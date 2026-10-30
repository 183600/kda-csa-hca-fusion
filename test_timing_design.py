"""Tests for the pre-registered timing design layer."""
from __future__ import annotations

import json
import math

import timing_design as td


def _benchmark_cells(cv=0.02, repeats=20, workloads=(128, 256, 384)):
    ops = ['hybrid', 'hybrid_mamba2', 'kda_rec', 'kda_chunk',
           'mamba2_scan', 'mamba2_chunked_scan']
    values = {'hybrid': 1.0, 'hybrid_mamba2': 1.3,
              'kda_rec': 1.0, 'kda_chunk': 0.7,
              'mamba2_scan': 1.0, 'mamba2_chunked_scan': 0.8}
    schedule = td.balanced_schedule(ops, len(workloads), seed=7)
    cells = []
    for block, order in enumerate(schedule):
        for position, op in enumerate(order):
            cells.append({
                'workload': workloads[block],
                'op': op,
                'value_ms': values[op],
                'cv': cv,
                'n': repeats,
                'warmup': 2,
                'schedule_block': block,
                'order_position': position,
                'device': 'cpu',
            })
    return cells, schedule


def test_registry_is_valid():
    errors = td.validate_plans()
    assert set(errors) == set(td.TIMING_PLAN_IDS)
    assert all(not messages for messages in errors.values())


def test_planning_responds_to_cv_family_and_repeats():
    plan = td.get_plan('exp2_benchmark')
    base = td.planning_report(plan)
    assert base['required_repeats'] == plan['planned_repeats']
    assert base['planned_power'] >= plan['target_power'] - 1e-9
    assert base['planned_mde_ratio'] <= plan['sesoi_ratio']

    noisier = td.planning_report(plan, cv=0.20)
    assert noisier['required_repeats'] > base['required_repeats']

    larger_family = td.planning_report(plan, repeats=base['required_repeats'],
                                       family_size=plan['planned_family_size'] * 2)
    assert larger_family['planned_power'] < base['planned_power']

    more_repeats = td.planning_report(plan,
                                      repeats=plan['planned_repeats'] * 2)
    assert more_repeats['planned_power'] > base['planned_power']
    assert more_repeats['planned_mde_ratio'] < base['planned_mde_ratio']


def test_ratio_log_se_unit_accounts_for_pairing():
    independent = td.ratio_log_se_unit(0.1, 0.2)
    assert math.isclose(independent, math.sqrt(0.01 + 0.04))
    paired = td.ratio_log_se_unit(0.1, 0.2, paired=True, rho=0.5)
    assert paired < independent
    high = td.ratio_log_se_unit(0.1, 0.2, paired=True, rho=0.9)
    assert high < paired


def test_balanced_schedule_is_deterministic_and_position_balanced():
    ops = ['a', 'b', 'c', 'd', 'e']
    first = td.balanced_schedule(ops, 5, seed=11)
    second = td.balanced_schedule(ops, 5, seed=11)
    assert first == second
    assert all(sorted(block) == sorted(ops) for block in first)
    for position in range(len(ops)):
        assert sorted(block[position] for block in first) == sorted(ops)
    info = td._schedule_balance(first)
    assert info['balanced']
    assert info['complete_permutations']
    assert info['max_position_span'] == 0
    assert math.isclose(info['max_position_share'], 1.0 / 5.0)


def test_fixed_order_is_not_a_balanced_schedule():
    ops = ['a', 'b', 'c']
    fixed = [list(ops) for _ in range(4)]
    info = td._schedule_balance(fixed)
    assert not info['balanced']
    assert info['complete_permutations']
    assert info['max_position_span'] == 4


def test_audit_confirmatory_when_budget_schedule_and_precision_hold():
    cells, schedule = _benchmark_cells()
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2,
        schedule=schedule, family_size=15)
    assert report['status'] == 'confirmatory', report['design_warnings']
    assert 'relative_latency_ratio_at_sesoi' in report['claims_allowed']
    assert report['analysis_population']['complete']
    assert report['max_mde_ratio'] <= 1.15
    assert all(row['claim_allowed'] for row in report['contrast_rows'])


def test_audit_screening_when_repeats_or_precision_are_low():
    cells, schedule = _benchmark_cells(cv=0.10, repeats=3)
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=3, warmup=2,
        schedule=schedule, family_size=9)
    assert report['status'] == 'screening'
    assert 'relative_latency_ratio_at_sesoi' not in report['claims_allowed']
    assert any('below the SESOI' in warning
               for warning in report['design_warnings'])


def test_zero_cv_is_screening_not_a_crash():
    cells, schedule = _benchmark_cells(cv=0.0)
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2,
        schedule=schedule, family_size=15)
    assert report['status'] == 'screening'
    assert any('zero observed CV' in warning
               for warning in report['design_warnings'])
    assert not any(row['claim_allowed'] for row in report['contrast_rows'])


def test_reduced_workload_grid_is_not_confirmatory():
    cells, schedule = _benchmark_cells()
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2,
        schedule=schedule, family_size=9)
    assert report['status'] == 'screening'
    assert any('smaller than the registered plan' in warning
               for warning in report['design_warnings'])


def test_audit_flags_order_and_warmup_confounds():
    cells, _schedule = _benchmark_cells()
    ops = sorted({cell['op'] for cell in cells})
    fixed = [ops for _ in range(3)]
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=0,
        schedule=fixed, family_size=9)
    assert report['status'] == 'order_confounded'
    assert any('not a balanced schedule' in warning
               for warning in report['design_warnings'])
    assert any('warmup' in warning for warning in report['design_warnings'])


def test_audit_flags_entire_missing_workload_block():
    cells, schedule = _benchmark_cells()
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2,
        schedule=schedule, family_size=15,
        workloads=(128, 256, 384, 512))
    assert report['status'] == 'incomplete_cells'
    assert report['analysis_population']['missing_workloads'] == [512]
    assert any('expected workload block' in warning
               for warning in report['design_warnings'])


def test_audit_flags_missing_contrast_cell():
    cells, schedule = _benchmark_cells()
    cells = [cell for cell in cells
             if not (cell['workload'] == 256 and cell['op'] == 'hybrid')]
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2,
        schedule=schedule, family_size=9)
    assert report['status'] == 'incomplete_cells'
    assert report['missing_cells']


def test_audit_derives_schedule_from_recorded_positions():
    cells, _schedule = _benchmark_cells()
    report = td.audit_run(
        'exp2_benchmark', cells, repeats=20, warmup=2, family_size=15)
    assert report['schedule']['balanced']
    assert report['status'] == 'confirmatory'


def test_pilot_report_is_conservative_and_never_confirmatory():
    cells, _schedule = _benchmark_cells(cv=0.02)
    cells[0]['cv'] = 0.20
    report = td.pilot_report('exp2_benchmark', cells, family_size=9)
    assert report['pilot_is_confirmatory'] is False
    assert report['cv_max'] >= report['cv_median']
    assert (report['required_repeats_conservative']
            >= report['required_repeats_pooled_cv'])
    assert (report['recommended_repeats']
            == report['required_repeats_conservative'])


def test_result_row_normalization_and_manifest_are_json_safe():
    rows = [{
        'T': 128, 'op': 'hybrid', 'time_ms': 4.0, 'time_cv': 0.05,
        'time_n': 7, 'warmup': 2, 'schedule_block': 0,
        'order_position': 0, 'device': 'cpu',
    }]
    cells = td.cells_from_benchmark_rows(rows)
    assert len(cells) == 1
    assert cells[0]['workload'] == 128
    assert cells[0]['n'] == 7
    manifest = td.design_manifest(repeats_by_plan={'exp2_benchmark': 7})
    json.dumps(manifest, allow_nan=False)
    row = next(item for item in manifest['results']
               if item['plan_id'] == 'exp2_benchmark')
    assert row['n_repeats'] == 7
    assert row['status'] == 'underpowered_at_sesoi'
    assert row['plan_sha256'] == td.plan_sha256('exp2_benchmark')


def test_result_plan_id_detects_envelope_and_raw_rows(tmp_path):
    exp2 = tmp_path / 'exp2.json'
    exp2.write_text(json.dumps({'kind': 'exp2_benchmark', 'results': []}))
    assert td.result_plan_id(str(exp2)) == 'exp2_benchmark'
    exp6 = tmp_path / 'exp6.json'
    exp6.write_text(json.dumps([
        {'prefill_len': 128, 'op': 'kda',
         'median_decode_ms_per_token': 0.5}]))
    assert td.result_plan_id(str(exp6)) == 'exp6_decoding'
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps([{'T': 128, 'op': 'kda', 'time_ms': 1.0}]))
    assert td.result_plan_id(str(raw)) == 'exp2_benchmark'


def test_decoding_cells_use_trial_replicates_not_token_samples():
    legacy = [{
        'prefill_len': 128, 'op': 'kda',
        'median_decode_ms_per_token': 0.5,
        'decode_cv': 0.05, 'decode_n': 60, 'repeats': 3,
        'warmup': 1, 'schedule_block': 0, 'order_position': 0,
        'device': 'cpu',
    }]
    cell = td.cells_from_decoding_rows(legacy)[0]
    assert cell['n'] == 3
    assert cell['cv'] == 0.05

    with_trials = [dict(legacy[0], decode_trial_cv=0.12,
                        decode_trial_n=3)]
    cell = td.cells_from_decoding_rows(with_trials)[0]
    assert cell['n'] == 3
    assert cell['cv'] == 0.12


def test_raw_manifest_round_trip_and_raw_contrasts(tmp_path):
    cells = [
        {'T': 128, 'op': 'hybrid', 'samples_ms': [1.0, 1.1, 0.9, 1.05]},
        {'T': 128, 'op': 'hybrid_mamba2',
         'samples_ms': [1.4, 1.5, 1.3, 1.45]},
        {'T': 128, 'op': 'kda_rec', 'samples_ms': [1.0, 1.0, 1.1, 0.9]},
        {'T': 128, 'op': 'kda_chunk',
         'samples_ms': [0.7, 0.72, 0.68, 0.71]},
    ]
    manifest = td.raw_manifest('exp2_benchmark', cells, warmup=2,
                               schedule_id='balanced_rotation:0')
    path = tmp_path / 'exp2_raw.json'
    td.write_raw_manifest(str(path), manifest=manifest)
    assert td.result_plan_id(str(path)) == 'exp2_benchmark'
    loaded = td.load_raw_cells(str(path))
    assert [cell['op'] for cell in loaded] == [cell['op'] for cell in cells]
    rows = td.raw_contrast_rows('exp2_benchmark', loaded, family_size=15)
    by_id = {row['contrast']: row for row in rows}
    assert by_id['route_swap_hybrid']['ci_excludes_one'] is True
    assert by_id['kda_chunk_vs_rec']['ci_excludes_one'] is True
    assert by_id['mamba2_chunked_vs_scan']['missing'] == [
        'mamba2_chunked_scan', 'mamba2_scan']


def test_cli_check_and_json_manifest(tmp_path):
    assert td.main(['--check']) == 0
    path = tmp_path / 'timing_design.json'
    assert td.main(['--json', str(path)]) == 0
    payload = json.loads(path.read_text())
    assert payload['kind'] == 'timing_design'
    assert {row['plan_id'] for row in payload['results']} == set(
        td.TIMING_PLAN_IDS)


def test_fingerprint_changes_with_plan_definition():
    plan = td.get_plan('exp6_decoding')
    original = td.plan_sha256(plan)
    plan['sesoi_ratio'] = 1.20
    assert td.plan_sha256(plan) != original
