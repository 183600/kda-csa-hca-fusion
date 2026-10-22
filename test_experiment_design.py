"""Tests for the pre-registered experiment-design and planning module.

The test names follow ``test_*`` so pytest can collect them directly::

    pytest -q test_experiment_design.py

They pin the properties the planning layer is supposed to guarantee:

* every committed plan is internally consistent and its declared
  confirmatory seed budget reaches its own target power;
* planning uses the multiplicity-corrected per-test alpha, not the raw
  family-wise alpha;
* a run is labelled confirmatory only when it reaches the required seed
  count AND contains the required controls;
* a pilot SD is extracted from per-seed rows (or from reported
  ``sd_diff`` rows) and never promotes the pilot itself to confirmatory;
* the manifest writer produces strict JSON that can be read back.
"""
from __future__ import annotations

import json
import math

import pytest

import experiment_design as D


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_is_internally_consistent():
    errors = D.validate_plans()
    assert errors == [], errors
    assert set(D.PLANS) == set(D.PLAN_IDS)
    for plan_id, plan in D.PLANS.items():
        assert plan['plan_id'] == plan_id
        # The confirmatory budget must be above the screening budget;
        # otherwise the "screening vs confirmatory" labels are noise.
        assert plan['planned_seeds'] > plan['screening_seeds'] >= 2
        # Every plan must name at least one result artifact.
        assert plan['result_files']
        # A returned plan is a copy: mutating it cannot corrupt the registry.
        copy = D.get_plan(plan_id)
        copy['planned_seeds'] = -1
        assert D.PLANS[plan_id]['planned_seeds'] > 1
    with pytest.raises(D.DesignError):
        D.get_plan('not_a_plan')


def test_validate_plan_rejects_stale_or_incomplete_designs():
    base = D.get_plan('exp4_mqar')
    # 10 is above the screening budget but still below the required 11.
    changed = dict(base, planned_seeds=10)
    errors = D.validate_plan(changed)
    assert any('below the prospective required_seeds' in e for e in errors)
    too_small = dict(base, planned_seeds=2)
    assert any('screening_seeds must be below' in e
               for e in D.validate_plan(too_small))
    missing_control = dict(base, required_levels={'operator': ['softmax']})
    # A required level absent from the factor itself is a registry error.
    missing_control['required_levels'] = {'operator': ['not-an-op']}
    assert any('not in the factor' in e
               for e in D.validate_plan(missing_control))
    bad_factor = dict(base, factors={'operator': {'role': 'treatment',
                                                  'levels': [1, 1]}})
    assert any('duplicate levels' in e for e in D.validate_plan(bad_factor))
    assert D.validate_plan('not a dict')


# ---------------------------------------------------------------------------
# Prospective planning
# ---------------------------------------------------------------------------

def test_planning_uses_corrected_alpha_and_mde():
    plan = D.get_plan('exp4_mqar')
    rep = D.planning_report(plan, seeds=plan['planned_seeds'])
    assert rep['status'] == 'confirmatory_ready'
    assert rep['required_seeds'] is not None
    assert plan['planned_seeds'] >= rep['required_seeds']
    assert rep['planned_power'] >= rep['target_power'] - 1e-9
    assert rep['alpha_per_test'] == pytest.approx(
        rep['alpha_family_wise'] / rep['family_size'])
    # The MDE is reported in endpoint units using the same normal model:
    # (z_crit + z_power) * SD / sqrt(n), with z_crit at the corrected alpha.
    from statistics import NormalDist
    z_crit = NormalDist().inv_cdf(1.0 - rep['alpha_per_test'])
    z_power = NormalDist().inv_cdf(rep['target_power'])
    expected_mde = ((z_crit + z_power) * rep['assumed_sd']
                    / math.sqrt(rep['seeds']))
    assert rep['planned_mde'] == pytest.approx(expected_mde)
    assert rep['planned_mde'] > 0

    # A larger family needs a larger n at the same SESOI and target power.
    small = D.planning_report(plan, seeds=5, family_size=1)
    large = D.planning_report(plan, seeds=5, family_size=20)
    assert large['required_seeds'] >= small['required_seeds']
    assert large['planned_power'] < small['planned_power']

    # The default screening budget is explicitly below the confirmatory n.
    screening = D.planning_report(plan, seeds=plan['screening_seeds'])
    assert screening['status'] == 'underpowered_at_sesoi'
    assert screening['planned_power'] < plan['target_power']


def test_two_sided_planning_applies_bonferroni_once():
    # exp5_ratio is a two-sided family: alpha_per_test is the one-sided TAIL
    # probability (alpha/(2*family)).  multi_seed_stats takes a test level,
    # so passing the tail alpha directly would halve it a second time and
    # overstate both the MDE and the required seed count.
    plan = D.get_plan('exp5_ratio')
    rep = D.planning_report(plan)
    assert rep['alternative'] == 'two-sided'
    assert rep['alpha_per_test'] == pytest.approx(
        rep['alpha_family_wise'] / (2.0 * rep['family_size']))
    from statistics import NormalDist
    z_crit = NormalDist().inv_cdf(1.0 - rep['alpha_per_test'])
    z_power = NormalDist().inv_cdf(rep['target_power'])
    expected_mde = ((z_crit + z_power) * rep['assumed_sd']
                    / math.sqrt(rep['seeds']))
    assert rep['planned_mde'] == pytest.approx(expected_mde)
    expected_required = math.ceil(
        ((z_crit + z_power) / rep['standardized_effect_dz']) ** 2)
    assert rep['required_seeds'] == max(2, expected_required)
    assert plan['planned_seeds'] >= rep['required_seeds']
    ncp = rep['standardized_effect_dz'] * math.sqrt(rep['seeds'])
    expected_power = ((1.0 - NormalDist().cdf(z_crit - ncp))
                      + NormalDist().cdf(-z_crit - ncp))
    assert rep['planned_power'] == pytest.approx(expected_power)


def test_planning_rejects_invalid_overrides():
    plan = D.get_plan('exp4_mqar')
    with pytest.raises(D.DesignError):
        D.planning_report(plan, seeds=-1)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, family_size=0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, alpha=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, sd=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, sesoi=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, alternative='either')
    # n < 2 leaves the inferential quantities undefined, not zero.
    tiny = D.planning_report(plan, seeds=1)
    assert tiny['planned_power'] is None
    assert tiny['planned_mde'] is None


# ---------------------------------------------------------------------------
# Runtime audit
# ---------------------------------------------------------------------------

def test_audit_labels_screening_confirmatory_and_invalid_runs():
    plan = D.get_plan('exp4_mqar')
    screening = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['screening_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=1)
    assert screening['status'] == 'screening'
    assert screening['claims_allowed'] == [
        'descriptive', 'exploratory_inference_with_caveat']

    confirmatory = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['planned_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=plan['planned_family_size'])
    assert confirmatory['status'] == 'confirmatory'
    assert confirmatory['checks']['family_size_at_least_planned'] is True
    assert 'paired_inference_at_planned_alpha' in \
        confirmatory['claims_allowed']

    # Reaching the required n only because the family shrank is not the
    # pre-registered experiment; it is a confirmatory test of a subset.
    reduced = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['planned_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']}, family_size=1)
    assert reduced['status'] == 'reduced_family'
    assert reduced['family_reduced'] is True
    assert reduced['claims_allowed'] == [
        'descriptive', 'paired_inference_for_tested_subset']

    missing_control = D.audit_run(
        'exp4_mqar', seeds=[42, 43, 44],
        levels={'operator': ['softmax', 'kda']}, family_size=1)
    assert missing_control['status'] == 'exploratory_missing_controls'
    assert 'operator:linear' in missing_control['missing_required_levels']
    assert missing_control['claims_allowed'] == ['descriptive_only']

    duplicate = D.audit_run('exp4_mqar', seeds=[42, 42],
                            levels={'operator': ['linear', 'softmax']})
    assert duplicate['status'] == 'invalid'
    assert duplicate['claims_allowed'] == []
    assert duplicate['checks']['seeds_unique'] is False

    one_seed = D.audit_run('exp4_mqar', seeds=[42],
                           levels={'operator': ['linear', 'softmax']})
    assert one_seed['status'] == 'invalid'


def test_audit_does_not_require_unobserved_levels_to_fail():
    # A caller that does not pass a factor cannot have it silently pass OR
    # silently fail: the check is recorded as not_evaluated.
    audit = D.audit_run('exp8_mechanism', seeds=[42, 43, 44, 45, 46],
                        family_size=4)
    assert audit['level_checks']['variant']['status'] == 'not_evaluated'
    assert audit['checks']['required_levels_present'] is True
    assert audit['status'] in ('screening', 'confirmatory')


# ---------------------------------------------------------------------------
# Pilot -> power -> confirmatory
# ---------------------------------------------------------------------------

def _exp4_rows(n_seeds=5):
    seeds = [42 + i for i in range(n_seeds)]
    base = [0.50, 0.52, 0.48, 0.51, 0.49][:n_seeds]
    diffs = [0.10, 0.12, 0.08, 0.15, 0.09][:n_seeds]
    rows = []
    for op, deltas in (('softmax', [0.0] * n_seeds), ('kda', diffs)):
        rows.append({
            'op': op, 'task': 'mqar', 'n_kv': 1,
            'per_seed': [
                {'seed': seed, 'final_acc': b + d}
                for seed, b, d in zip(seeds, base, deltas)
            ],
        })
    return rows


def test_pilot_sd_from_per_seed_rows_and_recommendation():
    info = D.pilot_sd_from_rows(_exp4_rows(), group_key='op',
                                reference='softmax',
                                cell_keys=['task', 'n_kv'])
    expected = [0.10, 0.12, 0.08, 0.15, 0.09]
    mean = sum(expected) / len(expected)
    expected_sd = math.sqrt(sum((d - mean) ** 2 for d in expected) / 4)
    assert info['method'] == 'per_seed_paired_differences'
    assert info['n_pairs'] == 1
    assert info['n_differences'] == 5
    assert info['sd_pooled'] == pytest.approx(expected_sd)
    assert info['sd'] == pytest.approx(expected_sd)
    assert info['sd_max'] == pytest.approx(expected_sd)

    report = D.pilot_report('exp4_mqar', _exp4_rows())
    assert report['status'] in ('pilot_reaches_required_n',
                                'pilot_is_screening_only')
    assert report['pilot_is_confirmatory'] is False
    assert report['pilot_seeds'] == 5
    assert report['observed_sd']['method'] == 'per_seed_paired_differences'
    assert report['required_seeds_pooled_sd'] >= 2
    assert report['recommended_seeds'] >= report['required_seeds_pooled_sd']


def test_pilot_report_with_a_large_sd_recommends_more_seeds():
    report = D.pilot_report('exp4_mqar', _exp4_rows(), family_size=5,
                            sd=0.25)
    assert report['observed_sd']['method'] == 'supplied_sd'
    assert report['required_seeds_pooled_sd'] > \
        D.PLANS['exp4_mqar']['screening_seeds']
    assert report['recommended_seeds'] >= report['required_seeds_pooled_sd']
    assert report['pilot_is_confirmatory'] is False


def test_pilot_uses_reported_contrast_sd_rows():
    rows = [
        {'contrast': 'B-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 0.10, 'mean_diff': 0.05},
        {'contrast': 'C-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 0.20, 'mean_diff': -0.05},
    ]
    info = D.pilot_sd_from_rows(rows, group_key='contrast',
                                reference='A', cell_keys=['n_kv'])
    assert info['method'] == 'reported_sd_diff'
    assert info['sd_pooled'] == pytest.approx(0.15)
    assert info['sd_max'] == pytest.approx(0.20)
    report = D.pilot_report('exp5b_matrix', rows, family_size=2)
    assert report['required_seeds_conservative'] >= \
        report['required_seeds_pooled_sd']


def test_pilot_can_exclude_the_memoryless_control():
    rows = _exp4_rows()
    rows.append({
        'op': 'linear', 'task': 'mqar', 'n_kv': 1,
        'per_seed': [{'seed': 42 + i, 'final_acc': 0.05 + 0.04 * i}
                     for i in range(5)],
    })
    all_pairs = D.pilot_sd_from_rows(
        rows, group_key='op', reference='softmax', cell_keys=['task', 'n_kv'])
    no_linear = D.pilot_sd_from_rows(
        rows, group_key='op', reference='softmax',
        cell_keys=['task', 'n_kv'], exclude=['linear'])
    assert all_pairs['n_pairs'] == 2
    assert no_linear['n_pairs'] == 1
    assert no_linear['sd_max'] < all_pairs['sd_max']


def test_pilot_with_zero_variance_is_not_a_license_for_two_seeds():
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.5},
                      {'seed': 43, 'final_acc': 0.5}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.6},
                      {'seed': 43, 'final_acc': 0.6}]},
    ]
    report = D.pilot_report('exp4_mqar', rows)
    assert report['status'] == 'degenerate_zero_sd'
    assert report['pilot_is_confirmatory'] is False


def test_pilot_without_two_common_seeds_is_insufficient():
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.5}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.6}]},
    ]
    assert D.pilot_sd_from_rows(rows, group_key='op', reference='softmax',
                               cell_keys=['task', 'n_kv']) is None
    report = D.pilot_report('exp4_mqar', rows)
    assert report['status'] == 'insufficient_pilot_data'
    assert report['observed_sd'] is None


# ---------------------------------------------------------------------------
# Manifest + CLI
# ---------------------------------------------------------------------------

def test_manifest_roundtrip_and_cli(tmp_path):
    manifest = D.design_manifest()
    assert manifest['schema_version'] == '1.0'
    assert manifest['kind'] == 'experiment_design'
    assert {row['plan_id'] for row in manifest['results']} == set(D.PLANS)
    # Strict JSON: nothing in the plan/planning path may be NaN or Inf.
    encoded = json.dumps(manifest, allow_nan=False)
    assert json.loads(encoded)['kind'] == 'experiment_design'

    path = tmp_path / 'design.json'
    D.write_design_manifest(str(path), manifest=manifest)
    with open(path, encoding='utf-8') as fh:
        loaded = json.load(fh)
    assert loaded['results'] == manifest['results']

    # The CLI check path validates the committed registry.
    assert D.main(['--check']) == 0
    # The CLI writes the same envelope.
    out = tmp_path / 'cli_design.json'
    assert D.main(['--plan', 'exp4_mqar', '--json', str(out)]) == 0
    written = json.loads(out.read_text(encoding='utf-8'))
    assert written['kind'] == 'experiment_design'
    assert len(written['results']) == 1
    assert written['results'][0]['plan_id'] == 'exp4_mqar'

    # Pilot CLI reads a result file and emits a planner report.
    pilot_path = tmp_path / 'exp4.json'
    pilot_path.write_text(json.dumps({'metadata': {},
                                      'results': _exp4_rows()}),
                          encoding='utf-8')
    rc = D.main(['--plan', 'exp4_mqar', '--pilot', str(pilot_path)])
    assert rc == 0


def test_manifest_seed_override_is_recorded():
    manifest = D.design_manifest(['exp4_mqar'], seeds_by_plan={
        'exp4_mqar': [42, 43, 44]})
    row = manifest['results'][0]
    assert row['seeds'] == [42, 43, 44]
    assert row['n_seeds'] == 3
    # Three seeds are below the confirmatory budget and stay labelled so.
    assert row['status'] == 'underpowered_at_sesoi'


def test_run_all_wires_the_design_manifest(monkeypatch, tmp_path):
    # run_all is importable without torch (the heavy imports live inside
    # run_all()); the design manifest must be registered as a raw side file
    # and its seed reader must agree with parse_int_env's fallback.
    import run_all
    assert 'results/experiment_design.json' in \
        run_all._EXPERIMENT_OUTPUTS['experiment_design']
    assert 'results/experiment_design.json' in run_all._RAW_SIDE_FILES
    monkeypatch.setenv('MQAR_SEEDS', '3')
    assert run_all._env_seed_list('MQAR_SEEDS', 5) == [42, 43, 44]
    monkeypatch.setenv('MQAR_SEEDS', 'not-an-int')
    assert run_all._env_seed_list('MQAR_SEEDS', 5) == [42, 43, 44, 45, 46]

    # The pre-run writer records the actual budget before the experiments.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MQAR_SEEDS', '3')
    monkeypatch.setenv('ABL_SEEDS', '4')
    monkeypatch.setenv('MECH_SEEDS', '2')
    block = run_all._write_experiment_design_manifest('smoke')
    assert block['path'] == 'results/experiment_design.json'
    assert block['validation_errors'] == []
    assert block['plans']['exp4_mqar']['n_seeds'] == 3
    assert block['plans']['exp5_ratio']['n_seeds'] == 4
    assert block['plans']['exp8_mechanism']['n_seeds'] == 2
    written = json.loads(
        (tmp_path / 'results' / 'experiment_design.json').read_text(
            encoding='utf-8'))
    assert written['kind'] == 'experiment_design'
    assert {row['plan_id']: row['n_seeds']
            for row in written['results']}['exp4_mqar'] == 3
