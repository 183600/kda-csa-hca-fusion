"""Tests for the results-contract and experiment-integrity layer.

The schema layer (``results_schema``) guarantees SHAPE; these tests cover
what shape alone cannot: whether the rows make sense. Three failure modes
motivate the suite:

1. **Partial eight-group matrix.** A crash mid-sweep that persists a
   survivor-only ``results/exp5b_ablation_matrix.json`` looks perfectly
   schema-valid. The per-kind row contract compares the written group set
   against the ``metadata.expected_groups`` the preset promised.
2. **Aggregate/contrast drift.** ``mean_acc`` must equal the mean of the
   per-seed ``final_acc`` values, and every paired contrast in
   ``results/exp5b_ablation_matrix_contrasts.json`` is re-derived from the
   raw per-seed rows by an independent implementation
   (``run_ablation._contrast_consistency_violations``) before the file is
   persisted.
3. **Dishonest incremental-decoding metadata.** Exp 6 rows carry
   ``uses_incremental_cache`` / ``prefill_cache_build`` flags used in the
   fairness discussion. Every ``hybrid*`` row threads one CSA/HCA
   incremental cache per sub-layer regardless of the memory route, so a
   hybrid row claiming native state is a results bug.

Also covered: the eight-group design itself
(``run_ablation.build_experiment_matrix``), the paired-contrast
arithmetic, chunk-schedule invariance of the incremental decoding
wrappers (one-shot vs token-by-token vs mixed continuation chunks), and
the registration of the matrix's contrast side file as a purgeable
experiment output.

How to run::

    pytest -q test_results_contracts.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import results_schema as rs  # noqa: E402
from run_ablation import (  # noqa: E402
    CONTRAST_EQUIVALENCE_MARGIN,
    MATRIX_UNIT,
    _annotate_contrast_significance,
    _paired_contrast,
    _attach_tuned_baseline_flags,
    _contrast_consistency_violations,
    _matrix_contrast_report,
    _matrix_design_audit,
    _tuned_baseline_contrast_report,
    build_experiment_matrix,
    build_simple_baseline_groups,
)
from run_decoding import (  # noqa: E402
    HYBRID_OPS,
    INCREMENTAL_CACHE_OPS,
    CSAAttnDecoding,
    HCAAttnDecoding,
    HybridDecoding,
)


def _matrix_row(group, memory, accs, *, n_kv=1, n_layers=10, n_params=1234,
                seeds=None, error=None, layout=None):
    """One exp5b-shaped matrix row for contract/contrast tests."""
    if seeds is None:
        seeds = [42 + i for i in range(len(accs))]
    if layout is None:
        head = memory.upper() if memory else 'KDA'
        layout = '-'.join([head] * max(n_layers - 2, 1) + ['CSA', 'HCA'])
    per_seed = [{'seed': s, 'final_acc': a, 'final_loss': 1.0 - a}
                for s, a in zip(seeds, accs)]
    row = {
        'group': group, 'memory': memory, 'n_kv': n_kv,
        'n_seeds': len(per_seed), 'n_seeds_ok': len(per_seed),
        'per_seed': per_seed, 'layout': layout, 'n_layers': n_layers,
        'n_params': n_params,
        'mean_acc': (sum(accs) / len(accs)) if accs else None,
    }
    if error is not None:
        row.update({'error': error, 'layout': None, 'mean_acc': None,
                    'n_params': None, 'per_seed': []})
    return row


def _exp6_row(op, **overrides):
    row = {
        'op': op, 'prefill_len': 128,
        'n_layers': 5 if op in HYBRID_OPS else 1,
        'prefill_ms': 1.0, 'mean_decode_ms_per_token': 0.1,
        'median_decode_ms_per_token': 0.1, 'peak_mem_MB': None,
        'n_decode': 20, 'repeats': 3,
        'uses_incremental_cache': op in INCREMENTAL_CACHE_OPS,
        'prefill_cache_build': (
            'batched_populate_layer_by_layer' if op in HYBRID_OPS
            else ('batched_append_from_naive_output'
                  if op in ('csa', 'hca') else 'native_operator_state')),
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Row contracts
# ---------------------------------------------------------------------------

def test_row_contract_matrix_accepts_valid_and_flags_defects():
    rows = [
        _matrix_row('KDA-A', 'kda', (0.2, 0.4)),
        _matrix_row('KDA-B', 'kda', (0.5, 0.8)),
        _matrix_row('MAMBA2-A', 'mamba2', (0.1, 0.3)),
        _matrix_row('MAMBA2-B', 'mamba2', (0.2, 0.5)),
    ]
    meta = {'expected_groups': [r['group'] for r in rows],
            'total_layers': 10}
    assert rs.validate_rows('exp5b_ablation_matrix', rows, meta) == []

    # Finite per-seed accuracies near the float limit overflow their raw
    # sum; the contract must re-derive the (finite) mean without rejecting
    # a valid row.
    huge = _matrix_row('KDA-H', 'kda', (1e308, 1e308))
    huge['mean_acc'] = 1e308
    assert rs._recomputed_mean(huge['per_seed'])[0] == 1e308
    huge_meta = {'expected_groups': meta['expected_groups'] + ['KDA-H'],
                 'total_layers': 10}
    assert rs.validate_rows(
        'exp5b_ablation_matrix', rows + [huge], huge_meta) == []

    # Duplicate (group, n_kv) key.
    assert rs.validate_rows('exp5b_ablation_matrix',
                            rows + [dict(rows[0])], meta)
    # mean_acc inconsistent with the per-seed accuracies.
    tampered = [dict(rows[0], mean_acc=rows[0]['mean_acc'] + 0.01)] + rows[1:]
    assert rs.validate_rows('exp5b_ablation_matrix', tampered, meta)
    # A crashed run that silently drops groups.
    assert rs.validate_rows('exp5b_ablation_matrix', rows[:-1], meta)
    # A layout whose memory tokens contradict the row's memory route.
    wrong = [dict(rows[0], layout='MAMBA2-MAMBA2-MAMBA2-CSA-HCA')]
    assert rs.validate_rows('exp5b_ablation_matrix', wrong, meta)
    # A failed group is structurally legal (it carries the reason).
    errored = rows + [_matrix_row('KDA-C', 'kda', (), error='OOM')]
    full_meta = dict(meta, expected_groups=meta['expected_groups'] + ['KDA-C'])
    assert rs.validate_rows('exp5b_ablation_matrix', errored, full_meta) == []


def test_row_contract_contrasts_requires_real_pairs():
    good = [{'n_kv': 1, 'memory': 'kda',
             'contrast': 'B-A (CSA replaces memory)', 'mean_diff': 0.35,
             't_stat': 7.0, 'n_common_seeds': 2, 'metric': 'final_acc',
             'direction': 'higher_is_better'}]
    assert rs.validate_rows('exp5b_ablation_matrix_contrasts', good, {}) == []
    assert rs.validate_rows('exp5b_ablation_matrix_contrasts',
                            [dict(good[0], n_common_seeds=1)], {})
    assert rs.validate_rows('exp5b_ablation_matrix_contrasts',
                            [dict(good[0], mean_diff=float('nan'))], {})
    assert rs.validate_rows('exp5b_ablation_matrix_contrasts',
                            good + [dict(good[0])], {})
    for field in ('memory', 'contrast'):
        violations = rs.validate_rows(
            'exp5b_ablation_matrix_contrasts',
            [dict(good[0], **{field: []})], {})
        assert any(field in v for v in violations), violations


def test_tuned_baseline_contrasts_pair_each_group_with_softmax():
    """Every non-baseline row is tested against softmax at the same n_kv."""
    rows = [
        {'group': 'BASELINE-softmax', 'memory': 'softmax', 'n_kv': 1,
         'lr': 1e-2, 'per_seed': [
             {'seed': 42, 'final_acc': 0.50},
             {'seed': 43, 'final_acc': 0.60}]},
        {'group': 'BASELINE-softmax', 'memory': 'softmax', 'n_kv': 2,
         'lr': 1e-2, 'per_seed': [
             {'seed': 42, 'final_acc': 0.40},
             {'seed': 43, 'final_acc': 0.50}]},
        {'group': 'KDA-D', 'memory': 'kda', 'n_kv': 1, 'lr': 3e-3,
         'per_seed': [
             {'seed': 42, 'final_acc': 0.70},
             {'seed': 43, 'final_acc': 0.80}]},
        {'group': 'BASELINE-linear', 'memory': 'linear', 'n_kv': 1,
         'lr': 3e-3, 'per_seed': [
             {'seed': 42, 'final_acc': 0.05},
             {'seed': 43, 'final_acc': 0.05}]},
    ]
    contrasts = _tuned_baseline_contrast_report(rows)
    assert len(contrasts) == 1
    row = contrasts[0]
    assert row['candidate_group'] == 'KDA-D'
    assert row['reference_group'] == 'BASELINE-softmax'
    assert row['baseline_lr'] == 1e-2 and row['candidate_lr'] == 3e-3
    assert row['mean_diff'] == pytest.approx(0.2)
    summary = _annotate_contrast_significance(
        contrasts, hypotheses={'tuned_baseline': {'direction': 'greater'}})
    assert summary['n_contrasts'] == 1
    assert row['significant_bonferroni'] is False
    assert row['hypothesis_status'] == 'not_supported'

    _attach_tuned_baseline_flags(rows, contrasts)
    assert rows[2]['beats_tuned_baseline'] is False
    assert rows[2]['tuned_baseline_diff'] == pytest.approx(0.2)
    assert rows[0]['beats_tuned_baseline'] is None
    assert rows[3]['beats_tuned_baseline'] is None

    # No softmax reference at this n_kv -> no directly testable pair.
    assert _tuned_baseline_contrast_report(rows[2:]) == []


def test_row_contract_exp6_cache_flags_are_honest():
    ops = ('softmax', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca', 'hybrid',
           'hybrid_mamba2', 'hybrid_rwkv7')
    rows = [_exp6_row(op) for op in ops]
    meta = {'prefill_lens': [128], 'ops': list(ops)}
    assert rs.validate_rows('exp6_decoding', rows, meta) == []

    # Hybrid routes run the incremental caches on every memory route, so
    # False is a contradiction.
    bug = [dict(r, uses_incremental_cache=False)
           if r['op'] == 'hybrid_mamba2' else r for r in rows]
    violations = rs.validate_rows('exp6_decoding', bug, meta)
    assert any('hybrid_mamba2' in v and 'incremental' in v
               for v in violations)
    # Duplicate cell and non-positive latency.
    assert rs.validate_rows('exp6_decoding', rows + [dict(rows[0])], meta)
    assert rs.validate_rows(
        'exp6_decoding',
        [dict(r, prefill_ms=0.0) if r['op'] == 'kda' else r for r in rows],
        meta)
    # An incomplete benchmark grid (op list recorded in metadata).
    assert rs.validate_rows('exp6_decoding', rows[:-1], meta)
    # An unhashable build tag must be reported, not crash the validator.
    violations = rs.validate_rows(
        'exp6_decoding',
        [dict(r, prefill_cache_build=[])
         if r['op'] == 'kda' else r for r in rows],
        meta)
    assert any('prefill_cache_build' in v for v in violations), violations
    # A CI block with the median missing must be reported, not crash the
    # validator with float(None).
    broken = dict(_exp6_row('kda'), decode_std_ms=0.1,
                  decode_q10_ms=0.05, decode_q90_ms=0.2,
                  decode_ci95_low_ms=0.04, decode_ci95_high_ms=0.21,
                  decode_ci95_halfwidth_ms=0.085, decode_cv=0.5,
                  decode_n=3, timing_ci_method='percentile_bootstrap_median',
                  timing_bootstrap_reps=2000)
    del broken['median_decode_ms_per_token']
    violations = rs.validate_rows('exp6_decoding', [broken], meta)
    assert any('median_decode_ms_per_token' in v for v in violations), violations


def test_row_contract_exp6_decode_state_certificate_fields():
    """Certificate fields are optional (legacy rows) but validated when present."""
    ops = ('softmax', 'kda', 'csa')
    base = [_exp6_row(op) for op in ops]
    meta = {'prefill_lens': [128], 'ops': list(ops)}
    # Legacy rows without the fields stay valid.
    assert rs.validate_rows('exp6_decoding', base, meta) == []

    good = [dict(r, decode_state_bytes=2048, decode_state_allocated_bytes=4096,
                 decode_state_breakdown={'recurrent_state': {
                     'allocated_bytes': 4096, 'valid_bytes': 2048}})
            for r in base]
    assert rs.validate_rows('exp6_decoding', good, meta) == []

    # valid bytes must be a positive int.
    assert rs.validate_rows(
        'exp6_decoding',
        [dict(r, decode_state_bytes=0, decode_state_allocated_bytes=4096)
         if r['op'] == 'kda' else r for r in good],
        meta)
    # allocated >= valid.
    assert rs.validate_rows(
        'exp6_decoding',
        [dict(r, decode_state_bytes=8192, decode_state_allocated_bytes=4096)
         if r['op'] == 'csa' else r for r in good],
        meta)
    # breakdown must be an object.
    assert rs.validate_rows(
        'exp6_decoding',
        [dict(r, decode_state_breakdown=[1, 2, 3])
         if r['op'] == 'softmax' else r for r in good],
        meta)


def test_row_contract_exp14_parity_counts_reject_bools():
    """Parity counts are integer counts, not truthy/falsy flags."""
    base = {'section': 'parity', 'case': 'random', 'n_cases': 10,
            'mismatches': 0, 'crashes': 0}
    assert rs.validate_rows('exp14_indexer_selector', [base], {}) == []
    for key in ('mismatches', 'crashes'):
        bad = dict(base, **{key: False})
        violations = rs.validate_rows('exp14_indexer_selector', [bad], {})
        assert any(key in v and 'exactly 0' in v for v in violations)
        bad = dict(base, **{key: 0.0})
        violations = rs.validate_rows('exp14_indexer_selector', [bad], {})
        assert any(key in v and 'exactly 0' in v for v in violations)
    # Malformed section/case tokens are violations, not unhashable-set
    # crashes during duplicate detection.
    for field in ('section', 'case'):
        bad = dict(base, **{field: []})
        violations = rs.validate_rows('exp14_indexer_selector', [bad], {})
        assert any(field in v for v in violations), violations


def test_unknown_kinds_and_legacy_payloads_pass():
    assert rs.validate_rows('no_such_kind', [{'anything': 1}], {}) == []
    assert rs.validate_payload([{'legacy': True}]) == []
    assert rs.validate_payload({
        'schema_version': '1.0', 'kind': 'unregistered_kind',
        'metadata': {}, 'results': [{'whatever': 1}]}) == []


def test_write_checked_roundtrip_and_cli(tmp_path):
    rows = [_matrix_row('KDA-A', 'kda', (0.2, 0.4)),
            _matrix_row('KDA-B', 'kda', (0.5, 0.8))]
    meta = {'expected_groups': ['KDA-A', 'KDA-B'], 'total_layers': 10}
    good = tmp_path / 'matrix.json'
    rs.write_checked(str(good), rows, kind='exp5b_ablation_matrix',
                     metadata=meta)
    loaded_meta, loaded_rows = rs.load(str(good))
    assert len(loaded_rows) == 2
    assert loaded_meta['expected_groups'] == ['KDA-A', 'KDA-B']

    tampered = [dict(rows[0], mean_acc=rows[0]['mean_acc'] + 0.01)] + rows[1:]
    bad = tmp_path / 'matrix_bad.json'
    rs.write(str(bad), tampered, kind='exp5b_ablation_matrix', metadata=meta)
    with pytest.raises(rs.RowContractError):
        rs.write_checked(str(bad), tampered, kind='exp5b_ablation_matrix',
                         metadata=meta)

    assert rs.main(['--check-rows', str(good)]) == 0
    assert rs.main(['--check-rows', str(bad)]) == 1
    # Plain --check stays shape-only (backwards compatible).
    assert rs.main(['--check', str(bad)]) == 0


# ---------------------------------------------------------------------------
# Eight-group design + paired contrasts
# ---------------------------------------------------------------------------

def test_experiment_matrix_design():
    g8 = build_experiment_matrix('main8', total_layers=10,
                                 memory_types=('kda', 'mamba2'))
    ids = [spec[0] for spec in g8]
    assert ids == [f'{m.upper()}-{u}'
                   for m in ('kda', 'mamba2') for u in 'ABCD']
    tokens = {t for unit in MATRIX_UNIT.values() for t in unit}
    assert all(len(spec[2]) == 10 and set(spec[2]) <= tokens for spec in g8)
    # Depth and operator counts are the controlled variables.
    counts = {spec[0]: (spec[2].count('memory'), spec[2].count('csa'),
                        spec[2].count('hca')) for spec in g8}
    # Two 5-layer units at depth 10: B/C sacrifice one memory layer per
    # unit (not per model), D sacrifices two.
    assert counts['KDA-A'] == (10, 0, 0)
    assert counts['KDA-B'] == (8, 2, 0)
    assert counts['KDA-C'] == (8, 0, 2)
    assert counts['KDA-D'] == (6, 2, 2)

    # Default memory_types covers all three routes (12 groups).
    g12 = build_experiment_matrix('main8', 10)
    assert len(g12) == 12 and len({spec[0] for spec in g12}) == 12

    # Budget preset: 5-tuples carrying the per-group cfg overrides.
    budget = build_experiment_matrix('budget', 10, ('kda',))
    assert len(budget) == 5
    assert all(len(spec) == 5 and spec[4]['csa_m'] > 0
               and spec[4]['hca_m2'] > 0 for spec in budget)

    # Position preset: same operator counts, different order.
    position = build_experiment_matrix('position', 10, ('kda',))
    alternating, back_loaded = position[0], position[1]
    for kind in ('memory', 'csa', 'hca'):
        assert (alternating[2].count(kind)
                == back_loaded[2].count(kind))
    assert alternating[2] != back_loaded[2]

    with pytest.raises(ValueError):
        build_experiment_matrix('main8', 7)
    with pytest.raises(ValueError):
        build_experiment_matrix('not-a-preset', 10)


def test_matrix_contrast_report_seed_matched_arithmetic():
    accmaps = {
        'kda': {'A': (0.2, 0.4), 'B': (0.5, 0.8), 'C': (0.4, 0.6),
                'D': (0.8, 0.9)},
        'mamba2': {'A': (0.1, 0.3), 'B': (0.2, 0.25), 'C': (0.3, 0.35),
                   'D': (0.5, 0.55)},
    }
    rows = [_matrix_row(f'{mem.upper()}-{g}', mem, accs)
            for mem, per_group in accmaps.items()
            for g, accs in per_group.items()]
    contrasts = _matrix_contrast_report(rows)
    index = {(c['n_kv'], c['memory'], c['contrast']): c for c in contrasts}

    def mean_of(name, memory='kda'):
        return index[(1, memory, name)]['mean_diff']

    assert mean_of('B-A (CSA replaces memory)') == pytest.approx(0.35)
    assert mean_of('C-A (HCA replaces memory)') == pytest.approx(0.2)
    assert mean_of('D-B (add HCA to CSA)') == pytest.approx(0.2)
    assert mean_of('D-C (add CSA to HCA)') == pytest.approx(0.35)
    # I = D - B - C + A per seed, then averaged.
    expected_i = ((0.8 - 0.5 - 0.4 + 0.2) + (0.9 - 0.8 - 0.6 + 0.4)) / 2
    assert mean_of('I=D-B-C+A (CSA/HCA interaction)') == pytest.approx(
        expected_i)
    assert index[(1, 'kda vs mamba2', 'A_KDA - A_MAMBA2')][
        'mean_diff'] == pytest.approx(0.1)
    expected_d = ((0.8 - 0.5) + (0.9 - 0.55)) / 2
    assert index[(1, 'kda vs mamba2', 'D_KDA - D_MAMBA2')][
        'mean_diff'] == pytest.approx(expected_d)
    assert all(c['n_common_seeds'] == 2 for c in contrasts)

    # Every reported row now also carries the sd of the per-seed
    # differences (the input to the TOST equivalence verdict).
    assert all(c['sd_diff'] is not None and c['sd_diff'] >= 0.0
               for c in contrasts)

    # Independent audit accepts the report and rejects tampering.
    assert _contrast_consistency_violations(rows, contrasts) == []
    tampered = [dict(contrasts[0],
                     mean_diff=contrasts[0]['mean_diff'] + 0.5)] + contrasts[1:]
    assert _contrast_consistency_violations(rows, tampered)
    tampered_sd = [dict(contrasts[0],
                        sd_diff=contrasts[0]['sd_diff'] + 0.5)] + contrasts[1:]
    assert _contrast_consistency_violations(rows, tampered_sd)

    # Fewer than two common seeds -> no paired contrast is reported.
    disjoint = [
        _matrix_row('KDA-A', 'kda', (0.2, 0.4), seeds=[42, 43]),
        _matrix_row('KDA-B', 'kda', (0.5, 0.8), seeds=[100, 101]),
    ]
    assert _matrix_contrast_report(disjoint) == []


def _controlled_matrix_rows():
    groups = build_experiment_matrix('main8', 10, ('kda',))
    rows = []
    for spec in groups:
        for n_kv in (1, 2):
            row = _matrix_row(spec[0], 'kda', (0.2, 0.4), n_kv=n_kv)
            row.update({'steps': 100, 'train_batch': 16, 'lr': 3e-3,
                        'd_model': 32})
            rows.append(row)
    return groups, rows


def test_matrix_design_audit_verifies_factorial_recipe_and_seed_parity():
    groups, rows = _controlled_matrix_rows()
    audit = _matrix_design_audit(
        rows, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'complete_factorial'
    assert audit['n_expected_cells'] == audit['n_present_cells'] == 8
    assert audit['missing_groups'] == []
    assert audit['violations'] == []
    assert audit['controls']['hybrid_parity']['steps'] is True
    assert audit['controls']['hybrid_parity']['train_batch'] is True
    assert audit['controls']['hybrid_parity']['lr'] is True
    assert audit['controls']['seed_alignment_ok'] is True
    assert all(c['factorial_complete'] for c in audit['cells'])
    assert all(c['n_common_seeds'] == 2 for c in audit['cells'])

    drifted = [dict(r) for r in rows]
    drifted[0]['lr'] = 1e-2
    audit = _matrix_design_audit(
        drifted, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'recipe_mismatch'
    assert any('hybrid_recipe_mismatch' in v for v in audit['violations'])

    unpaired = [dict(r) for r in rows]
    unpaired[1]['per_seed'] = unpaired[1]['per_seed'][:1]
    audit = _matrix_design_audit(
        unpaired, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'seed_mismatch'
    assert audit['controls']['seed_mismatch_cells']

    survivor_only = rows[1:]
    audit = _matrix_design_audit(
        survivor_only, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'incomplete'
    assert audit['missing_groups'] == ['KDA-A']


def test_matrix_design_audit_counts_and_flags_simple_baseline_cells():
    groups = (build_experiment_matrix('main8', 10, ('kda',))
              + build_simple_baseline_groups(10))
    rows = []
    for spec in groups:
        for n_kv in (1, 2):
            row = _matrix_row(spec[0], spec[1], (0.2, 0.4), n_kv=n_kv)
            row.update({'steps': 100, 'train_batch': 16, 'lr': 3e-3,
                        'd_model': 32})
            rows.append(row)
    audit = _matrix_design_audit(
        rows, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'complete_factorial'
    assert audit['n_expected_cells'] == audit['n_present_cells'] == 12
    assert audit['missing_groups'] == []
    baseline_cells = [c for c in audit['cells']
                      if c['memory'] in ('linear', 'softmax')]
    assert len(baseline_cells) == 4
    assert all(c['factorial_complete'] for c in baseline_cells)
    assert all(c['present_groups'] == ['BASELINE-' + c['memory']]
               for c in baseline_cells)

    dropped = [r for r in rows
               if not (r['group'] == 'BASELINE-linear' and r['n_kv'] == 1)]
    audit = _matrix_design_audit(
        dropped, groups, preset='main8', n_kv_list=[1, 2], n_seeds=2)
    assert audit['status'] == 'incomplete'
    assert audit['missing_groups'] == ['BASELINE-linear']


def test_contrast_annotation_reports_effect_size_and_power_plan():
    rows = [
        _matrix_row('KDA-A', 'kda', (0.20, 0.25, 0.30)),
        _matrix_row('KDA-B', 'kda', (0.30, 0.45, 0.60)),
    ]
    contrasts = _matrix_contrast_report(rows)
    assert len(contrasts) == 1
    summary = _annotate_contrast_significance(contrasts)
    c = contrasts[0]
    # Per-seed differences [0.1, 0.2, 0.3]: mean 0.2, SD 0.1, d_z = 2.
    assert c['effect_size_cohens_dz'] == pytest.approx(2.0)
    assert c['min_detectable_effect'] == pytest.approx(
        (1.6448536269514722 + 0.8416212335729143) * 0.1 / (3 ** 0.5))
    assert c['seeds_required_80pct_power'] == 2
    assert c['below_default_min_effect'] is False
    assert c['significant_bonferroni'] is True
    assert summary['n_significant_but_below_min_effect'] == 0
    assert summary['effect_size']['power_target'] == 0.8

    small = [
        _matrix_row('KDA-A', 'kda', (0.200, 0.210, 0.220)),
        _matrix_row('KDA-B', 'kda', (0.201, 0.212, 0.223)),
    ]
    small_contrasts = _matrix_contrast_report(small)
    small_summary = _annotate_contrast_significance(small_contrasts)
    assert small_contrasts[0]['significant_bonferroni'] is True
    assert small_contrasts[0]['below_default_min_effect'] is True
    assert small_contrasts[0][
        'significant_but_below_default_min_effect'] is True
    assert small_summary['n_significant_but_below_min_effect'] == 1


def test_tune_baseline_lr_ignores_malformed_validation_points(monkeypatch):
    """A JSON-null/nonfinite validation accuracy must not crash the tuner."""
    import run_quality as rq
    calls = []

    def fake_train(op_name, **kwargs):
        calls.append(kwargs['lr'])
        return {'per_seed': [
            {'val_history': [{'acc': 0.2}, {'acc': 0.3},
                             {'acc': None, 'loss': 1.0}]},
            {'val_history': [{'acc': 0.4}, {'acc': float('nan')},
                             {'acc': 0.6}]},
            'not-a-dict',
        ]}

    monkeypatch.setattr(rq, 'train_multi_seed', fake_train)
    best, record = rq.tune_baseline_lr(
        'softmax', [1e-3, 3e-3], [('mqar', 1, 8, 4)],
        steps=2, softmax_steps=2, n_seeds=2, device='cpu',
        val_every=1, val_batches=1, val_batch=1)
    assert best == 1e-3
    cell = record['lr_grid']['0.001']
    # Last finite tail values: [0.2,0.3] -> 0.25 and [0.4,0.6] -> 0.5.
    assert cell['mean_val_tail_acc'] == pytest.approx(0.375)
    assert cell['mean_final_val_acc'] == pytest.approx(0.6)


def test_benchmark_input_seed_aliases_the_compiled_chunk_row():
    from run_benchmark import benchmark_input_seed
    assert (benchmark_input_seed('kda_chunk_compiled', 64)
            == benchmark_input_seed('kda_chunk', 64))
    assert (benchmark_input_seed('kda_chunk', 64)
            != benchmark_input_seed('kda_rec', 64))


def test_directional_contrast_contradicted_and_one_sided_exact_p():
    """A unanimous negative sweep falsifies a directional prediction.

    The exact one-sided sign-flip p for n unanimous pairs is 2^-n (the
    old 2^-(n-1) was the two-sided value). With n=5 and one contrast the
    corrected alpha is 0.05, so the negative sweep must be significant
    and labelled ``contradicted`` rather than ``not_supported``.
    """
    rows = [
        _matrix_row('KDA-A', 'kda', (0.50, 0.50, 0.50, 0.50, 0.50)),
        _matrix_row('KDA-B', 'kda', (0.10, 0.10, 0.10, 0.10, 0.10)),
    ]
    contrasts = _matrix_contrast_report(rows)
    summary = _annotate_contrast_significance(contrasts)
    c = contrasts[0]
    assert c['mean_diff'] == pytest.approx(-0.4)
    assert c['t_stat'] is None
    assert c['significant_bonferroni'] is True
    assert c['significant_negative'] is True
    assert c['significant_positive'] is False
    assert c['hypothesis_status'] == 'contradicted'
    assert summary['n_contradicted'] == 1
    assert summary['n_supported'] == 0


def test_two_sided_contrast_planning_uses_the_tail_alpha_once():
    """A two-sided row's MDE must use its one-sided tail threshold once.

    The significance test uses the upper-tail quantile at alpha/(2*n); the
    planning helper also halves a two-sided TEST level internally, so the
    reported MDE has to be computed at alpha/n, not at alpha/(2*n) again.
    """
    from statistics import NormalDist
    contrasts = [{
        'predicted_direction': 'two_sided',
        'n_common_seeds': 5,
        'mean_diff': 0.20,
        'sd_diff': 0.10,
        't_stat': 0.20 / (0.10 / (5 ** 0.5)),
    }]
    _annotate_contrast_significance(contrasts)
    c = contrasts[0]
    assert c['test'] == 'paired_two_sided_t'
    assert c['alpha_bonferroni'] == pytest.approx(0.05 / 2.0)
    z_crit = NormalDist().inv_cdf(1.0 - c['alpha_bonferroni'])
    z_power = NormalDist().inv_cdf(0.8)
    expected = ((z_crit + z_power) * 0.10 / (5 ** 0.5))
    assert c['min_detectable_effect'] == pytest.approx(expected)
    # d_z = 2; required n from the same corrected threshold.
    expected_n = 2  # ceil(((z_crit + z_power) / 2)^2) stays below 2 here
    assert c['seeds_required_80pct_power'] == expected_n


def test_contrast_tost_separates_equivalent_from_different():
    """A non-significant contrast is not equivalence; TOST decides it.

    One route with only A/B present yields a single-contrast family, so
    the Bonferroni correction is the identity and the verdicts are exact.
    """
    rows = [
        _matrix_row('KDA-A', 'kda', (0.30, 0.31)),
        _matrix_row('KDA-B', 'kda', (0.30, 0.30)),
    ]
    contrasts = _matrix_contrast_report(rows)
    assert len(contrasts) == 1 and contrasts[0]['contrast'] == (
        'B-A (CSA replaces memory)')
    assert contrasts[0]['sd_diff'] > 0.0
    summary = _annotate_contrast_significance(
        contrasts, equivalence_margin=CONTRAST_EQUIVALENCE_MARGIN)
    c = contrasts[0]
    assert c['tost_equivalent'] is True
    assert c['equivalence_verdict'] == 'equivalent'
    assert c['tost_p_value'] < c['tost_alpha_bonferroni']
    assert c['tost_ci_equiv_lo'] > -CONTRAST_EQUIVALENCE_MARGIN
    assert c['tost_ci_equiv_hi'] < CONTRAST_EQUIVALENCE_MARGIN
    assert summary['n_equivalent'] == 1
    assert summary['tost']['margin'] == CONTRAST_EQUIVALENCE_MARGIN
    assert summary['tost']['n_equivalent'] == 1

    # Same layout, a large unanimous difference (zero-variance diffs):
    # clearly NOT equivalent, even though the row is also significant.
    far_rows = [
        _matrix_row('KDA-A', 'kda', (0.20, 0.22)),
        _matrix_row('KDA-B', 'kda', (0.80, 0.82)),
    ]
    far = _matrix_contrast_report(far_rows)
    _annotate_contrast_significance(
        far, equivalence_margin=CONTRAST_EQUIVALENCE_MARGIN)
    assert far[0]['sd_diff'] == 0.0
    assert far[0]['tost_equivalent'] is False
    assert far[0]['equivalence_verdict'] == 'not_equivalent'
    assert far[0]['tost_p_value'] == 1.0


def test_contrast_annotation_without_margin_makes_no_equivalence_claim():
    rows = [
        _matrix_row('KDA-A', 'kda', (0.30, 0.31)),
        _matrix_row('KDA-B', 'kda', (0.30, 0.30)),
    ]
    contrasts = _matrix_contrast_report(rows)
    summary = _annotate_contrast_significance(contrasts)
    assert summary['tost'] is None
    assert summary['equivalence_margin'] is None
    assert summary['n_equivalent'] == 0
    assert 'tost_equivalent' not in contrasts[0]
    assert 'not evidence of no effect' in summary['verdict_scope']
    for bad_margin in (0.0, float('inf'), 10 ** 400):
        with pytest.raises(ValueError):
            _annotate_contrast_significance(
                contrasts, equivalence_margin=bad_margin)


# ---------------------------------------------------------------------------
# Incremental decoding: chunk-schedule invariance
# ---------------------------------------------------------------------------

def test_chunked_prefill_matches_one_shot_forward():
    torch.manual_seed(11)
    d_model, length = 32, 10
    x = torch.randn(1, length, d_model) * 0.1
    factories = [
        ('csa', lambda: CSAAttnDecoding(d_model, topk=100)),
        ('hca', lambda: HCAAttnDecoding(d_model)),
        ('hybrid_kda', lambda: HybridDecoding(d_model, total_layers=5,
                                              csa_topk=100)),
        ('hybrid_mamba2', lambda: HybridDecoding(
            d_model, total_layers=5, csa_topk=100, memory_type='mamba2')),
    ]
    schedules = [[length], [1] * length, [4, 3, 3]]
    for name, factory in factories:
        model = factory().eval()
        with torch.no_grad():
            reference = None
            worst = 0.0
            for schedule in schedules:
                model.reset()
                outs, pos = [], 0
                for chunk in schedule:
                    chunk = min(chunk, length - pos)
                    if chunk <= 0:
                        break
                    outs.append(model(x[:, pos:pos + chunk]))
                    pos += chunk
                if pos < length:
                    outs.append(model(x[:, pos:]))
                y = torch.cat(outs, dim=1)
                if reference is None:
                    reference = y
                worst = max(worst, (y - reference).abs().max().item())
        assert worst < 1e-5, (
            f'{name}: chunked prefill diverged from the one-shot forward '
            f'(max diff {worst:.3e})')


def test_incremental_cache_op_sets_are_consistent():
    assert set(HYBRID_OPS) == {'hybrid', 'hybrid_mamba2', 'hybrid_rwkv7'}
    assert set(INCREMENTAL_CACHE_OPS) == set(HYBRID_OPS) | {'csa', 'hca'}


# ---------------------------------------------------------------------------
# Output registration (stale-file purge + schema gate)
# ---------------------------------------------------------------------------

def test_matrix_contrasts_output_registered(tmp_path, monkeypatch):
    import run_all

    rel = 'results/exp5b_ablation_matrix_contrasts.json'
    outputs = run_all._EXPERIMENT_OUTPUTS['exp5b_ablation_matrix_smoke']
    assert rel in outputs
    assert rel not in run_all._RAW_SIDE_FILES
    monkeypatch.chdir(tmp_path)
    os.makedirs('results', exist_ok=True)
    for entry in outputs:
        with open(entry, 'w', encoding='utf-8') as f:
            f.write('{}')
    run_all._purge_stale_outputs('exp5b_ablation_matrix_smoke')
    assert all(not os.path.exists(entry) for entry in outputs)


def test_matrix_rows_survive_json_roundtrip_validation(tmp_path):
    """The contract holds on RELOADED rows (tuples -> lists, sanitized)."""
    rows = [
        _matrix_row('KDA-D', 'kda', (0.8, 0.9)),
        _matrix_row('MAMBA2-D', 'mamba2', (0.5, 0.55)),
    ]
    meta = {'expected_groups': ['KDA-D', 'MAMBA2-D'], 'total_layers': 10}
    path = tmp_path / 'matrix.json'
    rs.write_checked(str(path), rows, kind='exp5b_ablation_matrix',
                     metadata=meta)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    assert rs.validate_payload(data) == []


# ---------------------------------------------------------------------------
# Exp 2 / latency-profile / Exp 9 contracts (uncertainty + join audit)
# ---------------------------------------------------------------------------

def _exp2_row(**overrides):
    row = {
        'T': 128, 'op': 'softmax', 'device': 'cpu', 'repeats': 5,
        'time_ms': 1.0, 'time_min_ms': 0.8, 'time_max_ms': 1.2,
        'time_std_ms': 0.1, 'time_mean_ms': 1.0, 'time_q10_ms': 0.9,
        'time_q90_ms': 1.1, 'time_ci95_low_ms': 0.95,
        'time_ci95_high_ms': 1.05, 'time_ci95_halfwidth_ms': 0.05,
        'time_cv': 0.1, 'time_n': 5,
        'time_ci_method': 'percentile_bootstrap_median',
        'time_bootstrap_reps': 2000,
    }
    row.update(overrides)
    return row


def test_row_contract_exp2_requires_ordered_uncertainty():
    good = _exp2_row()
    assert rs.validate_rows('exp2_benchmark', [good], {}) == []
    assert rs.validate_rows('exp2_benchmark', [{
        'T': 128, 'op': 'softmax', 'device': 'cpu', 'repeats': 5,
        'time_ms': 1.0, 'time_min_ms': 0.8, 'time_max_ms': 1.2,
        'time_std_ms': 0.1}], {}) == []  # legacy row without CI block
    for bad, needle in (
        (_exp2_row(time_q10_ms=1.2), 'median outside'),
        (_exp2_row(time_ci95_low_ms=1.1, time_ci95_high_ms=1.0), 'low_ms >'),
        (_exp2_row(time_n=4), 'time_n must equal repeats'),
        (_exp2_row(time_ci95_halfwidth_ms=0.5), 'halfwidth'),
    ):
        violations = rs.validate_rows('exp2_benchmark', [bad], {})
        assert any(needle in v for v in violations), violations
    assert any('duplicate' in v for v in rs.validate_rows(
        'exp2_benchmark', [good, good], {}))


def _profile_row(**overrides):
    row = {
        'T': 128, 'mode': 'sequential', 'chunk_size': None,
        'time_ms': 1.0, 'time_min_ms': 0.8, 'time_max_ms': 1.2,
        'time_std_ms': 0.1, 'time_q10_ms': 0.9, 'time_q90_ms': 1.1,
        'time_ci95_low_ms': 0.95, 'time_ci95_high_ms': 1.05,
        'times_ms': [0.9, 1.0, 1.1], 'n': 3, 'finite': True,
        'parity_max_abs': 0.0, 'parity_ok': True,
    }
    row.update(overrides)
    return row


def test_row_contract_latency_profile_parity_gate():
    meta = {'parity_tol': 1e-4}
    assert rs.validate_rows('mamba2_scan_latency_profile',
                            [_profile_row()], meta) == []
    row = _profile_row(mode='chunked', chunk_size=16,
                       parity_max_abs=1e-3)
    violations = rs.validate_rows('mamba2_scan_latency_profile', [row], meta)
    assert any('parity_ok' in v for v in violations), violations
    violations = rs.validate_rows(
        'mamba2_scan_latency_profile',
        [_profile_row(n=2)], meta)
    assert any('n=' in v for v in violations), violations
    for field in ('T', 'mode'):
        violations = rs.validate_rows(
            'mamba2_scan_latency_profile',
            [_profile_row(**{field: []})], meta)
        assert any(field in v for v in violations), violations
    violations = rs.validate_rows(
        'mamba2_scan_latency_profile',
        [_profile_row(mode='chunked', chunk_size=[])], meta)
    assert any('chunk_size' in v for v in violations), violations
    # A non-numeric metadata tolerance must be reported once, not raised.
    violations = rs.validate_rows(
        'mamba2_scan_latency_profile', [_profile_row()],
        {'parity_tol': 'not-a-number'})
    assert any('parity_tol' in v for v in violations), violations


def _exp9_contract_rows():
    commit = 'a' * 40
    audit = {
        'section': 'source_audit', 'source': 'exp5_ablation',
        'path': 'results/exp5_ablation.json', 'exists': True,
        'readable': True, 'schema_version': '0.0', 'n_rows': 1,
        'sha256': '0' * 64, 'git_commit': commit, 'git_dirty': False,
        'captured_at_utc': '2026-01-01T00:00:00+00:00',
        'seeds': [42], 'n_seeds': 1, 'backends': {},
        'provenance_source': 'sidecar', 'error': None,
    }
    cross = {
        'section': 'source_audit', 'source': 'cross_source',
        'status': 'comparable', 'comparable_for_ranking': True,
        'n_sources_requested': 1, 'n_sources_readable': 1,
        'missing_sources': [], 'missing_provenance': [],
        'git_commits': [commit], 'dirty_sources': [],
        'seed_status': 'shared', 'shared_seeds': [42],
        'n_shared_seeds': 1, 'backends_consistent': True,
        'backend_mismatches': [], 'warnings': [],
    }
    return [
        {'section': 'budget_curve', 'row_kind': 'op', 'geometry': 'mqar',
         'T': 16, 'name': 'csa', 'total_read_elements': 100.0},
        {'section': 'budget_curve', 'row_kind': 'layout', 'geometry': 'mqar',
         'T': 16, 'name': '3:1:1', 'ratio': [3, 1, 1],
         'total_read_elements': 120.0},
        {'section': 'decode_budget_curve', 'row_kind': 'op',
         'geometry': 'mqar', 't': 16, 'name': 'csa',
         'total_read_elements': 100.0},
        {'section': 'decode_sweep', 'csa_topk': 0, 'prefill_len': 256,
         'n_decode': 2, 'repeats': 1, 'prefill_ms': 1.0,
         'mean_decode_ms_per_token': 0.5,
         'total_read_elements_per_query': 100.0},
        {'section': 'quality_join', 'source': 'exp5_ablation',
         'label': '3:1:1', 'context_T': 16, 'mean_acc': 0.5,
         'ci95_acc': 0.01, 'n_seeds_ok': 1, 'mean_fwd_ms': None,
         'total_read_elements_per_query': 120.0,
         'source_git_commit': commit, 'source_git_dirty': False,
         'source_seeds': [42], 'source_sha256': '0' * 64,
         'source_schema_version': '0.0'},
        {'section': 'ranking_summary', 'geometry': 'mqar', 'context_T': 16,
         'decode_budget_position': 16, 'n_layouts': 3,
         'labels': ['3:1:1', '4:1:0', '2:1:2'],
         'layout_order_by_budget_ascending': ['3:1:1', '4:1:0', '2:1:2'],
         'layout_order_by_decode_budget_ascending': ['4:1:0', '3:1:1',
                                                     '2:1:2'],
         'spearman_acc_vs_budget': 0.5,
         'spearman_acc_vs_decode_budget': -0.5,
         'source_comparability': 'comparable', 'source_git_commits': [commit],
         'source_shared_seeds': [42], 'comparable_across_sources': True},
        audit, cross,
    ]


def test_row_contract_exp9_audits_cross_source_ranking():
    rows = _exp9_contract_rows()
    assert rs.validate_rows('exp9_budget_frontier', rows, {}) == []
    # The same ratio on the two memory routes is two rows, not a duplicate.
    both_routes = [dict(r) for r in rows]
    both_routes.append(dict(
        both_routes[1], memory_op='mamba2', total_read_elements=140.0))
    sweep_idx = next(i for i, r in enumerate(both_routes)
                     if r['section'] == 'decode_sweep')
    both_routes.append(dict(both_routes[sweep_idx], memory_op='mamba2'))
    assert rs.validate_rows('exp9_budget_frontier', both_routes, {}) == []
    bad_route = [dict(r) for r in both_routes]
    bad_route[-1] = dict(bad_route[-1], memory_op='rwkv7')
    assert any('memory_op' in v for v in
               rs.validate_rows('exp9_budget_frontier', bad_route, {}))
    duplicate = [dict(r) for r in both_routes]
    duplicate.append(dict(duplicate[-1]))
    assert any('duplicate key' in v for v in
               rs.validate_rows('exp9_budget_frontier', duplicate, {}))
    # A mixed-revision verdict must name at least two commits.
    bad = [dict(r) for r in rows]
    bad[-1] = dict(bad[-1], status='mixed_revisions')
    violations = rs.validate_rows('exp9_budget_frontier', bad, {})
    assert any('mixed_revisions' in v for v in violations), violations
    # Ranking rows cannot claim comparability with a non-bool flag.
    bad = [dict(r) for r in rows]
    idx = next(i for i, r in enumerate(bad)
               if r['section'] == 'ranking_summary')
    bad[idx] = dict(bad[idx], comparable_across_sources='yes')
    assert any('comparable_across_sources' in v for v in
               rs.validate_rows('exp9_budget_frontier', bad, {}))
    # Every joined source needs a source_audit row.
    bad = [r for r in rows if r.get('source') != 'exp5_ablation'
           or r['section'] != 'source_audit']
    violations = rs.validate_rows('exp9_budget_frontier', bad, {})
    assert any('no audit row' in v for v in violations), violations
    # Partial seed metadata is a first-class non-comparable verdict.
    partial = [dict(r) for r in rows]
    cross_idx = next(i for i, r in enumerate(partial)
                     if r.get('source') == 'cross_source')
    partial[cross_idx] = dict(
        partial[cross_idx], status='partial_seeds',
        comparable_for_ranking=False, seed_status='partial')
    rank_idx = next(i for i, r in enumerate(partial)
                    if r['section'] == 'ranking_summary')
    partial[rank_idx] = dict(
        partial[rank_idx], source_comparability='partial_seeds',
        comparable_across_sources=False)
    assert rs.validate_rows('exp9_budget_frontier', partial, {}) == []
    # Different dirty working trees at the same commit block the ranking and
    # must carry at least two digests as evidence.
    mixed_trees = [dict(r) for r in rows]
    cross_idx = next(i for i, r in enumerate(mixed_trees)
                     if r.get('source') == 'cross_source')
    mixed_trees[cross_idx] = dict(
        mixed_trees[cross_idx], status='mixed_source_trees',
        comparable_for_ranking=False,
        source_tree_sha256s=['d' * 64, 'e' * 64])
    rank_idx = next(i for i, r in enumerate(mixed_trees)
                    if r['section'] == 'ranking_summary')
    mixed_trees[rank_idx] = dict(
        mixed_trees[rank_idx], source_comparability='mixed_source_trees',
        comparable_across_sources=False)
    assert rs.validate_rows('exp9_budget_frontier', mixed_trees, {}) == []
    one_digest = [dict(r) for r in mixed_trees]
    one_digest[cross_idx] = dict(
        one_digest[cross_idx], source_tree_sha256s=['d' * 64])
    assert any('mixed_source_trees' in v for v in
               rs.validate_rows('exp9_budget_frontier', one_digest, {}))


def test_row_contract_exp9_malformed_types_do_not_crash():
    import copy

    rows = _exp9_contract_rows()
    for index, patch in (
            (0, {'section': []}),
            (0, {'row_kind': []}),
            (0, {'geometry': []}),
            (0, {'name': []}),
    ):
        bad = copy.deepcopy(rows)
        bad[index] = dict(bad[index], **patch)
        violations = rs.validate_rows('exp9_budget_frontier', bad, {})
        assert violations, patch

    # An error row keeps the malformed key out of the duplicate-detection
    # set: hashing a list used to abort validation with a TypeError instead
    # of reporting the csa_topk violation.
    bad = copy.deepcopy(rows)
    idx = next(i for i, r in enumerate(bad)
               if r['section'] == 'decode_sweep')
    bad[idx] = {k: v for k, v in bad[idx].items()
                if k not in ('prefill_ms', 'mean_decode_ms_per_token',
                             'total_read_elements_per_query')}
    bad[idx].update(csa_topk=[], error='boom')
    violations = rs.validate_rows('exp9_budget_frontier', bad, {})
    assert any('csa_topk' in v for v in violations), violations


def test_row_contracts_reject_unhashable_metadata_entries():
    """Malformed metadata lists must be violations, not TypeError crashes."""
    violations = rs.validate_rows(
        'exp5b_ablation_matrix', [],
        {'expected_groups': [['g1']], 'total_layers': 2})
    assert any('expected_groups' in v for v in violations), violations

    violations = rs.validate_rows(
        'exp6_decoding', [],
        {'prefill_lens': [[1]], 'ops': [['kda']]})
    assert any('prefill_lens' in v for v in violations), violations
    assert any('metadata.ops' in v for v in violations), violations

    violations = rs.validate_rows(
        'exp16_paired_stats_calibration', [],
        {'settings': {'reps': [], 'alpha': 'not-a-level'}})
    assert any('settings.reps' in v for v in violations), violations
    assert any('settings.alpha' in v for v in violations), violations


def test_pairwise_tests_ignore_nonfinite_per_seed_accuracy():
    """A diverged seed must not abort the paired-comparison report.

    ``train_multi_seed`` filters non-finite seeds from its aggregate, but the
    ``per_seed`` list still carries the raw record. The paired tests used to
    treat ``final_acc is not None`` as usable, so one NaN made
    ``multi_seed_stats`` raise and the whole quality report disappear. The
    seed must instead be excluded from the common population, exactly like
    the aggregate path does.
    """
    from run_quality import _finite_per_seed_acc, _pairwise_op_comparisons

    records = [
        {'seed': 1, 'final_acc': 0.4},
        {'seed': 2, 'final_acc': float('nan')},
        {'seed': 3, 'final_acc': float('inf')},
        {'seed': 4, 'final_acc': '0.5'},
        {'seed': 5, 'final_acc': True},
        {'seed': 6, 'final_acc': None},
        {'seed': 7, 'final_acc': 0.7, 'error': 'diverged'},
        {'seed': 8, 'final_acc': 0.8},
    ]
    assert _finite_per_seed_acc(records) == {1: 0.4, 8: 0.8}
    assert _finite_per_seed_acc(None) == {}
    assert _finite_per_seed_acc([None, 'x', {'seed': True, 'final_acc': 1.0}]) == {}

    rows = [
        {'op': 'a', 'task': 'mqar', 'n_kv': 1, 'mean_acc': 0.5,
         'per_seed': [{'seed': 1, 'final_acc': 0.4},
                      {'seed': 2, 'final_acc': float('nan')},
                      {'seed': 3, 'final_acc': 0.6}]},
        {'op': 'b', 'task': 'mqar', 'n_kv': 1, 'mean_acc': 0.4,
         'per_seed': [{'seed': 1, 'final_acc': 0.3},
                      {'seed': 2, 'final_acc': 0.5},
                      {'seed': 3, 'final_acc': 0.5}]},
    ]
    report = _pairwise_op_comparisons(
        rows, equivalence_margin=0.05, n_boot=50, n_permutations=50)
    pair = report['cells']['mqar/n_kv=1']['pairs']['a~b']
    assert pair['common_seeds'] == [1, 3]
    assert pair['n_pairs'] == 2
    assert abs(pair['mean_diff'] - 0.1) < 1e-12

    # A malformed metric is not a measurement either: with only one valid
    # pair the comparison is explicitly skipped instead of crashing.
    rows[0]['per_seed'] = [
        {'seed': 1, 'final_acc': '0.4'},
        {'seed': 2, 'final_acc': float('nan')},
        {'seed': 3, 'final_acc': True},
    ]
    report = _pairwise_op_comparisons(rows, n_boot=20, n_permutations=20)
    pair = report['cells']['mqar/n_kv=1']['pairs']['a~b']
    assert pair['verdict'] == 'insufficient_paired_seeds'


def test_lower_tail_bonferroni_quantile_is_the_mirror():
    """``alpha > 0.5`` must return the (negative) lower-tail t quantile.

    The dependency-free inverse-CDF path only brackets the upper tail, so
    it used to return a near-zero positive value for every lower-tail
    request. The t distribution is symmetric: ``t_{1-alpha} = -t_alpha``.
    """
    from multi_seed_stats import _t_quantile
    from run_quality import _bonferroni_crit_q

    for n in (2, 3, 5, 10, 100, 101):
        for alpha in (0.51, 0.6, 0.75, 0.9, 0.975, 0.999):
            got = _bonferroni_crit_q(n, alpha=alpha)
            want = _t_quantile(1.0 - alpha, n - 1)
            assert abs(got - want) <= 1e-7 * max(1.0, abs(want)), (
                n, alpha, got, want)

    for bad in (0, 1, -0.1, 1.1, float('nan'), True, '0.05', 10 ** 400):
        try:
            _bonferroni_crit_q(5, alpha=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'alpha={bad!r} was silently accepted')

    # n < 2 remains the documented "undefined" case.
    assert _bonferroni_crit_q(1, alpha=0.05) is None


def test_bonferroni_extreme_tail_is_not_truncated_at_t_squared_overflow():
    """The Cauchy tail must stay accurate past ``t**2 = inf``.

    For dof=1 the survival function is ``1/(pi*t)`` in the far tail, so
    ``t_{1-alpha} = 1/(pi*alpha)``.  Computing ``x = dof/(dof+t**2)`` used
    to underflow once ``t`` exceeded ~1e154, ending the bracket early and
    returning a critical value dozens of orders of magnitude too small.
    """
    from run_quality import _bonferroni_crit_q

    for alpha in (1e-100, 1e-200, 1e-300):
        got = _bonferroni_crit_q(2, alpha=alpha)
        want = 1.0 / (math.pi * alpha)
        assert abs(got / want - 1.0) < 1e-10, (alpha, got, want)


def test_bonferroni_huge_dof_and_near_median_are_stable():
    """Huge dof must use the normal limit instead of flattening the tail.

    ``x = dof/(dof+t**2)`` rounds to 1.0 for huge dof even at the median;
    the dependency-free path then returned 8.0 instead of ~1.645 (or
    overflowed for a JSON-sized ``n``).  A much smaller dof exposed the
    same ratio at near-median alpha, where ``x`` also rounded to 1.0.
    """
    from statistics import NormalDist

    import quality.stats as stats
    from run_quality import _bonferroni_crit_q, _t_crit_975

    # The guard applies regardless of whether scipy is installed.
    for n in (10 ** 12, 10 ** 18):
        for alpha in (0.05, 1e-6, 0.5):
            got = _bonferroni_crit_q(n, alpha=alpha)
            want = -NormalDist().inv_cdf(alpha)
            assert abs(got - want) <= 1e-15 * max(1.0, abs(want)), (
                n, alpha, got, want)
    huge = 10 ** 400
    assert abs(_bonferroni_crit_q(huge, alpha=0.05)
               - -NormalDist().inv_cdf(0.05)) <= 1e-15
    assert abs(_t_crit_975(huge)
               - NormalDist().inv_cdf(0.975)) <= 1e-15

    # Dependency-free path: with dof=1e6 the near-median quantile is tiny,
    # so the old x ratio rounded to 1.0 and SF stayed flat at 0.5.  The
    # stable y complement recovers the linear normal tail
    # t ~ (0.5 - alpha) * sqrt(2*pi).
    saved = (stats._T_PP, stats._T_ISF)
    stats._T_PP = stats._T_ISF = False
    try:
        got = _bonferroni_crit_q(10 ** 6, alpha=0.5 - 1e-10)
    finally:
        stats._T_PP, stats._T_ISF = saved
    want = 1e-10 * math.sqrt(2.0 * math.pi)
    assert abs(got - want) <= 1e-6 * want, (got, want)


def test_seed_accuracy_map_ignores_malformed_seed_metrics():
    """Paired contrasts use only finite numeric per-seed accuracies."""
    from run_ablation import _seed_accuracy_map

    result = {'per_seed': [
        {'seed': 1, 'final_acc': 0.4},
        {'seed': 2, 'final_acc': float('nan')},
        {'seed': 3, 'final_acc': float('inf')},
        {'seed': 4, 'final_acc': '0.5'},
        {'seed': 5, 'final_acc': True},
        {'seed': 6, 'final_acc': None},
        {'seed': 7, 'final_acc': 0.7, 'error': 'diverged'},
        {'seed': 8, 'final_acc': 0.8},
        {'seed': '9', 'final_acc': 0.9},
        {'seed': True, 'final_acc': 0.95},
        'not-a-row',
    ]}
    assert _seed_accuracy_map(result) == {1: 0.4, 8: 0.8}
    assert _seed_accuracy_map({'per_seed': None}) == {}


def test_matrix_design_audit_ignores_malformed_per_seed_rows():
    """The seed-alignment audit must skip bad rows, not crash or count them.

    ``_finite_seed_set`` feeds the matrix design audit's "all groups share
    the requested seeds" check. It must use the same survivorship rule as
    ``_seed_accuracy_map``: malformed records, non-int seed ids and
    non-finite/string/bool accuracies are not successful seeds.
    """
    from run_ablation import _finite_seed_set, _row_design_field

    row = {'per_seed': [
        {'seed': 1, 'final_acc': 0.4},
        {'seed': 2, 'final_acc': float('nan')},
        {'seed': 3, 'final_acc': '0.5'},
        {'seed': 4, 'final_acc': True},
        {'seed': '5', 'final_acc': 0.9},
        {'seed': 6, 'final_acc': 0.8},
        {'seed': 7, 'final_acc': 0.7, 'error': 'diverged'},
        'not-a-row',
        None,
    ]}
    assert _finite_seed_set(row) == {1, 6}
    assert _finite_seed_set({'per_seed': None}) == set()
    assert _finite_seed_set(None) == set()

    # The recipe-field fallback has the same malformed-row contract and
    # still gives the aggregate field precedence.
    assert _row_design_field({'per_seed': ['x', None,
                                           {'lr': 0.2}]}, 'lr') == 0.2
    assert _row_design_field({'lr': 0.1,
                              'per_seed': [{'lr': 0.2}]}, 'lr') == 0.1
    assert _row_design_field({'per_seed': ['x', None]}, 'lr') is None
    assert _row_design_field(None, 'lr') is None


def test_validation_tail_score_skips_malformed_and_nonfinite_points():
    """The baseline-LR tuner must not crash on a malformed per-seed record.

    ``_validation_tail_score`` is fed result rows that may contain error
    stubs or hand-edited placeholders. The rest of ``run_ablation`` skips
    such records in paired contrasts; the tuning selector must apply the
    same rule, and a NaN/Inf point must not poison the plateau mean.
    """
    from run_ablation import _validation_tail_score

    assert _validation_tail_score(None) is None
    assert _validation_tail_score({'per_seed': None}) is None
    record = {'per_seed': [
        'not-a-row',
        None,
        {'val_history': None},
        {'val_history': ['not-a-point', None, {'loss': 1.0}]},
        {'val_history': [{'acc': 0.2}, {'acc': 0.4},
                         {'acc': float('nan')}, {'acc': '0.9'}]},
        {'val_history': [{'acc': 0.6}, {'acc': float('inf')},
                         {'acc': 0.8}]},
    ]}
    # Last 3 points per row: row 5 -> [0.4, NaN, '0.9'] keeps [0.4]; row 6
    # -> [0.6, Inf, 0.8] keeps [0.6, 0.8]. Mean of the two per-seed
    # plateaus = (0.4 + 0.7) / 2 = 0.55.
    score = _validation_tail_score(record, tail_k=3)
    assert score == pytest.approx(0.55)
    assert math.isfinite(score)
    # No usable point -> None (the tuner falls back to the grid ordering).
    assert _validation_tail_score(
        {'per_seed': [{'val_history': [{'acc': float('nan')}]}]}) is None

def test_softmax_gqa_prefill_uses_the_requested_head_width():
    """``prefill_flops('softmax_gqa')`` must derive n_q from ``K``.

    The module default ``K=GQA_HEAD_DIM=128`` hides the bug; a caller that
    prices a 64-wide head geometry used to keep ``n_q=d//128`` and
    undercount every query-side term (core, q_proj, o_proj).
    """
    from run_kv_cache import prefill_flops

    T, d, H, V = 10, 4096, 8, 128
    for K in (128, 64, 32):
        got = prefill_flops('softmax_gqa', T, d=d, H=H, K=K, V=V)
        n_q = d // K
        entries = T * (T + 1) // 2
        core = 2 * entries * n_q * (K + V)
        proj = 2 * T * d * (n_q * K + H * K + H * V)
        out_proj = 2 * T * n_q * V * d
        assert got == core + proj + out_proj, (K, got, core + proj + out_proj)


def test_schema_finite_numbers_reject_huge_json_integers():
    huge = 10 ** 400
    assert rs._finite_number(huge) is False
    # A row with an out-of-float-range integer must produce a contract
    # violation, not abort the whole validation pass.
    row = {'group': 'g', 'memory': 'kda', 'n_kv': 1, 'mean_acc': huge,
           'n_seeds_ok': 1, 'layout': 'KDA-KDA-KDA-CSA-HCA',
           'n_params': 10, 'n_layers': 5, 'per_seed': []}
    violations = rs.validate_rows('exp5b_ablation_matrix', [row])
    assert violations
    assert 'mean_acc' in violations[0]


def test_schema_mixed_expected_group_labels_report_a_violation():
    rows = [{'group': 'a', 'memory': 'kda', 'n_kv': 1, 'mean_acc': 1.0,
             'per_seed': [{'final_acc': 1.0, 'final_loss': 0.5}],
             'n_seeds_ok': 1, 'layout': 'KDA-CSA-HCA',
             'n_params': 1, 'n_layers': 5}]
    violations = rs.validate_rows(
        'exp5b_ablation_matrix', rows,
        {'expected_groups': [1, 'b'], 'total_layers': 5})
    assert violations
    assert 'matrix incomplete' in violations[0]

def test_paired_contrast_keeps_a_finite_mean_under_sum_overflow():
    """Two finite large wins must not produce an infinite mean contrast.

    ``sum([1.7e308, 1.7e308]) / 2`` overflows to ``inf`` although the
    paired mean difference is the finite ``1.7e308``. The contrast row is
    written into the strict-JSON ablation report, so the helper must use
    the overflow-safe mean (and keep the ordinary sample SD).
    """
    a = {0: 1.7e308, 1: 1.7e308}
    b = {0: 0.0, 1: 0.0}
    mean_diff, t_stat, n, unanimous, sd = _paired_contrast(a, b)
    assert math.isfinite(mean_diff)
    assert mean_diff == pytest.approx(1.7e308)
    assert n == 2 and unanimous == 'positive'
    assert sd == 0.0 and t_stat is None


def test_paired_contrast_keeps_an_unrepresentable_sd_undefined():
    """A finite mean with an overflowed SD is not a zero-variance sweep."""
    d = 1.7e308
    a = {0: d, 1: -d, 2: d}
    b = {0: 0.0, 1: 0.0, 2: 0.0}
    mean_diff, t_stat, n, unanimous, sd = _paired_contrast(a, b)
    assert math.isfinite(mean_diff)
    assert n == 3 and unanimous == 'mixed'
    assert sd is None
    assert t_stat is None


def test_per_seed_filters_skip_oversized_json_integers():
    """A JSON int too large for float is a malformed seed, not a crash."""
    from run_quality import _finite_per_seed_acc
    from run_ablation import _finite_seed_set, _seed_accuracy_map
    rows = [{'seed': 1, 'final_acc': 10 ** 400},
            {'seed': 2, 'final_acc': 0.5}]
    assert _finite_per_seed_acc(rows) == {2: 0.5}
    assert _seed_accuracy_map({'per_seed': rows}) == {2: 0.5}
    assert _finite_seed_set({'per_seed': rows}) == {2}


def test_curve_summaries_skip_oversized_json_integers():
    """Curve/gradient summaries must not crash on an out-of-float JSON int."""
    from run_ablation import (_aggregate_curve_diagnosis,
                              _curve_diagnosis_summary)
    huge = 10 ** 400
    per_seed = [{
        'generalization_gap': huge,
        'gradient_norms': {'max_over_clip': huge, 'notes': []},
        'curve_diagnosis': {'verdict': 'overfitting'},
        'best_val': {'step': 1},
    }, {
        'generalization_gap': 0.1,
        'gradient_norms': {'max_over_clip': 2.0, 'notes': []},
        'curve_diagnosis': {'verdict': 'overfitting'},
        'best_val': {'step': 1},
    }]
    diag = _aggregate_curve_diagnosis(per_seed)
    assert diag['mean_generalization_gap'] == pytest.approx(0.1)
    assert diag['max_gradient_over_clip'] == pytest.approx(2.0)
    json.dumps(diag, allow_nan=False)

    summary = _curve_diagnosis_summary([{
        'ratio': 'a', 'curve_verdict': 'overfitting',
        'mean_generalization_gap': huge,
        'max_gradient_over_clip': huge,
    }])
    assert summary['mean_generalization_gap'] is None
    assert summary['max_gradient_over_clip'] is None
    json.dumps(summary, allow_nan=False)



# ---------------------------------------------------------------------------
# Route x layout transfer interaction + capacity audit
# ---------------------------------------------------------------------------

def test_matrix_route_interaction_is_seed_paired_difference_in_differences():
    accmaps = {
        'kda': {'A': (0.20, 0.40), 'B': (0.50, 0.80),
                'C': (0.40, 0.60), 'D': (0.80, 0.90)},
        'mamba2': {'A': (0.10, 0.30), 'B': (0.20, 0.25),
                   'C': (0.30, 0.35), 'D': (0.50, 0.55)},
    }
    rows = [_matrix_row(f'{mem.upper()}-{g}', mem, accs)
            for mem, per_group in accmaps.items()
            for g, accs in per_group.items()]
    contrasts = _matrix_contrast_report(rows)
    row = next(c for c in contrasts
               if c.get('interaction_type') == 'route_x_layout_transfer')
    # (D-A)_KDA = [(.8-.2)+(.9-.4)]/2 = .55
    # (D-A)_MAMBA2 = [(.5-.1)+(.55-.3)]/2 = .325
    assert row['mean_diff'] == pytest.approx(0.225)
    assert row['n_common_seeds'] == 2
    assert row['predicted_direction'] == 'two_sided'
    assert row['candidate_group'] is None
    assert len(row['interaction_cells']) == 4
    assert _contrast_consistency_violations(rows, contrasts) == []

    tampered = [dict(c, mean_diff=(
        c['mean_diff'] + 0.1
        if c is row else c['mean_diff'])) for c in contrasts]
    assert any('I_route' in v for v in _contrast_consistency_violations(
        rows, tampered))


def test_matrix_capacity_audit_labels_operator_plus_capacity():
    rows = [
        _matrix_row('KDA-A', 'kda', (0.2, 0.4), n_params=1000),
        _matrix_row('KDA-B', 'kda', (0.5, 0.8), n_params=1400),
    ]
    contrasts = _matrix_contrast_report(rows)
    assert len(contrasts) == 1
    c = contrasts[0]
    assert c['capacity_matched'] is False
    assert c['capacity_status'] == 'candidate_larger'
    assert c['param_delta_pct'] == pytest.approx(40.0)
    summary = _annotate_contrast_significance(contrasts)
    assert c['claim_scope'] == 'operator_plus_capacity'
    assert c['capacity_confounded'] is True
    assert summary['n_capacity_confounded'] == 1

    # A matched pair keeps the same arithmetic but a capacity-neutral label.
    matched = [
        _matrix_row('KDA-A', 'kda', (0.2, 0.4), n_params=1000),
        _matrix_row('KDA-B', 'kda', (0.5, 0.8), n_params=1040),
    ]
    mcontrasts = _matrix_contrast_report(matched)
    _annotate_contrast_significance(mcontrasts)
    assert mcontrasts[0]['capacity_matched'] is True
    assert mcontrasts[0]['claim_scope'] == 'capacity_matched'
    assert mcontrasts[0]['capacity_confounded'] is False


def test_matrix_design_audit_reports_capacity_confound_without_failing_design():
    groups = build_experiment_matrix('main8', 10, ('kda',))
    params = {'A': 1000, 'B': 1400, 'C': 1300, 'D': 1700}
    rows = []
    for spec in groups:
        group = spec[0]
        unit = group.rsplit('-', 1)[-1]
        row = _matrix_row(group, 'kda', (0.2, 0.4),
                          n_params=params.get(unit, 1000))
        row.update({'steps': 100, 'train_batch': 16, 'lr': 3e-3,
                    'd_model': 32})
        rows.append(row)
    audit = _matrix_design_audit(
        rows, groups, preset='main8', n_kv_list=[1], n_seeds=2)
    assert audit['status'] == 'complete_factorial'
    cap = audit['capacity_audit']
    assert cap['status'] == 'capacity_confounded'
    assert cap['n_confounded'] == 3
    assert cap['n_matched'] == 0
    # The capacity audit labels the evidence; it does not invalidate the
    # factorial design or change any p-value.
    assert audit['violations'] == []

    matched_rows = [dict(r, n_params=1000) for r in rows]
    matched = _matrix_design_audit(
        matched_rows, groups, preset='main8', n_kv_list=[1], n_seeds=2)
    assert matched['capacity_audit']['status'] == 'capacity_matched'
    assert matched['capacity_audit']['n_confounded'] == 0


def test_two_sided_zero_variance_interaction_uses_exact_sign_floor():
    def row(n):
        return {
            'n_kv': 1, 'memory': 'kda vs mamba2',
            'contrast': 'I_route(D)=(D-A)_KDA - (D-A)_MAMBA2',
            'mean_diff': 0.2, 't_stat': None, 'sd_diff': 0.0,
            'n_common_seeds': n, 'unanimous': 'positive',
            'predicted_direction': 'two_sided', 'metric': 'final_acc',
            'direction': 'higher_is_better',
        }
    five = [row(5)]
    _annotate_contrast_significance(five)
    # Exact two-sided floor at n=5 is 2/2^5 = .0625 > .05.
    assert five[0]['significant_bonferroni'] is False
    # At n=6 the exact two-sided p 2/2^6 = .03125 is still above the
    # corrected .025 two-sided alpha; n=7 is the first rejection region.
    six = [row(6)]
    _annotate_contrast_significance(six)
    assert six[0]['significant_bonferroni'] is False
    seven = [row(7)]
    _annotate_contrast_significance(seven)
    assert seven[0]['significant_bonferroni'] is True
