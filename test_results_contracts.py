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
    _contrast_consistency_violations,
    _matrix_contrast_report,
    build_experiment_matrix,
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
    with pytest.raises(ValueError):
        _annotate_contrast_significance(contrasts, equivalence_margin=0.0)


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
