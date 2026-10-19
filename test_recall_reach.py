"""Regression tests for ``recall_reach.py`` (Exp 9: block-size x
layer-interleaving associative-recall reachability).

What is pinned here
-------------------
1. **Analytic reachability is hand-exact.** The two exact-recall paths
   (same compressed block strictly preceding the cue; sliding window
   ``[t-win+1, t]``) are hand-derived for a small plan and compared
   placement-by-placement, including block boundaries, the causal edge
   and the window edge. The CSA uniform-selection chance and the
   ``1/m + 1/m2`` budget grid are checked against closed forms.
2. **Plan extraction matches the built model.** ``plan_from_config`` /
   ``plan_from_model`` reproduce the layer kinds, compression factors,
   windows and top-k exactly; unresolved 'memory' tokens are rejected.
3. **Empirical indexer recall is a real property of the model.** With
   select-all ``csa_topk`` every resolved gold block MUST be hit (recall
   exactly 1.0); with sparse top-k the recall is bounded, the chance
   level is the closed form ``min(k, a) / a``, and — the critical pin —
   the recomputed per-layer indices equal the indices the operator
   actually used for EVERY CSA layer of a multi-CSA layout.
4. **Analytic vs empirical-free cross-check.** ``hca_block_coverage``'s
   resolved fraction over an explicit placement sweep equals the analytic
   ``block_hit_fraction`` of the HCA plan entry (same predicate, two
   independent code paths).
5. **The report is strict JSON and self-consistent.** A tiny trained
   report round-trips through ``json.dumps(allow_nan=False)``, carries
   one record per cell and seed, and keeps per-layer recall list lengths
   equal to the CSA count; the structural-only path skips training.
6. **Statistical design.** The mean/std/CI helper is exact, the default
   seed list has >= 3 seeds (the repo's claim policy), and a multi-seed
   report carries per-row mean +/- sample std for accuracy and indexer
   union recall plus the exploratory warning for short seed lists.

Run via pytest::

    pytest -q test_recall_reach.py

Or directly::

    python3 test_recall_reach.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import recall_reach as R  # noqa: E402
from ops_fused import HybridKCHAttention  # noqa: E402
from run_quality import make_long_range_batch  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _plan(seq_len=16):
    return [
        R.LayerSpec(0, 'csa', block=4, window=4, topk=2),
        R.LayerSpec(1, 'hca', block=8, window=4, topk=None),
        R.LayerSpec(2, 'kda', block=None, window=0, topk=None),
    ]


def test_structural_recall_hand_derived():
    """Placement-by-placement values match a hand derivation.

    T=16 -> cue=15, m=4, m2=8, win=4. CSA block path covers p in
    {0,1,2,4,5,6,8,9,10} (same block AND block < 15//4=3), the window
    covers p in {12,13} (15-p <= 3). HCA's block path covers p in
    {0..6} (same 8-block AND strictly preceding). p=7 and p=11 are
    therefore the only uncovered placements -> union = 12/14.
    """
    plan = _plan()
    res = R.structural_recall(plan, seq_len=16)
    _ok('n_placements = cue-1', res['n_placements'] == 14,
        f"{res['n_placements']} (cue={res['cue_pos']})")
    _ok('cue position', res['cue_pos'] == 15, str(res['cue_pos']))

    csa, hca = res['per_layer'][0], res['per_layer'][1]
    _ok('csa block_hit fraction = 9/14',
        abs(csa['block_hit_fraction'] - 9 / 14) < 1e-12,
        f"{csa['block_hit_fraction']!r}")
    _ok('csa window_hit fraction = 2/14',
        abs(csa['window_hit_fraction'] - 2 / 14) < 1e-12,
        f"{csa['window_hit_fraction']!r}")
    _ok('hca block_hit fraction = 7/14',
        abs(hca['block_hit_fraction'] - 7 / 14) < 1e-12,
        f"{hca['block_hit_fraction']!r}")
    _ok('union exact fraction = 12/14',
        abs(res['any_exact_fraction'] - 12 / 14) < 1e-12,
        f"{res['any_exact_fraction']!r}")
    _ok('mean exact paths = 20/14',
        abs(res['n_exact_paths_mean'] - 20 / 14) < 1e-12,
        f"{res['n_exact_paths_mean']!r}")

    by_p = {r['key_pos']: r for r in res['rows']}
    _ok('p=7 uncovered', not by_p[7]['any_exact'], str(by_p[7]['any_exact']))
    _ok('p=11 uncovered (causal edge)', not by_p[11]['any_exact'],
        str(by_p[11]['any_exact']))
    _ok('p=3 covered by HCA only',
        by_p[3]['any_exact'] and not by_p[3]['layers'][0]['exact']
        and by_p[3]['layers'][1]['block_hit'], 'hca fallback')
    _ok('p=12 covered by window only',
        by_p[12]['layers'][0]['window_hit']
        and not by_p[12]['layers'][0]['block_hit'], 'window edge')
    _ok('window edge d=win is out, d=win-1 is in',
        not by_p[11]['layers'][0]['window_hit']
        and by_p[12]['layers'][0]['window_hit'], 'p=11 vs p=12')

    # CSA chance: a = 15//4 = 3 visible blocks, k=2 -> 2/3; the expected
    # chance over placements is (9/14) * (2/3) = 3/7.
    _ok('csa chance closed form',
        abs(csa['selection_chance'] - 3 / 7) < 1e-12,
        f"{csa['selection_chance']!r}")
    _ok('csa visible blocks at cue', csa['visible_blocks_at_cue'] == 3,
        str(csa['visible_blocks_at_cue']))
    _ok('union selection chance (single CSA layer)',
        abs(res['csa_union_selection_chance_indep'] - 3 / 7) < 1e-12,
        f"{res['csa_union_selection_chance_indep']!r}")

    # Explicit placements include the value-at-cue edge (p = cue-1).
    edge = R.structural_recall(plan, seq_len=16, key_positions=[14])
    _ok('explicit key position at cue-1 accepted',
        edge['n_placements'] == 1
        and edge['rows'][0]['key_pos'] == 14, 'p=14')


def test_structural_validation_and_distance():
    plan = _plan()
    for kwargs in ({'key_positions': [-1]}, {'key_positions': [15]},
                   {'key_positions': []}):
        try:
            R.structural_recall(plan, seq_len=16, **kwargs)
            raised = False
        except ValueError:
            raised = True
        _ok(f'invalid key_positions rejected {kwargs}', raised, '')
    for seq_len in (2, 0, -1, 'x'):
        try:
            R.structural_recall(plan, seq_len=seq_len)
            raised = False
        except (ValueError, TypeError):
            raised = True
        _ok(f'invalid seq_len rejected {seq_len!r}', raised, '')
    try:
        R.structural_recall([], seq_len=16)
        raised = False
    except ValueError:
        raised = True
    _ok('empty plan rejected', raised, '')
    try:
        R.LayerSpec(0, 'softmax', block=4, window=4, topk=2)
        raised = False
    except ValueError:
        raised = True
    _ok('unknown layer kind rejected', raised, '')
    try:
        R.LayerSpec(0, 'csa', block=0, window=4, topk=2)
        raised = False
    except ValueError:
        raised = True
    _ok('zero block rejected', raised, '')

    res = R.structural_recall(plan, seq_len=16, key_positions=[13])
    _ok('distance field = cue - key_pos',
        res['rows'][0]['distance'] == 2, str(res['rows'][0]['distance']))
    _ok('by_distance keyed by distance',
        '2' in res['by_distance'], str(list(res['by_distance'])))


def test_plan_from_config_and_model():
    layout = ('csa', 'kda', 'hca', 'csa')
    cfg = R.make_grid_cfg(3, 6, layout)
    plan = R.plan_from_config(cfg, cfg.layout)
    _ok('plan length = layout length', len(plan) == 4, str(len(plan)))
    _ok('kinds resolved',
        [s.kind for s in plan] == list(layout), str([s.kind for s in plan]))
    _ok('geometry from config',
        plan[0].block == 3 and plan[2].block == 6
        and plan[0].window == cfg.csa_sliding_window
        and plan[2].window == cfg.hca_sliding_window
        and plan[0].topk == cfg.csa_topk and plan[2].topk is None,
        f'{plan[0]} / {plan[2]}')
    model = HybridKCHAttention(cfg, total_layers=len(layout))
    plan2 = R.plan_from_model(model)
    _ok('plan_from_model == plan_from_config', plan2 == plan,
        str(plan2))
    try:
        R.plan_from_config(cfg, ('memory', 'csa'))
        raised = False
    except ValueError:
        raised = True
    _ok('unresolved memory token rejected', raised, '')
    try:
        R.plan_from_config('not-a-config', ('csa',))
        raised = False
    except TypeError:
        raised = True
    _ok('non-config rejected', raised, '')


def _two_csa_model(topk, seed=0):
    layout = ('csa', 'csa', 'hca', 'kda', 'kda')
    cfg = R.make_grid_cfg(4, 8, layout, csa_topk=topk)
    torch.manual_seed(seed)
    model = HybridKCHAttention(cfg, total_layers=len(cfg.layout)).eval()
    embed = nn.Embedding(R.VOCAB, R.D_MODEL)
    gen = torch.Generator().manual_seed(3)
    x_emb, _t, cue, meta = make_long_range_batch(
        64, 32, 1, R.VOCAB, embed, 'cpu', generator=gen, return_meta=True)
    return model, embed, x_emb, cue, meta


def test_indexer_recall_dense_all_hit_and_alignment():
    """Select-all top-k: every resolved gold block is hit, and the
    recomputed selection is bit-identical to the operator's own indices
    for BOTH CSA layers (the anti-drift pin)."""
    model, _embed, x_emb, cue, meta = _two_csa_model(topk=100)
    out = R.per_layer_indexer_recall(
        model, x_emb, meta['key_pos'], cue_pos=cue, capture_operator=True)
    _ok('two CSA layers discovered', len(out['layers']) == 2,
        str([l['layer_index'] for l in out['layers']]))
    _ok('operator alignment pin', out['operator_alignment'] is True, '')
    _ok('per-layer alignment',
        all(l['operator_alignment'] for l in out['layers']), '')
    for l in out['layers']:
        _ok(f"dense recall layer {l['layer_index']} == 1.0",
            l['recall'] == 1.0, f"{l['recall']!r}")
        _ok(f"dense chance layer {l['layer_index']} == 1.0",
            l['chance'] == 1.0, f"{l['chance']!r}")
    _ok('union recall == 1.0', out['union_recall'] == 1.0,
        f"{out['union_recall']!r}")
    _ok('first-hit histogram attributes layer 0',
        out['first_hit_layer_hist'] == {'0': out['union_n_resolved']},
        str(out['first_hit_layer_hist']))
    _ok('skipped_reason empty', out['skipped_reason'] is None, '')


def test_indexer_recall_sparse_chance_and_union():
    model, _embed, x_emb, cue, meta = _two_csa_model(topk=2)
    out = R.per_layer_indexer_recall(
        model, x_emb, meta['key_pos'], cue_pos=cue, capture_operator=True)
    _ok('sparse alignment pin', out['operator_alignment'] is True, '')
    for l in out['layers']:
        _ok(f"sparse recall layer {l['layer_index']} in [0,1]",
            l['recall'] is not None and 0.0 <= l['recall'] <= 1.0,
            f"{l['recall']!r}")
        expected = min(l['topk'], l['visible_blocks_at_cue']) / \
            l['visible_blocks_at_cue']
        _ok(f"sparse chance layer {l['layer_index']} == min(k,a)/a",
            l['chance'] == expected, f"{l['chance']!r} vs {expected!r}")
    _ok('union recall is well-formed',
        out['union_recall'] is not None
        and 0.0 <= out['union_recall'] <= 1.0, f"{out['union_recall']!r}")
    _ok('union resolved >= per-layer resolved',
        out['union_n_resolved'] >= out['layers'][0]['n_resolved'], '')
    # The per-layer chance must match the analytic block_hit coverage times
    # the closed-form selection probability: a = 31//4 = 7, k = 2.
    cfg = model.cfg
    summary = R.structural_summary(R.plan_from_model(model), seq_len=32)
    csa_entries = [e for e in summary['per_layer'] if e['kind'] == 'csa']
    _ok('analytic visible blocks at cue = 7',
        all(e['visible_blocks_at_cue'] == 31 // cfg.csa_m
            for e in csa_entries),
        str([e['visible_blocks_at_cue'] for e in csa_entries]))


def test_indexer_recall_error_paths_and_no_csa():
    layout = ('kda', 'hca', 'kda')
    cfg = R.make_grid_cfg(4, 8, layout)
    torch.manual_seed(0)
    model = HybridKCHAttention(cfg, total_layers=len(cfg.layout)).eval()
    x = torch.randn(2, 8, cfg.d_model) * 0.1
    out = R.per_layer_indexer_recall(model, x, [1, 1], cue_pos=7)
    _ok('no-CSA layout reports skipped_reason',
        out['layers'] == [] and out['skipped_reason'] is not None
        and out['union_recall'] is None, str(out['skipped_reason']))

    model2, _embed, x_emb, cue, meta = _two_csa_model(topk=4)
    for kwargs in (
        {'key_pos': [0, 0], 'cue_pos': 99},
        {'key_pos': [0, 0], 'cue_pos': -1},
        {'key_pos': [0], 'cue_pos': cue},
        {'key_pos': [-1, 0], 'cue_pos': cue},
        {'key_pos': [0, 99], 'cue_pos': cue},
    ):
        try:
            R.per_layer_indexer_recall(model2, x_emb, **kwargs)
            raised = False
        except ValueError:
            raised = True
        _ok(f'indexer-recall arg guard fires {kwargs}', raised, '')
    try:
        R.per_layer_indexer_recall(model2, x_emb[0], meta['key_pos'],
                                   cue_pos=cue)
        raised = False
    except ValueError:
        raised = True
    _ok('non-3D input rejected', raised, '')
    try:
        R.per_layer_indexer_recall(model2, x_emb[:1, :2], [0],
                                   cue_pos=1)
        raised = False
    except ValueError:
        raised = True
    _ok('T < 3 rejected', raised, '')


def test_hca_coverage_matches_structural():
    layout = ('hca', 'csa', 'hca', 'kda')
    cfg = R.make_grid_cfg(4, 8, layout)
    torch.manual_seed(0)
    model = HybridKCHAttention(cfg, total_layers=len(layout)).eval()
    seq_len = 32
    cue = seq_len - 1
    key_pos = torch.arange(0, cue - 1)
    cov = R.hca_block_coverage(model, key_pos, cue_pos=cue)
    structural = R.structural_summary(R.plan_from_model(model), seq_len)
    by_index = {e['layer_index']: e for e in structural['per_layer']}
    _ok('both HCA layers reported', len(cov['layers']) == 2,
        str([l['layer_index'] for l in cov['layers']]))
    for l in cov['layers']:
        _ok(f"hca layer {l['layer_index']} resolved == analytic block_hit",
            abs(l['resolved_fraction']
                - by_index[l['layer_index']]['block_hit_fraction']) < 1e-12,
            f"{l['resolved_fraction']!r} vs "
            f"{by_index[l['layer_index']]['block_hit_fraction']!r}")


def test_report_strict_json_and_budget():
    report = R.run_recall_report(
        pairs=((4, 8),), layouts=['alternating'], steps=2, seeds=(42,),
        seq_len=16, train_batch=8, eval_batches=1, eval_batch=8)
    text = json.dumps(report, allow_nan=False)
    _ok('report is strict JSON', isinstance(text, str) and len(text) > 0, '')
    _ok('one row', len(report['rows']) == 1, str(len(report['rows'])))
    row = report['rows'][0]
    _ok('row carries one seed record', len(row['per_seed']) == 1, '')
    seed0 = row['per_seed'][0]
    _ok('per-layer recall list length = n_csa',
        len(seed0['indexer_recall_per_layer']) == row['n_csa'],
        f"{len(seed0['indexer_recall_per_layer'])} vs {row['n_csa']}")
    _ok('accuracy in [0,1]',
        seed0['accuracy'] is not None and 0.0 <= seed0['accuracy'] <= 1.0,
        f"{seed0['accuracy']!r}")
    _ok('structural summary attached',
        row['structural']['any_exact_fraction'] > 0.0
        and row['structural']['n_placements'] == 14,
        str(row['structural']['n_placements']))
    _ok('cost = n_csa/m + n_hca/m2',
        abs(row['compressed_entries_per_token']
            - (row['n_csa'] / 4 + row['n_hca'] / 8)) < 1e-12,
        f"{row['compressed_entries_per_token']!r}")

    structural_only = R.run_recall_report(
        pairs=R.BUDGET_PAIRS, layouts=list(R.LAYOUT_PRESETS)[:2],
        train=False)
    _ok('no-train has one row per cell',
        len(structural_only['rows']) == len(R.BUDGET_PAIRS) * 2,
        str(len(structural_only['rows'])))
    by_cost = {}
    for r in structural_only['rows']:
        by_cost.setdefault(round(r['compressed_entries_per_token'], 12),
                           set()).add((r['m'], r['m2']))
    _ok('budget family 2/m+2/m2 shared by (4,8) and (8,4)',
        by_cost.get(0.75) == {(4, 8), (8, 4)}, str(by_cost.get(0.75)))
    _ok('budget family 1.0 shared by (4,4),(3,6),(6,3)',
        by_cost.get(1.0) == {(4, 4), (3, 6), (6, 3)},
        str(by_cost.get(1.0)))
    _ok('all no-train rows have empty per_seed',
        all(r['per_seed'] == [] for r in structural_only['rows']), '')
    # Every layout preset must keep the same operator counts (6 memory,
    # 2 CSA, 2 HCA), so interleaving changes position only.
    layouts = list(R.LAYOUT_PRESETS.values())
    _ok('all presets share operator counts',
        all((sum(1 for k in l if k == 'csa'),
             sum(1 for k in l if k == 'hca'),
             sum(1 for k in l if k in R.MEMORY_KINDS)) == (2, 2, 6)
            for l in layouts), 'fixed-depth position ablation')


def test_seed_statistics_contract():
    """Mean/std/CI math + the >= 3-seed default policy (regression pin)."""
    import inspect

    mean, std, ci = R._mean_std_ci95([])
    _ok('empty values -> all None', (mean, std, ci) == (None, None, None), '')
    mean, std, ci = R._mean_std_ci95([0.5])
    _ok('single value -> mean, undefined std/CI',
        (mean, std, ci) == (0.5, None, None), f'{mean}/{std}/{ci}')
    mean, std, ci = R._mean_std_ci95([1.0, 2.0, 3.0])
    t_crit = R._t_crit_975(3)
    _ok('sample mean', mean == 2.0, str(mean))
    _ok('sample std (ddof=1)', abs(std - 1.0) < 1e-12, str(std))
    _ok('95% CI half-width = se * t_{n-1}',
        abs(ci - (1.0 / (3 ** 0.5)) * t_crit) < 1e-12, str(ci))

    default = inspect.signature(R.run_recall_report).parameters['seeds'].default
    _ok('run_recall_report defaults to >= 3 seeds',
        tuple(default) == (42, 43, 44), str(default))


def test_report_multi_seed_statistics():
    """A 2-seed report carries mean/std and flags the short seed budget."""
    report = R.run_recall_report(
        pairs=((4, 8),), layouts=['alternating'], steps=1, seeds=(42, 43),
        seq_len=16, train_batch=8, eval_batches=1, eval_batch=8)
    json.dumps(report, allow_nan=False)
    row = report['rows'][0]
    _ok('two per-seed records', len(row['per_seed']) == 2, '')
    _ok('accuracy mean/std/n populated',
        row['accuracy_mean'] is not None and row['accuracy_std'] is not None
        and row['accuracy_std'] >= 0.0 and row['accuracy_n_seeds'] == 2,
        f"mean={row['accuracy_mean']} std={row['accuracy_std']}")
    if row['indexer_union_recall_mean'] is not None:
        _ok('indexer union recall std populated',
            row['indexer_union_recall_std'] is not None,
            f"std={row['indexer_union_recall_std']}")
    _ok('HCA resolved fraction aggregated',
        row['hca_resolved_fraction_mean'] is not None, '')
    design = report['design']
    stats = design['statistical_design']
    _ok('shared seed list recorded', stats['shared_seed_list'] is True, '')
    _ok('per-cell seed count recorded', stats['seeds_per_cell'] == 2, '')
    _ok('short seed budget flagged exploratory',
        stats['warning'] is not None and 'exploratory' in stats['warning'],
        str(stats['warning']))

    single = R.run_recall_report(
        pairs=((4, 8),), layouts=['alternating'], steps=1, seeds=(42,),
        seq_len=16, train_batch=8, eval_batches=1, eval_batch=8)
    row1 = single['rows'][0]
    _ok('single-seed row has mean but no std/CI',
        row1['accuracy_mean'] is not None and row1['accuracy_std'] is None
        and row1['accuracy_ci95'] is None, '')
    _ok('single-seed design warns',
        single['design']['statistical_design']['warning'] is not None, '')


def test_hypothesis_checks_contract():
    """Pre-registered hypotheses + analytic/empirical checks are reported."""
    structural = R.run_recall_report(train=False)
    json.dumps(structural, allow_nan=False)
    design = structural['design']
    _ok('design carries the hypothesis table',
        set(design.get('hypotheses', {})) == set(R.HYPOTHESES),
        f"keys={sorted(design.get('hypotheses', {}))}")
    checks = design.get('hypothesis_checks', {})
    _ok('checks cover every hypothesis',
        set(checks) == set(R.HYPOTHESES), f'keys={sorted(checks)}')
    win = checks['window_reach_matches_geometry']
    _ok('window geometry verdict True with zero deviations',
        win['verdict'] is True and win['n_deviations'] == 0, str(win))
    far = checks['far_placements_covered_by_blocks']
    _ok('far placements covered by the block path',
        far['verdict'] is True and far['min_far_exact_fraction'] > 0.0,
        str(far))
    mono = checks['block_coverage_not_monotone_in_block_size']
    _ok('monotonicity comparison actually ran over >0 pairs',
        mono['n_comparable_pairs'] > 0 and mono['verdict'] is not None,
        str({k: v for k, v in mono.items() if k != 'violation_examples'}))
    _ok('violation examples are capped but consistent',
        len(mono['violation_examples'])
        == min(mono['n_monotonicity_violations'], 8), str(mono))

    trained = R.run_recall_report(
        pairs=((4, 8),), layouts=['alternating'], steps=1, seeds=(42,),
        seq_len=16, train_batch=8, eval_batches=1, eval_batch=8)
    json.dumps(trained, allow_nan=False)
    row = trained['rows'][0]
    check = row.get('hypothesis_checks', {}).get('indexer_union_above_chance')
    _ok('trained row carries the indexer-vs-chance check',
        check is not None and 'uniform_union_chance' in check
        and check['verdict'] in (True, False, None), str(check))
    _ok('design-level empirical check aggregated',
        'verdict' in trained['design']['hypothesis_checks']
        ['indexer_union_above_chance'])


def main() -> int:
    test_structural_recall_hand_derived()
    test_structural_validation_and_distance()
    test_plan_from_config_and_model()
    test_indexer_recall_dense_all_hit_and_alignment()
    test_indexer_recall_sparse_chance_and_union()
    test_indexer_recall_error_paths_and_no_csa()
    test_hca_coverage_matches_structural()
    test_report_strict_json_and_budget()
    test_seed_statistics_contract()
    test_report_multi_seed_statistics()
    test_hypothesis_checks_contract()
    print('\nAll recall-reach tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
