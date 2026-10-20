"""Regression tests for the greedy layerwise architecture search (Exp 12).

What is pinned here
-------------------
``layerwise_search`` turns layerwise layout choice into a search over
``HybridConfig.layout`` tuples under the repository's exact decode-state
bytes (Exp 3 / ``cache_accounting``) and KV read budget (Exp 9b /
``kv_budget``). These tests lock the pieces that must never drift:

* candidate validation and the canonical (deterministic) single-edit
  neighborhood — substitution / adjacent swap / top-k move, in a fixed
  order with no duplicates;
* the cost model — state bytes match an independent walk over
  ``analytic_cache_components`` (PARAMETER-pool sinks excluded), a
  memory-only layout is exactly constant in ``T`` while any CSA/HCA
  layout grows, and the ``kv_budget`` geometry agrees with the committed
  Exp 9b builder;
* read-budget monotonicity in ``csa_topk`` (the top-k selection knob);
* the search loop — greedy best-improvement on injected landscapes
  (target recovery, budget exclusion, error records, determinism,
  Pareto-frontier correctness), including the uncertainty-aware
  acceptance rule (a mean gain that is not seed-paired supported is
  recorded but not accepted);
* the probe statistics — ``make_mqar_objective`` reports mean / sample
  std / SE / 95% CI over >= 3 seeds by default, and the paired support
  rule matches the closed-form t threshold and the zero-variance
  sign-flip bound;
* integration — searched candidates build through
  ``recall_reach.make_grid_cfg`` / ``HybridKCHAttention`` and run a
  forward; the MQAR objective wiring produces a finite score.

Run via pytest::

    pytest -q test_layerwise_search.py

Or directly::

    python3 test_layerwise_search.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402

from cache_accounting import (  # noqa: E402
    COMPONENT_SPECS,
    PARAMETER,
    analytic_cache_components,
)
from kv_budget import instant_layout_read_elements  # noqa: E402
from layerwise_search import (  # noqa: E402
    Candidate,
    DEFAULT_TOPK_GRID,
    SearchProblem,
    budget_spec,
    candidate_config,
    candidate_from_ratio,
    cost_report,
    greedy_search,
    layout_neighbors,
    make_mqar_objective,
    pareto_frontier,
    read_budget,
)

SMALL = dict(m=4, m2=8, d_model=32)
START = Candidate(('kda', 'kda', 'kda', 'csa', 'hca'), csa_topk=2)


@pytest.fixture
def device(request):
    """Standalone ``--device`` override (falls back to the pytest option)."""
    try:
        return request.config.getoption('--device')
    except (ValueError, KeyError):
        return 'cpu'


# ---------------------------------------------------------------------------
# candidates + neighborhood
# ---------------------------------------------------------------------------

def test_candidate_validation():
    with pytest.raises(ValueError):
        Candidate(())
    with pytest.raises(ValueError):
        Candidate(('kda', 'mla'))
    with pytest.raises(ValueError):
        Candidate(('kda',), csa_topk=-1)
    with pytest.raises(TypeError):
        Candidate(('kda',), csa_topk=True)
    c = Candidate(('csa', 'kda'), csa_topk=3)
    assert c.counts == {'kda': 1, 'csa': 1, 'hca': 0}
    assert c.label == 'csa,kda/topk=3'


def test_candidate_from_ratio_order_and_validation():
    c = candidate_from_ratio((2, 1, 1), csa_topk=4)
    assert c.layout == ('kda', 'kda', 'csa', 'hca')
    assert c.csa_topk == 4
    with pytest.raises(ValueError):
        candidate_from_ratio((2, 1))
    with pytest.raises(ValueError):
        candidate_from_ratio((-1, 1, 1))


def test_neighbors_are_exact_and_deterministic():
    neighbors = layout_neighbors(START)
    labels = [c.label for c in neighbors]
    assert len(labels) == len(set(labels)), 'neighbors must not duplicate'
    assert START not in neighbors
    # substitutions: 5 slots x 2 alternative kinds (hamming distance 1)
    substitutions = [c for c in neighbors
                     if sum(a != b for a, b in zip(c.layout, START.layout))
                     == 1]
    assert len(substitutions) == 10
    # adjacent swaps: slots (2,3) and (3,4) are the only unequal pairs
    swaps = [c for c in neighbors
             if sum(a != b for a, b in zip(c.layout, START.layout)) == 2
             and sorted(c.layout) == sorted(START.layout)]
    assert len(swaps) == 2
    # top-k moves: the grid minus the current value
    topk_moves = [c for c in neighbors if c.layout == START.layout]
    assert {c.csa_topk for c in topk_moves} == \
        set(DEFAULT_TOPK_GRID) - {START.csa_topk}
    assert labels == [c.label for c in layout_neighbors(START)], \
        'neighbor order must be reproducible'


def test_neighbors_skip_topk_without_csa():
    c = Candidate(('kda', 'kda', 'hca'))
    neighbors = layout_neighbors(c)
    assert all(x.layout != c.layout for x in neighbors)
    # 6 substitutions (3 slots x 2 alternative kinds) + 1 unequal swap
    # (slots 1,2); no top-k moves because the layout has no CSA layer.
    assert len(neighbors) == 7


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

def test_state_bytes_match_analytic_components():
    cfg = candidate_config(START, **SMALL)
    report = cost_report(START, T=512, **SMALL)
    comps = analytic_cache_components(cfg, 512)
    # Independent walk with the report's own rule: PARAMETER-pool
    # components (attention sinks) are shared parameters, not per-request
    # decode state.
    expected_total = sum(
        int(comp['elements_allocated']) * 4
        for name, comp in comps.items()
        if COMPONENT_SPECS[name][1] != PARAMETER)
    assert report['state_bytes_allocated'] == expected_total
    assert sum(report['state_bytes_breakdown'].values()) == expected_total
    assert report['parameter_bytes'] > 0, 'sink params are reported apart'
    by_op = report['state_bytes_by_op']
    assert by_op['kda'] > 0 and by_op['csa'] > 0 and by_op['hca'] > 0
    assert 'sink' not in by_op


def test_memory_only_layout_is_constant_in_T():
    c = Candidate(('kda', 'kda', 'kda'))
    low = cost_report(c, T=64, **SMALL)
    high = cost_report(c, T=8192, **SMALL)
    assert low['state_bytes_allocated'] == high['state_bytes_allocated']
    assert low['state_constant'] is True
    assert high['state_constant'] is True


def test_layouts_with_retrieval_grow_with_T():
    for cand in (START, Candidate(('kda', 'hca')),
                 Candidate(('csa', 'kda'))):
        low = cost_report(cand, T=64, **SMALL)
        high = cost_report(cand, T=512, **SMALL)
        assert high['state_bytes_allocated'] > low['state_bytes_allocated']
        assert high['state_constant'] is False
        assert low['state_constant'] is False


def test_budget_spec_agrees_with_exp9b_builder():
    from run_budget_frontier import _spec_from_hybrid_cfg

    cfg = candidate_config(START, **SMALL)
    mine = budget_spec(cfg)
    theirs = _spec_from_hybrid_cfg(
        cfg, softmax_geo={'n_kv_heads': cfg.n_heads_qk,
                          'd_k': cfg.head_dim_k, 'd_v': cfg.head_dim_v})
    for op in ('softmax', 'kda', 'csa', 'hca'):
        assert mine[op] == theirs[op], f'geometry drift for {op!r}'


def test_read_budget_matches_kv_budget_directly():
    cfg = candidate_config(START, **SMALL)
    report = read_budget(cfg, 256)
    counts = (cfg.layout.count('kda'), cfg.layout.count('csa'),
              cfg.layout.count('hca'))
    expected = instant_layout_read_elements(counts, 256, budget_spec(cfg))
    assert report['decode_marginal_read_elements'] == pytest.approx(
        expected['total_read_elements'], rel=1e-12)
    by_op = report['decode_marginal_by_op']
    assert by_op['kda'] + by_op['csa'] + by_op['hca'] == pytest.approx(
        expected['total_read_elements'], rel=1e-12)


def test_topk_increases_decode_reads():
    totals = []
    for topk in (0, 1, 2, 4):
        cand = Candidate(START.layout, csa_topk=topk)
        totals.append(cost_report(cand, T=2048, **SMALL)[
            'decode_marginal_read_elements'])
    assert totals == sorted(totals) and totals[0] < totals[-1], \
        f'top-k must buy selected-block reads: {totals}'


# ---------------------------------------------------------------------------
# search loop
# ---------------------------------------------------------------------------

def _fake_problem(score_fn, *, max_state_bytes=None, cost_fn=None):
    def evaluate(cand):
        return {'score': score_fn(cand), 'metrics': {}}

    def cost(cand):
        return cost_report(cand, T=4096, **SMALL)

    if cost_fn is not None:
        cost = cost_fn
    return SearchProblem(evaluate=evaluate, cost=cost,
                         max_state_bytes=max_state_bytes)


def test_greedy_search_recovers_target():
    target = Candidate(('kda', 'hca', 'csa', 'kda', 'hca'), csa_topk=2)

    def score(cand):
        return -sum(int(a != b) for a, b in zip(cand.layout, target.layout))

    result = greedy_search(_fake_problem(score), START, max_rounds=5)
    assert result['greedy_endpoint']['label'] == target.label
    assert result['best']['layout'] == list(target.layout)
    assert result['best']['score'] == 0.0
    # The accepted path is a chain of single edits: an adjacent swap fixes
    # two slots at once (hamming 3 -> 1), then one substitution reaches the
    # target (0). The final round is a rejected no-improvement round.
    accepted = [r for r in result['rounds'] if r['accepted']]
    assert len(accepted) == 2
    assert result['rounds'][-1]['accepted'] is False


def test_greedy_search_respects_state_budget():
    global_best = Candidate(('csa', 'csa', 'csa', 'csa', 'csa'), csa_topk=2)

    def score(cand):
        return float(cand.layout.count('csa'))

    start = Candidate(('kda', 'kda', 'kda', 'csa', 'hca'), csa_topk=2)
    start_bytes = cost_report(start, T=4096, **SMALL)['state_bytes_allocated']
    # Budget for exactly one extra CSA layer (one substitution move).
    one_csa = Candidate(('csa', 'kda', 'kda', 'csa', 'hca'), csa_topk=2)
    budget = cost_report(one_csa, T=4096, **SMALL)['state_bytes_allocated']
    assert start_bytes < budget
    result = greedy_search(
        _fake_problem(score, max_state_bytes=budget), start, max_rounds=3)
    best = result['best']
    assert best['layout'].count('csa') == 2, \
        f'budget must stop the all-CSA attractor: {best["label"]}'
    assert best['state_bytes_allocated'] <= budget
    assert global_best.label != best['label']
    # Every accepted move is feasible; infeasible records are still kept.
    assert any(not r['feasible'] for r in result['records'])
    assert all(r['feasible'] for r in result['records']
               if r['label'] == best['label'])


def test_search_records_errors_and_skips_them():
    def evaluate(cand):
        if 'csa' in cand.layout:
            raise RuntimeError('boom')
        return {'score': 1.0, 'metrics': {}}

    problem = SearchProblem(
        evaluate=evaluate,
        cost=lambda cand: cost_report(cand, T=256, **SMALL))
    result = greedy_search(problem, START, max_rounds=2)
    errored = [r for r in result['records'] if r['error'] is not None]
    assert errored, 'errors must be recorded'
    assert all(r['score'] is None for r in errored)
    assert result['best']['score'] is not None
    assert 'csa' not in result['best']['layout'] or \
        result['best']['label'] == START.label


def test_greedy_search_is_deterministic():
    def score(cand):
        return 3.0 - 0.25 * sum(int(a != b)
                                for a, b in zip(cand.layout, START.layout))

    r1 = greedy_search(_fake_problem(score), START, max_rounds=3)
    r2 = greedy_search(_fake_problem(score), START, max_rounds=3)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_plateau_closure_prefers_cheaper_tie():
    def synthetic_cost(cand):
        return {
            'state_bytes_allocated': 100 + 10 * cand.layout.count('hca'),
            'state_bytes_valid': 0,
            'decode_marginal_read_elements': float(cand.layout.count('csa')),
        }

    def flat_score(cand):
        return 1.0

    result = greedy_search(
        _fake_problem(flat_score, cost_fn=synthetic_cost), START,
        max_rounds=1)
    endpoint = result['greedy_endpoint']
    best = result['best']
    assert endpoint['state_bytes_allocated'] == 110  # start keeps its HCA
    # The strict-improvement rule refuses to move on an equal score, but the
    # reported best closes the plateau to the cheapest tied candidate.
    assert best['score'] == endpoint['score'] == 1.0
    assert best['state_bytes_allocated'] == 100
    assert best['layout'].count('hca') == 0


def test_pareto_frontier_hand_case():
    records = [
        {'label': 'a', 'score': 1.0, 'state_bytes_allocated': 10,
         'decode_marginal_read_elements': 1.0},
        {'label': 'b', 'score': 2.0, 'state_bytes_allocated': 20,
         'decode_marginal_read_elements': 1.0},
        {'label': 'c', 'score': 1.5, 'state_bytes_allocated': 30,
         'decode_marginal_read_elements': 1.0},   # dominated by b
        {'label': 'd', 'score': 3.0, 'state_bytes_allocated': 40,
         'decode_marginal_read_elements': 0.5},
        {'label': 'e', 'score': None, 'state_bytes_allocated': 5,
         'decode_marginal_read_elements': 0.1},   # no score -> excluded
    ]
    frontier = pareto_frontier(records)
    assert [r['label'] for r in frontier] == ['a', 'b', 'd']
    # Deterministic ordering by (cost tuple, -score, label).
    assert frontier[0]['label'] == 'a'


def test_search_result_is_json_safe():
    def score(cand):
        return float(cand.layout.count('hca'))

    result = greedy_search(_fake_problem(score), START, max_rounds=1)
    json.dumps(result, allow_nan=False)  # must not raise
    assert result['pareto'], 'frontier must be non-empty'
    assert result['best_vs_start'] is None, \
        'no per-seed metrics -> paired test must be reported unavailable'
    assert result['acceptance_policy']['require_supported'] is True


def test_paired_support_rule_matches_closed_forms():
    import layerwise_search as L

    stats, supported = L._paired_support([0.05, 0.10, 0.15])
    assert stats['n_pairs'] == 3 and stats['t_stat'] == pytest.approx(
        3.4641, rel=1e-3)
    assert supported, 't=3.46 must clear the one-sided 95% crit at df=2'
    _, not_supported = L._paired_support([0.10, -0.05, 0.02])
    assert not not_supported, 'mean gain within noise must not be supported'
    _, zero_var3 = L._paired_support([0.05] * 3)
    assert not zero_var3, '2^{-2}=0.25 > 0.05: n=3 cannot clear the bound'
    _, zero_var6 = L._paired_support([0.05] * 6)
    assert zero_var6, '2^{-5}=0.03125 <= 0.05'
    _, single = L._paired_support([0.30])
    assert not single, 'a single seed cannot support a paired claim'


def test_greedy_search_requires_seed_reproducibility():
    """A mean gain must be seed-paired supported to be ACCEPTED."""
    target_layout = ('csa', 'kda', 'kda', 'csa', 'hca')
    target_label = Candidate(target_layout).label

    def problem_with(per_seed_for_target):
        def evaluate(cand):
            per_seed = (list(per_seed_for_target)
                        if tuple(cand.layout) == target_layout
                        else [0.0, 0.0, 0.0])
            return {'score': sum(per_seed) / len(per_seed),
                    'metrics': {'seeds': [42, 43, 44],
                                'per_seed_accuracy': per_seed}}

        return SearchProblem(
            evaluate=evaluate,
            cost=lambda cand: {'state_bytes_allocated': 100,
                               'decode_marginal_read_elements': 1.0})

    # Reproducible gain: per-seed diffs [0.05, 0.10, 0.15] -> t=3.46.
    result = greedy_search(problem_with([0.05, 0.10, 0.15]), START,
                           max_rounds=1)
    round0 = result['rounds'][0]
    assert round0['improvement'] is True
    assert round0['improvement_supported'] is True
    assert round0['accepted'] is True
    assert round0['paired']['n_pairs'] == 3
    assert result['greedy_endpoint']['label'] == target_label
    assert result['best_vs_start']['significant'] is True
    json.dumps(result, allow_nan=False)

    # Same mean gain with zero spread across only 3 seeds: the exact
    # sign-flip bound cannot clear alpha, so the move is recorded but
    # rejected (single-seed-style noise promotion is not accepted).
    result = greedy_search(problem_with([0.10, 0.10, 0.10]), START,
                           max_rounds=1)
    round0 = result['rounds'][0]
    assert round0['improvement'] is True
    assert round0['improvement_supported'] is False
    assert round0['accepted'] is False
    assert 'not reproducible' in round0['reason']
    assert result['greedy_endpoint']['label'] == START.label
    assert result['best']['label'] == START.label, \
        'an unsupported noise-sized gain must not promote a candidate'
    assert result['best_vs_start']['significant'] is False


def test_selection_correction_penalizes_winner_curse():
    """The argmax of many candidates is not a pre-registered comparison."""
    target_layout = ('csa', 'kda', 'kda', 'csa', 'hca')

    def evaluate(cand):
        per_seed = ([0.05, 0.10, 0.15]
                    if tuple(cand.layout) == target_layout
                    else [0.0, 0.0, 0.0])
        return {'score': sum(per_seed) / len(per_seed),
                'metrics': {'seeds': [42, 43, 44],
                            'per_seed_accuracy': per_seed}}

    problem = SearchProblem(
        evaluate=evaluate,
        cost=lambda cand: {'state_bytes_allocated': 100,
                           'decode_marginal_read_elements': 1.0})
    uncorrected = greedy_search(problem, START, max_rounds=1)
    corrected = greedy_search(problem, START, max_rounds=1,
                              selection_correction=True)
    assert uncorrected['rounds'][0]['accepted'] is True
    round0 = corrected['rounds'][0]
    assert round0['improvement'] is True
    assert round0['selection_family_size'] > 1
    assert round0['alpha_used'] == pytest.approx(
        0.05 / round0['selection_family_size'])
    assert round0['paired']['alpha_used'] == pytest.approx(
        round0['alpha_used'])
    assert round0['accepted'] is False
    assert corrected['best']['label'] == START.label
    assert corrected['acceptance_policy']['selection_correction'] is True
    assert corrected['best_vs_start']['selection_correction'] is True
    json.dumps(corrected, allow_nan=False)


def test_plateau_does_not_promote_unsupported_cheaper_gain():
    """A cheaper neighbour with an unsupported score bump is not a tie."""
    target_layout = ('csa', 'kda', 'kda', 'csa', 'hca')
    target_label = Candidate(target_layout).label

    def evaluate(cand):
        per_seed = ([0.10, 0.10, 0.10]
                    if tuple(cand.layout) == target_layout
                    else [0.0, 0.0, 0.0])
        return {'score': sum(per_seed) / len(per_seed),
                'metrics': {'seeds': [42, 43, 44],
                            'per_seed_accuracy': per_seed}}

    def cost(cand):
        cheaper = tuple(cand.layout) == target_layout
        return {
            'state_bytes_allocated': 100 if cheaper else 200,
            'decode_marginal_read_elements': 1.0,
        }

    result = greedy_search(
        SearchProblem(evaluate=evaluate, cost=cost), START, max_rounds=1)
    round0 = result['rounds'][0]
    assert round0['best_neighbor'] == target_label
    assert round0['improvement'] is True
    assert round0['improvement_supported'] is False
    assert round0['accepted'] is False
    assert result['greedy_endpoint']['label'] == START.label
    assert result['best']['label'] == START.label, \
        'a cheaper but statistically unsupported gain must not be promoted'
    assert result['best_vs_start']['significant'] is False


def test_mqar_objective_seed_statistics():
    import inspect
    import layerwise_search as L

    default = inspect.signature(
        L.make_mqar_objective).parameters['seeds'].default
    assert tuple(default) == (42, 43, 44), \
        f'probe must default to >= 3 seeds, got {default}'

    objective = make_mqar_objective(
        m=4, m2=8, seeds=(0, 1), steps=1, seq_len=8, d_model=32,
        train_batch=4, eval_batches=1, eval_batch=8, device='cpu')
    result = objective(Candidate(('kda', 'kda', 'kda', 'csa', 'hca')))
    metrics = result['metrics']
    assert metrics['n_seeds'] == 2
    assert metrics['accuracy_std'] is not None and metrics['accuracy_std'] >= 0
    assert metrics['accuracy_se'] is not None and metrics['accuracy_se'] >= 0
    assert metrics['accuracy_ci95'] is not None
    assert len(metrics['per_seed_accuracy']) == 2
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

def test_candidates_build_and_forward(device):
    from ops_fused import HybridKCHAttention

    torch.manual_seed(0)
    for layout in (('kda', 'csa', 'hca'), ('kda', 'kda', 'csa', 'hca'),
                   ('csa', 'kda', 'hca')):
        cand = Candidate(layout, csa_topk=2)
        cfg = candidate_config(cand, **SMALL)
        model = HybridKCHAttention(cfg, total_layers=len(cfg.layout))
        model = model.to(device)
        model.eval()
        x = torch.randn(2, 8, cfg.d_model, device=device)
        with torch.no_grad():
            model.reset_state()
            out = model(x)
        assert out.shape == (2, 8, cfg.d_model)
        assert torch.isfinite(out).all(), f'non-finite output for {layout}'


def test_mqar_objective_smoke(device):
    objective = make_mqar_objective(
        m=4, m2=8, seeds=(0,), steps=1, seq_len=8, d_model=32,
        train_batch=4, eval_batches=1, eval_batch=8, device=device)
    result = objective(Candidate(('kda', 'kda', 'kda', 'csa', 'hca')))
    assert result['score'] is not None
    assert 0.0 <= result['score'] <= 1.0
    assert result['metrics']['per_seed_accuracy'][0] is not None
    assert result['metrics']['probe'] == 'mqar-long-range-cue-last'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
