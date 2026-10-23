"""Tests for the exact-test resolution and nonparametric CI layer.

Covers the small-n discreteness problem that ordinary power calculations
miss: an exact sign-flip test has only ``2**n`` sign assignments, so a
Holm-corrected family can have no rejection region at all. Also pins the
Clopper-Pearson win-rate interval, the order-statistic median interval,
and their integration into ``summarize_paired`` / ``run_quality`` /
``experiment_design``.

How to run::

    pytest -q test_inference_resolution.py
"""
from __future__ import annotations

import json
import math
import random

import pytest

import experiment_design as D
import multi_seed_stats as M
from run_quality import _pairwise_op_comparisons


# ---------------------------------------------------------------------------
# Exact-test floor arithmetic
# ---------------------------------------------------------------------------

def test_floor_arithmetic_for_small_exact_families():
    # 5 pairs, one two-sided comparison: p = 2/32 = 0.0625; still no
    # rejection at 0.05, and one more pair would fix it.
    r = M.exact_test_resolution(5)
    assert r['min_attainable_p_raw'] == 0.0625
    assert r['raw_testable'] is False
    assert r['family_testable'] is False
    assert r['min_effective_pairs_raw'] == 6
    assert r['shortfall_pairs_raw'] == 1
    assert r['max_comparisons_at_alpha'] == 0

    # A directional test halves the floor: 1/32 = 0.03125 <= 0.05.
    one = M.exact_test_resolution(5, alternative='greater')
    assert one['min_attainable_p_raw'] == 0.03125
    assert one['raw_testable'] is True
    assert one['min_effective_pairs_raw'] == 5

    # Six pairs with a 21-comparison family: raw testable, Holm is not.
    fam = M.exact_test_resolution(6, family_size=21)
    assert fam['min_attainable_p_raw'] == 0.03125
    assert fam['min_attainable_p_holm'] == pytest.approx(0.65625)
    assert fam['raw_testable'] is True
    assert fam['family_testable'] is False
    # The raw floor is below alpha, so a later-rank rejection is not ruled
    # out by the p floor alone; it could only happen if other, more
    # powerful comparisons were rejected first. With equal n it cannot
    # happen, which is why the family-level condition uses the first rank.
    assert fam['later_rank_possible'] is True
    assert fam['min_effective_pairs_raw'] == 6
    assert fam['min_effective_pairs_holm'] == 10
    assert fam['shortfall_pairs_holm'] == 4

    # Ten pairs clear the same 21-comparison family floor.
    ok = M.exact_test_resolution(10, family_size=21)
    assert ok['min_attainable_p_holm'] == pytest.approx(21 * 2 / 1024)
    assert ok['family_testable'] is True
    assert ok['shortfall_pairs_holm'] == 0

    # The largest family this n can support follows alpha / p_floor.
    assert M.exact_test_resolution(7)['max_comparisons_at_alpha'] == 3
    assert M.exact_test_resolution(8)['max_comparisons_at_alpha'] == 6


def test_ties_reduce_effective_pairs_and_raise_the_floor():
    r = M.exact_test_resolution(5, n_zero=1)
    assert r['n_effective'] == 4
    assert r['min_attainable_p_raw'] == 0.125
    assert r['min_effective_pairs_raw'] == 6
    assert r['shortfall_pairs_raw'] == 2


def test_extreme_positive_floors_do_not_overflow():
    """A subnormal p floor times an arbitrary-size family must stay finite.

    Both the Holm product and ``alpha / p_floor`` can exceed the float
    range for large effective n. They are still ordinary real quantities:
    the adjusted p is 1.0 and the supported family is an exact big integer,
    not an OverflowError.
    """
    tiny = M.exact_test_resolution(1030)
    assert 0.0 < tiny['min_attainable_p_raw'] < 1e-300
    assert tiny['family_testable'] is True
    assert tiny['max_comparisons_at_alpha'] > 10 ** 300

    huge_family = M.exact_test_resolution(1030, family_size=10 ** 400)
    assert huge_family['min_attainable_p_holm'] == 1.0
    assert huge_family['family_testable'] is False
    assert huge_family['min_effective_pairs_holm'] > 1300
    assert huge_family['max_comparisons_at_alpha'] > 10 ** 300
    json.dumps(huge_family, allow_nan=False)


def test_underflowed_raw_floor_keeps_the_exact_family_answer():
    """A p floor below the smallest float still has an exact family bound.

    At n_eff=2000 the two-sided floor ``2/2**2000`` underflows to 0.0, but
    it is not zero. A 10**400-comparison family therefore has a representable
    Holm-adjusted p (about 1e-202), and ``max_comparisons_at_alpha`` is still
    the exact floor(alpha * 2**n_eff / 2), not ``None``.
    """
    from fractions import Fraction

    r = M.exact_test_resolution(2000, family_size=10 ** 400)
    assert r['min_attainable_p_raw'] == 0.0
    expected_holm = math.exp(math.log(2.0) + math.log(10 ** 400)
                             - 2000 * math.log(2.0))
    assert r['min_attainable_p_holm'] == pytest.approx(expected_holm,
                                                       rel=1e-12)
    assert r['family_testable'] is True
    expected_family = int(
        (Fraction.from_float(0.05) * (1 << 2000) / 2).__floor__())
    assert r['max_comparisons_at_alpha'] == expected_family
    assert r['max_comparisons_at_alpha'] > 10 ** 300
    json.dumps(r, allow_nan=False)


def test_exactly_attainable_boundary_counts_the_boundary():
    """The minimum-n search must not round an exact integer boundary up.

    ``2 * 2 / 2**n <= 0.5`` first holds at n=3 (the left side is exactly
    0.5). Computing the log ratio and applying ``ceil`` can round 3.0 to
    4, overstating the seed budget by one.
    """
    r = M.exact_test_resolution(0, alpha=0.5, family_size=2)
    assert r['min_effective_pairs_holm'] == 3
    assert r['shortfall_pairs_holm'] == 3

    # 8 * 2 / 2**n <= 0.25 first holds at n=6 (again exact equality).
    r = M.exact_test_resolution(0, alpha=0.25, family_size=8)
    assert r['min_effective_pairs_holm'] == 6
    assert r['shortfall_pairs_holm'] == 6


def test_resolution_validation():
    for bad in (-1, 1.5, True, '5'):
        with pytest.raises(ValueError):
            M.exact_test_resolution(bad)
    for bad_zero in (-1, 6, 1.5, True):
        with pytest.raises(ValueError):
            M.exact_test_resolution(5, n_zero=bad_zero)
    for bad_family in (0, -2, 1.5, True):
        with pytest.raises(ValueError):
            M.exact_test_resolution(5, family_size=bad_family)
    for bad_alpha in (0.0, 1.0, -0.1, 1.1, float('nan'), True):
        with pytest.raises(ValueError):
            M.exact_test_resolution(5, alpha=bad_alpha)
    with pytest.raises(ValueError):
        M.exact_test_resolution(5, alternative='bogus')


# ---------------------------------------------------------------------------
# Clopper-Pearson win-rate interval
# ---------------------------------------------------------------------------

def test_clopper_pearson_known_endpoints_and_symmetry():
    five = M.binomial_proportion_ci(5, 5)
    assert five['ci_low'] == pytest.approx(0.4781762499, rel=1e-9)
    assert five['ci_high'] == 1.0
    zero = M.binomial_proportion_ci(0, 5)
    assert zero['ci_low'] == 0.0
    assert zero['ci_high'] == pytest.approx(0.5218237501, rel=1e-9)
    empty = M.binomial_proportion_ci(0, 0)
    assert empty['ci_low'] is None and empty['ci_high'] is None

    # k and n-k are mirror images: 1 - upper(k) == lower(n-k).
    for n in (6, 10, 21):
        for k in range(n + 1):
            a = M.binomial_proportion_ci(k, n)
            b = M.binomial_proportion_ci(n - k, n)
            assert a['ci_low'] <= k / n <= a['ci_high']
            assert a['ci_low'] == pytest.approx(1.0 - b['ci_high'],
                                                abs=1e-9)
            assert a['ci_high'] == pytest.approx(1.0 - b['ci_low'],
                                                 abs=1e-9)


def test_sign_test_reports_win_rate_and_exact_interval():
    s = M.sign_test([1, 1, 1, 1, 1])
    assert s['win_rate'] == 1.0
    lo, hi = s['win_rate_ci95']
    assert lo == pytest.approx(0.4781762499, rel=1e-9)
    assert hi == 1.0
    assert s['win_rate_ci_method'] == 'clopper_pearson_exact'
    # Ties are excluded from the proportion exactly as they are from p.
    tied = M.sign_test([1, 1, -1, 0, 0])
    assert tied['n_effective'] == 3
    assert tied['win_rate'] == pytest.approx(2 / 3)
    empty = M.sign_test([0, 0])
    assert empty['win_rate'] is None
    assert empty['win_rate_ci95'] == [None, None]


# ---------------------------------------------------------------------------
# Distribution-free median interval
# ---------------------------------------------------------------------------

def test_median_order_statistic_interval():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    five = M.median_order_statistic_ci(values)
    # The full range at n=5 covers only 93.75%: no 95% interval exists.
    assert five['ci_low'] is None and five['ci_high'] is None
    assert five['min_pairs_for_ci'] == 6

    six = M.median_order_statistic_ci([1, 2, 3, 4, 5, 6])
    assert six['k'] == 1
    assert (six['ci_low'], six['ci_high']) == (1, 6)
    assert six['coverage'] == pytest.approx(0.96875)

    twenty = M.median_order_statistic_ci(list(range(20)))
    assert twenty['ci_low'] == twenty['k'] - 1
    assert twenty['ci_high'] == 20 - twenty['k']
    assert 0.95 <= twenty['coverage'] < 1.0
    # The interval is an order-statistic interval, not a normal/t one.
    assert twenty['ci_low'] <= 9.5 <= twenty['ci_high']


# ---------------------------------------------------------------------------
# summarize_paired integration
# ---------------------------------------------------------------------------

def test_summarize_paired_attaches_resolution_and_nonparametric_cis():
    d = [0.10, 0.12, 0.14, 0.11, 0.13, 0.15]
    s = M.summarize_paired(d, n_boot=100, n_permutations=100,
                           family_size=21)
    json.dumps(s, allow_nan=False)
    expected = M.exact_test_resolution(
        6, alpha=0.05, family_size=21, alternative='two-sided', n_zero=0)
    assert s['test_resolution'] == expected
    assert s['test_resolution']['family_testable'] is False
    assert s['win_rate_ci95'] == M.sign_test(d)['win_rate_ci95']
    assert s['ci95_median_diff'] == [
        M.median_order_statistic_ci(d)['ci_low'],
        M.median_order_statistic_ci(d)['ci_high']]
    assert s['median_ci']['k'] == 1
    # The default family_size=1 record is the per-comparison view.
    default = M.summarize_paired(d, n_boot=50, n_permutations=50)
    assert default['test_resolution']['family_size'] == 1


def test_summarize_paired_rejects_bad_family_size():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            M.summarize_paired([0.1, 0.2], n_boot=10, family_size=bad)


# ---------------------------------------------------------------------------
# run_quality integration
# ---------------------------------------------------------------------------

def _row(op, accs, task='mqar', n_kv=1):
    return {
        'op': op, 'task': task, 'n_kv': n_kv,
        'mean_acc': sum(accs) / len(accs),
        'per_seed': [{'seed': 42 + i, 'final_acc': a}
                     for i, a in enumerate(accs)],
    }


def test_pairwise_report_flags_resolution_limited_family():
    ops = ['linear', 'softmax', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca']
    rows = []
    for oi, op in enumerate(ops):
        # Distinct per-seed values avoid accidental ties.
        accs = [0.10 + 0.01 * oi + 0.001 * i for i in range(5)]
        rows.append(_row(op, accs))
    report = _pairwise_op_comparisons(rows, n_boot=50,
                                      n_permutations=50)
    assert report['n_tests'] == 21
    assert report['n_resolution_limited'] == 21
    assert report['resolution']['all_pairs_testable'] is False
    assert report['resolution']['any_pair_testable'] is False
    # All 21 pairs have 5 seeds: raw p floor 0.0625 > alpha, so even a
    # later rank is impossible.
    assert report['resolution']['n_later_rank_possible'] == 0
    assert report['resolution']['worst_min_attainable_p_holm'] == 1.0
    cell = report['cells']['mqar/n_kv=1']
    assert cell['resolution']['status'] == 'resolution_limited'
    assert cell['n_resolution_limited'] == 21
    rec = next(iter(cell['pairs'].values()))
    assert rec['test_resolution']['family_size'] == 21
    assert rec['test_resolution']['family_testable'] is False
    json.dumps(report, allow_nan=False)


def test_pairwise_report_single_comparison_stays_testable():
    rows = [_row('op_a', [0.9] * 6), _row('op_b', [0.1] * 6)]
    report = _pairwise_op_comparisons(rows, n_boot=100,
                                      n_permutations=100)
    assert report['n_tests'] == 1
    assert report['n_resolution_limited'] == 0
    assert report['resolution']['all_pairs_testable'] is True
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert rec['test_resolution']['family_testable'] is True
    assert rec['test_resolution']['min_attainable_p_raw'] == 0.03125


# ---------------------------------------------------------------------------
# experiment_design integration
# ---------------------------------------------------------------------------

def test_planning_reports_the_exact_test_floor():
    plan = D.get_plan('exp4_mqar')
    screening = D.planning_report(plan, seeds=5, family_size=21)
    assert screening['exact_test_resolution']['family_testable'] is False
    assert screening['exact_test_warning']
    assert 'no rejection region' in screening['exact_test_warning']

    planned = D.planning_report(plan, seeds=plan['planned_seeds'],
                                family_size=plan['planned_family_size'])
    assert planned['exact_test_resolution']['family_testable'] is True
    assert planned['exact_test_warning'] is None


def test_registry_validation_rejects_a_resolution_limited_plan():
    plan = dict(D.get_plan('exp4_mqar'))
    # A tiny assumed SD makes the normal-approximation budget tiny, but a
    # 6-seed x 21-comparison exact family still has no rejection region.
    plan['assumed_sd'] = 0.0001
    plan['planned_seeds'] = 6
    plan['screening_seeds'] = 5
    plan['planned_family_size'] = 21
    errors = D.validate_plan(plan)
    assert any('no exact-test rejection region' in e for e in errors), errors


def test_audit_downgrades_a_resolution_limited_confirmatory_power():
    # 5 seeds with a 21-pair family: the normal approximation for a huge
    # effect can return confirmatory-ready, but the exact test cannot
    # reject. The audit must not call that confirmatory.
    audit = D.audit_run(
        'exp4_mqar', seeds=[42, 43, 44, 45, 46],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=21, sd=0.001)
    assert audit['checks']['exact_test_has_rejection_region'] is False
    assert audit['status'] == 'screening'
    assert any('no rejection region' in w for w in audit['design_warnings'])


def test_resolution_helpers_do_not_touch_global_rng():
    state = random.getstate()
    M.exact_test_resolution(6, family_size=21)
    M.binomial_proportion_ci(5, 5)
    M.median_order_statistic_ci([1, 2, 3, 4, 5, 6])
    M.summarize_paired([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       n_boot=20, n_permutations=20)
    assert random.getstate() == state
