"""Regression tests for the multi-seed paired-inference toolkit.

``multi_seed_stats.py`` decides which operator-vs-operator claims the
quality experiments (Exp 4 / Exp 5) are allowed to make, so its exact
statistics are pinned here against hand-computed values:

* exact binomial sign test (combinatorial tails);
* exact sign-flip permutation test (full ``2^n`` enumeration) plus the
  deterministic, RNG-isolated Monte-Carlo branch;
* Student-t paired test against a two-sided p verified by direct
  Simpson integration of the t density (t=4.24264, df=4,
  p=0.0132355995637);
* Holm-Bonferroni step-down on a hand-worked family;
* bootstrap determinism AND global-RNG isolation (resampling statistics
  must never perturb the training seed streams);
* power-planning helpers;
* the ``run_quality._pairwise_op_comparisons`` integration (row
  annotations, Holm verdicts, JSON safety).

How to run::

    pytest -q test_multi_seed_stats.py          # from the repo root

The file is intentionally standalone (no conftest dependency);
``pyproject.toml`` sets ``testpaths = ["."]`` so any bare ``pytest`` run
collects it.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multi_seed_stats as M  # noqa: E402
from run_quality import _pairwise_op_comparisons  # noqa: E402


# ---------------------------------------------------------------------------
# Student-t paired test
# ---------------------------------------------------------------------------

def test_paired_t_statistic_and_pvalue_match_hand_values():
    # diffs [1,2,3,4,5] -> mean 3, sample var 2.5, se=sqrt(0.5),
    # t = 3/0.70710678 = 4.242640687, df=4.
    # Two-sided p cross-checked here by Simpson integration of the t
    # density (t_pdf via the gamma function): 0.0132355995637.
    t = M.paired_t_test([1, 2, 3, 4, 5])
    assert t['n_pairs'] == 5
    assert t['dof'] == 4
    assert math.isclose(t['t_stat'], 4.242640687119285, rel_tol=1e-12)
    assert math.isclose(t['p_value'], 0.013235599563682587, rel_tol=1e-9)
    assert t['all_positive'] and not t['all_negative']


def test_paired_t_numerical_integration_cross_check():
    """Verify the analytic p-value against direct numerical integration."""
    diffs = [1, 2, 3, 4, 5]
    t = M.paired_t_test(diffs)['t_stat']

    def t_pdf(x, dof):
        return (math.gamma((dof + 1) / 2)
                / (math.sqrt(dof * math.pi) * math.gamma(dof / 2))
                * (1.0 + x * x / dof) ** (-(dof + 1) / 2))

    lo, hi, n = -t, t, 40000
    h = (hi - lo) / n
    s = t_pdf(lo, 4) + t_pdf(hi, 4)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * t_pdf(lo + i * h, 4)
    cdf = s * h / 3.0
    assert abs((1.0 - cdf) - M.paired_t_test(diffs)['p_value']) < 1e-9


def test_paired_t_directional_pvalue_is_half_the_two_sided():
    t2 = M.paired_t_test([1, 2, 3, 4, 5])
    tg = M.paired_t_test([1, 2, 3, 4, 5], alternative='greater')
    tl = M.paired_t_test([1, 2, 3, 4, 5], alternative='less')
    assert math.isclose(tg['p_value'], t2['p_value'] / 2.0, rel_tol=1e-12)
    # The opposite direction is the t CDF at +4.2426 (df=4) = 0.993382...
    assert math.isclose(tl['p_value'], 1.0 - t2['p_value'] / 2.0,
                        rel_tol=1e-9)


def test_paired_t_edge_cases():
    empty = M.paired_t_test([])
    assert empty['n_pairs'] == 0 and empty['p_value'] is None
    single = M.paired_t_test([0.3])
    assert single['n_pairs'] == 1 and single['p_value'] is None
    assert single['all_positive'] is True
    # Zero variance with a non-zero mean: t is undefined, and the record
    # must say so rather than fabricate +-inf / p=0.
    flat = M.paired_t_test([0.5, 0.5, 0.5])
    assert flat['t_stat'] is None and flat['p_value'] is None
    assert flat['all_positive'] is True
    with pytest.raises(ValueError):
        M.paired_t_test([1.0], alternative='bogus')


# ---------------------------------------------------------------------------
# Exact sign test
# ---------------------------------------------------------------------------

def test_sign_test_exact_binomial_tails():
    # 5/5 positive: p = 2 * (1/2)^5 = 0.0625.
    s1 = M.sign_test([1, 1, 1, 1, 1])
    assert s1['n_effective'] == 5
    assert math.isclose(s1['p_value'], 0.0625, rel_tol=1e-15)
    # 4/5 positive: p = 2 * (C(5,5) + C(5,4)) / 2^5 = 12/32 = 0.375.
    s2 = M.sign_test([1, 1, 1, 1, -1])
    assert math.isclose(s2['p_value'], 0.375, rel_tol=1e-15)
    # 6/6 positive: p = 2/64 = 0.03125 (the first significant unanimous
    # sweep at alpha=0.05).
    s3 = M.sign_test([0.2] * 6)
    assert math.isclose(s3['p_value'], 0.03125, rel_tol=1e-15)


def test_sign_test_zeros_are_dropped_steel_convention():
    s = M.sign_test([0.0, 0.0, 0.0])
    assert s['n_effective'] == 0 and s['n_zero'] == 3
    assert s['p_value'] == 1.0
    mixed = M.sign_test([0.0, 1.0, 1.0, -1.0])
    assert mixed['n_effective'] == 3 and mixed['n_zero'] == 1
    assert math.isclose(mixed['p_value'], 1.0, rel_tol=1e-15)  # 2/2 = 1


# ---------------------------------------------------------------------------
# Exact / Monte-Carlo sign-flip permutation test
# ---------------------------------------------------------------------------

def test_permutation_test_exact_enumeration_values():
    # 3/3 positive: only the two all-agree sign vectors reach |mean| = 1.
    p3 = M.permutation_test([1, 1, 1])
    assert p3['method'] == 'exact' and p3['n_permutations'] == 8
    assert math.isclose(p3['p_value'], 0.25, rel_tol=1e-15)
    p4 = M.permutation_test([1, 1, 1, 1])
    assert p4['n_permutations'] == 16
    assert math.isclose(p4['p_value'], 0.125, rel_tol=1e-15)
    # All-zero diffs: every sign vector ties the observed statistic, so
    # the tail counts all of them and p = 1 (never 0).
    p0 = M.permutation_test([0.0, 0.0, 0.0])
    assert p0['p_value'] == 1.0
    # Symmetric data: the observed mean is achieved by many sign vectors.
    ps = M.permutation_test([1.0, -1.0])
    assert math.isclose(ps['p_value'], 1.0, rel_tol=1e-15)


def test_permutation_test_monte_carlo_is_deterministic_and_rng_isolated():
    diffs = [0.5, -0.4, 0.3, -0.2, 0.6, -0.1, 0.2, 0.4, 0.1, -0.3,
             0.5, 0.7, -0.6, 0.2, 0.3, 0.1, -0.2, 0.4, 0.6, 0.2]
    state = random.getstate()
    first = M.permutation_test(diffs, n_permutations=200, seed=11,
                               max_exact=16)
    assert random.getstate() == state, (
        'permutation_test leaked into the global RNG stream')
    second = M.permutation_test(diffs, n_permutations=200, seed=11,
                                max_exact=16)
    assert first['method'] == 'monte_carlo'
    assert first['p_value'] == second['p_value']
    assert 0.0 < first['p_value'] <= 1.0
    third = M.permutation_test(diffs, n_permutations=200, seed=12,
                               max_exact=16)
    assert isinstance(third['p_value'], float)


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down
# ---------------------------------------------------------------------------

def test_holm_adjust_matches_hand_worked_family():
    # p = [0.005, 0.01, 0.03, 0.04], m=4:
    #   rank1 0.005*4 = 0.02, rank2 0.01*3 = 0.03,
    #   rank3 0.03*2 = 0.06, rank4 0.04*1 = 0.04
    #   running max -> [0.02, 0.03, 0.06, 0.06].
    holm = M.holm_adjust([0.005, 0.01, 0.03, 0.04], alpha=0.05)
    assert holm['adjusted'] == [0.02, 0.03, 0.06, 0.06]
    assert holm['rejected'] == [True, True, False, False]
    assert holm['n_tests'] == 4


def test_holm_adjust_edge_cases():
    assert M.holm_adjust([])['adjusted'] == []
    one = M.holm_adjust([0.03], alpha=0.05)
    assert one['adjusted'] == [0.03] and one['rejected'] == [True]
    # Adjusted p-values are monotonically ordered by the raw p-value rank.
    h = M.holm_adjust([0.9, 0.001, 0.2])
    assert h['adjusted'][1] <= h['adjusted'][2] <= h['adjusted'][0]
    with pytest.raises(ValueError):
        M.holm_adjust([0.5, None])
    with pytest.raises(ValueError):
        M.holm_adjust([1.5])


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ci_is_reproducible_and_rng_isolated():
    state = random.getstate()
    first = M.bootstrap_paired_ci([1, 2, 3, 4, 5], n_boot=2000, seed=7)
    assert random.getstate() == state, (
        'bootstrap_paired_ci leaked into the global RNG stream')
    second = M.bootstrap_paired_ci([1, 2, 3, 4, 5], n_boot=2000, seed=7)
    assert first == second
    assert first['ci_lo'] <= 3.0 <= first['ci_hi']


def test_bootstrap_ci_undefined_for_single_pair():
    b = M.bootstrap_paired_ci([0.5])
    assert b['ci_lo'] is None and b['ci_hi'] is None
    with pytest.raises(ValueError):
        M.bootstrap_paired_ci([0.5, 1.0], n_boot=0)


# ---------------------------------------------------------------------------
# Power planning
# ---------------------------------------------------------------------------

def test_seeds_required_hand_values_and_monotonicity():
    # ((1.959964 + 0.841621) / 1.0)^2 = 7.849 -> 8.
    assert M.seeds_required(1.0) == 8
    # ((1.959964 + 0.841621) / 0.5)^2 = 31.396 -> 32.
    assert M.seeds_required(0.5) == 32
    assert M.seeds_required(0.0) is None
    assert M.seeds_required(0.2) > M.seeds_required(0.4)


def test_min_detectable_effect_decreases_with_seeds():
    mde_5 = M.min_detectable_effect(5, 0.1)
    mde_20 = M.min_detectable_effect(20, 0.1)
    assert 0.0 < mde_20 < mde_5
    assert M.min_detectable_effect(1, 0.1) is None


# ---------------------------------------------------------------------------
# summarize_paired panel
# ---------------------------------------------------------------------------

def test_summarize_paired_is_json_safe_and_direction_aware():
    summary = M.summarize_paired([0.9, 0.8, 0.7, 0.85, 0.95, 0.75],
                                 n_boot=200, seed=3)
    json.dumps(summary, allow_nan=False)
    assert summary['direction'] == 'greater'
    assert summary['n_pairs'] == 6
    assert summary['win_rate'] == 1.0
    assert summary['cohens_dz'] > 0
    lo, hi = summary['ci95_mean_diff_t']
    assert lo <= summary['mean_diff'] <= hi
    lo_b, hi_b = summary['ci95_mean_diff_bootstrap']
    assert lo_b <= summary['mean_diff'] <= hi_b


def test_summarize_paired_zero_variance_sweep():
    # A unanimous, constant-magnitude sweep: the t-test is undefined, but
    # the exact sign/permutation tests still produce a finite p-value.
    flat = M.summarize_paired([0.01] * 5, n_boot=200, seed=0)
    assert flat['t_test']['p_value_two_sided'] is None
    assert flat['t_test']['all_positive'] is True
    assert math.isclose(flat['sign_test']['p_value_two_sided'], 0.0625,
                        rel_tol=1e-15)
    assert math.isclose(flat['permutation_test']['p_value_two_sided'],
                        0.0625, rel_tol=1e-15)
    six = M.summarize_paired([0.01] * 6, n_boot=200, seed=0)
    assert math.isclose(six['permutation_test']['p_value_two_sided'],
                        0.03125, rel_tol=1e-15)
    assert six['seeds_required_80pct_power'] is None  # d_z undefined


def test_summarize_paired_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):
        M.summarize_paired([1.0, float('nan')])
    with pytest.raises(ValueError):
        M.summarize_paired([1.0, float('inf')])


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


def test_pairwise_op_comparisons_significant_unanimous_sweep():
    rows = [_row('op_a', [0.9] * 6), _row('op_b', [0.1] * 6)]
    report = _pairwise_op_comparisons(rows, alpha=0.05, seed=0,
                                      n_boot=200, n_permutations=200)
    json.dumps(report, allow_nan=False)
    assert report['n_tests'] == 1 and report['n_significant'] == 1
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert rec['verdict'] == 'op_a_better'
    assert rec['significant_holm_permutation'] is True
    assert math.isclose(rec['permutation_test']['p_value_two_sided'],
                        0.03125, rel_tol=1e-12)
    # Row annotations must point at each other.
    assert rows[0]['pairwise_significant_wins'] == ['op_b']
    assert rows[1]['pairwise_significant_losses'] == ['op_a']
    assert rows[0]['pairwise_tests'] == 1
    # The significant pair carries a forward-looking seed budget.
    assert rec['seeds_required_80pct_power'] is None  # d_z undefined (flat)
    assert rec['mean_diff'] == pytest.approx(0.8)


def test_pairwise_op_comparisons_five_seed_sweep_stays_inconclusive():
    # 5/5 unanimous has exact p = 2/32 = 0.0625 > 0.05: the toolkit must
    # NOT promote it via an ad-hoc "all positive" heuristic.
    rows = [_row('op_a', [0.9] * 5), _row('op_b', [0.1] * 5)]
    report = _pairwise_op_comparisons(rows, alpha=0.05, seed=0,
                                      n_boot=200, n_permutations=200)
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert rec['verdict'] == 'not_significant'
    assert rec['significant_holm_permutation'] is False
    assert report['n_significant'] == 0
    assert rows[0]['pairwise_significant_wins'] == []


def test_pairwise_op_comparisons_skips_when_fewer_than_two_common_seeds():
    a = _row('op_a', [0.9, 0.8])
    b = _row('op_b', [0.1, 0.2])
    b['per_seed'][0]['seed'] = 100  # break the pairing: only seed 43 common
    report = _pairwise_op_comparisons([a, b], alpha=0.05, seed=0,
                                      n_boot=100, n_permutations=100)
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert rec['verdict'] == 'insufficient_paired_seeds'
    assert report['n_tests'] == 0 and report['n_skipped'] == 1
    assert 'pairwise_tests' not in a


def test_pairwise_op_comparisons_holm_family_spans_the_cell():
    # Three ops in one cell -> 3 pairwise tests in one Holm family; a
    # marginal raw p-value must be adjusted upwards.
    rows = [_row('op_a', [0.9] * 6),
            _row('op_b', [0.1] * 6),
            _row('op_c', [0.105] * 6)]
    report = _pairwise_op_comparisons(rows, alpha=0.05, seed=0,
                                      n_boot=200, n_permutations=200)
    cell = report['cells']['mqar/n_kv=1']
    assert cell['n_tests'] == 3
    recs = cell['pairs']
    # a vs b is unanimous (p=0.03125 raw) -> adjusted upward by Holm.
    rec_ab = recs['op_a~op_b']
    assert rec_ab['holm_adjusted_p_permutation'] >= \
        rec_ab['permutation_test']['p_value_two_sided']
    # b vs c differs on 5/6 seeds (one tie? no: strict ordering) -> exact
    # enumeration p-value is finite and in (0,1].
    rec_bc = recs['op_b~op_c']
    assert 0.0 < rec_bc['permutation_test']['p_value_two_sided'] <= 1.0
