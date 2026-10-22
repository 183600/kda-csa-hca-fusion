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
import statistics
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


def test_student_t_survival_tail_stays_positive():
    """The upper-tail p-value must not collapse to 0 below machine epsilon.

    Closed form for dof=1: P(T > t) = 0.5 - atan(t)/pi. Computing it as
    1 - CDF used to underflow to exactly 0.0, turning a tiny but real
    p-value into a fabricated certainty.
    """
    t_value = 1.0e6
    tail = 0.5 - math.atan(t_value) / math.pi
    assert math.isclose(M.student_t_sf(t_value, 1), tail, rel_tol=1e-9)
    assert M.student_t_sf(t_value, 1) > 0.0
    for dof in (2, 3, 5, 10):
        cdf = M.student_t_cdf(5.0, dof)
        sf = M.student_t_sf(5.0, dof)
        assert math.isclose(cdf + sf, 1.0, rel_tol=1e-12)
        assert sf > 0.0
        assert M.paired_t_test([0.0] * (dof - 1) + [10.0 * math.sqrt(dof)])[
            'p_value'] > 0.0


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
# TOST equivalence test
# ---------------------------------------------------------------------------

def _t_pdf_dof4(x):
    """Student-t(4) density, used only by the independent TOST check."""
    return (math.gamma(2.5) / (math.sqrt(4 * math.pi) * math.gamma(2.0))
            * (1.0 + x * x / 4.0) ** (-2.5))


def _simpson_prefix(t):
    """``int_0^t f(x) dx`` for the t(4) density via composite Simpson.

    ``x >= 0`` only. Combined with the density's symmetry, ``0.5 -
    int_0^t`` gives the one-sided tail probability ``P(T > t)`` without
    relying on ``multi_seed_stats``' own incomplete-beta CDF, so the TOST
    p-values can be cross-checked against an independent numerical
    computation (the same verification strategy as the paired-t test
    above).
    """
    n, h = 40000, t / 40000.0
    s = _t_pdf_dof4(0.0) + _t_pdf_dof4(t)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * _t_pdf_dof4(i * h)
    return s * h / 3.0


def test_tost_paired_hand_values_and_numerical_cross_check():
    # diffs [0.1..0.5]: mean 0.3, sd 0.15811388, se 0.07071068, df=4.
    # margin 0.5: t_upper = (0.3-0.5)/se = -2.82842712 ->
    # p_upper = P(T <= -2.82842712) = 0.0237103277921598 (verified with
    # mpmath.betainc at 40 digits and by direct density integration);
    # t_lower = (0.3+0.5)/se = 11.31370850 -> p_lower = 0.0001739457...
    # TOST p = max = 0.0237103277921598 < 0.05 -> equivalent.
    r = M.tost_paired([0.1, 0.2, 0.3, 0.4, 0.5], margin=0.5)
    assert r['n_pairs'] == 5 and r['dof'] == 4
    assert math.isclose(r['t_upper'], -2.8284271247461903, rel_tol=1e-12)
    assert math.isclose(r['t_lower'], 11.313708498984761, rel_tol=1e-12)
    assert math.isclose(r['p_upper'], 0.02371032779215981, rel_tol=1e-9)
    assert math.isclose(r['p_lower'], 0.00017394572028528454, rel_tol=1e-9)
    assert math.isclose(r['p_value'], r['p_upper'], rel_tol=1e-15)
    assert r['equivalent'] is True
    # Independent numerical check via density symmetry: P(T <= t) =
    # 0.5 - int_0^{-t} f for t < 0, and P(T >= t) = 0.5 - int_0^t f.
    p_upper_num = 0.5 - _simpson_prefix(-r['t_upper'])
    p_lower_num = 0.5 - _simpson_prefix(r['t_lower'])
    assert abs(r['p_upper'] - p_upper_num) < 1e-9
    assert abs(r['p_lower'] - p_lower_num) < 1e-9
    # (1 - 2*alpha) CI is centred on the mean and agrees with the verdict.
    assert math.isclose(r['ci_equiv_lo'], 0.14925566809376792, rel_tol=1e-9)
    assert math.isclose(r['ci_equiv_hi'], 0.4507443319062321, rel_tol=1e-9)
    assert r['ci_equiv_lo'] > -0.5 and r['ci_equiv_hi'] < 0.5
    # A tighter margin flips the verdict without changing the data.
    tight = M.tost_paired([0.1, 0.2, 0.3, 0.4, 0.5], margin=0.2)
    assert tight['equivalent'] is False
    assert math.isclose(tight['p_value'], 0.8849001794597504, rel_tol=1e-9)


def test_tost_paired_degenerate_and_error_cases():
    single = M.tost_paired([0.1], margin=0.2)
    assert single['p_value'] is None and single['equivalent'] is None
    empty = M.tost_paired([], margin=0.2)
    assert empty['p_value'] is None and empty['equivalent'] is None
    # Zero variance: point CI, exact 0 / 0.5 / 1 p at inside / boundary /
    # outside, and the verdict must follow |mean| vs margin exactly.
    inside = M.tost_paired([0.1] * 5, margin=0.2)
    assert inside['p_value'] == 0.0 and inside['equivalent'] is True
    boundary = M.tost_paired([0.2] * 5, margin=0.2)
    assert boundary['p_value'] == 0.5 and boundary['equivalent'] is False
    outside = M.tost_paired([0.3] * 5, margin=0.2)
    assert outside['p_value'] == 1.0 and outside['equivalent'] is False
    for bad_margin in (0.0, -0.1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            M.tost_paired([0.1, 0.2], margin=bad_margin)
    for bad_alpha in (0.0, 0.5, 0.9, 1.0):
        with pytest.raises(ValueError):
            M.tost_paired([0.1, 0.2], margin=0.2, alpha=bad_alpha)
    with pytest.raises(ValueError):
        M.tost_paired([0.1, float('nan')], margin=0.2)


def test_tost_from_summary_matches_vector_entry_point():
    d = [0.1, 0.2, 0.3, 0.4, 0.5]
    summary = M.tost_from_summary(sum(d) / len(d), statistics.stdev(d),
                                  len(d), margin=0.5)
    assert summary == pytest.approx(M.tost_paired(d, margin=0.5))
    # Degenerate branches mirror the vector entry point exactly.
    assert M.tost_from_summary(0.1, None, 1, margin=0.2) == \
        M.tost_paired([0.1], margin=0.2)
    assert M.tost_from_summary(None, None, 0, margin=0.2) == \
        M.tost_paired([], margin=0.2)
    assert M.tost_from_summary(0.3, 0.0, 5, margin=0.2) == \
        M.tost_paired([0.3] * 5, margin=0.2)
    for bad_n in (-1, 2.0, True):
        with pytest.raises(ValueError):
            M.tost_from_summary(0.1, 0.05, bad_n, margin=0.2)
    with pytest.raises(ValueError):
        M.tost_from_summary(None, 0.05, 3, margin=0.2)
    with pytest.raises(ValueError):
        M.tost_from_summary(float('nan'), 0.05, 3, margin=0.2)
    with pytest.raises(ValueError):
        M.tost_from_summary(0.1, None, 3, margin=0.2)
    with pytest.raises(ValueError):
        M.tost_from_summary(0.1, -0.05, 3, margin=0.2)


def test_equivalence_margin_supported_matches_ci_half_width():
    d = [0.1, 0.2, 0.3, 0.4, 0.5]
    supported = M.equivalence_margin_supported(d)
    # |mean| + t_{1-alpha,4} * se = the upper CI endpoint, computed by the
    # module's own tost CI for the same data.
    r = M.tost_paired(d, margin=0.5)
    assert math.isclose(supported, r['ci_equiv_hi'], rel_tol=1e-12)
    assert math.isclose(supported, 0.4507443319062321, rel_tol=1e-9)
    # Cross-checked on a second vector with an independent SD (statistics)
    # and the tabulated t_{0.95,5} = 2.0150483733.
    d2 = [-0.2, 0.05, 0.3, 0.15, -0.05, 0.1]
    sup2 = M.equivalence_margin_supported(d2)
    mean2 = sum(d2) / len(d2)
    se2 = statistics.stdev(d2) / math.sqrt(len(d2))
    assert math.isclose(sup2, abs(mean2) + 2.0150483733 * se2, rel_tol=1e-9)
    # A margin a hair above the supported infimum declares equivalence;
    # a hair below must not.
    assert M.tost_paired(d2, margin=sup2 + 1e-9)['equivalent'] is True
    assert M.tost_paired(d2, margin=sup2 - 1e-9)['equivalent'] is False
    assert M.equivalence_margin_supported([0.1]) is None
    # Zero variance: infimum is |mean| (strict inequality documented).
    assert M.equivalence_margin_supported([0.4] * 5) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        M.equivalence_margin_supported(d, alpha=0.5)


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


def test_power_at_effect_inverts_seeds_required():
    # The forward power and the inverse sample-size rule must agree: at the
    # n returned by seeds_required, power is at least the requested target.
    for dz in (0.5, 1.0, 1.5):
        n = M.seeds_required(dz, alpha=0.05, power=0.8)
        power = M.power_at_effect(n, sd=1.0, effect=dz, alpha=0.05)
        assert power is not None and power >= 0.8 - 1e-9
    # One-sided tests need fewer pairs than two-sided tests for the same
    # effect; the null effect has power alpha (two-sided).
    assert M.power_at_effect(5, 1.0, 1.0, alternative='greater') > \
        M.power_at_effect(5, 1.0, 1.0)
    assert M.power_at_effect(5, 1.0, 0.0) == pytest.approx(0.05)
    assert M.power_at_effect(5, 1.0, 0.0, alternative='greater') == \
        pytest.approx(0.05)
    # Degenerate / undefined cases are explicit, and invalid arguments still
    # fail loudly before the n < 2 shortcut.
    assert M.power_at_effect(1, 1.0, 1.0) is None
    assert M.power_at_effect(5, 0.0, 0.0) == pytest.approx(0.05)
    assert M.power_at_effect(5, 0.0, 1.0) == 1.0
    # Directional power follows the SIGN of the effect.  The old magnitude-
    # only implementation returned the ``greater`` power for ``less`` and
    # full noiseless power for an effect pointing the wrong way.
    assert M.power_at_effect(5, 1.0, 1.0, alternative='less') < 0.05
    assert M.power_at_effect(5, 1.0, -1.0, alternative='less') > \
        M.power_at_effect(5, 1.0, 1.0, alternative='less')
    assert M.power_at_effect(5, 1.0, -1.0, alternative='greater') < 0.05
    assert M.power_at_effect(5, 0.0, 1.0, alternative='greater') == 1.0
    assert M.power_at_effect(5, 0.0, -1.0, alternative='greater') == 0.0
    assert M.power_at_effect(5, 0.0, -1.0, alternative='less') == 1.0
    assert M.power_at_effect(5, 0.0, 1.0, alternative='less') == 0.0
    assert M.power_at_effect(5, 0.0, -1.0) == 1.0
    with pytest.raises(ValueError):
        M.power_at_effect(True, 1.0, 1.0)
    with pytest.raises(ValueError):
        M.power_at_effect(5, -0.1, 1.0)
    with pytest.raises(ValueError):
        M.power_at_effect(5, 1.0, float('nan'))
    with pytest.raises(ValueError):
        M.power_at_effect(5, 1.0, 1.0, alpha=0.0)
    with pytest.raises(ValueError):
        M.power_at_effect(5, 1.0, 1.0, alternative='both')


def test_power_planning_rejects_invalid_arguments_before_undefined_n():
    # An undefined seed count (n < 2) must not mask invalid planning
    # inputs: an alpha/sd typo has to fail loudly even when the result is
    # None, otherwise a broken analysis plan looks like a valid "n/a".
    for bad_n in (True, 1.5, -1):
        with pytest.raises(ValueError):
            M.min_detectable_effect(bad_n, 0.1)
    with pytest.raises(ValueError):
        M.min_detectable_effect(1, 0.1, alpha=0.0)
    with pytest.raises(ValueError):
        M.min_detectable_effect(1, float('nan'))
    with pytest.raises(ValueError):
        M.min_detectable_effect(5, True)
    with pytest.raises(ValueError):
        M.seeds_required(True)
    with pytest.raises(ValueError):
        M.seeds_required('1.0')
    # Zero and non-finite effects stay the documented None cases.
    assert M.seeds_required(float('nan')) is None
    assert M.seeds_required(float('inf')) is None


def test_resampling_arguments_reject_bool_float_and_zero():
    """Bootstrap / permutation counts and seeds are exact ints.

    Previously ``n_boot=True`` silently meant one resample, ``n_boot=1.5``
    leaked a raw ``TypeError`` from ``range``, and float seeds were
    truncated by ``random.Random(int(seed))``. All are caller bugs that
    must fail before the undefined-n / exact-enumeration shortcuts hide
    them.
    """
    d = [0.1, 0.2, 0.3, 0.4] * 5          # n > max_exact for Monte Carlo
    for bad in (True, 1.5, 0):
        with pytest.raises(ValueError):
            M.bootstrap_paired_ci(d, n_boot=bad)
        with pytest.raises(ValueError):
            M.permutation_test(d, n_permutations=bad)
    for bad in (True, 1.5):
        with pytest.raises(ValueError):
            M.bootstrap_paired_ci(d, seed=bad)
        with pytest.raises(ValueError):
            M.permutation_test(d, seed=bad)
    for bad in (True, 1.5, -1):
        with pytest.raises(ValueError):
            M.permutation_test(d, max_exact=bad)
    # Validation runs even on the undefined / exact shortcuts.
    with pytest.raises(ValueError):
        M.bootstrap_paired_ci([0.1], n_boot=0)
    with pytest.raises(ValueError):
        M.permutation_test([0.1], n_permutations=0)


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


def test_sample_sd_keeps_real_spread_at_a_large_offset():
    """Only an exactly constant vector is numerically zero-variance.

    The old clamp compared the variance with ``(1e-12 * max|value|)**2``,
    so a genuine spread of 1 around 1e12 was reported as zero and the
    paired t-test / d_z / TOST all became undefined.
    """
    flat = [1e12, 1e12 + 1.0, 1e12 + 2.0]
    mean = sum(flat) / len(flat)
    assert M._sample_sd(flat, mean) == pytest.approx(1.0)
    t = M.paired_t_test(flat)
    assert t['sd_diff'] == pytest.approx(1.0)
    assert t['t_stat'] == pytest.approx(mean / (1.0 / (3 ** 0.5)))
    assert t['p_value'] is not None
    assert M.cohens_dz(flat) == pytest.approx(mean)
    tost = M.tost_paired(flat, margin=0.5)
    assert tost['sd_diff'] == pytest.approx(1.0)
    assert tost['p_value'] is not None
    assert tost['ci_equiv_lo'] < tost['ci_equiv_hi']
    # Exactly constant vectors (including non-binary values) stay zero.
    const = [0.1, 0.1, 0.1]
    assert M._sample_sd(const, sum(const) / 3) == 0.0
    assert M.paired_t_test(const)['t_stat'] is None


def test_summarize_paired_without_margin_has_no_equivalence_claim():
    # Backward compatibility: without a pre-specified margin there must be
    # no TOST block and no equivalence verdict -- a non-significant test
    # can never silently become a "same" claim.
    s = M.summarize_paired([0.4, -0.2, 0.1], n_boot=200, seed=0)
    assert 'tost' not in s
    assert s['practically_equivalent'] is None
    assert s['equivalence_note']


def test_summarize_paired_with_margin_reports_tost_block():
    diffs = [0.004, -0.002, 0.003, 0.001, -0.003, 0.002]
    s = M.summarize_paired(diffs, n_boot=200, seed=0,
                           equivalence_margin=0.05)
    json.dumps(s, allow_nan=False)
    assert s['practically_equivalent'] is True
    assert s['tost']['margin'] == 0.05
    assert s['tost']['equivalent'] is True
    assert s['tost']['p_value'] < 0.05
    assert s['tost']['ci_equiv_level'] == pytest.approx(0.90)
    assert s['equivalence_margin_supported'] is not None
    # The nested TOST record is exactly the standalone function's output.
    assert s['tost'] == M.tost_paired(diffs, margin=0.05)
    far = M.summarize_paired([0.9, 0.1, 0.5], n_boot=200, seed=0,
                             equivalence_margin=0.05)
    assert far['practically_equivalent'] is False
    assert far['tost']['p_value'] > 0.05
    assert far['tost']['p_value'] == far['tost']['p_upper']


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


def test_pairwise_op_comparisons_equivalence_separates_same_from_different():
    # a1/a2 differ by <= 0.005 on every seed: practically the same at the
    # 0.05 margin. a1/c differs by ~0.8: definitely not equivalent. The
    # near-identical pair's superiority test is non-significant, which is
    # exactly the case that must NOT be misread as equivalence without
    # the TOST.
    rows = [_row('op_a1', [0.900, 0.905, 0.895, 0.901, 0.899, 0.903]),
            _row('op_a2', [0.904, 0.902, 0.894, 0.903, 0.897, 0.900]),
            _row('op_c', [0.100, 0.105, 0.095, 0.101, 0.099, 0.103])]
    report = _pairwise_op_comparisons(
        rows, alpha=0.05, seed=0, n_boot=200, n_permutations=200,
        equivalence_margin=0.05)
    json.dumps(report, allow_nan=False)
    assert report['equivalence']['enabled'] is True
    cell = report['cells']['mqar/n_kv=1']
    rec = cell['pairs']['op_a1~op_a2']
    assert rec['verdict'] == 'not_significant'
    assert rec['equivalence_verdict'] == 'equivalent'
    assert rec['significant_equivalence_holm'] is True
    assert rec['holm_adjusted_p_tost'] >= rec['tost']['p_value']
    assert rec['holm_adjusted_p_tost'] <= 0.05
    assert rec['equivalence_margin_supported'] < 0.05
    rec_far = cell['pairs']['op_a1~op_c']
    assert rec_far['equivalence_verdict'] == 'not_equivalent'
    assert rec_far['significant_equivalence_holm'] is False
    assert report['n_equivalent'] == 1
    assert 'op_a2' in rows[0]['pairwise_equivalent_to']
    assert 'op_c' not in rows[0]['pairwise_equivalent_to']


def test_pairwise_op_comparisons_equivalence_disabled_by_default():
    rows = [_row('op_a', [0.90] * 6), _row('op_b', [0.905] * 6)]
    report = _pairwise_op_comparisons(rows, alpha=0.05, seed=0,
                                      n_boot=200, n_permutations=200)
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert 'tost' not in rec
    assert 'equivalence_verdict' not in rec
    assert 'pairwise_equivalent_to' not in rows[0]
    assert report['n_equivalent'] == 0
    assert report['equivalence']['enabled'] is False


# ---------------------------------------------------------------------------
# Leave-one-seed-out sensitivity
# ---------------------------------------------------------------------------

def test_leave_one_out_stable_unanimous_seven_seed_sweep():
    # 7/7 positive: exact two-sided p = 2/128 = 0.015625. Deleting one
    # seed leaves 6/6 positive, p = 2/64 = 0.03125, still below alpha, so
    # neither the direction nor the primary decision flips.
    d = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.30]
    r = M.leave_one_out_sensitivity(d)
    assert r['n_pairs'] == 7 and r['n_dropped'] == 7
    assert r['status'] == 'stable'
    assert r['fragile'] is False
    assert r['direction_fragile'] is False
    assert r['decision_sensitive'] is False
    assert r['sign_stable'] is True
    assert r['primary_significance_stable'] is True
    assert r['n_sign_flips'] == 0
    assert r['n_primary_significance_flips'] == 0
    assert math.isclose(r['baseline']['permutation_p_value'],
                        0.015625, rel_tol=1e-15)
    # Dropping the largest value (0.30) moves the mean the most.
    assert r['most_influential_index'] == 6
    assert r['most_influential_value'] == 0.30
    assert math.isclose(r['max_abs_mean_shift'], 0.025, rel_tol=1e-12)
    assert math.isclose(r['drops'][6]['mean_diff'], 0.75 / 6, rel_tol=1e-12)
    assert math.isclose(r['drops'][6]['mean_shift'], -0.025, rel_tol=1e-12)
    assert math.isclose(r['max_relative_mean_shift'],
                        0.025 / 0.15, rel_tol=1e-12)
    assert r['min_mean_diff'] < r['max_mean_diff']
    json.dumps(r, allow_nan=False)


def test_leave_one_out_flags_an_exact_test_boundary():
    # 6/6 positive: p = 0.03125 clears alpha, but every 5/5 deletion has
    # p = 0.0625 and never can clear it. The direction is stable; the
    # decision is at the exact test's resolution boundary.
    d = [0.10, 0.12, 0.14, 0.11, 0.13, 0.15]
    r = M.leave_one_out_sensitivity(d)
    assert r['status'] == 'boundary_sensitive'
    assert r['fragile'] is True
    assert r['direction_fragile'] is False
    assert r['decision_sensitive'] is True
    assert r['sign_stable'] is True
    assert r['primary_significance_stable'] is False
    assert r['n_sign_flips'] == 0
    assert r['n_primary_significance_flips'] == 6
    assert all(math.isclose(x['permutation_p_value'], 0.0625,
                            rel_tol=1e-15) for x in r['drops'])
    json.dumps(r, allow_nan=False)


def test_leave_one_out_detects_a_direction_flip():
    # Six tiny positive differences and one huge negative one: the full
    # mean is negative, but deleting the outlier makes every remaining
    # seed positive. The exact test is not significant on the full set
    # (the outlier dominates the sign-flip distribution), yet deleting it
    # manufactures a significant 6/6 result — exactly the sensitivity this
    # report must expose.
    d = [0.1] * 6 + [-2.0]
    r = M.leave_one_out_sensitivity(d)
    assert r['status'] == 'direction_fragile'
    assert r['direction_fragile'] is True
    assert r['decision_sensitive'] is False
    assert r['fragile'] is True
    assert r['n_sign_flips'] == 1
    assert r['most_influential_index'] == 6
    assert r['most_influential_value'] == -2.0
    assert r['baseline']['direction'] == 'less'
    assert r['drops'][6]['direction'] == 'greater'
    assert r['drops'][6]['sign_flip'] is True
    assert math.isclose(r['drops'][6]['mean_diff'], 0.1, rel_tol=1e-12)
    assert math.isclose(r['min_mean_diff'], (-1.5) / 6, rel_tol=1e-12)
    assert math.isclose(r['max_mean_diff'], 0.1, rel_tol=1e-12)
    # Deleting the outlier turns a non-significant full sample into a
    # significant exact 6/6 sweep; the report must count that boundary
    # flip without presenting the deleted result as the finding.
    assert r['n_primary_significance_flips'] == 1


def test_leave_one_out_no_direction_and_insufficient_data():
    flat = M.leave_one_out_sensitivity([0.1, -0.1, 0.2, -0.2, 0.0])
    assert flat['status'] == 'no_direction'
    assert flat['baseline']['mean_diff'] == 0.0
    assert flat['fragile'] is False
    assert flat['direction_fragile'] is False
    assert flat['sign_stable'] is None
    for d, n in (([], 0), ([0.5], 1), ([0.5, 0.6], 2)):
        r = M.leave_one_out_sensitivity(d)
        assert r['n_pairs'] == n
        assert r['status'] == 'insufficient_data'
        assert r['n_dropped'] == 0
        assert r['drops'] == []
        assert r['fragile'] is None
        assert r['sign_stable'] is None
        assert r['primary_significance_stable'] is None
        assert r['most_influential_index'] is None
        json.dumps(r, allow_nan=False)


def test_leave_one_out_equivalence_verdict_stability_is_separate():
    # baseline: mean 0.24, wide spread -> not equivalent at margin 0.3.
    # Deleting the 0.8 outlier leaves four identical 0.1 values (point CI)
    # -> equivalent. The direction never flips, so `status` stays stable;
    # the equivalence verdict is reported separately.
    d = [0.1, 0.1, 0.1, 0.1, 0.8]
    r = M.leave_one_out_sensitivity(d, equivalence_margin=0.3)
    assert r['status'] == 'stable'
    assert r['fragile'] is False
    assert r['equivalence_verdict_stable'] is False
    assert r['n_equivalence_verdict_flips'] == 1
    assert r['baseline']['practically_equivalent'] is False
    assert r['drops'][4]['practically_equivalent'] is True
    assert r['drops'][4]['equivalence_verdict_flip'] is True
    json.dumps(r, allow_nan=False)


def test_leave_one_out_mc_is_deterministic_and_rng_isolated():
    d = [0.05, 0.07, 0.06, 0.04, 0.05, 0.08, 0.06, 0.05,
         0.07, 0.04, 0.06, 0.05, 0.09, 0.05, 0.06, 0.04, 0.07, 0.05]
    state = random.getstate()
    first = M.leave_one_out_sensitivity(
        d, max_exact=4, n_permutations=50, seed=7)
    assert random.getstate() == state, (
        'leave_one_out_sensitivity leaked into the global RNG stream')
    second = M.leave_one_out_sensitivity(
        d, max_exact=4, n_permutations=50, seed=7)
    assert first == second
    assert first['n_dropped'] == len(d)
    assert all(x['n_pairs'] == len(d) - 1 for x in first['drops'])
    json.dumps(first, allow_nan=False)


def test_leave_one_out_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], alpha=0.0)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], alpha=1.0)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], max_exact=-1)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], max_exact=True)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], n_permutations=0)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], seed=True)
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, float('nan'), 0.3])
    for bad_margin in (0.0, -0.1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            M.leave_one_out_sensitivity([0.1, 0.2, 0.3],
                                        equivalence_margin=bad_margin)
    # The TOST equivalence CI only exists for alpha < 0.5.
    with pytest.raises(ValueError):
        M.leave_one_out_sensitivity([0.1, 0.2, 0.3], alpha=0.5,
                                    equivalence_margin=0.05)


def test_summarize_paired_seed_influence_is_opt_in_and_matches_standalone():
    d = [0.10, 0.12, 0.14, 0.11, 0.13, 0.15]
    default = M.summarize_paired(d, n_boot=100, n_permutations=100)
    assert 'seed_influence' not in default
    s = M.summarize_paired(d, n_boot=100, n_permutations=100,
                           include_seed_influence=True)
    assert s['seed_influence'] == M.leave_one_out_sensitivity(
        d, n_permutations=100)
    assert s['seed_influence']['status'] == 'boundary_sensitive'
    json.dumps(s, allow_nan=False)
    with_margin = M.summarize_paired(
        d, n_boot=100, n_permutations=100, equivalence_margin=0.5,
        include_seed_influence=True)
    assert with_margin['seed_influence']['equivalence_margin'] == 0.5


def test_pairwise_report_integrates_seed_influence_counts():
    rows = [_row('op_a', [0.9] * 6), _row('op_b', [0.1] * 6)]
    report = _pairwise_op_comparisons(rows, alpha=0.05, seed=0,
                                      n_boot=200, n_permutations=200)
    rec = report['cells']['mqar/n_kv=1']['pairs']['op_a~op_b']
    assert rec['seed_influence']['status'] == 'boundary_sensitive'
    assert report['n_direction_fragile'] == 0
    assert report['n_boundary_sensitive'] == 1
    assert report['n_seed_influence_stable'] == 0
    assert report['seed_influence']['n_boundary_sensitive'] == 1
    assert rows[0]['pairwise_direction_fragile_wins'] == []
    json.dumps(report, allow_nan=False)

    # Direction fragility on a non-significant pair is still aggregated,
    # but it must not fabricate a winning row annotation.
    rows2 = [_row('op_c', [0.6] * 6 + [-1.5]), _row('op_d', [0.5] * 7)]
    report2 = _pairwise_op_comparisons(rows2, alpha=0.05, seed=0,
                                       n_boot=200, n_permutations=200)
    rec2 = report2['cells']['mqar/n_kv=1']['pairs']['op_c~op_d']
    assert rec2['seed_influence']['status'] == 'direction_fragile'
    assert rec2['verdict'] == 'not_significant'
    assert report2['n_direction_fragile'] == 1
    assert rows2[0]['pairwise_direction_fragile_wins'] == []
    assert rows2[1]['pairwise_direction_fragile_wins'] == []



def test_permutation_test_is_scale_invariant():
    """The sign-flip test must not depend on the units of the differences."""
    for scale in (1.0, 1e-13, 1e13):
        d = [1.0 * scale, 2.0 * scale, 3.0 * scale]
        assert math.isclose(M.permutation_test(d)['p_value'], 0.25,
                            rel_tol=1e-15)
        unanimous = [0.1 * scale] * 6
        assert math.isclose(M.permutation_test(unanimous)['p_value'],
                            0.03125, rel_tol=1e-15)
    # Exact ties are still counted at a large scale.
    assert M.permutation_test([1e300, -1e300])['p_value'] == 1.0


def test_sample_sd_survives_finite_large_differences():
    sd = M._sample_sd([1e200, 3e200], 2e200)
    assert sd == pytest.approx(2e200 / math.sqrt(2))
    t = M.paired_t_test([1e200, 3e200])
    assert t['t_stat'] is not None and math.isfinite(t['p_value'])


def test_seeds_required_never_recommends_an_unrunnable_design():
    assert M.seeds_required(3.0) == 2
    assert M.seeds_required(1e200) == 2
    assert M.seeds_required(1e-200) is None
    s = M.summarize_paired([10.0, 11.0, 12.0], n_boot=50, seed=0)
    assert s['seeds_required_80pct_power'] == 2


def test_holm_adjust_rejects_bool_p_values():
    with pytest.raises(ValueError):
        M.holm_adjust([False, 0.5])
    with pytest.raises(ValueError):
        M.holm_adjust([True])


def test_permutation_tie_epsilon_does_not_swallow_the_observed_effect():
    """An outlier pair must not make every sign flip count as a tie.

    The tie tolerance used to scale with ``max|d_i|``; for
    ``[1e12, -1e12+1, 2]`` that epsilon exceeded the observed mean and
    the exact two-sided p became 1.0 instead of 0.75.
    """
    d = [1e12, -1e12 + 1, 2.0]
    exact = M.permutation_test(d)
    assert exact['method'] == 'exact' and exact['n_permutations'] == 8
    # Independent enumeration with the documented strict tail.
    n = len(d)
    observed = sum(d) / n
    hits = 0
    for mask in range(2 ** n):
        stat = sum(d[i] if (mask >> i) & 1 else -d[i]
                   for i in range(n)) / n
        if abs(stat) >= abs(observed):
            hits += 1
    assert exact['p_value'] == pytest.approx(hits / 2 ** n)
    assert exact['p_value'] == pytest.approx(0.75)
    # The scale-invariance contract still holds at both extremes.
    assert M.permutation_test([x * 1e-13 for x in (1.0, 2.0, 3.0)])[
        'p_value'] == pytest.approx(0.25)
    assert M.permutation_test([1e300, -1e300])['p_value'] == 1.0


def test_finite_extreme_differences_keep_reports_json_safe():
    """Finite inputs of 1e308 must not overflow mean/median to ``inf``."""
    record = M.summarize_paired([1e308, 1e308], n_boot=20)
    assert record['mean_diff'] == 1e308
    assert record['median_diff'] == 1e308
    assert math.isfinite(record['mean_diff'])
    json.dumps(record, allow_nan=False)  # must not raise
    t = M.paired_t_test([1e308, 1e308])
    assert t['mean_diff'] == 1e308 and math.isfinite(t['mean_diff'])
    loo = M.leave_one_out_sensitivity([1e308, 1e308, 1e308],
                                      n_permutations=10)
    assert loo['baseline']['mean_diff'] == 1e308
    assert math.isfinite(loo['baseline']['mean_diff'])
    assert all(math.isfinite(d['mean_diff']) for d in loo['drops'])
    json.dumps(loo, allow_nan=False)  # must not raise
    # Opposite-sign extremes can make the t half-width / TOST interval
    # unrepresentable. Those fields become None instead of +/-inf, and a
    # sample SD beyond the float range is reported as undefined rather
    # than raising from tost_paired().
    for diffs in ([1e308, -1e308], [1.5e308, -1.5e308]):
        rep = M.summarize_paired(diffs, n_boot=20,
                                 equivalence_margin=1.0)
        assert rep['ci95_mean_diff_t'] == [None, None]
        assert rep['tost']['ci_equiv_lo'] is None
        assert rep['tost']['ci_equiv_hi'] is None
        json.dumps(rep, allow_nan=False)
    assert M.tost_paired([1.5e308, -1.5e308], margin=1.0)[
        'p_value'] is None
    # A normal-range sample keeps the exact legacy arithmetic.
    assert M.summarize_paired([1.0, 2.0, 3.0, 4.0],
                              n_boot=10)['median_diff'] == 2.5
