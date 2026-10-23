"""Dependency-free multi-seed inference toolkit for paired comparisons.

Motivation
----------
Every quality / ablation experiment in this repository trains each operator
over a small number of seeds (5-7 by default) and reports means, CIs and
t-tests. Those numbers answer "is the operator above chance?" but they do
NOT answer the question readers actually care about: *is seed-paired
operator A better than operator B, and would that ordering reproduce?*
The standard mistakes this module exists to prevent:

* comparing two operators via **overlapping confidence intervals** — a
  heuristic that is neither necessary nor sufficient for a difference
  (two CIs can overlap while the paired difference is highly significant,
  and disjoint CIs are only one of many sufficient conditions);
* relying on a **single parametric test** whose p-value silently breaks
  when one seed diverges or the per-seed differences are skewed (the
  standard t-test has no error control against a single outlier seed);
* treating **uncorrected p < 0.05** as a finding while scanning all
  operator pairs — the multiple-comparison problem;
* reporting a "significant" result from **n = 2-3 seeds** without any
  indication of the realized power (the same effect can flip sign on the
  next seed draw);
* interpreting a **non-significant superiority test as evidence of no
  difference** — absence of evidence is not evidence of absence. Only a
  pre-specified equivalence test (TOST) licenses a "practically the same"
  claim, and only against a margin fixed BEFORE the data were seen;
* **non-reproducible resampling statistics** — if the bootstrap /
  permutation test draws from the global RNG, re-running the experiment
  changes the reported p-values.

`multi_seed_stats` provides one deterministic, JSON-safe implementation of
the whole pipeline for a list of seed-paired differences:

1. :func:`paired_t_test` — Student t on the paired differences, two-sided
   and directional p-values, exact t CDF (regularized incomplete beta +
   continued fraction; no SciPy required).
2. :func:`sign_test` — the distribution-free exact binomial sign test
   (Steel's all-zeros convention), which is immune to scale/outliers.
3. :func:`permutation_test` — exact sign-flip permutation test (all
   ``2^n`` sign assignments) for small seed counts, deterministic
   Monte-Carlo for larger ones. This is the exact paired permutation test
   under the symmetric null and needs no distributional assumption at all.
4. :func:`bootstrap_paired_ci` — percentile bootstrap CI for the mean
   paired difference, drawn from a *local* ``random.Random`` instance so
   the result is reproducible to the bit and never perturbs the global
   RNG stream (important: resampling must not change the training seeds).
5. :func:`holm_adjust` — Holm-Bonferroni step-down FWER control over the
   family of pairwise tests. Uniformly more powerful than plain
   Bonferroni and just as safe, which matters at the 5-7-seed scale where
   every extra rejectable pair is informative.
6. :func:`seeds_required` / :func:`min_detectable_effect` — the planning
   side of multi-seed evaluation: with the observed standardized effect
   from this run, how many seeds would the next run need for 80% power?
   (Normal approximation, documented as such.)
7. :func:`tost_paired` — the two one-sided paired t-tests (TOST) for
   PRACTICAL EQUIVALENCE against a pre-specified margin. This is the only
   test here that can positively support "A and B are the same, for
   practical purposes"; a non-significant superiority test can never do
   that. The margin must be fixed by the analysis plan, never chosen after
   seeing the data.
8. :func:`equivalence_margin_supported` — the planning companion: given
   the observed per-seed differences, the smallest margin at which a TOST
   would declare equivalence. It makes "inconclusive at this seed budget"
   actionable (widen the margin you can defend or run more seeds).
9. :func:`summarize_paired` — one call that runs the whole panel and
   returns a single JSON-safe record for the results file.
10. :func:`leave_one_out_sensitivity` — the post-hoc robustness companion:
    delete each seed in turn and report whether the direction or the
    primary significance decision depends on that single seed. It is
    explicitly descriptive (not a test, no multiplicity correction) and
    cannot upgrade a claim; it prevents a one-seed-driven ordering from
    being read as a structural result.
11. :func:`exact_test_resolution` — the discreteness check that must be
    read BEFORE a non-significant exact test: an exact sign/sign-flip
    test has only ``2**n`` sign assignments, so at ``n = 5`` the smallest
    attainable two-sided p is 0.0625 and a Holm-corrected family of 21
    comparisons can never reject at alpha=0.05.  The function returns the
    attainable p floor, the minimum seed count that would give the family
    a rejection region, and the largest family the current n can support.
12. :func:`binomial_proportion_ci` — the exact Clopper-Pearson interval
    for the seed win rate, so a 5/5 sweep is reported as ``[0.478, 1]``
    rather than as a confident "100% win rate".
13. :func:`median_order_statistic_ci` — the distribution-free median
    interval from the binomial order statistics.  At n=5 no 95% interval
    exists; the function says so (minimum n=6) instead of returning a
    spuriously narrow interval.

Design constraints
------------------
* **Stdlib only** (``math``, ``random``). The repository must run in
  minimal Kaggle/CPU environments; SciPy is optional everywhere else and
  the existing runners already carry a pure-Python t-CDF fallback.
* **Deterministic.** Bootstrap / permutation randomness is isolated in a
  local ``random.Random(seed)``; nothing here touches the global RNG.
* **JSON-safe.** Every returned value is int / float / bool / str / None
  or a nested container of those. Non-finite inputs raise ``ValueError``
  so a diverged seed can never silently poison an aggregate.
* **Honest about n.** Every record carries ``n_pairs`` and a
  ``seeds_required_80pct_power`` planning estimate; a p-value from three
  seeds is labelled as such instead of being presented as a structural
  finding.

The exact paired t CDF here intentionally duplicates the incomplete-beta
core of ``run_quality._bonferroni_crit_q`` instead of importing it: that
function solves the *inverse* problem (quantile for a corrected alpha),
while this module needs the forward CDF, and keeping this file import-free
makes it usable from tests without pulling in torch / the experiment
runner. The two implementations agree to <1e-9 on the shared domain.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction
from statistics import NormalDist

__all__ = [
    'betainc_regularized',
    'student_t_cdf',
    'student_t_sf',
    'paired_t_test',
    'sign_test',
    'permutation_test',
    'bootstrap_paired_ci',
    'cohens_dz',
    'tost_paired',
    'tost_from_summary',
    'equivalence_margin_supported',
    'holm_adjust',
    'min_detectable_effect',
    'power_at_effect',
    'seeds_required',
    'summarize_paired',
    'leave_one_out_sensitivity',
    'exact_test_resolution',
    'binomial_proportion_ci',
    'median_order_statistic_ci',
]

# ---------------------------------------------------------------------------
# Student-t distribution (forward CDF / survival function)
# ---------------------------------------------------------------------------

_MAX_BETACF_ITER = 200
_BETACF_EPS = 3.0e-14
_FPMIN = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (NR 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_BETACF_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            break
    return h


def betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` for ``a, b > 0``."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f'betainc requires a,b > 0, got a={a!r}, b={b!r}')
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, dof: int) -> float:
    """CDF of Student's t with ``dof > 0`` degrees of freedom."""
    if dof <= 0:
        raise ValueError(f'student_t_cdf requires dof > 0, got {dof!r}')
    if not math.isfinite(t_value):
        return 1.0 if t_value > 0 else 0.0
    if t_value == 0.0:
        return 0.5
    x = dof / (dof + t_value * t_value)
    ib = betainc_regularized(0.5 * dof, 0.5, x)
    if t_value > 0.0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_sf(t_value: float, dof: int) -> float:
    """Survival function ``P(T > t)`` of Student's t.

    Computed from the same incomplete-beta tail as the CDF rather than as
    ``1 - cdf``: the subtraction cancels catastrophically in the upper
    tail, where it used to return exactly 0.0 for every p-value below
    machine epsilon (e.g. t=100, dof=5 should be 9.48e-10). P-values must
    stay positive however extreme the evidence is.
    """
    if dof <= 0:
        raise ValueError(f'student_t_sf requires dof > 0, got {dof!r}')
    if not math.isfinite(t_value):
        return 0.0 if t_value > 0 else 1.0
    if t_value == 0.0:
        return 0.5
    x = dof / (dof + t_value * t_value)
    ib = betainc_regularized(0.5 * dof, 0.5, x)
    if t_value > 0.0:
        return 0.5 * ib
    return 1.0 - 0.5 * ib


# ---------------------------------------------------------------------------
# Paired summaries
# ---------------------------------------------------------------------------

def _as_clean_diffs(diffs) -> list[float]:
    """Validate a paired-difference array and return a plain list.

    Non-finite values are rejected loudly: a diverged seed must be either
    excluded by the caller (with the exclusion recorded) or fail the run,
    never silently turn a p-value into NaN.
    """
    out = []
    for i, d in enumerate(diffs):
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            raise ValueError(
                f'diffs[{i}]={d!r} is not a real number; filter diverged '
                f'seeds before calling multi_seed_stats')
        try:
            d = float(d)
        except OverflowError:
            # A JSON integer can be larger than the float range. It is a
            # malformed/non-representable difference, not a usable seed.
            raise ValueError(
                f'diffs[{i}]={d!r} is not representable as a finite float; '
                f'filter diverged seeds before calling multi_seed_stats')
        if not math.isfinite(d):
            raise ValueError(
                f'diffs[{i}]={d!r} is not finite; filter diverged seeds '
                f'before calling multi_seed_stats')
        out.append(d)
    return out


def _safe_mean(values) -> float | None:
    """Mean of finite values without overflowing the intermediate sum.

    ``sum([1e308, 1e308]) / 2`` is ``inf / 2`` even though both inputs and
    the true mean are finite. The ordinary path keeps the exact previous
    arithmetic for every in-range sample; only an overflow falls back to a
    scaled sum, which is value-equal up to ordinary floating-point
    round-off. A result that is still non-finite is a programming error in
    this module's "JSON-safe" contract and is rejected loudly rather than
    inserted into a report.
    """
    n = len(values)
    if n == 0:
        return None
    total = sum(values)
    if math.isfinite(total):
        return total / n
    scale = max(abs(x) for x in values)
    if scale == 0.0:
        return 0.0
    mean = scale * (math.fsum(x / scale for x in values) / n)
    if not math.isfinite(mean):
        raise ValueError(
            f'mean is not finite after scaled summation (scale={scale!r}); '
            'the inputs exceed the representable range of their mean')
    return mean


def _safe_median(values):
    """Median whose even-sample midpoint cannot overflow to ``inf``.

    ``(a + b) / 2`` overflows for two finite large samples; ``a/2 + b/2``
    is the same midpoint without that overflow.
    """
    n = len(values)
    if n == 0:
        return None
    ordered = sorted(values)
    if n % 2:
        return ordered[n // 2]
    a, b = ordered[n // 2 - 1], ordered[n // 2]
    mid = (a + b) / 2.0
    if math.isfinite(mid):
        return mid
    # Same midpoint without the overflowing intermediate sum. This branch
    # is reached only for two same-sign finite values near the float max.
    mid = a / 2.0 + b / 2.0
    if not math.isfinite(mid):
        raise ValueError(
            'median is not finite after overflow-safe midpoint')
    return mid


# ---------------------------------------------------------------------------
# Exact-test resolution and nonparametric effect-size uncertainty
# ---------------------------------------------------------------------------

def _validate_alpha(alpha) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    try:
        alpha = float(alpha)
    except OverflowError:
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    return alpha


def _is_finite_real(value) -> bool:
    """True for a non-bool real representable as a finite float.

    JSON integers are unbounded; a value too large for ``float()`` is a
    malformed numeric input, not a reason to leak ``OverflowError``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _beta_quantile(q: float, a: float, b: float) -> float:
    """Quantile of the Beta(a, b) distribution by forward-CDF bisection.

    ``betainc_regularized`` is monotone in x, so bisection is robust for
    the small integer shape parameters used by Clopper-Pearson. The cost
    (a few hundred incomplete-beta evaluations) is negligible for the
    n = 5-7 seed scale this module targets.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f'q must be in [0,1], got {q!r}')
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if betainc_regularized(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binomial_proportion_ci(k, n, *, alpha: float = 0.05) -> dict:
    """Exact Clopper-Pearson interval for a binomial proportion ``k/n``.

    The seed win rate is a binomial proportion over the ``n_effective``
    non-tied pairs. A bare ``5/5 = 100%`` hides that five seeds are also
    consistent with a true win probability as low as 0.478; the exact
    interval makes that uncertainty explicit. ``n=0`` returns undefined
    endpoints (``None``) instead of fabricating [0, 1].

    The interval is conservative by construction (its coverage is at least
    ``1 - alpha``), and ties are excluded by the caller, exactly as in
    :func:`sign_test`.
    """
    alpha = _validate_alpha(alpha)
    if (isinstance(n, bool) or not isinstance(n, int) or n < 0):
        raise ValueError(f'n must be a non-negative int, got {n!r}')
    if (isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > n):
        raise ValueError(f'k must be an int in [0, n], got {k!r} (n={n!r})')
    out = {
        'k': int(k), 'n': int(n), 'alpha': alpha,
        'ci_level': 1.0 - alpha,
        'ci_low': None, 'ci_high': None,
        'method': 'clopper_pearson_exact',
        'note': ('Exact binomial interval over non-tied seed pairs; '
                 'conservative (coverage >= ci_level).'),
    }
    if n == 0:
        return out
    low = 0.0 if k == 0 else _beta_quantile(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else _beta_quantile(
        1.0 - alpha / 2.0, k + 1, n - k)
    # Guard against a last-ulp inversion on extreme shapes.
    low = min(max(low, 0.0), 1.0)
    high = min(max(high, 0.0), 1.0)
    if low > high:
        low, high = high, low
    out['ci_low'] = low
    out['ci_high'] = high
    return out


def median_order_statistic_ci(diffs, *, alpha: float = 0.05) -> dict:
    """Distribution-free confidence interval for the paired median.

    For ``n`` independent paired differences, the interval
    ``[x_(k), x_(n-k+1)]`` (1-based order statistics) covers the population
    median with probability
    ``1 - 2 * P(Binomial(n, 0.5) <= k - 1)``. Pick the largest ``k`` whose
    lower tail is at most ``alpha/2``; that gives the shortest such
    interval. The construction assumes only exchangeability/continuity and
    is robust to a single diverging seed.

    At ``n = 5`` even the full range ``[x_(1), x_(5)]`` has only 93.75%
    coverage, so no 95% interval exists; the function returns ``None``
    endpoints and ``min_pairs_for_ci = 6`` rather than pretending. Ties
    make the interval conservative.
    """
    alpha = _validate_alpha(alpha)
    d = _as_clean_diffs(diffs)
    n = len(d)
    min_pairs = max(1, int(math.ceil(
        math.log2(2.0) - math.log2(alpha))))
    out = {
        'n_pairs': n, 'alpha': alpha, 'ci_level': 1.0 - alpha,
        'k': None, 'ci_low': None, 'ci_high': None,
        'coverage': None, 'min_pairs_for_ci': min_pairs,
        'method': 'binomial_order_statistics',
        'note': ('Distribution-free interval for the population median. '
                 'The largest admissible k with P(Bin(n,.5)<=k-1) <= '
                 'alpha/2 is used; ties are conservative.'),
    }
    if n == 0:
        return out
    # P(X <= k-1) = I_{0.5}(n-k+1, k), monotone in k. Use the
    # incomplete-beta form (no 2**n) and binary-search the largest
    # admissible k, so the routine stays stable for large n as well.
    def _tail(k: int) -> float:
        return betainc_regularized(n - k + 1, k, 0.5)

    k_best = 0
    tail_best = None
    lo, hi = 1, (n + 1) // 2
    # A relative (not absolute) slack absorbs the last-ulp error of the
    # incomplete-beta tail without letting a tiny alpha be dominated by a
    # fixed 1e-15 epsilon: for alpha below that scale the old check accepted
    # every tail computed as <= 1e-15 and returned an interval whose true
    # coverage was far below 1 - alpha.
    threshold = alpha / 2.0
    slack = threshold * 1e-12
    while lo <= hi:
        mid = (lo + hi) // 2
        tail = _tail(mid)
        if tail <= threshold + slack:
            k_best, tail_best = mid, tail
            lo = mid + 1
        else:
            hi = mid - 1
    if k_best == 0:
        return out
    ordered = sorted(d)
    out['k'] = int(k_best)
    out['ci_low'] = ordered[k_best - 1]
    out['ci_high'] = ordered[n - k_best]
    out['coverage'] = 1.0 - 2.0 * tail_best
    return out


def exact_test_resolution(n_pairs, *, alpha: float = 0.05,
                          family_size: int = 1,
                          alternative: str = 'two-sided',
                          n_zero: int = 0) -> dict:
    """Rejection-region floor for the exact sign / sign-flip test.

    An exact paired sign-flip (or sign) test has only ``2**n_effective``
    sign assignments. Its smallest attainable two-sided p is therefore
    ``2 / 2**n_effective`` (``1 / 2**n_effective`` for a directional test),
    and a Holm family of ``m`` comparisons has an adjusted floor of
    ``m`` times that value. When the floor exceeds ``alpha`` the family has
    **no rejection region at all**: a non-significant result is forced by
    the design, not evidence of no effect. This is the small-n discreteness
    problem that ordinary power calculations (which use continuous normal
    or t distributions) do not see.

    ``n_zero`` is the number of exactly tied pairs. Ties carry no sign
    information, so the effective count is ``n_pairs - n_zero``; the floor
    uses that count exactly as :func:`sign_test` does.

    Returns a JSON-safe record with:

    * ``min_attainable_p_raw`` / ``min_attainable_p_holm`` — p-value
      floors before/after the family correction (capped at 1);
    * ``raw_testable`` / ``family_testable`` — whether the comparison
      could reject on its own, and whether it could be the *first*
      (smallest-p) rejection in the Holm family. When every family member
      has the same n (the usual complete-seed-block case), the latter is
      exactly whether the family has any rejection region;
    * ``later_rank_possible`` — the raw floor is at or below ``alpha``, so
      a later-rank rejection is not ruled out even when this comparison
      could not be the first one. It is necessary, not sufficient;
    * ``min_effective_pairs_raw`` / ``min_effective_pairs_holm`` — the
      smallest effective n that would restore a rejection region;
    * ``shortfall_pairs_raw`` / ``shortfall_pairs_holm`` — how many
      additional informative pairs the current run needs;
    * ``max_comparisons_at_alpha`` — the largest family this n can
      support (0 means not even one comparison is testable).

    This is a **necessary** condition, not a power guarantee: a family can
    be testable and still be badly underpowered at the SESOI.
    """
    alpha = _validate_alpha(alpha)
    if (isinstance(n_pairs, bool) or not isinstance(n_pairs, int)
            or n_pairs < 0):
        raise ValueError(
            f'n_pairs must be a non-negative int, got {n_pairs!r}')
    if (isinstance(n_zero, bool) or not isinstance(n_zero, int)
            or n_zero < 0 or n_zero > n_pairs):
        raise ValueError(
            f'n_zero must be an int in [0, n_pairs], got {n_zero!r} '
            f'(n_pairs={n_pairs!r})')
    if (isinstance(family_size, bool) or not isinstance(family_size, int)
            or family_size < 1):
        raise ValueError(
            f'family_size must be a positive int, got {family_size!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f'got {alternative!r}')
    n_eff = n_pairs - n_zero
    sided = 2 if alternative == 'two-sided' else 1
    if n_eff == 0:
        p_min = 1.0
    elif n_eff <= 1023:
        # Exact power-of-two arithmetic for every realistic seed count;
        # the float form avoids the last-ulp noise of exp(log(...)).
        p_min = min(1.0, sided / float(1 << n_eff))
    else:
        # exp(log) keeps the floor finite for n far beyond 2**1024.
        log_floor = math.log(sided) - n_eff * math.log(2.0)
        p_min = 1.0 if log_floor >= 0.0 else math.exp(log_floor)
    if p_min <= 0.0:
        # The exact floor ``sided / 2**n_eff`` underflowed to 0.0 for a very
        # large n_eff, but ``family_size * floor`` can still be a perfectly
        # representable p-value (e.g. n_eff=2000 with an enormous family).
        # Evaluate that product in log space instead of reporting a p of
        # exactly zero: the floor is small, not zero.
        log_product = (math.log(family_size) + math.log(sided)
                       - n_eff * math.log(2.0))
        p_holm = 1.0 if log_product >= 0.0 else math.exp(log_product)
    else:
        # ``family_size`` is an arbitrary Python int: converting a very large
        # one to float (implicitly, in ``family_size * p_min``) raises
        # OverflowError. Fall back to log-space only for that case, so the
        # ordinary integral products keep their exact float arithmetic.
        try:
            p_holm = min(1.0, family_size * p_min)
        except OverflowError:
            log_product = math.log(family_size) + math.log(p_min)
            p_holm = 1.0 if log_product >= 0.0 else math.exp(log_product)

    alpha_frac = Fraction.from_float(alpha)

    def _min_effective(multiplier: int) -> int:
        # Smallest n >= 1 with multiplier * sided / 2**n <= alpha.
        # Compare exact rationals instead of logs: when the boundary is
        # exactly attainable (e.g. alpha=0.5, family=2 -> 4/2**n <= 0.5 at
        # n=3), floating-point log/ceil can round 3.0 up to 4 and overstate
        # the required seed budget by one.
        need = multiplier * sided * alpha_frac.denominator
        num = alpha_frac.numerator
        if num >= need:
            return 0
        # Bit lengths give log2(need/num) up to one; correct exactly below.
        n = max(0, need.bit_length() - num.bit_length())
        while n > 0 and num * (1 << (n - 1)) >= need:
            n -= 1
        while num * (1 << n) < need:
            n += 1
        return max(1, n)

    min_raw = _min_effective(1)
    min_holm = _min_effective(family_size)
    if p_min <= 0.0:
        # The floor underflowed, so ``alpha / p_min`` would be infinite and
        # returning None would hide a well-defined (if enormous) answer. The
        # largest supported family is floor(alpha * 2**n_eff / sided); build
        # it with integer/Fraction arithmetic for every practical n_eff and
        # refuse only a shift so large that the integer itself is absurd.
        if n_eff <= 1_000_000:
            max_family = int(
                (alpha_frac * (1 << n_eff) / sided).__floor__())
        else:
            max_family = None
    else:
        ratio = alpha / p_min
        if math.isfinite(ratio):
            max_family = int(math.floor(ratio))
        else:
            # ``alpha / p_min`` overflowed for a tiny (but positive) floor.
            # The true quotient is still an ordinary (if enormous) integer;
            # compute its floor exactly from the float representations rather
            # than returning None or crashing in int(inf).
            max_family = int(
                (alpha_frac / Fraction.from_float(p_min)).__floor__())
    return {
        'n_pairs': n_pairs,
        'n_zero': n_zero,
        'n_effective': n_eff,
        'alternative': alternative,
        'sided': sided,
        'alpha': alpha,
        'family_size': family_size,
        'min_attainable_p_raw': p_min,
        'min_attainable_p_holm': p_holm,
        'raw_testable': bool(p_min <= alpha),
        # ``family_testable`` is the first-rank (smallest-p) condition.
        # With equal n across the family it is exactly the family-level
        # feasibility; with unequal n a later-ranked comparison can still
        # reject, which ``later_rank_possible`` records as a necessary
        # (not sufficient) condition.
        'family_testable': bool(p_holm <= alpha),
        'first_rejection_testable': bool(p_holm <= alpha),
        'later_rank_possible': bool(p_min <= alpha),
        'min_effective_pairs_raw': min_raw,
        'min_effective_pairs_holm': min_holm,
        'shortfall_pairs_raw': max(0, min_raw - n_eff),
        'shortfall_pairs_holm': max(0, min_holm - n_eff),
        'max_comparisons_at_alpha': max_family,
        'method': 'exact_sign_flip_resolution',
        'note': ('Exact-test discreteness floor, a NECESSARY condition for '
                 'rejection. family_testable=False means this comparison '
                 'cannot be the first (smallest-p) rejection at alpha; with '
                 'equal n across the family, no outcome of the family could '
                 'have been significant. later_rank_possible records the '
                 'weaker condition that its own p floor is <= alpha. The '
                 'floor is not evidence of equivalence or of a null effect, '
                 'and does not apply to the continuous t-test.'),
    }


# Numerical-zero rule: a difference vector whose entries are all the same
# float has exactly zero sample variance. Values that differ only by a few
# ULPs of their own magnitude can also be a rounding residue (e.g. per-seed
# accuracy differences computed from inputs that are mathematically equal),
# so those are treated as numerically constant too. The threshold is based
# on the ULP of the DIFFERENCES, not on ``eps * max|value|``: an accuracy
# residue of a few 1e-17 is zeroed, while a real spread of 1 at 1e15 (8 ULP
# of that magnitude) is not, so the paired t-test / d_z / TOST stay defined
# for large-offset data.
_ZERO_VAR_ULP_FACTOR = 8.0


def _sample_sd(values: list[float], mean: float) -> float | None:
    n = len(values)
    if n < 2:
        return None
    # Constant vectors (including values like 0.8 / 0.1 with no exact binary
    # representation) are the only exact-zero case.
    if all(v == values[0] for v in values):
        return 0.0
    max_abs = max(abs(v) for v in values)
    if max_abs > 0.0:
        span = max(values) - min(values)
        # Strict ``<`` keeps a spread of exactly 8 ULP (e.g. 1 at 1e15)
        # distinct; only a strictly sub-8-ULP spread is last-bit residue.
        if span < _ZERO_VAR_ULP_FACTOR * math.ulp(max_abs):
            return 0.0
    # Square SCALED deviations: ``(v - mean) ** 2`` overflows for finite
    # differences above ~1e154 even though the sample SD itself is
    # representable. Scaling is exact for the ratio, then the square root
    # is multiplied back, which is value-equal up to round-off for the
    # ordinary range.
    deviations = [v - mean for v in values]
    if all(math.isfinite(x) for x in deviations):
        dev_scale = max(abs(x) for x in deviations)
        if dev_scale == 0.0:
            return 0.0
        var_ratio = math.fsum((x / dev_scale) ** 2
                              for x in deviations) / (n - 1)
        sd = dev_scale * math.sqrt(var_ratio) if var_ratio > 0.0 else 0.0
    else:
        # ``v - mean`` itself overflowed (opposite-sign values near the
        # float limit). Evaluate the same SD on values divided by a common
        # scale so the intermediate sums stay finite.
        value_scale = max(abs(v) for v in values)
        if value_scale == 0.0:
            return 0.0
        scaled = [v / value_scale for v in values]
        scaled_mean = _safe_mean(scaled)
        scaled_dev = [v - scaled_mean for v in scaled]
        dev_scale = max(abs(x) for x in scaled_dev)
        if dev_scale == 0.0:
            return 0.0
        var_ratio = math.fsum((x / dev_scale) ** 2
                              for x in scaled_dev) / (n - 1)
        sd = (value_scale * dev_scale * math.sqrt(var_ratio)
              if var_ratio > 0.0 else 0.0)
    if not math.isfinite(sd):
        # The true SD exceeds the float range; ``None`` is the module's
        # honest "not representable" value (an ``inf`` would poison every
        # strict-JSON report).
        return None
    return sd


def paired_t_test(diffs, alternative: str = 'two-sided') -> dict:
    """Student t-test on seed-paired differences ``d_i = A_i - B_i``.

    ``alternative='greater'`` tests ``H1: mean(d) > 0`` (A beats B). The
    p-value uses the exact forward t CDF implemented above, so no SciPy
    dependency is required.

    Returns ``p_value=None`` when the test is undefined (n < 2, or zero
    variance with a non-zero mean, where t is infinite/undefined). For the
    zero-variance case ``all_positive`` / ``all_negative`` record whether
    every seed difference agreed in sign, so callers can still treat a
    unanimous sweep as evidence via the sign/permutation tests.
    """
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n == 0:
        return {'n_pairs': 0, 'mean_diff': None, 'sd_diff': None,
                't_stat': None, 'dof': None, 'p_value': None,
                'all_positive': False, 'all_negative': False,
                'alternative': alternative}
    mean = _safe_mean(d)
    if n < 2:
        return {'n_pairs': n, 'mean_diff': mean, 'sd_diff': None,
                't_stat': None, 'dof': None, 'p_value': None,
                'all_positive': all(x > 0 for x in d),
                'all_negative': all(x < 0 for x in d),
                'alternative': alternative}
    sd = _sample_sd(d, mean)
    all_pos = all(x > 0 for x in d)
    all_neg = all(x < 0 for x in d)
    if sd is None or sd <= 0.0:
        return {'n_pairs': n, 'mean_diff': mean, 'sd_diff': sd,
                't_stat': None, 'dof': n - 1, 'p_value': None,
                'all_positive': all_pos, 'all_negative': all_neg,
                'alternative': alternative}
    t_stat = mean / (sd / math.sqrt(n))
    dof = n - 1
    if alternative == 'two-sided':
        p = 2.0 * student_t_sf(abs(t_stat), dof)
    elif alternative == 'greater':
        p = student_t_sf(t_stat, dof)
    else:
        p = student_t_cdf(t_stat, dof)
    p = min(1.0, max(0.0, p))
    return {'n_pairs': n, 'mean_diff': mean, 'sd_diff': sd,
            't_stat': t_stat, 'dof': dof, 'p_value': p,
            'all_positive': all_pos, 'all_negative': all_neg,
            'alternative': alternative}


def sign_test(diffs, *, alpha: float = 0.05) -> dict:
    """Exact two-sided binomial sign test on paired differences.

    Zeros are dropped (Steel's convention): a seed that produced *exactly*
    equal accuracy for both operators carries no directional information.
    The p-value is the exact ``2 * P(X >= max(n+, n-))`` for
    ``X ~ Binomial(n+, n-, 0.5)``, capped at 1.0. Distribution-free and
    robust to a single divergent magnitude; it is the exact paired
    permutation test restricted to sign information.

    The record also carries the observed seed win rate over the
    ``n_effective`` non-tied pairs and its exact Clopper-Pearson interval
    (``win_rate_ci95``). A 5/5 sweep is reported as ``[0.478, 1]``, not as
    a confident 100% win probability.
    """
    alpha = _validate_alpha(alpha)
    d = _as_clean_diffs(diffs)
    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    n_zero = len(d) - n_pos - n_neg
    n_eff = n_pos + n_neg
    if n_eff == 0:
        p = 1.0
        win_rate = None
    else:
        k = max(n_pos, n_neg)
        tail = sum(math.comb(n_eff, i) for i in range(k, n_eff + 1))
        if n_eff <= 1023:
            p = min(1.0, 2.0 * tail / float(2 ** n_eff))
        else:
            # ``float(2**n_eff)`` overflows for more than 1023 informative
            # pairs. The tail sum is an exact integer, so evaluate the same
            # ratio in log space; the exact-test resolution helper uses the
            # identical large-n branch.
            log_p = (math.log(2.0) + math.log(tail)
                     - n_eff * math.log(2.0))
            p = 1.0 if log_p >= 0.0 else math.exp(log_p)
        win_rate = n_pos / float(n_eff)
    ci = binomial_proportion_ci(n_pos, n_eff, alpha=alpha)
    return {'n_pairs': len(d), 'n_positive': n_pos, 'n_negative': n_neg,
            'n_zero': n_zero, 'n_effective': n_eff, 'p_value': p,
            'win_rate': win_rate,
            'win_rate_ci95': [ci['ci_low'], ci['ci_high']],
            'win_rate_ci_method': ci['method'],
            'win_rate_alpha': alpha}


def permutation_test(diffs, alternative: str = 'two-sided',
                     n_permutations: int = 10000, seed: int = 0,
                     max_exact: int = 16) -> dict:
    """Seed-paired sign-flip permutation test.

    Under the null that the per-seed differences are symmetric around
    zero, the signs are exchangeable; the test statistic is the mean of
    the signed differences. For ``n <= max_exact`` every ``2^n`` sign
    assignment is enumerated (exact test). For larger n a deterministic
    Monte-Carlo sample is drawn from a *local* RNG seeded with ``seed``
    (the global RNG stream is never touched), with the standard +1
    correction so the p-value can never be 0.

    This is the assumption-lightest test in the panel: no normality, no
    scale assumption, exact for small n. Ties (``|stat| >= |observed|``)
    count toward the tail, which keeps the test conservative.
    """
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    if (isinstance(n_permutations, bool)
            or not isinstance(n_permutations, int) or n_permutations < 1):
        raise ValueError(
            f'n_permutations must be a positive int, got '
            f'{n_permutations!r}')
    if (isinstance(max_exact, bool)
            or not isinstance(max_exact, int) or max_exact < 0):
        raise ValueError(
            f'max_exact must be a non-negative int, got {max_exact!r}')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f'seed must be an int, got {seed!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n == 0:
        return {'n_pairs': 0, 'method': 'exact', 'n_permutations': 1,
                'p_value': None, 'alternative': alternative}
    observed = _safe_mean(d)
    # The sign-flip test is scale-invariant, so the tie tolerance must be
    # relative to the two statistics being compared, never to the largest
    # raw difference. The old data-scale epsilon (1e-12 * max|d|) exceeded
    # the observed mean for an outlier pair that nearly cancelled
    # (d=[1e12, -1e12+1, 2]) and made every sign assignment count as a
    # tie (p=1 instead of the exact 0.75). ``math.isclose`` only bridges
    # last-ulp differences between two near-equal statistics, so exact
    # ties still count at any unit scale and no below-observed statistic
    # is admitted by a data-driven epsilon.
    def _same_stat(a: float, b: float) -> bool:
        return a == b or math.isclose(
            a, b, rel_tol=1e-12, abs_tol=0.0)

    def _tail_hit(stat: float) -> bool:
        if alternative == 'two-sided':
            a, b = abs(stat), abs(observed)
            return a >= b or _same_stat(a, b)
        if alternative == 'greater':
            return stat >= observed or _same_stat(stat, observed)
        return stat <= observed or _same_stat(stat, observed)

    if n <= max_exact:
        n_perm = 2 ** n
        hits = 0
        for mask in range(n_perm):
            signed = [d[i] if (mask >> i) & 1 else -d[i]
                      for i in range(n)]
            if _tail_hit(_safe_mean(signed)):
                hits += 1
        return {'n_pairs': n, 'method': 'exact', 'n_permutations': n_perm,
                'p_value': min(1.0, hits / float(n_perm)),
                'alternative': alternative}
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_permutations):
        signed = [x if rng.random() < 0.5 else -x for x in d]
        if _tail_hit(_safe_mean(signed)):
            hits += 1
    p = min(1.0, (hits + 1) / float(n_permutations + 1))
    return {'n_pairs': n, 'method': 'monte_carlo',
            'n_permutations': n_permutations, 'p_value': p,
            'alternative': alternative}


def bootstrap_paired_ci(diffs, alpha: float = 0.05, n_boot: int = 10000,
                        seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples the per-seed differences with replacement using a local
    ``random.Random(seed)`` (bit-reproducible; the global RNG stream is
    untouched, so enabling this CI can never change the training seeds).
    With fewer than 2 pairs the CI is undefined (``None``), never a
    misleading zero-width interval.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f'n_boot must be a positive int, got {n_boot!r}')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f'seed must be an int, got {seed!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n < 2:
        return {'n_pairs': n, 'n_boot': n_boot, 'seed': seed,
                'ci_lo': None, 'ci_hi': None,
                'ci_level': 1.0 - alpha}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [d[rng.randrange(n)] for _ in range(n)]
        means.append(_safe_mean(sample))
    means.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_boot))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    lo_idx = min(max(lo_idx, 0), n_boot - 1)
    hi_idx = min(max(hi_idx, 0), n_boot - 1)
    return {'n_pairs': n, 'n_boot': n_boot, 'seed': seed,
            'ci_lo': means[lo_idx], 'ci_hi': means[hi_idx],
            'ci_level': 1.0 - alpha}


def cohens_dz(diffs) -> float | None:
    """Cohen's ``d_z``: mean paired difference in per-seed SD units.

    This is the effect size the paired t-test is powered against; unlike
    the raw mean difference it allows power comparisons across tasks with
    different accuracy scales. Returns ``None`` when undefined (n < 2 or
    zero SD).
    """
    d = _as_clean_diffs(diffs)
    if len(d) < 2:
        return None
    mean = _safe_mean(d)
    sd = _sample_sd(d, mean)
    if sd is None or sd <= 0.0:
        return None
    return mean / sd


def tost_paired(diffs, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided paired t-tests (TOST) for practical equivalence.

    Tests ``H0: |mean(d)| >= margin`` against ``H1: |mean(d)| < margin`` on
    the seed-paired differences, at level ``alpha``. ``p_value`` is the
    maximum of the two one-sided p-values:

    * lower test: ``H0: mean(d) <= -margin`` vs ``mean(d) > -margin``;
    * upper test: ``H0: mean(d) >= +margin`` vs ``mean(d) < +margin``.

    ``equivalent`` is ``True`` iff ``p_value < alpha`` (equivalently, the
    two-sided ``(1 - 2*alpha)`` t confidence interval lies strictly inside
    ``(-margin, +margin)``; both views are returned and agree by
    construction).

    The margin is a SCIENTIFIC choice (the smallest effect that would
    matter), not a statistical one: it must be pre-specified in the
    analysis plan before the data are inspected. Choosing it from the
    observed CI would make ``p_value`` meaningless. For the same reason a
    non-significant superiority test is NOT evidence of equivalence, but
    a significant TOST is (subject to the usual paired-t assumptions).

    ``margin`` must be finite and strictly positive. Degenerate inputs
    follow the repo-wide convention: ``n < 2`` gives ``p_value=None`` /
    ``equivalent=None`` rather than a fabricated verdict; a zero-variance
    difference vector (every seed identical) has a point CI, so it is
    equivalent iff ``|mean| < margin`` and the exact p is reported as
    ``0.0`` (inside), ``0.5`` (exactly on the boundary) or ``1.0``
    (outside) instead of a NaN.

    The statistics are delegated to :func:`tost_from_summary`; runners
    that store only per-contrast summary rows (n, mean, sd) can call that
    directly, so both entry points can never drift apart.
    """
    d = _as_clean_diffs(diffs)
    n = len(d)
    mean = _safe_mean(d)
    sd = _sample_sd(d, mean) if n >= 2 else None
    if n >= 2 and sd is None:
        # The sample SD exceeds the float range (e.g. two opposite-sign
        # differences near 1.7e308). The t interval is unrepresentable, so
        # report an undefined TOST instead of raising or emitting inf.
        out = tost_from_summary(mean, 0.0, n, margin, alpha=alpha)
        out.update(sd_diff=None, t_lower=None, t_upper=None,
                   p_lower=None, p_upper=None, p_value=None,
                   equivalent=None, ci_equiv_lo=None, ci_equiv_hi=None)
        return out
    return tost_from_summary(mean, sd, n, margin, alpha=alpha)


def tost_from_summary(mean_diff, sd_diff, n_pairs: int, margin: float,
                      alpha: float = 0.05) -> dict:
    """TOST from summary statistics of the seed-paired differences.

    Same contract and result shape as :func:`tost_paired` (which delegates
    here), for callers that keep only ``n``, ``mean`` and sample ``sd`` of
    the per-seed differences rather than the raw vector (e.g. the
    ablation-runner contrast rows). ``mean_diff`` must be finite when
    ``n_pairs >= 1``; ``sd_diff`` must be finite and non-negative and is
    required when ``n_pairs >= 2`` (pass ``0.0`` for a numerically
    identical-difference vector). ``dof`` is recorded for the t branches.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError(
            f'alpha must be in (0, 0.5) for an equivalence CI to exist, '
            f'got {alpha!r}')
    try:
        margin = float(margin)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f'margin must be finite and > 0, got {margin!r}')
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(
            f'margin must be finite and > 0, got {margin!r}')
    if not (isinstance(n_pairs, int) and not isinstance(n_pairs, bool)
            and n_pairs >= 0):
        raise ValueError(
            f'n_pairs must be a non-negative int, got {n_pairs!r}')
    mean = None
    if n_pairs >= 1:
        if mean_diff is None:
            raise ValueError('mean_diff is required when n_pairs >= 1')
        try:
            mean = float(mean_diff)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                f'mean_diff must be finite, got {mean_diff!r}')
        if not math.isfinite(mean):
            raise ValueError(f'mean_diff must be finite, got {mean_diff!r}')
    sd = None
    if n_pairs >= 2:
        if sd_diff is None:
            raise ValueError('sd_diff is required when n_pairs >= 2')
        try:
            sd = float(sd_diff)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(
                f'sd_diff must be finite and >= 0, got {sd_diff!r}')
        if not math.isfinite(sd) or sd < 0.0:
            raise ValueError(
                f'sd_diff must be finite and >= 0, got {sd_diff!r}')
    out = {'n_pairs': n_pairs, 'margin': margin, 'alpha': alpha,
           'mean_diff': mean, 'sd_diff': sd, 'dof': None,
           't_lower': None, 't_upper': None,
           'p_lower': None, 'p_upper': None, 'p_value': None,
           'equivalent': None,
           'ci_equiv_level': 1.0 - 2.0 * alpha,
           'ci_equiv_lo': None, 'ci_equiv_hi': None,
           'method': 'two_one_sided_paired_t'}
    if n_pairs < 2:
        return out
    dof = n_pairs - 1
    out['dof'] = dof
    if sd <= 0.0:
        # Every per-seed difference is identical (numerically): the
        # (1-2*alpha) CI degenerates to the point [mean, mean], so the
        # verdict is decided by where that point sits relative to the
        # margin. The exact p is 0 / 0.5 / 1 at inside / boundary /
        # outside.
        if abs(mean) < margin:
            p = 0.0
        elif abs(mean) == margin:
            p = 0.5
        else:
            p = 1.0
        out.update(t_lower=None, t_upper=None,
                   p_lower=p, p_upper=p, p_value=p,
                   equivalent=bool(abs(mean) < margin),
                   ci_equiv_lo=mean, ci_equiv_hi=mean)
        return out
    se = sd / math.sqrt(n_pairs)
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = student_t_sf(t_lower, dof)
    p_upper = student_t_cdf(t_upper, dof)
    p = min(1.0, max(0.0, max(p_lower, p_upper)))
    half = _t_quantile(1.0 - alpha, dof) * se
    ci_lo, ci_hi = mean - half, mean + half
    # A finite mean/SD can still produce an interval endpoint beyond the
    # float range (half ~1e308).  ``inf`` is not JSON-serializable, so an
    # unrepresentable interval is reported as undefined rather than
    # poisoning the whole record.
    if not (math.isfinite(ci_lo) and math.isfinite(ci_hi)):
        ci_lo = ci_hi = None
    out.update(t_lower=t_lower if math.isfinite(t_lower) else None,
               t_upper=t_upper if math.isfinite(t_upper) else None,
               p_lower=p_lower, p_upper=p_upper, p_value=p,
               equivalent=bool(p < alpha),
               ci_equiv_lo=ci_lo, ci_equiv_hi=ci_hi)
    return out


def equivalence_margin_supported(diffs, alpha: float = 0.05) -> float | None:
    """Smallest pre-specifiable margin at which TOST would declare equivalence.

    The largest defensible claim from one run is ``|mean| + t_{1-alpha} *
    se`` (the half-width of the ``(1 - 2*alpha)`` t CI around the observed
    mean difference), i.e. the margin must strictly exceed this number for
    the same data to reject non-equivalence. Returning it next to a
    non-significant AND non-equivalent comparison turns "inconclusive"
    into an actionable choice: either commit to a margin at least this
    wide before the next run, or spend the seeds needed to shrink it.

    ``None`` when undefined (``n < 2``). For a zero-variance difference
    vector the infimum is ``|mean|`` (strict inequality, documented as
    such). This is a data-dependent PLANNING quantity; the margin used in
    a confirmatory ``tost_paired`` call must still be pre-specified.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError(
            f'alpha must be in (0, 0.5) for an equivalence CI to exist, '
            f'got {alpha!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n < 2:
        return None
    mean = _safe_mean(d)
    sd = _sample_sd(d, mean)
    if sd is None or sd <= 0.0:
        return abs(mean)
    se = sd / math.sqrt(n)
    supported = abs(mean) + _t_quantile(1.0 - alpha, n - 1) * se
    # The true infimum can exceed the float range for extreme finite
    # differences; ``inf`` cannot be stored in the strict-JSON record.
    return supported if math.isfinite(supported) else None


def holm_adjust(p_values, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down adjustment for a family of p-values.

    Returns adjusted p-values (same order as the input), the step-down
    rejection mask at level ``alpha``, and the family size. The step-down
    procedure is uniformly at least as powerful as plain Bonferroni while
    retaining FWER <= alpha; adjusted p-values are the running maximum of
    ``(m - rank + 1) * p`` over the ascending p-value order.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    vals = []
    for i, p in enumerate(p_values):
        if p is None:
            raise ValueError(
                f'p_values[{i}] is None; exclude untestable comparisons '
                f'from the family or set p=1.0 explicitly')
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(
                f'p_values[{i}]={p!r} is not a real number in [0,1]')
        try:
            p = float(p)
        except OverflowError:
            raise ValueError(f'p_values[{i}]={p!r} is not in [0,1]')
        if not (0.0 <= p <= 1.0) or not math.isfinite(p):
            raise ValueError(f'p_values[{i}]={p!r} is not in [0,1]')
        vals.append(p)
    m = len(vals)
    if m == 0:
        return {'adjusted': [], 'rejected': [], 'n_tests': 0, 'alpha': alpha,
                'method': 'holm_bonferroni'}
    order = sorted(range(m), key=lambda i: (vals[i], i))
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * vals[idx])
        adjusted[idx] = min(1.0, running)
    rejected = [adjusted[i] <= alpha for i in range(m)]
    return {'adjusted': adjusted, 'rejected': rejected, 'n_tests': m,
            'alpha': alpha, 'method': 'holm_bonferroni'}


# ---------------------------------------------------------------------------
# Multi-seed planning: power / required seed count
# ---------------------------------------------------------------------------

_ND = NormalDist()


def min_detectable_effect(n: int, sd: float, alpha: float = 0.05,
                          power: float = 0.8,
                          alternative: str = 'two-sided') -> float | None:
    """Smallest true mean paired difference detectable with the given power.

    Normal approximation: ``MDE = (z_crit + z_power) * sd / sqrt(n)``.
    ``sd`` is the per-seed difference SD observed (or assumed) on the same
    scale as the mean difference. This is a planning estimate, not a
    guarantee — at n = 5 the normal approximation to the t-test is
    optimistic by roughly 10%; use the exact tests above for inference.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f'n must be a non-negative int, got {n!r}')
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError(
            f'alpha and power must be in (0,1), got {alpha!r}, {power!r}')
    if not _is_finite_real(sd) or sd < 0.0:
        raise ValueError(f'sd must be finite and >= 0, got {sd!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    if n < 2:
        return None
    z_crit = _ND.inv_cdf(1.0 - alpha / 2.0) if alternative == 'two-sided' \
        else _ND.inv_cdf(1.0 - alpha)
    z_power = _ND.inv_cdf(power)
    value = (z_crit + z_power) * sd / math.sqrt(n)
    # A finite SD can still exceed the representable MDE range; the module
    # convention for an unrepresentable statistic is ``None``, never inf.
    return value if math.isfinite(value) else None


def seeds_required(dz: float, alpha: float = 0.05, power: float = 0.8,
                   alternative: str = 'two-sided') -> int | None:
    """Paired-test seed count needed to reach ``power`` for effect ``dz``.

    ``dz`` is the standardized effect ``mean(diff) / sd(diff)``.
    Normal approximation ``n = ceil(((z_crit + z_power) / dz)^2)``.
    Returns ``None`` for a zero / non-finite effect (a null effect can
    never be detected by adding seeds). This is the number the *next*
    multi-seed run would need — printing it next to a non-significant
    comparison turns "inconclusive" into an actionable seed budget.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError(
            f'alpha and power must be in (0,1), got {alpha!r}, {power!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    if isinstance(dz, bool) or not isinstance(dz, (int, float)):
        raise ValueError(f'dz must be a real number, got {dz!r}')
    try:
        dz = float(dz)
    except OverflowError:
        # A JSON-sized integer is not representable; like inf/nan it has no
        # actionable finite seed budget.
        return None
    if not math.isfinite(dz) or dz == 0.0:
        return None
    z_crit = _ND.inv_cdf(1.0 - alpha / 2.0) if alternative == 'two-sided' \
        else _ND.inv_cdf(1.0 - alpha)
    z_power = _ND.inv_cdf(power)
    try:
        required = ((z_crit + z_power) / abs(dz)) ** 2
    except OverflowError:
        return None
    if not math.isfinite(required):
        # The exact effect is non-zero but no finite seed count can reach
        # the requested power; report that instead of raising/returning 0.
        return None
    # A paired test is undefined below two seed pairs. The normal
    # approximation can round a very large effect down to 0 or 1, so clamp
    # the recommendation to the smallest runnable design.
    return max(2, int(math.ceil(required)))


def power_at_effect(n: int, sd: float, effect: float,
                    alpha: float = 0.05,
                    alternative: str = 'two-sided') -> float | None:
    """Normal-approximation power for the paired test at ``effect``.

    This is the forward companion of :func:`seeds_required`: it answers
    "if the true mean paired difference is ``effect`` and the per-seed
    difference SD is ``sd``, what is the probability that a run with
    ``n`` seed pairs rejects at ``alpha``?".  ``effect`` and ``sd`` are
    on the same scale (raw difference units); standardize by
    ``effect / sd`` for a ``d_z`` formulation.

    The normal approximation is the same planning model used by
    :func:`min_detectable_effect` / :func:`seeds_required`; it is
    deliberately NOT used for inference.  At ``n = 5`` it is optimistic
    by roughly 10%, so a planned power of 0.80 is a target, not a
    promise.  Returns ``None`` when fewer than two seed pairs make the
    sample SD (and therefore the test) undefined.  A zero SD is handled
    exactly: power is ``alpha`` for a zero effect, 1 when a non-zero effect
    points in the alternative's direction, and 0 when it points the other
    way (the degenerate noiseless case).  The sign of ``effect`` therefore
    matters for ``alternative='greater'|'less'``; a positive effect has
    power below ``alpha`` under ``less``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f'n must be a non-negative int, got {n!r}')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    if not _is_finite_real(sd) or sd < 0.0:
        raise ValueError(f'sd must be finite and >= 0, got {sd!r}')
    if not _is_finite_real(effect):
        raise ValueError(f'effect must be finite, got {effect!r}')
    if n < 2:
        return None
    effect = float(effect)
    if float(sd) == 0.0:
        if effect == 0.0:
            return float(alpha)
        # A noiseless paired test rejects with probability 1 when the sign
        # of the effect matches the one-sided alternative and with
        # probability 0 when it points the other way.  Two-sided rejection
        # is sign-agnostic.  Returning 1.0 unconditionally would claim
        # full power for an effect in the WRONG direction.
        if alternative == 'two-sided':
            return 1.0
        if alternative == 'greater':
            return 1.0 if effect > 0.0 else 0.0
        return 1.0 if effect < 0.0 else 0.0
    # Keep the sign for the directional alternatives: ``less`` rejects in
    # the lower tail, so a positive effect has power BELOW alpha rather
    # than the ``greater`` power.  The two-sided computation is symmetric
    # and uses the magnitude.
    ncp = effect / (float(sd) / math.sqrt(n))
    if alternative == 'two-sided':
        z_crit = _ND.inv_cdf(1.0 - alpha / 2.0)
        power = ((1.0 - _ND.cdf(z_crit - abs(ncp)))
                 + _ND.cdf(-z_crit - abs(ncp)))
    elif alternative == 'greater':
        z_crit = _ND.inv_cdf(1.0 - alpha)
        power = 1.0 - _ND.cdf(z_crit - ncp)
    else:
        z_crit = _ND.inv_cdf(1.0 - alpha)
        power = _ND.cdf(-z_crit - ncp)
    return min(1.0, max(0.0, float(power)))


# ---------------------------------------------------------------------------
# Leave-one-seed-out sensitivity
# ---------------------------------------------------------------------------

def _direction_of(mean: float | None) -> str:
    """Signed direction label used by the influence report."""
    if mean is None or mean == 0.0:
        return 'none'
    return 'greater' if mean > 0.0 else 'less'


def leave_one_out_sensitivity(
        diffs, *, alpha: float = 0.05,
        equivalence_margin: float | None = None,
        max_exact: int = 16, n_permutations: int = 10000,
        seed: int = 0) -> dict:
    """Drop each seed in turn and re-run the paired decision.

    At the 5-7-seed scale a single diverging seed can carry a mean
    difference, flip its sign, or move a p-value across alpha. The
    three-p-value panel in :func:`summarize_paired` can show that the
    statistic is fragile, but not WHICH seed is responsible or whether
    the reported direction survives. This function answers that by
    re-computing the full paired panel on every ``n - 1`` subset (the
    leave-one-seed-out / deletion-jackknife view).

    For every deleted position it records the remaining mean / SD /
    direction, the t, sign and sign-flip permutation p-values, whether
    the direction or the primary (permutation) decision changed, and —
    when a pre-specified ``equivalence_margin`` is supplied — the TOST
    verdict. The summary names the most influential seed, the range of
    the remaining mean, and a status:

    ``stable``
        The direction and the primary significance decision are unchanged
        by deleting any single seed.
    ``direction_fragile``
        Deleting at least one seed reverses (or erases) the sign of the
        mean difference.
    ``boundary_sensitive``
        The direction survives, but the primary permutation decision
        flips at ``alpha`` when at least one seed is deleted. Note that a
        unanimous 6-pair exact permutation result is *always* in this
        class: deleting one seed leaves 5 pairs, whose smallest attainable
        two-sided p is 0.0625. That is a resolution property of the exact
        test, not evidence against the effect; read
        ``n_primary_significance_flips`` alongside ``n_pairs``.
    ``no_direction``
        The full-sample mean difference is exactly zero, so there is no
        direction to test for stability.
    ``insufficient_data``
        Fewer than 3 pairs: at least two pairs must remain after a
        deletion, so no leave-one-out subset is testable.

    This is a **descriptive sensitivity analysis, not a test**. The
    leave-one-out p-values re-use the same data, are not independent,
    are not corrected for multiplicity, and are not a license to delete
    the seed that "fixes" the result. A stable report also cannot upgrade
    a non-significant or underpowered claim; it can only show that the
    observed decision does not hinge on one seed. The confirmatory
    inference remains :func:`summarize_paired` on the pre-registered seed
    set.

    When ``equivalence_margin`` is supplied, the TOST flags
    (``equivalence_verdict_stable`` / ``n_equivalence_verdict_flips``) are
    computed at the raw per-pair ``alpha``; the family is not re-Holm-ed
    for each deletion, so they describe the raw TOST verdict, not the
    reported Holm-corrected equivalence verdict. Likewise, the
    ``primary_significance_*`` fields use the raw per-pair ``alpha``, not
    the Holm-adjusted family decision (deleting a seed changes the family
    as well, which this per-pair view deliberately does not re-run). The
    ``status`` concerns only the direction and the raw-alpha superiority
    decision; a direction flip is alpha-independent and remains the
    relevant robustness signal for a Holm-corrected claim.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    if (isinstance(max_exact, bool)
            or not isinstance(max_exact, int) or max_exact < 0):
        raise ValueError(
            f'max_exact must be a non-negative int, got {max_exact!r}')
    if (isinstance(n_permutations, bool)
            or not isinstance(n_permutations, int) or n_permutations < 1):
        raise ValueError(
            f'n_permutations must be a positive int, got '
            f'{n_permutations!r}')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f'seed must be an int, got {seed!r}')
    margin = None
    if equivalence_margin is not None:
        if isinstance(equivalence_margin, bool):
            raise ValueError(
                f'equivalence_margin must be a positive number, got '
                f'{equivalence_margin!r}')
        try:
            margin = float(equivalence_margin)
        except (TypeError, ValueError):
            raise ValueError(
                f'equivalence_margin must be a positive number, got '
                f'{equivalence_margin!r}') from None
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                f'equivalence_margin must be finite and > 0, got '
                f'{equivalence_margin!r}')
        # The TOST equivalence CI only exists for alpha < 0.5.
        if not (0.0 < alpha < 0.5):
            raise ValueError(
                f'alpha must be in (0, 0.5) when an equivalence margin is '
                f'given, got {alpha!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    # Use the same overflow-safe mean as the rest of the panel. A raw
    # ``sum(d) / n`` turns a finite vector like [1e308, 1e308] into ``inf``
    # here, which then poisons ``mean_shift`` / ``relative_mean_shift`` and
    # makes the JSON report non-strict (the drop-one means below already use
    # ``_safe_mean``).
    mean = _safe_mean(d)
    sd = _sample_sd(d, mean) if n >= 2 else None
    t = paired_t_test(d)
    sign = sign_test(d)
    perm = permutation_test(d, max_exact=max_exact,
                            n_permutations=n_permutations, seed=seed)
    base_direction = _direction_of(mean)
    base_t_sig = (t['p_value'] < alpha) if t['p_value'] is not None else None
    base_sign_sig = bool(sign['p_value'] < alpha)
    base_perm_sig = (bool(perm['p_value'] < alpha)
                     if perm['p_value'] is not None else None)
    base_tost = (tost_paired(d, margin, alpha=alpha)
                 if margin is not None else None)
    baseline = {
        'n_pairs': n,
        'mean_diff': mean,
        'sd_diff': sd,
        'direction': base_direction,
        't_p_value': t['p_value'],
        't_significant': base_t_sig,
        'sign_test_p_value': sign['p_value'],
        'sign_test_significant': base_sign_sig,
        'permutation_p_value': perm['p_value'],
        'permutation_significant': base_perm_sig,
    }
    if base_tost is not None:
        baseline['tost_p_value'] = base_tost['p_value']
        baseline['practically_equivalent'] = base_tost['equivalent']

    note = (
        'Leave-one-seed-out sensitivity is DESCRIPTIVE: it re-uses the '
        'same data, tests no hypothesis, is not multiplicity-corrected, '
        'and cannot upgrade or rescue a claim. It shows only whether the '
        'observed direction or the primary permutation decision depends '
        'on one training seed.')
    if n < 3:
        return {
            'n_pairs': n, 'n_dropped': 0,
            'method': 'leave_one_seed_out',
            'primary_test': 'sign_flip_permutation',
            'correction': 'none (raw per-pair alpha; descriptive only)',
            'alpha': alpha, 'equivalence_margin': margin,
            'status': 'insufficient_data',
            'fragile': None,
            'direction_fragile': None,
            'decision_sensitive': None,
            'baseline': baseline,
            'drops': [],
            'min_mean_diff': None, 'max_mean_diff': None,
            'max_abs_mean_shift': None,
            'max_relative_mean_shift': None,
            'most_influential_index': None,
            'most_influential_value': None,
            'sign_stable': None,
            'primary_significance_stable': None,
            't_significance_stable': None,
            'sign_test_significance_stable': None,
            'equivalence_verdict_stable': None,
            'n_sign_flips': 0,
            'n_primary_significance_flips': 0,
            'n_t_significance_flips': None,
            'n_sign_test_significance_flips': 0,
            'n_equivalence_verdict_flips': None,
            'note': note,
        }

    drops = []
    for i in range(n):
        rest = d[:i] + d[i + 1:]
        m = _safe_mean(rest)
        s = _sample_sd(rest, m)
        rt = paired_t_test(rest)
        rs = sign_test(rest)
        rp = permutation_test(rest, max_exact=max_exact,
                              n_permutations=n_permutations, seed=seed)
        direction = _direction_of(m)
        shift = m - mean
        if not math.isfinite(shift):
            shift = None
        relative_shift = None
        if shift is not None and mean != 0.0:
            relative_shift = abs(shift) / abs(mean)
            if not math.isfinite(relative_shift):
                relative_shift = None
        t_sig = (rt['p_value'] < alpha) if rt['p_value'] is not None else None
        perm_sig = (bool(rp['p_value'] < alpha)
                    if rp['p_value'] is not None else None)
        sign_sig = bool(rs['p_value'] < alpha)
        entry = {
            'dropped_index': i,
            'dropped_value': d[i],
            'n_pairs': len(rest),
            'mean_diff': m,
            'sd_diff': s,
            'direction': direction,
            'mean_shift': shift,
            'relative_mean_shift': relative_shift,
            'sign_flip': bool(base_direction != 'none'
                              and direction != base_direction),
            't_p_value': rt['p_value'],
            't_significant': t_sig,
            'sign_test_p_value': rs['p_value'],
            'sign_test_significant': sign_sig,
            'permutation_p_value': rp['p_value'],
            'permutation_significant': perm_sig,
            'primary_significance_flip': bool(
                base_perm_sig is not None and perm_sig != base_perm_sig),
        }
        if margin is not None:
            rtost = tost_paired(rest, margin, alpha=alpha)
            entry['tost_p_value'] = rtost['p_value']
            entry['practically_equivalent'] = rtost['equivalent']
            entry['equivalence_verdict_flip'] = (
                None if (base_tost['equivalent'] is None
                         or rtost['equivalent'] is None)
                else bool(rtost['equivalent'] != base_tost['equivalent']))
        drops.append(entry)

    finite_shifts = [(abs(x['mean_shift']), i) for i, x in enumerate(drops)
                     if x['mean_shift'] is not None]
    if finite_shifts:
        max_shift = max(shift for shift, _ in finite_shifts)
        most = next(i for shift, i in finite_shifts if shift == max_shift)
    else:
        max_shift = most = None
    finite_relative = [x['relative_mean_shift'] for x in drops
                       if x['relative_mean_shift'] is not None]
    n_sign_flips = sum(1 for x in drops if x['sign_flip'])
    n_perm_flips = sum(1 for x in drops if x['primary_significance_flip'])
    t_flags = [x['t_significant'] for x in drops]
    t_all_defined = (base_t_sig is not None
                     and all(v is not None for v in t_flags))
    t_stable = (all(v == base_t_sig for v in t_flags)
                if t_all_defined else None)
    sign_stable = (n_sign_flips == 0) if base_direction != 'none' else None
    perm_stable = (n_perm_flips == 0) if base_perm_sig is not None else None
    if margin is not None:
        eq_flags = [x['practically_equivalent'] for x in drops]
        eq_defined = (base_tost['equivalent'] is not None
                      and all(v is not None for v in eq_flags))
        n_eq_flips = (sum(1 for x in drops
                          if x['equivalence_verdict_flip'])
                      if eq_defined else None)
        eq_stable = (n_eq_flips == 0) if eq_defined else None
    else:
        n_eq_flips = None
        eq_stable = None
    if base_direction == 'none':
        status = 'no_direction'
    elif n_sign_flips:
        status = 'direction_fragile'
    elif n_perm_flips:
        status = 'boundary_sensitive'
    else:
        status = 'stable'
    direction_fragile = bool(status == 'direction_fragile')
    decision_sensitive = bool(status == 'boundary_sensitive')
    return {
        'n_pairs': n, 'n_dropped': n,
        'method': 'leave_one_seed_out',
        'primary_test': 'sign_flip_permutation',
        'correction': 'none (raw per-pair alpha; descriptive only)',
        'alpha': alpha, 'equivalence_margin': margin,
        'status': status,
        'fragile': bool(direction_fragile or decision_sensitive),
        'direction_fragile': direction_fragile,
        'decision_sensitive': decision_sensitive,
        'baseline': baseline,
        'drops': drops,
        'min_mean_diff': min(x['mean_diff'] for x in drops),
        'max_mean_diff': max(x['mean_diff'] for x in drops),
        'max_abs_mean_shift': max_shift,
        'max_relative_mean_shift': (max(finite_relative)
                                    if finite_relative else None),
        'most_influential_index': most,
        'most_influential_value': d[most] if most is not None else None,
        'sign_stable': sign_stable,
        'primary_significance_stable': perm_stable,
        't_significance_stable': t_stable,
        'sign_test_significance_stable': bool(
            all(v == base_sign_sig
                for v in (x['sign_test_significant'] for x in drops))),
        'equivalence_verdict_stable': eq_stable,
        'n_sign_flips': n_sign_flips,
        'n_primary_significance_flips': n_perm_flips,
        'n_t_significance_flips': (
            sum(1 for v in t_flags if v != base_t_sig)
            if t_all_defined else None),
        'n_sign_test_significance_flips': sum(
            1 for x in drops if x['sign_test_significant'] != base_sign_sig),
        'n_equivalence_verdict_flips': n_eq_flips,
        'note': note,
    }


# ---------------------------------------------------------------------------
# One-call panel
# ---------------------------------------------------------------------------

def summarize_paired(diffs, seed: int = 0, n_boot: int = 10000,
                     n_permutations: int = 10000,
                     max_exact: int = 16,
                     alpha: float = 0.05,
                     power: float = 0.8,
                     equivalence_margin: float | None = None,
                     include_seed_influence: bool = False,
                     family_size: int = 1) -> dict:
    """Run the full multi-seed inference panel on paired differences.

    Returns a single JSON-safe record containing the effect estimate,
    t-based and bootstrap CIs, the parametric t-test, the exact sign test,
    the sign-flip permutation test, Cohen's ``d_z``, the seed win rate,
    and the number of seeds a follow-up run would need for ``power`` at
    the observed effect. ``direction`` is ``'greater'`` when the mean
    difference is positive, ``'less'`` when negative, ``'none'`` when
    exactly zero.

    When ``equivalence_margin`` is given (a PRE-SPECIFIED smallest effect
    of interest, on the same scale as the differences), the record
    additionally carries a ``tost`` block and a top-level
    ``practically_equivalent`` flag from :func:`tost_paired`. Without it
    the record is unchanged (``practically_equivalent=None`` and no
    ``tost`` key), so a non-significant result can never be silently
    upgraded to an equivalence claim.

    The record deliberately reports three p-values for the same
    comparison: if they disagree materially at n = 5, the per-seed
    differences are skew/outlier-driven and the result should be treated
    as fragile rather than decided by whichever test looks best.

    ``family_size`` is the number of comparisons the caller will correct
    with Holm. The record then carries a ``test_resolution`` block from
    :func:`exact_test_resolution`: the smallest p the exact test could
    possibly produce at this n, the smallest Holm-adjusted p for this
    family, and the seed count needed to restore a rejection region.
    Without this block a forced non-significant exact test is easy to
    misread as evidence of no difference. The default ``family_size=1``
    describes the comparison in isolation; callers that correct a family
    should pass the actual family size (or overwrite the block from the
    family-level report).

    The record also carries the exact Clopper-Pearson interval for the
    seed win rate (``win_rate_ci95``) and the distribution-free order-
    statistic interval for the paired median (``median_ci``). Both are
    undefined at the sample sizes where no interval exists, rather than
    replaced by a misleading point.

    ``include_seed_influence=True`` additionally attaches a
    ``seed_influence`` block from :func:`leave_one_out_sensitivity`: which
    seed, if deleted, would change the direction or the primary
    permutation decision. It is opt-in because it re-runs the panel
    ``n`` times; enable it for the operator comparisons whose claim
    depends on the seed set, not for bulk per-cell summaries where the
    extra record would dominate the output.
    """
    if (isinstance(family_size, bool) or not isinstance(family_size, int)
            or family_size < 1):
        raise ValueError(
            f'family_size must be a positive int, got {family_size!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    mean = _safe_mean(d)
    median = _safe_median(d)
    sd = _sample_sd(d, mean) if n >= 2 else None
    t = paired_t_test(d, alternative='two-sided')
    t_dir = paired_t_test(
        d, alternative='greater' if (mean is not None and mean >= 0)
        else 'less')
    sign = sign_test(d, alpha=alpha)
    median_ci = median_order_statistic_ci(d, alpha=alpha)
    resolution = exact_test_resolution(
        n, alpha=alpha, family_size=family_size,
        alternative='two-sided', n_zero=sign['n_zero'])
    perm = permutation_test(d, alternative='two-sided', seed=seed,
                            n_permutations=n_permutations,
                            max_exact=max_exact)
    boot = bootstrap_paired_ci(d, alpha=alpha, n_boot=n_boot, seed=seed)
    dz = cohens_dz(d)
    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    n_nonzero = n_pos + n_neg
    win_rate = (n_pos / n_nonzero) if n_nonzero else None
    # t-based CI for the mean difference (companion to the bootstrap CI).
    ci_lo = ci_hi = None
    if n >= 2 and sd is not None and sd > 0.0:
        # Two-sided t critical value via bisection on the forward CDF.
        crit = _t_quantile(1.0 - alpha / 2.0, n - 1)
        half = crit * sd / math.sqrt(n)
        ci_lo, ci_hi = mean - half, mean + half
        # Finite inputs can produce a half-width beyond the float range;
        # an unrepresentable interval is reported as undefined so the
        # record stays strict-JSON.
        if not (math.isfinite(ci_lo) and math.isfinite(ci_hi)):
            ci_lo = ci_hi = None
    direction = ('none' if mean is None or mean == 0.0
                 else ('greater' if mean > 0 else 'less'))
    tost = (tost_paired(d, equivalence_margin, alpha=alpha)
            if equivalence_margin is not None else None)
    record = {
        'n_pairs': n,
        'mean_diff': mean,
        'median_diff': median,
        'sd_diff': sd,
        'cohens_dz': dz,
        'direction': direction,
        'win_rate': win_rate,
        'win_rate_ci95': sign['win_rate_ci95'],
        'n_seeds_a_wins': n_pos,
        'n_seeds_b_wins': n_neg,
        'n_seeds_tied': n - n_nonzero,
        'ci95_mean_diff_t': [ci_lo, ci_hi],
        'ci95_mean_diff_bootstrap': [boot['ci_lo'], boot['ci_hi']],
        'ci95_median_diff': [median_ci['ci_low'], median_ci['ci_high']],
        'median_ci': median_ci,
        'bootstrap': {'n_boot': boot['n_boot'], 'seed': boot['seed']},
        't_test': {'t_stat': t['t_stat'], 'dof': t['dof'],
                   'p_value_two_sided': t['p_value'],
                   'p_value_directional': t_dir['p_value'],
                   'all_positive': t['all_positive'],
                   'all_negative': t['all_negative']},
        'sign_test': {'n_effective': sign['n_effective'],
                      'n_zero': sign['n_zero'],
                      'win_rate': sign['win_rate'],
                      'win_rate_ci95': sign['win_rate_ci95'],
                      'p_value_two_sided': sign['p_value']},
        'permutation_test': {'method': perm['method'],
                             'n_permutations': perm['n_permutations'],
                             'p_value_two_sided': perm['p_value']},
        'seeds_required_80pct_power': (
            seeds_required(dz, alpha=alpha, power=power) if dz else None),
        'power_note': ('seeds_required_80pct_power is a normal-approximation '
                       'planning estimate for a FUTURE run at the observed '
                       '|d_z| (two-sided alpha, 80% power), not a guarantee '
                       'for this one.'),
        'test_resolution': resolution,
        'practically_equivalent': (tost['equivalent'] if tost else None),
        'equivalence_note': (
            'practically_equivalent is None unless a PRE-SPECIFIED '
            'equivalence_margin was supplied: a non-significant superiority '
            'test is not evidence of equivalence. When a TOST is present, '
            'it tests |mean_diff| < margin and can positively support a '
            '"practically the same" claim.'),
    }
    if tost is not None:
        record['tost'] = tost
        record['equivalence_margin_supported'] = equivalence_margin_supported(
            d, alpha=alpha)
    if include_seed_influence:
        record['seed_influence'] = leave_one_out_sensitivity(
            d, alpha=alpha, equivalence_margin=equivalence_margin,
            max_exact=max_exact, n_permutations=n_permutations, seed=seed)
    return record


def _t_quantile(p: float, dof: int) -> float:
    """Inverse t-CDF by bisection on :func:`student_t_cdf` (p in (0,1))."""
    if not (0.0 < p < 1.0):
        raise ValueError(f'p must be in (0,1), got {p!r}')
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, dof) > p:
        lo *= 2.0
    while student_t_cdf(hi, dof) < p:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
