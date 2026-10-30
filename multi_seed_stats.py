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
14. :func:`interaction_contrast` — the seed-paired 2x2 difference-in-
    differences ``(d - b) - (c - a)``.  A main-effect contrast cannot
    answer an interaction question: "does the benefit of CSA depend on
    the memory route?" or "does removing RoPE hurt more when selection is
    dense?" needs the paired interaction estimand, not two separate
    within-route tests read side by side.

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
from decimal import Decimal, localcontext
from fractions import Fraction
from statistics import NormalDist

__all__ = [
    'betainc_regularized',
    'student_t_cdf',
    'student_t_sf',
    'paired_t_test',
    'interaction_contrast',
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
    'cross_cell_paired_analysis',
]

# ---------------------------------------------------------------------------
# Student-t distribution (forward CDF / survival function)
# ---------------------------------------------------------------------------

_MIN_BETACF_ITER = 200
_MAX_BETACF_ITER = 100_000
_BETACF_EPS = 3.0e-14
_FPMIN = 1.0e-300
_STABLE_LOG_BETA_MIN = 1.0e3
_LARGE_BETA_NORMAL_MIN = 1.0e12
_LARGE_BETA_NORMAL_Z = 8.0
_NORMAL_T_DOF = 200_000_000
_MAX_EXACT_PERMUTATION_BITS = 20


def _sqrt_count(n: int) -> float:
    """Square root of a non-negative integer, ``inf`` if unrepresentable."""
    if n <= 0:
        return 0.0
    try:
        return math.sqrt(n)
    except OverflowError:
        shift = (n.bit_length() - 1) // 2
        mantissa = n / (1 << (2 * shift))
        try:
            return math.ldexp(math.sqrt(mantissa), shift)
        except OverflowError:
            return math.inf


def _float_count(n: int) -> float | None:
    """Float value of an integer count, or ``None`` beyond float range."""
    try:
        return float(n)
    except OverflowError:
        return None


def _log_gamma_ratio_asym(z: float, m: float) -> float:
    """``logGamma(z + m) - logGamma(z)`` for large ``z``.

    Subtracting two ``math.lgamma`` values loses about
    ``eps * lgamma(z)`` absolute precision when ``z`` is large and ``m`` is
    small.  Stirling's expansion keeps the small difference directly.
    """
    log_ratio = math.log1p(m / z)
    out = m * (math.log(z) - 1.0) + (z + m - 0.5) * log_ratio
    inv_z = 1.0 / z
    inv_sum = 1.0 / (z + m)
    out += (inv_sum - inv_z) / 12.0
    out -= (inv_sum ** 3 - inv_z ** 3) / 360.0
    return out


def _log_beta(a: float, b: float) -> float:
    """``log(Beta(a, b))`` with a cancellation-safe large-shape path."""
    z, m = (a, b) if a >= b else (b, a)
    if (z >= _STABLE_LOG_BETA_MIN and math.isfinite(z + m)):
        return math.lgamma(m) - _log_gamma_ratio_asym(z, m)
    # Keep the historical subtraction order and sign exactly: the callers
    # used ``-(lgamma(a+b) - lgamma(a) - lgamma(b))`` for the log-beta.
    return -(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def _stirling_beta_correction(a: float, b: float, s: float) -> float:
    """Stirling correction for ``log(Beta(a, b))`` at large shapes."""
    ia, ib, isum = 1.0 / a, 1.0 / b, 1.0 / s
    return ((ia + ib - isum) / 12.0
            - (ia ** 3 + ib ** 3 - isum ** 3) / 360.0
            + (ia ** 5 + ib ** 5 - isum ** 5) / 1260.0)


def _log_beta_exponent(a: float, b: float, x: float) -> float:
    """``log(x**a * (1-x)**b / Beta(a, b))`` without large cancellation."""
    if min(a, b) < _STABLE_LOG_BETA_MIN:
        return -_log_beta(a, b) + a * math.log(x) + b * math.log1p(-x)
    s = a + b
    if not math.isfinite(s):
        return -_log_beta(a, b) + a * math.log(x) + b * math.log1p(-x)
    y = 1.0 - x
    log_x = math.log1p(-y) if x > 0.5 else math.log(x)
    log_y = math.log1p(-x) if y > 0.5 else math.log(y)
    if a <= b:
        p = a / s
        log_p = math.log(p)
        log_q = math.log1p(-p)
        ratio_x = (x - p) / p
        log_ratio_x = (math.log1p(ratio_x) if -0.5 < ratio_x < 1.0
                       else log_x - log_p)
        ratio_y = (p - x) / (1.0 - p)
        log_ratio_y = (math.log1p(ratio_y) if -0.5 < ratio_y < 1.0
                       else log_y - log_q)
    else:
        q = b / s
        log_q = math.log(q)
        log_p = math.log1p(-q)
        diff = math.fsum([1.0, -x, -q])
        ratio_y = diff / q
        log_ratio_y = (math.log1p(ratio_y) if -0.5 < ratio_y < 1.0
                       else log_y - log_q)
        ratio_x = -diff / (1.0 - q)
        log_ratio_x = (math.log1p(ratio_x) if -0.5 < ratio_x < 1.0
                       else log_x - log_p)
    log_scale = (math.log(2.0 * math.pi) - math.log(s)
                 - log_p - log_q)
    return (a * log_ratio_x + b * log_ratio_y
            - 0.5 * log_scale - _stirling_beta_correction(a, b, s))


def _large_beta_normal_cdf(a: float, b: float,
                           x: float) -> tuple[float, float] | None:
    """Normal/Edgeworth approximation for a huge-shape Beta CDF.

    The continued fraction loses its scale once both shapes are large:
    at ``a=b=1e14`` the first denominator rounds away and the iteration
    cap truncates before convergence, yielding values outside ``[0, 1]``.
    The Beta distribution is then concentrated within ~``1/sqrt(a)`` of
    its mean, so the normal limit with the exact moments and a first-order
    skew correction is accurate to well below float resolution.
    """
    try:
        total = a + b
        total_is_finite = math.isfinite(total)
    except OverflowError:
        total = math.inf
        total_is_finite = False
    if total_is_finite:
        mean = a / total
    elif a == b:
        mean = 0.5
    else:
        ratio = b / a
        if math.isfinite(ratio):
            mean = 1.0 / (1.0 + ratio)
        elif ratio == math.inf:
            mean = 0.0
        else:
            mean = 0.5
    if not total_is_finite:
        if x < mean:
            return 0.0, -math.inf
        if x > mean:
            return 1.0, 0.0
        return 0.5, math.log(0.5)
    variance = mean * (1.0 - mean) / (total + 1.0)
    if variance <= 0.0:
        if x < mean:
            return 0.0, -math.inf
        if x > mean:
            return 1.0, 0.0
        return 0.5, math.log(0.5)
    sd = math.sqrt(variance)
    z = (x - mean) / sd
    if not math.isfinite(z) or abs(z) > _LARGE_BETA_NORMAL_Z:
        return None
    sqrt_ab = math.sqrt(a) * math.sqrt(b)
    gamma1 = (2.0 * ((b - a) / sqrt_ab)
              * (math.sqrt(total + 1.0) / (total + 2.0)))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    correction = gamma1 / 6.0 * (1.0 - z * z) * phi
    if z <= 0.0:
        lower = 0.5 * math.erfc(-z / math.sqrt(2.0))
        cdf = lower + correction
        log_cdf = (math.log(lower) + math.log1p(correction / lower)
                   if lower > 0.0 else -math.inf)
    else:
        upper = 0.5 * math.erfc(z / math.sqrt(2.0))
        cdf = 1.0 - upper + correction
        log_cdf = math.log1p(-upper + correction)
    return min(1.0, max(0.0, cdf)), log_cdf


def _betacf_max_iter(a: float, b: float) -> int:
    """Enough CF iterations for the largest shape parameter.

    The continued fraction needs roughly ``0.25 * sqrt(max(a, b))``
    iterations near the distribution's mode; the old fixed cap of 200
    truncated it for large shapes and could even return a negative
    regularized beta.  Small shapes still converge in a few iterations and
    exit early.
    """
    try:
        scaled = 2.0 * math.sqrt(max(a, b)) + _MIN_BETACF_ITER
    except (OverflowError, ValueError):
        return _MAX_BETACF_ITER
    return min(_MAX_BETACF_ITER, max(_MIN_BETACF_ITER, int(scaled)))


def _betacf_high_precision(a: float, b: float, x: float,
                           y: float | None = None) -> float | None:
    """Decimal evaluation of the beta continued fraction near ``x = 1``.

    The double-precision Lentz recurrence accumulates a relative error of
    a few ``1e-9`` for a huge first shape and ``x`` close to one: the early
    ``delta == 1`` terms are individually below the double epsilon but are
    numerous, and the double evaluation cannot recover their tail.  A
    short 40-digit fallback (a few dozen iterations for the Student-t
    regimes) keeps the direct branch accurate without altering the
    dependency-free path for ordinary shapes.
    """
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(x)):
        return None
    if not (a > 0.0 and b > 0.0 and 0.0 < x < 1.0):
        return None
    with localcontext() as ctx:
        ctx.prec = 40
        da, db = Decimal(a), Decimal(b)
        # The caller's ``y = 1 - x`` is often formed without cancellation
        # (e.g. t^2/(dof+t^2)); rebuild x from it so the CF is evaluated
        # at the unrounded complement of the accurate small argument.
        dx = (Decimal(1) - Decimal(y) if y is not None else Decimal(x))
        if dx <= 0:
            return None
        qab, qap, qam = da + db, da + 1, da - 1
        tiny = Decimal('1e-200')
        c = Decimal(1)
        d = Decimal(1) - qab * dx / qap
        if abs(d) < tiny:
            d = tiny
        d = 1 / d
        h = d
        max_iter = min(_betacf_max_iter(a, b), 5000)
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            dm = Decimal(m)
            aa = dm * (db - dm) * dx / ((qam + m2) * (da + m2))
            d = 1 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            h *= d * c
            aa = (-(da + dm) * (qab + dm) * dx
                  / ((da + m2) * (qap + m2)))
            d = 1 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < Decimal('1e-30'):
                try:
                    precise = float(h)
                except (OverflowError, ValueError):
                    return None
                return precise if math.isfinite(precise) else None
    return None


def _betacf(a: float, b: float, x: float, *,
            y: float | None = None) -> float:
    """Continued fraction for the incomplete beta function (NR 6.4)."""
    small = y if y is not None else 1.0 - x
    if a >= 1.0e6 and x > 0.5 and small * a < 1.0e4:
        precise = _betacf_high_precision(a, b, x, y)
        if precise is not None and precise > 0.0:
            return precise
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _betacf_max_iter(a, b) + 1):
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


def _complement_tail_value(a: float, b: float, x: float, y: float,
                           log_bt: float) -> float | None:
    """Left-tail mass from the complementary CF when that side is stable.

    For a huge first shape and ``x`` just below the direct-CF branch
    threshold, the direct fraction subtracts nearly equal ``a*x``-scale
    terms and loses several digits, while the complementary fraction is
    still in its convergence region and the final ``1 - I_y`` is not a
    catastrophic cancellation. Return ``None`` when that regime does not
    apply so callers keep the direct value.
    """
    if not (x > 0.5 and a >= 1.0e6 and y <= 1.0e-2):
        return None
    y_cut = (b + 1.0) / (a + b + 2.0)
    if y > 4.0 * y_cut:
        return None
    h = _betacf(b, a, y)
    if not (h > 0.0 and math.isfinite(log_bt)):
        return None
    try:
        p = math.exp(log_bt + math.log(h) - math.log(b))
    except (OverflowError, ValueError):
        return None
    # If ``p`` is itself close to one, ``1 - p`` is dominated by the CF
    # error instead of the desired small tail: the direct branch is then
    # both well conditioned and more accurate. Only use the complement
    # when it is the genuinely smaller side of the distribution.
    if not (0.0 < p < 0.9):
        return None
    return 1.0 - p


def _betainc_regularized_from_logs(a: float, b: float, x: float,
                                   y: float, log_x: float,
                                   log_y: float) -> float:
    """``I_x(a, b)`` with caller-supplied accurate ``log(x)``/``log(1-x)``."""
    log_bt = -_log_beta(a, b) + a * log_x + b * log_y
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * _betacf(a, b, x, y=y) / a
        if value > 1.0e-8:
            stable = _complement_tail_value(a, b, x, y, log_bt)
            if stable is not None:
                return stable
        return value
    return 1.0 - bt * _betacf(b, a, y) / b


def _log_betainc_regularized_from_logs(a: float, b: float, x: float,
                                       y: float, log_x: float,
                                       log_y: float) -> float:
    """``log(I_x(a, b))`` with caller-supplied accurate logs."""
    log_bt = -_log_beta(a, b) + a * log_x + b * log_y
    if x < (a + 1.0) / (a + b + 2.0):
        h = _betacf(a, b, x, y=y)
        if h > 0.0 and math.isfinite(log_bt):
            log_ib = log_bt + math.log(h) - math.log(a)
            if log_ib <= 0.0:
                if log_ib > math.log(1.0e-8):
                    stable = _complement_tail_value(a, b, x, y, log_bt)
                    if stable is not None and stable > 0.0:
                        return math.log(stable)
                return log_ib
    else:
        h = _betacf(b, a, y)
        if h > 0.0 and math.isfinite(log_bt):
            log_iy = log_bt + math.log(h) - math.log(b)
            if log_iy < 0.0:
                return math.log1p(-math.exp(log_iy))
    return _log_betainc_regularized(a, b, x)


def betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` for ``a, b > 0``."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f'betainc requires a,b > 0, got a={a!r}, b={b!r}')
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if min(a, b) >= _LARGE_BETA_NORMAL_MIN:
        approx = _large_beta_normal_cdf(a, b, x)
        if approx is not None:
            return approx[0]
    log_bt = _log_beta_exponent(a, b, x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        value = bt * _betacf(a, b, x, y=1.0 - x) / a
        if value > 1.0e-8:
            stable = _complement_tail_value(a, b, x, 1.0 - x, log_bt)
            if stable is not None:
                return stable
        return value
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_sf_positive(t_value: float, dof: int) -> float:
    """``P(T > t)`` for ``t >= 0``, evaluated in the non-cancelling tail."""
    if t_value == 0.0:
        return 0.5
    if dof >= _NORMAL_T_DOF:
        # ``NormalDist().cdf(-t)`` is ``0.5 * (1 + erf(-t/sqrt(2)))``:
        # once erf rounds to -1 (t around 9), the subtraction cancels to
        # exactly zero even though the tail is still representable
        # (t=10 -> 7.6e-24). ``erfc`` evaluates that tail directly.
        tail = 0.5 * math.erfc(t_value / math.sqrt(2.0))
        t_sq = t_value * t_value
        if math.isfinite(t_sq) and dof <= 10 ** 300:
            phi = math.exp(-0.5 * t_sq) / math.sqrt(2.0 * math.pi)
            if phi > 0.0:
                t_four = t_sq * t_sq
                first = (t_sq + 1.0) * t_value * phi / 4.0
                second = (tail / 6.0 + t_value * (t_sq - 3.0)
                          * (3.0 * t_four + 2.0 * t_sq + 1.0)
                          * phi / 96.0)
                tail += first / dof + second / dof / dof
        return tail
    t_sq = t_value * t_value
    if math.isfinite(t_sq):
        # The survival-function identity is
        #     SF(t) = 0.5 * I_x(dof/2, 1/2),  x = dof / (dof + t^2).
        # For small |t| and large dof, ``x`` rounds to exactly 1.0 even
        # though ``1 - x`` is still representable: e.g. dof=999999,
        # t=1e-6 gives x=1.0 and a spurious SF of exactly 0.5 instead of
        # 0.4999996010578. When the incomplete beta would be evaluated on
        # the near-1 side, form ``y = 1 - x = t^2 / (dof + t^2)`` directly
        # (no cancellation) and use the symmetry
        # ``I_x(a, b) = 1 - I_y(b, a)``. The branch threshold is the same
        # one ``betainc_regularized`` uses internally, so ``I_y(b, a)`` is
        # evaluated by its direct (non-complemented) continued fraction.
        a = 0.5 * dof
        b = 0.5
        denom = dof + t_sq
        if math.isfinite(denom):
            y = t_sq / denom
            if y < (b + 1.0) / (a + b + 2.0):
                if y == 0.0:
                    # |t| is below the representable deviation from 0.5.
                    return 0.5
                return 0.5 * (1.0 - betainc_regularized(b, a, y))
            x = dof / denom
            if x > 0.0:
                log_x = math.log1p(-y) if x > 0.5 else math.log(x)
                log_y = math.log1p(-x) if y > 0.5 else math.log(y)
                return 0.5 * _betainc_regularized_from_logs(
                    a, b, x, y, log_x, log_y)
    # ``t*t`` overflowed (or ``x`` underflowed to zero) even though the
    # tail itself can still be representable: for t=1e155, dof=1 the true
    # tail is 3.18e-156 but forming t**2 first returns 0. Use the x -> 0
    # limit ``I_x(a, b) ~ x**a / (a B(a, b))`` in log space instead; the
    # dropped O(x) correction is far below float resolution here (x is
    # below the smallest subnormal).
    a = 0.5 * dof
    b = 0.5
    log_t = math.log(t_value)
    log_ratio = math.log(dof) - 2.0 * log_t
    log_x = math.log(dof) - 2.0 * log_t
    if log_ratio > _MIN_LOG_PROB:
        log_x -= math.log1p(math.exp(log_ratio))
    log_ib = (a * log_x - math.log(a)
              - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    return 0.5 * math.exp(log_ib)


def _log_student_t_sf_positive(t_value: float, dof: int) -> float:
    if dof <= 0:
        raise ValueError(f'student_t_sf requires dof > 0, got {dof!r}')
    if t_value == 0.0:
        return math.log(0.5)
    if dof >= _NORMAL_T_DOF:
        log_tail = _log_normal_upper_tail(t_value)
        t_sq = t_value * t_value
        if math.isfinite(t_sq) and dof <= 10 ** 300:
            log_phi = -0.5 * t_sq - 0.5 * math.log(2.0 * math.pi)
            log_ratio = (math.log(t_value) + math.log1p(t_sq)
                         - math.log(dof) - math.log(4.0)
                         + log_phi - log_tail)
            if log_ratio < math.log(0.1):
                t_four = t_sq * t_sq
                log_poly = (math.log(t_value)
                            + math.log(abs(t_sq - 3.0))
                            + math.log(3.0 * t_four + 2.0 * t_sq + 1.0)
                            - math.log(96.0))
                signed = math.exp(log_poly + log_phi - log_tail)
                if t_sq < 3.0:
                    signed = -signed
                ratio = (math.exp(log_ratio)
                         + (1.0 / 6.0 + signed) / dof / dof)
                if 0.0 < ratio < 0.1:
                    return log_tail + math.log1p(ratio)
        return log_tail
    t_sq = t_value * t_value
    if math.isfinite(t_sq):
        denom = dof + t_sq
        if math.isfinite(denom):
            a = 0.5 * dof
            b = 0.5
            y = t_sq / denom
            if y < (b + 1.0) / (a + b + 2.0):
                if y == 0.0:
                    return math.log(0.5)
                log_iy = _log_betainc_regularized(b, a, y)
                if log_iy >= 0.0:
                    return -math.inf
                return math.log(0.5) + math.log1p(-math.exp(log_iy))
            x = dof / denom
            if x > 0.0:
                log_x = math.log1p(-y) if x > 0.5 else math.log(x)
                log_y = math.log1p(-x) if y > 0.5 else math.log(y)
                return (math.log(0.5)
                        + _log_betainc_regularized_from_logs(
                            a, b, x, y, log_x, log_y))
    a = 0.5 * dof
    b = 0.5
    log_x = math.log(dof) - 2.0 * math.log(t_value)
    if log_x > _MIN_LOG_PROB:
        log_x -= math.log1p(math.exp(log_x))
    log_ib = (a * log_x - math.log(a)
              - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    return math.log(0.5) + log_ib


def student_t_cdf(t_value: float, dof: int) -> float:
    """CDF of Student's t with ``dof > 0`` degrees of freedom."""
    if dof <= 0:
        raise ValueError(f'student_t_cdf requires dof > 0, got {dof!r}')
    try:
        t_value = float(t_value)
    except OverflowError:
        # A JSON-sized integer is beyond the float range; the limiting
        # CDF is the same as +/-infinity.
        t_value = math.inf if t_value > 0 else -math.inf
    if not math.isfinite(t_value):
        return 1.0 if t_value > 0 else 0.0
    if t_value == 0.0:
        return 0.5
    tail = _student_t_sf_positive(abs(t_value), dof)
    return (1.0 - tail) if t_value > 0.0 else tail


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
    try:
        t_value = float(t_value)
    except OverflowError:
        t_value = math.inf if t_value > 0 else -math.inf
    if not math.isfinite(t_value):
        return 0.0 if t_value > 0 else 1.0
    if t_value == 0.0:
        return 0.5
    tail = _student_t_sf_positive(abs(t_value), dof)
    return tail if t_value > 0.0 else 1.0 - tail


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


def _log_betainc_regularized(a: float, b: float, x: float) -> float:
    """``log(I_x(a, b))`` without exponentiating an underflowing tail."""
    if x <= 0.0:
        return -math.inf
    if x >= 1.0:
        return 0.0
    if min(a, b) >= _LARGE_BETA_NORMAL_MIN:
        approx = _large_beta_normal_cdf(a, b, x)
        if approx is not None:
            return approx[1]
    log_bt = _log_beta_exponent(a, b, x)
    if x < (a + 1.0) / (a + b + 2.0):
        h = _betacf(a, b, x, y=1.0 - x)
        if h > 0.0 and math.isfinite(log_bt):
            log_ib = log_bt + math.log(h) - math.log(a)
            if log_ib <= 0.0:
                return log_ib
        value = betainc_regularized(a, b, x)
        return math.log(value) if value > 0.0 else -math.inf
    y = 1.0 - x
    if min(a, b) >= _STABLE_LOG_BETA_MIN:
        log_bt = _log_beta_exponent(b, a, y)
    else:
        log_bt = (
            -_log_beta(a, b)
            + b * math.log(y) + a * math.log1p(-y))
    h = _betacf(b, a, y)
    if h > 0.0 and math.isfinite(log_bt):
        log_iy = log_bt + math.log(h) - math.log(b)
        if log_iy < 0.0:
            return math.log1p(-math.exp(log_iy))
    value = betainc_regularized(a, b, x)
    return math.log(value) if value > 0.0 else -math.inf


def _beta_quantile_log_space(log_q: float, a: float, b: float,
                             hi: float) -> float:
    x_min = 5e-324
    log_lo = math.log(x_min)
    log_hi = math.log(hi)
    if _log_betainc_regularized(a, b, x_min) > log_q:
        return 0.0
    if log_hi <= log_lo:
        return x_min
    for _ in range(100):
        log_mid = 0.5 * (log_lo + log_hi)
        mid = math.exp(log_mid)
        if mid <= x_min or mid >= hi:
            break
        if _log_betainc_regularized(a, b, mid) < log_q:
            log_lo = log_mid
        else:
            log_hi = log_mid
    return math.exp(0.5 * (log_lo + log_hi))


def _beta_quantile_from_log(log_q: float, a: float, b: float) -> float:
    """Beta quantile for a tail probability given as ``log(q)``."""
    if log_q >= 0.0:
        return 1.0
    if log_q == -math.inf:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _log_betainc_regularized(a, b, mid) < log_q:
            lo = mid
        else:
            hi = mid
    if lo == 0.0:
        return _beta_quantile_log_space(log_q, a, b, hi)
    return 0.5 * (lo + hi)


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
    if lo == 0.0:
        return _beta_quantile_log_space(math.log(q), a, b, hi)
    return 0.5 * (lo + hi)


_SMALL_BINOMIAL_TAIL = 1024
_LARGE_BINOMIAL_N = 100_000_000
_MIN_LOG_PROB = math.log(5e-324)


def _binomial_cdf_small(n: int, k: int, p: float) -> float:
    """Exact ``P(X <= k)`` for a small ``k`` without huge log-gamma terms."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    n_log_q = math.log(n) + math.log(-log_q)
    n_q_term = -math.exp(n_log_q) if n_log_q <= 709.0 else -math.inf
    log_comb = 0.0
    terms = []
    for i in range(k + 1):
        if i:
            log_comb += math.log(n - i + 1) - math.log(i)
        terms.append(log_comb + i * log_p + n_q_term - i * log_q)
    peak = max(terms)
    if peak == -math.inf:
        return 0.0
    return math.exp(peak) * math.fsum(
        math.exp(term - peak) for term in terms)


def _binomial_log_cdf_small(n: int, k: int, p: float) -> float:
    """``log P(X <= k)`` for a small ``k`` without underflowing the sum."""
    if k < 0:
        return -math.inf
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf
    log_p = math.log(p)
    log_q = math.log1p(-p)
    n_log_q = math.log(n) + math.log(-log_q)
    n_q_term = -math.exp(n_log_q) if n_log_q <= 709.0 else -math.inf
    log_comb = 0.0
    terms = []
    for i in range(k + 1):
        if i:
            log_comb += math.log(n - i + 1) - math.log(i)
        terms.append(log_comb + i * log_p + n_q_term - i * log_q)
    peak = max(terms)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(
        math.exp(term - peak) for term in terms))


def _binomial_tail_root(n: int, k: int, target: float, *,
                        log_target: float | None = None) -> float:
    """Solve ``P(Binomial(n, p) <= k) = target`` in log-probability space."""
    lo = _MIN_LOG_PROB
    hi = 0.0
    if log_target is not None:
        if log_target == -math.inf:
            return 0.0
        if log_target >= 0.0:
            return 1.0
        if _binomial_log_cdf_small(n, k, math.exp(lo)) < log_target:
            return 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _binomial_log_cdf_small(n, k, math.exp(mid)) > log_target:
                lo = mid
            else:
                hi = mid
        return math.exp(0.5 * (lo + hi))
    if _binomial_cdf_small(n, k, math.exp(lo)) < target:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _binomial_cdf_small(n, k, math.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def _scale_int_by_float(n: int, value: float) -> float:
    """``n * value`` for an integer count that may exceed the float range.

    ``n * value`` raises ``OverflowError`` when ``n`` cannot be converted
    to float, even if the product is representable (a huge count times a
    tiny log-probability). Fall back to a log-space product in that case.
    """
    if n == 0 or value == 0.0:
        return 0.0
    try:
        return n * value
    except OverflowError:
        log_mag = math.log(n) + math.log(abs(value))
        if log_mag > 709.782712893384:
            return math.copysign(math.inf, value)
        return math.copysign(math.exp(log_mag), value)


def _binomial_log_sf_downward(n: int, k: int, log_p: float,
                              log_q: float) -> float:
    """``log P(X >= k)`` when ``k`` is at/above the mode.

    The ratio recurrence is evaluated downward from ``k``, so every term
    is non-increasing and the sum cannot overflow.
    """
    log_comb = 0.0
    for i in range(1, k + 1):
        log_comb += math.log(n - i + 1)
    log_comb -= math.lgamma(k + 1)
    n_log_q = math.log(n) + math.log(-log_q)
    n_q_term = -math.exp(n_log_q) if n_log_q <= 709.0 else -math.inf
    log_term = log_comb + k * log_p + n_q_term - k * log_q
    if not math.isfinite(log_term):
        return log_term
    term = 1.0
    total = 1.0
    i = k
    while i < n:
        log_ratio = (math.log(n - i) - math.log(i + 1)
                     + log_p - log_q)
        term *= math.exp(log_ratio)
        total += term
        i += 1
        if term <= total * 1e-18:
            break
    return log_term + math.log(total)


def _binomial_log_upper_tail_down(n: int, k: int, log_s: float,
                                  log_f: float) -> float:
    """Stable upper tail for a start point above the mode.

    ``log_s``/``log_f`` are the log success and failure probabilities.
    Terms decay from ``k``, so this is used for the mirrored binomial when
    the original start point lies below its mode.
    """
    if k <= 0:
        return 0.0
    if k >= n:
        return _scale_int_by_float(n, log_s)
    if k <= 200_000:
        log_comb = 0.0
        for i in range(1, k + 1):
            log_comb += math.log(n - i + 1) - math.log(i)
    else:
        try:
            log_comb = (math.lgamma(n + 1) - math.lgamma(k + 1)
                        - math.lgamma(n - k + 1))
        except OverflowError:
            log_comb = 0.0
            for i in range(1, k + 1):
                log_comb += math.log(n - i + 1) - math.log(i)
    log_term = (log_comb + _scale_int_by_float(k, log_s)
                + _scale_int_by_float(n - k, log_f))
    if not math.isfinite(log_term):
        return log_term
    term = 1.0
    total = 1.0
    i = k
    while i < n:
        log_ratio = (math.log(n - i) - math.log(i + 1)
                     + log_s - log_f)
        term *= math.exp(log_ratio)
        total += term
        i += 1
        if term <= total * 1e-18:
            break
    return log_term + math.log(total)


def _binomial_log_lower_tail_below_mean(n: int, j: int, log_p: float,
                                        log_q: float) -> float:
    """``log P(X <= j)`` for ``j`` below the binomial mean.

    Lagging terms are summed relative to ``q**n`` (so a tiny ``q**n`` does
    not erase a moderate lower tail), or through the mirrored upper tail
    when ``j`` is in the upper half.
    """
    if j < 0:
        return -math.inf
    if j >= n:
        return 0.0
    if j == 0:
        bound = _scale_int_by_float(n, log_q)
    else:
        x = j / n
        p = math.exp(log_p)
        q = math.exp(log_q)
        if 0.0 < x < p and q > 0.0:
            d = (x * math.log(x / p)
                 + (1.0 - x) * math.log((1.0 - x) / q))
            bound = _scale_int_by_float(n, -d)
        else:
            bound = 0.0
    if bound < _MIN_LOG_PROB:
        # The lower tail is below the smallest positive float, so the
        # survival probability rounds to exactly one.
        return -math.inf
    if j <= n - j:
        log_comb = 0.0
        term = 0.0
        max_term = 0.0
        scaled = 1.0
        for i in range(1, j + 1):
            log_comb += math.log(n - i + 1) - math.log(i)
            term = log_comb + i * (log_p - log_q)
            if term > max_term:
                scaled = scaled * math.exp(max_term - term) + 1.0
                max_term = term
            else:
                scaled += math.exp(term - max_term)
        return (_scale_int_by_float(n, log_q) + max_term
                + math.log(scaled))
    kk = n - j
    return _binomial_log_upper_tail_down(n, kk, log_q, log_p)


def _binomial_log_sf(n: int, k: int, p: float) -> float:
    """``log P(Binomial(n, p) >= k)`` for a small ``k``.

    Sums the upper tail directly in log space.  The lower-tail root
    ``P(X <= k-1) = 1 - alpha/2`` cannot be solved by bisection for
    sub-epsilon ``alpha`` because ``1 - alpha/2`` rounds to exactly 1.0;
    this survival-function form keeps the requested tail probability.
    Terms are evaluated by the ratio recurrence, so no huge log-gamma
    difference or ``n * log1p(-p)`` product is formed.  When ``k`` lies
    below the mode the direct recurrence grows through the mode and can
    overflow; that case is evaluated as the complement of the lower tail.
    """
    if k <= 0:
        return 0.0
    if k > n:
        return -math.inf
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return 0.0
    if k == n:
        return _scale_int_by_float(n, math.log(p))
    log_p = math.log(p)
    log_q = math.log1p(-p)
    if math.log(k) < math.log(n) + log_p:
        downward = _binomial_log_sf_downward(n, k, log_p, log_q)
        if math.isfinite(downward) and downward <= 0.0:
            return downward
        lower = _binomial_log_lower_tail_below_mean(
            n, k - 1, log_p, log_q)
        if lower >= 0.0:
            lower = 0.0
        if lower == -math.inf:
            return 0.0
        return math.log1p(-math.exp(lower))
    return _binomial_log_sf_downward(n, k, log_p, log_q)


def _binomial_sf_root(n: int, k: int, target: float, *,
                      log_target: float | None = None) -> float:
    """Solve ``P(Binomial(n, p) >= k) = target`` for a small ``k``."""
    if log_target is None:
        if target <= 0.0:
            return 0.0
        if target >= 1.0:
            return 1.0
        log_target = math.log(target)
    else:
        if log_target == -math.inf:
            return 0.0
        if log_target >= 0.0:
            return 1.0
    log_hi = math.log(k) - math.log(n)
    if log_hi < _MIN_LOG_PROB:
        # The root is below the smallest positive float.
        return 0.0
    lo = _MIN_LOG_PROB
    hi = log_hi
    if _binomial_log_sf(n, k, math.exp(lo)) > log_target:
        return 0.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _binomial_log_sf(n, k, math.exp(mid)) > log_target:
            hi = mid
        else:
            lo = mid
    return math.exp(0.5 * (lo + hi))


def _normal_beta_quantile(q: float, a: int, b: int, *,
                          log_q: float | None = None) -> float:
    """Normal-moment approximation to a Beta quantile for huge shapes.

    The continued fraction used by :func:`betainc_regularized` loses its
    scale when ``a`` and ``b`` are both around 1e9 (the first denominator
    ``1 - (a+b)x/(a+1)`` rounds to zero).  At that size the Beta
    distribution is extremely well approximated by a normal with the exact
    Beta mean and variance, so use that stable limit instead of emitting a
    wrongly narrow interval or an OverflowError.
    """
    total = a + b
    mean = a / total
    if q <= 0.0:
        if log_q is None:
            return 0.0
        z = -_normal_upper_quantile_from_log_tail(log_q)
    else:
        if q >= 1.0:
            return 1.0
        z = NormalDist().inv_cdf(q)
    if q >= 1.0:
        return 1.0
    total_float = _float_count(total)
    if total_float is None:
        # The variance (at most 1/(4*(total+1))) is below the smallest
        # representable float, so its quantile offset underflows to zero.
        # Returning the exact mean is the closest representable bound.
        return mean
    variance = mean * (1.0 - mean) / (total_float + 1.0)
    if variance <= 0.0:
        return mean
    return min(max(mean + z * math.sqrt(variance), 0.0), 1.0)


def _small_tail_proportion_ci(k: int, n: int, alpha: float) -> tuple:
    tail = alpha / 2.0
    if tail == 0.0:
        # Halving a subnormal alpha underflows. Use the representable log
        # of the requested tail rather than treating it as exactly zero,
        # which collapsed a valid interval endpoint to 0 or 1.
        log_tail = math.log(alpha) - math.log(2.0)
        low_args = {'log_target': log_tail}
        if k <= n - k:
            low = 0.0 if k == 0 else _binomial_sf_root(
                n, k, 0.0, **low_args)
            high = 1.0 if k == n else _binomial_tail_root(
                n, k, 0.0, **low_args)
            return low, high
        kk = n - k
        low = 1.0 - (1.0 if kk == n else _binomial_tail_root(
            n, kk, 0.0, **low_args))
        high = 1.0 - (0.0 if kk == 0 else _binomial_sf_root(
            n, kk, 0.0, **low_args))
        return low, high
    if k <= n - k:
        low = 0.0 if k == 0 else _binomial_sf_root(
            n, k, tail)
        high = 1.0 if k == n else _binomial_tail_root(
            n, k, tail)
        return low, high
    kk = n - k
    low = 1.0 - (1.0 if kk == n else _binomial_tail_root(
        n, kk, tail))
    high = 1.0 - (0.0 if kk == 0 else _binomial_sf_root(
        n, kk, tail))
    return low, high


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
    if n > 1000 and min(k, n - k) <= _SMALL_BINOMIAL_TAIL:
        low, high = _small_tail_proportion_ci(k, n, alpha)
    elif n > _LARGE_BINOMIAL_N:
        tail = alpha / 2.0
        log_tail = (math.log(alpha) - math.log(2.0)
                    if tail == 0.0 else None)
        low = (0.0 if k == 0 else
               _normal_beta_quantile(tail, k, n - k + 1,
                                     log_q=log_tail))
        # Upper bound by symmetry: BetaInv(1 - q; a, b) =
        # 1 - BetaInv(q; b, a). This avoids ``1 - alpha/2`` rounding to
        # exactly 1.0 (which returned a spurious upper bound of 1.0).
        high = (1.0 if k == n else 1.0 - _normal_beta_quantile(
            tail, n - k, k + 1, log_q=log_tail))
    else:
        tail = alpha / 2.0
        if tail == 0.0:
            # ``alpha`` is a subnormal float, so halving it underflowed to
            # zero. Bisecting on q=0 collapses the interval to [0, 1];
            # keep the representable log tail instead.
            log_tail = math.log(alpha) - math.log(2.0)

            def _quantile(a, b):
                return _beta_quantile_from_log(log_tail, a, b)
        else:
            def _quantile(a, b):
                return _beta_quantile(tail, a, b)
        low = 0.0 if k == 0 else _quantile(k, n - k + 1)
        high = 1.0 if k == n else 1.0 - _quantile(n - k, k + 1)
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

    def _log_tail(k: int) -> float:
        return _log_betainc_regularized(n - k + 1, k, 0.5)

    k_best = 0
    tail_best = None
    lo, hi = 1, (n + 1) // 2
    # A relative (not absolute) slack absorbs the last-ulp error of the
    # incomplete-beta tail without letting a tiny alpha be dominated by a
    # fixed 1e-15 epsilon: for alpha below that scale the old check accepted
    # every tail computed as <= 1e-15 and returned an interval whose true
    # coverage was far below 1 - alpha.
    threshold = alpha / 2.0
    if threshold > 0.0:
        log_threshold = math.log(threshold)
    else:
        log_threshold = math.log(alpha) - math.log(2.0)
    log_slack = math.log1p(1e-12)
    while lo <= hi:
        mid = (lo + hi) // 2
        if _log_tail(mid) <= log_threshold + log_slack:
            k_best, tail_best = mid, _tail(mid)
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
        try:
            log_floor = math.log(sided) - n_eff * math.log(2.0)
        except OverflowError:
            # A JSON-sized count cannot be converted to float. Its floor is
            # far below the smallest representable float.
            log_floor = -math.inf
        p_min = 1.0 if log_floor >= 0.0 else math.exp(log_floor)
    if p_min <= 0.0:
        # The exact floor ``sided / 2**n_eff`` underflowed to 0.0 for a very
        # large n_eff, but ``family_size * floor`` can still be a perfectly
        # representable p-value (e.g. n_eff=2000 with an enormous family).
        # Evaluate that product in log space instead of reporting a p of
        # exactly zero: the floor is small, not zero.
        try:
            log_product = (math.log(family_size) + math.log(sided)
                           - n_eff * math.log(2.0))
        except OverflowError:
            log_product = -math.inf
        p_holm = 1.0 if log_product >= 0.0 else math.exp(log_product)
    else:
        # ``family_size`` is an arbitrary Python int: converting a very large
        # one to float (implicitly, in ``family_size * p_min``) raises
        # OverflowError. Fall back to log-space only for that case, so the
        # ordinary integral products keep their exact float arithmetic.
        try:
            p_holm = min(1.0, family_size * p_min)
        except OverflowError:
            try:
                log_product = math.log(family_size) + math.log(p_min)
            except OverflowError:
                log_product = -math.inf
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
    # The largest supported family is floor(alpha * 2**n_eff / sided).
    # Build it with exact integer/Fraction arithmetic for every practical
    # n_eff, not only when the float floor underflowed: for n_eff in
    # 1024..1030 ``p_min`` is a rounded exp(log_floor), so ``alpha / p_min``
    # inherits its relative error and lands one comparison below the exact
    # boundary. Refuse only a shift so large that the integer itself is
    # absurd.
    if n_eff <= 1_000_000:
        max_family = int(
            (alpha_frac * (1 << n_eff) / sided).__floor__())
    elif p_min <= 0.0:
        max_family = None
    else:
        ratio = alpha / p_min
        if math.isfinite(ratio):
            max_family = int(math.floor(ratio))
        else:
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


def _clean_seed_cell(cell, label, *,
                     caller: str = 'interaction_contrast'
                     ) -> dict[int, float]:
    """Validate one ``seed -> metric`` cell of a paired factorial design.

    The cell is the unit the runner stores as a per-seed result block. It
    must be keyed by a non-negative integer seed and carry only finite real
    values: a malformed seed or a diverged metric must be filtered by the
    caller (and reported as attrition) rather than silently dropped here,
    because dropping one cell and not its partners would break the pairing.
    """
    if not isinstance(cell, dict):
        raise ValueError(
            f'cell {label!r} must be a seed -> metric mapping, got '
            f'{type(cell).__name__}')
    out: dict[int, float] = {}
    for seed, value in cell.items():
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                f'cell {label!r} has a non-negative-integer seed key '
                f'{seed!r}; filter malformed rows before calling '
                f'{caller}')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f'cell {label!r}[seed {seed}]={value!r} is not a real '
                f'number; filter diverged seeds before calling '
                f'{caller}')
        try:
            value = float(value)
        except OverflowError:
            raise ValueError(
                f'cell {label!r}[seed {seed}]={value!r} is not '
                f'representable as a finite float') from None
        if not math.isfinite(value):
            raise ValueError(
                f'cell {label!r}[seed {seed}]={value!r} is not finite; '
                f'filter diverged seeds before calling '
                f'{caller}')
        out[seed] = value
    return out


def interaction_contrast(cell_a, cell_b, cell_c, cell_d, *,
                         labels=('a', 'b', 'c', 'd')) -> dict:
    """Seed-paired 2x2 difference-in-differences (interaction) contrast.

    Given four cells of a 2x2 design — ``a`` = neither factor, ``b`` =
    factor 1 only, ``c`` = factor 2 only, ``d`` = both — the interaction
    is estimated **per seed** as::

        I_i = (d_i - b_i) - (c_i - a_i) = d_i - b_i - c_i + a_i

    and then averaged over the seeds common to all four cells. The pairing
    is what makes this a within-seed contrast: every term in ``I_i`` comes
    from the same training seed, so seed-to-seed variance in task
    difficulty cancels. Testing ``b - a`` and ``d - c`` separately and
    comparing their p-values is **not** an interaction test — the two
    estimates are correlated, and neither p-value bounds the difference
    between them.

    This is the estimand behind claims such as "the CSA benefit transfers
    across memory routes" (route x layout interaction), "removing RoPE
    hurts more under dense selection" (rope x sparsity interaction), or
    "the two operators are complementary" (``I = D - B - C + A``). A
    non-significant interaction is inconclusive; it is not evidence of
    additivity.

    ``labels`` names the four cells in the returned ``cell_means`` /
    ``cell_n`` blocks (in a/b/c/d order). Inputs are validated like every
    other function here: non-integer seeds, booleans, non-finite values
    and non-numeric metrics raise ``ValueError`` instead of poisoning the
    report. Fewer than two common seeds yields a JSON-safe record with
    ``mean_diff=None`` and ``n_pairs < 2`` rather than a fabricated
    estimate.

    Returns a JSON-safe dict with the per-seed definition, the four cell
    means and sample sizes, the common seed set, the mean/SD of the
    interaction differences, the paired t statistic (``None`` when the
    differences are all identical), Cohen's ``d_z``, and a
    ``unanimous`` / ``direction`` description.
    """
    if (isinstance(labels, (str, bytes))
            or not hasattr(labels, '__len__')
            or len(labels) != 4):
        raise ValueError(
            f"labels must be four names in (a, b, c, d) order, got "
            f"{labels!r}")
    labels = tuple(str(x) for x in labels)
    cells = {
        'a': _clean_seed_cell(cell_a, labels[0]),
        'b': _clean_seed_cell(cell_b, labels[1]),
        'c': _clean_seed_cell(cell_c, labels[2]),
        'd': _clean_seed_cell(cell_d, labels[3]),
    }
    common = sorted(set.intersection(*(set(c) for c in cells.values())))
    n = len(common)
    record = {
        'definition': ('interaction I = (d - b) - (c - a) = '
                       'd - b - c + a, paired by seed'),
        'labels': list(labels),
        'n_pairs': n,
        'common_seeds': common,
        'cell_means': {
            key: _safe_mean([cells[key][s] for s in common])
            if common else None
            for key in ('a', 'b', 'c', 'd')},
        'cell_n': {key: len(cells[key]) for key in ('a', 'b', 'c', 'd')},
        'mean_diff': None,
        'sd_diff': None,
        't_stat': None,
        'dof': None,
        'effect_size_cohens_dz': None,
        'unanimous': None,
        'direction': 'none',
        'numerically_zero_tolerance': None,
        'note': ('fewer than two seeds are common to all four cells; no '
                 'paired interaction is estimable'),
    }
    if n < 2:
        return record
    # ``d - b - c + a`` can overflow to +-inf for finite cells even when
    # the true interaction is representable (e.g. all four cells near
    # 1e308). Keep the historical left-to-right arithmetic for every
    # in-range seed; only an overflowing seed falls back to a scaled
    # evaluation, which is the same difference up to round-off.
    diffs = []
    diff_scale = max(
        abs(v) for cell in cells.values() for v in cell.values())
    for s in common:
        diff = (cells['d'][s] - cells['b'][s]
                - cells['c'][s] + cells['a'][s])
        if not math.isfinite(diff):
            if diff_scale == 0.0:
                diff = 0.0
            else:
                diff = diff_scale * math.fsum([
                    cells['d'][s] / diff_scale,
                    -cells['b'][s] / diff_scale,
                    -cells['c'][s] / diff_scale,
                    cells['a'][s] / diff_scale,
                ])
        diffs.append(diff)
    diffs_finite = all(math.isfinite(d) for d in diffs)
    if diffs_finite:
        mean = _safe_mean(diffs)
    else:
        # At least one per-seed interaction exceeds the float range. The
        # MEAN can still be representable through cancellation across
        # seeds, so evaluate the aggregate in the same scaled space and
        # report the SD/effect size as undefined.
        mean = diff_scale * (
            math.fsum(
                cells['d'][s] / diff_scale
                - cells['b'][s] / diff_scale
                - cells['c'][s] / diff_scale
                + cells['a'][s] / diff_scale
                for s in common)
            / n)
        if not math.isfinite(mean):
            mean = None
    sd = (_sample_sd(diffs, mean)
          if mean is not None and diffs_finite else None)
    t_stat = (mean / (sd / math.sqrt(n))) if sd and sd > 0.0 else None
    if all(x > 0 for x in diffs):
        unanimous = 'positive'
    elif all(x < 0 for x in diffs):
        unanimous = 'negative'
    else:
        unanimous = 'mixed'
    # A difference-in-differences of values that are themselves rounded
    # can be a few ULPs away from an exact zero even when the design is
    # additive by construction. Do not report that residue as a direction:
    # the tolerance scales with the largest cell value, so it is a
    # relative-precision floor, not an absolute effect-size threshold.
    cell_scale = max(abs(v) for cell in cells.values()
                     for v in cell.values()) if common else 0.0
    zero_tol = (4.0 * math.ulp(cell_scale)) if cell_scale > 0.0 else 0.0
    if mean is None:
        direction = 'none'
    elif abs(mean) <= zero_tol:
        direction = 'none'
    else:
        direction = 'greater' if mean > 0 else 'less'
    record.update({
        'mean_diff': mean,
        'sd_diff': sd,
        't_stat': t_stat,
        'dof': n - 1,
        'effect_size_cohens_dz': (
            mean / sd if mean is not None and sd and sd > 0.0 else None),
        'unanimous': unanimous,
        'direction': direction,
        'numerically_zero_tolerance': zero_tol,
        'note': (
            'interaction is a within-seed difference-in-differences; a '
            'non-significant I does not establish additivity, and a '
            'significant I is the only evidence for a non-additive '
            'combination. Use the paired t / exact sign-flip test with the '
            'same multiplicity correction as the main contrasts.'
            if mean is not None else
            'the interaction exceeds the representable float range; '
            'mean_diff and the paired statistics are undefined'),
    })
    return record


def _f_survival(f_stat, df1: int, df2: int) -> float | None:
    """Upper-tail probability ``P(F > f_stat)`` for the F distribution."""
    if df1 <= 0 or df2 <= 0:
        raise ValueError(
            f'F survival requires positive dof, got {df1!r}, {df2!r}')
    if isinstance(f_stat, bool):
        raise ValueError(f'f_stat must be a real number, got {f_stat!r}')
    try:
        f_stat = float(f_stat)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(f_stat):
        return None
    if f_stat == math.inf:
        return 0.0
    if f_stat <= 0.0:
        return 1.0
    denom = df2 + df1 * f_stat
    if not math.isfinite(denom):
        return 0.0
    x = df2 / denom
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return betainc_regularized(0.5 * df2, 0.5 * df1, x)


def _cross_cell_heterogeneity(cells, names, common) -> dict:
    """Repeated-measures cell x seed F test on a balanced cell matrix."""
    k = len(names)
    n = len(common)
    if k < 2 or n < 2:
        return {
            'method': 'repeated_measures_cell_x_seed_f_test',
            'F': None,
            'df1': None,
            'df2': None,
            'p_value': None,
            'between_cell_sd': None,
            'within_seed_sd': None,
            'zero_residual_variance': None,
            'sphericity_assumed': True,
            'note': ('fewer than two cells or two common seeds; '
                     'heterogeneity is not estimable'),
        }
    values = [cells[name][s] for s in common for name in names]
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        scaled = {(s, name): 0.0 for s in common for name in names}
    else:
        scaled = {(s, name): cells[name][s] / scale
                  for s in common for name in names}
    cell_means = {
        name: _safe_mean([scaled[(s, name)] for s in common])
        for name in names}
    seed_means = {
        s: _safe_mean([scaled[(s, name)] for name in names])
        for s in common}
    grand = _safe_mean(
        [scaled[(s, name)] for s in common for name in names])
    ss_cells = n * math.fsum(
        (cell_means[name] - grand) ** 2 for name in names)
    ss_error = math.fsum(
        (scaled[(s, name)] - seed_means[s] - cell_means[name] + grand) ** 2
        for s in common for name in names)
    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    original_means = {
        name: _safe_mean([cells[name][s] for s in common])
        for name in names}
    original_grand = _safe_mean(
        [cells[name][s] for s in common for name in names])
    between_sd = _sample_sd(
        [original_means[name] for name in names], original_grand)
    # Constant cell offsets leave only last-bit residue in ``ss_error``.
    # Treat that residue as zero instead of reporting an astronomical F
    # from round-off; the tolerance is on the unit-scaled matrix.
    residual_tol = 64.0 * math.ulp(1.0) * max(1.0, ss_cells + n * k)
    zero_residual = ss_error <= residual_tol
    if zero_residual:
        f_stat = None
        p_value = 0.0 if ss_cells > 0.0 else 1.0
        within_sd = 0.0
    else:
        f_stat = (ss_cells / df1) / (ss_error / df2)
        if not math.isfinite(f_stat):
            f_stat = None
            p_value = 0.0
        else:
            p_value = _f_survival(f_stat, df1, df2)
        within_sd = scale * math.sqrt(ss_error / df2)
        if not math.isfinite(within_sd):
            within_sd = None
    return {
        'method': 'repeated_measures_cell_x_seed_f_test',
        'F': f_stat,
        'df1': df1,
        'df2': df2,
        'p_value': p_value,
        'between_cell_sd': between_sd,
        'within_seed_sd': within_sd,
        'zero_residual_variance': bool(zero_residual),
        'sphericity_assumed': True,
        'note': (
            'Descriptive repeated-measures check: a large F means the '
            'cell means differ by more than the within-seed cell-to-cell '
            'variation. The F test assumes sphericity (exact for two '
            'cells; potentially anti-conservative for more) and is not '
            'the replication test. The seed-level paired panel in '
            'primary is the inferential result.'),
    }


def cross_cell_paired_analysis(cells, *, alpha: float = 0.05,
                               seed: int = 0, n_boot: int = 10000,
                               n_permutations: int = 10000,
                               max_exact: int = 16,
                               family_size: int = 1,
                               min_cells: int = 2,
                               include_seed_influence: bool = False) -> dict:
    """Replicate one paired contrast across several cells (tasks/datasets).

    ``cells`` maps a cell name to a ``seed -> paired difference`` mapping,
    where the difference is oriented so positive means the first operator
    is better. The per-cell analysis in ``run_quality`` tests each
    ``(task, n_kv)`` cell in its own Holm family. That is the right unit
    for a cell-specific claim, but it also means a reader can select the
    cell where an operator ordering happens to be significant. This
    function asks the complementary question: does the ordering hold when
    every pre-specified cell is considered together?

    The primary test is **seed-balanced**, not observation-weighted: for
    each seed present in every cell, average that seed's paired difference
    across cells, then run the ordinary paired panel
    (:func:`summarize_paired`) on the averaged differences. The seed is
    the experimental unit, so a cell with more surviving seeds cannot
    dominate, and the cells are not treated as independent studies. The
    estimand is the average paired difference across the included cells;
    a cell-specific effect must still be read from the per-cell panel.

    Two descriptive checks guard against over-reading the pooled mean:

    * ``mixed_direction`` — the per-cell mean differences do not all share
      a sign, so the pooled average is not a replicated ordering.
    * ``aggregation_reversal`` — the naive observation-weighted mean (all
      cell/seed pairs) has the opposite sign from the seed-balanced mean.
      This is the Simpson-style aggregation failure: pooling raw rows
      across unequal cells can reverse the effect.

    ``heterogeneity`` is a balanced repeated-measures F test for a
    cell x seed interaction. It is descriptive (it assumes sphericity),
    not a confirmatory test.

    Only seeds present in every cell enter the primary test; seeds dropped
    by that balance requirement are listed explicitly. A contrast seen in
    fewer than ``min_cells`` cells is labelled ``insufficient_cells`` and
    must not be presented as a replication. The caller remains
    responsible for multiplicity correction across contrasts; the
    ``run_quality`` integration applies Holm across the cross-cell family.
    """
    alpha = _validate_alpha(alpha)
    if (isinstance(family_size, bool) or not isinstance(family_size, int)
            or family_size < 1):
        raise ValueError(
            f'family_size must be a positive int, got {family_size!r}')
    if (isinstance(min_cells, bool) or not isinstance(min_cells, int)
            or min_cells < 2):
        raise ValueError(
            f'min_cells must be an int >= 2, got {min_cells!r}')
    if hasattr(cells, 'items'):
        items = list(cells.items())
    else:
        items = list(cells)
    cleaned: dict[str, dict[int, float]] = {}
    for name, cell in items:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f'cell names must be non-empty strings, got {name!r}')
        if name in cleaned:
            raise ValueError(f'duplicate cell name {name!r}')
        cleaned[name] = _clean_seed_cell(
            cell, name, caller='cross_cell_paired_analysis')
    names = list(cleaned)
    k = len(names)
    cell_summaries = {}
    all_diffs = []
    for name in names:
        vals = list(cleaned[name].values())
        all_diffs.extend(vals)
        mean = _safe_mean(vals) if vals else None
        sd = (_sample_sd(vals, mean)
              if mean is not None and len(vals) >= 2 else None)
        cell_summaries[name] = {
            'n_pairs': len(vals),
            'mean_diff': mean,
            'sd_diff': sd,
            'direction': ('none' if mean is None or mean == 0.0
                          else ('greater' if mean > 0 else 'less')),
        }
    if k:
        common = sorted(set.intersection(
            *(set(cell) for cell in cleaned.values())))
        all_seeds = sorted(set().union(
            *(set(cell) for cell in cleaned.values())))
    else:
        common = []
        all_seeds = []
    common_set = set(common)
    dropped = [s for s in all_seeds if s not in common_set]
    n_common = len(common)
    naive_pooled = _safe_mean(all_diffs) if all_diffs else None
    seed_balanced = None
    primary = None
    if n_common >= 2:
        seed_diffs = [
            _safe_mean([cleaned[name][s] for name in names])
            for s in common]
        seed_balanced = _safe_mean(seed_diffs)
        if k >= min_cells:
            primary = summarize_paired(
                seed_diffs, seed=seed, n_boot=n_boot,
                n_permutations=n_permutations, max_exact=max_exact,
                alpha=alpha, family_size=family_size,
                include_seed_influence=include_seed_influence)
    cell_means = [cell_summaries[name]['mean_diff'] for name in names]
    n_pos = sum(1 for m in cell_means if m is not None and m > 0.0)
    n_neg = sum(1 for m in cell_means if m is not None and m < 0.0)
    n_zero = sum(1 for m in cell_means if m is not None and m == 0.0)
    if seed_balanced is None or seed_balanced == 0.0:
        same_direction = 0
    elif seed_balanced > 0.0:
        same_direction = n_pos
    else:
        same_direction = n_neg
    all_same = bool(k and (n_pos == k or n_neg == k))
    mixed = bool(n_pos and n_neg)
    reversal = bool(
        naive_pooled is not None and seed_balanced is not None
        and naive_pooled != 0.0 and seed_balanced != 0.0
        and (naive_pooled > 0.0) != (seed_balanced > 0.0))
    if k < min_cells:
        status = 'insufficient_cells'
    elif n_common < 2:
        status = 'insufficient_balanced_seeds'
    elif seed_balanced is None:
        status = 'undefined'
    elif reversal:
        status = 'aggregation_reversal'
    elif mixed:
        status = 'mixed_direction'
    else:
        status = 'consistent_direction'
    return {
        'method': 'seed_balanced_cross_cell_paired_analysis',
        'definition': (
            'for each seed common to every cell, average the paired '
            'differences across cells; test the averaged differences '
            'across seeds with the paired panel'),
        'n_cells': k,
        'cell_names': names,
        'min_cells': min_cells,
        'n_seeds_common': n_common,
        'common_seeds': common,
        'n_seeds_dropped': len(dropped),
        'dropped_seeds': dropped,
        'n_observations': len(all_diffs),
        'cells': cell_summaries,
        'cell_direction_counts': {
            'positive': n_pos,
            'negative': n_neg,
            'zero': n_zero,
            'same_as_seed_balanced': same_direction,
        },
        'all_cells_same_direction': all_same,
        'mixed_direction': mixed,
        'naive_pooled_mean_diff': naive_pooled,
        'seed_balanced_mean_diff': seed_balanced,
        'aggregation_reversal': reversal,
        'heterogeneity': _cross_cell_heterogeneity(
            cleaned, names, common),
        'primary': primary,
        'status': status,
        'note': (
            'Cells share the same training seeds, so this is not a '
            'meta-analysis of independent studies. Only seeds present in '
            'every cell enter the primary test; the estimand is the '
            'average paired difference across the included cells, not a '
            'cell-specific effect. mixed_direction or '
            'aggregation_reversal forbids reading the pooled mean as a '
            'replicated ordering. Significance requires the caller\'s '
            'multiplicity correction on primary.'),
    }


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
    assignment is enumerated (exact test), up to a hard resource cap of
    ``2**20`` assignments; a larger ``max_exact`` cannot turn an otherwise
    Monte-Carlo-sized seed count into an unbounded enumeration. For larger
    n a deterministic Monte-Carlo sample is drawn from a *local* RNG
    seeded with ``seed`` (the global RNG stream is never touched), with
    the standard +1 correction so the p-value can never be 0.

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

    if n <= max_exact and n <= _MAX_EXACT_PERMUTATION_BITS:
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
    se = sd / _sqrt_count(n_pairs) if sd > 0.0 else 0.0
    if se == 0.0 or _float_count(dof) is None:
        # Every per-seed difference is identical (numerically), or the seed
        # count is so large that its standard error underflows. The
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
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = student_t_sf(t_lower, dof)
    p_upper = student_t_cdf(t_upper, dof)
    p = min(1.0, max(0.0, max(p_lower, p_upper)))
    half = _positive_t_quantile(alpha, dof) * se
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
    if sd is None:
        # The sample SD exceeds the float range: the supported margin is
        # unrepresentable, not the zero-variance infimum.
        return None
    if sd <= 0.0:
        return abs(mean)
    se = sd / math.sqrt(n)
    supported = abs(mean) + _positive_t_quantile(alpha, n - 1) * se
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


def _normal_upper_quantile(tail: float) -> float:
    """Standard-normal ``z`` with upper-tail probability ``tail``.

    ``NormalDist.inv_cdf(1 - tail)`` loses all tail precision once
    ``tail`` is below the float spacing at 1 (and raises for tail < 5e-17
    because ``1 - tail`` rounds to exactly 1.0). Mirroring the tail first
    keeps the relative accuracy for sub-epsilon alphas.
    """
    if tail <= 0.0:
        return math.inf
    if tail >= 1.0:
        return -math.inf
    if tail >= 1e-4:
        return _ND.inv_cdf(1.0 - tail)
    return -_ND.inv_cdf(tail)


def _log_normal_upper_tail(z: float) -> float:
    """``log P(Z > z)`` for large positive ``z`` without erfc underflow."""
    if z < 30.0:
        return math.log(0.5 * math.erfc(z / math.sqrt(2.0)))
    z2 = z * z
    inv = 1.0 / z2
    series = (1.0 - inv + 3.0 * inv ** 2 - 15.0 * inv ** 3
              + 105.0 * inv ** 4 - 945.0 * inv ** 5)
    return (-0.5 * z2 - math.log(z) - 0.5 * math.log(2.0 * math.pi)
            + math.log(series))


def _normal_upper_quantile_from_log_tail(log_tail: float) -> float:
    """Standard-normal ``z`` for an upper-tail probability given in logs."""
    if log_tail >= 0.0:
        return -math.inf
    if log_tail == -math.inf:
        return math.inf
    if math.exp(log_tail) > 0.0:
        return _normal_upper_quantile(math.exp(log_tail))
    lo, hi = 0.0, 1.0
    while _log_normal_upper_tail(hi) > log_tail:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _log_normal_upper_tail(mid) > log_tail:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _normal_twosided_critical(alpha: float) -> float:
    """``z`` with two-sided normal tail ``alpha``, even if alpha/2 underflows."""
    half = alpha / 2.0
    if half > 0.0:
        return _normal_upper_quantile(half)
    return _normal_upper_quantile_from_log_tail(
        math.log(alpha) - math.log(2.0))


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
    z_crit = (_normal_twosided_critical(alpha)
              if alternative == 'two-sided'
              else _normal_upper_quantile(alpha))
    z_power = _ND.inv_cdf(power)
    value = (z_crit + z_power) * sd / _sqrt_count(n)
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
    z_crit = (_normal_twosided_critical(alpha)
              if alternative == 'two-sided'
              else _normal_upper_quantile(alpha))
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
    sqrt_n = _sqrt_count(n)
    se = float(sd) / sqrt_n
    if se == 0.0:
        # The standard error underflows for an enormous seed count; the
        # normal power is at its limiting value instead of 0/0.
        if effect == 0.0:
            return float(alpha)
        if alternative == 'two-sided':
            return 1.0
        if alternative == 'greater':
            return 1.0 if effect > 0.0 else 0.0
        return 1.0 if effect < 0.0 else 0.0
    ncp = effect / se
    if alternative == 'two-sided':
        z_crit = _normal_twosided_critical(alpha)
        power = ((1.0 - _ND.cdf(z_crit - abs(ncp)))
                 + _ND.cdf(-z_crit - abs(ncp)))
    elif alternative == 'greater':
        z_crit = _normal_upper_quantile(alpha)
        power = 1.0 - _ND.cdf(z_crit - ncp)
    else:
        z_crit = _normal_upper_quantile(alpha)
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
        except (TypeError, ValueError, OverflowError):
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
        # Two-sided t critical value via bisection on the log survival
        # function; ``alpha / 2`` underflows for a subnormal alpha.
        half_alpha = alpha / 2.0
        if half_alpha > 0.0:
            crit = _positive_t_quantile(half_alpha, n - 1)
        else:
            crit = _positive_t_quantile_from_log(
                math.log(alpha) - math.log(2.0), n - 1)
        # Scale the SD by sqrt(n) BEFORE multiplying by the critical value:
        # ``crit * sd`` can overflow even when the half-width
        # ``crit * (sd / sqrt(n))`` and both endpoints are representable.
        half = crit * (sd / math.sqrt(n))
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


def _positive_t_quantile_from_log(log_q: float, dof: int) -> float:
    if log_q >= math.log(0.5):
        return 0.0
    if log_q == -math.inf:
        return math.inf
    lo, hi = 0.0, 1.0
    while _log_student_t_sf_positive(hi, dof) > log_q:
        hi *= 2.0
        if not math.isfinite(hi):
            return math.inf
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _log_student_t_sf_positive(mid, dof) > log_q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _positive_t_quantile(q: float, dof: int) -> float:
    """Upper-tail t quantile ``x > 0`` with ``P(T > x) = q`` (q < 1/2)."""
    if q <= 0.0:
        # The requested tail is below the smallest representable
        # probability; the limiting critical value is unbounded.
        return math.inf
    return _positive_t_quantile_from_log(math.log(q), dof)


def _t_quantile(p: float, dof: int) -> float:
    """Inverse t-CDF (p in (0,1)).

    The t distribution is symmetric, so the upper half is evaluated through
    :func:`student_t_sf`: for ``p`` close to 1 the forward CDF is a tiny
    difference from 1 and loses most of its significant digits (at
    ``p=1-1e-15`` it underflowed to a 5% quantile error), while the
    survival function keeps the tail relative precision. This also makes
    the critical values used by the epsilon-level TOST/power helpers
    accurate instead of systematically too small.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f'p must be in (0,1), got {p!r}')
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_positive_t_quantile(p, dof)
    return _positive_t_quantile(1.0 - p, dof)
