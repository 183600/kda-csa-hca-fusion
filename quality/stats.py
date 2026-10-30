"""run_quality split module: stats."""

from __future__ import annotations

from statistics import NormalDist

from ._shared import *  # noqa: F401,F403


_T_PP = None
_T_ISF = None

# Above this many degrees of freedom the normal quantile is at least as
# accurate as the dependency-free t inversion (whose ``lgamma`` difference
# loses ~4e-7 at dof=1e8 and ~5e-6 at dof=1e9), while the t-vs-normal gap
# is only ~1.5e-8.  The guard is also required for huge dof: the stable
# ratio ``dof/(dof+t^2)`` rounds to 1.0 even at the median, which used to
# flatten the survival function and return values like 8.0 instead of
# ~1.645.
_NORMAL_DOF_LIMIT = 1.0e8


def _t_crit_975(n):
    """Two-sided 95% critical value of the t-distribution with ``n-1`` dof.

    Returns ``t`` such that ``P(-t < T < t) = 0.95`` where ``T`` follows a
    Student's t distribution with ``n - 1`` degrees of freedom (i.e. the
    one-sided 97.5% quantile, ``scipy.stats.t.ppf(0.975, n - 1)``).

    Resolution order:
      0. For ``n - 1 >= _NORMAL_DOF_LIMIT`` (including integers too large
         for a float) the standard-normal 97.5% quantile is returned; the
         t/normal gap is below 1e-7 there and the direct t inversion loses
         precision to ``lgamma`` cancellation.
      1. ``scipy.stats.t.ppf`` when scipy is importable (exact for any n).
      2. A hardcoded table covering ``n = 2..100`` (df = 1..99).
      3. For ``n > 100`` delegates to ``_bonferroni_crit_q(n, alpha=0.025)``
         (the exact dependency-free inverse CDF). A first-order
         Cornish-Fisher expansion remains only as a last-resort fallback
         if that path raises.
      4. For ``n < 2`` the CI is undefined; returns ``None`` (NOT ``0.0``),
         so a 1-seed estimate never becomes a zero-width CI that reads as
         "the estimate is exact". Callers MUST handle ``None`` (the
         one-sample t-test path in this file and ``run_ablation`` already
         guard with ``crit is not None`` before comparing).

    The scipy availability check runs once and is cached in ``_T_PP`` so
    repeated calls are essentially free.
    """
    global _T_PP, _T_ISF
    if n < 2:
        # The CI is undefined for n<2 (zero degrees of freedom); ``None``
        # propagates as "undefined" and the t-test guard already handles it
        # (``crit is not None and ...``).
        return None
    try:
        dof_float = float(n - 1)
    except (OverflowError, TypeError):
        # A JSON-sized integer can exceed the float range; the t limit is
        # the normal quantile and no further arithmetic is representable.
        return NormalDist().inv_cdf(0.975)
    if not math.isfinite(dof_float) or dof_float >= _NORMAL_DOF_LIMIT:
        return NormalDist().inv_cdf(0.975)
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
            _T_ISF = _t_dist.isf
        except ImportError:
            _T_PP = False
            _T_ISF = False
    if _T_PP:
        return _T_PP(0.975, n - 1)
    # Hardcoded two-sided 95% critical values, n = 2..100 (df = 1..99),
    # covering the common multi-seed CI range without falling back to the
    # normal approximation (which has ~4% relative error at n=31..100).
    # Values are scipy.stats.t.ppf(0.975, df) rounded to 3 decimals.
    _TABLE = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
        22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045, 31: 2.042,
        32: 2.040, 33: 2.037, 34: 2.035, 35: 2.032, 36: 2.030,
        37: 2.028, 38: 2.026, 39: 2.024, 40: 2.023, 41: 2.021,
        42: 2.020, 43: 2.018, 44: 2.017, 45: 2.015, 46: 2.014,
        47: 2.013, 48: 2.012, 49: 2.011, 50: 2.010, 51: 2.009,
        52: 2.008, 53: 2.007, 54: 2.006, 55: 2.005, 56: 2.004,
        57: 2.003, 58: 2.002, 59: 2.002, 60: 2.001, 61: 2.000,
        62: 2.000, 63: 1.999, 64: 1.998, 65: 1.998, 66: 1.997,
        67: 1.997, 68: 1.996, 69: 1.995, 70: 1.995, 71: 1.994,
        72: 1.994, 73: 1.993, 74: 1.993, 75: 1.993, 76: 1.992,
        77: 1.992, 78: 1.991, 79: 1.991, 80: 1.990, 81: 1.990,
        82: 1.990, 83: 1.989, 84: 1.989, 85: 1.989, 86: 1.988,
        87: 1.988, 88: 1.988, 89: 1.987, 90: 1.987, 91: 1.987,
        92: 1.986, 93: 1.986, 94: 1.986, 95: 1.986, 96: 1.985,
        97: 1.985, 98: 1.985, 99: 1.984, 100: 1.984,
    }
    if n in _TABLE:
        return _TABLE[n]
    # n > 100: delegate to the exact Student-t inverse CDF used by
    # _bonferroni_crit_q (regularized incomplete beta + bisection). The
    # exact path guarantees <1e-9 accuracy without scipy and keeps the CI
    # consistent with the Bonferroni critical values (which already use the
    # exact path).
    try:
        return _bonferroni_crit_q(n, alpha=0.025)
    except Exception:
        # Fallback to Cornish-Fisher if _bonferroni path itself fails
        # (should never happen, but keep a safe net).  For dof beyond the
        # float range the expansion's ``4.0*(n-1)`` would itself overflow;
        # the normal limit is the only representable answer.
        z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
        try:
            dof_float = float(n - 1)
        except (OverflowError, TypeError):
            return z
        if not math.isfinite(dof_float) or dof_float >= _NORMAL_DOF_LIMIT:
            return z
        return z + (z ** 3 + z) / (4.0 * dof_float)


def _bonferroni_crit_q(n, alpha=0.05):
    """Bonferroni-corrected ONE-SIDED t critical value with ``n-1`` dof.

    Resolution order (mirrors ``_t_crit_975``):
      0. For ``n - 1 >= _NORMAL_DOF_LIMIT`` (including integers too large
         for a float) the standard-normal quantile is returned; this also
         keeps the huge-dof ratio ``dof/(dof+t^2)`` from rounding to 1.0.
      1. ``scipy.stats.t.ppf`` when scipy is importable (exact for any n).
      2. For ``n`` in 2..100 a hardcoded table of the 97.5% one-sided
         quantiles (same table as ``_t_crit_975``, reused via lookup).
         Under Bonferroni correction alpha is typically divided by 20-30,
         but the table is the EXACT scipy value at the 97.5% level — for
         the corrected alphas used downstream (e.g. 0.05/28 ~= 0.0018,
         one-sided) we need the ``1-alpha`` upper-tail quantile, which is
         far in the tail. We therefore do NOT use the 97.5% table for
         corrected alphas; we fall through to the dependency-free inverse-CDF
         computation below.
      3. Dependency-free Student-t inverse CDF using the regularized
         incomplete beta function plus bisection for the requested
         ``1 - alpha`` level. This matches the one-sided ``t_stat > crit``
         comparisons in Exp4/Exp5; using ``1-alpha/2`` would be a two-sided
         threshold and silently make the test too conservative.
      4. For ``n < 2`` returns ``None`` (CI undefined — caller must guard).

    Being at module scope means ``run_ablation.py`` can import and reuse
    this function instead of re-implementing the same logic, and it is
    unit-testable in isolation.
    """
    if n < 2:
        return None
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(
            f'_bonferroni_crit_q: alpha={alpha!r} must be a finite number '
            f'in (0, 1).')
    try:
        alpha = float(alpha)
    except OverflowError:
        raise ValueError(
            f'_bonferroni_crit_q: alpha={alpha!r} must be a finite number '
            f'in (0, 1).')
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(
            f'_bonferroni_crit_q: alpha={alpha!r} must be a finite number '
            f'in (0, 1).')
    # t_{1-alpha} = -t_alpha by the symmetry of the t distribution. The
    # dependency-free fallback below only brackets the UPPER tail, so asking
    # it for a quantile below the median (alpha > 0.5) used to return ~0
    # instead of the correct negative critical value. (scipy's ppf already
    # handles the lower tail, but taking the mirror first keeps both paths
    # numerically identical and lets the fallback stay upper-tail only.)
    if alpha > 0.5:
        return -_bonferroni_crit_q(n, 1.0 - alpha)
    try:
        dof_float = float(n - 1)
    except (OverflowError, TypeError):
        dof_float = math.inf
    if not math.isfinite(dof_float) or dof_float >= _NORMAL_DOF_LIMIT:
        # The Student-t limit is the standard normal; use it for huge or
        # unrepresentable dof so the result is finite and correct instead
        # of overflowing inside scipy/lgamma.
        return -NormalDist().inv_cdf(alpha)
    # Try scipy first (exact for any alpha / any n). Use the inverse
    # survival function rather than ``ppf(1 - alpha)``: for alpha below
    # the float spacing at 1, ``1 - alpha`` rounds to exactly 1.0 and
    # scipy returns ``inf`` even when the true quantile is representable.
    global _T_PP, _T_ISF
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
            _T_ISF = _t_dist.isf
        except ImportError:
            _T_PP = False
            _T_ISF = False
    if _T_PP:
        if _T_ISF is not None:
            value = float(_T_ISF(alpha, n - 1))
        else:
            value = float(_T_PP(1 - alpha, n - 1))
        if math.isfinite(value):
            return value
    # scipy unavailable, or its tail underflowed to +/-inf even though the
    # true quantile is finite: compute the Student-t quantile directly via
    # the regularized incomplete beta function and bisection. The numerical
    # integration below is dependency-free and accurate for the small seed
    # counts used by Exp4/Exp5.
    target_p = 1.0 - alpha

    def _betacf(a, b, x):
        # Continued fraction for incomplete beta (Numerical Recipes).
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    def _betai(a, b, x):
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

    dof = n - 1
    # Compute the upper-tail probability directly from the incomplete beta
    # function.  ``1 - CDF`` is unusable here: for a large dof the CDF
    # rounds to exactly 1.0 well before the tail probability reaches alpha,
    # so an anchored bracket either never opens (RuntimeError) or stops at
    # a 1e-15 floor.  The identity
    #
    #     SF(t) = 0.5 * I_{dof/(dof+t^2)}(dof/2, 1/2)
    #
    # keeps the tail's relative precision (and its log) down to underflow,
    # which is exactly the regime of a Bonferroni-corrected alpha.
    def _log_sf(t_value):
        if t_value <= 0.0:
            return math.log(0.5)
        t_sq = t_value * t_value
        if math.isfinite(t_sq):
            denom = dof + t_sq
            if math.isfinite(denom):
                a = 0.5 * dof
                b = 0.5
                # ``y = t^2/(dof+t^2)`` is the stable complement of
                # ``x = dof/(dof+t^2)``.  When y is small the direct x
                # ratio rounds to exactly 1.0 and ``_betai`` returns 1.0,
                # flattening SF at 0.5 even for quantiles near the median
                # (large dof).  Use I_x(a,b) = 1 - I_y(b,a) there, which
                # keeps the small tail probability ``I_y`` accurate.
                y = t_sq / denom
                if y < (b + 1.0) / (a + b + 2.0):
                    ib = _betai(b, a, y)
                    if ib < 1.0:
                        return math.log(0.5) + math.log1p(-ib)
                x = dof / denom
                ib = _betai(a, b, x)
                if ib >= 2.2250738585072014e-308:
                    # Guard only against a last-bit overshoot from the
                    # continued fraction; the true regularized beta is in
                    # [0, 1].
                    return math.log(0.5) + math.log(min(ib, 1.0))
                if (x > 0.0 and x < (a + 1.0) / (a + b + 2.0)):
                    # A subnormal (or zero) ``ib`` has too few significant
                    # bits for ``log(ib)`` to locate the quantile; a zero
                    # also used to fall through to the x -> 0 asymptote,
                    # which is only valid for x near zero and returned a
                    # quantile near 107 instead of 38 for dof=9999,
                    # alpha=1e-300. Re-evaluate the SAME incomplete beta
                    # in log space on this side (where the direct
                    # continued fraction is the right branch).
                    log_bt = (
                        math.lgamma(a + b) - math.lgamma(a)
                        - math.lgamma(b)
                        + a * math.log(x) + b * math.log1p(-x))
                    h = _betacf(a, b, x)
                    if h > 0.0 and math.isfinite(log_bt):
                        log_ib = log_bt + math.log(h) - math.log(a)
                        return math.log(0.5) + log_ib
                if ib > 0.0:
                    return math.log(0.5) + math.log(min(ib, 1.0))
        # ``t**2`` overflowed or the incomplete beta underflowed. The true
        # x = dof / (dof + t^2) is then far below the smallest normal
        # float, so use the x -> 0 limit in log space:
        #     I_x(a, b) ~ x**a / (a * B(a, b)).
        # Without this fallback the bracket stopped at t ~ 1e154 and
        # returned a critical value orders of magnitude too small for
        # extreme alphas (e.g. n=2, alpha=1e-300).
        a = 0.5 * dof
        b = 0.5
        log_x = math.log(dof) - 2.0 * math.log(t_value)
        log_ib = (a * log_x - math.log(a)
                  - (math.lgamma(a) + math.lgamma(b)
                     - math.lgamma(a + b)))
        return math.log(0.5) + log_ib

    target_log_sf = math.log(alpha)  # log(1 - target_p) = log(alpha)
    # Coarse bracket: find lo, hi such that log_sf(lo) > target_log_sf >
    # log_sf(hi).  The old 1e6 cap was below the quantile for dof=1 even at
    # alpha=1e-12, and the saturation check above could not see that the
    # true tail had already crossed alpha.
    lo = 0.0
    hi = 1.0
    while _log_sf(hi) > target_log_sf:
        # Keep the last bracketing point. Starting every bisection from
        # [0, hi] leaves only absolute resolution; for a quantile near
        # 1e300 the 80 halvings cannot resolve its leading digits at all.
        lo = hi
        hi *= 2.0
        if not math.isfinite(hi) or hi > 1.0e308:
            raise ValueError(
                f"_bonferroni_crit_q: alpha={alpha!r} for n={n} needs a "
                f"critical value outside the float range")
    # Refine with log-survival bisection (80 iterations, ~1e-24 relative
    # precision on the quantile).
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _log_sf(mid) > target_log_sf:
            lo = mid
        else:
            hi = mid
    crit = 0.5 * (lo + hi)
    if not math.isfinite(crit):
        raise RuntimeError(
            f"_bonferroni_crit_q fallback produced non-finite critical value: {crit}")
    return float(crit)


def _is_finite_real(value) -> bool:
    """True for a real (non-bool) value representable as a finite float.

    JSON integers have no upper bound; a result row can therefore carry an
    integer too large for ``float()``. That is a malformed metric to skip,
    not an OverflowError that aborts the whole report.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _finite_per_seed_acc(per_seed, *, seed_key='seed',
                         acc_key='final_acc', error_key='error'):
    """Map ``seed -> finite accuracy`` for the usable per-seed records.

    A diverged seed can survive into a result row with ``final_acc`` set to
    ``NaN`` (or a string placeholder) when the training guard did not fire.
    The aggregate path filters those seeds, so the paired tests must use
    the SAME surviving population instead of treating a non-finite value as
    a measurement. Otherwise one NaN poisons every paired difference and
    aborts the whole quality report inside ``multi_seed_stats``.

    Only real finite numeric accuracies are kept; error stubs, missing
    values, bools, and stringified metrics are excluded. The returned
    values are left in their original numeric type.
    """
    out = {}
    if not isinstance(per_seed, (list, tuple)):
        return out
    for record in per_seed:
        if not isinstance(record, dict) or error_key in record:
            continue
        seed = record.get(seed_key)
        acc = record.get(acc_key)
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        if not _is_finite_real(acc):
            continue
        out[seed] = acc
    return out


def _paired_one_sided_t(diffs):
    """One-sided paired t-test statistics for an array of paired differences.

    H1: mean(diffs) > 0 (the candidate beats the reference). Returns a
    JSON-safe dict. ``t_stat`` is None when undefined (n < 2, or zero
    variance); in the zero-variance case ``all_positive`` records whether
    every seed difference was strictly positive so the caller can still
    treat a unanimous sweep as evidence.

    Zero variance is detected with ``multi_seed_stats._sample_sd``'s
    relative-tolerance rule: a perfectly constant difference vector (e.g.
    every seed improved by exactly 0.1) leaves a ~1e-17 floating-point
    residue in the naive variance, which produces a fake ``t ~ 1e16`` and
    would bypass the exact sign-flip evidence rule. The same numerical-zero
    clamp is used repo-wide.
    """
    n = len(diffs)
    if n == 0:
        return {'n_pairs': 0, 'mean_diff': None, 't_stat': None,
                'all_positive': False}
    mean_diff = _safe_mean(diffs) if n else None
    if mean_diff is None or not math.isfinite(mean_diff):
        return {'n_pairs': n, 'mean_diff': None, 't_stat': None,
                'all_positive': all(d > 0 for d in diffs)}
    if n < 2:
        return {'n_pairs': n, 'mean_diff': mean_diff, 't_stat': None,
                'all_positive': all(d > 0 for d in diffs)}
    sd = _sample_sd(diffs, mean_diff)
    if sd is None or sd <= 0.0:
        return {'n_pairs': n, 'mean_diff': mean_diff, 't_stat': None,
                'all_positive': all(d > 0 for d in diffs)}
    t_stat = mean_diff / (sd / math.sqrt(n))
    if not math.isfinite(t_stat):
        t_stat = None
    return {'n_pairs': n, 'mean_diff': mean_diff,
            't_stat': t_stat,
            'all_positive': all(d > 0 for d in diffs)}


_HYPOTHESIS_STATUSES = ('supported', 'contradicted', 'not_supported',
                        'not_computable', 'not_applicable')


def _hypothesis_verdicts(all_results, baseline_ops, baseline_ref):
    """Machine-readable verdicts for the pre-registered hypotheses.

    The repository records WHAT each cell was testing
    (``TASK_HYPOTHESES`` / ``OPERATOR_HYPOTHESES``). Each verdict below
    applies a rule fixed in code, reports its evidence, and uses one of
    five statuses:

      * ``supported``      — the prediction's signature is present;
      * ``contradicted``   — the falsification condition fired;
      * ``not_supported``  — the data are inconclusive (no significant
                             effect either way);
      * ``not_computable`` — required rows/statistics are missing;
      * ``not_applicable`` — the row is a tuned comparison reference, so
                             a superiority check does not apply to it.

    Checks (all pre-registered):

    1. ``linear_chance_control``: the memoryless ``linear`` baseline must
       stay at chance. Contradicted if it is significantly ABOVE chance in
       any cell (the task would then be solvable without token mixing,
       confounding the memory claim). A non-significant cell whose observed
       mean is still ABOVE chance is ``not_supported`` (inconclusive): a
       failed superiority test is not evidence that the control is at
       chance. Only a cell whose mean is at or below chance (or
       significantly below) supports the prediction.
    2. ``kda_nkv_monotonicity``: KDA accuracy is predicted to degrade as
       the number of key-value pairs grows. Supported when accuracy is
       non-increasing over the swept n_kv values; contradicted when a
       higher n_kv unanimously improves every paired seed.
    3. ``beats_tuned_baseline``: every complex operator must beat the
       TUNED simple baseline on the seed-paired one-sided test before a
       superiority claim is allowed. A cell is contradicted only when the
       REVERSE paired test (reference - candidate) is itself significant,
       so absence of evidence is not mislabelled as refutation.

    Returns ``(checks, counts)``: the per-rule verdict records and the
    per-row primary-status counts embedded in the run metadata.
    """
    def _cell(r):
        return f"{r.get('task', 'mqar')}/n_kv={r['n_kv']}"

    checks = OrderedDict()
    per_row = {}

    def _assign(r, name, verdict):
        per_row.setdefault(id(r), OrderedDict())[name] = verdict

    # ------------------------------------------------------------------
    # 1. Memoryless linear control.
    # ------------------------------------------------------------------
    linear_rows = [r for r in all_results
                   if r.get('op') == 'linear' and 'error' not in r
                   and r.get('mean_acc') is not None]
    evidence = []
    contradicted_cells = []
    missing_chance_cells = []
    inconclusive_cells = []
    for r in linear_rows:
        chance = r.get('chance_acc')
        sig = bool(r.get('significant_bonferroni'))
        above = chance is not None and r['mean_acc'] > chance
        if chance is None:
            cell_verdict = 'not_computable'
            prediction_holds = None
        elif sig and above:
            cell_verdict = 'contradicted'
            prediction_holds = False
        elif not above:
            # Observed mean is at or below chance (significantly or not):
            # this cell does not show the memoryless control solving the
            # task, so the prediction is supported for this cell.
            cell_verdict = 'supported'
            prediction_holds = True
        else:
            # Above chance but not significantly so: the test is
            # inconclusive. Calling this "supported" would be the classic
            # absence-of-evidence error, so record it explicitly.
            cell_verdict = 'not_supported'
            prediction_holds = None
        rec = {
            'cell': _cell(r),
            'mean_acc': float(r['mean_acc']),
            'chance_acc': chance,
            't_stat_vs_chance': r.get('t_stat_vs_chance'),
            'significant_bonferroni': sig,
            'prediction_holds': prediction_holds,
        }
        evidence.append(rec)
        if cell_verdict == 'contradicted':
            contradicted_cells.append(_cell(r))
        elif cell_verdict == 'not_computable':
            missing_chance_cells.append(_cell(r))
        elif cell_verdict == 'not_supported':
            inconclusive_cells.append(_cell(r))
        _assign(r, 'linear_chance_control', {
            'rule': ('linear must NOT be significantly above chance '
                     '(otherwise the task is solvable without token mixing)'),
            'status': cell_verdict,
            'falsified': cell_verdict == 'contradicted',
            'evidence': rec,
        })
    if not linear_rows:
        status = 'not_computable'
    elif contradicted_cells:
        status = 'contradicted'
    elif missing_chance_cells:
        status = 'not_computable'
    elif inconclusive_cells:
        # At least one cell is above chance without clearing the corrected
        # threshold: the overall "at chance" claim is inconclusive, not
        # supported.
        status = 'not_supported'
    else:
        status = 'supported'
    checks['linear_chance_control'] = {
        'rule': ('linear must NOT be significantly above chance '
                 '(otherwise the task is solvable without token mixing); '
                 'cells above chance without significance are inconclusive'),
        'status': status,
        'falsified': status == 'contradicted',
        'n_cells': len(linear_rows),
        'n_contradicting_cells': len(contradicted_cells),
        'contradicting_cells': contradicted_cells,
        'n_inconclusive_cells': len(inconclusive_cells),
        'inconclusive_cells': inconclusive_cells,
        'evidence': evidence,
    }

    # ------------------------------------------------------------------
    # 2. KDA accuracy degrades with n_kv.
    # ------------------------------------------------------------------
    kda_by_task = OrderedDict()
    for r in all_results:
        if r.get('op') != 'kda' or 'error' in r:
            continue
        if r.get('mean_acc') is None:
            continue
        kda_by_task.setdefault(r.get('task', 'mqar'), []).append(r)
    pair_evidence = []
    monotone = True
    missing_pairs = []
    unanimous_increase = []
    for task, rows in kda_by_task.items():
        rows = sorted(rows, key=lambda x: x['n_kv'])
        for lo, hi in zip(rows, rows[1:]):
            base = _finite_per_seed_acc(lo.get('per_seed', []))
            cand = _finite_per_seed_acc(hi.get('per_seed', []))
            common = sorted(set(base) & set(cand))
            diffs = [cand[s] - base[s] for s in common]
            mean_diff = (sum(diffs) / len(diffs)) if diffs else None
            all_up = bool(diffs) and all(d > 1e-12 for d in diffs)
            all_down = bool(diffs) and all(d < -1e-12 for d in diffs)
            rec = {
                'task': task,
                'n_kv_low': int(lo['n_kv']),
                'n_kv_high': int(hi['n_kv']),
                'mean_acc_low': float(lo['mean_acc']),
                'mean_acc_high': float(hi['mean_acc']),
                'mean_diff_high_minus_low': mean_diff,
                'n_pairs': len(common),
                'all_paired_seeds_increase': all_up,
                'all_paired_seeds_decrease': all_down,
            }
            pair_evidence.append(rec)
            if mean_diff is None:
                monotone = False
                missing_pairs.append(
                    f"{task}: n_kv {lo['n_kv']}->{hi['n_kv']}")
            elif mean_diff > 1e-12:
                monotone = False
            if all_up:
                unanimous_increase.append(
                    f"{task}: n_kv {lo['n_kv']}->{hi['n_kv']}")
    if len(pair_evidence) == 0:
        kda_status = 'not_computable'
    elif unanimous_increase:
        kda_status = 'contradicted'
    elif missing_pairs:
        # A missing common-seed pair means monotonicity was not evaluated
        # for every requested step; that is missing evidence, not evidence
        # that the prediction failed.
        kda_status = 'not_computable'
    elif monotone:
        kda_status = 'supported'
    else:
        kda_status = 'not_supported'
    kda_verdict = {
        'rule': ('KDA mean accuracy must be non-increasing across the swept '
                 'n_kv values; a unanimous paired increase at higher n_kv '
                 'refutes the capacity-degradation prediction'),
        'status': kda_status,
        'falsified': kda_status == 'contradicted',
        'unanimous_increases': unanimous_increase,
        'missing_pairs': missing_pairs,
        'evidence': pair_evidence,
    }
    checks['kda_nkv_monotonicity'] = kda_verdict
    for rows in kda_by_task.values():
        for r in rows:
            _assign(r, 'kda_nkv_monotonicity', kda_verdict)

    # ------------------------------------------------------------------
    # 3. Complex operators vs the tuned simple baseline.
    # ------------------------------------------------------------------
    ref_by_cell = {
        (r.get('task', 'mqar'), r['n_kv']): r
        for r in all_results
        if r.get('op') == baseline_ref and 'error' not in r
        and r.get('mean_acc') is not None
    }
    candidates = [r for r in all_results
                  if 'error' not in r and r.get('op') not in set(baseline_ops)
                  and r.get('mean_acc') is not None]
    n_reverse_tests = sum(
        1 for r in candidates
        if ref_by_cell.get((r.get('task', 'mqar'), r['n_kv'])) is not None)
    reverse_alpha = (0.05 / n_reverse_tests) if n_reverse_tests else None
    complex_evidence = []
    for r in candidates:
        key = _cell(r)
        if r.get('beats_tuned_baseline') is True:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'supported',
                'falsified': False,
                'evidence': {
                    'cell': key,
                    'reference_op': baseline_ref,
                    'vs_tuned_baseline': r.get('vs_tuned_baseline'),
                },
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('supported', key, r.get('op')))
            continue
        ref = ref_by_cell.get((r.get('task', 'mqar'), r['n_kv']))
        if ref is None:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'not_computable',
                'falsified': False,
                'evidence': {'cell': key,
                             'reason': 'reference_row_unavailable'},
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('not_computable', key, r.get('op')))
            continue
        cand = _finite_per_seed_acc(r.get('per_seed', []))
        base = _finite_per_seed_acc(ref.get('per_seed', []))
        common = sorted(set(cand) & set(base))
        if len(common) < 2:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'not_computable',
                'falsified': False,
                'evidence': {'cell': key, 'n_pairs': len(common),
                             'reason': 'insufficient_paired_seeds'},
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('not_computable', key, r.get('op')))
            continue
        reverse = _paired_one_sided_t([base[s] - cand[s] for s in common])
        crit = (_bonferroni_crit_q(reverse['n_pairs'], alpha=reverse_alpha)
                if (reverse_alpha is not None
                    and reverse['n_pairs'] >= 2) else None)
        ref_better = False
        if reverse['mean_diff'] is not None and reverse['mean_diff'] > 0:
            if (reverse['t_stat'] is not None and crit is not None
                    and reverse['t_stat'] > crit):
                ref_better = True
            elif (reverse['t_stat'] is None and reverse['all_positive']
                  and reverse['n_pairs'] >= 2 and reverse_alpha is not None
                  and 2.0 ** (-reverse['n_pairs']) <= reverse_alpha):
                ref_better = True
        status = 'contradicted' if ref_better else 'not_supported'
        verdict = {
            'rule': ('complex op must beat the TUNED simple baseline; a '
                     'significant REVERSE paired test refutes superiority '
                     '(failure to beat is only "not_supported")'),
            'status': status,
            'falsified': status == 'contradicted',
            'evidence': {
                'cell': key,
                'reference_op': baseline_ref,
                'reverse_mean_diff_reference_minus_op':
                    reverse['mean_diff'],
                'reverse_t_stat': reverse['t_stat'],
                'reverse_t_crit_bonferroni': crit,
                'reverse_alpha_bonferroni': reverse_alpha,
                'n_pairs': reverse['n_pairs'],
                'vs_tuned_baseline': r.get('vs_tuned_baseline'),
            },
        }
        _assign(r, 'beats_tuned_baseline', verdict)
        complex_evidence.append((status, key, r.get('op')))

    complex_statuses = [s for s, _, _ in complex_evidence]
    if not complex_statuses:
        complex_status = 'not_computable'
    elif all(s == 'not_computable' for s in complex_statuses):
        complex_status = 'not_computable'
    elif 'contradicted' in complex_statuses:
        complex_status = 'contradicted'
    elif all(s == 'supported' for s in complex_statuses):
        complex_status = 'supported'
    else:
        complex_status = 'not_supported'
    checks['beats_tuned_baseline'] = {
        'rule': ('every complex operator must beat the tuned simple '
                 'baseline, seed-paired; a significant reverse test '
                 'contradicts superiority'),
        'status': complex_status,
        'falsified': complex_status == 'contradicted',
        'reference_op': baseline_ref,
        'reverse_alpha_bonferroni': reverse_alpha,
        'n_reverse_tests': n_reverse_tests,
        'per_cell': [
            {'status': s, 'cell': cell, 'op': op}
            for s, cell, op in complex_evidence
        ],
    }

    # Attach a compact primary status per row (first applicable check) so
    # downstream consumers get one machine-readable field without having
    # to know the check list.
    for r in all_results:
        applicable = per_row.get(id(r)) or OrderedDict()
        if 'error' in r:
            r['hypothesis_checks'] = {
                'row': {'status': 'not_computable', 'rule': 'row_error',
                        'falsified': False,
                        'evidence': {'error': r.get('error')}}}
            r['hypothesis_status'] = 'not_computable'
            continue
        if not applicable and r.get('op') in set(baseline_ops):
            r['hypothesis_checks'] = {
                'reference_baseline': {
                    'rule': ('this row IS the tuned comparison reference; '
                             'no superiority claim applies to it'),
                    'status': 'not_applicable',
                    'falsified': False,
                    'evidence': {'op': r.get('op')},
                }}
            r['hypothesis_status'] = 'not_applicable'
            continue
        r['hypothesis_checks'] = applicable
        statuses = [v.get('status') for v in applicable.values()]
        if 'contradicted' in statuses:
            r['hypothesis_status'] = 'contradicted'
        elif 'not_supported' in statuses:
            r['hypothesis_status'] = 'not_supported'
        elif statuses and all(s == 'supported' for s in statuses):
            r['hypothesis_status'] = 'supported'
        else:
            r['hypothesis_status'] = 'not_computable'

    counts = {s: 0 for s in _HYPOTHESIS_STATUSES}
    for r in all_results:
        counts[r.get('hypothesis_status', 'not_computable')] = (
            counts.get(r.get('hypothesis_status', 'not_computable'), 0) + 1)
    return checks, counts


def _pairwise_family_resolution(records, alpha):
    """Summarize the exact-test rejection floor of one pairwise family.

    Every record carries its own ``test_resolution`` computed with the
    actual family size. A family with unequal common-seed counts can be
    testable for one pair and resolution-limited for another, so the
    aggregate reports both the worst and the best floor rather than a
    single misleading number.
    """
    if not records:
        return None
    flags = [bool(r['test_resolution']['family_testable'])
             for r in records]
    later = [bool(r['test_resolution']['later_rank_possible'])
             for r in records]
    n_limited = sum(1 for f in flags if not f)
    worst = max(records,
                key=lambda r: r['test_resolution'][
                    'min_attainable_p_holm'])
    best = min(records,
               key=lambda r: r['test_resolution'][
                   'min_attainable_p_holm'])
    if n_limited == 0:
        status = 'testable'
    elif n_limited == len(records):
        status = 'resolution_limited'
    else:
        status = 'partially_resolution_limited'
    return {
        'family_size': len(records),
        'min_n_pairs': min(r['n_pairs'] for r in records),
        'max_n_pairs': max(r['n_pairs'] for r in records),
        'status': status,
        'all_pairs_testable': bool(n_limited == 0),
        'any_pair_testable': bool(n_limited < len(records)),
        'n_resolution_limited': n_limited,
        # With unequal common-seed counts a later-ranked comparison can
        # still reject after the smaller-p members; record that weaker
        # possibility separately so the aggregate does not overstate it.
        'n_later_rank_possible': sum(1 for f in later if f),
        'any_pair_later_rank_possible': bool(any(later)),
        'worst_min_attainable_p_holm': worst['test_resolution'][
            'min_attainable_p_holm'],
        'best_min_attainable_p_holm': best['test_resolution'][
            'min_attainable_p_holm'],
        'min_effective_pairs_for_holm': max(
            r['test_resolution']['min_effective_pairs_holm']
            for r in records),
        'max_shortfall_pairs_for_holm': max(
            r['test_resolution']['shortfall_pairs_holm']
            for r in records),
        'note': ('The exact sign-flip test takes only 2**n values, and the '
                 'Holm correction multiplies its p floor by the family '
                 'size. A resolution_limited pair cannot be the first '
                 '(smallest-p) rejection; with equal n across the family '
                 '(the usual complete seed block) no outcome could have '
                 'crossed alpha, so a non-significant result is forced by '
                 'the design rather than evidence of equivalence. With '
                 'unequal n, n_later_rank_possible counts pairs whose own '
                 'p floor could still clear alpha at a later Holm rank.'),
    }


def _cross_cell_pairwise_report(pair_cell_diffs, *, alpha, n_boot,
                                n_permutations, max_exact):
    """Holm-corrected replication test across (task, n_kv) cells.

    Each pair's per-cell seed-paired differences are combined by
    ``multi_seed_stats.cross_cell_paired_analysis``: the seed is the unit,
    so a cell with more surviving seeds cannot dominate, and a pair that
    appears in only one cell is explicitly ``insufficient_cells`` rather
    than called a replication.
    """
    eligible = {}
    for pair_key, cells in pair_cell_diffs.items():
        if len(cells) < 2:
            continue
        common = set.intersection(
            *(set(seed_values) for seed_values in cells.values()))
        if len(common) >= 2:
            eligible[pair_key] = cells
    family_size = len(eligible)
    records = {}
    p_values = []
    for pair_key, cells in pair_cell_diffs.items():
        rec = cross_cell_paired_analysis(
            cells, alpha=alpha, n_boot=n_boot,
            n_permutations=n_permutations, max_exact=max_exact,
            family_size=family_size if pair_key in eligible else 1)
        records[pair_key] = rec
        if pair_key in eligible:
            primary = rec.get('primary')
            if primary is not None:
                p_values.append(
                    (pair_key,
                     primary['permutation_test']['p_value_two_sided']))
    holm = holm_adjust([p for _, p in p_values], alpha=alpha)
    for idx, (pair_key, _p) in enumerate(p_values):
        rec = records[pair_key]
        rec['holm_adjusted_p_permutation'] = holm['adjusted'][idx]
        rec['significant_holm_permutation'] = bool(holm['rejected'][idx])
        mean = rec['primary']['mean_diff']
        if rec['significant_holm_permutation']:
            rec['verdict'] = ('op_a_better' if mean > 0
                              else 'op_b_better')
        else:
            rec['verdict'] = 'not_significant'
    for rec in records.values():
        rec.setdefault('holm_adjusted_p_permutation', None)
        rec.setdefault('significant_holm_permutation', False)
        if 'verdict' not in rec:
            rec['verdict'] = (
                'insufficient_balanced_seeds'
                if rec.get('n_cells', 0) >= 2 else 'insufficient_cells')
    statuses = [rec.get('status') for rec in records.values()]
    verdicts = [rec.get('verdict') for rec in records.values()]
    significant = ('op_a_better', 'op_b_better')
    return {
        'alpha': alpha,
        'correction': 'holm_bonferroni',
        'family_scope': (
            'one family over operator pairs testable (>=2 common seeds) '
            'in at least two (task, n_kv) cells'),
        'n_pairs_total': len(records),
        'n_tests': len(p_values),
        'n_significant': sum(1 for v in verdicts if v in significant),
        'n_significant_and_consistent': sum(
            1 for rec in records.values()
            if rec.get('verdict') in significant
            and rec.get('status') == 'consistent_direction'),
        'n_consistent_direction': statuses.count('consistent_direction'),
        'n_mixed_direction': statuses.count('mixed_direction'),
        'n_aggregation_reversal': statuses.count(
            'aggregation_reversal'),
        'n_insufficient_cells': statuses.count('insufficient_cells'),
        'n_insufficient_balanced_seeds': statuses.count(
            'insufficient_balanced_seeds'),
        'pairs': records,
        'note': (
            'Cross-cell records re-use the per-cell seed-paired '
            'differences, so only cells where the pair had >=2 common '
            'seeds enter the cross-cell set. The primary test averages '
            'within seed across those cells, so the seed (not the cell) '
            'is the experimental unit and the cells are not treated as '
            'independent studies. A pair seen in only one testable cell '
            'is not a replication. aggregation_reversal means the naive '
            'observation-weighted mean has the opposite sign from the '
            'seed-balanced mean; mixed_direction means the per-cell means '
            'disagree. Neither is repaired by the pooled test, and '
            'neither licenses a cell-specific claim.'),
    }


def _pairwise_op_comparisons(all_results, alpha=0.05, seed=0,
                             n_boot=2000, n_permutations=10000,
                             max_exact=16, equivalence_margin=None):
    """Seed-paired, Holm-corrected operator-vs-operator comparisons.

    The per-row ``significant_bonferroni`` flag only says "this operator
    learned the task" and the ``vs_tuned_baseline`` test says "this
    operator beat the SIMPLE baseline". Neither answers the question the
    paper reader actually asks: *is operator A better than operator B,
    and is that ordering reproducible across seeds?* Comparing mean
    accuracies (or, worse, CI overlap) across two independently trained
    operators is not a valid test: the seeds are shared, so the correct
    statistic is the SEED-PAIRED difference.

    For every unordered operator pair inside a ``(task, n_kv)`` cell this
    function computes the per-seed differences ``acc(op_a) - acc(op_b)``
    on the COMMON seed set and runs:

    * the exact sign-flip permutation test (``multi_seed_stats``) as the
      PRIMARY statistic (two-sided, assumption-free for the 5-7-seed
      regime; exact enumeration below ``max_exact`` pairs, deterministic
      Monte-Carlo above);
    * the paired t-test as a secondary, normality-dependent view;
    * the exact binomial sign test, Cohen's ``d_z``, win rate with its
      exact Clopper-Pearson interval, the distribution-free median
      interval, and two bootstrap/t CIs for the mean difference;
    * ``seeds_required_80pct_power`` — how many seeds a follow-up run
      would need at the observed |d_z|. This converts an inconclusive
      comparison into an actionable multi-seed budget;
    * ``test_resolution`` — the smallest p the exact test could possibly
      produce at this n and family size, and whether the family has ANY
      rejection region. With seven operators (21 pairs) and five seeds the
      exact two-sided floor is 1.0, so every pair is forced to
      ``not_significant``; the report labels that a resolution limit
      instead of evidence of equivalence.

    When ``equivalence_margin`` is not None, every testable pair ALSO
    gets a TOST equivalence test against that PRE-SPECIFIED margin
    (``multi_seed_stats.tost_paired``), with its p-values Holm-corrected
    in a second family over the same pairs. This separates two claims
    that are routinely conflated:

    * ``verdict`` — a significant seed-paired difference (superiority);
    * ``equivalence_verdict`` — the ``(1 - 2*alpha)`` CI for the mean
      paired difference lies strictly inside +-margin (practical
      equivalence).

    A pair can be non-significant AND non-equivalent ("inconclusive at
    this seed budget"), and — for tiny-but-consistent differences — even
    significant AND practically equivalent; both are reported separately.
    A non-significant pair is NEVER labelled 'equivalent' by default:
    the margin comes from ``PAIRWISE_EQUIVALENCE_MARGIN`` /
    ``MQAR_EQUIV_MARGIN`` before the run, not from the observed CI.
    Each equivalence record also carries the smallest margin this run
    could have supported (``equivalence_margin_supported``) so an
    inconclusive pair is actionable.

    Multiple-comparison control: Holm-Bonferroni step-down within each
    cell (family = all testable unordered pairs of that cell), which
    controls FWER <= alpha per experimental question and is uniformly
    more powerful than plain Bonferroni at this seed count. p-values are
    adjusted for the permutation, the t-test, AND (when enabled) the TOST
    equivalence test — each in its own family.

    Single-seed robustness: every pair record also carries
    ``seed_influence`` from ``multi_seed_stats.leave_one_out_sensitivity``.
    It is descriptive (it re-uses the same data and is not a test): a
    significant ``verdict`` whose direction reverses when one seed is
    deleted is labelled ``direction_fragile`` and the winning row records
    the opponent under ``pairwise_direction_fragile_wins``. The report
    aggregates the per-cell counts under ``seed_influence``.

    Cross-cell replication: the per-cell family is the right unit for a
    cell-specific claim, but it also lets a reader select whichever
    ``(task, n_kv)`` cell happens to show a significant ordering. The
    report therefore also carries ``cross_cell_replication`` from
    ``multi_seed_stats.cross_cell_paired_analysis``: for every operator
    pair, the per-seed differences are averaged across cells (seed = the
    experimental unit, not the cell) and tested in one Holm family. It
    also reports mixed per-cell directions and Simpson-style aggregation
    reversal. A pair observed in only one cell is ``insufficient_cells``,
    not a replication.

    Side effect: every row participating in >= 1 testable pair gets
    ``pairwise_tests``, ``pairwise_significant_wins`` and
    ``pairwise_significant_losses`` (operator-name lists) so downstream
    consumers can read a row's pairwise standing without re-deriving it.
    With equivalence enabled they also get
    ``pairwise_equivalent_to``.

    Returns a JSON-safe report (also stored in the exp4 metadata under
    ``op_pairwise_comparisons``).
    """
    groups: dict = OrderedDict()
    for r in all_results:
        if 'error' in r or r.get('mean_acc') is None:
            continue
        groups.setdefault((r.get('task', 'mqar'), r['n_kv']), []).append(r)
    pair_cell_diffs: dict = {}

    report = {
        'alpha': alpha,
        'correction': 'holm_bonferroni',
        'family_scope': (
            'one family per (task, n_kv) cell: all unordered operator '
            'pairs with >= 2 common seeds'),
        'primary_test': (
            'two-sided exact sign-flip permutation test on the seed-paired '
            'accuracy differences; the paired t-test is reported as a '
            'secondary normality-dependent view'),
        'direction_convention': (
            'mean_diff = acc(op_a) - acc(op_b) on the common seed set'),
        'bootstrap': {'n_boot': n_boot, 'seed': seed},
        'ci95_overlap_note': (
            'CI overlap between two operators is a heuristic only and is '
            'neither necessary nor sufficient for a difference; the '
            'seed-paired permutation test here is the authoritative '
            'pairwise statistic.'),
        'n_tests': 0,
        'n_significant': 0,
        'n_skipped': 0,
        'n_equivalent': 0,
        'n_direction_fragile': 0,
        'n_boundary_sensitive': 0,
        'n_seed_influence_stable': 0,
        'n_resolution_limited': 0,
        'resolution': None,
        'equivalence': {
            'enabled': equivalence_margin is not None,
            'margin': equivalence_margin,
            'margin_source': (
                'pre_specified_config (PAIRWISE_EQUIVALENCE_MARGIN, '
                'overridable via the pre-run env MQAR_EQUIV_MARGIN; '
                '0 disables)'),
            'method': 'two one-sided paired t-tests (TOST)',
            'ci_level': (1.0 - 2.0 * alpha) if equivalence_margin is not None
            else None,
            'correction': ('holm_bonferroni over the same per-cell pairs '
                           'family (separate family from the superiority '
                           'tests)'),
            'note': (
                'equivalence_verdict=equivalent means the (1-2*alpha) CI '
                'of the mean paired difference lies strictly inside '
                '+-margin: the two operators are practically the same '
                'WITHIN the pre-specified margin. not_equivalent covers '
                'both "real difference" and "inconclusive"; use the '
                'superiority verdict and equivalence_margin_supported to '
                'tell those apart.'),
        },
        'cells': {},
    }

    for (task_name, n_kv_key), rows in groups.items():
        cell_key = f'{task_name}/n_kv={n_kv_key}'
        rows_sorted = sorted(rows, key=lambda r: str(r.get('op')))
        cell = {'n_tests': 0, 'n_significant': 0, 'n_skipped': 0,
                'n_equivalent': 0, 'n_direction_fragile': 0,
                'n_boundary_sensitive': 0, 'n_seed_influence_stable': 0,
                'pairs': OrderedDict()}
        records = []
        for i in range(len(rows_sorted)):
            for j in range(i + 1, len(rows_sorted)):
                a, b = rows_sorted[i], rows_sorted[j]
                pair_key = f"{a.get('op')}~{b.get('op')}"
                a_seed = _finite_per_seed_acc(a.get('per_seed', []))
                b_seed = _finite_per_seed_acc(b.get('per_seed', []))
                common = sorted(set(a_seed) & set(b_seed))
                if len(common) < 2:
                    cell['pairs'][pair_key] = {
                        'op_a': a.get('op'), 'op_b': b.get('op'),
                        'common_seeds': common, 'n_pairs': len(common),
                        'verdict': 'insufficient_paired_seeds',
                    }
                    cell['n_skipped'] += 1
                    continue
                diffs = [a_seed[s] - b_seed[s] for s in common]
                pair_cell_diffs.setdefault(pair_key, {})[cell_key] = {
                    s: a_seed[s] - b_seed[s] for s in common}
                stats = summarize_paired(
                    diffs, seed=seed, n_boot=n_boot,
                    n_permutations=n_permutations,
                    max_exact=max_exact, alpha=alpha,
                    equivalence_margin=equivalence_margin,
                    include_seed_influence=True)
                rec = {'op_a': a.get('op'), 'op_b': b.get('op'),
                       'common_seeds': common, **stats}
                records.append((rec, a, b))
                cell['pairs'][pair_key] = rec

        # The exact-test floor depends on the actual family size, which is
        # only known after every testable pair has been collected. Replace
        # the per-comparison default (family_size=1) from summarize_paired
        # with the family-aware record.
        for rec, _, _ in records:
            rec['test_resolution'] = exact_test_resolution(
                rec['n_pairs'], alpha=alpha, family_size=len(records),
                alternative='two-sided',
                n_zero=rec['sign_test']['n_zero'])

        p_perm = [rec['permutation_test']['p_value_two_sided']
                  for rec, _, _ in records]
        p_t_raw = [rec['t_test']['p_value_two_sided']
                   for rec, _, _ in records]
        holm_perm = holm_adjust(p_perm, alpha=alpha)
        # The t-test is undefined for a zero-variance difference vector
        # (t = +-inf); those pairs stay in the family with p=1.0 so the
        # family size (and therefore the correction strength) matches the
        # number of comparisons actually run, but they can never reject.
        holm_t = holm_adjust([1.0 if p is None else p for p in p_t_raw],
                             alpha=alpha)
        # TOST equivalence family (same pairs, separate question). Pairs
        # never have < 2 common seeds here (they were skipped above), so
        # the TOST p-value is always defined; the None guard is defensive.
        holm_tost = None
        if equivalence_margin is not None:
            p_tost = [rec['tost']['p_value'] for rec, _, _ in records]
            holm_tost = holm_adjust(
                [1.0 if p is None else p for p in p_tost], alpha=alpha)
        for idx, (rec, a, b) in enumerate(records):
            rec['holm_adjusted_p_permutation'] = holm_perm['adjusted'][idx]
            rec['significant_holm_permutation'] = bool(
                holm_perm['rejected'][idx])
            if p_t_raw[idx] is None:
                rec['holm_adjusted_p_t'] = None
                rec['significant_holm_t'] = False
            else:
                rec['holm_adjusted_p_t'] = holm_t['adjusted'][idx]
                rec['significant_holm_t'] = bool(holm_t['rejected'][idx])
            if equivalence_margin is not None:
                p_tost = rec.get('tost', {}).get('p_value')
                if p_tost is None:
                    rec['holm_adjusted_p_tost'] = None
                    rec['significant_equivalence_holm'] = False
                else:
                    rec['holm_adjusted_p_tost'] = holm_tost['adjusted'][idx]
                    rec['significant_equivalence_holm'] = bool(
                        holm_tost['rejected'][idx])
                rec['equivalence_verdict'] = (
                    'equivalent' if rec['significant_equivalence_holm']
                    else 'not_equivalent')
            mean = rec.get('mean_diff')
            if rec['significant_holm_permutation'] and mean:
                rec['verdict'] = 'op_a_better' if mean > 0 else 'op_b_better'
            else:
                rec['verdict'] = 'not_significant'
            for r in (a, b):
                r['pairwise_tests'] = r.get('pairwise_tests', 0) + 1
                r.setdefault('pairwise_significant_wins', [])
                r.setdefault('pairwise_significant_losses', [])
                r.setdefault('pairwise_direction_fragile_wins', [])
                if equivalence_margin is not None:
                    r.setdefault('pairwise_equivalent_to', [])
            if rec['verdict'] == 'op_a_better':
                a['pairwise_significant_wins'].append(b.get('op'))
                b['pairwise_significant_losses'].append(a.get('op'))
            elif rec['verdict'] == 'op_b_better':
                b['pairwise_significant_wins'].append(a.get('op'))
                a['pairwise_significant_losses'].append(b.get('op'))
            if (rec['verdict'] != 'not_significant'
                    and rec.get('seed_influence', {}).get(
                        'direction_fragile')):
                if rec['verdict'] == 'op_a_better':
                    a['pairwise_direction_fragile_wins'].append(b.get('op'))
                else:
                    b['pairwise_direction_fragile_wins'].append(a.get('op'))
            if equivalence_margin is not None and \
                    rec['equivalence_verdict'] == 'equivalent':
                a['pairwise_equivalent_to'].append(b.get('op'))
                b['pairwise_equivalent_to'].append(a.get('op'))
        cell['n_tests'] = len(records)
        cell['resolution'] = _pairwise_family_resolution(
            [rec for rec, _, _ in records], alpha)
        cell['n_resolution_limited'] = (
            0 if cell['resolution'] is None
            else cell['resolution']['n_resolution_limited'])
        cell['n_significant'] = sum(
            1 for rec, _, _ in records if rec['verdict'] != 'not_significant')
        cell['n_equivalent'] = sum(
            1 for rec, _, _ in records
            if rec.get('equivalence_verdict') == 'equivalent')
        cell['n_direction_fragile'] = sum(
            1 for rec, _, _ in records
            if rec.get('seed_influence', {}).get('direction_fragile'))
        cell['n_boundary_sensitive'] = sum(
            1 for rec, _, _ in records
            if rec.get('seed_influence', {}).get('decision_sensitive'))
        cell['n_seed_influence_stable'] = sum(
            1 for rec, _, _ in records
            if rec.get('seed_influence', {}).get('status') == 'stable')
        report['cells'][cell_key] = cell
        report['n_tests'] += cell['n_tests']
        report['n_significant'] += cell['n_significant']
        report['n_equivalent'] += cell['n_equivalent']
        report['n_skipped'] += cell['n_skipped']
        report['n_direction_fragile'] += cell['n_direction_fragile']
        report['n_boundary_sensitive'] += cell['n_boundary_sensitive']
        report['n_seed_influence_stable'] += cell['n_seed_influence_stable']
        report['n_resolution_limited'] += cell['n_resolution_limited']

    report['cross_cell_replication'] = _cross_cell_pairwise_report(
        pair_cell_diffs, alpha=alpha, n_boot=n_boot,
        n_permutations=n_permutations, max_exact=max_exact)

    cells_with_resolution = [
        cell['resolution'] for cell in report['cells'].values()
        if cell.get('resolution') is not None]
    report['resolution'] = {
        'family_scope': (
            'one family per (task, n_kv) cell, exactly the pairs that were '
            'Holm-corrected; the floor depends on the family size and on '
            'each pair\'s common-seed count'),
        'n_pairs': report['n_tests'],
        'n_resolution_limited': report['n_resolution_limited'],
        'all_pairs_testable': (
            all(c['all_pairs_testable'] for c in cells_with_resolution)
            if cells_with_resolution else None),
        'any_pair_testable': (
            any(c['any_pair_testable'] for c in cells_with_resolution)
            if cells_with_resolution else None),
        'max_shortfall_pairs_for_holm': (
            max(c['max_shortfall_pairs_for_holm']
                for c in cells_with_resolution)
            if cells_with_resolution else None),
        'n_later_rank_possible': sum(
            c['n_later_rank_possible'] for c in cells_with_resolution),
        'any_pair_later_rank_possible': (
            any(c['any_pair_later_rank_possible']
                for c in cells_with_resolution)
            if cells_with_resolution else None),
        'worst_min_attainable_p_holm': (
            max(c['worst_min_attainable_p_holm']
                for c in cells_with_resolution)
            if cells_with_resolution else None),
        'best_min_attainable_p_holm': (
            min(c['best_min_attainable_p_holm']
                for c in cells_with_resolution)
            if cells_with_resolution else None),
        'note': (
            'Exact sign/sign-flip tests have only 2**n sign assignments, '
            'so their p-value has a floor; Holm multiplies that floor by '
            'the family size. When n_resolution_limited > 0 the affected '
            'pairs cannot reject at alpha for ANY data, and their '
            'not_significant verdict is a property of the design, not '
            'evidence of equivalence. The continuous t-test is not subject '
            'to this floor.'),
    }

    report['seed_influence'] = {
        'method': 'leave_one_seed_out',
        'primary_test': 'sign_flip_permutation',
        'alpha': alpha,
        'correction': 'none (raw per-pair alpha; the Holm family is not '
                      're-run for each deletion)',
        'n_direction_fragile': report['n_direction_fragile'],
        'n_boundary_sensitive': report['n_boundary_sensitive'],
        'n_stable': report['n_seed_influence_stable'],
        'note': (
            'For every testable pair, each seed is deleted in turn and the '
            'paired panel is recomputed. direction_fragile means some '
            'single deletion reverses (or erases) the sign of the mean '
            'difference; boundary_sensitive means the direction survives '
            'but the primary permutation decision flips at alpha. This is '
            'a descriptive sensitivity check, not a test: it re-uses the '
            'same data, is not multiplicity-corrected, and never upgrades '
            'a claim.'),
    }

    report['note'] = (
        'Pairwise claims must cite these tests, not raw mean rankings or '
        'CI overlap. When a comparison is inconclusive, '
        'seeds_required_80pct_power estimates the seed budget the next '
        'run would need at the observed effect size. A "same/practically '
        'equivalent" claim requires equivalence_verdict=equivalent from '
        'the pre-specified-margin TOST; a non-significant superiority test '
        'alone never supports it. Read the resolution block first: a '
        'family whose exact-test floor exceeds alpha has no rejection '
        'region at all, so its non-significance is not a finding.')
    return report


def _print_pairwise_report(op_pairwise):
    """Print the per-cell paired-comparison table (stdout, not just logs)."""
    if not op_pairwise.get('n_tests'):
        print('\nSeed-paired op-vs-op comparisons: no testable pairs '
              '(need >= 2 common seeds per pair).')
        return
    eq = op_pairwise.get('equivalence', {})
    eq_on = bool(eq.get('enabled'))
    title = ('(permutation two-sided; Holm-corrected within cell'
             + (f"; TOST equivalence margin={eq.get('margin')!r}"
                if eq_on else '') + '):')
    print(f'\nSeed-paired op-vs-op comparisons {title}')
    header = (f"{'cell':>28} | {'op_a':>10} | {'op_b':>10} | {'d_acc':>8} | "
              f"{'n':>3} | {'p_perm':>9} | {'p_holm':>9} | {'seeds@80%':>9} | "
              f"verdict | {'infl':>9}")
    if eq_on:
        header += f" | {'p_tost':>9} | equivalence"
    print(header)
    infl_short = {
        'stable': 'stable',
        'direction_fragile': 'DIR-FRAG',
        'boundary_sensitive': 'boundary',
        'no_direction': 'none',
        'insufficient_data': 'n/a',
    }
    for cell_key, cell in op_pairwise['cells'].items():
        for rec in cell['pairs'].values():
            if rec.get('verdict') in (None, 'insufficient_paired_seeds'):
                continue
            d = rec.get('mean_diff')
            d_str = f'{d:+.4f}' if d is not None else 'n/a'
            p = rec['permutation_test']['p_value_two_sided']
            p_str = f'{p:.4f}' if p is not None else 'n/a'
            ph = rec.get('holm_adjusted_p_permutation')
            ph_str = f'{ph:.4f}' if ph is not None else 'n/a'
            need = rec.get('seeds_required_80pct_power')
            need_str = str(need) if need is not None else 'n/a'
            status = rec.get('seed_influence', {}).get('status', 'n/a')
            status_str = infl_short.get(status, status)
            row = (f"{cell_key:>28} | {str(rec.get('op_a')):>10} | "
                   f"{str(rec.get('op_b')):>10} | {d_str:>8} | "
                   f"{rec.get('n_pairs', 0):>3} | {p_str:>9} | "
                   f"{ph_str:>9} | {need_str:>9} | {rec['verdict']} | "
                   f"{status_str:>9}")
            if eq_on:
                pt = rec.get('holm_adjusted_p_tost')
                pt_str = f'{pt:.4f}' if pt is not None else 'n/a'
                row += (f" | {pt_str:>9} | "
                        f"{rec.get('equivalence_verdict', 'n/a')}")
            print(row)
    res = op_pairwise.get('resolution') or {}
    n_limited = res.get('n_resolution_limited') or 0
    if n_limited:
        floor = res.get('worst_min_attainable_p_holm')
        floor_str = f'{floor:.4f}' if floor is not None else 'n/a'
        short = res.get('max_shortfall_pairs_for_holm')
        print(f"  RESOLUTION: {n_limited}/{res.get('n_pairs', 0)} pair(s) "
              f"cannot be the first rejection in this Holm family "
              f"(smallest attainable adjusted p={floor_str}); "
              f"{short} more effective pair(s) would be needed. With equal "
              f"seed counts across the family, no rejection is possible; "
              f"read 'not_significant' as a design limit, not evidence of "
              f"equivalence.")


__all__ = [
    '_T_PP',
    '_t_crit_975',
    '_bonferroni_crit_q',
    '_is_finite_real',
    '_finite_per_seed_acc',
    '_paired_one_sided_t',
    '_HYPOTHESIS_STATUSES',
    '_hypothesis_verdicts',
    '_pairwise_family_resolution',
    '_pairwise_op_comparisons',
    '_print_pairwise_report',
]
