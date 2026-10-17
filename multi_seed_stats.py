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
7. :func:`summarize_paired` — one call that runs the whole panel and
   returns a single JSON-safe record for the results file.

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
    'holm_adjust',
    'min_detectable_effect',
    'seeds_required',
    'summarize_paired',
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
    """Survival function ``P(T > t)`` of Student's t."""
    return 1.0 - student_t_cdf(t_value, dof)


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
        d = float(d)
        if not math.isfinite(d):
            raise ValueError(
                f'diffs[{i}]={d!r} is not finite; filter diverged seeds '
                f'before calling multi_seed_stats')
        out.append(d)
    return out


# Relative tolerance for "numerically zero" variance. A perfectly
# constant difference vector (e.g. every seed improved by a fixed 0.8)
# can still produce a ~1e-16 residue in the floating-point variance from
# the mean-subtraction round-off. Treating that residue as real variance
# turns Cohen's d_z into ~1e16 and the power estimate into 1 seed, so it
# must be clamped to zero.
_ZERO_VAR_REL_EPS = 1.0e-12


def _sample_sd(values: list[float], mean: float) -> float | None:
    n = len(values)
    if n < 2:
        return None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    scale = max(abs(mean), max(abs(v) for v in values), 1.0e-300)
    if var <= (_ZERO_VAR_REL_EPS * scale) ** 2:
        return 0.0
    return math.sqrt(var)


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
    mean = sum(d) / n
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


def sign_test(diffs) -> dict:
    """Exact two-sided binomial sign test on paired differences.

    Zeros are dropped (Steel's convention): a seed that produced *exactly*
    equal accuracy for both operators carries no directional information.
    The p-value is the exact ``2 * P(X >= max(n+, n-))`` for
    ``X ~ Binomial(n+, n-, 0.5)``, capped at 1.0. Distribution-free and
    robust to a single divergent magnitude; it is the exact paired
    permutation test restricted to sign information.
    """
    d = _as_clean_diffs(diffs)
    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    n_zero = len(d) - n_pos - n_neg
    n_eff = n_pos + n_neg
    if n_eff == 0:
        p = 1.0
    else:
        k = max(n_pos, n_neg)
        tail = sum(math.comb(n_eff, i) for i in range(k, n_eff + 1))
        p = min(1.0, 2.0 * tail / float(2 ** n_eff))
    return {'n_pairs': len(d), 'n_positive': n_pos, 'n_negative': n_neg,
            'n_zero': n_zero, 'n_effective': n_eff, 'p_value': p}


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
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n == 0:
        return {'n_pairs': 0, 'method': 'exact', 'n_permutations': 1,
                'p_value': None, 'alternative': alternative}
    observed = sum(d) / n
    eps = 1e-12

    def _tail_hit(stat: float) -> bool:
        if alternative == 'two-sided':
            return abs(stat) >= abs(observed) - eps
        if alternative == 'greater':
            return stat >= observed - eps
        return stat <= observed + eps

    if n <= max_exact:
        n_perm = 2 ** n
        hits = 0
        for mask in range(n_perm):
            stat = 0.0
            for i in range(n):
                stat += d[i] if (mask >> i) & 1 else -d[i]
            if _tail_hit(stat / n):
                hits += 1
        return {'n_pairs': n, 'method': 'exact', 'n_permutations': n_perm,
                'p_value': min(1.0, hits / float(n_perm)),
                'alternative': alternative}
    rng = random.Random(seed)
    if n_permutations < 1:
        raise ValueError(
            f'n_permutations must be >= 1, got {n_permutations!r}')
    hits = 0
    for _ in range(n_permutations):
        stat = 0.0
        for x in d:
            stat += x if rng.random() < 0.5 else -x
        if _tail_hit(stat / n):
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
    if n_boot < 1:
        raise ValueError(f'n_boot must be >= 1, got {n_boot!r}')
    d = _as_clean_diffs(diffs)
    n = len(d)
    if n < 2:
        return {'n_pairs': n, 'n_boot': n_boot, 'seed': seed,
                'ci_lo': None, 'ci_hi': None,
                'ci_level': 1.0 - alpha}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
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
    mean = sum(d) / len(d)
    sd = _sample_sd(d, mean)
    if sd is None or sd <= 0.0:
        return None
    return mean / sd


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
        p = float(p)
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
    if n < 2:
        return None
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError(
            f'alpha and power must be in (0,1), got {alpha!r}, {power!r}')
    if not math.isfinite(sd) or sd < 0.0:
        raise ValueError(f'sd must be finite and >= 0, got {sd!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise ValueError(
            f"alternative must be 'two-sided'|'greater'|'less', "
            f"got {alternative!r}")
    z_crit = _ND.inv_cdf(1.0 - alpha / 2.0) if alternative == 'two-sided' \
        else _ND.inv_cdf(1.0 - alpha)
    z_power = _ND.inv_cdf(power)
    return (z_crit + z_power) * sd / math.sqrt(n)


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
    dz = float(dz)
    if not math.isfinite(dz) or dz == 0.0:
        return None
    z_crit = _ND.inv_cdf(1.0 - alpha / 2.0) if alternative == 'two-sided' \
        else _ND.inv_cdf(1.0 - alpha)
    z_power = _ND.inv_cdf(power)
    return int(math.ceil(((z_crit + z_power) / abs(dz)) ** 2))


# ---------------------------------------------------------------------------
# One-call panel
# ---------------------------------------------------------------------------

def summarize_paired(diffs, seed: int = 0, n_boot: int = 10000,
                     n_permutations: int = 10000,
                     max_exact: int = 16,
                     alpha: float = 0.05,
                     power: float = 0.8) -> dict:
    """Run the full multi-seed inference panel on paired differences.

    Returns a single JSON-safe record containing the effect estimate,
    t-based and bootstrap CIs, the parametric t-test, the exact sign test,
    the sign-flip permutation test, Cohen's ``d_z``, the seed win rate,
    and the number of seeds a follow-up run would need for ``power`` at
    the observed effect. ``direction`` is ``'greater'`` when the mean
    difference is positive, ``'less'`` when negative, ``'none'`` when
    exactly zero.

    The record deliberately reports three p-values for the same
    comparison: if they disagree materially at n = 5, the per-seed
    differences are skew/outlier-driven and the result should be treated
    as fragile rather than decided by whichever test looks best.
    """
    d = _as_clean_diffs(diffs)
    n = len(d)
    mean = (sum(d) / n) if n else None
    median = None
    if n:
        s = sorted(d)
        median = (s[(n - 1) // 2] if n % 2 else
                  0.5 * (s[n // 2 - 1] + s[n // 2]))
    sd = _sample_sd(d, mean) if n >= 2 else None
    t = paired_t_test(d, alternative='two-sided')
    t_dir = paired_t_test(
        d, alternative='greater' if (mean is not None and mean >= 0)
        else 'less')
    sign = sign_test(d)
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
    direction = ('none' if mean is None or mean == 0.0
                 else ('greater' if mean > 0 else 'less'))
    return {
        'n_pairs': n,
        'mean_diff': mean,
        'median_diff': median,
        'sd_diff': sd,
        'cohens_dz': dz,
        'direction': direction,
        'win_rate': win_rate,
        'n_seeds_a_wins': n_pos,
        'n_seeds_b_wins': n_neg,
        'n_seeds_tied': n - n_nonzero,
        'ci95_mean_diff_t': [ci_lo, ci_hi],
        'ci95_mean_diff_bootstrap': [boot['ci_lo'], boot['ci_hi']],
        'bootstrap': {'n_boot': boot['n_boot'], 'seed': boot['seed']},
        't_test': {'t_stat': t['t_stat'], 'dof': t['dof'],
                   'p_value_two_sided': t['p_value'],
                   'p_value_directional': t_dir['p_value'],
                   'all_positive': t['all_positive'],
                   'all_negative': t['all_negative']},
        'sign_test': {'n_effective': sign['n_effective'],
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
    }


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
