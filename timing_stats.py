"""Timing summaries with uncertainty for the latency experiments.

The latency experiments used to persist only a median (plus min/max/std in
Exp 2). A point estimate without an interval invites over-reading small
differences, and Exp 6 compared per-token means/medians with no dispersion
at all. This module centralizes the summary so every producer reports the
same quantities:

* median / mean / min / max / sample std;
* q10 / q90 (robust spread for small repeat counts);
* a deterministic percentile-bootstrap CI95 for the median;
* the coefficient of variation and the sample count;
* ``ratio_of_medians_ci`` for A-vs-B latency ratios, with an unpaired
  option (independent timing blocks) and a paired option (same trial
  indices).

Percentile bootstrap is used instead of a normal/t interval because repeat
counts are small (often 3-5) and timing distributions are right-skewed.
The RNG is seeded locally (``seed=0`` by default), so a rerun on the same
timings returns the same interval; it never touches global RNG state.

All functions are pure and dependency-light (stdlib only), and monotone
quantities are checked by ``test_timing_stats.py``.
"""
from __future__ import annotations

import math
import random
import statistics

DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_ALPHA = 0.05


def _percentile(sorted_values, q):
    """Linear-interpolation percentile of an already-sorted non-empty list."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _clean(values):
    """Finite, non-negative floats; non-finite/negative samples are dropped."""
    out = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(v) and v >= 0.0:
            out.append(v)
    return out


def _safe_mean(values):
    """Mean of finite non-negative values without an overflowing sum."""
    n = len(values)
    total = sum(values)
    if math.isfinite(total):
        return total / n
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    mean = scale * (math.fsum(v / scale for v in values) / n)
    return mean if math.isfinite(mean) else None


def _safe_median(values):
    """Median whose even-sample midpoint cannot overflow to ``inf``."""
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
    # ``a + (b-a)/2`` avoids the overflowing sum and is exact for the
    # non-negative timing samples this module handles.
    mid = a + (b - a) / 2.0
    if math.isfinite(mid):
        return mid
    mid = a / 2.0 + b / 2.0
    return mid if math.isfinite(mid) else None


def _safe_stdev(values):
    """Sample SD with a scaled fallback when the raw variance overflows."""
    n = len(values)
    if n < 2:
        return 0.0
    try:
        std = statistics.stdev(values)
    except (OverflowError, ValueError):
        std = None
    if std is not None and math.isfinite(std):
        return std
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    scaled = [v / scale for v in values]
    mean = _safe_mean(scaled)
    if mean is None:
        return None
    var = math.fsum((v - mean) ** 2 for v in scaled) / (n - 1)
    std = scale * math.sqrt(var)
    return std if math.isfinite(std) else None


def _check_bootstrap(n_boot, alpha):
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f'n_boot={n_boot!r} must be a positive int.')
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)):
        raise ValueError(f'alpha={alpha!r} must be in (0, 1).')
    try:
        alpha_value = float(alpha)
    except OverflowError:
        raise ValueError(f'alpha={alpha!r} must be in (0, 1).') from None
    if not 0.0 < alpha_value < 1.0:
        raise ValueError(f'alpha={alpha!r} must be in (0, 1).')


def bootstrap_median_ci(values, *, n_boot=DEFAULT_BOOTSTRAP_REPS,
                        alpha=DEFAULT_ALPHA, seed=0):
    """Percentile-bootstrap CI for the median of ``values``.

    Returns ``(low, high)`` in the same units as the input. A single
    observation yields a degenerate interval at that observation (no
    uncertainty is observable from one sample); an empty sample yields
    ``(None, None)``.
    """
    _check_bootstrap(n_boot, alpha)
    vals = _clean(values)
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(seed)
    n = len(vals)
    meds = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        meds.append(_safe_median(sample))
    meds.sort()
    return (_percentile(meds, 100.0 * float(alpha) / 2.0),
            _percentile(meds, 100.0 * (1.0 - float(alpha) / 2.0)))


def summarize_timings_ms(times_ms, *, n_boot=DEFAULT_BOOTSTRAP_REPS,
                         alpha=DEFAULT_ALPHA, seed=0):
    """Summary of timings already expressed in milliseconds."""
    _check_bootstrap(n_boot, alpha)
    vals = _clean(times_ms)
    if not vals:
        return {
            'median_ms': None, 'mean_ms': None, 'min_ms': None,
            'max_ms': None, 'std_ms': None, 'q10_ms': None,
            'q90_ms': None, 'ci95_low_ms': None, 'ci95_high_ms': None,
            'ci95_halfwidth_ms': None, 'cv': None, 'n': 0,
            'ci_method': 'percentile_bootstrap_median',
            'bootstrap_reps': n_boot, 'alpha': float(alpha),
        }
    ordered = sorted(vals)
    mean = _safe_mean(vals)
    std = _safe_stdev(vals)
    low, high = bootstrap_median_ci(vals, n_boot=n_boot, alpha=alpha,
                                    seed=seed)
    median = _safe_median(vals)
    return {
        'median_ms': median,
        'mean_ms': mean,
        'min_ms': ordered[0],
        'max_ms': ordered[-1],
        'std_ms': std,
        'q10_ms': _percentile(ordered, 10.0),
        'q90_ms': _percentile(ordered, 90.0),
        'ci95_low_ms': low,
        'ci95_high_ms': high,
        'ci95_halfwidth_ms': ((high - low) / 2.0
                              if low is not None and high is not None
                              else None),
        'cv': ((std / mean) if (std is not None and mean not in (None, 0.0))
               else None),
        'n': len(vals),
        'ci_method': 'percentile_bootstrap_median',
        'bootstrap_reps': n_boot,
        'alpha': float(alpha),
    }


def summarize_timings_s(times_s, *, n_boot=DEFAULT_BOOTSTRAP_REPS,
                        alpha=DEFAULT_ALPHA, seed=0):
    """Summary of timings expressed in seconds, returned in milliseconds."""
    # Clean before converting so a ``None`` / NaN / negative entry is
    # dropped by the same contract as ``summarize_timings_ms`` instead of
    # raising in ``float(None)``.
    vals_s = _clean(times_s)
    return summarize_timings_ms([v * 1e3 for v in vals_s],
                                n_boot=n_boot, alpha=alpha, seed=seed)


def ratio_of_medians_ci(a_ms, b_ms, *, paired=False,
                        n_boot=DEFAULT_BOOTSTRAP_REPS,
                        alpha=DEFAULT_ALPHA, seed=0):
    """Bootstrap CI for ``median(a) / median(b)`` (A over B).

    ``paired=True`` resamples trial indices jointly and requires equal
    lengths; that is the correct mode when A and B are measured on the
    same trial inputs/blocks. The default is unpaired (independent timing
    blocks). Returns a dict with the point ratio, CI, and the two medians.
    """
    _check_bootstrap(n_boot, alpha)
    if paired:
        # Pairing is by trial index, so the two raw samples must line up
        # BEFORE any invalid entry is removed. Cleaning each side
        # independently and then comparing lengths can silently pair
        # different trials (e.g. a NaN at index 0 of A and at index 2 of B
        # both disappear, leaving equal-length but misaligned lists).
        raw_a = list(a_ms)
        raw_b = list(b_ms)
        if len(raw_a) != len(raw_b):
            raise ValueError('paired ratio requires equal-length samples: '
                             f'{len(raw_a)} != {len(raw_b)}')
        a = []
        b = []
        for value_a, value_b in zip(raw_a, raw_b):
            clean_a = _clean([value_a])
            clean_b = _clean([value_b])
            if not clean_a or not clean_b:
                # Drop the whole pair: one missing/invalid side makes the
                # trial unusable, and keeping the other would break the
                # index correspondence.
                continue
            a.append(clean_a[0])
            b.append(clean_b[0])
    else:
        a = _clean(a_ms)
        b = _clean(b_ms)
    if not a or not b:
        raise ValueError('ratio requires two non-empty samples.')
    med_a = _safe_median(a)
    med_b = _safe_median(b)
    point = ((med_a / med_b)
             if (med_a is not None and med_b is not None and med_b > 0)
             else None)
    if point is not None and not math.isfinite(point):
        point = None
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        if paired:
            idx = [rng.randrange(len(a)) for _ in range(len(a))]
            sa = [a[i] for i in idx]
            sb = [b[i] for i in idx]
        else:
            sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
            sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        mb = _safe_median(sb)
        ma = _safe_median(sa)
        if mb is not None and ma is not None and mb > 0:
            ratio = ma / mb
            if math.isfinite(ratio):
                ratios.append(ratio)
    if not ratios:
        low = high = None
    else:
        ratios.sort()
        low = _percentile(ratios, 100.0 * float(alpha) / 2.0)
        high = _percentile(ratios, 100.0 * (1.0 - float(alpha) / 2.0))
    return {
        'ratio_median': point,
        'ratio_ci95_low': low,
        'ratio_ci95_high': high,
        'a_median_ms': med_a,
        'b_median_ms': med_b,
        'n_a': len(a),
        'n_b': len(b),
        'paired': bool(paired),
        'ci_method': 'percentile_bootstrap_median_ratio',
        'bootstrap_reps': n_boot,
        'alpha': float(alpha),
    }
