"""Regression tests for ``timing_stats.py`` and its producer wiring.

The latency experiments now report a percentile-bootstrap CI95 for the
median plus q10/q90 and CV. These tests pin the summary arithmetic on
hand-computed samples, check determinism and edge cases (empty, one
sample, non-finite entries), and verify that the Exp 2 helper and
``latency_profile.time_call`` expose the same fields. They never touch
the real ``results/`` directory.
"""
from __future__ import annotations

import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import latency_profile as lp  # noqa: E402
import timing_stats as ts  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}: {detail}')
    assert cond, f'{name}: {detail}'
    return cond


def test_summary_hand_values_and_quantiles():
    print('\nTest: summarize_timings_ms hand values')
    stats = ts.summarize_timings_ms([1.0, 2.0, 3.0, 4.0])
    _ok('median', abs(stats['median_ms'] - 2.5) < 1e-12, stats['median_ms'])
    _ok('mean', abs(stats['mean_ms'] - 2.5) < 1e-12, stats['mean_ms'])
    _ok('min/max', stats['min_ms'] == 1.0 and stats['max_ms'] == 4.0,
        f"{stats['min_ms']}/{stats['max_ms']}")
    _ok('sample std', abs(stats['std_ms'] - statistics.stdev(
        [1.0, 2.0, 3.0, 4.0])) < 1e-12, stats['std_ms'])
    _ok('q10 is linear interpolation', abs(stats['q10_ms'] - 1.3) < 1e-12,
        stats['q10_ms'])
    _ok('q90 is linear interpolation', abs(stats['q90_ms'] - 3.7) < 1e-12,
        stats['q90_ms'])
    _ok('CI brackets the median',
        stats['ci95_low_ms'] <= stats['median_ms'] <= stats['ci95_high_ms'],
        f"{stats['ci95_low_ms']}..{stats['ci95_high_ms']}")
    _ok('CI halfwidth is consistent',
        abs(stats['ci95_halfwidth_ms']
            - (stats['ci95_high_ms'] - stats['ci95_low_ms']) / 2.0) < 1e-12,
        stats['ci95_halfwidth_ms'])
    _ok('n and method recorded',
        stats['n'] == 4
        and stats['ci_method'] == 'percentile_bootstrap_median'
        and stats['bootstrap_reps'] == ts.DEFAULT_BOOTSTRAP_REPS, '')
    _ok('CV = std / mean',
        abs(stats['cv'] - stats['std_ms'] / stats['mean_ms']) < 1e-12,
        stats['cv'])


def test_summary_seconds_and_edge_cases():
    print('\nTest: summarize_timings_s and edge cases')
    ms = ts.summarize_timings_ms([1.0, 2.0, 3.0, 4.0], seed=7)
    sec = ts.summarize_timings_s([0.001, 0.002, 0.003, 0.004], seed=7)
    _ok('seconds wrapper preserves every statistic',
        all(sec[k] == ms[k] for k in ms if k not in ('alpha',)),
        f"{sec['median_ms']} vs {ms['median_ms']}")
    empty = ts.summarize_timings_ms([])
    _ok('empty sample is explicit (n=0, no point estimate)',
        empty['n'] == 0 and empty['median_ms'] is None
        and empty['ci95_low_ms'] is None, '')
    one = ts.summarize_timings_ms([5.0])
    _ok('one sample has a degenerate CI',
        one['n'] == 1 and one['ci95_low_ms'] == 5.0
        and one['ci95_high_ms'] == 5.0 and one['std_ms'] == 0.0, '')
    dirty = ts.summarize_timings_ms([1.0, float('nan'), float('inf'), -2.0])
    _ok('non-finite dropped, negative clamped to zero',
        dirty['n'] == 2 and dirty['median_ms'] == 0.5, dirty)
    a = ts.summarize_timings_ms([1.0, 2.0, 3.0], seed=11)
    b = ts.summarize_timings_ms([1.0, 2.0, 3.0], seed=11)
    _ok('bootstrap is deterministic for a fixed seed',
        a['ci95_low_ms'] == b['ci95_low_ms']
        and a['ci95_high_ms'] == b['ci95_high_ms'], '')


def test_ratio_of_medians_ci():
    print('\nTest: ratio_of_medians_ci (unpaired and paired)')
    stats = ts.ratio_of_medians_ci([10.0, 11.0, 9.0, 10.0],
                                   [20.0, 21.0, 19.0, 20.0])
    _ok('point ratio is median(a)/median(b)',
        abs(stats['ratio_median'] - 0.5) < 1e-12, stats['ratio_median'])
    _ok('ratio CI brackets the point estimate',
        stats['ratio_ci95_low'] <= stats['ratio_median']
        <= stats['ratio_ci95_high'],
        f"{stats['ratio_ci95_low']}..{stats['ratio_ci95_high']}")
    _ok('ratio CI excludes 1.0 for a clear 2x gap',
        stats['ratio_ci95_high'] < 1.0, stats['ratio_ci95_high'])
    paired = ts.ratio_of_medians_ci([10.0, 11.0, 9.0],
                                    [20.0, 22.0, 18.0], paired=True)
    _ok('paired ratio is marked and finite',
        paired['paired'] is True and math.isfinite(paired['ratio_median'])
        and paired['n_a'] == paired['n_b'] == 3, paired)
    raised = False
    try:
        ts.ratio_of_medians_ci([1.0, 2.0], [3.0], paired=True)
    except ValueError:
        raised = True
    _ok('paired mismatch is rejected', raised, '')
    for kwargs in ({'n_boot': 0}, {'alpha': 0.0}, {'alpha': 1.0}):
        raised = False
        try:
            ts.summarize_timings_ms([1.0, 2.0], **kwargs)
        except ValueError:
            raised = True
        _ok(f'rejects {kwargs}', raised, '')


def test_paired_ratio_drops_whole_trials_after_invalid_values():
    print('\nTest: paired ratio keeps trial indices aligned')
    # A has NaN at index 0, B at index 2. A pairwise drop leaves trials
    # (a=1,b=2) and (a=9,b=90); independently cleaning each side would
    # instead pair (1,10), (3,2), (9,90).
    stats = ts.ratio_of_medians_ci(
        [float('nan'), 1.0, 3.0, 9.0],
        [10.0, 2.0, float('nan'), 90.0], paired=True, seed=3)
    _ok('paired ratio drops only complete pairs',
        stats['n_a'] == stats['n_b'] == 2, stats)
    _ok('paired ratio uses the aligned medians',
        abs(stats['ratio_median'] - 5.0 / 46.0) < 1e-12,
        stats['ratio_median'])
    # A fully valid paired sample keeps the original behavior.
    valid = ts.ratio_of_medians_ci([1.0, 2.0, 3.0], [2.0, 4.0, 6.0],
                                   paired=True, seed=3)
    _ok('valid paired ratio is unchanged',
        valid['n_a'] == valid['n_b'] == 3
        and abs(valid['ratio_median'] - 0.5) < 1e-12, valid)
    raised = False
    try:
        ts.ratio_of_medians_ci([float('nan'), float('nan')],
                               [2.0, 4.0], paired=True)
    except ValueError:
        raised = True
    _ok('a paired sample with no usable trials is rejected', raised, '')


def test_producer_wiring_reports_uncertainty():
    print('\nTest: latency_profile.time_call and run_benchmark expose CI fields')
    stats = lp.time_call(lambda: None, warmup=0, repeats=4,
                         timer=lambda: 1.0)
    _ok('time_call keeps the legacy summary keys',
        all(k in stats for k in ('median_ms', 'min_ms', 'max_ms', 'std_ms',
                                 'times_ms')), '')
    _ok('time_call adds q10/q90 and CI95',
        all(k in stats for k in ('q10_ms', 'q90_ms', 'ci95_low_ms',
                                 'ci95_high_ms', 'n', 'ci_method'))
        and stats['n'] == 4, '')
    from run_benchmark import _LAST_TIMING_STATS, _record_timing_stats
    _record_timing_stats([0.001, 0.002, 0.003, 0.004])
    _ok('Exp 2 timing cache carries the uncertainty summary',
        _LAST_TIMING_STATS['ci95_low_ms']
        <= _LAST_TIMING_STATS['median_ms']
        <= _LAST_TIMING_STATS['ci95_high_ms']
        and _LAST_TIMING_STATS['n'] == 4
        and _LAST_TIMING_STATS['ci_method']
        == 'percentile_bootstrap_median', _LAST_TIMING_STATS)


_TEST_FUNCS = [
    test_summary_hand_values_and_quantiles,
    test_summary_seconds_and_edge_cases,
    test_ratio_of_medians_ci,
    test_paired_ratio_drops_whole_trials_after_invalid_values,
    test_producer_wiring_reports_uncertainty,
]


def main() -> int:
    print('=' * 70)
    print('Timing-uncertainty regression tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f'  [FAIL] {fn.__name__}: AssertionError: {e}')
            results.append(False)
        except Exception as e:
            print(f'  [CRASH] {fn.__name__}: {type(e).__name__}: {e}')
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
