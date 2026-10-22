"""Regression tests for the multi-seed inference calibration study.

These tests pin the simulator's population properties, the row arithmetic
of the three studies, the diagnostic-vs-gate logic, and the global-RNG
isolation that makes the report reproducible. Runtime is kept small by
using reduced replicate counts; the committed protocol runs in ~2 s.
"""
from __future__ import annotations

import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import multi_seed_calibration as mc  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}: {detail}')
    assert cond, f'{name}: {detail}'
    return cond


def test_simulator_known_population_and_determinism():
    print('\nTest: paired-difference simulator')
    for dist in mc.DISTRIBUTIONS:
        a = mc.simulate_paired_diffs(
            4000, random.Random(7), effect=0.5, noise_sd=1.0,
            distribution=dist)
        b = mc.simulate_paired_diffs(
            4000, random.Random(7), effect=0.5, noise_sd=1.0,
            distribution=dist)
        _ok(f'{dist}: deterministic', a == b, '')
        _ok(f'{dist}: finite draws', all(math.isfinite(x) for x in a), '')
        mean = sum(a) / len(a)
        sd = statistics.stdev(a)
        _ok(f'{dist}: population mean ~= effect',
            abs(mean - 0.5) < 0.08, f'mean={mean:.3f}')
        _ok(f'{dist}: marginal SD ~= noise_sd',
            abs(sd - 1.0) < 0.08, f'sd={sd:.3f}')
    for kwargs in ({'n_pairs': 1}, {'distribution': 'cauchy'},
                   {'noise_sd': 0.0}, {'effect': float('nan')}):
        raised = False
        call = dict(kwargs)
        call.setdefault('n_pairs', 5)
        try:
            mc.simulate_paired_diffs(
                call.pop('n_pairs'), random.Random(0), **call)
        except ValueError:
            raised = True
        _ok(f'rejects {kwargs}', raised, '')


def test_size_study_rows_and_diagnostics():
    print('\nTest: null-size study rows and diagnostic gating')
    scenarios = [
        {'name': 'normal_n5', 'n_pairs': 5, 'distribution': 'normal',
         'alpha': 0.05, 'tol': 0.04},
        {'name': 'contaminated_n7', 'n_pairs': 7,
         'distribution': 'contaminated', 'alpha': 0.05, 'tol': 0.06,
         'diagnostic_methods': ['paired_t']},
    ]
    rows = mc.size_study(scenarios, reps=40, seed=3)
    _ok('one row per scenario and method', len(rows) == 6, len(rows))
    _ok('counts -> rate is exact',
        all(abs(r['rejection_rate'] - r['n_significant'] / r['n_reps'])
            < 1e-12 for r in rows), '')
    _ok('threshold = alpha + tol',
        all(abs(r['threshold'] - (r['alpha'] + r['tol'])) < 1e-12
            for r in rows), '')
    diag = [r for r in rows if r['name'] == 'contaminated_n7'
            and r['method'] == 'paired_t']
    _ok('t-test under contamination is diagnostic, not a gate',
        len(diag) == 1 and diag[0]['gate'] is False
        and diag[0]['diagnostic_only'] is True, diag)
    rerun = mc.size_study(scenarios, reps=40, seed=3)
    _ok('deterministic report', rows == rerun, '')
    raised = False
    try:
        mc.size_study([{'n_pairs': 5, 'tol': 0.0}], reps=5)
    except ValueError:
        raised = True
    _ok('missing/invalid tolerance is rejected', raised, '')


def test_coverage_and_power_rows():
    print('\nTest: coverage and power row arithmetic')
    cov = mc.coverage_study(
        [{'name': 'normal_n10', 'n_pairs': 10, 'distribution': 'normal',
          'effect': 0.5, 'alpha': 0.05, 'tol': 0.05}],
        reps=30, n_boot=40, seed=4)
    row = cov[0]
    _ok('coverage = n_covered / n_reps',
        abs(row['coverage'] - row['n_covered'] / row['n_reps']) < 1e-12,
        row['coverage'])
    _ok('finite CIs counted and ordered',
        0 <= row['n_covered'] <= row['n_finite'] <= row['n_reps'], row)
    _ok('nominal and bias are consistent',
        abs(row['nominal'] - 0.95) < 1e-12
        and abs(row['bias'] - (row['coverage'] - row['nominal'])) < 1e-12,
        '')
    pow_rows = mc.power_study(
        [{'name': 'strong', 'n_pairs': 10, 'distribution': 'normal',
          'effect': 2.0, 'alpha': 0.05, 'min_power': 0.8},
         {'name': 'weak', 'n_pairs': 5, 'distribution': 'normal',
          'effect': 0.05, 'alpha': 0.05, 'min_power': 0.2}],
        reps=40, seed=5)
    strong = next(r for r in pow_rows if r['name'] == 'strong')
    weak = next(r for r in pow_rows if r['name'] == 'weak')
    _ok('large effect has high power', strong['rejection_rate'] > 0.8,
        strong['rejection_rate'])
    _ok('near-zero effect has low power',
        weak['rejection_rate'] <= 0.2, weak['rejection_rate'])
    _ok('power rows record their floor',
        all(r['min_power'] == r['threshold'] if 'threshold' in r
            else r['min_power'] > 0 for r in pow_rows), '')


def test_report_is_json_safe_and_gate_consistent():
    print('\nTest: calibration report structure and gate consistency')
    report = mc.calibration_report(reps=40, n_boot=40, seed=6)
    _ok('report has rows and settings',
        isinstance(report['rows'], list) and report['rows']
        and report['settings']['reps'] == 40, '')
    _ok('all_pass matches the gated failures',
        report['all_pass']
        == (not any(r.get('gate', True) and not r['meets']
                    for r in report['rows'])), '')
    _ok('diagnostic failures are reported separately',
        isinstance(report['diagnostic_failed'], list)
        and report['n_diagnostic_failed'] == len(report['diagnostic_failed']),
        '')
    import json
    encoded = json.dumps(report, allow_nan=False)
    _ok('strict-JSON serializable', isinstance(encoded, str), '')


def test_calibration_does_not_touch_global_rng():
    print('\nTest: calibration isolates the global RNG stream')
    random.seed(999)
    expected = [random.random() for _ in range(3)]
    random.seed(999)
    mc.calibration_report(reps=20, n_boot=20, seed=1)
    got = [random.random() for _ in range(3)]
    _ok('global RNG sequence is unchanged', got == expected, f'{got}')


def test_family_study_controls_holm_fwer():
    print('\nTest: family-wise error of Holm')
    scenarios = mc.default_scenarios()['family'][:2]
    rows = mc.family_study(scenarios, reps=120, seed=0)
    _ok('one row per family scenario', len(rows) == len(scenarios), rows)
    for row in rows:
        _ok(f"FWER within bound for {row['name']}",
            row['family_error_rate'] <= row['threshold'],
            row['family_error_rate'])
        _ok(f"family row gates for {row['name']}",
            row['gate'] and not row['diagnostic_only'], row)
        _ok(f"count and rate agree for {row['name']}",
            abs(row['family_error_rate']
                - row['n_significant'] / row['n_reps']) < 1e-12, row)


def test_exp16_row_contract_recomputes_rates_and_verdicts():
    print('\nTest: Exp 16 row contract')
    import results_schema as rs
    report = mc.calibration_report(reps=60, n_boot=60, seed=0)
    meta = {'settings': report['settings']}
    _ok('generated certificate passes the contract',
        rs.validate_rows('exp16_paired_stats_calibration',
                         report['rows'], meta) == [], '')
    bad = [dict(r) for r in report['rows']]
    idx = next(i for i, r in enumerate(bad) if r['section'] == 'size')
    bad[idx] = dict(bad[idx], rejection_rate=1.0)
    violations = rs.validate_rows('exp16_paired_stats_calibration',
                                  bad, meta)
    _ok('tampered rate is rejected',
        any('rejection_rate' in v for v in violations), violations)
    for patch in ({'rejection_rate': None}, {'tol': None},
                  {'method': []}, {'section': []}, {'name': []}):
        bad = [dict(r) for r in report['rows']]
        idx = next(i for i, r in enumerate(bad) if r['section'] == 'size')
        bad[idx] = dict(bad[idx], **patch)
        violations = rs.validate_rows('exp16_paired_stats_calibration',
                                      bad, meta)
        _ok(f'malformed {next(iter(patch))} is rejected without crashing',
            bool(violations), violations)
    bad = [dict(r) for r in report['rows']]
    idx = next(i for i, r in enumerate(bad) if r['section'] == 'coverage')
    bad[idx] = dict(bad[idx], verdict='calibrated', meets=False)
    violations = rs.validate_rows('exp16_paired_stats_calibration',
                                  bad, meta)
    _ok('contradictory coverage verdict is rejected',
        any('verdict' in v for v in violations), violations)


def test_exp16_runner_archives_only_on_pass():
    print('\nTest: Exp 16 runner archives a passing certificate')
    import tempfile
    import run_stats_calibration as runner
    old_cwd = os.getcwd()
    old_env = {k: os.environ.pop(k, None) for k in
               ('STATS_CAL_REPS', 'STATS_CAL_NBOOT', 'STATS_CAL_SEED')}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            rc = runner.main()
            _ok('runner returns 0 on the committed grid', rc == 0, rc)
            path = os.path.join(tmp, runner.RESULT_PATH)
            _ok('result + provenance files exist',
                os.path.exists(path)
                and os.path.exists(os.path.join(tmp, runner.PROVENANCE_PATH)),
                '')
            import json
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            _ok('envelope kind and pass flag',
                data['kind'] == runner.KIND
                and data['metadata']['all_pass'] is True
                and len(data['results']) > 0, '')
    finally:
        os.chdir(old_cwd)
        for key, value in old_env.items():
            if value is not None:
                os.environ[key] = value
    # A failed calibration must not leave a certificate behind.
    old_report = runner.MC.calibration_report
    try:
        runner.MC.calibration_report = lambda **kwargs: {
            'rows': [{'section': 'size', 'name': 'forced',
                      'n_pairs': 5, 'method': 'paired_t', 'alpha': 0.05,
                      'n_reps': 1, 'n_significant': 1,
                      'rejection_rate': 1.0, 'se': 0.0, 'threshold': 0.05,
                      'tol': 0.0, 'meets': False, 'verdict': 'liberal',
                      'gate': True, 'diagnostic_only': False,
                      'note': 'forced'}],
            'all_pass': False, 'n_failed': 1,
            'failed': [{'section': 'size', 'name': 'forced',
                        'method': 'paired_t', 'verdict': 'liberal'}],
            'diagnostic_failed': [], 'n_diagnostic_failed': 0,
            'settings': {}, 'method': 'forced', 'caveats': [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            rc = runner.main()
            _ok('failed gate returns non-zero and writes nothing',
                rc == 1
                and not os.path.exists(os.path.join(tmp, runner.RESULT_PATH)),
                rc)
    finally:
        runner.MC.calibration_report = old_report
        os.chdir(old_cwd)


_TEST_FUNCS = [
    test_simulator_known_population_and_determinism,
    test_size_study_rows_and_diagnostics,
    test_coverage_and_power_rows,
    test_report_is_json_safe_and_gate_consistent,
    test_calibration_does_not_touch_global_rng,
    test_family_study_controls_holm_fwer,
    test_exp16_row_contract_recomputes_rates_and_verdicts,
    test_exp16_runner_archives_only_on_pass,
]


def main() -> int:
    print('=' * 70)
    print('Multi-seed inference calibration tests')
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
