"""Simulation-based calibration of the multi-seed paired-inference panel.

``multi_seed_stats`` decides which operator-vs-operator claims Exp 4/5/8
are allowed to make. Its arithmetic is pinned exactly by
``test_multi_seed_stats.py``, but exact arithmetic is not the same as
calibrated operating characteristics: a perfectly implemented t-test can
still be liberal under a heavy-tailed seed distribution, a percentile
bootstrap can under-cover at n=5, and a "non-significant" result says
nothing about power.

This module closes that gap the same way ``ppl_calibration`` closes it
for the perplexity inference: draw seed-paired differences from a KNOWN
population, run the SHIPPED functions from :mod:`multi_seed_stats`, and
measure:

* **Null size** (:func:`size_study`): rejection rate at ``effect=0`` for
  the paired t-test, exact sign test, and exact sign-flip permutation
  test. The exact tests are distribution-free under the symmetric null;
  the t-test is the one whose normality assumption can fail under
  contaminated errors, so that scenario is reported as a diagnostic and
  does not gate the certificate.
* **CI coverage** (:func:`coverage_study`): does the percentile
  bootstrap interval for the mean paired difference contain the known
  population effect at the nominal rate?
* **Power** (:func:`power_study`): rejection rate at pre-registered
  standardized effects, with a pre-registered floor. A test that never
  rejects cannot pass as "well calibrated".

Design constraints
------------------
* Stdlib only; local ``random.Random`` seeded from
  ``(seed, scenario, replicate)``; no global RNG use.
* Committed scenario grid in :func:`default_scenarios`; every row carries
  its threshold/tolerance, binomial SE and a verdict, so the pass/fail
  rule is auditable in the result file rather than implicit.
* Normal and Laplace noise have the same total standard deviation;
  the contaminated mixture is rescaled so its marginal SD also matches,
  making ``effect`` a standardized effect size across scenarios.
* A row marked ``gate=False`` is diagnostic only (the t-test under
  contaminated errors is expected to be somewhat liberal); it is still
  reported so the assumption violation is visible.

Run the committed protocol directly with::

    python3 multi_seed_calibration.py
"""
from __future__ import annotations

import math
import random
import sys

from multi_seed_stats import (
    bootstrap_paired_ci,
    holm_adjust,
    paired_t_test,
    permutation_test,
    sign_test,
)


def _is_finite_real(value) -> bool:
    """True for a non-bool real representable as a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False

__all__ = [
    'simulate_paired_diffs',
    'size_study',
    'coverage_study',
    'power_study',
    'family_study',
    'default_scenarios',
    'calibration_report',
]

DEFAULT_REPS = 300
DEFAULT_NBOOT = 400
DEFAULT_ALPHA = 0.05
DEFAULT_NOISE_SD = 1.0
OUTLIER_PROB = 0.10
OUTLIER_SCALE = 5.0
DISTRIBUTIONS = ('normal', 'laplace', 'contaminated')
SIZE_METHODS = ('paired_t', 'sign', 'permutation')


def _derive_seed(base_seed, *keys):
    """Deterministic 31-bit seed from integer keys (no salted ``hash``)."""
    value = int(base_seed) % (2 ** 31 - 1)
    for key in keys:
        value = (value * 1_000_003 + int(key) + 1) % (2 ** 31 - 1)
    return value


def _require_int(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an exact int, got {value!r}')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    return value


def _validate_study_args(reps, n_boot, seed):
    _require_int(reps, 'reps', minimum=1)
    _require_int(n_boot, 'n_boot', minimum=1)
    _require_int(seed, 'seed')


def _se(rate, n):
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, rate * (1.0 - rate) / n))


def _laplace(rng, scale):
    """Zero-mean Laplace draw via inverse CDF, using only ``random()``."""
    u = rng.random() - 0.5
    if u == 0.0:
        return 0.0
    # ``random()`` can return exactly 0.0 (probability 2**-53), which maps
    # to the endpoint u=-0.5 and a log(0) = -inf draw. Clamp the magnitude
    # one representable step inside the interval so the inverse CDF stays
    # finite (the same tail draw, rounded to the nearest representable u).
    a = min(abs(u), 0.5 - 2.0 ** -53)
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * a)


def _base_sd_for_mixture(noise_sd, outlier_prob, outlier_scale):
    """Base SD so the contaminated mixture's marginal SD equals ``noise_sd``."""
    factor = math.sqrt((1.0 - outlier_prob)
                       + outlier_prob * outlier_scale ** 2)
    return noise_sd / factor


def simulate_paired_diffs(n_pairs, rng, *, effect=0.0, noise_sd=1.0,
                          distribution='normal', outlier_prob=OUTLIER_PROB,
                          outlier_scale=OUTLIER_SCALE):
    """Draw ``n_pairs`` seed-paired differences from a known population.

    The population mean is exactly ``effect``; ``noise_sd`` is the marginal
    standard deviation for all three distributions. ``laplace`` uses scale
    ``noise_sd / sqrt(2)``; ``contaminated`` draws a Gaussian outlier with
    probability ``outlier_prob`` and rescales the base component so the
    mixture SD stays ``noise_sd``.
    """
    _require_int(n_pairs, 'n_pairs', minimum=2)
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f'distribution={distribution!r} not in '
                         f'{DISTRIBUTIONS}')
    if not (isinstance(effect, (int, float))
            and not isinstance(effect, bool)
            and _is_finite_real(effect)):
        raise ValueError(f'effect must be finite, got {effect!r}')
    if not (isinstance(noise_sd, (int, float))
            and not isinstance(noise_sd, bool)
            and _is_finite_real(noise_sd) and float(noise_sd) > 0.0):
        raise ValueError(f'noise_sd must be finite and > 0, got {noise_sd!r}')
    if not (isinstance(outlier_prob, (int, float))
            and not isinstance(outlier_prob, bool)
            and _is_finite_real(outlier_prob)
            and 0.0 <= float(outlier_prob) < 1.0):
        raise ValueError(f'outlier_prob must be in [0, 1), got {outlier_prob!r}')
    if not (isinstance(outlier_scale, (int, float))
            and not isinstance(outlier_scale, bool)
            and _is_finite_real(outlier_scale)
            and float(outlier_scale) > 0.0):
        raise ValueError(f'outlier_scale must be finite and > 0, got '
                         f'{outlier_scale!r}')
    diffs = []
    for _ in range(n_pairs):
        if distribution == 'normal':
            noise = rng.gauss(0.0, float(noise_sd))
        elif distribution == 'laplace':
            noise = _laplace(rng, float(noise_sd) / math.sqrt(2.0))
        else:
            base = _base_sd_for_mixture(
                float(noise_sd), float(outlier_prob), float(outlier_scale))
            scale = (base * float(outlier_scale)
                     if rng.random() < float(outlier_prob) else base)
            noise = rng.gauss(0.0, scale)
        diffs.append(float(effect) + noise)
    return diffs


def _p_value(method, diffs, *, alpha, seed):
    if method == 'paired_t':
        return paired_t_test(diffs, alternative='two-sided')['p_value']
    if method == 'sign':
        return sign_test(diffs)['p_value']
    if method == 'permutation':
        return permutation_test(diffs, alternative='two-sided', seed=seed,
                                max_exact=16)['p_value']
    raise ValueError(f'unknown method {method!r}')


def size_study(scenarios, *, reps=DEFAULT_REPS, seed=0):
    """Empirical null rejection rate for every scenario/method.

    Scenario keys: ``n_pairs``, ``distribution``, ``alpha``, ``tol``
    (absolute slack above alpha), optional ``noise_sd``,
    ``diagnostic_methods`` (methods that are reported but do not gate the
    certificate), and ``name``.
    """
    _validate_study_args(reps, DEFAULT_NBOOT, seed)
    rows = []
    for s_i, scenario in enumerate(scenarios):
        n_pairs = _require_int(scenario['n_pairs'], f'scenario {s_i} n_pairs',
                               minimum=2)
        alpha = float(scenario.get('alpha', DEFAULT_ALPHA))
        tol = float(scenario['tol'])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'scenario {s_i}: alpha must be in (0, 1)')
        if not (math.isfinite(tol) and tol > 0.0):
            raise ValueError(f'scenario {s_i}: tol must be positive')
        distribution = str(scenario.get('distribution', 'normal'))
        noise_sd = float(scenario.get('noise_sd', DEFAULT_NOISE_SD))
        diagnostics = set(scenario.get('diagnostic_methods', ()))
        name = str(scenario.get('name', f'{distribution}_n{n_pairs}'))
        counts = {m: 0 for m in SIZE_METHODS}
        for r in range(int(reps)):
            rng = random.Random(_derive_seed(seed, 10 + s_i, r))
            diffs = simulate_paired_diffs(
                n_pairs, rng, effect=0.0, noise_sd=noise_sd,
                distribution=distribution)
            for method in SIZE_METHODS:
                p = _p_value(method, diffs, alpha=alpha,
                             seed=_derive_seed(seed, 10 + s_i, r, 99))
                if p is not None and p < alpha:
                    counts[method] += 1
        threshold = alpha + tol
        for method in SIZE_METHODS:
            rate = counts[method] / int(reps)
            meets = rate <= threshold
            # A rate far BELOW the nominal level is a conservative size
            # control, not "within bound" in the symmetric sense. Check it
            # before the meets branch: otherwise the conservative verdict
            # is unreachable (rate < alpha - tol already implies meets).
            if rate < alpha - tol:
                verdict = 'conservative'
            elif meets:
                verdict = 'within_bound'
            else:
                verdict = 'liberal'
            rows.append({
                'section': 'size',
                'name': name,
                'n_pairs': n_pairs,
                'distribution': distribution,
                'noise_sd': noise_sd,
                'method': method,
                'alpha': alpha,
                'n_reps': int(reps),
                'n_significant': counts[method],
                'rejection_rate': rate,
                'se': _se(rate, int(reps)),
                'threshold': threshold,
                'tol': tol,
                'meets': meets,
                'verdict': verdict,
                'gate': method not in diagnostics,
                'diagnostic_only': method in diagnostics,
                'note': ('fraction of null replicates rejected; exact '
                         'tests are distribution-free under a symmetric '
                         'null, the t-test is diagnostic under contaminated '
                         'errors'),
            })
    return rows


def coverage_study(scenarios, *, reps=DEFAULT_REPS, n_boot=DEFAULT_NBOOT,
                   seed=0):
    """Empirical coverage of ``bootstrap_paired_ci`` for a known effect."""
    _validate_study_args(reps, n_boot, seed)
    rows = []
    for s_i, scenario in enumerate(scenarios):
        n_pairs = _require_int(scenario['n_pairs'], f'scenario {s_i} n_pairs',
                               minimum=2)
        alpha = float(scenario.get('alpha', DEFAULT_ALPHA))
        tol = float(scenario['tol'])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'scenario {s_i}: alpha must be in (0, 1)')
        if not (math.isfinite(tol) and tol > 0.0):
            raise ValueError(f'scenario {s_i}: tol must be positive')
        distribution = str(scenario.get('distribution', 'normal'))
        noise_sd = float(scenario.get('noise_sd', DEFAULT_NOISE_SD))
        effect = float(scenario.get('effect', 0.5))
        name = str(scenario.get('name', f'{distribution}_n{n_pairs}'))
        diagnostic = bool(scenario.get('diagnostic', False))
        n_covered = 0
        n_finite = 0
        for r in range(int(reps)):
            rng = random.Random(_derive_seed(seed, 100 + s_i, r))
            diffs = simulate_paired_diffs(
                n_pairs, rng, effect=effect, noise_sd=noise_sd,
                distribution=distribution)
            ci = bootstrap_paired_ci(
                diffs, alpha=alpha, n_boot=n_boot,
                seed=_derive_seed(seed, 100 + s_i, r, 7))
            lo, hi = ci['ci_lo'], ci['ci_hi']
            if lo is not None and hi is not None:
                n_finite += 1
                if lo <= effect <= hi:
                    n_covered += 1
        coverage = n_covered / int(reps)
        nominal = 1.0 - alpha
        bias = coverage - nominal
        meets = abs(bias) <= tol
        rows.append({
            'section': 'coverage',
            'name': name,
            'n_pairs': n_pairs,
            'distribution': distribution,
            'noise_sd': noise_sd,
            'effect': effect,
            'alpha': alpha,
            'n_boot': int(n_boot),
            'n_reps': int(reps),
            'n_finite': n_finite,
            'n_covered': n_covered,
            'coverage': coverage,
            'nominal': nominal,
            'se': _se(coverage, int(reps)),
            'tol': tol,
            'bias': bias,
            'meets': meets,
            'verdict': 'calibrated' if meets else 'biased',
            'gate': not diagnostic,
            'diagnostic_only': diagnostic,
            'note': ('fraction of percentile-bootstrap CIs containing the '
                     'known population mean paired difference'
                     + ('; DIAGNOSTIC: small-n / heavy-tail percentile '
                        'bootstrap is expected to under-cover, so this row '
                        'does not gate the certificate' if diagnostic else '')),
        })
    return rows


def power_study(scenarios, *, reps=DEFAULT_REPS, seed=0):
    """Empirical power of the paired t-test at pre-registered effects."""
    _validate_study_args(reps, DEFAULT_NBOOT, seed)
    rows = []
    for s_i, scenario in enumerate(scenarios):
        n_pairs = _require_int(scenario['n_pairs'], f'scenario {s_i} n_pairs',
                               minimum=2)
        alpha = float(scenario.get('alpha', DEFAULT_ALPHA))
        effect = float(scenario['effect'])
        floor = float(scenario['min_power'])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'scenario {s_i}: alpha must be in (0, 1)')
        if not (math.isfinite(effect) and effect > 0.0):
            raise ValueError(f'scenario {s_i}: effect must be > 0')
        if not (math.isfinite(floor) and 0.0 < floor <= 1.0):
            raise ValueError(f'scenario {s_i}: min_power must be in (0, 1]')
        distribution = str(scenario.get('distribution', 'normal'))
        noise_sd = float(scenario.get('noise_sd', DEFAULT_NOISE_SD))
        name = str(scenario.get('name', f'{distribution}_n{n_pairs}'))
        n_significant = 0
        for r in range(int(reps)):
            rng = random.Random(_derive_seed(seed, 200 + s_i, r))
            diffs = simulate_paired_diffs(
                n_pairs, rng, effect=effect, noise_sd=noise_sd,
                distribution=distribution)
            p = paired_t_test(diffs, alternative='two-sided')['p_value']
            if p is not None and p < alpha:
                n_significant += 1
        rate = n_significant / int(reps)
        meets = rate >= floor
        rows.append({
            'section': 'power',
            'name': name,
            'n_pairs': n_pairs,
            'distribution': distribution,
            'noise_sd': noise_sd,
            'effect': effect,
            'alpha': alpha,
            'method': 'paired_t',
            'n_reps': int(reps),
            'n_significant': n_significant,
            'rejection_rate': rate,
            'se': _se(rate, int(reps)),
            'min_power': floor,
            'meets': meets,
            'verdict': 'met' if meets else 'below_floor',
            'gate': True,
            'note': ('fraction of shifted-effect replicates rejected by the '
                     'two-sided paired t-test; floor is pre-registered'),
        })
    return rows


def family_study(scenarios, *, reps=DEFAULT_REPS, seed=0):
    """Empirical family-wise error rate of Holm over paired t-tests.

    Scenario keys: ``m`` (family size >= 2), ``n_pairs``,
    ``distribution``, ``alpha``, ``tol``, optional ``noise_sd`` and
    ``name``. Every comparison is a null (``effect=0``) paired t-test and
    the replicate is a family error if Holm rejects at least one. This
    calibrates the correction actually used by Exp 4/5/8, not just the
    per-comparison tests.
    """
    _validate_study_args(reps, DEFAULT_NBOOT, seed)
    rows = []
    for s_i, scenario in enumerate(scenarios):
        m = _require_int(scenario['m'], f'scenario {s_i} m', minimum=2)
        n_pairs = _require_int(scenario['n_pairs'], f'scenario {s_i} n_pairs',
                               minimum=2)
        alpha = float(scenario.get('alpha', DEFAULT_ALPHA))
        tol = float(scenario['tol'])
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'scenario {s_i}: alpha must be in (0, 1)')
        if not (math.isfinite(tol) and tol > 0.0):
            raise ValueError(f'scenario {s_i}: tol must be positive')
        distribution = str(scenario.get('distribution', 'normal'))
        noise_sd = float(scenario.get('noise_sd', DEFAULT_NOISE_SD))
        diagnostic = bool(scenario.get('diagnostic', False))
        name = str(scenario.get('name', f'{distribution}_m{m}_n{n_pairs}'))
        n_family = 0
        for r in range(int(reps)):
            p_values = []
            for j in range(m):
                rng = random.Random(_derive_seed(seed, 300 + s_i, r, j))
                diffs = simulate_paired_diffs(
                    n_pairs, rng, effect=0.0, noise_sd=noise_sd,
                    distribution=distribution)
                p_values.append(
                    paired_t_test(diffs, alternative='two-sided')['p_value'])
            holm = holm_adjust(
                [1.0 if p is None else p for p in p_values], alpha=alpha)
            if any(holm['rejected']):
                n_family += 1
        rate = n_family / int(reps)
        threshold = alpha + tol
        meets = rate <= threshold
        rows.append({
            'section': 'family',
            'name': name,
            'm': m,
            'n_pairs': n_pairs,
            'distribution': distribution,
            'noise_sd': noise_sd,
            'method': 'holm_paired_t',
            'alpha': alpha,
            'n_reps': int(reps),
            'n_significant': n_family,
            'family_error_rate': rate,
            'se': _se(rate, int(reps)),
            'threshold': threshold,
            'tol': tol,
            'meets': meets,
            'verdict': 'within_bound' if meets else 'liberal',
            'gate': not diagnostic,
            'diagnostic_only': diagnostic,
            'note': ('fraction of null families where Holm rejected at '
                     'least one paired t-test; FWER must stay <= alpha + '
                     'tol'),
        })
    return rows


def default_scenarios():
    """The committed calibration grid (normal / Laplace / contaminated)."""
    return {
        'size': [
            {'name': 'normal_n5', 'n_pairs': 5, 'distribution': 'normal',
             'alpha': 0.05, 'tol': 0.04},
            {'name': 'normal_n10', 'n_pairs': 10, 'distribution': 'normal',
             'alpha': 0.05, 'tol': 0.03},
            {'name': 'laplace_n7', 'n_pairs': 7, 'distribution': 'laplace',
             'alpha': 0.05, 'tol': 0.05},
            {'name': 'contaminated_n7', 'n_pairs': 7,
             'distribution': 'contaminated', 'alpha': 0.05, 'tol': 0.06,
             'diagnostic_methods': ['paired_t']},
        ],
        'coverage': [
            {'name': 'normal_n5', 'n_pairs': 5, 'distribution': 'normal',
             'effect': 0.5, 'alpha': 0.05, 'tol': 0.06,
             'diagnostic': True},
            {'name': 'normal_n10', 'n_pairs': 10, 'distribution': 'normal',
             'effect': 0.5, 'alpha': 0.05, 'tol': 0.05},
            {'name': 'contaminated_n10', 'n_pairs': 10,
             'distribution': 'contaminated', 'effect': 0.5, 'alpha': 0.05,
             'tol': 0.06, 'diagnostic': True},
        ],
        'power': [
            {'name': 'normal_n7', 'n_pairs': 7, 'distribution': 'normal',
             'effect': 1.0, 'alpha': 0.05, 'min_power': 0.55},
            {'name': 'normal_n10', 'n_pairs': 10, 'distribution': 'normal',
             'effect': 0.8, 'alpha': 0.05, 'min_power': 0.55},
            {'name': 'normal_n10_strong', 'n_pairs': 10,
             'distribution': 'normal', 'effect': 1.2, 'alpha': 0.05,
             'min_power': 0.85},
        ],
        'family': [
            {'name': 'normal_m6_n5', 'm': 6, 'n_pairs': 5,
             'distribution': 'normal', 'alpha': 0.05, 'tol': 0.04},
            {'name': 'normal_m12_n5', 'm': 12, 'n_pairs': 5,
             'distribution': 'normal', 'alpha': 0.05, 'tol': 0.04},
            {'name': 'contaminated_m6_n7', 'm': 6, 'n_pairs': 7,
             'distribution': 'contaminated', 'alpha': 0.05, 'tol': 0.06,
             'diagnostic': True},
        ],
    }


def calibration_report(*, reps=DEFAULT_REPS, n_boot=DEFAULT_NBOOT, seed=0):
    """Run the committed grid and return a JSON-safe certificate."""
    scenarios = default_scenarios()
    rows = size_study(scenarios['size'], reps=reps, seed=seed)
    rows += coverage_study(scenarios['coverage'], reps=reps, n_boot=n_boot,
                           seed=seed)
    rows += power_study(scenarios['power'], reps=reps, seed=seed)
    rows += family_study(scenarios['family'], reps=reps, seed=seed)
    failed = [r for r in rows if r.get('gate', True) and not r['meets']]
    diagnostic_failed = [
        r for r in rows if not r.get('gate', True) and not r['meets']]
    return {
        'rows': rows,
        'all_pass': not failed,
        'n_failed': len(failed),
        'failed': [{'section': r['section'], 'name': r['name'],
                    'method': r.get('method'), 'verdict': r['verdict']}
                   for r in failed],
        'n_diagnostic_failed': len(diagnostic_failed),
        'diagnostic_failed': [
            {'section': r['section'], 'name': r['name'],
             'method': r.get('method'), 'verdict': r['verdict']}
            for r in diagnostic_failed],
        'caveats': [
            'n=5 percentile-bootstrap intervals for the mean paired '
            'difference under-cover at nominal 95% in the committed grid '
            '(~0.84 measured); they are reported as diagnostic only. Use '
            'the exact sign/permutation tests for 5-seed directional '
            'claims and treat the BCI as a lower bound on uncertainty.',
            'The contaminated scenario has 10% outliers at 5x the base '
            'scale; the t-test and the mean bootstrap are expected to lose '
            'efficiency there. That scenario is diagnostic, not a gate.',
            'The family study measures FWER of Holm over independent null '
            'paired t-tests; real operator comparisons are positively '
            'dependent, which makes Holm conservative rather than liberal.',
        ],
        'settings': {
            'reps': int(reps),
            'n_boot': int(n_boot),
            'seed': int(seed),
            'alpha': DEFAULT_ALPHA,
            'noise_sd': DEFAULT_NOISE_SD,
            'outlier_prob': OUTLIER_PROB,
            'outlier_scale': OUTLIER_SCALE,
        },
        'method': ('simulation-based calibration of multi_seed_stats on '
                   'synthetic seed differences with known population mean; '
                   'local random.Random(seed, scenario, replicate) streams'),
    }


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    reps = DEFAULT_REPS
    n_boot = DEFAULT_NBOOT
    seed = 0
    output = None
    i = 0
    while i < len(args):
        if args[i] == '--reps' and i + 1 < len(args):
            reps = int(args[i + 1]); i += 2
        elif args[i] == '--n-boot' and i + 1 < len(args):
            n_boot = int(args[i + 1]); i += 2
        elif args[i] == '--seed' and i + 1 < len(args):
            seed = int(args[i + 1]); i += 2
        elif args[i] == '--json' and i + 1 < len(args):
            output = args[i + 1]; i += 2
        else:
            raise SystemExit(f'unknown argument {args[i]!r}')
    report = calibration_report(reps=reps, n_boot=n_boot, seed=seed)
    if output:
        # Imported lazily so this pure-statistics module stays importable in
        # a minimal environment without torch (kaggle_setup imports torch).
        from kaggle_setup import write_json_atomic
        write_json_atomic(report, output, indent=2, allow_nan=False)
    for row in report['rows']:
        if row['section'] == 'size':
            print(f"size {row['name']:>16s} {row['method']:>10s} "
                  f"rate={row['rejection_rate']:.3f} "
                  f"threshold={row['threshold']:.3f} {row['verdict']}")
        elif row['section'] == 'coverage':
            print(f"cov  {row['name']:>16s} coverage={row['coverage']:.3f} "
                  f"nominal={row['nominal']:.3f} {row['verdict']}")
        elif row['section'] == 'power':
            print(f"pow  {row['name']:>16s} power={row['rejection_rate']:.3f} "
                  f"floor={row['min_power']:.3f} {row['verdict']}")
        else:
            print(f"fam  {row['name']:>16s} m={row['m']} "
                  f"FWER={row['family_error_rate']:.3f} "
                  f"threshold={row['threshold']:.3f} {row['verdict']}")
    print(f"all_pass={report['all_pass']} failed={report['n_failed']}")
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
