"""Simulation-based calibration of the unit-level perplexity inference.

Why this module?
----------------
``ppl_bootstrap`` turns held-out perplexity into a confidence interval and
two models into a paired test. Those are the statistics every
"significantly better / not significantly worse" claim in the LM results
tables rests on, but the machinery itself was never validated: a CI that
covers the truth 80% of the time at a nominal 95% -- or a paired test that
fires on identical models -- would keep producing plausible-looking
reports. This module closes that gap the standard way: draw synthetic
evaluation corpora from a KNOWN population, run the SHIPPED inference
functions on them, and measure the operating characteristics.

What is measured
----------------
* **Coverage** (:func:`coverage_study`): each replicate draws ``n_units``
  scored units from a population whose token-weighted mean NLL is exactly
  ``mean_nll`` (unit difficulty and token counts independent), runs
  :func:`ppl_bootstrap.bootstrap_ppl_ci`, and checks whether the interval
  contains the population value. The empirical coverage is compared with
  the nominal ``1 - alpha`` at a scenario-specific tolerance. The
  small-unit scenario (``n_units=12``, the Kaggle/AutoDL regime) is
  expected to sit slightly below nominal -- that is a documented property
  of percentile intervals at tiny evaluation sets, not a defect.
* **Size control + power** (:func:`paired_study`): the paired test is run
  (a) on two models drawn from the SAME population (shared unit
  difficulty + independent per-model noise) -- the rejection rate must
  not exceed ``alpha`` by more than its binomial tolerance; and (b) on
  two models separated by a known NLL shift ``delta`` -- the rejection
  rate (power) must clear a pre-registered floor, so a test that never
  rejects anything cannot pass as "well calibrated".

Simulation design
-----------------
* Population: unit difficulty ``d_i = mean_nll + spread * (u1 + u2 - 1)``
  (triangular; ``u1, u2 ~ U(0, 1)``) with ``sd(d) = spread / sqrt(6)``;
  token counts uniform on ``[token_lo, token_hi]``, indepenent of
  difficulty, so the population token-weighted mean NLL is exactly
  ``mean_nll``. Paired models share ``d_i`` (the corpus difficulty that
  cancels in the paired design) and add independent noise
  ``noise_sd * sqrt(3) * (2u - 1)`` per model (sd = ``noise_sd``).
* Every replicate uses a LOCAL ``random.Random`` seeded from
  ``(seed, scenario, replicate)`` with only ``random()``/``randint()``
  primitives (stable across Python versions); the global RNG stream and
  torch's RNG are never touched. The same call reproduces bit-identical
  numbers.
* All reported rates carry their binomial standard error, and every
  scenario declares its own tolerance so the pass/fail logic lives in the
  data (auditable in the result file) rather than in the reader's head.

The public entry point :func:`calibration_report` runs the committed
scenario grid and returns a strict-JSON-safe report (rows + verdicts +
``all_pass``). ``run_ppl_calibration.py`` archives it as Exp 15 after its
parity gate (a failing calibration or a non-parity decode-vs-prefill
check refuses to write the file), and consumes only this module's
functions plus the shipped :mod:`ppl_bootstrap` API.
"""
from __future__ import annotations

import math
import random

import ppl_bootstrap as _pb

__all__ = [
    'simulate_units',
    'simulate_paired_units',
    'coverage_study',
    'paired_study',
    'default_scenarios',
    'calibration_report',
]

# Committed population parameters (used by default_scenarios and the
# docstring numbers): triangular unit difficulty with spread 0.7
# (sd ~= 0.29 nats), token counts uniform on [8, 48].
MEAN_NLL = 3.0
SPREAD = 0.7
TOKEN_LO = 8
TOKEN_HI = 48


def _derive_seed(base_seed: int, *keys: int) -> int:
    """Deterministic 31-bit seed from the base seed and integer keys.

    ``hash()`` is deliberately avoided (string hashing is salted per
    process); with integer-only arithmetic the same call reproduces the
    same seed on every platform/Python version.
    """
    value = int(base_seed) % (2 ** 31 - 1)
    for key in keys:
        value = (value * 1_000_003 + int(key) + 1) % (2 ** 31 - 1)
    return value


def _se(rate: float, n: int) -> float:
    """Binomial standard error of a rate measured over ``n`` trials."""
    if n <= 0:
        return 0.0
    return math.sqrt(max(0.0, rate * (1.0 - rate)) / n)


def _validate_common(n_units: int, n_reps: int, n_boot: int) -> None:
    if int(n_units) < 2:
        raise ValueError(f'n_units must be >= 2 to resample units, '
                         f'got {n_units!r}')
    if int(n_reps) < 1:
        raise ValueError(f'n_reps must be >= 1, got {n_reps!r}')
    if int(n_boot) < 1:
        raise ValueError(f'n_boot must be >= 1, got {n_boot!r}')


def simulate_units(n_units: int, rng: random.Random, *, mean_nll: float = MEAN_NLL,
                   spread: float = SPREAD, token_lo: int = TOKEN_LO,
                   token_hi: int = TOKEN_HI) -> list[tuple[float, int]]:
    """One synthetic scored corpus: ``[(nll_sum, n_tokens), ...]``.

    The population token-weighted mean NLL is exactly ``mean_nll``
    (difficulty and token counts are independent), which is the target the
    coverage study checks the CI against.
    """
    if not 0 < int(token_lo) <= int(token_hi):
        raise ValueError(f'token_lo={token_lo!r} must be in [1, token_hi]')
    if float(spread) < 0.0:
        raise ValueError(f'spread must be >= 0, got {spread!r}')
    if not math.isfinite(float(mean_nll)):
        raise ValueError(f'mean_nll must be finite, got {mean_nll!r}')
    units = []
    for _ in range(int(n_units)):
        tok = rng.randint(int(token_lo), int(token_hi))
        difficulty = float(mean_nll) + float(spread) * (
            rng.random() + rng.random() - 1.0)
        # Guard against the extreme tail crossing zero (nll must be > 0 for
        # exp(); the probability is negligible but the clamp keeps the
        # simulation total).
        difficulty = max(0.05, difficulty)
        units.append((difficulty * tok, tok))
    return units


def simulate_paired_units(n_units: int, rng: random.Random, *,
                          mean_nll: float = MEAN_NLL, spread: float = SPREAD,
                          noise_sd: float = 0.2, delta: float = 0.0,
                          token_lo: int = TOKEN_LO, token_hi: int = TOKEN_HI,
                          ) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Two models scored on the SAME units: shared difficulty + per-model noise.

    Model A has per-unit mean NLL ``d_i + eps_a``, model B ``d_i + delta +
    eps_b``, where the ``eps`` are independent zero-mean uniforms with
    standard deviation ``noise_sd`` (half-width ``noise_sd * sqrt(3)``).
    Both lists carry identical per-unit token counts, so they satisfy
    ``ppl_bootstrap``'s alignment contract and represent exactly the
    paired design used by ``run_quant_lm`` / ``run_lm_consistency``.

    ``delta = 0`` is the null (both models from the same distribution).
    """
    if float(noise_sd) < 0.0:
        raise ValueError(f'noise_sd must be >= 0, got {noise_sd!r}')
    if not math.isfinite(float(delta)):
        raise ValueError(f'delta must be finite, got {delta!r}')
    half_width = float(noise_sd) * math.sqrt(3.0)
    toks = [rng.randint(int(token_lo), int(token_hi))
            for _ in range(int(n_units))]
    diffs = [float(mean_nll) + float(spread) * (rng.random() + rng.random() - 1.0)
             for _ in range(int(n_units))]
    a: list[tuple[float, int]] = []
    b: list[tuple[float, int]] = []
    for i, tok in enumerate(toks):
        base = max(0.05, diffs[i])
        a_mean = max(0.05, base + half_width * (2.0 * rng.random() - 1.0))
        b_mean = max(0.05, base + float(delta)
                     + half_width * (2.0 * rng.random() - 1.0))
        a.append((a_mean * tok, tok))
        b.append((b_mean * tok, tok))
    return a, b


def coverage_study(scenarios, *, n_boot: int = 500, reps: int = 400,
                   seed: int = 0) -> list[dict]:
    """Empirical coverage of ``bootstrap_ppl_ci`` over the given scenarios.

    Scenario keys: ``n_units``, ``alpha``, ``tol`` (absolute tolerance on
    ``coverage - (1 - alpha)``; required so the criterion is pre-declared
    per scenario). Returns one JSON-safe row per scenario with the
    binomial standard error, the signed bias and a
    ``{'calibrated', 'biased'}`` verdict.
    """
    rows: list[dict] = []
    for s_i, scenario in enumerate(scenarios):
        n_units = int(scenario['n_units'])
        alpha = float(scenario['alpha'])
        tol = float(scenario['tol'])
        _validate_common(n_units, reps, n_boot)
        _pb._validate_boot_args(alpha, n_boot)
        if not (math.isfinite(tol) and tol > 0.0):
            raise ValueError(
                f'scenario {s_i}: tol must be positive and finite, got {tol!r}')
        n_covered = 0
        for r in range(int(reps)):
            rng = random.Random(_derive_seed(seed, s_i, r))
            units = simulate_units(n_units, rng)
            report = _pb.bootstrap_ppl_ci(
                units, alpha=alpha, n_boot=n_boot,
                seed=_derive_seed(seed, s_i, r, 7))
            lo, hi = report['ci_mean_nll']
            if lo <= MEAN_NLL <= hi:
                n_covered += 1
        coverage = n_covered / int(reps)
        nominal = 1.0 - alpha
        bias = coverage - nominal
        rows.append({
            'section': 'coverage',
            'n_units': n_units,
            'alpha': alpha,
            'n_boot': int(n_boot),
            'n_reps': int(reps),
            'n_covered': n_covered,
            'coverage': coverage,
            'nominal': nominal,
            'se': _se(coverage, int(reps)),
            'tol': tol,
            'bias': bias,
            'verdict': 'calibrated' if abs(bias) <= tol else 'biased',
            'note': ('fraction of percentile-bootstrap CIs containing the '
                     'known population token-weighted mean NLL'),
        })
    return rows


def paired_study(scenarios, *, n_boot: int = 500, reps: int = 400,
                 seed: int = 0) -> list[dict]:
    """Size control (delta=0) and power (delta>0) of ``paired_bootstrap_ppl``.

    Scenario keys: ``n_units``, ``alpha``, and either ``check='size'``
    (null, ``delta`` must be 0; the pass bound is ``alpha + tol`` with
    ``tol`` required) or ``check='power'`` (shifted alternative,
    ``delta > 0``; the pass floor is ``min_power`` required). Rates carry
    their binomial SE; each row records the bound it was judged against.
    """
    rows: list[dict] = []
    for s_i, scenario in enumerate(scenarios):
        n_units = int(scenario['n_units'])
        alpha = float(scenario['alpha'])
        delta = float(scenario.get('delta', 0.0))
        noise_sd = float(scenario.get('noise_sd', 0.2))
        check = str(scenario.get('check', 'size'))
        _validate_common(n_units, reps, n_boot)
        _pb._validate_boot_args(alpha, n_boot)
        if check not in ('size', 'power'):
            raise ValueError(f'scenario {s_i}: check must be size or power, '
                             f'got {check!r}')
        if check == 'size':
            if delta != 0.0:
                raise ValueError(f'scenario {s_i}: size control needs '
                                 f'delta=0, got {delta!r}')
            tol = float(scenario['tol'])
            if not (math.isfinite(tol) and tol > 0.0):
                raise ValueError(f'scenario {s_i}: tol must be positive, '
                                 f'got {tol!r}')
            threshold = alpha + tol
        else:
            if not (math.isfinite(delta) and delta > 0.0):
                raise ValueError(f'scenario {s_i}: power scenario needs '
                                 f'delta > 0, got {delta!r}')
            floor = float(scenario['min_power'])
            if not (math.isfinite(floor) and 0.0 < floor <= 1.0):
                raise ValueError(f'scenario {s_i}: min_power must be in '
                                 f'(0, 1], got {floor!r}')
            threshold = floor
        n_significant = 0
        for r in range(int(reps)):
            rng = random.Random(_derive_seed(seed, 100 + s_i, r))
            a, b = simulate_paired_units(
                n_units, rng, noise_sd=noise_sd, delta=delta)
            report = _pb.paired_bootstrap_ppl(
                a, b, alpha=alpha, n_boot=n_boot,
                seed=_derive_seed(seed, 100 + s_i, r, 7))
            if report['significant']:
                n_significant += 1
        rate = n_significant / int(reps)
        if check == 'size':
            meets = rate <= threshold
            verdict = 'within_bound' if meets else 'liberal'
        else:
            meets = rate >= threshold
            verdict = 'met' if meets else 'below_floor'
        row = {
            'section': 'paired',
            'check': check,
            'n_units': n_units,
            'alpha': alpha,
            'delta': delta,
            'noise_sd': noise_sd,
            'n_boot': int(n_boot),
            'n_reps': int(reps),
            'n_significant': n_significant,
            'rejection_rate': rate,
            'se': _se(rate, int(reps)),
            'threshold': threshold,
            'verdict': verdict,
            'meets': meets,
            'note': ('fraction of paired-bootstrap tests declared '
                     'significant; size control must stay <= alpha + tol, '
                     'power must clear its floor'),
        }
        if check == 'size':
            row['tol'] = tol
        else:
            row['min_power'] = floor
        rows.append(row)
    return rows


def default_scenarios(*, small_n_units: int = 12) -> dict:
    """The committed Exp 15 protocol (scenarios + study settings).

    * coverage: ``n_units`` in {40, small_n_units} x ``alpha`` in
      {0.10, 0.05}; tolerance 0.05 for the moderate set, 0.08 for the
      small set (documented small-sample undercoverage);
    * paired: size control at the same two unit counts (tolerance 0.06
      for the moderate set; 0.10 for the small set, which is documented
      as mildly liberal at 12 units), plus power at ``delta`` in
      {0.10, 0.20} with pre-registered floors {0.55, 0.90} (a power
      analysis at this geometry measured ~0.67 and ~1.00).
    """
    cov_tol_mid, cov_tol_small = 0.05, 0.08
    coverage = [
        {'n_units': 40, 'alpha': 0.10, 'tol': cov_tol_mid},
        {'n_units': 40, 'alpha': 0.05, 'tol': cov_tol_mid},
        {'n_units': int(small_n_units), 'alpha': 0.10, 'tol': cov_tol_small},
        {'n_units': int(small_n_units), 'alpha': 0.05, 'tol': cov_tol_small},
    ]
    paired = [
        {'n_units': 40, 'alpha': 0.10, 'delta': 0.0, 'check': 'size',
         'tol': 0.06},
        {'n_units': int(small_n_units), 'alpha': 0.10, 'delta': 0.0,
         'check': 'size', 'tol': 0.10},
        {'n_units': 40, 'alpha': 0.10, 'delta': 0.10, 'check': 'power',
         'min_power': 0.55},
        {'n_units': 40, 'alpha': 0.10, 'delta': 0.20, 'check': 'power',
         'min_power': 0.90},
    ]
    return {'coverage': coverage, 'paired': paired}


def calibration_report(*, reps: int = 400, n_boot: int = 500, seed: int = 0,
                       small_n_units: int = 12) -> dict:
    """Run the committed scenario grid; JSON-safe rows + verdict.

    Returns ``{'rows': [...], 'all_pass': bool, 'n_failed': int,
    'settings': {...}, 'method': '...'}``. ``all_pass`` is the parity
    gate: any biased coverage row, liberal size row or below-floor power
    row fails the report.
    """
    scenarios = default_scenarios(small_n_units=small_n_units)
    rows = coverage_study(scenarios['coverage'], n_boot=n_boot, reps=reps,
                          seed=seed)
    rows += paired_study(scenarios['paired'], n_boot=n_boot, reps=reps,
                         seed=seed)
    failed = [r for r in rows if (
        r.get('verdict') in ('biased', 'liberal', 'below_floor'))]
    return {
        'rows': rows,
        'all_pass': not failed,
        'n_failed': len(failed),
        'settings': {
            'reps': int(reps),
            'n_boot': int(n_boot),
            'seed': int(seed),
            'mean_nll': MEAN_NLL,
            'spread': SPREAD,
            'token_lo': TOKEN_LO,
            'token_hi': TOKEN_HI,
        },
        'method': ('simulation-based calibration of ppl_bootstrap on '
                   'synthetic corpora with a known population mean NLL; '
                   'local random.Random(seed, scenario, replicate) streams'),
    }
