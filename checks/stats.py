"""Correctness checks: multi-seed statistics."""

from __future__ import annotations

import json
import random

from ._shared import *  # noqa: F401,F403


def test_effect_size_ci(device='cpu'):
    """Cohen's d_z carries a reproducible, scale-invariant bootstrap CI."""
    from multi_seed_stats import cohens_dz, cohens_dz_ci, summarize_paired

    results = []
    d = [0.4, -0.1, 0.7, 0.2, 0.5]
    point = cohens_dz(d)
    ci = cohens_dz_ci(d, alpha=0.05, n_boot=2000, seed=7)
    results.append(_ok(
        'd_z CI has the requested level and finite ordered bounds',
        ci['ci_level'] == 0.95 and ci['ci_lo'] is not None
        and ci['ci_hi'] is not None and ci['ci_lo'] <= ci['ci_hi'],
        f"ci=[{ci['ci_lo']}, {ci['ci_hi']}]"))
    results.append(_ok(
        'd_z CI is JSON-safe and counts undefined resamples',
        isinstance(ci['n_undefined'], int)
        and 0 <= ci['n_undefined'] <= ci['n_boot'],
        f"n_undefined={ci['n_undefined']}"))
    json.dumps(ci, allow_nan=False)

    # The same seed must produce the identical interval, and the local RNG
    # must never perturb the global stream.
    random.seed(123)
    expected = random.random()
    random.seed(123)
    ci_again = cohens_dz_ci(d, alpha=0.05, n_boot=2000, seed=7)
    results.append(_ok(
        'd_z CI is seed-deterministic and leaves the global RNG alone',
        ci_again == ci and random.random() == expected, ''))

    # d_z is a ratio: rescaling every difference leaves both the point
    # estimate and the standardized interval unchanged.
    ci_scaled = cohens_dz_ci([10.0 * x for x in d], n_boot=2000, seed=7)
    results.append(_ok(
        'd_z point estimate and CI are scale-invariant',
        abs(cohens_dz([10.0 * x for x in d]) - point) < 1e-12
        and abs(ci_scaled['ci_lo'] - ci['ci_lo']) < 1e-12
        and abs(ci_scaled['ci_hi'] - ci['ci_hi']) < 1e-12, ''))

    # Degenerate inputs stay undefined instead of fabricating a zero-width
    # interval.
    for name, values in [('n<2', [0.5]), ('zero variance', [0.5, 0.5])]:
        degenerate = cohens_dz_ci(values, n_boot=100)
        results.append(_ok(
            f'd_z CI is None for {name}',
            degenerate['ci_lo'] is None and degenerate['ci_hi'] is None,
            f'{degenerate["ci_lo"]}, {degenerate["ci_hi"]}'))

    # The one-call panel must report the same interval as the standalone
    # helper (same resampling seed and scheme).
    summary = summarize_paired(d, seed=7, n_boot=2000,
                               n_permutations=1000)
    results.append(_ok(
        'summarize_paired reports the d_z CI and provenance',
        summary['ci95_cohens_dz'] == [ci['ci_lo'], ci['ci_hi']]
        and summary['cohens_dz_bootstrap']['n_boot'] == 2000
        and summary['cohens_dz_bootstrap']['seed'] == 7, ''))
    json.dumps(summary, allow_nan=False)
    return results


__all__ = ['test_effect_size_ci']
