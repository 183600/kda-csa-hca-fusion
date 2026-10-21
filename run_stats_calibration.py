"""Experiment 16 — calibration certificate for multi-seed inference.

Exp 4/5/8 claims ("layout A beats B", "the effect survives Holm", "the
comparison is equivalent within the margin") all rest on
``multi_seed_stats``. Exp 15 validates the perplexity inference; this
experiment validates the seed-paired inference panel against known
populations:

* null size of the paired t-test, exact sign test, and exact sign-flip
  permutation test (normal / Laplace / contaminated errors);
* percentile-bootstrap CI coverage of a known mean paired difference;
* power at pre-registered standardized effects.

The committed grid is in :func:`multi_seed_calibration.default_scenarios`.
Diagnostic rows (small-n / heavy-tail bootstrap under-coverage and the
t-test under contaminated errors) are reported but do not gate the
certificate; every gated row must meet its pre-registered bound before
the result file is written. The run is pure Python and CPU-cheap (~2 s
with the committed defaults).

Outputs::

    results/exp16_paired_stats_calibration.json
    results/exp16_paired_stats_calibration_provenance.json

Environment overrides: ``STATS_CAL_REPS`` (default 300),
``STATS_CAL_NBOOT`` (default 400), ``STATS_CAL_SEED`` (default 0).
"""
from __future__ import annotations

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kaggle_setup import capture_provenance, write_json_atomic  # noqa: E402
import multi_seed_calibration as MC  # noqa: E402
from results_schema import write_checked  # noqa: E402

logger = logging.getLogger(__name__)

RESULT_PATH = os.path.join('results', 'exp16_paired_stats_calibration.json')
PROVENANCE_PATH = os.path.join(
    'results', 'exp16_paired_stats_calibration_provenance.json')
KIND = 'exp16_paired_stats_calibration'
EXP_REPS = 300
EXP_NBOOT = 400
EXP_SEED = 0


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning('%s is not an int; using default %d', name, default)
        return default


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s')
    reps = _env_int('STATS_CAL_REPS', EXP_REPS)
    n_boot = _env_int('STATS_CAL_NBOOT', EXP_NBOOT)
    seed = _env_int('STATS_CAL_SEED', EXP_SEED)
    logger.info('=' * 70)
    logger.info('Experiment 16: multi-seed inference calibration')
    logger.info('=' * 70)
    logger.info(f'  reps={reps}  n_boot={n_boot}  seed={seed}')

    report = MC.calibration_report(reps=reps, n_boot=n_boot, seed=seed)
    for row in report['rows']:
        if row['section'] == 'size':
            logger.info(f"  size {row['name']:>16s} {row['method']:>11s} "
                        f"rate={row['rejection_rate']:.3f} "
                        f"threshold={row['threshold']:.3f} "
                        f"{row['verdict']}")
        elif row['section'] == 'coverage':
            logger.info(f"  cov  {row['name']:>16s} "
                        f"coverage={row['coverage']:.3f} "
                        f"nominal={row['nominal']:.3f} {row['verdict']}")
        else:
            logger.info(f"  pow  {row['name']:>16s} "
                        f"power={row['rejection_rate']:.3f} "
                        f"floor={row['min_power']:.3f} {row['verdict']}")
    if report['n_diagnostic_failed']:
        logger.warning('  diagnostic rows outside nominal: '
                       f"{report['n_diagnostic_failed']} "
                       '(expected for small-n/heavy-tail bootstrap)')
    logger.info(f"  all_pass={report['all_pass']} "
                f"failed={report['n_failed']}")
    if not report['all_pass']:
        logger.error('  calibration gate FAILED; refusing to archive the '
                     'certificate. Increase seeds/reps or fix the '
                     'inference path. Failed rows: '
                     f"{report['failed']}")
        return 1

    metadata = {
        'experiment': 'exp16_paired_stats_calibration',
        'design': (
            'Simulation-based calibration of the multi_seed_stats panel on '
            'synthetic seed-paired differences with a known population '
            'mean. Null size, percentile-bootstrap CI coverage, and power '
            'are measured at the committed scenario grid; diagnostic rows '
            'are reported but excluded from the gate.'),
        'settings': report['settings'],
        'method': report['method'],
        'caveats': report['caveats'],
        'diagnostic_failed': report['diagnostic_failed'],
        'all_pass': True,
        'provenance': capture_provenance(),
    }
    os.makedirs(os.path.dirname(RESULT_PATH) or '.', exist_ok=True)
    write_checked(RESULT_PATH, report['rows'], kind=KIND, metadata=metadata)
    try:
        write_json_atomic(capture_provenance(), PROVENANCE_PATH,
                          indent=2, allow_nan=False)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info(f'\nSaved: {RESULT_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
