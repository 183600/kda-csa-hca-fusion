"""Experiment 4 — MQAR / quality experiments.

Thin entry point.  Task construction, models, error analysis, training,
statistics and the CLI orchestration live in the :mod:`quality` package.
All historical ``run_quality`` symbols are re-exported here.
"""

from __future__ import annotations

import sys

PAIRWISE_EQUIVALENCE_MARGIN = 0.05

from quality._shared import (  # noqa: F401
    _ERROR_ANALYSIS_SEED_OFFSET,
    _RULE_BASELINE_SEED_OFFSET,
    _VAL_MONITOR_SEED_OFFSET,
    logger,
)
from quality.utils import *  # noqa: F401,F403
from quality.stats import *  # noqa: F401,F403
from quality.tasks import *  # noqa: F401,F403
from quality.models import *  # noqa: F401,F403
from quality.errors import *  # noqa: F401,F403
from quality.training import *  # noqa: F401,F403
from quality.main import *  # noqa: F401,F403


if __name__ == '__main__':
    sys.exit(main())
