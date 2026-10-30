from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (configure_torch_for_device, parse_int_env,
                          sanitize_for_json, write_json_atomic,
                          write_results_json,
                          make_seeded_generator, set_global_seed,
                          capture_provenance, grad_flow_report,
                          make_run_manifest_entry, append_run_manifest)
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_csa import naive_csa, _l2_normalize
from ops_hca import naive_hca
from ops_fused import HybridConfig
from ops_mamba import build_mamba2_mixer
from ops_rwkv import RWKV7_UPSTREAM_BINDING, build_rwkv7_mixer
from multi_seed_stats import (_safe_mean, _sample_sd,
                               cross_cell_paired_analysis,
                               exact_test_resolution, holm_adjust,
                               summarize_paired)
from experiment_design import (audit_run as audit_experiment_design,
                               common_analyzed_seeds)
from results_analysis import (annotate_rows_with_pareto, diagnose_nonfinite,
                              diagnose_training_curves,
                              multiclass_classification_metrics)
from train_diagnostics import (
    ValPoint,
    ValidationCurve,
    optimizer_update_report,
    pool_error_analyses,
    snapshot_parameters,
    summarize_gradient_norms,
)
from results_schema import CURRENT_SCHEMA_VERSION, attach_integrity

logger = logging.getLogger(__name__)

_VAL_MONITOR_SEED_OFFSET = 6_000_000
# Dedicated stream for bad-case / error analysis. Never reused for
# training, validation monitoring, or the headline eval, so the examples
# examined are not the samples any metric was computed on.
_ERROR_ANALYSIS_SEED_OFFSET = 7_000_000
# Dedicated stream for the non-learned rule-based reference baselines
# (key lookup / majority count). Disjoint from training (+1M), headline
# eval (+5M), validation monitoring (+6M), and error analysis (+7M), so
# computing the rule reference can never perturb any reported metric.
_RULE_BASELINE_SEED_OFFSET = 8_000_000

PAIRWISE_EQUIVALENCE_MARGIN = 0.05

__all__ = [n for n in globals() if not n.startswith('__')]
