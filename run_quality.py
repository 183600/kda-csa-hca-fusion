"""Experiment 4 — synthetic quality proxy (Multi-Query Associative Recall).

MQAR (arXiv:2302.13071) is the canonical associative-recall probe used to
evaluate linear-attention variants: the model must recall one of ``n_kv``
key-value pairs presented earlier in the sequence when cued by its key.
KDA's delta rule + fine-grained gating is designed exactly for this kind of
in-context associative memory; CSA's sparse retrieval should also help; HCA's
heavy compression trades recall granularity for global context.

We train a tiny single-layer classifier on top of each attention operator for
a fixed number of steps and report final accuracy. Absolute accuracy is not
meaningful at this scale, but the *relative ranking* across operators is.

Methodology (experiment-design contract — see also TASK_HYPOTHESES and
OPERATOR_HYPOTHESES below, which pre-register what each cell tests):

  * **Baseline ladder, tuned.** The sweep includes the SIMPLEST baselines —
    ``linear`` (no token mixing at all; a memoryless control that must stay
    at chance on a memory task) and ``softmax`` (the simplest real mixer).
    Baseline learning rates are selected on the held-out validation stream
    (``MQAR_BASELINE_TUNE=1``, default), and every complex operator is
    additionally tested against the tuned reference baseline with a
    SEED-PAIRED one-sided t-test, Bonferroni-corrected over the comparison
    family (``beats_tuned_baseline``). A complex model must clear that bar
    before "better than the baseline" may be claimed.
  * **Multi-seed runs with confidence intervals.** Every operator is trained
    over ``n_seeds`` (default 5) seeds; we report mean +/- std, a
    one-sample t-test vs the chance baseline, and — for the multi-class
    probes — accuracy PLUS the confusion-derived metrics: macro / micro /
    weighted F1 (explicitly labelled), balanced accuracy, per-class
    precision/recall/support, and one-vs-rest PR-AUC / ROC-AUC. The
    synthetic MQAR targets are class-balanced by construction, so the
    metric record also VERIFIES the class balance (``class_balance``)
    instead of assuming it, and a collapsed predictor cannot hide behind
    accuracy. PR-AUC is the primary ranking metric when positives are rare;
    these probes are 1-of-vocab ranked decisions, so the same convention
    applies.
  * **Equivalence, not just significance.** Pairwise operator comparisons
    additionally run a TOST against a PRE-SPECIFIED smallest effect of
    interest (``PAIRWISE_EQUIVALENCE_MARGIN`` / ``MQAR_EQUIV_MARGIN``).
    A non-significant superiority test is never read as "no difference";
    only the TOST can positively support a "practically the same" claim,
    and each pair records the smallest margin the current seed budget
    could have supported.
  * **Softmax baseline convergence.** The softmax-attention baseline can be
    given more steps (``softmax_steps``) so it actually converges; a
    converged softmax is the right upper bound.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling back
    to CPU.
  * **Controlled evaluation.** A larger final eval batch (256) and multiple
    eval batches for a more stable accuracy estimate.
  * **Hypothesis-driven, not exploratory.** Each task row carries a
    pre-registered mechanism hypothesis with its predicted signature and
    falsification criterion (TASK_HYPOTHESES / OPERATOR_HYPOTHESES); the
    flags and metadata record which predictions the data supported.
"""

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
                          make_seeded_generator, set_global_seed,
                          capture_provenance, grad_flow_report,
                          make_run_manifest_entry, append_run_manifest)
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_fused import HybridConfig
from ops_mamba import build_mamba2_mixer
from ops_rwkv import RWKV7_UPSTREAM_BINDING, build_rwkv7_mixer
from multi_seed_stats import _sample_sd, holm_adjust, summarize_paired
from results_analysis import (annotate_rows_with_pareto, diagnose_nonfinite,
                              diagnose_training_curves,
                              multiclass_classification_metrics)
from train_diagnostics import ValPoint, ValidationCurve
from results_schema import CURRENT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Dedicated synthetic-data stream for the held-out validation monitoring
# curve. Kept distinct from the training stream (seed + 1_000_000) and from
# the final headline eval stream (seed + 5_000_000) so that (a) the
# validation curve can never leak into the reported eval metric, and
# (b) enabling monitoring cannot change any reported number (monitoring is
# read-only and RNG-isolated).
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

# Pre-specified smallest effect of interest (SESOI) for the seed-paired
# operator comparisons, in ABSOLUTE accuracy units. This is the margin
# parameter of the TOST equivalence test in ``multi_seed_stats``: two
# operators separated by less than this are declared practically
# equivalent (at the corrected alpha), and a non-significant superiority
# test alone NEVER licenses a "same" claim (absence of evidence is not
# evidence of absence — see ``_pairwise_op_comparisons``).
#
# Rationale for 0.05: the MQAR probe's chance level is 1/16 = 0.0625, and
# the README fairness note warns that several ablations land within a few
# percentage points of chance, so differences below 5 accuracy points are
# not practically meaningful at this model/task scale. The margin is
# fixed HERE, and overridable only via the pre-run ``MQAR_EQUIV_MARGIN``
# environment variable (``0`` disables equivalence testing). It must not
# be tuned against the observed CIs: choosing the margin after seeing the
# data would invalidate the TOST p-value.
PAIRWISE_EQUIVALENCE_MARGIN = 0.05


def _parse_equivalence_margin(env_var, default):
    """Parse the pre-specified TOST margin from an env var.

    ``None`` / unset -> ``default``; ``0`` -> ``None`` (disable TOST);
    a malformed or non-positive value falls back to ``default`` with a
    warning. Returning ``None`` is the explicit "no equivalence testing"
    signal consumed by ``_pairwise_op_comparisons``.
    """
    raw = os.environ.get(env_var)
    if raw is None or not str(raw).strip():
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            f'{env_var}={raw!r} is not a float; using the pre-specified '
            f'default margin {default!r}')
        return default
    if val == 0.0:
        return None
    if not math.isfinite(val) or val < 0.0:
        logger.warning(
            f'{env_var}={raw!r} must be finite and >= 0; using the '
            f'pre-specified default margin {default!r}')
        return default
    return val


def _build_param_groups(*modules, weight_decay=0.01):
    """Build AdamW parameter groups with proper weight-decay exclusion.

    Standard ML practice: embeddings, biases, and LayerNorm parameters should
    NOT be weight-decayed. Weight decay on these can hurt training quality
    (e.g. shrinking the embedding table towards zero degrades representation
    quality; decaying LayerNorm affine params breaks normalization statistics).

    Grouping rules:
      * Parameters from ``nn.Embedding`` modules -> no decay
      * Parameters from ``nn.LayerNorm`` modules -> no decay
      * 1-D parameters (biases, e.g. ``nn.Conv1d.bias``, ``nn.Linear.bias``)
        -> no decay
      * All other parameters (e.g. ``nn.Linear.weight``, ``nn.Conv1d.weight``,
        ``nn.Parameter`` tensors of ndim >= 2) -> decay

    Note: we check the MODULE TYPE (not the parameter name) to identify
    embeddings and LayerNorms, because ``named_parameters()`` on a submodule
    returns names RELATIVE to that submodule (e.g. just ``weight``, not
    ``embed.weight``). A name-based heuristic would miss the embedding table.

    The CSA/HCA positional biases (``Ba``, ``Bb``, ``B_idx``, ``B_pos``) are
    2-D ``nn.Parameter`` tensors of shape ``(m, c)`` or ``(m, c_I)`` but
    function analogously to embeddings (they are lookup tables indexed by
    block-position, not weights applied to activations). Decaying them
    shrinks the table toward zero and degrades the model's ability to
    represent position-dependent compression patterns. We exclude them
    from decay via an explicit name suffix match (their ndim == 2 means
    the generic ``p.ndim <= 1`` rule does not catch them).

    Returns a list of param groups suitable for ``torch.optim.AdamW``.
    """
    # nn.Parameter names ending in any of these suffixes are positional
    # bias tables and should NOT be weight-decayed. Listed explicitly
    # because they are 2-D (so the ``ndim <= 1`` rule misses them) and
    # are not submodules of nn.Embedding/nn.LayerNorm (so the module-type
    # rule misses them too).
    _POSITIONAL_BIAS_SUFFIXES = ('Ba', 'Bb', 'B_idx', 'B_pos')

    no_decay = []
    decay = []
    for module in modules:
        # Collect the ids of parameters that belong to Embedding / LayerNorm
        # submodules. We walk the module tree once so that nested modules
        # (e.g. ``layer.norm`` inside ``ResidualAttnLayer``) are correctly
        # identified by their module type, not by a fragile name match.
        no_decay_ids = set()
        for submod in module.modules():
            if isinstance(submod, (nn.Embedding, nn.LayerNorm)):
                for p in submod.parameters(recurse=False):
                    no_decay_ids.add(id(p))
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # Strip the trailing ``.weight`` / ``.bias`` etc. so the suffix
            # check below matches the leaf parameter's own name, not the
            # parent module's. e.g. ``csa_layers.0.Ba`` -> ``Ba``.
            leaf_name = name.rsplit('.', 1)[-1]
            # Exclude: Embedding/LayerNorm params (by module type), 1-D params
            # (biases), and 2-D positional-bias parameters (Ba/Bb/B_idx/B_pos,
            # which function like embeddings and would be shrunk by decay).
            is_no_decay = (
                id(p) in no_decay_ids
                or p.ndim <= 1
                or leaf_name in _POSITIONAL_BIAS_SUFFIXES
            )
            if is_no_decay:
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def _fmt_tstat(t, width=12, prec=2):
    """Format a t-statistic for display, None-safe.

    The one-sample t-test vs chance is undefined for n == 1 or when the
    sample standard deviation is zero; in those cases ``t_stat`` is None
    and we render ``"n/a"`` instead of a number.
    """
    if t is None:
        return 'n/a'.rjust(width)
    return f'{t:.{prec}f}'.rjust(width)


# Module-level cache for scipy.stats.t.ppf.
#   None  -> not yet probed
#   False -> scipy unavailable, use the fallback table
#   <fn>  -> scipy.stats.t.ppf callable
_T_PP = None


def _t_crit_975(n):
    """Two-sided 95% critical value of the t-distribution with ``n-1`` dof.

    Returns ``t`` such that ``P(-t < T < t) = 0.95`` where ``T`` follows a
    Student's t distribution with ``n - 1`` degrees of freedom (i.e. the
    one-sided 97.5% quantile, ``scipy.stats.t.ppf(0.975, n - 1)``).

    Resolution order:
      1. ``scipy.stats.t.ppf`` when scipy is importable (exact for any n).
      2. A hardcoded table covering ``n = 2..100`` (df = 1..99).
      3. For ``n > 100`` a first-order Cornish-Fisher expansion
         ``1.96 + (1.96^3 + 1.96) / (4 * (n - 1))`` is used. Under
         Bonferroni correction (e.g. alpha=0.05/28 ~= 0.0018) the relevant
         critical value is the 99.9% quantile, where the normal approx's
         error balloons past 60% (``t.ppf(0.999, 100) ~= 3.17`` vs ``3.09``
         for the normal). The Cornish-Fisher expansion keeps the error
         below 0.1% at n=101 for the 97.5% level and is a much better
         approximation for the corrected alphas used downstream.
      4. For ``n < 2`` the CI is undefined; returns ``None`` (NOT ``0.0``),
         so a 1-seed estimate never becomes a zero-width CI that reads as
         "the estimate is exact". Callers MUST handle ``None`` (the
         one-sample t-test path in this file and ``run_ablation`` already
         guard with ``crit is not None`` before comparing).

    The scipy availability check runs once and is cached in ``_T_PP`` so
    repeated calls are essentially free.
    """
    global _T_PP
    if n < 2:
        # The CI is undefined for n<2 (zero degrees of freedom); ``None``
        # propagates as "undefined" and the t-test guard already handles it
        # (``crit is not None and ...``).
        return None
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
        except ImportError:
            _T_PP = False
    if _T_PP:
        return _T_PP(0.975, n - 1)
    # Hardcoded two-sided 95% critical values, n = 2..100 (df = 1..99),
    # covering the common multi-seed CI range without falling back to the
    # normal approximation (which has ~4% relative error at n=31..100).
    # Values are scipy.stats.t.ppf(0.975, df) rounded to 3 decimals.
    _TABLE = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
        22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045, 31: 2.042,
        32: 2.040, 33: 2.037, 34: 2.035, 35: 2.032, 36: 2.030,
        37: 2.028, 38: 2.026, 39: 2.024, 40: 2.023, 41: 2.021,
        42: 2.020, 43: 2.018, 44: 2.017, 45: 2.015, 46: 2.014,
        47: 2.013, 48: 2.012, 49: 2.011, 50: 2.010, 51: 2.009,
        52: 2.008, 53: 2.007, 54: 2.006, 55: 2.005, 56: 2.004,
        57: 2.003, 58: 2.002, 59: 2.002, 60: 2.001, 61: 2.000,
        62: 2.000, 63: 1.999, 64: 1.998, 65: 1.998, 66: 1.997,
        67: 1.997, 68: 1.996, 69: 1.995, 70: 1.995, 71: 1.994,
        72: 1.994, 73: 1.993, 74: 1.993, 75: 1.993, 76: 1.992,
        77: 1.992, 78: 1.991, 79: 1.991, 80: 1.990, 81: 1.990,
        82: 1.990, 83: 1.989, 84: 1.989, 85: 1.989, 86: 1.988,
        87: 1.988, 88: 1.988, 89: 1.987, 90: 1.987, 91: 1.987,
        92: 1.986, 93: 1.986, 94: 1.986, 95: 1.986, 96: 1.985,
        97: 1.985, 98: 1.985, 99: 1.984, 100: 1.984,
    }
    if n in _TABLE:
        return _TABLE[n]
    # n > 100: delegate to the exact Student-t inverse CDF used by
    # _bonferroni_crit_q (regularized incomplete beta + bisection). The
    # exact path guarantees <1e-9 accuracy without scipy and keeps the CI
    # consistent with the Bonferroni critical values (which already use the
    # exact path).
    try:
        return _bonferroni_crit_q(n, alpha=0.025)
    except Exception:
        # Fallback to Cornish-Fisher if _bonferroni path itself fails
        # (should never happen, but keep a safe net).
        z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
        return z + (z ** 3 + z) / (4.0 * (n - 1))


def _bonferroni_crit_q(n, alpha=0.05):
    """Bonferroni-corrected ONE-SIDED t critical value with ``n-1`` dof.

    Resolution order (mirrors ``_t_crit_975``):
      1. ``scipy.stats.t.ppf`` when scipy is importable (exact for any n).
      2. For ``n`` in 2..100 a hardcoded table of the 97.5% one-sided
         quantiles (same table as ``_t_crit_975``, reused via lookup).
         Under Bonferroni correction alpha is typically divided by 20-30,
         but the table is the EXACT scipy value at the 97.5% level — for
         the corrected alphas used downstream (e.g. 0.05/28 ~= 0.0018,
         one-sided) we need the ``1-alpha`` upper-tail quantile, which is
         far in the tail. We therefore do NOT use the 97.5% table for
         corrected alphas; we fall through to the dependency-free inverse-CDF
         computation below.
      3. Dependency-free Student-t inverse CDF using the regularized
         incomplete beta function plus bisection for the requested
         ``1 - alpha`` level. This matches the one-sided ``t_stat > crit``
         comparisons in Exp4/Exp5; using ``1-alpha/2`` would be a two-sided
         threshold and silently make the test too conservative.
      4. For ``n < 2`` returns ``None`` (CI undefined — caller must guard).

    Being at module scope means ``run_ablation.py`` can import and reuse
    this function instead of re-implementing the same logic, and it is
    unit-testable in isolation.
    """
    if n < 2:
        return None
    # Try scipy first (exact for any alpha / any n).
    global _T_PP
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
        except ImportError:
            _T_PP = False
    if _T_PP:
        return float(_T_PP(1 - alpha, n - 1))
    # scipy unavailable: compute the Student-t quantile directly via the
    # regularized incomplete beta function and bisection. The numerical
    # integration below is dependency-free and accurate for the small seed
    # counts used by Exp4/Exp5.
    target_p = 1.0 - alpha

    def _betacf(a, b, x):
        # Continued fraction for incomplete beta (Numerical Recipes).
        max_iter = 200
        eps = 3.0e-14
        fpmin = 1.0e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    def _betai(a, b, x):
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

    def _student_t_cdf(t_value, dof):
        if t_value == 0.0:
            return 0.5
        x = dof / (dof + t_value * t_value)
        ib = _betai(0.5 * dof, 0.5, x)
        if t_value > 0.0:
            return 1.0 - 0.5 * ib
        return 0.5 * ib

    dof = n - 1
    # For small dof and extreme tail quantiles (e.g. dof=4, alpha=0.0125,
    # target_p=0.9875, true crit ≈ 3.5), the continued-fraction _betacf can
    # lose precision when x is very small.  We bracket the quantile with a
    # coarse exponential scan, then refine with a *regularised* bisection
    # that uses the log-survival function log(1 - CDF) instead of the raw
    # CDF.  The log-survival function is numerically much better behaved in
    # the far upper tail: 1 - CDF(t) ≈ 0.5 * ib for large t, and ib itself
    # is computed from _betai which is stable for small x.  Taking logs
    # avoids catastrophic cancellation when CDF(t) is extremely close to 1.
    import math
    target_log_sf = math.log(max(alpha, 1e-300))  # log(1 - target_p) = log(alpha)
    # Coarse bracket: find lo, hi such that log_sf(lo) > target_log_sf > log_sf(hi)
    lo = 0.0
    hi = 1.0
    while True:
        cdf_hi = _student_t_cdf(hi, dof)
        if cdf_hi >= 1.0 - 1e-15:
            # Already saturated; push hi further
            hi *= 2.0
            if hi > 1.0e6:
                raise RuntimeError(
                    f"_bonferroni_crit_q fallback could not bracket quantile "
                    f"for n={n}, alpha={alpha}")
            continue
        log_sf_hi = math.log(max(1.0 - cdf_hi, 1e-300))
        if log_sf_hi <= target_log_sf:
            break
        hi *= 2.0
        if hi > 1.0e6:
            raise RuntimeError(
                f"_bonferroni_crit_q fallback could not bracket quantile "
                f"for n={n}, alpha={alpha}")
    # Refine with log-survival bisection (80 iterations, ~1e-24 relative
    # precision on the quantile).
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf_mid = _student_t_cdf(mid, dof)
        if cdf_mid >= 1.0 - 1e-15:
            # mid is already in the saturated region; move hi down
            hi = mid
            continue
        log_sf_mid = math.log(max(1.0 - cdf_mid, 1e-300))
        if log_sf_mid > target_log_sf:
            lo = mid
        else:
            hi = mid
    crit = 0.5 * (lo + hi)
    if not math.isfinite(crit):
        raise RuntimeError(
            f"_bonferroni_crit_q fallback produced non-finite critical value: {crit}")
    return float(crit)


# ---------------------------------------------------------------------------
# Shared small-model architecture spec — used by both run_quality.CSAAttn /
# HCAAttn (Experiment 4: standalone MQAR) and run_ablation._make_cfg
# (Experiment 5: hybrid layout ablation). Lifting the spec to a single
# module-level constant makes both experiments use the SAME CSA/HCA sub-layer
# widths, so cross-experiment comparisons (e.g. "the hybrid block's CSA
# contributes X to MQAR accuracy") are apples-to-apples. The values match the
# wider run_quality spec since that is the more informative regime for CSA's
# sparse retrieval.
# ---------------------------------------------------------------------------
SMALL_MODEL_SPEC = {
    # CSA sub-layer
    'csa_c':      32,   # compressed KV dim
    'csa_cI':     16,   # indexer key dim
    'csa_dc':     64,   # down-projected query dim
    'csa_m':      4,    # compression factor
    'csa_nh':     2,    # number of attention heads
    'csa_nIh':    2,    # number of indexer heads
    'csa_topk':   2,    # top-k blocks per query (must be < n_blocks so the
                        #   sparse selection is actually exercised; with
                        #   seq_len=16 and csa_m=4 (n_blocks=4) a topk of 4
                        #   would collapse CSA to dense attention)
    'csa_sliding_window': 4,
    # HCA sub-layer (m2 >> m so HCA produces fewer compressed blocks)
    'hca_c':      32,
    'hca_dc':     64,
    'hca_m2':     8,    # heavy compression (2x CSA's m=4)
    'hca_nh':     2,
    'hca_sliding_window': 4,
    # Partial RoPE (DeepSeek-V4 §2.3.3) for the CSA/HCA core attention —
    # paper-faithful default: rotate the last 64 dims of the queries /
    # compressed KV entries / sliding-window keys and inverse-rotate the
    # core-attention output. Clamped to the head dim (c=32 -> rope_dim
    # 32) for this small spec. Set to None/0 to disable.
    'csa_rope_dim': 64,
    'hca_rope_dim': 64,
    'rope_base':  10000.0,
}


def make_mqar_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                    embed: nn.Embedding, device=None, generator=None,
                    return_meta: bool = False):
    """Build an MQAR batch with *learnable* embeddings.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv): alternating (key, value) tokens
      positions [2*n_kv, seq_len-1): distractor tokens (sampled from the
        vocab ids that are neither a key nor a value, so the answer token
        never appears redundantly in the filler)
      position  seq_len-1: a cue key (repeat of one of the keys)
    Target = the value that was paired with the cued key.

    Fully vectorized: all ``batch`` sequences are constructed in parallel
    via a single argsort-based per-row permutation and advanced indexing,
    avoiding the per-sample Python loop and ``.item()`` calls that broke
    async kernel launches.

    ``generator`` (optional ``torch.Generator``) makes the batch generation
    reproducible and INDEPENDENT of the global RNG. This matters for
    multi-operator comparisons: each operator has a different parameter
    count, so it consumes a different number of global-RNG draws during
    init. Without a dedicated generator, the *same* seed produced
    *different* training batches across operators — a silent confound in
    the multi-seed CI. Pass a per-seed generator to guarantee that, for a
    given seed, all operators see the same training data.

    ``return_meta`` (keyword-only, default False) preserves the historical
    3-tuple return ``(x_emb, target, cue_pos)``. When True, a fourth element
    is returned: ``{'key_idx': j}`` where ``j`` is the ``[batch]`` tensor of
    which KV pair (0..n_kv-1) was cued for each sequence. Downstream
    mechanism diagnostics (e.g. ``run_mechanism_ablation``'s indexer
    gold-block recall) need to localize the cued pair WITHOUT re-deriving
    the task layout, so the index is surfaced here rather than duplicated
    in the caller.
    """
    if device is None:
        device = embed.weight.device

    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1`` — ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # silent shape corruption / trivially-solvable-task bugs these guards
    # are meant to prevent. Mirrors the convention established in ops_kda.py.
    if n_kv < 1:
        raise ValueError(f"n_kv={n_kv} must be >= 1")
    # Guard against silent shape corruption: if 2*n_kv exceeds vocab, the
    # argsort slice returns fewer than 2*n_kv ids and keys/vals end up with
    # mismatched shapes ([batch, vocab//2] vs the expected [batch, n_kv]).
    # If 2*n_kv >= seq_len, the cue at position seq_len-1 would overwrite a
    # KV pair (or fall inside the KV region), making the task trivially
    # solvable or unsolvable.
    if 2 * n_kv > vocab:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be <= vocab={vocab} (need 2*n_kv distinct ids)")
    if 2 * n_kv >= seq_len:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be < seq_len={seq_len} (need room for cue token)")

    cue_pos = seq_len - 1
    n_noise = seq_len - 2 * n_kv - 1       # filler slots between KV region and cue

    # One uniformly-random permutation of [0, vocab) per sequence, obtained
    # by argsort of iid uniform keys (ties have measure zero in float32).
    # Slicing the first 2*n_kv columns yields 2*n_kv distinct ids per row.
    pair_ids = torch.rand(batch, vocab, device=device, generator=generator).argsort(dim=-1)
    keys = pair_ids[:, 0:2 * n_kv:2]   # [batch, n_kv]
    vals = pair_ids[:, 1:2 * n_kv:2]   # [batch, n_kv]

    # Random noise base (covers noise positions; KV positions overwritten below).
    # Noise MUST be drawn only from the distractor ids — the ids in [0, vocab)
    # that are neither a key nor a value for that sequence. Drawing the filler
    # from the full vocabulary lets the cued *answer* token reappear in the
    # noise as a redundant copy that token-level operators (softmax/KDA) can
    # copy without resolving the key-value association, while the block-
    # compressed operators (CSA/HCA) only see it after compression/dilution —
    # a structural difficulty mismatch that would bias the reported ranking.
    distractor = pair_ids[:, 2 * n_kv:]          # [batch, vocab - 2*n_kv]
    if n_noise > 0:
        if 2 * n_kv >= vocab:
            raise ValueError(
                f"vocab={vocab} leaves no distractor ids for {n_noise} noise "
                f"slots; need vocab > 2*n_kv={2 * n_kv}")
        x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
        noise_draw = torch.randint(
            0, distractor.shape[1], (batch, n_noise),
            device=device, generator=generator)
        x[:, 2 * n_kv:cue_pos] = distractor.gather(1, noise_draw)
    else:
        x = torch.empty(batch, seq_len, dtype=torch.long, device=device)

    # Place (key, value) pairs at positions [0, 2*n_kv) via broadcast
    # advanced indexing: x[b_idx, even_pos] has shape [batch, n_kv].
    b_idx = torch.arange(batch, device=device).unsqueeze(-1)          # [batch, 1]
    even_pos = torch.arange(0, 2 * n_kv, 2, device=device)            # [n_kv]
    odd_pos = torch.arange(1, 2 * n_kv, 2, device=device)             # [n_kv]
    x[b_idx, even_pos] = keys
    x[b_idx, odd_pos] = vals

    # Vectorized cue selection: pick one key index per sequence in parallel.
    j = torch.randint(0, n_kv, (batch,), device=device, generator=generator)  # [batch]
    row = torch.arange(batch, device=device)                          # [batch]
    x[:, cue_pos] = keys[row, j]
    target = vals[row, j]

    # Apply embedding (gradients flow to the embedding table).
    x_emb = embed(x)
    if return_meta:
        # ``input_ids`` is included so downstream bad-case analysis can
        # decode an error in task terms (which key was cued, what the
        # other values were) without re-deriving the layout.
        return x_emb, target, cue_pos, {'key_idx': j, 'input_ids': x}
    return x_emb, target, cue_pos


# ---------------------------------------------------------------------------
# Additional synthetic quality probes (framework design §8).
#
# MQAR stays the canonical associative-recall probe; the three tasks below
# cover the other capability axes the hybrid-operator comparison needs:
# memory UPDATE under interference, LONG-RANGE retrieval at controlled
# distance, and MULTI-SITE aggregation that no single-position retrieval
# can solve. All generators share make_mqar_batch's signature and
# generator-isolation contract (same seed -> same batches for every
# operator), so they plug into train_one/_eval_model unchanged.
# ---------------------------------------------------------------------------
def make_kv_update_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                         embed: nn.Embedding, device=None, generator=None,
                         return_meta: bool = False):
    """Key-value UPDATE task: same key re-presented with a NEW value.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv):          (key, value_old) pairs
      positions [2*n_kv, 4*n_kv):     the SAME keys in a shuffled order,
                                       each paired with value_new
      positions [4*n_kv, seq_len-1):  distractor tokens (never a key or
                                       any old/new value)
      position  seq_len-1:            a cue key (repeat of one key)
    Target = the NEWEST value paired with the cued key. The model must
    overwrite the earlier association, i.e. memory UPDATE + interference
    handling, rather than plain recall.

    Constraints: ``3*n_kv <= vocab`` (keys + old/new values must be
    distinct) and ``4*n_kv < seq_len`` (room for the cue). ``return_meta``
    additionally returns ``{'key_idx': j}`` like ``make_mqar_batch``.
    """
    if device is None:
        device = embed.weight.device
    if n_kv < 1:
        raise ValueError(f"n_kv={n_kv} must be >= 1")
    if 3 * n_kv > vocab:
        raise ValueError(
            f"3*n_kv={3 * n_kv} must be <= vocab={vocab} (need n_kv keys "
            f"plus 2*n_kv distinct old/new values)")
    if 4 * n_kv >= seq_len:
        raise ValueError(
            f"4*n_kv={4 * n_kv} must be < seq_len={seq_len} (old pairs + "
            f"updated pairs + cue token)")

    cue_pos = seq_len - 1
    n_noise = seq_len - 4 * n_kv - 1

    perm = torch.rand(batch, vocab, device=device,
                      generator=generator).argsort(dim=-1)
    keys = perm[:, 0:n_kv]                    # [batch, n_kv]
    vals_old = perm[:, n_kv:2 * n_kv]         # [batch, n_kv]
    vals_new = perm[:, 2 * n_kv:3 * n_kv]     # [batch, n_kv]
    distractor = perm[:, 3 * n_kv:]           # excludes keys and ALL values

    # Shuffled second presentation: the cued key must be found by CONTENT,
    # not by position alignment with the first presentation.
    order2 = torch.rand(batch, n_kv, device=device,
                        generator=generator).argsort(dim=-1)
    keys2 = keys.gather(1, order2)
    vals2 = vals_new.gather(1, order2)

    x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    if n_noise > 0:
        noise_draw = torch.randint(
            0, distractor.shape[1], (batch, n_noise),
            device=device, generator=generator)
        x[:, 4 * n_kv:cue_pos] = distractor.gather(1, noise_draw)

    b_idx = torch.arange(batch, device=device).unsqueeze(-1)
    x[b_idx, torch.arange(0, 2 * n_kv, 2, device=device)] = keys
    x[b_idx, torch.arange(1, 2 * n_kv, 2, device=device)] = vals_old
    x[b_idx, torch.arange(2 * n_kv, 4 * n_kv, 2, device=device)] = keys2
    x[b_idx, torch.arange(2 * n_kv + 1, 4 * n_kv, 2, device=device)] = vals2

    j = torch.randint(0, n_kv, (batch,), device=device, generator=generator)
    row = torch.arange(batch, device=device)
    x[:, cue_pos] = keys[row, j]
    # Target = the NEWEST value paired with the cued key. ``vals_new`` is
    # indexed by the ORIGINAL key index (the same index space as ``keys``
    # and ``j``), so the answer is directly ``vals_new[row, j]``. Note that
    # ``order2`` maps second-presentation slots -> original key indices, so
    # ``order2[row, j]`` would instead select the new value of whichever
    # *other* key happens to occupy slot ``j`` (the inverse permutation
    # ``order2.argsort(-1)`` is what locates the cued key's slot).
    target = vals_new[row, j]

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {'key_idx': j, 'input_ids': x}
    return x_emb, target, cue_pos


def make_long_range_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                          embed: nn.Embedding, device=None, generator=None,
                          return_meta: bool = False):
    """Long-range retrieval: ONE (key, value) pair at a random distance.

    Layout per sequence: a single (key, value) pair at a uniformly random
    position ``p in [0, seq_len-3]``, distractors everywhere else, and the
    cue key at the final position. Target = the paired value.
    ``distance = (seq_len-1) - p`` spans ``[2, seq_len-2]`` per sequence,
    so accuracy-vs-distance curves (binned from ``return_meta``'s
    ``distance`` field) measure long-range LOCALISATION, not just recall.

    ``n_kv`` is accepted for signature compatibility and ignored (the
    distance is the controlled variable, not the pair count).
    """
    if device is None:
        device = embed.weight.device
    if vocab < 3:
        raise ValueError(
            f"vocab={vocab} must be >= 3 (one key, one value, and at least "
            f"one distractor id)")
    if seq_len < 4:
        raise ValueError(
            f"seq_len={seq_len} must be >= 4 (key + value + at least one "
            f"distractor + cue)")

    cue_pos = seq_len - 1
    perm = torch.rand(batch, vocab, device=device,
                      generator=generator).argsort(dim=-1)
    key = perm[:, 0]                # [batch]
    val = perm[:, 1]                # [batch]
    distractor = perm[:, 2:]

    # Distractor base everywhere, then overwrite the pair + cue.
    noise_draw = torch.randint(
        0, distractor.shape[1], (batch, seq_len - 1),
        device=device, generator=generator)
    x = torch.empty(batch, seq_len, dtype=torch.long, device=device)
    x[:, :cue_pos] = distractor.gather(1, noise_draw)

    p = torch.randint(0, seq_len - 2, (batch,),
                      device=device, generator=generator)
    row = torch.arange(batch, device=device)
    x[row, p] = key
    x[row, p + 1] = val
    x[:, cue_pos] = key
    target = val

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {
            'distance': (cue_pos - p), 'key_pos': p, 'input_ids': x}
    return x_emb, target, cue_pos


def make_multi_agg_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                         embed: nn.Embedding, device=None, generator=None,
                         return_meta: bool = False):
    """Multi-site aggregation: predict the MAJORITY category.

    The sequence is a bag of ``vocab``-category tokens (length
    ``seq_len``); the target is the majority category. The answer depends
    on MANY distant positions simultaneously — no single-position
    retrieval solves it, which is exactly the capability HCA's dense
    global aggregation (and to a lesser degree any recurrent accumulator)
    is hypothesised to provide. The readout position is the final token
    (itself a category token and part of the count).

    Ties for the majority are resolved deterministically: the
    smallest-index tied class wins one flipped token from the next tied
    class, so the target is always UNIQUE and the chance level is exactly
    ``1 / vocab``.

    ``n_kv`` is accepted for signature compatibility and ignored
    (``vocab`` doubles as the class count).
    """
    if device is None:
        device = embed.weight.device
    n_classes = vocab
    if n_classes < 2:
        raise ValueError(f"vocab(=n_classes)={vocab} must be >= 2")
    if seq_len < 3:
        raise ValueError(f"seq_len={seq_len} must be >= 3")

    cue_pos = seq_len - 1
    x = torch.randint(0, n_classes, (batch, seq_len),
                      device=device, generator=generator)
    counts = torch.zeros(batch, n_classes, device=device, dtype=torch.long)
    counts.scatter_add_(1, x, torch.ones_like(x))

    max_count = counts.max(dim=1, keepdim=True).values
    tied = counts == max_count
    needs_fix = tied.sum(dim=1) > 1
    fix_rows = needs_fix.nonzero(as_tuple=True)[0]
    if fix_rows.numel() > 0:
        # winner = smallest tied class index; loser = the next tied class.
        winner = tied[fix_rows].float().argmax(dim=1)
        non_winner_tied = tied[fix_rows].clone()
        non_winner_tied[torch.arange(fix_rows.numel(), device=device),
                        winner] = False
        loser = non_winner_tied.float().argmax(dim=1)
        # Flip ONE token of the loser class to the winner: the winner then
        # leads by 2 and is the strict majority.
        match = x[fix_rows] == loser.unsqueeze(1)
        first_pos = match.float().argmax(dim=1)
        x[fix_rows, first_pos] = winner
        counts = torch.zeros(batch, n_classes, device=device,
                             dtype=torch.long)
        counts.scatter_add_(1, x, torch.ones_like(x))
    target = counts.argmax(dim=1)

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {
            'counts': counts.tolist(),
            'n_distinct': (counts > 0).sum(dim=1).tolist(),
            'input_ids': x,
        }
    return x_emb, target, cue_pos


# Task registry consumed by train_one / _eval_model / main. ``n_kv`` marks
# whether the task is swept over the MQAR_NKV list (mqar, kv_update) or
# uses a fixed default (long_range: distance is the variable; multi_agg:
# vocab is the class count). ``default_n_kv`` is the fixed value used for
# the non-swept tasks.
SYNTHETIC_TASKS = {
    'mqar': {
        'make': make_mqar_batch, 'seq_len': 16, 'vocab': 16,
        'n_kv_swept': True,
        'description': 'associative recall of one of n_kv KV pairs '
                       '(canonical probe)',
    },
    'kv_update': {
        'make': make_kv_update_batch, 'seq_len': 16, 'vocab': 32,
        'n_kv_swept': True, 'default_n_kv': 2,
        'description': 'same key re-presented with a NEW value; retrieve '
                       'the LATEST value (memory update + interference)',
    },
    'long_range': {
        'make': make_long_range_batch, 'seq_len': 64, 'vocab': 16,
        'n_kv_swept': False, 'default_n_kv': 1,
        'description': 'single KV pair at a uniformly random distance '
                       '(accuracy-vs-distance curves from the meta field)',
    },
    'multi_agg': {
        'make': make_multi_agg_batch, 'seq_len': 33, 'vocab': 4,
        'n_kv_swept': False, 'default_n_kv': 1,
        'description': 'majority category over the whole sequence '
                       '(multi-site aggregation; no single-position '
                       'retrieval shortcut)',
    },
}


# ---------------------------------------------------------------------------
# Non-learned rule-based reference baselines (experiment-design
# requirement: the baseline ladder starts with the SIMPLEST baseline — a
# rule system — before any learned or complex model).
#
# Deterministic SYMBOLIC rules over the raw token ids, no parameters and
# nothing to tune:
#   * ``recency_key_lookup``: at the cue position, find the most recent
#     EARLIER occurrence of the cue token and predict the token that
#     follows it. This is the exact associative-lookup algorithm MQAR /
#     kv_update / long_range are built around.
#   * ``majority_count``: predict the most frequent token in the sequence
#     (multi_agg's exact target rule).
#
# ROLE: a task-ceiling / learnability reference, NOT a competitive tuned
# baseline. A rule that reads the raw sequence perfectly solves the
# synthetic task by construction (its accuracy is the ceiling a
# fixed-budget probe is measured against); the scientific claim "complex
# operator beats the baseline" still requires ``beats_tuned_baseline=True``
# against the TUNED LEARNED baseline (softmax), never the rule. The rule
# answers a different question: is the task symbolically solvable at all,
# and how much of it does the trained probe recover?
# ---------------------------------------------------------------------------
_RULE_NAMES = {
    'mqar': 'recency_key_lookup',
    'kv_update': 'recency_key_lookup',
    'long_range': 'recency_key_lookup',
    'multi_agg': 'majority_count',
}


def _rule_recency_key_lookup(input_ids, cue_pos):
    """Symbolic recency lookup over raw token ids.

    For each row: scan ``input_ids[b, :cue_pos]`` for the cue token, take
    its LAST occurrence, and predict the immediately following token.
    Falls back to token id 0 when the cue never occurs before the cue
    position (impossible for the generated tasks, kept as a total-
    function guard).
    """
    batch = input_ids.shape[0]
    preds = torch.zeros(batch, dtype=input_ids.dtype, device=input_ids.device)
    for b in range(batch):
        cue = input_ids[b, cue_pos]
        history = input_ids[b, :cue_pos]
        matches = (history == cue).nonzero(as_tuple=False)
        if matches.numel():
            i = int(matches[-1].item())
            preds[b] = input_ids[b, i + 1]
            # The token following the last cue occurrence must itself be
            # in-sequence (i + 1 <= cue_pos); for the generated layouts
            # the pair precedes the cue, so this always holds.
    return preds


def _rule_majority_count(input_ids):
    """Symbolic majority-count rule (multi_agg's exact target rule).

    Counts every token in the sequence (the readout position is part of
    the bag) and predicts the argmax class. The generator breaks ties so
    the target is always a unique strict majority; the rule therefore
    cannot be ambiguous on generated batches.
    """
    vocab = int(input_ids.max().item()) + 1
    counts = torch.zeros(input_ids.shape[0], vocab,
                         dtype=torch.long, device=input_ids.device)
    counts.scatter_add_(1, input_ids,
                        torch.ones_like(input_ids, dtype=torch.long))
    return counts.argmax(dim=1)


def evaluate_rule_baseline(task, n_kv, seq_len, vocab, device,
                           n_batches=4, batch=64):
    """Evaluate the task's non-learned rule on a dedicated RNG stream.

    Returns a JSON-safe record with the rule accuracy, the chance level,
    the stream provenance, and an explicit ROLE note. The stream seed
    offset (+8M) is disjoint from training / eval / monitoring / error
    analysis, so enabling this reference cannot change any reported
    metric. No parameters are trained and nothing is tuned: the rule is
    evaluated once on fresh batches.
    """
    make_fn = SYNTHETIC_TASKS[task]['make']
    rule = _RULE_NAMES.get(task)
    if rule is None:
        raise ValueError(f'no rule-based reference registered for task '
                         f'{task!r}')
    # Zero-initialized throwaway embedding (the rule never reads it). The
    # construction is RNG-forked so probing the reference cannot perturb
    # the global stream the sweep seeds itself from.
    with torch.random.fork_rng():
        embed = nn.Embedding(vocab, 1)
        with torch.no_grad():
            embed.weight.zero_()
    embed = embed.to(device)
    generator = make_seeded_generator(_RULE_BASELINE_SEED_OFFSET,
                                      device=device)
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            _x_emb, target, cue_pos, meta = make_fn(
                batch, seq_len, n_kv, vocab, embed, device,
                generator=generator, return_meta=True)
            input_ids = meta['input_ids']
            if rule == 'majority_count':
                preds = _rule_majority_count(input_ids)
            else:
                preds = _rule_recency_key_lookup(input_ids, cue_pos)
            correct += int((preds == target).sum().item())
            total += int(target.numel())
    accuracy = (correct / total) if total else None
    return {
        'task': task,
        'n_kv': n_kv,
        'rule': rule,
        'accuracy': accuracy,
        'n_sequences': total,
        'n_batches': n_batches,
        'batch': batch,
        'chance_acc': 1.0 / vocab,
        'seed_offset': _RULE_BASELINE_SEED_OFFSET,
        'role': 'non_learned_rule_reference',
        'tunable_hyperparameters': [],
        'note': ('Task-ceiling / learnability reference only: a symbolic '
                 'rule with no parameters, evaluated once on a dedicated '
                 'RNG stream. It is NOT the comparison baseline for '
                 'superiority claims (those require beats_tuned_baseline '
                 'against the tuned learned baseline); it establishes '
                 'that the synthetic task is solvable by an exact rule '
                 'and how much of it the fixed-budget probe recovers.'),
    }


# ---------------------------------------------------------------------------
# Explicit scientific hypotheses (experiment-design requirement).
#
# Each task/operator row in the results is annotated with the mechanism
# hypothesis it is meant to test, the predicted signature, and what would
# FALSIFY the prediction. This keeps the sweep hypothesis-driven: a row is
# not "op X on task Y" but "operator X's memory mechanism should show THIS
# signature, and HERE is the evidence that would count against it".
# Multiple-comparison hygiene: the sweep runs many (task, op) cells, so the
# per-row significance flags are Bonferroni-corrected and the hypotheses
# below are pre-registered (fixed before the numbers are read).
# ---------------------------------------------------------------------------
TASK_HYPOTHESES = OrderedDict([
    ('mqar', {
        'hypothesis': (
            'Associative recall requires an in-context memory that binds '
            'each key token to a later value token and retrieves it from '
            'the cue, independent of distance and of interference from '
            'other pairs.'),
        'prediction': (
            'Memoryless models stay at chance; mixers with content-based '
            'addressing (softmax, KDA delta rule, sparse CSA retrieval) '
            'rise above chance; accuracy degrades with n_kv (more '
            'interference).'),
        'falsified_if': (
            'The memoryless ``linear`` baseline beats chance (task is '
            'solvable without token mixing => confounded), or no memory '
            'operator exceeds the tuned baseline.'),
    }),
    ('kv_update', {
        'hypothesis': (
            'When the same key is re-presented with a new value, the '
            'memory must OVERWRITE the old association (delta-rule '
            'update), not merely accumulate it.'),
        'prediction': (
            'Operators with an explicit delta/erase rule (KDA) keep one '
            'value per key and answer with the latest; pure accumulators '
            'mix old and new values and lose accuracy as interference '
            'grows.'),
        'falsified_if': (
            'An operator with no overwrite mechanism matches or beats the '
            'delta-rule operator, or the error mix is dominated by '
            '``used_stale_value`` for the delta-rule operator.'),
    }),
    ('long_range', {
        'hypothesis': (
            'Long-context retrieval is a LOCALISATION problem: the model '
            'must keep early (key, value) pairs addressable as the '
            'distance grows, not only aggregate them.'),
        'prediction': (
            'O(1)-state recurrent operators (KDA / Mamba-2 / RWKV-7) show '
            'accuracy that degrades with the key-cue distance; CSA/HCA '
            'compression + sparse selection should flatten the far-distance '
            'curve because old blocks remain individually addressable.'),
        'falsified_if': (
            'Accuracy is flat or distance-independent for every operator '
            '(the probe does not exercise long-range memory), or the '
            'accuracy-vs-distance curve is non-monotone in a way the '
            'mechanism cannot explain.'),
    }),
    ('multi_agg', {
        'hypothesis': (
            'Majority-counting is a MULTI-SITE aggregation problem: no '
            'single-position retrieval solves it; the model must combine '
            'evidence from all tokens.'),
        'prediction': (
            'Dense global aggregation (HCA compressed MQA, any recurrent '
            'accumulator with slow decay) beats operators whose memory is '
            'read at one location; ``linear`` stays at chance.'),
        'falsified_if': (
            'A no-mixing or single-retrieval-only model matches the '
            'aggregation-capable operators (the task has a positional '
            'shortcut), or errors concentrate in ``echoed_last_token`` '
            '(recency copy rather than aggregation).'),
    }),
])

OPERATOR_HYPOTHESES = OrderedDict([
    ('linear', {
        'mechanism': 'no token mixing (position-wise MLP only)',
        'hypothesis': (
            'CONTROL: without inter-token information flow the target of a '
            'memory task cannot be computed.'),
        'predicted_signature': 'chance accuracy on all four probes',
    }),
    ('softmax', {
        'mechanism': 'dense causal attention (exact retrieval over all '
                     'past tokens)',
        'hypothesis': (
            'The simplest mixer is the right upper-bound baseline: full '
            'attention can represent any past-token lookup exactly, so it '
            'should win MQAR / kv_update / long_range once trained.'),
        'predicted_signature': (
            'highest or near-highest accuracy on recall probes; cost grows '
            'with T (the trade-off the efficient operators are trying to '
            'beat).'),
    }),
    ('kda', {
        'mechanism': 'recurrent delta rule with per-channel log-decay '
                     '(O(1) state)',
        'hypothesis': (
            'A delta-rule state implements key-value binding with '
            'overwrite, saturating its capacity as the number of pairs '
            'grows.'),
        'predicted_signature': (
            'strong on mqar/kv_update at small n_kv, degrading with n_kv; '
            'no distance-dependent collapse while state capacity suffices.'),
    }),
    ('mamba2', {
        'mechanism': 'selective state-space (SSD) recurrence, O(1) matrix '
                     'state',
        'hypothesis': (
            'A time-varying linear recurrence provides the same O(1)-state '
            'route with a different (input-selective, non-delta) update '
            'rule, so it should trade recall precision for stability.'),
        'predicted_signature': (
            'competitive with KDA at small n_kv; different failure mix '
            '(dilution vs overwrite) in the bad-case categories.'),
    }),
    ('rwkv7', {
        'mechanism': 'RWKV-7 delta-style recurrence (third memory route)',
        'hypothesis': (
            'An independent delta-style memory implementation should track '
            'KDA on recall probes; the trio KDA/Mamba-2/RWKV-7 separates '
            '"delta-style binding" from "the specific KDA implementation".'),
        'predicted_signature': (
            'accuracy within the KDA/Mamba-2 band at matched budget.'),
    }),
    ('csa', {
        'mechanism': 'compressed sparse attention: overlapped block KV '
                     'compression + learned top-k indexer',
        'hypothesis': (
            'Sparse per-block retrieval keeps old associations individually '
            'addressable at bounded cost, so far-distance recall should not '
            'collapse the way a fixed-size recurrent state does.'),
        'predicted_signature': (
            'flat-ish accuracy-vs-distance curve on long_range; errors '
            'concentrated in indexer selection misses (gold-block recall '
            'below 1.0) rather than in value dilution.'),
    }),
    ('hca', {
        'mechanism': 'heavy compression + dense MQA over few entries '
                     '(global aggregation)',
        'hypothesis': (
            'Global summarisation should help aggregate tasks (multi_agg) '
            'and can substitute for exact retrieval when compression is '
            'fine enough.'),
        'predicted_signature': (
            'best or near-best on multi_agg; sharper degradation than CSA '
            'on mqar/long_range because per-block detail is compressed '
            'further.'),
    }),
])


class MQARHead(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor, cue_pos: int):
        return self.fc(self.norm(x[:, cue_pos]))


class ResidualAttnLayer(nn.Module):
    """Generic residual attention layer: x = x + Attn(LN(x))."""

    def __init__(self, attn: nn.Module, d_model: int):
        super().__init__()
        self.attn = attn
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class SoftmaxAttn(nn.Module):
    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V
        self.scale = K ** -0.5
        # Lazily-built causal mask cache. T is not known at __init__ (it
        # depends on the input), so we build the mask on the first forward
        # and cache it. Subsequent forwards with the same T reuse the cached
        # mask instead of allocating a new [T, T] tensor every call. The
        # cache is keyed by (T, device) to handle seq-len or device changes.
        # Registered as a non-persistent buffer so .to(device) moves it and
        # state_dict skips it (it is derived, not learned).
        self.register_buffer('_causal_mask', None, persistent=False)
        self._mask_key = None  # (T, device) the cached mask was built for

    def _get_causal_mask(self, T, device):
        """Return the cached [T, T] strictly-upper-triangular bool mask,
        rebuilding it only when T or device changes."""
        key = (T, str(device))
        if self._mask_key != key or self._causal_mask is None:
            self._causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
            )
            self._mask_key = key
        return self._causal_mask

    def forward(self, x):
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.H, self.K)
        k = self.k(x).view(B, T, self.H, self.K)
        v = self.v(x).view(B, T, self.H, self.V)
        s = torch.einsum('bthk,bshk->bhts', q, k) * self.scale
        mask = self._get_causal_mask(T, x.device)
        s = s.masked_fill(mask, float('-inf'))
        p = torch.softmax(s, dim=-1)
        out = torch.einsum('bhts,bshv->bthv', p, v)
        return self.o(out.reshape(B, T, self.H * self.V))


class NoMixAttn(nn.Module):
    """Trivial memoryless baseline: an identity token mixer.

    Hypothesis (baseline ladder, experiment-design requirement): a
    multi-query associative-recall target is a function of the cued key
    AND of the earlier (key, value) pair, so a model with NO token mixing
    cannot recover it and must stay at chance. This operator makes that
    prediction testable with the SAME training recipe as every complex
    mixer: wrapped in ``ResidualAttnLayer`` it computes
    ``x + LayerNorm(x)`` per position (a position-wise MLP with no
    information flow across time), and the head reads only the cue
    position.

    Role in the comparison
    ----------------------
    * ``linear`` is the simplest rung of the baseline ladder. A complex
      operator that does not beat it is not doing anything useful.
    * It is NOT a strawman: it gets the same data streams, optimizer, and
      budget as every other operator, and the tuned-baseline protocol
      gives it its own learning-rate sweep, so "beats the simple
      baseline" means beating a properly tuned simple model.
    * On tasks where it DOES beat chance (e.g. if the target is predictable
      from the current token alone), the task is confounded and the
      operator ranking on it is not evidence about memory mechanisms.

    Zero parameters and zero inter-token communication by construction.
    """

    def forward(self, x):
        return x


class KDAAttn(nn.Module):
    # Named default for the per-channel log-decay scale, so all KDA modules
    # (KDAHybridLayer, KDAAttn, KDAAttnDecoding, _kda_heads) share the same
    # value via HybridConfig.kda_decay_scale (the fused model) or this
    # attribute (the standalone KDAAttn).
    # Mirrors run_decoding.KDAAttnDecoding.decay_scale.
    DECAY_SCALE = 0.1

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        # Match KDAHybridLayer's low-rank gate parameterization: d -> K -> H*K,
        # so standalone Exp4 KDA shares the hybrid KDA layer's operator
        # boundary and the Exp3 FLOPs formula.
        self.g_down = nn.Linear(d_model, K, bias=False)
        self.g_up = nn.Linear(K, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        # Causal depthwise short-conv (kernel=3) matching KDAHybridLayer
        # (ops_fused.py:307) and KDAAttnDecoding (run_decoding.py:283), so
        # cross-experiment KDA comparisons and the run_kv_cache
        # ``short_conv`` FLOPs term (line 340) see the same operator.
        # Exp 4 runs on full sequences (not streaming), so we left-pad with
        # zeros each call (causal, no future leakage) and do NOT carry a
        # persistent ``_conv_lookback`` buffer (unlike KDAHybridLayer /
        # KDAAttnDecoding which need it for autoregressive decode).
        self.short_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=0,
            groups=d_model, bias=True,
        )
        # Keep quality experiments on the reference backend by default so
        # historical accuracy numbers remain comparable. Set KDA_BACKEND=fla
        # (after installing the optional [fla] extra) only for an explicitly
        # accelerated run; KDA_BACKEND=auto uses FLA on CUDA when available.
        self.kda_backend = validate_kda_backend(
            os.environ.get('KDA_BACKEND', 'reference')
        )
        self.H, self.K, self.V = H, K, V

    def forward(self, x):
        B, T, d = x.shape
        # Apply the causal depthwise short-conv before the q/k/v/g/beta
        # projections, mirroring KDAHybridLayer.forward (ops_fused.py:501)
        # and KDAAttnDecoding.forward
        # (run_decoding.py:326). Left-pad by (ksize-1)=2 zeros so position
        # t sees {t-2, t-1, t} and NEVER t+1 (future leakage). Conv1d
        # padding=0; the F.pad does the causal left-pad explicitly.
        ksize = self.short_conv.kernel_size[0]
        x_conv = self.short_conv(
            F.pad(x.transpose(1, 2), (ksize - 1, 0))
        ).transpose(1, 2)
        # View BEFORE normalize: F.normalize(dim=-1) must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector —
        # otherwise each head's L2 norm becomes ~1/sqrt(H) instead of 1 and
        # the q.k dot products shrink by a factor of 1/H. Mirrors
        # ops_fused.py::KDAHybridLayer.
        q = F.normalize(F.silu(self.q(x_conv)).view(B, T, self.H, self.K), dim=-1)
        k = F.normalize(F.silu(self.k(x_conv)).view(B, T, self.H, self.K), dim=-1)
        v = F.silu(self.v(x_conv)).view(B, T, self.H, self.V)
        # Log-space gate: low-rank down/up with a softplus-style decay,
        # matching ops_fused.KDAHybridLayer and run_kv_cache.prefill_flops.
        # Uses the named DECAY_SCALE constant so all KDA instantiations agree.
        g = -F.softplus(self.g_up(self.g_down(x_conv))).view(
            B, T, self.H, self.K) * self.DECAY_SCALE
        beta = torch.sigmoid(self.beta(x_conv))
        out, _ = kda_forward(
            q, k, v, g, beta,
            output_final_state=False,
            use_chunk=False,
            backend=self.kda_backend,
        )
        return self.o(out.reshape(B, T, self.H * self.V))


class Mamba2Attn(nn.Module):
    """Standalone single-layer Mamba-2 operator for the quality probes.

    Wraps the reference :class:`ops_mamba.Mamba2Mixer` (via the backend
    factory, so ``MAMBA_BACKEND=official`` selects the mamba_ssm kernels
    where available). The geometry knobs are sized for the tiny quality
    models and are env-overridable: ``MAMBA2_D_STATE`` (default 16),
    ``MAMBA2_HEADDIM`` (default 16; clamped down to the largest divisor
    of ``expand * d_model``), ``MAMBA2_EXPAND`` (default 2),
    ``MAMBA2_D_CONV`` (default 4).

    Like :class:`KDAAttn`, the operator is STATELESS per call: the MQAR
    training loop feeds an independent random batch each step, so any
    conv context carried across calls would be cross-batch leakage.
    ``forward`` therefore always starts from a fresh conv state and a
    fresh SSM state.
    """

    def __init__(self, d_model):
        super().__init__()
        expand = parse_int_env('MAMBA2_EXPAND', 2, min_value=1, logger=logger)
        d_state = parse_int_env('MAMBA2_D_STATE', 16, min_value=1,
                                logger=logger)
        headdim = parse_int_env('MAMBA2_HEADDIM', 16, min_value=1,
                                logger=logger)
        d_conv = parse_int_env('MAMBA2_D_CONV', 4, min_value=2, logger=logger)
        d_inner = expand * d_model
        # Clamp headdim down to the largest divisor of d_inner (a
        # misconfigured headdim would otherwise abort the whole sweep with
        # a geometry error; the env knobs are convenience, not precision
        # instruments).
        headdim_eff = min(headdim, d_inner)
        while d_inner % headdim_eff != 0:
            headdim_eff //= 2
        cfg = HybridConfig(
            d_model=d_model, memory_type='mamba2',
            mamba_d_state=d_state, mamba_d_conv=d_conv,
            mamba_expand=expand, mamba_headdim=headdim_eff,
            mamba_backend=os.environ.get('MAMBA_BACKEND', 'reference'),
        )
        self.mixer = build_mamba2_mixer(cfg)

    def forward(self, x):
        out, _, _ = self.mixer.forward_functional(x, None, None)
        return out


class Rwkv7Attn(nn.Module):
    """Standalone single-layer RWKV-7 operator for the quality probes.

    Wraps the reference :class:`ops_rwkv.Rwkv7Mixer` (via the backend
    factory, so ``RWKV_BACKEND=fla`` selects the flash-linear-attention
    kernels where available). The geometry knobs are sized for the tiny
    quality models and are env-overridable: ``RWKV_HEAD_DIM`` (default
    16; clamped down to the largest divisor of ``d_model``),
    ``RWKV_DECAY_LORA`` / ``RWKV_A_LORA`` (default 4),
    ``RWKV_GATE_LORA`` (default 8) — the same env knobs run_ablation's
    matrix uses, so the standalone probes and the matrix stacks test
    the same mixer geometry at a given d_model.

    Like :class:`Mamba2Attn`, the operator is STATELESS per call: the
    MQAR training loop feeds an independent random batch each step, so
    any shift state carried across calls would be cross-batch leakage.
    ``forward`` therefore always starts from a fresh shift state and a
    fresh matrix state.
    """

    def __init__(self, d_model):
        super().__init__()
        head_dim = parse_int_env('RWKV_HEAD_DIM', 16, min_value=1,
                                 logger=logger)
        decay_lora = parse_int_env('RWKV_DECAY_LORA', 4, min_value=1,
                                   logger=logger)
        a_lora = parse_int_env('RWKV_A_LORA', 4, min_value=1, logger=logger)
        gate_lora = parse_int_env('RWKV_GATE_LORA', 8, min_value=1,
                                  logger=logger)
        head_dim_eff = min(head_dim, d_model)
        while d_model % head_dim_eff != 0:
            head_dim_eff //= 2
        cfg = HybridConfig(
            d_model=d_model, memory_type='rwkv7',
            rwkv_head_dim=head_dim_eff,
            rwkv_decay_lora=min(decay_lora, d_model),
            rwkv_a_lora=min(a_lora, d_model),
            rwkv_gate_lora=min(gate_lora, d_model),
            rwkv_backend=os.environ.get('RWKV_BACKEND', 'reference'),
        )
        # A standalone single-layer probe: this layer is RWKV layer 0 of
        # 1 (the depth-dependent init constants degenerate gracefully —
        # see Rwkv7Mixer._initialize_weights).
        self.mixer = build_rwkv7_mixer(cfg, rwkv_layer_idx=0,
                                        n_rwkv_layers=1)

    @property
    def effective_backend(self):
        return getattr(self.mixer, 'effective_backend', 'reference')

    def forward(self, x):
        out, _, _ = self.mixer.forward_functional(x, None, None)
        return out


class CSAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Use the shared small-model spec so Experiment 4 (standalone MQAR)
        # and Experiment 5 (ablation) test the SAME CSA sub-layer widths.
        spec = SMALL_MODEL_SPEC
        c, dc = spec['csa_c'], spec['csa_dc']
        m, nh, nIh, cI, topk = (
            spec['csa_m'], spec['csa_nh'], spec['csa_nIh'],
            spec['csa_cI'], spec['csa_topk'],
        )
        self.m, self.topk, self.nh, self.nIh, self.c, self.cI, self.dc = \
            m, topk, nh, nIh, c, cI, dc
        self.W_aKV = nn.Linear(d_model, c, bias=False)
        self.W_bKV = nn.Linear(d_model, c, bias=False)
        self.W_aZ = nn.Linear(d_model, c, bias=False)
        self.W_bZ = nn.Linear(d_model, c, bias=False)
        self.Ba = nn.Parameter(torch.randn(m, c) * 0.02)
        self.Bb = nn.Parameter(torch.randn(m, c) * 0.02)
        self.W_DQ = nn.Linear(d_model, dc, bias=False)
        self.W_UQ = nn.Linear(dc, c * nh, bias=False)
        self.W_IUQ = nn.Linear(dc, cI * nIh, bias=False)
        self.W_w = nn.Linear(d_model, nIh, bias=False)
        self.W_KV_idx = nn.Linear(d_model, cI, bias=False)
        self.W_Z_idx = nn.Linear(d_model, cI, bias=False)
        self.B_idx = nn.Parameter(torch.randn(m, cI) * 0.02)
        self.sink = nn.Parameter(torch.zeros(nh))
        self.o = nn.Linear(c * nh, d_model, bias=False)

    def forward(self, x):
        # ``naive_csa`` handles non-divisible T via internal right-padding
        # and trims its output back to the original T, so no external
        # pad/trim is needed (see ``test_csa_hca_non_divisible_T``). Keeping
        # this wrapper thin avoids doing the pad/trim twice.
        o = naive_csa(
            x, self.W_aKV.weight, self.W_bKV.weight,
            self.W_aZ.weight, self.W_bZ.weight, self.Ba, self.Bb,
            self.W_DQ.weight, self.W_UQ.weight, self.W_IUQ.weight,
            self.W_w.weight, self.W_KV_idx.weight, self.W_Z_idx.weight,
            self.B_idx,
            m=self.m, topk=self.topk, nh=self.nh, nIh=self.nIh,
            c=self.c, c_I=self.cI, dc=self.dc,
            sliding_window=SMALL_MODEL_SPEC['csa_sliding_window'],
            sink_logits=self.sink,
            # Use cosine-style indexer scoring for the quality experiment so
            # top-k selection is not confounded by q_idx / K_idx vector norms.
            normalize_qk=True,
            # Partial RoPE (DeepSeek-V4 §2.3.3), paper-faithful default.
            rope_dim=SMALL_MODEL_SPEC.get('csa_rope_dim', 64),
            rope_base=SMALL_MODEL_SPEC.get('rope_base', 10000.0),
        )
        return self.o(o)


class HCAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Use the shared small-model spec (mirrors CSAAttn). HCA's defining
        # feature is *heavy* compression: m2 >> m so the HCA branch produces
        # far fewer compressed blocks than CSA, trading recall granularity
        # for global context. With seq_len=16 and CSA m=4 (n_blocks_CSA=4),
        # setting m2=8 gives n_blocks_HCA=2, exercising the heavier-
        # compression regime.
        spec = SMALL_MODEL_SPEC
        c, dc = spec['hca_c'], spec['hca_dc']
        m2, nh = spec['hca_m2'], spec['hca_nh']
        self.m2, self.nh, self.c, self.dc = m2, nh, c, dc
        self.W_KV = nn.Linear(d_model, c, bias=False)
        self.W_Z = nn.Linear(d_model, c, bias=False)
        self.B_pos = nn.Parameter(torch.randn(m2, c) * 0.02)
        self.W_DQ = nn.Linear(d_model, dc, bias=False)
        self.W_UQ = nn.Linear(dc, c * nh, bias=False)
        self.sink = nn.Parameter(torch.zeros(nh))
        self.o = nn.Linear(c * nh, d_model, bias=False)

    def forward(self, x):
        # ``naive_hca`` handles non-divisible T via internal right-padding
        # and trims its output back to the original T, so no pad/trim is
        # needed here. See the comment in ``CSAAttn.forward`` for the
        # full rationale.
        o = naive_hca(x, self.W_KV.weight, self.W_Z.weight, self.B_pos,
                      self.W_DQ.weight, self.W_UQ.weight,
                      m2=self.m2, nh=self.nh, c=self.c, dc=self.dc,
                      sliding_window=SMALL_MODEL_SPEC['hca_sliding_window'],
                      sink_logits=self.sink,
                      # Partial RoPE (DeepSeek-V4 §2.3.3), paper-faithful
                      # default.
                      rope_dim=SMALL_MODEL_SPEC.get('hca_rope_dim', 64),
                      rope_base=SMALL_MODEL_SPEC.get('rope_base', 10000.0))
        return self.o(o)


def _eval_model(layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=4, batch=64, seed=42, task='mqar',
                gen_offset=5_000_000, collect_predictions=False):
    """Evaluate accuracy over multiple fresh batches for a stable estimate.

    ``task`` selects the synthetic-task generator from ``SYNTHETIC_TASKS``
    (default ``'mqar'`` keeps the historical behaviour).

    Uses a dedicated ``torch.Generator`` for batch generation so the eval
    batches are reproducible and independent of any global-RNG state left
    over by training. This also makes the eval pass deterministic across
    re-runs, which is useful for debugging.

    The eval generator is derived from the training ``seed`` (not a fixed
    constant), so each seed is evaluated on FRESH eval batches. This makes
    the multi-seed ``std_acc`` / CI / t-test in ``train_multi_seed`` capture
    BOTH training stochasticity and eval-sampling variance — i.e. the full
    variance of repeating the experiment from scratch. A fixed eval seed
    would measure every seed on the SAME eval batches, excluding
    eval-sampling variance from the CI and overstating the significance of
    the operator comparisons. Because the
    generator depends only on ``seed``, all operators for a given seed still
    see the SAME eval batches (apples-to-apples), and the far ``+ 5_000_000``
    offset keeps the eval stream clear of every seed's ``+ 1_000_000``
    training stream (mirrors ``run_ablation.py::_eval_model``).

    ``gen_offset`` selects which seed offset defines this eval stream. The
    default (5_000_000) is the headline eval the reported accuracy comes
    from. The held-out validation-monitoring curve uses
    ``_VAL_MONITOR_SEED_OFFSET`` (6_000_000) instead, so monitored examples
    are never the examples the headline metric was computed on, and (with
    the error-analysis stream at 7_000_000) monitoring cannot leak into the
    reported result.

    ``collect_predictions`` (default False, historical 2-tuple return)
    additionally returns the concatenated per-sample ``pred`` / ``target``
    and per-class ``scores`` tensors (on CPU) so the caller can compute
    confusion-matrix metrics (macro/micro/weighted F1, per-class
    precision/recall, PR-AUC / ROC-AUC one-vs-rest) without a second
    forward pass. The accumulated tensors are detached, so collecting them
    can never change gradients or global RNG state.

    Restores each module's train/eval mode after evaluation so a future
    caller that evaluates mid-training does not silently resume training
    in eval mode (no dropout, BN using running stats).
    """
    # Save the prior train/eval state so we can restore it after eval —
    # a latent footgun if a caller ever invokes _eval_model mid-training.
    was_training = {m: m.training for m in (layer, head, embed)}
    try:
        layer.eval()
        head.eval()
        embed.eval()  # nn.Embedding has no dropout/batchnorm so this is a no-op,
                      # but we set it for symmetry with layer/head so future
                      # additions (e.g. embedding dropout) do not silently stay
                      # in train mode during evaluation.
        # Fresh eval batches per seed (see the docstring for the statistical
        # rationale): seed+5_000_000 is far from every seed's training stream
        # at seed+1_000_000 and from run_ablation's own eval offset usage.
        eval_gen = make_seeded_generator(seed + gen_offset, device=device)
        make_batch = SYNTHETIC_TASKS[task]['make']
        correct, total = 0, 0
        losses = []
        pred_chunks, target_chunks, score_chunks = [], [], []
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_batch(
                    batch, seq_len, n_kv, vocab, embed, device,
                    generator=eval_gen)
                h = layer(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
                if collect_predictions:
                    pred_chunks.append(logits.argmax(-1).detach().to('cpu'))
                    target_chunks.append(target.detach().to('cpu'))
                    score_chunks.append(logits.detach().float().to('cpu'))
        # Guard against n_batches=0 (or batch=0): without this the function
        # raises ZeroDivisionError on ``correct / total`` and
        # ``sum([]) / len([])``. Returns 0.0 for both metrics so the caller
        # gets a finite (if meaningless) value rather than a crash. Mirrors
        # the steps=0 guard in train_one.
        if total == 0 or not losses:
            if collect_predictions:
                return 0.0, 0.0, {
                    'pred': torch.zeros(0, dtype=torch.long),
                    'target': torch.zeros(0, dtype=torch.long),
                    'scores': torch.zeros(0, 0),
                }
            return 0.0, 0.0
        acc = correct / total
        mean_loss = sum(losses) / len(losses)
        if collect_predictions:
            return acc, mean_loss, {
                'pred': torch.cat(pred_chunks),
                'target': torch.cat(target_chunks),
                'scores': torch.cat(score_chunks),
            }
        return acc, mean_loss
    finally:
        for m, was in was_training.items():
            m.train(was)


# Operators with NO inter-token communication. They are valid scientific
# CONTROLS (the chance-accuracy prediction is the point), but the
# fixed-batch memorization gate does not apply to them: with no cross-token
# pathway the target is not a deterministic function of the cue, so even a
# FIXED batch is not memorizable by construction. The gate skips them
# (recorded as ``skipped``, not passed and not failed) instead of flagging a
# healthy control as a broken training recipe.
_MEMORYLESS_OPS = frozenset({'linear'})


def _op_factory(op_name, d_model):
    """Return the attention-operator factory for a quality-probe op.

    Lifted out of ``train_one`` so ``run_overfit_sanity_check`` constructs
    the operator through the IDENTICAL code path (same classes, same
    ``d_model``). The factory dictionary itself consumes no RNG draws, so
    routing both callers through this function does not change the seeded
    initialization stream.
    """
    factories = {
        # Simplest rung of the baseline ladder: no token mixing at all.
        # Any operator that cannot beat this on a memory task is not
        # demonstrating a memory mechanism.
        'linear':  lambda: NoMixAttn(),
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'mamba2':  lambda: Mamba2Attn(d_model),
        'rwkv7':   lambda: Rwkv7Attn(d_model),
        'csa':     lambda: CSAAttn(d_model),
        'hca':     lambda: HCAAttn(d_model),
    }
    if op_name not in factories:
        raise ValueError(
            f"op_name={op_name!r} must be one of {sorted(factories)}.")
    return factories[op_name]


def _build_task_model(op_name, d_model, vocab, device):
    """Build ``(embed, head, layer)`` in the seeded construction order.

    Order matters for reproducibility: embed and head are created BEFORE
    the operator-specific layer so their initial weights are IDENTICAL
    across operators for a given seed. Different operators consume a
    different number of RNG draws during init, so creating the layer first
    would desync the downstream head init — a silent confound in the
    multi-seed CI. Callers must invoke this after ``set_global_seed(seed)``
    and must not reorder the three constructions.

    Shared by ``train_one`` and ``run_overfit_sanity_check`` so the gate
    validates the exact model that the full experiment trains.
    """
    embed = nn.Embedding(vocab, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    layer = ResidualAttnLayer(_op_factory(op_name, d_model)(), d_model).to(device)
    return embed, head, layer


def train_one(op_name, d_model=32, seq_len=16, n_kv=1, vocab=16,
              steps=100, lr=3e-3, seed=42, device='cpu',
              softmax_steps=None, eval_batches=4, eval_batch=64,
              train_batch=None, task='mqar',
              val_every=0, val_batches=2, val_batch=32,
              error_batches=0, error_batch=64, max_error_examples=2):
    """Train a single operator on a synthetic task for ``steps`` steps.

    ``task`` selects the batch generator from ``SYNTHETIC_TASKS``
    (default ``'mqar'``: the historical Multi-Query Associative Recall
    probe; see the registry above for the update / long-range /
    multi-aggregation probes).

    ``softmax_steps`` lets the softmax baseline train longer to reach
    convergence (the original 100 steps left softmax at ~10%, barely above the
    6.25% chance — a meaningless upper bound).

    ``train_batch`` overrides the per-step training batch size (default 32,
    overridable for memory-constrained or GPU runs via the
    ``MQAR_TRAIN_BATCH`` env var).

    Results-analysis hooks (all OFF by default through the zero/None
    defaults, so direct callers get the historical behaviour bit-for-bit):

      * ``val_every > 0`` records a held-out validation loss/accuracy curve
        every ``val_every`` steps (plus at step 0), drawn from the dedicated
        ``_VAL_MONITOR_SEED_OFFSET`` stream — a DIFFERENT stream from the
        headline eval, so monitoring cannot leak into the reported metric.
        The train-vs-val curves feed ``diagnose_training_curves``.
      * ``error_batches > 0`` runs a bad-case analysis on
        ``error_batches`` x ``error_batch`` held-out samples from the
        dedicated ``_ERROR_ANALYSIS_SEED_OFFSET`` stream: misclassified
        samples are printed / stored with their task-level structure
        (which key was cued, which value was predicted) so the failure
        mode can be classified rather than guessed.

    RNG isolation: the model is initialized with ``torch.manual_seed(seed)``,
    but a SEPARATE per-step generator (also seeded with ``seed``) drives
    ``make_mqar_batch``. This is critical for the multi-seed comparison: the
    different operators have different parameter counts, so they consume a
    different number of RNG draws during init. Without a separate generator
    for batch generation, the *same* seed produced *different* training
    batches across operators, undermining the apples-to-apples comparison
    that multi-seed CI is meant to strengthen.

    ``device`` may be passed as a string ('cpu'/'cuda') or a
    ``torch.device``; string inputs are coerced to ``torch.device`` so
    callers from notebooks don't hit ``AttributeError`` on ``device.type``.
    """
    # Coerce string device (e.g. 'cpu', 'cuda') to torch.device so callers
    # passing a string from a notebook don't hit AttributeError on
    # ``device.type`` inside this function. main() already passes a real
    # torch.device, but train_one is a public function and the coercion is
    # cheap and idempotent.
    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    if softmax_steps is None:
        softmax_steps = steps
    if train_batch is None:
        # Robust env var parsing: ``parse_int_env`` logs a warning and falls
        # back to the default on a malformed ``MQAR_TRAIN_BATCH=abc`` (or
        # ``=0``, which would crash on the first batch with
        # ZeroDivisionError), matching the robustness pattern already used
        # for BENCH_REPEATS in run_benchmark.py.
        train_batch = parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1,
                                    logger=logger)
    actual_steps = softmax_steps if op_name == 'softmax' else steps

    # Seed Python random / NumPy / torch / CUDA from one value with
    # deterministic cuDNN+algorithm selection. torch.manual_seed alone left
    # Python/NumPy streams unseeded and cuDNN free to pick non-reproducible
    # kernels on GPU.
    set_global_seed(seed, deterministic=True)
    # Shared construction helper (validates op_name, builds embed/head/layer
    # in the fixed seeded order — see _build_task_model's docstring).
    embed, head, layer = _build_task_model(op_name, d_model, vocab, device)
    # Record the operator's ACTUAL parameter count: same d_model does NOT
    # mean same capacity across operators (Mamba-2's internal expansion /
    # state dimensions differ from KDA's head geometry), and the quality
    # comparison must annotate the residual difference rather than assume
    # parity.
    n_params = sum(p.numel() for p in layer.parameters())
    # Build parameter groups with proper weight-decay exclusion: embeddings,
    # biases, and LayerNorm parameters are NOT weight-decayed (standard ML
    # practice — decaying these can hurt training quality).
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    # Separate generator for batch generation so the per-step batches are
    # IDENTICAL across operators for a given seed (the model init consumed a
    # different number of RNG draws per operator, which would otherwise
    # desync the global RNG and produce different training data per op).
    # A LARGE offset (1_000_000) instead of ``+ 1`` keeps the batch RNG
    # stream in a completely different region of the seed space from the
    # model initialization stream, eliminating any overlap that would weaken
    # the t-test's independence assumption.
    # Routed through make_seeded_generator for CPU-fallback on older torch
    # builds.
    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    # Held-out validation monitoring (off when val_every <= 0). The monitor
    # stream is seed + 6_000_000 — distinct from training (+1_000_000) and
    # from the headline eval (+5_000_000) — so the curve is a genuine
    # held-out signal and enabling it cannot change any reported metric
    # (read-only eval, dedicated RNG). See _VAL_MONITOR_SEED_OFFSET.
    #
    # ValidationCurve (rather than a bare list) is used so the record keeps
    # the best-validation point and the early-stopping regret alongside the
    # final number: a run whose final weights sit past the turn is visible
    # as a positive regret instead of only as "final_acc = 0.62".
    val_every = int(val_every or 0)
    val_curve = None
    if val_every > 0:
        def _monitor_validation(step):
            acc_v, loss_v = _eval_model(
                layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=val_batches, batch=val_batch, seed=seed, task=task,
                gen_offset=_VAL_MONITOR_SEED_OFFSET)
            return ValPoint(loss=loss_v, acc=acc_v)

        val_curve = ValidationCurve(
            _monitor_validation, steps=actual_steps, val_every=val_every,
            label=f'{op_name}/{task}/n_kv={n_kv}/seed={seed}', ceiling=1.0)
        val_curve.record(0)

    losses, accs = [], []
    # Pre-clip gradient norms. The clip call already computes this value,
    # so recording it is free and turns "the loss stayed finite" into
    # "the optimizer stayed healthy" (a long tail of huge norms is the
    # first sign of an LR that is too large).
    grad_norms = []
    # Set train mode ONCE before the loop, not per step (calling
    # layer.train()/head.train() inside the loop is a redundant O(steps)
    # no-op once the modules are already in train mode).
    layer.train()
    head.train()
    embed.train()
    for step in range(actual_steps):
        x_emb, target, cue_pos = make_batch(
            train_batch, seq_len, n_kv, vocab, embed, device, generator=batch_gen)
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        # NaN/Inf guard: if a single step diverges (e.g. AdamW + lr=3e-3
        # instability, or a single Inf in KDA state from a bad beta/g
        # combination), backward() would propagate NaN into ALL parameter
        # grads, opt.step() would write NaN into ALL parameters, and the
        # rest of training would silently produce NaN logits — the loss
        # curve would look like [..., 2.4, 2.5, NaN, NaN, NaN, ...] with
        # no error raised. Worse, the final eval would still return a
        # *finite* accuracy (argmax on NaN is undefined but deterministic,
        # often returning 0), so the divergent seed would NOT be caught by
        # the per-seed try/except in train_multi_seed and would silently
        # corrupt the aggregate mean/CI. Raise here so the per-seed
        # try/except catches it as a failed seed.
        if not torch.isfinite(loss):
            # Systematic triage before raising: the report names the first
            # failing stage (data -> loss -> params -> logits) so the
            # per-seed failure message is actionable, not just "NaN".
            _diag = diagnose_nonfinite(
                loss=loss, inputs={'x_emb': x_emb}, params=params,
                logits=logits,
                context=f'{op_name}/task={task}/seed={seed}/step={step}')
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(op={op_name}, seed={seed}); diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad()
        loss.backward()
        # Guard against NaN/Inf gradients BEFORE step. ``clip_grad_norm_``
        # computes ``total_norm`` and (if NaN) the comparison
        # ``total_norm > max_norm`` is False, so no clipping happens and the
        # NaN grads pass through to ``opt.step()`` unchecked. That corrupts
        # ALL parameters in one step; the NEXT iteration's forward then
        # produces a NaN loss, which the finite-loss guard above catches —
        # but by then every parameter is already NaN and the seed is lost
        # without a clear root cause.
        #
        # The returned norm is already computed by the clip call, so
        # checking it costs no extra kernel and runs EVERY step (the old
        # every-50-steps per-parameter scan could let a NaN step through
        # between checks). Only when it is non-finite do we pay for the
        # per-parameter scan that names the offending tensors.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grad_norms.append(float(grad_norm))
        if not torch.isfinite(grad_norm):
            bad_grads = {f'p{i}': p.grad for i, p in enumerate(params)
                         if p.grad is not None
                         and not torch.isfinite(p.grad).all()}
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                grads=bad_grads or None, grad_norm=float('nan'),
                context=f'{op_name}/task={task}/seed={seed}/step={step}')
            raise RuntimeError(
                f"non-finite gradient norm at step {step} "
                f"({len(bad_grads)} non-finite param grads; op={op_name}, "
                f"seed={seed}); diagnosis [{_diag['verdict']}]: "
                f"{_diag['summary']}")
        opt.step()
        losses.append(loss.item())
        accs.append(acc)
        if val_curve is not None and (step + 1) % val_every == 0:
            val_curve.record(step + 1)

    # Ensure the monitored curve ends at the final trained state (e.g.
    # steps=25 with val_every=10 would otherwise stop at step 20).
    if val_curve is not None:
        val_curve.close(actual_steps)

    # Final eval on multiple fresh batches for a stable accuracy estimate.
    # Timed so the results carry an in-experiment inference-cost proxy,
    # used by the efficiency-vs-accuracy (Pareto) annotation in main().
    # Predictions are collected in the SAME pass (collect_predictions=True)
    # to compute imbalance-aware metrics (macro/micro/weighted F1,
    # balanced accuracy, per-class support, one-vs-rest PR-AUC/ROC-AUC)
    # without a second forward pass.
    eval_t0 = time.time()
    final_acc, final_loss, final_eval = _eval_model(
        layer, head, embed, seq_len, n_kv, vocab, device,
        n_batches=eval_batches, batch=eval_batch, seed=seed, task=task,
        collect_predictions=True,
    )
    final_eval_time_s = time.time() - eval_t0
    chance = 1.0 / vocab

    # Confusion-derived metrics. The MULTI-CLASS synthetic probes are
    # class-balanced BY CONSTRUCTION (targets uniform over the task's
    # vocab), but the metrics are reported anyway so (a) the class-balance
    # assumption is VERIFIED in the record (``class_balance``) instead of
    # assumed, (b) a collapsed predictor cannot hide behind accuracy, and
    # (c) the averaging convention of every F1 is explicit. If a future
    # task variant becomes imbalanced, the same record already carries the
    # imbalance-robust metrics.
    final_metrics = None
    if final_eval['pred'].numel() > 0:
        final_metrics = multiclass_classification_metrics(
            final_eval['pred'], final_eval['target'], num_classes=vocab,
            scores=final_eval['scores'])

    # Post-training results analysis, all from streams the training never
    # saw: the held-out validation curve (monitor stream), its
    # over/underfitting verdict, and the bad-case analysis (error stream).
    if val_curve is not None:
        curve_block = val_curve.finalize(losses, accs)
        curve_block['val_monitor_seed_offset'] = _VAL_MONITOR_SEED_OFFSET
        val_curve.log_verdict(curve_block)
    else:
        curve_block = None
    val_history = (curve_block['val_history'] if curve_block else [])
    val_losses = [v['loss'] for v in val_history]
    curve_diagnosis = (curve_block['curve_diagnosis'] if curve_block
                       else diagnose_training_curves(losses, None))
    error_analysis = None
    if int(error_batches or 0) > 0:
        error_analysis = analyze_errors(
            layer, head, embed, op_name=op_name, task=task, seq_len=seq_len,
            n_kv=n_kv, vocab=vocab, device=device,
            n_batches=int(error_batches), batch=int(error_batch), seed=seed,
            max_examples=int(max_error_examples))
        logger.info(
            f"  [bad-case] {op_name}/{task} n_kv={n_kv} seed={seed}: "
            f"{error_analysis['n_wrong']}/{error_analysis['n_samples']} "
            f"wrong; categories: "
            f"{ {c: v['count'] for c, v in error_analysis['categories'].items() if v['count']} }")

    gradient_norms = None
    if grad_norms:
        gradient_norms = {
            'n_steps': len(grad_norms),
            'max': max(grad_norms),
            'mean': sum(grad_norms) / len(grad_norms),
            'last': grad_norms[-1],
            'clip_norm': 1.0,
            # Pre-clip norm much larger than the clip budget means the raw
            # gradients were exploding and clipping is masking it.
            'n_above_clip': sum(1 for g in grad_norms if g > 1.0),
        }
    curve_fields = {}
    if curve_block is not None:
        # Additive curve-reading fields. ``final_acc`` above still means
        # "accuracy at the fixed step budget"; these say whether that
        # endpoint was the right place to read the curve, and whether an
        # apparently flat validation loss is actually accuracy saturation.
        for key in ('best_val', 'val_loss_regret_from_best',
                    'val_acc_regret_from_best', 'steps_since_best',
                    'best_val_is_final', 'early_stopping_summary',
                    'accuracy_regime', 'val_acc_first', 'val_acc_last',
                    'val_acc_peak'):
            if key in curve_block:
                curve_fields[key] = curve_block[key]
    analysis_fields = {
        'val_history': val_history,
        'curve_diagnosis': curve_diagnosis,
        'error_analysis': error_analysis,
        'gradient_norms': gradient_norms,
        'final_eval_time_s': final_eval_time_s,
        'final_eval_ms_per_sample': (
            1000.0 * final_eval_time_s / max(1, int(eval_batches) * int(eval_batch))),
        'generalization_gap': (
            curve_block['generalization_gap'] if curve_block is not None
            else None),
        **curve_fields,
        # Full confusion-derived metric block (each key names its averaging
        # convention — see results_analysis.multiclass_classification_metrics)
        # plus flat convenience keys so a table/DataFrame does not need to
        # reach into the nested dict.
        'final_metrics': final_metrics,
        'final_f1_macro': (final_metrics or {}).get('f1_macro'),
        'final_f1_weighted': (final_metrics or {}).get('f1_weighted'),
        'final_balanced_accuracy': (final_metrics or {}).get('balanced_accuracy'),
        'final_pr_auc_macro_ovr': (final_metrics or {}).get('pr_auc_macro_ovr'),
        'final_roc_auc_macro_ovr': (final_metrics or {}).get('roc_auc_macro_ovr'),
    }
    # Guard against steps=0 (would crash on accs[-1] / sum([])/0 below).
    # Returns a stub with the eval result and zeros for the training
    # trajectory fields. ``actual_steps`` is reported as 0 so downstream
    # consumers can detect the degenerate case.
    if not losses:
        return {
            'op': op_name,
            'task': task,
            'n_kv': n_kv,
            'n_params': n_params,
            'final_acc': final_acc,
            'final_loss': final_loss,
            'chance_acc': chance,
            'last_train_acc': 0.0,
            'mean_last10_loss': 0.0,
            'mean_last10_acc': 0.0,
            'steps': actual_steps,
            'seed': seed,
            'train_batch': train_batch,
            'lr': lr,
            'loss_curve': [],
            'acc_curve': [],
            **analysis_fields,
        }
    return {
        'op': op_name,
        'task': task,
        'n_kv': n_kv,
        'n_params': n_params,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'chance_acc': chance,
        'last_train_acc': accs[-1],
        'mean_last10_loss': sum(losses[-10:]) / min(10, len(losses)),
        'mean_last10_acc': sum(accs[-10:]) / min(10, len(accs)),
        'steps': actual_steps,
        'seed': seed,
        'train_batch': train_batch,
        # Full per-step training curves (the experiment log, not just the
        # endpoints) so a run can be audited/plotted without re-training.
        'lr': lr,
        'loss_curve': losses,
        'acc_curve': accs,
        **analysis_fields,
    }


# Error categories a task-aware bad-case analysis can distinguish. Labels
# on these synthetic tasks are exact (the generator computes the target),
# so a wrong prediction cannot be label noise: it is either a mechanism
# blind spot (e.g. KDA overwriting instead of accumulating), an incorrect
# association (choosing another key's value), or a positional shortcut
# (copying the cue / a distractor). Recording WHICH category dominates is
# the raw material for the next architecture idea.
_ERROR_CATEGORY_DEFINITIONS = {
    'mqar': OrderedDict([
        ('confused_with_other_value',
         'predicted the value paired with a DIFFERENT key (association '
         'mix-up between the n_kv pairs; grows with n_kv)'),
        ('echoed_key',
         'predicted a key token instead of its value (retrieval stopped at '
         'the key; positional copy of the cue region)'),
        ('copied_distractor_token',
         'predicted a filler token from the noise region (attention/state '
         'dilution instead of retrieval)'),
        ('other', 'none of the above'),
    ]),
    'kv_update': OrderedDict([
        ('used_stale_value',
         'predicted the OLD value for the cued key (delta-rule update failed '
         'to overwrite; classic interference failure)'),
        ('confused_with_other_new_value',
         'predicted another key\'s new value (update happened, association '
         'mix-up)'),
        ('echoed_key',
         'predicted a key token instead of its latest value'),
        ('other', 'none of the above'),
    ]),
    'long_range': OrderedDict([
        ('echoed_key',
         'predicted the key itself instead of its value'),
        ('copied_adjacent_token',
         'predicted the distractor immediately after the pair (local copy '
         'instead of long-range retrieval)'),
        ('other', 'none of the above'),
    ]),
    'multi_agg': OrderedDict([
        ('runner_up_category',
         'predicted the second-most-frequent category (off-by-one count; '
         'aggregation is close but not exact)'),
        ('echoed_last_token',
         'predicted the final token itself (recency copy rather than '
         'majority aggregation)'),
        ('other', 'none of the above'),
    ]),
}


def _classify_task_errors(task, meta, target, pred, cue_pos, n_kv):
    """Classify every sample in a batch into a wrong-answer category.

    Returns a list of category names (``'correct'`` for correct rows).
    Vectorized masks decide the category; the per-sample loop is only over
    the batch dimension and uses the task's known layout, not heuristics
    on the raw ids.
    """
    ids = meta['input_ids']
    b = int(ids.shape[0])
    device = ids.device
    correct = (pred == target)
    cat = ['correct'] * b
    extras = {}
    if task == 'mqar':
        keys = ids[:, 0:2 * n_kv:2]
        vals = ids[:, 1:2 * n_kv:2]
        is_other_val = (vals == pred[:, None]).any(dim=1)
        is_key = (keys == pred[:, None]).any(dim=1)
        noise = ids[:, 2 * n_kv:cue_pos]
        is_noise = ((noise == pred[:, None]).any(dim=1)
                    if noise.shape[1] else torch.zeros(
                        b, dtype=torch.bool, device=device))
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_other_val[i]):
                cat[i] = 'confused_with_other_value'
            elif bool(is_key[i]):
                cat[i] = 'echoed_key'
            elif bool(is_noise[i]):
                cat[i] = 'copied_distractor_token'
            else:
                cat[i] = 'other'
    elif task == 'kv_update':
        vals_old = ids[:, 1:2 * n_kv:2]
        keys2 = ids[:, 2 * n_kv:4 * n_kv:2]
        vals2 = ids[:, 2 * n_kv + 1:4 * n_kv:2]
        j = meta['key_idx']
        cue_key = ids[:, cue_pos]
        old_for_cue = vals_old.gather(1, j[:, None]).squeeze(1)
        is_stale = (pred == old_for_cue)
        is_other_new = (vals2 == pred[:, None]).any(dim=1)
        is_key2 = (keys2 == pred[:, None]).any(dim=1)
        extras['cue_key'] = cue_key
        extras['stale_value'] = old_for_cue
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_stale[i]):
                cat[i] = 'used_stale_value'
            elif bool(is_other_new[i]):
                cat[i] = 'confused_with_other_new_value'
            elif bool(is_key2[i]):
                cat[i] = 'echoed_key'
            else:
                cat[i] = 'other'
    elif task == 'long_range':
        p = meta['key_pos']
        key = ids.gather(1, p[:, None]).squeeze(1)
        is_key = (pred == key)
        adj_pos = torch.clamp(p + 2, max=max(0, cue_pos - 1))
        adj = ids.gather(1, adj_pos[:, None]).squeeze(1)
        is_adj = (pred == adj)
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_key[i]):
                cat[i] = 'echoed_key'
            elif bool(is_adj[i]):
                cat[i] = 'copied_adjacent_token'
            else:
                cat[i] = 'other'
    elif task == 'multi_agg':
        counts = torch.as_tensor(meta['counts'], device=device)
        top2 = counts.topk(2, dim=1).values
        pred_count = counts.gather(1, pred[:, None]).squeeze(1)
        is_runner_up = (pred_count == top2[:, 1])
        is_last = (pred == ids[:, -1])
        extras['counts'] = counts
        for i in range(b):
            if bool(correct[i]):
                continue
            if bool(is_runner_up[i]):
                cat[i] = 'runner_up_category'
            elif bool(is_last[i]):
                cat[i] = 'echoed_last_token'
            else:
                cat[i] = 'other'
    return cat, extras


def _error_example(task, meta, target, pred, i, cue_pos, extras):
    """Decode one misclassified sample into human-readable task terms."""
    ids = meta['input_ids']
    ex = {
        'input_ids': [int(v) for v in ids[i].tolist()],
        'target': int(target[i]),
        'pred': int(pred[i]),
        'cue_pos': int(cue_pos),
    }
    if task in ('mqar', 'kv_update'):
        ex['key_idx'] = int(meta['key_idx'][i])
    if task == 'kv_update':
        ex['cued_key'] = int(extras['cue_key'][i])
        ex['stale_value'] = int(extras['stale_value'][i])
    elif task == 'long_range':
        ex['key_pos'] = int(meta['key_pos'][i])
        ex['distance'] = int(meta['distance'][i])
    elif task == 'multi_agg':
        ex['counts'] = [int(c) for c in meta['counts'][i]]
        ex['pred_count'] = int(extras['counts'][i, int(pred[i])])
    return ex


def analyze_errors(layer, head, embed, *, op_name, task='mqar', seq_len=16,
                   n_kv=1, vocab=16, device='cpu', n_batches=2, batch=64,
                   seed=42, max_examples=2):
    """Bad-case analysis: classify and print held-out misclassifications.

    Runs the trained model (read-only) on ``n_batches`` x ``batch`` samples
    from the dedicated ``_ERROR_ANALYSIS_SEED_OFFSET`` stream, classifies
    every wrong prediction into a task-level category (see
    ``_ERROR_CATEGORY_DEFINITIONS``), keeps up to ``max_examples`` decoded
    examples per category, and prints a compact per-category summary with
    the first example. Synthetic labels are exact, so a wrong answer
    cannot be label noise: the category distribution localizes the
    mechanism blind spot (association mix-up, failed update, positional
    copy, ...) that the aggregate accuracy hides.

    The report is JSON-serializable and records the stream/seed so an
    example can be regenerated exactly.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    gen = make_seeded_generator(seed + _ERROR_ANALYSIS_SEED_OFFSET,
                                device=device)
    was_training = {m: m.training for m in (layer, head, embed)}
    counts: dict = OrderedDict()
    examples: dict = OrderedDict()
    n_total = n_wrong = 0
    dist_correct: dict = {}
    dist_total: dict = {}
    try:
        layer.eval()
        head.eval()
        embed.eval()
        with torch.no_grad():
            for _ in range(int(n_batches)):
                x_emb, target, cue_pos, meta = make_batch(
                    batch, seq_len, n_kv, vocab, embed, device,
                    generator=gen, return_meta=True)
                h = layer(x_emb)
                logits = head(h, cue_pos)
                pred = logits.argmax(-1)
                cats, extras = _classify_task_errors(
                    task, meta, target, pred, cue_pos, n_kv)
                n_total += int(target.numel())
                for i, c in enumerate(cats):
                    if c == 'correct':
                        continue
                    n_wrong += 1
                    counts[c] = counts.get(c, 0) + 1
                    bucket = examples.setdefault(c, [])
                    if len(bucket) < int(max_examples):
                        bucket.append(_error_example(
                            task, meta, target, pred, i, cue_pos, extras))
                if task == 'long_range':
                    ok = (pred == target)
                    d = meta['distance']
                    for i in range(int(target.numel())):
                        di = int(d[i])
                        dist_total[di] = dist_total.get(di, 0) + 1
                        dist_correct[di] = dist_correct.get(di, 0) + int(ok[i])
    finally:
        for m, was in was_training.items():
            m.train(was)

    categories = OrderedDict()
    for name, definition in _ERROR_CATEGORY_DEFINITIONS[task].items():
        n = counts.get(name, 0)
        categories[name] = {
            'count': int(n),
            'rate_of_wrong': (n / n_wrong) if n_wrong else 0.0,
            'rate_of_all': (n / n_total) if n_total else 0.0,
            'definition': definition,
        }
    for name in counts:
        if name not in categories:
            categories[name] = {
                'count': int(counts[name]),
                'rate_of_wrong': counts[name] / n_wrong if n_wrong else 0.0,
                'rate_of_all': (counts[name] / n_total) if n_total else 0.0,
                'definition': 'category not covered by the task layout',
            }

    by_distance = None
    if dist_total:
        by_distance = []
        for lo in sorted(dist_total):
            by_distance.append({
                'distance': int(lo),
                'n': int(dist_total[lo]),
                'accuracy': (dist_correct[lo] / dist_total[lo]
                             if dist_total[lo] else None),
            })

    if n_wrong:
        top = sorted(categories.items(), key=lambda kv: -kv[1]['count'])
        summary = ', '.join(f"{c}={v['count']}" for c, v in top if v['count'])
        logger.info(f"  [bad-case] {op_name}/{task} seed={seed}: "
                    f"{n_wrong}/{n_total} wrong ({100.0 * n_wrong / max(1, n_total):.1f}%); "
                    f"{summary}")
        shown = 0
        for c, v in top:
            if not v['count'] or c not in examples or not examples[c]:
                continue
            ex = examples[c][0]
            logger.info(f"  [bad-case] e.g. {c}: target={ex['target']} "
                        f"pred={ex['pred']} cue_pos={ex['cue_pos']} "
                        f"meta={ {k: v2 for k, v2 in ex.items() if k not in ('input_ids', 'target', 'pred', 'cue_pos')} }")
            logger.info(f"  [bad-case]   input_ids={ex['input_ids']}")
            shown += 1
            if shown >= 2:
                break
    else:
        logger.info(f"  [bad-case] {op_name}/{task} seed={seed}: no errors "
                    f"in {n_total} held-out samples")

    return {
        'op': op_name,
        'task': task,
        'n_kv': int(n_kv),
        'seed': int(seed),
        'stream': 'held-out error-analysis stream',
        'generator_seed': int(seed + _ERROR_ANALYSIS_SEED_OFFSET),
        'n_batches': int(n_batches),
        'batch': int(batch),
        'n_samples': int(n_total),
        'n_wrong': int(n_wrong),
        'error_rate': (n_wrong / n_total) if n_total else None,
        'categories': categories,
        'examples': examples,
        'accuracy_by_distance_bin': by_distance,
        'note': ('Errors are inspected on a stream never used for training, '
                 'validation monitoring, or the headline eval. Synthetic '
                 'labels are exact, so wrong answers are mechanism blind '
                 'spots / positional shortcuts, not label noise; use the '
                 'category mix to pick the next architecture experiment.'),
    }


# Fixed seed offset for the overfit gate's batch generator. Far away from
# the training offset (1_000_000) and the eval offset (5_000_000) so the
# gate's data stream can never collide with either.
_OVERFIT_BATCH_SEED_OFFSET = 3_000_000


def _task_batch_fingerprint(x_emb, target, cue_pos):
    """sha256 over the fixed overfit batch (embedded inputs + targets).

    Makes the gate report self-describing: a future reader can verify that
    a re-run with the same gate seed reproduces the same fixed batch, and
    detect a silently changed task generator or tokenizer between runs.
    """
    import hashlib

    h = hashlib.sha256()
    for name, t in (('x_emb', x_emb), ('target', target)):
        arr = t.detach().to('cpu').contiguous()
        h.update(f'{name}:{tuple(arr.shape)}:'.encode())
        h.update(str(arr.view(-1).tolist()).encode())
    h.update(f'cue_pos:{cue_pos}'.encode())
    return h.hexdigest()


def run_overfit_sanity_check(op_name, task='mqar', d_model=32, seq_len=16,
                             n_kv=1, vocab=16, steps=200, lr=3e-3,
                             seed=2026, device='cpu', train_batch=32,
                             acc_threshold=0.99, loss_ratio_threshold=0.5,
                             grad_clip=1.0, log_every=0):
    """Overfit ONE fixed batch as a pre-training gradient-flow gate.

    Trains a FRESH model (built through the exact ``train_one`` construction
    path) on a single fixed synthetic batch with all regularization
    disabled: ``weight_decay=0``, no dropout (none of these operators has
    any), and no data augmentation (the generators produce raw task batches;
    the same batch is regenerated deterministically every step from
    ``seed + _OVERFIT_BATCH_SEED_OFFSET``, so the autograd graph — including
    the embedding's gradient path — is fresh each step while the DATA is
    held constant).

    If the model cannot drive a fixed batch of a few dozen examples to
    near-100% train accuracy, something is fundamentally broken (gradients
    not flowing, loss mis-wired, labels reversed, optimizer not stepping)
    and the full multi-seed sweep would burn compute on a broken recipe.

    Pass criterion (either): mean train accuracy over the last ``tail``
    steps >= ``acc_threshold`` (default 0.99: "classification accuracy
    close to 100%"), OR the tail-mean loss dropped below
    ``loss_ratio_threshold`` times the head-mean loss ("loss rapidly
    approaching 0").

    This is a NECESSARY, NOT SUFFICIENT check: passing proves gradients
    flow and the loss responds on a fixed batch, but data-augmentation,
    feature-alignment and label-mapping bugs can still pass it. It must be
    complemented by inspecting intermediate results.

    Returns a JSON-serializable report with the loss/accuracy curves, the
    fixed-batch fingerprint, the seeded construction info, and the pass
    verdict.
    """
    import gc

    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    steps = int(steps)
    if steps < 10:
        # A handful of steps is dominated by early-loss noise (the loss can
        # rise for a few steps before collapsing); requiring >= 10 avoids a
        # misleading FAIL on a healthy model (mirrors
        # train_lm_autodl.run_overfit_sanity_check).
        raise ValueError(
            f"overfit check needs steps >= 10 to be meaningful, got {steps}")
    train_batch = int(train_batch)
    if train_batch < 1:
        raise ValueError(f"train_batch must be >= 1, got {train_batch}")
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    # Deterministic, regularization-free recipe. Re-seeding here makes the
    # probe model's init and data independent of any RNG consumption before
    # the call; the subsequent full training re-seeds per op/seed anyway, so
    # the gate cannot perturb the experiment's numbers.
    seed_info = set_global_seed(seed, deterministic=True)
    embed, head, layer = _build_task_model(op_name, d_model, vocab, device)
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.0)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layer.train()
    head.train()
    embed.train()

    losses, accs, elapsed_s = [], [], []
    first_batch = None
    grad_flow = None
    t0 = time.time()
    for step in range(steps):
        gen = make_seeded_generator(seed + _OVERFIT_BATCH_SEED_OFFSET,
                                    device=device)
        x_emb, target, cue_pos = make_batch(
            train_batch, seq_len, n_kv, vocab, embed, device,
            generator=gen)
        if first_batch is None:
            first_batch = (x_emb.detach(), target.detach(), cue_pos)
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()), inputs={'x_emb': x_emb},
                params=params, logits=logits,
                context=f'overfit sanity check step={step} '
                        f'op={op_name} task={task}')
            raise RuntimeError(
                f"overfit sanity check: non-finite loss at step {step} "
                f"({loss.item()}) for op={op_name} task={task}; diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            # One-shot gradient-flow audit on the first backward: names the
            # parameters the loss never reached (a detached / dead branch
            # can hide behind a passing accuracy or loss criterion).
            grad_flow = grad_flow_report(
                [(f'layer.{n}', p) for n, p in layer.named_parameters()]
                + [(f'head.{n}', p) for n, p in head.named_parameters()]
                + [(f'embed.{n}', p) for n, p in embed.named_parameters()])
        bad_grads = [p for p in params
                     if p.grad is not None
                     and not torch.isfinite(p.grad).all()]
        clip_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()), inputs={'x_emb': x_emb},
                params=params, logits=logits, grads=bad_grads or None,
                grad_norm=float(clip_norm),
                context=f'overfit sanity check step={step} '
                        f'op={op_name} task={task}')
            raise RuntimeError(
                f"overfit sanity check: non-finite gradient at step {step} "
                f"for op={op_name} task={task}; diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        opt.step()
        losses.append(float(loss.detach()))
        accs.append(float(acc))
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[overfit-check] {op_name}/{task} step {step:4d} "
                  f"loss {losses[-1]:.4f} acc {acc:.4f} "
                  f"({elapsed_s[-1]:.1f}s)", flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail
    loss_reduction = ((initial_loss - final_loss) / initial_loss
                      if initial_loss > 0 else None)
    passed = ((final_acc >= acc_threshold)
              or (final_loss < loss_ratio_threshold * initial_loss))
    report = {
        'passed': bool(passed),
        'op': op_name,
        'task': task,
        'n_kv': int(n_kv),
        'seq_len': int(seq_len),
        'vocab': int(vocab),
        'd_model': int(d_model),
        'steps': steps,
        'train_batch': train_batch,
        'lr': float(lr),
        'weight_decay': 0.0,
        'dropout': 0.0,
        'data_augmentation': 'none',
        'grad_clip_norm': float(grad_clip),
        'seed': int(seed),
        'batch_generator_seed': int(seed + _OVERFIT_BATCH_SEED_OFFSET),
        'seed_info': seed_info,
        'fixed_batch_sha256': _task_batch_fingerprint(*first_batch),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': min(losses),
        'loss_reduction': loss_reduction,
        'final_train_acc': final_acc,
        'best_train_acc': max(accs),
        'acc_threshold': float(acc_threshold),
        'loss_ratio_threshold': float(loss_ratio_threshold),
        'loss_curve': losses,
        'acc_curve': accs,
        'elapsed_s': elapsed_s,
        # Gradient-flow audit (first step): which trainable tensors the loss
        # actually reached, recorded next to the pass verdict.
        'grad_flow': grad_flow,
        'note': ('Necessary but NOT sufficient: passing proves gradients '
                 'flow and the loss responds on a fixed batch; it cannot '
                 'rule out data-augmentation / feature-alignment / '
                 'label-mapping bugs.'),
    }
    del embed, head, layer, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if grad_flow and grad_flow['n_missing_grad']:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the fixed batch for "
              f"{op_name}/{task} (e.g. "
              f"{grad_flow['missing_grad_names'][:3]})", flush=True)
    return report


def _compact_overfit_report(report):
    """Strip the per-step curves/seed block for compact metadata embedding.

    The full curves live in ``results/exp4_overfit_gate.json`` (written once
    per run); the exp4/exp4b metadata only needs the verdict and headline
    numbers, so embedding the curves twice would bloat every result file.
    """
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def train_multi_seed(op_name, n_seeds=5, steps=100, softmax_steps=None,
                     device='cpu', **kw):
    """Train ``op_name`` over ``n_seeds`` seeds.

    Returns a dict with per-seed results plus aggregate stats (mean, std,
    95% CI half-width via t-distribution). Any ``task=...`` keyword is
    forwarded to :func:`train_one` (default ``'mqar'``).

    Per-seed error handling: a single divergent seed (NaN loss, OOM, etc.)
    is caught and recorded as a stub entry in ``per_seed`` rather than
    crashing the whole operator's multi-seed run. Aggregate stats are
    computed over the surviving (successful) seeds. If ALL seeds fail, the
    function raises a ``RuntimeError`` so the caller can record a stub
    result for the operator (mirrors ``run_ablation.py::eval_layout_multi_seed``).

    ``device`` may be passed as a string or a ``torch.device``; string
    inputs are coerced for notebook callers.
    """
    # Coerce string device -> torch.device so callers passing 'cpu'/'cuda'
    # from a notebook don't hit AttributeError on ``device.type`` below.
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    # Compute the actual training-step count for THIS op once, mirroring
    # ``train_one``'s ``actual_steps = softmax_steps if op_name == 'softmax'
    # else steps`` logic, so the recorded ``steps`` field in the per-seed
    # error stub matches what a SUCCESS row would have recorded.
    actual_steps_for_op = softmax_steps if op_name == 'softmax' else steps
    stub_train_batch = kw.get('train_batch')
    if stub_train_batch is None:
        stub_train_batch = parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1,
                                         logger=logger)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        # Per-seed try/except: one divergent seed should not crash the
        # whole operator. We log and record a stub; the aggregate stats
        # are computed over whichever seeds succeeded.
        try:
            r = train_one(op_name, seed=s, steps=steps, device=device,
                          softmax_steps=softmax_steps, **kw)
            r['train_time_s'] = time.time() - t0
            # Log BEFORE appending to per_seed, so a failure inside
            # ``logger.info`` cannot record the seed as a success first and
            # then append a duplicate error stub for the same seed.
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"steps={r['steps']}  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            # Per-seed error stub: include ALL success-row fields (set to
            # None / 0 where not applicable) so downstream consumers using
            # ``pd.DataFrame(per_seed)`` or ``per_seed[0].get('steps', ...)``
            # see a consistent schema. ``pd.DataFrame(per_seed)`` uses the
            # first record's keys as columns, and
            # ``make_figures._plot_mqar_group`` reads
            # ``per_seed[0].get('steps', 100)`` for the figure title.
            per_seed.append({
                'op': op_name,
                'task': kw.get('task', 'mqar'),
                'n_kv': kw.get('n_kv', 1),
                'n_params': None,
                'final_acc': None,
                'final_loss': None,
                'chance_acc': 1.0 / kw.get('vocab', 16) if 'vocab' in kw else None,
                'last_train_acc': None,
                'mean_last10_loss': None,
                'mean_last10_acc': None,
                # Use the OP-SPECIFIC step count (softmax_steps for softmax,
                # steps for others) so the stub matches what a success row
                # would record, even when
                # MQAR_SOFTMAX_STEPS != MQAR_STEPS.
                'steps': actual_steps_for_op,
                'seed': s,
                'train_batch': stub_train_batch,
                'train_time_s': time.time() - t0,
                'error': str(e),
                # Mirror the success-row training-curve keys (empty) so the
                # per_seed schema is uniform across success/error records.
                'lr': kw.get('lr', 3e-3),
                'loss_curve': [],
                'acc_curve': [],
                # Mirror the results-analysis keys so a DataFrame over
                # per_seed has one schema for success and error rows.
                'val_history': [],
                'curve_diagnosis': None,
                'error_analysis': None,
                'final_eval_time_s': None,
                'final_eval_ms_per_sample': None,
                'generalization_gap': None,
                'best_val': None,
                'val_loss_regret_from_best': None,
                'val_acc_regret_from_best': None,
                'steps_since_best': None,
                'best_val_is_final': None,
                'early_stopping_summary': None,
                'accuracy_regime': None,
                'val_acc_first': None,
                'val_acc_last': None,
                'val_acc_peak': None,
                'gradient_norms': None,
                'final_metrics': None,
                'final_f1_macro': None,
                'final_f1_weighted': None,
                'final_balanced_accuracy': None,
                'final_pr_auc_macro_ovr': None,
                'final_roc_auc_macro_ovr': None,
            })
        # On GPU, clear the CUDA cache between seeds so the allocator does
        # not accumulate freed-but-unreleased blocks across seeds.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Filter out failed seeds before computing aggregate stats.
    # A seed that diverged to NaN loss is caught by train_one's NaN guard
    # and recorded with an 'error' key — but a seed that produced a NaN
    # final_acc/final_loss via a path the guard did not cover (e.g. eval
    # on NaN params from a step that crashed before the guard fired) would
    # survive the 'error' filter and silently corrupt the aggregate mean
    # (NaN propagates through sum()/std()). Defensively reject any seed
    # whose final_acc or final_loss is None or non-finite.
    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        # All seeds failed: propagate the error so the caller can record
        # a stub result for this operator.
        raise RuntimeError(
            f"all {n_seeds} seeds failed for op '{op_name}'; "
            f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    losses = [r['final_loss'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_loss = sum(losses) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / (n - 1)
        std_acc = math.sqrt(var_acc)
        std_loss = math.sqrt(var_loss)
        # 95% CI half-width using t-distribution with n-1 dof.
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
        ci_loss = t * std_loss / math.sqrt(n)
    else:
        # With a single seed, the sample std is undefined and the t-CI is
        # NOT zero — the uncertainty is maximal. Mirror the convention
        # used in run_ablation.py::eval_layout_multi_seed: return None so
        # downstream consumers (summary table, make_figures.py) can render
        # 'n/a' instead of a misleading '0.0000' that implies perfect
        # precision.
        std_acc = std_loss = 0.0
        ci_acc = ci_loss = None

    # One-sample t-test vs chance: tests whether mean_acc differs from the
    # chance level. The t-statistic is only defined when n > 1 and the
    # sample standard deviation is strictly positive; otherwise we return
    # None (the test is not computable, not "infinitely significant").
    chance = ok_per_seed[0]['chance_acc']
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    # Curve-diagnosis verdict tally over the surviving seeds: the experiment
    # record says HOW MANY seeds showed overfitting / underfitting, instead
    # of leaving the reader to infer it from the per-seed curves.
    verdicts: dict = OrderedDict()
    for r in ok_per_seed:
        v = (r.get('curve_diagnosis') or {}).get('verdict', 'unknown')
        verdicts[v] = verdicts.get(v, 0) + 1

    # Aggregate the curve-reading fields that a final-number-only table
    # hides: how far the final weights sit past the best validation point
    # (early-stopping regret), and how often validation accuracy was pinned
    # at the task ceiling (saturation, not overfitting).
    regrets = [r['val_loss_regret_from_best'] for r in ok_per_seed
               if r.get('val_loss_regret_from_best') is not None]
    regimes = [r.get('accuracy_regime') for r in ok_per_seed
               if r.get('accuracy_regime')]
    n_saturated = sum(1 for reg in regimes if reg.get('saturated'))
    tail_accs = [reg['val_acc_tail_mean'] for reg in regimes
                 if reg.get('val_acc_tail_mean') is not None]

    # Pooled bad-case analysis across seeds: counts add up over every
    # analyzed sample, while the EXAMPLES come from the worst seed (lowest
    # final accuracy) so the artifact shows the most informative failures
    # rather than a lucky seed's.
    error_aggregate = None
    ea_seeds = [r for r in ok_per_seed if r.get('error_analysis')]
    if ea_seeds:
        pooled: dict = OrderedDict()
        n_samples = n_wrong = 0
        for r in ea_seeds:
            ea = r['error_analysis']
            n_samples += ea['n_samples']
            n_wrong += ea['n_wrong']
            for c, v in ea['categories'].items():
                acc = pooled.setdefault(
                    c, {'count': 0, 'definition': v.get('definition')})
                acc['count'] += int(v['count'])
        for v in pooled.values():
            v['rate_of_wrong'] = v['count'] / n_wrong if n_wrong else 0.0
            v['rate_of_all'] = v['count'] / n_samples if n_samples else 0.0
        worst = min(ea_seeds, key=lambda r: r['final_acc'])
        error_aggregate = {
            'n_seeds': len(ea_seeds),
            'n_samples': n_samples,
            'n_wrong': n_wrong,
            'error_rate': (n_wrong / n_samples) if n_samples else None,
            'pooled_categories': pooled,
            'representative_seed': worst['seed'],
            'representative_examples': worst['error_analysis']['examples'],
            'accuracy_by_distance_bin': worst['error_analysis'].get(
                'accuracy_by_distance_bin'),
            'note': ('Pooled over seeds; examples come from the worst seed '
                     '(lowest final accuracy) so the artifact shows the '
                     'most informative failures. See the per-seed '
                     'error_analysis for each seed.'),
        }

    gaps = [r['generalization_gap'] for r in ok_per_seed
            if r.get('generalization_gap') is not None]

    # Seed-aggregated confusion metrics. Macro and weighted F1 are kept
    # SEPARATE (they differ under imbalance); micro is omitted here because
    # for single-label multi-class it is identical to accuracy, which the
    # row already carries (the per-seed metrics dict records it explicitly).
    def _seed_metric_mean(key):
        vals = [r['final_metrics'][key] for r in ok_per_seed
                if r.get('final_metrics')
                and r['final_metrics'].get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def _seed_metric_std(key):
        vals = [r['final_metrics'][key] for r in ok_per_seed
                if r.get('final_metrics')
                and r['final_metrics'].get(key) is not None]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var)

    mean_final_metrics = {
        'f1_macro_mean': _seed_metric_mean('f1_macro'),
        'f1_macro_std': _seed_metric_std('f1_macro'),
        'f1_weighted_mean': _seed_metric_mean('f1_weighted'),
        'f1_weighted_std': _seed_metric_std('f1_weighted'),
        'balanced_accuracy_mean': _seed_metric_mean('balanced_accuracy'),
        'pr_auc_macro_ovr_mean': _seed_metric_mean('pr_auc_macro_ovr'),
        'roc_auc_macro_ovr_mean': _seed_metric_mean('roc_auc_macro_ovr'),
        'class_balance': ((ok_per_seed[0].get('final_metrics') or {})
                          .get('class_balance')),
        'averaging_note': (
            'F1 variants are labelled explicitly: f1_macro = unweighted '
            'per-class mean, f1_weighted = support-weighted mean. Micro F1 '
            'equals accuracy for single-label multi-class and is recorded '
            'per seed in per_seed[i].final_metrics.f1_micro.'),
    }

    return {
        'op': op_name,
        'task': ok_per_seed[0].get('task', 'mqar'),
        'n_kv': ok_per_seed[0]['n_kv'],
        'n_params': ok_per_seed[0].get('n_params'),
        # Report the REQUESTED seed count (len(per_seed)), not the surviving
        # count (n), so the figure title and downstream consumers reflect
        # how many seeds were actually run. ``n_seeds_ok`` / ``n_seeds_failed``
        # carry the surviving / failed breakdown. Mirrors run_ablation.py's
        # convention (which already used len(per_seed) here).
        'n_seeds': len(per_seed),
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds_total': len(per_seed),
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'ci95_loss': ci_loss,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat,
        # The lr actually used for every seed of this row (tuned for the
        # baseline ops, 3e-3 otherwise). Per-seed values live in
        # per_seed[i]['lr']; they are identical across seeds.
        'lr': ok_per_seed[0].get('lr', 3e-3),
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
        'mean_eval_ms_per_sample': sum(
            r.get('final_eval_ms_per_sample', 0.0)
            for r in ok_per_seed) / n,
        'curve_diagnosis_summary': {
            'verdicts': dict(verdicts),
            'n_seeds_summarized': len(ok_per_seed),
        },
        'mean_generalization_gap': (sum(gaps) / len(gaps)) if gaps else None,
        # Best-validation regret: a positive value means the fixed-budget
        # final weights were past the turning point, so the final metric
        # understates what this operator achieved during training.
        'mean_val_loss_regret_from_best': (
            sum(regrets) / len(regrets)) if regrets else None,
        'n_seeds_with_positive_val_regret': sum(1 for x in regrets if x > 0),
        'accuracy_regime_summary': {
            'n_seeds_saturated': int(n_saturated),
            'n_seeds_with_regime': len(regimes),
            'mean_val_acc_tail': (
                sum(tail_accs) / len(tail_accs)) if tail_accs else None,
            'note': ('saturated = validation accuracy at the task ceiling '
                     'while train accuracy is still rising; this is '
                     'convergence, not overfitting, and more steps cannot '
                     'improve the metric.'),
        },
        'error_analysis': error_aggregate,
        'mean_final_metrics': mean_final_metrics,
    }


def _parse_nkv_list(env_var, default='1'):
    """Parse a comma-separated list of n_kv values from an env var.

    ``MQAR_NKV=1``       -> [1]           (backward compatible)
    ``MQAR_NKV=1,2,4``   -> [1, 2, 4]
    """
    raw = os.environ.get(env_var, default)
    try:
        vals = [int(x.strip()) for x in raw.split(',') if x.strip()]
    except ValueError:
        raise ValueError(
            f'{env_var}={raw!r} must be comma-separated ints, e.g. "1,2,4"')
    if not vals:
        vals = [int(default)]
    if any(v < 1 for v in vals):
        raise ValueError(
            f'{env_var} values must be positive integers, got {vals!r}')
    # Preserve user order while removing duplicates; duplicate task settings
    # would repeat identical experiments and overweight them in summaries.
    return list(dict.fromkeys(vals))


def _parse_task_list(env_var, default='mqar'):
    """Parse a comma-separated task list (or 'all') from an env var.

    ``MQAR_TASKS=all``            -> every task in SYNTHETIC_TASKS
    ``MQAR_TASKS=mqar,kv_update`` -> [mqar, kv_update]
    """
    raw = os.environ.get(env_var, default)
    if raw.strip().lower() == 'all':
        return list(SYNTHETIC_TASKS.keys())
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    if not vals:
        vals = [default]
    unknown = [v for v in vals if v not in SYNTHETIC_TASKS]
    if unknown:
        raise ValueError(
            f'{env_var} includes unknown task(s) {unknown}; '
            f'valid tasks: {sorted(SYNTHETIC_TASKS)} (or "all").')
    return list(dict.fromkeys(vals))


def _parse_op_list(env_var, default='linear,softmax,kda,mamba2,rwkv7,csa,hca'):
    """Parse the operator list for the quality sweep.

    ``MQAR_OPS`` selects which single-layer operators run. The default now
    starts with the SIMPLE-BASELINE LADDER — ``linear`` (no token mixing
    at all) and ``softmax`` (the simplest mixer) — followed by the
    complex operators (KDA / Mamba-2 / RWKV-7 / CSA / HCA). Every complex
    row is compared against the tuned reference baseline, not only
    against chance; see ``MQAR_BASELINE_*`` below. Set
    ``MQAR_OPS=softmax,kda,mamba2,rwkv7,csa,hca`` to reproduce the
    pre-baseline table exactly (or ``MQAR_OPS=softmax,kda,csa,hca`` for
    the historical four-operator table).
    """
    valid = ['linear', 'softmax', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca']
    raw = os.environ.get(env_var, default)
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    if not vals:
        vals = list(valid)
    unknown = [v for v in vals if v not in valid]
    if unknown:
        raise ValueError(
            f'{env_var} includes unknown op(s) {unknown}; valid ops: {valid}.')
    return list(dict.fromkeys(vals))


def _parse_lr_list(env_var, default='3e-4,1e-3,3e-3,1e-2,3e-2'):
    """Parse a comma-separated list of positive learning rates from an env var.

    Raises on a malformed / empty list (a silently empty tuning grid would
    turn the tuning protocol into a no-op).
    """
    raw = os.environ.get(env_var, default)
    vals = []
    for x in raw.split(','):
        x = x.strip()
        if not x:
            continue
        try:
            v = float(x)
        except ValueError:
            raise ValueError(
                f'{env_var}={raw!r} must be comma-separated floats, '
                f'e.g. "1e-3,3e-3,1e-2"')
        if not (v > 0):
            raise ValueError(f'{env_var} values must be > 0, got {v!r}')
        vals.append(v)
    if not vals:
        raise ValueError(
            f'{env_var}={raw!r} must contain at least one learning rate')
    return list(dict.fromkeys(vals))


def tune_baseline_lr(op_name, lrs, sweep, steps, softmax_steps, n_seeds,
                     device, val_every, val_batches, val_batch):
    """Select a baseline operator's learning rate on the VALIDATION stream.

    Experiment-design requirement: "implement the simplest baseline and TUNE
    IT WELL" before any complex model is declared better. This runs the
    baseline over ``lrs`` and picks the rate maximizing the held-out
    VALIDATION accuracy PLATEAU (mean of the last up-to-3 monitor points per
    seed; seed + ``_VAL_MONITOR_SEED_OFFSET`` stream), never the headline
    eval stream (seed + 5M), so selection cannot leak into any reported
    number. A single lr is selected per operator across all sweep
    cells (the row schema stays one-lr-per-op); the record keeps the full
    per-lr grid so the choice is auditable.

    Ties break toward the EARLIER entry in ``lrs`` (keep the more stable /
    smaller lr by convention, and keep the default 3e-3 grid order).

    Returns ``(best_lr, record)``. ``record`` is JSON-serializable and
    embedded in the run metadata.
    """
    actual_steps = softmax_steps if op_name == 'softmax' else steps
    # Ensure a validation signal exists even when the sweep disabled the
    # val monitor (MQAR_VAL_EVERY=0): the tuning runs then use a private
    # quarter-grid. This cannot change the headline metric — the monitor
    # stream is disjoint from the eval stream.
    tune_val_every = int(val_every) if int(val_every or 0) > 0 \
        else max(1, actual_steps // 4)

    # Selection statistic: mean of the LAST (up to) ``_TUNE_TAIL_K``
    # validation points per seed, averaged over seeds. The final point
    # alone is noisy (one unlucky monitor batch can flip the lr choice),
    # while the full curve would favour lrs that are still improving.
    # A tail window measures the converged plateau, which is what "tuned
    # well" should mean. The final-point mean is still recorded for
    # backward compatibility.
    _TUNE_TAIL_K = 3
    per_lr = OrderedDict()
    best_lr = lrs[0]
    best_score = None
    for lr in lrs:
        tail_scores = []
        final_scores = []
        tail_ks = []
        n_cell_runs_failed = 0
        for task, n_kv, task_seq_len, task_vocab in sweep:
            try:
                r = train_multi_seed(
                    op_name, n_seeds=n_seeds, steps=steps,
                    softmax_steps=softmax_steps, device=device,
                    lr=lr, task=task, n_kv=n_kv, vocab=task_vocab,
                    seq_len=task_seq_len, val_every=tune_val_every,
                    val_batches=val_batches, val_batch=val_batch,
                    error_batches=0)
            except Exception as e:
                n_cell_runs_failed += 1
                logger.warning(
                    f'    [tune] {op_name} lr={lr:g} {task}/n_kv={n_kv} '
                    f'FAILED: {e}')
                continue
            for seed_row in r.get('per_seed', []):
                vh = seed_row.get('val_history') or []
                if not vh:
                    continue
                final_scores.append(float(vh[-1]['acc']))
                k = min(_TUNE_TAIL_K, len(vh))
                tail_ks.append(k)
                tail_vals = [float(p['acc']) for p in vh[-k:]
                             if p.get('acc') is not None]
                if tail_vals:
                    tail_scores.append(sum(tail_vals) / len(tail_vals))
        mean_score = ((sum(tail_scores) / len(tail_scores))
                      if tail_scores else None)
        mean_final = ((sum(final_scores) / len(final_scores))
                      if final_scores else None)
        per_lr[f'{lr:g}'] = {
            'mean_val_tail_acc': mean_score,
            'mean_final_val_acc': mean_final,
            'val_tail_k': min(tail_ks) if tail_ks else 0,
            'n_val_points': len(tail_scores),
            'n_cell_runs_failed': n_cell_runs_failed,
        }
        logger.info(
            f'    [tune] {op_name:8s} lr={lr:g}  '
            f'val_tail_acc='
            f'{mean_score if mean_score is None else f"{mean_score:.4f}"} '
            f'(final='
            f'{mean_final if mean_final is None else f"{mean_final:.4f}"}, '
            f'{len(tail_scores)} points, {n_cell_runs_failed} failed cells)')
        # Strictly-greater: ties keep the earlier (user-order) lr.
        if mean_score is not None and (best_score is None
                                       or mean_score > best_score):
            best_score = mean_score
            best_lr = lr

    selected_final = (per_lr.get(f'{best_lr:g}', {})
                      .get('mean_final_val_acc'))
    return best_lr, {
        'op': op_name,
        'selected_lr': best_lr,
        'selected_lr_mean_val_acc': best_score,
        'selected_lr_mean_final_val_acc': selected_final,
        'selection_statistic': (f'tail mean: mean of the last min(3, n) '
                                f'validation points per seed, averaged '
                                f'over seeds'),
        'val_tail_k': _TUNE_TAIL_K,
        'selection_stream': f'val_monitor_seed+{_VAL_MONITOR_SEED_OFFSET}',
        'tune_seeds': n_seeds,
        'tune_val_every': tune_val_every,
        'lr_grid': per_lr,
        'note': ('Learning rate selected by the held-out validation '
                 'accuracy PLATEAU (tail mean, not a single final point), '
                 'and never by the headline eval stream; the winning lr is '
                 'then re-trained with the full seed count. Ties keep the '
                 'earlier/lower lr.'),
    }


def _paired_one_sided_t(diffs):
    """One-sided paired t-test statistics for an array of paired differences.

    H1: mean(diffs) > 0 (the candidate beats the reference). Returns a
    JSON-safe dict. ``t_stat`` is None when undefined (n < 2, or zero
    variance); in the zero-variance case ``all_positive`` records whether
    every seed difference was strictly positive so the caller can still
    treat a unanimous sweep as evidence.

    Zero variance is detected with ``multi_seed_stats._sample_sd``'s
    relative-tolerance rule: a perfectly constant difference vector (e.g.
    every seed improved by exactly 0.1) leaves a ~1e-17 floating-point
    residue in the naive variance, which produces a fake ``t ~ 1e16`` and
    would bypass the exact sign-flip evidence rule. The same numerical-zero
    clamp is used repo-wide.
    """
    n = len(diffs)
    if n == 0:
        return {'n_pairs': 0, 'mean_diff': None, 't_stat': None,
                'all_positive': False}
    mean_diff = sum(diffs) / n
    if n < 2:
        return {'n_pairs': n, 'mean_diff': mean_diff, 't_stat': None,
                'all_positive': all(d > 0 for d in diffs)}
    sd = _sample_sd(diffs, mean_diff)
    if sd is None or sd <= 0.0:
        return {'n_pairs': n, 'mean_diff': mean_diff, 't_stat': None,
                'all_positive': all(d > 0 for d in diffs)}
    return {'n_pairs': n, 'mean_diff': mean_diff,
            't_stat': mean_diff / (sd / math.sqrt(n)),
            'all_positive': all(d > 0 for d in diffs)}


_HYPOTHESIS_STATUSES = ('supported', 'contradicted', 'not_supported',
                        'not_computable', 'not_applicable')


def _hypothesis_verdicts(all_results, baseline_ops, baseline_ref):
    """Machine-readable verdicts for the pre-registered hypotheses.

    The repository records WHAT each cell was testing
    (``TASK_HYPOTHESES`` / ``OPERATOR_HYPOTHESES``). Each verdict below
    applies a rule fixed in code, reports its evidence, and uses one of
    five statuses:

      * ``supported``      — the prediction's signature is present;
      * ``contradicted``   — the falsification condition fired;
      * ``not_supported``  — the data are inconclusive (no significant
                             effect either way);
      * ``not_computable`` — required rows/statistics are missing;
      * ``not_applicable`` — the row is a tuned comparison reference, so
                             a superiority check does not apply to it.

    Checks (all pre-registered):

    1. ``linear_chance_control``: the memoryless ``linear`` baseline must
       stay at chance. Contradicted if it is significantly ABOVE chance in
       any cell (the task would then be solvable without token mixing,
       confounding the memory claim).
    2. ``kda_nkv_monotonicity``: KDA accuracy is predicted to degrade as
       the number of key-value pairs grows. Supported when accuracy is
       non-increasing over the swept n_kv values; contradicted when a
       higher n_kv unanimously improves every paired seed.
    3. ``beats_tuned_baseline``: every complex operator must beat the
       TUNED simple baseline on the seed-paired one-sided test before a
       superiority claim is allowed. A cell is contradicted only when the
       REVERSE paired test (reference - candidate) is itself significant,
       so absence of evidence is not mislabelled as refutation.

    Returns ``(checks, counts)``: the per-rule verdict records and the
    per-row primary-status counts embedded in the run metadata.
    """
    def _cell(r):
        return f"{r.get('task', 'mqar')}/n_kv={r['n_kv']}"

    checks = OrderedDict()
    per_row = {}

    def _assign(r, name, verdict):
        per_row.setdefault(id(r), OrderedDict())[name] = verdict

    # ------------------------------------------------------------------
    # 1. Memoryless linear control.
    # ------------------------------------------------------------------
    linear_rows = [r for r in all_results
                   if r.get('op') == 'linear' and 'error' not in r
                   and r.get('mean_acc') is not None]
    evidence = []
    contradicted_cells = []
    missing_chance_cells = []
    for r in linear_rows:
        chance = r.get('chance_acc')
        sig = bool(r.get('significant_bonferroni'))
        above = chance is not None and r['mean_acc'] > chance
        if chance is None:
            cell_verdict = 'not_computable'
        elif sig and above:
            cell_verdict = 'contradicted'
        else:
            cell_verdict = 'supported'
        rec = {
            'cell': _cell(r),
            'mean_acc': float(r['mean_acc']),
            'chance_acc': chance,
            't_stat_vs_chance': r.get('t_stat_vs_chance'),
            'significant_bonferroni': sig,
            'prediction_holds': (None if chance is None
                                 else not (sig and above)),
        }
        evidence.append(rec)
        if cell_verdict == 'contradicted':
            contradicted_cells.append(_cell(r))
        elif cell_verdict == 'not_computable':
            missing_chance_cells.append(_cell(r))
        _assign(r, 'linear_chance_control', {
            'rule': ('linear must NOT be significantly above chance '
                     '(otherwise the task is solvable without token mixing)'),
            'status': cell_verdict,
            'falsified': cell_verdict == 'contradicted',
            'evidence': rec,
        })
    if not linear_rows:
        status = 'not_computable'
    elif contradicted_cells:
        status = 'contradicted'
    elif missing_chance_cells:
        status = 'not_computable'
    else:
        status = 'supported'
    checks['linear_chance_control'] = {
        'rule': ('linear must NOT be significantly above chance '
                 '(otherwise the task is solvable without token mixing)'),
        'status': status,
        'falsified': status == 'contradicted',
        'n_cells': len(linear_rows),
        'n_contradicting_cells': len(contradicted_cells),
        'contradicting_cells': contradicted_cells,
        'evidence': evidence,
    }

    # ------------------------------------------------------------------
    # 2. KDA accuracy degrades with n_kv.
    # ------------------------------------------------------------------
    kda_by_task = OrderedDict()
    for r in all_results:
        if r.get('op') != 'kda' or 'error' in r:
            continue
        if r.get('mean_acc') is None:
            continue
        kda_by_task.setdefault(r.get('task', 'mqar'), []).append(r)
    pair_evidence = []
    monotone = True
    missing_pairs = []
    unanimous_increase = []
    for task, rows in kda_by_task.items():
        rows = sorted(rows, key=lambda x: x['n_kv'])
        for lo, hi in zip(rows, rows[1:]):
            base = {s['seed']: s['final_acc']
                    for s in lo.get('per_seed', [])
                    if 'error' not in s and s.get('final_acc') is not None}
            cand = {s['seed']: s['final_acc']
                    for s in hi.get('per_seed', [])
                    if 'error' not in s and s.get('final_acc') is not None}
            common = sorted(set(base) & set(cand))
            diffs = [cand[s] - base[s] for s in common]
            mean_diff = (sum(diffs) / len(diffs)) if diffs else None
            all_up = bool(diffs) and all(d > 1e-12 for d in diffs)
            all_down = bool(diffs) and all(d < -1e-12 for d in diffs)
            rec = {
                'task': task,
                'n_kv_low': int(lo['n_kv']),
                'n_kv_high': int(hi['n_kv']),
                'mean_acc_low': float(lo['mean_acc']),
                'mean_acc_high': float(hi['mean_acc']),
                'mean_diff_high_minus_low': mean_diff,
                'n_pairs': len(common),
                'all_paired_seeds_increase': all_up,
                'all_paired_seeds_decrease': all_down,
            }
            pair_evidence.append(rec)
            if mean_diff is None:
                monotone = False
                missing_pairs.append(
                    f"{task}: n_kv {lo['n_kv']}->{hi['n_kv']}")
            elif mean_diff > 1e-12:
                monotone = False
            if all_up:
                unanimous_increase.append(
                    f"{task}: n_kv {lo['n_kv']}->{hi['n_kv']}")
    if len(pair_evidence) == 0:
        kda_status = 'not_computable'
    elif unanimous_increase:
        kda_status = 'contradicted'
    elif missing_pairs:
        # A missing common-seed pair means monotonicity was not evaluated
        # for every requested step; that is missing evidence, not evidence
        # that the prediction failed.
        kda_status = 'not_computable'
    elif monotone:
        kda_status = 'supported'
    else:
        kda_status = 'not_supported'
    kda_verdict = {
        'rule': ('KDA mean accuracy must be non-increasing across the swept '
                 'n_kv values; a unanimous paired increase at higher n_kv '
                 'refutes the capacity-degradation prediction'),
        'status': kda_status,
        'falsified': kda_status == 'contradicted',
        'unanimous_increases': unanimous_increase,
        'missing_pairs': missing_pairs,
        'evidence': pair_evidence,
    }
    checks['kda_nkv_monotonicity'] = kda_verdict
    for rows in kda_by_task.values():
        for r in rows:
            _assign(r, 'kda_nkv_monotonicity', kda_verdict)

    # ------------------------------------------------------------------
    # 3. Complex operators vs the tuned simple baseline.
    # ------------------------------------------------------------------
    ref_by_cell = {
        (r.get('task', 'mqar'), r['n_kv']): r
        for r in all_results
        if r.get('op') == baseline_ref and 'error' not in r
        and r.get('mean_acc') is not None
    }
    candidates = [r for r in all_results
                  if 'error' not in r and r.get('op') not in set(baseline_ops)
                  and r.get('mean_acc') is not None]
    n_reverse_tests = sum(
        1 for r in candidates
        if ref_by_cell.get((r.get('task', 'mqar'), r['n_kv'])) is not None)
    reverse_alpha = (0.05 / n_reverse_tests) if n_reverse_tests else None
    complex_evidence = []
    for r in candidates:
        key = _cell(r)
        if r.get('beats_tuned_baseline') is True:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'supported',
                'falsified': False,
                'evidence': {
                    'cell': key,
                    'reference_op': baseline_ref,
                    'vs_tuned_baseline': r.get('vs_tuned_baseline'),
                },
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('supported', key, r.get('op')))
            continue
        ref = ref_by_cell.get((r.get('task', 'mqar'), r['n_kv']))
        if ref is None:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'not_computable',
                'falsified': False,
                'evidence': {'cell': key,
                             'reason': 'reference_row_unavailable'},
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('not_computable', key, r.get('op')))
            continue
        cand = {s['seed']: s['final_acc']
                for s in r.get('per_seed', [])
                if 'error' not in s and s.get('final_acc') is not None}
        base = {s['seed']: s['final_acc']
                for s in ref.get('per_seed', [])
                if 'error' not in s and s.get('final_acc') is not None}
        common = sorted(set(cand) & set(base))
        if len(common) < 2:
            verdict = {
                'rule': ('complex op must beat the TUNED simple baseline '
                         'on the seed-paired one-sided test'),
                'status': 'not_computable',
                'falsified': False,
                'evidence': {'cell': key, 'n_pairs': len(common),
                             'reason': 'insufficient_paired_seeds'},
            }
            _assign(r, 'beats_tuned_baseline', verdict)
            complex_evidence.append(('not_computable', key, r.get('op')))
            continue
        reverse = _paired_one_sided_t([base[s] - cand[s] for s in common])
        crit = (_bonferroni_crit_q(reverse['n_pairs'], alpha=reverse_alpha)
                if (reverse_alpha is not None
                    and reverse['n_pairs'] >= 2) else None)
        ref_better = False
        if reverse['mean_diff'] is not None and reverse['mean_diff'] > 0:
            if (reverse['t_stat'] is not None and crit is not None
                    and reverse['t_stat'] > crit):
                ref_better = True
            elif (reverse['t_stat'] is None and reverse['all_positive']
                  and reverse['n_pairs'] >= 2 and reverse_alpha is not None
                  and 2.0 ** (-(reverse['n_pairs'] - 1)) <= reverse_alpha):
                ref_better = True
        status = 'contradicted' if ref_better else 'not_supported'
        verdict = {
            'rule': ('complex op must beat the TUNED simple baseline; a '
                     'significant REVERSE paired test refutes superiority '
                     '(failure to beat is only "not_supported")'),
            'status': status,
            'falsified': status == 'contradicted',
            'evidence': {
                'cell': key,
                'reference_op': baseline_ref,
                'reverse_mean_diff_reference_minus_op':
                    reverse['mean_diff'],
                'reverse_t_stat': reverse['t_stat'],
                'reverse_t_crit_bonferroni': crit,
                'reverse_alpha_bonferroni': reverse_alpha,
                'n_pairs': reverse['n_pairs'],
                'vs_tuned_baseline': r.get('vs_tuned_baseline'),
            },
        }
        _assign(r, 'beats_tuned_baseline', verdict)
        complex_evidence.append((status, key, r.get('op')))

    complex_statuses = [s for s, _, _ in complex_evidence]
    if not complex_statuses:
        complex_status = 'not_computable'
    elif all(s == 'not_computable' for s in complex_statuses):
        complex_status = 'not_computable'
    elif 'contradicted' in complex_statuses:
        complex_status = 'contradicted'
    elif all(s == 'supported' for s in complex_statuses):
        complex_status = 'supported'
    else:
        complex_status = 'not_supported'
    checks['beats_tuned_baseline'] = {
        'rule': ('every complex operator must beat the tuned simple '
                 'baseline, seed-paired; a significant reverse test '
                 'contradicts superiority'),
        'status': complex_status,
        'falsified': complex_status == 'contradicted',
        'reference_op': baseline_ref,
        'reverse_alpha_bonferroni': reverse_alpha,
        'n_reverse_tests': n_reverse_tests,
        'per_cell': [
            {'status': s, 'cell': cell, 'op': op}
            for s, cell, op in complex_evidence
        ],
    }

    # Attach a compact primary status per row (first applicable check) so
    # downstream consumers get one machine-readable field without having
    # to know the check list.
    for r in all_results:
        applicable = per_row.get(id(r)) or OrderedDict()
        if 'error' in r:
            r['hypothesis_checks'] = {
                'row': {'status': 'not_computable', 'rule': 'row_error',
                        'falsified': False,
                        'evidence': {'error': r.get('error')}}}
            r['hypothesis_status'] = 'not_computable'
            continue
        if not applicable and r.get('op') in set(baseline_ops):
            r['hypothesis_checks'] = {
                'reference_baseline': {
                    'rule': ('this row IS the tuned comparison reference; '
                             'no superiority claim applies to it'),
                    'status': 'not_applicable',
                    'falsified': False,
                    'evidence': {'op': r.get('op')},
                }}
            r['hypothesis_status'] = 'not_applicable'
            continue
        r['hypothesis_checks'] = applicable
        statuses = [v.get('status') for v in applicable.values()]
        if 'contradicted' in statuses:
            r['hypothesis_status'] = 'contradicted'
        elif 'not_supported' in statuses:
            r['hypothesis_status'] = 'not_supported'
        elif statuses and all(s == 'supported' for s in statuses):
            r['hypothesis_status'] = 'supported'
        else:
            r['hypothesis_status'] = 'not_computable'

    counts = {s: 0 for s in _HYPOTHESIS_STATUSES}
    for r in all_results:
        counts[r.get('hypothesis_status', 'not_computable')] = (
            counts.get(r.get('hypothesis_status', 'not_computable'), 0) + 1)
    return checks, counts


def _pairwise_op_comparisons(all_results, alpha=0.05, seed=0,
                             n_boot=2000, n_permutations=10000,
                             max_exact=16, equivalence_margin=None):
    """Seed-paired, Holm-corrected operator-vs-operator comparisons.

    The per-row ``significant_bonferroni`` flag only says "this operator
    learned the task" and the ``vs_tuned_baseline`` test says "this
    operator beat the SIMPLE baseline". Neither answers the question the
    paper reader actually asks: *is operator A better than operator B,
    and is that ordering reproducible across seeds?* Comparing mean
    accuracies (or, worse, CI overlap) across two independently trained
    operators is not a valid test: the seeds are shared, so the correct
    statistic is the SEED-PAIRED difference.

    For every unordered operator pair inside a ``(task, n_kv)`` cell this
    function computes the per-seed differences ``acc(op_a) - acc(op_b)``
    on the COMMON seed set and runs:

    * the exact sign-flip permutation test (``multi_seed_stats``) as the
      PRIMARY statistic (two-sided, assumption-free for the 5-7-seed
      regime; exact enumeration below ``max_exact`` pairs, deterministic
      Monte-Carlo above);
    * the paired t-test as a secondary, normality-dependent view;
    * the exact binomial sign test, Cohen's ``d_z``, win rate, and two
      bootstrap/t CIs for the mean difference;
    * ``seeds_required_80pct_power`` — how many seeds a follow-up run
      would need at the observed |d_z|. This converts an inconclusive
      comparison into an actionable multi-seed budget.

    When ``equivalence_margin`` is not None, every testable pair ALSO
    gets a TOST equivalence test against that PRE-SPECIFIED margin
    (``multi_seed_stats.tost_paired``), with its p-values Holm-corrected
    in a second family over the same pairs. This separates two claims
    that are routinely conflated:

    * ``verdict`` — a significant seed-paired difference (superiority);
    * ``equivalence_verdict`` — the ``(1 - 2*alpha)`` CI for the mean
      paired difference lies strictly inside +-margin (practical
      equivalence).

    A pair can be non-significant AND non-equivalent ("inconclusive at
    this seed budget"), and — for tiny-but-consistent differences — even
    significant AND practically equivalent; both are reported separately.
    A non-significant pair is NEVER labelled 'equivalent' by default:
    the margin comes from ``PAIRWISE_EQUIVALENCE_MARGIN`` /
    ``MQAR_EQUIV_MARGIN`` before the run, not from the observed CI.
    Each equivalence record also carries the smallest margin this run
    could have supported (``equivalence_margin_supported``) so an
    inconclusive pair is actionable.

    Multiple-comparison control: Holm-Bonferroni step-down within each
    cell (family = all testable unordered pairs of that cell), which
    controls FWER <= alpha per experimental question and is uniformly
    more powerful than plain Bonferroni at this seed count. p-values are
    adjusted for the permutation, the t-test, AND (when enabled) the TOST
    equivalence test — each in its own family.

    Side effect: every row participating in >= 1 testable pair gets
    ``pairwise_tests``, ``pairwise_significant_wins`` and
    ``pairwise_significant_losses`` (operator-name lists) so downstream
    consumers can read a row's pairwise standing without re-deriving it.
    With equivalence enabled they also get
    ``pairwise_equivalent_to``.

    Returns a JSON-safe report (also stored in the exp4 metadata under
    ``op_pairwise_comparisons``).
    """
    groups: dict = OrderedDict()
    for r in all_results:
        if 'error' in r or r.get('mean_acc') is None:
            continue
        groups.setdefault((r.get('task', 'mqar'), r['n_kv']), []).append(r)

    report = {
        'alpha': alpha,
        'correction': 'holm_bonferroni',
        'family_scope': (
            'one family per (task, n_kv) cell: all unordered operator '
            'pairs with >= 2 common seeds'),
        'primary_test': (
            'two-sided exact sign-flip permutation test on the seed-paired '
            'accuracy differences; the paired t-test is reported as a '
            'secondary normality-dependent view'),
        'direction_convention': (
            'mean_diff = acc(op_a) - acc(op_b) on the common seed set'),
        'bootstrap': {'n_boot': n_boot, 'seed': seed},
        'ci95_overlap_note': (
            'CI overlap between two operators is a heuristic only and is '
            'neither necessary nor sufficient for a difference; the '
            'seed-paired permutation test here is the authoritative '
            'pairwise statistic.'),
        'n_tests': 0,
        'n_significant': 0,
        'n_skipped': 0,
        'n_equivalent': 0,
        'equivalence': {
            'enabled': equivalence_margin is not None,
            'margin': equivalence_margin,
            'margin_source': (
                'pre_specified_config (PAIRWISE_EQUIVALENCE_MARGIN, '
                'overridable via the pre-run env MQAR_EQUIV_MARGIN; '
                '0 disables)'),
            'method': 'two one-sided paired t-tests (TOST)',
            'ci_level': (1.0 - 2.0 * alpha) if equivalence_margin is not None
            else None,
            'correction': ('holm_bonferroni over the same per-cell pairs '
                           'family (separate family from the superiority '
                           'tests)'),
            'note': (
                'equivalence_verdict=equivalent means the (1-2*alpha) CI '
                'of the mean paired difference lies strictly inside '
                '+-margin: the two operators are practically the same '
                'WITHIN the pre-specified margin. not_equivalent covers '
                'both "real difference" and "inconclusive"; use the '
                'superiority verdict and equivalence_margin_supported to '
                'tell those apart.'),
        },
        'cells': {},
    }

    for (task_name, n_kv_key), rows in groups.items():
        cell_key = f'{task_name}/n_kv={n_kv_key}'
        rows_sorted = sorted(rows, key=lambda r: str(r.get('op')))
        cell = {'n_tests': 0, 'n_significant': 0, 'n_skipped': 0,
                'n_equivalent': 0, 'pairs': OrderedDict()}
        records = []
        for i in range(len(rows_sorted)):
            for j in range(i + 1, len(rows_sorted)):
                a, b = rows_sorted[i], rows_sorted[j]
                pair_key = f"{a.get('op')}~{b.get('op')}"
                a_seed = {s['seed']: s['final_acc']
                          for s in a.get('per_seed', [])
                          if 'error' not in s and s.get('final_acc') is not None}
                b_seed = {s['seed']: s['final_acc']
                          for s in b.get('per_seed', [])
                          if 'error' not in s and s.get('final_acc') is not None}
                common = sorted(set(a_seed) & set(b_seed))
                if len(common) < 2:
                    cell['pairs'][pair_key] = {
                        'op_a': a.get('op'), 'op_b': b.get('op'),
                        'common_seeds': common, 'n_pairs': len(common),
                        'verdict': 'insufficient_paired_seeds',
                    }
                    cell['n_skipped'] += 1
                    continue
                diffs = [a_seed[s] - b_seed[s] for s in common]
                stats = summarize_paired(
                    diffs, seed=seed, n_boot=n_boot,
                    n_permutations=n_permutations,
                    max_exact=max_exact, alpha=alpha,
                    equivalence_margin=equivalence_margin)
                rec = {'op_a': a.get('op'), 'op_b': b.get('op'),
                       'common_seeds': common, **stats}
                records.append((rec, a, b))
                cell['pairs'][pair_key] = rec

        p_perm = [rec['permutation_test']['p_value_two_sided']
                  for rec, _, _ in records]
        p_t_raw = [rec['t_test']['p_value_two_sided']
                   for rec, _, _ in records]
        holm_perm = holm_adjust(p_perm, alpha=alpha)
        # The t-test is undefined for a zero-variance difference vector
        # (t = +-inf); those pairs stay in the family with p=1.0 so the
        # family size (and therefore the correction strength) matches the
        # number of comparisons actually run, but they can never reject.
        holm_t = holm_adjust([1.0 if p is None else p for p in p_t_raw],
                             alpha=alpha)
        # TOST equivalence family (same pairs, separate question). Pairs
        # never have < 2 common seeds here (they were skipped above), so
        # the TOST p-value is always defined; the None guard is defensive.
        holm_tost = None
        if equivalence_margin is not None:
            p_tost = [rec['tost']['p_value'] for rec, _, _ in records]
            holm_tost = holm_adjust(
                [1.0 if p is None else p for p in p_tost], alpha=alpha)
        for idx, (rec, a, b) in enumerate(records):
            rec['holm_adjusted_p_permutation'] = holm_perm['adjusted'][idx]
            rec['significant_holm_permutation'] = bool(
                holm_perm['rejected'][idx])
            if p_t_raw[idx] is None:
                rec['holm_adjusted_p_t'] = None
                rec['significant_holm_t'] = False
            else:
                rec['holm_adjusted_p_t'] = holm_t['adjusted'][idx]
                rec['significant_holm_t'] = bool(holm_t['rejected'][idx])
            if equivalence_margin is not None:
                p_tost = rec.get('tost', {}).get('p_value')
                if p_tost is None:
                    rec['holm_adjusted_p_tost'] = None
                    rec['significant_equivalence_holm'] = False
                else:
                    rec['holm_adjusted_p_tost'] = holm_tost['adjusted'][idx]
                    rec['significant_equivalence_holm'] = bool(
                        holm_tost['rejected'][idx])
                rec['equivalence_verdict'] = (
                    'equivalent' if rec['significant_equivalence_holm']
                    else 'not_equivalent')
            mean = rec.get('mean_diff')
            if rec['significant_holm_permutation'] and mean:
                rec['verdict'] = 'op_a_better' if mean > 0 else 'op_b_better'
            else:
                rec['verdict'] = 'not_significant'
            for r in (a, b):
                r['pairwise_tests'] = r.get('pairwise_tests', 0) + 1
                r.setdefault('pairwise_significant_wins', [])
                r.setdefault('pairwise_significant_losses', [])
                if equivalence_margin is not None:
                    r.setdefault('pairwise_equivalent_to', [])
            if rec['verdict'] == 'op_a_better':
                a['pairwise_significant_wins'].append(b.get('op'))
                b['pairwise_significant_losses'].append(a.get('op'))
            elif rec['verdict'] == 'op_b_better':
                b['pairwise_significant_wins'].append(a.get('op'))
                a['pairwise_significant_losses'].append(b.get('op'))
            if equivalence_margin is not None and \
                    rec['equivalence_verdict'] == 'equivalent':
                a['pairwise_equivalent_to'].append(b.get('op'))
                b['pairwise_equivalent_to'].append(a.get('op'))
        cell['n_tests'] = len(records)
        cell['n_significant'] = sum(
            1 for rec, _, _ in records if rec['verdict'] != 'not_significant')
        cell['n_equivalent'] = sum(
            1 for rec, _, _ in records
            if rec.get('equivalence_verdict') == 'equivalent')
        report['cells'][cell_key] = cell
        report['n_tests'] += cell['n_tests']
        report['n_significant'] += cell['n_significant']
        report['n_equivalent'] += cell['n_equivalent']
        report['n_skipped'] += cell['n_skipped']

    report['note'] = (
        'Pairwise claims must cite these tests, not raw mean rankings or '
        'CI overlap. When a comparison is inconclusive, '
        'seeds_required_80pct_power estimates the seed budget the next '
        'run would need at the observed effect size. A "same/practically '
        'equivalent" claim requires equivalence_verdict=equivalent from '
        'the pre-specified-margin TOST; a non-significant superiority test '
        'alone never supports it.')
    return report


def _print_pairwise_report(op_pairwise):
    """Print the per-cell paired-comparison table (stdout, not just logs)."""
    if not op_pairwise.get('n_tests'):
        print('\nSeed-paired op-vs-op comparisons: no testable pairs '
              '(need >= 2 common seeds per pair).')
        return
    eq = op_pairwise.get('equivalence', {})
    eq_on = bool(eq.get('enabled'))
    title = ('(permutation two-sided; Holm-corrected within cell'
             + (f"; TOST equivalence margin={eq.get('margin')!r}"
                if eq_on else '') + '):')
    print(f'\nSeed-paired op-vs-op comparisons {title}')
    header = (f"{'cell':>28} | {'op_a':>10} | {'op_b':>10} | {'d_acc':>8} | "
              f"{'n':>3} | {'p_perm':>9} | {'p_holm':>9} | {'seeds@80%':>9} | "
              f"verdict")
    if eq_on:
        header += f" | {'p_tost':>9} | equivalence"
    print(header)
    for cell_key, cell in op_pairwise['cells'].items():
        for rec in cell['pairs'].values():
            if rec.get('verdict') in (None, 'insufficient_paired_seeds'):
                continue
            d = rec.get('mean_diff')
            d_str = f'{d:+.4f}' if d is not None else 'n/a'
            p = rec['permutation_test']['p_value_two_sided']
            p_str = f'{p:.4f}' if p is not None else 'n/a'
            ph = rec.get('holm_adjusted_p_permutation')
            ph_str = f'{ph:.4f}' if ph is not None else 'n/a'
            need = rec.get('seeds_required_80pct_power')
            need_str = str(need) if need is not None else 'n/a'
            row = (f"{cell_key:>28} | {str(rec.get('op_a')):>10} | "
                   f"{str(rec.get('op_b')):>10} | {d_str:>8} | "
                   f"{rec.get('n_pairs', 0):>3} | {p_str:>9} | "
                   f"{ph_str:>9} | {need_str:>9} | {rec['verdict']}")
            if eq_on:
                pt = rec.get('holm_adjusted_p_tost')
                pt_str = f'{pt:.4f}' if pt is not None else 'n/a'
                row += (f" | {pt_str:>9} | "
                        f"{rec.get('equivalence_verdict', 'n/a')}")
            print(row)


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 4: Synthetic Quality Probes (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device        : {device}')
    # Task and operator selection. Defaults keep the historical MQAR-only
    # output file (results/exp4_mqar.json) while ADDING the mamba2 operator
    # to the sweep; the additional probes are opt-in via MQAR_TASKS so the
    # historical table stays reproducible with MQAR_TASKS=mqar +
    # MQAR_OPS=softmax,kda,csa,hca.
    task_list = _parse_task_list('MQAR_TASKS', 'mqar')
    # Default ops include the baseline ladder (linear, softmax) followed
    # by the complex mixers; see _parse_op_list. The historical table is
    # reproducible with MQAR_OPS=softmax,kda,mamba2,rwkv7,csa,hca.
    ops_list = _parse_op_list('MQAR_OPS')
    logger.info(f'  tasks         : {task_list}')
    logger.info(f'  ops           : {ops_list}')
    # Tuned-baseline protocol (experiment-design requirement). The simple
    # baselines get their own learning-rate sweep, selected on the held-out
    # VALIDATION stream, and every complex op is tested against the tuned
    # reference baseline — not only against chance. MQAR_BASELINE_TUNE=0
    # restores the historical fixed lr=3e-3 for every op.
    baseline_tune_enabled = parse_int_env(
        'MQAR_BASELINE_TUNE', 1, min_value=0, logger=logger) != 0
    baseline_ops = [x.strip() for x in
                    os.environ.get('MQAR_BASELINE_OPS', 'linear,softmax')
                    .split(',') if x.strip()]
    # Silently intersect with the sweep's op list: naming a baseline op that
    # is not being swept (e.g. MQAR_OPS without 'linear') is not an error.
    baseline_ops = [op for op in baseline_ops if op in ops_list]
    baseline_lrs = _parse_lr_list('MQAR_BASELINE_LRS')
    # Reference baseline for the paired comparisons. Defaults to softmax
    # (the simplest mixer); if it is not in the sweep, fall back to the
    # first swept baseline op. None => no pairwise test possible.
    _baseline_ref_env = os.environ.get('MQAR_BASELINE_REF')
    baseline_ref = (_baseline_ref_env or 'softmax').strip()
    if baseline_ref not in ops_list:
        if _baseline_ref_env is not None:
            logger.warning(
                f'  MQAR_BASELINE_REF={baseline_ref!r} is not in the op '
                f'list {ops_list}; falling back to the first swept '
                f'baseline op.')
        baseline_ref = baseline_ops[0] if baseline_ops else None
    if baseline_tune_enabled and baseline_ops:
        logger.info(f'  baseline tune : ops={baseline_ops} '
                    f'lrs={baseline_lrs} (validation-stream selection); '
                    f'reference={baseline_ref}')
    elif not baseline_tune_enabled:
        logger.info('  baseline tune : OFF (MQAR_BASELINE_TUNE=0; '
                    'historical fixed lr=3e-3 for every op)')
    # MQAR's task hyperparameters. Lifted to named constants so the
    # chance-accuracy log line below uses the SAME value as the per-seed
    # ``chance = 1.0 / vocab`` computation in ``train_one``.
    vocab = 16
    seq_len = 16
    n_kv_list = _parse_nkv_list('MQAR_NKV', '1')
    # Pre-validate n_kv against vocab and seq_len so the user gets a clear
    # error message instead of an opaque AssertionError from deep inside
    # ``make_mqar_batch`` during training.
    for n_kv in n_kv_list:
        if 2 * n_kv > vocab:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"vocab={vocab}; reduce n_kv or increase vocab.")
        if 2 * n_kv >= seq_len:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< seq_len={seq_len} (need room for the cue token at the end).")
    logger.info(f'  mqar: vocab={vocab}, seq_len={seq_len}, n_kv={n_kv_list}')
    logger.info('  per-task defaults: ' + '; '.join(
        f"{t}: seq_len={s['seq_len']}, vocab={s['vocab']}"
        for t, s in SYNTHETIC_TASKS.items() if t != 'mqar'))
    # Robust env var parsing: ``parse_int_env`` logs a warning and falls
    # back to the default on a malformed value (e.g. ``MQAR_SEEDS=abc``
    # that would otherwise raise ``ValueError: invalid literal for
    # int()``), matching the robustness pattern already used for
    # BENCH_REPEATS / BENCH_LENGTHS in run_benchmark.py.
    n_seeds = parse_int_env('MQAR_SEEDS', 5, min_value=1, logger=logger)
    steps = parse_int_env('MQAR_STEPS', 200, min_value=1, logger=logger)
    # Fair primary comparison: every operator receives the same optimizer-step
    # budget by default. MQAR_SOFTMAX_STEPS remains available for an explicitly
    # labelled convergence/long-training sensitivity run, but no method is
    # advantaged in the default result table.
    softmax_steps = parse_int_env('MQAR_SOFTMAX_STEPS', steps, min_value=1,
                                  logger=logger)
    # Results-analysis controls (monitoring + bad cases). Defaults are
    # cheap enough for every run: 5 held-out validation points and 2
    # held-out analysis batches per seed, both on dedicated RNG streams.
    # Set MQAR_VAL_EVERY=0 / MQAR_ERROR_BATCHES=0 to disable.
    val_every = parse_int_env('MQAR_VAL_EVERY', max(1, steps // 4),
                              min_value=0, logger=logger)
    val_batches = parse_int_env('MQAR_VAL_BATCHES', 2, min_value=1,
                                logger=logger)
    val_batch = parse_int_env('MQAR_VAL_BATCH', 32, min_value=1,
                              logger=logger)
    error_batches = parse_int_env('MQAR_ERROR_BATCHES', 2, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('MQAR_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    logger.info(f'  n_seeds       : {n_seeds}')
    logger.info(f'  steps         : {steps}  (softmax: {softmax_steps})')
    logger.info(f'  val monitor   : every {val_every} steps '
                f'({val_batches}x{val_batch}; stream +{_VAL_MONITOR_SEED_OFFSET})')
    logger.info(f'  bad-case      : {error_batches}x{error_batch} held-out '
                f'samples per seed (stream +{_ERROR_ANALYSIS_SEED_OFFSET})')
    logger.info(f"  mamba_backend : {os.environ.get('MAMBA_BACKEND', 'reference')}")
    logger.info(f"  rwkv_backend  : {os.environ.get('RWKV_BACKEND', 'reference')}")
    logger.info(f"  rwkv7_binding : {RWKV7_UPSTREAM_BINDING['parameterization_init']['repo']}"
                f"@{RWKV7_UPSTREAM_BINDING['parameterization_init']['commit'][:12]}"
                f" ({RWKV7_UPSTREAM_BINDING['variant']})")

    # Pre-compute the (task, n_kv) sweep cells so every downstream count
    # (Bonferroni family size, log lines, result grouping) matches what
    # actually runs.
    sweep = []
    for task in task_list:
        spec = SYNTHETIC_TASKS[task]
        kv_values = n_kv_list if spec.get('n_kv_swept') \
            else [spec.get('default_n_kv', 1)]
        for n_kv in kv_values:
            sweep.append((task, n_kv, spec['seq_len'], spec['vocab']))

    # ------------------------------------------------------------------
    # Non-learned rule-based reference (baseline-ladder rung 0): the
    # simplest possible "model" is not a network at all but the exact
    # symbolic rule the synthetic task is built around. Evaluated once on
    # a dedicated stream (+8M), before any training, so it cannot perturb
    # the sweep; disabled with MQAR_RULE_BASELINE=0. See evaluate_rule_
    # baseline's docstring for why this is a ceiling reference, NOT the
    # comparison baseline for superiority claims.
    # ------------------------------------------------------------------
    rule_baselines: dict = OrderedDict()
    rule_baseline_enabled = parse_int_env(
        'MQAR_RULE_BASELINE', 1, min_value=0, logger=logger) != 0
    if rule_baseline_enabled:
        logger.info('\n' + '=' * 70)
        logger.info('Non-learned rule-based reference (task ceiling; '
                    'dedicated stream +8M)')
        logger.info('=' * 70)
        for task, n_kv, task_seq_len, task_vocab in sweep:
            rec = evaluate_rule_baseline(
                task, n_kv, task_seq_len, task_vocab, device)
            rule_baselines[(task, n_kv)] = rec
            logger.info(
                f"  [rule] {task:10s} n_kv={n_kv}: {rec['rule']:20s} "
                f"acc={rec['accuracy']:.4f} "
                f"(chance={rec['chance_acc']:.4f}, "
                f"n={rec['n_sequences']})")
        logger.info('=' * 70)

    # ------------------------------------------------------------------
    # Pre-training overfit sanity gate (before any multi-seed training):
    # for every (task cell, operator), train a FRESH model on ONE fixed
    # batch with regularization off and check it memorizes the batch
    # (train accuracy -> ~100%, loss -> ~0). This catches broken gradient
    # flow / loss wiring / label mapping cheaply before 5 seeds x N steps
    # of sweep time. Necessary, NOT sufficient (see the function
    # docstring). The gate RE-SEEDS internally and the sweep re-seeds per
    # op/seed, so enabling it does not change the experiment's numbers.
    # Controlled by env vars:
    #   MQAR_OVERFIT_CHECK=1|0  (default 1: run the gate)
    #   MQAR_OVERFIT_STRICT=1|0 (default 1: return non-zero if a cell fails)
    #   MQAR_OVERFIT_STEPS=N    (default 200 optimizer steps per cell)
    #   MQAR_OVERFIT_SEED=N     (default 2026; gate model + fixed batch)
    # ------------------------------------------------------------------
    overfit_check_enabled = parse_int_env(
        'MQAR_OVERFIT_CHECK', 1, min_value=0, logger=logger) != 0
    overfit_strict = parse_int_env(
        'MQAR_OVERFIT_STRICT', 1, min_value=0, logger=logger) != 0
    overfit_steps = parse_int_env(
        'MQAR_OVERFIT_STEPS', 200, min_value=10, logger=logger)
    overfit_seed = parse_int_env(
        'MQAR_OVERFIT_SEED', 2026, min_value=0, logger=logger)
    # Counts recorded in the run manifest even when the gate is disabled.
    n_gate_failed = 0
    # Cap the gate batch so a large MQAR_TRAIN_BATCH does not make the
    # memorization target unnecessarily hard / slow: "a fixed batch of a few
    # dozen examples" is the probe; 64 sequences still exercises every
    # code path of the real training batch.
    gate_train_batch = min(
        parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1, logger=logger), 64)
    overfit_reports = []
    if overfit_check_enabled:
        logger.info('\n' + '=' * 70)
        logger.info('Pre-training overfit sanity gate (fixed batch, '
                    'weight_decay=0, no dropout / augmentation)')
        logger.info(f'  steps={overfit_steps} seed={overfit_seed} '
                    f'batch={gate_train_batch} lr=3e-3  '
                    f'strict={overfit_strict}')
        logger.info('=' * 70)
        for task, n_kv, task_seq_len, task_vocab in sweep:
            for op in ops_list:
                # Memoryless controls cannot memorize a fixed batch BY
                # CONSTRUCTION (no cross-token pathway; the cue->value map is
                # not a function of the cue alone), so the gate is skipped
                # for them — recorded explicitly instead of counted as a
                # failure. See _MEMORYLESS_OPS.
                if op in _MEMORYLESS_OPS:
                    overfit_reports.append({
                        'passed': None,
                        'skipped': True,
                        'op': op, 'task': task, 'n_kv': n_kv,
                        'steps': overfit_steps, 'seed': overfit_seed,
                        'lr': 3e-3, 'train_batch': gate_train_batch,
                        'note': ('Memoryless control: the fixed-batch '
                                 'memorization criterion does not apply '
                                 '(no inter-token pathway). Skipped, not '
                                 'passed.'),
                    })
                    logger.info(
                        f'  [overfit] {op:8s} task={task} n_kv={n_kv}: '
                        'SKIP (memoryless control; gate not applicable)')
                    continue
                try:
                    rep = run_overfit_sanity_check(
                        op, task=task, seq_len=task_seq_len, n_kv=n_kv,
                        vocab=task_vocab, steps=overfit_steps,
                        seed=overfit_seed, device=device,
                        train_batch=gate_train_batch)
                except Exception as e:
                    # A gate crash is itself a red flag (e.g. non-finite
                    # loss); record it as a failed report rather than
                    # aborting with a traceback, so the side file keeps a
                    # complete record of every (cell, op) probe.
                    logger.error(f'  [overfit] {op:8s} task={task} '
                                 f'n_kv={n_kv}: ERROR {e}')
                    rep = {
                        'passed': False, 'op': op, 'task': task,
                        'n_kv': n_kv, 'steps': overfit_steps,
                        'seed': overfit_seed,
                        'train_batch': gate_train_batch,
                        'lr': 3e-3,
                        'error': str(e),
                        'note': ('Gate crashed; treat as a failed sanity '
                                 'check and inspect before trusting the '
                                 'sweep.'),
                    }
                overfit_reports.append(rep)
                logger.info(
                    f'  [overfit] {op:8s} task={task} n_kv={n_kv}: '
                    f"{'PASS' if rep.get('passed') else 'FAIL'} "
                    f"loss {rep.get('initial_loss', float('nan')):.3f} -> "
                    f"{rep.get('final_loss', float('nan')):.4f}, "
                    f"train_acc -> {rep.get('final_train_acc', float('nan')):.3f}")
        # Dedicated overfit-gate artifact (full per-step curves + batch
        # fingerprints), written BEFORE the strict check returns so a failed
        # gate leaves its evidence on disk. This is an experiment-tracking
        # record in its own right: if a later run's sweep numbers shift, this
        # file shows whether the probe model still memorized its fixed batch
        # under the SAME seed.
        os.makedirs('results', exist_ok=True)
        gate_payload = {
            'kind': 'exp4_overfit_gate',
            'schema_version': CURRENT_SCHEMA_VERSION,
            'metadata': {
                'schema_version': CURRENT_SCHEMA_VERSION,
                'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
                'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
                'rwkv_backend': os.environ.get('RWKV_BACKEND', 'reference'),
                'rwkv7_upstream_binding': RWKV7_UPSTREAM_BINDING,
                'provenance': capture_provenance(),
                'settings': {
                    'enabled': True,
                    'strict': bool(overfit_strict),
                    'steps': overfit_steps,
                    'seed': overfit_seed,
                    'train_batch': gate_train_batch,
                    'lr': 3e-3,
                    'weight_decay': 0.0,
                    'acc_threshold': 0.99,
                    'loss_ratio_threshold': 0.5,
                    'batch_generator_seed_offset': _OVERFIT_BATCH_SEED_OFFSET,
                },
                'data': {
                    'kind': 'synthetic',
                    'task_specs': {
                        t: {'seq_len': SYNTHETIC_TASKS[t]['seq_len'],
                            'vocab': SYNTHETIC_TASKS[t]['vocab']}
                        for t in task_list
                    },
                },
                'note': ('Necessary but NOT sufficient gradient-flow check. '
                         'Full training curves for each (task, op) probe are '
                         'in the reports below.'),
            },
            'reports': overfit_reports,
        }
        try:
            write_json_atomic(gate_payload, 'results/exp4_overfit_gate.json',
                              indent=2, allow_nan=False)
        except ValueError as e:
            logger.error(f'non-finite value in overfit gate payload; '
                         f'sanitizing to null: {e}')
            write_json_atomic(sanitize_for_json(gate_payload),
                              'results/exp4_overfit_gate.json',
                              indent=2, allow_nan=False)
        logger.info('Saved: results/exp4_overfit_gate.json')
        # Failed = checked cells (not skipped memoryless controls) that did
        # not pass the memorization criterion.
        n_gate_checked = sum(1 for r in overfit_reports
                             if not r.get('skipped'))
        n_gate_skipped = sum(1 for r in overfit_reports
                             if r.get('skipped'))
        n_gate_failed = sum(1 for r in overfit_reports
                            if not r.get('passed') and not r.get('skipped'))
        if n_gate_failed:
            logger.error(
                f'  {n_gate_failed}/{len(overfit_reports)} overfit gate '
                'cells FAILED: a fresh model could not memorize a single '
                'fixed batch of at most '
                f'{gate_train_batch} sequences in {overfit_steps} steps. '
                'This usually means gradients are not flowing, the loss is '
                'mis-wired, or labels are wrong — do NOT trust the sweep '
                'results. Debug with a single cell, or bypass (NOT '
                'recommended) with MQAR_OVERFIT_CHECK=0.')
            if overfit_strict:
                logger.error(
                    '  MQAR_OVERFIT_STRICT=1: refusing to start the full '
                    'multi-seed sweep on a broken training recipe. Set '
                    'MQAR_OVERFIT_STRICT=0 to record the failure and '
                    'continue (diagnostic-only mode).')
                return 1
        else:
            logger.info(
                f'  all {n_gate_checked} checked overfit gate cells PASSED'
                + (f' ({n_gate_skipped} memoryless control cell(s) skipped)'
                   if n_gate_skipped else ''))

    # ------------------------------------------------------------------
    # Tuned-baseline protocol (runs BEFORE the sweep): select each baseline
    # operator's lr on the validation stream, then use it for that op's
    # rows in the sweep. The selection runs use the shared train_multi_seed
    # path, so "tuned" means "same recipe, more lr candidates", not a
    # bespoke pipeline. Disabled with MQAR_BASELINE_TUNE=0 (historical
    # behavior: fixed lr=3e-3 for every op).
    # ------------------------------------------------------------------
    tuned_lrs = {}
    baseline_tuning = {}
    if baseline_tune_enabled and baseline_ops:
        tune_seeds = parse_int_env('MQAR_BASELINE_TUNE_SEEDS',
                                   min(n_seeds, 3),
                                   min_value=1, logger=logger)
        logger.info('\n' + '=' * 70)
        logger.info('Tuned-baseline protocol: selecting baseline lr on the '
                    f'validation stream (candidates={baseline_lrs}, '
                    f'seeds={tune_seeds})')
        logger.info('=' * 70)
        for op in baseline_ops:
            best_lr, rec = tune_baseline_lr(
                op, baseline_lrs, sweep, steps=steps,
                softmax_steps=softmax_steps, n_seeds=tune_seeds,
                device=device, val_every=val_every, val_batches=val_batches,
                val_batch=val_batch)
            tuned_lrs[op] = best_lr
            rec['grid_size'] = len(baseline_lrs)
            baseline_tuning[op] = rec
            logger.info(f'  [tune] {op}: selected lr={best_lr:g}')
        logger.info('=' * 70)

    all_results = []
    for task, n_kv, task_seq_len, task_vocab in sweep:
        spec = SYNTHETIC_TASKS[task]
        chance = 1.0 / task_vocab
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  task = {task} (n_kv={n_kv}, seq_len={task_seq_len}, '
                    f'vocab={task_vocab}) — {spec["description"]}')
        logger.info(f'  chance accuracy = {chance:.4f}')
        logger.info(f'{"=" * 70}')
        for op in ops_list:
            logger.info(f'\nTraining {op} (task={task}, n_kv={n_kv}, '
                        f'{n_seeds} seeds)...')
            # Per-operator try/except so ONE failing operator (all seeds
            # diverged / OOMed) does not crash the whole experiment and
            # lose the other operators' results. The error is logged and
            # recorded as a stub result so the JSON file is always written
            # and downstream figure generation can skip the missing op
            # gracefully (mirrors run_ablation.py's per-ratio try/except).
            try:
                r = train_multi_seed(op, n_seeds=n_seeds, steps=steps,
                                     softmax_steps=softmax_steps, device=device,
                                     n_kv=n_kv, vocab=task_vocab,
                                     seq_len=task_seq_len, task=task,
                                     lr=tuned_lrs.get(op, 3e-3),
                                     val_every=val_every,
                                     val_batches=val_batches,
                                     val_batch=val_batch,
                                     error_batches=error_batches,
                                     error_batch=error_batch)
                # Attach the non-learned rule reference for this cell next
                # to the row it contextualizes (see evaluate_rule_baseline:
                # ceiling reference, not the tuned comparison baseline).
                r['rule_baseline'] = rule_baselines.get((task, n_kv))
                all_results.append(r)
                # ci95_acc may be None when only one seed survived (see
                # train_multi_seed). Render as 'n/a' instead of crashing on
                # ``f'{None:.4f}'``.
                ci_str = f"{r['ci95_acc']:.4f}" if r['ci95_acc'] is not None else 'n/a'
                logger.info(f"  -> mean_acc={r['mean_acc']:.4f} +/- {ci_str} "
                            f"(std={r['std_acc']:.4f}, t_vs_chance={_fmt_tstat(r['t_stat_vs_chance'], width=0, prec=2)})")
                # Curve verdicts: over/underfitting is a property of the
                # training CURVE, reported per op so a good mean accuracy
                # from an overfit best-epoch cannot go unnoticed.
                cd = r.get('curve_diagnosis_summary') or {}
                ea = r.get('error_analysis')
                logger.info(
                    f"  -> curves: {cd.get('verdicts', {})}; "
                    f"bad-case error rate: "
                    f"{(ea or {}).get('error_rate') if ea else 'n/a'}"
                    + (f" ({ {c: v['count'] for c, v in (ea or {}).get('pooled_categories', {}).items() if v['count']} })"
                       if ea else ''))
            except Exception as e:
                import traceback as _tb
                logger.error(f"  op '{op}' FAILED: {e}")
                _tb.print_exc()
                # Error stub MUST include every key that success rows
                # carry, set to None/empty/zero as appropriate, so
                # downstream JSON consumers (make_figures.py, pandas
                # DataFrames, etc.) iterating over fields do not KeyError
                # on error rows. The success-row schema is defined by
                # train_multi_seed's return dict — mirror it exactly.
                all_results.append({
                    'op': op,
                    'task': task,
                    'n_kv': n_kv,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_loss': None,
                    'std_loss': None,
                    'ci95_loss': None,
                    'chance_acc': 1.0 / task_vocab,
                    't_stat_vs_chance': None,
                    'lr': tuned_lrs.get(op, 3e-3),
                    'mean_train_time_s': None,
                    'mean_eval_ms_per_sample': None,
                    'curve_diagnosis_summary': None,
                    'mean_generalization_gap': None,
                    'error_analysis': None,
                    'mean_final_metrics': None,
                    'pareto': None,
                    'rule_baseline': rule_baselines.get((task, n_kv)),
                    'per_seed': [],
                })

    # Summary table (grouped by task + n_kv). The lr column makes the
    # tuned-baseline protocol visible in the headline table: baseline ops
    # may show a non-default lr selected on the validation stream.
    print('\n' + '=' * 99)
    print(f"{'task':>10} | {'n_kv':>4} | {'op':>10} | {'lr':>7} | {'chance':>7} | {'mean_acc':>10} | "
          f"{'+/- CI95':>10} | {'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 99)
    for r in all_results:
        # Error rows have None for all numeric fields; render them as
        # dashes instead of crashing on ``f'{None:.4f}'``.
        if 'error' in r:
            err_msg = (r.get('error') or '')[:40]
            print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
                  f"{'-':>7} | {'-':>7} | {'-':>10} | {'-':>10} | {'-':>8} | {'-':>12} | "
                  f"{'-':>10}   ERROR: {err_msg}")
            continue
        # ci95_acc may be None when only one seed survived; render 'n/a'.
        if r['ci95_acc'] is not None:
            ci_str = f"{r['ci95_acc']:>10.4f}"
        else:
            ci_str = f"{'n/a':>10}"
        lr_str = f"{r.get('lr', 3e-3):>7.1e}"
        print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
              f"{lr_str} | {r['chance_acc']:>7.4f} | {r['mean_acc']:>10.4f} | "
              f"{ci_str} | {r['std_acc']:>8.4f} | "
              f"{_fmt_tstat(r['t_stat_vs_chance'], width=12, prec=2)} | {r['mean_loss']:>10.4f}")
    # Fairness annotation: the softmax baseline is trained for
    # ``softmax_steps`` steps while the other ops get ``steps`` steps. This
    # is intentional (softmax needs more steps to converge above chance —
    # see the comment block at the top of this file), but a reader of the
    # summary table alone has no way to know that. Print the asymmetry
    # explicitly so cross-op comparisons are not misread as "softmax is
    # unfairly advantaged" or "softmax is just better".
    if softmax_steps != steps:
        print(f"\nNOTE: softmax trained for {softmax_steps} steps; "
              f"other ops trained for {steps} steps. See MQAR_SOFTMAX_STEPS "
              f"env var / README 'Fairness notes' section for rationale.")

    # Bonferroni correction + ``conclusions_valid`` for Exp 4, mirroring the
    # logic already present in run_ablation.py::main for Exp 5. The README
    # documents Exp 4's JSON schema as including ``conclusions_valid`` and a
    # Bonferroni-corrected t-test.
    #
    # Logic:
    #   1. Compute the number of one-sample t-tests vs chance we are running
    #      (4 ops * len(n_kv_list) = 4 by default).
    #   2. Compute the Bonferroni-corrected alpha (0.05 / n_tests) and the
    #      corresponding t-critical value (scipy if available, else None).
    #   3. For each result, set ``significant_bonferroni`` and
    #      ``t_crit_bonferroni`` fields based on the one-sided
    #      comparison ``t_stat > crit``.
    #   4. Compute an experiment-level ``conclusions_valid`` flag combining
    #      seed count, minimum surviving seeds, presence of any significant
    #      result, and the fraction of near-chance results.
    #   5. Attach ``conclusions_valid`` to every result record (matching
    #      run_ablation.py's convention) so downstream consumers can check
    #      it per-record without recomputing.
    n_tests = len(ops_list) * len(sweep)
    # The t-test is one-sided (H1: mean_acc > chance). For a one-sided
    # Bonferroni correction, the per-test alpha is 0.05 / n_tests to
    # maintain the family-wise error rate at 0.05. Using 0.025 / n_tests
    # would make the test twice as conservative as intended.
    alpha_corrected = 0.05 / n_tests
    # ``_bonferroni_crit_q`` is a module-level function with an exact
    # beta-CDF/bisection fallback when scipy is unavailable. We just detect
    # scipy availability for the log line below.
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'\n  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=exact-beta-bisection)')
    for r in all_results:
        # Skip error rows: they have t_stat_vs_chance=None already.
        if 'error' in r:
            r['t_crit_bonferroni'] = None
            r['significant_bonferroni'] = False
            continue
        t_stat = r.get('t_stat_vs_chance')
        n_ok = r.get('n_seeds_ok', 0)
        if t_stat is not None and n_ok >= 2:
            crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
            r['t_crit_bonferroni'] = crit
            # ``crit`` is None when scipy is unavailable — guard the
            # comparison to avoid TypeError (mirrors run_ablation.py).
            #
            # The test is ONE-SIDED (``t_stat > crit``): the research question
            # is "does this op learn the task ABOVE chance", which is
            # directional. A below-chance result indicates the model is
            # systematically wrong (e.g. a sign error, a reversed label, or
            # pure noise that happens to anti-correlate), NOT that the op
            # "works". The Bonferroni-corrected critical value
            # ``_bonferroni_crit_q`` is already the one-sided upper-tail
            # quantile, so the one-sided comparison is the correct use of
            # that quantile.
            r['significant_bonferroni'] = (
                crit is not None and t_stat > crit
            )
        else:
            r['t_crit_bonferroni'] = None
            r['significant_bonferroni'] = False

    # ------------------------------------------------------------------
    # Paired comparison vs the tuned reference baseline.
    #
    # The vs-chance test above only says "the op learned the task at all".
    # It does NOT support "the complex op is better than a properly tuned
    # simple one" — with enough seeds, a tiny difference can be
    # significant, and the complex model could still lose to a well-tuned
    # baseline. So every non-baseline op is additionally tested against
    # the reference baseline on the SAME (task, n_kv) cell with a
    # SEED-PAIRED one-sided t-test (same seed list => paired). The
    # family of comparisons is Bonferroni-corrected.
    #
    # Flags per row:
    #   is_baseline          op is a baseline rung (no pairwise test)
    #   vs_tuned_baseline    {reference_op, n_pairs, mean_diff, t_stat,
    #                         t_crit_bonferroni, significant, verdict}
    #   beats_tuned_baseline bool (True only when the paired test is
    #                        significant in the positive direction);
    #                        None when not testable (baseline row itself,
    #                        missing reference row, < 2 paired seeds).
    # The scientific claim "complex op X is better" requires
    # beats_tuned_baseline=True; a row significant vs chance but not vs
    # the tuned baseline is only evidence that the task is learnable.
    # ------------------------------------------------------------------
    for r in all_results:
        r['is_baseline'] = r.get('op') in baseline_ops
        r['vs_tuned_baseline'] = None
        r['beats_tuned_baseline'] = None
        # Attach the pre-registered mechanism hypothesis for this
        # (task, operator) cell, so every row carries what it was TESTING
        # (and what would falsify it) next to the number it produced.
        th = TASK_HYPOTHESES.get(r.get('task', 'mqar'))
        oh = OPERATOR_HYPOTHESES.get(r.get('op'))
        r['hypothesis'] = ({'task': th, 'operator': oh}
                           if (th or oh) else None)
    baseline_pairwise = {
        'reference_op': baseline_ref,
        'baseline_ops': list(baseline_ops),
        'n_tests': 0,
        'alpha_bonferroni': None,
        'comparisons': {},
        'note': ('Seed-paired one-sided t-test of each non-baseline op vs '
                 'the reference baseline on the same (task, n_kv) cell '
                 '(H1: mean per-seed accuracy difference > 0), '
                 'Bonferroni-corrected over all comparisons. '
                 'beats_tuned_baseline=True is required before claiming '
                 'a complex operator is better than the simple baseline.'),
    }
    if baseline_ref is not None:
        ref_by_cell = {
            (r.get('task', 'mqar'), r['n_kv']): r
            for r in all_results
            if r.get('op') == baseline_ref and 'error' not in r
            and r.get('mean_acc') is not None
        }
        comparisons = []
        for r in all_results:
            if 'error' in r or r.get('op') == baseline_ref:
                continue
            ref = ref_by_cell.get((r.get('task', 'mqar'), r['n_kv']))
            if ref is None:
                r['vs_tuned_baseline'] = {
                    'reference_op': baseline_ref,
                    'n_pairs': 0,
                    'mean_diff': None,
                    't_stat': None,
                    'significant': False,
                    'verdict': 'reference_row_unavailable',
                }
                continue
            cand = {s['seed']: s['final_acc']
                    for s in r.get('per_seed', [])
                    if 'error' not in s and s.get('final_acc') is not None}
            base = {s['seed']: s['final_acc']
                    for s in ref.get('per_seed', [])
                    if 'error' not in s and s.get('final_acc') is not None}
            common = sorted(set(cand) & set(base))
            if len(common) < 2:
                r['vs_tuned_baseline'] = {
                    'reference_op': baseline_ref,
                    'n_pairs': len(common),
                    'mean_diff': None,
                    't_stat': None,
                    'significant': False,
                    'verdict': 'insufficient_paired_seeds',
                }
                continue
            stats = _paired_one_sided_t([cand[s] - base[s]
                                         for s in common])
            stats['reference_op'] = baseline_ref
            comparisons.append((r, stats))
        n_pair_tests = len(comparisons)
        # One-sided family-wise alpha 0.05 over the comparison family: the
        # per-test alpha is 0.05 / n_pair_tests. The critical value is then
        # the one-sided upper-tail quantile at that corrected alpha with the
        # COMPARISON's own dof (n_pairs - 1) — mirroring
        # ``run_ablation._annotate_contrast_significance``.
        alpha_pair = (0.05 / n_pair_tests) if n_pair_tests else None
        for r, stats in comparisons:
            t = stats.get('t_stat')
            n_pairs = stats.get('n_pairs', 0)
            crit_pair = (
                _bonferroni_crit_q(n_pairs, alpha=alpha_pair)
                if (alpha_pair is not None and n_pairs >= 2) else None)
            sig = False
            if stats.get('mean_diff') is not None and stats['mean_diff'] > 0:
                if t is not None and crit_pair is not None and t > crit_pair:
                    sig = True
                elif (t is None and stats.get('all_positive')
                      and n_pairs >= 2):
                    # Zero-variance unanimous improvement (every seed
                    # strictly better): the exact one-sided sign-flip
                    # p-value of the all-positive pattern is 2^{-(n-1)}; it
                    # counts as significant only when that clears the
                    # Bonferroni-corrected alpha (recorded via t_stat=None).
                    exact_p = 2.0 ** (-(n_pairs - 1))
                    sig = bool(alpha_pair is not None
                               and exact_p <= alpha_pair)
            stats['t_crit_bonferroni'] = crit_pair
            stats['significant'] = bool(sig)
            stats['verdict'] = ('beats_tuned_baseline' if sig
                                else 'not_significantly_better')
            r['vs_tuned_baseline'] = stats
            r['beats_tuned_baseline'] = bool(sig)
            baseline_pairwise['comparisons'][
                f"{r.get('task', 'mqar')}/n_kv={r['n_kv']}/{r['op']}"] = stats
        baseline_pairwise['n_tests'] = n_pair_tests
        baseline_pairwise['alpha_bonferroni'] = (
            (0.05 / n_pair_tests) if n_pair_tests else None)
        if n_pair_tests:
            n_beat = sum(1 for r in all_results
                         if r.get('beats_tuned_baseline'))
            logger.info(f'\n  paired vs tuned baseline ({baseline_ref}): '
                        f'{n_beat}/{n_pair_tests} comparisons significant '
                        f'(Bonferroni alpha='
                        f'{baseline_pairwise["alpha_bonferroni"]:.4f})')
            for key, stats in baseline_pairwise['comparisons'].items():
                logger.info(
                    f"    {key}: d_acc={stats['mean_diff']:+.4f} "
                    f"(n={stats['n_pairs']}, "
                    f"t={_fmt_tstat(stats['t_stat'], width=0, prec=2)}) "
                    f"-> {stats['verdict']}")
            if n_beat == 0:
                logger.warning(
                    '  WARNING: no operator significantly beats the TUNED '
                    'baseline. Complex-operator superiority is NOT '
                    'supported on this cell; treat any raw ranking as '
                    'inconclusive (the baseline may simply be strong '
                    'enough here).')
        # Printed baseline-comparison table (stdout, not just logs).
        print('\nPaired vs tuned baseline (one-sided, Bonferroni-corrected):')
        print(f"{'task':>10} | {'n_kv':>4} | {'op':>10} | {'ref':>8} | "
              f"{'d_acc':>8} | {'n':>3} | {'t':>7} | verdict")
        for r in all_results:
            st = r.get('vs_tuned_baseline')
            if 'error' in r or not st:
                continue
            d = st.get('mean_diff')
            d_str = f"{d:+.4f}" if d is not None else 'n/a'
            print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | "
                  f"{r['op']:>10} | {st.get('reference_op') or '-':>8} | "
                  f"{d_str:>8} | {st.get('n_pairs', 0):>3} | "
                  f"{_fmt_tstat(st.get('t_stat'), width=7, prec=2)} | "
                  f"{'YES' if st.get('significant') else 'no'}"
                  f" ({st.get('verdict')})")

    # ------------------------------------------------------------------
    # Machine-readable verdicts for the pre-registered hypotheses.
    # Each row gets ``hypothesis_checks`` (per-rule evidence + status) and
    # a compact ``hypothesis_status``; the metadata records the aggregate
    # counts. The rules are fixed in code BEFORE the numbers are read.
    # ------------------------------------------------------------------
    hypothesis_verdicts, hypothesis_counts = _hypothesis_verdicts(
        all_results, baseline_ops, baseline_ref)
    logger.info(
        '  hypothesis verdicts: '
        + ', '.join(f'{k}={v}' for k, v in hypothesis_counts.items())
        + ' (row counts)')
    for name, rec in hypothesis_verdicts.items():
        logger.info(f'    [hypothesis] {name}: {rec["status"]}'
                    + (' (FALSIFIED)' if rec.get('falsified') else ''))

    # ------------------------------------------------------------------
    # Seed-paired ALL-PAIRS operator comparisons (Holm-corrected).
    #
    # Completes the statistical claim ladder:
    #   1. significant_bonferroni          -> the op learned the task;
    #   2. beats_tuned_baseline            -> it beats a tuned simple model;
    #   3. pairwise_significant_wins       -> it beats a SPECIFIC complex
    #      operator on the same seeds, robust to multiplicity;
    #   4. pairwise_equivalent_to          -> the pre-specified-margin TOST
    #      says it is practically the SAME as that operator (a positive
    #      equivalence claim; non-significance alone never licenses it).
    # The CI-overlap check below is a heuristic companion: overlapping
    # CIs do not imply "no difference", so only the paired tests here
    # support an A-vs-B claim.
    # ------------------------------------------------------------------
    # Pre-specified SESOI for the TOST; MQAR_EQUIV_MARGIN=0 disables
    # equivalence testing. The margin is consumed BEFORE the reported
    # statistics are computed and is recorded in the metadata.
    equivalence_margin = _parse_equivalence_margin(
        'MQAR_EQUIV_MARGIN', PAIRWISE_EQUIVALENCE_MARGIN)
    op_pairwise = _pairwise_op_comparisons(
        all_results,
        alpha=float(os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')),
        equivalence_margin=equivalence_margin)
    _print_pairwise_report(op_pairwise)
    if op_pairwise['n_tests']:
        logger.info(
            f'\n  seed-paired op-vs-op comparisons: '
            f"{op_pairwise['n_significant']}/{op_pairwise['n_tests']} "
            f"significant at Holm alpha={op_pairwise['alpha']} "
            f"({op_pairwise['n_skipped']} pair(s) skipped: <2 common seeds)")
        if equivalence_margin is not None:
            logger.info(
                f"  TOST equivalence (pre-specified margin="
                f"{equivalence_margin:g}): "
                f"{op_pairwise['n_equivalent']}/{op_pairwise['n_tests']} "
                f"pair(s) practically equivalent at Holm "
                f"alpha={op_pairwise['alpha']}")
        else:
            logger.info('  TOST equivalence: disabled (MQAR_EQUIV_MARGIN=0)')

    # Experiment-level statistical validity summary. Mirrors run_ablation.py
    # but with the appropriate thresholds for the 4-op MQAR sweep.
    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    # Include error rows: they carry n_seeds_ok=0, so a fully-failed op
    # forces min_seeds_ok=0 and thus conclusions_valid=False (mirrors the
    # run_mechanism_ablation.py convention of taking the min over ALL rows).
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results),
                       default=0)
    # Split "near chance" (slightly above chance, not yet significant) from
    # "far below chance" (training diverged / recipe broken). Chance is
    # PER-ROW (each task has its own vocab), so use r['chance_acc']:
    #   * near_chance:   0.5*chance < acc < 1.5*chance  (statistically ~chance)
    #   * far_below_chance: acc <= 0.5*chance           (training/recipe broken)
    # Any far_below row forces conclusions_valid=False regardless of other
    # counts, since it indicates a real problem rather than weak power.
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and r.get('chance_acc') is not None
                   and 0.5 * r['chance_acc'] < r['mean_acc']
                   < 1.5 * r['chance_acc']]
    far_below = [r for r in all_results
                 if 'error' not in r
                 and r.get('mean_acc') is not None
                 and r.get('chance_acc') is not None
                 and r['mean_acc'] <= 0.5 * r['chance_acc']]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0
                         and len(near_chance) < len(all_results) // 2
                         and len(far_below) == 0)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ops with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
    logger.info(f'  ops near chance (0.5x-1.5x): {len(near_chance)}/{len(all_results)}')
    logger.info(f'  ops far below chance (<=0.5x): {len(far_below)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    # Baseline-ladder verdict: how many complex ops actually clear the
    # tuned reference baseline (the claim-supporting test), separate from
    # the vs-chance validity above.
    n_beat_baseline = sum(1 for r in all_results
                          if r.get('beats_tuned_baseline'))
    n_complex_rows = sum(1 for r in all_results
                         if 'error' not in r and not r.get('is_baseline'))
    if baseline_pairwise.get('n_tests'):
        logger.info(
            f'  complex ops beating tuned baseline '
            f'({baseline_pairwise["reference_op"]}): {n_beat_baseline}/'
            f'{n_complex_rows} rows '
            f'({baseline_pairwise["n_tests"]} Bonferroni-corrected paired '
            f'tests)')
    else:
        logger.info('  paired vs tuned baseline: not run '
                    '(no reference baseline row)')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: The MQAR results do NOT support strong structural\n'
            '  conclusions. Either the seed count is too low (<5), no op\n'
            '  reaches Bonferroni significance, or most accuracies are near\n'
            '  chance. Treat the ranking as exploratory, not confirmatory.\n'
            '  To improve power: increase MQAR_SEEDS (>=7), increase MQAR_STEPS,\n'
            '  or use a simpler task where the signal is stronger.')
    logger.info('=' * 70)
    # Attach the validity flag to every result record so downstream
    # consumers (make_figures, reports) can check it without recomputing.
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds

    # ------------------------------------------------------------------
    # Efficiency-vs-accuracy annotation (per task/n_kv group).
    #
    # "Report the gain WITH its cost": for every sweep cell, mark each op
    # as Pareto-optimal or dominated on (mean_acc maximise; parameters,
    # wall-clock train time and per-sample inference latency minimise).
    # Accuracy gains that come at a disproportionate cost are visible
    # directly in the record instead of requiring a cross-reference to the
    # latency/cache experiments.
    #
    # The cross-op claim check answers "is the winner actually better?":
    # when the best op's 95% CI overlaps another op's CI, the ranking
    # between them is not statistically supported at this seed count, and
    # the record says so explicitly (microscopic, non-significant gains
    # must not be over-claimed).
    # ------------------------------------------------------------------
    groups: dict = OrderedDict()
    for r in all_results:
        groups.setdefault((r.get('task', 'mqar'), r['n_kv']), []).append(r)

    pareto_reports = {}
    cross_op_checks = {}
    for (task_name, n_kv_key), rows in groups.items():
        live = [r for r in rows if 'error' not in r
                and r.get('mean_acc') is not None
                and r.get('mean_train_time_s') is not None
                and r.get('n_params') is not None]
        if live:
            rep = annotate_rows_with_pareto(
                live, accuracy_key='mean_acc',
                cost_keys=('mean_train_time_s', 'n_params',
                           'mean_eval_ms_per_sample'),
                name_key='op')
            pareto_reports[f'{task_name}/n_kv={n_kv_key}'] = {
                'frontier': rep['frontier'],
                'dominated_by': rep['dominated_by'],
                'cost_keys': rep['cost_keys'],
                'points': {
                    op: {'accuracy': p['accuracy'], **p['costs'],
                         'acc_per_train_second': p['acc_per_train_second'],
                         'acc_per_1k_params': p['acc_per_1k_params'],
                         'acc_per_eval_ms': p.get('acc_per_eval_ms')}
                    for op, p in rep['points'].items()
                },
                'skipped': rep['skipped'],
            }
            best = max(live, key=lambda r: r['mean_acc'])
            overlaps = []
            for r in live:
                if r is best:
                    r['ci95_overlaps_best'] = False
                    continue
                ci_b, ci_r = best.get('ci95_acc'), r.get('ci95_acc')
                if ci_b is None or ci_r is None:
                    r['ci95_overlaps_best'] = None
                    continue
                overlap = abs(best['mean_acc'] - r['mean_acc']) <= (ci_b + ci_r)
                r['ci95_overlaps_best'] = bool(overlap)
                if overlap:
                    overlaps.append(r['op'])
            best['ci95_overlaps_best'] = False
            # Seed-paired, Holm-corrected standing of the best op (from
            # _pairwise_op_comparisons, computed above). This — not the CI
            # overlap list — is what may be claimed.
            best_sig_wins = [
                op for op in best.get('pairwise_significant_wins', [])
                if any(r.get('op') == op for r in live)]
            best_sig_wins = sorted(best_sig_wins, key=str)
            if best_sig_wins:
                paired_claim = (
                    f"Seed-paired Holm-corrected tests support {best['op']} "
                    f"over: {', '.join(best_sig_wins)}.")
            else:
                paired_claim = (
                    f"No seed-paired Holm-corrected test separates "
                    f"{best['op']} from any other op at this seed count.")
            # Positive equivalence standing (only when TOST is enabled):
            # ops the best op is PRACTICALLY THE SAME as, within the
            # pre-specified margin. This is the allowed counterpart of
            # "no significant difference".
            best_equiv_to = [
                op for op in best.get('pairwise_equivalent_to', [])
                if any(r.get('op') == op for r in live)]
            best_equiv_to = sorted(best_equiv_to, key=str)
            equiv_claim = ''
            if equivalence_margin is not None:
                if best_equiv_to:
                    equiv_claim = (
                        f"Within the pre-specified margin "
                        f"{equivalence_margin:g}, the TOST declares it "
                        f"practically equivalent to: "
                        f"{', '.join(best_equiv_to)}.")
                else:
                    equiv_claim = (
                        'No op is declared practically equivalent to it at '
                        'the pre-specified margin.')
            cross_op_checks[f'{task_name}/n_kv={n_kv_key}'] = {
                'best_op': best['op'],
                'best_mean_acc': best['mean_acc'],
                'best_ci95_acc': best.get('ci95_acc'),
                'ops_with_overlapping_ci95': overlaps,
                'best_op_pairwise_significant_wins': best_sig_wins,
                'best_op_practically_equivalent_to': (
                    best_equiv_to if equivalence_margin is not None
                    else None),
                'paired_test_note': (
                    'CI overlap is a continuity heuristic, not a test: '
                    'overlapping CIs do not imply a non-significant '
                    'difference. The authoritative pairwise evidence is '
                    'metadata.op_pairwise_comparisons (seed-paired '
                    'permutation tests AND pre-specified-margin TOST, '
                    'Holm-corrected per cell).'),
                'claim': (
                    f"{best['op']} has the highest mean accuracy. "
                    f"{paired_claim} "
                    + (f"Its 95% CI overlaps {overlaps}."
                       if overlaps else
                       "Its 95% CI is disjoint from every other op.")
                    + (f" {equiv_claim}" if equiv_claim else '')),
            }
            if not best_sig_wins:
                logger.warning(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: best='
                    f"{best['op']} (acc={best['mean_acc']:.4f}) has no "
                    f"significant seed-paired win (Holm) — treat the "
                    f"ranking as inconclusive.")
            dominated = rep['dominated_by']
            if dominated:
                logger.info(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: Pareto '
                    f'dominated: '
                    + ', '.join(f'{a}<-{b}' for a, b in dominated.items()))
            else:
                logger.info(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: all live '
                    f'ops on the accuracy/cost Pareto frontier')

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a divergent seed slipped
    # past the NaN guard and the per_seed filter, Python's default
    # json.dump would emit literal ``NaN``/``Infinity`` tokens, which are
    # INVALID JSON per RFC 8259 and cause strict parsers (js, jq, pandas
    # with ``orient='records'``) to reject the whole file. With
    # allow_nan=False the call raises ValueError instead — surfacing the
    # corruption loudly rather than shipping a broken file.
    #
    # CRITICAL: serialize to a STRING first (json.dumps), then write the
    # string to the file, so a mid-write ValueError (NaN) cannot leave a
    # PARTIAL JSON document behind. Serializing to a string first
    # guarantees atomicity: either the complete JSON is written or nothing
    # is.
    # The payload is a single top-level object
    # ``{"metadata": ..., "results": [...]}``; ``make_figures.load`` also
    # accepts the legacy bare-array format. ``write_json_atomic`` below
    # writes the file atomically (temp + fsync + os.replace).
    # ``write_json_atomic`` writes atomically (temp file + fsync +
    # os.replace) so a process kill or disk-full mid-write leaves the
    # target file as the OLD version (or absent) rather than a truncated
    # partial JSON document — see kaggle_setup.write_json_atomic's
    # docstring for the full rationale. It serializes to a string first,
    # raising ValueError on any non-serializable value BEFORE touching the
    # filesystem.
    #
    # Output split: the canonical MQAR rows keep the historical
    # results/exp4_mqar.json file (name, schema, and figure consumers
    # unchanged); rows from the additional probes go to a separate
    # results/exp4b_quality_tasks.json so the two result families never
    # blur together. When MQAR is not in the task list, the historical
    # file is left untouched rather than overwritten with an empty result
    # set (a stale-but-valid file beats a silently emptied one).
    def _base_metadata():
        return {
            'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
            'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
            'rwkv_backend': os.environ.get('RWKV_BACKEND', 'reference'),
            'rwkv7_upstream_binding': RWKV7_UPSTREAM_BINDING,
            'ops': list(ops_list),
            'tasks': list(task_list),
            # Reproducibility block: code version, environment, seeds, and
            # the exact training recipe. Without these a results file cannot
            # be traced back to the run that produced it.
            'provenance': capture_provenance(),
            'seeds': [42 + i for i in range(n_seeds)],
            'n_seeds': n_seeds,
            # Data version block: this experiment's "dataset" is synthetic
            # and fully determined by (generator function, task spec, seed
            # + offset). Recording all three makes the data version explicit
            # instead of implicit in the code revision.
            'data': {
                'kind': 'synthetic',
                'generator_functions': {
                    t: SYNTHETIC_TASKS[t]['make'].__name__
                    for t in task_list
                },
                'task_specs': {
                    t: {
                        'seq_len': SYNTHETIC_TASKS[t]['seq_len'],
                        'vocab': SYNTHETIC_TASKS[t]['vocab'],
                        'n_kv_swept': bool(SYNTHETIC_TASKS[t].get('n_kv_swept')),
                    }
                    for t in task_list
                },
                'seed_offsets': {
                    'train': 1_000_000,
                    'eval': 5_000_000,
                    'val_monitor': _VAL_MONITOR_SEED_OFFSET,
                    'error_analysis': _ERROR_ANALYSIS_SEED_OFFSET,
                    'overfit_gate': _OVERFIT_BATCH_SEED_OFFSET,
                    'rule_baseline': _RULE_BASELINE_SEED_OFFSET,
                },
                'note': ('Synthetic batches are a deterministic function of '
                         '(task generator, task spec, seed + offset); no '
                         'external dataset version applies. Code revision in '
                         'provenance pins the generator implementation.'),
            },
            # Train/validation/test separation, spelled out because the
            # monitoring and bad-case analysis must not contaminate the
            # reported metric:
            #   train (+1M)          -> fits the model
            #   val monitor (+6M)    -> held-out curve for the over/underfit
            #                           diagnosis and error analysis; read-only
            #   headline eval (+5M)  -> the reported final_acc/final_loss
            #   bad cases (+7M)      -> error categorization examples
            'monitoring': {
                'val_every': val_every,
                'val_batches': val_batches,
                'val_batch': val_batch,
                'error_batches': error_batches,
                'error_batch': error_batch,
                'note': ('The validation-monitoring and error-analysis '
                         'streams are disjoint from both the training and '
                         'the headline-eval streams, and all passes are '
                         'read-only with dedicated generators: enabling '
                         'monitoring cannot change any reported metric.'),
            },
            # Overfit sanity gate verdicts (compact; full curves with the
            # per-step history live in results/exp4_overfit_gate.json).
            'overfit_sanity_check': {
                'enabled': bool(overfit_check_enabled),
                'strict': bool(overfit_strict),
                'steps': overfit_steps,
                'seed': overfit_seed,
                'train_batch': gate_train_batch,
                'reports': [_compact_overfit_report(r)
                            for r in overfit_reports],
                'note': ('Necessary but NOT sufficient: passing proves '
                         'gradients flow and the loss responds on a fixed '
                         'batch; it cannot rule out data-augmentation / '
                         'feature-alignment / label-mapping bugs.'),
            },
            'training': {
                'optimizer': 'AdamW',
                # Historical fixed lr; the per-op EFFECTIVE lr is in
                # lr_by_op (baseline ops use their lr selected on the
                # validation stream when MQAR_BASELINE_TUNE=1).
                'lr': 3e-3,
                'lr_by_op': {op: tuned_lrs.get(op, 3e-3)
                             for op in ops_list},
                'weight_decay_decay_group': 0.01,
                'weight_decay_no_decay_group': 0.0,
                'grad_clip_norm': 1.0,
                'steps': steps,
                'softmax_steps': softmax_steps,
                'batch_gen_seed_offset': 1_000_000,
                'eval_gen_seed_offset': 5_000_000,
            },
            # Efficiency-vs-accuracy: per sweep cell, which ops are Pareto
            # optimal / dominated on (accuracy, params, train time, eval
            # latency), plus the cost-normalized accuracies. Wall-clock
            # timing is a same-run proxy (shared CI hardware), not a kernel
            # benchmark: the dedicated latency experiment (Exp 2/6) remains
            # the authority for latency claims.
            'efficiency_accuracy': {
                'pareto': pareto_reports,
                'cost_note': ('cost axes are within-run wall-clock train '
                              'time, parameter count and per-sample eval '
                              'latency; timing is noisy on shared hardware '
                              'and is a screening proxy, not a latency '
                              'benchmark (use Exp 2/6 for latency claims).'),
                'accessor': ('per-row `pareto` = {optimal, dominated_by, '
                             'acc_per_train_second, acc_per_1k_params, '
                             'acc_per_eval_ms}'),
            },
            'cross_op_claim_check': cross_op_checks,
            # Seed-paired ALL-PAIRS operator comparison: per (task, n_kv)
            # cell, every unordered operator pair tested on the common
            # seeds with an exact sign-flip permutation test (primary), the
            # paired t-test (secondary), the exact sign test, bootstrap CI
            # and Cohen's d_z; p-values Holm-corrected within the cell.
            # `seeds_required_80pct_power` turns inconclusive pairs into an
            # actionable seed budget for the next multi-seed run. When the
            # equivalence margin is enabled (default), each pair ALSO
            # carries a TOST against the pre-specified margin, so "no
            # significant difference" can be separated from a positive
            # "practically equivalent" claim (and from a genuinely
            # inconclusive comparison).
            'op_pairwise_comparisons': op_pairwise,
            'equivalence_testing': {
                'enabled': equivalence_margin is not None,
                'margin': equivalence_margin,
                'margin_source': (
                    'pre_specified (PAIRWISE_EQUIVALENCE_MARGIN constant, '
                    'MQAR_EQUIV_MARGIN env override; 0 disables); never '
                    'derived from the observed CIs'),
                'method': ('two one-sided paired t-tests (TOST) per operator '
                           'pair on the common seeds; Holm-Bonferroni '
                           'within each (task, n_kv) cell'),
                'alpha': float(os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')),
                'ci_level': ((1.0 - 2.0 * float(
                    os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')))
                    if equivalence_margin is not None else None),
                'accessor': ('per-pair `tost`, `holm_adjusted_p_tost`, '
                             '`significant_equivalence_holm` and '
                             '`equivalence_verdict`; per-row '
                             '`pairwise_equivalent_to`'),
            },
            # Tuned-baseline protocol record: how the simple-baseline lrs
            # were selected (validation stream, full grid), what they were,
            # and the seed-paired comparison of every complex op against
            # the tuned reference baseline. This is the metadata that
            # supports (or refutes) "the complex model is better".
            'baseline_protocol': {
                'enabled': bool(baseline_tune_enabled and baseline_ops),
                'baseline_ops': list(baseline_ops),
                'reference_op': baseline_ref,
                'lr_grid': baseline_lrs,
                'tune_seeds': (
                    baseline_tuning[baseline_ops[0]]['tune_seeds']
                    if baseline_tuning else None),
                'selection_stream': f'val_monitor_seed+{_VAL_MONITOR_SEED_OFFSET}',
                'selected_lrs': {op: lr for op, lr in tuned_lrs.items()},
                'selection_records': baseline_tuning,
                'pairwise_vs_reference': baseline_pairwise,
                'headline_required_for_claims': (
                    'beats_tuned_baseline=True on the paired one-sided '
                    'Bonferroni-corrected test. A row only significant vs '
                    'chance may be a learnable task, not evidence that the '
                    'complex operator is better than a tuned simple one.'),
                'note': ('Model selection (lr) used ONLY the held-out '
                         'validation stream (seed +6M), never the headline '
                         'eval stream (seed +5M); the winning lr is then '
                         're-trained from scratch with the full seed count. '
                         'MQAR_BASELINE_TUNE=0 restores the historical '
                         'fixed lr=3e-3 recipe. The complex operators keep '
                         'the historical lr=3e-3: the baseline deliberately '
                         'gets the benefit of tuning, so a complex-operator '
                         'win over the tuned baseline is conservative '
                         '(against the complex op).'),
            },
            # Baseline-ladder rung 0: the non-learned symbolic rule the
            # synthetic tasks are built around (see evaluate_rule_baseline).
            # This is a task-ceiling / learnability reference, NOT a
            # competitive tuned baseline — superiority claims still require
            # beats_tuned_baseline=True against the tuned LEARNED baseline.
            'rule_baseline_reference': {
                'enabled': bool(rule_baseline_enabled),
                'seed_offset': _RULE_BASELINE_SEED_OFFSET,
                'cells': {
                    f'{task}/n_kv={n_kv}': dict(rec)
                    for (task, n_kv), rec in rule_baselines.items()
                },
                'accessor': ('per-row `rule_baseline` = the cell record '
                             '(rule name, accuracy, chance, seed stream) or '
                             'None for error stubs'),
                'role': ('non_learned_task_ceiling_reference: a '
                         'parameter-free symbolic rule evaluated once on a '
                         'dedicated RNG stream. NOT the comparison baseline '
                         'for superiority claims (those require '
                         'beats_tuned_baseline against the tuned learned '
                         'baseline); it establishes that the task is '
                         'solvable by an exact rule and quantifies how much '
                         'of it the fixed-budget trained probe recovers.'),
                'note': ('The baseline ladder is therefore: rule reference '
                         '(0 params) -> tuned learned baseline (softmax, lr '
                         'selected on the held-out stream) -> complex '
                         'operators, which must beat the TUNED baseline '
                         '(paired, Bonferroni-corrected) to support a '
                         'superiority claim.'),
            },
            # Pre-registered mechanism hypotheses (fixed in code BEFORE the
            # numbers were read): what each task probes and what signature
            # each operator is predicted to show, with the falsification
            # criterion. The sweep is hypothesis-driven, not a fishing
            # expedition; each result row carries its own copy.
            'hypotheses': {
                'tasks': {t: dict(TASK_HYPOTHESES[t])
                          for t in task_list if t in TASK_HYPOTHESES},
                'operators': {op: dict(OPERATOR_HYPOTHESES[op])
                              for op in ops_list
                              if op in OPERATOR_HYPOTHESES},
                'note': ('Pre-registered in run_quality.TASK_HYPOTHESES / '
                         'OPERATOR_HYPOTHESES. Statistical claims: a complex '
                         'op "beats" the baseline only if '
                         'beats_tuned_baseline=True (paired, Bonferroni); '
                         'a row significant vs chance but not vs the tuned '
                         'baseline only shows the task is learnable.'),
            },
            # Machine-readable rule application for the hypotheses above.
            # Every result row also carries its own ``hypothesis_status``
            # and ``hypothesis_checks`` (rule + status + evidence).
            'hypothesis_verdicts': {
                'checks': hypothesis_verdicts,
                'row_status_counts': hypothesis_counts,
                'statuses': list(_HYPOTHESIS_STATUSES),
                'accessor': ('per-row `hypothesis_status` (one of '
                             f'{list(_HYPOTHESIS_STATUSES)}) and '
                             '`hypothesis_checks` (per-rule status + '
                             'numeric evidence)'),
                'note': ('Verdicts are computed by fixed code rules, not '
                         'assigned by hand: contradicted only when the '
                         'pre-registered falsification condition fires '
                         '(a significant effect in the predicted-wrong '
                         'direction), not_supported when the test is '
                         'inconclusive, not_computable when rows or paired '
                         'seeds are missing.'),
            },
            'significance_scope': (
                'vs_chance_baseline_and_seed_paired_vs_tuned_reference_'
                'baseline_and_seed_paired_op_vs_op_holm'
                if baseline_pairwise.get('n_tests')
                else ('vs_chance_baseline_and_seed_paired_op_vs_op_holm'
                      if op_pairwise.get('n_tests')
                      else 'vs_chance_baseline_not_pairwise_between_ops')),
            'csa_indexer_trained': True,
            'csa_ste_enabled': True,
            'csa_indexer_normalize_qk': True,
            'csa_caveat': (
                "CSA's lightning indexer is trained via a straight-through "
                "estimator (STE): the forward pass uses hard top-k indices "
                "(genuine sparse selection), but the backward pass routes "
                "gradients through a differentiable soft distribution over "
                "all compressed blocks. After backward(), the indexer "
                "parameters (W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx) receive "
                "non-None .grad and are updated by the optimizer. The STE "
                "does NOT change the forward semantics — CSA is still "
                "sparse retrieval — but makes the indexer learnable."
            ),
            'training_steps_fairness': {
                'softmax_steps': softmax_steps,
                'other_ops_steps': steps,
                # Make the note CONDITIONAL on the actual asymmetry,
                # matching the summary-table note, so it never contradicts
                # the numeric fields above when softmax_steps == steps.
                'note': (
                    "The softmax baseline is trained for more steps than "
                    "the other operators so it actually converges (the "
                    "original 100 steps left softmax at ~10% accuracy, "
                    "barely above the 6.25% chance level — a useless "
                    "upper bound). Any cross-op accuracy comparison must "
                    "annotate this asymmetry. See README 'Fairness notes' "
                    "section."
                ) if softmax_steps != steps else (
                    "Default fair comparison: every operator (including "
                    "softmax) receives the same optimizer-step budget "
                    "(MQAR_STEPS). An optional longer-softmax "
                    "sensitivity run can be requested via the "
                    "MQAR_SOFTMAX_STEPS env var; in that case "
                    "softmax_steps > other_ops_steps and the asymmetry "
                    "must be annotated as a non-fixed-budget comparison. "
                    "See README 'Fairness notes' section."
                ),
            },
            'schema_version': 1,
        }

    mqar_rows = [r for r in all_results if r.get('task', 'mqar') == 'mqar']
    other_rows = [r for r in all_results if r.get('task', 'mqar') != 'mqar']
    if 'mqar' in task_list:
        payload = {'metadata': _base_metadata(), 'results': mqar_rows}
        try:
            write_json_atomic(payload, 'results/exp4_mqar.json',
                              indent=2, allow_nan=False)
        except ValueError as e:
            logger.error(f'non-finite value in payload; sanitizing to null: {e}')
            write_json_atomic(sanitize_for_json(payload),
                              'results/exp4_mqar.json',
                              indent=2, allow_nan=False)
        logger.info('\nSaved: results/exp4_mqar.json')
    if other_rows:
        payload_b = {
            'metadata': {
                **_base_metadata(),
                'task_descriptions': {
                    t: SYNTHETIC_TASKS[t]['description']
                    for t in task_list if t in SYNTHETIC_TASKS
                },
            },
            'results': other_rows,
        }
        try:
            write_json_atomic(payload_b, 'results/exp4b_quality_tasks.json',
                              indent=2, allow_nan=False)
        except ValueError as e:
            logger.error(f'non-finite value in payload; sanitizing to null: {e}')
            write_json_atomic(sanitize_for_json(payload_b),
                              'results/exp4b_quality_tasks.json',
                              indent=2, allow_nan=False)
        logger.info('Saved: results/exp4b_quality_tasks.json')
    # Return non-zero if any op's training crashed (``'error' in r``), so
    # ``run_all._run`` records the experiment as ``status='fail'`` instead
    # of treating a partial run as success (``run_all._run``'s
    # ``None == success`` contract). Returning 1 forces the failure to
    # propagate to ``run_all``'s summary and exit code.
    n_errors = sum(1 for r in all_results if 'error' in r)
    n_incomplete = sum(
        1 for r in all_results
        if 'error' not in r
        and r.get('n_seeds_ok', n_seeds) != r.get('n_seeds', n_seeds)
    )
    # Append the experiment to the shared run manifest (JSONL ledger) so a
    # campaign-level index links this run's seeds / recipe / data version to
    # its result files — automatic bookkeeping, never manual.
    quality_meta = _base_metadata()
    append_run_manifest(make_run_manifest_entry(
        experiment='exp4_quality',
        seeds=quality_meta['seeds'],
        provenance=quality_meta['provenance'],
        hyperparameters=quality_meta['training'],
        data=quality_meta['data'],
        artifacts={
            'mqar_results': ('results/exp4_mqar.json'
                             if 'mqar' in task_list else None),
            'quality_tasks_results': ('results/exp4b_quality_tasks.json'
                                      if other_rows else None),
            'overfit_gate_results': (
                'results/exp4_overfit_gate.json'
                if overfit_check_enabled else None),
        },
        results={
            'n_ops': len(all_results),
            'n_seeds': int(n_seeds),
            'errors': int(n_errors),
            'incomplete_seed_sets': int(n_incomplete),
            'overfit_gate_enabled': bool(overfit_check_enabled),
            'overfit_gate_failures': int(n_gate_failed),
        },
        status=('completed' if not (n_errors or n_incomplete)
                else 'partial'),
    ), logger=logger)
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_results)} ops errored and '
            f'{n_incomplete}/{len(all_results)} have incomplete seed sets. '
            'Returning non-zero to prevent survivor-only aggregates from '
            'being treated as a successful experiment.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
