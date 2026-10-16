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

Methodology:

  * **Multi-seed runs with confidence intervals.** Every operator is trained
    over ``n_seeds`` (default 5) seeds; we report mean +/- std and a
    one-sample t-test vs the chance baseline.
  * **Softmax baseline convergence.** The softmax-attention baseline is given
    more steps (``softmax_steps``) so it actually converges; a converged
    softmax is the right upper bound.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling back
    to CPU.
  * **Controlled evaluation.** A larger final eval batch (256) and multiple
    eval batches for a more stable accuracy estimate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (configure_torch_for_device, parse_int_env,
                          sanitize_for_json, write_json_atomic,
                          make_seeded_generator, set_global_seed,
                          capture_provenance)
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_fused import HybridConfig
from ops_mamba import build_mamba2_mixer

logger = logging.getLogger(__name__)


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
        return x_emb, target, cue_pos, {'key_idx': j}
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
    # Position of the cued key inside the (shuffled) second presentation.
    new_pos = order2[row, j]
    target = vals_new[row, new_pos]

    x_emb = embed(x)
    if return_meta:
        return x_emb, target, cue_pos, {'key_idx': j}
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
            'distance': (cue_pos - p), 'key_pos': p}
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
                n_batches=4, batch=64, seed=42, task='mqar'):
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
        eval_gen = make_seeded_generator(seed + 5_000_000, device=device)
        make_batch = SYNTHETIC_TASKS[task]['make']
        correct, total = 0, 0
        losses = []
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
        # Guard against n_batches=0 (or batch=0): without this the function
        # raises ZeroDivisionError on ``correct / total`` and
        # ``sum([]) / len([])``. Returns 0.0 for both metrics so the caller
        # gets a finite (if meaningless) value rather than a crash. Mirrors
        # the steps=0 guard in train_one.
        if total == 0 or not losses:
            return 0.0, 0.0
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def _op_factory(op_name, d_model):
    """Return the attention-operator factory for a quality-probe op.

    Lifted out of ``train_one`` so ``run_overfit_sanity_check`` constructs
    the operator through the IDENTICAL code path (same classes, same
    ``d_model``). The factory dictionary itself consumes no RNG draws, so
    routing both callers through this function does not change the seeded
    initialization stream.
    """
    factories = {
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'mamba2':  lambda: Mamba2Attn(d_model),
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
              train_batch=None, task='mqar'):
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

    losses, accs = [], []
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
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(op={op_name}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad()
        loss.backward()
        # Guard against NaN/Inf gradients BEFORE clip+step. ``clip_grad_norm_``
        # computes ``total_norm`` and (if NaN) the comparison
        # ``total_norm > max_norm`` is False, so no clipping happens and the
        # NaN grads pass through to ``opt.step()`` unchecked. That corrupts
        # ALL parameters in one step; the NEXT iteration's forward then
        # produces a NaN loss, which the finite-loss guard above catches —
        # but by then every parameter is already NaN and the seed is lost
        # without a clear root cause. Check here so the per-seed try/except
        # surfaces the real failure mode.
        #
        # The grad-finiteness check runs only every 50 steps to avoid a
        # GPU→CPU sync per parameter on every step (still catches NaN
        # propagation within ~50 steps, well before it can corrupt the
        # final accuracy estimate).
        if step % 50 == 0:
            bad_grads = [p for p in params
                         if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad_grads:
                raise RuntimeError(
                    f"non-finite gradient at step {step} in {len(bad_grads)} "
                    f"params (op={op_name}, seed={seed}); aborting this seed "
                    f"to prevent NaN propagation into parameters")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        accs.append(acc)

    # Final eval on multiple fresh batches for a stable accuracy estimate.
    final_acc, final_loss = _eval_model(
        layer, head, embed, seq_len, n_kv, vocab, device,
        n_batches=eval_batches, batch=eval_batch, seed=seed, task=task,
    )
    chance = 1.0 / vocab
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
            raise RuntimeError(
                f"overfit sanity check: non-finite loss at step {step} "
                f"({loss.item()}) for op={op_name} task={task}; aborting "
                f"before the full sweep.")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
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
        'note': ('Necessary but NOT sufficient: passing proves gradients '
                 'flow and the loss responds on a fixed batch; it cannot '
                 'rule out data-augmentation / feature-alignment / '
                 'label-mapping bugs.'),
    }
    del embed, head, layer, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
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
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
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


def _parse_op_list(env_var, default='softmax,kda,mamba2,csa,hca'):
    """Parse the operator list for the quality sweep.

    ``MQAR_OPS`` selects which single-layer operators run. The default
    includes the Mamba-2 route (the framework's second memory operator);
    set ``MQAR_OPS=softmax,kda,csa,hca`` to reproduce the historical
    four-operator table exactly.
    """
    valid = ['softmax', 'kda', 'mamba2', 'csa', 'hca']
    raw = os.environ.get(env_var, default)
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    if not vals:
        vals = list(valid)
    unknown = [v for v in vals if v not in valid]
    if unknown:
        raise ValueError(
            f'{env_var} includes unknown op(s) {unknown}; valid ops: {valid}.')
    return list(dict.fromkeys(vals))


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
    ops_list = _parse_op_list('MQAR_OPS', 'softmax,kda,mamba2,csa,hca')
    logger.info(f'  tasks         : {task_list}')
    logger.info(f'  ops           : {ops_list}')
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
    logger.info(f'  n_seeds       : {n_seeds}')
    logger.info(f'  steps         : {steps}  (softmax: {softmax_steps})')
    logger.info(f"  mamba_backend : {os.environ.get('MAMBA_BACKEND', 'reference')}")

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
            'metadata': {
                'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
                'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
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
        n_gate_failed = sum(1 for r in overfit_reports
                            if not r.get('passed'))
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
            logger.info('  all overfit gate cells PASSED')

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
                                     seq_len=task_seq_len, task=task)
                all_results.append(r)
                # ci95_acc may be None when only one seed survived (see
                # train_multi_seed). Render as 'n/a' instead of crashing on
                # ``f'{None:.4f}'``.
                ci_str = f"{r['ci95_acc']:.4f}" if r['ci95_acc'] is not None else 'n/a'
                logger.info(f"  -> mean_acc={r['mean_acc']:.4f} +/- {ci_str} "
                            f"(std={r['std_acc']:.4f}, t_vs_chance={_fmt_tstat(r['t_stat_vs_chance'], width=0, prec=2)})")
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
                    'mean_train_time_s': None,
                    'per_seed': [],
                })

    # Summary table (grouped by task + n_kv). Header is 89 chars wide.
    print('\n' + '=' * 89)
    print(f"{'task':>10} | {'n_kv':>4} | {'op':>10} | {'chance':>7} | {'mean_acc':>10} | "
          f"{'+/- CI95':>10} | {'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 89)
    for r in all_results:
        # Error rows have None for all numeric fields; render them as
        # dashes instead of crashing on ``f'{None:.4f}'``.
        if 'error' in r:
            err_msg = (r.get('error') or '')[:40]
            print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
                  f"{'-':>7} | {'-':>10} | {'-':>10} | {'-':>8} | {'-':>12} | "
                  f"{'-':>10}   ERROR: {err_msg}")
            continue
        # ci95_acc may be None when only one seed survived; render 'n/a'.
        if r['ci95_acc'] is not None:
            ci_str = f"{r['ci95_acc']:>10.4f}"
        else:
            ci_str = f"{'n/a':>10}"
        print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
              f"{r['chance_acc']:>7.4f} | {r['mean_acc']:>10.4f} | "
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
                    'overfit_gate': _OVERFIT_BATCH_SEED_OFFSET,
                },
                'note': ('Synthetic batches are a deterministic function of '
                         '(task generator, task spec, seed + offset); no '
                         'external dataset version applies. Code revision in '
                         'provenance pins the generator implementation.'),
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
                'lr': 3e-3,
                'weight_decay_decay_group': 0.01,
                'weight_decay_no_decay_group': 0.0,
                'grad_clip_norm': 1.0,
                'steps': steps,
                'softmax_steps': softmax_steps,
                'batch_gen_seed_offset': 1_000_000,
                'eval_gen_seed_offset': 5_000_000,
            },
            'csa_indexer_trained': True,
            'csa_ste_enabled': True,
            'csa_indexer_normalize_qk': True,
            'significance_scope': 'vs_chance_baseline_not_pairwise_between_ops',
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
