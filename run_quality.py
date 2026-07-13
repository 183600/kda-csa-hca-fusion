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

Kaggle / review-driven additions (address reviewer concerns):

  * **Multi-seed runs with confidence intervals.** Every operator is trained
    over ``n_seeds`` (default 5) seeds; we report mean +/- std and a
    one-sample t-test vs the chance baseline. This addresses the concern
    that the original single-seed numbers were within random noise of
    chance.
  * **Softmax baseline convergence.** The softmax-attention baseline is given
    more steps (``softmax_steps``) so it actually converges — the original
    paper's softmax only reached 10.2% (barely above 6.25% chance), making the
    comparison meaningless. A converged softmax is the *right* upper bound.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling back
    to CPU. The T4 makes the CSA per-token Python loop bearable.
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
                          make_seeded_generator)
from ops_kda import naive_recurrent_kda
from ops_csa import naive_csa
from ops_hca import naive_hca

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
      2. A hardcoded table covering ``n = 2..100`` (df = 1..99). This fixes
         the n = 11..100 range where the previous table (n=2..30 only) fell
         back to the normal approximation 1.96 and lost up to ~8% accuracy
         at n=11..30, and ~4% at n=31..100.
      3. For ``n > 100`` a first-order Cornish-Fisher expansion
         ``1.96 + (1.96^3 + 1.96) / (4 * (n - 1))`` is used. The previous
         fallback to the bare normal approximation ``1.96`` had ~0.4% relative
         error at n=101 for the uncorrected 95% level — but under
         Bonferroni correction (e.g. alpha=0.05/28 ~= 0.0018) the relevant
         critical value is the 99.9% quantile, where the normal approx's
         error balloons past 60% (``t.ppf(0.999, 100) ~= 3.17`` vs ``3.09``
         for the normal). The Cornish-Fisher expansion keeps the error
         below 0.1% at n=101 for the 97.5% level and is a much better
         approximation for the corrected alphas used downstream.
      4. For ``n < 2`` the CI is undefined; returns ``None`` (NOT ``0.0``).
         The previous ``return 0.0`` silently turned a 1-seed estimate into
         a zero-width CI, which downstream consumers interpreted as "the
         estimate is exact" — a false-green that hid the lack of
         replication. Callers MUST handle ``None`` (the one-sample t-test
         path in this file and ``run_ablation`` already guards with
         ``crit is not None`` before comparing).

    The scipy availability check runs once and is cached in ``_T_PP`` so
    repeated calls are essentially free.
    """
    global _T_PP
    if n < 2:
        # P0 fix: return None instead of 0.0. The CI is undefined for n<2
        # (zero degrees of freedom), and ``0.0`` was a load-bearing lie:
        # callers doing ``abs(t_stat) > crit`` would get ``abs(anything) > 0``
        # = True for any non-zero t_stat, falsely marking single-seed
        # estimates as "significant". ``None`` propagates as "undefined",
        # which the t-test guard already handles (``crit is not None and ...``).
        return None
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
        except ImportError:
            _T_PP = False
    if _T_PP:
        return _T_PP(0.975, n - 1)
    # Hardcoded two-sided 95% critical values, n = 2..100 (df = 1..99).
    # Extended from the previous n=2..30 table to cover the common
    # multi-seed CI range up to n=100 without falling back to the
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
        62: 2.000, 63: 1.999, 64: 1.999, 65: 1.998, 66: 1.998,
        67: 1.997, 68: 1.997, 69: 1.996, 70: 1.996, 71: 1.995,
        72: 1.995, 73: 1.994, 74: 1.994, 75: 1.994, 76: 1.993,
        77: 1.993, 78: 1.992, 79: 1.992, 80: 1.992, 81: 1.991,
        82: 1.991, 83: 1.990, 84: 1.990, 85: 1.990, 86: 1.989,
        87: 1.989, 88: 1.989, 89: 1.988, 90: 1.988, 91: 1.988,
        92: 1.987, 93: 1.987, 94: 1.987, 95: 1.986, 96: 1.986,
        97: 1.986, 98: 1.986, 99: 1.985, 100: 1.985,
    }
    if n in _TABLE:
        return _TABLE[n]
    # n > 100: delegate to the exact Student-t inverse CDF used by
    # _bonferroni_crit_q (regularized incomplete beta + bisection). The
    # previous Cornish-Fisher expansion was accurate for the 97.5% level
    # (<0.1% error) but is still an approximation; using the exact path
    # guarantees <1e-9 accuracy without scipy and keeps the CI consistent
    # with the Bonferroni critical values (which already use the exact path).
    # P2-1 fix (2026-07-13): n=200 seeds now gets an exact CI instead of
    # an approximate one.
    try:
        return _bonferroni_crit_q(n, alpha=0.025)
    except Exception:
        # Fallback to Cornish-Fisher if _bonferroni path itself fails
        # (should never happen, but keep a safe net).
        z = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
        return z + (z ** 3 + z) / (4.0 * (n - 1))


def _bonferroni_crit_q(n, alpha=0.05):
    """Bonferroni-corrected ONE-SIDED t critical value with ``n-1`` dof.

    P0-3 fix (lifted to module scope): the previous implementation was a
    nested function inside ``main()`` that returned ``None`` whenever scipy
    was unavailable. This silently zeroed out ``significant_bonferroni``
    for every row (the caller guards with ``crit is not None and ...``),
    so a missing scipy dependency caused ALL significance conclusions to
    disappear without any warning the user could act on.

    The fix mirrors ``_t_crit_975``'s resolution order:
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
         threshold and silently make the test too conservative. A previous
         Cornish-Fisher approximation was finite but still too liberal for
         small seed counts, so the fallback now computes the t CDF directly.
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
    # regularized incomplete beta function and bisection. Earlier fallback
    # revisions used a Cornish-Fisher approximation; even the third-order
    # form can be a few percent liberal at n=5 in Bonferroni-tail regimes,
    # which is enough to flip borderline ``significant_bonferroni`` flags.
    # The numerical integration below is dependency-free and accurate for
    # the small seed counts used by Exp4/Exp5.
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
    lo, hi = 0.0, 1.0
    while _student_t_cdf(hi, dof) < target_p:
        hi *= 2.0
        if hi > 1.0e6:
            raise RuntimeError(
                f"_bonferroni_crit_q fallback could not bracket quantile "
                f"for n={n}, alpha={alpha}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, dof) < target_p:
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
# (Experiment 5: hybrid layout ablation). Previously these two experiments
# used INCONSISTENT CSA/HCA sub-layer widths:
#   * run_quality.CSAAttn:      c=32, cI=16, m=4, nh=2, nIh=2, topk=4, dc=64
#   * run_ablation._make_cfg:   c=16, cI=8,  m=4, nh=2, nIh=2, topk=4, dc=32
# I.e. the ablation's CSA sub-layer was HALF the width of the standalone
# MQAR experiment's CSA. Cross-experiment comparisons (e.g. "the hybrid
# block's CSA contributes X to MQAR accuracy") were silently confounded by
# this width difference. We lift the spec to a single module-level constant
# so both experiments use the SAME widths, making cross-experiment
# comparisons apples-to-apples. The values match the (wider) run_quality
# spec since that is the more informative regime for CSA's sparse retrieval.
# ---------------------------------------------------------------------------
SMALL_MODEL_SPEC = {
    # CSA sub-layer
    'csa_c':      32,   # compressed KV dim
    'csa_cI':     16,   # indexer key dim
    'csa_dc':     64,   # down-projected query dim
    'csa_m':      4,    # compression factor
    'csa_nh':     2,    # number of attention heads
    'csa_nIh':    2,    # number of indexer heads
    'csa_topk':   4,    # top-k blocks per query
    'csa_sliding_window': 4,
    # HCA sub-layer (m2 >> m so HCA produces fewer compressed blocks)
    'hca_c':      32,
    'hca_dc':     64,
    'hca_m2':     8,    # heavy compression (2x CSA's m=4)
    'hca_nh':     2,
    'hca_sliding_window': 4,
}


def make_mqar_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                    embed: nn.Embedding, device=None, generator=None):
    """Build an MQAR batch with *learnable* embeddings.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv): alternating (key, value) tokens
      positions [2*n_kv, seq_len-1): random noise tokens
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
    # solvable or unsolvable. Both used to fail silently.
    if 2 * n_kv > vocab:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be <= vocab={vocab} (need 2*n_kv distinct ids)")
    if 2 * n_kv >= seq_len:
        raise ValueError(
            f"2*n_kv={2*n_kv} must be < seq_len={seq_len} (need room for cue token)")

    # Random noise base (covers noise positions; KV positions overwritten below).
    x = torch.randint(0, vocab, (batch, seq_len), device=device, generator=generator)
    cue_pos = seq_len - 1

    # One uniformly-random permutation of [0, vocab) per sequence, obtained
    # by argsort of iid uniform keys (ties have measure zero in float32).
    # Slicing the first 2*n_kv columns yields 2*n_kv distinct ids per row.
    pair_ids = torch.rand(batch, vocab, device=device, generator=generator).argsort(dim=-1)[:, :2 * n_kv]
    keys = pair_ids[:, 0::2]   # [batch, n_kv]
    vals = pair_ids[:, 1::2]   # [batch, n_kv]

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
    return embed(x), target, cue_pos


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
    # Module-level default for the per-channel log-decay scale. Historically
    # a magic 0.1 was hardcoded in 4 independent KDA instantiations
    # (KDAHybridLayer, KDAAttn, KDAAttnDecoding, _kda_heads). We lift it to
    # a named constant here so all KDA modules share the same value via
    # HybridConfig.kda_decay_scale (the fused model) or this attribute (the
    # standalone KDAAttn). The default (0.1) preserves the historical
    # behaviour. Mirrors run_decoding.KDAAttnDecoding.decay_scale.
    DECAY_SCALE = 0.1

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        # Match KDAHybridLayer's low-rank gate parameterization: d -> K -> H*K.
        # The previous direct d -> H*K gate made standalone Exp4 KDA use a
        # different operator boundary than the hybrid KDA layer and the Exp3
        # FLOPs formula, silently confounding cross-experiment comparisons.
        self.g_down = nn.Linear(d_model, K, bias=False)
        self.g_up = nn.Linear(K, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V

    def forward(self, x):
        B, T, d = x.shape
        # View BEFORE normalize: F.normalize(dim=-1) must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector. The
        # previous form normalized the full H*K vector, shrinking each
        # head's L2 norm to ~1/sqrt(H) and under-scaling q.k dot products
        # by 1/H. Mirrors the fix in ops_fused.py::KDAHybridLayer.
        q = F.normalize(F.silu(self.q(x)).view(B, T, self.H, self.K), dim=-1)
        k = F.normalize(F.silu(self.k(x)).view(B, T, self.H, self.K), dim=-1)
        v = F.silu(self.v(x)).view(B, T, self.H, self.V)
        # Log-space gate: low-rank down/up with a softplus-style decay,
        # matching ops_fused.KDAHybridLayer and run_kv_cache.prefill_flops.
        # Uses the named DECAY_SCALE constant so all KDA instantiations agree.
        g = -F.softplus(self.g_up(self.g_down(x))).view(
            B, T, self.H, self.K) * self.DECAY_SCALE
        beta = torch.sigmoid(self.beta(x))
        out, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
        return self.o(out.reshape(B, T, self.H * self.V))


class CSAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Use the shared small-model spec so Experiment 4 (standalone MQAR)
        # and Experiment 5 (ablation) test the SAME CSA sub-layer widths.
        # Previously this used c=32, cI=16, dc=64 while run_ablation._make_cfg
        # used c=16, cI=8, dc=32 — half the width — making cross-experiment
        # comparisons silently confounded.
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
        # and trims its output back to the original T, so we no longer need
        # to pad/trim here. The external padding was originally added to fix
        # a LEFT-padding bug, but that fix now lives inside ``naive_csa``
        # itself (see ``test_csa_hca_non_divisible_T``). Removing the
        # redundant padding keeps this wrapper thin and avoids doing the
        # pad/trim twice.
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
        # and trims its output back to the original T, so we no longer need
        # to pad/trim here. See the comment in ``CSAAttn.forward`` for the
        # full rationale.
        o = naive_hca(x, self.W_KV.weight, self.W_Z.weight, self.B_pos,
                      self.W_DQ.weight, self.W_UQ.weight,
                      m2=self.m2, nh=self.nh, c=self.c, dc=self.dc,
                      sliding_window=SMALL_MODEL_SPEC['hca_sliding_window'],
                      sink_logits=self.sink)
        return self.o(o)


def _eval_model(layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=4, batch=64):
    """Evaluate accuracy over multiple fresh batches for a stable estimate.

    Uses a dedicated ``torch.Generator`` for batch generation so the eval
    batches are reproducible and independent of any global-RNG state left
    over by training. This also makes the eval pass deterministic across
    re-runs, which is useful for debugging.

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
        # Fixed seed for the eval generator so every operator sees the SAME eval
        # batches (apples-to-apples comparison at eval time too, not just train).
        eval_gen = make_seeded_generator(12345, device=device)
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, vocab, embed, device, generator=eval_gen)
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


def train_one(op_name, d_model=32, seq_len=16, n_kv=1, vocab=16,
              steps=100, lr=3e-3, seed=42, device='cpu',
              softmax_steps=None, eval_batches=4, eval_batch=64,
              train_batch=None):
    """Train a single operator on MQAR for ``steps`` steps.

    ``softmax_steps`` lets the softmax baseline train longer to reach
    convergence (the original 100 steps left softmax at ~10%, barely above the
    6.25% chance — a meaningless upper bound).

    ``train_batch`` overrides the per-step training batch size (default 32,
    previously a hardcoded local magic number — now overridable for
    memory-constrained or GPU runs via the ``MQAR_TRAIN_BATCH`` env var).

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
    if softmax_steps is None:
        softmax_steps = steps
    if train_batch is None:
        # Robust env var parsing: a malformed ``MQAR_TRAIN_BATCH=abc`` (or
        # ``=0``, which would crash on the first batch with ZeroDivisionError)
        # previously crashed the whole experiment with no informative error.
        # ``parse_int_env`` logs a warning and falls back to the default,
        # matching the robustness pattern already used for BENCH_REPEATS in
        # run_benchmark.py.
        train_batch = parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1,
                                    logger=logger)
    actual_steps = softmax_steps if op_name == 'softmax' else steps

    torch.manual_seed(seed)
    factories = {
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'csa':     lambda: CSAAttn(d_model),
        'hca':     lambda: HCAAttn(d_model),
    }
    # Create embed and head BEFORE the operator-specific layer so their
    # initial weights are IDENTICAL across operators for a given seed.
    # Different operators have different parameter counts, so creating the
    # layer first would consume a different number of RNG draws and desync
    # the downstream head init — a silent confound in the multi-seed CI.
    # (The previous order was embed -> layer -> head, which left the head's
    # init operator-dependent.)
    embed = nn.Embedding(vocab, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    layer = ResidualAttnLayer(factories[op_name](), d_model).to(device)
    # Build parameter groups with proper weight-decay exclusion: embeddings,
    # biases, and LayerNorm parameters are NOT weight-decayed (standard ML
    # practice — decaying these can hurt training quality). The previous
    # version applied weight decay uniformly to all parameters.
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    # Separate generator for batch generation so the per-step batches are
    # IDENTICAL across operators for a given seed (the model init consumed a
    # different number of RNG draws per operator, which would otherwise
    # desync the global RNG and produce different training data per op).
    # BQ9 fix: use a LARGE offset (1_000_000) instead of ``+ 1``. The
    # previous offset collided with the next seed's model init: seed 42's
    # batch used seed 43, which is the SAME RNG stream as seed 43's
    # model initialization. The two RNG streams overlapped, weakening
    # the t-test's independence assumption. A large offset puts the
    # batch RNG stream in a completely different region of the seed
    # space, eliminating the overlap.
    # P2-1 fix (round 3): route through make_seeded_generator for
    # CPU-fallback on older torch builds.
    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    losses, accs = [], []
    # Set train mode ONCE before the loop, not per step (the previous code
    # called layer.train()/head.train() inside the loop, which is a redundant
    # O(steps) no-op once the modules are already in train mode).
    layer.train()
    head.train()
    embed.train()
    for step in range(actual_steps):
        x_emb, target, cue_pos = make_mqar_batch(
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
        # BQ6 fix: the previous check did ``torch.isfinite(p.grad).all()``
        # PER PARAMETER, which forces a GPU→CPU sync per parameter
        # (~15 params × 200 steps × 5 seeds × 4 ops = 60k syncs). The
        # loss is already finite (checked above), and a finite loss with
        # a NaN grad is extremely rare (would require an intermediate
        # activation to overflow to inf then collapse back to a finite
        # loss). We move the grad-finiteness check to run only every 50
        # steps (still catches NaN propagation within ~50 steps, well
        # before it can corrupt the final accuracy estimate), cutting
        # the sync count by ~50x.
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
        n_batches=eval_batches, batch=eval_batch,
    )
    chance = 1.0 / vocab
    # Guard against steps=0 (would crash on accs[-1] / sum([])/0 below).
    # Returns a stub with the eval result and zeros for the training
    # trajectory fields. ``actual_steps`` is reported as 0 so downstream
    # consumers can detect the degenerate case.
    if not losses:
        return {
            'op': op_name,
            'n_kv': n_kv,
            'final_acc': final_acc,
            'final_loss': final_loss,
            'chance_acc': chance,
            'last_train_acc': 0.0,
            'mean_last10_loss': 0.0,
            'mean_last10_acc': 0.0,
            'steps': actual_steps,
            'seed': seed,
            'train_batch': train_batch,
        }
    return {
        'op': op_name,
        'n_kv': n_kv,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'chance_acc': chance,
        'last_train_acc': accs[-1],
        'mean_last10_loss': sum(losses[-10:]) / min(10, len(losses)),
        'mean_last10_acc': sum(accs[-10:]) / min(10, len(accs)),
        'steps': actual_steps,
        'seed': seed,
        'train_batch': train_batch,
    }


def train_multi_seed(op_name, n_seeds=5, steps=100, softmax_steps=500,
                     device='cpu', **kw):
    """Train ``op_name`` over ``n_seeds`` seeds.

    Returns a dict with per-seed results plus aggregate stats (mean, std,
    95% CI half-width via t-distribution).

    Per-seed error handling: a single divergent seed (NaN loss, OOM, etc.)
    is caught and recorded as a stub entry in ``per_seed`` rather than
    crashing the whole operator's multi-seed run. Aggregate stats are
    computed over the surviving (successful) seeds. If ALL seeds fail, the
    function raises a ``RuntimeError`` so the caller can record a stub
    result for the operator (mirrors ``run_ablation.py::eval_layout_multi_seed``).
    Previously, a single seed failure crashed the entire experiment, losing
    all other operators' results — inconsistent with the ablation runner's
    more robust pattern.

    ``device`` may be passed as a string or a ``torch.device``; string
    inputs are coerced for notebook callers.
    """
    # Coerce string device -> torch.device so callers passing 'cpu'/'cuda'
    # from a notebook don't hit AttributeError on ``device.type`` below.
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
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
            # Log BEFORE appending to per_seed. The previous order appended
            # first, so if ``logger.info`` raised (e.g. a handler error, or
            # a formatting error on an unexpected None field) the seed had
            # already been recorded as a success and the except branch would
            # append a second stub entry for the same seed — corrupting
            # aggregate stats with a duplicate seed.
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"steps={r['steps']}  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            per_seed.append({
                'seed': s, 'error': str(e),
                'final_acc': None, 'final_loss': None,
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
        # precision. The previous code returned ``0.0``, which lied about
        # the precision of a single-seed mean.
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
        'n_kv': ok_per_seed[0]['n_kv'],
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
    return vals


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 4: MQAR Synthetic Quality Probe (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device        : {device}')
    # These are the task hyperparameters. Lifted to named constants so the
    # chance-accuracy log line below uses the SAME value as the per-seed
    # ``chance = 1.0 / vocab`` computation in ``train_one`` (the previous
    # code hardcoded 1/16 in three print/log lines, which would have silently
    # lied if vocab were ever changed).
    vocab = 16
    seq_len = 16
    chance = 1.0 / vocab
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
    logger.info(f'  vocab={vocab}, seq_len={seq_len}, n_kv={n_kv_list}')
    logger.info(f'  chance accuracy = {chance:.4f} (independent of n_kv; target is 1-of-vocab)')
    # Robust env var parsing: a single malformed value (e.g. ``MQAR_SEEDS=abc``)
    # previously crashed the whole multi-seed experiment with a bare
    # ``ValueError: invalid literal for int()``. ``parse_int_env`` logs a
    # warning and falls back to the default, matching the robustness pattern
    # already used for BENCH_REPEATS / BENCH_LENGTHS in run_benchmark.py.
    n_seeds = parse_int_env('MQAR_SEEDS', 5, min_value=1, logger=logger)
    steps = parse_int_env('MQAR_STEPS', 200, min_value=1, logger=logger)
    # Softmax gets more steps to actually converge (original paper's 100 left
    # it at ~10%, barely above 6.25% chance — a useless upper bound).
    softmax_steps = parse_int_env('MQAR_SOFTMAX_STEPS', 500, min_value=1,
                                  logger=logger)
    logger.info(f'  n_seeds       : {n_seeds}')
    logger.info(f'  steps         : {steps}  (softmax: {softmax_steps})')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for op in ['softmax', 'kda', 'csa', 'hca']:
            logger.info(f'\nTraining {op} (n_kv={n_kv}, {n_seeds} seeds)...')
            # Per-operator try/except so ONE failing operator (all seeds
            # diverged / OOMed) does not crash the whole experiment and
            # lose the other operators' results. The error is logged and
            # recorded as a stub result so the JSON file is always written
            # and downstream figure generation can skip the missing op
            # gracefully (mirrors run_ablation.py's per-ratio try/except).
            try:
                r = train_multi_seed(op, n_seeds=n_seeds, steps=steps,
                                     softmax_steps=softmax_steps, device=device,
                                     n_kv=n_kv, vocab=vocab, seq_len=seq_len)
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
                    'chance_acc': 1.0 / vocab,
                    't_stat_vs_chance': None,
                    'mean_train_time_s': None,
                    'per_seed': [],
                })

    # Summary table (grouped by n_kv). Header is 82 chars wide:
    # 4+3+10+3+10+3+10+3+8+3+12+3+10 = 82. Use the same width for the rules.
    print('\n' + '=' * 82)
    print(f"{'n_kv':>4} | {'op':>10} | {'mean_acc':>10} | {'+/- CI95':>10} | "
          f"{'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 82)
    for r in all_results:
        # Error rows have None for all numeric fields; render them as
        # dashes instead of crashing on ``f'{None:.4f}'``.
        if 'error' in r:
            err_msg = (r.get('error') or '')[:40]
            print(f"{r['n_kv']:>4} | {r['op']:>10} | {'-':>10} | "
                  f"{'-':>10} | {'-':>8} | {'-':>12} | {'-':>10}   ERROR: {err_msg}")
            continue
        # ci95_acc may be None when only one seed survived; render 'n/a'.
        if r['ci95_acc'] is not None:
            ci_str = f"{r['ci95_acc']:>10.4f}"
        else:
            ci_str = f"{'n/a':>10}"
        print(f"{r['n_kv']:>4} | {r['op']:>10} | {r['mean_acc']:>10.4f} | "
              f"{ci_str} | {r['std_acc']:>8.4f} | "
              f"{_fmt_tstat(r['t_stat_vs_chance'], width=12, prec=2)} | {r['mean_loss']:>10.4f}")
    # Chance row: fill ALL columns so the table renders as a clean grid
    # (the previous version only filled 3 of 7 columns, leaving the right
    # side of the row ragged with no trailing pipe separators).
    print(f"{'':>4} | {'chance':>10} | {chance:>10.4f} | "
          f"{'':>10} | {'':>8} | {'':>12} | {'':>10}")
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

    # P1-1 fix — add Bonferroni correction + ``conclusions_valid`` to Exp 4,
    # mirroring the logic already present in run_ablation.py::main for Exp 5.
    # The README documents Exp 4's JSON schema as including
    # ``conclusions_valid`` and a Bonferroni-corrected t-test, but
    # run_quality.py previously emitted only the raw ``t_stat_vs_chance``
    # field. This made the README's "authoritative signal" claim false:
    # consumers reading ``conclusions_valid`` from ``exp4_mqar.json`` would
    # KeyError (the field was absent), and the Bonferroni correction was
    # silently missing.
    #
    # The fix mirrors run_ablation.py's logic:
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
    n_tests = len(['softmax', 'kda', 'csa', 'hca']) * len(n_kv_list)
    alpha_corrected = 0.05 / n_tests
    # P0-3 fix: _bonferroni_crit_q is now a module-level function with a
    # proper Cornish-Fisher fallback when scipy is unavailable (instead of
    # silently returning None and zeroing out all significance conclusions).
    # We just detect scipy availability for the log line below.
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'\n  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=Cornish-Fisher)')
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
            # P0-3 fix: use a ONE-SIDED test (``t_stat > crit``) instead of
            # ``abs(t_stat) > crit``. The research question is "does this op
            # learn the task ABOVE chance", which is directional. The previous
            # two-sided test flagged an op as "significant" even when its
            # accuracy was significantly BELOW chance (large negative t_stat),
            # which is the opposite of what "this op works" means. A below-chance
            # result indicates the model is systematically wrong (e.g. a sign
            # bug, a reversed label, or pure noise that happens to anti-correlate),
            # NOT that the op "works". The Bonferroni-corrected critical value
            # ``_bonferroni_crit_q`` is already the one-sided upper-tail quantile, so the
            # one-sided comparison is the correct use of that quantile.
            r['significant_bonferroni'] = (
                crit is not None and t_stat > crit
            )
        else:
            r['t_crit_bonferroni'] = None
            r['significant_bonferroni'] = False

    # Experiment-level statistical validity summary. Mirrors run_ablation.py
    # but with the appropriate thresholds for the 4-op MQAR sweep.
    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results
                        if 'error' not in r), default=0)
    # A result is "near chance" if mean_acc < 1.5x the chance level.
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and r['mean_acc'] < 1.5 * chance]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0 and len(near_chance) < len(all_results) // 2)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary (P1-1 fix):')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ops with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
    logger.info(f'  ops near chance (<1.5x): {len(near_chance)}/{len(all_results)}')
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
    # string to the file. The previous pattern called json.dump directly
    # on the file object inside a try/except — when the first dump raised
    # ValueError mid-write (on encountering a NaN), the file was left
    # with a PARTIAL JSON document. The fallback json.dump then APPENDED
    # to the partial content, producing invalid JSON (two concatenated
    # fragments) that no parser could read. Serializing to a string first
    # guarantees atomicity: either the complete JSON is written or nothing
    # is, so the fallback can safely overwrite the (empty) file.
    # P1-1 fix's payload structure: emit a single top-level object
    # ``{"metadata": ..., "results": [...]}`` (NOT two concatenated
    # documents). The previous implementation prepended a standalone
    # metadata object to the results array, producing invalid JSON.
    # ``make_figures.load`` accepts both the envelope and the legacy
    # bare-array format. The P1-5 fix below uses ``write_json_atomic``
    # so the file is written atomically (temp + fsync + os.replace),
    # eliminating the truncated-partial-JSON failure mode.
    payload = {
        'metadata': {
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
                "non-None .grad and are updated by the optimizer. This "
                "closes the P0-4 gap where the indexer stayed at random "
                "initialization and CSA's sparse selection was effectively "
                "random. The STE does NOT change the forward semantics — "
                "CSA is still sparse retrieval — but makes the indexer "
                "learnable."
            ),
            'training_steps_fairness': {
                'softmax_steps': softmax_steps,
                'other_ops_steps': steps,
                'note': (
                    "The softmax baseline is trained for more steps than "
                    "the other operators so it actually converges (the "
                    "original 100 steps left softmax at ~10% accuracy, "
                    "barely above the 6.25% chance level — a useless "
                    "upper bound). Any cross-op accuracy comparison must "
                    "annotate this asymmetry. See README 'Fairness notes' "
                    "section."
                ),
            },
            'schema_version': 1,
        },
        'results': all_results,
    }
    # P1-5 fix: use the shared atomic JSON writer (temp file + fsync +
    # os.replace) so a process kill or disk-full mid-write leaves the
    # target file as the OLD version (or absent) rather than a truncated
    # partial JSON document. The previous ``with open(...) as f: f.write(text)``
    # pattern was NOT atomic — see kaggle_setup.write_json_atomic's
    # docstring for the full rationale. The regression guard (re-parse
    # the serialized text) is now inside write_json_atomic itself: it
    # serializes to a string first, which raises ValueError on any
    # non-serializable value BEFORE touching the filesystem.
    try:
        write_json_atomic(payload, 'results/exp4_mqar.json',
                          indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in payload; sanitizing to null: {e}')
        write_json_atomic(sanitize_for_json(payload),
                          'results/exp4_mqar.json',
                          indent=2, allow_nan=False)
    logger.info('\nSaved: results/exp4_mqar.json')
    # P0-2 fix: return non-zero if any op's training crashed (``'error' in r``),
    # so ``run_all._run`` records the experiment as ``status='fail'`` instead of
    # silently treating a partial run as success. The previous ``main()``
    # implicitly returned ``None`` even when every op crashed, which combined
    # with ``run_all._run``'s ``None == success`` contract produced a green
    # summary on a fully-red experiment. Returning 1 forces the failure to
    # propagate to ``run_all``'s summary and exit code.
    n_errors = sum(1 for r in all_results if 'error' in r)
    if n_errors:
        logger.error(
            f'\n[P0-2] {n_errors}/{len(all_results)} ops errored out. '
            f'Returning non-zero so run_all records this experiment as failed.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
