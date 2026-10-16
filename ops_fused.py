"""Fused KDA / Mamba-2 + CSA + HCA hybrid attention stack.

This module defines a hybrid *layer-level* fusion of the attention
operators studied in this repository:

  * **KDA** (Kimi Delta Attention, Kimi Linear arXiv:2510.26692):
        linear-time recurrent memory with fine-grained per-channel gating.
        O(1) state per head; ideal for streaming / decoding.
  * **Mamba-2** (SSD, arXiv:2405.21060, ``ops_mamba.Mamba2Mixer``):
        the second selectable *recurrent memory* route. ``memory_type``
        switches the stack between KDA and Mamba-2 while CSA/HCA stay
        untouched, so the sparse-retrieval (CSA) and global-aggregation
        (HCA) contributions can be ablated independently of the memory
        operator.
  * **RWKV-7** ("Goose", ``ops_rwkv.Rwkv7Mixer``): the third
        selectable *recurrent memory* route (``memory_type='rwkv7'``).
        Parameterization + initialization are bound to
        RWKV-Vibe/rwkv-fla @ ``074deaf``; the recurrence is
        cross-checked against BlinkDL/RWKV-LM @ ``a963e4a`` (see
        ``ops_rwkv.py`` for the bindings and the clearly-labelled
        ``rwkv7-nofirst`` variant). The same one-knob contract applies:
        the RWKV-7 TimeMix (token shift, dynamic bounded decay, matrix
        state update, readout, group norm, gated output projection)
        fills the memory slots while CSA/HCA stay untouched, so the
        KDA→RWKV-7 replacement ablation isolates the memory operator.
  * **CSA** (Compressed Sparse Attention, DeepSeek-V4 arXiv:2606.19348):
        block-wise KV compression + sparse top-k retrieval; good for
        content-addressable long-context recall.
  * **HCA** (Heavily Compressed Attention, DeepSeek-V4):
        aggressive block-wise compression + dense attention + sliding window;
        global context at a cost independent of sequence length.

The :class:`HybridAttentionStack` interleaves the operators in a layout
driven by one of two modes:

  * **Ratio mode (default).** ``(n_kda : n_csa : n_hca)``
    (default ``3 : 1 : 1``) is repeated to ``total_layers`` — the
    layout ``HybridKCHAttention`` uses.
  * **Explicit layout mode.** ``cfg.layout`` is an explicit tuple of layer
    kinds using the abstract token ``'memory'`` (resolved via
    ``cfg.memory_type``) plus ``'csa'`` / ``'hca'``. This is how the
    eight-group experiment matrix (KDA/Mamba2 x A/B/C/D, fixed total
    depth; see ``run_ablation.py``'s ``ABL_MODE=matrix``) is expressed.

``HybridKCHAttention`` is a subclass of :class:`HybridAttentionStack` for
call sites (runners, tests, checkpoints) using that name; with its default
config it produces the KDA ratio layout and numerics.

This is a research-oriented CPU-first implementation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_mamba import Mamba2Mixer, build_mamba2_mixer, validate_mamba2_geometry
from ops_rwkv import Rwkv7Mixer, build_rwkv7_mixer, validate_rwkv7_geometry
from ops_rwkv_backend import validate_rwkv_backend


# Layer kinds accepted in an explicit ``HybridConfig.layout``. ``'memory'``
# is the abstract placeholder resolved to ``cfg.memory_type``; the concrete
# ``'kda'`` / ``'mamba2'`` / ``'rwkv7'`` tokens are also accepted so a
# layout can pin a specific memory operator regardless of the global
# ``memory_type`` (a stack still refuses to MIX two of them — one memory
# route per model, which is what the KDA-vs-Mamba-2-vs-RWKV-7 comparison
# design requires).
LAYOUT_KINDS = ('memory', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca')

# The concrete kinds that fill the stack's *memory* slots (the recurrent
# routes). Every kind here must satisfy the memory-layer contract:
# ``forward(x, state) -> (out, new_state)``,
# ``forward_functional(x, state, conv_state) -> (out, new_state,
# new_conv_state)``, ``reset_conv_state()``. CSA/HCA are stateless.
MEMORY_LAYER_KINDS = ('kda', 'mamba2', 'rwkv7')


def _resolve_layout_kinds(layout, memory_type: str) -> tuple[str, ...]:
    """Validate an explicit layout and resolve ``'memory'`` tokens.

    Returns the concrete kind tuple. Raises ``ValueError`` / ``TypeError``
    with an actionable message on unknown kinds, an empty layout, or a
    layout that mixes KDA and Mamba-2 memory layers in one stack.
    """
    if isinstance(layout, str) or not isinstance(layout, (list, tuple)):
        raise TypeError(
            f"HybridConfig.layout={layout!r} must be a list/tuple of layer "
            f"kinds (e.g. ('memory', 'memory', 'memory', 'csa', 'hca')), "
            f"not a {type(layout).__name__}.")
    if len(layout) == 0:
        raise ValueError(
            "HybridConfig.layout must be non-empty when set explicitly; "
            "use layout=None to fall back to the n_kda:n_csa:n_hca ratio "
            "expansion.")
    resolved = []
    for i, kind in enumerate(layout):
        if not isinstance(kind, str) or kind not in LAYOUT_KINDS:
            raise ValueError(
                f"HybridConfig.layout[{i}]={kind!r} is not a valid layer "
                f"kind; expected one of {LAYOUT_KINDS}.")
        resolved.append(memory_type if kind == 'memory' else kind)
    # Generalized memory-route purity check: a layout may contain memory
    # layers of ONE concrete kind only (kda, mamba2, or rwkv7 — including
    # via the 'memory' placeholder, which resolves to memory_type). The
    # check is the set-difference over MEMORY_LAYER_KINDS, so every
    # route is covered, including future ones.
    used_memory_kinds = sorted(
        {kind for kind in resolved if kind in MEMORY_LAYER_KINDS})
    if len(used_memory_kinds) > 1:
        raise ValueError(
            f"HybridConfig.layout mixes multiple memory layer kinds "
            f"({used_memory_kinds}) in one stack. The comparison "
            f"framework treats the recurrent memory operator as a "
            f"per-model route (memory_type), not a per-layer mix; build "
            f"one stack per memory route instead.")
    return tuple(resolved)


def _memory_layer_lookback(layer: nn.Module) -> torch.Tensor | None:
    """Read a memory layer's streaming lookback buffer.

    KDA / Mamba-2 register ``_conv_lookback`` (the short-conv left
    context); RWKV-7 registers ``_shift_state`` (the token-shift
    history). The stack treats the channel opaquely — this helper is the
    single place that knows the per-route attribute name.
    """
    if isinstance(layer, Rwkv7Mixer):
        return layer._shift_state
    return getattr(layer, '_conv_lookback', None)


def _set_memory_layer_lookback(layer: nn.Module, lb: torch.Tensor | None) -> None:
    """Write a memory layer's streaming lookback buffer (see above).

    Writing the WRONG attribute (e.g. ``_conv_lookback`` on an RWKV-7
    mixer) would create a plain attribute while the REGISTERED buffer
    stayed stale — ``.to(device)`` / ``.half()`` would then move a
    different tensor than the one being threaded through forwards.
    """
    if isinstance(layer, Rwkv7Mixer):
        layer._shift_state = lb
    else:
        layer._conv_lookback = lb


class _Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm that computes in fp32 and returns the input dtype.

    ``torch.nn.LayerNorm`` has no fp16/bf16 CPU kernel. This subclass
    upcasts the input (and the affine params) to fp32, runs the norm, and
    casts the output back to the input dtype; fp32 inputs take the
    standard ``nn.LayerNorm`` path, so values are unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            weight = self.weight.to(torch.float32)
            bias = self.bias.to(torch.float32)
            out = F.layer_norm(
                x.float(), self.normalized_shape, weight, bias, self.eps)
            return out.to(dtype)
        return super().forward(x)

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    d_model: int = 256
    n_heads_qk: int = 4         # H
    n_heads_v: int = 4          # HV (GVA factor = HV // H)
    head_dim_k: int = 64        # K
    head_dim_v: int = 64        # V
    kda_chunk_size: int = 64
    # Recompute-vs-cache knob for the reference chunkwise KDA path. False
    # (default) caches the q-weighted intra-chunk attention matrices for the
    # whole sequence; True drops that cache and recomputes each chunk's
    # matrix inside the sequential loop. Both settings produce the same
    # values. Ignored when ``kda_backend`` selects FLA and when the
    # recurrent path runs (T < kda_chunk_size).
    kda_recompute_attn: bool = False
    # Associative-scan knob for the reference chunkwise KDA path. False
    # (default) resolves the per-chunk state recurrence with the NT-step
    # sequential loop. True selects the Hillis-Steele affine prefix scan
    # over chunks (ceil(log2 NT) batched levels); its results match the
    # sequential path to fp round-off.
    # Only takes effect when the chunk path runs
    # (``kda_chunk_size >= 1`` and ``T >= kda_chunk_size``) and the
    # reference backend is selected; mutually exclusive with
    # ``kda_recompute_attn``.
    kda_assoc_scan: bool = False
    # KDA implementation backend. The default is the reference path; ``fla``
    # selects the optional flash-linear-attention backend; ``auto`` selects it
    # only for CUDA tensors when it is installed and otherwise falls back to
    # the reference path.
    kda_backend: str = "reference"
    # CSA config
    csa_m: int = 16             # compression factor
    csa_topk: int = 8
    csa_nh: int = 4
    csa_c: int = 64
    csa_dc: int = 128
    csa_nIh: int = 2
    csa_cI: int = 32
    csa_sliding_window: int = 16
    # HCA config
    hca_m2: int = 64            # heavy compression (>> csa_m)
    hca_nh: int = 4
    hca_c: int = 64
    hca_dc: int = 128
    hca_sliding_window: int = 16
    # Partial RoPE (DeepSeek-V4 §2.3.3) for the CSA/HCA core attention:
    # queries, compressed KV entries (keys AND values), sliding-window keys
    # are rotated on their last ``rope_dim`` dims, and the core-attention
    # output is inverse-rotated at the negated query position so each KV
    # entry's contribution carries the relative (entry − query) distance.
    # The paper value is 64; ``None``/0 disables RoPE. Small test geometries
    # auto-clamp to the head dim
    # (``min(rope_dim, c)`` rounded down to even).
    csa_rope_dim: int | None = 64
    hca_rope_dim: int | None = 64
    rope_base: float = 10000.0
    # Query/KV normalization form for the CSA/HCA core attention:
    # 'l2'  — cosine attention (unit-norm vectors, scale 1.0);
    # 'rms' — the paper §2.3.3 RMSNorm form (norm ~sqrt(D) vectors, standard
    #         1/sqrt(D) temperature). The paper does not pin the core
    #         temperature, so both are paper-compatible.
    csa_qk_norm_mode: str = 'l2'
    hca_qk_norm_mode: str = 'l2'
    # Hybrid layout
    n_kda: int = 3
    n_csa: int = 1
    n_hca: int = 1
    # KDA decay scale: the per-channel log-decay gate is computed as
    # ``g = -F.softplus(...) * kda_decay_scale``. The value 0.1 is shared by
    # all KDA instantiations (KDAHybridLayer, run_quality.KDAAttn,
    # run_decoding.KDAAttnDecoding, method_analysis._kda_heads); a config
    # field keeps them consistent and makes the value tunable.
    kda_decay_scale: float = 0.1
    # init_std for learnable positional biases (Ba, Bb, B_idx,
    # B_pos). Lifting to a config field makes it tunable and keeps the
    # value consistent across instantiations.
    init_std: float = 0.02
    # NOTE: ``dropout`` is not implemented in any sub-layer (KDA/CSA/HCA).
    # A non-zero value is rejected here so a caller who sets it expecting
    # dropout to be applied gets a clear error instead of silently running
    # without dropout. Remove this guard once dropout is wired into the
    # forward passes.
    dropout: float = 0.0
    # ------------------------------------------------------------------
    # Memory route (KDA vs. Mamba-2) + explicit layout. These fields keep
    # the KDA mainline as the DEFAULT: memory_type='kda' and layout=None
    # select ratio mode.
    # ------------------------------------------------------------------
    # Which recurrent memory operator fills the 'memory' slots: 'kda',
    # 'mamba2', or 'rwkv7'. The CSA / HCA layers and every other field are
    # shared, so switching this ONE value ablates the memory operator while
    # holding the retrieval/global-aggregation operators fixed.
    memory_type: str = "kda"
    # Explicit layer layout. None (default) -> ratio mode (n_kda:n_csa:n_hca
    # repeated to total_layers). A tuple/list pins the exact per-layer
    # kinds, e.g. ('memory','memory','memory','csa','hca') * 2 for the
    # 10-layer eight-group matrix. When set, its length MUST equal the
    # stack's total_layers and the n_kda/n_csa/n_hca ratio fields are
    # ignored (a debug log records the override).
    layout: tuple[str, ...] | list[str] | None = None
    # Mamba-2 implementation backend (mirrors kda_backend): 'reference'
    # (pure PyTorch, default), 'official' (mamba_ssm kernels; CUDA + the
    # optional dependency required — errors loudly when unavailable),
    # 'auto' (official when possible, else reference; the resolved backend
    # is recorded on the mixer's effective_backend attribute).
    mamba_backend: str = "reference"
    # Mamba-2 geometry. Defaults follow the official module; small-scale
    # experiments override them (e.g. headdim=16 for d_model=32).
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    # Number of B/C groups shared across heads (official default 1 = every
    # head sees the same B/C). Kept configurable for the official-backend
    # parity tests; the reference path supports any divisor of nheads.
    mamba_ngroups: int = 1
    # dt low-rank dimension: 'auto' -> ceil(d_model / 16) (official rule).
    mamba_dt_rank: int | str = "auto"
    # Chunked-SSD scan knob for the REFERENCE Mamba-2 path (mirrors
    # kda_assoc_scan): 0 (default) runs the sequential reference recurrence;
    # a positive value selects ops_mamba.chunked_mamba2_scan with that chunk
    # size (quadratic intra-chunk + an associative scan over the scalar
    # chunk transitions, with chunk-local log-space gating). Both paths
    # match to fp round-off and satisfy the same state/dtype contract. The
    # official backend ignores it (its kernels do their own internal
    # chunking), so requesting the official backend with a chunked scan is
    # rejected instead of silently dropping the knob.
    mamba_chunk_size: int = 0
    # ------------------------------------------------------------------
    # RWKV-7 route (memory_type='rwkv7'). Defaults follow the fla layer
    # recipe evaluated at the main-table geometry (d_model=256):
    # head_dim=64 -> H=4 heads; decay/a LoRA=32, gate LoRA=64 (fla's
    # ``max(32, round(coeff * sqrt(d) / 32) * 32)`` with coefficients
    # 2.5 / 2.5 / 5). Small-scale runs override (e.g. run_ablation passes
    # head_dim clamped to a divisor of d_model).
    # ------------------------------------------------------------------
    # RWKV-7 implementation backend (mirrors kda_backend / mamba_backend):
    # 'reference' (pure PyTorch, default), 'fla' (flash-linear-attention's
    # fla.ops.rwkv7 kernels; CUDA + the optional dependency required —
    # errors loudly when unavailable), 'auto' (fla when possible, else
    # reference; the resolution is recorded on the mixer's
    # effective_backend attribute).
    rwkv_backend: str = "reference"
    # RWKV-7 head dimension (upstream: 64, "don't change"). H =
    # d_model // rwkv_head_dim heads; the recurrent matrix state is
    # H x head_dim x head_dim per layer.
    rwkv_head_dim: int = 64
    # LoRA widths of the decay / a / gate branches (see the comment
    # block above; the v_lora of upstream is absent — this is the
    # 'rwkv7-nofirst' variant, see ops_rwkv.py).
    rwkv_decay_lora: int = 32
    rwkv_a_lora: int = 32
    rwkv_gate_lora: int = 64

    def __post_init__(self):
        # Validate dropout TYPE / finite / range BEFORE the
        # ``NotImplementedError`` guard, so a caller passing ``dropout=NaN``,
        # ``dropout="abc"``, or ``dropout=-0.1`` gets a proper TypeError /
        # ValueError pointing at the actual problem. The ``NotImplementedError``
        # is raised only for valid floats in [0, 1] that are non-zero (the
        # "not yet wired in" case).
        if not isinstance(self.dropout, (int, float)):
            raise TypeError(
                f"HybridConfig.dropout={self.dropout!r} must be a float, "
                f"got {type(self.dropout).__name__}.")
        # bool is a subclass of int; reject it explicitly so ``dropout=True``
        # does not silently become ``dropout=1``.
        if isinstance(self.dropout, bool):
            raise TypeError(
                f"HybridConfig.dropout={self.dropout!r} must be a float, "
                f"got bool.")
        if math.isnan(self.dropout) or math.isinf(self.dropout):
            raise ValueError(
                f"HybridConfig.dropout={self.dropout} must be finite "
                f"(not NaN or Inf).")
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(
                f"HybridConfig.dropout={self.dropout} must be in [0.0, 1.0].")
        # Now the legitimate-but-unimplemented case: a valid float in [0,1]
        # that is not 0.0. Raise NotImplementedError pointing at the missing
        # implementation (NOT at the value itself, which is a valid config).
        if self.dropout != 0.0:
            raise NotImplementedError(
                f"HybridConfig.dropout={self.dropout} is not yet implemented. "
                f"The KDA/CSA/HCA sub-layers do not apply dropout in their "
                f"forward passes. Set dropout=0.0 (the default) or implement "
                f"dropout in KDAHybridLayer / CSAHybridLayer / HCAHybridLayer "
                f"before enabling it."
            )
        # KDA backend selection is deliberately validated at config creation,
        # while the optional FLA import remains lazy until a forward pass. This
        # keeps the default reference experiments dependency-free and makes a
        # typo fail before any model parameters are allocated.
        validate_kda_backend(self.kda_backend)
        # GVA (Grouped Value Attention) constraint: KDA's recurrence requires
        # HV to be an integer multiple of H so that ``repeat_interleave(G, dim=2)``
        # can expand q/k from H heads to HV heads (G = HV // H). Without this
        # check, a misconfigured ``n_heads_qk`` / ``n_heads_v`` pair (e.g.
        # H=3, HV=4) would only surface deep inside ``naive_recurrent_kda``
        # as ``AssertionError: HV=4 must be divisible by H=3 (GVA factor)``
        # — at the first forward pass, with no hint that the *config* (not
        # the call site) is the root cause. Validate here so the error fires
        # at construction time with a clear, actionable message.
        if not isinstance(self.n_heads_qk, int) or isinstance(self.n_heads_qk, bool) or self.n_heads_qk < 1:
            raise ValueError(
                f"n_heads_qk={self.n_heads_qk!r} must be a positive int")
        if not isinstance(self.n_heads_v, int) or isinstance(self.n_heads_v, bool) or self.n_heads_v < 1:
            raise ValueError(
                f"n_heads_v={self.n_heads_v!r} must be a positive int")
        if self.n_heads_v % self.n_heads_qk != 0:
            raise ValueError(
                f"n_heads_v={self.n_heads_v} must be divisible by "
                f"n_heads_qk={self.n_heads_qk} (KDA GVA factor G = HV // H "
                f"must be an integer). Adjust n_heads_v or n_heads_qk so the "
                f"ratio is a whole number (e.g. H=2, HV=4 -> G=2).")
        # Validate strictly-positive dimensional params. ``KDAHybridLayer``
        # computes ``self.scale = K ** -0.5`` (line 149), which raises
        # ``ZeroDivisionError`` if ``head_dim_k == 0``. Similarly, ``csa_cI``
        # feeds into ``csa_lightning_indexer``'s ``scale = DI ** -0.5``. A
        # zero value in any of these would otherwise surface as a cryptic
        # torch error deep inside the first forward pass. Validate here so
        # the error fires at config construction with a clear message.
        for name, val in [
            ('d_model', self.d_model),
            ('head_dim_k', self.head_dim_k),
            ('head_dim_v', self.head_dim_v),
            ('csa_m', self.csa_m),
            ('csa_c', self.csa_c),
            ('csa_dc', self.csa_dc),
            ('csa_cI', self.csa_cI),
            ('csa_nIh', self.csa_nIh),
            ('hca_m2', self.hca_m2),
            ('hca_c', self.hca_c),
            ('hca_dc', self.hca_dc),
        ]:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"HybridConfig.{name}={val!r} must be a positive int.")
            if val < 1:
                raise ValueError(
                    f"HybridConfig.{name}={val} must be >= 1. A zero or "
                    f"negative value would cause a division-by-zero or "
                    f"shape error inside the sub-layer forward pass.")
        # ``kda_chunk_size`` controls whether KDAHybridLayer uses the
        # step-by-step recurrent path (``naive_recurrent_kda``) or the
        # chunkwise-parallel path (``naive_chunk_kda``).
        #   * ``kda_chunk_size <= 0`` -> always use the recurrent path
        #     (the default behaviour; also used during autoregressive
        #     decoding where T is small).
        #   * ``kda_chunk_size >= 1`` -> use the chunk path when T is at
        #     least one full chunk (T >= kda_chunk_size), and the
        #     recurrent path for shorter sequences.
        # The chunk path matches the recurrent path to fp tolerance
        # (verified by test_kda_chunk_vs_recurrent in run_correctness.py).
        # NOTE: the chunk path does not support carrying the short-conv
        # lookback across calls (naive_chunk_kda accepts an initial_state but
        # the conv lookback is applied to x BEFORE the q/k/v/g/beta
        # projections, which are the same for both paths — so the conv state
        # is independent of which KDA path is taken). Streaming-decode callers
        # should leave kda_chunk_size at its default if they want chunked
        # training, OR set it to 0 to force the recurrent path during decode.
        if not isinstance(self.kda_chunk_size, int):
            raise ValueError(
                f"kda_chunk_size={self.kda_chunk_size!r} must be an int "
                f"(>= 1 enables the chunk path; <= 0 forces the recurrent "
                f"path; default 64).")
        # ``kda_recompute_attn`` must be a real bool. ``bool`` is the only
        # accepted type; ints are rejected explicitly (Python's ``1 == True``
        # would otherwise silently enable the flag).
        if not isinstance(self.kda_recompute_attn, bool):
            raise TypeError(
                f"kda_recompute_attn={self.kda_recompute_attn!r} must be a "
                f"bool (False caches the intra-chunk attention matrices; "
                f"True recomputes them per chunk).")
        # ``kda_assoc_scan`` follows the same strict-bool contract. The two
        # chunk-path strategy flags select different algorithms and cannot
        # be combined (validated here so the error fires at config creation,
        # not deep inside the first forward pass).
        if not isinstance(self.kda_assoc_scan, bool):
            raise TypeError(
                f"kda_assoc_scan={self.kda_assoc_scan!r} must be a bool "
                f"(False runs the sequential chunk loop; True resolves the "
                f"chunk states with an associative prefix scan).")
        if self.kda_assoc_scan and self.kda_recompute_attn:
            raise ValueError(
                "kda_assoc_scan=True needs the cached intra-chunk attention "
                "matrix and conflicts with kda_recompute_attn=True; enable "
                "exactly one chunk-path strategy.")
        # Validate the remaining structural params. Negative values for
        # these fields would either silently produce an empty layer list
        # (Python's ``[x] * -1 == []``), raise a cryptic torch error
        # inside the first forward pass, or silently disable a branch
        # (looking like the caller intentionally turned it off). Validate
        # them here so the error fires at config construction with a
        # clear message pointing at the misconfigured field.
        #
        # Layer counts: negative values silently produce empty lists via
        # Python's ``[layer] * n`` semantics (``[x] * -1 == []``), so a
        # typo like ``n_kda=-1`` would silently build a model with NO
        # KDA layers instead of raising. Reject negative values up-front.
        for name, val in [
            ('n_kda', self.n_kda),
            ('n_csa', self.n_csa),
            ('n_hca', self.n_hca),
        ]:
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"HybridConfig.{name}={val!r} must be an int.")
            if val < 0:
                raise ValueError(
                    f"HybridConfig.{name}={val} must be >= 0. A negative "
                    f"value would silently produce an empty layer list "
                    f"(Python's ``[x] * n`` semantics), building a model "
                    f"with no {name.replace('n_', '').upper()} layers "
                    f"instead of crashing loudly.")
        if self.n_kda + self.n_csa + self.n_hca == 0:
            raise ValueError(
                f"HybridConfig has n_kda=n_csa=n_hca=0; the model would "
                f"have no attention layers at all. Set at least one of "
                f"n_kda / n_csa / n_hca to a positive value.")
        # CSA-specific: topk=0 is valid (disables sparse selection) but
        # negative values would fail inside ``torch.topk``. nh must be >= 1
        # (it's a head count). sliding_window=0 disables the branch
        # (valid); negative would silently skip the branch.
        if (not isinstance(self.csa_topk, int) or isinstance(self.csa_topk, bool)
                or self.csa_topk < 0):
            raise ValueError(
                f"csa_topk={self.csa_topk!r} must be a non-negative int "
                f"(0 disables sparse selection; negative would crash "
                f"torch.topk).")
        if (not isinstance(self.csa_nh, int) or isinstance(self.csa_nh, bool)
                or self.csa_nh < 1):
            raise ValueError(
                f"csa_nh={self.csa_nh!r} must be a positive int (head count).")
        if (not isinstance(self.csa_sliding_window, int)
                or isinstance(self.csa_sliding_window, bool)
                or self.csa_sliding_window < 0):
            raise ValueError(
                f"csa_sliding_window={self.csa_sliding_window!r} must be a "
                f"non-negative int (0 disables the branch; negative would "
                f"silently skip the sliding-window code path).")
        # HCA-specific: mirrors CSA's checks for the HCA analogs.
        if (not isinstance(self.hca_nh, int) or isinstance(self.hca_nh, bool)
                or self.hca_nh < 1):
            raise ValueError(
                f"hca_nh={self.hca_nh!r} must be a positive int (head count).")
        if (not isinstance(self.hca_sliding_window, int)
                or isinstance(self.hca_sliding_window, bool)
                or self.hca_sliding_window < 0):
            raise ValueError(
                f"hca_sliding_window={self.hca_sliding_window!r} must be a "
                f"non-negative int (0 disables the branch; negative would "
                f"silently skip the sliding-window code path).")
        # kda_decay_scale must be a finite, non-negative float. A NaN or
        # inf would propagate into the gate ``g = -softplus(...) * scale``
        # and silently poison the entire recurrence. A negative scale
        # would invert the decay sign (making g positive -> exp(g) > 1
        # -> state explosion), which is never a meaningful configuration.
        if not isinstance(self.kda_decay_scale, (int, float)):
            raise ValueError(
                f"kda_decay_scale={self.kda_decay_scale!r} must be a number.")
        if not math.isfinite(self.kda_decay_scale):
            raise ValueError(
                f"kda_decay_scale={self.kda_decay_scale} must be finite "
                f"(NaN or inf would poison the KDA gate computation).")
        if self.kda_decay_scale < 0:
            raise ValueError(
                f"kda_decay_scale={self.kda_decay_scale} must be >= 0 "
                f"(negative would invert the decay sign and cause state "
                f"explosion).")
        # --- Partial RoPE config (DeepSeek-V4 §2.3.3) -----------------
        # rope dims: None or a non-negative int. None/0 disables RoPE; a
        # positive value is clamped to the head dim (even) at use time, so
        # over-sized requests on small geometries degrade gracefully
        # instead of raising.
        for name, val in [
            ('csa_rope_dim', self.csa_rope_dim),
            ('hca_rope_dim', self.hca_rope_dim),
        ]:
            if val is None:
                continue
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ValueError(
                    f"HybridConfig.{name}={val!r} must be None or a "
                    f"non-negative int (None/0 disables partial RoPE; the "
                    f"paper value is 64).")
        if not isinstance(self.rope_base, (int, float)) \
                or isinstance(self.rope_base, bool) \
                or not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError(
                f"HybridConfig.rope_base={self.rope_base!r} must be a "
                f"finite positive float.")
        # qk_norm_mode: 'l2' (cosine) or 'rms' (paper §2.3.3).
        for name, val in [
            ('csa_qk_norm_mode', self.csa_qk_norm_mode),
            ('hca_qk_norm_mode', self.hca_qk_norm_mode),
        ]:
            if val not in ('l2', 'rms'):
                raise ValueError(
                    f"HybridConfig.{name}={val!r} must be 'l2' or 'rms'.")
        # --- Memory route (KDA / Mamba-2) + explicit layout -------------
        # Validated at config creation (fail before any parameters are
        # allocated), mirroring the kda_backend validation above. The
        # optional mamba_ssm import stays lazy inside ops_mamba_backend.
        if self.memory_type not in MEMORY_LAYER_KINDS:
            raise ValueError(
                f"HybridConfig.memory_type={self.memory_type!r} must be "
                f"one of {MEMORY_LAYER_KINDS}. This field selects the "
                f"recurrent memory operator that fills the memory slots; "
                f"CSA/HCA are controlled by the layout / ratio fields.")
        from ops_mamba_backend import validate_mamba_backend
        validate_mamba_backend(self.mamba_backend)
        # ``mamba_chunk_size`` selects the reference chunked-SSD scan. It is
        # a no-op for the official backend (whose kernels chunk internally),
        # so an explicit 'official' request combined with a positive value
        # is a contradiction — fail loudly rather than silently dropping the
        # knob. 'auto' is allowed: it may resolve to either backend and the
        # mixer records the resolved ``scan_mode`` for provenance.
        if isinstance(self.mamba_chunk_size, bool) \
                or not isinstance(self.mamba_chunk_size, int) \
                or self.mamba_chunk_size < 0:
            raise ValueError(
                f"HybridConfig.mamba_chunk_size={self.mamba_chunk_size!r} "
                f"must be a non-negative int (0 = sequential reference "
                f"scan).")
        if self.mamba_chunk_size > 0 and self.mamba_backend == 'official':
            raise ValueError(
                "HybridConfig.mamba_chunk_size is a reference-backend-only "
                "knob (the official mamba_ssm kernels use their own internal "
                "chunking). Set mamba_backend='reference' (or 'auto') or "
                "mamba_chunk_size=0.")
        # RWKV backend validation mirrors the mamba one: eager at config
        # creation, lazy optional import inside ops_rwkv_backend.
        validate_rwkv_backend(self.rwkv_backend)
        # Validate + normalize the explicit layout. The RESOLVED tuple
        # (with 'memory' replaced by the concrete memory_type) is stored
        # back on the config so downstream consumers (stack, experiment
        # runners, result rows) never re-resolve the abstraction.
        resolved_layout: tuple[str, ...] | None = None
        if self.layout is not None:
            resolved_layout = _resolve_layout_kinds(self.layout,
                                                    self.memory_type)
            self.layout = resolved_layout
        # Mamba-2 geometry is validated ONLY when a Mamba-2 layer will
        # actually be built. KDA-only configs (the default)
        # must not be constrained by Mamba-2's headdim divisibility —
        # tiny test geometries like d_model=32 with the default
        # headdim=64 would otherwise be rejected for no KDA-relevant
        # reason.
        uses_mamba2 = (
            self.memory_type == 'mamba2'
            or (resolved_layout is not None and 'mamba2' in resolved_layout)
        )
        if uses_mamba2:
            validate_mamba2_geometry(
                self.d_model, self.mamba_d_state, self.mamba_d_conv,
                self.mamba_expand, self.mamba_headdim, self.mamba_ngroups,
                self.mamba_dt_rank)
        # RWKV-7 geometry is validated ONLY when an RWKV-7 layer will
        # actually be built (mirrors the Mamba-2 guard: KDA-only configs
        # must not be constrained by another route's divisibility — tiny
        # test geometries like d_model=32 with the default head_dim=64
        # would otherwise be rejected for no KDA-relevant reason).
        uses_rwkv7 = (
            self.memory_type == 'rwkv7'
            or (resolved_layout is not None and 'rwkv7' in resolved_layout)
        )
        if uses_rwkv7:
            validate_rwkv7_geometry(
                self.d_model, self.rwkv_head_dim, self.rwkv_decay_lora,
                self.rwkv_a_lora, self.rwkv_gate_lora)
        if resolved_layout is not None and self.n_kda + self.n_csa \
                + self.n_hca != len(resolved_layout):
            # Informational, not an error: an explicit layout overrides the
            # ratio fields by design (the eight-group matrix's B/C/D groups
            # are not expressible as a repeating n:n:n unit). Record it at
            # debug level so a surprised reader of the logs finds the hint.
            logger.debug(
                "HybridConfig: explicit layout with %d layers overrides the "
                "n_kda/n_csa/n_hca ratio fields (%d:%d:%d).",
                len(resolved_layout), self.n_kda, self.n_csa, self.n_hca)


class KDAHybridLayer(nn.Module):
    """A single KDA sub-layer with the KDA-style neural parameterization."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        d, H, K, V = cfg.d_model, cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v
        self.q_proj = nn.Linear(d, H * K, bias=False)
        self.k_proj = nn.Linear(d, H * K, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_heads_v * V, bias=False)
        # per-channel log-decay gate (low-rank parameterization as in the paper)
        self.g_down = nn.Linear(d, K, bias=False)
        self.g_up = nn.Linear(K, cfg.n_heads_v * K, bias=False)
        self.beta = nn.Linear(d, cfg.n_heads_v, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads_v * V, d, bias=False)
        # Causal depthwise short-conv: pad only on the left so that
        # position t sees {t-2, t-1, t} and NEVER t+1 (future leakage).
        # Conv1d padding=0; left-pad by (k-1) in forward via F.pad.
        self.short_conv = nn.Conv1d(d, d, kernel_size=3, padding=0, groups=d, bias=True)
        # Register ``scale`` as a non-persistent buffer so that
        # ``model.half()`` / ``model.bfloat16()`` cast it along with the
        # rest of the module's parameters. A plain Python float attribute
        # would always stay fp64 and be passed to ``naive_recurrent_kda`` /
        # ``naive_chunk_kda`` as a fp64 Python float even after
        # ``model.half()`` — a precision inconsistency between the scale
        # used inside the KDA math and the rest of the computation.
        self.register_buffer(
            'scale', torch.tensor(K ** -0.5), persistent=False,
        )
        # Persistent short-conv lookback buffer of shape ``[B, k-1, d]`` (i.e.
        # the last ``kernel_size - 1`` time steps of the previous chunk). This
        # is REQUIRED for streaming / autoregressive decoding, where the model
        # is called multiple times with the next chunk: without carrying the
        # conv context, each chunk boundary loses ``k-1`` tokens of left
        # context and produces a boundary artifact (the conv output for the
        # first ``k-1`` tokens of each new chunk is computed against a
        # zero-padded left edge instead of the actual previous tokens).
        #
        # For one-shot forward (training on a full sequence) this buffer is
        # ``None`` and the existing left-pad-with-zeros path is used; the
        # output is identical because the conv is causal and the first ``k-1``
        # positions of a fresh sequence genuinely have no left context.
        #
        # Registered as a non-persistent buffer so ``.to(device)`` /
        # ``.half()`` move/cast it automatically (a plain attribute would be
        # left on the source device / dtype and fail the next forward).
        self.register_buffer('_conv_lookback', None, persistent=False)

    def reset_conv_state(self) -> None:
        """Clear the persistent short-conv lookback buffer.

        Call this between independent sequences so the conv does not "see"
        tokens from the previous sequence as left context for the next one.
        ``HybridKCHAttention.reset_state`` calls this for every KDA layer.
        """
        self._conv_lookback = None

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """Stateful forward.

        Reads ``self._conv_lookback`` and writes the new lookback + new
        recurrent state back to ``self``. This is the API used by
        ``HybridKCHAttention.forward`` and the experiment runners.

        For a **pure / side-effect-free** path (needed for concurrent
        inference, DDP, ``torch.compile``, gradient checkpointing, and
        cross-chunk BPTT), use :meth:`forward_functional` instead — it
        takes the conv lookback as an argument and returns the new
        lookback without mutating ``self``.
        """
        o, new_state, new_lookback = self.forward_functional(
            x, state, self._conv_lookback)
        self._conv_lookback = new_lookback
        return o, new_state

    def forward_functional(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        conv_lookback: torch.Tensor | None,
        *,
        detach_lookback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pure functional forward — does NOT mutate ``self``.

        Side-effect-free state management.

        ``forward()`` reads from and writes to ``self._conv_lookback``
        internally, which breaks:
          * **Concurrent inference** (multiple sequences sharing one module
            instance would clobber each other's lookback).
          * **DDP** (in-place mutation of a registered buffer during forward
            triggers ``RuntimeError: Expected to mark a variable ready``
            because the buffer is not a parameter and DDP does not know
            when to sync it).
          * **Gradient checkpointing** (``torch.utils.checkpoint`` re-runs
            forward during backward; the second run would see the mutated
            lookback from the first run, producing wrong gradients).
          * **``torch.compile``** (graph breaks on Python-side mutation of
            module state; the compiled graph cannot reason about the
            lookback's value across calls).
          * **Cross-chunk BPTT** (the lookback carries activations from the
            previous chunk; if it is silently detached inside ``forward``,
            the gradient cannot flow across chunk boundaries).

        This method takes ``conv_lookback`` as an EXPLICIT argument and
        returns the new lookback as part of the output tuple, leaving
        ``self._conv_lookback`` untouched. Callers who want the stateful
        behavior use :meth:`forward`; callers who need the pure
        path (DDP, checkpointing, ``torch.compile``, BPTT) use this method
        directly and manage the state externally.

        Args:
            x: ``[B, T, d]`` input hidden states.
            state: ``[B, HV, K, V]`` KDA recurrent state from the previous
                call, or ``None`` for a fresh sequence.
            conv_lookback: ``[B, ksize-1, d]`` short-conv left context from
                the previous call, or ``None`` for a fresh sequence.
            detach_lookback: if True (the default), the incoming
                ``conv_lookback`` is detached before use so the autograd
                graph from the previous chunk is not retained. Set to
                **False** to enable cross-chunk BPTT
                (the caller is then responsible for memory management:
                without detaching, the graph retains activations across
                chunks and memory grows linearly in the number of
                chunks). The recurrent ``state`` argument is NOT
                controlled by this flag — see
                :meth:`HybridKCHAttention.forward_functional`'s
                ``detach_state`` parameter for that.

        Returns:
            ``(output, new_state, new_conv_lookback)`` where:
              * ``output`` is ``[B, T, d]`` (after o_proj).
              * ``new_state`` is ``[B, HV, K, V]``.
              * ``new_conv_lookback`` is ``[B, ksize-1, d]`` (or ``None``
                if ``T == 0``).
        """
        B, T, d = x.shape
        cfg = self.cfg
        H, K, V, HV = cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v, cfg.n_heads_v
        # Degenerate empty-sequence case. ``nn.Conv1d`` cannot consume a
        # length-0 input, while the lower-level KDA operators already define a
        # sensible T==0 contract. Preserve/initialize the recurrent state and
        # leave the conv lookback unchanged (modulo batch/device/dtype guards),
        # returning a zero-length output so HybridKCHAttention can handle empty
        # batches consistently with naive_csa / naive_hca.
        if T == 0:
            compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float
            if state is not None and state.shape[0] == B:
                new_state = state.to(device=x.device, dtype=compute_dtype)
            else:
                new_state = torch.zeros(
                    B, HV, K, V, device=x.device, dtype=compute_dtype)
            new_lookback = conv_lookback
            if new_lookback is not None:
                if new_lookback.shape[0] != B:
                    new_lookback = None
                else:
                    new_lookback = new_lookback.to(device=x.device, dtype=x.dtype)
                    if detach_lookback:
                        new_lookback = new_lookback.detach()
                    new_lookback = new_lookback.clone()
            return x.new_zeros(B, 0, d), new_state, new_lookback
        ksize = self.short_conv.kernel_size[0]
        # Build the conv input with proper LEFT context:
        #   * If ``conv_lookback`` is None (fresh sequence / first call),
        #     left-pad with zeros (the conv is causal; the first
        #     ``ksize-1`` positions have no left context).
        #   * If ``conv_lookback`` is set (streaming / autoregressive decode),
        #     prepend the last ``ksize - 1`` time steps from the previous call
        #     so the conv at positions 0..ksize-2 of the new chunk sees the
        #     actual previous tokens instead of zeros.
        # After the conv, we trim the prepended context off so ``x_conv`` has
        # the same length ``T`` as the input. The final ``ksize - 1`` time
        # steps of ``x`` are returned as the new lookback for the next call.
        lookback = conv_lookback
        # Detach the incoming lookback so the autograd graph from the previous
        # step is not retained. The conv lookback is just a tensor of numbers
        # (the previous chunk's activations); it does not need to carry
        # gradients — UNLESS the caller is doing cross-chunk BPTT and wants
        # gradients to flow. In that case they should pass a NON-detached
        # lookback AND ``detach_lookback=False`` (they own the state and the
        # memory-management contract). The stateful ``forward()`` wrapper
        # passes ``self._conv_lookback`` which is always detached (see the
        # assignment below), so the default path remains detach-by-default
        # for memory safety.
        if lookback is not None:
            # Batch-size / device / dtype guards: if any of these changed
            # between calls (e.g. train B=16 -> eval B=8, model.to(cuda),
            # model.half()), the lookback is invalid. Drop it on batch-size
            # change (the lookback is per-sequence and cannot be reused
            # across different batch sizes); cast/move on device/dtype change
            # to preserve the conv context where possible. Mirrors the
            # state-handling pattern in HybridKCHAttention.forward.
            if lookback.shape[0] != B:
                lookback = None
            elif lookback.device != x.device or lookback.dtype != x.dtype:
                lookback = lookback.to(device=x.device, dtype=x.dtype)
                if detach_lookback:
                    lookback = lookback.detach()
            elif detach_lookback:
                lookback = lookback.detach()
        if lookback is None:
            # Fresh sequence: left-pad with zeros (no previous context).
            x_conv_in = F.pad(x.transpose(1, 2), (ksize - 1, 0))
        else:
            # Streaming: prepend the previous chunk's last (ksize-1) tokens.
            # ``lookback`` is [B, ksize-1, d]; concatenate along the time axis
            # (dim=2 after transpose to [B, d, T]).
            x_conv_in = torch.cat(
                [lookback.transpose(1, 2), x.transpose(1, 2)], dim=2)
        # ``.contiguous()`` after the transpose: the conv returns
        # ``[B, d, T]`` while every downstream projection consumes
        # ``[B, T, d]``. Materializing the transposed layout ONCE here is
        # ~2x faster than letting each of the 5+ linears materialize its
        # own internal copy of the same non-contiguous input.
        x_conv = self.short_conv(x_conv_in).transpose(1, 2).contiguous()  # [B, T, d]
        # Compute the new lookback (the last ksize-1 time steps of THIS
        # chunk) to RETURN to the caller. We do NOT write it to self —
        # that is the caller's responsibility (or the stateful
        # ``forward()`` wrapper does it).
        #
        # Respect ``detach_lookback``: build the new lookback from the
        # (possibly non-detached) ``x`` tensor first, then detach it ONLY
        # if ``detach_lookback=True``. When ``detach_lookback=False``, the
        # caller receives a lookback that still carries the autograd graph
        # from the current chunk's input, so the next chunk's forward can
        # backprop into the current chunk's parameters (true cross-chunk
        # BPTT, matching the docstring contract at the top of
        # ``_kda_forward``). The ``.clone()`` is always applied so the
        # caller can mutate the returned tensor without aliasing the
        # source slice (a defensive copy).
        if T >= ksize - 1:
            new_lookback = x[:, -(ksize - 1):]
        else:
            # Chunk shorter than the lookback window: keep what we have and
            # prepend the existing lookback (the most recent ``ksize-1`` tokens
            # overall). This branch is rare (only the final chunk of a stream
            # can be shorter than ksize-1=2) but the right thing to do.
            if lookback is not None:
                combined = torch.cat([lookback, x], dim=1)
                new_lookback = combined[:, -(ksize - 1):]
            else:
                # No previous lookback and chunk shorter than window: pad with
                # zeros on the left to keep the shape contract.
                pad_len = (ksize - 1) - T
                new_lookback = torch.cat(
                    [torch.zeros(B, pad_len, d, device=x.device, dtype=x.dtype),
                     x], dim=1)
        if detach_lookback:
            new_lookback = new_lookback.detach()
        new_lookback = new_lookback.clone()
        # View BEFORE normalize: ``F.normalize(dim=-1)`` must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector —
        # otherwise each head's L2 norm becomes ~1/sqrt(H) instead of 1 and
        # the q.k dot products shrink by a factor of 1/H. Mirrors the
        # CSA/HCA branches below.
        q = F.normalize(F.silu(self.q_proj(x_conv)).view(B, T, H, K), dim=-1)
        k = F.normalize(F.silu(self.k_proj(x_conv)).view(B, T, H, K), dim=-1)
        v = F.silu(self.v_proj(x_conv)).view(B, T, HV, V)
        # log-space gate: low-rank down/up with a softplus-style decay. The
        # value 0.1 is exposed as ``HybridConfig.kda_decay_scale`` so all KDA
        # instantiations (this layer, run_quality.KDAAttn,
        # run_decoding.KDAAttnDecoding, method_analysis._kda_heads) use the
        # same value.
        decay_scale = cfg.kda_decay_scale
        g = -F.softplus(self.g_up(self.g_down(x_conv))).view(B, T, HV, K) * decay_scale
        beta = torch.sigmoid(self.beta(x_conv))                   # [B, T, HV]
        # Choose between the step-by-step recurrent path and the
        # chunkwise-parallel path based on ``cfg.kda_chunk_size``:
        #   * ``kda_chunk_size <= 0`` -> always recurrent (also the only
        #     option for streaming decode where T is small).
        #   * ``kda_chunk_size >= 1`` AND ``T >= kda_chunk_size`` -> chunk
        #     path (matches the recurrent path to fp tolerance).
        #   * ``kda_chunk_size >= 1`` AND ``T < kda_chunk_size`` -> recurrent
        #     path (T is shorter than one chunk).
        # The chunk path is wired in (``kda_chunk_size`` is consumed in
        # ``HybridConfig.__post_init__`` to validate/select it).
        use_chunk = (cfg.kda_chunk_size >= 1 and T >= cfg.kda_chunk_size)
        # Dispatch through the reference/FLA adapter. The adapter receives the
        # already-normalized q/k, log-space g and post-sigmoid beta produced
        # above, so an external backend must not activate them a second time.
        o, new_state = kda_forward(
            q, k, v, g, beta,
            scale=self.scale,
            initial_state=state,
            output_final_state=True,
            chunk_size=cfg.kda_chunk_size,
            use_chunk=use_chunk,
            backend=cfg.kda_backend,
            recompute_attn=cfg.kda_recompute_attn,
            assoc_scan=cfg.kda_assoc_scan,
        )
        return self.o_proj(o.reshape(B, T, HV * V)), new_state, new_lookback


class CSAHybridLayer(nn.Module):
    """A single CSA sub-layer (compression + sparse selection + MQA).

        The lightning indexer uses ``torch.topk`` which returns integer
        indices that do NOT propagate gradients directly. To make the
        indexer parameters (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``,
        ``B_idx``) trainable, ``ops_csa.naive_csa`` uses a straight-through
        estimator (STE): the forward pass still uses the HARD top-k
        indices (so the algorithm remains genuinely sparse), but the
        backward pass routes gradients through a differentiable soft
        distribution over all compressed blocks. After ``backward()``,
        the indexer parameters have non-None ``.grad`` and are updated by
        the optimizer. The STE does NOT change the forward semantics —
        CSA is still sparse retrieval — but it makes the indexer
        *learnable*.

        ``indexer_is_trained`` is ``True`` by default. The
        ``_maybe_warn_indexer`` method only fires when an explicit caller
        disables STE via ``use_ste=False`` (e.g. for ablation against the
        untrained-indexer baseline).
    """

    # Class-level flag so the warning fires once per process, not once per
    # forward call (which would spam the training log). Reset to False at
    # the start of each training run by ``CSAHybridLayer.reset_warned``.
    _indexer_warned = False

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        d, c = cfg.d_model, cfg.csa_c
        self.W_aKV = nn.Linear(d, c, bias=False)
        self.W_bKV = nn.Linear(d, c, bias=False)
        self.W_aZ = nn.Linear(d, c, bias=False)
        self.W_bZ = nn.Linear(d, c, bias=False)
        self.Ba = nn.Parameter(torch.randn(cfg.csa_m, c) * cfg.init_std)
        self.Bb = nn.Parameter(torch.randn(cfg.csa_m, c) * cfg.init_std)
        self.W_DQ = nn.Linear(d, cfg.csa_dc, bias=False)
        self.W_UQ = nn.Linear(cfg.csa_dc, c * cfg.csa_nh, bias=False)
        self.W_IUQ = nn.Linear(cfg.csa_dc, cfg.csa_cI * cfg.csa_nIh, bias=False)
        self.W_w = nn.Linear(d, cfg.csa_nIh, bias=False)
        self.W_KV_idx = nn.Linear(d, cfg.csa_cI, bias=False)
        self.W_Z_idx = nn.Linear(d, cfg.csa_cI, bias=False)
        self.B_idx = nn.Parameter(torch.randn(cfg.csa_m, cfg.csa_cI) * cfg.init_std)
        self.sink = nn.Parameter(torch.zeros(cfg.csa_nh))
        self.o_proj = nn.Linear(c * cfg.csa_nh, d, bias=False)
        self.indexer_is_trained = True
        # Controls whether ``forward`` passes ``use_ste=True`` to
        # ``naive_csa``. Exposed as an instance attribute (not a config
        # field) so ablation code can flip it on a per-layer basis
        # without rebuilding the config.
        self.use_ste = True

    @classmethod
    def reset_warned(cls):
        """Reset the one-time indexer-not-trained warning flag.

        Call this at the start of a fresh training run if you want the
        warning to fire again (e.g. after rotating log files).
        """
        cls._indexer_warned = False

    def _maybe_warn_indexer(self):
        # The indexer IS trained (via STE) by default. This warning only
        # fires when STE is explicitly disabled, which is an opt-in ablation
        # against the untrained baseline.
        if not self.indexer_is_trained and not CSAHybridLayer._indexer_warned:
            import warnings
            warnings.warn(
                "CSAHybridLayer: STE is disabled (use_ste=False), so the "
                "lightning indexer parameters (W_IUQ, W_w, W_KV_idx, "
                "W_Z_idx, B_idx) will NOT be trained — torch.topk returns "
                "integer indices that do not propagate gradients. CSA's "
                "sparse top-k selection will be effectively RANDOM over "
                "the (learned) compressed KV entries. This mode is "
                "intended ONLY for ablation against the untrained-indexer "
                "baseline; production use should keep use_ste=True.",
                stacklevel=3,
            )
            CSAHybridLayer._indexer_warned = True

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        cfg = self.cfg
        # Only warn when STE is explicitly disabled (ablation mode).
        if self.training and not self.use_ste:
            self._maybe_warn_indexer()
        o = naive_csa(
            x, self.W_aKV.weight, self.W_bKV.weight,
            self.W_aZ.weight, self.W_bZ.weight, self.Ba, self.Bb,
            self.W_DQ.weight, self.W_UQ.weight, self.W_IUQ.weight,
            self.W_w.weight, self.W_KV_idx.weight, self.W_Z_idx.weight,
            self.B_idx,
            m=cfg.csa_m, topk=cfg.csa_topk, nh=cfg.csa_nh, nIh=cfg.csa_nIh,
            c=cfg.csa_c, c_I=cfg.csa_cI, dc=cfg.csa_dc,
            sliding_window=cfg.csa_sliding_window, sink_logits=self.sink,
            # Partial RoPE (DeepSeek-V4 §2.3.3): paper-faithful default
            # (64 dims, clamped to the head dim for small geometries).
            rope_dim=cfg.csa_rope_dim,
            rope_base=cfg.rope_base,
            qk_norm_mode=cfg.csa_qk_norm_mode,
            use_ste=self.use_ste,
            # Match the cosine-style CSA indexer contract: rankings should be
            # driven by direction, not by arbitrary q_idx / K_idx magnitudes.
            normalize_qk=True,
        )
        return self.o_proj(o), None


class HCAHybridLayer(nn.Module):
    """A single HCA sub-layer (heavy compression + dense MQA + SW)."""

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        d, c = cfg.d_model, cfg.hca_c
        self.W_KV = nn.Linear(d, c, bias=False)
        self.W_Z = nn.Linear(d, c, bias=False)
        self.B_pos = nn.Parameter(torch.randn(cfg.hca_m2, c) * cfg.init_std)
        self.W_DQ = nn.Linear(d, cfg.hca_dc, bias=False)
        self.W_UQ = nn.Linear(cfg.hca_dc, c * cfg.hca_nh, bias=False)
        self.sink = nn.Parameter(torch.zeros(cfg.hca_nh))
        self.o_proj = nn.Linear(c * cfg.hca_nh, d, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        cfg = self.cfg
        o = naive_hca(
            x, self.W_KV.weight, self.W_Z.weight, self.B_pos,
            self.W_DQ.weight, self.W_UQ.weight,
            m2=cfg.hca_m2, nh=cfg.hca_nh, c=cfg.hca_c, dc=cfg.hca_dc,
            sliding_window=cfg.hca_sliding_window, sink_logits=self.sink,
            # Partial RoPE (DeepSeek-V4 §2.3.3): paper-faithful default
            # (64 dims, clamped to the head dim for small geometries).
            rope_dim=cfg.hca_rope_dim,
            rope_base=cfg.rope_base,
            qk_norm_mode=cfg.hca_qk_norm_mode,
        )
        return self.o_proj(o), None


def build_mixer(kind: str, cfg: HybridConfig, *,
                memory_layer_idx: int | None = None,
                n_memory_layers: int | None = None) -> nn.Module:
    """Unified mixer factory: map a layer kind to a mixer module.

    This is the single construction point for every sub-layer in the
    hybrid stack, so switching the memory route (KDA <-> Mamba-2 <->
    RWKV-7) or adding a future operator touches exactly one dispatch
    table.

    Args:
        kind: one of ``'memory'`` (resolved via ``cfg.memory_type``), the
            concrete memory kinds ``'kda'`` / ``'mamba2'`` / ``'rwkv7'``,
            or ``'csa'`` / ``'hca'``. The stack always passes an
            already-resolved concrete kind; the abstract ``'memory'``
            token is accepted for direct callers (matching the factory
            interface in the framework design).
        cfg: the shared :class:`HybridConfig`. All mixers read only the
            fields relevant to their operator, and every mixer satisfies
            the same outer contract: ``forward(x[, state]) ->
            (out, new_state)``, ``forward_functional(x, state,
            conv_state) -> (out, new_state, new_conv_state)`` (memory
            layers), stateless ``forward(x) -> (out, None)`` (CSA/HCA).
        memory_layer_idx: 0-based index of this layer among the model's
            *memory* layers (None -> 0). Only the RWKV-7 route consumes
            it: its upstream initialization is depth-dependent
            (per-layer decay slope / lerp powers / group-norm gain via
            ``layer_idx / n_layer``), so the stack passes each RWKV
            layer's index among the RWKV layers. KDA / Mamba-2 / CSA /
            HCA have depth-independent parameterizations and ignore it.
        n_memory_layers: total memory layers in the model (None -> 1);
            RWKV-7's init ratios use it exactly as upstream uses
            ``n_layer``.

    Returns:
        The mixer module. The returned class for ``'mamba2'`` depends on
        ``cfg.mamba_backend`` (reference by default; the official
        ``mamba_ssm`` wrapper when explicitly selected and available),
        and ``'rwkv7'`` on ``cfg.rwkv_backend`` (reference by default;
        the FLA kernels when explicitly selected and available).
    """
    if kind == 'memory':
        kind = cfg.memory_type
    if kind == 'kda':
        return KDAHybridLayer(cfg)
    if kind == 'mamba2':
        return build_mamba2_mixer(cfg)
    if kind == 'rwkv7':
        # The stack supplies the depth position; direct single-layer
        # callers get the layer-0-of-1 defaults (guarded upstream).
        return build_rwkv7_mixer(
            cfg,
            rwkv_layer_idx=0 if memory_layer_idx is None else memory_layer_idx,
            n_rwkv_layers=1 if n_memory_layers is None else max(1, n_memory_layers),
        )
    if kind == 'csa':
        return CSAHybridLayer(cfg)
    if kind == 'hca':
        return HCAHybridLayer(cfg)
    raise ValueError(
        f"build_mixer: unsupported mixer kind {kind!r}; expected one of "
        f"{LAYOUT_KINDS}.")


class HybridAttentionStack(nn.Module):
    """Stack of interleaved memory (KDA / Mamba-2 / RWKV-7) / CSA / HCA sub-layers.

    Two layout modes (see the module docstring):

      * **Ratio mode** (``cfg.layout is None``): the pattern
        ``[memory * n_kda, CSA * n_csa, HCA * n_hca]`` repeated to
        ``total_layers``, where ``memory`` is ``cfg.memory_type``. With the
        default ``memory_type='kda'`` the layout and numerics match
        ``HybridKCHAttention``'s ratio mode, and
        ``memory_type='mamba2'`` yields the Mamba-2 route
        (``MAMBA2-MAMBA2-MAMBA2-CSA-HCA`` per 5-layer unit) while
        ``memory_type='rwkv7'`` yields the RWKV-7 route
        (``RWKV7-RWKV7-RWKV7-CSA-HCA``).
      * **Explicit layout mode** (``cfg.layout`` set): the resolved layout
        pins every layer's kind; its length must equal ``total_layers``.

    Every sub-layer shares the SAME outer structure — pre-norm residual
    ``x = x + mixer(norm(x))`` with identical LayerNorm placement, input /
    output dimensions, and residual wiring — so only the token mixer
    differs. The memory layers (KDA, Mamba-2, or RWKV-7)
    carry a stacked recurrent state plus per-layer shift/conv lookbacks;
    CSA/HCA remain stateless inside the stack (their incremental decode state
    lives in ``ops_decoding_cache``).
    """

    LAYER_TYPES = ('kda', 'mamba2', 'rwkv7', 'csa', 'hca')

    def __init__(self, cfg: HybridConfig, total_layers: int = 5):
        super().__init__()
        self.cfg = cfg
        # Validate total_layers BEFORE anything else. A negative value is
        # never a meaningful configuration, but ``_build_layout`` silently
        # accepts it: ``while len(layout) < total_layers`` is False from
        # the start (0 < -1 is False), so the loop never runs and
        # ``layout[:total_layers]`` returns ``[]`` (empty list). The model
        # then has zero layers and ``forward`` is a no-op (returns the
        # input unchanged) — a silently broken model with no diagnostic.
        # total_layers == 0 IS a valid (if useless) no-op and is allowed.
        if not isinstance(total_layers, int) or total_layers < 0:
            raise ValueError(
                f"total_layers={total_layers!r} must be a non-negative int "
                f"(0 produces an empty no-op model; use >= 1 for a real model)."
            )
        self.total_layers = total_layers
        # Guard against the all-zero-ratio infinite loop BEFORE calling
        # _build_layout. Only applies to ratio mode — an explicit layout
        # pins its own (non-empty, config-validated) layer list. If
        # n_kda == n_csa == n_hca == 0 the repeating ``unit`` is empty and
        # ``while len(layout) < total_layers`` never terminates (layout
        # never grows). A user running a custom ablation with (0, 0, 0)
        # would hang the interpreter with no diagnostic.
        # total_layers == 0 is a valid (if useless) no-op and is allowed:
        # _build_layout returns [] and the ModuleLists are empty.
        if (self.cfg.layout is None and total_layers > 0
                and cfg.n_kda == 0 and cfg.n_csa == 0 and cfg.n_hca == 0):
            raise ValueError(
                "HybridAttentionStack requires at least one non-zero layer "
                f"count (n_kda, n_csa, n_hca) in ratio mode; got "
                f"({cfg.n_kda}, {cfg.n_csa}, {cfg.n_hca}) with "
                f"total_layers={total_layers}, which would produce an "
                f"empty repeating unit and an infinite loop in "
                f"_build_layout. Set cfg.layout explicitly for a custom "
                f"layout instead."
            )
        self.layout = self._build_layout()
        self.layers = nn.ModuleList()
        # Number of memory layers is computed BEFORE the build loop so
        # build_mixer can hand each RWKV-7 layer its depth position (the
        # RWKV-7 init is depth-dependent; KDA/Mamba-2/CSA/HCA ignore the
        # extra kwargs).
        self.n_memory_layers = sum(1 for k in self.layout
                                   if k in MEMORY_LAYER_KINDS)
        memory_seen = 0
        for kind in self.layout:
            # Unified mixer construction: the factory is the single place
            # that maps a (resolved) layer kind to a module, so adding a
            # new operator or switching the memory route touches exactly
            # one dispatch table.
            self.layers.append(build_mixer(
                kind, cfg,
                memory_layer_idx=memory_seen,
                n_memory_layers=self.n_memory_layers))
            if kind in MEMORY_LAYER_KINDS:
                memory_seen += 1
        self.norms = nn.ModuleList([_Fp32LayerNorm(cfg.d_model) for _ in self.layout])
        # Memory layers (KDA, Mamba-2, or RWKV-7 — in practice exactly one
        # kind per stack, enforced by the config validation), in layout
        # order. Kept as an nn.ModuleList (not a plain list) so .to(device)
        # / .half() / print(model) keep working on the submodule set.
        self.memory_layers = nn.ModuleList(
            [layer for layer, kind in zip(self.layers, self.layout)
             if kind in MEMORY_LAYER_KINDS])
        # Number of memory layers in the stack (each one needs its OWN
        # state slot in the stacked tensor below). Different memory layers
        # have different parameters, so they must not share recurrent
        # state: layer 2 would otherwise be seeded with layer 0's state.
        # (Computed above, before the build loop; the alias assignment
        # below exposes it under the ``n_kda_layers`` name.)
        # ``n_kda_layers`` is the stacked memory-state count. With the
        # default memory_type='kda' this is literally the number of KDA
        # layers (the name read by run_decoding.py and the correctness
        # tests); for Mamba-2 stacks it counts the Mamba-2 layers filling
        # the same state slots.
        self.n_kda_layers = self.n_memory_layers
        # Persistent memory-layer recurrent states, one per memory layer.
        # Survives across forward calls so the O(1) per-head memory is
        # preserved during autoregressive decoding (the whole point of a
        # recurrent memory operator). CSA/HCA are stateless and must NOT
        # touch this attribute.
        #
        # Stored as a single stacked tensor of shape
        # ``[n_memory_layers, B, ...]`` (KDA: ``[B, HV, K, V]`` per layer;
        # Mamba-2: ``[B, nheads, headdim, d_state]``) — or ``None`` when
        # freshly reset — and registered as a non-persistent buffer so
        # that model.to(device) moves the whole stack along with the
        # parameters (a plain Python list attribute would be left on the
        # original device, causing a device-mismatch error on the next
        # forward) and it is NOT saved into state_dict (persistent=False;
        # it is runtime state that should be reset between sequences /
        # sessions via reset_state(), not restored from a checkpoint).
        # The attribute is named ``_kda_state``; on Mamba-2 stacks it
        # holds the Mamba-2 SSM states, and on RWKV-7 stacks the per-layer
        # matrix states (``[B, H, K, V]`` per layer). run_decoding.py and
        # the correctness tests read this attribute.
        self.register_buffer('_kda_state', None, persistent=False)

    def reset_state(self) -> None:
        """Clear the persistent memory recurrent state AND short-conv lookbacks.

        Call this at the start of training, at the start of each sequence
        during evaluation, or between independent generation sessions.

        Also resets the class-level ``_stateful_warned`` flag so the next
        training-time forward re-emits the "stateful forward in training
        mode" warning if the caller forgot to switch to
        ``forward_functional``. Without this reset, the warning would fire
        only once per PROCESS (the flag is class-level), so a second
        training run in the same process (e.g. a notebook re-execution)
        would silently lose the warning even though the same risk
        applies.
        """
        # Assigning None to a registered buffer is supported by nn.Module
        # and keeps the buffer slot present (just empty) so a subsequent
        # model.to(device) / state_dict save still works correctly.
        self._kda_state = None
        # Also clear the short-conv lookback on every memory layer (KDA or
        # Mamba-2 — both register a ``_conv_lookback`` buffer) so the conv
        # does not "see" tokens from the previous sequence as left context
        # for the next one. Each memory layer owns its own lookback buffer
        # (registered in its __init__); we clear them here in one place so
        # callers only have to remember a single reset call.
        for layer in self.memory_layers:
            layer.reset_conv_state()
        # Reset the class-level warning flag so the next training run
        # re-emits the stateful-forward warning. The flag is class-level
        # (not instance-level) because the warning is about the API
        # contract (stateful forward in training is dangerous regardless
        # of which instance triggers it), but resetting it on
        # ``reset_state()`` is the right place: ``reset_state()`` is
        # called at the start of every training run / eval sequence, so
        # the warning re-fires for each new "session" of use.
        HybridAttentionStack._stateful_warned = False

    def _build_layout(self) -> list[str]:
        # Explicit layout mode: cfg.layout was validated and the 'memory'
        # tokens were resolved to cfg.memory_type in
        # HybridConfig.__post_init__. An explicit layout pins the exact
        # depth, so a mismatch with total_layers is a configuration error
        # (not silently truncated / padded — that would rebuild a model
        # other than the one the caller thinks they specified).
        if self.cfg.layout is not None:
            if len(self.cfg.layout) != self.total_layers:
                raise ValueError(
                    f"HybridAttentionStack: cfg.layout has "
                    f"{len(self.cfg.layout)} layers but "
                    f"total_layers={self.total_layers}. An explicit layout "
                    f"is exact: pass total_layers=len(cfg.layout) or adjust "
                    f"the layout tuple.")
            return list(self.cfg.layout)
        # Ratio mode: one repeating unit = n_kda memory slots + n_csa CSA
        # + n_hca HCA. The memory slots are filled with cfg.memory_type
        # ('kda' by default -> the KDA ratio layout;
        # 'mamba2' -> the MAMBA2-MAMBA2-MAMBA2-CSA-HCA route).
        unit = ([self.cfg.memory_type] * self.cfg.n_kda
                + ['csa'] * self.cfg.n_csa
                + ['hca'] * self.cfg.n_hca)
        # Defensive guard: if the unit is empty (all ratios zero) AND
        # total_layers > 0, the while loop below would never terminate.
        # __init__ already rejects this case, but we guard here too so a
        # future subclass that overrides __init__ cannot reintroduce the
        # hang by calling _build_layout without the pre-check.
        if not unit and self.total_layers > 0:
            raise ValueError(
                "_build_layout: cannot build a non-empty layout from an "
                "empty repeating unit (n_kda=n_csa=n_hca=0)."
            )
        layout = []
        while len(layout) < self.total_layers:
            layout.extend(unit)
        return layout[:self.total_layers]

    def forward(self, x: torch.Tensor):
        """Stateful forward.

        Reads ``self._kda_state`` and the per-KDA-layer
        ``_conv_lookback`` buffers, and writes the new states back to
        ``self``. This is the API used by the experiment runners
        (benchmark, decoding, quality, ablation).

        For a **pure / side-effect-free** path (needed for concurrent
        inference, DDP, ``torch.compile``, gradient checkpointing, and
        cross-chunk BPTT), use :meth:`forward_functional` instead — it
        takes the full state (stacked KDA recurrent state + per-layer
        conv lookbacks) as an argument and returns the new state without
        mutating ``self``.

        .. warning::

            DDP / ``torch.compile`` warning.

            This method mutates ``self._kda_state`` and each KDA layer's
            ``self._conv_lookback`` during forward, which is incompatible
            with:

              * **DDP**: in-place mutation of a registered buffer during
                forward triggers
                ``RuntimeError: Expected to mark a variable ready``
                because the buffer is not a parameter and DDP does not
                know when to sync it.
              * **``torch.compile``**: graph breaks on Python-side
                mutation of module state; the compiled graph cannot
                reason about the lookback / state value across calls.
              * **Concurrent inference**: multiple sequences sharing one
                module instance would clobber each other's state.

            When ``self.training`` is True, we emit a one-shot
            ``UserWarning`` advising the caller to switch to
            :meth:`forward_functional` for DDP / ``torch.compile`` /
            BPTT use. The warning is throttled to once per process per
            class so it does not spam the training log. To silence the
            warning explicitly, set ``HybridKCHAttention._stateful_warned = True``
            after the first forward (or use :meth:`forward_functional`).
        """
        if self.training and not HybridAttentionStack._stateful_warned:
            import warnings as _warnings
            _warnings.warn(
                "HybridAttentionStack.forward() mutates self._kda_state "
                "and per-layer _conv_lookback buffers during forward, "
                "which is incompatible with DDP, torch.compile, "
                "gradient checkpointing, and concurrent inference. "
                "Use forward_functional() (which takes state as an "
                "explicit argument and does not mutate self) for those "
                "use cases. This warning fires once per process per "
                "class. To silence it for legitimate stateful use "
                "(e.g. the experiment runners), set "
                "HybridAttentionStack._stateful_warned = True after the "
                "first forward call.",
                stacklevel=2,
            )
            HybridAttentionStack._stateful_warned = True
        o, new_kda_state, new_conv_lookbacks = self.forward_functional(
            x, self._kda_state,
            [_memory_layer_lookback(layer) for layer in self.memory_layers])
        self._kda_state = new_kda_state
        # Persist the per-layer lookbacks back to the memory layers
        # (route-aware: KDA/Mamba-2 ``_conv_lookback`` vs RWKV-7
        # ``_shift_state``).
        for layer, lb in zip(self.memory_layers, new_conv_lookbacks):
            _set_memory_layer_lookback(layer, lb)
        return o

    # Class-level one-shot warning flag for the stateful-forward DDP /
    # torch.compile incompatibility warning. Reset to False at the start
    # of a new training run by ``reset_state()`` if the caller wants the
    # warning to fire again (e.g. after rotating log files).
    _stateful_warned = False

    def forward_functional(
        self,
        x: torch.Tensor,
        kda_state: torch.Tensor | None,
        conv_lookbacks: list[torch.Tensor | None],
        *,
        detach_state: bool = True,
        detach_lookback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor | None]]:
        """Pure functional forward — does NOT mutate ``self``.

        Side-effect-free state management.

        ``forward()`` mutates ``self._kda_state`` and each
        KDA layer's ``self._conv_lookback`` internally, which breaks
        concurrent inference, DDP, gradient checkpointing,
        ``torch.compile``, and cross-chunk BPTT — see the rationale in
        :meth:`KDAHybridLayer.forward_functional`.

        This method takes the ENTIRE model state as explicit arguments
        and returns the new state without touching ``self``. Callers who
        want the stateful behavior use :meth:`forward`; callers who
        need the pure path use this method directly and manage the state
        externally.

        Args:
            x: ``[B, T, d]`` input hidden states.
            kda_state: ``[n_kda_layers, B, HV, K, V]`` stacked KDA
                recurrent state from the previous call, or ``None`` for
                a fresh sequence.
            conv_lookbacks: list of ``n_kda_layers`` tensors, each
                ``[B, ksize-1, d]`` (or ``None``), the short-conv left
                context for each KDA layer.
            detach_state: if True (the default), the incoming
                ``kda_state`` is detached before use so the autograd
                graph from the previous chunk is not retained. Set to
                **False** to enable cross-chunk BPTT
                through the recurrent state (the caller is then
                responsible for memory management: without detaching,
                the graph retains activations across chunks and memory
                grows linearly in the number of chunks).
            detach_lookback: forwarded to each KDA layer's
                :meth:`KDAHybridLayer.forward_functional`. See that
                method's docstring for details. Default ``True``.

        Returns:
            ``(output, new_kda_state, new_conv_lookbacks)`` where:
              * ``output`` is ``[B, T, d]``.
              * ``new_kda_state`` is ``[n_kda_layers, B, HV, K, V]`` or
                ``None`` if ``n_kda_layers == 0``.
              * ``new_conv_lookbacks`` is a list of ``n_kda_layers``
                tensors (or ``None`` entries).
        """
        stacked = kda_state
        if stacked is not None:
            # Batch dim is axis 1 (axis 0 is the per-layer index).
            if stacked.shape[1] != x.shape[0]:
                logger.debug(
                    "HybridKCHAttention: dropping KDA recurrent state because "
                    "batch size changed (was %d, now %d). This is expected when "
                    "switching between train/eval batches; call reset_state() "
                    "explicitly to suppress this message.",
                    stacked.shape[1], x.shape[0])
                stacked = None
            elif stacked.device != x.device:
                # Device mismatch (e.g. model moved to cuda, stale CPU state):
                # move the state to x's device. Do NOT cast dtype here — the
                # KDA op's ``initial_state.to(compute_dtype)`` handles any
                # dtype mismatch internally (the state is returned in
                # ``compute_dtype`` by default, so casting it to ``x.dtype``
                # here would reintroduce the fp32→fp16→fp32 round-trip that
                # accumulates quantization error in long streaming
                # inference).
                stacked = stacked.to(device=x.device)
                if detach_state:
                    stacked = stacked.detach()
            elif detach_state:
                stacked = stacked.detach()
        if stacked is not None:
            # Validate ``stacked.shape[0]`` matches ``n_kda_layers``.
            # A length mismatch (e.g. caller loaded a stale state_dict
            # from a different config) would otherwise cause an
            # IndexError at ``states[kda_idx]`` with no useful message.
            # Raise a clear error up-front instead.
            if stacked.shape[0] != self.n_memory_layers:
                raise ValueError(
                    f"HybridAttentionStack.forward_functional: stacked "
                    f"memory state has {stacked.shape[0]} layers but the "
                    f"model has {self.n_memory_layers} memory layers. The "
                    f"state was likely persisted from a different config — "
                    f"reinitialize or resize before calling "
                    f"forward_functional.")
            # ``unbind`` the stacked memory state into a list of per-layer
            # states (equivalent to a Python list comprehension over
            # indices).
            states = list(stacked.unbind(0))
        else:
            states = [None] * self.n_memory_layers

        # Build a mutable copy of the conv lookbacks so we can update per
        # layer without touching self.
        lookback_list = list(conv_lookbacks)
        # A length mismatch between ``conv_lookbacks`` and
        # ``n_kda_layers`` means the caller is feeding state from a
        # different config. Raise a clear ValueError rather than
        # silently padding with ``None`` or truncating, which would
        # hide the mismatch and produce confusing downstream errors
        # (e.g. a layer receiving ``None`` when it expects a real tensor).
        if len(lookback_list) != self.n_memory_layers:
            raise ValueError(
                f"HybridAttentionStack.forward_functional: conv_lookbacks "
                f"has length {len(lookback_list)} but the model has "
                f"{self.n_memory_layers} memory layers. The lookbacks were "
                f"likely persisted from a different config — reinitialize "
                f"before calling forward_functional.")

        mem_idx = 0
        new_lookbacks: list[torch.Tensor | None] = []
        for layer, norm, kind in zip(self.layers, self.norms, self.layout):
            residual = x
            x = norm(x)
            if kind in MEMORY_LAYER_KINDS:
                # Use the FUNCTIONAL memory path so we do NOT mutate the
                # layer's shift/conv state. The new lookback is
                # collected into ``new_lookbacks`` for the caller. KDA,
                # Mamba-2, and RWKV-7 implement the SAME functional
                # contract ((x, state, conv_state) -> (out, new_state,
                # new_conv_state)), so the dispatch is uniform.
                o, new_state, new_lb = layer.forward_functional(
                    x, states[mem_idx], lookback_list[mem_idx],
                    detach_lookback=detach_lookback)
                states[mem_idx] = new_state
                new_lookbacks.append(new_lb)
                mem_idx += 1
            else:
                # Stateless CSA/HCA — no state to thread.
                o, _ = layer(x, None)
            x = residual + o

        # Restack the per-layer states for the caller. All memory layers
        # share the same config (KDA: HV/K/V; Mamba-2: nheads/headdim/
        # d_state; RWKV-7: n_heads/head_dim/head_dim), so every entry has
        # the same shape and torch.stack is safe.
        if self.n_memory_layers > 0:
            if any(s is None for s in states):
                raise RuntimeError(
                    "HybridAttentionStack: a memory layer returned "
                    "new_state=None despite output_final_state=True. "
                    "Refusing to persist a partial state stack (would "
                    "silently drop other layers' states). Check the memory "
                    "layer implementation."
                )
            # Explicit shape-consistency check before torch.stack.
            # ``torch.stack`` requires all tensors to have the SAME shape;
            # a future per-layer config (e.g. per-layer head_dim) would
            # otherwise raise a cryptic ``RuntimeError: stack expects each
            # tensor to be equal size`` with no hint about WHICH layer
            # diverged. Check here so the error message points at the
            # mismatched layer indices and shapes.
            ref_shape = tuple(states[0].shape)
            for i, s in enumerate(states):
                if tuple(s.shape) != ref_shape:
                    raise RuntimeError(
                        f"HybridAttentionStack: memory layer {i} returned "
                        f"new_state with shape {tuple(s.shape)} but layer 0 "
                        f"returned shape {ref_shape}. torch.stack requires "
                        f"all per-layer states to have the same shape — "
                        f"the current HybridConfig shares the memory "
                        f"geometry across all memory layers, so a shape "
                        f"mismatch indicates either a bug in the memory "
                        f"layer's state computation or a future per-layer "
                        f"config (e.g. per-layer head_dim) that needs "
                        f"separate state management."
                    )
            new_kda_state = torch.stack(states, dim=0)
        else:
            new_kda_state = None
        return x, new_kda_state, new_lookbacks

    def layout_str(self) -> str:
        return '-'.join(s.upper() for s in self.layout)


class HybridKCHAttention(HybridAttentionStack):
    """Alias class for :class:`HybridAttentionStack`.

    ``HybridKCHAttention`` and :class:`HybridAttentionStack` are the same
    stack; this class subclasses it without changes. The name is used by
    the experiment runners, the correctness suite, and saved notebooks.

    Behaviour:

      * A config with the default ratio mode (``memory_type='kda'``,
        ``layout=None``) produces the SAME layout strings, parameter
        names, state-buffer names, and numerics as
        :class:`HybridAttentionStack`.
      * Passing a ``memory_type='mamba2'`` / explicit-layout config to
        this class works exactly like ``HybridAttentionStack``;
        ``HybridAttentionStack`` is the preferred name for new code.
    """
