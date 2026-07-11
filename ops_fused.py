"""Fused KDA + CSA + HCA hybrid attention block.

This module defines a hybrid *layer-level* fusion of the three attention
operators studied in this paper:

  * **KDA** (Kimi Delta Attention, Kimi Linear arXiv:2510.26692):
        linear-time recurrent memory with fine-grained per-channel gating.
        O(1) state per head; ideal for streaming / decoding.
  * **CSA** (Compressed Sparse Attention, DeepSeek-V4 arXiv:2606.19348):
        block-wise KV compression + sparse top-k retrieval; good for
        content-addressable long-context recall.
  * **HCA** (Heavily Compressed Attention, DeepSeek-V4):
        aggressive block-wise compression + dense attention + sliding window;
        cheapest global context at extreme lengths.

The ``HybridKCHAttention`` class interleaves the three in a configurable ratio
``(n_kda : n_csa : n_hca)`` (default ``3 : 1 : 1``, mirroring the
literature's hybrid ratios — Kimi Linear uses 3:1 KDA:MLA, DeepSeek-V4
alternates CSA/HCA). The forward pass runs a stack of sub-layers; KDA layers
carry a recurrent state across the stack (and across calls), CSA/HCA layers
operate on the raw hidden sequence.

This is a research-oriented CPU implementation: correctness and small-scale
behaviour, not throughput.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_kda import naive_recurrent_kda

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    d_model: int = 256
    n_heads_qk: int = 4         # H
    n_heads_v: int = 4          # HV (GVA factor = HV // H)
    head_dim_k: int = 64        # K
    head_dim_v: int = 64        # V
    kda_chunk_size: int = 64
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
    # Hybrid layout
    n_kda: int = 3
    n_csa: int = 1
    n_hca: int = 1
    # KDA decay scale: the per-channel log-decay gate is computed as
    # ``g = -F.softplus(...) * kda_decay_scale``. The historical magic value
    # 0.1 appeared in 4 independent KDA instantiations (KDAHybridLayer,
    # run_quality.KDAAttn, run_decoding.KDAAttnDecoding,
    # method_analysis._kda_heads); lifting it to a config field keeps all
    # instantiations consistent and makes the value tunable. The default
    # preserves the historical behaviour.
    kda_decay_scale: float = 0.1
    # NOTE: ``dropout`` is accepted for API compatibility / future use but is
    # NOT yet implemented in any sub-layer (KDA/CSA/HCA). A non-zero value is
    # rejected here so a caller who sets it expecting dropout to be applied
    # gets a clear error instead of silently running without dropout (which
    # would be a silent correctness bug in their training recipe). Remove
    # this guard once dropout is actually wired into the forward passes.
    dropout: float = 0.0

    def __post_init__(self):
        if self.dropout != 0.0:
            raise NotImplementedError(
                f"HybridConfig.dropout={self.dropout} is not yet implemented. "
                f"The KDA/CSA/HCA sub-layers do not apply dropout in their "
                f"forward passes. Set dropout=0.0 (the default) or implement "
                f"dropout in KDAHybridLayer / CSAHybridLayer / HCAHybridLayer "
                f"before enabling it."
            )
        # GVA (Grouped Value Attention) constraint: KDA's recurrence requires
        # HV to be an integer multiple of H so that ``repeat_interleave(G, dim=2)``
        # can expand q/k from H heads to HV heads (G = HV // H). Without this
        # check, a misconfigured ``n_heads_qk`` / ``n_heads_v`` pair (e.g.
        # H=3, HV=4) would only surface deep inside ``naive_recurrent_kda``
        # as ``AssertionError: HV=4 must be divisible by H=3 (GVA factor)``
        # — at the first forward pass, with no hint that the *config* (not
        # the call site) is the root cause. Validate here so the error fires
        # at construction time with a clear, actionable message.
        if self.n_heads_qk < 1:
            raise ValueError(
                f"n_heads_qk={self.n_heads_qk} must be >= 1")
        if self.n_heads_v < 1:
            raise ValueError(
                f"n_heads_v={self.n_heads_v} must be >= 1")
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
            if val < 1:
                raise ValueError(
                    f"HybridConfig.{name}={val} must be >= 1. A zero or "
                    f"negative value would cause a division-by-zero or "
                    f"shape error inside the sub-layer forward pass.")
        # ``kda_chunk_size`` controls whether KDAHybridLayer uses the
        # step-by-step recurrent path (``naive_recurrent_kda``) or the
        # chunkwise-parallel path (``naive_chunk_kda``).
        #   * ``kda_chunk_size <= 0`` -> always use the recurrent path
        #     (the historical default behaviour; also used during
        #     autoregressive decoding where T is small).
        #   * ``kda_chunk_size >= 1`` -> use the chunk path when T is at
        #     least one full chunk (T >= kda_chunk_size), falling back to
        #     the recurrent path for short sequences (where the chunk
        #     path's overhead exceeds the parallelism win).
        # Previously this field was unused (always recurrent) and emitted a
        # "UNUSED" warning; the warning is now removed because the field IS
        # wired in. The chunk path matches the recurrent path to fp tolerance
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
        self.scale = K ** -0.5
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
        # left on the source device / dtype and crash the next forward).
        self.register_buffer('_conv_lookback', None, persistent=False)

    def reset_conv_state(self) -> None:
        """Clear the persistent short-conv lookback buffer.

        Call this between independent sequences so the conv does not "see"
        tokens from the previous sequence as left context for the next one.
        ``HybridKCHAttention.reset_state`` calls this for every KDA layer.
        """
        self._conv_lookback = None

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, d = x.shape
        cfg = self.cfg
        H, K, V, HV = cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v, cfg.n_heads_v
        ksize = self.short_conv.kernel_size[0]
        # Build the conv input with proper LEFT context:
        #   * If ``_conv_lookback`` is None (fresh sequence / first call),
        #     left-pad with zeros — identical to the previous behaviour.
        #   * If ``_conv_lookback`` is set (streaming / autoregressive decode),
        #     prepend the last ``ksize - 1`` time steps from the previous call
        #     so the conv at positions 0..ksize-2 of the new chunk sees the
        #     actual previous tokens instead of zeros.
        # After the conv, we trim the prepended context off so ``x_conv`` has
        # the same length ``T`` as the input. The final ``ksize - 1`` time
        # steps of ``x`` are saved into ``_conv_lookback`` for the next call.
        lookback = self._conv_lookback
        # Detach the incoming lookback so the autograd graph from the previous
        # step is not retained. Mirrors the always-detach pattern used for the
        # KDA recurrent state in HybridKCHAttention.forward. The conv lookback
        # is just a tensor of numbers (the previous chunk's activations); it
        # does not need to carry gradients.
        if lookback is not None:
            # Batch-size / device / dtype guards: if any of these changed
            # between calls (e.g. train B=16 -> eval B=8, model.to(cuda),
            # model.half()), the cached lookback is invalid. Drop it on
            # batch-size change (the lookback is per-sequence and cannot be
            # reused across different batch sizes); cast/move on device/dtype
            # change to preserve the conv context where possible. Mirrors the
            # state-handling pattern in HybridKCHAttention.forward.
            if lookback.shape[0] != B:
                lookback = None
            elif lookback.device != x.device or lookback.dtype != x.dtype:
                lookback = lookback.to(device=x.device, dtype=x.dtype).detach()
            else:
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
        x_conv = self.short_conv(x_conv_in).transpose(1, 2)        # [B, T, d]
        # Persist the last (ksize-1) time steps of THIS chunk as the next
        # call's lookback. Detach so we don't retain the graph across calls.
        if T >= ksize - 1:
            new_lookback = x[:, -(ksize - 1):].detach().clone()
        else:
            # Chunk shorter than the lookback window: keep what we have and
            # prepend the existing lookback (the most recent ``ksize-1`` tokens
            # overall). This branch is rare (only the final chunk of a stream
            # can be shorter than ksize-1=2) but the right thing to do.
            if lookback is not None:
                combined = torch.cat([lookback, x], dim=1)
                new_lookback = combined[:, -(ksize - 1):].detach().clone()
            else:
                # No previous lookback and chunk shorter than window: pad with
                # zeros on the left to keep the shape contract.
                pad_len = (ksize - 1) - T
                new_lookback = torch.cat(
                    [torch.zeros(B, pad_len, d, device=x.device, dtype=x.dtype),
                     x], dim=1).detach().clone()
        self._conv_lookback = new_lookback
        # View BEFORE normalize: ``F.normalize(dim=-1)`` must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector. The
        # previous form ``F.normalize(F.silu(...), dim=-1).view(B, T, H, K)``
        # L2-normalized the full H*K vector, so each head's L2 norm became
        # ~1/sqrt(H) instead of 1. This silently shrinks q.k dot products by
        # a factor of 1/H, which propagates into the KDA recurrence as
        # under-scaled delta-rule updates. Mirrors the (correct) CSA/HCA
        # branches below and the fix in method_analysis.py.
        q = F.normalize(F.silu(self.q_proj(x_conv)).view(B, T, H, K), dim=-1)
        k = F.normalize(F.silu(self.k_proj(x_conv)).view(B, T, H, K), dim=-1)
        v = F.silu(self.v_proj(x_conv)).view(B, T, HV, V)
        # log-space gate: low-rank down/up with a softplus-style decay. The
        # magic constant 0.1 is exposed as ``HybridConfig.kda_decay_scale`` so
        # all KDA instantiations (this layer, run_quality.KDAAttn,
        # run_decoding.KDAAttnDecoding, method_analysis._kda_heads) use the
        # same value. The default (0.1) preserves the historical behaviour.
        decay_scale = getattr(cfg, 'kda_decay_scale', 0.1)
        g = -F.softplus(self.g_up(self.g_down(x_conv))).view(B, T, HV, K) * decay_scale
        beta = torch.sigmoid(self.beta(x_conv))                   # [B, T, HV]
        # Choose between the step-by-step recurrent path and the
        # chunkwise-parallel path based on ``cfg.kda_chunk_size``:
        #   * ``kda_chunk_size <= 0`` -> always recurrent (historical
        #     behaviour; also the only option for streaming decode where
        #     T is small).
        #   * ``kda_chunk_size >= 1`` AND ``T >= kda_chunk_size`` -> chunk
        #     path (faster on CPU/GPU for long training sequences; matches
        #     the recurrent path to fp tolerance — verified by
        #     ``test_kda_chunk_vs_recurrent`` in run_correctness.py).
        #   * ``kda_chunk_size >= 1`` AND ``T < kda_chunk_size`` -> recurrent
        #     path (the chunk path's per-chunk overhead exceeds the win at
        #     short T; also avoids the chunk path's right-padding cost).
        # The chunk path is now wired in (previously ``kda_chunk_size`` was
        # a dead field — see the P1-4 fix in ``HybridConfig.__post_init__``).
        use_chunk = (cfg.kda_chunk_size >= 1 and T >= cfg.kda_chunk_size)
        if use_chunk:
            from ops_kda import naive_chunk_kda
            o, new_state = naive_chunk_kda(
                q, k, v, g, beta, scale=self.scale,
                initial_state=state, output_final_state=True,
                chunk_size=cfg.kda_chunk_size,
            )
        else:
            o, new_state = naive_recurrent_kda(
                q, k, v, g, beta, scale=self.scale,
                initial_state=state, output_final_state=True,
            )
        return self.o_proj(o.reshape(B, T, HV * V)), new_state


class CSAHybridLayer(nn.Module):
    """A single CSA sub-layer (compression + sparse selection + MQA).

    .. note:: P0-4 fix — indexer is now trainable via straight-through estimator.

        The lightning indexer uses ``torch.topk`` which returns integer
        indices that do NOT propagate gradients directly. The previous
        implementation left the indexer parameters
        (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx``) at
        their random initialization after ``backward()`` (their ``.grad``
        was ``None`` and AdamW silently skipped them), which made CSA's
        sparse top-k selection effectively **random** over the learned
        compressed KV entries.

        The fix adds a straight-through estimator (STE) in
        ``ops_csa.naive_csa``: the forward pass still uses the HARD
        top-k indices (so the algorithm remains genuinely sparse), but
        the backward pass routes gradients through a differentiable soft
        distribution over all compressed blocks. After ``backward()``,
        the indexer parameters now have non-None ``.grad`` and are
        updated by the optimizer. The STE does NOT change the forward
        semantics — CSA is still sparse retrieval — but it makes the
        indexer *learnable*.

        ``indexer_is_trained`` is now ``True`` by default. The
        ``_maybe_warn_indexer`` method is kept for backward
        compatibility but only fires when an explicit caller disables
        STE via ``use_ste=False`` (e.g. for ablation against the
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
        self.Ba = nn.Parameter(torch.randn(cfg.csa_m, c) * 0.02)
        self.Bb = nn.Parameter(torch.randn(cfg.csa_m, c) * 0.02)
        self.W_DQ = nn.Linear(d, cfg.csa_dc, bias=False)
        self.W_UQ = nn.Linear(cfg.csa_dc, c * cfg.csa_nh, bias=False)
        self.W_IUQ = nn.Linear(cfg.csa_dc, cfg.csa_cI * cfg.csa_nIh, bias=False)
        self.W_w = nn.Linear(d, cfg.csa_nIh, bias=False)
        self.W_KV_idx = nn.Linear(d, cfg.csa_cI, bias=False)
        self.W_Z_idx = nn.Linear(d, cfg.csa_cI, bias=False)
        self.B_idx = nn.Parameter(torch.randn(cfg.csa_m, cfg.csa_cI) * 0.02)
        self.sink = nn.Parameter(torch.zeros(cfg.csa_nh))
        self.o_proj = nn.Linear(c * cfg.csa_nh, d, bias=False)
        # P0-4 fix: the indexer is now trainable via the STE in
        # ``naive_csa``. This flag is read by experiment runners to
        # include the training status in result JSON metadata. Set to
        # ``False`` only when ``use_ste=False`` is explicitly passed to
        # ``naive_csa`` (e.g. for the untrained-indexer ablation).
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
        # After the P0-4 fix the indexer IS trained (via STE) by default.
        # The warning now only fires when STE is explicitly disabled,
        # which is an opt-in ablation against the untrained baseline.
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
            x, self.W_aKV.weight.T, self.W_bKV.weight.T,
            self.W_aZ.weight.T, self.W_bZ.weight.T, self.Ba, self.Bb,
            self.W_DQ.weight.T, self.W_UQ.weight.T, self.W_IUQ.weight.T,
            self.W_w.weight.T, self.W_KV_idx.weight.T, self.W_Z_idx.weight.T,
            self.B_idx,
            m=cfg.csa_m, topk=cfg.csa_topk, nh=cfg.csa_nh, nIh=cfg.csa_nIh,
            c=cfg.csa_c, c_I=cfg.csa_cI, dc=cfg.csa_dc,
            sliding_window=cfg.csa_sliding_window, sink_logits=self.sink,
            use_ste=self.use_ste,
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
        self.B_pos = nn.Parameter(torch.randn(cfg.hca_m2, c) * 0.02)
        self.W_DQ = nn.Linear(d, cfg.hca_dc, bias=False)
        self.W_UQ = nn.Linear(cfg.hca_dc, c * cfg.hca_nh, bias=False)
        self.sink = nn.Parameter(torch.zeros(cfg.hca_nh))
        self.o_proj = nn.Linear(c * cfg.hca_nh, d, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        cfg = self.cfg
        o = naive_hca(
            x, self.W_KV.weight.T, self.W_Z.weight.T, self.B_pos,
            self.W_DQ.weight.T, self.W_UQ.weight.T,
            m2=cfg.hca_m2, nh=cfg.hca_nh, c=cfg.hca_c, dc=cfg.hca_dc,
            sliding_window=cfg.hca_sliding_window, sink_logits=self.sink,
        )
        return self.o_proj(o), None


class HybridKCHAttention(nn.Module):
    """Stack of interleaved KDA / CSA / HCA sub-layers.

    The layout follows the pattern ``[KDA * n_kda, CSA, HCA]`` repeated to
    reach ``total_layers``. This mirrors the layerwise hybrid approach of
    Kimi Linear (3:1 KDA:full-attn) extended with the DeepSeek-V4 CSA/HCA
    pair: KDA handles the bulk of token mixing, CSA adds sparse long-range
    retrieval, HCA adds heavily-compressed global context.
    """

    LAYER_TYPES = ('kda', 'csa', 'hca')

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
        # _build_layout. If n_kda == n_csa == n_hca == 0 the repeating
        # ``unit`` is empty and ``while len(layout) < total_layers`` never
        # terminates (layout never grows). A user running a custom ablation
        # with (0, 0, 0) would hang the interpreter with no diagnostic.
        # total_layers == 0 is a valid (if useless) no-op and is allowed:
        # _build_layout returns [] and the ModuleLists are empty.
        if total_layers > 0 and cfg.n_kda == 0 and cfg.n_csa == 0 and cfg.n_hca == 0:
            raise ValueError(
                "HybridKCHAttention requires at least one non-zero layer "
                f"count (n_kda, n_csa, n_hca); got "
                f"({cfg.n_kda}, {cfg.n_csa}, {cfg.n_hca}) with "
                f"total_layers={total_layers}, which would produce an "
                f"empty repeating unit and an infinite loop in _build_layout."
            )
        self.layout = self._build_layout()
        self.layers = nn.ModuleList()
        for kind in self.layout:
            if kind == 'kda':
                self.layers.append(KDAHybridLayer(cfg))
            elif kind == 'csa':
                self.layers.append(CSAHybridLayer(cfg))
            elif kind == 'hca':
                self.layers.append(HCAHybridLayer(cfg))
            else:
                raise ValueError(kind)
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in self.layout])
        # Number of KDA layers in the stack (each one needs its OWN state).
        # Different KDA layers have different parameters (q_proj, k_proj, ...),
        # so they must not share recurrent state. Sharing state across layers
        # is a correctness bug that silently corrupts autoregressive decoding
        # and training: layer 2 would be seeded with layer 0's state instead
        # of its own, and on the next forward call layer 0 would be seeded
        # with the last layer's state from the previous call.
        self.n_kda_layers = sum(1 for k in self.layout if k == 'kda')
        # Persistent KDA recurrent states, one per KDA layer. Survives across
        # forward calls so that KDA's O(1) per-head memory is preserved during
        # autoregressive decoding (the whole point of KDA). CSA/HCA are
        # stateless and must NOT touch this attribute.
        #
        # Stored as a single stacked tensor of shape
        # ``[n_kda_layers, B, HV, K, V]`` (or ``None`` when freshly reset) and
        # registered as a non-persistent buffer so that:
        #   * model.to(device) moves the whole stack along with the parameters
        #     (a plain Python list attribute would be left on the original
        #     device, causing a device-mismatch crash on the next forward);
        #   * it is NOT saved into state_dict (persistent=False), since it is
        #     runtime state that should be reset between sequences / sessions
        #     via reset_state(), not restored from a checkpoint.
        self.register_buffer('_kda_state', None, persistent=False)

    def reset_state(self) -> None:
        """Clear the persistent KDA recurrent state AND short-conv lookback.

        Call this at the start of training, at the start of each sequence
        during evaluation, or between independent generation sessions.
        """
        # Assigning None to a registered buffer is supported by nn.Module
        # and keeps the buffer slot present (just empty) so a subsequent
        # model.to(device) / state_dict save still works correctly.
        self._kda_state = None
        # Also clear the short-conv lookback on every KDA layer so the conv
        # does not "see" tokens from the previous sequence as left context
        # for the next one. Each KDA layer owns its own lookback buffer
        # (registered in KDAHybridLayer.__init__); we clear them here in one
        # place so callers only have to remember a single reset call.
        for layer in self.layers:
            if isinstance(layer, KDAHybridLayer):
                layer.reset_conv_state()

    def _build_layout(self) -> list[str]:
        # One repeating unit = n_kda KDA + n_csa CSA + n_hca HCA.
        unit = (['kda'] * self.cfg.n_kda
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
        # Unpack the stacked per-layer KDA states. ``_kda_state`` is either
        # ``None`` (freshly reset) or a stacked tensor of shape
        # ``[n_kda_layers, B, HV, K, V]``. We split it into a per-layer list
        # so each KDA layer can be seeded with its OWN state from the previous
        # call — sharing a single state across layers is a correctness bug
        # (each KDA layer has different parameters, so its recurrent state is
        # not interchangeable with another layer's state).
        #
        # Detach the incoming state so the autograd graph from the previous
        # step is not retained. In training mode this prevents
        # "backward through the graph a second time" errors; in eval mode it
        # prevents an O(N) memory leak across N forward calls when the caller
        # forgets to wrap inference in ``torch.no_grad()`` (each call would
        # otherwise retain the previous call's graph, accumulating
        # unbounded memory during long autoregressive decoding). The graph is
        # never needed for inference — eval-with-backward is unusual; if a
        # caller genuinely needs gradients through the recurrent state across
        # calls (e.g. BPTT), they should keep the graph manually rather than
        # relying on this implicit behavior.
        #
        # If the batch size changed (e.g. train batch=16, eval batch=8), the
        # old state is invalid and we drop it. For a DEVICE mismatch (which
        # should not happen now that the state is a registered buffer moved
        # by .to(), but cheap to guard) we MOVE the state instead of dropping
        # it: dropping irreversibly loses the recurrent memory, while moving
        # preserves it. The previous behavior (drop on device mismatch) would
        # silently reset the KDA state whenever the caller forgot to call
        # ``model.to(device)`` before passing inputs on a different device.
        #
        # DTYPE mismatch (e.g. caller did ``model.half()`` between forwards):
        # ``stacked.to(x.device)`` only moves the device, NOT the dtype. The
        # downstream ``naive_recurrent_kda`` does implicitly cast
        # ``initial_state`` to ``compute_dtype`` (fp32 for fp16 inputs), so
        # the recurrence itself does not crash — but the returned
        # ``new_state`` is in ``v.dtype`` (= ``x.dtype`` after ``model.half()``),
        # while any OTHER KDA layer whose state we have not yet overwritten
        # is still in the OLD dtype. ``torch.stack(states, dim=0)`` at the
        # end of forward then crashes with
        # ``RuntimeError: Expected object of scalar type Half but got scalar
        # type Float`` because the per-layer states have mixed dtypes.
        # The fix: explicitly cast ``stacked`` to ``x.dtype`` alongside the
        # device move (and in the same-dtype/device branch too, defensively).
        # This makes the contract explicit: the state is ALWAYS in
        # ``x.dtype`` before being passed to the KDA layers, so all
        # ``new_state`` returns are in ``x.dtype`` and ``torch.stack`` is
        # guaranteed homogeneous.
        stacked = self._kda_state
        if stacked is not None:
            # Batch dim is axis 1 (axis 0 is the per-layer index).
            if stacked.shape[1] != x.shape[0]:
                # Log the batch-size change so the user understands WHY the
                # recurrent state was dropped. Without this log the drop is
                # silent, and a user switching from train (large batch) to
                # eval (small batch) may not realize their KDA state is being
                # reset on every call — which is correct behavior (the state
                # is per-sequence and cannot be reused across different batch
                # sizes) but surprising if you expected streaming continuity.
                # Use a debug-level message (not warning) because this is a
                # routine occurrence during train/eval switches, not an error.
                logger.debug(
                    "HybridKCHAttention: dropping KDA recurrent state because "
                    "batch size changed (was %d, now %d). This is expected when "
                    "switching between train/eval batches; call reset_state() "
                    "explicitly to suppress this message.",
                    stacked.shape[1], x.shape[0])
                stacked = None
            elif stacked.device != x.device or stacked.dtype != x.dtype:
                # Move (device AND dtype) and keep detached rather than drop.
                # ``.to(device=x.device, dtype=x.dtype)`` is a no-op for the
                # dimension that already matches, so this also covers the
                # dtype-only-mismatch case (device matches, dtype differs).
                stacked = stacked.to(device=x.device, dtype=x.dtype).detach()
            else:
                # Always detach: training (BPTT safety) AND eval (memory leak
                # prevention). See the comment above for the full rationale.
                stacked = stacked.detach()
        if stacked is not None:
            states = [stacked[i] for i in range(stacked.shape[0])]
        else:
            states = [None] * self.n_kda_layers

        kda_idx = 0
        for layer, norm, kind in zip(self.layers, self.norms, self.layout):
            residual = x
            x = norm(x)
            if kind == 'kda':
                # KDA is stateful and sequence-continuous. It has no
                # compression factor, so T never needs padding here (padding
                # would desync the recurrent state anyway). Thread THIS
                # layer's own state; do not touch the others.
                o, new_state = layer(x, states[kda_idx])
                states[kda_idx] = new_state
                kda_idx += 1
            else:
                # Stateless CSA/HCA. ``naive_csa`` / ``naive_hca`` handle
                # non-divisible T via internal right-padding and trim their
                # output back to the original T (see
                # ``test_csa_hca_non_divisible_T``), so we no longer need the
                # external pad/trim that used to live here. The external
                # padding was originally added to fix a LEFT-padding bug; that
                # fix now lives inside the operators themselves, making the
                # wrapper-level pad/trim redundant. Do NOT pass ``state`` in,
                # and do NOT let the returned ``None`` overwrite the KDA
                # states list.
                o, _ = layer(x, None)
            x = residual + o
        # Restack the per-layer states for persistence. All KDA layers share
        # the same config (HV, K, V), so every entry has the same shape and
        # torch.stack is safe. If n_kda_layers == 0 (no KDA in the layout),
        # there is nothing to persist.
        #
        # If any single KDA layer returned ``None`` for its new_state (which
        # should not happen because KDA's ``output_final_state=True`` always
        # returns a tensor), we raise rather than silently nuking the OTHER
        # layers' valid states. The previous ``all(s is not None for s in
        # states)`` check would have set ``_kda_state = None``, throwing away
        # every layer's state because of one bad layer — a latent footgun for
        # any future KDA variant that conditionally returns None.
        if self.n_kda_layers > 0:
            if any(s is None for s in states):
                raise RuntimeError(
                    "HybridKCHAttention: a KDA layer returned new_state=None "
                    "despite output_final_state=True. Refusing to persist a "
                    "partial state stack (would silently drop other layers' "
                    "states). Check the KDA layer implementation."
                )
            self._kda_state = torch.stack(states, dim=0)
        else:
            self._kda_state = None
        return x

    def layout_str(self) -> str:
        return '-'.join(s.upper() for s in self.layout)
