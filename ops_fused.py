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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_kda import naive_recurrent_kda


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
    dropout: float = 0.0


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

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        B, T, d = x.shape
        cfg = self.cfg
        H, K, V, HV = cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v, cfg.n_heads_v
        # x: [B, T, d] -> [B, d, T]; left-pad by (k-1)=2, right-pad 0;
        # padding=0 conv keeps length T -> causal short-conv output.
        x_conv = F.pad(x.transpose(1, 2), (self.short_conv.kernel_size[0] - 1, 0))
        x_conv = self.short_conv(x_conv).transpose(1, 2)
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
        # log-space gate: low-rank down/up with a softplus-style decay
        g = -F.softplus(self.g_up(self.g_down(x_conv))).view(B, T, HV, K) * 0.1
        beta = torch.sigmoid(self.beta(x_conv))                   # [B, T, HV]
        o, new_state = naive_recurrent_kda(
            q, k, v, g, beta, scale=self.scale,
            initial_state=state, output_final_state=True,
        )
        return self.o_proj(o.reshape(B, T, HV * V)), new_state


class CSAHybridLayer(nn.Module):
    """A single CSA sub-layer (compression + sparse selection + MQA)."""

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

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        cfg = self.cfg
        o = naive_csa(
            x, self.W_aKV.weight.T, self.W_bKV.weight.T,
            self.W_aZ.weight.T, self.W_bZ.weight.T, self.Ba, self.Bb,
            self.W_DQ.weight.T, self.W_UQ.weight.T, self.W_IUQ.weight.T,
            self.W_w.weight.T, self.W_KV_idx.weight.T, self.W_Z_idx.weight.T,
            self.B_idx,
            m=cfg.csa_m, topk=cfg.csa_topk, nh=cfg.csa_nh, nIh=cfg.csa_nIh,
            c=cfg.csa_c, c_I=cfg.csa_cI, dc=cfg.csa_dc,
            sliding_window=cfg.csa_sliding_window, sink_logits=self.sink,
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
        self.total_layers = total_layers
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
        """Clear the persistent KDA recurrent state.

        Call this at the start of training, at the start of each sequence
        during evaluation, or between independent generation sessions.
        """
        # Assigning None to a registered buffer is supported by nn.Module
        # and keeps the buffer slot present (just empty) so a subsequent
        # model.to(device) / state_dict save still works correctly.
        self._kda_state = None

    def _build_layout(self) -> list[str]:
        # One repeating unit = n_kda KDA + n_csa CSA + n_hca HCA.
        unit = (['kda'] * self.cfg.n_kda
                + ['csa'] * self.cfg.n_csa
                + ['hca'] * self.cfg.n_hca)
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
        # old state is invalid and we drop it. We also drop it on a device
        # mismatch as a defensive measure (should not happen now that the
        # state is a registered buffer moved by .to(), but cheap to guard).
        stacked = self._kda_state
        if stacked is not None:
            # Batch dim is axis 1 (axis 0 is the per-layer index).
            if stacked.shape[1] != x.shape[0] or stacked.device != x.device:
                stacked = None
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
        if self.n_kda_layers > 0 and all(s is not None for s in states):
            self._kda_state = torch.stack(states, dim=0)
        else:
            self._kda_state = None
        return x

    def layout_str(self) -> str:
        return '-'.join(s.upper() for s in self.layout)
