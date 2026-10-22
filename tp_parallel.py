"""Tensor-parallel reference forward + prefill/decode latency ledger.

Why this module?
----------------
The repository pins the *numeric* contracts of every operator (KDA / CSA /
HCA) and separately prices their decode traffic and prefill FLOPs, but it
has no notion of **tensor parallelism (TP)**.  Serving the hybrid stack
with TP shards the value-head axis across ranks and then needs one
all-reduce per layer to reassemble the output projection.  Two questions
follow, and neither is answerable from the existing single-device
accounting:

1. **Correctness.** Does head sharding change the model?  MQAR accuracy
   is measured on the unsharded tower; if TP sharding altered logits, the
   measured recall would not transfer to a TP deployment.
2. **Latency.** TP divides weight/state traffic and FLOPs by the degree,
   but adds an all-reduce per layer.  Prefill (large activation messages,
   compute-bound) and decode (small messages, alpha-dominated, memory
   bound) respond very differently, so a single "TP speedup" number is
   meaningless.

This module supplies both:

* a **reference TP forward** for ``KDAHybridLayer`` / ``CSAHybridLayer`` /
  ``HCAHybridLayer`` and a whole ``HybridAttentionStack``.  Each "rank"
  computes a partial output; summing the partials is numerically equal to
  the unsharded layer (pinned by the regression suite).  KDA value heads
  (and the corresponding q/k GVA groups, gate rows, beta units and
  ``o_proj`` columns) are split, while the short conv / q,k projections /
  ``g_down`` are replicated.  CSA/HCA replicate the shared-KV compression
  and lightning indexer and split only the value heads, which is the
  standard MQA-under-TP layout (KV is replicated, query/output work is
  sharded).
* an **analytic prefill/decode latency model** over that exact sharding
  map: per-rank weights, per-rank decode state, FLOPs split into
  replicated and head-scaled terms, a ring all-reduce term
  (``2(P-1)/P · S/B + 2(P-1)·α``), and a sweep that reports the optimal TP
  degree separately for prefill and decode.

Scope and honesty
-----------------
* The TP forward is a **correctness reference**, not a distributed
  runtime.  It runs every simulated rank in one process and follows the
  layer's configured KDA backend (``cfg.kda_backend``; an explicit
  ``backend=`` argument overrides it for A/B checks).  Fresh-sequence
  stack checks use :func:`hybrid_stack_forward_tp`; streaming / incremental
  decoding of a KDA-only stack uses :func:`kda_stack_forward_tp`, which
  carries the per-head recurrent state and short-conv lookback across
  calls.  :func:`hybrid_stack_forward_tp_streaming` extends the same
  contract to a full hybrid stack by also carrying one shared-KV
  CSA/HCA decoding cache per layer.  These are the oracles a distributed
  kernel would be tested against.
* The latency model is analytic.  FLOPs reuse the canonical
  ``run_kv_cache.prefill_flops`` decomposition (a test pins that the
  replicated + sharded split sums to it exactly); decode bytes reuse
  ``decode_bandwidth.decode_traffic_breakdown`` / ``kv_budget``.  No
  Python/dispatch overhead, kernel inefficiency, or overlap between
  communication and compute is modelled.  All-reduce is added to the
  critical path, so the model is conservative on TP's behalf.
* MQA-side KV traffic is **replicated, not divided, by TP**: CSA/HCA
  compressed entries are shared across value heads, so every rank reads
  the same KV rows.  KDA recurrent state *is* head-scaled and therefore
  shrinks by the TP degree.  This asymmetry is the main decode-phase
  finding the ledger exposes, and it is why the streaming reference keeps
  ONE CSA/HCA cache per layer (shared by all ranks) rather than one per
  rank.
* Supported memory route: ``memory_type='kda'``.  Mamba-2 / RWKV-7 have
  their own state layouts and are rejected rather than silently priced
  with a wrong shard map.
"""
from __future__ import annotations

import argparse
import dataclasses
import math

import torch
import torch.nn.functional as F

import decode_bandwidth
from ops_csa import _qk_normalize, naive_csa
from ops_decoding_cache import CSADecodingCache, HCADecodingCache
from ops_fused import HybridConfig, build_mixer
from ops_hca import naive_hca
from ops_kda_backend import (kda_forward, resolve_kda_backend,
                             validate_kda_backend)
from run_kv_cache import (
    causal_block_entries,
    causal_selected_entries,
    prefill_flops,
)

#: Parameters that are sharded along the value-head axis.  Every other
#: parameter is replicated on each TP rank.  The names are the module
#: attribute paths exposed by ``build_mixer``; the regression suite pins
#: that this classification reproduces the real parameter counts.
SHARDED_PARAMETERS = {
    'kda': frozenset({'v_proj.weight', 'g_up.weight', 'beta.weight',
                      'o_proj.weight'}),
    'csa': frozenset({'W_UQ.weight', 'sink', 'o_proj.weight'}),
    'hca': frozenset({'W_UQ.weight', 'sink', 'o_proj.weight'}),
}

#: Head count a layer kind shards over.  CSA/HCA use their core-attention
#: head count, KDA uses the value-head count (``n_heads_v``).
HEAD_COUNT_FIELD = {
    'kda': 'n_heads_v',
    'csa': 'csa_nh',
    'hca': 'hca_nh',
}

SUPPORTED_MEMORY = ('kda',)


def _validate_tp_size(tp_size: int) -> None:
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        raise TypeError(
            f'tp_parallel: tp_size={tp_size!r} must be an int, got '
            f'{type(tp_size).__name__}.')
    if tp_size < 1:
        raise ValueError(
            f'tp_parallel: tp_size={tp_size} must be >= 1.')


def _validate_cfg(cfg: HybridConfig) -> None:
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'tp_parallel: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    memory_kinds = {k for k in decode_bandwidth.unit_layout(cfg)
                    if k in ('kda', 'mamba2', 'rwkv7')}
    if memory_kinds - set(SUPPORTED_MEMORY):
        raise NotImplementedError(
            f'tp_parallel: the TP shard map is defined only for the '
            f'KDA memory route; got memory layer kinds '
            f'{sorted(memory_kinds)}. Build a KDA-route config '
            f'(memory_type="kda") instead of pricing an unsupported '
            f'state layout.')
    for kind in decode_bandwidth.unit_layout(cfg):
        if kind not in SHARDED_PARAMETERS:
            raise NotImplementedError(
                f'tp_parallel: layer kind {kind!r} is not TP-shardable by '
                f'this reference; supported kinds are '
                f'{sorted(SHARDED_PARAMETERS)}.')


def head_ranges(n_heads: int, tp_size: int) -> list[tuple[int, int]]:
    """Contiguous, equal head slices for ``tp_size`` ranks.

    Tensor parallelism requires an even split: an uneven shard would make
    one rank the straggler and change the per-rank state geometry.  A
    non-divisible head count is rejected instead of padded, because a
    padded (masked) head still costs a full state slot and silently
    changes the memory ledger.
    """
    _validate_tp_size(tp_size)
    if isinstance(n_heads, bool) or not isinstance(n_heads, int):
        raise TypeError(f'head_ranges: n_heads={n_heads!r} must be an int.')
    if n_heads < 1:
        raise ValueError(f'head_ranges: n_heads={n_heads} must be >= 1.')
    if n_heads % tp_size:
        raise ValueError(
            f'head_ranges: n_heads={n_heads} must be divisible by '
            f'tp_size={tp_size} (no uneven shards / padding).')
    width = n_heads // tp_size
    return [(r * width, (r + 1) * width) for r in range(tp_size)]


def plan_layout(cfg: HybridConfig, total_layers: int | None = None) -> list[str]:
    """Resolved layer list the TP plan prices (one unit by default).

    Explicit layouts are exact (mirroring ``HybridAttentionStack``): a
    ``total_layers`` that disagrees with ``len(cfg.layout)`` is rejected
    instead of silently truncated / repeated.  Ratio mode repeats the
    ``n_kda:n_csa:n_hca`` unit to ``total_layers``.
    """
    _validate_cfg(cfg)
    unit = list(decode_bandwidth.unit_layout(cfg))
    if total_layers is None:
        return unit
    if isinstance(total_layers, bool) or not isinstance(total_layers, int):
        raise TypeError(
            f'plan_layout: total_layers={total_layers!r} must be an int.')
    if total_layers < 1:
        raise ValueError(
            f'plan_layout: total_layers={total_layers} must be >= 1.')
    if cfg.layout is not None:
        if len(unit) != total_layers:
            raise ValueError(
                f'plan_layout: cfg.layout has {len(unit)} layers but '
                f'total_layers={total_layers}; an explicit layout is exact.')
        return unit
    if not unit:
        return []
    layout = (unit * ((total_layers + len(unit) - 1) // len(unit)))[
        :total_layers]
    return layout


def validate_tp_size(cfg: HybridConfig, tp_size: int) -> dict:
    """Reject TP degrees that do not divide every used head axis."""
    _validate_cfg(cfg)
    _validate_tp_size(tp_size)
    layout = decode_bandwidth.unit_layout(cfg)
    used = set(layout)
    counts = {}
    for kind in sorted(used):
        if kind not in HEAD_COUNT_FIELD:
            raise NotImplementedError(
                f'validate_tp_size: kind {kind!r} has no head axis in the '
                f'TP plan.')
        n_heads = getattr(cfg, HEAD_COUNT_FIELD[kind])
        counts[kind] = n_heads
        if n_heads % tp_size:
            raise ValueError(
                f'tp_parallel: tp_size={tp_size} does not divide '
                f'{kind} head count {HEAD_COUNT_FIELD[kind]}={n_heads} '
                f'(remainder {n_heads % tp_size}). Choose a degree that '
                f'divides every head axis.')
    return {'tp_size': tp_size, 'heads_by_kind': counts,
            'heads_per_rank': {k: v // tp_size for k, v in counts.items()}}


def _layer_parameter_ledger(kind: str, cfg: HybridConfig,
                            tp_size: int) -> dict:
    """Exact per-layer parameter split measured from the real module."""
    rng_state = torch.random.get_rng_state()
    try:
        with torch.no_grad():
            layer = build_mixer(kind, cfg)
    finally:
        torch.random.set_rng_state(rng_state)
    sharded_names = SHARDED_PARAMETERS[kind]
    sharded = 0
    replicated = 0
    sharded_names_seen = set()
    for name, param in layer.named_parameters():
        n = int(param.numel())
        if name in sharded_names:
            if n % tp_size:
                raise ValueError(
                    f'tp_parallel: sharded parameter {kind}.{name} has '
                    f'{n} elements, not divisible by tp_size={tp_size}.')
            sharded += n
            sharded_names_seen.add(name)
        else:
            replicated += n
    missing = sharded_names - sharded_names_seen
    if missing:
        raise RuntimeError(
            f'tp_parallel: parameter(s) {sorted(missing)} expected in '
            f'{kind} were not found; the shard map is stale.')
    return {
        'kind': kind,
        'full_parameter_elements': sharded + replicated,
        'sharded_parameter_elements': sharded,
        'replicated_parameter_elements': replicated,
        'sharded_parameter_names': sorted(sharded_names),
    }


def tp_plan(cfg: HybridConfig, tp_size: int, *,
            total_layers: int | None = None, itemsize: int = 4) -> dict:
    """Static TP shard plan + exact per-rank parameter ledger.

    ``full = sharded + replicated`` and
    ``per_rank = sharded / tp_size + replicated`` hold by construction.
    Norm parameters (``2 * d_model`` per layer) are replicated.
    """
    meta = validate_tp_size(cfg, tp_size)
    if isinstance(itemsize, bool) or not isinstance(itemsize, int) \
            or itemsize < 1:
        raise ValueError(
            f'tp_plan: itemsize={itemsize!r} must be an int >= 1.')
    layout = plan_layout(cfg, total_layers)
    counts = {}
    for kind in layout:
        counts[kind] = counts.get(kind, 0) + 1
    by_kind = {kind: _layer_parameter_ledger(kind, cfg, tp_size)
               for kind in sorted(counts)}
    sharded = sum(counts[k] * v['sharded_parameter_elements']
                  for k, v in by_kind.items())
    replicated = sum(counts[k] * v['replicated_parameter_elements']
                     for k, v in by_kind.items())
    norm_elements = len(layout) * 2 * cfg.d_model
    full = sharded + replicated + norm_elements
    per_rank = sharded // tp_size + replicated + norm_elements
    for kind, entry in by_kind.items():
        entry['n_layers'] = counts[kind]
    return {
        'tp_size': tp_size,
        'layout': layout,
        'n_layers': len(layout),
        'heads_by_kind': meta['heads_by_kind'],
        'heads_per_rank': meta['heads_per_rank'],
        'head_ranges': {
            kind: head_ranges(cfg_n, tp_size)
            for kind, cfg_n in meta['heads_by_kind'].items()},
        'by_kind': by_kind,
        'sharded_parameter_elements': sharded,
        'replicated_parameter_elements': replicated,
        'norm_parameter_elements': norm_elements,
        'full_parameter_elements': full,
        'per_rank_parameter_elements': per_rank,
        'per_rank_weight_bytes': per_rank * itemsize,
        'full_weight_bytes': full * itemsize,
        'weight_shard_fraction': per_rank / full if full else 0.0,
        'itemsize': itemsize,
    }


# ---------------------------------------------------------------------------
# Reference TP forward
# ---------------------------------------------------------------------------
def _conv_features(layer, x: torch.Tensor,
                   conv_lookback: torch.Tensor | None):
    """Replicate ``KDAHybridLayer``'s causal short-conv preprocessing."""
    B, T, d = x.shape
    ksize = layer.short_conv.kernel_size[0]
    lookback = conv_lookback
    if lookback is not None:
        if lookback.shape[0] != B:
            lookback = None
        elif lookback.device != x.device or lookback.dtype != x.dtype:
            lookback = lookback.to(device=x.device, dtype=x.dtype)
    if lookback is None:
        x_conv_in = F.pad(x.transpose(1, 2), (ksize - 1, 0))
    else:
        x_conv_in = torch.cat(
            [lookback.transpose(1, 2), x.transpose(1, 2)], dim=2)
    x_conv = layer.short_conv(x_conv_in).transpose(1, 2).contiguous()
    if T >= ksize - 1:
        new_lookback = x[:, -(ksize - 1):]
    elif lookback is not None:
        new_lookback = torch.cat([lookback, x], dim=1)[:, -(ksize - 1):]
    else:
        pad = (ksize - 1) - T
        new_lookback = torch.cat(
            [torch.zeros(B, pad, d, device=x.device, dtype=x.dtype), x],
            dim=1)
    return x_conv, new_lookback.clone()


def kda_layer_forward_tp(
    layer,
    x: torch.Tensor,
    state_shard: torch.Tensor | None,
    tp_rank: int,
    tp_size: int,
    conv_lookback: torch.Tensor | None = None,
    *,
    backend: str | None = None,
) -> dict:
    """One rank's head-sharded KDA layer forward.

    Replicated: short conv, q/k projections, ``g_down``.  Sharded by the
    value-head axis: ``v_proj``, ``g_up``, ``beta``, the recurrent state
    and ``o_proj`` columns.  q/k are expanded by the GVA factor and then
    sliced to this rank's value-head range, so a rank boundary may fall
    inside one q/k head without changing the math.

    ``backend`` defaults to ``cfg.kda_backend`` so the TP oracle exercises
    the SAME kernel selection as the unsharded model.  Pass ``'reference'``
    explicitly to force the pure-PyTorch path; an invalid name raises via
    :func:`ops_kda_backend.validate_kda_backend`.
    """
    cfg = layer.cfg
    _validate_tp_size(tp_size)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(
            f'kda_layer_forward_tp: tp_rank={tp_rank} out of range for '
            f'tp_size={tp_size}.')
    requested_backend = cfg.kda_backend if backend is None else backend
    requested_backend = validate_kda_backend(requested_backend)
    resolved_backend = resolve_kda_backend(requested_backend)
    B, T, d = x.shape
    H, K, V, HV = (cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v,
                   cfg.n_heads_v)
    ranges = head_ranges(HV, tp_size)
    lo, hi = ranges[tp_rank]
    hv = hi - lo
    G = HV // H
    if T == 0:
        compute_dtype = (torch.float64 if x.dtype == torch.float64
                         else torch.float)
        if state_shard is not None and state_shard.shape[0] == B:
            new_state = state_shard.to(device=x.device,
                                       dtype=compute_dtype)
        else:
            new_state = torch.zeros(B, hv, K, V, device=x.device,
                                    dtype=compute_dtype)
        return {'output': x.new_zeros(B, 0, d), 'state': new_state,
                'conv_lookback': conv_lookback,
                'backend': requested_backend,
                'resolved_backend': resolved_backend}
    x_conv, new_lookback = _conv_features(layer, x, conv_lookback)
    q = F.normalize(
        F.silu(layer.q_proj(x_conv)).view(B, T, H, K), dim=-1)
    k = F.normalize(
        F.silu(layer.k_proj(x_conv)).view(B, T, H, K), dim=-1)
    q = q.repeat_interleave(G, dim=2)[:, :, lo:hi]
    k = k.repeat_interleave(G, dim=2)[:, :, lo:hi]
    v_weight = layer.v_proj.weight.view(HV, V, d)[lo:hi].reshape(hv * V, d)
    v = F.silu(F.linear(x_conv, v_weight)).view(B, T, hv, V)
    g_up_weight = layer.g_up.weight.view(HV, K, K)[lo:hi].reshape(hv * K, K)
    g = F.linear(layer.g_down(x_conv), g_up_weight).view(B, T, hv, K)
    if getattr(layer, '_decay_bias_active', False):
        # Match the unsharded layer: the prior is stored in fp32, but the
        # sharded projection can be bf16/fp16 under autocast, so cast the
        # slice before the add instead of promoting the gate and breaking
        # KDA's same-dtype contract.
        g = g + layer._decay_bias.view(HV, K)[lo:hi].to(dtype=g.dtype)
    g = -F.softplus(g) * cfg.kda_decay_scale
    beta = F.linear(x_conv, layer.beta.weight[lo:hi]).sigmoid()
    use_chunk = (cfg.kda_chunk_size >= 1 and T >= cfg.kda_chunk_size)
    o, new_state = kda_forward(
        q, k, v, g, beta,
        scale=layer.scale,
        initial_state=state_shard,
        output_final_state=True,
        chunk_size=cfg.kda_chunk_size,
        use_chunk=use_chunk,
        backend=requested_backend,
        recompute_attn=cfg.kda_recompute_attn,
        assoc_scan=cfg.kda_assoc_scan,
    )
    o_proj = layer.o_proj.weight[:, lo * V:hi * V]
    partial = F.linear(o.reshape(B, T, hv * V), o_proj)
    return {'output': partial, 'state': new_state,
            'conv_lookback': new_lookback,
            'backend': requested_backend,
            'resolved_backend': resolved_backend}


def csa_layer_forward_tp(
    layer,
    x: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> dict:
    """One rank's head-sharded CSA forward (shared KV + indexer replicated)."""
    cfg = layer.cfg
    _validate_tp_size(tp_size)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(
            f'csa_layer_forward_tp: tp_rank={tp_rank} out of range for '
            f'tp_size={tp_size}.')
    lo, hi = head_ranges(cfg.csa_nh, tp_size)[tp_rank]
    nh = hi - lo
    w_uq = layer.W_UQ.weight.view(
        cfg.csa_nh, cfg.csa_c, cfg.csa_dc)[lo:hi].reshape(
            nh * cfg.csa_c, cfg.csa_dc)
    o = naive_csa(
        x, layer.W_aKV.weight, layer.W_bKV.weight,
        layer.W_aZ.weight, layer.W_bZ.weight, layer.Ba, layer.Bb,
        layer.W_DQ.weight, w_uq, layer.W_IUQ.weight,
        layer.W_w.weight, layer.W_KV_idx.weight, layer.W_Z_idx.weight,
        layer.B_idx,
        m=cfg.csa_m, topk=cfg.csa_topk, nh=nh, nIh=cfg.csa_nIh,
        c=cfg.csa_c, c_I=cfg.csa_cI, dc=cfg.csa_dc,
        sliding_window=cfg.csa_sliding_window,
        sink_logits=layer.sink[lo:hi],
        rope_dim=cfg.csa_rope_dim,
        rope_base=cfg.rope_base,
        qk_norm_mode=cfg.csa_qk_norm_mode,
        use_ste=layer.use_ste,
        normalize_qk=True,
        indexer_chunk=cfg.csa_indexer_chunk,
    )
    o_proj = layer.o_proj.weight[:, lo * cfg.csa_c:hi * cfg.csa_c]
    return {'output': F.linear(o, o_proj), 'state': None}


def hca_layer_forward_tp(
    layer,
    x: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> dict:
    """One rank's head-sharded HCA forward (shared KV replicated)."""
    cfg = layer.cfg
    _validate_tp_size(tp_size)
    if not 0 <= tp_rank < tp_size:
        raise ValueError(
            f'hca_layer_forward_tp: tp_rank={tp_rank} out of range for '
            f'tp_size={tp_size}.')
    lo, hi = head_ranges(cfg.hca_nh, tp_size)[tp_rank]
    nh = hi - lo
    w_uq = layer.W_UQ.weight.view(
        cfg.hca_nh, cfg.hca_c, cfg.hca_dc)[lo:hi].reshape(
            nh * cfg.hca_c, cfg.hca_dc)
    o = naive_hca(
        x, layer.W_KV.weight, layer.W_Z.weight, layer.B_pos,
        layer.W_DQ.weight, w_uq,
        m2=cfg.hca_m2, nh=nh, c=cfg.hca_c, dc=cfg.hca_dc,
        sliding_window=cfg.hca_sliding_window,
        sink_logits=layer.sink[lo:hi],
        rope_dim=cfg.hca_rope_dim,
        rope_base=cfg.rope_base,
        qk_norm_mode=cfg.hca_qk_norm_mode,
    )
    o_proj = layer.o_proj.weight[:, lo * cfg.hca_c:hi * cfg.hca_c]
    return {'output': F.linear(o, o_proj), 'state': None}


def _split_linear_output(weight: torch.Tensor, o: torch.Tensor,
                        width: int, n_heads: int, tp_size: int) -> torch.Tensor:
    """Sum the per-rank slices of one sharded output projection."""
    partials = []
    for rank in range(tp_size):
        lo, hi = head_ranges(n_heads, tp_size)[rank]
        w = weight[:, lo * width:hi * width]
        partials.append(F.linear(o[..., lo * width:hi * width], w))
    return torch.stack(partials, dim=0).sum(dim=0)


def _csa_stream_project(layer, h: torch.Tensor):
    """Replicated CSA projections for the incremental cache path."""
    cfg = layer.cfg
    B, T, _ = h.shape
    combined = torch.cat(
        [layer.W_aKV.weight, layer.W_bKV.weight,
         layer.W_aZ.weight, layer.W_bZ.weight,
         layer.W_KV_idx.weight, layer.W_Z_idx.weight], dim=0)
    Ca, Cb, Za, Zb, K_idx, Z_idx = F.linear(h, combined).split(
        [cfg.csa_c, cfg.csa_c, cfg.csa_c, cfg.csa_c,
         cfg.csa_cI, cfg.csa_cI], dim=-1)
    cQ = F.linear(h, layer.W_DQ.weight)
    q = F.linear(cQ, layer.W_UQ.weight).view(B, T, cfg.csa_nh, cfg.csa_c)
    q_idx = F.linear(cQ, layer.W_IUQ.weight).view(
        B, T, cfg.csa_nIh, cfg.csa_cI)
    w_idx = F.linear(h, layer.W_w.weight)
    return Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx


def _hca_stream_project(layer, h: torch.Tensor):
    """Replicated HCA projections for the incremental cache path."""
    cfg = layer.cfg
    B, T, _ = h.shape
    C = F.linear(h, layer.W_KV.weight)
    Z = F.linear(h, layer.W_Z.weight)
    cQ = F.linear(h, layer.W_DQ.weight)
    q = F.linear(cQ, layer.W_UQ.weight).view(B, T, cfg.hca_nh, cfg.hca_c)
    return C, Z, q


def _ensure_csa_cache(layer, h: torch.Tensor,
                      cache: CSADecodingCache | None) -> CSADecodingCache:
    cfg = layer.cfg
    if (cache is None or cache.B != h.shape[0]
            or cache.device != h.device or cache.dtype != h.dtype):
        cache = CSADecodingCache(
            h.shape[0], cfg.csa_c, cfg.csa_cI, cfg.csa_m,
            cfg.csa_sliding_window, h.device, h.dtype,
            rope_dim=cfg.csa_rope_dim, rope_base=cfg.rope_base,
            qk_norm_mode=cfg.csa_qk_norm_mode)
    return cache


def _ensure_hca_cache(layer, h: torch.Tensor,
                      cache: HCADecodingCache | None) -> HCADecodingCache:
    cfg = layer.cfg
    if (cache is None or cache.B != h.shape[0]
            or cache.device != h.device or cache.dtype != h.dtype):
        cache = HCADecodingCache(
            h.shape[0], cfg.hca_c, cfg.hca_m2, cfg.hca_sliding_window,
            h.device, h.dtype,
            rope_dim=cfg.hca_rope_dim, rope_base=cfg.rope_base,
            qk_norm_mode=cfg.hca_qk_norm_mode)
    return cache


def csa_layer_forward_tp_streaming(layer, h: torch.Tensor,
                                   cache: CSADecodingCache | None,
                                   tp_size: int) -> dict:
    """One rank-summed CSA output for a streaming chunk.

    Shared-KV state (compressed blocks, indexer keys, sliding window) lives
    in ONE cache populated by the replicated branch.  A single-token call
    runs ``forward_step`` on each rank's value-head slice; multi-token
    chunks reuse the cache's vectorized ``forward_chunk`` and then split
    only the output projection, which is exactly the documented shard map.
    """
    cfg = layer.cfg
    _validate_tp_size(tp_size)
    validate_tp_size(cfg, tp_size)
    B, T, _ = h.shape
    out_dim = cfg.csa_nh * cfg.csa_c
    if T == 0:
        return {'output': h.new_zeros(B, 0, cfg.d_model), 'cache': cache}
    cache = _ensure_csa_cache(layer, h, cache)
    if T > 1 and cache.n_tokens_seen == 0:
        o, (Ca, Cb, Za, Zb, K_idx, Z_idx) = naive_csa(
            h, layer.W_aKV.weight, layer.W_bKV.weight,
            layer.W_aZ.weight, layer.W_bZ.weight, layer.Ba, layer.Bb,
            layer.W_DQ.weight, layer.W_UQ.weight, layer.W_IUQ.weight,
            layer.W_w.weight, layer.W_KV_idx.weight, layer.W_Z_idx.weight,
            layer.B_idx,
            m=cfg.csa_m, topk=cfg.csa_topk, nh=cfg.csa_nh,
            nIh=cfg.csa_nIh, c=cfg.csa_c, c_I=cfg.csa_cI,
            dc=cfg.csa_dc, sliding_window=cfg.csa_sliding_window,
            sink_logits=layer.sink, use_ste=False, normalize_qk=True,
            rope_dim=cfg.csa_rope_dim, rope_base=cfg.rope_base,
            qk_norm_mode=cfg.csa_qk_norm_mode, return_projections=True)
        cache.append_step(
            Ca.detach(), Cb.detach(), Za.detach(), Zb.detach(),
            K_idx.detach(), Z_idx.detach(),
            layer.Ba.detach(), layer.Bb.detach(), layer.B_idx.detach())
        return {'output': _split_linear_output(
            layer.o_proj.weight, o.reshape(B, T, out_dim),
            cfg.csa_c, cfg.csa_nh, tp_size), 'cache': cache}

    Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = _csa_stream_project(
        layer, h)
    compute_dtype = (torch.float64 if h.dtype == torch.float64
                     else torch.float)
    q_n = _qk_normalize(q.to(compute_dtype), cfg.csa_qk_norm_mode)
    if T == 1:
        cache.append_step(
            Ca.detach(), Cb.detach(), Za.detach(), Zb.detach(),
            K_idx.detach(), Z_idx.detach(),
            layer.Ba.detach(), layer.Bb.detach(), layer.B_idx.detach())
        partials = []
        for rank in range(tp_size):
            lo, hi = head_ranges(cfg.csa_nh, tp_size)[rank]
            nh = hi - lo
            o = cache.forward_step(
                q_n[:, :, lo:hi], q_idx.to(compute_dtype),
                w_idx.to(compute_dtype), topk=cfg.csa_topk, nh=nh,
                nIh=cfg.csa_nIh, scale=None,
                sink_logits=layer.sink[lo:hi], use_ste=False,
                normalize_qk=True)
            partials.append(F.linear(
                o.reshape(B, 1, nh * cfg.csa_c).to(h.dtype),
                layer.o_proj.weight[:, lo * cfg.csa_c:hi * cfg.csa_c]))
        return {'output': torch.stack(partials, dim=0).sum(dim=0),
                'cache': cache}

    o = cache.forward_chunk(
        Ca.detach(), Cb.detach(), Za.detach(), Zb.detach(),
        K_idx.detach(), Z_idx.detach(),
        layer.Ba.detach(), layer.Bb.detach(), layer.B_idx.detach(),
        q_n, q_idx.to(compute_dtype), w_idx.to(compute_dtype),
        topk=cfg.csa_topk, nh=cfg.csa_nh, nIh=cfg.csa_nIh, scale=None,
        sink_logits=layer.sink, use_ste=False, normalize_qk=True)
    return {'output': _split_linear_output(
        layer.o_proj.weight,
        o.reshape(B, T, out_dim).to(h.dtype),
        cfg.csa_c, cfg.csa_nh, tp_size), 'cache': cache}


def hca_layer_forward_tp_streaming(layer, h: torch.Tensor,
                                   cache: HCADecodingCache | None,
                                   tp_size: int) -> dict:
    """One rank-summed HCA output for a streaming chunk."""
    cfg = layer.cfg
    _validate_tp_size(tp_size)
    validate_tp_size(cfg, tp_size)
    B, T, _ = h.shape
    out_dim = cfg.hca_nh * cfg.hca_c
    if T == 0:
        return {'output': h.new_zeros(B, 0, cfg.d_model), 'cache': cache}
    cache = _ensure_hca_cache(layer, h, cache)
    if T > 1 and cache.n_tokens_seen == 0:
        o, (C, Z) = naive_hca(
            h, layer.W_KV.weight, layer.W_Z.weight, layer.B_pos,
            layer.W_DQ.weight, layer.W_UQ.weight,
            m2=cfg.hca_m2, nh=cfg.hca_nh, c=cfg.hca_c, dc=cfg.hca_dc,
            sliding_window=cfg.hca_sliding_window, sink_logits=layer.sink,
            rope_dim=cfg.hca_rope_dim, rope_base=cfg.rope_base,
            qk_norm_mode=cfg.hca_qk_norm_mode, return_projections=True)
        cache.append_step(C.detach(), Z.detach(), layer.B_pos.detach())
        return {'output': _split_linear_output(
            layer.o_proj.weight, o.reshape(B, T, out_dim),
            cfg.hca_c, cfg.hca_nh, tp_size), 'cache': cache}

    C, Z, q = _hca_stream_project(layer, h)
    compute_dtype = (torch.float64 if h.dtype == torch.float64
                     else torch.float)
    q_n = _qk_normalize(q.to(compute_dtype), cfg.hca_qk_norm_mode)
    if T == 1:
        cache.append_step(C.detach(), Z.detach(), layer.B_pos.detach())
        partials = []
        for rank in range(tp_size):
            lo, hi = head_ranges(cfg.hca_nh, tp_size)[rank]
            nh = hi - lo
            o = cache.forward_step(
                q_n[:, :, lo:hi], nh=nh, scale=None,
                sink_logits=layer.sink[lo:hi], normalize_qk=True)
            partials.append(F.linear(
                o.reshape(B, 1, nh * cfg.hca_c).to(h.dtype),
                layer.o_proj.weight[:, lo * cfg.hca_c:hi * cfg.hca_c]))
        return {'output': torch.stack(partials, dim=0).sum(dim=0),
                'cache': cache}

    o = cache.forward_chunk(
        C.detach(), Z.detach(), layer.B_pos.detach(), q_n,
        nh=cfg.hca_nh, scale=None, sink_logits=layer.sink)
    return {'output': _split_linear_output(
        layer.o_proj.weight,
        o.reshape(B, T, out_dim).to(h.dtype),
        cfg.hca_c, cfg.hca_nh, tp_size), 'cache': cache}


def layer_forward_tp(layer, kind: str, x: torch.Tensor, tp_rank: int,
                     tp_size: int, *,
                     state_shard: torch.Tensor | None = None,
                     conv_lookback: torch.Tensor | None = None,
                     backend: str | None = None) -> dict:
    """Dispatch one rank's forward for a layer kind."""
    if kind == 'kda':
        return kda_layer_forward_tp(
            layer, x, state_shard, tp_rank, tp_size, conv_lookback,
            backend=backend)
    if kind == 'csa':
        return csa_layer_forward_tp(layer, x, tp_rank, tp_size)
    if kind == 'hca':
        return hca_layer_forward_tp(layer, x, tp_rank, tp_size)
    raise NotImplementedError(
        f'tp_parallel: layer_forward_tp does not support kind {kind!r}.')


def hybrid_stack_forward_tp(model, x: torch.Tensor, tp_size: int, *,
                            backend: str | None = None) -> dict:
    """Fresh-sequence TP reference forward of a whole hybrid stack.

    Every simulated rank is executed locally and its partial mixer output
    is summed before the residual add.  Since LayerNorm and the residual
    stream are replicated, this is exactly the unsharded stack forward
    (up to fp reduction order), which is what makes MQAR logits / recall
    invariant to the TP degree.

    ``backend`` defaults to ``model.cfg.kda_backend`` (see
    :func:`kda_layer_forward_tp`).
    """
    _validate_tp_size(tp_size)
    _validate_cfg(model.cfg)
    for kind in model.layout:
        if kind not in SHARDED_PARAMETERS:
            raise NotImplementedError(
                f'hybrid_stack_forward_tp: layer kind {kind!r} is not '
                f'supported.')
    validate_tp_size(model.cfg, tp_size)
    requested_backend = _resolve_backend(model.cfg, backend)
    h = x
    for layer, norm, kind in zip(model.layers, model.norms, model.layout):
        mix_in = norm(h)
        partials = [layer_forward_tp(
            layer, kind, mix_in, rank, tp_size, backend=requested_backend
        )['output'] for rank in range(tp_size)]
        h = h + torch.stack(partials, dim=0).sum(dim=0)
    return {'output': h, 'tp_size': tp_size,
            'n_layers': len(model.layout),
            'backend': requested_backend,
            'resolved_backend': resolve_kda_backend(requested_backend)}


def _resolve_backend(cfg: HybridConfig, backend: str | None) -> str:
    """Resolve a per-call backend override against the configured one."""
    requested = cfg.kda_backend if backend is None else backend
    return validate_kda_backend(requested)


def kda_stack_forward_tp(
    model,
    x: torch.Tensor,
    tp_size: int,
    *,
    kda_state: torch.Tensor | None = None,
    conv_lookbacks: list[torch.Tensor | None] | None = None,
    backend: str | None = None,
) -> dict:
    """Streaming (incremental) TP forward for a KDA-only stack.

    This is the stateful counterpart of :func:`hybrid_stack_forward_tp`.
    It carries the KDA incremental rule across calls:

    * ``kda_state`` is the SAME stacked tensor used by
      ``HybridAttentionStack.forward_functional``, shape
      ``[n_layers, B, HV, K, V]``.  Each rank receives only its
      value-head slice ``state[:, lo:hi]``; the KDA recurrence never mixes
      heads, so a rank's shard is a complete recurrent state for its
      heads and **no state all-reduce is needed** (only the per-layer
      output all-reduce).
    * ``conv_lookbacks`` is the per-layer short-conv left context
      (``[B, ksize-1, d]`` or ``None``).  The short conv is replicated, so
      every rank uses the same lookback and returns the same new one.

    Only KDA layers are supported: CSA/HCA need their own KV/index caches
    to stream, which live in ``ops_decoding_cache`` and are outside this
    reference.  Use :func:`hybrid_stack_forward_tp` for a fresh hybrid
    sequence.

    Returns ``{'output', 'kda_state', 'conv_lookbacks', 'tp_size',
    'n_layers', 'backend'}``.  The returned state is in the full,
    unsharded head layout, so it can be fed back into this function or
    into the unsharded model's ``forward_functional`` interchange-ably.
    """
    _validate_tp_size(tp_size)
    _validate_cfg(model.cfg)
    if any(kind != 'kda' for kind in model.layout):
        raise NotImplementedError(
            'kda_stack_forward_tp: streaming TP is defined only for a '
            'KDA-only stack; CSA/HCA incremental state lives in '
            'ops_decoding_cache. Got layout '
            f'{list(model.layout)}.')
    validate_tp_size(model.cfg, tp_size)
    requested_backend = _resolve_backend(model.cfg, backend)
    cfg = model.cfg
    B, T, _ = x.shape
    n_layers = len(model.layout)
    HV, K, V = cfg.n_heads_v, cfg.head_dim_k, cfg.head_dim_v
    if kda_state is not None:
        if not isinstance(kda_state, torch.Tensor):
            raise TypeError(
                'kda_stack_forward_tp: kda_state must be a tensor of shape '
                f'[n_layers, B, HV, K, V], got {type(kda_state).__name__}.')
        want = (n_layers, B, HV, K, V)
        if tuple(kda_state.shape) != want:
            raise ValueError(
                'kda_stack_forward_tp: kda_state must have shape '
                f'{want} (n_layers, B, HV, K, V); got '
                f'{tuple(kda_state.shape)}. Drop the state when B or the '
                'stack geometry changes.')
    if conv_lookbacks is None:
        conv_lookbacks = [None] * n_layers
    elif len(conv_lookbacks) != n_layers:
        raise ValueError(
            'kda_stack_forward_tp: conv_lookbacks must have one entry per '
            f'layer ({n_layers}); got {len(conv_lookbacks)}.')
    if n_layers == 0:
        return {'output': x, 'kda_state': kda_state,
                'conv_lookbacks': list(conv_lookbacks),
                'tp_size': tp_size, 'n_layers': 0,
                'backend': requested_backend,
                'resolved_backend': resolve_kda_backend(requested_backend)}

    ranges = head_ranges(HV, tp_size)
    h = x
    new_states: list[torch.Tensor] = []
    new_lookbacks: list[torch.Tensor | None] = []
    for layer, norm, full_state, lookback in zip(
            model.layers, model.norms, _iter_state_layers(kda_state,
                                                          n_layers),
            conv_lookbacks):
        mix_in = norm(h)
        partials = []
        state_shards = []
        new_lookback = None
        for rank in range(tp_size):
            lo, hi = ranges[rank]
            state_shard = (None if full_state is None
                           else full_state[:, lo:hi])
            out = layer_forward_tp(
                layer, 'kda', mix_in, rank, tp_size,
                state_shard=state_shard,
                conv_lookback=lookback,
                backend=requested_backend,
            )
            partials.append(out['output'])
            state_shards.append(out['state'])
            if rank == 0:
                new_lookback = out['conv_lookback']
        h = h + torch.stack(partials, dim=0).sum(dim=0)
        new_states.append(torch.cat(state_shards, dim=1))
        new_lookbacks.append(new_lookback)
    return {'output': h, 'kda_state': torch.stack(new_states, dim=0),
            'conv_lookbacks': new_lookbacks,
            'tp_size': tp_size, 'n_layers': n_layers,
            'backend': requested_backend,
            'resolved_backend': resolve_kda_backend(requested_backend)}


def _iter_state_layers(kda_state: torch.Tensor | None, n_layers: int):
    """Yield each layer's full state slice (or ``None``) for the stack."""
    if kda_state is None:
        for _ in range(n_layers):
            yield None
    else:
        for idx in range(n_layers):
            yield kda_state[idx]


_STREAM_STATE_KEYS = ('kda_state', 'conv_lookbacks',
                      'csa_caches', 'hca_caches')


def hybrid_stack_forward_tp_streaming(
        model, x: torch.Tensor, tp_size: int, *,
        state: dict | None = None,
        backend: str | None = None) -> dict:
    """Incremental TP forward for a full KDA/CSA/HCA hybrid stack.

    This is the streaming counterpart of :func:`hybrid_stack_forward_tp`:
    it carries the KDA recurrent state / short-conv lookbacks AND the
    per-layer CSA/HCA decoding caches across calls, so a hybrid stack can
    be decoded token-by-token under the same head-sharded map.

    ``state`` is the dict returned by a previous call (``None`` for a
    fresh sequence). The caches are shared-KV replicas: every rank reads
    the same compressed blocks / sliding window, while attention is
    evaluated on that rank's value-head slice for a single-token step.
    A multi-token chunk uses the cache's vectorized ``forward_chunk`` and
    splits only ``o_proj``; this is the same shard map, just with the
    replicated attention factored out for speed.

    Returns ``{'output', 'state', 'kda_state', 'conv_lookbacks',
    'csa_caches', 'hca_caches', 'tp_size', 'n_layers', 'backend',
    'resolved_backend'}``.
    """
    _validate_tp_size(tp_size)
    _validate_cfg(model.cfg)
    for kind in model.layout:
        if kind not in SHARDED_PARAMETERS:
            raise NotImplementedError(
                f'hybrid_stack_forward_tp_streaming: layer kind {kind!r} '
                f'is not supported.')
    validate_tp_size(model.cfg, tp_size)
    requested_backend = _resolve_backend(model.cfg, backend)
    cfg = model.cfg
    B, T, _ = x.shape
    n_mem = sum(1 for kind in model.layout if kind == 'kda')
    n_csa = sum(1 for kind in model.layout if kind == 'csa')
    n_hca = sum(1 for kind in model.layout if kind == 'hca')
    HV, K, V = cfg.n_heads_v, cfg.head_dim_k, cfg.head_dim_v

    if state is None:
        kda_state = None
        conv_lookbacks = [None] * n_mem
        csa_caches: list[CSADecodingCache | None] = [None] * n_csa
        hca_caches: list[HCADecodingCache | None] = [None] * n_hca
    else:
        if not isinstance(state, dict):
            raise TypeError(
                'hybrid_stack_forward_tp_streaming: state must be the dict '
                f'returned by a previous call, got {type(state).__name__}.')
        missing = [k for k in _STREAM_STATE_KEYS if k not in state]
        if missing:
            raise ValueError(
                'hybrid_stack_forward_tp_streaming: state is missing '
                f'{missing}; expected keys {list(_STREAM_STATE_KEYS)}.')
        kda_state = state['kda_state']
        conv_lookbacks = state['conv_lookbacks']
        csa_caches = state['csa_caches']
        hca_caches = state['hca_caches']
        if kda_state is not None:
            if not isinstance(kda_state, torch.Tensor):
                raise TypeError(
                    'hybrid_stack_forward_tp_streaming: state["kda_state"] '
                    f'must be a tensor or None, got '
                    f'{type(kda_state).__name__}.')
            want = (n_mem, B, HV, K, V)
            if tuple(kda_state.shape) != want:
                raise ValueError(
                    'hybrid_stack_forward_tp_streaming: state["kda_state"] '
                    f'must have shape {want} (n_mem, B, HV, K, V); got '
                    f'{tuple(kda_state.shape)}. Drop the state when B or '
                    'the stack geometry changes.')
        if not isinstance(conv_lookbacks, (list, tuple)) \
                or len(conv_lookbacks) != n_mem:
            raise ValueError(
                'hybrid_stack_forward_tp_streaming: state["conv_lookbacks"] '
                f'must have one entry per KDA layer ({n_mem}).')
        if not isinstance(csa_caches, (list, tuple)) \
                or len(csa_caches) != n_csa:
            raise ValueError(
                'hybrid_stack_forward_tp_streaming: state["csa_caches"] '
                f'must have one entry per CSA layer ({n_csa}).')
        if not isinstance(hca_caches, (list, tuple)) \
                or len(hca_caches) != n_hca:
            raise ValueError(
                'hybrid_stack_forward_tp_streaming: state["hca_caches"] '
                f'must have one entry per HCA layer ({n_hca}).')
        conv_lookbacks = list(conv_lookbacks)
        csa_caches = list(csa_caches)
        hca_caches = list(hca_caches)
        for idx, cache in enumerate(csa_caches):
            if cache is not None and not isinstance(cache, CSADecodingCache):
                raise TypeError(
                    'hybrid_stack_forward_tp_streaming: '
                    f'state["csa_caches"][{idx}] must be a '
                    f'CSADecodingCache or None, got '
                    f'{type(cache).__name__}.')
        for idx, cache in enumerate(hca_caches):
            if cache is not None and not isinstance(cache, HCADecodingCache):
                raise TypeError(
                    'hybrid_stack_forward_tp_streaming: '
                    f'state["hca_caches"][{idx}] must be an '
                    f'HCADecodingCache or None, got '
                    f'{type(cache).__name__}.')

    h = x
    mem_idx = csa_idx = hca_idx = 0
    new_states: list[torch.Tensor] = []
    new_lookbacks: list[torch.Tensor | None] = []
    new_csa_caches: list[CSADecodingCache | None] = []
    new_hca_caches: list[HCADecodingCache | None] = []
    ranges = head_ranges(HV, tp_size)
    for layer, norm, kind in zip(model.layers, model.norms, model.layout):
        mix_in = norm(h)
        if kind == 'kda':
            full_state = (None if kda_state is None
                          else kda_state[mem_idx])
            partials = []
            state_shards = []
            new_lookback = None
            for rank in range(tp_size):
                lo, hi = ranges[rank]
                shard = None if full_state is None else full_state[:, lo:hi]
                out = layer_forward_tp(
                    layer, 'kda', mix_in, rank, tp_size,
                    state_shard=shard,
                    conv_lookback=conv_lookbacks[mem_idx],
                    backend=requested_backend)
                partials.append(out['output'])
                state_shards.append(out['state'])
                if rank == 0:
                    new_lookback = out['conv_lookback']
            h = h + torch.stack(partials, dim=0).sum(dim=0)
            new_states.append(torch.cat(state_shards, dim=1))
            new_lookbacks.append(new_lookback)
            mem_idx += 1
        elif kind == 'csa':
            out = csa_layer_forward_tp_streaming(
                layer, mix_in, csa_caches[csa_idx], tp_size)
            h = h + out['output']
            new_csa_caches.append(out['cache'])
            csa_idx += 1
        else:
            out = hca_layer_forward_tp_streaming(
                layer, mix_in, hca_caches[hca_idx], tp_size)
            h = h + out['output']
            new_hca_caches.append(out['cache'])
            hca_idx += 1

    new_kda_state = (torch.stack(new_states, dim=0)
                     if n_mem else None)
    stream_state = {
        'kda_state': new_kda_state,
        'conv_lookbacks': new_lookbacks,
        'csa_caches': new_csa_caches,
        'hca_caches': new_hca_caches,
    }
    return {
        'output': h,
        'state': stream_state,
        'kda_state': new_kda_state,
        'conv_lookbacks': new_lookbacks,
        'csa_caches': new_csa_caches,
        'hca_caches': new_hca_caches,
        'tp_size': tp_size,
        'n_layers': len(model.layout),
        'backend': requested_backend,
        'resolved_backend': resolve_kda_backend(requested_backend),
    }


# ---------------------------------------------------------------------------
# Prefill / decode FLOPs split
# ---------------------------------------------------------------------------
def _geo_from_cfg(cfg: HybridConfig) -> dict:
    """Map a config onto ``run_kv_cache.prefill_flops`` kwargs."""
    return dict(
        d=cfg.d_model, H=cfg.n_heads_qk, K=cfg.head_dim_k,
        V=cfg.head_dim_v, kda_hv=cfg.n_heads_v,
        kda_k=cfg.head_dim_k, kda_v=cfg.head_dim_v,
        kda_conv_ksize=3, kda_chunk_size=cfg.kda_chunk_size,
        csa_m=cfg.csa_m, csa_topk=cfg.csa_topk, csa_c=cfg.csa_c,
        csa_dc=cfg.csa_dc, csa_cI=cfg.csa_cI, csa_nIh=cfg.csa_nIh,
        csa_nh=cfg.csa_nh, csa_sliding_window=cfg.csa_sliding_window,
        hca_m2=cfg.hca_m2, hca_c=cfg.hca_c, hca_dc=cfg.hca_dc,
        hca_nh=cfg.hca_nh, hca_sliding_window=cfg.hca_sliding_window,
    )


def layer_flops_split(cfg: HybridConfig, kind: str, T: int) -> dict:
    """Split one layer's prefill FLOPs into replicated / head-scaled work.

    The split mirrors the sharding map used by the TP forward: q/k
    projections, ``g_down`` and the short conv are replicated for KDA;
    CSA/HCA compression, the lightning indexer and the indexer-side
    projections are replicated.  Everything that scales with the value
    heads is sharded.  ``replicated + sharded`` equals
    ``run_kv_cache.prefill_flops`` exactly (pinned by a test), so the
    latency model cannot drift from the canonical FLOP accounting.
    """
    if isinstance(T, bool) or not isinstance(T, int) or T < 0:
        raise ValueError(
            f'layer_flops_split: T={T!r} must be a non-negative int.')
    cfg_d, H, K, V = cfg.d_model, cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v
    HV = cfg.n_heads_v
    if kind == 'kda':
        q_proj = 2 * T * cfg_d * H * K
        k_proj = q_proj
        v_proj = 2 * T * cfg_d * HV * V
        g_down = 2 * T * cfg_d * K
        g_up = 2 * T * K * HV * K
        beta = 2 * T * cfg_d * HV
        short_conv = 2 * T * cfg_d * 3
        recurrent = 2 * 3 * T * HV * K * V
        chunk_inter = 3 * T * HV * cfg.kda_chunk_size * K
        chunk_a = T * HV * cfg.kda_chunk_size * cfg.kda_chunk_size
        out_proj = 2 * T * HV * V * cfg_d
        sharded = v_proj + g_up + beta + recurrent + chunk_inter + chunk_a \
            + out_proj
        replicated = q_proj + k_proj + g_down + short_conv
        components = {
            'q_proj': q_proj, 'k_proj': k_proj, 'v_proj': v_proj,
            'g_down': g_down, 'g_up': g_up, 'beta': beta,
            'short_conv': short_conv, 'recurrent': recurrent,
            'chunk_inter': chunk_inter, 'chunk_a': chunk_a,
            'out_proj': out_proj,
        }
    elif kind == 'csa':
        compress = 2 * T * cfg_d * (4 * cfg.csa_c + 2 * cfg.csa_cI)
        w_dq = 2 * T * cfg_d * cfg.csa_dc
        w_uq = 2 * T * cfg.csa_dc * cfg.csa_c * cfg.csa_nh
        w_iuq = 2 * T * cfg.csa_dc * cfg.csa_cI * cfg.csa_nIh
        w_w = 2 * T * cfg_d * cfg.csa_nIh
        block_entries = causal_block_entries(T, cfg.csa_m)
        indexer = (2 * block_entries * cfg.csa_cI * cfg.csa_nIh
                   + 2 * block_entries * cfg.csa_nIh)
        total_sel = causal_selected_entries(T, cfg.csa_m, cfg.csa_topk)
        eff_topk = total_sel / T if T else 0.0
        core = 2 * T * eff_topk * cfg.csa_c * cfg.csa_nh * 2
        eff_sw = min(T, cfg.csa_sliding_window)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * cfg.csa_c * cfg.csa_nh * 2
        out_proj = 2 * T * cfg.csa_nh * cfg.csa_c * cfg_d
        sharded = w_uq + core + sw + out_proj
        replicated = compress + w_dq + w_iuq + w_w + indexer
        components = {
            'compress': compress, 'w_dq': w_dq, 'w_uq': w_uq,
            'w_iuq': w_iuq, 'w_w': w_w, 'indexer': indexer,
            'core': core, 'sliding_window': sw, 'out_proj': out_proj,
        }
    elif kind == 'hca':
        compress = 2 * T * cfg_d * cfg.hca_c * 2
        w_dq = 2 * T * cfg_d * cfg.hca_dc
        w_uq = 2 * T * cfg.hca_dc * cfg.hca_c * cfg.hca_nh
        block_entries = causal_block_entries(T, cfg.hca_m2)
        core = 2 * block_entries * cfg.hca_c * cfg.hca_nh * 2
        eff_sw = min(T, cfg.hca_sliding_window)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * cfg.hca_c * cfg.hca_nh * 2
        out_proj = 2 * T * cfg.hca_nh * cfg.hca_c * cfg_d
        sharded = w_uq + core + sw + out_proj
        replicated = compress + w_dq
        components = {
            'compress': compress, 'w_dq': w_dq, 'w_uq': w_uq,
            'core': core, 'sliding_window': sw, 'out_proj': out_proj,
        }
    else:
        raise NotImplementedError(
            f'layer_flops_split: kind {kind!r} is not supported.')
    total = replicated + sharded
    canonical = float(prefill_flops(kind, T, **_geo_from_cfg(cfg)))
    if abs(total - canonical) > max(1e-6, 1e-9 * abs(canonical)):
        raise RuntimeError(
            f'layer_flops_split: {kind} split total {total} does not match '
            f'the canonical prefill_flops {canonical}; the FLOP model and '
            f'the sharding map have diverged.')
    return {'kind': kind, 'T': T, 'replicated': replicated,
            'sharded': sharded, 'total': total,
            'canonical_prefill_flops': canonical,
            'components': components}


def _rank_flops(cfg: HybridConfig, kind: str, T: int, tp_size: int,
                cache: dict | None = None) -> float:
    key = (kind, T)
    if cache is not None and key in cache:
        split = cache[key]
    else:
        split = layer_flops_split(cfg, kind, T)
        if cache is not None:
            cache[key] = split
    return float(split['replicated']) + float(split['sharded']) / tp_size


# ---------------------------------------------------------------------------
# Decode traffic under TP
# ---------------------------------------------------------------------------
def tp_decode_traffic(cfg: HybridConfig, T: int, tp_size: int, *,
                      itemsize: int = 4) -> dict:
    """Per-rank per-token decode traffic after head sharding.

    KDA's recurrent state is head-scaled, so its reads/writes divide by
    the TP degree.  CSA/HCA compressed entries are shared across value
    heads (MQA), so every rank reads the same KV rows: their traffic is
    replicated and does **not** shrink with TP.
    """
    _validate_cfg(cfg)
    _validate_tp_size(tp_size)
    if isinstance(itemsize, bool) or not isinstance(itemsize, int) \
            or itemsize < 1:
        raise ValueError(
            f'tp_decode_traffic: itemsize={itemsize!r} must be an int >= 1.')
    traffic = decode_bandwidth.decode_traffic_breakdown(
        cfg, T, itemsize=itemsize)
    mem_read = traffic['read_bytes_by_op']['memory'] / tp_size
    csa_read = traffic['read_bytes_by_op']['csa']
    hca_read = traffic['read_bytes_by_op']['hca']
    mem_write = traffic['write_bytes_by_op']['memory'] / tp_size
    csa_write = traffic['write_bytes_by_op']['csa']
    hca_write = traffic['write_bytes_by_op']['hca']
    read_rank = mem_read + csa_read + hca_read
    write_rank = mem_write + csa_write + hca_write
    full_read = traffic['read_bytes_per_token']
    full_write = traffic['write_bytes_per_token']
    return {
        'T': int(T),
        'tp_size': tp_size,
        'itemsize': itemsize,
        'memory_route': traffic['memory_route'],
        'read_bytes_by_op_per_rank': {
            'memory': mem_read, 'csa': csa_read, 'hca': hca_read},
        'write_bytes_by_op_per_rank': {
            'memory': mem_write, 'csa': csa_write, 'hca': hca_write},
        'read_bytes_per_token_per_rank': read_rank,
        'write_bytes_per_token_per_rank': write_rank,
        'state_bytes_per_token_per_rank': read_rank + write_rank,
        'full_read_bytes_per_token': full_read,
        'full_write_bytes_per_token': full_write,
        'full_state_bytes_per_token': full_read + full_write,
        'kv_replicated_bytes_per_token': csa_read + hca_read
        + csa_write + hca_write,
    }


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class HardwareModel:
    """Analytic hardware anchors for the TP latency model.

    ``peak_tflops`` / ``peak_bw_gbps`` are *achieved* dense fp32 rates (the
    same convention as ``roofline``).  ``link_bw_gbps`` is the one-way
    inter-rank bandwidth used by the ring all-reduce; ``allreduce_latency_us``
    is the per-ring-step launch/synchronization latency.  Defaults are
    nominal H100-class values, NOT a benchmark claim.
    """
    peak_tflops: float = 60.0
    peak_bw_gbps: float = 3000.0
    link_bw_gbps: float = 300.0
    allreduce_latency_us: float = 5.0
    itemsize: int = 4
    gpu_anchored: bool = False

    def __post_init__(self):
        for name in ('peak_tflops', 'peak_bw_gbps', 'link_bw_gbps'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f'HardwareModel.{name} must be a number, got '
                    f'{type(value).__name__}.')
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f'HardwareModel.{name}={value} must be finite and > 0.')
        if not isinstance(self.allreduce_latency_us, (int, float)) \
                or isinstance(self.allreduce_latency_us, bool) \
                or not math.isfinite(self.allreduce_latency_us) \
                or self.allreduce_latency_us < 0:
            raise ValueError(
                f'HardwareModel.allreduce_latency_us='
                f'{self.allreduce_latency_us} must be finite and >= 0.')
        if isinstance(self.itemsize, bool) or not isinstance(self.itemsize, int) \
                or self.itemsize < 1:
            raise ValueError(
                f'HardwareModel.itemsize={self.itemsize!r} must be an '
                f'int >= 1.')
        if not isinstance(self.gpu_anchored, bool):
            raise TypeError(
                f'HardwareModel.gpu_anchored must be a bool, got '
                f'{type(self.gpu_anchored).__name__}.')

    @classmethod
    def from_gpu(cls, mode: str = 'auto') -> 'HardwareModel':
        """Build a HardwareModel anchored to a real CUDA device when available.

        ``mode='nominal'`` always returns the H100-class defaults without
        touching ``torch.cuda``. ``mode='auto'`` inspects ``torch.cuda``:
        when CUDA is available the device's reported memory clock and bus
        width anchor ``peak_bw_gbps`` and ``gpu_anchored=True``; when CUDA is
        unavailable (CPU hosts, CI) it returns the nominal defaults with
        ``gpu_anchored=False``.  ``gpu_anchored`` stays ``False`` when the
        device does not report the clock / bus width, so the flag never
        overstates the provenance of the returned model.

        ``peak_tflops`` and ``link_bw_gbps`` are deliberately left nominal:
        ``torch.cuda`` exposes neither per-SM core counts (needed for an
        honest fp32 peak) nor interconnect bandwidth, so deriving them would
        require architecture-specific guesses. The model therefore stays
        analytic and conservative — a real deployment should measure its own
        peaks and pass a constructed :class:`HardwareModel` to
        :func:`tp_latency` / :func:`tp_sweep`.
        """
        if mode not in ('auto', 'nominal'):
            raise ValueError(
                f"HardwareModel.from_gpu: mode={mode!r} must be 'auto' or "
                "'nominal'.")
        if mode == 'nominal' or not torch.cuda.is_available():
            return cls()
        props = torch.cuda.get_device_properties(0)
        peak_bw = float(cls().peak_bw_gbps)
        # NOTE: torch exposes the bus width as ``memory_bus_width`` on
        # ``_CudaDeviceProperties`` (there is no ``bus_width`` attribute);
        # reading the wrong name silently disabled the anchor and left the
        # nominal H100 bandwidth in place on every real GPU.
        mem_clock = getattr(props, 'memory_clock_rate', None)
        bus_width = getattr(props, 'memory_bus_width', None)
        anchored = False
        if mem_clock is not None and bus_width is not None and bus_width > 0:
            # GB/s = (kHz * 1e3 cycles/s) * (bits / 8 bytes/bit) / 1e9.
            peak_bw = float(mem_clock) * 1e3 * float(bus_width) / 8.0 / 1e9
            if not math.isfinite(peak_bw) or peak_bw <= 0:
                peak_bw = float(cls().peak_bw_gbps)
            else:
                anchored = True
        # ``gpu_anchored`` is a provenance flag: only claim the model is
        # device-anchored when the device actually supplied the numbers.
        return cls(peak_bw_gbps=peak_bw, gpu_anchored=anchored)


def ring_allreduce_ms(nbytes: float, tp_size: int,
                      hardware: HardwareModel) -> float:
    """Ring all-reduce critical-path time for one message (milliseconds).

    Each rank sends ``2(P-1)/P`` of the message; the latency term is
    ``2(P-1)`` ring steps.  ``P == 1`` has no communication.
    """
    _validate_tp_size(tp_size)
    if nbytes < 0:
        raise ValueError(
            f'ring_allreduce_ms: nbytes={nbytes} must be >= 0.')
    if tp_size == 1 or nbytes == 0:
        return 0.0
    volume = 2.0 * (tp_size - 1) / tp_size * nbytes
    bandwidth_ms = volume / (hardware.link_bw_gbps * 1e9) * 1e3
    latency_ms = 2.0 * (tp_size - 1) * hardware.allreduce_latency_us * 1e-3
    return bandwidth_ms + latency_ms


def tp_latency(cfg: HybridConfig, *, tp_size: int, prefill_len: int,
               decode_batch: int = 1, total_layers: int | None = None,
               prefill_batch: int = 1,
               decode_context_len: int | None = None,
               hardware: HardwareModel | None = None) -> dict:
    """Analytic prefill + per-decode-step latency at one TP degree.

    Prefill: ``max(FLOPs/P_eff / peak, weights/P / bw) + Σ all-reduce``.
    Decode: ``max(batch·FLOPs_tok/P_eff / peak,
    (weights/P + batch·state/P) / bw) + Σ all-reduce``.  Communication is
    serialized with compute (no overlap), which is deliberately
    conservative for TP.
    """
    _validate_cfg(cfg)
    _validate_tp_size(tp_size)
    hw = hardware or HardwareModel()
    if not isinstance(hw, HardwareModel):
        raise TypeError(
            f'tp_latency: hardware must be a HardwareModel, got '
            f'{type(hw).__name__}.')
    if isinstance(prefill_len, bool) or not isinstance(prefill_len, int) \
            or prefill_len < 0:
        raise ValueError(
            f'tp_latency: prefill_len={prefill_len!r} must be a '
            f'non-negative int.')
    if isinstance(prefill_batch, bool) or not isinstance(prefill_batch, int) \
            or prefill_batch < 1:
        raise ValueError(
            f'tp_latency: prefill_batch={prefill_batch!r} must be an '
            f'int >= 1.')
    if isinstance(decode_batch, bool) or not isinstance(decode_batch, int) \
            or decode_batch < 1:
        raise ValueError(
            f'tp_latency: decode_batch={decode_batch!r} must be an int >= 1.')
    context_len = prefill_len if decode_context_len is None \
        else decode_context_len
    if isinstance(context_len, bool) or not isinstance(context_len, int) \
            or context_len < 0:
        raise ValueError(
            f'tp_latency: decode_context_len={context_len!r} must be a '
            f'non-negative int.')

    plan = tp_plan(cfg, tp_size, total_layers=total_layers,
                   itemsize=hw.itemsize)
    layout = plan['layout']
    flop_cache: dict = {}
    prefill_flops_rank = 0.0
    full_prefill_flops = 0.0
    prefill_comm_ms = 0.0
    message_bytes = prefill_batch * prefill_len * cfg.d_model * hw.itemsize
    for kind in layout:
        # Each sequence in the prefill batch does the same per-layer work;
        # the all-reduce message already includes the batch (below), so
        # the compute and full-FLOPs fields must scale with it too.
        prefill_flops_rank += prefill_batch * _rank_flops(
            cfg, kind, prefill_len, tp_size, flop_cache)
        full_prefill_flops += prefill_batch * layer_flops_split(
            cfg, kind, prefill_len)['total']
        prefill_comm_ms += ring_allreduce_ms(message_bytes, tp_size, hw)
    prefill_compute_ms = (prefill_flops_rank / (hw.peak_tflops * 1e12)) * 1e3
    prefill_memory_ms = (plan['per_rank_weight_bytes']
                         / (hw.peak_bw_gbps * 1e9)) * 1e3
    prefill_core_ms = max(prefill_compute_ms, prefill_memory_ms)
    prefill_total_ms = prefill_core_ms + prefill_comm_ms

    traffic = tp_decode_traffic(cfg, context_len, tp_size,
                                itemsize=hw.itemsize)
    decode_flops_per_token = 0.0
    for kind in layout:
        if context_len == 0:
            continue
        decode_flops_per_token += _rank_flops(
            cfg, kind, context_len, tp_size, flop_cache) / context_len
    decode_compute_ms = (
        decode_batch * decode_flops_per_token
        / (hw.peak_tflops * 1e12)) * 1e3
    decode_step_bytes = (plan['per_rank_weight_bytes']
                         + decode_batch
                         * traffic['state_bytes_per_token_per_rank'])
    decode_memory_ms = (decode_step_bytes
                        / (hw.peak_bw_gbps * 1e9)) * 1e3
    decode_core_ms = max(decode_compute_ms, decode_memory_ms)
    decode_message_bytes = decode_batch * cfg.d_model * hw.itemsize
    decode_comm_ms = len(layout) * ring_allreduce_ms(
        decode_message_bytes, tp_size, hw)
    decode_total_ms = decode_core_ms + decode_comm_ms

    def _bound(core_ms, compute_ms, memory_ms, comm_ms):
        if comm_ms > core_ms:
            return 'communication'
        return 'compute' if compute_ms >= memory_ms else 'memory'

    return {
        'tp_size': tp_size,
        'layout': layout,
        'n_layers': len(layout),
        'workload': {
            'prefill_len': prefill_len,
            'prefill_batch': prefill_batch,
            'decode_context_len': context_len,
            'decode_batch': decode_batch,
        },
        'hardware': dataclasses.asdict(hw),
        'prefill': {
            'total_ms': prefill_total_ms,
            'core_ms': prefill_core_ms,
            'compute_ms': prefill_compute_ms,
            'memory_ms': prefill_memory_ms,
            'comm_ms': prefill_comm_ms,
            'comm_share': (prefill_comm_ms / prefill_total_ms
                           if prefill_total_ms else 0.0),
            'bound': _bound(prefill_core_ms, prefill_compute_ms,
                            prefill_memory_ms, prefill_comm_ms),
            'flops_per_rank': prefill_flops_rank,
            'full_flops': full_prefill_flops,
        },
        'decode': {
            'total_ms': decode_total_ms,
            'core_ms': decode_core_ms,
            'compute_ms': decode_compute_ms,
            'memory_ms': decode_memory_ms,
            'comm_ms': decode_comm_ms,
            'comm_share': (decode_comm_ms / decode_total_ms
                           if decode_total_ms else 0.0),
            'bound': _bound(decode_core_ms, decode_compute_ms,
                            decode_memory_ms, decode_comm_ms),
            'flops_per_token_per_rank': decode_flops_per_token,
            'tokens_per_s': (decode_batch * 1e3 / decode_total_ms
                             if decode_total_ms else 0.0),
            'weight_bytes_per_step': plan['per_rank_weight_bytes'],
            'state_bytes_per_sequence': traffic[
                'state_bytes_per_token_per_rank'],
            'state_bytes_per_step': (
                decode_batch * traffic['state_bytes_per_token_per_rank']),
            'step_bytes': decode_step_bytes,
            'kv_replicated_bytes_per_sequence': traffic[
                'kv_replicated_bytes_per_token'],
        },
        'plan': plan,
    }


def tp_sweep(cfg: HybridConfig, tp_sizes, **kwargs) -> dict:
    """Latency rows for several TP degrees + per-phase optimum.

    ``speedup_vs_tp1`` is relative to the TP=1 row (computed even when 1 is
    not in ``tp_sizes``).  Optima are separate for prefill and decode
    because the two phases have opposite communication geometry.
    """
    sizes = tuple(int(s) for s in tp_sizes)
    if not sizes:
        raise ValueError('tp_sweep: tp_sizes must be non-empty.')
    if any(s < 1 for s in sizes):
        raise ValueError(
            f'tp_sweep: tp_sizes must be >= 1, got {sizes!r}.')
    baseline = tp_latency(cfg, tp_size=1, **kwargs)
    rows = []
    for size in dict.fromkeys(sizes):
        row = tp_latency(cfg, tp_size=size, **kwargs)
        row['speedup_vs_tp1'] = {
            'prefill': (baseline['prefill']['total_ms']
                        / row['prefill']['total_ms']
                        if row['prefill']['total_ms'] else None),
            'decode': (baseline['decode']['total_ms']
                       / row['decode']['total_ms']
                       if row['decode']['total_ms'] else None),
        }
        rows.append(row)
    prefill_best = min(rows, key=lambda r: r['prefill']['total_ms'])
    decode_best = min(rows, key=lambda r: r['decode']['total_ms'])
    return {
        'rows': rows,
        'prefill_optimum_tp': prefill_best['tp_size'],
        'decode_optimum_tp': decode_best['tp_size'],
        'baseline_tp1': baseline,
        'workload': baseline['workload'],
        'hardware': baseline['hardware'],
    }


def mqar_probe_config() -> HybridConfig:
    """The Exp 4/5 MQAR probe geometry (3:1:1, d_model=32).

    Mirrors ``run_ablation._make_cfg(32, (3, 1, 1))`` without importing
    the experiment runner.  ``n_heads_v = csa_nh = hca_nh = 2``, so the
    maximum TP degree for this geometry is 2.
    """
    return HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=32, csa_dc=64,
        csa_nIh=2, csa_cI=16, csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=32, hca_dc=64, hca_sliding_window=4,
        n_kda=3, n_csa=1, n_hca=1,
    )


def adjacent_probe_config() -> HybridConfig:
    """A TP-8-capable KDA-route config with the paper's head geometry."""
    return HybridConfig(
        d_model=512, n_heads_qk=8, n_heads_v=8,
        head_dim_k=64, head_dim_v=64,
        csa_m=16, csa_topk=32, csa_nh=8, csa_c=128, csa_dc=128,
        csa_nIh=4, csa_cI=32, csa_sliding_window=64,
        hca_m2=64, hca_nh=8, hca_c=128, hca_dc=128,
        hca_sliding_window=64,
        n_kda=3, n_csa=1, n_hca=1,
    )


def _print_table(sweep: dict) -> None:
    print(f"{'TP':>3} | {'prefill ms':>10} | {'p comm%':>7} | "
          f"{'p bound':>13} | {'decode ms':>9} | {'d comm%':>7} | "
          f"{'d bound':>13} | {'tok/s':>9}")
    print('-' * 96)
    for row in sweep['rows']:
        p, d = row['prefill'], row['decode']
        print(f"{row['tp_size']:>3} | {p['total_ms']:>10.3f} | "
              f"{100 * p['comm_share']:>7.1f} | {p['bound']:>13} | "
              f"{d['total_ms']:>9.3f} | {100 * d['comm_share']:>7.1f} | "
              f"{d['bound']:>13} | {d['tokens_per_s']:>9.2f}")
    print(f"\nprefill optimum TP={sweep['prefill_optimum_tp']}, "
          f"decode optimum TP={sweep['decode_optimum_tp']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Tensor-parallel reference forward check + analytic '
                    'prefill/decode latency sweep for the KDA/CSA/HCA stack.')
    parser.add_argument('--config', choices=('mqar', 'paper'),
                        default='paper',
                        help='mqar = Exp 4/5 probe geometry (TP<=2); '
                             'paper = 8-head KDA route (TP<=8).')
    parser.add_argument('--tp', default='1,2,4,8')
    parser.add_argument('--prefill-len', type=int, default=2048)
    parser.add_argument('--decode-batch', type=int, default=16)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--peak-tflops', type=float, default=60.0)
    parser.add_argument('--peak-bw-gbps', type=float, default=3000.0)
    parser.add_argument('--link-bw-gbps', type=float, default=300.0)
    parser.add_argument('--allreduce-latency-us', type=float, default=5.0)
    parser.add_argument('--hardware', choices=('auto', 'nominal'),
                        default='auto',
                        help='auto = anchor HardwareModel to a real CUDA '
                             'device when available (CPU/CI falls back to '
                             'the nominal H100-class defaults); nominal = '
                             'always use the built-in defaults. Individual '
                             '--peak-* / --link-* / --allreduce-* flags '
                             'override the chosen anchor.')
    args = parser.parse_args(argv)
    cfg = mqar_probe_config() if args.config == 'mqar' \
        else adjacent_probe_config()
    sizes = tuple(int(part) for part in args.tp.split(',') if part.strip())
    _numeric_defaults = {
        'peak_tflops': parser.get_default('peak_tflops'),
        'peak_bw_gbps': parser.get_default('peak_bw_gbps'),
        'link_bw_gbps': parser.get_default('link_bw_gbps'),
        'allreduce_latency_us': parser.get_default('allreduce_latency_us'),
    }
    overrides = {k: getattr(args, k) for k in _numeric_defaults
                 if getattr(args, k) != _numeric_defaults[k]}
    hardware = HardwareModel.from_gpu(args.hardware)
    if overrides:
        hardware = dataclasses.replace(hardware, **overrides)
    sweep = tp_sweep(
        cfg, sizes, prefill_len=args.prefill_len,
        decode_batch=args.decode_batch, total_layers=args.layers,
        hardware=hardware)
    _print_table(sweep)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
