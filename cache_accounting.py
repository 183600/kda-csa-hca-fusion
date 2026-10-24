"""Live KV-cache accounting + autograd-hygiene audit for the decode stack.

Why this module?
----------------
``run_kv_cache`` (Exp 3) computes the *analytical* KV-cache footprint of the
reference design: ``mode='full_accounting'`` is described as "the number a
production inference engine would actually pay". This module checks that
claim against the tensors a real decode session actually retains —
``run_decoding.HybridDecoding`` plus ``ops_decoding_cache``. The two can
drift in ways that are invisible to the operator-level tests:

* allocation *history* (one bulk prefill ``append_step`` vs token-by-token
  appends) can change the geometric growth sequence. This module keeps
  live and analytical accounting equal for every fill history.
* incremental decode caches can accidentally retain autograd graphs
  (``grad_fn``) when a caller forgets ``torch.no_grad()`` — the classic
  unbounded-memory leak during long generation. :func:`autograd_graph_probe`
  pins the graph-growth contract for the decode stack.

Paper alignment
---------------
The component taxonomy is the KV-cache layout of DeepSeek-V4
(2606.19348v1, §3.5.1 / Figure 6):

* ``classical_kv_cache`` — CSA compressed KV, CSA indexer KV, HCA
  compressed KV (each cache block covers ``lcm(m, m')`` original tokens and
  produces ``k1 = lcm/m`` CSA entries and ``k2 = lcm/m'`` HCA entries);
* ``state_cache`` — SWA KV ring buffers, uncompressed tail accumulators
  ("unready-for-compression tokens"), and the per-layer linear-attention
  state (KDA, Kimi Linear 2510.26692v2 §3.2 / Eq. 15: constant-size
  ``S in R^{dk x dv}`` per head).

The byte model implements the §2.3.4 mixed storage format (BF16 for the
RoPE dimensions, FP8 for the rest) on top of the same taxonomy, so the
"nearly half vs pure BF16" storage claim can be recomputed for any
geometry instead of quoted.

Accounting conventions (mirrored from ``run_kv_cache``)
-------------------------------------------------------
* one *element* = one scalar of the retained representation;
* ``elements_allocated`` = what the runtime really holds, including spare
  geometric capacity and pre-allocated ring buffers;
* ``elements_valid`` = the logically live prefix (completed blocks, valid
  ring entries);
* per-sequence figures divide by the batch size B;
* sinks are ``nn.Parameter``s (not cache), included only to reconcile
  one-to-one with Exp 3's ``full_accounting`` totals.

All functions return JSON-strict Python scalars (no tensors/dtypes).
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from ops_fused import HybridConfig  # noqa: F401  (public re-export / typing)
from ops_rope import effective_rope_dim
from kaggle_setup import (capture_provenance, set_global_seed,
                          write_json_atomic)
from run_kv_cache import geometric_capacity, kv_cache_elements

# ---------------------------------------------------------------------------
# Component taxonomy (DeepSeek-V4 §3.5.1 / Figure 6)
# ---------------------------------------------------------------------------

CLASSICAL_KV_CACHE = 'classical_kv_cache'
STATE_CACHE = 'state_cache'
PARAMETER = 'parameter'

#: component name -> (op, pool)
COMPONENT_SPECS = {
    'kda_recurrent_state': ('kda', STATE_CACHE),
    'kda_conv_lookback': ('kda', STATE_CACHE),
    'mamba2_ssm_state': ('mamba2', STATE_CACHE),
    'mamba2_conv_state': ('mamba2', STATE_CACHE),
    'rwkv7_matrix_state': ('rwkv7', STATE_CACHE),
    'rwkv7_shift_state': ('rwkv7', STATE_CACHE),
    'csa_compressed_kv': ('csa', CLASSICAL_KV_CACHE),
    'csa_indexer_kv': ('csa', CLASSICAL_KV_CACHE),
    'csa_sliding_window': ('csa', STATE_CACHE),
    'csa_partial_tail': ('csa', STATE_CACHE),
    'csa_overlap_blocks': ('csa', STATE_CACHE),
    'hca_compressed_kv': ('hca', CLASSICAL_KV_CACHE),
    'hca_sliding_window': ('hca', STATE_CACHE),
    'hca_partial_tail': ('hca', STATE_CACHE),
    'sink_logits': ('sink', PARAMETER),
}

#: Components whose entries are RoPE-rotated KV vectors and therefore use
#: the paper's mixed BF16-RoPE / FP8 storage policy.
_ROPE_ROTATED = {
    'csa_compressed_kv', 'csa_sliding_window',
    'hca_compressed_kv', 'hca_sliding_window',
}

#: Components whose tensors have no batch axis (shared across the batch),
#: so their per-sequence element count equals their raw element count.
_BATCH_SHARED = {'sink_logits'}

_DTYPES_BY_OP = ('csa', 'hca', 'kda', 'mamba2', 'rwkv7')

#: The concrete layer kinds that fill the stack's memory slots (mirrors
#: ops_fused.MEMORY_LAYER_KINDS; kept local to avoid an import cycle).
_MEMORY_KINDS = ('kda', 'mamba2', 'rwkv7')


def _layer_counts(cfg: HybridConfig) -> dict[str, int]:
    """Per-op layer counts, EXPLICIT-LAYOUT aware.

    Ratio mode: the memory-slot count is ``cfg.n_kda`` attributed to
    ``cfg.memory_type`` (the ratio-mode semantics). Explicit-layout
    mode: count each kind directly from ``cfg.layout`` — the eight-group
    matrix pins 10-layer layouts whose memory/CSA/HCA counts are NOT the
    ratio fields (which are ignored in that mode), so using them here
    would mis-audit every matrix-configured model.
    """
    if cfg.layout is not None:
        counts = {op: 0 for op in _DTYPES_BY_OP}
        for kind in cfg.layout:
            if kind in counts:
                counts[kind] += 1
        return counts
    counts = {op: 0 for op in _DTYPES_BY_OP}
    counts[cfg.memory_type] = cfg.n_kda
    counts['csa'] = cfg.n_csa
    counts['hca'] = cfg.n_hca
    return counts


# ---------------------------------------------------------------------------
# Autograd graph helpers
# ---------------------------------------------------------------------------

def _reachable_graph_nodes(tensor: torch.Tensor | None) -> int:
    """Number of autograd nodes reachable from ``tensor``'s ``grad_fn``.

    Counts distinct ``grad_fn`` objects (including ``AccumulateGrad``
    leaves) via ``next_functions``. ``0`` means the tensor is a graph root
    (detached leaf): the hygiene property decode caches must satisfy.
    """
    if tensor is None or tensor.grad_fn is None:
        return 0
    seen: set[int] = set()
    stack = [tensor.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        for nxt, _ in getattr(fn, 'next_functions', ()):
            if nxt is not None:
                stack.append(nxt)
    return len(seen)


def _graph_stats(tensors) -> dict:
    """Aggregate ``requires_grad`` / ``grad_fn`` / node count over a list."""
    requires = False
    nodes = 0
    graph_tensors = 0
    for t in tensors:
        if t is None:
            continue
        if t.requires_grad:
            requires = True
        if t.grad_fn is not None:
            graph_tensors += 1
            nodes += _reachable_graph_nodes(t)
    return {
        'requires_grad': requires,
        'has_grad_fn': graph_tensors > 0,
        'graph_tensors': graph_tensors,
        'graph_nodes': nodes,
    }


# ---------------------------------------------------------------------------
# Live cache walker
# ---------------------------------------------------------------------------

class _Component:
    """Accumulator for one named cache component (internal)."""

    __slots__ = ('name', 'op', 'pool', 'tensors', 'allocated', 'valid',
                 'graph')

    def __init__(self, name: str, op: str, pool: str):
        self.name = name
        self.op = op
        self.pool = pool
        self.tensors: list[torch.Tensor] = []
        self.allocated = 0
        self.valid = 0
        self.graph = {'requires_grad': False, 'has_grad_fn': False,
                      'graph_tensors': 0, 'graph_nodes': 0}

    def add(self, tensor: torch.Tensor | None, *, valid: int | None = None):
        if tensor is None:
            return
        self.tensors.append(tensor)
        self.allocated += tensor.numel()
        self.valid += tensor.numel() if valid is None else int(valid)
        stats = _graph_stats([tensor])
        self.graph['requires_grad'] |= stats['requires_grad']
        self.graph['has_grad_fn'] |= stats['has_grad_fn']
        self.graph['graph_tensors'] += stats['graph_tensors']
        self.graph['graph_nodes'] += stats['graph_nodes']


def _empty_components() -> dict[str, _Component]:
    return {name: _Component(name, op, pool)
            for name, (op, pool) in COMPONENT_SPECS.items()}


def _model_layers(decoder) -> list:
    model = getattr(decoder, 'model', None)
    if model is None or not hasattr(model, 'layers'):
        raise TypeError(
            'live_cache_components: expected a run_decoding.HybridDecoding-'
            'style decoder exposing ``.model.layers`` / ``._csa_caches`` / '
            f'``._hca_caches``, got {type(decoder).__name__}.')
    return list(model.layers), list(getattr(model, 'layout', []))


def live_cache_components(decoder) -> dict:
    """Walk the *actual* tensors retained by a hybrid decode session.

    Returns a dict with one record per component (see ``COMPONENT_SPECS``)
    plus ``batch_size`` / ``total_*`` aggregates. Every element count is
    exact (``tensor.numel()``); ``elements_valid`` is the logically live
    prefix of capacity-grown stores. Per-sequence figures divide by the
    batch size except for batch-shared components (``_BATCH_SHARED``).
    Raises ``ValueError`` when the decoder holds no state at all (nothing
    was prefilled/decoded yet).
    """
    layers, layout = _model_layers(decoder)
    comps = _empty_components()

    # --- memory-route recurrent state + shift/conv lookback -------------
    # Route by the layout's memory kind: the stacked ``_kda_state`` buffer
    # holds whichever memory operator fills the slots (the attribute keeps
    # its ``_kda`` name for every route), and the per-layer streaming
    # buffer is
    # ``_conv_lookback`` for KDA / Mamba-2 vs ``_shift_state`` for RWKV-7
    # (the token-shift history). Attributing them to route-specific
    # components keeps the live audit and the dtypes report exact per
    # operator.
    model = decoder.model
    memory_kinds = [k for k in layout if k in _MEMORY_KINDS]
    route = memory_kinds[0] if memory_kinds else 'kda'
    if route == 'rwkv7':
        comps['rwkv7_matrix_state'].add(getattr(model, '_kda_state', None))
        for layer, kind in zip(layers, layout):
            if kind == 'rwkv7':
                comps['rwkv7_shift_state'].add(
                    getattr(layer, '_shift_state', None))
    elif route == 'mamba2':
        comps['mamba2_ssm_state'].add(getattr(model, '_kda_state', None))
        for layer, kind in zip(layers, layout):
            if kind == 'mamba2':
                comps['mamba2_conv_state'].add(
                    getattr(layer, '_conv_lookback', None))
    else:
        comps['kda_recurrent_state'].add(getattr(model, '_kda_state', None))
        for layer in layers:
            comps['kda_conv_lookback'].add(getattr(layer, '_conv_lookback', None))

    # --- CSA incremental caches ----------------------------------------
    for cache in getattr(decoder, '_csa_caches', []) or []:
        if cache is None:
            continue
        n_blocks = cache.n_blocks
        B, c, c_I = cache.B, cache.c, cache.c_I
        block_valid = n_blocks * c * B
        comps['csa_compressed_kv'].add(cache._C_comp, valid=block_valid)
        comps['csa_indexer_kv'].add(cache._K_IComp,
                                    valid=n_blocks * c_I * B)
        sw = cache._sw
        if sw is not None:
            sw_valid = min(cache.n_tokens_seen, sw.win) * c * B
            comps['csa_sliding_window'].add(sw._buf, valid=sw_valid)
        for tensor in (cache._acc_Ca + cache._acc_Cb + cache._acc_Za
                       + cache._acc_Zb + cache._acc_K_idx + cache._acc_Z_idx):
            comps['csa_partial_tail'].add(tensor)
        comps['csa_overlap_blocks'].add(cache._prev_Cb)
        comps['csa_overlap_blocks'].add(cache._prev_Zb)

    # --- HCA incremental caches ----------------------------------------
    for cache in getattr(decoder, '_hca_caches', []) or []:
        if cache is None:
            continue
        n_blocks = cache.n_blocks
        B, c = cache.B, cache.c
        block_valid = n_blocks * c * B
        comps['hca_compressed_kv'].add(cache._C_comp, valid=block_valid)
        sw = cache._sw
        if sw is not None:
            sw_valid = min(cache.n_tokens_seen, sw.win) * c * B
            comps['hca_sliding_window'].add(sw._buf, valid=sw_valid)
        for tensor in cache._acc_C + cache._acc_Z:
            comps['hca_partial_tail'].add(tensor)

    # --- sinks (parameters kept in the Err accounting) ------------------
    for layer, kind in zip(layers, layout):
        if kind in ('csa', 'hca'):
            comps['sink_logits'].add(getattr(layer, 'sink', None))

    tensors = [t for comp in comps.values() for t in comp.tensors]
    if not tensors:
        raise ValueError(
            'live_cache_components: the decoder holds no cache tensors; run '
            'a prefill (and optionally decode steps) before auditing.')

    # Batch size: KDA state axis 1 or any cache's B attribute.
    B = None
    kda_state = getattr(model, '_kda_state', None)
    if kda_state is not None and kda_state.dim() == 5:
        B = int(kda_state.shape[1])
    if B is None:
        for cache in (list(getattr(decoder, '_csa_caches', []) or [])
                      + list(getattr(decoder, '_hca_caches', []) or [])):
            if cache is not None:
                B = int(cache.B)
                break
    if B is None or B < 1:
        raise ValueError(
            'live_cache_components: could not determine the batch size from '
            'the live cache tensors.')

    out_components = {}
    for name, comp in comps.items():
        if not comp.tensors:
            continue
        dtypes: dict[str, int] = {}
        for t in comp.tensors:
            dtypes[str(t.dtype).replace('torch.', '')] = (
                dtypes.get(str(t.dtype).replace('torch.', ''), 0) + 1)
        out_components[name] = {
            'op': comp.op,
            'pool': comp.pool,
            'tensors': len(comp.tensors),
            'dtypes': dtypes,
            'elements_allocated': int(comp.allocated),
            'elements_valid': int(comp.valid),
            'elements_allocated_per_sequence': (
                comp.allocated if name in _BATCH_SHARED
                else comp.allocated / B),
            'elements_valid_per_sequence': (
                comp.valid if name in _BATCH_SHARED else comp.valid / B),
            'requires_grad': bool(comp.graph['requires_grad']),
            'has_grad_fn': bool(comp.graph['has_grad_fn']),
            'graph_nodes': int(comp.graph['graph_nodes']),
        }

    totals = {
        'elements_allocated': sum(c['elements_allocated']
                                  for c in out_components.values()),
        'elements_valid': sum(c['elements_valid']
                              for c in out_components.values()),
        'graph_nodes': sum(c['graph_nodes']
                           for c in out_components.values()),
    }
    return {
        'batch_size': B,
        'components': out_components,
        'totals': totals,
    }


# ---------------------------------------------------------------------------
# Analytical decomposition (Exp 3 ``full_accounting``, component-wise)
# ---------------------------------------------------------------------------

def exp3_geometry(cfg: HybridConfig, *, conv_ksize: int = 3) -> dict:
    """Map a :class:`HybridConfig` onto ``run_kv_cache`` keyword geometry.

    Mirrors ``lm_consistency.lm_tower_geometry`` (kept as a local copy so
    this module stays import-acyclic for the training probe).
    """
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'exp3_geometry: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    counts = _layer_counts(cfg)
    hybrid_n_mem = sum(counts[k] for k in _MEMORY_KINDS)
    return dict(
        d=cfg.d_model, H=cfg.n_heads_qk, K=cfg.head_dim_k, V=cfg.head_dim_v,
        kda_hv=cfg.n_heads_v, kda_k=cfg.head_dim_k, kda_v=cfg.head_dim_v,
        kda_conv_ksize=conv_ksize,
        csa_m=cfg.csa_m, csa_topk=cfg.csa_topk, csa_nh=cfg.csa_nh,
        csa_c=cfg.csa_c, csa_dc=cfg.csa_dc, csa_nIh=cfg.csa_nIh,
        csa_cI=cfg.csa_cI, csa_sliding_window=cfg.csa_sliding_window,
        hca_m2=cfg.hca_m2, hca_nh=cfg.hca_nh, hca_c=cfg.hca_c,
        hca_dc=cfg.hca_dc, hca_sliding_window=cfg.hca_sliding_window,
        # Memory-slot count (layout-aware; ratio mode uses the cfg.n_kda
        # value). run_kv_cache's hybrid_*_rows use this as the
        # memory-layer count regardless of the concrete memory route.
        hybrid_n_kda=hybrid_n_mem, hybrid_n_csa=counts['csa'],
        hybrid_n_hca=counts['hca'],
        # Mamba-2 / RWKV-7 geometry (used by the route-aware rows of
        # kv_cache_elements / prefill_flops when the model's memory route
        # is mamba2 / rwkv7).
        mamba_d_state=cfg.mamba_d_state, mamba_d_conv=cfg.mamba_d_conv,
        mamba_expand=cfg.mamba_expand, mamba_headdim=cfg.mamba_headdim,
        mamba_ngroups=cfg.mamba_ngroups, mamba_dt_rank=cfg.mamba_dt_rank,
        rwkv_head_dim=cfg.rwkv_head_dim, rwkv_decay_lora=cfg.rwkv_decay_lora,
        rwkv_a_lora=cfg.rwkv_a_lora, rwkv_gate_lora=cfg.rwkv_gate_lora,
    )


def lcm_packing(cfg: HybridConfig, T: int) -> dict:
    """Paper cache-block structure: one block per ``lcm(m, m')`` tokens.

    DeepSeek-V4 §3.5.1: a classical cache block covers ``lcm(m, m')``
    original tokens and yields ``k1 = lcm/m`` CSA rows and ``k2 = lcm/m'``
    HCA rows. Returns the exact block counts expected after ``T`` tokens
    (floor semantics, matching the caches).
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'lcm_packing: T must be a >= 0 int, got {T!r}')
    L = math.lcm(cfg.csa_m, cfg.hca_m2)
    k1, k2 = L // cfg.csa_m, L // cfg.hca_m2
    n_blocks = T // L
    # Trailing partial super-blocks still produce full CSA/HCA rows whenever
    # their own compression factor has completed, so derive the exact row
    # counts directly from T (identity: T // m == (T // L) * (L // m)
    # + (T % L) // m).
    return {
        'lcm_tokens': L,
        'k1_csa_rows_per_block': k1,
        'k2_hca_rows_per_block': k2,
        'complete_lcm_blocks': n_blocks,
        'csa_blocks_expected': T // cfg.csa_m,
        'hca_blocks_expected': T // cfg.hca_m2,
    }


def analytic_cache_components(cfg: HybridConfig, T: int, *,
                              conv_ksize: int = 3) -> dict:
    """Per-sequence analytical decomposition of ``full_accounting``.

    Same component names and conventions as :func:`live_cache_components`
    (``elements_allocated`` includes geometric spare capacity and full ring
    buffers; ``elements_valid`` is the live prefix). The sum over
    components is asserted against the committed
    ``run_kv_cache.kv_cache_elements(op, T, mode='full_accounting')`` in
    :func:`audit_cache_accounting`, so the two models can never diverge
    silently.
    """
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'analytic_cache_components: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(
            f'analytic_cache_components: T must be a >= 0 int, got {T!r}.')

    comps: dict[str, dict] = {}

    def put(name, allocated, valid):
        comps[name] = {
            'op': COMPONENT_SPECS[name][0],
            'pool': COMPONENT_SPECS[name][1],
            'elements_allocated': int(allocated),
            'elements_valid': int(valid),
        }

    # Memory route: constant recurrent state + per-layer streaming
    # buffer, attributed by the ACTUAL route (ratio mode: memory_type;
    # explicit layout: the memory kind present). All three routes keep
    # an O(1) sequence-independent state; only the shapes differ.
    counts = _layer_counts(cfg)
    if cfg.layout is not None:
        memory_kinds = [k for k in cfg.layout if k in _MEMORY_KINDS]
        route = memory_kinds[0] if memory_kinds else cfg.memory_type
    else:
        route = cfg.memory_type
    n_mem = counts.get(route, 0)
    if route == 'kda':
        hvkv = cfg.n_heads_v * cfg.head_dim_k * cfg.head_dim_v
        put('kda_recurrent_state', n_mem * hvkv, n_mem * hvkv)
        lookback = (conv_ksize - 1) * cfg.d_model
        put('kda_conv_lookback', n_mem * lookback, n_mem * lookback)
    elif route == 'mamba2':
        d_inner = cfg.mamba_expand * cfg.d_model
        mamba_heads = (d_inner // cfg.mamba_headdim
                       if cfg.mamba_headdim > 0 else 1)
        ssm = mamba_heads * cfg.mamba_headdim * cfg.mamba_d_state
        conv = d_inner * (cfg.mamba_d_conv - 1)
        put('mamba2_ssm_state', n_mem * ssm, n_mem * ssm)
        put('mamba2_conv_state', n_mem * conv, n_mem * conv)
    elif route == 'rwkv7':
        rwkv_heads = (cfg.d_model // cfg.rwkv_head_dim
                      if cfg.rwkv_head_dim > 0 else 1)
        matrix = rwkv_heads * cfg.rwkv_head_dim * cfg.rwkv_head_dim
        shift = cfg.d_model
        put('rwkv7_matrix_state', n_mem * matrix, n_mem * matrix)
        put('rwkv7_shift_state', n_mem * shift, n_mem * shift)

    # CSA / HCA (layout-aware counts; in ratio mode these equal the
    # ratio-field values).
    n_csa_l = counts['csa']
    n_hca_l = counts['hca']
    n_csa_blocks = T // cfg.csa_m
    csa_cap = geometric_capacity(n_csa_blocks)
    put('csa_compressed_kv',
        n_csa_l * csa_cap * cfg.csa_c,
        n_csa_l * n_csa_blocks * cfg.csa_c)
    put('csa_indexer_kv',
        n_csa_l * csa_cap * cfg.csa_cI,
        n_csa_l * n_csa_blocks * cfg.csa_cI)
    put('csa_sliding_window',
        n_csa_l * cfg.csa_sliding_window * cfg.csa_c,
        n_csa_l * min(T, cfg.csa_sliding_window) * cfg.csa_c)
    put('csa_partial_tail',
        n_csa_l * (T % cfg.csa_m) * (4 * cfg.csa_c + 2 * cfg.csa_cI),
        n_csa_l * (T % cfg.csa_m) * (4 * cfg.csa_c + 2 * cfg.csa_cI))
    overlap = (2 * cfg.csa_m * cfg.csa_c) if n_csa_blocks >= 1 else 0
    put('csa_overlap_blocks', n_csa_l * overlap, n_csa_l * overlap)

    # HCA.
    n_hca_blocks = T // cfg.hca_m2
    hca_cap = geometric_capacity(n_hca_blocks)
    put('hca_compressed_kv',
        n_hca_l * hca_cap * cfg.hca_c,
        n_hca_l * n_hca_blocks * cfg.hca_c)
    put('hca_sliding_window',
        n_hca_l * cfg.hca_sliding_window * cfg.hca_c,
        n_hca_l * min(T, cfg.hca_sliding_window) * cfg.hca_c)
    put('hca_partial_tail',
        n_hca_l * (T % cfg.hca_m2) * 2 * cfg.hca_c,
        n_hca_l * (T % cfg.hca_m2) * 2 * cfg.hca_c)

    # Sinks (parameters, kept for one-to-one reconciliation with Exp 3).
    sink = n_csa_l * cfg.csa_nh + n_hca_l * cfg.hca_nh
    put('sink_logits', sink, sink)

    return comps


# ---------------------------------------------------------------------------
# Live-vs-analytical audit
# ---------------------------------------------------------------------------

def _live_context_len(decoder, cfg: HybridConfig) -> int | None:
    """Tokens seen by the incremental caches (None when there are none)."""
    seen = []
    for cache in (list(getattr(decoder, '_csa_caches', []) or [])
                  + list(getattr(decoder, '_hca_caches', []) or [])):
        if cache is not None:
            seen.append(int(cache.n_tokens_seen))
    if not seen:
        return None
    if len(set(seen)) != 1:
        raise ValueError(
            f'audit_cache_accounting: incremental caches disagree on '
            f'n_tokens_seen={sorted(set(seen))}; the decoder is mid-prefill '
            f'or was fed mismatched chunks.')
    return seen[0]


def audit_cache_accounting(decoder, cfg: HybridConfig, *, T: int | None = None,
                           live: dict | None = None) -> dict:
    """Reconcile the live decode caches against Exp 3's analytical model.

    ``T`` defaults to the caches' ``n_tokens_seen``. When given explicitly
    it must match (guards against auditing a decoder that was filled with a
    different number of tokens than the analytical evaluation assumes).
    ``live`` lets the caller pass a previous :func:`live_cache_components`
    result to avoid re-walking the tensors.

    Returns a JSON-strict report: per-component deltas, per-operator totals
    against the committed ``kv_cache_elements(..., mode='full_accounting')``,
    and a single ``consistent`` verdict.
    """
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'audit_cache_accounting: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    if live is None:
        live = live_cache_components(decoder)
    ctx = _live_context_len(decoder, cfg)
    if T is None:
        if ctx is None:
            raise ValueError(
                'audit_cache_accounting: no CSA/HCA caches to derive T from; '
                'pass T explicitly.')
        T = ctx
    elif ctx is not None and ctx != T:
        raise ValueError(
            f'audit_cache_accounting: T={T} does not match the caches\' '
            f'n_tokens_seen={ctx}.')
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'audit_cache_accounting: T must be >= 0, got {T!r}.')

    live_comps = live['components']
    analytic = analytic_cache_components(cfg, T)

    mismatches: list[dict] = []
    per_component: dict[str, dict] = {}
    for name in COMPONENT_SPECS:
        a = analytic.get(name)
        l = live_comps.get(name)
        if a is None:
            continue
        l_alloc = (l['elements_allocated_per_sequence'] if l else 0.0)
        l_valid = (l['elements_valid_per_sequence'] if l else 0.0)
        entry = {
            'op': a['op'],
            'pool': a['pool'],
            'live_allocated': l_alloc,
            'analytic_allocated': a['elements_allocated'],
            'live_valid': l_valid,
            'analytic_valid': a['elements_valid'],
            'match': l_alloc == a['elements_allocated']
                     and l_valid == a['elements_valid'],
        }
        per_component[name] = entry
        if not entry['match']:
            mismatches.append({'component': name, **{
                k: entry[k] for k in
                ('live_allocated', 'analytic_allocated', 'live_valid',
                 'analytic_valid')}})

    # Per-operator totals against the committed Exp 3 function. Exp 3's
    # ``kv_cache_elements(op, ...)`` reports ONE layer, so compare against
    # the live layout total divided by the layer count (all layers share
    # the config geometry).
    geo = exp3_geometry(cfg)
    layers, layout = _model_layers(decoder)
    sink_per_seq = {'csa': 0, 'hca': 0}
    for layer, kind in zip(layers, layout):
        if kind in sink_per_seq:
            sink = getattr(layer, 'sink', None)
            if sink is not None:
                # Sinks are batch-shared parameters (no batch axis).
                sink_per_seq[kind] += sink.numel()
    n_layers = _layer_counts(cfg)
    op_totals = {}
    hybrid_live = 0.0
    for op in _DTYPES_BY_OP:
        live_layout = sum(
            entry['live_allocated'] for entry in per_component.values()
            if entry['op'] == op)
        if op in sink_per_seq:
            # Exp 3 folds each layer's sink (an ``nn.Parameter``) into its
            # operator total; include the same live contribution here.
            live_layout += sink_per_seq[op]
        hybrid_live += live_layout
        committed = float(kv_cache_elements(op, T, mode='full_accounting',
                                            **geo))
        if not n_layers[op]:
            op_totals[op] = {
                'live_layout': 0.0,
                'live_per_layer': None,
                'committed_full_accounting_per_layer': committed,
                'match': True,
                'note': 'op absent from the layout; nothing to reconcile',
            }
            continue
        live_per_layer = live_layout / n_layers[op]
        op_totals[op] = {
            'live_layout': live_layout,
            'live_per_layer': live_per_layer,
            'committed_full_accounting_per_layer': committed,
            'match': live_per_layer == committed,
        }
        if live_per_layer != committed:
            mismatches.append({'component': f'{op}_total',
                               'live': live_per_layer,
                               'analytic': committed})

    # The hybrid committed total follows the model's memory ROUTE (the
    # hybrid_kch row assumes KDA memory slots; auditing an mamba2 / rwkv7
    # model against it would miscount the memory-op total).
    if cfg.layout is not None:
        memory_route = next(
            (k for k in cfg.layout if k in _MEMORY_KINDS), cfg.memory_type)
    else:
        memory_route = cfg.memory_type
    hybrid_op = {'kda': 'hybrid_kch', 'mamba2': 'hybrid_mamba2',
                 'rwkv7': 'hybrid_rwkv7'}[memory_route]
    hybrid_committed = float(kv_cache_elements(
        hybrid_op, T, mode='full_accounting', **geo))
    op_totals[hybrid_op] = {
        'live_layout': hybrid_live,
        'committed_full_accounting_per_layer': hybrid_committed,
        'match': hybrid_live == hybrid_committed,
    }
    if hybrid_live != hybrid_committed:
        mismatches.append({'component': f'{hybrid_op}_total',
                           'live': hybrid_live,
                           'analytic': hybrid_committed})

    # In explicit-layout mode ``cfg.n_kda/n_csa/n_hca`` are ignored by
    # ``_layer_counts``; report the counts the audit actually used (with
    # the three memory kinds folded into the memory slot, matching ratio
    # mode's [memory, CSA, HCA] convention).
    memory_layers = sum(n_layers[op] for op in _MEMORY_KINDS)
    return {
        'context_len': T,
        'batch_size': live['batch_size'],
        'layout': [memory_layers, n_layers['csa'], n_layers['hca']],
        'memory_type': memory_route,
        'resolved_layout': list(cfg.layout) if cfg.layout is not None else None,
        'per_component': per_component,
        'per_operator': op_totals,
        'mismatches': mismatches,
        'consistent': not mismatches,
    }


# ---------------------------------------------------------------------------
# Autograd hygiene
# ---------------------------------------------------------------------------

def autograd_hygiene(decoder, *, live: dict | None = None) -> dict:
    """Check every live *cache* tensor for retained autograd graphs.

    A clean decode cache holds only detached tensors: ``requires_grad``
    False, ``grad_fn`` None, zero reachable graph nodes. Any graph-bearing
    tensor is reported by name so a leak is immediately attributable.
    Sink logits are ``nn.Parameter``s (legitimately ``requires_grad``) and
    are excluded — only the KV/state pools are cache.
    """
    if live is None:
        live = live_cache_components(decoder)
    bearing = [
        {'component': name, 'graph_nodes': rec['graph_nodes'],
         'requires_grad': rec['requires_grad'],
         'tensors': rec['tensors']}
        for name, rec in live['components'].items()
        if rec['pool'] != PARAMETER
        and (rec['has_grad_fn'] or rec['requires_grad'])
    ]
    cache_nodes = sum(rec['graph_nodes'] for rec in live['components'].values()
                      if rec['pool'] != PARAMETER)
    return {
        'clean': not bearing,
        'reachable_graph_nodes': cache_nodes,
        'graph_bearing': bearing,
    }


def autograd_graph_probe(cfg: HybridConfig, *, total_layers: int | None = None,
                         device='cpu', prefill_len: int = 4,
                         steps=(4, 12), seed: int = 0) -> dict:
    """Check that decode retains a BOUNDED autograd graph as steps grow.

    Runs a fresh decoder with grad tracking enabled (deliberately WITHOUT
    ``torch.no_grad()`` — a misuse a serving caller can hit) and counts
    the autograd nodes reachable from the retained cache tensors after
    each checkpoint in ``steps``. A detached-state decoder holds a
    constant node count (each step replaces the previous step's graph); a
    leak grows linearly with the number of decoded tokens.
    """
    from run_decoding import HybridDecoding

    if not isinstance(steps, (tuple, list)) or len(steps) < 2 or \
            any((not isinstance(s, int)) or isinstance(s, bool) or s < 1
                for s in steps):
        raise ValueError(
            f'autograd_graph_probe: steps must be >= 2 positive ints, '
            f'got {steps!r}')
    steps = sorted(int(s) for s in steps)
    if len(set(steps)) != len(steps):
        # Checkpoints must be distinct: duplicates make the returned
        # ``reachable_nodes`` shorter than ``checkpoints`` and can make
        # ``steps[-1] - steps[0]`` zero (division by zero below).
        raise ValueError(
            f'autograd_graph_probe: steps must be distinct checkpoints, '
            f'got {steps!r}')
    if total_layers is None:
        total_layers = (len(cfg.layout) if cfg.layout is not None
                        else max(1, cfg.n_kda + cfg.n_csa + cfg.n_hca))

    torch.manual_seed(seed)
    decoder = HybridDecoding(cfg=cfg, total_layers=total_layers).to(device)
    decoder.eval()
    x = torch.randn(1, prefill_len, cfg.d_model, device=device)
    x = x * 0.1
    x.requires_grad_(True)

    counts: list[int] = []
    try:
        with torch.enable_grad():
            decoder.reset()
            decoder(x)
            n = 0
            for _ in range(steps[-1]):
                decoder(x[:, :1])
                n += 1
                if n in steps:
                    counts.append(int(live_cache_components(decoder)
                                      ['totals']['graph_nodes']))
    finally:
        decoder.reset()

    n0, n1 = counts[0], counts[-1]
    return {
        'checkpoints': steps,
        'reachable_nodes': counts,
        'bounded': n1 == n0,
        'growth_per_step': (n1 - n0) / (steps[-1] - steps[0]),
    }


# ---------------------------------------------------------------------------
# Paper-aligned byte model (DeepSeek-V4 §2.3.4 mixed storage)
# ---------------------------------------------------------------------------

def paper_aligned_bytes(cfg: HybridConfig, components: dict, *,
                        policy: str = 'mixed_fp8') -> dict:
    """Bytes retained under a paper storage policy.

    ``policy='pure_bf16'``: 2 bytes for every element (the baseline the
    paper compares against). ``policy='mixed_fp8'``: RoPE-rotated KV
    entries use BF16 on their rotated ``rope_dim_eff`` dimensions and FP8
    on the remaining ones (DeepSeek-V4 §2.3.4); the indexer KV uses FP8
    (low-precision indexer, §2.3.4 / Kimi Linear §7.1); all other state
    (KDA recurrence, uncompressed tail accumulators, overlap blocks,
    sinks) stays BF16.

    ``components`` is any ``{name: {'elements_allocated': int, ...}}``
    mapping — live or analytical, so the same policy can be applied to
    both. Elements are per the caller's B convention (live = full batch).
    """
    if policy not in ('pure_bf16', 'mixed_fp8'):
        raise ValueError(
            f'paper_aligned_bytes: policy={policy!r} must be one of '
            f"('pure_bf16', 'mixed_fp8').")
    rope = {
        'csa': effective_rope_dim(cfg.csa_rope_dim, cfg.csa_c),
        'hca': effective_rope_dim(cfg.hca_rope_dim, cfg.hca_c),
    }
    dim = {'csa': cfg.csa_c, 'hca': cfg.hca_c}

    by_component: dict[str, dict] = {}
    pool_bytes = {CLASSICAL_KV_CACHE: 0, STATE_CACHE: 0, PARAMETER: 0}
    weighted_elements = []
    for name, meta in COMPONENT_SPECS.items():
        rec = components.get(name)
        if rec is None:
            continue
        elements = rec['elements_allocated']
        op, pool = meta
        if policy == 'pure_bf16':
            per_element = 2.0
        elif name in _ROPE_ROTATED:
            rope_dim = rope[op]
            per_element = (2.0 * rope_dim + 1.0
                           * (dim[op] - rope_dim)) / dim[op]
        elif name == 'csa_indexer_kv':
            # Low-precision indexer (DeepSeek-V4 §2.3.4 FP4 compute /
            # Kimi Linear §7.1 FP8 DSA indexer): 1 byte per element.
            per_element = 1.0
        else:
            per_element = 2.0
        try:
            elements_f = float(elements)
        except OverflowError:
            # JSON integers are unbounded; keep the byte totals JSON-safe
            # and let the scaled ratio below use the exact integer counts.
            elements_f = math.inf if elements >= 0 else -math.inf
        nbytes = elements_f * per_element
        weighted_elements.append((elements, per_element))
        by_component[name] = {
            'elements_allocated': elements,
            'bytes_per_element': per_element,
            'bytes': nbytes if math.isfinite(nbytes) else None,
        }
        pool_bytes[pool] = pool_bytes.get(pool, 0) + nbytes

    total = sum(pool_bytes.values())
    if not math.isfinite(total):
        total = None
    finite_pool_bytes = {
        key: (value if math.isfinite(value) else None)
        for key, value in pool_bytes.items()
    }
    if policy == 'pure_bf16':
        bf16_total = total
        ratio = None if total == 0.0 else 1.0
    else:
        bf16_total = paper_aligned_bytes(
            cfg, components, policy='pure_bf16')['total_bytes']
        ratio = None
        if total is not None and bf16_total:
            candidate = total / bf16_total
            if math.isfinite(candidate):
                ratio = candidate
        if ratio is None and weighted_elements:
            # Both totals can overflow even though their ratio is finite.
            # Evaluate the same ratio in element units (the common factor
            # cancels), using a shared scale so the sums stay finite.
            scale = max(abs(elements) for elements, _ in weighted_elements)
            if scale > 0.0:
                mixed_scaled = sum(
                    (elements / scale) * per_element
                    for elements, per_element in weighted_elements)
                bf16_scaled = sum(
                    (elements / scale) * 2.0
                    for elements, _ in weighted_elements)
                if bf16_scaled > 0.0:
                    candidate = mixed_scaled / bf16_scaled
                    if math.isfinite(candidate):
                        ratio = candidate
    return {
        'policy': policy,
        'rope_dim_effective': rope,
        'by_component': by_component,
        'bytes_by_pool': finite_pool_bytes,
        'total_bytes': total,
        'pure_bf16_total_bytes': bf16_total,
        'ratio_vs_pure_bf16': ratio,
    }


# ---------------------------------------------------------------------------
# One-shot audit report
# ---------------------------------------------------------------------------

def run_cache_audit(decoder, cfg: HybridConfig, *, T: int | None = None,
                    with_graph_probe: bool = False,
                    graph_probe_kwargs: dict | None = None,
                    bytes_policy: str = 'mixed_fp8',
                    seed: int | None = None) -> dict:
    """One full audit: live vs analytical accounting + hygiene + bytes.

    The result is JSON-strict and intended to be written next to a
    checkpoint (see ``train_lm_autodl.run_decode_consistency_probe``).
    ``seed`` is recorded in the report when given (the audit itself is
    seed-independent; it documents the caller's RNG state).
    """
    live = live_cache_components(decoder)
    comparison = audit_cache_accounting(decoder, cfg, T=T, live=live)
    hygiene = autograd_hygiene(decoder, live=live)
    bytes_mixed = paper_aligned_bytes(
        cfg, live['components'], policy=bytes_policy)
    report = {
        'context_len': comparison['context_len'],
        'batch_size': live['batch_size'],
        'layout': comparison['layout'],
        'components': live['components'],
        'totals': live['totals'],
        'comparison': comparison,
        'autograd_hygiene': hygiene,
        'bytes': bytes_mixed,
        'paper_alignment': {
            'layout': {
                CLASSICAL_KV_CACHE: sorted(
                    n for n, (_, p) in COMPONENT_SPECS.items()
                    if p == CLASSICAL_KV_CACHE),
                STATE_CACHE: sorted(
                    n for n, (_, p) in COMPONENT_SPECS.items()
                    if p == STATE_CACHE),
            },
            'lcm_packing': lcm_packing(cfg, comparison['context_len']),
            'sources': {
                'layout': 'DeepSeek-V4 (2606.19348v1) §3.5.1 / Figure 6',
                'mixed_storage': 'DeepSeek-V4 §2.3.4 (BF16 RoPE dims + FP8 '
                                 'body, mixed policy)',
                'indexer_low_precision': 'DeepSeek-V4 §2.3.4 (FP4 indexer '
                                         'compute) / Kimi Linear (2510.26692'
                                         'v2) §7.1 (FP8 indexer)',
                'kda_constant_state': 'Kimi Linear §3.2 / Eq. (15)',
                'rwkv7_constant_state': (
                    'RWKV-7 (Goose) recurrent matrix state — O(1) '
                    'sequence-independent per layer; see ops_rwkv.py for '
                    'the upstream binding'),
            },
        },
    }
    if with_graph_probe:
        kwargs = dict(graph_probe_kwargs or {})
        report['autograd_graph_probe'] = autograd_graph_probe(cfg, **kwargs)
    if seed is not None:
        report['seed'] = seed
    return report


def run_cache_audit_for_model(model, context_len: int, *,
                              device='cpu', decode_steps: int = 1,
                              with_graph_probe: bool = False,
                              seed: int = 0,
                              graph_probe_kwargs: dict | None = None) -> dict:
    """Audit an ``LMWithHybrid``-style model's decode cache end to end.

    Builds a weight-identical :class:`run_decoding.HybridDecoding` clone
    (mirroring ``lm_consistency.build_decoding_clone`` but kept local to
    avoid an import cycle), prefills ``context_len`` random hidden states,
    decodes ``decode_steps`` more, and audits at ``T = context_len +
    decode_steps``. Only the geometry/state matter for accounting, so
    random weights are fine.
    """
    from run_decoding import HybridDecoding

    hybrid = getattr(model, 'hybrid', None)
    cfg = getattr(model, 'cfg', None)
    if cfg is None and hybrid is not None:
        cfg = getattr(hybrid, 'cfg', None)
    if hybrid is None or not isinstance(cfg, HybridConfig):
        raise TypeError(
            'run_cache_audit_for_model: expected an LMWithHybrid-style model '
            'exposing ``.hybrid`` and a HybridConfig ``.cfg``, got '
            f'{type(model).__name__}.')
    if (not isinstance(context_len, int) or isinstance(context_len, bool)
            or context_len < 1):
        raise ValueError(
            f'run_cache_audit_for_model: context_len must be >= 1, got '
            f'{context_len!r}.')
    if (not isinstance(decode_steps, int) or isinstance(decode_steps, bool)
            or decode_steps < 0):
        raise ValueError(
            f'run_cache_audit_for_model: decode_steps must be >= 0, got '
            f'{decode_steps!r}.')

    torch.manual_seed(seed)
    decoder = HybridDecoding(cfg=cfg, total_layers=hybrid.total_layers)
    decoder.model.load_state_dict(hybrid.state_dict())
    decoder.eval().to(device)

    x = (torch.randn(1, context_len, cfg.d_model, device=device) * 0.1)
    try:
        with torch.no_grad():
            decoder.reset()
            decoder(x)
            for _ in range(decode_steps):
                decoder(x[:, :1])
            T = context_len + decode_steps
            return run_cache_audit(
                decoder, cfg, T=T, with_graph_probe=with_graph_probe,
                graph_probe_kwargs=graph_probe_kwargs, seed=seed)
    finally:
        decoder.reset()


# ---------------------------------------------------------------------------
# Tiny CLI
# ---------------------------------------------------------------------------

def _print_audit_summary(report: dict) -> None:
    comp = report['comparison']
    print(f"[cache-audit] T={comp['context_len']}, "
          f"B={comp['batch_size']}, layout={comp['layout']}, "
          f"consistent={comp['consistent']}")
    for name, rec in comp['per_component'].items():
        if not rec['match']:
            print(f"  [MISMATCH] {name}: live={rec['live_allocated']} vs "
                  f"analytic={rec['analytic_allocated']}")
    for op, tot in comp['per_operator'].items():
        if op == 'hybrid_kch':
            print(f"  [{op}] live={tot['live_layout']:.0f} "
                  f"committed={tot['committed_full_accounting_per_layer']:.0f} "
                  f"match={tot['match']}")
        elif tot.get('live_per_layer') is None:
            print(f"  [{op}] absent from layout")
        else:
            print(f"  [{op}] live/layer={tot['live_per_layer']:.0f} "
                  f"committed/layer="
                  f"{tot['committed_full_accounting_per_layer']:.0f} "
                  f"match={tot['match']}")
    hygiene = report['autograd_hygiene']
    print(f"[cache-audit] autograd clean={hygiene['clean']} "
          f"(graph nodes={hygiene['reachable_graph_nodes']})")
    if 'autograd_graph_probe' in report:
        probe = report['autograd_graph_probe']
        print(f"[cache-audit] graph probe steps={probe['checkpoints']} "
              f"nodes={probe['reachable_nodes']} bounded={probe['bounded']}")
    b = report['bytes']
    ratio = b['ratio_vs_pure_bf16']
    print(f"[cache-audit] bytes[{b['policy']}]="
          f"{b['total_bytes']:.0f} vs pure_bf16={b['pure_bf16_total_bytes']:.0f}"
          + (f" (ratio {ratio:.3f})" if ratio is not None else ''))
    print(f"[cache-audit] pools: {b['bytes_by_pool']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Live KV-cache accounting / autograd-hygiene audit of '
                    'the reference hybrid decode stack.')
    parser.add_argument('--T', type=int, default=64,
                        help='prefill length in tokens (default 64)')
    parser.add_argument('--decode-steps', type=int, default=1,
                        help='decode steps to run after the prefill')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-graph-probe', action='store_true',
                        help='skip the autograd graph-growth probe')
    parser.add_argument('--json', type=str, default=None,
                        help='write the full JSON-strict report here')
    args = parser.parse_args(argv)

    # Deterministic probe inputs: seed Python/NumPy/torch/CUDA and request
    # deterministic kernels so the audit report is reproducible run-to-run
    # on the same machine.
    seed_info = set_global_seed(0, deterministic=True)
    from run_decoding import HybridDecoding
    decoder = HybridDecoding().to(args.device).eval()
    cfg = decoder.cfg
    x = (torch.randn(1, args.T, cfg.d_model, device=args.device) * 0.1)
    with torch.no_grad():
        decoder.reset()
        decoder(x)
        for _ in range(args.decode_steps):
            decoder(x[:, :1])
        report = run_cache_audit(
            decoder, cfg, T=args.T + args.decode_steps,
            with_graph_probe=not args.no_graph_probe, seed=0)
    # Record the seed scheme and code/environment provenance alongside the
    # audit numbers so an archived --json report is a complete run record.
    report['seed'] = 0
    report['seed_info'] = seed_info
    report['provenance'] = capture_provenance()
    _print_audit_summary(report)
    if args.json:
        write_json_atomic(report, args.json, indent=2, allow_nan=False)
        print(f'[cache-audit] report saved: {args.json}')
    return 0 if report['comparison']['consistent'] else 1


if __name__ == '__main__':
    sys.exit(main())
