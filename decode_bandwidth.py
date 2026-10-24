"""Batch-aware decode-step memory-traffic ledger for the hybrid stack (Exp 13).

Why this module?
----------------
Exp 3 / Exp 9b answer *how much state is RETAINED* and Exp 7 answers the
*prefill* roofline. The phase where the recurrent memory routes
(KDA / Mamba-2 / RWKV-7) are supposed to win is token-by-token decode —
and there the design question is not retention but traffic: how many bytes
must MOVE per decoded step?

    traffic(B) = weight_bytes + B * state_bytes_per_sequence

The two terms scale differently, and that difference is the whole story:

* the tower's weights are streamed once per step and SHARED across the
  batch (batch-independent);
* the recurrent state + CSA/HCA caches are per-sequence and must be
  read/rewritten for every sequence in the batch (batch-proportional).

So there is a batch at which the step **switches** from
weight-bandwidth-bound to state-bandwidth-bound — and, because the
per-sequence term depends on the memory route (KDA's ``HV*K*V`` vs
Mamba-2's ``nheads*headdim*d_state`` vs RWKV-7's ``H*K*K``), that switch
point depends on the memory-type knob. This module computes the ledger,
the switch batch, and a per-route comparison table
(:func:`route_switch_table`).

The per-sequence term is additionally split by operator, so the decode
bandwidth PRICE of the retrieval / global-aggregation layers (CSA and
HCA — the operators the multi-aggregation probe of Exp 4 exercises) is
reported next to the memory route's own traffic.

Scope and honesty
-----------------
* **Attention-tower scope.** Weights are the parameters of
  ``HybridAttentionStack`` (the scope of Exp 2's hybrid-stack rows).
  Embeddings / lm_head / final norm stream once per step independently of
  the route and are out of scope.
* **Compulsory-traffic lower bound.** ``weight_bytes_per_step`` charges
  one stream of the tower parameters per step. A real kernel may keep
  weights resident across a large batch (turning the term into compute,
  not traffic) and the reference implementation's Python overhead moves
  far more bytes than the model — the ledger is the memory-bound lower
  bound, and conservative on the weight side.
* **One repeating layout unit.** Ratio mode prices
  ``n_kda + n_csa + n_hca`` layers; an explicit layout prices exactly
  ``len(cfg.layout)``. This is the same coverage as the Exp 3 / Exp 9b
  accounting (a 10-layer stack is two units and scales linearly).
* **Analysis only.** No forward pass, no timing, no measurement. Combine
  with ``roofline.measure_copy_bandwidth_gbps`` (or
  ``lm_consistency.host_peaks``) and :func:`bandwidth_bound_ms` to turn
  bytes into a bandwidth-bound time.
"""
from __future__ import annotations

import dataclasses
import math

import torch

import kv_budget
from kv_quant import FP8Spec, QuantSpec, bytes_per_element, spec_summary
from lm_consistency import decode_elements_per_token, decode_read_spec
from ops_fused import MEMORY_LAYER_KINDS, HybridAttentionStack, HybridConfig

#: The selectable recurrent memory routes (mirrors
#: ``ops_fused.MEMORY_LAYER_KINDS``; kept as an explicit tuple so the
#: comparison table has a stable default order).
MEMORY_ROUTES = ('kda', 'mamba2', 'rwkv7')

#: Exceptions a route can raise while being built for its weight count:
#: geometry validation (ValueError/TypeError) and optional-backend
#: unavailability (RuntimeError/ImportError). Used to degrade a route row
#: to ``status='unavailable'`` instead of faking or crashing a table.
_UNAVAILABLE = (ValueError, TypeError, RuntimeError, ImportError)


def _require_cfg(cfg: HybridConfig) -> None:
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'decode_bandwidth: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')


def _validate_itemsize(itemsize: int) -> None:
    if isinstance(itemsize, bool) or not isinstance(itemsize, int):
        raise TypeError(
            f'decode_bandwidth: itemsize={itemsize!r} must be an int '
            f'(bytes per element; 4 = fp32, the Exp 7 convention).')
    if itemsize < 1:
        raise ValueError(
            f'decode_bandwidth: itemsize={itemsize} must be >= 1.')
    try:
        float(itemsize)
    except OverflowError:
        # JSON integers are unbounded; a byte count outside the float
        # range cannot participate in the byte/bandwidth arithmetic.
        raise ValueError(
            f'decode_bandwidth: itemsize={itemsize!r} must be representable '
            f'as a finite float.') from None


def _validate_batch(batch: int) -> None:
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError(
            f'decode_bandwidth: batch={batch!r} must be an int.')
    if batch < 1:
        raise ValueError(
            f'decode_bandwidth: batch={batch} must be >= 1.')
    try:
        float(batch)
    except OverflowError:
        # JSON integers are unbounded; a batch outside the float range
        # cannot participate in the byte/bandwidth arithmetic below.
        raise ValueError(
            f'decode_bandwidth: batch={batch!r} must be representable as a '
            f'finite float.') from None


_STORAGE_OPS = ('memory', 'csa', 'hca')


def _finite_float(fn_name: str, label: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f'{fn_name}: {label} is unrepresentable as a finite float')
    return value


def _resolve_storage(storage, itemsize: int) -> dict[str, float]:
    """Resolve a per-operator storage policy into bytes/element.

    ``None`` keeps the legacy uniform ``itemsize``. A single
    :class:`~kv_quant.QuantSpec` / :class:`~kv_quant.FP8Spec` applies to
    every operator; a mapping may override ``'memory'`` (the recurrent
    state) and/or ``'csa'`` / ``'hca'`` (their read traffic), with ``'*'``
    as the fallback. Metadata (block scales) is charged by
    ``kv_quant.bytes_per_element`` exactly as in the storage contract.
    """
    if storage is None:
        return {op: float(itemsize) for op in _STORAGE_OPS}
    if isinstance(storage, (QuantSpec, FP8Spec)):
        bpe = bytes_per_element(storage)
        return {op: bpe for op in _STORAGE_OPS}
    if not isinstance(storage, dict):
        raise TypeError(
            f'decode_bandwidth: storage must be None, a QuantSpec, an '
            f'FP8Spec or a dict, got {type(storage)!r}')
    unknown = sorted(set(storage) - set(_STORAGE_OPS) - {'*'})
    if unknown:
        raise ValueError(
            f'decode_bandwidth: storage names unknown operator(s) '
            f'{unknown}; choose from {list(_STORAGE_OPS)} or "*".')
    default = storage.get('*', itemsize)
    out: dict[str, float] = {}
    for op in _STORAGE_OPS:
        spec = storage.get(op, default)
        if spec is None:
            out[op] = float(itemsize)
        elif isinstance(spec, (QuantSpec, FP8Spec)):
            out[op] = bytes_per_element(spec)
        elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
            try:
                numeric = float(spec)
            except OverflowError:
                numeric = None
            if numeric is not None and math.isfinite(numeric) and numeric > 0:
                out[op] = numeric
            else:
                raise TypeError(
                    f'decode_bandwidth: storage[{op!r}] must be a QuantSpec, '
                    f'an FP8Spec, a positive bytes/element number or None, '
                    f'got {spec!r}')
        else:
            raise TypeError(
                f'decode_bandwidth: storage[{op!r}] must be a QuantSpec, '
                f'an FP8Spec, a positive bytes/element number or None, '
                f'got {spec!r}')
    return out


def storage_summary(storage, itemsize: int = 4) -> dict:
    """JSON-safe provenance for a storage policy (no tensors)."""
    _validate_itemsize(itemsize)
    bpe = _resolve_storage(storage, itemsize)
    if isinstance(storage, (QuantSpec, FP8Spec)):
        spec = spec_summary(storage)
    elif isinstance(storage, dict):
        spec = {key: (spec_summary(value)
                      if isinstance(value, (QuantSpec, FP8Spec)) else value)
                for key, value in storage.items()}
    else:
        spec = None
    return {
        'itemsize': itemsize,
        'bytes_per_element_by_op': bpe,
        'spec': spec,
    }


def unit_layout(cfg: HybridConfig) -> tuple[str, ...]:
    """The exact layer list the ledger prices.

    Ratio mode -> one repeating unit ``memory*n_kda + csa*n_csa +
    hca*n_hca`` with ``memory`` resolved to ``cfg.memory_type`` (the same
    unit ``HybridAttentionStack._build_layout`` repeats). Explicit layout
    -> ``tuple(cfg.layout)`` verbatim.
    """
    _require_cfg(cfg)
    if cfg.layout is not None:
        return tuple(cfg.layout)
    return ((cfg.memory_type,) * cfg.n_kda
            + ('csa',) * cfg.n_csa
            + ('hca',) * cfg.n_hca)


def attention_tower_parameter_elements(cfg: HybridConfig) -> dict:
    """Exact parameter element count of the attention tower, by layer kind.

    The count is measured from the REAL modules (``build_mixer`` through
    ``HybridAttentionStack``), not re-derived from geometry formulas, so a
    parameterization change can never silently desync the ledger. The
    construction runs under a saved/restored global RNG state: counting
    weights must not perturb a caller's seeded initialization stream.

    Backends are forced to ``'reference'`` for the count: the optional
    FLA / mamba_ssm / fla-rwkv wrappers are contractually shape-identical
    (the backend parity tests enforce matching parameterization), and the
    forced reference build keeps the ledger runnable on a CPU host with no
    optional dependencies installed.

    Returns ``{'total', 'by_kind', 'n_layers', 'layout', 'unit'}`` where
    ``sum(by_kind.values()) == total`` and ``by_kind`` carries one entry
    per layer kind plus ``'norm'`` for the per-layer LayerNorm pairs.
    """
    _require_cfg(cfg)
    layout = unit_layout(cfg)
    if not layout:
        return {'total': 0, 'by_kind': {}, 'n_layers': 0,
                'layout': [], 'unit': []}
    shape_cfg = dataclasses.replace(
        cfg, kda_backend='reference', mamba_backend='reference',
        rwkv_backend='reference')
    rng_state = torch.random.get_rng_state()
    try:
        with torch.no_grad():
            stack = HybridAttentionStack(shape_cfg,
                                         total_layers=len(layout))
    finally:
        torch.random.set_rng_state(rng_state)
    by_kind: dict[str, int] = {}
    for kind, layer in zip(stack.layout, stack.layers):
        by_kind[kind] = by_kind.get(kind, 0) + sum(
            int(p.numel()) for p in layer.parameters())
    by_kind['norm'] = sum(int(p.numel()) for p in stack.norms.parameters())
    total = sum(int(p.numel()) for p in stack.parameters())
    return {'total': total, 'by_kind': by_kind, 'n_layers': len(stack.layout),
            'layout': list(stack.layout), 'unit': list(layout)}


def attention_tower_weight_bytes(cfg: HybridConfig, *,
                                 itemsize: int = 4) -> dict:
    """Element and byte counts of the tower weights at ``itemsize``."""
    _validate_itemsize(itemsize)
    params = attention_tower_parameter_elements(cfg)
    return {
        **params,
        'itemsize': itemsize,
        'bytes': params['total'] * itemsize,
        'bytes_by_kind': {k: v * itemsize
                          for k, v in params['by_kind'].items()},
    }


def decode_traffic_breakdown(cfg: HybridConfig, T: int, *,
                             itemsize: int = 4, storage=None) -> dict:
    """Per-sequence per-token decode traffic split by operator.

    Totals come from the canonical ``lm_consistency`` decode byte model
    (``kv_budget`` per-op reads averaged over the ``T`` prefill queries,
    plus the recurrent-state writes). The split subtracts the CSA / HCA
    read terms (recomputed from the SAME ``kv_budget`` call) from the
    canonical total, so the identity

        ``memory + csa + hca == total_read_elements_per_token``

    holds by construction and is pinned by the regression suite.

    Writes are attributed entirely to the memory route: the canonical
    model amortizes / excludes compressed-block and window writes (see
    ``lm_consistency.decode_elements_per_token``).

    ``storage`` prices each operator's elements independently (e.g. FP8
    recurrent state + fp16 CSA/HCA caches). ``None`` is the legacy uniform
    ``itemsize`` baseline. Returns read/write element and byte dicts keyed
    by ``'memory' / 'csa' / 'hca'`` plus the canonical totals and the
    resolved ``bytes_per_element_by_op``.
    """
    _require_cfg(cfg)
    _validate_itemsize(itemsize)
    bpe = _resolve_storage(storage, itemsize)
    elems = decode_elements_per_token(cfg, T)
    n_mem, n_csa, n_hca = elems['layout']
    spec = decode_read_spec(cfg)
    csa_read = (n_csa * kv_budget.op_read_elements('csa', T, spec)[
        'total_read_elements']) if n_csa else 0.0
    hca_read = (n_hca * kv_budget.op_read_elements('hca', T, spec)[
        'total_read_elements']) if n_hca else 0.0
    total_read = float(elems['read_elements_per_token'])
    memory_read = total_read - csa_read - hca_read
    if memory_read < -1e-6:
        # The split is derived by subtraction; a negative value means the
        # canonical model changed underneath us. Fail loudly instead of
        # returning a fabricated number.
        raise RuntimeError(
            f'decode_bandwidth: internal split invariant violated — memory '
            f'read {memory_read} < 0 (total={total_read}, csa={csa_read}, '
            f'hca={hca_read}); the canonical decode byte model changed.')
    memory_read = max(0.0, memory_read)
    write_total = float(elems['write_elements_per_token'])
    read_by_op = {'memory': memory_read, 'csa': float(csa_read),
                  'hca': float(hca_read)}
    write_by_op = {'memory': write_total, 'csa': 0.0, 'hca': 0.0}
    total_elements = total_read + write_total
    read_bytes_by_op = {
        op: _finite_float('decode_traffic_breakdown', f'{op} read bytes',
                          read_by_op[op] * bpe[op])
        for op in _STORAGE_OPS}
    write_bytes_by_op = {
        op: _finite_float('decode_traffic_breakdown', f'{op} write bytes',
                          write_by_op[op] * bpe[op])
        for op in _STORAGE_OPS}
    read_bytes = _finite_float(
        'decode_traffic_breakdown', 'total read bytes',
        sum(read_bytes_by_op.values()))
    write_bytes = _finite_float(
        'decode_traffic_breakdown', 'total write bytes',
        sum(write_bytes_by_op.values()))
    return {
        'T': int(T),
        'itemsize': itemsize,
        'memory_route': elems['memory_route'],
        'layout': list(elems['layout']),
        'bytes_per_element_by_op': bpe,
        'read_elements_by_op': read_by_op,
        'write_elements_by_op': write_by_op,
        'read_elements_per_token': total_read,
        'write_elements_per_token': write_total,
        'total_elements_per_token': total_elements,
        'read_bytes_by_op': read_bytes_by_op,
        'write_bytes_by_op': write_bytes_by_op,
        'read_bytes_per_token': read_bytes,
        'write_bytes_per_token': write_bytes,
        'total_bytes_per_token': _finite_float(
            'decode_traffic_breakdown', 'total bytes',
            read_bytes + write_bytes),
    }


def decode_step_ledger(cfg: HybridConfig, T: int, *, batch: int = 1,
                       itemsize: int = 4, storage=None,
                       weight_itemsize: int | None = None,
                       include_weights: bool = True) -> dict:
    """The batch-aware decode-step traffic ledger.

    ``total_bytes_per_step = weight_bytes + batch * state_bytes_per_seq``.
    ``bytes_per_sequence_per_token`` (the conventional per-token metric)
    is the step total divided by the batch, i.e.
    ``weight_bytes / batch + state_bytes_per_seq``: at small batch the
    weight term dominates and shrinks with batch; at large batch the
    per-sequence state/cache term saturates it.

    ``storage`` prices the state/cache term per operator (see
    :func:`decode_traffic_breakdown`); ``weight_itemsize`` lets weights use
    a different precision from the ``itemsize`` baseline. ``state_share``
    is the fraction of the step's traffic that is per-sequence
    state/cache; ``weights_state_switch_batch`` (float) is the batch at
    which the two terms are equal (``None`` when there is no state
    traffic). All values are JSON-strict Python scalars.
    """
    _require_cfg(cfg)
    _validate_itemsize(itemsize)
    _validate_batch(batch)
    if weight_itemsize is None:
        weight_itemsize = itemsize
    _validate_itemsize(weight_itemsize)
    traffic = decode_traffic_breakdown(cfg, T, itemsize=itemsize,
                                       storage=storage)
    if include_weights:
        weights = attention_tower_weight_bytes(cfg, itemsize=weight_itemsize)
        weight_elements = int(weights['total'])
        weight_bytes = int(weights['bytes'])
    else:
        weight_elements = 0
        weight_bytes = 0
    state_read_bytes = _finite_float(
        'decode_step_ledger', 'state read bytes per step',
        batch * traffic['read_bytes_per_token'])
    state_write_bytes = _finite_float(
        'decode_step_ledger', 'state write bytes per step',
        batch * traffic['write_bytes_per_token'])
    state_bytes = _finite_float(
        'decode_step_ledger', 'state bytes per step',
        state_read_bytes + state_write_bytes)
    state_bytes_per_token = traffic['total_bytes_per_token']
    total_bytes = _finite_float(
        'decode_step_ledger', 'total bytes per step',
        weight_bytes + state_bytes)
    switch_batch = (weight_bytes / state_bytes_per_token
                    if state_bytes_per_token > 0 else None)
    if switch_batch is not None:
        switch_batch = _finite_float(
            'decode_step_ledger', 'weights/state switch batch', switch_batch)
    state_per_sequence = _finite_float(
        'decode_step_ledger', 'state bytes per sequence per token',
        state_bytes / batch)
    bytes_per_sequence = _finite_float(
        'decode_step_ledger', 'bytes per sequence per token',
        total_bytes / batch)
    read_bytes_by_op_per_step = {
        op: _finite_float(
            'decode_step_ledger', f'{op} read bytes per step',
            batch * traffic['read_bytes_by_op'][op])
        for op in ('memory', 'csa', 'hca')}
    write_bytes_by_op_per_step = {
        op: _finite_float(
            'decode_step_ledger', f'{op} write bytes per step',
            batch * traffic['write_bytes_by_op'][op])
        for op in ('memory', 'csa', 'hca')}
    return {
        'T': int(T),
        'batch': batch,
        'itemsize': itemsize,
        'weight_itemsize': weight_itemsize,
        'memory_route': traffic['memory_route'],
        'layout': traffic['layout'],
        'bytes_per_element_by_op': traffic['bytes_per_element_by_op'],
        'weight_elements': weight_elements,
        'weight_bytes_per_step': weight_bytes,
        'state_read_bytes_per_step': state_read_bytes,
        'state_write_bytes_per_step': state_write_bytes,
        'state_bytes_per_step': state_bytes,
        'total_bytes_per_step': total_bytes,
        'state_bytes_per_sequence_per_token': state_per_sequence,
        'bytes_per_sequence_per_token': bytes_per_sequence,
        'weights_share': (weight_bytes / total_bytes) if total_bytes else 0.0,
        'state_share': (state_bytes / total_bytes) if total_bytes else 0.0,
        'weights_state_switch_batch': switch_batch,
        'read_bytes_by_op_per_step': read_bytes_by_op_per_step,
        'write_bytes_by_op_per_step': write_bytes_by_op_per_step,
    }


def weights_state_switch_batch(cfg: HybridConfig, T: int, *,
                               itemsize: int = 4, storage=None,
                               weight_itemsize: int | None = None
                               ) -> float | None:
    """Batch at which weight traffic equals per-sequence state traffic.

    With a uniform ``itemsize`` and no storage policy the element ratio
    cancels to the legacy value. With ``storage`` (and/or a different
    ``weight_itemsize``) the ratio is computed in BYTES, so FP8 state
    storage moves the switch to a larger batch (weights dominate longer).
    Returns ``None`` when the canonical model attributes zero state traffic
    to the configuration (then weights dominate at every batch).
    """
    _require_cfg(cfg)
    _validate_itemsize(itemsize)
    if weight_itemsize is not None:
        _validate_itemsize(weight_itemsize)
    if storage is None and weight_itemsize is None:
        weight_elements = attention_tower_parameter_elements(cfg)['total']
        state_elements = decode_elements_per_token(cfg, T)[
            'total_elements_per_token']
        if state_elements <= 0:
            return None
        return weight_elements / state_elements
    win = itemsize if weight_itemsize is None else weight_itemsize
    weight_bytes = attention_tower_weight_bytes(cfg, itemsize=win)['bytes']
    traffic = decode_traffic_breakdown(cfg, T, itemsize=itemsize,
                                       storage=storage)
    state_bytes = traffic['total_bytes_per_token']
    if state_bytes <= 0:
        return None
    return weight_bytes / state_bytes


def route_switch_table(base_cfg: HybridConfig, T: int, *,
                       batches: tuple[int, ...] = (1, 4, 16, 64),
                       routes: tuple[str, ...] = MEMORY_ROUTES,
                       itemsize: int = 4, storage=None,
                       weight_itemsize: int | None = None) -> list[dict]:
    """One ledger row per memory route, all else held fixed.

    Each row replaces ONLY ``memory_type`` on ``base_cfg`` (the ratio-mode
    one-knob contract), so the difference between rows is exactly the
    recurrent memory operator at identical CSA/HCA geometry. Explicit
    layouts are rejected: they pin concrete memory kinds and a
    route switch is not expressible there.

    ``storage`` and ``weight_itemsize`` follow
    :func:`decode_step_ledger`; the same policy is applied to every route,
    so the only per-row difference remains the recurrent operator.

    A route whose geometry is incompatible with the base config (or whose
    optional backend is unavailable) is recorded as
    ``{'route': ..., 'status': 'unavailable', 'reason': ...}`` instead of
    aborting the table — a comparison must never silently drop a route.
    """
    _require_cfg(base_cfg)
    _validate_itemsize(itemsize)
    if weight_itemsize is None:
        weight_itemsize = itemsize
    _validate_itemsize(weight_itemsize)
    _resolve_storage(storage, itemsize)
    if base_cfg.layout is not None:
        raise ValueError(
            'route_switch_table requires a ratio-mode config '
            '(layout=None); an explicit layout pins concrete memory kinds, '
            'so there is no route to switch. Build one config per route '
            'instead.')
    if not batches:
        raise ValueError('route_switch_table: batches must be non-empty.')
    for b in batches:
        _validate_batch(b)
    rows: list[dict] = []
    for route in routes:
        if route not in MEMORY_LAYER_KINDS:
            raise ValueError(
                f'route_switch_table: route={route!r} must be one of '
                f'{MEMORY_LAYER_KINDS}.')
        try:
            cfg = dataclasses.replace(base_cfg, memory_type=route)
            weights = attention_tower_weight_bytes(
                cfg, itemsize=weight_itemsize)
            switch = weights_state_switch_batch(
                cfg, T, itemsize=itemsize, storage=storage,
                weight_itemsize=weight_itemsize)
            by_batch: dict[str, dict] = {}
            for b in batches:
                led = decode_step_ledger(
                    cfg, T, batch=b, itemsize=itemsize, storage=storage,
                    weight_itemsize=weight_itemsize, include_weights=True)
                by_batch[str(b)] = {
                    'total_bytes_per_step': led['total_bytes_per_step'],
                    'bytes_per_sequence_per_token':
                        led['bytes_per_sequence_per_token'],
                    'state_share': led['state_share'],
                    'read_bytes_by_op_per_step':
                        led['read_bytes_by_op_per_step'],
                }
            breakdown = decode_traffic_breakdown(
                cfg, T, itemsize=itemsize, storage=storage)
            rows.append({
                'route': route,
                'status': 'ok',
                'weight_elements': int(weights['total']),
                'weight_bytes_per_step': int(weights['bytes']),
                'weight_bytes_by_kind': weights['bytes_by_kind'],
                'bytes_per_element_by_op':
                    breakdown['bytes_per_element_by_op'],
                'state_elements_per_token':
                    breakdown['total_elements_per_token'],
                'state_bytes_per_token': breakdown['total_bytes_per_token'],
                'state_read_bytes_per_token':
                    breakdown['read_bytes_per_token'],
                'state_write_bytes_per_token':
                    breakdown['write_bytes_per_token'],
                'read_bytes_by_op_per_token': breakdown['read_bytes_by_op'],
                'weights_state_switch_batch': switch,
                'by_batch': by_batch,
            })
        except _UNAVAILABLE as exc:
            rows.append({'route': route, 'status': 'unavailable',
                         'reason': f'{type(exc).__name__}: {exc}'})
    return rows


def bandwidth_bound_ms(nbytes: float, bw_gbps: float) -> float:
    """Lower-bound step time (ms) to move ``nbytes`` at ``bw_gbps``.

    This is the memory-side roofline ceiling for one decode step: bytes /
    bandwidth. It says nothing about compute, orchestration, or the naive
    reference's Python overhead — compare it with the roofline module's
    ``efficiency_label`` conventions, not with a wall-clock claim.
    """
    if (isinstance(nbytes, bool)
            or not isinstance(nbytes, (int, float))):
        raise ValueError(
            f'bandwidth_bound_ms: nbytes={nbytes!r} must be a finite '
            f'non-negative number.')
    try:
        nbytes_value = float(nbytes)
    except OverflowError:
        raise ValueError(
            f'bandwidth_bound_ms: nbytes={nbytes!r} must be a finite '
            f'non-negative number.') from None
    if not math.isfinite(nbytes_value) or nbytes_value < 0:
        raise ValueError(
            f'bandwidth_bound_ms: nbytes={nbytes!r} must be a finite '
            f'non-negative number.')
    if (isinstance(bw_gbps, bool)
            or not isinstance(bw_gbps, (int, float))):
        raise ValueError(
            f'bandwidth_bound_ms: bw_gbps={bw_gbps!r} must be finite and '
            f'positive.')
    try:
        bw_value = float(bw_gbps)
    except OverflowError:
        raise ValueError(
            f'bandwidth_bound_ms: bw_gbps={bw_gbps!r} must be finite and '
            f'positive.') from None
    if not math.isfinite(bw_value) or bw_value <= 0:
        raise ValueError(
            f'bandwidth_bound_ms: bw_gbps={bw_gbps!r} must be finite and '
            f'positive.')
    return _finite_float(
        'bandwidth_bound_ms', 'bandwidth-bound time',
        nbytes_value / (bw_value * 1e9) * 1e3)
