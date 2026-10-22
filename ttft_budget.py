"""Time-to-first-token (TTFT) lower-bound ledger for the hybrid tower.

Why this module?
----------------
The repository prices the two serving phases separately: ``prefill_profile``
measures chunked-prefill latency/drift and ``decode_bandwidth`` prices the
steady-state decode traffic. Neither answers the question a serving
operator actually asks first: **how long until the first token, and which
phase owns that time?**

This module composes a *lower bound* on TTFT in two stages:

    TTFT >= prefill_lower_bound + first_decode_lower_bound

* the **prefill** stage must execute the analytic FLOPs
  (``run_kv_cache.prefill_flops`` at the config geometry) and move the
  compulsory bytes (one weight stream + input/output activations + the
  state/cache written by the prompt). It cannot finish faster than the
  larger of its compute and memory times.
* the **first decode** stage reads the cache at position ``T`` — the
  *marginal* budget from ``kv_budget.instant_layout_read_elements``, not
  the prefill average used by the steady-state ledger — plus one recurrent
  state write and another weight stream. In the regime
  ``decode_bandwidth`` targets, that step is memory-bound, so only its
  bandwidth time is charged.

The first-decode distinction matters. ``decode_bandwidth`` uses
``lm_consistency.decode_elements_per_token``, which averages reads over the
``T`` prefill queries. The token that produces the first output is decoded
at position ``T`` and sees ``T // m`` compressed blocks, so on long
contexts the first decode step is more expensive than the steady-state
average. Reporting both makes the difference visible instead of silently
pricing TTFT with an optimistic average.

Storage policy
--------------
``weight_storage`` and ``state_storage`` accept the same
:class:`~kv_quant.QuantSpec` / :class:`~kv_quant.FP8Spec` contracts as the
decode ledger. Weights and activations are charged at ``itemsize`` unless a
weight spec is supplied; the recurrent state and the CSA/HCA caches use the
per-operator policy (``'memory'`` / ``'csa'`` / ``'hca'`` with ``'*'`` as
the fallback). This makes the central FP8 question falsifiable: shrinking
weights changes only the memory term of a *memory-bound* prefill, so the
TTFT speedup is exactly the byte reduction until the compute term takes
over. ``compute_speedup`` models the separate (and explicitly labelled)
assumption that FP8 arithmetic itself is faster; storage alone does not
change the FLOP count.

Scope / honesty
---------------
* Attention-tower scope, mirroring ``decode_bandwidth``: embeddings,
  lm_head, sampling and serving queueing are excluded. TTFT here is the
  model-side lower bound, not a wall-clock prediction.
* ``max(compute, memory)`` per phase assumes perfect overlap inside a
  phase and zero overlap across phases; the sum of two lower bounds is a
  lower bound on TTFT, not a schedule.
* Prefill state bytes are the *retained* cache footprint from
  ``kv_cache_elements`` (one write of every retained slot), not the
  temporary accumulator traffic of the reference Python path.
* A real kernel may keep weights resident across requests; charging one
  weight stream in prefill and one in the first decode is conservative on
  the weight side (the same convention as ``decode_bandwidth``).
"""
from __future__ import annotations

import math

import kv_budget
import kv_quant
from decode_bandwidth import (MEMORY_ROUTES,
                              attention_tower_parameter_elements,
                              storage_summary, unit_layout)
from lm_consistency import (decode_elements_per_token, decode_read_spec,
                            lm_tower_geometry)
from ops_fused import HybridConfig
from run_kv_cache import kv_cache_elements, prefill_flops

__all__ = [
    'SCENARIOS',
    'first_decode_elements',
    'layout_prefill_flops',
    'prefill_state_elements',
    'resolve_scenario',
    'ttft_ledger',
]

_HYBRID_OPS = {
    'kda': 'hybrid_kch',
    'mamba2': 'hybrid_mamba2',
    'rwkv7': 'hybrid_rwkv7',
}

#: Named storage scenarios used by ``run_ttft_budget.py``. The FP8-compute
#: row is a *separate assumption* (2x arithmetic rate), never implied by
#: the storage contract alone.
SCENARIOS = (
    'fp16',
    'fp8-weights',
    'fp8-weights-state',
    'fp8-weights-state-fp8-compute',
)


def _validate_positive_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an int, got {value!r}')
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return value


def _validate_positive_float(value, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive, got {value!r}')
    return float(value)


def _require_cfg(cfg) -> None:
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'ttft_budget: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')


def _memory_route(cfg: HybridConfig) -> str:
    if cfg.layout is not None:
        return next((k for k in cfg.layout if k in MEMORY_ROUTES),
                    cfg.memory_type)
    return cfg.memory_type


def _layout_counts(cfg: HybridConfig) -> tuple[int, int, int]:
    unit = unit_layout(cfg)
    n_mem = sum(1 for kind in unit if kind in MEMORY_ROUTES)
    n_csa = sum(1 for kind in unit if kind == 'csa')
    n_hca = sum(1 for kind in unit if kind == 'hca')
    return n_mem, n_csa, n_hca


def layout_prefill_flops(cfg: HybridConfig, T: int, *,
                         batch: int = 1) -> float:
    """Analytic prefill FLOPs of the resolved layout at length ``T``.

    Uses the same closed form as Exp 3 / Exp 7 (``prefill_flops`` with the
    config geometry from ``lm_consistency.lm_tower_geometry``), multiplied
    by ``batch``. The memory route selects the matching hybrid row, so the
    KDA / Mamba-2 / RWKV-7 comparison stays a one-knob ablation.
    """
    _require_cfg(cfg)
    _validate_positive_int(T, 'T')
    _validate_positive_int(batch, 'batch')
    geo = lm_tower_geometry(cfg)
    op = _HYBRID_OPS[_memory_route(cfg)]
    return float(batch) * float(prefill_flops(op, T, **geo))


def prefill_state_elements(cfg: HybridConfig, T: int, *,
                           mode: str = 'compressed_kv_only') -> dict:
    """Retained state/cache elements written by a ``T``-token prefill.

    Per-operator decomposition (memory route / CSA / HCA) from
    ``run_kv_cache.kv_cache_elements``. ``'compressed_kv_only'`` counts the
    semantic cache (recurrent state, compressed rows); ``'full_accounting'``
    adds the reference runtime buffers (window rings, partial accumulators,
    indexer keys). Elements are the same currency the decode ledger uses;
    multiply by the per-operator bytes/element for storage.
    """
    _require_cfg(cfg)
    _validate_positive_int(T, 'T')
    if mode not in ('compressed_kv_only', 'full_accounting'):
        raise ValueError(
            f"prefill_state_elements: mode={mode!r} must be "
            f"'compressed_kv_only' or 'full_accounting'.")
    geo = lm_tower_geometry(cfg)
    n_mem, n_csa, n_hca = _layout_counts(cfg)
    route = _memory_route(cfg)
    by_op = {
        'memory': float(n_mem) * float(kv_cache_elements(
            route, T, mode=mode, **geo)),
        'csa': (float(n_csa) * float(kv_cache_elements(
            'csa', T, mode=mode, **geo))) if n_csa else 0.0,
        'hca': (float(n_hca) * float(kv_cache_elements(
            'hca', T, mode=mode, **geo))) if n_hca else 0.0,
    }
    return {'by_op': by_op, 'total': sum(by_op.values()), 'mode': mode}


def first_decode_elements(cfg: HybridConfig, T: int) -> dict:
    """Elements read/written by the decode step that produces token ``T``.

    Reads are the *instant* (marginal) budget at position ``T``: the query
    sees ``T // m`` compressed blocks, not the ``0..T-1`` average. The
    memory route's recurrent read is position-independent and follows the
    canonical ``lm_consistency`` convention (KDA reads its state twice;
    Mamba-2 / RWKV-7 read it once). Writes come from
    ``decode_elements_per_token`` (one recurrent-state write per memory
    layer; CSA/HCA writes are amortized by the canonical model).
    """
    _require_cfg(cfg)
    _validate_positive_int(T, 'T')
    geo = lm_tower_geometry(cfg)
    spec = decode_read_spec(cfg)
    n_mem, n_csa, n_hca = _layout_counts(cfg)
    route = _memory_route(cfg)
    one_state = float(kv_cache_elements(
        route, 1, mode='compressed_kv_only', **geo))
    mem_read = (2.0 if route == 'kda' else 1.0) * one_state
    instant = {'memory': float(n_mem) * mem_read, 'csa': 0.0, 'hca': 0.0}
    average = {'memory': float(n_mem) * mem_read, 'csa': 0.0, 'hca': 0.0}
    if n_csa:
        instant['csa'] = float(n_csa) * float(
            kv_budget.instant_read_elements('csa', T, spec)[
                'total_read_elements'])
        average['csa'] = float(n_csa) * float(
            kv_budget.op_read_elements('csa', T, spec)[
                'total_read_elements'])
    if n_hca:
        instant['hca'] = float(n_hca) * float(
            kv_budget.instant_read_elements('hca', T, spec)[
                'total_read_elements'])
        average['hca'] = float(n_hca) * float(
            kv_budget.op_read_elements('hca', T, spec)[
                'total_read_elements'])
    write_total = float(decode_elements_per_token(cfg, T)[
        'write_elements_per_token'])
    return {
        'read_elements_by_op': instant,
        'average_read_elements_by_op': average,
        'read_elements': sum(instant.values()),
        'average_read_elements': sum(average.values()),
        'write_elements': write_total,
        'total_elements': sum(instant.values()) + write_total,
    }


def resolve_scenario(name: str, *, itemsize: int = 2) -> dict:
    """Storage assumptions of a named TTFT scenario.

    ``'fp16'`` is the uniform baseline. ``'fp8-weights'`` stores the tower
    weights in MXFP8 but keeps activations/state at ``itemsize``.
    ``'fp8-weights-state'`` additionally stores the recurrent state and the
    CSA/HCA caches in MXFP8. ``'fp8-weights-state-fp8-compute'`` keeps the
    same storage and assumes a 2x FP8 arithmetic rate — a separate hardware
    assumption recorded explicitly as ``compute_speedup``.
    """
    _validate_positive_int(itemsize, 'itemsize')
    spec = kv_quant.FP8Spec()
    if name == 'fp16':
        return {'weight_storage': None, 'state_storage': None,
                'compute_speedup': 1.0}
    if name == 'fp8-weights':
        return {'weight_storage': spec, 'state_storage': None,
                'compute_speedup': 1.0}
    if name == 'fp8-weights-state':
        return {'weight_storage': spec, 'state_storage': {'*': spec},
                'compute_speedup': 1.0}
    if name == 'fp8-weights-state-fp8-compute':
        return {'weight_storage': spec, 'state_storage': {'*': spec},
                'compute_speedup': 2.0}
    raise ValueError(
        f"resolve_scenario: name={name!r} must be one of {list(SCENARIOS)}.")


def ttft_ledger(cfg: HybridConfig, T: int, *, batch: int = 1,
                itemsize: int = 2, weight_storage=None, state_storage=None,
                activation_itemsize: int | None = None,
                peak_tflops: float, bw_gbps: float,
                compute_speedup: float = 1.0,
                state_mode: str = 'compressed_kv_only',
                weight_elements: int | None = None) -> dict:
    """Two-stage TTFT lower bound with a per-operator storage policy.

    ``itemsize`` prices activations and (without ``weight_storage``) tower
    weights. ``activation_itemsize`` overrides only the activation term.
    ``peak_tflops`` is the assumed arithmetic rate of the compute dtype;
    ``compute_speedup`` scales it (e.g. 2.0 for an FP8-arithmetic
    assumption). ``bw_gbps`` is the streaming bandwidth used for both
    phases. ``weight_elements`` may be supplied by a sweep to reuse the
    parameter count; when omitted it is measured from the real modules.
    """
    _require_cfg(cfg)
    _validate_positive_int(T, 'T')
    _validate_positive_int(batch, 'batch')
    _validate_positive_int(itemsize, 'itemsize')
    if activation_itemsize is None:
        activation_itemsize = itemsize
    _validate_positive_int(activation_itemsize, 'activation_itemsize')
    peak_tflops = _validate_positive_float(peak_tflops, 'peak_tflops')
    bw_gbps = _validate_positive_float(bw_gbps, 'bw_gbps')
    compute_speedup = _validate_positive_float(compute_speedup,
                                               'compute_speedup')
    if state_mode not in ('compressed_kv_only', 'full_accounting'):
        raise ValueError(
            f"ttft_ledger: state_mode={state_mode!r} must be "
            f"'compressed_kv_only' or 'full_accounting'.")
    for name, spec in (('weight_storage', weight_storage),
                       ('state_storage', state_storage)):
        if spec is not None and not isinstance(
                spec, (kv_quant.QuantSpec, kv_quant.FP8Spec, dict)):
            raise TypeError(
                f'ttft_ledger: {name} must be None, a QuantSpec, an '
                f'FP8Spec or a dict, got {type(spec)!r}.')

    if weight_elements is None:
        weight_elements = attention_tower_parameter_elements(cfg)['total']
    if (isinstance(weight_elements, bool)
            or not isinstance(weight_elements, int)
            or weight_elements < 0):
        raise ValueError(
            f'ttft_ledger: weight_elements={weight_elements!r} must be a '
            f'non-negative int.')

    if weight_storage is None:
        weight_bpe = float(itemsize)
        weight_spec = None
    else:
        weight_bpe = kv_quant.bytes_per_element(weight_storage)
        weight_spec = kv_quant.spec_summary(weight_storage)
    state_summary = storage_summary(state_storage, itemsize)
    state_bpe = state_summary['bytes_per_element_by_op']

    weight_bytes = float(weight_elements) * weight_bpe
    activation_bytes = (2.0 * float(batch) * float(T) * float(cfg.d_model)
                        * float(activation_itemsize))
    state = prefill_state_elements(cfg, T, mode=state_mode)
    state_bytes_by_op = {
        op: elements * state_bpe[op]
        for op, elements in state['by_op'].items()
    }
    state_bytes = sum(state_bytes_by_op.values())
    prefill_bytes = weight_bytes + activation_bytes + state_bytes

    flops_total = layout_prefill_flops(cfg, T, batch=batch)
    effective_peak_tflops = peak_tflops * compute_speedup
    compute_ms = flops_total / (effective_peak_tflops * 1e12) * 1e3
    memory_ms = prefill_bytes / (bw_gbps * 1e9) * 1e3
    prefill_ms = max(compute_ms, memory_ms)
    prefill_bound = 'compute' if compute_ms >= memory_ms else 'memory'
    arithmetic_intensity = (flops_total / prefill_bytes
                            if prefill_bytes > 0 else math.inf)
    ridge = effective_peak_tflops * 1e12 / (bw_gbps * 1e9)

    first = first_decode_elements(cfg, T)
    first_read_bytes_by_op = {
        op: elements * state_bpe[op]
        for op, elements in first['read_elements_by_op'].items()
    }
    first_write_bytes = first['write_elements'] * state_bpe['memory']
    first_state_bytes = (sum(first_read_bytes_by_op.values())
                         + first_write_bytes)
    first_decode_bytes = weight_bytes + float(batch) * first_state_bytes
    first_decode_ms = first_decode_bytes / (bw_gbps * 1e9) * 1e3
    instant_vs_average = (
        first['read_elements'] / first['average_read_elements']
        if first['average_read_elements'] > 0 else None)

    ttft_ms = prefill_ms + first_decode_ms
    return {
        'T': int(T),
        'batch': int(batch),
        'itemsize': int(itemsize),
        'activation_itemsize': int(activation_itemsize),
        'memory_route': _memory_route(cfg),
        'layout_unit': list(unit_layout(cfg)),
        'layout_counts': list(_layout_counts(cfg)),
        'storage': {
            'weight_bytes_per_element': weight_bpe,
            'weight_spec': weight_spec,
            'state': state_summary,
        },
        'machine': {
            'peak_tflops': peak_tflops,
            'compute_speedup': compute_speedup,
            'effective_peak_tflops': effective_peak_tflops,
            'bw_gbps': bw_gbps,
            'ridge_flops_per_byte': ridge,
        },
        'weights': {
            'elements': int(weight_elements),
            'bytes': weight_bytes,
            'bytes_per_element': weight_bpe,
        },
        'prefill': {
            'flops': flops_total,
            'arithmetic_intensity_flops_per_byte': arithmetic_intensity,
            'compute_ms': compute_ms,
            'memory_ms': memory_ms,
            'lower_bound_ms': prefill_ms,
            'bound': prefill_bound,
            'state_mode': state_mode,
            'state_elements_by_op': state['by_op'],
            'state_bytes_by_op': state_bytes_by_op,
            'bytes': {
                'weights': weight_bytes,
                'activations': activation_bytes,
                'state': state_bytes,
                'total': prefill_bytes,
            },
        },
        'first_decode': {
            'read_elements_by_op': first['read_elements_by_op'],
            'average_read_elements_by_op': first[
                'average_read_elements_by_op'],
            'read_elements': first['read_elements'],
            'average_read_elements': first['average_read_elements'],
            'instant_vs_average_reads': instant_vs_average,
            'write_elements': first['write_elements'],
            'read_bytes_by_op': first_read_bytes_by_op,
            'write_bytes': first_write_bytes,
            'bytes': {
                'weights': weight_bytes,
                'state': first_state_bytes,
                'total': first_decode_bytes,
            },
            'memory_bound_ms': first_decode_ms,
        },
        'ttft_lower_bound_ms': ttft_ms,
        'prefill_share': prefill_ms / ttft_ms if ttft_ms > 0 else 0.0,
        'first_decode_share': (first_decode_ms / ttft_ms
                               if ttft_ms > 0 else 0.0),
        'notes': [
            'attention-tower scope; embeddings / lm_head / sampling / '
            'queueing excluded',
            'prefill = max(analytic compute, compulsory memory); first '
            'decode = memory-bound step (weights + cache at position T)',
            'first-decode reads use the marginal budget at t=T; '
            'average_read_elements is the steady-state comparison',
            'storage changes bytes only; compute_speedup is a separate, '
            'explicit arithmetic assumption',
            'TTFT is a two-phase lower bound, not a wall-clock prediction',
        ],
    }
