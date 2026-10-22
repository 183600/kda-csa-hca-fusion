"""Analytical KV read-budget accounting for the fused attention operators.

Why a separate budget accounting?
---------------------------------
Experiment 3 answers "how much KV *state* must be retained" and Experiment 7
answers "how close to the hardware roofline is the reference implementation".
Neither answers the design question behind the top-k knob: **how many KV
elements does one query actually read, and which term dominates?** For CSA
the answer is not simply ``topk * c``:

* the sparse branch reads ``min(topk, visible_blocks)`` selected compressed
  entries, and
* the lightning indexer must *score every visible compressed block* before it
  can select the top-k, reading ``visible_block_entries * c_i * n_idx_heads``
  elements per query (the reference implementation scores all blocks even
  for ``topk=0``; the gather is only skipped afterwards).

At long context the O(T) selection scan overtakes the O(topk) selected reads,
so "cutting the top-k budget" alone stops reducing reads. KDA is the other
extreme: the delta-rule state is read twice per step (masked delta read +
output read) and never grows with T.

Currency definition (the honesty contract)
------------------------------------------
* One *element* is one scalar of the KV-side representation: a compressed
  shared-KV vector counts once (it serves as both key and value), a
  sliding-window token contributes ``c`` elements, a KDA state contributes
  ``hv * d_k * d_v`` elements.
* Reads are counted per query and averaged uniformly over positions
  ``0..T-1``.
* Causal conventions match the implementations: softmax and the sliding
  window include the query's own token (the implementations do), compressed
  blocks are strictly preceding (DeepSeek-V4 Eq. 16, as implemented by
  ``ops_csa._causal_block_mask`` / decoded by ``run_kv_cache``).
* Query-side projections (q/w) are excluded — they are compute, not KV
  retrieval.
* This is an analytical model, not profiled traffic: library-internal
  materialization (``unfold``, temporaries) is out of scope, mirroring the
  compulsory-traffic caveat of Exp 7.

Prefill / decode unification
----------------------------
The averaged prefill functions (:func:`op_read_elements`,
:func:`layout_read_elements`) and the decode path share ONE primitive: the
*marginal* budget :func:`instant_read_elements` charges exactly the elements
the query at position ``t`` reads (compressed blocks strictly preceding,
window inclusive of the own token — the same masks the implementations use).
Two identities hold by construction and are pinned by tests:

* ``op_read_elements(op, T, spec)`` equals the mean of
  ``instant_read_elements(op, t, spec)`` over ``t = 0..T-1`` (the prefill
  average is the rolling mean of the decode marginal), and
* ``instant_read_elements(op, T, spec)`` is the analytical budget of the
  FIRST token decoded after a prefill of ``T`` tokens.

:func:`windowed_read_elements` exposes the mean marginal over an arbitrary
decode window ``[t0, t1)`` in O(1) (closed-form subtraction of the averaged
functions), which is what a measured per-token decode latency should be
compared against — the prefill average at ``T`` understates the cost of the
last decoded positions because it mixes in the cheap early ones.

The low-level visible-pair counters are imported from :mod:`run_kv_cache`
(``causal_block_entries`` / ``causal_selected_entries``) so the budget model
and the Exp 3 KV/FLOPs accounting can never disagree on causality.
"""

from __future__ import annotations

import math

from kv_quant import quantized_row_bytes
from run_kv_cache import causal_block_entries, causal_selected_entries


def _validate_T(T: int, fn_name: str) -> None:
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'{fn_name}: T must be a non-negative int, got {T!r}')


def _validate_positive(value: int, name: str, fn_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            f'{fn_name}: {name} must be a positive int, got {value!r}')


def sliding_window_entries(T: int, window: int) -> int:
    """Sum over queries of ``min(t + 1, window)`` window entries.

    Matches ``ops_csa._sliding_window_scores``: the window is
    inclusive of the query's own token, so query ``t`` sees
    ``min(t + 1, window)`` real entries (left padding is masked).
    """
    _validate_T(T, 'sliding_window_entries')
    _validate_positive(window, 'window', 'sliding_window_entries')
    if T == 0:
        return 0
    if T <= window:
        return T * (T + 1) // 2
    return window * (window + 1) // 2 + (T - window) * window


def softmax_read_elements(T: int, n_kv_heads: int, d_k: int, d_v: int) -> float:
    """Average KV elements read per query by a dense causal softmax layer."""
    _validate_T(T, 'softmax_read_elements')
    _validate_positive(n_kv_heads, 'n_kv_heads', 'softmax_read_elements')
    _validate_positive(d_k, 'd_k', 'softmax_read_elements')
    _validate_positive(d_v, 'd_v', 'softmax_read_elements')
    if T == 0:
        return 0.0
    # Query t reads t+1 key/value pairs (inclusive causal), averaged over T.
    return (T + 1) / 2.0 * n_kv_heads * (d_k + d_v)


def kda_state_elements(hv: int, d_k: int, d_v: int) -> int:
    """Elements of the delta-rule recurrent state ``[hv, d_k, d_v]``."""
    _validate_positive(hv, 'hv', 'kda_state_elements')
    _validate_positive(d_k, 'd_k', 'kda_state_elements')
    _validate_positive(d_v, 'd_v', 'kda_state_elements')
    return hv * d_k * d_v


def mamba2_state_elements(nheads: int, headdim: int, d_state: int) -> int:
    """Elements of the Mamba-2 recurrent state ``[nheads, headdim, d_state]``.

    The short-conv lookback is excluded, matching KDA's accounting (its
    short conv is also outside the recurrent-state currency).
    """
    _validate_positive(nheads, 'nheads', 'mamba2_state_elements')
    _validate_positive(headdim, 'headdim', 'mamba2_state_elements')
    _validate_positive(d_state, 'd_state', 'mamba2_state_elements')
    return nheads * headdim * d_state


def mamba2_read_elements(T: int, nheads: int, headdim: int,
                         d_state: int) -> float:
    """Average state elements READ per query by a Mamba-2 SSD step.

    ``ops_mamba.naive_mamba2_scan`` reads the state twice per token: once
    for the decayed update ``S = decay * S + x`` and once for the readout
    ``C @ S^T``. The state is O(1) in T, exactly like KDA's delta-rule
    state; only the value-head geometry differs.
    """
    _validate_T(T, 'mamba2_read_elements')
    if T == 0:
        return 0.0
    return 2.0 * mamba2_state_elements(nheads, headdim, d_state)


def kda_read_elements(T: int, hv: int, d_k: int, d_v: int) -> float:
    """Average state elements READ per query by the KDA delta-rule step.

    Each step reads the state twice — once for the masked delta term
    ``(k * S).sum(-2)`` and once for the output ``q @ S`` (see
    ``ops_kda._recurrent_kda_loop_eager``). The state is O(1) in T;
    writes (``hv * d_k * d_v`` per step) are reported separately by
    :func:`op_read_elements` as ``state_elements``.
    """
    _validate_T(T, 'kda_read_elements')
    if T == 0:
        return 0.0
    return 2.0 * kda_state_elements(hv, d_k, d_v)


def csa_read_elements(T: int, *, m: int, topk: int, c: int, c_i: int,
                      n_idx_heads: int = 1, sliding_window: int = 0) -> dict:
    """Average KV-side reads per query for one CSA layer.

    Components:

    * ``selected`` — the top-k gather over compressed shared-KV blocks
      (strictly-preceding mask, capped at ``topk``).
    * ``window`` — the sliding-window branch.
    * ``indexer`` — the lightning-indexer scan over ALL visible compressed
      blocks (``visible_pairs * c_i * n_idx_heads``); this runs before the
      top-k gather in the reference implementation, including ``topk=0``.
    """
    _validate_T(T, 'csa_read_elements')
    _validate_positive(m, 'm', 'csa_read_elements')
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 0:
        raise ValueError(
            f'csa_read_elements: topk must be a non-negative int, got '
            f'{topk!r}')
    _validate_positive(c, 'c', 'csa_read_elements')
    _validate_positive(c_i, 'c_i', 'csa_read_elements')
    _validate_positive(n_idx_heads, 'n_idx_heads', 'csa_read_elements')
    if (not isinstance(sliding_window, int)
            or isinstance(sliding_window, bool) or sliding_window < 0):
        raise ValueError(
            f'csa_read_elements: sliding_window must be a non-negative int, '
            f'got {sliding_window!r}')
    if T == 0:
        return {'selected_read_elements': 0.0, 'window_read_elements': 0.0,
                'core_read_elements': 0.0, 'indexer_read_elements': 0.0,
                'total_read_elements': 0.0}
    selected = causal_selected_entries(T, m, topk) * float(c)
    window = (sliding_window_entries(T, sliding_window) * float(c)
              if sliding_window > 0 else 0.0)
    indexer = causal_block_entries(T, m) * float(c_i) * n_idx_heads
    core = selected + window
    return {
        'selected_read_elements': selected / T,
        'window_read_elements': window / T,
        'core_read_elements': core / T,
        'indexer_read_elements': indexer / T,
        'total_read_elements': (core + indexer) / T,
    }


def hca_read_elements(T: int, *, m2: int, c: int,
                      sliding_window: int = 0) -> dict:
    """Average KV-side reads per query for one HCA layer.

    HCA keeps ALL compressed blocks (no top-k, no indexer) and adds the
    same sliding-window branch. Blocks are strictly preceding.
    """
    _validate_T(T, 'hca_read_elements')
    _validate_positive(m2, 'm2', 'hca_read_elements')
    _validate_positive(c, 'c', 'hca_read_elements')
    if (not isinstance(sliding_window, int)
            or isinstance(sliding_window, bool) or sliding_window < 0):
        raise ValueError(
            f'hca_read_elements: sliding_window must be a non-negative int, '
            f'got {sliding_window!r}')
    if T == 0:
        return {'dense_read_elements': 0.0, 'window_read_elements': 0.0,
                'core_read_elements': 0.0, 'indexer_read_elements': 0.0,
                'total_read_elements': 0.0}
    dense = causal_block_entries(T, m2) * float(c)
    window = (sliding_window_entries(T, sliding_window) * float(c)
              if sliding_window > 0 else 0.0)
    core = dense + window
    return {
        'dense_read_elements': dense / T,
        'window_read_elements': window / T,
        'core_read_elements': core / T,
        'indexer_read_elements': 0.0,
        'total_read_elements': core / T,
    }


# Layout rows are ratio-based KDA:CSA:HCA; the per-op API additionally
# prices the alternative Mamba-2 memory route so the two routes can be
# compared on the same currency without changing existing layout rows.
_OP_FUNCS = ('softmax', 'kda', 'csa', 'hca')
_BUDGET_OPS = ('softmax', 'kda', 'mamba2', 'csa', 'hca')
_MEMORY_OPS = ('kda', 'mamba2')


def op_read_elements(op: str, T: int, spec: dict) -> dict:
    """Dispatch one operator's read budget at context length ``T``.

    ``spec`` is a geometry dict with one sub-dict per operator (see
    ``run_budget_frontier`` for the MQAR / paper-scale builders). Returns a
    uniform row: ``op``, ``T``, ``core_read_elements`` (KV actually
    attended / state read), ``indexer_read_elements`` (selection scan, CSA
    only), ``state_elements`` (KDA state size, else ``None``) and
    ``total_read_elements``.
    """
    if op not in _BUDGET_OPS:
        raise ValueError(
            f"op_read_elements: op={op!r} must be one of "
            f"{list(_BUDGET_OPS)}.")
    if op not in spec:
        raise ValueError(
            f"op_read_elements: spec is missing the {op!r} geometry block "
            f"(keys present: {sorted(spec)}).")
    geo = spec[op]
    components = {}
    if op == 'softmax':
        core = softmax_read_elements(T, geo['n_kv_heads'], geo['d_k'],
                                     geo['d_v'])
        indexer = 0.0
        state = None
    elif op == 'kda':
        core = kda_read_elements(T, geo['hv'], geo['d_k'], geo['d_v'])
        state = kda_state_elements(geo['hv'], geo['d_k'], geo['d_v'])
        indexer = 0.0
    elif op == 'mamba2':
        core = mamba2_read_elements(
            T, geo['nheads'], geo['headdim'], geo['d_state'])
        state = mamba2_state_elements(
            geo['nheads'], geo['headdim'], geo['d_state'])
        indexer = 0.0
    elif op == 'csa':
        parts = csa_read_elements(
            T, m=geo['m'], topk=geo['topk'], c=geo['c'], c_i=geo['c_i'],
            n_idx_heads=geo.get('n_idx_heads', 1),
            sliding_window=geo.get('sliding_window', 0))
        core = parts['core_read_elements']
        indexer = parts['indexer_read_elements']
        state = None
        components = {
            'selected_read_elements': parts['selected_read_elements'],
            'window_read_elements': parts['window_read_elements'],
        }
    else:  # hca
        parts = hca_read_elements(
            T, m2=geo['m2'], c=geo['c'],
            sliding_window=geo.get('sliding_window', 0))
        core = parts['core_read_elements']
        indexer = parts['indexer_read_elements']
        state = None
        components = {
            'dense_read_elements': parts['dense_read_elements'],
            'window_read_elements': parts['window_read_elements'],
        }
    return {
        'op': op,
        'T': int(T),
        'core_read_elements': core,
        'indexer_read_elements': indexer,
        'state_elements': state,
        'total_read_elements': core + indexer,
        **components,
    }


def _validate_t(t, fn_name: str) -> None:
    if not isinstance(t, int) or isinstance(t, bool) or t < 0:
        raise ValueError(f'{fn_name}: t must be a non-negative int, got {t!r}')


def _instant_op_components(op: str, t: int, geo: dict) -> dict:
    """Per-position (marginal) read components for one operator.

    Called only after :func:`op_read_elements` has validated the geometry
    (at ``T = max(t, 1)``), so the raw ``geo`` lookups below cannot hit an
    unvalidated value.
    """
    if op == 'softmax':
        core = (t + 1) * geo['n_kv_heads'] * (geo['d_k'] + geo['d_v'])
        return {'core_read_elements': float(core),
                'indexer_read_elements': 0.0,
                'total_read_elements': float(core)}
    if op == 'kda':
        # Every recurrent step reads the state twice (masked delta read +
        # output read); the delta-rule extreme is position-independent.
        core = 2.0 * kda_state_elements(geo['hv'], geo['d_k'], geo['d_v'])
        return {'core_read_elements': core,
                'indexer_read_elements': 0.0,
                'total_read_elements': core}
    if op == 'mamba2':
        # Same two-read contract as KDA: decayed update + C @ S^T readout.
        core = 2.0 * mamba2_state_elements(
            geo['nheads'], geo['headdim'], geo['d_state'])
        return {'core_read_elements': core,
                'indexer_read_elements': 0.0,
                'total_read_elements': core}
    if op == 'csa':
        selected = min(geo['topk'], t // geo['m']) * float(geo['c'])
        window = (min(t + 1, geo['sliding_window']) * float(geo['c'])
                  if geo.get('sliding_window', 0) > 0 else 0.0)
        indexer = (t // geo['m']) * float(geo['c_i']) \
            * geo.get('n_idx_heads', 1)
        return {
            'selected_read_elements': selected,
            'window_read_elements': window,
            'core_read_elements': selected + window,
            'indexer_read_elements': indexer,
            'total_read_elements': selected + window + indexer,
        }
    # hca
    dense = (t // geo['m2']) * float(geo['c'])
    window = (min(t + 1, geo['sliding_window']) * float(geo['c'])
              if geo.get('sliding_window', 0) > 0 else 0.0)
    return {
        'dense_read_elements': dense,
        'window_read_elements': window,
        'core_read_elements': dense + window,
        'indexer_read_elements': 0.0,
        'total_read_elements': dense + window,
    }


def instant_read_elements(op: str, t: int, spec: dict) -> dict:
    """KV-side reads charged to the query at position ``t`` (marginal form).

    This is the decode-side primitive of the budget model: the query at
    position ``t`` sees ``t // m`` strictly-preceding compressed blocks
    (``t // m2`` for HCA), ``min(t + 1, sliding_window)`` window entries,
    and — for CSA — scores every visible block with the lightning indexer.
    Averaging over ``t = 0..T-1`` reproduces :func:`op_read_elements`
    exactly (prefill/decode unification), and evaluating at ``t = T`` is the
    budget of the first token decoded after a prefill of ``T`` tokens.

    Geometry validation is delegated to :func:`op_read_elements` (the
    checks are position-independent), so invalid specs raise the same
    ``ValueError`` here as in the averaged path.
    """
    if op not in _BUDGET_OPS:
        raise ValueError(
            f"instant_read_elements: op={op!r} must be one of "
            f"{list(_BUDGET_OPS)}.")
    if op not in spec:
        raise ValueError(
            f"instant_read_elements: spec is missing the {op!r} geometry "
            f"block (keys present: {sorted(spec)}).")
    _validate_t(t, 'instant_read_elements')
    ref = op_read_elements(op, t if t > 0 else 1, spec)
    return {'op': op, 't': int(t), 'state_elements': ref['state_elements'],
            **_instant_op_components(op, t, spec[op])}


def windowed_read_elements(op: str, t0: int, t1: int, spec: dict) -> dict:
    """Mean marginal budget over decode positions ``t0 <= t < t1``.

    ``windowed_read_elements(op, 0, T, spec)`` is component-wise equal to
    :func:`op_read_elements` — the prefill average IS the rolling mean of
    the decode marginal. For ``t0 = T`` (a decode window after a prefill of
    ``T`` tokens) it gives the budget the measured per-token decode latency
    should be compared against.

    Computed in O(1) as ``(t1 * avg(t1) - t0 * avg(t0)) / (t1 - t0)`` from
    the closed-form averaged functions, which is exact because every
    component is additive over positions.
    """
    _validate_t(t0, 'windowed_read_elements')
    _validate_t(t1, 'windowed_read_elements')
    if t1 <= t0:
        raise ValueError(
            f'windowed_read_elements: need 0 <= t0 < t1, got '
            f't0={t0!r}, t1={t1!r}')
    row0 = op_read_elements(op, t0, spec)
    row1 = op_read_elements(op, t1, spec)
    out = {'op': op, 't0': int(t0), 't1': int(t1),
           'state_elements': row1['state_elements']}
    for key, v1 in row1.items():
        if key in ('op', 'T', 'state_elements'):
            continue
        out[key] = (t1 * float(v1) - t0 * float(row0[key])) / (t1 - t0)
    return out


def _validate_ratio(ratio, fn_name: str) -> tuple:
    if not (isinstance(ratio, (tuple, list)) and len(ratio) == 3):
        raise ValueError(
            f'{fn_name}: ratio must be a 3-sequence, got {ratio!r}')
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0
               for n in ratio):
        raise ValueError(
            f'{fn_name}: ratio components must be non-negative ints, '
            f'got {ratio!r}')
    n_kda, n_csa, n_hca = (int(n) for n in ratio)
    if n_kda + n_csa + n_hca == 0:
        raise ValueError(f'{fn_name}: empty layout')
    return n_kda, n_csa, n_hca


def _validate_memory_op(memory_op, fn_name: str) -> str:
    if memory_op not in _MEMORY_OPS:
        raise ValueError(
            f'{fn_name}: memory_op={memory_op!r} must be one of '
            f'{list(_MEMORY_OPS)}.')
    return memory_op


def _layout_row(time_key: str, time_value, n_kda: int, n_csa: int,
                n_hca: int, per_layer: dict, memory_op: str = 'kda') -> dict:
    """Shared row shape for the averaged / marginal / windowed layout views.

    ``n_kda`` is the number of memory slots (the ratio's first component);
    ``memory_op`` says which operator fills them. The legacy
    ``kda_read_elements`` / ``n_kda`` fields remain the memory-slot values
    for the default KDA route, while ``memory_read_elements`` is route
    neutral and ``mamba2_read_elements`` is populated only on that route.
    """
    memory = n_kda * per_layer[memory_op]
    csa = n_csa * per_layer['csa']
    hca = n_hca * per_layer['hca']
    return {
        time_key: time_value,
        'n_memory': n_kda,
        'n_kda': n_kda if memory_op == 'kda' else 0,
        'n_mamba2': n_kda if memory_op == 'mamba2' else 0,
        'n_csa': n_csa,
        'n_hca': n_hca,
        'memory_op': memory_op,
        'memory_read_elements': memory,
        'kda_read_elements': memory if memory_op == 'kda' else 0.0,
        'mamba2_read_elements': memory if memory_op == 'mamba2' else 0.0,
        'csa_read_elements': csa,
        'hca_read_elements': hca,
        'total_read_elements': memory + csa + hca,
    }


def layout_read_elements(T: int, ratio, spec: dict, *,
                         memory_op: str = 'kda') -> dict:
    """Read budget of a ``(n_kda, n_csa, n_hca)`` hybrid layout.

    The first ratio component is the number of *memory* slots; ``memory_op``
    chooses the recurrent operator that fills them (``'kda'`` default, so
    every historical result is unchanged, or ``'mamba2'`` for the
    alternative route). The layout budget is the per-layer sum because
    layers execute sequentially and each query passes through every layer
    once.
    """
    _validate_memory_op(memory_op, 'layout_read_elements')
    n_kda, n_csa, n_hca = _validate_ratio(ratio, 'layout_read_elements')
    per_layer = {
        op: op_read_elements(op, T, spec)['total_read_elements']
        for op in ('csa', 'hca', memory_op)}
    return _layout_row('T', int(T), n_kda, n_csa, n_hca, per_layer,
                       memory_op)


def instant_layout_read_elements(ratio, t: int, spec: dict, *,
                                 memory_op: str = 'kda') -> dict:
    """Marginal layout budget at position ``t`` (the first decode step after
    a ``t``-token prefill). Layers execute sequentially, so per-layer
    marginal budgets add.
    """
    _validate_memory_op(memory_op, 'instant_layout_read_elements')
    _validate_t(t, 'instant_layout_read_elements')
    n_kda, n_csa, n_hca = _validate_ratio(
        ratio, 'instant_layout_read_elements')
    per_layer = {
        op: instant_read_elements(op, t, spec)['total_read_elements']
        for op in ('csa', 'hca', memory_op)}
    return _layout_row('t', int(t), n_kda, n_csa, n_hca, per_layer,
                       memory_op)


def windowed_layout_read_elements(ratio, t0: int, t1: int,
                                  spec: dict, *,
                                  memory_op: str = 'kda') -> dict:
    """Mean marginal layout budget over decode positions ``[t0, t1)``.

    Component-wise ``windowed_layout_read_elements(ratio, 0, T, spec)``
    equals :func:`layout_read_elements`, and ``t0 = T`` gives the budget of
    a decode window after a ``T``-token prefill. ``memory_op`` selects the
    recurrent route exactly as in :func:`layout_read_elements`.
    """
    _validate_memory_op(memory_op, 'windowed_layout_read_elements')
    _validate_t(t0, 'windowed_layout_read_elements')
    _validate_t(t1, 'windowed_layout_read_elements')
    if t1 <= t0:
        raise ValueError(
            f'windowed_layout_read_elements: need 0 <= t0 < t1, got '
            f't0={t0!r}, t1={t1!r}')
    row0 = layout_read_elements(t0, ratio, spec, memory_op=memory_op)
    row1 = layout_read_elements(t1, ratio, spec, memory_op=memory_op)
    out = {'t0': int(t0), 't1': int(t1), 'n_memory': row1['n_memory'],
           'n_kda': row1['n_kda'], 'n_mamba2': row1['n_mamba2'],
           'n_csa': row1['n_csa'], 'n_hca': row1['n_hca'],
           'memory_op': memory_op}
    for key in ('memory_read_elements', 'kda_read_elements',
                'mamba2_read_elements', 'csa_read_elements',
                'hca_read_elements', 'total_read_elements'):
        out[key] = (t1 * row1[key] - t0 * row0[key]) / (t1 - t0)
    return out


def op_read_bytes(op: str, T: int, spec: dict, quant=None) -> dict:
    """Byte view of :func:`op_read_elements` under a quantization policy.

    Element counts are converted with :func:`kv_quant.quantized_row_bytes`:
    each ``<stem>_read_elements`` gains a ``<stem>_read_bytes`` sibling,
    ``state_elements`` (KDA / Mamba-2) gains ``state_bytes``, and
    ``total_read_bytes`` sums the read components (retained state
    excluded). ``quant=None`` prices every component at fp16 (2 bytes /
    element); a :class:`kv_quant.QuantSpec` applies to all components; a
    mapping prices per component stem (``'core'``, ``'indexer'``,
    ``'selected'``, ``'window'``, ``'state'``, ...) with an optional
    ``'*'`` default. Component names that do not exist for ``op`` raise
    ``ValueError`` (typo protection).
    """
    row = dict(op_read_elements(op, T, spec))
    row.update(quantized_row_bytes(row, quant))
    return row


def layout_read_bytes(T: int, ratio, spec: dict, quant=None, *,
                      memory_op: str = 'kda') -> dict:
    """Byte view of :func:`layout_read_elements` under a quantization policy.

    Layout rows aggregate by operator, so the policy stems are the memory
    routes (``'kda'``, ``'csa'``, ``'hca'``) — for a component-level
    (selected / indexer / window) treatment use :func:`op_read_bytes` per
    operator. Same ``quant`` semantics as :func:`op_read_bytes`.
    """
    row = dict(layout_read_elements(T, ratio, spec, memory_op=memory_op))
    row.update(quantized_row_bytes(row, quant))
    return row


def _average_ranks(values) -> list:
    """Average ranks (1-based) with ties sharing the mean rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation with average-rank tie handling.

    Returns ``None`` when the correlation is undefined (fewer than two
    pairs, unequal lengths raise, or either side is constant) — the
    repo-wide convention that an undefined statistic is ``None`` rather
    than a misleading 0.
    """
    if len(xs) != len(ys):
        raise ValueError(
            f'spearman: length mismatch {len(xs)} vs {len(ys)}')
    n = len(xs)
    if n < 2:
        return None
    rx = _average_ranks(list(xs))
    ry = _average_ranks(list(ys))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)
