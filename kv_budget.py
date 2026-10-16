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

The low-level visible-pair counters are imported from :mod:`run_kv_cache`
(``causal_block_entries`` / ``causal_selected_entries``) so the budget model
and the Exp 3 KV/FLOPs accounting can never disagree on causality.
"""

from __future__ import annotations

import math

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
    if not isinstance(sliding_window, int) or sliding_window < 0:
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
    if not isinstance(sliding_window, int) or sliding_window < 0:
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


_OP_FUNCS = ('softmax', 'kda', 'csa', 'hca')


def op_read_elements(op: str, T: int, spec: dict) -> dict:
    """Dispatch one operator's read budget at context length ``T``.

    ``spec`` is a geometry dict with one sub-dict per operator (see
    ``run_budget_frontier`` for the MQAR / paper-scale builders). Returns a
    uniform row: ``op``, ``T``, ``core_read_elements`` (KV actually
    attended / state read), ``indexer_read_elements`` (selection scan, CSA
    only), ``state_elements`` (KDA state size, else ``None``) and
    ``total_read_elements``.
    """
    if op not in _OP_FUNCS:
        raise ValueError(
            f"op_read_elements: op={op!r} must be one of {list(_OP_FUNCS)}.")
    if op not in spec:
        raise ValueError(
            f"op_read_elements: spec is missing the {op!r} geometry block "
            f"(keys present: {sorted(spec)}).")
    geo = spec[op]
    if op == 'softmax':
        core = softmax_read_elements(T, geo['n_kv_heads'], geo['d_k'],
                                     geo['d_v'])
        indexer = 0.0
        state = None
    elif op == 'kda':
        core = kda_read_elements(T, geo['hv'], geo['d_k'], geo['d_v'])
        state = kda_state_elements(geo['hv'], geo['d_k'], geo['d_v'])
        indexer = 0.0
    elif op == 'csa':
        parts = csa_read_elements(
            T, m=geo['m'], topk=geo['topk'], c=geo['c'], c_i=geo['c_i'],
            n_idx_heads=geo.get('n_idx_heads', 1),
            sliding_window=geo.get('sliding_window', 0))
        core = parts['core_read_elements']
        indexer = parts['indexer_read_elements']
        state = None
    else:  # hca
        parts = hca_read_elements(
            T, m2=geo['m2'], c=geo['c'],
            sliding_window=geo.get('sliding_window', 0))
        core = parts['core_read_elements']
        indexer = parts['indexer_read_elements']
        state = None
    return {
        'op': op,
        'T': int(T),
        'core_read_elements': core,
        'indexer_read_elements': indexer,
        'state_elements': state,
        'total_read_elements': core + indexer,
    }


def layout_read_elements(T: int, ratio, spec: dict) -> dict:
    """Read budget of a ``(n_kda, n_csa, n_hca)`` hybrid layout.

    The layout budget is the per-layer sum because layers execute
    sequentially and each query passes through every layer once.
    """
    if not (isinstance(ratio, (tuple, list)) and len(ratio) == 3):
        raise ValueError(
            f'layout_read_elements: ratio must be a 3-sequence, got {ratio!r}')
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0
               for n in ratio):
        raise ValueError(
            f'layout_read_elements: ratio components must be non-negative '
            f'ints, got {ratio!r}')
    n_kda, n_csa, n_hca = (int(n) for n in ratio)
    if n_kda + n_csa + n_hca == 0:
        raise ValueError('layout_read_elements: empty layout')
    per_layer = {op: op_read_elements(op, T, spec)['total_read_elements']
                 for op in _OP_FUNCS}
    kda = n_kda * per_layer['kda']
    csa = n_csa * per_layer['csa']
    hca = n_hca * per_layer['hca']
    return {
        'T': int(T),
        'n_kda': n_kda,
        'n_csa': n_csa,
        'n_hca': n_hca,
        'kda_read_elements': kda,
        'csa_read_elements': csa,
        'hca_read_elements': hca,
        'total_read_elements': kda + csa + hca,
    }


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
