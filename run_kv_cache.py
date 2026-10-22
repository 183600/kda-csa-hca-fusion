"""Experiment 3 — KV cache and FLOPs analysis.

The accounting covers every cache a real inference engine must retain, not
only the compressed KV entries:

  * the sliding-window KV (uncompressed, per-layer);
  * the lightning-indexer key cache (for CSA);
  * the compression weights / metadata (small but nonzero);
  * the attention sink (negligible, included for completeness).

We now report TWO accounting modes:

  * ``compressed_kv_only``  — the optimistic number: only the compressed
    KV entries (shared-KV design — one c-vector per compressed block serves
    as both key and value). At T=1,048,576 this is ~0.51% of the 1-layer
    GQA8 KV cache (K+V per head) and ~0.10% of the 5-layer baseline. This
    is what you get if you only count the compressed KV entries.
  * ``full_accounting``     — includes every auxiliary cache listed above plus
    incremental runtime state (partial-token accumulators and CSA's previous
    overlapped Cb/Zb block). This is the number a production inference engine
    would actually pay for the reference cache design.

We also make the baseline explicit: the GQA8 baseline is a *5-layer* unit
(5 full GQA8 attention layers) so that the comparison to the 3:1:1 hybrid
(5 sub-layers) is apples-to-apples. The original paper compared a 5-sub-layer
hybrid to a single GQA8 layer; we report both for transparency.

The numbers mirror the efficiency discussion in DeepSeek-V4 §2.3.4 and Kimi
Linear §3.2 / §7.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (append_run_manifest, sanitize_for_json,
                          write_json_atomic, capture_provenance,
                          make_run_manifest_entry)
from results_schema import envelope, write as write_results


# Reference GQA8, head_dim=128, BF16 baseline (as in DeepSeek-V4 §2.3.4).
GQA_H = 8
GQA_HEAD_DIM = 128
BF16_BYTES = 2

# Default architecture parameters (repo-chosen scale for the KV/FLOPs sweep).
DEFAULTS = dict(
    H=8, K=128, V=128, d=4096,
    csa_m=16, csa_c=128, csa_topk=512, csa_nIh=4, csa_cI=32, csa_sliding_window=2048,
    # Number of attention heads for CSA / HCA core. The default of 8 heads
    # matches the repo's H=8; the sink has ``nh`` elements per layer. The
    # value is explicit so the sink count is correct even if H is changed.
    csa_nh=8, csa_dc=128,
    hca_m2=64, hca_c=128, hca_sliding_window=2048,
    hca_nh=8, hca_dc=128,
    kda_hv=8, kda_k=128, kda_v=128,
    # KDA short-conv kernel size (depthwise Conv1d in KDAHybridLayer).
    kda_conv_ksize=3,
    # KDA chunk size for the chunk-parallel prefill kernel. The paper's Eq 13
    # (Kimi Linear §6.3) specifies a fixed chunk size C=64; the actual prefill
    # path (ops_fused.KDAHybridLayer / ops_kda.naive_chunk_kda) uses this value
    # for the inter-chunk A/u/w terms, so the FLOPs formula must match it.
    kda_chunk_size=64,
    # Hybrid layer-count ratio. Keeping these in DEFAULTS makes the documented
    # kwargs override path work (unknown-key validation below otherwise rejects
    # hybrid_n_kda / hybrid_n_csa / hybrid_n_hca before the hybrid branch sees
    # them).
    hybrid_n_kda=3, hybrid_n_csa=1, hybrid_n_hca=1,
    # Mamba-2 geometry for the 'mamba2' / 'hybrid_mamba2' accounting rows
    # (official-module defaults; d_inner = expand*d, nheads = d_inner/headdim).
    mamba_d_state=64, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64,
    mamba_ngroups=1, mamba_dt_rank='auto',
    # RWKV-7 geometry for the 'rwkv7' / 'hybrid_rwkv7' accounting rows.
    # head_dim=64 (upstream's "don't change" value; H = d/head_dim);
    # the LoRA widths follow the fla formula at this paper-scale d=4096:
    # max(32, round(coeff*sqrt(d)/32)*32) with coefficients 2.5 (decay),
    # 2.5 (a), 5 (gate) -> 160 / 160 / 320.
    rwkv_head_dim=64, rwkv_decay_lora=160, rwkv_a_lora=160,
    rwkv_gate_lora=320,
)


def causal_block_entries(T: int, m: int) -> int:
    """Count visible (query, compressed-block) pairs.

    Paper-exact strictly-preceding mask (Eq. 16): block ``s`` is visible
    to query ``t`` iff ``s < floor(t / m)``. A block therefore first
    becomes visible at query ``(s + 1) * m`` — the first token of the
    NEXT block — and a query never sees its own block through the sparse
    branch (the sliding-window branch serves that local context).
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'T must be a non-negative int, got {T!r}')
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise ValueError(f'm must be a positive int, got {m!r}')
    if T == 0:
        return 0
    n_full = T // m
    remainder = T % m
    # sum_{t=0}^{T-1} floor(t/m): every complete block b < n_full
    # contributes m queries that can see it (b*m .. (b+1)*m-1 all have
    # floor(t/m) > b... i.e. floor(t/m) >= b+1 for t >= (b+1)*m).
    # Direct derivation: sum floor(t/m) for t in [0, n_full*m + r):
    #   = m * (0 + 1 + ... + (n_full-1)) + n_full * r
    #   = m * n_full * (n_full - 1) // 2 + n_full * r.
    return m * n_full * (n_full - 1) // 2 + n_full * remainder


def geometric_capacity(n_rows: int) -> int:
    """Return the power-of-two storage capacity used by decode caches."""
    if (not isinstance(n_rows, int) or isinstance(n_rows, bool)
            or n_rows < 0):
        raise ValueError(
            f'n_rows must be a non-negative int, got {n_rows!r}')
    if n_rows == 0:
        return 0
    return 1 << (n_rows - 1).bit_length()


def causal_selected_entries(T: int, m: int, topk: int) -> int:
    """Count selected block slots after a top-k cap under block causality.

    Uses the paper-exact strictly-preceding mask (Eq. 16): query ``t``
    selects from ``floor(t / m)`` visible blocks, capped at ``topk``.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'T must be a non-negative int, got {T!r}')
    if not isinstance(m, int) or isinstance(m, bool) or m < 1:
        raise ValueError(f'm must be a positive int, got {m!r}')
    if (not isinstance(topk, int) or isinstance(topk, bool)
            or topk < 0):
        raise ValueError(f'topk must be a non-negative int, got {topk!r}')
    if T == 0 or topk == 0:
        return 0
    n_full = T // m
    remainder = T % m
    if n_full == 0:
        return 0
    # sum_{t=0}^{T-1} min(topk, floor(t/m)):
    #   * queries in block b (b = 0..n_full-1, m queries each) see
    #     min(topk, b) blocks (floor(t/m) = b for those queries).
    #   * the trailing partial block's r queries see min(topk, n_full).
    if topk >= n_full - 1:
        sum_before_last = n_full * (n_full - 1) // 2
    else:
        sum_before_last = (
            topk * (topk + 1) // 2
            + topk * (n_full - 1 - topk)
        )
    return m * sum_before_last + remainder * min(topk, n_full)


def kv_cache_elements(op: str, T: int, *, mode: str = 'compressed_kv_only', **kw):
    """Number of KV-cache *elements* retained for decoding token T+1.

    Parameters
    ----------
    op : str
        One of 'softmax_gqa', 'kda', 'csa', 'hca', 'hybrid_kch'.
    T : int
        Number of tokens already processed.
    mode : str
        'compressed_kv_only'  — only the compressed KV entries (original paper's
                                 optimistic accounting).
        'full_accounting'     — compressed rows actually materialized by the
                                 incremental cache + fixed sliding-window buffer
                                 capacity + indexer key cache + partial-token
                                 accumulators + CSA overlap state + sink.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(
            f'kv_cache_elements: T must be a non-negative int, got {T!r}.')
    p = {**DEFAULTS, **kw}
    _valid_keys = set(DEFAULTS.keys())
    _unknown = set(kw.keys()) - _valid_keys
    if _unknown:
        raise ValueError(
            f"kv_cache_elements: unknown keyword argument(s) "
            f"{sorted(_unknown)}. Valid keys: {sorted(_valid_keys)}.")
    _valid_ops = {'softmax_gqa', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca',
                  'hybrid_kch', 'hybrid_mamba2', 'hybrid_rwkv7'}
    if op not in _valid_ops:
        raise ValueError(
            f"kv_cache_elements: op={op!r} must be one of "
            f"{sorted(_valid_ops)}.")
    _valid_modes = {'compressed_kv_only', 'full_accounting'}
    if mode not in _valid_modes:
        raise ValueError(
            f"kv_cache_elements: mode={mode!r} must be one of "
            f"{sorted(_valid_modes)}.")
    H, K, V = p['H'], p['K'], p['V']
    csa_m, csa_c = p['csa_m'], p['csa_c']
    hca_m2, hca_c = p['hca_m2'], p['hca_c']
    kda_hv, kda_k, kda_v = p['kda_hv'], p['kda_k'], p['kda_v']
    csa_sw = p['csa_sliding_window']
    hca_sw = p['hca_sliding_window']
    csa_cI = p['csa_cI']  # csa_nIh not needed for KV-cache accounting (only for FLOPs)

    if op == 'softmax_gqa':
        # GQA: 8 KV heads, each with K=V=128. Cache is T * H_kv * (K + V).
        # We count elements (not bytes); K and V are both retained.
        if mode == 'full_accounting':
            # Full-accounting mode counts the capacity the reference decode
            # cache actually allocates: run_decoding.SoftmaxAttnDecoding
            # fresh-allocates ``max(2*T, 64)`` token slots on prefill and
            # grows geometrically from there, so after T tokens the engine
            # has paid ``max(2*T, 64)`` slots, not just the T valid ones.
            # Counting only the valid prefix would understate the GQA
            # baseline and inflate the compressed ops' ``kv_ratio_vs_gqa_*``
            # in this mode.
            return max(2 * T, 64) * H * (K + V)
        return T * H * (K + V)

    if op == 'kda':
        # KDA keeps a fixed recurrent state [HV, K, V]; no per-token KV cache.
        # The recurrent state is the dominant cost and is always counted.
        recurrent_state = kda_hv * kda_k * kda_v
        if mode == 'full_accounting':
            # KDA layers also carry a short-conv state of O(d) per layer
            # (the d-element convolutional lookahead buffer used to feed the
            # recurrent update). The actual ``nn.Conv1d(kernel_size=3, groups=d)``
            # in ``ops_fused.py::KDAHybridLayer`` needs ``(kernel_size - 1) * d``
            # = 2*d elements of left-padding buffer for streaming — not just d.
            # This is negligible next to the recurrent state but a production
            # engine must retain it.
            _conv_ksize = p.get('kda_conv_ksize', 3)
            short_conv_state = (_conv_ksize - 1) * p['d']
            return recurrent_state + short_conv_state
        # compressed_kv_only: just the recurrent state.
        return recurrent_state

    if op == 'mamba2':
        # Mamba-2 keeps a FIXED per-layer state — sequence-INDEPENDENT,
        # the same O(1) property as KDA:
        #   * SSM state: [nheads, headdim, d_state] (compute precision);
        #   * conv state: [d_inner, d_conv - 1] (input dtype).
        # ``compressed_kv_only`` reports just the SSM state (the analogue
        # of KDA's recurrent state); ``full_accounting`` adds the conv
        # state. NOTE: within a hybrid STACK the CSA/HCA compressed
        # entries still grow with T — the hybrid rows below carry that.
        d_inner = p['mamba_expand'] * p['d']
        nheads = d_inner // p['mamba_headdim'] if p['mamba_headdim'] > 0 else 1
        ssm_state = nheads * p['mamba_headdim'] * p['mamba_d_state']
        if mode == 'full_accounting':
            conv_state = d_inner * (p['mamba_d_conv'] - 1)
            return ssm_state + conv_state
        return ssm_state

    if op == 'rwkv7':
        # RWKV-7 keeps a FIXED per-layer state — sequence-INDEPENDENT,
        # the same O(1) property as KDA / Mamba-2:
        #   * matrix state: [H, K, V] per layer in compute precision
        #     (K = V = rwkv_head_dim; H = d // rwkv_head_dim), i.e.
        #     ``d * rwkv_head_dim`` elements — the M_matrix of the
        #     cache-budget model;
        #   * shift state: [1, d] (the previous token's input, for the
        #     token shift), input dtype.
        # ``compressed_kv_only`` reports just the matrix state (the
        # analogue of KDA's recurrent state / Mamba-2's SSM state);
        # ``full_accounting`` adds the shift state. NOTE: within a hybrid
        # STACK the CSA/HCA compressed entries still grow with T — the
        # hybrid rows below carry that, so only the RWKV-7 BRANCH of the
        # hybrid cache is fixed-size.
        nheads = p['d'] // p['rwkv_head_dim'] if p['rwkv_head_dim'] > 0 else 1
        matrix_state = nheads * p['rwkv_head_dim'] * p['rwkv_head_dim']
        if mode == 'full_accounting':
            shift_state = p['d']
            return matrix_state + shift_state
        return matrix_state

    if op == 'csa':
        n_blocks = max(1, (T + csa_m - 1) // csa_m)
        # ``compressed_kv_only`` keeps the historical allocated-capacity / padded
        # prefill semantics: reserve enough compressed slots for the trailing
        # partial block as if it were padded out to length m.
        compressed = n_blocks * csa_c
        if mode == 'full_accounting':
            # Full-accounting mode tracks the *incremental decoding cache* state
            # after T tokens have actually been appended. Compressed rows are
            # materialized after full blocks, and the runtime storage grows
            # geometrically; count allocated capacity rather than only the
            # valid prefix. The trailing partial block is represented by the
            # partial accumulator below.
            n_completed = T // csa_m
            compressed_capacity = geometric_capacity(n_completed)
            compressed_runtime = compressed_capacity * csa_c
            # Sliding-window branch: the reference decoding cache pre-allocates
            # a fixed ring buffer of length csa_sw (when enabled), so memory
            # accounting should count capacity, not just the number of currently
            # valid tokens. This is conservative and matches the actual tensor
            # allocated by _SlidingWindowRingBuffer.
            sw = csa_sw * csa_c
            # Indexer key cache: one compressed indexer key per completed block.
            indexer = compressed_capacity * csa_cI
            # Incremental decode also retains a partial-token accumulator until
            # a full block is available. CSA stores six per-token projections:
            # Ca/Cb/Za/Zb (4*c) plus indexer K/Z (2*c_I).
            partial_tokens = T % csa_m
            partial = partial_tokens * (4 * csa_c + 2 * csa_cI)
            # Overlapped CSA compression needs the previous block's Cb/Zb as
            # the b-branch partner for the NEXT block. Once at least one block
            # has completed, the decoding cache retains two [m, c] tensors.
            overlap_prev = (2 * csa_m * csa_c) if n_completed >= 1 else 0
            # Compression metadata: the per-block softmax weights Z are recomputed
            # from the input hidden state during decoding, so they are NOT cached.
            # Sink: nh elements (negligible, included for completeness —
            # documented here even though the value is tiny).
            sink = p.get('csa_nh', H)
            return compressed_runtime + sw + indexer + partial + overlap_prev + sink
        return compressed

    if op == 'hca':
        n_blocks = max(1, (T + hca_m2 - 1) // hca_m2)
        compressed = n_blocks * hca_c
        if mode == 'full_accounting':
            # Incremental HCA cache materializes storage only for completed
            # heavy-compression blocks and grows it geometrically; the trailing
            # partial block is represented by the partial accumulator below.
            n_completed = T // hca_m2
            compressed_capacity = geometric_capacity(n_completed)
            compressed_runtime = compressed_capacity * hca_c
            # HCA uses the same fixed-capacity sliding-window ring buffer as
            # CSA, so count the allocated window capacity.
            sw = hca_sw * hca_c
            # HCA's incremental cache retains a partial accumulator until m2
            # tokens are available: C and Z projections for each partial token.
            partial_tokens = T % hca_m2
            partial = partial_tokens * (2 * hca_c)
            sink = p.get('hca_nh', H)
            return compressed_runtime + sw + partial + sink
        return compressed

    if op == 'hybrid_kch':
        # 3 KDA + 1 CSA + 1 HCA per 5-layer unit (default 3:1:1).
        # Allow override via kwargs so non-default ratios are accounted for
        # correctly.
        n_kda = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        kda_part = n_kda * kv_cache_elements('kda', T, mode=mode, **p)
        csa_part = n_csa * kv_cache_elements('csa', T, mode=mode, **p)
        hca_part = n_hca * kv_cache_elements('hca', T, mode=mode, **p)
        return kda_part + csa_part + hca_part

    if op == 'hybrid_mamba2':
        # The Mamba-2-route hybrid: the SAME ratio slots as hybrid_kch but
        # the memory layers are Mamba-2 mixers. The CSA/HCA parts are
        # IDENTICAL to hybrid_kch (same operators, same geometry), so the
        # row-pair difference isolates the memory operator's state cost.
        n_mem = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        mem_part = n_mem * kv_cache_elements('mamba2', T, mode=mode, **p)
        csa_part = n_csa * kv_cache_elements('csa', T, mode=mode, **p)
        hca_part = n_hca * kv_cache_elements('hca', T, mode=mode, **p)
        return mem_part + csa_part + hca_part

    if op == 'hybrid_rwkv7':
        # The RWKV-7-route hybrid: the SAME ratio slots as hybrid_kch but
        # the memory layers are RWKV-7 mixers (the KDA-replacement
        # ablation's cache row). The CSA/HCA parts are IDENTICAL to
        # hybrid_kch, so the row-pair difference isolates the memory
        # operator's state cost.
        n_mem = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        mem_part = n_mem * kv_cache_elements('rwkv7', T, mode=mode, **p)
        csa_part = n_csa * kv_cache_elements('csa', T, mode=mode, **p)
        hca_part = n_hca * kv_cache_elements('hca', T, mode=mode, **p)
        return mem_part + csa_part + hca_part

    raise ValueError(op)


def kv_cache_breakdown(op: str, T: int, mode: str = 'full_accounting', **kw):
    """Component-wise state accounting (elements, per single layer).

    Splits each operator's retained state into the framework's canonical
    buckets — recurrent state, convolution state, compressed cache,
    sliding-window cache, and miscellaneous runtime state — so the
    quality-vs-cache figures and the budget-matching design can reason
    about WHERE the bytes go, not just the total. Values are ELEMENT
    counts (multiply by the dtype size for bytes). Buckets the operator
    does not retain are omitted from the returned dict.

    ``mode`` mirrors :func:`kv_cache_elements`: the reported components
    are the ones retained under that accounting mode, so they sum to
    ``kv_cache_elements(op, T, mode=mode, **kw)`` — except for the
    explicitly-labelled ``compressed_kv_only_total`` reference annotation,
    which is informational and never part of the total.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(
            f'kv_cache_breakdown: T must be a non-negative int, got {T!r}.')
    _valid_keys = set(DEFAULTS.keys())
    _unknown = set(kw.keys()) - _valid_keys
    if _unknown:
        raise ValueError(
            f"kv_cache_breakdown: unknown keyword argument(s) "
            f"{sorted(_unknown)}. Valid keys: {sorted(_valid_keys)}.")
    if mode not in ('compressed_kv_only', 'full_accounting'):
        raise ValueError(
            f"kv_cache_breakdown: mode={mode!r} must be one of "
            f"['compressed_kv_only', 'full_accounting'].")
    p = {**DEFAULTS, **kw}
    full = mode == 'full_accounting'
    if op == 'softmax_gqa':
        if full:
            # Capacity the reference decode cache allocates after T tokens
            # (see kv_cache_elements): max(2*T, 64) slots, not just T.
            return {'kv_cache': max(2 * T, 64) * p['H'] * (p['K'] + p['V'])}
        return {'kv_cache': T * p['H'] * (p['K'] + p['V'])}
    if op == 'kda':
        out = {'recurrent_state': p['kda_hv'] * p['kda_k'] * p['kda_v']}
        if full:
            _conv_ksize = p.get('kda_conv_ksize', 3)
            out['conv_state'] = p['d'] * (_conv_ksize - 1)
        return out
    if op == 'mamba2':
        d_inner = p['mamba_expand'] * p['d']
        nheads = d_inner // p['mamba_headdim'] if p['mamba_headdim'] > 0 else 1
        out = {'recurrent_state':
               nheads * p['mamba_headdim'] * p['mamba_d_state']}
        if full:
            out['conv_state'] = d_inner * (p['mamba_d_conv'] - 1)
        return out
    if op == 'rwkv7':
        nheads = p['d'] // p['rwkv_head_dim'] if p['rwkv_head_dim'] > 0 else 1
        out = {'recurrent_state':
               nheads * p['rwkv_head_dim'] * p['rwkv_head_dim']}
        if full:
            out['shift_state'] = p['d']
        return out
    if op == 'csa':
        n_blocks = max(1, (T + p['csa_m'] - 1) // p['csa_m'])
        # Optimistic-mode total kept for reference in the row (never part of
        # the summed components).
        ref = n_blocks * p['csa_c']
        if not full:
            return {'compressed_cache': ref,
                    'compressed_kv_only_total': ref}
        n_completed = T // p['csa_m']
        capacity = geometric_capacity(n_completed)
        partial_tokens = T % p['csa_m']
        return {
            'compressed_cache': capacity * p['csa_c'],
            'window_cache': p['csa_sliding_window'] * p['csa_c'],
            'indexer_cache': capacity * p['csa_cI'],
            'partial_accumulator': partial_tokens * (4 * p['csa_c']
                                                     + 2 * p['csa_cI']),
            'overlap_state': (2 * p['csa_m'] * p['csa_c'])
            if n_completed >= 1 else 0,
            'sink': p.get('csa_nh', p['H']),
            'compressed_kv_only_total': ref,
        }
    if op == 'hca':
        n_blocks = max(1, (T + p['hca_m2'] - 1) // p['hca_m2'])
        ref = n_blocks * p['hca_c']
        if not full:
            return {'compressed_cache': ref,
                    'compressed_kv_only_total': ref}
        n_completed = T // p['hca_m2']
        capacity = geometric_capacity(n_completed)
        partial_tokens = T % p['hca_m2']
        return {
            'compressed_cache': capacity * p['hca_c'],
            'window_cache': p['hca_sliding_window'] * p['hca_c'],
            'partial_accumulator': partial_tokens * (2 * p['hca_c']),
            'sink': p.get('hca_nh', p['H']),
            'compressed_kv_only_total': ref,
        }
    if op in ('hybrid_kch', 'hybrid_mamba2', 'hybrid_rwkv7'):
        mem_op = {'hybrid_kch': 'kda', 'hybrid_mamba2': 'mamba2',
                  'hybrid_rwkv7': 'rwkv7'}[op]
        n_mem = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        mem = kv_cache_breakdown(mem_op, T, mode=mode, **p)
        csa = kv_cache_breakdown('csa', T, mode=mode, **p)
        hca = kv_cache_breakdown('hca', T, mode=mode, **p)
        out = {}
        for bucket in ('recurrent_state', 'conv_state', 'shift_state',
                       'compressed_cache', 'window_cache', 'indexer_cache',
                       'partial_accumulator', 'overlap_state', 'sink'):
            total = 0
            for count, comp in ((n_mem, mem), (n_csa, csa), (n_hca, hca)):
                v = comp.get(bucket)
                if v is not None:
                    total += count * v
            if total:
                out[bucket] = total
        return out
    raise ValueError(
        f"kv_cache_breakdown: op={op!r} must be one of "
        f"['softmax_gqa', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca', "
        f"'hybrid_kch', 'hybrid_mamba2', 'hybrid_rwkv7'].")


def prefill_flops(op: str, T: int, **kw):
    """Approximate prefill FLOPs (2 * MACs) for a single attention layer.

    Accounting conventions
    ----------------------
    Every attention op has TWO matmuls in its core: ``QK^T`` (over the key
    dim) and ``softmax(P) @ V`` (over the value dim). Both are counted for
    every operator so the ``flops_ratio_vs_gqa_*`` columns are comparable
    across operators.

    For KDA, the recurrence (see ``ops_kda.py::naive_recurrent_kda``) has
    roughly 4 ``HV*K*V``-sized matvec operations per step:

      1. ``S * g_i.exp()``                — elementwise (no FLOPs)
      2. ``(k_i * S).sum(-2)``            — HV*V dots of length K  -> HV*V*K MACs
      3. ``b_i * k_i ⊗ (v_i - ...)``     — outer product HV*K*V MACs
      4. ``q_i^T S``                       — HV*V dots of length K  -> HV*V*K MACs

    i.e. ~3 * HV*K*V MACs per step (the dominant terms), or
    ~6 * T * HV*K*V FLOPs total. The paper's §6.3 Eq 13 counts the
    chunk-parallel kernel (fixed chunk size ``C=64``) as
    ``6T*dh^2 + 3T*C*dh + T*C^2`` per head; the prefill implementation
    (``ops_fused.KDAHybridLayer``, ``kda_chunk_size=64`` default) actually
    computes those inter-chunk terms (the block transfer matrix ``A`` and the
    ``u``/``w`` inter-chunk outputs), so we add
    ``HV * (3T*C*K + T*C^2)`` to the recurrent dominant term. We also
    include the input projection FLOPs (q/k/v/g/beta plus the grouped
    output projection) for parity with CSA/HCA and the softmax baseline.

    For CSA, the ``compress`` term counts the SIX input projections the
    implementation (``ops_csa.py::naive_csa``) performs:
    ``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx``.
    We also count each operator's grouped output projection (KDA ``o_proj``,
    CSA/HCA ``o_proj``), so Exp3's "single-layer FLOPs" boundary matches
    Exp2's end-to-end standalone CSA/HCA boundary.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(
            f'prefill_flops: T must be a non-negative int, got {T!r}.')
    _valid_keys = set(DEFAULTS.keys())
    _unknown = set(kw.keys()) - _valid_keys
    if _unknown:
        raise ValueError(
            f"prefill_flops: unknown keyword argument(s) "
            f"{sorted(_unknown)}. Valid keys: {sorted(_valid_keys)}.")
    p = {**DEFAULTS, **kw}
    H, K, V, d = p['H'], p['K'], p['V'], p['d']
    csa_m, csa_topk, csa_c = p['csa_m'], p['csa_topk'], p['csa_c']
    hca_m2, hca_c = p['hca_m2'], p['hca_c']
    kda_hv, kda_k, kda_v = p['kda_hv'], p['kda_k'], p['kda_v']

    if op == 'softmax_gqa':
        # CAUSAL attention (matching the SoftmaxAttn baseline in
        # run_quality.py and run_decoding.py::SoftmaxAttnDecoding which
        # apply a strictly-upper-triangular mask). Each query t attends
        # to keys [0, t], i.e. (t+1) keys. Total attention entries over
        # all queries = T*(T+1)/2 (the upper-triangular-inclusive count).
        # GQA8: H=8 KV heads shared by a full-rank query side. With
        # head_dim=128 and d=4096, the query side has n_q = d // head_dim = 32
        # heads; the KV side has n_kv = 8. Query-side terms (core, q_proj,
        # o_proj) scale with n_q; KV-side terms (k_proj, v_proj) with n_kv.
        n_q = d // GQA_HEAD_DIM
        n_kv = H
        causal_entries = T * (T + 1) // 2
        core = 2 * causal_entries * n_q * (K + V)
        # Input/output projections — counted for PARITY with KDA / CSA / HCA,
        # whose ``proj`` / ``compress`` / ``query_proj`` terms include them.
        #   q_proj : d -> n_q*K  -> T*d*n_q*K MACs
        #   k_proj : d -> n_kv*K -> T*d*n_kv*K MACs
        #   v_proj : d -> n_kv*V -> T*d*n_kv*V MACs
        #   o_proj : n_q*V -> d  -> T*n_q*V*d MACs
        proj = 2 * T * d * (n_q * K + n_kv * K + n_kv * V)
        out_proj = 2 * T * n_q * V * d
        return core + proj + out_proj
    if op == 'kda':
        # Input projections — count the ACTUAL matmul shapes from
        # ops_fused.py::KDAHybridLayer, not an approximation.
        #   q_proj  : d -> H*K    -> T*d*H*kda_k MACs
        #   k_proj  : d -> H*K    -> T*d*H*kda_k MACs
        #   v_proj  : d -> HV*V   -> T*d*kda_hv*kda_v MACs  (V==kda_v)
        #   g_down  : d -> K      -> T*d*kda_k MACs
        #   g_up    : K -> HV*K   -> T*kda_k*kda_hv*kda_k MACs
        #   beta    : d -> HV     -> T*d*kda_hv MACs
        proj = 2 * T * (
              d * (2 * H * kda_k + kda_hv * kda_v + kda_k + kda_hv)
            + kda_k * kda_hv * kda_k   # g_up: inner dim is kda_k, not d
        )
        # Causal depthwise short-conv (KDAHybridLayer.short_conv):
        # Conv1d(in=d, out=d, kernel_size=ksize, groups=d, bias=True).
        # Depthwise: each output channel = 1 * ksize MACs per timestep,
        # for a total of ``T * d * ksize`` MACs -> 2*T*d*ksize FLOPs.
        # The bias adds T*d FLOPs, negligible vs 6*T*d for ksize=3.
        ksize = p.get('kda_conv_ksize', 3)
        short_conv = 2 * T * d * ksize
        # Recurrence: ~3 HV*K*V MACs per step (see docstring).
        recurrent = 2 * 3 * T * kda_hv * kda_k * kda_v
        # Chunk-parallel inter-chunk terms (paper §6.3 Eq 13):
        # ``3T*C*dh + T*C^2`` per head, C = chunk size, dh = K = V. The
        # prefill path (ops_fused.KDAHybridLayer, default
        # ``kda_chunk_size=64``) computes these on top of the recurrent
        # term. The terms are already FLOPs (2*MACs), matching the paper.
        c = p['kda_chunk_size']
        chunk_inter = 3 * T * kda_hv * c * kda_k
        chunk_a = T * kda_hv * c * c
        # Grouped output projection: HV*V -> d (matches KDAHybridLayer.o_proj,
        # run_quality.KDAAttn.o, and run_decoding.KDAAttnDecoding.o).
        out_proj = 2 * T * kda_hv * kda_v * d
        return proj + short_conv + recurrent + chunk_inter + chunk_a + out_proj
    if op == 'mamba2':
        # Full mixer FLOPs, matching ops_mamba.Mamba2Mixer's actual matmul
        # shapes (headdim/expand/d_state from p; dt_rank auto = ceil(d/16)):
        #   in_proj : d -> 2*d_inner                     2*T*d*2*d_inner
        #   conv    : depthwise d_inner x d_conv         2*T*d_inner*d_conv
        #   x_proj  : d_inner -> dt_rank + 2*G*N         2*T*d_inner*(dt_rank+2GN)
        #   dt_proj : dt_rank -> nheads                   2*T*dt_rank*nheads
        #   scan    : ~3 MACs of headdim*d_state per head per step
        #             (decay mul + x(x)B outer + C.S readout)
        #             -> 2*3*T*nheads*headdim*d_state
        #   out_proj: d_inner -> d                        2*T*d_inner*d
        d_inner = p['mamba_expand'] * d
        nheads = d_inner // p['mamba_headdim'] if p['mamba_headdim'] > 0 else 1
        headdim = p['mamba_headdim']
        d_state = p['mamba_d_state']
        ngroups = p.get('mamba_ngroups', 1)
        dt_rank_cfg = p.get('mamba_dt_rank', 'auto')
        if isinstance(dt_rank_cfg, str):
            dt_rank = max(1, (d + 15) // 16)
        else:
            dt_rank = int(dt_rank_cfg)
        in_proj = 2 * T * d * 2 * d_inner
        conv = 2 * T * d_inner * p['mamba_d_conv']
        x_proj = 2 * T * d_inner * (dt_rank + 2 * ngroups * d_state)
        dt_proj = 2 * T * dt_rank * nheads
        scan = 2 * 3 * T * nheads * headdim * d_state
        out_proj = 2 * T * d_inner * d
        return in_proj + conv + x_proj + dt_proj + scan + out_proj
    if op == 'rwkv7':
        # Full mixer FLOPs, matching ops_rwkv.Rwkv7Mixer's actual matmul
        # shapes (K = V = rwkv_head_dim; H = d // rwkv_head_dim; r_w/r_a/
        # r_g = the three LoRA widths):
        #   r/k/v/o_proj : d -> d (four of them)      4 * 2*T*d*d
        #   w_lora       : d -> r_w -> d               2*T*d*r_w*2
        #   a_lora       : d -> r_a -> d               2*T*d*r_a*2
        #   g_lora       : d -> r_g -> d               2*T*d*r_g*2
        #   token shift / k_k / k_a / r_k / l2norm:   O(T*d) — negligible,
        #                                               counted as 5*2*T*d
        #   recurrence   : ~6 K×V-sized passes per head per step (decay
        #                  multiply, kk^T·S contract dot, outer update,
        #                  k⊗v outer, elementwise add, S^T·r readout)
        #                  -> 2*6*T*H*K*V
        #   shortcut+gate: (r·k·r_k).sum + ⊗v + ⊙g    ~2*T*(H*(K+V) + 2*d)
        #                  (groupnorm ~3*T*H*V folded in)
        nheads = p['d'] // p['rwkv_head_dim'] if p['rwkv_head_dim'] > 0 else 1
        kk_dim = p['rwkv_head_dim']
        proj = 4 * 2 * T * d * d
        loras = 2 * 2 * T * d * (p['rwkv_decay_lora'] + p['rwkv_a_lora']
                                 + p['rwkv_gate_lora'])
        misc = 5 * 2 * T * d
        scan = 2 * 6 * T * nheads * kk_dim * kk_dim
        shortcut = 2 * T * (nheads * (kk_dim + kk_dim) + 2 * d + 3 * nheads * kk_dim)
        return proj + loras + misc + scan + shortcut
    if op == 'csa':
        # KV-side compression: SIX input projections (W_aKV, W_bKV, W_aZ,
        # W_bZ, W_KV_idx, W_Z_idx). The first four are T*d*c; the last
        # two are T*d*c_I.
        compress = 2 * T * d * (4 * csa_c + 2 * p['csa_cI'])
        # Query-side projections (W_DQ, W_UQ, W_IUQ, W_w):
        #   W_DQ  : d -> dc       -> T*d*csa_dc MACs
        #   W_UQ  : dc -> c*nh    -> T*csa_dc*csa_c*H MACs
        #   W_IUQ : dc -> c_I*nIh -> T*csa_dc*csa_cI*csa_nIh MACs
        #   W_w   : d -> nIh      -> T*d*csa_nIh MACs
        csa_dc = p.get('csa_dc', 128)
        csa_nh = p.get('csa_nh', H)
        query_proj = 2 * T * (
              d * csa_dc
            + csa_dc * csa_c * csa_nh
            + csa_dc * p['csa_cI'] * p['csa_nIh']
            + d * p['csa_nIh']
        )
        # Indexer: per-head similarities T * n_blocks * c_I * nIh, then
        # weighted sum across heads T * n_blocks * nIh. The lightning
        # indexer applies the paper-exact strictly-preceding causal mask
        # before top-k (block ``s`` is visible iff ``s < floor(t / m)``),
        # so a block first becomes visible to the NEXT block's queries.
        # Keep the exact count in one helper so the experiment accounting
        # matches ops_csa._causal_block_mask and the decoding cache.
        causal_entries = causal_block_entries(T, csa_m)
        indexer = 2 * causal_entries * p['csa_cI'] * p['csa_nIh'] \
                  + 2 * causal_entries * p['csa_nIh']
        # Core sparse attention: QK^T (c term) + softmax·V (c term).
        # ``csa_lightning_indexer`` clamps topk to ``min(topk, n_blocks)``
        # AND masks non-causal blocks to -inf before top-k, so the EFFECTIVE
        # per-query topk is ``min(csa_topk, t // csa_m)``. The AVERAGE
        # effective topk over all queries is therefore
        # ``sum_t min(csa_topk, t // csa_m) / T`` under the paper-exact
        # strictly-preceding mask. The helper counts the exact selected
        # slots.
        total_sel = causal_selected_entries(T, csa_m, csa_topk)
        effective_topk = total_sel / T if T > 0 else 0
        core = 2 * T * effective_topk * csa_c * csa_nh * 2
        # Sliding window: causal window — query t attends to positions
        # [max(0, t-w+1), t], i.e. min(t+1, w) keys (NOT w keys for every
        # query). Total causal-window entries = T*w - w*(w-1)/2 when
        # T >= w, else T*(T+1)/2.
        sw_w = p['csa_sliding_window']
        eff_sw = min(T, sw_w)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        # The SW branch uses ``csa_nh`` heads (the ``q`` tensor is shared
        # with the sparse branch), NOT ``H``.
        sw = 2 * sw_entries * csa_c * csa_nh * 2
        # Grouped output projection: (csa_nh * csa_c) -> d.
        # This is part of the standalone CSA layer boundary in run_benchmark.py
        # and CSAHybridLayer.o_proj, so Exp3 must count it too.
        out_proj = 2 * T * csa_nh * csa_c * d
        return compress + query_proj + indexer + core + sw + out_proj
    if op == 'hca':
        # KV-side compression: TWO input projections (W_KV, W_Z), each T*d*c.
        compress = 2 * T * d * hca_c * 2
        # Query-side projections (W_DQ, W_UQ):
        #   W_DQ : d -> dc    -> T*d*hca_dc MACs
        #   W_UQ : dc -> c*nh -> T*hca_dc*hca_c*H MACs
        hca_dc = p.get('hca_dc', 128)
        hca_nh = p.get('hca_nh', H)
        query_proj = 2 * T * (d * hca_dc + hca_dc * hca_c * hca_nh)
        # Core dense attention over the compressed blocks. The dense branch
        # uses the same strictly-preceding block mask as ops_csa: heavy-
        # compression block ``s`` is visible iff ``s < floor(t / m2)``
        # (its source window must be complete AND the query must lie in a
        # later block). Keep the exact (query, block) count shared with
        # the test oracle.
        causal_entries = causal_block_entries(T, hca_m2)
        # Head count: the HCA core attention uses ``hca_nh`` heads (NOT ``H``).
        core = 2 * causal_entries * hca_c * hca_nh * 2
        # Sliding window: causal window.
        # The SW branch uses ``hca_nh`` heads.
        sw_w = p['hca_sliding_window']
        eff_sw = min(T, sw_w)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * hca_c * hca_nh * 2
        # Grouped output projection: (hca_nh * hca_c) -> d.
        # Mirrors HCAHybridLayer.o_proj and the standalone HCA benchmark.
        out_proj = 2 * T * hca_nh * hca_c * d
        return compress + query_proj + core + sw + out_proj
    if op == 'hybrid_kch':
        # Mirror the configurable ratio in kv_cache_elements.
        n_kda = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        return (n_kda * prefill_flops('kda', T, **p)
                + n_csa * prefill_flops('csa', T, **p)
                + n_hca * prefill_flops('hca', T, **p))
    if op == 'hybrid_mamba2':
        # Mamba-2-route hybrid at the same ratio slots; the CSA/HCA FLOPs
        # are identical to hybrid_kch, so the row pair isolates the memory
        # operator's compute cost.
        n_mem = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        return (n_mem * prefill_flops('mamba2', T, **p)
                + n_csa * prefill_flops('csa', T, **p)
                + n_hca * prefill_flops('hca', T, **p))
    if op == 'hybrid_rwkv7':
        # RWKV-7-route hybrid at the same ratio slots (the KDA-replacement
        # ablation's compute row); CSA/HCA FLOPs identical to hybrid_kch.
        n_mem = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        return (n_mem * prefill_flops('rwkv7', T, **p)
                + n_csa * prefill_flops('csa', T, **p)
                + n_hca * prefill_flops('hca', T, **p))
    raise ValueError(op)


def main():
    print('=' * 70)
    print('Experiment 3: KV Cache & FLOPs Analysis (improved accounting)')
    print('=' * 70)
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                   131072, 262144, 524288, 1048576]

    ops = ['softmax_gqa', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca',
           'hybrid_kch', 'hybrid_mamba2', 'hybrid_rwkv7']
    rows = []
    for T in seq_lengths:
        flops_base_1l = prefill_flops('softmax_gqa', T)
        flops_base_5l = 5 * flops_base_1l

        for op in ops:
            for mode in ['compressed_kv_only', 'full_accounting']:
                kv = kv_cache_elements(op, T, mode=mode)
                # The GQA baseline is itself mode-aware: in full_accounting
                # mode it counts the geometric capacity the reference decode
                # cache allocates (see kv_cache_elements), so the ratios
                # below stay apples-to-apples within each accounting mode.
                baseline_1l = kv_cache_elements('softmax_gqa', T, mode=mode)
                baseline_5l = 5 * baseline_1l
                fl = prefill_flops(op, T)
                row = {
                    'T': T,
                    'op': op,
                    'accounting_mode': mode,
                    'accounting_semantics': (
                        'full_gqa_kv_cache'
                        if op == 'softmax_gqa'
                        else 'compressed_kv_only'
                        if mode == 'compressed_kv_only'
                        else 'recurrent_state_plus_short_conv'
                        if op == 'kda'
                        else 'fixed_ssm_state_plus_conv_state'
                        if op == 'mamba2'
                        else 'fixed_matrix_state_plus_shift_state'
                        if op == 'rwkv7'
                        else 'compressed_rows_plus_runtime_decode_state'
                        if op in ('csa', 'hca')
                        else 'hybrid_full_runtime_decode_state'
                    ),
                    'kv_elements': kv,
                    # Component-wise split (recurrent state vs conv state vs
                    # compressed/window caches) for the quality-vs-cache
                    # figures and the budget-matching design. Mode-aware, so
                    # the components reconcile with ``kv_elements`` above in
                    # BOTH accounting modes.
                    'state_components': kv_cache_breakdown(op, T, mode=mode),
                    # Explicit byte count under the BF16 accounting convention used
                    # throughout the cache analysis. README has long documented a
                    # ``kv_bytes`` field; writing it here keeps the JSON schema
                    # self-contained so downstream reports do not have to remember
                    # to multiply elements by BF16_BYTES themselves.
                    'kv_bytes': kv * BF16_BYTES,
                    # Ratios against the 1-layer baseline (original paper's convention).
                    'kv_ratio_vs_gqa_1l': kv / baseline_1l,
                    'flops_ratio_vs_gqa_1l': fl / flops_base_1l,
                    # Ratios against the 5-layer baseline (apples-to-apples).
                    'kv_ratio_vs_gqa_5l': kv / baseline_5l,
                    'flops_ratio_vs_gqa_5l': fl / flops_base_5l,
                    'prefill_flops': fl,
                }
                rows.append(row)

    # Pretty-print a compact table for the full-accounting mode at key lengths.
    print(f"\n{'='*100}")
    print("Full accounting (compressed rows + SW buffer + indexer + partial/overlap state + sink)")
    print(f"{'='*100}")
    print(f"{'T':>8} | {'op':>14} | {'KV elems':>14} | {'KV/GQA(1L)':>10} | "
          f"{'KV/GQA(5L)':>10} | {'FL/GQA(1L)':>10} | {'FL/GQA(5L)':>10}")
    print('-' * 100)
    for r in rows:
        if r['accounting_mode'] != 'full_accounting':
            continue
        if r['T'] not in (4096, 65536, 1048576):
            continue
        print(f"{r['T']:>8} | {r['op']:>14} | {r['kv_elements']:>14} | "
              f"{r['kv_ratio_vs_gqa_1l']:>10.4f} | {r['kv_ratio_vs_gqa_5l']:>10.4f} | "
              f"{r['flops_ratio_vs_gqa_1l']:>10.4f} | {r['flops_ratio_vs_gqa_5l']:>10.4f}")

    # Also print the optimistic (compressed-only) mode for comparison.
    print(f"\n{'='*100}")
    print("Compressed-KV-only accounting (original paper's optimistic number)")
    print(f"{'='*100}")
    print(f"{'T':>8} | {'op':>14} | {'KV elems':>14} | {'KV/GQA(1L)':>10} | {'KV/GQA(5L)':>10}")
    print('-' * 80)
    for r in rows:
        if r['accounting_mode'] != 'compressed_kv_only':
            continue
        if r['T'] not in (4096, 65536, 1048576):
            continue
        print(f"{r['T']:>8} | {r['op']:>14} | {r['kv_elements']:>14} | "
              f"{r['kv_ratio_vs_gqa_1l']:>10.4f} | {r['kv_ratio_vs_gqa_5l']:>10.4f}")

    # Summary headline numbers.
    print(f"\n{'='*100}")
    print("Headline numbers at T=1,048,576 (1M tokens)")
    print(f"{'='*100}")
    for mode in ['compressed_kv_only', 'full_accounting']:
        for r in rows:
            if r['T'] == 1048576 and r['op'] == 'hybrid_kch' and r['accounting_mode'] == mode:
                print(f"  Hybrid 3:1:1 ({mode}):")
                print(f"    KV / GQA8 (1-layer baseline) = {r['kv_ratio_vs_gqa_1l']*100:.2f}%")
                print(f"    KV / GQA8 (5-layer baseline) = {r['kv_ratio_vs_gqa_5l']*100:.2f}%")
                print(f"    FLOPs / GQA8 (1-layer)        = {r['flops_ratio_vs_gqa_1l']*100:.2f}%")
                print(f"    FLOPs / GQA8 (5-layer)        = {r['flops_ratio_vs_gqa_5l']*100:.2f}%")

    os.makedirs('results', exist_ok=True)

    # Sanitize non-finite floats to null before serializing. ``json.dump``
    # with default ``allow_nan=True`` emits non-standard ``NaN``/``Infinity``
    # literals that most downstream parsers (JS ``JSON.parse``, pandas
    # ``read_json`` with default flags, jq) reject. The T=0 edge case
    # (currently absent from main() but reachable via direct API call)
    # would produce ``inf`` from ``kv / baseline_1l`` when baseline_1l == 0;
    # without sanitization, the whole JSON file would be unparseable.
    #
    # Uses the centralized ``sanitize_for_json`` helper from kaggle_setup.py
    # (was a local ``_sanitize`` closure; centralizing removes 5 copies of
    # the same logic across run_*.py).
    #
    # The payload is written as a versioned envelope via ``results_schema``
    # (schema_version "1.0") so the file carries its accounting semantics
    # and byte convention alongside the rows. Legacy bare-array files remain
    # readable — see results_schema.parse / make_figures.load.
    metadata = {
        'model': 'analytic (closed-form formulas; NOT profiled from a forward pass)',
        'bytes_convention': 'kv_bytes = kv_elements * BF16_BYTES (2 bytes), '
                            'matching the DeepSeek-V4 §2.3.4 comparison',
        'accounting_modes': ['compressed_kv_only', 'full_accounting'],
        # Code/environment provenance; this experiment has no RNG (analytic
        # closed-form accounting), so no seed block is recorded.
        'provenance': capture_provenance(),
    }
    try:
        write_results('results/exp3_kv_cache.json', rows,
                      kind='exp3_kv_cache', metadata=metadata)
    except (TypeError, ValueError) as e:
        # Fallback: log the corruption and write with a string coercion so
        # results are not lost entirely. Non-finite floats have already been
        # converted to None by results_schema's internal sanitizer, so this
        # branch only fires on truly unexpected types (e.g. a tensor slipped
        # into a row).
        print(f'[run_kv_cache] WARNING: envelope serialization failed: {e}')
        write_json_atomic(
            sanitize_for_json(envelope(rows, kind='exp3_kv_cache',
                                       metadata=metadata)),
            'results/exp3_kv_cache.json', indent=2, default=str)
    print('\nSaved: results/exp3_kv_cache.json')
    append_run_manifest(make_run_manifest_entry(
        experiment='exp3_kv_cache',
        seeds=[],
        provenance=metadata['provenance'],
        hyperparameters={
            'seq_lengths': seq_lengths,
            'ops': ops,
            'accounting_modes': metadata['accounting_modes'],
            'bytes_convention': metadata['bytes_convention'],
            'randomness': 'none (analytic closed-form accounting)',
        },
        artifacts={'results': 'results/exp3_kv_cache.json'},
        results={
            'n_rows': len(rows),
            'max_T': max(seq_lengths),
        },
    ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
