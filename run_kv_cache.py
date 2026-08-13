"""Experiment 3 — KV cache and FLOPs analysis (improved accounting).

This is a rewritten, more rigorous version of the original KV-cache analysis.
It addresses the reviewer concern that the original accounting only counted
the *compressed* KV and ignored several auxiliary caches that a real
inference engine must retain:

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
hybrid to a single GQA8 layer, which understated the ratio by ~5x; we report
both for transparency.

The numbers mirror the efficiency discussion in DeepSeek-V4 §2.3.4 and Kimi
Linear §3.2 / §7.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import sanitize_for_json, write_json_atomic


# Reference GQA8, head_dim=128, BF16 baseline (as in DeepSeek-V4 §2.3.4).
GQA_H = 8
GQA_HEAD_DIM = 128
BF16_BYTES = 2

# Default architecture parameters (matching the paper's §3.3).
DEFAULTS = dict(
    H=8, K=128, V=128, d=4096,
    csa_m=16, csa_c=128, csa_topk=512, csa_nIh=4, csa_cI=32, csa_sliding_window=2048,
    # Number of attention heads for CSA / HCA core. The paper's §3.3 uses 8
    # heads (matching H); the sink has ``nh`` elements per layer. Previously
    # these keys were absent from DEFAULTS, so ``kv_cache_elements`` fell back
    # to ``p.get('csa_nh', H)`` and silently used H=8 — which happened to be
    # correct, but only by accident. Make the value explicit so the sink
    # count is correct even if H is ever changed.
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
)


def causal_block_entries(T: int, m: int) -> int:
    """Count visible (query, compressed-block) pairs.

    A compressed block becomes visible when its complete source window closes,
    i.e. block ``b`` is available from query ``(b + 1) * m - 1`` onward.
    """
    if not isinstance(T, int) or T < 0:
        raise ValueError(f'T must be a non-negative int, got {T!r}')
    if not isinstance(m, int) or m < 1:
        raise ValueError(f'm must be a positive int, got {m!r}')
    if T == 0:
        return 0
    n_full = T // m
    remainder = T % m
    return m * n_full * (n_full - 1) // 2 + n_full * (remainder + 1)


def geometric_capacity(n_rows: int) -> int:
    """Return the power-of-two storage capacity used by decode caches."""
    if not isinstance(n_rows, int) or n_rows < 0:
        raise ValueError(f'n_rows must be a non-negative int, got {n_rows!r}')
    if n_rows == 0:
        return 0
    return 1 << (n_rows - 1).bit_length()


def causal_selected_entries(T: int, m: int, topk: int) -> int:
    """Count selected block slots after a top-k cap under block causality."""
    if not isinstance(T, int) or T < 0:
        raise ValueError(f'T must be a non-negative int, got {T!r}')
    if not isinstance(m, int) or m < 1:
        raise ValueError(f'm must be a positive int, got {m!r}')
    if not isinstance(topk, int) or topk < 0:
        raise ValueError(f'topk must be a non-negative int, got {topk!r}')
    if T == 0 or topk == 0:
        return 0
    n_full = T // m
    remainder = T % m
    if n_full == 0:
        return 0
    k_cap = min(topk, n_full)
    if topk >= n_full:
        sum_before_last = n_full * (n_full - 1) // 2
    else:
        sum_before_last = (
            topk * (topk + 1) // 2
            + topk * (n_full - 1 - topk)
        )
    return m * sum_before_last + (remainder + 1) * k_cap


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
    p = {**DEFAULTS, **kw}
    _valid_keys = set(DEFAULTS.keys())
    _unknown = set(kw.keys()) - _valid_keys
    if _unknown:
        raise ValueError(
            f"kv_cache_elements: unknown keyword argument(s) "
            f"{sorted(_unknown)}. Valid keys: {sorted(_valid_keys)}.")
    _valid_ops = {'softmax_gqa', 'kda', 'csa', 'hca', 'hybrid_kch'}
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
            # Ca/Cb/Za/Zb (4*c) plus indexer K/Z (2*c_I). The previous
            # ``full_accounting`` mode omitted this runtime state, so it was not
            # truly full for non-divisible T.
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
        # correctly — previously the ratio was hardcoded, which silently
        # produced wrong KV-cache numbers for any ablation ratio.
        n_kda = p.get('hybrid_n_kda', 3)
        n_csa = p.get('hybrid_n_csa', 1)
        n_hca = p.get('hybrid_n_hca', 1)
        kda_part = n_kda * kv_cache_elements('kda', T, mode=mode, **p)
        csa_part = n_csa * kv_cache_elements('csa', T, mode=mode, **p)
        hca_part = n_hca * kv_cache_elements('hca', T, mode=mode, **p)
        return kda_part + csa_part + hca_part

    raise ValueError(op)


def prefill_flops(op: str, T: int, **kw):
    """Approximate prefill FLOPs (2 * MACs) for a single attention layer.

    Accounting conventions
    ----------------------
    Every attention op has TWO matmuls in its core: ``QK^T`` (over the key
    dim) and ``softmax(P) @ V`` (over the value dim). Both must be counted
    for the comparison to be fair across operators. The previous version
    counted BOTH for ``softmax_gqa`` (``2 * T * T * H * (K + V)``) but
    ONLY the ``QK^T`` term for CSA / HCA — undercounting their core FLOPs
    by ~2x and biasing the ``flops_ratio_vs_gqa_*`` columns roughly 2x in
    the hybrid's favor.

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
    ``HV * (3T*C*K + T*C^2)`` to the recurrent dominant term. The previous
    formula used ``2 * T * HV*K*V``, a ~3x underestimate of the dominant
    term alone. We also include the input projection FLOPs (q/k/v/g/beta
    plus the grouped output projection) for parity with CSA/HCA and the
    softmax baseline.

    For CSA, the ``compress`` term previously counted only ``W_aKV``
    (one ``T*d*c`` projection). The actual implementation
    (``ops_csa.py::naive_csa``) does SIX input projections:
    ``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx``. We count all six.
    We also count each operator's grouped output projection (KDA ``o_proj``,
    CSA/HCA ``o_proj``), so Exp3's "single-layer FLOPs" boundary matches
    Exp2's end-to-end standalone CSA/HCA boundary and the softmax baseline.
    """
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
        # The previous formula ``2 * T * T * H * (K + V)`` assumed a FULL
        # T*T attention matrix (non-causal), overcounting FLOPs by ~2x.
        # Since ``flops_ratio_vs_gqa_* = flops(op) / flops(softmax_gqa)``,
        # this 2x baseline bias made every other operator look ~2x
        # cheaper than it really is.
        causal_entries = T * (T + 1) // 2
        core = 2 * causal_entries * H * (K + V)
        # Input/output projections — counted for PARITY with KDA / CSA / HCA,
        # whose ``proj`` / ``compress`` / ``query_proj`` terms already include
        # them. The previous version omitted them from softmax_gqa, making the
        # denominator of ``flops_ratio_vs_gqa_*`` artificially small and the
        # ratios artificially large. At short T (e.g. 512) this swung the KDA
        # ratio from 0.79x (KDA is cheaper) to 26x (KDA looks 26x more
        # expensive) — a ~33x error in the headline comparison. At long T the
        # core dominates and the asymmetry shrinks to <2%, but the swept table
        # includes short-T rows where the error is large.
        #   q_proj : d -> H*K  -> T*d*H*K MACs
        #   k_proj : d -> H*K  -> T*d*H*K MACs
        #   v_proj : d -> H*V  -> T*d*H*V MACs
        #   o_proj : H*V -> d  -> T*H*V*d MACs
        proj = 2 * T * d * (2 * H * K + H * V)
        out_proj = 2 * T * H * V * d
        return core + proj + out_proj
    if op == 'kda':
        # Input projections — count the ACTUAL matmul shapes from
        # ops_fused.py::KDAHybridLayer, not an approximation. The previous
        # formula ``2 * T * d * kda_k * 5`` treated all 5 projections as
        # ``T*d*K`` MACs, dropping the H/HV factor — a ~5x underestimate
        # at the default H=8, K=128.
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
        # indexer applies the window-close causal mask before top-k, so a
        # block becomes visible at its final source token. Keep the exact
        # count in one helper so the experiment accounting matches
        # ops_csa._causal_block_mask and the decoding cache.
        causal_entries = causal_block_entries(T, csa_m)
        indexer = 2 * causal_entries * p['csa_cI'] * p['csa_nIh'] \
                  + 2 * causal_entries * p['csa_nIh']
        # Core sparse attention: QK^T (c term) + softmax·V (c term).
        # ``csa_lightning_indexer`` clamps topk to ``min(topk, n_blocks)``
        # AND masks non-causal blocks to -inf before top-k, so the EFFECTIVE
        # per-query topk is ``min(csa_topk, (t + 1) // csa_m)``. The AVERAGE
        # effective topk over all queries is therefore
        # ``sum_t min(csa_topk, (t + 1) // csa_m) / T`` under the
        # window-close mask. The helper counts the exact selected slots,
        # including the final token of each complete source window.
        total_sel = causal_selected_entries(T, csa_m, csa_topk)
        effective_topk = total_sel / T if T > 0 else 0
        core = 2 * T * effective_topk * csa_c * csa_nh * 2
        # Sliding window: causal window — query t attends to positions
        # [max(0, t-w+1), t], i.e. min(t+1, w) keys (NOT w keys for every
        # query). The previous formula ``T * w`` assumed every query
        # attends to exactly w keys, which overcounts by ~8x at T=512
        # (where w=2048 but only ~131K of the 1M claimed entries exist).
        # Total causal-window entries = T*w - w*(w-1)/2 when T >= w,
        # else T*(T+1)/2.
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
        # uses the same window-close mask as ops_csa: a heavy-compression
        # block becomes visible when its m2-token source window closes.
        # Keep the exact (query, block) count shared with the test oracle.
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
    raise ValueError(op)


def main():
    print('=' * 70)
    print('Experiment 3: KV Cache & FLOPs Analysis (improved accounting)')
    print('=' * 70)
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                   131072, 262144, 524288, 1048576]

    ops = ['softmax_gqa', 'kda', 'csa', 'hca', 'hybrid_kch']
    rows = []
    for T in seq_lengths:
        # Baseline: a single GQA8 layer (original paper's convention).
        baseline_1l = kv_cache_elements('softmax_gqa', T)
        # Baseline: a 5-layer GQA8 unit (apples-to-apples vs the 5-sub-layer hybrid).
        baseline_5l = 5 * baseline_1l
        flops_base_1l = prefill_flops('softmax_gqa', T)
        flops_base_5l = 5 * flops_base_1l

        for op in ops:
            for mode in ['compressed_kv_only', 'full_accounting']:
                kv = kv_cache_elements(op, T, mode=mode)
                fl = prefill_flops(op, T)
                row = {
                    'T': T,
                    'op': op,
                    'accounting_mode': mode,
                    'accounting_semantics': (
                        'compressed_kv_only'
                        if mode == 'compressed_kv_only'
                        else 'full_gqa_kv_cache'
                        if op == 'softmax_gqa'
                        else 'recurrent_state_plus_short_conv'
                        if op == 'kda'
                        else 'compressed_rows_plus_runtime_decode_state'
                        if op in ('csa', 'hca')
                        else 'hybrid_full_runtime_decode_state'
                    ),
                    'kv_elements': kv,
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
    sanitized = [sanitize_for_json(r) for r in rows]
    try:
        write_json_atomic(sanitized, 'results/exp3_kv_cache.json',
                          indent=2, allow_nan=False)
    except (TypeError, ValueError) as e:
        # Fallback: log the corruption and write without allow_nan=False
        # so results are not lost entirely. Non-finite values would have
        # been converted to None by sanitize_for_json above, so this branch
        # only fires on truly unexpected types (e.g. a tensor slipped in).
        print(f'[run_kv_cache] WARNING: JSON serialization failed: {e}')
        write_json_atomic(sanitized, 'results/exp3_kv_cache.json',
                          indent=2, default=str)
    print('\nSaved: results/exp3_kv_cache.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
