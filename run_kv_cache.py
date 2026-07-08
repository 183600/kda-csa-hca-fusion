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

  * ``compressed_kv_only``  — the optimistic number (matches the original
    paper's "1.01% of GQA8" claim). This is what you get if you only count
    the compressed KV entries.
  * ``full_accounting``     — includes every auxiliary cache listed above.
    This is the number a production inference engine would actually pay.

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
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
)


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
        'full_accounting'     — compressed KV + sliding-window KV + indexer key
                                 cache + compression metadata + sink.
    """
    p = {**DEFAULTS, **kw}
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
            short_conv_state = 2 * p['d']
            return recurrent_state + short_conv_state
        # compressed_kv_only: just the recurrent state.
        return recurrent_state

    if op == 'csa':
        # Use max(1, ...) so T < csa_m still reports 1 block (the partial
        # block that a real engine would allocate). Without this, T=0 would
        # report 0 compressed KV elements, which is technically correct but
        # makes the KV/GQA ratio 0/0 = NaN at T=0. The slight overestimate at
        # T < csa_m (1 block instead of 0) is negligible next to the sliding-
        # window term and matches what a production engine actually allocates
        # (it reserves the block buffer upfront, not lazily per token).
        n_blocks = max(1, T // csa_m)
        # Compressed KV: n_blocks entries of c elements (keys serve as values).
        compressed = n_blocks * csa_c
        if mode == 'full_accounting':
            # Sliding-window branch: uncompressed local KV, c per token, for the
            # last `csa_sw` tokens. In decoding we only keep the last window.
            sw = min(T, csa_sw) * csa_c
            # Indexer key cache: n_blocks compressed indexer keys of c_I elements.
            indexer = n_blocks * csa_cI
            # Compression metadata: the per-block softmax weights Z are recomputed
            # from the input hidden state during decoding, so they are NOT cached.
            # Sink: nh elements (negligible, included for completeness —
            # documented here even though the value is tiny).
            sink = p.get('csa_nh', H)
            return compressed + sw + indexer + sink
        return compressed

    if op == 'hca':
        # Same max(1, ...) rationale as the 'csa' branch.
        n_blocks = max(1, T // hca_m2)
        compressed = n_blocks * hca_c
        if mode == 'full_accounting':
            sw = min(T, hca_sw) * hca_c
            sink = p.get('hca_nh', H)
            return compressed + sw + sink
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
    ~6 * T * HV*K*V FLOPs total. The previous formula used
    ``2 * T * HV*K*V``, a ~3x underestimate. We also include the input
    projection FLOPs (q/k/v/g/beta) for parity with CSA/HCA, whose
    ``compress`` term already includes the input projection.

    For CSA, the ``compress`` term previously counted only ``W_aKV``
    (one ``T*d*c`` projection). The actual implementation
    (``ops_csa.py::naive_csa``) does SIX input projections:
    ``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx``. We count all six.
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
        return 2 * causal_entries * H * (K + V)
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
        # Recurrence: ~3 HV*K*V MACs per step (see docstring).
        recurrent = 2 * 3 * T * kda_hv * kda_k * kda_v
        return proj + recurrent
    if op == 'csa':
        n_blocks = max(1, T // csa_m)
        # KV-side compression: SIX input projections (W_aKV, W_bKV, W_aZ,
        # W_bZ, W_KV_idx, W_Z_idx). The first four are T*d*c; the last
        # two are T*d*c_I.
        compress = 2 * T * d * (4 * csa_c + 2 * p['csa_cI'])
        # Query-side projections (W_DQ, W_UQ, W_IUQ, W_w) — previously
        # OMITTED, undercounting CSA's prefill FLOPs by ~8% at the default
        # config. The parity comment ("compress term already includes the
        # input projection") was wrong: compress only covers the KV side.
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
        # weighted sum across heads T * n_blocks * nIh.
        # The lightning indexer applies the causal block mask BEFORE top-k
        # (csa_lightning_indexer masks non-causal blocks to -inf), so query t
        # only scores floor(t / csa_m) valid blocks. Total causal block
        # entries = T*n_blocks - n_blocks*(n_blocks-1)/2 (triangular). The
        # previous formula used the full T*n_blocks product, overcounting the
        # aggregation term by ~2x and biasing flops_ratio_vs_gqa_*.
        causal_block_entries = T * n_blocks - n_blocks * (n_blocks - 1) // 2
        indexer = 2 * causal_block_entries * p['csa_cI'] * p['csa_nIh'] \
                  + 2 * causal_block_entries * p['csa_nIh']
        # Core sparse attention: QK^T (c term) + softmax·V (c term).
        # ``csa_lightning_indexer`` clamps topk to ``min(topk, n_blocks)``
        # AND masks non-causal blocks to -inf before top-k, so the EFFECTIVE
        # per-query topk is ``min(csa_topk, floor(t / csa_m))``. Average
        # effective topk over all queries is roughly
        # ``min(csa_topk, n_blocks / 2)`` (causal triangular). The previous
        # formula used ``csa_topk`` directly, overcounting by up to 16x at
        # short T (e.g. T=512 -> n_blocks=32 -> effective topk=16, but
        # csa_topk=512 in the default config -> 32x overcount). Use the
        # clamped value; for very long T (n_blocks >> csa_topk) this
        # converges to the original formula.
        effective_topk = min(csa_topk, max(1, n_blocks // 2))
        core = 2 * T * effective_topk * csa_c * H * 2
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
        sw = 2 * sw_entries * csa_c * H * 2
        return compress + query_proj + indexer + core + sw
    if op == 'hca':
        n_blocks = max(1, T // hca_m2)
        # KV-side compression: TWO input projections (W_KV, W_Z), each T*d*c.
        compress = 2 * T * d * hca_c * 2
        # Query-side projections (W_DQ, W_UQ) — previously OMITTED,
        # undercounting HCA's prefill FLOPs by ~11% at the default config.
        #   W_DQ : d -> dc    -> T*d*hca_dc MACs
        #   W_UQ : dc -> c*nh -> T*hca_dc*hca_c*H MACs
        hca_dc = p.get('hca_dc', 128)
        hca_nh = p.get('hca_nh', H)
        query_proj = 2 * T * (d * hca_dc + hca_dc * hca_c * hca_nh)
        # Core dense attention over the compressed blocks. The dense branch
        # in naive_hca applies the causal block mask (query t attends only
        # to blocks STRICTLY before floor(t / hca_m2)), so the actual number
        # of (query, block) entries is the causal triangular count
        # ``T*n_blocks - n_blocks*(n_blocks-1)/2``, NOT the full
        # ``T * n_blocks``. The previous formula used the full product,
        # overcounting HCA core FLOPs by ~2x and biasing
        # flops_ratio_vs_gqa_* accordingly. Mirrors the softmax causal-tri
        # fix above and the CSA indexer fix.
        causal_block_entries = T * n_blocks - n_blocks * (n_blocks - 1) // 2
        core = 2 * causal_block_entries * hca_c * H * 2
        # Sliding window: causal window (same fix as CSA above).
        sw_w = p['hca_sliding_window']
        eff_sw = min(T, sw_w)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * hca_c * H * 2
        return compress + query_proj + core + sw
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
                    'kv_elements': kv,
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
    print("Full accounting (compressed KV + sliding window + indexer + sink)")
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
    def _sanitize(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(x) for x in o]
        return o
    sanitized = [_sanitize(r) for r in rows]
    try:
        text = json.dumps(sanitized, indent=2, allow_nan=False)
    except (TypeError, ValueError) as e:
        # Fallback: log the corruption and write without allow_nan=False
        # so results are not lost entirely. Non-finite values would have
        # been converted to None by _sanitize above, so this branch only
        # fires on truly unexpected types (e.g. a tensor slipped in).
        print(f'[run_kv_cache] WARNING: JSON serialization failed: {e}')
        text = json.dumps(sanitized, indent=2, default=str)
    with open('results/exp3_kv_cache.json', 'w') as f:
        f.write(text)
    print('\nSaved: results/exp3_kv_cache.json')


if __name__ == '__main__':
    main()
