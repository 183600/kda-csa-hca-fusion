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
    hca_m2=64, hca_c=128, hca_sliding_window=2048,
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
    csa_cI, csa_nIh = p['csa_cI'], p['csa_nIh']

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
        # QK^T (K term) + softmax·V (V term) over T x T x H heads.
        return 2 * T * T * H * (K + V)
    if op == 'kda':
        # Input projections (q, k, v, g_down+g_up, beta) — five T*d*?
        # matmuls. q/k/v are d -> H*K / H*K / HV*V; g is d->K->HV*K (two
        # matmuls); beta is d->HV. We approximate as 5 * T * d * kda_k
        # (treating g's two-pass as one T*d*K + one T*K*HV*K, dominated by
        # the second pass at T*K*HV*K ≈ T*d*K under typical K≈d/HV).
        proj = 2 * T * d * kda_k * 5
        # Recurrence: ~3 HV*K*V MACs per step (see docstring).
        recurrent = 2 * 3 * T * kda_hv * kda_k * kda_v
        return proj + recurrent
    if op == 'csa':
        n_blocks = max(1, T // csa_m)
        # Compression: SIX input projections (W_aKV, W_bKV, W_aZ, W_bZ,
        # W_KV_idx, W_Z_idx), each T*d*c or T*d*c_I. The first four are
        # T*d*c; the last two are T*d*c_I.
        compress = 2 * T * d * (4 * csa_c + 2 * p['csa_cI'])
        # Indexer: per-head similarities T * n_blocks * c_I * nIh, then
        # weighted sum across heads T * n_blocks * nIh.
        indexer = 2 * T * p['csa_cI'] * p['csa_nIh'] * n_blocks \
                  + 2 * T * p['csa_nIh'] * n_blocks
        # Core sparse attention: QK^T (c term) + softmax·V (c term).
        core = 2 * T * csa_topk * csa_c * H * 2
        # Sliding window: QK^T + softmax·V over the local window.
        sw = 2 * T * p['csa_sliding_window'] * csa_c * H * 2
        return compress + indexer + core + sw
    if op == 'hca':
        n_blocks = max(1, T // hca_m2)
        # Compression: TWO input projections (W_KV, W_Z), each T*d*c.
        compress = 2 * T * d * hca_c * 2
        # Core dense attention over all n_blocks: QK^T (c) + softmax·V (c).
        core = 2 * T * n_blocks * hca_c * H * 2
        # Sliding window: QK^T + softmax·V over the local window.
        sw = 2 * T * p['hca_sliding_window'] * hca_c * H * 2
        return compress + core + sw
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
    with open('results/exp3_kv_cache.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print('\nSaved: results/exp3_kv_cache.json')


if __name__ == '__main__':
    main()
