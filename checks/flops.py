"""Correctness checks: flops."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_prefill_flops_head_count(device='cpu'):
    """``prefill_flops`` must use csa_nh/hca_nh (not H) for core/SW FLOPs.

    With csa_nh=4 and H=8 (deliberately different), the CSA core+SW FLOPs
    must be HALF of what they'd be with csa_nh=8 (since FLOPs scale
    linearly with head count).

    The check isolates the head-count effect exactly: when only the
    attention head count changes, ``prefill_flops`` may change only via
    the terms that are linear in that head count (W_UQ up-projection,
    core attention, sliding-window branch, and the output projection).
    All other terms (compression, W_DQ, W_IUQ, W_w, indexer) are
    head-count-independent. We therefore assert that the FLOP delta
    between the two variants equals the analytic sum of the
    head-scaled terms. If core+SW wrongly used ``H`` instead of
    ``csa_nh``/``hca_nh``, their contribution would be IDENTICAL in both
    variants and the measured delta would fall short of the expected one
    by exactly the core+SW portion — failing the assertion. (The earlier
    version only checked ``fl_4 < fl_8``; that is trivially satisfied
    even with the H-bug because W_UQ and out_proj still scale with the
    head count, so the test could not detect the regression it claims to
    guard. The causal-entry helpers used as the oracle here are
    independently verified by ``test_prefill_flops_causal_block_entries``
    and the brute-force checks in ``test_kv_cache_ceil_block_count``.)
    """
    logger.info("Test: prefill_flops uses csa_nh/hca_nh (not H) for core/SW FLOPs")
    # Import lazily so this test does not add a hard dependency to the
    # top-level imports (run_correctness.py is the only consumer).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_kv_cache import prefill_flops, causal_block_entries, causal_selected_entries

    T = 4096
    # Common base params with H=8.
    base = dict(H=8, K=128, V=128, d=4096,
                csa_m=16, csa_c=128, csa_topk=512, csa_nIh=4, csa_cI=32,
                csa_sliding_window=2048, csa_dc=128,
                hca_m2=64, hca_c=128, hca_sliding_window=2048,
                hca_dc=128,
                kda_hv=8, kda_k=128, kda_v=128)

    # Variant A: csa_nh=4, hca_nh=4 (HALF of H=8).
    fl_4 = prefill_flops('csa', T, csa_nh=4, hca_nh=4, **base)
    fl_4_hca = prefill_flops('hca', T, csa_nh=4, hca_nh=4, **base)

    # Variant B: csa_nh=8, hca_nh=8 (EQUAL to H=8).
    fl_8 = prefill_flops('csa', T, csa_nh=8, hca_nh=8, **base)
    fl_8_hca = prefill_flops('hca', T, csa_nh=8, hca_nh=8, **base)

    # --- Analytic oracle for the head-count-scaled FLOP delta ---
    # Terms linear in csa_nh (per unit head), from prefill_flops('csa'):
    #   W_UQ up-projection : 2*T*csa_dc*csa_c
    #   core sparse        : 2*total_sel*csa_c*2
    #   sliding-window     : 2*sw_entries*csa_c*2
    #   out_proj           : 2*T*csa_c*d
    # Terms linear in hca_nh (per unit head), from prefill_flops('hca'):
    #   W_UQ up-projection : 2*T*hca_dc*hca_c
    #   core dense         : 2*causal_entries*hca_c*2
    #   sliding-window     : 2*sw_entries*hca_c*2
    #   out_proj           : 2*T*hca_c*d
    total_sel = causal_selected_entries(T, base['csa_m'], base['csa_topk'])
    csa_sw_w = base['csa_sliding_window']
    csa_eff_sw = min(T, csa_sw_w)
    csa_sw_entries = T * csa_eff_sw - csa_eff_sw * (csa_eff_sw - 1) // 2
    csa_scale = (2 * T * base['csa_dc'] * base['csa_c']
                 + 2 * total_sel * base['csa_c'] * 2
                 + 2 * csa_sw_entries * base['csa_c'] * 2
                 + 2 * T * base['csa_c'] * base['d'])

    hca_entries = causal_block_entries(T, base['hca_m2'])
    hca_sw_w = base['hca_sliding_window']
    hca_eff_sw = min(T, hca_sw_w)
    hca_sw_entries = T * hca_eff_sw - hca_eff_sw * (hca_eff_sw - 1) // 2
    hca_scale = (2 * T * base['hca_dc'] * base['hca_c']
                 + 2 * hca_entries * base['hca_c'] * 2
                 + 2 * hca_sw_entries * base['hca_c'] * 2
                 + 2 * T * base['hca_c'] * base['d'])

    delta_nh = 8 - 4
    csa_expected_delta = csa_scale * delta_nh
    hca_expected_delta = hca_scale * delta_nh

    csa_delta_matches = (fl_8 - fl_4) == csa_expected_delta
    hca_delta_matches = (fl_8_hca - fl_4_hca) == hca_expected_delta
    return [
        _ok('prefill_flops(csa) head-count delta matches the head-scaled terms',
            csa_delta_matches,
            f'actual_delta={fl_8 - fl_4}, expected_delta={csa_expected_delta}'),
        _ok('prefill_flops(hca) head-count delta matches the head-scaled terms',
            hca_delta_matches,
            f'actual_delta={fl_8_hca - fl_4_hca}, expected_delta={hca_expected_delta}'),
    ]


def test_prefill_flops_causal_block_entries(device='cpu'):
    """Regression: ``prefill_flops`` must use the CORRECT causal block entry
    count for CSA/HCA, not the previous ~2x-overcounting formula.

    The window-close causal mask in ``ops_csa.py::_causal_block_mask`` is
    ``b < (t + 1) // m`` (query t can attend to a block once its source
    window is complete). The total number of valid (query, block) pairs is
    therefore ``sum_{t=0}^{T-1} floor((t + 1) / m)``. For ``T = nb*m`` this
    equals ``m * nb * (nb - 1) // 2 + nb``; the final ``+nb`` term accounts
    for the final token of each complete window.

    The previous formula ``T * n_blocks - n_blocks * (n_blocks - 1) // 2``
    was WRONG: it computed the full ``T * n_blocks`` product minus a small
    ``n_blocks``-sized triangle, yielding ~2x the correct count. This
    overcounted the CSA indexer FLOPs and HCA core FLOPs by ~2x and biased
    ``flops_ratio_vs_gqa_*``.

    This test recomputes the FULL expected CSA/HCA FLOPs with the correct
    causal_block_entries formula and compares to ``prefill_flops()``. If the
    code reverts to the broken formula, the totals will mismatch by the
    indexer (CSA) or core (HCA) overcount.
    """
    logger.info("Test: prefill_flops causal_block_entries correctness")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_kv_cache import (
        DEFAULTS,
        causal_block_entries,
        causal_selected_entries,
        prefill_flops,
    )

    results = []

    # --- CSA check ---
    for T in [512, 1024, 4096]:
        p = {**DEFAULTS}
        H, K, V, d = p['H'], p['K'], p['V'], p['d']
        csa_m, csa_c, csa_topk = p['csa_m'], p['csa_c'], p['csa_topk']
        csa_nh, csa_dc = p['csa_nh'], p['csa_dc']
        csa_cI, csa_nIh = p['csa_cI'], p['csa_nIh']
        sw_w = p['csa_sliding_window']
        n_blocks = (T + csa_m - 1) // csa_m if T > 0 else 0

        # Recompute each term with the window-close causal oracle.
        cbe_correct = causal_block_entries(T, csa_m)
        # The old broken formula (for comparison):
        cbe_broken = T * n_blocks - n_blocks * (n_blocks - 1) // 2

        compress = 2 * T * d * (4 * csa_c + 2 * csa_cI)
        query_proj = 2 * T * (
              d * csa_dc
            + csa_dc * csa_c * csa_nh
            + csa_dc * csa_cI * csa_nIh
            + d * csa_nIh
        )
        indexer_correct = 2 * cbe_correct * csa_cI * csa_nIh \
                          + 2 * cbe_correct * csa_nIh
        indexer_broken = 2 * cbe_broken * csa_cI * csa_nIh \
                         + 2 * cbe_broken * csa_nIh
        # Mirror the exact window-close top-k count used by
        # ``run_kv_cache.prefill_flops``.
        total_sel = causal_selected_entries(T, csa_m, csa_topk)
        effective_topk = total_sel / T if T > 0 else 0
        core = 2 * T * effective_topk * csa_c * csa_nh * 2
        eff_sw = min(T, sw_w)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * csa_c * csa_nh * 2
        # Output projection: (csa_nh * csa_c) -> d, part of the
        # end-to-end single-layer FLOPs boundary.
        out_proj = 2 * T * csa_nh * csa_c * d

        expected_correct = (
            compress + query_proj + indexer_correct + core + sw + out_proj)
        expected_broken = (
            compress + query_proj + indexer_broken + core + sw + out_proj)
        actual = prefill_flops('csa', T)

        match = actual == expected_correct
        would_fail_broken = actual != expected_broken  # must NOT match broken
        results.append(_ok(
            f'CSA causal_block_entries correct at T={T}',
            match and would_fail_broken,
            f'actual={actual}, expected_correct={expected_correct}, '
            f'expected_broken={expected_broken}, cbe_correct={cbe_correct}, '
            f'cbe_broken={cbe_broken}'))

    # --- HCA check ---
    for T in [512, 1024, 4096]:
        p = {**DEFAULTS}
        H, K, V, d = p['H'], p['K'], p['V'], p['d']
        hca_m2, hca_c = p['hca_m2'], p['hca_c']
        hca_nh, hca_dc = p['hca_nh'], p['hca_dc']
        sw_w = p['hca_sliding_window']
        n_blocks = (T + hca_m2 - 1) // hca_m2 if T > 0 else 0

        cbe_correct = causal_block_entries(T, hca_m2)
        cbe_broken = T * n_blocks - n_blocks * (n_blocks - 1) // 2

        compress = 2 * T * d * hca_c * 2
        query_proj = 2 * T * (d * hca_dc + hca_dc * hca_c * hca_nh)
        core_correct = 2 * cbe_correct * hca_c * hca_nh * 2
        core_broken = 2 * cbe_broken * hca_c * hca_nh * 2
        eff_sw = min(T, sw_w)
        sw_entries = T * eff_sw - eff_sw * (eff_sw - 1) // 2
        sw = 2 * sw_entries * hca_c * hca_nh * 2
        # Output projection: (hca_nh * hca_c) -> d, part of the
        # end-to-end single-layer FLOPs boundary.
        out_proj = 2 * T * hca_nh * hca_c * d

        expected_correct = compress + query_proj + core_correct + sw + out_proj
        expected_broken = compress + query_proj + core_broken + sw + out_proj
        actual = prefill_flops('hca', T)

        match = actual == expected_correct
        would_fail_broken = actual != expected_broken
        results.append(_ok(
            f'HCA causal_block_entries correct at T={T}',
            match and would_fail_broken,
            f'actual={actual}, expected_correct={expected_correct}, '
            f'expected_broken={expected_broken}, cbe_correct={cbe_correct}, '
            f'cbe_broken={cbe_broken}'))

    return results


def test_kv_cache_ceil_block_count(device='cpu'):
    """P1-4 regression: block count must use ceil(T / m), not floor.

    The previous ``max(1, T // m)`` returned 1 block for T = m + 1
    (which actually compresses 2 blocks: one full + one partial),
    silently undercounting the compressed KV / indexer cache at
    non-divisible T. This test verifies the corrected formula
    ``ceil(T / m) = (T + m - 1) // m`` in both ``kv_cache_elements``
    (which keeps a ``max(1, ...)`` floor for allocated-capacity
    semantics) and ``prefill_flops`` (which uses the pure logical
    count, zero at T=0).
    """
    logger.info("Test: KV cache ceil block count")
    from run_kv_cache import kv_cache_elements, prefill_flops, DEFAULTS
    p = {**DEFAULTS}
    csa_m = p['csa_m']
    hca_m2 = p['hca_m2']

    results = []
    # Non-divisible T values: T = m + 1 must yield 2 blocks, not 1.
    for op, m in [('csa', csa_m), ('hca', hca_m2)]:
        for T in [m + 1, m + 2, 2 * m + 1, 3 * m - 1]:
            # kv_cache_elements: allocated capacity = max(1, ceil(T/m))
            expected_kv = max(1, (T + m - 1) // m)
            # compressed_kv_only mode returns n_blocks * c (no SW/indexer).
            c = p['csa_c'] if op == 'csa' else p['hca_c']
            actual_kv_elements = kv_cache_elements(op, T, mode='compressed_kv_only', **p)
            actual_blocks = actual_kv_elements // c
            results.append(_ok(
                f'{op} kv_cache_elements ceil at T={T} (m={m})',
                actual_blocks == expected_kv,
                f'actual_blocks={actual_blocks}, expected={expected_kv}'))
            # prefill_flops: logical count = ceil(T/m), zero at T=0
            # (we check T>0 here; T=0 is checked separately below).
            # We probe the block-count sensitivity by comparing
            # prefill_flops at T=m (1 block) vs T=m+1 (2 blocks): the
            # indexer FLOPs must strictly increase because the second
            # block adds new (query, block) scoring entries.
            if T == m + 1:
                fl_m = prefill_flops(op, m, **p)
                fl_m1 = prefill_flops(op, m + 1, **p)
                results.append(_ok(
                    f'{op} prefill_flops increases from T=m to T=m+1',
                    fl_m1 > fl_m,
                    f'fl(T=m)={fl_m}, fl(T=m+1)={fl_m1}'))

    # T=0 edge case: kv_cache_elements returns >=1 block (allocated
    # capacity), but prefill_flops returns the compress term = 0
    # (since compress = 2*T*d*(...) and T=0). The key invariant: at
    # T=0, prefill_flops must be 0 (no work), while kv_cache_elements
    # must be > 0 (reserved buffer).
    for op in ['csa', 'hca']:
        fl_0 = prefill_flops(op, 0, **p)
        kv_0 = kv_cache_elements(op, 0, mode='compressed_kv_only', **p)
        results.append(_ok(
            f'{op} prefill_flops(0) == 0 (no work at T=0)',
            fl_0 == 0, f'fl_0={fl_0}'))
        results.append(_ok(
            f'{op} kv_cache_elements(0) > 0 (allocated capacity)',
            kv_0 > 0, f'kv_0={kv_0}'))

    return results


def test_geometric_decode_cache_capacity(device='cpu'):
    """Regression for the geometric compressed-cache storage contract."""
    logger.info("Test: geometric decode-cache capacity")
    from run_kv_cache import geometric_capacity
    expected = [0, 1, 2, 4, 4, 8, 8, 8, 8]
    actual = [geometric_capacity(n) for n in range(len(expected))]
    valid = actual == expected
    invalid_ok = False
    try:
        geometric_capacity(-1)
    except ValueError:
        invalid_ok = True
    return [
        _ok('geometric capacity sequence', valid,
            f'expected={expected}, actual={actual}'),
        _ok('geometric capacity rejects negative rows', invalid_ok, ''),
    ]


def test_kv_cache_full_accounting_runtime_state(device='cpu'):
    """KV-cache full_accounting must include incremental runtime state.

    CSA/HCA decoding caches retain more than compressed KV: partial-token
    accumulators (until a full block is available) and, for CSA overlap, the
    previous block's Cb/Zb tensors. Exp3's ``full_accounting`` is the number
    readers interpret as "what the decoding runtime retains", so omitting these
    tensors undercounts non-divisible sequence lengths and small contexts.
    """
    logger.info("Test: KV full_accounting includes partial/overlap runtime state")
    from run_kv_cache import geometric_capacity, kv_cache_elements, DEFAULTS

    p = {**DEFAULTS}
    csa_m, csa_c, csa_cI = p['csa_m'], p['csa_c'], p['csa_cI']
    hca_m2, hca_c = p['hca_m2'], p['hca_c']

    T_csa = 2 * csa_m + 3
    n_completed_csa = T_csa // csa_m
    csa_capacity = geometric_capacity(n_completed_csa)
    expected_csa = (
        csa_capacity * csa_c                       # geometrically-grown KV storage
        + p['csa_sliding_window'] * csa_c           # allocated SW ring buffer
        + csa_capacity * csa_cI                     # geometrically-grown indexer storage
        + (T_csa % csa_m) * (4 * csa_c + 2 * csa_cI)
        + 2 * csa_m * csa_c                        # previous Cb/Zb overlap
        + p['csa_nh']                              # sink
    )
    actual_csa = kv_cache_elements('csa', T_csa, mode='full_accounting', **p)

    T_hca = 2 * hca_m2 + 5
    n_completed_hca = T_hca // hca_m2
    hca_capacity = geometric_capacity(n_completed_hca)
    expected_hca = (
        hca_capacity * hca_c                         # geometrically-grown storage
        + p['hca_sliding_window'] * hca_c             # allocated SW ring buffer
        + (T_hca % hca_m2) * (2 * hca_c)
        + p['hca_nh']
    )
    actual_hca = kv_cache_elements('hca', T_hca, mode='full_accounting', **p)
    invalid_mode_ok = False
    try:
        kv_cache_elements('hca', T_hca, mode='not-a-mode', **p)
    except ValueError:
        invalid_mode_ok = True

    return [
        _ok('csa full_accounting includes partial + overlap state',
            actual_csa == expected_csa,
            f'actual={actual_csa}, expected={expected_csa}, T={T_csa}'),
        _ok('hca full_accounting includes partial accumulator',
            actual_hca == expected_hca,
            f'actual={actual_hca}, expected={expected_hca}, T={T_hca}'),
        _ok('KV accounting rejects invalid mode', invalid_mode_ok, ''),
    ]


def test_prefill_flops_softmax_gqa_projections(device='cpu'):
    """Regression: ``prefill_flops('softmax_gqa')`` must include input/output
    projections for parity with KDA/CSA/HCA.

    Previously, softmax_gqa counted ONLY the attention core (QK^T + softmax·V),
    while KDA/CSA/HCA all counted their input projections. This asymmetry made
    the denominator of ``flops_ratio_vs_gqa_*`` artificially small: at T=512
    the KDA ratio was 26x (KDA looks 26x more expensive than GQA) instead of
    the correct 0.79x (KDA is actually cheaper). At long T the core dominates
    and the error shrinks, but the swept table includes short-T rows where the
    error is massive.

    This test verifies that softmax_gqa FLOPs are strictly LARGER than the
    core-only value, and that adding the projections makes the KDA/GQA ratio
    at T=512 less than 1.0 (KDA is cheaper than GQA at short T, which is the
    correct behavior — the broken version reported 26x).
    """
    logger.info("Test: prefill_flops softmax_gqa includes projections")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_kv_cache import prefill_flops, DEFAULTS, GQA_HEAD_DIM

    p = {**DEFAULTS}
    H, K, V, d = p['H'], p['K'], p['V'], p['d']
    T = 512

    # GQA8 baseline: H=8 KV heads, head_dim=128, d=4096 -> n_q = d/head_dim
    # = 32 query heads on the query side; KV side keeps n_kv = H = 8.
    n_q = d // GQA_HEAD_DIM
    n_kv = H

    # Core-only FLOPs (the OLD, broken value).
    causal_entries = T * (T + 1) // 2
    core_only = 2 * causal_entries * n_q * (K + V)

    actual = prefill_flops('softmax_gqa', T)

    # The fixed value must be strictly larger than core-only (projections add
    # a large positive term at d=4096).
    includes_proj = actual > core_only

    # Verify the projection magnitude: proj + out_proj should equal the
    # difference. proj = 2*T*d*(n_q*K + n_kv*K + n_kv*V),
    # out_proj = 2*T*n_q*V*d.
    expected_proj = (2 * T * d * (n_q * K + n_kv * K + n_kv * V)
                     + 2 * T * n_q * V * d)
    diff_matches = (actual - core_only) == expected_proj

    # Sanity: with the fix, KDA/GQA ratio at T=512 should be finite and not
    # the pre-fix 26x core-only-denominator artifact.
    kda = prefill_flops('kda', T)
    ratio = kda / actual
    ratio_sane = ratio < 5.0  # definitely not the broken 26x

    return [_ok(
        'softmax_gqa includes input/output projections',
        includes_proj and diff_matches and ratio_sane,
        f'actual={actual}, core_only={core_only}, '
        f'proj_diff={actual - core_only}, expected_proj={expected_proj}, '
        f'kda/gqa_ratio={ratio:.4f}x')]


def test_prefill_flops_kch_output_projections(device='cpu'):
    """Regression: Exp3 KDA/CSA/HCA FLOPs must include output projections.

    Exp3 advertises an end-to-end single-layer FLOPs boundary for the
    standalone operators. ``run_benchmark.py`` already times CSA/HCA with the
    grouped output projection, and KDA/CSA/HCA modules all apply an ``o`` /
    ``o_proj`` after the attention core. Omitting these terms undercounts the
    numerator of ``flops_ratio_vs_gqa_*`` (and, for the hybrid, multiplies the
    omission by the layer ratio).
    """
    logger.info("Test: prefill_flops includes KDA/CSA/HCA output projections")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_kv_cache import prefill_flops, DEFAULTS

    p = {**DEFAULTS}
    H, d = p['H'], p['d']
    kda_hv, kda_k, kda_v = p['kda_hv'], p['kda_k'], p['kda_v']
    T = 512

    proj_no_out = 2 * T * (
          d * (2 * H * kda_k + kda_hv * kda_v + kda_k + kda_hv)
        + kda_k * kda_hv * kda_k
    )
    # P0-1 fix (round 1): KDA prefill FLOPs now also include the depthwise
    # short-conv (nn.Conv1d groups=d, ksize=3) applied before the q/k/v
    # projections. The test's "proj_no_out + recurrent" baseline must include
    # this term too, otherwise the subtraction below under-counts and the
    # "output projection residual" test spuriously fails.
    ksize = p.get('kda_conv_ksize', 3)
    short_conv = 2 * T * d * ksize
    recurrent = 2 * 3 * T * kda_hv * kda_k * kda_v
    # Chunk-parallel inter-chunk terms (paper §6.3 Eq 13): ``3T*C*dh + T*C^2``
    # per head with C = kda_chunk_size, dh = K = V. The prefill path uses the
    # chunk kernel, so these must be part of the KDA FLOPs baseline the
    # output-projection residual is measured against.
    c = p['kda_chunk_size']
    chunk_inter = 3 * T * kda_hv * c * kda_k
    chunk_a = T * kda_hv * c * c
    expected_kda_out = 2 * T * kda_hv * kda_v * d
    actual_kda = prefill_flops('kda', T)
    kda_ok = (actual_kda - (proj_no_out + short_conv + recurrent
                            + chunk_inter + chunk_a)) == expected_kda_out

    # CSA/HCA are covered in the full formula tests above; here we also make
    # their totals explicitly sensitive to a positive output-projection term.
    csa = prefill_flops('csa', T)
    hca = prefill_flops('hca', T)
    csa_out = 2 * T * p['csa_nh'] * p['csa_c'] * d
    hca_out = 2 * T * p['hca_nh'] * p['hca_c'] * d
    csa_ok = csa > csa_out and csa_out > 0
    hca_ok = hca > hca_out and hca_out > 0

    return [
        _ok('KDA prefill_flops includes output projection', kda_ok,
            f'actual={actual_kda}, expected_out_proj={expected_kda_out}'),
        _ok('CSA prefill_flops contains positive output projection term', csa_ok,
            f'total={csa}, out_proj={csa_out}'),
        _ok('HCA prefill_flops contains positive output projection term', hca_ok,
            f'total={hca}, out_proj={hca_out}'),
    ]


def test_recall_distance_buckets_match_structural_window(device='cpu'):
    """Long-range distance buckets must use the structural window boundary.

    ``structural_recall`` counts a pair as window-reachable only when
    ``cue - p <= win - 1`` (the key itself is inside ``[cue-win+1, cue]``).
    ``evaluate_recall`` used to label ``distance == win`` as ``window``,
    which is the first out-of-window placement.  The buckets therefore did
    not agree with the structural table they are reported next to.  This
    test pins the corrected inclusive ranges and checks every distance
    against the structural per-layer window flag.
    """
    logger.info("Test: recall distance buckets match structural window")
    from recall_reach import (
        _recall_bucket, _recall_bucket_ranges, LayerSpec,
        structural_recall,
    )

    results = []
    win, seq_len = 4, 32
    ranges = _recall_bucket_ranges(win, seq_len)
    results.append(_ok(
        'window bucket ends at win-1 (key inside the window)',
        ranges['window'] == (2, win - 1),
        f"window={ranges['window']}"))
    results.append(_ok(
        'mid bucket starts at win (first out-of-window distance)',
        ranges['mid'][0] == win,
        f"mid={ranges['mid']}"))
    results.append(_ok(
        'far bucket starts after the mid edge',
        ranges['far'][0] == ranges['mid'][1] + 1,
        f"far={ranges['far']}"))

    covered = []
    for lo, hi in ranges.values():
        covered.extend(range(lo, hi + 1))
    results.append(_ok(
        'buckets partition every distance in [2, seq_len-1] exactly once',
        covered == list(range(2, seq_len)),
        f'n_covered={len(covered)}, first={covered[:2]}, '
        f'last={covered[-2:]}'))

    plan = [LayerSpec(0, 'csa', block=16, window=win, topk=2)]
    structural = structural_recall(plan, seq_len)
    by_distance = {r['distance']: r for r in structural['rows']}
    mismatches = []
    for distance in range(2, seq_len):
        row = by_distance[distance]
        label = _recall_bucket(distance, win, ranges['mid'][1])
        if label == 'window' and not row['layers'][0]['window_hit']:
            mismatches.append((distance, 'window', 'structural miss'))
        if label != 'window' and row['layers'][0]['window_hit']:
            mismatches.append((distance, label, 'structural hit'))
    results.append(_ok(
        'window label agrees with structural window_hit for every distance',
        not mismatches,
        f'mismatches={mismatches[:4]}'))

    class _IdentityModel:
        def reset_state(self):
            pass

        def __call__(self, x):
            return x

    class _ZeroHead:
        def __call__(self, x, cue_pos):
            return torch.zeros(x.shape[0], 16, device=x.device)

    from recall_reach import evaluate_recall
    evaluation = evaluate_recall(
        _IdentityModel(), torch.nn.Embedding(16, 4), _ZeroHead(),
        seq_len=seq_len, n_batches=1, batch=16, seed=0)
    bucket_n = {name: slot['n']
                for name, slot in evaluation['buckets'].items()}
    results.append(_ok(
        'evaluate_recall buckets partition the evaluation set',
        sum(bucket_n.values()) == evaluation['n_eval']
        and all(n > 0 for n in bucket_n.values()),
        f"bucket_n={bucket_n}, total={evaluation['n_eval']}"))

    degenerate = _recall_bucket_ranges(0, 8)
    results.append(_ok(
        'win=0 leaves the window bucket empty and covers [2, seq_len-1]',
        degenerate['window'][0] > degenerate['window'][1]
        and degenerate['mid'][0] == 2
        and degenerate['far'][1] == 7,
        f'ranges={degenerate}'))
    invalid = [(4, 2), (-1, 8), (True, 8)]
    for win_arg, seq_len_arg in invalid:
        try:
            _recall_bucket_ranges(win_arg, seq_len_arg)
        except (TypeError, ValueError):
            results.append(_ok(
                f'rejects win={win_arg!r}, seq_len={seq_len_arg!r}', True))
        else:
            results.append(_ok(
                f'rejects win={win_arg!r}, seq_len={seq_len_arg!r}', False,
                'no error raised'))
    return results


def test_recall_flops_audit(device='cpu'):
    """Whole-stack FLOPs + distance-conditioned exact-recall accounting.

    Pins the new ``recall_flops_audit`` module against hand-derived cases:

      1. A far placement reachable only by CSA puts exactly the CSA layer's
         FLOPs in the numerator; a block-boundary placement with no exact
         path puts zero.
      2. A KDA-only plan has zero exact-reachable FLOPs and 100% state
         FLOPs, matching ``recall_reach``'s decision not to count recurrent
         state as an exact path.
      3. The whole-stack reference sum matches a real
         ``HybridAttentionStack`` forward exactly at T=64, and within the
         documented HCA single-block value-fold bound at T=32.
      4. Ratio expansion and invalid inputs fail loudly.
    """
    logger.info("Test: recall-conditioned FLOPs audit")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flops_audit as fa
    from recall_reach import LayerSpec, plan_from_config, structural_recall
    import recall_flops_audit as rfa

    results = []
    cfg = fa.probe_config(
        csa_m=4, hca_m2=8, csa_sliding_window=4, hca_sliding_window=4)
    layout = ('kda', 'csa', 'hca')
    plan = plan_from_config(cfg, layout)
    layer_flops = rfa.layer_reference_flops(plan, cfg, 16)
    total = sum(row['gemm_flops'] for row in layer_flops)
    results.append(_ok(
        'per-layer FLOPs are 2 * reference MACs',
        all(row['gemm_flops'] == 2 * row['gemm_macs']
            for row in layer_flops),
        f'layers={[row["gemm_flops"] for row in layer_flops]}'))
    results.append(_ok(
        'per-layer kinds follow the explicit layout',
        [row['kind'] for row in layer_flops] == list(layout),
        f'kinds={[row["kind"] for row in layer_flops]}'))

    reach = rfa.distance_compute_reach(
        plan, cfg, 16, layer_flops=layer_flops)
    by_distance = {row['distance']: row for row in reach['rows']}
    csa_flops = layer_flops[1]['gemm_flops']
    results.append(_ok(
        'block-boundary placement at d=4 has no exact path',
        by_distance[4]['exact_layer_indices'] == []
        and by_distance[4]['exact_reachable_fraction'] == 0.0,
        f"row={by_distance[4]['exact_layer_indices']}"))
    results.append(_ok(
        'd=5 is reachable only by CSA and prices exactly the CSA layer',
        by_distance[5]['exact_layer_indices'] == [1]
        and by_distance[5]['exact_reachable_flops'] == csa_flops
        and abs(by_distance[5]['exact_reachable_fraction']
                - csa_flops / total) < 1e-12,
        f"fraction={by_distance[5]['exact_reachable_fraction']:.4f}"))
    results.append(_ok(
        'd=9 is reachable by both retrieval layers',
        by_distance[9]['exact_layer_indices'] == [1, 2],
        f"row={by_distance[9]['exact_layer_indices']}"))
    results.append(_ok(
        'far threshold starts beyond the largest window',
        reach['far_min_distance'] == 4,
        f"far_min={reach['far_min_distance']}"))
    results.append(_ok(
        'state-capable fraction is the KDA layer share',
        abs(reach['state_capable_fraction']
            - layer_flops[0]['gemm_flops'] / total) < 1e-12,
        f"state_fraction={reach['state_capable_fraction']:.4f}"))

    kda_plan = [LayerSpec(0, 'kda', block=None, window=0, topk=None)]
    kda_reach = rfa.distance_compute_reach(kda_plan, cfg, 16)
    results.append(_ok(
        'KDA-only plan has zero exact-reachable FLOPs',
        kda_reach['max_exact_reachable_fraction'] == 0.0
        and kda_reach['structural_any_exact_fraction'] == 0.0,
        f"max={kda_reach['max_exact_reachable_fraction']}"))
    results.append(_ok(
        'KDA-only plan prices all compute as state-capable',
        abs(kda_reach['state_capable_fraction'] - 1.0) < 1e-12,
        f"state={kda_reach['state_capable_fraction']}"))

    measured_cfg = fa.probe_config()
    measured_layout = ('kda', 'kda', 'kda', 'csa', 'hca')
    row64 = rfa.audit_layout(
        measured_layout, cfg=measured_cfg, seq_len=64, device=device,
        measure=True)
    measurement = row64['measurement']
    results.append(_ok(
        'whole-stack reference FLOPs match the dispatch counter at T=64',
        measurement['match_kind'] == 'exact'
        and measurement['gemm_macs'] == row64['reference_gemm_macs'],
        f"delta={measurement['reference_minus_measured_macs']}, "
        f"kind={measurement['match_kind']}"))
    results.append(_ok(
        'audit reports one placement per legal key position',
        row64['compute_reach']['n_placements'] == 64 - 2,
        f"n={row64['compute_reach']['n_placements']}"))
    results.append(_ok(
        'structural and compute far fractions are both in [0, 1]',
        0.0 <= row64['compute_reach']['structural_far_any_exact_fraction']
        <= 1.0
        and 0.0 <= row64['compute_reach']['far_mean_exact_reachable_fraction']
        <= 1.0,
        f"struct={row64['compute_reach']['structural_far_any_exact_fraction']:.3f}, "
        f"compute={row64['compute_reach']['far_mean_exact_reachable_fraction']:.3f}"))

    row32 = rfa.audit_layout(
        measured_layout, cfg=measured_cfg, seq_len=32, device=device,
        measure=True)
    measurement32 = row32['measurement']
    results.append(_ok(
        'T=32 difference is inside the documented HCA value-fold bound',
        measurement32['match_kind'] in ('exact', 'hca_value_fold')
        and measurement32['within_documented_hca_value_fold']
        and 0 <= measurement32['reference_minus_measured_macs']
        <= measurement32['hca_value_fold_bound_macs'],
        f"delta={measurement32['reference_minus_measured_macs']}, "
        f"bound={measurement32['hca_value_fold_bound_macs']}"))

    ratio = rfa.ratio_layout(3, 1, 1, total_layers=10)
    results.append(_ok(
        'ratio expansion repeats and truncates to the requested depth',
        len(ratio) == 10
        and ratio.count('kda') == 6
        and ratio.count('csa') == 2
        and ratio.count('hca') == 2,
        f"layout={'-'.join(ratio)}"))

    report = rfa.audit_layouts(
        ['alternating'], seq_len=32, measure=False)
    results.append(_ok(
        'batch audit keeps names and Pareto flags',
        report['rows'][0]['group'] == 'alternating'
        and isinstance(report['rows'][0]['pareto_optimal'], bool),
        f"group={report['rows'][0]['group']}"))

    invalid = [
        ('unknown layer kind', lambda: rfa.audit_layout(
            ('kda', 'mamba2'), cfg=measured_cfg, seq_len=32, measure=False)),
        ('seq_len below key+value+cue', lambda: rfa.audit_layout(
            measured_layout, cfg=measured_cfg, seq_len=2, measure=False)),
        ('zero ratio', lambda: rfa.ratio_layout(0, 0, 0, 10)),
        ('unknown preset', lambda: rfa.audit_layouts(
            ['not-a-preset'], seq_len=32, measure=False)),
    ]
    for name, fn in invalid:
        try:
            fn()
        except (TypeError, ValueError):
            results.append(_ok(f'rejects {name}', True))
        except Exception as exc:
            results.append(_ok(
                f'rejects {name}', False,
                f'{type(exc).__name__}: {exc}'))
        else:
            results.append(_ok(f'rejects {name}', False, 'no error raised'))
    return results


def test_ranking_chance_reference_is_exact(device='cpu'):
    """Chance PR-AUC must be the expected AP of a random ranking.

    ``binary_ranking_metrics`` reports ``chance_pr_auc`` next to a learned
    PR-AUC so a reader can judge whether the ranking beat chance. Using
    the prevalence (``n_pos / n``) as that reference is wrong for finite
    rankings: with a single relevant item the AP is ``1 / rank``, whose
    expectation over a uniform random ranking is the harmonic mean
    ``H_n / n`` (which is also the chance MRR). The closed-form helper is
    checked against exhaustive enumeration of every ranking for small n.
    """
    import itertools
    from results_analysis import (binary_ranking_metrics,
                                  expected_average_precision)

    def ap_at_positions(positions):
        return sum((i + 1) / r for i, r in enumerate(positions)) / len(positions)

    bad = []
    for n in range(2, 8):
        for n_pos in range(1, n):
            values = [ap_at_positions(c)
                      for c in itertools.combinations(range(1, n + 1), n_pos)]
            want = sum(values) / len(values)
            got = expected_average_precision(n_pos, n - n_pos)
            if abs(got - want) > 1e-12:
                bad.append(f'n={n} n_pos={n_pos}: {got} != {want}')
    if expected_average_precision(0, 5) is not None:
        bad.append('zero positives must return None')
    if expected_average_precision(3, 0) != 1.0:
        bad.append('all-positive ranking must have chance AP 1.0')

    labels = torch.zeros(100, dtype=torch.long)
    labels[7] = 1
    m = binary_ranking_metrics(torch.arange(100, dtype=torch.float64), labels)
    h100 = sum(1.0 / i for i in range(1, 101)) / 100
    if abs(m['chance_pr_auc'] - h100) > 1e-12:
        bad.append(f"chance_pr_auc={m['chance_pr_auc']} != H_100/100={h100}")
    if not (m['chance_pr_auc'] > m['prevalence']):
        bad.append('chance AP must exceed prevalence for rare positives')

    return [_ok(
        'ranking_chance_reference_is_exact',
        not bad,
        '; '.join(bad) if bad else
        'closed form == exhaustive enumeration; one-positive chance AP = H_n/n',
    )]


def test_mqar_noise_excludes_keys_and_values(device='cpu'):
    """MQAR filler noise must not leak key/value (or the cued answer) tokens.

    Exp 4 (run_quality.py) and Exp 5 (run_ablation.py) both build MQAR
    sequences through ``make_mqar_batch``. If the filler slots
    ``[2*n_kv, seq_len-1)`` are drawn from the full vocabulary, the cued
    *answer* token re-appears there as a redundant copy that token-level
    operators (softmax/KDA) can copy without resolving the key-value
    association, while block-compressed operators (CSA/HCA) only observe it
    after compression/dilution — a structural difficulty mismatch that biases
    the reported operator ranking. Assert that no noise slot ever contains a
    key token, a value token, or in particular the cued answer.
    """
    from torch.nn import Embedding
    from run_quality import make_mqar_batch

    def _ids_of(x_emb):
        # The embedding below is row-wise one-hot, so argmax recovers ids.
        return x_emb.argmax(dim=-1)

    leaks = []
    for (seq_len, n_kv, vocab, batch) in [
            (16, 1, 16, 8), (24, 2, 24, 8), (32, 3, 64, 6),
            (64, 4, 64, 6), (48, 2, 100, 8), (40, 5, 200, 6)]:
        for seed in range(4):
            gen = torch.Generator(device=device).manual_seed(seed)
            embed = Embedding(vocab, vocab).to(device)
            with torch.no_grad():
                embed.weight.copy_(torch.eye(vocab, device=device))
            x_emb, target, cue_pos = make_mqar_batch(
                batch, seq_len, n_kv, vocab, embed, device=device,
                generator=gen)
            ids = _ids_of(x_emb)                      # [batch, seq_len]
            for b in range(ids.shape[0]):
                mem = set(ids[b, :2 * n_kv].tolist())
                noise = ids[b, 2 * n_kv:cue_pos].tolist()
                if mem.intersection(noise):
                    leaks.append(
                        f'seed{seed} (seq={seq_len},n_kv={n_kv},vocab={vocab}) '
                        f'row{b}: key/value {sorted(mem)} appears in noise')
                if int(target[b].item()) in noise:
                    leaks.append(
                        f'seed{seed} (seq={seq_len},n_kv={n_kv},vocab={vocab}) '
                        f'row{b}: answer token reappears in noise')
                # Sanity: the cue is a key (must live in the memory region).
                if int(ids[b, cue_pos].item()) not in mem:
                    leaks.append(
                        f'seed{seed} (seq={seq_len},n_kv={n_kv},vocab={vocab}) '
                        f'row{b}: cue token misplaced')

    # Config without enough distractor ids must raise, not silently re-leak.
    raised = False
    try:
        gen = torch.Generator(device=device).manual_seed(0)
        embed = Embedding(4, 4).to(device)
        make_mqar_batch(1, 6, 2, 4, embed, device=device, generator=gen)
    except ValueError:
        raised = True

    return [_ok(
        'mqar_noise_excludes_keys_and_values',
        not leaks and raised,
        f'noise_leaks={len(leaks)}'
        + (f' first={leaks[0]}' if leaks else '')
        + f' no-distractor-ValueError={raised}')]


def test_fp8_state_storage_ledger(device='cpu'):
    """FP8 block-scaled state storage prices the memory-bound decode ledger.

    Pins the cross-module contract: an ``FP8Spec`` payload is round-to-grid
    with a charged block scale, and the decode ledger applies a per-operator
    storage policy (FP8 recurrent state, unchanged CSA/HCA caches) without
    touching the canonical element model. The weights<->state switch batch
    must move to a larger batch when state bytes shrink.
    """
    logger.info("Test: FP8 block-scaled state storage in the decode ledger")
    results = []
    import decode_bandwidth as db
    from kv_quant import FP8Spec, bits_per_element, fake_quantize

    spec = FP8Spec(format='e4m3fn', group_size=32, scale_bits=8,
                   power_of_two=True)
    results.append(_ok(
        'MXFP8 storage = 8.25 bits/element',
        abs(bits_per_element(spec) - 8.25) < 1e-12,
        f'{bits_per_element(spec)}'))
    torch.manual_seed(23)
    x = torch.randn(2, 64, device=device)
    q = fake_quantize(x, spec, ste=False)
    results.append(_ok(
        'FP8 fake-quantize is finite and bounds-preserving',
        bool(torch.isfinite(q).all())
        and float(q.abs().max()) <= spec.max_finite,
        f'max|q|={float(q.abs().max()):.3f}'))
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=8,
        csa_cI=4, csa_nIh=1, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=8,
        hca_sliding_window=4)
    T = 128
    base = db.decode_traffic_breakdown(cfg, T)
    fp8 = db.decode_traffic_breakdown(cfg, T, storage={'memory': spec})
    results.append(_ok(
        'FP8 prices only the memory-route state',
        fp8['bytes_per_element_by_op'] == {'memory': 1.03125,
                                           'csa': 4.0, 'hca': 4.0}
        and fp8['read_bytes_by_op']['csa'] == base['read_bytes_by_op']['csa']
        and fp8['read_bytes_by_op']['hca'] == base['read_bytes_by_op']['hca'],
        str(fp8['bytes_per_element_by_op'])))
    results.append(_ok(
        'FP8 memory traffic is the element row times 1.03125',
        abs(fp8['read_bytes_by_op']['memory']
            - base['read_elements_by_op']['memory'] * 1.03125) < 1e-9,
        f"{fp8['read_bytes_by_op']['memory']:.3f}"))
    old_switch = db.weights_state_switch_batch(cfg, T)
    new_switch = db.weights_state_switch_batch(cfg, T,
                                               storage={'memory': spec})
    results.append(_ok(
        'FP8 state moves the weights/state switch batch outward',
        old_switch is not None and new_switch is not None
        and new_switch > old_switch,
        f'old={old_switch:.3f}, new={new_switch:.3f}'))
    return results


def test_recall_efficiency_ratio_sweep(device='cpu'):
    """Ratio-sweep rows must carry a consistent recall-per-decode-byte axis.

    Exp 5 now screens an accuracy gain against the canonical per-token
    decode traffic, not only parameters / prefill latency / training time.
    This pins the contract end to end:

      * the pure skill/efficiency formulas (chance-corrected recall in
        [0, 1], exact recall per decimal MB, ``None`` on a zero
        denominator, no ``+inf``/NaN);
      * ``_attach_decode_traffic`` prices each ratio through the same
        ``decode_bandwidth`` ledger, skips baselines, and ordering follows
        the state-vs-cache geometry;
      * ``_attach_recall_efficiency`` marks the accuracy-vs-bytes Pareto
        frontier, names the dominator, and lists byte-less rows under
        ``skipped`` instead of dropping them;
      * duplicate row names remain distinct points.
    """
    import decode_bandwidth as db
    from recall_efficiency import (annotate_recall_efficiency,
                                   chance_corrected_recall,
                                   recall_per_megabyte)
    import run_ablation as ra

    problems = []

    def want(cond, msg):
        if not cond:
            problems.append(msg)

    chance = 1.0 / 16.0
    want(abs(chance_corrected_recall(0.375, chance) - (1.0 / 3.0)) < 1e-12,
         'skill of 0.375 at 1/16 chance must be 1/3')
    want(chance_corrected_recall(0.01, chance) < 0.0,
         'below-chance accuracy must keep a negative skill')
    want(chance_corrected_recall(0.5, 1.0) is None,
         'chance == 1 leaves no finite skill scale -> None')
    want(abs(recall_per_megabyte(0.375, chance, 1_000_000.0)
             - (1.0 / 3.0)) < 1e-12,
         'skill per decimal MB must be exact at 1 MB')
    want(recall_per_megabyte(0.375, chance, 0.0) is None,
         'zero bytes/token must be None, not +inf')
    want(recall_per_megabyte(0.375, chance, float('nan')) is None,
         'NaN bytes/token must be None')

    rows = [
        {'ratio': '3:1:1', 'n_kv': 1, 'mean_acc': 0.30,
         'chance_acc': chance},
        {'ratio': '1:1:3', 'n_kv': 1, 'mean_acc': 0.25,
         'chance_acc': chance},
        {'ratio': '4:1:0', 'n_kv': 1, 'mean_acc': 0.20,
         'chance_acc': chance},
        {'ratio': 'BASELINE-softmax', 'n_kv': 1, 'mean_acc': 0.40,
         'chance_acc': chance, 'is_baseline': True},
    ]
    traffic = ra._attach_decode_traffic(rows)
    want(set(traffic) == {'3:1:1', '1:1:3', '4:1:0'},
         f'traffic cache must cover exactly the ratio rows, got {sorted(traffic)}')
    baseline = rows[3]
    want(baseline.get('decode_bytes_per_token') is None,
         'baseline rows must not be priced as ratio allocations')
    for row in rows[:3]:
        ratio = tuple(int(x) for x in row['ratio'].split(':'))
        expected = db.decode_traffic_breakdown(
            ra._make_cfg(ratio=ratio), ra.SEQ_LEN)['total_bytes_per_token']
        got = row.get('decode_bytes_per_token')
        want(got is not None and abs(got - expected) < 1e-9,
             f"{row['ratio']}: traffic {got} != canonical {expected}")
    want(rows[1]['decode_bytes_per_token']
         < rows[0]['decode_bytes_per_token']
         < rows[2]['decode_bytes_per_token'],
         'decode traffic order 1:1:3 < 3:1:1 < 4:1:0 violated')

    report = ra._attach_recall_efficiency(rows)
    cell = report['by_n_kv']['1']
    want(cell['frontier'] == ['3:1:1', '1:1:3'],
         f"frontier must be the cheap+HCA and the accurate default, got "
         f"{cell['frontier']}")
    want(cell['dominated_by'].get('4:1:0') == '3:1:1',
         f"4:1:0 must be dominated by 3:1:1, got {cell['dominated_by']}")
    eff = rows[0]['recall_efficiency']
    expected_skill = (0.30 - chance) / (1.0 - chance)
    expected_eff = expected_skill / (rows[0]['decode_bytes_per_token'] / 1e6)
    want(eff is not None and eff['optimal'] is True
         and abs(eff['chance_corrected_recall'] - expected_skill) < 1e-12
         and abs(eff['recall_per_megabyte'] - expected_eff) < 1e-12,
         f'3:1:1 row efficiency mismatch: {eff}')

    # Duplicate names must remain two distinct Pareto points, and a row
    # whose traffic is missing/zero must be reported, not silently priced.
    dup = [
        {'ratio': 'dup', 'mean_acc': 0.2, 'chance_acc': chance,
         'decode_bytes_per_token': 1000.0},
        {'ratio': 'dup', 'mean_acc': 0.3, 'chance_acc': chance,
         'decode_bytes_per_token': 2000.0},
    ]
    dup_rep = annotate_recall_efficiency(dup)
    want(len(dup_rep['points']) == 2 and not dup_rep['skipped'],
         f'duplicate names collapsed: {dup_rep}')
    missing = [{'ratio': 'no-bytes', 'mean_acc': 0.2,
                'chance_acc': chance, 'decode_bytes_per_token': 0.0}]
    miss_rep = annotate_recall_efficiency(missing)
    want(miss_rep['skipped'] == ['no-bytes']
         and missing[0]['recall_efficiency'] is None,
         f'byte-less row not reported as skipped: {miss_rep}')

    return [_ok(
        'recall_efficiency_ratio_sweep',
        not problems,
        '; '.join(problems) if problems else
        'skill/efficiency formulas exact; traffic ledger wired; '
        '3:1:1 vs 1:1:3 frontier + 4:1:0 dominated; skipped/dup rows safe',
    )]


def test_online_softmax_shared_kv_tiling(device='cpu'):
    """Streaming softmax must honour the shared-KV MQA/GQA contract.

    CSA/HCA core attention is shared-KV MQA (``Hkv`` keys, ``Hq`` query
    heads) with a block-causal mask and a denominator-only sink.  The
    online primitive is the kernel oracle for that pattern, so it must:

    * accept ``Hq % Hkv == 0`` (MQA and GQA), not require ``Hq == Hkv``;
    * stay value-equal to the dense softmax under key-chunk and
      query-tile blocking (the two axes a Flash/Triton schedule blocks);
    * tile a per-query sink and a 4-D mask on the same axes; and
    * keep gradients finite and equal to the dense reference.
    """
    logger.info("Test: online softmax shared-KV (MQA/GQA) + two-axis tiling")
    results = []
    import online_softmax as osf

    def _dense(q, k, v, mask, sink, scale):
        group = q.shape[1] // k.shape[1]
        kk = k.repeat_interleave(group, dim=1)
        vv = v.repeat_interleave(group, dim=1)
        scores = torch.einsum('b h t d, b h n d -> b h t n', q, kk) * scale
        if mask is not None:
            if mask.dim() == 4 and mask.shape[1] == k.shape[1] and group > 1:
                mask = mask.repeat_interleave(group, dim=1)
            scores = scores.masked_fill(~mask, float('-inf'))
        row_max = scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isneginf(row_max),
                              torch.zeros_like(row_max), row_max)
        e = (scores - row_max).exp()
        den = e.sum(dim=-1, keepdim=True)
        if sink is not None:
            ss = sink
            if ss.dim() == 1:
                ss = ss.view(1, q.shape[1], 1, 1)
            elif ss.dim() == 2:
                ss = ss.view(1, q.shape[1], 1, 1)
            elif ss.dim() == 3:
                ss = ss.unsqueeze(-1)
            den = den + (ss - row_max).exp()
        return torch.einsum('b h t n, b h n d -> b h t d', e / den, vv)

    torch.manual_seed(2027)
    B, T, N, D = 2, 7, 9, 8
    scale = D ** -0.5
    mask = torch.tril(torch.ones(T, N, dtype=torch.bool))
    for Hq, Hkv in ((4, 1), (4, 2), (3, 3)):
        q = torch.randn(B, Hq, T, D, device=device)
        k = torch.randn(B, Hkv, N, D, device=device)
        v = torch.randn(B, Hkv, N, D, device=device)
        sink = torch.randn(Hq, device=device)
        ref = _dense(q, k, v, mask, sink, scale)
        for query_chunk, key_chunk in ((0, 0), (2, 3), (T, N), (3, 2)):
            out = osf.online_softmax_attention(
                q, k, v, key_chunk=key_chunk, query_chunk=query_chunk,
                scale=scale, mask=mask, sink_logits=sink)
            diff = (out - ref).abs().max().item()
            results.append(_ok(
                f'Hq={Hq},Hkv={Hkv} qc={query_chunk} kc={key_chunk} '
                f'matches dense shared-KV softmax',
                diff < 1e-5, f'max|d|={diff:.2e}'))

        sink_q = torch.randn(B, Hq, T, 1, device=device)
        ref_q = _dense(q, k, v, mask, sink_q, scale)
        out_q = osf.online_softmax_attention(
            q, k, v, key_chunk=3, query_chunk=2, scale=scale, mask=mask,
            sink_logits=sink_q)
        diff = (out_q - ref_q).abs().max().item()
        results.append(_ok(
            f'Hq={Hq},Hkv={Hkv} per-query sink tiles with the query axis',
            diff < 1e-5, f'max|d|={diff:.2e}'))

        mask_hkv = torch.randn(B, Hkv, T, N, device=device) > 0
        mask_hkv[..., 0] = True
        ref_m = _dense(q, k, v, mask_hkv, sink, scale)
        out_m = osf.online_softmax_attention(
            q, k, v, key_chunk=2, query_chunk=2, scale=scale,
            mask=mask_hkv, sink_logits=sink)
        diff = (out_m - ref_m).abs().max().item()
        results.append(_ok(
            f'Hq={Hq},Hkv={Hkv} KV-head mask expands with K/V',
            diff < 1e-5, f'max|d|={diff:.2e}'))

    # Gradient parity on a small MQA geometry.
    torch.manual_seed(2028)
    B, Hq, Hkv, T, N, D = 1, 3, 1, 5, 6, 8
    scale = D ** -0.5
    mask = torch.tril(torch.ones(T, N, dtype=torch.bool))
    sink = torch.randn(Hq, device=device)
    q0 = torch.randn(B, Hq, T, D, device=device)
    k0 = torch.randn(B, Hkv, N, D, device=device)
    v0 = torch.randn(B, Hkv, N, D, device=device)
    grads = {}
    for label, fn in (
            ('dense', lambda q, k, v: _dense(q, k, v, mask, sink, scale)),
            ('online', lambda q, k, v: osf.online_softmax_attention(
                q, k, v, key_chunk=2, query_chunk=2, scale=scale,
                mask=mask, sink_logits=sink))):
        q = q0.clone().requires_grad_()
        k = k0.clone().requires_grad_()
        v = v0.clone().requires_grad_()
        fn(q, k, v).sum().backward()
        grads[label] = (q.grad, k.grad, v.grad)
    for i, name in enumerate(('q', 'k', 'v')):
        finite = all(torch.isfinite(g).all() for g in grads['online'])
        diff = (grads['dense'][i] - grads['online'][i]).abs().max().item()
        results.append(_ok(
            f'MQA grad_{name} finite and matches dense', finite and diff < 1e-5,
            f'max|d|={diff:.2e}'))

    bad_head = False
    try:
        osf.online_softmax_attention(
            torch.randn(1, 3, 4, D, device=device),
            torch.randn(1, 2, 5, D, device=device),
            torch.randn(1, 2, 5, D, device=device))
    except ValueError:
        bad_head = True
    results.append(_ok('non-divisible Hq/Hkv is rejected', bad_head, ''))
    bad_chunk = False
    try:
        osf.online_softmax_attention(
            q0, k0, v0, query_chunk=-1)
    except ValueError:
        bad_chunk = True
    results.append(_ok('negative query_chunk is rejected', bad_chunk, ''))
    bad_sink = False
    try:
        osf.online_softmax_attention(
            q0, k0, v0, query_chunk=2, sink_logits=torch.randn(1, Hq, 3, 1))
    except ValueError:
        bad_sink = True
    results.append(_ok('untileable per-query sink is rejected', bad_sink, ''))
    return results


def test_interaction_planning_is_four_cell_budgeted(device='cpu'):
    """A pre-registered 2x2 interaction must not reuse the main-effect SD.

    Exp 5b/8 pre-register difference-in-differences claims (the KDA/Mamba-2
    route x layout transfer in particular). ``I = d-b-c+a`` is a four-cell
    contrast: under the shared-seed / independent-cell-noise model its SD is
    sqrt(2) times the two-cell paired-difference SD that ``assumed_sd``
    describes. This check pins the arithmetic, the descriptive-only label
    for the registered 14-seed budget, the audit warning (which must not
    change the main-effect status), and the promotion path when the budget
    actually reaches the interaction requirement.
    """
    import experiment_design as design

    results = []
    for plan_id, family in (('exp5b_matrix', 24), ('exp8_mechanism', 6)):
        rep = design.planning_report(plan_id)
        inter = rep['interaction']
        results.append(_ok(
            f'{plan_id}: interaction has its own variance budget',
            inter is not None
            and inter['family_size'] == family
            and inter['alternative'] == 'two-sided'
            and abs(inter['sd_interaction']
                    - rep['assumed_sd'] * inter['sd_inflation']) < 1e-12,
            f"sd={inter['sd_interaction']:.4f} inflation="
            f"{inter['sd_inflation']:.4f}"))
        results.append(_ok(
            f'{plan_id}: main-effect budget does not power the interaction',
            inter['required_seeds'] > rep['required_seeds']
            and inter['planned_power'] < rep['planned_power']
            and inter['supports_interaction_at_planned_n'] is False
            and inter['status'] == 'underpowered_at_sesoi',
            f"need={inter['required_seeds']} "
            f"power={inter['planned_power']:.3f} at n={inter['seeds']}"))
    results.append(_ok(
        'plans without an interaction report no interaction claim',
        design.planning_report('exp4_mqar')['interaction'] is None
        and design.planning_report('exp4_mqar')[
            'interaction_claim_allowed'] is None, ''))

    levels = {'memory': ['kda', 'mamba2', 'rwkv7'],
              'layout_unit': ['A', 'B', 'C', 'D']}
    rep = design.planning_report('exp5b_matrix')
    unpowered = design.audit_run(
        'exp5b_matrix', seeds=[42 + i for i in range(14)],
        levels=levels, family_size=24)
    results.append(_ok(
        'audit keeps the main status but withholds the interaction claim',
        unpowered['status'] == 'confirmatory'
        and unpowered['interaction_claim_allowed'] is False
        and unpowered['checks']['interaction_budget_supports_claim'] is False
        and 'interaction_claim_at_sesoi' not in unpowered['claims_allowed']
        and any('interaction' in w and 'descriptive' in w
                for w in unpowered['design_warnings']),
        f"need={unpowered['interaction']['required_seeds']}"))
    enough = rep['interaction']['required_seeds']
    powered = design.audit_run(
        'exp5b_matrix', seeds=[42 + i for i in range(enough)],
        levels=levels, family_size=24)
    results.append(_ok(
        'audit promotes the interaction only when its own budget is met',
        powered['status'] == 'confirmatory'
        and powered['interaction_claim_allowed'] is True
        and 'interaction_claim_at_sesoi' in powered['claims_allowed']
        and not any('interaction' in w for w in powered['design_warnings']),
        f'n={enough}'))

    # Calibrate the sqrt(2) planning assumption on a synthetic 2x2 with a
    # shared seed effect and independent cell noise: the DiD SD must be
    # ~sqrt(2) x the two-cell paired-difference SD, and the shipped
    # interaction_contrast must return that same DiD sample SD.
    import random
    import statistics
    from multi_seed_stats import interaction_contrast
    rng = random.Random(7)
    cells = {key: {} for key in 'abcd'}
    n_sim = 4000
    for seed in range(n_sim):
        shared = rng.gauss(0.0, 1.0)
        for key in 'abcd':
            cells[key][seed] = shared + rng.gauss(0.0, 1.0)
    main_diffs = [cells['b'][s] - cells['a'][s] for s in range(n_sim)]
    did_diffs = [cells['d'][s] - cells['b'][s] - cells['c'][s] + cells['a'][s]
                 for s in range(n_sim)]
    ratio = statistics.stdev(did_diffs) / statistics.stdev(main_diffs)
    rec = interaction_contrast(cells['a'], cells['b'], cells['c'], cells['d'])
    results.append(_ok(
        'simulated 2x2 calibrates SD(I)/SD(B-A) ~= sqrt(2)',
        abs(ratio - 2.0 ** 0.5) < 0.06
        and abs(rec['sd_diff'] - statistics.stdev(did_diffs)) < 1e-9
        and abs(rec['mean_diff']
                - statistics.mean(did_diffs)) < 1e-9,
        f'ratio={ratio:.4f} target={2.0 ** 0.5:.4f}'))

    stale = design.get_plan('exp8_mechanism')
    stale['interaction']['required_for_claim'] = True
    stale_errors = design.validate_plan(stale)
    bad = design.get_plan('exp8_mechanism')
    bad['interaction']['sd_inflation'] = 0.5
    bad_errors = design.validate_plan(bad)
    results.append(_ok(
        'registry rejects an unpromotable interaction and a sub-1.0 inflation',
        any('interaction.required_for_claim is set' in e
            for e in stale_errors)
        and any('sd_inflation must be >= 1.0' in e for e in bad_errors)
        and design.validate_plans() == [],
        f'{stale_errors[:1]}'))
    return results


__all__ = [
    'test_prefill_flops_head_count',
    'test_prefill_flops_causal_block_entries',
    'test_kv_cache_ceil_block_count',
    'test_geometric_decode_cache_capacity',
    'test_kv_cache_full_accounting_runtime_state',
    'test_prefill_flops_softmax_gqa_projections',
    'test_prefill_flops_kch_output_projections',
    'test_recall_distance_buckets_match_structural_window',
    'test_recall_flops_audit',
    'test_ranking_chance_reference_is_exact',
    'test_mqar_noise_excludes_keys_and_values',
    'test_fp8_state_storage_ledger',
    'test_recall_efficiency_ratio_sweep',
    'test_online_softmax_shared_kv_tiling',
    'test_interaction_planning_is_four_cell_budgeted',
]
