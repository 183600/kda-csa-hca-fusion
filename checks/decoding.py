"""Correctness checks: decoding."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_decoding_batch_size_change(device='cpu'):
    """Regression: ``SoftmaxAttnDecoding`` / ``KDAAttnDecoding`` must handle
    batch-size changes between forward calls without crashing.

    Previously, ``SoftmaxAttnDecoding`` cached K/V on the first forward and
    concatenated on subsequent forwards. If the batch size changed between
    calls (e.g. prefill with B>1 then decode with B=1, or train batch=16
    then eval batch=8), ``torch.cat`` would raise
    ``RuntimeError: Sizes of tensors must match except in dimension 1``
    because the batch dims (dim=0) differed.

    Similarly, ``KDAAttnDecoding`` cached the recurrent state, which would
    crash inside the recurrence on a batch-size change (the state has the
    old B, the new q/k/v have the new B, and the einsums broadcast-
    incompatibly).

    The fix adds a batch-size guard to both classes: on a batch-size
    change, the cache/state is reset and the new batch starts fresh.
    Mirrors the batch-size guard in ``HybridKCHAttention``.
    """
    logger.info("Test: decoding modules handle batch-size change without crashing")
    torch.manual_seed(222)
    # Import lazily so this test does not add a hard dependency to the
    # top-level imports.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_decoding import SoftmaxAttnDecoding, KDAAttnDecoding

    d_model = 32
    results = []

    # --- SoftmaxAttnDecoding ---
    sm = SoftmaxAttnDecoding(d_model).to(device).eval()
    # First forward: B=2, T=4 (prefill with batch=2).
    x_b2 = torch.randn(2, 4, d_model, device=device) * 0.1
    with torch.no_grad():
        y1 = sm(x_b2)
    # Second forward: B=1, T=1 (decode with batch=1).
    # Without the guard, torch.cat([B=2 cache, B=1 k], dim=1) crashes.
    x_b1 = torch.randn(1, 1, d_model, device=device) * 0.1
    try:
        with torch.no_grad():
            y2 = sm(x_b1)
        sm_ok = (y2.shape == (1, 1, d_model)
                 and torch.isfinite(y2).all().item())
        sm_err = ''
    except Exception as e:
        sm_ok = False
        sm_err = f'{type(e).__name__}: {e}'
    results.append(_ok('SoftmaxAttnDecoding survives B=2 -> B=1', sm_ok,
                       f'y2.shape={tuple(y2.shape) if sm_ok else "n/a"}, err={sm_err}'))

    # --- KDAAttnDecoding ---
    kda = KDAAttnDecoding(d_model).to(device).eval()
    # First forward: B=2, T=4.
    with torch.no_grad():
        y3 = kda(x_b2)
    # Second forward: B=1, T=1. Without the guard, the recurrence crashes
    # because the state has B=2 but q/k/v have B=1.
    try:
        with torch.no_grad():
            y4 = kda(x_b1)
        kda_ok = (y4.shape == (1, 1, d_model)
                  and torch.isfinite(y4).all().item())
        kda_err = ''
    except Exception as e:
        kda_ok = False
        kda_err = f'{type(e).__name__}: {e}'
    results.append(_ok('KDAAttnDecoding survives B=2 -> B=1', kda_ok,
                       f'y4.shape={tuple(y4.shape) if kda_ok else "n/a"}, err={kda_err}'))

    return results


def test_csa_decoding_cache_correctness(device='cpu'):
    """Incremental CSA decoding == naive_csa on the full sequence.

    Feeds T tokens one at a time through CSADecodingCache and compares
    the per-token outputs to naive_csa called once on the full
    sequence. Uses topk=100 (select all valid blocks) so the output is
    permutation-invariant over the selected set, sidestepping the
    torch.topk tie-breaking artifact.
    """
    from ops_decoding_cache import CSADecodingCache
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 100, 4
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=1)
    T = 17
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    o_full = naive_csa(
        H, p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
        p['Ba'], p['Bb'], p['W_DQ'], p['W_UQ'], p['W_IUQ'],
        p['W_w'], p['W_KV_idx'], p['W_Z_idx'], p['B_idx'],
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=win, sink_logits=None, use_ste=False,
        normalize_qk=True,
    )

    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    outs = []
    for t in range(T):
        H_t = H[:, t:t+1]
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, q_idx, w_idx,
            topk=topk, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=None, use_ste=False,
        )
        outs.append(o_t.reshape(B, 1, nh * c))
    o_inc = torch.cat(outs, dim=1).to(o_full.dtype)

    max_diff = (o_full - o_inc).abs().max().item()
    # Tolerance tightened from 1e-4 to 1e-5 (round 9 audit). The observed
    # max_diff is ~1.8e-7; the previous 1e-4 tolerance was ~1000x looser than
    # the actual error and 10x looser than the file's own TOL['match'][fp32]
    # standard (1e-5). A real cache bug producing a 5e-5 error (wrong cache
    # update, off-by-one causal mask, partial-accumulator bug) would have
    # passed silently. 1e-5 is 100x tighter than 1e-4 while still 100x
    # looser than the observed error, giving cross-platform headroom.
    return [_ok(
        'csa_decoding_cache_matches_naive',
        max_diff < 1e-5,
        f'max_diff={max_diff:.6e} (tol=1e-5), T={T}, m={m}, win={win}, '
        f'topk={topk} (select-all)')]


def test_hca_decoding_cache_correctness(device='cpu'):
    """Incremental HCA decoding == naive_hca on the full sequence."""
    from ops_decoding_cache import HCADecodingCache
    torch.manual_seed(42)
    d, c, m2, nh, dc = 32, 8, 4, 2, 8
    win = 4
    p = _make_hca_decoding_params(d, c, m2, nh, dc, device, seed=2)
    T = 17
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    o_full = naive_hca(
        H, p['W_KV'], p['W_Z'], p['B_pos'],
        p['W_DQ'], p['W_UQ'],
        m2=m2, nh=nh, c=c, dc=dc,
        sliding_window=win, sink_logits=None,
    )

    cache = HCADecodingCache(B, c, m2, win, device, torch.float32)
    outs = []
    for t in range(T):
        H_t = H[:, t:t+1]
        C, Z, q = _project_hca_for_cache(H_t, p, c, dc, nh)
        cache.append_step(C, Z, p['B_pos'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, nh=nh, scale=1.0, sink_logits=None,
        )
        outs.append(o_t.reshape(B, 1, nh * c))
    o_inc = torch.cat(outs, dim=1).to(o_full.dtype)

    max_diff = (o_full - o_inc).abs().max().item()
    # Tolerance tightened from 1e-4 to 1e-5 (round 9 audit). The observed
    # max_diff is ~1.2e-7; the previous 1e-4 tolerance was ~1000x looser than
    # the actual error. See test_csa_decoding_cache_correctness for the full
    # rationale.
    return [_ok(
        'hca_decoding_cache_matches_naive',
        max_diff < 1e-5,
        f'max_diff={max_diff:.6e} (tol=1e-5), T={T}, m2={m2}, win={win}')]


def test_csa_decoding_cache_compressed_blocks_match(device='cpu'):
    """The compressed block cache stores EXACTLY naive_csa's C_comp."""
    from ops_decoding_cache import CSADecodingCache
    from ops_csa import csa_compress_kv_overlapped, csa_compress_kv
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 100, 4
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=3)
    T = 16  # divisible by m for a clean comparison
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    # Full compression.
    Ca_f, Cb_f, Za_f, Zb_f, K_idx_f, Z_idx_f, _, _, _ = _project_csa_for_cache(
        H, p, c, c_I, dc, nh, nIh)
    C_comp_full = csa_compress_kv_overlapped(
        Ca_f, Cb_f, Za_f, Zb_f, p['Ba'], p['Bb'], m)
    K_IComp_full = csa_compress_kv(K_idx_f, Z_idx_f, p['B_idx'], m)

    # Incremental.
    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    for t in range(T):
        H_t = H[:, t:t+1]
        Ca, Cb, Za, Zb, K_idx, Z_idx, _, _, _ = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])

    c_diff = (C_comp_full - cache.C_comp).abs().max().item()
    k_diff = (K_IComp_full - cache.K_IComp).abs().max().item()
    n_blocks_match = cache.n_blocks == T // m
    return [_ok(
        'csa_decoding_cache_compressed_blocks_match',
        c_diff < 1e-5 and k_diff < 1e-5 and n_blocks_match,
        f'C_comp_diff={c_diff:.6e}, K_IComp_diff={k_diff:.6e}, '
        f'n_blocks_inc={cache.n_blocks} (expected {T // m})')]


def test_csa_decoding_cache_sliding_window_ring_buffer(device='cpu'):
    """The SW ring buffer holds the last ``win`` entries in causal order."""
    from ops_decoding_cache import CSADecodingCache, _SlidingWindowRingBuffer
    torch.manual_seed(42)
    B, win, c = 1, 4, 8
    buf = _SlidingWindowRingBuffer(B, win, c, device, torch.float32)

    # Append 2*win + 1 entries; the buffer should hold only the last win.
    entries = []
    for i in range(2 * win + 1):
        x = torch.full((B, 1, c), float(i), device=device)
        buf.append(x)
        entries.append(i)
    contents = buf.get()  # [B, win, c]
    expected_last_win = [entries[-win:]]  # [[win+1, win+2, ..., 2*win]]
    actual = contents[0, :, 0].tolist()  # first channel of each entry
    n_valid_ok = buf.n_valid == win
    contents_ok = actual == expected_last_win[0]
    return [_ok(
        'csa_decoding_cache_sw_ring_buffer',
        n_valid_ok and contents_ok,
        f'n_valid={buf.n_valid} (expected {win}), '
        f'contents={actual} (expected {expected_last_win[0]})')]


def test_csa_decoding_cache_indexer_dynamic_update(device='cpu'):
    """The indexer key cache grows in lockstep with the compressed block cache."""
    from ops_decoding_cache import CSADecodingCache
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 100, 4
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=5)
    T = 13  # 3 full blocks + 1 partial
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    expected_n_blocks = 0
    for t in range(T):
        H_t = H[:, t:t+1]
        Ca, Cb, Za, Zb, K_idx, Z_idx, _, _, _ = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])
        # After appending token t, the number of completed blocks should
        # be (t+1) // m.
        expected_n_blocks = (t + 1) // m
        if cache.n_blocks != expected_n_blocks:
            return [_ok(
                'csa_decoding_cache_indexer_dynamic_update',
                False,
                f'at t={t}: n_blocks={cache.n_blocks} '
                f'(expected {expected_n_blocks})')]
    # Final check: C_comp and K_IComp should have the same length.
    final_ok = (cache.C_comp is not None and cache.K_IComp is not None
                and cache.C_comp.shape[1] == cache.K_IComp.shape[1]
                == cache.n_blocks)
    return [_ok(
        'csa_decoding_cache_indexer_dynamic_update',
        final_ok,
        f'final n_blocks={cache.n_blocks}, '
        f'C_comp.shape={tuple(cache.C_comp.shape) if cache.C_comp is not None else None}, '
        f'K_IComp.shape={tuple(cache.K_IComp.shape) if cache.K_IComp is not None else None}')]


def test_csa_decoding_cache_prefill_then_decode(device='cpu'):
    """Prefill (T_new > 1) then decode (T_new == 1) produces correct shapes."""
    from ops_decoding_cache import CSADecodingCache
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 100, 4
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=6)
    prefill_len = 16
    n_decode = 5
    B = 1
    H_pre = torch.randn(B, prefill_len, d, device=device) * 0.1

    # Build the cache by feeding the prefill token-by-token (the
    # ``append_step`` path that CSAAttnDecoding uses during prefill).
    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    for t in range(prefill_len):
        H_t = H_pre[:, t:t+1]
        Ca, Cb, Za, Zb, K_idx, Z_idx, _, _, _ = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])

    prefill_n_blocks_ok = cache.n_blocks == prefill_len // m
    prefill_acc_ok = cache.accumulator_len == prefill_len % m

    # Decode: feed one token at a time and verify the cache state evolves.
    decode_shapes_ok = True
    for t in range(n_decode):
        H_t = torch.randn(B, 1, d, device=device) * 0.1
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, q_idx, w_idx,
            topk=topk, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=None, use_ste=False,
        )
        if o_t.shape != (B, 1, nh, c):
            decode_shapes_ok = False
            break

    final_n_blocks = cache.n_blocks
    expected_final_n_blocks = (prefill_len + n_decode) // m
    final_n_blocks_ok = final_n_blocks == expected_final_n_blocks

    return [_ok(
        'csa_decoding_cache_prefill_then_decode',
        prefill_n_blocks_ok and prefill_acc_ok and decode_shapes_ok
        and final_n_blocks_ok,
        f'prefill_n_blocks={cache.n_blocks if not decode_shapes_ok else "ok"} '
        f'(prefill_blocks_ok={prefill_n_blocks_ok}, acc_ok={prefill_acc_ok}, '
        f'decode_shapes_ok={decode_shapes_ok}, '
        f'final_n_blocks={final_n_blocks} expected={expected_final_n_blocks})')]


def test_hca_decoding_cache_compressed_blocks_match(device='cpu'):
    """The HCA compressed block cache stores EXACTLY naive_hca's C_comp."""
    from ops_decoding_cache import HCADecodingCache
    from ops_csa import csa_compress_kv
    torch.manual_seed(42)
    d, c, m2, nh, dc = 32, 8, 4, 2, 8
    win = 4
    p = _make_hca_decoding_params(d, c, m2, nh, dc, device, seed=7)
    T = 16  # divisible by m2
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    C_f, Z_f, _ = _project_hca_for_cache(H, p, c, dc, nh)
    C_comp_full = csa_compress_kv(C_f, Z_f, p['B_pos'], m2)

    cache = HCADecodingCache(B, c, m2, win, device, torch.float32)
    for t in range(T):
        H_t = H[:, t:t+1]
        C, Z, _ = _project_hca_for_cache(H_t, p, c, dc, nh)
        cache.append_step(C, Z, p['B_pos'])

    c_diff = (C_comp_full - cache.C_comp).abs().max().item()
    n_blocks_match = cache.n_blocks == T // m2
    return [_ok(
        'hca_decoding_cache_compressed_blocks_match',
        c_diff < 1e-5 and n_blocks_match,
        f'C_comp_diff={c_diff:.6e}, n_blocks_inc={cache.n_blocks} '
        f'(expected {T // m2})')]


def test_decoding_cache_reset_clears_state(device='cpu'):
    """reset() clears all cache state (blocks, accumulator, SW buffer)."""
    from ops_decoding_cache import CSADecodingCache
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 100, 4
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=8)
    B = 1
    H = torch.randn(B, 10, d, device=device) * 0.1

    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    for t in range(10):
        H_t = H[:, t:t+1]
        Ca, Cb, Za, Zb, K_idx, Z_idx, _, _, _ = _project_csa_for_cache(
            H_t, p, c, c_I, dc, nh, nIh)
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])

    # State should be non-empty.
    pre_n_blocks = cache.n_blocks
    pre_acc = cache.accumulator_len
    pre_sw = cache.sw_buffer.n_valid if cache.sw_buffer else 0
    pre_reset_nonempty = (pre_n_blocks > 0 or pre_acc > 0 or pre_sw > 0)
    cache.reset()
    post_reset_empty = (cache.n_blocks == 0 and cache.accumulator_len == 0
                        and (cache.sw_buffer is None
                             or cache.sw_buffer.n_valid == 0)
                        and cache.C_comp is None and cache.K_IComp is None)
    return [_ok(
        'decoding_cache_reset_clears_state',
        pre_reset_nonempty and post_reset_empty,
        f'pre_reset: n_blocks={pre_n_blocks}, acc={pre_acc}, '
        f'sw_valid={pre_sw}; '
        f'post_reset: n_blocks={cache.n_blocks}, acc={cache.accumulator_len}, '
        f'sw_valid={cache.sw_buffer.n_valid if cache.sw_buffer else 0}')]


def test_hybrid_decode_bf16_cache_precision(device='cpu'):
    """bf16 token-by-token decode must match a one-shot bf16 prefill.

    The persistent KDA / Mamba-2 / RWKV-7 recurrent state already lives in
    compute precision (fp32 for bf16 inputs). The CSA/HCA incremental
    caches historically stored their compressed blocks at the INPUT dtype,
    which added one extra bf16 rounding relative to the full-sequence
    prefill. That rounding can move near-tied lightning-indexer scores
    across a tie boundary, changing the hard top-k selection and making
    the decode suffix differ from prefill by many bf16 ulps on some seeds
    (observed: ~11 ulp on seed 1 of the 3:1:1 hybrid, while other seeds
    were within 1 ulp).

    This is a MULTI-SEED certificate, not a one-draw smoke test: a single
    lucky seed would have passed the old bf16-cache path. All three memory
    routes share the same CSA/HCA cache code, so each route is checked
    across seeds. The budget is two bf16 ulps of the output scale (the
    incremental and one-shot paths use different softmax reduction
    orders; with compute-precision caches the observed diff is exactly
    zero, so 2 ulp leaves headroom for platform-specific reduction ULP
    without admitting the old requantization error).
    """
    import copy
    from run_decoding import HybridDecoding

    torch.manual_seed(0)
    d_model, total_layers = 32, 5
    T, prefill = 64, 32
    seeds = (0, 1, 2)
    results = []
    for route in ('kda', 'mamba2', 'rwkv7'):
        for seed in seeds:
            torch.manual_seed(seed)
            ref = HybridDecoding(d_model=d_model, total_layers=total_layers,
                                 memory_type=route).to(device).eval()
            low = copy.deepcopy(ref).to(dtype=torch.bfloat16).eval()
            x = torch.randn(1, T, d_model, device=device)
            with torch.no_grad():
                full = low(x.bfloat16())
                low.reset()
                outs = [low(x[:, :prefill].bfloat16())]
                for t in range(prefill, T):
                    outs.append(low(x[:, t:t + 1].bfloat16()))
                dec = torch.cat(outs, dim=1)
            diff = (dec.float() - full.float()).abs().max().item()
            scale = full.float().abs().max().item()
            tol = 2.0 * float(torch.finfo(torch.bfloat16).eps) * max(1.0, scale)
            caches = [c for c in (*low._csa_caches, *low._hca_caches)
                      if c is not None]
            cache_fp32 = bool(caches) and all(
                c.dtype == torch.float32 for c in caches)
            state = low.model._kda_state
            state_fp32 = state is None or state.dtype == torch.float32
            results.append(_ok(
                f'{route} bf16 decode == prefill (seed={seed})',
                diff <= tol and cache_fp32 and state_fp32,
                f'max_diff={diff:.3e} tol={tol:.3e} scale={scale:.3e} '
                f'cache_dtypes={[str(c.dtype) for c in caches]} '
                f'state_dtype={None if state is None else state.dtype}'))

    # The standalone CSA/HCA benchmark wrappers share the same helper; pin
    # their cache storage dtype directly so a future refactor cannot
    # silently reintroduce the bf16 requantization only in those paths.
    from run_decoding import CSAAttnDecoding, HCAAttnDecoding
    for name, cls in (('csa', CSAAttnDecoding), ('hca', HCAAttnDecoding)):
        mod = cls(d_model).to(device).to(torch.bfloat16).eval()
        with torch.no_grad():
            mod(torch.randn(1, 8, d_model, device=device).bfloat16())
            mod(torch.randn(1, 1, d_model, device=device).bfloat16())
        results.append(_ok(
            f'standalone {name} bf16 cache stores fp32',
            mod._cache is not None and mod._cache.dtype == torch.float32,
            f'cache_dtype={None if mod._cache is None else mod._cache.dtype}'))
    return results


__all__ = [
    'test_decoding_batch_size_change',
    'test_csa_decoding_cache_correctness',
    'test_hca_decoding_cache_correctness',
    'test_csa_decoding_cache_compressed_blocks_match',
    'test_csa_decoding_cache_sliding_window_ring_buffer',
    'test_csa_decoding_cache_indexer_dynamic_update',
    'test_csa_decoding_cache_prefill_then_decode',
    'test_hca_decoding_cache_compressed_blocks_match',
    'test_decoding_cache_reset_clears_state',
    'test_hybrid_decode_bf16_cache_precision',
]
