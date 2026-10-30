"""Correctness checks: csa."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_csa_causality(device='cpu'):
    logger.info("Test: CSA compression + indexer causality")
    torch.manual_seed(2)
    B, T, m, topk = 1, 64, 8, 4
    n_blocks = T // m
    c = 16
    Ca = torch.randn(B, T, c, device=device)
    Cb = torch.randn(B, T, c, device=device)
    Za = torch.randn(B, T, c, device=device)
    Zb = torch.randn(B, T, c, device=device)
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
    cbm = _causal_block_mask(T, n_blocks, m, device)

    HI, DI = 2, 8
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    w_idx = torch.randn(B, T, HI, device=device)
    indices = csa_lightning_indexer(q_idx, k_idx, w_idx, topk, causal_block_mask=cbm)

    # Each query's selected blocks must be strictly preceding (cbm True).
    valid = True
    for t in range(T):
        sel = indices[0, t]
        sel = sel[sel >= 0]
        if not cbm[t, sel].all():
            valid = False
            break
    return [
        _ok('CSA compressed shape', C_comp.shape == (B, n_blocks, c), str(tuple(C_comp.shape))),
        _ok('CSA indices shape', indices.shape == (B, T, topk), str(tuple(indices.shape))),
        _ok('CSA causal selection', valid, 'all selected blocks precede the query'),
    ]


def test_overlap_causality(device='cpu'):
    """Verify CSA two-branch overlapped compression never leaks future tokens.

    The overlapped scheme fuses ``m`` entries from ``Ca`` (current block i)
    with ``m`` entries from ``Cb`` (previous block i-1). A correct
    implementation must ensure compressed block ``i`` depends only on source
    tokens ``[i*m - m, i*m + m)`` — i.e. block ``i`` and block ``i-1``, never
    block ``i+1`` or later.

    We test this by zeroing every token outside the allowed window and checking
    that the compressed output for block ``i`` is unchanged.
    """
    logger.info("Test: CSA overlapped compression causality (no future leakage)")
    torch.manual_seed(5)
    B, T, m, c = 1, 64, 8, 16
    n_blocks = T // m
    Ca = torch.randn(B, T, c, dtype=torch.float64, device=device)
    Cb = torch.randn(B, T, c, dtype=torch.float64, device=device)
    Za = torch.randn(B, T, c, dtype=torch.float64, device=device)
    Zb = torch.randn(B, T, c, dtype=torch.float64, device=device)
    Ba = torch.randn(m, c, dtype=torch.float64, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=torch.float64, device=device) * 0.1

    # Reference: full computation.
    ref = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)

    # For each compressed block i, verify only source tokens [max(0,(i-1)*m),
    # (i+1)*m) affect it. We do this by perturbing every token *after* the
    # allowed window and confirming the compressed block i is unchanged.
    max_diff = 0.0
    for i in range(n_blocks):
        allowed_lo = max(0, (i - 1) * m)
        allowed_hi = (i + 1) * m
        # Perturb tokens outside [allowed_lo, allowed_hi).
        Ca_p = Ca.clone()
        Cb_p = Cb.clone()
        Za_p = Za.clone()
        Zb_p = Zb.clone()
        # Add a large perturbation to all forbidden positions.
        for pos in range(T):
            if not (allowed_lo <= pos < allowed_hi):
                Ca_p[:, pos] += 10.0 * torch.randn_like(Ca[:, pos])
                Cb_p[:, pos] += 10.0 * torch.randn_like(Cb[:, pos])
                Za_p[:, pos] += 10.0 * torch.randn_like(Za[:, pos])
                Zb_p[:, pos] += 10.0 * torch.randn_like(Zb[:, pos])
        perturbed = csa_compress_kv_overlapped(Ca_p, Cb_p, Za_p, Zb_p, Ba, Bb, m)
        diff = (ref[:, i] - perturbed[:, i]).abs().max().item()
        max_diff = max(max_diff, diff)

    # Also verify block 0 has no "previous" dependency (Cb[:, -1] is never
    # read for block 0). The implementation pads with -inf for the b-branch.
    Cb_shifted = Cb.clone()
    Cb_shifted[:, :m] += 100.0  # perturb the would-be "previous" of block 0
    out_shifted = csa_compress_kv_overlapped(Ca, Cb_shifted, Za, Zb, Ba, Bb, m)
    block0_diff = (ref[:, 0] - out_shifted[:, 0]).abs().max().item()

    return [
        _ok('overlap no-future-leakage', max_diff < 1e-9,
            f'max diff over all blocks = {max_diff:.2e} (fp64)'),
        _ok('block 0 no-prev-dependency', block0_diff < 1e-9,
            f'block 0 diff when perturbing prev = {block0_diff:.2e}'),
    ]


def test_csa_indexer_validity(device='cpu'):
    """Check CSA top-k indices are in range and the count is correct."""
    logger.info("Test: CSA indexer top-k validity")
    torch.manual_seed(7)
    B, T, m, topk = 1, 64, 8, 4
    n_blocks = T // m
    HI, DI = 2, 8
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    w_idx = torch.randn(B, T, HI, device=device)
    cbm = _causal_block_mask(T, n_blocks, m, device)
    indices = csa_lightning_indexer(q_idx, k_idx, w_idx, topk, causal_block_mask=cbm)

    # All non-negative indices must be < n_blocks.
    valid_idx = indices >= 0
    in_range = (indices[valid_idx] < n_blocks).all().item()
    # Every selected block must be causal (strictly preceding the query).
    # Vectorized: clamp -1 to 0, gather cbm[t, idx] for every (b, t, k),
    # then OR with the (indices < 0) mask so invalid slots don't trip the
    # check. Replaces the per-t Python loop with a single batched gather.
    idx_safe = indices.clamp(min=0)                              # [B, T, topk]
    t_grid = torch.arange(T, device=device).view(T, 1).expand(T, idx_safe.shape[-1])
    causal_per_slot = cbm[t_grid, idx_safe[0]] | (indices[0] < 0)  # [T, topk]
    causal_ok = causal_per_slot.all().item()
    # For queries in the FIRST block (t < m), floor(t/m) = 0, so no
    # strictly-preceding block exists and all indices should be -1.
    # At t=m (first token of block 1) block 0 becomes visible.
    early_ok = (indices[0, :m] == -1).all().item()
    # Count check: every query t should have exactly min(topk, (t + 1) // m)
    # valid (non -1) indices. The docstring promises this check but the
    # original implementation never performed it; a bug where the indexer
    # returned e.g. only 2 of 4 valid indices (or 6 padded -1s with 2
    # valid) would have passed silently. We verify the count for EVERY
    # query, not just a sample, so an off-by-one or wrong-padding bug at
    # any t is caught.
    expected_counts = torch.tensor(
        [min(topk, t // m) for t in range(T)], device=device)
    actual_counts = (indices[0] >= 0).sum(-1)                   # [T]
    count_ok = (actual_counts == expected_counts).all().item()

    return [
        _ok('CSA indices in range', in_range, f'topk={topk}, n_blocks={n_blocks}'),
        _ok('CSA indices causal', causal_ok, 'all selected blocks precede query'),
        _ok('CSA early queries empty', early_ok,
            f'queries t<{m} have no preceding block -> all -1'),
        _ok('CSA index count per query', count_ok,
            f'each query t has min(topk={topk}, t//m) valid indices'),
    ]


def test_causal_block_mask_window_close(device='cpu'):
    """Boundary regression for the DeepSeek-V4 strictly-preceding mask.

    Paper-exact contract (Eq. 16 + §2.3.3): a query attends only to
    STRICTLY PRECEDING compressed blocks — ``s < floor(t / m)``. Block 0
    (covering positions 0..m-1) is therefore hidden from every query in
    its own block (t = 0..m-1) and first becomes visible at t = m, the
    first token of the NEXT block. The historical window-close variant
    ``b < (t + 1) // m`` wrongly exposed the block at t = m-1 (its own
    last source token) — causally safe but off-by-one against the paper.
    """
    logger.info("Test: strictly-preceding block-causal mask boundary")
    T, m = 2 * 8 + 1, 8
    mask = _causal_block_mask(T, T // m, m, device)
    expected_counts = torch.tensor(
        [t // m for t in range(T)], device=device,
    )
    actual_counts = mask.sum(dim=-1)
    boundary_ok = bool(torch.equal(actual_counts, expected_counts))
    # Block 0 must be hidden at t = m-1 (its own last source token) and
    # visible at t = m (the first token of the next block).
    hidden_before_ok = not bool(mask[m - 1, 0].item()) if mask.shape[1] > 0 else False
    visible_at_ok = bool(mask[m, 0].item()) if mask.shape[1] > 0 else False
    return [
        _ok("strictly-preceding mask counts", boundary_ok,
            f"expected={expected_counts.tolist()}, actual={actual_counts.tolist()}"),
        _ok("first block hidden at t=m-1, visible at t=m",
            hidden_before_ok and visible_at_ok,
            "block 0 serves only queries from the NEXT block onward "
            "(paper Eq. 16: s < floor(t/m))"),
    ]


def test_csa_indexer_w_idx_none(device='cpu'):
    """P0 regression: ``csa_lightning_indexer`` must handle ``w_idx=None``.

    The ``w_idx is None`` branch (``logits = score.sum(1)``) was
    historically never exercised by ``run_quality``'s ``CSAAttn``
    (which always passes ``W_w``), so a regression that broke it
    (e.g. a wrong axis on the sum, a dtype mismatch, or a missing
    causal-mask application) could ship silently. This test calls
    ``csa_lightning_indexer`` directly with ``w_idx=None`` and
    verifies:
      1. The output shape is correct (``[B, T, topk]``).
      2. Indices are in range and causal (mirror the
         ``test_csa_indexer_validity`` checks).
      3. The ``w_idx=None`` path produces the SAME top-k ranking as
         passing ``w_idx = ones`` (since ``score.sum(1)`` is the same
         as ``score * 1`` summed). This pins the algebraic identity
         between the two branches.
      4. The STE path also works with ``w_idx=None`` (returns
         differentiable ``soft_weights``).
    """
    logger.info("Test: CSA indexer with w_idx=None (no head-mixing weights)")
    torch.manual_seed(13)
    B, T, m, topk = 1, 64, 8, 4
    n_blocks = T // m
    HI, DI = 2, 8
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    cbm = _causal_block_mask(T, n_blocks, m, device)

    # --- Branch 1: w_idx=None (the path under test) ---
    indices_none = csa_lightning_indexer(
        q_idx, k_idx, None, topk, causal_block_mask=cbm)
    shape_ok = indices_none.shape == (B, T, topk)
    in_range = (indices_none[indices_none >= 0] < n_blocks).all().item()
    idx_safe = indices_none.clamp(min=0)
    t_grid = torch.arange(T, device=device).view(T, 1).expand(T, topk)
    causal_per_slot = cbm[t_grid, idx_safe[0]] | (indices_none[0] < 0)
    causal_ok = causal_per_slot.all().item()

    # --- Branch 2: w_idx=ones (should give the SAME ranking) ---
    # ``score`` is [B, HI, T, n_blocks] after the einsum.
    # With ``w_idx`` of shape [B, T, HI]:
    #   logits_w = einsum('bhtn, bth -> btn', score, w_idx)
    # With w_idx = ones(B, T, HI), this is exactly score.sum(1) over HI,
    # i.e. identical to the ``w_idx is None`` branch.
    w_idx_ones = torch.ones(B, T, HI, device=device)
    indices_ones = csa_lightning_indexer(
        q_idx, k_idx, w_idx_ones, topk, causal_block_mask=cbm)
    # Use exact equality — the two code paths produce bit-identical
    # floating-point logits (same einsum reduction over HI), so the
    # top-k ranking must be identical. A non-equal result indicates
    # the two branches diverged (e.g. a wrong axis on the sum).
    ranking_match = torch.equal(indices_none, indices_ones)

    # --- Branch 3: w_idx=None + return_soft_weights=True (STE path) ---
    # Verify the STE branch still works when ``w_idx`` is None. The
    # soft_weights returned must have the full [B, T, n_blocks] shape
    # and be differentiable (the gradient path back to q_idx/k_idx
    # does not depend on w_idx).
    q_idx_g = q_idx.clone().requires_grad_(True)
    k_idx_g = k_idx.clone().requires_grad_(True)
    indices_ste, soft_weights = csa_lightning_indexer(
        q_idx_g, k_idx_g, None, topk,
        causal_block_mask=cbm, return_soft_weights=True)
    soft_shape_ok = soft_weights.shape == (B, T, n_blocks)
    soft_finite = torch.isfinite(soft_weights).all().item()
    # Backward through soft_weights (sum-reduce) and verify q_idx/k_idx
    # receive finite gradient. The hard indices tensor has no grad_fn
    # (it is integer), so the gradient flows ONLY through soft_weights.
    soft_weights.sum().backward()
    q_grad_ok = (q_idx_g.grad is not None
                 and torch.isfinite(q_idx_g.grad).all().item()
                 and q_idx_g.grad.abs().sum().item() > 0)
    k_grad_ok = (k_idx_g.grad is not None
                 and torch.isfinite(k_idx_g.grad).all().item()
                 and k_idx_g.grad.abs().sum().item() > 0)

    return [
        _ok('CSA w_idx=None output shape', shape_ok,
            f'expected {(B, T, topk)}, got {tuple(indices_none.shape)}'),
        _ok('CSA w_idx=None indices in range', in_range,
            f'topk={topk}, n_blocks={n_blocks}'),
        _ok('CSA w_idx=None indices causal', causal_ok,
            'all selected blocks precede query'),
        _ok('CSA w_idx=None ranking == w_idx=ones ranking',
            ranking_match,
            'score.sum(1) == einsum(score, ones) -> identical top-k'),
        _ok('CSA w_idx=None STE soft_weights shape', soft_shape_ok,
            f'expected {(B, T, n_blocks)}, got {tuple(soft_weights.shape)}'),
        _ok('CSA w_idx=None STE soft_weights finite', soft_finite,
            'no NaN/Inf in the differentiable soft distribution'),
        _ok('CSA w_idx=None STE q_idx receives finite non-zero grad',
            q_grad_ok, f'q_idx.grad.sum={q_idx_g.grad.abs().sum().item() if q_idx_g.grad is not None else None}'),
        _ok('CSA w_idx=None STE k_idx receives finite non-zero grad',
            k_grad_ok, f'k_idx.grad.sum={k_idx_g.grad.abs().sum().item() if k_idx_g.grad is not None else None}'),
    ]


def test_csa_indexer_topk_zero(device='cpu'):
    """P0 regression: ``csa_lightning_indexer`` must handle ``topk=0`` directly.

    The ``topk=0`` guard historically lived only in ``naive_csa``; a
    direct caller of the public ``csa_lightning_indexer`` would hit a
    cryptic ``RuntimeError`` from ``torch.topk``. The P0 fix pushed
    the guard down into the indexer itself. This test verifies:
      1. Direct call with ``topk=0`` returns shape ``[B, T, 0]`` (no
         crash).
      2. With ``return_soft_weights=True``, the STE branch still
         returns the full ``[B, T, n_blocks]`` differentiable
         distribution (useful for an auxiliary loss even when no
         hard selection is made).
    """
    logger.info("Test: CSA indexer with topk=0 (direct call, P0 guard)")
    torch.manual_seed(14)
    B, T, m = 1, 32, 8
    n_blocks = T // m
    HI, DI = 2, 8
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    cbm = _causal_block_mask(T, n_blocks, m, device)

    # topk=0, no STE.
    indices = csa_lightning_indexer(q_idx, k_idx, None, topk=0,
                                    causal_block_mask=cbm)
    shape_ok = indices.shape == (B, T, 0)

    # topk=0, with STE.
    indices_ste, soft_weights = csa_lightning_indexer(
        q_idx, k_idx, None, topk=0,
        causal_block_mask=cbm, return_soft_weights=True)
    shape_ste_ok = indices_ste.shape == (B, T, 0)
    soft_shape_ok = soft_weights.shape == (B, T, n_blocks)
    soft_finite = torch.isfinite(soft_weights).all().item()

    return [
        _ok('CSA indexer topk=0 returns empty indices', shape_ok,
            f'expected {(B, T, 0)}, got {tuple(indices.shape)}'),
        _ok('CSA indexer topk=0 + STE returns empty indices', shape_ste_ok,
            f'expected {(B, T, 0)}, got {tuple(indices_ste.shape)}'),
        _ok('CSA indexer topk=0 + STE soft_weights shape', soft_shape_ok,
            f'expected {(B, T, n_blocks)}, got {tuple(soft_weights.shape)}'),
        _ok('CSA indexer topk=0 + STE soft_weights finite', soft_finite,
            'no NaN/Inf in the differentiable soft distribution'),
    ]


def test_csa_indexer_ste_gradient(device='cpu'):
    """P0-4 regression: STE must make the indexer parameters trainable.

    Before the P0-4 fix, ``torch.topk`` returned integer indices that did
    not propagate gradients, so the indexer parameters (``W_IUQ``,
    ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx``) stayed at random
    initialization after ``backward()`` (their ``.grad`` was ``None``).
    This test verifies that the straight-through estimator (STE) in
    ``naive_csa`` closes that gap: after a forward + backward pass with
    ``use_ste=True`` (the default), every indexer parameter has a
    non-None, finite ``.grad``.

    We also verify the STE's forward-value contract: the output of
    ``naive_csa(use_ste=True)`` must be BIT-IDENTICAL to
    ``naive_csa(use_ste=False)`` (the STE only changes the backward
    path, not the forward value). If a future refactor breaks this
    invariant, the STE is silently changing the algorithm's forward
    semantics — which would invalidate every existing CSA result.
    """
    logger.info("Test: CSA indexer STE gradient flow (P0-4 fix)")
    torch.manual_seed(42)
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, cI, dc = 4, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device, dtype=torch.float64)

    def _make_params():
        return dict(
            W_aKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_aZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            Ba=torch.randn(m, c, dtype=torch.float64) * 0.02,
            Bb=torch.randn(m, c, dtype=torch.float64) * 0.02,
            W_DQ=torch.randn(dc, d, dtype=torch.float64) * 0.1,
            W_UQ=torch.randn(c * nh, dc, dtype=torch.float64) * 0.1,
            W_IUQ=torch.randn(cI * nIh, dc, dtype=torch.float64) * 0.1,
            W_w=torch.randn(nIh, d, dtype=torch.float64) * 0.1,
            W_KV_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            W_Z_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            B_idx=torch.randn(m, cI, dtype=torch.float64) * 0.02,
        )

    # --- Part 1: STE forward-value invariance ---
    # The STE must NOT change the forward output. Run the same input
    # through ``use_ste=True`` and ``use_ste=False`` and assert the
    # outputs are equal to fp64 precision. We use ``torch.allclose``
    # (rtol=0, atol=0 is too strict because the STE adds and subtracts
    # ``soft_kv``, which introduces ULP-level rounding from the extra
    # float ops even though the result is mathematically identical).
    # ``atol=1e-12`` is well below any meaningful accuracy threshold
    # for fp64 (machine epsilon ~2.2e-16) while still tolerating the
    # rounding from the extra add/subtract in the STE path.
    #
    # IMPORTANT: this comparison must run with GRAD ENABLED (i.e. NOT
    # inside ``torch.no_grad()``). ``naive_csa`` gates the STE on
    # ``use_ste and torch.is_grad_enabled()`` (see ops_csa.py): under
    # ``no_grad`` both calls take the same lean hard-index path, so the
    # assertion would compare the code path to itself and trivially pass
    # even if the STE silently changed the forward value whenever grad is
    # enabled. With grad enabled, ``use_ste=True`` executes the STE
    # surrogate (``kv = kv + (soft_kv - soft_kv.detach())``) whose forward
    # value is mathematically identical to the hard gather.
    import copy
    p_ref = _make_params()
    p_ste = copy.deepcopy(p_ref)
    common = dict(m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=cI, dc=dc,
                  sliding_window=0, sink_logits=None)
    o_ref = naive_csa(H, **p_ref, use_ste=False, **common)
    o_ste = naive_csa(H, **p_ste, use_ste=True, **common)
    fwd_invariant = torch.allclose(o_ref, o_ste, rtol=0, atol=1e-12)

    # --- Part 2: indexer parameters receive gradient under STE ---
    # Make indexer params require grad, run forward + backward, and
    # assert each has a non-None, finite .grad.
    p = _make_params()
    indexer_param_names = ['W_IUQ', 'W_w', 'W_KV_idx', 'W_Z_idx', 'B_idx']
    for name in indexer_param_names:
        p[name] = p[name].clone().requires_grad_(True)
    o = naive_csa(H, **p, use_ste=True, **common)
    # Sum-reduce to a scalar and backward.
    o.sum().backward()
    grad_results = {}
    for name in indexer_param_names:
        g = p[name].grad
        grad_results[name] = (
            g is not None and torch.isfinite(g).all().item()
            and g.abs().sum().item() > 0
        )

    # --- Part 3: without STE, indexer params get NO useful gradient ---
    # This pins the substantive old (buggy) behavior so a future "fix" that
    # removes STE is forced to also update this test, rather than silently
    # reverting to the untrained-indexer regime.
    #
    # Issue 2.3 fix note: the F.linear fusion in ``naive_csa`` merges
    # ``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx`` into a single matmul.
    # Under ``use_ste=False`` the indexer's top-k selection still doesn't
    # propagate gradient, but ``W_KV_idx`` and ``W_Z_idx`` now receive a
    # ZERO (but non-None) grad because autograd flows backward through the
    # merged matmul into all 6 weight slices. The substantive contract —
    # "indexer params don't learn without STE" — is preserved: a zero grad
    # means the optimizer step is a no-op. We therefore check for "no
    # useful gradient" (None OR all-zero) instead of requiring None
    # specifically. ``W_IUQ``, ``W_w`` and ``B_idx`` are NOT in the merged
    # matmul, so they still receive ``None`` grad under ``use_ste=False``.
    #
    # We must make a NON-indexer parameter (W_aKV) require grad too,
    # otherwise the output has no grad_fn at all (under use_ste=False
    # the indexer params don't participate in the differentiable graph,
    # so if W_aKV also doesn't require grad, ``backward()`` raises
    # "element 0 of tensors does not require grad"). With W_aKV
    # requiring grad, the output has a grad_fn, backward() succeeds,
    # and we can assert the indexer params STILL get no useful grad
    # (the bug we're pinning).
    p_noste = _make_params()
    for name in indexer_param_names:
        p_noste[name] = p_noste[name].clone().requires_grad_(True)
    # W_aKV is a non-indexer param that DOES participate in the
    # differentiable compression+attention path. Make it require grad
    # so the output has a grad_fn.
    p_noste['W_aKV'] = p_noste['W_aKV'].clone().requires_grad_(True)
    o_noste = naive_csa(H, **p_noste, use_ste=False, **common)
    o_noste.sum().backward()
    # Under use_ste=False, the indexer params get NO useful gradient
    # (None for W_IUQ/W_w/B_idx which aren't in the merged matmul;
    # all-zero for W_KV_idx/W_Z_idx which ARE in the merged matmul and
    # therefore receive a zero slice of the merged weight's grad).
    # Either way, the optimizer step is a no-op — the indexer doesn't
    # learn without STE. W_aKV (non-indexer) DOES get a non-trivial
    # gradient, confirming the backward pass actually ran.
    def _no_useful_grad(g):
        if g is None:
            return True
        return bool(g.abs().sum().item() < 1e-12)
    noste_grads_useless = all(_no_useful_grad(p_noste[name].grad)
                              for name in indexer_param_names)
    noste_wakv_has_grad = (p_noste['W_aKV'].grad is not None
                            and p_noste['W_aKV'].grad.abs().sum().item() > 0)

    return [
        _ok('STE forward == no-STE forward (bit-identical)',
            fwd_invariant,
            f'max_abs_diff={(o_ref - o_ste).abs().max().item() if not fwd_invariant else 0}'),
        _ok('STE: W_IUQ receives finite non-zero gradient',
            grad_results['W_IUQ'], f'grad={p["W_IUQ"].grad}'),
        _ok('STE: W_w receives finite non-zero gradient',
            grad_results['W_w'], f'grad={p["W_w"].grad}'),
        _ok('STE: W_KV_idx receives finite non-zero gradient',
            grad_results['W_KV_idx'], f'grad={p["W_KV_idx"].grad}'),
        _ok('STE: W_Z_idx receives finite non-zero gradient',
            grad_results['W_Z_idx'], f'grad={p["W_Z_idx"].grad}'),
        _ok('STE: B_idx receives finite non-zero gradient',
            grad_results['B_idx'], f'grad={p["B_idx"].grad}'),
        _ok('no-STE: indexer params get no useful grad (None or all-zero; pins old behavior)',
            noste_grads_useless,
            f'grads={[(_none_or_norm(p_noste[n].grad)) for n in indexer_param_names]}'),
        _ok('no-STE: W_aKV (non-indexer) DOES get non-trivial grad (backward ran)',
            noste_wakv_has_grad,
            f'W_aKV.grad_norm={(_none_or_norm(p_noste["W_aKV"].grad))}'),
    ]


def test_csa_fuse_projections_grad_contract(device='cpu'):
    """Review-fix 1.3 regression: ``fuse_projections`` forward/backward contract.

    Issue 2.3 fused the 6 KV/indexer-compression projections
    (``W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx``) into a single
    ``F.linear`` call for performance. This is forward-value-preserving,
    but under ``use_ste=False`` it changes ``W_KV_idx``/``W_Z_idx``'s
    ``.grad`` from ``None`` (untouched by autograd — the pre-fusion
    behaviour) to a zero-but-non-None tensor (the fused behaviour),
    because they now share a matmul with 4 other, differentiable weights.

    ``naive_csa`` now exposes ``fuse_projections=True`` (default,
    performance path) vs ``fuse_projections=False`` (restores the
    original 6-separate-matmul path with the ``None``-grad convention).
    This test pins:

      1. The forward output is bit-identical between the two modes
         (fp64 tolerance) — the fusion must never change the algorithm's
         result.
      2. Under ``use_ste=False`` + ``fuse_projections=True``,
         ``W_KV_idx``/``W_Z_idx`` receive a non-None, all-zero gradient
         (the "fused" contract).
      3. Under ``use_ste=False`` + ``fuse_projections=False``,
         ``W_KV_idx``/``W_Z_idx`` receive ``None`` gradient (the
         "unfused" contract, matching the pre-issue-2.3 behaviour).
      4. Under ``use_ste=True`` (STE enabled), BOTH modes give
         ``W_KV_idx``/``W_Z_idx`` a real (non-zero) gradient — the STE
         gradient path does not depend on ``fuse_projections`` at all.
    """
    logger.info("Test: naive_csa fuse_projections forward/backward contract (review-fix 1.3)")
    torch.manual_seed(43)
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, cI, dc = 4, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device, dtype=torch.float64)

    def _make_params():
        return dict(
            W_aKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_aZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            Ba=torch.randn(m, c, dtype=torch.float64) * 0.02,
            Bb=torch.randn(m, c, dtype=torch.float64) * 0.02,
            W_DQ=torch.randn(dc, d, dtype=torch.float64) * 0.1,
            W_UQ=torch.randn(c * nh, dc, dtype=torch.float64) * 0.1,
            W_IUQ=torch.randn(cI * nIh, dc, dtype=torch.float64) * 0.1,
            W_w=torch.randn(nIh, d, dtype=torch.float64) * 0.1,
            W_KV_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            W_Z_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            B_idx=torch.randn(m, cI, dtype=torch.float64) * 0.02,
        )

    common = dict(m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=cI, dc=dc,
                  sliding_window=0, sink_logits=None)

    # --- Part 1: forward value is identical regardless of fuse_projections ---
    import copy
    p_a = _make_params()
    p_b = copy.deepcopy(p_a)
    with torch.no_grad():
        o_fused = naive_csa(H, **p_a, use_ste=False, fuse_projections=True, **common)
        o_unfused = naive_csa(H, **p_b, use_ste=False, fuse_projections=False, **common)
    fwd_match = torch.allclose(o_fused, o_unfused, rtol=0, atol=1e-12)

    indexer_names = ['W_KV_idx', 'W_Z_idx']

    def _run_backward(fuse_projections, use_ste):
        p = _make_params()
        p['W_aKV'] = p['W_aKV'].clone().requires_grad_(True)
        for name in indexer_names:
            p[name] = p[name].clone().requires_grad_(True)
        o = naive_csa(H, **p, use_ste=use_ste, fuse_projections=fuse_projections, **common)
        o.sum().backward()
        return {name: p[name].grad for name in indexer_names}

    # --- Part 2: use_ste=False, fuse_projections=True -> zero, non-None grad ---
    grads_fused_noste = _run_backward(fuse_projections=True, use_ste=False)
    fused_noste_ok = all(
        g is not None and g.abs().sum().item() < 1e-12
        for g in grads_fused_noste.values())

    # --- Part 3: use_ste=False, fuse_projections=False -> None grad ---
    grads_unfused_noste = _run_backward(fuse_projections=False, use_ste=False)
    unfused_noste_ok = all(g is None for g in grads_unfused_noste.values())

    # --- Part 4: use_ste=True -> both modes give a REAL (non-zero) grad ---
    grads_fused_ste = _run_backward(fuse_projections=True, use_ste=True)
    grads_unfused_ste = _run_backward(fuse_projections=False, use_ste=True)
    fused_ste_ok = all(
        g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
        for g in grads_fused_ste.values())
    unfused_ste_ok = all(
        g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
        for g in grads_unfused_ste.values())

    return [
        _ok('fuse_projections=True/False: forward output bit-identical',
            fwd_match,
            f'max_abs_diff={(o_fused - o_unfused).abs().max().item() if not fwd_match else 0}'),
        _ok('use_ste=False + fuse_projections=True: indexer weights get zero-but-non-None grad',
            fused_noste_ok,
            f'grads={[(_none_or_norm(g)) for g in grads_fused_noste.values()]}'),
        _ok('use_ste=False + fuse_projections=False: indexer weights get None grad',
            unfused_noste_ok,
            f'grads={[(_none_or_norm(g)) for g in grads_unfused_noste.values()]}'),
        _ok('use_ste=True + fuse_projections=True: indexer weights get real gradient',
            fused_ste_ok,
            f'grads={[(_none_or_norm(g)) for g in grads_fused_ste.values()]}'),
        _ok('use_ste=True + fuse_projections=False: indexer weights get real gradient',
            unfused_ste_ok,
            f'grads={[(_none_or_norm(g)) for g in grads_unfused_ste.values()]}'),
    ]


def test_csa_indexer_ste_full_softmax(device='cpu'):
    """``ste_mode='full_softmax'`` must be distinct from
    ``'topk_columns'`` and from the no-STE path.

    1. **Forward equivalence**: ``full_softmax`` and ``topk_columns``
       produce bit-identical forward outputs (the STE only changes the
       backward path).

    2. **Backward distinctness**: under ``full_softmax``, the gradient
       on ``soft_weights`` (and therefore on the indexer parameters) is
       NON-ZERO for non-selected blocks; under ``topk_columns`` it is
       EXACTLY ZERO for non-selected blocks. We verify this by checking
       that the indexer parameter ``W_KV_idx`` receives a LARGER
       gradient norm under ``full_softmax`` than under ``topk_columns``
       (because more blocks contribute).

    3. **No-STE distinctness**: ``ste_mode`` has NO effect when
       ``use_ste=False`` — both modes produce bit-identical forward
       outputs and no gradient on indexer params. This is a sanity
       check that the ``ste_mode`` branch is correctly guarded by
       ``use_ste``.
    """
    logger.info("Test: CSA indexer STE full_softmax distinctness")
    torch.manual_seed(7)
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, cI, dc = 4, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device, dtype=torch.float64)

    def _make_params():
        return dict(
            W_aKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bKV=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_aZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            W_bZ=torch.randn(c, d, dtype=torch.float64) * 0.1,
            Ba=torch.randn(m, c, dtype=torch.float64) * 0.02,
            Bb=torch.randn(m, c, dtype=torch.float64) * 0.02,
            W_DQ=torch.randn(dc, d, dtype=torch.float64) * 0.1,
            W_UQ=torch.randn(c * nh, dc, dtype=torch.float64) * 0.1,
            W_IUQ=torch.randn(cI * nIh, dc, dtype=torch.float64) * 0.1,
            W_w=torch.randn(nIh, d, dtype=torch.float64) * 0.1,
            W_KV_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            W_Z_idx=torch.randn(cI, d, dtype=torch.float64) * 0.1,
            B_idx=torch.randn(m, cI, dtype=torch.float64) * 0.02,
        )

    common = dict(m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=cI, dc=dc,
                  sliding_window=0, sink_logits=None)

    # --- Part 1: forward equivalence between the two STE modes ---
    # Run with GRAD ENABLED (not ``torch.no_grad()``): ``naive_csa`` gates
    # the STE on ``use_ste and torch.is_grad_enabled()``, so under
    # ``no_grad`` neither ``ste_mode`` branch would execute and the
    # assertion would compare two identical hard-index paths (vacuously
    # true even if a STE mode corrupted the forward value).
    import copy
    p_ref = _make_params()
    p_fs = copy.deepcopy(p_ref)
    o_tk = naive_csa(H, **p_ref, use_ste=True,
                     ste_mode='topk_columns', **common)
    o_fs = naive_csa(H, **p_fs, use_ste=True,
                     ste_mode='full_softmax', **common)
    fwd_equiv = torch.allclose(o_tk, o_fs, rtol=0, atol=1e-12)
    fwd_diff = (o_tk - o_fs).abs().max().item() if not fwd_equiv else 0.0

    # --- Part 2: backward distinctness ---
    # Under ``full_softmax``, gradient flows to ALL blocks (including
    # non-selected ones), so the total gradient norm on the indexer
    # params is larger than under ``topk_columns`` (where only the
    # selected blocks contribute). We measure the L1 norm of the
    # gradient on ``W_KV_idx`` (which directly produces the indexer
    # keys) under both modes and assert full_softmax > topk_columns.
    #
    # We also verify both modes produce non-None gradients (they're
    # both STE paths).
    indexer_param_names = ['W_IUQ', 'W_w', 'W_KV_idx', 'W_Z_idx', 'B_idx']

    # Controlled comparison: BOTH modes must start from the SAME
    # parameter values so the gradient-norm difference isolates the
    # ``ste_mode`` effect. (Previously each mode drew a fresh
    # ``_make_params()``, so ``full_softmax`` vs ``topk_columns`` was
    # confounded by random initialization.)
    p_base = _make_params()
    p_tk = copy.deepcopy(p_base)
    p_fs = copy.deepcopy(p_base)
    for p in (p_tk, p_fs):
        for name in indexer_param_names:
            p[name] = p[name].clone().requires_grad_(True)

    def _run_backward(p, ste_mode):
        o = naive_csa(H, **p, use_ste=True, ste_mode=ste_mode, **common)
        o.sum().backward()
        return {name: p[name].grad for name in indexer_param_names}

    grads_tk = _run_backward(p_tk, 'topk_columns')
    grads_fs = _run_backward(p_fs, 'full_softmax')

    # All grads must be non-None and finite.
    tk_grads_finite = all(
        g is not None and torch.isfinite(g).all() for g in grads_tk.values())
    fs_grads_finite = all(
        g is not None and torch.isfinite(g).all() for g in grads_fs.values())

    # full_softmax must produce STRICTLY LARGER total L1 grad norm on
    # W_KV_idx than topk_columns. The "strictly larger" comes from the
    # additional gradient path through non-selected blocks. We use a
    # margin of 1e-12 to allow for fp64 rounding noise.
    tk_norm = grads_tk['W_KV_idx'].abs().sum().item()
    fs_norm = grads_fs['W_KV_idx'].abs().sum().item()
    fs_dense_signal = fs_norm > tk_norm + 1e-12

    # --- Part 3: no-STE invariance ---
    # ste_mode has no effect when use_ste=False. Both modes produce
    # identical forward outputs. Run with GRAD ENABLED and
    # ``use_ste=False`` so ``effective_use_ste`` is False: this verifies
    # the ``ste_mode`` branch is guarded by ``use_ste`` (not merely by
    # ``torch.is_grad_enabled()``, which would have made the old
    # ``no_grad`` version of this check vacuous).
    p_noste_a = _make_params()
    p_noste_b = copy.deepcopy(p_noste_a)
    o_noste_tk = naive_csa(H, **p_noste_a, use_ste=False,
                           ste_mode='topk_columns', **common)
    o_noste_fs = naive_csa(H, **p_noste_b, use_ste=False,
                           ste_mode='full_softmax', **common)
    noste_equiv = torch.allclose(o_noste_tk, o_noste_fs, rtol=0, atol=0)

    return [
        _ok('full_softmax forward == topk_columns forward (bit-identical)',
            fwd_equiv, f'max_abs_diff={fwd_diff}'),
        _ok('topk_columns: all indexer grads finite & non-None',
            tk_grads_finite,
            f'W_KV_idx.grad is None: {grads_tk["W_KV_idx"] is None}'),
        _ok('full_softmax: all indexer grads finite & non-None',
            fs_grads_finite,
            f'W_KV_idx.grad is None: {grads_fs["W_KV_idx"] is None}'),
        _ok('full_softmax has denser grad signal (||W_KV_idx.grad||_1 larger)',
            fs_dense_signal,
            f'topk_columns={tk_norm:.6e}, full_softmax={fs_norm:.6e}'),
        _ok('no-STE: ste_mode has no effect (forward bit-identical)',
            noste_equiv,
            f'diff={(o_noste_tk - o_noste_fs).abs().max().item() if not noste_equiv else 0}'),
    ]


def test_csa_full_pipeline_causality(device='cpu'):
    """Verify the full ``naive_csa`` pipeline (compression + indexer + sparse
    MQA core attention) is causal end-to-end.

    This complements ``test_csa_causality``, which only checks the compression
    and indexer stages *in isolation*.  Here we run the complete CSA forward
    and perturb a single source position ``p`` of ``H``, then verify that
    ``output[t]`` is unchanged for every ``t`` that should NOT depend on ``p``.

    For a perturbation at source position ``p`` (with ``p`` in compressed block
    ``bp = p // m``):
      * Sliding-window branch: ``t in [p, p+win-1]``  (``p`` is in ``t``'s
        window).
      * Sparse MQA branch: ``t >= (floor(p/m)+1)*m - 1``. Compressed block
        ``bp`` depends on ``p`` via the a-branch (``Ca[:, bp]`` reads position
        ``p``), and is selectable once its source window closes. Compressed
        block ``bp+1`` also depends on ``p`` via the b-branch, but becomes
        visible one full window later.
    Everywhere else -- including all ``t < p`` (strict future must not affect
    past) and the gap ``[p+win, (floor(p/m)+1)*m - 1)`` between the SW and sparse
    regions -- ``output[t]`` must be unchanged.  The gap check additionally
    verifies the SW window size is exactly ``win`` (a larger window would leak
    into the gap and be detected).

    Note on the lightning indexer: perturbing ``p`` also changes
    ``K_IComp`` (indexer keys) for blocks ``bp`` and ``bp+1``, and changes
    ``q_idx``/``w_idx`` at position ``p`` only.  A block is masked until its
    source window closes, so the first affected compressed block becomes
    selectable at ``t >= (bp+1)*m - 1``. The ``q_idx``/``w_idx`` perturbation only affects the
    selection (and thus output) at ``t == p``, which is already inside the SW
    affected region.
    """
    logger.info("Test: CSA full pipeline causality (compression + indexer + core)")
    torch.manual_seed(9)
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    win = 4
    dtype = torch.float64

    H = torch.randn(B, T, d, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(c, d, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(c, d, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(c, d, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(c, d, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(dc, d, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(c * nh, dc, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(c_I * nIh, dc, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(nIh, d, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(c_I, d, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(c_I, d, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    sink = torch.zeros(nh, dtype=dtype, device=device)

    # --- Reference forward (unperturbed) ---
    o_ref = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                      W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                      m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                      sliding_window=win, sink_logits=sink)  # [B, T, nh*c]

    # --- Batched single-position perturbation ---
    # Same transpose-trick as test_hca_sliding_window_causality: build T
    # copies of H, each perturbed at exactly one position p, and run a single
    # batched forward instead of T separate forwards.
    H_batch = H.unsqueeze(0).repeat(T, 1, 1, 1)              # [T, B, T, d]
    perturb = 10.0 * torch.randn(T, B, d, dtype=dtype, device=device)
    p_idx = torch.arange(T, device=device)
    H_batch[p_idx, :, p_idx, :] += perturb
    H_batch = H_batch.reshape(T * B, T, d)

    o_pert = naive_csa(H_batch, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                       W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                       m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                       sliding_window=win, sink_logits=sink)
    o_pert = o_pert.reshape(T, B, T, nh * c)                 # [p, B, t, nh*c]

    # diff[p, t] = max |o_ref[t] - o_pert[p, t]|
    diff = (o_ref[0].unsqueeze(0) - o_pert[:, 0]).abs().max(dim=-1).values  # [T, T]

    # Expected affected region for perturbing position p:
    #   SW branch:     p <= t < p + win
    #   Sparse branch: t >= (floor(p/m) + 1) * m - 1
    p_grid = torch.arange(T, device=device)[:, None]
    t_grid = torch.arange(T, device=device)[None, :]
    sw_affected = (t_grid >= p_grid) & (t_grid < p_grid + win)
    #     (paper Eq. 16: the block containing p is visible only to
    #      queries in strictly LATER blocks)
    sparse_affected = t_grid >= ((p_grid // m) + 1) * m
    expected_affected = sw_affected | sparse_affected

    # Outside the expected region, diff must be ~0.  This covers both
    # causality (t < p: future must not affect past) and the window-size gap
    # (p+win <= t < (floor(p/m)+1)*m: outside both branches).
    #
    # BIDIRECTIONAL assertion (mirrors test_hca_sliding_window_causality):
    # also verify that the SW AFFECTED region actually sees a non-trivial diff,
    # so a degenerate implementation that zeros out the SW branch cannot pass
    # the causality check trivially. We only check the SW region (NOT the
    # sparse region) because the sparse_affected mask is an *upper bound*:
    # block ``bp`` is selectable by queries ``t >= (bp+1)*m``, but it might
    # not be in the top-k for a given query — so a zero diff at some
    # (p, t) in sparse_affected is legitimate (block not selected), not a bug.
    if (~expected_affected).any():
        max_diff = diff[~expected_affected].max().item()
    else:
        max_diff = 0.0
    if sw_affected.any():
        min_sw_affected = diff[sw_affected].min().item()
    else:
        min_sw_affected = float('inf')

    return [
        _ok('CSA full pipeline causal (unaffected region is stable)',
            max_diff < 1e-9,
            f'max diff in unaffected region = {max_diff:.2e} (fp64, win={win})'),
        _ok('CSA full pipeline SW-affected region actually changes',
            min_sw_affected > 1e-6,
            f'min diff in SW-affected region = {min_sw_affected:.2e} '
            f'(guards against zero-SW-output regressions; sparse region '
            f'is not checked because top-k may legitimately not select '
            f'block bp)'),
    ]


def test_csa_topk_edge_cases(device='cpu'):
    """Verify CSA handles topk == n_blocks and topk > n_blocks.

    When topk >= n_blocks, every selected index is valid (no -1 padding
    needed). When topk > n_blocks, the lightning indexer pads with -1 and
    the sparse MQA core must mask those slots to -inf in the softmax. A bug
    in either the padding or the masking would produce NaN or wrong outputs.
    """
    logger.info("Test: CSA topk edge cases (==n_blocks, >n_blocks)")
    torch.manual_seed(202)
    results = []
    B, T, d = 1, 32, 16
    m, nh, nIh, c, c_I, dc = 8, 2, 2, 8, 4, 8
    n_blocks = T // m
    H = torch.randn(B, T, d, device=device) * 0.1
    W_aKV = torch.randn(c, d, device=device) * 0.1
    W_bKV = torch.randn(c, d, device=device) * 0.1
    W_aZ = torch.randn(c, d, device=device) * 0.1
    W_bZ = torch.randn(c, d, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(dc, d, device=device) * 0.1
    W_UQ = torch.randn(c * nh, dc, device=device) * 0.1
    W_IUQ = torch.randn(c_I * nIh, dc, device=device) * 0.1
    W_w = torch.randn(nIh, d, device=device) * 0.1
    W_KV_idx = torch.randn(c_I, d, device=device) * 0.1
    W_Z_idx = torch.randn(c_I, d, device=device) * 0.1
    B_idx = torch.randn(m, c_I, device=device) * 0.1
    sink = torch.zeros(nh, device=device)

    for topk, label in [(n_blocks, 'topk==n_blocks'),
                        (n_blocks + 4, 'topk>n_blocks')]:
        try:
            o = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=0, sink_logits=sink)
            ok_ = o.shape == (B, T, nh * c) and torch.isfinite(o).all().item()
            results.append(_ok(f'CSA {label}', ok_,
                               f'shape={tuple(o.shape)}, finite={ok_}, topk={topk}'))
        except Exception as e:
            results.append(_ok(f'CSA {label}', False, f'{type(e).__name__}: {e}'))
    return results


def test_csa_topk_zero(device='cpu'):
    """Regression: ``naive_csa`` must handle ``topk=0`` without crashing.

    Previously ``topk=0`` caused ``IndexError: amax(): Expected reduction
    dim -1 to have non-zero size`` because the sparse attention ``scores``
    tensor had shape ``[B, T, nh, 0]`` and ``scores.amax(-1)`` cannot reduce
    an empty dim. With ``topk=0`` the user is asking for "no sparse
    selection", which is a degenerate but valid configuration: the sparse
    branch contributes exactly zero, and only the sliding-window branch
    (if enabled) produces non-zero output.

    This test verifies:
      1. ``topk=0`` with ``sliding_window=0`` produces a finite all-zero
         output (no sparse, no SW -> everything is zero).
      2. ``topk=0`` with ``sliding_window>0`` produces a finite non-zero
         output (SW branch still contributes).
    """
    logger.info("Test: CSA with topk=0 (no sparse selection)")
    torch.manual_seed(207)
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 0, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device) * 0.1
    W_aKV = torch.randn(c, d, device=device) * 0.1
    W_bKV = torch.randn(c, d, device=device) * 0.1
    W_aZ = torch.randn(c, d, device=device) * 0.1
    W_bZ = torch.randn(c, d, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(dc, d, device=device) * 0.1
    W_UQ = torch.randn(c * nh, dc, device=device) * 0.1
    W_IUQ = torch.randn(c_I * nIh, dc, device=device) * 0.1
    W_w = torch.randn(nIh, d, device=device) * 0.1
    W_KV_idx = torch.randn(c_I, d, device=device) * 0.1
    W_Z_idx = torch.randn(c_I, d, device=device) * 0.1
    B_idx = torch.randn(m, c_I, device=device) * 0.1
    sink = torch.zeros(nh, device=device)

    # topk=0, no SW -> all-zero output (sparse branch contributes 0, no SW).
    o_no_sw = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                        W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                        sliding_window=0, sink_logits=sink)
    no_sw_ok = (o_no_sw.shape == (B, T, nh * c)
                and torch.isfinite(o_no_sw).all().item()
                and o_no_sw.abs().max().item() < 1e-12)

    # topk=0, with SW -> non-zero output (SW branch contributes).
    o_with_sw = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=4, sink_logits=sink)
    with_sw_ok = (o_with_sw.shape == (B, T, nh * c)
                  and torch.isfinite(o_with_sw).all().item()
                  and o_with_sw.abs().max().item() > 0.0)

    return [
        _ok('CSA topk=0 no-SW produces zero output', no_sw_ok,
            f'shape={tuple(o_no_sw.shape)}, max|o|={o_no_sw.abs().max().item():.2e}'),
        _ok('CSA topk=0 with-SW produces non-zero output', with_sw_ok,
            f'shape={tuple(o_with_sw.shape)}, max|o|={o_with_sw.abs().max().item():.2e}'),
    ]


def test_csa_compress_kv_overlapped_zero_length(device='cpu'):
    """Regression: ``csa_compress_kv_overlapped`` must accept T=0 directly.

    The public ``naive_csa`` guards T=0 with an early return, so the
    overlapped compression function is never reached through that API.
    However ``csa_compress_kv_overlapped`` is itself a public function
    (no underscore prefix) imported directly by
    ``run_correctness.py::test_overlap_causality`` and ``method_analysis.py``.
    Calling it with T=0 previously crashed with a confusing
    ``RuntimeError: Sizes of tensors must match except in dimension 2``
    because the ``-inf`` pad always inserts one block while ``A_logits``
    was empty (0 blocks), making the ``torch.cat([A_logits, Bb_logits],
    dim=2)`` fail on the mismatched dim-1 sizes.

    The sibling ``csa_compress_kv`` (single-branch) already handled T=0
    correctly via the view operation returning an empty ``[B, 0, m, c]``
    tensor. This test verifies the defensive guard added to
    ``csa_compress_kv_overlapped`` makes its contract match the
    single-branch version: T=0 returns ``[B, 0, c]`` without raising.
    """
    logger.info("Test: csa_compress_kv_overlapped accepts T=0 (direct call)")
    torch.manual_seed(211)
    B, T, m, c = 2, 0, 8, 16
    Ca = torch.randn(B, T, c, device=device)
    Cb = torch.randn(B, T, c, device=device)
    Za = torch.randn(B, T, c, device=device)
    Zb = torch.randn(B, T, c, device=device)
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    try:
        out = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
        ok = out.shape == (B, 0, c) and torch.isfinite(out).all().item()
        err = ''
    except Exception as e:
        out = None
        ok = False
        err = f'{type(e).__name__}: {e}'
    return [
        _ok('overlapped compression T=0 returns [B,0,c]', ok,
            f'shape={tuple(out.shape) if out is not None else "n/a"}, err={err}'),
    ]


def test_overlapped_compress_no_pad_parity(device='cpu'):
    """No-grad overlapped compression must match the padded autograd path.

    The inference path writes the shifted b-branch directly into the logits
    buffer instead of materializing full ``[B, T, c]`` ``Zb_prev`` /
    ``Cb_prev`` pads. This regression test pins both value parity and the
    absence of ``F.pad`` on that path.
    """
    logger.info("Test: overlapped compression no-pad inference parity")
    results = []
    cases = [
        (1, 8, 8, 16, torch.float32),
        (3, 16, 4, 8, torch.float32),
        (1, 16, 16, 4, torch.float64),
        (2, 32, 8, 32, torch.bfloat16),
    ]
    for B, T, m, c, dtype in cases:
        torch.manual_seed(311 + T + m)
        Ca = torch.randn(B, T, c, dtype=dtype, device=device)
        Cb = torch.randn(B, T, c, dtype=dtype, device=device)
        Za = torch.randn(B, T, c, dtype=dtype, device=device)
        Zb = torch.randn(B, T, c, dtype=dtype, device=device)
        Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
        Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1

        pad_calls = []
        orig_pad = torch.nn.functional.pad

        def _record_pad(*args, **kwargs):
            pad_calls.append(1)
            return orig_pad(*args, **kwargs)

        torch.nn.functional.pad = _record_pad
        try:
            with torch.no_grad():
                fast = csa_compress_kv_overlapped(
                    Ca, Cb, Za, Zb, Ba, Bb, m)
        finally:
            torch.nn.functional.pad = orig_pad

        ref = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
        diff = (fast.float() - ref.float()).abs().max().item()
        label = f'B={B},T={T},m={m},c={c},{str(dtype).split(".")[-1]}'
        results += [
            _ok(f'overlapped no-pad parity ({label})', diff == 0.0,
                f'max|diff|={diff:.2e}'),
            _ok(f'overlapped no-pad avoids F.pad ({label})',
                not pad_calls, f'pad calls={len(pad_calls)}'),
        ]
    return results


def test_compress_chunk_tiling_parity(device='cpu'):
    """Block-tiled KV compression must match the full-buffer reference.

    Both compression kernels are row-wise over compressed blocks, so tiling
    the block axis only changes the size of the transient logits / softmax
    buffers. This pins (a) the automatic heuristic engages for a large
    buffer, (b) every explicit tile size agrees with the forced full path,
    and (c) ``compress_chunk`` rejects non-int / negative values. fp64 is
    expected to be bit-identical; fp32/bf16 may differ by the shape-dependent
    softmax reduction ULP (< 1e-6 on the test data).
    """
    logger.info("Test: block-tiled KV compression parity")
    from ops_csa import (csa_compress_kv, csa_compress_kv_overlapped,
                         _resolve_compress_tile)

    cases = [
        (1, 64, 8, 16, torch.float32),
        (2, 32, 4, 8, torch.float32),
        (1, 64, 16, 4, torch.float64),
        (1, 64, 1, 8, torch.float32),
        (3, 48, 16, 12, torch.bfloat16),
    ]
    results = []
    for B, T, m, c, dtype in cases:
        torch.manual_seed(911 + T + m)
        C = torch.randn(B, T, c, dtype=dtype, device=device)
        Z = torch.randn(B, T, c, dtype=dtype, device=device)
        bias = torch.randn(m, c, dtype=dtype, device=device) * 0.1
        Ca = torch.randn(B, T, c, dtype=dtype, device=device)
        Cb = torch.randn(B, T, c, dtype=dtype, device=device)
        Za = torch.randn(B, T, c, dtype=dtype, device=device)
        Zb = torch.randn(B, T, c, dtype=dtype, device=device)
        Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
        Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
        n_blocks = T // m
        with torch.no_grad():
            ref_s = csa_compress_kv(
                C, Z, bias, m, compress_chunk=n_blocks)
            ref_o = csa_compress_kv_overlapped(
                Ca, Cb, Za, Zb, Ba, Bb, m, compress_chunk=n_blocks)
            auto_s = csa_compress_kv(C, Z, bias, m)
            auto_o = csa_compress_kv_overlapped(
                Ca, Cb, Za, Zb, Ba, Bb, m)
            max_diff = 0.0
            for tile in (1, 2, 3, n_blocks):
                got_s = csa_compress_kv(
                    C, Z, bias, m, compress_chunk=tile)
                got_o = csa_compress_kv_overlapped(
                    Ca, Cb, Za, Zb, Ba, Bb, m, compress_chunk=tile)
                max_diff = max(
                    max_diff,
                    (got_s.float() - ref_s.float()).abs().max().item(),
                    (got_o.float() - ref_o.float()).abs().max().item())
            max_diff = max(
                max_diff,
                (auto_s.float() - ref_s.float()).abs().max().item(),
                (auto_o.float() - ref_o.float()).abs().max().item())
        label = f'B={B},T={T},m={m},c={c},{str(dtype).split(".")[-1]}'
        # fp64 must be exact; float32/bf16 tolerate the softmax-kernel ULP.
        tol = 0.0 if dtype == torch.float64 else 1e-6
        results.append(_ok(
            f'compress tiled parity ({label})', max_diff <= tol,
            f'max|diff|={max_diff:.2e} (tol={tol:.0e})'))

    # The auto heuristic must keep small buffers on the full path and tile a
    # multi-megabyte buffer. The model-level long-context prefill is exactly
    # the regime the tiling exists for.
    small = _resolve_compress_tile(
        0, 1, 4, 16, 8, overlapped=False)
    large = _resolve_compress_tile(
        0, 1, 4096, 16, 64, overlapped=True)
    explicit = _resolve_compress_tile(
        7, 1, 100, 16, 64, overlapped=True)
    results.append(_ok(
        'compress auto tile heuristic',
        small == 4 and 0 < large < 4096 and explicit == 7,
        f'small={small}, large={large}, explicit={explicit}'))

    bad_accepted = []
    for bad in (True, 1.5, -1):
        for fn, args in (
                (csa_compress_kv, (C, Z, bias, m)),
                (csa_compress_kv_overlapped,
                 (Ca, Cb, Za, Zb, Ba, Bb, m))):
            try:
                fn(*args, compress_chunk=bad)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001
                bad_accepted.append(f'{bad!r}: {type(exc).__name__}')
                continue
            bad_accepted.append(repr(bad))
    results.append(_ok(
        'compress_chunk validates geometry', not bad_accepted,
        f'rejected={bad_accepted}'))
    return results


def test_partial_rope_properties(device='cpu'):
    """Unit properties of the partial RoPE (DeepSeek-V4 §2.3.3).

    Verifies on ``ops_rope.partial_rope``:
      1. Norm preservation (the rotation is orthogonal).
      2. Inverse identity: rotating by negated positions undoes the
         rotation (this is the core-attention OUTPUT countermeasure).
      3. Relative property: <R(p1)x, R(p2)y> depends only on p1-p2 (the
         RoPE score invariance the paper's queries/keys rely on).
      4. The non-rotated leading dims are untouched (PARTIAL rope).
      5. ``effective_rope_dim`` clamps to the head dim and to even values.
    """
    logger.info("Test: partial RoPE properties (§2.3.3)")
    from ops_rope import partial_rope, effective_rope_dim, token_positions

    torch.manual_seed(123)
    results = []

    # 1-2-4: norm preservation + inverse identity + partial-ness on a
    # query-shaped tensor [B, T, nh, c] with the sequence axis at dim=1.
    x = torch.randn(2, 7, 3, 16, dtype=torch.float64, device=device)
    pos = token_positions(7, device)
    xr = partial_rope(x, pos, 6, dim=1)
    norm_ok = torch.allclose(
        x.norm(dim=-1), xr.norm(dim=-1), atol=1e-12)
    xi = partial_rope(xr, -pos, 6, dim=1)
    inverse_ok = torch.allclose(x, xi, atol=1e-12)
    # The leading (16 - 6) dims must be untouched.
    partial_ok = torch.equal(x[..., :10], xr[..., :10])
    results += [
        _ok('partial_rope preserves norms', norm_ok, ''),
        _ok('partial_rope inverse identity (negated positions)',
            inverse_ok, f'max_err={(x - xi).abs().max().item():.2e}'),
        _ok('partial_rope leaves non-rope dims untouched', partial_ok, ''),
    ]

    # 3: relative property.
    xv = torch.randn(16, dtype=torch.float64, device=device)
    yv = torch.randn(16, dtype=torch.float64, device=device)

    def rot(v, p):
        return partial_rope(
            v.view(1, 1, 16), torch.tensor([p], device=device), 8, dim=1,
        ).view(-1)

    a = (rot(xv, 10) * rot(yv, 3)).sum()
    b = (rot(xv, 20) * rot(yv, 13)).sum()
    rel_ok = torch.allclose(a, b, atol=1e-12)
    results.append(_ok(
        'partial_rope relative property (score depends on p1-p2 only)',
        rel_ok, f'a={a.item():.6f}, b={b.item():.6f}'))

    # 5: effective_rope_dim clamping.
    clamp_ok = (
        effective_rope_dim(64, 32) == 32
        and effective_rope_dim(64, 8) == 8
        and effective_rope_dim(None, 32) == 0
        and effective_rope_dim(0, 32) == 0
        and effective_rope_dim(5, 8) == 4  # odd -> largest even below
    )
    results.append(_ok('effective_rope_dim clamping', clamp_ok, ''))
    return results


__all__ = [
    'test_csa_causality',
    'test_overlap_causality',
    'test_csa_indexer_validity',
    'test_causal_block_mask_window_close',
    'test_csa_indexer_w_idx_none',
    'test_csa_indexer_topk_zero',
    'test_csa_indexer_ste_gradient',
    'test_csa_fuse_projections_grad_contract',
    'test_csa_indexer_ste_full_softmax',
    'test_csa_full_pipeline_causality',
    'test_csa_topk_edge_cases',
    'test_csa_topk_zero',
    'test_csa_compress_kv_overlapped_zero_length',
    'test_overlapped_compress_no_pad_parity',
    'test_compress_chunk_tiling_parity',
    'test_partial_rope_properties',
]
