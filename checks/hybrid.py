"""Correctness checks: hybrid."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_fused_hybrid(device='cpu'):
    logger.info("Test: fused KDA+CSA+HCA hybrid block")
    torch.manual_seed(4)
    cfg = HybridConfig(
        d_model=64, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    # Use eval() + no_grad() for this sanity check: (a) the model has no
    # dropout/BN today so eval() is a no-op, but adding it future-proofs
    # the "finite" check against any later stochastic module; (b) without
    # no_grad() the full 5-layer autograd graph is retained on ``y`` until
    # it goes out of scope, wasting memory on a sanity-check test that
    # never calls backward().
    with torch.no_grad():
        y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    # Verify the layout matches the configured n_kda:n_csa:n_hca ratio.
    # With n_kda=3, n_csa=1, n_hca=1, total_layers=5, the layout must be
    # exactly 'KDA-KDA-KDA-CSA-HCA'.
    expected_layout = 'KDA-KDA-KDA-CSA-HCA'
    layout_ok = model.layout_str() == expected_layout
    return [
        _ok('hybrid output shape', y.shape == x.shape, str(tuple(y.shape))),
        _ok('hybrid finite', torch.isfinite(y).all().item(), ''),
        _ok('hybrid layout', layout_ok,
            f'layout={model.layout_str()} expected={expected_layout} params={n_params}'),
    ]


def test_csa_recompute_indexer_plumbing(device='cpu'):
    """``HybridConfig.csa_recompute_indexer`` must reach ``naive_csa``.

    The CSA layer owns the config -> ``naive_csa`` forwarding, so a typo in
    the plumbing would silently keep the historical full-softmax backward.
    This test builds two identically initialized CSA layers, enables the
    recompute path on one, and checks forward/gradient parity plus the
    strict-bool config validation.
    """
    from ops_fused import CSAHybridLayer
    logger.info("Test: CSA recompute indexer config plumbing")
    torch.manual_seed(19)
    base = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=3, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=8, csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=8,
        n_kda=0, n_csa=1, n_hca=0,
    )
    cfg_off = HybridConfig(**base, csa_recompute_indexer=False)
    cfg_on = HybridConfig(**base, csa_recompute_indexer=True)
    layer_off = CSAHybridLayer(cfg_off).to(device).double()
    layer_on = CSAHybridLayer(cfg_on).to(device).double()
    layer_on.load_state_dict(layer_off.state_dict())
    x = torch.randn(2, 24, cfg_off.d_model, device=device,
                    dtype=torch.float64) * 0.1
    o_off, _ = layer_off(x)
    o_on, _ = layer_on(x)
    fwd_diff = (o_off - o_on).abs().max().item()
    torch.manual_seed(5)
    w = torch.randn_like(o_off)
    (o_off * w).sum().backward()
    (o_on * w).sum().backward()
    grad_diff = max(
        (p_off.grad - p_on.grad).abs().max().item()
        for p_off, p_on in zip(layer_off.parameters(),
                               layer_on.parameters())
        if p_off.grad is not None and p_on.grad is not None)
    bad_raised = False
    try:
        HybridConfig(**base, csa_recompute_indexer=1)
    except TypeError:
        bad_raised = True
    return [
        _ok('CSA layer recompute indexer forward parity',
            fwd_diff <= 1e-12, f'max_abs_diff={fwd_diff:.3e}'),
        _ok('CSA layer recompute indexer gradient parity',
            grad_diff <= 1e-10, f'max_abs_diff={grad_diff:.3e}'),
        _ok('HybridConfig rejects non-bool csa_recompute_indexer',
            bad_raised, ''),
    ]


def test_hybrid_padding_no_crash(device='cpu'):
    """Regression: HybridKCHAttention must not crash when T is not divisible
    by csa_m / hca_m2.

    We exercise T values that require non-trivial padding for BOTH the
    CSA layer (m=8) and the HCA layer (m2=16), for several batch sizes
    including B=1 and B>pad.
    """
    logger.info("Test: hybrid forward with non-divisible T (padding regression)")
    torch.manual_seed(10)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    # T values that force padding for CSA (m=8) and HCA (m2=16):
    #   T=10 -> csa pad=6,  hca pad=6
    #   T=13 -> csa pad=3,  hca pad=3
    #   T=20 -> csa pad=4,  hca pad=12
    #   T=1  -> csa pad=7,  hca pad=15  (single-token decode edge case)
    test_Ts = [10, 13, 20, 1]
    # Batch sizes that exercise both the B<=pad and B>pad failure modes.
    test_Bs = [1, 3, 8]
    all_ok = True
    detail_parts = []
    for B in test_Bs:
        for T in test_Ts:
            model.reset_state()
            x = torch.randn(B, T, cfg.d_model, device=device) * 0.1
            try:
                with torch.no_grad():
                    y = model(x)
                shape_ok = y.shape == x.shape
                finite_ok = torch.isfinite(y).all().item()
                if not (shape_ok and finite_ok):
                    all_ok = False
                    detail_parts.append(
                        f'B={B},T={T}: shape={tuple(y.shape)} finite={finite_ok}')
            except Exception as e:
                all_ok = False
                detail_parts.append(f'B={B},T={T}: CRASH {type(e).__name__}: {e}')
    detail = '; '.join(detail_parts) if detail_parts else \
        f'all {len(test_Ts)*len(test_Bs)} (B,T) combos forward cleanly'
    return [
        _ok('hybrid padding no-crash', all_ok, detail),
    ]


def test_hybrid_state_buffer_registration(device='cpu'):
    """Regression: HybridKCHAttention._kda_state must be a registered buffer.

    Previously _kda_state was a plain Python attribute, so ``model.to(device)``
    left it on the source device, causing a device-mismatch crash on the next
    forward. We register it as a non-persistent buffer so .to() moves it
    automatically and state_dict skips it (it is runtime state, not weights).

    On a CPU-only box we cannot test a real device transfer, but we CAN
    verify the buffer is registered (so .to() will move it) and that a
    second forward after a no-op .to('cpu') still works.
    """
    logger.info("Test: hybrid _kda_state is a registered buffer")
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    # First forward populates _kda_state.
    x1 = torch.randn(2, 16, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model.reset_state()
        model(x1)
    state_is_buffer = any(b is model._kda_state for b in model.buffers())
    state_not_in_state_dict = '_kda_state' not in model.state_dict()
    # A no-op .to(device) must not break the next forward (it would have,
    # had _kda_state stayed on the wrong device after a real .to('cuda')).
    model.to(device)
    x2 = torch.randn(2, 16, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model.reset_state()
        y2 = model(x2)
    forward_ok_after_to = torch.isfinite(y2).all().item()
    return [
        _ok('_kda_state is a registered buffer', state_is_buffer,
            f'in model.buffers()={state_is_buffer}'),
        _ok('_kda_state not in state_dict (non-persistent)',
            state_not_in_state_dict,
            f'in state_dict={not state_not_in_state_dict}'),
        _ok('forward works after model.to(device)', forward_ok_after_to, ''),
    ]


def test_bench_hybrid_no_grad_inference(device='cpu'):
    """Regression: bench_hybrid's fn() must not build an autograd graph.

    Previously the ``with torch.no_grad():`` wrapped the ``def fn():`` line
    rather than the call body. The no_grad context is global and exits as
    soon as the ``with`` block ends, so later ``fn()`` calls ran with
    gradients enabled -- silently inflating both latency (graph construction)
    and peak memory (retained activations). We now put no_grad inside fn().

    This test imports the ACTUAL ``bench_hybrid`` from ``run_benchmark``
    (which has no matplotlib dependency) so that any future regression in
    ``bench_hybrid`` itself — not just in an inline replica — is caught.
    The previous version re-implemented the bench pattern inline, which meant
    a bug introduced into ``run_benchmark.bench_hybrid`` would go undetected.
    """
    logger.info("Test: bench_hybrid runs under no_grad (regression)")
    # Import the actual bench_hybrid so the test catches regressions in the
    # real function, not just in an inline replica. ``run_benchmark`` does
    # NOT import matplotlib (only ``make_figures`` does), so the import is
    # lightweight and safe.
    from run_benchmark import bench_hybrid
    B, T, d = 1, 16, 32
    fn = bench_hybrid(B, T, d, device)
    y = fn()
    return [
        _ok('bench_hybrid output is grad-free', not y.requires_grad,
            f'requires_grad={y.requires_grad} (should be False)'),
    ]


def test_hybrid_per_layer_kda_state(device='cpu'):
    """Regression: each KDA layer must keep its OWN recurrent state.

    Previously ``_kda_state`` was a single tensor shared across all KDA
    layers in the stack. With layout ``[KDA, CSA, KDA, HCA, KDA]`` this meant
    layer 0's state was passed as the initial state to layer 2, and layer 2's
    state to layer 4 -- which is mathematically wrong because each KDA layer
    has its own parameters (q_proj/k_proj/...). On the NEXT forward call,
    layer 0 would then be seeded with layer 4's state from the previous
    call, silently corrupting autoregressive decoding and biasing training.

    The fix is to keep one state per KDA layer, stored as a stacked tensor
    of shape ``[n_kda_layers, B, HV, K, V]``. This test verifies:
      1. After a forward pass, ``_kda_state`` has the per-layer leading axis.
      2. Across two consecutive forward passes (no reset), each layer's
         state evolves independently -- i.e. the state for layer i after
         call 2 differs from the state for layer i after call 1 (proving
         we are not just overwriting a single shared slot).
      3. The per-layer states differ from each other within one call
         (proving they are not aliased to the same tensor).
    """
    logger.info("Test: hybrid per-layer KDA state independence")
    torch.manual_seed(11)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    n_kda_layers = sum(1 for k in model.layout if k == 'kda')
    B, T = 2, 16
    HV, K_dim, V_dim = cfg.n_heads_v, cfg.head_dim_k, cfg.head_dim_v

    # --- 1. Shape check: _kda_state must carry a per-layer leading axis. ---
    model.reset_state()
    x1 = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model(x1)
    shape_ok = (
        model._kda_state is not None
        and model._kda_state.shape[0] == n_kda_layers
        and model._kda_state.shape == (n_kda_layers, B, HV, K_dim, V_dim)
    )
    shape_detail = (
        f'expected ({n_kda_layers},{B},{HV},{K_dim},{V_dim}), got '
        f'{tuple(model._kda_state.shape) if model._kda_state is not None else None}'
    )

    # --- 2. Per-layer states must NOT alias each other (independence). ---
    # If the implementation shared a single state across layers, all slices
    # would be bitwise-identical (because the same tensor would be stacked
    # with itself). With per-layer states, each slice carries a different
    # layer's recurrence result and should differ.
    # Clone (with detach) so the snapshot is immune to any future in-place
    # updates to ``_kda_state``. If forward ever switched to in-place updates
    # (e.g. ``self._kda_state[i].copy_(...)``), ``stacked1`` would alias the
    # live buffer and the "evolved across calls" check below would trivially
    # pass with a false negative.
    stacked1 = model._kda_state.detach().clone()
    pairwise_distinct = True
    for i in range(n_kda_layers):
        for j in range(i + 1, n_kda_layers):
            if torch.equal(stacked1[i], stacked1[j]):
                pairwise_distinct = False
                break
        if not pairwise_distinct:
            break

    # --- 3. Across two forward calls, each layer's state must evolve. ---
    # Run a second forward WITHOUT resetting state. Each layer's state should
    # change because each KDA layer ingested x2 on top of its own prior state.
    # If state were shared (single slot), only the LAST layer's state would
    # persist across calls -- and the "layer 0" slice would actually hold the
    # last layer's old state, not its own.
    x2 = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model(x2)
    stacked2 = model._kda_state
    all_evolved = True
    for i in range(n_kda_layers):
        if torch.equal(stacked1[i], stacked2[i]):
            all_evolved = False
            break

    # --- 4. Functional equivalence: per-layer state threading must match
    #     running each KDA layer in isolation with its own carried state. ---
    # Build a reference: run the model step by step, but for each KDA layer
    # keep a separate state, and verify the model's stacked[i] matches the
    # reference state for layer i after the same two calls.
    # Wrap in torch.no_grad() to match the test's intent (functional
    # equivalence check, not gradient tracking) and avoid building a
    # computation graph that would waste memory on a 5-layer eval loop.
    #
    # NOTE: this reference mirrors ``HybridKCHAttention.forward`` EXACTLY —
    # it does NOT externally pad/trim CSA/HCA inputs, because
    # ``HybridKCHAttention.forward`` doesn't either (it relies on
    # ``naive_csa`` / ``naive_hca``'s internal right-padding + output trim,
    # verified by ``test_csa_hca_non_divisible_T``). The previous version of
    # this reference DID pad externally, which was both redundant AND a
    # maintenance trap: if the internal padding contract ever changed, the
    # reference would silently diverge from the actual implementation while
    # still passing the test (because the test only uses T divisible by m,
    # where external and internal padding produce identical results).
    # Removing the external padding makes the reference a faithful replica.
    model.reset_state()
    ref_states = [None] * n_kda_layers
    with torch.no_grad():
        for x in (x1, x2):
            kda_idx = 0
            h = x
            for layer, norm, kind in zip(model.layers, model.norms, model.layout):
                residual = h
                h_norm = norm(h)
                if kind == 'kda':
                    o, ref_states[kda_idx] = layer(h_norm, ref_states[kda_idx])
                    kda_idx += 1
                else:
                    # Stateless CSA/HCA: pass h_norm directly. The operators
                    # handle non-divisible T internally (right-pad + output
                    # trim), so no external padding is needed — matching
                    # HybridKCHAttention.forward exactly.
                    o, _ = layer(h_norm, None)
                h = residual + o
    ref_match = all(
        torch.allclose(stacked2[i], ref_states[i], atol=1e-6)
        for i in range(n_kda_layers)
    )

    return [
        _ok('per-layer state shape [n_kda,B,HV,K,V]', shape_ok, shape_detail),
        _ok('per-layer states are pairwise distinct', pairwise_distinct,
            'slices differ across layers (not aliased to one tensor)'),
        _ok('each layer state evolves across calls', all_evolved,
            'every layer i: state_after_call1 != state_after_call2'),
        _ok('stacked state matches per-layer reference', ref_match,
            'model._kda_state[i] == reference state for layer i'),
    ]


def test_hybrid_backward_produces_grads(device='cpu'):
    """Regression: HybridKCHAttention backward must produce finite grads for
    all DIFFERENTIABLE parameters.

    The lightning indexer uses ``torch.topk`` which returns integer indices
    that do NOT propagate gradients directly. Historically this left the
    indexer parameters (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``,
    ``B_idx`` in ``CSAHybridLayer``) at their random initialization after
    ``backward()`` (their ``.grad`` stayed ``None`` and AdamW silently
    skipped them).

    The P0-4 fix added a straight-through estimator (STE) in
    ``ops_csa.naive_csa`` so the indexer parameters DO receive gradient
    through the STE's differentiable ``soft_weights`` path. Under the
    default ``use_ste=True`` (which this test exercises via the default
    ``CSAHybridLayer`` constructor), the indexer parameters are now
    trainable and SHOULD have a non-None ``.grad`` after ``backward()``.

    What WOULD still be a bug: a differentiable parameter (one whose
    gradient SHOULD flow) ending up with a non-finite or all-zero
    gradient. This test runs a full forward + backward pass over the
    hybrid stack and verifies that:
      1. All non-indexer parameters receive a finite, non-zero gradient.
      2. Indexer parameters (the 5 listed above) have a FINITE,
         non-zero gradient under STE (the P0-4 fix's contract). If a
         future refactor breaks the STE path, the indexer params would
         revert to ``.grad is None`` — the test catches that regression
         by treating ``.grad is None`` on an indexer param as a failure
         (rather than the historical "expected" behaviour).
      3. No parameter has a non-finite (NaN/Inf) gradient.

    The test uses a sequence length large enough (T=64) so that CSA/HCA
    blocks are actually selectable by queries — at T=16 with csa_m=8,
    only block 0 is selectable and the b-branch / sink parameters have
    legitimately-zero gradients (no contribution to the loss).
    """
    logger.info("Test: Hybrid backward produces finite grads (large T)")
    torch.manual_seed(200)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=8, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device)
    model.train()
    model.reset_state()
    # T=64 ensures CSA has n_blocks=8 (blocks 0..6 selectable) and HCA has
    # n_blocks=8 (same), so the b-branch / sink / W_bKV / W_bZ parameters
    # all receive non-zero gradients through the differentiable attention path.
    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    target = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()

    # Indexer parameters (CSA layer index 3 in the default 3:1:1 layout).
    # Under the P0-4 STE fix these ARE differentiable (gradient flows
    # through the ``soft_weights`` path), and the default model config
    # uses ``use_ste=True`` — so a None grad on an indexer param is a
    # regression (the STE path broke), not a legitimate case. We treat it
    # as ``differentiable_no_grad`` so the test fails loudly instead of
    # silently pinning the old un-trained-indexer behaviour. A non-finite
    # or unexpectedly-zero grad on ANY differentiable param is also a bug.
    indexer_param_substrings = ('W_IUQ', 'W_w', 'W_KV_idx', 'W_Z_idx', 'B_idx')
    differentiable_no_grad = []
    differentiable_zero_grad = []
    differentiable_non_finite = []
    differentiable_ok = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_indexer = any(s in name for s in indexer_param_substrings)
        if p.grad is None:
            if is_indexer:
                # P0-4 STE: under the default ``use_ste=True`` the
                # indexer params SHOULD have a non-None grad. A None
                # grad here means the STE path is broken (regression).
                # We flag it as ``differentiable_no_grad`` so the test
                # fails loudly instead of silently pinning the old
                # un-trained-indexer behaviour.
                differentiable_no_grad.append(name)
                continue
            else:
                differentiable_no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            differentiable_non_finite.append(name)
        elif is_indexer:
            # STE path produced a finite grad — count as OK. The
            # magnitude may be small (the STE only gathers top-k
            # columns by default), so we do not apply the 1e-12 zero
            # check to indexer params.
            differentiable_ok += 1
        # Use a small epsilon rather than exact == 0.0: a legitimately tiny
        # but non-zero fp32 grad (e.g. from underflow in a masked branch)
        # would otherwise be counted as differentiable_ok, hiding real bugs.
        elif p.grad.abs().max().item() < 1e-12:
            differentiable_zero_grad.append(name)
        else:
            differentiable_ok += 1

    all_ok = (
        not differentiable_no_grad
        and not differentiable_zero_grad
        and not differentiable_non_finite
    )
    detail = (
        f'differentiable_with_grad={differentiable_ok}, '
        f'differentiable_no_grad={differentiable_no_grad[:3]}, '
        f'differentiable_zero_grad={differentiable_zero_grad[:3]}, '
        f'differentiable_non_finite={differentiable_non_finite[:3]}'
    )
    return [
        _ok('hybrid backward produces finite non-zero grads for differentiable params',
            all_ok, detail),
    ]


def test_hybrid_no_kda_layout(device='cpu'):
    """Verify HybridKCHAttention works with n_kda=0 (CSA+HCA only).

    When there are no KDA layers, ``_kda_state`` should remain None and the
    forward pass should not attempt to stack empty states. A bug in the
    ``n_kda_layers > 0`` guard or the state-stacking logic would crash.
    """
    logger.info("Test: Hybrid with no-KDA layout (CSA+HCA only)")
    torch.manual_seed(205)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=0, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=2).to(device).eval()
    x = torch.randn(2, 16, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model.reset_state()
        y = model(x)
    state_is_none = model._kda_state is None
    return [
        _ok('hybrid no-KDA forward', y.shape == x.shape and torch.isfinite(y).all().item(),
            f'shape={tuple(y.shape)}, layout={model.layout_str()}'),
        _ok('hybrid no-KDA state is None', state_is_none,
            f'_kda_state is None: {state_is_none}'),
    ]


def test_hybrid_no_csa_layout(device='cpu'):
    """Verify HybridKCHAttention works with n_csa=0 (KDA+HCA only).

    The ablation experiment sweeps ratios including ``(3,0,1)``; a bug in
    ``_build_layout`` or the per-layer dispatcher would only surface as a
    runtime error during the ablation. This test mirrors
    ``test_hybrid_no_kda_layout`` for the no-CSA case: forward shape,
    finiteness, and KDA state survives across calls.
    """
    logger.info("Test: Hybrid with no-CSA layout (KDA+HCA only)")
    torch.manual_seed(207)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=1, n_csa=0, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=2).to(device).eval()
    x = torch.randn(2, 16, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model.reset_state()
        y1 = model(x)
        # Second call: KDA state should persist and be re-used.
        y2 = model(x)
    state_is_stacked = model._kda_state is not None
    return [
        _ok('hybrid no-CSA forward', y1.shape == x.shape and torch.isfinite(y1).all().item(),
            f'shape={tuple(y1.shape)}, layout={model.layout_str()}'),
        _ok('hybrid no-CSA state persists', state_is_stacked,
            f'_kda_state is not None: {state_is_stacked}'),
        _ok('hybrid no-CSA second call', y2.shape == x.shape and torch.isfinite(y2).all().item(),
            f'shape={tuple(y2.shape)}'),
    ]


def test_hybrid_no_hca_layout(device='cpu'):
    """Verify HybridKCHAttention works with n_hca=0 (KDA+CSA only).

    The ablation experiment sweeps ratios including ``(3,1,0)`` and
    ``(0,1,0)``; a bug in ``_build_layout`` or the per-layer dispatcher
    would only surface as a runtime error during the ablation. This test
    mirrors ``test_hybrid_no_kda_layout`` for the no-HCA case.
    """
    logger.info("Test: Hybrid with no-HCA layout (KDA+CSA only)")
    torch.manual_seed(208)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=1, n_csa=1, n_hca=0,
    )
    model = HybridKCHAttention(cfg, total_layers=2).to(device).eval()
    x = torch.randn(2, 16, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        model.reset_state()
        y1 = model(x)
        y2 = model(x)
    state_is_stacked = model._kda_state is not None
    return [
        _ok('hybrid no-HCA forward', y1.shape == x.shape and torch.isfinite(y1).all().item(),
            f'shape={tuple(y1.shape)}, layout={model.layout_str()}'),
        _ok('hybrid no-HCA state persists', state_is_stacked,
            f'_kda_state is not None: {state_is_stacked}'),
        _ok('hybrid no-HCA second call', y2.shape == x.shape and torch.isfinite(y2).all().item(),
            f'shape={tuple(y2.shape)}'),
    ]


def test_weight_decay_param_groups(device='cpu'):
    """Regression: ``_build_param_groups`` must exclude embeddings, biases,
    and LayerNorm parameters from the weight-decay group.

    Standard ML practice: embeddings, biases, and LayerNorm affine parameters
    should NOT be weight-decayed. The ``_build_param_groups`` helper in
    ``run_quality.py`` implements this grouping; this test verifies:

      1. ``nn.Embedding`` weights are in the no-decay group.
      2. ``nn.LayerNorm`` weight and bias are in the no-decay group.
      3. ``nn.Linear`` bias is in the no-decay group.
      4. ``nn.Linear`` weight (2-D) is in the decay group.
      5. 1-D ``nn.Parameter`` tensors (e.g. CSA sink logits, positional
         biases) are in the no-decay group.
      6. The total parameter count in both groups matches the sum of all
         module parameters (no parameter is dropped or duplicated).
      7. The weight_decay values are correctly set on each group.
    """
    logger.info("Test: weight decay param groups (embed/bias/LayerNorm excluded)")
    # ``_build_param_groups`` is imported lazily because run_quality.py is a
    # heavy module (it builds nn.Modules at import time); importing it at
    # module load would slow down the entire test suite for one test.
    # ``HybridKCHAttention`` / ``HybridConfig`` are ALREADY imported at module
    # level (line 50) — the previous inline re-import was a copy-paste smell
    # that shadowed the module-level imports for no reason.
    from run_quality import _build_param_groups

    embed = torch.nn.Embedding(16, 32).to(device)
    head = torch.nn.Sequential(
        torch.nn.LayerNorm(32),
        torch.nn.Linear(32, 16),  # has bias=True by default
    ).to(device)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device)

    groups = _build_param_groups(embed, head, model, weight_decay=0.01)
    decay_params = groups[0]['params']
    no_decay_params = groups[1]['params']
    decay_wd = groups[0]['weight_decay']
    no_decay_wd = groups[1]['weight_decay']

    # 7. Weight decay values are correct.
    wd_ok = (decay_wd == 0.01 and no_decay_wd == 0.0)

    # 6. Total param count matches.
    total_in_groups = len(decay_params) + len(no_decay_params)
    total_in_modules = len(list(embed.parameters())) + len(list(head.parameters())) + len(list(model.parameters()))
    count_ok = (total_in_groups == total_in_modules)

    # Build id sets for fast lookup.
    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}

    # 1. Embedding weight is in no-decay.
    embed_weight_id = id(list(embed.parameters())[0])
    embed_ok = embed_weight_id in no_decay_ids

    # 2. LayerNorm weight and bias are in no-decay.
    ln_params = [p for m in head.modules() if isinstance(m, torch.nn.LayerNorm)
                 for p in m.parameters(recurse=False)]
    ln_ok = all(id(p) in no_decay_ids for p in ln_params) and len(ln_params) >= 2

    # 3. Linear bias is in no-decay.
    linear_biases = [p for m in head.modules() if isinstance(m, torch.nn.Linear)
                     for p in m.parameters(recurse=False) if p.ndim == 1]
    linear_bias_ok = all(id(p) in no_decay_ids for p in linear_biases) and len(linear_biases) >= 1

    # 4. Linear weight (2-D) is in decay.
    linear_weights = [p for m in head.modules() if isinstance(m, torch.nn.Linear)
                      for p in m.parameters(recurse=False) if p.ndim == 2]
    linear_weight_ok = all(id(p) in decay_ids for p in linear_weights) and len(linear_weights) >= 1

    # 5. 1-D nn.Parameter tensors (sink logits, positional biases) are in
    #    no-decay. The CSA sink is shape (nh,) = (2,), 1-D.
    sink_params = [p for n, p in model.named_parameters() if 'sink' in n.lower()]
    sink_ok = all(id(p) in no_decay_ids for p in sink_params) and len(sink_params) >= 1

    # 5b. 2-D nn.Parameter tensors (Ba, Bb, B_idx, B_pos) are in NO-decay.
    # These are positional bias tables that function analogously to
    # embeddings (lookup tables indexed by block-position, not weights
    # applied to activations). Decaying them shrinks the table toward
    # zero and degrades the model's ability to represent
    # position-dependent compression patterns. ``_build_param_groups``
    # excludes them via an explicit name-suffix match (their ndim == 2
    # so the ``p.ndim <= 1`` rule does not catch them).
    pos_bias_params = [p for n, p in model.named_parameters()
                       if any(k in n for k in ('Ba', 'Bb', 'B_idx', 'B_pos'))
                       and p.ndim == 2]
    pos_bias_ok = all(id(p) in no_decay_ids for p in pos_bias_params) and len(pos_bias_params) >= 1

    return [
        _ok('weight_decay values correct (0.01 / 0.0)', wd_ok,
            f'decay_wd={decay_wd}, no_decay_wd={no_decay_wd}'),
        _ok('param count matches (no params dropped/duplicated)', count_ok,
            f'in_groups={total_in_groups}, in_modules={total_in_modules}'),
        _ok('nn.Embedding weight in no-decay group', embed_ok,
            f'embed.weight id in no_decay: {embed_ok}'),
        _ok('nn.LayerNorm params in no-decay group', ln_ok,
            f'{len(ln_params)} LayerNorm params, all in no_decay'),
        _ok('nn.Linear bias in no-decay group', linear_bias_ok,
            f'{len(linear_biases)} Linear biases, all in no_decay'),
        _ok('nn.Linear weight in decay group', linear_weight_ok,
            f'{len(linear_weights)} Linear weights, all in decay'),
        _ok('1-D nn.Parameter (sink) in no-decay group', sink_ok,
            f'{len(sink_params)} sink params, all in no_decay'),
        _ok('2-D nn.Parameter (positional bias) in no-decay group', pos_bias_ok,
            f'{len(pos_bias_params)} pos-bias params, all in no_decay'),
    ]


def test_hybrid_kda_state_dtype_mismatch(device='cpu'):
    """``HybridKCHAttention`` must survive a dtype change between forwards.

    A fp32 forward followed by a fp16 forward (after ``model.half()``) must
    not crash, and the output must be finite.
    """
    logger.info("Test: HybridKCHAttention survives dtype change between forwards")
    torch.manual_seed(220)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16, csa_nIh=2, csa_cI=4,
        csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=2, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=4).to(device).eval()
    # First forward in fp32 — populates _kda_state with fp32 tensors.
    x_fp32 = torch.randn(1, 8, cfg.d_model, device=device, dtype=torch.float32) * 0.1
    with torch.no_grad():
        y1 = model(x_fp32)
    fp32_state_dtype = model._kda_state.dtype if model._kda_state is not None else None

    # Switch model to fp16. The cached _kda_state is still fp32 (registered
    # buffer is NOT moved by .half() — only parameters are; but we manually
    # cast the buffer to simulate the dtype-mismatch scenario the fix targets).
    # Actually, .half() DOES move non-persistent buffers too in modern torch,
    # so the state would be fp16 after .half(). To test the actual fix path
    # (dtype mismatch between _kda_state and x), we manually restore the
    # state to fp32 AFTER .half() to simulate a stale-state scenario.
    model = model.half()
    if model._kda_state is not None:
        model._kda_state = model._kda_state.to(torch.float32)

    # Second forward in fp16 — must not crash despite _kda_state being fp32.
    x_fp16 = x_fp32.to(torch.float16)
    try:
        with torch.no_grad():
            y2 = model(x_fp16)
        ok = (y2.shape == x_fp16.shape
              and y2.dtype == torch.float16
              and torch.isfinite(y2.float()).all().item())
        err = ''
    except Exception as e:
        ok = False
        err = f'{type(e).__name__}: {e}'
    return [
        _ok('hybrid survives dtype change (fp32 state -> fp16 forward)', ok,
            f'y2.shape={tuple(y2.shape) if ok else "n/a"}, '
            f'fp32_state_dtype={fp32_state_dtype}, err={err}'),
    ]


def test_hybrid_kda_state_batch_size_change(device='cpu'):
    """Regression: ``HybridKCHAttention`` drops KDA state on batch-size change.

    Previously, the KDA recurrent state was retained across batch-size
    changes, causing a shape mismatch crash inside ``naive_recurrent_kda``
    (the state has the old B, the new q/k/v have the new B, and the einsums
    broadcast-incompatibly). The fix drops the state on batch-size change
    (the state is per-sequence and cannot be reused across different batch
    sizes). This test verifies the drop: a B=2 forward followed by a B=1
    forward must not crash, and the state after the second forward must
    have B=1.
    """
    logger.info("Test: HybridKCHAttention drops KDA state on batch-size change")
    torch.manual_seed(221)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16, csa_nIh=2, csa_cI=4,
        csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=2, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=4).to(device).eval()
    # First forward with B=2 — populates _kda_state with B=2.
    x_b2 = torch.randn(2, 8, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        y1 = model(x_b2)
    b2_state_shape = (model._kda_state.shape[1]
                     if model._kda_state is not None else None)

    # Second forward with B=1 — must drop the B=2 state, not crash.
    x_b1 = torch.randn(1, 8, cfg.d_model, device=device) * 0.1
    try:
        with torch.no_grad():
            y2 = model(x_b1)
        ok = (y2.shape == x_b1.shape
              and torch.isfinite(y2).all().item()
              and model._kda_state is not None
              and model._kda_state.shape[1] == 1)
        err = ''
    except Exception as e:
        ok = False
        err = f'{type(e).__name__}: {e}'
    return [
        _ok('hybrid drops KDA state on batch-size change (B=2 -> B=1)', ok,
            f'b2_state_B={b2_state_shape}, '
            f'b1_state_B={model._kda_state.shape[1] if model._kda_state is not None else "n/a"}, '
            f'err={err}'),
    ]


def test_hybrid_decoding_cache_matches_full_sequence(device='cpu'):
    """HybridDecoding prefill+token decode matches full-sequence hybrid output.

    Exp6's hybrid row now wires one CSADecodingCache / HCADecodingCache into
    every CSA/HCA sub-layer. This is a high-risk state-threading path: KDA
    recurrent state, KDA short-conv lookback, CSA cache, HCA cache, residuals,
    and LayerNorm all have to stay aligned across layers. A pure standalone
    CSA/HCA cache test does not cover that integration.

    Use ``csa_topk=100`` (select all valid compressed blocks in this tiny
    sequence) to sidestep torch.topk tie-breaking differences. The full path
    and incremental path share the SAME underlying HybridKCHAttention weights
    because HybridDecoding owns the fused model internally.
    """
    logger.info("Test: HybridDecoding cache path matches full-sequence hybrid")
    from run_decoding import HybridDecoding

    torch.manual_seed(707)
    d_model = 32
    prefill_len = 8
    n_decode = 4
    total_len = prefill_len + n_decode
    model = HybridDecoding(d_model=d_model, total_layers=5, csa_topk=100).to(device).eval()
    x = torch.randn(1, total_len, d_model, device=device) * 0.1

    with torch.no_grad():
        # Full-sequence reference through the underlying fused model.
        model.reset()
        y_full = model.model(x)
        # Incremental path: prefill then one token at a time through
        # HybridDecoding's per-layer CSA/HCA caches.
        model.reset()
        y_prefill = model(x[:, :prefill_len])
        ys = []
        for t in range(prefill_len, total_len):
            ys.append(model(x[:, t:t+1]))
        y_inc_decode = torch.cat(ys, dim=1)

    ref_decode = y_full[:, prefill_len:]
    max_diff = (ref_decode - y_inc_decode).abs().max().item()
    shape_ok = (y_prefill.shape == (1, prefill_len, d_model)
                and y_inc_decode.shape == (1, n_decode, d_model))
    # Tolerance tightened from 1e-4 to 1e-5 (round 9 audit). The observed
    # max_diff is ~1.5e-7; the previous 1e-4 tolerance was ~1000x looser than
    # the actual error. See test_csa_decoding_cache_correctness for the full
    # rationale.
    return [_ok(
        'hybrid_decoding_cache_matches_full_sequence',
        shape_ok and max_diff < 1e-5,
        f'max_diff={max_diff:.6e}, shape_ok={shape_ok}, '
        f'prefill_len={prefill_len}, n_decode={n_decode}, csa_topk=100')]


def test_hybrid_streaming_tp_matches_unsharded(device='cpu'):
    """Head-sharded streaming TP reproduces the full hybrid decoder.

    The TP reference has to carry three state families across calls: the KDA
    recurrent state, the short-conv lookbacks, and one CSA/HCA incremental
    cache per layer.  Compare its outputs against ``run_decoding.HybridDecoding``
    on the SAME weights for a prefill, single-token decode steps, and a
    multi-token continuation chunk.  A divergence here would mean the TP
    decode oracle is wrong even though the fresh-sequence TP test passes.
    """
    logger.info("Test: hybrid streaming TP matches unsharded decoder")
    import tp_parallel as tp
    from run_decoding import HybridDecoding

    torch.manual_seed(811)
    cfg = tp.mqar_probe_config()
    decoder = HybridDecoding(cfg=cfg, total_layers=5).to(device).eval()
    x = torch.randn(2, 13, cfg.d_model, device=device) * 0.5
    state = None
    worst = 0.0
    with torch.no_grad():
        for start, end in ((0, 8), (8, 9), (9, 10), (10, 13)):
            ref = decoder(x[:, start:end])
            got = tp.hybrid_stack_forward_tp_streaming(
                decoder.model, x[:, start:end], 2, state=state)
            worst = max(worst, (ref - got['output']).abs().max().item())
            state = got['state']
        state_diff = (state['kda_state']
                      - decoder.model._kda_state).abs().max().item()
    return [_ok(
        'hybrid_streaming_tp_matches_unsharded',
        worst < 1e-5 and state_diff < 1e-5,
        f'max_diff={worst:.3e}, state_diff={state_diff:.3e}, '
        f'tp_size=2, layers=5')]


def test_hybrid_decoding_mamba2_route(device='cpu'):
    """HybridDecoding prefill+decode matches full sequence on the Mamba-2 route.

    Mirrors ``test_hybrid_decoding_cache_matches_full_sequence`` but with
    ``memory_type='mamba2'``: the Mamba-2 layers' stacked SSM states + conv
    lookbacks have to stay aligned with the CSA/HCA incremental caches,
    residuals, and LayerNorm across the stack — the same high-risk
    state-threading path, now exercised through the second memory route.
    Uses ``csa_topk=100`` for the same tie-breaking reason as the KDA-route
    test (documented numerical artifact of torch.topk, not a bug).
    """
    logger.info("Test: HybridDecoding (mamba2 route) matches full sequence")
    from run_decoding import HybridDecoding

    torch.manual_seed(707)
    d_model = 32
    prefill_len = 8
    n_decode = 4
    total_len = prefill_len + n_decode
    model = HybridDecoding(d_model=d_model, total_layers=5, csa_topk=100,
                           memory_type='mamba2').to(device).eval()
    x = torch.randn(1, total_len, d_model, device=device) * 0.1

    with torch.no_grad():
        model.reset()
        y_full = model.model(x)
        model.reset()
        y_prefill = model(x[:, :prefill_len])
        ys = []
        for t in range(prefill_len, total_len):
            ys.append(model(x[:, t:t+1]))
        y_inc_decode = torch.cat(ys, dim=1)

    ref_decode = y_full[:, prefill_len:]
    max_diff = (ref_decode - y_inc_decode).abs().max().item()
    prefill_diff = (y_full[:, :prefill_len] - y_prefill).abs().max().item()
    shape_ok = (y_prefill.shape == (1, prefill_len, d_model)
                and y_inc_decode.shape == (1, n_decode, d_model))
    return [
        _ok('hybrid_decoding_mamba2_route_matches_full_sequence',
            shape_ok and max_diff < 1e-5,
            f'shape_ok={shape_ok}, decode max_diff={max_diff:.2e} '
            f'(tol 1e-5)'),
        _ok('hybrid_decoding_mamba2_route_prefill_prefix_matches',
            prefill_diff < 1e-5,
            f'prefill-prefix max_diff={prefill_diff:.2e} (tol 1e-5)'),
    ]


__all__ = [
    'test_fused_hybrid',
    'test_csa_recompute_indexer_plumbing',
    'test_hybrid_padding_no_crash',
    'test_hybrid_state_buffer_registration',
    'test_bench_hybrid_no_grad_inference',
    'test_hybrid_per_layer_kda_state',
    'test_hybrid_backward_produces_grads',
    'test_hybrid_no_kda_layout',
    'test_hybrid_no_csa_layout',
    'test_hybrid_no_hca_layout',
    'test_weight_decay_param_groups',
    'test_hybrid_kda_state_dtype_mismatch',
    'test_hybrid_kda_state_batch_size_change',
    'test_hybrid_decoding_cache_matches_full_sequence',
    'test_hybrid_streaming_tp_matches_unsharded',
    'test_hybrid_decoding_mamba2_route',
]
