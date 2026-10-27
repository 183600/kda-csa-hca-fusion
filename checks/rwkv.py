"""Correctness checks: rwkv."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_rwkv7_scan_reference_properties(device='cpu'):
    """Algebraic semantics of the reference recurrence (layout + decay).

    With ``kk = 0`` (no low-rank correction) and ``log_w = 0`` (decay
    1), the recurrence must reduce to ``S_t = S_0 + sum_{s<=t} k_s
    \\otimes v_s`` with the readout ``y_t = S_t^T r_t`` — this pins the
    [K, V] state orientation (rows = key dim, columns = value dim).
    A strongly negative log-decay on ONE key channel must zero that
    channel's ROW of the state (the decay acts on the key index), not a
    column — the exact convention an accelerated-backend adapter could
    silently transpose.
    """
    logger.info("Test: RWKV-7 reference scan algebraic properties")
    results = []
    torch.manual_seed(11)
    B, T, H, K, V = 1, 5, 2, 3, 4
    r = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)
    kk = torch.zeros(B, T, H, K, device=device)
    a = torch.zeros(B, T, H, K, device=device)
    log_w = torch.zeros(B, T, H, K, device=device)
    S0 = torch.randn(B, H, K, V, device=device)

    o, S = rwkv7_reference_scan(r, log_w, k, v, kk, a,
                                initial_state=S0,
                                output_final_state=True)
    max_err = 0.0
    S_exp = S0.clone()
    for t in range(T):
        S_exp = S_exp + k[:, t][..., None] * v[:, t][..., None, :]
        y_exp = (r[:, t][..., None] * S_exp).sum(dim=-2)
        max_err = max(max_err, (o[:, t] - y_exp).abs().max().item())
    results.append(_ok(
        'kk=0, decay=1: S_t = S_0 + sum k⊗v, y = S^T r (fp32 tol 1e-6)',
        max_err < 1e-6, f'max_err={max_err:.2e}'))
    results.append(_ok(
        'final state matches the closed form',
        (S - S_exp).abs().max().item() < 1e-6,
        f'max_err={(S - S_exp).abs().max().item():.2e}'))

    # Decay orientation: a strong decay on key channel 1 zeroes ROW 1.
    # k is zeroed here so the k⊗v write (which refreshes EVERY row each
    # step) does not mask the decay effect — with k = kk = 0 the state
    # evolves purely under the diagonal decay.
    log_w2 = torch.zeros(B, T, H, K, device=device)
    log_w2[..., 1] = -30.0          # ~exp(-30) ~ 1e-13 decay on channel 1
    k_zero = torch.zeros_like(k)
    S0b = torch.randn(B, H, K, V, device=device)
    with torch.no_grad():
        _, S2 = rwkv7_reference_scan(r, log_w2, k_zero, v, kk, a,
                                     initial_state=S0b,
                                     output_final_state=True)
    row_decayed = S2[:, :, 1, :].abs().max().item()
    rows_alive = max(S2[:, :, 0, :].abs().max().item(),
                     S2[:, :, 2, :].abs().max().item())
    results.append(_ok(
        'decay acts on the KEY dim (rows of [K, V] state)',
        row_decayed < 1e-8 and rows_alive > 0.1,
        f'row1={row_decayed:.2e}, rows0/2={rows_alive:.3f}'))
    return results


def test_rwkv7_geometry_validation(device='cpu'):
    """Config/geometry validation fires with actionable errors."""
    logger.info("Test: RWKV-7 geometry validation")
    results = []
    geo = validate_rwkv7_geometry(32, 16, 4, 4, 8)
    results.append(_ok(
        'geometry derived', geo == {'n_heads': 2, 'head_dim': 16,
                                    'decay_lora': 4, 'a_lora': 4,
                                    'gate_lora': 8}, str(geo)))
    for name, kwargs, err in [
            ('head_dim not dividing d_model', dict(rwkv_head_dim=5),
             ValueError),
            ('zero head_dim', dict(rwkv_head_dim=0), ValueError),
            ('zero decay_lora', dict(rwkv_decay_lora=0), ValueError),
            ('lora wider than d_model', dict(rwkv_gate_lora=64),
             ValueError),
    ]:
        try:
            _rwkv7_small_cfg(**kwargs)
            results.append(_ok(f'rejects {name}', False, 'no error raised'))
        except err:
            results.append(_ok(f'rejects {name}', True))
        except Exception as e:
            results.append(_ok(f'rejects {name}', False,
                               f'{type(e).__name__}: {e}'))
    return results


def test_rwkv7_forward_smoke(device='cpu'):
    """Shape / dtype / state contract of the mixer's forward paths."""
    logger.info("Test: RWKV-7 mixer forward smoke")
    results = []
    torch.manual_seed(12)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    B, T, d = 2, 17, 32
    x = torch.randn(B, T, d, device=device)
    with torch.no_grad():
        out, state, shift = mixer.forward_functional(x, None, None)
    results.append(_ok('output shape [B, T, d]',
                       tuple(out.shape) == (B, T, d), str(tuple(out.shape))))
    results.append(_ok('output dtype preserved', out.dtype == x.dtype,
                       str(out.dtype)))
    H, K = mixer.n_heads, mixer.head_dim
    results.append(_ok('matrix state shape [B, H, K, V]',
                       tuple(state.shape) == (B, H, K, K),
                       str(tuple(state.shape))))
    results.append(_ok('matrix state in compute precision (fp32)',
                       state.dtype == torch.float32, str(state.dtype)))
    results.append(_ok('shift state shape [B, 1, d]',
                       tuple(shift.shape) == (B, 1, d),
                       str(tuple(shift.shape))))
    results.append(_ok('output finite', bool(torch.isfinite(out).all())))
    # o_proj is zero-initialized upstream (identity-in-residual at init):
    # the OUTPUT is exactly zero at init BY DESIGN — assert it so a future
    # accidental non-zero init is caught as a deviation from the binding.
    results.append(_ok('zero output at init (upstream o_proj zero-init)',
                       float(out.abs().max()) == 0.0,
                       f'max|out|={float(out.abs().max()):.2e}'))
    # stateful forward persists the shift state and matches the
    # functional path.
    mixer.reset_conv_state()
    with torch.no_grad():
        out_stateful, state2 = mixer(x)
    results.append(_ok('stateful forward matches functional forward',
                       torch.equal(out_stateful, out)
                       and torch.equal(state2, state)))
    results.append(_ok('stateful forward persisted the shift state',
                       mixer._shift_state is not None
                       and torch.equal(mixer._shift_state, shift)))
    # reset clears the streaming state (the stack's reset_state contract).
    mixer.reset_conv_state()
    results.append(_ok('reset_conv_state clears the shift state',
                       mixer._shift_state is None))
    return results


def test_rwkv7_prefill_step_matches_full(device='cpu'):
    """Full forward == prefill + single-token decode, within tolerance."""
    logger.info("Test: RWKV-7 prefill + step decode matches full forward")
    results = []
    torch.manual_seed(1)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    # Break the zero-init so the comparison is non-trivial (see the smoke
    # test: o_proj starts at zero by upstream design).
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, T, n_extra, d = 2, 33, 6, 32
    x = torch.randn(B, T, d, device=device)
    xt = torch.randn(B, n_extra, d, device=device)
    tol = 1e-5
    with torch.no_grad():
        out_full, _, _ = mixer.forward_functional(
            torch.cat([x, xt], dim=1), None, None)
        out_pre, state = mixer.prefill(x)
        diff_pre = (out_pre - out_full[:, :T]).abs().max().item()
        results.append(_ok('prefill matches full forward prefix',
                           diff_pre < tol, f'maxdiff={diff_pre:.2e}'))
        steps = []
        for t in range(n_extra):
            o_t, state = mixer.step(xt[:, t:t + 1], state)
            steps.append(o_t)
        out_steps = torch.cat(steps, dim=1)
        diff = (out_steps - out_full[:, T:]).abs().max().item()
        results.append(_ok('prefill+step decode matches full forward',
                           diff < tol, f'maxdiff={diff:.2e}'))
        # prefill/step state is detached (inference lifecycle contract).
        results.append(_ok('inference protocol state detached',
                           not state['matrix_state'].requires_grad
                           and not state['shift_state'].requires_grad))
    # step() rejects multi-token input (protocol ambiguity guard).
    try:
        mixer.step(xt, None)
        results.append(_ok('step rejects T>1 input', False, 'no error'))
    except ValueError:
        results.append(_ok('step rejects T>1 input', True))
    return results


def test_rwkv7_segmented_prefill(device='cpu'):
    """One-shot prefill == arbitrary-chunk segmented prefill.

    The chunk boundaries deliberately include a length-1 chunk, an empty
    chunk, and non-block-length boundaries (the acceptance list's
    'non-zero initial state, non-block-length multiples, length-1
    inputs' — the boundary cases where a shift/state handoff bug would
    show).
    """
    logger.info("Test: RWKV-7 segmented prefill")
    results = []
    torch.manual_seed(2)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, T, d = 2, 37, 32
    x = torch.randn(B, T, d, device=device)
    tol = 1e-5
    with torch.no_grad():
        out_full, s_full, c_full = mixer.forward_functional(x, None, None)
        outs, s_prev, c_prev = [], None, None
        for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 5), (5, 20), (20, 37)]:
            o, s_prev, c_prev = mixer.forward_functional(
                x[:, lo:hi], s_prev, c_prev)
            outs.append(o)
        out_seg = torch.cat(outs, dim=1)
        diff = (out_seg - out_full).abs().max().item()
        results.append(_ok('segmented prefill output matches one-shot',
                           diff < tol, f'maxdiff={diff:.2e}'))
        diff_s = (s_prev - s_full).abs().max().item()
        results.append(_ok('segmented final matrix state matches one-shot',
                           diff_s < tol, f'maxdiff={diff_s:.2e}'))
        diff_c = (c_prev - c_full).abs().max().item()
        results.append(_ok('segmented final shift state matches one-shot',
                           diff_c < tol, f'maxdiff={diff_c:.2e}'))
    # Token-by-token (every chunk length 1) is the degenerate case.
    with torch.no_grad():
        outs = []
        s_prev = c_prev = None
        for t in range(T):
            o, s_prev, c_prev = mixer.forward_functional(
                x[:, t:t + 1], s_prev, c_prev)
            outs.append(o)
        out_tok = torch.cat(outs, dim=1)
        diff_tok = (out_tok - out_full).abs().max().item()
        results.append(_ok('token-by-token forward matches one-shot',
                           diff_tok < tol, f'maxdiff={diff_tok:.2e}'))
    return results


def test_rwkv7_causality(device='cpu'):
    """Perturbing future tokens leaves past outputs unchanged."""
    logger.info("Test: RWKV-7 causality")
    results = []
    torch.manual_seed(3)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, T, d = 2, 28, 32
    x = torch.randn(B, T, d, device=device)
    with torch.no_grad():
        out1, _, _ = mixer.forward_functional(x, None, None)
        for cut in (1, 7, T - 1):
            x2 = x.clone()
            x2[:, cut:] += 5.0
            out2, _, _ = mixer.forward_functional(x2, None, None)
            diff = (out2[:, :cut] - out1[:, :cut]).abs().max().item()
            results.append(_ok(
                f'future perturbation at t>={cut} leaves outputs[:{cut}] '
                f'unchanged', diff < 1e-6, f'maxdiff={diff:.2e}'))
    return results


def test_rwkv7_shift_boundary_cases(device='cpu'):
    """T == 0 preserves the state; length-0 chunks compose."""
    logger.info("Test: RWKV-7 shift/state boundary cases")
    results = []
    torch.manual_seed(4)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, d = 2, 32
    x = torch.randn(B, 9, d, device=device)
    with torch.no_grad():
        out_full, s_full, _ = mixer.forward_functional(x, None, None)
        # Chunks including an empty one and single tokens.
        outs, s_prev, c_prev = [], None, None
        bounds = [(0, 0), (0, 1), (1, 1), (1, 3), (3, 3), (3, 4), (4, 9)]
        for lo, hi in bounds:
            o, s_prev, c_prev = mixer.forward_functional(
                x[:, lo:hi], s_prev, c_prev)
            outs.append(o)
        out_seg = torch.cat(outs, dim=1)
        diff = (out_seg - out_full).abs().max().item()
        results.append(_ok(
            'empty + single-token chunks compose to one-shot',
            diff < 1e-5, f'maxdiff={diff:.2e}'))
        # Direct T=0 call preserves a provided state.
        out0, s0, c0 = mixer.forward_functional(x[:, :0], s_prev, c_prev)
        results.append(_ok('T=0 returns empty output',
                           tuple(out0.shape) == (B, 0, d)))
        diff_s0 = (s0 - s_prev).abs().max().item()
        results.append(_ok('T=0 preserves the matrix state',
                           diff_s0 == 0.0, f'maxdiff={diff_s0:.2e}'))
        diff_c0 = (c0 - c_prev).abs().max().item()
        results.append(_ok('T=0 preserves the shift state',
                           diff_c0 == 0.0, f'maxdiff={diff_c0:.2e}'))
    return results


def test_rwkv7_gradient(device='cpu'):
    """Backward produces finite grads for every mixer parameter.

    (Gradients MAY be exactly zero for the upstream-init layers — o_proj
    is zero-initialized, so upstream parameters receive exact-zero
    gradients until o_proj moves off zero; zero is finite and the
    contract is grad-not-None.)
    """
    logger.info("Test: RWKV-7 gradient flow")
    results = []
    torch.manual_seed(5)
    mixer = Rwkv7Mixer(_rwkv7_small_cfg()).to(device)
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    x = torch.randn(2, 16, 32, device=device)
    out, _ = mixer(x)
    out.square().mean().backward()
    missing = [n for n, p in mixer.named_parameters()
               if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in mixer.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    results.append(_ok('every parameter receives a gradient',
                       not missing, str(missing)))
    results.append(_ok('all gradients finite', not nonfinite, str(nonfinite)))
    # o_proj itself must receive a NON-zero gradient (its input is
    # non-zero), otherwise the zero-init would be a dead layer.
    results.append(_ok(
        'o_proj gradient non-zero (lifts the zero init)',
        mixer.o_proj.weight.grad is not None
        and float(mixer.o_proj.weight.grad.abs().max()) > 0.0))
    return results


def test_rwkv7_state_save_load(device='cpu'):
    """Saving and loading weights reproduces outputs exactly."""
    logger.info("Test: RWKV-7 weight save/load roundtrip")
    results = []
    torch.manual_seed(6)
    mixer_a = Rwkv7Mixer(_rwkv7_small_cfg()).to(device).eval()
    mixer_b = Rwkv7Mixer(_rwkv7_small_cfg()).to(device).eval()
    # Non-trivial weights (upstream zero-inits o_proj; randomize it so the
    # comparison is not vacuously zero).
    with torch.no_grad():
        mixer_a.o_proj.weight.normal_(0.0, 0.02)
        mixer_b.o_proj.weight.normal_(0.0, 0.02)
    x = torch.randn(2, 15, 32, device=device)
    with torch.no_grad():
        out_a, s_a, c_a = mixer_a.forward_functional(x, None, None)
    # Sanity: the two random inits must DIFFER (otherwise the roundtrip
    # check is vacuous).
    diff_init = (out_a - mixer_b.forward_functional(x, None, None)[0]) \
        .abs().max().item()
    results.append(_ok('independent inits differ (non-vacuous check)',
                       diff_init > 1e-6, f'maxdiff={diff_init:.2e}'))
    mixer_b.load_state_dict(mixer_a.state_dict())
    with torch.no_grad():
        out_b, s_b, c_b = mixer_b.forward_functional(x, None, None)
    results.append(_ok('outputs bit-identical after state_dict roundtrip',
                       torch.equal(out_a, out_b)))
    results.append(_ok('matrix state identical after roundtrip',
                       torch.equal(s_a, s_b)))
    results.append(_ok('shift state identical after roundtrip',
                       torch.equal(c_a, c_b)))
    return results


def test_rwkv7_state_request_independence(device='cpu'):
    """Two interleaved requests keep independent, unpolluted state."""
    logger.info("Test: RWKV-7 per-request state independence")
    results = []
    torch.manual_seed(7)
    mixer = Rwkv7Mixer(_rwkv7_small_cfg()).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    d = 32
    xa = torch.randn(2, 12, d, device=device)
    xb = torch.randn(3, 12, d, device=device)   # DIFFERENT batch size
    with torch.no_grad():
        # Interleave: A prefill, B prefill, A step, B step...
        out_a, st_a = mixer.prefill(xa[:, :8])
        out_b, st_b = mixer.prefill(xb[:, :8])
        o_a, st_a = mixer.step(xa[:, 8:9], st_a)
        o_b, st_b = mixer.step(xb[:, 8:9], st_b)
        o_a2, st_a = mixer.step(xa[:, 9:10], st_a)
        o_b2, st_b = mixer.step(xb[:, 9:10], st_b)
        out_a_steps = torch.cat([o_a, o_a2], dim=1)
        out_b_steps = torch.cat([o_b, o_b2], dim=1)
        # Full references for each request independently.
        ref_a, _, _ = mixer.forward_functional(xa[:, :10], None, None)
        ref_b, _, _ = mixer.forward_functional(xb[:, :10], None, None)
        tol = 1e-5
        diff_a = (out_a_steps - ref_a[:, 8:10]).abs().max().item()
        diff_b = (out_b_steps - ref_b[:, 8:10]).abs().max().item()
        results.append(_ok('request A state unpolluted by request B',
                           diff_a < tol, f'maxdiff={diff_a:.2e}'))
        results.append(_ok('request B state unpolluted by request A',
                           diff_b < tol, f'maxdiff={diff_b:.2e}'))
    # Analytic state accounting (the state_nbytes helper mirrors
    # Mamba2Mixer.state_nbytes; the fp32 figures feed run_kv_cache's
    # analytic rows and the cache audit).
    nb = mixer.state_nbytes(batch=2)
    B, H, K, dmod = 2, mixer.n_heads, mixer.head_dim, mixer.d_model
    results.append(_ok(
        'state_nbytes matches the analytic element counts',
        nb['matrix_state_bytes'] == B * H * K * K * 4
        and nb['shift_state_bytes'] == B * dmod * 4
        and nb['total_bytes'] == (B * H * K * K + B * dmod) * 4,
        str(nb)))
    return results


def test_rwkv7_nonzero_initial_state(device='cpu'):
    """A non-zero initial state participates in forward AND backward."""
    logger.info("Test: RWKV-7 non-zero initial state")
    results = []
    torch.manual_seed(8)
    mixer = Rwkv7Mixer(_rwkv7_small_cfg()).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, T, d = 2, 12, 32
    x = torch.randn(B, T, d, device=device)
    H, K = mixer.n_heads, mixer.head_dim
    S0 = torch.randn(B, H, K, K, device=device)
    with torch.no_grad():
        out_zero, _, _ = mixer.forward_functional(x, None, None)
        out_nonzero, _, _ = mixer.forward_functional(x, S0, None)
    diff = (out_zero - out_nonzero).abs().max().item()
    results.append(_ok('non-zero initial state changes the output',
                       diff > 1e-6, f'maxdiff={diff:.2e}'))
    # Backward: gradients flow into the initial state (cross-chunk BPTT
    # contract) and remain finite.
    S0g = S0.clone().requires_grad_(True)
    out, new_state, _ = mixer.forward_functional(x, S0g, None)
    out.square().mean().backward()
    results.append(_ok('gradient reaches the non-zero initial state',
                       S0g.grad is not None
                       and float(S0g.grad.abs().max()) > 0.0))
    results.append(_ok('initial-state gradient finite',
                       S0g.grad is not None
                       and bool(torch.isfinite(S0g.grad).all())))
    return results


def test_hybrid_stack_rwkv7_route(device='cpu'):
    """Ratio + explicit layouts, state stacking, and the RWKV-7 route."""
    logger.info("Test: hybrid stack RWKV-7 route")
    results = []
    torch.manual_seed(9)
    cfg = _rwkv7_small_cfg(n_heads_qk=2, n_heads_v=2,
                           head_dim_k=16, head_dim_v=16)
    m = HybridAttentionStack(cfg, total_layers=5).to(device).eval()
    results.append(_ok(
        'rwkv7 ratio-mode layout',
        m.layout == ['rwkv7'] * 3 + ['csa', 'hca'], str(m.layout)))
    results.append(_ok('rwkv7 layout_str',
                       m.layout_str() == 'RWKV7-RWKV7-RWKV7-CSA-HCA',
                       m.layout_str()))
    results.append(_ok('n_memory_layers counts rwkv7 layers',
                       m.n_memory_layers == 3 and m.n_kda_layers == 3))
    x = torch.randn(2, 11, 32, device=device)
    with torch.no_grad():
        m.reset_state()
        out = m(x)
        results.append(_ok('rwkv7 stack forward shape',
                           tuple(out.shape) == (2, 11, 32),
                           str(tuple(out.shape))))
        first_mem = m.memory_layers[0]
        expected_state = (3, 2, first_mem.n_heads, first_mem.head_dim,
                          first_mem.head_dim)
        results.append(_ok(
            'rwkv7 stacked memory state shape',
            m._kda_state is not None
            and tuple(m._kda_state.shape) == expected_state,
            str(None if m._kda_state is None
                else tuple(m._kda_state.shape))))
    # Stateful == functional on the same stack.
    with torch.no_grad():
        m.reset_state()
        out_stateful = m(x)
        m.reset_state()
        out_fn, new_state, _ = m.forward_functional(
            x, None, [None] * m.n_memory_layers)
        m._kda_state = new_state
    results.append(_ok('stateful == functional forward (rwkv7 stack)',
                       torch.allclose(out_stateful, out_fn, atol=1e-6),
                       f'maxdiff={(out_stateful - out_fn).abs().max():.2e}'))
    # Explicit 10-layer matrix layout (the eight-group design's depth).
    layout = ('memory', 'memory', 'memory', 'csa', 'hca') * 2
    cfg_lay = _rwkv7_small_cfg(layout=layout)
    results.append(_ok(
        'cfg.layout resolved to the concrete rwkv7 kinds',
        cfg_lay.layout == ('rwkv7', 'rwkv7', 'rwkv7', 'csa', 'hca') * 2,
        str(cfg_lay.layout)))
    m_lay = HybridAttentionStack(cfg_lay, total_layers=10).to(device).eval()
    results.append(_ok('explicit-layout stack depth', len(m_lay.layers) == 10))
    results.append(_ok('explicit-layout memory count', m_lay.n_memory_layers == 6))
    with torch.no_grad():
        m_lay.reset_state()
        out = m_lay(torch.randn(2, 7, 32, device=device))
        results.append(_ok('explicit-layout stack forward',
                           tuple(out.shape) == (2, 7, 32)))
    # Depth-dependent init: two RWKV layers at different positions must
    # receive DIFFERENT init constants (the upstream recipe is
    # layer_idx-dependent).
    lay_params = [dict(p.named_parameters()) for p in
                  [l for l, k in zip(m_lay.layers, m_lay.layout)
                   if k == 'rwkv7']]
    results.append(_ok(
        'depth-dependent init differs across RWKV layers',
        not torch.equal(lay_params[0]['x_r'], lay_params[1]['x_r'])))
    # Layout validation error paths (route purity generalization).
    for name, kwargs, err in [
            ('mixed memory kinds (kda + rwkv7)',
             dict(layout=('kda', 'rwkv7')), ValueError),
            ('mixed memory kinds (mamba2 + rwkv7)',
             dict(layout=('mamba2', 'rwkv7')), ValueError),
            ('unknown kind', dict(layout=('rwkv7', 'softmax')), ValueError),
            ('bad memory_type', dict(memory_type='rwkv6'), ValueError),
    ]:
        try:
            cfg_bad = HybridConfig(
                d_model=32, n_heads_qk=2, n_heads_v=2,
                head_dim_k=16, head_dim_v=16, **kwargs)
            HybridAttentionStack(cfg_bad, total_layers=5)
            results.append(_ok(f'rejects {name}', False, 'no error raised'))
        except err:
            results.append(_ok(f'rejects {name}', True))
        except Exception as e:
            results.append(_ok(f'rejects {name}', False,
                               f'{type(e).__name__}: {e}'))
    # Mixer factory follows the route.
    results.append(_ok("build_mixer('memory') follows memory_type=rwkv7",
                       isinstance(build_mixer('memory', cfg), Rwkv7Mixer)))
    # Parameter counts differ across routes (must be REPORTED, not assumed
    # equal — the budget-matching experiment exists precisely for this).
    cfg_kda = HybridConfig(d_model=32, n_heads_qk=2, n_heads_v=2,
                           head_dim_k=16, head_dim_v=16)
    m_kda = HybridAttentionStack(cfg_kda, total_layers=5).to(device).eval()
    n_params_kda = sum(p.numel() for p in m_kda.parameters())
    n_params_rwkv = sum(p.numel() for p in m.parameters())
    results.append(_ok('kda vs rwkv7 param counts recorded (not assumed)',
                       n_params_kda > 0 and n_params_rwkv > 0
                       and n_params_kda != n_params_rwkv,
                       f'kda={n_params_kda}, rwkv7={n_params_rwkv}'))
    return results


def test_hybrid_decoding_rwkv7_route(device='cpu'):
    """HybridDecoding prefill+decode matches full sequence on the RWKV-7 route.

    Mirrors ``test_hybrid_decoding_mamba2_route``: the RWKV-7 layers'
    stacked matrix states + shift lookbacks must stay aligned with the
    CSA/HCA incremental caches, residuals, and LayerNorms across the
    stack. Uses ``csa_topk=100`` for the same torch.topk tie-breaking
    reason as the KDA-route test.
    """
    logger.info("Test: HybridDecoding (rwkv7 route) matches full sequence")
    from run_decoding import HybridDecoding

    torch.manual_seed(708)
    d_model = 32
    prefill_len = 8
    n_decode = 4
    total_len = prefill_len + n_decode
    model = HybridDecoding(d_model=d_model, total_layers=5, csa_topk=100,
                           memory_type='rwkv7').to(device).eval()
    with torch.no_grad():
        for layer in model.model.memory_layers:
            layer.o_proj.weight.normal_(0.0, 0.02)
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
        _ok('hybrid_decoding_rwkv7_route_matches_full_sequence',
            shape_ok and max_diff < 1e-5,
            f'shape_ok={shape_ok}, decode max_diff={max_diff:.2e} '
            f'(tol 1e-5)'),
        _ok('hybrid_decoding_rwkv7_route_prefill_prefix_matches',
            prefill_diff < 1e-5,
            f'prefill-prefix max_diff={prefill_diff:.2e} (tol 1e-5)'),
    ]


def test_rwkv7_backend_selection_contract(device='cpu'):
    """Backend selection resolves/records correctly and errors loudly."""
    logger.info("Test: RWKV-7 backend selection contract")
    results = []
    from ops_rwkv_backend import validate_rwkv_backend
    try:
        validate_rwkv_backend('tensorrt')
        results.append(_ok('unknown rwkv_backend rejected', False,
                           'no error'))
    except ValueError:
        results.append(_ok('unknown rwkv_backend rejected', True))
    if torch.cuda.is_available() and rwkv7_fla_available():
        results.append(_ok('fla backend available (CUDA box)', True))
    else:
        try:
            cfg = _rwkv7_small_cfg(rwkv_backend='fla')
            build_rwkv7_mixer(cfg)
            results.append(_ok(
                "build_rwkv7_mixer('fla') errors when unavailable",
                False, 'no error'))
        except (ImportError, RuntimeError):
            results.append(_ok(
                "build_rwkv7_mixer('fla') errors when unavailable",
                True))
        # resolve records the would-be resolution without raising.
        results.append(_ok(
            "resolve_rwkv_backend('fla') degrades to 'reference'",
            resolve_rwkv_backend('fla') == 'reference'))
    # 'auto' falls back with a recorded marker on this box.
    cfg_auto = _rwkv7_small_cfg(rwkv_backend='auto')
    mixer_auto = build_rwkv7_mixer(cfg_auto)
    expected = ('fla' if (torch.cuda.is_available()
                          and rwkv7_fla_available())
                else 'reference (auto fallback)')
    results.append(_ok("auto backend resolution recorded",
                       mixer_auto.effective_backend == expected,
                       mixer_auto.effective_backend))
    # The reference request resolves to the reference mixer.
    mixer_ref = build_rwkv7_mixer(_rwkv7_small_cfg(rwkv_backend='reference'))
    results.append(_ok("reference backend resolution recorded",
                       mixer_ref.effective_backend == 'reference',
                       mixer_ref.effective_backend))
    return results


def test_rwkv7_fla_backend_parity(device='cpu'):
    """Reference vs FLA outputs/states match, when installed.

    Skipped (recorded as PASS-with-note) when the optional
    flash-linear-attention with ``fla.ops.rwkv7`` or CUDA is unavailable
    — mirroring how the KDA FLA and Mamba-2 official backends are
    handled. When available, the SAME mixer is run through both the
    reference scan and the FLA kernels (the decay-contract and state
    layout are shared by construction — see ops_rwkv_backend) and the
    outputs / final states must agree within a kernel-reduction-order
    tolerance.
    """
    logger.info("Test: RWKV-7 FLA backend parity")
    results = []
    if device != 'cuda' or not rwkv7_fla_available():
        results.append(_ok(
            'fla backend parity (SKIPPED: needs CUDA + fla.ops.rwkv7)',
            True,
            f"device={device}, fla_available={rwkv7_fla_available()}"))
        return results
    torch.manual_seed(10)
    cfg = _rwkv7_small_cfg()
    mixer = Rwkv7Mixer(cfg).to(device).eval()
    with torch.no_grad():
        mixer.o_proj.weight.normal_(0.0, 0.02)
    B, T, d = 2, 40, 32
    x = torch.randn(B, T, d, device=device)
    with torch.no_grad():
        out_ref, s_ref, c_ref = mixer.forward_functional(x, None, None)
        mixer.enable_fla_backend()
        out_fla, s_fla, c_fla = mixer.forward_functional(x, None, None)
    tol = 1e-4
    diff = (out_ref.float() - out_fla.float()).abs().max().item()
    results.append(_ok('fla backend output parity (fp32)',
                       diff < tol, f'maxdiff={diff:.2e}'))
    diff_s = (s_ref.float() - s_fla.float()).abs().max().item()
    results.append(_ok('fla backend matrix state parity',
                       diff_s < tol, f'maxdiff={diff_s:.2e}'))
    diff_c = (c_ref - c_fla).abs().max().item()
    results.append(_ok('fla backend shift state parity (pass-through)',
                       diff_c == 0.0, f'maxdiff={diff_c:.2e}'))
    # Decode: the fused recurrent kernel path (T == 1).
    with torch.no_grad():
        _, st_ref = mixer.prefill(x[:, :30])
        mixer.backend = 'reference'
        o1_ref, st1 = mixer.step(x[:, 30:31], st_ref)
        mixer.backend = 'fla'
        o1_fla, st1f = mixer.step(x[:, 30:31], st_ref)
    diff_step = (o1_ref.float() - o1_fla.float()).abs().max().item()
    results.append(_ok('fla single-step decode parity',
                       diff_step < tol, f'maxdiff={diff_step:.2e}'))
    return results


def test_rwkv7_cache_live_accounting(device='cpu'):
    """Live RWKV-7 decode state reconciles with the analytic accounting.

    Mirrors the KDA route's cache audit: after a prefill + decode steps
    on the RWKV-7-route HybridDecoding clone, the live walker must
    attribute the matrix / shift states to the RWKV-7 components (with
    their ACTUAL dtypes — fp32 matrix state), and the audit against
    Exp 3's closed-form formulas must be consistent.
    """
    logger.info("Test: RWKV-7 live cache accounting")
    results = []
    import cache_accounting
    from run_decoding import HybridDecoding

    torch.manual_seed(13)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2, head_dim_k=16, head_dim_v=16,
        memory_type='rwkv7', rwkv_head_dim=16, rwkv_decay_lora=4,
        rwkv_a_lora=4, rwkv_gate_lora=8, rwkv_backend='reference',
        csa_m=4, csa_topk=100, csa_nh=2, csa_c=8, csa_dc=8,
        csa_nIh=1, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=8, hca_sliding_window=4,
        n_kda=3, n_csa=1, n_hca=1)
    decoder = HybridDecoding(cfg=cfg, total_layers=5,
                             memory_type='rwkv7').to(device).eval()
    T, n_decode = 16, 3
    x = torch.randn(1, T + n_decode, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        decoder.reset()
        decoder(x[:, :T])
        for t in range(T, T + n_decode):
            decoder(x[:, t:t + 1])
    live = cache_accounting.live_cache_components(decoder)
    comps = live['components']
    results.append(_ok(
        'rwkv7_matrix_state attributed live',
        'rwkv7_matrix_state' in comps, str(sorted(comps))))
    if 'rwkv7_matrix_state' in comps:
        rec = comps['rwkv7_matrix_state']
        # The stacked [n_mem, B, H, K, V] buffer is ONE tensor; the
        # per-sequence element count is n_mem * H * K * V (B == 1 here).
        expected = 3 * 2 * 16 * 16
        results.append(_ok(
            'matrix state elements match the layout count',
            rec['elements_allocated_per_sequence'] == expected,
            f"live={rec['elements_allocated_per_sequence']}, "
            f"expected={expected}"))
        results.append(_ok(
            'matrix state dtype is fp32 (compute precision)',
            rec['dtypes'] == {'float32': 1}, str(rec['dtypes'])))
    results.append(_ok(
        'rwkv7_shift_state attributed live (3 per-layer tensors, fp32)',
        'rwkv7_shift_state' in comps
        and comps['rwkv7_shift_state']['dtypes'] == {'float32': 3},
        str(comps.get('rwkv7_shift_state'))))
    audit = cache_accounting.audit_cache_accounting(
        decoder, cfg, T=T + n_decode, live=live)
    results.append(_ok(
        'live vs analytic accounting consistent (rwkv7 route)',
        audit['consistent'], f"mismatches={audit['mismatches'][:2]}"))
    results.append(_ok(
        'audit reports the memory route',
        audit.get('memory_type') == 'rwkv7', str(audit.get('memory_type'))))
    return results


__all__ = [
    'test_rwkv7_scan_reference_properties',
    'test_rwkv7_geometry_validation',
    'test_rwkv7_forward_smoke',
    'test_rwkv7_prefill_step_matches_full',
    'test_rwkv7_segmented_prefill',
    'test_rwkv7_causality',
    'test_rwkv7_shift_boundary_cases',
    'test_rwkv7_gradient',
    'test_rwkv7_state_save_load',
    'test_rwkv7_state_request_independence',
    'test_rwkv7_nonzero_initial_state',
    'test_hybrid_stack_rwkv7_route',
    'test_hybrid_decoding_rwkv7_route',
    'test_rwkv7_backend_selection_contract',
    'test_rwkv7_fla_backend_parity',
    'test_rwkv7_cache_live_accounting',
]
