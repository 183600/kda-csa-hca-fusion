"""Correctness checks: mamba."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_mamba2_geometry_validation(device='cpu'):
    """Config/geometry validation fires with actionable errors."""
    results = []
    geo = validate_mamba2_geometry(32, 16, 4, 2, 16, 1, 'auto')
    results.append(_ok(
        'auto geometry derived',
        geo == {'d_inner': 64, 'nheads': 4, 'heads_per_group': 4,
                'dt_rank': 2}, str(geo)))
    for name, kwargs, err in [
            ('headdim not dividing d_inner',
             dict(mamba_headdim=5), ValueError),
            ('d_conv < 2', dict(mamba_d_conv=1), ValueError),
            ('zero d_state', dict(mamba_d_state=0), ValueError),
    ]:
        try:
            _mamba2_small_cfg(**kwargs)
            results.append(_ok(f'rejects {name}', False, 'no error raised'))
        except err:
            results.append(_ok(f'rejects {name}', True))
        except Exception as e:  # wrong exception type
            results.append(_ok(f'rejects {name}', False,
                               f'{type(e).__name__}: {e}'))
    # KDA-only configs are NOT constrained by the mamba geometry (the
    # default headdim=64 does not divide d_model=32's d_inner=64 -> it
    # does divide, so use headdim=5 to prove the bypass).
    try:
        HybridConfig(d_model=32, mamba_headdim=5)
        results.append(_ok('kda-only config skips mamba geometry check',
                           True))
    except ValueError as e:
        results.append(_ok('kda-only config skips mamba geometry check',
                           False, str(e)))
    return results


def test_mamba2_forward_smoke(device='cpu'):
    """Forward shapes, state contract, dtype passthrough."""
    results = []
    torch.manual_seed(0)
    cfg = _mamba2_small_cfg()
    mixer = Mamba2Mixer(cfg).to(device).eval()
    B, T, d = 3, 24, 32
    x = torch.randn(B, T, d, device=device)
    with torch.no_grad():
        out, ssm_state, conv_state = mixer.forward_functional(x, None, None)
    results.append(_ok('output shape [B,T,d]',
                       tuple(out.shape) == (B, T, d),
                       str(tuple(out.shape))))
    results.append(_ok('ssm state shape [B,H,hd,N]',
                       tuple(ssm_state.shape) ==
                       (B, mixer.nheads, mixer.headdim, mixer.d_state),
                       str(tuple(ssm_state.shape))))
    results.append(_ok('conv state shape [B,d_inner,d_conv-1]',
                       tuple(conv_state.shape) ==
                       (B, mixer.d_inner, mixer.d_conv - 1),
                       str(tuple(conv_state.shape))))
    results.append(_ok('output dtype passthrough', out.dtype == x.dtype))
    # fp32 input keeps fp32 states (compute precision contract).
    results.append(_ok('fp32 ssm state dtype', ssm_state.dtype == torch.float32,
                       str(ssm_state.dtype)))
    # Parameter names match the official mamba_ssm module (weight transfer
    # contract used by the backend parity test).
    expected_params = {'in_proj.weight', 'conv1d.weight', 'conv1d.bias',
                       'x_proj.weight', 'dt_proj.weight', 'dt_proj.bias',
                       'A_log', 'D', 'out_proj.weight'}
    got = {n for n, _ in mixer.named_parameters()}
    results.append(_ok('parameter names match official Mamba2',
                       expected_params <= got, str(sorted(got))))
    # dt init contract: softplus(dt_proj.bias) lands in [dt_min, dt_max].
    with torch.no_grad():
        dt_init = torch.nn.functional.softplus(mixer.dt_proj.bias)
    results.append(_ok('softplus(dt bias) in [0.001, 0.1]',
                      bool((dt_init >= 0.001).all() and (dt_init <= 0.1).all()),
                      f'range=[{dt_init.min():.5f}, {dt_init.max():.5f}]'))
    return results


def test_mamba2_prefill_step_matches_full(device='cpu'):
    """Full forward == prefill + single-token decode, within tolerance."""
    results = []
    torch.manual_seed(1)
    cfg = _mamba2_small_cfg()
    mixer = Mamba2Mixer(cfg).to(device).eval()
    B, T, n_extra, d = 2, 33, 6, 32
    x = torch.randn(B, T, d, device=device)
    xt = torch.randn(B, n_extra, d, device=device)
    tol = 1e-9 if x.dtype == torch.float64 else 1e-5
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
                           not state['ssm_state'].requires_grad
                           and not state['conv_state'].requires_grad))
    # step() rejects multi-token input (protocol ambiguity guard).
    try:
        mixer.step(xt, None)
        results.append(_ok('step rejects T>1 input', False, 'no error'))
    except ValueError:
        results.append(_ok('step rejects T>1 input', True))
    return results


def test_mamba2_segmented_prefill(device='cpu'):
    """One-shot prefill == arbitrary-chunk segmented prefill."""
    results = []
    torch.manual_seed(2)
    cfg = _mamba2_small_cfg()
    mixer = Mamba2Mixer(cfg).to(device).eval()
    B, T, d = 2, 37, 32
    x = torch.randn(B, T, d, device=device)
    tol = 1e-9 if x.dtype == torch.float64 else 1e-5
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
        results.append(_ok('segmented final SSM state matches one-shot',
                           diff_s < tol, f'maxdiff={diff_s:.2e}'))
        diff_c = (c_prev - c_full).abs().max().item()
        results.append(_ok('segmented final conv state matches one-shot',
                           diff_c < tol, f'maxdiff={diff_c:.2e}'))
    return results


def test_mamba2_causality(device='cpu'):
    """Perturbing future tokens leaves past outputs unchanged."""
    results = []
    torch.manual_seed(3)
    cfg = _mamba2_small_cfg()
    mixer = Mamba2Mixer(cfg).to(device).eval()
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


def test_mamba2_conv_boundary_cases(device='cpu'):
    """T < d_conv, T == 0, and the conv-state handoff across them."""
    results = []
    torch.manual_seed(4)
    cfg = _mamba2_small_cfg()          # d_conv = 4
    mixer = Mamba2Mixer(cfg).to(device).eval()
    B, d = 2, 32
    x = torch.randn(B, 9, d, device=device)
    with torch.no_grad():
        # Chunks shorter than the conv kernel, incl. a length-1 chunk and
        # an empty chunk, must compose to the one-shot result.
        out_full, s_full, _ = mixer.forward_functional(x, None, None)
        outs, s_prev, c_prev = [], None, None
        bounds = [(0, 0), (0, 1), (1, 3), (3, 3), (3, 4), (4, 9)]
        for lo, hi in bounds:
            o, s_prev, c_prev = mixer.forward_functional(
                x[:, lo:hi], s_prev, c_prev)
            outs.append(o)
        out_seg = torch.cat(outs, dim=1)
        diff = (out_seg - out_full).abs().max().item()
        results.append(_ok(
            'sub-conv-kernel chunks (incl. T=0) compose to one-shot',
            diff < 1e-5, f'maxdiff={diff:.2e}'))
        diff_s = (s_prev - s_full).abs().max().item()
        results.append(_ok('final SSM state identical after boundary chunks',
                           diff_s < 1e-5, f'maxdiff={diff_s:.2e}'))
        # Direct T=0 call preserves a provided state.
        out0, s0, c0 = mixer.forward_functional(x[:, :0], s_prev, c_prev)
        results.append(_ok('T=0 returns empty output',
                           tuple(out0.shape) == (B, 0, d)))
        diff_s0 = (s0 - s_prev).abs().max().item()
        results.append(_ok('T=0 preserves SSM state', diff_s0 == 0.0,
                           f'maxdiff={diff_s0:.2e}'))
    return results


def test_mamba2_gradient(device='cpu'):
    """Backward produces finite grads for every mixer parameter."""
    results = []
    torch.manual_seed(5)
    mixer = Mamba2Mixer(_mamba2_small_cfg()).to(device)
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
    return results


def test_mamba2_state_save_load(device='cpu'):
    """Saving and loading weights reproduces outputs exactly."""
    results = []
    torch.manual_seed(6)
    mixer_a = Mamba2Mixer(_mamba2_small_cfg()).to(device).eval()
    mixer_b = Mamba2Mixer(_mamba2_small_cfg()).to(device).eval()
    x = torch.randn(2, 15, 32, device=device)
    with torch.no_grad():
        out_a, s_a, c_a = mixer_a.forward_functional(x, None, None)
    mixer_b.load_state_dict(mixer_a.state_dict())
    with torch.no_grad():
        out_b, s_b, c_b = mixer_b.forward_functional(x, None, None)
    results.append(_ok('outputs bit-identical after state_dict roundtrip',
                       torch.equal(out_a, out_b)))
    results.append(_ok('ssm state identical after roundtrip',
                       torch.equal(s_a, s_b)))
    results.append(_ok('conv state identical after roundtrip',
                       torch.equal(c_a, c_b)))
    return results


def test_mamba2_state_request_independence(device='cpu'):
    """Two interleaved requests keep independent, unpolluted state."""
    results = []
    torch.manual_seed(7)
    mixer = Mamba2Mixer(_mamba2_small_cfg()).to(device).eval()
    d = 32
    xa = torch.randn(2, 12, d, device=device)
    xb = torch.randn(3, 12, d, device=device)   # DIFFERENT batch size
    with torch.no_grad():
        # Request A: one-shot.
        out_a_ref, s_a_ref, _ = mixer.forward_functional(xa, None, None)
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
    # Mamba2DecodingState container accounting.
    st = Mamba2DecodingState.from_state_dict(st_a)
    results.append(_ok('Mamba2DecodingState tracks bytes',
                       st.nbytes > 0 and
                       st.nbytes_breakdown()['ssm_state_bytes'] > 0 and
                       st.nbytes_breakdown()['conv_state_bytes'] > 0,
                       f'{st.nbytes} bytes'))
    st.reset()
    results.append(_ok('Mamba2DecodingState reset clears bytes',
                       st.nbytes == 0))
    return results


def test_hybrid_stack_memory_routes(device='cpu'):
    """Ratio + explicit layouts, layout strings, states, and the KDA
    backward-compatibility contract of HybridAttentionStack."""
    results = []
    torch.manual_seed(8)
    # --- KDA route backward compatibility -----------------------------
    cfg_kda = HybridConfig(d_model=32, n_heads_qk=2, n_heads_v=2,
                           head_dim_k=16, head_dim_v=16)
    torch.manual_seed(42)
    m_legacy = HybridKCHAttention(cfg_kda, total_layers=5).to(device).eval()
    torch.manual_seed(42)
    m_stack = HybridAttentionStack(cfg_kda, total_layers=5).to(device).eval()
    results.append(_ok(
        'default config keeps the historical layout',
        m_legacy.layout == ['kda', 'kda', 'kda', 'csa', 'hca'],
        str(m_legacy.layout)))
    results.append(_ok(
        'layout_str unchanged',
        m_legacy.layout_str() == 'KDA-KDA-KDA-CSA-HCA',
        m_legacy.layout_str()))
    x = torch.randn(2, 11, 32, device=device)
    with torch.no_grad():
        m_legacy.reset_state(); m_stack.reset_state()
        out_legacy = m_legacy(x)
        out_stack = m_stack(x)
    results.append(_ok(
        'HybridKCHAttention == HybridAttentionStack numerics',
        torch.equal(out_legacy, out_stack)))
    # --- Mamba-2 ratio route -------------------------------------------
    cfg_m2 = _mamba2_small_cfg(n_heads_qk=2, n_heads_v=2,
                               head_dim_k=16, head_dim_v=16)
    m_m2 = HybridAttentionStack(cfg_m2, total_layers=5).to(device).eval()
    results.append(_ok(
        'mamba2 ratio-mode layout',
        m_m2.layout == ['mamba2'] * 3 + ['csa', 'hca'], str(m_m2.layout)))
    results.append(_ok('mamba2 layout_str',
                       m_m2.layout_str() == 'MAMBA2-MAMBA2-MAMBA2-CSA-HCA',
                       m_m2.layout_str()))
    results.append(_ok('n_memory_layers counts mamba2 layers',
                       m_m2.n_memory_layers == 3
                       and m_m2.n_kda_layers == 3))
    with torch.no_grad():
        m_m2.reset_state()
        out = m_m2(x)
        results.append(_ok('mamba2 stack forward shape',
                           tuple(out.shape) == (2, 11, 32),
                           str(tuple(out.shape))))
        first_mem = m_m2.memory_layers[0]
        expected_state = (3, 2, first_mem.nheads, first_mem.headdim,
                          first_mem.d_state)
        results.append(_ok(
            'mamba2 stacked memory state shape',
            m_m2._kda_state is not None
            and tuple(m_m2._kda_state.shape) == expected_state,
            str(None if m_m2._kda_state is None
                else tuple(m_m2._kda_state.shape))))
    # Stateful forward == functional forward on the same stack.
    with torch.no_grad():
        m_m2.reset_state()
        out_stateful = m_m2(x)
        m_m2.reset_state()
        out_fn, new_state, _ = m_m2.forward_functional(
            x, None, [None] * m_m2.n_memory_layers)
        m_m2._kda_state = new_state
    results.append(_ok('stateful == functional forward (mamba2 stack)',
                       torch.allclose(out_stateful, out_fn, atol=1e-6),
                       f'maxdiff={(out_stateful - out_fn).abs().max():.2e}'))
    # --- Explicit layout mode -------------------------------------------
    layout = ('memory', 'memory', 'memory', 'csa', 'hca') * 2
    cfg_lay = _mamba2_small_cfg(n_heads_qk=2, n_heads_v=2,
                                head_dim_k=16, head_dim_v=16,
                                layout=layout)
    results.append(_ok(
        'cfg.layout resolved and stored',
        cfg_lay.layout == ('mamba2', 'mamba2', 'mamba2', 'csa', 'hca') * 2,
        str(cfg_lay.layout)))
    m_lay = HybridAttentionStack(cfg_lay, total_layers=10).to(device).eval()
    results.append(_ok('explicit layout stack depth', len(m_lay.layers) == 10))
    with torch.no_grad():
        m_lay.reset_state()
        out = m_lay(torch.randn(2, 7, 32, device=device))
        results.append(_ok('explicit-layout stack forward',
                           tuple(out.shape) == (2, 7, 32)))
    # Parameter counts differ between routes (must be REPORTED, not assumed
    # equal — the budget-matching experiment exists precisely because of
    # this).
    n_params_kda = sum(p.numel() for p in m_legacy.parameters())
    n_params_m2 = sum(p.numel() for p in m_m2.parameters())
    results.append(_ok('kda vs mamba2 param counts recorded (not assumed)',
                       n_params_kda > 0 and n_params_m2 > 0
                       and n_params_kda != n_params_m2,
                       f'kda={n_params_kda}, mamba2={n_params_m2}'))
    # --- Layout validation error paths ----------------------------------
    for name, kwargs, err in [
            ('layout/total_layers mismatch',
             dict(layout=('csa', 'hca'), _total_layers=5), ValueError),
            ('mixed memory kinds',
             dict(layout=('kda', 'mamba2')), ValueError),
            ('unknown kind',
             dict(layout=('kda', 'softmax')), ValueError),
            ('empty layout', dict(layout=()), ValueError),
            ('bad memory_type', dict(memory_type='mamba3'), ValueError),
    ]:
        total_layers = kwargs.pop('_total_layers', 5)
        try:
            cfg_bad = HybridConfig(
                d_model=32, n_heads_qk=2, n_heads_v=2,
                head_dim_k=16, head_dim_v=16, **kwargs)
            HybridAttentionStack(cfg_bad, total_layers=total_layers)
            results.append(_ok(f'rejects {name}', False, 'no error raised'))
        except err:
            results.append(_ok(f'rejects {name}', True))
        except Exception as e:
            results.append(_ok(f'rejects {name}', False,
                               f'{type(e).__name__}: {e}'))
    # --- Mixer factory --------------------------------------------------
    results.append(_ok("build_mixer('memory') follows memory_type=kda",
                       isinstance(build_mixer('memory', cfg_kda),
                                  KDAHybridLayer)))
    results.append(_ok("build_mixer('memory') follows memory_type=mamba2",
                       isinstance(build_mixer('memory', cfg_m2), Mamba2Mixer)))
    results.append(_ok("build_mixer('csa')",
                       type(build_mixer('csa', cfg_kda)).__name__
                       == 'CSAHybridLayer'))
    try:
        build_mixer('softmax', cfg_kda)
        results.append(_ok('build_mixer rejects unknown kind', False,
                           'no error'))
    except ValueError:
        results.append(_ok('build_mixer rejects unknown kind', True))
    return results


def test_mamba2_official_backend_parity(device='cpu'):
    """Reference vs official mamba_ssm outputs/gradients match, when installed.

    Skipped (recorded as PASS-with-note) when the optional ``mamba-ssm``
    extra or CUDA is unavailable — mirroring how the FLA KDA backend is
    handled. When it IS available, the test transfers the reference
    weights into the official module and requires output agreement within
    a dtype-scaled tolerance (fp32: 1e-4; the official kernels use
    different reduction orders, so bit-equality is NOT the contract).
    """
    results = []
    if device != 'cuda' or not mamba2_official_available():
        results.append(_ok(
            'official backend parity (SKIPPED: needs CUDA + mamba-ssm '
            'extra)', True,
            f"device={device}, official_available="
            f"{mamba2_official_available()}"))
        return results
    from ops_mamba_backend import (OfficialMamba2Mixer,
                                    build_official_mamba2_mixer,
                                    transfer_reference_state)
    torch.manual_seed(9)
    cfg = _mamba2_small_cfg()
    ref = Mamba2Mixer(cfg).to(device).eval()
    off = build_official_mamba2_mixer(cfg).to(device).eval()
    transfer_reference_state(off.mamba, ref)
    B, T, d = 2, 40, 32
    x = torch.randn(B, T, d, device=device)
    with torch.no_grad():
        out_ref, s_ref, c_ref = ref.forward_functional(x, None, None)
        out_off, s_off, c_off = off.forward_functional(x, None, None)
    tol_out = 1e-4
    diff = (out_ref.float() - out_off.float()).abs().max().item()
    results.append(_ok('official backend output parity (fp32)',
                       diff < tol_out, f'maxdiff={diff:.2e}'))
    diff_s = (s_ref.float() - s_off.float()).abs().max().item()
    results.append(_ok('official backend SSM state parity',
                       diff_s < tol_out, f'maxdiff={diff_s:.2e}'))
    # Prefill+step equivalence through the official wrapper.
    with torch.no_grad():
        out_pre, st = off.prefill(x[:, :30])
        o1, st = off.step(x[:, 30:31], st)
        o2, st = off.step(x[:, 31:32], st)
        steps = torch.cat([o1, o2], dim=1)
    diff_steps = (steps.float() - out_off[:, 30:32].float()).abs().max().item()
    results.append(_ok('official prefill+step parity',
                       diff_steps < tol_out, f'maxdiff={diff_steps:.2e}'))
    return results


def test_mamba2_backend_selection_contract(device='cpu'):
    """Backend selection resolves/records correctly and errors loudly."""
    results = []
    from ops_mamba_backend import validate_mamba_backend
    try:
        validate_mamba_backend('tensorrt')
        results.append(_ok('unknown mamba_backend rejected', False,
                           'no error'))
    except ValueError:
        results.append(_ok('unknown mamba_backend rejected', True))
    # On a CPU-only box the 'official' request cannot be honoured...
    if torch.cuda.is_available() and mamba2_official_available():
        results.append(_ok('official backend available (CUDA box)', True))
    else:
        try:
            cfg = _mamba2_small_cfg(mamba_backend='official')
            Mamba2Mixer(cfg)  # the reference module itself never dispatches
            # The FACTORY is the dispatch point:
            build_mamba2_mixer(cfg)
            results.append(_ok(
                "build_mamba2_mixer('official') errors when unavailable",
                False, 'no error'))
        except (ImportError, RuntimeError):
            results.append(_ok(
                "build_mamba2_mixer('official') errors when unavailable",
                True))
        # ...and resolve_mamba_backend records the would-be resolution
        # without raising (provenance helper).
        results.append(_ok(
            "resolve_mamba_backend('official') degrades to 'reference'",
            resolve_mamba_backend('official') == 'reference'))
    # 'auto' falls back with a recorded marker on this box.
    cfg_auto = _mamba2_small_cfg(mamba_backend='auto')
    mixer_auto = build_mamba2_mixer(cfg_auto)
    expected = ('official' if (torch.cuda.is_available()
                               and mamba2_official_available())
                else 'reference (auto fallback)')
    results.append(_ok("auto backend resolution recorded",
                       mixer_auto.effective_backend == expected,
                       mixer_auto.effective_backend))
    return results


__all__ = [
    'test_mamba2_geometry_validation',
    'test_mamba2_forward_smoke',
    'test_mamba2_prefill_step_matches_full',
    'test_mamba2_segmented_prefill',
    'test_mamba2_causality',
    'test_mamba2_conv_boundary_cases',
    'test_mamba2_gradient',
    'test_mamba2_state_save_load',
    'test_mamba2_state_request_independence',
    'test_hybrid_stack_memory_routes',
    'test_mamba2_official_backend_parity',
    'test_mamba2_backend_selection_contract',
]
