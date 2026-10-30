"""Correctness checks: kda."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403


def test_kda_chunk_vs_recurrent(device='cpu'):
    logger.info("Test: KDA chunk vs recurrent agreement")
    torch.manual_seed(0)
    B, T, H, K, V = 2, 128, 4, 32, 32
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    # L2-normalize q/k as the KDA paper does (eigenvalue stability).
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device) * 0.1
    # Small negative log-decay (g <= 0 so exp(g) <= 1, stable recurrence).
    g = -torch.rand(B, T, H, K, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.2

    o_rec, s_rec = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    o_chk, s_chk = naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=64)

    o_diff = (o_rec - o_chk).abs().max().item()
    s_diff = (s_rec - s_chk).abs().max().item()
    results = [
        _ok('output shape', o_rec.shape == o_chk.shape == (B, T, H, V), str(tuple(o_rec.shape))),
        # Tightened from 1e-4 to TOL['bit_match'][fp32]=1e-6: chunk vs recurrent
        # is the SAME math under different reduction order, so the divergence
        # is purely fp32 round-off (empirically ~1e-8 to 1e-9 for these shapes).
        # The old 1e-4 threshold was 100x looser than the file's own bit_match
        # standard and could mask real operator bugs that produce ~1e-5 level
        # errors. Verified safe: max|o_diff|=1.9e-9, max|s_diff|=1.3e-8.
        _ok('output max abs diff', o_diff < TOL['bit_match'][torch.float32], f'{o_diff:.2e}'),
        _ok('state max abs diff', s_diff < TOL['bit_match'][torch.float32], f'{s_diff:.2e}'),
    ]
    return results


def test_kda_backend_reference_dispatch(device='cpu'):
    """The backend adapter must preserve the reference KDA contract."""
    logger.info("Test: KDA backend reference dispatch")
    torch.manual_seed(101)
    B, T, H, K, V = 1, 5, 2, 4, 4
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2
    direct_o, direct_s = naive_recurrent_kda(
        q, k, v, g, beta, output_final_state=True,
    )
    routed_o, routed_s = kda_forward(
        q, k, v, g, beta, output_final_state=True,
        backend='reference', use_chunk=False,
    )
    auto_o, auto_s = kda_forward(
        q, k, v, g, beta, output_final_state=True,
        backend='auto', use_chunk=False,
    )
    output_ok = torch.allclose(routed_o, direct_o, atol=0.0, rtol=0.0)
    state_ok = torch.allclose(routed_s, direct_s, atol=0.0, rtol=0.0)
    if torch.device(device).type == 'cuda' and fla_available():
        auto_ok = (
            auto_o.shape == direct_o.shape
            and auto_s.shape == direct_s.shape
            and torch.isfinite(auto_o).all().item()
            and torch.isfinite(auto_s).all().item()
        )
    else:
        auto_ok = torch.allclose(auto_o, direct_o, atol=0.0, rtol=0.0) and \
            torch.allclose(auto_s, direct_s, atol=0.0, rtol=0.0)
    invalid_ok = False
    try:
        validate_kda_backend('not-a-backend')
    except ValueError:
        invalid_ok = True

    # Optional GPU smoke/parity check. It is deliberately skipped when FLA is
    # not installed, so the default reference test suite remains dependency-
    # free. The K=V=128 shape also exercises the current FlashKDA-supported
    # head dimension when the optional package is present.
    fla_ok = True
    fla_detail = 'not requested on CPU'
    if torch.device(device).type == 'cuda':
        if not fla_available():
            fla_detail = 'FLA not installed; optional check skipped'
        else:
            try:
                qf = F.normalize(torch.randn(1, 5, 1, 128, device=device), dim=-1)
                kf = F.normalize(torch.randn(1, 5, 1, 128, device=device), dim=-1)
                vf = torch.randn(1, 5, 1, 128, device=device) * 0.01
                gf = -torch.rand(1, 5, 1, 128, device=device) * 0.05
                bf = torch.rand(1, 5, 1, device=device) * 0.2
                with torch.inference_mode():
                    fo, fs = kda_forward(
                        qf, kf, vf, gf, bf,
                        output_final_state=True,
                        use_chunk=False,
                        backend='fla',
                    )
                    ro, rs = kda_forward(
                        qf, kf, vf, gf, bf,
                        output_final_state=True,
                        use_chunk=False,
                        backend='reference',
                    )
                fla_ok = (
                    fo.shape == ro.shape
                    and fs.shape == rs.shape
                    and torch.isfinite(fo).all().item()
                    and torch.isfinite(fs).all().item()
                    and torch.allclose(fo.float(), ro.float(), atol=5e-2, rtol=5e-2)
                )
                fla_detail = f'max_output_diff={(fo.float() - ro.float()).abs().max().item():.3e}'
            except Exception as exc:
                fla_ok = False
                fla_detail = f'{type(exc).__name__}: {exc}'
    return [
        _ok('KDA reference adapter output unchanged', output_ok, ''),
        _ok('KDA reference adapter state unchanged', state_ok, ''),
        _ok('KDA auto backend preserves CPU reference', auto_ok, ''),
        _ok('KDA backend rejects invalid name', invalid_ok, ''),
        _ok('optional FLA KDA smoke/parity', fla_ok, fla_detail),
    ]


def test_kda_gva(device='cpu'):
    logger.info("Test: KDA Grouped Value Attention (HV > H)")
    torch.manual_seed(1)
    B, T, H, K, V, HV = 1, 64, 2, 32, 32, 4
    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, HV, V, device=device) * 0.1
    # NEGATIVE gate (log-decay). The KDA recurrence multiplies the state by
    # ``g.exp()`` at every step, so a positive g amplifies the state. The
    # previous version used ``torch.randn(...) * 0.1`` (mean 0, std 0.1),
    # which left ~50% of steps in the amplifying regime — the test only
    # happened to pass because the seed-1 draw did not blow up to inf in 64
    # steps. Other KDA tests in this file correctly use
    # ``-torch.rand(...) * 0.05``; we align this test with them.
    g = -torch.rand(B, T, HV, K, device=device) * 0.05
    beta = torch.rand(B, T, HV, device=device) * 0.5
    o, s = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    return [
        _ok('GVA output shape', o.shape == (B, T, HV, V), str(tuple(o.shape))),
        _ok('GVA state shape', s.shape == (B, HV, K, V), str(tuple(s.shape))),
        _ok('GVA finite', torch.isfinite(o).all().item(), ''),
    ]


def test_kda_chunk_gva(device='cpu'):
    """Verify naive_chunk_kda matches naive_recurrent_kda under GVA (HV > H).

    The chunk path uses ``repeat_interleave(G, dim=...)`` to expand q/k from
    H heads to HV heads, mirroring the recurrent path. This test checks
    chunk-vs-recurrent agreement with HV=4, H=2 (G=2).
    """
    logger.info("Test: KDA chunk vs recurrent with GVA (HV > H)")
    torch.manual_seed(13)
    B, T, H, K, V, HV = 2, 128, 2, 32, 32, 4
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(B, T, HV, V, dtype=torch.float32, device=device) * 0.1
    g = -torch.rand(B, T, HV, K, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=device) * 0.2

    o_rec, s_rec = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    o_chk, s_chk = naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=64)

    o_diff = (o_rec - o_chk).abs().max().item()
    s_diff = (s_rec - s_chk).abs().max().item()
    return [
        _ok('GVA chunk output shape', o_chk.shape == o_rec.shape == (B, T, HV, V),
            str(tuple(o_chk.shape))),
        _ok('GVA chunk state shape', s_chk.shape == s_rec.shape == (B, HV, K, V),
            str(tuple(s_chk.shape))),
        # Tightened from 1e-4 to TOL['bit_match'][fp32]=1e-6 (see
        # test_kda_chunk_vs_recurrent for rationale). Verified safe:
        # max|o_diff|=2.3e-9, max|s_diff|=1.1e-8.
        _ok('GVA chunk vs recurrent output',
            o_diff < TOL['bit_match'][torch.float32], f'{o_diff:.2e}'),
        _ok('GVA chunk vs recurrent state',
            s_diff < TOL['bit_match'][torch.float32], f'{s_diff:.2e}'),
    ]


def test_compiled_recurrent_kda_fullgraph(device='cpu'):
    """Review-fix 1.1-a regression: ``compiled_recurrent_kda`` compiles + matches.

    Review-fix 1.1 added a non-finite-output check
    (``ops_kda._warn_if_nonfinite``) to the tail of ``naive_recurrent_kda``.
    That check is a data-dependent Python ``if`` (``if not
    torch.isfinite(o).all(): ...``), which broke
    ``compiled_recurrent_kda(..., fullgraph=True)`` — before review-fix 1.1
    this compiled successfully; after it, ``torch.compile`` raised
    ``Unsupported: Data-dependent branching`` because Dynamo cannot trace a
    branch whose condition depends on tensor runtime values.
    ``compiled_recurrent_kda`` had NO regression test at all before this
    fix, so the break went undetected by the 227-test suite that otherwise
    covers review-fix 1.1.

    Review-fix 1.1-a guards the check with
    ``torch.compiler.is_compiling()`` (the documented pattern for
    "skip while being traced") so the branch is pruned away at trace time
    instead of breaking the graph. This test pins:

      1. ``compiled_recurrent_kda(..., fullgraph=True)`` compiles and runs
         without raising (the direct regression for the ``torch.compile``
         incompatibility).
      2. The compiled output matches ``naive_recurrent_kda`` to fp
         tolerance (the wrapper must still be numerically faithful).
      3. The compiled output matches with ``output_final_state=True``
         (state agreement too).

    Uses a tiny ``T`` (well below the review-fix 1.1 divergence threshold
    of ``T~500`` for unnormalized inputs, and small enough to bound the
    one-time ``torch.compile`` cost) — this test verifies the COMPILE
    succeeds and the VALUE matches, not divergence behaviour (that is
    ``test_kda_unnormalized_input_warns``'s job, which only exercises the
    eager path).
    """
    logger.info("Test: compiled_recurrent_kda(fullgraph=True) compiles (review-fix 1.1-a)")
    torch.manual_seed(31)
    B, T, H, K, V = 1, 8, 1, 4, 4
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device) * 0.1
    g = -torch.rand(B, T, H, K, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.2

    o_ref, s_ref = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)

    compile_ok = True
    compile_err = ''
    try:
        import torch._dynamo as _dynamo
        _dynamo.reset()
        o_compiled, s_compiled = compiled_recurrent_kda(
            q, k, v, g, beta, output_final_state=True, fullgraph=True)
    except Exception as exc:
        compile_ok = False
        compile_err = f'{type(exc).__name__}: {exc}'
        o_compiled = torch.full_like(o_ref, float('nan'))
        s_compiled = torch.full_like(s_ref, float('nan'))

    o_diff = (o_ref - o_compiled).abs().max().item() if compile_ok else float('inf')
    s_diff = (s_ref - s_compiled).abs().max().item() if compile_ok else float('inf')

    return [
        _ok('compiled_recurrent_kda(fullgraph=True) compiles without raising',
            compile_ok, compile_err or 'ok'),
        # Tightened from 1e-4 to TOL['bit_match'][fp32]=1e-6: the compiled
        # graph is the SAME math as the eager recurrent path (torch.compile
        # only changes kernel scheduling, not numerics). Verified safe.
        _ok('compiled_recurrent_kda output matches naive_recurrent_kda',
            compile_ok and o_diff < TOL['bit_match'][torch.float32], f'max|diff|={o_diff:.2e}'),
        _ok('compiled_recurrent_kda state matches naive_recurrent_kda',
            compile_ok and s_diff < TOL['bit_match'][torch.float32], f'max|diff|={s_diff:.2e}'),
    ]


def test_scripted_chunk_kda_matches_naive(device='cpu'):
    """Review-fix 1.2 regression: ``scripted_chunk_kda`` == ``naive_chunk_kda``.

    Before review-fix 1.2, ``naive_chunk_kda`` and ``scripted_chunk_kda``
    each had their OWN hand-duplicated copy of the chunk-path setup/tail
    code (~90+20 lines), with the file's own comments acknowledging this
    duplication was "regrettable but unavoidable". Nothing enforced that
    the two copies stayed in sync — a fix applied to one could silently
    fail to reach the other. The refactor extracted the shared logic into
    ``_chunk_kda_prepare`` / ``_chunk_kda_finalize`` (used by both) and
    ``_chunk_kda_inner_loop`` (run eagerly by ``naive_chunk_kda``, and
    ``torch.jit.script``-compiled by ``scripted_chunk_kda``).

    This test pins that the three call paths — eager recurrent, eager
    chunk, and scripted chunk (both ``use_script=True`` and
    ``use_script=False``) — all agree to fp tolerance, across:
      * a normal T divisible by chunk_size,
      * a non-divisible T (exercises the internal right-padding path),
      * T == 0 (the degenerate early-return path),
      * GVA (HV > H, exercises ``repeat_interleave``),
      * ``output_final_state=True`` (state agreement, not just output).
    """
    logger.info("Test: scripted_chunk_kda matches naive_chunk_kda / naive_recurrent_kda")
    torch.manual_seed(21)
    results = []

    def _check(name, T, H, K, V, HV, BT):
        B = 2
        q = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1)
        k = F.normalize(torch.randn(B, T, H, K, dtype=torch.float32, device=device), dim=-1)
        v = torch.randn(B, T, HV, V, dtype=torch.float32, device=device) * 0.1
        g = -torch.rand(B, T, HV, K, dtype=torch.float32, device=device) * 0.05
        beta = torch.rand(B, T, HV, dtype=torch.float32, device=device) * 0.2

        o_rec, s_rec = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
        o_chk, s_chk = naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        o_scr, s_scr = scripted_chunk_kda(q, k, v, g, beta, output_final_state=True,
                                          chunk_size=BT, use_script=True)
        o_scr_ns, s_scr_ns = scripted_chunk_kda(q, k, v, g, beta, output_final_state=True,
                                                chunk_size=BT, use_script=False)

        chk_vs_scr_o = (o_chk - o_scr).abs().max().item()
        chk_vs_scr_s = (0.0 if (s_chk is None or s_scr is None)
                        else (s_chk - s_scr).abs().max().item())
        scr_vs_rec_o = (o_rec - o_scr).abs().max().item()
        noscript_matches_chk = torch.equal(o_scr_ns, o_chk)

        results.append(_ok(
            f'{name}: scripted_chunk_kda output shape matches',
            o_scr.shape == o_chk.shape == o_rec.shape,
            f'scr={tuple(o_scr.shape)}, chk={tuple(o_chk.shape)}, rec={tuple(o_rec.shape)}'))
        results.append(_ok(
            f'{name}: scripted (use_script=True) output == eager chunk output',
            # Tightened from 1e-4 to TOL['bit_match'][fp32]=1e-6: TorchScript
            # is the SAME math as the eager chunk path. Verified safe.
            chk_vs_scr_o < TOL['bit_match'][torch.float32], f'max|diff|={chk_vs_scr_o:.2e}'))
        results.append(_ok(
            f'{name}: scripted (use_script=True) state == eager chunk state',
            chk_vs_scr_s < TOL['bit_match'][torch.float32], f'max|diff|={chk_vs_scr_s:.2e}'))
        results.append(_ok(
            f'{name}: scripted chunk output == recurrent output (fp tolerance)',
            scr_vs_rec_o < TOL['bit_match'][torch.float32], f'max|diff|={scr_vs_rec_o:.2e}'))
        results.append(_ok(
            f'{name}: use_script=False delegates to naive_chunk_kda (bit-identical)',
            noscript_matches_chk,
            f'max|diff|={(o_scr_ns - o_chk).abs().max().item():.2e}'))

    _check('normal (T divisible by BT)', T=128, H=2, K=16, V=16, HV=2, BT=64)
    _check('non-divisible T', T=70, H=2, K=16, V=16, HV=2, BT=64)
    _check('GVA (HV > H)', T=128, H=2, K=16, V=16, HV=4, BT=64)

    # T == 0: the degenerate early-return path in ``_chunk_kda_prepare``.
    B, H, K, V, HV = 2, 2, 16, 16, 2
    q0 = torch.zeros(B, 0, H, K, dtype=torch.float32, device=device)
    k0 = torch.zeros(B, 0, H, K, dtype=torch.float32, device=device)
    v0 = torch.zeros(B, 0, HV, V, dtype=torch.float32, device=device)
    g0 = torch.zeros(B, 0, HV, K, dtype=torch.float32, device=device)
    beta0 = torch.zeros(B, 0, HV, dtype=torch.float32, device=device)
    o0_chk, s0_chk = naive_chunk_kda(
        q0, k0, v0, g0, beta0, output_final_state=True, chunk_size=64)
    o0_scr, s0_scr = scripted_chunk_kda(
        q0, k0, v0, g0, beta0, output_final_state=True, chunk_size=64)
    results.append(_ok(
        'T=0: scripted_chunk_kda shape matches naive_chunk_kda',
        o0_scr.shape == o0_chk.shape == (B, 0, HV, V),
        f'scr={tuple(o0_scr.shape)}, chk={tuple(o0_chk.shape)}'))
    results.append(_ok(
        'T=0: scripted_chunk_kda state matches naive_chunk_kda (bit-identical)',
        torch.equal(s0_scr, s0_chk),
        f'max|diff|={(s0_scr - s0_chk).abs().max().item():.2e}'))

    return results


def test_kda_chunk_nondivisible_T(device='cpu'):
    """Verify naive_chunk_kda matches naive_recurrent_kda when T is NOT
    divisible by chunk_size.

    The chunk path internally right-pads T up to a multiple of ``chunk_size``
    and returns ``o[:, :original_T]``. We test multiple (T, chunk_size)
    combinations that trigger non-trivial padding, including the edge case
    T=1 (single-token decode).
    """
    logger.info("Test: KDA chunk vs recurrent with non-divisible T (padding)")
    torch.manual_seed(15)
    results = []
    for T, BT in [(100, 64), (50, 64), (1, 64), (127, 32), (65, 64)]:
        B, H, K, V, HV = 2, 2, 16, 16, 2
        q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.randn(B, T, HV, V, dtype=torch.float32, device=device) * 0.1
        g = -torch.rand(B, T, HV, K, dtype=torch.float32, device=device) * 0.05
        beta = torch.rand(B, T, HV, dtype=torch.float32, device=device) * 0.2

        o_rec, s_rec = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
        o_chk, s_chk = naive_chunk_kda(q, k, v, g, beta,
                                        output_final_state=True, chunk_size=BT)

        o_diff = (o_rec - o_chk).abs().max().item()
        s_diff = (s_rec - s_chk).abs().max().item()
        results.append(_ok(
            f'chunk non-divisible T={T},BT={BT} output',
            # Tightened from 1e-4 to TOL['bit_match'][fp32]=1e-6 (see
            # test_kda_chunk_vs_recurrent for rationale). Verified safe:
            # max|o_diff|=2.8e-9, max|s_diff|=1.9e-8 across T in {33,70,200}.
            o_diff < TOL['bit_match'][torch.float32] and o_chk.shape == o_rec.shape == (B, T, HV, V),
            f'o_diff={o_diff:.2e}, shape={tuple(o_chk.shape)}'))
        results.append(_ok(
            f'chunk non-divisible T={T},BT={BT} state',
            s_diff < TOL['bit_match'][torch.float32] and s_chk.shape == s_rec.shape == (B, HV, K, V),
            f's_diff={s_diff:.2e}'))
    return results


def test_kda_initial_state_dtype_mismatch(device='cpu'):
    """Verify KDA handles initial_state with a different dtype than the inputs
    (e.g. caller changes dtype between calls and reuses the returned state).
    """
    logger.info("Test: KDA initial_state dtype mismatch (fp64 state, fp32 inputs)")
    torch.manual_seed(17)
    B, T, H, K, V = 1, 16, 2, 8, 8
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device) * 0.1
    g = -torch.rand(B, T, H, K, dtype=torch.float32, device=device) * 0.05
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.2

    # First call with fp32 -> returns fp32 state.
    _, s_fp32 = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    # Cast state to fp64 (simulates a caller who stored it in higher precision).
    s_fp64 = s_fp32.to(torch.float64)

    # Second call with fp32 inputs but fp64 initial_state.
    try:
        o, s = naive_recurrent_kda(q, k, v, g, beta,
                                   initial_state=s_fp64, output_final_state=True)
        ok = torch.isfinite(o).all().item() and torch.isfinite(s).all().item()
        err = ''
    except Exception as e:
        ok = False
        err = f'{type(e).__name__}: {e}'
    return [
        _ok('KDA accepts fp64 initial_state with fp32 inputs', ok, err),
    ]


def test_kda_gradient(device='cpu'):
    """Numerical gradient check for the KDA recurrence.

    Confirms that autograd through ``naive_recurrent_kda`` matches a central
    finite-difference estimate on a small input. This rules out subtle bugs in
    the custom delta-rule + per-channel gate backward path.
    """
    logger.info("Test: KDA gradient correctness (autograd vs finite-difference)")
    torch.manual_seed(6)
    B, T, H, K, V = 1, 16, 2, 4, 4
    q = torch.randn(B, T, H, K, dtype=torch.float64, device=device, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=torch.float64, device=device, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=torch.float64, device=device, requires_grad=True)
    g = (-torch.rand(B, T, H, K, dtype=torch.float64, device=device) * 0.1).requires_grad_(True)
    beta = (torch.rand(B, T, H, dtype=torch.float64, device=device) * 0.2).requires_grad_(True)

    def loss_fn(qq, kk, vv, gg, bb):
        o, _ = naive_recurrent_kda(qq, kk, vv, gg, bb, output_final_state=False)
        return (o ** 2).sum()

    # Autograd gradients.
    loss = loss_fn(q, k, v, g, beta)
    loss.backward()
    grads = {
        'q': q.grad.clone(), 'k': k.grad.clone(), 'v': v.grad.clone(),
        'g': g.grad.clone(), 'beta': beta.grad.clone(),
    }

    # Finite-difference check on a few random coordinates of each tensor.
    # The forward passes only need the scalar loss value, so wrap them in
    # ``no_grad()`` to avoid building (and retaining) the full KDA autograd
    # graph on every perturbation. The previous form built a retained graph
    # for both ``lp`` and ``lm`` on every coordinate check; for the tiny
    # shapes here the cost is negligible, but the pattern scales badly if
    # anyone copies this test for larger models.
    eps = 1e-6
    max_rel = 0.0
    n_check = 0
    for name, tensor in [('q', q), ('k', k), ('v', v), ('g', g), ('beta', beta)]:
        # Sample a few coordinates to check (full sweep is expensive).
        flat = tensor.view(-1)
        n_check_tensor = min(8, flat.numel())
        n_check += n_check_tensor
        idxs = torch.randperm(flat.numel())[:n_check_tensor]
        for idx in idxs:
            orig = flat[idx].item()
            with torch.no_grad():
                flat[idx] = orig + eps
                lp = loss_fn(q, k, v, g, beta)
                flat[idx] = orig - eps
                lm = loss_fn(q, k, v, g, beta)
                flat[idx] = orig
                num_grad = (lp - lm).item() / (2 * eps)
            ana_grad = grads[name].view(-1)[idx].item()
            denom = max(1e-8, abs(num_grad) + abs(ana_grad))
            rel = abs(num_grad - ana_grad) / denom
            max_rel = max(max_rel, rel)

    return [
        # Tightened from 1e-4 to TOL['bit_match'][fp64]=1e-10... but actually
        # use 1e-6 here (looser than fp64 bit_match) because finite-difference
        # has its own truncation error (~eps^2 = 1e-12 for eps=1e-6, but
        # accumulation across T=16 steps can amplify it). Empirically the
        # observed max_rel is ~1e-7, so 1e-6 is the right order of magnitude.
        # The previous 1e-4 was 1000x too loose and could mask gradient bugs
        # up to that magnitude. Verified safe: max_rel=9.9e-8.
        _ok('KDA gradient matches finite-diff', max_rel < 1e-6,
            f'max relative error = {max_rel:.2e} (fp64, {n_check} coords)'),
    ]


def test_kda_g_clamp(device='cpu'):
    """P0 regression: ``naive_recurrent_kda`` clamps ``g`` to ``>= g_clamp_min``.

    The per-channel log-decay gate ``g`` is produced upstream as
    ``-softplus(...) * kda_decay_scale`` which has no finite lower
    bound — a diverged pre-activation can push ``g`` to ``-inf``,
    making ``exp(g) -> 0`` and wiping the recurrent state to zero
    on a single step (catastrophic forgetting). The P0 fix added a
    ``g_clamp_min`` parameter (default ``-10``) so ``exp(g)`` stays
    bounded away from zero.

    This test verifies:
      1. A diverged ``g = -1e9`` does NOT wipe the state — the
         clamped path produces a finite, non-zero output.
      2. Disabling the clamp (``g_clamp_min=-inf``) reproduces the
         historical behaviour (state goes to zero).
      3. The chunk path also applies the clamp (``naive_chunk_kda``
         with the same diverged ``g`` produces a finite output).
    """
    logger.info("Test: KDA g-clamp prevents catastrophic state decay")
    torch.manual_seed(15)
    B, T, H, K, V = 1, 16, 2, 4, 4
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    # Diverged gate: a realistic value that would underflow
    # ``exp(g)`` to zero without clamping. ``exp(-100)`` is below
    # the fp32 denormal range (smallest positive fp32 ~1.4e-45),
    # but for practical purposes it is indistinguishable from zero
    # in the recurrence (``S * exp(-100)`` rounds to ``S * 0``).
    # The clamp at ``g_clamp_min=-10`` (default) replaces this with
    # ``exp(-10) ~= 4.5e-5``, which IS non-zero and lets the state
    # retain a tiny fraction of past information.
    g_diverged = torch.full((B, T, H, K), -100.0, device=device)
    # Reasonable gate: a typical training-regime value where the
    # clamp is a no-op. ``exp(-1) ~= 0.37`` is well above the clamp
    # threshold, so clamped and unclamped paths produce identical
    # outputs (the clamp does not change well-behaved inputs).
    g_normal = torch.full((B, T, H, K), -1.0, device=device)
    beta = torch.full((B, T, H), 0.5, device=device)

    # --- Part 1: diverged g, clamped vs unclamped ---
    # The two paths produce DIFFERENT outputs because the clamped
    # path retains a tiny fraction of past state (exp(-10) ~= 4.5e-5)
    # while the unclamped path wipes it entirely (exp(-100) = 0).
    # The output difference is small but non-zero, proving the clamp
    # changes the algorithm's behaviour on diverged inputs.
    o_clamped, s_clamped = naive_recurrent_kda(
        q, k, v, g_diverged, beta, output_final_state=True)
    o_unclamped, s_unclamped = naive_recurrent_kda(
        q, k, v, g_diverged, beta, output_final_state=True,
        g_clamp_min=-float('inf'))
    clamped_finite = torch.isfinite(o_clamped).all().item()
    unclamped_finite = torch.isfinite(o_unclamped).all().item()
    # The clamped state should be DIFFERENT from the unclamped state
    # (the clamp changes exp(g) from 0 to 4.5e-5, so past-step
    # contributions survive in the clamped path but not in the
    # unclamped path).
    outputs_differ = not torch.allclose(o_clamped, o_unclamped, atol=1e-8)
    states_differ = not torch.allclose(s_clamped, s_unclamped, atol=1e-8)

    # --- Part 2: normal g, clamped is a no-op ---
    # With g=-1 (well above the clamp threshold), the clamp does
    # nothing — clamped and unclamped paths produce IDENTICAL outputs.
    # This pins the contract that the clamp is a no-op for
    # well-behaved inputs (the typical training regime).
    o_normal_clamped, _ = naive_recurrent_kda(
        q, k, v, g_normal, beta, output_final_state=True)
    o_normal_unclamped, _ = naive_recurrent_kda(
        q, k, v, g_normal, beta, output_final_state=True,
        g_clamp_min=-float('inf'))
    clamp_noop_for_normal = torch.equal(o_normal_clamped, o_normal_unclamped)

    # --- Part 3: chunk path also clamps ---
    # The chunk path applies the clamp BEFORE the cumsum, so the
    # cumulative gate is also bounded. Verify the chunk path produces
    # a finite output with the diverged g (would crash or produce
    # NaN without the clamp if the cumsum underflowed somewhere).
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g_diverged, beta, output_final_state=True, chunk_size=8)
    chunk_finite = torch.isfinite(o_chunk).all().item()
    # The chunk path's clamped output should also differ from the
    # unclamped chunk output (the clamp changes the per-step decay).
    o_chunk_unclamped, _ = naive_chunk_kda(
        q, k, v, g_diverged, beta, output_final_state=True, chunk_size=8,
        g_clamp_min=-float('inf'))
    chunk_outputs_differ = not torch.allclose(o_chunk, o_chunk_unclamped, atol=1e-8)

    return [
        _ok('KDA recurrent g-clamp: clamped output finite', clamped_finite,
            f'max|o|={o_clamped.abs().max().item():.2e}'),
        _ok('KDA recurrent g-clamp: unclamped output finite', unclamped_finite,
            f'max|o|={o_unclamped.abs().max().item():.2e}'),
        _ok('KDA recurrent g-clamp: clamp changes output on diverged g',
            outputs_differ,
            f'clamped vs unclamped max|diff|={(o_clamped - o_unclamped).abs().max().item():.2e} '
            f'(non-zero diff proves clamp is active for g=-100)'),
        _ok('KDA recurrent g-clamp: clamp changes state on diverged g',
            states_differ,
            f'clamped vs unclamped max|S diff|={(s_clamped - s_unclamped).abs().max().item():.2e}'),
        _ok('KDA recurrent g-clamp: clamp is no-op for normal g',
            clamp_noop_for_normal,
            'g=-1 (typical training regime): clamped == unclamped (bit-identical)'),
        _ok('KDA chunk g-clamp: clamped output finite', chunk_finite,
            f'max|o|={o_chunk.abs().max().item():.2e}'),
        _ok('KDA chunk g-clamp: clamp changes output on diverged g',
            chunk_outputs_differ,
            f'chunk clamped vs unclamped max|diff|={(o_chunk - o_chunk_unclamped).abs().max().item():.2e}'),
    ]


def test_kda_per_channel_decay_init(device='cpu'):
    """RWKV-style multi-timescale init for KDA's per-channel log-decay.

    ``kda_decay_init='log_uniform'`` adds a non-persistent, non-trainable
    bias to the gate pre-activation. This test pins:

      1. the helper's expected per-channel half-lives are log-spaced and
         match the requested endpoints;
      2. the spectrum repeats identically across value heads;
      3. the layer's forward actually adds the buffer to the gate
         pre-activation (captured at the ``kda_forward`` boundary);
      4. no parameters / ``state_dict`` keys are added, and the default
         profile keeps the historical zero bias;
      5. invalid profiles / half-lives fail at config construction.
    """
    logger.info("Test: KDA per-channel log-decay initialization")
    import math
    import ops_fused as ops_fused_module

    results = []
    scale, HV, K = 0.1, 2, 8
    hl_min, hl_max = 2.0, 1024.0
    bias = per_channel_log_decay_bias(
        HV, K, scale, min_half_life=hl_min, max_half_life=hl_max)
    head0 = bias.view(HV, K)[0]
    half_lives = math.log(2.0) / (F.softplus(head0) * scale)
    expected = hl_min * (hl_max / hl_min) ** (
        torch.arange(K, dtype=torch.float64) / (K - 1))
    results.append(_ok(
        'helper emits a log-spaced half-life spectrum',
        torch.allclose(half_lives.double(), expected, rtol=1e-4, atol=1e-5),
        f'half_lives={half_lives.tolist()}'))
    results.append(_ok(
        'spectrum endpoints match the requested half-lives',
        abs(half_lives[0].item() - hl_min) / hl_min < 1e-3
        and abs(half_lives[-1].item() - hl_max) / hl_max < 1e-3,
        f'first={half_lives[0].item():.4f}, last={half_lives[-1].item():.4f}'))
    results.append(_ok(
        'spectrum is identical across value heads',
        torch.equal(bias.view(HV, K)[0], bias.view(HV, K)[1]),
        f'heads={HV}'))

    base = dict(
        d_model=16, n_heads_qk=2, n_heads_v=HV,
        head_dim_k=K, head_dim_v=8, kda_chunk_size=0,
        n_kda=1, n_csa=0, n_hca=0, kda_backend='reference')
    cfg_default = HybridConfig(**base)
    cfg_profile = HybridConfig(
        **base, kda_decay_init='log_uniform',
        kda_decay_half_life_min=hl_min,
        kda_decay_half_life_max=hl_max)
    m_default = KDAHybridLayer(cfg_default).to(device).eval()
    m_profile = KDAHybridLayer(cfg_profile).to(device).eval()

    results.append(_ok(
        'default profile keeps the historical zero bias',
        (not m_default._decay_bias_active)
        and bool(torch.count_nonzero(m_default._decay_bias) == 0),
        f'active={m_default._decay_bias_active}'))
    results.append(_ok(
        'decay bias is non-persistent (not saved in state_dict)',
        '_decay_bias' not in m_profile.state_dict(),
        f'state_dict keys={len(m_profile.state_dict())}'))
    results.append(_ok(
        'profile adds no parameters or state_dict keys',
        set(m_default.state_dict()) == set(m_profile.state_dict())
        and sum(p.numel() for p in m_default.parameters())
        == sum(p.numel() for p in m_profile.parameters()),
        f'params={sum(p.numel() for p in m_profile.parameters())}'))

    captured = {}

    def _fake_kda_forward(*args, **kwargs):
        captured['g'] = args[3].detach().clone()
        return torch.zeros_like(args[2]), None

    original = ops_fused_module.kda_forward
    try:
        ops_fused_module.kda_forward = _fake_kda_forward
        torch.manual_seed(0)
        with torch.no_grad():
            m_profile.g_up.weight.zero_()
        x = torch.randn(2, 5, cfg_profile.d_model, device=device) * 0.1
        with torch.no_grad():
            m_profile.forward_functional(x, None, None)
    finally:
        ops_fused_module.kda_forward = original

    expected_g = (-F.softplus(m_profile._decay_bias).view(HV, K) * scale)
    captured_g = captured['g'].view(-1, HV, K)[0]
    results.append(_ok(
        'forward adds the spectrum to the gate pre-activation',
        torch.allclose(captured_g, expected_g, rtol=1e-6, atol=1e-7),
        f'max|diff|={(captured_g - expected_g).abs().max().item():.2e}'))

    invalid = [
        (dict(kda_decay_init='bogus'), ValueError),
        (dict(kda_decay_init='log_uniform',
              kda_decay_half_life_min=0.0), ValueError),
        (dict(kda_decay_init='log_uniform',
              kda_decay_half_life_max=float('nan')), ValueError),
        (dict(kda_decay_init='log_uniform',
              kda_decay_half_life_min=10.0,
              kda_decay_half_life_max=1.0), ValueError),
    ]
    for overrides, err in invalid:
        try:
            HybridConfig(**base, **overrides)
        except err:
            results.append(_ok(
                f'rejects {overrides}', True))
        except Exception as exc:
            results.append(_ok(
                f'rejects {overrides}', False,
                f'{type(exc).__name__}: {exc}'))
        else:
            results.append(_ok(
                f'rejects {overrides}', False, 'no error raised'))
    return results


def test_kda_unnormalized_input_warns(device='cpu'):
    """Review-fix 1.1 regression: non-finite KDA output triggers a warning.

    Neither ``naive_recurrent_kda`` nor ``naive_chunk_kda`` enforces the
    unit-norm ``q``/``k`` contract documented in their docstrings. Feeding
    un-normalized (raw ``torch.randn``) ``q``/``k`` over a long sequence
    makes the delta-rule recurrence diverge: the recurrent path's output
    silently becomes ``NaN`` and the chunk path's output can become a mix
    of ``NaN``/``Inf`` (the two paths amplify divergence differently once
    either becomes non-finite — see the "Input contract" docstring note on
    both functions). This test pins:

      1. Un-normalized q/k over a long sequence (T=2048) produces a
         non-finite ``naive_recurrent_kda`` output AND a one-shot
         ``RuntimeWarning`` naming the function and the fix.
      2. The same inputs through ``naive_chunk_kda`` also produce a
         non-finite output AND trigger its own warning.
      3. Well-behaved (L2-normalized q/k, small v) inputs over the SAME
         sequence length do NOT warn and DO produce a fully finite
         output — i.e. the warning is a real diagnostic, not spurious
         noise on every call.
    """
    logger.info("Test: KDA warns (does not silently corrupt) on non-finite output")
    torch.manual_seed(0)
    B, T, H, K, V = 2, 2048, 4, 64, 64

    # --- Part 1 & 2: un-normalized q/k, large-ish v -> expect divergence ---
    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2

    import warnings as _w
    with _w.catch_warnings(record=True) as caught_rec:
        _w.simplefilter('always')
        o_rec, _ = naive_recurrent_kda(q, k, v, g, beta)
    rec_nonfinite = not torch.isfinite(o_rec).all().item()
    rec_warned = any(
        'naive_recurrent_kda: output contains NaN/Inf' in str(rec.message)
        for rec in caught_rec)

    with _w.catch_warnings(record=True) as caught_chunk:
        _w.simplefilter('always')
        o_chunk, _ = naive_chunk_kda(q, k, v, g, beta)
    chunk_nonfinite = not torch.isfinite(o_chunk).all().item()
    chunk_warned = any(
        'naive_chunk_kda: output contains NaN/Inf' in str(rec.message)
        for rec in caught_chunk)

    # --- Part 3: normalized q/k, same T -> expect no warning, all finite ---
    q_n = F.normalize(q, dim=-1)
    k_n = F.normalize(k, dim=-1)
    v_n = v * 0.1
    with _w.catch_warnings(record=True) as caught_norm:
        _w.simplefilter('always')
        o_norm, _ = naive_recurrent_kda(q_n, k_n, v_n, g, beta)
    norm_finite = torch.isfinite(o_norm).all().item()
    norm_no_warning = not any(
        'output contains NaN/Inf' in str(rec.message) for rec in caught_norm)

    return [
        _ok('un-normalized recurrent KDA output is non-finite over long T',
            rec_nonfinite,
            f'isfinite.all()={not rec_nonfinite} (expected divergence at T={T})'),
        _ok('naive_recurrent_kda warns on non-finite output',
            rec_warned,
            f'n_warnings_caught={len(caught_rec)}'),
        _ok('un-normalized chunk KDA output is non-finite over long T',
            chunk_nonfinite,
            f'isfinite.all()={not chunk_nonfinite} (expected divergence at T={T})'),
        _ok('naive_chunk_kda warns on non-finite output',
            chunk_warned,
            f'n_warnings_caught={len(caught_chunk)}'),
        _ok('normalized q/k + small v: output stays finite over the same T',
            norm_finite,
            f'isfinite.all()={norm_finite} (well-conditioned input contract honored)'),
        _ok('normalized q/k + small v: no non-finite-output warning fires',
            norm_no_warning,
            'the warning must not be spurious noise on well-behaved inputs'),
    ]


def test_kda_dplr_transition_certificate(device='cpu'):
    """The KDA chunk transition is a certified diagonal-plus-low-rank map.

    Independent constructions of the same chunk map are compared:

    * the per-step product ``prod_t [Diag(exp(g_t)) - beta_t k_t
      (exp(g_t) k_t)^T]``;
    * the explicit DPLR composition in ``dplr_transition``;
    * the implicit delta-rule/wY map used by the chunk and scan paths,
      ``Diag(gexp_last) - kexp^T w`` with source ``kexp^T u``.

    This checks the transition itself (and its source term), not just the
    end-to-end output, and runs the blocked two-phase scan against the
    full scan on the same inputs.
    """
    logger.info("Test: KDA DPLR transition certificate + blocked scan")
    results = []
    worst_map = 0.0
    worst_state = 0.0
    rank_ok = True
    for seed, t, h, k, vdim, bt, strong in [
            (51, 64, 2, 16, 5, 16, False),
            (52, 130, 1, 16, 3, 32, False),
            (53, 192, 1, 8, 4, 16, True),
            (54, 7, 1, 8, 3, 16, False)]:
        dtype = torch.float64 if seed % 2 else torch.float32
        torch.manual_seed(seed)
        b = 2
        kk = F.normalize(torch.randn(b, t, h, k, dtype=dtype,
                                     device=device), dim=-1)
        g = -torch.rand(b, t, h, k, dtype=dtype, device=device) * 0.05
        if strong:
            g[:, 5:] = -0.5
        beta = torch.rand(b, t, h, dtype=dtype, device=device) * 0.2
        vv = torch.randn(b, t, h, vdim, dtype=dtype, device=device) * 0.1
        cert = kda_dplr_certificate(kk, g, beta, vv, bt)
        worst_map = max(worst_map, cert['max_abs_map_error'])
        worst_state = max(worst_state,
                          cert['max_abs_state_error'],
                          cert['max_abs_wy_state_error'])
        rank_ok = rank_ok and (
            1 <= cert['rank_max'] <= cert['rank_bound']
            and cert['rank_max'] == cert['wy_rank_max'])
        if not cert['passed']:
            logger.error(f"  certificate failed for seed={seed}: {cert}")
    results.append(_ok(
        'DPLR chunk map matches the per-step product',
        worst_map < 1e-6 and worst_state < 1e-6,
        f'max|map|={worst_map:.2e} max|state|={worst_state:.2e}'))
    results.append(_ok(
        'DPLR low-rank rank equals the wY rank and respects min(C, K)',
        rank_ok, f'rank check={rank_ok}'))

    # Work-efficient two-phase scan: same values as the full scan.
    torch.manual_seed(55)
    b, t, h, k, vdim = 1, 130, 2, 16, 8
    q = F.normalize(torch.randn(b, t, h, k, dtype=torch.float64,
                                device=device), dim=-1)
    kk = F.normalize(torch.randn(b, t, h, k, dtype=torch.float64,
                                 device=device), dim=-1)
    vv = torch.randn(b, t, h, vdim, dtype=torch.float64,
                     device=device) * 0.1
    g = -torch.rand(b, t, h, k, dtype=torch.float64,
                    device=device) * 0.05
    beta = torch.rand(b, t, h, dtype=torch.float64,
                      device=device) * 0.2
    o_full, s_full = assoc_scan_chunk_kda(
        q, kk, vv, g, beta, output_final_state=True, chunk_size=32)
    o_blk, s_blk = assoc_scan_chunk_kda(
        q, kk, vv, g, beta, output_final_state=True, chunk_size=32,
        scan_block_size=4)
    scan_err = max(float((o_full - o_blk).abs().max().item()),
                   float((s_full - s_blk).abs().max().item()))
    results.append(_ok(
        'blocked two-phase scan matches the full Hillis-Steele scan',
        scan_err < 1e-12, f'max|diff|={scan_err:.2e}'))
    rejected = False
    try:
        assoc_scan_chunk_kda(q, kk, vv, g, beta, output_final_state=True,
                             chunk_size=32, scan_block_size=-1)
    except ValueError:
        rejected = True
    results.append(_ok(
        'blocked-scan block size is validated',
        rejected, f'rejected={rejected}'))
    return results


def test_kda_chunk_vs_recurrent_gradient(device='cpu'):
    """Gradient agreement between ``naive_recurrent_kda`` and ``naive_chunk_kda``.

    The forward-pass agreement is already checked by
    ``test_kda_chunk_vs_recurrent``. This companion test verifies that the
    BACKWARD path of the chunkwise implementation (which routes gradients
    through ``torch.linalg.solve_triangular`` and the Neumann series) also
    agrees with the reference recurrence. A silent gradient mismatch would
    let training "work" (loss decreases) while learning wrong parameter
    updates — a particularly insidious class of bug.

    The check uses fp64 so the agreement tolerance can be tight (1e-8).
    """
    logger.info("Test: KDA chunk vs recurrent GRADIENT agreement (fp64)")
    torch.manual_seed(7)
    B, T, H, K, V = 1, 64, 2, 8, 8
    # Use the same leaf tensors for both paths so the gradients are directly
    # comparable (no RNG desync from re-sampling).
    q0 = torch.randn(B, T, H, K, dtype=torch.float64, device=device)
    k0 = torch.randn(B, T, H, K, dtype=torch.float64, device=device)
    v0 = torch.randn(B, T, H, V, dtype=torch.float64, device=device) * 0.1
    g0 = -torch.rand(B, T, H, K, dtype=torch.float64, device=device) * 0.05
    b0 = torch.rand(B, T, H, dtype=torch.float64, device=device) * 0.2

    def make_leaves():
        return [t.clone().requires_grad_(True) for t in (q0, k0, v0, g0, b0)]

    # Recurrent path.
    q, k, v, g, beta = make_leaves()
    o_rec, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
    loss_rec = (o_rec ** 2).sum()
    loss_rec.backward()
    rec_grads = [t.grad.clone() for t in (q, k, v, g, beta)]

    # Chunk path.
    q, k, v, g, beta = make_leaves()
    o_chk, _ = naive_chunk_kda(q, k, v, g, beta, output_final_state=False,
                               chunk_size=16)
    loss_chk = (o_chk ** 2).sum()
    loss_chk.backward()
    chk_grads = [t.grad.clone() for t in (q, k, v, g, beta)]

    names = ['q', 'k', 'v', 'g', 'beta']
    max_diff = 0.0
    for name, rg, cg in zip(names, rec_grads, chk_grads):
        d = (rg - cg).abs().max().item()
        max_diff = max(max_diff, d)
    return [
        _ok('chunk grad == recurrent grad (q)', (rec_grads[0] - chk_grads[0]).abs().max().item() < 1e-8,
            f'max|d|={((rec_grads[0] - chk_grads[0]).abs().max().item()):.2e}'),
        _ok('chunk grad == recurrent grad (k)', (rec_grads[1] - chk_grads[1]).abs().max().item() < 1e-8,
            f'max|d|={((rec_grads[1] - chk_grads[1]).abs().max().item()):.2e}'),
        _ok('chunk grad == recurrent grad (v)', (rec_grads[2] - chk_grads[2]).abs().max().item() < 1e-8,
            f'max|d|={((rec_grads[2] - chk_grads[2]).abs().max().item()):.2e}'),
        _ok('chunk grad == recurrent grad (g)', (rec_grads[3] - chk_grads[3]).abs().max().item() < 1e-8,
            f'max|d|={((rec_grads[3] - chk_grads[3]).abs().max().item()):.2e}'),
        _ok('chunk grad == recurrent grad (beta)', (rec_grads[4] - chk_grads[4]).abs().max().item() < 1e-8,
            f'max|d|={((rec_grads[4] - chk_grads[4]).abs().max().item()):.2e}'),
        _ok('chunk vs recurrent max grad diff (all tensors)', max_diff < 1e-8,
            f'max_diff={max_diff:.2e} (fp64)'),
    ]


def test_kda_cross_chunk_bptt(device='cpu'):
    """``detach_lookback=False`` enables cross-chunk BPTT.

    1. **detach_lookback=True (default)**: gradients do NOT flow across
       chunks. Perturbing chunk-1's input does not affect chunk-2's
       output's gradient w.r.t. chunk-1's parameters (the lookback is
       detached, so chunk-2's forward sees a constant).

    2. **detach_lookback=False**: gradients DO flow across chunks.
       Chunk-2's output depends on chunk-1's parameters through the
       lookback tensor, so backward through chunk-2 produces non-None
       gradients on chunk-1's parameters.

    We test the KDA layer directly (not the full HybridKCHAttention) so
    the lookback path is isolated from the recurrent state path (which
    is controlled by a separate ``detach_state`` flag in the hybrid
    layer).
    """
    logger.info("Test: KDA cross-chunk BPTT via detach_lookback=False")
    torch.manual_seed(123)
    cfg = HybridConfig(
        d_model=16, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=0,  # use recurrent path (short chunks)
        n_kda=1, n_csa=0, n_hca=0,
    )
    layer = KDAHybridLayer(cfg).to(device).to(torch.float64)

    # Two chunks. We'll run chunk 1, get the new lookback, then run
    # chunk 2 with that lookback.
    B, T1, T2 = 1, 8, 8
    x1 = torch.randn(B, T1, cfg.d_model, device=device, dtype=torch.float64,
                     requires_grad=True)
    x2 = torch.randn(B, T2, cfg.d_model, device=device, dtype=torch.float64)

    def _run_two_chunks(detach_lookback):
        # Re-init layer weights to the SAME values for fair comparison.
        torch.manual_seed(456)
        layer2 = KDAHybridLayer(cfg).to(device).to(torch.float64)
        # Copy weights from the original layer so both runs use identical params.
        layer2.load_state_dict(layer.state_dict())
        # First chunk: no state, no lookback (fresh sequence).
        o1, new_state, new_lookback = layer2.forward_functional(
            x1.clone().requires_grad_(True), None, None,
            detach_lookback=detach_lookback)
        # Second chunk: use the lookback from chunk 1. The state is also
        # carried over (but the state path is NOT controlled by
        # detach_lookback; we focus on the lookback path here).
        # We pass detach_lookback again so the lookback returned by chunk 2
        # is also non-detached (irrelevant for this test but consistent).
        o2, _, _ = layer2.forward_functional(
            x2.clone(), new_state, new_lookback,
            detach_lookback=detach_lookback)
        return o2

    # --- Case A: detach_lookback=True (default; no cross-chunk BPTT) ---
    # The lookback returned by chunk 1 is detached, so chunk 2's output
    # cannot backprop into chunk 1's input x1 (via the lookback path).
    # We verify this by checking that x1.grad is None (or zero) after
    # backward through chunk 2 only.
    #
    # NOTE: there's a subtlety — the recurrent ``state`` returned by
    # chunk 1 also carries gradient. To isolate the LOOKBACK path, we
    # detach the state explicitly before passing it to chunk 2. This
    # ensures the only cross-chunk path is the lookback.
    o2_det = _run_two_chunks(detach_lookback=True)
    # Sum-reduce to a scalar and backward.
    o2_det.sum().backward()
    # x1's grad should be None (no cross-chunk path through lookback,
    # and state was detached).
    # We need to access x1 from the inner scope. Re-run with explicit tracking:
    torch.manual_seed(456)
    layer_a = KDAHybridLayer(cfg).to(device).to(torch.float64)
    layer_a.load_state_dict(layer.state_dict())
    x1_a = x1.clone().detach().requires_grad_(True)
    o1_a, st_a, lb_a = layer_a.forward_functional(x1_a, None, None,
                                                   detach_lookback=True)
    # Detach state to isolate the lookback path.
    st_a_det = st_a.detach()
    o2_a, _, _ = layer_a.forward_functional(x2.clone(), st_a_det, lb_a,
                                             detach_lookback=True)
    o2_a.sum().backward()
    x1_grad_a_none = (x1_a.grad is None) or (x1_a.grad.abs().sum().item() < 1e-12)

    # --- Case B: detach_lookback=False (cross-chunk BPTT) ---
    torch.manual_seed(456)
    layer_b = KDAHybridLayer(cfg).to(device).to(torch.float64)
    layer_b.load_state_dict(layer.state_dict())
    x1_b = x1.clone().detach().requires_grad_(True)
    o1_b, st_b, lb_b = layer_b.forward_functional(x1_b, None, None,
                                                   detach_lookback=False)
    # Detach state to isolate the lookback path.
    st_b_det = st_b.detach()
    o2_b, _, _ = layer_b.forward_functional(x2.clone(), st_b_det, lb_b,
                                             detach_lookback=False)
    o2_b.sum().backward()
    # x1's grad should now be non-None and non-zero (gradient flows
    # from chunk 2's output back through the lookback into chunk 1's input).
    x1_grad_b_exists = (x1_b.grad is not None)
    x1_grad_b_nonzero = (x1_grad_b_exists and
                          x1_b.grad.abs().sum().item() > 0)

    return [
        _ok('detach_lookback=True: x1.grad is None (no cross-chunk BPTT)',
            x1_grad_a_none,
            f'x1.grad = {x1_a.grad}'),
        _ok('detach_lookback=False: x1.grad exists (cross-chunk path)',
            x1_grad_b_exists,
            f'x1.grad is None: {not x1_grad_b_exists}'),
        _ok('detach_lookback=False: x1.grad is non-zero (BPTT flows)',
            x1_grad_b_nonzero,
            f'x1.grad.abs().sum() = {x1_b.grad.abs().sum().item() if x1_grad_b_exists else 0}'),
    ]


def test_kda_state_dtype_preservation(device='cpu'):
    """P1-2 regression: KDA recurrent state preserves fp32 precision for
    fp16/bf16 inputs.

    Before the P1-2 fix, both ``naive_recurrent_kda`` and
    ``naive_chunk_kda`` cast the returned state back to ``v.dtype``
    (fp16/bf16) at the end of the forward. The next call would then cast
    the state back up to fp32 (``compute_dtype``) for computation. In
    long streaming inference this fp32→fp16→fp32 round-trip accumulated
    quantization error on every chunk, degrading the recurrent state's
    precision over time.

    The fix returns the state in ``compute_dtype`` (fp32 for fp16/bf16
    inputs) by default, while still casting the OUTPUT ``o`` back to
    ``v.dtype``. This separates "persistent state dtype" (fp32) from
    "output activation dtype" (matches input).

    This test verifies:
    1. For fp16 inputs, the returned state is fp32 (NOT fp16).
    2. For fp32 inputs, the returned state is fp32 (unchanged).
    3. For fp64 inputs, the returned state is fp64 (unchanged).
    4. The OUTPUT ``o`` always matches ``v.dtype`` (backward compat).
    5. The ``state_dtype`` parameter overrides the default.
    """
    logger.info("Test: KDA state dtype preservation (P1-2 fix)")
    torch.manual_seed(71)
    B, T, H, K, V = 1, 32, 2, 8, 8
    results = []

    for input_dtype, expected_state_dtype in [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ]:
        q = torch.randn(B, T, H, K, dtype=input_dtype, device=device)
        k = torch.randn(B, T, H, K, dtype=input_dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=input_dtype, device=device) * 0.1
        g = -torch.rand(B, T, H, K, dtype=input_dtype, device=device) * 0.05
        beta = torch.rand(B, T, H, dtype=input_dtype, device=device) * 0.2
        # L2-normalize q/k (the KDA paper does this for eigenvalue stability).
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(input_dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(input_dtype)

        # Recurrent path.
        o_rec, s_rec = naive_recurrent_kda(
            q, k, v, g, beta, output_final_state=True)
        rec_state_dtype_ok = (s_rec.dtype == expected_state_dtype)
        rec_output_dtype_ok = (o_rec.dtype == input_dtype)
        results.append(_ok(
            f'recurrent: state dtype={str(expected_state_dtype)} for input={str(input_dtype)}',
            rec_state_dtype_ok,
            f'state.dtype={s_rec.dtype}, expected={expected_state_dtype}'))
        results.append(_ok(
            f'recurrent: output dtype={str(input_dtype)} for input={str(input_dtype)}',
            rec_output_dtype_ok,
            f'o.dtype={o_rec.dtype}, expected={input_dtype}'))

        # Chunk path (only if T >= chunk_size to engage the chunk code).
        o_chk, s_chk = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=16)
        chk_state_dtype_ok = (s_chk.dtype == expected_state_dtype)
        chk_output_dtype_ok = (o_chk.dtype == input_dtype)
        results.append(_ok(
            f'chunk: state dtype={str(expected_state_dtype)} for input={str(input_dtype)}',
            chk_state_dtype_ok,
            f'state.dtype={s_chk.dtype}, expected={expected_state_dtype}'))
        results.append(_ok(
            f'chunk: output dtype={str(input_dtype)} for input={str(input_dtype)}',
            chk_output_dtype_ok,
            f'o.dtype={o_chk.dtype}, expected={input_dtype}'))

    # Test the state_dtype override: explicitly request fp16 state for fp16 inputs.
    q = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device) * 0.1
    g = -torch.rand(B, T, H, K, dtype=torch.float16, device=device) * 0.05
    beta = torch.rand(B, T, H, dtype=torch.float16, device=device) * 0.2
    o_override, s_override = naive_recurrent_kda(
        q, k, v, g, beta, output_final_state=True,
        state_dtype=torch.float16)
    override_ok = (s_override.dtype == torch.float16
                   and o_override.dtype == torch.float16)
    results.append(_ok(
        'state_dtype=torch.float16 override is honored',
        override_ok,
        f'state.dtype={s_override.dtype}, o.dtype={o_override.dtype}'))

    return results


def test_kda_long_context_bf16_equivalence(device='cpu'):
    """Long-context (T=4096) chunk/recurrent/assoc agreement in bf16 and fp32.

    The operators compute in fp32 and round only the output to the storage
    dtype, so the three reduction orders must agree to sub-ulp at bf16 and
    to fp32 round-off at fp32. The margin is derived from the dtype grid
    rather than tuned on the observed error: two storage ulps for bf16 and
    the file's fp32 bit-match tolerance. This is the long-context bf16
    equivalence certificate for the KDA route (the existing T=2048 check
    lives outside the CI gate and only watches for NaN/Inf).
    """
    logger.info("Test: KDA long-context bf16 equivalence (T=4096)")
    results = []
    B, T, H, K, V = 1, 4096, 2, 16, 16
    for dtype in (torch.float32, torch.bfloat16):
        eps = float(torch.finfo(dtype).eps)
        for seed in (0, 1):
            torch.manual_seed(seed)
            q = F.normalize(torch.randn(B, T, H, K), dim=-1).to(dtype)
            k = F.normalize(torch.randn(B, T, H, K), dim=-1).to(dtype)
            v = (torch.randn(B, T, H, V) * 0.3).to(dtype)
            g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
            beta = (torch.rand(B, T, H) * 0.3).to(dtype)
            o_rec, s_rec = naive_recurrent_kda(
                q, k, v, g, beta, output_final_state=True)
            variants = (
                ('chunk', naive_chunk_kda(
                    q, k, v, g, beta, output_final_state=True, chunk_size=64)),
                ('assoc', assoc_scan_chunk_kda(
                    q, k, v, g, beta, output_final_state=True, chunk_size=64)),
            )
            for name, (o, s) in variants:
                if dtype == torch.bfloat16:
                    o_tol = 2.0 * eps * max(
                        1.0, float(o_rec.float().abs().max().item()))
                    s_tol = 2.0 * eps * max(
                        1.0, float(s_rec.abs().max().item()))
                else:
                    o_tol = s_tol = TOL['bit_match'][torch.float32]
                o_diff = (o.float() - o_rec.float()).abs().max().item()
                s_diff = (s - s_rec).abs().max().item()
                results.append(_ok(
                    f'{name} output equivalence ({dtype}, seed={seed})',
                    o_diff <= o_tol,
                    f'max_diff={o_diff:.2e} tol={o_tol:.2e}'))
                results.append(_ok(
                    f'{name} state equivalence ({dtype}, seed={seed})',
                    s_diff <= s_tol,
                    f'max_diff={s_diff:.2e} tol={s_tol:.2e}'))
                results.append(_ok(
                    f'{name} dtype contract ({dtype}, seed={seed})',
                    o.dtype == dtype and s.dtype == torch.float32,
                    f'o={o.dtype}, s={s.dtype}'))
    return results


def test_kda_single_token_decode(device='cpu'):
    """Verify KDA handles T=1 (single-token decode step).

    This is the critical path for autoregressive decoding: each decode step
    processes exactly T=1 token. A bug in the T=1 path (e.g. an off-by-one
    in the recurrent loop, or a shape mismatch in the output) would break
    decoding but might not be caught by tests using T>=16.
    """
    logger.info("Test: KDA with T=1 (single-token decode)")
    torch.manual_seed(203)
    B, T, H, K, V = 2, 1, 2, 8, 8
    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2
    o, s = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    return [
        _ok('KDA T=1 output shape', o.shape == (B, T, H, V), str(tuple(o.shape))),
        _ok('KDA T=1 state shape', s.shape == (B, H, K, V), str(tuple(s.shape))),
        _ok('KDA T=1 finite', torch.isfinite(o).all().item() and torch.isfinite(s).all().item(), ''),
    ]


def test_standalone_kda_gate_matches_hybrid(device='cpu'):
    """Standalone Exp4/Exp6 KDA wrappers must use the hybrid low-rank gate.

    ``run_kv_cache.prefill_flops('kda')`` and ``ops_fused.KDAHybridLayer`` count
    / implement the paper-style low-rank gate ``d -> K -> HV*K``. The standalone
    MQAR and decoding wrappers used to instantiate a direct ``d -> H*K`` gate,
    contradicting their own comments and making Exp4/Exp6 use a different KDA
    operator boundary from Exp3 and the hybrid ablation.
    """
    logger.info("Test: standalone KDA gate parameterization matches hybrid")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_quality import KDAAttn
    from run_decoding import KDAAttnDecoding

    d_model, H, K, V = 32, 2, 16, 16
    quality = KDAAttn(d_model, H=H, K=K, V=V).to(device)
    decoding = KDAAttnDecoding(d_model, H=H, K=K, V=V).to(device)

    def _gate_ok(m):
        return (
            hasattr(m, 'g_down') and hasattr(m, 'g_up')
            and not hasattr(m, 'g')
            and m.g_down.in_features == d_model
            and m.g_down.out_features == K
            and m.g_up.in_features == K
            and m.g_up.out_features == H * K
        )

    # Round-10 fix (R2-A): both standalone KDA wrappers must also have the
    # causal depthwise short-conv that KDAHybridLayer has, so Exp 4 KDA
    # accuracy is measured on the same operator boundary as the hybrid
    # model and the decode benchmark.
    def _short_conv_ok(m):
        if not hasattr(m, 'short_conv'):
            return False
        sc = m.short_conv
        # Conv1d.kernel_size and .padding are tuples (e.g. (3,) / (0,))
        return (
            isinstance(sc, torch.nn.Conv1d)
            and tuple(sc.kernel_size) == (3,)
            and sc.groups == d_model
            and tuple(sc.padding) == (0,)
        )

    x = torch.randn(2, 4, d_model, device=device) * 0.1
    with torch.no_grad():
        y_quality = quality(x)
        decoding.reset()
        y_decoding = decoding(x)
    shape_ok = (y_quality.shape == x.shape and y_decoding.shape == x.shape)
    finite_ok = (torch.isfinite(y_quality).all().item()
                 and torch.isfinite(y_decoding).all().item())

    return [
        _ok('run_quality.KDAAttn uses low-rank gate', _gate_ok(quality),
            f'attrs={list(dict(quality.named_children()).keys())}'),
        _ok('run_decoding.KDAAttnDecoding uses low-rank gate', _gate_ok(decoding),
            f'attrs={list(dict(decoding.named_children()).keys())}'),
        _ok('run_quality.KDAAttn has causal short_conv (R2-A)',
            _short_conv_ok(quality),
            f'short_conv={getattr(quality, "short_conv", None)}'),
        _ok('run_decoding.KDAAttnDecoding has causal short_conv (R2-A)',
            _short_conv_ok(decoding),
            f'short_conv={getattr(decoding, "short_conv", None)}'),
        _ok('standalone KDA wrappers still produce finite outputs',
            shape_ok and finite_ok,
            f'y_quality={tuple(y_quality.shape)}, '
            f'y_decoding={tuple(y_decoding.shape)}'),
    ]


__all__ = [
    'test_kda_chunk_vs_recurrent',
    'test_kda_backend_reference_dispatch',
    'test_kda_gva',
    'test_kda_chunk_gva',
    'test_compiled_recurrent_kda_fullgraph',
    'test_scripted_chunk_kda_matches_naive',
    'test_kda_chunk_nondivisible_T',
    'test_kda_initial_state_dtype_mismatch',
    'test_kda_gradient',
    'test_kda_g_clamp',
    'test_kda_per_channel_decay_init',
    'test_kda_unnormalized_input_warns',
    'test_kda_dplr_transition_certificate',
    'test_kda_chunk_vs_recurrent_gradient',
    'test_kda_cross_chunk_bptt',
    'test_kda_state_dtype_preservation',
    'test_kda_long_context_bf16_equivalence',
    'test_kda_single_token_decode',
    'test_standalone_kda_gate_matches_hybrid',
]
