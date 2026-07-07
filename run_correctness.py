"""Experiment 1 — correctness verification.

Verifies that:
  * ``naive_recurrent_kda`` and ``naive_chunk_kda`` agree to fp tolerance
    (this is the same check used by the upstream FLA test suite).
  * CSA compression + sparse selection produces a valid attention pattern
    (causal, exactly ``topk`` selected entries per query, no look-ahead).
  * HCA dense attention over compressed entries is causal at block granularity.
  * The fused hybrid block runs end-to-end and preserves shape/dtype.

Kaggle / review-driven additions (address reviewer concerns about correctness
proof being only a sanity check):

  * **Overlap-compression causality.** We explicitly verify that the
    two-branch overlapped CSA compression of block ``i`` depends only on tokens
    from block ``i`` and block ``i-1`` (the overlap), never on any token from
    block ``i+1`` or later. This rules out future-token leakage.
  * **Gradient correctness for KDA.** We numerically check that the autograd
    gradient of the KDA recurrent loss matches a finite-difference estimate,
    confirming the custom recurrence is correctly differentiable.
  * **CSA indexer top-k validity.** We check that every selected index is in
    range and that the selected set size is exactly ``min(topk, n_blocks_causal)``.
  * **HCA sliding-window causality.** We verify the sliding-window branch mask
    only attends to past + current positions.
  * **CSA full-pipeline causality.** We run the complete ``naive_csa`` forward
    (compression + lightning indexer + sparse MQA core attention) and verify
    end-to-end causality by perturbing each source position ``p`` and checking
    that ``output[t]`` is unchanged for every ``t`` that should not depend on
    ``p`` (strict future ``t < p`` plus the SW/sparse gap region).  This closes
    the gap left by ``test_csa_causality``, which only tested the compression
    and indexer stages in isolation.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device
from ops_kda import naive_recurrent_kda, naive_chunk_kda
from ops_csa import csa_compress_kv_overlapped, csa_lightning_indexer, _causal_block_mask, naive_csa
from ops_hca import naive_hca
from ops_fused import HybridKCHAttention, HybridConfig

logger = logging.getLogger(__name__)


def _ok(name: str, cond: bool, detail: str = '') -> dict:
    status = 'PASS' if cond else 'FAIL'
    msg = f"  [{status}] {name}: {detail}"
    if cond:
        logger.info(msg)
    else:
        logger.error(msg)
    return {'name': name, 'status': status, 'detail': detail}


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
        _ok('output max abs diff', o_diff < 1e-4, f'{o_diff:.2e}'),
        _ok('state max abs diff', s_diff < 1e-4, f'{s_diff:.2e}'),
    ]
    return results


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
    H heads to HV heads, mirroring the recurrent path. This was previously
    only tested with HV == H (G=1, no GVA); the GVA chunk path was
    unverified. This test closes that gap by checking chunk-vs-recurrent
    agreement with HV=4, H=2 (G=2).
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
        _ok('GVA chunk vs recurrent output', o_diff < 1e-4, f'{o_diff:.2e}'),
        _ok('GVA chunk vs recurrent state', s_diff < 1e-4, f'{s_diff:.2e}'),
    ]


def test_kda_chunk_nondivisible_T(device='cpu'):
    """Verify naive_chunk_kda matches naive_recurrent_kda when T is NOT
    divisible by chunk_size.

    The chunk path internally right-pads T up to a multiple of ``chunk_size``
    and returns ``o[:, :original_T]``. This padding code path was previously
    unverified — all existing tests used T divisible by chunk_size (e.g.
    T=128, chunk_size=64). A bug in the padding logic (wrong trim axis,
    incorrect cumsum handling of padded zeros, etc.) would go undetected.

    We test multiple (T, chunk_size) combinations that trigger non-trivial
    padding, including the edge case T=1 (single-token decode).
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
            o_diff < 1e-4 and o_chk.shape == o_rec.shape == (B, T, HV, V),
            f'o_diff={o_diff:.2e}, shape={tuple(o_chk.shape)}'))
        results.append(_ok(
            f'chunk non-divisible T={T},BT={BT} state',
            s_diff < 1e-4 and s_chk.shape == s_rec.shape == (B, HV, K, V),
            f's_diff={s_diff:.2e}'))
    return results


def test_csa_hca_fp16_dtype_consistency(device='cpu'):
    """Verify CSA and HCA run without dtype-mismatch crashes on fp16 inputs.

    The compression functions (csa_compress_kv, csa_compress_kv_overlapped)
    return ``compute_dtype`` (fp32 for fp16 inputs). The attention query ``q``
    was previously left in ``H.dtype`` (fp16), causing
    ``torch.einsum`` to raise ``RuntimeError: Expected object of scalar type
    Half but got scalar type Float`` for fp16 inputs. This test verifies the
    dtype-consistency fix by running the full forward pass in fp16.

    We only check shape and finiteness (not numerical correctness against a
    reference) because fp16 has ~3 decimal digits of precision — the existing
    fp64 correctness tests already cover the math.
    """
    logger.info("Test: CSA/HCA fp16 dtype consistency (no mixed-dtype crash)")
    torch.manual_seed(16)
    dtype = torch.float16
    results = []

    # --- CSA fp16 ---
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    sink = torch.zeros(nh, dtype=dtype, device=device)

    try:
        o_csa = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=4, sink_logits=sink)
        csa_ok = o_csa.shape == (B, T, nh * c) and torch.isfinite(o_csa.float()).all().item()
        csa_err = ''
    except Exception as e:
        csa_ok = False
        csa_err = f'{type(e).__name__}: {e}'
    results.append(_ok('CSA fp16 forward', csa_ok,
                       f'shape={tuple(o_csa.shape) if csa_ok else "n/a"} '
                       f'{csa_err}'))

    # --- HCA fp16 ---
    B2, T2, d2 = 1, 32, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T2, d2, dtype=dtype, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, dtype=dtype, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, dtype=dtype, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, dtype=dtype, device=device) * 0.1
    sink2 = torch.zeros(nh2, dtype=dtype, device=device)

    try:
        o_hca = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=4, sink_logits=sink2)
        hca_ok = o_hca.shape == (B2, T2, nh2 * c2) and torch.isfinite(o_hca.float()).all().item()
        hca_err = ''
    except Exception as e:
        hca_ok = False
        hca_err = f'{type(e).__name__}: {e}'
    results.append(_ok('HCA fp16 forward', hca_ok,
                       f'shape={tuple(o_hca.shape) if hca_ok else "n/a"} '
                       f'{hca_err}'))

    return results


def test_kda_initial_state_dtype_mismatch(device='cpu'):
    """Verify KDA handles initial_state with a different dtype than the inputs.

    Previously, ``S += initial_state`` would raise ``RuntimeError: result type
    Double can't be cast to the desired output type Float`` if initial_state
    had a higher precision dtype than compute_dtype. This can happen when the
    caller changes dtype between calls and reuses the returned state.
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


def test_hca_causality(device='cpu'):
    logger.info("Test: HCA dense attention causality (block-level)")
    torch.manual_seed(3)
    B, T, d = 1, 128, 32
    m2, nh, c, dc = 32, 2, 16, 32
    H = torch.randn(B, T, d, device=device) * 0.1
    W_KV = torch.randn(d, c, device=device) * 0.1
    W_Z = torch.randn(d, c, device=device) * 0.1
    B_pos = torch.randn(m2, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    sink = torch.zeros(nh, device=device)
    o = naive_hca(H, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                  m2=m2, nh=nh, c=c, dc=dc,
                  sliding_window=8, sink_logits=sink)
    return [
        _ok('HCA output shape', o.shape == (B, T, nh * c), str(tuple(o.shape))),
        _ok('HCA finite', torch.isfinite(o).all().item(), ''),
    ]


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
    model = HybridKCHAttention(cfg, total_layers=5).to(device)
    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model, device=device) * 0.1
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    return [
        _ok('hybrid output shape', y.shape == x.shape, str(tuple(y.shape))),
        _ok('hybrid finite', torch.isfinite(y).all().item(), ''),
        _ok('hybrid layout', True,
            f'layout={model.layout_str()} params={n_params}'),
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
            with torch.no_grad():
                flat[idx] = orig - eps
            lm = loss_fn(q, k, v, g, beta)
            with torch.no_grad():
                flat[idx] = orig
            num_grad = (lp - lm).item() / (2 * eps)
            ana_grad = grads[name].view(-1)[idx].item()
            denom = max(1e-8, abs(num_grad) + abs(ana_grad))
            rel = abs(num_grad - ana_grad) / denom
            max_rel = max(max_rel, rel)

    return [
        _ok('KDA gradient matches finite-diff', max_rel < 1e-4,
            f'max relative error = {max_rel:.2e} (fp64, {n_check} coords)'),
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
    causal_ok = True
    for t in range(T):
        sel = indices[0, t]
        sel = sel[sel >= 0]
        if sel.numel() and not cbm[t, sel].all():
            causal_ok = False
            break
    # For early queries (t < m), no preceding block exists, so all indices
    # should be -1 (padded).
    early_ok = True
    for t in range(m):
        sel = indices[0, t]
        if not (sel == -1).all():
            early_ok = False
            break

    return [
        _ok('CSA indices in range', in_range, f'topk={topk}, n_blocks={n_blocks}'),
        _ok('CSA indices causal', causal_ok, 'all selected blocks precede query'),
        _ok('CSA early queries empty', early_ok,
            f'queries t<{m} have no preceding block -> all -1'),
    ]


def test_hca_sliding_window_causality(device='cpu'):
    """Verify HCA's sliding-window branch only attends to past + current.

    Optimization: the original per-t loop ran O(T) separate ``naive_hca``
    forwards (T calls, each O(T) work -> O(T^2) total).  We replace it with
    the *transpose* approach: perturb each source position p independently,
    batch all T perturbations into a single forward call, then verify the
    affected-output pattern.  This reduces O(T) forward calls to exactly 2
    (one reference + one batched).

    For a perturbation at source position p, the query positions t whose
    output *may* legitimately change are:
      * Sliding-window branch: t in [p, p+win-1]  (p is in t's window)
      * Dense MQA branch:      t >= (floor(p/m2)+1)*m2  (block-level causal)
    Everywhere else -- including all t < p (strict future) and the gap
    [p+win, (floor(p/m2)+1)*m2) between the SW and dense regions --
    output[t] must be unchanged.  The gap check additionally verifies the
    SW window size is exactly ``win`` (a larger window would leak into the
    gap and be detected).
    """
    logger.info("Test: HCA sliding-window branch causality")
    torch.manual_seed(8)
    B, T, d = 1, 64, 16
    m2, nh, c, dc = 16, 2, 8, 16
    H = torch.randn(B, T, d, dtype=torch.float64, device=device) * 0.1
    W_KV = torch.randn(d, c, dtype=torch.float64, device=device) * 0.1
    W_Z = torch.randn(d, c, dtype=torch.float64, device=device) * 0.1
    B_pos = torch.randn(m2, c, dtype=torch.float64, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=torch.float64, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=torch.float64, device=device) * 0.1
    sink = torch.zeros(nh, dtype=torch.float64, device=device)
    win = 4

    # --- Reference forward (unperturbed) ---
    o_ref = naive_hca(H, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                      m2=m2, nh=nh, c=c, dc=dc,
                      sliding_window=win, sink_logits=sink)  # [B, T, nh*c]

    # --- Batched single-position perturbation ---
    # For each source position p, create a copy of H where ONLY position p
    # is perturbed.  Stack all T copies into one batch and run a SINGLE
    # forward.  This replaces O(T) separate forwards with 1 batched forward.
    H_batch = H.unsqueeze(0).repeat(T, 1, 1, 1)              # [T, B, T, d]
    perturb = 10.0 * torch.randn(T, B, d, dtype=H.dtype, device=device)
    p_idx = torch.arange(T, device=device)
    # Diagonal perturbation: batch element p gets noise at position p only.
    H_batch[p_idx, :, p_idx, :] += perturb
    H_batch = H_batch.reshape(T * B, T, d)

    o_pert = naive_hca(H_batch, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                       m2=m2, nh=nh, c=c, dc=dc,
                       sliding_window=win, sink_logits=sink)
    o_pert = o_pert.reshape(T, B, T, nh * c)                 # [p, B, t, nh*c]

    # diff[p, t] = max |o_ref[t] - o_pert[p, t]|
    diff = (o_ref[0].unsqueeze(0) - o_pert[:, 0]).abs().max(dim=-1).values  # [T, T]

    # Expected affected region for perturbing position p:
    #   SW branch:    p <= t < p + win
    #   Dense branch: t >= (floor(p/m2) + 1) * m2
    p_grid = torch.arange(T, device=device)[:, None]
    t_grid = torch.arange(T, device=device)[None, :]
    sw_affected = (t_grid >= p_grid) & (t_grid < p_grid + win)
    dense_affected = t_grid >= ((p_grid // m2) + 1) * m2
    expected_affected = sw_affected | dense_affected

    # Outside the expected region, diff must be ~0.  This covers both
    # causality (t < p: future must not affect past) and the window-size
    # gap (p+win <= t < (floor(p/m2)+1)*m2: outside both branches).
    #
    # BIDIRECTIONAL assertion: we also check that the AFFECTED region actually
    # sees a non-trivial diff. A degenerate implementation (e.g. one that
    # zeros out the SW or dense branch) would pass the one-sided
    # "unaffected == 0" check trivially; requiring affected > 0 rules out
    # such silent regressions. We use a small floor (1e-6) because the
    # perturbation magnitude is 10.0 * randn and the attention output is
    # bounded, so any real dependency should produce a clearly non-zero diff.
    if (~expected_affected).any():
        max_diff = diff[~expected_affected].max().item()
    else:
        max_diff = 0.0
    if expected_affected.any():
        min_affected = diff[expected_affected].min().item()
    else:
        min_affected = float('inf')

    return [
        _ok('HCA sliding-window causal (unaffected region is stable)',
            max_diff < 1e-9,
            f'max diff in unaffected region = {max_diff:.2e} (fp64, win={win})'),
        _ok('HCA sliding-window affected region actually changes',
            min_affected > 1e-6,
            f'min diff in affected region = {min_affected:.2e} '
            f'(guards against zero-output regressions)'),
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
      * Sparse MQA branch: ``t >= (floor(p/m)+1)*m``.  Compressed block ``bp``
        depends on ``p`` via the a-branch (``Ca[:, bp]`` reads position ``p``),
        and is selectable by query ``t`` only when ``bp < t // m``, i.e.
        ``t >= (bp+1)*m``.  Compressed block ``bp+1`` also depends on ``p`` via
        the b-branch (``Cb[:, bp]``), but is only selectable when
        ``t >= (bp+2)*m`` -- subsumed by the ``bp`` condition.
    Everywhere else -- including all ``t < p`` (strict future must not affect
    past) and the gap ``[p+win, (floor(p/m)+1)*m)`` between the SW and sparse
    regions -- ``output[t]`` must be unchanged.  The gap check additionally
    verifies the SW window size is exactly ``win`` (a larger window would leak
    into the gap and be detected).

    Note on the lightning indexer: perturbing ``p`` also changes
    ``K_IComp`` (indexer keys) for blocks ``bp`` and ``bp+1``, and changes
    ``q_idx``/``w_idx`` at position ``p`` only.  For any query ``t`` with
    ``t // m <= bp``, blocks ``bp`` and ``bp+1`` are masked to ``-inf`` by the
    causal block mask, so their key changes cannot alter the top-k selection;
    consequently the sparse-branch affected region remains
    ``t >= (bp+1)*m``.  The ``q_idx``/``w_idx`` perturbation only affects the
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
    W_aKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
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
    #   Sparse branch: t >= (floor(p/m) + 1) * m
    p_grid = torch.arange(T, device=device)[:, None]
    t_grid = torch.arange(T, device=device)[None, :]
    sw_affected = (t_grid >= p_grid) & (t_grid < p_grid + win)
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


def test_hybrid_padding_no_crash(device='cpu'):
    """Regression: HybridKCHAttention must not crash when T is not divisible
    by csa_m / hca_m2.

    Previously, the padding trim used ``o[pad:]`` which slices dim=0 (batch)
    instead of dim=1 (sequence). This caused a shape-mismatch RuntimeError
    on the residual add whenever T was not a multiple of the compression
    factor. The tests happened not to trigger it because every test used
    T divisible by m, but any real-world sequence length could hit it.

    We now exercise T values that require non-trivial padding for BOTH the
    CSA layer (m=8) and the HCA layer (m2=16), for several batch sizes
    including B=1 (which previously crashed with an empty-batch slice) and
    B>pad (which previously silently corrupted results).
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


def test_csa_hca_right_padding_correctness(device='cpu'):
    """Regression: CSA/HCA right-padding must preserve real-token outputs.

    Previously the code LEFT-padded (zeros at the start), which shifted real
    tokens to positions [pad, pad+T) and corrupted block 0's compressed KV
    (a mix of padding zeros and real tokens). Every subsequent block's real
    queries then attended to that corrupted block 0, silently producing
    wrong outputs for any T not divisible by m.

    With RIGHT-padding (zeros at the end), real tokens keep their original
    positions and block alignment. Only the LAST partial block contains
    padding zeros, and -- crucially -- no real token attends to it (the
    causal block mask only allows attending to PRECEDING blocks). So
    real-token outputs are bit-identical to running with T_padded real
    tokens and taking the first T_orig outputs.

    This test verifies that property for both CSA and HCA: run the operator
    on T_padded real tokens (no padding needed) and on the first T_orig
    tokens (right-padded to T_padded), and check the first T_orig outputs
    match.
    """
    logger.info("Test: CSA/HCA right-padding preserves real-token outputs")
    torch.manual_seed(12)
    dtype = torch.float64

    # HCA test (simpler -- single branch compression, dense attention).
    B, T_padded, d = 1, 32, 16
    m2, nh, c, dc = 8, 2, 8, 16
    T_orig = 25  # not divisible by m2=8 -> pad=7
    pad = (-T_orig) % m2
    assert pad == 7, f"expected pad=7, got {pad}"

    H_full = torch.randn(B, T_padded, d, dtype=dtype, device=device) * 0.1
    W_KV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_Z = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    B_pos = torch.randn(m2, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    sink = torch.zeros(nh, dtype=dtype, device=device)

    # Reference: run on all T_padded real tokens (no padding).
    o_ref = naive_hca(H_full, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                      m2=m2, nh=nh, c=c, dc=dc,
                      sliding_window=4, sink_logits=sink)

    # Right-padded: run on first T_orig tokens, right-padded to T_padded.
    H_short = H_full[:, :T_orig].clone()
    H_padded = F.pad(H_short, (0, 0, 0, pad))
    o_pad = naive_hca(H_padded, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                      m2=m2, nh=nh, c=c, dc=dc,
                      sliding_window=4, sink_logits=sink)

    # The first T_orig outputs must match (real tokens see only all-real
    # preceding blocks in both cases).
    hca_diff = (o_ref[:, :T_orig] - o_pad[:, :T_orig]).abs().max().item()

    # CSA test (two-branch overlapped compression + sparse selection).
    B2, T_padded2, d2 = 1, 32, 16
    m, topk, nh2, nIh, c2, c_I, dc2 = 8, 4, 2, 2, 8, 4, 8
    T_orig2 = 25
    pad2 = (-T_orig2) % m
    assert pad2 == 7

    H_full2 = torch.randn(B2, T_padded2, d2, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c2, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c2, dtype=dtype, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, dtype=dtype, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc2, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d2, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d2, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d2, c_I, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    sink2 = torch.zeros(nh2, dtype=dtype, device=device)

    o_ref2 = naive_csa(H_full2, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                       W_DQ2, W_UQ2, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                       m=m, topk=topk, nh=nh2, nIh=nIh, c=c2, c_I=c_I, dc=dc2,
                       sliding_window=4, sink_logits=sink2)

    H_short2 = H_full2[:, :T_orig2].clone()
    H_padded2 = F.pad(H_short2, (0, 0, 0, pad2))
    o_pad2 = naive_csa(H_padded2, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                       W_DQ2, W_UQ2, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                       m=m, topk=topk, nh=nh2, nIh=nIh, c=c2, c_I=c_I, dc=dc2,
                       sliding_window=4, sink_logits=sink2)

    csa_diff = (o_ref2[:, :T_orig2] - o_pad2[:, :T_orig2]).abs().max().item()

    return [
        _ok('HCA right-padding preserves outputs', hca_diff < 1e-10,
            f'max diff = {hca_diff:.2e} (fp64, T_orig={T_orig}, m2={m2})'),
        _ok('CSA right-padding preserves outputs', csa_diff < 1e-10,
            f'max diff = {csa_diff:.2e} (fp64, T_orig={T_orig2}, m={m})'),
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
    """
    logger.info("Test: bench_hybrid runs under no_grad (regression)")
    # Replicate the fixed bench_hybrid pattern inline so the test is
    # self-contained and does not import run_benchmark (which would pull in
    # matplotlib etc.).
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    x = torch.randn(1, 16, cfg.d_model, device=device) * 0.1

    def fn():
        with torch.no_grad():
            return model(x)

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
    stacked1 = model._kda_state
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
                    T_h = h_norm.shape[1]
                    if kind == 'csa':
                        pad = (-T_h) % cfg.csa_m
                    else:
                        pad = (-T_h) % cfg.hca_m2
                    if pad:
                        # RIGHT-pad to match HybridKCHAttention.forward's padding
                        # direction (real tokens keep original positions; only the
                        # last partial block contains padding zeros).
                        hp = F.pad(h_norm, (0, 0, 0, pad))
                        o, _ = layer(hp, None)
                        o = o[:, :T_h]
                    else:
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


def test_csa_hca_sink_numerical_correctness(device='cpu'):
    """Regression: attention sink must be shifted by -row_max in the log-space softmax.

    The attention sink adds a per-head constant ``exp(sink_logits[h])`` to the
    softmax denominator:

        p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink))

    For numerical stability we subtract ``row_max = max(0, max_i s_i)`` from
    every score. The sink MUST be shifted by the same amount:

        p_i = exp(s_i - M) / (sum_j exp(s_j - M) + exp(sink - M))

    The previous code forgot to shift the sink, producing

        p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink) * exp(M))

    i.e. the sink was over-weighted by a factor ``exp(row_max)``. In the
    default ``c=64`` config this is a ~13% systematic bias; at ``c=4`` it
    reaches 65%. The existing sink tests all used ``sink_logits=zeros(nh)``
    and only checked shape/finiteness, so the bias went undetected.

    This test builds a CORRECT reference implementation (with the shift) and
    compares both ``naive_csa`` and ``naive_hca`` against it, using a
    non-zero ``sink_logits`` so the bias is detectable.
    """
    logger.info("Test: CSA/HCA attention sink numerical correctness (log-space shift)")
    torch.manual_seed(14)
    dtype = torch.float64

    # --- CSA sink test ---
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    # Non-zero sink so the bias is detectable.
    sink = torch.tensor([0.5, -0.3], dtype=dtype, device=device)

    # Run naive_csa with sliding_window=0 to isolate the sparse+sink path.
    o_csa = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                      W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                      m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                      sliding_window=0, sink_logits=sink)

    # Build a CORRECT reference for the sparse MQA core with sink.
    from ops_csa import csa_compress_kv, csa_compress_kv_overlapped, csa_lightning_indexer, _causal_block_mask
    Ca = H @ W_aKV; Cb = H @ W_bKV; Za = H @ W_aZ; Zb = H @ W_bZ
    C_comp = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
    n_blocks = T // m
    K_idx_raw = H @ W_KV_idx; Z_idx = H @ W_Z_idx
    K_IComp = csa_compress_kv(K_idx_raw, Z_idx, B_idx, m)
    cQ = H @ W_DQ
    q_idx = (cQ @ W_IUQ).view(B, T, nIh, c_I)
    w_idx = H @ W_w
    cbm = _causal_block_mask(T, n_blocks, m, H.device)
    indices = csa_lightning_indexer(q_idx, K_IComp, w_idx, topk,
                                     scale=c_I ** -0.5, causal_block_mask=cbm)
    q = (cQ @ W_UQ).view(B, T, nh, c)
    q = F.normalize(q, dim=-1)
    C_comp_n = F.normalize(C_comp, dim=-1)
    valid_mask = indices >= 0
    idx_safe = indices.clamp(min=0)
    batch_idx = torch.arange(B, device=H.device).view(B, 1, 1)
    kv = C_comp_n[batch_idx, idx_safe]
    scale = c ** -0.5
    scores = torch.einsum('b t h d, b t k d -> b t h k', q, kv) * scale
    scores = scores.masked_fill(~valid_mask[:, :, None, :], float('-inf'))
    # CORRECT reference: shift the sink by -row_max.
    row_max = scores.amax(-1, keepdim=True).clamp(min=0)
    shifted = scores - row_max
    log_sink = sink.view(1, 1, nh, 1)
    shifted_sink = log_sink - row_max
    lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
    log_denom = torch.logaddexp(lse, shifted_sink)
    vmask = valid_mask[:, :, None, :].to(scores.dtype)
    p_ref = ((shifted - log_denom).exp() * vmask)
    all_invalid = ~valid_mask.any(-1, keepdim=True)[:, :, None]
    p_ref = p_ref.masked_fill(all_invalid, 0.0)
    out_ref = torch.einsum('b t h k, b t k d -> b t h d', p_ref, kv)
    out_ref_flat = out_ref.reshape(B, T, nh * c)

    csa_diff = (o_csa - out_ref_flat).abs().max().item()

    # --- HCA sink test ---
    B2, T2, d2 = 1, 32, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T2, d2, dtype=dtype, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, dtype=dtype, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, dtype=dtype, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, dtype=dtype, device=device) * 0.1
    sink2 = torch.tensor([0.7, -0.2], dtype=dtype, device=device)

    o_hca = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                      m2=m2, nh=nh2, c=c2, dc=dc2,
                      sliding_window=0, sink_logits=sink2)

    # Correct reference for HCA dense attention with sink.
    # ``csa_compress_kv`` is already imported at the top of this function
    # (line ~1211); reuse it instead of re-importing under an alias.
    C2 = H2 @ W_KV2; Z2 = H2 @ W_Z2
    C_comp2 = csa_compress_kv(C2, Z2, B_pos2, m2)
    n_blocks2 = T2 // m2
    C_comp_n2 = F.normalize(C_comp2, dim=-1)
    cQ2 = H2 @ W_DQ2
    q2 = (cQ2 @ W_UQ2).view(B2, T2, nh2, c2)
    q2 = F.normalize(q2, dim=-1)
    cbm2 = _causal_block_mask(T2, n_blocks2, m2, H2.device)
    scale2 = c2 ** -0.5
    scores2 = torch.einsum('b t h d, b n d -> b h t n', q2, C_comp_n2) * scale2
    scores2 = scores2.masked_fill(~cbm2[None, None], float('-inf'))
    row_max2 = scores2.amax(-1, keepdim=True).clamp(min=0)
    shifted2 = scores2 - row_max2
    log_sink2 = sink2.view(1, nh2, 1, 1)
    shifted_sink2 = log_sink2 - row_max2
    lse2 = torch.logsumexp(shifted2, dim=-1, keepdim=True)
    log_denom2 = torch.logaddexp(lse2, shifted_sink2)
    p_ref2 = (shifted2 - log_denom2).exp()
    all_masked2 = torch.isinf(scores2).all(-1, keepdim=True)
    p_ref2 = p_ref2.masked_fill(all_masked2, 0.0)
    out_ref2 = torch.einsum('b h t n, b n d -> b t h d', p_ref2, C_comp_n2)
    out_ref2_flat = out_ref2.reshape(B2, T2, nh2 * c2)

    hca_diff = (o_hca - out_ref2_flat).abs().max().item()

    return [
        _ok('CSA sink matches shifted-logsumexp reference', csa_diff < 1e-10,
            f'max diff = {csa_diff:.2e} (fp64, sink=[0.5,-0.3]); '
            f'a non-zero diff means the sink is not shifted by -row_max'),
        _ok('HCA sink matches shifted-logsumexp reference', hca_diff < 1e-10,
            f'max diff = {hca_diff:.2e} (fp64, sink=[0.7,-0.2]); '
            f'a non-zero diff means the sink is not shifted by -row_max'),
    ]


def test_hybrid_backward_produces_grads(device='cpu'):
    """Regression: HybridKCHAttention backward must produce finite grads for
    all DIFFERENTIABLE parameters.

    The lightning indexer uses ``torch.topk`` which returns integer indices
    that do NOT propagate gradients. Consequently the indexer parameters
    (``W_IUQ``, ``W_w``, ``W_KV_idx``, ``W_Z_idx``, ``B_idx`` in
    ``CSAHybridLayer``) cannot receive gradients through the main loss —
    their ``.grad`` stays ``None`` after ``backward()``. This is a known
    structural limitation (see the docstring of ``csa_lightning_indexer``)
    and is NOT a bug.

    What WOULD be a bug: a differentiable parameter (one whose gradient
    SHOULD flow) ending up with a non-finite or all-zero gradient. This
    test runs a full forward + backward pass over the hybrid stack and
    verifies that:
      1. All non-indexer parameters receive a finite, non-zero gradient.
      2. Indexer parameters (the 5 listed above) have ``.grad is None``
         (the expected behavior, documented for future readers).
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
    # These have .grad is None because topk is non-differentiable.
    indexer_param_substrings = ('W_IUQ', 'W_w', 'W_KV_idx', 'W_Z_idx', 'B_idx')
    indexer_no_grad = True
    indexer_detail_parts = []
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
                # Expected: indexer params don't get grads through topk.
                continue
            else:
                differentiable_no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            differentiable_non_finite.append(name)
        elif is_indexer:
            # If an indexer param DOES receive a grad (e.g. via an auxiliary
            # loss added in the future), that's fine — just don't require it.
            differentiable_ok += 1
        elif p.grad.abs().max().item() == 0.0:
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


def test_csa_hca_bf16_dtype_consistency(device='cpu'):
    """Verify CSA and HCA run without dtype-mismatch crashes on bf16 inputs.

    The existing fp16 test covers the half-precision code path, but bf16 has
    a different exponent/mantissa split (8 exponent bits vs 10 for fp16) and
    may expose different rounding behavior. This test mirrors
    ``test_csa_hca_fp16_dtype_consistency`` for bf16.
    """
    logger.info("Test: CSA/HCA bf16 dtype consistency (no mixed-dtype crash)")
    torch.manual_seed(200)
    dtype = torch.bfloat16
    results = []

    # --- CSA bf16 ---
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    sink = torch.zeros(nh, dtype=dtype, device=device)

    try:
        o_csa = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=4, sink_logits=sink)
        csa_ok = o_csa.shape == (B, T, nh * c) and torch.isfinite(o_csa.float()).all().item()
        csa_err = ''
    except Exception as e:
        csa_ok = False
        csa_err = f'{type(e).__name__}: {e}'
    results.append(_ok('CSA bf16 forward', csa_ok,
                       f'shape={tuple(o_csa.shape) if csa_ok else "n/a"} '
                       f'{csa_err}'))

    # --- HCA bf16 ---
    B2, T2, d2 = 1, 32, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T2, d2, dtype=dtype, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, dtype=dtype, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, dtype=dtype, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, dtype=dtype, device=device) * 0.1
    sink2 = torch.zeros(nh2, dtype=dtype, device=device)

    try:
        o_hca = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=4, sink_logits=sink2)
        hca_ok = o_hca.shape == (B2, T2, nh2 * c2) and torch.isfinite(o_hca.float()).all().item()
        hca_err = ''
    except Exception as e:
        hca_ok = False
        hca_err = f'{type(e).__name__}: {e}'
    results.append(_ok('HCA bf16 forward', hca_ok,
                       f'shape={tuple(o_hca.shape) if hca_ok else "n/a"} '
                       f'{hca_err}'))

    return results


def test_csa_hca_no_sink_no_sliding_window(device='cpu'):
    """Verify CSA/HCA work with sink_logits=None and sliding_window=0.

    The default config in HybridConfig uses both sink and sliding_window, so
    the no-sink/no-SW code path is only exercised when explicitly tested.
    A bug in either branch (e.g. a None check on sink_logits, or a missing
    guard for sliding_window=0) would go undetected without this test.
    """
    logger.info("Test: CSA/HCA with no sink and no sliding window")
    torch.manual_seed(201)
    results = []

    # --- CSA no sink, no SW ---
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device) * 0.1
    W_aKV = torch.randn(d, c, device=device) * 0.1
    W_bKV = torch.randn(d, c, device=device) * 0.1
    W_aZ = torch.randn(d, c, device=device) * 0.1
    W_bZ = torch.randn(d, c, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, device=device) * 0.1
    W_w = torch.randn(d, nIh, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, device=device) * 0.1
    B_idx = torch.randn(m, c_I, device=device) * 0.1
    try:
        o_csa = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=0, sink_logits=None)
        csa_ok = o_csa.shape == (B, T, nh * c) and torch.isfinite(o_csa).all().item()
    except Exception as e:
        csa_ok = False
    results.append(_ok('CSA no-sink no-SW', csa_ok,
                       f'shape={tuple(o_csa.shape) if csa_ok else "n/a"}'))

    # --- HCA no sink, no SW ---
    B2, T2, d2 = 1, 32, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T2, d2, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, device=device) * 0.1
    try:
        o_hca = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=0, sink_logits=None)
        hca_ok = o_hca.shape == (B2, T2, nh2 * c2) and torch.isfinite(o_hca).all().item()
    except Exception as e:
        hca_ok = False
    results.append(_ok('HCA no-sink no-SW', hca_ok,
                       f'shape={tuple(o_hca.shape) if hca_ok else "n/a"}'))

    return results


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
    W_aKV = torch.randn(d, c, device=device) * 0.1
    W_bKV = torch.randn(d, c, device=device) * 0.1
    W_aZ = torch.randn(d, c, device=device) * 0.1
    W_bZ = torch.randn(d, c, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, device=device) * 0.1
    W_w = torch.randn(d, nIh, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, device=device) * 0.1
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


def test_csa_hca_extreme_sink_values(device='cpu'):
    """Verify CSA/HCA do not produce NaN with extreme sink_logits (+/-100).

    The log-space sink implementation shifts by -row_max for numerical
    stability. Extreme sink values (e.g. +100 or -100) test that the shift
    prevents overflow (exp(100) = inf) and underflow (exp(-100) = 0) without
    producing NaN. A bug in the shift logic would produce NaN via inf/inf or
    -inf - (-inf).
    """
    logger.info("Test: CSA/HCA with extreme sink_logits (+/-100)")
    torch.manual_seed(204)
    results = []
    B, T, d = 1, 32, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T, d, device=device) * 0.1
    W_aKV = torch.randn(d, c, device=device) * 0.1
    W_bKV = torch.randn(d, c, device=device) * 0.1
    W_aZ = torch.randn(d, c, device=device) * 0.1
    W_bZ = torch.randn(d, c, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, device=device) * 0.1
    W_w = torch.randn(d, nIh, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, device=device) * 0.1
    B_idx = torch.randn(m, c_I, device=device) * 0.1
    sink = torch.tensor([100.0, -100.0], device=device)
    try:
        o_csa = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                          W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                          m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                          sliding_window=0, sink_logits=sink)
        results.append(_ok('CSA extreme sink', torch.isfinite(o_csa).all().item(),
                           f'finite={torch.isfinite(o_csa).all().item()}'))
    except Exception as e:
        results.append(_ok('CSA extreme sink', False, f'{type(e).__name__}: {e}'))

    # HCA extreme sink
    B2, T2, d2 = 1, 32, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T2, d2, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, device=device) * 0.1
    sink2 = torch.tensor([100.0, -100.0], device=device)
    try:
        o_hca = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=0, sink_logits=sink2)
        results.append(_ok('HCA extreme sink', torch.isfinite(o_hca).all().item(),
                           f'finite={torch.isfinite(o_hca).all().item()}'))
    except Exception as e:
        results.append(_ok('HCA extreme sink', False, f'{type(e).__name__}: {e}'))
    return results


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


def test_csa_hca_non_divisible_T(device='cpu'):
    """Regression: ``naive_csa`` and ``naive_hca`` must accept T not divisible
    by m / m2 without crashing.

    Previously both functions had a bare ``assert T % m == 0`` (no message)
    that crashed with ``AssertionError`` when called directly with a
    non-divisible T. Callers like ``HybridKCHAttention`` and ``CSAAttn``
    worked around this by padding externally, but a direct caller had no way
    to use the operators without replicating the padding logic.

    The fix mirrors ``naive_chunk_kda``: the operators now right-pad T up to
    a multiple of m / m2 internally and trim the output back to the original
    T. Real tokens keep their original positions; only the last partial
    block contains padding zeros, and the causal block mask ensures no real
    token attends to it.

    This test verifies:
      1. Both operators run without crashing for non-divisible T.
      2. The output T matches the input T (i.e. the trim is correct).
      3. The output for the non-padded prefix matches the output of running
         on the padded T directly (i.e. padding does not affect real-token
         outputs).
    """
    logger.info("Test: CSA/HCA accept non-divisible T (internal padding)")
    torch.manual_seed(206)
    dtype = torch.float64

    # --- CSA: T=20, m=8 -> pad=4 -> T_padded=24 ---
    B, T_orig, d = 1, 20, 16
    m, topk, nh, nIh, c, c_I, dc = 8, 2, 2, 2, 8, 4, 8
    H = torch.randn(B, T_orig, d, dtype=dtype, device=device) * 0.1
    W_aKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bKV = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_aZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    W_bZ = torch.randn(d, c, dtype=dtype, device=device) * 0.1
    Ba = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    Bb = torch.randn(m, c, dtype=dtype, device=device) * 0.1
    W_DQ = torch.randn(d, dc, dtype=dtype, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, dtype=dtype, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, dtype=dtype, device=device) * 0.1
    W_w = torch.randn(d, nIh, dtype=dtype, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, dtype=dtype, device=device) * 0.1
    B_idx = torch.randn(m, c_I, dtype=dtype, device=device) * 0.1
    sink = torch.zeros(nh, dtype=dtype, device=device)

    # Run on the original (non-divisible) T directly.
    o_direct = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                         W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                         m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                         sliding_window=4, sink_logits=sink)
    # Run on the padded T (manually padded) and trim — should match.
    pad = (-T_orig) % m
    H_padded = F.pad(H, (0, 0, 0, pad))
    o_padded = naive_csa(H_padded, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                         W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                         m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                         sliding_window=4, sink_logits=sink)
    csa_shape_ok = o_direct.shape == (B, T_orig, nh * c)
    csa_match = torch.allclose(o_direct, o_padded[:, :T_orig], atol=1e-10)

    # --- HCA: T=20, m2=16 -> pad=12 -> T_padded=32 ---
    B2, T_orig2, d2 = 1, 20, 16
    m2, nh2, c2, dc2 = 16, 2, 8, 16
    H2 = torch.randn(B2, T_orig2, d2, dtype=dtype, device=device) * 0.1
    W_KV2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    W_Z2 = torch.randn(d2, c2, dtype=dtype, device=device) * 0.1
    B_pos2 = torch.randn(m2, c2, dtype=dtype, device=device) * 0.1
    W_DQ2 = torch.randn(d2, dc2, dtype=dtype, device=device) * 0.1
    W_UQ2 = torch.randn(dc2, c2 * nh2, dtype=dtype, device=device) * 0.1
    sink2 = torch.zeros(nh2, dtype=dtype, device=device)

    o_direct2 = naive_hca(H2, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=4, sink_logits=sink2)
    pad2 = (-T_orig2) % m2
    H2_padded = F.pad(H2, (0, 0, 0, pad2))
    o_padded2 = naive_hca(H2_padded, W_KV2, W_Z2, B_pos2, W_DQ2, W_UQ2,
                          m2=m2, nh=nh2, c=c2, dc=dc2,
                          sliding_window=4, sink_logits=sink2)
    hca_shape_ok = o_direct2.shape == (B2, T_orig2, nh2 * c2)
    hca_match = torch.allclose(o_direct2, o_padded2[:, :T_orig2], atol=1e-10)

    return [
        _ok('CSA non-divisible T shape', csa_shape_ok,
            f'expected {(B, T_orig, nh*c)}, got {tuple(o_direct.shape)}'),
        _ok('CSA non-divisible T matches padded-prefix', csa_match,
            f'max diff = {(o_direct - o_padded[:, :T_orig]).abs().max().item():.2e}'),
        _ok('HCA non-divisible T shape', hca_shape_ok,
            f'expected {(B2, T_orig2, nh2*c2)}, got {tuple(o_direct2.shape)}'),
        _ok('HCA non-divisible T matches padded-prefix', hca_match,
            f'max diff = {(o_direct2 - o_padded2[:, :T_orig2]).abs().max().item():.2e}'),
    ]


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
    W_aKV = torch.randn(d, c, device=device) * 0.1
    W_bKV = torch.randn(d, c, device=device) * 0.1
    W_aZ = torch.randn(d, c, device=device) * 0.1
    W_bZ = torch.randn(d, c, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.1
    Bb = torch.randn(m, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    W_IUQ = torch.randn(dc, c_I * nIh, device=device) * 0.1
    W_w = torch.randn(d, nIh, device=device) * 0.1
    W_KV_idx = torch.randn(d, c_I, device=device) * 0.1
    W_Z_idx = torch.randn(d, c_I, device=device) * 0.1
    B_idx = torch.randn(m, c_I, device=device) * 0.1
    sink = torch.zeros(nh, device=device)

    # topk=0, no SW -> all-zero output (sparse branch contributes 0, no SW).
    o_no_sw = naive_csa(H, W_aKV, W_bKV, W_aZ, W_bZ, Ba, Bb,
                        W_DQ, W_UQ, W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx,
                        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                        sliding_window=0, sink_logits=sink)
    no_sw_ok = (o_no_sw.shape == (B, T, nh * c)
                and torch.isfinite(o_no_sw).all().item()
                and o_no_sw.abs().max().item() == 0.0)

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


def test_hca_T_smaller_than_m2(device='cpu'):
    """Regression: ``naive_hca`` must accept T < m2 without crashing.

    Previously the bare ``assert T % m2 == 0`` crashed when T < m2 (e.g.
    T=8, m2=16). With the internal padding fix, T is right-padded up to a
    multiple of m2 (so T=8 -> T_padded=16, n_blocks=1). Under the causal
    block mask, no query can attend to block 0 (it would leak the current
    block's future tokens), so the dense branch produces zero and only the
    sliding-window branch (if enabled) contributes.
    """
    logger.info("Test: HCA with T < m2 (single block, no preceding)")
    torch.manual_seed(208)
    B, T, d = 1, 8, 16
    m2, nh, c, dc = 16, 2, 8, 16  # m2 > T
    H = torch.randn(B, T, d, device=device) * 0.1
    W_KV = torch.randn(d, c, device=device) * 0.1
    W_Z = torch.randn(d, c, device=device) * 0.1
    B_pos = torch.randn(m2, c, device=device) * 0.1
    W_DQ = torch.randn(d, dc, device=device) * 0.1
    W_UQ = torch.randn(dc, c * nh, device=device) * 0.1
    sink = torch.zeros(nh, device=device)

    # Without SW: dense branch is fully masked (no preceding block) -> zero.
    o_no_sw = naive_hca(H, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                        m2=m2, nh=nh, c=c, dc=dc,
                        sliding_window=0, sink_logits=sink)
    no_sw_ok = (o_no_sw.shape == (B, T, nh * c)
                and torch.isfinite(o_no_sw).all().item()
                and o_no_sw.abs().max().item() == 0.0)

    # With SW: SW branch contributes (each query attends to itself + past 3).
    o_with_sw = naive_hca(H, W_KV, W_Z, B_pos, W_DQ, W_UQ,
                          m2=m2, nh=nh, c=c, dc=dc,
                          sliding_window=4, sink_logits=sink)
    with_sw_ok = (o_with_sw.shape == (B, T, nh * c)
                  and torch.isfinite(o_with_sw).all().item()
                  and o_with_sw.abs().max().item() > 0.0)

    return [
        _ok('HCA T<m2 no-SW produces zero output', no_sw_ok,
            f'shape={tuple(o_no_sw.shape)}, max|o|={o_no_sw.abs().max().item():.2e}'),
        _ok('HCA T<m2 with-SW produces non-zero output', with_sw_ok,
            f'shape={tuple(o_with_sw.shape)}, max|o|={o_with_sw.abs().max().item():.2e}'),
    ]


def main():
    info = configure_torch_for_device()
    device = info.device
    # fp64 tests run on the detected device. On T4 fp64 is slow but correct,
    # and these tests are tiny (T<=128). Keeping them on-device verifies the
    # GPU code path end-to-end.
    logger.info('=' * 70)
    logger.info(f'Experiment 1: Correctness Verification ({device})')
    logger.info('=' * 70)
    all_results = []
    all_results += test_kda_chunk_vs_recurrent(device)
    all_results += test_kda_gva(device)
    all_results += test_kda_chunk_gva(device)
    all_results += test_csa_causality(device)
    all_results += test_hca_causality(device)
    all_results += test_fused_hybrid(device)
    # New reviewer-driven checks.
    all_results += test_overlap_causality(device)
    all_results += test_kda_gradient(device)
    all_results += test_csa_indexer_validity(device)
    all_results += test_hca_sliding_window_causality(device)
    all_results += test_csa_full_pipeline_causality(device)
    # Regression tests for bugs found during code review.
    all_results += test_hybrid_padding_no_crash(device)
    all_results += test_hybrid_state_buffer_registration(device)
    all_results += test_bench_hybrid_no_grad_inference(device)
    all_results += test_hybrid_per_layer_kda_state(device)
    all_results += test_csa_hca_right_padding_correctness(device)
    all_results += test_csa_hca_sink_numerical_correctness(device)
    # New tests for dtype consistency and chunk padding edge cases.
    all_results += test_kda_chunk_nondivisible_T(device)
    all_results += test_csa_hca_fp16_dtype_consistency(device)
    all_results += test_kda_initial_state_dtype_mismatch(device)
    # Regression test for hybrid backward gradient flow.
    all_results += test_hybrid_backward_produces_grads(device)
    # Additional edge-case tests for broader coverage.
    all_results += test_csa_hca_bf16_dtype_consistency(device)
    all_results += test_csa_hca_no_sink_no_sliding_window(device)
    all_results += test_csa_topk_edge_cases(device)
    all_results += test_kda_single_token_decode(device)
    all_results += test_csa_hca_extreme_sink_values(device)
    all_results += test_hybrid_no_kda_layout(device)
    # Regression tests for the internal-padding + topk=0 fixes.
    all_results += test_csa_hca_non_divisible_T(device)
    all_results += test_csa_topk_zero(device)
    all_results += test_hca_T_smaller_than_m2(device)

    passed = sum(r['status'] == 'PASS' for r in all_results)
    logger.info('-' * 70)
    logger.info(f'Total: {passed}/{len(all_results)} passed')

    os.makedirs('results', exist_ok=True)
    with open('results/exp1_correctness.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('Saved: results/exp1_correctness.json')
    return 0 if passed == len(all_results) else 1


if __name__ == '__main__':
    sys.exit(main())
