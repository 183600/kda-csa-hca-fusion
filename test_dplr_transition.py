"""Regression tests for the explicit DPLR transition algebra.

The KDA recurrence advances the state through a diagonal-plus-low-rank
map ::

    S_t = [Diag(exp(g_t)) - beta_t k_t (exp(g_t) * k_t)^T] S_{t-1}
          + beta_t k_t v_t^T

and the chunked path builds the same map implicitly through the
delta-rule/wY triangular solve.  These tests pin the algebra, the two
constructions against each other, the RWKV-7 specialisation (same DPLR
class, rank 1), and the blocked two-phase scan that exploits the chunk
map's structure.

Standalone by design (own ``device`` fixture, no conftest dependency).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dplr_transition as D  # noqa: E402
from ops_kda import (  # noqa: E402
    assoc_scan_chunk_kda,
    naive_chunk_kda,
    naive_recurrent_kda,
)
from ops_kda_backend import kda_forward  # noqa: E402


@pytest.fixture
def device(request):
    try:
        return request.config.getoption('--device')
    except (ValueError, KeyError):
        return 'cpu'


def _kda_case(seed, B, T, H, K, V, *, dtype=torch.float64, decay=0.05,
              strong=False):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, dtype=dtype) * decay
    if strong:
        g[:, 5:] = -1.5
    beta = torch.rand(B, T, H, dtype=dtype) * 0.2
    return q, k, v, g, beta


def _dense_from_steps(k, g, beta):
    """Reference chunk map for ``k/g [B, C, H, K]`` and ``beta [B, C, H]``.

    Explicit product of per-step dense matrices, with the step applied on
    the left (``M = M_C ... M_1``).
    """
    C, K = k.shape[1], k.shape[-1]
    M = torch.eye(K, dtype=k.dtype, device=k.device).expand(
        k.shape[0], k.shape[2], K, K).clone()
    for t in range(C):
        alpha = g[:, t].exp()                        # [B, H, K]
        step = torch.diag_embed(alpha) - beta[:, t].unsqueeze(-1).unsqueeze(
            -1) * (k[:, t].unsqueeze(-1) *
                   (alpha * k[:, t]).unsqueeze(-2))
        M = step @ M
    return M


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------
def test_compose_matches_dense_and_rank_adds(device):
    torch.manual_seed(0)
    b, h, k = 2, 3, 8
    d1 = torch.rand(b, h, k, dtype=torch.float64) * 0.9 + 0.1
    d2 = torch.rand(b, h, k, dtype=torch.float64) * 0.9 + 0.1
    l1 = torch.randn(b, h, k, 3, dtype=torch.float64) * 0.1
    r1 = torch.randn(b, h, k, 3, dtype=torch.float64) * 0.1
    l2 = torch.randn(b, h, k, 2, dtype=torch.float64) * 0.1
    r2 = torch.randn(b, h, k, 2, dtype=torch.float64) * 0.1
    d, l, r = D.compose(d1, l1, r1, d2, l2, r2)
    assert l.shape[-1] == r.shape[-1] == 5
    torch.testing.assert_close(
        D.dense(d, l, r), D.dense(d2, l2, r2) @ D.dense(d1, l1, r1),
        rtol=1e-12, atol=1e-12)

    # Non-commutative order matters: M1 M2 != M2 M1 in general.
    d_rev, l_rev, r_rev = D.compose(d2, l2, r2, d1, l1, r1)
    assert (D.dense(d, l, r) - D.dense(d_rev, l_rev, r_rev)).abs().max() > 1e-6

    # Associativity of three compositions.
    d3 = torch.rand(b, h, k, dtype=torch.float64) * 0.9 + 0.1
    l3 = torch.randn(b, h, k, 1, dtype=torch.float64) * 0.1
    r3 = torch.randn(b, h, k, 1, dtype=torch.float64) * 0.1
    left = D.compose(*D.compose(d1, l1, r1, d2, l2, r2), d3, l3, r3)
    right = D.compose(d1, l1, r1, *D.compose(d2, l2, r2, d3, l3, r3))
    torch.testing.assert_close(D.dense(*left), D.dense(*right),
                               rtol=1e-12, atol=1e-12)


def test_compose_compresses_without_changing_the_map(device):
    torch.manual_seed(1)
    b, h, k, rank = 2, 2, 5, 9
    d1 = torch.rand(b, h, k, dtype=torch.float64) * 0.5 + 0.5
    d2 = torch.rand(b, h, k, dtype=torch.float64) * 0.5 + 0.5
    l1 = torch.randn(b, h, k, rank, dtype=torch.float64) * 0.1
    r1 = torch.randn(b, h, k, rank, dtype=torch.float64) * 0.1
    l2 = torch.randn(b, h, k, rank, dtype=torch.float64) * 0.1
    r2 = torch.randn(b, h, k, rank, dtype=torch.float64) * 0.1
    d, l, r = D.compose(d1, l1, r1, d2, l2, r2)
    assert l.shape[-1] == k, 'rank must be compressed back to K exactly'
    assert D.effective_rank(l, r) == k
    torch.testing.assert_close(
        D.dense(d, l, r), D.dense(d2, l2, r2) @ D.dense(d1, l1, r1),
        rtol=1e-11, atol=1e-11)


def test_step_factors_match_the_documented_recurrences(device):
    torch.manual_seed(2)
    b, h, k, v = 2, 2, 6, 4
    kk = torch.randn(b, h, k, dtype=torch.float64) * 0.2
    g = -torch.rand(b, h, k, dtype=torch.float64) * 0.3
    beta = torch.rand(b, h, dtype=torch.float64) * 0.4
    s = torch.randn(b, h, k, v, dtype=torch.float64) * 0.1
    val = torch.randn(b, h, v, dtype=torch.float64) * 0.1

    d, l, r = D.from_kda_step(kk, g, beta)
    alpha = g.exp()
    m = D.dense(d, l, r)
    ref = torch.diag_embed(alpha) - beta.unsqueeze(-1).unsqueeze(-1) * (
        kk.unsqueeze(-1) * (alpha * kk).unsqueeze(-2))
    torch.testing.assert_close(m, ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        m @ s + beta.unsqueeze(-1).unsqueeze(-1) *
        kk.unsqueeze(-1) * val.unsqueeze(-2),
        alpha.unsqueeze(-1) * s - beta.unsqueeze(-1).unsqueeze(-1) *
        kk.unsqueeze(-1) *
        (alpha * kk).unsqueeze(-2) @ s +
        beta.unsqueeze(-1).unsqueeze(-1) * kk.unsqueeze(-1) *
        val.unsqueeze(-2),
        rtol=1e-12, atol=1e-12)

    w = torch.rand(b, h, k, dtype=torch.float64) * 0.9
    kka = torch.randn(b, h, k, dtype=torch.float64) * 0.1
    d, l, r = D.from_rwkv7_step(w, kk, kka)
    ref_update = w.unsqueeze(-1) * s + kka.unsqueeze(-1) * (
        kk.unsqueeze(-1) * s).sum(-2, keepdim=True)
    torch.testing.assert_close(D.dense(d, l, r) @ s, ref_update,
                               rtol=1e-12, atol=1e-12)


def test_chunk_factors_match_the_product_of_steps(device):
    b, t, h, k, v = 2, 37, 2, 8, 3
    q, kk, vv, g, beta = _kda_case(3, b, t, h, k, v)
    for chunk_size in (4, 16, 64):
        dc, lc, rc = D.kda_chunk_factors(kk, g, beta, chunk_size)
        nt = dc.shape[2]
        # The dense construction uses the padded token axis.
        pad = (-t) % chunk_size
        kpad = F.pad(kk, (0, 0, 0, 0, 0, pad))
        gpad = F.pad(g, (0, 0, 0, 0, 0, pad))
        bpad = F.pad(beta, (0, 0, 0, pad))
        for n in range(nt):
            sl = slice(n * chunk_size, (n + 1) * chunk_size)
            ref = _dense_from_steps(
                kpad[:, sl], gpad[:, sl], bpad[:, sl])
            got = D.dense(dc[:, :, n], lc[:, :, n], rc[:, :, n])
            torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


def test_rwkv7_chunk_factors_place_the_chunk_axis_before_the_key_axis(device):
    """The documented layout is ``[..., NT, K, r]``, not ``[..., K, NT, r]``.

    ``diag`` has one axis fewer than ``left``/``right``, so the chunk axis
    must be inserted at a different relative position; an off-by-one here
    makes every consumer's ``[..., n, :]`` slice select a key row instead
    of a chunk.
    """
    torch.manual_seed(25)
    b, h, t, k, chunk = 2, 3, 10, 5, 4
    w = torch.rand(b, h, t, k, dtype=torch.float64) * 0.9 + 0.1
    kk = torch.randn(b, h, t, k, dtype=torch.float64) * 0.1
    kka = torch.randn(b, h, t, k, dtype=torch.float64) * 0.1

    dc, lc, rc = D.rwkv7_chunk_factors(w, kk, kka, chunk)
    nt = math.ceil(t / chunk)
    assert dc.shape == (b, h, nt, k)
    assert lc.shape[:3] == rc.shape[:3] == (b, h, nt)
    assert lc.shape[-2:] == (k, lc.shape[-1])
    assert lc.shape[-1] == rc.shape[-1] <= min(chunk, k)

    pad = (-t) % chunk
    wp = F.pad(w, (0, 0, 0, pad), value=1.0)
    kkp = F.pad(kk, (0, 0, 0, pad), value=0.0)
    kkap = F.pad(kka, (0, 0, 0, pad), value=0.0)
    for n in range(nt):
        ref = torch.eye(k, dtype=w.dtype).expand(b, h, k, k).clone()
        for c in range(chunk):
            idx = n * chunk + c
            step = (torch.diag_embed(wp[:, :, idx])
                    + kkap[:, :, idx].unsqueeze(-1)
                    * kkp[:, :, idx].unsqueeze(-2))
            ref = step @ ref
        got = D.dense(dc[:, :, n], lc[:, :, n], rc[:, :, n])
        torch.testing.assert_close(got, ref, rtol=1e-11, atol=1e-11)

    # The same contract for the reduced 2-D form used by RWKV helpers.
    d2, l2, r2 = D.rwkv7_chunk_factors(w[0, 0], kk[0, 0], kka[0, 0], chunk)
    assert d2.shape == (nt, k)
    assert l2.shape[:2] == r2.shape[:2] == (nt, k)
    torch.testing.assert_close(
        D.dense(d2[0], l2[0], r2[0]),
        D.dense(dc[0, 0, 0], lc[0, 0, 0], rc[0, 0, 0]),
        rtol=1e-12, atol=1e-12)


def test_certificate_matches_the_wy_construction_over_a_regression_matrix(
        device):
    cases = [
        # (seed, T, H, K, V, chunk_size, decay, strong)
        (10, 64, 2, 16, 5, 16, 0.05, False),
        (11, 130, 1, 16, 3, 32, 0.05, False),
        (12, 64, 2, 8, 4, 64, 0.10, False),
        (13, 192, 1, 16, 3, 16, 0.05, True),
        (14, 7, 1, 8, 3, 16, 0.05, False),
    ]
    worst = 0.0
    for seed, t, h, k, v, chunk, decay, strong in cases:
        q, kk, vv, g, beta = _kda_case(
            seed, 2, t, h, k, v, decay=decay, strong=strong)
        cert = D.kda_dplr_certificate(kk, g, beta, vv, chunk)
        assert cert['n_chunks'] == math.ceil(t / chunk)
        assert cert['rank_max'] <= cert['rank_bound']
        assert cert['rank_max'] >= 1, cert
        assert cert['passed'], cert
        worst = max(worst, cert['max_abs_map_error'],
                    cert['max_abs_state_error'])
    assert worst < 1e-9, f'worst certificate error {worst:.2e}'


# ---------------------------------------------------------------------------
# Blocked two-phase scan (the work-efficient scan)
# ---------------------------------------------------------------------------
def test_blocked_scan_matches_the_full_scan_and_recurrent(device):
    torch.manual_seed(20)
    b, t, h, k, v = 1, 130, 2, 16, 8
    q, kk, vv, g, beta = _kda_case(21, b, t, h, k, v)
    o_full, s_full = assoc_scan_chunk_kda(
        q, kk, vv, g, beta, output_final_state=True, chunk_size=32)
    o_rec, s_rec = naive_recurrent_kda(
        q, kk, vv, g, beta, output_final_state=True)
    for block_size in (2, 4, 7, 64):
        o_b, s_b = assoc_scan_chunk_kda(
            q, kk, vv, g, beta, output_final_state=True, chunk_size=32,
            scan_block_size=block_size)
        torch.testing.assert_close(o_b, o_full, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(s_b, s_full, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(o_b, o_rec, rtol=1e-10, atol=1e-10)


def test_blocked_scan_plan_counts(device):
    from ops_kda import _blocked_chunk_states
    torch.manual_seed(22)
    b, hv, nt, bt, k, v = 1, 2, 11, 4, 8, 3
    kexp = torch.randn(b, hv, nt, bt, k, dtype=torch.float64) * 0.1
    w = torch.randn(b, hv, nt, bt, k, dtype=torch.float64) * 0.1
    u = torch.randn(b, hv, nt, bt, v, dtype=torch.float64) * 0.1
    glast = torch.rand(b, hv, nt, k, dtype=torch.float64) * 0.9
    s0 = torch.randn(b, hv, k, v, dtype=torch.float64) * 0.1
    _, _, meta = _blocked_chunk_states(kexp, w, u, glast, s0, 4)
    assert meta['n_blocks'] == math.ceil(nt / 4)
    assert meta['within_block_compositions'] == 8  # 4 + 4 (last block has 3)
    assert meta['total_compositions'] == (
        meta['within_block_compositions'] + meta['block_scan_compositions'])
    assert meta['hillis_steele_compositions'] > meta['total_compositions']
    assert meta['composition_savings_factor'] > 1.0


def test_blocked_scan_gradient_parity_and_backend_passthrough(device):
    q, kk, vv, g, beta = _kda_case(23, 1, 96, 1, 8, 8)

    def run(block):
        tensors = [x.clone().requires_grad_(True) for x in (q, kk, vv, g, beta)]
        o, s = assoc_scan_chunk_kda(
            *tensors, output_final_state=True, chunk_size=16,
            scan_block_size=block)
        (o.sum() + s.sum()).backward()
        return [t.grad for t in tensors]

    full = run(0)
    blocked = run(4)
    worst = max((a - b).abs().max().item() for a, b in zip(full, blocked))
    assert worst < 1e-9, f'gradient parity {worst:.2e}'

    o0, s0 = kda_forward(
        q, kk, vv, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference', assoc_scan=True)
    o1, s1 = kda_forward(
        q, kk, vv, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference', assoc_scan=True,
        scan_block_size=4)
    torch.testing.assert_close(o1, o0, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(s1, s0, rtol=1e-11, atol=1e-11)
    with pytest.raises(ValueError):
        kda_forward(
            q, kk, vv, g, beta, output_final_state=True, chunk_size=16,
            use_chunk=True, backend='reference', assoc_scan=True,
            scan_block_size=-1)


def test_certificate_rejects_malformed_inputs(device):
    q, k, v, g, beta = _kda_case(24, 1, 16, 1, 4, 2)
    with pytest.raises(ValueError):
        D.kda_dplr_certificate(k, g, beta, v, 0)
    with pytest.raises(ValueError):
        D.kda_dplr_certificate(k, g, beta[:, :, 0], v, 4)
    with pytest.raises(ValueError):
        D.compose(g, k.unsqueeze(-1), k[..., :3].unsqueeze(-1),
                  g, k.unsqueeze(-1), k.unsqueeze(-1))  # K mismatch


def test_two_phase_scan_work_plan(device):
    """The work plan is internally consistent and exposes the trade-off."""
    plan = D.scan_work_plan(128, 8, dim=128, value_dim=64, chunk_size=64)
    assert plan['n_blocks'] == 16
    assert plan['producer_compositions'] == 112  # NT - NB
    assert plan['scan_compositions'] == (
        plan['producer_compositions'] + plan['scanner_compositions'])
    assert plan['scan_compositions'] < plan['full_scan_compositions']
    assert plan['composition_savings_factor'] > 2.0
    assert plan['transition_memory_ratio'] == pytest.approx(16 / 128)
    assert plan['blocked_transition_scalars'] * 8 == (
        plan['dense_transition_scalars'])

    # Short chunks: producer work fits inside the consumer chain.
    short = D.scan_work_plan(128, 8, dim=128, value_dim=64, chunk_size=16)
    assert short['producer_hidden_by_consumer'] is True
    # Long chunks: the producer is the bottleneck.
    long = D.scan_work_plan(128, 8, dim=128, value_dim=64, chunk_size=64)
    assert long['producer_hidden_by_consumer'] is False
    assert plan['specialized_transition_macs'] < plan['serial_transition_macs']
    with pytest.raises(ValueError):
        D.scan_work_plan(0, 8)
    with pytest.raises(ValueError):
        D.scan_work_plan(16, 0)


def test_blocked_scan_gva_and_initial_state(device):
    """GVA (HV > H) and a nonzero carried state select the right slices."""
    torch.manual_seed(30)
    b, t, h, hv, k, v = 2, 96, 2, 4, 8, 5
    q = F.normalize(torch.randn(b, t, h, k, dtype=torch.float64), dim=-1)
    kk = F.normalize(torch.randn(b, t, h, k, dtype=torch.float64), dim=-1)
    vv = torch.randn(b, t, hv, v, dtype=torch.float64) * 0.1
    g = -torch.rand(b, t, hv, k, dtype=torch.float64) * 0.05
    beta = torch.rand(b, t, hv, dtype=torch.float64) * 0.2
    s0 = torch.randn(b, hv, k, v, dtype=torch.float64) * 0.1
    o_full, s_full = assoc_scan_chunk_kda(
        q, kk, vv, g, beta, initial_state=s0, output_final_state=True,
        chunk_size=16)
    o_blk, s_blk = assoc_scan_chunk_kda(
        q, kk, vv, g, beta, initial_state=s0, output_final_state=True,
        chunk_size=16, scan_block_size=4)
    torch.testing.assert_close(o_blk, o_full, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(s_blk, s_full, rtol=1e-11, atol=1e-11)
