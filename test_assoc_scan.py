"""Regression tests for the associative-scan chunkwise-KDA path.

``assoc_scan_chunk_kda`` resolves the per-chunk KDA state recurrence with a
Hillis-Steele affine prefix scan (``ceil(log2 NT)`` batched levels) instead
of the ``NT``-step sequential inner loop used by ``naive_chunk_kda``. The
per-chunk maps are built from the same triangular-solve outputs as the
sequential path::

    M_n = Diag(exp(g_last)) - kexp_n^T w_n ,   b_n = kexp_n^T u_n
    C_n = q * exp(g)        - Aqk_n w_n    ,   a_n = Aqk_n u_n

with ``kexp = k * exp(g_last - g)`` carrying the PER-CHANNEL log-decay.
Composition of these affine maps is associative, so all incoming chunk
states are recovered in log depth. The scan is an independent numerical
path (different summation order), not a bit-exact rewrite — these tests
lock its agreement with the sequential chunk path to fp round-off, its
logarithmic depth contract, its gradients, TorchScript/torch.compile
tracing, and the end-to-end adapter/config wiring.

How to run::

    pytest -q test_assoc_scan.py          # from the repo root

Standalone by design (own ``device`` fixture, no conftest dependency);
``pyproject.toml``'s ``testpaths = ["."]`` makes a bare ``pytest`` collect
it.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ops_kda import (  # noqa: E402
    _assoc_scan_chunk_states,
    assoc_scan_chunk_kda,
    naive_chunk_kda,
    naive_recurrent_kda,
)
from ops_kda_backend import kda_forward  # noqa: E402
from ops_fused import HybridConfig, HybridKCHAttention  # noqa: E402


BIT_MATCH = 1e-6  # repository TOL['bit_match'][torch.float32]


@pytest.fixture
def device(request):
    """Standalone ``--device`` override (falls back to the pytest option)."""
    try:
        return request.config.getoption('--device')
    except (ValueError, KeyError):
        return 'cpu'


def _make_case(seed, B, T, H, K, V, G=1, strong_decay=False, device='cpu',
               dtype=torch.float32):
    torch.manual_seed(seed)
    HV = H * G
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).to(dtype)
    v = (torch.randn(B, T, HV, V, device=device) * 0.1).to(dtype)
    g = (-torch.rand(B, T, HV, K, device=device) * 0.05).to(dtype)
    if strong_decay:
        # >= 50 cumulative decay inside a chunk so the exp(clamp(max=50))
        # branch (``clamp_g_diff=True``) runs in the shared prepare step.
        g[:, 5:] = -5.0
    beta = (torch.rand(B, T, HV, device=device) * 0.2).to(dtype)
    return q, k, v, g, beta


CASES = [
    # (seed, B, T, H, K, V, G, strong_decay, chunk_size)
    (71, 2, 130, 2, 16, 16, 1, False, 32),   # non-divisible T
    (72, 1, 96, 2, 16, 16, 1, False, 64),    # divisible T
    (73, 1, 130, 2, 16, 16, 2, False, 32),   # GVA G=2
    (74, 1, 48, 1, 8, 8, 1, True, 16),       # clamp_g_diff branch
    (75, 1, 7, 1, 8, 8, 1, False, 16),       # T < chunk_size
    (76, 1, 64, 1, 8, 8, 1, False, 64),      # NT == 1
]


def test_assoc_scan_vs_chunk_and_recurrent(device):
    """The scan matches the sequential chunk path to fp round-off."""
    worst_scan_chunk = 0.0
    worst_chunk_rec = 0.0
    for seed, B, T, H, K, V, G, strong, BT in CASES:
        q, k, v, g, beta = _make_case(seed, B, T, H, K, V, G, strong, device)
        o_scan, s_scan = assoc_scan_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        o_chunk, s_chunk = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        o_rec, s_rec = naive_recurrent_kda(
            q, k, v, g, beta, output_final_state=True)
        assert o_scan.shape == o_chunk.shape == o_rec.shape
        assert s_scan.shape == s_chunk.shape == s_rec.shape
        worst_scan_chunk = max(
            worst_scan_chunk,
            (o_scan - o_chunk).abs().max().item(),
            (s_scan - s_chunk).abs().max().item())
        worst_chunk_rec = max(
            worst_chunk_rec,
            (o_chunk - o_rec).abs().max().item(),
            (s_chunk - s_rec).abs().max().item())
    assert worst_scan_chunk < BIT_MATCH, f'max|scan-chunk|={worst_scan_chunk:.2e}'
    assert worst_chunk_rec < BIT_MATCH, f'max|chunk-rec|={worst_chunk_rec:.2e}'


def test_assoc_scan_anisotropic_per_channel_decay(device):
    """Per-channel decay: half the key channels decay ~1000x faster."""
    torch.manual_seed(77)
    B, T, H, K, V = 1, 192, 2, 16, 16
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    g = torch.empty(B, T, H, K, device=device)
    g[..., :K // 2] = -2.0 - 3.0 * torch.rand(B, T, H, K // 2, device=device)
    g[..., K // 2:] = -0.001 * torch.rand(B, T, H, K // 2, device=device)
    beta = torch.rand(B, T, H, device=device) * 0.2
    o_scan, s_scan = assoc_scan_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=32)
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=32)
    assert (o_scan - o_chunk).abs().max().item() < BIT_MATCH
    assert (s_scan - s_chunk).abs().max().item() < BIT_MATCH


def test_assoc_scan_fp64_tight(device):
    """In fp64 the scan agrees with the sequential chunk path far below fp32."""
    q, k, v, g, beta = _make_case(
        78, 1, 192, 2, 16, 16, G=1, device=device, dtype=torch.float64)
    o_scan, s_scan = assoc_scan_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=32)
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=32)
    assert o_scan.dtype == torch.float64
    assert (o_scan - o_chunk).abs().max().item() < 1e-12
    assert (s_scan - s_chunk).abs().max().item() < 1e-12


def _reference_affine_scan(M, b, S0):
    """Textbook sequential reference for the affine prefix scan."""
    NT = M.shape[2]
    S = S0
    S_ins = []
    for n in range(NT):
        S_ins.append(S)
        S = M[:, :, n] @ S + b[:, :, n]
    return torch.stack(S_ins, dim=2), S


def test_assoc_scan_states_closed_form_and_depth(device):
    """The scan mechanics match a hand-rolled loop and run in log depth.

    This is independent of KDA: random affine maps, a plain Python prefix
    loop, and a white-box assertion that the Hillis-Steele core performs
    exactly ``ceil(log2 NT)`` doubling levels and passes the initial state
    through bit-exactly for chunk 0.
    """
    torch.manual_seed(79)
    for NT in (1, 2, 3, 5, 16):
        for B, HV in ((2, 2), (2, 3)):
            K, V = 4, 3
            M = torch.randn(B, HV, NT, K, K, device=device) * 0.2
            b = torch.randn(B, HV, NT, K, V, device=device) * 0.2
            S0 = torch.randn(B, HV, K, V, device=device) * 0.2
            S_in, S_final, levels = _assoc_scan_chunk_states(M, b, S0)
            ref_in, ref_final = _reference_affine_scan(M, b, S0)
            expected_levels = 0 if NT == 1 else math.ceil(math.log2(NT))
            assert levels == expected_levels, (NT, levels)
            assert torch.equal(S_in[:, :, 0], S0), 'chunk 0 must see S0 exactly'
            assert (S_in - ref_in).abs().max().item() < BIT_MATCH
            assert (S_final - ref_final).abs().max().item() < BIT_MATCH


def test_assoc_scan_scripted_core_parity(device):
    """The scan core is TorchScript-compilable and bit-identical to eager."""
    M = torch.randn(1, 2, 12, 4, 4, device=device) * 0.1
    b = torch.randn(1, 2, 12, 4, 3, device=device) * 0.1
    S0 = torch.randn(1, 2, 4, 3, device=device) * 0.1
    eager = _assoc_scan_chunk_states(M, b, S0)
    scripted = torch.jit.script(_assoc_scan_chunk_states)
    out = scripted(M, b, S0)
    assert torch.equal(out[0], eager[0])
    assert torch.equal(out[1], eager[1])
    assert out[2] == eager[2] == 4


def test_assoc_scan_compiled_core_smoke(device):
    """Fast ``torch.compile`` (fullgraph) parity for the scan core.

    Mirrors the fast compile-contract smoke test used for the recurrent
    path: guards against a dynamo regression in the scan (the tracing
    contract, not speed) without paying for a full-model compile.
    """
    M = torch.randn(1, 2, 8, 4, 4, device=device) * 0.1
    b = torch.randn(1, 2, 8, 4, 3, device=device) * 0.1
    S0 = torch.randn(1, 2, 4, 3, device=device) * 0.1
    eager = _assoc_scan_chunk_states(M, b, S0)
    compiled = torch.compile(
        _assoc_scan_chunk_states, dynamic=False, fullgraph=True)
    out = compiled(M, b, S0)
    assert torch.equal(out[0], eager[0])
    assert torch.equal(out[1], eager[1])
    assert out[2] == eager[2] == 3


def test_assoc_scan_gradient_parity(device):
    """Backward through the scan graph matches the sequential chunk graph."""
    q, k, v, g, beta = _make_case(80, 1, 128, 1, 8, 8, device=device)
    grads = {}
    for tag, fn in [
        ('chunk', lambda a, b_, c, d, e: naive_chunk_kda(
            a, b_, c, d, e, output_final_state=True, chunk_size=32)),
        ('scan', lambda a, b_, c, d, e: assoc_scan_chunk_kda(
            a, b_, c, d, e, output_final_state=True, chunk_size=32)),
    ]:
        tensors = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        o, s = fn(*tensors)
        (o.sum() + s.sum()).backward()
        grads[tag] = [t.grad for t in tensors]
        assert all(gr is not None for gr in grads[tag])
    worst = max((a - b_).abs().max().item()
                for a, b_ in zip(grads['chunk'], grads['scan']))
    assert worst < 1e-5, f'max|grad diff|={worst:.2e}'


def test_assoc_scan_initial_state_and_empty(device):
    """Carried initial state, T=0, output_final_state, state_dtype edges.

    Uses B=2 with GVA G=2 and a NON-zero initial state on purpose: with
    ``S0 == 0`` (the common no-initial-state case) a wrong batch/head
    broadcast in the final-state matmul is invisible, so this test would
    silently pass on a broken scan without the nonzero state.
    """
    q, k, v, g, beta = _make_case(81, 2, 64, 2, 8, 8, G=2, device=device)
    torch.manual_seed(99)
    state = torch.randn(2, 4, 8, 8, device=device) * 0.1
    o_scan, s_scan = assoc_scan_chunk_kda(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        chunk_size=16)
    o_chunk, s_chunk = naive_chunk_kda(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        chunk_size=16)
    assert (o_scan - o_chunk).abs().max().item() < BIT_MATCH
    assert (s_scan - s_chunk).abs().max().item() < BIT_MATCH
    # T == 0 short-circuits with the same contract as the chunk path.
    q0 = torch.randn(1, 0, 1, 4, device=device)
    k0 = torch.randn(1, 0, 1, 4, device=device)
    v0 = torch.randn(1, 0, 1, 4, device=device)
    g0 = torch.randn(1, 0, 1, 4, device=device)
    b0 = torch.randn(1, 0, 1, device=device)
    o, s = assoc_scan_chunk_kda(
        q0, k0, v0, g0, b0, output_final_state=True, chunk_size=16)
    assert o.shape == (1, 0, 1, 4) and s.shape == (1, 1, 4, 4)
    assert torch.equal(o, torch.zeros_like(o))
    # output_final_state=False returns None state, dtype cast honored.
    o2, s2 = assoc_scan_chunk_kda(
        q, k, v, g, beta, output_final_state=False, chunk_size=16)
    assert s2 is None and o2.shape == o_scan.shape
    _, s3 = assoc_scan_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        state_dtype=torch.float64)
    assert s3.dtype == torch.float64


def test_chunk_padding_is_identity_with_positive_gate_floor(device):
    """Padded steps must not clamp their synthetic g=0 up to a positive floor.

    ``T % chunk_size != 0`` pads ``g`` with zeros, which are identity decay
    steps. Applying a positive ``g_clamp_min`` after padding turns them into
    ``exp(g_clamp_min) > 1`` decays and corrupts the returned final state.
    """
    B, T, H, K, V = 1, 5, 1, 2, 2
    q = torch.ones(B, T, H, K, device=device)
    k = torch.full((B, T, H, K), K ** -0.5, device=device)
    v = torch.full((B, T, H, V), 0.1, device=device)
    g = torch.ones(B, T, H, K, device=device)
    beta = torch.full((B, T, H), 0.1, device=device)
    state = torch.ones(B, H, K, V, device=device)
    _, s_rec = naive_recurrent_kda(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        g_clamp_min=1.0)
    for fn in (naive_chunk_kda, assoc_scan_chunk_kda):
        _, s_chunk = fn(
            q, k, v, g, beta, initial_state=state, output_final_state=True,
            chunk_size=4, g_clamp_min=1.0)
        torch.testing.assert_close(
            s_chunk, s_rec, rtol=1e-5, atol=1e-5,
            msg=lambda m: f'{fn.__name__}: {m}')


def test_kda_forward_assoc_scan_adapter(device):
    """``kda_forward(assoc_scan=...)`` routes to the scan on the chunk path."""
    q, k, v, g, beta = _make_case(82, 1, 64, 1, 8, 8, device=device)
    o_seq, s_seq = kda_forward(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference')
    o_scan, s_scan = kda_forward(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference', assoc_scan=True)
    assert (o_seq - o_scan).abs().max().item() < BIT_MATCH
    assert (s_seq - s_scan).abs().max().item() < BIT_MATCH
    # Mutually exclusive strategies must fail loudly, not silently pick one.
    with pytest.raises(ValueError):
        kda_forward(
            q, k, v, g, beta, output_final_state=True, chunk_size=16,
            use_chunk=True, backend='reference',
            assoc_scan=True, recompute_attn=True)
    # The knob is inert when the recurrent path runs (short T).
    o_r, _ = kda_forward(
        q, k, v, g, beta, output_final_state=True, use_chunk=False,
        backend='reference', assoc_scan=True)
    assert o_r.shape == o_seq.shape


def test_hybrid_config_assoc_scan_parity(device):
    """``HybridConfig.kda_assoc_scan`` yields the same hybrid forward."""
    def make_cfg(assoc):
        return HybridConfig(
            d_model=32, n_heads_qk=2, n_heads_v=2,
            head_dim_k=8, head_dim_v=8,
            kda_chunk_size=16, kda_assoc_scan=assoc,
            csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=4,
            csa_nIh=1, csa_cI=4, csa_sliding_window=4,
            hca_m2=8, hca_nh=2, hca_c=8, hca_dc=4, hca_sliding_window=4,
        )

    torch.manual_seed(83)
    model_seq = HybridKCHAttention(make_cfg(False), total_layers=5).to(device).eval()
    model_scan = HybridKCHAttention(make_cfg(True), total_layers=5).to(device).eval()
    model_scan.load_state_dict(model_seq.state_dict())
    x = torch.randn(2, 48, 32, device=device) * 0.1
    with torch.no_grad():
        y_seq = model_seq(x)
        y_scan = model_scan(x)
    diff = (y_seq - y_scan).abs().max().item()
    assert diff < BIT_MATCH, f'max|diff|={diff:.2e}'
    # Type / exclusivity guards fire at config construction.
    with pytest.raises(TypeError):
        make_cfg(assoc=1)
    with pytest.raises(ValueError):
        HybridConfig(
            d_model=32, n_heads_qk=2, n_heads_v=2,
            head_dim_k=8, head_dim_v=8,
            kda_assoc_scan=True, kda_recompute_attn=True)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
