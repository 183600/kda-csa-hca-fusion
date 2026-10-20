"""Regression tests for the chunkwise-KDA recompute-vs-cache knob.

``naive_chunk_kda(..., recompute_attn=True)`` (and
``scripted_chunk_kda(..., recompute_attn=True)``) switches the reference
chunk path from the default *cache* policy — where every chunk's
q-weighted intra-chunk attention matrix ``Aqk`` is materialized up-front
in ``_chunk_kda_prepare`` with shape ``[B, HV, NT, BT, BT]``, i.e.
``O(T * chunk_size)`` extra elements per head — to a *recompute* policy
that rebuilds ``Aqk`` for the current chunk only inside the sequential
inner loop (``O(chunk_size ** 2)`` extra elements per head). The feature
is exposed end-to-end via ``ops_kda_backend.kda_forward(recompute_attn=)``
and ``ops_fused.HybridConfig.kda_recompute_attn``.

These tests lock the equivalence between the two policies (forward
values, gradients, TorchScript path, hybrid integration) and add a fast
``torch.compile`` parity check.

How to run::

    pytest -q test_recompute_attn.py          # from the repo root

The file is intentionally standalone (own ``device`` fixture, no
conftest dependency); ``pyproject.toml`` sets ``testpaths = ["."]`` so any
bare ``pytest`` run collects it.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ops_kda import (  # noqa: E402
    _chunk_kda_attn_loop_eager,
    _chunk_kda_prepare,
    compiled_chunk_kda,
    compiled_recurrent_kda,
    naive_chunk_kda,
    naive_recurrent_kda,
    scripted_chunk_kda,
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


def _make_case(seed, B, T, H, K, V, G=1, strong_decay=False, device='cpu'):
    torch.manual_seed(seed)
    HV = H * G
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, HV, V, device=device) * 0.1
    g = -torch.rand(B, T, HV, K, device=device) * 0.05
    if strong_decay:
        # >= 50 cumulative decay inside a chunk so the exp(clamp(max=50))
        # branch (``clamp_g_diff=True``) runs in BOTH policies.
        g[:, 5:] = -5.0
    beta = torch.rand(B, T, HV, device=device) * 0.2
    return q, k, v, g, beta


CASES = [
    # (seed, B, T, H, K, V, G, strong_decay, chunk_size)
    (11, 2, 130, 2, 16, 16, 1, False, 32),   # non-divisible T
    (12, 1, 96, 2, 16, 16, 1, False, 64),    # divisible T
    (13, 1, 130, 2, 16, 16, 2, False, 32),   # GVA G=2
    (14, 1, 48, 1, 8, 8, 1, True, 16),       # clamp branch
]


def test_recompute_vs_cache_forward_parity(device):
    """recompute_attn=True == recompute_attn=False (and == recurrent)."""
    worst_cache_re = 0.0
    worst_re_rec = 0.0
    for seed, B, T, H, K, V, G, strong, BT in CASES:
        q, k, v, g, beta = _make_case(seed, B, T, H, K, V, G, strong, device)
        o_cache, s_cache = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        o_re, s_re = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT,
            recompute_attn=True)
        o_rec, s_rec = naive_recurrent_kda(
            q, k, v, g, beta, output_final_state=True)
        assert o_cache.shape == o_re.shape == o_rec.shape
        assert s_cache.shape == s_re.shape == s_rec.shape
        worst_cache_re = max(
            worst_cache_re,
            (o_cache - o_re).abs().max().item(),
            (s_cache - s_re).abs().max().item())
        worst_re_rec = max(
            worst_re_rec,
            (o_rec - o_re).abs().max().item(),
            (s_rec - s_re).abs().max().item())
        # The clamped-gate branch must be genuinely exercised by case 4.
        if strong:
            prep = _chunk_kda_prepare(
                q, k, v, g, beta, None, None, False, BT, -10.0, None,
                'test', recompute_attn=True)
            assert prep['clamp_g_diff'], 'strong-decay case must clamp'
    assert worst_cache_re < BIT_MATCH, f'max|diff|={worst_cache_re:.2e}'
    assert worst_re_rec < BIT_MATCH, f'max|diff|={worst_re_rec:.2e}'


def test_recompute_drops_aqk_cache(device):
    """White-box: the persistent ``Aqk`` cache is actually skipped."""
    q, k, v, g, beta = _make_case(21, 1, 32, 1, 8, 8, device=device)
    prep_re = _chunk_kda_prepare(
        q, k, v, g, beta, None, None, False, 16, -10.0, None, 'test',
        recompute_attn=True)
    prep_ca = _chunk_kda_prepare(
        q, k, v, g, beta, None, None, False, 16, -10.0, None, 'test',
        recompute_attn=False)
    assert prep_re['Aqk'] is None
    assert prep_ca['Aqk'] is not None
    # Cache size is O(B * HV * T * BT) elements; recompute holds one chunk.
    B, HV, T, BT = 1, 1, 32, 16
    assert prep_ca['Aqk'].numel() == B * HV * T * BT
    assert prep_re['clamp_g_diff'] == prep_ca['clamp_g_diff']
    assert prep_re['upper_mask'].shape == (BT, BT)


def test_recompute_scripted_parity(device):
    """The TorchScript inner loop supports the recompute policy."""
    for seed, B, T, H, K, V, G, strong, BT in CASES:
        q, k, v, g, beta = _make_case(seed, B, T, H, K, V, G, strong, device)
        o_eager, s_eager = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT,
            recompute_attn=True)
        o_scr, s_scr = scripted_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT,
            recompute_attn=True)
        assert (o_scr - o_eager).abs().max().item() < BIT_MATCH
        assert (s_scr - s_eager).abs().max().item() < BIT_MATCH
        # use_script=False must route straight to the eager recompute path.
        o_ns, s_ns = scripted_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=BT,
            recompute_attn=True, use_script=False)
        assert torch.equal(o_ns, o_eager) and torch.equal(s_ns, s_eager)


def test_recompute_gradient_parity(device):
    """Backward through the recompute graph matches the cache graph."""
    q, k, v, g, beta = _make_case(22, 1, 64, 1, 8, 8, device=device)
    grads = {}
    for mode, tag in [(False, 'cache'), (True, 'recompute')]:
        qc = q.clone().requires_grad_(True)
        kc = k.clone().requires_grad_(True)
        vc = v.clone().requires_grad_(True)
        gc = g.clone().requires_grad_(True)
        bc = beta.clone().requires_grad_(True)
        o, s = naive_chunk_kda(
            qc, kc, vc, gc, bc, output_final_state=True, chunk_size=16,
            recompute_attn=mode)
        (o.sum() + s.sum()).backward()
        grads[tag] = [t.grad for t in (qc, kc, vc, gc, bc)]
    assert all(gr is not None for gr in grads['recompute'])
    worst = max((a - b).abs().max().item()
                for a, b in zip(grads['cache'], grads['recompute']))
    assert worst < BIT_MATCH, f'max|grad diff|={worst:.2e}'


def test_recompute_initial_state_parity(device):
    """The recompute policy carries an incoming recurrent state identically."""
    q, k, v, g, beta = _make_case(23, 1, 64, 1, 8, 8, device=device)
    torch.manual_seed(99)
    state = torch.randn(1, 1, 8, 8, device=device) * 0.1
    o_cache, s_cache = naive_chunk_kda(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        chunk_size=16)
    o_re, s_re = naive_chunk_kda(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        chunk_size=16, recompute_attn=True)
    assert (o_cache - o_re).abs().max().item() < BIT_MATCH
    assert (s_cache - s_re).abs().max().item() < BIT_MATCH


def test_recompute_empty_sequence(device):
    """T == 0 short-circuits before the inner loop in both policies."""
    q = torch.randn(1, 0, 1, 4, device=device)
    k = torch.randn(1, 0, 1, 4, device=device)
    v = torch.randn(1, 0, 1, 4, device=device)
    g = torch.randn(1, 0, 1, 4, device=device)
    beta = torch.randn(1, 0, 1, device=device)
    o, s = naive_chunk_kda(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        recompute_attn=True)
    assert o.shape == (1, 0, 1, 4)
    assert s.shape == (1, 1, 4, 4)
    assert torch.equal(o, torch.zeros_like(o))


def test_recompute_kda_forward_adapter(device):
    """``kda_forward(recompute_attn=...)`` reaches the reference chunk path."""
    q, k, v, g, beta = _make_case(24, 1, 64, 1, 8, 8, device=device)
    o_cache, s_cache = kda_forward(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference')
    o_re, s_re = kda_forward(
        q, k, v, g, beta, output_final_state=True, chunk_size=16,
        use_chunk=True, backend='reference', recompute_attn=True)
    assert (o_cache - o_re).abs().max().item() < BIT_MATCH
    assert (s_cache - s_re).abs().max().item() < BIT_MATCH
    # The flag is reference-path-only and documented as ignored when the
    # recurrent path runs (use_chunk=False) — verify it does not break it.
    o_rec, _ = kda_forward(
        q, k, v, g, beta, output_final_state=True, use_chunk=False,
        backend='reference', recompute_attn=True)
    assert o_rec.shape == o_cache.shape


def test_hybrid_config_recompute_parity(device):
    """``HybridConfig.kda_recompute_attn`` yields the same hybrid forward."""
    torch.manual_seed(31)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=16,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=4, csa_nIh=1, csa_cI=4,
        csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=4, hca_sliding_window=4,
    )
    cfg_re = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=16, kda_recompute_attn=True,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=4, csa_nIh=1, csa_cI=4,
        csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=4, hca_sliding_window=4,
    )
    model_cache = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    model_re = HybridKCHAttention(cfg_re, total_layers=5).to(device).eval()
    model_re.load_state_dict(model_cache.state_dict())
    x = torch.randn(2, 48, cfg.d_model, device=device) * 0.1
    with torch.no_grad():
        y_cache = model_cache(x)
        y_re = model_re(x)
    diff = (y_cache - y_re).abs().max().item()
    assert diff < BIT_MATCH, f'max|diff|={diff:.2e}'
    # Type guard: ints must not silently flip the memory policy.
    with pytest.raises(TypeError):
        HybridConfig(
            d_model=32, n_heads_qk=2, n_heads_v=2,
            head_dim_k=8, head_dim_v=8, kda_recompute_attn=1)


def test_compiled_recurrent_kda_fast_parity(device):
    """Fast ``torch.compile`` (fullgraph) parity smoke test.

    The longer compile regression test in ``run_correctness.py`` is marked
    ``slow`` and therefore skipped in the default fast test run. This
    tiny-shape check keeps the dynamo tracing contract exercised in any
    fast run of this file.
    """
    q, k, v, g, beta = _make_case(41, 1, 8, 1, 4, 4, device=device)
    o_ref, s_ref = naive_recurrent_kda(
        q, k, v, g, beta, output_final_state=True)
    o_cmp, s_cmp = compiled_recurrent_kda(
        q, k, v, g, beta, output_final_state=True,
        dynamic=False, fullgraph=True)
    assert (o_ref - o_cmp).abs().max().item() < BIT_MATCH
    assert (s_ref - s_cmp).abs().max().item() < BIT_MATCH



def test_compiled_chunk_kda_fullgraph_per_channel_decay(device):
    """``compiled_chunk_kda`` is fullgraph-traceable and decay-faithful.

    The chunk setup previously reduced the gate range to a host scalar and
    compared storage pointers, both of which are untraceable under
    ``torch.compile(fullgraph=True)``. The branch-free clamp and the
    metadata-only ownership check fix that. This test runs both the
    cached-``Aqk`` and the ``recompute_attn`` memory policy on an
    anisotropic per-channel-decay input and checks output + final state.
    """
    import torch._dynamo as _dynamo

    q, k, v, g, beta = _make_case(51, 1, 32, 1, 8, 8, device=device)
    # Per-channel anisotropy: the first half of the key channels forgets
    # ~8x faster; the late prefix forces the >50 clamp branch.
    g = g.clone()
    g[..., :4] *= 8.0
    g[:, 16:] = -5.0
    for recompute in (False, True):
        o_ref, s_ref = naive_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=16,
            recompute_attn=recompute)
        _dynamo.reset()
        o_cmp, s_cmp = compiled_chunk_kda(
            q, k, v, g, beta, output_final_state=True, chunk_size=16,
            recompute_attn=recompute, dynamic=False, fullgraph=True)
        assert (o_ref - o_cmp).abs().max().item() < BIT_MATCH
        assert (s_ref - s_cmp).abs().max().item() < BIT_MATCH


def test_unconditional_decay_clamp_is_below_threshold_identity(device):
    """Always-clamping is bit-identical when every gate difference is <= 50.

    Removing the data-dependent range check is only valid if the clamp is a
    numerical no-op in the regime where the old code skipped it.
    """
    q, k, v, g, beta = _make_case(52, 1, 32, 1, 8, 8, device=device)
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, None, None, False, 16, -10.0, None, 'test',
        recompute_attn=True)
    assert prep['clamp_g_diff'] is True
    A_open = torch.zeros(*prep['g'].shape[:-1], 16,
                         dtype=prep['g'].dtype, device=device)
    A_closed = A_open.clone()
    _chunk_kda_attn_loop_eager(
        prep['g'], prep['k'], prep['q'], A_open, A_open, 16,
        False, True, prep['upper_mask'], 1)
    _chunk_kda_attn_loop_eager(
        prep['g'], prep['k'], prep['q'], A_closed, A_closed, 16,
        True, True, prep['upper_mask'], 1)
    assert torch.equal(A_open, A_closed)


def test_compiled_chunk_kda_delegates_degenerate_and_initial_state(device):
    """T=0 and ``initial_state`` keep the eager contract (no compile cost)."""
    q, k, v, g, beta = _make_case(53, 1, 16, 1, 8, 8, device=device)
    S0 = torch.randn(1, 1, 8, 8, device=device) * 0.1
    o_ref, s_ref = naive_chunk_kda(
        q, k, v, g, beta, initial_state=S0, output_final_state=True,
        chunk_size=16)
    o_cmp, s_cmp = compiled_chunk_kda(
        q, k, v, g, beta, initial_state=S0, output_final_state=True,
        chunk_size=16, dynamic=False, fullgraph=True)
    assert (o_ref - o_cmp).abs().max().item() < BIT_MATCH
    assert (s_ref - s_cmp).abs().max().item() < BIT_MATCH
    q0 = torch.zeros(1, 0, 1, 8, device=device)
    v0 = torch.zeros(1, 0, 1, 8, device=device)
    b0 = torch.zeros(1, 0, 1, device=device)
    o0, s0 = compiled_chunk_kda(
        q0, q0, v0, q0, b0, output_final_state=True)
    assert o0.shape == (1, 0, 1, 8)
    assert s0.shape == (1, 1, 8, 8)

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
