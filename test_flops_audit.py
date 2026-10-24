"""Regression tests for the measured FLOPs audit and online softmax.

Pins three contracts:

1. **The audit counter is exact.**  mm / addmm / bmm, broadcasted
   ``einsum`` (which ``torch.utils.flop_counter.FlopCounterMode`` misses),
   grouped convolution, triangular solve, and the special-op ledger are
   all checked against hand-computed element/MAC counts.
2. **The exact reference model matches the operators.**  For the default
   KDA (recurrent and chunk paths, including a padded sequence), CSA and
   HCA layers, ``reference_gemm_macs`` equals the measured GEMM+TRSM work
   to the MAC.  The KDA recurrent path additionally satisfies
   ``measured + broadcast_encoded == executed canonical``.
3. **The online-softmax primitive is value-equal to dense softmax** (with
   and without the denominator-only sink), chunk-size invariant,
   re-association invariant, and performs the same GEMM MACs.

How to run::

    pytest -q test_flops_audit.py
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flops_audit as fa  # noqa: E402
import online_softmax as osf  # noqa: E402


@pytest.fixture
def device(request):
    try:
        return request.config.getoption('--device')
    except (ValueError, KeyError):
        return 'cpu'


def _count(fn):
    """Run ``fn`` under the audit counter and return the ledger."""
    mode = fa._GemmAuditMode()
    with mode:
        fn()
    return mode.ledger()


# ---------------------------------------------------------------------------
# 1. Counter exactness
# ---------------------------------------------------------------------------
def test_counter_counts_matmul_and_broadcast_einsum():
    def run():
        a = torch.randn(4, 8)
        b = torch.randn(8, 16)
        torch.mm(a, b)
        x = torch.randn(2, 4, 5)
        y = torch.randn(2, 5, 6)
        torch.bmm(x, y)
        q = torch.randn(2, 7, 3, 5)
        c = torch.randn(2, 4, 5)
        torch.einsum('b t h d, b n d -> b h t n', q, c)

    ledger = _count(run)
    assert ledger.macs_by_op['aten::mm'] == 4 * 8 * 16
    assert ledger.macs_by_op['aten::bmm'] == 2 * 4 * 5 * 6 + 2 * 7 * 3 * 4 * 5


def test_counter_counts_depthwise_conv1d():
    conv = nn.Conv1d(8, 8, kernel_size=3, groups=8, bias=False)
    x = torch.randn(1, 8, 32)
    ledger = _count(lambda: conv(x))
    out_len = 32 - 3 + 1
    assert ledger.macs_by_op['aten::convolution'] == 1 * 8 * out_len * 1 * 3


def test_counter_counts_triangular_solve():
    a = torch.randn(2, 3, 8, 8).tril()
    b = torch.randn(2, 3, 8, 8)
    ledger = _count(lambda: torch.linalg.solve_triangular(a, b, upper=False))
    systems = 2 * 3
    assert ledger.trsm_calls == 1
    assert ledger.trsm_macs == systems * 8 * 8 * (8 - 1) // 2


def test_counter_records_special_ops():
    x = torch.randn(2, 4, 16)
    tiny = nn.Linear(16, 16, bias=False)

    def run():
        torch.softmax(x, dim=-1)
        torch.sin(x)
        tiny(x).sigmoid_()
        torch.nn.functional.silu(x)

    ledger = _count(run)
    elems = 2 * 4 * 16
    assert ledger.special_ops['softmax'] == elems
    assert ledger.special_ops['sin'] == elems
    assert ledger.special_ops['sigmoid'] == elems
    assert ledger.special_ops['silu'] == elems


# ---------------------------------------------------------------------------
# 2. Exact reference model vs measured
# ---------------------------------------------------------------------------
def _cfg(**overrides):
    return fa.probe_config(**overrides)


@pytest.mark.parametrize('T', [8, 32, 63])
def test_kda_exact_model_recurrent(T, device):
    cfg = _cfg()
    row = fa.audit_layer('kda', cfg, T, device=device)
    assert row['path'] == 'recurrent'
    assert row['model_matches_measurement'], (
        f'T={T}: model={row["reference_model_macs"]} '
        f'measured={row["reference_gemm_macs"]} '
        f'diff={row["model_minus_measured_macs"]}')


@pytest.mark.parametrize('T', [64, 128, 256])
def test_kda_exact_model_chunk(T, device):
    cfg = _cfg()
    row = fa.audit_layer('kda', cfg, T, device=device)
    assert row['path'] == 'chunk'
    assert row['ledger'].trsm_calls > 0
    assert row['model_matches_measurement'], (
        f'T={T}: model={row["reference_model_macs"]} '
        f'measured={row["reference_gemm_macs"]}')


def test_kda_exact_model_padded_sequence(device):
    cfg = _cfg()
    row = fa.audit_layer('kda', cfg, 100, device=device)
    assert row['path'] == 'chunk'
    assert row['model_matches_measurement'], row['model_minus_measured_macs']
    BT = cfg.kda_chunk_size
    padded = math.ceil(100 / BT) * BT
    NT = padded // BT
    B, HV, K = 1, cfg.n_heads_v, cfg.head_dim_k
    from ops_kda import _attn_block_rows
    rows = _attn_block_rows(B, HV, NT, BT, K)
    a_build = B * HV * NT * K * sum(rows * c1 for c1 in range(rows, BT + 1, rows))
    decay = NT * B * HV * BT * K
    gate = B * HV * padded * K
    assert row['ledger'].special_ops['exp'] == a_build + decay + gate
    assert row['ledger'].special_ops['softplus'] == 100 * HV * K


@pytest.mark.parametrize('T', [1, 8, 32, 63, 100])
def test_mamba2_exact_model_sequential(T, device):
    cfg = _cfg()
    row = fa.audit_layer('mamba2', cfg, T, device=device)
    assert row['path'] == 'sequential'
    assert row['model_matches_measurement'], (
        f'T={T}: model={row["reference_model_macs"]} '
        f'measured={row["reference_gemm_macs"]} '
        f'diff={row["model_minus_measured_macs"]}')
    assert row['canonical_flops'] > 0


def test_mamba2_canonical_terms_and_executed_identity(device):
    cfg = _cfg()
    for T in (1, 8, 32, 63, 100):
        row = fa.audit_layer('mamba2', cfg, T, device=device)
        terms = row['canonical_terms']
        assert abs(sum(terms.values()) - row['canonical_flops']) < 1e-9
        assert abs(row['executed_flops'] - row['canonical_flops']) < 1e-9, (
            T, row['executed_flops'], row['canonical_flops'])
        assert row['executed_flops'] == (
            2 * row['reference_gemm_macs']
            + 2 * row['broadcast_encoded_macs'])
        assert terms['scan'] == 3 * row['broadcast_encoded_macs']


def test_mamba2_exact_model_rejects_unmodelled_path(device):
    cfg = _cfg(mamba_chunk_size=16)
    row = fa.audit_layer('mamba2', cfg, 64, device=device)
    assert row['path'] == 'chunked'
    assert row['reference_model_macs'] is None
    with pytest.raises(NotImplementedError):
        fa.mamba2_reference_gemm_macs(cfg, 64)


def test_csa_exact_model(device):
    cfg = _cfg()
    for T in (64, 128, 100):
        row = fa.audit_layer('csa', cfg, T, device=device)
        assert row['model_matches_measurement'], (
            f'T={T}: model={row["reference_model_macs"]} '
            f'measured={row["reference_gemm_macs"]}')


def test_csa_topk1_folded_value_term_is_not_charged(device):
    cfg = _cfg(csa_topk=1)
    for T in (17, 64, 100):
        row = fa.audit_layer('csa', cfg, T, device=device)
        assert row['model_matches_measurement'], (
            f'T={T}: model={row["reference_model_macs"]} '
            f'measured={row["reference_gemm_macs"]} '
            f'diff={row["model_minus_measured_macs"]}')


def test_hca_exact_model(device):
    cfg = _cfg()
    for T in (96, 128, 100):
        row = fa.audit_layer('hca', cfg, T, device=device)
        assert row['model_matches_measurement'], (
            f'T={T}: model={row["reference_model_macs"]} '
            f'measured={row["reference_gemm_macs"]}')


def test_hca_single_block_value_fold_documented(device):
    cfg = _cfg()
    row = fa.audit_layer('hca', cfg, 32, device=device)
    Tp = 32
    shortfall = row['model_minus_measured_macs']
    assert 0 <= shortfall <= Tp * cfg.hca_c * cfg.hca_nh


def test_geometry_kwargs_matches_exp14_mapping():
    from run_train_roofline import hybrid_geometry_kwargs
    cfg = _cfg()
    assert fa.geometry_kwargs(cfg) == hybrid_geometry_kwargs(cfg)


def test_canonical_terms_sum_matches_prefill_flops():
    cfg = _cfg()
    for T in (32, 64, 128, 100):
        terms = fa.kda_canonical_terms(cfg, T)
        assert sum(terms.values()) == pytest.approx(
            fa.canonical_flops('kda', cfg, T), rel=1e-12)


def test_kda_recurrent_identity_with_canonical():
    cfg = _cfg()
    T = 32
    executed = fa.kda_executed_flops(cfg, T)
    measured = 2 * fa.kda_reference_gemm_macs(cfg, T)
    broadcast = 2 * fa.kda_broadcast_encoded_macs(cfg, T)
    assert measured + broadcast == pytest.approx(executed, rel=1e-12)
    assert executed < fa.canonical_flops('kda', cfg, T)


def test_kda_path_selection():
    cfg = _cfg()
    assert fa.kda_path(cfg, 63) == 'recurrent'
    assert fa.kda_path(cfg, 64) == 'chunk'
    assert fa.kda_executed_flops(cfg, 32) < fa.canonical_flops('kda', cfg, 32)
    always_recurrent = _cfg(kda_chunk_size=0)
    assert fa.kda_path(always_recurrent, 4096) == 'recurrent'
    assert (fa.kda_executed_flops(always_recurrent, 128)
            == pytest.approx(fa.canonical_flops('kda', always_recurrent, 128),
                             rel=1e-12))


def test_kda_chunk_reference_tax_is_scale_invariant():
    cfg = _cfg()
    row128 = fa.audit_layer('kda', cfg, 128)
    row256 = fa.audit_layer('kda', cfg, 256)
    assert row128['reference_tax_vs_canonical'] > 1.2
    assert (row128['reference_tax_vs_canonical']
            == pytest.approx(row256['reference_tax_vs_canonical'], rel=1e-9))


def test_kda_executed_equals_canonical_on_chunk_path():
    cfg = _cfg()
    for T in (64, 128):
        assert fa.kda_executed_flops(cfg, T) == pytest.approx(
            fa.canonical_flops('kda', cfg, T), rel=1e-12)


def test_audit_special_ledger_gate_counts(device):
    cfg = _cfg()
    T = 32
    row = fa.audit_layer('kda', cfg, T, device=device)
    special = row['ledger'].special_ops
    K, HV = cfg.head_dim_k, cfg.n_heads_v
    assert special['softplus'] == T * HV * K
    assert special['exp'] == T * HV * K
    assert special['silu'] == T * (2 * cfg.n_heads_qk * K + HV * cfg.head_dim_v)
    assert special['sigmoid'] == T * HV
    assert special['l2_norm'] == 2 * T * cfg.n_heads_qk * K


def test_audit_softmax_ledger_csa_hca(device):
    cfg = _cfg()
    for kind in ('csa', 'hca'):
        row = fa.audit_layer(kind, cfg, 128, device=device)
        special = row['ledger'].special_ops
        assert special['softmax'] > 0
        assert special['logsumexp'] > 0
        assert special['sin'] == special['cos'] > 0
        assert row['canonical_flops'] > 0


def test_audit_validation():
    cfg = _cfg()
    with pytest.raises(ValueError):
        fa.audit_layer('rwkv7', cfg, 64)
    with pytest.raises(ValueError):
        fa.audit_layer('kda', cfg, -1)
    with pytest.raises(TypeError):
        fa.audit_layer('kda', cfg, 64.0)
    with pytest.raises(ValueError):
        fa.canonical_flops('kda', cfg, 64, batch=0)
    huge = 10 ** 400
    for args in ((64, dict(batch=huge)), (huge, {})):
        with pytest.raises(ValueError):
            fa.canonical_flops('kda', cfg, args[0], **args[1])
    representable = 10 ** 308
    with pytest.raises(ValueError):
        fa.canonical_flops('kda', cfg, representable)
    with pytest.raises(ValueError):
        fa.canonical_flops('kda', cfg, 10 ** 200, batch=10 ** 200)
    with pytest.raises(ValueError):
        fa.kda_executed_flops(cfg, representable)
    with pytest.raises(ValueError):
        fa.mamba2_executed_flops(cfg, representable)
    assert math.isfinite(fa.canonical_flops('kda', cfg, 64, batch=2))


# ---------------------------------------------------------------------------
# 3. Online softmax primitive
# ---------------------------------------------------------------------------
def _dense_attention(q, k, v, mask=None, sink_logits=None):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum('b h t d, b h n d -> b h t n', q, k) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    if sink_logits is None:
        p = torch.softmax(scores, dim=-1)
    else:
        row_max = scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isneginf(row_max),
                              torch.zeros_like(row_max), row_max)
        e = (scores - row_max).exp()
        denom = e.sum(-1, keepdim=True) + (
            sink_logits.view(1, -1, 1, 1) - row_max).exp()
        p = e / denom
    return torch.einsum('b h t n, b h n d -> b h t d', p, v)


def test_online_softmax_matches_dense():
    torch.manual_seed(0)
    B, H, T, N, D = 2, 3, 16, 40, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    sink = torch.randn(H)
    dense = _dense_attention(q, k, v)
    dense_sink = _dense_attention(q, k, v, sink_logits=sink)
    for key_chunk in (0, 5, 7, N):
        out = osf.online_softmax_attention(q, k, v, key_chunk=key_chunk)
        assert (out - dense).abs().max().item() < 1e-5
        out_sink = osf.online_softmax_attention(
            q, k, v, key_chunk=key_chunk, sink_logits=sink)
        assert (out_sink - dense_sink).abs().max().item() < 1e-5


def test_online_softmax_causal_mask_matches_dense():
    torch.manual_seed(1)
    B, H, T, D = 2, 2, 12, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    dense = _dense_attention(q, k, v, mask=mask)
    out = osf.online_softmax_attention(q, k, v, key_chunk=5, mask=mask)
    assert (out - dense).abs().max().item() < 1e-5


def test_online_softmax_merge_associativity():
    torch.manual_seed(2)
    B, H, T, N, D = 1, 2, 10, 21, 6
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    sink = torch.randn(H)
    states = [
        osf.online_softmax_state(q, k[:, :, lo:hi], v[:, :, lo:hi])
        for lo, hi in ((0, 5), (5, 13), (13, N))
    ]
    left = osf.online_softmax_merge(
        osf.online_softmax_merge(states[0], states[1]), states[2])
    right = osf.online_softmax_merge(
        states[0], osf.online_softmax_merge(states[1], states[2]))
    out_left = osf.online_softmax_finalize(*left, sink_logits=sink)
    out_right = osf.online_softmax_finalize(*right, sink_logits=sink)
    assert (out_left - out_right).abs().max().item() < 1e-6
    assert (out_left - _dense_attention(q, k, v, sink_logits=sink)
            ).abs().max().item() < 1e-5


def test_online_softmax_all_masked_rows_are_zero():
    torch.manual_seed(3)
    B, H, T, N, D = 1, 1, 4, 3, 3
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, N, D)
    mask = torch.zeros(T, N, dtype=torch.bool)
    out = osf.online_softmax_attention(q, k, k, key_chunk=2, mask=mask)
    assert torch.equal(out, torch.zeros_like(out))


def test_online_softmax_gemm_work_equals_dense():
    torch.manual_seed(4)
    B, H, T, N, D = 2, 3, 16, 40, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    dense = _count(lambda: _dense_attention(q, k, v))
    online = _count(lambda: osf.online_softmax_attention(
        q, k, v, key_chunk=7))
    assert online.gemm_macs == dense.gemm_macs
    assert online.gemm_macs == 2 * B * H * T * N * D
    assert 'softmax' in dense.special_ops
    assert 'softmax' not in online.special_ops


def test_online_softmax_validation():
    q = torch.randn(1, 2, 4, 3)
    k = torch.randn(1, 2, 5, 3)
    v = torch.randn(1, 2, 5, 3)
    with pytest.raises(ValueError):
        osf.online_softmax_attention(q, k, v, key_chunk=-1)
    with pytest.raises(ValueError):
        osf.online_softmax_attention(q, k, torch.randn(1, 2, 5, 4))
    with pytest.raises(ValueError):
        osf.online_softmax_state(q, k, torch.randn(1, 3, 5, 3))
    empty = torch.randn(1, 2, 0, 3)
    with pytest.raises(ValueError):
        osf.online_softmax_attention(q, empty, empty)


def test_counter_counts_rectangular_triangular_solve_rhs():
    """``solve_triangular(A, B)`` counts N from A, columns from B."""
    a = torch.randn(2, 8, 8).tril()
    b = torch.randn(2, 8, 3)
    ledger = _count(lambda: torch.linalg.solve_triangular(
        a, b, upper=False, unitriangular=True))
    # 2 batched systems * (8*7/2 MACs per column) * 3 RHS columns.
    assert ledger.trsm_calls == 1
    assert ledger.trsm_macs == 2 * (8 * (8 - 1) // 2) * 3
