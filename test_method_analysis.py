"""Regression tests for the headwise-fusion prototype in ``method_analysis``.

The prototype's CSA / HCA branches used ``torch.isinf(...).all()`` as their
fully-masked predicate.  ``isinf`` also matches ``+inf``, so an overflowed
row (all visible scores ``+inf``) was silently zeroed instead of taking the
softmax limit (uniform mass over the ``+inf`` entries).  Rows that mixed
``+inf`` with ``-inf`` also passed a bare ``torch.softmax`` and produced
``NaN``.  Both branches now share ``ops_csa._nan_safe_softmax``, the same
contract used by the reference operators.

Run via pytest::

    pytest -q test_method_analysis.py

Or directly::

    python3 test_method_analysis.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from method_analysis import HeadwiseConfig, HeadwiseFusedAttention  # noqa: E402
from ops_csa import csa_compress_kv, _causal_block_mask  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _positive_model(seed=0):
    """A model whose projections keep every visible score strictly positive.

    Positive weights and a positive hidden state make ``q`` and the
    compressed values positive after normalization, so multiplying by a huge
    ``scale`` yields ``+inf`` (not ``-inf``) on every visible score.
    """
    torch.manual_seed(seed)
    cfg = HeadwiseConfig(
        d_model=32, H_total=3, H_kda=1, H_csa=1, H_hca=1, head_dim=8,
        csa_m=4, csa_topk=2, hca_m2=4, csa_c=8, hca_c=8)
    model = HeadwiseFusedAttention(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.abs_()
        h = torch.ones(1, 16, cfg.d_model) * 0.5
    model.scale = 1e39
    return model, h


def _csa_reference(model, h):
    cfg = model.cfg
    B, T, _ = h.shape
    H, c, m = cfg.H_csa, cfg.csa_c, cfg.csa_m
    pad = (-T) % m
    if pad:
        h = F.pad(h, (0, 0, 0, pad))
    Tp = h.shape[1]
    n_blocks = Tp // m
    C = model.csa_kv(h)
    Z = model.csa_z(h)
    C_comp_n = F.normalize(csa_compress_kv(C, Z, model.csa_B, m), dim=-1)
    q = F.normalize(
        model.csa_q(h).view(B, Tp, H, c).to(C_comp_n.dtype), dim=-1)
    cbm = _causal_block_mask(Tp, n_blocks, m, h.device)
    scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n)
    scores = scores * model.scale
    scores = scores.masked_fill(~cbm[None, None, :, :], float('-inf'))
    topk = min(cfg.csa_topk, n_blocks)
    topk_scores, topk_idx = torch.topk(scores, topk, dim=-1)
    pos = topk_scores.isposinf()
    p = pos.to(scores.dtype) / pos.sum(dim=-1, keepdim=True).clamp(min=1)
    b_idx = torch.arange(B, device=h.device).view(B, 1, 1, 1)
    C_sel = C_comp_n[b_idx, topk_idx]
    out = torch.einsum('b h t k, b h t k d -> b h t d', p, C_sel)
    if pad:
        out = out[:, :, :T]
    return out


def _hca_reference(model, h):
    cfg = model.cfg
    B, T, _ = h.shape
    m2 = cfg.hca_m2
    pad = (-T) % m2
    if pad:
        h = F.pad(h, (0, 0, 0, pad))
    Tp = h.shape[1]
    n_blocks = Tp // m2
    C = model.hca_kv(h)
    Z = model.hca_z(h)
    C_comp_n = F.normalize(csa_compress_kv(C, Z, model.hca_B, m2), dim=-1)
    cbm = _causal_block_mask(Tp, n_blocks, m2, h.device)
    visible = cbm[None, None, :, :]
    p = visible.to(C_comp_n.dtype)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1)
    out = torch.einsum('b h t n, b n d -> b h t d', p, C_comp_n)
    if pad:
        out = out[:, :, :T]
    return out


def test_headwise_csa_and_hca_use_the_softmax_limit_for_positive_inf():
    """Overflowed scores keep the softmax limit instead of being zeroed."""
    model, h = _positive_model()
    with torch.no_grad():
        csa = model._csa_heads(h)
        hca = model._hca_heads(h)
    csa_ref = _csa_reference(model, h)
    hca_ref = _hca_reference(model, h)

    _ok('CSA +inf scores stay finite', bool(torch.isfinite(csa).all()),
        f'nan_count={(~torch.isfinite(csa)).sum().item()}')
    _ok('HCA +inf scores stay finite', bool(torch.isfinite(hca).all()),
        f'nan_count={(~torch.isfinite(hca)).sum().item()}')
    _ok('CSA matches the uniform limit over +inf entries',
        torch.allclose(csa, csa_ref, atol=1e-6, rtol=0),
        f'max|d|={(csa - csa_ref).abs().max().item():.3e}')
    _ok('HCA matches the uniform limit over visible +inf blocks',
        torch.allclose(hca, hca_ref, atol=1e-6, rtol=0),
        f'max|d|={(hca - hca_ref).abs().max().item():.3e}')
    _ok('overflowed output is not silently zeroed',
        float(csa.abs().max()) > 0 and float(hca.abs().max()) > 0,
        f'csa={float(csa.abs().max()):.3e} hca={float(hca.abs().max()):.3e}')


def test_headwise_fully_masked_rows_are_still_zeroed():
    """The first block sees no preceding block for CSA / HCA: output zeros."""
    model, h = _positive_model()
    with torch.no_grad():
        csa = model._csa_heads(h)
        hca = model._hca_heads(h)
    _ok('CSA block-0 rows are zeroed',
        torch.equal(csa[:, :, :model.cfg.csa_m], torch.zeros_like(
            csa[:, :, :model.cfg.csa_m])), '')
    _ok('HCA block-0 rows are zeroed',
        torch.equal(hca[:, :, :model.cfg.hca_m2], torch.zeros_like(
            hca[:, :, :model.cfg.hca_m2])), '')


_TEST_FUNCS = [
    test_headwise_csa_and_hca_use_the_softmax_limit_for_positive_inf,
    test_headwise_fully_masked_rows_are_still_zeroed,
]


if __name__ == '__main__':
    for _fn in _TEST_FUNCS:
        _fn()
    print('all method_analysis tests passed')
