"""Regression tests for the dense multi-query attention reference (Exp 14).

What is pinned here
-------------------
``mqa_attention`` supplies the softmax ladder's KV-sharing rung. Its
correctness rests on two independent
certificates, and both are tested:

* **grouped execution == expanded MHA.** :meth:`MQAAttn.to_mha` expands
  the shared K/V projection rows with ``repeat_interleave``; the grouped
  einsum path and the expanded-MHA path must produce the same outputs.
  An explicit, independently written oracle built on
  :func:`mqa_attention.expand_kv_heads` is a second witness (guards
  against a shared-but-wrong convention).
* **incremental decode == full-sequence.** The append-only cache path is
  the same code path as the full forward by construction, so the test
  appends in single-token AND multi-token steps and compares against the
  cached full-sequence forward.

Also pinned: causality, the parameter/FLOP/KV-byte accounting identities
(the sharing factor divides decode traffic while prefill FLOPs differ only
in the K/V projection), and the validation surface.

Run via pytest::

    pytest -q test_mqa_attention.py

Or directly::

    python3 test_mqa_attention.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402

import mqa_attention as mqa  # noqa: E402

D_MODEL, H, HEAD_DIM = 32, 4, 8
TOL = 1e-6


def make_attn(n_kv_heads=2, *, bias=False, seed=0):
    torch.manual_seed(seed)
    return mqa.MQAAttn(D_MODEL, H, n_kv_heads, HEAD_DIM, bias=bias)


def oracle_forward(attn, x):
    """Independent expanded-MHA oracle over the layer's own weights."""
    B, T, _ = x.shape
    q = attn.q(x).view(B, T, attn.n_heads, attn.head_dim)
    k = attn.k(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    v = attn.v(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    k_e, v_e = mqa.expand_kv_heads(
        k, v, n_heads=attn.n_heads, n_kv_heads=attn.n_kv_heads)
    scores = torch.einsum('bthd,bshd->bhts', q, k_e) * attn.scale
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device),
                      diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum('bhts,bshd->bthd', probs, v_e)
    return attn.o(out.reshape(B, T, attn.n_heads * attn.head_dim))


# ---------------------------------------------------------------------------
# equivalence certificates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_kv_heads', [4, 2, 1])
def test_grouped_equals_expanded_mha(n_kv_heads):
    attn = make_attn(n_kv_heads)
    mha = attn.to_mha()
    assert mha.n_kv_heads == mha.n_heads == H
    assert mha.group_size == 1
    x = torch.randn(2, 9, D_MODEL)
    out_grouped = attn(x)
    out_expanded = mha(x)
    assert torch.allclose(out_grouped, out_expanded, atol=TOL), \
        (out_grouped - out_expanded).abs().max().item()


@pytest.mark.parametrize('n_kv_heads', [4, 2, 1])
def test_grouped_equals_independent_oracle(n_kv_heads):
    attn = make_attn(n_kv_heads)
    x = torch.randn(2, 7, D_MODEL)
    assert torch.allclose(attn(x), oracle_forward(attn, x), atol=TOL)


def test_to_mha_is_independent_copy():
    attn = make_attn(2)
    mha = attn.to_mha()
    assert mha.q.weight.data_ptr() != attn.q.weight.data_ptr()
    before = attn.q.weight.detach().clone()
    with torch.no_grad():
        mha.q.weight.add_(1.0)
    # The grouped source must be untouched by mutating the expansion.
    assert torch.equal(attn.q.weight, before)
    x = torch.randn(1, 4, D_MODEL)
    assert torch.allclose(attn(x), oracle_forward(attn, x), atol=TOL)


def test_to_mha_bias_expansion():
    attn = make_attn(2, bias=True)
    mha = attn.to_mha()
    x = torch.randn(1, 5, D_MODEL)
    assert torch.allclose(attn(x), mha(x), atol=TOL)


# ---------------------------------------------------------------------------
# incremental decode
# ---------------------------------------------------------------------------

def test_incremental_decode_single_tokens_matches_full():
    attn = make_attn(2)
    x = torch.randn(2, 6, D_MODEL)
    full = attn(x)
    cache = None
    outs = []
    for t in range(x.shape[1]):
        out, cache = attn.forward_incremental(x[:, t:t + 1], cache)
        outs.append(out)
    dec = torch.cat(outs, dim=1)
    assert torch.allclose(full, dec, atol=TOL), (full - dec).abs().max().item()
    assert cache.seen == x.shape[1]
    assert cache.k.shape == (2, 6, attn.n_kv_heads, HEAD_DIM)


def test_incremental_decode_chunked_matches_full():
    attn = make_attn(1)
    x = torch.randn(1, 7, D_MODEL)
    full = attn(x)
    cache = None
    outs = []
    for start, end in ((0, 2), (2, 4), (4, 5), (5, 7)):
        out, cache = attn.forward_incremental(x[:, start:end], cache)
        outs.append(out)
    dec = torch.cat(outs, dim=1)
    assert torch.allclose(full, dec, atol=TOL), (full - dec).abs().max().item()


def test_causality_future_tokens_do_not_change_past():
    attn = make_attn(2)
    x = torch.randn(1, 8, D_MODEL)
    out_a = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, D_MODEL)
    out_b = attn(x2)
    assert torch.allclose(out_a[:, :5], out_b[:, :5], atol=TOL)


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

def test_parameter_accounting_matches_module():
    for bias in (False, True):
        attn = make_attn(2, bias=bias)
        live = int(sum(p.numel() for p in attn.parameters()))
        expected = mqa.attention_parameter_elements(
            D_MODEL, H, 2, HEAD_DIM, bias=bias)
        assert live == expected


def test_prefill_flops_difference_is_projection_only():
    T = 12
    f_mha = mqa.attention_prefill_flops(T, D_MODEL, H, H, HEAD_DIM)
    f_gqa = mqa.attention_prefill_flops(T, D_MODEL, H, 2, HEAD_DIM)
    f_mqa = mqa.attention_prefill_flops(T, D_MODEL, H, 1, HEAD_DIM)
    per_head = 2 * T * D_MODEL * HEAD_DIM
    assert f_mha - f_gqa == pytest.approx(2 * per_head * (H - 2))
    assert f_mha - f_mqa == pytest.approx(2 * per_head * (H - 1))


def test_kv_budget_identities():
    assert mqa.kv_state_elements_per_token(1, HEAD_DIM) == 2 * HEAD_DIM
    assert mqa.kv_bytes_reduction(H, H) == 1.0
    assert mqa.kv_bytes_reduction(H, 2) == 2.0
    assert mqa.kv_bytes_reduction(H, 1) == 4.0
    T, batch, itemsize = 5, 2, 4
    read_mha = mqa.attention_kv_read_elements(T, H, HEAD_DIM, batch=batch)
    read_mqa = mqa.attention_kv_read_elements(T, 1, HEAD_DIM, batch=batch)
    assert read_mha == batch * T * mqa.kv_state_elements_per_token(H, HEAD_DIM)
    assert read_mha == 4 * read_mqa
    assert read_mqa * itemsize == batch * T * 2 * HEAD_DIM * itemsize
    assert mqa.attention_parameter_elements(D_MODEL, H, H, HEAD_DIM) \
        > mqa.attention_parameter_elements(D_MODEL, H, 1, HEAD_DIM)


def test_expand_kv_heads_matches_repeat_interleave():
    k = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    v = k + 1.0
    k_e, v_e = mqa.expand_kv_heads(k, v, n_heads=6, n_kv_heads=2)
    assert k_e.shape == (2, 3, 6, 4)
    assert torch.equal(k_e, k.repeat_interleave(3, dim=-2))
    # Contiguous grouping: query heads h*3, h*3+1, h*3+2 copy source head h.
    assert torch.equal(k_e[:, :, 2], k[:, :, 0])
    assert torch.equal(k_e[:, :, 3], k[:, :, 1])
    assert torch.equal(k_e[:, :, 5], k[:, :, 1])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_construction_validation():
    with pytest.raises(ValueError):
        mqa.MQAAttn(D_MODEL, 5, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.MQAAttn(D_MODEL, H, 0, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.MQAAttn(D_MODEL, H, H, 0)
    with pytest.raises(TypeError):
        mqa.MQAAttn(D_MODEL, H, H, HEAD_DIM, bias='yes')
    with pytest.raises(ValueError):
        mqa.MQAAttn(0, H, H, HEAD_DIM)


def test_forward_validation():
    attn = make_attn(2)
    with pytest.raises(ValueError):
        attn(torch.randn(2, 3, D_MODEL + 1))
    with pytest.raises(ValueError):
        attn(torch.randn(2, D_MODEL))
    with pytest.raises(ValueError):
        attn(torch.randn(2, 0, D_MODEL))
    with pytest.raises(TypeError):
        attn.forward_incremental(torch.randn(2, 1, D_MODEL), cache='nope')
    with pytest.raises(ValueError):
        attn.forward_incremental(
            torch.randn(3, 1, D_MODEL),
            cache=mqa.MQACache(torch.zeros(2, 4, 2, HEAD_DIM),
                               torch.zeros(2, 4, 2, HEAD_DIM)))
    with pytest.raises(ValueError):
        attn.forward_incremental(
            torch.randn(2, 1, D_MODEL),
            cache=mqa.MQACache(torch.zeros(2, 4, 3, HEAD_DIM),
                               torch.zeros(2, 4, 3, HEAD_DIM)))


def test_mqa_cache_validates_the_kv_pair():
    """K and V form one cache: shape/dtype/device mismatches fail early."""
    with pytest.raises(ValueError):
        mqa.MQACache(torch.zeros(2, 4, 2, HEAD_DIM),
                     torch.zeros(2, 4, 3, HEAD_DIM))
    with pytest.raises(ValueError):
        mqa.MQACache(torch.zeros(2, 4, 2, HEAD_DIM),
                     torch.zeros(2, 4, 2, HEAD_DIM, dtype=torch.float64))
    with pytest.raises(ValueError):
        mqa.MQACache(torch.zeros(2, 4, 2),
                     torch.zeros(2, 4, 2))
    with pytest.raises(TypeError):
        mqa.MQACache('k', 'v')

    # A valid cache from a float64 model must not be fed a float32 input
    # (torch.cat would promote or fail deep inside the einsum).
    attn = make_attn(2).double()
    cache = mqa.MQACache(torch.zeros(2, 4, 2, HEAD_DIM),
                         torch.zeros(2, 4, 2, HEAD_DIM))
    with pytest.raises(ValueError):
        attn.forward_incremental(
            torch.randn(2, 1, D_MODEL, dtype=torch.float64), cache)


def test_accounting_validation():
    with pytest.raises(ValueError):
        mqa.attention_parameter_elements(D_MODEL, 5, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.attention_prefill_flops(-1, D_MODEL, H, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.attention_prefill_flops(4, D_MODEL, 5, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.kv_bytes_reduction(4, 3)
    with pytest.raises(ValueError):
        mqa.kv_state_elements_per_token(0, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.attention_kv_read_elements(4, 2, HEAD_DIM, batch=0)
    huge = 10 ** 400
    with pytest.raises(ValueError):
        mqa.attention_prefill_flops(huge, D_MODEL, H, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.attention_prefill_flops(4, huge, H, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.kv_bytes_reduction(huge, 1)
    with pytest.raises(ValueError):
        mqa.attention_kv_read_elements(huge, 2, HEAD_DIM)
    with pytest.raises(ValueError):
        mqa.attention_kv_read_elements(4, 2, HEAD_DIM, batch=huge)
    with pytest.raises(ValueError):
        mqa.expand_kv_heads(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4),
                            n_heads=5, n_kv_heads=2)
    with pytest.raises(ValueError):
        mqa.expand_kv_heads(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4),
                            n_heads=4, n_kv_heads=2)
    with pytest.raises(ValueError):
        mqa.expand_kv_heads(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 3),
                            n_heads=4, n_kv_heads=2)
    with pytest.raises(ValueError):
        mqa.expand_kv_heads(torch.zeros(1, 2, 2, 4), torch.zeros(1, 3, 2, 4),
                            n_heads=4, n_kv_heads=2)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
