"""Regression tests for the batched chunked-prefill cache path.

``CSADecodingCache.forward_chunk`` / ``HCADecodingCache.forward_chunk``
are the throughput fast path used by ``run_decoding`` for continuation
chunks. They must be *exactly* the per-token ``append_step`` +
``forward_step`` loop (same cache state, same outputs), including:

* an empty cache (``n0 = 0``) and a partial accumulator;
* block-boundary crossings inside the chunk;
* ``win = 0`` and windows shorter/longer than the chunk;
* sparse top-k (ties included) and ``topk >= n_blocks``;
* L2 and paper §2.3.3 RMS normalization;
* partial RoPE on/off and the inverse-rotation output path;
* sinks, explicit scales, and batch > 1;
* float64 caches (the fp32 fast paths must not quantize them);
* mixed cache/query dtypes (fp64 cache + fp32 query and vice versa),
  where both paths must promote to fp64 instead of disagreeing on the
  compute precision.

Run via pytest::

    pytest -q test_chunked_prefill_cache.py
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402

from ops_csa import _qk_normalize  # noqa: E402
from ops_decoding_cache import (  # noqa: E402
    CSADecodingCache,
    HCADecodingCache,
)


def _csa_projs(B, T, c, c_I, scale=0.3):
    return [torch.randn(B, T, c) * scale for _ in range(4)] + [
        torch.randn(B, T, c_I) * scale for _ in range(2)]


def _make_csa(n0=5, T=7, m=4, win=4, topk=2, rope=4, qk='l2',
              sink=True, scale=None, dtype=torch.float32, B=2,
              c=4, c_I=3, nh=2, nIh=2, seed=0):
    torch.manual_seed(seed)
    cache = CSADecodingCache(
        B, c, c_I, m, win, torch.device('cpu'), dtype,
        rope_dim=rope, qk_norm_mode=qk)
    cache._normalize_qk = True
    Ba = torch.randn(m, c) * 0.02
    Bb = torch.randn(m, c) * 0.02
    B_idx = torch.randn(m, c_I) * 0.02
    if n0:
        cache.append_step(*_csa_projs(B, n0, c, c_I), Ba, Bb, B_idx)
    P = _csa_projs(B, T, c, c_I)
    q = torch.randn(B, T, nh, c) * 0.5
    q_idx = torch.randn(B, T, nIh, c_I) * 0.5
    w_idx = torch.randn(B, T, nIh) * 0.5
    q_n = _qk_normalize(q.to(dtype), qk)
    sink_logits = (torch.tensor([0.2, -0.3], dtype=dtype) if sink else None)
    return {
        'cache': cache, 'P': P, 'Ba': Ba, 'Bb': Bb, 'B_idx': B_idx,
        'q': q_n, 'q_idx': q_idx.to(dtype), 'w_idx': w_idx.to(dtype),
        'topk': topk, 'nh': nh, 'nIh': nIh, 'scale': scale,
        'sink': sink_logits, 'T': T,
    }


def _make_hca(n0=5, T=7, m=4, win=4, rope=4, qk='l2', sink=True,
              scale=None, dtype=torch.float32, B=2, c=4, nh=2, seed=0):
    torch.manual_seed(seed)
    cache = HCADecodingCache(
        B, c, m, win, torch.device('cpu'), dtype,
        rope_dim=rope, qk_norm_mode=qk)
    cache._normalize_qk = True
    B_pos = torch.randn(m, c) * 0.02
    if n0:
        cache.append_step(
            torch.randn(B, n0, c) * 0.3, torch.randn(B, n0, c) * 0.3,
            B_pos)
    C = torch.randn(B, T, c) * 0.3
    Z = torch.randn(B, T, c) * 0.3
    q = torch.randn(B, T, nh, c) * 0.5
    q_n = _qk_normalize(q.to(dtype), qk)
    sink_logits = (torch.tensor([0.2, -0.3], dtype=dtype) if sink else None)
    return {
        'cache': cache, 'C': C, 'Z': Z, 'B_pos': B_pos, 'q': q_n,
        'nh': nh, 'scale': scale, 'sink': sink_logits, 'T': T,
    }


def _csa_token_outputs(d):
    cache = copy.deepcopy(d['cache'])
    outs = []
    with torch.no_grad():
        for i in range(d['T']):
            cache.append_step(
                *(x[:, i:i + 1] for x in d['P']),
                d['Ba'], d['Bb'], d['B_idx'])
            outs.append(cache.forward_step(
                d['q'][:, i:i + 1], d['q_idx'][:, i:i + 1],
                d['w_idx'][:, i:i + 1], topk=d['topk'], nh=d['nh'],
                nIh=d['nIh'], scale=d['scale'], sink_logits=d['sink'],
                use_ste=False, normalize_qk=True))
    return torch.cat(outs, dim=1), cache


def _csa_chunk_output(d):
    cache = copy.deepcopy(d['cache'])
    with torch.no_grad():
        out = cache.forward_chunk(
            *d['P'], d['Ba'], d['Bb'], d['B_idx'], d['q'],
            d['q_idx'], d['w_idx'], topk=d['topk'], nh=d['nh'],
            nIh=d['nIh'], scale=d['scale'], sink_logits=d['sink'],
            use_ste=False, normalize_qk=True)
    return out, cache


def _hca_token_outputs(d):
    cache = copy.deepcopy(d['cache'])
    outs = []
    with torch.no_grad():
        for i in range(d['T']):
            cache.append_step(
                d['C'][:, i:i + 1], d['Z'][:, i:i + 1], d['B_pos'])
            outs.append(cache.forward_step(
                d['q'][:, i:i + 1], nh=d['nh'], scale=d['scale'],
                sink_logits=d['sink']))
    return torch.cat(outs, dim=1), cache


def _hca_chunk_output(d):
    cache = copy.deepcopy(d['cache'])
    with torch.no_grad():
        out = cache.forward_chunk(
            d['C'], d['Z'], d['B_pos'], d['q'], nh=d['nh'],
            scale=d['scale'], sink_logits=d['sink'])
    return out, cache


def _assert_cache_state_equal(a, b, tol=0.0):
    assert a.n_tokens_seen == b.n_tokens_seen
    assert a.n_blocks == b.n_blocks
    assert a.accumulator_len == b.accumulator_len
    if a.sw_buffer is not None:
        assert b.sw_buffer is not None
        assert a.sw_buffer.n_valid == b.sw_buffer.n_valid
        assert torch.equal(a.sw_buffer.get(), b.sw_buffer.get())
    if a.C_comp is not None:
        if tol == 0.0:
            assert torch.equal(a.C_comp, b.C_comp)
        else:
            assert (a.C_comp - b.C_comp).abs().max().item() <= tol
    if getattr(a, 'K_IComp', None) is not None:
        assert torch.equal(a.K_IComp, b.K_IComp)


@pytest.mark.parametrize('rope', [0, 4])
@pytest.mark.parametrize('win', [0, 1, 4, 12])
@pytest.mark.parametrize('n0,T', [(0, 9), (5, 7), (3, 12)])
def test_csa_forward_chunk_matches_per_token(rope, win, n0, T):
    d = _make_csa(n0=n0, T=T, win=win, rope=rope)
    ref, ref_cache = _csa_token_outputs(d)
    out, chunk_cache = _csa_chunk_output(d)
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 1e-5
    _assert_cache_state_equal(ref_cache, chunk_cache)


@pytest.mark.parametrize('rope', [0, 4])
@pytest.mark.parametrize('win', [0, 1, 4, 12])
@pytest.mark.parametrize('n0,T', [(0, 9), (5, 7), (3, 12)])
def test_hca_forward_chunk_matches_per_token(rope, win, n0, T):
    d = _make_hca(n0=n0, T=T, win=win, rope=rope)
    ref, ref_cache = _hca_token_outputs(d)
    out, chunk_cache = _hca_chunk_output(d)
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 1e-5
    _assert_cache_state_equal(ref_cache, chunk_cache)


@pytest.mark.parametrize('qk', ['l2', 'rms'])
@pytest.mark.parametrize('topk', [0, 2, 100])
def test_csa_forward_chunk_modes(qk, topk):
    d = _make_csa(n0=3, T=12, win=4, rope=4, qk=qk, topk=topk,
                  sink=True, scale=1.0)
    ref, _ = _csa_token_outputs(d)
    out, _ = _csa_chunk_output(d)
    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize('qk', ['l2', 'rms'])
def test_hca_forward_chunk_modes(qk):
    d = _make_hca(n0=3, T=12, win=4, rope=4, qk=qk, sink=True,
                  scale=1.0)
    ref, _ = _hca_token_outputs(d)
    out, _ = _hca_chunk_output(d)
    assert (out - ref).abs().max().item() < 1e-5


def test_forward_chunk_fp64_is_exact_to_roundoff():
    dc = _make_csa(n0=5, T=7, rope=4, qk='rms', dtype=torch.float64)
    ref, _ = _csa_token_outputs(dc)
    out, _ = _csa_chunk_output(dc)
    assert out.dtype == torch.float64
    assert (out - ref).abs().max().item() < 1e-12

    dh = _make_hca(n0=5, T=7, rope=4, qk='rms', dtype=torch.float64)
    ref, _ = _hca_token_outputs(dh)
    out, _ = _hca_chunk_output(dh)
    assert out.dtype == torch.float64
    assert (out - ref).abs().max().item() < 1e-12


@pytest.mark.parametrize('which', ['csa', 'hca'])
@pytest.mark.parametrize('cache_dtype,q_dtype', [
    (torch.float64, torch.float32),
    (torch.float32, torch.float64),
])
def test_forward_chunk_mixed_dtype_matches_step(which, cache_dtype, q_dtype):
    # The cache and the query disagree on precision: the shared policy
    # must promote to fp64 in BOTH the token loop and the chunked path,
    # so values and output dtype stay identical.
    make, token, chunk = (
        (_make_csa, _csa_token_outputs, _csa_chunk_output) if which == 'csa'
        else (_make_hca, _hca_token_outputs, _hca_chunk_output))
    d = make(n0=5, T=7, win=4, rope=4, qk='rms', sink=True,
             dtype=cache_dtype)
    d['q'] = d['q'].to(q_dtype)
    ref, _ = token(d)
    out, _ = chunk(d)
    assert ref.dtype == torch.float64
    assert out.dtype == torch.float64
    assert (out - ref).abs().max().item() < 1e-10


def test_forward_chunk_no_keys_returns_zeros():
    # n0=0, win=0, and T <= m: the first queries have no completed block
    # and no window, so every joint-softmax row is empty.
    dc = _make_csa(n0=0, T=3, m=4, win=0, rope=0, sink=True)
    out, _ = _csa_chunk_output(dc)
    assert torch.equal(out, torch.zeros_like(out))

    dh = _make_hca(n0=0, T=3, m=4, win=0, rope=0, sink=False)
    out, _ = _hca_chunk_output(dh)
    assert torch.equal(out, torch.zeros_like(out))


def test_forward_chunk_singleton_delegates_to_forward_step():
    d = _make_hca(n0=5, T=1, win=4, rope=4)
    ref, ref_cache = _hca_token_outputs(d)
    out, chunk_cache = _hca_chunk_output(d)
    assert torch.equal(out, ref)
    _assert_cache_state_equal(ref_cache, chunk_cache)

    d = _make_csa(n0=5, T=1, win=4, rope=4)
    ref, ref_cache = _csa_token_outputs(d)
    out, chunk_cache = _csa_chunk_output(d)
    assert torch.equal(out, ref)
    _assert_cache_state_equal(ref_cache, chunk_cache)


def test_forward_chunk_validates_shapes():
    dh = _make_hca(n0=4, T=4)
    cache = dh['cache']
    with pytest.raises(ValueError):
        cache.forward_chunk(
            dh['C'][:, :0], dh['Z'][:, :0], dh['B_pos'], dh['q'][:, :0],
            nh=dh['nh'])
    with pytest.raises(ValueError):
        cache.forward_chunk(
            dh['C'], dh['Z'], dh['B_pos'], dh['q'][:, :1], nh=dh['nh'])
    with pytest.raises(ValueError):
        cache.forward_chunk(
            dh['C'][:1], dh['Z'][:1], dh['B_pos'], dh['q'][:1],
            nh=dh['nh'])

    dc = _make_csa(n0=4, T=4)
    cache = dc['cache']
    with pytest.raises(ValueError):
        cache.forward_chunk(
            *dc['P'], dc['Ba'], dc['Bb'], dc['B_idx'],
            dc['q'][:, :1], dc['q_idx'][:, :1], dc['w_idx'][:, :1],
            topk=dc['topk'], nh=dc['nh'], nIh=dc['nIh'])
    with pytest.raises(ValueError):
        cache.forward_chunk(
            *dc['P'], dc['Ba'], dc['Bb'], dc['B_idx'],
            dc['q'], dc['q_idx'], dc['w_idx'][:, :1],
            topk=dc['topk'], nh=dc['nh'], nIh=dc['nIh'])


def test_chunk_window_keys_torchscript_parity():
    """The scripted window gather is bit-identical to the eager body.

    The continuation-chunk fast path calls ``_chunk_window_keys`` for every
    CSA/HCA layer; compiling the pure gather removes Python control flow
    without changing the gather geometry.  Values, validity masks, and the
    gradient w.r.t. the key tensor must match exactly; the eager fallback
    keeps working on a build where ``torch.jit.script`` is unavailable.
    """
    from ops_decoding_cache import (_chunk_window_keys,
                                    _chunk_window_keys_eager,
                                    _get_scripted_chunk_window_keys)

    cases = ((10, 4, 5), (6, 0, 0), (2, 0, 3), (5, 5, 4),
             (0, 0, 1), (8, 2, 9))
    for total, n_pre, win in cases:
        torch.manual_seed(1000 + total * 100 + n_pre * 10 + win)
        keys = torch.randn(2, total, 3)
        want, want_valid = _chunk_window_keys_eager(keys, n_pre, win)
        got, got_valid = _chunk_window_keys(keys, n_pre, win)
        assert torch.equal(got, want), (total, n_pre, win)
        assert torch.equal(got_valid, want_valid), (total, n_pre, win)

    with pytest.raises(ValueError):
        _chunk_window_keys(torch.zeros(2, 4), 0, 2)
    with pytest.raises(ValueError):
        _chunk_window_keys(torch.zeros(2, 4, 3), 0, -1)
    with pytest.raises(ValueError):
        _chunk_window_keys(torch.zeros(2, 4, 3), 5, 2)

    scripted = _get_scripted_chunk_window_keys()
    if scripted is None:
        pytest.skip('torch.jit.script unavailable; eager fallback is pinned')

    keys = torch.randn(2, 9, 3, requires_grad=True)
    got, got_valid = scripted(keys, 3, 4)
    got.sum().backward()
    scripted_grad = keys.grad.clone()
    keys.grad = None
    want, want_valid = _chunk_window_keys_eager(keys, 3, 4)
    want.sum().backward()
    assert torch.equal(got, want)
    assert torch.equal(got_valid, want_valid)
    assert torch.equal(scripted_grad, keys.grad)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
