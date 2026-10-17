"""Regression tests for autograd through the incremental decoding caches.

What is pinned here
-------------------
A single backward pass may span SEVERAL ``forward_step`` calls of
``CSADecodingCache`` / ``HCADecodingCache`` (e.g. a loss over the whole
decoded continuation, or any teacher-forced decode under autograd). The
window-scores einsums save the window keys for the query gradient AND for
the gradient w.r.t. the softmax weights (which can require grad on their
own, e.g. through a learnable attention sink), and
``_SlidingWindowRingBuffer.get()`` returns a VIEW into the ring buffer; a
later ``append_step`` overwrites that buffer in place, which would
invalidate the saved view. Both caches materialize a stable copy of the
window keys before the einsums, so these tests keep the multi-step
backward contract green.

Run via pytest::

    pytest -q test_decoding_cache_autograd.py

Or directly::

    python3 test_decoding_cache_autograd.py
"""
from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ops_decoding_cache import (  # noqa: E402
    CSADecodingCache, HCADecodingCache,
)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _csa_backward(device='cpu', rope_dim=None, sink=None, q_grad=True):
    B, d, c, c_I, m, nh, nIh, dc = 1, 32, 8, 4, 4, 2, 1, 8
    win, topk, T = 4, 100, 6
    torch.manual_seed(11)
    H = torch.randn(B, T, d, device=device) * 0.1

    W_aKV = (torch.randn(c, d, device=device) * 0.1).requires_grad_(True)
    W_DQ = (torch.randn(dc, d, device=device) * 0.1).requires_grad_(True)
    W_UQ = (torch.randn(c * nh, dc, device=device) * 0.1).requires_grad_(True)
    Ba = (torch.randn(m, c, device=device) * 0.02).requires_grad_(True)
    sink_par = None if sink is None else torch.full((nh,), sink, device=device)
    if not q_grad:
        # The only grad-carrying operand is the sink: the window-scores
        # einsum still puts ``p_w`` on the graph, so the saved ring-buffer
        # view must have been materialized independently of q.
        assert sink_par is not None
        sink_par.requires_grad_(True)

    cache = CSADecodingCache(
        B, c, c_I, m, win, device, torch.float32, rope_dim=rope_dim)
    outs = []
    for t in range(T):
        H_t = H[:, t:t + 1]
        # Keys enter the cache detached (the decode contract: cache state
        # must not retain the training graph / cross-step history).
        Ca = F.linear(H_t, W_aKV).detach()
        cache.append_step(
            Ca, Ca, Ca, Ca,
            torch.zeros(B, 1, c_I, device=device),
            torch.zeros(B, 1, c_I, device=device),
            Ba.detach(), Ba.detach(), torch.zeros(m, c_I, device=device))
        with torch.no_grad() if not q_grad else contextlib.nullcontext():
            q = F.linear(F.linear(H_t, W_DQ), W_UQ).view(B, 1, nh, c)
        q_n = F.normalize(q, dim=-1)
        outs.append(cache.forward_step(
            q_n, torch.zeros(B, 1, nIh, c_I, device=device),
            torch.zeros(B, 1, nIh, device=device),
            topk=topk, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=sink_par, use_ste=False))
    loss = torch.cat(outs, dim=1).sum()
    loss.backward()  # multi-step backward must succeed
    return W_DQ.grad, W_UQ.grad, (
        None if sink_par is None else sink_par.grad)


def _hca_backward(device='cpu', rope_dim=None, sink=None, q_grad=True):
    B, d, c, m2, nh, dc = 1, 32, 8, 4, 2, 8
    win, T = 4, 6
    torch.manual_seed(11)
    H = torch.randn(B, T, d, device=device) * 0.1

    W_KV = (torch.randn(c, d, device=device) * 0.1).requires_grad_(True)
    W_DQ = (torch.randn(dc, d, device=device) * 0.1).requires_grad_(True)
    W_UQ = (torch.randn(c * nh, dc, device=device) * 0.1).requires_grad_(True)
    B_pos = (torch.randn(m2, c, device=device) * 0.02).requires_grad_(True)
    sink_par = None if sink is None else torch.full((nh,), sink, device=device)
    if not q_grad:
        # Sink-only gradient: see the CSA helper above.
        assert sink_par is not None
        sink_par.requires_grad_(True)

    cache = HCADecodingCache(
        B, c, m2, win, device, torch.float32, rope_dim=rope_dim)
    outs = []
    for t in range(T):
        H_t = H[:, t:t + 1]
        C = F.linear(H_t, W_KV).detach()
        cache.append_step(C, C, B_pos.detach())
        with torch.no_grad() if not q_grad else contextlib.nullcontext():
            q = F.linear(F.linear(H_t, W_DQ), W_UQ).view(B, 1, nh, c)
        q_n = F.normalize(q, dim=-1)
        outs.append(cache.forward_step(
            q_n, nh=nh, scale=1.0, sink_logits=sink_par))
    loss = torch.cat(outs, dim=1).sum()
    loss.backward()  # multi-step backward must succeed
    return W_DQ.grad, W_UQ.grad, (
        None if sink_par is None else sink_par.grad)


def test_csa_cache_backward_spans_decode_steps(device='cpu'):
    """CSA: one backward over many decode steps must not raise."""
    for rope_dim in (None, 4):
        for sink in (None, 0.5):
            g_dq, g_uq, _ = _csa_backward(device, rope_dim, sink)
            _ok(
                f'csa_cache_backward rope={rope_dim} sink={sink}',
                g_dq is not None and g_uq is not None
                and torch.isfinite(g_dq).all() and torch.isfinite(g_uq).all()
                and g_uq.abs().sum() > 0,
                f'W_DQ.grad finite={g_dq is not None}, '
                f'W_UQ.grad nonzero={float(g_uq.abs().sum()) > 0}')


def test_hca_cache_backward_spans_decode_steps(device='cpu'):
    """HCA: one backward over many decode steps must not raise."""
    for rope_dim in (None, 4):
        for sink in (None, 0.5):
            g_dq, g_uq, _ = _hca_backward(device, rope_dim, sink)
            _ok(
                f'hca_cache_backward rope={rope_dim} sink={sink}',
                g_dq is not None and g_uq is not None
                and torch.isfinite(g_dq).all() and torch.isfinite(g_uq).all()
                and g_uq.abs().sum() > 0,
                f'W_DQ.grad finite={g_dq is not None}, '
                f'W_UQ.grad nonzero={float(g_uq.abs().sum()) > 0}')


def test_csa_cache_backward_sink_only_spans_decode_steps(device='cpu'):
    """CSA: grad flows only through the sink; the window-scores einsum still
    saves the ring-buffer view, so the copy must not be tied to q.requires_grad."""
    g_dq, g_uq, g_sink = _csa_backward(device, rope_dim=None, sink=0.5,
                                       q_grad=False)
    _ok(
        'csa_cache_backward_sink_only',
        g_sink is not None and torch.isfinite(g_sink).all()
        and g_sink.abs().sum() > 0 and g_dq is None and g_uq is None,
        f'sink.grad nonzero={g_sink is not None and float(g_sink.abs().sum()) > 0}, '
        f'q grads absent={g_dq is None and g_uq is None}')


def test_hca_cache_backward_sink_only_spans_decode_steps(device='cpu'):
    """HCA: same sink-only multi-step backward contract as CSA."""
    g_dq, g_uq, g_sink = _hca_backward(device, rope_dim=None, sink=0.5,
                                       q_grad=False)
    _ok(
        'hca_cache_backward_sink_only',
        g_sink is not None and torch.isfinite(g_sink).all()
        and g_sink.abs().sum() > 0 and g_dq is None and g_uq is None,
        f'sink.grad nonzero={g_sink is not None and float(g_sink.abs().sum()) > 0}, '
        f'q grads absent={g_dq is None and g_uq is None}')


def test_csa_cache_backward_normalize_qk_off(device='cpu'):
    """CSA: normalize_qk=False must not save live compressed-cache views.

    Without normalization the compressed rows and indexer keys are read
    straight from the cache's backing buffers, so a later ``append_step``
    can overwrite storage saved for backward. Both STE modes must span
    several block completions and still backpropagate.
    """
    B, d, c, c_I, m, nh, nIh, dc = 1, 16, 8, 4, 4, 2, 1, 8
    win, topk, T = 3, 2, 20
    torch.manual_seed(13)
    H = torch.randn(B, T, d, device=device) * 0.1
    W_aKV = torch.randn(c, d, device=device) * 0.1
    W_KV_idx = torch.randn(c_I, d, device=device) * 0.1
    W_Z_idx = torch.randn(c_I, d, device=device) * 0.1
    Ba = torch.randn(m, c, device=device) * 0.02
    Bb = torch.randn(m, c, device=device) * 0.02
    B_idx = torch.randn(m, c_I, device=device) * 0.02
    for ste_mode in ('topk_columns', 'full_softmax'):
        torch.manual_seed(17)
        W_DQ = (torch.randn(dc, d, device=device) * 0.1).requires_grad_(True)
        W_UQ = (torch.randn(c * nh, dc, device=device)
                * 0.1).requires_grad_(True)
        W_IUQ = (torch.randn(c_I * nIh, d, device=device)
                 * 0.1).requires_grad_(True)
        W_w = (torch.randn(nIh, d, device=device)
               * 0.1).requires_grad_(True)
        cache = CSADecodingCache(
            B, c, c_I, m, win, device, torch.float32)
        outs = []
        for t in range(T):
            H_t = H[:, t:t + 1]
            Ca = F.linear(H_t, W_aKV).detach()
            K_i = F.linear(H_t, W_KV_idx).detach()
            Z_i = F.linear(H_t, W_Z_idx).detach()
            cache.append_step(
                Ca, Ca, Ca, Ca, K_i, Z_i,
                Ba.detach(), Bb.detach(), B_idx.detach())
            q = F.linear(F.linear(H_t, W_DQ), W_UQ).view(B, 1, nh, c)
            q_idx = F.linear(H_t, W_IUQ).view(B, 1, nIh, c_I)
            w_idx = F.linear(H_t, W_w).view(B, 1, nIh)
            outs.append(cache.forward_step(
                q, q_idx, w_idx, topk=topk, nh=nh, nIh=nIh,
                scale=None, sink_logits=None, use_ste=True,
                ste_mode=ste_mode, normalize_qk=False))
        torch.cat(outs, dim=1).sum().backward()
        grads = [W_DQ.grad, W_UQ.grad, W_IUQ.grad, W_w.grad]
        _ok(
            f'csa_cache_backward_normalize_qk_off ste_mode={ste_mode}',
            all(g is not None and torch.isfinite(g).all() for g in grads),
            f'grads present={[g is not None for g in grads]}')


def test_hca_cache_backward_normalize_qk_off(device='cpu'):
    """HCA: normalize_qk=False must not save live compressed-cache views."""
    B, d, c, m2, nh, dc = 1, 16, 8, 4, 2, 8
    win, T = 3, 20
    torch.manual_seed(19)
    H = torch.randn(B, T, d, device=device) * 0.1
    W_KV = torch.randn(c, d, device=device) * 0.1
    B_pos = torch.randn(m2, c, device=device) * 0.02
    W_DQ = (torch.randn(dc, d, device=device) * 0.1).requires_grad_(True)
    W_UQ = (torch.randn(c * nh, dc, device=device)
            * 0.1).requires_grad_(True)
    cache = HCADecodingCache(B, c, m2, win, device, torch.float32)
    outs = []
    for t in range(T):
        H_t = H[:, t:t + 1]
        C = F.linear(H_t, W_KV).detach()
        cache.append_step(C, C, B_pos.detach())
        q = F.linear(F.linear(H_t, W_DQ), W_UQ).view(B, 1, nh, c)
        outs.append(cache.forward_step(
            q, nh=nh, scale=None, sink_logits=None, normalize_qk=False))
    torch.cat(outs, dim=1).sum().backward()
    grads = [W_DQ.grad, W_UQ.grad]
    _ok(
        'hca_cache_backward_normalize_qk_off',
        all(g is not None and torch.isfinite(g).all() for g in grads),
        f'grads present={[g is not None for g in grads]}')


if __name__ == '__main__':
    test_csa_cache_backward_spans_decode_steps()
    test_hca_cache_backward_spans_decode_steps()
    test_csa_cache_backward_sink_only_spans_decode_steps()
    test_hca_cache_backward_sink_only_spans_decode_steps()
    test_csa_cache_backward_normalize_qk_off()
    test_hca_cache_backward_normalize_qk_off()
    print('all decoding-cache autograd tests passed')
