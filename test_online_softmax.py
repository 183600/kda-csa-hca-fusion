"""Regression tests for chunked ``online_softmax`` mask handling.

What is pinned here
-------------------
``online_softmax.online_softmax_attention`` shards the KEY axis into
``key_chunk``-sized blocks.  Its mask contract (``online_softmax_state``
docstring) accepts any boolean mask broadcastable to ``[B, H, T, Nc]``,
which includes masks whose last axis is a broadcast singleton (size 1 —
a per-query keep / drop-all-everything flag).

Slicing that singleton axis at a non-zero chunk offset yields an EMPTY
block, which ``masked_fill`` / ``amax`` cannot reduce:

* before the fix, ``mask=[T, 1]`` + ``key_chunk=2`` died with
  ``IndexError: amax(): Expected reduction dim -1 to have non-zero size``
  inside the second chunk, while ``key_chunk=0`` worked (Python slicing
  clamps at the start of the axis).

The chunked result must therefore equal the result of the equivalent
materialized mask, up to the summation-order rounding of the online
merge.

Run via pytest::

    pytest -q test_online_softmax.py

Or directly::

    python3 test_online_softmax.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from online_softmax import (  # noqa: E402
    online_softmax_attention, online_softmax_state)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _batch(dtype=torch.float32):
    torch.manual_seed(7)
    q = torch.randn(2, 3, 6, 8, dtype=dtype)
    k = torch.randn(2, 3, 6, 8, dtype=dtype)
    v = torch.randn(2, 3, 6, 8, dtype=dtype)
    return q, k, v


def test_singleton_key_axis_mask_survives_key_chunking():
    """A [T, 1] mask must not be sliced into an empty chunk block."""
    q, k, v = _batch()
    for label, mask in (
            ('all-keep', torch.ones(6, 1, dtype=torch.bool)),
            ('per-query keep/drop',
             torch.tensor([[True], [True], [False],
                           [True], [True], [False]]))):
        chunked = online_softmax_attention(q, k, v, mask=mask, key_chunk=2)
        whole = online_softmax_attention(q, k, v, mask=mask, key_chunk=0)
        _ok(f'{label}: key_chunk=2 runs without crashing',
            chunked.shape == q.shape, f'{tuple(chunked.shape)}')
        _ok(f'{label}: chunked == unchunked (fp32 merge rounding)',
            bool((chunked - whole).abs().max() < 1e-5),
            f'max|d|={(chunked - whole).abs().max().item():.2e}')
        # The broadcast reading of the singleton mask is the reference.
        ref = online_softmax_attention(
            q, k, v, mask=mask.expand(6, 6), key_chunk=0)
        _ok(f'{label}: matches the materialized-mask reference',
            bool((whole - ref).abs().max() == 0.0),
            f'max|d|={(whole - ref).abs().max().item():.2e}')


def test_materialized_masks_are_still_sliced_per_chunk():
    """A full [T, N] / [B, H, T, N] mask keeps chunk == unchunked."""
    q, k, v = _batch()
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    b4 = torch.ones(2, 3, 6, 6, dtype=torch.bool)
    b4[..., 4:] = False
    for label, mask in (('2-D causal', causal), ('4-D', b4)):
        chunked = online_softmax_attention(q, k, v, mask=mask, key_chunk=2)
        whole = online_softmax_attention(q, k, v, mask=mask, key_chunk=0)
        _ok(f'{label}: chunked == unchunked (fp32 merge rounding)',
            bool((chunked - whole).abs().max() < 1e-5),
            f'max|d|={(chunked - whole).abs().max().item():.2e}')


def test_masked_reference_matches_dense_reference():
    """Chunked causal output equals the dense masked softmax result."""
    q, k, v = _batch()
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    out = online_softmax_attention(q, k, v, mask=mask, key_chunk=2)
    scores = (q @ k.transpose(-1, -2)) * (8 ** -0.5)
    probs = torch.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
    ref = probs @ v
    _ok('chunked causal == dense masked softmax',
        bool((out - ref).abs().max() < 1e-5),
        f'max|d|={(out - ref).abs().max().item():.2e}')


def test_positive_infinity_scores_use_the_softmax_limit():
    """``+inf`` scores must not turn the online state/merge into NaN."""
    B, H, T, N, D = 1, 2, 4, 4, 4
    torch.manual_seed(11)
    q = torch.zeros(B, H, T, D)
    k = torch.zeros(B, H, N, D)
    v = torch.randn(B, H, N, D)
    # Query 1 sees key 1 as +inf; the other rows are ordinary finite rows.
    q[:, :, 1, :] = 1e20
    k[:, :, 1, :] = 1e20
    sink = torch.zeros(H)
    ref = online_softmax_attention(q, k, v, sink_logits=sink, key_chunk=0)
    _ok('single +inf row selects that value',
        bool((ref[:, :, 1] - v[:, :, 1]).abs().max() == 0.0),
        f'max|d|={(ref[:, :, 1] - v[:, :, 1]).abs().max().item():.2e}')
    _ok('output is finite', bool(torch.isfinite(ref).all()))
    for key_chunk in (1, 2, 3):
        out = online_softmax_attention(q, k, v, sink_logits=sink,
                                       key_chunk=key_chunk)
        _ok(f'key_chunk={key_chunk} finite rows match (merge rounding)',
            bool((out - ref).abs().max() < 1e-5),
            f'max|d|={(out - ref).abs().max().item():.2e}')
        _ok(f'key_chunk={key_chunk} keeps the +inf row exact',
            bool((out[:, :, 1] - v[:, :, 1]).abs().max() == 0.0),
            f'max|d|={(out[:, :, 1] - v[:, :, 1]).abs().max().item():.2e}')

    # A huge but finite sink must still lose to the +inf score.
    out = online_softmax_attention(q, k, v,
                                   sink_logits=torch.full((H,), 1e30),
                                   key_chunk=2)
    _ok('sink cannot take mass from a +inf row',
        bool((out[:, :, 1] - v[:, :, 1]).abs().max() == 0.0),
        f'max|d|={(out[:, :, 1] - v[:, :, 1]).abs().max().item():.2e}')

    # Two +inf keys in different chunks: merging must add both states.
    k2 = k.clone()
    k2[:, :, 2, :] = 1e20
    expected = (v[:, :, 1] + v[:, :, 2]) / 2
    for key_chunk in (0, 1, 2, 3):
        out = online_softmax_attention(q, k2, v, sink_logits=sink,
                                       key_chunk=key_chunk)
        _ok(f'key_chunk={key_chunk} averages two +inf keys',
            bool((out[:, :, 1] - expected).abs().max() < 1e-6),
            f'max|d|={(out[:, :, 1] - expected).abs().max().item():.2e}')

    # Gradients through a batch containing a +inf row must stay finite.
    qg = q.clone().requires_grad_()
    kg = k2.clone().requires_grad_()
    vg = v.clone().requires_grad_()
    loss = online_softmax_attention(qg, kg, vg, sink_logits=sink,
                                    key_chunk=1).sum()
    loss.backward()
    grads = {'q': qg.grad, 'k': kg.grad, 'v': vg.grad}
    for name, grad in grads.items():
        _ok(f'+inf row: grad_{name} finite',
            bool(torch.isfinite(grad).all()),
            f'max|g|={grad.abs().max().item():.2e}')


def test_all_positive_infinity_scores_average_all_values():
    """An all-``+inf`` row is uniform over every key, not NaN."""
    B, H, T, N, D = 1, 1, 2, 3, 4
    torch.manual_seed(13)
    q = torch.zeros(B, H, T, D)
    k = torch.zeros(B, H, N, D)
    v = torch.randn(B, H, N, D)
    q[:, :, 1, :] = 1e20
    k[:, :, :, :] = 1e20
    expected = v.mean(dim=2)
    for key_chunk in (0, 1, 2):
        out = online_softmax_attention(q, k, v, key_chunk=key_chunk)
        _ok(f'key_chunk={key_chunk} uniforms the all-+inf row',
            bool((out[:, :, 1] - expected).abs().max() < 1e-6),
            f'max|d|={(out[:, :, 1] - expected).abs().max().item():.2e}')



def test_per_batch_sink_broadcasts_over_heads_when_b_equals_h():
    """A leading batch axis must not be mistaken for the head axis.

    ``[B,1,1]`` (one sink per sequence) and ``[B,1]`` are valid
    broadcastable forms. When B == H the old shape heuristic read them as
    ``[H,1,1]`` / ``[H,1]`` (one sink per head), silently swapping head
    logits between the two heads.
    """
    B = H = T = N = D = 2
    torch.manual_seed(29)
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, N, D, dtype=torch.float64)
    v = torch.randn(B, H, N, D, dtype=torch.float64)
    per_batch = torch.tensor([[[1.0]], [[-1.0]]], dtype=torch.float64)
    intended = per_batch.unsqueeze(1).expand(B, H, 1, 1)
    want = online_softmax_attention(
        q, k, v, sink_logits=intended)
    for sink, label in (
            (per_batch, '[B,1,1]'),
            (per_batch[:, :, 0], '[B,1]')):
        got = online_softmax_attention(q, k, v, sink_logits=sink)
        _ok(f'per-batch sink {label} broadcasts over heads',
            bool((got - want).abs().max() < 1e-12),
            f'max|d|={(got - want).abs().max().item():.2e}')


def test_shared_kv_mqa_gqa_and_query_tiling():
    """MQA/GQA + query tiles must equal the dense shared-KV reference."""
    torch.manual_seed(21)
    B, Hq, Hkv, T, N, D = 2, 4, 2, 7, 9, 8
    scale = D ** -0.5
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, N, D)
    v = torch.randn(B, Hkv, N, D)
    mask = torch.tril(torch.ones(T, N, dtype=torch.bool))
    sink = torch.randn(Hq)
    group = Hq // Hkv
    scores = torch.einsum(
        'b h t d, b h n d -> b h t n', q,
        k.repeat_interleave(group, dim=1)) * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    row_max = scores.amax(dim=-1, keepdim=True)
    e = (scores - row_max).exp()
    den = e.sum(dim=-1, keepdim=True) + (
        sink.view(1, Hq, 1, 1) - row_max).exp()
    ref = torch.einsum(
        'b h t n, b h n d -> b h t d', e / den,
        v.repeat_interleave(group, dim=1))
    for query_chunk, key_chunk in ((0, 0), (2, 3), (T, N)):
        out = online_softmax_attention(
            q, k, v, key_chunk=key_chunk, query_chunk=query_chunk,
            scale=scale, mask=mask, sink_logits=sink)
        _ok(f'qc={query_chunk} kc={key_chunk}: MQA/GQA matches dense',
            bool((out - ref).abs().max() < 1e-5),
            f'max|d|={(out - ref).abs().max().item():.2e}')
    try:
        online_softmax_attention(
            torch.randn(1, 3, 4, D), torch.randn(1, 2, 5, D),
            torch.randn(1, 2, 5, D))
    except ValueError:
        _ok('non-divisible Hq/Hkv is rejected', True, '')
    else:
        _ok('non-divisible Hq/Hkv is rejected', False, 'no ValueError')
    try:
        online_softmax_attention(q, k, v, query_chunk=-1)
    except ValueError:
        _ok('negative query_chunk is rejected', True, '')
    else:
        _ok('negative query_chunk is rejected', False, 'no ValueError')


def test_empty_key_block_is_rejected_with_a_clear_error():
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 0, 4)
    v = torch.randn(1, 1, 0, 4)
    try:
        online_softmax_state(q, k, v)
    except ValueError as exc:
        _ok('empty key block raises ValueError',
            'empty' in str(exc).lower(), str(exc))
    else:
        _ok('empty key block raises ValueError', False, 'no ValueError')


_TEST_FUNCS = [
    test_empty_key_block_is_rejected_with_a_clear_error,
    test_singleton_key_axis_mask_survives_key_chunking,
    test_materialized_masks_are_still_sliced_per_chunk,
    test_masked_reference_matches_dense_reference,
    test_positive_infinity_scores_use_the_softmax_limit,
    test_all_positive_infinity_scores_average_all_values,
    test_per_batch_sink_broadcasts_over_heads_when_b_equals_h,
    test_shared_kv_mqa_gqa_and_query_tiling,
]


def main() -> int:
    print('=' * 70)
    print('online_softmax key-chunk mask regression tests')
    print('=' * 70)
    ok = True
    for fn in _TEST_FUNCS:
        try:
            fn()
        except AssertionError as e:
            print(f'  [FAIL] {fn.__name__}: {e}')
            ok = False
    print('-' * 70)
    print('all tests passed' if ok else 'FAILURES')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
