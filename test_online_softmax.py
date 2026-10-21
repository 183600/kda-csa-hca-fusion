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

from online_softmax import online_softmax_attention  # noqa: E402


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


_TEST_FUNCS = [
    test_singleton_key_axis_mask_survives_key_chunking,
    test_materialized_masks_are_still_sliced_per_chunk,
    test_masked_reference_matches_dense_reference,
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
