# Sliding-window auto-chunking: the budget must hold for every window

`ops_csa._sliding_window_attention`, `_sliding_window_scores` (when
`return_windows=False`) and `_sliding_window_output` evaluate the window
branch in bounded query tiles of `chunk_t` rows. The transient window slab
is `[B, chunk_t, win, c]`, so the tile size is chosen to keep
`chunk_t * win * c <= CHUNKED_SW_THRESHOLD` (8M elements by default).

## The bug

The old auto rule was

```python
target = max(64, CHUNKED_SW_THRESHOLD // max(1, win * c))
chunk_t = min(T, ((target + 63) // 64) * 64)
```

Two operations break the stated bound:

1. the `max(64, ...)` floor, and
2. rounding the target **up** to a multiple of 64.

For `win=4096`, `c=64` the budget allows `8M / 262144 = 30` rows, but the
rule selected 64 rows: a 16.8M-element slab, 2.1x the budget. The same
mistake affected the joint-softmax path (`_sw_query_chunk`), where the
window scores share one denominator with the compressed entries and the
attention sink.

## The fix

`_auto_sw_chunk_t(T, win, c)` now:

* rounds **down** to a 64-row multiple when the budget allows at least 64
  rows (preserves the aligned-GEMM shape),
* otherwise keeps the exact target (minimum 1 row, capped at `T`).

A single row larger than the whole budget (`win * c > CHUNKED_SW_THRESHOLD`)
uses 1 row; a row cannot be split further. The helper is shared by
`_sliding_window_attention` and `_sw_query_chunk`, so the standalone helper
and the joint-softmax path cannot drift apart.

`_sliding_window_attention` also stops densifying the overlapping window
view when the slab exceeds `_SW_TILE_DENSIFY_MAX`; the extra dense copy
would have doubled the transient the budget is meant to bound. Small slabs
still use the dense layout where it measures faster.

Tiling is strictly row-wise: each query row's dot products, causal mask and
softmax are unchanged. The only difference versus the full path is the
per-row reduction order, pinned to 1e-6 in the regression (the mask keeps
its exact `-inf` pattern).

## Regression

`checks/csa.py::test_sliding_window_auto_chunk_budget` (registered in
`run_correctness.py`):

* sweeps `T x win x c` and asserts every auto tile satisfies the element
  budget, with the documented 1-row fallback for oversized single rows;
* lowers the threshold temporarily to force a real multi-tile run at
  `T=300, win=128, c=16`, then compares the tiled scores against the full
  unfold path (including the `-inf` padding pattern);
* runs the same comparison through a **joint softmax with a
  denominator-only attention sink**, checking the window value contraction
  as well as the scores.

The attention sink is a virtual entry with zero value, so it changes only
the shared denominator; the test therefore covers the exact interaction that
motivated keeping one joint softmax instead of the standalone chunked helper.
