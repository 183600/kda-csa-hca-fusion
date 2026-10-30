# Causal key-range tiling for HCA dense attention

*How the long-context HCA reference path stops paying for compressed blocks
that the causal mask would discard, and why the sink-only tile is exact.*

## The problem

`naive_hca` attends densely to every compressed KV entry. With the paper's
strictly-preceding block mask (Eq. 16), query `t` sees block `b` iff
`b < t // m2`. The untiled forward scores the full `[T, n_blocks]`
rectangle and then writes `-inf` into the future-block triangle.

The tiled long-context path inherited that rectangle: for a query tile
`[lo, hi)` it built `[B, nh, hi-lo, n_blocks]` scores, even though the last
query in the tile can only see `(hi - 1) // m2` blocks. The analytic cost
model already charges the causal triangle
(`run_kv_cache.causal_block_entries(T, m2)`), so the implementation was
doing up to ~2x more dense score/value work than its own ledger.

## The design

For each query tile `[lo, hi)` the tiled path now computes

```
k_max = (hi - 1) // m2
scores = q[:, lo:hi] @ C_comp_n[:, :k_max].T
```

and masks only within that prefix. The last tile can still use all
`n_blocks` entries, so the memory bound is unchanged. For long sequences
the total dense score/value entries become

```
sum_tiles (hi - lo) * ((hi - 1) // m2)  ~=  T * n_blocks / 2
```

instead of `T * n_blocks` (the small excess over the exact triangle is the
per-tile boundary slack).

### Sink-only tiles

When `k_max == 0` and `sliding_window == 0`, the tile has no softmax entry
at all. The attention sink is a virtual denominator-only entry with zero
value, so the output is exactly zero. The tiled path returns a zero tensor
directly; the untiled path reaches the same value through its empty-column
branch. With a sliding window the window entries are still present, so the
joint softmax runs normally and the sink shares its denominator as before.

## Why it stays correct

Removing columns that are `-inf` for every row of the tile does not change
the per-row softmax:

* the row maximum is unchanged (the removed entries cannot be the maximum);
* the removed `exp(-inf)` terms contribute exactly zero to the denominator;
* the surviving entries keep their original order, so the reduction is the
  same prefix reduction.

The value contraction uses the same prefix of `C_comp_n`. Gradients to the
discarded future blocks are zero, which is the same gradient the masked
untiled path produces.

Pinned by `checks/hca.py::test_hca_causal_score_tiling` in
`run_correctness.py`:

* forward parity with the untiled path for `T` around the `m2` boundary
  (`1, 8, 16, 17, 33, 96`), with and without a sliding window and partial
  RoPE;
* exact-zero output for sink-only tiles (`T <= m2`, `sliding_window=0`);
* backward parity for input, sink logits and every projection parameter.

The original untiled/tiled parity test
(`test_hca_score_tiling_parity`) continues to pass unchanged.

## Measured effect (CPU, fp32, this host)

Median of 3 runs, `_HCA_SCORE_TILE_ELEMS = 1` to force the tiled path,
`B=1`, `nh=4`, `c=64`, `win=64`:

| T | m2 | old tiled | causal tiled | speedup |
|---|---|---|---|---|
| 8,192 | 32 | 479 ms | 438 ms | 1.09x |
| 16,384 | 64 | 957 ms | 868 ms | 1.10x |
| 16,384 | 16 | 1,407 ms | 1,100 ms | 1.28x |

The dense score tensor shrinks by ~47% at these shapes (0.53x entries); the
wall-clock gain is smaller because the sliding-window branch, softmax and
Python tile loop are unchanged. The benefit grows with `n_blocks`, i.e. at
longer context or smaller `m2`, which is exactly where HCA's dense branch
is expensive.

## Related

* `docs/compress-block-tiling.md` — block-axis tiling of the compression
  kernel itself.
* `docs/flops-accounting.md` — the causal block-entry ledger this path now
  matches.
