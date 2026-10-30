# Block tiling of CSA/HCA KV compression

*How a row-wise compression kernel gets a bounded-memory, faster long-context
forward without changing its math.*

## The problem

Both compression primitives materialize one softmax buffer for the whole
sequence:

| operator | source | logits transient |
|---|---|---|
| `csa_compress_kv` (HCA) | Eq. 22–23 | `[B, T/m, m, c]` |
| `csa_compress_kv_overlapped` (CSA) | Eq. 11–12 | `[B, T/m, 2m, c]` |

At `T = 1M`, `c = 128`, `B = 1`, the overlapped logits buffer alone is
`1M / m · 2m · 128 · 4 B ≈ 1 GiB` in fp32 (plus the same-sized softmax
output). Unlike the indexer score tensor, this is not a quadratic blow-up —
it is linear in `T` — but it is still the largest transient in the CSA/HCA
prefill path, and it is assembled in one allocation with poor cache reuse.

Compression is **row-wise over compressed blocks**. Block `i` reads its own
`m` source tokens and, for the overlapped variant, block `i-1`'s `m` tokens.
No block reads another block's *output*. Tiling the block axis therefore
changes only the size/locality of the transient, not the computation.

## The design

`csa_compress_kv` and `csa_compress_kv_overlapped` accept a keyword-only
`compress_chunk`:

* `0` (default) — automatic heuristic (`_resolve_compress_tile`): keep the
  historical full-buffer path while the materialized logits fit in
  `262,144` elements, otherwise process a tile of 128 blocks (single branch)
  or 64 blocks (overlapped);
* positive — explicit number of compressed blocks per tile;
* negative / non-int — rejected with `ValueError` (same contract as `m`).

`naive_csa` and `naive_hca` forward the knob to their compression calls, so
long-context callers can pin a tile size:

```python
o = naive_csa(..., compress_chunk=256)   # 0 = automatic
o = naive_hca(..., compress_chunk=256)
```

Tiling applies to the `torch.no_grad()` / inference path (the same path that
already avoids the full-size shifted `Zb_prev` / `Cb_prev` pads). The autograd
path keeps the single full-buffer graph so training gradients are untouched.

### Why it stays correct

For every compressed block `i`, the tile that contains `i` computes

```
A = Za[i] + Ba                # [m, c]
B = Zb[i-1] + Bb              # [m, c]   (-inf for i == 0)
S = softmax(cat(A, B), dim=0)
out[i] = (S[:m] * Ca[i]).sum(0) + (S[m:] * Cb[i-1]).sum(0)
```

which is exactly the full-buffer expression restricted to that block. The
b-branch of a tile starting at `i0 > 0` is read from the physical block
`i0 - 1`; the first tile writes the `-inf` row directly, as the untiled
no-pad path already does.

The only observable difference is the CPU softmax kernel's shape-dependent
reduction: fp64 is bit-identical, while fp32/bf16 can differ by a few ULP
(`< 1e-6` in the regression cases). Pinned by
`checks/csa.py::test_compress_chunk_tiling_parity` in `run_correctness.py`
and by `test_ops_csa_compress.py`.

## Measured forward effect (CPU, fp32, this host)

Median of 7 runs; `full` forces one tile (`compress_chunk = n_blocks`),
`auto` uses the heuristic above:

| operator | B | m | c | T | full | auto | speedup |
|---|---|---|---|---|---|---|---|
| overlapped | 1 | 16 | 64 | 65,536 | 29.7 ms | 12.6 ms | 2.36x |
| overlapped | 1 | 16 | 128 | 65,536 | 61.6 ms | 19.1 ms | 3.22x |
| overlapped | 1 | 64 | 64 | 65,536 | 27.5 ms | 9.7 ms | 2.84x |
| overlapped | 4 | 16 | 64 | 16,384 | 30.7 ms | 8.4 ms | 3.65x |
| single (HCA) | 1 | 16 | 64 | 65,536 | 14.6 ms | 4.1 ms | 3.58x |
| single (HCA) | 1 | 64 | 64 | 65,536 | 15.3 ms | 3.9 ms | 3.90x |
| single (HCA) | 1 | 64 | 128 | 65,536 | 26.9 ms | 7.3 ms | 3.70x |

The gain comes from keeping the `[B, tile, 2m, c]` logits / softmax /
weighted-sum working set cache-resident and avoiding a single large
allocation. Small sequences stay on the historical path (the extra Python
iterations would not amortize).

## Related

* `docs/csa-indexer-chunking.md` — the same idea on the indexer's query axis.
* `docs/chunked-prefill-cache.md` — the incremental decode/prefill cache,
  which already compresses one block at a time via
  `_csa_compress_kv_overlapped_single` / `_csa_compress_kv_overlapped_prev`.
