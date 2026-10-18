# Query-axis chunking of the CSA lightning indexer

*How a long-context prefill bounds the biggest intermediate of Compressed
Sparse Attention without changing a single selection.*

## The problem

CSA first compresses every `m` tokens into one KV entry and then runs the
lightning indexer (DeepSeek-V4 §2.3.1, Eq. 13–17) to choose the top-`k`
compressed blocks per query. The indexer is cheap in FLOPs (`O(T · n_blocks ·
nIh · c_I)` with `n_blocks = T/m`) but the reference implementation scored the
whole sequence at once, materializing a per-head similarity tensor of shape

```
[B, nIh, T, n_blocks]
```

before reducing over the indexer heads and applying softmax. At `T = 1M`,
`m = 16`, `nIh = 2` that intermediate holds `1.25e11` elements — ~500 GB in
fp32 — even though the final selection per query is only `k` integers.

This is the classic reason linear-attention kernels are *chunked* over the
sequence axis (FLA's chunked KDA/DPLR kernels do the same thing): the
algorithm is row-wise in the query, so the sequence axis is embarrassingly
parallel and tiling it changes memory, not math.

## The design

`ops_csa.csa_lightning_indexer` gained a keyword-only `query_chunk: int`
parameter (default `0` = disabled, i.e. the historical full-sequence score
tensor):

* `query_chunk = 0` or `query_chunk >= T`: one call to the shared
  `_lightning_indexer_chunk` helper over the full sequence (the historical
  code path, now shared);
* `0 < query_chunk < T`: the query axis is cut into tiles of at most
  `query_chunk` rows; each tile computes its own scores → logits → causal
  mask → canonical top-`k` → NaN-safe softmax, and the per-tile results are
  concatenated along the query axis.

`naive_csa` forwards the tile size as `indexer_chunk`, and `HybridConfig`
exposes it as `csa_indexer_chunk` (default `0`, so every existing experiment
configuration is untouched).

Peak intermediate memory (per-head scores):

```
untiled: B · nIh · T        · n_blocks
tiled:   B · nIh · query_chunk · n_blocks
```

## Why the selection is provably unchanged

Every step between the projections and the selected indices is row-wise along
the query axis:

| step | axis touched |
|---|---|
| `q_idx · K_IComp` (einsum `bthd,bnd->bhtn`) | reduces over `d`, keeps rows |
| head mixing by `w_idx` | reduces over `nIh`, keeps rows |
| causal block mask | row-wise mask |
| `_canonical_topk_indices` | row-wise, shape-independent tie-break |
| `_nan_safe_softmax` | row-wise softmax |

The tie-break deserves the emphasis: the indexer's deterministic rule
(equal scores → ascending block index, see `test_csa_tiebreak.py`) is a pure
function of each query's score *row*, so it cannot depend on how many rows
were scored together. That is exactly why the incremental-decoding path
already agrees with batched prefill, and the same argument makes tiling
transparent.

The only differences the tests can observe are fp round-off from batched-GEMM
reduction order in the einsums (indices: bit-equal in the regression suite;
soft weights: ≤ ~6e-8 on the test data).

## Usage

```python
from ops_csa import naive_csa

# Long-context prefill with a bounded indexer score tensor:
o = naive_csa(H, ..., indexer_chunk=4096)

# Or through the hybrid config:
from ops_fused import HybridConfig, HybridAttentionStack
cfg = HybridConfig(csa_indexer_chunk=4096)
```

Rules of thumb:

* inference (`use_ste=False` / `torch.no_grad()`): the indexer only returns
  indices, so a tile of a few thousand queries keeps the transient score
  tensor in the tens of MB at million-token contexts;
* training: the STE path additionally needs the full `[B, T, n_blocks]` soft
  distribution (it *is* returned), but the per-head score tensor — `nIh×`
  larger — is still bounded by the tile;
* `query_chunk >= T` is a no-op; negative values are rejected with
  `ValueError` rather than silently running the untiled path.

## Verification

`test_csa_chunk_fla_adapter.py` pins, on CPU:

* bit-equal indices (including exact-tie rows, `topk` > `n_blocks`, and the
  `w_idx=None` / `topk=0` branches) across tile sizes `{1, 5, T±1}`;
* equality of the soft weights within ~1e-6 and gradient parity
  of `q_idx` / `k_idx` / `w_idx` under tiling;
* the actual memory bound by spying on `torch.einsum` outputs: the peak
  score tensor is exactly `B·nIh·chunk·n_blocks` when tiled and
  `B·nIh·T·n_blocks` when not;
* `naive_csa(indexer_chunk=...)` end-to-end output identical (atol = 0).

## Limitations

* The returned `soft_weights` tensor is still `[B, T, n_blocks]` when the STE
  is active — tiling bounds the *transient* per-head scores, not the STE
  distribution the backward pass consumes in full.
* The untiled `topk=0` path also materializes the full logits; `topk=0` is an
  inference-only degenerate configuration, so it is deliberately not tiled.
* The core attention over the selected blocks is already `O(T · topk · c)`
  and is not affected.
