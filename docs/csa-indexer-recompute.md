# Memory-efficient CSA indexer STE (training)

*How long-context CSA training drops the retained `[B, T, n_blocks]` soft
distribution without changing the forward value or the parameter gradients.*

## The problem

The CSA lightning indexer selects top-`k` compressed blocks per query. The
selection is non-differentiable, so training uses a straight-through
estimator (STE): `naive_csa` computes the full softmax distribution over all
compressed blocks and gathers the selected columns.

Query-axis chunking (`csa_indexer_chunk`, see
[`csa-indexer-chunking.md`](csa-indexer-chunking.md)) bounds the *transient*
per-head score tensor, but the STE still returns the full

```
[B, T, n_blocks]
```

soft distribution to the autograd graph. At `T = 1M`, `m = 16` that is
`6.25e10` elements (~250 GB in fp32) retained until backward — the dominant
activation of CSA training, even though only `topk` columns per query carry
gradient into the core attention.

## The design

`naive_csa(..., recompute_indexer=True)` switches the `topk_columns` STE to a
checkpointed formulation:

* the query axis is cut into tiles (auto-sized so one tile's
  `[B, nIh, tile, n_blocks]` logits tensor stays within
  `ops_csa._CSA_INDEXER_STE_TILE_ELEMS`, or an explicit
  `indexer_chunk` when positive);
* each tile runs `_lightning_indexer_chunk(..., return_soft_weights=True)`,
  gathers the selected columns, and masks the `-1` padding slots *inside* a
  non-reentrant `torch.utils.checkpoint.checkpoint` call;
* only the hard indices `[B, T, topk]` and the gathered selected weights
  `[B, T, topk]` are returned to the caller. The per-tile logits / softmax
  intermediates are rebuilt during backward.

Because the gather lives inside the checkpointed function, the full soft
distribution is never a retained activation. The selected weights have the
same mathematical gradient path as the historical gather: only the selected
columns of the full softmax receive gradient.

`HybridConfig` exposes the switch as `csa_recompute_indexer` (default
`False`), forwarded by `CSAHybridLayer` and the tensor-parallel CSA path.

## Usage

```python
from ops_csa import naive_csa

# Long-context CSA training: retain [B, T, topk] instead of [B, T, n_blocks].
o = naive_csa(H, ..., use_ste=True, recompute_indexer=True)

# Or through the hybrid config:
from ops_fused import HybridConfig, HybridAttentionStack
cfg = HybridConfig(csa_recompute_indexer=True)
```

Rules of thumb:

* default `False` preserves the historical graph exactly, so existing
  experiments and checkpoints are untouched;
* enable for long-context training where `n_blocks >> topk`;
* `recompute_indexer` is ignored for `ste_mode='full_softmax'` (that mode
  needs every column's gradient) and when STE is inactive
  (`use_ste=False` / `torch.no_grad()`);
* a positive `csa_indexer_chunk` also sets the recompute tile size; `0`
  uses the automatic budget.

## Verification

`run_correctness.py` pins on CPU:

* `test_csa_indexer_recompute_ste_parity`: forward values and all parameter
  gradients match the historical path in fp64 for both scale policies
  (`normalize_qk=False/True`) and both auto / explicit (`7`) tiling, with
  sink logits, sliding-window scores, and partial RoPE enabled; the flag is
  a no-op for `full_softmax`;
* `test_csa_indexer_recompute_ste_memory`: saved-tensor hooks confirm the
  historical path retains a `[B, T, n_blocks]` tensor, the recompute path
  never does, and the recompute path retains strictly fewer bytes.

## Limitations

* Recompute trades backward FLOPs for activation memory: the indexer logits
  are evaluated twice per training step. Use it when memory, not compute,
  is the binding constraint.
* The checkpointed path targets the `topk_columns` STE. `full_softmax`
  still retains the full distribution by design.
* Batched-GEMM reduction order may differ by tile size; parity is pinned at
  fp64 (`≤1e-12` forward, `≤1e-10` gradient in the regression test).
