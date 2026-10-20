# Compiled chunked KDA (`torch.compile`)

`ops_kda.compiled_chunk_kda` is a cached `torch.compile` wrapper around the
chunkwise-parallel KDA reference (`naive_chunk_kda`). It keeps the exact
per-channel log-decay recurrence and adds an execution-policy knob for the
memory-bounded `recompute_attn` variant.

## Why

The chunked path is the training / prefill workhorse, but its setup is a long
chain of small elementwise kernels plus a batched triangular solve:

1. per-channel decay clamp + within-chunk cumsum,
2. intra-chunk attention matrix `A` and q-weighted `Aqk`,
3. unit-lower-triangular solve, `u = A v`, `w = A (exp(g) k)`,
4. one sequential state carry per chunk.

Inductor can fuse the elementwise part of that chain (the decay map and the
attention-matrix build). On CPU the measured benefit is largest when that
fused elementwise work dominates (e.g. ~2.3x at T=128 in this host's smoke
run); at longer T the batched GEMM / triangular-solve portion dominates and
the gain shrinks toward 1x. The wrapper is mainly a GPU-oriented execution
policy and a way to make `fullgraph=True` auditable, not a claim that CPU
eager is slow. `recompute_attn=True` drops the persistent `Aqk` cache
(`O(B · HV · T · chunk_size)` elements) and rebuilds one
`chunk_size × chunk_size` matrix per chunk; compiling that policy makes the
memory-bounded configuration practical.

## API

```python
import torch
from ops_kda import compiled_chunk_kda

o, S = compiled_chunk_kda(
    q, k, v, g, beta,
    output_final_state=True,
    chunk_size=64,
    # memory policy: rebuild Aqk per chunk instead of caching all chunks
    recompute_attn=False,
    # compile policy
    dynamic=False,
    fullgraph=True,
    mode=None,
)
```

Requirements / contract:

* `q`, `k` must be L2-normalized by the caller (same as every KDA path).
* `g` is the per-channel log-decay gate; `beta` is post-sigmoid.
* Output shape/dtype, final-state dtype, padding, GVA (`HV > H`) and
  `initial_state` semantics are identical to `naive_chunk_kda`.
* `T == 0` and invalid inputs delegate to the eager path/validators, so the
  degenerate case never pays a compile and never hides a shape error.
* The compiled callable is cached per `(shape, dtype, requires_grad,
  chunk_size, recompute_attn, compile options, initial-state signature,
  scale, g_clamp_min)`; cache size is bounded (LRU, 32 entries).

## What changed to make `fullgraph=True` possible

Two Python-level operations in `_chunk_kda_prepare` were untraceable:

1. **Data-dependent decay-range reduction.**
   The old code computed `bool(max(g) - min(g) <= 50)` on the device and
   skipped the clamp when the range was small. This is a device sync and a
   data-dependent branch. The clamp is now applied unconditionally, which is
   value-identical:
   * if the range is `<= 50`, every pairwise difference is `<= 50`, so the
     clamp is an elementwise identity;
   * if any difference exceeds `50` (or is `NaN`/`+inf`), the old code
     clamped every entry anyway.

   `_chunk_kda_prepare` still returns `clamp_g_diff=True` so the eager,
   TorchScript and recompute inner loops share the same explanation.

2. **Storage-pointer ownership test.**
   The old no-grad in-place scale compared
   `q.untyped_storage().data_ptr()` before/after rearrangement. Dynamo cannot
   trace `UntypedStorage.data_ptr`. The code now uses a metadata-only rule:
   a contiguous input rearranges to a view of the caller's storage, so the
   scale stays out-of-place; padding or a non-contiguous input produces an
   owned buffer, so the in-place multiply is safe. Values are unchanged; the
   only cost is one extra temporary for contiguous inputs.

The regression tests in `test_recompute_attn.py` pin both policies under
`fullgraph=True`, including anisotropic per-channel decay and the `>50` clamp
branch.

## Exp 2 benchmark row

`run_benchmark.py` adds `kda_chunk_compiled` next to `kda_chunk`:

* the two rows share the `kda_chunk` generator key, so q/k/v/g/beta are
  identical;
* before timing, output **and** final state are parity-gated against the
  eager chunk path (max-abs `<= 1e-4`); a divergent operator raises instead
  of emitting a fast-but-wrong row;
* the row records `kda_execution`, `kda_compile_fullgraph`,
  `kda_compile_dynamic`, `kda_chunk_size`, `kda_recompute_attn`,
  `kda_chunk_parity_max_abs`, `kda_chunk_state_parity_max_abs` and
  `kda_chunk_parity_ok`;
* `KDA_RECOMPUTE_ATTN=1` switches the compiled row to the memory-bounded
  policy (the eager `kda_chunk` row remains the cached-Aqk reference).

`run_roofline._account_row` prices the new row with the **chunked** KDA
core model (`kda_core_flops(..., chunked=True)`,
`core_bytes('kda', ..., chunked=True)`) at the recorded `chunk_size`, and
the roofline point carries `kda_execution` and the parity residual. The
Exp 2 figure uses a dotted line / distinct marker for the compiled row.

## Reading the numbers

* The comparison isolates execution policy, not algorithm: both rows perform
  the same per-channel-decay chunked recurrence.
* `torch.compile` has a one-time compilation cost. `_measure` warms up
  before timing, so the reported latency excludes compilation, but the first
  call in a fresh process is slow (seconds, not milliseconds).
* `fullgraph=True` is deliberately used by the benchmark row so a graph
  break fails loudly rather than silently measuring a partially-eager
  program.
* CPU/GPU results are host-specific; compare rows from the same run only.

## Code map

| Location | Role |
| --- | --- |
| `ops_kda.compiled_chunk_kda` | cached compile wrapper + policy knobs |
| `ops_kda._chunk_kda_prepare` | branch-free clamp, metadata-only ownership, setup |
| `ops_kda._chunk_kda_inner_loop` | sequential per-chunk state carry |
| `ops_kda._chunk_kda_inner_loop_recompute` | memory-bounded `Aqk` rebuild |
| `run_benchmark.bench_kda_chunk_compiled` | parity-gated Exp 2 row |
| `run_roofline._core_accounting` | chunked KDA pricing for the new row |
| `test_recompute_attn.py` | fullgraph + clamp-identity + memory-policy tests |
