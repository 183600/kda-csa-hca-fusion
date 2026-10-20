# Batched chunked-prefill cache (`forward_chunk`)

*Why a continuation chunk used to cost `T` decode dispatches, how
`CSADecodingCache.forward_chunk` / `HCADecodingCache.forward_chunk` remove
them without changing a single output, and how `prefill_profile.py` makes
the throughput/numerical-drift trade-off measurable.*

## The problem

The decoding wrappers (`run_decoding.py`) accept a prompt in several calls:
the first call seeds the cache, later *continuation chunks* extend it. This
is the primitive a serving engine needs for chunked prefill, where a long
prompt is interleaved with decode work instead of running as one
head-of-line-blocking forward.

Correctness was already pinned (`test_chunked_prefill_matches_one_shot_forward`),
but the continuation path still paid for a Python token loop:

```
for each token i in the chunk:
    project one token        # GEMV dispatches
    cache.append_step(...)   # ring-buffer / compressor update
    cache.forward_step(...)  # per-token attention
```

At `T = 4096` that is three dispatches per token per layer, and the
per-token GEMMs cannot use the machine's full matrix throughput. The
one-shot path is `O(1)` calls but ignores the cache, so it cannot be used
for a continuation chunk.

## The design

Two layers changed independently:

1. **One projection GEMM per chunk.** Token projections are a per-row
   linear map, so `self._project(x)` over all `T` rows is numerically
   equivalent to `T` calls on single rows. `run_decoding` now projects the
   whole chunk once and slices the results for caching.
2. **One batched attention call per chunk.** `CSADecodingCache.forward_chunk`
   and `HCADecodingCache.forward_chunk` take the whole chunk, append it to
   the cache once, and build the per-query attention directly. The
   per-token `append_step`/`forward_step` APIs remain unchanged for decode.

`HybridDecoding._forward_csa_layer` / `_forward_hca_layer` use the same
fast path, so the win applies to every hybrid route (`kda`, `mamba2`,
`rwkv7`) and both token-level wrapper classes.

## Why the math is identical

Chunked prefill changes only *dispatch order*, never the model:

* **Block mask.** The cache already owns the strictly-preceding causal
  block mask. `forward_chunk` reconstructs it from
  `n_tokens_seen + [0..T)`, exactly what the per-token loop would have
  seen at each step. `T > 1` indexer calls receive the same
  `causal_block_mask` (row `i` enables blocks completed before that token,
  not blocks completed by the end of the chunk).
* **Sliding window.** A snapshot of the ring buffer is taken *before* the
  chunk append (`_chunk_window_keys`), then rows are sliced by absolute
  position. A query never sees its own key or a later token. `win = 0`,
  partial windows, and windows longer than the cache all fall out of the
  same index math.
* **RoPE.** Positions are absolute (`n_tokens_seen + i`), so the chunked
  path rotates exactly like the one-shot forward; partial RoPE and the
  inverse rotation on the output are untouched.
* **Joint softmax.** `_joint_softmax_chunk` mirrors `forward_step`'s sink
  handling (one shared normalization over sparse + window + sink columns)
  and uses the indexer's `indices >= 0` validity mask — not the dense
  block mask — because top-k may include padding slots.
* **Cache state.** Appends happen in the same order, so after the chunk
  `n_tokens_seen`, the block count, the sliding-window contents, and the
  compressor accumulators are bit-identical to the token loop (pinned in
  `test_chunked_prefill_cache.py`).

The only observable difference is batched-GEMM reduction order: fp32
outputs agree with the token loop to ~`1e-7` (`~1e-16` in fp64), which is
far below the `1e-5` mixed-tolerance contract.

## Measuring it

`prefill_profile.py` profiles a decoder wrapper over a prompt and a set of
schedules (`full`, uniform size, or explicit chunk list). For each schedule
it reports median latency, tokens/s, and drift against the one-shot
reference, computed in fp64 with explicit NaN/Inf failure:

```bash
python3 prefill_profile.py --model hybrid --total-len 256
python3 prefill_profile.py --model csa --total-len 128 \
    --schedules full,64,32,16,1 --repeats 5 --warmup 2 \
    --json results/prefill_profile.json
```

## Verification

* `test_chunked_prefill_cache.py` compares `forward_chunk` with a
  `deepcopy`'d cache driven token-by-token across empty/partial caches,
  block crossings, `win ∈ {0,1,4,12}`, `topk ∈ {0,2,100}`, sinks,
  L2/RMS normalization, partial RoPE on/off, batch 2, and fp32/fp64. It
  also compares the *final cache state*, not just the outputs.
* `test_prefill_profile.py` checks schedule expansion, drift metrics,
  non-finite detection, and that the wrapper path issues exactly one
  projection call per chunk.
* `test_results_contracts.py::test_chunked_prefill_matches_one_shot_forward`
  remains the end-to-end schedule-invariance contract.

## Limitations

* The first prefill call still uses the one-shot `naive_csa`/`naive_hca`
  path; `forward_chunk` is an incremental/continuation optimization.
* Per-token decode remains per-token: chunking a `T=1` schedule cannot be
  batched, and the profiler shows that floor rather than hiding it.
* Timing is host-specific. The portable contract is the drift column:
  every schedule must stay inside `atol + rtol·max|reference|`.
