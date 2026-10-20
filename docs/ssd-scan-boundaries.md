# SSD scan boundaries: metadata, checkpoint resume, and measured throughput

*Why the chunked Mamba-2 path now exposes the decomposition it executes,
and how Exp 2 makes the throughput difference measurable without giving up
the correctness contract.*

## The problem

`ops_mamba.chunked_mamba2_scan` is the throughput-oriented view of the
Mamba-2 / SSD operator: quadratic intra-chunk attention plus a
Hillis–Steele scan over scalar-affine chunk transitions. Its work and
memory profile depend on an *operator boundary* that the sequential
reference does not have:

* how many chunks (`ceil(T / chunk_size)`),
* how much right-padding is added,
* how many doubling levels the associative scan runs,
* where a prefix state can be checkpointed and resumed.

Before this change those numbers existed only as local variables inside
the scan. Analytic pricing (`roofline.ssd_core_flops_breakdown`), the
benchmark row, and downstream tooling each had to re-derive them. Any
disagreement would silently mis-price a measured row or claim a
checkpoint boundary that the operator does not actually have.

## The metadata contract

`ops_mamba.mamba2_scan_boundary_metadata(seqlen, chunk_size=64)` returns
the decomposition as a JSON-safe dict:

| field | meaning |
|---|---|
| `seqlen`, `chunk_size` | validated inputs |
| `n_chunks` | `ceil(seqlen / chunk_size)` (0 for `T=0`) |
| `padded_len` | `n_chunks * chunk_size` |
| `pad` | right-padding added by the scan |
| `scan_levels` | `ceil(log2(n_chunks))` doubling levels |
| `chunk_starts` | zero-based start index of every chunk |
| `chunk_lengths` | real (unpadded) token count per chunk |
| `last_chunk_len` | length of the final chunk (0 when empty) |

`chunked_mamba2_scan` now allocates from this exact record, so the
metadata is the implementation's definition rather than a parallel
calculation. Invalid `seqlen` / `chunk_size` values (negative, bool,
non-int) raise `ValueError`.

## Boundaries are exact checkpoints

A `chunk_starts` entry is not just descriptive. Because the decomposition
restarts on aligned chunk boundaries, running the chunked scan on a prefix
and then feeding its final state into a second chunked call on the suffix
reproduces the full call to floating-point round-off. This is pinned in
`test_chunked_ssd.py::test_chunked_boundary_metadata` at fp64 for every
interior boundary of a non-divisible sequence.

That property is what makes the metadata useful for chunked prefill and
for checkpointing a long scan: a serving engine can cut at a recorded
boundary instead of guessing a position that only *looks* safe.

## Measured throughput in Exp 2

`run_benchmark.py` gains the core-boundary row:

```
mamba2_chunked_scan   (C=64, same inputs as mamba2_scan)
```

* Inputs and geometry are shared with `mamba2_scan` through
  `_mamba2_scan_inputs` (same seeded generator key), so the latency ratio
  compares kernel organization, not different random data.
* **Parity gate before timing.** The row first runs both scans and
  requires finite outputs and `max|chunked - sequential| <= 1e-4` (fp32).
  A divergent operator raises, producing an explicit error row instead of
  a fast-but-wrong measurement.
* Each measured row now records `tokens_per_s` (`B*T / median_seconds`)
  alongside latency; the core rows have no projections or weights, so this
  is the scan's actual token throughput at the recorded boundary.
* The chunked row also records `scan_boundary` and
  `scan_parity_max_abs`/`scan_parity_ok`; `op_boundary` and `op_workload`
  mark it `compute_boundary='core'` with `mamba_chunk_size=64`.

`run_roofline.py` prices both scan rows through the same
`_mamba2_scan_accounting` path: `mamba_chunk_size>0` selects
`ssd_core_flops(..., chunked=True)`, and the row's `scan_boundary` is
copied into the roofline point so the analytic decomposition is auditable.
`make_figures.fig_benchmark` labels the sequential and chunked scan rows
separately (solid vs dashed) inside the core panel.

## How to read the numbers

The chunked decomposition does more arithmetic than the sequential
recurrence (the `T*C` intra-chunk quadratic term) but replaces `T`
sequential state updates with batched matmuls and `ceil(log2 NT)`
parallel scan levels. On CPU at the benchmarked geometry this is a large
win; the measured `time_ms` and `tokens_per_s` columns are the
authoritative device-specific answer. The analytic model is a work
estimate used for crossover analysis, not a latency prediction.

Do not compare the scan rows with `mamba2_layer` / `hybrid_mamba2`: those
are end-to-end rows that include projections, convolution, gating and
`out_proj`. The subplot split by `compute_boundary` exists to keep that
comparison honest.

## Code map

* `ops_mamba.mamba2_scan_boundary_metadata` — metadata contract and
  validation; used by `chunked_mamba2_scan`.
* `run_benchmark.bench_mamba2_chunked_scan` — parity-gated core row and
  row metadata.
* `run_roofline._mamba2_scan_accounting` — analytic join for both modes.
* `test_chunked_ssd.py` — hand-computed decomposition, agreement with the
  roofline model, invalid-argument contracts, and checkpoint-resume
  equivalence.
* `test_latency_profile.py` — chunked benchmark-row metadata and roofline
  accounting.
