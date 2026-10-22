# Time-to-first-token budget with FP8 weights and state

`ttft_budget.py` adds the serving-phase bound the repository was missing.
`prefill_profile.py` measures chunked-prefill latency and
`decode_bandwidth.py` prices steady-state decode traffic; neither says how
long the *first* token takes or which phase owns that time.

## Model

```
TTFT >= prefill_lower_bound + first_decode_lower_bound
```

**Prefill**

```
flops  = batch * prefill_flops(hybrid_<route>, T, config geometry)
bytes  = weight_bytes
       + 2 * batch * T * d_model * activation_itemsize   # input + output
       + state_bytes                                      # retained cache
time   = max(flops / effective_peak_flops, bytes / bw)
```

The FLOP row comes from `run_kv_cache.prefill_flops` with the config
geometry from `lm_consistency.lm_tower_geometry`, so KDA / Mamba-2 / RWKV-7
and explicit layouts use the same accounting as Exp 3 / Exp 7.

**First decode**

Reads use the *marginal* budget at position `T`
(`kv_budget.instant_layout_read_elements`), not the `0..T-1` prefill
average used by the steady-state decode ledger. The token that produces the
first output sees `T // m` CSA blocks and `T // m2` HCA blocks, so on long
contexts it reads ~2x the average. The step also pays one recurrent-state
write and one weight stream. In the memory-bound decode regime only its
bandwidth time is charged.

`first_decode.instant_vs_average_reads` reports the ratio explicitly, so a
reader can see when the steady-state model under-prices TTFT.

## Storage policy

`weight_storage` and `state_storage` accept the same `QuantSpec` /
`FP8Spec` contracts as the decode ledger. State is priced per operator
(`'memory'`, `'csa'`, `'hca'`, with `'*'` as the fallback), so FP8 weights
and FP8 recurrent state are separable knobs.

Storage-only FP8 changes **bytes, not FLOPs**. It helps TTFT only while the
prefill is memory-bound. `compute_speedup` is a separate, explicit
assumption (the `fp8-weights-state-fp8-compute` scenario uses 2x).

## Running

```bash
python3 run_ttft_budget.py \
    --lengths 1,16,128,1024,8192 --batches 1,8 \
    --routes kda,mamba2,rwkv7 --d-model 256 \
    --json results/exp17_ttft_budget.json
```

Host anchors default to measured PyTorch GEMM / copy throughput
(`lm_consistency.host_peaks`), with `--peak-tflops` / `--bw-gbps`
overrides. A measured fp32 GEMM is a conservative anchor for an fp16/FP8
tensor-core run; pass the intended arithmetic peak explicitly when
comparing compute assumptions.

## What the sweep shows (d_model=256, 67 TFLOP/s, 3.35 TB/s)

| route | T | batch | prefill bound | TTFT speedup: FP8 weights | TTFT speedup: FP8 weights+state | TTFT speedup: + FP8 compute |
|---|---:|---:|---|---:|---:|---:|
| KDA | 1 | 1 | memory | 1.92x | 1.94x | 1.94x |
| KDA | 16 | 1 | memory | 1.46x | 1.47x | 1.91x |
| KDA | 16 | 8 | compute | 1.06x | 1.07x | 1.99x |
| KDA | 128 | 1 | compute | 1.06x | 1.07x | 1.99x |
| KDA | 8192 | 1 | compute | 1.00x | 1.00x | 2.00x |
| Mamba-2 | 16 | 1 | memory | 1.48x | 1.50x | 1.93x |
| Mamba-2 | 128 | 1 | compute | 1.07x | 1.07x | 1.99x |

Two conclusions:

1. **Weight-only FP8 is a short-prompt / low-batch lever.** The prefill
   arithmetic intensity is roughly `batch * T` FLOP/byte for a dense
   weight stream, so it crosses the machine ridge (peak FLOP/s / bandwidth)
   at a small `batch * T`. Above that, halving the weight bytes buys at
   most the remaining weight share of memory time.
2. **Long-context TTFT needs FP8 arithmetic, not just FP8 storage.** The
   `+ FP8 compute` column approaches 2x because the compute term dominates;
   storage-only rows converge to 1.00x. At 8192 tokens the first-decode
   share is also small, so cache quantization alone cannot move TTFT.

The repo's tiny `d_model=32` matrix geometry produces the same shape of
result at smaller absolute milliseconds. Absolute times are not a hardware
claim; the bound classification and the speedup ordering are the portable
outputs.

## Scope / honesty

* Attention-tower scope: embeddings, lm_head, sampling, and serving
  queueing are excluded. TTFT here is a model-side lower bound.
* `max(compute, memory)` assumes perfect intra-phase overlap and no
  cross-phase overlap; the sum of two lower bounds is a lower bound.
* Prefill state bytes are the retained cache footprint (one write of every
  retained slot), not the reference Python path's temporary traffic.
* A weight stream is charged in both phases; a kernel that keeps weights
  resident across requests can do better on the weight term.
