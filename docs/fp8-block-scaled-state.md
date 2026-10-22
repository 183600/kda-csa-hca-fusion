# Block-scaled FP8 state storage and the decode traffic ledger

`kv_quant.FP8Spec` adds a real FP8 storage contract next to the integer
`QuantSpec`, and `decode_bandwidth` can now price a **per-operator** storage
policy. Together they answer a memory-bound question the all-fp16/fp32
ledger could not: *how far does FP8 recurrent state move the batch at which
decode becomes state-bandwidth-bound, and does the answer differ between
KDA, Mamba-2, and RWKV-7?*

## Format and cost

`FP8Spec` uses an IEEE-style payload and one shared scale per contiguous
block along the last axis:

| field | default | meaning |
|---|---|---|
| `format` | `'e4m3fn'` | E4M3FN (±448) or `'e5m2'` (±57344) |
| `group_size` | 32 | elements sharing one scale |
| `scale_bits` | 8 | E8M0 scale (16 for an fp16 scale) |
| `power_of_two` | `True` | snap the block scale up to a power of two (MXFP8) |

The default is therefore MXFP8:

```
bits/element  = 8 + 8/32 = 8.25
bytes/element = 1.03125          (1.94x smaller than fp16)
```

`scale_bits=0` is the idealized payload-only bound and is reported
separately, exactly like the integer codec's free-metadata lower bound.

## Numerics

* Values are rounded to the finite FP8 grid with round-to-nearest-even.
* E4M3FN has no Inf; E5M2's finite overflow saturates. Finite inputs never
  produce Inf/NaN — an overflowed quantizer is not a storage contract.
* Subnormal magnitudes use the format's subnormal step (`2^-9` for E4M3FN,
  `2^-16` for E5M2), so the low end of the grid is exact.
* `power_of_two=True` computes `scale = 2**ceil(log2(amax/max_finite))`,
  which can be up to 2x coarser than the exact absmax scale but can never
  overflow the format. `power_of_two=False` uses `amax/max_finite` and the
  full mantissa range.
* Scales are computed in fp32 (fp64 input stays fp64); groups are padded
  internally on a non-divisible tail, matching `QuantSpec`.
* Non-finite inputs are rejected, `ste=True` has an identity gradient, and
  `fake_quantize_` remains inference-only.

```python
from kv_quant import FP8Spec, fake_quantize, bits_per_element

spec = FP8Spec()                       # MXFP8
state = fake_quantize(state, spec)     # QAT (STE)
state = fake_quantize(state, spec, ste=False)   # inference
bits_per_element(spec)                 # 8.25
```

The `kv_budget` bridge accepts an `FP8Spec` anywhere a `QuantSpec` is
accepted:

```python
from kv_budget import op_read_elements
from kv_quant import FP8Spec, quantized_row_bytes

row = op_read_elements('kda', T=128, spec=...)
quantized_row_bytes(row, {'state': FP8Spec()})['state_bytes']
```

## Per-operator decode traffic

`decode_bandwidth.decode_traffic_breakdown(..., storage=...)` resolves a
spec (or a mapping over `'memory'`, `'csa'`, `'hca'`, with `'*'` as the
fallback) into bytes/element and applies it to that operator's element
row. `decode_step_ledger`, `weights_state_switch_batch`, and
`route_switch_table` accept the same policy plus an optional
`weight_itemsize`.

The switch batch is now computed in **bytes**, not elements:

```
switch_batch = weight_bytes_per_step / state_bytes_per_token
```

so shrinking only the recurrent state moves the crossover outward while a
simultaneous weight-precision change is accounted for explicitly.

```bash
# fp16 weights/caches, FP8 recurrent state
python3 run_decode_bandwidth.py --itemsize 2 --storage mixed-fp8
```

## Example (d_model=32, T=128, fp16 baseline)

| route | state B/token (fp16) | state B/token (MXFP8 memory) | switch batch (fp16) | switch batch (MXFP8) |
|---|---:|---:|---:|---:|
| KDA | 11,316 | 6,852 | 7.51 | 12.41 |
| Mamba-2 | 14,388 | 8,436 | 7.30 | 12.46 |

At T=4096 the same policy saves only ~4–8% of state bytes because the
CSA/HCA compressed caches dominate the per-token read; the memory route's
recurrent state is a small share of long-context decode traffic. This is
the useful negative result: FP8 state storage is a short/medium-context
lever, not a long-context one, unless the retrieval caches are quantized
too.

## Scope / honesty

* `FP8Spec` simulates storage numerics, not a fused FP8 kernel's throughput.
  The ledger prices bytes only; `bandwidth_bound_ms` remains a memory-side
  lower bound.
* Power-of-two block scaling adds up to 1 bit of effective precision loss
  versus the exact absmax scale; the exact-scale variant is available.
* The canonical decode element model still amortizes / excludes cache
  writes (see `lm_consistency.decode_elements_per_token`); the FP8 policy
  changes bytes per element, not the underlying element counts.
* Weights and CSA/HCA caches are priced at the caller's `itemsize`; this
  module does not model FP8 weights or FP8 attention compute.
