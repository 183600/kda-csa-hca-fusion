# Measured FLOPs accounting (and the online-softmax oracle)

`prefill_flops` (`run_kv_cache.py`) is an **analytic** ledger: it models
the matmul shapes from the paper and never instruments a forward. The
README says so explicitly ("approximate ... not measured") and suggests a
counter cross-check. `flops_audit.py` performs that cross-check for a
single reference layer and closes the loop with an exact accounting of
the work the reference implementation actually dispatches.

Companion: `online_softmax.py` is the streaming form of the joint CSA/HCA
softmax (sink in the denominator only), checked against the dense path.

## Why not `torch.utils.flop_counter.FlopCounterMode`

`FlopCounterMode` understands `mm`/`addmm`/`bmm`/`conv`, but returns
**zero** for the most common attention contraction in this repo:

```python
torch.einsum('b t h d, b n d -> b h t n', q, kv)   # 0 FLOPs reported
```

The broadcasted operand is expanded and the underlying kernel is not one
of the counter's recognized shapes. `flops_audit._GemmAuditMode` is a
`TorchDispatchMode` that prices the **dispatched** operators instead:

| op | MACs |
|---|---|
| `mm`, `addmm`, `bmm`, `baddbmm` | product of the contracted and output dims |
| `convolution` | `C_out * out_spatial * (C_in/groups) * kernel` |
| `linalg_solve_triangular` | `systems * N^2 (N-1) / 2`, `systems = numel / N^2` |
| special elementwise/reduction ops | element count, kept in a separate ledger |

`einsum`, `linear`, and `matmul` decompose into the first two, so the
counter stays exact as long as the decomposition is a matmul. It is not a
performance timer; it is an exact MAC count for the reference graph.

## What is counted

`audit_layer(kind, cfg, T)` builds one `build_mixer` layer, runs a forward
under the counter, and reports:

| field | meaning |
|---|---|
| `ledger.gemm_macs` | `mm + bmm + conv + trsm` MACs (2 FLOPs each) |
| `ledger.special_ops` | `exp`, `sigmoid`, `silu`, `softplus`, `sin`, `cos`, `softmax`, `logsumexp`, `l2_norm` element counts |
| `canonical_flops` | `run_kv_cache.prefill_flops` (the repo's analytic number) |
| `reference_model_macs` | the closed-form model of the reference forward |
| `reference_tax_vs_canonical` | `2 * gemm_macs / canonical_flops` |
| `path` | KDA core actually dispatched: `recurrent` or `chunk` |

## Probe-geometry results (d=64, H=2, HV=4, K=V=32, BT=64)

| kind | T | path | canonical | ref GEMM | ref/canon |
|---|---|---|---|---|---|
| KDA | 32 | recurrent | 4.092M | 2.257M | **0.552** |
| KDA | 64 | chunk | 8.184M | 10.854M | **1.326** |
| KDA | 128 | chunk | 16.368M | 21.709M | **1.326** |
| CSA | 128 | — | 2.634M | 2.691M | 1.021 |
| HCA | 128 | — | 1.713M | 1.769M | 1.033 |

The KDA numbers are the two actionable findings:

1. **Short sequences are overcounted.** Below `kda_chunk_size` the layer
   never enters the chunk path, yet the analytic formula always adds
   `3T*C*K + T*C^2` per head. At `T=32` the executed work is 55% of the
   reported number. `kda_executed_flops` zeroes those terms when
   `kda_path == 'recurrent'`.

2. **The chunk path is undercounted by ~33% at `BT=64`.** The reference
   builds `A`/`Aqk` with an elementwise sub/clamp/exp/multiply chain,
   solves a batched triangular system, and runs `A@(gexp*k)`,
   `A@v`, `w@S`, `q@S`, `Aqk@v`, `k^T@v`. The paper's
   `3T*C*dh + T*C^2` term only approximates this. The ratio is
   scale-invariant (`T=128` and `T=256` agree to 1e-9) because both the
   reference and canonical terms are linear in `T`.

The recurrent path is also special: `(k * S).sum(-2)` and the outer-product
update are **elementwise broadcast-reduce**, not GEMMs. They are real work
(`2 * T * HV * K * V` MACs) that no dispatcher-based counter sees, so
`kda_reference_gemm_macs + kda_broadcast_encoded_macs ==
kda_executed_flops` is the identity pinned by the tests.

## Exact reference model

All projections run on the padded length `Tp` (ceil to `m`/`m2`/`BT`);
only `o_proj` runs on the original `T`. The model matches the measured
GEMM MACs **exactly** for the tested `T` (including padded `T=100`):

* **KDA**: A/Aqk block entries from `_attn_block_rows` (prepare) or
  `_recompute_block_rows` (recompute); `trsm` counted as
  `systems * N^2(N-1)/2`; one term per `A@`, `w@S`, `q@S`, `Aqk@v`,
  `k^T@v`.
* **CSA**: dense indexer masks at `Tp`, `core` on the full `topk` (not
  clamped to the block count), `window` on the sliding window.
* **HCA**: `core` on `nb` blocks, `window` on the sliding window. Caveat:
  with a single block (`nb == 1`) the reference folds the PV product into
  `mul + sum`, so no `bmm` dispatches; the model over-reports by
  `Tp * c * nh`. Tests allow that documented shortfall.

`associated_scan` (KDA's alternative core) is not modelled: `audit_layer`
raises rather than mis-pricing it.

## Online-softmax oracle

`online_softmax_attention(q, k, v, key_chunk=..., mask=..., sink_logits=...)`
streams key blocks through the standard `(m, den, acc)` state. It is
value-equal to the dense joint softmax `< 1e-5` fp32, independent of
`key_chunk`, and its merge is associative up to fp re-association.
`den == 0` rows (all-masked) return exact zeros.

The sink is a virtual key with value zero: it appears in the denominator
only, matching `ops_csa._joint_p` / `ops_hca._joint_p` and DeepSeek-V4
Eq 27. The counter confirms online and dense perform the same `QK^T` and
`P@V` MACs (`2 * B * H * T * N * D`); only the dense `softmax` kernel
disappears from the special-op ledger. That is the point of the
primitive: memory footprint changes, arithmetic does not.

## Scope and caveats

* CPU reference modules in eval mode, fp32, batch 1 by default. The
  models price the reference implementation, not a fused kernel.
* `special_ops` mix unary map counts with differing costs (`l2_norm` and
  `softmax` are multi-op reductions); they are reported separately and
  deliberately excluded from `gemm_macs`.
* The KDA chunk-path exp count includes the `A`/`Aqk` build; only the
  gate map (`padded * HV * K`) is a clean closed form. Tests pin the full
  count from `_attn_block_rows` instead of assuming one gate pass.

Run:

```
python3 flops_audit.py --seq-lens 32,64,128,100
python3 -m pytest -q test_flops_audit.py
```
