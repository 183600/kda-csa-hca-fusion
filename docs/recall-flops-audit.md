# Recall-conditioned FLOPs audit (`recall_flops_audit.py`)

## Why this exists

The repository has two independent diagnostic ledgers:

* `flops_audit.py` measures the exact GEMM work of a KDA / CSA / HCA layer
  and pins a closed-form reference model against the dispatched operators.
* `recall_reach.py` derives which planted (key, value) pairs a layout can
  recover *exactly*: the pair must share a compressed block that strictly
  precedes the cue's block, or the key must be inside the query's sliding
  window. Recurrent KDA / Mamba layers are capacity-limited and are not
  counted as exact paths.

Neither ledger answers the architecture question: **how much prefill compute
sits in layers that can actually carry the long-range association?** A
retrieval-heavy layer may be cheap in isolation, but the default 3:1:1 stack
spends most of its FLOPs in KDA, which has no exact-recall guarantee. The new
module joins the two ledgers and reports that allocation per distance.

## Definitions

For a plan and a prefill length `T = seq_len`, every layer has an exact GEMM
work `F_l = 2 * reference_gemm_macs(kind, cfg, T)` (the same accounting as
`flops_audit.audit_layer`). For a key at position `p`, value at `p+1`, cue at
`seq_len-1`, and distance `d = cue - p`:

```
exact_layers(d) = { l : block_hit(l, d) or window_hit(l, d) }
exact_reachable_flops(d) = sum_{l in exact_layers(d)} F_l
exact_reachable_fraction(d) = exact_reachable_flops(d) / sum_l F_l
```

`window_hit` uses the repo's strict window rule: the key itself must be in
`[cue-win+1, cue]`, i.e. `d <= win - 1`. This is the same rule as
`structural_recall` and `test_recall_reach.py`; it is why
`evaluate_recall`'s distance buckets were corrected from `d <= win` to
`d <= win - 1`.

Two aggregates are reported:

* `far_mean_exact_reachable_fraction`: the compute share above, averaged
  over placements beyond every sliding window (`d >= max(2, max_window)`).
* `structural_far_any_exact_fraction`: the *placement* coverage — what
  fraction of far placements has at least one exact path.

They are intentionally separate: one says how much compute is in reachable
layers, the other says how many placements are reachable at all.

## What the audit checks

`audit_layout` sums the per-layer reference model and, by default, runs the
assembled `HybridAttentionStack` through the dispatch counter. At `T >= 2 *
hca_m2` the measured count must match the reference sum exactly. With a
single HCA compressed block, PyTorch may fold the value contraction into a
mul+sum; the audit reports the documented bound
`batch * n_hca * Tp * c * nh` and classifies the row as `hca_value_fold`
rather than reporting a spurious mismatch.

## Usage

```bash
python3 recall_flops_audit.py --seq-len 128
python3 recall_flops_audit.py \
  --ratios 3:1:1,1:2:2,1:0:0,0:1:1 --total-layers 10 \
  --seq-len 128 --json /tmp/recall_flops.json
```

The default geometry is `flops_audit.probe_config()` (`d_model=64`,
`csa_m=16`, `hca_m2=32`, windows `16`). Named presets come from
`recall_reach.LAYOUT_PRESETS`; arbitrary ratios are expanded by
`ratio_layout` with the same repeat-and-truncate convention as
`HybridAttentionStack` ratio mode.

## Example result (probe geometry, `seq_len=128`)

```
                 group   L     GFLOPs    KDA%  far reach% struct far%      measure
----------------------------------------------------------------------------------
           alternating  10     0.1415   93.7%       5.64%      96.43%        exact
        retrieval_back  10     0.1415   93.7%       5.64%      96.43%        exact
       retrieval_front  10     0.1415   93.7%       5.64%      96.43%        exact
                spread  10     0.1415   93.7%       5.64%      96.43%        exact
           ratio-3:1:1  10     0.1415   93.7%       5.64%      96.43%        exact
           ratio-1:2:2  10     0.0620   71.2%      25.74%      96.43%        exact
           ratio-1:0:0  10     0.2210  100.0%       0.00%       0.00%        exact
           ratio-0:1:1  10     0.0223    0.0%      89.50%      96.43%        exact
```

The four interleaving presets have identical per-kind counts, so they tie by
construction; the audit is measuring allocation, not layer order. The ratio
rows show the trade-off: at this small geometry the 3:1:1 stack spends 93.7%
of its GEMM FLOPs in KDA while only 5.64% of far-placement compute sits in
structurally exact layers. A retrieval-only stack is both cheaper *and*
allocates most of its compute to exact paths, but it gives up the recurrent
bulk-mixing capacity that is the other half of the hybrid design — the
frontier is not a ranking.

## Caveats

* Structural reachability is not end-to-end accuracy. The CSA indexer can
  miss a reachable block, and a recurrent state can store associations
  without an exact guarantee.
* GEMM FLOPs exclude layernorm/residual/elementwise work and are not a
  latency prediction. The special-op ledger is reported by
  `flops_audit`, not converted to FLOPs here.
* The exact reference model currently covers the KDA memory route only.
  Mamba-2 / RWKV-7 layouts are rejected rather than silently approximated.
* The `pareto_optimal` flag is a two-axis diagnostic (total FLOPs vs far
  exact-reachable share with structural coverage as a constraint), not a
  recommendation. Parameter count and model quality are not in the objective.

## Tests

`run_correctness.py::test_recall_flops_audit` pins the hand-derived
distance cases, the KDA-only zero case, the whole-stack exact match at
`T=64`, the HCA value-fold bound at `T=32`, ratio expansion, and input
validation. `run_correctness.py::test_recall_distance_buckets_match_structural_window`
pins the corrected recall bucket boundaries.
