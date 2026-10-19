# Tensor-parallel sharding reference + prefill/decode latency

`tp_parallel.py` adds the missing serving axis to the hybrid stack:
**tensor parallelism (TP)**. It provides (a) a numerically exact
head-sharded reference forward and (b) an analytic prefill/decode latency
ledger over that same shard map.

## Shard map

TP shards the value-head axis and replicates everything shared. The map is
measured from the real modules (`build_mixer`), not re-derived from
formulas.

| Layer | Sharded by the value-head axis | Replicated on every rank |
|---|---|---|
| KDA (`KDAHybridLayer`) | `v_proj`, `g_up`, `beta`, `o_proj` columns, recurrent state | short conv, `q_proj`, `k_proj`, `g_down` |
| CSA (`CSAHybridLayer`) | `W_UQ`, `sink`, `o_proj` columns | compression (`W_aKV/bKV/aZ/bZ`, `B_a/b`), lightning indexer (`W_IUQ`, `W_w`, `W_KV_idx`, `W_Z_idx`, `B_idx`), `W_DQ` |
| HCA (`HCAHybridLayer`) | `W_UQ`, `sink`, `o_proj` columns | compression (`W_KV`, `W_Z`, `B_pos`), `W_DQ` |

KDA's q/k are expanded by the GVA factor and sliced to each rank's
value-head range, so a rank boundary may fall inside one q/k head group
(H=2, HV=4 at TP=2) without changing the math. CSA/HCA replicate the
shared-KV MQA path and split only the attention heads — the standard
layout when KV heads are not themselves sharded.

Each rank produces a partial output; one ring **all-reduce per layer**
reassembles the `o_proj` result. LayerNorm and the residual stream are
replicated.

Numerical equivalence (pinned by `test_tp_parallel.py`):

```
sum_r rank_r(layer)          == layer_output            (< 1e-5 fp32)
cat_r rank_r(state)          == full recurrent state    (< 1e-5 fp32)
hybrid_stack_forward_tp(m,x,2) == m(x)                  (< 1e-5 fp32)
```

The last identity is measured on an actual MQAR batch
(`run_quality.make_mqar_batch`) at the Exp 4/5 probe geometry: **MQAR
representations — hence measured MQAR accuracy — are invariant to the TP
degree**. TP is a serving/throughput transformation, not a model change.

Scope: KDA memory route only (Mamba-2 / RWKV-7 state layouts are rejected
rather than mis-priced). The reference always selects the reference KDA
backend and assumes fresh state for the stack-level check. It simulates
every rank in one process; it is the oracle for a distributed kernel, not
a runtime.

## FLOP split

`layer_flops_split(cfg, kind, T)` splits the canonical
`run_kv_cache.prefill_flops` into `replicated + sharded`, mirroring the
table above. A regression test pins exact equality
(`replicated + sharded == prefill_flops`) for every kind and several `T`,
so the TP model cannot drift from the repository's FLOP accounting.

## Latency model

`tp_latency` / `tp_sweep` price one prefill and one decode step per TP
degree:

```
prefill = max(FLOPs_rank / peak_flops, weights_rank / bw) + Σ_layers AR(B·T·d)
decode  = max(batch·FLOPs_tok_rank / peak_flops,
              (weights_rank + batch·state_rank) / bw) + Σ_layers AR(B·d)
```

where `FLOPs_tok_rank` is the amortized per-token share of the prefill
FLOPs (the `lm_consistency` convention). The ring all-reduce term is

```
AR(S) = 2(P-1)/P · S / link_bw  +  2(P-1) · α
```

Communication is serialized with compute (no overlap), which is
deliberately conservative for TP. `HardwareModel` carries the achieved
fp32 peaks, the one-way link bandwidth and the per-step all-reduce
latency; all defaults are nominal H100-class values, not benchmark
claims.

### Decode-phase asymmetry (the main finding)

* **KDA state is head-scaled**, so its per-token read/write traffic
  divides by the TP degree.
* **CSA/HCA compressed KV is shared across value heads (MQA)**, so every
  rank reads the same rows: that traffic is *replicated* and does **not**
  shrink with TP. `tp_decode_traffic` reports the split.

Consequently TP helps prefill (compute-bound, large messages amortize the
fixed α) much earlier than it helps decode. At small batch,
`2(P-1)·α` per layer is on the critical path of every decode step, so
the sweep often reports `decode_optimum_tp` = 1 even when prefill prefers
a larger degree. The MQAR probe geometry (HV = csa_nh = hca_nh = 2) caps
the degree at 2; `adjacent_probe_config()` (8 heads per axis) exercises
TP up to 8.

## Usage

```bash
python3 tp_parallel.py --config paper --tp 1,2,4,8 \
    --prefill-len 2048 --decode-batch 16 --layers 5
python3 tp_parallel.py --config mqar --tp 1,2 --prefill-len 32 --decode-batch 8
pytest -q test_tp_parallel.py
```

Caveats: the model is analytic (no Python/dispatch overhead, kernel
inefficiency, or comm/compute overlap), and the reference TP forward is
fp32-CPU-first. Absolute numbers are host/geometry-specific; the
replicated-vs-sharded identities and the prefill/decode ordering are the
transferable results.
