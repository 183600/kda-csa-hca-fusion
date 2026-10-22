# Decode-state ablation and causality audit

`decode_state_ablation.py` supplies the missing **negative controls** for the
incremental hybrid decoder. The existing equivalence tests compare the cached
path against a full-sequence forward; that comparison can pass *vacuously* if
an integration silently stops reading a cache (especially on a short suffix).
The ablation makes each state family prove that it is load-bearing.

The three state families are:

| Part | Owned by | Dropped by |
|---|---|---|
| `memory_state` | `HybridKCHAttention._kda_state` (KDA / Mamba-2 / RWKV-7) | setting it to `None` |
| `memory_lookback` | each memory layer (`_conv_lookback` / `_shift_state`) | `layer.reset_conv_state()` |
| `csa_cache` | `HybridDecoding._csa_caches` | replacing the slot with `None` |
| `hca_cache` | `HybridDecoding._hca_caches` | replacing the slot with `None` |

## Controlled ablation

All arms share the same weights, the same token stream, and the same
prefill. The only difference is the dropped state:

* `full` — ordinary incremental decode.
* `at_prefill` — drop the family once after the prefill, then decode normally.
  This is a counterfactual branch: it measures whether the prefill-populated
  state is used.
* `every_step` — drop the family before every decoded token. This is a
  stateless decoder and measures whether the family accumulates.

A family is `load_bearing` iff at least one suffix element moves by more than
`tol` (default `1e-5`). The prefill prefix must remain bit-identical, because
the drop happens after it.

```python
from decode_state_ablation import decode_state_ablation
from run_decoding import HybridDecoding

decoder = HybridDecoding(d_model=32, total_layers=5, csa_topk=100).eval()
res = decode_state_ablation(decoder, x, prefill_len=8)
for arm in res['arms']:
    print(arm['part'], arm['mode'], arm['max_abs_diff'],
          arm['first_divergence_step'], arm['load_bearing'])
```

Representative tiny-model result (fp32, seed 3, `T=14`, `prefill_len=8`,
`csa_topk=100`; values depend on the seed / geometry and are not a benchmark):

| part | mode | max abs diff | rel L2 |
|---|---|---:|---:|
| `memory_state` | `at_prefill` | 3.6e-2 | 2.4e-2 |
| `memory_state` | `every_step` | 5.1e-2 | 3.2e-2 |
| `memory_lookback` | `at_prefill` | 2.8e-2 | 1.8e-2 |
| `memory_lookback` | `every_step` | 4.0e-2 | 3.0e-2 |
| `csa_cache` | `at_prefill` | 2.6e-1 | 1.6e-1 |
| `csa_cache` | `every_step` | 3.0e-1 | 2.5e-1 |
| `hca_cache` | `at_prefill` | 3.1e-1 | 1.7e-1 |
| `hca_cache` | `every_step` | 3.6e-1 | 2.7e-1 |

Every family diverges immediately (`first_divergence_step == 0`), so the
equivalence tests are not passing because a cache is bypassed.

## Causality audit

`decode_causality_audit` runs three checks on the same weights:

1. **Future independence** — replace every token at/after `cut` with fresh
   noise; outputs at `t < cut` must not move.
2. **Non-vacuous perturbation** — the same noise must change at least one
   suffix element. Without this control, a zero-difference "causality" check
   could simply be ignoring the input.
3. **Incremental prefix** — a prefix decoded as a prefill plus optional
   continuation chunks and then token-by-token must match the one-shot causal
   forward on `t < cut`. `prefix_chunks=(5, 5)` exercises the
   `forward_chunk` continuation path instead of only the first prefill.

```python
from decode_state_ablation import decode_causality_audit

audit = decode_causality_audit(decoder, x, cut=10, prefix_chunks=(5, 5))
assert audit['prefix_unchanged']
assert audit['future_perturbation_nonvacuous']
assert audit['incremental_prefix_matches_one_shot']
```

The same contract is applied to the streaming tensor-parallel reference in
`test_tp_parallel.py::test_hybrid_streaming_tp_causality`: two chunk schedules
(prefill + continuation) run on the original and future-perturbed streams, the
prefix must match, the suffix must move, and the TP prefix must agree with the
unsharded one-shot forward.

## Scope and honesty

* This is a correctness/validity audit, not a trained accuracy ablation and
  not a latency measurement. The divergence magnitudes depend on the random
  initialization and the geometry; the contract is the sign of the effect
  (`load_bearing`) plus the exact prefix.
* `memory_state` and `memory_lookback` are route-neutral: the same helper
  drops the KDA, Mamba-2, or RWKV-7 state through ``reset_conv_state``. The
  reference tests use the KDA route.
* The audit calls the reference decoder token-by-token, so it is intentionally
  small. It is meant to run in CI and on a laptop, not to replace Exp 6's
  decode benchmark.
