# Switchable KDA / Mamba-2 Hybrid Framework — Design Document

This document records the design decisions behind the framework extension
that makes the recurrent memory operator (KDA vs. Mamba-2) a switchable
route while keeping CSA (sparse retrieval) and HCA (global aggregation)
as independently ablatable operators. It is the implementation-side
companion to the experiment design; the code is the source of truth for
behaviour, and every claim below is pinned by a test in
`run_correctness.py`.

**Core research question.** Under a matched training budget, do sparse
retrieval (CSA) and heavily-compressed global attention (HCA) improve the
long-context ability of fixed-size recurrent-state models — and does the
benefit transfer across BOTH KDA and Mamba-2?

---

## 1. What stayed, what was added

```
Existing (unchanged default):       Added route:

KDA  KDA  KDA  CSA  HCA            Mamba2 Mamba2 Mamba2 CSA HCA
```

* The three operators and the default 3:1:1 layout are untouched; every
  historical config, checkpoint, and result reproduces bit-for-bit
  (`test_hybrid_stack_memory_routes` pins the layout string, parameter
  names, and numerics of `HybridKCHAttention` vs. the renamed
  `HybridAttentionStack`).
* `HybridConfig.memory_type` (`'kda'` | `'mamba2'`, default `'kda'`)
  selects the operator that fills the memory slots. In ratio mode the
  `n_kda` slots ARE memory slots; in explicit-layout mode the abstract
  `'memory'` token resolves through the same field.
* One memory route per model: a layout mixing concrete `'kda'` and
  `'mamba2'` kinds is rejected at config validation. The comparison
  design treats the memory operator as a per-model route, not a per-layer
  mix; a mixed stack would also break the single stacked-state tensor.

## 2. Unified layer structure — only the token mixer differs

Every sub-layer in `HybridAttentionStack` shares the same outer
structure:

```
x = x + mixer(norm(x))
```

identical LayerNorm placement (`_Fp32LayerNorm`), identical residual
wiring, identical input/output dimensions (`d_model`). No outer MLP was
added in this revision: the historical operators embed their projections
inside the mixer, so adding an outer MLP would have changed every
existing baseline's numerics. The fairness contract the benchmarks rely
on ("same boundary, only the mixer differs") holds without it; if a
future revision adds an outer MLP it must land behind a default-off
config flag with the same backward-compatibility proof.

Mamba-2 keeps the official mixer module's structure (no RoPE — its
recurrent state carries positions implicitly like KDA's), and the
official `Mamba2` module is wrapped at the MIXER level only, never as a
full block, so the stack does not double-wrap norms or residuals.

**Same `d_model` does not mean same capacity.** KDA's head geometry
(`HV×K×V`) and Mamba-2's internal expansion (`expand`, `headdim`,
`d_state`) differ; at `d_model=32` the 10-layer matrix groups measure
~53K (KDA-A) vs. ~105K (MAMBA2-D) parameters. Every result row records
`n_params`, and the budget-matching follow-up (§5) exists precisely to
close that gap before any capacity claim.

## 3. The mixer interface

The stack layers implement the repo's memory-layer contract:

| Method | Contract |
|---|---|
| `forward(x, state=None)` | stateful wrapper; persists conv state on the layer |
| `forward_functional(x, state, conv_state, *, detach_lookback=True)` | pure path for DDP / `torch.compile` / checkpointing / concurrent inference |
| `reset_conv_state()` | clear streaming conv context between sequences |
| `prefill(x)` / `step(x_t, state)` | inference protocol; returns detached state dicts |

State layouts (per layer):

| Module | Recurrent state | Conv state |
|---|---|---|
| KDA | `[B, HV, K, V]` | `[B, 2, d]` (raw hidden lookback) |
| Mamba-2 | `[B, nheads, headdim, d_state]` | `[B, d_inner, d_conv-1]` (in_proj features) |
| CSA | — (compressed blocks in `CSADecodingCache`) | — |
| HCA | — (compressed blocks in `HCADecodingCache`) | — |

The stack threads the memory layers' recurrent states as ONE stacked
tensor (axis 0 = layer index; the historical `_kda_state` buffer name is
kept for source compatibility) plus a per-layer conv-state list. CSA/HCA
keep their existing incremental decoding caches — their state grows with
context (compressed entries), which is why the hybrid's total cache is
NOT sequence-independent even though the memory operators' states are
(§6, `run_kv_cache.kv_cache_breakdown`).

Correctness pins (all in `run_correctness.py`): full forward == prefill +
single-token decode; one-shot == segmented prefill (outputs AND final
states); future-token perturbation leaves past outputs unchanged;
sub-conv-kernel chunks and T=0; interleaved requests with different batch
sizes keep unpolluted state; weight save/load roundtrip; both hybrid
routes' prefill+decode == full-sequence equivalence (~1e-7, with the
documented `torch.topk` tie-breaking caveat handled by the `csa_topk=100`
test protocol).

## 4. Backends

| | KDA | Mamba-2 |
|---|---|---|
| reference | `ops_kda` (this repo) | `ops_mamba.Mamba2Mixer` (this repo) |
| accelerated | FLA (`kda_backend='fla'`) | `mamba_ssm` (`mamba_backend='official'`) |
| auto | FLA on CUDA when installed | official on CUDA when installed |

Contract (mirrored for both operators): an EXPLICITLY requested
accelerated backend errors loudly when unavailable — it never silently
falls back and keeps reporting accelerated numbers. `'auto'` may fall
back, and the resolution is recorded on the mixer
(`effective_backend`) and in every result row
(`mamba_effective_backend`), so provenance is auditable.

The reference Mamba-2 names its parameters exactly like the official
module (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `A_log`, `D`,
`out_proj`), so weights transfer between backends;
`transfer_reference_state` tolerates the `dt_proj.bias` vs. `dt_bias`
layout drift across `mamba_ssm` revisions. The parity test
(`test_mamba2_official_backend_parity`) copies reference weights into
the official module and requires output/state agreement within
dtype-scaled tolerance (reduction orders differ; bit-equality is not the
contract). It self-skips with a recorded note on CPU-only boxes.

Benchmark interpretation follows the two-set rule: reference rows are
correctness baselines, never speed rankings; accelerated rows are
"current implementation" system numbers, not architecture-intrinsic
costs.

## 5. The eight-group main experiment

`ABL_MODE=matrix` (run_ablation.py) crosses the memory route with four
layout groups at FIXED total depth (`ABL_TOTAL_LAYERS`, default 10 =
the 5-layer unit twice):

| Group | Per-5-layer unit | Purpose |
|---|---|---|
| A | `R R R R R` | pure recurrent baseline |
| B | `R R R CSA R` | CSA replaces a fixed recurrent slot |
| C | `R R R R HCA` | HCA replaces a fixed recurrent slot |
| D | `R R R CSA HCA` | full hybrid |

yielding `KDA-A…D` and `MAMBA2-A…D`. Replacing a slot (rather than
deleting a layer) keeps depth constant, so CSA's and HCA's contributions
are ablated independently of depth and of the memory operator.

This is a fixed-STRUCTURE comparison, not a matched-parameter one. The
two-step protocol from the design:

1. **Fixed-structure screening** (what `ABL_MODE=matrix` runs): same
   `d_model`, depth, and outer structure; report the actual `n_params`
   per row.
2. **Budget matching** (follow-up): for the surviving combinations,
   tune internal dimensions (MLP width / expansion / state size) until
   parameter counts agree within ~±5%, and report the residual.

The `budget` matrix preset (`ABL_MATRIX=budget`) is the hook for the
cache-byte side of that follow-up: the D layout at equal
compressed-entry budgets (`1/csa_m + 1/hca_m2` matched within families).

Follow-up ablation presets (design §10): `csa_csa` (is HCA better than a
second CSA layer? A/D/E), `position` (alternating vs. back-loaded
placement at identical operator counts — fix depth AND ratio, vary only
position).

## 6. Evaluation and measurement

Quality probes (`run_quality.py`, all generator-isolated so every
operator sees identical batches per seed):

| Task | Measures | Default |
|---|---|---|
| `mqar` | associative recall (canonical) | seq 16, vocab 16 |
| `kv_update` | memory UPDATE under interference (latest value wins) | seq 16, vocab 32 |
| `long_range` | retrieval at controlled distance (per-sequence `distance` meta) | seq 64, vocab 16 |
| `multi_agg` | multi-site aggregation (majority category; no single-position shortcut) | seq 33, vocab 4 |

`MQAR_TASKS` selects tasks (`'mqar'` default for full backward
compatibility; `'all'` or a comma list enables the probes). MQAR rows
keep `results/exp4_mqar.json`; the other tasks write
`results/exp4b_quality_tasks.json`. `MQAR_OPS` selects operators
(default now includes `mamba2`).
**Language-model NLL stays the decisive quality metric** — the synthetic
probes are screening instruments, and length extrapolation (train
length ≠ test length) is reported separately per the design.

Latency (`run_benchmark.py`) / decode (`run_decoding.py`) / cache
(`run_kv_cache.py`) rows exist for both routes at matched boundaries:
`mamba2_scan` is capacity-matched to `kda_rec` (4 heads × 32×32 state);
`mamba2_layer` mirrors the csa/hca single-layer boundary;
`hybrid_mamba2` mirrors the hybrid stack boundary with identical CSA/HCA
geometry. `kv_cache_breakdown` splits every row into recurrent / conv /
compressed / window / indexer / partial buckets, and the two new figures
(`quality_vs_latency.png`, `quality_vs_cache.png`) draw the frontier.

## 7. Run tiers and decision rules

`RUN_ALL_PROFILE` = `smoke` (plumbing validation: tiny budgets + a
1-seed 5-step pass over all eight matrix groups) | `screen` (default =
historical budgets) | `full` (7 seeds across the multi-seed experiments).
Presets only touch env vars the caller did NOT export.

Result recording follows the design's minimum provenance: git commit,
seed, full config, dataset/tokenizer ids, actual token counts, total
parameters, requested AND effective backends, device/software/precision,
validation metrics, prefill/decode latency, actual cache bytes, peak
memory; multi-seed reports carry mean/std/CI and per-seed rows, with the
run (not the sample) as the statistical unit.

Final default-route decision (from the experiment design):

| Result | Choice |
|---|---|
| KDA more stable, Mamba-2 no cost win | keep KDA default |
| Mamba-2 similar quality at clearly lower cost | ship a Mamba-2 performance default |
| HCA helps both memory operators | prioritise HCA's cross-module role |
| HCA helps only in specific tasks/lengths | scope conclusions to those conditions |
| HCA loses to more CSA / more recurrence | simplify; drop the three-operator design |

The first deliverable's goal is to answer "is the swap worth it" fairly —
not to pre-empt the answer by changing the default route.

## 8. Known limitations

* The official Mamba-2 backend could not be executed in this sandbox (no
  CUDA); its adapter and parity test are written defensively against
  `mamba_ssm >= 2.2` and self-skip with a recorded note until they run on
  a CUDA box. The reference path is fully tested.
* The reference Mamba-2 scan is a sequential Python loop (like the KDA
  reference): correctness-first, not throughput. Speed conclusions must
  use the official-backend rows.
* Hybrid-decode equality uses the `csa_topk=100` protocol because of the
  pre-existing `torch.topk` tie-breaking artifact documented in the
  README (a numerical artifact, not a correctness bug; identical for
  both memory routes).
* No experiments were run as part of landing this framework — by design.
  The smoke profile exists precisely so the next runner can validate the
  whole pipeline cheaply before committing GPU time.
