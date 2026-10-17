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

## 1. Operator layout and routes

```
Default (3:1:1):                    Switchable memory route:

KDA  KDA  KDA  CSA  HCA            Mamba2 Mamba2 Mamba2 CSA HCA
```

* The three operators and the default 3:1:1 layout are preserved; every
  config, checkpoint, and result reproduces bit-for-bit
  (`test_hybrid_stack_memory_routes` pins the layout string, parameter
  names, and numerics of `HybridKCHAttention` and its
  `HybridAttentionStack` alias).
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
wiring, identical input/output dimensions (`d_model`). No outer MLP is
used: the operators embed their projections inside the mixer, so an
outer MLP would change every baseline's numerics. The fairness contract
the benchmarks rely on ("same boundary, only the mixer differs") holds
without it; if an outer MLP is ever added it must land behind a
default-off config flag with the same backward-compatibility proof.

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
tensor (axis 0 = layer index; the buffer is exposed under the
`_kda_state` name) plus a per-layer conv-state list. CSA/HCA
keep their existing incremental decoding caches — their state grows with
context (compressed entries), which is why the hybrid's total cache is
NOT sequence-independent even though the memory operators' states are
(§6, `run_kv_cache.kv_cache_breakdown`).

Correctness pins (all in `run_correctness.py`): full forward == prefill +
single-token decode; one-shot == segmented prefill (outputs AND final
states); future-token perturbation leaves past outputs unchanged;
sub-conv-kernel chunks and T=0; interleaved requests with different batch
sizes keep unpolluted state; weight save/load roundtrip; both hybrid
routes' prefill+decode == full-sequence equivalence (~1e-7). The CSA
indexer's hard top-k is deterministic and size-independent (ties break on
ascending block index, `ops_csa._canonical_topk_indices`), so the
prefill/decode selection agreement is independent of the score-tensor
size; the `csa_topk=100` select-all protocol is retained as
defense-in-depth in the legacy decoding-cache tests.

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

The reference Mamba-2 shares part of its parameter layout with the
official module (`A_log`, `D`, `out_proj`, and the dt bias, which
`transfer_reference_state` maps between the `dt_proj.bias` and `dt_bias`
spellings across `mamba_ssm` revisions). The official module fuses
`[z, x, B, C, dt]` into `in_proj` and convolves `xBC` jointly, whereas the
reference keeps the Mamba-1-style `x_proj` + `dt_proj` head and an x-only
short conv, so those tensors are structurally incompatible;
`transfer_reference_state` raises a `ValueError` listing them rather than
silently producing wrong outputs. The parity test
(`test_mamba2_official_backend_parity`) copies the transferable weights
into the official module and requires output/state agreement within
dtype-scaled tolerance (reduction orders differ; bit-equality is not the
contract). It self-skips with a recorded note on CPU-only boxes.

The two backends also expose different runtime state layouts: the SSM
state is identical (`[B, nheads, headdim, d_state]`), but the official
conv state spans the joint xBC channels over `d_conv` steps
(`[B, d_inner + 2 * ngroups * d_state, d_conv]`) and cannot be continued
from the reference's x-only `[B, d_inner, d_conv - 1]` buffer. The
official parallel path always starts from a fresh prefix, so chunked
prefill continuation is reference-only; the official wrapper rejects
`T > 1` with an incoming state with an actionable error.

Benchmark interpretation follows the two-set rule: reference rows are
correctness baselines, never speed rankings; accelerated rows are
"current implementation" system numbers, not architecture-intrinsic
costs.

### 4.1 The SSD duality oracle (reference-scan correctness)

The reference recurrence's other correctness chains share its code path
or self-skip on CPU: prefill/step/segmented-prefill all run the same
scan, and the official-backend parity test needs CUDA + `mamba-ssm`. A
semantic error in the recurrence (wrong decay convention, wrong causal
alignment, wrong group-to-head mapping) could therefore survive CI
unnoticed. The **state-space duality** gives an independent check.

`ops_mamba.quadratic_mamba2_scan` evaluates the same operator as causally
masked quadratic attention with a semi-separable decay
(arXiv:2405.21060 Sections 3-4). With per-head cumulative log-decays
`G_t = sum_{r<=t} dt_r * A`:

```
y_t = sum_{s<=t} exp(G_t - G_s) * dt_s * (C_t . B_s) * x_s
      + exp(G_t) * (C_t . S_0) + D * x_t
S_T = exp(G_{T-1}) * S_0 + sum_s exp(G_{T-1} - G_s) * dt_s * (x_s (x) B_s)
```

It is implemented from `cumsum` + matmuls + one elementwise `exp` — no
sequential recurrence — and shares the input contract with the reference
through ONE validator (`_validate_mamba2_scan_inputs`), so the two views
cannot drift apart in what they accept. Non-causal pairwise entries are
masked **before** the `exp` (a positive exponent would overflow and
`inf * 0 = NaN` would poison the mask). It is O(T^2) by design: a test
oracle and the semantic reference for future chunked/parallel SSD
kernels, never a production path.

`test_ssd_duality.py` pins, to fp64 machine precision:

* hand-computed closed forms (two-step y/state, initial-state decay),
* forward parity across group counts 1/2/4, with/without initial state,
  at T=13 and T=64,
* contract parity: T=0, `output_final_state=False`, fp32/fp16 dtype
  passthrough, and identical rejection of malformed inputs,
* gradient parity for every input (x, dt, B, C, A, D, S0).

The same validator also rejects `B`/`C` batch or seqlen shapes that
differ from `x`, instead of silently broadcasting/indexing them.

### 4.2 The chunked SSD path (associative scan over chunks)

`ops_mamba.chunked_mamba2_scan` is a third evaluation of the same operator
(not an oracle — a drop-in for the recurrence). It splits the sequence into
chunks of `chunk_size` and evaluates the duality expansion where it is
cheap and well-conditioned:

* **intra-chunk outputs** with the masked quadratic form restricted to the
  chunk (`O(C^2)` per chunk instead of `O(T^2)` globally),
* **inter-chunk state** via the affine chunk recurrence
  `S_{n+1} = a_n S_n + b_n`, with the SCALAR per-head transition
  `a_n = exp(sum_chunk dt*A)` and the chunk source `b_n`, resolved by a
  Hillis-Steele associative prefix scan over chunks
  (`ceil(log2 NT)` batched levels). This is the
  Mamba-2 analogue of `ops_kda._assoc_scan_chunk_states`; because the SSD
  decay is scalar-per-head, the scan combine is elementwise.

**Log-space gating.** All exponents are anchored at the *state before the
chunk starts*: with `G_t = sum_{r<=t} dt_r A` (non-increasing because
`A < 0`, `dt >= 0`), the code forms `g_c = G_{t0+c} - G_{t0-1}` per chunk,
so every intra-chunk exponent `g_c - g_s` (`s <= c`) is `<= 0`, the chunk
transition is one `exp(g_last)`, and the inter-chunk readout is
`exp(g_c)`. The global dual form differences two large cumulative
log-decays; the chunk anchor keeps every exponent's magnitude bounded by
the chunk's own decay budget.

The path is selected by `HybridConfig.mamba_chunk_size` (`0` = sequential
default; positive = chunked; reference backend only — the official kernels
chunk internally, and the contradictory combination is rejected at config
time). `Mamba2Mixer.scan_mode` records the resolved mode for provenance.
`test_chunked_ssd.py` pins forward/gradient parity against BOTH existing
views at fp64 machine precision, chunk-size invariance, huge-cumulative-
decay stability (up to `|G| ~ 8e4`), the `ceil(log2 NT)` scan-depth
contract, TorchScript parity of the scan core, and the mixer integration.

### 4.3 Latency profiling the scan: model vs measurement

`latency_profile.py` provides an operator-level profiler for the scan:

* **Analytic side** (`roofline.ssd_core_flops_breakdown`): dominant-MAC
  FLOPs for the four chunked phases, at a per-batch-element granularity —
  `intra_chunk_attention` (the causal masked quadratic attention operator
  restricted to a chunk, `2*Tp*C*H*(headdim + d_state)`),
  `chunk_source` and `inter_chunk_readout` (each
  `2*Tp*H*headdim*d_state`), and `assoc_scan`
  (`2*ceil(log2 NT)*Tp*H*headdim*d_state/C`), with `Tp = NT*C` the padded
  length. The sequential mode is the flat `6*T*H*D*N` state work, the same
  shape as `kda_core_flops`. `core_bytes('ssd', ...)` adds the
  compulsory-traffic counterpart (activation I/O + per-step or per-chunk
  state read/write).
* **Measured side** (`latency_profile.profile_scan_modes`): seeded inputs at
  the `mamba2_scan` core geometry (4 heads × 32×32 state), warmup +
  repeated timing with median/min/max/std, an injectable clock/runner, and a
  **parity gate on every row** (max deviation from the sequential reference
  at the same T — a profile can never silently measure a divergent
  operator). `--quadratic` adds the global dual-form oracle as the
  full-attention reference point; `--json` writes a versioned
  `results_schema` envelope (`kind="mamba2_scan_latency_profile"`).

```
python3 latency_profile.py --lengths 128,256,512,1024,2048 \
    --chunk-sizes 16,32,64,128 [--quadratic] [--json results/scan_latency.json]
```

The profiler prints the work-model argmin next to the measured optimum
rather than trusting the model: the analytic model has no per-call
overhead and no BLAS-efficiency term, so the two can differ.

`test_latency_profile.py` pins the model formulas (hand-computed phases,
chunk-size monotonicity, tie-breaking), the harness mechanics (fake clock /
fake runner: exact median/min/max/std, sync bracketing, parity-gate
detection of an injected divergence), and one real-torch end-to-end profile
including the JSON round-trip.

**Roofline join.** `run_roofline.py` (Exp 7) joins the Mamba-2 route as
`mamba2_scan` → the SSD core model above; `mamba2_layer` /
`hybrid_mamba2` → `run_kv_cache` at the recorded realization geometry
(the benchmark records `csa_nIh`/`csa_cI` on the hybrid_mamba2 workload
so the join is exact).

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

### 6.1 Mixed-precision parity sweep (`precision_sweep.py`)

The correctness tests pin the operators in fp32; `train_lm_autodl.py`
deploys them under bf16/fp16 autocast. `precision_sweep.py` closes that
gap with a dedicated evaluation harness over the seven core paths
(KDA `rec`/`chunk`/`assoc`, Mamba-2 `naive`/`quad`/`chunk`, RWKV-7):

* every (path, dtype, seed) is compared against an **fp64 oracle of the
  same path** on the same generator draw (the tested input is that draw
  rounded to the dtype), and errors are reported in units of the dtype's
  unit roundoff `u` — fp16's grid is ~150x finer than bf16's, so an
  absolute number means nothing without it;
* a **seed sweep** (default 3) reports max/median/spread and the
  `worst_seed`, so a budget failure is reproducible and a lucky single
  seed cannot green-light a regression;
* the **dtype contract** is checked on every row (output in the input
  dtype; state in compute precision — fp32 for fp16/bf16/fp32 inputs);
* two modes: `explicit` (inputs cast to the dtype; ops upcast to fp32) and
  `autocast` (fp32 storage + `torch.autocast`, the real AMP path);
* budgets: T-independent `factor*u` for storage dtypes (observed ~2u on
  the shipped cores), and the textbook `accum_factor*T*u` round-off
  envelope for compute dtypes (the quadratic/chunked SSD state measured
  0.87*T*u at T=1024).

`--json` writes a versioned envelope (`kind='precision_sweep'`) whose
aggregate rows are checked by a registered `results_schema` contract
(verdict/flag consistency, max ≥ median, unique groups). The shared dtype
contract — every core path returns its state in compute precision, never
the autocast dtype of the state einsum, which is autocast-eligible — is
pinned across all paths by `test_precision_sweep.py`.

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
standard budgets) | `full` (7 seeds across the multi-seed experiments).
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
  CUDA); its adapter targets the released `mamba_ssm` API
  (`InferenceParams` in `mamba_ssm.utils.generation`, states threaded via
  `inference_params.key_value_memory_dict[layer_idx]`) and its parity test
  self-skips with a recorded note until it runs on a CUDA box. The
  reference path is fully tested. Because the official module's parameter
  and conv-state layouts are structurally different (see section 4), the
  parity test can only compare the transferable subset; the reference
  backend remains the weight-compatible baseline.
* The reference Mamba-2 scan is a sequential Python loop (like the KDA
  reference). Its *semantics* are independently pinned by the SSD
  dual-form oracle (section 4.1), so the CPU-only official-backend gap
  does not leave the reference unverified. The chunked path (section
  4.2) is an alternative reference evaluation for hosts without
  `mamba-ssm`; it is parity-pinned to both other views but is not wired
  into the main experiment rows yet (`mamba_chunk_size` defaults to 0).
  Its device-level latency is measured by `latency_profile.py` (section
  4.3); those numbers are host- and torch-build-specific and must not be
  compared across machines.
* Hybrid-decode equality tests use the `csa_topk=100` protocol (select
  every visible block); with the deterministic ascending-index tie-break
  in `ops_csa._canonical_topk_indices` this is defense-in-depth, not a
  workaround for a tie-breaking artifact (a numerical artifact, identical
  for both memory routes).
* The smoke profile validates the pipeline before a full run.
* The precision sweep's budgets are smoke-level regression envelopes, not
  numerical-error theorems, and its autocast numbers are torch-build- and
  host-specific: CPU AMP is bf16 in practice (CPU fp16 autocast is
  emulated and excluded by default), and the harness does not run on
  CUDA in this sandbox. The fp64 oracle pins semantics, not the deployed
  kernel stack.
