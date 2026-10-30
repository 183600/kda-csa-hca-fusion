# KDA-CSA-HCA Fusion

**English** · [中文](#中文版)

This repository provides a readable, correctness-first PyTorch reference
implementation and experimental harness for a hybrid linear-attention stack.
The stack pairs a switchable recurrent-memory operator with two complementary
retrieval operators from recent long-context / efficient-attention literature:

* **KDA** — Kimi-style Delta Attention (recurrent delta-rule with per-channel
  log-decay). Bulk token mixing, O(1) state per head.
* **Mamba-2** — SSD (arXiv:2405.21060) mixer with per-head matrix state;
  the second selectable recurrent-memory route (`memory_type='mamba2'`),
  same O(1)-state decoding contract as KDA.
* **CSA** — DeepSeek-V4 Compressed Sparse Attention (overlapped two-branch KV
  compression + lightning indexer that selects top-k compressed blocks per
  query). Sparse long-range retrieval.
* **HCA** — DeepSeek-V4 Heavily Compressed Attention (single-branch heavy
  compression, dense MQA over a tiny set of compressed entries, optional
  sliding window). Cheap global context.

The hybrid stack interleaves them in a `3:1:1` memory:CSA:HCA ratio by default
(see `ops_fused.HybridAttentionStack`; `HybridKCHAttention` is kept as the
backward-compatible name). The recurrent memory slots are switchable between
KDA and Mamba-2 via ONE config field — see
[docs/mamba2-framework.md](docs/mamba2-framework.md) for the design and the
eight-group experiment matrix.

> **What this is.** A research-grade, correctness-first reference
> implementation. Numerical edge cases (causality, dtype, empty sequences,
> sink stability, state management across decode steps) are handled more
> carefully than in a typical "research dump" — see the extensive regression
> suite in `run_correctness.py`.
>
> **What this is NOT.** It is **not** a production kernel. KDA uses a Python
> `for` loop over time; CSA/HCA use `unfold`-based sliding windows; nothing
> is fused or Tritonized. Latency numbers from `run_benchmark.py` measure
> this naive reference and **must not** be compared to FLA / FlashAttention
> / DeepSeek production kernels. See the *Limitations* section below.
>
> **Accelerated paths.** The naive paths above remain the default for
> correctness / readability, but the repo ships two opt-in accelerated
> wrappers that preserve the exact numerical contract:
>
> - `ops_kda.compiled_recurrent_kda` — `torch.compile` wrapper around
>   `naive_recurrent_kda`, cached per (shape, dtype, requires_grad) signature.
>   Best for moderate `T` (≥1024); typical speedup 5–20× on GPU.
> - `ops_kda.scripted_chunk_kda` — `torch.jit.script` wrapper around the
>   inner per-chunk loop of `naive_chunk_kda`, with a graceful fallback to
>   the eager path if TorchScript rejects the input.
> - `ops_csa._sliding_window_attention` — standalone memory-efficient
>   sliding-window attention; it auto-chunks when `T * win * c > 8M`
>   elements, keeping peak memory at `O(chunk_t · win · c)` instead of
>   `O(T · win · c)`. `naive_csa` / `naive_hca` use the unfold-based
>   `_sliding_window_scores` instead, so the window entries share one joint
>   softmax with the compressed-branch scores and the attention sink.
> - `naive_csa` itself fuses the 6 `F.linear(H, W_*)` calls
>   (`W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx`) into a single matmul
>   via `torch.cat` + `tensor.split`, reducing kernel-launch overhead by 6×.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running the experiments](#running-the-experiments)
- [Tests](#tests)
- [Result files](#result-files)
- [Fairness notes](#fairness-notes-read-before-interpreting-figures)
- [Limitations](#limitations)
- [Papers](#papers)
- [License](#license)

---

## Repository layout

```
.
├── ops_kda.py             # KDA reference: naive_recurrent_kda, naive_chunk_kda
├── ops_kda_backend.py     # Optional KDA backend adapter: reference / FLA / auto
├── ops_kda_autograd.py    # Custom autograd for the recurrent KDA path
├── ops_mamba.py           # Mamba-2 (SSD) reference mixer: scan + unified prefill/step
├── ops_mamba_backend.py   # Optional Mamba-2 backend adapter: reference / mamba_ssm / auto
├── ops_rwkv.py            # RWKV-7 reference mixer (third memory route)
├── ops_rwkv_backend.py    # Optional RWKV-7 backend adapter
├── ops_rope.py            # Partial RoPE (DeepSeek-V4 §2.3.3) for CSA/HCA
├── ops_norm.py            # RMSNorm reference helpers shared by the operators
├── ops_csa.py             # CSA: compress_kv (overlapped), lightning indexer, naive_csa
├── ops_hca.py             # HCA: heavy compression + dense MQA + SW, naive_hca
├── ops_decoding_cache.py  # CSADecodingCache / HCADecodingCache / Mamba2DecodingState
├── ops_fused.py           # HybridConfig + build_mixer factory + HybridAttentionStack
│                          #   (HybridKCHAttention kept as the compat name)
├── checks/                # Correctness checks grouped by subsystem
├── quality/               # MQAR task, model, training, and statistics helpers
├── docs/                  # Design notes and experiment contracts
├── docs/mamba2-framework.md # Switchable KDA/Mamba-2 framework design document
├── COMPARISON.md          # ops_* vs early toy implementation comparison
├── run_correctness.py     # Regression checks (custom runner; pytest-importable)
├── run_benchmark.py       # Exp 2: latency vs. sequence length (with op_boundary metadata)
├── run_quality.py         # Exp 4: MQAR + kv-update/long-range/multi-agg probes (multi-seed)
├── run_ablation.py        # Exp 5: ratio sweep (ABL_MODE=ratio) | eight-group matrix (matrix)
├── run_decoding.py        # Exp 6: prefill + per-token decode latency (both memory routes)
├── run_kv_cache.py        # Exp 3: analytic KV-cache + FLOPs accounting (+ state breakdown)
├── method_analysis.py     # Headwise prototype (CSA simplified to dense for the demo)
├── make_figures.py        # Generate figures/* from results/* (incl. quality-vs-cost)
├── run_all.py             # Single-entry runner (RUN_ALL_PROFILE=smoke|screen|full)
├── experiment_design.py   # Pre-registered plans, prospective power/MDE, runtime audit
├── kaggle_setup.py        # CUDA bootstrap + shared utilities (sanitize_for_json, parse_int_env)
├── near_duplicates.py     # Token n-gram MinHash/LSH near-duplicate leakage detector
├── docs/data-integrity.md # Near-duplicate + fingerprint data-integrity contract
├── test_figures.py        # Tests for the figure generation pipeline
├── results/               # Generated JSON outputs (gitignored; regenerate via run_all.py)
├── figures/               # Generated PDF/PNG outputs (gitignored; regenerate via make_figures.py)
├── requirements.txt
├── pyproject.toml         # `pip install -e .` makes the modules importable as a package
└── README.md              # this file
```

---

## Installation

**Requires Python 3.10–3.13** (the source uses PEP 604 `X | None` union syntax,
and the full regression suite is validated on the 3.10–3.13 interpreter
matrix). The `pyproject.toml` declares `requires-python = ">=3.10,<3.14"`.

```bash
# 1. Clone
git clone https://gitee.com/qwe12345678/kda-csa-hca-fusion.git
cd kda-csa-hca-fusion

# 2. Install the package and its runtime dependencies
pip install -e .        # installs torch, einops, matplotlib, numpy, scipy
#    OR, if you only need the runtime deps without the dev toolchain:
#    pip install -r requirements.txt

# 3. (Optional) Install dev tooling (pytest, pytest-xdist, mypy, ruff):
pip install -e .[dev]

# 4. (Optional) Install the FLA KDA backend. This does not change the
#    default reference experiments; it is selected explicitly in HybridConfig.
pip install -e '.[fla]'
```

The default `HybridConfig(kda_backend="reference")` preserves the original
PyTorch KDA implementation. To opt into FLA for GPU experiments:

```python
from ops_fused import HybridConfig
cfg = HybridConfig(kda_backend="fla")
```

Use `kda_backend="auto"` to try FLA only for CUDA tensors when it is installed
and otherwise fall back to the reference path. For the standalone quality,
decoding, and benchmark runners, set the same choice before launching:

```bash
KDA_BACKEND=fla python run_benchmark.py
KDA_BACKEND=auto python run_decoding.py
```

The adapter passes the already normalized q/k, log-space gate, and post-sigmoid
beta values without applying those transformations a second time. CSA/HCA
remain on the repository reference path; no Hugging Face or vLLM dependency is
required for the normal experiments.

After `pip install -e .`, the experiment scripts can be run as modules:

```bash
python -m run_correctness       # or: python run_correctness.py
python -m run_benchmark
python -m run_all
```

### Quick start

```python
import torch
from ops_fused import HybridConfig, HybridAttentionStack

cfg = HybridConfig(d_model=64, n_heads_qk=2, n_heads_v=2,
                   head_dim_k=16, head_dim_v=16, kda_chunk_size=4)
model = HybridAttentionStack(cfg, total_layers=5).eval()

x = torch.randn(2, 8, cfg.d_model)
with torch.no_grad():
    y = model(x)

print(y.shape)  # torch.Size([2, 8, 64])
```

`HybridAttentionStack` applies the configured memory/CSA/HCA layout and
returns a tensor shaped `[batch, sequence, d_model]`. See
[Switchable KDA / Mamba-2 memory route](#switchable-kda--mamba-2-memory-route-quick-start)
for layout overrides and per-op configuration.

### Kaggle

Upload the repo as a Dataset, then in a notebook cell:

```python
!pip install -q einops matplotlib
import sys; sys.path.insert(0, '/kaggle/input/<your-dataset-name>')
%run /kaggle/input/<your-dataset-name>/run_all.py
```

For GPU runs on Kaggle T4, follow the CUDA bootstrap procedure documented in
`kaggle_setup.bootstrap_kaggle_cuda()` (install the CUDA wheel, then restart
the kernel — `setup_kaggle()` only *verifies* CUDA availability, it does not
install in-place). On current Kaggle images torch ships WITH CUDA
preinstalled, so the bootstrap is normally a no-op. The bootstrap pins an
exact torch version **matched to the detected CUDA wheel index**
(`_BOOTSTRAP_TORCH_PINS` in `kaggle_setup.py`): e.g. `torch==2.6.0` for
cu118/cu124/cu126, `torch==2.5.1` for cu121 (its last release), and
`torch==2.7.1` for cu128 (where cu128 wheels start) — and it verifies the
installed wheel actually carries a `+cuXXX` build tag before telling you to
restart, so a CPU-only fallback install can no longer masquerade as success.

Dependency policy for preinstalled environments: `run_all.py` no longer
hard-rejects torch/numpy/matplotlib versions outside the historically-tested
bounds — it WARNS and continues (the operator code has no version-sensitive
API dependencies for correctness; only softmax-baseline benchmark latency may
drift across torch releases). Missing packages are still installed. Set
`STRICT_DEPS=1` to restore the hard version gate for bit-reproducible
historical reruns. Correspondingly, `torch` keeps only a permissive upper
bound (`<3`) in `requirements.txt` / `pyproject.toml` so `pip install -e .`
never downgrades Kaggle's preinstalled CUDA torch; pin `torch==2.6.*`
yourself to reproduce the committed historical numbers. The advisory ranges
for numpy / matplotlib / scipy are extended whenever the full suite is
re-validated on newer releases (last: 2026-08, numpy 2.5.2 / matplotlib
3.11.1 / scipy 1.18.1 / torch 2.13.0).

Resource footprint on the Kaggle free tier (2×T4, 30h/week quota):
`run_all.py` with the default knobs is CPU-scale (~10 minutes) — GPU is not
required. The default `train_lm_autodl.py --kaggle` profile (d=256, 5 layers,
seq 512, 500 optimizer steps) takes ~40 minutes on 2×T4 and a few GB of GPU
memory — comfortably inside the free quota. `train_lm_autodl.py` also runs
without `transformers`/`datasets` (char-level fallback tokenizer over
synthetic text), so the pipeline executes even on a bare CPU session.

---

## Running the experiments

| Experiment | Script | Output | Wall-clock (CPU) |
|---|---|---|---|
| 1. Correctness | `run_correctness.py` | `results/exp1_correctness.json` | ~10 s |
| 2. Latency benchmark | `run_benchmark.py` | `results/exp2_benchmark.json` | ~30 s |
| 3. KV-cache + FLOPs | `run_kv_cache.py` | `results/exp3_kv_cache.json` | ~1 s |
| 4. Quality probes | `run_quality.py` | `results/exp4_mqar.json` (+ `exp4b_quality_tasks.json`) | ~3 min (5 seeds × 200 steps) |
| 5. Ratio ablation | `ABL_MODE=ratio python run_ablation.py` | `results/exp5_ablation.json` | ~5 min (7 seeds × 7 layouts) |
| 5b. Eight-group matrix | `ABL_MODE=matrix python run_ablation.py` | `results/exp5b_ablation_matrix.json` | see below |
| 6. Decode latency | `run_decoding.py` | `results/exp6_decoding.json` | ~30 s |
| Figures | `make_figures.py` | `figures/fig_*.{pdf,png}` + `quality_vs_*.png` | ~5 s |

Run everything end-to-end:

```bash
python run_all.py                        # historical screening budget (default)
RUN_ALL_PROFILE=smoke python run_all.py  # minutes: plumbing validation only
RUN_ALL_PROFILE=full python run_all.py   # 7 seeds for the multi-seed statistics
```

Before any experiment runs, `run_all.py` writes `results/experiment_design.json`:
the pre-registered SESOI, analysis family, required seed count, and the power
the current budget actually delivers. Each multi-seed runner also embeds its
runtime audit as `metadata.experiment_design` (`confirmatory` /
`reduced_family` / `screening` / `incomplete_attrition` /
`exploratory_missing_controls`). The audit compares the REQUESTED seed block
with the seeds that actually survived into the analysis, so a run that
silently dropped failed seeds is labelled `incomplete_attrition` instead of
`confirmatory`. Validate the registry with
`python experiment_design.py --check`, and use
`python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json`
to turn a pilot's paired SD into a confirmatory seed budget. See
[docs/experiment-design.md](docs/experiment-design.md).

The comparative/ablation design also tests **interactions**, not just main
effects: Exp 5b reports the route × layout transfer estimand
`I_route = (D-A)_route1 - (D-A)_route2`, and Exp 8 reports the
`RoPE × selection-density` 2×2 interaction. Both are seed-paired
difference-in-differences rows in the same multiplicity family as the main
contrasts. Every two-cell matrix contrast also carries
`capacity_matched` / `claim_scope`; the fixed-structure groups differ in
parameter count, and a contrast outside the pre-registered 5% tolerance is
an operator-bundle + capacity effect, not a capacity-neutral mechanism
result. See [docs/comparative-ablation.md](docs/comparative-ablation.md).

### Switchable KDA / Mamba-2 memory route (quick start)

```python
from ops_fused import HybridConfig, HybridAttentionStack

# Historical KDA route (default; bit-identical to the old behaviour):
cfg = HybridConfig(d_model=256)                  # memory_type='kda'
model = HybridAttentionStack(cfg, total_layers=5)

# Mamba-2 route: ONE field switches the memory operator, CSA/HCA untouched:
cfg = HybridConfig(d_model=256, memory_type='mamba2',
                   mamba_d_state=64, mamba_headdim=64)
model = HybridAttentionStack(cfg, total_layers=5)

# Explicit layout (e.g. the eight-group matrix's D group, 10 layers):
cfg = HybridConfig(d_model=256, memory_type='mamba2',
                   layout=('memory', 'memory', 'memory', 'csa', 'hca') * 2)
model = HybridAttentionStack(cfg, total_layers=10)
```

The eight-group main experiment (KDA/Mamba2 × A/B/C/D at fixed depth):

```bash
ABL_MODE=matrix ABL_TOTAL_LAYERS=10 ABL_D_MODEL=32 python run_ablation.py
# Follow-up presets: ABL_MATRIX=csa_csa | position | budget
```

Environment knobs (set before launching):

| Variable | Default | Effect |
|---|---|---|
| `MQAR_SEEDS` | `5` | seeds for Exp 4 |
| `MQAR_STEPS` | `200` | training steps for non-softmax ops in Exp 4 |
| `MQAR_SOFTMAX_STEPS` | same as `MQAR_STEPS` | optional explicitly labelled softmax long-training sensitivity run |
| `MQAR_OPS` | `softmax,kda,mamba2,csa,hca` | operator list for Exp 4 (set `softmax,kda,csa,hca` to reproduce the historical 4-op table) |
| `MQAR_TASKS` | `mqar` | task list for Exp 4: `mqar`, `kv_update`, `long_range`, `multi_agg`, or `all` |
| `ABL_SEEDS` | `7` | seeds for Exp 5 |
| `ABL_STEPS` | `100` | training steps for Exp 5 |
| `ABL_MODE` | `ratio` | `ratio` (historical sweep) or `matrix` (eight-group design) |
| `ABL_MATRIX` | `main8` | matrix preset: `main8`, `csa_csa`, `position`, `budget` |
| `ABL_TOTAL_LAYERS` | `10` | fixed depth for all matrix groups (multiple of 5) |
| `ABL_MEMORY` | `kda,mamba2` | memory routes crossed in the matrix |
| `ABL_D_MODEL` | `32` | matrix model width (raise for the screening→formal stages) |
| `BENCH_LENGTHS` | `128,256,512,1024,2048` | sequence lengths for Exp 2 |
| `BENCH_REPEATS` | `5` | timed repeats per (T, op) in Exp 2 |
| `MQAR_TRAIN_BATCH` | `32` | training batch size for Exp 4 |
| `ABL_TRAIN_BATCH` | `16` | training batch size for Exp 5 |
| `MQAR_NKV` | `1` | comma-separated MQAR `n_kv` values for Exp 4 (e.g. `1,2,4`) |
| `ABL_NKV` | `1` | comma-separated MQAR `n_kv` values for Exp 5 |
| `SKIP_SLOW` | `0` | `1` truncates Exp 2/4/5 on CPU |
| `SKIP_CUDA_CHECK` | `0` | `1` bypasses the CUDA-availability guard |
| `KDA_BACKEND` | `reference` | `reference`, `fla`, or `auto`; applies to hybrid, standalone KDA quality/decode, ablation, and KDA benchmark paths |
| `MAMBA_BACKEND` | `reference` | `reference`, `official` (needs CUDA + `pip install -e '.[mamba]'`; errors loudly when unavailable), or `auto`; the resolved backend is recorded per result row |
| `MAMBA2_D_STATE` / `MAMBA2_HEADDIM` / `MAMBA2_EXPAND` / `MAMBA2_D_CONV` | `16` / `16` / `2` / `4` | small-model Mamba-2 geometry for the quality/ablation/matrix runners (headdim auto-clamped to divide `expand × d_model`) |
| `RUN_ALL_PROFILE` | `screen` | `smoke` / `screen` / `full` budget tiers for `run_all.py` (presets only touch env vars you did NOT export) |

---

## Tests

```bash
# Custom runner (290+ checks, includes long-running correctness checks):
python run_correctness.py

# pytest-compatible: the test functions use the standard `test_*` naming
# convention, so they are also discoverable by pytest:
pip install pytest
pytest -q run_correctness.py
pytest -q test_figures.py
```

The custom runner is the canonical entry point — it emits a structured
`results/exp1_correctness.json` and a pass/fail summary that the
`run_all.py` orchestrator consumes. Use `pytest` for selective / parallel
runs during development.

---

## Result files

| File | Schema | Notes |
|---|---|---|
| `results/exp1_correctness.json` | `[{name, status, detail}, ...]` | per-check rows from the custom runner |
| `results/exp2_benchmark.json` | `[{T, op, time_ms, peak_mem_MB, device, repeats, compute_boundary, n_layers, note}, ...]` | **`compute_boundary` differs per op** — see *Fairness notes* |
| `results/exp3_kv_cache.json` | `[{T, op, accounting_mode, accounting_semantics, kv_bytes, kv_elements, ...}, ...]` | analytic model, not profiled |
| `results/exp4_mqar.json` | `{metadata, results: [{op, n_kv, per_seed: [...], mean_acc, std_acc, ci95_acc, chance_acc, conclusions_valid, ...}, ...]}` | multi-seed with CI95 + Bonferroni; rows now carry `task` + `n_params` and include the `mamba2` operator by default |
| `results/exp4b_quality_tasks.json` | `{metadata, results: [...]}` | same envelope for the kv-update / long-range / multi-agg probes (`MQAR_TASKS`) |
| `results/exp5_ablation.json` | `[{ratio, layout, n_kv, per_seed, mean_acc, ...}, ...]` | same envelope as exp4 minus the metadata wrapper (`ABL_MODE=ratio`) |
| `results/exp5b_ablation_matrix.json` | `[{group, memory, layout, n_kv, per_seed, mean_acc, n_params, total_layers, d_model, ...}, ...]` | the eight-group matrix and its presets (`ABL_MODE=matrix`); `metadata.design_audit.capacity_audit` labels parameter-count confounds |
| `results/exp5b_ablation_matrix_contrasts.json` | `[{n_kv, memory, contrast, mean_diff, t_stat, sd_diff, n_common_seeds, predicted_direction, capacity_matched, claim_scope, interaction_type, interaction_cells, ...}, ...]` | paired contrasts; includes `I_route(D)=(D-A)_route1-(D-A)_route2` and per-row capacity bookkeeping |
| `results/exp8_mechanism_ablation.json` | `{metadata, results: [{variant, n_kv, per_seed, mean_acc, paired_vs_baseline_*, gold_pr_auc, ...}, ...]}` | `metadata.factorial_interactions.results` holds the pre-registered RoPE × selection-density 2×2 interaction |
| `results/exp6_decoding.json` | `[{op, prefill_ms, mean_decode_ms_per_token, median_decode_ms_per_token, peak_mem_MB, uses_incremental_cache, prefill_cache_build, compute_boundary, n_layers, ...}, ...]` | softmax / KDA / CSA / HCA / hybrid (standalone CSA/HCA and hybrid use incremental decoding caches; `n_layers` marks hybrid as a 5-layer stack vs single-layer methods) |
| `results/experiment_design.json` | `{schema_version, kind: "experiment_design", metadata, results: [{plan_id, factors, sesoi, planned_seeds, planning, status}, ...]}` | prospective plan + required n, written by `run_all.py` before the experiments start |
| `results/summary.json` | `{env: {git_commit, torch_version, cuda_version, kda_backend, ...}, runs: [{name, status, time_s}], n_ok, n_fail, total_time_s}` | produced by `run_all.py` |

> **Known schema inconsistency.** Exp 4 wraps its results in
> `{metadata, results: [...]}`; the other experiments emit a bare list. We
> keep both shapes for backward compatibility with downstream figure code
> (`make_figures.load` handles both). Future work: unify on the envelope
> with a `schema_version` field.

---

## Fairness notes (READ BEFORE INTERPRETING FIGURES)

Several experimental decisions affect cross-op comparisons. The figures and
JSON outputs annotate these where possible, but they are easy to miss:

### 1. Benchmark compute boundary (Exp 2)

The operators in `run_benchmark.py` are timed under **different
boundaries**:

| Op | Boundary | What's included |
|---|---|---|
| `softmax`, `kda_rec`, `kda_chunk`, `mamba2_scan` | `core` | only the attention / recurrence kernel; q/k/v (and g/beta, x/dt/B/C) are pre-projected outside the timed region. `mamba2_scan` is capacity-matched to `kda_rec` (4 heads × 32×32 state) |
| `csa`, `hca`, `mamba2_layer` | `end_to_end_single_layer` | a single layer end-to-end: input projection + compression / conv + indexer / scan + attention + output projection |
| `hybrid`, `hybrid_mamba2` | `end_to_end_multi_layer` | a 5-layer stack with LayerNorm, projections, attention, state management; the two rows share identical CSA/HCA geometry so the pair isolates the memory-operator swap |

These numbers are **not directly comparable** as "operator latency".
`make_figures.fig_benchmark` splits them into separate subplots by
`compute_boundary` and prints a warning in the caption. The stale
`results/exp2_benchmark.json` committed in this repo predates the
`compute_boundary` field; regenerate it with `python run_benchmark.py` to
get the metadata, or rely on the `_OP_BOUNDARY_FALLBACK` mapping in
`make_figures.py` for the legacy file.

### 2. Softmax baseline training steps (Exp 4)

The primary comparison gives **every operator the same optimizer-step
budget** (`MQAR_STEPS`, default 200). This avoids advantaging softmax with
extra optimization. Because softmax may converge more slowly, an optional
sensitivity run can explicitly set `MQAR_SOFTMAX_STEPS` to a larger value;
such a run is not a fixed-budget comparison and must be labelled separately.
The per-seed `steps` field records the actual budget used.

### 3. MQAR statistical power (Exp 4)

The default MQAR config (`vocab=16, seq_len=16, n_kv=1`) has chance
accuracy 6.25%. Several ablations land within a few percentage points of
chance, so the `conclusions_valid` flag and the Bonferroni-corrected
t-test in `run_quality.py` are the authoritative signal — `mean_acc` alone
is misleading. Treat near-chance results as a *smoke quality probe*, not a
structural claim. The statistical tests currently reported are primarily
**vs. the chance baseline**, not pairwise operator-vs-operator tests; do not
claim that operator A is significantly better than operator B unless you add
and report an explicit pairwise test.

### 4. Decoding experiment scope (Exp 6)

`run_decoding.py` benchmarks **softmax, KDA, Mamba-2, CSA, HCA, and the
hybrid stacks (both memory routes)** for both prefill and per-token
decode latency. CSA and HCA use the incremental decoding cache
implemented in `ops_decoding_cache.py` (`CSADecodingCache` /
`HCADecodingCache`), which maintains:

* a **partial-token accumulator** (Python `list`) — buffers 0..m-1 new
  tokens until a compressed block can be formed;
* a **compressed-block cache** (`[B, n_blocks, c]` tensor) — grows by
  one row every `m` (CSA) / `m2` (HCA) tokens;
* a **sliding-window ring buffer** (`[B, win, c]` tensor) — fixed-size
  FIFO of the most recent `win` uncompressed local keys;
* a **dynamically-updated indexer key cache** (CSA only,
  `[B, n_blocks, c_I]` tensor) — grows in lockstep with the
  compressed-block cache so the lightning indexer can score new blocks
  without recomputing the full history.

The cache avoids recomputing compression over the full prefix on every decode
step, but the attention/indexer work still scales with the number of completed
compressed blocks: roughly **O(T/m + win)** for CSA and **O(T/m2 + win)** for
HCA (plus Python-reference overhead). The standalone CSA/HCA rows and
the hybrid row use these caches during token decode; the hybrid wrapper
threads one cache per CSA/HCA sub-layer through the full KDA+CSA+HCA
stack, so CSA/HCA layers can see the prefill history instead of only the
current token. The small decode benchmark uses a lighter geometry than the
Exp 4/5 models (`csa_topk=2`, `c=8`, `dc=8`, `m2=4` at `d_model=64`,
preserving the sparse top-k path); its per-token decode numbers must not be
compared head-to-head with Exp 4/5 quality results. The `prefill_ms` for cache-enabled CSA/HCA/hybrid rows
includes correctness-first Python cache population (`prefill_cache_build` in
the JSON is `"batched_append_from_naive_output"` for standalone CSA/HCA and
`"batched_populate_layer_by_layer"` for the hybrid); it is therefore a
conservative reference-wrapper prefill number. The per-token decode timings are the main
cache-efficiency signal.

### 5. KV-cache + FLOPs are analytic (Exp 3)

`run_kv_cache.py` computes KV-bytes and FLOPs from closed-form formulas
(derived from the operator definitions, with corrections for ceil-block
counts, causal entries, input **and grouped output** projection terms, and the
incremental CSA/HCA runtime state such as partial-token accumulators, fixed
sliding-window ring-buffer capacity, and CSA overlap state). They are **not**
measured from a real forward pass. The unit tests in `run_correctness.py`
(`test_prefill_flops_*`, `test_kv_cache_ceil_block_count`) verify the
formulas against hand-computed expected values; for production claims,
cross-check with `torch.cuda.memory_allocated` and a FLOP counter.

### 6. Near-duplicate leakage and data fingerprints

The LM data protocol removes exact and whitespace/Unicode-normalized
duplicates before splitting, and audits identical truncated model windows.
It also audits **near duplicates** — the same story with a few words changed,
which exact hashing cannot see and which would inflate a held-out score if
the two copies straddled train/val/test. `near_duplicates.py` compares the
truncated `seq_len+1` token windows by token n-gram Jaccard (default
threshold `0.8`, 5-grams), using MinHash+LSH for candidates and an exact
second pass for the reported similarity. The cross-split audit runs by
default in `train_lm_autodl.py`; `--dedup_near` removes near duplicates
before the split and `--strict_near_dup` refuses to train while any
cross-split pair remains (or while the candidate cap left the audit
incomplete). The audit is structural (no model metric), so it
can run while the test split is still sealed. See
[docs/data-integrity.md](docs/data-integrity.md) for the full contract,
the recorded `near_duplicate_audit` fields, the candidate-truncation
caveat, and the corpus-fingerprint framing fix.

---

## Limitations

* **KDA requires unit-norm `q`/`k` for numerical stability.** Neither
  `naive_recurrent_kda` nor `naive_chunk_kda` enforces this — nothing stops
  a caller from passing raw (non-normalized) `q`/`k`. Always
  `F.normalize(q, dim=-1)` / `F.normalize(k, dim=-1)` before calling into
  KDA unless you have a specific reason not to.
* **Reference Python loops.** KDA's recurrent path is a Python `for` loop
  over time; the chunked path still has a Python loop over chunks. CSA's
  indexer loops over heads. Optional accelerated paths
  (`ops_kda.compiled_recurrent_kda`, `ops_kda.scripted_chunk_kda`, and
  `HybridConfig(kda_backend="fla")` for GPU) are available for better
  performance. The naive paths remain the default for correctness /
  readability.
* **STE for CSA indexer.** The default straight-through estimator
  (`ste_mode='topk_columns'`) routes gradient through the top-k selected
  columns of `soft_weights` (see `ops_csa.py::naive_csa`). Non-selected
  blocks receive no "you should have been picked" gradient unless the
  optional `ste_mode='full_softmax'` dense surrogate is selected;
  `aux_contrastive_loss` remains future work. The STE path is training-only:
  `naive_csa` and the decoding cache automatically skip the soft surrogate
  under `torch.no_grad()` so inference/benchmark latency measures the hard
  top-k path, not the training proxy.
* **Fusing the 6 KV/indexer projections.** `naive_csa` fuses
  `W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx` into one `F.linear` call
  by default (`fuse_projections=True`), purely as a kernel-launch-count
  optimization. Pass `fuse_projections=False` if you require the original
  6-separate-matmul path (e.g., to preserve `None` grads for `W_KV_idx`/
  `W_Z_idx` under `use_ste=False`).
* **Sliding window in the reference ops is unfold-based.** `naive_csa` and
  `naive_hca` compute the window branch with `_sliding_window_scores`
  (`unfold`, peak memory `O(T · win · c)`) because the window scores must
  share ONE softmax with the compressed-branch scores and the attention
  sink (DeepSeek-V4 Eq. 27). `ops_csa._sliding_window_attention` is a
  standalone chunked helper (`O(chunk_t · win · c)` peak memory) for
  callers that do not need the joint softmax; it is numerically identical
  to the unfold softmax path. Long-context experiments that need the memory
  bound call that chunked helper directly (`ops_csa._sliding_window_attention`
  in `ops_csa.py`); the reference `naive_csa` / `naive_hca` paths remain
  unfold-based.
* **Incremental decoding cache (CSA / HCA).** The
  `ops_decoding_cache.CSADecodingCache` / `HCADecodingCache` enable
  token-by-token autoregressive decoding for CSA / HCA. The cache maintains
  a partial-token accumulator, a compressed-block cache, a sliding-window
  ring buffer, and (for CSA) a dynamically-updated indexer key cache. The
  cache avoids full-prefix recompression on every decode step; per-token
  work still scales with completed compressed blocks (about O(T/m + win) for
  CSA and O(T/m2 + win) for HCA). The Exp 6 hybrid wrapper wires one such
  cache into every CSA/HCA sub-layer of the full stack.
* **Dropout unimplemented.** `HybridConfig.dropout != 0` raises
  `NotImplementedError` rather than silently no-op'ing. MQAR-scale models
  don't need dropout; larger runs should wire it in (or remove the field).
* **Causal block mask is strictly preceding (paper-exact).**
  `_causal_block_mask` uses `b < t // m` (DeepSeek-V4 Eq. 16: a query
  attends only to blocks with index strictly below `floor(t/m)`).
  A compressed block therefore first becomes visible at the first token
  of the NEXT block; the query's own block — complete or not — is never
  reachable through the sparse branch, and its local context is served
  by the sliding-window branch instead (§2.3.3). The decoding caches and
  the FLOPs accounting (`run_kv_cache.causal_block_entries`) use the
  same rule. (Historical note: an earlier revision used the
  window-close variant `b < (t + 1) // m`, which exposed the query's own
  block at its final source token — causally safe but off-by-one against
  the paper; aligned to the paper-exact mask in the 2026-08 fix pass
  together with the partial-RoPE addition below.)
* **Partial RoPE (DeepSeek-V4 §2.3.3) is implemented and ON by default at
  the model level.** Queries, compressed KV entries (which serve as both
  keys and values in the shared-KV MQA), and sliding-window keys are
  rotated on their last `rope_dim` dims (paper value 64, auto-clamped to
  the head dim `c` for the small experiment geometries), and the
  core-attention output is inverse-rotated at the negated query position
  so each KV entry's contribution carries the relative (entry − query)
  distance. Interpretation choices the paper leaves open are documented
  in `ops_rope.py`: compressed entry `s` is anchored at position `s·m`
  (block start), and the output countermeasure rotates by −t (the query
  position). `HybridConfig.csa_rope_dim` / `hca_rope_dim` default to 64;
  set them to `None`/0 to disable (the pre-fix historical behaviour). The
  raw operators (`naive_csa` / `naive_hca`) keep `rope_dim=None` as their
  default for backward compatibility, and the incremental decoding
  caches accept the same `rope_dim` / `rope_base` arguments — prefill and
  token-by-token decoding agree to fp32 tolerance with RoPE enabled
  (pinned by `test_csa_hca_rope_prefill_vs_cache`). The indexer is
  intentionally NOT roped: §2.3.3 scopes RoPE to the core attention, and
  Eq. 15–16 contain no positional term.
* **Query/KV normalization: L2 by default, paper-form RMSNorm optional.**
  §2.3.3 specifies an RMSNorm on each head of the queries and on the
  compressed KV entries; the repository's historical behaviour L2-normalizes
  both (cosine attention, `scale=1.0`). `naive_csa` / `naive_hca` accept
  `qk_norm_mode='rms'` for the paper-form RMSNorm (values then keep norm
  ~`sqrt(c)` and the standard `1/sqrt(c)` temperature is auto-selected,
  mirroring how DeepSeek-V2/V3 combine per-head RMSNorm with standard
  scaling). The paper does not pin the core-attention temperature, so
  both forms are paper-compatible; `'l2'` remains the default so the
  repository's historical experiment numbers stay reproducible.
* **Cosine scale is fixed at 1.0 (default 'l2' mode).** `naive_csa` /
  `naive_hca` L2-normalize `q` and `C_comp` and use `scale=1.0` (a
  deliberate fix — the old `c ** -0.5` flattened softmax). The repository
  experiments also pass `normalize_qk=True` for the CSA lightning indexer
  so top-k selection is direction-based rather than dominated by
  q_idx / K_idx vector norms — note this indexer normalization is a
  repository-side choice, NOT a paper requirement (Eq. 15–16 define a
  plain weighted ReLU dot product). No learnable temperature is exposed.
  Pass `scale=` explicitly to override.

---

## Papers

The implementation follows these two arXiv reports. The PDFs are no longer
committed to this repository (see `.gitignore`); download them from arXiv:

* [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv:2606.19348)](https://arxiv.org/abs/2606.19348)
* [Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)

## License

See `LICENSE`.


---

## What's new: real LM training + AutoDL for under ¥120 (2026-07)

The repository's original six experiments are correctness / benchmark / MQAR
probes. The new `train_lm_autodl.py` performs real language-model training
with `ops_fused.HybridKCHAttention`, supports both Kaggle and AutoDL, and
costs far less than ¥120.

### 2026-08 paper alignment + regression fixes (changes experiment numbers — please regenerate results/)

Fixes applied after re-reviewing the implementation against
arXiv:2510.26692 / arXiv:2606.19348:

1. **KDA chunked-path regression fix (P0).** An earlier automated fix had
   mistakenly changed the A-matrix mask from `diagonal=0` back to
   `diagonal=1`, which made `naive_chunk_kda` / `scripted_chunk_kda` deviate
   from the recurrent path by O(1e-2) (25 correctness checks failed,
   including fp64 gradient consistency). `diagonal=0` has been restored, and
   a comment was added to prevent the regression from recurring.
2. **Causal block mask off-by-one (P2-1).** `_causal_block_mask` changed from
   the window-close form `b < (t+1)//m` to the strictly-preceding form
   `b < t//m` of the paper's Eq. 16. The decoding caches, the FLOPs
   accounting, and the corresponding tests were updated in sync.
3. **Partial RoPE and output counter-rotation per §2.3.3 (P2-2).** New
   `ops_rope.py`; the CSA/HCA query, compressed KV, and sliding-window keys
   are rotated on their last `rope_dim` dims (paper value 64, clamped to the
   head dim), and the output is counter-rotated by −t. Enabled by default at
   the model level; off by default in the raw operators (backward
   compatible); the incremental decoding caches support it as well.
4. **Sliding-window branch restored to the Eq. 27 joint softmax.** Compressed
   blocks and sliding-window tokens share a single softmax denominator
   (including the sink), instead of each branch applying its own softmax and
   the two results being added.
5. Others: `qk_norm_mode='rms'` option (the paper §2.3.3 RMSNorm form); a
   fp16 round-trip precision fix for the KDA decode state; a corrected
   attribution comment for the indexer cosine normalization; Kaggle
   compatibility (the `run_all.py` version gate relaxed to a warning; the
   torch upper bound removed).

The correctness suite is now 267 checks, all passing. Because of the mask and
RoPE changes, CSA/HCA-related experiment numbers will change — please
regenerate `results/` with `python run_all.py`.

### Why the original ops were not replaced

See [COMPARISON.md](COMPARISON.md) for the comparison: the existing
`ops_kda/csa/hca` already handle 20+ edge cases (g clamping, STE, sink,
NaN-safety, chunked SW, decoding cache, ...), which makes them more suitable
for a paper than the early toy implementation. The original implementations
are therefore kept, and only the training pipeline was added.

### AutoDL one-command run

```bash
# max_steps counts optimizer steps; with the small default config the cost on
# a 3090/4090 stays far below ¥120
pip install -r requirements.txt
pip install transformers datasets accelerate
python train_lm_autodl.py --autodl --max_steps 2000 --seq_len 1024 --batch_size 2
```

### Kaggle

```python
!pip install -q transformers datasets
!python train_lm_autodl.py --kaggle
```

Default Kaggle config: d=256, 5 layers (3:1:1), seq 512, 500 steps — roughly
40 minutes on a single T4 (the script uses one GPU, no DDP; on a T4x2
instance only one of the two cards is used).

> **Note (avoiding a torch downgrade).** Kaggle GPU images ship a CUDA build
> of torch preinstalled. Just run `!pip install -q transformers datasets`,
> and do not perform any torch reinstall on Kaggle beyond
> `pip install -r requirements.txt` / `pip install -e .`. This repository's
> dependency range is `torch>=2.2,<3`, a no-op for the preinstalled version;
> if you manually pin an old upper bound such as `<2.7`, pip will downgrade
> the CUDA torch to a CPU wheel and the GPU becomes unusable.

### Cost breakdown

| Phase | Duration on a 3090 | Cost |
|---|---|---|
| Phase1 1024 ctx 2000 optimizer steps | several hours | far below ¥120 |
| Phase2 4096 ctx 500 optimizer steps | GPU-dependent | far below ¥120 |
| Total | GPU-dependent | << ¥120 |

Even scaled up to d=512, 12 layers, 4000 steps: 4 h on a 4090 ≈ ¥11 — still
within budget.

The run produces `checkpoints_lm/final_lm.pt`, which can be further evaluated
with `run_quality.py` / `run_decoding.py`.

---

# 中文版

[English](#kda-csa-hca-fusion) · **中文**

本仓库提供一个可读、正确性优先的 PyTorch 参考实现与实验框架，用于一种混合线性注意力栈。该栈把一个可切换的递归记忆算子与两个互补的检索算子组合在一起：

* **KDA** —— Kimi 风格的 Delta Attention（带逐通道对数衰减的递归 delta-rule）。承担主要的 token 混合，每头状态为 O(1)。
* **Mamba-2** —— SSD（arXiv:2405.21060）混合器，带逐头矩阵状态；是第二条可选的递归记忆路线（`memory_type='mamba2'`），与 KDA 遵循相同的 O(1) 状态解码契约。
* **CSA** —— DeepSeek-V4 压缩稀疏注意力（重叠式双分支 KV 压缩 + 为每个 query 选出 top-k 压缩块的 lightning 索引器）。负责稀疏长程检索。
* **HCA** —— DeepSeek-V4 重度压缩注意力（单分支重度压缩，在极少量压缩条目上做稠密 MQA，可选滑动窗口）。提供廉价的全局上下文。

默认情况下，混合栈按 `3:1:1` 的 memory:CSA:HCA 比例交错堆叠（见 `ops_fused.HybridAttentionStack`；`HybridKCHAttention` 保留为向后兼容名）。递归记忆槽位可以通过一个配置字段在 KDA 与 Mamba-2 之间切换——设计细节与八组实验矩阵见 [docs/mamba2-framework.md](docs/mamba2-framework.md)。

> **它是什么。** 一个研究级、正确性优先的参考实现。数值边界情况（因果性、dtype、空序列、sink 稳定性、跨解码步的状态管理）的处理比典型的"研究代码堆"更细致——见 `run_correctness.py` 中大量的回归检查。
>
> **它不是什么。** 它**不是**生产级 kernel。KDA 用 Python `for` 循环逐时间步递推；CSA/HCA 用基于 `unfold` 的滑窗；没有任何算子做过融合或 Triton 化。`run_benchmark.py` 的延迟数字测的是这个朴素参考实现，**不得**拿来与 FLA / FlashAttention / DeepSeek 生产 kernel 比较。见下文*局限性*一节。
>
> **加速路径。** 出于正确性 / 可读性考虑，上述朴素路径仍是默认路径，但仓库提供可选的加速封装，保持完全相同的数值契约：
>
> - `ops_kda.compiled_recurrent_kda` —— `naive_recurrent_kda` 的 `torch.compile` 封装，按（形状、dtype、requires_grad）签名缓存。适合中等规模的 `T`（≥1024）；GPU 上典型加速 5–20×。
> - `ops_kda.scripted_chunk_kda` —— `naive_chunk_kda` 内层逐 chunk 循环的 `torch.jit.script` 封装；当 TorchScript 拒绝输入时会优雅回退到 eager 路径。
> - `ops_csa._sliding_window_attention` —— 独立的显存高效滑窗注意力；当 `T * win * c > 8M` 个元素时自动分块，把峰值显存从 `O(T · win · c)` 降为 `O(chunk_t · win · c)`。`naive_csa` / `naive_hca` 用的是基于 unfold 的 `_sliding_window_scores`，使窗口条目与压缩分支得分、attention sink 共享同一个联合 softmax。
> - `naive_csa` 本身把 6 次 `F.linear(H, W_*)` 调用（`W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx`）通过 `torch.cat` + `tensor.split` 融合为单个 matmul，把 kernel 启动开销降低 6×。

---

## 目录

- [仓库结构](#仓库结构)
- [安装](#安装)
- [运行实验](#运行实验)
- [测试](#测试)
- [结果文件](#结果文件)
- [公平性说明](#公平性说明解读图表前必读)
- [局限性](#局限性)
- [论文](#论文)
- [许可证](#许可证)

---

## 仓库结构

```
.
├── ops_kda.py             # KDA 参考实现：naive_recurrent_kda、naive_chunk_kda
├── ops_kda_backend.py     # 可选 KDA 后端适配器：reference / FLA / auto
├── ops_kda_autograd.py    # 递归 KDA 路径的自定义 autograd
├── ops_mamba.py           # Mamba-2 (SSD) 参考混合器：scan + 统一的 prefill/step
├── ops_mamba_backend.py   # 可选 Mamba-2 后端适配器：reference / mamba_ssm / auto
├── ops_rwkv.py            # RWKV-7 参考混合器（第三条记忆路线）
├── ops_rwkv_backend.py    # 可选 RWKV-7 后端适配器
├── ops_rope.py            # CSA/HCA 用的部分 RoPE（DeepSeek-V4 §2.3.3）
├── ops_norm.py            # 各算子共享的 RMSNorm 参考实现
├── ops_csa.py             # CSA：compress_kv（重叠式）、lightning 索引器、naive_csa
├── ops_hca.py             # HCA：重度压缩 + 稠密 MQA + SW，naive_hca
├── ops_decoding_cache.py  # CSADecodingCache / HCADecodingCache / Mamba2DecodingState
├── ops_fused.py           # HybridConfig + build_mixer 工厂 + HybridAttentionStack
│                          #   （HybridKCHAttention 保留为兼容名）
├── checks/                # 按子系统分组的正确性检查
├── quality/               # MQAR 任务、模型、训练与统计辅助
├── docs/                  # 设计说明与实验契约
├── docs/mamba2-framework.md # 可切换 KDA/Mamba-2 框架的设计文档
├── COMPARISON.md          # ops_* 与早期 toy 实现的对比
├── run_correctness.py     # 回归检查（自定义 runner；可被 pytest 导入）
├── run_benchmark.py       # 实验 2：延迟 vs 序列长度（含 op_boundary 元数据）
├── run_quality.py         # 实验 4：MQAR + kv-update/long-range/multi-agg 探针（多种子）
├── run_ablation.py        # 实验 5：比例扫描（ABL_MODE=ratio）| 八组矩阵（matrix）
├── run_decoding.py        # 实验 6：prefill + 逐 token 解码延迟（两条记忆路线都测）
├── run_kv_cache.py        # 实验 3：解析式 KV-cache + FLOPs 核算（+ 状态分解）
├── method_analysis.py     # 逐头原型（demo 中把 CSA 简化为稠密）
├── make_figures.py        # 从 results/ 生成 figures/*（含 quality-vs-cost）
├── run_all.py             # 单入口 runner（RUN_ALL_PROFILE=smoke|screen|full）
├── experiment_design.py   # 预注册方案、前瞻功效/MDE、运行时审计
├── kaggle_setup.py        # CUDA 引导 + 共享工具（sanitize_for_json、parse_int_env）
├── near_duplicates.py     # token n-gram MinHash/LSH 近重复泄漏检测器
├── docs/data-integrity.md # 近重复 + 指纹的数据完整性契约
├── test_figures.py        # 图表生成管线的测试
├── results/               # 生成的 JSON 输出（gitignored；用 run_all.py 重新生成）
├── figures/               # 生成的 PDF/PNG 输出（gitignored；用 make_figures.py 重新生成）
├── requirements.txt
├── pyproject.toml         # `pip install -e .` 之后各模块可作为包导入
└── README.md              # 本文件（中英双语）
```

---

## 安装

**需要 Python 3.10–3.13**（源码使用 PEP 604 的 `X | None` 联合类型语法，完整回归套件在 3.10–3.13 解释器矩阵上验证过）。`pyproject.toml` 声明了 `requires-python = ">=3.10,<3.14"`。

```bash
# 1. 克隆
git clone https://gitee.com/qwe12345678/kda-csa-hca-fusion.git
cd kda-csa-hca-fusion

# 2. 安装包及其运行时依赖
pip install -e .        # 安装 torch、einops、matplotlib、numpy、scipy
#    或者，只需要运行时依赖、不需要开发工具链时：
#    pip install -r requirements.txt

# 3.（可选）安装开发工具（pytest、pytest-xdist、mypy、ruff）：
pip install -e .[dev]

# 4.（可选）安装 FLA KDA 后端。这不会改变默认的参考实验；
#    需要在 HybridConfig 里显式选择。
pip install -e '.[fla]'
```

默认的 `HybridConfig(kda_backend="reference")` 保留原始的 PyTorch KDA 实现。要在 GPU 实验中启用 FLA：

```python
from ops_fused import HybridConfig
cfg = HybridConfig(kda_backend="fla")
```

使用 `kda_backend="auto"` 时，只有当 FLA 已安装且张量在 CUDA 上才会尝试 FLA，否则回退到参考路径。对独立的质量 / 解码 / 基准 runner，在启动前设置同样的选择：

```bash
KDA_BACKEND=fla python run_benchmark.py
KDA_BACKEND=auto python run_decoding.py
```

适配器直接传入已归一化的 q/k、对数空间的门控以及 sigmoid 之后的 beta 值，不会重复施加这些变换。CSA/HCA 始终走仓库参考路径；常规实验不需要任何 Hugging Face 或 vLLM 依赖。

`pip install -e .` 之后，实验脚本可以作为模块运行：

```bash
python -m run_correctness       # 或者：python run_correctness.py
python -m run_benchmark
python -m run_all
```

### 快速开始

```python
import torch
from ops_fused import HybridConfig, HybridAttentionStack

cfg = HybridConfig(d_model=64, n_heads_qk=2, n_heads_v=2,
                   head_dim_k=16, head_dim_v=16, kda_chunk_size=4)
model = HybridAttentionStack(cfg, total_layers=5).eval()

x = torch.randn(2, 8, cfg.d_model)
with torch.no_grad():
    y = model(x)

print(y.shape)  # torch.Size([2, 8, 64])
```

`HybridAttentionStack` 按配置的记忆 / CSA / HCA 布局执行前向，返回形状为
`[batch, sequence, d_model]` 的张量。布局覆盖与各算子配置见
[可切换的 KDA / Mamba-2 记忆路由](#可切换的-kda--mamba-2-记忆路由快速上手)。

### Kaggle

把仓库作为 Dataset 上传，然后在 notebook 单元格里：

```python
!pip install -q einops matplotlib
import sys; sys.path.insert(0, '/kaggle/input/<your-dataset-name>')
%run /kaggle/input/<your-dataset-name>/run_all.py
```

在 Kaggle T4 上跑 GPU 时，请按 `kaggle_setup.bootstrap_kaggle_cuda()` 记录的 CUDA 引导流程操作（先装 CUDA wheel，再重启 kernel——`setup_kaggle()` 只*验证* CUDA 可用性，不会就地安装）。当前 Kaggle 镜像的 torch 自带 CUDA 预装，因此引导通常是无操作。引导会锁定与检测到的 CUDA wheel 索引**精确匹配**的 torch 版本（`kaggle_setup.py` 里的 `_BOOTSTRAP_TORCH_PINS`）：例如 cu118/cu124/cu126 对应 `torch==2.6.0`，cu121 对应 `torch==2.5.1`（它的最后一个版本），cu128 对应 `torch==2.7.1`（cu128 wheel 的起点）——而且在提示重启之前会验证已安装的 wheel 确实带 `+cuXXX` 构建标签，因此 CPU 回退安装无法再伪装成成功。

面向预装环境的依赖策略：`run_all.py` 不再硬性拒绝超出历史测试边界的 torch/numpy/matplotlib 版本——改为警告并继续（算子代码在正确性上没有版本敏感的 API 依赖；只有 softmax 基线的基准延迟可能随 torch 版本漂移）。缺失的包仍会被安装。设置 `STRICT_DEPS=1` 可以恢复硬版本门禁，用于比特级可复现的历史重跑。相应地，`torch` 在 `requirements.txt` / `pyproject.toml` 中只保留宽松上界（`<3`），保证 `pip install -e .` 绝不降级 Kaggle 预装的 CUDA torch；要复现已提交的历史数字，请自行 pin `torch==2.6.*`。numpy / matplotlib / scipy 的建议区间会在完整套件于新版本上重新验证后扩展（最近一次：2026-08，numpy 2.5.2 / matplotlib 3.11.1 / scipy 1.18.1 / torch 2.13.0）。

Kaggle 免费额度（2×T4、每周 30 小时）下的资源占用：默认参数的 `run_all.py` 是 CPU 量级（约 10 分钟）——不需要 GPU。默认的 `train_lm_autodl.py --kaggle` 配置（d=256、5 层、序列 512、500 个 optimizer 步）在 2×T4 上约 40 分钟、占用几 GB 显存——轻松落在免费额度内。`train_lm_autodl.py` 在没有 `transformers`/`datasets` 时也能运行（对合成文本使用字符级回退分词器），因此纯 CPU 会话也能跑通整条管线。

---

## 运行实验

| 实验 | 脚本 | 输出 | 耗时（CPU） |
|---|---|---|---|
| 1. 正确性 | `run_correctness.py` | `results/exp1_correctness.json` | ~10 秒 |
| 2. 延迟基准 | `run_benchmark.py` | `results/exp2_benchmark.json` | ~30 秒 |
| 3. KV-cache + FLOPs | `run_kv_cache.py` | `results/exp3_kv_cache.json` | ~1 秒 |
| 4. 质量探针 | `run_quality.py` | `results/exp4_mqar.json`（+ `exp4b_quality_tasks.json`） | ~3 分钟（5 种子 × 200 步） |
| 5. 比例消融 | `ABL_MODE=ratio python run_ablation.py` | `results/exp5_ablation.json` | ~5 分钟（7 种子 × 7 种布局） |
| 5b. 八组矩阵 | `ABL_MODE=matrix python run_ablation.py` | `results/exp5b_ablation_matrix.json` | 见下文 |
| 6. 解码延迟 | `run_decoding.py` | `results/exp6_decoding.json` | ~30 秒 |
| 图表 | `make_figures.py` | `figures/fig_*.{pdf,png}` + `quality_vs_*.png` | ~5 秒 |

端到端全部运行：

```bash
python run_all.py                        # 历史筛选预算（默认）
RUN_ALL_PROFILE=smoke python run_all.py  # 分钟级：只验证链路
RUN_ALL_PROFILE=full python run_all.py   # 7 种子，用于多种子统计
```

任何实验开始之前，`run_all.py` 会先写入 `results/experiment_design.json`：预注册的 SESOI（最小感兴趣效应量）、分析族、所需种子数，以及当前预算实际交付的功效。每个多种子 runner 还会把运行时审计嵌入 `metadata.experiment_design`（`confirmatory` / `reduced_family` / `screening` / `incomplete_attrition` / `exploratory_missing_controls`）。审计会比较"请求的种子块"与"实际进入分析的种子"，因此悄悄丢弃了失败种子的运行会被标记为 `incomplete_attrition` 而非 `confirmatory`。用 `python experiment_design.py --check` 校验注册表；用 `python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json` 把试点的配对 SD 换算成验证性种子预算。见 [docs/experiment-design.md](docs/experiment-design.md)。

比较 / 消融设计还检验**交互效应**，而不只是主效应：实验 5b 报告 route × layout 迁移估计量 `I_route = (D-A)_route1 - (D-A)_route2`，实验 8 报告 `RoPE × selection-density` 的 2×2 交互。两者都是与主对比同一多重性族内的种子配对双重差分行。每个二格矩阵对比还携带 `capacity_matched` / `claim_scope`；固定结构的组之间参数量不同，超出预注册 5% 容差的对比是"算子束 + 容量"效应，而不是容量中立的机制结论。见 [docs/comparative-ablation.md](docs/comparative-ablation.md)。

### 可切换的 KDA / Mamba-2 记忆路由（快速上手）

```python
from ops_fused import HybridConfig, HybridAttentionStack

# 历史 KDA 路由（默认；与旧行为比特级一致）：
cfg = HybridConfig(d_model=256)                  # memory_type='kda'
model = HybridAttentionStack(cfg, total_layers=5)

# Mamba-2 路由：一个字段切换记忆算子，CSA/HCA 原封不动：
cfg = HybridConfig(d_model=256, memory_type='mamba2',
                   mamba_d_state=64, mamba_headdim=64)
model = HybridAttentionStack(cfg, total_layers=5)

# 显式布局（例如八组矩阵的 D 组，10 层）：
cfg = HybridConfig(d_model=256, memory_type='mamba2',
                   layout=('memory', 'memory', 'memory', 'csa', 'hca') * 2)
model = HybridAttentionStack(cfg, total_layers=10)
```

八组主实验（固定深度下的 KDA/Mamba2 × A/B/C/D）：

```bash
ABL_MODE=matrix ABL_TOTAL_LAYERS=10 ABL_D_MODEL=32 python run_ablation.py
# 后续预设：ABL_MATRIX=csa_csa | position | budget
```

环境变量开关（启动前设置）：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `MQAR_SEEDS` | `5` | 实验 4 的种子数 |
| `MQAR_STEPS` | `200` | 实验 4 中非 softmax 算子的训练步数 |
| `MQAR_SOFTMAX_STEPS` | 同 `MQAR_STEPS` | 可选的、显式标注的 softmax 长训练敏感性运行 |
| `MQAR_OPS` | `softmax,kda,mamba2,csa,hca` | 实验 4 的算子列表（设为 `softmax,kda,csa,hca` 可复现历史 4 算子表格） |
| `MQAR_TASKS` | `mqar` | 实验 4 的任务列表：`mqar`、`kv_update`、`long_range`、`multi_agg` 或 `all` |
| `ABL_SEEDS` | `7` | 实验 5 的种子数 |
| `ABL_STEPS` | `100` | 实验 5 的训练步数 |
| `ABL_MODE` | `ratio` | `ratio`（历史扫描）或 `matrix`（八组设计） |
| `ABL_MATRIX` | `main8` | 矩阵预设：`main8`、`csa_csa`、`position`、`budget` |
| `ABL_TOTAL_LAYERS` | `10` | 所有矩阵组的固定深度（5 的倍数） |
| `ABL_MEMORY` | `kda,mamba2` | 矩阵中交叉的记忆路由 |
| `ABL_D_MODEL` | `32` | 矩阵模型宽度（从筛选到正式阶段可调大） |
| `BENCH_LENGTHS` | `128,256,512,1024,2048` | 实验 2 的序列长度 |
| `BENCH_REPEATS` | `5` | 实验 2 中每个 (T, op) 的计时重复次数 |
| `MQAR_TRAIN_BATCH` | `32` | 实验 4 的训练 batch size |
| `ABL_TRAIN_BATCH` | `16` | 实验 5 的训练 batch size |
| `MQAR_NKV` | `1` | 逗号分隔的实验 4 MQAR `n_kv` 取值（例如 `1,2,4`） |
| `ABL_NKV` | `1` | 逗号分隔的实验 5 MQAR `n_kv` 取值 |
| `SKIP_SLOW` | `0` | `1` 时在 CPU 上截断实验 2/4/5 |
| `SKIP_CUDA_CHECK` | `0` | `1` 时绕过 CUDA 可用性守卫 |
| `KDA_BACKEND` | `reference` | `reference`、`fla` 或 `auto`；作用于混合模型、独立 KDA 质量 / 解码、消融以及 KDA 基准路径 |
| `MAMBA_BACKEND` | `reference` | `reference`、`official`（需要 CUDA + `pip install -e '.[mamba]'`；不可用时会明确报错）或 `auto`；解析到的后端会记录到每个结果行 |
| `MAMBA2_D_STATE` / `MAMBA2_HEADDIM` / `MAMBA2_EXPAND` / `MAMBA2_D_CONV` | `16` / `16` / `2` / `4` | 质量 / 消融 / 矩阵 runner 的小模型 Mamba-2 几何参数（headdim 自动钳位到能整除 `expand × d_model` 的值） |
| `RUN_ALL_PROFILE` | `screen` | `run_all.py` 的 `smoke` / `screen` / `full` 预算档（预设只碰你没有显式 export 的环境变量） |

---

## 测试

```bash
# 自定义 runner（290+ 项检查，含长耗时的正确性检查）：
python run_correctness.py

# pytest 兼容：测试函数遵循标准 `test_*` 命名约定，
# 因此也能被 pytest 发现：
pip install pytest
pytest -q run_correctness.py
pytest -q test_figures.py
```

自定义 runner 是规范入口——它输出结构化的 `results/exp1_correctness.json` 和供 `run_all.py` 编排器消费的通过 / 失败汇总。开发过程中用 `pytest` 做选择性 / 并行运行。

---

## 结果文件

| 文件 | Schema | 说明 |
|---|---|---|
| `results/exp1_correctness.json` | `[{name, status, detail}, ...]` | 自定义 runner 的逐检查行 |
| `results/exp2_benchmark.json` | `[{T, op, time_ms, peak_mem_MB, device, repeats, compute_boundary, n_layers, note}, ...]` | **`compute_boundary` 因算子而异**——见*公平性说明* |
| `results/exp3_kv_cache.json` | `[{T, op, accounting_mode, accounting_semantics, kv_bytes, kv_elements, ...}, ...]` | 解析模型，非实测 |
| `results/exp4_mqar.json` | `{metadata, results: [{op, n_kv, per_seed: [...], mean_acc, std_acc, ci95_acc, chance_acc, conclusions_valid, ...}, ...]}` | 多种子 + CI95 + Bonferroni；行内带 `task` + `n_params`，默认包含 `mamba2` 算子 |
| `results/exp4b_quality_tasks.json` | `{metadata, results: [...]}` | kv-update / long-range / multi-agg 探针（`MQAR_TASKS`）使用同一信封 |
| `results/exp5_ablation.json` | `[{ratio, layout, n_kv, per_seed, mean_acc, ...}, ...]` | 与 exp4 相同的信封但无 metadata 包装（`ABL_MODE=ratio`） |
| `results/exp5b_ablation_matrix.json` | `[{group, memory, layout, n_kv, per_seed, mean_acc, n_params, total_layers, d_model, ...}, ...]` | 八组矩阵及其预设（`ABL_MODE=matrix`）；`metadata.design_audit.capacity_audit` 标注参数量混杂 |
| `results/exp5b_ablation_matrix_contrasts.json` | `[{n_kv, memory, contrast, mean_diff, t_stat, sd_diff, n_common_seeds, predicted_direction, capacity_matched, claim_scope, interaction_type, interaction_cells, ...}, ...]` | 配对对比；含 `I_route(D)=(D-A)_route1-(D-A)_route2` 与逐行容量记账 |
| `results/exp8_mechanism_ablation.json` | `{metadata, results: [{variant, n_kv, per_seed, mean_acc, paired_vs_baseline_*, gold_pr_auc, ...}, ...]}` | `metadata.factorial_interactions.results` 保存预注册的 RoPE × selection-density 2×2 交互 |
| `results/exp6_decoding.json` | `[{op, prefill_ms, mean_decode_ms_per_token, median_decode_ms_per_token, peak_mem_MB, uses_incremental_cache, prefill_cache_build, compute_boundary, n_layers, ...}, ...]` | softmax / KDA / CSA / HCA / 混合（独立 CSA/HCA 与混合模型使用增量解码缓存；`n_layers` 标注混合模型是 5 层栈而其余是单层方法） |
| `results/experiment_design.json` | `{schema_version, kind: "experiment_design", metadata, results: [{plan_id, factors, sesoi, planned_seeds, planning, status}, ...]}` | 前瞻方案 + 所需 n，由 `run_all.py` 在实验开始前写入 |
| `results/summary.json` | `{env: {git_commit, torch_version, cuda_version, kda_backend, ...}, runs: [{name, status, time_s}], n_ok, n_fail, total_time_s}` | 由 `run_all.py` 产出 |

> **已知的 schema 不一致。** 实验 4 把结果包在 `{metadata, results: [...]}` 里；其他实验输出裸列表。为了对下游图表代码（`make_figures.load` 两种都能处理）保持向后兼容，我们保留两种形状。后续工作：统一为带 `schema_version` 字段的信封。

---

## 公平性说明（解读图表前必读）

若干实验决策会影响跨算子比较。图表和 JSON 输出会尽可能标注这些点，但它们很容易被忽略：

### 1. 基准测试的计算边界（实验 2）

`run_benchmark.py` 里的算子在**不同的边界**下计时：

| 算子 | 边界 | 包含的内容 |
|---|---|---|
| `softmax`, `kda_rec`, `kda_chunk`, `mamba2_scan` | `core` | 只计注意力 / 递推 kernel；q/k/v（以及 g/beta、x/dt/B/C）在计时区之外预先投影。`mamba2_scan` 与 `kda_rec` 容量匹配（4 头 × 32×32 状态） |
| `csa`, `hca`, `mamba2_layer` | `end_to_end_single_layer` | 单层端到端：输入投影 + 压缩 / conv + 索引器 / scan + 注意力 + 输出投影 |
| `hybrid`, `hybrid_mamba2` | `end_to_end_multi_layer` | 5 层栈，含 LayerNorm、投影、注意力、状态管理；两行的 CSA/HCA 几何完全相同，因此这一对隔离了记忆算子的替换 |

这些数字**不可**直接当作"算子延迟"来比较。`make_figures.fig_benchmark` 会按 `compute_boundary` 拆成不同子图，并在图注中打印警告。本仓库提交的旧 `results/exp2_benchmark.json` 早于 `compute_boundary` 字段；请用 `python run_benchmark.py` 重新生成以获得该元数据，或依赖 `make_figures.py` 中的 `_OP_BOUNDARY_FALLBACK` 映射处理旧文件。

### 2. softmax 基线的训练步数（实验 4）

主对比给**每个算子相同的 optimizer 步数预算**（`MQAR_STEPS`，默认 200）。这避免让 softmax 占到额外优化的便宜。由于 softmax 收敛可能更慢，可选的敏感性运行可以显式把 `MQAR_SOFTMAX_STEPS` 设为更大的值；这类运行不是固定预算对比，必须单独标注。每种子的 `steps` 字段记录实际使用的预算。

### 3. MQAR 的统计功效（实验 4）

默认 MQAR 配置（`vocab=16, seq_len=16, n_kv=1`）的随机水平准确率为 6.25%。若干消融落在随机水平上下几个百分点内，因此 `run_quality.py` 中的 `conclusions_valid` 标志和 Bonferroni 校正的 t 检验才是权威信号——只看 `mean_acc` 会有误导性。请把接近随机水平的结果当作*冒烟级质量探针*，而不是结构性结论。目前报告的统计检验主要是**相对随机基线**的，不是算子两两之间的检验；除非你补做并报告显式的两两检验，否则不要声称算子 A 显著优于算子 B。

### 4. 解码实验的范围（实验 6）

`run_decoding.py` 对 **softmax、KDA、Mamba-2、CSA、HCA 以及混合栈（两条记忆路由）** 同时测量 prefill 和逐 token 解码延迟。CSA 和 HCA 使用 `ops_decoding_cache.py` 实现的增量解码缓存（`CSADecodingCache` / `HCADecodingCache`），它维护：

* **部分 token 累加器**（Python `list`）——缓冲 0..m-1 个新 token，直到能凑出一个压缩块；
* **压缩块缓存**（`[B, n_blocks, c]` 张量）——每 `m`（CSA）/ `m2`（HCA）个 token 增加一行；
* **滑窗环形缓冲**（`[B, win, c]` 张量）——最近 `win` 个未压缩局部 key 的定长 FIFO；
* **动态更新的索引器 key 缓存**（仅 CSA，`[B, n_blocks, c_I]` 张量）——与压缩块缓存同步增长，使 lightning 索引器不必重算完整历史就能为新块打分。

缓存避免了每步解码都对整个前缀重新压缩，但注意力 / 索引器的工作量仍随已完成压缩块的数量增长：CSA 大约是 **O(T/m + win)**，HCA 大约是 **O(T/m2 + win)**（另有 Python 参考实现的开销）。独立 CSA/HCA 行和混合行在 token 解码期间使用这些缓存；混合包装器为完整 KDA+CSA+HCA 栈的每个 CSA/HCA 子层各接入一个缓存，使 CSA/HCA 层能看到 prefill 历史而不只是当前 token。小型解码基准使用的几何比实验 4/5 的模型更轻（`csa_topk=2`、`c=8`、`dc=8`、`m2=4`、`d_model=64`，保留稀疏 top-k 路径）；它的逐 token 解码数字不能与实验 4/5 的质量结果正面比较。启用缓存的 CSA/HCA/混合行的 `prefill_ms` 包含正确性优先的 Python 缓存填充（JSON 中的 `prefill_cache_build`：独立 CSA/HCA 为 `"batched_append_from_naive_output"`，混合模型为 `"batched_populate_layer_by_layer"`）；因此它是一个偏保守的参考封装 prefill 数字。逐 token 解码耗时才是缓存效率的主要信号。

### 5. KV-cache + FLOPs 是解析式（实验 3）

`run_kv_cache.py` 用闭式公式计算 KV 字节数和 FLOPs（由算子定义推导，并对 ceil 块数、因果条目、输入**与分组输出**投影项、以及增量 CSA/HCA 运行时状态——部分 token 累加器、固定滑窗环形缓冲容量、CSA 重叠状态——做了修正）。它们**不是**从真实前向传播测得的。`run_correctness.py` 中的单元测试（`test_prefill_flops_*`、`test_kv_cache_ceil_block_count`）用手算期望值验证公式；生产级结论请用 `torch.cuda.memory_allocated` 和 FLOP 计数器交叉验证。

### 6. 近重复泄漏与数据指纹

LM 数据协议在切分前去除精确重复以及空白 / Unicode 归一化后的重复，并审计相同的截断模型窗口。它还审计**近重复**——同一个故事只改了几个词——精确哈希看不到这种情况，而如果两份拷贝横跨 train/val/test，就会抬高留出集分数。`near_duplicates.py` 按 token n-gram Jaccard（默认阈值 `0.8`，5-gram）比较截断的 `seq_len+1` token 窗口，用 MinHash+LSH 找候选，再用精确的第二遍计算所报告的相似度。跨切分审计默认在 `train_lm_autodl.py` 中运行；`--dedup_near` 在切分前去除近重复；`--strict_near_dup` 在仍存在任何跨切分对（或候选上限导致审计不完整）时拒绝训练。审计是结构性的（不依赖模型指标），因此可以在测试集仍处于封存状态时运行。完整契约、记录的 `near_duplicate_audit` 字段、候选截断的注意事项以及语料指纹框架的修正见 [docs/data-integrity.md](docs/data-integrity.md)。

---

## 局限性

* **KDA 要求数值稳定的单位范数 `q`/`k`。** `naive_recurrent_kda` 和 `naive_chunk_kda` 都不强制这一点——没有任何机制阻止调用方传入原始（未归一化）的 `q`/`k`。除非有特别理由，调用 KDA 前务必先 `F.normalize(q, dim=-1)` / `F.normalize(k, dim=-1)`。
* **参考实现是 Python 循环。** KDA 的递归路径是对时间的 Python `for` 循环；分块路径仍有对 chunk 的 Python 循环。CSA 的索引器按头循环。可选的加速路径（`ops_kda.compiled_recurrent_kda`、`ops_kda.scripted_chunk_kda`，以及 GPU 上的 `HybridConfig(kda_backend="fla")`）可以提供更好的性能。朴素路径仍是正确性 / 可读性优先的默认选择。
* **CSA 索引器的 STE。** 默认的直通估计器（`ste_mode='topk_columns'`）把梯度路由到 `soft_weights` 中 top-k 选中的列（见 `ops_csa.py::naive_csa`）。未被选中的块收不到"你本该被选中"的梯度，除非选择可选的 `ste_mode='full_softmax'` 稠密代理；`aux_contrastive_loss` 仍是未来工作。STE 路径仅用于训练：`naive_csa` 和解码缓存在 `torch.no_grad()` 下自动跳过软代理，使推理 / 基准延迟测量的是硬 top-k 路径而非训练代理。
* **融合 6 个 KV/索引器投影。** `naive_csa` 默认把 `W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx` 融合为一次 `F.linear` 调用（`fuse_projections=True`），纯粹是 kernel 启动次数的优化。如果你需要原始的 6 次独立 matmul 路径（例如在 `use_ste=False` 下保留 `W_KV_idx`/`W_Z_idx` 的 `None` 梯度），请传 `fuse_projections=False`。
* **参考算子的滑窗基于 unfold。** `naive_csa` 和 `naive_hca` 用 `_sliding_window_scores` 计算窗口分支（`unfold`，峰值显存 `O(T · win · c)`），因为窗口得分必须与压缩分支得分和 attention sink 共享同一个 softmax（DeepSeek-V4 Eq. 27）。`ops_csa._sliding_window_attention` 是独立的分块辅助函数（峰值显存 `O(chunk_t · win · c)`），供不需要联合 softmax 的调用方使用；它与 unfold softmax 路径数值一致。需要显存上界的大上下文实验直接调用该分块辅助函数（`ops_csa.py` 中的 `ops_csa._sliding_window_attention`）；参考 `naive_csa` / `naive_hca` 路径保持 unfold 实现。
* **增量解码缓存（CSA / HCA）。** `ops_decoding_cache.CSADecodingCache` / `HCADecodingCache` 让 CSA / HCA 支持逐 token 自回归解码。缓存维护部分 token 累加器、压缩块缓存、滑窗环形缓冲以及（CSA 的）动态更新的索引器 key 缓存。缓存避免了每步解码的全前缀重压缩；每 token 的工作量仍随已完成的压缩块增长（CSA 约 O(T/m + win)，HCA 约 O(T/m2 + win)）。实验 6 的混合包装器为完整栈的每个 CSA/HCA 子层各接入一个这样的缓存。
* **Dropout 未实现。** `HybridConfig.dropout != 0` 会抛出 `NotImplementedError`，而不是静默无操作。MQAR 规模的模型不需要 dropout；更大规模的训练应自行接入（或删掉该字段）。
* **因果块掩码是严格前置的（与论文精确一致）。** `_causal_block_mask` 使用 `b < t // m`（DeepSeek-V4 Eq. 16：query 只关注索引严格小于 `floor(t/m)` 的块）。因此一个压缩块最早在下一个块的第一个 token 处才可见；query 自身所在的块——无论是否完整——永远无法通过稀疏分支被访问，其局部上下文由滑窗分支提供（§2.3.3）。解码缓存与 FLOPs 核算（`run_kv_cache.causal_block_entries`）使用同一规则。（历史注记：更早的修订使用窗口关闭变体 `b < (t + 1) // m`，它会在块的最后一个源 token 处暴露 query 自身的块——因果上安全但与论文差一；在 2026-08 修复轮中与下文的部分 RoPE 一起对齐到论文精确掩码。）
* **部分 RoPE（DeepSeek-V4 §2.3.3）已实现，且模型级默认开启。** query、压缩 KV 条目（在共享 KV 的 MQA 中同时充当 key 与 value）以及滑窗 key 在其末 `rope_dim` 维旋转（论文值 64，小实验几何下自动钳位到头维 `c`），核心注意力的输出在取负的 query 位置处逆旋转，使每个 KV 条目的贡献携带相对（条目 − query）距离。论文留白的解释性选择记录在 `ops_rope.py`：压缩条目 `s` 锚定在位置 `s·m`（块起点），输出对策按 −t（query 位置）旋转。`HybridConfig.csa_rope_dim` / `hca_rope_dim` 默认为 64；设为 `None`/0 可关闭（修复前的历史行为）。原始算子（`naive_csa` / `naive_hca`）为向后兼容保持 `rope_dim=None` 默认值，增量解码缓存也接受同样的 `rope_dim` / `rope_base` 参数——开启 RoPE 时 prefill 与逐 token 解码在 fp32 容差内一致（由 `test_csa_hca_rope_prefill_vs_cache` 钉住）。索引器刻意不做 RoPE：§2.3.3 把 RoPE 的范围限定在核心注意力，Eq. 15–16 不含位置项。
* **query/KV 归一化：默认 L2，可选论文式 RMSNorm。** §2.3.3 规定对 query 的每个头以及压缩 KV 条目做 RMSNorm；仓库的历史行为是对两者做 L2 归一化（余弦注意力，`scale=1.0`）。`naive_csa` / `naive_hca` 接受 `qk_norm_mode='rms'` 以使用论文式 RMSNorm（此时数值保持范数 ~`sqrt(c)`，并自动选用标准的 `1/sqrt(c)` 温度，与 DeepSeek-V2/V3 将逐头 RMSNorm 与标准缩放结合的方式一致）。论文没有固定核心注意力的温度，因此两种形式都与论文兼容；`'l2'` 仍为默认值，以保证仓库历史实验数字可复现。
* **余弦缩放固定为 1.0（默认 'l2' 模式）。** `naive_csa` / `naive_hca` 对 `q` 和 `C_comp` 做 L2 归一化并使用 `scale=1.0`（一次刻意的修复——旧的 `c ** -0.5` 会把 softmax 压平）。仓库实验还为 CSA lightning 索引器传 `normalize_qk=True`，使 top-k 选择基于方向而非被 q_idx / K_idx 的向量范数主导——注意这个索引器归一化是仓库侧的选择，不是论文要求（Eq. 15–16 定义的是普通的加权 ReLU 点积）。没有暴露可学习温度。传 `scale=` 可显式覆盖。

---

## 论文

实现遵循以下两篇 arXiv 报告。PDF 不再提交到本仓库（见 `.gitignore`）；请从 arXiv 下载：

* [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv:2606.19348)](https://arxiv.org/abs/2606.19348)
* [Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)

## 许可证

见 `LICENSE`。

---

## 新增：真实LM训练 + AutoDL <120元 (2026-07)

本仓库原有6个实验为正确性/benchmark/MQAR探针，新增 `train_lm_autodl.py` 使用 `ops_fused.HybridKCHAttention` 做真实语言模型训练，支持Kaggle和AutoDL，成本远低于120元。

### 2026-08 论文对齐 + 回归修复（影响实验数值，请重新生成 results/）

对照 arXiv:2510.26692 / arXiv:2606.19348 复审后修复：

1. **KDA 分块路径回归修复（P0）**：此前一轮自动修复把 A 矩阵掩码从
   `diagonal=0` 误改回 `diagonal=1`，导致 `naive_chunk_kda` /
   `scripted_chunk_kda` 与递归路径偏差 O(1e-2)（25 项正确性检查失败，
   含 fp64 梯度一致性）。已恢复 `diagonal=0` 并加注释防止再次回归。
2. **因果块掩码 off-by-one（P2-1）**：`_causal_block_mask` 由窗口关闭式
   `b < (t+1)//m` 改为论文 Eq. 16 的严格前置式 `b < t//m`。同步更新了
   解码缓存、FLOPs 统计及对应测试。
3. **补齐 §2.3.3 部分 RoPE 与输出逆旋转（P2-2）**：新增 `ops_rope.py`；
   CSA/HCA 的 query、压缩 KV、滑窗 key 在末 `rope_dim` 维（论文值 64，
   按头维钳位）旋转，输出按 −t 逆旋转。模型层默认开启；原始算子默认
   关闭（向后兼容）；增量解码缓存同样支持。
4. **滑窗分支改回 Eq. 27 联合 softmax**：压缩块与滑窗 token 共享单一
   softmax 分母（含 sink），不再是两支路各自 softmax 后相加。
5. 其他：`qk_norm_mode='rms'` 选项（论文 §2.3.3 RMSNorm 形式）；KDA 解码
   fp16 状态往返精度修复；索引器 cosine 归一化归因注释修正；Kaggle
   兼容（放宽 `run_all.py` 版本门禁为警告、取消 torch 上界）。

正确性套件现为 267 项检查，全部通过。由于掩码与 RoPE 变更，CSA/HCA
相关实验数值会有变化，请用 `python run_all.py` 重新生成 `results/`。

### 为什么不替换原有ops
对比见 [COMPARISON.md](COMPARISON.md)：原有 `ops_kda/csa/hca` 已处理 g clamp、STE、sink、NaN-safe、chunked SW、decoding cache 等 20+ 处边界 case，比早期 toy 实现更适合发论文。因此保留原实现，仅增加训练管线。

### AutoDL一键运行
```bash
# max_steps 表示 optimizer steps；小模型默认配置在3090/4090上成本远低于120元
pip install -r requirements.txt
pip install transformers datasets accelerate
python train_lm_autodl.py --autodl --max_steps 2000 --seq_len 1024 --batch_size 2
```

### Kaggle
```python
!pip install -q transformers datasets
!python train_lm_autodl.py --kaggle
```
默认Kaggle配置：d=256, 5层(3:1:1), seq512, 500 steps，T4 单卡约40分钟完成
（脚本使用单 GPU，无需 DDP；在 T4x2 实例上只会用到其中一张卡）。

> **注意（避免 torch 被降级）**：Kaggle GPU 镜像预装 CUDA 版 torch，直接
> `!pip install -q transformers datasets` 即可，不要在 Kaggle 上执行
> `pip install -r requirements.txt` / `pip install -e .` 之外的 torch 重装操作。
> 本仓库的依赖区间是 `torch>=2.2,<3`，对预装版本是 no-op；若你手动 pin 了
> `<2.7` 之类的旧上限，pip 会把 CUDA torch 降级成 CPU wheel，GPU 将不可用。

### 成本拆解
| 阶段 | 时长 3090 | 成本 |
|---|---|---|
| Phase1 1024 ctx 2000 optimizer steps | 数小时级 | 远低于120元 |
| Phase2 4096 ctx 500 optimizer steps | 视GPU而定 | 远低于120元 |
| 总计 | 视GPU而定 | <<120元 |

即便放大到 d=512, 12层, 4000 steps，4090 4h ≈ 11元，仍在预算内。

运行产生 `checkpoints_lm/final_lm.pt` 可用 `run_quality.py`/`run_decoding.py` 进一步评估。
