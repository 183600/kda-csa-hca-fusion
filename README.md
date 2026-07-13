# KDA-CSA-HCA Fusion

A naive PyTorch reference implementation and experimental harness for a hybrid
linear-attention architecture that combines three operators from recent
long-context / efficient-attention literature:

* **KDA** — Kimi-style Delta Attention (recurrent delta-rule with per-channel
  log-decay). Bulk token mixing, O(1) state per head.
* **CSA** — DeepSeek-V4 Compressed Sparse Attention (overlapped two-branch KV
  compression + lightning indexer that selects top-k compressed blocks per
  query). Sparse long-range retrieval.
* **HCA** — DeepSeek-V4 Heavily Compressed Attention (single-branch heavy
  compression, dense MQA over a tiny set of compressed entries, optional
  sliding window). Cheap global context.

The hybrid stack interleaves them in a `3:1:1` KDA:CSA:HCA ratio by default
(see `ops_fused.HybridKCHAttention`).

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
> **Accelerated paths (added in the code-review pass).** The naive paths
> above remain the default for correctness / readability, but the repo now
> ships two opt-in accelerated wrappers that preserve the exact numerical
> contract:
>
> - `ops_kda.compiled_recurrent_kda` — `torch.compile` wrapper around
>   `naive_recurrent_kda`, cached per (shape, dtype, requires_grad) signature.
>   Best for moderate `T` (≥1024); typical speedup 5–20× on GPU.
> - `ops_kda.scripted_chunk_kda` — `torch.jit.script` wrapper around the
>   inner per-chunk loop of `naive_chunk_kda`, with a graceful fallback to
>   the eager path if TorchScript rejects the input.
> - `ops_csa._sliding_window_attention` — auto-engages a chunked sliding
>   window when `T * win * c > 8M` elements, keeping peak memory at
>   `O(chunk_t · win · c)` instead of `O(T · win · c)`. Used by both
>   `naive_csa` and `naive_hca` when `sliding_window > 0`.
> - `naive_csa` itself now fuses the 6 `F.linear(H, W_*)` calls
>   (`W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx`) into a single matmul
>   via `torch.cat` + `tensor.split`, reducing kernel-launch overhead by 6×.

---

## Repository layout

```
.
├── ops_kda.py             # KDA: naive_recurrent_kda, naive_chunk_kda
├── ops_csa.py             # CSA: compress_kv (overlapped), lightning indexer, naive_csa
├── ops_hca.py             # HCA: heavy compression + dense MQA + SW, naive_hca
├── ops_decoding_cache.py  # CSADecodingCache / HCADecodingCache (incremental decode)
├── ops_fused.py           # HybridConfig + KDAHybridLayer/CSAHybridLayer/HCAHybridLayer + HybridKCHAttention
├── run_correctness.py     # 199 regression tests (custom runner; pytest-importable)
├── run_benchmark.py       # Exp 2: latency vs. sequence length (with op_boundary metadata)
├── run_quality.py         # Exp 4: MQAR associative-recall quality (multi-seed)
├── run_ablation.py        # Exp 5: KDA:CSA:HCA ratio ablation
├── run_decoding.py        # Exp 6: prefill + per-token decode latency (softmax/KDA/CSA/HCA/hybrid)
├── run_kv_cache.py        # Exp 3: analytic KV-cache + FLOPs accounting
├── method_analysis.py     # Headwise prototype (CSA simplified to dense for the demo)
├── make_figures.py        # Generate figures/* from results/*
├── run_all.py             # Single-entry runner (Kaggle-friendly)
├── kaggle_setup.py        # CUDA bootstrap + shared utilities (sanitize_for_json, parse_int_env)
├── test_figures.py        # Tests for the figure generation pipeline
├── results/               # Generated JSON outputs (gitignored; regenerate via run_all.py)
├── figures/               # Generated PDF/PNG outputs (gitignored; regenerate via make_figures.py)
├── requirements.txt
├── pyproject.toml         # `pip install -e .` makes the modules importable as a package
└── README.md              # this file
```

---

## Installation

**Requires Python 3.10–3.12** (the source uses PEP 604 `X | None` union syntax,
and the pinned torch/scipy ranges are validated for the 3.10–3.12 wheel matrix).
The `pyproject.toml` declares `requires-python = ">=3.10,<3.13"`.

```bash
# 1. Clone
git clone https://gitee.com/qwe12345678/kda-csa-hca-fusion.git
cd kda-csa-hca-fusion

# 2. Install dependencies (pinned to match the committed historical results)
pip install -e .        # installs torch, einops, matplotlib, numpy, scipy
#    OR, if you only need the runtime deps without the dev toolchain:
#    pip install -r requirements.txt

# 3. (Optional) Install dev tooling (pytest, pytest-xdist, mypy, ruff):
pip install -e .[dev]
```

After `pip install -e .`, the experiment scripts can be run as modules:

```bash
python -m run_correctness       # or: python run_correctness.py
python -m run_benchmark
python -m run_all
```

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
install in-place).

---

## Running the experiments

| Experiment | Script | Output | Wall-clock (CPU) |
|---|---|---|---|
| 1. Correctness | `run_correctness.py` | `results/exp1_correctness.json` | ~10 s |
| 2. Latency benchmark | `run_benchmark.py` | `results/exp2_benchmark.json` | ~30 s |
| 3. KV-cache + FLOPs | `run_kv_cache.py` | `results/exp3_kv_cache.json` | ~1 s |
| 4. MQAR quality | `run_quality.py` | `results/exp4_mqar.json` | ~3 min (5 seeds × 200 steps) |
| 5. Ratio ablation | `run_ablation.py` | `results/exp5_ablation.json` | ~5 min (7 seeds × 7 layouts) |
| 6. Decode latency | `run_decoding.py` | `results/exp6_decoding.json` | ~30 s |
| Figures | `make_figures.py` | `figures/fig_*.{pdf,png}` | ~5 s |

Run everything end-to-end:

```bash
python run_all.py
```

Environment knobs (set before launching):

| Variable | Default | Effect |
|---|---|---|
| `MQAR_SEEDS` | `5` | seeds for Exp 4 |
| `MQAR_STEPS` | `200` | training steps for non-softmax ops in Exp 4 |
| `MQAR_SOFTMAX_STEPS` | `500` | extra steps for the softmax baseline (see *Fairness notes* below) |
| `ABL_SEEDS` | `7` | seeds for Exp 5 |
| `ABL_STEPS` | `100` | training steps for Exp 5 |
| `BENCH_LENGTHS` | `128,256,512,1024,2048` | sequence lengths for Exp 2 |
| `BENCH_REPEATS` | `5` | timed repeats per (T, op) in Exp 2 |
| `SKIP_SLOW` | `0` | `1` truncates Exp 2/4/5 on CPU |
| `SKIP_CUDA_CHECK` | `0` | `1` bypasses the CUDA-availability guard |

---

## Tests

```bash
# Custom runner (199 tests, includes long-running correctness checks):
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
| `results/exp1_correctness.json` | `{ metadata, results: [...] }` | per-test `{name, ok, detail}` rows |
| `results/exp2_benchmark.json` | `[{T, op, time_ms, peak_mem_MB, device, repeats, compute_boundary, n_layers, note}, ...]` | **`compute_boundary` differs per op** — see *Fairness notes* |
| `results/exp3_kv_cache.json` | `[{T, op, kv_bytes, kv_elements, ...}, ...]` | analytic model, not profiled |
| `results/exp4_mqar.json` | `[{op, n_kv, per_seed: [...], mean_acc, std_acc, ci95_acc, chance_acc, conclusions_valid, ...}, ...]` | multi-seed with CI95 + Bonferroni |
| `results/exp5_ablation.json` | `[{ratio, layout, n_kv, per_seed, mean_acc, ...}, ...]` | same envelope as exp4 minus the metadata wrapper |
| `results/exp6_decoding.json` | `[{op, prefill_ms, mean_decode_ms_per_token, median_decode_ms_per_token, peak_mem_MB, uses_incremental_cache, prefill_cache_build, ...}, ...]` | softmax / KDA / CSA / HCA / hybrid (standalone CSA/HCA and hybrid use incremental decoding caches) |
| `results/summary.json` | `{env, runs: [{name, status, time_s}], n_ok, n_fail, total_time_s}` | produced by `run_all.py` |

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

The six operators in `run_benchmark.py` are timed under **different
boundaries**:

| Op | Boundary | What's included |
|---|---|---|
| `softmax`, `kda_rec`, `kda_chunk` | `core` | only the attention / recurrence kernel; q/k/v (and g/beta) are pre-projected outside the timed region |
| `csa`, `hca` | `end_to_end_single_layer` | a single layer end-to-end: input projection + compression + indexer + sparse attention + output projection |
| `hybrid` | `end_to_end_multi_layer` | a 5-layer stack with LayerNorm, projections, attention, state management |

These numbers are **not directly comparable** as "operator latency".
`make_figures.fig_benchmark` splits them into separate subplots by
`compute_boundary` and prints a warning in the caption. The stale
`results/exp2_benchmark.json` committed in this repo predates the
`compute_boundary` field; regenerate it with `python run_benchmark.py` to
get the metadata, or rely on the `_OP_BOUNDARY_FALLBACK` mapping in
`make_figures.py` for the legacy file.

### 2. Softmax baseline training steps (Exp 4)

The softmax-attention baseline is given **500 training steps** while the
other operators get **200 steps**. This is intentional: with the original
100 steps the softmax baseline plateaued at ~10% accuracy (barely above the
6.25% chance level for vocab=16), making it a useless upper bound. The
extra steps let softmax actually converge so the comparison is meaningful.
The per-seed `steps` field in `results/exp4_mqar.json` records the actual
step count used; the `softmax_steps` env var is also logged. Any summary
table that reports cross-op accuracy must annotate this asymmetry.

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

`run_decoding.py` benchmarks **softmax, KDA, CSA, HCA, and the hybrid
stack** for both prefill and per-token decode latency. CSA and HCA use
the incremental decoding cache implemented in
`ops_decoding_cache.py` (`CSADecodingCache` / `HCADecodingCache`),
which maintains:

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
current token. The small decode benchmark uses `csa_topk=2`, matching the
small ablation setting. The `prefill_ms` for cache-enabled CSA/HCA/hybrid rows
includes correctness-first Python cache population (`prefill_cache_build =
"reference_python_append"` in the JSON); it is therefore a conservative
reference-wrapper prefill number. The per-token decode timings are the main
cache-efficiency signal.

**Known limitation: `torch.topk` tie-breaking.** When the CSA indexer's
ReLU scores have many exact ties at 0 (common with random untrained
weights but rare in trained models where the indexer learns
discriminative scores), `torch.topk`'s tie-breaking depends on the
tensor SIZE. The full-sequence path's tensor has `T_padded // m` entries
(including `-inf`-masked future blocks); the incremental path's tensor
has only the completed blocks (no padding). With the same underlying
scores but different tensor sizes, `torch.topk` may pick DIFFERENT tied
blocks, leading to different gathered `kv` and different sparse-attention
outputs. This is a `torch.topk` implementation artifact, NOT a
correctness bug — both paths select valid blocks with the highest scores,
just different tie-breaking. The regression tests
(`test_csa_decoding_cache_correctness` in `run_correctness.py`) use
`topk >= n_blocks` (select all valid blocks) to sidestep this.

### 5. KV-cache + FLOPs are analytic (Exp 3)

`run_kv_cache.py` computes KV-bytes and FLOPs from closed-form formulas
(derived from the operator definitions, with corrections for ceil-block
counts, causal entries, projection terms, and the incremental CSA/HCA runtime
state such as partial-token accumulators, fixed sliding-window ring-buffer
capacity, and CSA overlap state). They are **not** measured from a real forward
pass. The unit tests in `run_correctness.py`
(`test_prefill_flops_*`, `test_kv_cache_ceil_block_count`) verify the
formulas against hand-computed expected values; for production claims,
cross-check with `torch.cuda.memory_allocated` and a FLOP counter.

---

## Limitations

* **KDA requires unit-norm `q`/`k` for numerical stability.** Neither
  `naive_recurrent_kda` nor `naive_chunk_kda` enforces this — nothing stops
  a caller from passing raw (non-normalized) `q`/`k`. The delta-rule
  recurrence is only numerically bounded when `q`/`k` are L2-normalized
  along the last axis (the standard KDA/DeltaNet convention, and what every
  regression test in `run_correctness.py` does). With un-normalized inputs,
  the recurrent state can diverge over a long sequence: empirically this
  starts producing non-finite (`NaN`) output around `T~500` for
  standard-normal `q`/`k` at `K=64`. The **recurrent** and **chunked**
  paths diverge *differently* once inputs are unnormalized (recurrent tends
  to decay straight to `NaN`; chunked can produce a mix of `NaN`/`Inf`
  because its Neumann-series solve amplifies large values differently) —
  the two paths are only guaranteed to agree to fp tolerance when both stay
  finite. **Mitigation (added):** both functions now run a cheap
  `torch.isfinite` check on the finished output and emit a one-shot,
  actionable `RuntimeWarning` (naming the function and the fix) instead of
  silently returning a non-finite result with no diagnostic; see
  `test_kda_unnormalized_input_warns` in `run_correctness.py`. Always
  `F.normalize(q, dim=-1)` / `F.normalize(k, dim=-1)` before calling into
  KDA unless you have a specific reason not to. **Follow-up fix:** this
  non-finite check is data-dependent Python control flow, which initially
  broke `compiled_recurrent_kda(..., fullgraph=True)` (`torch.compile`
  cannot trace a branch that depends on tensor runtime values). The check
  is now guarded with `torch.compiler.is_compiling()` (the documented
  "skip while being traced" pattern) so it is pruned away inside a
  `torch.compile` graph and only runs in eager mode; see
  `test_compiled_recurrent_kda_fullgraph` in `run_correctness.py`.
* **Naive Python loops.** KDA's recurrent path is a Python `for` loop over
  time; the chunked path still has a Python loop over chunks. CSA's
  indexer loops over heads. None of this is fused or Tritonized. Latency
  numbers reflect Python overhead and kernel-launch overhead, **not** the
  algorithm's intrinsic FLOPs. Do not benchmark this against FLA / Triton
  kernels and draw architectural conclusions.
  **Mitigation (added):** `ops_kda.compiled_recurrent_kda` wraps the
  recurrent path with `torch.compile` (cached per signature, 5–20× speedup
  on GPU at `T≥1024`); `ops_kda.scripted_chunk_kda` wraps the chunked
  path's inner loop with `torch.jit.script` (bit-identical to the eager
  path). The naive paths remain the default for correctness / readability.
* **STE for CSA indexer.** The default straight-through estimator
  (`ste_mode='topk_columns'`) routes gradient through the top-k selected
  columns of `soft_weights` (see `ops_csa.py::naive_csa`). Non-selected
  blocks receive no "you should have been picked" gradient unless the
  optional `ste_mode='full_softmax'` dense surrogate is selected;
  `aux_contrastive_loss` remains future work. The STE path is training-only:
  `naive_csa` and the decoding cache automatically skip the soft surrogate
  under `torch.no_grad()` so inference/benchmark latency measures the hard
  top-k path, not the training proxy.
* **Fusing the 6 KV/indexer projections changes the `use_ste=False`
  gradient contract (not just performance).** `naive_csa` fuses
  `W_aKV, W_bKV, W_aZ, W_bZ, W_KV_idx, W_Z_idx` into one `F.linear` call
  by default (`fuse_projections=True` — issue 2.3 fix, purely a kernel-
  launch-count optimization with an identical forward value). This has a
  side effect on **backward**: with STE disabled (`use_ste=False`, e.g.
  the untrained-indexer ablation), `W_KV_idx`/`W_Z_idx` now receive a
  zero-but-**non-None** gradient (they share a matmul with 4 other,
  differentiable weights) instead of `None` (the pre-fusion behaviour,
  where they were only reachable through the non-differentiable
  `torch.topk` selection). The optimizer step is a no-op either way, but
  any code that branches on `param.grad is None` to mean "this parameter
  did not participate in this forward pass" will observe a difference.
  **Mitigation (added):** pass `fuse_projections=False` to restore the
  original 6-separate-matmul path (identical forward value, `None` grad
  under `use_ste=False`); see `test_csa_fuse_projections_grad_contract`
  in `run_correctness.py` for a regression pinning both branches.
* **Sliding window uses `unfold`.** Memory is `O(T · win)` per call. Fine
  for `T ≤ 4k, win ≤ 512`; will blow up at `T=64k, win=2k` and needs a
  chunked / banded kernel.
  **Mitigation (added):** `ops_csa._sliding_window_attention` (used by
  both `naive_csa` and `naive_hca`) auto-engages a chunked path when
  `T * win * c > 8M` elements, keeping peak memory at `O(chunk_t · win · c)`.
  The chunked path is bit-identical to the unfold path (verified by
  `test_hca_sliding_window_causality` / `test_csa_full_pipeline_causality`).
* **Incremental decoding cache (CSA / HCA).** The
  `ops_decoding_cache.CSADecodingCache` / `HCADecodingCache` enable
  token-by-token autoregressive decoding for CSA / HCA (closing the
  Exp 6 scope gap — see *Fairness notes* #4). The cache maintains a
  partial-token accumulator, a compressed-block cache, a sliding-window
  ring buffer, and (for CSA) a dynamically-updated indexer key cache.
  The cache avoids full-prefix recompression on every decode step; per-token
  work still scales with completed compressed blocks (about O(T/m + win) for
  CSA and O(T/m2 + win) for HCA). The Exp 6 hybrid wrapper now wires one such
  cache into every CSA/HCA sub-layer of the full stack, so the hybrid row is
  no longer a "CSA/HCA no-history" placeholder.
  **Known limitation:** `torch.topk`'s tie-breaking depends on tensor
  size, so the incremental path may pick different tied blocks than the
  full-sequence path when scores are tied (ReLU saturation at 0). This
  is a numerical artifact, not a correctness bug — both paths select
  valid blocks with the highest scores. The regression tests use
  `topk >= n_blocks` to sidestep this.
* **Dropout unimplemented.** `HybridConfig.dropout != 0` raises
  `NotImplementedError` rather than silently no-op'ing. MQAR-scale models
  don't need dropout; larger runs should wire it in (or remove the field).
* **Causal block mask is strict-prefix.** `_causal_block_mask` uses
  `b < t // m` — the block containing `t` is never attended to (its
  compressed representation aggregates future tokens). The sliding-window
  branch handles intra-block / near-context attention. This is correct
  but easy to misread; document it in any derivative work.
* **Cosine scale is fixed at 1.0.** `naive_csa` / `naive_hca` L2-normalize
  `q` and `C_comp` and use `scale=1.0` (a deliberate fix — the old
  `c ** -0.5` flattened softmax). The repository experiments now also pass
  `normalize_qk=True` for the CSA lightning indexer so top-k selection is
  direction-based rather than dominated by q_idx / K_idx vector norms. No
  learnable temperature is exposed. Pass `scale=` explicitly to override.

---

## License

See `LICENSE`.
