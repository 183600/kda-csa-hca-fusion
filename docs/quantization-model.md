# Quantized-state recall / cost model (Exp 10)

`kv_quant.py` adds the **quantization axis** the rest of the repository is
missing: costs are otherwise priced in fp16 *elements*, which cannot express
"the same association at 4 bits per element". Exp 10
(`run_quant_recall.py`) crosses the Exp 5 eight-group matrix's
**memory-route axis** (KDA vs Mamba-2) with a **state-bits axis** and
measures where planted associative recall survives as the state shrinks.

## Storage contract (`QuantSpec`)

| field | meaning |
|---|---|
| `bits` | payload bits per element (int in `[2, 16]`) |
| `group_size` | elements sharing one scale / zero point (`>= 1`, last axis) |
| `symmetric` | `True` → absmax-scaled signed grid; `False` → min/max scale + zero point |
| `scale_bits` | bits charged per stored scale (default 16 = fp16; `0` = free) |
| `zero_bits` | bits per zero point; defaults to `bits` for asymmetric, forbidden for symmetric |

Cost formulas (the honesty contract — metadata is **always** charged):

```
symmetric  qmax = 2**(bits-1) - 1        asymmetric qmax = 2**bits - 1
bits_per_element  = bits + (scale_bits + zero_bits) / group_size
bytes_per_element = bits_per_element / 8
compression_vs_fp16 = 16 / bits_per_element
```

So 4-bit payload with group 64 and fp16 scales is **4.25 bits/element**
(3.76x smaller than fp16), not a nominal 4x; `scale_bits=0` is the
idealized payload-only bound and is reported separately.

## `fake_quantize` semantics

Grouped round-to-grid + dequantize in full precision:

* Groups run along the **last tensor axis** (`group_size` contiguous
  elements share a scale); a non-divisible tail is zero-padded internally
  and unpadded on the way out.
* Zero groups (all-zero) map to exact zero; constant groups reconstruct
  exactly (the asymmetric `span == 0` branch).
* Scales/zero-points are computed in fp32 under `no_grad`; the
  **straight-through estimator** (`ste=True`, default) makes
  `d(out)/d(in) == 1` almost everywhere, so the module is QAT-ready.
* Non-finite inputs are rejected with `ValueError` — a quantizer must not
  launder NaN/Inf into a plausible-looking grid value.

## Byte bridge into `kv_budget`

Element rows from the Exp 9 budget model convert to bytes with the same
component structure:

```python
from kv_budget import op_read_bytes, layout_read_bytes
from kv_quant import QuantSpec

row = op_read_bytes('kda', T=16, spec=mqar_spec,
                    quant={'state': QuantSpec(4, 64)})
row['state_bytes']              # retained state, priced at 4.25 bits/elem
row['core_read_bytes']          # per-query reads (fp16 default: 2 B/elem)
row['total_read_bytes']         # reads only; retained state excluded
row['quant_bytes_per_element']  # which policy priced which component
```

* `quant=None` prices everything at fp16 (2.0 B/element).
* A single `QuantSpec` applies to all components; a mapping prices per
  field stem (`'core'`, `'indexer'`, `'selected'`, `'window'`, `'dense'`,
  `'state'` for operator rows; `'kda'`, `'csa'`, `'hca'` for layout rows)
  with an optional `'*'` default.
* Policy names that do not exist in the row raise `ValueError` (typo
  protection: a policy that silently does nothing is worse than an error).
* `total_read_bytes` never folds in the retained state: the state is
  storage, the total is a per-query read charge (same split as
  `state_elements` vs `total_read_elements`).

## Exp 10 probe design

Planted associative recall through the **real reference operators**
(`ops_kda.naive_recurrent_kda`, `ops_mamba.naive_mamba2_scan`), one token
at a time with `initial_state`, **fake-quantizing the state after every
write**. Keys are the rows of a random orthogonal matrix (the exact-recall
best case, already pinned for KDA by
`test_budget_frontier.test_delta_rule_orthonormal_keys_exact_recall`);
values are random and decoded by nearest neighbour. The stepwise write
path with quantization disabled reproduces the single-call full-sequence
state exactly (pinned by `test_quant_recall.py`), so any recall change is
attributable to the quantizer and not to the stepping.

Geometry: `n_keys = 16`, `value_dim = 8` (state = 128 elements both
routes); Mamba-2 configured with `dt=1`, `A=0`, `D=0`, `ngroups=nheads=1`
so the state is `S[i, j] = sum_t v_t[i] * k_t[j]` and reading with
`C = k_i` returns `v_i`. The state is quantized as one flattened
contiguous buffer. Result rows: mean/min/max recall over seeds, max
read-back error, `state_bytes`, `bits_per_element`, compression, and a
conservative `exact_recall` flag (min-over-seeds == 1.0).

## Measured recall-cost frontier (3 seeds: 0, 1, 2)

```
route   cheapest exact-recall cell     bits/el   state B   vs fp16   recall(min)
kda     3-bit, group 16 (symmetric)      4.00      64.0     4.00x    1.000
mamba2  4-bit, group 64 (symmetric)      4.25      68.0     3.76x    1.000
```

Both routes collapse at 2 bits (mean recall 0.23–0.46; min 0.125) and are
exact at 4+ bits. The interesting middle is **route- and group-dependent**:

| cell | KDA recall (min..max) | Mamba-2 recall (min..max) |
|---|---|---|
| 2-bit / group 64 | 0.229 (0.125..0.312) | 0.188 (0.125..0.250) |
| 3-bit / group 64 | 0.896 (0.875..0.938) | 0.771 (0.688..0.812) |
| 3-bit / group 16 | **1.000 (exact)** | 0.917 (0.812..1.000) |
| 4-bit / group 64 | 1.000 | **1.000 (exact)** |

On this probe the delta-rule state tolerates 3-bit codes with fine groups
(4.00 bits/element, 4x smaller than fp16) while the SSD matrix state needs
4 bits (or finer groups at 3 bits) for exact recall — i.e. the
memory-route axis of the eight-group matrix buys a measurable ~1 bit of
quantization headroom on exact associative recall, in KDA's favour.

## Scope / honesty

* Single-operator probe, **no training, no weights**: it measures
  state-quantization damage to stored associations, not end-task quality.
* Orthonormal keys are the best case for exact recall; real keys are
  correlated and low-bit cells will be worse.
* Only the recurrent *state* is quantized. Activations, weights, and the
  CSA/HCA compressed-KV caches are out of scope; the byte bridge is the
  accounting tool for those components (looping it into Exp 9 result rows
  is future work).
* "Exact" = min-over-seeds == 1.0 (conservative); 3 seeds keep the probe
  at ~2 s, and the 2-bit cells carry the visible variance (min..max
  reported).
* Flattened-buffer grouping matches a state cache stored as one contiguous
  quantized buffer; per-head-dim groupings give different low-bit numbers.

## Exp 11: state-quantized decode perplexity (end-task evaluation)

Exp 10 measures stored associations; Exp 11 (`run_quant_lm.py`) asks the
end-task question: **what does state quantization cost the language-model
perplexity, and can QAT buy it back?** Two complementary parts:

* **Part A — PTQ perplexity, fp16 reference.** A tiny hybrid LM (2 KDA +
  1 CSA + 1 HCA blocks, `d_model=32`, 400 training steps) is trained on a
  deterministic offline synthetic corpus (`train_lm_autodl`'s fallback
  char tokenizer; no network). Held-out perplexity is scored
  token-by-token through the real incremental decode path
  (`run_decoding.HybridDecoding`) with the **stacked KDA state
  fake-quantized after every write**, sweeping the Exp 10 spec grid
  (bits × group sizes + asymmetric probe). `decode_ppl(spec) /
  decode_ppl(fp16)` on the same seed/path is the delta; the
  `ppl_preserving` flag uses the conservative min-over-seeds ratio.
  Crucially, the stacked state is grouped along the **flattened
  per-layer buffer**: grouping only the last state dim makes
  `group_size` a no-op (all 16s == all 64s), which is why the codec
  contract threads per-layer grouping through `quantize_stacked_state`.

* **Part B — QAT recovery (autograd support).** A standalone
  `StepwiseKDALM` (a `run_quality.KDAAttn` driven by a per-token
  `ops_kda.naive_recurrent_kda` recurrence) is trained twice **from the
  same initialization** on the same batches: once with fp16 state writes,
  once with 2-bit grouped STE state writes
  (`kv_quant.fake_quantize(..., ste=True)` on every state write). The
  recovery fraction is
  `1 - (qat_ppl - fp16_ppl) / (ptq_ppl - fp16_ppl)`, clipped to
  `[0, 1]` and aggregated mean/min over seeds; inference scoring always
  re-quantizes with `ste=False` on the same spec.

`kv_quant.fake_quantize_` is the in-place inference-only variant of
the codec: it mutates the state buffer without cloning, preserves dtype,
is idempotent, and **refuses tensors that require grad** so it can never
silently detach a training graph. The autograd contract of the STE path
is pinned by `test_quant_lm.py` (gradient exactly equals the unquantized
surrogate; `ste=False` gradient is exactly zero; a multi-step quantized
recurrence still reaches `q/k/v/g/beta`).

Measured (3 seeds: 0, 1, 2; 400 train / 400 QAT steps, held-out ppl):

| part | fp16 | 2-bit PTQ (g16) | 2-bit QAT | recovery |
|---|---|---|---|---|
| A: hybrid LM decode ppl | 1.1523 | 1.2759 (ratio 1.108) | — | — |
| B: stepwise KDA LM | 1.1708 | 36415 (ratio ~3.1e4) | 1.1876 | 100.00% (min 99.99%) |

Part A grid: 3-bit/group-16 lands at ratio 1.0034 (group-64 at 1.0106,
right at the 1.01 tolerance); 2-bit/group-64 degrades to ratio 1.689
while 2-bit/group-16 stays at 1.108 — fine groups again buy roughly a
bit of headroom. The 4/6/8-bit cells and the asymmetric probe are all
within 0.1%. Part B's standalone per-token recurrence is far more
2-bit-sensitive (perplexity explodes to ~3e4), and STE QAT pulls it
back to the fp16 level on every seed. Absolute perplexities are
toy-corpus artifacts; the measured quantity is the relative damage at
fixed weights and path.

## CI gating for the quantization track

`run_all.py` integrates the quantization track: it runs
`test_quant_recall` / `test_quant_lm` alongside the other unit-test
modules, executes Exp 10 and Exp 11 with smoke / SKIP_SLOW CPU budget
truncation, and finishes with a `results_schema_check` step asserting
every schema-managed result file present in `results/` parses as a
compatible `results_schema` payload (raw provenance side files and the
Exp 4 overfit-gate report are explicitly exempt). `ci_gate.py` covers
`run_correctness.py` / `test_figures.py`, so this wiring provides the
quantization experiments' pipeline coverage.

## bfloat16 delta-rule numerical stability

The KDA delta-rule recurrence is sensitive to reduced precision because it
repeatedly applies `exp(g_t)` (per-channel log-decay) and accumulates the
state `S` over T steps. In bfloat16 the 8-bit exponent range makes both
overflow (large positive `g_t` → `exp(g_t) → Inf`) and underflow (large
negative `g_t` → `exp(g_t) → 0`) more likely than in fp32, and the
T-step accumulation can amplify small errors.

`ops_kda_autograd.py` (the custom-autograd KDA path) includes three
bfloat16-specific stabilizers that activate **only when the compute dtype
is bfloat16**:

1. **`_bf16_safe_exp(g, clamp_min=-10.0, clamp_max=50.0)`** — clamps the
   log-gate before `exp()` and clamps the result to `[1e-5, 1e5]`. This
   prevents Inf/0 from a single channel's gate while preserving the
   effective decay range for normal values.

2. **`_state_scale_correction(S, max_norm=1e4)`** — after each state
   update, computes the Frobenius norm of `S` per head and scales it down
   if `||S||_F > max_norm`. This bounds the recurrent state magnitude and
   prevents gradual overflow over long sequences (T > 1024).

3. **`_bf16_safe_addcmul(S, beta_k, delta, max_val=1e4)`** — clamps the
   per-step update `beta_k * delta` before adding to the state. This
   limits single-step blowup from a large residual.

These patches are **no-ops for fp32/fp64** (the compute dtypes for
non-bfloat16 inputs per `ops_kda._compute_dtype`), so they do not affect
the bit-identical fp32/fp64 reference paths.

The precision sweep (`precision_sweep.py`, case `'kda'`, path `'custom'`)
verifies that with these stabilizers the bfloat16 explicit/autocast error
stays within the storage-dtype budget (`~2 u`, where `u = 2^-8` for
bfloat16) up to T=256. For longer sequences the state-scale correction
keeps the recurrence stable (tested up to T=2048 in
`test_bf16_gated_kda.py::test_bf16_long_sequence_stability`).

## GatedAttentionMixin — unified gating for KDA/CSA/HCA

`ops_gated_attention.py` provides a reusable `GatedAttentionMixin` that
adds optional per-head or per-channel gating to any attention operator:

```python
from ops_gated_attention import GatedAttentionMixin

class MyAttention(GatedAttentionMixin, nn.Module):
    def __init__(self, ..., gate_type='per_channel', gate_dim=64, num_heads=4):
        super().__init__()
        self._init_gate(gate_type, gate_dim, num_heads)

    def forward(self, q, k, v, ...):
        k = self.apply_gate(k, dim=-1)  # gate keys
        v = self.apply_gate(v, dim=-1)  # gate values
        ...
```

Gate types:
- `'none'` — identity (no parameters)
- `'per_head'` — one scalar gate per head `[H]`, broadcast over channels
- `'per_channel'` — full `[H, D]` gate matrix per head

The mixin initializes gates near 1.0 (`nn.Parameter(torch.ones(...))`) so
the initial forward is an identity mapping. This makes it safe to add
gating to existing operators without changing their initialization
semantics.

`GatedDeltaRule` is a standalone fused layer that combines per-channel
gating with the delta-rule recurrence, usable as a drop-in gated memory
module or as a building block for gated CSA/HCA variants.

## Exp 12: needle-in-a-haystack retrieval with quantization-aware evaluation

Exp 10/11 measure intrinsic state-quantization damage (planted recall) and
end-task perplexity (LM). Exp 12 (`run_needle_haystack.py`) adds the
canonical **needle-in-a-haystack** long-context benchmark: a fixed Q&A pair
("needle") is embedded in a long distractor context ("haystack"), and the
model must retrieve the answer when prompted at the end. This directly
measures long-context retrieval capability — the primary motivation for
hybrid architectures like Kimi-Linear and DeepSeek-V4.

Three integrated evaluation axes:

1. **bf16-native evaluation** — uses the bfloat16 numerical stability patches
   from ``ops_kda_autograd.py`` (for KDA) and extends them to CSA/HCA
   (``ops_csa.py``, ``ops_hca.py``) for a unified low-precision decode path
   across all operators.

2. **Quantization-aware PTQ evaluation** — reuses ``kv_quant.QuantSpec`` and
   ``fake_quantize_`` to fake-quantize the recurrent state (KDA/Mamba-2/RWKV-7)
   and compressed caches (CSA/HCA) after every decode write, measuring the
   recall-cost curve on the *same* needle task (not a planted proxy).

3. **QAT recovery probe** — trains a small hybrid LM from scratch with 2-bit
   STE state writes (mirroring Exp 11 Part B) and evaluates needle retrieval
   on the QAT model vs the fp16 baseline, reporting a recovery fraction.

The evaluation uses a **deterministic, offline haystack generator** (no
network, no external tokenizer) so it runs in CI. The haystack is constructed
from a fixed vocabulary of distractor sentences; the needle is a fixed Q&A
pair inserted at a controlled position. Retrieval is measured by exact-match
of the answer token(s) at the final position.

Output: ``results/exp12_needle_haystack.json`` (versioned envelope via
``results_schema``) with ``row_kind`` in {``needle_config``,
``needle_recall``, ``needle_summary``, ``qat_needle``,
``qat_needle_summary``}.

Entry points:
```bash
python3 run_needle_haystack.py                 # writes results/exp12_needle_haystack.json
python3 test_needle_haystack.py                # or: pytest -q test_needle_haystack.py
python3 run_needle_haystack.py --quick --no-qat # CI smoke run
```

## bfloat16 numerical stability (extended to CSA/HCA)

The KDA delta-rule recurrence (Exp 10/11) includes three bfloat16-specific
stabilizers in ``ops_kda_autograd.py``. This round extends the same pattern
to CSA and HCA operators (``ops_csa.py``, ``ops_hca.py``), which also
perform repeated softmax and attention score computations sensitive to
bfloat16's 8-bit exponent range.

Added to ``ops_csa.py`` and ``ops_hca.py`` (activate **only for bfloat16**):

1. **`_bf16_safe_softmax(logits, dim=-1)`** — clamps logits to `[-50, 50]`
   before softmax to prevent exp overflow (exp(50) ~ 5e21, well below
   bf16 max ~3e38) while preserving relative probabilities for normal values.

2. **`_bf16_safe_exp(x, clamp_max=50.0)`** — clamps before exp to prevent
   overflow in any exponential computation.

3. **`_bf16_scale_if_needed(x, max_norm=1e4)`** — scales tensor down if its
   L2 norm exceeds ``max_norm`` per head/channel, bounding recurrent state
   magnitude and preventing gradual overflow over long sequences.

4. **`_bf16_safe_einsum_sum(x, dim, max_val=1e4)`** — clamps before summing
   to limit single-step blowup from large residuals.

Applied at critical points:
- KV compression softmax (``csa_compress_kv``, ``csa_compress_kv_overlapped``)
- Sliding-window attention softmax (``_sliding_window_attention``)
- Lightning indexer STE softmax (``_nan_safe_softmax`` → ``_bf16_safe_softmax``)
- Joint core-attention softmax (``naive_csa``, ``naive_hca``)

These patches are **no-ops for fp32/fp64**, so they do not affect the
bit-identical reference paths. The precision sweep and
``test_bf16_gated_kda.py`` verify bfloat16 stability up to T=2048 for KDA;
``test_needle_haystack.py::test_bf16_csa_hca_no_nan_inf`` pins the CSA/HCA
bf16 path.

## GatedAttentionMixin — unified gating for KDA/CSA/HCA

`ops_gated_attention.py` provides a reusable `GatedAttentionMixin` that
adds optional per-head or per-channel gating to any attention operator:

```python
from ops_gated_attention import GatedAttentionMixin

class MyAttention(GatedAttentionMixin, nn.Module):
    def __init__(self, ..., gate_type='per_channel', gate_dim=64, num_heads=4):
        super().__init__()
        self._init_gate(gate_type, gate_dim, num_heads)

    def forward(self, q, k, v, ...):
        k = self.apply_gate(k, dim=-1)  # gate keys
        v = self.apply_gate(v, dim=-1)  # gate values
        ...
```

Gate types:
- `'none'` — identity (no parameters)
- `'per_head'` — one scalar gate per head `[H]`, broadcast over channels
- `'per_channel'` — full `[H, D]` gate matrix per head

The mixin initializes gates near 1.0 (`nn.Parameter(torch.ones(...))`) so
the initial forward is an identity mapping. This makes it safe to add
gating to existing operators without changing their initialization
semantics.

`GatedDeltaRule` is a standalone fused layer that combines per-channel
gating with the delta-rule recurrence, usable as a drop-in gated memory
module or as a building block for gated CSA/HCA variants.

## Entry points

```bash
python3 run_quant_recall.py                 # writes results/exp10_quant_recall.json
python3 test_quant_recall.py                # or: pytest -q test_quant_recall.py
QUANT_BITS=2,3,4 QUANT_SEEDS=0,1 python3 run_quant_recall.py

python3 run_quant_lm.py                     # writes results/exp11_quant_lm.json
python3 test_quant_lm.py                    # or: pytest -q test_quant_lm.py
QLM_STEPS=100 QLM_QAT_STEPS=100 QLM_SEEDS=0 python3 run_quant_lm.py
QLM_SKIP_QAT=1 python3 run_quant_lm.py      # Part A only (fast)

python3 run_needle_haystack.py                 # writes results/exp12_needle_haystack.json
python3 test_needle_haystack.py                # or: pytest -q test_needle_haystack.py
python3 run_needle_haystack.py --quick --no-qat # CI smoke run
```
