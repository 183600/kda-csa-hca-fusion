# KV read-budget model (Exp 9)

`kv_budget.py` prices every fused-attention operator in ONE currency: the
number of **KV-side elements read per query per layer** (one element = one
scalar of the KV-side representation; query-side projections are excluded).
It backs Experiment 9 (`run_budget_frontier.py`) and is written to be
adjusted independently of the operator implementations, then cross-checked
against their causal conventions.

## Two views, one model

The model has a prefill view and a decode view that are **the same
function aggregated differently**:

| View | API | Meaning |
|---|---|---|
| Prefill average | `op_read_elements(op, T, spec)` / `layout_read_elements(T, ratio, spec)` | mean reads per query over positions `0..T-1` |
| Decode marginal | `instant_read_elements(op, t, spec)` / `instant_layout_read_elements(ratio, t, spec)` | reads of the single query at position `t`; `t = T` is the first token decoded after a `T`-token prefill |
| Decode window | `windowed_read_elements(op, t0, t1, spec)` / `windowed_layout_read_elements(ratio, t0, t1, spec)` | mean marginal over `t in [t0, t1)` (O(1) closed form) |

Identities pinned by `test_budget_frontier.py`:

```
mean_{t < T} instant(op, t)                  == op_read_elements(op, T)
windowed(op, 0, T)                           == op_read_elements(op, T)
windowed(op, t0, t1)                         == mean_{t in [t0,t1)} instant(op, t)
mean_{t < T} instant_layout(ratio, t)        == layout_read_elements(T, ratio)
```

Because the marginal view is exactly what a token-by-token decoder charges
at each step, a measured per-token decode latency should be compared against
the **decode-window** budget of the positions actually decoded
(`[prefill_len, prefill_len + n)`), not against the prefill average — the
latter mixes in cheap early positions and understates late-context cost.

## Causal conventions

* Compressed CSA/HCA blocks are **strictly preceding** (DeepSeek-V4
  Eq. 16: block `s` is visible to query `t` iff `s < t // m`), matching
  `ops_csa._causal_block_mask` and `run_kv_cache.causal_block_entries`.
* The sliding window and dense softmax include the query's own token.
* KDA reads its `hv x d_k x d_v` delta-rule state twice per step (masked
  delta read + output read) and is therefore O(1) in `T`.
* Mamba-2 reads its `nheads x headdim x d_state` SSD state twice per step
  (decayed update + `C @ S^T` readout), so it is the same O(1) currency
  with a different state geometry. `op_read_elements('mamba2', ...)` and
  `instant_read_elements('mamba2', ...)` price that route; the short-conv
  lookback is excluded, matching KDA's convention.
* CSA charges the lightning-indexer scan over **every visible block**
  (the reference implementation scores before top-k selection, including
  `topk=0`); the top-k gather itself is capped at `min(topk, visible)`.
* The low-level pair counters (`causal_block_entries`,
  `causal_selected_entries`) are imported from `run_kv_cache`, so the read
  budget and the Exp 3 KV/FLOPs accounting cannot disagree on causality.

## Exp 9 sections

| `section` | Rows | Content |
|---|---|---|
| `budget_curve` | `row_kind` `op`/`layout` | prefill-average curves over `BUDGET_T_GRID` for softmax/KDA/Mamba-2/CSA/HCA and the layouts |
| `decode_budget_curve` | `row_kind` `op`/`layout` | marginal (one-query) curves at position `t` |
| `decode_sweep` | measured | per-token decode latency joined with both views (`total_read_elements_per_query` = prefill average, `decode_step_*` / `decode_window_*` = decode view) |
| `quality_join` | measured | Exp 4/5/8 accuracy rows annotated with their analytic budget |
| `ranking_summary` | derived | layout orders and Spearman correlations against accuracy under both views |

## Quantization / bytes

The element counts above assume fp16. `kv_quant.py` adds a byte view:
`op_read_bytes` / `layout_read_bytes` price the same rows under a
component→`QuantSpec` policy (payload bits + per-group scale/zero-point
metadata), and `run_quant_recall.py` (Exp 10) measures the recall side of
the trade-off on planted associations for both recurrent routes. See
[quantization-model.md](quantization-model.md).

Layout rows are expanded from a ``memory:CSA:HCA`` ratio. The memory slot
is KDA by default; pass ``memory_op='mamba2'`` to
``layout_read_elements`` / ``instant_layout_read_elements`` /
``windowed_layout_read_elements`` to price the alternative route. The row
keeps ``n_memory`` and ``memory_op``, sets ``kda_read_elements`` or
``mamba2_read_elements`` accordingly, and leaves the default KDA totals
bit-identical. `run_budget_frontier` still emits its layout curves on the
KDA route; the Mamba-2 route is available per-op (Exp 4 join) and through
this explicit API rather than being silently assumed away.

## Scope / honesty

This is an **analytical read count, not profiled traffic**: library-internal
materialization (`unfold`, temporaries) and Python-loop overhead are out of
scope (same caveat as Exp 7). The decode marginal describes the *shape* of
the decode cost, not a latency prediction. Hybrid layouts are
depth-controlled, not budget-controlled, so the ranking summaries are
exploratory; a positive correlation between accuracy and budget confounds
operator composition with budget and does not establish a causal effect.
The byte view inherits the same analytical caveat: it prices the *model*
under a quantization policy, not a profiled quantized kernel.
