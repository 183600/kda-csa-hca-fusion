# The CSA lightning-indexer top-k selector: two kernels, one rule

*How the per-token decode selection stays a pure function of the score
vector, and how NaN scores stay unselectable.*

## The problem

CSA's lightning indexer (DeepSeek-V4 §2.3.1, Eq. 13–17) scores every
completed compressed block per query and keeps the top-`k`.
`ops_csa._canonical_topk_indices` makes that selection a pure function of
the score vector — ties break on ascending block index — so batched
prefill over `n_blocks_full` scores (future blocks masked to `-inf`) and
per-token decode over the growing `n_blocks_inc` prefix select
index-for-index the same blocks. `torch.topk` cannot be used directly:
its tie choice depends on the tensor SIZE (and backend).

A second problem: **NaN scores**. A NaN loses both the `>` and the `==`
comparison, so a naive selector can under-fill a row or treat the NaN
slot as a real block index. The selector makes NaN scores unselectable
instead.

## The two kernels

| | stable-sort kernel | boundary kernel (`_boundary_selected_mask`) |
|---|---|---|
| cost | `O(n log n)` | `O(n)`: one `kthvalue` + comparison masks |
| used for | dense `2*s >= n`, or small arrays | sparse selections on larger arrays |
| tile size | one sort | chunked boolean/int32 masks (`_TOPK_SELECT_CHUNK_ELEMS`) |
| NaN handling | re-sort a NaN→`-inf` copy iff a NaN entered the top-`s` slots | under-fill detection (`selected.sum(-1) < s`) → re-run on a NaN-free copy |

Both kernels implement the SAME canonical rule. The kernel choice is a
pure **cost heuristic** (`2*s >= n or rows*n <= _TOPK_SORT_MAX_ELEMS`,
default 2048 elements), documented at the constant. Because both kernels
are covered by the same parity tests, the heuristic cannot change any
result — only which kernel runs.

## The NaN policy

NaNs are **unselectable**: a NaN score never occupies a returned slot.
A NaN competes as `-inf` for the boundary/whole-row rules, and a slot that
ends up selected at `-inf` keeps the "invalid slot" contract of `-1`.
Downstream consumers already treat `-1` as "no block"
(`valid_mask = indices >= 0`, `clamp(min=0)`), so a NaN-poisoned indexer
signal degrades to "selects nothing" (sink / sliding-window only) rather
than desynchronizing decode from prefill.

This matches the module's existing philosophy: `_nan_safe_softmax` exists
precisely so non-finite logits do not take down the forward pass.

## Verification

* `test_indexer_selector.py` — grid that provably exercises BOTH kernels
  against an independent reference (stable sort over `NaN -> -inf`),
  the NaN hazard shapes, valid-slot-count invariants, and cache-level
  decode == `naive_csa` with sink + sparse tied top-k, the sink-only
  degenerate config, and NaN-poisoned indexer weights.
* `test_csa_tiebreak.py` — the canonical tie-break contract.
* `run_correctness.py` (gate) — full prefill/decode equivalence suite.
* `run_indexer_selector.py` — Exp 14: parity gate runs BEFORE any timing
  row is archived; the `exp14_indexer_selector` results contract refuses
  any file with a non-zero mismatch/crash count.

## Limitations

* The dispatch threshold is an element-count heuristic
  (`_TOPK_SORT_MAX_ELEMS = 2048` elements). On accelerators the crossover
  may differ; the constant is documented for revisiting per device, and a
  wrong choice costs performance, never correctness (parity tests pin
  both kernels).
* The NaN policy makes NaN scores behave as `-inf`; it does not diagnose
  *why* NaNs appeared upstream (that is the caller's job — the selector
  only guarantees NaN cannot become a real block index or a crash).
