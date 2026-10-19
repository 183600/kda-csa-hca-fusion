# The CSA lightning-indexer top-k selector: two kernels, one rule

*Why the per-token decode selection no longer pays `O(n log n)`, and how
NaN scores went from "crash or silent wrong index" to "unselectable".*

## The problem

CSA's lightning indexer (DeepSeek-V4 §2.3.1, Eq. 13–17) scores every
completed compressed block per query and keeps the top-`k`.
`ops_csa._canonical_topk_indices` makes that selection a pure function of
the score vector — ties break on ascending block index — so batched
prefill over `n_blocks_full` scores (future blocks masked to `-inf`) and
per-token decode over the growing `n_blocks_inc` prefix select
index-for-index the same blocks. `torch.topk` cannot be used directly:
its tie choice depends on the tensor SIZE (and backend).

Historically the selector ran a full **stable sort** for the
single-query decode shape (the sort's output *is* the canonical rule) and
an `O(n)` boundary pass only for multi-row sparse scores. At long context
that made every decode step pay `O(n_blocks log n_blocks)` inside the
indexer:

```
selector cost, 1 query row, topk=16       n=4096     n=16384    n=65536
stable sort (historical decode kernel)    0.16 ms     0.90 ms    4.23 ms
O(n) boundary kernel                      0.05 ms     0.18 ms    0.55 ms
```

— in exactly the regime (`T` in the tens of thousands, `m = 16`) CSA
exists for. A second, sharper problem: **NaN scores**. A NaN loses both
the `>` and the `==` comparison, so

* the multi-row boundary path under-filled a row and crashed on the
  `nonzero(...).view(rows, s)` reshape
  (`RuntimeError: shape '[2, 1]' is invalid for input of size ...`), and
* the single-row sort path ranked NaN *first* and silently returned a NaN
  block index (`[[nan, nan, 1.0]]` with `k=1` gave `[[0]]`, the NaN slot,
  instead of the finite block `2`).

## The two kernels

| | stable-sort kernel | boundary kernel (`_boundary_selected_mask`) |
|---|---|---|
| cost | `O(n log n)` | `O(n)`: one `kthvalue` + comparison masks |
| used for | dense `2*s >= n`, or small arrays | sparse selections on larger arrays |
| tile size | one sort | chunked boolean/int32 masks (`_TOPK_SELECT_CHUNK_ELEMS`) |
| NaN handling | re-sort a NaN→`-inf` copy iff a NaN entered the top-`s` slots | under-fill detection (`selected.sum(-1) < s`) → re-run on a NaN-free copy |

Both kernels implement the SAME canonical rule. The kernel choice is a
pure **cost heuristic** (`2*s >= n or rows*n <= _TOPK_SORT_MAX_ELEMS`,
default 2048 elements), documented at the constant: below the element
budget the sort's lower constant wins by up to ~2x; above it the boundary
kernel wins by 3–8x. Because both kernels are covered by the same parity
tests, the heuristic cannot change any result — only the cost.

## The NaN policy

NaNs are **unselectable**: a NaN score never occupies a returned slot.
A NaN competes as `-inf` for the boundary/whole-row rules, and a slot that
ends up selected at `-inf` keeps the historical "invalid slot" contract
of `-1`. Downstream consumers already treat `-1` as "no block"
(`valid_mask = indices >= 0`, `clamp(min=0)`), so a NaN-poisoned indexer
signal degrades to "selects nothing" (sink / sliding-window only) instead
of crashing prefill or desynchronizing decode from prefill.

This matches the module's existing philosophy: `_nan_safe_softmax` exists
precisely so non-finite logits do not take down the forward pass.

## Measured cost (CPU, Exp 14: `run_indexer_selector.py`)

Decode geometry `m=16, c=64, c_I=32, nh=4, nIh=2, win=64, topk=16`,
median wall-clock:

| T | n_blocks | dispatch | indexer (historical sort → shipped) | full decode step |
|---|---|---|---|---|
| 16 384 | 1 024 | sort (unchanged) | 0.047 → 0.050 ms | 0.19 → 0.20 ms |
| 65 536 | 4 096 | boundary | 0.19 → 0.12 ms (**1.6x**) | 0.36 → 0.31 ms (**1.2x**) |

The `T = 16384` row is deliberately included: below the element budget
the shipped selector *is* the sort kernel plus the NaN-safety check
(~2–3 µs per call, <1.5% of a decode step at that scale). The larger,
asymptotic win is the point; the small-array cost is the price of not
crashing on NaN.

## Verification

* `test_indexer_selector.py` — grid that provably exercises BOTH kernels
  against an independent reference (stable sort over `NaN -> -inf`),
  the exact historical hazard shapes, valid-slot-count invariants, and
  cache-level decode == `naive_csa` with sink + sparse tied top-k, the
  sink-only degenerate config, and NaN-poisoned indexer weights.
* `test_csa_tiebreak.py` — the pre-existing canonical tie-break contract
  (unchanged, must keep passing).
* `run_correctness.py` (gate) — full prefill/decode equivalence suite.
* `run_indexer_selector.py` — Exp 14: parity gate runs BEFORE any timing
  row is archived; the `exp14_indexer_selector` results contract refuses
  any file with a non-zero mismatch/crash count.

## Limitations

* The dispatch threshold is a CPU measurement (`_TOPK_SORT_MAX_ELEMS =
  2048` elements). On accelerators the crossover may differ; the constant
  is documented for revisiting per device, and a wrong choice costs
  performance, never correctness (parity tests pin both kernels).
* The NaN policy makes NaN scores behave as `-inf`; it does not diagnose
  *why* NaNs appeared upstream (that is the caller's job — the selector
  only guarantees NaN cannot become a real block index or a crash).
* Selector timings are CPU wall-clock medians under the repo's standard
  timing protocol; absolute microseconds are machine- and thread-count-
  dependent, the recorded ratios are the portable claim.
