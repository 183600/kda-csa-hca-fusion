# Single-seed influence: leave-one-seed-out sensitivity

*Why a significant multi-seed result can still hinge on one training seed,
and how the repository reports that without turning it into another test.*

## The problem

The multi-seed experiments in this repository train 5-7 seeds per operator
and decide orderings with a seed-paired sign-flip permutation test. That is
the right primary statistic for such a small sample, but it answers only
"is the mean paired difference large relative to the sign-flip
distribution?". It does not answer a reader's next question:

> If one seed had landed differently, would the reported direction or the
> significant/not-significant verdict still hold?

The three-p-value panel in `summarize_paired` (t, exact sign, permutation)
can show that a result is *fragile* when the tests disagree, but it does not
say **which seed** is responsible, how far the estimate moves when that seed
is removed, or whether the direction reverses. At `n = 5-7`, a single
diverged optimization run can do all three.

## The method

`multi_seed_stats.leave_one_out_sensitivity` deletes each seed in turn and
recomputes the same paired panel on the remaining `n - 1` pairs:

* mean / sample SD / direction of the remaining differences;
* the paired t-test, exact sign test, and sign-flip permutation test;
* the shift of the mean relative to the full-sample mean, and whether the
  sign flipped;
* whether the primary (permutation) reject/not-reject decision flipped;
* when a PRE-SPECIFIED `equivalence_margin` is supplied, the TOST verdict
  and whether it flipped.

The summary reports the range of the remaining means, the largest absolute
shift, the most influential seed (position and value), the counts of sign /
decision / TOST flips, and a status.

| `status` | Meaning |
|---|---|
| `stable` | Deleting any single seed leaves both the direction and the primary decision unchanged. |
| `direction_fragile` | At least one deletion reverses (or erases) the sign of the mean difference. |
| `boundary_sensitive` | The direction survives, but the primary permutation decision flips at `alpha` for at least one deletion. |
| `no_direction` | The full-sample mean difference is exactly zero; there is no direction to assess. |
| `insufficient_data` | Fewer than 3 pairs; no `n - 1` subset is testable. |

`fragile` is `True` for `direction_fragile` and `boundary_sensitive`;
`direction_fragile` and `decision_sensitive` are also exposed separately.

### The `n = 6` exact-test caveat

A unanimous 6-pair exact permutation result has two-sided
`p = 2 / 2^6 = 0.03125`. Deleting one seed leaves 5 pairs, whose smallest
attainable two-sided p is `2 / 2^5 = 0.0625`. Such a result is therefore
*always* `boundary_sensitive`. That is the resolution of an exact test at
this sample size, not evidence against the effect. Read
`n_primary_significance_flips` next to `n_pairs`; a stable 7-pair unanimous
sweep, by contrast, has `p = 0.015625` and remains significant after one
deletion (`p = 0.03125`).

## What this is not

Leave-one-seed-out is a **descriptive sensitivity analysis, not a
hypothesis test**:

* it re-uses the same data, so its p-values are not independent and are not
  corrected for multiplicity;
* it is not a license to delete the seed that "fixes" the result — the
  seed set is part of the pre-registered design;
* a `stable` report cannot upgrade a non-significant or underpowered claim;
  it only says the observed decision does not depend on one seed;
* a `direction_fragile` report does not by itself make the full-sample
  result invalid; it says the ordering should not be presented as
  structural without more seeds or an investigation of the influential run.

The confirmatory inference remains `summarize_paired` on the
pre-registered seed set, with the Holm-corrected pairwise family.

## Where it is used

* `run_quality._pairwise_op_comparisons` enables
  `include_seed_influence=True` for every testable operator pair. Each pair
  record carries the full `seed_influence` block. A significant `verdict`
  whose direction is fragile also records the opponent under the winning
  row's `pairwise_direction_fragile_wins`.
* The exp4 `op_pairwise_comparisons` report aggregates
  `n_direction_fragile`, `n_boundary_sensitive`, and
  `n_seed_influence_stable` per cell and overall.
* The best-op claim (`cross_op_checks`) splits its significant wins into
  `best_op_single_seed_robust_wins` and
  `best_op_direction_fragile_wins`, and the claim text states when a cited
  ordering reverses under a single deletion.
* `summarize_paired(..., include_seed_influence=True)` attaches the block to
  any caller that wants it. It is opt-in because it re-runs the panel `n`
  times; bulk per-cell summaries leave it off.

Example:

```python
from multi_seed_stats import leave_one_out_sensitivity

diffs = [0.10, 0.12, 0.11, 0.09, 0.10, -1.90]   # one diverged seed
rep = leave_one_out_sensitivity(diffs)
rep['status']                 # 'direction_fragile'
rep['most_influential_index'] # 5
rep['drops'][5]['mean_diff']  # the remaining mean after deleting seed 5
```

## Limitations

* It is a one-at-a-time deletion view, not an exhaustive subset or
  influence-function analysis. Two seeds can jointly drive a result while
  no single deletion flips it.
* A `boundary_sensitive` label is unavoidable when the exact test has no
  rejection region to begin with (e.g. 5 seeds x 21 Holm-corrected
  comparisons). Read `n_primary_significance_flips` together with
  `test_resolution` from `summarize_paired`; the latter reports whether
  any p-value in the family could have crossed alpha. See
  [inference-resolution.md](inference-resolution.md).
* The permutation Monte-Carlo branch is reused with the same local seed, so
  the report is deterministic; its resolution is still
  `1 / (n_permutations + 1)`.
* It is only as meaningful as the pairing. Seeds that were not actually
  paired (different data order, different stopping point) must be excluded
  before the differences are formed, exactly as for `summarize_paired`.
* The status thresholds are fixed at the same `alpha` as the primary test.
  A result that is significant at a corrected family alpha but not at the
  raw alpha will be labelled `boundary_sensitive` more often; the
  `n_primary_significance_flips` count makes that visible. The per-pair
  view uses the raw alpha and does not re-run the Holm family for each
  deletion (deleting a seed changes the family too). A **direction** flip
  is alpha-independent, so `direction_fragile` remains the relevant
  robustness signal for a Holm-corrected superiority claim.
* With an equivalence margin, the TOST stability flags are computed at the
  raw per-pair alpha; the Holm-corrected family is not re-run for each
  deletion. They describe the raw TOST verdict, not the reported
  family-corrected equivalence verdict, and `status` itself concerns only
  the direction and the superiority decision.

## Code map

* `multi_seed_stats.leave_one_out_sensitivity` — implementation (stdlib
  only, deterministic, JSON-safe).
* `multi_seed_stats.summarize_paired` — `include_seed_influence` opt-in.
* `run_quality._pairwise_op_comparisons` — per-cell integration, row
  annotations, and aggregate counts.
* `run_quality._print_pairwise_report` — the `infl` column in the printed
  pairwise table.
* `test_multi_seed_stats.py` — hand-computed sign/decision flips,
  degenerate cases, determinism/JSON safety, and the `run_quality`
  integration.

For the complementary question — whether an ordering replicates across
all `(task, n_kv)` cells rather than within one cell — see
[cross-cell-replication.md](cross-cell-replication.md). For whether a
**data-selected winner** survives on held-out seeds (the winner's curse),
see [selection-inference.md](selection-inference.md).
