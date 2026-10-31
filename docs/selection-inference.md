# Split-half selection validation: the winner's curse in a data-selected ranking

*Why "the best operator has the highest mean accuracy" is not a replicated
result, and how the repository re-selects the winner on held-out seeds.*

## The problem

`run_quality` trains every operator on the same seed block, then labels the
operator with the largest full-data mean as `best_op`. That label is
**data-selected**. With 5-7 seeds, the maximum of several noisy means is
biased upward: the selected operator tends to look better than its population
mean, and the winner can change if one seed lands differently.

The existing pairwise machinery is not the problem. Each pre-registered
contrast is tested with a seed-paired permutation test and Holm correction,
and the CI-overlap list is explicitly labelled a heuristic. But those tests
answer "is this fixed pair different?" They do not answer "would this
data-driven winner have been selected again, and does it still win on unseen
seeds?" Reporting the selected mean with an ordinary CI can therefore still
read as an unbiased performance estimate.

## The method

`multi_seed_stats.split_half_selection_validation(cells)` takes a mapping
`candidate -> seed -> metric` and forms balanced seed-half partitions:

1. intersect the seed sets (complete-case, as everywhere else) and list the
   dropped seeds;
2. for every half, select the candidate with the highest mean on one half
   and evaluate it on the complement; repeat in the reverse direction;
3. aggregate the selection events into:

| Field | Meaning |
|---|---|
| `full_ranking` | full-data means with deterministic rank (ties break by name) |
| `full_data_best` / `full_data_best_mean` | the label being audited |
| `full_winner_selection_frequency` | fraction of half-selections that choose the full-data winner |
| `full_winner_holdout_top1_rate` | fraction of held-out halves where the full-data winner is actually top-1 |
| `top1_agreement_rate` | fraction of events where the held-out top-1 equals the selected candidate |
| `mean_selection_optimism` | mean(selection-half mean − holdout mean) for the selected candidate |
| `cross_fitted_selected_mean` | mean held-out value of the candidate the procedure selects |
| `selection_frequency` / `holdout_best_frequency` | per-candidate rates, so a near-tie is visible |
| `full_winner_mean_holdout_rank` | mean rank of the full-data winner on held-out halves |

`mean_selection_optimism` is the direct winner's-curse estimate: selecting on
noise makes the selection-half mean exceed the held-out mean. A negative value
means the selected candidate happened to do even better out of sample; it is
not evidence against selection bias.

The partition search is exhaustive when the number of balanced halves is at
most `max_partitions` (default 4096). Above that it uses a **deterministic
local random sample** of balanced halves, never the global RNG, and records
`partition_sampling` plus `partitions_truncated`, so a capped run is visible
rather than silently presented as exhaustive.

## Reading the status

| `status` | Meaning |
|---|---|
| `stable` | the full-data winner is selected on at least `stability_threshold` (0.8) of the half-selections |
| `fragile` | it is selected on at least `fragile_threshold` (0.5) but below the stability threshold |
| `unstable` | it is selected on fewer than half of the half-selections; the data-driven ranking does not reproduce |
| `insufficient_data` | fewer than two usable candidates or fewer than `min_seeds` (4) common seeds |

`stable` means the **ranking** is robust to halving the seed block. It does
not mean the winner is significantly better than every other candidate; that
question still belongs to the pre-registered pairwise family. A `fragile` or
`unstable` label qualifies the ranking claim, not the p-values of the
individual contrasts.

## What this is not

* It is not a hypothesis test and reports no p-value.
* It reuses the same seeds; the halves are not independent replications and
  the rates are descriptive.
* It is not multiplicity-corrected and does not change any pairwise p-value.
* It cannot upgrade a non-significant contrast or an underpowered run, and it
  does not license deleting the seeds that produce an inconvenient result.
* It only audits the **selection label**. A pre-registered contrast remains
  the confirmatory evidence regardless of the winner's status.

## Integration

`quality/_shared._pairwise_op_comparisons` attaches the report to every
`(task, n_kv)` cell as `selection_validation`, and the exp4 report aggregates
`n_selection_stable` / `n_selection_fragile` / `n_selection_unstable` /
`n_selection_insufficient` across cells.

`run_quality` copies the cell report into `cross_op_checks[cell]`, and its
human-readable `claim` adds a sentence when the full-data winner is not
stable, e.g.:

> Split-half seed validation selects the full-data winner on 50% of held-out
> seed halves (status=fragile); the ranking is not a replicated ordering.

## Code map

* `multi_seed_stats.split_half_selection_validation` — implementation
  (stdlib only, deterministic, strict-JSON-safe).
* `quality/stats.py::_pairwise_op_comparisons` — per-cell integration and
  aggregate counts.
* `quality/main.py` — copies the block into `cross_op_checks` and qualifies
  the best-op claim text.
* `test_multi_seed_stats.py` — hand-computed stable/fragile cases, seed
  intersection, capped deterministic sampling, validation, and JSON safety.

See [seed-influence.md](seed-influence.md) for leave-one-seed-out sensitivity
and [cross-cell-replication.md](cross-cell-replication.md) for replication
across `(task, n_kv)` cells.
