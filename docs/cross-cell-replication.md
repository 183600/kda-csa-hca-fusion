# Cross-cell replication: one paired ordering across every cell

*Why a significant per-cell ordering is not automatically a replicated
result, and how the repository tests the ordering across all pre-specified
`(task, n_kv)` cells without treating cells as independent studies.*

## The problem

`run_quality._pairwise_op_comparisons` tests every unordered operator pair
inside one `(task, n_kv)` cell and applies Holm correction to that cell's
family. That is the right unit for a **cell-specific** claim: "on MQAR with
`n_kv = 1`, operator A beats operator B". It is not evidence that the
ordering replicates.

Because each cell has its own family, a paper can report the cell where a
pair happened to cross alpha and omit the cells where it did not. The
per-cell record cannot detect that selection: it never looks across cells.
A pooled mean over all rows is not a fix either, because cells can have
different seed counts and different within-seed variability; the
observation-weighted mean can even reverse the seed-balanced sign
(Simpson-style aggregation).

## The method

`multi_seed_stats.cross_cell_paired_analysis` takes a mapping

```python
cells = {
    'mqar/n_kv=1': {0: 0.10, 1: 0.12, 2: 0.09, 3: 0.11},
    'kv/n_kv=2':   {0: 0.08, 1: 0.11, 2: 0.10, 3: 0.12},
}
```

where each inner mapping is `seed -> paired difference` oriented so that a
positive value means operator A is better. It then:

1. intersects the seed sets and keeps only seeds present in **every** cell;
2. for each common seed, averages that seed's paired difference across
   cells;
3. runs the ordinary `summarize_paired` panel (t, exact sign, sign-flip
   permutation) on those seed-level averages.

The seed, not the cell, is the experimental unit. A cell with more
surviving seeds therefore cannot dominate the estimate, and the cells are
not treated as independent studies (which would be a meta-analysis of
separate experiments). The estimand is the average paired difference
across the included cells; a cell-specific effect must still be read from
the per-cell panel.

Seeds excluded by the balance requirement are reported explicitly in
`dropped_seeds`, with `n_seeds_dropped` and `n_observations`. The report is
stdlib-only, deterministic, and JSON-safe.

## Reading the status

| `status` | Meaning |
|---|---|
| `insufficient_cells` | The contrast appears in fewer than `min_cells` (default 2) cells. This is **not** a replication. |
| `insufficient_balanced_seeds` | The cells exist but fewer than two seeds are common to all of them; no paired test is possible. |
| `undefined` | The balanced estimate could not be formed (for example, an empty common seed set). |
| `aggregation_reversal` | The naive observation-weighted mean has the opposite sign from the seed-balanced mean. |
| `mixed_direction` | The per-cell mean differences do not all share a sign, so the pooled average is not a replicated ordering. |
| `consistent_direction` | All cells point the same way (or the balanced estimate is defined and no stronger warning applies). |

`all_cells_same_direction` is true only when every cell's mean is strictly
positive or every cell's mean is strictly negative. `mixed_direction` is
true when at least one cell is positive and at least one is negative.
`cell_direction_counts` exposes the positive / negative / zero counts and
how many cells agree with the seed-balanced sign.

The status is a **guard against over-reading**, not a replacement for the
paired p-value. A `consistent_direction` result can still be
non-significant at the current seed budget, and an `aggregation_reversal`
result can still have a significant balanced test — the latter means the
balanced test and the raw pooled mean disagree about the sign, which is
exactly the situation in which a cell-weighted claim is unsafe.

## Heterogeneity is descriptive

`heterogeneity` is a balanced repeated-measures cell x seed F test on the
common-seed matrix:

* `F`, `df1 = k - 1`, `df2 = (n - 1)(k - 1)`;
* `p_value` from the F upper tail;
* `between_cell_sd` in the original units;
* `within_seed_sd`, the residual cell-to-cell variation after removing
  seed and cell effects;
* `zero_residual_variance`, true when the residual is only floating-point
  round-off (a constant offset across seeds);
* `sphericity_assumed`, always true in this implementation.

The F test assumes sphericity (exact for two cells, potentially
anti-conservative for more) and is **not** the replication test. The
seed-level paired panel in `primary` is the inferential result. A large F
means the cell means differ by more than the within-seed cell-to-cell
variation; it does not by itself say which cell is better.

## Multiplicity

`cross_cell_paired_analysis` does not correct for multiplicity across
contrasts. In `run_quality._pairwise_op_comparisons` the integration forms
one Holm family over every operator pair that is testable (at least two
common seeds) in at least two cells. Pairs testable in only one cell are
labelled `insufficient_cells`; pairs that enter two or more testable cells
but share fewer than two seeds across them are labelled
`insufficient_balanced_seeds`. Neither enters the family, so neither
inflates the correction. Pairs that are not testable in any cell remain in
the per-cell `insufficient_paired_seeds` records and are absent from the
cross-cell block. Each cross-cell record receives:

* `holm_adjusted_p_permutation`;
* `significant_holm_permutation`;
* `verdict` (`op_a_better`, `op_b_better`, `not_significant`,
  `insufficient_cells`, or `insufficient_balanced_seeds`).

The aggregate block also reports `n_tests`, `n_significant`,
`n_consistent_direction`, `n_mixed_direction`,
`n_aggregation_reversal`, `n_insufficient_cells`,
`n_insufficient_balanced_seeds`, and
`n_significant_and_consistent` (significant **and**
`consistent_direction`). The last count is the one to cite for a replicated
ordering; a significant mixed-direction or reversal result is not a
replication.

## What this is not

* It is not a meta-analysis of independent experiments. The cells share
  seeds and training design; treating them as independent would
  double-count and overstate the evidence.
* It is not a license to drop a cell. The cell set is part of the
  pre-registered design; `dropped_seeds` exists only because the balanced
  estimator requires complete seed blocks, not because inconvenient cells
  may be removed.
* It does not repair a cell-specific claim. If only one cell is
  significant, the cross-cell record will usually be non-significant or
  mixed-direction; the correct conclusion is that the ordering is not
  replicated, not that the significant cell should be promoted.
* It does not replace the per-cell Holm family for a claim explicitly
  scoped to one cell. The two questions are complementary: "does this
  ordering hold here?" and "does it hold across the cells we ran?".

## Code map

* `multi_seed_stats.cross_cell_paired_analysis` — implementation
  (stdlib only, deterministic, JSON-safe).
* `multi_seed_stats._cross_cell_heterogeneity` — descriptive balanced
  repeated-measures F check.
* `quality/stats.py::_cross_cell_pairwise_report` — cross-cell Holm
  family, verdicts, and aggregate counts.
* `quality/stats.py::_pairwise_op_comparisons` — collects the per-cell
  seed-paired differences and attaches `cross_cell_replication` to the
  exp4 `op_pairwise_comparisons` report.
* `test_multi_seed_stats.py` — hand-computed balancing, mixed-direction,
  aggregation-reversal, heterogeneity, validation, and runner integration
  tests.

See [seed-influence.md](seed-influence.md) for the single-seed
leave-one-out check and [inference-resolution.md](inference-resolution.md)
for the exact-test resolution floor.
