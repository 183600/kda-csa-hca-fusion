# Equivalence testing (TOST) in the multi-seed comparisons

*Why "no significant difference" is not "the same", and how this repository
tests the difference between those two claims.*

## The problem

Every multi-seed comparison in this repository (Exp 4 quality probes,
Exp 5 ablation, Exp 8 mechanism ablation) can end in three states:

1. a **significant** seed-paired difference (operator A is better);
2. a **non-significant** difference at the current seed budget;
3. **practically the same** behaviour.

State 2 is routinely over-read as state 3. It is not: a non-significant
superiority test means the data are also compatible with a real difference
that the 5-7-seed budget is simply too small to detect (absence of evidence
is not evidence of absence). Only an **equivalence test** — a test whose null
hypothesis is "the difference is at least as large as some margin" — can
positively support state 3.

## The method

`multi_seed_stats.tost_paired` implements the two one-sided tests (TOST)
procedure for seed-paired differences `d_i = A_i - B_i`:

* lower test: `H0: mean(d) <= -margin` vs `mean(d) > -margin`;
* upper test: `H0: mean(d) >= +margin` vs `mean(d) < +margin`;
* `p_value = max(p_lower, p_upper)`, both from Student's t with `n - 1`
  degrees of freedom;
* `equivalent = p_value < alpha`, which is mathematically identical to the
  two-sided `(1 - 2*alpha)` confidence interval of the mean difference lying
  **strictly inside** `(-margin, +margin)`. Both views are returned in the
  same record.

Degenerate inputs follow the repository-wide convention: fewer than two
pairs gives `None` (not a fabricated verdict), and a zero-variance
difference vector yields the point interval with an exact `0 / 0.5 / 1`
p-value at inside / boundary / outside the margin.

## The margin must be pre-specified

The equivalence margin is the **smallest effect size of interest** (SESOI),
a scientific judgement, not a statistic. It is fixed in
`run_quality.PAIRWISE_EQUIVALENCE_MARGIN` and
`run_ablation.CONTRAST_EQUIVALENCE_MARGIN` (both default `0.05` absolute
accuracy: the MQAR probes start at chance `1/16 = 0.0625`, so sub-5-point
differences are not practically meaningful at this scale) and can only be
overridden before a run via the `MQAR_EQUIV_MARGIN` / `ABL_EQUIV_MARGIN`
environment variables (`0` disables equivalence testing). Choosing the
margin after looking at the confidence interval would invalidate the
p-value, so the runners record the margin and its provenance in the results
metadata (`equivalence_testing` / the `contrast_tests.tost` block).

Each pair additionally reports `equivalence_margin_supported`: the smallest
margin at which the *observed* data would have declared equivalence
(`|mean| + t_{1-alpha} * se`). This is a planning quantity — it answers
"what could this seed budget have shown?" — and must not be fed back into
the confirmatory TOST.

## Reading the results

In `results/exp4_mqar.json` (and the other quality-result files) each pair
record carries both questions separately:

| field | question | allowed claim when true |
|---|---|---|
| `significant_holm_permutation` | is the difference real? | "A is better than B" (`verdict`) |
| `significant_equivalence_holm` | is the difference smaller than the margin? | "A and B are practically the same" (`equivalence_verdict = equivalent`) |
| neither | — | "inconclusive at this seed budget"; use `seeds_required_80pct_power` for the next run |

The two can even hold at once for a tiny but perfectly consistent
difference: statistically detectable, practically negligible. The
per-cell family for the TOST is the same set of operator pairs as for the
superiority tests, and both families are Holm-Bonferroni corrected
separately.

In `results/exp5b_ablation_matrix_contrasts.json` and the ratio-sweep
provenance every contrast row carries the same split: `significant_
bonferroni` (superiority, Bonferroni family over all contrast rows) and
`tost_equivalent` + `equivalence_verdict` (equivalence, TOST over the same
family at the same corrected alpha, computed from the stored
`sd_diff`). A ratio row whose TOST rejects non-equivalence gets
`vs_reference_ratio.verdict = 'equivalent_to_reference'` — a claim the
plain non-significant superiority test can never make.

## Limitations

* TOST inherits the paired t-test's normality assumption on the
  differences. At `n = 5-7` the exact sign-flip permutation test remains
  the primary superiority statistic; treat a marginal TOST near the
  boundary with the same caution.
* Equivalence is always relative to the chosen margin. A "practically the
  same" claim is only as strong as the justification for that margin.
* The margin is in absolute accuracy units. For metrics with different
  scales (loss, PPL) pass a margin on that scale instead; `ppl_bootstrap`
  comparisons use their own paired-bootstrap machinery.

## Code map

* `multi_seed_stats.tost_paired`, `tost_from_summary` (the summary-
  statistics core both entry points share), `equivalence_margin_supported`
  — implementation (stdlib only, deterministic, JSON-safe).
* `run_quality._pairwise_op_comparisons` — per-cell integration and Holm
  correction; `pairwise_equivalent_to` row annotations.
* `run_ablation._annotate_contrast_significance` — matrix/ratio contrast
  family: `sd_diff` stored per row, TOST verdicts and the
  `equivalent_to_reference` ratio verdict.
* `test_multi_seed_stats.py`, `test_results_contracts.py` — hand-computed
  and numerical-integration cross-checks of the TOST p-values,
  degenerate-case contracts, and the runner integrations.

An equivalence claim is still a claim about a *seed set*. See
[seed-influence.md](seed-influence.md) for the leave-one-seed-out check that
reports whether the direction (or the TOST verdict) reverses when one seed
is deleted; it is descriptive and never upgrades either claim.
