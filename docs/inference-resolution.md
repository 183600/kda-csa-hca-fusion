# Exact-test resolution: when a non-significant result is forced by the design

*Why an exact 5-seed sign-flip test with a 21-comparison family cannot
reject at `alpha = 0.05`, and how the results-analysis layer reports that
instead of calling it "no difference".*

## The problem

The repository's primary paired statistic is the exact sign-flip
permutation test (`multi_seed_stats.permutation_test`) and its
distribution-free companion, the exact sign test. Both enumerate the
`2**n_effective` sign assignments of the `n_effective` non-tied per-seed
differences. Their p-value therefore cannot be smaller than

```
p_floor = 2 / 2**n_effective      (two-sided)
p_floor = 1 / 2**n_effective      (one-sided)
```

At `n_effective = 5` the two-sided floor is `2/32 = 0.0625`. If the
analysis also Holm-corrects a family of `m` comparisons, the smallest
attainable adjusted p is

```
p_floor_holm = min(1, m * p_floor)
```

For the default quality sweep (`linear, softmax, kda, mamba2, rwkv7, csa,
hca`, so `m = C(7, 2) = 21`) at the default screening budget of 5 seeds,
`p_floor_holm = min(1, 21 * 0.0625) = 1.0`. **No pair can reject for any
data.** A table of 21 `not_significant` verdicts is not evidence that the
operators are equivalent; it is a property of the discrete test and the
seed budget. Ordinary power arithmetic (normal/t approximation) does not
see this floor because it uses a continuous test distribution.

The same arithmetic applies to a pre-registered plan: a plan can pass a
normal-approximation power target and still name an exact test with no
rejection region. That is why the planning layer checks both.

## The method

`multi_seed_stats.exact_test_resolution(n_pairs, alpha, family_size,
alternative, n_zero)` returns a JSON-safe record:

| Field | Meaning |
|---|---|
| `min_attainable_p_raw` | `p_floor` for one comparison |
| `min_attainable_p_holm` | `min(1, family_size * p_floor)`, the smallest adjusted p the family could produce |
| `raw_testable` | the comparison could reject on its own |
| `family_testable` | the comparison could be the **first** (smallest-p) rejection in the Holm family; with equal n across the family this is exactly whether any rejection is possible |
| `later_rank_possible` | the raw floor is <= alpha, so a later-rank rejection is not ruled out when the comparison cannot be first (necessary, not sufficient) |
| `min_effective_pairs_raw`, `min_effective_pairs_holm` | smallest effective n that would restore a rejection region |
| `shortfall_pairs_raw`, `shortfall_pairs_holm` | additional informative pairs the current run needs |
| `max_comparisons_at_alpha` | largest family the current n can support (`0` means not even one comparison) |
| `n_zero`, `n_effective` | tied pairs carry no sign information and are excluded |

Two nonparametric uncertainty companions are reported with it:

* `binomial_proportion_ci(k, n)` — the exact Clopper-Pearson interval
  for the seed win rate. A 5/5 sweep is `[0.478, 1]`, not a confident
  100%.
* `median_order_statistic_ci(diffs)` — the distribution-free interval
  `[x_(k), x_(n-k+1)]` for the paired median, with `k` the largest order
  statistic whose binomial tail is at most `alpha/2`. At `n = 5` no 95%
  interval exists (the full range has 93.75% coverage), so the function
  returns `None` and `min_pairs_for_ci = 6` rather than pretending.

These are **necessary-condition diagnostics**, not tests and not power
guarantees. A family can be testable and still badly underpowered at the
SESOI; conversely, a resolution-limited family is not evidence of a null
effect or of equivalence.

## Worked examples

The table assumes the usual complete seed block (the same `n` for every
comparison in the family), where `family_testable` is the family-level
feasibility condition. If common-seed counts differ, a pair with a larger
`p_floor` can still be rejected at a later Holm rank after more powerful
pairs are rejected; `later_rank_possible` records that weaker condition and
`run_quality` reports `n_later_rank_possible` separately.

| Design | `p_floor` (two-sided) | `p_floor_holm` | Can reject? | Effective pairs needed for Holm |
|---|---|---|---|---|
| 5 seeds, 1 comparison | 0.0625 | 0.0625 | no | 6 |
| 5 seeds, 7 ops (21 pairs) | 0.0625 | 1.0 | no | 10 |
| 6 seeds, 1 comparison | 0.03125 | 0.03125 | yes | 6 |
| 6 seeds, 7 ops (21 pairs) | 0.03125 | 0.65625 | no | 10 |
| 7 seeds, 7 ops (21 pairs) | 0.015625 | 0.328125 | no | 10 |
| 10 seeds, 7 ops (21 pairs) | 0.0009766 | 0.0205 | yes | 10 |
| 11 seeds, 5 vs-reference (planned Exp 4, **one-sided**) | 0.0004883 | 0.00244 | yes | 7 |

All rows are two-sided unless marked. The planned Exp 4 budget
(11 seeds, 5 pre-registered one-sided vs-reference contrasts) is testable; the default 5-seed screening sweep is not. That
is exactly the screening/confirmatory distinction the design audit
already draws, now made visible on the statistical side as well.

## Where it is used

* `multi_seed_stats.summarize_paired(..., family_size=m)` always attaches
  a `test_resolution` block (default `family_size=1` describes the
  comparison in isolation). The same record also carries
  `win_rate_ci95`, `ci95_median_diff`, and the nested `median_ci`.
* `run_quality._pairwise_op_comparisons` recomputes the block with the
  actual number of testable pairs in each `(task, n_kv)` cell, stores the
  per-cell `resolution` summary, and aggregates `n_resolution_limited`
  into the report. `_print_pairwise_report` prints a `RESOLUTION:` line
  when any pair is affected, and the cross-op claim text says
  "no test CAN separate" (rather than "no test separates") when the
  family has no rejection region.
* `experiment_design.planning_report` adds `exact_test_resolution` and
  `exact_test_warning`; `audit_run` adds the
  `exact_test_has_rejection_region` check and labels a run whose exact
  test cannot reject as `screening`, even if the normal-approximation
  power target is met. `validate_plan` rejects a registry entry whose
  declared confirmatory budget has no exact-test rejection region.
* `docs/experiment-design.md` (planning) and `docs/seed-influence.md`
  (single-seed deletion) are the companion documents.

Example:

```python
from multi_seed_stats import exact_test_resolution, summarize_paired

rep = exact_test_resolution(5, alpha=0.05, family_size=21)
rep['family_testable']            # False
rep['min_attainable_p_holm']      # 1.0
rep['min_effective_pairs_holm']   # 10
rep['max_comparisons_at_alpha']   # 0

panel = summarize_paired(diffs, family_size=21)
panel['test_resolution']['family_testable']  # False
panel['win_rate_ci95']                       # exact CP interval
```

## What this is not

* It is not a substitute for the permutation/t/sign tests. It says only
  whether the exact test *could* have rejected.
* It does not apply to the paired t-test, whose p-value is continuous.
  A t-test can report `p < 0.05` at `n = 5`; that is a different,
  normality-dependent claim.
* A resolution-limited family does not license a switch to whichever test
  happens to reject. The primary test is part of the pre-registered
  analysis plan; the honest response is to report the limit and add seed
  pairs (or reduce the confirmatory family to the pre-registered
  contrasts) in a new run.
* The floor assumes the sign-flip null and the recorded number of exactly
  tied pairs. Ties reduce `n_effective` and therefore *raise* the floor;
  the function reports both counts.

## Limitations

* `n_effective` is based on exact ties. A difference that is numerically
  tiny but nonzero still counts as informative, exactly as in
  `sign_test`; the floor is a property of the test, not of measurement
  resolution.
* `family_testable` is the first-rank condition. In a family with unequal
  common-seed counts, a later-ranked comparison can still reject after the
  smaller-p members; use `later_rank_possible` (and
  `run_quality.resolution.n_later_rank_possible`) for that weaker
  statement. The worked examples and the default runner use equal seed
  blocks, where the two questions coincide.
* The minimum-`n` arithmetic uses the smallest attainable p, not the
  attained p. A family can be testable while the observed configuration
  is still far from the rejection boundary.
* Monte-Carlo permutation (`n > max_exact`) has a resolution of
  `1 / (n_permutations + 1)` on top of the exact floor; the exact floor
  is still the relevant small-n limit.
* The median interval is distribution-free but assumes independent,
  exchangeable pairs. Shared data order or unpaired seeds invalidate it,
  just as they invalidate the paired tests.

## Code map

* `multi_seed_stats.exact_test_resolution` — the floor and minimum-n
  arithmetic.
* `multi_seed_stats.binomial_proportion_ci` — Clopper-Pearson.
* `multi_seed_stats.median_order_statistic_ci` — order-statistic median
  interval.
* `multi_seed_stats.sign_test` / `summarize_paired` — attach the win-rate
  CI and the resolution block.
* `run_quality._pairwise_family_resolution` and
  `_pairwise_op_comparisons` — per-cell and aggregate integration.
* `experiment_design.planning_report` / `audit_run` — prospective and
  post-run feasibility checks.
* `test_inference_resolution.py` — hand-computed floors, Clopper-Pearson
  endpoints, median coverage, JSON safety, and the runner integration.
