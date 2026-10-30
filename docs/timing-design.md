# Pre-registered timing measurement design

This document describes `timing_design.py`, the planning layer for the two
latency experiments: Exp 2 (`run_benchmark.py`) and Exp 6
(`run_decoding.py`). It complements
[experiment-design.md](experiment-design.md), which covers the multi-seed
training comparisons.

## Why timing needs its own design

The training experiments use a trained model as the experimental unit and
seed-to-seed variation as the uncertainty. Timing experiments have neither:
the unit is a timed trial; repeats estimate within-cell noise; and the main
threats are order effects, warmup transients, allocator state, and device
frequency drift. Treating repeats as independent seeds, or ignoring the order
in which operators are measured, produces confident-looking ratios that do
not reproduce.

`timing_design.py` decides **before a run**:

* which A/B contrasts are primary, and their multiplicity family;
* the smallest effect of interest (SESOI) as a latency **ratio**;
* the assumed coefficient of variation (CV) and the resulting repeat budget;
* a deterministic, order-balanced measurement schedule;
* what claim a completed table can support.

The module is stdlib-only so the design manifest can be written before torch
is installed or a GPU is available.

## Registry

`TIMING_PLANS` contains one record per experiment. Both plans use a
two-sided 5% family-wise alpha, 80% target power, a 1.15 latency-ratio SESOI
(15% slower/faster), and an assumed per-cell CV of 10%.

| Plan | Runner | Contrasts | Family | Screening | Confirmatory | Warmup |
|---|---|---|---|---:|---:|---:|
| `exp2_benchmark` | `run_benchmark.py` | `hybrid_mamba2/hybrid`, `kda_chunk/kda_rec`, `mamba2_chunked_scan/mamba2_scan` | 15 | 5 | 15 | 2 |
| `exp6_decoding` | `run_decoding.py` | `hybrid_mamba2/hybrid`, `csa/hca`, `kda/mamba2` | 12 | 3 | 15 | 1 |

The family sizes are the number of registered contrasts times the default
number of workload blocks (5 sequence lengths for Exp 2, 4 prefill lengths
for Exp 6). `audit_run` accepts the actual `family_size` so a run with a
different workload set uses the correspondingly corrected alpha.

Each plan also records the endpoint, allocation rule, analysis, decision
rule, and stopping rule. Changing a SESOI, CV, family size, or repeat budget
is a reviewable diff, and `plan_sha256` fingerprints the plan definition.
Per-invocation overrides (e.g. `BENCH_REPEATS`) are recorded separately in
the manifest and do not rewrite the registered plan.

## Planning arithmetic

Latency is analysed as a **ratio of medians**. On the log scale the standard
error for one cell with CV `c` and `n` repeats is approximately
`c / sqrt(n)`. For two independently timed cells the unit standard error is

```
sd_log = sqrt(cv_a^2 + cv_b^2)
```

and the required repeats are

```
n = ceil( ((z_crit + z_power) * sd_log / log(SESOI_ratio))^2 )
```

where `z_crit` uses the multiplicity-corrected alpha:

* one-sided family: `alpha / family_size`;
* two-sided family: `alpha / (2 * family_size)`.

At the registered inputs this gives `n = 15` for both plans, with planned
power 0.81 (Exp 2) and 0.83 (Exp 6). `ratio_log_se_unit` also supports a
paired correlation (`rho`), but the two benchmark cells are treated as
independently timed by default.

This is **planning only**. The exact bootstrap ratio interval from raw
trials is `timing_stats.ratio_of_medians_ci`. When only per-cell summaries
(CV and `n`) are available, the audit reports a normal log-ratio interval and
labels it as a summary-level approximation.

## Balanced schedule

Within each workload block every operator is measured exactly once; the
block order is a cyclic rotation of a seeded random permutation. This means:

* operator identity is not confounded with position when the number of
  blocks covers the operator set;
* the same seed reproduces the same schedule (`DEFAULT_SCHEDULE_SEED = 0`);
* global RNG state is never touched (a local `random.Random(seed)` is used).

`balanced_schedule` and `_schedule_balance` report whether the realized
schedule is position-balanced. `audit_run` derives the schedule from the
recorded `schedule_block` / `order_position` fields when the runner does not
pass one explicitly, so an archived result file can be audited without
re-running.

The runners record `warmup`, `schedule_id`, `schedule_block`, and
`order_position` on every row. Warmup passes are untimed: Exp 2 runs
`min(2, repeats)` warmup calls per cell; Exp 6 runs one full untimed
prefill + decode loop per cell.

Exp 2 reads its repeat budget from `BENCH_REPEATS`; Exp 6 from
`DECODE_REPEATS` (default 3). Exp 6 also records `decode_trial_cv` and
`decode_trial_n`: the CV across per-trial median token latencies and the
number of full prefill+decode trials. Those are the independent replicates.
Token-level `decode_cv` / `decode_n` are still reported for continuity, but
the audit uses the trial-level fields so correlated token timings are not
counted as independent observations. The reported point estimate remains the
existing median per-token latency; the trial-level CV is the uncertainty
input, not a replacement endpoint.

## Run audit

`audit_run(plan_id, cells, ...)` assigns one status:

| Status | Meaning |
|---|---|
| `invalid` | cells cannot be normalized (non-finite, non-positive, bad metadata) |
| `incomplete_cells` | a registered contrast cell is missing |
| `repeats_inconsistent` | repeat counts differ across cells |
| `order_confounded` | realized schedule is not position-balanced |
| `unplanned_warmup` | warmup missing or below the registered minimum |
| `device_mixed` | one table contains more than one device |
| `screening` | below the confirmatory repeat budget, a reduced workload grid, or realized precision at the SESOI is insufficient |
| `confirmatory` | complete, balanced, warmed up, at budget, and every contrast MDE ≤ SESOI |

The runners pass the intended workload grid explicitly, so losing an entire
sequence/prefill block is reported as `incomplete_cells`; without that input
the audit can only infer the grid from the rows that survived.

Only `confirmatory` adds the `relative_latency_ratio_at_sesoi` claim. All
other statuses may still report `descriptive_relative_latency`, but not a
confirmatory latency claim. The audit never changes a timing; it changes what
the table is allowed to support.

`pilot_report` reads per-cell CVs from a pilot run and returns a conservative
repeat recommendation (the maximum over cells). A pilot selects the budget;
it cannot also supply the confirmatory evidence. After fixing the budget,
re-run and audit the new table.

## Integration

* `run_benchmark.py` and `run_decoding.py` use the balanced schedule, write
  the design fields on every row, and attach `timing_design` to their
  provenance/metadata.
* Both runners archive the per-trial timings in
  `results/exp2_benchmark_raw.json` / `results/exp6_decoding_raw.json`
  (`kind: timing_raw`). Exp 6 archives the per-trial median decode latency,
  i.e. one sample per full prefill+decode trial, not every token. These files
  feed `timing_stats.ratio_of_medians_ci`; the provenance also records exact
  bootstrap contrast rows via `timing_design.raw_contrast_rows`.
  `python timing_design.py --raw results/exp2_benchmark_raw.json` prints them.
* `run_all.py` writes `results/timing_design.json` **before** the experiments
  start, using the same `BENCH_REPEATS` / `DECODE_REPEATS` env vars the
  runners read. `RUN_ALL_PROFILE=full` sets both to the 15-repeat
  confirmatory budget; `smoke` / `screen` remain screening runs.
* `results_schema.py` validates the core row contract; the design fields are
  additive metadata.
* `timing_design.py --check` validates the registry; `--plan PLAN_ID` prints
  the planning report; `--pilot results/....json` recommends a repeat
  budget; `--json PATH` writes the design manifest.

## Limitations

* The normal log-ratio model is an approximation; the realized CV can differ
  from the assumed 10%. The audit recomputes the MDE from the observed CV and
  downgrades the table to `screening` if the budget no longer meets the
  SESOI. The assumption is still an assumption, not a measurement.
* The schedule removes position confounding between operators but does not
  remove time trends within a block (thermal drift, other processes). The
  balanced order makes such drift spread across operators instead of
  favouring one, but it does not eliminate it.
* The audit's interval is summary-level (normal log-ratio from CV and
  repeats). Use the archived `*_raw.json` files and
  `timing_stats.ratio_of_medians_ci` / `timing_design.raw_contrast_rows` for
  the reported ratio interval, and treat the audit interval as a planning
  diagnostic. The exact bootstrap still assumes independent trial cells; it
  does not repair a confounded order.
* Exp 2's workload blocks are separate sequence lengths, and Exp 6's are
  separate prefill lengths. Effects are audited per workload; pooling across
  workloads requires a pre-registered aggregation rule, which these plans do
  not claim.
* The registry covers Exp 2 and Exp 6. Exp 9 (recall) and the training
  experiments have different units and remain in `experiment_design.py`.
