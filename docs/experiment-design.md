# Pre-registered experiment design and prospective power planning

This document describes `experiment_design.py`, the planning layer that sits
in front of the multi-seed quality/ablation experiments. It does not replace
the statistical tests in `multi_seed_stats.py`; it decides **before a run**
what those tests will be asked to detect, with how many seed pairs, and
under which correction.

## Why a separate planning layer?

The result files already carry post-hoc `min_detectable_effect` and
`seeds_required_80pct_power` fields. Those are useful for planning a follow-up,
but they cannot justify the run that produced them: computing the effect size
after seeing the data makes the seed budget look adequate by construction. The
same problem appears in subtler forms:

* choosing the SESOI after seeing which effects were significant;
* reporting the raw family-wise alpha when the analysis actually tests many
  contrasts;
* treating a 5-seed screening run as if it were the pre-registered
  confirmatory run;
* dropping the memoryless `linear` control (or the tuned `softmax`
  reference) from an operator sweep and still reading the remaining rows as a
  controlled comparison;
* using the same pilot rows both to estimate the variance and to test the
  hypothesis.

`experiment_design.py` makes those decisions explicit and machine-readable.

## The registry

`experiment_design.PLANS` contains one record per multi-seed experiment:

| Field | Meaning |
|---|---|
| `question` | the research question the experiment answers |
| `design` | the design type (seed-blocked paired comparison, factorial, leave-one-factor-out) |
| `unit` | the experimental unit (one trained model at one factor combination and seed) |
| `factors` | treatment factors, their levels, and the seed block |
| `required_levels` | controls that must be present for a claim (e.g. `linear` and `softmax`) |
| `primary_contrast` | the pre-registered estimand |
| `endpoint` / `direction` | the outcome and its interpretation |
| `sesoi` | smallest effect of interest, in endpoint units |
| `assumed_sd` | working per-seed difference SD used for planning |
| `alpha` / `planned_family_size` / `alternative` | the analysis family and correction |
| `target_power` | planning target (80% by default) |
| `screening_seeds` | the cheap default budget, explicitly non-confirmatory |
| `planned_seeds` | the confirmatory budget, validated against the power arithmetic |
| `analysis` / `decision_rule` / `stopping_rule` | how the data will be analysed and what will not be done |

The registry is code, so a change to the SESOI, the assumed SD, or the family
size is a reviewable diff. `validate_plan` also recomputes the power of the
declared confirmatory budget: a plan whose `planned_seeds` is below its own
required n is rejected instead of silently drifting.

## Planning arithmetic

For a seed-paired difference with mean effect `SESOI` and per-seed SD `sd`,
the standardized effect is `d_z = SESOI / sd`. The planning model is the
normal approximation already used by `multi_seed_stats`:

```
n_required = ceil(((z_crit + z_power) / d_z)^2)
```

where `z_crit` is the **corrected** per-test critical value:

* one-sided family: `alpha / family_size`;
* two-sided family: `alpha / (2 * family_size)`.

The MDE at the planned n is `(z_crit + z_power) * sd / sqrt(n)`, and the
planned power at the SESOI is the forward normal approximation in
`multi_seed_stats.power_at_effect`. The module reports all of these next to
the inputs so the decision is auditable.

The approximation is for **planning only**. Inference continues to use the
exact sign and sign-flip permutation tests and the paired t-test in
`multi_seed_stats`; the normal approximation is slightly optimistic at the
5-7-seed scale.

## Statuses and claims

`audit_run(plan_id, seeds=..., levels=..., family_size=...)` compares an actual
run with the plan:

| Status | Meaning | Allowed claims |
|---|---|---|
| `confirmatory` | seed budget reaches the required n, the full planned family is present, and required controls are present | corrected paired inference at the planned alpha |
| `reduced_family` | the seed budget is adequate only for a subset of the pre-registered contrast family | confirmatory only for the tested subset |
| `screening` | fewer seeds than required (or a larger-than-planned family) | descriptive / exploratory inference only |
| `exploratory_missing_controls` | a required control or level is absent | descriptive only; the treatment effect is confounded |
| `invalid` | fewer than two unique non-negative integer seeds | none |

The runners attach this audit to their result metadata as
`metadata.experiment_design`, so a reader can see whether a result was
confirmatory without reconstructing the seed count by hand. The audit never
changes the experiment or its exit code; it only labels the evidence.

## Pilot -> power -> confirmatory

A screening run can estimate the paired SD. Given a result file:

```bash
python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json
```

`pilot_sd_from_rows` supports two shapes:

* per-seed rows (Exp 4/5/8): for every treatment/reference cell it forms the
  seed-paired differences and computes the pooled within-pair SD;
* contrast rows that already carry `sd_diff` (Exp 5b side file): the median
  reported SD is the primary estimate and the maximum is the conservative one.

Control levels listed in `pilot.exclude` (for Exp 4, the memoryless `linear`
rung) are not treatments and are excluded from the SD estimate.

The report returns `required_seeds_pooled_sd`,
`required_seeds_conservative` (using the largest per-contrast SD) and
`recommended_seeds`. A pilot is **never** confirmatory: the same rows cannot
both choose the sample size and test the hypothesis, so
`pilot_is_confirmatory` is always `False`, even when the pilot happens to have
enough seeds.

The SD estimate from a small pilot is itself uncertain, which is why the
conservative recommendation uses the maximum rather than the pooled estimate.
After the confirmatory run, use the audit on the new seed count, not on the
pilot.

## The pipeline manifest

`run_all.py` writes `results/experiment_design.json` before any experiment
starts. It records the actual seed budget planned for each experiment in this
pipeline invocation (`MQAR_SEEDS`, `ABL_SEEDS`, `MECH_SEEDS`), the required n
at the declared SESOI, and the resulting status. This is a provenance artifact:
if the plan is edited after the run, the committed manifest no longer matches
the result metadata.

## CLI

```bash
# Show the plan table (screening vs confirmatory budgets).
python experiment_design.py

# Validate the committed registry; exit 1 if a plan is stale/malformed.
python experiment_design.py --check

# Write the versioned manifest.
python experiment_design.py --json results/experiment_design.json

# What-if planning: override the seed count, the analysis family, or the SD.
python experiment_design.py --plan exp5b_matrix --seeds 20 --family-size 15
python experiment_design.py --plan exp4_mqar --sd 0.05

# Pilot analysis from a result file (or supply an SD directly).
python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json
python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json --sd 0.12
```

## Post-run sensitivity

Planning fixes the seed budget and the primary analysis before the data
exist. It does not make the realized seed set interchangeable: at 5-7 seeds
one diverged run can still carry a mean difference. After the confirmatory
test, `multi_seed_stats.leave_one_out_sensitivity` deletes each seed in turn
and reports whether the direction or the primary decision depends on that
seed. The exp4 operator comparisons carry the block automatically and use a
`direction_fragile` label to qualify a significant ordering. This is
descriptive only — it re-uses the same data, is not multiplicity-corrected,
and never upgrades a claim. See
[seed-influence.md](seed-influence.md) for the status definitions and the
`n = 6` exact-test caveat.

## Limitations

* The normal approximation is optimistic at small n; treat the target power as
  a floor to beat, not a guarantee.
* The assumed SD is an assumption. Replace it with a pilot estimate and use
  the conservative recommendation before committing a confirmatory budget.
* The family size used for planning is a conservative model of the tests the
  run will perform. If a runner adds cells or contrasts, pass the actual
  `family_size` to `audit_run`; the planning report then uses the corrected
  alpha for that larger family.
* The registry covers the multi-seed inference experiments. Timing experiments
  (Exp 2/6/9) need a different design vocabulary (repeats, interleaving,
  order effects); they are not squeezed into the seed-power model.
* A plan is not a substitute for a clean data split. The data-separation
  contract lives in the runners (`run_quality.py`'s train / validation /
  headline-eval / bad-case streams).
