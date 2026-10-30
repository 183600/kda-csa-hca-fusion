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
| `assumed_sd` | working per-seed difference SD used for planning (a two-cell paired difference) |
| `equivalence` | optional TOST block: margin, family, alpha, target power, and whether an equivalence claim is required |
| `interaction` | optional 2×2 block: SESOI, family, alternative, `sd_inflation` (`sqrt(2)` by default), and whether an interaction claim is required |
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

### Exact-test resolution is a separate feasibility check

A continuous normal/t power calculation can certify a design that the
discrete exact test can never reject. The exact sign-flip test has only
`2**n_effective` sign assignments, so its two-sided p floor is
`2 / 2**n_effective`; a Holm family of `f` comparisons has a floor of
`f` times that. `planning_report` therefore also returns
`exact_test_resolution` (`family_testable`, `min_attainable_p_holm`,
`min_effective_pairs_holm`, `shortfall_pairs_holm`) and an
`exact_test_warning`, and `audit_run` adds the
`exact_test_has_rejection_region` check. If the floor exceeds `alpha`, no
outcome could have been significant and the run is labelled `screening`
even when the normal power target is met; `validate_plan` also rejects a
registry entry whose declared confirmatory budget has no exact-test
rejection region. See
[inference-resolution.md](inference-resolution.md) for the arithmetic and
worked examples (the default 5-seed, 21-pair quality sweep is
resolution-limited; the planned 11-seed, 5-contrast Exp 4 budget is not).

## Superiority and practical equivalence are different budgets

The registry contains **two** pre-specified effects, not one:

* `sesoi` (0.10 accuracy points) is the smallest **superiority** difference
  the confirmatory run is powered to detect;
* `equivalence.margin` (0.05 accuracy points) is the **TOST** margin:
  differences inside `±0.05` are called practically the same.

They answer different questions and need different sample sizes. For a
seed-paired design with per-seed difference SD `sd`, the number of seed pairs
needed to *declare* equivalence at a true difference of zero is

```
n_eq = ceil(((z_{1-alpha/f} + z_{1-beta/2}) * sd / margin)^2)
```

where `alpha` is the family-wise alpha and `f` is the TOST family size. The
`z_{1-beta/2}` term (rather than the superiority `z_{1-beta}`) appears because
the `(1-2*alpha)` interval must clear the margin on **both** sides. The
function is `experiment_design.equivalence_seeds_required`; the forward power
is `equivalence_power`.

The TOST family is wider than the superiority family. `run_quality` applies
one Holm correction over *every* operator pair in a cell (21 pairs for the
default seven operators), not just the five vs-softmax contrasts, and
`run_ablation` uses one family over all contrast rows. At `sd = 0.10`,
`margin = 0.05`, and 80% target power the planning report gives:

| plan | registered n | TOST family | TOST n at margin 0.05 |
|---|---:|---:|---:|
| `exp4_mqar` | 11 | 21 | 68 |
| `exp5_ratio` | 14 | 6 | 55 |
| `exp5b_matrix` | 14 | 24 | 69 |

At the registered budgets and the **planning** SD, the `(1-2*alpha)`
interval half-width already exceeds 0.05, so with a sample SD at the planning
value no observed mean could produce an equivalence verdict:
`planned_power_at_zero` is 0 and `minimum_supportable_margin_at_n` is
0.098-0.124, above the registered margin.
(If a *pre-run pilot* shows a much smaller SD, the planning report recomputes
from that SD; plugging the run's own realized SD back in to upgrade a
post-hoc TOST is circular and is not a confirmatory design.)
Those plans therefore record `required_for_claim: False`. Equivalence is a
secondary/descriptive endpoint for them; only an equivalence-powered run may
support a "practically the same" claim. This is deliberate and auditable, not
an omission:

* `planning_report(plan)['equivalence']` reports the margin, TOST family,
  corrected alpha, `required_seeds_at_zero`, the realized power at zero /
  half-margin / margin edge, `minimum_supportable_margin_at_n`, and
  `supports_equivalence_at_planned_n`;
* `audit_run` returns `equivalence_claim_allowed` and adds
  `practical_equivalence_at_margin` to `claims_allowed` **only** when the
  seed budget supports it;
* an underpowered TOST produces a `design_warnings` entry; a
  `required_for_claim: True` plan whose budget does not reach `n_eq` fails
  `validate_plan`, so the registry cannot claim a powered equivalence test
  it does not have.

`exp8_mechanism` intentionally has no equivalence block: that runner reports
non-significance as inconclusive and makes no TOST claim.

## Interactions are a separate four-cell budget

A pre-registered 2×2 interaction is a different estimand from a main effect,
and it needs a different seed budget. The registry's `assumed_sd` is the SD of
a **two-cell** paired difference (`sd_diff`, the quantity the pilot extracts).
An interaction `I = d - b - c + a` is a **four-cell** difference-in-
differences. Under the shared-seed / independent-cell-noise model that
motivates the paired design (cell value = treatment mean + shared seed effect +
independent cell noise of variance `σ²`),

```
Var(B - A) = 2 σ²          (the shared seed effect cancels)
Var(I)     = 4 σ²          (the shared seed effect also cancels)
SD(I)      = sqrt(2) · SD(B - A)
```

so the interaction's planning SD is `sqrt(2) * assumed_sd`. Reusing the
main-effect SD for a four-cell contrast overstates its power and understates
the seed requirement. The planning formula is the same normal approximation
with the inflated SD:

```
n_int = ceil(((z_crit + z_power) * sqrt(2) * sd / SESOI_int)^2)
```

and `z_crit` uses the alpha corrected for the family the interaction is
actually tested in — the **shared** run-wide family, not the number of
interaction rows. `experiment_design.interaction_planning` is the helper;
`planning_report(plan)['interaction']` carries `sd_inflation`,
`sd_interaction`, `required_seeds`, `planned_power`, `planned_mde`,
`shortfall_seeds`, the exact-test resolution, and
`supports_interaction_at_planned_n`.

At `sd = 0.10` (paired-difference SD), `SESOI_int = 0.10`, 80% target power:

| plan | interaction estimand | shared family | registered n | interaction n | power at registered n | MDE at registered n |
|---|---|---:|---:|---:|---:|---:|
| `exp5b_matrix` | `I_route(D) = (D-A)_r1 - (D-A)_r2` | 24 | 14 | 31 | 0.33 | 0.148 |
| `exp8_mechanism` | `no_rope_dense - no_rope - dense + baseline` | 6 | 14 | 25 | 0.50 | 0.132 |

The transfer/complementarity question is therefore **descriptive** at the
registered 14-seed budgets, even though those budgets are confirmatory for the
main effects. This is deliberate and auditable:

* `planning_report` reports the interaction budget next to (not inside) the
  main-effect record, so "confirmatory for a two-cell contrast" cannot be read
  as "confirmatory for the difference-in-differences";
* `audit_run` returns `interaction_claim_allowed` and adds
  `interaction_claim_at_sesoi` to `claims_allowed` **only** when the run
  reaches the interaction requirement; otherwise it adds a `design_warnings`
  entry and leaves the main-effect status unchanged (the main experiment is
  still confirmatory; only the interaction claim is withheld);
* both plans record `required_for_claim: False`. A plan that sets it to `True`
  while its `planned_seeds` falls short fails `validate_plan`, exactly like an
  unpromotable TOST endpoint;
* `sd_inflation` may be replaced by a value calibrated from a dedicated
  four-cell pilot, but `validate_plan` rejects a value below 1.0: a four-cell
  contrast cannot be less variable than the two-cell contrast it is built from
  without a negative cross-cell correlation, which the design does not assume.

`pilot_report` surfaces `required_seeds_interaction` next to the superiority
and TOST recommendations. A pilot that is meant to size an interaction claim
should still measure the difference-in-differences SD directly (the same four
variants on the same seeds) rather than relying on the `sqrt(2)` model
assumption; the pilot path uses the model because the generic pilot extractor
only sees two-cell `sd_diff` rows.

The assumption is a model, not a measurement. Positive cross-cell correlation
of the treatment noise makes the `sqrt(2)` factor conservative; unequal cell
variances or a negative correlation invalidate it. The exact-test feasibility
check is also recomputed for the interaction family, because a small family
can be testable for the main effects but not for a larger interaction family.

## Statuses and claims

`audit_run(plan_id, seeds=..., levels=..., family_size=...,
analyzed_seeds=...)` compares an actual run with the plan:

| Status | Meaning | Allowed claims |
|---|---|---|
| `confirmatory` | seed budget reaches the required n, the full planned family is present, required controls are present, and the analyzed population is the complete requested block | corrected paired inference at the planned alpha; a practical-equivalence claim additionally requires `equivalence_claim_allowed=True` |
| `reduced_family` | the seed budget is adequate only for a subset of the pre-registered contrast family | confirmatory only for the tested subset |
| `screening` | fewer seeds than required (or a larger-than-planned family) | descriptive / exploratory inference only |
| `incomplete_attrition` | the analyzed seed block is smaller than the requested block, or contains seeds outside it (see below) | descriptive; complete-case paired inference only |
| `exploratory_missing_controls` | a required control or level is absent | descriptive only; the treatment effect is confounded |
| `invalid` | fewer than two unique non-negative integer seeds | none |

The runners attach this audit to their result metadata as
`metadata.experiment_design`, so a reader can see whether a result was
confirmatory without reconstructing the seed count by hand. The audit never
changes the experiment or its exit code; it only labels the evidence.

## Single-factor claims are audited against the built config and observed run

A "leave-one-factor-out" table is only as trustworthy as its implementation.
Exp 8 therefore carries three complementary audits in
`metadata.ablation_design`:

* `status` / `missing_single_factor_variants` classify the *declared* variant
  table: every factor needs the baseline plus a variant that differs from it
  in exactly that factor, and a variant changing two or more factors is an
  interaction cell.
* `config_audit` rebuilds each variant's `HybridConfig` and compares it with
  the baseline. The fields that actually change (plus the post-construction
  `use_ste` switch) must be exactly the switches mapped from the declared
  factors. A table entry that says "RoPE only" while the config also changes
  `csa_topk` is reported as `mismatch`, not as a single-factor arm.
  Non-baseline variants whose built state is identical to the baseline (or to
  another variant) are reported as `no_change_variants` /
  `duplicate_treatments`; they are not treatments.
* `controlled_variables` checks the observed rows. Every declared variant
  must have a result row, and those arms must use the same training recipe
  (steps, batch, learning rate), parameter count, depth and successfully
  evaluated seed set. A drift is reported as `missing_results`,
  `recipe_mismatch`, `capacity_mismatch`, `seed_mismatch` or
  `partially_observed`. A mechanism switch that silently changes capacity or
  training budget is a second factor even if the built config fields match.

`single_factor_claims_valid` is true only when all three checks pass (the
runtime check is required whenever rows are supplied). The reasons for a
false verdict are listed in `single_factor_claim_failures`, and the
corresponding interaction verdict in `interaction_claims_valid` /
`interaction_claim_failures`. The runner logs a warning otherwise; the audit
never deletes or rewrites a result row, it labels the evidence.

The `rope` factor is a **declared bundle**: disabling partial RoPE on both
the CSA and HCA core-attention layers is one treatment factor, so the audit
expects both `csa_rope_dim` and `hca_rope_dim` to change together. It does
not support a claim about RoPE in CSA alone; that would need a separate
variant that leaves HCA's rotation intact.

## Requested vs analyzed seeds (attrition)

A pre-registered `n = 11` is a statement about the seeds the experiment
*requests*, not about the seeds that happen to survive. A training run may
crash, diverge, or produce a non-finite metric; the aggregate then drops it
and reports the mean over the remaining seeds. That is a **complete-case
analysis**. If the failure is related to the treatment — for example a wider
CSA layout that OOMs, or a different memory route that diverges — the
surviving-seed difference is biased, and no amount of post-hoc correction can
recover the missing counterfactual.

`audit_run(..., analyzed_seeds=[...])` therefore takes the population that
actually entered the analysis. `common_analyzed_seeds(rows)` computes the
conservative default: the intersection of the seed ids with usable `per_seed`
records in every row (a row-level error or a missing `per_seed` list yields
the empty set). The audit reports:

* `analysis_population.n_requested` / `n_analyzed` / `n_missing`;
* `missing_seeds` (requested but absent) and `unexpected_seeds` (analyzed
  but outside the requested block);
* `attrition_rate` and `complete_case`;
* `planning_at_analyzed_n` — the realized power/MDE at the analyzed n, next
  to the planned power at the requested n.

An incomplete block is labelled `incomplete_attrition` even when the
surviving n still clears the planned required n. The estimate is not
"wrong"; it is a complete-case estimate, and the label prevents it from
being presented as the pre-registered result. The only ways back to
`confirmatory` are to analyse the full block or to establish
(pre-specifiably) that the missing seeds are unrelated to the treatment.

The Exp 4, Exp 5/5b and Exp 8 runners pass this population automatically.
Exp 4 intersects across every operator/task/n_kv row; the ablation matrix and
ratio sweep intersect across every group; the mechanism ablation intersects
across every variant. A seed lost anywhere in the block therefore downgrades
the whole block, which is conservative on purpose: the pre-registered family
is a joint claim, not a collection of independently salvageable cells.

```python
from experiment_design import audit_run, common_analyzed_seeds

# rows: the result-block rows, each with a per_seed list of success/error
# records. The intersection is the conservative complete-case population.
population = common_analyzed_seeds(
    rows, finite_keys=('final_acc', 'final_loss'))
audit = audit_run(
    'exp4_mqar', seeds=[42 + i for i in range(11)],
    levels={'operator': ['linear', 'softmax', 'kda']},
    family_size=5, analyzed_seeds=population)
audit['status']              # confirmatory / incomplete_attrition / ...
audit['analysis_population']  # missing seeds, attrition rate, complete_case
```

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
`recommended_seeds`. It also returns `required_seeds_equivalence` and
`required_seeds_equivalence_conservative` when the plan registers a TOST
endpoint; those are separate, usually much larger numbers, and the
superiority `recommended_seeds` is not sufficient for an equivalence claim. A
pilot is **never** confirmatory: the same rows cannot both choose the sample
size and test the hypothesis, so `pilot_is_confirmatory` is always `False`,
even when the pilot happens to have enough seeds.

The SD estimate from a small pilot is itself uncertain, which is why the
conservative recommendation uses the maximum rather than the pooled estimate.
After the confirmatory run, use the audit on the new seed count, not on the
pilot.

## The pipeline manifest

`run_all.py` writes `results/experiment_design.json` before any experiment
starts. It records the actual seed budget planned for each experiment in this
pipeline invocation (`MQAR_SEEDS`, `ABL_SEEDS`, `MECH_SEEDS`,
`RECALL_SEEDS`), the required n at the declared SESOI, the resulting status,
and a canonical `plan_sha256` per row. When provenance capture succeeds it also embeds the
`capture_provenance()` block, binding the pre-registration to the working
tree and runtime that wrote it. This is a provenance artifact: if the plan is
edited after the run, the committed manifest and the result metadata no
longer agree.

## Binding a result to the plan it was audited against

The audit's status alone does not show *which* plan definition produced it,
and the registry is code that can be edited after a run. `audit_run()`
therefore records the same canonical `plan_sha256`, its documented scope, and
a full `plan_definition` snapshot in every result that embeds the audit.
`reproducibility.compare_design_binding()` re-hashes the snapshot and compares
it with the current registry: a post-hoc change to the SESOI, seed budget,
required controls, analysis family, or decision rule is reported as
`registry_drift` (exit 1), and an edited snapshot as `snapshot_mismatch`.
Legacy results without a fingerprint stay readable and are reported as
`unbound` with a warning.

```bash
# The design: line is added to the usual provenance/integrity report.
python reproducibility.py results/exp4_mqar.json
```

See [preregistration-binding.md](preregistration-binding.md) for the field
contract and the full status table. The snapshot is evidence of the plan
content, not a cryptographic signature; the before-the-run ordering still
comes from the committed history and from `run_all.py` writing the manifest
before any experiment starts.

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

## Interactions and capacity are separate estimands

The registry's main-effect budgets do not automatically cover interaction
claims. Exp 5b now pre-registers the route × layout transfer interaction
`I_route(D) = (D-A)_route1 - (D-A)_route2` (3 rows, family 24), and Exp 8
pre-registers the `RoPE × selection-density` interaction
`I = no_rope_dense - no_rope - dense + baseline` (family 6). Both use
`multi_seed_stats.interaction_contrast`, a seed-paired
difference-in-differences; the separate main-effect contrasts cannot
substitute for them. A non-significant interaction is inconclusive, never
evidence of additivity or route-independent transfer.

Planning treats the interaction as its own endpoint (see “Interactions are a
separate four-cell budget” above). The `int_n` column of the CLI table is the
interaction seed requirement; `audit_run.interaction_claim_allowed` is true
only when the run reaches it; and the registered 14-seed budgets are
explicitly `required_for_claim: False` for the interaction, so they are
descriptive for the transfer/complementarity question.

The fixed-structure matrix is not parameter-matched. Every two-cell contrast
carries `capacity_matched` / `claim_scope`, and
`_matrix_design_audit.capacity_audit` reports the per-cell parameter gap
against the pre-registered 5% tolerance. Outside that tolerance, a
significant layout effect is an **operator bundle plus capacity** effect and
must not be written as a capacity-neutral mechanism result. See
[comparative-ablation.md](comparative-ablation.md) for the full decision
rules and field definitions.

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
  a floor to beat, not a guarantee. The TOST planning formula is the standard
  normal shortcut: it uses `z` rather than the `t` critical value and ignores
  the correlation between the two one-sided statistics, so `n_eq` is a lower
  bound and the realized power is a little below the nominal target at
  moderate n.
* Superiority and equivalence are separate budgets. A run powered to detect
  the SESOI is not automatically powered to declare equivalence at the tighter
  TOST margin, and a non-significant superiority test is never evidence of
  equivalence. The current registry records `required_for_claim: False` for
  every equivalence endpoint, so those TOST verdicts are descriptive; an
  equivalence claim needs a dedicated run at `required_seeds_at_zero`.
* The assumed SD is an assumption. Replace it with a pilot estimate and use
  the conservative recommendation before committing a confirmatory budget.
* The family size used for planning is a conservative model of the tests the
  run will perform. If a runner adds cells or contrasts, pass the actual
  `family_size` to `audit_run`; the planning report then uses the corrected
  alpha for that larger family.
* The attrition audit labels a complete-case analysis; it does not repair
  one. There is no imputation for a diverged or crashed seed, and no
  statistical test can show that the dropout is unrelated to the treatment.
  If the missing seeds are plausibly treatment-related, the honest options
  are to report the complete-case estimate with its attrition, or to fix the
  failure and re-run the whole block.
* The registry covers the multi-seed inference experiments. Timing experiments
  (Exp 2/6/9) need a different design vocabulary (repeats, interleaving,
  order effects); they are not squeezed into the seed-power model.
* A plan is not a substitute for a clean data split. The data-separation
  contract lives in the runners (`run_quality.py`'s train / validation /
  headline-eval / bad-case streams).
