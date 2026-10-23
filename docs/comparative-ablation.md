# Comparative experiments and ablations: controls, interactions, and claim scope

This document is about **what a contrast is allowed to say**, not about which
number is largest. It covers the comparative design of Exp 5b (layout ×
memory-route matrix) and Exp 8 (CSA mechanism ablation), the two new
interaction estimands, and the capacity audit that qualifies the
fixed-structure matrix.

## 1. The baseline ladder is part of the experiment

A hybrid result is only interpretable after the simpler rungs have been run
with the **same data stream, depth, optimizer-step budget, and outer
structure**:

| Rung | Group | What it rules out |
|---|---|---|
| memoryless | `BASELINE-linear` | the task is solvable from the current token alone |
| standard | `BASELINE-softmax` | the hybrid only beats an untuned or under-budgeted attention baseline |
| pure recurrent | `A` | the gain comes only from replacing a recurrent layer with *any* other layer |
| single operator | `B`, `C` | the gain requires both CSA and HCA |
| full hybrid | `D` | the combination is complementary rather than additive |

`BASELINE-softmax` is tuned on the held-out validation stream, never on the
headline evaluation stream. Exp 8 has a related requirement: all mechanism
variants share depth and parameter count, so a mechanism contrast is not a
capacity contrast. The mechanism design audit (`ablation_design.config_audit`,
`controlled_variables`) checks the built config, parameter count, depth,
training recipe, and analyzed seed set rather than trusting the variant table.

## 2. Main effects do not test interactions

Exp 5b estimates, within each memory route `R`:

```
B_R - A_R      CSA replaces one recurrent slot
C_R - A_R      HCA replaces one recurrent slot
D_R - B_R      HCA added on top of CSA
D_R - C_R      CSA added on top of HCA
I_R = D_R - B_R - C_R + A_R       CSA × HCA complementarity
```

`I_R` is a **difference-in-differences** estimated per seed. It is not a
comparison of the `B-A` and `C-A` p-values: the two estimates share the `A`
cell and are correlated, and neither bounds their difference.

Exp 8 has a complete `RoPE × selection-density` 2×2 cell:

```
baseline          RoPE on,  sparse top-k
no_rope           RoPE off, sparse top-k
dense             RoPE on,  dense selection
no_rope_dense     RoPE off, dense selection

I = no_rope_dense - no_rope - dense + baseline
```

The pre-registered prediction is `I <= 0` (not super-additive). Only a
corrected **significant positive** `I` contradicts it. A non-significant `I`
is **inconclusive**: it is not evidence that the two ablations are additive.
Testing `no_rope`, `dense`, and `no_rope_dense` each against `baseline` cannot
substitute for this row.

The estimator is `multi_seed_stats.interaction_contrast`. It validates the
four seed→metric cells, intersects the seed sets, forms one interaction value
per seed, and reports `mean_diff`, `sd_diff`, `t_stat`, `dof`, Cohen's
`d_z`, `n_pairs`, and the unanimity of the per-seed signs. The interaction
p-value uses the same multiplicity family and the same exact/robustness
conventions as the main contrasts.

## 3. Route × layout transfer is its own estimand

The matrix's replacement contrasts `D_KDA - D_MAMBA2` and
`A_KDA - A_MAMBA2` answer "which route is better at this layout?" and "which
route has the better baseline?". They do **not** answer the transfer
question: "is the *benefit* of CSA+HCA the same under both routes?"

That estimand is the route × layout interaction:

```
I_route(D) = (D - A)_KDA - (D - A)_MAMBA2
```

It is zero when the hybrid benefit is route-independent, even if the two
routes differ by a large constant. It is estimated on the seeds common to
all four cells and is written as a two-sided contrast because no direction
is pre-registered. Non-significance is inconclusive, not evidence of
transfer; a transfer claim at this budget needs either a significant
interaction (route-dependent) or an equivalence-powered interaction test
(route-independent).

Exp 5b's pre-registered family is therefore 24 contrasts per default sweep
with three routes: 15 within-route (5 × 3), 6 replacement (A/D × 3 route
pairs), and 3 transfer interactions.

## 4. Fixed-structure is not capacity-matched

The matrix holds `d_model`, depth, outer residual/LayerNorm structure, data,
and optimizer budget fixed. Replacing one recurrent layer with CSA/HCA still
changes the trainable parameter count — often by tens of percent. A
significant `B - A` is therefore an **operator-bundle + capacity** effect
until the budget-matching follow-up closes the gap.

Every two-cell contrast row carries:

| Field | Meaning |
|---|---|
| `candidate_n_params` / `reference_n_params` | reported parameter counts |
| `param_ratio`, `param_delta_pct` | relative capacity change |
| `capacity_match_tolerance` | pre-registered tolerance (`0.05`) |
| `capacity_matched` | `True` only inside the tolerance; `None` if unobserved |
| `capacity_status` | `matched` / `candidate_larger` / `candidate_smaller` / `unknown` |
| `claim_scope` | `capacity_matched`, `operator_plus_capacity`, `interaction`, or `unknown_capacity` |
| `capacity_note` | human-readable consequence |

`_matrix_design_audit` aggregates the same cells into
`capacity_audit.status` (`capacity_matched` / `capacity_confounded` /
`unknown` / `not_computable`) and the per-cell gaps. This audit **labels** the
evidence; it never deletes a contrast or changes a p-value.

The label matters for wording:

* `capacity_matched`: "the CSA bundle changes accuracy" is still a bundle
  claim, but capacity is not a plausible alternative explanation.
* `operator_plus_capacity`: write "the CSA layer plus its parameter budget",
  and do not write "the CSA mechanism alone". Run the budget-matching
  follow-up (`docs/mamba2-framework.md` §5) before a mechanism claim.

Four-cell interactions are reported as `four_cell_functional`: there is no
single candidate/reference pair, so the capacity check is per cell. For
`I_route`, the transfer interaction is capacity-neutral only if **both**
routes' `(D - A)` contrasts are matched.

## 5. Multiplicity, attrition, and sensitivity

The interaction rows join the **same** Bonferroni family as the main
contrasts. They are not a separate exploratory table with an uncorrected
alpha. The runners record the shared family in `contrast_tests` (Exp 5b) or
`factorial_interactions.n_tests_in_shared_family` (Exp 8). Each Exp 8
interaction row also carries its own `min_detectable_effect` and
`seeds_required_80pct_power`, because a 2×2 interaction is usually much
less powered than either main effect.

That post-hoc read-out is not a prospective budget. `experiment_design`
therefore plans the interaction with its own four-cell SD
(`sqrt(2) * assumed_sd` under the shared-seed / independent-cell-noise
model) and reports `planning_report(plan)['interaction']`. At the planning
SD the registered 14-seed budgets give only 33% (Exp 5b, family 24) and 50%
(Exp 8, family 6) power for an interaction SESOI of 0.10; the required
budgets are 31 and 25 seeds. The registry records
`interaction.required_for_claim: False` for both plans, so
`audit_run.interaction_claim_allowed` is false at the registered budget and
the transfer/complementarity verdict is descriptive even when the main
effects are confirmatory. A run may only cite `I` as a confirmatory claim
after reaching the interaction requirement (or after a dedicated four-cell
pilot replaces the `sqrt(2)` assumption with a measured DiD SD). See
[experiment-design.md](experiment-design.md#interactions-are-a-separate-four-cell-budget).

The existing evidence-quality rules still apply and are not replaced by the
new rows:

* a non-significant contrast is inconclusive, never equivalence;
* TOST is the only route to a "practically the same" claim, and only against
  a margin fixed before the run;
* `common_analyzed_seeds` / `audit_run` label attrition instead of silently
  dropping failed seeds;
* `leave_one_out_sensitivity` reports whether a significant ordering depends
  on one seed;
* exact-test resolution decides whether the family could reject at all.

## 6. Decision rules

For Exp 5b:

1. A layout main effect may be cited only with
   `significant_bonferroni=True`; read `claim_scope` before calling it a
   mechanism effect.
2. `I=D-B-C+A` supports complementarity only when its own row is
   significant. A non-significant `I` is not additivity.
3. `I_route(D)` is the transfer claim. The replacement contrasts do not
   substitute for it.
4. A non-significant `I_route` is not evidence of transfer; the TOST fields
   decide only if the interaction was equivalence-powered (it is not at the
   registered budget).

For Exp 8:

1. Single-factor claims require the baseline + exactly-one-factor variant,
   the built config audit must pass, and the observed arms must share the
   training recipe, parameter count, depth and seed set.
2. The factorial interaction is a separate row; the four main effects do not
   replace it.
3. `no_rope_dense` supports only an interaction claim, not a RoPE-only or
   density-only claim.

## 7. Limitations

* The capacity audit measures parameter count, not compute or state size.
  Matching parameters does not match FLOPs, cache bytes, or optimization
  difficulty.
* The interaction test is a difference-in-differences under the paired-seed
  design. It assumes the same seeds are exchangeable across the four cells
  (they are), but it does not remove order/position confounds inside the
  layout — those are tested by the `position` preset.
* A 2×2 interaction has low power at 5-7 seeds. The interaction rows are
  reported with their own MDE/seed requirements; a planned interaction claim
  needs its own prospective power calculation, not the main-effect budget.
* The audit's 5% tolerance is a convention from the budget-matching protocol,
  not a physical threshold. It is pre-registered so it cannot be selected
  after seeing which contrasts were significant.
