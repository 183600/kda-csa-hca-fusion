"""Pre-registered experiment designs, prospective power planning, and run audits.

Motivation
----------
The repository already reports post-hoc minimum detectable effects and
``seeds_required_80pct_power`` estimates *after* a run. That is useful for
deciding what a follow-up would need, but it is not an experimental design:
the seed budget, the control set, and the smallest effect of interest have
to be fixed BEFORE looking at the outcome, otherwise the analysis quietly
becomes a search for a defensible effect size.

This module supplies the missing planning side of the contract. It keeps a
committed, machine-readable registry of the multi-seed experiments and, for
each one, records:

* the research question, experimental unit, factors, levels and controls;
* the primary endpoint and the pre-specified smallest effect of interest
  (SESOI), in the endpoint's own units;
* the analysis family and the multiplicity correction, so the planning
  alpha is the SAME corrected alpha the inference will use;
* the confirmatory seed budget derived from the SESOI, an assumed
  per-seed difference SD, and the target power (normal approximation,
  clearly labelled as planning rather than inference);
* the SEPARATE pre-specified equivalence margin used by the TOST, with
  its own family, so a "practically the same" claim is not smuggled into a
  budget that was sized for superiority. The equivalence margin and the
  SESOI are different scientific thresholds (the runner uses a margin of
  0.05 accuracy points while the superiority SESOI is 0.10), and a run
  powered to detect the SESOI is almost never powered to declare
  equivalence at the tighter margin;
* the stopping rule (fixed n, no optional stopping, no replacing failed
  seeds) and the decision rule.

Three things follow from that registry:

1. :func:`planning_report` computes the prospective required seed count and
   the power the current budget would actually have. A plan whose
   ``planned_seeds`` is below the required count fails :func:`validate_plan`,
   so the registry cannot silently drift away from its own arithmetic.
2. :func:`audit_run` compares an ACTUAL run configuration (seed list,
   observed factor levels, family size, the ANALYZED seed set, and a pilot
   SD) with the plan. It labels the run ``confirmatory`` /
   ``reduced_family`` / ``screening`` / ``incomplete_attrition`` /
   ``exploratory_missing_controls`` / ``invalid`` and states which claims
   the run is allowed to support. Runners attach this audit to their result
   metadata, so neither a 5-seed screening run nor a run that silently
   dropped failed seeds can be mistaken for the pre-registered confirmatory
   experiment.
3. :func:`plan_sha256` fingerprints the canonical plan definition and
   :func:`audit_run` carries both the fingerprint and a plan snapshot. A
   result can therefore be checked against the registry after the fact
   (``reproducibility.compare_design_binding``): changing the SESOI, seed
   budget, required controls or decision rule after a run is detectable,
   not just a silent code edit.

The distinction between the REQUESTED seed block and the seeds that
actually entered the analysis matters. A training run that crashes, diverges
or produces a non-finite metric is normally excluded from the aggregate, but
excluding it turns the pre-registered experiment into a complete-case
analysis. If failures are related to the treatment (OOM under a wider
layout, divergence under a different memory route, ...), the surviving-seed
mean is biased. :func:`common_analyzed_seeds` computes the seed ids that
survived in every row, and :func:`audit_run` downgrades an incomplete block
to ``incomplete_attrition`` instead of calling it confirmatory.

The pilot workflow
------------------
A screening run can estimate the paired-difference SD. :func:`pilot_report`
extracts that SD from the per-seed rows (or from contrast rows that already
carry ``sd_diff``), recomputes the required seed count, and returns a
recommendation. The pilot is still a pilot: the SD estimate is noisy, so the
report also carries a conservative recommendation based on the largest
observed per-contrast SD. This makes the screening -> power -> confirmatory
sequence explicit instead of an informal judgement call.

CLI
---
``python experiment_design.py`` prints the plan table.
``python experiment_design.py --check`` validates the committed registry
(exit 1 on a stale or malformed plan).
``python experiment_design.py --json results/experiment_design.json``
writes the versioned plan manifest.
``python experiment_design.py --plan exp4_mqar --pilot results/exp4_mqar.json``
computes the observed paired SD and the recommended confirmatory seed budget
without re-running the experiment.

Design constraints
------------------
* Stdlib only (``json`` / ``math`` / ``statistics``). The planning path must
  run before torch is installed or a GPU is available.
* Deterministic and JSON-safe. No global RNG is touched.
* The normal approximation is used for PLANNING only. Inference continues to
  use the exact sign / permutation tests and the t-test in
  :mod:`multi_seed_stats`; the docs and every returned record say so.
* The SESOI and the assumed SD are explicit, editable assumptions. They are
  not estimated from the data the plan is supposed to decide, and a pilot
  cannot become confirmatory merely by exceeding its own recommended n.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Mapping

from multi_seed_stats import (exact_test_resolution, min_detectable_effect,
                              power_at_effect, seeds_required)

__all__ = [
    'DesignError',
    'PLANS',
    'PLAN_IDS',
    'get_plan',
    'plan_sha256',
    'validate_plan',
    'validate_plans',
    'planning_report',
    'equivalence_seeds_required',
    'equivalence_power',
    'equivalence_plan',
    'interaction_planning',
    'INTERACTION_SD_INFLATION',
    'common_analyzed_seeds',
    'audit_run',
    'pilot_sd_from_rows',
    'pilot_report',
    'design_manifest',
    'write_design_manifest',
    'main',
]

DEFAULT_SEED_BASE = 42
DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
DEFAULT_ASSUMED_SD = 0.10

# A 2x2 interaction I = d - b - c + a is a FOUR-cell contrast, not a
# two-cell one. Under the shared-seed / independent-cell-noise model that
# motivates the paired design (cell value = treatment mean + shared seed
# effect + independent cell noise of variance sigma^2):
#
#   Var(B - A) = 2 * sigma^2
#   Var(I)     = 4 * sigma^2      (the shared seed effect cancels)
#
# so SD(I) = sqrt(2) * SD(B - A). The registry's ``assumed_sd`` is already
# a paired-DIFFERENCE SD (the pilot extracts ``sd_diff``), so interaction
# planning must inflate it by sqrt(2); reusing the main-effect SD for a
# four-cell contrast overstates its power and understates the seed budget
# by a factor of about two. Positive cross-cell correlation of the
# treatment noise makes this conservative; a negative correlation would
# violate it, so the assumption is recorded in every interaction report.
INTERACTION_SD_INFLATION = math.sqrt(2.0)


class DesignError(ValueError):
    """A plan, a run configuration, or a navigation key is malformed."""


# ---------------------------------------------------------------------------
# The committed registry
# ---------------------------------------------------------------------------
#
# The SESOI is deliberately expressed in ABSOLUTE accuracy points, because
# that is the scale the runners report and the scale a reader can judge. The
# assumed SD is a working value from the historical 5-7 seed screenings; use
# the --pilot path to replace it with an observed paired SD before planning a
# confirmatory run. Raising one seed count is cheap compared with a
# underpowered campaign, so the defaults are conservative.
#
# planned_seeds is the CONFIRMATORY budget checked by validate_plan; the
# default screening budget is recorded separately and is always labelled
# non-confirmatory by audit_run.
PLANS: dict[str, dict] = {
    'exp4_mqar': {
        'plan_id': 'exp4_mqar',
        'title': 'Exp 4 - synthetic quality probes (MQAR + task probes)',
        'runner': 'run_quality.py',
        'result_files': ['results/exp4_mqar.json',
                         'results/exp4b_quality_tasks.json'],
        'question': (
            'Does a memory operator beat a TUNED simple baseline on the '
            'pre-registered synthetic probes, at a matched optimizer-step '
            'budget?'),
        'design': 'seed-blocked paired comparison (same seed, same data '
                  'stream, treatment vs reference)',
        'unit': ('one trained model: (operator, task, n_kv, seed); seeds are '
                 'complete blocks shared by every operator'),
        'factors': {
            'operator': {
                'role': 'treatment',
                'levels': ['linear', 'softmax', 'kda', 'mamba2', 'rwkv7',
                           'csa', 'hca'],
            },
            'task': {
                'role': 'treatment',
                'levels': ['mqar', 'kv_update', 'long_range', 'multi_agg'],
                'note': ('the default sweep runs mqar only; the other probes '
                         'are opt-in via MQAR_TASKS'),
            },
            'n_kv': {
                'role': 'treatment',
                'levels': [1, 2, 4],
                'note': 'default MQAR_NKV=1',
            },
            'seed': {
                'role': 'block',
                'levels': '42 .. 42 + n_seeds - 1',
            },
        },
        'required_levels': {
            'operator': ['linear', 'softmax'],
        },
        'primary_contrast': (
            'each non-baseline operator minus the tuned softmax reference, '
            'paired by seed within (task, n_kv)'),
        'endpoint': 'final_acc',
        'direction': 'higher_is_better',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        # Complex ops vs the tuned reference in the default sweep. The
        # all-pairs family is larger, so 5 is the optimistic (not the
        # conservative) planning family; that is intentional, because the
        # pre-registered primary claim is the vs-reference one.
        'planned_family_size': 5,
        'alternative': 'greater',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 5,
        'planned_seeds': 11,
        # The TOST margin is HALF the superiority SESOI and it belongs to a
        # wider family: run_quality applies Holm to every operator pair in a
        # cell (21 pairs for the default seven operators), not just to the
        # five vs-softmax contrasts. The confirmatory budget above was sized
        # for superiority, so equivalence is explicitly a secondary
        # (descriptive) endpoint unless a dedicated equivalence-powered run
        # is registered. See docs/experiment-design.md.
        'equivalence': {
            'margin': 0.05,
            'family_size': 21,
            'alpha': DEFAULT_ALPHA,
            'required_for_claim': False,
            'source': ('run_quality.PAIRWISE_EQUIVALENCE_MARGIN, overridable '
                       'only at run start by MQAR_EQUIV_MARGIN; the TOST '
                       'family is one Holm correction over all operator '
                       'pairs in a cell (21 for the default seven ops)'),
            'note': ('the TOST margin is a stricter threshold than the 0.10 '
                     'superiority SESOI; the 11-seed budget cannot support '
                     'an equivalence claim at this margin (the planning '
                     'report states the required n)'),
        },
        'analysis': (
            'one-sided seed-paired t-test for the vs-reference contrast, '
            'Bonferroni/Holm over the planned family; exact sign and '
            'sign-flip permutation tests as robustness checks; TOST at the '
            'separately pre-specified equivalence margin (0.05), which is a '
            'secondary endpoint and is not powered at the 11-seed budget'),
        'decision_rule': (
            'a superiority claim requires beats_tuned_baseline=True; a '
            'non-significant contrast is inconclusive, never evidence of '
            'equivalence, and only the TOST can support practical '
            'equivalence'),
        'stopping_rule': (
            'fixed seed count fixed before the run; no optional stopping, '
            'no peeking, no replacing failed/diverged seeds to reach '
            'significance'),
        'pilot': {
            'group_key': 'op',
            'reference': 'softmax',
            'cell_keys': ['task', 'n_kv'],
            # The memoryless linear control is not a treatment; its variance
            # should not set the treatment seed budget.
            'exclude': ['linear'],
        },
    },
    'exp5_ratio': {
        'plan_id': 'exp5_ratio',
        'title': 'Exp 5 - KDA:CSA:HCA ratio sweep',
        'runner': 'run_ablation.py (ABL_MODE=ratio)',
        'result_files': ['results/exp5_ablation.json',
                         'results/exp5_ablation_provenance.json'],
        'question': (
            'Is an alternative KDA:CSA:HCA allocation different from the '
            'default 3:1:1 at fixed depth and fixed training budget?'),
        'design': ('seed-blocked two-sided allocation screen against the '
                   '3:1:1 reference; layouts are not ordered'),
        'unit': 'one trained 5-layer layout at one (ratio, n_kv, seed)',
        'factors': {
            'ratio': {
                'role': 'treatment',
                'levels': ['3:1:1', '4:1:0', '4:0:1', '2:2:1', '2:1:2',
                           '1:3:1', '1:1:3'],
            },
            'n_kv': {'role': 'treatment', 'levels': [1, 2, 4],
                     'note': 'default ABL_NKV=1'},
            'seed': {'role': 'block', 'levels': '42 .. 42 + n_seeds - 1'},
        },
        'required_levels': {'ratio': ['3:1:1']},
        'primary_contrast': (
            'each alternative ratio minus the 3:1:1 reference, paired by '
            'seed within n_kv'),
        'endpoint': 'final_acc',
        'direction': 'different_from_reference',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        'planned_family_size': 6,
        'alternative': 'two-sided',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 7,
        'planned_seeds': 14,
        'equivalence': {
            'margin': 0.05,
            'family_size': 6,
            'alpha': DEFAULT_ALPHA,
            'required_for_claim': False,
            'source': ('run_ablation.CONTRAST_EQUIVALENCE_MARGIN, overridable '
                       'only at run start by ABL_EQUIV_MARGIN'),
            'note': ('the ratio sweep reports a TOST verdict per alternative '
                     'ratio, but its 14-seed budget is powered for the 0.10 '
                     'superiority SESOI, not equivalence at 0.05'),
        },
        'analysis': (
            'two-sided seed-paired t-test against the reference under one '
            'Bonferroni family; TOST at the separately pre-specified '
            'equivalence margin (0.05) as a secondary endpoint; no raw '
            'mean ranking'),
        'decision_rule': (
            'only corrected-significant rows may be called different from '
            '3:1:1; non-significance is inconclusive at this seed budget'),
        'stopping_rule': (
            'fixed seed count and fixed ratio set fixed before the run; no '
            'optional stopping and no adding ratios after seeing the table'),
        'pilot': {
            'group_key': 'ratio',
            'reference': '3:1:1',
            'cell_keys': ['n_kv'],
        },
    },
    'exp5b_matrix': {
        'plan_id': 'exp5b_matrix',
        'title': 'Exp 5b - eight-group layout x memory-route matrix',
        'runner': 'run_ablation.py (ABL_MODE=matrix)',
        'result_files': ['results/exp5b_ablation_matrix.json',
                         'results/exp5b_ablation_matrix_contrasts.json',
                         'results/exp5b_ablation_matrix_provenance.json'],
        'question': (
            'Do CSA and HCA improve MQAR when they replace recurrent slots, '
            'and does the layout conclusion transfer across the KDA and '
            'Mamba-2 memory routes?'),
        'design': ('complete layout factorial (A/B/C/D) per memory route at '
                   'fixed depth, plus cross-route replacement contrasts and '
                   'the pre-registered layout x route interaction; seeds are '
                   'complete blocks'),
        'unit': ('one trained 10-layer matrix group at one (memory, layout, '
                 'n_kv, seed)'),
        'factors': {
            'memory': {'role': 'treatment',
                       'levels': ['kda', 'mamba2', 'rwkv7']},
            'layout_unit': {'role': 'treatment',
                            'levels': ['A', 'B', 'C', 'D'],
                            'note': ('main8 preset; csa_csa adds E and the '
                                     'position/budget presets answer '
                                     'separate questions')},
            'n_kv': {'role': 'treatment', 'levels': [1, 2, 4],
                     'note': 'default ABL_NKV=1'},
            'seed': {'role': 'block', 'levels': '42 .. 42 + n_seeds - 1'},
        },
        'required_levels': {},
        'primary_contrast': (
            'B-A, C-A, D-B, D-C and the I=D-B-C+A interaction within each '
            'memory route, the A/D cross-route replacement contrasts, and '
            'the route x layout transfer interaction '
            'I_route(D) = (D-A)_route1 - (D-A)_route2'),
        'endpoint': 'final_acc',
        'direction': 'higher_is_better',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        # 15 within-route layout contrasts (5 x 3 routes), 6 cross-route
        # A/D replacement contrasts for 3 route pairs, and 3 route x layout
        # transfer interactions. Planning with the full family is
        # conservative and keeps the two-sided tests honest.
        'planned_family_size': 24,
        'alternative': 'greater',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 7,
        'planned_seeds': 14,
        'equivalence': {
            'margin': 0.05,
            'family_size': 24,
            'alpha': DEFAULT_ALPHA,
            'required_for_claim': False,
            'source': ('run_ablation.CONTRAST_EQUIVALENCE_MARGIN, overridable '
                       'only at run start by ABL_EQUIV_MARGIN'),
            'note': ('the TOST is applied to every contrast row under one '
                     'Bonferroni family; the 14-seed budget is sized for the '
                     '0.10 superiority SESOI, so equivalence at 0.05 is '
                     'reported but not powered'),
        },
        # The transfer question is an INTERACTION, so it needs its own
        # variance budget: I_route = (D-A)_r1 - (D-A)_r2 is a four-cell
        # contrast whose SD is sqrt(2) times the two-cell paired-difference
        # SD that ``assumed_sd`` describes. Planning it at the main-effect
        # SD would quietly overstate the transfer power. The interaction
        # is tested inside the SAME run-wide family (24), so the record
        # inherits that family rather than pretending it is 3 independent
        # tests. It is marked required_for_claim=False for the same reason
        # as the TOST endpoint: the registered 14-seed budget is sized for
        # the main effects and does not reach the interaction requirement.
        'interaction': {
            'estimand': ('route x layout transfer '
                         'I_route(D) = (D-A)_route1 - (D-A)_route2, one row '
                         'per memory-route pair (KDA/Mamba-2 included)'),
            'sesoi': 0.10,
            'alternative': 'two-sided',
            'sd_inflation': INTERACTION_SD_INFLATION,
            'required_for_claim': False,
            'source': ('run_ablation._matrix_contrast_report computes '
                       'I_route with multi_seed_stats.interaction_contrast '
                       'and tests it two-sided inside the same run-wide '
                       'Bonferroni family as the main contrasts'),
            'note': ('a 2x2 interaction has SD sqrt(2) x the paired '
                     'difference SD assumed for a main effect; the '
                     'interaction budget is therefore larger than the '
                     'main-effect budget and is reported separately'),
        },
        'analysis': (
            'seed-paired t-test under one Bonferroni family over every '
            'contrast row; one-sided for the pre-registered layout '
            'directions, two-sided for cross-route replacement and the '
            'route x layout interaction; TOST at the separately '
            'pre-specified equivalence margin (0.05) as a secondary '
            'endpoint. A non-significant interaction is inconclusive, not '
            'evidence that the benefit transfers'),
        'decision_rule': (
            'only significant_bonferroni=True rows may be cited; the '
            'within-route interaction row is a claim about complementarity '
            'and the I_route row is the transfer claim, not a license to '
            'rank raw means. Every two-cell contrast reports its parameter-'
            'count gap; outside the 5% tolerance it is an operator-bundle + '
            'capacity effect and cannot be quoted as a capacity-neutral '
            'mechanism result'),
        'stopping_rule': (
            'fixed group set, depth, recipe and seed count fixed before the '
            'run; no optional stopping and no survivor-only matrix'),
        'pilot': {
            'group_key': 'group',
            'reference': 'KDA-A',
            'cell_keys': ['n_kv'],
            'note': ('prefer the contrast side file when available: it '
                     'already carries sd_diff, so pilot_sd_from_rows uses '
                     'the reported paired-difference SD directly'),
        },
    },
    'exp8_mechanism': {
        'plan_id': 'exp8_mechanism',
        'title': 'Exp 8 - CSA mechanism ablation (RoPE / top-k / STE)',
        'runner': 'run_mechanism_ablation.py',
        'result_files': ['results/exp8_mechanism_ablation.json'],
        'question': (
            'Which CSA mechanisms - partial RoPE, sparse top-k selection, '
            'and indexer STE learning - change MQAR accuracy?'),
        'design': ('leave-one-factor-out mechanism ablation inside the '
                   'frozen 3:1:1 layout, including the pre-registered '
                   'RoPE x selection-density 2x2 interaction; all variants '
                   'share depth and parameter count, seeds are complete '
                   'blocks'),
        'unit': 'one trained variant at one (variant, n_kv, seed)',
        'factors': {
            'variant': {
                'role': 'treatment',
                'levels': ['baseline', 'no_rope', 'topk1', 'dense',
                           'frozen_indexer', 'no_rope_dense'],
            },
            'n_kv': {'role': 'treatment', 'levels': [1, 2, 4],
                     'note': 'default MECH_NKV=1'},
            'seed': {'role': 'block', 'levels': '42 .. 42 + n_seeds - 1'},
        },
        'required_levels': {'variant': ['baseline']},
        'primary_contrast': (
            'each mechanism variant minus the baseline, paired by seed, '
            'with the pre-registered sign in '
            'run_mechanism_ablation.VARIANTS, plus the pre-registered '
            '2x2 interaction I = no_rope_dense - no_rope - dense + '
            'baseline'),
        'endpoint': 'final_acc',
        'direction': 'pre_registered_sign',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        # 5 non-baseline main-effect contrasts + 1 factorial interaction.
        'planned_family_size': 6,
        'alternative': 'two-sided',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 5,
        'planned_seeds': 14,
        # The pre-registered RoPE x density interaction is a four-cell
        # contrast: its SD is sqrt(2) x the paired-difference SD used for
        # the main effects, and it is tested inside the shared family of 6.
        # The 14-seed main-effect budget is not the interaction budget, so
        # the interaction record is descriptive unless a dedicated run
        # reaches its own required n (see docs/experiment-design.md).
        'interaction': {
            'estimand': ('RoPE x selection-density interaction '
                         'I = no_rope_dense - no_rope - dense + baseline'),
            'sesoi': 0.10,
            'alternative': 'two-sided',
            'sd_inflation': INTERACTION_SD_INFLATION,
            'required_for_claim': False,
            'source': ('run_mechanism_ablation.mechanism_interaction_contrasts '
                       'via multi_seed_stats.interaction_contrast; one test '
                       'in the shared family of 6'),
            'note': ('the registered 14-seed budget is sized for the '
                     'main-effect SESOI; a 2x2 interaction needs a larger '
                     'seed block at the same SESOI'),
        },
        'analysis': (
            'two-sided seed-paired t-test vs baseline, Bonferroni-corrected '
            'over the compared variants and the pre-registered interaction; '
            'the pre-registered sign decides supported vs contradicted; '
            'exact tests as robustness checks. The interaction is a '
            'difference-in-differences over the four 2x2 cells, not a '
            'comparison of the four baseline contrasts'),
        'decision_rule': (
            'a prediction is supported only by a corrected significant '
            'difference in the pre-registered direction; non-significance '
            'is not_supported (inconclusive), never a confirmed null. A '
            'non-significant interaction is inconclusive, never evidence of '
            'additivity'),
        'stopping_rule': (
            'fixed variant set and seed count fixed before the run; the '
            'audit refuses single-factor claims when the one-factor '
            'counterpart is missing or when the built variant config changes '
            'a switch outside its declared factor'),
        'pilot': {
            'group_key': 'variant',
            'reference': 'baseline',
            'cell_keys': ['n_kv'],
        },
    },
}

PLAN_IDS = tuple(PLANS)


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _need_number(value, name, *, positive=False, allow_zero=True):
    if not _is_number(value):
        raise DesignError(f'{name} must be a finite number, got {value!r}')
    value = float(value)
    if positive and value <= 0.0:
        raise DesignError(f'{name} must be > 0, got {value!r}')
    if not allow_zero and value == 0.0:
        raise DesignError(f'{name} must be non-zero, got {value!r}')
    return value


def _seed_list(n_or_seeds, *, name='seeds'):
    """Normalize an int seed COUNT or an explicit seed sequence to a list."""
    if _is_int(n_or_seeds):
        if n_or_seeds < 0:
            raise DesignError(f'{name} count must be >= 0, got {n_or_seeds!r}')
        return [DEFAULT_SEED_BASE + i for i in range(n_or_seeds)]
    if isinstance(n_or_seeds, (str, bytes)) or not hasattr(n_or_seeds,
                                                           '__iter__'):
        raise DesignError(f'{name} must be an int count or an iterable of '
                          f'ints, got {n_or_seeds!r}')
    return list(n_or_seeds)


def _cell_key(row, keys):
    return tuple(row.get(k) for k in keys)


def _rows_from_payload(payload):
    """Accept an envelope, a bare row list, or a path to either."""
    if isinstance(payload, str):
        with open(payload, encoding='utf-8') as fh:
            payload = json.load(fh)
    if isinstance(payload, dict):
        rows = payload.get('results')
        if not isinstance(rows, list):
            raise DesignError('result payload has no "results" list')
        return rows
    if isinstance(payload, list):
        return payload
    raise DesignError('result payload must be a list or an envelope object')


def _clean_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _is_hashable(value) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Plan access and static validation
# ---------------------------------------------------------------------------

def get_plan(plan_id: str) -> dict:
    """Return a deep copy of a registered plan (callers cannot mutate it)."""
    if plan_id not in PLANS:
        raise DesignError(f'unknown experiment plan {plan_id!r}; valid ids: '
                          f'{list(PLANS)}')
    return copy.deepcopy(PLANS[plan_id])


#: Scope string stored next to every plan fingerprint. It documents exactly
#: what the digest covers so a reader does not have to guess the canonical
#: form. Changing the scope is a breaking change: bump the version suffix.
PLAN_FINGERPRINT_SCOPE = (
    'sha256 of the canonical JSON (sorted keys, compact separators, UTF-8, '
    'strict numbers) of one pre-registered plan dictionary')


def plan_sha256(plan) -> str:
    """Stable SHA-256 fingerprint of a pre-registered plan definition.

    ``plan`` may be a plan id or the plan dict itself. The digest is
    canonical (sorted keys, compact separators, strict JSON), so key order
    and pretty-printing do not change it, while ANY semantic change to the
    plan -- SESOI, assumed SD, seed budget, required controls, analysis or
    decision rule -- does.

    The fingerprint is embedded in every runtime audit and in the
    prospective design manifest. It turns "the plan was pre-registered" into
    a checkable claim: a result recorded under a different plan definition
    can be detected after the fact, even though the registry is code and can
    be edited. Use :func:`reproducibility.compare_design_binding` to check a
    result file against the current registry.

    Raises :class:`DesignError` for an unknown plan id or a plan that is not
    a JSON object with strict (finite, JSON-native) values.
    """
    if isinstance(plan, str):
        plan = get_plan(plan)
    if not isinstance(plan, dict):
        raise DesignError(
            f'plan must be a plan id or a dict, got {type(plan).__name__}')
    try:
        canonical = json.dumps(plan, sort_keys=True, separators=(',', ':'),
                               allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DesignError(f'plan is not strict JSON: {exc}') from None
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_plan(plan) -> list[str]:
    """Return a list of design errors for one plan (empty when valid)."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return [f'plan must be a dict, got {type(plan).__name__}']
    for key in ('plan_id', 'title', 'runner', 'question', 'design', 'unit',
                'endpoint', 'analysis', 'decision_rule', 'stopping_rule'):
        if not _is_nonempty_str(plan.get(key)):
            errors.append(f'{key} must be a non-empty string')
    if not isinstance(plan.get('result_files'), list) \
            or not all(_is_nonempty_str(p) for p in plan.get('result_files')):
        errors.append('result_files must be a list of non-empty paths')
    factors = plan.get('factors')
    if not isinstance(factors, dict) or not factors:
        errors.append('factors must be a non-empty dict')
        factors = {}
    for name, factor in factors.items():
        if not _is_nonempty_str(name):
            errors.append('factor names must be non-empty strings')
            continue
        if not isinstance(factor, dict):
            errors.append(f'factor {name!r} must be a dict')
            continue
        if not _is_nonempty_str(factor.get('role')):
            errors.append(f'factor {name!r} needs a non-empty role')
        levels = factor.get('levels')
        if isinstance(levels, str) and levels.strip():
            # A block factor may describe its levels as a formula (e.g. the
            # shared seed range) instead of enumerating them.
            continue
        if not isinstance(levels, list) or not levels:
            errors.append(f'factor {name!r} needs a non-empty levels list')
            continue
        if len({repr(level) for level in levels}) != len(levels):
            errors.append(f'factor {name!r} has duplicate levels')
    required = plan.get('required_levels')
    if not isinstance(required, dict):
        errors.append('required_levels must be a dict')
        required = {}
    for name, levels in required.items():
        if name not in factors:
            errors.append(f'required_levels names unknown factor {name!r}')
            continue
        if not isinstance(levels, list) or not levels:
            errors.append(f'required_levels[{name!r}] must be a non-empty '
                          'list')
            continue
        factor = factors[name]
        if not isinstance(factor, dict):
            # Already reported above as "factor ... must be a dict"; do not
            # crash on the follow-up lookup.
            errors.append(f'required_levels names factor {name!r} that is '
                          'not a dict')
            continue
        factor_levels = factor.get('levels', [])
        if not isinstance(factor_levels, list):
            errors.append(f'required_levels names a factor {name!r} whose '
                          'levels are not an explicit list')
            continue
        known = {repr(level) for level in factor_levels}
        missing = [level for level in levels if repr(level) not in known]
        if missing:
            errors.append(f'required_levels[{name!r}] contains levels not in '
                          f'the factor: {missing!r}')
    for key in ('sesoi', 'assumed_sd'):
        if not _is_number(plan.get(key)) or float(plan[key]) <= 0.0:
            errors.append(f'{key} must be a finite number > 0')
    alpha = plan.get('alpha')
    if not _is_number(alpha) or not (0.0 < float(alpha) < 1.0):
        errors.append('alpha must be in (0,1)')
    if plan.get('alternative') not in ('two-sided', 'greater', 'less'):
        errors.append("alternative must be 'two-sided'|'greater'|'less'")
    target = plan.get('target_power')
    if not _is_number(target) or not (0.0 < float(target) < 1.0):
        errors.append('target_power must be in (0,1)')
    family = plan.get('planned_family_size')
    if not _is_int(family) or family < 1:
        errors.append('planned_family_size must be an int >= 1')
    for key in ('screening_seeds', 'planned_seeds'):
        if not _is_int(plan.get(key)) or plan[key] < 2:
            errors.append(f'{key} must be an int >= 2')
    if isinstance(plan.get('screening_seeds'), int) \
            and isinstance(plan.get('planned_seeds'), int) \
            and plan['screening_seeds'] >= plan['planned_seeds']:
        errors.append('screening_seeds must be below the confirmatory '
                      'planned_seeds (otherwise the labels are meaningless)')
    pilot = plan.get('pilot')
    if not isinstance(pilot, dict):
        errors.append('pilot must be a dict')
    else:
        for key in ('group_key', 'reference'):
            if not _is_nonempty_str(pilot.get(key)):
                errors.append(f'pilot.{key} must be a non-empty string')
        cell_keys = pilot.get('cell_keys')
        if not isinstance(cell_keys, list) or not cell_keys \
                or not all(_is_nonempty_str(k) for k in cell_keys):
            errors.append('pilot.cell_keys must be a non-empty list of names')
        exclude = pilot.get('exclude', [])
        if not isinstance(exclude, list) \
                or not all(_is_nonempty_str(k) for k in exclude):
            errors.append('pilot.exclude must be a list of names')
    # The equivalence (TOST) endpoint has its own margin, family, alpha and
    # power requirement. It is optional (some mechanism runs make no
    # equivalence claim), but when present it must be internally coherent and
    # it may not be looser than the superiority SESOI.
    equiv = plan.get('equivalence')
    equiv_required = False
    if equiv is not None:
        if not isinstance(equiv, dict):
            errors.append('equivalence must be a dict when present')
        else:
            eq_margin = equiv.get('margin')
            if not _is_number(eq_margin) or float(eq_margin) <= 0.0:
                errors.append('equivalence.margin must be a finite number > 0')
            elif _is_number(plan.get('sesoi')) \
                    and float(eq_margin) > float(plan['sesoi']) + 1e-12:
                errors.append(
                    'equivalence.margin must not exceed the superiority '
                    'sesoi: declaring equivalence at a margin wider than the '
                    'smallest effect of interest would call a practically '
                    'relevant difference "equivalent"')
            eq_family = equiv.get('family_size', family)
            if not _is_int(eq_family) or eq_family < 1:
                errors.append('equivalence.family_size must be an int >= 1')
            eq_alpha = equiv.get('alpha', alpha)
            if not _is_number(eq_alpha) or not (0.0 < float(eq_alpha) < 1.0):
                errors.append('equivalence.alpha must be in (0,1)')
            elif _is_int(eq_family) and eq_family >= 1 \
                    and float(eq_alpha) / eq_family >= 0.5:
                errors.append('equivalence.alpha / equivalence.family_size '
                              'must be below 0.5')
            eq_target = equiv.get('target_power', target)
            if not _is_number(eq_target) or not (0.0 < float(eq_target) < 1.0):
                errors.append('equivalence.target_power must be in (0,1)')
            flag = equiv.get('required_for_claim', False)
            if not isinstance(flag, bool):
                errors.append('equivalence.required_for_claim must be a bool')
            else:
                equiv_required = flag
    # The interaction endpoint has its own SESOI, family, alternative and
    # variance inflation. It is optional (only plans that pre-register a
    # 2x2 difference-in-differences need it), but when present it must be
    # internally coherent and it may not silently reuse a looser test.
    inter = plan.get('interaction')
    interaction_required = False
    if inter is not None:
        if not isinstance(inter, dict):
            errors.append('interaction must be a dict when present')
        else:
            i_sesoi = inter.get('sesoi', plan.get('sesoi'))
            if not _is_number(i_sesoi) or float(i_sesoi) <= 0.0:
                errors.append('interaction.sesoi must be a finite number > 0')
            i_alt = inter.get('alternative', 'two-sided')
            if i_alt not in ('two-sided', 'greater', 'less'):
                errors.append(
                    "interaction.alternative must be "
                    "'two-sided'|'greater'|'less'")
            i_family = inter.get('family_size', family)
            if not _is_int(i_family) or i_family < 1:
                errors.append('interaction.family_size must be an int >= 1')
            i_alpha = inter.get('alpha', alpha)
            if not _is_number(i_alpha) or not (0.0 < float(i_alpha) < 1.0):
                errors.append('interaction.alpha must be in (0,1)')
            elif _is_int(i_family) and i_family >= 1 \
                    and float(i_alpha) / i_family >= 0.5:
                errors.append(
                    'interaction.alpha / interaction.family_size must be '
                    'below 0.5')
            i_target = inter.get('target_power', target)
            if not _is_number(i_target) or not (0.0 < float(i_target) < 1.0):
                errors.append('interaction.target_power must be in (0,1)')
            i_inflation = inter.get('sd_inflation', INTERACTION_SD_INFLATION)
            if not _is_number(i_inflation) or float(i_inflation) <= 0.0:
                errors.append(
                    'interaction.sd_inflation must be a finite number > 0')
            elif float(i_inflation) < 1.0:
                errors.append(
                    'interaction.sd_inflation must be >= 1.0: a four-cell '
                    'interaction cannot be less variable than the two-cell '
                    'contrast it is built from')
            flag = inter.get('required_for_claim', False)
            if not isinstance(flag, bool):
                errors.append('interaction.required_for_claim must be a bool')
            else:
                interaction_required = flag
    if errors:
        return errors
    # A plan whose declared confirmatory budget does not reach its own
    # required power is stale: fix the registry or the assumptions.
    try:
        rep = planning_report(plan)
    except DesignError as exc:
        return [f'planning_report failed: {exc}']
    required = rep.get('required_seeds')
    if required is None:
        errors.append('required_seeds is undefined for this plan')
    elif plan['planned_seeds'] < required:
        errors.append(
            f"planned_seeds={plan['planned_seeds']} is below the "
            f"prospective required_seeds={required} at the declared SESOI / "
            f"SD / corrected alpha; raise planned_seeds or revise the "
            f"assumptions explicitly")
    if rep.get('planned_power') is None \
            or rep['planned_power'] < float(plan['target_power']) - 1e-9:
        errors.append(
            f"planned_power={rep.get('planned_power')!r} does not reach "
            f"target_power={plan['target_power']!r}")
    exact = rep.get('exact_test_resolution') or {}
    if exact and not exact.get('family_testable', True):
        errors.append(
            'the planned budget has no exact-test rejection region: '
            f"n={plan.get('planned_seeds')!r}, "
            f"family_size={plan.get('planned_family_size')!r}, "
            f"min attainable Holm p="
            f"{exact.get('min_attainable_p_holm')!r} > alpha; raise "
            f"planned_seeds to at least "
            f"{exact.get('min_effective_pairs_holm')!r} effective pairs or "
            'shrink the confirmatory family')
    if equiv_required:
        eq = rep.get('equivalence') or {}
        eq_required_n = eq.get('required_seeds_at_zero')
        if not eq.get('supports_equivalence_at_planned_n'):
            errors.append(
                'equivalence.required_for_claim is set, but the planned '
                f"seeds={plan.get('planned_seeds')!r} do not reach the TOST "
                f"requirement required_seeds_at_zero={eq_required_n!r} at "
                f"margin={eq.get('margin')!r} and the corrected per-test "
                f"alpha={eq.get('alpha_per_test')!r}; raise planned_seeds, "
                'widen the pre-specified margin, or record '
                'required_for_claim=False and treat equivalence as '
                'descriptive')
    if interaction_required:
        ip = rep.get('interaction') or {}
        if not ip.get('supports_interaction_at_planned_n'):
            errors.append(
                'interaction.required_for_claim is set, but the planned '
                f"seeds={plan.get('planned_seeds')!r} do not reach the "
                f"interaction requirement "
                f"required_seeds={ip.get('required_seeds')!r} at "
                f"sesoi={ip.get('sesoi')!r}, sd_inflation="
                f"{ip.get('sd_inflation')!r} and the corrected per-test "
                f"alpha={ip.get('alpha_per_test')!r}; raise planned_seeds, "
                'reduce the interaction SESOI, or record '
                'required_for_claim=False and treat the interaction as '
                'descriptive')
    return errors


def validate_plans() -> list[str]:
    """Validate every registered plan; errors are prefixed with the id."""
    errors: list[str] = []
    for plan_id, plan in PLANS.items():
        if plan.get('plan_id') != plan_id:
            errors.append(f'{plan_id}: plan_id does not match the registry key')
        errors.extend(f'{plan_id}: {e}' for e in validate_plan(plan))
    return errors


# ---------------------------------------------------------------------------
# Prospective planning
# ---------------------------------------------------------------------------

_NORMAL = statistics.NormalDist()


def equivalence_seeds_required(margin, sd, *, alpha=DEFAULT_ALPHA,
                               family_size=1,
                               power=DEFAULT_TARGET_POWER):
    """Seed pairs needed to DECLARE equivalence at ``margin``.

    This is the TOST companion of :func:`planning_report`, not a
    superiority calculation. Two one-sided tests at the Bonferroni-corrected
    tail ``alpha / family_size`` are run against ``+-margin``. Under the
    normal approximation at a true difference of zero, the probability of
    declaring equivalence is ``2*Phi(margin/se - z) - 1`` with
    ``se = sd/sqrt(n)``. Requiring target power ``1-beta`` gives

    ``n = ceil(((z_{1-alpha/family} + z_{1-beta/2}) * sd / margin)^2)``.

    The ``z_{1-beta/2}`` (not ``z_{1-beta}``) is the TOST-specific term: a
    two-sided interval must clear the margin on BOTH sides. Equivalence
    therefore needs substantially more seeds than superiority at the same
    standardized effect, and a budget planned only for the SESOI is
    normally not powered to declare equivalence.

    This is a planning approximation, like :func:`planning_report`; it is
    not used for inference. It uses the normal critical value rather than
    the t critical value, so it is slightly optimistic (a lower bound on the
    seeds needed). The margin must be positive and finite.
    """
    margin = _need_number(margin, 'equivalence margin', positive=True)
    sd_value = _need_number(sd, 'sd', positive=True)
    alpha_value = _need_number(alpha, 'alpha')
    if not (0.0 < alpha_value < 1.0):
        raise DesignError(f'alpha must be in (0,1), got {alpha_value!r}')
    if not _is_int(family_size) or family_size < 1:
        raise DesignError(
            f'equivalence family_size must be an int >= 1, got '
            f'{family_size!r}')
    if alpha_value / family_size >= 0.5:
        raise DesignError(
            'the per-test equivalence alpha must be below 0.5 so a '
            f'(1-2*alpha) interval exists, got {alpha_value / family_size!r}')
    power_value = _need_number(power, 'power', positive=True)
    if not (0.0 < power_value < 1.0):
        raise DesignError(f'power must be in (0,1), got {power_value!r}')
    z_crit = _NORMAL.inv_cdf(1.0 - alpha_value / family_size)
    # At a true difference of zero the TOST power is
    # 2*Phi(m/se - z) - 1, so target power needs Phi(...) = 1 - beta/2.
    z_power = _NORMAL.inv_cdf(0.5 * (1.0 + power_value))
    try:
        required = ((z_crit + z_power) * sd_value / margin) ** 2
    except OverflowError:
        return None
    if not math.isfinite(required):
        return None
    # A paired design needs at least two seed pairs.
    return max(2, int(math.ceil(required)))


def equivalence_power(n, margin, sd, *, alpha=DEFAULT_ALPHA,
                      family_size=1, true_diff=0.0):
    """Approximate probability that TOST declares equivalence at ``n``.

    ``true_diff`` is the hypothesized true mean paired difference. The
    approximate power is

    ``Phi((margin + delta)/se - z) + Phi((margin - delta)/se - z) - 1``,

    with ``se = sd/sqrt(n)`` and ``z = z_{1-alpha/family_size}``. At
    ``delta = 0`` this is the usual ``2*Phi(margin/se - z) - 1``. The
    approximation ignores the correlation between the two one-sided
    statistics (it is the standard planning shortcut) and is not used for
    inference.

    Returns ``None`` when fewer than two seed pairs make the sample SD
    undefined. A true difference on or beyond the margin has power at or
    below the per-test alpha: no finite seed budget turns a real
    difference into equivalence.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise DesignError(f'n must be a non-negative int, got {n!r}')
    margin = _need_number(margin, 'equivalence margin', positive=True)
    sd_value = _need_number(sd, 'sd')
    if sd_value < 0.0:
        raise DesignError(f'sd must be >= 0, got {sd_value!r}')
    alpha_value = _need_number(alpha, 'alpha')
    if not (0.0 < alpha_value < 1.0):
        raise DesignError(f'alpha must be in (0,1), got {alpha_value!r}')
    if not _is_int(family_size) or family_size < 1:
        raise DesignError(
            f'equivalence family_size must be an int >= 1, got '
            f'{family_size!r}')
    if alpha_value / family_size >= 0.5:
        raise DesignError(
            'the per-test equivalence alpha must be below 0.5 so a '
            f'(1-2*alpha) interval exists, got {alpha_value / family_size!r}')
    delta = _need_number(true_diff, 'true_diff')
    if n < 2:
        return None
    if sd_value == 0.0:
        # A noiseless difference vector declares equivalence iff it lies
        # strictly inside the margin (the TOST p is 0 / 0.5 / 1 inside /
        # boundary / outside).
        return 1.0 if abs(delta) < margin else 0.0
    tail_alpha = alpha_value / family_size
    z_crit = _NORMAL.inv_cdf(1.0 - tail_alpha)
    se = sd_value / math.sqrt(n)
    lower = _NORMAL.cdf((delta + margin) / se - z_crit)
    upper = _NORMAL.cdf((margin - delta) / se - z_crit)
    return min(1.0, max(0.0, float(lower + upper - 1.0)))


def equivalence_plan(*, margin, sd, seeds, alpha=DEFAULT_ALPHA,
                     family_size=1, target_power=DEFAULT_TARGET_POWER,
                     required_for_claim=False, source=None, note=None) -> dict:
    """Prospective read-out for a TOST equivalence endpoint.

    The superiority budget in :func:`planning_report` answers a different
    question from the TOST budget computed here. Reporting both, side by
    side, prevents a run powered for a 0.10 accuracy-point superiority
    difference from being read as a confirmatory equivalence result at a
    tighter 0.05 margin.
    """
    margin = _need_number(margin, 'equivalence margin', positive=True)
    sd_value = _need_number(sd, 'sd', positive=True)
    if not _is_int(seeds) or seeds < 0:
        raise DesignError(f'seeds must be a non-negative int, got {seeds!r}')
    alpha_value = _need_number(alpha, 'alpha')
    if not (0.0 < alpha_value < 1.0):
        raise DesignError(f'alpha must be in (0,1), got {alpha_value!r}')
    if not _is_int(family_size) or family_size < 1:
        raise DesignError(
            f'equivalence family_size must be an int >= 1, got '
            f'{family_size!r}')
    target = _need_number(target_power, 'target_power', positive=True)
    if not (0.0 < target < 1.0):
        raise DesignError(f'target_power must be in (0,1), got {target!r}')
    if not isinstance(required_for_claim, bool):
        raise DesignError(
            'required_for_claim must be a bool, got '
            f'{required_for_claim!r}')
    required = equivalence_seeds_required(
        margin, sd_value, alpha=alpha_value, family_size=family_size,
        power=target)
    power_zero = equivalence_power(
        seeds, margin, sd_value, alpha=alpha_value,
        family_size=family_size, true_diff=0.0)
    power_half = equivalence_power(
        seeds, margin, sd_value, alpha=alpha_value,
        family_size=family_size, true_diff=margin / 2.0)
    power_edge = equivalence_power(
        seeds, margin, sd_value, alpha=alpha_value,
        family_size=family_size, true_diff=margin)
    supported = bool(
        required is not None and seeds >= required
        and power_zero is not None and power_zero >= target - 1e-9)
    if seeds < 2:
        status = 'not_estimable'
    elif supported:
        status = 'equivalence_supported'
    else:
        status = 'equivalence_underpowered'
    # Smallest margin (in endpoint units) at which THIS n reaches the
    # target power at a true difference of zero. If it is wider than the
    # registered margin, the run cannot declare equivalence at that margin.
    minimum_margin = None
    if seeds >= 2:
        z_crit = _NORMAL.inv_cdf(1.0 - alpha_value / family_size)
        z_power = _NORMAL.inv_cdf(0.5 * (1.0 + target))
        minimum_margin = (z_crit + z_power) * sd_value / math.sqrt(seeds)
        if not math.isfinite(minimum_margin):
            minimum_margin = None
    shortfall = (max(0, int(required) - int(seeds))
                 if required is not None else None)
    return {
        'margin': margin,
        'margin_source': source,
        'note': note,
        'alpha_family_wise': alpha_value,
        'family_size': family_size,
        'alpha_per_test': alpha_value / family_size,
        'ci_level': 1.0 - 2.0 * alpha_value / family_size,
        'target_power': target,
        'seeds': int(seeds),
        'required_seeds_at_zero': required,
        'shortfall_seeds': shortfall,
        'planned_power_at_zero': power_zero,
        'planned_power_at_half_margin': power_half,
        'planned_power_at_margin_edge': power_edge,
        'minimum_supportable_margin_at_n': minimum_margin,
        'supports_equivalence_at_planned_n': supported,
        'required_for_claim': bool(required_for_claim),
        'claim_allowed': supported,
        'status': status,
        'method': (
            'TOST normal approximation for planning only: '
            'n = ceil(((z_{1-alpha/family} + z_{1-beta/2}) * sd / '
            'margin)^2) at a true difference of zero; inference uses '
            'multi_seed_stats.tost_paired / tost_from_summary'),
        'caveat': (
            'the equivalence margin is a separate scientific threshold '
            'from the superiority SESOI. A run powered to detect the SESOI '
            'is not automatically powered to declare equivalence; only '
            'claim_allowed=True may support a "practically the same" '
            'conclusion'),
    }


def interaction_planning(*, sesoi, sd, seeds, alpha=DEFAULT_ALPHA,
                         family_size=1, alternative='two-sided',
                         target_power=DEFAULT_TARGET_POWER,
                         sd_inflation=INTERACTION_SD_INFLATION,
                         required_for_claim=False, estimand=None,
                         source=None, note=None) -> dict:
    """Prospective read-out for a pre-registered 2x2 interaction endpoint.

    A difference-in-differences ``I = d - b - c + a`` is a four-cell
    contrast. ``sd`` is the paired-difference SD used for a two-cell main
    effect; the interaction SD is ``sd * sd_inflation`` because
    ``Var(I) = 4*sigma^2`` while ``Var(B-A) = 2*sigma^2`` under the
    shared-seed / independent-cell-noise model. ``sd_inflation`` is
    exposed so a plan can replace the default with a value calibrated from
    a dedicated interaction pilot, but it may not be below 1.0: a
    four-cell contrast cannot be less variable than a two-cell one without
    a negative cross-cell correlation, which the design does not assume.

    ``family_size`` is the multiplicity family the interaction is tested
    in (the same run-wide family as the main contrasts for Exp 5b/8), not
    the number of interaction rows. ``required_for_claim`` mirrors the TOST
    endpoint: when true the registry must actually reach this budget.
    """
    sesoi = _need_number(sesoi, 'interaction sesoi', positive=True)
    sd_value = _need_number(sd, 'sd', positive=True)
    inflation = _need_number(sd_inflation, 'interaction sd_inflation',
                             positive=True)
    if inflation < 1.0:
        raise DesignError(
            'interaction sd_inflation must be >= 1.0: a four-cell '
            'difference-in-differences cannot have a smaller SD than the '
            f'two-cell contrast it is built from, got {inflation!r}')
    if not _is_int(seeds) or seeds < 0:
        raise DesignError(f'seeds must be a non-negative int, got {seeds!r}')
    alpha_value = _need_number(alpha, 'alpha')
    if not (0.0 < alpha_value < 1.0):
        raise DesignError(f'alpha must be in (0,1), got {alpha_value!r}')
    if not _is_int(family_size) or family_size < 1:
        raise DesignError(
            f'interaction family_size must be an int >= 1, got '
            f'{family_size!r}')
    if alternative not in ('two-sided', 'greater', 'less'):
        raise DesignError(
            "interaction alternative must be "
            f"'two-sided'|'greater'|'less', got {alternative!r}")
    target = _need_number(target_power, 'target_power', positive=True)
    if not (0.0 < target < 1.0):
        raise DesignError(f'target_power must be in (0,1), got {target!r}')
    if not isinstance(required_for_claim, bool):
        raise DesignError(
            f'required_for_claim must be a bool, got {required_for_claim!r}')
    sd_interaction = sd_value * inflation
    if not math.isfinite(sd_interaction):
        raise DesignError(
            'sd * interaction sd_inflation is not a finite float')
    # ``alpha_test`` is the one-sided TAIL probability the runner's
    # corrected threshold uses; multi_seed_stats takes a TEST level and
    # halves it internally, so undo that for the symmetric alternatives.
    alpha_test = (alpha_value / family_size
                  if alternative in ('greater', 'less')
                  else alpha_value / (2.0 * family_size))
    planning_alpha = (alpha_test if alternative in ('greater', 'less')
                      else 2.0 * alpha_test)
    dz = sesoi / sd_interaction
    required = seeds_required(
        dz, alpha=planning_alpha, power=target, alternative=alternative)
    if required is not None:
        required = max(2, int(required))
    mde = min_detectable_effect(
        seeds, sd_interaction, alpha=planning_alpha, power=target,
        alternative=alternative)
    planning_effect = -sesoi if alternative == 'less' else sesoi
    achieved = power_at_effect(
        seeds, sd_interaction, planning_effect, alpha=planning_alpha,
        alternative=alternative)
    exact = exact_test_resolution(
        seeds, alpha=alpha_value, family_size=family_size,
        alternative=alternative)
    supported = bool(
        required is not None and seeds >= required
        and achieved is not None and achieved >= target - 1e-9
        and exact.get('family_testable', True))
    if seeds < 2 or required is None:
        status = 'not_estimable'
    elif supported:
        status = 'confirmatory_ready'
    else:
        status = 'underpowered_at_sesoi'
    exact_warning = None
    if not exact.get('family_testable', True):
        exact_warning = (
            'the exact sign-flip test has no rejection region for this '
            f'interaction at n={seeds} and family_size={family_size}: the '
            'smallest attainable Holm-adjusted p is '
            f"{exact.get('min_attainable_p_holm')!r} > alpha="
            f'{alpha_value:g}; the interaction is screening-only')
    shortfall = (max(0, int(required) - int(seeds))
                 if required is not None else None)
    return {
        'estimand': estimand,
        'sesoi': sesoi,
        'assumed_sd': sd_value,
        'sd_inflation': inflation,
        'sd_interaction': sd_interaction,
        'standardized_effect_dz': dz,
        'alpha_family_wise': alpha_value,
        'family_size': family_size,
        'alpha_per_test': alpha_test,
        'alternative': alternative,
        'target_power': target,
        'seeds': int(seeds),
        'required_seeds': required,
        'shortfall_seeds': shortfall,
        'planned_mde': mde,
        'planned_power': achieved,
        'power_gap': (None if achieved is None else target - achieved),
        'supports_interaction_at_planned_n': supported,
        'required_for_claim': bool(required_for_claim),
        'claim_allowed': bool(supported),
        'status': status,
        'exact_test_resolution': exact,
        'exact_test_warning': exact_warning,
        'source': source,
        'note': note,
        'method': (
            'normal approximation for planning only: '
            'n = ceil(((z_crit + z_power) * sd_interaction / SESOI)^2), '
            'with sd_interaction = sd * sqrt(2) under the shared-seed / '
            'independent-cell-noise model; inference uses the paired t / '
            'exact sign-flip tests on the per-seed difference-in-'
            'differences in multi_seed_stats.interaction_contrast'),
        'caveat': (
            'the sqrt(2) inflation is a model assumption, not a measured '
            'quantity: a shared seed effect cancels in I but independent '
            'cell noise does not. Positive cross-cell correlation makes it '
            'conservative; a dedicated four-cell pilot can replace it. The '
            'main-effect seed budget is not the interaction budget, and a '
            'non-significant interaction is inconclusive, never evidence '
            'of additivity'),
    }


def planning_report(plan, *, seeds=None, family_size=None, sd=None,
                    alpha=None, alternative=None, sesoi=None,
                    target_power=None) -> dict:
    """Compute the prospective seed budget and the power it delivers.

    All overrides are for "what if" planning and are validated; the
    defaults come from the plan. ``alpha`` is the FAMILY-WISE alpha. The
    per-test threshold used here is the one-sided TAIL probability used by
    the runners: ``alpha / family_size`` for a directional family and
    ``alpha / (2 * family_size)`` for a two-sided family.  The planning
    helpers in :mod:`multi_seed_stats` take a TEST level (and halve it
    internally for a two-sided test), so the two-sided case is passed
    ``2 * alpha_per_test``; otherwise the correction would be applied twice.

    The returned record is JSON-safe and includes the arithmetic inputs so a
    reader can audit the sample-size decision without importing this module.
    """
    plan = get_plan(plan) if isinstance(plan, str) else plan
    if not isinstance(plan, dict):
        raise DesignError('plan must be a plan id or a dict')
    alpha = _need_number(
        plan.get('alpha', DEFAULT_ALPHA) if alpha is None else alpha, 'alpha')
    if not (0.0 < alpha < 1.0):
        raise DesignError(f'alpha must be in (0,1), got {alpha!r}')
    family = plan.get('planned_family_size', 1) if family_size is None \
        else family_size
    if not _is_int(family) or family < 1:
        raise DesignError(f'family_size must be an int >= 1, got {family!r}')
    alt = plan.get('alternative', 'two-sided') if alternative is None \
        else alternative
    if alt not in ('two-sided', 'greater', 'less'):
        raise DesignError(f'invalid alternative {alt!r}')
    sesoi_value = _need_number(
        plan.get('sesoi', 0.10) if sesoi is None else sesoi, 'sesoi',
        positive=True)
    sd_value = _need_number(
        plan.get('assumed_sd', DEFAULT_ASSUMED_SD) if sd is None else sd,
        'sd', positive=True)
    target = _need_number(
        plan.get('target_power', DEFAULT_TARGET_POWER)
        if target_power is None else target_power, 'target_power',
        positive=True)
    if not (0.0 < target < 1.0):
        raise DesignError(f'target_power must be in (0,1), got {target!r}')
    n_seeds = plan.get('planned_seeds', 2) if seeds is None else seeds
    if not _is_int(n_seeds) or n_seeds < 0:
        raise DesignError(f'seeds must be a non-negative int, got {n_seeds!r}')
    alpha_test = (alpha / family if alt in ('greater', 'less')
                  else alpha / (2.0 * family))
    # ``multi_seed_stats`` interprets ``alpha`` as a two-sided TEST level and
    # halves it again internally.  ``alpha_test`` here is the one-sided TAIL
    # probability the significance test will actually use (matching
    # run_ablation's corrected threshold), so undo the internal halving for
    # the two-sided planning calls.  Passing ``alpha_test`` directly would
    # correct a two-sided plan twice (alpha/(4*family)) and overstate both
    # the required n and the MDE.
    planning_alpha = (alpha_test if alt in ('greater', 'less')
                      else 2.0 * alpha_test)
    dz = sesoi_value / sd_value
    required = seeds_required(dz, alpha=planning_alpha, power=target,
                              alternative=alt)
    # A paired test is undefined below two seed pairs. The normal
    # approximation can return 1 for a very large effect; clamp the
    # planning answer to the smallest design that can actually be run.
    if required is not None:
        required = max(2, int(required))
    mde = min_detectable_effect(n_seeds, sd_value, alpha=planning_alpha,
                                power=target, alternative=alt)
    # The SESOI is a magnitude.  For a lower-tail alternative the effect that
    # the plan is powered against points DOWN, while for 'greater' and the
    # symmetric two-sided test it points up.  Passing the positive magnitude
    # to a directional power calculation would make every 'less' plan look
    # powerless even when the seed budget is ample.
    planning_effect = (-sesoi_value if alt == 'less' else sesoi_value)
    achieved = power_at_effect(n_seeds, sd_value, planning_effect,
                               alpha=planning_alpha, alternative=alt)
    if required is None:
        status = 'not_estimable'
    elif achieved is not None and achieved >= target - 1e-9:
        status = 'confirmatory_ready'
    else:
        status = 'underpowered_at_sesoi'
    # The power arithmetic above uses a continuous normal approximation.
    # The runners' primary paired test is exact (sign-flip permutation),
    # whose p-value takes only 2**n discrete values. At small n and/or a
    # large corrected family the exact test can have NO rejection region,
    # so a normal-power "confirmatory_ready" verdict would be unattainable.
    # Report that floor before the run; do not let a p-value that could
    # never cross alpha be read as evidence.
    exact_resolution = exact_test_resolution(
        n_seeds, alpha=alpha, family_size=family, alternative=alt)
    exact_warning = None
    if not exact_resolution['family_testable']:
        exact_warning = (
            f'the exact sign-flip test has no rejection region at '
            f'n={n_seeds} and family_size={family}: the smallest '
            f'attainable Holm-adjusted p is '
            f"{exact_resolution['min_attainable_p_holm']:.6g} > "
            f'alpha={alpha:g}. The normal-approximation power number is '
            f'therefore unattainable; this is a screening design, not a '
            f'confirmatory one. It needs '
            f"{exact_resolution['min_effective_pairs_holm']} effective "
            f'seed pairs (shortfall '
            f"{exact_resolution['shortfall_pairs_holm']}).")
    # A TOST equivalence claim is a DIFFERENT design question from the
    # superiority SESOI above. Compute its prospective budget separately so
    # a run that is confirmatory-ready for superiority cannot be presented
    # as a confirmatory equivalence result by accident.
    equivalence = None
    equiv_cfg = plan.get('equivalence')
    if equiv_cfg is not None:
        if not isinstance(equiv_cfg, Mapping):
            raise DesignError(
                'plan.equivalence must be a mapping of margin/family_size/'
                f'alpha, got {type(equiv_cfg).__name__}')
        required_flag = equiv_cfg.get('required_for_claim', False)
        if not isinstance(required_flag, bool):
            raise DesignError(
                'plan.equivalence.required_for_claim must be a bool, got '
                f'{required_flag!r}')
        equivalence = equivalence_plan(
            margin=equiv_cfg.get('margin'),
            sd=sd_value,
            seeds=n_seeds,
            alpha=(alpha if equiv_cfg.get('alpha') is None
                   else equiv_cfg['alpha']),
            family_size=(family if equiv_cfg.get('family_size') is None
                         else equiv_cfg['family_size']),
            target_power=(target if equiv_cfg.get('target_power') is None
                          else equiv_cfg['target_power']),
            required_for_claim=required_flag,
            source=equiv_cfg.get('source'),
            note=equiv_cfg.get('note'),
        )
    # Interactions are a separate endpoint with a DIFFERENT variance
    # budget from the main effects. Reporting it next to (not inside) the
    # main planning record keeps "this run is confirmatory for a two-cell
    # contrast" from being read as "this run is confirmatory for the
    # difference-in-differences".
    interaction = None
    interaction_cfg = plan.get('interaction')
    if interaction_cfg is not None:
        if not isinstance(interaction_cfg, Mapping):
            raise DesignError(
                'plan.interaction must be a mapping of sesoi/alternative/'
                f'family_size/sd_inflation, got '
                f'{type(interaction_cfg).__name__}')
        interaction_flag = interaction_cfg.get('required_for_claim', False)
        if not isinstance(interaction_flag, bool):
            raise DesignError(
                'plan.interaction.required_for_claim must be a bool, got '
                f'{interaction_flag!r}')
        interaction = interaction_planning(
            sesoi=interaction_cfg.get('sesoi', sesoi_value),
            sd=sd_value,
            seeds=n_seeds,
            alpha=(alpha if interaction_cfg.get('alpha') is None
                   else interaction_cfg['alpha']),
            family_size=(family if interaction_cfg.get('family_size') is None
                         else interaction_cfg['family_size']),
            alternative=interaction_cfg.get('alternative', 'two-sided'),
            target_power=(target if interaction_cfg.get('target_power') is None
                          else interaction_cfg['target_power']),
            sd_inflation=interaction_cfg.get(
                'sd_inflation', INTERACTION_SD_INFLATION),
            required_for_claim=interaction_flag,
            estimand=interaction_cfg.get('estimand'),
            source=interaction_cfg.get('source'),
            note=interaction_cfg.get('note'),
        )
    return {
        'sesoi': sesoi_value,
        'assumed_sd': sd_value,
        'standardized_effect_dz': dz,
        'alpha_family_wise': alpha,
        'family_size': family,
        'alpha_per_test': alpha_test,
        'alternative': alt,
        'target_power': target,
        'seeds': int(n_seeds),
        'required_seeds': required,
        'planned_mde': mde,
        'planned_power': achieved,
        'power_gap': (None if achieved is None else target - achieved),
        'status': status,
        'exact_test_resolution': exact_resolution,
        'exact_test_warning': exact_warning,
        'equivalence': equivalence,
        'equivalence_status': (
            None if equivalence is None else equivalence['status']),
        'equivalence_claim_allowed': (
            None if equivalence is None
            else equivalence['supports_equivalence_at_planned_n']),
        'interaction': interaction,
        'interaction_status': (
            None if interaction is None else interaction['status']),
        'interaction_claim_allowed': (
            None if interaction is None
            else interaction['supports_interaction_at_planned_n']),
        'method': ('normal approximation for planning only: '
                   'n = ceil(((z_crit + z_power) / (SESOI / SD))^2); '
                   'inference uses the exact/t tests in multi_seed_stats'),
        'note': ('A plan is confirmatory-ready only when the seed count '
                 'reaches required_seeds at the corrected per-test alpha; '
                 'the normal approximation is slightly optimistic at small '
                 'n, so treat the target power as a floor to beat, not a '
                 'promise. Equivalence is a separate endpoint: see the '
                 'nested equivalence record, whose required n is usually '
                 'larger. A 2x2 interaction is a four-cell contrast with a '
                 'larger SD than the main-effect paired difference: see the '
                 'nested interaction record, whose required n is usually '
                 'larger still. exact_test_resolution is an independent '
                 'feasibility check on the discrete exact test: when '
                 'family_testable is False no p-value from that family '
                 'could have crossed alpha, whatever the observed effect.'),
    }


# ---------------------------------------------------------------------------
# Runtime audit
# ---------------------------------------------------------------------------

def common_analyzed_seeds(rows, *, per_seed_key: str = 'per_seed',
                          seed_key: str = 'seed',
                          error_key: str = 'error',
                          finite_keys=()) -> list[int]:
    """Return the seed ids that survived in EVERY row of a result block.

    Multi-seed rows carry a ``per_seed`` list whose entries may be error
    stubs (a diverged or crashed training run). The pre-registered design
    promises a COMPLETE seed block, so averaging only the survivors is a
    complete-case analysis, not the confirmatory experiment. This helper
    computes the intersection of the seed ids with usable data in every
    row; runners pass the result to :func:`audit_run` as
    ``analyzed_seeds``.

    The function is deliberately conservative: a row that is itself an
    error stub, or that has no usable ``per_seed`` list, yields ``[]`` (no
    complete-case population can be certified). ``finite_keys`` optionally
    names per-seed fields that must be present and finite (for example
    ``('final_acc', 'final_loss')``).

    Returns a sorted list of non-negative integer seed ids. It never
    raises on a malformed result row: the audit should report missing
    evidence, not abort the run that produced it.
    """
    if isinstance(rows, Mapping):
        raise DesignError('rows must be an iterable of row mappings, not a '
                          'single mapping')
    try:
        row_list = list(rows)
    except TypeError:
        raise DesignError('rows must be an iterable of row mappings') from None
    if not row_list:
        return []
    common: set[int] | None = None
    for row in row_list:
        if (not isinstance(row, Mapping)
                or row.get(error_key) is not None):
            return []
        per_seed = row.get(per_seed_key)
        if not isinstance(per_seed, list):
            return []
        valid: set[int] = set()
        for record in per_seed:
            if (not isinstance(record, Mapping)
                    or record.get(error_key) is not None):
                continue
            seed = record.get(seed_key)
            if not _is_int(seed) or seed < 0:
                continue
            if any(_clean_float(record.get(key)) is None
                   for key in finite_keys):
                continue
            valid.add(seed)
        common = valid if common is None else (common & valid)
        if not common:
            return []
    return sorted(common)


def audit_run(plan_id: str, *, seeds, levels=None, family_size=None,
              sd=None, analyzed_seeds=None) -> dict:
    """Compare an actual run configuration with its pre-registered plan.

    ``seeds`` is the explicit seed list REQUESTED by the run. ``levels``
    maps factor name -> observed levels (e.g. ``{'operator': ['linear',
    'softmax', ...]}``); a required control missing from the observed
    levels changes the verdict to ``exploratory_missing_controls``.
    ``family_size`` is the number of tests actually planned for THIS run
    (defaults to the registered family); passing the observed count keeps
    the audit honest when a runner adds or drops cells.

    ``analyzed_seeds`` is the optional explicit set of seed ids that
    actually entered the analysis. When it is supplied and is not exactly
    the requested block, the run is labelled ``incomplete_attrition``:
    dropping failed seeds is a complete-case analysis, and it is only
    confirmatory if the missing seeds can be treated as missing completely
    at random, which the data cannot establish. Omitting the argument keeps
    the historical behaviour (the audit reports only the requested block).

    The audit never changes the experiment; it is attached to the result
    metadata so a reader can see whether a claim is confirmatory. It does
    not raise for a merely underpowered, control-less, or attrition-afflicted
    run: those are exactly the states the audit exists to report.
    """
    plan = get_plan(plan_id)
    seed_list = _seed_list(seeds)
    n_seeds = len(seed_list)
    if levels is None:
        levels = {}
    elif not isinstance(levels, Mapping):
        raise DesignError(
            f'levels must be a mapping of factor name -> observed levels, '
            f'got {type(levels).__name__}')
    else:
        try:
            levels = dict(levels)
        except (TypeError, ValueError) as exc:
            raise DesignError(f'levels is not a usable mapping: {exc}') from None

    seed_errors: list[str] = []
    if n_seeds < 2:
        seed_errors.append('the paired design needs at least two seeds')
    bad_seeds = [s for s in seed_list if not _is_int(s) or s < 0]
    unique_seeds = (not bad_seeds
                    and len(set(seed_list)) == n_seeds)
    if not unique_seeds and not bad_seeds:
        seed_errors.append('seed list contains duplicates')
    if bad_seeds:
        seed_errors.append(f'seeds must be non-negative ints, got {bad_seeds!r}')

    # ---- Analysis population (optional post-run attrition audit) --------
    population_provided = analyzed_seeds is not None
    analyzed: list[int] | None = None
    population_bad: list[str] = []
    population_duplicates: list[int] = []
    population_errors: list[str] = []
    if population_provided:
        if (isinstance(analyzed_seeds, (str, bytes))
                or not hasattr(analyzed_seeds, '__iter__')):
            raise DesignError(
                'analyzed_seeds must be an explicit iterable of seed ids '
                f'(an int is a seed COUNT, which is ambiguous here), got '
                f'{analyzed_seeds!r}')
        analyzed = []
        seen_seeds: set[int] = set()
        for seed in list(analyzed_seeds):
            if not _is_int(seed) or seed < 0:
                population_bad.append(repr(seed))
                continue
            if seed in seen_seeds:
                population_duplicates.append(seed)
                continue
            seen_seeds.add(seed)
            analyzed.append(seed)
        if population_bad:
            population_errors.append(
                f'analyzed seeds must be non-negative ints, got '
                f'{population_bad!r}')
        if population_duplicates:
            population_errors.append(
                f'analyzed seed list contains duplicates: '
                f'{sorted(population_duplicates)!r}')

    requested_set = {s for s in seed_list
                     if _is_int(s) and s >= 0}
    analyzed_set = set(analyzed or ())
    missing_seeds = (sorted(requested_set - analyzed_set)
                     if population_provided else None)
    unexpected_seeds = (sorted(analyzed_set - requested_set)
                        if population_provided else None)
    n_analyzed = len(analyzed) if population_provided else None
    attrition_rate = (
        len(missing_seeds) / len(requested_set)
        if population_provided and requested_set else None)
    complete_case = bool(
        population_provided
        and not population_bad
        and not population_duplicates
        and not missing_seeds
        and not unexpected_seeds
        and n_analyzed is not None
        and n_analyzed >= 2)
    if (population_provided and n_analyzed is not None
            and n_analyzed < 2):
        population_errors.append(
            'the analyzed population has fewer than two seeds; no paired '
            'inference is defined')
    analysis_population = {
        'provided': population_provided,
        'requested_seeds': seed_list,
        'analyzed_seeds': analyzed,
        'n_requested': len(seed_list),
        'n_analyzed': n_analyzed,
        'missing_seeds': missing_seeds,
        'unexpected_seeds': unexpected_seeds,
        'duplicate_seeds': (population_duplicates
                            if population_provided else None),
        'invalid_seeds': (population_bad if population_provided else None),
        'n_missing': (len(missing_seeds)
                      if missing_seeds is not None else None),
        'attrition_rate': attrition_rate,
        'complete_case': (complete_case if population_provided else None),
        'errors': population_errors,
        'note': (
            'analyzed_seeds is the population that actually entered the '
            'analysis. A block smaller than the requested one is a '
            'complete-case analysis: if failures are related to the '
            'treatment, its estimate is biased. The pre-registered '
            'experiment is the complete block, not the survivors.'),
    }

    level_checks: dict[str, dict] = {}
    missing_required: list[str] = []
    for factor, required_levels in plan.get('required_levels', {}).items():
        observed = levels.get(factor)
        if observed is None:
            level_checks[factor] = {'status': 'not_evaluated',
                                    'required': list(required_levels),
                                    'observed': None}
            continue
        if isinstance(observed, (str, bytes)) or not hasattr(
                observed, '__iter__'):
            raise DesignError(
                f'levels[{factor!r}] must be an iterable of observed levels, '
                f'got {type(observed).__name__}; a bare string is ambiguous '
                'and is rejected rather than iterated character by character')
        observed = list(observed)
        missing = [level for level in required_levels
                   if level not in observed]
        missing_required.extend(f'{factor}:{level}' for level in missing)
        level_checks[factor] = {
            'status': 'missing' if missing else 'present',
            'required': list(required_levels),
            'observed': observed,
            'missing': missing,
        }

    planning = planning_report(plan, seeds=n_seeds, family_size=family_size,
                               sd=sd)
    exact_testable = bool(
        planning['exact_test_resolution']['family_testable'])
    equivalence = planning.get('equivalence')
    equivalence_supported = bool(
        equivalence is not None
        and equivalence.get('supports_equivalence_at_planned_n'))
    interaction = planning.get('interaction')
    interaction_supported = bool(
        interaction is not None
        and interaction.get('supports_interaction_at_planned_n'))
    planned_family = plan.get('planned_family_size')
    family_reduced = (family_size is not None
                      and family_size < int(planned_family))
    family_expanded = (family_size is not None
                       and family_size > int(planned_family))
    checks = {
        'seeds_present': bool(seed_list),
        'at_least_two_seeds': n_seeds >= 2,
        'seeds_unique': unique_seeds,
        'seeds_are_non_negative_ints': not bad_seeds,
        'required_levels_present': not missing_required,
        # None = the caller did not pass the observed family, so it cannot
        # be checked either way.
        'family_size_at_least_planned': (
            None if family_size is None else not family_reduced),
        # None = the caller did not pass the observed family. A family LARGER
        # than the pre-registered one is not the pre-registered experiment
        # either: it tests extra comparisons and needs its own (larger)
        # multiplicity correction, so it cannot be promoted to confirmatory.
        'family_size_at_most_planned': (
            None if family_size is None else not family_expanded),
        # None = the caller did not pass the analyzed population, so the
        # requested-seed audit is all that can be computed.
        'analysis_population_provided': population_provided,
        'analyzed_seeds_valid': (
            None if not population_provided
            else not population_bad and n_analyzed is not None
            and n_analyzed >= 2),
        'complete_case_population': (
            None if not population_provided else complete_case),
        # None = the plan makes no TOST equivalence claim.
        'equivalence_supported': (
            None if equivalence is None else equivalence_supported),
        # None = the plan makes no pre-registered interaction claim. A
        # four-cell interaction has its own (larger) seed requirement; the
        # main-effect confirmatory status does not cover it.
        'interaction_budget_supports_claim': (
            None if interaction is None else interaction_supported),
        # A necessary (not sufficient) feasibility check: if the discrete
        # exact test has no rejection region at this n/family, no outcome
        # could have crossed alpha.
        'exact_test_has_rejection_region': exact_testable,
    }
    valid_seeds = all(checks[key] for key in (
        'seeds_present', 'at_least_two_seeds', 'seeds_unique',
        'seeds_are_non_negative_ints'))
    design_warnings: list[str] = []
    if equivalence is not None and not equivalence_supported:
        design_warnings.append(
            f"the TOST margin {equivalence['margin']!r} needs "
            f"{equivalence.get('required_seeds_at_zero')!r} seeds for "
            f"{equivalence['target_power']:.0%} power at a true difference "
            f"of zero; this run has {n_seeds}. Equivalence verdicts are "
            "descriptive only and must not be cited as confirmatory")
    if interaction is not None and not interaction_supported:
        design_warnings.append(
            f"the pre-registered interaction {interaction.get('estimand')!r} "
            f"needs {interaction.get('required_seeds')!r} seeds for "
            f"{interaction['target_power']:.0%} power at its SESOI "
            f"{interaction['sesoi']!r} with sd_inflation "
            f"{interaction['sd_inflation']!r} (four-cell contrast); this run "
            f"has {n_seeds} seeds at family size "
            f"{interaction['family_size']}. The interaction verdict is "
            "descriptive only and must not be cited as a confirmatory "
            "transfer/complementarity claim")
    if family_expanded:
        design_warnings.append(
            f"the observed family size {family_size} exceeds the "
            f"pre-registered planned_family_size={planned_family}; the extra "
            "comparisons were not part of the confirmatory plan, so the run "
            "is screening/exploratory even if its seed budget is ample")
    if not exact_testable:
        design_warnings.append(
            planning.get('exact_test_warning')
            or ('the exact sign-flip test has no rejection region at this '
                'seed count and family size; treat the run as screening'))
    if not valid_seeds:
        status = 'invalid'
        claims = []
    elif missing_required:
        status = 'exploratory_missing_controls'
        claims = ['descriptive_only']
    elif population_provided and not complete_case:
        status = 'incomplete_attrition'
        if n_analyzed is not None and n_analyzed >= 2:
            claims = ['descriptive',
                      'complete_case_paired_inference_only']
        else:
            claims = []
    elif (planning['status'] == 'confirmatory_ready'
          and not exact_testable):
        # The normal-approximation power target is met, but the exact test
        # the plan actually names cannot reject at any p-value. That is a
        # resolution-limited design, not a confirmatory one.
        status = 'screening'
        claims = ['descriptive', 'exploratory_inference_with_caveat']
    elif planning['status'] != 'confirmatory_ready':
        status = 'screening'
        claims = ['descriptive', 'exploratory_inference_with_caveat']
    elif family_reduced:
        # A subset of the pre-registered family needs fewer seeds; that does
        # not make it the pre-registered experiment. Claims are limited to
        # the tested subset and cannot borrow the full family's coverage.
        status = 'reduced_family'
        claims = ['descriptive', 'paired_inference_for_tested_subset']
    elif family_expanded:
        # Extra comparisons beyond the pre-registered family were added
        # after the design was fixed. Even with enough seeds at the larger
        # corrected alpha, this is an exploratory scan, not the registered
        # experiment (the plan's family size and decision rule no longer
        # describe the analysis that was run).
        status = 'screening'
        claims = ['descriptive', 'exploratory_inference_with_caveat']
    else:
        status = 'confirmatory'
        claims = ['descriptive', 'paired_inference_at_planned_alpha',
                  'directional_claim_at_sesoi']
        if equivalence_supported:
            claims.append('practical_equivalence_at_margin')
    # The interaction is a separate estimand: even a confirmatory main-effect
    # run may only cite it when its own four-cell budget reaches its own
    # required n. This never upgrades a screening/reduced-family run.
    if interaction is not None and interaction_supported \
            and status == 'confirmatory':
        claims.append('interaction_claim_at_sesoi')
    analysis_planning = None
    if population_provided and n_analyzed is not None:
        # Realized sensitivity of the population that was actually
        # analyzed. This is reported next to the planned power so a reader
        # can see how much attrition cost, even when the survivors still
        # clear the planned n.
        analysis_planning = planning_report(
            plan, seeds=n_analyzed, family_size=family_size, sd=sd)
    return {
        'plan_id': plan_id,
        # Content fingerprint of the plan this run was audited against. The
        # full definition travels with the audit so a reader can see the
        # registered SESOI/controls/decision rule without the historical
        # checkout, and can re-hash it independently.
        'plan_sha256': plan_sha256(plan),
        'plan_fingerprint_scope': PLAN_FINGERPRINT_SCOPE,
        'plan_definition': plan,
        'status': status,
        'n_seeds': n_seeds,
        'n_analyzed': n_analyzed,
        'seeds': seed_list,
        'analysis_population': analysis_population,
        'screening_seeds': plan.get('screening_seeds'),
        'planned_seeds': plan.get('planned_seeds'),
        'family_size': family_size,
        'planned_family_size': planned_family,
        'family_reduced': family_reduced,
        'family_expanded': family_expanded,
        'checks': checks,
        'level_checks': level_checks,
        'missing_required_levels': missing_required,
        'seed_errors': seed_errors,
        'population_errors': population_errors,
        'planning': planning,
        'planning_at_analyzed_n': analysis_planning,
        'equivalence': equivalence,
        'equivalence_claim_allowed': (
            None if equivalence is None
            else bool(equivalence_supported and status == 'confirmatory')),
        'interaction': interaction,
        'interaction_claim_allowed': (
            None if interaction is None
            else bool(interaction_supported and status == 'confirmatory')),
        'design_warnings': design_warnings,
        'claims_allowed': claims,
        'interpretation': {
            'confirmatory': ('the seed budget reaches the pre-registered '
                             'required n, the full planned contrast family '
                             'is present, and the required controls are '
                             'present; corrected paired inference may be '
                             'reported as confirmatory. A practical-'
                             'equivalence claim additionally requires '
                             'equivalence_claim_allowed=True, which uses '
                             'the separate TOST margin and family. An '
                             'interaction claim additionally requires '
                             'interaction_claim_allowed=True, which uses '
                             'the separate four-cell SD budget'),
            'reduced_family': (
                'the run reaches the required n only because it tests a '
                'subset of the pre-registered contrast family; report it as '
                'a confirmatory test of that subset, not as the full '
                'pre-registered experiment'),
            'screening': ('the run is below the pre-registered required n '
                          'or tests a LARGER family than the pre-registered '
                          'one; report the seed budget and the MDE, treat '
                          'the result as exploratory, and do not promote it '
                          'to a confirmatory claim without a pre-registered '
                          'run at the matching family size'),
            'incomplete_attrition': (
                'the analyzed population is smaller than the pre-registered '
                'seed block (or contains seeds outside it), so the reported '
                'estimate is a complete-case estimate. Report the missing '
                'seeds and the realized power; do not present it as the '
                'pre-registered confirmatory result unless the failures can '
                'be shown to be unrelated to the treatment'),
            'exploratory_missing_controls': (
                'a required control/level is absent, so treatment '
                'differences cannot be attributed to the factor under '
                'test; re-run with the complete control set'),
            'invalid': ('the seed design is malformed (fewer than two '
                        'unique non-negative integer seeds); no paired '
                        'inference is defined'),
        }[status],
    }


# ---------------------------------------------------------------------------
# Pilot -> power -> confirmatory
# ---------------------------------------------------------------------------

def _safe_mean(values):
    """Mean of finite values without overflowing the intermediate sum."""
    n = len(values)
    if n == 0:
        return None
    total = sum(values)
    if math.isfinite(total):
        return total / n
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    mean = scale * (math.fsum(v / scale for v in values) / n)
    if not math.isfinite(mean):
        raise DesignError(
            f'mean is not finite after scaled summation (scale={scale!r})')
    return mean


def _sample_sd(values):
    """Sample SD, falling back to scaled arithmetic on an overflow.

    ``statistics.stdev`` is exact for ordinary pilots but can raise /
    return a non-finite value when two finite differences are near the
    float limits (``1.7e308 - -1.7e308`` overflows its float conversion).
    The fallback evaluates the same SD after dividing every value by a
    common scale, so ordinary samples keep the historical result.
    """
    if len(values) < 2:
        return None
    try:
        sd = statistics.stdev(values)
    except (OverflowError, ValueError):
        sd = None
    if sd is not None and math.isfinite(sd):
        return sd
    scale = max(abs(v) for v in values)
    if scale == 0.0:
        return 0.0
    scaled = [v / scale for v in values]
    mean = _safe_mean(scaled)
    var = math.fsum((v - mean) ** 2 for v in scaled) / (len(values) - 1)
    sd = scale * math.sqrt(var)
    return sd if math.isfinite(sd) else None


def pilot_sd_from_rows(rows, *, group_key: str, reference,
                       cell_keys=('task', 'n_kv'),
                       exclude=()) -> dict | None:
    """Estimate the paired-difference SD from a pilot result payload.

    Two shapes are supported:

    * contrast rows that already report ``sd_diff`` / ``n_common_seeds``
      (Exp 5b side file) - the median and maximum reported SD are used
      directly;
    * per-seed rows (Exp 4 / 5 / 8) - for every non-reference level in a
      cell, the seed-paired differences against ``reference`` are formed
      and the pooled within-pair SD is computed.

    The pooled SD is the primary estimate. ``sd_max`` is the largest
    per-contrast SD and drives the conservative recommendation, because a
    pilot's SD estimate is itself noisy and the plan should not be built on
    the most optimistic cell. ``exclude`` removes control levels (e.g. the
    memoryless ``linear`` rung) from the treatment SD estimate. Returns
    ``None`` when no contrast has at least two common seeds.
    """
    rows = _rows_from_payload(rows)
    cell_keys = list(cell_keys)
    excluded = {str(name) for name in exclude}
    direct = []
    for row in rows:
        if not isinstance(row, dict) or 'error' in row:
            continue
        label = row.get('contrast', group_key)
        if str(label) in excluded:
            continue
        sd = _clean_float(row.get('sd_diff'))
        n_common = row.get('n_common_seeds')
        if not _is_int(n_common) or n_common < 0:
            n_common = 0
        if sd is not None and sd >= 0.0 and n_common >= 2:
            direct.append({'sd': sd, 'n': n_common,
                           'cell': _cell_key(row, cell_keys),
                           'group': label})
    if direct:
        sds = [item['sd'] for item in direct]
        median_sd = statistics.median(sds)
        return {
            'method': 'reported_sd_diff',
            'group_key': group_key,
            'reference': reference,
            'cell_keys': cell_keys,
            'n_pairs': len(direct),
            'n_differences': sum(item['n'] for item in direct),
            'sd': median_sd,
            'sd_pooled': median_sd,
            'sd_median': median_sd,
            'sd_max': max(sds),
            'pairs': [
                {'cell': list(item['cell']), 'group': item['group'],
                 'n': item['n'], 'sd_diff': item['sd']}
                for item in direct],
            'note': ('the payload already reported seed-paired SDs; the '
                     'median is the primary pilot estimate and the maximum '
                     'is the conservative one'),
        }

    cells: dict[tuple, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or 'error' in row:
            continue
        group = row.get(group_key)
        if group is None or not _is_hashable(group) \
                or str(group) in excluded:
            # A malformed (e.g. list-valued) group cannot be a dict key or
            # compared to the reference level; skip it like any other
            # unusable pilot row rather than aborting the extraction.
            continue
        cell = _cell_key(row, cell_keys)
        if not _is_hashable(cell):
            continue
        per_seed = row.get('per_seed') or []
        if not isinstance(per_seed, (list, tuple)):
            continue
        acc_by_seed = {}
        for seed_row in per_seed:
            if not isinstance(seed_row, dict) or 'error' in seed_row:
                continue
            seed = seed_row.get('seed')
            acc = seed_row.get('final_acc')
            if acc is None:
                acc = seed_row.get('acc')
            acc = _clean_float(acc)
            if seed is None or not _is_hashable(seed) or acc is None:
                continue
            acc_by_seed[seed] = acc
        if acc_by_seed:
            cells.setdefault(cell, {})[group] = acc_by_seed

    pairs = []
    for cell, groups in cells.items():
        ref = groups.get(reference)
        if ref is None:
            continue
        for group, acc in groups.items():
            if group == reference:
                continue
            # Seed ids are hashable but not necessarily mutually ordered
            # (an integer-vs-string mix would make ``sorted`` raise). The
            # order is irrelevant to the paired statistics, so a stable
            # string key keeps a malformed seed type from aborting the
            # whole pilot extraction.
            common = sorted(set(ref) & set(acc), key=str)
            if len(common) < 2:
                continue
            diffs = [acc[seed] - ref[seed] for seed in common]
            mean_diff = _safe_mean(diffs)
            sd = _sample_sd(diffs)
            if sd is None:
                continue
            pairs.append({
                'cell': list(cell),
                'group': group,
                'n': len(common),
                'mean_diff': mean_diff,
                'sd_diff': sd,
                'diffs': diffs,
            })
    if not pairs:
        return None
    # Pooled within-pair SD: the standard fixed-effect estimate, which
    # weights a contrast by its degrees of freedom instead of letting a
    # 2-seed contrast with a huge SD dominate.
    ss = 0.0
    dof = 0
    try:
        for pair in pairs:
            mean_diff = pair['mean_diff']
            ss += math.fsum((d - mean_diff) ** 2 for d in pair['diffs'])
            dof += pair['n'] - 1
    except OverflowError:
        # Squaring a finite ~1e308 difference raises instead of producing
        # inf on CPython; mark the raw sum as unusable and use the scaled
        # path below.
        ss = math.inf
        dof = sum(pair['n'] - 1 for pair in pairs)
    pooled = None
    if dof > 0:
        if math.isfinite(ss):
            pooled = math.sqrt(ss / dof)
            if not math.isfinite(pooled):
                pooled = None
        if pooled is None:
            # Finite differences up to ~1e154 square to values the float
            # range cannot hold; evaluate the same pooled variance in
            # scaled units so the reported SD stays finite.
            scale = max(abs(d) for pair in pairs for d in pair['diffs'])
            if scale == 0.0:
                pooled = 0.0
            else:
                ss_scaled = math.fsum(
                    ((d / scale) - (pair['mean_diff'] / scale)) ** 2
                    for pair in pairs for d in pair['diffs'])
                candidate = scale * math.sqrt(ss_scaled / dof)
                pooled = candidate if math.isfinite(candidate) else None
    sds = [pair['sd_diff'] for pair in pairs]
    median_sd = statistics.median(sds)
    return {
        'method': 'per_seed_paired_differences',
        'group_key': group_key,
        'reference': reference,
        'cell_keys': cell_keys,
        'n_pairs': len(pairs),
        'n_differences': sum(pair['n'] for pair in pairs),
        'sd': pooled if pooled is not None else median_sd,
        'sd_pooled': pooled,
        'sd_median': median_sd,
        'sd_max': max(sds),
        'pairs': [
            {'cell': pair['cell'], 'group': pair['group'],
             'n': pair['n'], 'mean_diff': pair['mean_diff'],
             'sd_diff': pair['sd_diff']}
            for pair in pairs],
        'note': ('pooled within-pair SD across every treatment/reference '
                 'cell; a pilot SD is itself uncertain, so the maximum '
                 'per-contrast SD is also reported for a conservative plan'),
    }


def pilot_report(plan_id: str, rows, *, family_size=None, sd=None) -> dict:
    """Turn a pilot result payload into a confirmatory seed recommendation.

    ``rows`` may be a path, a bare list, or a ``{metadata, results}``
    envelope. ``sd`` overrides the extraction when the caller already has a
    per-seed difference SD from another source (it is then labelled as
    supplied, not observed).
    """
    plan = get_plan(plan_id)
    pilot_cfg = plan['pilot']
    family = plan['planned_family_size'] if family_size is None \
        else family_size
    if sd is not None:
        sd_info = {
            'method': 'supplied_sd',
            'sd': _need_number(sd, 'sd', positive=True),
            'sd_pooled': _need_number(sd, 'sd', positive=True),
            'sd_median': None,
            'sd_max': None,
            'n_pairs': None,
            'n_differences': None,
            'pairs': [],
            'note': 'caller-supplied SD; not estimated from this pilot',
        }
    else:
        sd_info = pilot_sd_from_rows(
            rows, group_key=pilot_cfg['group_key'],
            reference=pilot_cfg['reference'],
            cell_keys=pilot_cfg.get('cell_keys', ('task', 'n_kv')),
            exclude=pilot_cfg.get('exclude', ()))
    if sd_info is None:
        return {
            'plan_id': plan_id,
            'status': 'insufficient_pilot_data',
            'planning': planning_report(plan, family_size=family),
            'observed_sd': None,
            'recommendation': (
                'the pilot payload has no contrast with two common seeds; '
                'run the screening tier first or pass --sd explicitly'),
        }
    if sd_info.get('sd') is None or float(sd_info['sd']) <= 0.0:
        # A zero pilot SD is not a license for n=2: it can mean identical
        # seeds, a clipped/constant metric, or a degenerate model. Report it
        # instead of dividing by zero and inventing a sample size.
        return {
            'plan_id': plan_id,
            'status': 'degenerate_zero_sd',
            'pilot_is_confirmatory': False,
            'planning': planning_report(plan, family_size=family),
            'observed_sd': sd_info,
            'recommendation': (
                'the pilot shows zero paired-difference variance; inspect '
                'whether the metric is constant/clipped before planning, or '
                'pass --sd from a valid pilot'),
        }
    pilot_n = (int(statistics.median([p['n'] for p in sd_info['pairs']]))
               if sd_info.get('pairs') else None)
    pooled_plan = planning_report(plan, seeds=pilot_n or plan['screening_seeds'],
                                  family_size=family, sd=sd_info['sd'])
    required = pooled_plan['required_seeds']
    conservative = (
        planning_report(plan, seeds=plan['planned_seeds'], family_size=family,
                        sd=sd_info['sd_max'])['required_seeds']
        if sd_info.get('sd_max') is not None and sd_info['sd_max'] > 0 else None)
    if required is None:
        status = 'not_estimable'
    elif pilot_n is not None and pilot_n >= required:
        status = 'pilot_reaches_required_n'
    else:
        status = 'pilot_is_screening_only'
    recommended = max([n for n in (required, conservative) if n is not None],
                      default=None)
    equivalence = pooled_plan.get('equivalence') or {}
    required_equivalence = equivalence.get('required_seeds_at_zero')
    # The equivalence requirement is more sensitive to the SD than the
    # superiority requirement; report the conservative (max-SD) companion
    # next to it so a pilot cannot under-budget the TOST endpoint.
    required_equivalence_conservative = None
    if equivalence and sd_info.get('sd_max') is not None \
            and sd_info['sd_max'] > 0:
        conservative_plan = planning_report(
            plan, seeds=plan['planned_seeds'], family_size=family,
            sd=sd_info['sd_max'])
        required_equivalence_conservative = (
            conservative_plan.get('equivalence') or {}
        ).get('required_seeds_at_zero')
    # Interaction endpoints are a four-cell contrast on the same pilot
    # difference SD. Report their own (larger) requirement next to the
    # main-effect recommendation; a pilot meant to size an interaction
    # claim should still measure the difference-in-differences SD directly
    # rather than relying on the sqrt(2) model assumption.
    interaction = pooled_plan.get('interaction') or {}
    required_interaction = interaction.get('required_seeds')
    return {
        'plan_id': plan_id,
        'status': status,
        # A pilot contributes an SD estimate, never confirmatory rows: the
        # same data cannot both choose the sample size and test the
        # hypothesis.
        'pilot_is_confirmatory': False,
        'pilot_seeds': pilot_n,
        'observed_sd': {
            'method': sd_info['method'],
            'sd': sd_info['sd'],
            'sd_pooled': sd_info.get('sd_pooled'),
            'sd_median': sd_info.get('sd_median'),
            'sd_max': sd_info.get('sd_max'),
            'n_pairs': sd_info.get('n_pairs'),
            'n_differences': sd_info.get('n_differences'),
            'note': sd_info.get('note'),
            'pairs': sd_info.get('pairs', []),
        },
        'planning': pooled_plan,
        'pilot_power_at_sesoi': pooled_plan['planned_power'],
        'required_seeds_pooled_sd': required,
        'required_seeds_conservative': conservative,
        'required_seeds_equivalence': required_equivalence,
        'required_seeds_equivalence_conservative':
            required_equivalence_conservative,
        'required_seeds_interaction': required_interaction,
        'recommended_seeds': recommended,
        'planned_confirmatory_seeds': plan['planned_seeds'],
        'recommendation': (
            ('the SD implies no finite sample size (the SESOI is zero or '
             'undefined); revise the design assumptions')
            if recommended is None else
            ('the pilot is a screening run: plan the confirmatory run at '
             f'n >= {recommended} seeds, then re-run the audit; do not '
             'promote the pilot rows themselves to confirmatory evidence')),
        'equivalence_recommendation': (
            'this plan makes no TOST equivalence claim'
            if not equivalence else
            (f'equivalence at margin {equivalence.get("margin")} needs '
             f'{required_equivalence} seeds at the pooled SD'
             + (f' (or {required_equivalence_conservative} at the '
                'conservative max SD)'
                if required_equivalence_conservative is not None else '')
             + '; the superiority recommendation above is not sufficient '
             'for a practical-equivalence claim')),
        'interaction_recommendation': (
            'this plan makes no pre-registered interaction claim'
            if not interaction else
            (f'the four-cell interaction at SESOI '
             f'{interaction.get("sesoi")} needs '
             f'{required_interaction} seeds at the pooled SD '
             f'(SD_interaction = {interaction.get("sd_inflation")} x the '
             'pilot difference SD); the main-effect recommendation above '
             'is not sufficient for a transfer/complementarity claim')),
        'caveat': (
            'the pilot SD is estimated from few seed pairs and is itself '
            'uncertain; the conservative recommendation uses the largest '
            'per-contrast SD, and the normal approximation is optimistic '
            'at small n'),
    }


# ---------------------------------------------------------------------------
# Manifest and CLI
# ---------------------------------------------------------------------------

def design_manifest(plan_ids=None, *, seeds_by_plan=None, provenance=None) -> dict:
    """Build a versioned, JSON-safe manifest of the registered plans.

    ``seeds_by_plan`` maps plan id -> an explicit seed list or a seed
    COUNT. Overrides are recorded in the manifest next to the plan so the
    audit is reproducible. The manifest is intentionally not an experiment
    result with row contracts: it records the plan, not measurements.

    ``provenance`` is the optional ``capture_provenance()`` block captured
    at the moment the manifest is written. Supplying it binds the
    pre-registration to the exact working tree and runtime that produced
    it. It is an explicit argument rather than an internal import so this
    module stays stdlib-only and usable before torch is installed; callers
    that already have :mod:`kaggle_setup` (e.g. ``run_all``) pass it in.

    Every row also carries ``plan_sha256`` (the same fingerprint that
    :func:`audit_run` embeds in each result file), so the prospective
    manifest and the retrospective audit can be matched mechanically. The
    fingerprint covers the REGISTERED plan definition; per-invocation seed
    overrides remain visible in the row's ``seeds`` / ``n_seeds`` fields and
    are not part of the registry identity.
    """
    ids = list(PLAN_IDS if plan_ids is None else plan_ids)
    unknown = [p for p in ids if p not in PLANS]
    if unknown:
        raise DesignError(f'unknown experiment plan(s) {unknown!r}')
    overrides = dict(seeds_by_plan or {})
    unknown = [p for p in overrides if p not in PLANS]
    if unknown:
        raise DesignError(f'seeds_by_plan names unknown plan(s) {unknown!r}')
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise DesignError(
                'provenance must be the dict returned by '
                f'capture_provenance(), got {type(provenance).__name__}')
        try:
            # Reject NaN/Inf or non-JSON values now: the manifest is written
            # with allow_nan=False and a failure at write time would lose the
            # pre-registration that is supposed to exist BEFORE the run.
            json.dumps(provenance, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DesignError(f'provenance is not strict JSON: {exc}') from None
        provenance = copy.deepcopy(provenance)
    rows = []
    for plan_id in ids:
        plan = get_plan(plan_id)
        seeds = overrides.get(plan_id, plan['planned_seeds'])
        seed_list = _seed_list(seeds, name=f'{plan_id} seeds')
        planning = planning_report(plan, seeds=len(seed_list))
        row = dict(plan)
        row['seeds'] = seed_list
        row['n_seeds'] = len(seed_list)
        row['plan_sha256'] = plan_sha256(plan)
        row['plan_fingerprint_scope'] = PLAN_FINGERPRINT_SCOPE
        row['planning'] = planning
        row['status'] = planning['status']
        rows.append(row)
    metadata = {
        'generated_by': 'experiment_design.design_manifest',
        'seed_base': DEFAULT_SEED_BASE,
        'note': ('Registry of the pre-registered multi-seed designs. '
                 'planning is prospective (SESOI + assumed SD + '
                 'corrected alpha), not post-hoc. A run whose audit '
                 'status is screening is not confirmatory. Equivalence and '
                 '2x2-interaction endpoints carry their own budgets (TOST '
                 'margin and four-cell SD respectively); a superiority '
                 'budget does not cover either'),
        'method': ('normal approximation for planning only; inference '
                   'uses multi_seed_stats exact sign/permutation and '
                   't tests'),
        'plan_fingerprint_scope': PLAN_FINGERPRINT_SCOPE,
    }
    if provenance is not None:
        metadata['provenance'] = provenance
    return {
        'schema_version': '1.0',
        'kind': 'experiment_design',
        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'metadata': metadata,
        'results': rows,
    }


def write_design_manifest(path: str, *, plan_ids=None,
                          seeds_by_plan=None, manifest=None,
                          provenance=None) -> str:
    """Atomically write a design manifest (strict JSON) and return the path."""
    if manifest is None:
        manifest = design_manifest(plan_ids, seeds_by_plan=seeds_by_plan,
                                   provenance=provenance)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.experiment_design_', suffix='.json',
                               dir=parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True,
                      allow_nan=False)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _print_table(manifest: dict) -> None:
    header = (f"{'plan':<16} {'n':>3} {'family':>6} {'SESOI':>6} "
              f"{'SD':>5} {'need':>5} {'power':>6} {'eq_n':>5} "
              f"{'int_n':>5}  status")
    print(header)
    print('-' * len(header))
    for row in manifest['results']:
        p = row['planning']
        power = ('  n/a' if p['planned_power'] is None
                 else f"{p['planned_power']:.2f}")
        need = ('n/a' if p['required_seeds'] is None
                else str(p['required_seeds']))
        eq = p.get('equivalence') or {}
        eq_need = ('-' if eq.get('required_seeds_at_zero') is None
                   else str(eq['required_seeds_at_zero']))
        inter = p.get('interaction') or {}
        int_need = ('-' if inter.get('required_seeds') is None
                    else str(inter['required_seeds']))
        print(f"{row['plan_id']:<16} {row['n_seeds']:>3} "
              f"{p['family_size']:>6} {p['sesoi']:>6.2f} "
              f"{p['assumed_sd']:>5.2f} {need:>5} {power:>6} "
              f"{eq_need:>5} {int_need:>5}  {row['status']}")
    print('\nPlanning is normal-approximation; status=screening means the '
          'seed budget is below the pre-registered confirmatory n.')
    print('eq_n is the separate TOST budget at the registered equivalence '
          'margin; "-" means the plan makes no equivalence claim. A plan can '
          'be confirmatory_ready for superiority and still not support an '
          'equivalence claim.')
    print('int_n is the separate four-cell interaction budget (SD = '
          'sqrt(2) x the main-effect paired-difference SD); "-" means the '
          'plan makes no interaction claim. The main-effect budget is not '
          'the interaction budget.')


def main(argv=None) -> int:
    """CLI: print/validate the plan table, write a manifest, or run a pilot."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Pre-registered experiment designs and prospective power '
                    'planning.')
    parser.add_argument('--plan', action='append', default=None,
                        help='plan id to show/write (repeatable; default all)')
    parser.add_argument('--check', action='store_true',
                        help='validate the committed registry and exit')
    parser.add_argument('--json', metavar='PATH', default=None,
                        help='write the versioned design manifest to PATH')
    parser.add_argument('--pilot', metavar='RESULTS_JSON', default=None,
                        help='estimate the paired SD from a pilot result file')
    parser.add_argument('--seeds', type=int, default=None,
                        help='override the seed count for the manifest')
    parser.add_argument('--family-size', type=int, default=None,
                        help='override the analysis family size')
    parser.add_argument('--sd', type=float, default=None,
                        help='supply a per-seed difference SD for --pilot')
    args = parser.parse_args(argv)

    if args.check:
        errors = validate_plans()
        if errors:
            print('[experiment_design] INVALID registry:')
            for error in errors:
                print(f'  - {error}')
            return 1
        print(f'[experiment_design] {len(PLANS)} plans valid; every '
              'declared confirmatory budget reaches its target power, and '
              'every equivalence endpoint either reaches its TOST budget or '
              'is marked required_for_claim=False')
        return 0

    plan_ids = args.plan or list(PLAN_IDS)
    unknown = [p for p in plan_ids if p not in PLANS]
    if unknown:
        parser.error(f'unknown plan(s) {unknown!r}; valid ids: {list(PLANS)}')
    if args.pilot:
        if len(plan_ids) != 1:
            parser.error('--pilot needs exactly one --plan id')
        try:
            report = pilot_report(plan_ids[0], args.pilot,
                                  family_size=args.family_size, sd=args.sd)
        except (OSError, ValueError) as exc:
            print(f'[experiment_design] pilot analysis failed: {exc}',
                  file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report['status'] != 'insufficient_pilot_data' else 1

    seeds_by_plan = ({plan_id: args.seeds for plan_id in plan_ids}
                     if args.seeds is not None else None)
    manifest = design_manifest(plan_ids, seeds_by_plan=seeds_by_plan)
    if args.family_size is not None or args.sd is not None:
        for row in manifest['results']:
            row['planning'] = planning_report(
                row, seeds=row['n_seeds'], family_size=args.family_size,
                sd=args.sd)
            row['status'] = row['planning']['status']
    if args.json:
        write_design_manifest(args.json, manifest=manifest)
        print(f'[experiment_design] wrote {args.json}')
    else:
        _print_table(manifest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
