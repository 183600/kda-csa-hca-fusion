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
* the stopping rule (fixed n, no optional stopping, no replacing failed
  seeds) and the decision rule.

Two things follow from that registry:

1. :func:`planning_report` computes the prospective required seed count and
   the power the current budget would actually have. A plan whose
   ``planned_seeds`` is below the required count fails :func:`validate_plan`,
   so the registry cannot silently drift away from its own arithmetic.
2. :func:`audit_run` compares an ACTUAL run configuration (seed list,
   observed factor levels, family size, optionally a pilot SD) with the
   plan. It labels the run ``confirmatory`` / ``screening`` /
   ``exploratory_missing_controls`` / ``invalid`` and states which claims
   the run is allowed to support. Runners attach this audit to their result
   metadata, so a 5-seed screening run cannot be mistaken for the
   pre-registered confirmatory run.

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
import json
import math
import os
import statistics
import sys
import tempfile

from multi_seed_stats import (min_detectable_effect, power_at_effect,
                              seeds_required)

__all__ = [
    'DesignError',
    'PLANS',
    'PLAN_IDS',
    'get_plan',
    'validate_plan',
    'validate_plans',
    'planning_report',
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
        'analysis': (
            'one-sided seed-paired t-test for the vs-reference contrast, '
            'Bonferroni/Holm over the planned family; exact sign and '
            'sign-flip permutation tests as robustness checks; TOST at the '
            'pre-specified SESOI for a "practically the same" claim'),
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
        'analysis': (
            'two-sided seed-paired t-test against the reference under one '
            'Bonferroni family; TOST at the pre-specified margin; no raw '
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
                   'fixed depth, plus cross-route replacement contrasts; '
                   'seeds are complete blocks'),
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
            'memory route, plus the A/D cross-route replacement contrasts'),
        'endpoint': 'final_acc',
        'direction': 'higher_is_better',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        # 15 within-route layout contrasts (5 x 3 routes) plus 6 cross-route
        # A/D contrasts for 3 route pairs. Planning with the full family is
        # conservative and keeps the cross-route two-sided tests honest.
        'planned_family_size': 21,
        'alternative': 'greater',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 7,
        'planned_seeds': 14,
        'analysis': (
            'seed-paired t-test under one Bonferroni family over every '
            'contrast row; one-sided for the pre-registered layout '
            'directions, two-sided for cross-route replacement; TOST for '
            'practical equivalence'),
        'decision_rule': (
            'only significant_bonferroni=True rows may be cited; the '
            'interaction row is a claim about complementarity, not a '
            'license to rank raw means'),
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
                   'frozen 3:1:1 layout; all variants share depth and '
                   'parameter count, seeds are complete blocks'),
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
            'with the pre-registered sign in run_mechanism_ablation.VARIANTS'),
        'endpoint': 'final_acc',
        'direction': 'pre_registered_sign',
        'sesoi': 0.10,
        'assumed_sd': DEFAULT_ASSUMED_SD,
        'alpha': DEFAULT_ALPHA,
        'planned_family_size': 5,
        'alternative': 'two-sided',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_seeds': 5,
        'planned_seeds': 14,
        'analysis': (
            'two-sided seed-paired t-test vs baseline, Bonferroni-corrected '
            'over the compared variants; the pre-registered sign decides '
            'supported vs contradicted; exact tests as robustness checks'),
        'decision_rule': (
            'a prediction is supported only by a corrected significant '
            'difference in the pre-registered direction; non-significance '
            'is not_supported (inconclusive), never a confirmed null'),
        'stopping_rule': (
            'fixed variant set and seed count fixed before the run; the '
            'audit refuses single-factor claims when the one-factor '
            'counterpart is missing'),
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
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


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
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Plan access and static validation
# ---------------------------------------------------------------------------

def get_plan(plan_id: str) -> dict:
    """Return a deep copy of a registered plan (callers cannot mutate it)."""
    if plan_id not in PLANS:
        raise DesignError(f'unknown experiment plan {plan_id!r}; valid ids: '
                          f'{list(PLANS)}')
    return copy.deepcopy(PLANS[plan_id])


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
        factor_levels = factors[name].get('levels', [])
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

def planning_report(plan, *, seeds=None, family_size=None, sd=None,
                    alpha=None, alternative=None, sesoi=None,
                    target_power=None) -> dict:
    """Compute the prospective seed budget and the power it delivers.

    All overrides are for "what if" planning and are validated; the
    defaults come from the plan. ``alpha`` is the FAMILY-WISE alpha. The
    per-test threshold used here is ``alpha / family_size`` for a
    directional family and ``alpha / (2 * family_size)`` for a two-sided
    family, matching the Bonferroni convention in the runners.

    The returned record is JSON-safe and includes the arithmetic inputs so a
    reader can audit the sample-size decision without importing this module.
    """
    plan = get_plan(plan) if isinstance(plan, str) else plan
    if not isinstance(plan, dict):
        raise DesignError('plan must be a plan id or a dict')
    alpha = float(plan.get('alpha', DEFAULT_ALPHA) if alpha is None
                  else _need_number(alpha, 'alpha'))
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
    sesoi_value = float(plan.get('sesoi', 0.10) if sesoi is None
                        else _need_number(sesoi, 'sesoi', positive=True))
    if sesoi_value <= 0.0:
        raise DesignError(f'sesoi must be > 0, got {sesoi_value!r}')
    sd_value = float(plan.get('assumed_sd', DEFAULT_ASSUMED_SD) if sd is None
                     else _need_number(sd, 'sd', positive=True))
    if sd_value <= 0.0:
        raise DesignError(f'sd must be > 0, got {sd_value!r}')
    target = float(plan.get('target_power', DEFAULT_TARGET_POWER)
                   if target_power is None
                   else _need_number(target_power, 'target_power',
                                     positive=True))
    if not (0.0 < target < 1.0):
        raise DesignError(f'target_power must be in (0,1), got {target!r}')
    n_seeds = plan.get('planned_seeds', 2) if seeds is None else seeds
    if not _is_int(n_seeds) or n_seeds < 0:
        raise DesignError(f'seeds must be a non-negative int, got {n_seeds!r}')
    alpha_test = (alpha / family if alt in ('greater', 'less')
                  else alpha / (2.0 * family))
    dz = sesoi_value / sd_value
    required = seeds_required(dz, alpha=alpha_test, power=target,
                              alternative=alt)
    # A paired test is undefined below two seed pairs. The normal
    # approximation can return 1 for a very large effect; clamp the
    # planning answer to the smallest design that can actually be run.
    if required is not None:
        required = max(2, int(required))
    mde = min_detectable_effect(n_seeds, sd_value, alpha=alpha_test,
                                power=target, alternative=alt)
    achieved = power_at_effect(n_seeds, sd_value, sesoi_value,
                               alpha=alpha_test, alternative=alt)
    if required is None:
        status = 'not_estimable'
    elif achieved is not None and achieved >= target - 1e-9:
        status = 'confirmatory_ready'
    else:
        status = 'underpowered_at_sesoi'
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
        'method': ('normal approximation for planning only: '
                   'n = ceil(((z_crit + z_power) / (SESOI / SD))^2); '
                   'inference uses the exact/t tests in multi_seed_stats'),
        'note': ('A plan is confirmatory-ready only when the seed count '
                 'reaches required_seeds at the corrected per-test alpha; '
                 'the normal approximation is slightly optimistic at small '
                 'n, so treat the target power as a floor to beat, not a '
                 'promise.'),
    }


# ---------------------------------------------------------------------------
# Runtime audit
# ---------------------------------------------------------------------------

def audit_run(plan_id: str, *, seeds, levels=None, family_size=None,
              sd=None) -> dict:
    """Compare an actual run configuration with its pre-registered plan.

    ``seeds`` is the explicit seed list used by the run. ``levels`` maps
    factor name -> observed levels (e.g. ``{'operator': ['linear',
    'softmax', ...]}``); a required control missing from the observed
    levels changes the verdict to ``exploratory_missing_controls``.
    ``family_size`` is the number of tests actually planned for THIS run
    (defaults to the registered family); passing the observed count keeps
    the audit honest when a runner adds or drops cells.

    The audit never changes the experiment; it is attached to the result
    metadata so a reader can see whether a claim is confirmatory. It does
    not raise for a merely underpowered or control-less run: those are
    exactly the states the audit exists to report.
    """
    plan = get_plan(plan_id)
    seed_list = _seed_list(seeds)
    n_seeds = len(seed_list)
    levels = dict(levels or {})

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

    level_checks: dict[str, dict] = {}
    missing_required: list[str] = []
    for factor, required_levels in plan.get('required_levels', {}).items():
        observed = levels.get(factor)
        if observed is None:
            level_checks[factor] = {'status': 'not_evaluated',
                                    'required': list(required_levels),
                                    'observed': None}
            continue
        missing = [level for level in required_levels
                   if level not in list(observed)]
        missing_required.extend(f'{factor}:{level}' for level in missing)
        level_checks[factor] = {
            'status': 'missing' if missing else 'present',
            'required': list(required_levels),
            'observed': list(observed),
            'missing': missing,
        }

    planning = planning_report(plan, seeds=n_seeds, family_size=family_size,
                               sd=sd)
    planned_family = plan.get('planned_family_size')
    family_reduced = (family_size is not None
                      and family_size < int(planned_family))
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
    }
    valid_seeds = all(checks[key] for key in (
        'seeds_present', 'at_least_two_seeds', 'seeds_unique',
        'seeds_are_non_negative_ints'))
    if not valid_seeds:
        status = 'invalid'
        claims = []
    elif missing_required:
        status = 'exploratory_missing_controls'
        claims = ['descriptive_only']
    elif planning['status'] != 'confirmatory_ready':
        status = 'screening'
        claims = ['descriptive', 'exploratory_inference_with_caveat']
    elif family_reduced:
        # A subset of the pre-registered family needs fewer seeds; that does
        # not make it the pre-registered experiment. Claims are limited to
        # the tested subset and cannot borrow the full family's coverage.
        status = 'reduced_family'
        claims = ['descriptive', 'paired_inference_for_tested_subset']
    else:
        status = 'confirmatory'
        claims = ['descriptive', 'paired_inference_at_planned_alpha',
                  'directional_claim_at_sesoi']
    return {
        'plan_id': plan_id,
        'status': status,
        'n_seeds': n_seeds,
        'seeds': seed_list,
        'screening_seeds': plan.get('screening_seeds'),
        'planned_seeds': plan.get('planned_seeds'),
        'family_size': family_size,
        'planned_family_size': planned_family,
        'family_reduced': family_reduced,
        'checks': checks,
        'level_checks': level_checks,
        'missing_required_levels': missing_required,
        'seed_errors': seed_errors,
        'planning': planning,
        'claims_allowed': claims,
        'interpretation': {
            'confirmatory': ('the seed budget reaches the pre-registered '
                             'required n, the full planned contrast family '
                             'is present, and the required controls are '
                             'present; corrected paired inference may be '
                             'reported as confirmatory'),
            'reduced_family': (
                'the run reaches the required n only because it tests a '
                'subset of the pre-registered contrast family; report it as '
                'a confirmatory test of that subset, not as the full '
                'pre-registered experiment'),
            'screening': ('the run is below the pre-registered required n; '
                          'report the seed budget and the MDE, treat the '
                          'result as exploratory, and do not promote it to '
                          'a confirmatory claim without a larger run'),
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

def _sample_sd(values):
    return statistics.stdev(values) if len(values) >= 2 else None


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
        try:
            n_common = int(row.get('n_common_seeds'))
        except (TypeError, ValueError):
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
        if group is None or str(group) in excluded:
            continue
        acc_by_seed = {}
        for seed_row in row.get('per_seed') or []:
            if not isinstance(seed_row, dict) or 'error' in seed_row:
                continue
            seed = seed_row.get('seed')
            acc = seed_row.get('final_acc')
            if acc is None:
                acc = seed_row.get('acc')
            acc = _clean_float(acc)
            if seed is None or acc is None:
                continue
            acc_by_seed[seed] = acc
        if acc_by_seed:
            cells.setdefault(_cell_key(row, cell_keys), {})[group] = \
                acc_by_seed

    pairs = []
    for cell, groups in cells.items():
        ref = groups.get(reference)
        if ref is None:
            continue
        for group, acc in groups.items():
            if group == reference:
                continue
            common = sorted(set(ref) & set(acc))
            if len(common) < 2:
                continue
            diffs = [acc[seed] - ref[seed] for seed in common]
            mean_diff = sum(diffs) / len(diffs)
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
    for pair in pairs:
        mean_diff = pair['mean_diff']
        ss += sum((d - mean_diff) ** 2 for d in pair['diffs'])
        dof += pair['n'] - 1
    pooled = math.sqrt(ss / dof) if dof > 0 else None
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
        'recommended_seeds': recommended,
        'planned_confirmatory_seeds': plan['planned_seeds'],
        'recommendation': (
            ('the SD implies no finite sample size (the SESOI is zero or '
             'undefined); revise the design assumptions')
            if recommended is None else
            ('the pilot is a screening run: plan the confirmatory run at '
             f'n >= {recommended} seeds, then re-run the audit; do not '
             'promote the pilot rows themselves to confirmatory evidence')),
        'caveat': (
            'the pilot SD is estimated from few seed pairs and is itself '
            'uncertain; the conservative recommendation uses the largest '
            'per-contrast SD, and the normal approximation is optimistic '
            'at small n'),
    }


# ---------------------------------------------------------------------------
# Manifest and CLI
# ---------------------------------------------------------------------------

def design_manifest(plan_ids=None, *, seeds_by_plan=None) -> dict:
    """Build a versioned, JSON-safe manifest of the registered plans.

    ``seeds_by_plan`` maps plan id -> an explicit seed list or a seed
    COUNT. Overrides are recorded in the manifest next to the plan so the
    audit is reproducible. The manifest is intentionally not an experiment
    result with row contracts: it records the plan, not measurements.
    """
    ids = list(PLAN_IDS if plan_ids is None else plan_ids)
    unknown = [p for p in ids if p not in PLANS]
    if unknown:
        raise DesignError(f'unknown experiment plan(s) {unknown!r}')
    overrides = dict(seeds_by_plan or {})
    unknown = [p for p in overrides if p not in PLANS]
    if unknown:
        raise DesignError(f'seeds_by_plan names unknown plan(s) {unknown!r}')
    rows = []
    for plan_id in ids:
        plan = get_plan(plan_id)
        seeds = overrides.get(plan_id, plan['planned_seeds'])
        seed_list = _seed_list(seeds, name=f'{plan_id} seeds')
        planning = planning_report(plan, seeds=len(seed_list))
        row = dict(plan)
        row['seeds'] = seed_list
        row['n_seeds'] = len(seed_list)
        row['planning'] = planning
        row['status'] = planning['status']
        rows.append(row)
    return {
        'schema_version': '1.0',
        'kind': 'experiment_design',
        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'metadata': {
            'generated_by': 'experiment_design.design_manifest',
            'seed_base': DEFAULT_SEED_BASE,
            'note': ('Registry of the pre-registered multi-seed designs. '
                     'planning is prospective (SESOI + assumed SD + '
                     'corrected alpha), not post-hoc. A run whose audit '
                     'status is screening is not confirmatory.'),
            'method': ('normal approximation for planning only; inference '
                       'uses multi_seed_stats exact sign/permutation and '
                       't tests'),
        },
        'results': rows,
    }


def write_design_manifest(path: str, *, plan_ids=None,
                          seeds_by_plan=None, manifest=None) -> str:
    """Atomically write a design manifest (strict JSON) and return the path."""
    if manifest is None:
        manifest = design_manifest(plan_ids, seeds_by_plan=seeds_by_plan)
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
              f"{'SD':>5} {'need':>5} {'power':>6}  status")
    print(header)
    print('-' * len(header))
    for row in manifest['results']:
        p = row['planning']
        power = ('  n/a' if p['planned_power'] is None
                 else f"{p['planned_power']:.2f}")
        need = ('n/a' if p['required_seeds'] is None
                else str(p['required_seeds']))
        print(f"{row['plan_id']:<16} {row['n_seeds']:>3} "
              f"{p['family_size']:>6} {p['sesoi']:>6.2f} "
              f"{p['assumed_sd']:>5.2f} {need:>5} {power:>6}  "
              f"{row['status']}")
    print('\nPlanning is normal-approximation; status=screening means the '
          'seed budget is below the pre-registered confirmatory n.')


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
              'declared confirmatory budget reaches its target power')
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
