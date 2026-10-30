"""Pre-registered measurement designs for the latency experiments.

Motivation
----------
``experiment_design`` plans the multi-seed training comparisons, where the
experimental unit is a trained model and the uncertainty is seed-to-seed
variation. The latency experiments (Exp 2 benchmark and Exp 6 decoding)
have a different design vocabulary: there are no training seeds, the
experimental unit is a timed trial, and the threats are warmup transients,
allocator state, device frequency drift, and the order in which operators
are measured. Treating repeated timings as if they were independent seeds
(or ignoring order entirely) produces confident-looking ratios that a
rerun does not reproduce.

This module supplies the missing planning side for those experiments:

* a committed registry of the measurement plans, each with the primary
  A/B latency-ratio contrasts, the smallest effect of interest (SESOI) as
  a ratio, an assumed coefficient of variation, the analysis family, and
  the repeat budget the plan requires;
* prospective planning arithmetic on the log-ratio scale, so "how many
  repeats do I need?" is answered before the timings exist rather than
  after a ratio happens to look large;
* a deterministic balanced schedule (randomized complete blocks with a
  cyclic rotation) so one operator is not always measured first or last
  within a workload sweep;
* a run audit that labels a timing table ``confirmatory`` /
  ``screening`` / ``incomplete_cells`` / ``order_confounded`` and reports
  the realized precision at the pre-registered SESOI. The audit never
  changes a timing, only the claim it can support;
* an archive format for per-trial timings plus exact bootstrap ratio
  contrasts (``raw_manifest`` / ``raw_contrast_rows``), so the summary-level
  audit interval can be replaced by ``timing_stats.ratio_of_medians_ci``
  on the retained trials.

Design constraints
------------------
* Stdlib only (``math`` / ``random`` / ``statistics``). The planning path
  must run before torch is installed or a GPU is available.
* Deterministic and JSON-safe. Schedule randomness is isolated in a local
  ``random.Random(seed)``; no global RNG is touched.
* The planning arithmetic is a normal approximation on log medians. The
  ratio interval reported by the audit is explicitly labelled as a
  summary-level approximation; when raw per-trial timings are available,
  ``timing_stats.ratio_of_medians_ci`` remains the inference path.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
from collections.abc import Mapping
from statistics import NormalDist

from multi_seed_stats import (min_detectable_effect, power_at_effect,
                              seeds_required)
from timing_stats import DEFAULT_BOOTSTRAP_REPS, ratio_of_medians_ci

__all__ = [
    'TimingDesignError',
    'TIMING_PLANS',
    'TIMING_PLAN_IDS',
    'DEFAULT_SCHEDULE_SEED',
    'DEFAULT_WARMUP_PASSES',
    'get_plan',
    'plan_sha256',
    'validate_plan',
    'validate_plans',
    'ratio_log_se_unit',
    'planning_report',
    'balanced_schedule',
    'cells_from_benchmark_rows',
    'cells_from_decoding_rows',
    'load_cells',
    'load_raw_cells',
    'result_plan_id',
    'raw_manifest',
    'write_raw_manifest',
    'raw_contrast_rows',
    'pilot_report',
    'audit_run',
    'design_manifest',
    'write_design_manifest',
    'main',
]

DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
DEFAULT_SESOI_RATIO = 1.15
DEFAULT_ASSUMED_CV = 0.10
DEFAULT_SCHEDULE_SEED = 0
DEFAULT_WARMUP_PASSES = 2
PLAN_FINGERPRINT_SCOPE = (
    'timing plan definition (contrasts, SESOI ratio, assumed CV, analysis '
    'family, repeat budgets, warmup and schedule); the per-invocation '
    'repeat override is recorded separately')


class TimingDesignError(ValueError):
    """A timing plan, schedule, cell table, or planning argument is invalid."""


TIMING_PLANS: dict[str, dict] = {
    'exp2_benchmark': {
        'plan_id': 'exp2_benchmark',
        'title': 'Exp 2 - operator latency across sequence lengths',
        'runner': 'run_benchmark.py',
        'result_files': ['results/exp2_benchmark.json',
                         'results/exp2_benchmark_provenance.json'],
        'question': (
            'At a fixed workload and compute boundary, does operator A '
            'differ from operator B in median latency by at least the '
            'pre-registered ratio?'),
        'design': (
            'randomized complete block over operator order: every workload '
            'block measures each operator once per repetition cycle, and '
            'the base order is rotated across workload blocks so position '
            'is not confounded with operator'),
        'unit': (
            'one timed forward pass of one (operator, workload) cell after '
            'the warmup; repeats within a cell are the units of the ratio '
            'interval'),
        'factors': {
            'workload': {'role': 'block',
                         'levels': 'BENCH_LENGTHS (default 128..2048)'},
            'operator': {'role': 'treatment',
                         'levels': ['softmax', 'kda_rec', 'kda_chunk',
                                    'kda_chunk_compiled', 'kda_rec_hybrid',
                                    'kda_chunk_hybrid', 'mamba2_scan',
                                    'mamba2_chunked_scan', 'mamba2_layer',
                                    'csa', 'hca', 'hybrid', 'hybrid_mamba2']},
            'trial': {'role': 'replicate', 'levels': 'BENCH_REPEATS'},
        },
        'primary_contrasts': [
            {'id': 'route_swap_hybrid',
             'numerator': 'hybrid_mamba2', 'denominator': 'hybrid'},
            {'id': 'kda_chunk_vs_rec',
             'numerator': 'kda_chunk', 'denominator': 'kda_rec'},
            {'id': 'mamba2_chunked_vs_scan',
             'numerator': 'mamba2_chunked_scan',
             'denominator': 'mamba2_scan'},
        ],
        'endpoint': 'ratio_of_medians_ms',
        'direction': 'two_sided',
        'sesoi_ratio': DEFAULT_SESOI_RATIO,
        'assumed_cv': DEFAULT_ASSUMED_CV,
        'alpha': DEFAULT_ALPHA,
        'planned_family_size': 15,
        'alternative': 'two-sided',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_repeats': 5,
        'planned_repeats': 15,
        'warmup': 2,
        'schedule': 'balanced_rotation',
        'allocation_rule': (
            'every operator receives the same repeat count at every '
            'workload; a slow or erroring operator is reported, not '
            'dropped from the table'),
        'analysis': (
            'ratio of medians on the log scale; the primary interval is '
            'timing_stats.ratio_of_medians_ci when raw trials are '
            'available, otherwise the audit reports the normal log-ratio '
            'approximation from the per-cell CV and repeat count'),
        'decision_rule': (
            'a relative-latency claim requires a corrected interval that '
            'excludes a ratio of 1 AND a realized MDE ratio at or below '
            'the pre-registered SESOI. Two overlapping intervals, or a '
            'ratio computed from a single repeat, are not evidence'),
        'stopping_rule': (
            'repeat count and operator set are fixed before the run; no '
            'optional stopping, no adding repeats until a ratio crosses '
            'the threshold, and no dropping a slow operator'),
        'pilot': {'cv_key': 'time_cv', 'cell_keys': ['T']},
    },
    'exp6_decoding': {
        'plan_id': 'exp6_decoding',
        'title': 'Exp 6 - per-token decoding latency after a prefill',
        'runner': 'run_decoding.py',
        'result_files': ['results/exp6_decoding.json',
                         'results/exp6_decoding_provenance.json'],
        'question': (
            'After a fixed prefill, does operator A differ from operator B '
            'in median per-token decode latency by at least the '
            'pre-registered ratio?'),
        'design': (
            'randomized complete block over operator order within each '
            'prefill-length block; every block warms the decode path once '
            'before the timed trials and keeps the prefill input fixed '
            'across operators'),
        'unit': (
            'one timed decode trial (a full prefill + decode loop) of one '
            '(operator, prefill length) cell; repeated trials are the units '
            'of the ratio interval'),
        'factors': {
            'workload': {'role': 'block',
                         'levels': 'DECODE_PREFILL_LENS (default 128..2048)'},
            'operator': {'role': 'treatment',
                         'levels': ['softmax', 'kda', 'mamba2', 'rwkv7',
                                    'csa', 'hca', 'hybrid', 'hybrid_mamba2',
                                    'hybrid_rwkv7']},
            'trial': {'role': 'replicate', 'levels': 'DECODE_REPEATS'},
        },
        'primary_contrasts': [
            {'id': 'route_swap_hybrid',
             'numerator': 'hybrid_mamba2', 'denominator': 'hybrid'},
            {'id': 'csa_vs_hca',
             'numerator': 'csa', 'denominator': 'hca'},
            {'id': 'kda_vs_mamba2',
             'numerator': 'kda', 'denominator': 'mamba2'},
        ],
        'endpoint': 'ratio_of_medians_ms_per_token',
        'direction': 'two_sided',
        'sesoi_ratio': DEFAULT_SESOI_RATIO,
        'assumed_cv': DEFAULT_ASSUMED_CV,
        'alpha': DEFAULT_ALPHA,
        'planned_family_size': 12,
        'alternative': 'two-sided',
        'target_power': DEFAULT_TARGET_POWER,
        'screening_repeats': 3,
        'planned_repeats': 15,
        'warmup': 1,
        'schedule': 'balanced_rotation',
        'allocation_rule': (
            'every operator receives the same repeat count at every '
            'prefill length; the per-token decode metric is paired with '
            'its prefill but not across operators'),
        'analysis': (
            'ratio of median per-token decode latencies on the log scale; '
            'the same precision and multiplicity rules as Exp 2'),
        'decision_rule': (
            'a relative decode-latency claim requires a corrected interval '
            'excluding a ratio of 1 AND a realized MDE ratio at or below '
            'the pre-registered SESOI; a flat-vs-growing shape comparison '
            'is descriptive unless a ratio contrast was registered'),
        'stopping_rule': (
            'repeat count and operator set are fixed before the run; no '
            'optional stopping and no dropping an operator after seeing '
            'its decode cost'),
        'pilot': {'cv_key': 'decode_cv', 'cell_keys': ['prefill_len']},
    },
}

TIMING_PLAN_IDS = tuple(TIMING_PLANS)

_SCHEDULES = ('balanced_rotation',)


def _is_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        # JSON integers are unbounded; one beyond the float range is a
        # malformed number, not a reason to abort the design audit.
        return False


def _need_number(value, name, *, positive=False):
    if not _is_number(value):
        raise TimingDesignError(f'{name} must be a finite number, got {value!r}')
    value = float(value)
    if positive and value <= 0.0:
        raise TimingDesignError(f'{name} must be > 0, got {value!r}')
    return value


def _need_int(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimingDesignError(f'{name} must be an int, got {value!r}')
    if minimum is not None and value < minimum:
        raise TimingDesignError(f'{name} must be >= {minimum}, got {value!r}')
    return value


def _safe_exp(value):
    if value is None:
        return None
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _safe_log(value):
    if value is None or not _is_number(value) or float(value) <= 0.0:
        return None
    result = math.log(float(value))
    return result if math.isfinite(result) else None


def _planning_alpha(alpha, family_size, alternative):
    """Test-level alpha whose one-sided tail is the corrected threshold."""
    tail = (alpha / family_size
            if alternative != 'two-sided'
            else alpha / (2.0 * family_size))
    return tail if alternative != 'two-sided' else 2.0 * tail


def get_plan(plan_id: str) -> dict:
    try:
        plan = TIMING_PLANS[plan_id]
    except (KeyError, TypeError):
        raise TimingDesignError(
            f'unknown timing plan {plan_id!r}; known: '
            f'{sorted(TIMING_PLANS)}') from None
    return copy.deepcopy(plan)


def plan_sha256(plan) -> str:
    plan = get_plan(plan) if isinstance(plan, str) else plan
    if not isinstance(plan, Mapping):
        raise TimingDesignError('plan must be a plan id or a mapping')
    payload = json.dumps(plan, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def ratio_log_se_unit(cv_numerator, cv_denominator=None, *,
                      paired=False, rho=0.0) -> float:
    """Per-repeat SD of the log latency ratio under the delta method.

    ``cv`` is the coefficient of variation of a timing sample. For two
    independently timed cells the log-ratio variance per repeat is
    ``cv_a**2 + cv_b**2``. With ``paired=True`` the same trial indices
    must be used for both cells and a correlation ``rho`` removes the
    shared trial effect: ``cv_a**2 + cv_b**2 - 2*rho*cv_a*cv_b``.
    """
    cv_a = _need_number(cv_numerator, 'cv_numerator', positive=True)
    cv_b = (_need_number(cv_denominator, 'cv_denominator', positive=True)
            if cv_denominator is not None else cv_a)
    if not isinstance(paired, bool):
        raise TimingDesignError(f'paired must be a bool, got {paired!r}')
    if not _is_number(rho) or not -1.0 <= float(rho) <= 1.0:
        raise TimingDesignError(f'rho must be in [-1, 1], got {rho!r}')
    if paired:
        variance = (cv_a ** 2 + cv_b ** 2
                    - 2.0 * float(rho) * cv_a * cv_b)
        if variance < 0.0:
            raise TimingDesignError(
                'paired log-ratio variance is negative; check rho and the CVs')
    else:
        variance = cv_a ** 2 + cv_b ** 2
    return math.sqrt(variance)


def planning_report(plan, *, repeats=None, cv=None, family_size=None,
                    alpha=None, target_power=None,
                    alternative=None) -> dict:
    """Prospective repeat budget and the precision it delivers.

    The estimand is a ratio of medians, analysed on the log scale. The
    planning SD is the per-repeat log-ratio SD from
    :func:`ratio_log_se_unit`; the required repeat count is the ordinary
    normal-approximation sample size with the family-corrected alpha.
    """
    plan = get_plan(plan) if isinstance(plan, str) else plan
    if not isinstance(plan, Mapping):
        raise TimingDesignError('plan must be a plan id or a mapping')
    alpha_value = _need_number(
        plan.get('alpha', DEFAULT_ALPHA) if alpha is None else alpha, 'alpha')
    if not 0.0 < alpha_value < 1.0:
        raise TimingDesignError(f'alpha must be in (0, 1), got {alpha_value!r}')
    family = _need_int(
        plan.get('planned_family_size', 1)
        if family_size is None else family_size, 'family_size', minimum=1)
    alternative = (plan.get('alternative', 'two-sided')
                   if alternative is None else alternative)
    if alternative not in ('two-sided', 'greater', 'less'):
        raise TimingDesignError(
            f'invalid alternative {alternative!r}')
    ratio = _need_number(
        plan.get('sesoi_ratio', DEFAULT_SESOI_RATIO), 'sesoi_ratio',
        positive=True)
    if ratio <= 1.0:
        raise TimingDesignError(
            f'sesoi_ratio must be > 1 (a ratio below 1 is inverted), '
            f'got {ratio!r}')
    cv_value = _need_number(
        plan.get('assumed_cv', DEFAULT_ASSUMED_CV) if cv is None else cv,
        'assumed_cv', positive=True)
    target = _need_number(
        plan.get('target_power', DEFAULT_TARGET_POWER)
        if target_power is None else target_power, 'target_power',
        positive=True)
    if not 0.0 < target < 1.0:
        raise TimingDesignError(
            f'target_power must be in (0, 1), got {target!r}')
    repeats_value = _need_int(
        plan.get('planned_repeats', 2) if repeats is None else repeats,
        'repeats', minimum=0)
    se_unit = ratio_log_se_unit(cv_value)
    corrected = _planning_alpha(alpha_value, family, alternative)
    if corrected <= 0.0:
        required = None
        power = None
        mde_ratio = None
        status = 'not_estimable'
    else:
        dz = math.log(ratio) / se_unit
        required = seeds_required(
            dz, alpha=corrected, power=target, alternative=alternative)
        if required is not None:
            required = max(2, int(required))
        mde_log = min_detectable_effect(
            repeats_value, se_unit, alpha=corrected, power=target,
            alternative=alternative)
        mde_ratio = _safe_exp(mde_log)
        power = power_at_effect(
            repeats_value, se_unit, math.log(ratio),
            alpha=corrected, alternative=alternative)
        if (required is not None and repeats_value >= required
                and power is not None and power >= target - 1e-9):
            status = 'confirmatory_ready'
        else:
            status = 'underpowered_at_sesoi'
    return {
        'sesoi_ratio': ratio,
        'log_sesoi': math.log(ratio),
        'assumed_cv': cv_value,
        'log_ratio_sd_per_repeat': se_unit,
        'alpha_family_wise': alpha_value,
        'family_size': family,
        'alpha_per_test_tail': (alpha_value / family
                                if alternative != 'two-sided'
                                else alpha_value / (2.0 * family)),
        'alternative': alternative,
        'target_power': target,
        'repeats': int(repeats_value),
        'required_repeats': required,
        'planned_power': power,
        'planned_mde_ratio': mde_ratio,
        'power_gap': (None if power is None else target - power),
        'status': status,
        'method': ('normal approximation on log medians: '
                   'n = ceil(((z_crit + z_power) * sd_log / log(SESOI))^2) '
                   'with sd_log = sqrt(cv_a^2 + cv_b^2) for independently '
                   'timed cells'),
        'note': ('Planning only. The ratio interval from a run should be '
                 'computed from the raw trials (timing_stats) or, when '
                 'only summaries exist, labelled as a normal log-ratio '
                 'approximation. A confirmatory claim also requires a '
                 'balanced measurement order and a warmup that is excluded '
                 'from the timed trials.'),
    }


def balanced_schedule(names, blocks, *, seed=DEFAULT_SCHEDULE_SEED):
    """Deterministic randomized complete-block order with rotation.

    Each block is a permutation of ``names``. The base permutation is
    shuffled with a local ``random.Random(seed)`` and then rotated by one
    position per block, so over ``len(names)`` consecutive blocks every
    operator occupies every position exactly once. This removes the
    "operator X is always measured first" confound without touching the
    measurement code or the cell definitions.
    """
    ops = list(names)
    if not ops:
        raise TimingDesignError('names must contain at least one operator')
    if len(set(ops)) != len(ops):
        raise TimingDesignError('names must be unique')
    for op in ops:
        if not isinstance(op, str) or not op:
            raise TimingDesignError(
                f'operator names must be non-empty strings, got {op!r}')
    n_blocks = _need_int(blocks, 'blocks', minimum=1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TimingDesignError(f'seed must be an int, got {seed!r}')
    base = list(ops)
    random.Random(seed).shuffle(base)
    schedule = []
    for block in range(n_blocks):
        shift = block % len(base)
        schedule.append(base[shift:] + base[:shift])
    return schedule


def _schedule_balance(schedule) -> dict:
    """Summarize whether a schedule balances operator positions."""
    if not isinstance(schedule, (list, tuple)) or not schedule:
        return {'n_blocks': 0, 'n_ops': 0, 'complete_permutations': False,
                'balanced': False, 'max_position_span': None,
                'max_position_share': None,
                'per_operator_distinct_positions': {}}
    blocks = []
    for block in schedule:
        if not isinstance(block, (list, tuple)):
            return {'n_blocks': len(schedule), 'n_ops': 0,
                    'complete_permutations': False, 'balanced': False,
                    'max_position_span': None, 'max_position_share': None,
                    'per_operator_distinct_positions': {}}
        blocks.append(list(block))
    op_set = set()
    for block in blocks:
        op_set.update(block)
    n_ops = len(op_set)
    complete = (n_ops > 0
                and all(len(block) == n_ops
                        and len(set(block)) == n_ops
                        and set(block) == op_set
                        for block in blocks))
    positions: dict[str, dict[int, int]] = {op: {} for op in op_set}
    if complete:
        for block in blocks:
            for position, op in enumerate(block):
                positions[op][position] = positions[op].get(position, 0) + 1
    distinct = {
        op: sum(1 for count in positions[op].values() if count)
        for op in op_set
    }
    spans = []
    for op in op_set:
        counts = [positions[op].get(position, 0) for position in range(n_ops)]
        spans.append(max(counts) - min(counts))
    max_span = max(spans) if spans else None
    expected_distinct = min(len(blocks), n_ops) if n_ops else 0
    balanced = bool(complete and len(blocks) >= 2 and max_span is not None
                    and max_span <= 1
                    and all(value == expected_distinct
                            for value in distinct.values()))
    if complete:
        max_position_share = max(
            (count for counts in positions.values()
             for count in counts.values()), default=0) / len(blocks)
    else:
        max_position_share = None
    return {
        'n_blocks': len(blocks),
        'n_ops': n_ops,
        'complete_permutations': complete,
        'balanced': balanced,
        'max_position_span': max_span,
        'max_position_share': max_position_share,
        'per_operator_distinct_positions': dict(sorted(distinct.items())),
    }


def _cell_float(value, name, *, positive=False):
    if value is None:
        return None
    if not _is_number(value):
        raise TimingDesignError(f'{name} must be finite, got {value!r}')
    value = float(value)
    if positive and value <= 0.0:
        raise TimingDesignError(f'{name} must be > 0, got {value!r}')
    return value


def _normalize_cell(workload, op, value, *, cv=None, n=None,
                    warmup=None, schedule_block=None, order_position=None,
                    device=None) -> dict:
    if not isinstance(op, str) or not op:
        raise TimingDesignError(f'op must be a non-empty string, got {op!r}')
    value_ms = _cell_float(value, f'{op} value_ms', positive=True)
    cv_value = _cell_float(cv, f'{op} cv')
    if cv_value is not None and cv_value < 0.0:
        raise TimingDesignError(f'{op} cv must be >= 0, got {cv_value!r}')
    n_value = None
    if n is not None:
        n_value = _need_int(n, f'{op} n', minimum=1)
    warmup_value = None
    if warmup is not None:
        warmup_value = _need_int(warmup, f'{op} warmup', minimum=0)
    block_value = None
    if schedule_block is not None:
        block_value = _need_int(schedule_block, f'{op} schedule_block',
                                minimum=0)
    position_value = None
    if order_position is not None:
        position_value = _need_int(order_position, f'{op} order_position',
                                   minimum=0)
    return {
        'workload': workload,
        'op': op,
        'value_ms': value_ms,
        'cv': cv_value,
        'n': n_value,
        'warmup': warmup_value,
        'schedule_block': block_value,
        'order_position': position_value,
        'device': str(device) if device is not None else None,
    }


def cells_from_benchmark_rows(rows) -> list[dict]:
    """Normalize Exp 2 result rows into timing-design cells."""
    if isinstance(rows, Mapping):
        raise TimingDesignError('rows must be an iterable of row mappings')
    cells = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TimingDesignError(f'row[{index}] must be a mapping')
        if row.get('error') is not None:
            continue
        if 'T' not in row or 'op' not in row or 'time_ms' not in row:
            continue
        cells.append(_normalize_cell(
            row.get('T'), row.get('op'), row.get('time_ms'),
            cv=row.get('time_cv'), n=row.get('time_n'),
            warmup=row.get('warmup'),
            schedule_block=row.get('schedule_block'),
            order_position=row.get('order_position'),
            device=row.get('device')))
    return cells


def cells_from_decoding_rows(rows) -> list[dict]:
    """Normalize Exp 6 result rows into timing-design cells."""
    if isinstance(rows, Mapping):
        raise TimingDesignError('rows must be an iterable of row mappings')
    cells = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TimingDesignError(f'row[{index}] must be a mapping')
        if row.get('error') is not None:
            continue
        if ('prefill_len' not in row or 'op' not in row
                or 'median_decode_ms_per_token' not in row):
            continue
        # The independent replicate is the full prefill+decode trial, not
        # the per-token sample: tokens within a trial share one cache and
        # thermal state. Prefer the across-trial CV/count; older rows only
        # have ``decode_n`` (token samples), so fall back to ``repeats`` to
        # avoid treating correlated token timings as independent trials.
        cells.append(_normalize_cell(
            row.get('prefill_len'), row.get('op'),
            row.get('median_decode_ms_per_token'),
            cv=row.get('decode_trial_cv', row.get('decode_cv')),
            n=row.get('decode_trial_n', row.get('repeats')),
            warmup=row.get('warmup'),
            schedule_block=row.get('schedule_block'),
            order_position=row.get('order_position'),
            device=row.get('device')))
    return cells


def result_plan_id(path: str) -> str:
    """Identify which timing plan wrote a result file (or raw row list)."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    rows = None
    if isinstance(payload, Mapping):
        kind = payload.get('kind')
        if kind in TIMING_PLANS:
            return kind
        plan_id = payload.get('plan_id')
        if plan_id in TIMING_PLANS:
            return plan_id
        rows = payload.get('results')
        if not isinstance(rows, list):
            raise TimingDesignError(
                f'{path}: envelope has no results list')
    elif isinstance(payload, list):
        rows = payload
    else:
        raise TimingDesignError(f'{path}: expected a JSON list or envelope')
    if rows and isinstance(rows[0], Mapping):
        if 'prefill_len' in rows[0]:
            return 'exp6_decoding'
        if 'T' in rows[0] or 'time_ms' in rows[0]:
            return 'exp2_benchmark'
    raise TimingDesignError(f'{path}: cannot identify the timing schema')


def load_cells(path: str) -> list[dict]:
    """Load cells from an Exp 2 or Exp 6 result file."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        kind = payload.get('kind')
        rows = payload.get('results')
        if not isinstance(rows, list):
            raise TimingDesignError(
                f'{path}: envelope has no results list')
        if kind == 'exp2_benchmark':
            return cells_from_benchmark_rows(rows)
        if kind == 'exp6_decoding':
            return cells_from_decoding_rows(rows)
        if rows and isinstance(rows[0], Mapping):
            if 'T' in rows[0] and 'time_ms' in rows[0]:
                return cells_from_benchmark_rows(rows)
            if 'prefill_len' in rows[0]:
                return cells_from_decoding_rows(rows)
        raise TimingDesignError(f'{path}: cannot identify the timing schema')
    if isinstance(payload, list):
        if payload and isinstance(payload[0], Mapping):
            if 'T' in payload[0] and 'time_ms' in payload[0]:
                return cells_from_benchmark_rows(payload)
            if 'prefill_len' in payload[0]:
                return cells_from_decoding_rows(payload)
        raise TimingDesignError(f'{path}: cannot identify the timing schema')
    raise TimingDesignError(f'{path}: expected a JSON list or envelope')


def _normalize_raw_cell(cell, index) -> dict:
    if not isinstance(cell, Mapping):
        raise TimingDesignError(f'raw cell[{index}] must be a mapping')
    op = cell.get('op')
    if not isinstance(op, str) or not op:
        raise TimingDesignError(f'raw cell[{index}].op must be a string')
    workload = cell.get('workload', cell.get('T', cell.get('prefill_len')))
    if workload is None:
        raise TimingDesignError(
            f'raw cell[{index}] has no workload/T/prefill_len')
    samples = cell.get('samples_ms', cell.get('times_ms'))
    if isinstance(samples, Mapping) or not isinstance(samples, (list, tuple)):
        raise TimingDesignError(
            f'raw cell[{index}].samples_ms must be a list')
    clean = []
    for value in samples:
        number = _need_number(value, f'raw cell[{index}] sample')
        if number < 0.0:
            raise TimingDesignError(
                f'raw cell[{index}] samples must be >= 0')
        clean.append(number)
    if not clean:
        raise TimingDesignError(f'raw cell[{index}] has no samples')
    return {
        'workload': workload,
        'op': op,
        'samples_ms': clean,
        'device': (str(cell['device']) if cell.get('device') is not None
                   else None),
    }


def load_raw_cells(path: str) -> list[dict]:
    """Load per-trial latency samples from a ``timing_raw`` side file."""
    with open(path, encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        kind = payload.get('kind')
        if kind is not None and kind != 'timing_raw':
            raise TimingDesignError(
                f'{path}: expected kind timing_raw, got {kind!r}')
        rows = payload.get('results')
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TimingDesignError(f'{path}: raw file has no results list')
    return [_normalize_raw_cell(row, index) for index, row in enumerate(rows)]


def raw_manifest(plan_id: str, cells, *, warmup=None,
                 schedule_id=None) -> dict:
    """Build the archived per-trial table that backs a timing run."""
    get_plan(plan_id)
    if warmup is not None:
        warmup = _need_int(warmup, 'warmup', minimum=0)
    if schedule_id is not None and not isinstance(schedule_id, str):
        raise TimingDesignError(
            f'schedule_id must be a string, got {schedule_id!r}')
    normalized = [_normalize_raw_cell(cell, index)
                  for index, cell in enumerate(cells)]
    seen = set()
    for cell in normalized:
        key = (str(cell['workload']), cell['op'])
        if key in seen:
            raise TimingDesignError(f'duplicate raw timing cell {key!r}')
        seen.add(key)
    return {
        'schema_version': '1.0',
        'kind': 'timing_raw',
        'plan_id': plan_id,
        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'metadata': {
            'warmup': warmup,
            'schedule_id': schedule_id,
            'n_cells': len(normalized),
            'note': ('Per-trial timings in milliseconds, with warmup '
                     'already excluded. This file is the input to '
                     'timing_stats.ratio_of_medians_ci and to '
                     'timing_design.raw_contrast_rows.'),
        },
        'results': normalized,
    }


def write_raw_manifest(path: str, *, plan_id=None, cells=None,
                       manifest=None) -> str:
    if manifest is None:
        if plan_id is None or cells is None:
            raise TimingDesignError(
                'write_raw_manifest needs a manifest or plan_id+cells')
        manifest = raw_manifest(plan_id, cells)
    payload = json.dumps(manifest, indent=2, sort_keys=True,
                         allow_nan=False) + '\n'
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.timing_raw_', suffix='.json',
                               dir=parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def raw_contrast_rows(plan_id: str, cells, *, family_size=None,
                      alpha=None, alternative=None,
                      n_boot=DEFAULT_BOOTSTRAP_REPS) -> list[dict]:
    """Bootstrap ratio intervals from archived per-trial timings."""
    plan = get_plan(plan_id)
    raw = [_normalize_raw_cell(cell, index)
           for index, cell in enumerate(cells)]
    by_key = {}
    for cell in raw:
        key = (str(cell['workload']), cell['op'])
        if key in by_key:
            raise TimingDesignError(f'duplicate raw timing cell {key!r}')
        by_key[key] = cell
    workloads = []
    for cell in raw:
        if cell['workload'] not in workloads:
            workloads.append(cell['workload'])
    alpha_value = _need_number(
        plan.get('alpha', DEFAULT_ALPHA) if alpha is None else alpha, 'alpha')
    if not 0.0 < alpha_value < 1.0:
        raise TimingDesignError(f'alpha must be in (0, 1), got {alpha_value!r}')
    family = _need_int(
        plan.get('planned_family_size', 1)
        if family_size is None else family_size, 'family_size', minimum=1)
    alternative = (plan.get('alternative', 'two-sided')
                   if alternative is None else alternative)
    if alternative not in ('two-sided', 'greater', 'less'):
        raise TimingDesignError(f'invalid alternative {alternative!r}')
    corrected = _planning_alpha(alpha_value, family, alternative)
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise TimingDesignError(f'n_boot must be a positive int, got {n_boot!r}')
    rows = []
    for workload in workloads:
        for contrast in plan['primary_contrasts']:
            left = by_key.get((str(workload), contrast['numerator']))
            right = by_key.get((str(workload), contrast['denominator']))
            missing = []
            if left is None:
                missing.append(contrast['numerator'])
            if right is None:
                missing.append(contrast['denominator'])
            if missing:
                rows.append({
                    'workload': workload,
                    'contrast': contrast['id'],
                    'numerator': contrast['numerator'],
                    'denominator': contrast['denominator'],
                    'missing': missing,
                    'ratio_median': None,
                    'ratio_ci95_low': None,
                    'ratio_ci95_high': None,
                    'ci_excludes_one': None,
                    'method': 'raw contrast cell(s) missing',
                })
                continue
            stats = ratio_of_medians_ci(
                left['samples_ms'], right['samples_ms'],
                paired=False, n_boot=n_boot, alpha=corrected, seed=0)
            low = stats['ratio_ci95_low']
            high = stats['ratio_ci95_high']
            if low is None or high is None:
                excludes = None
            elif alternative == 'greater':
                excludes = low > 1.0
            elif alternative == 'less':
                excludes = high < 1.0
            else:
                excludes = low > 1.0 or high < 1.0
            rows.append({
                'workload': workload,
                'contrast': contrast['id'],
                'numerator': contrast['numerator'],
                'denominator': contrast['denominator'],
                'missing': [],
                'ratio_median': stats['ratio_median'],
                'ratio_ci95_low': low,
                'ratio_ci95_high': high,
                'ci_excludes_one': excludes,
                'n_numerator': stats['n_a'],
                'n_denominator': stats['n_b'],
                'paired': False,
                'ci_method': stats['ci_method'],
                'bootstrap_reps': stats['bootstrap_reps'],
                'alpha_per_test': corrected,
                'method': 'percentile bootstrap on raw per-trial timings',
            })
    return rows


def _contrast_sesoi(plan, contrast) -> float:
    ratio = contrast.get('sesoi_ratio', plan['sesoi_ratio'])
    ratio = _need_number(ratio, 'contrast sesoi_ratio', positive=True)
    if ratio <= 1.0:
        raise TimingDesignError(
            f'contrast sesoi_ratio must be > 1, got {ratio!r}')
    return ratio


def pilot_report(plan_id: str, cells, *, family_size=None) -> dict:
    """Turn a pilot timing table into a repeat-budget recommendation.

    The pilot's per-cell CVs are noisy, so the report returns the pooled
    (median) and conservative (maximum) recommendations. As with the
    seed-power pilot, the same rows can never be treated as the
    confirmatory run.
    """
    plan = get_plan(plan_id)
    cell_list = list(cells)
    cvs = []
    for cell in cell_list:
        if isinstance(cell, Mapping):
            cv = cell.get('cv')
            if cv is not None:
                cv_value = _need_number(cv, 'cell cv')
                if cv_value < 0.0:
                    raise TimingDesignError(
                        f'cell cv must be >= 0, got {cv_value!r}')
                cvs.append(cv_value)
    if not cvs:
        raise TimingDesignError('pilot cells carry no CV values')
    cv_median = statistics.median(cvs)
    cv_max = max(cvs)
    if cv_max <= 0.0:
        raise TimingDesignError(
            'pilot CVs are all zero; precision cannot be estimated from '
            'a noiseless table')
    pooled_cv = (cv_median if cv_median > 0.0
                 else float(plan.get('assumed_cv', DEFAULT_ASSUMED_CV)))
    pooled = planning_report(plan, cv=pooled_cv, family_size=family_size)
    conservative = planning_report(plan, cv=cv_max, family_size=family_size)
    candidates = [value for value in (
        pooled['required_repeats'], conservative['required_repeats'])
        if value is not None]
    return {
        'plan_id': plan_id,
        'n_cells': len(cvs),
        'cv_median': cv_median,
        'cv_max': cv_max,
        'required_repeats_pooled_cv': pooled['required_repeats'],
        'required_repeats_conservative': conservative['required_repeats'],
        'recommended_repeats': (max(candidates) if candidates else None),
        'pilot_is_confirmatory': False,
        'note': ('A pilot chooses the repeat budget; it cannot also supply '
                 'the confirmatory evidence. Re-run after the budget is '
                 'fixed and audit the new table.'),
    }


def _derive_schedule(cells):
    blocks: dict[int, list[tuple[int, str]]] = {}
    for cell in cells:
        block = cell.get('schedule_block')
        position = cell.get('order_position')
        if block is None or position is None:
            return None
        blocks.setdefault(int(block), []).append((int(position), cell['op']))
    if not blocks:
        return None
    schedule = []
    for block in sorted(blocks):
        ordered = sorted(blocks[block])
        if len({position for position, _op in ordered}) != len(ordered):
            return None
        schedule.append([op for _position, op in ordered])
    return schedule


def audit_run(plan_id: str, cells, *, repeats=None, warmup=None,
              schedule=None, family_size=None, alpha=None,
              device=None, workloads=None) -> dict:
    """Label a timing table against its pre-registered measurement plan.

    Passing ``workloads`` (the intended sequence/prefill grid) makes the
    audit distinguish "all registered contrasts are present" from "the run
    silently lost an entire workload block"; without it, the expected grid
    is inferred from the rows that survived.
    """
    plan = get_plan(plan_id)
    if isinstance(cells, Mapping):
        raise TimingDesignError('cells must be an iterable of cell mappings')
    cell_list = list(cells)
    valid = []
    invalid = []
    seen = set()
    for index, cell in enumerate(cell_list):
        if not isinstance(cell, Mapping):
            invalid.append({'index': index, 'reason': 'not a mapping'})
            continue
        op = cell.get('op')
        workload = cell.get('workload')
        if not isinstance(op, str) or not op:
            invalid.append({'index': index, 'reason': 'invalid op'})
            continue
        try:
            normalized = _normalize_cell(
                workload, op, cell.get('value_ms'), cv=cell.get('cv'),
                n=cell.get('n'), warmup=cell.get('warmup'),
                schedule_block=cell.get('schedule_block'),
                order_position=cell.get('order_position'),
                device=cell.get('device'))
        except TimingDesignError as exc:
            invalid.append({'index': index, 'reason': str(exc)})
            continue
        key = (str(workload), op)
        if key in seen:
            invalid.append({'index': index, 'reason': 'duplicate cell'})
            continue
        seen.add(key)
        valid.append(normalized)
    if not valid:
        return {
            'plan_id': plan_id,
            'status': 'invalid',
            'claims_allowed': [],
            'design_warnings': ['no valid timing cells'],
            'invalid_cells': invalid,
            'missing_cells': [],
            'missing_workloads': [],
            'plan_sha256': plan_sha256(plan),
            'plan_fingerprint_scope': PLAN_FINGERPRINT_SCOPE,
            'plan_definition': copy.deepcopy(plan),
        }

    observed_workloads = []
    for cell in valid:
        if cell['workload'] not in observed_workloads:
            observed_workloads.append(cell['workload'])
    if workloads is None:
        expected_workloads = observed_workloads
    else:
        expected_workloads = []
        seen_workloads = set()
        for workload in workloads:
            key = str(workload)
            if key in seen_workloads:
                continue
            seen_workloads.add(key)
            expected_workloads.append(workload)
        if not expected_workloads:
            raise TimingDesignError(
                'workloads must contain at least one expected block')
    observed_keys = {str(cell['workload']) for cell in valid}
    missing_workloads = [workload for workload in expected_workloads
                         if str(workload) not in observed_keys]
    workloads = expected_workloads
    ops = sorted({cell['op'] for cell in valid})
    n_values = [cell['n'] for cell in valid if cell['n'] is not None]
    observed_repeats = (min(n_values) if repeats is None and n_values
                        else repeats)
    if observed_repeats is not None:
        _need_int(observed_repeats, 'repeats', minimum=1)
    warmup_values = [cell['warmup'] for cell in valid
                     if cell['warmup'] is not None]
    observed_warmup = (warmup if warmup is not None
                       else (min(warmup_values) if warmup_values else None))
    if observed_warmup is not None:
        _need_int(observed_warmup, 'warmup', minimum=0)
    devices = sorted({cell['device'] for cell in valid
                      if cell['device'] is not None})
    if device is not None and devices and str(device) not in devices:
        devices.append(str(device))

    derived_schedule = schedule if schedule is not None else _derive_schedule(valid)
    schedule_info = _schedule_balance(derived_schedule)
    contrasts = plan['primary_contrasts']
    family = _need_int(
        plan['planned_family_size'] if family_size is None else family_size,
        'family_size', minimum=1)
    alpha_value = _need_number(
        plan['alpha'] if alpha is None else alpha, 'alpha')
    if not 0.0 < alpha_value < 1.0:
        raise TimingDesignError(f'alpha must be in (0,1), got {alpha_value!r}')
    alternative = plan.get('alternative', 'two-sided')
    corrected = _planning_alpha(alpha_value, family, alternative)
    z_crit = NormalDist().inv_cdf(1.0 - (
        alpha_value / family if alternative != 'two-sided'
        else alpha_value / (2.0 * family)))

    by_key = {(str(cell['workload']), cell['op']): cell for cell in valid}
    contrast_rows = []
    missing = []
    for workload in workloads:
        for contrast in contrasts:
            numerator = contrast['numerator']
            denominator = contrast['denominator']
            left = by_key.get((str(workload), numerator))
            right = by_key.get((str(workload), denominator))
            absent = [op for op, cell in (
                (numerator, left), (denominator, right)) if cell is None]
            if absent:
                missing.append({'workload': workload,
                                'contrast': contrast['id'],
                                'missing': absent})
                continue
            sesoi = _contrast_sesoi(plan, contrast)
            n_pair = min(value for value in (left['n'], right['n'])
                         if value is not None) \
                if (left['n'] is not None and right['n'] is not None) else None
            if n_pair is None:
                n_pair = observed_repeats
            ratio = (left['value_ms'] / right['value_ms']
                     if right['value_ms'] else None)
            log_ratio = _safe_log(ratio)
            if (left['cv'] is None or right['cv'] is None
                    or n_pair is None or n_pair < 2 or corrected <= 0.0):
                contrast_rows.append({
                    'workload': workload,
                    'contrast': contrast['id'],
                    'numerator': numerator,
                    'denominator': denominator,
                    'observed_ratio': ratio,
                    'n_repeats': n_pair,
                    'cv_numerator': left['cv'],
                    'cv_denominator': right['cv'],
                    'sesoi_ratio': sesoi,
                    'required_repeats': None,
                    'mde_ratio': None,
                    'powered_at_sesoi': False,
                    'ci95_ratio_low': None,
                    'ci95_ratio_high': None,
                    'ci_excludes_one': None,
                    'claim_allowed': False,
                    'method': 'precision unavailable (CV or repeats missing)',
                })
                continue
            # Zero observed CV is valid data (identical timings), but it
            # cannot support an infinite-precision claim. One zero CV is
            # fine as long as the other cell has observed variation; two
            # zeros leave the log-ratio SE at exactly zero, so the row is
            # reported as precision-unavailable rather than powered.
            se_unit = math.sqrt(left['cv'] ** 2 + right['cv'] ** 2)
            if se_unit <= 0.0:
                contrast_rows.append({
                    'workload': workload,
                    'contrast': contrast['id'],
                    'numerator': numerator,
                    'denominator': denominator,
                    'observed_ratio': ratio,
                    'n_repeats': n_pair,
                    'cv_numerator': left['cv'],
                    'cv_denominator': right['cv'],
                    'sesoi_ratio': sesoi,
                    'required_repeats': None,
                    'mde_ratio': None,
                    'powered_at_sesoi': False,
                    'ci95_ratio_low': None,
                    'ci95_ratio_high': None,
                    'ci_excludes_one': None,
                    'claim_allowed': False,
                    'method': ('precision unavailable: both cells report '
                               'zero observed CV'),
                })
                continue
            required = seeds_required(
                math.log(sesoi) / se_unit, alpha=corrected,
                power=plan['target_power'], alternative=alternative)
            if required is not None:
                required = max(2, int(required))
            mde_log = min_detectable_effect(
                n_pair, se_unit, alpha=corrected, power=plan['target_power'],
                alternative=alternative)
            mde_ratio = _safe_exp(mde_log)
            log_se = se_unit / math.sqrt(n_pair)
            low_log = (log_ratio - z_crit * log_se
                       if log_ratio is not None else None)
            high_log = (log_ratio + z_crit * log_se
                        if log_ratio is not None else None)
            low = _safe_exp(low_log)
            high = _safe_exp(high_log)
            if low is None or high is None:
                excludes = None
            elif alternative == 'greater':
                excludes = low > 1.0
            elif alternative == 'less':
                excludes = high < 1.0
            else:
                excludes = low > 1.0 or high < 1.0
            powered = (mde_ratio is not None
                       and mde_ratio <= sesoi + 1e-12)
            contrast_rows.append({
                'workload': workload,
                'contrast': contrast['id'],
                'numerator': numerator,
                'denominator': denominator,
                'observed_ratio': ratio,
                'n_repeats': n_pair,
                'cv_numerator': left['cv'],
                'cv_denominator': right['cv'],
                'sesoi_ratio': sesoi,
                'required_repeats': required,
                'mde_ratio': mde_ratio,
                'powered_at_sesoi': bool(powered),
                'ci95_ratio_low': low,
                'ci95_ratio_high': high,
                'ci_excludes_one': excludes,
                'claim_allowed': bool(powered and excludes),
                'method': 'normal log-ratio from per-cell CV and repeats',
            })

    warnings = []
    if invalid:
        warnings.append(f'{len(invalid)} invalid cell(s)')
    if missing:
        warnings.append(f'{len(missing)} contrast cell(s) missing')
    expected_rows = len(workloads) * len(contrasts)
    complete = (not missing and not missing_workloads
                and len(contrast_rows) == expected_rows)
    if missing_workloads:
        warnings.append(
            f'{len(missing_workloads)} expected workload block(s) missing: '
            f'{missing_workloads}')
    if not complete:
        warnings.append(
            f'expected {expected_rows} contrast rows, got '
            f'{len(contrast_rows)}')
    if n_values and len(set(n_values)) > 1:
        warnings.append('repeat count differs across cells')
    if (observed_repeats is not None
            and plan['planned_repeats'] is not None):
        if observed_repeats < plan['planned_repeats']:
            warnings.append(
                f"observed repeats ({observed_repeats}) below the "
                f"pre-registered budget ({plan['planned_repeats']})")
    planned_family = plan.get('planned_family_size')
    if planned_family is not None and family < planned_family:
        warnings.append(
            f'analysis family ({family}) is smaller than the registered '
            f'plan ({planned_family}); a reduced workload grid is not the '
            f'confirmatory design')
    if observed_warmup is None:
        warnings.append('warmup not recorded')
    elif observed_warmup < plan['warmup']:
        warnings.append(
            f"warmup ({observed_warmup}) below the plan ({plan['warmup']})")
    if not schedule_info['balanced']:
        warnings.append('operator order is not a balanced schedule')
    if len(devices) > 1:
        warnings.append(f'multiple devices in one table: {devices}')
    if any(row['cv_numerator'] is None or row['cv_denominator'] is None
           for row in contrast_rows):
        warnings.append('CV unavailable for at least one cell')
    if any(cell['cv'] == 0.0 for cell in valid):
        warnings.append('zero observed CV for at least one cell; precision '
                        'is a lower bound, not infinite')
    if contrast_rows and not all(row['powered_at_sesoi']
                                 for row in contrast_rows):
        warnings.append('realized precision is below the SESOI for at least '
                        'one contrast')

    if not complete or invalid:
        status = 'incomplete_cells'
    elif n_values and len(set(n_values)) > 1:
        status = 'repeats_inconsistent'
    elif not schedule_info['balanced']:
        status = 'order_confounded'
    elif observed_warmup is None or observed_warmup < plan['warmup']:
        status = 'unplanned_warmup'
    elif len(devices) > 1:
        status = 'device_mixed'
    elif (observed_repeats is not None
          and observed_repeats < plan['planned_repeats']):
        status = 'screening'
    elif planned_family is not None and family < planned_family:
        status = 'screening'
    elif not contrast_rows or not all(row['powered_at_sesoi']
                                      for row in contrast_rows):
        status = 'screening'
    else:
        status = 'confirmatory'

    mdes = [row['mde_ratio'] for row in contrast_rows
            if row['mde_ratio'] is not None]
    observed_cvs = [cell['cv'] for cell in valid if cell['cv'] is not None]
    claims = []
    if contrast_rows:
        claims.append('descriptive_relative_latency')
    if status == 'confirmatory':
        claims.append('relative_latency_ratio_at_sesoi')
    return {
        'plan_id': plan_id,
        'status': status,
        'claims_allowed': claims,
        'design_warnings': warnings,
        'analysis_population': {
            'n_cells': len(valid),
            'n_workloads': len(workloads),
            'n_operators': len(ops),
            'n_contrasts': len(contrasts),
            'expected_workloads': workloads,
            'missing_workloads': missing_workloads,
            'expected_contrast_rows': expected_rows,
            'observed_contrast_rows': len(contrast_rows),
            'complete': complete,
        },
        'observed_repeats': observed_repeats,
        'planned_repeats': plan['planned_repeats'],
        'observed_warmup': observed_warmup,
        'planned_warmup': plan['warmup'],
        'observed_cv_median': (statistics.median(observed_cvs)
                               if observed_cvs else None),
        'observed_cv_max': (max(observed_cvs) if observed_cvs else None),
        'max_mde_ratio': (max(mdes) if mdes else None),
        'schedule': schedule_info,
        'family_size': family,
        'alpha_family_wise': alpha_value,
        'devices': devices,
        'contrast_rows': contrast_rows,
        'missing_cells': missing,
        'invalid_cells': invalid,
        'planning_at_observed_repeats': (
            planning_report(plan, repeats=observed_repeats,
                            family_size=family)
            if observed_repeats is not None else None),
        'planning_at_planned_repeats': planning_report(
            plan, repeats=plan['planned_repeats'], family_size=family),
        'plan_sha256': plan_sha256(plan),
        'plan_fingerprint_scope': PLAN_FINGERPRINT_SCOPE,
        'plan_definition': copy.deepcopy(plan),
    }


def validate_plan(plan) -> list[str]:
    errors = []
    if not isinstance(plan, Mapping):
        return ['plan must be a mapping']
    plan_id = plan.get('plan_id')
    if not isinstance(plan_id, str) or not plan_id:
        errors.append('plan_id must be a non-empty string')
    for key in ('title', 'runner', 'question', 'design', 'unit', 'endpoint',
                'analysis', 'decision_rule', 'stopping_rule',
                'allocation_rule'):
        if not isinstance(plan.get(key), str) or not plan.get(key):
            errors.append(f'{key} must be a non-empty string')
    result_files = plan.get('result_files')
    if (not isinstance(result_files, (list, tuple)) or not result_files
            or not all(isinstance(path, str) and path
                       for path in result_files)):
        errors.append('result_files must be a non-empty list of strings')
    if plan.get('schedule') not in _SCHEDULES:
        errors.append(f"schedule must be one of {_SCHEDULES}, got "
                      f"{plan.get('schedule')!r}")
    try:
        alpha = _need_number(plan.get('alpha'), 'alpha')
        if not 0.0 < alpha < 1.0:
            raise TimingDesignError(f'alpha must be in (0,1), got {alpha!r}')
        _need_number(plan.get('assumed_cv'), 'assumed_cv', positive=True)
        ratio = _need_number(plan.get('sesoi_ratio'), 'sesoi_ratio',
                             positive=True)
        if ratio <= 1.0:
            raise TimingDesignError(
                f'sesoi_ratio must be > 1, got {ratio!r}')
        target = _need_number(plan.get('target_power'), 'target_power',
                              positive=True)
        if not 0.0 < target < 1.0:
            raise TimingDesignError(
                f'target_power must be in (0,1), got {target!r}')
        _need_int(plan.get('planned_family_size'), 'planned_family_size',
                  minimum=1)
        _need_int(plan.get('screening_repeats'), 'screening_repeats',
                  minimum=1)
        _need_int(plan.get('planned_repeats'), 'planned_repeats', minimum=2)
        _need_int(plan.get('warmup'), 'warmup', minimum=0)
    except TimingDesignError as exc:
        errors.append(str(exc))
    if plan.get('alternative') not in ('two-sided', 'greater', 'less'):
        errors.append(
            f"alternative must be 'two-sided'|'greater'|'less', got "
            f"{plan.get('alternative')!r}")
    contrasts = plan.get('primary_contrasts')
    if not isinstance(contrasts, (list, tuple)) or not contrasts:
        errors.append('primary_contrasts must be a non-empty list')
    else:
        seen = set()
        for index, contrast in enumerate(contrasts):
            if not isinstance(contrast, Mapping):
                errors.append(f'primary_contrasts[{index}] must be a mapping')
                continue
            contrast_id = contrast.get('id')
            if not isinstance(contrast_id, str) or not contrast_id:
                errors.append(
                    f'primary_contrasts[{index}].id must be a string')
            elif contrast_id in seen:
                errors.append(f'duplicate contrast id {contrast_id!r}')
            else:
                seen.add(contrast_id)
            for key in ('numerator', 'denominator'):
                value = contrast.get(key)
                if not isinstance(value, str) or not value:
                    errors.append(
                        f'primary_contrasts[{index}].{key} must be a '
                        'non-empty string')
            if contrast.get('numerator') == contrast.get('denominator'):
                errors.append(
                    f'primary_contrasts[{index}] compares an operator with '
                    'itself')
            try:
                _contrast_sesoi(plan, contrast)
            except TimingDesignError as exc:
                errors.append(f'primary_contrasts[{index}]: {exc}')
    if not errors:
        try:
            report = planning_report(plan)
        except TimingDesignError as exc:
            errors.append(str(exc))
        else:
            required = report['required_repeats']
            planned = plan['planned_repeats']
            if required is not None and planned < required:
                errors.append(
                    f"planned_repeats={planned} is below the plan's own "
                    f'required n={required} at SESOI ratio '
                    f"{plan['sesoi_ratio']} and CV={plan['assumed_cv']}")
    return errors


def validate_plans() -> dict[str, list[str]]:
    return {plan_id: validate_plan(plan)
            for plan_id, plan in TIMING_PLANS.items()}


def design_manifest(plan_ids=None, *, repeats_by_plan=None,
                    provenance=None) -> dict:
    ids = list(TIMING_PLAN_IDS if plan_ids is None else plan_ids)
    unknown = [plan_id for plan_id in ids if plan_id not in TIMING_PLANS]
    if unknown:
        raise TimingDesignError(
            f'unknown timing plan(s) {unknown!r}')
    overrides = dict(repeats_by_plan or {})
    unknown = [plan_id for plan_id in overrides if plan_id not in TIMING_PLANS]
    if unknown:
        raise TimingDesignError(
            f'repeats_by_plan names unknown plan(s) {unknown!r}')
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise TimingDesignError(
                'provenance must be the dict returned by capture_provenance()')
        try:
            json.dumps(provenance, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TimingDesignError(
                f'provenance is not strict JSON: {exc}') from None
        provenance = copy.deepcopy(provenance)
    rows = []
    for plan_id in ids:
        plan = get_plan(plan_id)
        repeats = overrides.get(plan_id, plan['planned_repeats'])
        repeats = _need_int(repeats, f'{plan_id} repeats', minimum=1)
        planning = planning_report(plan, repeats=repeats)
        row = copy.deepcopy(plan)
        row['repeats'] = repeats
        row['n_repeats'] = repeats
        row['plan_sha256'] = plan_sha256(plan)
        row['plan_fingerprint_scope'] = PLAN_FINGERPRINT_SCOPE
        row['planning'] = planning
        row['status'] = planning['status']
        rows.append(row)
    metadata = {
        'generated_by': 'timing_design.design_manifest',
        'note': (
            'Registry of the pre-registered latency measurement plans. '
            'Planning is prospective (SESOI ratio + assumed CV + corrected '
            'alpha); a run below the planned repeat budget or without a '
            'balanced operator order is screening, not confirmatory.'),
        'method': ('normal approximation on log medians; inference uses '
                   'timing_stats bootstrap ratio intervals when raw trials '
                   'are available'),
        'plan_fingerprint_scope': PLAN_FINGERPRINT_SCOPE,
    }
    if provenance is not None:
        metadata['provenance'] = provenance
    return {
        'schema_version': '1.0',
        'kind': 'timing_design',
        'created_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'metadata': metadata,
        'results': rows,
    }


def write_design_manifest(path: str, *, plan_ids=None,
                          repeats_by_plan=None, manifest=None,
                          provenance=None) -> str:
    if manifest is None:
        manifest = design_manifest(
            plan_ids, repeats_by_plan=repeats_by_plan,
            provenance=provenance)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.timing_design_', suffix='.json',
                               dir=parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _print_table(manifest: dict) -> None:
    header = (f"{'plan':<16} {'n':>4} {'family':>6} {'SESOI':>6} "
              f"{'CV':>5} {'need':>5} {'power':>6}  status")
    print(header)
    print('-' * len(header))
    for row in manifest['results']:
        planning = row['planning']
        power = ('  n/a' if planning['planned_power'] is None
                 else f"{planning['planned_power']:.2f}")
        required = ('n/a' if planning['required_repeats'] is None
                    else str(planning['required_repeats']))
        print(f"{row['plan_id']:<16} {row['n_repeats']:>4} "
              f"{planning['family_size']:>6} {row['sesoi_ratio']:>6.2f} "
              f"{row['assumed_cv']:>5.2f} {required:>5} {power:>6}  "
              f"{row['status']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Pre-registered timing experiment designs')
    parser.add_argument('--check', action='store_true',
                        help='validate the committed timing registry')
    parser.add_argument('--json', metavar='PATH',
                        help='write the versioned timing manifest')
    parser.add_argument('--plan', metavar='PLAN_ID',
                        help='print a planning report for one plan')
    parser.add_argument('--repeats', type=int, default=None,
                        help='override the repeat budget for --plan')
    parser.add_argument('--cv', type=float, default=None,
                        help='override the assumed CV for --plan')
    parser.add_argument('--family-size', type=int, default=None,
                        help='override the analysis family size')
    parser.add_argument('--pilot', metavar='RESULT_JSON',
                        help='extract CVs from a timing result file')
    parser.add_argument('--raw', metavar='RAW_JSON',
                        help='bootstrap ratio intervals from raw trials')
    args = parser.parse_args(argv)

    if args.check:
        errors = validate_plans()
        bad = {key: value for key, value in errors.items() if value}
        if bad:
            for plan_id, messages in bad.items():
                for message in messages:
                    print(f'[timing-design] {plan_id}: {message}')
            return 1
        print(f'[timing-design] {len(errors)} plan(s) valid')
        return 0

    if args.json:
        try:
            path = write_design_manifest(args.json)
        except (TimingDesignError, OSError, ValueError) as exc:
            print(f'[timing-design] could not write manifest: {exc}')
            return 1
        print(f'[timing-design] wrote {path}')
        return 0

    if args.raw:
        try:
            cells = load_raw_cells(args.raw)
            plan_id = args.plan or result_plan_id(args.raw)
            rows = raw_contrast_rows(
                plan_id, cells, family_size=args.family_size)
        except (TimingDesignError, OSError, ValueError) as exc:
            print(f'[timing-design] raw contrast analysis failed: {exc}')
            return 1
        print(json.dumps({
            'plan_id': plan_id,
            'family_size': (args.family_size
                            or get_plan(plan_id)['planned_family_size']),
            'contrasts': rows,
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.pilot:
        try:
            cells = load_cells(args.pilot)
            plan_id = args.plan
            if plan_id is None:
                plan_id = result_plan_id(args.pilot)
            report = pilot_report(plan_id, cells,
                                  family_size=args.family_size)
        except (TimingDesignError, OSError, ValueError) as exc:
            print(f'[timing-design] pilot analysis failed: {exc}')
            return 1
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.plan:
        try:
            report = planning_report(
                args.plan, repeats=args.repeats, cv=args.cv,
                family_size=args.family_size)
        except TimingDesignError as exc:
            print(f'[timing-design] planning failed: {exc}')
            return 1
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    _print_table(design_manifest())
    return 0


if __name__ == '__main__':
    sys.exit(main())
