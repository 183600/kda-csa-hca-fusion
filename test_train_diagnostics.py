"""Tests for the held-out monitoring / claim-judging helpers.

These pin the contract that makes the results-analysis layer trustworthy:

* a validation curve reports the early-stopping regret (best vs final), not
  just the endpoint;
* the accuracy-vs-cost screen refuses sub-noise gains, untested gains and
  disproportionate cost;
* the staged NaN triage names the first failing stage;
* per-seed bad-case reports pool into one row-level category distribution.

Run with ``pytest -q test_train_diagnostics.py``.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_diagnostics import (
    NonFiniteTrainingError,
    ValPoint,
    ValidationCurve,
    accuracy_cost_tradeoff,
    pool_error_analyses,
    raise_non_finite,
    triage_gradient,
    triage_nonfinite,
)


def _curve(points, *, steps, val_every=1, ceiling=None):
    it = iter(points)
    curve = ValidationCurve(lambda step: ValPoint(*next(it)),
                            steps=steps, val_every=val_every,
                            label='test', ceiling=ceiling)
    return curve


def test_validation_curve_quantifies_early_stopping_regret():
    # 0..3 improve, then 4..5 overfit: the final point is past the optimum.
    points = [(2.0, 0.10), (1.6, 0.30), (1.3, 0.50), (1.1, 0.70),
              (1.4, 0.62), (1.7, 0.55)]
    curve = _curve(points, steps=5, val_every=1)
    for step in range(6):
        curve.record(step)
    block = curve.finalize([3.0, 2.2, 1.6, 1.2, 0.9, 0.7])

    assert block['val_loss_regret_from_best'] == pytest.approx(0.6)
    assert block['val_acc_regret_from_best'] == pytest.approx(0.15)
    assert block['steps_since_best'] == 2
    assert block['best_val_is_final'] is False
    summary = block['early_stopping_summary']
    assert summary['selected_step'] == 3
    assert summary['steps_saved'] == 2
    assert summary['val_loss_regret'] == pytest.approx(0.6)
    assert summary['val_acc_at_selection'] == pytest.approx(0.70)
    assert block['curve_diagnosis']['verdict'] == 'overfitting'


def test_validation_curve_no_regret_when_final_is_best():
    curve = _curve([(2.0, 0.1), (1.5, 0.3), (1.0, 0.6)],
                   steps=2, val_every=1)
    for step in range(3):
        curve.record(step)
    block = curve.finalize([2.0, 1.4, 0.9])
    assert block['best_val_is_final'] is True
    assert block['val_loss_regret_from_best'] == pytest.approx(0.0)
    assert block['early_stopping_summary']['steps_saved'] == 0


def test_validation_curve_rejects_swapped_fields_and_disabled_is_empty():
    curve = _curve([(1.0, 3.0)], steps=1, val_every=1)
    with pytest.raises(ValueError, match='accuracy must lie in'):
        curve.record(0)

    disabled = ValidationCurve(lambda step: (_ for _ in ()).throw(
        AssertionError('must not evaluate')), steps=10, val_every=0)
    block = disabled.finalize([1.0, 0.9])
    assert block['val_history'] == []
    assert 'early_stopping_summary' not in block


def test_accuracy_cost_tradeoff_screens_effect_significance_and_cost():
    base = {'mean_acc': 0.60, 'n_params': 1000, 'mean_train_time_s': 10.0,
            'mean_fwd_ms': 1.0}

    below = accuracy_cost_tradeoff({**base, 'mean_acc': 0.602}, base)
    assert below['verdict'] == 'below_min_effect'

    untested = accuracy_cost_tradeoff({**base, 'mean_acc': 0.65}, base)
    assert untested['verdict'] == 'warranted'
    assert 'NOT tested' in untested['recommendation']

    noisy = accuracy_cost_tradeoff({**base, 'mean_acc': 0.65}, base,
                                   paired_pvalue=0.20)
    assert noisy['verdict'] == 'not_significant'

    expensive = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.65, 'mean_train_time_s': 100.0}, base,
        significant=True)
    assert expensive['verdict'] == 'cost_prohibitive'
    assert expensive['worst_cost_ratio']['cost'] == 'mean_train_time_s'

    cheaper = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.65, 'mean_fwd_ms': 0.5}, base,
        significant=True)
    assert cheaper['verdict'] == 'warranted'


def test_nonfinite_triage_names_first_failing_stage_and_gradient_triage():
    bad = torch.tensor([float('nan'), 1.0])
    diag = triage_nonfinite(inputs={'x': bad}, loss=float('nan'))
    assert diag['verdict'] == 'data_nonfinite'
    assert diag['first_failing_stage'] == 'data_nonfinite'

    amp = triage_nonfinite(loss=float('nan'), amp_enabled=True,
                           fp32_forward_loss=1.2)
    assert amp['verdict'] == 'amp_overflow'

    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    p1.grad = torch.tensor([float('inf'), 1.0])
    p2.grad = torch.ones(2)
    assert triage_gradient([p1, p2]) == [p1]

    with pytest.raises(NonFiniteTrainingError) as exc:
        raise_non_finite('loss', loss=float('nan'),
                         token_ids={'ids': torch.tensor([99])},
                         id_range=(0, 10))
    assert exc.value.diagnosis['verdict'] == 'token_ids_out_of_range'
    assert 'token_ids_out_of_range' in str(exc.value)


def test_pooled_error_analyses_use_worst_seed_examples():
    def seed_record(seed, acc, counts, examples):
        return {
            'seed': seed,
            'final_acc': acc,
            'error_analysis': {
                'n_samples': 10,
                'n_wrong': sum(counts.values()),
                'categories': {
                    c: {'count': n, 'definition': c}
                    for c, n in counts.items()
                },
                'examples': examples,
                'accuracy_by_distance_bin': [{'bin': 'far', 'accuracy': acc}],
            },
        }

    per_seed = [
        seed_record(0, 0.7, {'near_miss': 3}, {'near_miss': ['a']}),
        seed_record(1, 0.4, {'copyable_missed': 6},
                    {'copyable_missed': ['worst']}),
    ]
    pooled = pool_error_analyses(per_seed)
    assert pooled['n_seeds'] == 2
    assert pooled['n_wrong'] == 9
    assert pooled['pooled_categories']['near_miss']['count'] == 3
    assert pooled['pooled_categories']['copyable_missed']['count'] == 6
    assert pooled['representative_seed'] == 1
    assert pooled['representative_examples'] == {'copyable_missed': ['worst']}
    assert pool_error_analyses([{'seed': 0}]) is None
