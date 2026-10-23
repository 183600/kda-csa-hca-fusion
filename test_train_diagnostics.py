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

import json
import math
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
    accuracy_regime,
    optimizer_update_report,
    pool_error_analyses,
    raise_non_finite,
    snapshot_parameters,
    summarize_gradient_norms,
    triage_gradient,
    triage_nonfinite,
)


def _curve(points, *, steps, val_every=1, ceiling=None):
    it = iter(points)
    curve = ValidationCurve(lambda step: ValPoint(*next(it)),
                            steps=steps, val_every=val_every,
                            label='test', ceiling=ceiling)
    return curve


def test_gradient_norm_summary_names_masked_explosions():
    # Raw norms far above the clip budget mean clipping is masking an
    # explosion; a growing tail adds a second, independent warning.
    summary = summarize_gradient_norms([0.5, 1.0, 2.0, 20.0],
                                       clip_norm=1.0)
    assert summary['n_steps'] == 4
    assert summary['max'] == 20.0
    assert summary['n_above_clip'] == 2
    assert summary['max_over_clip'] == 20.0
    assert summary['n_over_10x_clip'] == 1
    assert any('masking an explosion' in n for n in summary['notes'])
    assert any('grew from' in n for n in summary['notes'])
    json.dumps(summary, allow_nan=False)

    # A healthy, clipped run must not raise a false alarm.
    quiet = summarize_gradient_norms([0.8, 0.9, 0.85, 0.7], clip_norm=1.0)
    assert quiet['n_above_clip'] == 0
    assert quiet['notes'] == []

    # Non-finite norms are counted but never poison the finite statistics.
    mixed = summarize_gradient_norms([1.0, float('inf'), float('nan'), 3.0],
                                     clip_norm=1.0)
    assert mixed['n_nonfinite'] == 2
    assert mixed['max'] == 3.0
    assert mixed['mean'] == 2.0
    json.dumps(mixed, allow_nan=False)

    # Finite values near the float maximum must not overflow the summary to
    # ``inf``: the mean/median and the last-k tail mean stay representable
    # and the report remains JSON-strict.
    huge = summarize_gradient_norms([1e308, 1e308], clip_norm=1e10)
    assert huge['mean'] == 1e308 and huge['median'] == 1e308
    assert huge['tail_mean'] == 1e308
    assert all(math.isfinite(huge[k]) for k in ('mean', 'median',
                                                'tail_mean'))
    json.dumps(huge, allow_nan=False)

    assert summarize_gradient_norms([]) is None
    dead = summarize_gradient_norms([float('nan')], clip_norm=1.0)
    assert dead['max'] is None
    assert 'every recorded gradient norm is non-finite' in dead['notes'][0]


def test_experiment_rows_carry_gradient_health():
    """Exp 5/8 per-seed rows now expose the same clipping/explosion signal
    as Exp 7, so a layout or mechanism row can be rejected for an unstable
    optimizer even when its endpoint loss happens to look fine."""
    import run_ablation
    import run_mechanism_ablation

    layout = run_ablation.eval_layout(
        (1, 1, 1), steps=3, device='cpu', train_batch=8,
        eval_batches=1, eval_batch=8, val_every=0,
        error_batches=1, error_batch=8, max_error_examples=2)
    assert layout['gradient_norms']['n_steps'] == 3
    assert layout['gradient_norms']['clip_norm'] == 1.0
    json.dumps(layout['gradient_norms'], allow_nan=False)

    group = run_ablation.eval_group(
        'g', 'kda', ('memory', 'csa', 'hca'), steps=3, device='cpu',
        train_batch=8, eval_batches=1, eval_batch=8, val_every=0)
    assert group['gradient_norms']['n_steps'] == 3

    variant = run_mechanism_ablation.train_variant(
        run_mechanism_ablation.VARIANTS[0], seed=0, steps=3, device='cpu',
        train_batch=8, eval_batches=1, eval_batch=8)
    assert variant['gradient_norms']['n_steps'] == 3


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


def test_validation_curve_ignores_nonfinite_points_for_best_and_regret():
    """A NaN checkpoint must not win the best slot forever.

    NaN compares False against every later loss, so the old min-rule kept
    a non-finite first point as ``best`` and poisoned the regret/summary
    with NaN even after the curve recovered. The monitor must select over
    finite points and record where the non-finite points were.
    """
    curve = _curve([(float('nan'), 0.10), (1.0, 0.50), (0.8, 0.70)],
                   steps=2, val_every=1)
    for step in range(3):
        curve.record(step)
    block = curve.finalize([2.0, 1.5, 1.0])
    assert block['best_val']['step'] == 2
    assert block['best_val']['loss'] == pytest.approx(0.8)
    assert block['n_nonfinite_val_points'] == 1
    assert block['nonfinite_val_steps'] == [0]
    assert block['val_loss_last_finite'] == pytest.approx(0.8)
    assert block['val_loss_regret_from_best'] == pytest.approx(0.0)
    assert block['generalization_gap'] == pytest.approx(0.8 - 1.5)
    json.dumps(block, allow_nan=False)

    # A non-finite FINAL checkpoint has no regret against the best finite
    # point: report None (and the reason) rather than NaN.
    curve = _curve([(1.0, 0.50), (float('inf'), 0.60)],
                   steps=1, val_every=1)
    curve.record(0)
    curve.record(1)
    block = curve.finalize([2.0, 1.0])
    assert block['best_val']['step'] == 0
    assert block['val_loss_regret_from_best'] is None
    assert block['early_stopping_summary'] is None
    assert block['best_val_is_final'] is False
    assert 'non-finite' in block['early_stopping_unavailable_reason']
    assert block['generalization_gap'] is None
    json.dumps(block, allow_nan=False)


def test_validation_curve_gap_ignores_json_null_training_losses():
    """A sanitized non-finite training loss must not abort the report.

    ``diagnose_training_curves`` accepts ``None`` entries (the JSON-null
    sentinel written for a non-finite loss), but ``finalize`` used to add
    them with ``sum(...)`` and raise TypeError. The gap must average the
    finite tail entries instead.
    """
    curve = _curve([(1.0, 0.5), (0.9, 0.6), (0.8, 0.7)],
                   steps=2, val_every=1)
    for step in range(3):
        curve.record(step)
    block = curve.finalize([2.0, None, 1.0])
    assert block['generalization_gap'] == pytest.approx(0.8 - 1.5)
    json.dumps(block, allow_nan=False)

    # No finite entry in the tail -> an explicit undefined gap, not NaN.
    block = curve.finalize([None, float('nan'), float('inf')])
    assert block['generalization_gap'] is None
    json.dumps(block, allow_nan=False)


def test_validation_curve_gap_is_finite_for_extreme_finite_losses():
    """Finite losses must not make the reported gap +/-inf.

    ``sum([1e308] * 10)`` overflows to inf even though the mean and the
    gap are finite. The gap must use the scaled mean and stay strict-JSON.
    """
    curve = _curve([(1e308, 0.5), (1e308, 0.5), (1e308, 0.5)],
                   steps=2, val_every=1)
    for step in range(3):
        curve.record(step)
    block = curve.finalize([1e308] * 20)
    assert block['generalization_gap'] == 0.0
    json.dumps(block, allow_nan=False)

    # The best-to-final regret is also a difference of finite endpoints
    # that can overflow; it must be None, not +inf.
    curve = _curve([(-1e308, 0.5), (1e308, 0.5)], steps=1, val_every=1)
    curve.record(0)
    curve.record(1)
    block = curve.finalize([1.0])
    assert block['val_loss_regret_from_best'] is None
    assert block['early_stopping_summary']['val_loss_regret'] is None
    json.dumps(block, allow_nan=False)


def test_validation_curve_rejects_swapped_fields_and_disabled_is_empty():
    curve = _curve([(1.0, 3.0)], steps=1, val_every=1)
    with pytest.raises(ValueError, match='accuracy must lie in'):
        curve.record(0)

    disabled = ValidationCurve(lambda step: (_ for _ in ()).throw(
        AssertionError('must not evaluate')), steps=10, val_every=0)
    block = disabled.finalize([1.0, 0.9])
    assert block['val_history'] == []
    assert 'early_stopping_summary' not in block


def test_accuracy_regime_separates_saturation_from_overfitting():
    # Validation loss is flat while held-out accuracy is pinned at 1.0:
    # that is convergence, not a generalization failure.
    regime = accuracy_regime([0.2, 0.6, 1.0, 1.0, 1.0],
                             [0.1, 0.3, 0.7, 0.9, 1.0], ceiling=1.0)
    assert regime['saturated'] is True
    assert regime['val_acc_tail_mean'] == pytest.approx(1.0)
    assert regime['train_acc_still_rising'] is True
    assert regime['ceiling'] == 1.0

    # With no declared ceiling, saturation is unknown, not silently False.
    unknown = accuracy_regime([0.2, 0.3, 0.4])
    assert unknown['saturated'] is None
    assert accuracy_regime([0.5]) is None


def test_train_lm_accuracy_regime_reports_optimizer_step_not_index():
    from train_lm_autodl import _accuracy_regime_for_history

    history = [
        {'step': 0, 'token_accuracy': 0.10},
        {'step': 10, 'token_accuracy': None},
        {'step': 20, 'token_accuracy': 0.50},
        {'step': 30, 'token_accuracy': 0.90},
    ]
    regime = _accuracy_regime_for_history(history, ceiling=1.0)
    assert regime['val_acc_peak'] == pytest.approx(0.90)
    assert regime['val_acc_best_step'] == 30


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

    # Compute / memory / eval latency are part of the same budget screen
    # when the rows carry them.
    memory_hog = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.65, 'peak_memory_bytes': 3 * 1024 ** 3},
        {**base, 'peak_memory_bytes': 1024 ** 3}, significant=True)
    assert memory_hog['verdict'] == 'cost_prohibitive'
    assert memory_hog['worst_cost_ratio']['cost'] == 'peak_memory_bytes'

    slow_eval = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.65, 'mean_eval_ms_per_sample': 9.0},
        {**base, 'mean_eval_ms_per_sample': 1.0}, significant=True)
    assert slow_eval['verdict'] == 'cost_prohibitive'
    assert slow_eval['worst_cost_ratio']['cost'] == 'mean_eval_ms_per_sample'

    more_flops = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.65, 'flops': 10.0},
        {**base, 'flops': 1.0}, significant=True)
    assert more_flops['verdict'] == 'cost_prohibitive'
    assert more_flops['worst_cost_ratio']['cost'] == 'flops'

    # "warranted" must say which cost axes it could NOT compare: an
    # unmeasured memory axis must not be silently treated as free.
    assert 'peak_memory_bytes' in untested['missing_cost_keys']
    assert 'peak_memory_bytes' not in untested['measured_cost_keys']
    assert 'not compared' in untested['recommendation']

    full = {**base, 'flops': 1.0, 'mean_eval_ms_per_sample': 1.0,
            'peak_memory_bytes': 1024 ** 2}
    complete = accuracy_cost_tradeoff(
        {**full, 'mean_acc': 0.65}, full, significant=True)
    assert complete['verdict'] == 'warranted'
    assert complete['missing_cost_keys'] == []
    assert len(complete['measured_cost_keys']) == 6


def test_accuracy_cost_tradeoff_is_json_safe_at_float_limits():
    """An unrepresentable delta/ratio must not leak inf into the report."""
    # Opposite-sign extremes overflow the subtraction.
    rep = accuracy_cost_tradeoff(
        {'mean_acc': 1e308}, {'mean_acc': -1e308})
    assert rep['verdict'] == 'indeterminate'
    assert rep['accuracy_delta'] is None
    json.dumps(rep, allow_nan=False)

    # A ratio whose true value exceeds the float range is not JSON-safe;
    # the axis is reported as missing instead of inf.
    rep = accuracy_cost_tradeoff(
        {'mean_acc': 0.6, 'n_params': 1e308},
        {'mean_acc': 0.5, 'n_params': 1e-300}, paired_pvalue=float('nan'))
    assert 'n_params' in rep['missing_cost_keys']
    assert rep['paired_pvalue'] is None
    json.dumps(rep, allow_nan=False)


def test_oversized_json_integers_are_skipped_by_tradeoff_and_pooling():
    """JSON ints beyond the float range are malformed metrics, not crashes.

    They must not raise from ``float()`` in the gradient summary, the
    accuracy/cost screen, or the bad-case pooling.
    """
    huge = 10 ** 400

    grads = summarize_gradient_norms([huge, 1.0], clip_norm=1.0)
    assert grads['n_steps'] == 2 and grads['n_nonfinite'] == 1
    assert grads['max'] == 1.0
    json.dumps(grads, allow_nan=False)

    base = {'mean_acc': 0.5, 'n_params': 10, 'mean_train_time_s': 1.0}
    no_acc = accuracy_cost_tradeoff({**base, 'mean_acc': huge}, base)
    assert no_acc['verdict'] == 'indeterminate'
    assert 'no comparable accuracy' in no_acc['recommendation']
    huge_cost = accuracy_cost_tradeoff(
        {**base, 'mean_acc': 0.6, 'n_params': huge}, base)
    assert 'n_params' in huge_cost['missing_cost_keys']
    json.dumps(huge_cost, allow_nan=False)

    def seed(seed_id, acc, confidence):
        return {
            'seed': seed_id, 'final_acc': acc,
            'error_analysis': {
                'n_samples': 10, 'n_wrong': 1,
                'categories': {'other': {
                    'count': 1, 'definition': 'other',
                    'confidence': confidence}},
                'examples': {'other': [seed_id]},
            },
        }
    pooled = pool_error_analyses([
        seed(1, huge, {'n_scored': 2, 'n_confident_wrong': 1,
                       'mean_pred_prob': huge, 'mean_target_prob': 0.1,
                       'mean_margin': 0.2, 'mean_entropy_ratio': 0.3}),
        seed(2, 0.4, {'n_scored': 2, 'n_confident_wrong': 1,
                      'mean_pred_prob': 0.7, 'mean_target_prob': 0.1,
                      'mean_margin': 0.2, 'mean_entropy_ratio': 0.3}),
    ])
    assert pooled['representative_seed'] == 2
    assert math.isfinite(pooled['confidence']['mean_pred_prob'])
    json.dumps(pooled, allow_nan=False)

    # Finite extreme confidence values times large sample counts must not
    # overflow the pooled weighted sum to inf.
    extreme = pool_error_analyses([
        seed(1, 0.5, {'n_scored': 10 ** 9, 'n_confident_wrong': 1,
                      'mean_pred_prob': 1e308, 'mean_target_prob': 1e308,
                      'mean_margin': 1e308, 'mean_entropy_ratio': 1e308}),
        seed(2, 0.4, {'n_scored': 10 ** 9, 'n_confident_wrong': 1,
                      'mean_pred_prob': 1e308, 'mean_target_prob': 1e308,
                      'mean_margin': 1e308, 'mean_entropy_ratio': 1e308}),
    ])
    assert extreme['confidence']['mean_pred_prob'] == 1e308
    json.dumps(extreme, allow_nan=False)


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
                    c: {
                        'count': n,
                        'definition': c,
                        'confidence': {
                            'n_scored': n,
                            'n_confident_wrong': 1,
                            'mean_pred_prob': 0.8,
                            'mean_target_prob': 0.1,
                            'mean_margin': 0.7,
                            'mean_entropy_ratio': 0.2,
                        },
                    }
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
    # Confidence is pooled with count weights, not by averaging seed means.
    conf = pooled['pooled_categories']['copyable_missed']['confidence']
    assert conf['n_scored'] == 6
    assert conf['n_confident_wrong'] == 1
    assert conf['mean_pred_prob'] == pytest.approx(0.8)
    assert pooled['confidence']['n_scored_wrong'] == 9
    assert pooled['confidence']['n_confident_wrong'] == 2
    assert pool_error_analyses([{'seed': 0}]) is None


def test_optimizer_update_report_detects_coverage_and_frozen_params():
    # A healthy step: every trainable tensor is covered and moved.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(),
                                torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    loss = model(torch.arange(12, dtype=torch.float32).reshape(3, 4)).sum()
    opt.zero_grad()
    loss.backward()
    named = list(model.named_parameters())
    before = snapshot_parameters(named)
    opt.step()
    report = optimizer_update_report(before, named, opt)
    assert report['ok'] is True
    assert report['n_not_in_optimizer'] == 0
    assert report['n_not_updated'] == 0
    assert report['n_grad_nonzero'] == report['n_trainable_tensors']
    json.dumps(report, allow_nan=False)

    # A parameter group that omits the head: loss can still fall through the
    # trunk, so the update audit must fail even though gradients reached it.
    opt_missing = torch.optim.AdamW(model[0].parameters(), lr=1e-2)
    opt_missing.zero_grad()
    loss = model(torch.arange(12, dtype=torch.float32).reshape(3, 4)).sum()
    loss.backward()
    before = snapshot_parameters(named)
    opt_missing.step()
    missing = optimizer_update_report(before, named, opt_missing)
    assert missing['ok'] is False
    assert '2.weight' in missing['not_in_optimizer_names']
    assert '2.bias' in missing['not_in_optimizer_names']

    # lr=0 freezes every covered parameter although gradients are non-zero.
    opt_frozen = torch.optim.AdamW(model.parameters(), lr=0.0)
    opt_frozen.zero_grad()
    loss = model(torch.arange(12, dtype=torch.float32).reshape(3, 4)).sum()
    loss.backward()
    before = snapshot_parameters(named)
    opt_frozen.step()
    frozen = optimizer_update_report(before, named, opt_frozen)
    assert frozen['ok'] is False
    assert frozen['n_not_updated'] == frozen['n_grad_nonzero']
