"""Tests for the results-analysis tooling.

Covers ``results_analysis`` (curve-shape diagnosis, systematic NaN/Inf
triage, efficiency/accuracy Pareto annotation, token-level LM bad-case
diagnostics) and the wiring: held-out monitoring in ``run_quality`` is
RNG-inert, bad-case classification is task-correct and recordable, and the
LM training path's ``analyze_lm_errors`` is read-only, deterministic and
JSON-serializable.

How to run::

    pytest -q test_results_analysis.py
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_analysis import (  # noqa: E402
    lm_token_diagnostics,
    annotate_rows_with_pareto,
    binary_ranking_metrics,
    diagnose_nonfinite,
    expected_average_precision,
    diagnose_training_curves,
    multiclass_classification_metrics,
    confusion_matrix,
    pareto_frontier,
    summarize_lm_diagnostics,
)
from run_quality import (  # noqa: E402
    _classify_task_errors,
    _error_confidence,
    _hypothesis_verdicts,
    _paired_one_sided_t,
    _parse_lr_list,
    train_one,
)


# ---------------------------------------------------------------------------
# Curve-shape diagnosis
# ---------------------------------------------------------------------------

def test_curve_diagnosis_overfitting():
    train = [3.0 - 0.05 * i for i in range(60)]        # keeps falling
    val = [3.0 - 0.04 * i for i in range(40)] + [1.6 + 0.02 * i
                                                 for i in range(20)]
    rep = diagnose_training_curves(train, val)
    assert rep['verdict'] == 'overfitting'
    assert rep['signals']['train_improved'] is True
    assert rep['signals']['val_rose_from_best'] is True
    assert any('regularization' in r or 'early stopping' in r
               for r in rep['recommendations'])
    json.dumps(rep, allow_nan=False)


def test_curve_diagnosis_underfitting():
    train = [3.0 + 0.001 * (-1) ** i for i in range(60)]
    val = [3.1 + 0.001 * (-1) ** i for i in range(20)]
    rep = diagnose_training_curves(train, val)
    assert rep['verdict'] == 'underfitting'
    assert rep['signals']['train_improved'] is False
    assert any('learning rate' in r for r in rep['recommendations'])


def test_curve_diagnosis_healthy_and_diverged():
    rep = diagnose_training_curves([3.0 - 0.03 * i for i in range(50)],
                                   [3.0 - 0.02 * i for i in range(20)])
    assert rep['verdict'] == 'healthy'

    rep = diagnose_training_curves([3.0, 2.9, float('nan'), 2.8])
    assert rep['verdict'] == 'diverged'
    assert rep['train']['non_finite_at'] == 2

    rep = diagnose_training_curves([3.0, 2.5],
                                   [2.9, float('inf')])
    assert rep['verdict'] == 'diverged'
    assert rep['val']['non_finite_at'] == 1


def test_curve_diagnosis_keeps_extreme_finite_curves_json_safe():
    """A finite loss curve must never emit an inf/NaN summary.

    ``sum([1e308, 1e308])`` overflows even though both losses and their
    mean are finite, and an opposite-sign swing can produce a slope larger
    than the float range.  The report is embedded in strict JSON, so the
    ordinary path is kept and only these overflow cases degrade to a
    finite scaled mean or a documented ``None`` slope.
    """
    for train in ([1e308, 1e308, 1e308],
                  [1e308, -1e308, 1e308],
                  [1e308, -1e308],
                  [1e308, 1e308]):
        rep = diagnose_training_curves(train)
        json.dumps(rep, allow_nan=False)
        assert rep['train']['head_mean'] is not None
        assert rep['train']['tail_mean'] is not None
    # Ordinary curves still have the historical values.
    rep = diagnose_training_curves([3.0, 2.0, 1.0])
    assert rep['train']['head_mean'] == 3.0
    assert rep['train']['tail_mean'] == 1.0
    assert rep['train']['tail_slope_per_point'] == pytest.approx(-1.0)


def test_curve_diagnosis_flags_a_budget_limited_run():
    """A healthy run that is still descending at the last step is NOT converged.

    Head/tail means cannot distinguish "plateaued" from "still falling";
    the tail slope can, and the report must say the fixed-budget endpoint
    should not be ranked as a converged result.
    """
    train = [3.0 - 0.04 * i for i in range(60)]
    val = [3.0 - 0.03 * i for i in range(20)]
    rep = diagnose_training_curves(train, val)
    assert rep['verdict'] == 'healthy'
    assert rep['budget_limited'] is True
    assert rep['signals']['train_still_improving_at_end'] is True
    assert rep['signals']['val_still_improving_at_end'] is True
    assert any('not converged' in r or 'Extend the budget' in r
               for r in rep['recommendations'])
    json.dumps(rep, allow_nan=False)

    # Same early drop, but the final segment is flat: converged, not
    # budget-limited (the tail mean is similar; the slope is not).
    plateau_train = [3.0 - 0.04 * i for i in range(20)] + [2.2] * 20
    plateau_val = [3.0 - 0.03 * i for i in range(10)] + [2.7] * 10
    rep = diagnose_training_curves(plateau_train, plateau_val)
    assert rep['budget_limited'] is False
    assert rep['signals']['val_still_improving_at_end'] is False

    # A late rebound must not be mistaken for "still improving".
    rebound = [3.0 - 0.03 * i for i in range(15)] + [2.8, 3.0, 3.2]
    rep = diagnose_training_curves(rebound, None)
    assert rep['signals']['train_still_improving_at_end'] is False


def test_curve_diagnosis_train_improving_only():
    rep = diagnose_training_curves([3.0 - 0.05 * i for i in range(40)])
    assert rep['verdict'] == 'train_improving_only'
    assert any('validation' in r for r in rep['recommendations'])


def test_curve_diagnosis_single_validation_point_is_insufficient():
    """One val point has no SHAPE: no over/under verdict is supported."""
    rep = diagnose_training_curves([3.0 - 0.05 * i for i in range(30)],
                                   [2.5])
    assert rep['verdict'] == 'insufficient_validation_curve'
    assert rep['signals']['val_n_points'] == 1
    assert rep['signals']['train_improved'] is True
    json.dumps(rep, allow_nan=False)


def test_sweep_summary_prefers_saturation_over_loss_shape():
    """No accuracy headroom is a stronger fact than a rising val loss.

    A fully saturated sweep has no room to improve, so recommending
    regularization would send the next experiment in the wrong direction;
    the probe must get harder instead.
    """
    from run_ablation import _curve_diagnosis_summary

    rows = [
        {'ratio': '3:1:1', 'curve_verdict': 'overfitting',
         'n_seeds_ok': 5, 'n_seeds_at_ceiling': 4,
         'curve_verdict_stable': True,
         'mean_generalization_gap': 0.1, 'n_seeds_budget_limited': 0},
        {'ratio': '2:2:1', 'curve_verdict': 'overfitting',
         'n_seeds_ok': 5, 'n_seeds_at_ceiling': 4,
         'curve_verdict_stable': True,
         'mean_generalization_gap': 0.2, 'n_seeds_budget_limited': 0},
    ]
    summary = _curve_diagnosis_summary(rows, 'ratio')
    assert summary['regime'] == 'saturated'
    assert 'too easy' in summary['interpretation']
    assert 'regularization' not in summary['recommended_action']

    # One lucky seed at 100% must not label the whole PROBE too easy.
    rows[1]['n_seeds_at_ceiling'] = 1
    summary = _curve_diagnosis_summary(rows, 'ratio')
    assert summary['regime'] == 'overfitting'
    assert summary['rows_partially_saturated'] == ['2:2:1']

    # Without universal saturation, the loss-shape verdict still drives.
    rows[1]['n_seeds_at_ceiling'] = 0
    summary = _curve_diagnosis_summary(rows, 'ratio')
    assert summary['regime'] == 'overfitting'
    assert summary['rows_partially_saturated'] == []


def test_curve_rollup_surfaces_masked_gradient_explosions():
    """A healthy loss curve must not hide clipping-masked explosions.

    The per-seed gradient summary is now rolled up to the row and the
    sweep, so the action line can say "lower the LR" even when the loss
    verdict itself is healthy.
    """
    from run_ablation import (_aggregate_curve_diagnosis,
                              _curve_diagnosis_summary)

    seeds = [
        {'gradient_norms': {
            'notes': ['peak gradient norm is 20.0x the clip budget'],
            'n_above_clip': 3, 'max_over_clip': 20.0}},
        {'gradient_norms': {
            'notes': [], 'n_above_clip': 0, 'max_over_clip': 1.0}},
    ]
    agg = _aggregate_curve_diagnosis(seeds)
    assert agg['n_seeds_with_gradient_warnings'] == 1
    assert agg['n_seeds_clipped'] == 1
    assert agg['max_gradient_over_clip'] == 20.0
    assert agg['gradient_warning_notes'] == [
        'peak gradient norm is 20.0x the clip budget']
    json.dumps(agg, allow_nan=False)

    rows = [{'ratio': '3:1:1', 'curve_verdict': 'healthy',
             'n_seeds_ok': 2, 'n_seeds_with_gradient_warnings': 1,
             'max_gradient_over_clip': 20.0}]
    summary = _curve_diagnosis_summary(rows, 'ratio')
    assert summary['rows_with_gradient_warnings'] == ['3:1:1']
    assert summary['max_gradient_over_clip'] == 20.0
    assert 'Separately, raw gradients' in summary['recommended_action']


# ---------------------------------------------------------------------------
# Systematic non-finite triage
# ---------------------------------------------------------------------------

def test_nonfinite_diagnosis_data_first():
    rep = diagnose_nonfinite(
        loss=torch.tensor(float('nan')),
        inputs={'x': torch.tensor([1.0, float('nan')])},
        params={'w': torch.ones(2)},
    )
    assert rep['verdict'] == 'data_nonfinite'
    assert rep['checks'][0]['check'] == 'data'


def test_nonfinite_diagnosis_token_ids_and_params():
    rep = diagnose_nonfinite(
        loss=torch.tensor(1.0),
        token_ids={'input_ids': torch.tensor([0, 5, 99])},
        id_range=(0, 16),
    )
    assert rep['verdict'] == 'token_ids_out_of_range'

    rep = diagnose_nonfinite(
        loss=torch.tensor(1.0),
        params={'w': torch.tensor([float('inf'), 1.0])},
    )
    assert rep['verdict'] == 'parameters_nonfinite'


def test_nonfinite_diagnosis_amp_overflow_and_logits():
    rep = diagnose_nonfinite(
        loss=torch.tensor(float('nan')),
        params={'w': torch.ones(1)},
        logits=torch.tensor([[1.0, 2.0]]),
        amp_enabled=True,
        amp_scaler=None,
        fp32_forward_loss=torch.tensor(2.0),
    )
    assert rep['verdict'] == 'amp_overflow'

    rep = diagnose_nonfinite(
        loss=torch.tensor(1.0),
        logits=torch.tensor([[1e5, 0.0]]),
    )
    logits_check = [c for c in rep['checks'] if c['check'] == 'logits'][0]
    assert logits_check['status'] == 'warn'


def test_nonfinite_diagnosis_gradients_and_clean():
    rep = diagnose_nonfinite(
        loss=torch.tensor(1.0),
        grads={'w': torch.tensor([float('nan')])},
    )
    assert rep['verdict'] == 'gradients_nonfinite'

    rep = diagnose_nonfinite(
        loss=torch.tensor(1.0),
        params={'w': torch.ones(2)},
        grads={'w': torch.ones(2)},
        grad_norm=0.5,
        logits=torch.zeros(1, 3),
    )
    assert rep['verdict'] == 'no_nonfinite_signal'
    json.dumps(rep, allow_nan=False)


# ---------------------------------------------------------------------------
# Efficiency / accuracy Pareto annotation
# ---------------------------------------------------------------------------

def test_pareto_frontier_dominance():
    rows = [
        {'op': 'a', 'mean_acc': 0.90, 'n_params': 100, 'mean_train_time_s': 1.0,
         'mean_eval_ms_per_sample': 1.0},
        {'op': 'b', 'mean_acc': 0.90, 'n_params': 500, 'mean_train_time_s': 5.0,
         'mean_eval_ms_per_sample': 2.0},
        {'op': 'c', 'mean_acc': 0.95, 'n_params': 900, 'mean_train_time_s': 9.0,
         'mean_eval_ms_per_sample': 3.0},
    ]
    rep = pareto_frontier(rows)
    assert set(rep['frontier']) == {'a', 'c'}
    assert rep['dominated_by'] == {'b': 'a'}
    # accuracy per 1000 parameters: 0.90 * 1000 / 100 = 9.0
    assert rep['points']['a']['acc_per_1k_params'] == pytest.approx(9.0)

    annotate_rows_with_pareto(rows)
    by_op = {r['op']: r['pareto'] for r in rows}
    assert by_op['a']['optimal'] is True
    assert by_op['b']['dominated_by'] == 'a'
    assert by_op['c']['acc_per_eval_ms'] == pytest.approx(0.95 / 3.0)


def test_pareto_frontier_keeps_duplicate_names_apart():
    """Two rows with the same name must not collapse into one point.

    The points are keyed by name, so a repeated name used to overwrite
    the earlier row: that row vanished from the dominance analysis AND
    both rows were annotated with the surviving row's ratios (the
    first row reported acc_per_train_second 0.9 instead of 0.05).
    """
    rows = [
        {'op': 'kda', 'mean_acc': 0.5, 'n_params': 100,
         'mean_train_time_s': 10.0},
        {'op': 'kda', 'mean_acc': 0.9, 'n_params': 50,
         'mean_train_time_s': 1.0},
        {'op': 'csa', 'mean_acc': 0.8, 'n_params': 80,
         'mean_train_time_s': 5.0},
    ]
    report = pareto_frontier(rows)
    # Both 'kda' rows are present under distinct keys.
    assert set(report['points']) == {'kda', 'kda#2', 'csa'}
    assert report['points']['kda']['acc_per_train_second'] == pytest.approx(0.05)
    assert report['points']['kda#2']['acc_per_train_second'] == pytest.approx(0.9)
    # The weaker 'kda' row is dominated by the stronger one.
    assert report['dominated_by']['kda'] == 'kda#2'

    annotate_rows_with_pareto(rows)
    assert rows[0]['pareto']['acc_per_train_second'] == pytest.approx(0.05)
    assert rows[0]['pareto']['optimal'] is False
    assert rows[0]['pareto']['dominated_by'] == 'kda#2'
    assert rows[1]['pareto']['acc_per_train_second'] == pytest.approx(0.9)
    assert rows[1]['pareto']['optimal'] is True
    assert rows[2]['pareto']['acc_per_train_second'] == pytest.approx(0.16)


def test_pareto_frontier_suffix_collision_stays_unique():
    """A literal ``a#2`` name must not collide with the duplicate suffix."""
    rows = [
        {'op': 'a', 'mean_acc': 0.5, 'n_params': 100,
         'mean_train_time_s': 10.0},
        {'op': 'a#2', 'mean_acc': 0.9, 'n_params': 50,
         'mean_train_time_s': 1.0},
        {'op': 'a', 'mean_acc': 0.8, 'n_params': 200,
         'mean_train_time_s': 4.0},
    ]
    report = pareto_frontier(rows)
    assert set(report['points']) == {'a', 'a#2', 'a#3'}
    assert report['points']['a']['acc_per_train_second'] == pytest.approx(0.05)
    assert report['points']['a#2']['acc_per_train_second'] == pytest.approx(0.9)
    assert report['points']['a#3']['acc_per_train_second'] == pytest.approx(0.2)
    assert report['dominated_by'] == {'a': 'a#2', 'a#3': 'a#2'}

    annotate_rows_with_pareto(rows)
    assert rows[0]['pareto']['dominated_by'] == 'a#2'
    assert rows[1]['pareto']['optimal'] is True
    assert rows[2]['pareto']['dominated_by'] == 'a#2'
    assert rows[2]['pareto']['acc_per_train_second'] == pytest.approx(0.2)


def test_pareto_frontier_skips_missing_costs():
    rows = [
        {'op': 'a', 'mean_acc': 0.5, 'n_params': 10, 'mean_train_time_s': 1.0},
        {'op': 'b', 'mean_acc': 0.6, 'n_params': None,
         'mean_train_time_s': 2.0},
    ]
    rep = pareto_frontier(rows)
    assert rep['skipped'] == ['b']
    assert rep['frontier'] == ['a']


def test_pareto_frontier_skips_malformed_values_instead_of_crashing():
    """Non-numeric fields and unhashable names are reported, not fatal.

    The dominance pass used to call ``float()`` directly on every metric and
    hash the row name, so one malformed row aborted the whole annotation
    instead of being listed under ``skipped``.
    """
    rows = [
        {'op': 'bad_acc', 'mean_acc': 'n/a',
         'n_params': 10, 'mean_train_time_s': 1.0},
        {'op': 'bad_cost', 'mean_acc': 0.5,
         'n_params': [10], 'mean_train_time_s': 1.0},
        {'op': 'string_metric', 'mean_acc': '0.9', 'n_params': '10',
         'mean_train_time_s': '1.0'},
        {'op': ['unhashable'], 'mean_acc': 0.7,
         'n_params': 10, 'mean_train_time_s': 1.0},
        {'op': 'good', 'mean_acc': 0.5,
         'n_params': 10, 'mean_train_time_s': 1.0},
        'not-a-row',
    ]
    rep = pareto_frontier(rows)
    assert rep['skipped'] == ['bad_acc', 'bad_cost', 'string_metric',
                              'row#5']
    # The unhashable name is keyed by repr, so it can still participate.
    assert rep['frontier'] == [str(['unhashable'])]
    annotate_rows_with_pareto(rows)
    assert rows[0]['pareto'] is None and rows[1]['pareto'] is None
    assert rows[2]['pareto'] is None
    assert rows[3]['pareto']['optimal'] is True
    assert rows[4]['pareto']['dominated_by'] == str(['unhashable'])


def test_pareto_ratios_stay_finite_for_tiny_costs():
    """A finite subnormal cost can overflow the derived efficiency ratio."""
    tiny = float.fromhex('0x0.0000000000001p-1022')  # smallest subnormal
    rows = [{'op': 'subnormal', 'mean_acc': 1.0,
             'mean_train_time_s': tiny, 'n_params': tiny,
             'mean_eval_ms_per_sample': tiny}]
    rep = pareto_frontier(rows, cost_keys=('mean_train_time_s',))
    point = rep['points']['subnormal']
    assert point['acc_per_train_second'] is None
    assert point['acc_per_1k_params'] is None
    assert point['acc_per_eval_ms'] is None
    json.dumps(rep, allow_nan=False)
    annotate_rows_with_pareto(rows, cost_keys=('mean_train_time_s',))
    json.dumps(rows, allow_nan=False)


def test_confusion_matrix_rejects_out_of_range_explicit_classes():
    """An explicit num_classes must bound every observed id, not alias them."""
    with pytest.raises(ValueError):
        confusion_matrix([2], [0], num_classes=2)
    with pytest.raises(ValueError):
        confusion_matrix([0], [4], num_classes=2)
    # In-range explicit classes still keep the full zero row/column.
    cm = confusion_matrix([1], [1], num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[1, 1].item() == 1 and cm.sum().item() == 1
    # No explicit count: the derived shape covers the observed maximum.
    assert confusion_matrix([2], [2]).shape == (3, 3)
    # The class count is exact, not silently truncated from a bool/float.
    for bad_classes in (True, 2.5, '3'):
        with pytest.raises(ValueError):
            confusion_matrix([0], [0], num_classes=bad_classes)

    # Class ids are exact too: 1.9 must not silently become class 1.
    for bad_pred, bad_target in (
            ([0.0, 1.9], [0, 1]),
            ([0, 1], [0.0, 1.0]),
            ([True], [1]),
            ([True, 2], [1, 0]),
            ([0, 1], [False, 1]),
            ([[False, 2]], [[0, 1]]),
    ):
        with pytest.raises(ValueError):
            confusion_matrix(bad_pred, bad_target)


# ---------------------------------------------------------------------------
# Imbalance-aware classification / ranking metrics
# ---------------------------------------------------------------------------

def test_multiclass_metrics_expose_accuracy_hiding_collapse():
    """90% accuracy from a collapsed predictor must not look good.

    The metric block exists so an imbalanced / collapsed result cannot
    hide behind accuracy: macro F1 and balanced accuracy must expose it,
    and every F1 variant must be labelled with its averaging convention.
    """
    # 90 samples of class 0 + one sample of each class 1..10; the model
    # always predicts class 0. 90% accuracy, but only 1 of 11 classes is
    # ever recalled.
    target = torch.cat([torch.zeros(90, dtype=torch.long),
                        torch.arange(1, 11)])
    pred = torch.zeros(100, dtype=torch.long)
    m = multiclass_classification_metrics(pred, target, num_classes=11)
    assert m['accuracy'] == pytest.approx(0.9)
    assert m['f1_micro'] == pytest.approx(m['accuracy'])
    # Macro averaging (unweighted over classes) exposes the collapse:
    # class 0 recall is 1.0 (balanced accuracy = 1/11), its precision is
    # 90/100 (the predictor also claims the 10 minority samples), so
    # F1_class0 = 2*0.9/1.9 = 0.947...; the other 10 classes have F1 0.0.
    assert m['balanced_accuracy'] == pytest.approx(1 / 11)
    assert m['f1_macro'] == pytest.approx((2 * 0.9 / 1.9) / 11)
    # Support-weighted averaging mostly shows the dominant class.
    assert m['f1_weighted'] == pytest.approx(0.9 * (2 * 0.9 / 1.9))
    # Explicit averaging labels and per-class support are recorded.
    assert 'macro' in m['averaging_note'] and 'micro' in m['averaging_note']
    assert sum(c['support'] for c in m['per_class']) == m['n_samples']
    # class_balance = max support / min support (90 / 1 here).
    assert m['class_balance'] == pytest.approx(90.0)


def test_multiclass_metrics_perfect_and_single_class():
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    pred = target.clone()
    m = multiclass_classification_metrics(pred, target, num_classes=3,
                                          scores=torch.nn.functional.one_hot(
                                              target, 3).double())
    assert m['accuracy'] == 1.0
    assert m['f1_macro'] == pytest.approx(1.0)
    assert m['f1_weighted'] == pytest.approx(1.0)
    assert m['pr_auc_macro_ovr'] == pytest.approx(1.0)
    assert m['roc_auc_macro_ovr'] == pytest.approx(1.0)
    # A single class present: accuracy is defined, one-vs-rest AUCs are
    # undefined for every class (no negatives / no positives) -> reported
    # as None, never 0.0 or NaN.
    one = multiclass_classification_metrics(
        torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
        num_classes=3, scores=torch.zeros(4, 3))
    assert one['accuracy'] == 1.0
    assert one['pr_auc_macro_ovr'] is None
    assert one['roc_auc_macro_ovr'] is None
    assert one['balanced_accuracy'] == pytest.approx(1.0)


def test_expected_average_precision_matches_enumeration():
    """The closed-form random-ranking AP equals exhaustive enumeration."""
    import itertools

    def ap_at_positions(positions, n):
        # 1-based positions of the relevant items in the ranking.
        return sum((i + 1) / r for i, r in enumerate(positions)) / len(positions)

    for n in range(2, 8):
        for n_pos in range(1, n):
            values = [ap_at_positions(c, n)
                      for c in itertools.combinations(range(1, n + 1), n_pos)]
            assert expected_average_precision(n_pos, n - n_pos) == pytest.approx(
                sum(values) / len(values), rel=0, abs=1e-12)
    assert expected_average_precision(0, 5) is None
    assert expected_average_precision(3, 0) == 1.0
    for bad_counts in ((True, 1), (1, True), (1.5, 2), (1, '2')):
        with pytest.raises(ValueError):
            expected_average_precision(*bad_counts)
    # One positive among n is H_n / n, identical to chance MRR.
    assert expected_average_precision(1, 99) == pytest.approx(
        sum(1.0 / i for i in range(1, 101)) / 100)


def test_binary_ranking_metrics_rare_positive_and_ties():
    # 1 positive among 100 candidates: PR-AUC is the informative metric.
    labels = torch.zeros(100, dtype=torch.long)
    labels[7] = 1
    perfect = torch.zeros(100)
    perfect[7] = 1.0
    m = binary_ranking_metrics(perfect, labels)
    assert m['pr_auc'] == pytest.approx(1.0)
    assert m['roc_auc'] == pytest.approx(1.0)
    assert m['mrr'] == pytest.approx(1.0)
    assert m['recall_at_1'] == pytest.approx(1.0)
    # Chance AP is the EXACT expectation under a uniformly random ranking,
    # H_100 / 100 for a single positive -- not the prevalence 0.01.
    assert m['chance_pr_auc'] == pytest.approx(
        sum(1.0 / i for i in range(1, 101)) / 100)
    assert m['chance_pr_auc'] > m['prevalence']
    # All-equal (for example quantized) scores: ROC-AUC is the exact chance
    # value 0.5 thanks to average-rank tie handling. PR-AUC breaks ties by
    # candidate order (stable sort): the positive sits at index 7, so the
    # first 8 ranked slots contain exactly one hit -> AP = 1/8, and MRR
    # follows the same deterministic tie order.
    tied = binary_ranking_metrics(torch.zeros(100), labels)
    assert tied['roc_auc'] == pytest.approx(0.5)
    assert tied['pr_auc'] == pytest.approx(1 / 8)
    assert tied['mrr'] == pytest.approx(1 / 8)
    assert tied['recall_at_1'] is False
    # Several relevant items: recall@1 is the fraction found in the top
    # slot (1/n_pos when the top candidate is relevant), not a boolean.
    multi = binary_ranking_metrics(
        torch.tensor([0.9, 0.8, 0.7, 0.1]),
        torch.tensor([1, 1, 0, 0]))
    assert multi['precision_at_1'] == pytest.approx(1.0)
    assert multi['recall_at_1'] == pytest.approx(0.5)
    missed = binary_ranking_metrics(
        torch.tensor([0.9, 0.8, 0.7, 0.1]),
        torch.tensor([0, 1, 1, 0]))
    assert missed['precision_at_1'] == pytest.approx(0.0)
    assert missed['recall_at_1'] == pytest.approx(0.0)
    # No positives at all: ranking metrics are undefined (None).
    none_pos = binary_ranking_metrics(torch.rand(10), torch.zeros(10,
                                                                  dtype=torch.long))
    assert none_pos['pr_auc'] is None and none_pos['roc_auc'] is None
    # valid mask removes causally unavailable candidates.
    scores = torch.tensor([0.9, 0.1, 0.1])
    labels = torch.tensor([0, 1, 0])
    masked = binary_ranking_metrics(scores, labels,
                                    valid=torch.tensor([False, True, True]))
    assert masked['n'] == 2 and masked['n_pos'] == 1

    # Negative / non-binary labels must not be treated as positives.
    for bad in ([-1, 0, 1], [0, 2, 0], [0.0, 0.5, 1.0]):
        with pytest.raises(ValueError):
            binary_ranking_metrics(torch.arange(3.0), torch.tensor(bad))
    # A non-finite score is not a rank, and a 0.5 mask value is not
    # "False with extra confidence": both silently corrupt the metric.
    for bad_scores in ([0.0, float('nan'), 1.0],
                       [0.0, float('inf'), 1.0]):
        with pytest.raises(ValueError):
            binary_ranking_metrics(torch.tensor(bad_scores),
                                   torch.tensor([0, 1, 0]))
    with pytest.raises(ValueError):
        binary_ranking_metrics(torch.arange(3.0), torch.tensor([0, 1, 0]),
                               valid=torch.tensor([1.0, 0.5, 1.0]))
    # Non-finite scores on explicitly EXCLUDED candidates are harmless.
    allowed = binary_ranking_metrics(
        torch.tensor([float('nan'), 0.1, 0.9]),
        torch.tensor([0, 0, 1]), valid=torch.tensor([False, True, True]))
    assert allowed['n'] == 2 and allowed['recall_at_1'] == pytest.approx(1.0)
    # The same holds for labels on excluded candidates: a masked slot must
    # not inject a sentinel class into the binary label contract.
    allowed_labels = binary_ranking_metrics(
        torch.tensor([0.3, 0.1, 0.9]),
        torch.tensor([-1, 0, 1]), valid=torch.tensor([False, True, True]))
    assert allowed_labels['n'] == 2
    assert allowed_labels['recall_at_1'] == pytest.approx(1.0)


def test_tune_baseline_lr_selects_on_validation_stream():
    """Baseline lr selection must run on the held-out monitor stream."""
    from run_quality import tune_baseline_lr

    sweep = [('mqar', 1, 16, 16)]
    best, rec = tune_baseline_lr(
        'linear', [1e-3, 1e-2], sweep, steps=10, softmax_steps=10,
        n_seeds=1, device='cpu', val_every=5, val_batches=1, val_batch=16)
    assert best in (1e-3, 1e-2)
    assert rec['selected_lr'] == best
    assert rec['selection_stream'] == 'val_monitor_seed+6000000'
    assert set(rec['lr_grid']) == {'0.001', '0.01'}
    for entry in rec['lr_grid'].values():
        assert entry['n_val_points'] == 1
        assert entry['mean_final_val_acc'] is not None
        assert entry['n_cell_runs_failed'] == 0


def test_paired_one_sided_t_math_and_degenerate_cases():
    stats = _paired_one_sided_t([0.1, 0.2, 0.3])
    assert stats['n_pairs'] == 3
    assert stats['mean_diff'] == pytest.approx(0.2)
    # mean 0.2, sample std 0.1, n 3 -> t = 0.2 / (0.1 / sqrt(3)).
    assert stats['t_stat'] == pytest.approx(0.2 / (0.1 / (3 ** 0.5)))
    assert stats['all_positive'] is True
    # Zero-variance unanimous improvement: t undefined, but the caller
    # still has ``all_positive`` as evidence.
    flat = _paired_one_sided_t([0.5, 0.5, 0.5])
    assert flat['t_stat'] is None and flat['all_positive'] is True
    # Floating-point zero variance: a constant sweep of a value without an
    # exact binary representation (0.1) leaves a ~1e-17 residue in the
    # naive variance; without the numerical-zero clamp that becomes a fake
    # t ~ 1e16 and silently bypasses the sign-flip evidence rule.
    fp_flat = _paired_one_sided_t([0.1, 0.1, 0.1])
    assert fp_flat['t_stat'] is None and fp_flat['all_positive'] is True
    neg = _paired_one_sided_t([-0.1, -0.2])
    assert neg['all_positive'] is False
    assert _paired_one_sided_t([])['n_pairs'] == 0
    assert _paired_one_sided_t([0.3])['t_stat'] is None


# ---------------------------------------------------------------------------
# run_quality wiring
# ---------------------------------------------------------------------------

def test_monitoring_does_not_change_training_results():
    """Held-out monitoring must be RNG-inert: same curves and metrics."""
    base = dict(seed=7, steps=25, device='cpu', eval_batches=1, eval_batch=16)
    plain = train_one('kda', **base)
    monitored = train_one('kda', **base, val_every=10, val_batches=1,
                          val_batch=16, error_batches=1, error_batch=16)
    assert plain['loss_curve'] == monitored['loss_curve']
    assert plain['acc_curve'] == monitored['acc_curve']
    assert plain['final_acc'] == monitored['final_acc']
    assert plain['final_loss'] == monitored['final_loss']
    # The monitor stream is a different stream from the headline eval.
    assert len(monitored['val_history']) == 4  # steps 0, 10, 20 + final 25
    assert monitored['curve_diagnosis']['verdict'] in {
        'healthy', 'overfitting', 'underfitting', 'diverged'}
    # Reading the curve, not only its endpoint: best-validation point,
    # early-stopping regret, accuracy regime and optimizer health are all
    # recorded next to ``final_acc`` (which keeps its fixed-budget meaning).
    assert monitored['best_val']['step'] in {0, 10, 20, 25}
    assert monitored['early_stopping_summary']['selected_step'] == \
        monitored['best_val']['step']
    assert monitored['steps_since_best'] >= 0
    assert monitored['accuracy_regime']['saturated'] is False
    assert monitored['gradient_norms']['n_steps'] == 25
    assert monitored['gradient_norms']['median'] is not None
    assert monitored['gradient_norms']['n_nonfinite'] == 0
    assert isinstance(monitored['gradient_norms']['notes'], list)


def test_train_one_records_confusion_and_ranking_metrics():
    """Every per-seed record must carry the imbalance-aware metric block.

    Exercises the new simplest baseline ('linear', no token mixing) so the
    metric wiring is also pinned on the identity-mixer path.
    """
    r = train_one('linear', seed=3, steps=10, device='cpu',
                  eval_batches=1, eval_batch=32)
    fm = r['final_metrics']
    assert fm is not None
    assert fm['n_samples'] == 32
    # Averaging conventions are explicit in the key names.
    for key in ('accuracy', 'balanced_accuracy', 'f1_macro', 'f1_micro',
                'f1_weighted', 'precision_macro', 'recall_macro',
                'precision_weighted', 'recall_weighted',
                'pr_auc_macro_ovr', 'roc_auc_macro_ovr', 'per_class',
                'class_balance', 'averaging_note'):
        assert key in fm, key
    # Same pass, same batches: the metric accuracy equals the headline acc.
    assert fm['accuracy'] == pytest.approx(r['final_acc'])
    assert fm['f1_micro'] == pytest.approx(fm['accuracy'])
    assert sum(c['support'] for c in fm['per_class']) == fm['n_samples']


def test_train_multi_seed_single_surviving_seed_reports_undefined_std(
        monkeypatch):
    """One surviving seed -> std/CI are None, never a fake ``0.0000``.

    The sample std is undefined for ``n == 1``; reporting ``0.0`` claims
    perfect precision, and the summary table / logger already render
    ``ci95_acc is None`` as 'n/a'. Pre-fix the std fields were ``0.0``
    while the comment above them documented None.
    """
    import run_quality as rq_mod

    def _fake_train_one(op_name, seed=42, steps=10, device=None,
                        softmax_steps=None, **kw):
        return {
            'op': op_name, 'seed': seed, 'steps': steps, 'n_kv': 1,
            'task': 'mqar', 'n_params': 123, 'lr': 3e-3, 'train_batch': 32,
            'final_acc': 0.5 + 0.01 * seed, 'final_loss': 1.0,
            'chance_acc': 0.0625, 'val_history': [], 'curve_diagnosis': None,
            'error_analysis': None, 'accuracy_regime': None,
            'final_metrics': None, 'loss_curve': [], 'acc_curve': [],
            'generalization_gap': None, 'val_loss_regret_from_best': None,
            'best_val_is_final': None, 'early_stopping_summary': None,
            'final_eval_ms_per_sample': 1.5, 'gradient_norms': None,
        }

    monkeypatch.setattr(rq_mod, 'train_one', _fake_train_one)
    one = rq_mod.train_multi_seed('kda', n_seeds=1, steps=1, device='cpu')
    assert one['n_seeds_ok'] == 1
    assert one['std_acc'] is None and one['std_loss'] is None
    assert one['ci95_acc'] is None and one['ci95_loss'] is None
    assert one['t_stat_vs_chance'] is None
    assert one['mean_acc'] == pytest.approx(0.92)

    two = rq_mod.train_multi_seed('kda', n_seeds=2, steps=1, device='cpu')
    assert two['n_seeds_ok'] == 2
    assert two['std_acc'] is not None and two['std_acc'] > 0

    def _flaky(op_name, seed=42, **kw):
        if seed == 43:
            raise RuntimeError('seed diverged')
        return _fake_train_one(op_name, seed=seed, **kw)

    monkeypatch.setattr(rq_mod, 'train_one', _flaky)
    partial = rq_mod.train_multi_seed('kda', n_seeds=2, steps=1, device='cpu')
    assert partial['n_seeds_ok'] == 1 and partial['n_seeds_failed'] == 1
    # A partially failed multi-seed run is the same n==1 case: undefined
    # std, and the summary table must render it without crashing.
    assert partial['std_acc'] is None and partial['ci95_acc'] is None
    std_str = (f"{partial['std_acc']:>8.4f}"
               if partial['std_acc'] is not None else f"{'n/a':>8}")
    assert std_str.strip() == 'n/a'


def test_error_classification_mqar_layout():
    torch.manual_seed(0)
    # Layout: [k0, v0, k1, v1, noise, cue]; cue = k0, target = v0.
    ids = torch.tensor([[3, 9, 4, 8, 5, 3]])
    target = torch.tensor([9])
    cue_pos = 5
    meta = {'input_ids': ids, 'key_idx': torch.tensor([0])}
    for pred, expected in (
            (torch.tensor([8]), 'confused_with_other_value'),
            (torch.tensor([4]), 'echoed_key'),
            (torch.tensor([5]), 'copied_distractor_token'),
            (torch.tensor([9]), 'correct'),
    ):
        cats, _ = _classify_task_errors('mqar', meta, target, pred, cue_pos, 2)
        assert cats == [expected], (int(pred[0]), cats)


def test_error_confidence_separates_blind_spot_from_near_tie():
    """Wrong is not one failure mode: confidence distinguishes them."""
    logits = torch.tensor([[5.0, 0.0, 0.0]])
    conf = _error_confidence(logits, torch.tensor([2]),
                             torch.tensor([0]), 0)
    probs = torch.softmax(logits[0], dim=-1)
    assert conf['pred_prob'] == pytest.approx(float(probs[0]))
    assert conf['target_prob'] == pytest.approx(float(probs[2]))
    assert conf['margin'] == pytest.approx(float(probs[0] - probs[2]))
    assert conf['target_rank'] == 2
    assert 0.0 <= conf['entropy_ratio'] <= 1.0
    assert conf['pred_prob'] >= 0.5  # confident blind spot

    near_tie = torch.tensor([[0.01, 0.0, 0.0]])
    conf = _error_confidence(near_tie, torch.tensor([2]),
                             torch.tensor([0]), 0)
    assert conf['margin'] < 0.01
    assert conf['entropy_ratio'] > 0.9  # ambiguous for the model

    bad = torch.tensor([[float('nan'), 0.0, 0.0]])
    assert _error_confidence(bad, torch.tensor([2]),
                             torch.tensor([0]), 0) is None


def test_error_classification_kv_update_stale_value():
    # Layout: [k0,v0_old, k1,v1_old, k1,v1_new, k0,v0_new, cue]; cue=k0.
    # keys2 (shuffled) = [k1, k0]; old value for k0 is 10; new value is 13.
    ids = torch.tensor([[3, 10, 4, 11, 4, 12, 3, 13, 3]])
    target = torch.tensor([13])
    cue_pos = 8
    meta = {'input_ids': ids, 'key_idx': torch.tensor([0])}
    cats, extras = _classify_task_errors('kv_update', meta, target,
                                         torch.tensor([10]), cue_pos, 2)
    assert cats == ['used_stale_value']
    assert int(extras['stale_value'][0]) == 10
    cats, _ = _classify_task_errors('kv_update', meta, target,
                                    torch.tensor([12]), cue_pos, 2)
    assert cats == ['confused_with_other_new_value']


def test_error_classification_multi_agg_runner_up():
    ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    counts = torch.tensor([[3, 2, 1]])
    meta = {'input_ids': ids, 'counts': counts.tolist()}
    target = torch.tensor([0])
    cats, extras = _classify_task_errors('multi_agg', meta, target,
                                         torch.tensor([1]), cue_pos=5,
                                         n_kv=1)
    assert cats == ['runner_up_category']
    assert int(extras['counts'][0, 1]) == 2
    cats, _ = _classify_task_errors('multi_agg', meta, target,
                                    torch.tensor([2]), cue_pos=5, n_kv=1)
    assert cats == ['echoed_last_token']


def test_evaluate_lm_loss_nonfinite_refuses_with_diagnosis():
    """A diverged held-out pass must refuse the metric AND say why."""
    from train_lm_autodl import evaluate_lm_loss

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {'input_ids': torch.tensor([1, 2, 3]),
                    'labels': torch.tensor([2, 3, 4])}

    class NaNLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hybrid = torch.nn.Module()
            self.hybrid.reset_state = lambda: None

        def forward(self, input_ids, labels=None):
            return {'loss': torch.tensor(float('nan'))}

    with pytest.raises(RuntimeError, match=r'Diagnosis \[loss_nonfinite\]'):
        evaluate_lm_loss(NaNLossModel(), DummyDataset(), batch_size=1)


def test_analyze_errors_refuses_nonfinite_logits():
    """A diverged forward pass must not produce a plausible error mix."""
    from run_quality import analyze_errors

    vocab = 8

    class _BadLayer(torch.nn.Module):
        def forward(self, x):
            return torch.full_like(x, float('nan'))

    class _BadHead(torch.nn.Module):
        def forward(self, h, cue_pos):
            return torch.full((int(h.shape[0]), vocab), float('nan'))

    with pytest.raises(ValueError, match='non-finite'):
        analyze_errors(_BadLayer(), _BadHead(), torch.nn.Embedding(vocab, 4),
                       op_name='bad', task='mqar', seq_len=8, n_kv=1,
                       vocab=vocab, n_batches=1, batch=4, max_examples=1)


def test_error_analysis_report_is_json_serializable():
    r = train_one('kda', seed=3, steps=12, device='cpu', val_every=6,
                  val_batches=1, val_batch=8, error_batches=1,
                  error_batch=8, eval_batches=1, eval_batch=8)
    ea = r['error_analysis']
    assert ea is not None
    assert ea['n_samples'] == 8
    assert sum(v['count'] for v in ea['categories'].values()) == ea['n_wrong']
    # A wrong answer carries the model's confidence, so an ambiguous
    # near-tie can be separated from a confident blind spot.
    assert ea['confidence'] is not None
    assert ea['confidence']['n_scored_wrong'] == ea['n_wrong']
    for cat in ea['categories'].values():
        if cat['count']:
            assert cat['confidence'] is not None
    for bucket in ea['examples'].values():
        for ex in bucket:
            for key in ('pred_prob', 'target_prob', 'margin',
                        'target_rank', 'entropy_nats', 'entropy_ratio'):
                assert key in ex
            assert 0.0 <= ex['pred_prob'] <= 1.0
            assert 0.0 <= ex['target_prob'] <= 1.0
    json.dumps(r, allow_nan=False)


# ---------------------------------------------------------------------------
# Token-level LM bad-case diagnostics (train_lm_autodl wiring)
# ---------------------------------------------------------------------------

def _row_logits(vocab, target, rank, top1=None):
    """Logits where the target has exactly ``rank - 1`` larger entries."""
    logits = torch.full((vocab,), -1.0)
    logits[target] = 0.0
    others = [i for i in range(vocab) if i != target]
    for i in others[:rank - 1]:
        logits[i] = 1.0
    if top1 is not None:
        logits[others[0]] = top1
    return logits


def _toy_diagnostics():
    """Five positions covering every error category in the partition."""
    vocab = 8
    # Context for the prediction at t is input_ids[:t+1] (the current token
    # is already consumed); distance is measured from the predicted t+1.
    input_ids = torch.tensor([[1, 2, 1, 4, 2]])
    labels = torch.tensor([[2, 1, 4, 5, 2]])
    rows = [
        _row_logits(vocab, 2, 1),              # correct
        _row_logits(vocab, 1, 2),              # near_miss (seen, dist 2)
        _row_logits(vocab, 4, 6),              # novel_token
        _row_logits(vocab, 5, 6, top1=8.0),    # confident_wrong
        _row_logits(vocab, 2, 6),              # copyable_missed (dist 1)
    ]
    logits = torch.stack(rows).unsqueeze(0)
    return lm_token_diagnostics(logits, labels, input_ids,
                                top_k=5, confidence=0.5)


def test_lm_token_diagnostics_categories_and_copy_distance():
    d = _toy_diagnostics()
    assert d['category'][0] == ['correct', 'near_miss', 'novel_token',
                                'confident_wrong', 'copyable_missed']
    assert d['pred'].tolist() == [[2, 0, 0, 0, 0]]
    assert d['target_rank'].tolist() == [[1, 2, 6, 6, 6]]
    assert d['seen_earlier'].tolist() == [[False, True, False, False, True]]
    assert d['last_seen_distance'].tolist() == [[-1, 2, -1, -1, 1]]
    # Confident wrong really is confident; the copy miss is not.
    assert float(d['top1_prob'][0, 3]) > 0.9
    assert float(d['top1_prob'][0, 4]) < 0.5


def test_lm_token_diagnostics_counts_immediate_repetition_as_copyable():
    """The token just consumed is context for the next prediction.

    A strict-prefix mask made ``... 3, 3`` look like an unseen target, so a
    failed immediate copy was reported as ``novel_token`` (intrinsic
    difficulty) instead of ``copyable_missed`` (retrieval failure).
    """
    vocab = 6
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 3]])
    logits = torch.stack([
        _row_logits(vocab, 2, 1),
        _row_logits(vocab, 3, 1),
        _row_logits(vocab, 3, 5),
    ]).unsqueeze(0)
    d = lm_token_diagnostics(logits, labels, input_ids, top_k=2)
    assert d['seen_earlier'].tolist() == [[False, False, True]]
    assert d['last_seen_distance'].tolist() == [[-1, -1, 1]]
    assert d['category'][0] == ['correct', 'correct', 'copyable_missed']

    s = summarize_lm_diagnostics(d)
    dist = {b['bucket']: b for b in s['accuracy_by_repeat_distance']}
    assert dist['d01_08']['n'] == 1
    assert dist['not_in_context']['n'] == 2


def test_summarize_lm_diagnostics_reports_both_axes():
    s = summarize_lm_diagnostics([_toy_diagnostics(), _toy_diagnostics()])
    assert s['n_tokens'] == 10 and s['n_wrong'] == 8
    assert s['token_accuracy'] == pytest.approx(0.2)
    counts = {k: v['count'] for k, v in s['categories'].items()}
    assert counts == {'correct': 2, 'near_miss': 2, 'confident_wrong': 2,
                      'copyable_missed': 2, 'novel_token': 2, 'ignored': 0}
    dist = {b['bucket']: b for b in s['accuracy_by_repeat_distance']}
    assert dist['d01_08']['n'] == 4 and dist['d01_08']['accuracy'] == 0.0
    assert dist['not_in_context']['n'] == 6
    assert dist['not_in_context']['accuracy'] == pytest.approx(1 / 3)
    pos = {b['quartile']: b for b in s['accuracy_by_position_quartile']}
    assert pos['q1_of_4']['n'] == 4
    assert pos['q1_of_4']['accuracy'] == pytest.approx(0.5)
    assert s['target_rank_on_errors']['median'] == pytest.approx(6.0)
    json.dumps(s, allow_nan=False)


def _all_correct_diag(T):
    logits = torch.zeros(1, T, 3)
    labels = torch.zeros(1, T, dtype=torch.long)
    input_ids = torch.zeros(1, T, dtype=torch.long)
    return lm_token_diagnostics(logits, labels, input_ids)


def test_position_quartiles_use_full_sequence_length_not_last_valid():
    """An ignored padded tail must not compress the position axis.

    With T=8 and only positions 0..1 valid, deriving the width from the
    last VALID position (the old ``pos.max()+1``) put position 1 in q3
    instead of q1. Quartiles are relative to the sequence length.
    """
    T = 8
    logits = torch.zeros(1, T, 3)
    labels = torch.full((1, T), -100, dtype=torch.long)
    labels[0, :2] = 0
    input_ids = torch.zeros(1, T, dtype=torch.long)
    s = summarize_lm_diagnostics(lm_token_diagnostics(
        logits, labels, input_ids))
    q = {b['quartile']: b for b in s['accuracy_by_position_quartile']}
    assert [q[f'q{i}_of_4']['n'] for i in range(1, 5)] == [2, 0, 0, 0]
    assert q['q1_of_4']['position_range'] == [0, 2]
    assert q['q4_of_4']['position_range'] == [6, 8]


def test_position_quartiles_are_relative_to_each_sequence_length():
    """Batched diagnostics of different lengths get per-sequence bins."""
    s = summarize_lm_diagnostics([_all_correct_diag(4),
                                  _all_correct_diag(8)])
    q = {b['quartile']: b for b in s['accuracy_by_position_quartile']}
    assert [q[f'q{i}_of_4']['n'] for i in range(1, 5)] == [3, 3, 3, 3]
    for i in range(1, 5):
        entry = q[f'q{i}_of_4']
        assert entry['position_range'] is None
        assert entry['relative_position_range'] == [
            (i - 1) / 4, i / 4]
    json.dumps(s, allow_nan=False)


def test_lm_token_diagnostics_rejects_malformed_inputs():
    vocab = 4
    logits = torch.zeros(1, 2, vocab)
    labels = torch.tensor([[0, 1]])
    input_ids = torch.tensor([[1, 0]])
    lm_token_diagnostics(logits, labels, input_ids)

    for bad_labels in (torch.tensor([[0.0, 1.0]]),
                       torch.tensor([[False, True]])):
        with pytest.raises(ValueError):
            lm_token_diagnostics(logits, bad_labels, input_ids)
    for bad_labels in (torch.tensor([[0, vocab]]),
                       torch.tensor([[0, -2]])):
        with pytest.raises(ValueError, match='outside'):
            lm_token_diagnostics(logits, bad_labels, input_ids)
    # The ignore index may itself be negative and is not a bad label.
    lm_token_diagnostics(logits, torch.tensor([[0, -100]]), input_ids)
    with pytest.raises(ValueError):
        lm_token_diagnostics(logits, labels, input_ids.to(torch.float32))
    with pytest.raises(ValueError):
        lm_token_diagnostics(logits.to(torch.long), labels, input_ids)
    for bad_value in (float('nan'), float('inf'), float('-inf')):
        bad_logits = logits.clone()
        bad_logits[0, 0, 0] = bad_value
        with pytest.raises(ValueError, match='non-finite'):
            lm_token_diagnostics(bad_logits, labels, input_ids)
    for bad_k in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            lm_token_diagnostics(logits, labels, input_ids, top_k=bad_k)
    for bad_conf in (-0.1, 1.1, float('nan'), True):
        with pytest.raises(ValueError):
            lm_token_diagnostics(logits, labels, input_ids,
                                 confidence=bad_conf)
    for bad_ignore in (True, 1.5):
        with pytest.raises(ValueError):
            lm_token_diagnostics(logits, labels, input_ids,
                                 ignore_index=bad_ignore)


def test_lm_token_diagnostics_masked_and_empty():
    vocab = 4
    logits = torch.stack([_row_logits(vocab, 2, 1),
                          _row_logits(vocab, 1, 1),
                          _row_logits(vocab, 3, 1)]).unsqueeze(0)
    d = lm_token_diagnostics(logits, torch.tensor([[2, -100, 3]]),
                             torch.tensor([[1, 2, 3]]))
    assert d['category'][0] == ['correct', 'ignored', 'correct']
    s = summarize_lm_diagnostics(d)
    assert s['n_tokens'] == 2 and s['n_wrong'] == 0
    assert s['target_rank_on_errors']['mean'] is None
    json.dumps(s, allow_nan=False)

    d0 = lm_token_diagnostics(torch.zeros(2, 0, 5),
                              torch.zeros(2, 0, dtype=torch.long),
                              torch.zeros(2, 0, dtype=torch.long))
    s0 = summarize_lm_diagnostics(d0)
    assert s0['n_tokens'] == 0
    json.dumps(s0, allow_nan=False)


def test_rate_of_wrong_only_covers_error_categories():
    """'rate_of_wrong' is the share of the WRONG tokens per category.

    'correct' and 'ignored' complete the position partition but are not
    errors: a correct prediction is not a mistake and an ignored position
    is masked out of the loss. Dividing their counts by the error count
    reported nonsense (e.g. "67% of the errors were masked positions",
    and a rate above 1.0 on padding-heavy batches).
    """
    vocab = 4
    logits = torch.stack([
        _row_logits(vocab, 2, 1),   # correct
        _row_logits(vocab, 1, 2),   # near_miss
        _row_logits(vocab, 3, 2),   # near_miss
        _row_logits(vocab, 1, 2),
    ]).unsqueeze(0)
    labels = torch.tensor([[2, 1, 3, 1]])
    d = lm_token_diagnostics(logits, labels, torch.tensor([[1, 2, 3, 4]]))
    s = summarize_lm_diagnostics(d)
    cats = s['categories']
    assert s['n_wrong'] == 3
    # The error categories still partition the wrong tokens exactly.
    error_total = sum(cats[name]['rate_of_wrong']
                      for name in cats if name not in ('correct', 'ignored'))
    assert error_total == pytest.approx(1.0)
    # The non-error categories contribute no share of the wrong tokens.
    assert cats['correct']['rate_of_wrong'] == 0.0
    assert cats['ignored']['rate_of_wrong'] == 0.0
    # Every category's rate_of_wrong stays within [0, 1].
    for name, v in cats.items():
        assert 0.0 <= v['rate_of_wrong'] <= 1.0, (name, v)


class _FixedLogitModel(torch.nn.Module):
    """Deterministic LM stand-in: logits depend only on the input ids."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.hybrid = torch.nn.Module()
        self.hybrid.reset_state = lambda: None
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels=None):
        eye = torch.eye(self.vocab)
        return {'logits': eye[input_ids.clamp(0, self.vocab - 1)]}


class _IdsDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, seq_len=6, vocab=8):
        self.items = []
        for i in range(n):
            ids = [(i + j) % vocab for j in range(seq_len + 1)]
            self.items.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:])))
        self.split = 'validation'

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {'input_ids': self.items[i][0], 'labels': self.items[i][1]}


def test_analyze_lm_errors_read_only_deterministic_json(capsys):
    from train_lm_autodl import _FallbackCharTokenizer, analyze_lm_errors

    model = _FixedLogitModel(8)
    model.train()
    dataset = _IdsDataset()
    rng_before = torch.get_rng_state()
    rep1 = analyze_lm_errors(model, dataset, batch_size=2, max_batches=1,
                             seed=0, max_examples=1)
    rep2 = analyze_lm_errors(model, dataset, batch_size=2, max_batches=1,
                             seed=0, max_examples=1)
    # Read-only contract: eval mode restored and the global RNG untouched
    # (dedicated DataLoader generator), so the analysis cannot perturb
    # training or the sealed test evaluation.
    assert model.training is True
    assert torch.equal(rng_before, torch.get_rng_state())
    assert json.dumps(rep1, allow_nan=False) == json.dumps(rep2, allow_nan=False)
    assert rep1['split'] == 'validation'
    assert rep1['n_batches_analyzed'] == 1
    assert rep1['n_tokens'] == 12
    assert rep1['n_wrong'] == 12
    assert rep1['stream'].startswith('held-out split')
    out = capsys.readouterr().out
    assert '[bad-case] LM held-out' in out
    # max_batches=0 must analyze nothing (the training script's disable path).
    rep0 = analyze_lm_errors(model, dataset, batch_size=2, max_batches=0,
                             seed=0)
    assert rep0['n_batches_analyzed'] == 0

    tok = _FallbackCharTokenizer()
    assert tok.decode(tok.encode('hello world')) == 'hello world'


def test_analyze_lm_errors_sample_ids_point_at_the_dataset_row():
    """The recorded ``sample`` id must be the dataset row, not a batch count.

    A batch whose labels carry no supervised token is skipped by the
    analyzer.  The row index of the samples in the FOLLOWING batches must
    still be the dataset rows they came from, so the bad-case report can be
    traced back to the exact held-out examples.
    """
    from train_lm_autodl import analyze_lm_errors

    class _PartlyUnsupervisedDataset(torch.utils.data.Dataset):
        """Rows 0-1 have no supervised token; rows 2-5 do."""

        def __init__(self):
            self.split = 'validation'

        def __len__(self):
            return 6

        def __getitem__(self, i):
            # Valid token ids for the stand-in's vocab=8, with a deliberate
            # off-by-one label so a wrong prediction is available. The test
            # is about row indices, not label validity.
            ids = (torch.arange(4) + i) % 8
            labels = (ids + 1) % 8
            if i < 2:
                labels[:] = -100
            return {'input_ids': ids, 'labels': labels}

    model = _FixedLogitModel(8)
    rep = analyze_lm_errors(model, _PartlyUnsupervisedDataset(),
                             batch_size=2, max_batches=5, seed=0,
                             max_examples=10)
    samples = [ex['sample']
               for bucket in rep['examples'].values() for ex in bucket]
    assert samples, 'expected at least one collected bad case'
    # The first batch (rows 0-1) has no supervised token and is skipped;
    # every recorded sample must therefore be a row of a LATER batch
    # (rows 2-5), never a row of the skipped one.
    assert min(samples) >= 2, f'sample ids {sorted(samples)} point into the skipped batch'
    assert max(samples) <= 5, f'sample ids {sorted(samples)} exceed the dataset'


def test_hypothesis_verdicts_are_machine_readable_and_rule_based():
    """P2-2: pre-registered hypotheses get rule-based, per-row verdicts."""
    chance = 0.0625

    def row(op, n_kv, seed_accs, *, chance_acc=chance, sig_chance=False,
            beats=None):
        mean = sum(seed_accs) / len(seed_accs)
        r = {
            'op': op, 'task': 'mqar', 'n_kv': n_kv,
            'mean_acc': mean, 'chance_acc': chance_acc,
            'significant_bonferroni': sig_chance,
            't_stat_vs_chance': 0.0 if not sig_chance else 99.0,
            'per_seed': [{'seed': 42 + i, 'final_acc': a}
                         for i, a in enumerate(seed_accs)],
        }
        if beats is not None:
            r['beats_tuned_baseline'] = beats
            r['vs_tuned_baseline'] = {
                'reference_op': 'softmax', 'n_pairs': len(seed_accs),
                'mean_diff': 0.1, 't_stat': None, 'significant': beats,
                'verdict': ('beats_tuned_baseline' if beats
                            else 'not_significantly_better'),
            }
        return r

    linear = row('linear', 1, [chance, chance])
    softmax = row('softmax', 1, [0.3, 0.3])
    kda_low = row('kda', 1, [0.5, 0.5], beats=True)
    kda_high = row('kda', 2, [0.3, 0.3], beats=True)
    # Reference is unanimously (8/8) better than hca: the reverse one-sided
    # exact sign-flip p = 2^-7 = 0.0078 clears alpha=0.05, so the
    # superiority claim for hca is contradicted (not merely unsupported).
    hca = row('hca', 1, [0.2] * 8)
    softmax['per_seed'] = [{'seed': 42 + i, 'final_acc': 0.3}
                           for i in range(8)]
    hca['per_seed'] = [{'seed': 42 + i, 'final_acc': 0.2}
                       for i in range(8)]

    checks, counts = _hypothesis_verdicts(
        [linear, softmax, kda_low, kda_high, hca],
        baseline_ops=['linear', 'softmax'], baseline_ref='softmax')

    assert checks['linear_chance_control']['status'] == 'supported'
    assert checks['kda_nkv_monotonicity']['status'] == 'supported'
    assert checks['beats_tuned_baseline']['status'] == 'contradicted'
    assert checks['beats_tuned_baseline']['falsified'] is True
    # Per-row compact status + per-rule evidence.
    assert linear['hypothesis_status'] == 'supported'
    assert softmax['hypothesis_status'] == 'not_applicable'
    assert kda_high['hypothesis_status'] == 'supported'
    assert hca['hypothesis_status'] == 'contradicted'
    assert hca['hypothesis_checks']['beats_tuned_baseline']['falsified'] is True
    assert counts == {'supported': 3, 'contradicted': 1,
                      'not_supported': 0, 'not_computable': 0,
                      'not_applicable': 1}
    # JSON-safe: the metadata block must serialize into the result file.
    json.dumps({'checks': checks, 'counts': counts}, allow_nan=False)


def test_linear_control_above_chance_without_significance_is_not_supported():
    """A failed superiority test is not evidence that linear is at chance."""
    chance = 0.0625
    linear = {
        'op': 'linear', 'task': 'mqar', 'n_kv': 1,
        # 3x chance but not significant at this seed budget.
        'mean_acc': 0.1875, 'chance_acc': chance,
        'significant_bonferroni': False,
        't_stat_vs_chance': 1.2,
        'per_seed': [{'seed': 42 + i, 'final_acc': a}
                     for i, a in enumerate([0.10, 0.20, 0.26])],
    }
    checks, counts = _hypothesis_verdicts(
        [linear], baseline_ops=['linear'], baseline_ref='softmax')
    assert checks['linear_chance_control']['status'] == 'not_supported'
    assert checks['linear_chance_control']['falsified'] is False
    assert checks['linear_chance_control']['n_inconclusive_cells'] == 1
    assert checks['linear_chance_control']['inconclusive_cells'] == [
        'mqar/n_kv=1']
    assert linear['hypothesis_status'] == 'not_supported'
    assert counts['supported'] == 0 and counts['not_supported'] == 1
    assert linear['hypothesis_checks'][
        'linear_chance_control']['evidence']['prediction_holds'] is None


def test_hypothesis_verdicts_mark_missing_evidence_not_computable():
    """Missing chance / paired seeds are not evidence for or against."""
    def row(op, n_kv, seed_accs, *, chance_acc=0.1, seeds=None):
        return {
            'op': op, 'task': 'mqar', 'n_kv': n_kv,
            'mean_acc': sum(seed_accs) / len(seed_accs),
            'chance_acc': chance_acc,
            'significant_bonferroni': False,
            't_stat_vs_chance': 0.0,
            'per_seed': [
                {'seed': (seeds[i] if seeds else 42 + i), 'final_acc': a}
                for i, a in enumerate(seed_accs)],
        }

    linear = row('linear', 1, [0.3, 0.3], chance_acc=None)
    softmax = row('softmax', 1, [0.5, 0.5])
    # The two KDA cells share no seed, so their monotonicity pair is not
    # computable even though both cells have a mean accuracy.
    kda_low = row('kda', 1, [0.9, 0.9], seeds=[42, 43])
    kda_high = row('kda', 2, [0.1, 0.1], seeds=[44, 45])

    checks, _ = _hypothesis_verdicts(
        [linear, softmax, kda_low, kda_high],
        baseline_ops=['linear', 'softmax'], baseline_ref='softmax')

    assert checks['linear_chance_control']['status'] == 'not_computable'
    assert linear['hypothesis_checks'][
        'linear_chance_control']['status'] == 'not_computable'
    assert linear['hypothesis_status'] == 'not_computable'
    assert checks['kda_nkv_monotonicity']['status'] == 'not_computable'
    assert checks['kda_nkv_monotonicity']['missing_pairs'] == [
        'mqar: n_kv 1->2']


def test_parse_lr_list_rejects_non_finite(monkeypatch):
    monkeypatch.setenv('TEST_LR_LIST', '1e-3,1e999')
    with pytest.raises(ValueError, match='finite'):
        _parse_lr_list('TEST_LR_LIST')
    assert _parse_lr_list('TEST_LR_LIST_MISSING', default='1e-3,3e-3') == [
        1e-3, 3e-3]


def test_tune_baseline_lr_selects_on_tail_plateau(monkeypatch):
    """P2-1: selection uses the converged tail, not one noisy final point."""
    import run_quality

    curves = {
        1e-3: [0.9, 0.9, 0.9, 0.1],   # best plateau, unlucky final point
        1e-2: [0.5, 0.5, 0.5, 0.5],   # worse plateau, good final point
    }

    def fake_train_multi_seed(op_name, n_seeds=1, steps=10,
                              softmax_steps=None, device='cpu', lr=3e-3,
                              **kw):
        return {'per_seed': [{'val_history': [{'acc': p} for p in curves[lr]]}]}

    monkeypatch.setattr(run_quality, 'train_multi_seed',
                        fake_train_multi_seed)
    best, rec = run_quality.tune_baseline_lr(
        'linear', [1e-3, 1e-2], [('mqar', 1, 16, 16)],
        steps=10, softmax_steps=10, n_seeds=1, device='cpu',
        val_every=1, val_batches=1, val_batch=16)
    assert best == 1e-3
    assert rec['val_tail_k'] == 3
    assert 'tail' in rec['selection_statistic']
    assert rec['selected_lr_mean_val_acc'] == pytest.approx(1.9 / 3)
    assert rec['selected_lr_mean_final_val_acc'] == pytest.approx(0.1)
    assert rec['lr_grid']['0.001']['mean_val_tail_acc'] == pytest.approx(1.9 / 3)
    assert rec['lr_grid']['0.001']['mean_final_val_acc'] == pytest.approx(0.1)


def test_sweep_cells_match_task_defaults():
    """The design-audit family must count every actual sweep cell.

    Tasks that do not sweep n_kv run only their default cell; tasks that do
    run one cell per requested n_kv. Both the sweep construction and the
    prospective Exp 4 family size now share this rule.
    """
    import run_quality
    assert run_quality._sweep_cells(['mqar'], [1, 2]) == [
        ('mqar', 1), ('mqar', 2)]
    assert run_quality._sweep_cells(['long_range'], [1, 2, 4]) == [
        ('long_range', 1)]
    assert run_quality._sweep_cells(['mqar', 'long_range'], [1, 2]) == [
        ('mqar', 1), ('mqar', 2), ('long_range', 1)]


def test_nonfinite_diagnosis_skips_non_tensor_parameter_maps():
    import json
    for params in ({'w': None}, [None], {'w': 'not-a-tensor'}):
        rep = diagnose_nonfinite(loss=1.0, params=params)
        check = [c for c in rep['checks']
                 if c['check'] == 'parameters'][0]
        assert check['status'] == 'skipped', check
        json.dumps(rep, allow_nan=False)
