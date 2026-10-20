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
    pareto_frontier,
    summarize_lm_diagnostics,
)
from run_quality import (  # noqa: E402
    _classify_task_errors,
    _hypothesis_verdicts,
    _paired_one_sided_t,
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


def test_pareto_frontier_skips_missing_costs():
    rows = [
        {'op': 'a', 'mean_acc': 0.5, 'n_params': 10, 'mean_train_time_s': 1.0},
        {'op': 'b', 'mean_acc': 0.6, 'n_params': None,
         'mean_train_time_s': 2.0},
    ]
    rep = pareto_frontier(rows)
    assert rep['skipped'] == ['b']
    assert rep['frontier'] == ['a']


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


def test_error_analysis_report_is_json_serializable():
    r = train_one('kda', seed=3, steps=12, device='cpu', val_every=6,
                  val_batches=1, val_batch=8, error_batches=1,
                  error_batch=8, eval_batches=1, eval_batch=8)
    ea = r['error_analysis']
    assert ea is not None
    assert ea['n_samples'] == 8
    assert sum(v['count'] for v in ea['categories'].values()) == ea['n_wrong']
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
    # Prefix at t: input_ids[:t]; target seen earlier drives copy distance.
    input_ids = torch.tensor([[1, 2, 1, 4, 2]])
    labels = torch.tensor([[2, 1, 4, 5, 2]])
    rows = [
        _row_logits(vocab, 2, 1),              # correct
        _row_logits(vocab, 1, 2),              # near_miss (seen, dist 1)
        _row_logits(vocab, 4, 6),              # novel_token
        _row_logits(vocab, 5, 6, top1=8.0),    # confident_wrong
        _row_logits(vocab, 2, 6),              # copyable_missed (dist 3)
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
    assert d['last_seen_distance'].tolist() == [[-1, 1, -1, -1, 3]]
    # Confident wrong really is confident; the copy miss is not.
    assert float(d['top1_prob'][0, 3]) > 0.9
    assert float(d['top1_prob'][0, 4]) < 0.5


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
