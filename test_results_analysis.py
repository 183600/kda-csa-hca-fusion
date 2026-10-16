"""Tests for the results-analysis tooling.

Covers ``results_analysis`` (curve-shape diagnosis, systematic NaN/Inf
triage, efficiency/accuracy Pareto annotation) and the ``run_quality``
wiring (held-out monitoring is RNG-inert; bad-case classification is
task-correct and recordable).

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
    annotate_rows_with_pareto,
    diagnose_nonfinite,
    diagnose_training_curves,
    pareto_frontier,
)
from run_quality import _classify_task_errors, train_one  # noqa: E402


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
