"""Results-analysis helpers for the experiment scripts.

Each helper addresses one concrete failure mode of "look at the final
number and move on":

1. :func:`diagnose_training_curves` — turns a train/val loss history into
   an overfitting / underfitting / healthy / diverged verdict with the
   numeric evidence and concrete next actions, instead of only comparing
   the last loss value.
2. :func:`diagnose_nonfinite` — the systematic NaN/Inf triage path, in
   order: data -> token-id range -> loss -> parameters -> logits
   (softmax overflow) -> AMP scaler / fp32-forward comparison -> gradients
   (explosion vs bad LR). Returns a structured report naming the FIRST
   failing stage, not just "loss is NaN".
3. :func:`pareto_frontier` — efficiency-vs-accuracy annotation: which
   operators are dominated once parameter count and train time are taken
   into account, so a small accuracy gain at a large cost is visible in
   the results rather than buried in a separate latency experiment.

All functions are pure (no RNG, no global state) so they can be called
from training loops, tests, and notebooks without perturbing an
experiment.
"""

from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from typing import Iterable, Mapping, Sequence

import torch


def _finite(values: Sequence[float]) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


def _window_mean(values: Sequence[float], k: int) -> float | None:
    if not values:
        return None
    k = max(1, min(int(k), len(values)))
    return float(sum(values[-k:]) / k)


def _head_mean(values: Sequence[float], frac: float = 0.1) -> float | None:
    if not values:
        return None
    k = max(1, int(math.ceil(len(values) * frac)))
    return float(sum(values[:k]) / k)


def _curve_stats(losses: Sequence[float]) -> dict:
    """Head/tail/best summary of one loss curve."""
    n = len(losses)
    head_k = max(1, int(math.ceil(n * 0.1)))
    tail_k = max(1, int(math.ceil(n * 0.1)))
    head = float(sum(losses[:head_k]) / head_k)
    tail = float(sum(losses[-tail_k:]) / tail_k)
    best = float(min(losses))
    best_step = int(min(range(n), key=lambda i: losses[i]))
    relative_change = ((tail - head) / abs(head)) if head != 0.0 else None
    return {
        'n_points': n,
        'first': float(losses[0]),
        'last': float(losses[-1]),
        'head_mean': head,
        'tail_mean': tail,
        'best': best,
        'best_step': best_step,
        'relative_change': relative_change,
    }


def diagnose_training_curves(train_losses: Sequence[float],
                             val_losses: Sequence[float] | None = None,
                             *, relative_improvement: float = 0.02,
                             val_rise_fraction: float = 0.02) -> dict:
    """Diagnose over-/under-fitting from the monitored curves.

    ``train_losses`` is the per-optimizer-step training loss; ``val_losses``
    (optional) is the held-out loss measured at checkpoints. The verdict is
    based on the SHAPE of the curves, not on the final values:

      * ``diverged``    — a non-finite loss anywhere (run
        :func:`diagnose_nonfinite` on the failing step).
      * ``overfitting`` — train improved by more than
        ``relative_improvement`` while validation regressed from its best by
        more than ``val_rise_fraction`` (or the train/val gap grew while
        validation stopped improving).
      * ``underfitting``— neither curve improved: wrong LR, optimizer or
        data-pipeline problem, not a capacity problem yet.
      * ``healthy``     — train improved and validation improved or tracked.
      * ``train_improving_only`` — train improved but no validation curve
        was recorded; generalization cannot be judged, so monitoring is
        recommended before trusting the run.
      * ``insufficient_validation_curve`` — a validation curve was recorded,
        but with fewer than 2 points it has no shape, so no over/under
        verdict is claimed.

    Returns a JSON-serializable report with the numeric evidence
    (head/tail/best means, gap) and a list of concrete recommendations.
    """
    report: dict = {
        'verdict': 'no_data',
        'train': None,
        'val': None,
        'signals': {},
        'recommendations': [],
    }
    train = [float(v) for v in train_losses]
    if not train:
        report['recommendations'].append(
            'no training curve was recorded; log the per-step loss so the '
            'run can be diagnosed without re-training.')
        return report

    report['train'] = _curve_stats(train) if _finite(train) else {
        'n_points': len(train),
        'non_finite_at': next(
            (i for i, v in enumerate(train)
             if v is None or not math.isfinite(float(v))), None),
    }
    if not _finite(train):
        report['verdict'] = 'diverged'
        report['recommendations'].append(
            'training loss became non-finite: run diagnose_nonfinite() on '
            'the failing step (check data, log(0)/div-by-zero, softmax '
            'overflow, AMP fp16 overflow, then gradients/LR).')
        return report

    val = None
    if val_losses is not None:
        val = [float(v) for v in val_losses]
    if val is not None and val and not _finite(val):
        report['val'] = {
            'n_points': len(val),
            'non_finite_at': next(
                (i for i, v in enumerate(val)
                 if v is None or not math.isfinite(float(v))), None),
        }
        report['verdict'] = 'diverged'
        report['recommendations'].append(
            'validation loss became non-finite (training itself may still '
            'be finite): run diagnose_nonfinite() on the eval forward pass; '
            'check the eval batch for NaN/Inf and any fp16 eval autocast.')
        return report

    tr = report['train']
    train_head, train_tail = tr['head_mean'], tr['tail_mean']
    train_improved = (train_head > 0
                      and train_tail < train_head * (1.0 - relative_improvement))
    signals = {
        'train_improved': bool(train_improved),
        'train_head_mean': train_head,
        'train_tail_mean': train_tail,
        'train_relative_change': tr['relative_change'],
    }

    if val is not None and val:
        va = _curve_stats(val)
        report['val'] = va
        signals['val_n_points'] = va['n_points']
        if len(val) < 2:
            # A single validation point has no shape: it can neither show a
            # turning point nor a flat curve, so "overfitting" / "healthy"
            # would be an over-claim. Say what is missing instead.
            report['verdict'] = 'insufficient_validation_curve'
            report['signals'] = signals
            report['recommendations'] = [
                'only one validation point was recorded: the curve has no '
                'shape, so no over/underfitting verdict is supported. Lower '
                'the validation interval (or raise the step budget) so the '
                'validation curve has at least 2-3 points, then re-diagnose; '
                'until then the run is only known to be '
                '"train_improving_only".',
            ]
            return report
        val_head, val_tail, val_best = va['head_mean'], va['tail_mean'], va['best']
        val_rose_from_best = val_tail > val_best * (1.0 + val_rise_fraction)
        val_improved = (val_head > 0
                        and val_tail < val_head * (1.0 - relative_improvement))
        head_gap = val_head - train_head
        tail_gap = val_tail - train_tail
        signals.update({
            'val_improved': bool(val_improved),
            'val_head_mean': val_head,
            'val_tail_mean': val_tail,
            'val_best': val_best,
            'val_best_step': va['best_step'],
            'val_rose_from_best': bool(val_rose_from_best),
            'generalization_gap_head': float(head_gap),
            'generalization_gap_tail': float(tail_gap),
        })
        gap_grew = (tail_gap > max(0.0, head_gap) * 2.0 + 0.1
                    and val_tail > val_head * 0.999)
        signals['gap_grew'] = bool(gap_grew)
        if train_improved and (val_rose_from_best or gap_grew):
            report['verdict'] = 'overfitting'
            report['recommendations'] = [
                'train loss fell while validation regressed or the gap '
                'widened: add regularization (weight decay / dropout), '
                'increase data, or use early stopping at the best '
                'validation checkpoint (best step recorded above).',
                'report the best-validation metric, not the final-epoch '
                'metric; the final weights are past the turning point.',
            ]
        elif train_improved and val_improved:
            report['verdict'] = 'healthy'
            report['recommendations'] = [
                'both curves fell: the recipe is learning and generalizing; '
                'continue and use the validation minimum for model '
                'selection.',
            ]
        elif train_improved:
            report['verdict'] = 'overfitting'
            report['recommendations'] = [
                'train improved but validation flattened: the model is '
                'using capacity that does not transfer. Prefer the '
                'validation-best checkpoint and consider more data or '
                'stronger regularization before scaling the model.',
            ]
        else:
            report['verdict'] = 'underfitting'
            report['recommendations'] = [
                'neither curve improved: treat as underfitting / broken '
                'optimization before adding capacity. Check the learning '
                'rate (try a sweep over 3-10x), the optimizer wiring, and '
                'the data pipeline itself (shuffling, label alignment, '
                'task difficulty vs chance level).',
                'run the fixed-batch overfit sanity gate: if a fresh model '
                'cannot memorize one batch, the bug is in the pipeline, '
                'not in the model.',
            ]
    else:
        if train_improved:
            report['verdict'] = 'train_improving_only'
            report['recommendations'] = [
                'training loss fell but no validation curve was recorded; '
                'generalization is unverified (the loss may be falling on '
                'memorized data). Enable periodic held-out validation and '
                'select the model on its minimum.',
            ]
        else:
            report['verdict'] = 'underfitting'
            report['recommendations'] = [
                'training loss never improved: suspect the learning rate, '
                'optimizer wiring, or the data pipeline itself before '
                'changing the model.',
            ]
    report['signals'] = signals
    return report


def _finiteness_detail(t: torch.Tensor) -> dict:
    td = t.detach()
    detail = {
        'shape': list(td.shape),
        'dtype': str(td.dtype),
        'finite': True,
        'n_nan': 0,
        'n_inf': 0,
        'absmax': None,
    }
    if td.is_floating_point() or td.is_complex():
        finite = torch.isfinite(td)
        detail['finite'] = bool(finite.all())
        if not detail['finite']:
            detail['n_nan'] = int(torch.isnan(td).sum())
            detail['n_inf'] = int(torch.isinf(td).sum())
        with torch.no_grad():
            detail['absmax'] = float(td.abs().max()) if td.numel() else 0.0
    return detail


def _tensor_map_finiteness(tensors: Mapping[str, torch.Tensor] | None) -> dict:
    out = {}
    if not tensors:
        return out
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            out[name] = _finiteness_detail(t)
    return out


def diagnose_nonfinite(*, loss: torch.Tensor | float | None = None,
                       inputs: Mapping[str, torch.Tensor] | None = None,
                       token_ids: Mapping[str, torch.Tensor] | None = None,
                       id_range: tuple[int, int] | None = None,
                       params: Mapping[str, torch.Tensor]
                       | Iterable[torch.Tensor] | None = None,
                       logits: torch.Tensor | None = None,
                       grads: Mapping[str, torch.Tensor]
                       | Iterable[torch.Tensor] | None = None,
                       grad_norm: float | None = None,
                       amp_enabled: bool = False,
                       amp_scaler=None,
                       fp32_forward_loss: torch.Tensor | float | None = None,
                       logits_absmax_warn: float = 1e4,
                       context: str = '') -> dict:
    """Systematic NaN/Inf triage in the documented order of suspicion.

    Checks, in order (the FIRST failing stage is the verdict):

    1. **data** — ``inputs`` (activations / embeddings) contain NaN/Inf;
    2. **token ids** — ids outside ``id_range`` (embedding gather holes);
    3. **loss** — the scalar itself is non-finite;
    4. **parameters** — a weight/bias already non-finite (corruption from
       an earlier step);
    5. **logits** — non-finite or implausibly large (manual softmax
       overflow; ``F.cross_entropy`` shifts by the max and would not
       overflow by itself);
    6. **amp** — fp16/bf16 overflow: when ``fp32_forward_loss`` is finite
       but the autocast ``loss`` is not, the overflow is in mixed
       precision, not in the math; also reports a collapsing GradScaler
       (repeatedly halved => many skipped steps);
    7. **gradients** — non-finite grads, or a huge ``grad_norm``
       (explosion / learning rate too large).

    ``params`` / ``grads`` may be a ``{name: tensor}`` mapping (better
    messages) or any iterable of tensors. The verdict is the first stage
    that can EXPLAIN a non-finite value (the loss itself is a symptom, so
    a non-finite loss with every other stage clean falls back to
    ``loss_nonfinite``). Returns a JSON-serializable report with per-stage
    status, the verdict, and concrete remediation steps.
    """
    def _as_map(obj):
        if obj is None:
            return {}
        if isinstance(obj, Mapping):
            return dict(obj)
        return {f'param[{i}]': p for i, p in enumerate(obj)}

    checks: list[dict] = []
    verdict = None
    summary = 'no non-finite signal found in the supplied tensors'

    def record(name, status, detail):
        checks.append({'check': name, 'status': status, 'detail': detail})

    data_map = _as_map(inputs)
    data_info = _tensor_map_finiteness(data_map)
    bad_data = [k for k, v in data_info.items() if not v['finite']]
    if bad_data:
        verdict = 'data_nonfinite'
        summary = (f'input data contains NaN/Inf in {bad_data}: fix the '
                   f'data pipeline (clean/validate inputs) before blaming '
                   f'the model')
        record('data', 'fail', {'non_finite': bad_data, 'tensors': data_info})
    else:
        record('data', 'pass' if data_info else 'skipped', data_info)

    if token_ids and id_range is not None:
        lo, hi = int(id_range[0]), int(id_range[1])
        bad_ids = {}
        for name, ids in token_ids.items():
            if not isinstance(ids, torch.Tensor) or not ids.numel():
                continue
            imin, imax = int(ids.min()), int(ids.max())
            if imin < lo or imax >= hi:
                bad_ids[name] = {'min': imin, 'max': imax,
                                 'allowed': [lo, hi - 1]}
        if bad_ids:
            if verdict is None:
                verdict = 'token_ids_out_of_range'
                summary = (f'token ids outside [{lo}, {hi}) in {list(bad_ids)}: '
                           f'an embedding gather would produce undefined/garbage '
                           f'rows; fix the data/tokenizer, not the model')
            record('token_ids', 'fail', bad_ids)
        else:
            record('token_ids', 'pass',
                   {name: {'min': int(ids.min()), 'max': int(ids.max())}
                    for name, ids in token_ids.items()
                    if isinstance(ids, torch.Tensor) and ids.numel()})
    else:
        record('token_ids', 'skipped', 'no token ids / range supplied')

    loss_val = None
    if loss is not None:
        loss_val = float(loss) if not isinstance(loss, torch.Tensor) else float(loss.detach())
        loss_finite = math.isfinite(loss_val)
        # The loss is the SYMPTOM, not a stage: record it in order but let
        # the more specific stages below (parameters / logits / AMP /
        # gradients) claim the verdict when they fail.
        record('loss', 'pass' if loss_finite else 'fail',
               {'value': loss_val})
    else:
        record('loss', 'skipped', 'no loss supplied')

    params_map = _as_map(params)
    param_info = _tensor_map_finiteness(params_map)
    bad_params = [k for k, v in param_info.items() if not v['finite']]
    if params_map:
        if bad_params:
            if verdict is None:
                verdict = 'parameters_nonfinite'
                summary = (f'{len(bad_params)} parameter tensor(s) already '
                           f'non-finite (first: {bad_params[0]}): an earlier '
                           f'step corrupted the weights; check the first step '
                           f'where the loss/grad became non-finite, not the '
                           f'current one')
            record('parameters', 'fail',
                   {'non_finite': bad_params[:10], 'n_total': len(params_map)})
        else:
            record('parameters', 'pass',
                   {'n_tensors': len(params_map),
                    'absmax': max((v['absmax'] or 0.0)
                                  for v in param_info.values())})
    else:
        record('parameters', 'skipped', 'no parameters supplied')

    if logits is not None and isinstance(logits, torch.Tensor):
        info = _finiteness_detail(logits)
        if not info['finite']:
            if verdict is None:
                verdict = 'logits_nonfinite'
                summary = ('logits contain NaN/Inf: if the loss is a manual '
                           'softmax/log-softmax, check overflow and use the '
                           'max-shift stable form (F.cross_entropy)')
            record('logits', 'fail', info)
        elif info['absmax'] is not None and info['absmax'] > logits_absmax_warn:
            record('logits', 'warn',
                   {'absmax': info['absmax'],
                    'threshold': logits_absmax_warn,
                    'note': 'large logits saturate softmax; fp16 would '
                            'overflow near 65504'})
        else:
            record('logits', 'pass', info)
    else:
        record('logits', 'skipped', 'no logits supplied')

    amp_detail: dict = {'amp_enabled': bool(amp_enabled)}
    if amp_scaler is not None:
        try:
            amp_detail['scale'] = float(amp_scaler.get_scale())
        except Exception:
            pass
    fp32_finite = None
    if fp32_forward_loss is not None:
        fp32_val = (float(fp32_forward_loss)
                    if not isinstance(fp32_forward_loss, torch.Tensor)
                    else float(fp32_forward_loss.detach()))
        fp32_finite = math.isfinite(fp32_val)
        amp_detail['fp32_forward_loss'] = fp32_val
    if amp_enabled:
        if (loss_val is not None and not math.isfinite(loss_val)
                and fp32_finite):
            if verdict is None:
                verdict = 'amp_overflow'
                summary = ('the autocast (fp16/bf16) loss is non-finite but '
                           'the same batch is finite in fp32: mixed-precision '
                           'overflow. Keep the loss/softmax in fp32, check '
                           'GradScaler, or lower the LR')
            record('amp', 'fail', amp_detail)
        elif (amp_detail.get('scale') is not None
                and amp_detail['scale'] < 16.0):
            record('amp', 'warn', amp_detail)
        else:
            record('amp', 'pass', amp_detail)
    else:
        record('amp', 'skipped', amp_detail)

    grads_map = _as_map(grads)
    grad_info = _tensor_map_finiteness(grads_map)
    bad_grads = [k for k, v in grad_info.items() if not v['finite']]
    if grads_map or grad_norm is not None:
        detail = {'non_finite': bad_grads[:10], 'grad_norm': grad_norm}
        if bad_grads:
            if verdict is None:
                verdict = 'gradients_nonfinite'
                summary = (f'{len(bad_grads)} gradient tensor(s) non-finite '
                           f'(first: {bad_grads[0]}): add clip_grad_norm_, '
                           f'check for log(0)/division by zero in the loss, '
                           f'and lower the LR')
            record('gradients', 'fail', detail)
        elif grad_norm is not None and not math.isfinite(float(grad_norm)):
            if verdict is None:
                verdict = 'gradients_nonfinite'
                summary = 'total gradient norm is non-finite (explosion)'
            record('gradients', 'fail', detail)
        elif grad_norm is not None and float(grad_norm) > 1e3:
            record('gradients', 'warn',
                   {**detail, 'note': 'gradients are exploding; clip and/or '
                                      'lower the learning rate'})
        else:
            record('gradients', 'pass', detail)
    else:
        record('gradients', 'skipped', 'no gradients / grad norm supplied')

    if (verdict is None and loss_val is not None
            and not math.isfinite(loss_val)):
        verdict = 'loss_nonfinite'
        summary = (f'the loss is {loss_val} while every supplied stage was '
                   f'finite: trace the forward pass for log(0) / division '
                   f'by zero / unsafe softmax (supply logits and grads for '
                   f'a narrower verdict)')

    return {
        'context': context,
        'verdict': verdict or 'no_nonfinite_signal',
        'summary': summary,
        'first_failing_stage': verdict,
        'checks': checks,
    }


def pareto_frontier(rows: Sequence[Mapping],
                    *, accuracy_key: str = 'mean_acc',
                    cost_keys: Sequence[str] = ('mean_train_time_s',
                                                'n_params'),
                    name_key: str = 'op') -> dict:
    """Dominance annotation over (accuracy maximise, costs minimise).

    A row A dominates a row B iff A's accuracy is >= B's and every cost is
    <= B's, with at least one strict inequality. Rows with missing /
    non-finite accuracy or costs are excluded (listed under ``skipped``).
    Ties do not dominate.

    Returns ``{'frontier': [names], 'dominated_by': {name: dominator},
    'points': {...}, 'skipped': [...]}``.
    """
    points: dict = {}
    skipped: list = []
    for r in rows:
        name = r.get(name_key, r.get('op', '?'))
        acc = r.get(accuracy_key)
        if acc is None or not math.isfinite(float(acc)):
            skipped.append(name)
            continue
        costs = {}
        ok = True
        for k in cost_keys:
            v = r.get(k)
            if v is None or not math.isfinite(float(v)):
                ok = False
                break
            costs[k] = float(v)
        if not ok:
            skipped.append(name)
            continue
        eval_ms = r.get('mean_eval_ms_per_sample')
        if eval_ms is not None and (not math.isfinite(float(eval_ms))
                                    or float(eval_ms) <= 0):
            eval_ms = None
        points[name] = {
            'accuracy': float(acc),
            'costs': costs,
            'acc_per_train_second': (
                float(acc) / costs['mean_train_time_s']
                if costs.get('mean_train_time_s', 0.0) > 0 else None),
            'acc_per_1k_params': (
                float(acc) / (costs['n_params'] / 1000.0)
                if costs.get('n_params', 0.0) > 0 else None),
            'acc_per_eval_ms': (
                float(acc) / float(eval_ms) if eval_ms else None),
        }

    names = list(points)
    dominated_by: dict = {}
    for b in names:
        dominators = []
        for a in names:
            if a == b:
                continue
            acc_ge = points[a]['accuracy'] >= points[b]['accuracy']
            costs_le = all(points[a]['costs'][k] <= points[b]['costs'][k]
                           for k in cost_keys)
            strict = (points[a]['accuracy'] > points[b]['accuracy']
                      or any(points[a]['costs'][k] < points[b]['costs'][k]
                             for k in cost_keys))
            if acc_ge and costs_le and strict:
                dominators.append(a)
        if dominators:
            dominators.sort(key=lambda a: (-points[a]['accuracy'],
                                           sum(points[a]['costs'].values())))
            dominated_by[b] = dominators[0]
    frontier = [n for n in names if n not in dominated_by]
    return {
        'accuracy_key': accuracy_key,
        'cost_keys': list(cost_keys),
        'frontier': frontier,
        'dominated_by': dominated_by,
        'points': points,
        'skipped': skipped,
    }


def annotate_rows_with_pareto(rows: Sequence[dict], **kwargs) -> dict:
    """Run :func:`pareto_frontier` and attach per-row fields in place.

    Each row gets ``pareto`` = ``{'optimal': bool, 'dominated_by': name or
    None, 'acc_per_train_second': ..., 'acc_per_1k_params': ...}``.
    Returns the frontier report.
    """
    report = pareto_frontier(rows, **kwargs)
    for r in rows:
        name = r.get(kwargs.get('name_key', 'op'))
        p = report['points'].get(name)
        if p is None:
            r['pareto'] = None
            continue
        r['pareto'] = {
            'optimal': name in report['frontier'],
            'dominated_by': report['dominated_by'].get(name),
            'acc_per_train_second': p['acc_per_train_second'],
            'acc_per_1k_params': p['acc_per_1k_params'],
            'acc_per_eval_ms': p.get('acc_per_eval_ms'),
        }
    return report


# Error categories for token-level LM bad-case analysis. A natural-language
# model has no exact labels to check against (unlike the synthetic probes),
# so a wrong next-token prediction is categorized by WHAT INFORMATION the
# model failed to use, not by which task layout the sample came from:
#
#   * ``near_miss``        — the target was in the top-k but top-1 disagreed:
#                            a ranking/calibration miss, the cheapest to fix
#                            and the most likely to be sample noise;
#   * ``confident_wrong``  — the model strongly preferred a wrong token:
#                            a systematic bias / blind spot, not ambiguity;
#   * ``copyable_missed``  — the target token DID occur earlier in the
#                            context but ranked outside the top-k: a
#                            retrieval/copy failure (the actionable class for
#                            a recurrent / compressed-attention hybrid, whose
#                            whole selling point is long-range memory);
#   * ``novel_token``      — the target never occurred in the context and
#                            the model was not confident: an intrinsic
#                            prediction error (ambiguity / capacity).
#
# ``correct`` and the masked ``ignored`` positions complete the partition.
LM_ERROR_CATEGORIES = OrderedDict([
    ('correct', 'top-1 prediction equals the target'),
    ('near_miss', 'target ranked 2..top_k: ranking/calibration miss'),
    ('confident_wrong', 'wrong top-1 with top-1 probability >= confidence '
                        'threshold: systematic bias / blind spot'),
    ('copyable_missed', 'target occurred earlier in the context but ranked '
                        'outside the top-k and was predicted without '
                        'confidence: retrieval/copy failure'),
    ('novel_token', 'target never occurred in the context and was predicted '
                    'without confidence: intrinsic prediction difficulty'),
    ('ignored', 'position masked out (ignore_index label)'),
])

_LM_ERROR_CATEGORY_ORDER = tuple(LM_ERROR_CATEGORIES)


def lm_token_diagnostics(logits: torch.Tensor, labels: torch.Tensor,
                         input_ids: torch.Tensor, *, top_k: int = 5,
                         confidence: float = 0.5,
                         ignore_index: int = -100) -> dict:
    """Per-token bad-case diagnostics for next-token prediction.

    Pure tensor math (no RNG, no global state). ``logits`` is
    ``(B, T, V)``, ``labels`` / ``input_ids`` are ``(B, T)``. For every
    position the function computes:

      * ``category`` — one of :data:`LM_ERROR_CATEGORIES` (see there for the
        definition of each class; the model's own output decides, so this
        works for any tokenizer and any corpus);
      * ``target_rank`` — 1-based rank of the target among the logits
        (rank 1 = the target is (one of) the argmax token(s));
      * ``top1_prob`` — softmax probability of the predicted token, computed
        with the max-shift (logsumexp) stable form;
      * ``seen_earlier`` / ``last_seen_distance`` — whether (and how long
        ago) the target token occurred earlier in the same input. Distance
        buckets turn "the model can't copy" into "the model can copy at
        distance d but not at distance d'", which is exactly the evidence a
        long-context architecture change needs.

    Tensors are returned with the same ``(B, T)`` shape as the inputs; the
    category is also returned as JSON-ready nested lists.
    """
    if logits.ndim != 3:
        raise ValueError(f'logits must be (B, T, V), got shape '
                         f'{tuple(logits.shape)}')
    if labels.shape != logits.shape[:2] or input_ids.shape != labels.shape:
        raise ValueError(
            f'labels/input_ids must be {tuple(logits.shape[:2])}, got '
            f'{tuple(labels.shape)} / {tuple(input_ids.shape)}')
    b, t, v = logits.shape
    if t == 0:
        empty = torch.zeros(b, 0, dtype=torch.long, device=logits.device)
        return {
            'valid': torch.zeros(b, 0, dtype=torch.bool,
                                 device=logits.device),
            'correct': torch.zeros(b, 0, dtype=torch.bool,
                                   device=logits.device),
            'pred': empty,
            'target_rank': empty,
            'top1_prob': torch.zeros(b, 0, device=logits.device),
            'seen_earlier': torch.zeros(b, 0, dtype=torch.bool,
                                        device=logits.device),
            'last_seen_distance': torch.full((b, 0), -1, dtype=torch.long,
                                             device=logits.device),
            'position': empty,
            'category': [[] for _ in range(b)],
            'top_k': int(top_k),
            'confidence': float(confidence),
        }

    valid = labels != int(ignore_index)
    pred = logits.argmax(dim=-1)
    correct = valid & (pred == labels)

    safe_labels = labels.clamp(min=0, max=v - 1).unsqueeze(-1)
    target_logit = logits.gather(-1, safe_labels).squeeze(-1)
    target_rank = (logits > target_logit.unsqueeze(-1)).sum(dim=-1) + 1
    target_rank = torch.where(valid, target_rank, torch.zeros_like(target_rank))

    # Stable top-1 probability: max-shift by the row max via logsumexp, so a
    # manual softmax overflow can never fabricate a "confident" verdict.
    log_norm = torch.logsumexp(logits, dim=-1)
    top1_prob = torch.exp(logits.max(dim=-1).values - log_norm)

    # seen[b, t, s] = labels[b, t] == input_ids[b, s] (s < t): the target
    # token occurred earlier in the same sample. B*T*T bool is small for the
    # analysis batch sizes (a 1k-token sequence is ~1M entries).
    seen = input_ids.unsqueeze(1) == labels.unsqueeze(2)
    lower = torch.ones(t, t, dtype=torch.bool, device=logits.device)
    lower = lower.tril(diagonal=-1)
    seen = seen & lower.unsqueeze(0)
    seen_earlier = seen.any(dim=-1) & valid
    idx = torch.arange(t, device=logits.device)
    no_pos = torch.full(seen.shape, -1, dtype=torch.long,
                        device=logits.device)
    last_pos = torch.where(seen, idx.view(1, 1, t).expand_as(seen),
                           no_pos).max(dim=-1).values
    last_seen_distance = torch.where(
        last_pos >= 0, idx.view(1, t) - last_pos,
        torch.full_like(last_pos, -1))

    k = max(1, int(top_k))
    conf = float(confidence)
    cat_idx = torch.zeros(b, t, dtype=torch.long, device=logits.device)
    cat_idx = torch.where(~valid, torch.full_like(cat_idx, 5), cat_idx)
    wrong = valid & ~correct
    cat_idx = torch.where(wrong & (target_rank <= k),
                          torch.full_like(cat_idx, 1), cat_idx)
    cat_idx = torch.where(wrong & (target_rank > k) & (top1_prob >= conf),
                          torch.full_like(cat_idx, 2), cat_idx)
    cat_idx = torch.where(wrong & (target_rank > k) & (top1_prob < conf)
                          & seen_earlier,
                          torch.full_like(cat_idx, 3), cat_idx)
    cat_idx = torch.where(wrong & (target_rank > k) & (top1_prob < conf)
                          & ~seen_earlier,
                          torch.full_like(cat_idx, 4), cat_idx)
    names = _LM_ERROR_CATEGORY_ORDER
    category = [[names[i] for i in row] for row in cat_idx.tolist()]

    return {
        'valid': valid,
        'correct': correct,
        'pred': pred,
        'target_rank': target_rank,
        'top1_prob': top1_prob,
        'seen_earlier': seen_earlier,
        'last_seen_distance': last_seen_distance,
        'position': idx.view(1, t).expand(b, t).clone(),
        'category': category,
        'top_k': k,
        'confidence': conf,
    }


def summarize_lm_diagnostics(diags: Mapping | Sequence) -> dict:
    """Aggregate token diagnostics into a JSON-safe bad-case report.

    Accepts one diagnostics dict or a sequence of them (e.g. one per
    analysis batch); computes the token accuracy, the category mix, the
    target-rank / confidence profile of the errors, and accuracy sliced by
    (a) relative position in the sequence and (b) distance to the target
    token's last earlier occurrence — the two axes that turn "the model is
    wrong 30% of the time" into "it is wrong at long copy distances / early
    positions".
    """
    items = ([diags] if isinstance(diags, Mapping)
             else list(diags))
    if not items:
        return {'n_tokens': 0, 'n_wrong': 0, 'token_accuracy': None,
                'categories': {}, 'accuracy_by_position_quartile': [],
                'accuracy_by_repeat_distance': [],
                'note': 'no diagnostics supplied'}

    valid = torch.cat([d['valid'].reshape(-1) for d in items])
    correct = torch.cat([d['correct'].reshape(-1) for d in items])
    target_rank = torch.cat([d['target_rank'].reshape(-1) for d in items])
    top1_prob = torch.cat([d['top1_prob'].reshape(-1) for d in items])
    seen_earlier = torch.cat([d['seen_earlier'].reshape(-1) for d in items])
    distance = torch.cat([d['last_seen_distance'].reshape(-1) for d in items])
    position = torch.cat([d['position'].reshape(-1) for d in items])
    categories = [c for d in items for row in d['category'] for c in row]

    n_tokens = int(valid.sum())
    n_wrong = int((valid & ~correct).sum())
    counts: dict = OrderedDict((name, 0) for name in LM_ERROR_CATEGORIES)
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    # Every position (valid or ignored) receives exactly one category, so
    # the category counts sum to the TOTAL number of positions. Using
    # ``n_tokens`` (valid only) understated the denominator for categories
    # that can include masked positions — ``ignored`` then reported a
    # "rate of all" above 1.0 whenever padding outnumbered real tokens.
    n_all = sum(counts.values())
    cat_report = OrderedDict()
    for name, definition in LM_ERROR_CATEGORIES.items():
        c = counts.get(name, 0)
        cat_report[name] = {
            'count': int(c),
            'rate_of_all': (c / n_all) if n_all else 0.0,
            'rate_of_wrong': (c / n_wrong) if n_wrong else 0.0,
            'definition': definition,
        }

    err = valid & ~correct
    rank_on_err = target_rank[err].float()
    prob_err = top1_prob[err].float()
    prob_ok = top1_prob[valid & correct].float()

    def _acc(mask: torch.Tensor) -> dict:
        n = int(mask.sum())
        return {
            'n': n,
            'accuracy': (float(correct[mask].float().mean()) if n else None),
        }

    # (a) Relative position quartiles: a recurrent state early in the
    # sequence has seen less context, so this separates cold-start errors
    # from steady-state errors.
    pos = position[valid]
    width = int(pos.max().item()) + 1 if pos.numel() else 1
    pos_buckets = []
    if pos.numel():
        for qi in range(4):
            mask = valid & ((position * 4) // max(1, width) == qi)
            entry = _acc(mask)
            entry['quartile'] = f'q{qi + 1}_of_4'
            # Half-open interval of the SAME mask condition
            # ``(position*4)//width == qi``  <=>
            # ceil(qi*width/4) <= position < ceil((qi+1)*width/4).
            entry['position_range'] = [
                -(-qi * width // 4), -(-(qi + 1) * width // 4)]
            pos_buckets.append(entry)

    # (b) Distance to the target's last earlier occurrence: copy accuracy is
    # the retrieval curve of a memory architecture.
    dist_steps = [(1, 8, 'd01_08'), (9, 32, 'd09_32'),
                  (33, 128, 'd33_128'), (129, None, 'd129_plus')]
    dist_buckets = []
    for lo, hi, label in dist_steps:
        mask = valid & seen_earlier & (distance >= lo)
        if hi is not None:
            mask = mask & (distance <= hi)
        entry = _acc(mask)
        entry['bucket'] = label
        entry['distance_range'] = [lo, hi]
        dist_buckets.append(entry)
    entry = _acc(valid & ~seen_earlier)
    entry['bucket'] = 'not_in_context'
    entry['distance_range'] = None
    dist_buckets.append(entry)

    return {
        'n_tokens': n_tokens,
        'n_wrong': n_wrong,
        'token_accuracy': (float(correct[valid].float().mean())
                           if n_tokens else None),
        'categories': cat_report,
        'target_rank_on_errors': {
            'mean': float(rank_on_err.mean()) if rank_on_err.numel() else None,
            # ``statistics.median`` (not ``torch.median``, which returns the
            # LOWER of the two middle values for even-sized samples): error
            # counts are typically even, and the lower-middle convention
            # biases the reported headline rank downward.
            'median': (float(statistics.median(rank_on_err.tolist()))
                       if rank_on_err.numel() else None),
        },
        'top1_prob_on_errors': {
            'mean': float(prob_err.mean()) if prob_err.numel() else None,
        },
        'top1_prob_on_correct': {
            'mean': float(prob_ok.mean()) if prob_ok.numel() else None,
        },
        'accuracy_by_position_quartile': pos_buckets,
        'accuracy_by_repeat_distance': dist_buckets,
        'note': ('Token-level bad cases on a held-out split: category mix '
                 'separates near-misses / confident blind spots / copy '
                 'failures / novel tokens, and the repeat-distance curve '
                 'localizes long-range retrieval failures.'),
    }
