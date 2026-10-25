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
4. :func:`multiclass_classification_metrics` — confusion-matrix metrics
   with the averaging convention EXPLICIT in every key name (macro /
   micro / weighted), plus balanced accuracy and per-class support, so a
   multi-class result can never be read as "some F1" without knowing
   which F1 it is (the three differ substantially under imbalance).
5. :func:`binary_ranking_metrics` — threshold-free ranking metrics
   (PR-AUC / average precision, ROC-AUC with tie correction, MRR,
   recall@1) for binary retrieval-style decisions whose positives are
   rare. PR-AUC is the primary metric in that regime: with a huge
   negative base, FPR grows slowly and ROC-AUC stays optimistic.
6. :func:`confidence_calibration` — whether a reported confidence score
   means what it says: reliability bins, equal-width AND equal-mass
   ECE/MCE, Brier score/skill, over/under-confidence gap, and a
   discrimination block (ROC-AUC). A model can rank correct above wrong
   (high AUC) while being badly overconfident, and the token-level
   ``confident_wrong`` category in this module *assumes* the raw top-1
   probability is calibrated; this report tests that assumption instead
   of leaving it implicit.

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
    for v in values:
        if v is None:
            return False
        try:
            if not math.isfinite(float(v)):
                return False
        except (TypeError, ValueError, OverflowError):
            # JSON has unbounded integers; a value too large for float is
            # malformed for this numeric report, not a reason to crash.
            return False
    return True


def _safe_mean(values: Sequence[float]) -> float | None:
    """Mean of finite values without overflowing the intermediate sum.

    ``sum([1e308, 1e308]) / 2`` is ``inf / 2`` even though both inputs and
    the true mean are finite.  The ordinary path keeps the exact previous
    arithmetic; only an overflowing sum falls back to a scaled sum, which
    is value-equal up to ordinary floating-point round-off.
    """
    if not values:
        return None
    total = sum(values)
    if math.isfinite(total):
        return float(total / len(values))
    scale = max(abs(float(v)) for v in values)
    if scale == 0.0:
        return 0.0
    mean = float(scale * (math.fsum(float(v) / scale for v in values)
                          / len(values)))
    if not math.isfinite(mean):
        raise ValueError(
            f'mean is not finite after scaled summation (scale={scale!r})')
    return mean


def _window_mean(values: Sequence[float], k: int) -> float | None:
    if not values:
        return None
    k = max(1, min(int(k), len(values)))
    return _safe_mean(values[-k:])


def _head_mean(values: Sequence[float], frac: float = 0.1) -> float | None:
    if not values:
        return None
    k = max(1, int(math.ceil(len(values) * frac)))
    return _safe_mean(values[:k])


def _tail_slope(losses: Sequence[float], *, frac: float = 0.25
                ) -> tuple[float | None, float | None, int]:
    """Least-squares slope over the last ``frac`` of a curve.

    A head/tail MEAN comparison cannot tell a curve that reached a plateau
    from one that is still falling when the budget runs out: both can have
    the same tail mean. The slope of the final segment answers that
    question. Returns ``(slope_per_point, slope_relative_to_tail_mean,
    n_points_used)``; the relative value is ``None`` when the tail mean is
    zero, so it can be stored in strict JSON.  ``slope_per_point`` is also
    ``None`` when the true slope is not representable as a finite float
    (e.g. an opposite-sign swing of two ``1e308`` losses); returning
    ``inf``/``NaN`` would break the strict-JSON contract.
    """
    n = len(losses)
    if n < 2:
        return 0.0, None, n
    k = max(2, int(math.ceil(n * float(frac))))
    k = min(k, n)
    ys = [float(v) for v in losses[-k:]]
    xs = list(range(k))
    mean_x = (k - 1) / 2.0
    mean_y = _safe_mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = None
    if denom:
        deviations = [y - mean_y for y in ys]
        try:
            num = math.fsum((x - mean_x) * d
                            for x, d in zip(xs, deviations))
            candidate = num / denom
        except OverflowError:
            # ``fsum`` raises when an intermediate partial sum overflows,
            # even if the true slope is finite. Treat it like an
            # unrepresentable candidate and retry in scaled units below.
            candidate = math.inf
        if math.isfinite(candidate):
            slope = candidate
        else:
            # The true slope is finite in units of the curve but can exceed
            # the float range.  Recompute in scaled units and only multiply
            # back when the product is representable.
            scale = max(abs(y) for y in ys)
            if scale > 0.0:
                scaled = [y / scale for y in ys]
                scaled_mean = _safe_mean(scaled)
                scaled_num = math.fsum(
                    (x - mean_x) * (y - scaled_mean)
                    for x, y in zip(xs, scaled))
                scaled_slope = scaled_num / denom
                candidate = scaled_slope * scale
                if math.isfinite(candidate):
                    slope = candidate
    else:
        slope = 0.0
    relative = (slope / abs(mean_y)) if (
        slope is not None and mean_y not in (None, 0.0)) else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    return slope, relative, k


def _curve_stats(losses: Sequence[float]) -> dict:
    """Head/tail/best/tail-trend summary of one loss curve."""
    n = len(losses)
    head_k = max(1, int(math.ceil(n * 0.1)))
    tail_k = max(1, int(math.ceil(n * 0.1)))
    head = _safe_mean(losses[:head_k])
    tail = _safe_mean(losses[-tail_k:])
    best = float(min(losses))
    best_step = int(min(range(n), key=lambda i: losses[i]))
    relative_change = None
    if head not in (None, 0.0):
        delta = tail - head
        if math.isfinite(delta):
            relative_change = delta / abs(head)
        else:
            # Same ratio, evaluated without the overflowing subtraction.
            relative_change = (tail / abs(head)) - math.copysign(1.0, head)
        if not math.isfinite(relative_change):
            relative_change = None
    tail_slope, tail_slope_rel, tail_slope_points = _tail_slope(losses)
    return {
        'n_points': n,
        'first': float(losses[0]),
        'last': float(losses[-1]),
        'head_mean': head,
        'tail_mean': tail,
        'best': best,
        'best_step': best_step,
        'relative_change': relative_change,
        # Negative = still falling at the final checkpoint. Compare the
        # RELATIVE slope across tasks/architectures; the absolute one is in
        # the row's loss units.
        'tail_slope_per_point': tail_slope,
        'tail_slope_relative_per_point': tail_slope_rel,
        'tail_slope_points': int(tail_slope_points),
    }


#: A relative tail slope more negative than this (per point) means the curve
#: has not flattened at the final checkpoint. Being within
#: ``TAIL_BEST_TOLERANCE`` of its own minimum rules out a late rebound.
TAIL_STILL_IMPROVING_SLOPE = 1e-3
TAIL_BEST_TOLERANCE = 0.02


def _still_improving_at_end(stats: Mapping) -> bool:
    """Whether a loss curve was still falling when the budget ran out.

    Uses the final-segment slope, not the head/tail means: a curve that
    dropped early and then plateaued and one that dropped early and is
    still dropping have the same tail mean but opposite next-step
    prospects. The endpoint must also be near its own minimum, so a late
    rebound is not mislabelled as "still improving".
    """
    slope = stats.get('tail_slope_relative_per_point')
    if slope is None:
        return False
    best = float(stats.get('best', stats.get('last', 0.0)))
    last = float(stats.get('last', best))
    tolerance = TAIL_BEST_TOLERANCE * max(1.0, abs(best))
    return bool(slope < -TAIL_STILL_IMPROVING_SLOPE
                and last <= best + tolerance)


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
    # Preserve ``None`` entries (JSON ``null``, e.g. a non-finite loss that
    # was sanitized before being persisted) so the non-finite branch below
    # can report *where* the curve broke instead of crashing on float(None).
    # Oversized integers are also mapped to None: JSON permits them, but
    # they are not representable here and used to raise OverflowError before
    # the non-finite verdict could be formed.
    train = [_finite_float_or_none(v) for v in train_losses]
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
        val = [_finite_float_or_none(v) for v in val_losses]
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
    train_still_improving = _still_improving_at_end(tr)
    signals = {
        'train_improved': bool(train_improved),
        'train_head_mean': train_head,
        'train_tail_mean': train_tail,
        'train_relative_change': tr['relative_change'],
        'train_tail_slope_relative_per_point': tr[
            'tail_slope_relative_per_point'],
        'train_still_improving_at_end': bool(train_still_improving),
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
        # The difference of two finite means can overflow; a gap that the
        # float range cannot represent is reported as unavailable, never
        # as +-inf in the JSON record.
        head_gap = head_gap if math.isfinite(head_gap) else None
        tail_gap = tail_gap if math.isfinite(tail_gap) else None
        val_still_improving = _still_improving_at_end(va)
        signals.update({
            'val_improved': bool(val_improved),
            'val_head_mean': val_head,
            'val_tail_mean': val_tail,
            'val_best': val_best,
            'val_best_step': va['best_step'],
            'val_rose_from_best': bool(val_rose_from_best),
            'val_tail_slope_relative_per_point': va[
                'tail_slope_relative_per_point'],
            'val_still_improving_at_end': bool(val_still_improving),
            'generalization_gap_head': head_gap,
            'generalization_gap_tail': tail_gap,
        })
        gap_grew = (head_gap is not None and tail_gap is not None
                    and tail_gap > max(0.0, head_gap) * 2.0 + 0.1
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
    # Budget status is a separate axis from over/underfitting: a healthy
    # run can still be mid-descent when the fixed step budget expires, and
    # then the final number is not a converged property of the recipe.
    # Requiring BOTH curves to still fall keeps a noisy train curve from
    # triggering the label when validation has already flattened.
    budget_limited = bool(val is not None and len(val) >= 2
                          and train_still_improving
                          and signals.get('val_still_improving_at_end'))
    signals['budget_limited'] = budget_limited
    if budget_limited:
        report['recommendations'].append(
            'both training and validation loss were still falling at the '
            'final checkpoint (relative tail slopes '
            f"{signals.get('train_tail_slope_relative_per_point'):+.4g} / "
            f"{signals.get('val_tail_slope_relative_per_point'):+.4g} per "
            'point): the fixed-budget endpoint is not converged, so do not '
            'rank configurations on it alone. Extend the budget (or lower '
            'the validation interval) until the tail slope flattens, then '
            're-compare.')
    elif train_still_improving and not val:
        report['recommendations'].append(
            'the training curve was still falling at the final checkpoint; '
            'the run may be budget-limited. Add held-out validation to tell '
            'a genuine descent from memorization before extending it.')
    report['signals'] = signals
    report['budget_limited'] = budget_limited
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
    loss_malformed = False
    if loss is not None:
        try:
            loss_val = (float(loss) if not isinstance(loss, torch.Tensor)
                        else float(loss.detach()))
        except (TypeError, ValueError, OverflowError):
            # A non-numeric or out-of-float-range loss is malformed; keep
            # the report JSON-safe instead of leaking OverflowError.
            loss_val = None
            loss_malformed = True
        loss_finite = loss_val is not None and math.isfinite(loss_val)
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
        elif param_info:
            record('parameters', 'pass',
                   {'n_tensors': len(param_info),
                    'absmax': max((v['absmax'] or 0.0)
                                  for v in param_info.values())})
        else:
            # A non-empty map of non-tensors (e.g. ``{'w': None}``) has no
            # parameter to triage. Report it as skipped instead of calling
            # ``max()`` on an empty sequence.
            record('parameters', 'skipped',
                   'parameter map contained no tensor values')
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
        try:
            fp32_val = (float(fp32_forward_loss)
                        if not isinstance(fp32_forward_loss, torch.Tensor)
                        else float(fp32_forward_loss.detach()))
        except (TypeError, ValueError, OverflowError):
            fp32_val = None
        fp32_finite = fp32_val is not None and math.isfinite(fp32_val)
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
        try:
            grad_norm_val = (None if grad_norm is None
                             else float(grad_norm))
        except (TypeError, ValueError, OverflowError):
            # A malformed/oversized norm cannot be a healthy measurement;
            # classify it with the non-finite norms and keep the report
            # JSON-safe.
            grad_norm_val = None
        grad_norm_bad = (grad_norm is not None and grad_norm_val is None)
        detail = {'non_finite': bad_grads[:10], 'grad_norm': grad_norm_val}
        if bad_grads:
            if verdict is None:
                verdict = 'gradients_nonfinite'
                summary = (f'{len(bad_grads)} gradient tensor(s) non-finite '
                           f'(first: {bad_grads[0]}): add clip_grad_norm_, '
                           f'check for log(0)/division by zero in the loss, '
                           f'and lower the LR')
            record('gradients', 'fail', detail)
        elif grad_norm_bad or (grad_norm_val is not None
                               and not math.isfinite(grad_norm_val)):
            if verdict is None:
                verdict = 'gradients_nonfinite'
                summary = 'total gradient norm is non-finite (explosion)'
            record('gradients', 'fail', detail)
        elif grad_norm_val is not None and grad_norm_val > 1e3:
            record('gradients', 'warn',
                   {**detail, 'note': 'gradients are exploding; clip and/or '
                                      'lower the learning rate'})
        else:
            record('gradients', 'pass', detail)
    else:
        record('gradients', 'skipped', 'no gradients / grad norm supplied')

    if verdict is None and (loss_malformed
                            or (loss_val is not None
                                and not math.isfinite(loss_val))):
        verdict = 'loss_nonfinite'
        if loss_malformed:
            summary = ('the loss is not representable as a finite float '
                       'while every supplied stage was finite: trace the '
                       'forward pass for log(0) / division by zero / unsafe '
                       'softmax (supply logits and grads for a narrower '
                       'verdict)')
        else:
            summary = (f'the loss is {loss_val} while every supplied stage '
                       f'was finite: trace the forward pass for log(0) / '
                       f'division by zero / unsafe softmax (supply logits and '
                       f'grads for a narrower verdict)')

    return {
        'context': context,
        'verdict': verdict or 'no_nonfinite_signal',
        'summary': summary,
        'first_failing_stage': verdict,
        'checks': checks,
    }


def _finite_float_or_none(value):
    """Finite non-bool float for a metric, or ``None`` when not usable.

    Malformed result rows (a string placeholder, a list, ``True``) must be
    listed under ``skipped`` like a missing/NaN value; calling ``float()``
    on them used to abort the whole domination pass with TypeError /
    ValueError instead of annotating the remaining rows.
    """
    if isinstance(value, (bool, str, bytes)):
        # A JSON string (even a numeric-looking one) is a malformed metric,
        # not a measurement: the row must be skipped so stringified output
        # cannot silently enter the Pareto frontier.
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_ratio(numerator, denominator) -> float | None:
    """Ratio with a non-positive / non-finite denominator, or overflow → None."""
    if denominator is None:
        return None
    try:
        denominator = float(denominator)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None
    try:
        ratio = float(numerator) / denominator
    except (OverflowError, ZeroDivisionError):
        return None
    return ratio if math.isfinite(ratio) else None


def _pareto_row_keys(rows: Sequence[Mapping], name_key: str) -> list[str]:
    """Unique keys for the rows, disambiguating repeated names.

    ``pareto_frontier`` keys its points by the row name.  Two rows with
    the SAME name used to collide: the later one overwrote the earlier
    point, so the earlier row vanished from the dominance analysis AND
    both rows were annotated with the SURVIVING row's ratios.  Duplicates
    now get a stable ``name#2`` / ``name#3`` suffix.

    The generated suffix can itself be a literal name (``a``, ``a#2``,
    ``a``), so keys are checked against every already-assigned key and
    the suffix is advanced until it is unique.  A non-hashable name (e.g.
    a list from a malformed row) is keyed by its ``repr`` so it cannot
    abort the pass.
    """
    seen: dict = {}
    used: set = set()
    keys: list[str] = []
    for index, r in enumerate(rows):
        if not isinstance(r, Mapping):
            name = f'row#{index}'
        else:
            name = r.get(name_key, r.get('op', '?'))
        try:
            hash(name)
        except TypeError:
            name = repr(name)
        count = seen.get(name, 0) + 1
        key = name if count == 1 else f'{name}#{count}'
        while key in used:
            count += 1
            key = f'{name}#{count}'
        seen[name] = count
        used.add(key)
        keys.append(key)
    return keys


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

    Repeated names are disambiguated with a ``name#2`` suffix (see
    :func:`_pareto_row_keys`) so that no row is silently dropped.

    Returns ``{'frontier': [names], 'dominated_by': {name: dominator},
    'points': {...}, 'skipped': [...]}``.
    """
    points: dict = {}
    skipped: list = []
    row_keys = _pareto_row_keys(rows, name_key)
    for row, key in zip(rows, row_keys):
        if not isinstance(row, Mapping):
            skipped.append(key)
            continue
        r = row
        acc = _finite_float_or_none(r.get(accuracy_key))
        if acc is None:
            skipped.append(key)
            continue
        costs = {}
        ok = True
        for k in cost_keys:
            v = _finite_float_or_none(r.get(k))
            if v is None:
                ok = False
                break
            costs[k] = v
        if not ok:
            skipped.append(key)
            continue
        eval_ms = _finite_float_or_none(r.get('mean_eval_ms_per_sample'))
        if eval_ms is not None and eval_ms <= 0:
            eval_ms = None
        points[key] = {
            'accuracy': acc,
            'costs': costs,
            'acc_per_train_second': _safe_ratio(
                acc, costs.get('mean_train_time_s')),
            'acc_per_1k_params': _safe_ratio(
                acc, (costs['n_params'] / 1000.0)
                if costs.get('n_params', 0.0) > 0 else None),
            'acc_per_eval_ms': _safe_ratio(acc, eval_ms),
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
    name_key = kwargs.get('name_key', 'op')
    row_keys = _pareto_row_keys(rows, name_key)
    for r, key in zip(rows, row_keys):
        if not isinstance(r, dict):
            # ``pareto_frontier`` already listed the malformed entry under
            # ``skipped``; there is nowhere to annotate it in place.
            continue
        p = report['points'].get(key)
        if p is None:
            r['pareto'] = None
            continue
        r['pareto'] = {
            'optimal': key in report['frontier'],
            'dominated_by': report['dominated_by'].get(key),
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

# The two categories that are NOT errors: a 'correct' prediction is by
# definition not a mistake, and an 'ignored' position is masked out of the
# loss entirely.  They are part of the position partition, so they carry a
# share of ALL positions, but they contribute NO share of the wrong ones.
_LM_NON_ERROR_CATEGORIES = frozenset(('correct', 'ignored'))


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
      * ``seen_earlier`` / ``last_seen_distance`` — whether the target
        occurred in the context available for this prediction, and the
        distance from the PREDICTED position to its most recent occurrence.
        The token just consumed (``input_ids[:, t]``) is part of that
        context, so an immediate repetition has distance 1; excluding it
        mislabelled repeat failures as ``novel_token``. Distance buckets
        turn "the model can't copy" into "the model can copy at distance d
        but not at distance d'", which is exactly the evidence a
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
    if not logits.is_floating_point():
        raise ValueError(
            f'logits must be a floating-point tensor, got {logits.dtype}')
    # A NaN/+Inf logit would silently fall through every category test and
    # be recorded as a plausible-looking error (usually ``novel_token``),
    # and its probability/entropy would make the summary NaN. Refuse the
    # report instead: the caller already has diagnose_nonfinite() for the
    # staged data -> log(0)/div0 -> softmax -> AMP -> grad/LR triage.
    if logits.numel() and not bool(torch.isfinite(logits).all()):
        n_bad = int((~torch.isfinite(logits)).sum())
        raise ValueError(
            f'logits contain {n_bad} non-finite value(s); run '
            f'diagnose_nonfinite() on the forward pass before computing '
            f'token bad-case diagnostics')
    for name, ids in (('labels', labels), ('input_ids', input_ids)):
        if (ids.dtype == torch.bool or ids.is_floating_point()
                or ids.is_complex()):
            raise ValueError(
                f'{name} must contain integer token ids, got {ids.dtype}')
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise ValueError(
            f'ignore_index={ignore_index!r} must be an integer')
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f'top_k={top_k!r} must be a positive int')
    if top_k < 1:
        raise ValueError(f'top_k={top_k} must be a positive int')
    if (isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))):
        raise ValueError(
            f'confidence={confidence!r} must be a number in [0, 1]')
    try:
        confidence = float(confidence)
    except OverflowError:
        raise ValueError(
            f'confidence={confidence} must be a finite number in [0, 1]'
        ) from None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f'confidence={confidence} must be a finite number in [0, 1]')
    if t and v < 1:
        raise ValueError('logits must have a non-empty vocabulary')
    valid = labels != int(ignore_index)
    if valid.any():
        # A label outside the vocabulary used to be clamped into range for
        # the rank lookup, silently scoring a different token and emitting
        # a plausible category. Refuse it: a malformed label is a data bug,
        # not a model failure.
        bad_label = valid & ((labels < 0) | (labels >= v))
        if bool(bad_label.any()):
            raise ValueError(
                f'labels contain {int(bad_label.sum())} token id(s) outside '
                f'[0, vocab={v}) (ignoring {ignore_index}); fix the label '
                f'pipeline before computing token bad-case diagnostics')
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
    # The model predicts position t+1 after consuming input_ids[t], so the
    # current token is context too (diagonal 0, not -1). With a strict
    # prefix mask, ``... 3, 3`` made the second 3 look unseen and turned an
    # easy copy into a "novel token".
    lower = torch.ones(t, t, dtype=torch.bool, device=logits.device)
    lower = lower.tril(diagonal=0)
    seen = seen & lower.unsqueeze(0)
    seen_earlier = seen.any(dim=-1) & valid
    idx = torch.arange(t, device=logits.device)
    no_pos = torch.full(seen.shape, -1, dtype=torch.long,
                        device=logits.device)
    last_pos = torch.where(seen, idx.view(1, 1, t).expand_as(seen),
                           no_pos).max(dim=-1).values
    # Distance is measured from the predicted position (t+1): an immediate
    # repetition is distance 1, the previous token is distance 2, etc.
    last_seen_distance = torch.where(
        last_pos >= 0, (idx + 1).view(1, t) - last_pos,
        torch.full_like(last_pos, -1))

    k = top_k
    conf = confidence
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
                'categories': {},
                'target_rank_on_errors': {'mean': None, 'median': None},
                'top1_prob_on_errors': {'mean': None},
                'top1_prob_on_correct': {'mean': None},
                'confidence_calibration': confidence_calibration([], []),
                'accuracy_by_position_quartile': [],
                'accuracy_by_repeat_distance': [],
                'note': 'no diagnostics supplied'}

    valid = torch.cat([d['valid'].reshape(-1) for d in items])
    correct = torch.cat([d['correct'].reshape(-1) for d in items])
    target_rank = torch.cat([d['target_rank'].reshape(-1) for d in items])
    top1_prob = torch.cat([d['top1_prob'].reshape(-1) for d in items])
    seen_earlier = torch.cat([d['seen_earlier'].reshape(-1) for d in items])
    distance = torch.cat([d['last_seen_distance'].reshape(-1) for d in items])
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
        # 'rate_of_wrong' is the share of the WRONG tokens that fall into
        # this category.  'correct' and 'ignored' are not errors, so they
        # contribute no share of them (dividing their count by ``n_wrong``
        # reported e.g. "67% of the errors were masked positions" and could
        # exceed 1.0 on padding-heavy batches).  The error categories still
        # sum to exactly 1.
        is_error = name not in _LM_NON_ERROR_CATEGORIES
        cat_report[name] = {
            'count': int(c),
            'rate_of_all': (c / n_all) if n_all else 0.0,
            'rate_of_wrong': ((c / n_wrong)
                              if (n_wrong and is_error) else 0.0),
            'definition': definition,
        }

    err = valid & ~correct
    rank_on_err = target_rank[err].float()
    prob_err = top1_prob[err].float()
    prob_ok = top1_prob[valid & correct].float()
    calibration = confidence_calibration(top1_prob[valid], correct[valid])

    def _acc(mask: torch.Tensor) -> dict:
        n = int(mask.sum())
        return {
            'n': n,
            'accuracy': (float(correct[mask].float().mean()) if n else None),
        }

    lengths = [int(d['position'].shape[-1]) for d in items]
    pos_buckets = []
    if n_tokens:
        position_bucket = torch.cat([
            (d['position'].reshape(-1) * 4)
            // max(1, int(d['position'].shape[-1]))
            for d in items
        ])
        uniform_width = lengths[0] if len(set(lengths)) == 1 else None
        for qi in range(4):
            mask = valid & (position_bucket == qi)
            entry = _acc(mask)
            entry['quartile'] = f'q{qi + 1}_of_4'
            if uniform_width is None:
                entry['position_range'] = None
                entry['relative_position_range'] = [qi / 4, (qi + 1) / 4]
            else:
                entry['position_range'] = [
                    -(-qi * uniform_width // 4),
                    -(-(qi + 1) * uniform_width // 4)]
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
        # The confidence-based ``confident_wrong`` category assumes that a
        # raw top-1 probability means what it says. Test that assumption on
        # the same tokens instead of leaving it to the reader: ECE/MCE +
        # Brier skill + discrimination, under two binning rules.
        'confidence_calibration': calibration,
        'accuracy_by_position_quartile': pos_buckets,
        'accuracy_by_repeat_distance': dist_buckets,
        'note': ('Token-level bad cases on a held-out split: category mix '
                 'separates near-misses / confident blind spots / copy '
                 'failures / novel tokens, and the repeat-distance curve '
                 'localizes long-range retrieval failures. '
                 'confidence_calibration reports whether top1_prob is a '
                 'usable probability, which the confidence threshold in the '
                 'category assignment implicitly assumes.'),
    }


# ---------------------------------------------------------------------------
# Classification / ranking metrics
# ---------------------------------------------------------------------------
#
# Design rules (scientific-methodology contract):
#
#   * The averaging convention is part of the metric NAME. ``f1_macro`` is
#     NOT comparable to ``f1_weighted`` under class imbalance, and a bare
#     "F1" is ambiguous — callers must pick one and the JSON says which.
#   * Undefined values are ``None`` (never 0.0 / NaN): a class with no
#     support has undefined recall, and a ranking with only positives (or
#     only negatives) has undefined AUC. Emitting 0.0 would silently drag
#     an average down; emitting NaN would break strict JSON.
#   * Accuracy is reported alongside the F1 variants but is never the only
#     metric: under imbalance it hides a collapsed predictor that always
#     predicts the majority class (high accuracy, near-zero minority F1).


def _class_id_tensor(values, name: str) -> torch.Tensor:
    """Flatten integer class ids; never truncate floats or coerce bools."""
    if not isinstance(values, torch.Tensor):
        def _has_bool(value) -> bool:
            if isinstance(value, bool):
                return True
            if isinstance(value, (list, tuple)):
                return any(_has_bool(item) for item in value)
            return False

        if _has_bool(values):
            raise ValueError(
                f'{name} must contain integer class ids, got a bool element')
    try:
        tensor = torch.as_tensor(values)
    except (TypeError, ValueError, OverflowError, RuntimeError) as e:
        raise ValueError(
            f'{name} must contain integer class ids '
            f'(got {type(values).__name__})') from e
    if tensor.numel() and (tensor.dtype == torch.bool
                           or tensor.is_floating_point()
                           or tensor.is_complex()):
        raise ValueError(
            f'{name} must contain integer class ids, got dtype {tensor.dtype}')
    return tensor.reshape(-1).to(dtype=torch.long)


def confusion_matrix(pred, target, *, num_classes: int | None = None):
    """Integer confusion matrix ``C[i, j] = #(target == i, pred == j)``.

    Returns a ``[num_classes, num_classes]`` ``torch.long`` tensor on CPU.
    ``num_classes`` defaults to ``max(observed class ids) + 1``; pass it
    explicitly to include classes that never appear in this sample (their
    rows/columns stay zero, which is exactly the evidence needed to detect
    a class that is never predicted).
    """
    pred = _class_id_tensor(pred, 'pred')
    target = _class_id_tensor(target, 'target')
    if pred.numel() != target.numel():
        raise ValueError(
            f'pred/target length mismatch: {pred.numel()} vs {target.numel()}')
    if target.numel() and int(target.min()) < 0:
        raise ValueError('target contains negative class ids')
    if pred.numel() and int(pred.min()) < 0:
        raise ValueError('pred contains negative class ids')
    explicit_num_classes = num_classes is not None
    if num_classes is None:
        max_id = int(max(pred.max(), target.max()).item()) if pred.numel() else -1
        num_classes = max_id + 1
    elif isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ValueError(
            f'num_classes={num_classes!r} must be a non-negative int or None')
    if num_classes < 0:
        raise ValueError(f'num_classes={num_classes} must be >= 0')
    if num_classes > 0:
        max_cells = torch.iinfo(torch.long).max
        if num_classes > max_cells // num_classes:
            raise ValueError(
                f'num_classes={num_classes} is too large to materialize a '
                f'confusion matrix')
    if explicit_num_classes and pred.numel():
        max_id = int(max(pred.max(), target.max()).item())
        if max_id >= num_classes:
            # ``flat = target * num_classes + pred`` would otherwise alias
            # an out-of-range id into another class's cell (or overflow the
            # reshape), silently corrupting the matrix.
            raise ValueError(
                f'class id {max_id} is outside [0, num_classes={num_classes})')
    if num_classes == 0:
        return torch.zeros(0, 0, dtype=torch.long)
    flat = target * num_classes + pred
    try:
        counts = torch.bincount(flat, minlength=num_classes * num_classes)
        return counts.reshape(num_classes, num_classes).to(dtype=torch.long)
    except (OverflowError, RuntimeError, MemoryError) as e:
        raise ValueError(
            f'num_classes={num_classes} is too large to materialize a '
            f'confusion matrix') from e


def _prf(tp: float, fp: float, fn: float):
    """(precision, recall, f1) with the standard zero_division=0 convention.

    A class that OCCURS in the targets but is never predicted gets
    precision/recall/f1 = 0.0 (the degenerate predictor is exactly what
    these metrics must expose, not hide behind ``None``). ``None`` is
    reserved for a class absent from both the targets and the
    predictions, where the metric is genuinely undefined.
    """
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0 if fn > 0 else None
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0 if fp > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


_AVERAGING_NOTE = (
    'macro = unweighted mean over classes with support > 0; micro = global '
    'tp/fp/fn counts (equals accuracy for single-label multi-class); '
    'weighted = support-weighted per-class mean. Under class imbalance the '
    'three can disagree substantially; prefer macro/balanced accuracy for '
    'minority-class behaviour, and PR-AUC over ROC-AUC when positives are '
    'rare.')


def _mean_defined(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def multiclass_classification_metrics(pred, target, *, num_classes: int | None = None,
                                      scores=None) -> dict:
    """Confusion-derived metrics with EXPLICIT averaging conventions.

    Parameters
    ----------
    pred, target : array-like[int], shape [N]
        Hard predictions and ground-truth labels (single-label
        multi-class; ``ignore_index`` positions must be filtered by the
        caller).
    num_classes : int, optional
        Class count; defaults to ``max(observed) + 1``.
    scores : array-like[float], shape [N, C], optional
        Per-class scores/probabilities. When given, one-vs-rest PR-AUC and
        ROC-AUC are computed per class and macro-averaged over classes
        with support > 0 (that is the imbalance-robust summary).

    Returns
    -------
    dict with, among others (all keys carry their averaging rule):

      * ``accuracy`` and ``balanced_accuracy`` (macro recall);
      * ``precision_macro`` / ``recall_macro`` / ``f1_macro`` — unweighted
        mean over classes that actually occur (support > 0);
      * ``micro_*`` — global tp/fp/fn counts. For single-label multi-class
        micro precision/recall/F1 all equal ``accuracy``; kept explicit so
        the JSON cannot be misread;
      * ``weighted_*`` — support-weighted mean (equivalent to micro for
        single-label multi-class but computed per class, so per-class
        degenerate classes with support 0 are excluded rather than
        counted as 0);
      * ``per_class`` — support / precision / recall / f1 per class;
      * ``class_balance`` — max/min per-class support ratio (1.0 = balanced;
        large values flag the imbalance that makes accuracy misleading);
      * ``pr_auc_macro_ovr`` / ``roc_auc_macro_ovr`` — only with
        ``scores``; PR-AUC is the preferred summary under imbalance.
    """
    cm = confusion_matrix(pred, target, num_classes=num_classes)
    num_classes = int(cm.shape[0])
    n = int(cm.sum().item())
    if n == 0:
        return {
            'n_samples': 0, 'num_classes': num_classes,
            'accuracy': None, 'balanced_accuracy': None,
            'precision_macro': None, 'recall_macro': None, 'f1_macro': None,
            'precision_micro': None, 'recall_micro': None, 'f1_micro': None,
            'precision_weighted': None, 'recall_weighted': None,
            'f1_weighted': None,
            'per_class': [], 'class_balance': None,
            'imbalance_note': None,
            'averaging_note': _AVERAGING_NOTE,
        }

    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)

    per_class = []
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        p, r, f = _prf(float(tp[c]), float(fp[c]), float(fn[c]))
        per_class.append({
            'class': c,
            'support': int(support[c]),
            'predicted': int(tp[c] + fp[c]),
            'precision': p,
            'recall': r,
            'f1': f,
        })
        # Macro averages only over classes that actually occur in the
        # targets (support > 0): a class absent from this held-out sample
        # has an undefined metric, and folding a 0.0 in would fabricate a
        # penalty. This is the convention named in the key ``_macro``.
        if int(support[c]) > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

    accuracy = float(tp.sum().item()) / n
    balanced_accuracy = _mean_defined(recalls)

    sum_tp = float(tp.sum())
    sum_fp = float(fp.sum())
    sum_fn = float(fn.sum())
    p_micro, r_micro, f_micro = _prf(sum_tp, sum_fp, sum_fn)

    total_support = float(support.sum().item())

    def _weighted(values):
        # ``values`` must be indexed by CLASS ID (e.g. the per-class
        # precision/recall/f1 pulled out of ``per_class``), not a compact
        # support>0 list — the weights below are per-class supports.
        if total_support <= 0:
            return None
        num = 0.0
        present = False
        for c in range(num_classes):
            v = values[c]
            if v is None or int(support[c]) <= 0:
                continue
            present = True
            num += (float(support[c]) / total_support) * v
        # The weights over support>0 classes sum to 1 by construction; a
        # None entry means that class never appeared in targets or
        # predictions and was excluded from the average entirely.
        return float(num) if present else None

    nonzero_support = [int(s) for s in support.tolist() if s > 0]
    class_balance = None
    if nonzero_support:
        class_balance = (max(nonzero_support) / min(nonzero_support)
                         if min(nonzero_support) > 0 else None)
    # Imbalance-aware guidance (metadata only; does not alter any metric).
    # Under extreme imbalance accuracy and ROC-AUC are overly optimistic
    # (the huge negative mass makes FPR grow slowly), so PR-AUC / macro F1
    # should be the primary ranking metrics; see the README fairness notes.
    imbalance_note = None
    if class_balance is not None:
        if class_balance >= 1000:
            imbalance_note = (
                f'extreme class imbalance (max/min support = {class_balance:.0f}:1): '
                'accuracy and ROC-AUC are overly optimistic; prefer PR-AUC '
                '(pr_auc_macro_ovr) and macro F1 (f1_macro) as the primary '
                'ranking metrics.'
            )
        elif class_balance >= 100:
            imbalance_note = (
                f'class imbalance (max/min support = {class_balance:.0f}:1): '
                'prefer macro F1 / PR-AUC over raw accuracy.'
            )

    report = {
        'n_samples': n,
        'num_classes': num_classes,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision_macro': _mean_defined(precisions),
        'recall_macro': _mean_defined(recalls),
        'f1_macro': _mean_defined(f1s),
        'precision_micro': p_micro,
        'recall_micro': r_micro,
        'f1_micro': f_micro,
        'precision_weighted': _weighted(
            [pc['precision'] for pc in per_class]),
        'recall_weighted': _weighted([pc['recall'] for pc in per_class]),
        'f1_weighted': _weighted([pc['f1'] for pc in per_class]),
        'per_class': per_class,
        'class_balance': class_balance,
        'imbalance_note': imbalance_note,
        'averaging_note': _AVERAGING_NOTE,
    }

    if scores is not None:
        try:
            scores_t = torch.as_tensor(scores, dtype=torch.float32)
        except (TypeError, ValueError, OverflowError, RuntimeError) as e:
            raise ValueError(
                'scores must be a numeric [N, num_classes] array or tensor '
                f'(got {type(scores).__name__})') from e
        if scores_t.ndim != 2 or scores_t.shape[1] != num_classes:
            raise ValueError(
                f'scores must be [N, {num_classes}], got '
                f'{tuple(scores_t.shape)}')
        target_t = torch.as_tensor(target, dtype=torch.long).reshape(-1)
        pr_aucs, roc_aucs = [], []
        for c in range(num_classes):
            if int(support[c]) <= 0:
                continue
            labels = (target_t == c)
            sub = binary_ranking_metrics(scores_t[:, c], labels)
            pr_aucs.append(sub['pr_auc'])
            roc_aucs.append(sub['roc_auc'])
        report['pr_auc_macro_ovr'] = _mean_defined(pr_aucs)
        report['roc_auc_macro_ovr'] = _mean_defined(roc_aucs)
    return report


def expected_average_precision(n_pos: int, n_neg: int) -> float | None:
    """Exact expected average precision of a uniformly random ranking.

    For one query with ``n_pos`` relevant items among
    ``n = n_pos + n_neg`` candidates, average precision (AP) is

        AP = (1 / n_pos) * sum_{i: y_i = 1} (TP_i / i).

    If the ranking is a uniformly random permutation, the expected AP is
    **not** the prevalence ``n_pos / n`` (a common but incorrect
    "chance AP" shorthand).  Conditioning on a relevant item at rank
    ``r``, the expected precision there is::

        [1 + (r - 1) * (n_pos - 1) / (n - 1)] / r,

    because each earlier slot is relevant with probability
    ``(n_pos - 1) / (n - 1)``.  Summing over ranks and simplifying gives

        E[AP] = (H_n + (n_pos - 1) / (n - 1) * (n - H_n)) / n,

    where ``H_n`` is the n-th harmonic number.  For a single relevant
    item this is ``H_n / n`` -- identical to chance mean reciprocal rank
    (which is why the two chance references must agree for the
    one-gold-block indexer diagnostic).

    The prevalence is the area under the *expected* precision-recall
    curve, a different (smoothed) quantity; it understates the expected
    AP of a finite random ranking, badly so for rare positives
    (``H_100 / 100 ~= 0.052`` vs prevalence ``0.01``).

    Returns ``None`` when there are no positives (AP undefined) and 1.0
    when every candidate is relevant.
    """
    for name, value in (('n_pos', n_pos), ('n_neg', n_neg)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name}={value!r} must be a non-negative int')
        if value < 0:
            raise ValueError('n_pos and n_neg must be >= 0')
    if n_pos == 0:
        return None
    n = n_pos + n_neg
    if n_neg == 0:
        return 1.0
    try:
        n_float = float(n)
    except OverflowError:
        raise ValueError(
            f'expected_average_precision: n_pos+n_neg={n!r} is too large '
            f'to represent as a finite float.') from None
    if not math.isfinite(n_float):
        raise ValueError(
            f'expected_average_precision: n_pos+n_neg={n!r} is too large '
            f'to represent as a finite float.')
    if n <= 1_000_000:
        harmonic = sum(1.0 / i for i in range(1, n + 1))
    else:
        # Euler-Maclaurin: the exact loop is O(n), and JSON integers are
        # unbounded. The first terms are accurate far beyond float64
        # precision for n > 1e6, while a too-large n is rejected above.
        inv = 1.0 / n_float
        harmonic = (math.log(n_float) + 0.5772156649015328606
                    + 0.5 * inv - inv * inv / 12.0 + inv ** 4 / 120.0)
    return float((harmonic
                  + (n_pos - 1) / (n - 1) * (n - harmonic)) / n)


def binary_ranking_metrics(scores, labels, *, valid=None) -> dict:
    """Threshold-free ranking metrics for a binary decision / retrieval.

    ``scores`` is a 1-D score per candidate (higher = more likely
    positive), ``labels`` the 0/1 ground truth. ``valid`` optionally masks
    candidates (e.g. causally unavailable blocks) out of the ranking.

    Metrics
    -------
    pr_auc
        Average precision (area under the precision-recall curve, step
        interpolation). PRIMARY metric when positives are rare: ROC-AUC
        stays high because FPR grows slowly over a huge negative base,
        while PR-AUC drops toward its (finite-sample) chance expectation
        as the ranking degrades.
    roc_auc
        Mann-Whitney U / rank statistic with average ranks for ties.
    mrr
        Reciprocal rank of the highest-scored positive (1.0 = best).
    recall_at_1 / precision_at_1
        ``precision_at_1`` is the top-ranked candidate's label (1.0 or
        0.0). ``recall_at_1`` is the fraction of all relevant items found
        in that one slot, so with several positives it is ``1/n_pos``
        when the top candidate is relevant (and 0.0 otherwise). The two
        coincide for a single relevant item.
    chance_pr_auc / chance_roc_auc
        Chance references so a number can be judged. ``chance_roc_auc``
        is 0.5; ``chance_pr_auc`` is the EXACT expected AP under a
        uniformly random tie-free ranking (``expected_average_precision``),
        not the prevalence -- for a single positive among n candidates it
        is ``H_n / n``. The two differ most when positives are rare.

    Returns ``None`` for pr_auc / roc_auc / mrr when the sample has only
    one class (undefined), never 0.0 or NaN. Equal scores are handled
    deterministically: ROC-AUC uses average ranks (tie-corrected);
    PR-AUC breaks ties by candidate order — document/track that when
    scores come from a quantized model.
    """
    try:
        scores_t = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
        labels_raw = torch.as_tensor(labels).reshape(-1)
    except (TypeError, ValueError, OverflowError, RuntimeError) as e:
        # JSON integers are unbounded and a numeric-looking string is not a
        # score: both must become an actionable ValueError rather than an
        # OverflowError / torch conversion error from the parser.
        raise ValueError(
            'scores/labels must be numeric 1-D arrays or tensors '
            f'(got {type(scores).__name__}/{type(labels).__name__})') from e
    if scores_t.numel() != labels_raw.numel():
        raise ValueError(
            f'scores/labels length mismatch: {scores_t.numel()} vs '
            f'{labels_raw.numel()}')
    if labels_raw.dtype.is_complex:
        raise ValueError('labels must be boolean or exact 0/1 values')
    if valid is not None:
        try:
            valid_raw = torch.as_tensor(valid).reshape(-1)
        except (TypeError, ValueError, OverflowError, RuntimeError) as e:
            raise ValueError(
                'valid must be a boolean mask (or exact 0/1 values), got '
                f'{type(valid).__name__}') from e
        if valid_raw.numel() != scores_t.numel():
            raise ValueError('valid mask length mismatch')
        if valid_raw.dtype.is_complex:
            raise ValueError(
                'valid must be a boolean mask (or exact 0/1 values), '
                f'got {valid!r}')
        if valid_raw.dtype != torch.bool:
            if valid_raw.numel() and not bool(
                    ((valid_raw == 0) | (valid_raw == 1)).all().item()):
                raise ValueError(
                    'valid must be a boolean mask (or exact 0/1 values), '
                    f'got {valid!r}')
            valid_raw = valid_raw.to(dtype=torch.bool)
        scores_t = scores_t[valid_raw]
        labels_raw = labels_raw[valid_raw]
    if labels_raw.numel() and not bool(
            ((labels_raw == 0) | (labels_raw == 1)).all().item()):
        raise ValueError('labels must contain only 0/1 binary values')
    labels_t = labels_raw.to(dtype=torch.bool)
    if scores_t.numel() and not bool(torch.isfinite(scores_t).all().item()):
        raise ValueError('scores must be finite for the kept candidates')

    n = int(scores_t.numel())
    n_pos = int(labels_t.sum().item())
    n_neg = n - n_pos
    prevalence = (n_pos / n) if n else None
    report = {
        'n': n,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'prevalence': prevalence,
        'pr_auc': None,
        'roc_auc': None,
        'mrr': None,
        'recall_at_1': None,
        'precision_at_1': None,
        'chance_pr_auc': expected_average_precision(n_pos, n_neg),
        'chance_roc_auc': 0.5 if n else None,
        'note': ('PR-AUC is the primary metric when positives are rare; '
                 'ROC-AUC is optimistic under a large negative base.'),
    }
    if n_pos == 0 or n_neg == 0:
        return report

    # ---- ROC-AUC via average ranks (tie-corrected) -------------------
    order = torch.argsort(scores_t)
    sorted_scores = scores_t[order]
    diff = torch.ones(n, dtype=torch.bool)
    diff[1:] = sorted_scores[1:] != sorted_scores[:-1]
    group_id = diff.cumsum(0) - 1
    group_sizes = torch.bincount(group_id)
    group_starts = torch.cat([
        torch.zeros(1, dtype=torch.long),
        group_sizes.cumsum(0)[:-1],
    ]).to(dtype=torch.float64)
    avg_rank_sorted = (group_starts[group_id]
                       + (group_sizes[group_id].to(dtype=torch.float64) + 1.0) / 2.0)
    avg_rank = torch.empty(n, dtype=torch.float64)
    avg_rank[order] = avg_rank_sorted
    rank_sum_pos = float(avg_rank[labels_t].sum().item())
    roc_auc = ((rank_sum_pos - n_pos * (n_pos + 1) / 2.0)
               / (n_pos * n_neg))
    report['roc_auc'] = float(roc_auc)

    # ---- PR-AUC (average precision, step interpolation) --------------
    # Descending score order; ties broken by candidate index so the
    # result is deterministic across runs (torch.sort is not guaranteed
    # stable for all backends).
    order_desc = torch.argsort(scores_t, descending=True, stable=True)
    labels_desc = labels_t[order_desc].to(dtype=torch.float64)
    tp_cum = torch.cumsum(labels_desc, dim=0)
    k = torch.arange(1, n + 1, dtype=torch.float64)
    precision_k = tp_cum / k
    recall_k = tp_cum / n_pos
    recall_prev = torch.cat([torch.zeros(1, dtype=torch.float64),
                             recall_k[:-1]])
    report['pr_auc'] = float(((recall_k - recall_prev) * precision_k).sum().item())

    # ---- MRR / recall@1 ---------------------------------------------
    first_pos_rank = int(torch.nonzero(labels_desc, as_tuple=False)[0].item()) + 1
    report['mrr'] = 1.0 / first_pos_rank
    top1 = float(labels_desc[0].item())
    report['precision_at_1'] = top1
    # With one positive the old boolean is numerically identical to the
    # fraction and remains the documented type; with several positives,
    # recall@1 must divide by the number of relevant items.
    report['recall_at_1'] = bool(top1) if n_pos == 1 else top1 / n_pos
    return report


def _calibration_table(probabilities: torch.Tensor,
                       outcomes: torch.Tensor,
                       assignments: torch.Tensor, *, n_bins: int,
                       binning: str,
                       equal_width_bounds: bool) -> dict:
    """Reliability table + ECE/MCE for one fixed bin assignment.

    ``assignments[i]`` is the bin index of observation ``i``. Empty bins
    are kept in the table (with ``count = 0`` and undefined statistics) so
    the JSON schema does not change with the sample. ECE is the
    count-weighted mean absolute gap; MCE is the largest absolute gap
    among non-empty bins. Both are descriptive, not a test.
    """
    n = int(probabilities.numel())
    reliability = []
    ece = 0.0
    mce = None
    for b in range(n_bins):
        mask = assignments == b
        count = int(mask.sum())
        entry = {
            'bin': b,
            'lower': (b / n_bins if equal_width_bounds else None),
            'upper': ((b + 1) / n_bins if equal_width_bounds else None),
            'count': count,
            'fraction': (count / n if n else None),
            'mean_confidence': None,
            'accuracy': None,
            'confidence_minus_accuracy': None,
            'abs_confidence_gap': None,
        }
        if count:
            mean_conf = float(probabilities[mask].mean().item())
            accuracy = float(outcomes[mask].mean().item())
            gap = mean_conf - accuracy
            abs_gap = abs(gap)
            ece += (count / n) * abs_gap
            mce = abs_gap if mce is None else max(mce, abs_gap)
            entry.update({
                'mean_confidence': mean_conf,
                'accuracy': accuracy,
                'confidence_minus_accuracy': gap,
                'abs_confidence_gap': abs_gap,
            })
            if not equal_width_bounds:
                entry['lower'] = float(probabilities[mask].min().item())
                entry['upper'] = float(probabilities[mask].max().item())
        reliability.append(entry)
    return {
        'binning': binning,
        'n_bins': n_bins,
        'ece': (ece if n else None),
        'mce': mce,
        'reliability': reliability,
        'boundary_note': ('fixed equal-width probability bins'
                          if equal_width_bounds else
                          'equal-count bins; lower/upper are the observed '
                          'confidence range, and tied scores may be split '
                          'across adjacent bins'),
    }


def _equal_mass_assignments(probabilities: torch.Tensor,
                            n_bins: int) -> torch.Tensor:
    """Stable equal-count bin indices for ``probabilities``.

    Ties are ordered by input position (stable sort), so a run is
    deterministic and never depends on torch's default unstable sort.
    """
    n = int(probabilities.numel())
    assignments = torch.zeros(n, dtype=torch.long,
                              device=probabilities.device)
    if n == 0:
        return assignments
    order = torch.argsort(probabilities, stable=True)
    base, extra = divmod(n, n_bins)
    start = 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        if size:
            assignments[order[start:start + size]] = b
        start += size
    return assignments


def confidence_calibration(probabilities, correct, *, n_bins: int = 10) -> dict:
    """Calibration analysis of a confidence score against binary outcomes.

    This answers a question that accuracy / AUC / error categories cannot:
    when the model says "0.9", is it right about 90% of the time? A model
    can discriminate well (rank correct predictions above wrong ones) and
    still be systematically overconfident, which makes any confidence-
    thresholded diagnostic (e.g. the ``confident_wrong`` category in
    :func:`lm_token_diagnostics`) systematically misleading.

    Parameters
    ----------
    probabilities : array-like[float], shape [N]
        Predicted confidence in the event that ``correct`` is 1. Values
        must be finite and lie in ``[0, 1]``; invalid values raise instead
        of being silently dropped, because dropping low-confidence or
        high-confidence tokens would change the calibration estimate.
    correct : array-like[bool or 0/1], shape [N]
        Whether the prediction was correct (1) or wrong (0).
    n_bins : int, optional
        Number of reliability bins (default 10).

    Returns
    -------
    dict with:

      * ``n`` / ``accuracy`` / ``mean_confidence``;
      * ``confidence_minus_accuracy`` — positive means overconfidence
        (the reported confidence exceeds the observed hit rate);
      * ``brier_score`` — mean squared error of the confidence forecast;
      * ``brier_baseline`` — Brier score of always predicting the base
        rate (``accuracy``), and ``brier_skill_score = 1 - Brier /
        baseline``. A skill score ``<= 0`` means the confidence score has
        no probabilistic resolution beyond the base rate; it is ``None``
        when every outcome is the same class (baseline 0).
      * ``discrimination`` — ROC-AUC / PR-AUC of confidence as a ranking
        of correct over wrong predictions, with their chance references.
        This is **not** a calibration measure: high AUC + poor ECE is a
        common and important combination.
      * ``equal_width`` / ``equal_mass`` — two ECE/MCE + reliability
        tables. ECE is sensitive to the binning rule, so both are
        reported and ``binning_sensitivity`` is their absolute
        difference; a conclusion that depends on one binning is fragile.
      * ``notes`` — interpretation contract, including the fact that ECE /
        MCE are descriptive and that empty bins are retained.

    All values are strict-JSON scalars/containers (``None`` for
    undefined), and the function is deterministic (no RNG).
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise ValueError(f'n_bins={n_bins!r} must be a positive int')
    if n_bins < 1:
        raise ValueError(f'n_bins={n_bins} must be a positive int')
    if n_bins > 1_000_000:
        raise ValueError(
            f'n_bins={n_bins} is too large to build a reliability table')

    try:
        prob = torch.as_tensor(probabilities,
                               dtype=torch.float64).reshape(-1)
        outcomes_raw = torch.as_tensor(correct).reshape(-1)
    except (TypeError, ValueError, OverflowError, RuntimeError) as e:
        raise ValueError(
            'probabilities/correct must be numeric 1-D arrays or tensors '
            f'(got {type(probabilities).__name__}/'
            f'{type(correct).__name__})') from e
    if prob.numel() != outcomes_raw.numel():
        raise ValueError(
            f'probabilities/correct length mismatch: {prob.numel()} vs '
            f'{outcomes_raw.numel()}')
    if outcomes_raw.dtype.is_complex:
        raise ValueError('correct must be boolean or exact 0/1 values')
    if outcomes_raw.dtype == torch.bool:
        outcomes = outcomes_raw.to(dtype=torch.float64)
    else:
        if (outcomes_raw.numel()
                and bool(((outcomes_raw != 0)
                          & (outcomes_raw != 1)).any().item())):
            raise ValueError(
                'correct must contain only boolean or exact 0/1 values')
        outcomes = outcomes_raw.to(dtype=torch.float64)
    if prob.numel() and not bool(torch.isfinite(prob).all().item()):
        raise ValueError('probabilities must be finite')
    if prob.numel() and bool(((prob < 0.0) | (prob > 1.0)).any().item()):
        raise ValueError('probabilities must lie in [0, 1]')

    n = int(prob.numel())
    accuracy = float(outcomes.mean().item()) if n else None
    mean_conf = float(prob.mean().item()) if n else None
    gap = (mean_conf - accuracy) if n else None
    brier = (float(((prob - outcomes) ** 2).mean().item())
             if n else None)
    baseline = (accuracy * (1.0 - accuracy)
                if accuracy is not None else None)
    ratio = _safe_ratio(brier, baseline)
    bss = (1.0 - ratio) if ratio is not None else None
    if bss is not None and not math.isfinite(bss):
        bss = None

    disc = binary_ranking_metrics(prob, outcomes_raw.to(dtype=torch.bool))
    discrimination = {
        'roc_auc': disc['roc_auc'],
        'pr_auc': disc['pr_auc'],
        'chance_roc_auc': disc['chance_roc_auc'],
        'chance_pr_auc': disc['chance_pr_auc'],
        'positive_class': 'correct',
        'note': ('ranking quality of confidence, not calibration: a high '
                 'AUC can coexist with a large overconfidence gap.'),
    }

    width_assign = torch.clamp((prob * n_bins).to(torch.long),
                               min=0, max=n_bins - 1)
    equal_width = _calibration_table(
        prob, outcomes, width_assign, n_bins=n_bins,
        binning='equal_width', equal_width_bounds=True)
    equal_mass = _calibration_table(
        prob, outcomes, _equal_mass_assignments(prob, n_bins),
        n_bins=n_bins, binning='equal_mass', equal_width_bounds=False)

    sensitivity = None
    if equal_width['ece'] is not None and equal_mass['ece'] is not None:
        sensitivity = abs(equal_width['ece'] - equal_mass['ece'])

    return {
        'n': n,
        'accuracy': accuracy,
        'mean_confidence': mean_conf,
        'confidence_minus_accuracy': gap,
        'brier_score': brier,
        'brier_baseline': baseline,
        'brier_skill_score': bss,
        'discrimination': discrimination,
        'equal_width': equal_width,
        'equal_mass': equal_mass,
        'binning_sensitivity': sensitivity,
        'notes': [
            'ECE/MCE are binning-dependent descriptive statistics, not a '
            'hypothesis test; equal-width and equal-mass values are both '
            'reported so a conclusion cannot hinge on one binning rule.',
            'Brier skill <= 0 means this confidence score adds no '
            'probabilistic resolution beyond always predicting the base '
            'rate; it is None when the base rate is 0 or 1.',
            'A high discrimination (ROC-)AUC only means confidence ranks '
            'correct above wrong; calibrate the probabilities before '
            'interpreting confidence thresholds as error probabilities.',
            'Empty reliability bins are retained (count 0, undefined '
            'statistics) so the record has a stable schema.',
        ],
    }
