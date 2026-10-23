"""Held-out training-curve monitoring and cost/claim judging for the probes.

Motivation
----------
``run_quality.train_one`` (Exp 4) records a held-out validation curve, feeds
it to :func:`results_analysis.diagnose_training_curves` and prints a
bad-case report, so an Exp-4 cell can be *labelled*: overfitting,
underfitting, healthy or saturated. The ablation trainers — ``run_ablation.
eval_layout`` and ``run_ablation.eval_group`` (Exp 5) and
``run_mechanism_ablation.train_variant`` (Exp 8) — only logged the training
loss and the *final* held-out accuracy. That is the "only look at the final
number" failure mode: a cell scoring 0.62 could be overfitting, underfitting
or already at the task ceiling, and the artifact could not say which. Two
consequences:

* an ablation gap (layout A 0.62 vs layout B 0.64) could not be attributed
  to a mechanism, because "B wins" and "A was still improving while B
  saturated" are indistinguishable from the final numbers alone;
* a non-finite loss raised a bare ``RuntimeError("non-finite loss")``, which
  names none of the staged suspects (data -> log(0)/div-by-zero -> softmax
  overflow -> fp16 AMP -> gradient explosion / LR), so a single bad seed
  cost a debugging cycle instead of naming the first failing stage.

This module supplies the missing pieces and is deliberately **additive**:
``final_acc`` keeps its exact meaning (accuracy at the fixed step budget), so
every existing statistic, schema check and figure keeps its semantics. Only
new keys are added.

Three things live here:

1. :class:`ValidationCurve` — a held-out monitor that records
   ``(loss, accuracy)`` at checkpoints, tracks the validation minimum and
   turns train+val curves into a diagnosis, including the *saturated*
   regime (train accuracy still rising, validation accuracy already at the
   ceiling) that a loss-only diagnosis cannot express.
2. :func:`raise_non_finite` — the staged NaN/Inf triage, wrapping
   :func:`results_analysis.diagnose_nonfinite` and raising
   :class:`NonFiniteTrainingError`, whose message names the first failing
   stage.
3. :func:`accuracy_cost_tradeoff` — refuses to call a change an improvement
   when the accuracy gain is below a pre-specified minimum effect or is not
   separable from noise, or when it buys a disproportionate amount of
   parameters / train time / inference latency / peak memory.
4. :func:`pool_error_analyses` — rolls the per-seed bad-case reports up to
   one row-level category distribution (with the worst seed's examples), so
   "which mistake dominates this configuration" is visible without walking
   the per-seed list.

RNG isolation
-------------
Every stream this module relies on is disjoint from the ones the trainers
already use, following ``run_quality`` exactly:

===================  =========================  ==========================
stream               seed offset                used for
===================  =========================  ==========================
training             ``seed + 1_000_000``      the optimization stream
headline eval        ``seed + 5_000_000``      ``final_acc`` / ``final_loss``
validation monitor   ``seed + 6_000_000``      the curve in this module
bad-case analysis    ``seed + 7_000_000``      the error report
===================  =========================  ==========================

so adding monitoring cannot perturb the training batches, the headline
evaluation, or the reproducibility of the existing numbers.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Mapping, NamedTuple, Sequence

import torch

from results_analysis import diagnose_nonfinite, diagnose_training_curves

logger = logging.getLogger(__name__)

# Held-out monitoring stream. Disjoint from the training offset (1_000_000)
# and the headline-eval offset (5_000_000) used by ``_eval_model``, mirroring
# ``run_quality._VAL_MONITOR_SEED_OFFSET``.
VAL_MONITOR_SEED_OFFSET = 6_000_000

# Bad-case stream. Mirrors ``run_quality._ERROR_ANALYSIS_SEED_OFFSET`` so an
# example printed here can never be one the model was trained or selected on.
ERROR_ANALYSIS_SEED_OFFSET = 7_000_000


def _finite_float_or_none(value):
    """Finite real float, or ``None`` for a malformed/oversized metric.

    Result rows are JSON, whose integers are unbounded; ``float(10**400)``
    raises ``OverflowError``. Such a value is not a usable measurement, so
    callers skip it instead of aborting the analysis.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None

# --------------------------------------------------------------------------
# 1. Systematic NaN/Inf triage
# --------------------------------------------------------------------------
class NonFiniteTrainingError(RuntimeError):
    """Raised when a training step produced a non-finite loss or gradient.

    Subclasses :class:`RuntimeError` so the existing ``except Exception``
    handlers around each seed keep behaving exactly as before; the staged
    diagnosis is additionally available programmatically as ``.diagnosis``.
    """

    def __init__(self, message: str, diagnosis: Mapping | None = None):
        super().__init__(message)
        self.diagnosis = dict(diagnosis) if diagnosis else {}


def triage_nonfinite(*, loss=None, context: str = '', **stages) -> dict:
    """Run :func:`results_analysis.diagnose_nonfinite` and return the report.

    Thin wrapper so callers get the same seven-stage order everywhere: data,
    token ids, loss, parameters, logits (softmax overflow), mixed precision,
    gradients/LR. The first stage that can *explain* the value is the
    verdict.
    """
    return diagnose_nonfinite(loss=loss, context=context, **stages)


def _failed_checks(diag: Mapping, limit: int = 3) -> str:
    failed = [c for c in diag.get('checks', ())
              if c.get('status') == 'fail']
    if not failed:
        return ''
    parts = []
    for c in failed[:limit]:
        detail = c.get('detail')
        parts.append(f"{c.get('check')}={detail}" if detail
                     else str(c.get('check')))
    suffix = f" (+{len(failed) - limit} more)" if len(failed) > limit else ''
    return '; '.join(parts) + suffix


def raise_non_finite(kind: str, *, loss=None, context: str = '',
                     **stages) -> None:
    """Diagnose a non-finite training value and raise with the verdict.

    ``kind`` is ``'loss'`` or ``'gradient_norm'`` and only shapes the
    leading sentence. Every tensor worth inspecting should be passed as a
    keyword (``inputs``, ``token_ids``, ``id_range``, ``params``,
    ``logits``, ``grads``, ``grad_norm``, ``amp_enabled``, ``amp_scaler``,
    ``fp32_forward_loss``) — the more stages are supplied, the narrower the
    verdict, because a bare "non-finite loss" is a symptom, not a cause.
    """
    diag = triage_nonfinite(loss=loss, context=context, **stages)
    detail = _failed_checks(diag)
    raise NonFiniteTrainingError(
        f'non-finite {kind} ({context or "training step"}): '
        f"diagnosis [{diag['verdict']}] {diag['summary']}"
        + (f' | failing stages: {detail}' if detail else ''),
        diag)


def triage_gradient(params: Sequence[torch.nn.Parameter],
                    *, grad_norm=None, context: str = '',
                    **stages) -> list[torch.nn.Parameter]:
    """Return the parameters whose gradient is non-finite (empty if none).

    Called after ``backward()`` and BEFORE ``clip_grad_norm_``, so the
    explosion is attributed to the gradients rather than to whatever the
    clipper then did to them.
    """
    return [p for p in params
            if p.grad is not None and not torch.isfinite(p.grad).all()]


def snapshot_parameters(named_parameters) -> dict[str, torch.Tensor]:
    """CPU snapshot of every trainable tensor, for a post-step update audit.

    The overfit gates already check that gradients REACH the parameters
    (:func:`snapshot_parameters` is called before ``optimizer.step()``).
    They did not check that the optimizer actually covers and MOVES those
    parameters: a parameter group that accidentally omits the head, or a
    frozen / detached copy passed to the optimizer, can leave the loss
    falling through the remaining weights and still pass the gate. This
    snapshot lets :func:`optimizer_update_report` close that gap on the
    first optimizer step.

    Tensors are copied to CPU so the audit does not keep a second
    device-resident copy of the model alive during training.
    """
    return {
        str(name): p.detach().to(device='cpu', copy=True)
        for name, p in named_parameters
        if getattr(p, 'requires_grad', False)
    }


def optimizer_update_report(before, named_parameters, optimizer, *,
                            step: int = 0,
                            max_names: int = 10) -> dict:
    """Audit optimizer coverage and whether covered parameters moved.

    ``before`` must be the :func:`snapshot_parameters` output taken before
    ``optimizer.step()``. The report distinguishes three failure modes the
    loss-only overfit criterion cannot see:

      * a trainable tensor absent from every ``optimizer.param_groups``
        (the optimizer can never update it);
      * a tensor with a non-zero gradient that did not change at all after
        the step (frozen / detached copy / wrong device);
      * duplicate parameters registered in several groups (ambiguous
        update semantics; reported but not itself a failure).

    Returns a JSON-serializable dict. ``ok`` is True only when every
    trainable tensor is covered exactly once and every tensor that received
    a non-zero gradient changed. Tensors with missing / zero gradients are
    excluded from the "should have moved" set; the separate
    ``grad_flow_report`` names those.
    """
    if not isinstance(before, Mapping):
        raise TypeError('before must be a snapshot_parameters() mapping')

    grouped: dict[int, torch.nn.Parameter] = {}
    duplicates: list[str] = []
    for group in getattr(optimizer, 'param_groups', ()):
        for p in group.get('params', ()):
            if id(p) in grouped:
                duplicates.append(str(getattr(p, 'name', id(p))))
            grouped[id(p)] = p

    named = [(str(name), p) for name, p in named_parameters
             if getattr(p, 'requires_grad', False)]
    not_in_optimizer = [name for name, p in named if id(p) not in grouped]
    grad_nonzero = [
        name for name, p in named
        if p.grad is not None
        and bool(torch.isfinite(p.grad).all())
        and float(p.grad.detach().float().norm()) > 0.0
    ]
    not_updated = []
    for name, p in named:
        if name not in grad_nonzero or name not in before:
            continue
        after = p.detach().to(device='cpu')
        if torch.equal(before[name], after):
            not_updated.append(name)

    def _cap(names):
        return names[:int(max_names)]

    return {
        'step': int(step),
        'n_trainable_tensors': len(named),
        'n_optimizer_tensors': len(grouped),
        'n_not_in_optimizer': len(not_in_optimizer),
        'not_in_optimizer_names': _cap(not_in_optimizer),
        'n_duplicate_optimizer_tensors': len(duplicates),
        'duplicate_optimizer_names': _cap(duplicates),
        'n_grad_nonzero': len(grad_nonzero),
        'n_not_updated': len(not_updated),
        'not_updated_names': _cap(not_updated),
        'names_truncated_at': int(max_names),
        'ok': bool(not not_in_optimizer and not not_updated),
        'note': ('Coverage/update audit for the first optimizer step: a '
                 'trainable tensor missing from the optimizer, or one with '
                 'a non-zero gradient that did not move, makes the fixed-'
                 'batch gate fail even if another path lowers the loss.'),
    }


# --------------------------------------------------------------------------
# 1b. Gradient-norm health summary
# --------------------------------------------------------------------------
def _finite_safe_mean(values: list[float]) -> float:
    """Mean of finite floats without overflowing the intermediate sum.

    ``sum([1e308, 1e308]) / 2`` is ``inf`` although both the inputs and the
    true mean are finite. Keep the ordinary expression for every in-range
    sample; only fall back to a scaled summation when it overflows.
    """
    n = len(values)
    total = sum(values)
    if math.isfinite(total):
        return total / n
    scale = max(abs(x) for x in values)
    if scale == 0.0:
        return 0.0
    mean = scale * (math.fsum(x / scale for x in values) / n)
    if not math.isfinite(mean):
        raise ValueError(
            f'gradient-norm mean is not finite after scaled summation '
            f'(scale={scale!r})')
    return mean


def _finite_safe_median(ordered: list[float]) -> float:
    """Median whose even-sample midpoint cannot overflow to ``inf``."""
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    a, b = ordered[n // 2 - 1], ordered[n // 2]
    mid = (a + b) / 2.0
    if math.isfinite(mid):
        return mid
    mid = a / 2.0 + b / 2.0
    if not math.isfinite(mid):
        raise ValueError(
            'gradient-norm median is not finite after overflow-safe midpoint')
    return mid


def summarize_gradient_norms(grad_norms, *, clip_norm: float | None = None
                             ) -> dict | None:
    """Summarize the per-step PRE-CLIP gradient norm curve.

    The loss curve says whether the objective fell; this says whether the
    optimizer stayed healthy while it fell. A run whose raw norm sits far
    above the clip budget is being carried by clipping, and a norm that is
    still growing at the end is the first sign of an LR that is too large.
    ``None`` is returned when no norms were recorded, so a caller can keep
    the field absent/None for a disabled probe.

    All values are finite-only; non-finite norms are counted separately
    (``n_nonfinite``) rather than poisoning the summary with NaN.
    """
    vals = []
    for g in grad_norms:
        if g is None:
            continue
        number = _finite_float_or_none(g)
        # A malformed/oversized norm is counted like any other non-finite
        # value instead of raising from float().
        vals.append(float('nan') if number is None else number)
    if not vals:
        return None
    finite = [v for v in vals if math.isfinite(v)]
    out = {
        'n_steps': len(vals),
        'n_nonfinite': len(vals) - len(finite),
        'max': None,
        'mean': None,
        'median': None,
        'p95': None,
        'first': None,
        'last': None,
        'tail_mean': None,
        'clip_norm': None,
        'n_above_clip': None,
        'max_over_clip': None,
        'n_over_10x_clip': None,
        'notes': [],
    }
    if not finite:
        out['notes'].append(
            'every recorded gradient norm is non-finite: run '
            'diagnose_nonfinite() on the first failing step.')
        return out
    ordered = sorted(finite)
    n = len(ordered)
    out.update({
        'max': ordered[-1],
        'mean': _finite_safe_mean(finite),
        'median': _finite_safe_median(ordered),
        'p95': ordered[max(0, math.ceil(0.95 * n) - 1)],
        'first': finite[0],
        'last': finite[-1],
    })
    tail_k = max(1, min(10, math.ceil(0.1 * n)))
    out['tail_mean'] = _finite_safe_mean(finite[-tail_k:])
    if clip_norm is not None:
        try:
            clip = float(clip_norm)
        except (TypeError, ValueError):
            clip = None
        if clip is not None and math.isfinite(clip) and clip > 0:
            out['clip_norm'] = clip
            out['n_above_clip'] = sum(1 for v in finite if v > clip)
            out['max_over_clip'] = out['max'] / clip
            out['n_over_10x_clip'] = sum(1 for v in finite if v > 10.0 * clip)
            if out['max_over_clip'] > 10.0:
                out['notes'].append(
                    f"peak gradient norm is {out['max_over_clip']:.1f}x the "
                    f'clip budget: clipping is masking an explosion, not '
                    f'just stabilizing a healthy step.')
            elif out['n_above_clip'] == n:
                out['notes'].append(
                    'every step exceeded the clip budget: lower the learning '
                    'rate or inspect the loss for large early gradients.')
    if out['first'] > 0 and out['tail_mean'] > 2.0 * out['first']:
        out['notes'].append(
            f"gradient norms grew from {out['first']:.3g} to a tail mean of "
            f"{out['tail_mean']:.3g}: the optimization is diverging even if "
            f'the loss has not caught up yet.')
    return out


# --------------------------------------------------------------------------
# 2. Held-out validation curve
# --------------------------------------------------------------------------
def accuracy_regime(val_accs: Sequence[float],
                    train_accs: Sequence[float] | None = None, *,
                    ceiling: float | None = None) -> dict | None:
    """Summarise the held-out ACCURACY curve, which loss shape cannot.

    The loss verdict can call a curve "overfitting" when validation loss
    flattens, but on a bounded metric a flat validation loss while held-out
    accuracy sits at the task ceiling is *convergence*, not a generalization
    failure. This block records the head/tail means, peak accuracy and (when
    ``ceiling`` is supplied) whether the tail is pinned at it, so a reader
    can tell the two apart. It deliberately does not override the loss
    verdict; it is additive evidence.

    Returns ``None`` when there are fewer than two points (no shape).
    """
    if val_accs is None or len(val_accs) < 2:
        return None
    vals = [float(v) for v in val_accs]
    head_k = max(1, len(vals) // 5)
    val_head = sum(vals[:head_k]) / head_k
    val_tail = sum(vals[-head_k:]) / head_k
    out = {
        'val_acc_head_mean': val_head,
        'val_acc_tail_mean': val_tail,
        'val_acc_peak': max(vals),
        'val_acc_best_step': int(max(range(len(vals)),
                                     key=lambda i: vals[i])),
    }
    if ceiling is not None:
        out['ceiling'] = float(ceiling)
        out['saturated'] = bool(
            val_tail >= float(ceiling) * (1.0 - 1e-3))
    else:
        # No declared ceiling: say so honestly instead of guessing "not
        # saturated" from an arbitrary threshold.
        out['ceiling'] = None
        out['saturated'] = None
    if train_accs:
        train = [float(v) for v in train_accs]
        t_head_k = max(1, len(train) // 5)
        out['train_acc_head_mean'] = sum(train[:t_head_k]) / t_head_k
        out['train_acc_tail_mean'] = sum(train[-t_head_k:]) / t_head_k
        out['train_acc_still_rising'] = bool(
            out['train_acc_tail_mean'] > out['train_acc_head_mean'])
    return out


class ValPoint(NamedTuple):
    """One held-out measurement.

    A ``NamedTuple`` rather than a bare ``(loss, acc)`` tuple: the two
    fields are ``float``-identical in type and order-swappable, and a swap
    is *silent* — it turns "validation accuracy 0.11" into "validation
    accuracy 2.97" and then reports the model as saturated at the ceiling.
    Naming the fields makes that mistake impossible to express.
    """

    loss: float
    acc: float


class ValidationCurve:
    """Record a held-out curve and turn it into a training diagnosis.

    ``evaluate`` is called as ``evaluate(step) -> ValPoint(loss, acc)``
    with the modules already in ``eval()`` mode; the monitor restores the
    previous mode itself. Points are taken at step 0, every ``val_every``
    steps, and once more at the final step (so a budget that is not a
    multiple of ``val_every`` still ends on a real measurement rather than
    on a stale one).

    The curve is **read-only with respect to the optimization**: it never
    changes the loss, the optimizer or the step count, so turning it on
    cannot change what the trainer reports. That is deliberate — the
    headline ``final_acc`` stays the fixed-budget endpoint and every
    existing statistic keeps its meaning, while the diagnosis says whether
    that endpoint was the right place to stop reading the curve.
    """

    def __init__(self, evaluate: Callable[[int], ValPoint], *, steps: int,
                 val_every: int = 0, label: str = '',
                 ceiling: float | None = None):
        self._evaluate = evaluate
        self.steps = int(steps)
        self.val_every = max(0, int(val_every or 0))
        self.label = str(label)
        self.ceiling = ceiling
        self.history: list[dict] = []
        self.best: dict | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.val_every > 0

    def record(self, step: int) -> dict | None:
        """Evaluate and append one held-out point (no-op when disabled)."""
        if not self.enabled:
            return None
        step = int(step)
        if self.history and step <= self.history[-1]['step']:
            return self.history[-1]
        point_in = self._evaluate(step)
        if not isinstance(point_in, ValPoint):
            raise TypeError(
                f'ValidationCurve.evaluate must return a ValPoint(loss, acc); '
                f'got {type(point_in).__name__}. Build it as '
                f'ValPoint(loss=..., acc=...) so the two fields cannot be '
                f'silently swapped.')
        loss, acc = float(point_in.loss), float(point_in.acc)
        # Accuracy is a proportion. A value outside [0, 1] means the fields
        # arrived swapped (or the evaluator is broken), and every downstream
        # verdict would be wrong in a way that still looks plausible.
        if not (0.0 <= acc <= 1.0):
            raise ValueError(
                f'validation accuracy must lie in [0, 1]; got {acc} '
                f'(loss={loss}). Either the (loss, acc) pair is swapped or '
                f'the evaluator is returning a loss in the accuracy slot.')
        loss_finite = math.isfinite(loss)
        # Persist a non-finite checkpoint as JSON null (the same sentinel
        # ``diagnose_training_curves`` already understands) instead of NaN/
        # Infinity, which would make the whole artifact fail strict JSON
        # serialization long after the run recovered. ``loss_finite`` keeps
        # the distinction explicit.
        point = {'step': step, 'loss': loss if loss_finite else None,
                 'acc': acc, 'loss_finite': bool(loss_finite)}
        self.history.append(point)
        # Track the validation minimum by LOSS, which is the quantity the
        # early-stopping rule would use; accuracy is kept alongside so the
        # diagnosis can distinguish "val loss fell" from "val accuracy rose"
        # (they disagree when the probe saturates).
        #
        # Only FINITE losses may win the best slot. A NaN compares False
        # against everything, so the old ``best is None or loss < best``
        # rule kept a NaN first checkpoint forever: every later finite
        # checkpoint lost to it, and the reported best/regret/early-stopping
        # summary became NaN even though the run recovered.
        if loss_finite and (self.best is None or loss < self.best['loss']):
            self.best = dict(point)
        return point

    def close(self, actual_steps: int) -> None:
        """Make sure the curve ends on the state that was actually trained."""
        if self._closed or not self.enabled:
            self._closed = True
            return
        actual_steps = int(actual_steps)
        if not self.history or self.history[-1]['step'] != actual_steps:
            self.record(actual_steps)
        self._closed = True

    def _accuracy_regime(self, train_accs: Sequence[float]) -> dict | None:
        """Name the accuracy-level regime, which a loss diagnosis cannot.

        ``saturated`` is the case that a final-number-only reading gets
        wrong most often: training accuracy is still climbing while
        held-out accuracy is pinned at the task ceiling. The recipe is not
        overfitting and not underfitting — it is *done*, and additional
        steps buy nothing, so a claim of "more training would help" would be
        unsupported.
        """
        if self.ceiling is None or len(self.history) < 2:
            return None
        out = accuracy_regime([h['acc'] for h in self.history],
                              train_accs, ceiling=self.ceiling)
        if out is not None:
            # Report the STEP of the peak (the module-level helper works on
            # bare sequences and therefore only knows the index).
            out['val_acc_best_step'] = self.history[
                out['val_acc_best_step']]['step']
        return out

    def finalize(self, train_losses: Sequence[float],
                 train_accs: Sequence[float] | None = None) -> dict:
        """Return the JSON-serializable monitoring block for one row.

        Keys mirror ``run_quality.train_one``'s ``analysis_fields`` so a
        consumer can read Exp 4 / Exp 5 / Exp 8 artifacts with one code
        path: ``val_history``, ``curve_diagnosis``,
        ``generalization_gap``, plus the best-validation point and the
        accuracy regime.
        """
        self.close(self.steps)
        val_losses = [h['loss'] for h in self.history]
        diagnosis = diagnose_training_curves(
            train_losses, val_losses or None)
        def _point_finite(h):
            flag = h.get('loss_finite')
            if flag is not None:
                return bool(flag)
            return h.get('loss') is not None and math.isfinite(
                float(h['loss']))

        finite = [h for h in self.history if _point_finite(h)]
        final = self.history[-1] if self.history else None
        final_finite = bool(final and _point_finite(final))
        last_finite = finite[-1] if finite else None
        # Same definition run_quality uses, so the two are comparable:
        # final held-out loss minus the mean of the last 10 training steps.
        # A non-finite endpoint has no meaningful gap, so report None
        # instead of propagating NaN into every downstream mean.
        gap = None
        if final_finite and train_losses:
            # ``train_losses`` may carry the same JSON-null sentinel for a
            # sanitized non-finite step that ``val_history`` does. Average
            # the finite entries of the last 10 steps; if none are finite
            # the gap is undefined (None), never a TypeError or NaN.
            tail = []
            for value in train_losses[-10:]:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    tail.append(value)
            if tail:
                # Use the overflow-safe mean: ten finite 1e308 losses sum
                # to inf, which made the gap -inf even though every input
                # (and the true gap) is finite. An unrepresentable true gap
                # is reported as None, not +/-inf.
                gap = float(final['loss']) - _finite_safe_mean(tail)
                if not math.isfinite(gap):
                    gap = None
        block = {
            'val_every': self.val_every,
            'val_history': self.history,
            'curve_diagnosis': diagnosis,
            # Convenience copies of the two budget-status fields a reader
            # needs most often; the full evidence stays in the diagnosis.
            'budget_limited': bool(diagnosis.get('budget_limited')),
            'val_still_improving_at_end': bool(
                (diagnosis.get('signals') or {}).get(
                    'val_still_improving_at_end')),
            'generalization_gap': gap,
            'best_val': self.best,
            'val_loss_first': val_losses[0] if val_losses else None,
            'val_loss_last': val_losses[-1] if val_losses else None,
            'val_loss_last_finite': (last_finite['loss']
                                     if last_finite else None),
            'val_acc_first': self.history[0]['acc'] if self.history else None,
            'val_acc_last': self.history[-1]['acc'] if self.history else None,
            'val_acc_peak': (max(h['acc'] for h in self.history)
                             if self.history else None),
            'n_nonfinite_val_points': len(self.history) - len(finite),
            'nonfinite_val_steps': [int(h['step']) for h in self.history
                                    if not _point_finite(h)],
        }
        # The overfitting verdict says "use early stopping"; these fields
        # quantify what skipping it actually cost, so the recommendation is
        # backed by a number from THIS run instead of a generic rule. They
        # are derived only from the recorded history (no extra evaluation)
        # and are None when the curve is empty/disabled OR the final
        # checkpoint is non-finite (the run diverged before the endpoint).
        if final is not None and self.best is not None:
            block['final_val'] = dict(final)
            block['val_acc_regret_from_best'] = (
                max(h['acc'] for h in self.history) - final['acc'])
            block['steps_since_best'] = int(final['step']
                                            - self.best['step'])
            block['best_val_is_final'] = (
                bool(final_finite and final['step'] == self.best['step']))
            if final_finite:
                # The final-minus-best loss can exceed the float range even
                # though both endpoints are finite (e.g. best=-1e308,
                # final=+1e308). A non-representable regret is reported as
                # None, not inf, so the record stays strict-JSON.
                loss_regret = final['loss'] - self.best['loss']
                loss_regret = (float(loss_regret)
                               if math.isfinite(loss_regret) else None)
                block['val_loss_regret_from_best'] = loss_regret
                block['early_stopping_summary'] = {
                    'selected_step': int(self.best['step']),
                    'final_step': int(final['step']),
                    'steps_saved': int(final['step'] - self.best['step']),
                    'val_loss_at_selection': float(self.best['loss']),
                    'val_loss_at_final': float(final['loss']),
                    'val_loss_regret': loss_regret,
                    'val_acc_at_selection': float(self.best['acc']),
                    'val_acc_at_final': float(final['acc']),
                    'val_acc_regret': float(
                        max(h['acc'] for h in self.history)
                        - final['acc']),
                }
            else:
                # Keep the schema stable while making the missing arithmetic
                # explicit: no finite endpoint, no regret.
                block['val_loss_regret_from_best'] = None
                block['early_stopping_summary'] = None
                block['early_stopping_unavailable_reason'] = (
                    'the final validation checkpoint is non-finite; the '
                    'loss regret against the best finite checkpoint is '
                    'undefined. Diagnose the eval pass (data -> log(0)/'
                    'div_by_zero -> softmax overflow -> AMP -> grads/LR).')
        regime = self._accuracy_regime(
            train_accs or [])
        if regime is not None:
            block['accuracy_regime'] = regime
        return block

    def log_verdict(self, block: Mapping) -> None:
        """One compact line so the verdict is visible without opening JSON."""
        if not self.enabled:
            return
        verdict = (block.get('curve_diagnosis') or {}).get('verdict', '?')
        regime = block.get('accuracy_regime') or {}
        suffix = (' saturated_at_ceiling' if regime.get('saturated') else '')
        if block.get('budget_limited'):
            suffix += ' still_improving_at_budget'
        best = block.get('best_val') or {}
        regret = block.get('val_loss_regret_from_best')
        logger.info(
            f'  [curve] {self.label}: {verdict}{suffix} '
            f'({len(self.history)} val points, best val loss '
            f"{best.get('loss')} @ step {best.get('step')}"
            + (f", final-vs-best regret {regret:+.4f}" if regret is not None
               else '') + ')')


# --------------------------------------------------------------------------
# 3. Accuracy-vs-cost acceptance
# --------------------------------------------------------------------------
#: Cost keys compared by default. These are the axes an ablation changes:
#: parameter count, analytic/measured compute, wall-clock training cost,
#: per-sample inference latency through the operator, per-sample evaluation
#: cost, and measured peak memory. Missing axes are skipped (the row simply
#: carries no measurement), so a caller that only knows parameters is not
#: penalised for the others; but when both sides measured memory or eval
#: latency, a 2x increase is enough to reject a sub-noise accuracy gain.
DEFAULT_COST_KEYS = ('n_params', 'flops', 'mean_train_time_s',
                     'mean_fwd_ms', 'mean_eval_ms_per_sample',
                     'peak_memory_bytes')

#: Smallest accuracy difference worth claiming (0.5 percentage points).
#: Below this a "win" is indistinguishable from seed noise at the 5-7 seed
#: scale these experiments run at, so claiming it spends credibility for
#: nothing.
DEFAULT_MIN_EFFECT = 0.005

#: Largest multiple of any single cost axis that a claimed win may cost.
#: A 0.5-point accuracy gain that doubles parameters, training time or
#: inference latency is a *different* operating point, not a better one.
DEFAULT_MAX_COST_RATIO = 2.0


def accuracy_cost_tradeoff(candidate: Mapping, baseline: Mapping, *,
                           cost_keys: Sequence[str] = DEFAULT_COST_KEYS,
                           min_effect: float = DEFAULT_MIN_EFFECT,
                           max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
                           significant: bool | None = None,
                           paired_pvalue: float | None = None) -> dict:
    """Decide whether ``candidate``'s gain justifies what it costs.

    A metric that goes up is not automatically an improvement. This applies
    the three screens that a final-number-only comparison skips, in order:

    1. **Effect size** — a gain smaller than ``min_effect`` (default 0.005)
       is below the seed-to-seed noise these budgets produce, so it is
       reported as ``below_min_effect`` rather than as a win.
    2. **Significance** — when the caller supplies ``significant`` (from
       the seed-paired test it already ran) or ``paired_pvalue``, a gain
       that the data cannot separate from zero is ``not_significant``.
       ``None`` means untested, which is reported honestly instead of
       defaulting to "significant".
    3. **Cost proportionality** — the ratio of every cost axis is computed.
       If any exceeds ``max_cost_ratio`` the verdict is ``cost_prohibitive``:
       the accuracy gain is real but the configuration is not a better
       operating point.

    Verdicts: ``indeterminate`` (no comparable accuracy), ``no_gain``,
    ``below_min_effect``, ``not_significant``, ``cost_prohibitive``,
    ``warranted``.
    """
    acc = candidate.get('mean_acc', candidate.get('final_acc'))
    base = baseline.get('mean_acc', baseline.get('final_acc'))
    min_effect_value = _finite_float_or_none(min_effect)
    max_cost_value = _finite_float_or_none(max_cost_ratio)
    if min_effect_value is None or max_cost_value is None:
        raise ValueError('min_effect and max_cost_ratio must be finite '
                         f'numbers, got {min_effect!r}, {max_cost_ratio!r}')
    paired = _finite_float_or_none(paired_pvalue)
    report: dict = {
        'verdict': 'indeterminate',
        'accuracy_delta': None,
        'relative_accuracy_gain': None,
        'cost_ratios': {},
        'worst_cost_ratio': None,
        'measured_cost_keys': [],
        'missing_cost_keys': [],
        'min_effect': min_effect_value,
        'max_cost_ratio': max_cost_value,
        'significant': significant,
        'paired_pvalue': paired,
        'cost_keys': list(cost_keys),
        'recommendation': '',
    }
    if acc is None or base is None:
        candidate_acc = baseline_acc = None
    else:
        candidate_acc = _finite_float_or_none(acc)
        baseline_acc = _finite_float_or_none(base)
    if candidate_acc is None or baseline_acc is None:
        report['recommendation'] = (
            'no comparable accuracy recorded for both sides; refusing to '
            'call this an improvement.')
        return report

    delta = candidate_acc - baseline_acc
    if not math.isfinite(delta):
        # Opposite-sign extremes can overflow the subtraction even though
        # both endpoints are finite; the difference is not representable.
        report['recommendation'] = (
            'the accuracy difference is not representable as a finite '
            'float; refusing to call this an improvement.')
        return report
    report['accuracy_delta'] = delta
    relative = (delta / baseline_acc) if baseline_acc else None
    report['relative_accuracy_gain'] = (
        relative if relative is not None and math.isfinite(relative)
        else None)

    ratios = {}
    missing = []
    for key in cost_keys:
        c = _finite_float_or_none(candidate.get(key))
        b = _finite_float_or_none(baseline.get(key))
        if c is None or b is None:
            missing.append(key)
            continue
        if not (math.isfinite(c) and math.isfinite(b)):
            missing.append(key)
            continue
        # A zero baseline cost is uninformative (division by zero), not
        # "infinitely cheaper"; skip rather than invent a ratio.
        if b <= 0.0:
            missing.append(key)
            continue
        ratio = c / b
        if not math.isfinite(ratio):
            # The true ratio exceeds the float range; it is not a usable
            # JSON metric, so record the axis as missing.
            missing.append(key)
            continue
        ratios[key] = ratio
    report['cost_ratios'] = ratios
    report['measured_cost_keys'] = list(ratios)
    report['missing_cost_keys'] = list(missing)
    if ratios:
        worst_key = max(ratios, key=lambda k: ratios[k])
        report['worst_cost_ratio'] = {'cost': worst_key,
                                      'ratio': ratios[worst_key]}
    else:
        report['cost_ratios_status'] = (
            'no comparable cost measured; the gain is reported without a '
            'cost verdict')

    cost_clause = ''
    if ratios:
        cost_clause = (
            f" Cost ratios: "
            + ', '.join(f'{k}={v:.2f}x' for k, v in ratios.items()) + '.')

    if delta <= 0.0:
        report['verdict'] = 'no_gain'
        report['recommendation'] = (
            f'accuracy did not improve ({delta:+.4f}); this configuration '
            f'should not be adopted over the baseline.')
        return report

    if delta < min_effect_value:
        report['verdict'] = 'below_min_effect'
        report['recommendation'] = (
            f'gain of {delta:+.4f} is below the pre-specified minimum '
            f'effect {min_effect:.4f}: at this seed count it is not '
            f'separable from noise, so report it as "no measurable '
            f'difference" rather than as a win.')
        return report

    if significant is False or (
            paired is not None and not (paired < 0.05)):
        report['verdict'] = 'not_significant'
        report['recommendation'] = (
            f'gain of {delta:+.4f} clears the minimum effect but is not '
            f'separable from seed noise'
            + (f' (paired p={paired:.4g})' if paired is not None
               else '')
            + '. Report as inconclusive, not as an improvement.')
        return report

    if ratios and report['worst_cost_ratio']['ratio'] > max_cost_value:
        worst = report['worst_cost_ratio']
        report['verdict'] = 'cost_prohibitive'
        report['recommendation'] = (
            f'gain of {delta:+.4f} is real, but it costs '
            f"{worst['ratio']:.2f}x {worst['cost']}, above the "
            f'{max_cost_ratio:.2f}x budget: this is a different operating '
            f'point, not a better one. Report the trade-off explicitly '
            f'rather than quoting the accuracy gain alone.')
        return report

    report['verdict'] = 'warranted'
    coverage_clause = ''
    if missing:
        coverage_clause = (
            f" Cost axes not compared (missing or uninformative): "
            f"{', '.join(missing)}; the 'warranted' verdict covers only the "
            f'measured axes.')
    report['recommendation'] = (
        f'gain of {delta:+.4f} clears the minimum effect'
        + ('' if significant is not None else
           ' but was NOT tested for significance (paired_pvalue/significant '
           'were not supplied), so it is still a candidate, not a claim')
        + f' and stays within the cost budget.{cost_clause}{coverage_clause}')
    return report


def log_tradeoff(verdict: Mapping, *, label: str = '') -> None:
    """Surface a :func:`accuracy_cost_tradeoff` verdict in the log."""
    if not verdict:
        return
    level = (logger.warning if verdict.get('verdict') in
             ('cost_prohibitive', 'not_significant', 'below_min_effect',
              'no_gain') else logger.info)
    prefix = f'  [cost/claim] {label}: ' if label else '  [cost/claim] '
    level(f'{prefix}{verdict["verdict"]}: {verdict.get("recommendation", "")}')


# --------------------------------------------------------------------------
# 4. Row-level pooling of the per-seed bad-case reports
# --------------------------------------------------------------------------
def pool_error_analyses(per_seed: Sequence[Mapping], *,
                        accuracy_key: str = 'final_acc') -> dict | None:
    """Pool per-seed bad-case reports into one row-level aggregate.

    ``run_quality.analyze_errors`` writes a per-sample classification into
    every per-seed record; a multi-seed row without this rollup hides which
    mistake DOMINATES the configuration behind a list of per-seed dicts.
    The pooled category counts answer "is this layout failing by retrieving
    a different key's value, by echoing the key, or by copying a
    distractor?", which is the raw material for the next architecture idea
    (see the brief's bad-case step).

    Counts add up over every analyzed sample; the EXAMPLES come from the
    worst seed (lowest accuracy) so the artifact shows the most informative
    failures rather than a lucky seed's. Returns ``None`` when no per-seed
    record carries an ``error_analysis`` block, so a disabled bad-case pass
    stays ``None`` instead of fabricating an empty report.
    """
    rows = [r for r in per_seed if r.get('error_analysis')]
    if not rows:
        return None
    pooled: dict = {}
    conf_acc: dict = {}
    n_samples = n_wrong = 0
    for r in rows:
        ea = r['error_analysis']
        n_samples += int(ea.get('n_samples', 0) or 0)
        n_wrong += int(ea.get('n_wrong', 0) or 0)
        for c, v in (ea.get('categories') or {}).items():
            acc = pooled.setdefault(
                c, {'count': 0, 'definition': v.get('definition')})
            acc['count'] += int(v.get('count', 0) or 0)
            # Confidence is a nested block; combine it with sample-count
            # weights (means cannot simply be averaged across seeds with
            # different numbers of errors).
            conf = v.get('confidence') or {}
            n_scored = int(conf.get('n_scored', 0) or 0)
            if n_scored:
                ca = conf_acc.setdefault(c, {
                    'n_scored': 0,
                    'n_confident_wrong': 0,
                    'sum_pred_prob': 0.0,
                    'sum_target_prob': 0.0,
                    'sum_margin': 0.0,
                    'sum_entropy_ratio': 0.0,
                })
                ca['n_scored'] += n_scored
                ca['n_confident_wrong'] += int(
                    conf.get('n_confident_wrong', 0) or 0)
                ca['sum_pred_prob'] += (
                    (_finite_float_or_none(conf.get('mean_pred_prob'))
                     or 0.0) * n_scored)
                ca['sum_target_prob'] += (
                    (_finite_float_or_none(conf.get('mean_target_prob'))
                     or 0.0) * n_scored)
                ca['sum_margin'] += (
                    (_finite_float_or_none(conf.get('mean_margin'))
                     or 0.0) * n_scored)
                ca['sum_entropy_ratio'] += (
                    (_finite_float_or_none(conf.get('mean_entropy_ratio'))
                     or 0.0) * n_scored)
    for v in pooled.values():
        v['rate_of_wrong'] = (v['count'] / n_wrong) if n_wrong else 0.0
        v['rate_of_all'] = (v['count'] / n_samples) if n_samples else 0.0
    for c, ca in conf_acc.items():
        n = ca['n_scored']
        pooled[c]['confidence'] = {
            'n_scored': int(n),
            'n_confident_wrong': int(ca['n_confident_wrong']),
            'rate_confident_wrong': ca['n_confident_wrong'] / n,
            'mean_pred_prob': ca['sum_pred_prob'] / n,
            'mean_target_prob': ca['sum_target_prob'] / n,
            'mean_margin': ca['sum_margin'] / n,
            'mean_entropy_ratio': ca['sum_entropy_ratio'] / n,
        }
    n_scored_all = sum(ca['n_scored'] for ca in conf_acc.values())
    confidence_summary = None
    if n_scored_all:
        n_conf = sum(ca['n_confident_wrong'] for ca in conf_acc.values())
        confidence_summary = {
            'n_scored_wrong': int(n_scored_all),
            'n_confident_wrong': int(n_conf),
            'rate_confident_wrong': n_conf / n_scored_all,
            'mean_pred_prob': (sum(ca['sum_pred_prob']
                                   for ca in conf_acc.values())
                               / n_scored_all),
            'mean_target_prob': (sum(ca['sum_target_prob']
                                     for ca in conf_acc.values())
                                 / n_scored_all),
            'mean_margin': (sum(ca['sum_margin']
                                for ca in conf_acc.values()) / n_scored_all),
            'note': ('Pooled confidence over every analyzed wrong sample: '
                     'high values mean the failures are confident blind '
                     'spots rather than ambiguous near-ties.'),
        }

    def _acc(r):
        value = _finite_float_or_none(r.get(accuracy_key))
        if value is None:
            return float('inf')
        return value

    worst = min(rows, key=_acc)
    worst_ea = worst['error_analysis']
    return {
        'n_seeds': len(rows),
        'n_samples': n_samples,
        'n_wrong': n_wrong,
        'error_rate': (n_wrong / n_samples) if n_samples else None,
        'pooled_categories': pooled,
        'confidence': confidence_summary,
        'representative_seed': worst.get('seed'),
        'representative_examples': worst_ea.get('examples'),
        'accuracy_by_distance_bin': worst_ea.get('accuracy_by_distance_bin'),
        'note': ('Pooled over seeds; examples come from the worst seed '
                 '(lowest accuracy) so the artifact shows the most '
                 'informative failures. See the per-seed error_analysis for '
                 'each seed.'),
    }


def log_pooled_errors(pooled: Mapping | None, *, label: str = '') -> None:
    """One compact log line for a pooled bad-case report."""
    if not pooled or not pooled.get('n_samples'):
        return
    counts = {c: v['count'] for c, v in
              (pooled.get('pooled_categories') or {}).items() if v['count']}
    logger.info(
        f'  [bad-case] {label}: {pooled["n_wrong"]}/{pooled["n_samples"]} '
        f"wrong ({100.0 * (pooled.get('error_rate') or 0.0):.1f}%); "
        f'categories={counts}')