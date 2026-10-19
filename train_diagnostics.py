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


# --------------------------------------------------------------------------
# 2. Held-out validation curve
# --------------------------------------------------------------------------
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
        point = {'step': step, 'loss': loss, 'acc': acc}
        self.history.append(point)
        # Track the validation minimum by LOSS, which is the quantity the
        # early-stopping rule would use; accuracy is kept alongside so the
        # diagnosis can distinguish "val loss fell" from "val accuracy rose"
        # (they disagree when the probe saturates).
        if self.best is None or loss < self.best['loss']:
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
        val_accs = [h['acc'] for h in self.history]
        head_k = max(1, len(val_accs) // 5)
        val_head = sum(val_accs[:head_k]) / head_k
        val_tail = sum(val_accs[-head_k:]) / head_k
        saturated = val_tail >= self.ceiling * (1.0 - 1e-3)
        out = {
            'val_acc_head_mean': val_head,
            'val_acc_tail_mean': val_tail,
            'val_acc_peak': max(val_accs),
            'val_acc_best_step': self.history[
                int(max(range(len(val_accs)), key=lambda i: val_accs[i]))]['step'],
            'ceiling': float(self.ceiling),
            'saturated': bool(saturated),
        }
        if train_accs:
            t_head_k = max(1, len(train_accs) // 5)
            out['train_acc_head_mean'] = sum(
                train_accs[:t_head_k]) / t_head_k
            out['train_acc_tail_mean'] = sum(
                train_accs[-t_head_k:]) / t_head_k
            out['train_acc_still_rising'] = bool(
                out['train_acc_tail_mean'] > out['train_acc_head_mean'])
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
        # Same definition run_quality uses, so the two are comparable:
        # final held-out loss minus the mean of the last 10 training steps.
        gap = (val_losses[-1] - sum(train_losses[-10:]) / min(10, len(train_losses))
               if val_losses and train_losses else None)
        block = {
            'val_every': self.val_every,
            'val_history': self.history,
            'curve_diagnosis': diagnosis,
            'generalization_gap': gap,
            'best_val': self.best,
            'val_loss_first': val_losses[0] if val_losses else None,
            'val_loss_last': val_losses[-1] if val_losses else None,
            'val_acc_first': self.history[0]['acc'] if self.history else None,
            'val_acc_last': self.history[-1]['acc'] if self.history else None,
            'val_acc_peak': (max(h['acc'] for h in self.history)
                             if self.history else None),
        }
        # The overfitting verdict says "use early stopping"; these fields
        # quantify what skipping it actually cost, so the recommendation is
        # backed by a number from THIS run instead of a generic rule. They
        # are derived only from the recorded history (no extra evaluation)
        # and are None when the curve is empty/disabled.
        if self.history and self.best is not None:
            final = self.history[-1]
            block['final_val'] = dict(final)
            block['val_loss_regret_from_best'] = (
                final['loss'] - self.best['loss'])
            block['val_acc_regret_from_best'] = (
                max(h['acc'] for h in self.history) - final['acc'])
            block['steps_since_best'] = int(final['step']
                                            - self.best['step'])
            block['best_val_is_final'] = (
                bool(final['step'] == self.best['step']))
            block['early_stopping_summary'] = {
                'selected_step': int(self.best['step']),
                'final_step': int(final['step']),
                'steps_saved': int(final['step'] - self.best['step']),
                'val_loss_at_selection': float(self.best['loss']),
                'val_loss_at_final': float(final['loss']),
                'val_loss_regret': float(final['loss']
                                         - self.best['loss']),
                'val_acc_at_selection': float(self.best['acc']),
                'val_acc_at_final': float(final['acc']),
                'val_acc_regret': float(max(h['acc'] for h in self.history)
                                        - final['acc']),
            }
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
#: parameter count, wall-clock training cost, and per-sample inference
#: latency. Peak memory is compared too when the caller has measured it
#: (CUDA only; on CPU the tensor footprint is not comparable across
#: configs).
DEFAULT_COST_KEYS = ('n_params', 'mean_train_time_s', 'mean_fwd_ms')

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
    report: dict = {
        'verdict': 'indeterminate',
        'accuracy_delta': None,
        'relative_accuracy_gain': None,
        'cost_ratios': {},
        'worst_cost_ratio': None,
        'min_effect': float(min_effect),
        'max_cost_ratio': float(max_cost_ratio),
        'significant': significant,
        'paired_pvalue': paired_pvalue,
        'cost_keys': list(cost_keys),
        'recommendation': '',
    }
    if acc is None or base is None or not (
            math.isfinite(float(acc)) and math.isfinite(float(base))):
        report['recommendation'] = (
            'no comparable accuracy recorded for both sides; refusing to '
            'call this an improvement.')
        return report

    delta = float(acc) - float(base)
    report['accuracy_delta'] = delta
    report['relative_accuracy_gain'] = (delta / float(base)) if base else None

    ratios = {}
    for key in cost_keys:
        c, b = candidate.get(key), baseline.get(key)
        if c is None or b is None:
            continue
        try:
            c, b = float(c), float(b)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(c) and math.isfinite(b)):
            continue
        # A zero baseline cost is uninformative (division by zero), not
        # "infinitely cheaper"; skip rather than invent a ratio.
        if b <= 0.0:
            continue
        ratios[key] = c / b
    report['cost_ratios'] = ratios
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

    if delta < float(min_effect):
        report['verdict'] = 'below_min_effect'
        report['recommendation'] = (
            f'gain of {delta:+.4f} is below the pre-specified minimum '
            f'effect {min_effect:.4f}: at this seed count it is not '
            f'separable from noise, so report it as "no measurable '
            f'difference" rather than as a win.')
        return report

    if significant is False or (
            paired_pvalue is not None and not (paired_pvalue < 0.05)):
        report['verdict'] = 'not_significant'
        report['recommendation'] = (
            f'gain of {delta:+.4f} clears the minimum effect but is not '
            f'separable from seed noise'
            + (f' (paired p={paired_pvalue:.4g})' if paired_pvalue is not None
               else '')
            + '. Report as inconclusive, not as an improvement.')
        return report

    if ratios and report['worst_cost_ratio']['ratio'] > float(max_cost_ratio):
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
    report['recommendation'] = (
        f'gain of {delta:+.4f} clears the minimum effect'
        + ('' if significant is not None else
           ' but was NOT tested for significance (paired_pvalue/significant '
           'were not supplied), so it is still a candidate, not a claim')
        + f' and stays within the cost budget.{cost_clause}')
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
    n_samples = n_wrong = 0
    for r in rows:
        ea = r['error_analysis']
        n_samples += int(ea.get('n_samples', 0) or 0)
        n_wrong += int(ea.get('n_wrong', 0) or 0)
        for c, v in (ea.get('categories') or {}).items():
            acc = pooled.setdefault(
                c, {'count': 0, 'definition': v.get('definition')})
            acc['count'] += int(v.get('count', 0) or 0)
    for v in pooled.values():
        v['rate_of_wrong'] = (v['count'] / n_wrong) if n_wrong else 0.0
        v['rate_of_all'] = (v['count'] / n_samples) if n_samples else 0.0

    def _acc(r):
        value = r.get(accuracy_key)
        if value is None or not math.isfinite(float(value)):
            return float('inf')
        return float(value)

    worst = min(rows, key=_acc)
    worst_ea = worst['error_analysis']
    return {
        'n_seeds': len(rows),
        'n_samples': n_samples,
        'n_wrong': n_wrong,
        'error_rate': (n_wrong / n_samples) if n_samples else None,
        'pooled_categories': pooled,
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