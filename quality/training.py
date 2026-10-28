"""run_quality split module: training."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403

from .errors import (  # noqa: F401
    analyze_errors,
)

from .models import (  # noqa: F401
    _MEMORYLESS_OPS,
    _build_task_model,
    _eval_model,
)

from .stats import (  # noqa: F401
    _is_finite_real,
    _t_crit_975,
)

from .tasks import (  # noqa: F401
    SYNTHETIC_TASKS,
)

from .utils import (  # noqa: F401
    _build_param_groups,
)


def _public_api(name, fallback):
    """Resolve a runner-level symbol so monkeypatching ``run_quality`` works.

    The historical single-module implementation let tests patch
    ``run_quality.train_one`` / ``run_quality.train_multi_seed`` and have
    the internal call sites see the patch.  After the split those call
    sites live here, so resolve through the public module at call time.
    """
    import sys

    mod = sys.modules.get('run_quality')
    if mod is not None:
        return getattr(mod, name, fallback)
    return fallback


def train_one(op_name, d_model=32, seq_len=16, n_kv=1, vocab=16,
              steps=100, lr=3e-3, seed=42, device='cpu',
              softmax_steps=None, eval_batches=4, eval_batch=64,
              train_batch=None, task='mqar',
              val_every=0, val_batches=2, val_batch=32,
              error_batches=0, error_batch=64, max_error_examples=2):
    """Train a single operator on a synthetic task for ``steps`` steps.

    ``task`` selects the batch generator from ``SYNTHETIC_TASKS``
    (default ``'mqar'``: the historical Multi-Query Associative Recall
    probe; see the registry above for the update / long-range /
    multi-aggregation probes).

    ``softmax_steps`` lets the softmax baseline train longer to reach
    convergence (the original 100 steps left softmax at ~10%, barely above the
    6.25% chance — a meaningless upper bound).

    ``train_batch`` overrides the per-step training batch size (default 32,
    overridable for memory-constrained or GPU runs via the
    ``MQAR_TRAIN_BATCH`` env var).

    Results-analysis hooks (all OFF by default through the zero/None
    defaults, so direct callers get the historical behaviour bit-for-bit):

      * ``val_every > 0`` records a held-out validation loss/accuracy curve
        every ``val_every`` steps (plus at step 0), drawn from the dedicated
        ``_VAL_MONITOR_SEED_OFFSET`` stream — a DIFFERENT stream from the
        headline eval, so monitoring cannot leak into the reported metric.
        The train-vs-val curves feed ``diagnose_training_curves``.
      * ``error_batches > 0`` runs a bad-case analysis on
        ``error_batches`` x ``error_batch`` held-out samples from the
        dedicated ``_ERROR_ANALYSIS_SEED_OFFSET`` stream: misclassified
        samples are printed / stored with their task-level structure
        (which key was cued, which value was predicted) so the failure
        mode can be classified rather than guessed.

    RNG isolation: the model is initialized with ``torch.manual_seed(seed)``,
    but a SEPARATE per-step generator (also seeded with ``seed``) drives
    ``make_mqar_batch``. This is critical for the multi-seed comparison: the
    different operators have different parameter counts, so they consume a
    different number of RNG draws during init. Without a separate generator
    for batch generation, the *same* seed produced *different* training
    batches across operators, undermining the apples-to-apples comparison
    that multi-seed CI is meant to strengthen.

    ``device`` may be passed as a string ('cpu'/'cuda') or a
    ``torch.device``; string inputs are coerced to ``torch.device`` so
    callers from notebooks don't hit ``AttributeError`` on ``device.type``.
    """
    # Coerce string device (e.g. 'cpu', 'cuda') to torch.device so callers
    # passing a string from a notebook don't hit AttributeError on
    # ``device.type`` inside this function. main() already passes a real
    # torch.device, but train_one is a public function and the coercion is
    # cheap and idempotent.
    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    if softmax_steps is None:
        softmax_steps = steps
    if train_batch is None:
        # Robust env var parsing: ``parse_int_env`` logs a warning and falls
        # back to the default on a malformed ``MQAR_TRAIN_BATCH=abc`` (or
        # ``=0``, which would crash on the first batch with
        # ZeroDivisionError), matching the robustness pattern already used
        # for BENCH_REPEATS in run_benchmark.py.
        train_batch = parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1,
                                    logger=logger)
    actual_steps = softmax_steps if op_name == 'softmax' else steps

    # Seed Python random / NumPy / torch / CUDA from one value with
    # deterministic cuDNN+algorithm selection. torch.manual_seed alone left
    # Python/NumPy streams unseeded and cuDNN free to pick non-reproducible
    # kernels on GPU.
    set_global_seed(seed, deterministic=True)
    # Shared construction helper (validates op_name, builds embed/head/layer
    # in the fixed seeded order — see _build_task_model's docstring).
    embed, head, layer = _build_task_model(op_name, d_model, vocab, device)
    # Record the operator's ACTUAL parameter count: same d_model does NOT
    # mean same capacity across operators (Mamba-2's internal expansion /
    # state dimensions differ from KDA's head geometry), and the quality
    # comparison must annotate the residual difference rather than assume
    # parity.
    n_params = sum(p.numel() for p in layer.parameters())
    # Build parameter groups with proper weight-decay exclusion: embeddings,
    # biases, and LayerNorm parameters are NOT weight-decayed (standard ML
    # practice — decaying these can hurt training quality).
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    # Separate generator for batch generation so the per-step batches are
    # IDENTICAL across operators for a given seed (the model init consumed a
    # different number of RNG draws per operator, which would otherwise
    # desync the global RNG and produce different training data per op).
    # A LARGE offset (1_000_000) instead of ``+ 1`` keeps the batch RNG
    # stream in a completely different region of the seed space from the
    # model initialization stream, eliminating any overlap that would weaken
    # the t-test's independence assumption.
    # Routed through make_seeded_generator for CPU-fallback on older torch
    # builds.
    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    # Held-out validation monitoring (off when val_every <= 0). The monitor
    # stream is seed + 6_000_000 — distinct from training (+1_000_000) and
    # from the headline eval (+5_000_000) — so the curve is a genuine
    # held-out signal and enabling it cannot change any reported metric
    # (read-only eval, dedicated RNG). See _VAL_MONITOR_SEED_OFFSET.
    #
    # ValidationCurve (rather than a bare list) is used so the record keeps
    # the best-validation point and the early-stopping regret alongside the
    # final number: a run whose final weights sit past the turn is visible
    # as a positive regret instead of only as "final_acc = 0.62".
    val_every = int(val_every or 0)
    val_curve = None
    if val_every > 0:
        def _monitor_validation(step):
            acc_v, loss_v = _eval_model(
                layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=val_batches, batch=val_batch, seed=seed, task=task,
                gen_offset=_VAL_MONITOR_SEED_OFFSET)
            return ValPoint(loss=loss_v, acc=acc_v)

        val_curve = ValidationCurve(
            _monitor_validation, steps=actual_steps, val_every=val_every,
            label=f'{op_name}/{task}/n_kv={n_kv}/seed={seed}', ceiling=1.0)
        val_curve.record(0)

    losses, accs = [], []
    # Pre-clip gradient norms. The clip call already computes this value,
    # so recording it is free and turns "the loss stayed finite" into
    # "the optimizer stayed healthy" (a long tail of huge norms is the
    # first sign of an LR that is too large).
    grad_norms = []
    # Set train mode ONCE before the loop, not per step (calling
    # layer.train()/head.train() inside the loop is a redundant O(steps)
    # no-op once the modules are already in train mode).
    layer.train()
    head.train()
    embed.train()
    for step in range(actual_steps):
        x_emb, target, cue_pos = make_batch(
            train_batch, seq_len, n_kv, vocab, embed, device, generator=batch_gen)
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        # NaN/Inf guard: if a single step diverges (e.g. AdamW + lr=3e-3
        # instability, or a single Inf in KDA state from a bad beta/g
        # combination), backward() would propagate NaN into ALL parameter
        # grads, opt.step() would write NaN into ALL parameters, and the
        # rest of training would silently produce NaN logits — the loss
        # curve would look like [..., 2.4, 2.5, NaN, NaN, NaN, ...] with
        # no error raised. Worse, the final eval would still return a
        # *finite* accuracy (argmax on NaN is undefined but deterministic,
        # often returning 0), so the divergent seed would NOT be caught by
        # the per-seed try/except in train_multi_seed and would silently
        # corrupt the aggregate mean/CI. Raise here so the per-seed
        # try/except catches it as a failed seed.
        if not torch.isfinite(loss):
            # Systematic triage before raising: the report names the first
            # failing stage (data -> loss -> params -> logits) so the
            # per-seed failure message is actionable, not just "NaN".
            _diag = diagnose_nonfinite(
                loss=loss, inputs={'x_emb': x_emb}, params=params,
                logits=logits,
                context=f'{op_name}/task={task}/seed={seed}/step={step}')
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(op={op_name}, seed={seed}); diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad()
        loss.backward()
        # Guard against NaN/Inf gradients BEFORE step. ``clip_grad_norm_``
        # computes ``total_norm`` and (if NaN) the comparison
        # ``total_norm > max_norm`` is False, so no clipping happens and the
        # NaN grads pass through to ``opt.step()`` unchecked. That corrupts
        # ALL parameters in one step; the NEXT iteration's forward then
        # produces a NaN loss, which the finite-loss guard above catches —
        # but by then every parameter is already NaN and the seed is lost
        # without a clear root cause.
        #
        # The returned norm is already computed by the clip call, so
        # checking it costs no extra kernel and runs EVERY step (the old
        # every-50-steps per-parameter scan could let a NaN step through
        # between checks). Only when it is non-finite do we pay for the
        # per-parameter scan that names the offending tensors.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grad_norms.append(float(grad_norm))
        if not torch.isfinite(grad_norm):
            bad_grads = {f'p{i}': p.grad for i, p in enumerate(params)
                         if p.grad is not None
                         and not torch.isfinite(p.grad).all()}
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                grads=bad_grads or None, grad_norm=float('nan'),
                context=f'{op_name}/task={task}/seed={seed}/step={step}')
            raise RuntimeError(
                f"non-finite gradient norm at step {step} "
                f"({len(bad_grads)} non-finite param grads; op={op_name}, "
                f"seed={seed}); diagnosis [{_diag['verdict']}]: "
                f"{_diag['summary']}")
        opt.step()
        losses.append(loss.item())
        accs.append(acc)
        if val_curve is not None and (step + 1) % val_every == 0:
            val_curve.record(step + 1)

    # Ensure the monitored curve ends at the final trained state (e.g.
    # steps=25 with val_every=10 would otherwise stop at step 20).
    if val_curve is not None:
        val_curve.close(actual_steps)

    # Final eval on multiple fresh batches for a stable accuracy estimate.
    # Timed so the results carry an in-experiment inference-cost proxy,
    # used by the efficiency-vs-accuracy (Pareto) annotation in main().
    # Predictions are collected in the SAME pass (collect_predictions=True)
    # to compute imbalance-aware metrics (macro/micro/weighted F1,
    # balanced accuracy, per-class support, one-vs-rest PR-AUC/ROC-AUC)
    # without a second forward pass.
    eval_t0 = time.time()
    final_acc, final_loss, final_eval = _eval_model(
        layer, head, embed, seq_len, n_kv, vocab, device,
        n_batches=eval_batches, batch=eval_batch, seed=seed, task=task,
        collect_predictions=True,
    )
    final_eval_time_s = time.time() - eval_t0
    chance = 1.0 / vocab

    # Confusion-derived metrics. The MULTI-CLASS synthetic probes are
    # class-balanced BY CONSTRUCTION (targets uniform over the task's
    # vocab), but the metrics are reported anyway so (a) the class-balance
    # assumption is VERIFIED in the record (``class_balance``) instead of
    # assumed, (b) a collapsed predictor cannot hide behind accuracy, and
    # (c) the averaging convention of every F1 is explicit. If a future
    # task variant becomes imbalanced, the same record already carries the
    # imbalance-robust metrics.
    final_metrics = None
    if final_eval['pred'].numel() > 0:
        final_metrics = multiclass_classification_metrics(
            final_eval['pred'], final_eval['target'], num_classes=vocab,
            scores=final_eval['scores'])

    # Post-training results analysis, all from streams the training never
    # saw: the held-out validation curve (monitor stream), its
    # over/underfitting verdict, and the bad-case analysis (error stream).
    if val_curve is not None:
        curve_block = val_curve.finalize(losses, accs)
        curve_block['val_monitor_seed_offset'] = _VAL_MONITOR_SEED_OFFSET
        val_curve.log_verdict(curve_block)
    else:
        curve_block = None
    val_history = (curve_block['val_history'] if curve_block else [])
    val_losses = [v['loss'] for v in val_history]
    curve_diagnosis = (curve_block['curve_diagnosis'] if curve_block
                       else diagnose_training_curves(losses, None))
    error_analysis = None
    if int(error_batches or 0) > 0:
        error_analysis = analyze_errors(
            layer, head, embed, op_name=op_name, task=task, seq_len=seq_len,
            n_kv=n_kv, vocab=vocab, device=device,
            n_batches=int(error_batches), batch=int(error_batch), seed=seed,
            max_examples=int(max_error_examples))
        logger.info(
            f"  [bad-case] {op_name}/{task} n_kv={n_kv} seed={seed}: "
            f"{error_analysis['n_wrong']}/{error_analysis['n_samples']} "
            f"wrong; categories: "
            f"{ {c: v['count'] for c, v in error_analysis['categories'].items() if v['count']} }")

    gradient_norms = summarize_gradient_norms(grad_norms, clip_norm=1.0)
    curve_fields = {}
    if curve_block is not None:
        # Additive curve-reading fields. ``final_acc`` above still means
        # "accuracy at the fixed step budget"; these say whether that
        # endpoint was the right place to read the curve, and whether an
        # apparently flat validation loss is actually accuracy saturation.
        for key in ('best_val', 'val_loss_regret_from_best',
                    'val_acc_regret_from_best', 'steps_since_best',
                    'best_val_is_final', 'early_stopping_summary',
                    'accuracy_regime', 'val_acc_first', 'val_acc_last',
                    'val_acc_peak'):
            if key in curve_block:
                curve_fields[key] = curve_block[key]
    analysis_fields = {
        'val_history': val_history,
        'curve_diagnosis': curve_diagnosis,
        'error_analysis': error_analysis,
        'gradient_norms': gradient_norms,
        'final_eval_time_s': final_eval_time_s,
        'final_eval_ms_per_sample': (
            1000.0 * final_eval_time_s / max(1, int(eval_batches) * int(eval_batch))),
        'generalization_gap': (
            curve_block['generalization_gap'] if curve_block is not None
            else None),
        **curve_fields,
        # Full confusion-derived metric block (each key names its averaging
        # convention — see results_analysis.multiclass_classification_metrics)
        # plus flat convenience keys so a table/DataFrame does not need to
        # reach into the nested dict.
        'final_metrics': final_metrics,
        'final_f1_macro': (final_metrics or {}).get('f1_macro'),
        'final_f1_weighted': (final_metrics or {}).get('f1_weighted'),
        'final_balanced_accuracy': (final_metrics or {}).get('balanced_accuracy'),
        'final_pr_auc_macro_ovr': (final_metrics or {}).get('pr_auc_macro_ovr'),
        'final_roc_auc_macro_ovr': (final_metrics or {}).get('roc_auc_macro_ovr'),
    }
    # Guard against steps=0 (would crash on accs[-1] / sum([])/0 below).
    # Returns a stub with the eval result and zeros for the training
    # trajectory fields. ``actual_steps`` is reported as 0 so downstream
    # consumers can detect the degenerate case.
    if not losses:
        return {
            'op': op_name,
            'task': task,
            'n_kv': n_kv,
            'n_params': n_params,
            'final_acc': final_acc,
            'final_loss': final_loss,
            'chance_acc': chance,
            'last_train_acc': 0.0,
            'mean_last10_loss': 0.0,
            'mean_last10_acc': 0.0,
            'steps': actual_steps,
            'seed': seed,
            'train_batch': train_batch,
            'lr': lr,
            'loss_curve': [],
            'acc_curve': [],
            **analysis_fields,
        }
    return {
        'op': op_name,
        'task': task,
        'n_kv': n_kv,
        'n_params': n_params,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'chance_acc': chance,
        'last_train_acc': accs[-1],
        'mean_last10_loss': sum(losses[-10:]) / min(10, len(losses)),
        'mean_last10_acc': sum(accs[-10:]) / min(10, len(accs)),
        'steps': actual_steps,
        'seed': seed,
        'train_batch': train_batch,
        # Full per-step training curves (the experiment log, not just the
        # endpoints) so a run can be audited/plotted without re-training.
        'lr': lr,
        'loss_curve': losses,
        'acc_curve': accs,
        **analysis_fields,
    }


_OVERFIT_BATCH_SEED_OFFSET = 3_000_000


def _task_batch_fingerprint(x_emb, target, cue_pos):
    """sha256 over the fixed overfit batch (embedded inputs + targets).

    Makes the gate report self-describing: a future reader can verify that
    a re-run with the same gate seed reproduces the same fixed batch, and
    detect a silently changed task generator or tokenizer between runs.
    """
    import hashlib

    h = hashlib.sha256()
    for name, t in (('x_emb', x_emb), ('target', target)):
        arr = t.detach().to('cpu').contiguous()
        h.update(f'{name}:{tuple(arr.shape)}:'.encode())
        h.update(str(arr.view(-1).tolist()).encode())
    h.update(f'cue_pos:{cue_pos}'.encode())
    return h.hexdigest()


def run_overfit_sanity_check(op_name, task='mqar', d_model=32, seq_len=16,
                             n_kv=1, vocab=16, steps=200, lr=3e-3,
                             seed=2026, device='cpu', train_batch=32,
                             acc_threshold=0.99, loss_ratio_threshold=0.5,
                             grad_clip=1.0, log_every=0):
    """Overfit ONE fixed batch as a pre-training gradient-flow gate.

    Trains a FRESH model (built through the exact ``train_one`` construction
    path) on a single fixed synthetic batch with all regularization
    disabled: ``weight_decay=0``, no dropout (none of these operators has
    any), and no data augmentation (the generators produce raw task batches;
    the same batch is regenerated deterministically every step from
    ``seed + _OVERFIT_BATCH_SEED_OFFSET``, so the autograd graph — including
    the embedding's gradient path — is fresh each step while the DATA is
    held constant).

    If the model cannot drive a fixed batch of a few dozen examples to
    near-100% train accuracy, something is fundamentally broken (gradients
    not flowing, loss mis-wired, labels reversed, optimizer not stepping)
    and the full multi-seed sweep would burn compute on a broken recipe.

    Pass criterion (either): mean train accuracy over the last ``tail``
    steps >= ``acc_threshold`` (default 0.99: "classification accuracy
    close to 100%"), OR the tail-mean loss dropped below
    ``loss_ratio_threshold`` times the head-mean loss ("loss rapidly
    approaching 0").

    This is a NECESSARY, NOT SUFFICIENT check: passing proves gradients
    flow and the loss responds on a fixed batch, but data-augmentation,
    feature-alignment and label-mapping bugs can still pass it. It must be
    complemented by inspecting intermediate results.

    Returns a JSON-serializable report with the loss/accuracy curves, the
    fixed-batch fingerprint, the seeded construction info, and the pass
    verdict.
    """
    import gc

    if isinstance(device, str):
        device = torch.device(device)
    if task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"task={task!r} must be one of {sorted(SYNTHETIC_TASKS)}.")
    make_batch = SYNTHETIC_TASKS[task]['make']
    steps = int(steps)
    if steps < 10:
        # A handful of steps is dominated by early-loss noise (the loss can
        # rise for a few steps before collapsing); requiring >= 10 avoids a
        # misleading FAIL on a healthy model (mirrors
        # train_lm_autodl.run_overfit_sanity_check).
        raise ValueError(
            f"overfit check needs steps >= 10 to be meaningful, got {steps}")
    train_batch = int(train_batch)
    if train_batch < 1:
        raise ValueError(f"train_batch must be >= 1, got {train_batch}")
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    # Deterministic, regularization-free recipe. Re-seeding here makes the
    # probe model's init and data independent of any RNG consumption before
    # the call; the subsequent full training re-seeds per op/seed anyway, so
    # the gate cannot perturb the experiment's numbers.
    seed_info = set_global_seed(seed, deterministic=True)
    embed, head, layer = _build_task_model(op_name, d_model, vocab, device)
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.0)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layer.train()
    head.train()
    embed.train()

    losses, accs, elapsed_s = [], [], []
    first_batch = None
    grad_flow = None
    update_report = None
    t0 = time.time()
    for step in range(steps):
        gen = make_seeded_generator(seed + _OVERFIT_BATCH_SEED_OFFSET,
                                    device=device)
        x_emb, target, cue_pos = make_batch(
            train_batch, seq_len, n_kv, vocab, embed, device,
            generator=gen)
        if first_batch is None:
            first_batch = (x_emb.detach(), target.detach(), cue_pos)
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()), inputs={'x_emb': x_emb},
                params=params, logits=logits,
                context=f'overfit sanity check step={step} '
                        f'op={op_name} task={task}')
            raise RuntimeError(
                f"overfit sanity check: non-finite loss at step {step} "
                f"({loss.item()}) for op={op_name} task={task}; diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            # One-shot gradient-flow audit on the first backward: names the
            # parameters the loss never reached (a detached / dead branch
            # can hide behind a passing accuracy or loss criterion).
            grad_flow = grad_flow_report(
                [(f'layer.{n}', p) for n, p in layer.named_parameters()]
                + [(f'head.{n}', p) for n, p in head.named_parameters()]
                + [(f'embed.{n}', p) for n, p in embed.named_parameters()])
        bad_grads = [p for p in params
                     if p.grad is not None
                     and not torch.isfinite(p.grad).all()]
        clip_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            _diag = diagnose_nonfinite(
                loss=float(loss.detach()), inputs={'x_emb': x_emb},
                params=params, logits=logits, grads=bad_grads or None,
                grad_norm=float(clip_norm),
                context=f'overfit sanity check step={step} '
                        f'op={op_name} task={task}')
            raise RuntimeError(
                f"overfit sanity check: non-finite gradient at step {step} "
                f"for op={op_name} task={task}; diagnosis "
                f"[{_diag['verdict']}]: {_diag['summary']}")
        named_params = (
            [(f'layer.{n}', p) for n, p in layer.named_parameters()]
            + [(f'head.{n}', p) for n, p in head.named_parameters()]
            + [(f'embed.{n}', p) for n, p in embed.named_parameters()])
        before_step = None
        if update_report is None:
            before_step = snapshot_parameters(named_params)
        opt.step()
        if update_report is None:
            update_report = optimizer_update_report(
                before_step, named_params, opt, step=step)
        losses.append(float(loss.detach()))
        accs.append(float(acc))
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[overfit-check] {op_name}/{task} step {step:4d} "
                  f"loss {losses[-1]:.4f} acc {acc:.4f} "
                  f"({elapsed_s[-1]:.1f}s)", flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail
    loss_reduction = ((initial_loss - final_loss) / initial_loss
                      if initial_loss > 0 else None)
    passed = ((final_acc >= acc_threshold)
              or (final_loss < loss_ratio_threshold * initial_loss))
    if update_report is not None and not update_report['ok']:
        passed = False
    report = {
        'passed': bool(passed),
        'op': op_name,
        'task': task,
        'n_kv': int(n_kv),
        'seq_len': int(seq_len),
        'vocab': int(vocab),
        'd_model': int(d_model),
        'steps': steps,
        'train_batch': train_batch,
        'lr': float(lr),
        'weight_decay': 0.0,
        'dropout': 0.0,
        'data_augmentation': 'none',
        'grad_clip_norm': float(grad_clip),
        'seed': int(seed),
        'batch_generator_seed': int(seed + _OVERFIT_BATCH_SEED_OFFSET),
        'seed_info': seed_info,
        'fixed_batch_sha256': _task_batch_fingerprint(*first_batch),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': min(losses),
        'loss_reduction': loss_reduction,
        'final_train_acc': final_acc,
        'best_train_acc': max(accs),
        'acc_threshold': float(acc_threshold),
        'loss_ratio_threshold': float(loss_ratio_threshold),
        'loss_curve': losses,
        'acc_curve': accs,
        'elapsed_s': elapsed_s,
        # Gradient-flow audit (first step): which trainable tensors the loss
        # actually reached, recorded next to the pass verdict.
        'grad_flow': grad_flow,
        # Optimizer coverage/update audit (first step): proves every
        # trainable tensor is in a param group and actually moved. A gate
        # that only watched the loss could pass while the optimizer skipped
        # a frozen or unwired branch.
        'optimizer_updates': update_report,
        'note': ('Necessary but NOT sufficient: passing proves gradients '
                 'flow and the loss responds on a fixed batch; it cannot '
                 'rule out data-augmentation / feature-alignment / '
                 'label-mapping bugs.'),
    }
    del embed, head, layer, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if grad_flow and grad_flow['n_missing_grad']:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the fixed batch for "
              f"{op_name}/{task} (e.g. "
              f"{grad_flow['missing_grad_names'][:3]})", flush=True)
    return report


def _compact_overfit_report(report):
    """Strip the per-step curves/seed block for compact metadata embedding.

    The full curves live in ``results/exp4_overfit_gate.json`` (written once
    per run); the exp4/exp4b metadata only needs the verdict and headline
    numbers, so embedding the curves twice would bloat every result file.
    """
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def train_multi_seed(op_name, n_seeds=5, steps=100, softmax_steps=None,
                     device='cpu', **kw):
    """Train ``op_name`` over ``n_seeds`` seeds.

    Returns a dict with per-seed results plus aggregate stats (mean, std,
    95% CI half-width via t-distribution). Any ``task=...`` keyword is
    forwarded to :func:`train_one` (default ``'mqar'``).

    Per-seed error handling: a single divergent seed (NaN loss, OOM, etc.)
    is caught and recorded as a stub entry in ``per_seed`` rather than
    crashing the whole operator's multi-seed run. Aggregate stats are
    computed over the surviving (successful) seeds. If ALL seeds fail, the
    function raises a ``RuntimeError`` so the caller can record a stub
    result for the operator (mirrors ``run_ablation.py::eval_layout_multi_seed``).

    ``device`` may be passed as a string or a ``torch.device``; string
    inputs are coerced for notebook callers.
    """
    # Coerce string device -> torch.device so callers passing 'cpu'/'cuda'
    # from a notebook don't hit AttributeError on ``device.type`` below.
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    # Compute the actual training-step count for THIS op once, mirroring
    # ``train_one``'s ``actual_steps = softmax_steps if op_name == 'softmax'
    # else steps`` logic, so the recorded ``steps`` field in the per-seed
    # error stub matches what a SUCCESS row would have recorded.
    actual_steps_for_op = softmax_steps if op_name == 'softmax' else steps
    stub_train_batch = kw.get('train_batch')
    if stub_train_batch is None:
        stub_train_batch = parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1,
                                         logger=logger)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        # Per-seed try/except: one divergent seed should not crash the
        # whole operator. We log and record a stub; the aggregate stats
        # are computed over whichever seeds succeeded.
        try:
            r = _public_api('train_one', train_one)(
                op_name, seed=s, steps=steps, device=device,
                          softmax_steps=softmax_steps, **kw)
            r['train_time_s'] = time.time() - t0
            # Log BEFORE appending to per_seed, so a failure inside
            # ``logger.info`` cannot record the seed as a success first and
            # then append a duplicate error stub for the same seed.
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"steps={r['steps']}  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            # Per-seed error stub: include ALL success-row fields (set to
            # None / 0 where not applicable) so downstream consumers using
            # ``pd.DataFrame(per_seed)`` or ``per_seed[0].get('steps', ...)``
            # see a consistent schema. ``pd.DataFrame(per_seed)`` uses the
            # first record's keys as columns, and
            # ``make_figures._plot_mqar_group`` reads
            # ``per_seed[0].get('steps', 100)`` for the figure title.
            per_seed.append({
                'op': op_name,
                'task': kw.get('task', 'mqar'),
                'n_kv': kw.get('n_kv', 1),
                'n_params': None,
                'final_acc': None,
                'final_loss': None,
                'chance_acc': 1.0 / kw.get('vocab', 16) if 'vocab' in kw else None,
                'last_train_acc': None,
                'mean_last10_loss': None,
                'mean_last10_acc': None,
                # Use the OP-SPECIFIC step count (softmax_steps for softmax,
                # steps for others) so the stub matches what a success row
                # would record, even when
                # MQAR_SOFTMAX_STEPS != MQAR_STEPS.
                'steps': actual_steps_for_op,
                'seed': s,
                'train_batch': stub_train_batch,
                'train_time_s': time.time() - t0,
                'error': str(e),
                # Mirror the success-row training-curve keys (empty) so the
                # per_seed schema is uniform across success/error records.
                'lr': kw.get('lr', 3e-3),
                'loss_curve': [],
                'acc_curve': [],
                # Mirror the results-analysis keys so a DataFrame over
                # per_seed has one schema for success and error rows.
                'val_history': [],
                'curve_diagnosis': None,
                'error_analysis': None,
                'final_eval_time_s': None,
                'final_eval_ms_per_sample': None,
                'generalization_gap': None,
                'best_val': None,
                'val_loss_regret_from_best': None,
                'val_acc_regret_from_best': None,
                'steps_since_best': None,
                'best_val_is_final': None,
                'early_stopping_summary': None,
                'accuracy_regime': None,
                'val_acc_first': None,
                'val_acc_last': None,
                'val_acc_peak': None,
                'gradient_norms': None,
                'final_metrics': None,
                'final_f1_macro': None,
                'final_f1_weighted': None,
                'final_balanced_accuracy': None,
                'final_pr_auc_macro_ovr': None,
                'final_roc_auc_macro_ovr': None,
            })
        # On GPU, clear the CUDA cache between seeds so the allocator does
        # not accumulate freed-but-unreleased blocks across seeds.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Filter out failed seeds before computing aggregate stats.
    # A seed that diverged to NaN loss is caught by train_one's NaN guard
    # and recorded with an 'error' key — but a seed that produced a NaN
    # final_acc/final_loss via a path the guard did not cover (e.g. eval
    # on NaN params from a step that crashed before the guard fired) would
    # survive the 'error' filter and silently corrupt the aggregate mean
    # (NaN propagates through sum()/std()). Defensively reject any seed
    # whose final_acc or final_loss is None or non-finite.
    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and _is_finite_real(r.get('final_acc'))
        and _is_finite_real(r.get('final_loss'))
    ]
    if not ok_per_seed:
        # All seeds failed: propagate the error so the caller can record
        # a stub result for this operator.
        raise RuntimeError(
            f"all {n_seeds} seeds failed for op '{op_name}'; "
            f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    losses = [r['final_loss'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_loss = sum(losses) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / (n - 1)
        std_acc = math.sqrt(var_acc)
        std_loss = math.sqrt(var_loss)
        # 95% CI half-width using t-distribution with n-1 dof.
        t = _t_crit_975(n)
        ci_acc = t * (std_acc / math.sqrt(n))
        ci_loss = t * (std_loss / math.sqrt(n))
    else:
        # With a single seed, the sample std is undefined and the t-CI is
        # NOT zero — the uncertainty is maximal. Return None (like ci95)
        # so downstream consumers (the summary table below) render 'n/a'
        # instead of a misleading '0.0000' that implies perfect precision.
        std_acc = std_loss = None
        ci_acc = ci_loss = None

    # One-sample t-test vs chance: tests whether mean_acc differs from the
    # chance level. The t-statistic is only defined when n > 1 and the
    # sample standard deviation is strictly positive; otherwise we return
    # None (the test is not computable, not "infinitely significant").
    chance = ok_per_seed[0]['chance_acc']
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    # Curve-diagnosis verdict tally over the surviving seeds: the experiment
    # record says HOW MANY seeds showed overfitting / underfitting, instead
    # of leaving the reader to infer it from the per-seed curves.
    verdicts: dict = OrderedDict()
    n_budget_limited = 0
    n_val_still = 0
    for r in ok_per_seed:
        diag = r.get('curve_diagnosis') or {}
        v = diag.get('verdict', 'unknown')
        verdicts[v] = verdicts.get(v, 0) + 1
        if diag.get('budget_limited'):
            n_budget_limited += 1
        if (diag.get('signals') or {}).get('val_still_improving_at_end'):
            n_val_still += 1

    # Aggregate the curve-reading fields that a final-number-only table
    # hides: how far the final weights sit past the best validation point
    # (early-stopping regret), and how often validation accuracy was pinned
    # at the task ceiling (saturation, not overfitting).
    regrets = [r['val_loss_regret_from_best'] for r in ok_per_seed
               if r.get('val_loss_regret_from_best') is not None]
    regimes = [r.get('accuracy_regime') for r in ok_per_seed
               if r.get('accuracy_regime')]
    n_saturated = sum(1 for reg in regimes if reg.get('saturated'))
    tail_accs = [reg['val_acc_tail_mean'] for reg in regimes
                 if reg.get('val_acc_tail_mean') is not None]

    # Pooled bad-case analysis across seeds: counts add up over every
    # analyzed sample, while the EXAMPLES come from the worst seed (lowest
    # final accuracy) so the artifact shows the most informative failures
    # rather than a lucky seed's.
    # One implementation for every caller: the helper also pools the
    # per-category confidence block (ambiguous near-ties vs confident blind
    # spots) and picks examples from the worst seed.
    error_aggregate = pool_error_analyses(ok_per_seed)

    gaps = [r['generalization_gap'] for r in ok_per_seed
            if r.get('generalization_gap') is not None]

    # Seed-aggregated confusion metrics. Macro and weighted F1 are kept
    # SEPARATE (they differ under imbalance); micro is omitted here because
    # for single-label multi-class it is identical to accuracy, which the
    # row already carries (the per-seed metrics dict records it explicitly).
    def _seed_metric_mean(key):
        vals = [r['final_metrics'][key] for r in ok_per_seed
                if r.get('final_metrics')
                and r['final_metrics'].get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def _seed_metric_std(key):
        vals = [r['final_metrics'][key] for r in ok_per_seed
                if r.get('final_metrics')
                and r['final_metrics'].get(key) is not None]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return math.sqrt(var)

    mean_final_metrics = {
        'f1_macro_mean': _seed_metric_mean('f1_macro'),
        'f1_macro_std': _seed_metric_std('f1_macro'),
        'f1_weighted_mean': _seed_metric_mean('f1_weighted'),
        'f1_weighted_std': _seed_metric_std('f1_weighted'),
        'balanced_accuracy_mean': _seed_metric_mean('balanced_accuracy'),
        'pr_auc_macro_ovr_mean': _seed_metric_mean('pr_auc_macro_ovr'),
        'roc_auc_macro_ovr_mean': _seed_metric_mean('roc_auc_macro_ovr'),
        'class_balance': ((ok_per_seed[0].get('final_metrics') or {})
                          .get('class_balance')),
        'averaging_note': (
            'F1 variants are labelled explicitly: f1_macro = unweighted '
            'per-class mean, f1_weighted = support-weighted mean. Micro F1 '
            'equals accuracy for single-label multi-class and is recorded '
            'per seed in per_seed[i].final_metrics.f1_micro.'),
    }

    return {
        'op': op_name,
        'task': ok_per_seed[0].get('task', 'mqar'),
        'n_kv': ok_per_seed[0]['n_kv'],
        'n_params': ok_per_seed[0].get('n_params'),
        # Report the REQUESTED seed count (len(per_seed)), not the surviving
        # count (n), so the figure title and downstream consumers reflect
        # how many seeds were actually run. ``n_seeds_ok`` / ``n_seeds_failed``
        # carry the surviving / failed breakdown. Mirrors run_ablation.py's
        # convention (which already used len(per_seed) here).
        'n_seeds': len(per_seed),
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds_total': len(per_seed),
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'ci95_loss': ci_loss,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat,
        # The lr actually used for every seed of this row (tuned for the
        # baseline ops, 3e-3 otherwise). Per-seed values live in
        # per_seed[i]['lr']; they are identical across seeds.
        'lr': ok_per_seed[0].get('lr', 3e-3),
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
        'mean_eval_ms_per_sample': sum(
            r.get('final_eval_ms_per_sample', 0.0)
            for r in ok_per_seed) / n,
        'curve_diagnosis_summary': {
            'verdicts': dict(verdicts),
            'n_seeds_summarized': len(ok_per_seed),
            # A healthy verdict does not mean the fixed-budget endpoint is
            # converged: count the seeds whose train AND validation tails
            # were still falling when training stopped.
            'n_seeds_budget_limited': int(n_budget_limited),
            'n_seeds_val_still_improving': int(n_val_still),
            'budget_note': (
                'budget_limited seeds were still descending on both curves '
                'at the final checkpoint; their endpoint is not a converged '
                'property of the operator and comparisons may reorder with '
                'more steps.' if n_budget_limited else None),
        },
        'mean_generalization_gap': (sum(gaps) / len(gaps)) if gaps else None,
        # Best-validation regret: a positive value means the fixed-budget
        # final weights were past the turning point, so the final metric
        # understates what this operator achieved during training.
        'mean_val_loss_regret_from_best': (
            sum(regrets) / len(regrets)) if regrets else None,
        'n_seeds_with_positive_val_regret': sum(1 for x in regrets if x > 0),
        'accuracy_regime_summary': {
            'n_seeds_saturated': int(n_saturated),
            'n_seeds_with_regime': len(regimes),
            'mean_val_acc_tail': (
                sum(tail_accs) / len(tail_accs)) if tail_accs else None,
            'note': ('saturated = validation accuracy at the task ceiling '
                     'while train accuracy is still rising; this is '
                     'convergence, not overfitting, and more steps cannot '
                     'improve the metric.'),
        },
        'error_analysis': error_aggregate,
        'mean_final_metrics': mean_final_metrics,
    }


def _parse_nkv_list(env_var, default='1'):
    """Parse a comma-separated list of n_kv values from an env var.

    ``MQAR_NKV=1``       -> [1]           (backward compatible)
    ``MQAR_NKV=1,2,4``   -> [1, 2, 4]
    """
    raw = os.environ.get(env_var, default)
    try:
        vals = [int(x.strip()) for x in raw.split(',') if x.strip()]
    except ValueError:
        raise ValueError(
            f'{env_var}={raw!r} must be comma-separated ints, e.g. "1,2,4"')
    if not vals:
        vals = [int(default)]
    if any(v < 1 for v in vals):
        raise ValueError(
            f'{env_var} values must be positive integers, got {vals!r}')
    # Preserve user order while removing duplicates; duplicate task settings
    # would repeat identical experiments and overweight them in summaries.
    return list(dict.fromkeys(vals))


def _parse_task_list(env_var, default='mqar'):
    """Parse a comma-separated task list (or 'all') from an env var.

    ``MQAR_TASKS=all``            -> every task in SYNTHETIC_TASKS
    ``MQAR_TASKS=mqar,kv_update`` -> [mqar, kv_update]
    """
    raw = os.environ.get(env_var, default)
    if raw.strip().lower() == 'all':
        return list(SYNTHETIC_TASKS.keys())
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    if not vals:
        vals = [default]
    unknown = [v for v in vals if v not in SYNTHETIC_TASKS]
    if unknown:
        raise ValueError(
            f'{env_var} includes unknown task(s) {unknown}; '
            f'valid tasks: {sorted(SYNTHETIC_TASKS)} (or "all").')
    return list(dict.fromkeys(vals))


def _sweep_cells(task_list, n_kv_list):
    """The concrete ``(task, n_kv)`` cells the quality sweep will run.

    Tasks that do not sweep ``n_kv`` run only their default value. Keeping
    this in one place means the paired-comparison family size recorded in
    the design audit cannot drift from the cells that actually run.
    """
    cells = []
    for task in task_list:
        spec = SYNTHETIC_TASKS[task]
        kv_values = (n_kv_list if spec.get('n_kv_swept')
                     else [spec.get('default_n_kv', 1)])
        cells.extend((task, n_kv) for n_kv in kv_values)
    return cells


def _parse_op_list(env_var, default='linear,softmax,kda,mamba2,rwkv7,csa,hca'):
    """Parse the operator list for the quality sweep.

    ``MQAR_OPS`` selects which single-layer operators run. The default now
    starts with the SIMPLE-BASELINE LADDER — ``linear`` (no token mixing
    at all) and ``softmax`` (the simplest mixer) — followed by the
    complex operators (KDA / Mamba-2 / RWKV-7 / CSA / HCA). Every complex
    row is compared against the tuned reference baseline, not only
    against chance; see ``MQAR_BASELINE_*`` below. Set
    ``MQAR_OPS=softmax,kda,mamba2,rwkv7,csa,hca`` to reproduce the
    pre-baseline table exactly (or ``MQAR_OPS=softmax,kda,csa,hca`` for
    the historical four-operator table).
    """
    valid = ['linear', 'softmax', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca']
    raw = os.environ.get(env_var, default)
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    if not vals:
        vals = list(valid)
    unknown = [v for v in vals if v not in valid]
    if unknown:
        raise ValueError(
            f'{env_var} includes unknown op(s) {unknown}; valid ops: {valid}.')
    return list(dict.fromkeys(vals))


def _parse_lr_list(env_var, default='3e-4,1e-3,3e-3,1e-2,3e-2'):
    """Parse a comma-separated list of positive learning rates from an env var.

    Raises on a malformed / empty list (a silently empty tuning grid would
    turn the tuning protocol into a no-op).
    """
    raw = os.environ.get(env_var, default)
    vals = []
    for x in raw.split(','):
        x = x.strip()
        if not x:
            continue
        try:
            v = float(x)
        except ValueError:
            raise ValueError(
                f'{env_var}={raw!r} must be comma-separated floats, '
                f'e.g. "1e-3,3e-3,1e-2"')
        if not math.isfinite(v) or not (v > 0):
            raise ValueError(
                f'{env_var} values must be finite and > 0, got {v!r}')
        vals.append(v)
    if not vals:
        raise ValueError(
            f'{env_var}={raw!r} must contain at least one learning rate')
    return list(dict.fromkeys(vals))


def tune_baseline_lr(op_name, lrs, sweep, steps, softmax_steps, n_seeds,
                     device, val_every, val_batches, val_batch):
    """Select a baseline operator's learning rate on the VALIDATION stream.

    Experiment-design requirement: "implement the simplest baseline and TUNE
    IT WELL" before any complex model is declared better. This runs the
    baseline over ``lrs`` and picks the rate maximizing the held-out
    VALIDATION accuracy PLATEAU (mean of the last up-to-3 monitor points per
    seed; seed + ``_VAL_MONITOR_SEED_OFFSET`` stream), never the headline
    eval stream (seed + 5M), so selection cannot leak into any reported
    number. A single lr is selected per operator across all sweep
    cells (the row schema stays one-lr-per-op); the record keeps the full
    per-lr grid so the choice is auditable.

    Ties break toward the EARLIER entry in ``lrs`` (keep the more stable /
    smaller lr by convention, and keep the default 3e-3 grid order).

    Returns ``(best_lr, record)``. ``record`` is JSON-serializable and
    embedded in the run metadata.
    """
    actual_steps = softmax_steps if op_name == 'softmax' else steps
    # Ensure a validation signal exists even when the sweep disabled the
    # val monitor (MQAR_VAL_EVERY=0): the tuning runs then use a private
    # quarter-grid. This cannot change the headline metric — the monitor
    # stream is disjoint from the eval stream.
    tune_val_every = int(val_every) if int(val_every or 0) > 0 \
        else max(1, actual_steps // 4)

    # Selection statistic: mean of the LAST (up to) ``_TUNE_TAIL_K``
    # validation points per seed, averaged over seeds. The final point
    # alone is noisy (one unlucky monitor batch can flip the lr choice),
    # while the full curve would favour lrs that are still improving.
    # A tail window measures the converged plateau, which is what "tuned
    # well" should mean. The final-point mean is still recorded for
    # backward compatibility.
    _TUNE_TAIL_K = 3
    per_lr = OrderedDict()
    best_lr = lrs[0]
    best_score = None
    for lr in lrs:
        tail_scores = []
        final_scores = []
        tail_ks = []
        n_cell_runs_failed = 0
        for task, n_kv, task_seq_len, task_vocab in sweep:
            try:
                r = _public_api('train_multi_seed', train_multi_seed)(
                    op_name, n_seeds=n_seeds, steps=steps,
                    softmax_steps=softmax_steps, device=device,
                    lr=lr, task=task, n_kv=n_kv, vocab=task_vocab,
                    seq_len=task_seq_len, val_every=tune_val_every,
                    val_batches=val_batches, val_batch=val_batch,
                    error_batches=0)
            except Exception as e:
                n_cell_runs_failed += 1
                logger.warning(
                    f'    [tune] {op_name} lr={lr:g} {task}/n_kv={n_kv} '
                    f'FAILED: {e}')
                continue
            for seed_row in r.get('per_seed', []):
                if not isinstance(seed_row, dict):
                    continue
                vh = seed_row.get('val_history')
                if not isinstance(vh, (list, tuple)) or not vh:
                    continue
                last = vh[-1]
                if isinstance(last, dict):
                    acc = last.get('acc')
                    if _is_finite_real(acc):
                        final_scores.append(float(acc))
                k = min(_TUNE_TAIL_K, len(vh))
                tail_ks.append(k)
                tail_vals = []
                for point in vh[-k:]:
                    if not isinstance(point, dict):
                        continue
                    acc = point.get('acc')
                    if _is_finite_real(acc):
                        tail_vals.append(float(acc))
                if tail_vals:
                    tail_scores.append(_safe_mean(tail_vals))
        mean_score = _safe_mean(tail_scores) if tail_scores else None
        mean_final = _safe_mean(final_scores) if final_scores else None
        per_lr[f'{lr:g}'] = {
            'mean_val_tail_acc': mean_score,
            'mean_final_val_acc': mean_final,
            'val_tail_k': min(tail_ks) if tail_ks else 0,
            'n_val_points': len(tail_scores),
            'n_cell_runs_failed': n_cell_runs_failed,
        }
        logger.info(
            f'    [tune] {op_name:8s} lr={lr:g}  '
            f'val_tail_acc='
            f'{mean_score if mean_score is None else f"{mean_score:.4f}"} '
            f'(final='
            f'{mean_final if mean_final is None else f"{mean_final:.4f}"}, '
            f'{len(tail_scores)} points, {n_cell_runs_failed} failed cells)')
        # Strictly-greater: ties keep the earlier (user-order) lr.
        if mean_score is not None and (best_score is None
                                       or mean_score > best_score):
            best_score = mean_score
            best_lr = lr

    selected_final = (per_lr.get(f'{best_lr:g}', {})
                      .get('mean_final_val_acc'))
    return best_lr, {
        'op': op_name,
        'selected_lr': best_lr,
        'selected_lr_mean_val_acc': best_score,
        'selected_lr_mean_final_val_acc': selected_final,
        'selection_statistic': (f'tail mean: mean of the last min(3, n) '
                                f'validation points per seed, averaged '
                                f'over seeds'),
        'val_tail_k': _TUNE_TAIL_K,
        'selection_stream': f'val_monitor_seed+{_VAL_MONITOR_SEED_OFFSET}',
        'tune_seeds': n_seeds,
        'tune_val_every': tune_val_every,
        'lr_grid': per_lr,
        'note': ('Learning rate selected by the held-out validation '
                 'accuracy PLATEAU (tail mean, not a single final point), '
                 'and never by the headline eval stream; the winning lr is '
                 'then re-trained with the full seed count. Ties keep the '
                 'earlier/lower lr.'),
    }


__all__ = [
    '_run_overfit_gate',
    'train_one',
    '_OVERFIT_BATCH_SEED_OFFSET',
    '_task_batch_fingerprint',
    'run_overfit_sanity_check',
    '_compact_overfit_report',
    'train_multi_seed',
    '_parse_nkv_list',
    '_parse_task_list',
    '_sweep_cells',
    '_parse_op_list',
    '_parse_lr_list',
    'tune_baseline_lr',
]


def _run_overfit_gate(*, sweep, ops_list, task_list, device):
    """Run the pre-training overfit sanity gate and return its record.

    Returns a dict with the gate settings, per-cell reports, failure count
    and an ``aborted`` flag (strict mode + a failed cell).  Kept separate
    from ``main`` so the entry orchestration stays readable.
    """
    def _result(aborted=False):
        return {
            'aborted': aborted,
            'enabled': overfit_check_enabled,
            'strict': overfit_strict,
            'steps': overfit_steps,
            'seed': overfit_seed,
            'train_batch': gate_train_batch,
            'reports': overfit_reports,
            'n_failed': n_gate_failed,
        }

    # ------------------------------------------------------------------
    # Pre-training overfit sanity gate (before any multi-seed training):
    # for every (task cell, operator), train a FRESH model on ONE fixed
    # batch with regularization off and check it memorizes the batch
    # (train accuracy -> ~100%, loss -> ~0). This catches broken gradient
    # flow / loss wiring / label mapping cheaply before 5 seeds x N steps
    # of sweep time. Necessary, NOT sufficient (see the function
    # docstring). The gate RE-SEEDS internally and the sweep re-seeds per
    # op/seed, so enabling it does not change the experiment's numbers.
    # Controlled by env vars:
    #   MQAR_OVERFIT_CHECK=1|0  (default 1: run the gate)
    #   MQAR_OVERFIT_STRICT=1|0 (default 1: return non-zero if a cell fails)
    #   MQAR_OVERFIT_STEPS=N    (default 200 optimizer steps per cell)
    #   MQAR_OVERFIT_SEED=N     (default 2026; gate model + fixed batch)
    # ------------------------------------------------------------------
    overfit_check_enabled = parse_int_env(
        'MQAR_OVERFIT_CHECK', 1, min_value=0, logger=logger) != 0
    overfit_strict = parse_int_env(
        'MQAR_OVERFIT_STRICT', 1, min_value=0, logger=logger) != 0
    overfit_steps = parse_int_env(
        'MQAR_OVERFIT_STEPS', 200, min_value=10, logger=logger)
    overfit_seed = parse_int_env(
        'MQAR_OVERFIT_SEED', 2026, min_value=0, logger=logger)
    # Counts recorded in the run manifest even when the gate is disabled.
    n_gate_failed = 0
    # Cap the gate batch so a large MQAR_TRAIN_BATCH does not make the
    # memorization target unnecessarily hard / slow: "a fixed batch of a few
    # dozen examples" is the probe; 64 sequences still exercises every
    # code path of the real training batch.
    gate_train_batch = min(
        parse_int_env('MQAR_TRAIN_BATCH', 32, min_value=1, logger=logger), 64)
    overfit_reports = []
    if overfit_check_enabled:
        logger.info('\n' + '=' * 70)
        logger.info('Pre-training overfit sanity gate (fixed batch, '
                    'weight_decay=0, no dropout / augmentation)')
        logger.info(f'  steps={overfit_steps} seed={overfit_seed} '
                    f'batch={gate_train_batch} lr=3e-3  '
                    f'strict={overfit_strict}')
        logger.info('=' * 70)
        for task, n_kv, task_seq_len, task_vocab in sweep:
            for op in ops_list:
                # Memoryless controls cannot memorize a fixed batch BY
                # CONSTRUCTION (no cross-token pathway; the cue->value map is
                # not a function of the cue alone), so the gate is skipped
                # for them — recorded explicitly instead of counted as a
                # failure. See _MEMORYLESS_OPS.
                if op in _MEMORYLESS_OPS:
                    overfit_reports.append({
                        'passed': None,
                        'skipped': True,
                        'op': op, 'task': task, 'n_kv': n_kv,
                        'steps': overfit_steps, 'seed': overfit_seed,
                        'lr': 3e-3, 'train_batch': gate_train_batch,
                        'note': ('Memoryless control: the fixed-batch '
                                 'memorization criterion does not apply '
                                 '(no inter-token pathway). Skipped, not '
                                 'passed.'),
                    })
                    logger.info(
                        f'  [overfit] {op:8s} task={task} n_kv={n_kv}: '
                        'SKIP (memoryless control; gate not applicable)')
                    continue
                try:
                    rep = run_overfit_sanity_check(
                        op, task=task, seq_len=task_seq_len, n_kv=n_kv,
                        vocab=task_vocab, steps=overfit_steps,
                        seed=overfit_seed, device=device,
                        train_batch=gate_train_batch)
                except Exception as e:
                    # A gate crash is itself a red flag (e.g. non-finite
                    # loss); record it as a failed report rather than
                    # aborting with a traceback, so the side file keeps a
                    # complete record of every (cell, op) probe.
                    logger.error(f'  [overfit] {op:8s} task={task} '
                                 f'n_kv={n_kv}: ERROR {e}')
                    rep = {
                        'passed': False, 'op': op, 'task': task,
                        'n_kv': n_kv, 'steps': overfit_steps,
                        'seed': overfit_seed,
                        'train_batch': gate_train_batch,
                        'lr': 3e-3,
                        'error': str(e),
                        'note': ('Gate crashed; treat as a failed sanity '
                                 'check and inspect before trusting the '
                                 'sweep.'),
                    }
                overfit_reports.append(rep)
                logger.info(
                    f'  [overfit] {op:8s} task={task} n_kv={n_kv}: '
                    f"{'PASS' if rep.get('passed') else 'FAIL'} "
                    f"loss {rep.get('initial_loss', float('nan')):.3f} -> "
                    f"{rep.get('final_loss', float('nan')):.4f}, "
                    f"train_acc -> {rep.get('final_train_acc', float('nan')):.3f}")
        # Dedicated overfit-gate artifact (full per-step curves + batch
        # fingerprints), written BEFORE the strict check returns so a failed
        # gate leaves its evidence on disk. This is an experiment-tracking
        # record in its own right: if a later run's sweep numbers shift, this
        # file shows whether the probe model still memorized its fixed batch
        # under the SAME seed.
        os.makedirs('results', exist_ok=True)
        gate_payload = {
            'kind': 'exp4_overfit_gate',
            'schema_version': CURRENT_SCHEMA_VERSION,
            'metadata': {
                'schema_version': CURRENT_SCHEMA_VERSION,
                'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
                'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
                'rwkv_backend': os.environ.get('RWKV_BACKEND', 'reference'),
                'rwkv7_upstream_binding': RWKV7_UPSTREAM_BINDING,
                'provenance': capture_provenance(),
                'settings': {
                    'enabled': True,
                    'strict': bool(overfit_strict),
                    'steps': overfit_steps,
                    'seed': overfit_seed,
                    'train_batch': gate_train_batch,
                    'lr': 3e-3,
                    'weight_decay': 0.0,
                    'acc_threshold': 0.99,
                    'loss_ratio_threshold': 0.5,
                    'batch_generator_seed_offset': _OVERFIT_BATCH_SEED_OFFSET,
                },
                'data': {
                    'kind': 'synthetic',
                    'task_specs': {
                        t: {'seq_len': SYNTHETIC_TASKS[t]['seq_len'],
                            'vocab': SYNTHETIC_TASKS[t]['vocab']}
                        for t in task_list
                    },
                },
                'note': ('Necessary but NOT sufficient gradient-flow check. '
                         'Full training curves for each (task, op) probe are '
                         'in the reports below.'),
            },
            'reports': overfit_reports,
        }
        write_results_json(gate_payload, 'results/exp4_overfit_gate.json',
                           logger=logger)
        logger.info('Saved: results/exp4_overfit_gate.json')
        # Failed = checked cells (not skipped memoryless controls) that did
        # not pass the memorization criterion.
        n_gate_checked = sum(1 for r in overfit_reports
                             if not r.get('skipped'))
        n_gate_skipped = sum(1 for r in overfit_reports
                             if r.get('skipped'))
        n_gate_failed = sum(1 for r in overfit_reports
                            if not r.get('passed') and not r.get('skipped'))
        if n_gate_failed:
            logger.error(
                f'  {n_gate_failed}/{len(overfit_reports)} overfit gate '
                'cells FAILED: a fresh model could not memorize a single '
                'fixed batch of at most '
                f'{gate_train_batch} sequences in {overfit_steps} steps. '
                'This usually means gradients are not flowing, the loss is '
                'mis-wired, or labels are wrong — do NOT trust the sweep '
                'results. Debug with a single cell, or bypass (NOT '
                'recommended) with MQAR_OVERFIT_CHECK=0.')
            if overfit_strict:
                logger.error(
                    '  MQAR_OVERFIT_STRICT=1: refusing to start the full '
                    'multi-seed sweep on a broken training recipe. Set '
                    'MQAR_OVERFIT_STRICT=0 to record the failure and '
                    'continue (diagnostic-only mode).')
                return _result(True)
        else:
            logger.info(
                f'  all {n_gate_checked} checked overfit gate cells PASSED'
                + (f' ({n_gate_skipped} memoryless control cell(s) skipped)'
                   if n_gate_skipped else ''))

    return _result(False)
