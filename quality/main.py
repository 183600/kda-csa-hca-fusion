"""run_quality split module: main."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403

from .models import (  # noqa: F401
    _MEMORYLESS_OPS,
)

from .stats import (  # noqa: F401
    _HYPOTHESIS_STATUSES,
    _bonferroni_crit_q,
    _finite_per_seed_acc,
    _hypothesis_verdicts,
    _paired_one_sided_t,
    _pairwise_op_comparisons,
    _print_pairwise_report,
)

from .tasks import (  # noqa: F401
    OPERATOR_HYPOTHESES,
    SYNTHETIC_TASKS,
    TASK_HYPOTHESES,
    evaluate_rule_baseline,
)

from .training import (  # noqa: F401
    _run_overfit_gate,
    _OVERFIT_BATCH_SEED_OFFSET,
    _compact_overfit_report,
    _parse_lr_list,
    _parse_nkv_list,
    _parse_op_list,
    _parse_task_list,
    _sweep_cells,
    run_overfit_sanity_check,
    train_multi_seed,
    tune_baseline_lr,
)

from .utils import (  # noqa: F401
    _fmt_tstat,
    _parse_equivalence_margin,
)


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 4: Synthetic Quality Probes (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device        : {device}')
    # Task and operator selection. Defaults keep the historical MQAR-only
    # output file (results/exp4_mqar.json) while ADDING the mamba2 operator
    # to the sweep; the additional probes are opt-in via MQAR_TASKS so the
    # historical table stays reproducible with MQAR_TASKS=mqar +
    # MQAR_OPS=softmax,kda,csa,hca.
    task_list = _parse_task_list('MQAR_TASKS', 'mqar')
    # Default ops include the baseline ladder (linear, softmax) followed
    # by the complex mixers; see _parse_op_list. The historical table is
    # reproducible with MQAR_OPS=softmax,kda,mamba2,rwkv7,csa,hca.
    ops_list = _parse_op_list('MQAR_OPS')
    logger.info(f'  tasks         : {task_list}')
    logger.info(f'  ops           : {ops_list}')
    # Tuned-baseline protocol (experiment-design requirement). The simple
    # baselines get their own learning-rate sweep, selected on the held-out
    # VALIDATION stream, and every complex op is tested against the tuned
    # reference baseline — not only against chance. MQAR_BASELINE_TUNE=0
    # restores the historical fixed lr=3e-3 for every op.
    baseline_tune_enabled = parse_int_env(
        'MQAR_BASELINE_TUNE', 1, min_value=0, logger=logger) != 0
    baseline_ops = [x.strip() for x in
                    os.environ.get('MQAR_BASELINE_OPS', 'linear,softmax')
                    .split(',') if x.strip()]
    # Silently intersect with the sweep's op list: naming a baseline op that
    # is not being swept (e.g. MQAR_OPS without 'linear') is not an error.
    baseline_ops = [op for op in baseline_ops if op in ops_list]
    baseline_lrs = _parse_lr_list('MQAR_BASELINE_LRS')
    # Reference baseline for the paired comparisons. Defaults to softmax
    # (the simplest mixer); if it is not in the sweep, fall back to the
    # first swept baseline op. None => no pairwise test possible.
    _baseline_ref_env = os.environ.get('MQAR_BASELINE_REF')
    baseline_ref = (_baseline_ref_env or 'softmax').strip()
    if baseline_ref not in ops_list:
        if _baseline_ref_env is not None:
            logger.warning(
                f'  MQAR_BASELINE_REF={baseline_ref!r} is not in the op '
                f'list {ops_list}; falling back to the first swept '
                f'baseline op.')
        baseline_ref = baseline_ops[0] if baseline_ops else None
    if baseline_tune_enabled and baseline_ops:
        logger.info(f'  baseline tune : ops={baseline_ops} '
                    f'lrs={baseline_lrs} (validation-stream selection); '
                    f'reference={baseline_ref}')
    elif not baseline_tune_enabled:
        logger.info('  baseline tune : OFF (MQAR_BASELINE_TUNE=0; '
                    'historical fixed lr=3e-3 for every op)')
    # MQAR's task hyperparameters. Lifted to named constants so the
    # chance-accuracy log line below uses the SAME value as the per-seed
    # ``chance = 1.0 / vocab`` computation in ``train_one``.
    vocab = 16
    seq_len = 16
    n_kv_list = _parse_nkv_list('MQAR_NKV', '1')
    # Pre-validate n_kv against vocab and seq_len so the user gets a clear
    # error message instead of an opaque AssertionError from deep inside
    # ``make_mqar_batch`` during training.
    for n_kv in n_kv_list:
        if 2 * n_kv > vocab:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"vocab={vocab}; reduce n_kv or increase vocab.")
        if 2 * n_kv >= seq_len:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< seq_len={seq_len} (need room for the cue token at the end).")
    logger.info(f'  mqar: vocab={vocab}, seq_len={seq_len}, n_kv={n_kv_list}')
    logger.info('  per-task defaults: ' + '; '.join(
        f"{t}: seq_len={s['seq_len']}, vocab={s['vocab']}"
        for t, s in SYNTHETIC_TASKS.items() if t != 'mqar'))
    # Robust env var parsing: ``parse_int_env`` logs a warning and falls
    # back to the default on a malformed value (e.g. ``MQAR_SEEDS=abc``
    # that would otherwise raise ``ValueError: invalid literal for
    # int()``), matching the robustness pattern already used for
    # BENCH_REPEATS / BENCH_LENGTHS in run_benchmark.py.
    n_seeds = parse_int_env('MQAR_SEEDS', 5, min_value=1, logger=logger)
    steps = parse_int_env('MQAR_STEPS', 200, min_value=1, logger=logger)
    # Fair primary comparison: every operator receives the same optimizer-step
    # budget by default. MQAR_SOFTMAX_STEPS remains available for an explicitly
    # labelled convergence/long-training sensitivity run, but no method is
    # advantaged in the default result table.
    softmax_steps = parse_int_env('MQAR_SOFTMAX_STEPS', steps, min_value=1,
                                  logger=logger)
    # Prospective design audit (experiment_design.py): record the actual
    # seed budget, control set and analysis-family size against the
    # pre-registered Exp 4 plan. This does not change the run; it prevents a
    # screening budget from being read as a confirmatory result. The family
    # is one vs-reference contrast per non-baseline operator in EVERY
    # (task, n_kv) cell, not just the default cell: a multi-task / multi-n_kv
    # sweep runs proportionally more tests and therefore needs a correction
    # at least as strong.
    design_family_size = max(
        1, len([op for op in ops_list if op not in baseline_ops])
        * len(_sweep_cells(task_list, n_kv_list)))
    experiment_design_audit = audit_experiment_design(
        'exp4_mqar',
        seeds=[42 + i for i in range(n_seeds)],
        levels={'operator': list(ops_list), 'task': list(task_list),
                'n_kv': list(n_kv_list)},
        family_size=design_family_size,
    )
    logger.info(
        f'  design audit  : {experiment_design_audit["status"]} '
        f'(n={experiment_design_audit["n_seeds"]}, '
        f'required={experiment_design_audit["planning"]["required_seeds"]}, '
        f'power={experiment_design_audit["planning"]["planned_power"]})')
    # Results-analysis controls (monitoring + bad cases). Defaults are
    # cheap enough for every run: 5 held-out validation points and 2
    # held-out analysis batches per seed, both on dedicated RNG streams.
    # Set MQAR_VAL_EVERY=0 / MQAR_ERROR_BATCHES=0 to disable.
    val_every = parse_int_env('MQAR_VAL_EVERY', max(1, steps // 4),
                              min_value=0, logger=logger)
    val_batches = parse_int_env('MQAR_VAL_BATCHES', 2, min_value=1,
                                logger=logger)
    val_batch = parse_int_env('MQAR_VAL_BATCH', 32, min_value=1,
                              logger=logger)
    error_batches = parse_int_env('MQAR_ERROR_BATCHES', 2, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('MQAR_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    logger.info(f'  n_seeds       : {n_seeds}')
    logger.info(f'  steps         : {steps}  (softmax: {softmax_steps})')
    logger.info(f'  val monitor   : every {val_every} steps '
                f'({val_batches}x{val_batch}; stream +{_VAL_MONITOR_SEED_OFFSET})')
    logger.info(f'  bad-case      : {error_batches}x{error_batch} held-out '
                f'samples per seed (stream +{_ERROR_ANALYSIS_SEED_OFFSET})')
    logger.info(f"  mamba_backend : {os.environ.get('MAMBA_BACKEND', 'reference')}")
    logger.info(f"  rwkv_backend  : {os.environ.get('RWKV_BACKEND', 'reference')}")
    logger.info(f"  rwkv7_binding : {RWKV7_UPSTREAM_BINDING['parameterization_init']['repo']}"
                f"@{RWKV7_UPSTREAM_BINDING['parameterization_init']['commit'][:12]}"
                f" ({RWKV7_UPSTREAM_BINDING['variant']})")

    # Pre-compute the (task, n_kv) sweep cells so every downstream count
    # (Bonferroni family size, log lines, result grouping) matches what
    # actually runs.
    sweep = []
    for task, n_kv in _sweep_cells(task_list, n_kv_list):
        spec = SYNTHETIC_TASKS[task]
        sweep.append((task, n_kv, spec['seq_len'], spec['vocab']))

    # ------------------------------------------------------------------
    # Non-learned rule-based reference (baseline-ladder rung 0): the
    # simplest possible "model" is not a network at all but the exact
    # symbolic rule the synthetic task is built around. Evaluated once on
    # a dedicated stream (+8M), before any training, so it cannot perturb
    # the sweep; disabled with MQAR_RULE_BASELINE=0. See evaluate_rule_
    # baseline's docstring for why this is a ceiling reference, NOT the
    # comparison baseline for superiority claims.
    # ------------------------------------------------------------------
    rule_baselines: dict = OrderedDict()
    rule_baseline_enabled = parse_int_env(
        'MQAR_RULE_BASELINE', 1, min_value=0, logger=logger) != 0
    if rule_baseline_enabled:
        logger.info('\n' + '=' * 70)
        logger.info('Non-learned rule-based reference (task ceiling; '
                    'dedicated stream +8M)')
        logger.info('=' * 70)
        for task, n_kv, task_seq_len, task_vocab in sweep:
            rec = evaluate_rule_baseline(
                task, n_kv, task_seq_len, task_vocab, device)
            rule_baselines[(task, n_kv)] = rec
            logger.info(
                f"  [rule] {task:10s} n_kv={n_kv}: {rec['rule']:20s} "
                f"acc={rec['accuracy']:.4f} "
                f"(chance={rec['chance_acc']:.4f}, "
                f"n={rec['n_sequences']})")
        logger.info('=' * 70)

    overfit_gate = _run_overfit_gate(
        sweep=sweep, ops_list=ops_list, task_list=task_list, device=device)
    overfit_check_enabled = overfit_gate['enabled']
    overfit_strict = overfit_gate['strict']
    overfit_steps = overfit_gate['steps']
    overfit_seed = overfit_gate['seed']
    gate_train_batch = overfit_gate['train_batch']
    overfit_reports = overfit_gate['reports']
    n_gate_failed = overfit_gate['n_failed']
    if overfit_gate['aborted']:
        return 1


    # ------------------------------------------------------------------
    # Tuned-baseline protocol (runs BEFORE the sweep): select each baseline
    # operator's lr on the validation stream, then use it for that op's
    # rows in the sweep. The selection runs use the shared train_multi_seed
    # path, so "tuned" means "same recipe, more lr candidates", not a
    # bespoke pipeline. Disabled with MQAR_BASELINE_TUNE=0 (historical
    # behavior: fixed lr=3e-3 for every op).
    # ------------------------------------------------------------------
    tuned_lrs = {}
    baseline_tuning = {}
    if baseline_tune_enabled and baseline_ops:
        tune_seeds = parse_int_env('MQAR_BASELINE_TUNE_SEEDS',
                                   min(n_seeds, 3),
                                   min_value=1, logger=logger)
        logger.info('\n' + '=' * 70)
        logger.info('Tuned-baseline protocol: selecting baseline lr on the '
                    f'validation stream (candidates={baseline_lrs}, '
                    f'seeds={tune_seeds})')
        logger.info('=' * 70)
        for op in baseline_ops:
            best_lr, rec = tune_baseline_lr(
                op, baseline_lrs, sweep, steps=steps,
                softmax_steps=softmax_steps, n_seeds=tune_seeds,
                device=device, val_every=val_every, val_batches=val_batches,
                val_batch=val_batch)
            tuned_lrs[op] = best_lr
            rec['grid_size'] = len(baseline_lrs)
            baseline_tuning[op] = rec
            logger.info(f'  [tune] {op}: selected lr={best_lr:g}')
        logger.info('=' * 70)

    all_results = []
    for task, n_kv, task_seq_len, task_vocab in sweep:
        spec = SYNTHETIC_TASKS[task]
        chance = 1.0 / task_vocab
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  task = {task} (n_kv={n_kv}, seq_len={task_seq_len}, '
                    f'vocab={task_vocab}) — {spec["description"]}')
        logger.info(f'  chance accuracy = {chance:.4f}')
        logger.info(f'{"=" * 70}')
        for op in ops_list:
            logger.info(f'\nTraining {op} (task={task}, n_kv={n_kv}, '
                        f'{n_seeds} seeds)...')
            # Per-operator try/except so ONE failing operator (all seeds
            # diverged / OOMed) does not crash the whole experiment and
            # lose the other operators' results. The error is logged and
            # recorded as a stub result so the JSON file is always written
            # and downstream figure generation can skip the missing op
            # gracefully (mirrors run_ablation.py's per-ratio try/except).
            try:
                r = train_multi_seed(op, n_seeds=n_seeds, steps=steps,
                                     softmax_steps=softmax_steps, device=device,
                                     n_kv=n_kv, vocab=task_vocab,
                                     seq_len=task_seq_len, task=task,
                                     lr=tuned_lrs.get(op, 3e-3),
                                     val_every=val_every,
                                     val_batches=val_batches,
                                     val_batch=val_batch,
                                     error_batches=error_batches,
                                     error_batch=error_batch)
                # Attach the non-learned rule reference for this cell next
                # to the row it contextualizes (see evaluate_rule_baseline:
                # ceiling reference, not the tuned comparison baseline).
                r['rule_baseline'] = rule_baselines.get((task, n_kv))
                all_results.append(r)
                # ci95_acc may be None when only one seed survived (see
                # train_multi_seed). Render as 'n/a' instead of crashing on
                # ``f'{None:.4f}'``. std_acc/std_loss are None in the same
                # n==1 case (the sample std is undefined, not zero).
                ci_str = f"{r['ci95_acc']:.4f}" if r['ci95_acc'] is not None else 'n/a'
                std_str = f"{r['std_acc']:.4f}" if r['std_acc'] is not None else 'n/a'
                logger.info(f"  -> mean_acc={r['mean_acc']:.4f} +/- {ci_str} "
                            f"(std={std_str}, t_vs_chance={_fmt_tstat(r['t_stat_vs_chance'], width=0, prec=2)})")
                # Curve verdicts: over/underfitting is a property of the
                # training CURVE, reported per op so a good mean accuracy
                # from an overfit best-epoch cannot go unnoticed.
                cd = r.get('curve_diagnosis_summary') or {}
                ea = r.get('error_analysis')
                logger.info(
                    f"  -> curves: {cd.get('verdicts', {})}; "
                    f"bad-case error rate: "
                    f"{(ea or {}).get('error_rate') if ea else 'n/a'}"
                    + (f" ({ {c: v['count'] for c, v in (ea or {}).get('pooled_categories', {}).items() if v['count']} })"
                       if ea else ''))
            except Exception as e:
                import traceback as _tb
                logger.error(f"  op '{op}' FAILED: {e}")
                _tb.print_exc()
                # Error stub MUST include every key that success rows
                # carry, set to None/empty/zero as appropriate, so
                # downstream JSON consumers (make_figures.py, pandas
                # DataFrames, etc.) iterating over fields do not KeyError
                # on error rows. The success-row schema is defined by
                # train_multi_seed's return dict — mirror it exactly.
                all_results.append({
                    'op': op,
                    'task': task,
                    'n_kv': n_kv,
                    'n_params': None,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_loss': None,
                    'std_loss': None,
                    'ci95_loss': None,
                    'chance_acc': 1.0 / task_vocab,
                    't_stat_vs_chance': None,
                    'lr': tuned_lrs.get(op, 3e-3),
                    'mean_train_time_s': None,
                    'mean_eval_ms_per_sample': None,
                    'curve_diagnosis_summary': None,
                    'mean_generalization_gap': None,
                    'mean_val_loss_regret_from_best': None,
                    'n_seeds_with_positive_val_regret': 0,
                    'accuracy_regime_summary': None,
                    'error_analysis': None,
                    'mean_final_metrics': None,
                    'pareto': None,
                    'rule_baseline': rule_baselines.get((task, n_kv)),
                    'per_seed': [],
                })

    # Summary table (grouped by task + n_kv). The lr column makes the
    # tuned-baseline protocol visible in the headline table: baseline ops
    # may show a non-default lr selected on the validation stream.
    print('\n' + '=' * 121)
    print(f"{'task':>10} | {'n_kv':>4} | {'op':>10} | {'lr':>7} | {'chance':>7} | {'mean_acc':>10} | "
          f"{'f1_macro':>10} | {'pr_auc_ovr':>10} | {'+/- CI95':>10} | {'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 121)
    for r in all_results:
        # Error rows have None for all numeric fields; render them as
        # dashes instead of crashing on ``f'{None:.4f}'``.
        if 'error' in r:
            err_msg = (r.get('error') or '')[:40]
            print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
                  f"{'-':>7} | {'-':>7} | {'-':>10} | {'-':>10} | {'-':>10} | "
                  f"{'-':>10} | {'-':>8} | {'-':>12} | "
                  f"{'-':>10}   ERROR: {err_msg}")
            continue
        # ci95_acc / std_acc may be None when only one seed survived;
        # render 'n/a' rather than crashing on ``f'{None:.4f}'``.
        if r['ci95_acc'] is not None:
            ci_str = f"{r['ci95_acc']:>10.4f}"
        else:
            ci_str = f"{'n/a':>10}"
        if r['std_acc'] is not None:
            std_str = f"{r['std_acc']:>8.4f}"
        else:
            std_str = f"{'n/a':>8}"
        lr_str = f"{r.get('lr', 3e-3):>7.1e}"
        # Imbalance-robust metrics live in mean_final_metrics (populated
        # whenever collect_predictions=True, which main() always sets).
        mfm = r.get('mean_final_metrics') or {}
        f1 = mfm.get('f1_macro_mean')
        pr = mfm.get('pr_auc_macro_ovr_mean')
        f1_str = f"{f1:>10.4f}" if f1 is not None else f"{'n/a':>10}"
        pr_str = f"{pr:>10.4f}" if pr is not None else f"{'n/a':>10}"
        print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | {r['op']:>10} | "
              f"{lr_str} | {r['chance_acc']:>7.4f} | {r['mean_acc']:>10.4f} | "
              f"{f1_str} | {pr_str} | "
              f"{ci_str} | {std_str} | "
              f"{_fmt_tstat(r['t_stat_vs_chance'], width=12, prec=2)} | {r['mean_loss']:>10.4f}")
    # Fairness annotation: the softmax baseline is trained for
    # ``softmax_steps`` steps while the other ops get ``steps`` steps. This
    # is intentional (softmax needs more steps to converge above chance —
    # see the comment block at the top of this file), but a reader of the
    # summary table alone has no way to know that. Print the asymmetry
    # explicitly so cross-op comparisons are not misread as "softmax is
    # unfairly advantaged" or "softmax is just better".
    if softmax_steps != steps:
        print(f"\nNOTE: softmax trained for {softmax_steps} steps; "
              f"other ops trained for {steps} steps. See MQAR_SOFTMAX_STEPS "
              f"env var / README 'Fairness notes' section for rationale.")

    # Bonferroni correction + ``conclusions_valid`` for Exp 4, mirroring the
    # logic already present in run_ablation.py::main for Exp 5. The README
    # documents Exp 4's JSON schema as including ``conclusions_valid`` and a
    # Bonferroni-corrected t-test.
    #
    # Logic:
    #   1. Compute the number of one-sample t-tests vs chance we are running
    #      (4 ops * len(n_kv_list) = 4 by default).
    #   2. Compute the Bonferroni-corrected alpha (0.05 / n_tests) and the
    #      corresponding t-critical value (scipy if available, else None).
    #   3. For each result, set ``significant_bonferroni`` and
    #      ``t_crit_bonferroni`` fields based on the one-sided
    #      comparison ``t_stat > crit``.
    #   4. Compute an experiment-level ``conclusions_valid`` flag combining
    #      seed count, minimum surviving seeds, presence of any significant
    #      result, and the fraction of near-chance results.
    #   5. Attach ``conclusions_valid`` to every result record (matching
    #      run_ablation.py's convention) so downstream consumers can check
    #      it per-record without recomputing.
    n_tests = len(ops_list) * len(sweep)
    # The t-test is one-sided (H1: mean_acc > chance). For a one-sided
    # Bonferroni correction, the per-test alpha is 0.05 / n_tests to
    # maintain the family-wise error rate at 0.05. Using 0.025 / n_tests
    # would make the test twice as conservative as intended.
    alpha_corrected = 0.05 / n_tests
    # ``_bonferroni_crit_q`` is a module-level function with an exact
    # beta-CDF/bisection fallback when scipy is unavailable. We just detect
    # scipy availability for the log line below.
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'\n  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=exact-beta-bisection)')
    for r in all_results:
        # Skip error rows: they have t_stat_vs_chance=None already.
        if 'error' in r:
            r['t_crit_bonferroni'] = None
            r['significant_bonferroni'] = False
            continue
        t_stat = r.get('t_stat_vs_chance')
        n_ok = r.get('n_seeds_ok', 0)
        if t_stat is not None and n_ok >= 2:
            crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
            r['t_crit_bonferroni'] = crit
            # ``crit`` is None when scipy is unavailable — guard the
            # comparison to avoid TypeError (mirrors run_ablation.py).
            #
            # The test is ONE-SIDED (``t_stat > crit``): the research question
            # is "does this op learn the task ABOVE chance", which is
            # directional. A below-chance result indicates the model is
            # systematically wrong (e.g. a sign error, a reversed label, or
            # pure noise that happens to anti-correlate), NOT that the op
            # "works". The Bonferroni-corrected critical value
            # ``_bonferroni_crit_q`` is already the one-sided upper-tail
            # quantile, so the one-sided comparison is the correct use of
            # that quantile.
            r['significant_bonferroni'] = (
                crit is not None and t_stat > crit
            )
        else:
            r['t_crit_bonferroni'] = None
            r['significant_bonferroni'] = False

    # ------------------------------------------------------------------
    # Paired comparison vs the tuned reference baseline.
    #
    # The vs-chance test above only says "the op learned the task at all".
    # It does NOT support "the complex op is better than a properly tuned
    # simple one" — with enough seeds, a tiny difference can be
    # significant, and the complex model could still lose to a well-tuned
    # baseline. So every non-baseline op is additionally tested against
    # the reference baseline on the SAME (task, n_kv) cell with a
    # SEED-PAIRED one-sided t-test (same seed list => paired). The
    # family of comparisons is Bonferroni-corrected.
    #
    # Flags per row:
    #   is_baseline          op is a baseline rung (no pairwise test)
    #   vs_tuned_baseline    {reference_op, n_pairs, mean_diff, t_stat,
    #                         t_crit_bonferroni, significant, verdict}
    #   beats_tuned_baseline bool (True only when the paired test is
    #                        significant in the positive direction);
    #                        None when not testable (baseline row itself,
    #                        missing reference row, < 2 paired seeds).
    # The scientific claim "complex op X is better" requires
    # beats_tuned_baseline=True; a row significant vs chance but not vs
    # the tuned baseline is only evidence that the task is learnable.
    # ------------------------------------------------------------------
    for r in all_results:
        r['is_baseline'] = r.get('op') in baseline_ops
        r['vs_tuned_baseline'] = None
        r['beats_tuned_baseline'] = None
        # Attach the pre-registered mechanism hypothesis for this
        # (task, operator) cell, so every row carries what it was TESTING
        # (and what would falsify it) next to the number it produced.
        th = TASK_HYPOTHESES.get(r.get('task', 'mqar'))
        oh = OPERATOR_HYPOTHESES.get(r.get('op'))
        r['hypothesis'] = ({'task': th, 'operator': oh}
                           if (th or oh) else None)
    baseline_pairwise = {
        'reference_op': baseline_ref,
        'baseline_ops': list(baseline_ops),
        'n_tests': 0,
        'alpha_bonferroni': None,
        'comparisons': {},
        'note': ('Seed-paired one-sided t-test of each non-baseline op vs '
                 'the reference baseline on the same (task, n_kv) cell '
                 '(H1: mean per-seed accuracy difference > 0), '
                 'Bonferroni-corrected over all comparisons. '
                 'beats_tuned_baseline=True is required before claiming '
                 'a complex operator is better than the simple baseline.'),
    }
    if baseline_ref is not None:
        ref_by_cell = {
            (r.get('task', 'mqar'), r['n_kv']): r
            for r in all_results
            if r.get('op') == baseline_ref and 'error' not in r
            and r.get('mean_acc') is not None
        }
        comparisons = []
        for r in all_results:
            # Every baseline rung (not only the reference one) is a
            # comparison reference, never a complex candidate: including
            # e.g. ``linear`` in the vs-softmax family would both give it
            # a misleading ``beats_tuned_baseline=False`` and inflate the
            # Bonferroni family.
            if 'error' in r or r.get('op') in baseline_ops:
                continue
            ref = ref_by_cell.get((r.get('task', 'mqar'), r['n_kv']))
            if ref is None:
                r['vs_tuned_baseline'] = {
                    'reference_op': baseline_ref,
                    'n_pairs': 0,
                    'mean_diff': None,
                    't_stat': None,
                    'significant': False,
                    'verdict': 'reference_row_unavailable',
                }
                continue
            cand = _finite_per_seed_acc(r.get('per_seed', []))
            base = _finite_per_seed_acc(ref.get('per_seed', []))
            common = sorted(set(cand) & set(base))
            if len(common) < 2:
                r['vs_tuned_baseline'] = {
                    'reference_op': baseline_ref,
                    'n_pairs': len(common),
                    'mean_diff': None,
                    't_stat': None,
                    'significant': False,
                    'verdict': 'insufficient_paired_seeds',
                }
                continue
            stats = _paired_one_sided_t([cand[s] - base[s]
                                         for s in common])
            stats['reference_op'] = baseline_ref
            comparisons.append((r, stats))
        n_pair_tests = len(comparisons)
        # One-sided family-wise alpha 0.05 over the comparison family: the
        # per-test alpha is 0.05 / n_pair_tests. The critical value is then
        # the one-sided upper-tail quantile at that corrected alpha with the
        # COMPARISON's own dof (n_pairs - 1) — mirroring
        # ``run_ablation._annotate_contrast_significance``.
        alpha_pair = (0.05 / n_pair_tests) if n_pair_tests else None
        for r, stats in comparisons:
            t = stats.get('t_stat')
            n_pairs = stats.get('n_pairs', 0)
            crit_pair = (
                _bonferroni_crit_q(n_pairs, alpha=alpha_pair)
                if (alpha_pair is not None and n_pairs >= 2) else None)
            sig = False
            if stats.get('mean_diff') is not None and stats['mean_diff'] > 0:
                if t is not None and crit_pair is not None and t > crit_pair:
                    sig = True
                elif (t is None and stats.get('all_positive')
                      and n_pairs >= 2):
                    # Zero-variance unanimous improvement (every seed
                    # strictly better): the exact one-sided sign-flip
                    # p-value of the all-positive pattern is 2^{-n}; it
                    # counts as significant only when that clears the
                    # Bonferroni-corrected alpha (recorded via t_stat=None).
                    exact_p = 2.0 ** (-n_pairs)
                    sig = bool(alpha_pair is not None
                               and exact_p <= alpha_pair)
            stats['t_crit_bonferroni'] = crit_pair
            stats['significant'] = bool(sig)
            stats['verdict'] = ('beats_tuned_baseline' if sig
                                else 'not_significantly_better')
            r['vs_tuned_baseline'] = stats
            r['beats_tuned_baseline'] = bool(sig)
            baseline_pairwise['comparisons'][
                f"{r.get('task', 'mqar')}/n_kv={r['n_kv']}/{r['op']}"] = stats
        baseline_pairwise['n_tests'] = n_pair_tests
        baseline_pairwise['alpha_bonferroni'] = (
            (0.05 / n_pair_tests) if n_pair_tests else None)
        if n_pair_tests:
            n_beat = sum(1 for r in all_results
                         if r.get('beats_tuned_baseline'))
            logger.info(f'\n  paired vs tuned baseline ({baseline_ref}): '
                        f'{n_beat}/{n_pair_tests} comparisons significant '
                        f'(Bonferroni alpha='
                        f'{baseline_pairwise["alpha_bonferroni"]:.4f})')
            for key, stats in baseline_pairwise['comparisons'].items():
                logger.info(
                    f"    {key}: d_acc={stats['mean_diff']:+.4f} "
                    f"(n={stats['n_pairs']}, "
                    f"t={_fmt_tstat(stats['t_stat'], width=0, prec=2)}) "
                    f"-> {stats['verdict']}")
            if n_beat == 0:
                logger.warning(
                    '  WARNING: no operator significantly beats the TUNED '
                    'baseline. Complex-operator superiority is NOT '
                    'supported on this cell; treat any raw ranking as '
                    'inconclusive (the baseline may simply be strong '
                    'enough here).')
        # Printed baseline-comparison table (stdout, not just logs).
        print('\nPaired vs tuned baseline (one-sided, Bonferroni-corrected):')
        print(f"{'task':>10} | {'n_kv':>4} | {'op':>10} | {'ref':>8} | "
              f"{'d_acc':>8} | {'n':>3} | {'t':>7} | verdict")
        for r in all_results:
            st = r.get('vs_tuned_baseline')
            if 'error' in r or not st:
                continue
            d = st.get('mean_diff')
            d_str = f"{d:+.4f}" if d is not None else 'n/a'
            print(f"{r.get('task', 'mqar'):>10} | {r['n_kv']:>4} | "
                  f"{r['op']:>10} | {st.get('reference_op') or '-':>8} | "
                  f"{d_str:>8} | {st.get('n_pairs', 0):>3} | "
                  f"{_fmt_tstat(st.get('t_stat'), width=7, prec=2)} | "
                  f"{'YES' if st.get('significant') else 'no'}"
                  f" ({st.get('verdict')})")

    # ------------------------------------------------------------------
    # Machine-readable verdicts for the pre-registered hypotheses.
    # Each row gets ``hypothesis_checks`` (per-rule evidence + status) and
    # a compact ``hypothesis_status``; the metadata records the aggregate
    # counts. The rules are fixed in code BEFORE the numbers are read.
    # ------------------------------------------------------------------
    hypothesis_verdicts, hypothesis_counts = _hypothesis_verdicts(
        all_results, baseline_ops, baseline_ref)
    logger.info(
        '  hypothesis verdicts: '
        + ', '.join(f'{k}={v}' for k, v in hypothesis_counts.items())
        + ' (row counts)')
    for name, rec in hypothesis_verdicts.items():
        logger.info(f'    [hypothesis] {name}: {rec["status"]}'
                    + (' (FALSIFIED)' if rec.get('falsified') else ''))

    # ------------------------------------------------------------------
    # Seed-paired ALL-PAIRS operator comparisons (Holm-corrected).
    #
    # Completes the statistical claim ladder:
    #   1. significant_bonferroni          -> the op learned the task;
    #   2. beats_tuned_baseline            -> it beats a tuned simple model;
    #   3. pairwise_significant_wins       -> it beats a SPECIFIC complex
    #      operator on the same seeds, robust to multiplicity;
    #   4. pairwise_equivalent_to          -> the pre-specified-margin TOST
    #      says it is practically the SAME as that operator (a positive
    #      equivalence claim; non-significance alone never licenses it).
    # The CI-overlap check below is a heuristic companion: overlapping
    # CIs do not imply "no difference", so only the paired tests here
    # support an A-vs-B claim.
    # ------------------------------------------------------------------
    # Pre-specified SESOI for the TOST; MQAR_EQUIV_MARGIN=0 disables
    # equivalence testing. The margin is consumed BEFORE the reported
    # statistics are computed and is recorded in the metadata.
    equivalence_margin = _parse_equivalence_margin(
        'MQAR_EQUIV_MARGIN', PAIRWISE_EQUIVALENCE_MARGIN)
    op_pairwise = _pairwise_op_comparisons(
        all_results,
        alpha=float(os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')),
        equivalence_margin=equivalence_margin)
    _print_pairwise_report(op_pairwise)
    if op_pairwise['n_tests']:
        logger.info(
            f'\n  seed-paired op-vs-op comparisons: '
            f"{op_pairwise['n_significant']}/{op_pairwise['n_tests']} "
            f"significant at Holm alpha={op_pairwise['alpha']} "
            f"({op_pairwise['n_skipped']} pair(s) skipped: <2 common seeds)")
        _resolution = op_pairwise.get('resolution') or {}
        if _resolution.get('n_resolution_limited'):
            logger.warning(
                f"  RESOLUTION LIMIT: "
                f"{_resolution['n_resolution_limited']}/"
                f"{_resolution['n_pairs']} pair(s) cannot be the first "
                f"rejection in this Holm family (smallest attainable "
                f"adjusted p="
                f"{_resolution.get('worst_min_attainable_p_holm')!r}). "
                f"With equal seed counts, no rejection is possible; a "
                f"'not_significant' verdict is forced by the exact test's "
                f"discreteness, not evidence of no difference. Add seed "
                f"pairs or reduce the family before claiming a "
                f"null/equivalence result.")
        if equivalence_margin is not None:
            logger.info(
                f"  TOST equivalence (pre-specified margin="
                f"{equivalence_margin:g}): "
                f"{op_pairwise['n_equivalent']}/{op_pairwise['n_tests']} "
                f"pair(s) practically equivalent at Holm "
                f"alpha={op_pairwise['alpha']}")
        else:
            logger.info('  TOST equivalence: disabled (MQAR_EQUIV_MARGIN=0)')
        logger.info(
            f"  single-seed robustness (leave-one-seed-out): "
            f"{op_pairwise['n_direction_fragile']} pair(s) direction-fragile, "
            f"{op_pairwise['n_boundary_sensitive']} boundary-sensitive, "
            f"{op_pairwise['n_seed_influence_stable']} stable "
            f"(descriptive; not a test)")

    # Experiment-level statistical validity summary. Mirrors run_ablation.py
    # but with the appropriate thresholds for the 4-op MQAR sweep.
    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    # Include error rows: they carry n_seeds_ok=0, so a fully-failed op
    # forces min_seeds_ok=0 and thus conclusions_valid=False (mirrors the
    # run_mechanism_ablation.py convention of taking the min over ALL rows).
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results),
                       default=0)
    # Split "near chance" (slightly above chance, not yet significant) from
    # "far below chance" (training diverged / recipe broken). Chance is
    # PER-ROW (each task has its own vocab), so use r['chance_acc']:
    #   * near_chance:   0.5*chance < acc < 1.5*chance  (statistically ~chance)
    #   * far_below_chance: acc <= 0.5*chance           (training/recipe broken)
    # Any far_below row forces conclusions_valid=False regardless of other
    # counts, since it indicates a real problem rather than weak power.
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and r.get('chance_acc') is not None
                   and 0.5 * r['chance_acc'] < r['mean_acc']
                   < 1.5 * r['chance_acc']]
    far_below = [r for r in all_results
                 if 'error' not in r
                 and r.get('mean_acc') is not None
                 and r.get('chance_acc') is not None
                 and r['mean_acc'] <= 0.5 * r['chance_acc']]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0
                         and len(near_chance) < len(all_results) // 2
                         and len(far_below) == 0)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ops with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
    logger.info(f'  ops near chance (0.5x-1.5x): {len(near_chance)}/{len(all_results)}')
    logger.info(f'  ops far below chance (<=0.5x): {len(far_below)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    # Baseline-ladder verdict: how many complex ops actually clear the
    # tuned reference baseline (the claim-supporting test), separate from
    # the vs-chance validity above.
    n_beat_baseline = sum(1 for r in all_results
                          if r.get('beats_tuned_baseline'))
    n_complex_rows = sum(1 for r in all_results
                         if 'error' not in r and not r.get('is_baseline'))
    if baseline_pairwise.get('n_tests'):
        logger.info(
            f'  complex ops beating tuned baseline '
            f'({baseline_pairwise["reference_op"]}): {n_beat_baseline}/'
            f'{n_complex_rows} rows '
            f'({baseline_pairwise["n_tests"]} Bonferroni-corrected paired '
            f'tests)')
    else:
        logger.info('  paired vs tuned baseline: not run '
                    '(no reference baseline row)')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: The MQAR results do NOT support strong structural\n'
            '  conclusions. Either the seed count is too low (<5), no op\n'
            '  reaches Bonferroni significance, or most accuracies are near\n'
            '  chance. Treat the ranking as exploratory, not confirmatory.\n'
            '  To improve power: increase MQAR_SEEDS (>=7), increase MQAR_STEPS,\n'
            '  or use a simpler task where the signal is stronger.')
    logger.info('=' * 70)
    # Attach the validity flag to every result record so downstream
    # consumers (make_figures, reports) can check it without recomputing.
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds

    # ------------------------------------------------------------------
    # Efficiency-vs-accuracy annotation (per task/n_kv group).
    #
    # "Report the gain WITH its cost": for every sweep cell, mark each op
    # as Pareto-optimal or dominated on (mean_acc maximise; parameters,
    # wall-clock train time and per-sample inference latency minimise).
    # Accuracy gains that come at a disproportionate cost are visible
    # directly in the record instead of requiring a cross-reference to the
    # latency/cache experiments.
    #
    # The cross-op claim check answers "is the winner actually better?":
    # when the best op's 95% CI overlaps another op's CI, the ranking
    # between them is not statistically supported at this seed count, and
    # the record says so explicitly (microscopic, non-significant gains
    # must not be over-claimed).
    # ------------------------------------------------------------------
    groups: dict = OrderedDict()
    for r in all_results:
        groups.setdefault((r.get('task', 'mqar'), r['n_kv']), []).append(r)

    pareto_reports = {}
    cross_op_checks = {}
    for (task_name, n_kv_key), rows in groups.items():
        cell_key = f'{task_name}/n_kv={n_kv_key}'
        cell_res = ((op_pairwise.get('cells', {}).get(cell_key) or {})
                    .get('resolution'))
        live = [r for r in rows if 'error' not in r
                and r.get('mean_acc') is not None
                and r.get('mean_train_time_s') is not None
                and r.get('n_params') is not None]
        if live:
            rep = annotate_rows_with_pareto(
                live, accuracy_key='mean_acc',
                cost_keys=('mean_train_time_s', 'n_params',
                           'mean_eval_ms_per_sample'),
                name_key='op')
            pareto_reports[f'{task_name}/n_kv={n_kv_key}'] = {
                'frontier': rep['frontier'],
                'dominated_by': rep['dominated_by'],
                'cost_keys': rep['cost_keys'],
                'points': {
                    op: {'accuracy': p['accuracy'], **p['costs'],
                         'acc_per_train_second': p['acc_per_train_second'],
                         'acc_per_1k_params': p['acc_per_1k_params'],
                         'acc_per_eval_ms': p.get('acc_per_eval_ms')}
                    for op, p in rep['points'].items()
                },
                'skipped': rep['skipped'],
            }
            best = max(live, key=lambda r: r['mean_acc'])
            overlaps = []
            for r in live:
                if r is best:
                    r['ci95_overlaps_best'] = False
                    continue
                ci_b, ci_r = best.get('ci95_acc'), r.get('ci95_acc')
                if ci_b is None or ci_r is None:
                    r['ci95_overlaps_best'] = None
                    continue
                overlap = abs(best['mean_acc'] - r['mean_acc']) <= (ci_b + ci_r)
                r['ci95_overlaps_best'] = bool(overlap)
                if overlap:
                    overlaps.append(r['op'])
            best['ci95_overlaps_best'] = False
            # Seed-paired, Holm-corrected standing of the best op (from
            # _pairwise_op_comparisons, computed above). This — not the CI
            # overlap list — is what may be claimed.
            best_sig_wins = [
                op for op in best.get('pairwise_significant_wins', [])
                if any(r.get('op') == op for r in live)]
            best_sig_wins = sorted(best_sig_wins, key=str)
            # Of those significant wins, which ones reverse direction when
            # a single seed is deleted? The pairwise verdict stays as
            # computed; the claim is qualified so a one-seed-driven win is
            # not read as a structural ordering.
            best_fragile_wins = sorted(
                [op for op in best_sig_wins
                 if op in best.get('pairwise_direction_fragile_wins', [])],
                key=str)
            best_robust_wins = [op for op in best_sig_wins
                                if op not in best_fragile_wins]
            if best_sig_wins:
                paired_claim = (
                    f"Seed-paired Holm-corrected tests support {best['op']} "
                    f"over: {', '.join(best_sig_wins)}.")
                if best_fragile_wins:
                    paired_claim += (
                        " The sign of that difference reverses when one "
                        "seed is deleted for: "
                        f"{', '.join(best_fragile_wins)} "
                        "(leave-one-seed-out; descriptive, not a test).")
            elif (cell_res is not None
                  and not cell_res.get('any_pair_testable')):
                _floor = cell_res.get('worst_min_attainable_p_holm')
                _floor_str = (f'{_floor:.4f}' if _floor is not None
                              else 'n/a')
                paired_claim = (
                    f"No seed-paired Holm-corrected test CAN separate "
                    f"{best['op']} from any other op: with "
                    f"{cell_res.get('min_n_pairs')}-"
                    f"{cell_res.get('max_n_pairs')} seed pairs across "
                    f"{cell_res.get('family_size')} comparisons, the exact "
                    f"test's smallest attainable Holm-adjusted p is "
                    f"{_floor_str} > alpha; with equal seed counts across "
                    f"the cell, the family has no rejection region. This is "
                    f"a resolution limit of the design, not evidence that "
                    f"the operators are equivalent.")
            else:
                paired_claim = (
                    f"No seed-paired Holm-corrected test separates "
                    f"{best['op']} from any other op at this seed count.")
            # Positive equivalence standing (only when TOST is enabled):
            # ops the best op is PRACTICALLY THE SAME as, within the
            # pre-specified margin. This is the allowed counterpart of
            # "no significant difference".
            best_equiv_to = [
                op for op in best.get('pairwise_equivalent_to', [])
                if any(r.get('op') == op for r in live)]
            best_equiv_to = sorted(best_equiv_to, key=str)
            equiv_claim = ''
            if equivalence_margin is not None:
                if best_equiv_to:
                    equiv_claim = (
                        f"Within the pre-specified margin "
                        f"{equivalence_margin:g}, the TOST declares it "
                        f"practically equivalent to: "
                        f"{', '.join(best_equiv_to)}.")
                else:
                    equiv_claim = (
                        'No op is declared practically equivalent to it at '
                        'the pre-specified margin.')
            cross_op_checks[f'{task_name}/n_kv={n_kv_key}'] = {
                'best_op': best['op'],
                'best_mean_acc': best['mean_acc'],
                'best_ci95_acc': best.get('ci95_acc'),
                'ops_with_overlapping_ci95': overlaps,
                'best_op_pairwise_significant_wins': best_sig_wins,
                'best_op_single_seed_robust_wins': best_robust_wins,
                'best_op_direction_fragile_wins': best_fragile_wins,
                'best_op_practically_equivalent_to': (
                    best_equiv_to if equivalence_margin is not None
                    else None),
                'family_resolution': cell_res,
                'paired_test_note': (
                    'CI overlap is a continuity heuristic, not a test: '
                    'overlapping CIs do not imply a non-significant '
                    'difference. The authoritative pairwise evidence is '
                    'metadata.op_pairwise_comparisons (seed-paired '
                    'permutation tests AND pre-specified-margin TOST, '
                    'Holm-corrected per cell).'),
                'claim': (
                    f"{best['op']} has the highest mean accuracy. "
                    f"{paired_claim} "
                    + (f"Its 95% CI overlaps {overlaps}."
                       if overlaps else
                       "Its 95% CI is disjoint from every other op.")
                    + (f" {equiv_claim}" if equiv_claim else '')),
            }
            if not best_sig_wins:
                logger.warning(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: best='
                    f"{best['op']} (acc={best['mean_acc']:.4f}) has no "
                    f"significant seed-paired win (Holm) — treat the "
                    f"ranking as inconclusive.")
            elif best_fragile_wins:
                logger.warning(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: best='
                    f"{best['op']} — the significant direction vs "
                    f"{', '.join(best_fragile_wins)} reverses when one seed "
                    f'is deleted (leave-one-seed-out); do not report that '
                    f'ordering as structural.')
            dominated = rep['dominated_by']
            if dominated:
                logger.info(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: Pareto '
                    f'dominated: '
                    + ', '.join(f'{a}<-{b}' for a, b in dominated.items()))
            else:
                logger.info(
                    f'  [cost/claim] {task_name}/n_kv={n_kv_key}: all live '
                    f'ops on the accuracy/cost Pareto frontier')

    # Post-run analysis-population audit. The prospective audit above only
    # knows the REQUESTED seed budget. Seeds that crashed, diverged or
    # produced a non-finite metric are filtered out of the aggregates, so
    # the run that actually contributes evidence can be smaller than the
    # pre-registered block. Passing the common surviving seed set turns
    # that silent complete-case analysis into an explicit
    # ``incomplete_attrition`` label (the planned power is still reported
    # beside the realized power at the analyzed n).
    analyzed_population = common_analyzed_seeds(
        all_results, finite_keys=('final_acc', 'final_loss'))
    experiment_design_audit = audit_experiment_design(
        'exp4_mqar',
        seeds=[42 + i for i in range(n_seeds)],
        levels={'operator': list(ops_list), 'task': list(task_list),
                'n_kv': list(n_kv_list)},
        family_size=design_family_size,
        analyzed_seeds=analyzed_population,
    )
    logger.info(
        f'  design audit (post-run): {experiment_design_audit["status"]} '
        f'(requested={experiment_design_audit["n_seeds"]}, '
        f'analyzed={experiment_design_audit["n_analyzed"]}, '
        f'missing={experiment_design_audit["analysis_population"]["n_missing"]})')
    if experiment_design_audit['status'] == 'incomplete_attrition':
        logger.warning(
            '  [design] failed seeds were dropped from the analysis; the '
            'reported estimates are complete-case, not the pre-registered '
            'complete block. See metadata.experiment_design.'
            'analysis_population.')

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a divergent seed slipped
    # past the NaN guard and the per_seed filter, Python's default
    # json.dump would emit literal ``NaN``/``Infinity`` tokens, which are
    # INVALID JSON per RFC 8259 and cause strict parsers (js, jq, pandas
    # with ``orient='records'``) to reject the whole file. With
    # allow_nan=False the call raises ValueError instead — surfacing the
    # corruption loudly rather than shipping a broken file.
    #
    # CRITICAL: serialize to a STRING first (json.dumps), then write the
    # string to the file, so a mid-write ValueError (NaN) cannot leave a
    # PARTIAL JSON document behind. Serializing to a string first
    # guarantees atomicity: either the complete JSON is written or nothing
    # is.
    # The payload is a single top-level object
    # ``{"metadata": ..., "results": [...]}``; ``make_figures.load`` also
    # accepts the legacy bare-array format. ``write_json_atomic`` below
    # writes the file atomically (temp + fsync + os.replace).
    # ``write_json_atomic`` writes atomically (temp file + fsync +
    # os.replace) so a process kill or disk-full mid-write leaves the
    # target file as the OLD version (or absent) rather than a truncated
    # partial JSON document — see kaggle_setup.write_json_atomic's
    # docstring for the full rationale. It serializes to a string first,
    # raising ValueError on any non-serializable value BEFORE touching the
    # filesystem.
    #
    # Output split: the canonical MQAR rows keep the historical
    # results/exp4_mqar.json file (name, schema, and figure consumers
    # unchanged); rows from the additional probes go to a separate
    # results/exp4b_quality_tasks.json so the two result families never
    # blur together. When MQAR is not in the task list, the historical
    # file is left untouched rather than overwritten with an empty result
    # set (a stale-but-valid file beats a silently emptied one).
    def _base_metadata():
        return {
            'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
            'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
            'rwkv_backend': os.environ.get('RWKV_BACKEND', 'reference'),
            'rwkv7_upstream_binding': RWKV7_UPSTREAM_BINDING,
            'ops': list(ops_list),
            'tasks': list(task_list),
            # Reproducibility block: code version, environment, seeds, and
            # the exact training recipe. Without these a results file cannot
            # be traced back to the run that produced it.
            'provenance': capture_provenance(),
            'seeds': [42 + i for i in range(n_seeds)],
            'n_seeds': n_seeds,
            # Runtime audit of the pre-registered Exp 4 design: status,
            # required vs actual seed budget, per-test alpha, MDE and the
            # claims this run is allowed to support (see experiment_design).
            'experiment_design': experiment_design_audit,
            # Data version block: this experiment's "dataset" is synthetic
            # and fully determined by (generator function, task spec, seed
            # + offset). Recording all three makes the data version explicit
            # instead of implicit in the code revision.
            'data': {
                'kind': 'synthetic',
                'generator_functions': {
                    t: SYNTHETIC_TASKS[t]['make'].__name__
                    for t in task_list
                },
                'task_specs': {
                    t: {
                        'seq_len': SYNTHETIC_TASKS[t]['seq_len'],
                        'vocab': SYNTHETIC_TASKS[t]['vocab'],
                        'n_kv_swept': bool(SYNTHETIC_TASKS[t].get('n_kv_swept')),
                    }
                    for t in task_list
                },
                'seed_offsets': {
                    'train': 1_000_000,
                    'eval': 5_000_000,
                    'val_monitor': _VAL_MONITOR_SEED_OFFSET,
                    'error_analysis': _ERROR_ANALYSIS_SEED_OFFSET,
                    'overfit_gate': _OVERFIT_BATCH_SEED_OFFSET,
                    'rule_baseline': _RULE_BASELINE_SEED_OFFSET,
                },
                'note': ('Synthetic batches are a deterministic function of '
                         '(task generator, task spec, seed + offset); no '
                         'external dataset version applies. Code revision in '
                         'provenance pins the generator implementation.'),
            },
            # Train/validation/test separation, spelled out because the
            # monitoring and bad-case analysis must not contaminate the
            # reported metric:
            #   train (+1M)          -> fits the model
            #   val monitor (+6M)    -> held-out curve for the over/underfit
            #                           diagnosis and error analysis; read-only
            #   headline eval (+5M)  -> the reported final_acc/final_loss
            #   bad cases (+7M)      -> error categorization examples
            'monitoring': {
                'val_every': val_every,
                'val_batches': val_batches,
                'val_batch': val_batch,
                'error_batches': error_batches,
                'error_batch': error_batch,
                'note': ('The validation-monitoring and error-analysis '
                         'streams are disjoint from both the training and '
                         'the headline-eval streams, and all passes are '
                         'read-only with dedicated generators: enabling '
                         'monitoring cannot change any reported metric.'),
            },
            # Overfit sanity gate verdicts (compact; full curves with the
            # per-step history live in results/exp4_overfit_gate.json).
            'overfit_sanity_check': {
                'enabled': bool(overfit_check_enabled),
                'strict': bool(overfit_strict),
                'steps': overfit_steps,
                'seed': overfit_seed,
                'train_batch': gate_train_batch,
                'reports': [_compact_overfit_report(r)
                            for r in overfit_reports],
                'note': ('Necessary but NOT sufficient: passing proves '
                         'gradients flow and the loss responds on a fixed '
                         'batch; it cannot rule out data-augmentation / '
                         'feature-alignment / label-mapping bugs.'),
            },
            'training': {
                'optimizer': 'AdamW',
                # Historical fixed lr; the per-op EFFECTIVE lr is in
                # lr_by_op (baseline ops use their lr selected on the
                # validation stream when MQAR_BASELINE_TUNE=1).
                'lr': 3e-3,
                'lr_by_op': {op: tuned_lrs.get(op, 3e-3)
                             for op in ops_list},
                'weight_decay_decay_group': 0.01,
                'weight_decay_no_decay_group': 0.0,
                'grad_clip_norm': 1.0,
                'steps': steps,
                'softmax_steps': softmax_steps,
                'batch_gen_seed_offset': 1_000_000,
                'eval_gen_seed_offset': 5_000_000,
            },
            # Efficiency-vs-accuracy: per sweep cell, which ops are Pareto
            # optimal / dominated on (accuracy, params, train time, eval
            # latency), plus the cost-normalized accuracies. Wall-clock
            # timing is a same-run proxy (shared CI hardware), not a kernel
            # benchmark: the dedicated latency experiment (Exp 2/6) remains
            # the authority for latency claims.
            'efficiency_accuracy': {
                'pareto': pareto_reports,
                'cost_note': ('cost axes are within-run wall-clock train '
                              'time, parameter count and per-sample eval '
                              'latency; timing is noisy on shared hardware '
                              'and is a screening proxy, not a latency '
                              'benchmark (use Exp 2/6 for latency claims).'),
                'accessor': ('per-row `pareto` = {optimal, dominated_by, '
                             'acc_per_train_second, acc_per_1k_params, '
                             'acc_per_eval_ms}'),
            },
            'cross_op_claim_check': cross_op_checks,
            # Seed-paired ALL-PAIRS operator comparison: per (task, n_kv)
            # cell, every unordered operator pair tested on the common
            # seeds with an exact sign-flip permutation test (primary), the
            # paired t-test (secondary), the exact sign test, bootstrap CI
            # and Cohen's d_z; p-values Holm-corrected within the cell.
            # `seeds_required_80pct_power` turns inconclusive pairs into an
            # actionable seed budget for the next multi-seed run. When the
            # equivalence margin is enabled (default), each pair ALSO
            # carries a TOST against the pre-specified margin, so "no
            # significant difference" can be separated from a positive
            # "practically equivalent" claim (and from a genuinely
            # inconclusive comparison).
            'op_pairwise_comparisons': op_pairwise,
            'equivalence_testing': {
                'enabled': equivalence_margin is not None,
                'margin': equivalence_margin,
                'margin_source': (
                    'pre_specified (PAIRWISE_EQUIVALENCE_MARGIN constant, '
                    'MQAR_EQUIV_MARGIN env override; 0 disables); never '
                    'derived from the observed CIs'),
                'method': ('two one-sided paired t-tests (TOST) per operator '
                           'pair on the common seeds; Holm-Bonferroni '
                           'within each (task, n_kv) cell'),
                'alpha': float(os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')),
                'ci_level': ((1.0 - 2.0 * float(
                    os.environ.get('MQAR_PAIRWISE_ALPHA', '0.05')))
                    if equivalence_margin is not None else None),
                'accessor': ('per-pair `tost`, `holm_adjusted_p_tost`, '
                             '`significant_equivalence_holm` and '
                             '`equivalence_verdict`; per-row '
                             '`pairwise_equivalent_to`'),
            },
            # Tuned-baseline protocol record: how the simple-baseline lrs
            # were selected (validation stream, full grid), what they were,
            # and the seed-paired comparison of every complex op against
            # the tuned reference baseline. This is the metadata that
            # supports (or refutes) "the complex model is better".
            'baseline_protocol': {
                'enabled': bool(baseline_tune_enabled and baseline_ops),
                'baseline_ops': list(baseline_ops),
                'reference_op': baseline_ref,
                'lr_grid': baseline_lrs,
                'tune_seeds': (
                    baseline_tuning[baseline_ops[0]]['tune_seeds']
                    if baseline_tuning else None),
                'selection_stream': f'val_monitor_seed+{_VAL_MONITOR_SEED_OFFSET}',
                'selected_lrs': {op: lr for op, lr in tuned_lrs.items()},
                'selection_records': baseline_tuning,
                'pairwise_vs_reference': baseline_pairwise,
                'headline_required_for_claims': (
                    'beats_tuned_baseline=True on the paired one-sided '
                    'Bonferroni-corrected test. A row only significant vs '
                    'chance may be a learnable task, not evidence that the '
                    'complex operator is better than a tuned simple one.'),
                'note': ('Model selection (lr) used ONLY the held-out '
                         'validation stream (seed +6M), never the headline '
                         'eval stream (seed +5M); the winning lr is then '
                         're-trained from scratch with the full seed count. '
                         'MQAR_BASELINE_TUNE=0 restores the historical '
                         'fixed lr=3e-3 recipe. The complex operators keep '
                         'the historical lr=3e-3: the baseline deliberately '
                         'gets the benefit of tuning, so a complex-operator '
                         'win over the tuned baseline is conservative '
                         '(against the complex op).'),
            },
            # Baseline-ladder rung 0: the non-learned symbolic rule the
            # synthetic tasks are built around (see evaluate_rule_baseline).
            # This is a task-ceiling / learnability reference, NOT a
            # competitive tuned baseline — superiority claims still require
            # beats_tuned_baseline=True against the tuned LEARNED baseline.
            'rule_baseline_reference': {
                'enabled': bool(rule_baseline_enabled),
                'seed_offset': _RULE_BASELINE_SEED_OFFSET,
                'cells': {
                    f'{task}/n_kv={n_kv}': dict(rec)
                    for (task, n_kv), rec in rule_baselines.items()
                },
                'accessor': ('per-row `rule_baseline` = the cell record '
                             '(rule name, accuracy, chance, seed stream) or '
                             'None for error stubs'),
                'role': ('non_learned_task_ceiling_reference: a '
                         'parameter-free symbolic rule evaluated once on a '
                         'dedicated RNG stream. NOT the comparison baseline '
                         'for superiority claims (those require '
                         'beats_tuned_baseline against the tuned learned '
                         'baseline); it establishes that the task is '
                         'solvable by an exact rule and quantifies how much '
                         'of it the fixed-budget trained probe recovers.'),
                'note': ('The baseline ladder is therefore: rule reference '
                         '(0 params) -> tuned learned baseline (softmax, lr '
                         'selected on the held-out stream) -> complex '
                         'operators, which must beat the TUNED baseline '
                         '(paired, Bonferroni-corrected) to support a '
                         'superiority claim.'),
            },
            # Pre-registered mechanism hypotheses (fixed in code BEFORE the
            # numbers were read): what each task probes and what signature
            # each operator is predicted to show, with the falsification
            # criterion. The sweep is hypothesis-driven, not a fishing
            # expedition; each result row carries its own copy.
            'hypotheses': {
                'tasks': {t: dict(TASK_HYPOTHESES[t])
                          for t in task_list if t in TASK_HYPOTHESES},
                'operators': {op: dict(OPERATOR_HYPOTHESES[op])
                              for op in ops_list
                              if op in OPERATOR_HYPOTHESES},
                'note': ('Pre-registered in run_quality.TASK_HYPOTHESES / '
                         'OPERATOR_HYPOTHESES. Statistical claims: a complex '
                         'op "beats" the baseline only if '
                         'beats_tuned_baseline=True (paired, Bonferroni); '
                         'a row significant vs chance but not vs the tuned '
                         'baseline only shows the task is learnable.'),
            },
            # Machine-readable rule application for the hypotheses above.
            # Every result row also carries its own ``hypothesis_status``
            # and ``hypothesis_checks`` (rule + status + evidence).
            'hypothesis_verdicts': {
                'checks': hypothesis_verdicts,
                'row_status_counts': hypothesis_counts,
                'statuses': list(_HYPOTHESIS_STATUSES),
                'accessor': ('per-row `hypothesis_status` (one of '
                             f'{list(_HYPOTHESIS_STATUSES)}) and '
                             '`hypothesis_checks` (per-rule status + '
                             'numeric evidence)'),
                'note': ('Verdicts are computed by fixed code rules, not '
                         'assigned by hand: contradicted only when the '
                         'pre-registered falsification condition fires '
                         '(a significant effect in the predicted-wrong '
                         'direction), not_supported when the test is '
                         'inconclusive, not_computable when rows or paired '
                         'seeds are missing.'),
            },
            'significance_scope': (
                'vs_chance_baseline_and_seed_paired_vs_tuned_reference_'
                'baseline_and_seed_paired_op_vs_op_holm'
                if baseline_pairwise.get('n_tests')
                else ('vs_chance_baseline_and_seed_paired_op_vs_op_holm'
                      if op_pairwise.get('n_tests')
                      else 'vs_chance_baseline_not_pairwise_between_ops')),
            'csa_indexer_trained': True,
            'csa_ste_enabled': True,
            'csa_indexer_normalize_qk': True,
            'csa_caveat': (
                "CSA's lightning indexer is trained via a straight-through "
                "estimator (STE): the forward pass uses hard top-k indices "
                "(genuine sparse selection), but the backward pass routes "
                "gradients through a differentiable soft distribution over "
                "all compressed blocks. After backward(), the indexer "
                "parameters (W_IUQ, W_w, W_KV_idx, W_Z_idx, B_idx) receive "
                "non-None .grad and are updated by the optimizer. The STE "
                "does NOT change the forward semantics — CSA is still "
                "sparse retrieval — but makes the indexer learnable."
            ),
            'training_steps_fairness': {
                'softmax_steps': softmax_steps,
                'other_ops_steps': steps,
                # Make the note CONDITIONAL on the actual asymmetry,
                # matching the summary-table note, so it never contradicts
                # the numeric fields above when softmax_steps == steps.
                'note': (
                    "The softmax baseline is trained for more steps than "
                    "the other operators so it actually converges (the "
                    "original 100 steps left softmax at ~10% accuracy, "
                    "barely above the 6.25% chance level — a useless "
                    "upper bound). Any cross-op accuracy comparison must "
                    "annotate this asymmetry. See README 'Fairness notes' "
                    "section."
                ) if softmax_steps != steps else (
                    "Default fair comparison: every operator (including "
                    "softmax) receives the same optimizer-step budget "
                    "(MQAR_STEPS). An optional longer-softmax "
                    "sensitivity run can be requested via the "
                    "MQAR_SOFTMAX_STEPS env var; in that case "
                    "softmax_steps > other_ops_steps and the asymmetry "
                    "must be annotated as a non-fixed-budget comparison. "
                    "See README 'Fairness notes' section."
                ),
            },
            'schema_version': 1,
        }

    mqar_rows = [r for r in all_results if r.get('task', 'mqar') == 'mqar']
    other_rows = [r for r in all_results if r.get('task', 'mqar') != 'mqar']
    if 'mqar' in task_list:
        payload = {'metadata': _base_metadata(), 'results': mqar_rows}
        try:
            write_json_atomic(payload, 'results/exp4_mqar.json',
                              indent=2, allow_nan=False)
        except ValueError as e:
            logger.error(f'non-finite value in payload; sanitizing to null: {e}')
            write_json_atomic(sanitize_for_json(payload),
                              'results/exp4_mqar.json',
                              indent=2, allow_nan=False)
        logger.info('\nSaved: results/exp4_mqar.json')
    if other_rows:
        payload_b = {
            'metadata': {
                **_base_metadata(),
                'task_descriptions': {
                    t: SYNTHETIC_TASKS[t]['description']
                    for t in task_list if t in SYNTHETIC_TASKS
                },
            },
            'results': other_rows,
        }
        try:
            write_json_atomic(payload_b, 'results/exp4b_quality_tasks.json',
                              indent=2, allow_nan=False)
        except ValueError as e:
            logger.error(f'non-finite value in payload; sanitizing to null: {e}')
            write_json_atomic(sanitize_for_json(payload_b),
                              'results/exp4b_quality_tasks.json',
                              indent=2, allow_nan=False)
        logger.info('Saved: results/exp4b_quality_tasks.json')
    # Return non-zero if any op's training crashed (``'error' in r``), so
    # ``run_all._run`` records the experiment as ``status='fail'`` instead
    # of treating a partial run as success (``run_all._run``'s
    # ``None == success`` contract). Returning 1 forces the failure to
    # propagate to ``run_all``'s summary and exit code.
    n_errors = sum(1 for r in all_results if 'error' in r)
    n_incomplete = sum(
        1 for r in all_results
        if 'error' not in r
        and r.get('n_seeds_ok', n_seeds) != r.get('n_seeds', n_seeds)
    )
    # Append the experiment to the shared run manifest (JSONL ledger) so a
    # campaign-level index links this run's seeds / recipe / data version to
    # its result files — automatic bookkeeping, never manual.
    quality_meta = _base_metadata()
    append_run_manifest(make_run_manifest_entry(
        experiment='exp4_quality',
        seeds=quality_meta['seeds'],
        provenance=quality_meta['provenance'],
        hyperparameters=quality_meta['training'],
        data=quality_meta['data'],
        artifacts={
            'mqar_results': ('results/exp4_mqar.json'
                             if 'mqar' in task_list else None),
            'quality_tasks_results': ('results/exp4b_quality_tasks.json'
                                      if other_rows else None),
            'overfit_gate_results': (
                'results/exp4_overfit_gate.json'
                if overfit_check_enabled else None),
        },
        results={
            'n_ops': len(all_results),
            'n_seeds': int(n_seeds),
            'errors': int(n_errors),
            'incomplete_seed_sets': int(n_incomplete),
            'overfit_gate_enabled': bool(overfit_check_enabled),
            'overfit_gate_failures': int(n_gate_failed),
        },
        status=('completed' if not (n_errors or n_incomplete)
                else 'partial'),
    ), logger=logger)
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_results)} ops errored and '
            f'{n_incomplete}/{len(all_results)} have incomplete seed sets. '
            'Returning non-zero to prevent survivor-only aggregates from '
            'being treated as a successful experiment.')
        return 1
    return 0


__all__ = [
    'main',
]
