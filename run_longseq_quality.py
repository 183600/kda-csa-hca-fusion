"""Experiment 12 — long-sequence MQAR quality comparison.

Motivation
----------
Exp 4/5/5b all run MQAR at ``SEQ_LEN=16``. At short range every operator
can solve the association task, so short-sequence accuracy does NOT
discriminate memory mechanisms: a softmax baseline can win at T=16 purely
by attending to the current token's neighborhood. The discriminative
regime is long range, where linear/recurrent operators (KDA, Mamba-2, RWKV-7)
carry O(1) state while softmax carries O(T) per-token state and compressed
operators (CSA/HCA) trade retrieval fidelity for cache bytes.

This experiment extends MQAR to long sequences (T=16..1024) so the
accuracy-vs-seq_len curve can be read as:
    * a *long-range quality* comparison (which operator degrades least as
      the cue-to-target distance grows), and
    * a *compression-robustness* comparison for CSA/HCA (does the compressed
      retrieval still hit the gold block at T=1024?).

Design
------
* Fixed 5-layer depth, same training recipe as Exp 5 (AdamW 3e-3, 100 steps
  by default, multi-seed), same VOCAB=16, n_kv=1.
* Seq lengths: {16, 64, 256, 512, 1024} (env-tunable via
  ``LS_SEQ_LENS``). At each length, every operator is trained from scratch
  over ``LS_SEEDS`` seeds paired across operators at the same seed, so the
  per-length comparison is seed-paired.
* Operators:
    * ``memory`` — KDA recurrent (3 memory layers + 1 CSA + 1 HCA, the
      3:1:1 hybrid with memory_type=kda).
    * ``csa`` — pure CSA stack (n_csa=5).
    * ``hca`` — pure HCA stack (n_hca=5).
    * ``hybrid`` — 3:1:1 KDA:CSA:HCA hybrid (the default).
    * ``softmax`` — causal dense softmax baseline (same depth/recipe).
* Primary statistic per (operator, length): mean_acc +/- CI95 vs the
  operator's own T=16 reference (seed-paired paired t-test,
  Bonferroni-corrected within the operator family), plus TOST equivalence
  at ``CONTRAST_EQUIVALENCE_MARGIN = 0.05``.
* Cross-operator contrast at each length: hybrid vs softmax, hybrid vs
  memory, CSA vs HCA (two-sided, Bonferroni across all cross-operator rows
  at that length).
* Latency probe per (operator, length) (same protocol as Exp 5's ``fwd_ms``)
  so the quality-vs-latency frontier is visible: softmax wins quality at
  short T but its per-token state grows linearly; linear operators keep
  O(1) state but may lose retrieval fidelity at long T.

Environment variables
---------------------
* ``LS_SEQ_LENS`` (default ``16,64,256,512,1024``), ``LS_SEEDS`` (default 5),
  ``LS_STEPS`` (default 100), ``LS_TRAIN_BATCH`` (default 16).
* ``LS_OPERATORS`` (default ``memory,csa,hca,hybrid,softmax``).
* ``LS_OVERFIT_CHECK`` / ``LS_OVERFIT_STRICT`` / ``LS_OVERFIT_STEPS`` /
  ``LS_OVERFIT_SEED`` / ``LS_OVERFIT_BATCH`` (mirror Exp 5's ABL_* gate).
* ``LS_DEVICE`` (default ``cpu``; ``cuda`` when a GPU is available).

Outputs
-------
``results/exp12_longseq_quality.json`` — envelope
``{metadata, results: [...]}`` with per-(operator, length) per-seed rows plus
aggregated summary rows and a ``contrasts`` block (per-length vs T=16
reference, and cross-operator at each length).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    configure_torch_for_device, parse_int_env, sanitize_for_json,
    write_json_atomic, write_results_json, capture_provenance,
    make_seeded_generator, set_global_seed, grad_flow_report,
    make_run_manifest_entry, append_run_manifest,
)
from results_schema import (RowContractError,
                            write_checked as write_results_checked)
from ops_fused import HybridAttentionStack, HybridConfig, HybridKCHAttention
from ops_kda_backend import validate_kda_backend
from run_ablation import (
    VOCAB, SEQ_LEN, CONTRAST_EQUIVALENCE_MARGIN, CAPACITY_MATCH_TOLERANCE,
    _make_cfg, _run_provenance, _run_ablation_overfit_gate,
    _attach_curve_diagnostics, _eval_model, eval_layout_one_seed,
    summarize_layout_results, run_overfit_sanity_check,
    _OVERFIT_BATCH_SEED_OFFSET, ERROR_ANALYSIS_SEED_OFFSET,
    VAL_MONITOR_SEED_OFFSET, _aggregate_curve_diagnosis,
    SimpleBaselineStack, _make_matrix_cfg,
)
from run_quality import (
    make_mqar_batch, MQARHead, _build_param_groups,
    _t_crit_975, _bonferroni_crit_q, _parse_equivalence_margin,
    analyze_errors, ResidualAttnLayer, NoMixAttn, SoftmaxAttn,
)
from results_analysis import multiclass_classification_metrics
from train_diagnostics import (
    DEFAULT_COST_KEYS, DEFAULT_MAX_COST_RATIO, DEFAULT_MIN_EFFECT,
    ERROR_ANALYSIS_SEED_OFFSET, VAL_MONITOR_SEED_OFFSET, ValPoint,
    ValidationCurve, accuracy_cost_tradeoff, log_pooled_errors,
    log_tradeoff, optimizer_update_report, pool_error_analyses,
    raise_non_finite, snapshot_parameters, summarize_gradient_norms,
    triage_gradient,
)
from multi_seed_stats import (
    paired_t_test, tost_from_summary, seeds_required, min_detectable_effect,
    _safe_mean, _sample_sd,
)
from experiment_design import (audit_run as audit_experiment_design,
                               seed_ids)

logger = logging.getLogger(__name__)

LS_PLAN_ID = 'exp12_longseq'
LS_RESULTS_FILE = 'results/exp12_longseq_quality.json'
LS_PROVENANCE_FILE = 'results/exp12_longseq_provenance.json'
LS_CONTRASTS_FILE = 'results/exp12_longseq_contrasts.json'

DEFAULT_SEQ_LENS = [16, 64, 256, 512, 1024]
DEFAULT_OPERATORS = ['memory', 'csa', 'hca', 'hybrid', 'softmax']
LS_EQUIV_MARGIN = CONTRAST_EQUIVALENCE_MARGIN

# Layout per operator (5 layers). 'memory' resolves via memory_type=kda.
LS_LAYOUT = {
    'memory': ('memory', 'memory', 'memory', 'csa', 'hca'),
    'csa': ('csa', 'csa', 'csa', 'csa', 'csa'),
    'hca': ('hca', 'hca', 'hca', 'hca', 'hca'),
    'hybrid': ('memory', 'memory', 'memory', 'csa', 'hca'),
    'softmax': ('softmax', 'softmax', 'softmax', 'softmax', 'softmax'),
}


def _parse_seq_lens(env_value: str | None) -> list[int]:
    if env_value is None:
        return list(DEFAULT_SEQ_LENS)
    out = []
    for tok in env_value.split(','):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_operators(env_value: str | None) -> list[str]:
    if env_value is None:
        return list(DEFAULT_OPERATORS)
    return [t.strip() for t in env_value.split(',') if t.strip()]


def _build_longseq_model(operator: str, seq_len: int, d_model: int,
                         device) -> tuple[nn.Module, nn.Module, nn.Embedding]:
    """Construct (model, head, embed) for one (operator, seq_len) cell.

    Uses the explicit-layout path of _make_matrix_cfg for hybrid-style
    operators and SimpleBaselineStack for softmax. The KDA memory route is
    the default memory_type, so 'memory'/'csa'/'hca'/'hybrid' all share
    the same KDA recurrent mixer; only the layout differs.
    """
    if isinstance(device, str):
        device = torch.device(device)
    layout = LS_LAYOUT[operator]
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    if operator == 'softmax':
        model = SimpleBaselineStack('softmax', d_model, len(layout)).to(device)
    else:
        memory_type = 'kda' if operator in ('memory', 'hybrid') else operator
        # For pure CSA/HCA stacks, memory_type is unused (no 'memory'
        # slots), but we still pass a valid value.
        cfg = _make_matrix_cfg(d_model, memory_type, list(layout),
                               cfg_overrides={})
        model = HybridAttentionStack(cfg, total_layers=len(layout)).to(device)
    return model, head, embed


def _train_eval_longseq_cell(operator: str, seq_len: int, d_model: int,
                             steps: int, lr: float, seed: int, device,
                             eval_batches: int = 4, eval_batch: int = 64,
                             train_batch: int | None = None,
                             val_every: int = 0, val_batches: int = 2,
                             val_batch: int = 32,
                             error_batches: int = 0, error_batch: int = 64,
                             max_error_examples: int = 2) -> dict:
    """Train + evaluate ONE (operator, seq_len) cell at ONE seed.

    Mirrors eval_layout_one_seed's protocol (same RNG stream offsets, same
    held-out curve, same bad-case pass, same latency probe).
    """
    if isinstance(device, str):
        device = torch.device(device)
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    layout = LS_LAYOUT[operator]
    if operator == 'softmax':
        model = SimpleBaselineStack('softmax', d_model, len(layout)).to(device)
    else:
        memory_type = 'kda' if operator in ('memory', 'hybrid') else operator
        cfg = _make_matrix_cfg(d_model, memory_type, list(layout),
                               cfg_overrides={})
        model = HybridAttentionStack(cfg, total_layers=len(layout)).to(device)
    param_groups = _build_param_groups(model, head, embed,
                                       weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout_str = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)
    if train_batch is None:
        train_batch = parse_int_env('LS_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    model.train()
    head.train()
    embed.train()

    def _val_eval(_step):
        acc, loss = _eval_model(
            model, head, embed, seq_len, n_kv=1, device=device,
            n_batches=int(val_batches or 2), batch=int(val_batch or 32),
            seed=seed, seed_offset=VAL_MONITOR_SEED_OFFSET)
        return ValPoint(loss=loss, acc=acc)

    curve = ValidationCurve(
        _val_eval, steps=steps,
        val_every=int(val_every or 0) or max(1, steps // 4),
        label=f'{operator} T={seq_len} seed={seed}', ceiling=1.0)
    curve.record(0)

    losses = []
    train_accs = []
    grad_norms = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, 1, VOCAB, embed, device,
            generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                params=params, logits=logits,
                context=f'step={step}, op={operator}, T={seq_len}, seed={seed}')
        opt.zero_grad()
        loss.backward()
        bad_grads = triage_gradient(params)
        clip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=float(loss.detach()),
                params=params,
                grads={f'p{i}': p.grad for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'step={step}, op={operator}, T={seq_len}, seed={seed}')
        grad_norms.append(float(clip_norm))
        opt.step()
        losses.append(loss.item())
        train_accs.append(float((logits.argmax(-1) == target).float().mean()))
        if curve.enabled and (step + 1) % curve.val_every == 0:
            curve.record(step + 1)

    final_acc, final_loss = _eval_model(
        model, head, embed, seq_len, n_kv=1, device=device,
        n_batches=eval_batches, batch=eval_batch, seed=seed,
    )

    lat_gen = make_seeded_generator(99, device=device)
    x = torch.randn(1, seq_len, d_model, device=device,
                    generator=lat_gen) * 0.1
    was_training = model.training
    model.eval()
    _prev_thr = torch.get_num_threads()
    if device.type != 'cuda':
        torch.set_num_threads(1)
    try:
        with torch.no_grad():
            model.reset_state()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                model.reset_state()
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / 5 * 1e3
    finally:
        if device.type != 'cuda':
            torch.set_num_threads(_prev_thr)
        model.train(was_training)

    last_loss = losses[-1] if losses else 0.0
    mean_last10 = (sum(losses[-10:]) / min(10, len(losses))) if losses else 0.0
    curve_block = _attach_curve_diagnostics(
        curve=curve, model=model, head=head, embed=embed,
        train_losses=losses, train_accs=train_accs, op_name='longseq',
        seq_len=seq_len, n_kv=1, device=device, steps=steps,
        error_batches=error_batches, error_batch=error_batch,
        max_error_examples=max_error_examples, seed=seed,
        label=f'{operator} T={seq_len} n_kv=1')

    return {
        'operator': operator,
        'seq_len': seq_len,
        'layout': layout_str,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'fwd_ms': fwd_ms,
        'n_params': n_params,
        'n_layers': len(layout),
        'seed': seed,
        'steps': steps,
        'train_batch': train_batch,
        'last_train_loss': last_loss,
        'mean_last10_loss': mean_last10,
        'loss_curve': losses,
        'acc_curve': train_accs,
        'gradient_norms': summarize_gradient_norms(grad_norms,
                                                   clip_norm=1.0),
        **curve_block,
    }


def _paired_t_pvalue(t_stat: float, df: int) -> float:
    """Two-sided paired t p-value via the regularized incomplete beta."""
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    a = df / 2.0
    b = 0.5

    def betacf(a_val, b_val, x_val):
        MAXIT = 200
        EPS = 3.0e-14
        FPMIN = 1.0e-30
        qab = a_val + b_val
        qap = a_val + 1.0
        qam = a_val - 1.0
        c = 1.0
        d = 1.0 - qab * x_val / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b_val - m) * x_val / ((qam + m2) * (a_val + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a_val + m) * (qab + m) * x_val / ((a_val + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return h

    def betai(a_val, b_val, x_val):
        if x_val <= 0.0:
            return 0.0
        if x_val >= 1.0:
            return 1.0
        bt = (math.exp(math.lgamma(a_val + b_val) - math.lgamma(a_val)
                       - math.lgamma(b_val) + a_val * math.log(x_val)
                       + b_val * math.log(1.0 - x_val)))
        if x_val < (a_val + 1.0) / (a_val + b_val + 2.0):
            return bt * betacf(a_val, b_val, x_val) / a_val
        return 1.0 - bt * betacf(b_val, a_val, 1.0 - x_val) / b_val

    return betai(0.5 * df, 0.5, x)


def _summarize_cell(operator: str, seq_len: int, per_seed: list[dict],
                    seeds: list[int],
                    reference_per_seed: list[dict] | None = None) -> dict:
    """Aggregate one cell's per-seed rows into a summary row.

    If ``reference_per_seed`` is given (the same operator at T=16), compute
    the seed-paired contrast: mean diff, paired t-test, Bonferroni-corrected
    significance, TOST equivalence.
    """
    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        raise RuntimeError(
            f"all {len(per_seed)} seeds failed for {operator} T={seq_len}; "
            f"first error: {per_seed[0].get('error')}")

    n = len(ok_per_seed)
    accs = [r['final_acc'] for r in ok_per_seed]
    fwds = [r['fwd_ms'] for r in ok_per_seed]
    mean_acc = sum(accs) / n
    mean_fwd = sum(fwds) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        std_acc = math.sqrt(var_acc)
        ci_acc = _t_crit_975(n) * (std_acc / math.sqrt(n))
    else:
        std_acc = None
        ci_acc = None

    chance = 1.0 / VOCAB
    t_stat_vs_chance = None
    if n > 1 and std_acc and std_acc > 0:
        t_stat_vs_chance = (mean_acc - chance) / (std_acc / math.sqrt(n))

    error_aggregate = pool_error_analyses(ok_per_seed)
    log_pooled_errors(error_aggregate,
                      label=f'{operator} T={seq_len}')

    summary = {
        'operator': operator,
        'seq_len': seq_len,
        'layout': ok_per_seed[0]['layout'],
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds': len(per_seed),
        'n_seeds_total': len(per_seed),
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat_vs_chance,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        **_aggregate_curve_diagnosis(ok_per_seed),
        'mean_train_time_s': (sum(r.get('train_time_s', 0.0) for r in ok_per_seed)
                              / n),
        'error_analysis': error_aggregate,
        'error_analysis_seed_offset': ERROR_ANALYSIS_SEED_OFFSET,
        'val_monitor_seed_offset': VAL_MONITOR_SEED_OFFSET,
    }

    # Paired contrast vs the same operator at T=16.
    contrast = None
    if reference_per_seed is not None:
        ok_ref = [r for r in reference_per_seed
                  if 'error' not in r
                  and r.get('final_acc') is not None
                  and math.isfinite(r['final_acc'])]
        common = sorted(set(r['seed'] for r in ok_per_seed)
                        & set(r['seed'] for r in ok_ref))
        if len(common) >= 2:
            acc_map = {r['seed']: r['final_acc'] for r in ok_per_seed}
            ref_map = {r['seed']: r['final_acc'] for r in ok_ref}
            diffs = [acc_map[s] - ref_map[s] for s in common]
            mean_diff = sum(diffs) / len(diffs)
            if len(diffs) > 1:
                sd_diff = math.sqrt(
                    sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1))
                t_stat = mean_diff / (sd_diff / math.sqrt(len(diffs)))
                p_two = _paired_t_pvalue(t_stat, len(diffs))
                p_one = p_two / 2.0 if mean_diff > 0 else 1.0 - p_two / 2.0
                crit = _bonferroni_crit_q(len(diffs), alpha=0.05 / 1)
                sig = abs(t_stat) >= crit
                tost = tost_from_summary(mean_diff, sd_diff, len(diffs),
                                         LS_EQUIV_MARGIN)
            else:
                sd_diff = None
                t_stat = None
                p_two = None
                p_one = None
                sig = False
                tost = None
            contrast = {
                'contrast': f"{operator} T={seq_len} vs T=16",
                'operator': operator,
                'seq_len': seq_len,
                'reference_seq_len': 16,
                'n_seeds_paired': len(common),
                'mean_diff': mean_diff,
                'sd_diff': sd_diff,
                't_stat': t_stat,
                'p_two_sided': p_two,
                'p_one_sided': p_one,
                'bonferroni_crit': crit,
                'significant_bonferroni': sig,
                'tost_equivalent': (tost['equivalent'] if tost else None),
                'tost': tost,
            }
    summary['contrast_vs_t16'] = contrast
    return summary


def _run_overfit_gate_longseq(device, seq_len: int) -> tuple[dict, bool]:
    """Pre-sweep overfit gate: hybrid at the smallest seq len."""
    label = f'hybrid T={seq_len} reference'

    def _build():
        return _build_longseq_model('hybrid', seq_len,
                                    int(os.environ.get('LS_D_MODEL', '32')),
                                    device)

    return _run_ablation_overfit_gate(
        [(label, _build)], device, n_kv=1, env_prefix='LS', seq_len=seq_len)


def main() -> int:
    device = torch.device(os.environ.get('LS_DEVICE', 'cpu'))
    d_model = parse_int_env('LS_D_MODEL', 32, min_value=8, logger=logger)
    seq_lens = _parse_seq_lens(os.environ.get('LS_SEQ_LENS'))
    operators = _parse_operators(os.environ.get('LS_OPERATORS'))
    steps = parse_int_env('LS_STEPS', 100, min_value=10, logger=logger)
    n_seeds = parse_int_env('LS_SEEDS', 5, min_value=2, logger=logger)
    lr = 3e-3
    eval_batches = parse_int_env('LS_ABL_EVAL_BATCHES', 4, min_value=1,
                                 logger=logger)
    eval_batch = parse_int_env('LS_ABL_EVAL_BATCH', 64, min_value=1,
                               logger=logger)
    train_batch = parse_int_env('LS_TRAIN_BATCH', 16, min_value=1,
                                logger=logger)
    error_batches = parse_int_env('LS_ABL_ERROR_BATCHES', 0, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('LS_ABL_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    max_error_examples = parse_int_env('LS_ABL_ERROR_EXAMPLES', 2,
                                       min_value=1, logger=logger)

    logger.info(f'[exp12] device={device} d_model={d_model} '
                f'seq_lens={seq_lens} operators={operators} '
                f'steps={steps} seeds={n_seeds}')

    # --- Overfit gate (hybrid at T=16 only) ---
    gate_record, proceed = _run_overfit_gate_longseq(device, seq_lens[0])

    # --- Enumerate cells ---
    seeds = seed_ids(n_seeds, plan_id=LS_PLAN_ID)
    results: list[dict] = []
    contrasts: list[dict] = []
    cross_operator_contrasts: list[dict] = []

    # Cache per-operator reference rows at T=16.
    ref_cache: dict[str, list[dict]] = {}

    for seq_len in seq_lens:
        logger.info(f'[exp12] seq_len={seq_len}')
        for operator in operators:
            logger.info(f'[exp12]  op={operator} T={seq_len}')
            per_seed = []
            for s in seeds:
                r = _train_eval_longseq_cell(
                    operator, seq_len, d_model, steps, lr, s, device,
                    eval_batches=eval_batches, eval_batch=eval_batch,
                    train_batch=train_batch,
                    error_batches=error_batches, error_batch=error_batch,
                    max_error_examples=max_error_examples)
                per_seed.append(r)

            if seq_len == 16:
                ref_cache[operator] = per_seed

            reference_per_seed = ref_cache.get(operator) if seq_len != 16 else None
            summary = _summarize_cell(operator, seq_len, per_seed, seeds,
                                      reference_per_seed)
            results.append(summary)
            if summary.get('contrast_vs_t16'):
                contrasts.append(summary['contrast_vs_t16'])

        # --- Cross-operator contrasts at this length ---
        # Compare hybrid vs each other operator (seed-paired at this length).
        if seq_len in seq_lens:
            hybrid_rows = [r for r in results if r['operator'] == 'hybrid'
                           and r['seq_len'] == seq_len and 'per_seed' in r]
            for other in operators:
                if other == 'hybrid':
                    continue
                other_rows = [r for r in results
                              if r['operator'] == other
                              and r['seq_len'] == seq_len and 'per_seed' in r]
                if not hybrid_rows or not other_rows:
                    continue
                h_row = hybrid_rows[0]
                o_row = other_rows[0]
                h_per_seed = [r for r in h_row['per_seed']
                              if 'error' not in r
                              and r.get('final_acc') is not None]
                o_per_seed = [r for r in o_row['per_seed']
                              if 'error' not in r
                              and r.get('final_acc') is not None]
                common = sorted(set(r['seed'] for r in h_per_seed)
                                & set(r['seed'] for r in o_per_seed))
                if len(common) < 2:
                    continue
                h_map = {r['seed']: r['final_acc'] for r in h_per_seed}
                o_map = {r['seed']: r['final_acc'] for r in o_per_seed}
                diffs = [h_map[s] - o_map[s] for s in common]
                mean_diff = sum(diffs) / len(diffs)
                if len(diffs) > 1:
                    sd_diff = math.sqrt(
                        sum((d - mean_diff) ** 2 for d in diffs)
                        / (len(diffs) - 1))
                    t_stat = mean_diff / (sd_diff / math.sqrt(len(diffs)))
                    p_two = _paired_t_pvalue(t_stat, len(diffs))
                    p_one = p_two / 2.0 if mean_diff > 0 else 1.0 - p_two / 2.0
                    crit = _bonferroni_crit_q(len(diffs),
                                              alpha=0.05 / (len(operators) - 1))
                    sig = abs(t_stat) >= crit
                    tost = tost_from_summary(mean_diff, sd_diff, len(diffs),
                                             LS_EQUIV_MARGIN)
                else:
                    sd_diff = None
                    t_stat = None
                    p_two = None
                    p_one = None
                    sig = False
                    tost = None
                cross_operator_contrasts.append({
                    'contrast': f"hybrid vs {other} at T={seq_len}",
                    'seq_len': seq_len,
                    'operator_a': 'hybrid',
                    'operator_b': other,
                    'n_seeds_paired': len(common),
                    'mean_diff': mean_diff,
                    'sd_diff': sd_diff,
                    't_stat': t_stat,
                    'p_two_sided': p_two,
                    'p_one_sided': p_one,
                    'bonferroni_crit': crit,
                    'significant_bonferroni': sig,
                    'tost_equivalent': (tost['equivalent'] if tost else None),
                    'tost': tost,
                })

    # --- Provenance + manifest ---
    prov = _run_provenance(n_seeds, steps, overfit_gate=gate_record,
                           plan_id=LS_PLAN_ID)
    prov['training']['lr'] = lr
    prov['data'] = {
        'kind': 'synthetic',
        'generator_function': make_mqar_batch.__name__,
        'task_spec': {'seq_len': 'varies', 'vocab': VOCAB},
        'seed_offsets': {'train': 1_000_000, 'eval': 5_000_000},
        'note': ('Synthetic batches are a deterministic function of '
                 '(generator, task spec, seed + offset); seq_len varies '
                 'across cells.'),
    }
    prov['seq_lens'] = seq_lens
    prov['operators'] = operators
    prov['equivalence_margin'] = LS_EQUIV_MARGIN

    # --- Experiment-design audit ---
    try:
        design_audit = audit_experiment_design(
            LS_PLAN_ID,
            seeds=seeds,
            observed_factor_levels={
                'operator': operators,
                'seq_len': [str(s) for s in seq_lens],
            },
            family_size=len(contrasts) + len(cross_operator_contrasts),
            analyzed_seed_ids=seeds,
            pilot_sd=None,
        )
    except Exception as e:  # pragma: no cover - audit must not kill a run
        logger.warning(f'[exp12] experiment-design audit failed: {e}')
        design_audit = {'status': 'audit_error', 'error': str(e)}

    metadata = {
        'experiment': 'exp12_longseq_quality',
        'plan_id': LS_PLAN_ID,
        'device': str(device),
        'd_model': d_model,
        'seq_lens': seq_lens,
        'operators': operators,
        'steps': steps,
        'n_seeds': n_seeds,
        'seeds': seeds,
        'lr': lr,
        'equivalence_margin': LS_EQUIV_MARGIN,
        'n_cells': len(results),
        'n_contrasts_vs_t16': len(contrasts),
        'n_cross_operator_contrasts': len(cross_operator_contrasts),
        'design_audit': design_audit,
        'overfit_gate': _compact_overfit_report(gate_record),
        'proceed': proceed,
    }

    envelope = {'metadata': metadata, 'results': results,
                'contrasts': contrasts,
                'cross_operator_contrasts': cross_operator_contrasts}

    os.makedirs('results', exist_ok=True)
    write_results_checked(envelope, LS_RESULTS_FILE)
    write_json_atomic(prov, LS_PROVENANCE_FILE)
    write_json_atomic({
        'metadata': metadata,
        'contrasts': contrasts,
        'cross_operator_contrasts': cross_operator_contrasts,
    }, LS_CONTRASTS_FILE)

    _append_ablation_manifest_longseq(len(results), 0, n_seeds, steps)

    # Console summary.
    for c in contrasts:
        sig_tag = 'SIG' if c['significant_bonferroni'] else 'ns'
        eq_tag = ('eq' if c['tost_equivalent'] else 'neq')
        logger.info(
            f"[exp12] {c['contrast']}: mean_diff={c['mean_diff']:+.4f} "
            f"t={c['t_stat'] if c['t_stat'] is not None else 'n/a'} "
            f"[{sig_tag}, {eq_tag}]")
    for c in cross_operator_contrasts:
        sig_tag = 'SIG' if c['significant_bonferroni'] else 'ns'
        eq_tag = ('eq' if c['tost_equivalent'] else 'neq')
        logger.info(
            f"[exp12] {c['contrast']}: mean_diff={c['mean_diff']:+.4f} "
            f"t={c['t_stat'] if c['t_stat'] is not None else 'n/a'} "
            f"[{sig_tag}, {eq_tag}]")
    logger.info(f'[exp12] wrote {LS_RESULTS_FILE} ({len(results)} cells, '
                f'{len(contrasts)} vs-T16 contrasts, '
                f'{len(cross_operator_contrasts)} cross-operator contrasts)')
    return 0


def _compact_overfit_report(report: dict) -> dict:
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def _append_ablation_manifest_longseq(n_cells: int, return_code: int,
                                      n_seeds: int, steps: int) -> None:
    prov = _run_provenance(n_seeds, steps, plan_id=LS_PLAN_ID)
    append_run_manifest(make_run_manifest_entry(
        experiment='exp12_longseq_quality',
        seeds=prov['seeds'],
        provenance=prov,
        hyperparameters=prov['training'],
        data=prov['data'],
        artifacts={
            'results': LS_RESULTS_FILE,
            'provenance': LS_PROVENANCE_FILE,
            'contrasts': LS_CONTRASTS_FILE,
        },
        results={'n_cells': n_cells, 'return_code': return_code},
        status='completed' if not return_code else 'partial',
    ), logger=logger)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    sys.exit(main())
