"""Experiment 11 — CSA/HCA compression-ratio ablation.

Motivation
----------
Exp 8 ablates CSA's *mechanisms* (partial RoPE, top-k selection density,
indexer STE learning) but holds the compression *ratios* fixed at the
paper defaults (csa_m=16, hca_m2=64, ...). It does not answer how much of
CSA's/HCA's quality comes from the compression ratio itself versus the
retrieval/aggregation mechanism. Two arms with different ``csa_m`` or
``hca_m2`` have different KV-cache footprints and different
information bottlenecks, so a quality difference can be a pure
compression-ratio effect rather than a mechanism effect.

Design
------
* Fixed 3:1:1 hybrid layout (n_kda=3, n_csa=1, n_hca=1), 5 layers,
  ``SEQ_LEN=16`` MQAR, same training recipe as Exp 5 (100 steps,
  AdamW 3e-3, multi-seed).
* One-factor-at-a-time grid around the Exp-5 reference:
    * CSA: ``csa_m`` ∈ {8, 16, 32, 64} (compression factor),
           ``csa_topk`` ∈ {1, 4, 8} (selection density),
           ``csa_c`` ∈ {32, 64, 128} (compressed KV channel width).
    * HCA: ``hca_m2`` ∈ {32, 64, 128, 256} (heavy compression),
           ``hca_c`` ∈ {32, 64, 128} (compressed channel width).
    * Reference arm: csa_m=16, csa_topk=8, csa_c=64, hca_m2=64, hca_c=64.
* Each arm runs ``ABL_SEEDS`` seeds paired with the reference at the same
  seeds; the primary statistic is a seed-paired paired t-test arm vs
  reference, Bonferroni-corrected within the CSA family and within the
  HCA family, plus TOST equivalence at ``CONTRAST_EQUIVALENCE_MARGIN =
  0.05``.
* Parameter-capacity audit: every arm records ``n_params``; an arm whose
  parameter count differs from the reference by more than
  ``CAPACITY_MATCH_TOLERANCE = 5%`` is labelled ``operator_plus_capacity``.
* Latency probe per arm (same protocol as Exp 5's ``fwd_ms``) so the
  quality-vs-cost tradeoff is visible: a tighter compression may lose a
  point of MQAR while halving the forward pass.

Environment variables
---------------------
* ``CC_ABL_STEPS`` (default 100), ``CC_ABL_SEEDS`` (default 7),
  ``CC_ABL_TRAIN_BATCH`` (default 16), ``CC_ABL_SEQ_LEN`` (default 16).
* ``CC_CSA=0`` skips the CSA grid; ``CC_HCA=0`` skips the HCA grid.
* ``CC_OVERFIT_CHECK`` / ``CC_OVERFIT_STRICT`` / ``CC_OVERFIT_STEPS`` /
  ``CC_OVERFIT_SEED`` / ``CC_OVERFIT_BATCH`` (mirror Exp 5's ABL_* gate).
* ``CC_DEVICE`` (default ``cpu``; ``cuda`` when a GPU is available).

Outputs
-------
``results/exp11_compression_ablation.json`` — envelope
``{metadata, results: [...]}`` with per-arm per-seed rows plus an aggregated
summary row per arm and a ``contrasts`` block (arm vs reference, paired
t-test, Bonferroni, TOST, capacity label).
"""

from __future__ import annotations

import itertools
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
from ops_fused import HybridAttentionStack, HybridConfig
from ops_kda_backend import validate_kda_backend
from run_ablation import (
    VOCAB, SEQ_LEN, CONTRAST_EQUIVALENCE_MARGIN, CAPACITY_MATCH_TOLERANCE,
    _make_cfg, _run_provenance, _run_ablation_overfit_gate,
    _attach_curve_diagnostics, _eval_model, eval_layout_one_seed,
    summarize_layout_results, run_overfit_sanity_check,
    _OVERFIT_BATCH_SEED_OFFSET, ERROR_ANALYSIS_SEED_OFFSET,
    VAL_MONITOR_SEED_OFFSET, _aggregate_curve_diagnosis,
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

CC_PLAN_ID = 'exp11_compression'
CC_RESULTS_FILE = 'results/exp11_compression_ablation.json'
CC_PROVENANCE_FILE = 'results/exp11_compression_provenance.json'
CC_CONTRASTS_FILE = 'results/exp11_compression_contrasts.json'

# Reference compression ratios (Exp-5 defaults from SMALL_MODEL_SPEC).
CC_REFERENCE = {
    'csa_m': 16, 'csa_topk': 8, 'csa_c': 64,
    'hca_m2': 64, 'hca_c': 64,
}

# One-factor-at-a-time grids around the reference.
CC_CSA_GRID = {
    'csa_m': [8, 16, 32, 64],
    'csa_topk': [1, 4, 8],
    'csa_c': [32, 64, 128],
}
CC_HCA_GRID = {
    'hca_m2': [32, 64, 128, 256],
    'hca_c': [32, 64, 128],
}

CC_EQUIV_MARGIN = CONTRAST_EQUIVALENCE_MARGIN


def _compression_override(values: dict) -> dict:
    """Build cfg_overrides for one compression arm.

    Only the compression fields present in ``values`` are touched; every
    other HybridConfig field keeps its Exp-5 default so the ablation
    isolates the single compression knob.
    """
    allowed = {'csa_m', 'csa_topk', 'csa_c', 'hca_m2', 'hca_c'}
    overrides = {}
    for key, val in values.items():
        if key in allowed:
            overrides[key] = val
    return overrides


def _arm_label_compression(values: dict) -> dict:
    """Human-readable arm label for a compression arm."""
    parts = []
    for key in ('csa_m', 'csa_topk', 'csa_c', 'hca_m2', 'hca_c'):
        if key in values:
            parts.append(f"{key}={values[key]}")
    return "compression: " + ", ".join(parts) if parts else "reference"


def _enumerate_compression_grid() -> list[dict]:
    """Return the list of arm value-dicts (reference first).

    One-factor-at-a-time: each arm changes exactly one compression knob
    away from the reference, so every contrast isolates one factor.
    """
    arms: list[dict] = []
    for family, grid in (('csa', CC_CSA_GRID), ('hca', CC_HCA_GRID)):
        for key, levels in grid.items():
            for level in levels:
                if level == CC_REFERENCE[key]:
                    continue  # the reference arm is added once below
                arms.append({key: level})
    # Reference arm first.
    arms.insert(0, {})
    return arms


def _build_compression_model(values: dict, d_model: int,
                             device) -> tuple[nn.Module, nn.Module,
                                              nn.Embedding]:
    """Construct (model, head, embed) for one compression arm.

    Uses the ratio-mode path of _make_cfg (3:1:1) with cfg_overrides
    injecting the compression knobs. Mirrors eval_layout's construction
    path exactly.
    """
    from run_ablation import _make_cfg
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio=(3, 1, 1))
    # Apply compression overrides on top of the default config.
    for key, val in _compression_override(values).items():
        setattr(cfg, key, val)
    total = 5  # 3+1+1
    model = HybridKCHAttention(cfg, total_layers=total).to(device)
    return model, head, embed


def _train_eval_compression_arm(values: dict, d_model: int, seq_len: int,
                                n_kv: int, steps: int, lr: float, seed: int,
                                device, eval_batches: int = 4,
                                eval_batch: int = 64,
                                train_batch: int | None = None,
                                val_every: int = 0, val_batches: int = 2,
                                val_batch: int = 32,
                                error_batches: int = 0, error_batch: int = 64,
                                max_error_examples: int = 2) -> dict:
    """Train + evaluate ONE compression arm at ONE seed.

    Mirrors eval_layout_one_seed's protocol (same RNG stream offsets, same
    held-out curve, same bad-case pass, same latency probe).
    """
    from ops_fused import HybridKCHAttention
    if isinstance(device, str):
        device = torch.device(device)
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio=(3, 1, 1))
    for key, val in _compression_override(values).items():
        setattr(cfg, key, val)
    total = 5
    model = HybridKCHAttention(cfg, total_layers=total).to(device)
    param_groups = _build_param_groups(model, head, embed,
                                       weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout_str = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)
    if train_batch is None:
        train_batch = parse_int_env('CC_ABL_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    model.train()
    head.train()
    embed.train()

    def _val_eval(_step):
        acc, loss = _eval_model(
            model, head, embed, seq_len, n_kv, device,
            n_batches=int(val_batches or 2), batch=int(val_batch or 32),
            seed=seed, seed_offset=VAL_MONITOR_SEED_OFFSET)
        return ValPoint(loss=loss, acc=acc)

    curve = ValidationCurve(
        _val_eval, steps=steps,
        val_every=int(val_every or 0) or max(1, steps // 4),
        label=f'compression {values} seed={seed}', ceiling=1.0)
    curve.record(0)

    losses = []
    train_accs = []
    grad_norms = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device,
            generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                params=params, logits=logits,
                context=f'step={step}, compression={values}, seed={seed}')
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
                context=f'step={step}, compression={values}, seed={seed}')
        grad_norms.append(float(clip_norm))
        opt.step()
        losses.append(loss.item())
        train_accs.append(float((logits.argmax(-1) == target).float().mean()))
        if curve.enabled and (step + 1) % curve.val_every == 0:
            curve.record(step + 1)

    final_acc, final_loss = _eval_model(
        model, head, embed, seq_len, n_kv, device,
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
        train_losses=losses, train_accs=train_accs, op_name='compression',
        seq_len=seq_len, n_kv=n_kv, device=device, steps=steps,
        error_batches=error_batches, error_batch=error_batch,
        max_error_examples=max_error_examples, seed=seed,
        label=f'compression {values} n_kv={n_kv}')

    return {
        'values': values,
        'label': _arm_label_compression(values),
        'n_kv': n_kv,
        'layout': layout_str,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'fwd_ms': fwd_ms,
        'n_params': n_params,
        'n_layers': total,
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


def _summarize_arm(values: dict, per_seed: list[dict], seeds: list[int],
                   reference_values: dict,
                   reference_per_seed: list[dict] | None = None) -> dict:
    """Aggregate one arm's per-seed rows into a summary row.

    If ``reference_per_seed`` is given, compute the seed-paired contrast
    arm vs reference: mean diff, paired t-test, Bonferroni-corrected
    significance, TOST equivalence, and the capacity label.
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
            f"all {len(per_seed)} seeds failed for compression arm "
            f"{values}; "
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
                      label=f'compression {values} n_kv={ok_per_seed[0]["n_kv"]}')

    summary = {
        'values': values,
        'label': _arm_label_compression(values),
        'n_kv': ok_per_seed[0]['n_kv'],
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

    # Paired contrast vs the reference (same seeds).
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
                crit = _bonferroni_crit_q(len(common),
                                          alpha=0.05 / 1)  # single contrast vs reference
                sig = abs(t_stat) >= crit
                tost = tost_from_summary(mean_diff, sd_diff, len(diffs),
                                         CC_EQUIV_MARGIN)
            else:
                sd_diff = None
                t_stat = None
                p_two = None
                p_one = None
                sig = False
                tost = None
            params_arm = ok_per_seed[0]['n_params']
            params_ref = ok_ref[0]['n_params'] if ok_ref else None
            capacity_gap = None
            capacity_label = 'capacity_matched'
            if params_arm and params_ref:
                capacity_gap = abs(params_arm - params_ref) / params_ref
                if capacity_gap > CAPACITY_MATCH_TOLERANCE:
                    capacity_label = 'operator_plus_capacity'
            contrast = {
                'contrast': f"{_arm_label_compression(values)} vs reference",
                'arm': values,
                'reference': reference_values,
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
                'capacity_gap': capacity_gap,
                'capacity_label': capacity_label,
            }
    summary['contrast'] = contrast
    return summary


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


def _run_overfit_gate_compression(device, seq_len: int) -> tuple[dict, bool]:
    """Pre-sweep overfit gate: the reference compression arm."""
    reference = CC_REFERENCE
    label = f'compression reference {reference}'

    def _build():
        return _build_compression_model(reference,
                                        int(os.environ.get('CC_D_MODEL', '32')),
                                        device)

    return _run_ablation_overfit_gate(
        [(label, _build)], device, n_kv=1, env_prefix='CC', seq_len=seq_len)


def main() -> int:
    device = torch.device(os.environ.get('CC_DEVICE', 'cpu'))
    d_model = parse_int_env('CC_D_MODEL', 32, min_value=8, logger=logger)
    seq_len = parse_int_env('CC_ABL_SEQ_LEN', SEQ_LEN, min_value=8,
                            logger=logger)
    steps = parse_int_env('CC_ABL_STEPS', 100, min_value=10, logger=logger)
    n_seeds = parse_int_env('CC_ABL_SEEDS', 7, min_value=2, logger=logger)
    lr = 3e-3
    eval_batches = parse_int_env('CC_ABL_EVAL_BATCHES', 4, min_value=1,
                                 logger=logger)
    eval_batch = parse_int_env('CC_ABL_EVAL_BATCH', 64, min_value=1,
                               logger=logger)
    train_batch = parse_int_env('CC_ABL_TRAIN_BATCH', 16, min_value=1,
                                logger=logger)
    error_batches = parse_int_env('CC_ABL_ERROR_BATCHES', 0, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('CC_ABL_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    max_error_examples = parse_int_env('CC_ABL_ERROR_EXAMPLES', 2,
                                       min_value=1, logger=logger)

    run_csa = parse_int_env('CC_CSA', 1, min_value=0, logger=logger) != 0
    run_hca = parse_int_env('CC_HCA', 1, min_value=0, logger=logger) != 0

    logger.info(f'[exp11] device={device} d_model={d_model} seq_len={seq_len} '
                f'steps={steps} seeds={n_seeds} csa={run_csa} hca={run_hca}')

    # --- Overfit gate (reference arm only) ---
    gate_record, proceed = _run_overfit_gate_compression(device, seq_len)

    # --- Enumerate arms ---
    arms: list[dict] = []
    if run_csa:
        for key, levels in CC_CSA_GRID.items():
            for level in levels:
                if level == CC_REFERENCE[key]:
                    continue
                arms.append({key: level})
    if run_hca:
        for key, levels in CC_HCA_GRID.items():
            for level in levels:
                if level == CC_REFERENCE[key]:
                    continue
                arms.append({key: level})

    reference_values = CC_REFERENCE
    seeds = seed_ids(n_seeds, plan_id=CC_PLAN_ID)

    results: list[dict] = []
    contrasts: list[dict] = []

    # Cache the reference per-seed rows once.
    reference_per_seed: list[dict] | None = None

    for arm in arms:
        values = arm
        is_reference = (values == {})

        logger.info(f'[exp11] arm {_arm_label_compression(values)} '
                    f'({"reference" if is_reference else "ablation"})')
        per_seed = []
        for s in seeds:
            r = _train_eval_compression_arm(
                values, d_model, seq_len, n_kv=1, steps=steps, lr=lr,
                seed=s, device=device, eval_batches=eval_batches,
                eval_batch=eval_batch, train_batch=train_batch,
                error_batches=error_batches, error_batch=error_batch,
                max_error_examples=max_error_examples)
            r['arm_is_reference'] = is_reference
            per_seed.append(r)

        if not is_reference and reference_per_seed is None:
            reference_per_seed = [
                _train_eval_compression_arm(
                    reference_values, d_model, seq_len, n_kv=1, steps=steps,
                    lr=lr, seed=s, device=device,
                    eval_batches=eval_batches, eval_batch=eval_batch,
                    train_batch=train_batch,
                    error_batches=error_batches, error_batch=error_batch,
                    max_error_examples=max_error_examples)
                for s in seeds
            ]

        summary = _summarize_arm(values, per_seed, seeds, reference_values,
                                 reference_per_seed)
        summary['arm_is_reference'] = is_reference
        results.append(summary)
        if summary.get('contrast'):
            contrasts.append(summary['contrast'])

    # --- Provenance + manifest ---
    prov = _run_provenance(n_seeds, steps, overfit_gate=gate_record,
                           plan_id=CC_PLAN_ID)
    prov['training']['lr'] = lr
    prov['data'] = {
        'kind': 'synthetic',
        'generator_function': make_mqar_batch.__name__,
        'task_spec': {'seq_len': seq_len, 'vocab': VOCAB},
        'seed_offsets': {'train': 1_000_000, 'eval': 5_000_000},
        'note': ('Synthetic batches are a deterministic function of '
                 '(generator, task spec, seed + offset).'),
    }
    prov['compression_grid'] = {
        'reference': CC_REFERENCE,
        'csa': CC_CSA_GRID,
        'hca': CC_HCA_GRID,
    }
    prov['equivalence_margin'] = CC_EQUIV_MARGIN
    prov['capacity_match_tolerance'] = CAPACITY_MATCH_TOLERANCE

    # --- Experiment-design audit ---
    try:
        design_audit = audit_experiment_design(
            CC_PLAN_ID,
            seeds=seeds,
            observed_factor_levels={
                'csa_m': sorted({str(a.get('csa_m')) for a in arms if 'csa_m' in a}),
                'csa_topk': sorted({str(a.get('csa_topk')) for a in arms if 'csa_topk' in a}),
                'csa_c': sorted({str(a.get('csa_c')) for a in arms if 'csa_c' in a}),
                'hca_m2': sorted({str(a.get('hca_m2')) for a in arms if 'hca_m2' in a}),
                'hca_c': sorted({str(a.get('hca_c')) for a in arms if 'hca_c' in a}),
            },
            family_size=len(contrasts),
            analyzed_seed_ids=seeds,
            pilot_sd=None,
        )
    except Exception as e:  # pragma: no cover - audit must not kill a run
        logger.warning(f'[exp11] experiment-design audit failed: {e}')
        design_audit = {'status': 'audit_error', 'error': str(e)}

    metadata = {
        'experiment': 'exp11_compression_ablation',
        'plan_id': CC_PLAN_ID,
        'device': str(device),
        'd_model': d_model,
        'seq_len': seq_len,
        'steps': steps,
        'n_seeds': n_seeds,
        'seeds': seeds,
        'lr': lr,
        'reference': CC_REFERENCE,
        'equivalence_margin': CC_EQUIV_MARGIN,
        'capacity_match_tolerance': CAPACITY_MATCH_TOLERANCE,
        'arms': [{'values': a, 'label': _arm_label_compression(a)} for a in arms],
        'n_arms': len(arms),
        'n_contrasts': len(contrasts),
        'design_audit': design_audit,
        'overfit_gate': _compact_overfit_report(gate_record),
        'proceed': proceed,
    }

    envelope = {'metadata': metadata, 'results': results,
                'contrasts': contrasts}

    os.makedirs('results', exist_ok=True)
    write_results_checked(envelope, CC_RESULTS_FILE)
    write_json_atomic(prov, CC_PROVENANCE_FILE)
    write_json_atomic({'metadata': metadata, 'contrasts': contrasts},
                      CC_CONTRASTS_FILE)

    _append_ablation_manifest_compression(len(arms), 0, n_seeds, steps)

    # Console summary.
    for c in contrasts:
        sig_tag = 'SIG' if c['significant_bonferroni'] else 'ns'
        eq_tag = ('eq' if c['tost_equivalent'] else 'neq')
        logger.info(
            f"[exp11] contrast {c['contrast']}: "
            f"mean_diff={c['mean_diff']:+.4f} "
            f"t={c['t_stat'] if c['t_stat'] is not None else 'n/a'} "
            f"p2={c['p_two_sided'] if c['p_two_sided'] is not None else 'n/a'} "
            f"[{sig_tag}, {eq_tag}, cap={c['capacity_label']}]")
    logger.info(f'[exp11] wrote {CC_RESULTS_FILE} ({len(results)} arms, '
                f'{len(contrasts)} contrasts)')
    return 0


def _compact_overfit_report(report: dict) -> dict:
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def _append_ablation_manifest_compression(n_arms: int, return_code: int,
                                          n_seeds: int, steps: int) -> None:
    prov = _run_provenance(n_seeds, steps, plan_id=CC_PLAN_ID)
    append_run_manifest(make_run_manifest_entry(
        experiment='exp11_compression_ablation',
        seeds=prov['seeds'],
        provenance=prov,
        hyperparameters=prov['training'],
        data=prov['data'],
        artifacts={
            'results': CC_RESULTS_FILE,
            'provenance': CC_PROVENANCE_FILE,
            'contrasts': CC_CONTRASTS_FILE,
        },
        results={'n_arms': n_arms, 'return_code': return_code},
        status='completed' if not return_code else 'partial',
    ), logger=logger)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    sys.exit(main())
