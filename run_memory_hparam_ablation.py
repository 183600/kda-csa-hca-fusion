"""Experiment 10 — memory-operator hyperparameter ablation.

Motivation
----------
Exp 5b ablates *which* memory operator fills the recurrent slots (KDA vs
Mamba-2 vs RWKV-7) at fixed default geometry. It does NOT ablate the
*internal hyperparameters* of those operators, which are at least as
consequential for quality vs latency:

    * KDA: ``kda_chunk_size`` (associative-scan chunk boundary), ``kda_decay_scale``
      (per-channel log-decay magnitude), ``kda_decay_init`` (default vs log_uniform
      multi-timescale init).
    * Mamba-2: ``mamba_d_state`` (SSM state dimension), ``mamba_expand`` (inner-dim
      expansion), ``mamba_headdim`` (per-head state width).

All of these change FLOPs/state and can change MQAR accuracy independently of
the operator identity, so a claim like "KDA beats Mamba-2" is confounded unless
each operator is also evaluated across its own geometry grid.

Design
------
* Fixed 3:1:1 hybrid layout (n_kda=3, n_csa=1, n_hca=1), 5 layers,
  ``SEQ_LEN=16`` MQAR, same training recipe as Exp 5 (100 steps, AdamW 3e-3,
  multi-seed).
* KDA grid: ``kda_chunk_size`` ∈ {32, 64, 128} × ``kda_decay_scale`` ∈
  {0.05, 0.1, 0.2} × ``kda_decay_init`` ∈ {default, log_uniform} = 18 arms,
  each vs the KDA reference (64, 0.1, default).
* Mamba-2 grid: ``mamba_d_state`` ∈ {16, 64, 256} × ``mamba_expand`` ∈
  {1, 2, 4} × ``mamba_headdim`` ∈ {32, 64, 128} = 27 arms, each vs the
  Mamba-2 reference (16, 2, 16).
* Each arm runs ``ABL_SEEDS`` seeds (default 7) paired with every other arm
  at the same seed, so the primary statistic is a seed-paired paired t-test
  against the arm's own reference, Bonferroni-corrected within the operator
  family, plus TOST equivalence at ``CONTRAST_EQUIVALENCE_MARGIN = 0.05``.
* Parameter-capacity audit: every arm records ``n_params``; an arm whose
  parameter count differs from its reference by more than
  ``CAPACITY_MATCH_TOLERANCE = 5%`` is labelled ``operator_plus_capacity``
  (same policy as Exp 5b) so a geometry-driven quality change is not read as
  a pure mechanism effect.

Environment variables
---------------------
* ``MH_ABL_STEPS`` (default 100), ``MH_ABL_SEEDS`` (default 7),
  ``MH_ABL_TRAIN_BATCH`` (default 16), ``MH_ABL_SEQ_LEN`` (default 16).
* ``MH_KDA=0`` skips the KDA grid; ``MH_MAMBA2=0`` skips the Mamba-2 grid.
* ``MH_OVERFIT_CHECK`` / ``MH_OVERFIT_STRICT`` / ``MH_OVERFIT_STEPS`` /
  ``MH_OVERFIT_SEED`` / ``MH_OVERFIT_BATCH`` (mirror Exp 5's ABL_* gate).
* ``MH_DEVICE`` (default ``cpu``; ``cuda`` when a GPU is available).

Outputs
-------
``results/exp10_memory_hparam_ablation.json`` — envelope
``{metadata, results: [...]}`` with per-arm per-seed rows plus an aggregated
summary row per arm and a ``contrasts`` block (arm vs its own reference,
paired t-test, Bonferroni, TOST, capacity label).
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
from ops_fused import HybridAttentionStack, HybridConfig
from ops_kda_backend import validate_kda_backend
from ops_mamba_backend import validate_mamba_backend
from run_ablation import (
    VOCAB, SEQ_LEN, CONTRAST_EQUIVALENCE_MARGIN, CAPACITY_MATCH_TOLERANCE,
    _make_cfg, _run_provenance, _run_ablation_overfit_gate,
    _attach_curve_diagnostics, _eval_model, eval_layout_one_seed,
    summarize_layout_results, run_overfit_sanity_check,
    _OVERFIT_BATCH_SEED_OFFSET, ERROR_ANALYSIS_SEED_OFFSET,
    VAL_MONITOR_SEED_OFFSET,
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
from run_ablation import (_aggregate_curve_diagnosis as
                          _run_ablation_aggregate_curve_diagnosis)

logger = logging.getLogger(__name__)

_REF_CACHE: dict[str, list[dict]] = {}

MH_PLAN_ID = 'exp10_memory_hparam'
MH_RESULTS_FILE = 'results/exp10_memory_hparam_ablation.json'
MH_PROVENANCE_FILE = 'results/exp10_memory_hparam_provenance.json'
MH_CONTRASTS_FILE = 'results/exp10_memory_hparam_contrasts.json'

# Reference geometries: (kda_chunk_size, kda_decay_scale, kda_decay_init)
# and (mamba_d_state, mamba_expand, mamba_headdim). The first entry of each
# grid is the reference arm; every other arm is tested against it.
KDA_REFERENCE = {'kda_chunk_size': 64, 'kda_decay_scale': 0.1,
                 'kda_decay_init': 'default'}
KDA_GRID = {
    'kda_chunk_size': [32, 64, 128],
    'kda_decay_scale': [0.05, 0.1, 0.2],
    'kda_decay_init': ['default', 'log_uniform'],
}

MAMBA2_REFERENCE = {'mamba_d_state': 16, 'mamba_expand': 2,
                    'mamba_headdim': 16}
MAMBA2_GRID = {
    'mamba_d_state': [16, 64, 256],
    'mamba_expand': [1, 2, 4],
    'mamba_headdim': [32, 64, 128],
}

# TOST equivalence margin (shared with Exp 5/5b so the two experiments sit
# in one pre-declared family).
MH_EQUIV_MARGIN = CONTRAST_EQUIVALENCE_MARGIN


def _memory_hparam_override(operator: str, values: dict) -> dict:
    """Build the cfg_overrides dict for one arm of the memory-hparam grid.

    Only the fields relevant to ``operator`` ('kda' or 'mamba2') are
    touched; every other HybridConfig field keeps its Exp-5 default so the
    ablation isolates the single geometry knob.
    """
    overrides: dict = {}
    if operator == 'kda':
        for key in ('kda_chunk_size', 'kda_decay_scale', 'kda_decay_init'):
            if key in values:
                overrides[key] = values[key]
    elif operator == 'mamba2':
        for key in ('mamba_d_state', 'mamba_expand', 'mamba_headdim'):
            if key in values:
                overrides[key] = values[key]
    else:
        raise ValueError(f"unknown operator {operator!r}")
    return overrides


def _arm_label(operator: str, values: dict) -> str:
    """Human-readable arm label for results and contrast names."""
    parts = []
    if operator == 'kda':
        parts.append(f"chunk={values.get('kda_chunk_size')}")
        parts.append(f"decay_scale={values.get('kda_decay_scale')}")
        parts.append(f"decay_init={values.get('kda_decay_init')}")
    elif operator == 'mamba2':
        parts.append(f"d_state={values.get('mamba_d_state')}")
        parts.append(f"expand={values.get('mamba_expand')}")
        parts.append(f"headdim={values.get('mamba_headdim')}")
    return f"{operator}: " + ", ".join(parts)


def _enumerate_grid(operator: str) -> list[dict]:
    """Return the list of arm value-dicts for ``operator`` (reference first)."""
    grid = KDA_GRID if operator == 'kda' else MAMBA2_GRID
    reference = KDA_REFERENCE if operator == 'kda' else MAMBA2_REFERENCE
    keys = list(grid.keys())
    arms: list[dict] = []
    import itertools
    for combo in itertools.product(*(grid[k] for k in keys)):
        values = dict(zip(keys, combo))
        arms.append(values)
    # Put the reference arm first so contrast indexing is stable.
    arms.sort(key=lambda v: 0 if v == reference else 1)
    return arms


def _build_hparam_model(operator: str, values: dict, d_model: int,
                        device) -> tuple[nn.Module, nn.Module, nn.Embedding]:
    """Construct (model, head, embed) for one memory-hparam arm.

    Uses the explicit-layout path of _make_matrix_cfg so the 3:1:1 layout
    is fixed and the memory route is the requested operator; ``cfg_overrides``
    injects the arm's geometry knobs. This mirrors eval_group's construction
    path exactly.
    """
    from run_ablation import _make_matrix_cfg, SimpleBaselineStack
    layout = ('memory', 'memory', 'memory', 'csa', 'hca')
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg_overrides = _memory_hparam_override(operator, values)
    cfg = _make_matrix_cfg(d_model, operator, layout, cfg_overrides=cfg_overrides)
    model = HybridAttentionStack(cfg, total_layers=len(layout)).to(device)
    return model, head, embed


def _train_eval_hparam_arm(operator: str, values: dict, d_model: int,
                           seq_len: int, n_kv: int, steps: int, lr: float,
                           seed: int, device, eval_batches: int = 4,
                           eval_batch: int = 64, train_batch: int | None = None,
                           val_every: int = 0, val_batches: int = 2,
                           val_batch: int = 32,
                           error_batches: int = 0, error_batch: int = 64,
                           max_error_examples: int = 2) -> dict:
    """Train + evaluate ONE memory-hparam arm at ONE seed.

    Mirrors eval_layout_one_seed's protocol (same RNG stream offsets, same
    held-out curve, same bad-case pass, same latency probe) so every arm is
    directly comparable to every other arm and to Exp 5 rows.
    """
    if isinstance(device, str):
        device = torch.device(device)
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    layout = ('memory', 'memory', 'memory', 'csa', 'hca')
    cfg_overrides = _memory_hparam_override(operator, values)
    from run_ablation import _make_matrix_cfg
    cfg = _make_matrix_cfg(d_model, operator, layout,
                           cfg_overrides=cfg_overrides)
    total = len(layout)
    model = HybridAttentionStack(cfg, total_layers=total).to(device)
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout_str = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)
    if train_batch is None:
        train_batch = parse_int_env('MH_ABL_TRAIN_BATCH', 16, min_value=1,
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
        label=f'{operator} {values} seed={seed}', ceiling=1.0)
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
                context=f'step={step}, arm={operator}/{values}, seed={seed}')
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
                context=f'step={step}, arm={operator}/{values}, seed={seed}')
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
    x = torch.randn(1, seq_len, d_model, device=device, generator=lat_gen) * 0.1
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
        train_losses=losses, train_accs=train_accs, op_name='memory_hparam',
        seq_len=seq_len, n_kv=n_kv, device=device, steps=steps,
        error_batches=error_batches, error_batch=error_batch,
        max_error_examples=max_error_examples, seed=seed,
        label=f'{operator} {values} n_kv={n_kv}')

    return {
        'operator': operator,
        'values': values,
        'label': _arm_label(operator, values),
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


def _summarize_arm(operator: str, values: dict, per_seed: list[dict],
                   seeds: list[int], reference_values: dict,
                   reference_per_seed: list[dict] | None = None) -> dict:
    """Aggregate one arm's per-seed rows into a summary row.

    If ``reference_per_seed`` is given (the arm's own reference at the same
    seeds), also compute the seed-paired contrast: mean diff, paired t-test,
    Bonferroni-corrected significance, TOST equivalence, and the capacity
    label. Returns the summary row plus an optional ``contrast`` block.
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
            f"all {len(per_seed)} seeds failed for arm "
            f"{operator}/{values}; "
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
                      label=f'{operator} {values} n_kv={ok_per_seed[0]["n_kv"]}')

    summary = {
        'operator': operator,
        'values': values,
        'label': _arm_label(operator, values),
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

    # Paired contrast vs this arm's own reference (same seeds).
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
                                          alpha=0.05 / 1)  # single contrast vs own reference
                sig = abs(t_stat) >= crit
                tost = tost_from_summary(mean_diff, sd_diff, len(diffs),
                                         MH_EQUIV_MARGIN)
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
                'contrast': f"{_arm_label(operator, values)} vs "
                            f"{_arm_label(operator, reference_values)}",
                'operator': operator,
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
    """Two-sided paired t p-value via the regularized incomplete beta.

    Thin local implementation so this module does not pull in SciPy; mirrors
    multi_seed_stats.paired_t_test's arithmetic but returns only the p-value
    for the inline contrast computation.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    # regularized incomplete beta I_x(df/2, 1/2) = two-sided p
    a = df / 2.0
    b = 0.5
    # continued fraction (Lentz) for the beta fraction
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


def _aggregate_curve_diagnosis(ok_per_seed: list[dict]) -> dict:
    """Modal curve verdict across surviving seeds (same as Exp 5's helper)."""
    return _run_ablation_aggregate_curve_diagnosis(ok_per_seed)


def _run_overfit_gate_hparam(device, seq_len: int) -> tuple[dict, bool]:
    """Pre-sweep overfit gate: one reference arm per operator."""
    build_models = []
    for operator in ('kda', 'mamba2'):
        reference = (KDA_REFERENCE if operator == 'kda'
                     else MAMBA2_REFERENCE)
        label = f'{operator} reference {reference}'
        def _build(operator=operator, reference=reference):
            return _build_hparam_model(operator, reference,
                                       int(os.environ.get('MH_D_MODEL', '32')),
                                       device)
        build_models.append((label, _build))
    return _run_ablation_overfit_gate(
        build_models, device, n_kv=1, env_prefix='MH', seq_len=seq_len)


def main() -> int:
    device = torch.device(os.environ.get('MH_DEVICE', 'cpu'))
    d_model = parse_int_env('MH_D_MODEL', 32, min_value=8, logger=logger)
    seq_len = parse_int_env('MH_ABL_SEQ_LEN', SEQ_LEN, min_value=8,
                            logger=logger)
    steps = parse_int_env('MH_ABL_STEPS', 100, min_value=10, logger=logger)
    n_seeds = parse_int_env('MH_ABL_SEEDS', 7, min_value=2, logger=logger)
    lr = 3e-3
    eval_batches = parse_int_env('MH_ABL_EVAL_BATCHES', 4, min_value=1,
                                 logger=logger)
    eval_batch = parse_int_env('MH_ABL_EVAL_BATCH', 64, min_value=1,
                               logger=logger)
    train_batch = parse_int_env('MH_ABL_TRAIN_BATCH', 16, min_value=1,
                                logger=logger)
    error_batches = parse_int_env('MH_ABL_ERROR_BATCHES', 0, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('MH_ABL_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    max_error_examples = parse_int_env('MH_ABL_ERROR_EXAMPLES', 2, min_value=1,
                                       logger=logger)

    run_kda = parse_int_env('MH_KDA', 1, min_value=0, logger=logger) != 0
    run_mamba2 = parse_int_env('MH_MAMBA2', 1, min_value=0, logger=logger) != 0

    logger.info(f'[exp10] device={device} d_model={d_model} seq_len={seq_len} '
                f'steps={steps} seeds={n_seeds} '
                f'kda={run_kda} mamba2={run_mamba2}')

    # --- Overfit gate (reference arms only) ---
    gate_record, proceed = _run_overfit_gate_hparam(device, seq_len)

    # --- Enumerate arms ---
    arms: list[dict] = []
    if run_kda:
        for values in _enumerate_grid('kda'):
            arms.append({'operator': 'kda', 'values': values})
    if run_mamba2:
        for values in _enumerate_grid('mamba2'):
            arms.append({'operator': 'mamba2', 'values': values})

    # Reference arm per operator (first entry of each grid).
    kda_ref = KDA_REFERENCE
    mamba2_ref = MAMBA2_REFERENCE

    results: list[dict] = []
    contrasts: list[dict] = []
    seeds = seed_ids(n_seeds, plan_id=MH_PLAN_ID)

    for arm in arms:
        operator = arm['operator']
        values = arm['values']
        reference_values = (kda_ref if operator == 'kda' else mamba2_ref)
        is_reference = (values == reference_values)

        logger.info(f'[exp10] arm {_arm_label(operator, values)} '
                    f'({"reference" if is_reference else "ablation"})')
        per_seed = []
        for s in seeds:
            r = _train_eval_hparam_arm(
                operator, values, d_model, seq_len, n_kv=1, steps=steps,
                lr=lr, seed=s, device=device, eval_batches=eval_batches,
                eval_batch=eval_batch, train_batch=train_batch,
                error_batches=error_batches, error_batch=error_batch,
                max_error_examples=max_error_examples)
            r['arm_is_reference'] = is_reference
            per_seed.append(r)

        # Reference per_seed (cached across arms of the same operator).
        reference_per_seed = None
        if not is_reference:
            ref_key = operator
            if ref_key not in _REF_CACHE:
                _REF_CACHE[ref_key] = [
                    _train_eval_hparam_arm(
                        operator, reference_values, d_model, seq_len,
                        n_kv=1, steps=steps, lr=lr, seed=s, device=device,
                        eval_batches=eval_batches, eval_batch=eval_batch,
                        train_batch=train_batch,
                        error_batches=error_batches, error_batch=error_batch,
                        max_error_examples=max_error_examples)
                    for s in seeds
                ]
            reference_per_seed = _REF_CACHE[ref_key]

        summary = _summarize_arm(
            operator, values, per_seed, seeds, reference_values,
            reference_per_seed)
        summary['arm_is_reference'] = is_reference
        results.append(summary)
        if summary.get('contrast'):
            contrasts.append(summary['contrast'])

    # --- Provenance + manifest ---
    prov = _run_provenance(n_seeds, steps, overfit_gate=gate_record,
                           plan_id=MH_PLAN_ID)
    prov['training']['lr'] = lr
    prov['data'] = {
        'kind': 'synthetic',
        'generator_function': make_mqar_batch.__name__,
        'task_spec': {'seq_len': seq_len, 'vocab': VOCAB},
        'seed_offsets': {'train': 1_000_000, 'eval': 5_000_000},
        'note': ('Synthetic batches are a deterministic function of '
                 '(generator, task spec, seed + offset).'),
    }
    prov['memory_hparam_grid'] = {
        'kda': {'reference': KDA_REFERENCE, 'grid': KDA_GRID},
        'mamba2': {'reference': MAMBA2_REFERENCE, 'grid': MAMBA2_GRID},
    }
    prov['equivalence_margin'] = MH_EQUIV_MARGIN
    prov['capacity_match_tolerance'] = CAPACITY_MATCH_TOLERANCE

    # --- Experiment-design audit ---
    try:
        design_audit = audit_experiment_design(
            MH_PLAN_ID,
            seeds=seeds,
            observed_factor_levels={
                'operator': sorted({a['operator'] for a in arms}),
                'kda_chunk_size': sorted({str(a['values'].get('kda_chunk_size'))
                                          for a in arms if a['operator'] == 'kda'}),
                'mamba_d_state': sorted({str(a['values'].get('mamba_d_state'))
                                         for a in arms if a['operator'] == 'mamba2'}),
            },
            family_size=len(contrasts),
            analyzed_seed_ids=seeds,
            pilot_sd=None,
        )
    except Exception as e:  # pragma: no cover - audit must not kill a run
        logger.warning(f'[exp10] experiment-design audit failed: {e}')
        design_audit = {'status': 'audit_error', 'error': str(e)}

    metadata = {
        'experiment': 'exp10_memory_hparam_ablation',
        'plan_id': MH_PLAN_ID,
        'device': str(device),
        'd_model': d_model,
        'seq_len': seq_len,
        'steps': steps,
        'n_seeds': n_seeds,
        'seeds': seeds,
        'lr': lr,
        'kda_reference': KDA_REFERENCE,
        'mamba2_reference': MAMBA2_REFERENCE,
        'equivalence_margin': MH_EQUIV_MARGIN,
        'capacity_match_tolerance': CAPACITY_MATCH_TOLERANCE,
        'arms': [{'operator': a['operator'], 'values': a['values'],
                  'label': a['label']} for a in arms],
        'n_arms': len(arms),
        'n_contrasts': len(contrasts),
        'design_audit': design_audit,
        'overfit_gate': _compact_overfit_report(gate_record),
        'proceed': proceed,
    }

    envelope = {'metadata': metadata, 'results': results,
                'contrasts': contrasts}

    os.makedirs('results', exist_ok=True)
    write_results_checked(envelope, MH_RESULTS_FILE)
    write_json_atomic(prov, MH_PROVENANCE_FILE)
    write_json_atomic({'metadata': metadata, 'contrasts': contrasts},
                      MH_CONTRASTS_FILE)

    _append_ablation_manifest_hparam(len(arms), 0, n_seeds, steps)

    # Console summary.
    for c in contrasts:
        sig_tag = 'SIG' if c['significant_bonferroni'] else 'ns'
        eq_tag = ('eq' if c['tost_equivalent'] else 'neq')
        logger.info(
            f"[exp10] contrast {c['contrast']}: "
            f"mean_diff={c['mean_diff']:+.4f} "
            f"t={c['t_stat'] if c['t_stat'] is not None else 'n/a'} "
            f"p2={c['p_two_sided'] if c['p_two_sided'] is not None else 'n/a'} "
            f"[{sig_tag}, {eq_tag}, cap={c['capacity_label']}]")
    logger.info(f'[exp10] wrote {MH_RESULTS_FILE} ({len(results)} arms, '
                f'{len(contrasts)} contrasts)')
    return 0


def _compact_overfit_report(report: dict) -> dict:
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def _append_ablation_manifest_hparam(n_arms: int, return_code: int,
                                     n_seeds: int, steps: int) -> None:
    prov = _run_provenance(n_seeds, steps, plan_id=MH_PLAN_ID)
    append_run_manifest(make_run_manifest_entry(
        experiment='exp10_memory_hparam_ablation',
        seeds=prov['seeds'],
        provenance=prov,
        hyperparameters=prov['training'],
        data=prov['data'],
        artifacts={
            'results': MH_RESULTS_FILE,
            'provenance': MH_PROVENANCE_FILE,
            'contrasts': MH_CONTRASTS_FILE,
        },
        results={'n_arms': n_arms, 'return_code': return_code},
        status='completed' if not return_code else 'partial',
    ), logger=logger)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    sys.exit(main())
