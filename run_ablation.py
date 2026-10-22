"""Experiment 5 — hybrid layout ablation (multi-seed, device-aware).

Two modes, selected by ``ABL_MODE``:

* ``ratio`` (default — the historical behaviour): sweeps KDA:CSA:HCA
  ratios in the fused hybrid block at fixed 5-layer depth and measures
  (a) end-to-end forward latency and (b) MQAR accuracy.
* ``matrix`` (the eight-group main experiment from the framework design):
  a fixed-depth explicit-layout matrix crossing the memory route
  (KDA vs. Mamba-2 vs. RWKV-7) with four layout groups —

        A: R R R R R     (pure recurrent baseline)
        B: R R R CSA R   (CSA replaces a fixed recurrent slot)
        C: R R R R HCA   (HCA replaces a fixed recurrent slot)
        D: R R R CSA HCA (full hybrid)

  where R is the recurrent memory operator selected by ``memory_type``.
  The default ``ABL_MEMORY=kda,mamba2,rwkv7`` yields the TWELVE-group
  total table (the RWKV-7 route is the KDA-replacement ablation:
  same depth/width/layout/training, only the memory mixer swapped);
  ``ABL_MEMORY=kda,rwkv7`` restricts to the eight-group main
  experiment. All groups share the SAME total depth
  (``ABL_TOTAL_LAYERS``, default 10 = the 5-layer unit repeated
  twice), the same outer layer structure, and the same training
  recipe, so CSA's and HCA's contributions are ablated independently
  of the memory operator. Paired seed-matched contrasts
  (B-A / C-A / D-B / D-C, the I=D-B-C+A interaction per route, and
  the cross-route A/D replacement differences) are computed, tested
  with a Bonferroni-corrected seed-paired t-test against their
  PRE-REGISTERED hypotheses (``MATRIX_CONTRAST_HYPOTHESES``), and
  written to ``results/exp5b_ablation_matrix_contrasts.json``. A raw
  t-statistic is not evidence: only rows with
  ``significant_bonferroni=true`` support their prediction (or, for
  the two-sided cross-route rows, show a significant difference).
  Additional presets (``ABL_MATRIX=csa_csa | position | budget``)
  cover the follow-up ablations: HCA-vs-more-CSA, layer placement,
  and compression-budget allocation at fixed cache bytes.

Methodology (both modes):

  * **Multi-seed runs.** Each configuration is trained over ``n_seeds``
    seeds with mean +/- CI95 reported, so the comparison is statistically
    meaningful.
  * **Training steps.** 100+ steps with convergence curves (a single-seed,
    25-step setting is too short for reliable convergence).
  * **Controlled depth.** Every configuration contains exactly the same
    number of layers, so layout effects are not confounded with model
    depth. ``n_params`` is still reported because operator types have
    different parameter counts — same d_model does NOT mean same
    capacity, and the budget-matching experiment exists precisely to
    close that gap.
  * **Tuned simple baselines.** In matrix mode the linear/softmax rungs
    get the same depth/recipe/budget, and the softmax lr is selected on
    the held-out validation stream (``ABL_BASELINE_TUNE``,
    ``ABL_BASELINE_LRS``, ``ABL_BASELINE_TUNE_SEEDS``) before the main
    sweep. Every non-baseline group (including the pure recurrent
    control) additionally gets a seed-paired one-sided test against that
    tuned softmax under its own
    Bonferroni family (``TUNED_BASELINE_HYPOTHESIS``), recorded in the
    matrix metadata; ``ABL_BASELINE_TUNE=0`` preserves the historical
    fixed lr=3e-3 and skips the tuning pass.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available.
  * **Equivalence, not just significance.** Every paired contrast also
    gets a TOST equivalence verdict against the PRE-SPECIFIED margin
    ``CONTRAST_EQUIVALENCE_MARGIN`` (env override ``ABL_EQUIV_MARGIN``;
    ``0`` disables it). A non-significant contrast is inconclusive on
    both questions; only ``tost_equivalent=True`` supports "no practical
    difference" (see ``docs/equivalence-testing.md``).
  * **Results analysis is part of the run, not an afterthought.** Every
    seed records a held-out validation curve (read-only; the headline
    ``final_acc`` keeps its exact fixed-budget meaning) which is labelled
    overfitting / underfitting / healthy / saturated; every row pools the
    per-seed bad-case reports (which mistake dominates); every paired
    contrast carries an accuracy-vs-cost verdict next to its corrected
    significance. The bad-case pass is ON by default (2 batches x 64
    held-out samples per seed, dedicated RNG stream) and can be tuned or
    disabled with ``ABL_ERROR_BATCHES`` / ``ABL_ERROR_BATCH`` /
    ``ABL_ERROR_EXAMPLES``.
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
from ops_fused import HybridAttentionStack, HybridKCHAttention, HybridConfig
from ops_kda_backend import validate_kda_backend
from ops_mamba_backend import validate_mamba_backend
from ops_rwkv import RWKV7_UPSTREAM_BINDING
from ops_rwkv_backend import validate_rwkv_backend
from run_quality import (
    make_mqar_batch, MQARHead, _parse_nkv_list, _fmt_tstat, _t_crit_975,
    _bonferroni_crit_q, _build_param_groups, _task_batch_fingerprint,
    _parse_equivalence_margin, _parse_lr_list, SMALL_MODEL_SPEC,
    analyze_errors, ResidualAttnLayer, NoMixAttn, SoftmaxAttn,
)
from results_analysis import multiclass_classification_metrics
from train_diagnostics import (
    DEFAULT_COST_KEYS, DEFAULT_MAX_COST_RATIO, DEFAULT_MIN_EFFECT,
    ERROR_ANALYSIS_SEED_OFFSET, VAL_MONITOR_SEED_OFFSET, ValPoint,
    ValidationCurve, accuracy_cost_tradeoff, log_pooled_errors, log_tradeoff,
    optimizer_update_report, pool_error_analyses, raise_non_finite,
    snapshot_parameters, summarize_gradient_norms, triage_gradient,
)
from multi_seed_stats import (
    min_detectable_effect, seeds_required, tost_from_summary,
    _safe_mean, _sample_sd,
)
from experiment_design import (audit_run as audit_experiment_design,
                               common_analyzed_seeds)

logger = logging.getLogger(__name__)

VOCAB = 16
SEQ_LEN = 16

# Fixed seed offset for the overfit gate's batch generator. Far away from
# the training offset (1_000_000) and the eval offset (5_000_000) so the
# gate's data stream can never collide with either.
_OVERFIT_BATCH_SEED_OFFSET = 3_000_000

# Pre-specified TOST equivalence margin for the paired contrast families,
# in absolute ``final_acc`` units (the scale of every ``mean_diff``). The
# task chance rate is 1/16 = 0.0625, so a five-accuracy-point difference
# is a substantial fraction of the whole learnable range: differences
# below it are treated as the smallest effect that would NOT change any
# practical conclusion (e.g. for the KDA-vs-RWKV-7 replacement question).
# Chosen BEFORE the numbers are read; the only legitimate override is the
# run-start environment variable ``ABL_EQUIV_MARGIN`` (``0`` disables
# equivalence testing, leaving every non-significant contrast
# inconclusive). Registering it here keeps superiority and equivalence
# verdicts on one pre-declared family.
CONTRAST_EQUIVALENCE_MARGIN = 0.05


def _fmt_num(value, prec: int = 4) -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.{prec}f}'


def _make_cfg(d_model=32, ratio=(3, 1, 1)):
    if not (isinstance(ratio, tuple) and len(ratio) == 3):
        raise ValueError(
            f"ratio must be a 3-tuple (n_kda, n_csa, n_hca), got {ratio!r}")
    if not all(isinstance(n, int) and n >= 0 for n in ratio):
        raise ValueError(
            f"ratio components must be non-negative ints, got {ratio!r}")
    n_kda, n_csa, n_hca = ratio
    spec = SMALL_MODEL_SPEC
    return HybridConfig(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=spec['csa_m'], csa_topk=spec['csa_topk'],
        csa_nh=spec['csa_nh'], csa_c=spec['csa_c'],
        csa_dc=spec['csa_dc'], csa_nIh=spec['csa_nIh'],
        csa_cI=spec['csa_cI'],
        csa_sliding_window=spec['csa_sliding_window'],
        hca_m2=spec['hca_m2'], hca_nh=spec['hca_nh'],
        hca_c=spec['hca_c'], hca_dc=spec['hca_dc'],
        hca_sliding_window=spec['hca_sliding_window'],
        n_kda=n_kda, n_csa=n_csa, n_hca=n_hca,
        kda_backend=validate_kda_backend(
            os.environ.get('KDA_BACKEND', 'reference')
        ),
    )


def _run_provenance(n_seeds, steps, overfit_gate=None):
    """Environment provenance + seed list + training recipe for side files.

    Each side file records the code/environment alongside the seeds and the
    training recipe that produced the adjacent results file, so the pair is
    a complete run record. ``overfit_gate`` (the record returned by
    ``_run_ablation_overfit_gate``) is embedded so the sanity-gate verdict
    travels with the run record.
    """
    prov = capture_provenance()
    prov['seeds'] = [42 + i for i in range(int(n_seeds))]
    prov['training'] = {
        'optimizer': 'AdamW',
        'lr': 3e-3,
        'weight_decay_decay_group': 0.01,
        'weight_decay_no_decay_group': 0.0,
        'grad_clip_norm': 1.0,
        'steps': int(steps),
    }
    # Memory-route implementation provenance: which backends were
    # REQUESTED for this run (the effective backend is recorded per
    # result row / on each mixer's effective_backend attribute), plus
    # the RWKV-7 upstream binding (implementation commits) so every run
    # record answers 'which RWKV-7 implementation produced this'.
    prov['memory_backends_requested'] = {
        'kda': os.environ.get('KDA_BACKEND', 'reference'),
        'mamba2': os.environ.get('MAMBA_BACKEND', 'reference'),
        'rwkv7': os.environ.get('RWKV_BACKEND', 'reference'),
    }
    prov['rwkv7_upstream_binding'] = RWKV7_UPSTREAM_BINDING
    # Data version block: MQAR batches are synthetic and fully determined by
    # (generator function, task spec, seed + offset). Recording them makes
    # the data version explicit instead of implicit in the code revision.
    prov['data'] = {
        'kind': 'synthetic',
        'generator_function': make_mqar_batch.__name__,
        'task_spec': {'seq_len': SEQ_LEN, 'vocab': VOCAB},
        'seed_offsets': {'train': 1_000_000, 'eval': 5_000_000},
        'note': ('Synthetic batches are a deterministic function of '
                 '(generator, task spec, seed + offset); no external dataset '
                 'version applies. Code revision in this provenance pins the '
                 'generator implementation.'),
    }
    if overfit_gate is not None:
        prov['overfit_sanity_check'] = overfit_gate
    return prov


def _append_ablation_manifest(mode, return_code, n_seeds, steps,
                              n_kv_list):
    """Append one entry for this ablation run to the shared JSONL manifest.

    Turns the per-mode provenance side file into a campaign-level index
    line (seeds, recipe, data version, artifact paths, status), so the
    ablation is findable without opening its result file first.
    """
    prov = _run_provenance(n_seeds, steps)
    if mode == 'matrix':
        experiment = 'exp5b_ablation_matrix'
        artifacts = {
            'results': 'results/exp5b_ablation_matrix.json',
            'provenance': 'results/exp5b_ablation_matrix_provenance.json',
            'contrasts': 'results/exp5b_ablation_matrix_contrasts.json',
            'overfit_gate_results': 'results/exp5b_overfit_gate.json',
        }
    else:
        experiment = 'exp5_ratio_ablation'
        artifacts = {
            'results': 'results/exp5_ablation.json',
            'provenance': 'results/exp5_ablation_provenance.json',
            'overfit_gate_results': 'results/exp5_overfit_gate.json',
        }
    append_run_manifest(make_run_manifest_entry(
        experiment=experiment,
        seeds=prov['seeds'],
        provenance=prov,
        hyperparameters=prov['training'],
        data=prov['data'],
        artifacts=artifacts,
        results={
            'mode': mode,
            'n_kv': list(n_kv_list),
            'return_code': return_code,
        },
        status='completed' if not return_code else 'partial',
    ), logger=logger)


def _eval_model(model, head, embed, seq_len, n_kv=1, device='cpu',
                n_batches=4, batch=64, seed=42, seed_offset=5_000_000,
                collect_predictions=False):
    was_training = {m: m.training for m in (model, head, embed)}
    preds_all, target_all, scores_all = [], [], []
    try:
        model.eval()
        head.eval()
        embed.eval()
        correct, total = 0, 0
        losses = []
        # Use a distinct, far-separated offset (5_000_000 by default) for the
        # eval RNG so it never collides with the *next* seed's training stream
        # (``seed + 1_000_000``), keeping the eval batches fresh and the
        # seeds independent for the multi-seed t-test. This mirrors the
        # RNG isolation in ``run_quality.py``.
        # ``seed_offset`` is parameterized so the held-out VALIDATION monitor
        # (train_diagnostics.VAL_MONITOR_SEED_OFFSET = 6_000_000) draws from
        # its own stream instead of re-using the batches the headline
        # ``final_acc`` is measured on — otherwise "generalization gap"
        # would compare a model against its own training stream.
        eval_gen = make_seeded_generator(seed + seed_offset, device=device)
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, VOCAB, embed, device, generator=eval_gen)
                model.reset_state()
                h = model(x_emb)
                logits = head(h, cue_pos)
                pred = logits.argmax(-1)
                correct += (pred == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
                if collect_predictions:
                    preds_all.append(pred.detach().cpu())
                    target_all.append(target.detach().cpu())
                    scores_all.append(
                        torch.softmax(logits.detach().float(), dim=-1).cpu())
        if total == 0 or not losses:
            if collect_predictions:
                return 0.0, 0.0, None
            return 0.0, 0.0
        if collect_predictions:
            extra = {
                'pred': torch.cat(preds_all),
                'target': torch.cat(target_all),
                'scores': torch.cat(scores_all),
            }
            return correct / total, sum(losses) / len(losses), extra
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def _build_layout_model(ratio, d_model, device):
    """Construct (model, head, embed) exactly as ``eval_layout`` does.

    Factored out so the overfit gate builds models through the same
    construction path (and therefore the same init distribution) as the
    sweep it guards.
    """
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio)
    model = HybridKCHAttention(cfg, total_layers=sum(ratio)).to(device)
    return model, head, embed


SIMPLE_BASELINE_KINDS = ('linear', 'softmax')


class SimpleBaselineStack(nn.Module):
    """Simplest-baseline ladder for the matrix: linear or dense softmax.

    ``linear`` stacks identity token mixers (position-wise only; cannot
    move information across time, so it MUST stay at chance on MQAR if the
    task really requires memory). ``softmax`` stacks causal dense
    attention, the standard architecture families are compared against.
    Both get the SAME outer residual/LayerNorm structure, depth, data
    stream, optimizer and budget as every hybrid group, so a complex
    operator that fails to beat them has no demonstrated advantage.
    """

    def __init__(self, kind: str, d_model: int, total_layers: int):
        super().__init__()
        if kind not in SIMPLE_BASELINE_KINDS:
            raise ValueError(
                f'SimpleBaselineStack: kind must be one of '
                f'{SIMPLE_BASELINE_KINDS}, got {kind!r}')
        self.kind = kind
        self.total_layers = int(total_layers)
        layers = []
        for _ in range(self.total_layers):
            attn = (NoMixAttn() if kind == 'linear'
                    else SoftmaxAttn(d_model, H=2, K=16, V=16))
            layers.append(ResidualAttnLayer(attn, d_model))
        self.layers = nn.ModuleList(layers)

    def reset_state(self) -> None:
        pass

    def layout_str(self) -> str:
        return '-'.join([self.kind.upper()] * self.total_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_simple_baseline_groups(total_layers: int) -> list:
    """Baseline matrix groups (same depth/outer structure as the hybrids)."""
    groups = []
    for kind in SIMPLE_BASELINE_KINDS:
        layout = (kind,) * int(total_layers)
        desc = (f'{kind} baseline: {total_layers} x '
                f'{"identity mixer" if kind == "linear" else "causal dense softmax attention"} '
                f'(same depth/training recipe as the hybrid groups)')
        groups.append((f'BASELINE-{kind}', kind, layout, desc))
    return groups


def _build_matrix_model(spec, d_model, device):
    """Construct (model, head, embed) exactly as ``eval_group`` does."""
    memory_type, layout = spec[1], spec[2]
    cfg_overrides = spec[4] if len(spec) > 4 else None
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    if memory_type in SIMPLE_BASELINE_KINDS:
        model = SimpleBaselineStack(
            memory_type, d_model, len(layout)).to(device)
    else:
        cfg = _make_matrix_cfg(d_model, memory_type, layout, cfg_overrides)
        model = HybridAttentionStack(cfg, total_layers=len(layout)).to(device)
    return model, head, embed


def run_overfit_sanity_check(build_model, make_batch, *, label, seed=2026,
                             steps=100, train_batch=32, device='cpu',
                             lr=3e-3, grad_clip=1.0, acc_threshold=0.99,
                             loss_ratio_threshold=0.5, log_every=0,
                             seq_len=SEQ_LEN):
    """Overfit ONE fixed batch as a pre-sweep gradient-flow gate.

    Trains a FRESH model (built by ``build_model`` immediately after a
    ``set_global_seed(seed, deterministic=True)`` call, so the init is
    reproducible) on a single fixed synthetic batch with all
    regularization disabled (``weight_decay=0``, no dropout anywhere in
    these operators, no data augmentation). The batch is regenerated
    deterministically every step from ``seed +
    _OVERFIT_BATCH_SEED_OFFSET``, so the autograd graph — including the
    embedding's gradient path — is fresh each step while the DATA is held
    constant (mirrors ``run_quality.run_overfit_sanity_check``).

    ``build_model()`` must return ``(model, head, embed)`` and
    ``make_batch(batch, embed, generator)`` must return
    ``(x_emb, target, cue_pos)``; both callables let this single gate
    guard the layout sweep, the mechanism ablation, and the recall-reach
    probe without duplicating the loop.

    Pass criterion (either): tail-mean train accuracy >= ``acc_threshold``
    (default 0.99), OR tail-mean loss < ``loss_ratio_threshold`` times the
    head-mean loss.

    This is a NECESSARY, NOT SUFFICIENT check: passing proves gradients
    flow and the loss responds on a fixed batch; augmentation,
    feature-alignment and label-mapping bugs can still pass it.

    The caller re-seeds per sweep cell anyway, so the gate cannot perturb
    the experiment's numbers.
    """
    import gc

    if isinstance(device, str):
        device = torch.device(device)
    steps = int(steps)
    if steps < 10:
        raise ValueError(
            f"overfit check needs steps >= 10 to be meaningful, got {steps}")
    train_batch = int(train_batch)
    if train_batch < 1:
        raise ValueError(f"train_batch must be >= 1, got {train_batch}")
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    seed_info = set_global_seed(seed, deterministic=True)
    model, head, embed = build_model()
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.0)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    model.train()
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
        x_emb, target, cue_pos = make_batch(train_batch, embed, gen)
        if first_batch is None:
            first_batch = (x_emb.detach(), target.detach(), cue_pos)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                context=f'overfit sanity check step={step} {label}')
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            # One-shot audit on the first backward: names the parameters the
            # loss never reached, so a hidden detached branch is visible even
            # when the loss criterion passes (mirrors the LM gate).
            grad_flow = grad_flow_report(
                list(model.named_parameters())
                + [(f'head.{n}', p) for n, p in head.named_parameters()]
                + [(f'embed.{n}', p) for n, p in embed.named_parameters()])
        named_params = (
            list(model.named_parameters())
            + [(f'head.{n}', p) for n, p in head.named_parameters()]
            + [(f'embed.{n}', p) for n, p in embed.named_parameters()])
        bad_grads = triage_gradient(params)
        clip_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                grads={f'p{i}': p.grad for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'overfit sanity check step={step} {label}')
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
            print(f"[overfit-check] {label} step {step:4d} "
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
        'label': label,
        'seq_len': int(seq_len),
        'vocab': VOCAB,
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
        # Optimizer coverage/update audit (first step): a param missing from
        # the optimizer, or a non-zero-gradient param that never moved, now
        # fails the gate instead of hiding behind a falling loss.
        'optimizer_updates': update_report,
        'note': ('Necessary but NOT sufficient: passing proves gradients '
                 'flow and the loss responds on a fixed batch; it cannot '
                 'rule out data-augmentation / feature-alignment / '
                 'label-mapping bugs.'),
    }
    del model, head, embed, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if grad_flow and grad_flow['n_missing_grad']:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the fixed batch for {label} "
              f"(e.g. {grad_flow['missing_grad_names'][:3]})", flush=True)
    return report


def _compact_overfit_report(report):
    """Strip per-step curves for compact metadata embedding."""
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'acc_curve', 'elapsed_s', 'seed_info')}


def _run_ablation_overfit_gate(build_models, device, *, n_kv=1,
                               env_prefix='ABL', make_task_batch=None,
                               seq_len=SEQ_LEN):
    """Run the pre-sweep overfit gate(s) and return ``(record, proceed)``.

    ``build_models`` is a list of ``(label, build_model)`` pairs; the gate
    runs once per entry so every distinct layout/mechanism that will be
    swept gets its own verdict (a broken CSA/HCA gradient path must not be
    hidden behind a passing pure-recurrent layout).

    Environment controls (mirroring ``run_quality``'s MQAR gate); the
    ``{env_prefix}`` is ``ABL`` for the Exp 5 sweep and reused with the
    experiment's own prefix by the other hybrid runners:
      * ``{env_prefix}_OVERFIT_CHECK=1|0``  (default 1: run the gate)
      * ``{env_prefix}_OVERFIT_STRICT=1|0`` (default 1: refuse to start the
        sweep when any gate cell fails)
      * ``{env_prefix}_OVERFIT_STEPS=N``    (default 100 optimizer steps)
      * ``{env_prefix}_OVERFIT_SEED=N``     (default 2026: model + batch)
      * ``{env_prefix}_OVERFIT_BATCH=N``    (default 32 sequences)
    """
    enabled = parse_int_env(f'{env_prefix}_OVERFIT_CHECK', 1, min_value=0,
                            logger=logger) != 0
    strict = parse_int_env(f'{env_prefix}_OVERFIT_STRICT', 1, min_value=0,
                           logger=logger) != 0
    if not enabled:
        logger.info(f'  overfit sanity gate disabled '
                    f'({env_prefix}_OVERFIT_CHECK=0)')
        return {'enabled': False, 'strict': bool(strict), 'reports': []}, True
    steps = parse_int_env(f'{env_prefix}_OVERFIT_STEPS', 100, min_value=10,
                          logger=logger)
    seed = parse_int_env(f'{env_prefix}_OVERFIT_SEED', 2026, min_value=0,
                         logger=logger)
    batch = parse_int_env(f'{env_prefix}_OVERFIT_BATCH', 32, min_value=1,
                          logger=logger)
    if make_task_batch is None:
        def make_task_batch(b, embed, gen):
            return make_mqar_batch(b, SEQ_LEN, n_kv, VOCAB, embed, device,
                                   generator=gen)
    logger.info(f'  overfit sanity gate: {len(build_models)} cell(s), '
                f'{steps} steps on one fixed batch each '
                f'(batch={batch}, seed={seed}, strict={bool(strict)})')
    reports = []
    all_passed = True
    for label, build_model in build_models:
        try:
            report = run_overfit_sanity_check(
                build_model, make_task_batch,
                label=label, seed=seed, steps=steps, train_batch=batch,
                device=device, log_every=max(10, steps // 5),
                seq_len=seq_len)
        except Exception as e:
            logger.error(f'  [overfit] {label}: gate crashed: {e}')
            report = {'passed': False, 'label': label, 'error': str(e),
                      'steps': steps, 'seed': seed, 'train_batch': batch}
        reports.append(report)
        if report.get('passed'):
            logger.info(
                f"  [overfit] {label}: PASS "
                f"(acc={report['final_train_acc']:.4f}, "
                f"loss {report['initial_loss']:.4f}->"
                f"{report['final_loss']:.4f})")
        else:
            all_passed = False
            logger.error(
                f"  [overfit] {label}: FAIL "
                f"(acc={report.get('final_train_acc')}, "
                f"loss={report.get('final_loss')}, "
                f"error={report.get('error')})")
    record = {'enabled': True, 'strict': bool(strict),
              'steps': steps, 'seed': seed, 'train_batch': batch,
              'reports': reports}
    if not all_passed and strict:
        logger.error(
            f'  {env_prefix}_OVERFIT_STRICT=1: refusing to start the sweep '
            f'on a recipe that cannot overfit a fixed batch. Fix the '
            f'training path, or set {env_prefix}_OVERFIT_STRICT=0 to '
            f'record the failure and continue (not recommended).')
        return record, False
    return record, True


def _attach_curve_diagnostics(*, curve, model, head, embed, train_losses,
                              train_accs, op_name, seq_len, n_kv, device,
                              steps, error_batches, error_batch,
                              max_error_examples, seed, label):
    """Close the held-out curve and run the bad-case pass for one row.

    Factored out so ``eval_layout`` and ``eval_group`` record *identical*
    diagnostic blocks (same keys, same streams, same definitions), the way
    they already share ``_eval_model`` and the latency probe. Keeping the
    protocol in one place is what makes an Exp-5 row and an Exp-8 row
    directly comparable.
    """
    block = curve.finalize(train_losses, train_accs)
    curve.log_verdict(block)
    block['error_analysis'] = None
    if int(error_batches or 0) > 0:
        # Bad cases come from the dedicated error stream
        # (seed + ERROR_ANALYSIS_SEED_OFFSET), never from the stream the
        # reported accuracy was measured on, so the printed examples are
        # evidence about generalization rather than a re-read of the
        # headline number.
        block['error_analysis'] = analyze_errors(
            model, head, embed, op_name=op_name, task='mqar',
            seq_len=seq_len, n_kv=n_kv, vocab=VOCAB, device=device,
            n_batches=int(error_batches), batch=int(error_batch), seed=seed,
            max_examples=int(max_error_examples))
    block['error_analysis_seed_offset'] = ERROR_ANALYSIS_SEED_OFFSET
    block['val_monitor_seed_offset'] = VAL_MONITOR_SEED_OFFSET
    return block


def eval_layout(ratio, d_model=32, seq_len=SEQ_LEN, n_kv=1, steps=100, lr=3e-3, seed=42,
                device='cpu', eval_batches=4, eval_batch=64, train_batch=None,
                val_every=0, val_batches=2, val_batch=32,
                error_batches=0, error_batch=64, max_error_examples=2):
    if isinstance(device, str):
        device = torch.device(device)
    # Seed Python/NumPy/torch/CUDA + deterministic cuDNN/algorithm selection
    # so the layout comparison is reproducible on the same machine.
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio)
    total = sum(ratio)
    model = HybridKCHAttention(cfg, total_layers=total).to(device)
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    if train_batch is None:
        train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    model.train()
    head.train()
    embed.train()

    # Held-out monitoring curve. READ-ONLY with respect to the optimization:
    # it changes no loss, no hyper-parameter and no step count, so the
    # headline ``final_acc`` below keeps its exact meaning and every
    # existing statistic/schema check is unaffected. What it buys is the
    # ability to say WHY a layout scored what it scored — overfitting,
    # underfitting, saturated or healthy — instead of only quoting the
    # endpoint. The curve is drawn from ``seed + VAL_MONITOR_SEED_OFFSET``,
    # a stream disjoint from training (1_000_000) and from the headline
    # eval (5_000_000).
    def _val_eval(_step):
        acc, loss = _eval_model(
            model, head, embed, seq_len, n_kv, device,
            n_batches=int(val_batches or 2), batch=int(val_batch or 32),
            seed=seed, seed_offset=VAL_MONITOR_SEED_OFFSET)
        # ``_eval_model`` returns (accuracy, loss); the monitor's order is
        # named, so this is checked rather than assumed.
        return ValPoint(loss=loss, acc=acc)

    curve = ValidationCurve(
        _val_eval, steps=steps,
        val_every=int(val_every or 0) or max(1, steps // 4),
        label=f'layout {ratio[0]}:{ratio[1]}:{ratio[2]} n_kv={n_kv} '
              f'seed={seed}', ceiling=1.0)
    curve.record(0)

    losses = []
    train_accs = []
    grad_norms = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            # Staged triage instead of a bare message: name the FIRST stage
            # that can explain the NaN (data -> log(0)/div-by-zero ->
            # softmax overflow -> AMP fp16 -> gradient explosion / LR too
            # large), so a diverged seed names its cause instead of costing
            # a blind debugging cycle. Same abort semantics as before: this
            # seed is dropped, never silently averaged in.
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                token_ids=None, params=params, logits=logits,
                context=f'step={step}, ratio={ratio}, seed={seed}')
        opt.zero_grad()
        loss.backward()
        bad_grads = triage_gradient(params)
        clip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad_grads or not torch.isfinite(clip_norm):
            # A finite per-parameter norm can still overflow the global
            # norm; report it through the same staged triage instead of
            # letting the optimizer step on a corrupted direction.
            raise_non_finite(
                'gradient', loss=float(loss.detach()),
                params=params,
                grads={f'p{i}': p.grad for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'step={step}, ratio={ratio}, seed={seed}')
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
    # Pin intra-op threads to 1 during the latency probe so the reported
    # ``fwd_ms`` is comparable across layouts and to run_benchmark (Exp2),
    # which measures with ``torch.set_num_threads(1)``. Dynamic intra-op
    # thread resizing on CPU causes large run-to-run jitter that would
    # otherwise dominate (or hide) the layout-vs-layout latency gap.
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
    mean_last10 = sum(losses[-10:]) / min(10, len(losses)) if losses else 0.0
    curve_block = _attach_curve_diagnostics(
        curve=curve, model=model, head=head, embed=embed,
        train_losses=losses, train_accs=train_accs, op_name='layout',
        seq_len=seq_len, n_kv=n_kv, device=device, steps=steps,
        error_batches=error_batches, error_batch=error_batch,
        max_error_examples=max_error_examples, seed=seed,
        label=f'layout {ratio[0]}:{ratio[1]}:{ratio[2]} n_kv={n_kv}')
    return {
        'ratio': f'{ratio[0]}:{ratio[1]}:{ratio[2]}',
        'n_kv': n_kv,
        'layout': layout,
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
        'gradient_norms': summarize_gradient_norms(grad_norms, clip_norm=1.0),
        # Additive results-analysis block: the held-out curve, its
        # over/under/saturation verdict, the generalization gap, the
        # best-validation point and the bad-case report. Additive on
        # purpose — ``final_acc`` above still means "accuracy at the fixed
        # step budget", so the aggregate statistics, the schema checks and
        # the figures keep their exact meaning.
        **curve_block,
    }


def eval_layout_multi_seed(ratio, n_seeds=5, steps=100, device='cpu', **kw):
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    stub_train_batch = kw.get('train_batch')
    if stub_train_batch is None:
        stub_train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                         logger=logger)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        try:
            r = eval_layout(ratio, seed=s, steps=steps, device=device, **kw)
            r['train_time_s'] = time.time() - t0
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"fwd={r['fwd_ms']:.2f}ms  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            ratio_str = f'{ratio[0]}:{ratio[1]}:{ratio[2]}' if isinstance(ratio, tuple) else str(ratio)
            per_seed.append({
                'ratio': ratio_str,
                'n_kv': kw.get('n_kv', 1),
                'layout': None,
                'final_acc': None,
                'final_loss': None,
                'fwd_ms': None,
                'n_params': None,
                'n_layers': None,
                'seed': s,
                'steps': steps,
                'train_batch': stub_train_batch,
                'last_train_loss': None,
                'mean_last10_loss': None,
                'loss_curve': [],
                'acc_curve': [],
                'gradient_norms': None,
                # Same keys as a live row, so a consumer can read every
                # per_seed entry with one code path (mirrors run_quality's
                # failed-seed stub).
                'val_every': (int(kw.get('val_every', 0) or 0)
                              or max(1, steps // 4)),
                'val_history': [],
                'curve_diagnosis': None,
                'generalization_gap': None,
                'best_val': None,
                'val_loss_first': None,
                'val_loss_last': None,
                'val_acc_first': None,
                'val_acc_last': None,
                'val_acc_peak': None,
                'error_analysis': None,
                'error_analysis_seed_offset': ERROR_ANALYSIS_SEED_OFFSET,
                'val_monitor_seed_offset': VAL_MONITOR_SEED_OFFSET,
                'train_time_s': time.time() - t0,
                'error': str(e),
            })
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        raise RuntimeError(f"all {n_seeds} seeds failed for ratio {ratio}; "
                           f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    fwds = [r['fwd_ms'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_fwd = sum(fwds) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        std_acc = math.sqrt(var_acc)
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
    else:
        std_acc = None
        ci_acc = None

    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    # Row-level rollup of the per-seed bad-case reports (None when the pass
    # is disabled). Counts pool over every analyzed sample; examples come
    # from the worst seed, so the row answers "which mistake dominates this
    # ratio" without walking per_seed.
    error_aggregate = pool_error_analyses(ok_per_seed)
    log_pooled_errors(error_aggregate,
                      label=f'ratio {ratio} n_kv={ok_per_seed[0]["n_kv"]}')

    return {
        'ratio': ok_per_seed[0]['ratio'],
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
        't_stat_vs_chance': t_stat,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        # Sweep-level view of the held-out curves: the modal verdict across
        # the surviving seeds, plus how often the seeds AGREE. Seeds that
        # disagree are the important signal — identical configurations
        # cannot really be underfitting on one seed and overfitting on
        # another, so a split verdict means the curve is too short/flat to
        # carry information at this budget, and the label should not be
        # quoted as if it were a property of the layout.
        **_aggregate_curve_diagnosis(ok_per_seed),
        'mean_train_time_s': sum(r.get('train_time_s', 0.0) for r in ok_per_seed) / n,
        # Pooled bad-case rollup for this ratio (None when disabled).
        'error_analysis': error_aggregate,
    }


# ---------------------------------------------------------------------------
# Eight-group experiment matrix (ABL_MODE=matrix).
# ---------------------------------------------------------------------------
# Per-5-layer layout units. 'memory' resolves via cfg.memory_type, so the
# SAME matrix runs with R = KDA or R = Mamba-2 (the framework's core
# question: do CSA/HCA help BOTH recurrent memory operators?).
MATRIX_UNIT = {
    'A': ('memory', 'memory', 'memory', 'memory', 'memory'),
    'B': ('memory', 'memory', 'memory', 'csa', 'memory'),
    'C': ('memory', 'memory', 'memory', 'memory', 'hca'),
    'D': ('memory', 'memory', 'memory', 'csa', 'hca'),
    # E is only used by the csa_csa preset (HCA vs. one-more-CSA).
    'E': ('memory', 'memory', 'memory', 'csa', 'csa'),
}

# Role of each layout group in the design (mirrors the module docstring;
# machine-readable so the result metadata carries the design intent).
MATRIX_GROUP_ROLES = {
    'A': 'pure recurrent baseline (all memory slots)',
    'B': 'CSA replaces one fixed recurrent slot',
    'C': 'HCA replaces one fixed recurrent slot',
    'D': 'full hybrid (CSA + HCA replace two recurrent slots)',
    'E': 'two CSA layers (csa_csa preset only: HCA vs another CSA)',
}

# Pre-registered hypotheses for the SIMPLEST baselines (baseline-ladder
# requirement: a complex mixer must beat the simplest thing that could
# work, trained with the same recipe/budget, before its mechanism can be
# credited). Disable with ABL_SIMPLE_BASELINES=0.
SIMPLE_BASELINE_HYPOTHESES = {
    'linear': {
        'role': 'simplest rung: position-wise identity mixer (no token mixing)',
        'hypothesis': ('MQAR requires associating a cued key with an earlier '
                       'value, so a model that cannot move information '
                       'across time must stay at chance.'),
        'prediction': ('linear accuracy is not significantly above '
                       '1/VOCAB and does not approach the hybrid groups.'),
        'falsified_if': ('linear is significantly above chance or close to '
                         'the hybrids: the task is solvable from the current '
                         'token alone, so the operator ranking is confounded '
                         'and cannot be attributed to memory.'),
    },
    'softmax': {
        'role': 'standard rung: causal dense softmax attention',
        'hypothesis': ('Dense attention can solve short MQAR at this depth; '
                       'it is the standard architecture baseline, not a '
                       'strawman (same depth, residual/LN structure, data '
                       'stream, optimizer and budget).'),
        'prediction': ('softmax reaches high MQAR accuracy; a hybrid claim '
                       'must be read against it (and against its O(T) '
                       'per-token state/cache cost).'),
        'falsified_if': ('softmax fails at this depth/budget: the training '
                         'recipe (not the memory operator) is the limiting '
                         'factor, and hybrid-vs-softmax differences are not '
                         'interpretable as memory-mechanism evidence.'),
    },
}

# Standard-rung reference for the direct baseline test. The baseline is
# TUNED on the held-out validation stream (never the headline eval stream)
# so "the hybrid beats softmax" cannot be explained by an untuned or
# under-budgeted standard architecture.
TUNED_BASELINE_GROUP = 'BASELINE-softmax'
DEFAULT_BASELINE_TUNE_LRS = '1e-3,3e-3,1e-2'
TUNED_BASELINE_HYPOTHESIS = {
    'reference_group': TUNED_BASELINE_GROUP,
    'role': ('standard rung: causal dense softmax attention, tuned on the '
             'same validation stream and re-trained with the full seed '
             'count under the same depth/recipe/budget as the hybrids'),
    'hypothesis': ('A hybrid layout improves MQAR accuracy over a properly '
                   'tuned dense-softmax baseline: the memory mechanism, not '
                   'an untuned baseline or extra optimization, explains the '
                   'difference.'),
    'prediction': ('each non-baseline group, including the pure recurrent '
                   'control, has a positive seed-paired final_acc '
                   'difference versus the tuned BASELINE-softmax at '
                   'the same n_kv.'),
    'falsified_if': ('the corrected one-sided test does not clear the '
                     'threshold (hybrid advantage not established) or is '
                     'significantly negative (the tuned baseline wins); '
                     'raw mean differences alone are not evidence.'),
    'direction': 'greater',
}

# ---------------------------------------------------------------------------
# Pre-registered hypotheses for the matrix's paired contrasts
# (experiment-design requirement: fix the prediction and the falsification
# criterion BEFORE reading the numbers). The layout contrasts and the
# interaction are DIRECTIONAL (predicted positive: the added operator
# helps); the cross-route replacement contrasts ask a two-sided question
# (does the layout benefit transfer across memory operators?) and have no
# pre-registered direction. The statistical test is the seed-paired
# t-statistic from ``_matrix_contrast_report``; significance is assigned by
# ``_annotate_contrast_significance`` under one Bonferroni family over all
# contrast rows of a run.
# ---------------------------------------------------------------------------
MATRIX_CONTRAST_HYPOTHESES = {
    'B-A (CSA replaces memory)': {
        'hypothesis': ('Sparse compressed retrieval (CSA) replacing one '
                       'recurrent memory slot improves MQAR accuracy.'),
        'prediction': 'B > A on the seed-paired accuracy difference.',
        'falsified_if': ('the corrected one-sided test is significant in '
                         'the NEGATIVE direction (CSA hurts), or the '
                         'positive effect does not clear the corrected '
                         'threshold (prediction unsupported).'),
        'direction': 'greater',
    },
    'C-A (HCA replaces memory)': {
        'hypothesis': ('Heavily compressed global attention (HCA) replacing '
                       'one recurrent memory slot improves MQAR accuracy.'),
        'prediction': 'C > A on the seed-paired accuracy difference.',
        'falsified_if': ('the corrected one-sided test is significant in '
                         'the NEGATIVE direction (HCA hurts), or the '
                         'positive effect does not clear the corrected '
                         'threshold (prediction unsupported).'),
        'direction': 'greater',
    },
    'D-B (add HCA to CSA)': {
        'hypothesis': ('Adding HCA on top of an existing CSA layer improves '
                       'MQAR accuracy (the two sparse operators are '
                       'complementary at this budget).'),
        'prediction': 'D > B on the seed-paired accuracy difference.',
        'falsified_if': ('the corrected one-sided test is significant in '
                         'the NEGATIVE direction (HCA hurts on top of '
                         'CSA), or the positive effect does not clear the '
                         'corrected threshold (prediction unsupported).'),
        'direction': 'greater',
    },
    'D-C (add CSA to HCA)': {
        'hypothesis': ('Adding CSA on top of an existing HCA layer improves '
                       'MQAR accuracy (the two sparse operators are '
                       'complementary at this budget).'),
        'prediction': 'D > C on the seed-paired accuracy difference.',
        'falsified_if': ('the corrected one-sided test is significant in '
                         'the NEGATIVE direction (CSA hurts on top of '
                         'HCA), or the positive effect does not clear the '
                         'corrected threshold (prediction unsupported).'),
        'direction': 'greater',
    },
    'I=D-B-C+A (CSA/HCA interaction)': {
        'hypothesis': ('CSA and HCA are complementary: the benefit of the '
                       'two together exceeds the sum of their individual '
                       'replacements (super-additive interaction).'),
        'prediction': 'I = D - B - C + A > 0 on the seed-paired differences.',
        'falsified_if': ('the corrected one-sided test rejects the '
                         'sub-additive side (I significantly < 0), or the '
                         'positive estimate does not clear the corrected '
                         'threshold (complementarity unsupported).'),
        'direction': 'greater',
    },
}

# Fallback hypothesis for the dynamic cross-route contrast names
# (``{A|D}_{ROUTE_A} - {A|D}_{ROUTE_B}``): the memory-operator transfer
# question. No direction is pre-registered; the test is two-sided.
CROSS_ROUTE_CONTRAST_HYPOTHESIS = {
    'hypothesis': ('Swapping the recurrent memory operator (KDA vs Mamba-2 '
                   'vs RWKV-7) changes the layout\'s MQAR performance (the '
                   'transfer question: does a layout conclusion hold under '
                   'BOTH memory routes?).'),
    'prediction': ('no pre-registered direction: the two memory routes '
                   'differ on the paired accuracy difference.'),
    'falsified_if': ('the two-sided corrected test is not significant: no '
                     'evidence the operator swap changes MQAR at this '
                     'budget/geometry (the layout result then transfers '
                     'without a detected route-dependence).'),
    'direction': 'two_sided',
}


def _contrast_hypothesis(name: str) -> dict:
    """Return the pre-registered hypothesis for a contrast row name."""
    if name in MATRIX_CONTRAST_HYPOTHESES:
        return MATRIX_CONTRAST_HYPOTHESES[name]
    return CROSS_ROUTE_CONTRAST_HYPOTHESIS


# ---------------------------------------------------------------------------
# Pre-registered decision rule for the historical RATIO sweep (ABL_MODE=
# ratio). The sweep is an allocation EXPLORATION, not a directional
# experiment: the default 3:1:1 hybrid is the reference, and each
# alternative ratio is tested against it with a TWO-SIDED corrected
# paired test. An alternative may be called better/worse only when that
# test clears the family-wise threshold; otherwise it is statistically
# indistinguishable from the default at this seed budget and must not be
# ranked by raw mean accuracy. Fixed BEFORE the numbers are read.
# ---------------------------------------------------------------------------
RATIO_REFERENCE_LABEL = '3:1:1'
RATIO_SWEEP_HYPOTHESES = {
    'reference': {
        'hypothesis': ('The default KDA:CSA:HCA=3:1:1 allocation is the '
                       'accuracy reference at fixed depth; alternatives '
                       'must clear a corrected paired test before being '
                       'called different.'),
        'prediction': 'no directional prior (allocation exploration).',
        'falsified_if': ('an alternative ratio shows a significant '
                         'corrected paired difference (two-sided) from the '
                         'default: the default is then not the unique '
                         'accuracy point at this budget.'),
        'direction': 'two_sided',
    },
    'alternatives': {
        '4:1:0': {
            'question': ('does dropping HCA (pure memory + CSA) change MQAR '
                         'accuracy vs the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
        '4:0:1': {
            'question': ('does dropping CSA (pure memory + HCA) change MQAR '
                         'accuracy vs the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
        '2:2:1': {
            'question': ('does more CSA (CSA:memory 1:1) change accuracy vs '
                         'the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
        '2:1:2': {
            'question': ('does more HCA (HCA:memory 1:1) change accuracy vs '
                         'the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
        '1:3:1': {
            'question': ('does a CSA-heavy allocation change accuracy vs '
                         'the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
        '1:1:3': {
            'question': ('does an HCA-heavy allocation change accuracy vs '
                         'the default 3:1:1?'),
            'prediction': ('no directional prior: the estimand is the '
                           'two-sided seed-paired accuracy difference vs '
                           'the default 3:1:1 (allocation exploration).'),
            'falsified_if': ('not applicable: an exploratory two-sided '
                             'contrast has no directional prediction to '
                             'falsify; a significant difference is reported '
                             'with its sign, and non-significance is '
                             'inconclusive at this seed budget (not '
                             'evidence of no difference).'),
        },
    },
    'note': ('Pre-registered in run_ablation.RATIO_SWEEP_HYPOTHESES. Only '
             'rows with vs_reference_ratio.significant_bonferroni=True may '
             'be called different from the default; non-significant rows '
             'are statistically indistinguishable at this seed budget '
             '(not evidence of equivalence either).'),
}


def _matrix_mamba_dims(d_model: int) -> dict:
    """Env-tunable Mamba-2 geometry for the matrix, auto-clamped.

    Mirrors run_quality.Mamba2Attn's knobs (MAMBA2_D_STATE / HEADDIM /
    EXPAND / D_CONV) so the standalone operator probes and the matrix
    stacks test the same mixer geometry at a given d_model.
    """
    expand = parse_int_env('MAMBA2_EXPAND', 2, min_value=1, logger=logger)
    d_state = parse_int_env('MAMBA2_D_STATE', 16, min_value=1, logger=logger)
    headdim = parse_int_env('MAMBA2_HEADDIM', 16, min_value=1, logger=logger)
    d_conv = parse_int_env('MAMBA2_D_CONV', 4, min_value=2, logger=logger)
    d_inner = expand * d_model
    headdim_eff = min(headdim, d_inner)
    while d_inner % headdim_eff != 0:
        headdim_eff //= 2
    return {'mamba_d_state': d_state, 'mamba_d_conv': d_conv,
            'mamba_expand': expand, 'mamba_headdim': headdim_eff}


def _matrix_rwkv_dims(d_model: int) -> dict:
    """Env-tunable RWKV-7 geometry for the matrix, auto-clamped.

    Mirrors run_quality.Rwkv7Attn's knobs (RWKV_HEAD_DIM / lora widths)
    so the standalone operator probes and the matrix stacks test the
    same mixer geometry at a given d_model. ``RWKV_HEAD_DIM`` (default
    16 for the small matrix models, mirroring MAMBA2_HEADDIM) is clamped
    down to the largest divisor of d_model; the LoRA widths default to
    small-scale values (fla-formula floored at 4/8) and are clamped to
    d_model. The MAIN-table geometry (d_model=256, head_dim=64, loras
    32/32/64) is applied by the formal runs via the same env knobs.
    """
    head_dim = parse_int_env('RWKV_HEAD_DIM', 16, min_value=1, logger=logger)
    decay_lora = parse_int_env('RWKV_DECAY_LORA', 4, min_value=1,
                               logger=logger)
    a_lora = parse_int_env('RWKV_A_LORA', 4, min_value=1, logger=logger)
    gate_lora = parse_int_env('RWKV_GATE_LORA', 8, min_value=1,
                              logger=logger)
    head_dim_eff = min(head_dim, d_model)
    while d_model % head_dim_eff != 0:
        head_dim_eff //= 2
    return {
        'rwkv_head_dim': head_dim_eff,
        'rwkv_decay_lora': min(decay_lora, d_model),
        'rwkv_a_lora': min(a_lora, d_model),
        'rwkv_gate_lora': min(gate_lora, d_model),
    }


def build_experiment_matrix(preset='main8', total_layers=10,
                            memory_types=('kda', 'mamba2', 'rwkv7')):
    """Build the explicit experiment matrix.

    Returns a list of ``(group_id, memory_type, layout, description)``
    tuples where ``layout`` is the RESOLVED abstract layout (still carrying
    the 'memory' token — HybridConfig resolves it via memory_type).

    Presets:
      * ``main8``   — A/B/C/D x memories: the primary eight-group design
        at two memory routes; the DEFAULT ``memory_types`` now covers
        all three routes (kda / mamba2 / rwkv7) so a bare run yields the
        TWELVE-group total table when the Mamba-2 results share the
        configuration. Restrict via ABL_MEMORY (e.g.
        ``ABL_MEMORY=kda,rwkv7`` for the eight-group KDA-vs-RWKV main
        experiment). Requires total_layers % 5 == 0 (the unit repeats).
      * ``csa_csa`` — A/D/E x memories: is HCA actually better than a
        second CSA layer at the same depth?
      * ``position``— alternating (D-unit repeated) vs. back-loaded
        (memory layers first, CSA+HCA concentrated at the end), same
        operator counts, same depth.
      * ``budget``  — the D layout at equal compressed-entry budgets:
        (csa_m, hca_m2) pairs sharing 1/m + 1/m2. The cfg overrides are
        returned via the description; callers apply them per group.
    """
    if not isinstance(total_layers, int) or total_layers < 1:
        raise ValueError(
            f"total_layers={total_layers!r} must be a positive int.")
    if preset in ('main8', 'csa_csa', 'budget'):
        if total_layers % 5 != 0:
            raise ValueError(
                f"ABL_MATRIX={preset!r} repeats a 5-layer unit, so "
                f"ABL_TOTAL_LAYERS={total_layers} must be a multiple of 5.")
        n_units = total_layers // 5
        if preset == 'budget':
            # Equal compressed-entry budgets: 1/csa_m + 1/hca_m2 equal
            # within each budget family. The cfg overrides ride along as
            # the group tuple's fifth element.
            budget_pairs = [
                (4, 8), (8, 4),            # 1/m + 1/m2 = 0.375
                (4, 4), (3, 6), (6, 3),    # 1/m + 1/m2 = 0.500
            ]
            groups = []
            for mem in memory_types:
                for (m, m2) in budget_pairs:
                    layout = MATRIX_UNIT['D'] * n_units
                    budget = 1.0 / m + 1.0 / m2
                    gid = f'{mem.upper()}-D-m{m}-m2_{m2}'
                    desc = (f'D layout, csa_m={m}, hca_m2={m2} '
                            f'(compressed entries/token={budget:.3f})')
                    groups.append((gid, mem, layout, desc,
                                   {'csa_m': m, 'hca_m2': m2}))
            return groups
        units = ['A', 'B', 'C', 'D'] if preset == 'main8' else ['A', 'D', 'E']
        groups = []
        for mem in memory_types:
            for u in units:
                layout = MATRIX_UNIT[u] * n_units
                desc = (f'{u}-unit x{n_units}: '
                        + ' '.join(k.upper() for k in MATRIX_UNIT[u]))
                groups.append((f'{mem.upper()}-{u}', mem, layout, desc))
        return groups
    if preset == 'position':
        if total_layers < 2:
            raise ValueError(
                f"ABL_MATRIX=position needs total_layers >= 2, got "
                f"{total_layers}.")
        if total_layers % 5 != 0:
            raise ValueError(
                f"ABL_MATRIX=position compares against the repeated D "
                f"unit, so ABL_TOTAL_LAYERS={total_layers} must be a "
                f"multiple of 5.")
        n_units = total_layers // 5
        # Same operator COUNTS as the alternating layout (3n memory + n CSA
        # + n HCA — the spec: fix depth and ratio, vary ONLY the position);
        # the back-loaded variant concentrates all CSA layers then all HCA
        # layers at the end of the stack instead of spreading one of each
        # per 5-layer unit.
        alternating = MATRIX_UNIT['D'] * n_units
        back_loaded = (('memory',) * (3 * n_units)
                       + ('csa',) * n_units + ('hca',) * n_units)
        groups = []
        for mem in memory_types:
            groups.append((f'{mem.upper()}-alternating', mem, alternating,
                           f'D-unit x{n_units} (CSA/HCA spread evenly)'))
            groups.append((f'{mem.upper()}-backloaded', mem, back_loaded,
                           f'{3 * n_units} memory layers, then {n_units} CSA '
                           f'and {n_units} HCA at the end (same counts, '
                           f'different positions)'))
        return groups
    raise ValueError(
        f"ABL_MATRIX={preset!r} must be one of "
        f"['main8', 'csa_csa', 'position', 'budget'].")


def _make_matrix_cfg(d_model, memory_type, layout, cfg_overrides=None):
    """HybridConfig for a matrix group (explicit layout + memory route)."""
    spec = SMALL_MODEL_SPEC
    kw = dict(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        memory_type=memory_type, layout=tuple(layout),
        csa_m=spec['csa_m'], csa_topk=spec['csa_topk'],
        csa_nh=spec['csa_nh'], csa_c=spec['csa_c'],
        csa_dc=spec['csa_dc'], csa_nIh=spec['csa_nIh'],
        csa_cI=spec['csa_cI'],
        csa_sliding_window=spec['csa_sliding_window'],
        hca_m2=spec['hca_m2'], hca_nh=spec['hca_nh'],
        hca_c=spec['hca_c'], hca_dc=spec['hca_dc'],
        hca_sliding_window=spec['hca_sliding_window'],
        kda_backend=validate_kda_backend(
            os.environ.get('KDA_BACKEND', 'reference')
        ),
        mamba_backend=validate_mamba_backend(
            os.environ.get('MAMBA_BACKEND', 'reference')
        ),
        rwkv_backend=validate_rwkv_backend(
            os.environ.get('RWKV_BACKEND', 'reference')
        ),
        **_matrix_mamba_dims(d_model),
        **_matrix_rwkv_dims(d_model),
    )
    if cfg_overrides:
        kw.update(cfg_overrides)
    return HybridConfig(**kw)


def eval_group(group_id, memory_type, layout, d_model=32, seq_len=SEQ_LEN,
               n_kv=1, steps=100, lr=3e-3, seed=42, device='cpu',
               eval_batches=4, eval_batch=64, train_batch=None,
               cfg_overrides=None, description='', val_every=0,
               val_batches=2, val_batch=32, error_batches=0,
               error_batch=64, max_error_examples=2):
    """Train + evaluate ONE matrix group (mirrors eval_layout's protocol).

    ``layout`` is the abstract layout (with 'memory' tokens); the returned
    row records the RESOLVED layout string, the memory route, and the
    actual parameter count (which differs between KDA and Mamba-2 at the
    same d_model — that residual is exactly what the budget-matching
    follow-up closes).

    The held-out curve, the NaN triage and the bad-case pass are the SAME
    protocol as :func:`eval_layout` (shared helpers, shared RNG stream
    offsets), so a matrix row and a ratio row are directly comparable.
    """
    if isinstance(device, str):
        device = torch.device(device)
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    total = len(layout)
    if memory_type in SIMPLE_BASELINE_KINDS:
        model = SimpleBaselineStack(memory_type, d_model, total).to(device)
    else:
        cfg = _make_matrix_cfg(d_model, memory_type, layout, cfg_overrides)
        model = HybridAttentionStack(cfg, total_layers=total).to(device)
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout_str = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    if train_batch is None:
        train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    model.train()
    head.train()
    embed.train()

    # Same held-out monitor as eval_layout; see _attach_curve_diagnostics
    # for why this is additive (final_acc keeps its exact meaning).
    def _val_eval(_step):
        acc, loss = _eval_model(
            model, head, embed, seq_len, n_kv, device,
            n_batches=int(val_batches or 2), batch=int(val_batch or 32),
            seed=seed, seed_offset=VAL_MONITOR_SEED_OFFSET)
        return ValPoint(loss=loss, acc=acc)

    curve = ValidationCurve(
        _val_eval, steps=steps,
        val_every=int(val_every or 0) or max(1, steps // 4),
        label=f'group {group_id} n_kv={n_kv} seed={seed}', ceiling=1.0)
    curve.record(0)

    losses = []
    train_accs = []
    grad_norms = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                params=params, logits=logits,
                context=f'step={step}, group={group_id}, seed={seed}')
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
                context=f'step={step}, group={group_id}, seed={seed}')
        grad_norms.append(float(clip_norm))
        opt.step()
        losses.append(loss.item())
        train_accs.append(float((logits.argmax(-1) == target).float().mean()))
        if curve.enabled and (step + 1) % curve.val_every == 0:
            curve.record(step + 1)

    final_acc, final_loss, eval_extra = _eval_model(
        model, head, embed, seq_len, n_kv, device,
        n_batches=eval_batches, batch=eval_batch, seed=seed,
        collect_predictions=True,
    )
    # Accuracy alone hides per-class failure on an imbalanced recall set.
    # Report the full confusion-derived panel on the SAME held-out samples
    # (macro/micro/weighted F1 + balanced accuracy + PR-/ROC-AUC), so every
    # matrix row (including the simple baselines) carries it.
    class_metrics = multiclass_classification_metrics(
        eval_extra['pred'], eval_extra['target'],
        num_classes=VOCAB, scores=eval_extra['scores'])

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
    mean_last10 = sum(losses[-10:]) / min(10, len(losses)) if losses else 0.0
    curve_block = _attach_curve_diagnostics(
        curve=curve, model=model, head=head, embed=embed,
        train_losses=losses, train_accs=train_accs, op_name='group',
        seq_len=seq_len, n_kv=n_kv, device=device, steps=steps,
        error_batches=error_batches, error_batch=error_batch,
        max_error_examples=max_error_examples, seed=seed,
        label=f'group {group_id} n_kv={n_kv}')
    return {
        'group': group_id,
        'memory': memory_type,
        'description': description,
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
        'lr': float(lr),
        'last_train_loss': last_loss,
        'class_metrics': class_metrics,
        'f1_macro': class_metrics.get('f1_macro'),
        'f1_micro': class_metrics.get('f1_micro'),
        'f1_weighted': class_metrics.get('f1_weighted'),
        'precision_macro': class_metrics.get('precision_macro'),
        'recall_macro': class_metrics.get('recall_macro'),
        'balanced_accuracy': class_metrics.get('balanced_accuracy'),
        'class_balance': class_metrics.get('class_balance'),
        'pr_auc_macro_ovr': class_metrics.get('pr_auc_macro_ovr'),
        'mean_last10_loss': mean_last10,
        'loss_curve': losses,
        'acc_curve': train_accs,
        'gradient_norms': summarize_gradient_norms(grad_norms, clip_norm=1.0),
        # Additive results-analysis block (see eval_layout).
        **curve_block,
    }


def eval_group_multi_seed(group_spec, n_seeds=5, steps=100, device='cpu',
                          **kw):
    """Multi-seed wrapper for a matrix group (mirrors eval_layout_multi_seed).

    ``group_spec`` is one tuple from :func:`build_experiment_matrix` —
    either ``(group_id, memory, layout, description)`` or the budget
    preset's 5-tuple ``(..., cfg_overrides)``.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if len(group_spec) == 5:
        group_id, memory_type, layout, description, cfg_overrides = group_spec
    else:
        group_id, memory_type, layout, description = group_spec
        cfg_overrides = None
    seeds = [42 + i for i in range(n_seeds)]
    stub_train_batch = kw.get('train_batch')
    if stub_train_batch is None:
        stub_train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                         logger=logger)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        try:
            r = eval_group(group_id, memory_type, layout, seed=s, steps=steps,
                           device=device, cfg_overrides=cfg_overrides,
                           description=description, **kw)
            r['train_time_s'] = time.time() - t0
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  "
                        f"loss={r['final_loss']:.4f}  fwd={r['fwd_ms']:.2f}ms"
                        f"  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            per_seed.append({
                'group': group_id, 'memory': memory_type,
                'description': description,
                'n_kv': kw.get('n_kv', 1), 'layout': None,
                'final_acc': None, 'final_loss': None, 'fwd_ms': None,
                'n_params': None, 'n_layers': None,
                'seed': s, 'steps': steps,
                'lr': kw.get('lr', 3e-3),
                'train_batch': stub_train_batch,
                'last_train_loss': None, 'mean_last10_loss': None,
                'loss_curve': [],
                'acc_curve': [],
                'gradient_norms': None,
                'val_every': (int(kw.get('val_every', 0) or 0)
                              or max(1, steps // 4)),
                'val_history': [],
                'curve_diagnosis': None,
                'generalization_gap': None,
                'best_val': None,
                'val_loss_first': None,
                'val_loss_last': None,
                'val_acc_first': None,
                'val_acc_last': None,
                'val_acc_peak': None,
                'error_analysis': None,
                'error_analysis_seed_offset': ERROR_ANALYSIS_SEED_OFFSET,
                'val_monitor_seed_offset': VAL_MONITOR_SEED_OFFSET,
                'train_time_s': time.time() - t0,
                'error': str(e),
            })
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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
            f"all {n_seeds} seeds failed for group {group_id}; "
            f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    fwds = [r['fwd_ms'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_fwd = sum(fwds) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        std_acc = math.sqrt(var_acc)
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
    else:
        std_acc = None
        ci_acc = None

    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    # Seed-level aggregation of the confusion-derived classification panel
    # (the per-seed rows carry the full per-class report).
    class_agg = {}
    for key in ('f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro',
                'recall_macro', 'balanced_accuracy', 'pr_auc_macro_ovr',
                'roc_auc_macro_ovr'):
        vals = [float(r[key]) for r in ok_per_seed
                if r.get(key) is not None]
        if not vals:
            class_agg[key] = {'mean': None, 'std': None, 'ci95': None,
                              'n_seeds': 0}
            continue
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            crit = _t_crit_975(len(vals))
            ci = (crit * sd / math.sqrt(len(vals))
                  if crit is not None else None)
        else:
            # One surviving seed has no sample SD; 0.0 would claim perfect
            # precision. Mirror run_quality's undefined-std convention.
            sd, ci = None, None
        class_agg[key] = {'mean': m, 'std': sd, 'ci95': ci,
                          'n_seeds': len(vals)}

    # Pooled bad-case rollup for this group (None when the pass is
    # disabled); same definition as eval_layout_multi_seed.
    error_aggregate = pool_error_analyses(ok_per_seed)
    log_pooled_errors(error_aggregate,
                      label=f'group {group_id} n_kv={ok_per_seed[0]["n_kv"]}')

    return {
        'group': group_id,
        'memory': memory_type,
        'description': ok_per_seed[0].get('description', ''),
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
        't_stat_vs_chance': t_stat,
        # Accuracy is not the only read: the imbalanced-class panel is
        # aggregated over the same seeds.
        'class_metrics_mean': class_agg,
        'mean_f1_macro': class_agg['f1_macro']['mean'],
        'mean_f1_micro': class_agg['f1_micro']['mean'],
        'mean_f1_weighted': class_agg['f1_weighted']['mean'],
        'mean_precision_macro': class_agg['precision_macro']['mean'],
        'mean_recall_macro': class_agg['recall_macro']['mean'],
        'mean_balanced_accuracy': class_agg['balanced_accuracy']['mean'],
        'mean_pr_auc_macro_ovr': class_agg['pr_auc_macro_ovr']['mean'],
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        'lr': ok_per_seed[0].get('lr'),
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
        # Same sweep-level curve rollup as the ratio mode.
        **_aggregate_curve_diagnosis(ok_per_seed),
        # Pooled bad-case rollup for this group (None when disabled).
        'error_analysis': error_aggregate,
    }


def _validation_tail_score(result: dict, tail_k: int = 3):
    """Mean held-out validation accuracy over the last <= ``tail_k`` points.

    Mirrors ``run_quality.tune_baseline_lr``: average the plateau per seed,
    then average over seeds. Selection uses only the validation stream, so
    the headline eval stream stays untouched.
    """
    if not isinstance(result, dict):
        return None
    per_seed = result.get('per_seed')
    if not isinstance(per_seed, (list, tuple)):
        return None
    scores = []
    for row in per_seed:
        if not isinstance(row, dict):
            continue
        vh = row.get('val_history')
        if not isinstance(vh, (list, tuple)):
            continue
        vals = []
        for point in vh[-int(tail_k):]:
            if not isinstance(point, dict):
                continue
            acc = point.get('acc')
            if (isinstance(acc, bool) or not isinstance(acc, (int, float))
                    or not math.isfinite(float(acc))):
                continue
            vals.append(float(acc))
        if vals:
            scores.append(_safe_mean(vals))
    return _safe_mean(scores) if scores else None


def tune_simple_baseline_lr(spec, lrs, *, n_kv_list, tune_seeds, steps,
                            device, d_model, val_batches=2, val_batch=32):
    """Select a simple baseline's lr on the validation stream.

    Experiment-design requirement: the standard softmax baseline must be
    tuned as carefully as the proposed hybrid before any hybrid-vs-baseline
    difference is interpreted. Every candidate lr is trained over the same
    cells/seeds with the SAME recipe; the winner maximizes the mean
    validation-tail accuracy (plateau, not one noisy final point) and is
    then re-trained with the full seed count. Ties keep the earlier grid
    entry, which preserves the historical ``3e-3`` default ordering.
    """
    if not lrs:
        raise ValueError('tune_simple_baseline_lr requires a non-empty lr grid')
    group_id = spec[0]
    tune_seeds = max(1, int(tune_seeds))
    per_lr = {}
    best_lr = float(lrs[0])
    best_score = None
    for lr in lrs:
        val_scores = []
        final_accs = []
        failed_cells = []
        for n_kv in n_kv_list:
            try:
                r = eval_group_multi_seed(
                    spec, n_seeds=tune_seeds, steps=steps, device=device,
                    n_kv=n_kv, d_model=d_model, lr=float(lr),
                    val_batches=val_batches, val_batch=val_batch,
                    error_batches=0, error_batch=64,
                    max_error_examples=0)
            except Exception as e:
                failed_cells.append({'n_kv': n_kv, 'error': str(e)})
                logger.warning(
                    f'    [baseline-tune] {group_id} lr={lr:g} '
                    f'n_kv={n_kv} FAILED: {e}')
                continue
            score = _validation_tail_score(r)
            if score is not None:
                val_scores.append(score)
            if r.get('mean_acc') is not None:
                final_accs.append(float(r['mean_acc']))
        mean_score = ((sum(val_scores) / len(val_scores))
                      if val_scores else None)
        mean_final = ((sum(final_accs) / len(final_accs))
                      if final_accs else None)
        per_lr[f'{lr:g}'] = {
            'mean_val_tail_acc': mean_score,
            'mean_final_eval_acc': mean_final,
            'n_val_cells': len(val_scores),
            'n_cells_failed': len(failed_cells),
            'failed_cells': failed_cells,
        }
        logger.info(
            f'    [baseline-tune] {group_id:16s} lr={lr:g}  '
            f'val_tail_acc='
            f'{mean_score if mean_score is None else f"{mean_score:.4f}"} '
            f'(eval={mean_final if mean_final is None else f"{mean_final:.4f}"}, '
            f'{len(val_scores)}/{len(n_kv_list)} cells)')
        if mean_score is not None and (best_score is None
                                       or mean_score > best_score):
            best_score = mean_score
            best_lr = float(lr)
    if best_score is None:
        logger.warning(
            f'    [baseline-tune] {group_id}: no usable validation signal; '
            f'falling back to lr={best_lr:g}')
    return best_lr, {
        'group': group_id,
        'selected_lr': best_lr,
        'selected_lr_mean_val_acc': best_score,
        'selection_statistic': ('mean of the last min(3, n) validation '
                                'accuracies per seed, averaged over seeds '
                                'and n_kv cells'),
        'selection_stream': f'val_monitor_seed+{VAL_MONITOR_SEED_OFFSET}',
        'tune_seeds': tune_seeds,
        'tune_n_kv': list(n_kv_list),
        'lr_grid': per_lr,
        'note': ('The baseline lr is chosen on held-out validation '
                 'accuracy, never on the headline eval stream, then '
                 're-trained with the full seed count. Ties keep the '
                 'earlier grid entry.'),
    }


def _seed_accuracy_map(result: dict) -> dict[int, float]:
    """Map seed -> final_acc from a group's per-seed rows (missing = skip).

    A diverged seed can survive with a NaN/string ``final_acc`` (the
    aggregate path already excludes it). It must not enter a paired
    contrast either: one NaN poisons the difference, and a string raises
    inside ``math.isfinite``.
    """
    out = {}
    per_seed = result.get('per_seed', [])
    if not isinstance(per_seed, (list, tuple)):
        return out
    for row in per_seed:
        if not isinstance(row, dict) or 'error' in row:
            continue
        acc = row.get('final_acc')
        seed = row.get('seed')
        if (isinstance(acc, bool) or not isinstance(acc, (int, float))
                or not math.isfinite(float(acc))):
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        out[seed] = float(acc)
    return out


def _paired_contrast(accs_a: dict[int, float], accs_b: dict[int, float]):
    """Paired mean diff + t-stat over the COMMON seeds of two groups.

    Returns ``(mean_diff, t_stat, n, unanimous, sd_diff)``, or
    ``(None, None, 0, None, None)`` when fewer than two common seeds
    exist. ``sd_diff`` is the sample SD of the per-seed differences
    (``0.0`` when every difference is numerically identical — the
    zero-variance convention of ``multi_seed_stats``); it rides in the
    contrast rows so the TOST equivalence test can be re-run from the
    stored summary statistics without keeping the raw vectors.
    Positive diff = a better than b for a higher-is-better metric
    (final_acc); loss-direction metrics must be sign-flipped BEFORE
    calling this. ``unanimous`` is ``'positive'`` when every per-seed
    difference is strictly > 0, ``'negative'`` when every difference is
    strictly < 0, and ``'mixed'`` otherwise. It lets the caller keep
    zero-variance (t undefined) UNANIMOUS evidence, mirroring the
    convention in ``run_quality._paired_one_sided_t``.
    """
    common = sorted(set(accs_a) & set(accs_b))
    if len(common) < 2:
        return None, None, 0, None, None
    diffs = [accs_a[s] - accs_b[s] for s in common]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    # Same exact-constant-vector zero-variance convention as
    # multi_seed_stats._sample_sd: only a genuinely constant sweep has
    # numerically zero spread, so a real spread around a large common offset
    # is not mistaken for a flat sweep.
    std = _sample_sd(diffs, mean_diff)
    if std is None:
        std = 0.0
    t_stat = (mean_diff / (std / math.sqrt(n))) if std > 0 else None
    if all(d > 0 for d in diffs):
        unanimous = 'positive'
    elif all(d < 0 for d in diffs):
        unanimous = 'negative'
    else:
        unanimous = 'mixed'
    return mean_diff, t_stat, n, unanimous, std


def _tuned_baseline_contrast_report(
        all_results: list,
        reference_group: str = TUNED_BASELINE_GROUP) -> list:
    """Direct paired tests of every non-baseline group vs tuned softmax.

    ``_matrix_contrast_report`` answers the CSA/HCA layout questions but
    never asks the prior question: does *any* hybrid beat the standard
    dense-attention rung after that rung has been tuned? Each row here is
    a seed-paired ``group - BASELINE-softmax`` accuracy difference at the
    same ``n_kv``; the annotation step applies its own Bonferroni family
    (separate from the layout-contrast family) and the pre-registered
    ``TUNED_BASELINE_HYPOTHESIS`` decision rule.
    """
    ok_results = [r for r in all_results if 'error' not in r]
    by_key = {(r.get('n_kv'), r.get('group')): r for r in ok_results}
    references = {
        r.get('n_kv'): r for r in ok_results
        if r.get('group') == reference_group
    }
    rows = []
    for r in ok_results:
        gid = r.get('group')
        if gid == reference_group or r.get('memory') in SIMPLE_BASELINE_KINDS:
            continue
        ref = references.get(r.get('n_kv'))
        if ref is None:
            continue
        diff, t_stat, n, unanimous, sd = _paired_contrast(
            _seed_accuracy_map(r), _seed_accuracy_map(ref))
        if diff is None:
            continue
        rows.append({
            'n_kv': r.get('n_kv'),
            'memory': r.get('memory'),
            'contrast': f'{gid} - {reference_group} (tuned)',
            'candidate_group': gid,
            'reference_group': reference_group,
            'mean_diff': diff,
            't_stat': t_stat,
            'sd_diff': sd,
            'n_common_seeds': n,
            'unanimous': unanimous,
            'candidate_lr': r.get('lr'),
            'baseline_lr': ref.get('lr'),
            'predicted_direction':
                TUNED_BASELINE_HYPOTHESIS['direction'],
            'hypothesis': TUNED_BASELINE_HYPOTHESIS['hypothesis'],
            'prediction': TUNED_BASELINE_HYPOTHESIS['prediction'],
            'falsified_if': TUNED_BASELINE_HYPOTHESIS['falsified_if'],
            'metric': 'final_acc',
            'direction': 'higher_is_better',
            'question': ('does this group beat a softmax baseline tuned '
                         'on the same validation stream and budget?'),
        })
    return rows


def _matrix_contrast_report(all_results: list) -> list:
    """Paired CSA/HCA contrasts per memory route (the matrix's key reads).

    For each ``n_kv`` and each memory route R present with the A/B/C/D
    groups, compute paired seed-matched differences on ``final_acc``
    (higher-is-better; loss metrics must be sign-flipped before reuse):

      * ``B_R - A_R``  — CSA replacing a fixed memory slot's benefit.
      * ``C_R - A_R``  — HCA replacing a fixed memory slot's benefit.
      * ``D_R - B_R``  — adding HCA on top of an existing CSA.
      * ``D_R - C_R``  — adding CSA on top of an existing HCA.
      * ``I_R = D_R - B_R - C_R + A_R`` — the CSA/HCA interaction
        (complementarity) under this layout; interpret only with
        statistical evidence.

    and, for every PAIR of memory routes (R, R') sharing the A/D groups
    (e.g. the KDA-vs-RWKV-7 replacement question), the full-model
    replacement difference ``D_R - D_R'`` plus the pure-baseline
    replacement difference ``A_R - A_R'``.

    Only rows with >= 2 common seeds produce a t-stat. The returned
    list is written to ``results/exp5b_ablation_matrix_contrasts.json``
    and logged as a table by ``_run_matrix_mode``.
    """
    ok_results = [r for r in all_results if 'error' not in r]
    by_key = {}
    for r in ok_results:
        by_key[(r.get('n_kv'), r.get('memory'), r.get('group'))] = r

    contrasts = []
    n_kvs = sorted({r.get('n_kv') for r in ok_results})
    routes = sorted({r.get('memory') for r in ok_results})
    for n_kv in n_kvs:
        # --- per-route layout contrasts --------------------------------
        for route in routes:
            groups = {g: by_key.get((n_kv, route, f'{route.upper()}-{g}'))
                      for g in 'ABCD'}
            accs = {g: _seed_accuracy_map(r) if r is not None else {}
                    for g, r in groups.items()}
            for name, ga, gb in [
                ('B-A (CSA replaces memory)', 'B', 'A'),
                ('C-A (HCA replaces memory)', 'C', 'A'),
                ('D-B (add HCA to CSA)', 'D', 'B'),
                ('D-C (add CSA to HCA)', 'D', 'C'),
            ]:
                if groups[ga] is None or groups[gb] is None:
                    continue
                diff, t_stat, n, unanimous, sd = _paired_contrast(
                    accs[ga], accs[gb])
                if diff is None:
                    continue
                hyp = _contrast_hypothesis(name)
                contrasts.append({
                    'n_kv': n_kv, 'memory': route, 'contrast': name,
                    'mean_diff': diff, 't_stat': t_stat,
                    'sd_diff': sd,
                    'n_common_seeds': n,
                    'unanimous': unanimous,
                    'predicted_direction': hyp['direction'],
                    'hypothesis': hyp['hypothesis'],
                    'prediction': hyp['prediction'],
                    'falsified_if': hyp['falsified_if'],
                    'metric': 'final_acc',
                    'direction': 'higher_is_better',
                    # Row identity of the two sides, so the accuracy-vs-cost
                    # screen (and any consumer) can join the contrast back to
                    # the group rows it was computed from without parsing the
                    # display strings.
                    'candidate_memory': route,
                    'candidate_group': f'{route.upper()}-{ga}',
                    'reference_memory': route,
                    'reference_group': f'{route.upper()}-{gb}',
                })
            # Interaction I_R = D - B - C + A on the COMMON seeds of all
            # four groups.
            if all(groups[g] is not None for g in 'ABCD'):
                common = (set(accs['A']) & set(accs['B'])
                          & set(accs['C']) & set(accs['D']))
                if len(common) >= 2:
                    inter = [accs['D'][s] - accs['B'][s] - accs['C'][s]
                             + accs['A'][s] for s in sorted(common)]
                    n = len(inter)
                    mean_i = sum(inter) / n
                    sd_i = _sample_sd(inter, mean_i)
                    if sd_i is None:
                        sd_i = 0.0
                    t_i = (mean_i / (sd_i / math.sqrt(n))) \
                        if sd_i > 0 else None
                    if all(x > 0 for x in inter):
                        unanimous_i = 'positive'
                    elif all(x < 0 for x in inter):
                        unanimous_i = 'negative'
                    else:
                        unanimous_i = 'mixed'
                    hyp = _contrast_hypothesis(
                        'I=D-B-C+A (CSA/HCA interaction)')
                    contrasts.append({
                        'n_kv': n_kv, 'memory': route,
                        'contrast': 'I=D-B-C+A (CSA/HCA interaction)',
                        'mean_diff': mean_i, 't_stat': t_i,
                        'sd_diff': sd_i,
                        'n_common_seeds': n,
                        'unanimous': unanimous_i,
                        'predicted_direction': hyp['direction'],
                        'hypothesis': hyp['hypothesis'],
                        'prediction': hyp['prediction'],
                        'falsified_if': hyp['falsified_if'],
                        'metric': 'final_acc',
                        'direction': 'higher_is_better',
                        # A four-cell functional, not a two-row difference:
                        # no single candidate/reference pair to cost-screen.
                        'candidate_memory': None,
                        'candidate_group': None,
                        'reference_memory': None,
                        'reference_group': None,
                    })
        # --- cross-route replacement contrasts ---------------------------
        for i, ra in enumerate(routes):
            for rb in routes[i + 1:]:
                for g, question in (('A', 'pure memory-baseline replacement'),
                                    ('D', 'full hybrid model replacement')):
                    res_a = by_key.get((n_kv, ra, f'{ra.upper()}-{g}'))
                    res_b = by_key.get((n_kv, rb, f'{rb.upper()}-{g}'))
                    if res_a is None or res_b is None:
                        continue
                    diff, t_stat, n, unanimous, sd = _paired_contrast(
                        _seed_accuracy_map(res_a), _seed_accuracy_map(res_b))
                    if diff is None:
                        continue
                    contrasts.append({
                        'n_kv': n_kv,
                        'memory': f'{ra} vs {rb}',
                        'contrast': f'{g}_{ra.upper()} - {g}_{rb.upper()}',
                        'mean_diff': diff, 't_stat': t_stat,
                        'sd_diff': sd,
                        'n_common_seeds': n,
                        'unanimous': unanimous,
                        'predicted_direction':
                            CROSS_ROUTE_CONTRAST_HYPOTHESIS['direction'],
                        'hypothesis':
                            CROSS_ROUTE_CONTRAST_HYPOTHESIS['hypothesis'],
                        'prediction':
                            CROSS_ROUTE_CONTRAST_HYPOTHESIS['prediction'],
                        'falsified_if':
                            CROSS_ROUTE_CONTRAST_HYPOTHESIS['falsified_if'],
                        'metric': 'final_acc',
                        'direction': 'higher_is_better',
                        'question': question,
                        # Row identity: the two sides live on DIFFERENT
                        # memory routes (the replacement question).
                        'candidate_memory': ra,
                        'candidate_group': f'{ra.upper()}-{g}',
                        'reference_memory': rb,
                        'reference_group': f'{rb.upper()}-{g}',
                    })
    return contrasts


_FACTORIAL_UNITS = {
    'main8': ('A', 'B', 'C', 'D'),
    'csa_csa': ('A', 'D', 'E'),
}


def _unit_of_group(group_id: str) -> str:
    return str(group_id).rsplit('-', 1)[-1]


def _row_design_field(row: dict, field: str):
    """Read a recipe field from the aggregate row or its per-seed rows.

    Malformed per-seed entries are skipped (a result file can contain an
    error stub or a hand-edited placeholder); the aggregate field, when
    present, always wins.
    """
    if not isinstance(row, dict):
        return None
    value = row.get(field)
    if value is None:
        per_seed = row.get('per_seed')
        if isinstance(per_seed, (list, tuple)):
            for seed_row in per_seed:
                if not isinstance(seed_row, dict):
                    continue
                value = seed_row.get(field)
                if value is not None:
                    break
    return value


def _finite_seed_set(row: dict) -> set:
    """Seed ids with a finite numeric ``final_acc`` in one result row.

    This is the seed-alignment audit's population. It must use exactly the
    same survivorship rule as the paired contrasts (see
    :func:`_seed_accuracy_map`): a malformed per-seed record, a non-int
    seed id, or a non-finite/string/bool accuracy is not a measurement and
    must not be treated as a successfully aligned seed.
    """
    seeds = set()
    if not isinstance(row, dict):
        return seeds
    per_seed = row.get('per_seed')
    if not isinstance(per_seed, (list, tuple)):
        return seeds
    for seed_row in per_seed:
        if not isinstance(seed_row, dict) or 'error' in seed_row:
            continue
        seed = seed_row.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            continue
        acc = seed_row.get('final_acc')
        if (isinstance(acc, bool) or not isinstance(acc, (int, float))
                or not math.isfinite(float(acc))):
            continue
        seeds.add(seed)
    return seeds


def _matrix_design_audit(all_results: list, groups: list, *, preset: str,
                         n_kv_list, n_seeds: int,
                         requested_seeds=None) -> dict:
    """Machine-readable audit of the matrix as a controlled experiment.

    Verifies three things that the contrast arithmetic cannot see:

    1. every preset group is present for every n_kv (no survivor-only
       matrix, no silently missing factorial cell);
    2. every non-baseline group was trained with the same recipe fields
       (steps, batch, lr, depth, width), apart from the intentionally
       tuned simple-baseline rungs; and
    3. every contrasted cell shares the same requested seed set, so the
       paired differences are genuinely paired.
    """
    if requested_seeds is None:
        requested_seeds = [42 + i for i in range(int(n_seeds))]
    requested = set(requested_seeds)
    recipe_fields = ('steps', 'train_batch', 'lr', 'n_layers', 'd_model')

    expected_groups = [spec[0] for spec in groups]
    expected_by_route = {}
    for spec in groups:
        expected_by_route.setdefault(spec[1], set()).add(spec[0])
    baseline_routes = set(SIMPLE_BASELINE_KINDS)

    ok_rows = [r for r in all_results
               if 'error' not in r and r.get('mean_acc') is not None]
    by_key = {(r.get('n_kv'), r.get('memory'), r.get('group')): r
              for r in ok_rows}

    cells = []
    missing_groups = []
    factorial_incomplete = []
    seed_mismatch_cells = []
    for n_kv in n_kv_list:
        for memory in sorted(expected_by_route):
            expected = expected_by_route[memory]
            present = {gid for gid in expected
                       if (n_kv, memory, gid) in by_key}
            missing = sorted(expected - present)
            for gid in missing:
                if gid not in missing_groups:
                    missing_groups.append(gid)
            # Simple baselines are appended to the matrix for the tuned
            # reference comparison but are not factorial units; still,
            # their promised cells must be present (and counted in
            # n_expected_cells), so audit presence without demanding a
            # preset unit.
            is_baseline = memory in baseline_routes
            units_present = {_unit_of_group(g) for g in present}
            required_units = (None if is_baseline
                              else _FACTORIAL_UNITS.get(preset))
            units_complete = (is_baseline or required_units is None
                              or units_present == set(required_units))
            if not units_complete:
                factorial_incomplete.append(
                    {'n_kv': n_kv, 'memory': memory,
                     'missing_units': sorted(
                         set(required_units) - units_present)})
            seed_sets = {gid: _finite_seed_set(by_key[(n_kv, memory, gid)])
                         for gid in present}
            common = (set.intersection(*seed_sets.values())
                      if seed_sets else set())
            seed_aligned = bool(seed_sets) and all(
                seeds == requested for seeds in seed_sets.values())
            if not seed_aligned:
                seed_mismatch_cells.append(
                    {'n_kv': n_kv, 'memory': memory,
                     'requested_seeds': sorted(requested),
                     'per_group_seeds': {
                         gid: sorted(s for s in seeds if s is not None)
                         for gid, seeds in sorted(seed_sets.items())}})
            cells.append({
                'n_kv': n_kv,
                'memory': memory,
                'present_groups': sorted(present),
                'missing_groups': missing,
                'factorial_complete': units_complete,
                'required_units': (list(required_units)
                                   if required_units else None),
                'n_common_seeds': len(common),
                'common_seeds': sorted(s for s in common if s is not None),
                'seed_alignment_ok': seed_aligned,
            })

    hybrid_rows = [r for r in ok_rows
                   if r.get('memory') not in SIMPLE_BASELINE_KINDS]
    baseline_rows = [r for r in ok_rows
                     if r.get('memory') in SIMPLE_BASELINE_KINDS]

    def values_for(rows, field):
        vals = []
        for row in rows:
            value = _row_design_field(row, field)
            if value is None:
                continue
            if field in ('steps', 'train_batch', 'n_layers', 'd_model'):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            elif field == 'lr':
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
            vals.append(value)
        return sorted(set(vals), key=repr)

    hybrid_values = {f: values_for(hybrid_rows, f) for f in recipe_fields}
    hybrid_parity = {f: len(hybrid_values[f]) <= 1 for f in recipe_fields}
    baseline_values = {f: values_for(baseline_rows, f) for f in recipe_fields}
    baseline_parity = {f: len(baseline_values[f]) <= 1
                       for f in recipe_fields}

    violations = []
    if missing_groups:
        violations.append(f'missing_groups:{missing_groups}')
    if factorial_incomplete:
        violations.append(
            f'factorial_incomplete:{factorial_incomplete}')
    mismatched = [f for f in ('steps', 'train_batch', 'lr',
                              'n_layers', 'd_model')
                  if not hybrid_parity[f]]
    if mismatched:
        violations.append(f'hybrid_recipe_mismatch:{mismatched}')
    if seed_mismatch_cells:
        violations.append(
            f'seed_mismatch:{len(seed_mismatch_cells)}_cells')

    if not ok_rows:
        status = 'not_computable'
    elif missing_groups:
        status = 'incomplete'
    elif factorial_incomplete:
        status = 'incomplete_factorial'
    elif mismatched:
        status = 'recipe_mismatch'
    elif seed_mismatch_cells:
        status = 'seed_mismatch'
    else:
        status = ('complete_factorial' if preset == 'main8'
                  else 'complete_preset')

    return {
        'status': status,
        'preset': preset,
        'required_units': (list(_FACTORIAL_UNITS[preset])
                           if preset in _FACTORIAL_UNITS else None),
        'expected_groups': list(expected_groups),
        'n_expected_cells': len(expected_groups) * len(n_kv_list),
        'n_present_cells': sum(len(c['present_groups']) for c in cells),
        'cells': cells,
        'missing_groups': missing_groups,
        'factorial_incomplete_cells': factorial_incomplete,
        'controls': {
            'recipe_fields': list(recipe_fields),
            'hybrid_values': hybrid_values,
            'hybrid_parity': hybrid_parity,
            'baseline_values': baseline_values,
            'baseline_parity': baseline_parity,
            'requested_seeds': sorted(requested),
            'seed_alignment_ok': not seed_mismatch_cells,
            'seed_mismatch_cells': seed_mismatch_cells,
            'note': ('Baseline lrs are tuned independently on a held-out '
                     'validation tail; every non-baseline group must share '
                     'the same steps/batch/lr/depth/width. A non-complete '
                     'status means the design claims are not verified.'),
        },
        'violations': violations,
    }


def _annotate_contrast_significance(contrasts: list,
                                    alpha: float = 0.05,
                                    hypotheses: dict | None = None,
                                    equivalence_margin: float | None = None
                                    ) -> dict:
    """Attach Bonferroni-corrected significance to the paired contrasts.

    Raw t-statistics invite post-hoc storytelling: with ~10-30 contrasts
    per run, some |t| > 2. We therefore declare ONE family — every
    contrast row of the run across all memory routes / layouts and n_kv
    cells — and assign each row a corrected threshold. Directional
    contrasts (predicted ``> 0``) use the one-sided upper-tail quantile
    at ``alpha / n`` (``_bonferroni_crit_q`` is one-sided); two-sided
    contrasts (no pre-registered direction) use the upper-tail quantile
    at ``alpha / (2n)``. Zero-variance differences: a unanimous
    all-positive sweep of a DIRECTIONAL contrast counts as significant
    (with ``t_stat=None`` recorded) only if the exact one-sided sign-flip
    p-value ``2^{-(n-1)}`` clears the corrected alpha; any other
    undefined row can never reject. ``hypotheses`` overrides the
    hypothesis block recorded in the summary (matrix vs ratio sweep).

    ``equivalence_margin`` (absolute final_acc units) additionally runs
    the TOST equivalence test on every row over the SAME family, using
    the row's stored ``sd_diff`` (``multi_seed_stats.tost_from_summary``)
    and the Bonferroni-corrected per-test alpha. This is what turns a
    non-significant contrast into a *decidable* statement: without TOST,
    "not significant" means only "inconclusive at this seed budget" — it
    is NEVER evidence that the compared groups are equivalent; with a
    pre-specified margin in hand, ``tost_equivalent=True`` IS positive
    evidence of practical equivalence (and ``not_equivalent`` remains
    inconclusive, not proof of a difference). The margin must be fixed
    before the numbers are read and is NOT derived from the observed CI.

    Mutates the contrast rows (additive keys) and returns a JSON-safe
    summary for the result metadata:

      * ``test``                  'paired_one_sided_t' | 'paired_two_sided_t'
      * ``alpha_bonferroni``      per-test alpha (family-wise / n_family)
      * ``t_crit_bonferroni``     corrected critical value (None if undefined)
      * ``significant_bonferroni``bool
      * ``hypothesis_status``     directional: 'supported' | 'contradicted'
                                  | 'not_supported'; two-sided:
                                  'difference_supported' | 'not_supported'
      * ``equivalence_margin``    the pre-specified TOST margin (None if off)
      * ``tost_p_value`` etc.     per-row equivalence verdict (margin on)
      * ``family_size`` etc. on the returned summary
    """
    if equivalence_margin is not None:
        equivalence_margin = float(equivalence_margin)
        if not math.isfinite(equivalence_margin) \
                or equivalence_margin <= 0.0:
            raise ValueError(
                'equivalence_margin must be finite and > 0, got '
                f'{equivalence_margin!r}')
    n_family = len(contrasts)
    summary = {
        'alpha_family_wise': alpha,
        'n_contrasts': n_family,
        'family_scope': ('all contrast rows of this run (every memory '
                         'route x n_kv x contrast)'),
        'correction': 'bonferroni',
        'test': ('seed-paired t-test on per-seed final_acc differences; '
                 'one-sided for the pre-registered directional layout '
                 'contrasts, two-sided for the cross-route replacement '
                 'contrasts'),
        'null_hypothesis': ('H0: the mean seed-paired accuracy difference '
                            'is zero (for directional contrasts, against '
                            'the pre-registered positive direction)'),
        'zero_variance_rule': ('a UNANIMOUS all-positive sweep of a '
                               'directional contrast (t_stat=None) counts '
                               'as significant only if the exact one-sided '
                               'sign-flip p-value 2^{-(n-1)} clears the '
                               'corrected alpha; all other zero-variance '
                               'rows can never reject'),
        'n_significant': 0,
        'n_supported': 0,
        'n_contradicted': 0,
        'n_significant_but_below_min_effect': 0,
        'effect_size': {
            'measure': ("Cohen's d_z = mean seed-paired difference / SD of "
                        'the paired differences'),
            'power_target': 0.8,
            'reference_min_effect': DEFAULT_MIN_EFFECT,
            'mde_note': (
                'min_detectable_effect is the smallest mean difference this '
                'run could detect at 80% power using the observed paired SD '
                'and the corrected per-test alpha. It is a post-hoc planning '
                'quantity, not a second test; a significant effect smaller '
                'than the reference_min_effect is statistically detectable '
                'but may not be scientifically meaningful.'),
        },
        'equivalence_margin': equivalence_margin,
        'n_equivalent': 0,
        'tost': None,
        'verdict_scope': (
            'Only significant_bonferroni=True rows may be cited as '
            'evidence for (or against) the pre-registered prediction; '
            'non-significant rows are inconclusive at this seed budget, '
            'not evidence of no effect.'),
    }
    for c in contrasts:
        direction = c.get('predicted_direction', 'two_sided')
        n = c.get('n_common_seeds', 0)
        t_stat = c.get('t_stat')
        mean_diff = c.get('mean_diff')
        if direction == 'greater':
            c['test'] = 'paired_one_sided_t'
            alpha_test = alpha / n_family if n_family else None
            crit = (_bonferroni_crit_q(n, alpha=alpha_test)
                    if (alpha_test is not None and n >= 2) else None)
            sig = bool(crit is not None and t_stat is not None
                       and t_stat > crit)
            if t_stat is None and c.get('unanimous') == 'positive' \
                    and n >= 2 and mean_diff is not None and mean_diff > 0:
                # Zero-variance unanimous improvement: the exact
                # one-sided sign-flip permutation p-value of the
                # all-positive pattern is 2^{-(n-1)}; it counts as
                # significant only when that clears the corrected alpha
                # (e.g. n >= 9 at a 10-contrast family with alpha=0.05).
                # Never claim a degenerate 2-seed sweep as evidence.
                exact_p = 2.0 ** (-(n - 1))
                sig = bool(alpha_test is not None and exact_p <= alpha_test)
        else:
            c['test'] = 'paired_two_sided_t'
            alpha_test = alpha / (2 * n_family) if n_family else None
            crit = (_bonferroni_crit_q(n, alpha=alpha_test)
                    if (alpha_test is not None and n >= 2) else None)
            sig = bool(crit is not None and t_stat is not None
                       and abs(t_stat) > crit)
        c['alpha_bonferroni'] = alpha_test
        c['t_crit_bonferroni'] = crit
        c['significant_bonferroni'] = sig

        # Standardized effect size + planning quantities. The t-statistic
        # alone hides whether a significant contrast is a large effect or
        # a tiny one bought with many seeds; d_z is on the paired-difference
        # SD scale, and the MDE/seed count show what this run could and
        # could not have resolved. Additive fields only.
        sd = c.get('sd_diff')
        dz = None
        if (isinstance(mean_diff, (int, float))
                and not isinstance(mean_diff, bool)
                and isinstance(sd, (int, float))
                and not isinstance(sd, bool)
                and math.isfinite(float(mean_diff))
                and math.isfinite(float(sd)) and float(sd) > 0.0):
            dz = float(mean_diff) / float(sd)
        mde = None
        required_n = None
        if (alpha_test is not None and n >= 2
                and isinstance(sd, (int, float))
                and not isinstance(sd, bool)
                and math.isfinite(float(sd)) and float(sd) >= 0.0):
            mde_alt = 'greater' if direction == 'greater' else 'two-sided'
            # ``alpha_test`` is the one-sided TAIL probability used by the
            # significance test above.  The planning helpers take a two-sided
            # TEST level and halve it internally, so pass the full test level
            # for a two-sided row; passing the tail alpha again would correct
            # twice and overstate both the MDE and the required seed count.
            planning_alpha = (alpha_test if direction == 'greater'
                              else 2.0 * alpha_test)
            mde = min_detectable_effect(
                n, float(sd), alpha=planning_alpha, power=0.8,
                alternative=mde_alt)
            if dz is not None:
                required_n = seeds_required(
                    dz, alpha=planning_alpha, power=0.8,
                    alternative=mde_alt)
        c['effect_size_cohens_dz'] = dz
        c['min_detectable_effect'] = mde
        c['min_detectable_effect_alpha'] = alpha_test
        c['min_detectable_effect_power'] = 0.8
        c['seeds_required_80pct_power'] = required_n
        below_min = (isinstance(mean_diff, (int, float))
                     and not isinstance(mean_diff, bool)
                     and math.isfinite(float(mean_diff))
                     and abs(float(mean_diff)) < DEFAULT_MIN_EFFECT)
        c['below_default_min_effect'] = below_min
        c['significant_but_below_default_min_effect'] = bool(
            sig and below_min)
        summary['n_significant_but_below_min_effect'] += int(
            sig and below_min)

        if direction == 'greater':
            if sig and mean_diff is not None and mean_diff > 0:
                c['hypothesis_status'] = 'supported'
            elif sig and mean_diff is not None and mean_diff < 0:
                c['hypothesis_status'] = 'contradicted'
            else:
                c['hypothesis_status'] = 'not_supported'
        else:
            c['hypothesis_status'] = ('difference_supported' if sig
                                      else 'not_supported')
        summary['n_significant'] += int(sig)
        summary['n_supported'] += int(
            c['hypothesis_status'] == 'supported')
        summary['n_contradicted'] += int(
            c['hypothesis_status'] == 'contradicted')
    if equivalence_margin is not None:
        # TOST over the SAME run-wide family: a non-significant
        # superiority row says nothing about equivalence, so each row
        # additionally gets the two-one-sided-tests verdict computed from
        # its stored summary statistics (n, mean, sd of the seed-paired
        # differences) at the Bonferroni-corrected alpha. Note the
        # asymmetry the summary records: equivalent=True is positive
        # evidence; equivalent=False is still inconclusive (TOST failed
        # to reject, it did not accept the null).
        alpha_tost = alpha / n_family if n_family else None
        n_equiv = 0
        for c in contrasts:
            n = c.get('n_common_seeds', 0)
            mean_diff = c.get('mean_diff')
            sd = c.get('sd_diff')
            if (alpha_tost is None or n < 2
                    or not isinstance(mean_diff, (int, float))
                    or isinstance(mean_diff, bool)
                    or not isinstance(sd, (int, float))
                    or isinstance(sd, bool)
                    or not math.isfinite(float(sd))):
                c['equivalence_margin'] = equivalence_margin
                c['tost_alpha_bonferroni'] = alpha_tost
                c['tost_p_value'] = None
                c['tost_equivalent'] = None
                c['tost_ci_equiv_lo'] = None
                c['tost_ci_equiv_hi'] = None
                c['equivalence_verdict'] = 'not_computed'
                continue
            tost = tost_from_summary(mean_diff, sd, n, equivalence_margin,
                                     alpha=alpha_tost)
            c['equivalence_margin'] = equivalence_margin
            c['tost_alpha_bonferroni'] = alpha_tost
            c['tost_p_value'] = tost['p_value']
            c['tost_equivalent'] = tost['equivalent']
            c['tost_ci_equiv_lo'] = tost['ci_equiv_lo']
            c['tost_ci_equiv_hi'] = tost['ci_equiv_hi']
            c['equivalence_verdict'] = ('equivalent'
                                        if tost['equivalent']
                                        else 'not_equivalent')
            n_equiv += int(bool(tost['equivalent']))
        summary['n_equivalent'] = n_equiv
        summary['tost'] = {
            'margin': equivalence_margin,
            'alpha_family_wise': alpha,
            'alpha_bonferroni': alpha_tost,
            'correction': 'bonferroni',
            'ci_equiv_level': (1.0 - 2.0 * alpha_tost
                               if alpha_tost is not None else None),
            'n_equivalent': n_equiv,
            'method': 'two_one_sided_paired_t',
            'null_hypothesis': (
                'H0: |mean seed-paired final_acc difference| >= margin '
                '(per contrast, from the stored n / mean / sd of the '
                'differences)'),
            'equivalence_rule': (
                'equivalent=True (tost_p_value < corrected alpha) is '
                'positive evidence of practical equivalence within the '
                'pre-specified margin; equivalent=False is inconclusive '
                '(TOST failed to reject), NOT evidence of a difference.'),
        }
        summary['verdict_scope'] = (
            'Only significant_bonferroni=True rows may be cited as '
            'evidence for (or against) the pre-registered prediction; '
            'non-significant rows are inconclusive at this seed budget, '
            'not evidence of no effect. ADDITIONALLY: rows with '
            'tost_equivalent=True may be cited as evidence of practical '
            'equivalence within the pre-specified margin '
            f'{equivalence_margin}; rows with tost_equivalent=False '
            'remain inconclusive on both questions.')
    summary['hypotheses'] = hypotheses if hypotheses is not None else {
        **MATRIX_CONTRAST_HYPOTHESES,
        'cross_route (dynamic names A_KDA - A_MAMBA2, ...)':
            CROSS_ROUTE_CONTRAST_HYPOTHESIS,
    }
    return summary


def _log_contrast_table(contrasts: list) -> None:
    """Log the paired-contrast rows as a readable table."""
    if not contrasts:
        logger.info('  (no paired contrasts computable: need >= 2 common '
                    'seeds across the compared groups)')
        return
    annotated = any('significant_bonferroni' in c for c in contrasts)
    equiv_annotated = any('tost_equivalent' in c for c in contrasts)
    logger.info('\n' + '=' * 104)
    logger.info('Paired memory-route contrasts (seed-matched; final_acc; '
                'positive = first operand better):')
    logger.info('=' * 104)
    for c in contrasts:
        t_str = _fmt_tstat(c.get('t_stat'), width=8, prec=2)
        line = (f"  n_kv={c['n_kv']:<3} {c['memory']:<12} "
                f"{c['contrast']:<34} "
                f"diff={c['mean_diff']:+.4f}  t={t_str}  "
                f"n={c['n_common_seeds']}")
        if annotated:
            sig = 'YES' if c.get('significant_bonferroni') else 'no'
            line += f"  sig={sig:<3} ({c.get('hypothesis_status', 'n/a')})"
        if equiv_annotated:
            eq = c.get('tost_equivalent')
            eq_str = 'YES' if eq else ('no' if eq is False else 'n/a')
            line += f"  equiv={eq_str}"
        logger.info(line)
    if annotated:
        logger.info(
            '  Significance = Bonferroni-corrected paired t-test within the '
            'run-wide contrast family (one-sided for the pre-registered '
            'directional contrasts, two-sided for the cross-route rows); '
            'non-significant rows are inconclusive, not "no effect".')
    if equiv_annotated:
        margin = contrasts[0].get('equivalence_margin')
        logger.info(
            f'  Equivalence = Bonferroni-corrected TOST within the same '
            f'family against the pre-specified margin {margin} accuracy '
            'points; equiv=YES is positive evidence of practical '
            'equivalence, equiv=no is inconclusive (not a difference).')
    logger.info(
        '  NOTE: interaction rows (I=D-B-C+A) support complementarity '
        'claims ONLY with statistical evidence;')
    logger.info(
        '  sign-flip loss/PPL metrics before reusing this helper.')
    logger.info('=' * 104)


_LAYOUT_CONTRAST_PAIRS = {
    'B-A (CSA replaces memory)': ('B', 'A'),
    'C-A (HCA replaces memory)': ('C', 'A'),
    'D-B (add HCA to CSA)': ('D', 'B'),
    'D-C (add CSA to HCA)': ('D', 'C'),
}


def _contrast_consistency_violations(all_results: list,
                                     contrasts: list) -> list[str]:
    """Re-derive every reported contrast from the raw per-seed rows.

    ``_matrix_contrast_report`` produces the contrast table; this function
    checks each reported number against an INDEPENDENT computation from
    the per-seed ``final_acc`` values (recomputed stable-interaction
    arithmetic, separately implemented pair resolution). A report bug —
    wrong group pair, stale aggregate, interaction computed over
    mismatched seed sets, a renamed contrast that no longer corresponds to
    any group combination — is caught before the side file is written
    next to a correct matrix.

    Only REPORTED rows are checked: a partial matrix (missing groups)
    legitimately reports fewer contrasts. Returns human-readable
    violation strings ([] = consistent).
    """
    accs: dict = {}
    for r in all_results:
        if 'error' in r:
            continue
        key = (r.get('n_kv'), r.get('memory'), r.get('group'))
        seed_map = _seed_accuracy_map(r)
        if seed_map:
            accs[key] = seed_map

    def paired_stats(diffs: list):
        """Mean and sample SD of per-seed differences.

        Uses the same exact-constant-vector convention as
        ``multi_seed_stats._sample_sd``: a real spread around a large
        common offset is not mistaken for a flat sweep.
        """
        n = len(diffs)
        mean = sum(diffs) / n
        sd = _sample_sd(diffs, mean)
        return mean, (0.0 if sd is None else sd)

    def paired_mean(a_map: dict, b_map: dict):
        common = sorted(set(a_map) & set(b_map))
        if len(common) < 2:
            return None, len(common), None
        diffs = [a_map[s] - b_map[s] for s in common]
        mean, sd = paired_stats(diffs)
        return mean, len(common), sd

    out: list[str] = []
    for c in contrasts:
        n_kv = c.get('n_kv')
        memory = c.get('memory', '')
        name = c.get('contrast', '')
        reported = c.get('mean_diff')
        if not (isinstance(reported, (int, float))
                and not isinstance(reported, bool)
                and math.isfinite(float(reported))):
            continue  # the row contract reports non-finite means
        where = f'{name!r} ({memory}, n_kv={n_kv})'
        if isinstance(memory, str) and ' vs ' in memory:
            route_a, route_b = (p.strip() for p in memory.split(' vs ', 1))
            # name is 'A_KDA - A_MAMBA2' -> the ablated group letter 'A'.
            group = name.split(' ', 1)[0].split('_', 1)[0]
            a_map = accs.get((n_kv, route_a, f'{route_a.upper()}-{group}'))
            b_map = accs.get((n_kv, route_b, f'{route_b.upper()}-{group}'))
            expected, n_common, expected_sd = paired_mean(a_map, b_map) \
                if a_map is not None and b_map is not None \
                else (None, 0, None)
        elif name.startswith('I='):
            members = [accs.get((n_kv, memory, f'{memory.upper()}-{g}'))
                       for g in 'ABCD']
            if any(m is None for m in members):
                out.append(f'{where}: references a missing group')
                continue
            common = sorted(set.intersection(*(set(m) for m in members)))
            if len(common) < 2:
                out.append(f'{where}: fewer than 2 common seeds in the rows')
                continue
            diffs = [members[3][s] - members[1][s] - members[2][s]
                     + members[0][s] for s in common]
            expected, expected_sd = paired_stats(diffs)
            n_common = len(common)
        else:
            pair = _LAYOUT_CONTRAST_PAIRS.get(name)
            if pair is None:
                out.append(f'{where}: unrecognized contrast name')
                continue
            ga, gb = pair
            a_map = accs.get((n_kv, memory, f'{memory.upper()}-{ga}'))
            b_map = accs.get((n_kv, memory, f'{memory.upper()}-{gb}'))
            expected, n_common, expected_sd = paired_mean(a_map, b_map) \
                if a_map is not None and b_map is not None \
                else (None, 0, None)
        if expected is None:
            out.append(f'{where}: rows imply < 2 common seeds but the '
                       f'contrast was reported')
            continue
        tol = 1e-9 * max(1.0, abs(expected))
        if abs(float(reported) - expected) > tol:
            out.append(f'{where}: mean_diff={reported!r} but the per-seed '
                       f'rows imply {expected!r}')
        if c.get('n_common_seeds') != n_common:
            out.append(f'{where}: n_common_seeds={c.get("n_common_seeds")!r} '
                       f'but the rows share {n_common} seed(s)')
        reported_sd = c.get('sd_diff')
        if (isinstance(reported_sd, (int, float))
                and not isinstance(reported_sd, bool)
                and math.isfinite(float(reported_sd))):
            # The stored SD feeds the TOST equivalence verdict; drift
            # between it and the raw rows would silently move the
            # equivalence CI.
            sd_tol = 1e-9 * max(1.0, abs(expected_sd))
            if abs(float(reported_sd) - expected_sd) > sd_tol:
                out.append(
                    f'{where}: sd_diff={reported_sd!r} but the per-seed '
                    f'rows imply {expected_sd!r}')
    return out


def _aggregate_curve_diagnosis(ok_per_seed: list) -> dict:
    """Roll the per-seed held-out verdicts up to one row-level summary.

    Two things a final-number-only comparison cannot show, and that this
    makes visible:

    * ``curve_verdict`` / ``curve_verdict_counts`` — the modal verdict, so a
      row can be read without walking its ``per_seed`` list;
    * ``curve_verdict_stable`` — whether the surviving seeds AGREE. They
      cannot legitimately disagree about one configuration: the same layout
      at the same budget is not underfitting on seed 42 and overfitting on
      seed 44. A split verdict therefore means the curve is too short or too
      flat to carry information at this step budget, i.e. the monitor needs
      more points before its label can be trusted — a fact about the
      MEASUREMENT, which has to be reported rather than averaged away.

    Also reports the mean generalization gap and the best-validation step
    seen across seeds, so a row can be re-read at the checkpoint it would
    have selected instead of at the last step.
    """
    verdicts = [v for r in ok_per_seed
                for v in [(r.get('curve_diagnosis') or {}).get('verdict')] if v]
    counts: dict = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    gaps = [float(r['generalization_gap']) for r in ok_per_seed
            if r.get('generalization_gap') is not None
            and math.isfinite(float(r['generalization_gap']))]
    best_vals = [r['best_val'] for r in ok_per_seed if r.get('best_val')]
    best_step = None
    if best_vals:
        best_step = min(int(b['step']) for b in best_vals)
    saturated = sum(1 for r in ok_per_seed
                    if (r.get('accuracy_regime') or {}).get('saturated'))
    # Budget status is separate from the over/under verdict: a healthy run
    # can still be descending at the last checkpoint. Count it per seed so
    # the sweep summary can refuse to treat such a ranking as converged.
    budget_limited = sum(
        1 for r in ok_per_seed
        if (r.get('curve_diagnosis') or {}).get('budget_limited'))
    val_still = sum(
        1 for r in ok_per_seed
        if ((r.get('curve_diagnosis') or {}).get('signals') or {}).get(
            'val_still_improving_at_end'))
    # Optimizer health rides along with the curve verdict: a row can have a
    # healthy-looking loss while clipping is masking a gradient explosion,
    # which makes its endpoint no more trustworthy than an overfit one.
    grad_records = [r.get('gradient_norms') for r in ok_per_seed
                    if r.get('gradient_norms')]
    grad_warned = [g for g in grad_records if g.get('notes')]
    grad_clipped = [g for g in grad_records
                    if int(g.get('n_above_clip') or 0) > 0]
    grad_over = [float(g['max_over_clip']) for g in grad_records
                 if g.get('max_over_clip') is not None
                 and math.isfinite(float(g['max_over_clip']))]
    grad_notes = sorted({note for g in grad_warned
                         for note in (g.get('notes') or [])})
    modal = (max(counts, key=lambda k: counts[k])
             if len(set(counts.values())) == 1 or counts else None)
    return {
        'curve_verdict_counts': counts,
        'curve_verdict': modal,
        'curve_verdict_stable': (len(set(counts)) <= 1 if counts else None),
        'n_seeds_budget_limited': int(budget_limited),
        'n_seeds_val_still_improving': int(val_still),
        'curve_verdict_note': (
            None if not counts or len(set(counts)) <= 1 else
            f'the surviving seeds disagree on the curve verdict ({counts}); '
            f'one configuration cannot be both underfitting and '
            f'overfitting, so the held-out curve is too short/flat to label '
            f'this row at the current step budget — raise the step count (or '
            f'the validation frequency) before quoting the verdict.'),
        'mean_generalization_gap': (sum(gaps) / len(gaps)) if gaps else None,
        'best_val_step_any_seed': best_step,
        'n_seeds_at_ceiling': saturated,
        'n_seeds_with_gradient_warnings': len(grad_warned),
        'n_seeds_clipped': len(grad_clipped),
        'max_gradient_over_clip': max(grad_over) if grad_over else None,
        'gradient_warning_notes': grad_notes,
    }


def _curve_diagnosis_summary(all_results: list, label_key: str = 'ratio'):
    """Tally the held-out curve verdict across every row of the sweep.

    A sweep that prints "mean_acc=0.62" for ten layouts hides *why* it got
    there, and the three reasons imply opposite next actions:

    * ``underfitting`` everywhere — the budget, not the layout, is the
      binding constraint; raising the step count may REORDER the ratios, so
      the present ranking is not converged and must not be reported as the
      final answer;
    * ``overfitting`` everywhere — the layouts are memorizing; more steps
      will not help, and the fix is regularization / more data / early
      stopping at the recorded best-validation step;
    * ``saturated`` everywhere — the probe is too easy and the ceiling, not
      the architecture, is what is being measured, so differences between
      rows are compressions of nothing.

    Reporting the tally makes the regime of the WHOLE sweep visible instead
    of leaving it to be guessed from a single row, and keeps the accuracy
    claim honest about what it is measuring.
    """
    rows = [r for r in all_results if 'error' not in r]
    counted = [r for r in rows if r.get('curve_verdict')]
    counts: dict = {}
    gaps = []
    for r in counted:
        counts[r['curve_verdict']] = counts.get(r['curve_verdict'], 0) + 1
        g = r.get('mean_generalization_gap')
        if g is not None and math.isfinite(float(g)):
            gaps.append(float(g))
    def _row_saturated(r) -> bool:
        n_ceiling = int(r.get('n_seeds_at_ceiling') or 0)
        if n_ceiling <= 0:
            return False
        n_ok = int(r.get('n_seeds_ok') or 0)
        # A single lucky seed at 100% is not evidence that the PROBE is too
        # easy; the row has to be at the ceiling for at least half of its
        # surviving seeds. Artifacts written before n_seeds_ok was carried
        # keep the old any-seed behaviour.
        return (n_ceiling * 2 >= n_ok) if n_ok else True

    saturated = sorted(
        str(r.get(label_key))
        for r in counted if _row_saturated(r))
    partially_saturated = sorted(
        str(r.get(label_key))
        for r in counted
        if r.get('n_seeds_at_ceiling') and not _row_saturated(r))
    unstable = sorted(
        str(r.get(label_key))
        for r in counted if r.get('curve_verdict_stable') is False)
    still_improving = sorted(
        str(r.get(label_key))
        for r in counted if r.get('n_seeds_budget_limited'))
    grad_warn_rows = sorted(
        str(r.get(label_key))
        for r in counted if r.get('n_seeds_with_gradient_warnings'))
    grad_over = [float(r['max_gradient_over_clip'])
                 for r in counted
                 if r.get('max_gradient_over_clip') is not None
                 and math.isfinite(float(r['max_gradient_over_clip']))]

    summary = {
        'rows_total': len(rows),
        'rows_with_curve': len(counted),
        'rows_without_curve': len(rows) - len(counted),
        'verdict_counts': counts,
        'mean_generalization_gap': (sum(gaps) / len(gaps)) if gaps else None,
        'max_generalization_gap': max(gaps) if gaps else None,
        'saturated_at_ceiling': saturated,
        'rows_partially_saturated': partially_saturated,
        'rows_with_unstable_verdict': unstable,
        'rows_still_improving_at_budget': still_improving,
        'rows_with_gradient_warnings': grad_warn_rows,
        'max_gradient_over_clip': max(grad_over) if grad_over else None,
        'regime': None,
        'interpretation': '',
        'recommended_action': '',
    }
    n = len(counted)
    if n == 0:
        summary['interpretation'] = (
            'no row carries a held-out validation curve, so this sweep '
            'reports final numbers only and cannot distinguish '
            'overfitting from underfitting from saturation.')
        summary['recommended_action'] = (
            'enable the validation monitor so the sweep can be labelled '
            'before its ranking is interpreted.')
        return summary

    def _frac(v):
        return counts.get(v, 0) / n

    # Saturation is checked before the loss-shape verdicts: when every row
    # is pinned at the task ceiling there is no accuracy headroom, so a
    # rising/flat validation LOSS cannot be repaired by regularization or a
    # larger step budget -- the probe itself has to get harder. Putting the
    # loss verdicts first used to mislabel a fully saturated sweep as
    # "overfitting" and recommend the wrong next experiment.
    if len(saturated) == n:
        summary['regime'] = 'saturated'
        summary['interpretation'] = (
            'every row reached the task ceiling: the probe is too easy to '
            'discriminate these layouts and the measured gaps are '
            'compressions of near-ceiling behaviour.')
        summary['recommended_action'] = (
            'raise the difficulty (more keys, more distractors, a longer '
            'context) so the metric has headroom before the architecture '
            'is ranked.')
    elif _frac('underfitting') >= 0.5:
        summary['regime'] = 'budget_limited'
        summary['interpretation'] = (
            f"{_frac('underfitting'):.0%} of rows are still underfitting at "
            f'the fixed step budget, so the sweep is comparing under-trained '
            f'models: the ranking is budget-dependent and may reorder with '
            f'a larger ABL_STEPS.')
        summary['recommended_action'] = (
            'raise the step budget (and re-run the paired contrasts) before '
            'treating the ordering as a structural result.')
    elif _frac('overfitting') >= 0.5:
        summary['regime'] = 'overfitting'
        summary['interpretation'] = (
            f"{_frac('overfitting'):.0%} of rows overfit: training loss kept "
            f'falling while held-out loss regressed, so the final-step '
            f'weights are past the turning point.')
        summary['recommended_action'] = (
            'select on the recorded best-validation step instead of the last '
            'step, and/or add regularization or more data.')
    elif len(still_improving) * 2 >= n:
        summary['regime'] = 'budget_limited'
        summary['interpretation'] = (
            f"{len(still_improving)}/{n} rows were still descending on BOTH "
            f'the training and validation curves at the final checkpoint '
            f'({still_improving}): their fixed-budget endpoints are not '
            f'converged, so the ordering may reorder with more steps even '
            f'though the curve verdicts look healthy.')
        summary['recommended_action'] = (
            'extend ABL_STEPS (or raise the validation frequency) until '
            'the tail slopes flatten, then re-run the paired contrasts; '
            'do not quote this ranking as a converged structural result.')
    else:
        summary['regime'] = 'mixed'
        summary['interpretation'] = (
            f"curve verdicts are mixed ({counts}); a single aggregate "
            f"accuracy hides per-row differences in what is being measured, "
            f"so read each row's curve_diagnosis before ranking.")
        summary['recommended_action'] = (
            'inspect the per-row curve_diagnosis / best_val fields and '
            're-run any row whose verdict is underfitting or overfitting '
            'before quoting an aggregate ranking.')

    if saturated and summary['regime'] != 'saturated':
        summary['interpretation'] += (
            f" {len(saturated)}/{n} rows are additionally pinned at the "
            f"task ceiling (saturated at ceiling={saturated}), which "
            f"compresses any accuracy gap between them.")
    if still_improving and summary['regime'] != 'budget_limited':
        summary['interpretation'] += (
            f" {len(still_improving)}/{n} rows were still improving at the "
            f'final checkpoint ({still_improving}); extend their budget '
            f'before treating their endpoint as converged.')

    # The measurement's own validity. Seeds of ONE configuration cannot
    # genuinely disagree about being under- or overfitting, so a split
    # verdict is evidence that the held-out curve is too short/flat to carry
    # information at this budget — and a label that unstable must not be
    # quoted as a property of the layout.
    if unstable:
        summary['interpretation'] += (
            f" {len(unstable)}/{n} rows have seeds that DISAGREE on the "
            f"curve verdict ({unstable}); at this step budget the held-out "
            f"curve is too short to label those rows reliably.")
        summary['recommended_action'] = (
            (summary['recommended_action'] + ' ' if
             summary['recommended_action'] else '')
            + 'raise ABL_STEPS (or the validation frequency) until the '
              'per-seed verdicts agree before quoting them.')
    if grad_warn_rows:
        peak = summary['max_gradient_over_clip']
        peak_txt = f'{peak:.1f}x' if peak is not None else '>10x'
        summary['recommended_action'] = (
            (summary['recommended_action'] + ' ').strip()
            + f' Separately, raw gradients exceeded the clip budget in '
            f'{grad_warn_rows} (peak {peak_txt}): lower the learning rate or '
            f'fix the exploding stage before treating the endpoints of those '
            f'rows as converged, even where the loss curve looks healthy.')
    return summary


def _attach_cost_tradeoffs(contrasts: list, all_results: list,
                           label_key: str = 'ratio') -> list:
    """Attach an accuracy-vs-cost verdict to each paired contrast row.

    The paired contrast answers "is the accuracy difference real?"; it does
    NOT answer "is it worth adopting?". A layout that gains 0.5 points while
    doubling parameters and inference latency, or a gain that survives no
    significance test, must not be quoted as an improvement. Each contrast
    therefore also carries the cost ratios and an explicit verdict (see
    ``train_diagnostics.accuracy_cost_tradeoff``).

    Significance is taken from the contrast's OWN corrected test, so the
    cost screen and the statistical screen judge the same evidence.
    """
    by_key = {(r.get('n_kv'), str(r.get(label_key))): r
              for r in all_results if 'error' not in r}
    for c in contrasts:
        cand = by_key.get((c.get('n_kv'), str(c.get('ratio'))))
        ref = by_key.get((c.get('n_kv'), str(c.get('reference_ratio'))))
        if cand is None or ref is None:
            c['cost_tradeoff'] = None
            c['cost_tradeoff_note'] = (
                'no comparable cost row on one side; the accuracy '
                'difference is reported without a cost verdict.')
            continue
        verdict = accuracy_cost_tradeoff(
            cand, ref,
            significant=(bool(c.get('significant_bonferroni'))
                         if c.get('significant_bonferroni') is not None
                         else None))
        c['cost_tradeoff'] = verdict
        label = (f"{c.get('contrast')} (n_kv={c.get('n_kv')})")
        log_tradeoff(verdict, label=label)
        c['cost_tradeoff_note'] = verdict.get('recommendation', '')
    return contrasts


def _attach_matrix_cost_tradeoffs(contrasts: list, all_results: list) -> list:
    """Accuracy-vs-cost screen for the matrix contrasts (ratio-mode parity).

    The matrix's paired tests answer "is the accuracy difference real?";
    they do not answer "is it worth adopting?". A group that gains half an
    accuracy point while doubling parameters or inference latency, or a
    difference that survives no significance test, must not be quoted as an
    improvement — the same rule the ratio sweep already enforces through
    :func:`_attach_cost_tradeoffs`.

    Each contrast row carries the identity of its two sides
    (``candidate_memory`` / ``candidate_group`` and the reference pair), so
    the underlying aggregate rows are looked up exactly; the functional
    interaction contrast (``I=D-B-C+A``) has no single candidate/reference
    pair and records ``cost_tradeoff=None`` with an explanatory note.
    Significance is taken from the row's OWN corrected test, so the cost
    screen and the statistical screen judge the same evidence.
    """
    by_key = {(r.get('n_kv'), r.get('memory'), r.get('group')): r
              for r in all_results if 'error' not in r}
    for c in contrasts:
        c_mem = c.get('candidate_memory')
        c_grp = c.get('candidate_group')
        r_mem = c.get('reference_memory')
        r_grp = c.get('reference_group')
        if not (c_mem and c_grp and r_mem and r_grp):
            c['cost_tradeoff'] = None
            c['cost_tradeoff_note'] = (
                'not a two-row contrast (functional interaction); no '
                'accuracy-vs-cost verdict is defined for it.')
            continue
        cand = by_key.get((c.get('n_kv'), c_mem, c_grp))
        ref = by_key.get((c.get('n_kv'), r_mem, r_grp))
        if cand is None or ref is None:
            c['cost_tradeoff'] = None
            c['cost_tradeoff_note'] = (
                f'missing group row for '
                f'candidate={c_mem}/{c_grp} or reference={r_mem}/{r_grp}; '
                f'the accuracy difference is reported without a cost '
                f'verdict.')
            continue
        verdict = accuracy_cost_tradeoff(
            cand, ref,
            significant=(bool(c.get('significant_bonferroni'))
                         if c.get('significant_bonferroni') is not None
                         else None))
        c['cost_tradeoff'] = verdict
        label = (f"{c.get('contrast')} @ {c.get('memory')} "
                 f"(n_kv={c.get('n_kv')})")
        log_tradeoff(verdict, label=label)
        c['cost_tradeoff_note'] = verdict.get('recommendation', '')
    return contrasts


def _ratio_reference_contrasts(all_results: list,
                               reference_label: str = RATIO_REFERENCE_LABEL):
    """Paired seed-matched differences of every ratio vs the reference.

    Mirrors the matrix contrast report for ``ABL_MODE=ratio``: the default
    3:1:1 allocation is the reference, and every other ratio is compared
    on the COMMON seeds inside each n_kv cell (paired design; the ratio
    rows share the seed list). The sweep is exploratory, so the
    pre-registered test (``RATIO_SWEEP_HYPOTHESES``) is two-sided; the
    corrected verdict is attached by ``_annotate_contrast_significance``
    with ``hypotheses=RATIO_SWEEP_HYPOTHESES``.

    Returns JSON-safe contrast rows; [] when the reference row is absent
    or no alternative shares >= 2 seeds with it.
    """
    rows = [r for r in all_results if 'error' not in r
            and r.get('ratio') is not None]
    by_key = {(r.get('n_kv'), str(r.get('ratio'))): r for r in rows}
    contrasts = []
    for n_kv in sorted({r.get('n_kv') for r in rows}):
        ref = by_key.get((n_kv, reference_label))
        if ref is None:
            continue
        ref_accs = _seed_accuracy_map(ref)
        for label in sorted({str(r.get('ratio')) for r in rows}):
            if label == reference_label:
                continue
            cand = by_key.get((n_kv, label))
            if cand is None:
                continue
            diff, t_stat, n, unanimous, sd = _paired_contrast(
                _seed_accuracy_map(cand), ref_accs)
            if diff is None:
                continue
            hyp = RATIO_SWEEP_HYPOTHESES['alternatives'].get(label, {})
            contrasts.append({
                'n_kv': n_kv,
                'reference_ratio': reference_label,
                'ratio': label,
                'contrast': f'{label} - {reference_label}',
                'mean_diff': diff, 't_stat': t_stat,
                'sd_diff': sd,
                'n_common_seeds': n,
                'unanimous': unanimous,
                'predicted_direction': 'two_sided',
                'question': hyp.get(
                    'question',
                    f'does ratio {label} differ from {reference_label}?'),
                'hypothesis': RATIO_SWEEP_HYPOTHESES['reference']['hypothesis'],
                'prediction': hyp.get(
                    'prediction', 'two-sided paired difference vs the '
                                  'reference ratio.'),
                'falsified_if': hyp.get(
                    'falsified_if',
                    'not applicable: exploratory two-sided contrast.'),
                'metric': 'final_acc',
                'direction': 'higher_is_better',
            })
    return contrasts


def _log_ratio_contrast_table(contrasts: list) -> None:
    """Log the ratio-vs-reference paired contrasts as a readable table."""
    if not contrasts:
        logger.info('  (no ratio-vs-reference contrasts computable: the '
                    'reference ratio row is absent or shares < 2 seeds)')
        return
    logger.info('\n' + '=' * 104)
    logger.info('Paired ratio contrasts vs the reference layout '
                '(seed-matched; final_acc; positive = alternative better):')
    logger.info('=' * 104)
    equiv_annotated = any('tost_equivalent' in c for c in contrasts)
    for c in contrasts:
        sig = 'YES' if c.get('significant_bonferroni') else 'no'
        t_str = _fmt_tstat(c.get('t_stat'), width=8, prec=2)
        line = (f"  n_kv={c['n_kv']:<3} {c['contrast']:<14} "
                f"diff={c['mean_diff']:+.4f}  t={t_str}  "
                f"n={c['n_common_seeds']}  sig={sig:<3} "
                f"({c.get('hypothesis_status', 'n/a')})")
        if equiv_annotated:
            eq = c.get('tost_equivalent')
            eq_str = 'YES' if eq else ('no' if eq is False else 'n/a')
            line += f"  equiv={eq_str}"
        logger.info(line)
    logger.info(
        '  Significance = Bonferroni-corrected TWO-SIDED paired t-test '
        'over the run-wide ratio-contrast family; non-significant ratios '
        'are statistically indistinguishable from the default 3:1:1 at '
        'this seed budget (not proof of equivalence).')
    if equiv_annotated:
        margin = contrasts[0].get('equivalence_margin')
        logger.info(
            f'  Equivalence = Bonferroni-corrected TOST over the same '
            f'family against the pre-specified margin {margin} accuracy '
            'points; equiv=YES is positive evidence that the ratio is '
            'practically equivalent to the default, equiv=no is '
            'inconclusive.')
    logger.info('=' * 104)


def _attach_tuned_baseline_flags(all_results: list,
                                 baseline_contrasts: list) -> None:
    """Copy the direct tuned-baseline verdict onto every per-group row.

    The result reader should not have to join a side table to see whether
    a hybrid cleared the standard rung: the corrected verdict, effect and
    threshold ride on the row itself. Baseline rows (and groups without a
    paired reference) carry explicit ``None`` rather than a misleading
    ``False``.
    """
    by_group = {(c.get('n_kv'), c.get('candidate_group')): c
                for c in baseline_contrasts}
    for r in all_results:
        if 'error' in r:
            continue
        r['tuned_baseline_group'] = TUNED_BASELINE_GROUP
        c = by_group.get((r.get('n_kv'), r.get('group')))
        if c is None:
            r['beats_tuned_baseline'] = None
            r['tuned_baseline_diff'] = None
            r['tuned_baseline_t_stat'] = None
            r['tuned_baseline_alpha_bonferroni'] = None
            continue
        r['beats_tuned_baseline'] = bool(
            c.get('significant_bonferroni')
            and c.get('mean_diff') is not None
            and c['mean_diff'] > 0)
        r['tuned_baseline_diff'] = c.get('mean_diff')
        r['tuned_baseline_t_stat'] = c.get('t_stat')
        r['tuned_baseline_alpha_bonferroni'] = c.get('alpha_bonferroni')


def _log_tuned_baseline_table(rows: list, summary: dict) -> None:
    """Log the direct group-vs-tuned-softmax tests."""
    if not rows:
        logger.info('\n  Direct tuned-baseline tests: not available '
                    '(ABL_SIMPLE_BASELINES=0 or no paired groups).')
        return
    logger.info('\n' + '=' * 104)
    logger.info('Direct tuned-baseline tests (seed-matched final_acc vs '
                f'{TUNED_BASELINE_GROUP}; positive = group better):')
    logger.info('=' * 104)
    for c in rows:
        sig = 'YES' if c.get('significant_bonferroni') else 'no'
        t_str = _fmt_tstat(c.get('t_stat'), width=8, prec=2)
        logger.info(
            f"  n_kv={c['n_kv']:<3} {c['candidate_group']:<16} "
            f"diff={c['mean_diff']:+.4f}  t={t_str}  "
            f"n={c['n_common_seeds']}  sig={sig:<3} "
            f"({c.get('hypothesis_status', 'n/a')})  "
            f"lr={c.get('candidate_lr')} vs base={c.get('baseline_lr')}")
    logger.info(
        '  Significance = Bonferroni-corrected ONE-SIDED seed-paired '
        't-test over this baseline-test family; non-significant rows are '
        'inconclusive, not evidence that the hybrid fails to help.')
    logger.info(f"  supported={summary.get('n_supported', 0)}/"
              f"{summary.get('n_contrasts', 0)}  "
              f"significant={summary.get('n_significant', 0)}/"
              f"{summary.get('n_contrasts', 0)}")
    logger.info('=' * 104)


def _run_matrix_mode(device, n_seeds, steps, n_kv_list, *,
                     error_batches=0, error_batch=64, error_examples=2):
    """ABL_MODE=matrix driver: build + run + report the explicit matrix."""
    preset = os.environ.get('ABL_MATRIX', 'main8').strip().lower()
    total_layers = parse_int_env('ABL_TOTAL_LAYERS', 10, min_value=1,
                                 logger=logger)
    d_model = parse_int_env('ABL_D_MODEL', 32, min_value=8, logger=logger)
    raw_mem = os.environ.get('ABL_MEMORY', 'kda,mamba2,rwkv7')
    memory_types = [m.strip() for m in raw_mem.split(',') if m.strip()]
    valid_mem = {'kda', 'mamba2', 'rwkv7'}
    unknown_mem = [m for m in memory_types if m not in valid_mem]
    if unknown_mem:
        raise ValueError(
            f"ABL_MEMORY includes unknown memory route(s) {unknown_mem}; "
            f"valid: {sorted(valid_mem)}.")
    memory_types = list(dict.fromkeys(memory_types))
    logger.info(f'  mode          : matrix (preset={preset})')
    logger.info(f'  total_layers  : {total_layers}')
    logger.info(f'  d_model       : {d_model}')
    logger.info(f'  memory routes : {memory_types}')
    groups = build_experiment_matrix(preset, total_layers, memory_types)
    include_baselines = parse_int_env(
        'ABL_SIMPLE_BASELINES', 1, min_value=0, logger=logger)
    if include_baselines:
        groups = groups + build_simple_baseline_groups(total_layers)
        logger.info('  simple baselines: '
                    f'{[g[0] for g in groups[-len(SIMPLE_BASELINE_KINDS):]]} '
                    '(same depth/recipe; ABL_SIMPLE_BASELINES=0 disables)')
    else:
        logger.info('  simple baselines: DISABLED (ABL_SIMPLE_BASELINES=0)')

    # Tune the simple baselines BEFORE the main sweep. The standard softmax
    # rung is the bar a hybrid must clear; leaving it at one fixed lr would
    # confound "the operator helps" with "the baseline was untuned". Each
    # candidate is selected on the held-out validation stream and then
    # re-trained with the full seed count in the main loop below.
    baseline_specs = [g for g in groups
                      if g[1] in SIMPLE_BASELINE_KINDS]
    baseline_lrs = {}
    baseline_tuning = {}
    baseline_tune_enabled = parse_int_env(
        'ABL_BASELINE_TUNE', 1, min_value=0, logger=logger)
    if baseline_specs and baseline_tune_enabled:
        baseline_lr_grid = _parse_lr_list(
            'ABL_BASELINE_LRS', DEFAULT_BASELINE_TUNE_LRS)
        tune_seeds = parse_int_env(
            'ABL_BASELINE_TUNE_SEEDS', min(int(n_seeds), 3),
            min_value=1, logger=logger)
        tune_seeds = max(1, min(tune_seeds, int(n_seeds)))
        logger.info(f'\nTuning simple baselines on validation stream '
                    f'(lr grid={baseline_lr_grid}, '
                    f'{tune_seeds} seed(s), {len(n_kv_list)} n_kv cell(s)):')
        for spec in baseline_specs:
            best_lr, record = tune_simple_baseline_lr(
                spec, baseline_lr_grid, n_kv_list=n_kv_list,
                tune_seeds=tune_seeds, steps=steps, device=device,
                d_model=d_model)
            baseline_lrs[spec[0]] = best_lr
            baseline_tuning[spec[0]] = record
    elif baseline_specs:
        logger.info('  baseline tuning: DISABLED '
                    '(ABL_BASELINE_TUNE=0); using lr=3e-3')
        for spec in baseline_specs:
            baseline_lrs[spec[0]] = 3e-3
            baseline_tuning[spec[0]] = {
                'group': spec[0],
                'selected_lr': 3e-3,
                'status': 'disabled_by_ABL_BASELINE_TUNE',
                'note': ('tuning disabled; the historical fixed lr is '
                         'kept so ABL_BASELINE_TUNE=0 reproduces the old '
                         'numeric behaviour'),
            }

    # Pre-sweep overfit sanity gate: one cell per DISTINCT layout on the
    # first memory route (the memory operators themselves already have
    # their own exp4 gradient gates). Gating every layout catches a broken
    # CSA/HCA gradient path that a pure-recurrent group would hide; the
    # route-level check is deliberately not repeated for every memory
    # operator to keep the gate cheap.
    gate_cells = []
    _seen_layouts = set()
    _first_memory = groups[0][1]
    for spec in groups:
        if spec[1] != _first_memory:
            continue
        _overrides = spec[4] if len(spec) > 4 else None
        _layout_key = (tuple(spec[2]),
                       tuple(sorted((_overrides or {}).items())))
        if _layout_key in _seen_layouts:
            continue
        _seen_layouts.add(_layout_key)
        _gid = spec[0]
        gate_cells.append(
            (f'group {_gid}',
             lambda spec=spec: _build_matrix_model(spec, d_model, device)))
    gate_record, proceed = _run_ablation_overfit_gate(
        gate_cells, device, n_kv=n_kv_list[0])
    if not proceed:
        return 1

    n_tests = len(groups) * len(n_kv_list)
    alpha_corrected = 0.05 / n_tests
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=exact-beta-bisection)')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}')
        logger.info(f'{"=" * 70}')
        for spec in groups:
            gid = spec[0]
            logger.info(f'\n-- group {gid} (n_kv={n_kv}, {n_seeds} seeds) --')
            try:
                res = eval_group_multi_seed(
                    spec, n_seeds=n_seeds, steps=steps, device=device,
                    n_kv=n_kv, d_model=d_model,
                    lr=baseline_lrs.get(gid, 3e-3),
                    error_batches=error_batches,
                    error_batch=error_batch,
                    max_error_examples=error_examples)
                t_stat = res.get('t_stat_vs_chance')
                n_ok = res.get('n_seeds_ok', 0)
                if t_stat is not None and n_ok >= 2:
                    crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
                    res['t_crit_bonferroni'] = crit
                    res['significant_bonferroni'] = (
                        crit is not None and t_stat > crit)
                else:
                    res['t_crit_bonferroni'] = None
                    res['significant_bonferroni'] = False
                all_results.append(res)
                logger.info(f"  layout={res['layout']}  "
                            f"n_params={res['n_params']}  "
                            f"n_layers={res['n_layers']}")
                _ci = res['ci95_acc']
                _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
                logger.info(f"  -> mean_acc={res['mean_acc']:.4f} +/- {_ci_str} "
                            f"(std={_fmt_num(res['std_acc'])}, "
                            f"t_vs_chance={_fmt_tstat(res['t_stat_vs_chance'], width=0, prec=2)}, "
                            f"sig_bonferroni={res['significant_bonferroni']})")
                logger.info(f"     mean_fwd={res['mean_fwd_ms']:.2f}ms")
            except Exception as e:
                import traceback as _tb
                logger.error(f'  group {gid} FAILED: {e}')
                _tb.print_exc()
                all_results.append({
                    'group': gid, 'memory': spec[1],
                    'description': spec[3] if len(spec) > 3 else '',
                    'n_kv': n_kv, 'n_seeds': n_seeds, 'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds, 'n_seeds_total': n_seeds,
                    'seeds': [], 'error': str(e), 'layout': None,
                    'mean_acc': None, 'ci95_acc': None, 'std_acc': None,
                    'mean_fwd_ms': None, 'n_params': None,
                    'n_layers': total_layers, 'chance_acc': 1.0 / VOCAB,
                    't_stat_vs_chance': None, 't_crit_bonferroni': None,
                    'significant_bonferroni': False,
                    'mean_train_time_s': None, 'per_seed': [],
                })

    print('\n' + '=' * 118)
    print(f"{'n_kv':>4} | {'group':>16} | {'memory':>7} | {'layout':>28} | "
          f"{'layers':>6} | {'params':>8} | {'mean_acc':>10} | {'+/- CI95':>10} | "
          f"{'t_vs_chance':>12} | {'fwd_ms':>8}")
    print('-' * 118)
    for r in all_results:
        if 'error' in r:
            print(f"{r['n_kv']:>4} | {r['group']:>16} | {r.get('memory', ''):>7} | "
                  f"{'(error)':>28} | {r['n_layers']:>6} | {'-':>8} | "
                  f"{'-':>10} | {'-':>10} | {'-':>12} | {'-':>8}   ERROR: {r['error']}")
            continue
        layout_str = (r.get('layout', '') or '')[:28]
        n_params = r['n_params'] if r.get('n_params') is not None else 0
        mean_acc = r['mean_acc'] if r.get('mean_acc') is not None else 0.0
        ci95 = r['ci95_acc']
        fwd = r['mean_fwd_ms'] if r.get('mean_fwd_ms') is not None else 0.0
        ci_str = f"{ci95:>10.4f}" if ci95 is not None else f"{'n/a':>10}"
        print(f"{r['n_kv']:>4} | {r['group']:>16} | {r['memory']:>7} | "
              f"{layout_str:>28} | {r['n_layers']:>6} | {n_params:>8} | "
              f"{mean_acc:>10.4f} | {ci_str} | "
              f"{_fmt_tstat(r.get('t_stat_vs_chance'), width=12, prec=2)} | {fwd:>8.2f}")
    print(f"{'':>4} | {'chance':>16} | {'':>7} | {'':>28} | {'':>6} | {'':>8} | "
          f"{1.0 / VOCAB:>10.4f} | {'':>10} | {'':>12} | {'':>8}")

    logger.info(f'\nControlled-depth design: all matrix groups contain '
                f'{total_layers} layers.')
    logger.info('  Parameter counts differ across memory routes / presets and are')
    logger.info('  reported per row; interpret layout effects together with')
    logger.info('  n_params, and run the budget-matching follow-up before claiming')
    logger.info('  capacity-neutral comparisons.')

    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results),
                       default=0)
    _c = 1.0 / VOCAB
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and 0.5 * _c < r['mean_acc'] < 1.5 * _c]
    far_below = [r for r in all_results
                 if 'error' not in r
                 and r.get('mean_acc') is not None
                 and r['mean_acc'] <= 0.5 * _c]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0
                         and len(near_chance) < len(all_results) // 2
                         and len(far_below) == 0)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  groups with significant_bonferroni=True: '
                f'{n_any_sig}/{len(all_results)}')
    logger.info(f'  groups near chance (0.5x-1.5x): '
                f'{len(near_chance)}/{len(all_results)}')
    logger.info(f'  groups far below chance (<=0.5x): '
                f'{len(far_below)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    logger.info('=' * 70)
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds
        r['abl_mode'] = 'matrix'
        r['abl_matrix_preset'] = preset
        r['total_layers'] = total_layers
        r['d_model'] = d_model
        r['csa_indexer_normalize_qk'] = True
        r['kda_backend'] = os.environ.get('KDA_BACKEND', 'reference')
        r['mamba_backend'] = os.environ.get('MAMBA_BACKEND', 'reference')
        r['rwkv_backend'] = os.environ.get('RWKV_BACKEND', 'reference')
        r['rwkv7_upstream_binding'] = RWKV7_UPSTREAM_BINDING
        r['significance_scope'] = (
            'vs_chance_bonferroni_and_seed_paired_layout_contrasts_'
            'bonferroni_and_tost_equivalence_and_tuned_softmax_'
            'baseline_bonferroni')

    # Paired seed-matched contrasts (B-A / C-A / D-B / D-C / interaction
    # per route; A/D replacement across routes — the KDA-vs-RWKV-7
    # replacement question) — written to their own side file so the
    # per-group rows' schema stays unchanged for existing consumers. Each
    # row is tested against its PRE-REGISTERED hypothesis under one
    # Bonferroni family over the run's contrast rows; only
    # significant_bonferroni=True rows may be cited as evidence.
    contrasts = _matrix_contrast_report(all_results)
    equiv_margin = _parse_equivalence_margin(
        'ABL_EQUIV_MARGIN', CONTRAST_EQUIVALENCE_MARGIN)
    contrast_tests = _annotate_contrast_significance(
        contrasts, equivalence_margin=equiv_margin)
    # Accuracy-vs-cost screen on the SAME contrasts (ratio-mode parity):
    # run after the corrected significance is attached so both screens
    # judge the same evidence.
    _attach_matrix_cost_tradeoffs(contrasts, all_results)
    _log_contrast_table(contrasts)

    # Direct answer to the prior question: does each hybrid beat the tuned
    # standard rung? This is a SEPARATE pre-registered family from the
    # layout contrasts; it must not be folded into their Bonferroni family
    # (that would make both screens harder to interpret).
    baseline_contrasts = []
    if include_baselines:
        baseline_contrasts = _tuned_baseline_contrast_report(all_results)
    if baseline_contrasts:
        baseline_tests = _annotate_contrast_significance(
            baseline_contrasts,
            hypotheses={'tuned_baseline': TUNED_BASELINE_HYPOTHESIS})
        baseline_tests['reference_group'] = TUNED_BASELINE_GROUP
        baseline_tests['family_scope'] = (
            'every non-baseline group x n_kv, tested against the tuned '
            f'{TUNED_BASELINE_GROUP} at the same n_kv')
        baseline_tests['rows'] = baseline_contrasts
        _log_tuned_baseline_table(baseline_contrasts, baseline_tests)
    else:
        baseline_tests = {
            'status': ('disabled' if not include_baselines
                       else 'no_paired_tuned_baseline_reference'),
            'reference_group': TUNED_BASELINE_GROUP,
            'n_contrasts': 0,
            'n_significant': 0,
            'n_supported': 0,
        }
        _log_tuned_baseline_table([], baseline_tests)
    _attach_tuned_baseline_flags(all_results, baseline_contrasts)

    # Independent design audit: a complete factorial is only meaningful if
    # every promised group ran, every non-baseline group used the same
    # training recipe, and every contrasted cell shares the requested seed
    # set. This catches a survivor-only matrix or a drifted run that the
    # contrast arithmetic itself cannot see.
    design_audit = _matrix_design_audit(
        all_results, groups, preset=preset, n_kv_list=n_kv_list,
        n_seeds=n_seeds)
    logger.info(f'Design audit: {design_audit["status"]} '
                f'({len(design_audit["missing_groups"])} missing groups, '
                f'{len(design_audit["violations"])} violations)')
    if design_audit['status'] not in ('complete_factorial',
                                      'complete_preset'):
        logger.warning(f'Matrix design is NOT verified: '
                       f'{design_audit["violations"]}')

    # Prospective audit of the pre-registered Exp 5b plan: the actual seed
    # budget and the number of contrast rows in THIS run are compared with
    # the declared SESOI / family. The post-hoc design_audit above checks
    # that the promised cells ran; this audit checks whether the run is
    # powered to make the claims those cells invite. The analyzed seed set
    # is the intersection of the seeds that survived in every group: a
    # dropped seed makes the matrix a complete-case analysis, which the
    # audit labels ``incomplete_attrition`` instead of confirmatory.
    matrix_analyzed_population = common_analyzed_seeds(
        all_results, finite_keys=('final_acc', 'final_loss'))
    matrix_experiment_design = audit_experiment_design(
        'exp5b_matrix',
        seeds=[42 + i for i in range(n_seeds)],
        levels={
            'memory': list(memory_types),
            'layout_unit': sorted({_unit_of_group(spec[0]) for spec in groups}),
        },
        family_size=max(1, len(contrasts)),
        analyzed_seeds=matrix_analyzed_population,
    )
    logger.info(f'Design audit: {matrix_experiment_design["status"]} '
                f'(requested={matrix_experiment_design["n_seeds"]}, '
                f'analyzed={matrix_experiment_design["n_analyzed"]}, '
                f'family={matrix_experiment_design["planning"]["family_size"]}, '
                f'required='
                f'{matrix_experiment_design["planning"]["required_seeds"]})')
    if matrix_experiment_design['status'] == 'incomplete_attrition':
        logger.warning(
            'Failed seeds were dropped from the matrix analysis; the '
            'reported contrasts are complete-case, not the pre-registered '
            'complete block (see metadata.experiment_design.'
            'analysis_population).')

    # Regime of the matrix as a whole (same vocabulary as the ratio sweep):
    # are the rows saturated / underfitting / overfitting, and do the seeds
    # agree on the verdict? Without this, a final-number-only table cannot
    # say whether a group's score is a converged property or a budget
    # artifact.
    matrix_curve_summary = _curve_diagnosis_summary(all_results, 'group')
    logger.info('\n' + '=' * 70)
    logger.info('Training-curve regime (held-out monitor, not final numbers):')
    logger.info(f'  rows with a curve: {matrix_curve_summary["rows_with_curve"]}'
                f'/{matrix_curve_summary["rows_total"]}   '
                f'verdicts={matrix_curve_summary["verdict_counts"]}')
    if matrix_curve_summary['mean_generalization_gap'] is not None:
        logger.info(f'  mean generalization gap: '
                    f"{matrix_curve_summary['mean_generalization_gap']:.4f} "
                    f"(max {matrix_curve_summary['max_generalization_gap']:.4f})")
    logger.info(f'  regime: {matrix_curve_summary["regime"]}')
    logger.info(f'  {matrix_curve_summary["interpretation"]}')
    if matrix_curve_summary['recommended_action']:
        logger.info(f'  ACTION: {matrix_curve_summary["recommended_action"]}')
    logger.info('=' * 70)

    os.makedirs('results', exist_ok=True)
    run_meta = {
        'preset': preset,
        'memory_routes': list(memory_types),
        'include_simple_baselines': bool(include_baselines),
        'baseline_tuning': {
            'enabled': bool(include_baselines and baseline_tune_enabled),
            'reference_group': TUNED_BASELINE_GROUP,
            'records': baseline_tuning,
            'note': ('Baseline lrs are selected on validation-tail accuracy '
                     '(held-out monitor stream), then re-trained with the '
                     'full seed count. ABL_BASELINE_TUNE=0 keeps the '
                     'historical fixed lr=3e-3.'),
        },
        'n_kv_list': list(n_kv_list),
        'n_seeds': n_seeds,
        'steps': steps,
        'total_layers': total_layers,
        'd_model': d_model,
        'device': str(device),
        # The preset's promised group ids; the row contract refuses to
        # write a survivor-only matrix (a crash mid-sweep) as if complete.
        'expected_groups': [spec[0] for spec in groups],
        # Machine-readable check that the promised cells actually ran with
        # a common recipe and a common seed set (see _matrix_design_audit).
        'design_audit': design_audit,
        # Prospective audit of the same run against the pre-registered
        # Exp 5b plan (seed budget, family size, required n at the SESOI).
        'experiment_design': matrix_experiment_design,
        # Pre-registered design intent (fixed before the numbers were
        # read): what each layout group is for, and the prediction +
        # falsification criterion for every paired contrast family. The
        # per-row copies ride with each contrast row; this block is the
        # single readable registration for the whole experiment.
        'hypotheses': {
            'groups': MATRIX_GROUP_ROLES,
            'simple_baselines': (SIMPLE_BASELINE_HYPOTHESES
                                 if include_baselines else {}),
            'tuned_baseline': (TUNED_BASELINE_HYPOTHESIS
                               if include_baselines else {}),
            'contrasts': MATRIX_CONTRAST_HYPOTHESES,
            'cross_route': CROSS_ROUTE_CONTRAST_HYPOTHESIS,
            'note': ('Pre-registered in run_ablation.MATRIX_CONTRAST_'
                     'HYPOTHESES / CROSS_ROUTE_CONTRAST_HYPOTHESIS. A '
                     'contrast supports its prediction only when '
                     'significant_bonferroni=True in the contrasts side '
                     'file; non-significant rows are inconclusive. '
                     'PRACTICAL EQUIVALENCE must be read from the TOST '
                     'fields (tost_equivalent=True), not from a '
                     'non-significant superiority test; the margin was '
                     'pre-specified before the run.'),
        },
        # Corrected significance + equivalence verdicts for the contrast
        # family (TOST margin is inside this block, so the registration
        # travels with the results).
        'contrast_tests': contrast_tests,
        'contrast_equivalence_margin': equiv_margin,
        # Separate family: every group's direct paired test against the
        # TUNED softmax rung (the baseline-superiority question).
        'baseline_tests': baseline_tests,
        # Sweep-wide training regime + the accuracy-vs-cost screen, so the
        # artifact records not only which group won but whether the win
        # survives the curve diagnosis and what it cost.
        'curve_diagnosis_summary': matrix_curve_summary,
        'cost_tradeoff_policy': {
            'min_effect': DEFAULT_MIN_EFFECT,
            'max_cost_ratio': DEFAULT_MAX_COST_RATIO,
            'cost_keys': list(DEFAULT_COST_KEYS),
            'note': (
                'A paired contrast is only an adoptable improvement if its '
                'corrected test is significant AND the accuracy gain clears '
                'min_effect AND no cost axis exceeds max_cost_ratio. The '
                'accuracy difference and the cost ratios are reported '
                'separately so either can be quoted on its own.'),
        },
        'overfit_sanity_check': {
            'enabled': gate_record['enabled'],
            'strict': gate_record['strict'],
            'steps': gate_record.get('steps'),
            'seed': gate_record.get('seed'),
            'train_batch': gate_record.get('train_batch'),
            'reports': [_compact_overfit_report(r)
                        for r in gate_record['reports']],
        },
    }
    # Independent re-derivation of every reported contrast from the raw
    # per-seed rows: the analysis side file must not contradict the matrix
    # it is derived from.
    violations = _contrast_consistency_violations(all_results, contrasts)
    if violations:
        raise RowContractError(
            'matrix contrast report is inconsistent with the per-seed rows: '
            + '; '.join(violations[:5])
            + (f' (+{len(violations) - 5} more)' if len(violations) > 5
               else ''))
    try:
        write_results_checked('results/exp5b_ablation_matrix.json',
                              all_results, kind='exp5b_ablation_matrix',
                              metadata=run_meta)
        write_results_checked(
            'results/exp5b_ablation_matrix_contrasts.json', contrasts,
            kind='exp5b_ablation_matrix_contrasts', metadata=run_meta)
    except RowContractError as e:
        logger.error(f'result rows failed their contract: {e}')
        return 1
    try:
        write_results_json(_run_provenance(n_seeds, steps, gate_record),
                           'results/exp5b_ablation_matrix_provenance.json',
                           logger=logger)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info('\nSaved: results/exp5b_ablation_matrix.json '
                '(schema kind=exp5b_ablation_matrix)')
    n_errors = sum(1 for r in all_results if 'error' in r)
    n_incomplete = sum(
        1 for r in all_results
        if 'error' not in r
        and r.get('n_seeds_ok', n_seeds) != r.get('n_seeds', n_seeds)
    )
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_results)} groups errored and '
            f'{n_incomplete}/{len(all_results)} have incomplete seed sets. '
            'Returning non-zero to prevent survivor-only aggregates from '
            'being treated as a successful experiment.')
        return 1
    return 0


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 5: Hybrid Layout Ablation (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')
    # ABL_MODE selects the ablation family:
    #   * 'ratio'  (default) — the historical KDA:CSA:HCA ratio sweep at
    #     fixed 5-layer depth. Output: results/exp5_ablation.json.
    #   * 'matrix' — the explicit-layout eight-group design (memory route
    #     x A/B/C/D), plus the csa_csa / position / budget follow-up
    #     presets. Output: results/exp5b_ablation_matrix.json.
    abl_mode = os.environ.get('ABL_MODE', 'ratio').strip().lower()
    if abl_mode not in ('ratio', 'matrix'):
        raise ValueError(
            f"ABL_MODE={abl_mode!r} must be 'ratio' or 'matrix'.")
    n_seeds = parse_int_env('ABL_SEEDS', 7, min_value=1, logger=logger)
    steps = parse_int_env('ABL_STEPS', 100, min_value=1, logger=logger)
    n_kv_list = _parse_nkv_list('ABL_NKV', '1')
    for n_kv in n_kv_list:
        if 2 * n_kv > VOCAB:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"VOCAB={VOCAB}; reduce n_kv or increase VOCAB.")
        if 2 * n_kv >= SEQ_LEN:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< SEQ_LEN={SEQ_LEN} (need room for the cue token at the end).")
    # Results-analysis controls. The held-out validation curve is always
    # recorded (every steps//4); the bad-case pass is ON by default with
    # Exp 4's budget (2 batches x 64 held-out samples per seed, dedicated
    # RNG stream) so the sweep's artifact carries WHY each layout fails.
    # Set ABL_ERROR_BATCHES=0 to disable it.
    error_batches = parse_int_env('ABL_ERROR_BATCHES', 2, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('ABL_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    error_examples = parse_int_env('ABL_ERROR_EXAMPLES', 2, min_value=0,
                                   logger=logger)
    logger.info(f'  n_seeds={n_seeds}, steps={steps}, n_kv={n_kv_list}')
    logger.info(f'  bad-case      : {error_batches}x{error_batch} held-out '
                f'samples per seed '
                f'(stream +{ERROR_ANALYSIS_SEED_OFFSET})')

    if abl_mode == 'matrix':
        rc = _run_matrix_mode(device, n_seeds, steps, n_kv_list,
                              error_batches=error_batches,
                              error_batch=error_batch,
                              error_examples=error_examples)
        _append_ablation_manifest('matrix', rc, n_seeds, steps, n_kv_list)
        return rc

    # Pre-sweep overfit sanity gate on the reference layout: if the shared
    # training path cannot drive one fixed MQAR batch to ~100% train
    # accuracy, every sweep row below would be measuring a broken recipe.
    gate_ratio = (3, 1, 1)
    _gate_label = (f'layout KDA:CSA:HCA='
                   f'{gate_ratio[0]}:{gate_ratio[1]}:{gate_ratio[2]}')
    gate_record, proceed = _run_ablation_overfit_gate(
        [(_gate_label,
          lambda: _build_layout_model(gate_ratio, 32, device))],
        device, n_kv=n_kv_list[0])
    if not proceed:
        return 1

    ratios = [
        (3, 1, 1),
        (4, 1, 0), (4, 0, 1),
        (2, 2, 1), (2, 1, 2),
        (1, 3, 1), (1, 1, 3),
    ]
    if any(sum(r) != 5 for r in ratios):
        raise AssertionError('ablation layouts must all have total depth 5')
    # Simple baselines (linear / softmax) at the SAME depth (5 layers) and
    # same training recipe as every ratio group, so the sweep can answer
    # "does the hybrid mix beat a tuned simple model?" instead of only
    # "which mix is least bad". Mirror matrix mode's ABL_SIMPLE_BASELINES.
    include_baselines = parse_int_env(
        'ABL_SIMPLE_BASELINES', 1, min_value=0, logger=logger)
    baseline_cells = []
    if include_baselines:
        for kind in SIMPLE_BASELINE_KINDS:
            layout = (kind,) * 5
            desc = (f'{kind} baseline: 5 x '
                    f"{'identity mixer (no token mixing)' if kind == 'linear' else 'causal dense softmax attention'} "
                    f'(same depth/training recipe as the ratio groups)')
            baseline_cells.append((f'BASELINE-{kind}', kind, layout, desc))
        logger.info('  simple baselines: '
                    f'{[c[0] for c in baseline_cells]} '
                    '(same depth 5 / recipe; ABL_SIMPLE_BASELINES=0 disables)')
    else:
        logger.info('  simple baselines: DISABLED (ABL_SIMPLE_BASELINES=0)')

    # Each cell is ('ratio', ratio_tuple) or ('baseline',
    # (group_id, memory_type, layout, description)).
    cells = [('ratio', r) for r in ratios] + \
        [('baseline', c) for c in baseline_cells]
    n_tests = len(cells) * len(n_kv_list)
    alpha_corrected = 0.05 / n_tests
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'  {n_tests} one-sample t-tests vs chance '
                f'({len(ratios)} ratios + {len(baseline_cells)} baselines x '
                f'{len(n_kv_list)} n_kv); '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=exact-beta-bisection)')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for kind, spec in cells:
            if kind == 'ratio':
                label = f'ratio KDA:CSA:HCA = {spec[0]}:{spec[1]}:{spec[2]}'
                ratio_str = f'{spec[0]}:{spec[1]}:{spec[2]}'
            else:
                group_id, _memory, _layout, desc = spec
                label = desc
                ratio_str = group_id
            logger.info(f'\n-- {label} (n_kv={n_kv}, {n_seeds} seeds) --')
            try:
                kw = dict(n_seeds=n_seeds, steps=steps, device=device,
                          n_kv=n_kv, error_batches=error_batches,
                          error_batch=error_batch,
                          max_error_examples=error_examples)
                if kind == 'ratio':
                    res = eval_layout_multi_seed(spec, **kw)
                else:
                    res = eval_group_multi_seed(spec, **kw)
                res['ratio'] = ratio_str
                res['is_baseline'] = (kind == 'baseline')
                t_stat = res.get('t_stat_vs_chance')
                n_ok = res.get('n_seeds_ok', 0)
                if t_stat is not None and n_ok >= 2:
                    crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
                    res['t_crit_bonferroni'] = crit
                    res['significant_bonferroni'] = (
                        crit is not None and t_stat > crit
                    )
                else:
                    res['t_crit_bonferroni'] = None
                    res['significant_bonferroni'] = False
                all_results.append(res)
                logger.info(f"  layout={res.get('layout')}  "
                            f"n_params={res.get('n_params')}  "
                            f"n_layers={res.get('n_layers')}")
                _ci = res.get('ci95_acc')
                _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
                logger.info(f"  -> mean_acc={res.get('mean_acc'):.4f} +/- {_ci_str} "
                            f"(std={_fmt_num(res.get('std_acc'))}, "
                            f"t_vs_chance={_fmt_tstat(res.get('t_stat_vs_chance'), width=0, prec=2)}, "
                            f"sig_bonferroni={res['significant_bonferroni']})")
                logger.info(f"     mean_fwd={res.get('mean_fwd_ms'):.2f}ms  "
                            f"macro_f1={_fmt_num(res.get('mean_f1_macro'))}  "
                            f"balanced_acc="
                            f"{_fmt_num(res.get('mean_balanced_accuracy'))}")
            except Exception as e:
                import traceback as _tb
                logger.error(f"  {ratio_str} FAILED: {e}")
                _tb.print_exc()
                all_results.append({
                    'ratio': ratio_str,
                    'n_kv': n_kv,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    'layout': None,
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_fwd_ms': None,
                    'n_params': None,
                    'n_layers': 5 if kind == 'baseline' else sum(spec),
                    'chance_acc': 1.0 / VOCAB,
                    't_stat_vs_chance': None,
                    't_crit_bonferroni': None,
                    'significant_bonferroni': False,
                    'curve_verdict_counts': {},
                    'curve_verdict': None,
                    'curve_verdict_stable': None,
                    'curve_verdict_note': None,
                    'mean_generalization_gap': None,
                    'best_val_step_any_seed': None,
                    'n_seeds_at_ceiling': 0,
                    'mean_train_time_s': None,
                    'error_analysis': None,
                    'per_seed': [],
                    'is_baseline': (kind == 'baseline'),
                })

    print('\n' + '=' * 112)
    print(f"{'n_kv':>4} | {'ratio':>8} | {'layout':>22} | {'layers':>6} | {'params':>8} | "
          f"{'mean_acc':>10} | {'+/- CI95':>10} | {'t_vs_chance':>12} | {'fwd_ms':>8}")
    print('-' * 112)
    for r in all_results:
        if 'error' in r:
            print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {'(error)':>22} | {r['n_layers']:>6} | "
                  f"{'-':>8} | {'-':>10} | {'-':>10} | {'-':>12} | {'-':>8}   ERROR: {r['error']}")
            continue
        layout_str = r.get('layout', '') or ''
        n_params = r['n_params'] if r.get('n_params') is not None else 0
        mean_acc = r['mean_acc'] if r.get('mean_acc') is not None else 0.0
        ci95 = r['ci95_acc']
        fwd = r['mean_fwd_ms'] if r.get('mean_fwd_ms') is not None else 0.0
        ci_str = f"{ci95:>10.4f}" if ci95 is not None else f"{'n/a':>10}"
        print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {layout_str:>22} | {r['n_layers']:>6} | "
              f"{n_params:>8} | {mean_acc:>10.4f} | {ci_str} | "
              f"{_fmt_tstat(r.get('t_stat_vs_chance'), width=12, prec=2)} | {fwd:>8.2f}")
    print(f"{'':>4} | {'chance':>8} | {'':>22} | {'':>6} | {'':>8} | "
          f"{1.0/VOCAB:>10.4f} | {'':>10} | {'':>12} | {'':>8}")

    # ------------------------------------------------------------------
    # Paired ratio-vs-reference contrasts. The vs-chance tests above only
    # establish that a layout learns the task; they do NOT support
    # "ratio X is better than the default". Since every ratio shares the
    # seed list, the correct comparison is the seed-paired difference vs
    # the 3:1:1 reference under a Bonferroni family over all
    # (alternative ratio, n_kv) comparisons. The sweep is exploratory, so
    # the tests are two-sided against the pre-registered decision rule in
    # RATIO_SWEEP_HYPOTHESES; raw mean ranking is not evidence.
    # ------------------------------------------------------------------
    ratio_contrasts = _ratio_reference_contrasts(all_results)
    ratio_equiv_margin = _parse_equivalence_margin(
        'ABL_EQUIV_MARGIN', CONTRAST_EQUIVALENCE_MARGIN)
    ratio_contrast_tests = _annotate_contrast_significance(
        ratio_contrasts, hypotheses=RATIO_SWEEP_HYPOTHESES,
        equivalence_margin=ratio_equiv_margin)
    # Accuracy-vs-cost screen on the SAME contrasts: a real accuracy
    # difference that costs a disproportionate amount of parameters, train
    # time or inference latency is a different operating point, not a
    # better one. Runs after _annotate_contrast_significance so the
    # significance screen and the cost screen judge the same corrected test.
    _attach_cost_tradeoffs(ratio_contrasts, all_results)
    _log_ratio_contrast_table(ratio_contrasts)

    # Prospective audit of the pre-registered Exp 5 ratio plan. The family
    # is the number of alternative-vs-reference contrasts this run actually
    # produced; it is recorded next to the seed budget so a 7-seed screening
    # sweep is never read as a confirmatory allocation claim. The analyzed
    # population is the intersection of the seeds that survived in every
    # ratio row, so a failed seed downgrades the claim to
    # ``incomplete_attrition`` rather than disappearing from the table.
    ratio_analyzed_population = common_analyzed_seeds(
        all_results, finite_keys=('final_acc', 'final_loss'))
    ratio_experiment_design = audit_experiment_design(
        'exp5_ratio',
        seeds=[42 + i for i in range(n_seeds)],
        levels={'ratio': sorted({str(r.get('ratio')) for r in all_results
                                 if r.get('ratio') is not None})},
        family_size=max(1, len(ratio_contrasts)),
        analyzed_seeds=ratio_analyzed_population,
    )
    logger.info(f'Design audit: {ratio_experiment_design["status"]} '
                f'(requested={ratio_experiment_design["n_seeds"]}, '
                f'analyzed={ratio_experiment_design["n_analyzed"]}, '
                f'family={ratio_experiment_design["planning"]["family_size"]}, '
                f'required='
                f'{ratio_experiment_design["planning"]["required_seeds"]})')
    if ratio_experiment_design['status'] == 'incomplete_attrition':
        logger.warning(
            'Failed seeds were dropped from the ratio sweep; the reported '
            'contrasts are complete-case, not the pre-registered complete '
            'block (see results/exp5_ablation_provenance.json).')

    # Regime of the sweep as a whole. Individual rows already carry their
    # own verdict; this is the one-line answer to "can this ranking be
    # trusted, or is it a budget artifact?".
    ratio_curve_summary = _curve_diagnosis_summary(all_results, 'ratio')
    logger.info('\n' + '=' * 70)
    logger.info('Training-curve regime (held-out monitor, not final numbers):')
    logger.info(f'  rows with a curve: {ratio_curve_summary["rows_with_curve"]}'
                f'/{ratio_curve_summary["rows_total"]}   '
                f'verdicts={ratio_curve_summary["verdict_counts"]}')
    if ratio_curve_summary['mean_generalization_gap'] is not None:
        logger.info(f'  mean generalization gap: '
                    f"{ratio_curve_summary['mean_generalization_gap']:.4f} "
                    f"(max {ratio_curve_summary['max_generalization_gap']:.4f})")
    logger.info(f'  regime: {ratio_curve_summary["regime"]}')
    logger.info(f'  {ratio_curve_summary["interpretation"]}')
    if ratio_curve_summary['recommended_action']:
        logger.info(f'  ACTION: {ratio_curve_summary["recommended_action"]}')
    logger.info('=' * 70)

    logger.info('\nControlled-depth design: all reported ratios contain 5 layers.')
    logger.info('  Parameter counts can still differ by operator type and are reported;')
    logger.info('  interpret ratio effects together with n_params and paired seed data.')

    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    # Include error rows: they carry n_seeds_ok=0, so a fully-failed ratio
    # forces min_seeds_ok=0 and thus conclusions_valid=False (mirrors the
    # run_mechanism_ablation.py convention of taking the min over ALL rows).
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results),
                       default=0)
    _c = 1.0 / VOCAB
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and 0.5 * _c < r['mean_acc'] < 1.5 * _c]
    far_below = [r for r in all_results
                 if 'error' not in r
                 and r.get('mean_acc') is not None
                 and r['mean_acc'] <= 0.5 * _c]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0
                         and len(near_chance) < len(all_results) // 2
                         and len(far_below) == 0)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ratios with vs-chance significant_bonferroni=True: '
                f'{n_any_sig}/{len(all_results)}  (paired-vs-reference '
                f'verdicts are in each row\'s vs_reference_ratio)')
    logger.info(f'  ratios near chance (0.5x-1.5x): {len(near_chance)}/{len(all_results)}')
    logger.info(f'  ratios far below chance (<=0.5x): {len(far_below)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: The ablation results do NOT support strong structural\n'
            '  conclusions. Either the seed count is too low (<5), no layout\n'
            '  reaches Bonferroni significance, or most accuracies are near\n'
            '  chance. Treat the ranking as exploratory, not confirmatory.\n'
            '  To improve power: increase ABL_SEEDS (>=7), increase ABL_STEPS,\n'
            '  or use a simpler task where the signal is stronger.')
    logger.info('=' * 70)
    contrast_by_key = {(c['n_kv'], c['ratio']): c for c in ratio_contrasts}
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds
        r['csa_indexer_normalize_qk'] = True
        r['kda_backend'] = os.environ.get('KDA_BACKEND', 'reference')
        r['is_reference_ratio'] = (str(r.get('ratio'))
                                   == RATIO_REFERENCE_LABEL)
        # Per-row paired standing vs the default ratio (None for the
        # reference row itself and for error stubs).
        if 'error' in r or r['is_reference_ratio']:
            r['vs_reference_ratio'] = None
        else:
            c = contrast_by_key.get((r.get('n_kv'), str(r.get('ratio'))))
            if c is None:
                r['vs_reference_ratio'] = None
            else:
                if c.get('significant_bonferroni') and \
                        c.get('mean_diff') is not None:
                    verdict = ('better_than_reference'
                               if c['mean_diff'] > 0
                               else 'worse_than_reference')
                elif c.get('tost_equivalent') is True:
                    # TOST rejected non-equivalence at the pre-specified
                    # margin: the ratio is practically equivalent to the
                    # default, which "not significant" alone cannot say.
                    verdict = 'equivalent_to_reference'
                else:
                    verdict = 'not_significantly_different'
                r['vs_reference_ratio'] = {
                    'reference_ratio': c['reference_ratio'],
                    'n_common_seeds': c['n_common_seeds'],
                    'mean_diff': c['mean_diff'],
                    't_stat': c['t_stat'],
                    'test': c.get('test'),
                    'alpha_bonferroni': c.get('alpha_bonferroni'),
                    't_crit_bonferroni': c.get('t_crit_bonferroni'),
                    'significant_bonferroni':
                        bool(c.get('significant_bonferroni')),
                    'hypothesis_status': c.get('hypothesis_status'),
                    'equivalence_margin': c.get('equivalence_margin'),
                    'tost_p_value': c.get('tost_p_value'),
                    'tost_equivalent': c.get('tost_equivalent'),
                    'question': c.get('question'),
                    'verdict': verdict,
                }
        r['significance_scope'] = (
            'vs_chance_bonferroni_and_seed_paired_vs_reference_ratio_'
            'bonferroni_and_tost_equivalence')

    os.makedirs('results', exist_ok=True)
    write_results_json(all_results, 'results/exp5_ablation.json',
                       logger=logger)
    try:
        provenance = _run_provenance(n_seeds, steps, gate_record)
        # The paired ratio contrasts + the pre-registered decision rule
        # travel with the run record (the historical rows stay a bare
        # list for backward compatibility with existing consumers).
        provenance['ratio_contrasts'] = ratio_contrasts
        provenance['contrast_tests'] = ratio_contrast_tests
        provenance['hypotheses'] = RATIO_SWEEP_HYPOTHESES
        # Prospective audit against the pre-registered Exp 5 plan: seed
        # budget, actual contrast family and required n at the SESOI.
        provenance['experiment_design'] = ratio_experiment_design
        # Sweep-wide training regime + the accuracy-vs-cost screen, so the
        # artifact records not only which layout won but whether the win
        # survives the curve diagnosis and what it cost.
        provenance['curve_diagnosis_summary'] = ratio_curve_summary
        provenance['cost_tradeoff_policy'] = {
            'min_effect': DEFAULT_MIN_EFFECT,
            'max_cost_ratio': DEFAULT_MAX_COST_RATIO,
            'cost_keys': list(DEFAULT_COST_KEYS),
            'note': (
                'A paired contrast is only an adoptable improvement if its '
                'corrected test is significant AND the accuracy gain clears '
                'min_effect AND no cost axis exceeds max_cost_ratio. The '
                'accuracy difference and the cost ratios are reported '
                'separately so either can be quoted on its own.'),
        }
        write_results_json(provenance,
                           'results/exp5_ablation_provenance.json',
                           logger=logger)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info('\nSaved: results/exp5_ablation.json')
    n_errors = sum(1 for r in all_results if 'error' in r)
    n_incomplete = sum(
        1 for r in all_results
        if 'error' not in r
        and r.get('n_seeds_ok', n_seeds) != r.get('n_seeds', n_seeds)
    )
    rc = 1 if (n_errors or n_incomplete) else 0
    _append_ablation_manifest('ratio', rc, n_seeds, steps, n_kv_list)
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_results)} layouts errored and '
            f'{n_incomplete}/{len(all_results)} have incomplete seed sets. '
            'Returning non-zero to prevent survivor-only aggregates from '
            'being treated as a successful experiment.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
