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
  the cross-route A/D replacement differences) are computed and
  written to ``results/exp5b_ablation_matrix_contrasts.json``.
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
  * **Device awareness.** Runs on GPU (Kaggle T4) when available.
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
    make_seeded_generator, set_global_seed,
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
    _bonferroni_crit_q, _build_param_groups, SMALL_MODEL_SPEC,
)

logger = logging.getLogger(__name__)

VOCAB = 16
SEQ_LEN = 16


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


def _run_provenance(n_seeds, steps):
    """Environment provenance + seed list + training recipe for side files.

    Each side file records the code/environment alongside the seeds and the
    training recipe that produced the adjacent results file, so the pair is
    a complete run record.
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
    return prov


def _eval_model(model, head, embed, seq_len, n_kv=1, device='cpu',
                n_batches=4, batch=64, seed=42):
    was_training = {m: m.training for m in (model, head, embed)}
    try:
        model.eval()
        head.eval()
        embed.eval()
        correct, total = 0, 0
        losses = []
        # Use a distinct, far-separated offset (5_000_000) for the eval RNG so
        # it never collides with the *next* seed's training stream
        # (``seed + 1_000_000``), keeping the eval batches fresh and the
        # seeds independent for the multi-seed t-test. This mirrors the
        # RNG isolation in ``run_quality.py``.
        eval_gen = make_seeded_generator(seed + 5_000_000, device=device)
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, VOCAB, embed, device, generator=eval_gen)
                model.reset_state()
                h = model(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
        if total == 0 or not losses:
            return 0.0, 0.0
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def eval_layout(ratio, d_model=32, seq_len=SEQ_LEN, n_kv=1, steps=100, lr=3e-3, seed=42,
                device='cpu', eval_batches=4, eval_batch=64, train_batch=None):
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

    losses = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(ratio={ratio}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
        opt.zero_grad()
        loss.backward()
        bad_grads = [p for p in params
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grads:
            raise RuntimeError(
                f"non-finite gradient at step {step} in {len(bad_grads)} "
                f"params (ratio={ratio}, seed={seed}); aborting this seed "
                f"to prevent NaN propagation into parameters")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

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
        std_acc = 0.0
        ci_acc = None

    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

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
        'mean_train_time_s': sum(r.get('train_time_s', 0.0) for r in ok_per_seed) / n,
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
               cfg_overrides=None, description=''):
    """Train + evaluate ONE matrix group (mirrors eval_layout's protocol).

    ``layout`` is the abstract layout (with 'memory' tokens); the returned
    row records the RESOLVED layout string, the memory route, and the
    actual parameter count (which differs between KDA and Mamba-2 at the
    same d_model — that residual is exactly what the budget-matching
    follow-up closes).
    """
    if isinstance(device, str):
        device = torch.device(device)
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_matrix_cfg(d_model, memory_type, layout, cfg_overrides)
    total = len(layout)
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

    losses = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(group={group_id}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
        opt.zero_grad()
        loss.backward()
        bad_grads = [p for p in params
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grads:
            raise RuntimeError(
                f"non-finite gradient at step {step} in {len(bad_grads)} "
                f"params (group={group_id}, seed={seed}); aborting this "
                f"seed to prevent NaN propagation into parameters")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

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
    mean_last10 = sum(losses[-10:]) / min(10, len(losses)) if losses else 0.0
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
        'last_train_loss': last_loss,
        'mean_last10_loss': mean_last10,
        'loss_curve': losses,
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
                'train_batch': stub_train_batch,
                'last_train_loss': None, 'mean_last10_loss': None,
                'loss_curve': [],
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
        std_acc = 0.0
        ci_acc = None

    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

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
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
    }


def _seed_accuracy_map(result: dict) -> dict[int, float]:
    """Map seed -> final_acc from a group's per-seed rows (missing = skip)."""
    out = {}
    for row in result.get('per_seed', []):
        acc = row.get('final_acc')
        seed = row.get('seed')
        if acc is None or seed is None or not math.isfinite(acc):
            continue
        out[int(seed)] = float(acc)
    return out


def _paired_contrast(accs_a: dict[int, float], accs_b: dict[int, float]):
    """Paired mean diff + t-stat over the COMMON seeds of two groups.

    Returns ``(mean_diff, t_stat, n)``, or ``(None, None, 0)`` when fewer
    than two common seeds exist. Positive diff = a better than b for a
    higher-is-better metric (final_acc); loss-direction metrics must be
    sign-flipped BEFORE calling this.
    """
    common = sorted(set(accs_a) & set(accs_b))
    if len(common) < 2:
        return None, None, 0
    diffs = [accs_a[s] - accs_b[s] for s in common]
    n = len(diffs)
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    t_stat = (mean_diff / (std / math.sqrt(n))) if std > 0 else None
    return mean_diff, t_stat, n


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
                diff, t_stat, n = _paired_contrast(accs[ga], accs[gb])
                if diff is None:
                    continue
                contrasts.append({
                    'n_kv': n_kv, 'memory': route, 'contrast': name,
                    'mean_diff': diff, 't_stat': t_stat,
                    'n_common_seeds': n,
                    'metric': 'final_acc',
                    'direction': 'higher_is_better',
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
                    var = sum((x - mean_i) ** 2 for x in inter) / (n - 1)
                    t_i = (mean_i / (math.sqrt(var) / math.sqrt(n))) \
                        if var > 0 else None
                    contrasts.append({
                        'n_kv': n_kv, 'memory': route,
                        'contrast': 'I=D-B-C+A (CSA/HCA interaction)',
                        'mean_diff': mean_i, 't_stat': t_i,
                        'n_common_seeds': n,
                        'metric': 'final_acc',
                        'direction': 'higher_is_better',
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
                    diff, t_stat, n = _paired_contrast(
                        _seed_accuracy_map(res_a), _seed_accuracy_map(res_b))
                    if diff is None:
                        continue
                    contrasts.append({
                        'n_kv': n_kv,
                        'memory': f'{ra} vs {rb}',
                        'contrast': f'{g}_{ra.upper()} - {g}_{rb.upper()}',
                        'mean_diff': diff, 't_stat': t_stat,
                        'n_common_seeds': n,
                        'metric': 'final_acc',
                        'direction': 'higher_is_better',
                        'question': question,
                    })
    return contrasts


def _log_contrast_table(contrasts: list) -> None:
    """Log the paired-contrast rows as a readable table."""
    if not contrasts:
        logger.info('  (no paired contrasts computable: need >= 2 common '
                    'seeds across the compared groups)')
        return
    logger.info('\n' + '=' * 92)
    logger.info('Paired memory-route contrasts (seed-matched; final_acc; '
                'positive = first operand better):')
    logger.info('=' * 92)
    for c in contrasts:
        t_str = _fmt_tstat(c.get('t_stat'), width=8, prec=2)
        logger.info(
            f"  n_kv={c['n_kv']:<3} {c['memory']:<12} {c['contrast']:<34} "
            f"diff={c['mean_diff']:+.4f}  t={t_str}  n={c['n_common_seeds']}")
    logger.info(
        '  NOTE: interaction rows (I=D-B-C+A) support complementarity '
        'claims ONLY with statistical evidence;')
    logger.info(
        '  sign-flip loss/PPL metrics before reusing this helper.')
    logger.info('=' * 92)


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

    def paired_mean(a_map: dict, b_map: dict):
        common = sorted(set(a_map) & set(b_map))
        if len(common) < 2:
            return None, len(common)
        diffs = [a_map[s] - b_map[s] for s in common]
        return sum(diffs) / len(diffs), len(common)

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
            expected, n_common = paired_mean(a_map, b_map) \
                if a_map is not None and b_map is not None else (None, 0)
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
            expected, n_common = sum(diffs) / len(diffs), len(common)
        else:
            pair = _LAYOUT_CONTRAST_PAIRS.get(name)
            if pair is None:
                out.append(f'{where}: unrecognized contrast name')
                continue
            ga, gb = pair
            a_map = accs.get((n_kv, memory, f'{memory.upper()}-{ga}'))
            b_map = accs.get((n_kv, memory, f'{memory.upper()}-{gb}'))
            expected, n_common = paired_mean(a_map, b_map) \
                if a_map is not None and b_map is not None else (None, 0)
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
    return out


def _run_matrix_mode(device, n_seeds, steps, n_kv_list):
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
                res = eval_group_multi_seed(spec, n_seeds=n_seeds,
                                            steps=steps, device=device,
                                            n_kv=n_kv, d_model=d_model)
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
                            f"(std={res['std_acc']:.4f}, "
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
        r['significance_scope'] = 'vs_chance_baseline_not_pairwise_between_groups'

    # Paired seed-matched contrasts (B-A / C-A / D-B / D-C / interaction
    # per route; A/D replacement across routes — the KDA-vs-RWKV-7
    # replacement question) — written to their own side file so the
    # per-group rows' schema stays unchanged for existing consumers.
    contrasts = _matrix_contrast_report(all_results)
    _log_contrast_table(contrasts)

    os.makedirs('results', exist_ok=True)
    run_meta = {
        'preset': preset,
        'memory_routes': list(memory_types),
        'n_kv_list': list(n_kv_list),
        'n_seeds': n_seeds,
        'steps': steps,
        'total_layers': total_layers,
        'd_model': d_model,
        'device': str(device),
        # The preset's promised group ids; the row contract refuses to
        # write a survivor-only matrix (a crash mid-sweep) as if complete.
        'expected_groups': [spec[0] for spec in groups],
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
        write_results_json(_run_provenance(n_seeds, steps),
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
    logger.info(f'  n_seeds={n_seeds}, steps={steps}, n_kv={n_kv_list}')

    if abl_mode == 'matrix':
        return _run_matrix_mode(device, n_seeds, steps, n_kv_list)

    ratios = [
        (3, 1, 1),
        (4, 1, 0), (4, 0, 1),
        (2, 2, 1), (2, 1, 2),
        (1, 3, 1), (1, 1, 3),
    ]
    if any(sum(r) != 5 for r in ratios):
        raise AssertionError('ablation layouts must all have total depth 5')
    n_tests = len(ratios) * len(n_kv_list)
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
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for r in ratios:
            logger.info(f'\n-- ratio KDA:CSA:HCA = {r[0]}:{r[1]}:{r[2]} '
                        f'(n_kv={n_kv}, {n_seeds} seeds) --')
            try:
                res = eval_layout_multi_seed(r, n_seeds=n_seeds, steps=steps,
                                             device=device, n_kv=n_kv)
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
                logger.info(f"  layout={res['layout']}  n_params={res['n_params']}  n_layers={res['n_layers']}")
                _ci = res['ci95_acc']
                _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
                logger.info(f"  -> mean_acc={res['mean_acc']:.4f} +/- {_ci_str} "
                            f"(std={res['std_acc']:.4f}, t_vs_chance={_fmt_tstat(res['t_stat_vs_chance'], width=0, prec=2)}, "
                            f"sig_bonferroni={res['significant_bonferroni']})")
                logger.info(f"     mean_fwd={res['mean_fwd_ms']:.2f}ms")
            except Exception as e:
                import traceback as _tb
                logger.error(f"  ratio {r[0]}:{r[1]}:{r[2]} FAILED: {e}")
                _tb.print_exc()
                all_results.append({
                    'ratio': f'{r[0]}:{r[1]}:{r[2]}',
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
                    'n_layers': sum(r),
                    'chance_acc': 1.0 / VOCAB,
                    't_stat_vs_chance': None,
                    't_crit_bonferroni': None,
                    'significant_bonferroni': False,
                    'mean_train_time_s': None,
                    'per_seed': [],
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
    logger.info(f'  ratios with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
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
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds
        r['csa_indexer_normalize_qk'] = True
        r['kda_backend'] = os.environ.get('KDA_BACKEND', 'reference')
        r['significance_scope'] = 'vs_chance_baseline_not_pairwise_between_layouts'

    os.makedirs('results', exist_ok=True)
    write_results_json(all_results, 'results/exp5_ablation.json',
                       logger=logger)
    try:
        write_results_json(_run_provenance(n_seeds, steps),
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
