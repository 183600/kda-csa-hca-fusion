#!/usr/bin/env python3
"""Experiment 14 — training-step roofline under gradient-checkpointing
policies, plus the dense multi-query KV-budget ladder.

Three questions, one table family:

1. **Gradient checkpointing** (``grad_checkpointing``): for each policy
   (``none`` / ``all`` / ``memory`` / ``retrieval``) on the REAL hybrid
   stack, what does the autograd graph retain for backward (measured with
   ``saved_tensors_hooks``), does the policy preserve values and
   gradients exactly, and what does the extra forward pass cost?
2. **Roofline** (``roofline`` / ``run_kv_cache.prefill_flops``): combine
   the analytic training FLOPs (forward + 2x backward + one recompute
   forward per checkpointed layer) with the compulsory training bytes
   (Adam state + gradients + parameters + measured retained activations +
   input batch) into an arithmetic-intensity point and a
   compute/memory-bound verdict against the host's measured (or preset)
   peaks.
3. **Dense multi-query attention** (``mqa_attention``): the softmax
   ladder's KV budget across MHA / GQA / MQA at a fixed
   ``(n_heads, head_dim)`` — prefill FLOPs are unchanged, decode traffic
   divides by the sharing factor — placed next to the hybrid memory
   routes' per-token decode state so the "why O(1) state" story has a
   quantitative softmax reference.

Usage::

    python3 run_train_roofline.py
    python3 run_train_roofline.py --T 128 --batch 4 --skip-quality \
        --json results/exp14_train_roofline.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import grad_checkpointing as gc  # noqa: E402
import mqa_attention as mqa  # noqa: E402
import results_schema  # noqa: E402
import roofline  # noqa: E402
from kaggle_setup import (append_run_manifest,  # noqa: E402
                          capture_provenance,
                          configure_torch_for_device,
                          make_run_manifest_entry, sanitize_for_json,
                          set_global_seed, write_results_json)
from multi_seed_stats import summarize_paired  # noqa: E402
from ops_fused import HybridAttentionStack  # noqa: E402
from run_decode_bandwidth import build_base_cfg  # noqa: E402
from run_kv_cache import prefill_flops  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp14_train_roofline.json')
KIND = 'exp14_train_roofline'

#: MQA/GQA/MHA variant ladder. The mapping's values are the historical
#: counts for ``n_heads == 4``; the actual counts are derived from the
#: configured ``n_heads`` by :func:`resolve_mqa_variants`, so the labels stay
#: correct for every head count. The dict shape is kept for callers that
#: imported this constant.
MQA_VARIANTS = {'mha': 4, 'gqa': 2, 'mqa': 1}


def resolve_mqa_variants(n_heads: int) -> list[tuple[str, int]]:
    """Resolve the variant ladder for ``n_heads`` (duplicates skipped).

    ``mha`` keeps every query head's own K/V head, ``gqa`` halves the KV
    head count (rounded down but at least one), and ``mqa`` shares one KV
    head. For ``n_heads == 4`` this is exactly the historical
    ``mha=4, gqa=2, mqa=1`` table; for ``n_heads == 2`` the GQA and MQA
    rows coincide, so only the first (GQA) label is emitted.
    """
    if isinstance(n_heads, bool) or not isinstance(n_heads, int)             or n_heads < 1:
        raise ValueError(
            f'n_heads={n_heads!r} must be a positive int')
    candidates = (('mha', n_heads),
                  ('gqa', max(1, n_heads // 2)),
                  ('mqa', 1))[:len(MQA_VARIANTS)]
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, n_kv in candidates:
        if n_heads % n_kv != 0 or n_kv in seen:
            continue
        seen.add(n_kv)
        out.append((name, n_kv))
    return out


def parse_int_list(raw: str, *, minimum: int = 0) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            raise SystemExit(f'expected an int list, got {part!r}')
        if v < minimum:
            raise SystemExit(f'{part!r} must be >= {minimum}')
        values.append(v)
    if not values:
        raise SystemExit('at least one value required')
    return tuple(dict.fromkeys(values))


def resolve_peaks(device, *, bw_gbps: float | None = None,
                  tflops: float | None = None) -> tuple[dict, str]:
    """Resolve machine peaks from explicit values, measurement, or presets.

    Returns ``(peaks, source)`` where ``peaks`` may have
    ``peak_bw_gbps`` and/or ``peak_tflops`` set to ``None``; the roofline
    fields that need a missing peak are omitted rather than guessed.
    """
    peaks: dict[str, float | None] = {'peak_bw_gbps': None,
                                      'peak_tflops': None}
    sources: list[str] = []
    if bw_gbps is not None:
        peaks['peak_bw_gbps'] = float(bw_gbps)
        sources.append('bw=explicit')
    else:
        measured = roofline.measure_copy_bandwidth_gbps(device)
        if measured is not None:
            peaks['peak_bw_gbps'] = measured
            sources.append('bw=measured_on_host')
        else:
            name = ''
            try:
                if str(device).startswith('cuda'):
                    name = torch.cuda.get_device_name(device)
            except Exception:
                name = ''
            spec = roofline.spec_for_device(name)
            if spec is not None:
                peaks['peak_bw_gbps'] = spec['peak_bw_gbps']
                sources.append(f"bw=preset:{spec['preset_key']}")
            else:
                sources.append('bw=unavailable')
    if tflops is not None:
        peaks['peak_tflops'] = float(tflops)
        sources.append('tflops=explicit')
    else:
        measured = roofline.measure_gemm_tflops(device)
        if measured is not None:
            peaks['peak_tflops'] = measured
            sources.append('tflops=measured_on_host')
        else:
            name = ''
            try:
                if str(device).startswith('cuda'):
                    name = torch.cuda.get_device_name(device)
            except Exception:
                name = ''
            spec = roofline.spec_for_device(name)
            if spec is not None:
                peaks['peak_tflops'] = spec['peak_tflops']
                sources.append(f"tflops=preset:{spec['preset_key']}")
            else:
                sources.append('tflops=unavailable')
    return peaks, '; '.join(sources)


def hybrid_geometry_kwargs(cfg) -> dict:
    """Map a ``HybridConfig`` onto ``run_kv_cache.prefill_flops`` kwargs."""
    return dict(
        d=cfg.d_model, H=cfg.n_heads_qk, K=cfg.head_dim_k, V=cfg.head_dim_v,
        kda_hv=cfg.n_heads_v, kda_k=cfg.head_dim_k, kda_v=cfg.head_dim_v,
        kda_conv_ksize=3, kda_chunk_size=cfg.kda_chunk_size,
        csa_m=cfg.csa_m, csa_topk=cfg.csa_topk, csa_c=cfg.csa_c,
        csa_dc=cfg.csa_dc, csa_cI=cfg.csa_cI, csa_nIh=cfg.csa_nIh,
        csa_nh=cfg.csa_nh, csa_sliding_window=cfg.csa_sliding_window,
        hca_m2=cfg.hca_m2, hca_c=cfg.hca_c, hca_dc=cfg.hca_dc,
        hca_nh=cfg.hca_nh, hca_sliding_window=cfg.hca_sliding_window,
        mamba_d_state=cfg.mamba_d_state, mamba_d_conv=cfg.mamba_d_conv,
        mamba_expand=cfg.mamba_expand, mamba_headdim=cfg.mamba_headdim,
        mamba_ngroups=cfg.mamba_ngroups, mamba_dt_rank=cfg.mamba_dt_rank,
        rwkv_head_dim=cfg.rwkv_head_dim, rwkv_decay_lora=cfg.rwkv_decay_lora,
        rwkv_a_lora=cfg.rwkv_a_lora, rwkv_gate_lora=cfg.rwkv_gate_lora,
    )


def _finite_flops(name: str, value) -> float:
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f'training_flops_model: {name}={value!r} must be representable '
            f'as a finite float.') from None
    if not math.isfinite(number):
        raise ValueError(
            f'training_flops_model: {name}={value!r} must be representable '
            f'as a finite float.')
    return number


def training_flops_model(cfg, policy: str, *, T: int) -> dict:
    """Analytic training FLOPs for one checkpointing policy.

    Forward per layer from ``run_kv_cache.prefill_flops`` (the repo's
    operator-boundary accounting); backward = 2x forward; each
    checkpointed layer adds one forward during recompute. The model is a
    first-order work estimate, not a latency prediction.
    """
    policy = gc.parse_policy(policy)
    layout = _layout_for(cfg)
    geo = hybrid_geometry_kwargs(cfg)
    fwd_by_kind: dict[str, float] = {}
    for kind in layout:
        fwd_by_kind[kind] = _finite_flops(
            f'prefill FLOPs ({kind})', prefill_flops(kind, T, **geo))
    # Sum over the LAYOUT (one term per layer), not over the deduplicated
    # kind dictionary: repeated memory layers (e.g. the default 3xKDA)
    # each contribute their own forward FLOPs.
    forward_total = _finite_flops(
        'forward FLOPs', sum(fwd_by_kind[kind] for kind in layout))
    checked = [k for k in layout if gc.should_checkpoint(k, policy)]
    recompute = _finite_flops(
        'recompute FLOPs', sum(fwd_by_kind[k] for k in checked))
    backward = _finite_flops('backward FLOPs', 2.0 * forward_total)
    total = _finite_flops(
        'training FLOPs', forward_total + backward + recompute)
    return {
        'policy': policy,
        'n_layers': len(layout),
        'n_layers_checked': len(checked),
        'forward_flops': forward_total,
        'backward_flops': backward,
        'recompute_flops': recompute,
        'training_flops': total,
        'flops_relative_to_forward_total': total / forward_total
        if forward_total else 1.0,
    }


def _layout_for(cfg) -> list[str]:
    """Resolved layout without building (and seeding) a stack."""
    unit = (cfg.layout if cfg.layout is not None else
            [cfg.memory_type] * cfg.n_kda + ['csa'] * cfg.n_csa
            + ['hca'] * cfg.n_hca)
    return list(unit)


def probe_policies(cfg, *, T: int, batch: int, policies: tuple[str, ...],
                   seed: int, device, timing_reps: int = 3) -> list[dict]:
    """Run the checkpointing policies on a real stack and audit them.

    For every policy: value parity vs ``forward_functional``, parameter
    gradient parity, retained activation bytes (probe), and a min-of-reps
    forward+backward wall time. The analytic overhead model is attached.
    """
    set_global_seed(seed, deterministic=True)
    torch.manual_seed(seed)
    layout = _layout_for(cfg)
    stack = HybridAttentionStack(cfg, total_layers=len(layout)).to(device)
    x = torch.randn(batch, T, cfg.d_model, device=device)
    n_params = int(sum(p.numel() for p in stack.parameters()))

    # Baseline: the canonical functional forward + backward.
    stack.zero_grad(set_to_none=True)
    out0, state0, _ = stack.forward_functional(
        x, None, [None] * stack.n_memory_layers)
    out0.pow(2).mean().backward()
    grads0 = {name: p.grad.detach().clone()
              for name, p in stack.named_parameters()}

    rows: list[dict] = []
    for policy in policies:
        row = gc.checkpoint_overhead_model(layout, policy)
        # 1) measured retained bytes (probe active during forward only).
        stack.zero_grad(set_to_none=True)
        probe = gc.SavedBytesProbe()
        with probe.measure():
            out, state, _ = gc.forward_with_policy(
                stack, x, policy=policy, probe=probe)
        row['saved_activation_bytes'] = probe.total_bytes()
        row['saved_activation_tensors'] = probe.total_tensors()
        row['saved_bytes_by_kind'] = probe.bytes_by_tag()
        row['retained_floor_bytes'] = gc.minimum_retained_bytes(
            row['n_layers'], batch, T, cfg.d_model)
        row['output_max_abs_diff_vs_baseline'] = float(
            (out - out0).abs().max().item())
        row['state_max_abs_diff_vs_baseline'] = float(
            (state - state0).abs().max().item())
        # 2) gradient parity on the same graph.
        out.pow(2).mean().backward()
        row['grad_max_abs_diff_vs_baseline'] = max(
            float((p.grad - grads0[name]).abs().max().item())
            for name, p in stack.named_parameters())
        # 3) wall time (probe-free, min of repeats).
        times = []
        for _ in range(max(1, timing_reps)):
            stack.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            out_t, _, _ = gc.forward_with_policy(stack, x, policy=policy)
            out_t.pow(2).mean().backward()
            times.append((time.perf_counter() - t0) * 1e3)
        row['fwd_bwd_ms_min'] = float(min(times))
        row['fwd_bwd_ms_median'] = float(statistics.median(times))
        rows.append(row)
    return rows, n_params


def assemble_roofline(rows: list[dict], flops_models: list[dict], *,
                      n_params: int, batch: int, T: int, d_model: int,
                      itemsize: int, peaks: dict, optimizer: str = 'adam',
                      ) -> list[dict]:
    """Merge measured bytes + analytic FLOPs into roofline points.

    Compulsory training bytes: optimizer state + gradients + parameters
    (Adam: ``4`` copies of the parameter tensor per element; SGD: ``2``),
    the measured retained activations, and one input batch. Compute/memory
    verdicts use the resolved peaks; a missing peak omits its fields.
    """
    if optimizer == 'adam':
        state_copies = 2  # first/second moments
    elif optimizer == 'sgd':
        state_copies = 0
    else:
        raise ValueError(f"optimizer={optimizer!r} must be 'adam' or 'sgd'.")
    param_bytes = n_params * itemsize
    opt_grad_bytes = param_bytes * (1 + 1 + state_copies)
    input_bytes = batch * T * d_model * itemsize
    peak_tflops = peaks.get('peak_tflops')
    peak_bw = peaks.get('peak_bw_gbps')
    out: list[dict] = []
    for row, model in zip(rows, flops_models):
        if row['policy'] != model['policy']:
            raise ValueError('policy rows and flops models are misaligned.')
        activation_bytes = int(row['saved_activation_bytes'])
        total_bytes = opt_grad_bytes + activation_bytes + input_bytes
        flops = float(model['training_flops'])
        ai = roofline.arithmetic_intensity(flops, total_bytes)
        point = {
            'policy': row['policy'],
            'training_flops': flops,
            'parameter_bytes': int(param_bytes),
            'optimizer_grad_bytes': int(opt_grad_bytes),
            'activation_bytes_retained': activation_bytes,
            'input_batch_bytes': int(input_bytes),
            'training_bytes': int(total_bytes),
            'arithmetic_intensity_flops_per_byte': ai,
            'activation_share': activation_bytes / total_bytes,
        }
        if peak_tflops and peak_bw:
            ridge = roofline.ridge_point(peak_tflops * 1e12,
                                         peak_bw)
            point['ridge_point_flops_per_byte'] = ridge
            point['roofline_bound'] = roofline.bound_for(
                ai, peak_tflops * 1e12, peak_bw)
            compute_ms = flops / (peak_tflops * 1e12) * 1e3
            memory_ms = total_bytes / (peak_bw * 1e9) * 1e3
            point['compute_bound_ms'] = compute_ms
            point['memory_bound_ms'] = memory_ms
            point['roofline_ceiling_ms'] = max(compute_ms, memory_ms)
            point['efficiency_label'] = roofline.efficiency_label(
                point['roofline_ceiling_ms'] / row['fwd_bwd_ms_min'] * 100.0
                if row['fwd_bwd_ms_min'] > 0 else None)
        out.append(point)
    return out


def mqa_variant_table(*, d_model: int, n_heads: int, head_dim: int, T: int,
                      itemsize: int, batch: int = 1) -> list[dict]:
    """MHA / GQA / MQA KV budget at fixed ``(n_heads, head_dim)``."""
    rows: list[dict] = []
    for name, n_kv in resolve_mqa_variants(n_heads):
        params = mqa.attention_parameter_elements(
            d_model, n_heads, n_kv, head_dim)
        kv_per_token = mqa.kv_state_elements_per_token(n_kv, head_dim)
        rows.append({
            'variant': name,
            'n_heads': n_heads,
            'n_kv_heads': n_kv,
            'head_dim': head_dim,
            'group_size': n_heads // n_kv,
            'parameter_elements': params,
            'prefill_flops': mqa.attention_prefill_flops(
                T, d_model, n_heads, n_kv, head_dim),
            'kv_elements_per_token': kv_per_token,
            'kv_bytes_per_token': kv_per_token * itemsize,
            'kv_read_bytes_at_T': mqa.attention_kv_read_elements(
                T, n_kv, head_dim, batch=batch) * itemsize,
            'kv_read_write_bytes_at_T': (T + 1) * kv_per_token * itemsize * batch,
            'kv_reduction_vs_mha': mqa.kv_bytes_reduction(n_heads, n_kv),
        })
    return rows


def hybrid_decode_reference(cfg, *, T: int, itemsize: int,
                            routes: tuple[str, ...] = ('kda', 'mamba2',
                                                       'rwkv7')) -> list[dict]:
    """Per-token decode state bytes of the hybrid memory routes (Exp 13)."""
    import decode_bandwidth as db
    table = db.route_switch_table(cfg, T, batches=(1,), routes=routes,
                                  itemsize=itemsize)
    out = []
    for row in table:
        if row['status'] != 'ok':
            out.append({'route': row['route'], 'status': 'unavailable',
                        'reason': row.get('reason')})
            continue
        out.append({
            'route': row['route'],
            'status': 'ok',
            'state_bytes_per_token': row['state_bytes_per_token'],
            'weights_state_switch_batch': row['weights_state_switch_batch'],
            'read_bytes_by_op_per_token': row['read_bytes_by_op_per_token'],
        })
    return out


def _train_variant(name: str, n_kv_heads: int, *, seed: int, device,
                   d_model: int, n_heads: int, head_dim: int, seq_len: int,
                   vocab: int, steps: int, lr: float, batch: int,
                   eval_batches: int, eval_batch: int) -> dict:
    """Tiny protocol-identical training run of one MQA variant (multi_agg).

    Same recipe for every variant (Adam, fixed LR/steps/batch; model init
    and batch stream both seeded) so the accuracy difference reflects the
    KV-head count, not the budget.
    """
    from run_quality import MQARHead, ResidualAttnLayer, make_multi_agg_batch
    # Seed Python/NumPy/Torch/CUDA and request deterministic kernels. This
    # probe is not used for the timing rows, so the deterministic path does
    # not bias the roofline measurement.
    set_global_seed(seed, deterministic=True)
    embed = torch.nn.Embedding(vocab, d_model).to(device)
    attn = mqa.MQAAttn(d_model, n_heads, n_kv_heads, head_dim).to(device)
    layer = ResidualAttnLayer(attn, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    params = [*embed.parameters(), *layer.parameters(), *head.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    train_gen = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(steps):
        x, target, cue = make_multi_agg_batch(
            batch, seq_len, 1, vocab, embed, 'cpu', generator=train_gen)
        if device != 'cpu':
            x, target = x.to(device), target.to(device)
        logits = head(layer(x), cue)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    eval_gen = torch.Generator(device='cpu').manual_seed(seed + 10_000)
    correct = total = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            x, target, cue = make_multi_agg_batch(
                eval_batch, seq_len, 1, vocab, embed, 'cpu',
                generator=eval_gen)
            if device != 'cpu':
                x, target = x.to(device), target.to(device)
            pred = head(layer(x), cue).argmax(dim=-1)
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return {
        'variant': name,
        'seed': seed,
        'final_acc': correct / total if total else float('nan'),
        'n_params': int(sum(p.numel() for p in params)),
    }


def quality_probe(*, seeds: tuple[int, ...], device, d_model: int,
                  n_heads: int, head_dim: int, seq_len: int, vocab: int,
                  steps: int, lr: float, batch: int,
                  eval_batches: int, eval_batch: int) -> list[dict]:
    runs: list[dict] = []
    variants = resolve_mqa_variants(n_heads)
    for name, n_kv in variants:
        for seed in seeds:
            runs.append(_train_variant(
                name, n_kv, seed=seed, device=device, d_model=d_model,
                n_heads=n_heads, head_dim=head_dim, seq_len=seq_len,
                vocab=vocab, steps=steps, lr=lr, batch=batch,
                eval_batches=eval_batches, eval_batch=eval_batch))
    # Per-variant, per-seed accuracies so the MQA/GQA/MHA contrast can be
    # seed-paired (the same recipe and seeds are used for every variant).
    by_variant: dict[str, dict[int, float]] = {}
    for run in runs:
        by_variant.setdefault(run['variant'], {})[run['seed']] = \
            run['final_acc']
    ref = by_variant.get('mha', {})
    summary: list[dict] = []
    for name, _n_kv in variants:
        seed_accs = by_variant.get(name)
        if not seed_accs:
            continue
        seeds = sorted(seed_accs)
        accs = [seed_accs[s] for s in seeds]
        mean = statistics.fmean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else None
        ci = None
        if std is not None:
            from run_quality import _t_crit_975
            crit = _t_crit_975(len(accs))
            if crit is not None:
                half = crit * std / (len(accs) ** 0.5)
                ci = [mean - half, mean + half]
        common = sorted(set(seed_accs) & set(ref))
        paired = None
        if name != 'mha' and len(common) >= 2:
            paired = summarize_paired(
                [seed_accs[s] - ref[s] for s in common])
            # Two non-baseline variants (GQA, MQA) are contrasted with MHA
            # inside each probe; apply a Bonferroni adjustment so the
            # family-wise error stays at 0.05.
            family = max(1, len(variants) - 1)
            p_raw = (paired.get('t_test') or {}).get('p_value_two_sided')
            if p_raw is not None:
                p_adj = min(1.0, p_raw * family)
                paired['p_value_bonferroni'] = p_adj
                paired['alpha_bonferroni'] = 0.05 / family
                paired['significant_bonferroni'] = bool(p_adj < 0.05)
                paired['bonferroni_family_size'] = family
        summary.append({
            'variant': name,
            'seeds': seeds,
            'accs': accs,
            'mean_acc': mean,
            # Sample standard deviation (n - 1), not the population
            # statistic: these seeds are a sample from the run
            # distribution, and the paired contrast uses the same units.
            'std_acc': std,
            'ci95_acc': ci,
            'min_acc': min(accs),
            'max_acc': max(accs),
            'n_seeds': len(accs),
            'n_params': next(r['n_params'] for r in runs
                             if r['variant'] == name),
            'paired_vs_mha': paired,
        })
    return summary


def build_report(*, T: int, batch: int, d_model: int, seed: int,
                 policies: tuple[str, ...], itemsize: int, device,
                 peaks: dict, peak_source: str, quality_seeds: tuple[int, ...],
                 quality_steps: int, quality_lr: float, skip_quality: bool,
                 timing_reps: int = 3) -> dict:
    """Assemble the JSON-safe Exp 14 payload (pure builder)."""
    if d_model < 4 or d_model % 4 != 0:
        raise ValueError(
            f"d_model={d_model} must be a multiple of 4 (the MQA ladder "
            f"uses n_heads=4 with head_dim=d_model//4).")
    if T < 1 or batch < 1:
        raise ValueError(f"T={T} and batch={batch} must be >= 1.")
    cfg = build_base_cfg(d_model)
    rows, n_params = probe_policies(
        cfg, T=T, batch=batch, policies=policies, seed=seed, device=device,
        timing_reps=timing_reps)
    flops_models = [training_flops_model(cfg, p, T=T) for p in policies]
    roofline_rows = assemble_roofline(
        rows, flops_models, n_params=n_params, batch=batch, T=T,
        d_model=d_model, itemsize=itemsize, peaks=peaks)
    mqa_rows = mqa_variant_table(
        d_model=d_model, n_heads=4, head_dim=d_model // 4, T=T,
        itemsize=itemsize, batch=1)
    hybrid_ref = hybrid_decode_reference(cfg, T=T, itemsize=itemsize)
    quality: dict = {'status': 'skipped'}
    if not skip_quality:
        probes = []
        for vocab in (4, 16):
            summary = quality_probe(
                seeds=quality_seeds,
                device='cpu' if device == 'cpu' else device,
                d_model=d_model, n_heads=4, head_dim=d_model // 4,
                seq_len=33, vocab=vocab, steps=quality_steps, lr=quality_lr,
                batch=32, eval_batches=4, eval_batch=64)
            by_variant = {r['variant']: r for r in mqa_rows}
            for row in summary:
                kv_bytes = by_variant[row['variant']]['kv_bytes_per_token']
                row['kv_bytes_per_token'] = kv_bytes
                row['acc_per_kv_kb'] = (
                    row['mean_acc'] / (kv_bytes / 1024.0))
            probes.append({'vocab': vocab, 'chance_acc': 1.0 / vocab,
                           'variants': summary})
        quality = {'status': 'ok', 'task': 'multi_agg',
                   'seeds': list(quality_seeds), 'steps': quality_steps,
                   'lr': quality_lr, 'probes': probes}
    return {
        'kind': KIND,
        'method': (
            'training-step memory/compute accounting under gradient-'
            'checkpointing policies on a real HybridAttentionStack: '
            'retained-activation bytes measured via saved_tensors_hooks; '
            'analytic training FLOPs = fwd + 2x bwd + one recompute fwd per '
            'checkpointed layer; roofline verdicts against resolved peaks; '
            'MHA/GQA/MQA KV ladder from mqa_attention'),
        'config': {
            'T': int(T),
            'batch': int(batch),
            'd_model': int(d_model),
            'seed': int(seed),
            'itemsize': int(itemsize),
            'policies': list(policies),
            'layout_unit': list(_layout_for(cfg)),
            'n_parameters_stack': n_params,
            'n_layers_stack': len(_layout_for(cfg)),
            'peaks_gbps': peaks.get('peak_bw_gbps'),
            'peaks_tflops': peaks.get('peak_tflops'),
            'peak_source': peak_source,
        },
        'checkpoint_policies': rows,
        'training_roofline': roofline_rows,
        'mqa_variants': mqa_rows,
        'hybrid_decode_reference': hybrid_ref,
        'quality_probe': quality,
        'notes': [
            'retained bytes = tensors saved for backward during the forward '
            '(saved_tensors_hooks); the checkpoint recompute traffic during '
            'backward is deliberately not counted',
            'gradient/value parity is a hard requirement: both diffs must be '
            '~0 for every policy (checkpointing may not change results)',
            'training FLOPs are the repo prefill_flops boundary scaled by '
            '3 (fwd+2xbwd) plus 1 extra forward per checkpointed layer; '
            'first-order work estimate, not measured',
            'training bytes are compulsory-only (Adam 4x params: param + '
            'grad + 2 moments, retained activations, one input batch); '
            'transient kernel traffic is excluded',
            'MQA/GQA/MQA at fixed n_heads: prefill FLOPs are identical, '
            'decode KV traffic divides by the sharing factor',
            'quality probe: multi_agg at vocab=4 (easy; variants saturate '
            'near 1.0 — the KV sharing is free there) and vocab=16 (hard; '
            'exposes the sharing-quality cost), same recipe per variant and '
            '3 seeds; std_acc is the sample standard deviation, ci95_acc is '
            'a two-sided t-interval, and paired_vs_mha is a seed-paired '
            'contrast with a Bonferroni-corrected p across the two '
            'non-baseline variants in each probe (diagnostic probe; the '
            '3-seed power is low, so non-significance is not equivalence)',
            'efficiency_label relates the roofline ceiling to the measured '
            'fwd+bwd time of the same policy; at this small scale python '
            'overhead dominates, so the achieved fraction is a lower bound',
        ],
        'provenance': capture_provenance(),
    }


def report_envelope(report: dict) -> dict:
    """Wrap the flat Exp 14 report in the repo's versioned envelope.

    The per-policy checkpoint rows become ``results`` (the grid most
    likely to be plotted); every other section rides in ``metadata``.
    ``results_schema`` then accepts the file (unknown kind -> no row
    contract yet) and future tooling can migrate it.
    """
    metadata = {k: report[k] for k in report
                if k not in ('kind', 'checkpoint_policies')}
    return results_schema.envelope(
        list(report['checkpoint_policies']), kind=report['kind'],
        metadata=metadata)


def _print_report(report: dict) -> None:
    cfg = report['config']
    print(f"[exp14] T={cfg['T']} batch={cfg['batch']} d_model={cfg['d_model']}"
          f" layers={cfg['n_layers_stack']} params={cfg['n_parameters_stack']:,}")
    print(f"[exp14] peaks: bw={cfg['peaks_gbps']} GB/s "
          f"tflops={cfg['peaks_tflops']} ({cfg['peak_source']})")
    print('[exp14] checkpointing policies (measured on the real stack):')
    for row in report['checkpoint_policies']:
        print(f"  {row['policy']:<10} checked="
              f"{row['n_layers_checked']}/{row['n_layers']} "
              f"saved={row['saved_activation_bytes']:>10,}B "
              f"floor={row['retained_floor_bytes']:>9,}B "
              f"out-diff={row['output_max_abs_diff_vs_baseline']:.1e} "
              f"grad-diff={row['grad_max_abs_diff_vs_baseline']:.1e} "
              f"fwd+bwd={row['fwd_bwd_ms_min']:.1f}ms")
    print('[exp14] training roofline:')
    for row in report['training_roofline']:
        extra = ''
        if 'roofline_bound' in row:
            extra = (f" bound={row['roofline_bound']} "
                     f"ceiling={row['roofline_ceiling_ms']:.2f}ms "
                     f"ai={row['arithmetic_intensity_flops_per_byte']:.2f}")
        print(f"  {row['policy']:<10} flops={row['training_flops']:>12.3e} "
              f"bytes={row['training_bytes']:>10,} "
              f"act-share={row['activation_share']:.3f}{extra}")
    print('[exp14] dense MQA KV ladder (d_model='
          f"{report['config']['d_model']}):")
    for row in report['mqa_variants']:
        print(f"  {row['variant']:<4} kv_heads={row['n_kv_heads']} "
              f"params={row['parameter_elements']:>6,} "
              f"prefill_flops={row['prefill_flops']:.3e} "
              f"kv={row['kv_bytes_per_token']:>4}B/token "
              f"read@T={row['kv_read_bytes_at_T']:>9,}B "
              f"x{row['kv_reduction_vs_mha']:.1f} vs MHA")
    for row in report['hybrid_decode_reference']:
        if row['status'] == 'ok':
            print(f"  hybrid {row['route']:<7} state="
                  f"{row['state_bytes_per_token']:>10,.1f}B/token "
                  f"switch@B={row['weights_state_switch_batch']:.2f}")
        else:
            print(f"  hybrid {row['route']:<7} unavailable")
    q = report['quality_probe']
    if q['status'] == 'ok':
        print('[exp14] multi_agg quality probe (same recipe per variant):')
        for frame in q['probes']:
            print(f"  vocab={frame['vocab']} (chance={frame['chance_acc']:.2f}):")
            for row in frame['variants']:
                std = row.get('std_acc')
                std_str = f'{std:.3f}' if std is not None else 'n/a'
                paired = row.get('paired_vs_mha') or {}
                p = paired.get('p_value_bonferroni')
                p_str = f'{p:.3g}' if p is not None else 'n/a'
                print(f"    {row['variant']:<4} acc={row['mean_acc']:.3f} "
                      f"+- {std_str} "
                      f"kv={row['kv_bytes_per_token']}B/token "
                      f"acc/KB={row['acc_per_kv_kb']:.3f} "
                      f"p_bonf_vs_mha={p_str}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Exp 14 — training roofline under gradient-checkpointing '
                    'policies + dense MQA KV ladder.')
    parser.add_argument('--T', type=int, default=64,
                        help='sequence length for the training step '
                             '(default 64)')
    parser.add_argument('--batch', type=int, default=2,
                        help='training batch size (default 2)')
    parser.add_argument('--d-model', type=int, default=32,
                        help='model width (default 32)')
    parser.add_argument('--seed', type=int, default=0,
                        help='stack-init / input seed (default 0)')
    parser.add_argument('--policies', default=','.join(gc.CHECKPOINT_POLICIES),
                        help='comma-separated checkpointing policies')
    parser.add_argument('--itemsize', type=int, default=4,
                        help='bytes per element (default 4 = fp32)')
    parser.add_argument('--timing-reps', type=int, default=3,
                        help='fwd+bwd timing repeats; min is reported')
    parser.add_argument('--bw-gbps', type=float, default=None,
                        help='override the measured host bandwidth')
    parser.add_argument('--tflops', type=float, default=None,
                        help='override the measured host GEMM rate')
    parser.add_argument('--quality-seeds', default='0,1,2',
                        help='seeds for the MQA quality probe')
    parser.add_argument('--quality-steps', type=int, default=150,
                        help='training steps for the quality probe')
    parser.add_argument('--quality-lr', type=float, default=3e-3,
                        help='learning rate for the quality probe')
    parser.add_argument('--skip-quality', action='store_true',
                        help='skip the MQA multi_agg quality probe')
    parser.add_argument('--json', default=RESULTS_PATH,
                        help=f'output path (default {RESULTS_PATH})')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    policies = tuple(p.strip() for p in args.policies.split(',')
                     if p.strip())
    for p in policies:
        gc.parse_policy(p)
    info = configure_torch_for_device()
    peaks, peak_source = resolve_peaks(
        info.device, bw_gbps=args.bw_gbps, tflops=args.tflops)
    report = build_report(
        T=args.T, batch=args.batch, d_model=args.d_model, seed=args.seed,
        policies=policies, itemsize=args.itemsize, device=info.device,
        peaks=peaks, peak_source=peak_source,
        quality_seeds=parse_int_list(args.quality_seeds),
        quality_steps=args.quality_steps, quality_lr=args.quality_lr,
        skip_quality=args.skip_quality, timing_reps=args.timing_reps)
    payload = sanitize_for_json(report_envelope(report))
    json.dumps(payload, allow_nan=False)
    _print_report(report)
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    write_results_json(payload, args.json, logger=logger)

    # Campaign-level ledger: timing seed + quality-probe seeds, recipe,
    # artifact and probe status in one strict-JSON line.
    quality_seeds = parse_int_list(args.quality_seeds)
    run_seeds = sorted(set([int(args.seed), *quality_seeds]))
    quality = report['quality_probe']
    append_run_manifest(make_run_manifest_entry(
        experiment=KIND,
        seeds=run_seeds,
        provenance=payload['metadata']['provenance'],
        hyperparameters={
            'T': int(args.T),
            'batch': int(args.batch),
            'd_model': int(args.d_model),
            'timing_seed': int(args.seed),
            'policies': list(policies),
            'itemsize': int(args.itemsize),
            'timing_reps': int(args.timing_reps),
            'quality_steps': int(args.quality_steps),
            'quality_lr': float(args.quality_lr),
            'quality_seeds': list(quality_seeds),
            'skip_quality': bool(args.skip_quality),
            'peak_source': peak_source,
            'peaks': dict(peaks),
        },
        data={
            'tasks': ['gradient-checkpointing parity/roofline',
                      'dense MQA/GQA/MHA KV ladder',
                      'multi_agg MQAR quality probe'],
            'seq_len': int(args.T),
            'd_model': int(args.d_model),
            'dtype_bytes': int(args.itemsize),
        },
        artifacts={'results': args.json},
        results={
            'n_policies': len(policies),
            'quality_status': quality.get('status'),
            'quality_probes': len(quality.get('probes', [])),
            'all_policy_parity_ok': bool(all(
                r['output_max_abs_diff_vs_baseline'] < 1e-5
                and r['grad_max_abs_diff_vs_baseline'] < 1e-5
                for r in report['checkpoint_policies'])),
        },
        status='completed',
    ), logger=logger)

    print(f'[exp14] wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
