"""Generate figures from experiment results.

Handles both the original single-seed result format and the new multi-seed
format (with mean / std / CI95). When multi-seed results are present, bars
are drawn with error bars showing the 95% CI half-width.
"""

from __future__ import annotations

import collections
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as _fm
for _fp in ('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    try:
        _fm.fontManager.addfont(_fp)
    except (FileNotFoundError, OSError):
        pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC',
                                  'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(_ROOT, 'results'))
_FIGURES_DIR = os.environ.get('FIGURES_DIR', os.path.join(_ROOT, 'figures'))


def load(name):
    """Load a results JSON file.

    Returns ``[]`` on FileNotFoundError or JSONDecodeError so the caller
    can degrade gracefully (skip the figure with a log line) rather than
    crashing the entire figure-generation step. A truncated/malformed
    JSON (common when an experiment is killed mid-write) used to crash
    every subsequent figure too.

    Supports two on-disk schemas:

    1. **Legacy bare array** ``[{...}, {...}]`` (the historical format;
       still used by all result files except ``exp4_mqar.json`` after the
       P0-1 fix).
    2. **Envelope object** ``{"metadata": {...}, "results": [...]}``
       introduced by the P0-1 fix so that a result file can carry
       provenance / caveats alongside the data array without producing
       the invalid concatenated-document form ``{...}\\n[...]`` that
       ``json.load`` rejects with ``Extra data``.

    For the envelope form, this function returns the inner ``results``
    array so downstream figure code keeps working unchanged. The
    ``metadata`` block is intentionally discarded here because no current
    figure consumes it; if a future figure needs provenance, add a
    sibling ``load_with_metadata()`` that returns ``(metadata, results)``.
    """
    path = os.path.join(_RESULTS_DIR, name)
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'[load] {path} not found; skipping', file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f'[load] {path} is malformed: {e}; skipping', file=sys.stderr)
        return []
    if isinstance(data, dict) and 'results' in data and isinstance(
            data['results'], list):
        return data['results']
    return data


def _ensure_figures_dir():
    """Ensure ``figures/`` exists before any savefig call."""
    os.makedirs(_FIGURES_DIR, exist_ok=True)


def _ratio_layers(ratio: str) -> int:
    """Return the total layer count encoded by a ratio string like '3:1:1'.

    Returns 0 if the string contains any non-numeric component (e.g.
    'baseline', '3:1:1 (5L)'), instead of raising ValueError.
    """
    try:
        return sum(int(x) for x in ratio.replace(' ', '').split(':'))
    except ValueError:
        return 0


def _has_multiseed(record):
    """True if a record is in the new multi-seed format."""
    return 'mean_acc' in record and 'per_seed' in record


def fig_benchmark():
    """Figure: latency vs sequence length for each operator."""
    data = load('exp2_benchmark.json')
    _OP_BOUNDARY_FALLBACK = {
        'softmax': 'core',
        'kda_rec': 'core',
        'kda_chunk': 'core',
        'csa': 'end_to_end_single_layer',
        'hca': 'end_to_end_single_layer',
        'hybrid': 'end_to_end_multi_layer',
    }
    ops = {}
    for r in data:
        if 'error' in r:
            continue
        t = r.get('time_ms')
        if t is None:
            continue
        boundary = r.get('compute_boundary')
        if boundary is None:
            boundary = _OP_BOUNDARY_FALLBACK.get(r['op'], 'unknown')
        n_layers = r.get('n_layers', 1)
        ops.setdefault(r['op'], {
            'points': [],
            'boundary': boundary,
            'n_layers': n_layers,
        })['points'].append((r['T'], t))
    boundary_groups = {}
    for op, info in ops.items():
        boundary_groups.setdefault(info['boundary'], []).append(op)

    markers = {'softmax': 'o-', 'kda_rec': 's-', 'kda_chunk': '^-',
               'csa': 'D-', 'hca': 'v-', 'hybrid': 'p-'}
    labels = {'softmax': 'Softmax attention', 'kda_rec': 'KDA (recurrent)',
              'kda_chunk': 'KDA (chunk)', 'csa': 'CSA', 'hca': 'HCA',
              'hybrid': 'Fused KDA+CSA+HCA'}
    boundary_titles = {
        'core': 'Core kernel only\n(q/k/v pre-projected; 1 layer)',
        'end_to_end_single_layer': 'End-to-end single layer\n(H -> projections -> attention -> o_proj)',
        'end_to_end_multi_layer': 'End-to-end multi-layer\n(5-layer stack with LayerNorm + state)',
        'unknown': 'Unknown boundary',
    }
    boundary_order = ['core', 'end_to_end_single_layer',
                      'end_to_end_multi_layer', 'unknown']
    ordered_boundaries = [b for b in boundary_order if b in boundary_groups]

    device = next((r.get('device', 'cpu') for r in data if 'error' not in r),
                  'cpu')
    backend_values = sorted({r.get('kda_backend') for r in data
                             if r.get('kda_backend') is not None})
    backend_note = (
        f"KDA backend: {', '.join(backend_values)}"
        if backend_values else "KDA backend: reference (legacy result schema)"
    )

    n_subplots = len(ordered_boundaries)
    if n_subplots == 0:
        print('Skipping fig_benchmark (no data to plot)')
        return

    fig, axes = plt.subplots(n_subplots, 1, figsize=(7, 3.0 * n_subplots),
                             sharex=True, constrained_layout=True)
    if n_subplots == 1:
        axes = [axes]

    for ax, boundary in zip(axes, ordered_boundaries):
        group_ops = boundary_groups[boundary]
        for op in group_ops:
            pts = ops[op]['points']
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, markers.get(op, 'o-'),
                    label=labels.get(op, op), markersize=5)
        ax.set_ylabel(f'Latency (ms, {device})')
        ax.set_title(boundary_titles.get(boundary, boundary), fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        if group_ops:
            ax.legend(fontsize=8, loc='upper left')

    axes[-1].set_xlabel('Sequence length T')
    axes[-1].set_xscale('log', base=2)

    fig.suptitle(
        'Operator latency vs. sequence length — split by compute boundary\n'
        'WARNING: each subplot uses a DIFFERENT measurement boundary; '
        'cross-subplot comparison is NOT apples-to-apples\n'
        + backend_note,
        fontsize=10, fontweight='bold')

    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_benchmark.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_benchmark.png'), dpi=150)
    finally:
        plt.close(fig)
    print('Saved figures/fig_benchmark.pdf')


def fig_kv_cache():
    """Figure: KV cache size ratio vs GQA baseline."""
    data = load('exp3_kv_cache.json')
    available_modes = {r.get('accounting_mode') for r in data if 'accounting_mode' in r}
    if 'full_accounting' in available_modes:
        mode = 'full_accounting'
    elif 'compressed_kv_only' in available_modes:
        mode = 'compressed_kv_only'
    else:
        mode = None
    ops = {}
    for r in data:
        if mode is not None and r.get('accounting_mode', mode) != mode:
            continue
        ratio = r.get('kv_ratio_vs_gqa_5l', r.get('kv_ratio_vs_gqa_1l',
                                                   r.get('kv_ratio_vs_gqa')))
        if ratio is None:
            continue
        ops.setdefault(r['op'], []).append((r['T'], ratio))
    if not ops:
        print('Skipping fig_kv_cache (no data to plot for the selected '
              f'accounting_mode={mode!r})')
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax_gqa': 'o-', 'kda': 's-', 'csa': 'D-',
               'hca': 'v-', 'hybrid_kch': 'p-'}
    labels = {'softmax_gqa': 'Softmax GQA8 (1 layer)', 'kda': 'KDA',
              'csa': 'CSA', 'hca': 'HCA', 'hybrid_kch': 'Fused hybrid (3:1:1)'}
    for op, pts in ops.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, markers.get(op, 'o-'), label=labels.get(op, op), markersize=5)
    ax.set_xlabel('Sequence length T (tokens)')
    ax.set_ylabel('KV cache size / GQA8 (5-layer baseline)')
    ax.set_title('KV cache compression vs. sequence length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.02, color='red', linestyle=':', alpha=0.5,
               label='DeepSeek-V4 target (2%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_kv_cache.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_kv_cache.png'), dpi=150)
    finally:
        plt.close(fig)
    print(f'Saved {os.path.join(_FIGURES_DIR, "fig_kv_cache.pdf")}')


def fig_flops():
    """Figure: prefill FLOPs ratio vs GQA baseline."""
    data = load('exp3_kv_cache.json')
    available_modes = {r.get('accounting_mode') for r in data if 'accounting_mode' in r}
    if 'full_accounting' in available_modes:
        mode = 'full_accounting'
    elif 'compressed_kv_only' in available_modes:
        mode = 'compressed_kv_only'
    else:
        mode = None
    ops = {}
    for r in data:
        if mode is not None and r.get('accounting_mode', mode) != mode:
            continue
        ratio = r.get('flops_ratio_vs_gqa_5l', r.get('flops_ratio_vs_gqa_1l',
                                                      r.get('flops_ratio_vs_gqa')))
        if ratio is None:
            continue
        ops.setdefault(r['op'], []).append((r['T'], ratio))
    if not ops:
        print('Skipping fig_flops (no data to plot for the selected '
              f'accounting_mode={mode!r})')
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax_gqa': 'o-', 'kda': 's-', 'csa': 'D-',
               'hca': 'v-', 'hybrid_kch': 'p-'}
    labels = {'softmax_gqa': 'Softmax GQA8 (1 layer)', 'kda': 'KDA',
              'csa': 'CSA', 'hca': 'HCA', 'hybrid_kch': 'Fused hybrid (3:1:1)'}
    for op, pts in ops.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, markers.get(op, 'o-'), label=labels.get(op, op), markersize=5)
    ax.set_xlabel('Sequence length T (tokens)')
    ax.set_ylabel('Prefill FLOPs / GQA8 (5-layer baseline)')
    ax.set_title('Prefill compute vs. sequence length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.27, color='red', linestyle=':', alpha=0.5,
               label='DeepSeek-V4 target (27%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_flops.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_flops.png'), dpi=150)
    finally:
        plt.close(fig)
    print(f'Saved {os.path.join(_FIGURES_DIR, "fig_flops.pdf")}')


def fig_mqar():
    """Figure: MQAR accuracy bar chart (multi-seed with CI95 error bars)."""
    data = load('exp4_mqar.json')
    if not data:
        print('Skipping MQAR figure (no data)')
        return
    groups = collections.defaultdict(list)
    for r in data:
        groups[r.get('n_kv', 1)].append(r)
    single = len(groups) == 1
    for n_kv in sorted(groups):
        _plot_mqar_group(groups[n_kv], n_kv, single)


def _plot_mqar_group(records, n_kv, write_legacy_name):
    """Plot MQAR accuracy bars for one ``n_kv`` group."""
    ops, means, cis = [], [], []
    chance = 1 / 16
    skipped = 0
    for r in records:
        if r.get('mean_acc') is None and r.get('final_acc') is None:
            skipped += 1
            continue
        ops.append(r['op'])
        if _has_multiseed(r):
            means.append(r['mean_acc'])
            _ci = r.get('ci95_acc')
            cis.append(float('nan') if _ci is None else float(_ci))
            chance = r.get('chance_acc') or chance
        else:
            means.append(r['final_acc'])
            cis.append(0.0)
            chance = r.get('chance_acc') or chance
    if not ops:
        print(f'Skipping MQAR figure for n_kv={n_kv} (all records are '
              f'error rows, no data to plot)')
        return
    if skipped:
        print(f'[fig_mqar] skipped {skipped} error row(s) for n_kv={n_kv}')
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(ops, means, yerr=cis, capsize=5,
                  color=colors[:len(ops)],
                  error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.axhline(chance, color='gray', linestyle='--', alpha=0.7,
               label=f'Chance ({chance:.3f})')
    ax.set_ylabel('MQAR accuracy (mean over seeds, 95% CI)')
    ok_records = [r for r in records
                  if r.get('mean_acc') is not None
                  or r.get('final_acc') is not None]
    if ok_records:
        n_seeds = ok_records[0].get('n_seeds', 1)
    else:
        n_seeds = 1

    title_rec = next((r for r in ok_records if r.get('op') != 'softmax'),
                     ok_records[0] if ok_records else None)
    softmax_rec = next((r for r in ok_records if r.get('op') == 'softmax'),
                       None)
    steps = 100
    softmax_steps = None
    if title_rec:
        per_seed = title_rec.get('per_seed') or []
        if per_seed:
            steps = per_seed[0].get('steps', 100)
    if softmax_rec is not None:
        ps = softmax_rec.get('per_seed') or []
        if ps:
            softmax_steps = ps[0].get('steps')
    if softmax_steps is not None and softmax_steps != steps:
        title_steps = f'softmax={softmax_steps}, others={steps}'
    else:
        title_steps = str(steps)
    _valid = all(r.get('conclusions_valid', True) for r in ok_records)
    _n_sig = sum(1 for r in ok_records if r.get('significant_bonferroni'))
    _n_total = len(ok_records)
    validity_note = ''
    if not _valid:
        validity_note = (f' WARNING: conclusions_valid=False '
                         f'({_n_sig}/{_n_total} Bonferroni-sig). '
                         f'Treat as exploratory, not confirmatory.')
    if validity_note:
        validity_note = '\n' + validity_note.lstrip()
    ax.set_title(f'Multi-Query Associative Recall (n_kv={n_kv}, '
                 f'{n_seeds} seeds, {title_steps} steps){validity_note}',
                 fontsize=8)
    _finite_upper = [m + c for m, c in zip(means, cis)
                     if math.isfinite(m + c)]
    acc_upper = (max(_finite_upper, default=0.0)) * 1.3
    ax.set_ylim(0, max(acc_upper, 0.35))
    for bar, m, c in zip(bars, means, cis):
        _c_offset = 0.0 if not math.isfinite(c) else c
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + _c_offset + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend(fontsize=9)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, f'fig_mqar_nkv{n_kv}.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, f'fig_mqar_nkv{n_kv}.png'), dpi=150)
        if write_legacy_name:
            fig.savefig(os.path.join(_FIGURES_DIR, 'fig_mqar.pdf'), dpi=150)
            fig.savefig(os.path.join(_FIGURES_DIR, 'fig_mqar.png'), dpi=150)
    finally:
        plt.close(fig)
    msg = f'Saved figures/fig_mqar_nkv{n_kv}.pdf'
    if write_legacy_name:
        msg += ' (and fig_mqar.pdf)'
    print(msg)


def fig_ablation():
    """Figure: ablation accuracy and latency (multi-seed with error bars)."""
    path = os.path.join(_RESULTS_DIR, 'exp5_ablation.json')
    if not os.path.exists(path):
        print('Skipping ablation figure (no results yet)')
        return
    data = load('exp5_ablation.json')
    if not data:
        print('Skipping ablation figure (no data)')
        return
    groups = collections.defaultdict(list)
    for r in data:
        groups[r.get('n_kv', 1)].append(r)
    single = len(groups) == 1
    for n_kv in sorted(groups):
        _plot_ablation_group(groups[n_kv], n_kv, single)


def _plot_ablation_group(records, n_kv, write_legacy_name):
    """Plot ablation accuracy + latency bars for one ``n_kv`` group."""
    ratios = []
    accs = []
    acc_cis = []
    fwds = []
    n_params = []
    n_layers = []
    error_flags = []
    skipped = 0
    for r in records:
        ratio = r['ratio']
        is_error = 'error' in r or (
            r.get('mean_acc') is None and r.get('final_acc') is None)
        if is_error:
            error_flags.append(True)
            ratios.append(ratio)
            accs.append(0.0)
            acc_cis.append(0.0)
            fwds.append(0.0)
            n_params.append(r.get('n_params') or 0)
            n_layers.append(r.get('n_layers')
                            or _ratio_layers(ratio))
            skipped += 1
        else:
            error_flags.append(False)
            ratios.append(ratio)
            accs.append(r.get('mean_acc', r.get('final_acc', 0.0)) or 0.0)
            _ci = r.get('ci95_acc')
            acc_cis.append(float('nan') if _ci is None else float(_ci))
            fwds.append(r.get('mean_fwd_ms', r.get('fwd_ms', 0.0)) or 0.0)
            n_params.append(r.get('n_params') or 0)
            n_layers.append(r.get('n_layers')
                            or _ratio_layers(ratio))

    if skipped:
        print(f'[fig_ablation] {skipped} ratio(s) had errors and are shown '
              f'as 0-height "ERR" bars for n_kv={n_kv}')

    if not ratios:
        print(f'Skipping ablation figure for n_kv={n_kv} (no records to plot)')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(ratios))

    bar_colors = ['#C44E52' if e else '#4C72B0' for e in error_flags]
    bars1 = ax1.bar(x, accs, yerr=acc_cis, capsize=4, color=bar_colors,
                    error_kw={'linewidth': 1.3, 'ecolor': '#333'})
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r}\n({l}L, {p}p)' for r, l, p in zip(ratios, n_layers, n_params)],
                        rotation=0, fontsize=8)
    ax1.set_ylabel('MQAR accuracy (mean +/- CI95)')
    ax1.set_title('Accuracy vs. KDA:CSA:HCA ratio')
    _ok_rec = next((r for r in records
                    if 'error' not in r
                    and (r.get('mean_acc') is not None
                         or r.get('final_acc') is not None)), {})
    chance_acc = _ok_rec.get('chance_acc') or 1/16
    ax1.axhline(chance_acc, color='gray', linestyle='--', alpha=0.5,
                label=f'chance ({chance_acc:.4f})')
    for bar, acc, is_err in zip(bars1, accs, error_flags):
        label = 'ERR' if is_err else f'{acc:.3f}'
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 label, ha='center', va='bottom', fontsize=9)
    ax1.legend(fontsize=8)
    _finite_upper = [float(a) + float(c) for a, c in zip(accs, acc_cis)
                     if math.isfinite(float(a) + float(c))]
    acc_upper = (max(_finite_upper, default=0.0)) * 1.3
    ax1.set_ylim(0, max(acc_upper, 0.2))

    bar_colors2 = ['#C44E52' if e else '#55A868' for e in error_flags]
    bars2 = ax2.bar(x, fwds, color=bar_colors2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{r}\n({l}L)' for r, l in zip(ratios, n_layers)],
                        rotation=0, fontsize=8)
    ax2.set_ylabel('Forward latency (ms)')
    ax2.set_title('Latency vs. KDA:CSA:HCA ratio')
    for bar, fwd, is_err in zip(bars2, fwds, error_flags):
        label = 'ERR' if is_err else f'{fwd:.1f}'
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 label, ha='center', va='bottom', fontsize=9)
    fwd_max = max((float(f) for f in fwds), default=0.0)
    ax2.set_ylim(0, max(fwd_max * 1.25, 1.0))

    _valid = all(r.get('conclusions_valid', True) for r in records
                 if 'error' not in r)
    _n_sig = sum(1 for r in records if r.get('significant_bonferroni'))
    _n_total = sum(1 for r in records if 'error' not in r)
    validity_note = ''
    if not _valid:
        validity_note = (f' WARNING: conclusions_valid=False '
                         f'({_n_sig}/{_n_total} Bonferroni-sig). '
                         f'Treat as exploratory, not confirmatory.')
    if n_layers:
        max_l = max(n_layers)
        min_l = min(n_layers)
        if max_l != min_l:
            max_r = ratios[n_layers.index(max_l)]
            min_r = ratios[n_layers.index(min_l)]
            depth_note = (f'{max_r} has {max_l}L vs {min_r} has {min_l}L — '
                          'depth confound noted in paper.')
        else:
            depth_note = 'all ratios have equal depth.'
    else:
        depth_note = 'depth confound noted in paper.'
    fig.suptitle(f'Ablation: ratio trade-off (n_kv={n_kv}, multi-seed). '
                 f'{depth_note}{validity_note}',
                 fontsize=8, y=0.98)
    fig.subplots_adjust(top=0.91, bottom=0.18, left=0.08, right=0.97, wspace=0.3)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, f'fig_ablation_nkv{n_kv}.pdf'),
                    dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, f'fig_ablation_nkv{n_kv}.png'),
                    dpi=150)
        if write_legacy_name:
            fig.savefig(os.path.join(_FIGURES_DIR, 'fig_ablation.pdf'), dpi=150)
            fig.savefig(os.path.join(_FIGURES_DIR, 'fig_ablation.png'), dpi=150)
    finally:
        plt.close(fig)
    msg = f'Saved figures/fig_ablation_nkv{n_kv}.pdf'
    if write_legacy_name:
        msg += ' (and fig_ablation.pdf)'
    print(msg)


def fig_decoding():
    """Figure: per-token decoding latency vs cached context length."""
    path = os.path.join(_RESULTS_DIR, 'exp6_decoding.json')
    if not os.path.exists(path):
        print('Skipping decoding figure (no results yet)')
        return
    data = load('exp6_decoding.json')
    ops = {}
    for r in data:
        if 'error' in r:
            continue
        v = r.get('median_decode_ms_per_token')
        if v is None:
            continue
        ops.setdefault(r['op'], []).append((r['prefill_len'], v))
    if not ops:
        print('Skipping decoding figure (no successful runs)')
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    markers = {'softmax': 'o-', 'kda': 's-', 'csa': '^-',
               'hca': 'D-', 'hybrid': 'v-'}
    labels = {'softmax': 'Softmax attention (growing KV cache)',
              'kda': 'KDA recurrent (O(1) state)',
              'csa': 'CSA sparse (incremental cache)',
              'hca': 'HCA dense (incremental cache)',
              'hybrid': 'Hybrid KDA+CSA+HCA stack (incremental caches)'}
    for op, pts in ops.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, markers.get(op, 'o-'), label=labels.get(op, op), markersize=7)
    device = next((r.get('device', 'cpu') for r in data if 'error' not in r),
                  'cpu')
    backend_values = sorted({r.get('kda_backend') for r in data
                             if r.get('kda_backend') is not None})
    backend_note = (
        f"KDA backend: {', '.join(backend_values)}"
        if backend_values else "KDA backend: reference (legacy result schema)"
    )
    ax.set_xlabel('Cached context length (tokens)')
    ax.set_ylabel(f'Per-token decode latency (ms, {device})')
    ax.set_title('Decoding latency vs. cached context length\n' + backend_note)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_decoding.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_decoding.png'), dpi=150)
    finally:
        plt.close(fig)
    print('Saved figures/fig_decoding.pdf')


def fig_architecture():
    """Figure: schematic of the fused hybrid architecture."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    blocks = [
        (0.5, 'KDA', '#4C72B0'),
        (2.0, 'KDA', '#4C72B0'),
        (3.5, 'KDA', '#4C72B0'),
        (5.0, 'CSA', '#C44E52'),
        (6.5, 'HCA', '#55A868'),
    ]
    for x, label, color in blocks:
        rect = plt.Rectangle((x, 1.5), 1.2, 1.0, facecolor=color,
                             edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.6, 2.0, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    ax.annotate('', xy=(0.5, 2.0), xytext=(0.0, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(0.0, 2.3, 'x', fontsize=11)
    ax.annotate('', xy=(8.2, 2.0), xytext=(7.7, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(8.2, 2.3, 'y', fontsize=11)
    for x, _, _ in blocks:
        ax.annotate('', xy=(x + 1.2, 1.2), xytext=(x, 1.2),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))
    ax.text(4.5, 0.5, 'KDA: fine-grained gated delta rule (linear, O(1) state)\n'
            'CSA: compressed sparse attention (top-k retrieval)\n'
            'HCA: heavily compressed attention (dense global context)',
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Fused KDA+CSA+HCA hybrid attention (3:1:1 layout)', fontsize=12)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_architecture.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_architecture.png'), dpi=150)
    finally:
        plt.close(fig)
    print('Saved figures/fig_architecture.pdf')


def main():
    _ensure_figures_dir()
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    fig_architecture()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp2_benchmark.json')):
        fig_benchmark()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp3_kv_cache.json')):
        fig_kv_cache()
        fig_flops()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp4_mqar.json')):
        fig_mqar()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp5_ablation.json')):
        fig_ablation()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp6_decoding.json')):
        fig_decoding()
    print('\nAll figures generated.')


if __name__ == '__main__':
    main()
