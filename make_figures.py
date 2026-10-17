"""Generate figures from experiment results.

Handles both the original single-seed result format and the new multi-seed
format (with mean / std / CI95). When multi-seed results are present, bars
are drawn with error bars showing the 95% CI half-width.
"""

from __future__ import annotations

import collections
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

from results_schema import load_or_empty  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(_ROOT, 'results'))
_FIGURES_DIR = os.environ.get('FIGURES_DIR', os.path.join(_ROOT, 'figures'))


def load_with_metadata(name):
    """Load a results JSON file, returning ``(metadata, rows)``.

    Thin wrapper over :func:`results_schema.load_or_empty` so figure code
    never crashes on a missing / malformed / future-version file (it logs
    the reason to stderr and yields ``({}, [])``).
    """
    path = os.path.join(_RESULTS_DIR, name)
    return load_or_empty(path, context='make_figures')


def load(name):
    """Load a results JSON file and return its row list.

    Returns ``[]`` on FileNotFoundError / JSONDecodeError / incompatible
    schema version so the caller can degrade gracefully (skip the figure
    with a log line) rather than crashing the entire figure-generation step.

    Schema handling is delegated to :mod:`results_schema`, which accepts:

    1. **Legacy bare array** ``[{...}, {...}]`` (written by Exp 1/2/3/5/6).
    2. **Envelope object** ``{"metadata": {...}, "results": [...]}`` in any
       revision (Exp 4's informal ``schema_version: 1``, or the versioned
       ``"schema_version": "1.0"`` envelopes written by current tooling),
       returning the inner ``results`` array so downstream figure code
       keeps working unchanged. A file declaring a FUTURE major version is
       refused loudly (stderr) instead of being misread — see
       ``results_schema.SchemaVersionError``.

    Use :func:`load_with_metadata` when the provenance block is needed
    (e.g. the roofline figure reads machine peaks from it).
    """
    _meta, rows = load_with_metadata(name)
    return rows


def _ensure_figures_dir():
    """Ensure ``figures/`` exists before any savefig call."""
    os.makedirs(_FIGURES_DIR, exist_ok=True)


def _ratio_vs_gqa(row, kind, default_mode='5l'):
    """Select the KV/FLOPs ratio to draw for one result row.

    ``run_kv_cache.py`` emits three ratio families against the GQA8
    baseline: ``*_vs_gqa_1l`` (denominator = ONE GQA8 layer) and
    ``*_vs_gqa_5l`` (denominator = FIVE GQA8 layers) plus a legacy
    ``*_vs_gqa``.

    The figure axes are labelled against the 5-layer baseline, so every
    row (including the single-layer ``softmax_gqa`` baseline, whose
    5-layer-relative value is 0.2) is drawn from the 5l field so the
    plotted values are consistent with the axis definition.
    """
    base = 'kv_ratio' if kind == 'kv' else 'flops_ratio'
    ratio = row.get(f'{base}_vs_gqa_5l',
                    row.get(f'{base}_vs_gqa_1l',
                            row.get(f'{base}_vs_gqa')))
    return ratio if ratio is not None else None


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
        'kda_rec_hybrid': 'core',
        'kda_chunk_hybrid': 'core',
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
               'kda_rec_hybrid': 's--', 'kda_chunk_hybrid': '^--',
               'csa': 'D-', 'hca': 'v-', 'hybrid': 'p-'}
    labels = {'softmax': 'Softmax attention', 'kda_rec': 'KDA (recurrent)',
              'kda_chunk': 'KDA (chunk)', 'kda_rec_hybrid': 'KDA (recurrent, hybrid-config)',
              'kda_chunk_hybrid': 'KDA (chunk, hybrid-config)', 'csa': 'CSA', 'hca': 'HCA',
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
        ratio = _ratio_vs_gqa(r, 'kv')
        if ratio is None:
            continue
        ops.setdefault(r['op'], []).append((r['T'], ratio))
    if not ops:
        print('Skipping fig_kv_cache (no data to plot for the selected '
              f'accounting_mode={mode!r})')
        return
    mode_note = {'full_accounting': 'full KV accounting',
                 'compressed_kv_only': 'compressed-KV-only accounting'}.get(
                     mode, f'accounting_mode={mode!r}')
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
    ax.set_title(f'KV cache compression vs. sequence length ({mode_note})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5,
               label='Softmax GQA8 (5-layer) = 1.0')
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
        ratio = _ratio_vs_gqa(r, 'flops')
        if ratio is None:
            continue
        ops.setdefault(r['op'], []).append((r['T'], ratio))
    if not ops:
        print('Skipping fig_flops (no data to plot for the selected '
              f'accounting_mode={mode!r})')
        return
    mode_note = {'full_accounting': 'full KV accounting',
                 'compressed_kv_only': 'compressed-KV-only accounting'}.get(
                     mode, f'accounting_mode={mode!r}')
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
    ax.set_title(f'Prefill compute vs. sequence length ({mode_note})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5,
               label='Softmax GQA8 (5-layer) = 1.0')
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
    # Per-op colours by name, matching fig_architecture (KDA blue, CSA red,
    # HCA green); the softmax baseline gets the fourth palette colour, so an
    # op's colour never depends on the order/skipping of records.
    _palette = ['#8172B2', '#4C72B0', '#C44E52', '#55A868']
    op_colors = {'softmax': '#8172B2', 'kda': '#4C72B0',
                 'csa': '#C44E52', 'hca': '#55A868'}
    bars = ax.bar(ops, means, yerr=cis, capsize=5,
                  color=[op_colors.get(op, _palette[i % len(_palette)])
                         for i, op in enumerate(ops)],
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
    _finite_upper = [m + (c if math.isfinite(c) else 0.0)
                     for m, c in zip(means, cis)]
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
            n_params.append(0 if r.get('n_params') is None else r['n_params'])
            n_layers.append(_ratio_layers(ratio)
                            if r.get('n_layers') is None else r['n_layers'])
            skipped += 1
        else:
            error_flags.append(False)
            ratios.append(ratio)
            _acc = r.get('mean_acc')
            if _acc is None:
                _acc = r.get('final_acc')
            accs.append(0.0 if _acc is None else _acc)
            _ci = r.get('ci95_acc')
            acc_cis.append(float('nan') if _ci is None else float(_ci))
            _fwd = r.get('mean_fwd_ms')
            if _fwd is None:
                _fwd = r.get('fwd_ms')
            fwds.append(0.0 if _fwd is None else _fwd)
            n_params.append(0 if r.get('n_params') is None else r['n_params'])
            n_layers.append(_ratio_layers(ratio)
                            if r.get('n_layers') is None else r['n_layers'])

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
    _finite_upper = [float(a) + (float(c) if math.isfinite(float(c)) else 0.0)
                     for a, c in zip(accs, acc_cis)]
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


def fig_mechanism_ablation():
    """Figure: CSA mechanism ablation (Exp 8, multi-seed with error bars).

    Two panels: MQAR accuracy per mechanism variant and the lightning
    indexer's gold-block recall@k at the cue position. Recall is a
    diagnostic of *selection quality*; its per-variant chance level
    (``min(k, available)/available``) is drawn alongside so a curve that
    fails to beat chance is visible as such.
    """
    path = os.path.join(_RESULTS_DIR, 'exp8_mechanism_ablation.json')
    if not os.path.exists(path):
        print('Skipping mechanism-ablation figure (no results yet)')
        return
    meta, data = load_with_metadata('exp8_mechanism_ablation.json')
    if not data:
        print('Skipping mechanism-ablation figure (no data)')
        return
    groups = collections.defaultdict(list)
    for r in data:
        groups[r.get('n_kv', 1)].append(r)
    single = len(groups) == 1
    for n_kv in sorted(groups):
        _plot_mechanism_group(groups[n_kv], n_kv, single, meta)


def _plot_mechanism_group(records, n_kv, write_legacy_name, meta):
    """Plot accuracy + gold-block recall bars for one ``n_kv`` group."""
    labels = []
    accs = []
    acc_cis = []
    recalls = []
    recall_cis = []
    recall_na = []
    chances = []
    error_flags = []
    skipped = 0
    for r in records:
        name = str(r.get('variant', '?'))
        label = (f"{name}\nrope={'on' if r.get('rope') else 'off'}, "
                 f"k={r.get('topk')}\n"
                 f"STE={'on' if r.get('use_ste', True) else 'off'}")
        labels.append(label)
        is_error = 'error' in r or (
            r.get('mean_acc') is None and r.get('final_acc') is None)
        if is_error:
            error_flags.append(True)
            accs.append(0.0)
            acc_cis.append(0.0)
            recalls.append(0.0)
            recall_cis.append(0.0)
            recall_na.append(True)
            chances.append(None)
            skipped += 1
            continue
        error_flags.append(False)
        accs.append(0.0 if r.get('mean_acc') is None else r['mean_acc'])
        _ci = r.get('ci95_acc')
        acc_cis.append(float('nan') if _ci is None else float(_ci))
        _rec = r.get('mean_gold_recall')
        has_recall = _rec is not None
        recalls.append(0.0 if not has_recall else float(_rec))
        _rci = r.get('ci95_gold_recall')
        recall_cis.append(float('nan') if _rci is None else float(_rci))
        recall_na.append(not has_recall)
        chances.append(None if r.get('chance_gold_recall') is None
                       else float(r['chance_gold_recall']))
        if not has_recall:
            skipped += 1
    if not labels:
        print(f'Skipping mechanism-ablation figure for n_kv={n_kv} '
              f'(no records to plot)')
        return
    if skipped:
        print(f'[fig_mechanism_ablation] {skipped} variant(s) errored or had '
              f'no recall diagnostic (shown as ERR/n-a) for n_kv={n_kv}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(labels))

    bar_colors = ['#C44E52' if e else '#4C72B0' for e in error_flags]
    bars1 = ax1.bar(x, accs, yerr=acc_cis, capsize=4, color=bar_colors,
                    error_kw={'linewidth': 1.3, 'ecolor': '#333'})
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=6.5)
    ax1.set_ylabel('MQAR accuracy (mean +/- CI95)')
    ax1.set_title('Mechanism effect on accuracy')
    chance_acc = next((r.get('chance_acc') for r in records
                       if r.get('chance_acc') is not None), 1 / 16)
    ax1.axhline(chance_acc, color='gray', linestyle='--', alpha=0.5,
                label=f'chance ({chance_acc:.4f})')
    for bar, acc, is_err in zip(bars1, accs, error_flags):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 'ERR' if is_err else f'{acc:.3f}',
                 ha='center', va='bottom', fontsize=8)
    ax1.legend(fontsize=8)
    upper = max((a + (c if math.isfinite(c) else 0.0)
                 for a, c in zip(accs, acc_cis)), default=0.0)
    ax1.set_ylim(0, max(upper * 1.3, 0.2))

    bar_colors2 = ['#C44E52' if e else ('#BBBBBB' if na else '#55A868')
                   for e, na in zip(error_flags, recall_na)]
    w = 0.38
    bars2 = ax2.bar(x - w / 2, recalls, width=w, yerr=recall_cis, capsize=3,
                    color=bar_colors2,
                    error_kw={'linewidth': 1.3, 'ecolor': '#333'},
                    label='model recall@k')
    chance_plot = [0.0 if c is None else c for c in chances]
    bars_c = ax2.bar(x + w / 2, chance_plot, width=w,
                     color=['#FFFFFF' if c is None else '#999999'
                            for c in chances],
                     edgecolor='#555', linewidth=0.6,
                     label='chance = min(k, n_avail) / n_avail')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, fontsize=6.5)
    ax2.set_ylabel('Gold-block recall@k (mean +/- CI95)')
    ax2.set_title('Lightning-indexer selection quality at the cue token')
    for bar, rec, is_err, na in zip(bars2, recalls, error_flags, recall_na):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 'ERR' if is_err else ('n/a' if na else f'{rec:.3f}'),
                 ha='center', va='bottom', fontsize=7)
    for bar, c in zip(bars_c, chances):
        if c is None:
            continue
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{c:.3f}', ha='center', va='bottom', fontsize=6,
                 color='#555')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.set_ylim(0, 1.25)

    valid = bool(meta.get('conclusions_valid', False))
    note = ('' if valid else
            ' WARNING: conclusions_valid=False (underpowered or no '
            'Bonferroni-significant effect) — treat as exploratory.')
    fig.suptitle(f'CSA mechanism ablation: partial RoPE / top-k block '
                 f'selection / indexer STE (n_kv={n_kv}, multi-seed).{note}',
                 fontsize=8, y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.07, right=0.98,
                        wspace=0.28)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(
            _FIGURES_DIR, f'fig_mechanism_ablation_nkv{n_kv}.pdf'), dpi=150)
        fig.savefig(os.path.join(
            _FIGURES_DIR, f'fig_mechanism_ablation_nkv{n_kv}.png'), dpi=150)
        if write_legacy_name:
            fig.savefig(os.path.join(
                _FIGURES_DIR, 'fig_mechanism_ablation.pdf'), dpi=150)
            fig.savefig(os.path.join(
                _FIGURES_DIR, 'fig_mechanism_ablation.png'), dpi=150)
    finally:
        plt.close(fig)
    msg = f'Saved figures/fig_mechanism_ablation_nkv{n_kv}.pdf'
    if write_legacy_name:
        msg += ' (and fig_mechanism_ablation.pdf)'
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
        nl = next((r.get('n_layers', 1) for r in data
                   if r.get('op') == op and 'error' not in r), 1)
        label = labels.get(op, op)
        if nl and nl > 1:
            label = f'{label} ({nl} layers)'
        ax.plot(xs, ys, markers.get(op, 'o-'), label=label, markersize=7)
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
    # CSA/HCA/hybrid rows are measured with incremental compressed-block /
    # sliding-window caches, whose per-token path (top-k selection, block
    # append) differs from softmax/KDA. The raw median is plotted as-is, so
    # keep the caveat visible on the figure rather than only in the JSON
    # metadata.
    max_layers = max((r.get('n_layers', 1) for r in data if 'error' not in r),
                     default=1)
    layer_note = (
        f'hybrid is a {max_layers}-layer stack; other methods are '
        'single-layer. Compare latency ratios, not raw times across methods.'
        if max_layers > 1 else
        'all methods measured on a single layer.')
    ax.text(0.02, 0.02,
            'CSA/HCA/hybrid use incremental caches; their decode latency '
            'includes per-token Python control flow.\n' + layer_note,
            transform=ax.transAxes, fontsize=7, color='0.4', ha='left',
            va='bottom')
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


def fig_roofline():
    """Figure: arithmetic intensity vs achieved throughput (roofline, Exp 7).

    Reads the machine peaks from the Exp 7 metadata so the ceiling line is
    drawn for the host that actually produced the timings (measured or
    preset; see ``run_roofline._resolve_peaks``). Points are the per-(T, op)
    Exp 2 measurements joined with analytic FLOPs / compulsory bytes.
    """
    meta, data = load_with_metadata('exp7_roofline.json')
    pts = [r for r in data
           if r.get('achieved_tflops') and
           r.get('arithmetic_intensity_flops_per_byte')]
    if not pts:
        print('Skipping fig_roofline (no data to plot)')
        return
    peak_t = meta.get('peak_tflops')
    peak_b = meta.get('peak_bw_gbps')
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    ais = [p['arithmetic_intensity_flops_per_byte'] for p in pts]
    xmin = max(min(ais) * 0.5, 1e-3)
    xmax = max(max(ais) * 2.0, xmin * 10.0)
    if peak_t and peak_b:
        ridge = peak_t * 1e12 / (peak_b * 1e9)
        xmax = max(xmax, ridge * 2.0)
        line_x = np.logspace(math.log10(xmin), math.log10(xmax), 256)
        line_y = np.minimum(peak_t, peak_b * line_x / 1000.0)
        ax.plot(line_x, line_y, 'k--', lw=1.6,
                label=f'roofline: {peak_t:.3g} TFLOP/s, '
                      f'{peak_b:.0f} GB/s')
        ax.axvline(ridge, color='gray', ls=':', lw=1.0)
        ax.text(ridge, 0.06, f'  ridge {ridge:.2f} FLOP/B', fontsize=7,
                color='gray', va='bottom',
                transform=ax.get_xaxis_transform())
    markers = {'softmax': 'o', 'kda_rec': 's', 'kda_chunk': '^',
               'kda_rec_hybrid': 's', 'kda_chunk_hybrid': '^',
               'csa': 'D', 'hca': 'v', 'hybrid': 'P'}
    labels = {'softmax': 'Softmax attention', 'kda_rec': 'KDA (recurrent)',
              'kda_chunk': 'KDA (chunk)',
              'kda_rec_hybrid': 'KDA (rec., hybrid cfg)',
              'kda_chunk_hybrid': 'KDA (chunk, hybrid cfg)',
              'csa': 'CSA', 'hca': 'HCA', 'hybrid': 'Fused KDA+CSA+HCA'}
    ops = sorted({p['op'] for p in pts})
    for op in ops:
        rows_op = [p for p in pts if p['op'] == op]
        ax.scatter([p['arithmetic_intensity_flops_per_byte']
                    for p in rows_op],
                   [p['achieved_tflops'] for p in rows_op],
                   marker=markers.get(op, 'o'), s=34,
                   label=labels.get(op, op))
        # Label only the smallest and largest T per op: annotating every
        # point makes the dense middle of the plot unreadable.
        ts = sorted(p['T'] for p in rows_op)
        labeled = set(ts) if len(ts) <= 4 else {ts[0], ts[-1]}
        for p in rows_op:
            if p['T'] not in labeled:
                continue
            ax.annotate(f"T={p['T']}",
                        (p['arithmetic_intensity_flops_per_byte'],
                         p['achieved_tflops']),
                        fontsize=5.5, xytext=(3, 3),
                        textcoords='offset points', color='0.35')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic intensity (FLOPs / byte)')
    ax.set_ylabel('Achieved throughput (TFLOP/s)')
    device = meta.get('device_name') or meta.get('device') or 'cpu'
    ax.set_title('Roofline of the reference implementation '
                 f'({device})', fontsize=11)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.text(0.98, 0.02,
            'Boundaries differ per op (core vs end-to-end); bytes are a\n'
            'compulsory-traffic lower bound with no unfold/library traffic.\n'
            'Reference PyTorch is expected to sit far below the roofline.',
            transform=ax.transAxes, fontsize=6.5, color='0.4',
            ha='right', va='bottom')
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_roofline.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_roofline.png'), dpi=150)
    finally:
        plt.close(fig)
    print('Saved figures/fig_roofline.pdf')


_BUDGET_COLORS = {'softmax': 'tab:blue', 'kda': 'tab:red',
                  'csa': 'tab:green', 'hca': 'tab:orange'}
_BUDGET_MARKERS = {'softmax': 'o', 'kda': 's', 'csa': 'D', 'hca': 'v'}


def _budget_curve_panel(ax, curve_rows, xkey, xlabel, title, note):
    """Draw the 4 operators x 2 geometries on one log-log budget panel."""
    any_curve = False
    for geom, ls in (('paper', '-'), ('mqar', '--')):
        for op in ('softmax', 'kda', 'csa', 'hca'):
            pts = sorted([r for r in curve_rows
                          if r.get('geometry') == geom and r['name'] == op
                          and xkey in r], key=lambda r: r[xkey])
            if not pts:
                continue
            any_curve = True
            ax.plot([p[xkey] for p in pts],
                    [p['total_read_elements'] for p in pts],
                    ls, color=_BUDGET_COLORS[op],
                    marker=_BUDGET_MARKERS[op], ms=4.5, lw=1.4,
                    label=f'{op} ({geom})')
    if any_curve:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('KV elements read per query per layer')
        ax.set_title(title, fontsize=11)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        ax.text(0.98, 0.03, note, transform=ax.transAxes, fontsize=6.5,
                color='0.4', ha='right', va='bottom')
    else:
        ax.text(0.5, 0.5, 'no budget-curve rows', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='0.4')
        ax.set_axis_off()
    return any_curve


def fig_budget_frontier():
    """Figure: KV read-budget curves, the decode-step frontier, and the
    top-k decode sweep (Exp 9).

    Left: analytical prefill-average budget vs context length for the four
    operators in both geometries (solid = paper-scale, dashed = MQAR
    probe). KDA is flat (delta-rule O(1) state); CSA and HCA grow through
    the window/scan terms. Middle: the marginal (per-position) view — the
    elements ONE query reads at decode position t; averaging this curve
    over the prefill reproduces the left panel exactly. Right: measured
    per-token decode latency of the fused hybrid vs its analytical
    decode-step budget for swept ``csa_topk`` values (annotated); stale
    result files without the decode-step fields fall back to the prefill
    average.
    """
    rows = load('exp9_budget_frontier.json')
    curves = [r for r in rows if r.get('section') == 'budget_curve'
              and r.get('row_kind') == 'op' and r.get('T')]
    marginal = [r for r in rows
                if r.get('section') == 'decode_budget_curve'
                and r.get('row_kind') == 'op' and 't' in r]
    decode = [r for r in rows if r.get('section') == 'decode_sweep'
              and r.get('error') is None
              and r.get('mean_decode_ms_per_token')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8),
                             constrained_layout=True)
    _budget_curve_panel(
        axes[0], curves, 'T', 'Context length T (tokens)',
        'Prefill-average read budget (lower bound)',
        'KDA: flat (delta-rule O(1) state)\n'
        'CSA: selected gather (capped) + O(T) indexer scan\n'
        'Not profiled traffic; q-side projections excluded.')
    _budget_curve_panel(
        axes[1], marginal, 't', 'Decode position t (context already seen)',
        'Decode-step marginal budget (one query)',
        'Same model as the left panel, evaluated at ONE position:\n'
        't = T is the first token decoded after a T-token prefill.\n'
        'Mean over t < T equals the prefill average.')
    ax = axes[2]
    if decode:
        def _budget_key(r):
            return ('decode_step_read_elements_per_query'
                    if 'decode_step_read_elements_per_query' in r
                    else 'total_read_elements_per_query')

        prefill_vals = sorted({r['prefill_len'] for r in decode})
        cmap = plt.get_cmap('viridis')
        for i, plen in enumerate(prefill_vals):
            pts = [r for r in decode if r['prefill_len'] == plen]
            pts.sort(key=lambda r: r['csa_topk'])
            ax.plot([p[_budget_key(p)] for p in pts],
                    [p['mean_decode_ms_per_token'] for p in pts],
                    'o-', color=cmap(0.15 + 0.7 * i / max(1, len(prefill_vals) - 1)),
                    ms=5, lw=1.2, label=f'prefill L={plen}')
            for p in pts:
                ax.annotate(f"k={p['csa_topk']}",
                            (p[_budget_key(p)],
                             p['mean_decode_ms_per_token']),
                            fontsize=6, xytext=(3, 3),
                            textcoords='offset points', color='0.35')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Analytical decode-step budget '
                      '(elements/query/layer)')
        ax.set_ylabel('Measured decode ms per token')
        ax.set_title('Top-k budget vs measured decode latency', fontsize=11)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=7)
        ax.text(0.02, 0.96,
                'Budget is indexer-scan dominated;\n'
                'cutting top-k alone moves little.',
                transform=ax.transAxes, fontsize=6.5, color='0.4',
                ha='left', va='top')
    else:
        ax.text(0.5, 0.5, 'no decode-sweep rows', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='0.4')
        ax.set_axis_off()
    fig.suptitle('Exp 9: KV read-budget frontier (prefill average | '
                 'decode marginal | top-k sweep; delta-rule O(1) state, '
                 'hybrid ratio)', fontsize=12)
    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_budget_frontier.pdf'),
                    dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_budget_frontier.png'),
                    dpi=150)
    finally:
        plt.close(fig)
    print('Saved figures/fig_budget_frontier.pdf')


def fig_quality_latency():
    """Quality-vs-latency scatter (framework design: the cost side of the
    quality/cost frontier).

    Joins Exp 4's per-operator MQAR accuracy with the matching Exp 2
    prefill latency (at the largest measured T, end-to-end single-layer
    rows where available) and Exp 6's per-token decode latency (at the
    largest prefill length). Each panel degrades independently when its
    results file is absent. The point is the FRONTIER (does the extra
    accuracy of operator X cost more latency than operator Y's?), not the
    individual dots, so both axes are annotated with their measurement
    boundary caveats from the README's fairness notes.
    """
    quality = [r for r in load('exp4_mqar.json')
               if isinstance(r, dict) and 'error' not in r]
    if not quality:
        print('[fig_quality_latency] no exp4 quality rows; skipping.')
        return
    bench = [r for r in load('exp2_benchmark.json')
             if isinstance(r, dict) and 'error' not in r]
    decoding = [r for r in load('exp6_decoding.json')
                if isinstance(r, dict) and 'error' not in r]

    # exp4 op name -> exp2 row name (single-layer end-to-end boundary
    # preferred; kda's reference recurrence row stands in for the kda op).
    # rwkv7 joins only when an Exp 2 rwkv7 row exists (run_benchmark has
    # not gained an rwkv7 row yet; the join is skipped gracefully).
    bench_name = {'softmax': 'softmax', 'kda': 'kda_rec',
                  'mamba2': 'mamba2_layer', 'rwkv7': 'rwkv7_layer',
                  'csa': 'csa', 'hca': 'hca'}
    decode_name = {'softmax': 'softmax', 'kda': 'kda', 'mamba2': 'mamba2',
                   'rwkv7': 'rwkv7', 'csa': 'csa', 'hca': 'hca'}

    # Accuracy per op (n_kv=1 rows; the canonical probe setting).
    acc_by_op = {}
    for r in quality:
        if r.get('n_kv') == 1 and r.get('mean_acc') is not None:
            acc_by_op.setdefault(r['op'], r['mean_acc'])

    panels = []
    if bench:
        max_t = max(r['T'] for r in bench)
        lat_by_op = {}
        for r in bench:
            if r['T'] == max_t and r.get('time_ms') is not None:
                lat_by_op.setdefault(r['op'], r['time_ms'])
        panels.append(('prefill', lat_by_op, bench_name,
                       f'prefill latency @ T={max_t} (ms, Exp 2)'))
    if decoding:
        max_plen = max(r['prefill_len'] for r in decoding)
        dec_by_op = {}
        for r in decoding:
            if (r['prefill_len'] == max_plen
                    and r.get('median_decode_ms_per_token') is not None):
                dec_by_op.setdefault(r['op'],
                                     r['median_decode_ms_per_token'])
        panels.append(('decode', dec_by_op, decode_name,
                       f'decode latency per token @ prefill={max_plen} '
                       f'(ms, Exp 6)'))
    if not panels:
        print('[fig_quality_latency] no latency rows; skipping.')
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.6),
                             squeeze=False, constrained_layout=True)
    colors = {'softmax': '#767676', 'kda': '#1f77b4', 'mamba2': '#d62728',
              'rwkv7': '#ff7f0e', 'csa': '#2ca02c', 'hca': '#9467bd'}
    for ax, (kind, lat, name_map, xlabel) in zip(axes[0], panels):
        plotted = False
        for op, acc in acc_by_op.items():
            row_name = name_map.get(op)
            lat_ms = lat.get(row_name)
            if lat_ms is None:
                continue
            ax.scatter(lat_ms, acc, s=70, color=colors.get(op, '#000000'),
                       zorder=3)
            ax.annotate(op, (lat_ms, acc), textcoords='offset points',
                        xytext=(6, 4), fontsize=9)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, 'no joinable rows', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='#767676')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('MQAR mean accuracy (n_kv=1)')
        ax.set_title(f'quality vs {kind} latency')
        ax.grid(True, alpha=0.3)
    _ensure_figures_dir()
    out = os.path.join(_FIGURES_DIR, 'quality_vs_latency.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[fig_quality_latency] saved {out}')


def fig_quality_cache():
    """Quality-vs-cache scatter (framework design: the memory-budget side).

    Joins Exp 4's per-operator accuracy with Exp 3's full-accounting cache
    bytes at the largest measured T (single-layer rows; the hybrid rows
    have no single-op quality counterpart and are skipped). Annotates that
    the cache axis is ANALYTIC (Exp 3's model, not a profiled number).
    """
    quality = [r for r in load('exp4_mqar.json')
               if isinstance(r, dict) and 'error' not in r]
    cache_rows = [r for r in load('exp3_kv_cache.json')
                  if isinstance(r, dict)]
    if not quality or not cache_rows:
        print('[fig_quality_cache] missing exp4/exp3 rows; skipping.')
        return
    acc_by_op = {}
    for r in quality:
        if r.get('n_kv') == 1 and r.get('mean_acc') is not None:
            acc_by_op.setdefault(r['op'], r['mean_acc'])

    cache_name = {'softmax': 'softmax_gqa', 'kda': 'kda',
                  'mamba2': 'mamba2', 'rwkv7': 'rwkv7', 'csa': 'csa',
                  'hca': 'hca'}
    max_t = max(r['T'] for r in cache_rows)
    bytes_by_op = {}
    for r in cache_rows:
        if (r['T'] == max_t and r.get('accounting_mode') == 'full_accounting'
                and r.get('kv_bytes') is not None):
            bytes_by_op.setdefault(r['op'], r['kv_bytes'])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    colors = {'softmax': '#767676', 'kda': '#1f77b4', 'mamba2': '#d62728',
              'rwkv7': '#ff7f0e', 'csa': '#2ca02c', 'hca': '#9467bd'}
    plotted = False
    for op, acc in acc_by_op.items():
        row_name = cache_name.get(op)
        kv_bytes = bytes_by_op.get(row_name)
        if kv_bytes is None:
            continue
        ax.scatter(kv_bytes / 1e6, acc, s=70,
                   color=colors.get(op, '#000000'), zorder=3)
        ax.annotate(op, (kv_bytes / 1e6, acc), textcoords='offset points',
                    xytext=(6, 4), fontsize=9)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, 'no joinable rows', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#767676')
    ax.set_xlabel(f'runtime cache at T={max_t} (MB, full accounting, '
                  f'analytic)')
    ax.set_ylabel('MQAR mean accuracy (n_kv=1)')
    ax.set_title('quality vs cache budget')
    ax.grid(True, alpha=0.3)
    _ensure_figures_dir()
    out = os.path.join(_FIGURES_DIR, 'quality_vs_cache.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[fig_quality_cache] saved {out}')


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
        fig_quality_latency()
        fig_quality_cache()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp5_ablation.json')):
        fig_ablation()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp6_decoding.json')):
        fig_decoding()
    if os.path.exists(os.path.join(_RESULTS_DIR, 'exp7_roofline.json')):
        fig_roofline()
    if os.path.exists(os.path.join(
            _RESULTS_DIR, 'exp8_mechanism_ablation.json')):
        fig_mechanism_ablation()
    if os.path.exists(os.path.join(
            _RESULTS_DIR, 'exp9_budget_frontier.json')):
        fig_budget_frontier()
    print('\nAll figures generated.')


if __name__ == '__main__':
    main()
