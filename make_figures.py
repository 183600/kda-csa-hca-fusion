"""Generate figures from experiment results.

Handles both the original single-seed result format and the new multi-seed
format (with mean / std / CI95). When multi-seed results are present, bars
are drawn with error bars showing the 95% CI half-width.
"""

from __future__ import annotations

import collections
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(f'results/{name}') as f:
        return json.load(f)


def _has_multiseed(record):
    """True if a record is in the new multi-seed format."""
    return 'mean_acc' in record and 'per_seed' in record


def fig_benchmark():
    """Figure: latency vs sequence length for each operator."""
    data = load('exp2_benchmark.json')
    ops = {}
    for r in data:
        if 'error' in r:
            continue
        ops.setdefault(r['op'], []).append((r['T'], r['time_ms']))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax': 'o-', 'kda_rec': 's-', 'kda_chunk': '^-',
               'csa': 'D-', 'hca': 'v-', 'hybrid': 'p-'}
    labels = {'softmax': 'Softmax attention', 'kda_rec': 'KDA (recurrent)',
              'kda_chunk': 'KDA (chunk)', 'csa': 'CSA', 'hca': 'HCA',
              'hybrid': 'Fused KDA+CSA+HCA'}
    for op, pts in ops.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, markers.get(op, 'o-'), label=labels.get(op, op), markersize=5)
    device = data[0].get('device', 'cpu') if data else 'cpu'
    ax.set_xlabel('Sequence length T')
    ax.set_ylabel(f'Wall-clock latency (ms, {device})')
    ax.set_title('Operator latency vs. sequence length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    # Only call legend() if at least one labeled artist was plotted. Without
    # this guard, an empty / all-error benchmark result file triggers a noisy
    # ``UserWarning: No artists with labels found to put in legend`` from
    # matplotlib. Skipping the legend when there's nothing to show is cleaner
    # than emitting the warning, and the figure remains structurally valid
    # (just empty axes).
    if ops:
        ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig_benchmark.pdf', dpi=150)
    fig.savefig('figures/fig_benchmark.png', dpi=150)
    plt.close(fig)
    print('Saved figures/fig_benchmark.pdf')


def fig_kv_cache():
    """Figure: KV cache size ratio vs GQA baseline.

    Uses the apples-to-apples 5-layer GQA8 baseline (``kv_ratio_vs_gqa_5l``)
    when available, falling back to the 1-layer baseline
    (``kv_ratio_vs_gqa_1l``) for older result files.

    The new ``run_kv_cache.py`` writes TWO records per ``(T, op)`` — one for
    ``accounting_mode='compressed_kv_only'`` and one for
    ``'full_accounting'``. Plotting both into the same series produces two
    y-values per x, which after sorting by T draws vertical segments and
    creates a zigzag artifact on the log-scale plot. We filter to a single
    accounting mode (defaulting to ``full_accounting`` since that is the more
    honest number; the other mode is reported separately in the JSON).
    """
    data = load('exp3_kv_cache.json')
    # Pick the accounting mode: prefer 'full_accounting' (the more honest
    # number), but only if it actually exists in the data. Fall back to
    # 'compressed_kv_only' if that's all that's available. Older result files
    # without the field fall through with mode=None and we plot all rows
    # (legacy). Previously the code unconditionally picked 'full_accounting'
    # whenever ANY record had the field, which would silently produce an empty
    # figure if the JSON only contained 'compressed_kv_only' records (e.g.
    # from a partial run).
    available_modes = {r.get('accounting_mode') for r in data if 'accounting_mode' in r}
    if 'full_accounting' in available_modes:
        mode = 'full_accounting'
    elif 'compressed_kv_only' in available_modes:
        mode = 'compressed_kv_only'
    else:
        mode = None  # legacy: no accounting_mode field, plot all rows
    ops = {}
    for r in data:
        if mode is not None and r.get('accounting_mode', mode) != mode:
            continue
        # Prefer the 5-layer (apples-to-apples) baseline; fall back to 1-layer.
        ratio = r.get('kv_ratio_vs_gqa_5l', r.get('kv_ratio_vs_gqa_1l',
                                                   r.get('kv_ratio_vs_gqa')))
        if ratio is None:
            continue
        ops.setdefault(r['op'], []).append((r['T'], ratio))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax_gqa': 'o-', 'kda': 's-', 'csa': 'D-',
               'hca': 'v-', 'hybrid_kch': 'p-'}
    labels = {'softmax_gqa': 'Softmax GQA8 (5-layer baseline)', 'kda': 'KDA',
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
    fig.tight_layout()
    fig.savefig('figures/fig_kv_cache.pdf', dpi=150)
    fig.savefig('figures/fig_kv_cache.png', dpi=150)
    plt.close(fig)
    print('Saved figures/fig_kv_cache.pdf')


def fig_flops():
    """Figure: prefill FLOPs ratio vs GQA baseline.

    Same accounting-mode filter as ``fig_kv_cache``: ``exp3_kv_cache.json``
    contains two records per ``(T, op)`` (one per ``accounting_mode``) and
    plotting both into the same series creates a zigzag on the log-scale plot.
    We default to ``full_accounting`` for consistency with ``fig_kv_cache``.
    """
    data = load('exp3_kv_cache.json')
    # Same mode-selection logic as fig_kv_cache: prefer 'full_accounting' but
    # fall back to 'compressed_kv_only' if that's all that exists. Prevents
    # an empty figure when the JSON only has one mode.
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
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax_gqa': 'o-', 'kda': 's-', 'csa': 'D-',
               'hca': 'v-', 'hybrid_kch': 'p-'}
    labels = {'softmax_gqa': 'Softmax GQA8 (5-layer baseline)', 'kda': 'KDA',
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
    fig.tight_layout()
    fig.savefig('figures/fig_flops.pdf', dpi=150)
    fig.savefig('figures/fig_flops.png', dpi=150)
    plt.close(fig)
    print('Saved figures/fig_flops.pdf')


def fig_mqar():
    """Figure: MQAR accuracy bar chart (multi-seed with CI95 error bars).

    When ``exp4_mqar.json`` contains records for multiple ``n_kv`` values,
    one figure is produced per ``n_kv`` (saved as
    ``figures/fig_mqar_nkv{n}.pdf/.png``). For backward compatibility, when
    only a single ``n_kv`` group is present the original
    ``figures/fig_mqar.pdf/.png`` filename is also written.
    """
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
    """Plot MQAR accuracy bars for one ``n_kv`` group.

    Handles error rows (where ``mean_acc`` is ``None`` because all seeds
    failed) by skipping them — a partial figure with the surviving
    operators is more useful than crashing the entire figure-generation
    step. The error is logged so the user knows data was dropped.
    """
    ops, means, cis = [], [], []
    chance = 1 / 16
    skipped = 0
    for r in records:
        # Skip error rows (mean_acc is None when all seeds failed).
        # Without this guard, the downstream ``m + c`` and ``f'{m:.3f}'``
        # calls crash with ``TypeError: unsupported operand type(s) for +:
        # 'int' and 'NoneType'`` (or ``ValueError: Unknown format code 'f'
        # for object of type 'str'``), taking down the whole figure step.
        if r.get('mean_acc') is None and r.get('final_acc') is None:
            skipped += 1
            continue
        ops.append(r['op'])
        if _has_multiseed(r):
            means.append(r['mean_acc'])
            cis.append(r.get('ci95_acc', 0.0) or 0.0)
            chance = r.get('chance_acc', chance)
        else:
            # Legacy single-seed format.
            means.append(r['final_acc'])
            cis.append(0.0)
            chance = r.get('chance_acc', chance)
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
    # Number of seeds (fall back to 1 for legacy / empty data).
    # Use the first non-error record so the title reflects the actual
    # seed count of the plotted data, not a failed row's stub.
    ok_records = [r for r in records
                  if r.get('mean_acc') is not None
                  or r.get('final_acc') is not None]
    if ok_records:
        n_seeds = ok_records[0].get('n_seeds', 1)
    else:
        n_seeds = 1

    # Training steps: take from the first per_seed entry when present.
    # Falls back to 100 for missing key, empty per_seed list, or empty data.
    steps = 100
    if ok_records:
        per_seed = ok_records[0].get('per_seed') or []
        if per_seed:
            steps = per_seed[0].get('steps', 100)
    ax.set_title(f'Multi-Query Associative Recall (n_kv={n_kv}, '
                 f'{n_seeds} seeds, {steps} steps)')
    ax.set_ylim(0, max(max(m + c for m, c in zip(means, cis)) * 1.3, 0.35))
    for bar, m, c in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + c + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f'figures/fig_mqar_nkv{n_kv}.pdf', dpi=150)
    fig.savefig(f'figures/fig_mqar_nkv{n_kv}.png', dpi=150)
    if write_legacy_name:
        fig.savefig('figures/fig_mqar.pdf', dpi=150)
        fig.savefig('figures/fig_mqar.png', dpi=150)
    plt.close(fig)
    msg = f'Saved figures/fig_mqar_nkv{n_kv}.pdf'
    if write_legacy_name:
        msg += ' (and fig_mqar.pdf)'
    print(msg)


def fig_ablation():
    """Figure: ablation accuracy and latency (multi-seed with error bars).

    When ``exp5_ablation.json`` contains records for multiple ``n_kv``
    values, one figure is produced per ``n_kv`` (saved as
    ``figures/fig_ablation_nkv{n}.pdf/.png``). For backward compatibility,
    when only a single ``n_kv`` group is present the original
    ``figures/fig_ablation.pdf/.png`` filename is also written.
    """
    path = 'results/exp5_ablation.json'
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
    """Plot ablation accuracy + latency bars for one ``n_kv`` group.

    Handles error rows (where ``mean_acc`` is ``None`` because all seeds
    for that ratio failed) by substituting 0.0 and labeling the bar as
    ``"ERR"``. A partial figure with the failed ratios marked is more
    useful than crashing the entire figure-generation step and losing the
    successful ratios' plots. The error is also logged.

    Previously, a single error row (``mean_acc=None``) crashed the whole
    function at ``max(a + c for a, c in zip(accs, acc_cis))`` with
    ``TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'``,
    because ``r.get('mean_acc', r.get('final_acc', 0.0))`` returns ``None``
    (not the default ``0.0``) when the key exists with value ``None`` —
    ``dict.get`` only falls back to the default when the key is *absent*,
    not when its value is ``None``.
    """
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
        # ``r.get('mean_acc', ...)`` returns None when the key exists with
        # value None (error row), NOT the fallback default. Detect error
        # rows explicitly via the 'error' key or by checking for None.
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
                            or sum(int(x) for x in ratio.split(':')))
            skipped += 1
        else:
            error_flags.append(False)
            ratios.append(ratio)
            accs.append(r.get('mean_acc', r.get('final_acc', 0.0)) or 0.0)
            acc_cis.append(r.get('ci95_acc', 0.0) or 0.0)
            fwds.append(r.get('mean_fwd_ms', r.get('fwd_ms', 0.0)) or 0.0)
            n_params.append(r.get('n_params') or 0)
            n_layers.append(r.get('n_layers')
                            or sum(int(x) for x in ratio.split(':')))

    if skipped:
        print(f'[fig_ablation] {skipped} ratio(s) had errors and are shown '
              f'as 0-height "ERR" bars for n_kv={n_kv}')

    # Guard against the empty-records case (e.g. all records were filtered
    # out before reaching this function, or the function was called with
    # an empty list). Previously the next line
    # ``max(max(a + c for a, c in zip(accs, acc_cis)) * 1.3, 0.2)`` would
    # raise ``ValueError: max() iterable argument is empty`` because the
    # inner generator produces nothing when accs is empty. Skip plotting
    # entirely and return — there is nothing to draw.
    if not ratios:
        print(f'Skipping ablation figure for n_kv={n_kv} (no records to plot)')
        return

    # Use a taller figure (5.0in vs the original 4.5in) to accommodate
    # the two-line x-tick labels (e.g. ``"3:1:1\n(5L, 26852p)"``) and the
    # suptitle. See the ``subplots_adjust`` comment below for why we don't
    # use ``tight_layout``.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(ratios))

    # Accuracy with CI95 error bars. Error rows get a distinct color so
    # they are visually identifiable rather than silently blending in.
    bar_colors = ['#C44E52' if e else '#4C72B0' for e in error_flags]
    bars1 = ax1.bar(x, accs, yerr=acc_cis, capsize=4, color=bar_colors,
                    error_kw={'linewidth': 1.3, 'ecolor': '#333'})
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r}\n({l}L, {p}p)' for r, l, p in zip(ratios, n_layers, n_params)],
                        rotation=0, fontsize=8)
    ax1.set_ylabel('MQAR accuracy (mean +/- CI95)')
    ax1.set_title('Accuracy vs. KDA:CSA:HCA ratio')
    ax1.axhline(1/16, color='gray', linestyle='--', alpha=0.5, label='chance (1/16)')
    for bar, acc, is_err in zip(bars1, accs, error_flags):
        label = 'ERR' if is_err else f'{acc:.3f}'
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 label, ha='center', va='bottom', fontsize=9)
    ax1.legend(fontsize=8)
    # Compute the y-axis upper bound safely. ``accs`` may be all zeros
    # (all-error case) — the previous ``max(max(...), 0.2)`` would still
    # work in that case (the inner max returns 0.0, then 0.2 wins), but
    # the empty-list case is now handled by the early return above.
    # Use a defensive ``or 0.0`` so a None in accs (should not happen
    # here, but cheap to guard) does not crash the multiplication.
    acc_upper = max((float(a) + float(c) for a, c in zip(accs, acc_cis)),
                    default=0.0) * 1.3
    ax1.set_ylim(0, max(acc_upper, 0.2))

    # Latency.
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

    # Place the suptitle and use ``subplots_adjust`` (not ``tight_layout``) to
    # manually control margins. ``tight_layout`` emits
    # ``UserWarning: Tight layout not applied. The bottom and top margins
    # cannot be made large enough to accommodate all Axes decorations.``
    # when the figure has both a suptitle AND two-line x-tick labels (e.g.
    # ``"3:1:1\n(5L, 26852p)"``), because its iterative constraint solver
    # cannot converge within the default margin bounds. ``subplots_adjust``
    # sets the margins directly (no solver), so it never warns. The values
    # are tuned to leave room for: top=0.91 (suptitle), bottom=0.18
    # (two-line x-tick labels), left=0.08 (y-axis label), right=0.97,
    # wspace=0.3 (gap between the two subplots).
    fig.suptitle(f'Ablation: ratio trade-off (n_kv={n_kv}, multi-seed). '
                 '4:1:1 has 6 layers vs 3:1:1 has 5 — depth confound noted in paper.',
                 fontsize=9, y=0.98)
    fig.subplots_adjust(top=0.91, bottom=0.18, left=0.08, right=0.97, wspace=0.3)
    fig.savefig(f'figures/fig_ablation_nkv{n_kv}.pdf', dpi=150,
                bbox_inches='tight')
    fig.savefig(f'figures/fig_ablation_nkv{n_kv}.png', dpi=150,
                bbox_inches='tight')
    if write_legacy_name:
        fig.savefig('figures/fig_ablation.pdf', dpi=150, bbox_inches='tight')
        fig.savefig('figures/fig_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    msg = f'Saved figures/fig_ablation_nkv{n_kv}.pdf'
    if write_legacy_name:
        msg += ' (and fig_ablation.pdf)'
    print(msg)


def fig_decoding():
    """Figure: per-token decoding latency vs cached context length."""
    path = 'results/exp6_decoding.json'
    if not os.path.exists(path):
        print('Skipping decoding figure (no results yet)')
        return
    data = load('exp6_decoding.json')
    ops = {}
    for r in data:
        if 'error' in r:
            continue
        ops.setdefault(r['op'], []).append(
            (r['prefill_len'], r['median_decode_ms_per_token']))
    if not ops:
        print('Skipping decoding figure (no successful runs)')
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {'softmax': 'o-', 'kda': 's-'}
    labels = {'softmax': 'Softmax attention (growing KV cache)',
              'kda': 'KDA recurrent (O(1) state)'}
    for op, pts in ops.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, markers.get(op, 'o-'), label=labels.get(op, op), markersize=7)
    device = data[0].get('device', 'cpu') if data else 'cpu'
    ax.set_xlabel('Cached context length (tokens)')
    ax.set_ylabel(f'Per-token decode latency (ms, {device})')
    ax.set_title('Decoding latency vs. cached context length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/fig_decoding.pdf', dpi=150)
    fig.savefig('figures/fig_decoding.png', dpi=150)
    plt.close(fig)
    print('Saved figures/fig_decoding.pdf')


def fig_architecture():
    """Figure: schematic of the fused hybrid architecture."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Draw the layer stack
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
    # Input/output arrows
    ax.annotate('', xy=(0.5, 2.0), xytext=(0.0, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(0.0, 2.3, 'x', fontsize=11)
    ax.annotate('', xy=(8.2, 2.0), xytext=(7.7, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(8.2, 2.3, 'y', fontsize=11)
    # Residual connections
    for x, _, _ in blocks:
        ax.annotate('', xy=(x + 1.2, 1.2), xytext=(x, 1.2),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))
    # Legend
    ax.text(4.5, 0.5, 'KDA: fine-grained gated delta rule (linear, O(1) state)\n'
            'CSA: compressed sparse attention (top-k retrieval)\n'
            'HCA: heavily compressed attention (dense global context)',
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Fused KDA+CSA+HCA hybrid attention (3:1:1 layout)', fontsize=12)
    fig.tight_layout()
    fig.savefig('figures/fig_architecture.pdf', dpi=150)
    fig.savefig('figures/fig_architecture.png', dpi=150)
    plt.close(fig)
    print('Saved figures/fig_architecture.pdf')


def main():
    os.makedirs('figures', exist_ok=True)
    fig_architecture()
    # exp2 may not exist if benchmark was skipped.
    if os.path.exists('results/exp2_benchmark.json'):
        fig_benchmark()
    if os.path.exists('results/exp3_kv_cache.json'):
        fig_kv_cache()
        fig_flops()
    if os.path.exists('results/exp4_mqar.json'):
        fig_mqar()
    if os.path.exists('results/exp5_ablation.json'):
        fig_ablation()
    if os.path.exists('results/exp6_decoding.json'):
        fig_decoding()
    print('\nAll figures generated.')


if __name__ == '__main__':
    main()
