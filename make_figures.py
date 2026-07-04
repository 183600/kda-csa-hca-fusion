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
    """
    data = load('exp3_kv_cache.json')
    ops = {}
    for r in data:
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
    """Figure: prefill FLOPs ratio vs GQA baseline."""
    data = load('exp3_kv_cache.json')
    ops = {}
    for r in data:
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
    ops, means, cis = [], [], []
    chance = 1 / 16
    for r in records:
        ops.append(r['op'])
        if _has_multiseed(r):
            means.append(r['mean_acc'])
            cis.append(r.get('ci95_acc', 0.0))
            chance = r.get('chance_acc', chance)
        else:
            # Legacy single-seed format.
            means.append(r['final_acc'])
            cis.append(0.0)
            chance = r.get('chance_acc', chance)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(ops, means, yerr=cis, capsize=5,
                  color=colors[:len(ops)],
                  error_kw={'linewidth': 1.5, 'ecolor': '#333'})
    ax.axhline(chance, color='gray', linestyle='--', alpha=0.7,
               label=f'Chance ({chance:.3f})')
    ax.set_ylabel('MQAR accuracy (mean over seeds, 95% CI)')
    # Number of seeds (fall back to 1 for legacy / empty data).
    if records:
        n_seeds = records[0].get('n_seeds', 1)
    else:
        n_seeds = 1

    # Training steps: take from the first per_seed entry when present.
    # Falls back to 100 for missing key, empty per_seed list, or empty data.
    steps = 100
    if records:
        per_seed = records[0].get('per_seed') or []
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
    ratios = [r['ratio'] for r in records]
    accs = [r.get('mean_acc', r.get('final_acc', 0.0)) for r in records]
    acc_cis = [r.get('ci95_acc', 0.0) for r in records]
    fwds = [r.get('mean_fwd_ms', r.get('fwd_ms', 0.0)) for r in records]
    n_params = [r.get('n_params', 0) for r in records]
    n_layers = [r.get('n_layers', sum(int(x) for x in r['ratio'].split(':'))) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(ratios))

    # Accuracy with CI95 error bars.
    bars1 = ax1.bar(x, accs, yerr=acc_cis, capsize=4, color='#4C72B0',
                    error_kw={'linewidth': 1.3, 'ecolor': '#333'})
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r}\n({l}L, {p}p)' for r, l, p in zip(ratios, n_layers, n_params)],
                        rotation=0, fontsize=8)
    ax1.set_ylabel('MQAR accuracy (mean +/- CI95)')
    ax1.set_title('Accuracy vs. KDA:CSA:HCA ratio')
    ax1.axhline(1/16, color='gray', linestyle='--', alpha=0.5, label='chance (1/16)')
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, max(max(a + c for a, c in zip(accs, acc_cis)) * 1.3, 0.2))

    # Latency.
    bars2 = ax2.bar(x, fwds, color='#55A868')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{r}\n({l}L)' for r, l in zip(ratios, n_layers)],
                        rotation=0, fontsize=8)
    ax2.set_ylabel('Forward latency (ms)')
    ax2.set_title('Latency vs. KDA:CSA:HCA ratio')
    for bar, fwd in zip(bars2, fwds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{fwd:.1f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f'Ablation: ratio trade-off (n_kv={n_kv}, multi-seed). '
                 '4:1:1 has 6 layers vs 3:1:1 has 5 — depth confound noted in paper.',
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(f'figures/fig_ablation_nkv{n_kv}.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(f'figures/fig_ablation_nkv{n_kv}.png', dpi=150, bbox_inches='tight')
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
