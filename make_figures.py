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
# Configure CJK font fallback so any non-ASCII label renders correctly
# instead of producing tofu boxes. English labels today, but future-proofs
# the file against later Chinese annotations. DejaVu Sans is the final
# fallback for symbols the CJK fonts lack.
import matplotlib.font_manager as _fm
for _fp in ('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    try:
        _fm.fontManager.addfont(_fp)
    except (FileNotFoundError, OSError):
        pass
import matplotlib.pyplot as plt
# Font priority order rationale:
#   1. DejaVu Sans — has full coverage of the mathtext glyphs that the log-scale
#      tick formatter needs (notably U+2212 MINUS SIGN, which appears in
#      ``$10^{-3}$`` style labels). WITHOUT DejaVu Sans first, matplotlib's
#      mathtext engine emits "Font 'default' does not have a glyph for '\u2212',
#      substituting with a dummy symbol" for every negative-exponent tick label,
#      and the rendered PDF/PNG has a broken (dummy) minus sign.
#   2. Noto Sans SC / WenQuanYi Zen Hei — CJK fallback for any future Chinese
#      annotations. Listed AFTER DejaVu Sans so that mathtext rendering always
#      resolves to a font with U+2212; regular text still falls through to the
#      CJK fonts for characters DejaVu Sans lacks (per-glyph fallback in
#      matplotlib 3.6+).
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC',
                                  'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Anchor every relative path to THIS FILE's directory by default so the
# script works regardless of the current working directory. Previously
# ``load()`` opened ``'results/{name}'`` (relative to cwd), so running the
# script from a different directory (e.g. ``python ../make_figures.py``)
# failed with ``FileNotFoundError`` on every figure.
#
# On Kaggle, however, ``run_all.py`` does ``os.chdir(out_root)`` (typically
# ``/kaggle/working``) before running the experiments, so the *experiment
# scripts* write their JSONs to ``/kaggle/working/results/`` while this
# module's ``_ROOT`` still points at the read-only ``/kaggle/input/...``.
# Without honoring an override, every data-driven figure was silently
# skipped (existence checks failed) and ``fig_architecture`` raised
# ``OSError [Errno 30]``. We therefore let the caller redirect both
# ``_RESULTS_DIR`` and ``_FIGURES_DIR`` via env vars (set by ``run_all.py``
# to ``{out_root}/results`` / ``{out_root}/figures`` on Kaggle).
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
    # Unwrap the envelope form. A top-level dict with a ``results`` key
    # holding a list unambiguously signals the envelope introduced by the
    # P0-1 fix. All current legacy result files are bare arrays (the
    # kv_cache / benchmark / correctness / ablation / decoding files all
    # serialize Python lists), so this branch only fires for the new
    # envelope form and leaves every other file's behavior unchanged.
    if isinstance(data, dict) and 'results' in data and isinstance(
            data['results'], list):
        return data['results']
    return data


def _ensure_figures_dir():
    """Ensure ``figures/`` exists before any savefig call.

    Previously ``os.makedirs('figures', ...)`` lived only in ``main()``,
    so calling any ``fig_*`` function directly (e.g. from a notebook or
    test) raised ``FileNotFoundError`` because the directory did not
    exist yet. Cheaper to call exist_ok=True here than to thread a
    directory argument through every fig_* signature.
    """
    os.makedirs(_FIGURES_DIR, exist_ok=True)


def _ratio_layers(ratio: str) -> int:
    """Return the total layer count encoded by a ratio string like '3:1:1'.

    Returns 0 if the string contains any non-numeric component (e.g.
    'baseline', '3:1:1 (5L)'), instead of raising ValueError. The previous
    ``sum(int(x) for x in ratio.split(':'))`` would crash the entire
    ablation figure if a future code path stored a non-numeric ratio
    string in the JSON.
    """
    try:
        return sum(int(x) for x in ratio.replace(' ', '').split(':'))
    except ValueError:
        return 0


def _has_multiseed(record):
    """True if a record is in the new multi-seed format."""
    return 'mean_acc' in record and 'per_seed' in record


def fig_benchmark():
    """Figure: latency vs sequence length for each operator.

    P2 fix — SEPARATE subplots by compute boundary (was: all ops in one plot).

    The benchmark measures three DIFFERENT compute boundaries:

      * **core** (softmax, kda_rec, kda_chunk): only the attention /
        recurrence kernel is timed, with q/k/v (and g/beta for KDA)
        pre-projected OUTSIDE the timed region.
      * **end_to_end_single_layer** (csa, hca): timing starts from the hidden
        state ``H`` and includes ALL input projections + compression +
        indexer + attention + output projection for ONE layer.
      * **end_to_end_multi_layer** (hybrid): a full 5-layer stack with
        LayerNorm, projections, attention, and state management.

    These numbers are NOT directly comparable as "operator latency" because
    the compute boundary differs — a single-axis plot with all 6 lines invites
    exactly the misleading cross-boundary comparison the issue calls out
    (e.g. "softmax is 10x faster than hybrid" ignores that hybrid is 5 layers
    end-to-end while softmax is just the core kernel).

    The fix splits the figure into **separate subplots** grouped by compute
    boundary, so only ops sharing the same boundary appear on the same axes.
    A prominent suptitle warns that cross-subplot comparison is not
    apples-to-apples. Within each subplot the comparison IS fair (same
    boundary, same measurement methodology).
    """
    data = load('exp2_benchmark.json')
    # Fallback boundary mapping for older result files that predate the
    # ``compute_boundary`` field (added by the P1-1 fix). Without this
    # fallback, every op would be grouped under 'unknown' and the split-by-
    # boundary subplot layout would collapse back to the single-plot
    # anti-pattern the P2 fix is specifically meant to eliminate.
    _OP_BOUNDARY_FALLBACK = {
        'softmax': 'core',
        'kda_rec': 'core',
        'kda_chunk': 'core',
        'csa': 'end_to_end_single_layer',
        'hca': 'end_to_end_single_layer',
        'hybrid': 'end_to_end_multi_layer',
    }
    # Collect (T, time_ms, compute_boundary, n_layers) per op.
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
    # Group ops by compute boundary so we only plot same-boundary ops together.
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
    # Order subplots: core first, then single-layer, then multi-layer.
    boundary_order = ['core', 'end_to_end_single_layer',
                      'end_to_end_multi_layer', 'unknown']
    ordered_boundaries = [b for b in boundary_order if b in boundary_groups]

    device = next((r.get('device', 'cpu') for r in data if 'error' not in r),
                  'cpu')

    n_subplots = len(ordered_boundaries)
    if n_subplots == 0:
        print('Skipping fig_benchmark (no data to plot)')
        return

    # Use a tall figure with one subplot per boundary group, sharing the x
    # axis so the reader can visually align sequence lengths across groups.
    # Each subplot has its OWN y-axis (log scale) because the absolute
    # latencies differ by orders of magnitude across boundaries — sharing a
    # y-axis would compress the core subplot to a flat line.
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

    # Prominent suptitle: warn that cross-subplot comparison is unfair.
    fig.suptitle(
        'Operator latency vs. sequence length — split by compute boundary\n'
        'WARNING: each subplot uses a DIFFERENT measurement boundary; '
        'cross-subplot comparison is NOT apples-to-apples',
        fontsize=10, fontweight='bold')

    _ensure_figures_dir()
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_benchmark.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_benchmark.png'), dpi=150)
    finally:
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
    # Empty-data guard: if filtering left no plottable series, skip the
    # figure entirely. An empty-axes PDF on disk is indistinguishable from a
    # real one and confuses downstream consumers (mirrors the guard in
    # ``fig_decoding`` / ``_plot_mqar_group``).
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
    # try/finally around savefig so a savefig failure (disk full, read-only
    # Kaggle input dir, etc.) does not leak the figure. Without the finally,
    # ``plt.close(fig)`` would be skipped on exception and the half-built
    # figure would persist into the next fig_* call. Mirrors the pattern in
    # the other fig_* functions.
    try:
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_kv_cache.pdf'), dpi=150)
        fig.savefig(os.path.join(_FIGURES_DIR, 'fig_kv_cache.png'), dpi=150)
    finally:
        plt.close(fig)
    print(f'Saved {os.path.join(_FIGURES_DIR, "fig_kv_cache.pdf")}')


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
    # Empty-data guard (mirrors fig_kv_cache).
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
            # ci95_acc is None when only one seed survived (see
            # run_quality.py::train_multi_seed / run_ablation.py::
            # eval_layout_multi_seed). Coercing None -> 0.0 (the previous
            # behaviour) produces a ZERO-WIDTH error bar that visually
            # implies perfect precision, contradicting the upstream intent
            # of writing None to signal "uncertainty is maximal / n/a".
            # Use NaN instead: matplotlib skips the error bar entirely
            # (visually: no error bar), which honestly reflects that the
            # CI is undefined rather than zero.
            _ci = r.get('ci95_acc')
            cis.append(float('nan') if _ci is None else float(_ci))
            # Same ``or`` guard for chance_acc: a record with
            # ``chance_acc: None`` (e.g. partially-failed seed wrote null)
            # would set ``chance = None`` and crash ``ax.axhline(None, ...)``
            # with TypeError. The function-local default ``chance = 1 / 16``
            # (set at the top of this function) is preserved when the value
            # is missing OR None.
            chance = r.get('chance_acc') or chance
        else:
            # Legacy single-seed format.
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

    # Training steps: take from the first per_seed entry of a NON-softmax
    # record when available. ``run_quality.py`` appends records in the fixed
    # order ``['softmax', 'kda', 'csa', 'hca']``, and softmax may use a
    # DIFFERENT step budget (``MQAR_SOFTMAX_STEPS``) than the other ops
    # (``MQAR_STEPS``). Using ``ok_records[0]`` (which is softmax whenever it
    # succeeds) would put softmax's step count in the title, mislabeling the
    # budget that kda/csa/hca actually trained for. Prefer a non-softmax
    # record so the title reflects the canonical ``MQAR_STEPS`` budget; fall
    # back to the first record (softmax) only if no non-softmax record
    # succeeded.
    title_rec = next((r for r in ok_records if r.get('op') != 'softmax'),
                     ok_records[0] if ok_records else None)
    # Also pick up softmax's step count so we can show both when they differ.
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
    # If softmax used a different step budget, surface BOTH in the title so
    # the figure is not misleading (the README's Fairness notes #2 explicitly
    # requires this case to be "labelled separately").
    if softmax_steps is not None and softmax_steps != steps:
        title_steps = f'softmax={softmax_steps}, others={steps}'
    else:
        title_steps = str(steps)
    # Round 9 audit: surface statistical validity in the title, mirroring
    # the ablation figure's suptitle warning (lines ~794-802). The README's
    # Fairness notes #3 explicitly says ``conclusions_valid`` is "the
    # authoritative signal — ``mean_acc`` alone is misleading." Without this
    # warning, a reader of fig_mqar could draw strong structural conclusions
    # from underpowered near-chance data — the exact failure mode the
    # ablation figure was patched to prevent.
    _valid = all(r.get('conclusions_valid', True) for r in ok_records)
    _n_sig = sum(1 for r in ok_records if r.get('significant_bonferroni'))
    _n_total = len(ok_records)
    validity_note = ''
    if not _valid:
        validity_note = (f' WARNING: conclusions_valid=False '
                         f'({_n_sig}/{_n_total} Bonferroni-sig). '
                         f'Treat as exploratory, not confirmatory.')
    ax.set_title(f'Multi-Query Associative Recall (n_kv={n_kv}, '
                 f'{n_seeds} seeds, {title_steps} steps){validity_note}',
                 fontsize=8)
    # Defensive ``default=0.0`` on the inner ``max`` so an empty ``means``
    # list (e.g. all records were error rows but somehow bypassed the early
    # return) yields 0.0 instead of raising ``ValueError: max() iterable
    # argument is empty``. Mirrors the fix in ``_plot_ablation_group``.
    # NaN-safe upper bound: a NaN ci (single-seed record) would propagate
    # through ``m + c`` and turn the whole max into NaN, breaking the ylim.
    # Filter to finite values only; if everything is NaN/empty, fall back
    # to 0.0 so the ``max(acc_upper, 0.35)`` floor still applies.
    _finite_upper = [m + c for m, c in zip(means, cis)
                     if math.isfinite(m + c)]
    acc_upper = (max(_finite_upper, default=0.0)) * 1.3
    ax.set_ylim(0, max(acc_upper, 0.35))
    for bar, m, c in zip(bars, means, cis):
        # Use 0.0 for the text y-offset when c is NaN (single-seed record)
        # so the accuracy label still renders at the top of the bar.
        _c_offset = 0.0 if not math.isfinite(c) else c
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + _c_offset + 0.005,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    ax.legend(fontsize=9)
    _ensure_figures_dir()
    # Use _FIGURES_DIR (env-overridable) instead of a relative path so the
    # file lands in the same directory as every other figure and respects
    # the Kaggle output-dir override set by run_all.py.
    # try/finally around savefig so a savefig failure (disk full, read-only
    # Kaggle input dir, etc.) does not leak the figure. Mirrors the pattern
    # in ``fig_kv_cache`` / ``fig_flops`` / ``fig_benchmark``.
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
    """Figure: ablation accuracy and latency (multi-seed with error bars).

    When ``exp5_ablation.json`` contains records for multiple ``n_kv``
    values, one figure is produced per ``n_kv`` (saved as
    ``figures/fig_ablation_nkv{n}.pdf/.png``). For backward compatibility,
    when only a single ``n_kv`` group is present the original
    ``figures/fig_ablation.pdf/.png`` filename is also written.
    """
    # Use _RESULTS_DIR (env-overridable) for the existence check, not a
    # relative path. The ``load()`` call below uses _RESULTS_DIR; checking a
    # different path meant the figure was silently skipped whenever the
    # script was run from a different cwd (e.g. via run_all.py on Kaggle).
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
                            or _ratio_layers(ratio))
            skipped += 1
        else:
            error_flags.append(False)
            ratios.append(ratio)
            accs.append(r.get('mean_acc', r.get('final_acc', 0.0)) or 0.0)
            # ci95_acc is None when only one seed survived (see comment in
            # _plot_mqar_group). Use NaN so the error bar is omitted rather
            # than rendered as a misleading zero-width bar.
            _ci = r.get('ci95_acc')
            acc_cis.append(float('nan') if _ci is None else float(_ci))
            fwds.append(r.get('mean_fwd_ms', r.get('fwd_ms', 0.0)) or 0.0)
            n_params.append(r.get('n_params') or 0)
            n_layers.append(r.get('n_layers')
                            or _ratio_layers(ratio))

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
    # Use the chance_acc carried by the records (defaulting to 1/16) so the
    # dashed reference line stays correct if the task vocab changes. The
    # previous code hardcoded ``1/16`` here, which would silently lie if the
    # upstream experiment ever changed VOCAB. Mirrors the fix in
    # run_ablation.py (chance = 1.0 / VOCAB).
    #
    # Pick the first NON-ERROR record's chance_acc — ``records[0]`` may be
    # an error row whose chance_acc is None, which would crash
    # ``axhline(None, ...)`` with TypeError. Use ``or 1/16`` so an explicit
    # None (or missing key) falls back to the default. Mirrors the None-safe
    # read in _plot_mqar_group.
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
    # Compute the y-axis upper bound safely. ``accs`` may be all zeros
    # (all-error case) — the previous ``max(max(...), 0.2)`` would still
    # work in that case (the inner max returns 0.0, then 0.2 wins), but
    # the empty-list case is now handled by the early return above.
    # Use a defensive ``or 0.0`` so a None in accs (should not happen
    # here, but cheap to guard) does not crash the multiplication.
    # NaN-safe upper bound (mirrors _plot_mqar_group): a NaN ci (single-
    # seed record) would propagate through ``a + c`` and turn the whole
    # max into NaN. Filter to finite values only.
    _finite_upper = [float(a) + float(c) for a, c in zip(accs, acc_cis)
                     if math.isfinite(float(a) + float(c))]
    acc_upper = (max(_finite_upper, default=0.0)) * 1.3
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
    # Explicit y-axis upper bound so the per-bar text labels (placed at
    # ``bar.get_height() + 0.5``) don't get clipped at the top of the axes.
    # Without this, matplotlib auto-scales to ~max(fwds)*1.05, and the 0.5-ms
    # offset pushes the tallest label above the axes box (invisible in the
    # saved PDF/PNG). Use a relative offset for the upper bound so it scales
    # with the data (a fixed ``+0.5`` would be too tight at large latencies
    # and too loose at small ones).
    fwd_max = max((float(f) for f in fwds), default=0.0)
    ax2.set_ylim(0, max(fwd_max * 1.25, 1.0))

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
    # Compute the depth-confound note dynamically from the records rather
    # than hardcoding "4:1:1 has 6 layers vs 3:1:1 has 5". The hardcoded
    # text was correct only for the default ablation set
    # [(3,1,1),(4,1,1),(2,1,1),(1,1,1),(3,0,1),(3,1,0),(0,1,1)]; a user
    # running a custom subset (e.g. only (2,1,1) vs (1,1,1)) would get a
    # misleading suptitle. We identify the max-depth and min-depth ratios
    # and only mention the confound when they actually differ.
    # P4 fix — surface statistical validity in the suptitle. The issue
    # flagged that the ablation had only 3 seeds, accuracies near chance,
    # and ALL significant_bonferroni=False. We now check the
    # ``conclusions_valid`` flag (written by run_ablation.py) and add a
    # prominent warning to the suptitle so the figure does not invite
    # strong structural conclusions from underpowered data.
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
            # Find the first ratio at max depth and the first at min depth.
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
    # Drop ``bbox_inches='tight'`` here (issue A5/A10): it overrides the
    # manual subplots_adjust margins AND makes the saved figure size
    # inconsistent with every other figure (which doesn't pass it). The
    # subplots_adjust call above is what controls the layout for this fig.
    _ensure_figures_dir()
    # try/finally around savefig so a savefig failure (disk full, read-only
    # Kaggle input dir, etc.) does not leak the figure. Without the finally,
    # ``plt.close(fig)`` would be skipped on exception and the half-built
    # figure would persist into the next fig_* call. Mirrors the pattern in
    # ``fig_kv_cache`` / ``fig_flops`` / ``fig_benchmark``.
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
        # Guard against None median_decode_ms_per_token (half-written row).
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
    ax.set_xlabel('Cached context length (tokens)')
    ax.set_ylabel(f'Per-token decode latency (ms, {device})')
    ax.set_title('Decoding latency vs. cached context length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    _ensure_figures_dir()
    # try/finally around savefig so a savefig failure (disk full, read-only
    # Kaggle input dir, etc.) does not leak the figure. Mirrors the pattern
    # in ``fig_kv_cache`` / ``fig_flops`` / ``fig_benchmark``.
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
    _ensure_figures_dir()
    # try/finally around savefig so a savefig failure (disk full, read-only
    # Kaggle input dir, etc.) does not leak the figure. Mirrors the pattern
    # in ``fig_kv_cache`` / ``fig_flops`` / ``fig_benchmark``.
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
    # exp2 may not exist if benchmark was skipped.
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
