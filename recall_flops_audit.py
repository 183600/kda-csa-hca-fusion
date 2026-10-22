"""Recall-conditioned FLOPs audit for hybrid linear-attention layouts.

Motivation
----------
The repository already has two separate diagnostic ledgers:

* ``flops_audit`` measures the exact GEMM work of each *layer kind* and
  pins a closed-form reference model against the dispatched operators.
* ``recall_reach`` derives the *exact* long-range reachability of a hybrid
  plan: a planted (key, value) pair is exactly recoverable only through a
  compressed block that strictly precedes the cue's block, or through the
  query's sliding window. Recurrent memory layers are capacity-limited and
  are deliberately NOT counted as exact paths.

Nothing joins the two ledgers.  As a result, a layout can look cheap in
aggregate FLOPs while almost all of that work sits in layers that cannot
provide a structural exact-recall path at the distance being measured.  This
module makes that allocation visible.  For every placement distance it
reports

* which layers have an exact path (``recall_reach.structural_recall``),
* the GEMM FLOPs of exactly those layers, and
* the resulting ``exact_reachable_flops_fraction`` = exact-path layer FLOPs /
  total stack FLOPs.

The metric is a *structural accounting ratio*, not a latency claim and not a
capability claim: a recurrent layer can still store associations in its
state; it just has no documented guarantee that a single planted pair
survives interference and decay, which is why ``recall_reach`` excludes it
from exact reachability.  Read the ratio together with the placement
coverage (``structural_far_any_exact_fraction``); one is compute, the other
is geometry.

What it adds beyond the layer audit
-----------------------------------
1. **Whole-stack FLOPs.**  Sums the exact per-layer reference model over the
   assembled layout and, optionally, cross-checks it against a real
   ``HybridAttentionStack`` forward under the dispatch counter.  The only
   documented discrepancy is HCA's single-compressed-block value fold; the
   audit reports that bound instead of hiding it.
2. **Distance-conditioned compute.**  A far placement may be reachable by
   two CSA/HCA layers out of ten; the audit reports the FLOPs actually
   residing in those reachable layers for every distance.
3. **Ratio/layout frontier.**  The CLI can compare named layout presets and
   arbitrary KDA:CSA:HCA ratios at a fixed depth, which exposes whether extra
   compute is being allocated to layers that cannot help exact recall.

Usage::

    python3 recall_flops_audit.py --seq-len 128
    python3 recall_flops_audit.py --ratios 3:1:1,1:2:2,1:0:0,0:1:1 \
        --total-layers 10 --json /tmp/recall_flops.json
    python3 -m pytest -q run_correctness.py -k recall_flops_audit

Tests: ``run_correctness.py::test_recall_flops_audit`` and
``run_correctness.py::test_recall_distance_buckets_match_structural_window``.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flops_audit as fa  # noqa: E402
from kaggle_setup import (  # noqa: E402
    capture_provenance,
    write_results_json,
)
from ops_fused import HybridAttentionStack, HybridConfig  # noqa: E402
from recall_reach import (  # noqa: E402
    BLOCK_KINDS,
    LAYOUT_PRESETS,
    plan_from_config,
    structural_recall,
    structural_summary,
)

__all__ = [
    'LAYOUT_PRESETS',
    'audit_layout',
    'audit_layouts',
    'distance_compute_reach',
    'layer_reference_flops',
    'ratio_layout',
]

SUPPORTED_LAYER_KINDS = ('kda', 'mamba2', 'csa', 'hca')
_MEMORY_KINDS = ('kda', 'mamba2')


def _validate_positive_int(name: str, value, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f'{name}={value!r} must be an int (got {type(value).__name__}).')
    if value < minimum:
        raise ValueError(f'{name}={value} must be >= {minimum}.')
    return value


def _validate_layout(layout) -> tuple[str, ...]:
    if isinstance(layout, str):
        raise TypeError(
            'layout must be a sequence of layer kinds, not a string; pass '
            'LAYOUT_PRESETS[name] for a preset.')
    layout = tuple(layout)
    if not layout:
        raise ValueError('layout must be non-empty.')
    for kind in layout:
        if kind not in SUPPORTED_LAYER_KINDS:
            raise ValueError(
                f'layout kind={kind!r} is not auditable; supported kinds are '
                f'{SUPPORTED_LAYER_KINDS}. RWKV-7 memory layers have no '
                f'exact GEMM reference model in flops_audit yet.')
    memory_kinds = [kind for kind in layout if kind in _MEMORY_KINDS]
    if len(set(memory_kinds)) > 1:
        raise ValueError(
            'layout mixes multiple memory routes '
            f'({sorted(set(memory_kinds))}); audit one pure KDA or Mamba-2 '
            'route at a time.')
    return layout


def _validate_cfg(cfg) -> None:
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'cfg must be a HybridConfig, got {type(cfg).__name__}.')
    if cfg.memory_type not in _MEMORY_KINDS:
        raise ValueError(
            f"cfg.memory_type={cfg.memory_type!r} is not auditable; the exact "
            f"FLOPs reference model covers the {'/'.join(_MEMORY_KINDS)} "
            f"memory routes only.")


def ratio_layout(n_kda: int, n_csa: int, n_hca: int,
                 total_layers: int, memory_kind: str = 'kda') -> tuple[str, ...]:
    """Expand a KDA:CSA:HCA ratio into an explicit fixed-depth layout.

    The repeating unit is ``[kda] * n_kda + [csa] * n_csa + [hca] * n_hca``;
    the unit is repeated and truncated to ``total_layers`` (the same
    convention as ``HybridAttentionStack``'s ratio mode).  The counts are
    explicit here because the audit's plan must know every layer.
    """
    for name, value in (('n_kda', n_kda), ('n_csa', n_csa),
                        ('n_hca', n_hca)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'{name}={value!r} must be an int.')
        if value < 0:
            raise ValueError(f'{name}={value} must be >= 0.')
    _validate_positive_int('total_layers', total_layers)
    if memory_kind not in _MEMORY_KINDS:
        raise ValueError(
            f'memory_kind={memory_kind!r} must be one of {_MEMORY_KINDS}.')
    if n_kda + n_csa + n_hca == 0:
        raise ValueError('at least one of n_kda/n_csa/n_hca must be > 0.')
    unit = ([memory_kind] * n_kda + ['csa'] * n_csa + ['hca'] * n_hca)
    repeats = math.ceil(total_layers / len(unit))
    return tuple((unit * repeats)[:total_layers])


def layer_reference_flops(plan, cfg: HybridConfig, T: int,
                          batch: int = 1) -> list[dict]:
    """Exact per-layer GEMM FLOPs for a validated plan and prefill length.

    The FLOPs are ``2 * flops_audit.reference_gemm_macs`` (MACs counted as
    two FLOPs), i.e. they describe the work the reference operators actually
    dispatch at length ``T``.  Both the layernorm and the residual adds are
    outside the GEMM ledger, exactly as in ``flops_audit.audit_layer``.
    """
    _validate_cfg(cfg)
    _validate_positive_int('T', T)
    _validate_positive_int('batch', batch)
    plan = list(plan)
    if not plan:
        raise ValueError('plan must be non-empty.')
    rows = []
    for spec in plan:
        macs = int(fa.reference_gemm_macs(spec.kind, cfg, T, batch))
        rows.append({
            'layer_index': spec.layer_index,
            'kind': spec.kind,
            'block': spec.block,
            'window': spec.window,
            'topk': spec.topk,
            'gemm_macs': macs,
            'gemm_flops': int(2 * macs),
            'canonical_flops': float(fa.canonical_flops(
                spec.kind, cfg, T, batch)),
        })
    return rows


def _structural_far_fraction(
        structural: dict, far_min_distance: int) -> float | None:
    """Exact-path placement fraction beyond ``far_min_distance``.

    ``None`` when the sequence has no placement at or beyond the threshold
    (e.g. a window longer than the whole sequence). Reporting ``0.0`` there
    would falsely claim that far placements exist and are all unreachable;
    the compute-side aggregate is likewise ``None`` with no far rows.
    """
    total = hit = 0
    for distance, slot in structural['by_distance'].items():
        if int(distance) < far_min_distance:
            continue
        n = int(slot['n'])
        total += n
        hit += int(slot['any_exact'])
    return (hit / total) if total else None


def distance_compute_reach(plan, cfg: HybridConfig, seq_len: int, *,
                           batch: int = 1,
                           layer_flops: list[dict] | None = None) -> dict:
    """Join structural exact reachability with per-layer FLOPs.

    For each key placement (key at ``p``, value at ``p+1``, cue at
    ``seq_len-1``) the exact-path layers are those that satisfy the
    documented block/window rules.  The returned ``exact_reachable_flops`` is
    the sum of those layers' GEMM FLOPs; dividing by the stack total gives
    the fraction of prefill compute executed in layers that could carry the
    planted association exactly.

    ``state_capable_*`` counts recurrent memory layers separately.  They are
    *not* exact paths (same convention as ``recall_reach``), so they are
    included in the denominator but never in ``exact_reachable_flops``.
    """
    _validate_cfg(cfg)
    _validate_positive_int('seq_len', seq_len, minimum=3)
    _validate_positive_int('batch', batch)
    plan = list(plan)
    if not plan:
        raise ValueError('plan must be non-empty.')
    if layer_flops is None:
        layer_flops = layer_reference_flops(plan, cfg, seq_len, batch=batch)
    if len(layer_flops) != len(plan):
        raise ValueError(
            f'layer_flops has {len(layer_flops)} rows for a {len(plan)}-layer '
            f'plan.')
    flops_by_index = {int(row['layer_index']): float(row['gemm_flops'])
                      for row in layer_flops}
    if len(flops_by_index) != len(plan):
        raise ValueError('layer_flops must have one unique layer_index per '
                         'plan entry.')
    total_flops = float(sum(flops_by_index.values()))
    structural = structural_recall(plan, seq_len)
    max_window = max((spec.window for spec in plan
                      if spec.kind in BLOCK_KINDS), default=0)
    far_min_distance = max(2, int(max_window))
    rows = []
    for row in structural['rows']:
        exact_indices = [int(layer['layer_index'])
                         for layer in row['layers'] if layer['exact']]
        state_indices = [int(layer['layer_index'])
                         for layer in row['layers'] if layer['state']]
        exact_flops = float(sum(flops_by_index[i] for i in exact_indices))
        state_flops = float(sum(flops_by_index[i] for i in state_indices))
        rows.append({
            'key_pos': int(row['key_pos']),
            'distance': int(row['distance']),
            'any_exact': bool(row['any_exact']),
            'exact_layer_indices': exact_indices,
            'n_exact_layers': len(exact_indices),
            'exact_reachable_flops': exact_flops,
            'exact_reachable_fraction': (
                exact_flops / total_flops if total_flops else None),
            'state_capable_layer_indices': state_indices,
            'state_capable_flops': state_flops,
            'state_capable_fraction': (
                state_flops / total_flops if total_flops else None),
        })
    n = len(rows)
    far_rows = [r for r in rows if r['distance'] >= far_min_distance]
    exact_fractions = [r['exact_reachable_fraction'] for r in rows
                       if r['exact_reachable_fraction'] is not None]
    far_exact_fractions = [r['exact_reachable_fraction'] for r in far_rows
                           if r['exact_reachable_fraction'] is not None]
    state_fraction = (rows[0]['state_capable_fraction'] if rows else None)
    return {
        'seq_len': int(seq_len),
        'n_placements': n,
        'total_gemm_flops': total_flops,
        'max_window': int(max_window),
        'far_min_distance': int(far_min_distance),
        'mean_exact_reachable_fraction': (
            sum(exact_fractions) / n if exact_fractions else None),
        'far_mean_exact_reachable_fraction': (
            sum(far_exact_fractions) / len(far_exact_fractions)
            if far_exact_fractions else None),
        'min_exact_reachable_fraction': (
            min(exact_fractions) if exact_fractions else None),
        'max_exact_reachable_fraction': (
            max(exact_fractions) if exact_fractions else None),
        'mean_exact_layers': (
            sum(r['n_exact_layers'] for r in rows) / n if n else None),
        'far_mean_exact_layers': (
            sum(r['n_exact_layers'] for r in far_rows) / len(far_rows)
            if far_rows else None),
        'state_capable_fraction': state_fraction,
        'structural_any_exact_fraction': structural['any_exact_fraction'],
        'structural_far_any_exact_fraction': _structural_far_fraction(
            structural, far_min_distance),
        'rows': rows,
    }


def _hca_value_fold_bound(cfg: HybridConfig, layout, T: int,
                          batch: int) -> int:
    """Upper bound on the documented HCA single-block value-fold shortfall.

    With exactly one compressed block, PyTorch may fold the value contraction
    into a mul+sum, so the measured GEMM count can trail the reference model
    by at most ``Tp * c * nh`` MACs per HCA layer (see
    ``flops_audit.hca_reference_gemm_macs``).  With two or more blocks the
    fold does not occur and the bound is zero.
    """
    Tp = T + (-T) % cfg.hca_m2
    n_blocks = Tp // cfg.hca_m2
    if n_blocks != 1:
        return 0
    n_hca = sum(1 for kind in layout if kind == 'hca')
    return int(batch * n_hca * Tp * cfg.hca_c * cfg.hca_nh)


def _measure_stack(cfg: HybridConfig, layout, seq_len: int, *,
                   batch: int, device: str, seed: int) -> dict:
    torch.manual_seed(seed)
    model = HybridAttentionStack(cfg, total_layers=len(layout))
    model = model.to(device).eval()
    model.reset_state()
    x = torch.randn(batch, seq_len, cfg.d_model, device=device)
    _, ledger = fa.measure_forward(model, x)
    return {
        'gemm_macs': int(ledger.gemm_macs),
        'gemm_flops': int(ledger.gemm_flops),
        'matmul_macs': int(ledger.matmul_macs),
        'trsm_macs': int(ledger.trsm_macs),
        'trsm_calls': int(ledger.trsm_calls),
        'special_ops': {k: int(v) for k, v in ledger.special_ops.items()},
    }


def audit_layout(layout, cfg: HybridConfig | None = None, *,
                 seq_len: int = 128, batch: int = 1, device: str = 'cpu',
                 measure: bool = True, seed: int = 0) -> dict:
    """Audit one explicit hybrid layout at one prefill length.

    Returns the exact per-layer FLOPs, the structural summary, the
    distance-conditioned exact-reachable compute, and (when ``measure``)
    the whole-stack dispatch ledger cross-check.
    """
    layout = _validate_layout(layout)
    _validate_positive_int('seq_len', seq_len, minimum=3)
    _validate_positive_int('batch', batch)
    if cfg is None:
        cfg = fa.probe_config()
    _validate_cfg(cfg)
    memory_kinds = {kind for kind in layout if kind in _MEMORY_KINDS}
    memory_type = (next(iter(memory_kinds)) if memory_kinds
                   else cfg.memory_type)
    cfg = dataclasses.replace(
        cfg, layout=layout, memory_type=memory_type)
    plan = plan_from_config(cfg, layout)
    layer_flops = layer_reference_flops(plan, cfg, seq_len, batch=batch)
    reference_macs = int(sum(row['gemm_macs'] for row in layer_flops))
    reach = distance_compute_reach(
        plan, cfg, seq_len, batch=batch, layer_flops=layer_flops)
    record = {
        'seq_len': int(seq_len),
        'batch': int(batch),
        'layout': list(layout),
        'n_layers': len(layout),
        'n_kda': sum(1 for kind in layout if kind == 'kda'),
        'n_mamba2': sum(1 for kind in layout if kind == 'mamba2'),
        'n_csa': sum(1 for kind in layout if kind == 'csa'),
        'n_hca': sum(1 for kind in layout if kind == 'hca'),
        'reference_gemm_macs': reference_macs,
        'reference_gemm_flops': int(2 * reference_macs),
        'kda_flops_fraction': (
            sum(row['gemm_flops'] for row in layer_flops
                if row['kind'] == 'kda') / (2 * reference_macs)
            if reference_macs else None),
        'memory_flops_fraction': (
            sum(row['gemm_flops'] for row in layer_flops
                if row['kind'] in _MEMORY_KINDS) / (2 * reference_macs)
            if reference_macs else None),
        'layer_flops': layer_flops,
        'structural': structural_summary(plan, seq_len),
        'compute_reach': reach,
        'config': dataclasses.asdict(cfg),
        'measurement': None,
    }
    if measure:
        measurement = _measure_stack(
            cfg, layout, seq_len, batch=batch, device=device, seed=seed)
        delta = reference_macs - measurement['gemm_macs']
        bound = _hca_value_fold_bound(cfg, layout, seq_len, batch)
        within = bool(0 <= delta <= bound)
        measurement.update({
            'reference_minus_measured_macs': int(delta),
            'hca_value_fold_bound_macs': int(bound),
            'match': bool(delta == 0),
            'within_documented_hca_value_fold': within,
            'match_kind': (
                'exact' if delta == 0 else
                ('hca_value_fold' if within else 'mismatch')),
        })
        record['measurement'] = measurement
    return record


def _resolve_layouts(layouts, labels) -> list[tuple[str, tuple[str, ...]]]:
    if layouts is None:
        layouts = list(LAYOUT_PRESETS)
    elif isinstance(layouts, str):
        layouts = [layouts]
    elif (isinstance(layouts, tuple) and layouts
          and all(isinstance(kind, str) and kind in SUPPORTED_LAYER_KINDS
                  for kind in layouts)):
        # A bare explicit layout is naturally written as a tuple of kinds
        # (``('kda', 'csa', 'hca')``). A tuple of preset NAMES is a
        # sequence of layouts instead: the two vocabularies are disjoint,
        # so this does not make the API ambiguous.
        layouts = [layouts]
    else:
        layouts = list(layouts)
    if labels is not None:
        labels = list(labels)
        if len(labels) != len(layouts):
            raise ValueError(
                f'labels has {len(labels)} entries for {len(layouts)} '
                f'layouts.')
    resolved = []
    for i, item in enumerate(layouts):
        if isinstance(item, str):
            if item not in LAYOUT_PRESETS:
                raise ValueError(
                    f'unknown layout preset {item!r}; choose from '
                    f'{sorted(LAYOUT_PRESETS)}.')
            resolved.append((item, tuple(LAYOUT_PRESETS[item])))
        else:
            layout = _validate_layout(item)
            name = (labels[i] if labels is not None else '-'.join(layout))
            resolved.append((name, layout))
    return resolved


def _pareto_flags(rows: list[dict]) -> list[bool]:
    flags = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            left = row['compute_reach']
            right = other['compute_reach']
            # A missing far-placement metric means the axis was not
            # measurable for that layout (no placement beyond its largest
            # window), not that its reach is zero. Coercing None to 0.0
            # would fabricate a domination claim on undefined evidence.
            left_metrics = (
                left['far_mean_exact_reachable_fraction'],
                left['structural_far_any_exact_fraction'],
            )
            right_metrics = (
                right['far_mean_exact_reachable_fraction'],
                right['structural_far_any_exact_fraction'],
            )
            if any(v is None for v in left_metrics + right_metrics):
                continue
            no_worse = (
                other['reference_gemm_flops'] <= row['reference_gemm_flops']
                and right_metrics[0] >= left_metrics[0]
                and right_metrics[1] >= left_metrics[1]
            )
            strictly_better = (
                other['reference_gemm_flops'] < row['reference_gemm_flops']
                or right_metrics[0] > left_metrics[0]
                or right_metrics[1] > left_metrics[1]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return flags


def audit_layouts(layouts=None, cfg: HybridConfig | None = None, *,
                  labels=None, seq_len: int = 128, batch: int = 1,
                  device: str = 'cpu', measure: bool = True,
                  seed: int = 0) -> dict:
    """Audit a set of named layouts / ratios at a common depth and length."""
    resolved = _resolve_layouts(layouts, labels)
    rows = []
    for name, layout in resolved:
        row = audit_layout(
            layout, cfg=cfg, seq_len=seq_len, batch=batch, device=device,
            measure=measure, seed=seed)
        row['group'] = name
        rows.append(row)
    flags = _pareto_flags(rows)
    for row, flag in zip(rows, flags):
        row['pareto_optimal'] = bool(flag)
    return {
        'experiment': 'exp17_recall_flops_audit',
        'design': {
            'question': (
                'how much of the prefill compute sits in layers that can '
                'carry a planted long-range association exactly?'),
            'seq_len': int(seq_len),
            'batch': int(batch),
            'measured': bool(measure),
            'layouts': [name for name, _ in resolved],
            'metric_definitions': {
                'exact_reachable_flops_fraction': (
                    'sum of GEMM FLOPs in layers with a documented exact '
                    'block/window path at that placement, divided by the '
                    'stack total; recurrent state layers are excluded from '
                    'the numerator (no exact guarantee)'),
                'structural_far_any_exact_fraction': (
                    'fraction of placements beyond every sliding window that '
                    'have at least one exact path'),
                'pareto_optimal': (
                    'not dominated on (total FLOPs minimized, far-placement '
                    'exact-reachable compute maximized, far structural '
                    'coverage maximized)'),
            },
            'caveats': [
                'structural reachability is not end-to-end accuracy; the '
                'indexer can still miss a reachable block and the state can '
                'store associations without an exact guarantee',
                'GEMM FLOPs exclude layernorm/residual elementwise work and '
                'are not a latency prediction',
                'the exact reference model covers the KDA and Mamba-2 '
                'memory routes; RWKV-7 is rejected rather than approximated',
            ],
            'provenance': capture_provenance(),
        },
        'rows': rows,
    }


def _parse_ratio(text: str) -> tuple[int, int, int]:
    parts = text.strip().split(':')
    if len(parts) != 3:
        raise ValueError(
            f'ratio {text!r} must have the form KDA:CSA:HCA, e.g. 3:1:1.')
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f'ratio {text!r} contains a non-integer component.') from exc
    if any(v < 0 for v in values) or sum(values) == 0:
        raise ValueError(
            f'ratio {text!r} must have non-negative components and at least '
            f'one positive.')
    return values  # type: ignore[return-value]


def _format_table(rows: list[dict]) -> str:
    header = (f"{'group':>22} {'L':>3} {'GFLOPs':>10} {'mem%':>7} "
              f"{'far reach%':>11} {'struct far%':>11} {'measure':>12}")
    lines = [header, '-' * len(header)]
    for row in rows:
        reach = row['compute_reach']
        far_reach = reach['far_mean_exact_reachable_fraction']
        struct_far = reach['structural_far_any_exact_fraction']
        measured = row['measurement']
        measure_label = ('n/a' if measured is None
                         else measured['match_kind'])
        lines.append(
            f"{row['group']:>22} {row['n_layers']:>3} "
            f"{row['reference_gemm_flops'] / 1e9:>10.4f} "
            f"{100 * (row['memory_flops_fraction'] or 0.0):>6.1f}% "
            f"{100 * (far_reach or 0.0):>10.2f}% "
            f"{100 * (struct_far or 0.0):>10.2f}% "
            f"{measure_label:>12}")
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Recall-conditioned FLOPs audit for hybrid layouts.')
    parser.add_argument('--layouts', default=None,
                        help='comma-separated preset names (default: all '
                             'presets unless --ratios is given)')
    parser.add_argument('--ratios', default=None,
                        help='comma-separated KDA:CSA:HCA ratios, e.g. '
                             '3:1:1,1:2:2,1:0:0')
    parser.add_argument('--total-layers', type=int, default=10,
                        help='depth used to expand --ratios (default: 10)')
    parser.add_argument('--memory-route', default='kda',
                        choices=list(_MEMORY_KINDS),
                        help='memory operator for --ratios (default: kda)')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='prefill length / cue length (default: 128)')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-measure', action='store_true',
                        help='skip the whole-stack dispatch counter')
    parser.add_argument('--json', default=None,
                        help='optional output JSON path')
    args = parser.parse_args(argv)

    layouts: list = []
    labels: list[str] = []
    if args.layouts:
        layouts.extend([s.strip() for s in args.layouts.split(',')
                        if s.strip()])
        labels.extend(layouts)
    if args.ratios:
        for text in [s.strip() for s in args.ratios.split(',') if s.strip()]:
            values = _parse_ratio(text)
            layouts.append(ratio_layout(
                *values, total_layers=args.total_layers,
                memory_kind=args.memory_route))
            labels.append(
                f'ratio-{values[0]}:{values[1]}:{values[2]}'
                f'-{args.memory_route}')
    if not layouts:
        layouts = list(LAYOUT_PRESETS)
        labels = list(LAYOUT_PRESETS)

    report = audit_layouts(
        layouts, labels=labels, seq_len=args.seq_len, batch=args.batch,
        device=args.device, measure=not args.no_measure, seed=args.seed)
    print(_format_table(report['rows']))
    if args.json:
        write_results_json(report, args.json)
        print(f'\n[recall-flops-audit] wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
