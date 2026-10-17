#!/usr/bin/env python3
"""Experiment 13 — batch-aware decode memory-traffic ledger.

Runs :mod:`decode_bandwidth` on a ratio-mode route-comparison config at the
repository's matrix geometry (mirroring ``run_ablation._matrix_*_dims`` /
``run_quality.SMALL_MODEL_SPEC``) and reports, per memory route
(KDA / Mamba-2 / RWKV-7, all else held fixed):

* the tower weight bytes streamed per step (batch-independent),
* the per-sequence state/cache bytes per decoded token, split by operator
  (memory route vs CSA vs HCA — the aggregation price),
* the batch-scaling ledger ``weights + B * state`` and the state share,
* the weights<->state dominance SWITCH BATCH, i.e. where the step stops
  being weight-bandwidth-bound and becomes state-bandwidth-bound,
* the memory-side bandwidth-bound step time against the host's measured
  copy bandwidth (purely a lower bound; see the module's honesty notes).

Usage::

    python3 run_decode_bandwidth.py
    python3 run_decode_bandwidth.py --T 8192 --batches 1,8,64,256 \
        --json results/exp13_decode_bandwidth.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roofline  # noqa: E402

import decode_bandwidth as db  # noqa: E402
from kaggle_setup import (capture_provenance, configure_torch_for_device,  # noqa: E402
                          sanitize_for_json, write_results_json)
from ops_fused import HybridConfig  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp13_decode_bandwidth.json')
KIND = 'exp13_decode_bandwidth'


def build_base_cfg(d_model: int = 32) -> HybridConfig:
    """Ratio-mode route-comparison config at the matrix geometry.

    CSA/HCA follow ``run_quality.SMALL_MODEL_SPEC`` and the memory-route
    geometry follows ``run_ablation._matrix_mamba_dims`` /
    ``_matrix_rwkv_dims`` (the repo's auto-clamped small-scale policy), so
    the ONLY field that differs across the compared rows is
    ``memory_type`` — exactly the one-knob memory-route switch.
    """
    from run_ablation import _matrix_mamba_dims, _matrix_rwkv_dims
    from run_quality import SMALL_MODEL_SPEC as spec
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
        **_matrix_mamba_dims(d_model),
        **_matrix_rwkv_dims(d_model),
    )


def parse_batches(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            b = int(part)
        except ValueError:
            raise SystemExit(f'--batches: {part!r} is not an int')
        if b < 1:
            raise SystemExit(f'--batches: {b} must be >= 1')
        values.append(b)
    if not values:
        raise SystemExit('--batches: at least one batch size required')
    return tuple(dict.fromkeys(values))


def parse_routes(raw: str) -> tuple[str, ...]:
    values = [p.strip() for p in raw.split(',') if p.strip()]
    for r in values:
        if r not in db.MEMORY_ROUTES:
            raise SystemExit(
                f'--routes: {r!r} must be one of {list(db.MEMORY_ROUTES)}')
    return tuple(dict.fromkeys(values)) or db.MEMORY_ROUTES


def measure_host_bandwidth(device) -> tuple[float | None, str]:
    """Measured streaming copy bandwidth (GB/s) + its provenance label.

    Priority: on-host measurement -> spec-sheet preset for a named GPU ->
    ``(None, 'unavailable')`` (the bandwidth-bound columns are then
    omitted rather than guessed).
    """
    bw = roofline.measure_copy_bandwidth_gbps(device)
    if bw is not None:
        return bw, 'measured_on_host'
    name = ''
    try:
        import torch
        if str(device).startswith('cuda'):
            name = torch.cuda.get_device_name(device)
    except Exception:
        name = ''
    spec = roofline.spec_for_device(name)
    if spec is not None:
        return spec['peak_bw_gbps'], f"preset:{spec['preset_key']}"
    return None, 'unavailable'


def _print_report(report: dict) -> None:
    cfg = report['config']
    print(f"[exp13] T={cfg['T']}  batches={cfg['batches']}  "
          f"itemsize={cfg['itemsize']}B  d_model={cfg['d_model']}  "
          f"layout={'-'.join(cfg['layout_unit'])}")
    if cfg['bw_gbps'] is None:
        print('[exp13] host bandwidth unavailable; bandwidth-bound '
              'columns omitted')
    else:
        print(f"[exp13] host copy bandwidth: {cfg['bw_gbps']:.1f} GB/s "
              f"({cfg['bw_source']})")
    for row in report['routes']:
        if row['status'] != 'ok':
            print(f"  {row['route']:<7} unavailable: {row['reason'][:90]}")
            continue
        reads = row['read_bytes_by_op_per_token']
        print(f"  {row['route']:<7} weights={row['weight_bytes_per_step']:>9,}B  "
              f"state={row['state_bytes_per_token']:>10,.1f}B/token  "
              f"switch@B={row['weights_state_switch_batch']:.2f}")
        print(f"          read split: memory={reads['memory']:>9,.1f}  "
              f"csa={reads['csa']:>8,.1f}  hca={reads['hca']:>8,.1f} B/token")
        for b, cell in row['by_batch'].items():
            line = (f"          B={int(b):<4} total/step="
                    f"{cell['total_bytes_per_step']:>12,}B  "
                    f"state-share={cell['state_share']:.3f}")
            if cell.get('bandwidth_bound_ms') is not None:
                line += (f"  bw-bound={cell['bandwidth_bound_ms']:.4f}ms")
            print(line)


def build_report(*, T: int, batches: tuple[int, ...],
                 routes: tuple[str, ...], d_model: int = 32,
                 itemsize: int = 4, bw_gbps: float | None = None,
                 bw_source: str = 'unavailable') -> dict:
    """Assemble the JSON-safe Exp 13 payload (pure; no I/O)."""
    base_cfg = build_base_cfg(d_model)
    table = db.route_switch_table(base_cfg, T, batches=batches,
                                  routes=routes, itemsize=itemsize)
    if bw_gbps is not None:
        for row in table:
            if row['status'] != 'ok':
                continue
            for cell in row['by_batch'].values():
                cell['bandwidth_bound_ms'] = db.bandwidth_bound_ms(
                    cell['total_bytes_per_step'], bw_gbps)
    weights = db.attention_tower_weight_bytes(base_cfg, itemsize=itemsize)
    return {
        'kind': KIND,
        'method': (
            'batch-aware decode-step traffic ledger: weight bytes streamed '
            'once per step (batch-independent) + B x per-sequence '
            'state/cache bytes; per-op split via lm_consistency/kv_budget; '
            'switch batch = weight elements / state elements per token'),
        'config': {
            'T': int(T),
            'batches': [int(b) for b in batches],
            'itemsize': int(itemsize),
            'd_model': int(d_model),
            'routes': list(routes),
            'layout_unit': list(db.unit_layout(base_cfg)),
            'layout_counts': [base_cfg.n_kda, base_cfg.n_csa, base_cfg.n_hca],
            'bw_gbps': bw_gbps,
            'bw_source': bw_source,
        },
        'base_weights': {
            'total_elements': weights['total'],
            'total_bytes': weights['bytes'],
            'bytes_by_kind': weights['bytes_by_kind'],
        },
        'routes': table,
        'notes': [
            'attention-tower scope: embeddings / lm_head / final norm are '
            'route-independent and excluded',
            'compulsory-traffic lower bound: one weight stream per step; a '
            'kernel may keep weights resident across a large batch',
            'state/cache bytes are the canonical lm_consistency decode '
            'model (CSA/HCA reads averaged over the T prefill queries + '
            'recurrent-state writes); see its docstring for exclusions',
            'bandwidth_bound_ms uses the host copy bandwidth and is a '
            'memory-side lower bound, not a latency prediction',
        ],
        'provenance': capture_provenance(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Exp 13 — batch-aware decode memory-traffic ledger.')
    parser.add_argument('--T', type=int, default=4096,
                        help='decode context length (default 4096)')
    parser.add_argument('--batches', default='1,2,4,8,16,32,64,128',
                        help='comma-separated decode batch sizes')
    parser.add_argument('--routes', default=','.join(db.MEMORY_ROUTES),
                        help='comma-separated memory routes')
    parser.add_argument('--d-model', type=int, default=32,
                        help='model width for the matrix geometry')
    parser.add_argument('--itemsize', type=int, default=4,
                        help='bytes per element (4 = fp32)')
    parser.add_argument('--bw-gbps', type=float, default=None,
                        help='override the measured host bandwidth')
    parser.add_argument('--json', default=RESULTS_PATH,
                        help=f'output path (default {RESULTS_PATH})')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    info = configure_torch_for_device()
    if args.bw_gbps is not None:
        bw, source = args.bw_gbps, 'explicit'
    else:
        bw, source = measure_host_bandwidth(info.device)
    report = build_report(
        T=args.T, batches=parse_batches(args.batches),
        routes=parse_routes(args.routes), d_model=args.d_model,
        itemsize=args.itemsize, bw_gbps=bw, bw_source=source)
    payload = sanitize_for_json(report)
    json.dumps(payload, allow_nan=False)
    _print_report(payload)
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    write_results_json(payload, args.json, logger=logger)
    print(f'[exp13] wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
