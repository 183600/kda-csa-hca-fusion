#!/usr/bin/env python3
"""Experiment 17 — time-to-first-token (TTFT) storage budget.

Sweeps prompt length, batch and memory route through
:func:`ttft_budget.ttft_ledger` at the matrix geometry used by Exp 13, and
compares four storage scenarios:

* ``fp16`` — fp16 weights, activations and state (baseline);
* ``fp8-weights`` — MXFP8 tower weights, fp16 activations/state;
* ``fp8-weights-state`` — MXFP8 weights + recurrent state + CSA/HCA caches;
* ``fp8-weights-state-fp8-compute`` — the same storage plus an explicit 2x
  FP8 arithmetic-rate assumption (a hardware assumption, not a storage
  claim).

The report answers three questions the decode ledger cannot:

1. Is the prefill (TTFT-dominant) phase compute- or memory-bound at this
   ``(T, batch)``? Weight-only FP8 can move the first phase at most by the
   weight share of its bytes.
2. How much of TTFT is already the first decode step, priced with the
   *marginal* cache budget at position ``T`` (not the prefill average)?
3. Which machine balance point makes storage-only FP8 irrelevant? The
   ledger reports arithmetic intensity next to the ridge point, so the
   answer is visible instead of hidden in a wall-clock number.

Usage::

    python3 run_ttft_budget.py
    python3 run_ttft_budget.py --lengths 1,8,64,512,4096 --batches 1,16 \
        --peak-tflops 67 --bw-gbps 3350 \
        --json results/exp17_ttft_budget.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode_bandwidth as db  # noqa: E402
from kaggle_setup import (append_run_manifest, capture_provenance,  # noqa: E402
                          configure_torch_for_device,
                          make_run_manifest_entry, sanitize_for_json,
                          write_results_json)
from lm_consistency import host_peaks  # noqa: E402
from run_decode_bandwidth import (build_base_cfg, parse_batches,  # noqa: E402
                                  parse_routes)
from ttft_budget import (SCENARIOS, resolve_scenario,  # noqa: E402
                         ttft_ledger)

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp17_ttft_budget.json')
KIND = 'exp17_ttft_budget'
DEFAULT_LENGTHS = (1, 16, 128, 1024, 8192)


def parse_lengths(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            T = int(part)
        except ValueError:
            raise SystemExit(f'--lengths: {part!r} is not an int')
        if T < 1:
            raise SystemExit(f'--lengths: {T} must be >= 1')
        values.append(T)
    if not values:
        raise SystemExit('--lengths: at least one prompt length required')
    return tuple(dict.fromkeys(values))


def build_report(*, lengths: tuple[int, ...], batches: tuple[int, ...],
                 routes: tuple[str, ...], d_model: int = 32,
                 itemsize: int = 2, peak_tflops: float,
                 bw_gbps: float, peak_sources: dict,
                 scenarios: tuple[str, ...] = SCENARIOS,
                 state_mode: str = 'compressed_kv_only') -> dict:
    """Assemble the JSON-safe TTFT sweep (pure; no I/O)."""
    if not lengths:
        raise ValueError('build_report: lengths must be non-empty.')
    for name in scenarios:
        if name not in SCENARIOS:
            raise ValueError(
                f'build_report: scenario={name!r} must be one of '
                f'{list(SCENARIOS)}.')
    base_cfg = build_base_cfg(d_model)
    rows: list[dict] = []
    for route in routes:
        cfg = dataclasses.replace(base_cfg, memory_type=route)
        weight_elements = db.attention_tower_parameter_elements(cfg)['total']
        for T in lengths:
            for batch in batches:
                base = ttft_ledger(
                    cfg, T, batch=batch, itemsize=itemsize,
                    peak_tflops=peak_tflops, bw_gbps=bw_gbps,
                    state_mode=state_mode,
                    weight_elements=weight_elements)
                baseline_ms = base['ttft_lower_bound_ms']
                cells = {}
                for name in scenarios:
                    spec = resolve_scenario(name, itemsize=itemsize)
                    led = ttft_ledger(
                        cfg, T, batch=batch, itemsize=itemsize,
                        peak_tflops=peak_tflops, bw_gbps=bw_gbps,
                        state_mode=state_mode,
                        weight_elements=weight_elements, **spec)
                    cells[name] = {
                        'ttft_lower_bound_ms': led['ttft_lower_bound_ms'],
                        'prefill_bound': led['prefill']['bound'],
                        'prefill_lower_bound_ms':
                            led['prefill']['lower_bound_ms'],
                        'first_decode_ms':
                            led['first_decode']['memory_bound_ms'],
                        'weight_bytes_per_element':
                            led['storage']['weight_bytes_per_element'],
                        'state_bytes_per_element':
                            led['storage']['state']['bytes_per_element_by_op'],
                        'compute_speedup': led['machine']['compute_speedup'],
                        'ttft_speedup_vs_fp16': (
                            baseline_ms / led['ttft_lower_bound_ms']
                            if led['ttft_lower_bound_ms'] > 0 else None),
                    }
                rows.append({
                    'T': int(T),
                    'batch': int(batch),
                    'route': route,
                    'layout_unit': base['layout_unit'],
                    'prefill': {
                        'flops': base['prefill']['flops'],
                        'bound': base['prefill']['bound'],
                        'arithmetic_intensity_flops_per_byte':
                            base['prefill']['arithmetic_intensity_flops_per_byte'],
                        'lower_bound_ms': base['prefill']['lower_bound_ms'],
                        'compute_ms': base['prefill']['compute_ms'],
                        'memory_ms': base['prefill']['memory_ms'],
                        'bytes': base['prefill']['bytes'],
                    },
                    'first_decode': {
                        'memory_bound_ms':
                            base['first_decode']['memory_bound_ms'],
                        'instant_vs_average_reads':
                            base['first_decode']['instant_vs_average_reads'],
                        'bytes': base['first_decode']['bytes'],
                    },
                    'weights': base['weights'],
                    'ttft_lower_bound_ms': baseline_ms,
                    'prefill_share': base['prefill_share'],
                    'scenarios': cells,
                })
    return {
        'kind': KIND,
        'method': (
            'TTFT >= max(prefill compute, prefill memory) + first-decode '
            'memory; prefill FLOPs from run_kv_cache.prefill_flops at the '
            'lm_consistency config geometry; memory = one weight stream + '
            'activations + retained state writes; first-decode reads use '
            'the marginal kv_budget budget at t=T'),
        'config': {
            'd_model': int(d_model),
            'lengths': [int(T) for T in lengths],
            'batches': [int(b) for b in batches],
            'routes': list(routes),
            'itemsize': int(itemsize),
            'state_mode': state_mode,
            'layout_unit': list(db.unit_layout(base_cfg)),
            'layout_counts': [base_cfg.n_kda, base_cfg.n_csa,
                              base_cfg.n_hca],
            'scenarios': list(scenarios),
            'peak_tflops': peak_tflops,
            'bw_gbps': bw_gbps,
            'peak_sources': peak_sources,
            'ridge_flops_per_byte': (peak_tflops * 1e12
                                     / (bw_gbps * 1e9)),
        },
        'rows': rows,
        'notes': [
            'attention-tower scope; embeddings / lm_head / sampling / '
            'queueing excluded',
            'storage-only FP8 changes bytes, not FLOPs; the '
            'fp8-*-fp8-compute row additionally assumes 2x arithmetic '
            'throughput (explicit, not measured)',
            'peak_tflops is the anchor for the compute dtype; a measured '
            'fp32 GEMM is conservative for a tensor-core FP8 run unless '
            '--peak-tflops is overridden',
            'TTFT is a lower bound, not a wall-clock prediction; measured '
            'latency is the prefill_profile / decode benchmark path',
        ],
        'provenance': capture_provenance(),
    }


def _print_report(report: dict) -> None:
    cfg = report['config']
    print(f"[exp17] d_model={cfg['d_model']} itemsize={cfg['itemsize']}B "
          f"peak={cfg['peak_tflops']:.2f} TFLOP/s "
          f"bw={cfg['bw_gbps']:.1f} GB/s "
          f"ridge={cfg['ridge_flops_per_byte']:.1f} FLOP/B")
    for row in report['rows']:
        cells = row['scenarios']
        speedups = '  '.join(
            f"{name}={cells[name]['ttft_speedup_vs_fp16']:.3f}x"
            for name in cfg['scenarios'])
        print(f"  {row['route']:<7} T={row['T']:<5} B={row['batch']:<3} "
              f"prefill={row['prefill']['lower_bound_ms']:>9.4f}ms "
              f"({row['prefill']['bound']:<7}) "
              f"AI={row['prefill']['arithmetic_intensity_flops_per_byte']:>8.1f} "
              f"TTFT={row['ttft_lower_bound_ms']:>9.4f}ms  {speedups}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Exp 17 — TTFT storage budget with FP8 weights/state.')
    parser.add_argument('--lengths', default=','.join(
        str(T) for T in DEFAULT_LENGTHS),
        help=f"comma-separated prompt lengths "
             f"(default {','.join(str(T) for T in DEFAULT_LENGTHS)})")
    parser.add_argument('--batches', default='1,8',
                        help='comma-separated prefill batch sizes')
    parser.add_argument('--routes', default=','.join(db.MEMORY_ROUTES),
                        help='comma-separated memory routes')
    parser.add_argument('--d-model', type=int, default=32,
                        help='model width for the matrix geometry')
    parser.add_argument('--itemsize', type=int, default=2,
                        help='bytes per element for activations/state '
                             '(2 = fp16, 4 = fp32)')
    parser.add_argument('--peak-tflops', type=float, default=None,
                        help='override the measured fp32 compute anchor')
    parser.add_argument('--bw-gbps', type=float, default=None,
                        help='override the measured streaming bandwidth')
    parser.add_argument('--scenarios', default=','.join(SCENARIOS),
                        help=f"comma-separated scenario names "
                             f"(default {','.join(SCENARIOS)})")
    parser.add_argument('--state-mode',
                        choices=('compressed_kv_only', 'full_accounting'),
                        default='compressed_kv_only',
                        help='retained-state accounting mode')
    parser.add_argument('--json', default=RESULTS_PATH,
                        help=f'output path (default {RESULTS_PATH})')
    args = parser.parse_args(argv)

    if args.d_model < 1:
        parser.error('--d-model must be >= 1')
    if args.itemsize < 1:
        parser.error('--itemsize must be >= 1')
    scenarios = tuple(s.strip() for s in args.scenarios.split(',')
                      if s.strip())
    for name in scenarios:
        if name not in SCENARIOS:
            parser.error(f'--scenarios: {name!r} must be one of '
                         f'{list(SCENARIOS)}')

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')
    info = configure_torch_for_device()
    peaks = host_peaks(info.device, peak_tflops=args.peak_tflops,
                       peak_bw_gbps=args.bw_gbps)
    if peaks['peak_tflops'] is None or peaks['peak_bw_gbps'] is None:
        parser.error('host peaks unavailable; pass --peak-tflops and '
                     '--bw-gbps')
    report = build_report(
        lengths=parse_lengths(args.lengths),
        batches=parse_batches(args.batches),
        routes=parse_routes(args.routes), d_model=args.d_model,
        itemsize=args.itemsize, peak_tflops=peaks['peak_tflops'],
        bw_gbps=peaks['peak_bw_gbps'],
        peak_sources=peaks['peak_sources'], scenarios=scenarios,
        state_mode=args.state_mode)
    payload = sanitize_for_json(report)
    json.dumps(payload, allow_nan=False)
    _print_report(payload)
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    write_results_json(payload, args.json, logger=logger)
    print(f'[exp17] wrote {args.json}')
    append_run_manifest(make_run_manifest_entry(
        experiment='exp17_ttft_budget',
        seeds=[],
        provenance=payload['provenance'],
        hyperparameters={
            'config': payload['config'],
            'method': payload['method'],
            'randomness': 'none (analytic TTFT budget; host peaks measured, '
                          'not sampled)',
        },
        data={
            'kind': 'analytic_ttft_budget',
            'lengths': payload['config']['lengths'],
            'batches': payload['config']['batches'],
            'routes': payload['config']['routes'],
            'scenarios': payload['config']['scenarios'],
        },
        artifacts={'results': args.json},
        results={'n_rows': len(payload['rows'])},
    ), logger=logger)
    return 0


if __name__ == '__main__':
    sys.exit(main())
