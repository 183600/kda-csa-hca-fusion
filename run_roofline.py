#!/usr/bin/env python3
"""Experiment 7 — roofline / memory-bandwidth analysis of the Exp 2 operators.

This runner joins two existing artifacts:

* ``results/exp2_benchmark.json`` — measured latency per (T, op) row, with the
  per-op ``workload`` geometry and (since this round) ``param_elements``;
* ``run_kv_cache.prefill_flops`` / ``kv_cache_elements`` — closed-form FLOPs
  and state-size formulas, evaluated at the EXACT Exp 2 geometry read from
  each row's ``workload`` metadata.

For every (T, op) cell it computes arithmetic intensity (FLOPs/byte), the
achieved throughput, the machine's roofline ceiling, and whether the cell is
memory- or compute-bound at the host's ridge point. Machine peaks come from
on-host microbenchmarks (:mod:`roofline`) with an optional spec-sheet preset
fallback, so the analysis works on CPU CI, Kaggle T4, and unknown GPUs alike.

Output: ``results/exp7_roofline.json`` written as a versioned envelope via
:mod:`results_schema` (``kind="exp7_roofline"``).

Environment knobs
-----------------
================================  =======  ============================
Variable                          Default  Effect
================================  =======  ============================
``ROOFLINE_SIZE_MB``              64       streaming-copy buffer size (MiB)
``ROOFLINE_REPEATS``              3        timed repeats per microbenchmark
``ROOFLINE_GEMM_N``               1024     square GEMM size for the compute peak
``ROOFLINE_PEAK_TFLOPS``          —        override measured/preset fp32 peak
``ROOFLINE_PEAK_BW_GBPS``         —        override measured/preset bandwidth
================================  =======  ============================

The interpretation caveat matters and is recorded in the output metadata:
the byte model is a **compulsory-traffic lower bound** (activation I/O +
weights + compressed/recurrent state). It excludes library-internal traffic
(unfold intermediates), so the naive reference paths legitimately land far
below the roofline; the model measures compulsory traffic, not the
reference implementation's actual traffic.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (configure_torch_for_device, parse_int_env,
                          set_global_seed, capture_provenance)
from results_schema import SchemaError, parse, write as write_results
from run_kv_cache import kv_cache_elements, prefill_flops
import roofline

logger = logging.getLogger(__name__)

# Seed for the on-host peak microbenchmarks (torch.randn inputs in
# roofline.measure_gemm_tflops). Recorded in the result metadata so the
# measured peaks are reproducible on the same machine.
ROOFLINE_SEED = 0

_BYTE_MODEL_NOTE = (
    'compulsory-traffic lower bound: activation I/O (input read + output '
    'write) + weights (param_elements, when recorded by the benchmark) + '
    'compressed/recurrent state (compressed_kv_only accounting). Library- '
    'internal traffic (unfold/softmax intermediates, allocator churn, '
    'Python interpreter overhead) is excluded, so achieved bandwidth vs. '
    'this model can exceed the measured copy bandwidth when the reference '
    'implementation moves more bytes than compulsory.'
)


def _csa_geometry(w: dict) -> dict:
    """Map an Exp 2 CSA ``workload`` record onto run_kv_cache kwargs."""
    return dict(
        d=w['d'],
        csa_m=w['m'], csa_topk=w['topk'], csa_nh=w['heads'],
        csa_c=w['c'], csa_dc=w['dc'], csa_nIh=w['nIh'], csa_cI=w['cI'],
        csa_sliding_window=w['sliding_window'],
    )


def _hca_geometry(w: dict) -> dict:
    """Map an Exp 2 HCA ``workload`` record onto run_kv_cache kwargs."""
    return dict(
        d=w['d'],
        hca_m2=w['m2'], hca_nh=w['heads'],
        hca_c=w['c'], hca_dc=w['dc'],
        hca_sliding_window=w['sliding_window'],
    )


def _hybrid_geometry(w: dict) -> dict:
    """Map an Exp 2 hybrid ``workload`` record onto run_kv_cache kwargs.

    Requires the extended geometry fields that ``run_benchmark.py`` records
    for the hybrid row (csa_c/dc/nIh/cI/nh, hca_c/dc/nh, kda chunk/conv
    sizes). Rows written by older benchmark revisions lack them and are
    reported as skipped rather than accounted with paper-scale defaults.
    """
    return dict(
        d=w['d'], H=w['heads_qk'], K=w['head_dim_k'], V=w['head_dim_v'],
        kda_hv=w['heads_v'], kda_k=w['head_dim_k'], kda_v=w['head_dim_v'],
        kda_chunk_size=w['kda_chunk_size'], kda_conv_ksize=w['kda_conv_ksize'],
        csa_m=w['csa_m'], csa_topk=w['csa_topk'], csa_nh=w['csa_nh'],
        csa_c=w['csa_c'], csa_dc=w['csa_dc'], csa_nIh=w['csa_nIh'],
        csa_cI=w['csa_cI'], csa_sliding_window=w['csa_sliding_window'],
        hca_m2=w['hca_m2'], hca_nh=w['hca_nh'], hca_c=w['hca_c'],
        hca_dc=w['hca_dc'], hca_sliding_window=w['hca_sliding_window'],
        hybrid_n_kda=w['n_kda'], hybrid_n_csa=w['n_csa'],
        hybrid_n_hca=w['n_hca'],
    )


def _core_accounting(op: str, T: int, w: dict) -> tuple[float, float]:
    """(flops, bytes) for the core-boundary ops (softmax / KDA variants)."""
    heads, kd = w['heads'], w['head_dim']
    vd = w.get('head_dim_v', kd)
    if op == 'softmax':
        flops = roofline.softmax_core_flops(T, heads, kd, vd)
        nbytes = roofline.core_bytes('softmax', T, heads=heads, head_dim=kd,
                                     value_dim=vd)
    else:
        chunked = op in ('kda_chunk', 'kda_chunk_hybrid')
        cs = w.get('chunk_size', 64)
        flops = roofline.kda_core_flops(T, heads, kd, vd, chunked=chunked,
                                        chunk_size=cs)
        nbytes = roofline.core_bytes('kda', T, heads=heads, head_dim=kd,
                                     value_dim=vd, chunked=chunked,
                                     chunk_size=cs)
    return flops, nbytes


def _e2e_accounting(op: str, T: int, w: dict,
                    param_elements: int | None) -> tuple[float, float, bool]:
    """(flops, bytes, weights_included) for end-to-end CSA/HCA/hybrid rows.

    FLOPs come from ``run_kv_cache.prefill_flops`` evaluated at the Exp 2
    geometry. ``run_kv_cache``'s accounting for these ops includes input
    projections + core + grouped output projection, matching Exp 2's
    ``end_to_end_single_layer`` / ``end_to_end_multi_layer`` boundary.
    """
    if op == 'csa':
        geo, kv_op = _csa_geometry(w), 'csa'
    elif op == 'hca':
        geo, kv_op = _hca_geometry(w), 'hca'
    else:
        geo, kv_op = _hybrid_geometry(w), 'hybrid_kch'
    flops = float(prefill_flops(kv_op, T, **geo))
    io_elements = 2 * T * w['d']
    state_elements = kv_cache_elements(kv_op, T, mode='compressed_kv_only',
                                       **geo)
    weights_included = param_elements is not None
    total_elements = io_elements + state_elements + (param_elements or 0)
    return flops, roofline.elements_to_bytes(total_elements), weights_included


def _account_row(row: dict) -> dict:
    """Build the analytic (flops, bytes) record for one Exp 2 benchmark row."""
    op, T = row['op'], row['T']
    w = row.get('workload') or {}
    if op in ('softmax', 'kda_rec', 'kda_chunk', 'kda_rec_hybrid',
              'kda_chunk_hybrid'):
        flops, nbytes = _core_accounting(op, T, w)
        return {'flops': flops, 'bytes': nbytes, 'weights_included': False,
                'analytic_boundary': 'core'}
    flops, nbytes, weights_included = _e2e_accounting(
        op, T, w, row.get('param_elements'))
    return {'flops': flops, 'bytes': nbytes,
            'weights_included': weights_included,
            'analytic_boundary': 'end_to_end'}


def _resolve_peaks(device: str, *, size_mb: int, repeats: int,
                   gemm_n: int) -> dict:
    """Measure or look up machine peaks; env overrides win over everything."""
    measured_bw = roofline.measure_copy_bandwidth_gbps(
        device, size_mb=size_mb, repeats=repeats)
    measured_gemm = roofline.measure_gemm_tflops(
        device, n=gemm_n, repeats=repeats)
    if str(device).startswith('cuda'):
        try:
            import torch
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = str(device)
    else:
        device_name = platform.processor() or 'cpu'
    preset = roofline.spec_for_device(device_name) or {}
    env_tflops = os.environ.get('ROOFLINE_PEAK_TFLOPS')
    env_bw = os.environ.get('ROOFLINE_PEAK_BW_GBPS')

    def _pick(env_val, measured, preset_val):
        if env_val:
            try:
                return float(env_val), 'env_override'
            except ValueError:
                logger.warning(
                    'ignoring non-numeric peak override value %r', env_val)
        if measured:
            return float(measured), 'measured_on_host'
        if preset_val:
            return float(preset_val), f"preset:{preset.get('preset_key')}"
        return None, 'unavailable'

    peak_tflops, tflops_src = _pick(env_tflops, measured_gemm,
                                    preset.get('peak_tflops'))
    peak_bw, bw_src = _pick(env_bw, measured_bw, preset.get('peak_bw_gbps'))
    return {
        'device_name': device_name,
        'peak_tflops': peak_tflops,
        'peak_tflops_source': tflops_src,
        'peak_bw_gbps': peak_bw,
        'peak_bw_source': bw_src,
        'measured_gemm_tflops': measured_gemm,
        'measured_copy_bw_gbps': measured_bw,
        'preset': preset or None,
    }


def main() -> int:
    configure_torch_for_device()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger.info('=' * 70)
    logger.info('Experiment 7: Roofline / memory-bandwidth analysis')
    logger.info('=' * 70)

    src = 'results/exp2_benchmark.json'
    try:
        with open(src, encoding='utf-8') as f:
            raw = json.load(f)
        _src_meta, rows, src_version = parse(raw)
    except FileNotFoundError:
        logger.error(f'{src} not found — run `python run_benchmark.py` first.')
        return 1
    except (json.JSONDecodeError, SchemaError) as e:
        logger.error(f'{src} cannot be read: {e}')
        return 1
    logger.info(f'Loaded {len(rows)} rows from {src} (schema v{src_version})')

    size_mb = parse_int_env('ROOFLINE_SIZE_MB', 64, min_value=1, logger=logger)
    repeats = parse_int_env('ROOFLINE_REPEATS', 3, min_value=1, logger=logger)
    gemm_n = parse_int_env('ROOFLINE_GEMM_N', 1024, min_value=64,
                           logger=logger)
    device = next((r.get('device') for r in rows if r.get('time_ms')),
                  'cpu')
    # Seed the host-peak microbenchmark inputs so the recorded
    # "measured_copy_bw_gbps" / "measured_gemm_tflops" metadata is
    # reproducible run to run.
    seed_info = set_global_seed(ROOFLINE_SEED)
    peaks = _resolve_peaks(device, size_mb=size_mb, repeats=repeats,
                           gemm_n=gemm_n)
    logger.info(
        f"Host: {peaks['device_name']} | fp32 GEMM peak "
        f"{peaks['peak_tflops']} TFLOP/s ({peaks['peak_tflops_source']}) | "
        f"copy BW {peaks['peak_bw_gbps']} GB/s ({peaks['peak_bw_source']})")
    peak_flops = (peaks['peak_tflops'] or 0) * 1e12 or None
    peak_bw = peaks['peak_bw_gbps'] or None

    points: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        if row.get('error') or row.get('time_ms') in (None, 0):
            skipped.append({'op': row.get('op'), 'T': row.get('T'),
                            'reason': row.get('error') or 'no timing'})
            continue
        try:
            acct = _account_row(row)
        except (KeyError, ValueError, TypeError) as e:
            skipped.append({'op': row.get('op'), 'T': row.get('T'),
                            'reason': f'analytic accounting failed: {e}'})
            continue
        backend = row.get('kda_backend')
        boundary_matched = backend in (None, 'reference')
        point = {
            'op': row['op'],
            'T': row['T'],
            'device': row.get('device'),
            'compute_boundary': row.get('compute_boundary'),
            'time_ms': row['time_ms'],
            'param_elements': row.get('param_elements'),
            'weights_bytes_included': acct['weights_included'],
            'analytic_boundary': acct['analytic_boundary'],
            'analytic_boundary_matched': boundary_matched,
        }
        if backend not in (None, 'reference'):
            point['kda_backend'] = backend
        point.update(roofline.roofline_point(
            flops=acct['flops'], nbytes=acct['bytes'],
            time_ms=row['time_ms'], peak_flops_per_s=peak_flops,
            peak_bw_gbps=peak_bw))
        points.append(point)

    if not points:
        logger.error('No roofline points could be computed; '
                     f'{len(skipped)} row(s) skipped.')
        return 1

    points.sort(key=lambda p: (p['T'], p['op']))
    width = max(len(p['op']) for p in points)
    logger.info('')
    logger.info(f"{'T':>6}  {'op':<{width}}  {'ms':>9}  {'TFLOP/s':>9}  "
                f"{'GB/s':>8}  {'AI':>8}  {'bound':>8}  {'%roof':>7}  label")
    for p in points:
        logger.info(
            f"{p['T']:>6}  {p['op']:<{width}}  {p['time_ms']:>9.4f}  "
            f"{p['achieved_tflops']:>9.4f}  {p['achieved_bw_gbps']:>8.2f}  "
            f"{p['arithmetic_intensity_flops_per_byte']:>8.3f}  "
            f"{p.get('roofline_bound', '?'):>8}  "
            f"{p.get('pct_of_roofline', float('nan')):>7.2f}  "
            f"{p.get('efficiency_label', 'unknown')}")
    if skipped:
        logger.warning(f'{len(skipped)} row(s) skipped:')
        for s in skipped[:10]:
            logger.warning(f"  T={s['T']} op={s['op']}: {s['reason']}")

    metadata = {
        'device': device,
        **peaks,
        'seed': ROOFLINE_SEED,
        'seed_info': seed_info,
        'provenance': capture_provenance(),
        'source': {'exp2_benchmark': {'path': src, 'schema_version': src_version,
                                      'n_rows': len(rows)}},
        'flops_model': (
            'analytic prefill FLOPs at the Exp 2 geometry: core-only '
            '(causal QK^T + AV, or the KDA recurrence + chunk terms) for '
            'core-boundary rows; run_kv_cache.prefill_flops end-to-end '
            'accounting for csa/hca/hybrid rows.'),
        'byte_model': _BYTE_MODEL_NOTE,
        'interpretation': (
            'Compare pct_of_roofline ACROSS rows with the same boundary only; '
            'Exp 2 boundaries differ per op (core vs end-to-end vs 5-layer). '
            'The roofline ceiling is a lower bound on achievable time: a '
            'reference PyTorch implementation is expected to sit far below '
            'it — the value of the number is in RANKING the operators and '
            'quantifying how much headroom a fused kernel could recover.'),
        'n_points': len(points),
        'n_skipped': len(skipped),
        'skipped': skipped,
    }
    os.makedirs('results', exist_ok=True)
    out = 'results/exp7_roofline.json'
    write_results(out, points, kind='exp7_roofline', metadata=metadata)
    logger.info(f'\nSaved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
