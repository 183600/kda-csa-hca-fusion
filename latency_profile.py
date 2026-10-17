#!/usr/bin/env python3
"""Operator-level latency profiling for the Mamba-2 (SSD) scan family.

Why a dedicated profiler?
-------------------------
``run_benchmark.py`` measures the sequential SSD scan at the core boundary
and ``test_chunked_ssd.py`` pins the chunked path's *correctness* — but the
repository owns no tool that answers the operator-level question the chunked
path was built for: **where does the time go, which chunk size should this
machine actually use, and how far is the implementation from the work model?**
This module supplies it:

* a seeded, dependency-light scan-input factory at the same core boundary
  and geometry as ``run_benchmark.bench_mamba2_scan`` (4 heads x 32x32
  state, capacity-matched to the KDA core row);
* ``time_call`` — warmup + repeated timing with median/min/max/std, an
  injectable clock and an optional synchronize hook (CUDA);
* ``profile_scan_modes`` — times the sequential recurrence, the shipped
  chunked path across candidate chunk sizes, and optionally the global
  quadratic dual-form oracle (the "full attention operator" reference
  point). Every row is **parity-checked against the sequential reference
  before its timing is kept**, so a profile can never silently measure a
  divergent operator;
* the analytic counterpart from :mod:`roofline`
  (``ssd_core_flops_breakdown``): the model's per-phase FLOPs and its
  argmin chunk size, printed next to the measured optimum.

Honest caveats
--------------
The model counts dominant MACs only (see
``roofline.ssd_core_flops_breakdown``): it has no Python/dispatch overhead,
no memory-hierarchy term, and no per-chunk fixed cost. Its argmin therefore
lands at smaller chunks than the measured optimum on CPU (small chunk
matmuls under-utilize BLAS and pay per-call overhead). Reporting BOTH
numbers is the point: the gap quantifies what a fused kernel could recover,
and the measured table is the authoritative device-specific answer.

Only the wall-clock of the *whole* scan call is measured here (no internal
phase hooks); the model supplies the phase decomposition. Profile numbers
are machine- and torch-build-specific and must never be compared across
hosts.

Usage::

    python3 latency_profile.py                           # default sweep
    python3 latency_profile.py --lengths 512,1024,2048 \
        --chunk-sizes 16,32,64,128 --quadratic
    python3 latency_profile.py --json results/scan_latency.json
"""
from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from ops_mamba import (chunked_mamba2_scan, naive_mamba2_scan,  # noqa: E402
                       quadratic_mamba2_scan)
import roofline  # noqa: E402

logger = logging.getLogger(__name__)

#: ``results_schema`` kind for the optional JSON output.
PROFILE_KIND = 'mamba2_scan_latency_profile'

DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048)
DEFAULT_CHUNK_SIZES = (16, 32, 64, 128)
#: Core-boundary geometry mirroring ``run_benchmark.bench_mamba2_scan``.
DEFAULT_GEOMETRY = {
    'batch': 1, 'heads': 4, 'headdim': 32, 'd_state': 32, 'ngroups': 1,
}
VALID_MODES = ('sequential', 'chunked', 'quadratic')


# ---------------------------------------------------------------------------
# Inputs and execution
# ---------------------------------------------------------------------------
def make_scan_inputs(
    T: int,
    *,
    batch: int = 1,
    heads: int = 4,
    headdim: int = 32,
    d_state: int = 32,
    ngroups: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device='cpu',
) -> dict:
    """Seeded scan inputs at the benchmark's core boundary.

    Value ranges mirror ``run_benchmark.bench_mamba2_scan``: ``x``/``B``/``C``
    are scaled normal draws, ``dt`` is uniform in ``[1e-3, 0.1]`` (post-
    softplus-like), ``A = -exp(log(1..heads))`` and ``D = 1``. A private
    CPU generator makes the draw reproducible on CPU and CUDA alike (the
    tensors are then materialized on ``device``).
    """
    if isinstance(T, bool) or not isinstance(T, int) or T < 0:
        raise ValueError(f'make_scan_inputs: T={T!r} must be a non-negative int.')
    gen = torch.Generator(device='cpu').manual_seed(int(seed) & 0xFFFFFFFF)
    x = (torch.randn(batch, T, heads, headdim, generator=gen,
                     dtype=torch.float32) * 0.1).to(device=device, dtype=dtype)
    dt = (torch.rand(batch, T, heads, generator=gen,
                     dtype=torch.float32) * 0.1 + 1e-3).to(device=device,
                                                            dtype=dtype)
    B = (torch.randn(batch, T, ngroups, d_state, generator=gen,
                     dtype=torch.float32) * 0.1).to(device=device, dtype=dtype)
    C = (torch.randn(batch, T, ngroups, d_state, generator=gen,
                     dtype=torch.float32) * 0.1).to(device=device, dtype=dtype)
    A = -torch.exp(torch.log(torch.arange(
        1, heads + 1, dtype=torch.float32))).to(device=device, dtype=dtype)
    D = torch.ones(heads, device=device, dtype=dtype)
    return {'x': x, 'dt': dt, 'B': B, 'C': C, 'A': A, 'D': D}


def run_scan_mode(mode: str, inputs: dict, chunk_size: int | None):
    """Execute one scan mode on pre-built inputs; returns ``(y, state)``."""
    if mode not in VALID_MODES:
        raise ValueError(
            f'run_scan_mode: mode={mode!r} must be one of {VALID_MODES}.')
    args = (inputs['x'], inputs['dt'], inputs['B'], inputs['C'],
            inputs['A'], inputs['D'])
    if mode == 'sequential':
        return naive_mamba2_scan(*args, output_final_state=True)
    if mode == 'quadratic':
        return quadratic_mamba2_scan(*args, output_final_state=True)
    if chunk_size is None:
        raise ValueError('run_scan_mode: chunked mode requires chunk_size.')
    return chunked_mamba2_scan(*args, output_final_state=True,
                               chunk_size=chunk_size)


def time_call(
    fn,
    *,
    warmup: int = 1,
    repeats: int = 3,
    timer=time.perf_counter,
    synchronize=None,
) -> dict:
    """Time ``fn`` with warmup runs and return median/min/max/std in ms.

    ``timer`` is any zero-argument clock (seconds); ``synchronize`` is
    called before and after each timed call (CUDA) and defaults to a no-op.
    The clock is injectable so the harness mechanics are testable without
    wall-clock flakiness. ``repeats >= 1`` is enforced: a median over an
    empty sample would be a silent lie.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f'time_call: warmup={warmup!r} must be a non-negative int.')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f'time_call: repeats={repeats!r} must be a positive int.')
    barrier = synchronize or (lambda: None)
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        barrier()
        t0 = timer()
        fn()
        barrier()
        times.append(max(0.0, timer() - t0))
    return {
        'median_ms': statistics.median(times) * 1e3,
        'min_ms': min(times) * 1e3,
        'max_ms': max(times) * 1e3,
        'std_ms': (statistics.stdev(times) * 1e3) if len(times) > 1 else 0.0,
        'times_ms': [t * 1e3 for t in times],
    }


# ---------------------------------------------------------------------------
# Model-side helpers
# ---------------------------------------------------------------------------
def model_best_chunk_size(T: int, *, heads: int, headdim: int, d_state: int,
                          candidates) -> int:
    """Chunk size minimizing the ANALYTIC scan FLOPs on ``candidates``.

    This is a work-model argmin, not a latency prediction (the profiler
    prints it next to the measured optimum). Ties break to the smaller
    chunk size, so the result is deterministic. ``candidates`` must be a
    non-empty iterable of positive ints.
    """
    cand = list(candidates)
    if not cand:
        raise ValueError('model_best_chunk_size: candidates must be non-empty.')
    scored = sorted(
        (roofline.ssd_core_flops(T, heads, headdim, d_state, chunked=True,
                                 chunk_size=int(c)),
         int(c))
        for c in cand)
    return scored[0][1]


def recommend_chunk_sizes(rows) -> dict:
    """Measured optimum per T over the chunked rows (median time).

    Returns ``{T: chunk_size}``; T's with no chunked row are omitted.
    Ties break to the smaller chunk size for determinism.
    """
    by_t: dict = {}
    for row in rows:
        if row.get('mode') != 'chunked' or row.get('time_ms') is None:
            continue
        T = row['T']
        cand = (row['time_ms'], row['chunk_size'])
        if T not in by_t or cand < by_t[T]:
            by_t[T] = cand
    return {T: C for T, (_, C) in by_t.items()}


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def _parity_max_abs(y, state, y_ref, s_ref) -> float:
    """Max absolute deviation from the reference (0.0 for empty tensors)."""
    dev = 0.0
    if y is not None and y_ref is not None and y.numel() and y_ref.numel():
        dev = max(dev, float((y - y_ref).abs().max().item()))
    if state is not None and s_ref is not None \
            and state.numel() and s_ref.numel():
        dev = max(dev, float((state - s_ref).abs().max().item()))
    return dev


def _all_finite(*tensors) -> bool:
    for t in tensors:
        if t is not None and not bool(torch.isfinite(t).all().item()):
            return False
    return True


def _profile_row(
    *, T, mode, chunk_size, inputs, y_ref, s_ref, geometry,
    warmup, repeats, timer, synchronize, runner, parity_tol,
) -> dict:
    """One (T, mode, chunk_size) row: parity gate first, timing second."""
    chunk_arg = chunk_size if mode == 'chunked' else None

    def call():
        with torch.no_grad():
            return runner(mode, inputs, chunk_arg)

    y, state = call()
    parity = 0.0 if mode == 'sequential' else _parity_max_abs(
        y, state, y_ref, s_ref)
    finite = _all_finite(y, state)
    timing = time_call(call, warmup=warmup, repeats=repeats, timer=timer,
                       synchronize=synchronize)
    H, D, N = geometry['heads'], geometry['headdim'], geometry['d_state']
    if mode == 'chunked':
        breakdown = roofline.ssd_core_flops_breakdown(
            T, H, D, N, chunk_size=int(chunk_size))
        flops = breakdown['total']
        n_chunks, levels = breakdown['n_chunks'], breakdown['scan_levels']
    elif mode == 'quadratic':
        flops = (roofline.ssd_core_flops_breakdown(
            T, H, D, N, chunk_size=max(T, 1))['total'] if T > 0 else 0.0)
        n_chunks, levels = None, None
    else:
        flops = roofline.ssd_core_flops(T, H, D, N)
        n_chunks, levels = None, None
    med = timing['median_ms']
    row = {
        'T': T,
        'mode': mode,
        'chunk_size': int(chunk_size) if chunk_size is not None else None,
        'time_ms': med,
        'time_min_ms': timing['min_ms'],
        'time_max_ms': timing['max_ms'],
        'time_std_ms': timing['std_ms'],
        'times_ms': timing['times_ms'],
        'analytic_flops': flops,
        'achieved_gflops': (flops / med * 1e-6) if med > 0 else None,
        'n_chunks': n_chunks,
        'scan_levels': levels,
        'parity_max_abs': parity,
        'parity_ok': bool(finite and parity <= parity_tol),
        'finite': bool(finite),
    }
    return row


def profile_scan_modes(
    lengths,
    *,
    chunk_sizes=DEFAULT_CHUNK_SIZES,
    geometry: dict | None = None,
    device='cpu',
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    warmup: int = 1,
    repeats: int = 3,
    timer=time.perf_counter,
    synchronize=None,
    runner=run_scan_mode,
    include_quadratic: bool = False,
    max_quadratic_len: int = 512,
    parity_tol: float = 1e-4,
) -> dict:
    """Measure the SSD scan modes over a ``lengths`` x ``chunk_sizes`` grid.

    Returns a profile dict:
      * ``rows`` — one dict per (T, mode[, chunk_size]); every non-sequential
        row records ``parity_max_abs`` against the sequential reference at
        the same T (and ``parity_ok = finite and parity <= parity_tol``),
        so the profile carries its own correctness gate;
      * ``model_best_chunk_size`` / ``measured_best_chunk_size`` — analytic
        vs measured optima per T;
      * the sweep settings and geometry for provenance.

    ``runner`` and ``timer`` are injectable for deterministic tests; the
    defaults execute the real ``ops_mamba`` scans on a real clock.
    """
    geo = dict(DEFAULT_GEOMETRY)
    geo.update(geometry or {})
    lengths = list(lengths)
    chunk_sizes = list(chunk_sizes)
    rows: list[dict] = []
    model_best: dict = {}
    for T in lengths:
        inputs = make_scan_inputs(T, device=device, dtype=dtype, seed=seed,
                                  **geo)
        with torch.no_grad():
            y_ref, s_ref = runner('sequential', inputs, None)
        rows.append(_profile_row(
            T=T, mode='sequential', chunk_size=None, inputs=inputs,
            y_ref=y_ref, s_ref=s_ref, geometry=geo, warmup=warmup,
            repeats=repeats, timer=timer, synchronize=synchronize,
            runner=runner, parity_tol=parity_tol))
        for C in chunk_sizes:
            rows.append(_profile_row(
                T=T, mode='chunked', chunk_size=int(C), inputs=inputs,
                y_ref=y_ref, s_ref=s_ref, geometry=geo, warmup=warmup,
                repeats=repeats, timer=timer, synchronize=synchronize,
                runner=runner, parity_tol=parity_tol))
        if include_quadratic and T <= max_quadratic_len:
            rows.append(_profile_row(
                T=T, mode='quadratic', chunk_size=None, inputs=inputs,
                y_ref=y_ref, s_ref=s_ref, geometry=geo, warmup=warmup,
                repeats=repeats, timer=timer, synchronize=synchronize,
                runner=runner, parity_tol=parity_tol))
        model_best[T] = model_best_chunk_size(
            T, heads=geo['heads'], headdim=geo['headdim'],
            d_state=geo['d_state'], candidates=chunk_sizes)
    return {
        'rows': rows,
        'geometry': geo,
        'device': str(device),
        'dtype': str(dtype).replace('torch.', ''),
        'seed': seed,
        'warmup': warmup,
        'repeats': repeats,
        'parity_tol': parity_tol,
        'chunk_sizes': [int(c) for c in chunk_sizes],
        'model_best_chunk_size': model_best,
        'measured_best_chunk_size': recommend_chunk_sizes(rows),
    }


def summarize(profile: dict) -> list[dict]:
    """Per-T summary joining measured optimum, model optimum and speedup."""
    out: list[dict] = []
    rows = profile['rows']
    for T in sorted({r['T'] for r in rows}):
        seq = next((r for r in rows if r['T'] == T and r['mode'] == 'sequential'),
                   None)
        best_C = profile['measured_best_chunk_size'].get(T)
        best = next((r for r in rows if r['T'] == T and r['mode'] == 'chunked'
                     and r['chunk_size'] == best_C), None)
        speedup = None
        if seq and best and seq['time_ms'] and best['time_ms']:
            speedup = seq['time_ms'] / best['time_ms']
        out.append({
            'T': T,
            'sequential_ms': seq['time_ms'] if seq else None,
            'best_chunk_size': best_C,
            'best_chunked_ms': best['time_ms'] if best else None,
            'speedup_vs_sequential': speedup,
            'model_best_chunk_size': profile['model_best_chunk_size'].get(T),
            'parity_ok': all(r['parity_ok'] for r in rows if r['T'] == T),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_int_list(raw: str, name: str, *, min_value: int = 1) -> list[int]:
    try:
        values = [int(part) for part in raw.split(',') if part.strip()]
    except ValueError:
        raise SystemExit(f'--{name}: expected comma-separated ints, got {raw!r}')
    if not values or any(v < min_value for v in values):
        raise SystemExit(
            f'--{name}: values must be >= {min_value}, got {values!r}')
    return list(dict.fromkeys(values))


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _write_json(path: str, profile: dict) -> None:
    """Write the profile as a versioned ``results_schema`` envelope."""
    from results_schema import write as write_results
    from kaggle_setup import capture_provenance
    geometry = profile['geometry']
    metadata = {
        'device': profile['device'],
        'dtype': profile['dtype'],
        'seed': profile['seed'],
        'warmup': profile['warmup'],
        'repeats': profile['repeats'],
        'parity_tol': profile['parity_tol'],
        'chunk_sizes': profile['chunk_sizes'],
        'geometry': geometry,
        'model_best_chunk_size': profile['model_best_chunk_size'],
        'measured_best_chunk_size': profile['measured_best_chunk_size'],
        'summary': summarize(profile),
        'flops_model': (
            'roofline.ssd_core_flops_breakdown: dominant-MAC accounting of '
            'the chunked SSD decomposition (intra-chunk masked quadratic '
            'attention + chunk source + inter-chunk readout + associative '
            'scan passes); exp/mask/skip terms excluded.'),
        'caveat': (
            'Wall-clock is host- and torch-build-specific. parity_max_abs '
            'is the max deviation from the sequential reference scan at the '
            'same T; parity_ok gates each row at parity_tol.'),
        'provenance': capture_provenance(),
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    write_results(path, profile['rows'], kind=PROFILE_KIND, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Latency-profile the Mamba-2 (SSD) scan modes and '
                    'compare the measured optimal chunk size with the '
                    'analytic work model.')
    parser.add_argument('--lengths', default=','.join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument('--chunk-sizes', default=','.join(
        map(str, DEFAULT_CHUNK_SIZES)))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--headdim', type=int, default=32)
    parser.add_argument('--d-state', type=int, default=32)
    parser.add_argument('--ngroups', type=int, default=1)
    parser.add_argument('--dtype', default='float32',
                        choices=('float32', 'float64'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quadratic', action='store_true',
                        help='also time the global O(T^2) dual-form oracle '
                             '(the full-attention reference point)')
    parser.add_argument('--max-quadratic-len', type=int, default=512)
    parser.add_argument('--threads', type=int, default=1,
                        help='torch CPU threads during measurement '
                             '(1 = stable CPU numbers, matching '
                             'run_benchmark)')
    parser.add_argument('--json', default=None,
                        help='optional path to write the profile as a '
                             'results_schema envelope')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    lengths = _parse_int_list(args.lengths, 'lengths', min_value=1)
    chunk_sizes = _parse_int_list(args.chunk_sizes, 'chunk-sizes')
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit('--repeats must be >= 1 and --warmup >= 0.')
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    device = _resolve_device(args.device)
    synchronize = torch.cuda.synchronize if str(device).startswith('cuda') else None

    geometry = {'batch': args.batch, 'heads': args.heads,
                'headdim': args.headdim, 'd_state': args.d_state,
                'ngroups': args.ngroups}
    logger.info('=' * 72)
    logger.info('Mamba-2 (SSD) scan latency profile | device=%s dtype=%s',
                device, args.dtype)
    logger.info('geometry: %s', geometry)
    logger.info('lengths=%s chunk_sizes=%s repeats=%d',
                lengths, chunk_sizes, args.repeats)
    logger.info('=' * 72)

    prev_threads = torch.get_num_threads()
    if not str(device).startswith('cuda'):
        torch.set_num_threads(max(1, args.threads))
    try:
        profile = profile_scan_modes(
            lengths, chunk_sizes=chunk_sizes, geometry=geometry,
            device=device, dtype=dtype, seed=args.seed, warmup=args.warmup,
            repeats=args.repeats, synchronize=synchronize,
            include_quadratic=args.quadratic,
            max_quadratic_len=args.max_quadratic_len)
    finally:
        torch.set_num_threads(prev_threads)

    width = max([len(r['mode']) for r in profile['rows']] + [8])
    logger.info('')
    logger.info('%6s  %-*s  %5s  %10s  %10s  %12s  %8s',
                'T', width, 'mode', 'C', 'ms', 'GFLOP/s', 'model MFLOPs',
                'parity')
    for r in profile['rows']:
        gflops = (f"{r['achieved_gflops']:.3f}"
                  if r['achieved_gflops'] is not None else 'n/a')
        logger.info('%6d  %-*s  %5s  %10.4f  %10s  %12.2f  %8.1e%s',
                    r['T'], width, r['mode'],
                    '-' if r['chunk_size'] is None else r['chunk_size'],
                    r['time_ms'], gflops, r['analytic_flops'] / 1e6,
                    r['parity_max_abs'], '' if r['parity_ok'] else '  PARITY!')
    logger.info('')
    for s in summarize(profile):
        if s['best_chunk_size'] is None:
            continue
        speed = (f"{s['speedup_vs_sequential']:.1f}x"
                 if s['speedup_vs_sequential'] else 'n/a')
        logger.info(
            'T=%d: measured best C=%s (%.4f ms, %s vs sequential); '
            'model argmin C=%s%s',
            s['T'], s['best_chunk_size'], s['best_chunked_ms'], speed,
            s['model_best_chunk_size'],
            '' if s['parity_ok'] else '  [PARITY FAILURE]')
    if args.json:
        _write_json(args.json, profile)
        logger.info('Saved: %s', args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
