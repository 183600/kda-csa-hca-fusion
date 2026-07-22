"""Experiment 2 — latency and memory benchmark (device-aware).

Measures wall-clock latency and peak memory of each attention operator
(softmax attention baseline, KDA, CSA, HCA, and the fused hybrid) across a
range of sequence lengths.

Kaggle / review-driven additions (address reviewer concerns):

  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling
    back to CPU. On GPU we use CUDA events for accurate timing and
    ``torch.cuda.max_memory_allocated`` for real peak memory.
  * **Real memory measurement.** On GPU we report the actual
    ``torch.cuda.max_memory_allocated`` (the number a production inference
    engine would pay), not just the Python-traced allocation. On CPU we keep
    ``tracemalloc`` for the resident Python heap.
  * **All advertised sequence lengths.** The original paper's text mentioned
    T in {128, 256, 512, 1024, 2048} but the table omitted T=256. We include
    every length.
  * **CUDA-event timing.** On GPU we synchronize and use CUDA events so the
    timing reflects actual kernel execution, not host-side launch overhead.
  * **Memory cleared between runs.** ``gc.collect()`` + ``torch.cuda.empty_cache()``
    (on GPU) between operators so peak memory is per-operator, not cumulative.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import statistics
import sys
import time
import zlib

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    configure_torch_for_device, parse_int_env, sanitize_for_json,
    write_json_atomic, capture_provenance, make_seeded_generator,
)
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_fused import HybridKCHAttention, HybridConfig

logger = logging.getLogger(__name__)


def _selected_kda_backend() -> str:
    """Return the explicitly requested benchmark KDA backend.

    The benchmark remains reference-first; set ``KDA_BACKEND=fla`` or
    ``KDA_BACKEND=auto`` to measure the optional FLA path separately.
    """
    return validate_kda_backend(os.environ.get('KDA_BACKEND', 'reference'))


_LAST_TIMING_STATS: dict = {}


def _rand(*shape, scale=0.1, device=None, dtype=None, generator=None):
    t = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    return t * scale


def _make_op_gen(op_name, T, device):
    """Build a seeded ``torch.Generator`` keyed on (op_name, T).

    Uses ``zlib.crc32`` which is a stable, process-independent hash of the
    byte string, so the seed is reproducible without any env-var cooperation.

    Older torch / CPU-only builds don't support ``torch.Generator(device='cuda')``
    (raises RuntimeError). Fall back to a CPU generator in that case;
    ``torch.randn(..., device='cuda', generator=cpu_gen)`` is supported and
    still produces reproducible (CPU-seeded, then GPU-materialized) draws.
    """
    name_hash = zlib.crc32(op_name.encode('utf-8')) & 0xFFFFFFFF
    t_hash = (T * 2654435761) & 0xFFFFFFFF
    seed = name_hash ^ t_hash
    return make_seeded_generator(seed, device=device)


def _clear_cache(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _measure(fn, repeats, device):
    """Return (median_wall_time_s, peak_memory_MB).

    On CUDA: uses CUDA events for timing and ``torch.cuda.max_memory_allocated``
    (with baseline subtraction) for the real activation peak memory — the
    number that characterises the operator's memory behavior, isolated from
    the constant model-parameter offset that varies per operator.

    On CPU: uses ``time.perf_counter`` for timing. Peak memory is reported
    as ``None`` because torch's native CPU allocator retains freed blocks
    in its pool, so RSS-based sampling between calls reports 0 for every
    operator after the first (the pool is reused, not returned to the OS).
    See the CPU-path comment below for details. JSON serializes ``None`` to
    ``null``, which is clearly "no data" rather than the misleading 0.0.
    """
    if device.type == 'cuda':
        # Warmup
        for _ in range(min(2, repeats)):
            out = fn()
            del out
        torch.cuda.synchronize()
        # Capture the baseline allocation AFTER the reset. The model
        # parameters and any persistent state (e.g. KDA recurrent state)
        # are still in ``memory_allocated()`` at this point, so without
        # subtracting the baseline the reported peak would be
        # ``params + peak_activations`` — a constant offset that varies
        # per operator (different parameter counts) and makes the
        # cross-operator comparison unfair. Subtracting the baseline
        # isolates the activation footprint, which is the number that
        # actually characterises an operator's memory behavior.
        baseline_bytes = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        peak_bytes = 0
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(device)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            # elapsed_time returns milliseconds; convert to seconds to
            # preserve the (seconds, MB) contract of this function.
            times.append(start.elapsed_time(end) / 1000.0)
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
            del out
            torch.cuda.synchronize()
        peak_mb = max(0.0, peak_bytes - baseline_bytes) / (1024 ** 2)
        _LAST_TIMING_STATS['times'] = list(times)
        _LAST_TIMING_STATS['min_ms'] = (min(times) * 1000.0) if times else 0.0
        _LAST_TIMING_STATS['max_ms'] = (max(times) * 1000.0) if times else 0.0
        _LAST_TIMING_STATS['std_ms'] = (
            (statistics.stdev(times) * 1000.0) if len(times) >= 2 else 0.0
        )
        # Guard against repeats=0: ``statistics.median([])`` raises
        # ``StatisticsError``. Returns 0.0 so the JSON row is well-formed.
        return (statistics.median(times) if times else 0.0), peak_mb
    else:
        # CPU path: we previously tried sampling VmRSS from /proc/self/status
        # between ``fn()`` calls, but torch's native CPU allocator (like
        # malloc) retains freed blocks in its pool for reuse rather than
        # returning them to the OS. So after the first operator pushes RSS
        # to a new high, every subsequent operator reuses the pool and the
        # sampled RSS delta is 0 — making every row after the first report
        # ``0.00 MB``, which is misleading (it looks like a real measurement
        # but is in fact "no new RSS growth detected").
        #
        # tracemalloc is even worse: it only traces the Python heap and
        # silently ignores native torch tensor allocations, so it reports
        # ~0 MB for tensors that are actually megabytes.
        #
        # The honest choice is to report ``None`` on CPU (matching
        # ``run_decoding.py``), so JSON serializes to ``null`` (clearly "no
        # data") rather than a misleading 0.0. On GPU we report the real
        # ``torch.cuda.max_memory_allocated`` (with baseline subtraction).
        # Warmup
        for _ in range(min(2, repeats)):
            out = fn()
            del out
        gc.collect()
        _prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                del out
        finally:
            torch.set_num_threads(_prev_threads)
        peak_mb = None  # CPU peak memory is not reliably measurable
        _LAST_TIMING_STATS['times'] = list(times)
        _LAST_TIMING_STATS['min_ms'] = (min(times) * 1000.0) if times else 0.0
        _LAST_TIMING_STATS['max_ms'] = (max(times) * 1000.0) if times else 0.0
        _LAST_TIMING_STATS['std_ms'] = (
            (statistics.stdev(times) * 1000.0) if len(times) >= 2 else 0.0
        )
        return (statistics.median(times) if times else 0.0), peak_mb


def bench_softmax_attn(B, T, H, K, V, device):
    gen = _make_op_gen('softmax', T, device)
    q = _rand(B, T, H, K, device=device, generator=gen)
    k = _rand(B, T, H, K, device=device, generator=gen)
    v = _rand(B, T, H, V, device=device, generator=gen)
    scale = K ** -0.5
    # Precompute the causal mask OUTSIDE the timed region so the benchmark
    # measures attention compute, not mask allocation. A production
    # implementation would cache this mask (it is the same every call) rather
    # than reconstructing a [T, T] tensor per forward. For T=2048 this avoids
    # a 4M-element allocation + fill on every timed iteration, which previously
    # inflated softmax's measured latency and made the cross-operator
    # comparison unfair (KDA/CSA/HCA don't have this per-call [T, T]
    # allocation overhead in their bench wrappers).
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
    ).view(1, 1, T, T)

    def fn():
        with torch.no_grad():
            scores = torch.einsum('bthk,bshk->bhts', q, k) * scale
            scores = scores.masked_fill(causal_mask, float('-inf'))
            p = torch.softmax(scores, dim=-1)
            return torch.einsum('bhts,bshv->bthv', p, v)
    return fn


def bench_kda_recurrent(B, T, H, K, V, device):
    gen = _make_op_gen('kda_rec', T, device)
    q = torch.nn.functional.normalize(_rand(B, T, H, K, device=device, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(_rand(B, T, H, K, device=device, generator=gen), dim=-1)
    v = _rand(B, T, H, V, device=device, generator=gen)
    g = -torch.rand(B, T, H, K, device=device, generator=gen) * 0.05
    beta = torch.rand(B, T, H, device=device, generator=gen) * 0.2
    backend = _selected_kda_backend()

    def fn():
        with torch.no_grad():
            return kda_forward(
                q, k, v, g, beta,
                output_final_state=True,
                use_chunk=False,
                backend=backend,
            )
    return fn


def bench_kda_chunk(B, T, H, K, V, device):
    gen = _make_op_gen('kda_chunk', T, device)
    q = torch.nn.functional.normalize(_rand(B, T, H, K, device=device, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(_rand(B, T, H, K, device=device, generator=gen), dim=-1)
    v = _rand(B, T, H, V, device=device, generator=gen)
    g = -torch.rand(B, T, H, K, device=device, generator=gen) * 0.05
    beta = torch.rand(B, T, H, device=device, generator=gen) * 0.2
    backend = _selected_kda_backend()
    BT = 64

    def fn():
        with torch.no_grad():
            o, s = kda_forward(
                q, k, v, g, beta,
                output_final_state=True,
                chunk_size=BT,
                use_chunk=True,
                backend=backend,
            )
            assert o.shape[1] == T, (
                f"naive_chunk_kda output shape {tuple(o.shape)} does not "
                f"match input T={T} on dim=1 (sequence). The chunk path "
                f"is supposed to right-pad T up to a multiple of "
                f"chunk_size={BT} and trim back to original_T internally; "
                f"a shape mismatch indicates a regression in the trim "
                f"logic.")
            return o, s
    return fn


def bench_csa(B, T, d, device):
    m, topk = 8, 4
    nh, c, dc, nIh, cI = 4, 16, 32, 2, 8
    gen = _make_op_gen('csa', T, device)
    H = _rand(B, T, d, device=device, generator=gen)
    cfg = dict(
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=cI, dc=dc,
        sliding_window=8, sink_logits=torch.zeros(nh, device=device),
        use_ste=False,
        normalize_qk=True,
    )
    weights = dict(
        W_aKV=_rand(c, d, device=device, generator=gen), W_bKV=_rand(c, d, device=device, generator=gen),
        W_aZ=_rand(c, d, device=device, generator=gen), W_bZ=_rand(c, d, device=device, generator=gen),
        Ba=_rand(m, c, device=device, generator=gen), Bb=_rand(m, c, device=device, generator=gen),
        W_DQ=_rand(dc, d, device=device, generator=gen), W_UQ=_rand(dc, c * nh, device=device, generator=gen),
        W_IUQ=_rand(cI * nIh, dc, device=device, generator=gen), W_w=_rand(nIh, d, device=device, generator=gen),
        W_KV_idx=_rand(cI, d, device=device, generator=gen), W_Z_idx=_rand(cI, d, device=device, generator=gen),
        B_idx=_rand(m, cI, device=device, generator=gen),
    )
    W_O = _rand(d, nh * c, device=device, generator=gen)

    def fn():
        with torch.no_grad():
            out = naive_csa(H, **weights, **cfg)
            return torch.nn.functional.linear(out, W_O)
    return fn


def bench_hca(B, T, d, device):
    m2, nh, c, dc = 16, 4, 16, 32
    gen = _make_op_gen('hca', T, device)
    H = _rand(B, T, d, device=device, generator=gen)
    cfg = dict(
        m2=m2, nh=nh, c=c, dc=dc,
        sliding_window=8, sink_logits=torch.zeros(nh, device=device),
    )
    weights = dict(
        W_KV=_rand(c, d, device=device, generator=gen), W_Z=_rand(c, d, device=device, generator=gen),
        B_pos=_rand(m2, c, device=device, generator=gen),
        W_DQ=_rand(dc, d, device=device, generator=gen), W_UQ=_rand(dc, c * nh, device=device, generator=gen),
    )
    W_O = _rand(d, nh * c, device=device, generator=gen)

    def fn():
        with torch.no_grad():
            out = naive_hca(H, **weights, **cfg)
            return torch.nn.functional.linear(out, W_O)
    return fn


def bench_hybrid(B, T, d, device):
    gen = _make_op_gen('hybrid', T, device)
    cfg = HybridConfig(
        d_model=d, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
        kda_backend=_selected_kda_backend(),
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    x = _rand(B, T, d, device=device, generator=gen)
    def fn():
        with torch.no_grad():
            model.reset_state()
            return model(x)
    return fn


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info(f'Experiment 2: Latency & Memory Benchmark ({device})')
    logger.info('=' * 70)
    default_lengths = '128,256,512,1024,2048'
    raw = os.environ.get('BENCH_LENGTHS', default_lengths)
    try:
        seq_lengths = [int(x) for x in raw.split(',') if x.strip()]
    except ValueError:
        logger.warning(f'[run_benchmark] invalid BENCH_LENGTHS={raw!r}; using default.')
        seq_lengths = [int(x) for x in default_lengths.split(',')]
    if not seq_lengths:
        seq_lengths = [int(x) for x in default_lengths.split(',')]
    if any(t < 1 for t in seq_lengths):
        raise ValueError(
            f'BENCH_LENGTHS must contain positive sequence lengths, got {seq_lengths!r}')
    seq_lengths = list(dict.fromkeys(seq_lengths))
    logger.info(f'[run_benchmark] seq_lengths = {seq_lengths}')
    B, H, K, V, d = 1, 4, 32, 32, 64

    benches = [
        ('softmax',  lambda T: bench_softmax_attn(B, T, H, K, V, device)),
        ('kda_rec',  lambda T: bench_kda_recurrent(B, T, H, K, V, device)),
        ('kda_chunk', lambda T: bench_kda_chunk(B, T, H, K, V, device)),
        ('csa',      lambda T: bench_csa(B, T, d, device)),
        ('hca',      lambda T: bench_hca(B, T, d, device)),
        ('hybrid',   lambda T: bench_hybrid(B, T, d, device)),
    ]
    op_boundary = {
        'softmax':   {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'attention core only; q/k/v pre-projected'},
        'kda_rec':   {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'recurrence core only; q/k/v/g/beta pre-projected'},
        'kda_chunk': {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'chunked recurrence core only; q/k/v/g/beta pre-projected'},
        'csa':       {'compute_boundary': 'end_to_end_single_layer',
                      'n_layers': 1,
                      'note': 'single CSA layer from hidden state H (includes all projections + compression + indexer + sparse attention + o_proj)'},
        'hca':       {'compute_boundary': 'end_to_end_single_layer',
                      'n_layers': 1,
                      'note': 'single HCA layer from hidden state H (includes all projections + compression + dense attention + o_proj)'},
        'hybrid':    {'compute_boundary': 'end_to_end_multi_layer',
                      'n_layers': 5,
                      'note': '5-layer KDA+CSA+HCA stack with LayerNorm, projections, attention, state management (3:1:1 default ratio)'},
    }

    results = []
    n_repeats = parse_int_env('BENCH_REPEATS', 5, min_value=1, logger=logger)
    for T in seq_lengths:
        logger.info(f'\n-- T = {T} --')
        for name, factory in benches:
            try:
                _clear_cache(device)
                fn = factory(T)
                t, mem = _measure(fn, repeats=n_repeats, device=device)
                row = {'T': T, 'op': name, 'time_ms': t * 1e3, 'peak_mem_MB': mem,
                       'device': str(device), 'repeats': n_repeats}
                if name in {'kda_rec', 'kda_chunk', 'hybrid'}:
                    row['kda_backend'] = _selected_kda_backend()
                if name in {'csa', 'hybrid'}:
                    row['csa_indexer_normalize_qk'] = True
                    row['csa_ste_in_timed_region'] = False
                row['time_min_ms'] = _LAST_TIMING_STATS.get('min_ms')
                row['time_max_ms'] = _LAST_TIMING_STATS.get('max_ms')
                row['time_std_ms'] = _LAST_TIMING_STATS.get('std_ms')
                row.update(op_boundary[name])
                results.append(row)
                mem_str = f'{mem:8.2f} MB' if mem is not None else '     n/a'
                print(f'  {name:12s}  time={t*1e3:8.2f} ms  mem={mem_str}')
            except Exception as e:
                err_row = {
                    'T': T, 'op': name, 'error': str(e),
                    'device': str(device),
                    'kda_backend': (os.environ.get('KDA_BACKEND', 'reference')
                                    if name in {'kda_rec', 'kda_chunk', 'hybrid'} else None),
                    'time_ms': None, 'peak_mem_MB': None,
                    'time_min_ms': None, 'time_max_ms': None,
                    'time_std_ms': None,
                    'repeats': n_repeats,
                }
                if name in {'csa', 'hybrid'}:
                    err_row['csa_indexer_normalize_qk'] = True
                    err_row['csa_ste_in_timed_region'] = False
                err_row.update(op_boundary[name])
                results.append(err_row)
                logger.error(f'  {name:12s}  ERROR: {e}')

    os.makedirs('results', exist_ok=True)
    try:
        write_json_atomic(results, 'results/exp2_benchmark.json',
                          indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in results; sanitizing to null: {e}')
        write_json_atomic(sanitize_for_json(results),
                          'results/exp2_benchmark.json',
                          indent=2, allow_nan=False)
    try:
        write_json_atomic(capture_provenance(),
                          'results/exp2_benchmark_provenance.json',
                          indent=2, allow_nan=False)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info('\nSaved: results/exp2_benchmark.json')
    n_errors = sum(1 for r in results if 'error' in r)
    if n_errors:
        logger.error(
            f'\n[P0-2] {n_errors}/{len(results)} (T, op) cells errored out. '
            f'Returning non-zero so run_all records this experiment as failed.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
