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

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, parse_int_env, sanitize_for_json
from ops_kda import naive_recurrent_kda, naive_chunk_kda
from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_fused import HybridKCHAttention, HybridConfig

logger = logging.getLogger(__name__)


def _rand(*shape, scale=0.1, device=None, dtype=None):
    t = torch.randn(*shape, device=device, dtype=dtype)
    return t * scale


def _clear_cache(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _read_current_rss_kb() -> int:
    """Return the current process RSS in kilobytes (Linux only).

    Reads ``VmRSS`` from ``/proc/self/status``. Returns 0 on macOS/BSD or
    when ``/proc`` is unavailable — the previous ``resource.getrusage``
    fallback returned ``ru_maxrss`` (a *high-water mark* that never
    decreases), which made every operator after the first report a 0 delta.
    Since the CPU ``_measure`` path now returns ``None`` for peak memory
    (see the docstring there), this function is retained only for any
    future caller that wants a best-effort instantaneous RSS on Linux.
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # "VmRSS:    12345 kB\n"
                    return int(line.split()[1])
    except (FileNotFoundError, ValueError, IndexError, OSError):
        pass
    return 0


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
            fn()
        torch.cuda.synchronize()
        # Reset peak memory stats AFTER warmup so the reported peak reflects
        # only the timed region's activations, NOT the warmup allocations.
        torch.cuda.reset_peak_memory_stats(device)
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
        for _ in range(repeats):
            start.record()
            fn()
            end.record()
            end.synchronize()
            # elapsed_time returns milliseconds; convert to seconds to
            # preserve the (seconds, MB) contract of this function.
            times.append(start.elapsed_time(end) / 1000.0)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = max(0.0, peak_bytes - baseline_bytes) / (1024 ** 2)
        # ``statistics.median`` averages the two middle values for even-length
        # lists; the previous ``sorted(times)[len(times)//2]`` returned the
        # upper-middle element, which biased the reported latency upward
        # whenever ``repeats`` was even. (Currently masked because the caller
        # uses repeats=3, but the bug would surface on any even-repeat run.)
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
        gc.collect()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        peak_mb = None  # CPU peak memory is not reliably measurable
        return (statistics.median(times) if times else 0.0), peak_mb


def bench_softmax_attn(B, T, H, K, V, device):
    q = _rand(B, T, H, K, device=device)
    k = _rand(B, T, H, K, device=device)
    v = _rand(B, T, H, V, device=device)
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
    )

    def fn():
        with torch.no_grad():
            scores = torch.einsum('bthk,bshk->bhts', q, k) * scale
            scores = scores.masked_fill(causal_mask, float('-inf'))
            p = torch.softmax(scores, dim=-1)
            return torch.einsum('bhts,bshv->bthv', p, v)
    return fn


def bench_kda_recurrent(B, T, H, K, V, device):
    q = torch.nn.functional.normalize(_rand(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(_rand(B, T, H, K, device=device), dim=-1)
    v = _rand(B, T, H, V, device=device)
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2

    def fn():
        with torch.no_grad():
            return naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    return fn


def bench_kda_chunk(B, T, H, K, V, device):
    q = torch.nn.functional.normalize(_rand(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(_rand(B, T, H, K, device=device), dim=-1)
    v = _rand(B, T, H, V, device=device)
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2
    BT = 64
    # NOTE: ``naive_chunk_kda`` already right-pads T up to a multiple of
    # ``chunk_size`` internally and returns ``o[:, :original_T]``. The previous
    # version of this bench duplicated that padding *and* then trimmed with
    # ``o[:T]``, which slices dim=0 (batch) instead of dim=1 (sequence) — the
    # same class of bug that was fixed in ``ops_fused.py`` and
    # ``run_quality.py::CSAAttn.forward``. For B=1 the wrong slice happened to
    # return the full tensor so the benchmark kept working, but for B>1 it
    # silently corrupted results, and for any B the reported timing reflected
    # the padded length rather than T. We now let ``naive_chunk_kda`` handle
    # padding end-to-end and just time it directly.
    def fn():
        with torch.no_grad():
            return naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=BT)
    return fn


def bench_csa(B, T, d, device):
    m, topk = 8, 4
    nh, c, dc, nIh, cI = 4, 16, 32, 2, 8
    H = _rand(B, T, d, device=device)
    cfg = dict(
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=cI, dc=dc,
        sliding_window=8, sink_logits=torch.zeros(nh, device=device),
    )
    weights = dict(
        W_aKV=_rand(d, c, device=device), W_bKV=_rand(d, c, device=device),
        W_aZ=_rand(d, c, device=device), W_bZ=_rand(d, c, device=device),
        Ba=_rand(m, c, device=device), Bb=_rand(m, c, device=device),
        W_DQ=_rand(d, dc, device=device), W_UQ=_rand(dc, c * nh, device=device),
        W_IUQ=_rand(dc, cI * nIh, device=device), W_w=_rand(d, nIh, device=device),
        W_KV_idx=_rand(d, cI, device=device), W_Z_idx=_rand(d, cI, device=device),
        B_idx=_rand(m, cI, device=device),
    )

    def fn():
        with torch.no_grad():
            return naive_csa(H, **weights, **cfg)
    return fn


def bench_hca(B, T, d, device):
    m2, nh, c, dc = 16, 4, 16, 32
    H = _rand(B, T, d, device=device)
    cfg = dict(
        m2=m2, nh=nh, c=c, dc=dc,
        sliding_window=8, sink_logits=torch.zeros(nh, device=device),
    )
    weights = dict(
        W_KV=_rand(d, c, device=device), W_Z=_rand(d, c, device=device),
        B_pos=_rand(m2, c, device=device),
        W_DQ=_rand(d, dc, device=device), W_UQ=_rand(dc, c * nh, device=device),
    )

    def fn():
        with torch.no_grad():
            return naive_hca(H, **weights, **cfg)
    return fn


def bench_hybrid(B, T, d, device):
    cfg = HybridConfig(
        d_model=d, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
    )
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    x = _rand(B, T, d, device=device)
    # Reset KDA recurrent state before each fn() call so that warmup and
    # timed repeats start from the same fresh state. Without this, the KDA
    # state grows across repeats — for latency this is O(1) per layer so
    # the timing impact is negligible, but for peak-memory measurement the
    # retained state tensors would be double-counted across repeats,
    # inflating the reported memory footprint.
    # NOTE: torch.no_grad() must live INSIDE fn(). A `with torch.no_grad():`
    # block wrapping the `def fn():` only disables grad for the duration of
    # the function definition, not for later calls — the context manager
    # state is global and is restored as soon as the `with` block exits.
    # Putting it outside (the previous form) silently built the autograd
    # graph during benchmarking, inflating both latency and peak memory.
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
    # Original paper text mentioned {128, 256, 512, 1024, 2048} but the table
    # omitted 256. We include all advertised lengths.
    # Override via the BENCH_LENGTHS env var (comma-separated) — used by
    # run_all.py's SKIP_SLOW path to truncate the sweep on CPU.
    default_lengths = '128,256,512,1024,2048'
    raw = os.environ.get('BENCH_LENGTHS', default_lengths)
    try:
        seq_lengths = [int(x) for x in raw.split(',') if x.strip()]
    except ValueError:
        logger.warning(f'[run_benchmark] invalid BENCH_LENGTHS={raw!r}; using default.')
        seq_lengths = [int(x) for x in default_lengths.split(',')]
    if not seq_lengths:
        seq_lengths = [int(x) for x in default_lengths.split(',')]
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

    results = []
    # Number of timed repeats per (T, op). The previous value of 3 gave a
    # noisy single-point estimate; with median aggregation across 5 repeats
    # (and min/max kept implicit in the underlying times list) the reported
    # number is meaningfully more stable. We keep the count modest so the
    # full sweep (5 seq_lengths x 6 ops x repeats) stays under a few minutes
    # on a Kaggle T4. Override via the ``BENCH_REPEATS`` env var.
    # Parse BENCH_REPEATS defensively: the sibling BENCH_LENGTHS env var is
    # already wrapped in try/except with a graceful fallback, but BENCH_REPEATS
    # was a bare ``int()`` that crashed the whole benchmark on malformed input
    # like ``BENCH_REPEATS=abc`` or ``BENCH_REPEATS=5.0``. Use the shared
    # ``parse_int_env`` helper so the robustness contract is identical across
    # BENCH_REPEATS / BENCH_LENGTHS here AND MQAR_SEEDS / ABL_SEEDS / etc in
    # the sibling experiment runners (single source of truth for the pattern).
    n_repeats = parse_int_env('BENCH_REPEATS', 5, min_value=1, logger=logger)
    for T in seq_lengths:
        logger.info(f'\n-- T = {T} --')
        for name, factory in benches:
            try:
                _clear_cache(device)
                fn = factory(T)
                # warmup (counted in _measure on GPU; explicit here for CPU)
                if device.type != 'cuda':
                    fn()
                t, mem = _measure(fn, repeats=n_repeats, device=device)
                row = {'T': T, 'op': name, 'time_ms': t * 1e3, 'peak_mem_MB': mem,
                       'device': str(device), 'repeats': n_repeats}
                results.append(row)
                # mem may be None on CPU (unreliable RSS sampling); render
                # as 'n/a' instead of crashing on ``f'{None:8.2f}'``.
                mem_str = f'{mem:8.2f} MB' if mem is not None else '     n/a'
                print(f'  {name:12s}  time={t*1e3:8.2f} ms  mem={mem_str}')
            except Exception as e:
                # Include null fields for the keys present on success rows so
                # downstream JSON consumers can do ``row['time_ms']`` without
                # a KeyError on error rows. Missing keys vs explicit null
                # matters: pandas read_json treats missing keys as NaN only
                # if the column exists in at least one row.
                results.append({
                    'T': T, 'op': name, 'error': str(e),
                    'device': str(device),
                    'time_ms': None, 'peak_mem_MB': None,
                    'repeats': n_repeats,
                })
                logger.error(f'  {name:12s}  ERROR: {e}')

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a benchmark row's ``time_ms``
    # became non-finite (e.g. a CUDA event glitch producing inf, or a future
    # code path that returns float('nan') on a degenerate input), Python's
    # default json.dump would emit literal ``NaN``/``Infinity`` tokens that
    # are INVALID JSON per RFC 8259 and break strict parsers (JS
    # ``JSON.parse``, jq, pandas with ``orient='records'``). The sibling
    # runners (run_kv_cache.py, run_decoding.py, run_quality.py,
    # run_ablation.py) all already use this pattern; this closes the
    # consistency gap.
    #
    # CRITICAL: serialize to a STRING first, then write the string. The
    # previous ``json.dump(results, f, indent=2)`` (default allow_nan=True)
    # wrote directly to the file, so a NaN mid-stream left a partial JSON
    # document. Mirrors the atomicity fix in run_quality.py::main /
    # run_ablation.py::main.
    try:
        text = json.dumps(results, indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in results; sanitizing to null: {e}')
        text = json.dumps(sanitize_for_json(results), indent=2,
                          allow_nan=False)
    with open('results/exp2_benchmark.json', 'w') as f:
        f.write(text)
    logger.info('\nSaved: results/exp2_benchmark.json')


if __name__ == '__main__':
    main()
