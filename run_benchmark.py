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
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, get_device
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
    """Return the current process RSS in kilobytes.

    Reads ``VmRSS`` from ``/proc/self/status`` on Linux (fast, no deps).
    Falls back to ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` on macOS/BSD
    — note that ru_maxrss is a high-water mark on Linux but the *current* RSS
    on macOS, so the fallback is less accurate on Linux but we only use it
    when /proc is unavailable.
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # "VmRSS:    12345 kB\n"
                    return int(line.split()[1])
    except (FileNotFoundError, ValueError, IndexError, OSError):
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError):
        return 0


def _measure(fn, repeats, device):
    """Return (median_wall_time_s, peak_memory_MB).

    On CUDA: uses CUDA events for timing and ``torch.cuda.max_memory_allocated``
    for real peak memory — the number a production inference engine pays.

    On CPU: uses ``time.perf_counter`` for timing and ``resource.getrusage``
    (RU) RSS for memory. We previously used ``tracemalloc``, but tracemalloc
    only traces the Python heap and silently ignores native torch tensor
    allocations — so it reported ~0 MB for tensors that were actually
    gigabytes, which made the CPU memory column misleading. RSS via
    ``resource`` captures the process-level resident set, which includes
    torch's native allocator. We subtract a baseline RSS taken just before
    the timed region so the reported number is the *delta* attributable to
    the benchmarked operator (not the Python interpreter baseline).
    """
    if device.type == 'cuda':
        # Warmup
        for _ in range(min(2, repeats)):
            fn()
        torch.cuda.synchronize()
        # Reset peak memory stats AFTER warmup so the reported peak reflects
        # only the timed region's activations, NOT the model parameters or
        # warmup allocations. Without this reset, max_memory_allocated()
        # returns the high-water mark since the last _clear_cache() call,
        # which includes the model .to(device) allocation and warmup tensors
        # -- inflating the reported memory by a constant offset that varies
        # per operator (different parameter counts) and makes the comparison
        # less fair.
        torch.cuda.reset_peak_memory_stats(device)
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
        peak_mb = peak_bytes / (1024 ** 2)
        return sorted(times)[len(times) // 2], peak_mb
    else:
        # CPU path: sample current RSS from /proc/self/status (Linux) or
        # resource.getrusage (mac/BSD fallback).
        #
        # We previously used tracemalloc, but tracemalloc only traces the
        # Python heap and silently ignores native torch tensor allocations —
        # so it reported ~0 MB for tensors that were actually megabytes,
        # making the CPU memory column meaningless.
        #
        # We then tried resource.getrusage(RUSAGE_SELF).ru_maxrss, but
        # ru_maxrss is a *high-water mark* — once the process reaches a given
        # RSS, it never decreases, so the delta is 0 for every operator after
        # the first one that pushes RSS to a new high. That made every
        # subsequent row show 0.00 MB, which is just as misleading.
        #
        # The current approach samples the *instantaneous* RSS (VmRSS from
        # /proc/self/status on Linux, or ru_idrss on BSD/macOS) before and
        # after each repeat and keeps the peak sampled value. This is not a
        # true peak (we can only sample between calls, not during), but it
        # captures the steady-state resident footprint attributable to each
        # operator, which is the number a serving engineer cares about.
        gc.collect()
        rss0 = _read_current_rss_kb()  # baseline before the timed region
        times = []
        peak_rss_kb = rss0
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            cur = _read_current_rss_kb()
            if cur > peak_rss_kb:
                peak_rss_kb = cur
        peak_mb = max(0.0, peak_rss_kb - rss0) / 1024.0  # kB -> MB
        return sorted(times)[len(times) // 2], peak_mb


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
    for T in seq_lengths:
        logger.info(f'\n-- T = {T} --')
        for name, factory in benches:
            try:
                _clear_cache(device)
                fn = factory(T)
                # warmup (counted in _measure on GPU; explicit here for CPU)
                if device.type != 'cuda':
                    fn()
                t, mem = _measure(fn, repeats=3, device=device)
                row = {'T': T, 'op': name, 'time_ms': t * 1e3, 'peak_mem_MB': mem,
                       'device': str(device)}
                results.append(row)
                print(f'  {name:12s}  time={t*1e3:8.2f} ms  mem={mem:8.2f} MB')
            except Exception as e:
                results.append({'T': T, 'op': name, 'error': str(e), 'device': str(device)})
                logger.error(f'  {name:12s}  ERROR: {e}')

    os.makedirs('results', exist_ok=True)
    with open('results/exp2_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('\nSaved: results/exp2_benchmark.json')


if __name__ == '__main__':
    main()
