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
import tracemalloc

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


def _measure(fn, repeats, device):
    """Return (median_wall_time_s, peak_memory_MB)."""
    if device.type == 'cuda':
        # Warmup
        for _ in range(min(2, repeats)):
            fn()
        torch.cuda.synchronize()
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
        # CPU path with tracemalloc.
        gc.collect()
        tracemalloc.start()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return sorted(times)[len(times) // 2], peak / (1024 ** 2)


def bench_softmax_attn(B, T, H, K, V, device):
    q = _rand(B, T, H, K, device=device)
    k = _rand(B, T, H, K, device=device)
    v = _rand(B, T, H, V, device=device)
    scale = K ** -0.5

    def fn():
        with torch.no_grad():
            scores = torch.einsum('bthk,bshk->bhts', q, k) * scale
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
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
    pad = (-T) % BT
    if pad:
        qp = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        kp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        vp = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
        gp = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad))
        bp = torch.nn.functional.pad(beta, (0, 0, 0, pad))

        def fn():
            with torch.no_grad():
                o, _ = naive_chunk_kda(qp, kp, vp, gp, bp, output_final_state=True, chunk_size=BT)
                return o[:T]
        return fn

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
    with torch.no_grad():
        def fn():
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
