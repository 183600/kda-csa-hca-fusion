#!/usr/bin/env python3
"""Benchmark the sequential vs associative-scan chunkwise KDA paths.

Round-2 companion script for ``test_assoc_scan.py``: measures wall time of

    * ``naive_chunk_kda``                      (sequential inner loop)
    * ``naive_chunk_kda(recompute_attn=True)`` (low-memory sequential)
    * ``assoc_scan_chunk_kda``                 (Hillis-Steele prefix scan)

on the same inputs, verifies the scan agrees with the sequential path to fp
round-off, and reports the Python-level *sequential depth* of each strategy
(NT loop iterations vs ``ceil(log2 NT)`` scan levels) plus the analytic
auxiliary memory. The CPU timings are a formality; the point of the scan is
the depth reduction (one batched kernel per level), which is what pays off
under CUDA / ``torch.compile``. Run it wherever you need the numbers:

    python3 bench_scan.py --device cpu
    python3 bench_scan.py --device cuda --T 4096 16384 --probe 4

Nothing here is imported by the test suite; the module is standalone.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from ops_kda import assoc_scan_chunk_kda, naive_chunk_kda


def make_case(B, T, H, K, V, chunk_size, device, dtype, seed=0):
    """Unit-norm q/k, small v, mild per-channel log-decay, beta in (0, 0.2)."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    v = torch.randn(B, T, H, V, generator=gen) * 0.1
    g = -torch.rand(B, T, H, K, generator=gen) * 0.05
    beta = torch.rand(B, T, H, generator=gen) * 0.2
    return tuple(x.to(device=device, dtype=dtype) for x in (q, k, v, g, beta))


def timed(fn, warmup=1, repeats=5):
    """Median wall time over ``repeats`` runs after ``warmup``."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.disable()
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
            gc.enable()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    return sorted(times)[len(times) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--T', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument('--B', type=int, default=1)
    ap.add_argument('--H', type=int, default=4, help='q/k heads (HV == H, GVA G=1)')
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--V', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--probe', type=int, default=5, help='timed repetitions (median)')
    ap.add_argument('--dtype', default='fp32', choices=['fp32', 'fp64'])
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == 'fp64' else torch.float32
    device = torch.device(args.device)
    BT, K, V = args.chunk, args.K, args.V
    aux_seq = args.B * args.H * args.K * args.V  # one chunk of state terms
    print(f"device={device} dtype={args.dtype} B={args.B} H={args.H} "
          f"K={K} V={V} BT={BT} probe={args.probe}")
    print(f"{'T':>7} {'NT':>5} {'depth':>5} {'seq ms':>9} {'recomp ms':>10} "
          f"{'scan ms':>9} {'scan/seq':>9} {'max|o-s|':>10} {'Mscan MB':>9}")

    for T in args.T:
        q, k, v, g, beta = make_case(args.B, T, args.H, K, V, BT, device, dtype)
        seq = lambda: naive_chunk_kda(  # noqa: E731
            q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        rec = lambda: naive_chunk_kda(  # noqa: E731
            q, k, v, g, beta, output_final_state=True, chunk_size=BT,
            recompute_attn=True)
        scan = lambda: assoc_scan_chunk_kda(  # noqa: E731
            q, k, v, g, beta, output_final_state=True, chunk_size=BT)
        t_seq = timed(seq, repeats=args.probe)
        t_rec = timed(rec, repeats=args.probe)
        t_scan = timed(scan, repeats=args.probe)
        o1, s1 = seq()
        o2, s2 = scan()
        err = max((o1 - o2).abs().max().item(), (s1 - s2).abs().max().item())
        NT = (T + BT - 1) // BT
        depth = 0 if NT <= 1 else (NT - 1).bit_length()
        # Per-chunk scan operators M [B,HV,NT,K,K]; the doubling scan holds
        # ~2-3 levels at peak.
        m_bytes = args.B * args.H * NT * K * K * torch.tensor([], dtype=dtype).element_size()
        print(f"{T:>7} {NT:>5} {depth:>5} {t_seq*1e3:>9.2f} {t_rec*1e3:>10.2f} "
              f"{t_scan*1e3:>9.2f} {t_scan/t_seq:>9.2f} {err:>10.2e} "
              f"{m_bytes/1e6:>9.1f}")
    print(f"\nsequential loop depth = NT; scan depth = ceil(log2 NT) batched levels.")
    print(f"sequential Aqk cache = B*H*T*BT elements; one chunk of state = {aux_seq} elems.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
