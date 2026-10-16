"""Roofline / memory-bandwidth analysis helpers (Exp 7).

Why a roofline pass?
--------------------
Exp 2 measures wall-clock latency for every operator but says nothing about
*why* an operator costs what it costs. Exp 3 supplies closed-form FLOPs and
state-size formulas at paper scale but never combines them with a time
measurement. This module supplies the missing piece: it turns
``(analytic FLOPs, compulsory bytes, measured time)`` into the standard
roofline coordinates:

* **arithmetic intensity** ``AI = FLOPs / bytes`` (x-axis),
* **achieved throughput** ``FLOPs / time`` (y-axis),
* the machine's **attainable ceiling** ``min(peak_flops, peak_bw * AI)``,
* the **bound** (memory vs compute) at the machine's ridge point.

Two flavours of machine peak are used, in priority order:

1. **measured** on the running host (:func:`measure_gemm_tflops`,
   :func:`measure_copy_bandwidth_gbps`) — no spec sheet needed, works on
   CPU CI and unknown GPUs;
2. **preset** from :data:`DEVICE_PEAK_SPECS` for well-known GPUs when the
   measurement is unavailable.

Peak numbers are fp32 CUDA-core / non-tensor-core values to match the
repository's fp32 reference implementation; TF32/BF16 tensor-core peaks are
much higher and are intentionally NOT used. Every preset value is a
documented nominal figure (H100 SXM ~67 TFLOP/s fp32, ~3.35 TB/s HBM3,
etc.), not a benchmark claim.

Byte model honesty
------------------
:func:`core_bytes` and the composition in ``run_roofline.py`` count a
**compulsory-traffic lower bound**: activation I/O + weights + recurrent /
compressed state read-write. Library-internal traffic (im2col/unfold
intermediates, allocator churn, PyTorch interpreter overhead) is NOT
included, so an operator's "achieved bandwidth" computed against this model
is a lower bound that can legitimately exceed the machine's measured copy
bandwidth when the real implementation moves more bytes than compulsory.
``run_roofline.py`` records that caveat in the output metadata.
"""
from __future__ import annotations

import math
import statistics
import time

FP32_BYTES = 4

# Nominal fp32 (non-tensor-core) peaks for well-known accelerators.
# Values are spec-sheet figures used only as a fallback when on-host
# measurement is unavailable; keys are lowercase substrings matched against
# the device name.
DEVICE_PEAK_SPECS = {
    'tesla t4': {'peak_tflops': 8.1, 'peak_bw_gbps': 320.0},
    'v100': {'peak_tflops': 15.7, 'peak_bw_gbps': 900.0},
    'a100': {'peak_tflops': 19.5, 'peak_bw_gbps': 2039.0},
    'l40s': {'peak_tflops': 91.6, 'peak_bw_gbps': 864.0},
    'rtx 3090': {'peak_tflops': 35.6, 'peak_bw_gbps': 936.0},
    'rtx 4090': {'peak_tflops': 82.6, 'peak_bw_gbps': 1008.0},
    'h100': {'peak_tflops': 67.0, 'peak_bw_gbps': 3350.0},
}


def spec_for_device(device_name: str) -> dict | None:
    """Return the matching :data:`DEVICE_PEAK_SPECS` entry, or ``None``.

    Matching is a case-insensitive substring test so ``"NVIDIA A100-SXM4"``
    and ``"a100"`` both find the A100 preset. When several keys match, the
    longest key wins (``'rtx 4090'`` beats a hypothetical ``'rtx'``).
    """
    if not device_name:
        return None
    name = device_name.lower()
    matches = [k for k in DEVICE_PEAK_SPECS if k in name]
    if not matches:
        return None
    key = max(matches, key=len)
    spec = dict(DEVICE_PEAK_SPECS[key])
    spec['preset_key'] = key
    return spec


def measure_copy_bandwidth_gbps(device: str = 'cpu', *, size_mb: int = 64,
                                repeats: int = 3) -> float | None:
    """Measure streaming copy bandwidth (GB/s) with a ``dst.copy_(src)``.

    Traffic per iteration = ``2 * n * itemsize`` bytes (one read + one
    write), counted in decimal GB. The buffer is sized to exceed last-level
    cache at the default 64 MiB so the number reflects DRAM streaming
    rather than cache reuse. Returns ``None`` (never raises) when torch is
    unavailable or the device fails.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        n = max(1, size_mb * 1024 * 1024 // FP32_BYTES)
        src = torch.empty(n, dtype=torch.float32, device=device)
        dst = torch.empty(n, dtype=torch.float32, device=device)
        is_cuda = str(device).startswith('cuda')
        times: list[float] = []
        with torch.no_grad():
            for _ in range(max(1, repeats)):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                dst.copy_(src)
                if is_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        if not times or min(times) <= 0:
            return None
        return (2.0 * n * FP32_BYTES) / statistics.median(times) / 1e9
    except Exception:
        return None


def measure_gemm_tflops(device: str = 'cpu', *, n: int = 1024,
                        repeats: int = 3) -> float | None:
    """Measure dense fp32 GEMM throughput (TFLOP/s) for an ``n x n @ n x n``.

    FLOPs = ``2 * n**3``. ``n=1024`` keeps the measurement under ~50 ms on a
    small CPU while large enough that launch overhead is negligible. This is
    an *achieved* GEMM rate (typically within ~10% of the CPU's practical
    fp32 ceiling), used as the compute-roof anchor. Returns ``None`` on any
    failure.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        a = torch.randn(n, n, dtype=torch.float32, device=device)
        b = torch.randn(n, n, dtype=torch.float32, device=device)
        is_cuda = str(device).startswith('cuda')
        times: list[float] = []
        with torch.no_grad():
            for _ in range(max(1, repeats)):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                a @ b
                if is_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        if not times or min(times) <= 0:
            return None
        return (2.0 * n ** 3) / statistics.median(times) / 1e12
    except Exception:
        return None


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    """``FLOPs / byte``; ``inf`` when bytes are zero but work is not."""
    if nbytes > 0:
        return flops / nbytes
    return math.inf if flops > 0 else 0.0


def ridge_point(peak_flops_per_s: float, peak_bw_gbps: float) -> float:
    """Machine balance point in FLOPs/byte: ``peak_flops / peak_bw``."""
    if peak_bw_gbps <= 0:
        raise ValueError('peak_bw_gbps must be positive')
    return peak_flops_per_s / (peak_bw_gbps * 1e9)


def bound_for(ai: float, peak_flops_per_s: float,
              peak_bw_gbps: float) -> str:
    """Classify a workload as ``'compute'`` or ``'memory'`` bound."""
    return 'compute' if ai >= ridge_point(peak_flops_per_s,
                                          peak_bw_gbps) else 'memory'


def roofline_point(*, flops: float, nbytes: float, time_ms: float,
                   peak_flops_per_s: float | None = None,
                   peak_bw_gbps: float | None = None) -> dict:
    """Combine analytic work, bytes and measured time into roofline fields.

    ``time_ms`` is the measured wall-clock time of the timed region the
    FLOPs / bytes accounting was derived for. Machine ceilings are optional:
    without them the achieved-coordinate fields are still returned.
    """
    out: dict = {
        'flops': flops,
        'bytes': nbytes,
        'observed_time_ms': time_ms,
        'arithmetic_intensity_flops_per_byte': arithmetic_intensity(flops,
                                                                    nbytes),
    }
    if time_ms and time_ms > 0:
        out['achieved_tflops'] = flops / (time_ms * 1e-3) / 1e12
        out['achieved_bw_gbps'] = nbytes / (time_ms * 1e-3) / 1e9
    else:
        out['achieved_tflops'] = None
        out['achieved_bw_gbps'] = None
    if peak_flops_per_s and peak_bw_gbps and time_ms and time_ms > 0:
        ridge = ridge_point(peak_flops_per_s, peak_bw_gbps)
        compute_time_s = flops / peak_flops_per_s
        memory_time_s = nbytes / (peak_bw_gbps * 1e9)
        ceiling_s = max(compute_time_s, memory_time_s)
        out['ridge_point_flops_per_byte'] = ridge
        out['roofline_bound'] = 'compute' if out[
            'arithmetic_intensity_flops_per_byte'] >= ridge else 'memory'
        out['roofline_ceiling_ms'] = ceiling_s * 1e3
        out['pct_of_roofline'] = ceiling_s * 1e3 / time_ms * 100.0
        out['efficiency_label'] = efficiency_label(
            out['pct_of_roofline'])
    return out


def efficiency_label(pct_of_roofline: float | None) -> str:
    """Coarse label for how close a measured point is to its ceiling.

    Thresholds are engineering rules of thumb, not theory; they exist so the
    markdown/JSON report can say ``overhead-bound`` instead of making the
    reader compare percentages manually.
    """
    if pct_of_roofline is None:
        return 'unknown'
    if pct_of_roofline >= 50.0:
        return 'near-roofline'
    if pct_of_roofline >= 10.0:
        return 'moderate'
    return 'overhead-bound'


def softmax_core_flops(T: int, heads: int, head_dim: int,
                       value_dim: int | None = None) -> float:
    """Causal softmax attention FLOPs for the *core only* (pre-projected).

    ``QK^T`` + ``softmax @ V`` over the ``T*(T+1)/2`` causal entries:
    ``2 * entries * heads * (K + V)``.
    """
    V = value_dim if value_dim is not None else head_dim
    entries = T * (T + 1) // 2
    return 2.0 * entries * heads * (head_dim + V)


def kda_core_flops(T: int, heads: int, k_dim: int, v_dim: int, *,
                   chunked: bool = False, chunk_size: int = 64) -> float:
    """KDA recurrence-core FLOPs matching ``run_kv_cache.prefill_flops``.

    Recurrent dominant term: ``6 * T * H * K * V`` (3 MACs/step over the
    ``H*K*V`` state). When ``chunked``, add the paper §6.3 Eq 13 inter-chunk
    terms ``3*T*H*C*K + T*H*C^2`` that ``naive_chunk_kda`` actually
    computes. Projections are excluded: Exp 2 times the core with q/k/v/g/beta
    pre-projected.
    """
    flops = 6.0 * T * heads * k_dim * v_dim
    if chunked:
        flops += 3.0 * T * heads * chunk_size * k_dim
        flops += T * heads * chunk_size * chunk_size
    return flops


def core_bytes(op: str, T: int, *, heads: int, head_dim: int,
               value_dim: int | None = None, chunked: bool = False,
               chunk_size: int = 64, itemsize: int = FP32_BYTES) -> float:
    """Compulsory-traffic lower bound (bytes) for a *core* timed region.

    ``softmax``: read ``q``, ``k``, ``v``; write ``o``.
    ``kda``: read ``q``, ``k``, ``g``, ``v``, ``beta``; write ``o``; plus the
    recurrent state read+write per step (``2*H*K*V`` elements), or per chunk
    in the chunked path. The state term dominates at the benchmark geometry
    and is exactly why the naive recurrent path is memory-traffic limited.
    """
    V = value_dim if value_dim is not None else head_dim
    if op == 'softmax':
        elements = 3 * T * heads * head_dim + T * heads * V
        return float(elements * itemsize)
    if op == 'kda':
        activation = T * heads * (3 * head_dim + 2 * V + 1)
        if chunked:
            n_chunks = math.ceil(T / chunk_size)
            state = 2 * n_chunks * heads * head_dim * V
        else:
            state = 2 * T * heads * head_dim * V
        return float((activation + state) * itemsize)
    raise ValueError(f"core_bytes: op must be 'softmax' or 'kda', got {op!r}")


def elements_to_bytes(elements: float, *, itemsize: int = FP32_BYTES) -> float:
    """Convert an element count to bytes under the fp32 accounting used here."""
    return float(elements) * itemsize
