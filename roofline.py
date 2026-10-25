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

Mamba-2 (SSD) scan model
------------------------
:func:`ssd_core_flops` / :func:`ssd_core_flops_breakdown` extend the
analytic accounting to the Mamba-2 scan in BOTH shipped modes: the
sequential recurrence and the chunked-SSD decomposition
(``ops_mamba.chunked_mamba2_scan``). The chunked breakdown names the
four phases — intra-chunk masked quadratic "attention" operator, chunk
source, inter-chunk readout, and the associative-scan passes — so the
dominant term can be identified per geometry. ``latency_profile.py``
uses these functions to compare the model's optimal chunk size against
the measured one.
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
    try:
        flops_value = float(flops)
        nbytes_value = float(nbytes)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'arithmetic_intensity: flops={flops!r} and nbytes={nbytes!r} '
            f'must be real numbers representable as finite floats'
        ) from None
    if not math.isfinite(flops_value) or not math.isfinite(nbytes_value):
        raise ValueError(
            f'arithmetic_intensity: flops={flops!r} and nbytes={nbytes!r} '
            f'must be real numbers representable as finite floats'
        )
    if nbytes_value < 0:
        raise ValueError(
            f'arithmetic_intensity: nbytes={nbytes!r} must be non-negative')
    if nbytes_value > 0:
        value = flops_value / nbytes_value
        if not math.isfinite(value):
            raise ValueError(
                f'arithmetic_intensity: flops={flops!r} and '
                f'nbytes={nbytes!r} produce an unrepresentable intensity')
        return value
    return math.inf if flops_value > 0 else 0.0


def ridge_point(peak_flops_per_s: float, peak_bw_gbps: float) -> float:
    """Machine balance point in FLOPs/byte: ``peak_flops / peak_bw``."""
    if isinstance(peak_bw_gbps, bool) or not isinstance(
            peak_bw_gbps, (int, float)):
        raise ValueError(
            f'peak_bw_gbps={peak_bw_gbps!r} must be a real number')
    try:
        bw_value = float(peak_bw_gbps)
    except OverflowError:
        raise ValueError(
            f'peak_bw_gbps={peak_bw_gbps!r} must be representable as a '
            f'finite float') from None
    if not math.isfinite(bw_value) or bw_value <= 0:
        raise ValueError(
            f'peak_bw_gbps={peak_bw_gbps!r} must be finite and positive')
    if isinstance(peak_flops_per_s, bool) or not isinstance(
            peak_flops_per_s, (int, float)):
        raise ValueError(
            f'peak_flops_per_s={peak_flops_per_s!r} must be a real number')
    try:
        flops_value = float(peak_flops_per_s)
    except OverflowError:
        raise ValueError(
            f'peak_flops_per_s={peak_flops_per_s!r} must be representable '
            f'as a finite float') from None
    if not math.isfinite(flops_value):
        raise ValueError(
            f'peak_flops_per_s={peak_flops_per_s!r} must be finite')
    value = flops_value / (bw_value * 1e9)
    if not math.isfinite(value):
        raise ValueError(
            f'ridge_point: peak_flops_per_s={peak_flops_per_s!r} and '
            f'peak_bw_gbps={peak_bw_gbps!r} produce an unrepresentable '
            f'ridge point')
    return value


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
    try:
        time_value = float(time_ms)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'roofline_point: time_ms={time_ms!r} must be representable as '
            f'a finite float') from None
    if not math.isfinite(time_value):
        raise ValueError(
            f'roofline_point: time_ms={time_ms!r} must be finite')
    if time_value > 0:
        # ``arithmetic_intensity`` already rejected unrepresentable
        # flops/nbytes, so these casts cannot overflow.
        flops_value = float(flops)
        nbytes_value = float(nbytes)
        time_s = time_value * 1e-3
        if time_s <= 0.0:
            raise ValueError(
                f'roofline_point: time_ms={time_ms!r} is too small to '
                f'represent as seconds')
        out['achieved_tflops'] = _finite_result(
            'roofline_point', 'achieved TFLOP/s',
            flops_value / time_s / 1e12)
        out['achieved_bw_gbps'] = _finite_result(
            'roofline_point', 'achieved bandwidth',
            nbytes_value / time_s / 1e9)
    else:
        out['achieved_tflops'] = None
        out['achieved_bw_gbps'] = None
    if peak_flops_per_s and peak_bw_gbps and time_value > 0:
        ridge = ridge_point(peak_flops_per_s, peak_bw_gbps)
        compute_time_s = _finite_result(
            'roofline_point', 'compute time',
            float(flops) / float(peak_flops_per_s))
        memory_time_s = _finite_result(
            'roofline_point', 'memory time',
            float(nbytes) / (float(peak_bw_gbps) * 1e9))
        ceiling_s = _finite_result(
            'roofline_point', 'roofline ceiling',
            max(compute_time_s, memory_time_s))
        out['ridge_point_flops_per_byte'] = ridge
        out['roofline_bound'] = 'compute' if out[
            'arithmetic_intensity_flops_per_byte'] >= ridge else 'memory'
        out['roofline_ceiling_ms'] = _finite_result(
            'roofline_point', 'roofline ceiling ms', ceiling_s * 1e3)
        out['pct_of_roofline'] = _finite_result(
            'roofline_point', 'percent of roofline',
            ceiling_s * 1e3 / time_value * 100.0)
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
    try:
        pct_value = float(pct_of_roofline)
    except (TypeError, ValueError, OverflowError):
        return 'unknown'
    if not math.isfinite(pct_value):
        return 'unknown'
    if pct_value >= 50.0:
        return 'near-roofline'
    if pct_value >= 10.0:
        return 'moderate'
    return 'overhead-bound'


def _validate_core_int(name: str, value, fn_name: str, *,
                       minimum: int = 0) -> int:
    """Exact int geometry for the core FLOPs models, float-representable.

    JSON integers are unbounded; one outside the float range cannot enter
    the ``float`` FLOP arithmetic and used to raise ``OverflowError``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f'{fn_name}: {name}={value!r} must be an int >= {minimum}.')
    try:
        float(value)
    except OverflowError:
        raise ValueError(
            f'{fn_name}: {name}={value!r} must be representable as a finite '
            f'float.') from None
    return value


def _finite_result(fn_name: str, name: str, value) -> float:
    """Return ``value`` as a finite float or reject it as malformed geometry.

    The individual geometry arguments are validated by the callers, but a
    product of representable integers can still exceed the float range
    (e.g. ``T=10**300`` makes ``T*(T+1)/2`` unrepresentable). JSON integers
    are unbounded, so such a request is a malformed configuration rather
    than an ``OverflowError`` from the arithmetic or an ``inf`` result.
    """
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f'{fn_name}: {name}={value!r} must be representable as a finite '
            f'float.') from None
    if not math.isfinite(number):
        raise ValueError(
            f'{fn_name}: {name}={value!r} must be representable as a finite '
            f'float.')
    return number


def softmax_core_flops(T: int, heads: int, head_dim: int,
                       value_dim: int | None = None) -> float:
    """Causal softmax attention FLOPs for the *core only* (pre-projected).

    ``QK^T`` + ``softmax @ V`` over the ``T*(T+1)/2`` causal entries:
    ``2 * entries * heads * (K + V)``.
    """
    _validate_core_int('T', T, 'softmax_core_flops')
    _validate_core_int('heads', heads, 'softmax_core_flops', minimum=1)
    _validate_core_int('head_dim', head_dim, 'softmax_core_flops',
                       minimum=1)
    if value_dim is not None:
        _validate_core_int('value_dim', value_dim, 'softmax_core_flops',
                           minimum=1)
    V = value_dim if value_dim is not None else head_dim
    entries = T * (T + 1) // 2
    try:
        flops = 2.0 * entries * heads * (head_dim + V)
    except OverflowError:
        raise ValueError(
            f'softmax_core_flops: T={T!r} makes the FLOP count '
            f'unrepresentable as a finite float.') from None
    return _finite_result('softmax_core_flops', 'FLOPs', flops)


def kda_core_flops(T: int, heads: int, k_dim: int, v_dim: int, *,
                   chunked: bool = False, chunk_size: int = 64) -> float:
    """KDA recurrence-core FLOPs matching ``run_kv_cache.prefill_flops``.

    Recurrent dominant term: ``6 * T * H * K * V`` (3 MACs/step over the
    ``H*K*V`` state). When ``chunked``, add the paper §6.3 Eq 13 inter-chunk
    terms ``3*T*H*C*K + T*H*C^2`` that ``naive_chunk_kda`` actually
    computes. Projections are excluded: Exp 2 times the core with q/k/v/g/beta
    pre-projected.
    """
    _validate_core_int('T', T, 'kda_core_flops')
    _validate_core_int('heads', heads, 'kda_core_flops', minimum=1)
    _validate_core_int('k_dim', k_dim, 'kda_core_flops', minimum=1)
    _validate_core_int('v_dim', v_dim, 'kda_core_flops', minimum=1)
    if chunked:
        _validate_core_int('chunk_size', chunk_size, 'kda_core_flops',
                           minimum=1)
    flops = _finite_result(
        'kda_core_flops', 'FLOPs', 6.0 * T * heads * k_dim * v_dim)
    if chunked:
        flops += _finite_result(
            'kda_core_flops', 'FLOPs',
            3.0 * T * heads * chunk_size * k_dim)
        flops += _finite_result(
            'kda_core_flops', 'FLOPs', T * heads * chunk_size * chunk_size)
    return _finite_result('kda_core_flops', 'FLOPs', flops)


def ssd_core_flops_breakdown(
    T: int,
    heads: int,
    headdim: int,
    d_state: int,
    *,
    chunk_size: int = 64,
) -> dict:
    """Chunked-SSD scan FLOPs split into its four computational phases.

    Matches ``ops_mamba.chunked_mamba2_scan``'s chunked decomposition at a
    per-batch-element granularity (multiply and add each count as one FLOP,
    i.e. an ``n``-term MAC dot product costs ``2n``):

    * ``intra_chunk_attention`` — the causal masked quadratic operator
      restricted to each chunk. Per ``(t, s)`` pair inside a chunk:
      ``C_t . B_s`` (``2*d_state``) plus the ``weights @ x`` contraction
      (``2*headdim``). Over ``NT`` chunks of ``C`` positions each:
      ``2*Tp*C*heads*(headdim + d_state)`` with ``Tp = NT*C`` the padded
      length.
    * ``chunk_source`` — the per-chunk source ``b_n`` (an outer-product
      accumulation over the chunk, ``2*C*headdim*d_state`` per chunk):
      ``2*Tp*heads*headdim*d_state``.
    * ``inter_chunk_readout`` — ``C_t . S_in`` for every position:
      ``2*Tp*heads*headdim*d_state``.
    * ``assoc_scan`` — the Hillis-Steele prefix scan over chunk
      transitions: ``ceil(log2 NT)`` levels, each one elementwise multiply
      plus one fused multiply-add over ``[BH, NT, headdim, d_state]``:
      ``2*levels*Tp*heads*headdim*d_state / C``.

    Excluded (all ``O(1)``-FLOP-per-element lower-order terms): the
    ``exp``/``cumsum``/mask elementwise work inside a chunk and the
    ``D * x`` skip connection. The model is a *work* estimate for
    crossover/phase analysis — it is NOT a latency prediction (see
    ``latency_profile.py`` for the measured counterpart).

    ``chunk_size >= T`` degenerates to one chunk (``levels = 0``) and the
    intra term becomes the global O(T^2) dual form, matching
    ``ops_mamba.quadratic_mamba2_scan``'s dominant cost.
    """
    _validate_ssd_flops_args(T, heads, headdim, d_state, chunk_size)
    n_chunks = math.ceil(T / chunk_size) if T > 0 else 0
    padded_len = n_chunks * chunk_size
    levels = math.ceil(math.log2(n_chunks)) if n_chunks > 1 else 0
    intra = 2.0 * padded_len * chunk_size * heads * (headdim + d_state)
    source = 2.0 * padded_len * heads * headdim * d_state
    readout = 2.0 * padded_len * heads * headdim * d_state
    scan = (2.0 * levels * padded_len * heads * headdim * d_state
            / chunk_size)
    total = _finite_result(
        'ssd_core_flops_breakdown', 'FLOPs',
        intra + source + readout + scan)
    return {
        'intra_chunk_attention': intra,
        'chunk_source': source,
        'inter_chunk_readout': readout,
        'assoc_scan': scan,
        'total': total,
        'n_chunks': n_chunks,
        'padded_len': padded_len,
        'scan_levels': levels,
    }


def _validate_ssd_flops_args(T, heads, headdim, d_state, chunk_size) -> None:
    """Shared validation for the SSD FLOPs model (fail loudly, never guess)."""
    for name, val in (('T', T), ('heads', heads), ('headdim', headdim),
                      ('d_state', d_state)):
        if isinstance(val, bool) or not isinstance(val, int) or val < 0 \
                or (name != 'T' and val < 1):
            raise ValueError(
                f"ssd_core_flops: {name}={val!r} must be "
                f"{'a non-negative' if name == 'T' else 'a positive'} int.")
        try:
            float(val)
        except OverflowError:
            raise ValueError(
                f"ssd_core_flops: {name}={val!r} must be representable as a "
                f"finite float.") from None
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) \
            or chunk_size < 1:
        raise ValueError(
            f"ssd_core_flops: chunk_size={chunk_size!r} must be a positive "
            f"int.")
    try:
        float(chunk_size)
    except OverflowError:
        raise ValueError(
            f"ssd_core_flops: chunk_size={chunk_size!r} must be "
            f"representable as a finite float.") from None


def ssd_core_flops(
    T: int,
    heads: int,
    headdim: int,
    d_state: int,
    *,
    chunked: bool = False,
    chunk_size: int = 64,
) -> float:
    """Mamba-2 (SSD) scan-core FLOPs, sequential or chunked.

    ``chunked=False`` is the sequential recurrence: per step and head the
    operator does two pointwise products (source outer product ``x (x) B``
    and its ``dt`` scaling), one state decay multiply, one state
    accumulate, and the readout dot ``C_t . S`` — ``3`` MACs over the
    ``headdim x d_state`` state, so ``6*T*heads*headdim*d_state`` FLOPs
    (the same ``6*T*H*K*V`` shape as :func:`kda_core_flops`).

    ``chunked=True`` returns
    :func:`ssd_core_flops_breakdown`'s ``total``.
    """
    if chunked:
        return ssd_core_flops_breakdown(
            T, heads, headdim, d_state, chunk_size=chunk_size)['total']
    _validate_ssd_flops_args(T, heads, headdim, d_state, chunk_size)
    return _finite_result(
        'ssd_core_flops', 'FLOPs', 6.0 * T * heads * headdim * d_state)


def core_bytes(op: str, T: int, *, heads: int, head_dim: int,
               value_dim: int | None = None, d_state: int | None = None,
               ngroups: int = 1, chunked: bool = False,
               chunk_size: int = 64, itemsize: int = FP32_BYTES) -> float:
    """Compulsory-traffic lower bound (bytes) for a *core* timed region.

    ``softmax``: read ``q``, ``k``, ``v``; write ``o``.
    ``kda``: read ``q``, ``k``, ``g``, ``v``, ``beta``; write ``o``; plus the
    recurrent state read+write per step (``2*H*K*V`` elements), or per chunk
    in the chunked path. The state term dominates at the benchmark geometry
    and is exactly why the naive recurrent path is memory-traffic limited.
    ``ssd``: read ``x``, ``dt``, ``B``, ``C``; write ``y``; plus the
    ``H x headdim x d_state`` recurrent state read+write per step
    (sequential) or per chunk (chunked). ``d_state`` is required for
    ``ssd``; ``ngroups`` sizes the shared ``B``/``C`` tensors.

    As with the other ops this is a compulsory lower bound: the chunked
    scan's materialized ``[C, C]`` intra-chunk weight matrices are
    library-internal intermediates and are NOT counted (same policy as the
    softmax operator's score matrix).
    """
    for name, val, minimum in (('T', T, 0), ('heads', heads, 1),
                               ('head_dim', head_dim, 1),
                               ('ngroups', ngroups, 1),
                               ('itemsize', itemsize, 1)):
        if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
            raise ValueError(
                f"core_bytes: {name}={val!r} must be an int >= {minimum}.")
        try:
            float(val)
        except OverflowError:
            raise ValueError(
                f"core_bytes: {name}={val!r} must be representable as a "
                'finite float.') from None
    for name, val in (('value_dim', value_dim), ('d_state', d_state)):
        if val is not None and (isinstance(val, bool)
                                or not isinstance(val, int) or val < 1):
            raise ValueError(
                f"core_bytes: {name}={val!r} must be a positive int or None.")
        if val is not None:
            try:
                float(val)
            except OverflowError:
                raise ValueError(
                    f"core_bytes: {name}={val!r} must be representable as "
                    'a finite float.') from None
    if chunked and (isinstance(chunk_size, bool)
                    or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError(
            f"core_bytes: chunk_size={chunk_size!r} must be a positive int "
            'when chunked=True.')
    if chunked:
        try:
            float(chunk_size)
        except OverflowError:
            raise ValueError(
                f"core_bytes: chunk_size={chunk_size!r} must be "
                'representable as a finite float.') from None
    V = value_dim if value_dim is not None else head_dim
    if op == 'softmax':
        # q + k read at head_dim, v read at value_dim, o written at
        # value_dim (the softmax operator's value/output dimension).
        elements = 2 * T * heads * head_dim + 2 * T * heads * V
        try:
            nbytes = float(elements * itemsize)
        except OverflowError:
            raise ValueError(
                f'core_bytes: T={T!r} makes the byte count unrepresentable '
                f'as a finite float.') from None
        return _finite_result('core_bytes', 'bytes', nbytes)
    if op == 'kda':
        activation = T * heads * (3 * head_dim + 2 * V + 1)
        if chunked:
            n_chunks = math.ceil(T / chunk_size)
            state = 2 * n_chunks * heads * head_dim * V
        else:
            state = 2 * T * heads * head_dim * V
        try:
            nbytes = float((activation + state) * itemsize)
        except OverflowError:
            raise ValueError(
                f'core_bytes: T={T!r} makes the byte count unrepresentable '
                f'as a finite float.') from None
        return _finite_result('core_bytes', 'bytes', nbytes)
    if op == 'ssd':
        if d_state is None:
            raise ValueError("core_bytes: op='ssd' requires d_state.")
        if chunked:
            n_chunks = math.ceil(T / chunk_size)
            state = 2 * n_chunks * heads * head_dim * d_state
        else:
            state = 2 * T * heads * head_dim * d_state
        # x + dt + B + C (read) + y (write).
        activation = (2 * T * heads * head_dim + T * heads
                      + 2 * T * ngroups * d_state)
        try:
            nbytes = float((activation + state) * itemsize)
        except OverflowError:
            raise ValueError(
                f'core_bytes: T={T!r} makes the byte count unrepresentable '
                f'as a finite float.') from None
        return _finite_result('core_bytes', 'bytes', nbytes)
    raise ValueError(
        f"core_bytes: op must be 'softmax', 'kda' or 'ssd', got {op!r}")


def elements_to_bytes(elements: float, *, itemsize: int = FP32_BYTES) -> float:
    """Convert an element count to bytes under the fp32 accounting used here."""
    try:
        elements = float(elements)
        itemsize = float(itemsize)
    except OverflowError:
        raise ValueError(
            'elements_to_bytes: elements/itemsize must be representable as '
            'finite floats.') from None
    return _finite_result(
        'elements_to_bytes', 'bytes', elements * itemsize)
