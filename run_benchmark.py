"""Experiment 2 — latency and memory benchmark (device-aware).

Measures wall-clock latency and peak memory of each attention operator
(softmax attention baseline, KDA, CSA, HCA, and the fused hybrid) across a
range of sequence lengths.

Methodology:

  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling
    back to CPU. On GPU we use CUDA events for accurate timing and
    ``torch.cuda.max_memory_allocated`` for real peak memory.
  * **Real memory measurement.** On GPU we report the actual
    ``torch.cuda.max_memory_allocated`` (the number a production inference
    engine would pay), not just the Python-traced allocation. On CPU we keep
    ``tracemalloc`` for the resident Python heap.
  * **All advertised sequence lengths.** T in {128, 256, 512, 1024, 2048}.
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
    append_run_manifest, configure_torch_for_device, parse_int_env,
    sanitize_for_json, write_json_atomic, capture_provenance,
    make_run_manifest_entry, make_seeded_generator,
)
from ops_kda_backend import kda_forward, validate_kda_backend
from ops_csa import naive_csa
from ops_hca import naive_hca
from ops_fused import HybridAttentionStack, HybridKCHAttention, HybridConfig
from ops_mamba import (Mamba2Mixer, build_mamba2_mixer, chunked_mamba2_scan,
                       mamba2_scan_boundary_metadata, naive_mamba2_scan)
from ops_mamba_backend import validate_mamba_backend
from timing_stats import summarize_timings_s
from results_schema import RowContractError, validate_rows
import ops_kda

logger = logging.getLogger(__name__)


def _selected_kda_backend() -> str:
    """Return the explicitly requested benchmark KDA backend.

    The benchmark remains reference-first; set ``KDA_BACKEND=fla`` or
    ``KDA_BACKEND=auto`` to measure the optional FLA path separately.
    """
    return validate_kda_backend(os.environ.get('KDA_BACKEND', 'reference'))


def _selected_mamba_backend() -> str:
    """Return the explicitly requested benchmark Mamba-2 backend.

    Mirrors the KDA contract: reference-first by default;
    ``MAMBA_BACKEND=official`` measures the mamba_ssm kernels (errors
    loudly when the optional dependency or CUDA is missing — an explicit
    accelerated-backend request never silently falls back);
    ``MAMBA_BACKEND=auto`` uses them when available.
    """
    return validate_mamba_backend(os.environ.get('MAMBA_BACKEND', 'reference'))


_LAST_TIMING_STATS: dict = {}


def _record_timing_stats(times):
    """Store the uncertainty summary for the row just measured."""
    stats = summarize_timings_s(times)
    _LAST_TIMING_STATS.clear()
    _LAST_TIMING_STATS.update(stats)
    _LAST_TIMING_STATS['times'] = list(times)
    return stats


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
        _record_timing_stats(times)
        # Guard against repeats=0: ``statistics.median([])`` raises
        # ``StatisticsError``. Returns 0.0 so the JSON row is well-formed.
        return (statistics.median(times) if times else 0.0), peak_mb
    else:
        # CPU path: VmRSS sampling from /proc/self/status between ``fn()``
        # calls is not usable, because torch's native CPU allocator (like
        # malloc) retains freed blocks in its pool for reuse rather than
        # returning them to the OS. Once the first operator pushes RSS to a
        # new high, every subsequent operator reuses the pool and the RSS
        # delta reads 0 — which looks like a real measurement but only means
        # "no new RSS growth detected".
        #
        # tracemalloc is even worse: it only traces the Python heap and
        # silently ignores native torch tensor allocations, so it reports
        # ~0 MB for tensors that are actually megabytes.
        #
        # The honest choice is to report ``None`` on CPU (matching
        # ``run_decoding.py``), so JSON serializes to ``null`` (clearly "no
        # data") rather than a misleading 0.0. On GPU we report the real
        # ``torch.cuda.max_memory_allocated`` (with baseline subtraction).
        _prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            # Warmup runs under the same single-thread configuration as the
            # timed loop, so the first timed iteration does not absorb the
            # one-time cost of reconfiguring the internal thread pool /
            # kernel dispatch, which would bias the measured latency.
            for _ in range(min(2, repeats)):
                out = fn()
                del out
            gc.collect()
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
        _record_timing_stats(times)
        return (statistics.median(times) if times else 0.0), peak_mb


def bench_softmax_attn(B, T, H, K, V, device):
    gen = _make_op_gen('softmax', T, device)
    q = _rand(B, T, H, K, device=device, generator=gen)
    k = _rand(B, T, H, K, device=device, generator=gen)
    v = _rand(B, T, H, V, device=device, generator=gen)
    scale = K ** -0.5
    # Precompute the causal mask OUTSIDE the timed region so the benchmark
    # measures attention compute rather than mask construction. The mask is
    # the same every call, and the timed region for the cross-operator
    # comparison then covers comparable work: KDA/CSA/HCA bench wrappers
    # have no per-call [T, T] mask to build.
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
            if o.shape[1] != T:
                # A plain ``assert`` here would be stripped under
                # ``python -O``, silently accepting a corrupted benchmark
                # row when the chunk path stops trimming its padding.
                raise RuntimeError(
                    f"naive_chunk_kda output shape {tuple(o.shape)} does "
                    f"not match input T={T} on dim=1 (sequence). The chunk "
                    f"path is supposed to right-pad T up to a multiple of "
                    f"chunk_size={BT} and trim back to original_T "
                    f"internally; a shape mismatch indicates a regression "
                    f"in the trim logic.")
            return o, s
    return fn


def bench_kda_chunk_compiled(B, T, H, K, V, device, meta=None):
    """``torch.compile`` chunked-KDA core row at the same boundary.

    Draws from the SAME ``kda_chunk`` generator key as ``bench_kda_chunk``
    (identical q/k/v/g/beta), gates parity against the eager chunk path
    BEFORE timing, and records the compile policy plus the per-channel
    decay parity residual. ``recompute_attn`` follows the
    ``KDA_RECOMPUTE_ATTN`` env knob (default 0 = cached ``Aqk``) so the
    memory-bounded policy can be measured without changing the eager
    reference row.
    """
    gen = _make_op_gen('kda_chunk', T, device)
    q = torch.nn.functional.normalize(
        _rand(B, T, H, K, device=device, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(
        _rand(B, T, H, K, device=device, generator=gen), dim=-1)
    v = _rand(B, T, H, V, device=device, generator=gen)
    g = -torch.rand(B, T, H, K, device=device, generator=gen) * 0.05
    beta = torch.rand(B, T, H, device=device, generator=gen) * 0.2
    BT = 64
    recompute = os.environ.get(
        'KDA_RECOMPUTE_ATTN', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
    with torch.no_grad():
        o_ref, s_ref = ops_kda.naive_chunk_kda(
            q, k, v, g, beta,
            output_final_state=True, chunk_size=BT,
            recompute_attn=recompute)
        o_cmp, s_cmp = ops_kda.compiled_chunk_kda(
            q, k, v, g, beta,
            output_final_state=True, chunk_size=BT,
            recompute_attn=recompute, fullgraph=True)
    finite = bool(torch.isfinite(o_cmp).all())
    if s_cmp is not None:
        finite = finite and bool(torch.isfinite(s_cmp).all())
    parity = (float((o_cmp - o_ref).abs().max().item())
              if o_cmp.numel() and o_ref.numel() else 0.0)
    state_parity = 0.0
    if s_cmp is not None and s_ref is not None and s_cmp.numel():
        state_parity = float((s_cmp - s_ref).abs().max().item())
    if meta is not None:
        meta['param_elements'] = 0
        meta['kda_chunk_size'] = BT
        meta['kda_recompute_attn'] = recompute
        meta['kda_execution'] = 'torch.compile'
        meta['kda_compile_fullgraph'] = True
        meta['kda_compile_dynamic'] = False
        meta['kda_chunk_parity_max_abs'] = parity
        meta['kda_chunk_state_parity_max_abs'] = state_parity
        meta['kda_chunk_parity_ok'] = bool(
            finite and parity <= 1e-4 and state_parity <= 1e-4)
    if not finite or parity > 1e-4 or state_parity > 1e-4:
        raise RuntimeError(
            f'compiled chunked-KDA parity gate failed at T={T}: '
            f'finite={finite} out={parity:.3e} state={state_parity:.3e}')

    def fn():
        with torch.no_grad():
            return ops_kda.compiled_chunk_kda(
                q, k, v, g, beta,
                output_final_state=True, chunk_size=BT,
                recompute_attn=recompute, fullgraph=True)
    return fn


def bench_csa(B, T, d, device, meta=None):
    m, topk = 8, 4
    nh, c, dc, nIh, cI = 2, 16, 32, 2, 8
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
        W_DQ=_rand(dc, d, device=device, generator=gen), W_UQ=_rand(c * nh, dc, device=device, generator=gen),
        W_IUQ=_rand(cI * nIh, dc, device=device, generator=gen), W_w=_rand(nIh, d, device=device, generator=gen),
        W_KV_idx=_rand(cI, d, device=device, generator=gen), W_Z_idx=_rand(cI, d, device=device, generator=gen),
        B_idx=_rand(m, cI, device=device, generator=gen),
    )
    W_O = _rand(d, nh * c, device=device, generator=gen)
    if meta is not None:
        # Exact element count of every weight tensor in the timed region.
        # Consumed by run_roofline for the weights-traffic term; measured
        # here instead of re-derived from formulas so it can never drift
        # from the actual benchmark weights.
        meta['param_elements'] = sum(w.numel() for w in weights.values()) \
            + W_O.numel()

    def fn():
        with torch.no_grad():
            out = naive_csa(H, **weights, **cfg)
            return torch.nn.functional.linear(out, W_O)
    return fn


def bench_hca(B, T, d, device, meta=None):
    m2, nh, c, dc = 16, 2, 16, 32
    gen = _make_op_gen('hca', T, device)
    H = _rand(B, T, d, device=device, generator=gen)
    cfg = dict(
        m2=m2, nh=nh, c=c, dc=dc,
        sliding_window=8, sink_logits=torch.zeros(nh, device=device),
    )
    weights = dict(
        W_KV=_rand(c, d, device=device, generator=gen), W_Z=_rand(c, d, device=device, generator=gen),
        B_pos=_rand(m2, c, device=device, generator=gen),
        W_DQ=_rand(dc, d, device=device, generator=gen), W_UQ=_rand(c * nh, dc, device=device, generator=gen),
    )
    W_O = _rand(d, nh * c, device=device, generator=gen)
    if meta is not None:
        meta['param_elements'] = sum(w.numel() for w in weights.values()) \
            + W_O.numel()

    def fn():
        with torch.no_grad():
            out = naive_hca(H, **weights, **cfg)
            return torch.nn.functional.linear(out, W_O)
    return fn


def _mamba2_scan_inputs(B, T, device):
    """Shared seeded inputs for the sequential/chunked SSD core rows.

    Both scan rows draw from the SAME ``mamba2_scan`` generator key, so the
    chunked row is a parity-checked, apples-to-apples re-execution of the
    sequential row's exact inputs (not a different random draw). Geometry
    is capacity-matched to the KDA core bench (H=4, K=V=32 -> 4 heads x
    32x32 state): 4 heads x (headdim=32 x d_state=32) = the same per-model
    recurrent-state size, so the scan-cost comparison is not confounded by
    state size.
    """
    H, headdim, d_state = 4, 32, 32
    gen = _make_op_gen('mamba2_scan', T, device)
    x = _rand(B, T, H, headdim, device=device, generator=gen)
    # dt: post-softplus-like positive values in the official init range.
    dt = torch.rand(B, T, H, device=device, generator=gen) * 0.1 + 1e-3
    Bm = _rand(B, T, 1, d_state, device=device, generator=gen)
    Cm = _rand(B, T, 1, d_state, device=device, generator=gen)
    A = -torch.exp(torch.log(torch.arange(1, H + 1, device=device,
                                           dtype=torch.float32)))
    D = torch.ones(H, device=device)
    return x, dt, Bm, Cm, A, D


def bench_mamba2_scan(B, T, d, device, meta=None):
    """Mamba-2 SSD scan at the CORE boundary (mirrors bench_kda_recurrent).

    Inputs (x, dt, B, C) are pre-projected OUTSIDE the timed region, so
    the row measures only the recurrence — the same boundary as the
    softmax / kda_rec / kda_chunk rows.
    """
    x, dt, Bm, Cm, A, D = _mamba2_scan_inputs(B, T, device)
    if meta is not None:
        meta['param_elements'] = 0  # the core boundary carries no weights

    def fn():
        with torch.no_grad():
            return naive_mamba2_scan(
                x, dt, Bm, Cm, A, D,
                output_final_state=True)
    return fn


def bench_mamba2_chunked_scan(B, T, d, device, meta=None, chunk_size=64):
    """Chunked-SSD scan at the SAME core boundary as ``bench_mamba2_scan``.

    Same inputs and geometry as the sequential row, so the latency ratio is
    a clean kernel/organization comparison. Correctness is gated BEFORE any
    timing: the chunked output must be finite and match the sequential
    reference on the same inputs (max-abs <= 1e-4 fp32); a divergent
    operator raises instead of producing a fast-but-wrong row.

    ``meta`` records the operator-boundary decomposition
    (``mamba2_scan_boundary_metadata``: n_chunks / padded_len / scan_levels
    / chunk starts) plus the parity residual, so the row is self-describing
    and ``run_roofline`` can price the exact chunked decomposition.
    """
    x, dt, Bm, Cm, A, D = _mamba2_scan_inputs(B, T, device)
    args = (x, dt, Bm, Cm, A, D)
    with torch.no_grad():
        y_ref, _ = naive_mamba2_scan(*args, output_final_state=True)
        y_ch, s_ch = chunked_mamba2_scan(
            *args, output_final_state=True, chunk_size=chunk_size)
    finite = bool(torch.isfinite(y_ch).all())
    if s_ch is not None:
        finite = finite and bool(torch.isfinite(s_ch).all())
    parity = (float((y_ch - y_ref).abs().max().item())
              if y_ch.numel() and y_ref.numel() else 0.0)
    boundary = mamba2_scan_boundary_metadata(T, chunk_size=chunk_size)
    if meta is not None:
        meta['param_elements'] = 0
        meta['mamba_chunk_size'] = chunk_size
        meta['scan_boundary'] = boundary
        meta['scan_parity_max_abs'] = parity
        meta['scan_parity_ok'] = bool(finite and parity <= 1e-4)
    if not finite or parity > 1e-4:
        raise RuntimeError(
            f'chunked SSD parity gate failed at T={T} C={chunk_size}: '
            f'finite={finite} max_abs={parity:.3e}')

    def fn():
        with torch.no_grad():
            return chunked_mamba2_scan(
                *args, output_final_state=True, chunk_size=chunk_size)
    return fn


def bench_mamba2_layer(B, T, d, device, meta=None):
    """Full Mamba-2 mixer at the end_to_end_single_layer boundary.

    Mirrors bench_csa / bench_hca: hidden state in, projected output out
    (in_proj -> causal conv -> SiLU -> x_proj -> scan -> gate ->
    out_proj). The backend follows MAMBA_BACKEND (reference default).
    """
    gen = _make_op_gen('mamba2_layer', T, device)
    cfg = HybridConfig(
        d_model=d, memory_type='mamba2',
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=32,
        mamba_backend=_selected_mamba_backend(),
    )
    seed = zlib.crc32(b'mamba2_layer_weights') & 0xFFFFFFFF
    _prev_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        mixer = build_mamba2_mixer(cfg).to(device).eval()
    finally:
        torch.random.set_rng_state(_prev_state)
    if meta is not None:
        meta['param_elements'] = sum(p.numel() for p in mixer.parameters())
        meta['mamba_effective_backend'] = getattr(
            mixer, 'effective_backend', 'reference')
    x = _rand(B, T, d, device=device, generator=gen)

    def fn():
        with torch.no_grad():
            out, _, _ = mixer.forward_functional(x, None, None)
            return out
    return fn


def bench_hybrid(B, T, d, device, meta=None):
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
    seed = zlib.crc32(b'hybrid_weights') & 0xFFFFFFFF
    _prev_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    finally:
        torch.random.set_rng_state(_prev_state)
    if meta is not None:
        meta['param_elements'] = sum(p.numel() for p in model.parameters())
    x = _rand(B, T, d, device=device, generator=gen)
    def fn():
        with torch.no_grad():
            model.reset_state()
            return model(x)
    return fn


def bench_hybrid_mamba2(B, T, d, device, meta=None):
    """5-layer Mamba-2-route hybrid stack at the SAME boundary as the
    KDA hybrid row (end_to_end_multi_layer: LayerNorm + mixer + state
    management, 3:1:1 memory:CSA:HCA ratio).

    Same CSA/HCA sub-layer geometry as ``bench_hybrid`` so the row pair
    isolates the memory-operator swap. The mamba geometry mirrors
    ``bench_mamba2_layer`` (d_state=16, headdim=32 -> d_inner/32 heads).
    """
    gen = _make_op_gen('hybrid_mamba2', T, device)
    cfg = HybridConfig(
        d_model=d, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32, csa_nIh=2, csa_cI=8,
        csa_sliding_window=8,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=32, hca_sliding_window=8,
        n_kda=3, n_csa=1, n_hca=1,
        memory_type='mamba2',
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=32,
        kda_backend=_selected_kda_backend(),
        mamba_backend=_selected_mamba_backend(),
    )
    seed = zlib.crc32(b'hybrid_mamba2_weights') & 0xFFFFFFFF
    _prev_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        model = HybridAttentionStack(cfg, total_layers=5).to(device).eval()
    finally:
        torch.random.set_rng_state(_prev_state)
    if meta is not None:
        meta['param_elements'] = sum(p.numel() for p in model.parameters())
        meta['mamba_effective_backend'] = getattr(
            model.memory_layers[0], 'effective_backend', 'reference')
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
    # Standalone KDA is benchmarked at TWO configs:
    #   * H/K/V matched to the softmax baseline (H=4, K=32, V=32): the
    #     matched-capacity core comparison (kda_rec / kda_chunk).
    #   * H2/K2/V2 matched to the hybrid stack's KDA sub-layers
    #     (n_heads_qk=2, head_dim_k=16, head_dim_v=16): a standalone KDA
    #     row that is directly comparable to the hybrid's per-KDA-layer
    #     cost (kda_rec_hybrid / kda_chunk_hybrid).
    # The hybrid-matched rows keep the latency comparison fair: a single
    # standalone KDA layer runs at 8x the compute of the hybrid's KDA
    # sub-layers (4*32*32 = 4096 vs 2*16*16 = 512 MACs/step), so the two
    # KDA configs are reported separately rather than in one row.
    H2, K2, V2 = 2, 16, 16

    benches = [
        ('softmax',  lambda T, meta=None: bench_softmax_attn(B, T, H, K, V, device)),
        ('kda_rec',  lambda T, meta=None: bench_kda_recurrent(B, T, H, K, V, device)),
        ('kda_chunk', lambda T, meta=None: bench_kda_chunk(B, T, H, K, V, device)),
        ('kda_chunk_compiled', lambda T, meta=None: bench_kda_chunk_compiled(
            B, T, H, K, V, device, meta=meta)),
        ('kda_rec_hybrid', lambda T, meta=None: bench_kda_recurrent(B, T, H2, K2, V2, device)),
        ('kda_chunk_hybrid', lambda T, meta=None: bench_kda_chunk(B, T, H2, K2, V2, device)),
        ('mamba2_scan', lambda T, meta=None: bench_mamba2_scan(B, T, d, device, meta=meta)),
        ('mamba2_chunked_scan', lambda T, meta=None: bench_mamba2_chunked_scan(
            B, T, d, device, meta=meta)),
        ('mamba2_layer', lambda T, meta=None: bench_mamba2_layer(B, T, d, device, meta=meta)),
        ('csa',      lambda T, meta=None: bench_csa(B, T, d, device, meta=meta)),
        ('hca',      lambda T, meta=None: bench_hca(B, T, d, device, meta=meta)),
        ('hybrid',   lambda T, meta=None: bench_hybrid(B, T, d, device, meta=meta)),
        ('hybrid_mamba2', lambda T, meta=None: bench_hybrid_mamba2(B, T, d, device, meta=meta)),
    ]
    op_boundary = {
        'softmax':   {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'attention core only; q/k/v pre-projected'},
        'kda_rec':   {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'recurrence core only; q/k/v/g/beta pre-projected (H=4, K=V=32, matched to softmax capacity)'},
        'kda_chunk': {'compute_boundary': 'core',
                      'n_layers': 1,
                      'note': 'chunked recurrence core only; q/k/v/g/beta pre-projected (H=4, K=V=32, matched to softmax capacity)'},
        'kda_chunk_compiled': {'compute_boundary': 'core',
                               'n_layers': 1,
                               'note': ('torch.compile chunked recurrence core; '
                                        'identical inputs to kda_chunk, '
                                        'output/state parity-gated against '
                                        'the eager chunk path before timing')},
        'kda_rec_hybrid': {'compute_boundary': 'core',
                           'n_layers': 1,
                           'note': 'recurrence core only; q/k/v/g/beta pre-projected (H=2, K=V=16, matched to hybrid KDA sub-layer config)'},
        'kda_chunk_hybrid': {'compute_boundary': 'core',
                             'n_layers': 1,
                             'note': 'chunked recurrence core only; q/k/v/g/beta pre-projected (H=2, K=V=16, matched to hybrid KDA sub-layer config)'},
        'csa':       {'compute_boundary': 'end_to_end_single_layer',
                      'n_layers': 1,
                      'note': 'single CSA layer from hidden state H (includes all projections + compression + indexer + sparse attention + o_proj)'},
        'hca':       {'compute_boundary': 'end_to_end_single_layer',
                      'n_layers': 1,
                      'note': 'single HCA layer from hidden state H (includes all projections + compression + dense attention + o_proj)'},
        'hybrid':    {'compute_boundary': 'end_to_end_multi_layer',
                      'n_layers': 5,
                      'note': '5-layer KDA+CSA+HCA stack with LayerNorm, projections, attention, state management (3:1:1 default ratio)'},
        'mamba2_scan': {'compute_boundary': 'core',
                        'n_layers': 1,
                        'note': ('SSD recurrence core only; x/dt/B/C '
                                 'pre-projected (4 heads x 32x32 state, '
                                 'capacity-matched to kda_rec); sequential '
                                 'reference scan')},
        'mamba2_chunked_scan': {'compute_boundary': 'core',
                                'n_layers': 1,
                                'note': ('chunked SSD scan core only; same '
                                         'inputs/geometry as mamba2_scan '
                                         '(C=64); quadratic intra-chunk + '
                                         'Hillis-Steele chunk transition '
                                         'scan; output parity-gated against '
                                         'the sequential reference before '
                                         'timing')},
        'mamba2_layer': {'compute_boundary': 'end_to_end_single_layer',
                         'n_layers': 1,
                         'note': 'single Mamba-2 mixer from hidden state (in_proj + causal conv + x_proj + scan + gate + out_proj)'},
        'hybrid_mamba2': {'compute_boundary': 'end_to_end_multi_layer',
                          'n_layers': 5,
                          'note': '5-layer Mamba-2+CSA+HCA stack at the same boundary as the hybrid row (3:1:1 memory:CSA:HCA)'},
    }
    # Workload-defining hyperparameters per operator. These are the values that
    # determine how many keys each method actually attends to at a given T
    # (e.g. CSA attends only to ``topk`` compressed blocks + a fixed
    # ``sliding_window``, so its cost is ~flat in T; HCA attends to all
    # compressed blocks, growing linearly in T; softmax attends to all T
    # keys). Serializing them next to the timing rows lets downstream
    # analysis reconstruct the measured workload instead of having to guess
    # it from source.
    op_workload = {
        'softmax':   {'heads': H, 'head_dim': K, 'attends_to_all_T': True,
                      'causal_mask': True},
        'kda_rec':   {'heads': H, 'head_dim': K,
                      'note': 'recurrent delta-rule; O(1) state, attends to all T via recurrence'},
        'kda_chunk': {'heads': H, 'head_dim': K, 'chunk_size': 64,
                      'recompute_attn': False,
                      'note': 'chunked recurrence; attends to all T via chunk-state passing'},
        'kda_chunk_compiled': {'heads': H, 'head_dim': K, 'chunk_size': 64,
                               'execution': 'torch.compile fullgraph=True',
                               'note': ('same per-channel-decay chunked '
                                        'recurrence as kda_chunk; '
                                        'torch.compile-fused setup + '
                                        'state carry')},
        'kda_rec_hybrid': {'heads': H2, 'head_dim': K2,
                           'note': 'recurrent delta-rule at hybrid KDA sub-layer config (H=2, K=V=16); directly comparable to the hybrid stack\'s per-KDA-layer cost'},
        'kda_chunk_hybrid': {'heads': H2, 'head_dim': K2, 'chunk_size': 64,
                             'note': 'chunked recurrence at hybrid KDA sub-layer config (H=2, K=V=16); directly comparable to the hybrid stack\'s per-KDA-layer cost'},
        'csa':       {'m': 8, 'topk': 4, 'sliding_window': 8, 'heads': 2,
                      'c': 16, 'dc': 32, 'nIh': 2, 'cI': 8, 'd': d,
                      'attends_to_all_T': False,
                      'note': 'attends to at most topk compressed blocks + sliding_window keys (~constant in T)'},
        'hca':       {'m2': 16, 'sliding_window': 8, 'heads': 2,
                      'c': 16, 'dc': 32, 'd': d,
                      'attends_to_all_T': False,
                      'note': 'attends to all causally-visible compressed entries (O(T/m2), linear in T) + sliding_window keys'},
        'hybrid':    {'n_layers': 5, 'n_kda': 3, 'n_csa': 1, 'n_hca': 1, 'd': d,
                      'heads_qk': 2, 'heads_v': 2, 'head_dim_k': 16, 'head_dim_v': 16,
                      'csa_m': 8, 'csa_topk': 4, 'csa_sliding_window': 8,
                      'hca_m2': 16, 'hca_sliding_window': 8,
                      # Full per-sublayer geometry, so downstream analytic
                      # accounting (run_roofline / run_kv_cache) can
                      # reconstruct FLOPs and byte traffic for the EXACT
                      # benchmarked stack instead of assuming paper-scale
                      # defaults.
                      'csa_c': 16, 'csa_dc': 32, 'csa_nIh': 2, 'csa_cI': 8,
                      'csa_nh': 2,
                      'hca_c': 16, 'hca_dc': 32, 'hca_nh': 2,
                      'kda_chunk_size': 64, 'kda_conv_ksize': 3,
                      'attends_to_all_T': False,
                      'note': 'hybrid stack: KDA recurrence covers all T; CSA layer is sparse (topk-capped, ~constant radius); HCA layer attends to all compressed blocks (linear in T)'},
        'mamba2_scan': {'heads': 4, 'headdim': 32, 'd_state': 32,
                        'ngroups': 1, 'mamba_chunk_size': 0,
                        'note': ('SSD scan core only; x/dt/B/C '
                                 'pre-projected (4 heads x 32x32 state, '
                                 'capacity-matched to the kda_rec core '
                                 'row); sequential reference scan')},
        'mamba2_chunked_scan': {'heads': 4, 'headdim': 32, 'd_state': 32,
                                'ngroups': 1, 'mamba_chunk_size': 64,
                                'note': ('same core boundary and inputs as '
                                         'mamba2_scan; chunk_size=64 '
                                         'chunked SSD decomposition '
                                         '(parity-gated against the '
                                         'sequential reference)')},
        'mamba2_layer': {'d': d, 'expand': 2, 'd_state': 16, 'd_conv': 4,
                         'headdim': 32, 'backend_env': 'MAMBA_BACKEND',
                         'note': 'full Mamba-2 mixer from hidden state (in_proj + causal conv + x_proj + scan + gate + out_proj)'},
        'hybrid_mamba2': {'n_layers': 5, 'memory': 'mamba2', 'n_memory': 3,
                          'n_csa': 1, 'n_hca': 1, 'd': d,
                          'csa_c': 16, 'csa_dc': 32, 'csa_nh': 2, 'csa_m': 8,
                          'csa_topk': 4, 'csa_sliding_window': 8,
                          # Full CSA geometry, mirroring the hybrid row: the
                          # indexer widths are needed by prefill_flops('csa')
                          # in the roofline join, and the config above sets
                          # them explicitly (csa_nIh=2, csa_cI=8).
                          'csa_nIh': 2, 'csa_cI': 8,
                          'hca_c': 16, 'hca_dc': 32, 'hca_nh': 2, 'hca_m2': 16,
                          'hca_sliding_window': 8,
                          'mamba_expand': 2, 'mamba_d_state': 16,
                          'mamba_d_conv': 4, 'mamba_headdim': 32,
                          'attends_to_all_T': False,
                          'note': 'same stack/boundary as the hybrid row with the memory slots filled by Mamba-2 instead of KDA (3:1:1)'},
    }

    results = []
    n_repeats = parse_int_env('BENCH_REPEATS', 5, min_value=1, logger=logger)
    prev_nonfinite_warn = ops_kda._KDA_NONFINITE_WARN
    ops_kda._KDA_NONFINITE_WARN = False
    try:
        for T in seq_lengths:
            logger.info(f'\n-- T = {T} --')
            for name, factory in benches:
                try:
                    _clear_cache(device)
                    meta = {}
                    fn = factory(T, meta=meta)
                    t, mem = _measure(fn, repeats=n_repeats, device=device)
                    row = {'T': T, 'op': name, 'time_ms': t * 1e3, 'peak_mem_MB': mem,
                           'device': str(device), 'repeats': n_repeats,
                           'tokens_per_s': ((B * T) / t) if t > 0 else None,
                           'param_elements': meta.get('param_elements')}
                    if name in {'kda_rec', 'kda_chunk', 'kda_chunk_compiled',
                                'kda_rec_hybrid', 'kda_chunk_hybrid',
                                'hybrid'}:
                        # The compiled row is always the reference operator
                        # executed by torch.compile (even when KDA_BACKEND
                        # selects FLA for the eager rows), so record the
                        # reference identity for the roofline boundary join
                        # and the execution mode separately.
                        row['kda_backend'] = (
                            'reference' if name == 'kda_chunk_compiled'
                            else _selected_kda_backend())
                    if meta.get('kda_execution') is not None:
                        row['kda_execution'] = meta['kda_execution']
                        row['kda_compile_fullgraph'] = meta.get(
                            'kda_compile_fullgraph')
                        row['kda_compile_dynamic'] = meta.get(
                            'kda_compile_dynamic')
                        row['kda_chunk_parity_max_abs'] = meta.get(
                            'kda_chunk_parity_max_abs')
                        row['kda_chunk_state_parity_max_abs'] = meta.get(
                            'kda_chunk_state_parity_max_abs')
                        row['kda_chunk_parity_ok'] = meta.get(
                            'kda_chunk_parity_ok')
                    if name in {'kda_chunk', 'kda_chunk_compiled'}:
                        row['kda_chunk_size'] = meta.get(
                            'kda_chunk_size', 64)
                        row['kda_recompute_attn'] = meta.get(
                            'kda_recompute_attn', False)
                    if name in {'mamba2_scan', 'mamba2_chunked_scan',
                                'mamba2_layer', 'hybrid_mamba2'}:
                        row['mamba_backend'] = _selected_mamba_backend()
                        row['mamba_effective_backend'] = meta.get(
                            'mamba_effective_backend', 'reference')
                    # Operator-boundary metadata for the chunked SSD row:
                    # decomposition + the parity residual measured against
                    # the sequential reference on the SAME inputs.
                    if meta.get('scan_boundary') is not None:
                        row['scan_boundary'] = meta['scan_boundary']
                        row['scan_parity_max_abs'] = meta.get(
                            'scan_parity_max_abs')
                        row['scan_parity_ok'] = meta.get('scan_parity_ok')
                    if name in {'csa', 'hybrid'}:
                        row['csa_indexer_normalize_qk'] = True
                        row['csa_ste_in_timed_region'] = False
                    row['time_min_ms'] = _LAST_TIMING_STATS.get('min_ms')
                    row['time_max_ms'] = _LAST_TIMING_STATS.get('max_ms')
                    row['time_std_ms'] = _LAST_TIMING_STATS.get('std_ms')
                    row['time_mean_ms'] = _LAST_TIMING_STATS.get('mean_ms')
                    row['time_q10_ms'] = _LAST_TIMING_STATS.get('q10_ms')
                    row['time_q90_ms'] = _LAST_TIMING_STATS.get('q90_ms')
                    row['time_ci95_low_ms'] = _LAST_TIMING_STATS.get(
                        'ci95_low_ms')
                    row['time_ci95_high_ms'] = _LAST_TIMING_STATS.get(
                        'ci95_high_ms')
                    row['time_ci95_halfwidth_ms'] = _LAST_TIMING_STATS.get(
                        'ci95_halfwidth_ms')
                    row['time_cv'] = _LAST_TIMING_STATS.get('cv')
                    row['time_n'] = _LAST_TIMING_STATS.get('n')
                    row['time_ci_method'] = _LAST_TIMING_STATS.get(
                        'ci_method')
                    row['time_bootstrap_reps'] = _LAST_TIMING_STATS.get(
                        'bootstrap_reps')
                    row.update(op_boundary[name])
                    row['workload'] = op_workload.get(name, {})
                    results.append(row)
                    mem_str = f'{mem:8.2f} MB' if mem is not None else '     n/a'
                    print(f'  {name:12s}  time={t*1e3:8.2f} ms  mem={mem_str}')
                except Exception as e:
                    err_row = {
                        'T': T, 'op': name, 'error': str(e),
                        'device': str(device),
                        'kda_backend': ('reference'
                                        if name == 'kda_chunk_compiled'
                                        else (os.environ.get(
                                            'KDA_BACKEND', 'reference')
                                            if name in {
                                                'kda_rec', 'kda_chunk',
                                                'kda_rec_hybrid',
                                                'kda_chunk_hybrid',
                                                'hybrid'} else None)),
                        'mamba_backend': (os.environ.get('MAMBA_BACKEND', 'reference')
                                          if name in {'mamba2_scan',
                                                      'mamba2_chunked_scan',
                                                      'mamba2_layer',
                                                      'hybrid_mamba2'} else None),
                        'time_ms': None, 'peak_mem_MB': None,
                        'tokens_per_s': None,
                        'time_min_ms': None, 'time_max_ms': None,
                        'time_std_ms': None, 'time_mean_ms': None,
                        'time_q10_ms': None, 'time_q90_ms': None,
                        'time_ci95_low_ms': None, 'time_ci95_high_ms': None,
                        'time_ci95_halfwidth_ms': None, 'time_cv': None,
                        'time_n': 0, 'time_ci_method': None,
                        'time_bootstrap_reps': None,
                        'repeats': n_repeats,
                        'param_elements': None,
                    }
                    if name == 'kda_chunk_compiled':
                        err_row['kda_execution'] = 'torch.compile'
                        err_row['kda_chunk_size'] = 64
                    if name in {'csa', 'hybrid'}:
                        err_row['csa_indexer_normalize_qk'] = True
                        err_row['csa_ste_in_timed_region'] = False
                    err_row.update(op_boundary[name])
                    err_row['workload'] = op_workload.get(name, {})
                    results.append(err_row)
                    logger.error(f'  {name:12s}  ERROR: {e}')
    finally:
        ops_kda._KDA_NONFINITE_WARN = prev_nonfinite_warn

    os.makedirs('results', exist_ok=True)
    violations = validate_rows('exp2_benchmark', results)
    if violations:
        shown = '; '.join(violations[:5])
        more = (f' (+{len(violations) - 5} more)'
                if len(violations) > 5 else '')
        raise RowContractError(
            f'exp2_benchmark: {len(violations)} row-contract violation(s): '
            f'{shown}{more}')
    try:
        write_json_atomic(results, 'results/exp2_benchmark.json',
                          indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in results; sanitizing to null: {e}')
        write_json_atomic(sanitize_for_json(results),
                          'results/exp2_benchmark.json',
                          indent=2, allow_nan=False)
    prov = None
    try:
        prov = capture_provenance()
        # This benchmark seeds per-(op, T) generators rather than a global
        # RNG; record the scheme explicitly so the numbers are traceable
        # instead of "which random weights was this?".
        prov['seeds'] = {
            'scheme': ('per-(op, T) seed = zlib.crc32(op_name) XOR '
                       '(T * 2654435761), used by _make_op_gen for inputs and '
                       'by the scoped weight-init seeding (torch.manual_seed '
                       'wrapped in get/set_rng_state); no global seed needed'),
        }
        write_json_atomic(prov,
                          'results/exp2_benchmark_provenance.json',
                          indent=2, allow_nan=False)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info('\nSaved: results/exp2_benchmark.json')
    n_errors = sum(1 for r in results if 'error' in r)
    # Record the exact per-(op, T) input seeds rather than a fake global
    # seed: the benchmark is deliberately independent of process RNG state.
    input_seeds = {
        (zlib.crc32(str(r['op']).encode('utf-8')) & 0xFFFFFFFF)
        ^ ((int(r['T']) * 2654435761) & 0xFFFFFFFF)
        for r in results if 'op' in r and 'T' in r}
    weight_init_seeds = {
        name: zlib.crc32(label.encode('utf-8')) & 0xFFFFFFFF
        for name, label in (
            ('mamba2_layer', b'mamba2_layer_weights'),
            ('hybrid', b'hybrid_weights'),
            ('hybrid_mamba2', b'hybrid_mamba2_weights'),
        )}
    used_seeds = sorted(input_seeds | set(weight_init_seeds.values()))
    append_run_manifest(make_run_manifest_entry(
        experiment='exp2_benchmark',
        seeds=used_seeds,
        provenance=prov,
        hyperparameters={
            'device': str(device),
            'seq_lengths': seq_lengths,
            'model_geometry': {'B': B, 'H': H, 'K': K, 'V': V, 'd': d},
            'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
            'mamba_backend': os.environ.get('MAMBA_BACKEND', 'reference'),
            'seed_scheme': ('per-(op, T) zlib.crc32 seeds; scoped weight '
                            'init restores the caller RNG state'),
            'weight_init_seeds': weight_init_seeds,
        },
        data={
            'kind': 'synthetic_benchmark_workloads',
            'seq_lengths': seq_lengths,
            'geometry': {'B': B, 'H': H, 'K': K, 'V': V, 'd': d},
            'inputs': 'seeded torch.Generator tensors; no corpus file',
        },
        artifacts={
            'results': 'results/exp2_benchmark.json',
            'provenance': 'results/exp2_benchmark_provenance.json',
        },
        results={
            'n_rows': len(results),
            'n_errors': int(n_errors),
        },
        status=('completed' if not n_errors else 'partial'),
    ), logger=logger)
    if n_errors:
        logger.error(
            f'\n{n_errors}/{len(results)} (T, op) cells errored out. '
            f'Returning non-zero so run_all records this experiment as failed.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
