"""Experiment 6 — autoregressive decoding latency benchmark.

The original paper only reported *prefill* latency on CPU. A reviewer flagged
that decoding latency (token-by-token generation) is the more relevant number
for long-context inference, and that CPU single-thread numbers cannot be
extrapolated to real serving hardware.

This benchmark addresses both points within the Kaggle-feasible envelope:

  * **Decoding latency.** For each operator we run a fixed-length prefill
    (build the full context once) and then decode ``n_new`` tokens one at a
    time, measuring the per-token wall-clock latency. KDA's recurrent form
    should shine here (O(1) per token), softmax attention should grow linearly
    in the cached sequence length.
  * **Real GPU memory.** On the T4 we report
    ``torch.cuda.max_memory_allocated`` during decoding — the actual number a
    serving engine pays.
  * **Honest scope.** We only measure the attention operator itself (not a
    full LM), on a Kaggle T4. We do not claim these numbers extrapolate to
    H100/H20; they are a *relative* comparison of the operators' decoding
    cost growth, which is exactly what the reviewer asked for.

Note: KDA's ``naive_recurrent_kda`` is a Python loop, so its wall-clock on
CPU is dominated by Python overhead. On GPU the per-step kernel launch
overhead dominates. The *trend* (flat vs growing) is the signal, not the
absolute microseconds.
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device
from ops_kda import naive_recurrent_kda


def _clear_cache(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class SoftmaxAttnDecoding(nn.Module):
    """Softmax attention that caches K/V for autoregressive decoding."""

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V
        self.scale = K ** -0.5
        # Register KV cache as non-persistent buffers so model.to(device)
        # moves them automatically (a plain attribute would stay on the
        # source device, causing a device-mismatch crash on the next forward).
        # Non-persistent => not saved into state_dict (runtime state, not
        # learned weights).
        self.register_buffer('_cache_k', None, persistent=False)
        self.register_buffer('_cache_v', None, persistent=False)

    def reset(self):
        self._cache_k = None
        self._cache_v = None

    def forward(self, x):
        # x: [B, T_new, d] — T_new = prefill_len during prefill, 1 during decoding.
        # ``d`` is intentionally not unpacked: it is unused here (we read
        # ``self.H * self.K`` etc. from the layer config, not from x.shape).
        B, T_new, _ = x.shape
        q = self.q(x).view(B, T_new, self.H, self.K)
        k = self.k(x).view(B, T_new, self.H, self.K)
        v = self.v(x).view(B, T_new, self.H, self.V)
        if self._cache_k is None:
            # Detach k/v before caching so the KV cache does not accumulate
            # autograd graph nodes across decode steps. Without this, each
            # ``torch.cat`` in the else-branch below would retain the previous
            # step's graph, causing an O(N) memory leak across N decode steps
            # when the caller forgets to wrap inference in ``torch.no_grad()``.
            # The cache is just a tensor of numbers (keys/values); it does not
            # need to carry gradients. Mirrors the always-detach pattern in
            # ops_fused.py::HybridKCHAttention.forward and the KDA state fix
            # in KDAAttnDecoding.forward below.
            self._cache_k = k.detach()
            self._cache_v = v.detach()
        else:
            # Cast incoming k/v to the cache's dtype before concat. The
            # cache is set on the FIRST forward call (prefill) and retains
            # whatever dtype prefill used. If a later call passes a
            # different dtype (e.g. fp16 prefill then fp32 decode, or
            # mixed-precision training where weights are fp32 but inputs
            # are fp16), torch.cat([fp16, fp32]) raises
            # ``RuntimeError: Expected object of scalar type Half but got
            # scalar type Float``. Casting to the cache dtype makes the
            # contract explicit: the cache dtype is fixed by the first
            # call, and all subsequent calls are coerced to match. This
            # mirrors the dtype-coercion pattern in
            # ops_kda.py::naive_recurrent_kda (initial_state.to(compute_dtype)).
            #
            # Detach before concat for the same reason as the first-call
            # branch: prevent graph accumulation across decode steps.
            cache_dtype = self._cache_k.dtype
            self._cache_k = torch.cat(
                [self._cache_k, k.to(cache_dtype).detach()], dim=1)
            self._cache_v = torch.cat(
                [self._cache_v, v.to(cache_dtype).detach()], dim=1)
        T_full = self._cache_k.shape[1]
        s = torch.einsum('bthk,bshk->bhts', q, self._cache_k) * self.scale
        # Causal mask: query at relative position t in the current chunk is at
        # absolute position (T_full - T_new + t); it may only attend to keys
        # at absolute positions <= (T_full - T_new + t).
        # For prefill (T_new == T_full) this reduces to the standard
        # lower-triangular mask. For decoding (T_new == 1) the single query
        # is at position T_full - 1, so it attends to all cached keys and the
        # mask is all-False (we skip the masked_fill entirely to avoid the
        # overhead of constructing a [1, T_full] mask per decode step).
        # Previously no causal mask was applied at all, which made the prefill
        # non-causal — the prefill output is discarded by the benchmark, but
        # the missing mask still made prefill_ms artificially low (no mask
        # construction / fill) and was incorrect for any autoregressive use.
        if T_new > 1:
            q_offset = T_full - T_new
            q_pos = torch.arange(T_new, device=x.device) + q_offset     # [T_new]
            k_pos = torch.arange(T_full, device=x.device)               # [T_full]
            causal_mask = k_pos[None, :] > q_pos[:, None]               # [T_new, T_full]
            s = s.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        p = torch.softmax(s, dim=-1)
        out = torch.einsum('bhts,bshv->bthv', p, self._cache_v)
        return self.o(out.reshape(B, T_new, self.H * self.V))


class KDAAttnDecoding(nn.Module):
    """KDA recurrent attention — O(1) state, no growing cache."""

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.g = nn.Linear(d_model, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V
        # Register the recurrent state as a non-persistent buffer so
        # model.to(device) moves it along with the parameters. A plain
        # attribute would be left on the source device, causing a
        # device-mismatch crash on the next forward — the same class of
        # bug that was fixed in ops_fused.py::HybridKCHAttention.
        self.register_buffer('_state', None, persistent=False)

    def reset(self):
        self._state = None

    def forward(self, x):
        B, T_new, _ = x.shape
        q = F.normalize(F.silu(self.q(x)), dim=-1).view(B, T_new, self.H, self.K)
        k = F.normalize(F.silu(self.k(x)), dim=-1).view(B, T_new, self.H, self.K)
        v = F.silu(self.v(x)).view(B, T_new, self.H, self.V)
        g = -F.softplus(self.g(x)).view(B, T_new, self.H, self.K) * 0.1
        beta = torch.sigmoid(self.beta(x))
        # Always detach the incoming state so the autograd graph from the
        # previous step is not retained. In training mode this prevents
        # "backward through the graph a second time" errors; in eval mode it
        # prevents an O(N) memory leak across N forward calls when the caller
        # forgets to wrap inference in ``torch.no_grad()`` (each call would
        # otherwise retain the previous call's graph, accumulating unbounded
        # memory during long autoregressive decoding). Stateful generation
        # works fine with a detached state — the state is just a tensor of
        # numbers, not a graph node.
        # Mirrors the fix in ops_fused.py::HybridKCHAttention.forward.
        state = self._state
        if state is not None:
            state = state.detach()
        o, self._state = naive_recurrent_kda(
            q, k, v, g, beta, scale=self.K ** -0.5,
            initial_state=state, output_final_state=True,
        )
        return self.o(o.reshape(B, T_new, self.H * self.V))


def bench_decoding(model, d_model, prefill_len, n_decode, device, repeats=3):
    """Measure per-token decoding latency after a fixed prefill.

    Returns dict with prefill_ms, mean_decode_ms_per_token, peak_mem_MB.

    ``repeats`` controls how many independent (prefill + decode-loop) trials
    are run. The median prefill and per-token decode latency across trials is
    reported, which is far more stable than the single-trial measurement the
    previous implementation used (the ``repeats`` parameter existed in the
    signature but was silently ignored — a clear bug).
    """
    # Move model to device FIRST, then clear cache and reset peak memory
    # stats. The previous order (_clear_cache -> model.to(device)) meant
    # max_memory_allocated captured the model parameter allocation too,
    # inflating peak_mem_MB by the parameter count (a constant offset that
    # varies per operator and makes the comparison less fair).
    model = model.to(device).eval()
    model.reset()
    _clear_cache(device)

    # Seed a dedicated generator (NOT the global RNG) so the input tensors
    # are identical across operators, prefill lengths, and runs. The
    # previous code called ``torch.randn(...)`` against the global RNG,
    # which consumed different states per operator (because each model's
    # ``nn.Linear`` init made a different number of RNG draws during
    # construction). This made run-to-run latency variance confounded
    # with input variance, and made the "median over repeats" noisier than
    # the inter-operator gap being measured.
    _seed_gen = torch.Generator(device=device)
    _seed_gen.manual_seed(0)

    # Pre-allocate the per-step decode input ONCE outside the timed loop.
    # The previous code allocated ``x_new = torch.randn(1, 1, d_model)`` inside
    # the timed region, so per-token latency included the cost of randn + the
    # scalar multiply — for KDA (tiny per-token compute) this overhead can
    # dominate the measurement.
    x_new = torch.randn(1, 1, d_model, device=device, generator=_seed_gen) * 0.1

    # One fixed prefill input (same across trials) so timing variance comes
    # from the model, not from input noise.
    x_prefill = torch.randn(1, prefill_len, d_model, device=device, generator=_seed_gen) * 0.1

    # Warmup: run prefill + a couple of decode steps once (untimed) so the
    # first timed trial is not paying one-time kernel compilation / autotune
    # / allocator-warmup costs. ``cudnn.benchmark=True`` (set in
    # kaggle_setup.py) triggers autotuning on the first conv/matmul call,
    # which can dominate the first prefill by 100x. Without warmup the
    # reported prefill_ms is "compute + one-time setup", not steady-state.
    with torch.no_grad():
        model.reset()
        model(x_prefill)
        for _ in range(min(3, n_decode)):
            model(x_new)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        # Reset peak memory AFTER warmup so the reported peak reflects the
        # timed trials, not the warmup allocations.
        torch.cuda.reset_peak_memory_stats(device)
        # IMPORTANT: reset the model state BEFORE capturing the baseline.
        # The warmup above populated the KV cache (softmax) / recurrent
        # state (KDA) with (prefill_len + 3) tokens of context. If we
        # captured baseline_bytes WITHOUT resetting, the reported
        # ``peak - baseline`` would only include the (n_decode - 3)-token
        # cache GROWTH during the timed region, NOT the full cache that
        # a serving engine pays for. For softmax with plen=2048, H=2,
        # K=V=16, fp32, the omitted cache is ~524 KB — making softmax
        # look artificially cheaper than KDA (whose state is ~2 KB).
        # Resetting here drops baseline to (params + persistent state)
        # so the reported peak reflects (params + full cache + activations).
        model.reset()
        # Re-capture baseline AFTER the reset: now it's just the model
        # parameters and any persistent buffers, NOT the warmup cache.
        baseline_bytes = torch.cuda.memory_allocated(device)
    else:
        baseline_bytes = 0

    prefill_times = []
    all_decode_times = []
    for _ in range(repeats):
        model.reset()
        # Prefill: process the whole context at once.
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x_prefill)
                torch.cuda.synchronize()
                prefill_times.append((time.perf_counter() - t0) * 1e3)
            else:
                t0 = time.perf_counter()
                model(x_prefill)
                prefill_times.append((time.perf_counter() - t0) * 1e3)

        # Decode n_decode tokens one at a time.
        decode_times = []
        with torch.no_grad():
            for _ in range(n_decode):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x_new)
                    torch.cuda.synchronize()
                    decode_times.append((time.perf_counter() - t0) * 1e3)
                else:
                    t0 = time.perf_counter()
                    model(x_new)
                    decode_times.append((time.perf_counter() - t0) * 1e3)
        all_decode_times.append(decode_times)

    # Aggregate across repeats: take the median across trials for each
    # summary statistic. ``statistics.median`` handles even-length lists
    # correctly (averages the two middle values) — the previous
    # ``sorted(times)[len(times)//2]`` returned the upper-middle for even n.
    import statistics
    # Guard against empty timing lists (n_decode=0 or repeats=0). Previously
    # ``statistics.median([])`` raised ``StatisticsError: no median for empty
    # data`` and ``sum([])/0`` raised ``ZeroDivisionError``. Both are
    # degenerate configurations, but a defensive guard prevents a confusing
    # crash and instead reports zeros so the JSON row is still well-formed.
    prefill_ms = statistics.median(prefill_times) if prefill_times else 0.0
    # Per-token: median across (trial, token-step) samples — flattens the
    # repeats x n_decode matrix into one list and takes its median.
    flat_decode = [t for trial in all_decode_times for t in trial]
    median_decode = statistics.median(flat_decode) if flat_decode else 0.0
    mean_decode = sum(flat_decode) / len(flat_decode) if flat_decode else 0.0

    if device.type == 'cuda':
        # ``max_memory_allocated`` returns the peak across all timed trials
        # (warmup was excluded by the reset_peak_memory_stats call above).
        # We subtract the baseline allocation captured right after the reset
        # so the reported peak isolates the activation footprint (prefill
        # scores + KV cache growth) from the constant model-parameter offset
        # that varies per operator. Without the subtraction, the hybrid
        # model (more params) would report a higher "peak memory" than
        # softmax even if their activation footprints were identical — an
        # unfair comparison.
        #
        # The peak is dominated by prefill (which allocates [1, H, T, T]
        # attention scores ≈ 32 MB at plen=2048, vs the per-step decode
        # cache ≈ 260 KB). To report the *decoding* footprint (what a
        # serving engine actually pays in steady state), we would need to
        # reset peak stats after prefill — but then the reported number
        # would miss the prefill activations a serving engine retains in
        # the KV cache. Reporting the global peak (minus baseline) is the
        # honest choice: it is the maximum activation memory the model
        # needs at any point, including prefill.
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = max(0.0, peak_bytes - baseline_bytes) / (1024 ** 2)
    else:
        # CPU: tracemalloc doesn't capture torch tensors (native memory), and
        # RSS-based approximations are unreliable across platforms. Report
        # None so JSON serializes to null (clearly "no data") rather than 0.0
        # (which would look like a real measured value and mislead reviewers).
        peak_mb = None

    return {
        'prefill_ms': prefill_ms,
        'mean_decode_ms_per_token': mean_decode,
        'median_decode_ms_per_token': median_decode,
        'peak_mem_MB': peak_mb,
        'prefill_len': prefill_len,
        'n_decode': n_decode,
        'repeats': repeats,
    }


def main():
    info = configure_torch_for_device()
    device = info.device
    print('=' * 70)
    print(f'Experiment 6: Decoding Latency Benchmark ({device})')
    print('=' * 70)
    d_model = 64
    prefill_lens = [128, 512, 1024, 2048]
    n_decode = 20
    # Hoist n_repeats into a module-visible variable so the error path can
    # record the same value as the success path. Previously success rows
    # recorded ``repeats=3`` (the bench_decoding default) while error rows
    # recorded ``repeats=None`` — a schema inconsistency that broke
    # downstream consumers doing arithmetic on ``repeats`` (e.g.
    # ``n_decode / repeats``) on error rows.
    N_REPEATS = 3

    models = {
        'softmax': lambda: SoftmaxAttnDecoding(d_model),
        'kda':     lambda: KDAAttnDecoding(d_model),
    }

    results = []
    for plen in prefill_lens:
        print(f'\n-- prefill_len = {plen}, decode {n_decode} tokens --')
        for name, factory in models.items():
            try:
                model = factory()
                r = bench_decoding(model, d_model, plen, n_decode, device,
                                   repeats=N_REPEATS)
                r['op'] = name
                r['device'] = str(device)
                results.append(r)
                peak_str = 'n/a' if r['peak_mem_MB'] is None else f"{r['peak_mem_MB']:.2f}MB"
                print(f"  {name:10s}  prefill={r['prefill_ms']:8.2f}ms  "
                      f"decode/tok={r['median_decode_ms_per_token']:8.3f}ms  "
                      f"peak_mem={peak_str:>10}")
            except Exception as e:
                # Include null fields for the keys present on success rows so
                # downstream JSON consumers can do ``r['prefill_ms']`` without
                # a KeyError on error rows (mirrors run_benchmark.py's pattern).
                # ``repeats`` records N_REPEATS (not None) so error rows
                # match the success-row schema.
                results.append({'op': name, 'prefill_len': plen, 'error': str(e),
                                'device': str(device),
                                'prefill_ms': None,
                                'mean_decode_ms_per_token': None,
                                'median_decode_ms_per_token': None,
                                'peak_mem_MB': None,
                                'n_decode': n_decode,
                                'repeats': N_REPEATS})
                print(f"  {name:10s}  ERROR: {e}")

    # Summary: decode latency growth rate.
    print('\n' + '=' * 70)
    print('Per-token decoding latency (ms) vs. cached context length')
    print('=' * 70)
    print(f"{'op':>10} | " + " | ".join(f"{p:>8}" for p in prefill_lens))
    print('-' * 70)
    for name in models:
        cells = [f"{name:>10}"]
        for plen in prefill_lens:
            vals = [r['median_decode_ms_per_token'] for r in results
                    if r.get('op') == name and r.get('prefill_len') == plen
                    and 'error' not in r]
            if vals:
                cells.append(f"{vals[0]:>8.3f}")
            else:
                cells.append(f"{'n/a':>8}")
        print(" | ".join(cells))

    if device.type == 'cpu':
        print('\nNote: CPU peak memory is not reported (torch tensors use native')
        print('memory not visible to tracemalloc). On GPU (Kaggle T4) we report')
        print('torch.cuda.max_memory_allocated, the real serving cost.')

    os.makedirs('results', exist_ok=True)

    # Sanitize non-finite floats to null before serializing (mirrors
    # run_kv_cache.py). Without this, a single NaN/Inf in
    # ``prefill_ms`` / ``decode_ms_per_token`` / ``peak_mem_MB`` (e.g. from
    # KDA recurrence overflow, or from ``statistics.median`` propagating
    # a NaN in the underlying times list) would cause ``json.dump`` to
    # emit non-standard ``NaN`` / ``Infinity`` literals, breaking
    # downstream parsers (JS ``JSON.parse``, pandas, jq).
    def _sanitize(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(x) for x in o]
        return o
    sanitized = [_sanitize(r) for r in results]
    try:
        text = json.dumps(sanitized, indent=2, allow_nan=False)
    except (TypeError, ValueError) as e:
        print(f'[run_decoding] WARNING: JSON serialization failed: {e}')
        text = json.dumps(sanitized, indent=2, default=str)
    with open('results/exp6_decoding.json', 'w') as f:
        f.write(text)
    print('\nSaved: results/exp6_decoding.json')


if __name__ == '__main__':
    main()
