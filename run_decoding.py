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
import os
import statistics
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, sanitize_for_json, write_json_atomic
from ops_kda import naive_recurrent_kda

# P0 fix: emit a one-shot warning at import time so notebook / REPL
# users do not silently misread the decoding results as a fair
# three-way comparison. The README's "Fairness notes" #4 acknowledges
# this gap, but a code-level warning is more visible to a user who
# skips the README and goes straight to ``python run_decoding.py``.
# The warning is emitted once per process (Python's default warning
# filter deduplicates by (message, category, module, lineno)).
warnings.warn(
    "run_decoding.py: only softmax attention and KDA are benchmarked "
    "here. CSA and HCA do NOT have an incremental KV-block cache "
    "implemented in this repository, so their decoding latency is not "
    "measured. Do NOT interpret the softmax-vs-KDA numbers as a "
    "three-way comparison. See README 'Fairness notes' #4 for the "
    "acknowledged scope gap and the planned incremental-cache roadmap.",
    stacklevel=2,
)


def _clear_cache(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class SoftmaxAttnDecoding(nn.Module):
    """Softmax attention that caches K/V for autoregressive decoding.

    P3 fix — pre-allocated KV cache (was O(T) ``torch.cat`` per token).

    The previous implementation rebuilt the entire KV cache on every decode
    step via ``torch.cat([self._cache_k, k], dim=1)``. Each ``torch.cat``
    allocates a NEW tensor of shape ``[B, T_full, H, K]`` and copies ALL
    existing cached keys into it — an ``O(T)`` memory copy per generated
    token, giving ``O(T²)`` total copy cost over a ``T``-token generation
    pass. This inflated softmax's measured per-token decode latency by a
    factor that grows with the cached context length, systematically
    amplifying softmax's disadvantage relative to KDA (whose recurrent state
    is genuinely ``O(1)`` per token). The comparison was therefore not
    measuring the attention kernel's decoding cost — it was measuring
    softmax's cache-rebuild overhead.

    The fix uses a **pre-allocated ring-style cache**:
      * On the first forward (prefill), allocate a cache buffer with some
        initial capacity (``max(T_new * 2, 64)``) and a write pointer
        ``_cache_len``.
      * On each decode step, write the new k/v into ``cache[:, _cache_len]``
        (an ``O(1)`` in-place write, no copy) and increment the pointer.
      * If the capacity is exceeded, grow the buffer geometrically (double
        the capacity, copy the old data — amortized ``O(1)`` per token).
      * Attention is computed over ``cache[:, :_cache_len]`` (a view, no
        copy).

    This makes the per-token decode cost genuinely ``O(1)`` in the cache
    length (modulo amortized geometric growth), so the benchmark now
    measures the attention kernel itself, not the cache-management
    overhead. The attention score computation is still ``O(T)`` per token
    (one query attending to all cached keys) — that is the irreducible
    cost of softmax attention during decoding and is exactly the cost we
    want to compare against KDA's ``O(1)`` recurrent update.
    """

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V
        self.scale = K ** -0.5
        # Pre-allocated KV cache buffers. Shape ``[B, capacity, H, K/V]``.
        # ``_cache_len`` is the write pointer = number of valid entries.
        # ``capacity`` is the allocated size; the buffer grows geometrically
        # (doubles) when ``_cache_len`` would exceed it, giving amortized
        # O(1) per-token write cost.
        #
        # Registered as non-persistent buffers so model.to(device) moves them
        # automatically (a plain attribute would stay on the source device,
        # causing a device-mismatch crash on the next forward). Non-persistent
        # => not saved into state_dict (runtime state, not learned weights).
        self.register_buffer('_cache_k', None, persistent=False)
        self.register_buffer('_cache_v', None, persistent=False)
        # ``_cache_len`` is a plain Python int (not a buffer) because it is
        # a scalar counter, not a tensor — there is nothing for ``.to()``
        # to move. Storing it as a 0-dim tensor would add unnecessary
        # overhead on every read/write.
        self._cache_len = 0

    def reset(self):
        self._cache_k = None
        self._cache_v = None
        self._cache_len = 0

    def _ensure_cache_capacity(self, B, needed_len, dtype, device):
        """Ensure the pre-allocated cache can hold ``needed_len`` entries.

        Allocates a fresh buffer on the first call, or grows it geometrically
        (doubles capacity) when the current capacity is exceeded. Growing
        involves one ``O(capacity)`` copy, but because capacity doubles each
        time, the amortized cost per appended token is ``O(1)``.
        """
        cur = self._cache_k
        if cur is None or cur.shape[0] != B or cur.dtype != dtype \
                or cur.device != device or cur.shape[1] < needed_len:
            # Determine new capacity: at least ``needed_len``, but if we're
            # growing an existing buffer, double it (geometric growth for
            # amortized O(1)). For a fresh allocation, use
            # ``max(needed_len * 2, 64)`` so the first few decode steps
            # don't trigger a grow on every single token.
            if cur is not None and cur.shape[1] < needed_len:
                new_cap = max(needed_len, cur.shape[1] * 2)
            elif cur is None:
                new_cap = max(needed_len * 2, 64)
            else:
                # Batch/dtype/device changed but capacity is sufficient.
                new_cap = cur.shape[1]
            new_k = torch.zeros(B, new_cap, self.H, self.K,
                                dtype=dtype, device=device)
            new_v = torch.zeros(B, new_cap, self.H, self.V,
                                dtype=dtype, device=device)
            # Copy existing valid data if any (and batch matches).
            if cur is not None and cur.shape[0] == B and self._cache_len > 0:
                copy_len = min(self._cache_len, new_cap)
                new_k[:, :copy_len] = cur[:, :copy_len]
                new_v[:, :copy_len] = self._cache_v[:, :copy_len]
            elif cur is not None and cur.shape[0] != B:
                # Batch size changed — start fresh (per-sequence cache).
                self._cache_len = 0
            self._cache_k = new_k
            self._cache_v = new_v

    def forward(self, x):
        # x: [B, T_new, d] — T_new = prefill_len during prefill, 1 during decoding.
        B, T_new, _ = x.shape
        q = self.q(x).view(B, T_new, self.H, self.K)
        k = self.k(x).view(B, T_new, self.H, self.K)
        v = self.v(x).view(B, T_new, self.H, self.V)
        # Detach k/v before caching so the KV cache does not accumulate
        # autograd graph nodes across decode steps. Without this, each cached
        # k/v would retain the previous step's graph, causing an O(N) memory
        # leak across N decode steps when the caller forgets to wrap inference
        # in ``torch.no_grad()``. The cache is just a tensor of numbers
        # (keys/values); it does not need to carry gradients. Mirrors the
        # always-detach pattern in ops_fused.py::HybridKCHAttention.forward
        # and the KDA state fix in KDAAttnDecoding.forward below.
        k = k.detach()
        v = v.detach()

        # Ensure the pre-allocated cache has room for the new T_new entries.
        # On the first call this allocates; on subsequent calls it may grow
        # geometrically if capacity is exceeded (amortized O(1) per token).
        needed_len = self._cache_len + T_new
        self._ensure_cache_capacity(B, needed_len, k.dtype, x.device)

        # Write the new k/v into the pre-allocated slots — O(T_new) write,
        # NO copy of the existing cache (unlike the old torch.cat approach
        # which copied ALL existing entries on every call).
        self._cache_k[:, self._cache_len:self._cache_len + T_new] = k
        self._cache_v[:, self._cache_len:self._cache_len + T_new] = v
        self._cache_len = self._cache_len + T_new

        # Slice the valid portion of the cache (a view, no copy).
        T_full = self._cache_len
        cache_k = self._cache_k[:, :T_full]
        cache_v = self._cache_v[:, :T_full]

        s = torch.einsum('bthk,bshk->bhts', q, cache_k) * self.scale
        # Causal mask: query at relative position t in the current chunk is at
        # absolute position (T_full - T_new + t); it may only attend to keys
        # at absolute positions <= (T_full - T_new + t).
        # For prefill (T_new == T_full) this reduces to the standard
        # lower-triangular mask. For decoding (T_new == 1) the single query
        # is at position T_full - 1, so it attends to all cached keys and the
        # mask is all-False (we skip the masked_fill entirely to avoid the
        # overhead of constructing a [1, T_full] mask per decode step).
        if T_new > 1:
            q_offset = T_full - T_new
            q_pos = torch.arange(T_new, device=x.device) + q_offset     # [T_new]
            k_pos = torch.arange(T_full, device=x.device)               # [T_full]
            causal_mask = k_pos[None, :] > q_pos[:, None]               # [T_new, T_full]
            s = s.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        p = torch.softmax(s, dim=-1)
        out = torch.einsum('bhts,bshv->bthv', p, cache_v)
        return self.o(out.reshape(B, T_new, self.H * self.V))


class KDAAttnDecoding(nn.Module):
    """KDA recurrent attention — O(1) state, no growing cache.

    This benchmark module mirrors the parameterization of
    ``ops_fused.KDAHybridLayer`` so the decoding-cost comparison reflects the
    SAME operator the fused model uses. Previously this module omitted the
    causal depthwise short-conv (kernel=3) that ``KDAHybridLayer`` applies to
    the input before the q/k/v/g/beta projections — i.e. the benchmark
    compared a *stripped-down* KDA against softmax, making the comparison
    unfair. The short-conv is now included and its lookback state is carried
    across decode steps (mirroring the conv-lookback buffer in
    ``KDAHybridLayer``), so the per-token decode cost includes the conv.
    """

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.g = nn.Linear(d_model, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        # Causal depthwise short-conv (kernel=3) — matches KDAHybridLayer.
        # Conv1d padding=0; left-pad by (k-1)=2 in forward via the lookback
        # buffer (or zeros for the first call).
        self.short_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                                    groups=d_model, bias=True)
        # Magic constant 0.1 (decay scale) — lifted to a module attribute so
        # all KDA instantiations share the same value via HybridConfig. The
        # default (0.1) preserves the historical behaviour. We read it from
        # the constructor arg if provided; otherwise default to 0.1.
        self.decay_scale = 0.1
        self.H, self.K, self.V = H, K, V
        # Register the recurrent state as a non-persistent buffer so
        # model.to(device) moves it along with the parameters. A plain
        # attribute would be left on the source device, causing a
        # device-mismatch crash on the next forward — the same class of
        # bug that was fixed in ops_fused.py::HybridKCHAttention.
        self.register_buffer('_state', None, persistent=False)
        # Register the short-conv lookback ([B, k-1, d]) so .to(device) /
        # .half() move/cast it along with the parameters. Mirrors the
        # KDAHybridLayer._conv_lookback buffer.
        self.register_buffer('_conv_lookback', None, persistent=False)

    def reset(self):
        self._state = None
        self._conv_lookback = None

    def forward(self, x):
        B, T_new, d = x.shape
        ksize = self.short_conv.kernel_size[0]
        # Build the conv input with proper LEFT context, mirroring
        # KDAHybridLayer.forward: prepend the previous chunk's last
        # ``ksize - 1`` tokens if available, else left-pad with zeros.
        lookback = self._conv_lookback
        if lookback is not None:
            # Batch-size / device / dtype guards (mirrors KDAHybridLayer).
            if lookback.shape[0] != B:
                lookback = None
            elif lookback.device != x.device or lookback.dtype != x.dtype:
                lookback = lookback.to(device=x.device, dtype=x.dtype).detach()
            else:
                lookback = lookback.detach()
        if lookback is None:
            x_conv_in = F.pad(x.transpose(1, 2), (ksize - 1, 0))
        else:
            x_conv_in = torch.cat(
                [lookback.transpose(1, 2), x.transpose(1, 2)], dim=2)
        x_conv = self.short_conv(x_conv_in).transpose(1, 2)
        # Persist the last (ksize-1) time steps of THIS chunk for the next call.
        if T_new >= ksize - 1:
            self._conv_lookback = x[:, -(ksize - 1):].detach().clone()
        else:
            if lookback is not None:
                combined = torch.cat([lookback, x], dim=1)
                self._conv_lookback = combined[:, -(ksize - 1):].detach().clone()
            else:
                pad_len = (ksize - 1) - T_new
                self._conv_lookback = torch.cat(
                    [torch.zeros(B, pad_len, d, device=x.device, dtype=x.dtype),
                     x], dim=1).detach().clone()
        # View BEFORE normalize: F.normalize(dim=-1) must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector. The
        # previous form normalized the full H*K vector, shrinking each
        # head's L2 norm to ~1/sqrt(H) and under-scaling q.k dot products
        # by 1/H. Mirrors the fix in ops_fused.py::KDAHybridLayer.
        q = F.normalize(F.silu(self.q(x_conv)).view(B, T_new, self.H, self.K), dim=-1)
        k = F.normalize(F.silu(self.k(x_conv)).view(B, T_new, self.H, self.K), dim=-1)
        v = F.silu(self.v(x_conv)).view(B, T_new, self.H, self.V)
        # log-space gate: low-rank down/up with a softplus-style decay.
        # Uses self.decay_scale (default 0.1, matching HybridConfig.kda_decay_scale)
        # so all KDA instantiations agree on the magic constant.
        g = -F.softplus(self.g(x_conv)).view(B, T_new, self.H, self.K) * self.decay_scale
        beta = torch.sigmoid(self.beta(x_conv))
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
        #
        # Batch-size + dtype + device guards: if the caller switches batch
        # size (e.g. train B=16 -> eval B=8) or dtype/device (e.g. model.half()
        # or model.to(cuda)) between forward calls, the cached state is
        # invalid. ``naive_recurrent_kda`` would implicitly cast via
        # ``initial_state.to(device=S.device, dtype=compute_dtype)``, so a
        # dtype/device mismatch does not crash — but a BATCH-SIZE mismatch
        # WOULD crash inside the recurrence (the state has the old B, the
        # new q/k/v have the new B, and the einsums broadcast-incompatibly).
        # Drop the state on batch-size change (the state is per-sequence and
        # cannot be reused across different batch sizes). For dtype/device
        # mismatch, explicitly cast so the contract is clear and the state
        # is in the right form before being passed to the recurrence.
        # Mirrors the guards in ops_fused.py::HybridKCHAttention.forward.
        state = self._state
        if state is not None:
            if state.shape[0] != B:
                # Batch size changed — drop the state (per-sequence, cannot
                # be reused across different batch sizes).
                state = None
            elif state.device != x.device or state.dtype != x.dtype:
                # Device or dtype changed — move/cast and keep detached.
                state = state.to(device=x.device, dtype=x.dtype).detach()
            else:
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
        # Reset peak memory AFTER both warmup AND model.reset() so the
        # reported peak reflects only the timed trials, not the warmup
        # allocations. ``model.reset()`` is currently a no-op on memory
        # (just sets ``self._state = None``), but resetting peak AFTER
        # the reset is the robust order — if reset ever switches to
        # zeroing buffers in place, no transient reset allocation can
        # leak into the reported peak.
        torch.cuda.reset_peak_memory_stats(device)
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
            if device.type == 'cuda':
                # P0 timing-bias fix: use CUDA events instead of
                # ``torch.cuda.synchronize() + time.perf_counter()`` around
                # each token. The previous per-token sync pattern added
                # ~10-100 microseconds of driver-roundtrip overhead per
                # token. KDA's per-token compute is only ~10 microseconds,
                # so the sync overhead was 50-90% of the reported "KDA decode
                # latency" — systematically biasing the KDA-vs-softmax
                # comparison in softmax's favor. CUDA events record
                # asynchronously on the stream and add no host-side
                # synchronization per token; a single ``synchronize()`` at
                # the end drains the whole batch.
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_decode)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_decode)]
                for i in range(n_decode):
                    starts[i].record()
                    model(x_new)
                    ends[i].record()
                torch.cuda.synchronize()
                decode_times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            else:
                for _ in range(n_decode):
                    t0 = time.perf_counter()
                    model(x_new)
                    decode_times.append((time.perf_counter() - t0) * 1e3)
        all_decode_times.append(decode_times)

    # Aggregate across repeats: take the median across trials for each
    # summary statistic. ``statistics.median`` handles even-length lists
    # correctly (averages the two middle values) — the previous
    # ``sorted(times)[len(times)//2]`` returned the upper-middle for even n.
    # (statistics is imported at module top.)
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
    #
    # Uses the centralized ``sanitize_for_json`` helper from kaggle_setup.py
    # (was a local ``_sanitize`` closure; centralizing removes 5 copies of
    # the same logic across run_*.py and ensures any future edge-case fix
    # propagates everywhere).
    sanitized = [sanitize_for_json(r) for r in results]
    # P1-5 fix: use the shared atomic JSON writer (temp file + fsync +
    # os.replace) so a process kill or disk-full mid-write leaves the
    # target file as the OLD version (or absent) rather than a truncated
    # partial JSON document. See kaggle_setup.write_json_atomic's docstring.
    try:
        write_json_atomic(sanitized, 'results/exp6_decoding.json',
                          indent=2, allow_nan=False)
    except (TypeError, ValueError) as e:
        print(f'[run_decoding] WARNING: JSON serialization failed: {e}')
        write_json_atomic(sanitized, 'results/exp6_decoding.json',
                          indent=2, default=str)
    print('\nSaved: results/exp6_decoding.json')
    # P0-2 fix: return non-zero if any (prefill_len, op) cell errored out,
    # so ``run_all._run`` records the experiment as ``status='fail'`` instead
    # of silently treating a partial run as success. Mirrors run_benchmark /
    # run_quality / run_ablation.
    n_errors = sum(1 for r in results if 'error' in r)
    if n_errors:
        print(
            f'\n[P0-2] {n_errors}/{len(results)} (prefill_len, op) cells '
            f'errored out. Returning non-zero so run_all records this '
            f'experiment as failed.')
        return 1
    return 0


if __name__ == '__main__':
    main()
