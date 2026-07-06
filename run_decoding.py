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
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, get_device
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
        B, T_new, d = x.shape
        q = self.q(x).view(B, T_new, self.H, self.K)
        k = self.k(x).view(B, T_new, self.H, self.K)
        v = self.v(x).view(B, T_new, self.H, self.V)
        if self._cache_k is None:
            self._cache_k = k
            self._cache_v = v
        else:
            self._cache_k = torch.cat([self._cache_k, k], dim=1)
            self._cache_v = torch.cat([self._cache_v, v], dim=1)
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
        B, T_new, d = x.shape
        q = F.normalize(F.silu(self.q(x)), dim=-1).view(B, T_new, self.H, self.K)
        k = F.normalize(F.silu(self.k(x)), dim=-1).view(B, T_new, self.H, self.K)
        v = F.silu(self.v(x)).view(B, T_new, self.H, self.V)
        g = -F.softplus(self.g(x)).view(B, T_new, self.H, self.K) * 0.1
        beta = torch.sigmoid(self.beta(x))
        # Detach the incoming state in training mode so the autograd graph
        # from the previous step is not retained (otherwise backward() would
        # raise "backward through the graph a second time"). In eval/decoding
        # mode we keep the graph so that stateful generation works.
        # Mirrors the fix in ops_fused.py::HybridKCHAttention.forward.
        state = self._state
        if state is not None and self.training:
            state = state.detach()
        o, self._state = naive_recurrent_kda(
            q, k, v, g, beta, scale=self.K ** -0.5,
            initial_state=state, output_final_state=True,
        )
        return self.o(o.reshape(B, T_new, self.H * self.V))


def bench_decoding(model, d_model, prefill_len, n_decode, device, repeats=3):
    """Measure per-token decoding latency after a fixed prefill.

    Returns dict with prefill_ms, mean_decode_ms_per_token, peak_mem_MB.
    """
    # Move model to device FIRST, then clear cache and reset peak memory
    # stats. The previous order (_clear_cache -> model.to(device)) meant
    # max_memory_allocated captured the model parameter allocation too,
    # inflating peak_mem_MB by the parameter count (a constant offset that
    # varies per operator and makes the comparison less fair).
    model = model.to(device).eval()
    model.reset()
    _clear_cache(device)

    # Prefill: process the whole context at once.
    x_prefill = torch.randn(1, prefill_len, d_model, device=device) * 0.1
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x_prefill)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t0) * 1e3
        else:
            t0 = time.perf_counter()
            model(x_prefill)
            prefill_ms = (time.perf_counter() - t0) * 1e3

    # Decode n_decode tokens one at a time.
    decode_times = []
    with torch.no_grad():
        for _ in range(n_decode):
            x_new = torch.randn(1, 1, d_model, device=device) * 0.1
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

    decode_times.sort()
    median_decode = decode_times[len(decode_times) // 2]
    mean_decode = sum(decode_times) / len(decode_times)

    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
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
                r = bench_decoding(model, d_model, plen, n_decode, device)
                r['op'] = name
                r['device'] = str(device)
                results.append(r)
                peak_str = 'n/a' if r['peak_mem_MB'] is None else f"{r['peak_mem_MB']:.2f}MB"
                print(f"  {name:10s}  prefill={r['prefill_ms']:8.2f}ms  "
                      f"decode/tok={r['median_decode_ms_per_token']:8.3f}ms  "
                      f"peak_mem={peak_str:>10}")
            except Exception as e:
                results.append({'op': name, 'prefill_len': plen, 'error': str(e),
                                'device': str(device)})
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
    with open('results/exp6_decoding.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved: results/exp6_decoding.json')


if __name__ == '__main__':
    main()
