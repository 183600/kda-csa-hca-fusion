"""Cross-mode regression oracle for module-level tensor caches.

Why this file exists
--------------------
``torch.inference_mode()`` is the recommended way to run inference. Tensors
created inside it are *inference tensors*: they have no version counter and
cannot participate in autograd. Several operators cache constant tensors in
module-level dictionaries (causal masks, triangular constants, decoding
scratch). If a cache is populated during an inference warm-up, a later
grad-enabled or plain-``no_grad`` call that reads the cached inference tensor
fails with one of:

* ``Inference tensors cannot be saved for backward`` (mask used by
  ``masked_fill`` in a grad-enabled forward), or
* ``Inplace update to inference tensor outside InferenceMode`` (decoding
  position scratch written by a later ``no_grad`` call).

The oracle here is deliberately cross-mode: it warms every cache inside
``torch.inference_mode()`` (using bfloat16 inputs where an operator accepts
them, matching the deployment path), then asserts that the next grad-enabled
call runs, produces finite gradients, and matches a fresh-cache reference.

Every cached tensor must therefore be built under
``ops_cache.normal_tensor_scope()``. The fix only changes tensor ownership
metadata; values and cache hits stay identical (hits remain zero-copy).

Run via pytest::

    pytest -q test_cache_inference_mode.py

Or directly::

    python3 test_cache_inference_mode.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import ops_csa  # noqa: E402
import ops_decoding_cache as dc  # noqa: E402
import ops_kda  # noqa: E402
import ops_mamba  # noqa: E402
from ops_csa import naive_csa  # noqa: E402
from ops_hca import naive_hca  # noqa: E402
from ops_kda import naive_chunk_kda  # noqa: E402
from ops_mamba import chunked_mamba2_scan, quadratic_mamba2_scan  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _clear_all_caches():
    for cache in (
        ops_csa._SLIDING_WINDOW_MASK_CACHE,
        ops_csa._CAUSAL_BLOCK_MASK_CACHE,
        ops_csa._CAUSAL_BLOCK_MASK_INVALID_CACHE,
        ops_mamba._CAUSAL_CHUNK_MASK_CACHE,
        ops_mamba._NONCAUSAL_CHUNK_MASK_CACHE,
        ops_kda._CHUNK_CONST_CACHE,
        dc._BLOCK_VALID_MEMO,
        dc._BOOL_ONES_MEMO,
        dc._POS_SCRATCH,
    ):
        cache.clear()


def _all_ordinary(tensors):
    return all(not t.is_inference() for t in tensors)


# ---------------------------------------------------------------------------
# 1. Every mask/constant cache is ordinary after an inference warm-up
# ---------------------------------------------------------------------------
def test_mask_and_constant_caches_are_ordinary(device='cpu'):
    _clear_all_caches()
    with torch.inference_mode():
        sw = ops_csa._sliding_window_valid_mask(17, 5, device)
        cbm = ops_csa._causal_block_mask(17, 4, 5, device)
        cbm_inv = ops_csa._causal_block_mask_invalid(17, 4, 5, device)
        causal = ops_mamba._causal_chunk_mask(17, device)
        noncausal = ops_mamba._noncausal_chunk_mask(17, device)
        eye, upper = ops_kda._chunk_constants(17, torch.float32, device)
        blk = dc._block_valid_mask(4, 2, device)
        ones = dc._bool_ones(1, 5, device)
        pos = dc._pos_tensor(7, device)
    named = {
        'csa sliding-window mask': sw,
        'csa causal block mask': cbm,
        'csa invalid block mask': cbm_inv,
        'mamba causal chunk mask': causal,
        'mamba noncausal chunk mask': noncausal,
        'kda chunk eye': eye,
        'kda chunk upper mask': upper,
        'decode block mask': blk,
        'decode ones mask': ones,
        'decode position scratch': pos,
    }
    for name, tensor in named.items():
        _ok(f'{name} is ordinary after inference warm-up',
            not tensor.is_inference(), f'dtype={tensor.dtype}')


# ---------------------------------------------------------------------------
# 2. CSA / HCA: bf16 inference warm-up -> fp32 grad-enabled forward
# ---------------------------------------------------------------------------
def _csa_case(device, dtype):
    torch.manual_seed(11)
    B, T, d = 1, 17, 32
    c, c_I, m, nh, nIh, dc = 8, 4, 4, 2, 1, 8
    topk, win, rope_dim = 100, 4, 8

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.1

    args = (
        rand(B, T, d), rand(c, d), rand(c, d), rand(c, d), rand(c, d),
        rand(m, c), rand(m, c), rand(dc, d), rand(c * nh, dc),
        rand(c_I * nIh, dc), rand(nIh, d), rand(c_I, d), rand(c_I, d),
        rand(m, c_I),
    )
    kwargs = dict(
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=win, sink_logits=None, use_ste=False,
        normalize_qk=True, rope_dim=rope_dim,
    )
    return args, kwargs


def _hca_case(device, dtype):
    torch.manual_seed(12)
    B, T, d = 1, 17, 32
    c, m2, nh, dc = 8, 4, 2, 8
    win, rope_dim = 4, 8

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.1

    args = (
        rand(B, T, d), rand(c, d), rand(c, d), rand(m2, c),
        rand(dc, d), rand(c * nh, dc),
    )
    kwargs = dict(
        m2=m2, nh=nh, c=c, dc=dc, sliding_window=win,
        sink_logits=None, rope_dim=rope_dim,
    )
    return args, kwargs


def test_csa_hca_train_after_bf16_inference_warmup(device='cpu'):
    for name, fn, case in (
        ('CSA', naive_csa, _csa_case),
        ('HCA', naive_hca, _hca_case),
    ):
        _clear_all_caches()
        args_bf16, kwargs = case(device, torch.bfloat16)
        # Warm every mask/RoPE cache inside inference mode with the deployment
        # dtype. The masks themselves are dtype-independent.
        with torch.inference_mode():
            fn(*args_bf16, **kwargs)

        _clear_all_caches()
        args_fp32, _ = case(device, torch.float32)
        # Fresh-cache reference (no inference warm-up).
        ref = fn(*args_fp32, **kwargs)

        # Same call again, now with gradients, after the warm-up repopulated
        # the caches with ordinary tensors.
        _clear_all_caches()
        with torch.inference_mode():
            fn(*args_bf16, **kwargs)
        grad_args = tuple(
            a.detach().clone().requires_grad_(i == 0)
            for i, a in enumerate(args_fp32)
        )
        out = fn(*grad_args, **kwargs)
        out.square().sum().backward()
        grad = grad_args[0].grad
        _ok(f'{name}: grad-enabled forward runs after bf16 inference warm-up',
            bool(torch.isfinite(out).all()),
            f'out_scale={out.abs().max().item():.3e}')
        _ok(f'{name}: backward produces finite gradients',
            grad is not None and bool(torch.isfinite(grad).all()),
            f'grad_norm={None if grad is None else grad.norm().item():.3e}')
        _ok(f'{name}: warmed output matches fresh-cache reference',
            torch.allclose(out, ref, atol=1e-6, rtol=1e-5),
            f'max_diff={(out - ref).abs().max().item():.2e}')


# ---------------------------------------------------------------------------
# 3. Mamba-2 chunked / quadratic scans: same cross-mode contract
# ---------------------------------------------------------------------------
def _mamba_inputs(device, dtype, T=34):
    torch.manual_seed(21)
    B, H, D, N = 1, 2, 8, 4
    x = torch.randn(B, T, H, D, device=device, dtype=dtype) * 0.1
    dt = torch.nn.functional.softplus(
        torch.randn(B, T, H, device=device, dtype=dtype) * 0.3)
    Bm = torch.randn(B, T, 1, N, device=device, dtype=dtype)
    C = torch.randn(B, T, 1, N, device=device, dtype=dtype)
    A = -torch.exp(torch.randn(H, device=device, dtype=dtype))
    Dv = torch.randn(H, device=device, dtype=dtype)
    return x, dt, Bm, C, A, Dv


def test_mamba_scan_train_after_bf16_inference_warmup(device='cpu'):
    for name, fn, extra in (
        ('chunked', chunked_mamba2_scan, {'chunk_size': 16}),
        ('quadratic', quadratic_mamba2_scan, {}),
    ):
        _clear_all_caches()
        warm = _mamba_inputs(device, torch.bfloat16)
        with torch.inference_mode():
            fn(*warm, output_final_state=True, **extra)

        _clear_all_caches()
        fp32 = _mamba_inputs(device, torch.float32)
        ref_out, ref_state = fn(*fp32, output_final_state=True, **extra)

        _clear_all_caches()
        with torch.inference_mode():
            fn(*warm, output_final_state=True, **extra)
        grad_inputs = tuple(
            t.detach().clone().requires_grad_(i == 0)
            for i, t in enumerate(fp32)
        )
        out, state = fn(*grad_inputs, output_final_state=True, **extra)
        (out.square().sum() + state.square().sum()).backward()
        _ok(f'Mamba-2 {name}: grad-enabled forward runs after bf16 warm-up',
            bool(torch.isfinite(out).all() and torch.isfinite(state).all()))
        _ok(f'Mamba-2 {name}: backward produces finite gradients',
            grad_inputs[0].grad is not None
            and bool(torch.isfinite(grad_inputs[0].grad).all()))
        _ok(f'Mamba-2 {name}: warmed output/state match fresh reference',
            torch.allclose(out, ref_out, atol=1e-6, rtol=1e-5)
            and torch.allclose(state, ref_state, atol=1e-6, rtol=1e-5),
            f'out_diff={(out - ref_out).abs().max().item():.2e}, '
            f'state_diff={(state - ref_state).abs().max().item():.2e}')


# ---------------------------------------------------------------------------
# 4. KDA chunk constants: ordinary under inference mode
# ---------------------------------------------------------------------------
def test_kda_chunk_constants_and_forward_cross_mode(device='cpu'):
    _clear_all_caches()
    with torch.inference_mode():
        naive_chunk_kda(
            *[t.to(torch.bfloat16) for t in _kda_inputs(device)],
            chunk_size=16,
        )
    eye, upper = ops_kda._CHUNK_CONST_CACHE[(16, torch.float32, device)]
    _ok('KDA chunk constants are ordinary after inference warm-up',
        _all_ordinary((eye, upper)))
    args = _kda_inputs(device)
    args[0] = args[0].requires_grad_(True)
    out, state = naive_chunk_kda(*args, chunk_size=16, output_final_state=True)
    (out.square().sum() + state.square().sum()).backward()
    _ok('KDA chunk backward runs after inference warm-up',
        args[0].grad is not None
        and bool(torch.isfinite(args[0].grad).all()))


def _kda_inputs(device):
    torch.manual_seed(31)
    B, T, H, K, V = 1, 34, 2, 8, 8
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device=device), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2
    return [q, k, v, g, beta]


# ---------------------------------------------------------------------------
# 5. Decoding scratch: inference_mode -> plain no_grad must not crash
# ---------------------------------------------------------------------------
def test_decoding_scratch_survives_mode_switch(device='cpu'):
    dc._POS_SCRATCH.clear()
    with torch.inference_mode():
        p_inf = dc._pos_tensor(7, device)
        _ok('decoding scratch is ordinary after inference warm-up',
            not p_inf.is_inference(), f'value={p_inf.item()}')
        _ok('decoding scratch holds the warmed value',
            p_inf.item() == 7, f'value={p_inf.item()}')
    # Plain no_grad is a different mode from inference_mode; a cached
    # inference tensor would raise on ``fill_`` here.
    with torch.no_grad():
        p_nograd = dc._pos_tensor(9, device)
        _ok('decoding scratch survives inference_mode -> no_grad',
            not p_nograd.is_inference() and p_nograd.item() == 9,
            f'value={p_nograd.item()}')
    _ok('decoding scratch is reused, not reallocated per call',
        p_nograd.data_ptr() == p_inf.data_ptr(),
        'same storage')


# ---------------------------------------------------------------------------
# 6. Cache hits stay zero-copy (the ownership fix must not clone)
# ---------------------------------------------------------------------------
def test_cache_hits_are_zero_copy(device='cpu'):
    _clear_all_caches()
    with torch.inference_mode():
        first = ops_csa._causal_block_mask_invalid(33, 5, 8, device)
    second = ops_csa._causal_block_mask_invalid(33, 5, 8, device)
    _ok('csa invalid-mask cache hit returns the same ordinary tensor',
        second is first and not second.is_inference(),
        f'data_ptr={second.data_ptr()}')

    _clear_all_caches()
    with torch.inference_mode():
        first = ops_mamba._noncausal_chunk_mask(32, device)
    second = ops_mamba._noncausal_chunk_mask(32, device)
    _ok('mamba noncausal-mask cache hit returns the same ordinary tensor',
        second is first and not second.is_inference(),
        f'data_ptr={second.data_ptr()}')

    _clear_all_caches()
    with torch.inference_mode():
        first = dc._pos_tensor(3, device)
    with torch.no_grad():
        second = dc._pos_tensor(4, device)
    _ok('decoding scratch hit reuses storage',
        second.data_ptr() == first.data_ptr(),
        f'data_ptr={second.data_ptr()}')


# ---------------------------------------------------------------------------
# 7. Informational benchmark: hit vs rebuild (not a timing gate)
# ---------------------------------------------------------------------------
def benchmark_cache_hit_vs_rebuild(device='cpu', BT=512, reps=200):
    """Print the median cost of a cache hit vs rebuilding the mask.

    This is informational: the correctness contract is the identity check in
    ``test_cache_hits_are_zero_copy``. The benchmark documents that the
    ownership fix preserves the original performance reason for the caches
    (cache hits do not allocate a mask or a clone).
    """
    _clear_all_caches()
    ops_mamba._causal_chunk_mask(BT, device)  # populate

    def _time(fn):
        for _ in range(5):
            fn()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        return samples[len(samples) // 2] * 1e6

    hit_us = _time(lambda: ops_mamba._causal_chunk_mask(BT, device))
    rebuild_us = _time(
        lambda: torch.tril(torch.ones(BT, BT, dtype=torch.bool,
                                      device=device)))
    print(f'  cache hit: {hit_us:.2f} us   rebuild {BT}x{BT}: '
          f'{rebuild_us:.2f} us   speedup {rebuild_us / hit_us:.1f}x')


if __name__ == '__main__':
    test_mask_and_constant_caches_are_ordinary()
    test_csa_hca_train_after_bf16_inference_warmup()
    test_mamba_scan_train_after_bf16_inference_warmup()
    test_kda_chunk_constants_and_forward_cross_mode()
    test_decoding_scratch_survives_mode_switch()
    test_cache_hits_are_zero_copy()
    benchmark_cache_hit_vs_rebuild()
    print('all cross-mode cache regression tests passed')
