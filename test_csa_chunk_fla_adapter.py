"""Regression tests for the round-1 improvements:

1. ``ops_csa`` — query-axis chunking of the lightning indexer
   (``query_chunk`` / ``naive_csa(indexer_chunk=...)``). The tiled path
   must be a PURE memory optimization: same indices (bit-compared), same
   soft weights / gradients up to batched-GEMM reduction order, no change
   to the canonical tie-break, and a bounded peak size for the per-head
   score tensor (verified by spying on ``torch.einsum`` outputs).

2. ``ops_kda_backend`` — CPU-executable conformance tests for the optional
   FLA adapter using an injected FAKE ``fla.ops.kda`` module (plus a
   class-level ``Tensor.is_cuda`` patch). Pins the "transform exactly
   once" contract flags, ``g_clamp_min`` forwarding (and the ``-inf``
   sentinel exclusion), output/state dtype normalization, the
   ``(output, final_state)`` result contract, the widened ``auto``
   fallback (RuntimeError -> warning + reference, CUDA OOM -> re-raise),
   the never-raising availability probe, and ``resolve_kda_backend``.

Run via pytest::

    pytest -q test_csa_chunk_fla_adapter.py

Or directly::

    python3 test_csa_chunk_fla_adapter.py
"""
from __future__ import annotations

import os
import sys
import types
import warnings
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ops_csa import (  # noqa: E402
    _causal_block_mask, csa_lightning_indexer, naive_csa,
)
from ops_kda import naive_recurrent_kda  # noqa: E402
from ops_kda_backend import kda_forward  # noqa: E402

import ops_kda_backend as kdb  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# ---------------------------------------------------------------------------
# 1. CSA lightning-indexer query-axis chunking
# ---------------------------------------------------------------------------
def test_csa_indexer_query_chunk_equivalence(device='cpu'):
    """Chunked indexer selection == monolithic selection, exactly for
    indices and to fp-round-off for the soft weights / gradients."""
    torch.manual_seed(23)
    B, T, HI, DI = 2, 29, 3, 16
    m = 8
    n_blocks = (T + m - 1) // m
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    w_idx = torch.randn(B, T, HI, device=device)
    cbm = _causal_block_mask(T, n_blocks, m, device)

    # Several topk values: sparse, dense-but-below, and beyond n_blocks
    # (exercises the -1 padding slots).
    for topk in (3, n_blocks, n_blocks + 4):
        idx_full, soft_full = csa_lightning_indexer(
            q_idx, k_idx, w_idx, topk, causal_block_mask=cbm,
            return_soft_weights=True, normalize_qk=True)
        for chunk in (1, 5, T - 1, T + 1):
            idx_c, soft_c = csa_lightning_indexer(
                q_idx, k_idx, w_idx, topk, causal_block_mask=cbm,
                return_soft_weights=True, normalize_qk=True,
                query_chunk=chunk)
            _ok(f'idx equal topk={topk} chunk={chunk}',
                torch.equal(idx_full, idx_c), '')
            d = (soft_full - soft_c).abs().max().item()
            _ok(f'soft_weights match topk={topk} chunk={chunk}', d < 1e-6,
                f'max_diff={d:.2e}')

    # w_idx=None branch (logits = score.sum(1)).
    idx_none = csa_lightning_indexer(q_idx, k_idx, None, 4,
                                     causal_block_mask=cbm,
                                     normalize_qk=True)
    idx_none_c = csa_lightning_indexer(q_idx, k_idx, None, 4,
                                       causal_block_mask=cbm,
                                       normalize_qk=True, query_chunk=3)
    _ok('w_idx=None branch chunk-equivalent', torch.equal(idx_none, idx_none_c),
        '')

    # topk=0 contract survives chunking (empty indices + full soft dist).
    idx_zero, soft_zero = csa_lightning_indexer(
        q_idx, k_idx, w_idx, 0, causal_block_mask=cbm,
        return_soft_weights=True, query_chunk=4)
    idx_zero_full, soft_zero_full = csa_lightning_indexer(
        q_idx, k_idx, w_idx, 0, causal_block_mask=cbm,
        return_soft_weights=True)
    _ok('topk=0 chunked shape/values', idx_zero.shape == idx_zero_full.shape
        and soft_zero.shape == soft_zero_full.shape
        and (soft_zero - soft_zero_full).abs().max().item() < 1e-6, '')

    # Exact ties (k_idx=0 -> every valid logit is exactly 0): the canonical
    # ascending-index rule must survive tiling bit-for-bit.
    k_zero = torch.zeros(B, n_blocks, DI, device=device)
    tie_full = csa_lightning_indexer(q_idx, k_zero, None, 4,
                                     causal_block_mask=cbm)
    for chunk in (1, 6, 64):
        tie_c = csa_lightning_indexer(q_idx, k_zero, None, 4,
                                      causal_block_mask=cbm, query_chunk=chunk)
        _ok(f'tie-break identical under chunk={chunk}',
            torch.equal(tie_full, tie_c), '')


def test_csa_indexer_query_chunk_memory_bound(device='cpu'):
    """The per-head score tensor must shrink to B*HI*chunk*n_blocks.

    Verified by recording every ``torch.einsum`` output produced during the
    indexer call: the first einsum materializes the per-head scores
    ``[B, HI, T_c, n_blocks]``, whose element count is the quantity the
    tile size is supposed to bound. This pins the *memory* claim, not just
    the numerical equivalence claim.
    """
    torch.manual_seed(31)
    B, T, HI, DI = 2, 40, 3, 8
    m = 8
    n_blocks = T // m
    q_idx = torch.randn(B, T, HI, DI, device=device)
    k_idx = torch.randn(B, n_blocks, DI, device=device)
    w_idx = torch.randn(B, T, HI, device=device)
    cbm = _causal_block_mask(T, n_blocks, m, device)

    def _max_score_numel(**kwargs):
        seen = []
        orig = torch.einsum

        def spy(*args, **kw):
            out = orig(*args, **kw)
            seen.append(out.numel())
            return out

        with mock.patch.object(torch, 'einsum', new=spy):
            csa_lightning_indexer(q_idx, k_idx, w_idx, 3,
                                  causal_block_mask=cbm, **kwargs)
        return max(seen) if seen else 0

    full_peak = _max_score_numel()
    tile = 5
    chunk_peak = _max_score_numel(query_chunk=tile)
    _ok('untiled peak == B*HI*T*n_blocks', full_peak == B * HI * T * n_blocks,
        f'{full_peak} vs {B * HI * T * n_blocks}')
    _ok('tiled peak == B*HI*chunk*n_blocks',
        chunk_peak == B * HI * tile * n_blocks,
        f'{chunk_peak} vs {B * HI * tile * n_blocks}')


def test_csa_indexer_query_chunk_gradients(device='cpu'):
    """Dense-indexer gradients must be tiling-invariant."""
    torch.manual_seed(37)
    B, T, HI, DI = 1, 18, 2, 8
    m = 6
    n_blocks = (T + m - 1) // m
    cbm = _causal_block_mask(T, n_blocks, m, device)

    def grads(chunk):
        q = torch.randn(B, T, HI, DI, device=device)
        k = torch.randn(B, n_blocks, DI, device=device)
        w = torch.randn(B, T, HI, device=device)
        q.requires_grad_(True)
        k.requires_grad_(True)
        w.requires_grad_(True)
        _, soft = csa_lightning_indexer(
            q, k, w, 4, causal_block_mask=cbm, return_soft_weights=True,
            query_chunk=chunk)
        soft.sum().backward()
        return q.grad, k.grad, w.grad

    g_full = grads(0)
    g_chunk = grads(4)
    diffs = [float((a - b).abs().max().item())
             for a, b in zip(g_full, g_chunk)]
    _ok('grad parity q/k/w under tiling', max(diffs) < 1e-6,
        f'max grad diffs={diffs}')


def test_csa_naive_indexer_chunk_end_to_end(device='cpu'):
    """``naive_csa(indexer_chunk=...)`` output must be unchanged."""
    torch.manual_seed(41)
    B, T, d = 2, 21, 32
    c, c_I, dc, nh, nIh, m, topk = 8, 4, 16, 2, 1, 7, 3

    def rp(out, inp):
        return (torch.randn(out, inp, device=device) * 0.2)

    args = (
        torch.randn(B, T, d, device=device) * 0.1,
        rp(c, d), rp(c, d), rp(c, d), rp(c, d),
        torch.randn(m, c, device=device) * 0.1,
        torch.randn(m, c, device=device) * 0.1,
        rp(dc, d), rp(c * nh, dc), rp(c_I * nIh, dc), rp(nIh, d),
        rp(c_I, d), rp(c_I, d),
        torch.randn(m, c_I, device=device) * 0.1,
    )
    kw = dict(m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
              sliding_window=4, sink_logits=torch.zeros(nh),
              normalize_qk=True)
    with torch.no_grad():
        o_full = naive_csa(*args, **kw)
        for chunk in (1, 5, T + 1):
            o_chunk = naive_csa(*args, **kw, indexer_chunk=chunk)
            _ok(f'naive_csa output unchanged (indexer_chunk={chunk})',
                torch.allclose(o_full, o_chunk, atol=0.0, rtol=0.0),
                f'max_diff={float((o_full - o_chunk).abs().max().item()):.2e}')

    bad_raised = False
    try:
        naive_csa(*args, **{**kw, 'indexer_chunk': -1})
    except ValueError:
        bad_raised = True
    _ok('naive_csa rejects negative indexer_chunk', bad_raised, '')

    q_small = torch.randn(B, 1, 1, d, device=device)
    k_small = torch.randn(B, 2, d, device=device)
    for bad_value in (-1, True):
        bad_raised = False
        try:
            csa_lightning_indexer(q_small, k_small, None, 1,
                                  query_chunk=bad_value)
        except ValueError:
            bad_raised = True
        _ok(f'csa_lightning_indexer rejects query_chunk={bad_value!r}',
            bad_raised, '')


# ---------------------------------------------------------------------------
# 2. FLA KDA adapter conformance (fake module, CPU)
# ---------------------------------------------------------------------------
def _install_fake_fla(chunk_fn, fused_fn):
    """Install fake ``fla`` / ``fla.ops`` / ``fla.ops.kda`` modules."""
    fake_fla = types.ModuleType('fla')
    fake_ops = types.ModuleType('fla.ops')
    fake_kda = types.ModuleType('fla.ops.kda')
    fake_kda.chunk_kda = chunk_fn
    fake_kda.fused_recurrent_kda = fused_fn
    sys.modules['fla'] = fake_fla
    sys.modules['fla.ops'] = fake_ops
    sys.modules['fla.ops.kda'] = fake_kda
    kdb._FLA_OPS = None


def test_kda_fla_backend_adapter_contract(device='cpu'):
    """Adapter contract, API-drift handling, and auto-fallback policy.

    See the module docstring. The fake kernels delegate to the reference
    operators, so any value mismatch is a genuine adapter bug.
    """
    saved_ops = kdb._FLA_OPS
    saved_modules = {
        name: sys.modules.get(name)
        for name in ('fla', 'fla.ops', 'fla.ops.kda')
    }
    calls: list[dict] = []

    def _delegate(q, k, v, g, beta, scale, initial_state,
                  output_final_state):
        return naive_recurrent_kda(
            q, k, v, g, beta, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state)

    def make_full_sig():
        unset = object()  # distinguishes "kwarg forwarded" from the default

        def fn(q, k, v, g, beta, scale=None, initial_state=None,
               output_final_state=False,
               use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
               use_beta_sigmoid_in_kernel=False, chunk_size=64,
               g_clamp_min=unset):
            calls.append({
                'use_qk_l2norm_in_kernel': use_qk_l2norm_in_kernel,
                'use_gate_in_kernel': use_gate_in_kernel,
                'use_beta_sigmoid_in_kernel': use_beta_sigmoid_in_kernel,
                'output_final_state': output_final_state,
                'initial_state': initial_state,
                'g_clamp_min': (None if g_clamp_min is unset
                                else g_clamp_min),
                'chunk_size': chunk_size,
            })
            return _delegate(q, k, v, g, beta, scale, initial_state,
                             output_final_state)
        return fn

    torch.manual_seed(11)
    B, T, H, K, V = 1, 6, 2, 4, 4
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1)
    v = torch.randn(B, T, H, V, device=device) * 0.1
    g = -torch.rand(B, T, H, K, device=device) * 0.05
    beta = torch.rand(B, T, H, device=device) * 0.2

    cuda_patch = mock.patch.object(
        torch.Tensor, 'is_cuda', property(lambda self: True))
    try:
        fn = make_full_sig()
        _install_fake_fla(fn, fn)
        with cuda_patch:
            _ok('fake FLA importable -> fla_available() True',
                kdb.fla_available() is True, '')
            o_fla, s_fla = kda_forward(
                q, k, v, g, beta, output_final_state=True,
                backend='fla', use_chunk=False)
            o_ref, s_ref = kda_forward(
                q, k, v, g, beta, output_final_state=True,
                backend='reference', use_chunk=False)
        _ok('fla dispatch equals reference (fake delegates)',
            torch.equal(o_fla, o_ref) and torch.equal(s_fla, s_ref), '')
        kw = calls[-1] if calls else {}
        _ok('transform-once flags + g_clamp_min forwarded',
            kw.get('use_qk_l2norm_in_kernel') is False
            and kw.get('use_gate_in_kernel') is False
            and kw.get('use_beta_sigmoid_in_kernel') is False
            and kw.get('output_final_state') is True
            and kw.get('initial_state') is None
            and kw.get('g_clamp_min') == -10.0, str(kw))

        calls.clear()
        with cuda_patch:
            kda_forward(q, k, v, g, beta, backend='fla', use_chunk=False,
                        g_clamp_min=float('-inf'))
        _ok('g_clamp_min=-inf sentinel not forwarded',
            bool(calls) and calls[-1]['g_clamp_min'] is None,
            str(calls[-1]) if calls else 'no call recorded')

        # Output/state dtype normalization.
        def half_kernel(q, k, v, g, beta, scale=None, initial_state=None,
                        output_final_state=False):
            out = _delegate(q, k, v, g, beta, scale, initial_state, True)[0]
            state = torch.zeros(1, 2, 4, 4, dtype=torch.float16,
                                device=v.device)
            return out.to(torch.float16), state
        _install_fake_fla(half_kernel, half_kernel)
        with cuda_patch:
            o_h, s_h = kda_forward(q, k, v, g, beta,
                                   output_final_state=True,
                                   backend='fla', use_chunk=False)
        _ok('fp32-in: output kept in v.dtype, state normalized to fp32',
            o_h.dtype == v.dtype and s_h.dtype == torch.float32,
            f'o={o_h.dtype}, s={s_h.dtype}')

        def half_kernel_fp64(q, k, v, g, beta, scale=None,
                             initial_state=None, output_final_state=False):
            out = _delegate(q, k, v, g, beta, scale, initial_state, True)[0]
            state = torch.zeros(1, 2, 4, 4, dtype=torch.float32,
                                device=v.device)
            return out.to(torch.float16), state
        _install_fake_fla(half_kernel_fp64, half_kernel_fp64)
        with cuda_patch:
            _, s64 = kda_forward(q.double(), k.double(), v.double(),
                                 g.double(), beta.double(),
                                 output_final_state=True,
                                 backend='fla', use_chunk=False)
        _ok('fp64-in: state normalized to fp64', s64.dtype == torch.float64,
            f's={s64.dtype}')

        # Dropped contract flags must warn, never silently.
        def no_flags_kernel(q, k, v, g, beta, scale=None, initial_state=None,
                            output_final_state=False):
            return _delegate(q, k, v, g, beta, scale, initial_state,
                             output_final_state)
        _install_fake_fla(no_flags_kernel, no_flags_kernel)
        with cuda_patch, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            kda_forward(q, k, v, g, beta, output_final_state=True,
                        backend='fla', use_chunk=False)
        _ok('dropped contract flags emit RuntimeWarning',
            any(issubclass(x.category, RuntimeWarning)
                and 'does not accept' in str(x.message) for x in w),
            f'{len(w)} warning(s)')

        # VAR_KEYWORD signature: everything forwarded, nothing warned.
        def var_kw_kernel(q, k, v, g, beta, **kwargs):
            calls.append({k2: kwargs.get(k2) for k2 in (
                'scale', 'initial_state', 'output_final_state',
                'use_qk_l2norm_in_kernel', 'chunk_size', 'g_clamp_min')})
            return _delegate(q, k, v, g, beta, kwargs.get('scale'),
                             kwargs.get('initial_state'),
                             kwargs.get('output_final_state', False))
        _install_fake_fla(var_kw_kernel, var_kw_kernel)
        calls.clear()
        with cuda_patch, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            kda_forward(q, k, v, g, beta, output_final_state=True,
                        backend='fla', use_chunk=False)
        _ok('VAR_KEYWORD signature passes all kwargs without warning',
            bool(calls) and calls[-1]['use_qk_l2norm_in_kernel'] is False
            and calls[-1]['g_clamp_min'] == -10.0
            and not any(issubclass(x.category, RuntimeWarning) for x in w),
            f'{len(w)} warning(s)')

        # Missing (output, final_state) tuple must raise.
        def bad_result(q, k, v, g, beta, **kwargs):
            return v.clone()
        _install_fake_fla(bad_result, bad_result)
        bad_raised = False
        with cuda_patch:
            try:
                kda_forward(q, k, v, g, beta, output_final_state=True,
                            backend='fla', use_chunk=False)
            except RuntimeError:
                bad_raised = True
        _ok('missing final_state result raises RuntimeError', bad_raised, '')

        # auto fallback on a runtime kernel failure; explicit fla raises.
        def broken_kernel(q, k, v, g, beta, **kwargs):
            raise RuntimeError('triton launch failed')
        _install_fake_fla(broken_kernel, broken_kernel)
        with cuda_patch, warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            o_fb, s_fb = kda_forward(q, k, v, g, beta,
                                     output_final_state=True, backend='auto',
                                     use_chunk=False)
            warned = any(issubclass(x.category, RuntimeWarning)
                         and 'auto' in str(x.message) for x in w)
        _ok('auto falls back on FLA RuntimeError (with warning)',
            warned and torch.equal(o_fb, o_ref) and torch.equal(s_fb, s_ref),
            f'warned={warned}')

        explicit_raised = False
        with cuda_patch:
            try:
                kda_forward(q, k, v, g, beta, output_final_state=True,
                            backend='fla', use_chunk=False)
            except RuntimeError as exc:
                explicit_raised = 'triton launch failed' in str(exc)
        _ok("explicit backend='fla' still raises (no silent fallback)",
            explicit_raised, '')

        # CUDA OOM propagates under auto (resource failure, not drift).
        def oom_kernel(q, k, v, g, beta, **kwargs):
            raise RuntimeError(
                'CUDA out of memory. Tried to allocate 2.00 GiB')
        _install_fake_fla(oom_kernel, oom_kernel)
        oom_raised = False
        with cuda_patch, warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            try:
                kda_forward(q, k, v, g, beta, output_final_state=True,
                            backend='auto', use_chunk=False)
            except RuntimeError as exc:
                oom_raised = 'out of memory' in str(exc)
        _ok('auto re-raises CUDA OOM instead of falling back', oom_raised, '')

        # Never-raising availability probe.
        with mock.patch.object(kdb, '_load_fla_ops',
                               side_effect=RuntimeError('triton mismatch')):
            probe_ok = kdb.fla_available() is False
        _ok('fla_available() False on non-ImportError import failure',
            probe_ok, '')

        # Provenance resolver.
        with mock.patch.object(torch.cuda, 'is_available',
                               return_value=False), \
                mock.patch.object(kdb, 'fla_available', return_value=True):
            resolve_cpu = kdb.resolve_kda_backend('auto')
        with mock.patch.object(torch.cuda, 'is_available',
                               return_value=True), \
                mock.patch.object(kdb, 'fla_available', return_value=True):
            resolve_cuda = kdb.resolve_kda_backend('auto')
        invalid_raised = False
        try:
            kdb.resolve_kda_backend('not-a-backend')
        except ValueError:
            invalid_raised = True
        _ok('resolve_kda_backend provenance resolution',
            resolve_cpu == 'reference' and resolve_cuda == 'fla'
            and kdb.resolve_kda_backend('reference') == 'reference'
            and invalid_raised,
            f'cpu={resolve_cpu}, cuda+fla={resolve_cuda}')
    finally:
        kdb._FLA_OPS = saved_ops
        for name, mod in saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_hybrid_config_indexer_chunk_wiring():
    """``HybridConfig.csa_indexer_chunk`` is plumbed through and inert.

    Two identical hybrid stacks (one with the tile enabled) must produce
    bit-identical outputs, and a negative config value must be rejected at
    construction time rather than silently running the untiled path.
    """
    from ops_fused import HybridConfig, HybridAttentionStack

    torch.manual_seed(43)
    common = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2, head_dim_k=8, head_dim_v=8,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=8, csa_nIh=1,
        csa_cI=4, csa_sliding_window=3, hca_m2=8, hca_nh=2, hca_c=8,
        hca_dc=8, hca_sliding_window=3, n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=None, hca_rope_dim=None,
    )
    cfg_off = HybridConfig(**common, csa_indexer_chunk=0)
    cfg_on = HybridConfig(**common, csa_indexer_chunk=5)
    model_off = HybridAttentionStack(cfg_off, total_layers=3)
    model_on = HybridAttentionStack(cfg_on, total_layers=3)
    model_on.load_state_dict(model_off.state_dict())
    model_off.eval()
    model_on.eval()
    x = torch.randn(2, 11, 32, device='cpu') * 0.1
    with torch.no_grad():
        o_off = model_off(x)
        o_on = model_on(x)
    _ok('hybrid stack identical with csa_indexer_chunk on/off',
        torch.equal(o_off, o_on),
        f'max_diff={float((o_off - o_on).abs().max().item()):.2e}')

    bad_raised = False
    try:
        HybridConfig(**common, csa_indexer_chunk=-3)
    except ValueError:
        bad_raised = True
    _ok('HybridConfig rejects negative csa_indexer_chunk', bad_raised, '')


if __name__ == '__main__':
    test_csa_indexer_query_chunk_equivalence()
    test_csa_indexer_query_chunk_memory_bound()
    test_csa_indexer_query_chunk_gradients()
    test_csa_naive_indexer_chunk_end_to_end()
    test_hybrid_config_indexer_chunk_wiring()
    test_kda_fla_backend_adapter_contract()
    print('all CSA-chunk / FLA-adapter tests passed')
