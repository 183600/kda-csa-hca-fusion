"""Regression tests for the shape-polymorphic RMSNorm reference (``ops_norm``).

Pins the primitive's contract independently of any attention call site:

1. arbitrary leading axes and arbitrary numbers of trailing normalized axes;
2. exact agreement with the inline ``x * rsqrt(mean(x**2) + eps)`` formula
   used by the fused operators (including bit-exact in-place equivalence);
3. dtype preservation (no hidden upcast), empty leading axes, non-contiguous
   inputs, and zero-vector handling;
4. explicit validation of ``eps``, ``normalized_shape``, and affine params;
5. autograd support via ``torch.autograd.gradcheck``;
6. the integration contract: ``ops_csa._qk_normalize('rms')`` delegates to
   this primitive bit-exactly.

Run via pytest::

    pytest -q test_ops_norm.py

Or directly::

    python3 test_ops_norm.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_norm import RMS_NORM_EPS, rms_norm, rms_norm_  # noqa: E402
from ops_csa import _qk_normalize, _qk_normalize_  # noqa: E402

EPS = RMS_NORM_EPS


def _check(results, name, cond, detail=''):
    """Record a PASS/FAIL row (same ``status`` contract as run_correctness).

    Returning a list of these dicts lets the repository ``conftest.py``
    convert any FAIL row into a real ``AssertionError`` under pytest, while
    ``main()`` below can still print a standalone summary.
    """
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    results.append({'name': name, 'status': status, 'detail': detail})
    return bool(cond)


def test_rmsnorm_shape_polymorphism(device='cpu'):
    results = []
    _check(results, 'RMS_NORM_EPS is 1e-6', EPS == 1e-6, f'EPS={EPS!r}')

    torch.manual_seed(2026)
    for shape in [(7,), (5, 7), (3, 5, 7), (2, 3, 5, 7)]:
        x = torch.randn(*shape, dtype=torch.float64, device=device) * 3.0
        y = rms_norm(x)
        ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        sqrt_ref = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        _check(results, f'rms_norm exact for shape {shape}', torch.equal(y, ref),
               f'max_diff={(y - ref).abs().max().item():.2e}')
        _check(results, f'rms_norm matches x/sqrt(mean(x^2)+eps) for {shape}',
               torch.allclose(y, sqrt_ref, atol=1e-12, rtol=1e-12), '')

    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, device=device)
    y_4d = rms_norm(x)
    y_3d = rms_norm(x.reshape(6, 5, 7))
    _check(results, 'rms_norm is invariant to regrouping leading axes',
           torch.equal(y_4d.reshape(6, 5, 7), y_3d), '')

    empty = torch.empty(0, 5, 7, dtype=torch.float32, device=device)
    y_empty = rms_norm(empty)
    _check(results, 'rms_norm passes empty leading axes through',
           y_empty.shape == empty.shape and y_empty.dtype == empty.dtype, '')

    base = torch.randn(5, 3, 7, dtype=torch.float64, device=device)
    nc = base.transpose(0, 1)
    _check(results, 'rms_norm handles non-contiguous input',
           not nc.is_contiguous() and torch.equal(rms_norm(nc),
                                                  rms_norm(nc.contiguous())), '')

    x = torch.randn(2, 3, 4, 5, dtype=torch.float64, device=device) * 2.0
    w = torch.randn(4, 5, dtype=torch.float64, device=device)
    b = torch.randn(4, 5, dtype=torch.float64, device=device)
    y = rms_norm(x, weight=w, bias=b, normalized_shape=(4, 5))
    ms = x.pow(2).mean(dim=(-2, -1), keepdim=True)
    ref = x * torch.rsqrt(ms + EPS) * w + b
    _check(results, 'rms_norm normalizes multiple trailing axes',
           torch.equal(y, ref), f'max_diff={(y - ref).abs().max().item():.2e}')
    x4 = torch.randn(2, 3, 4, dtype=torch.float64, device=device)
    _check(results, 'rms_norm promotes an int normalized_shape to a 1-tuple',
           torch.equal(rms_norm(x4, normalized_shape=4),
                       rms_norm(x4, normalized_shape=(4,))), '')
    return results


def test_rmsnorm_inplace_and_affine(device='cpu'):
    results = []
    a = torch.randn(3, 4, 6, dtype=torch.float64, device=device)
    b_t = a.clone()
    out = rms_norm(a)
    ret = rms_norm_(b_t)
    _check(results, 'rms_norm_ matches rms_norm bit-exactly and returns x',
           ret is b_t and torch.equal(out, b_t), '')

    w6 = torch.randn(6, dtype=torch.float64, device=device)
    b6 = torch.randn(6, dtype=torch.float64, device=device)
    b_t2 = a.clone()
    ret2 = rms_norm_(b_t2, weight=w6, bias=b6)
    _check(results, 'rms_norm_ applies affine parameters like rms_norm',
           ret2 is b_t2
           and torch.equal(ret2, rms_norm(a, weight=w6, bias=b6)), '')

    x = torch.randn(2, 3, 5, 7, dtype=torch.float64, device=device)
    w1 = torch.randn(7, dtype=torch.float64, device=device)
    ref_w = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)) * w1
    _check(results, 'rms_norm applies a [D] weight per row',
           torch.equal(rms_norm(x, weight=w1), ref_w), '')
    return results


def test_rmsnorm_dtype_and_autograd(device='cpu'):
    results = []
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(2, 5, 7, dtype=dtype, device=device)
        y = rms_norm(x)
        ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        _check(results, f'rms_norm preserves dtype {dtype}',
               y.dtype == dtype and torch.equal(y, ref), '')

    xb = torch.randn(2, 5, 7, dtype=torch.bfloat16, device=device)
    yb = rms_norm(xb)
    refb = (xb.float() * torch.rsqrt(
        xb.float().pow(2).mean(dim=-1, keepdim=True) + EPS)).bfloat16()
    _check(results, 'rms_norm preserves bfloat16 and tracks the fp32 reference',
           yb.dtype == torch.bfloat16
           and torch.allclose(yb.float(), refb.float(), atol=1e-2, rtol=1e-2),
           '')

    z = torch.zeros(2, 3, 4, device=device)
    _check(results, 'rms_norm maps the zero vector to zero',
           bool((rms_norm(z) == 0).all()), '')

    x = torch.randn(2, 3, 4, dtype=torch.float64, device=device,
                    requires_grad=True)
    w4 = torch.randn(4, dtype=torch.float64, device=device, requires_grad=True)
    b4 = torch.randn(4, dtype=torch.float64, device=device, requires_grad=True)
    gradcheck_ok = torch.autograd.gradcheck(
        lambda t: rms_norm(t, eps=EPS, weight=w4, bias=b4),
        (x,), atol=1e-6, rtol=1e-3)
    y_sum = rms_norm(x, eps=EPS, weight=w4, bias=b4).sum()
    gx, gw, gb = torch.autograd.grad(y_sum, (x, w4, b4))
    _check(results, 'rms_norm passes torch.autograd.gradcheck (fp64, affine)',
           gradcheck_ok, '')
    _check(results, 'rms_norm gradients finite and non-zero for x/weight/bias',
           all(bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0)
               for g in (gx, gw, gb)), '')
    return results


def test_rmsnorm_validation(device='cpu'):
    results = []

    def _raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    xf = torch.randn(2, 3, 4, device=device)
    wf = torch.randn(4, device=device)
    checks = [
        ('negative eps rejected', lambda: rms_norm(xf, eps=-1.0), ValueError),
        ('NaN eps rejected', lambda: rms_norm(xf, eps=float('nan')), ValueError),
        ('non-numeric eps rejected', lambda: rms_norm(xf, eps='x'), TypeError),
        ('numeric-string eps rejected', lambda: rms_norm(xf, eps='1e-6'),
         TypeError),
        ('bool eps rejected', lambda: rms_norm(xf, eps=True), TypeError),
        ('oversized int eps rejected', lambda: rms_norm(xf, eps=10 ** 400),
         ValueError),
        ('normalized_shape mismatch rejected',
         lambda: rms_norm(xf, normalized_shape=(4, 3)), ValueError),
        ('empty normalized_shape rejected',
         lambda: rms_norm(xf, normalized_shape=()), ValueError),
        ('wrong weight shape rejected',
         lambda: rms_norm(xf, weight=torch.randn(3, 4, device=device)),
         ValueError),
        ('weight dtype mismatch rejected',
         lambda: rms_norm(xf, weight=wf.double()), ValueError),
        ('integer input rejected',
         lambda: rms_norm(torch.ones(2, 3, dtype=torch.int64, device=device)),
         TypeError),
        ('zero-size normalized axis rejected',
         lambda: rms_norm(torch.empty(2, 0, device=device)), ValueError),
    ]
    for name, fn, exc in checks:
        _check(results, f'rms_norm validation: {name}', _raises(fn, exc), '')

    grad_x = torch.randn(2, 3, device=device, requires_grad=True)
    _check(results, 'rms_norm_ refuses autograd-enabled grad tensors',
           _raises(lambda: rms_norm_(grad_x), RuntimeError), '')
    return results


def test_rmsnorm_overflow_finite_inputs(device='cpu'):
    """Finite inputs at the dtype limit must not collapse to all zeros.

    ``x.pow(2)`` overflows to inf at ``finfo.max``; the historical
    ``rsqrt(inf) = 0`` then multiplied the whole row away.  The fallback
    keeps the normalized direction finite while leaving ordinary rows
    bit-identical (pinned by the other tests in this file).
    """
    results = []
    sqrt2 = 2.0 ** 0.5
    tolerances = {torch.float16: 2e-3, torch.bfloat16: 2e-3,
                  torch.float32: 1e-6, torch.float64: 1e-12}
    for dtype, tol in tolerances.items():
        limit = torch.finfo(dtype).max
        x = torch.tensor([[limit, 0.0], [limit, limit],
                          [-limit, limit]], dtype=dtype, device=device)
        y = rms_norm(x)
        expected = torch.tensor([[sqrt2, 0.0], [1.0, 1.0], [-1.0, 1.0]],
                                dtype=dtype, device=device)
        y_inplace = x.clone()
        rms_norm_(y_inplace)
        _check(results, f'rms_norm finite at {dtype} limit',
               y.dtype == dtype and bool(torch.isfinite(y).all())
               and torch.allclose(y, expected, atol=tol, rtol=tol),
               f'{y.tolist()}')
        _check(results, f'rms_norm_ matches rms_norm at {dtype} limit',
               torch.equal(y_inplace, y), '')

        normal_rows = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=dtype,
                                   device=device)
        mixed = torch.cat(
            [x[:1], normal_rows,
             torch.tensor([[float('nan'), 1.0]], dtype=dtype,
                          device=device)])
        mixed_y = rms_norm(mixed)
        _check(results, f'rms_norm mixed rows keep ordinary rows exact '
               f'at {dtype}',
               torch.equal(mixed_y[1:3], rms_norm(normal_rows)), '')
        _check(results, f'rms_norm mixed rows keep finite rows at {dtype}',
               bool(torch.isfinite(mixed_y[0]).all())
               and bool(mixed_y[0].abs().sum() > 0), f'{mixed_y.tolist()}')

    x = torch.tensor([[1.0, 2.0]], device=device)
    huge_eps = float(torch.finfo(torch.float32).max)
    y = rms_norm(x, eps=huge_eps)
    _check(results, 'rms_norm stays finite for an eps at the dtype limit',
           bool(torch.isfinite(y).all()) and bool((y > 0).all()),
           f'{y.tolist()}')

    limit = torch.finfo(torch.float32).max
    xg = torch.tensor([[limit, 0.0], [limit, limit]],
                      dtype=torch.float32, device=device,
                      requires_grad=True)
    yg = rms_norm(xg)
    (grad,) = torch.autograd.grad(yg.sum(), xg)
    _check(results, 'rms_norm fallback gradient finite at dtype limit',
           bool(torch.isfinite(grad).all()), f'{grad.tolist()}')
    return results


def test_qk_normalize_delegates_to_rmsnorm(device='cpu'):
    results = []
    q = torch.randn(2, 4, 3, 8, dtype=torch.float32, device=device) * 0.7
    q_n = _qk_normalize(q, 'rms')
    inline = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + EPS)
    _check(results, "_qk_normalize('rms') equals the reference bit-exactly",
           torch.equal(q_n, rms_norm(q)) and torch.equal(q_n, inline), '')
    q_inplace = q.clone()
    ret_q = _qk_normalize_(q_inplace, 'rms')
    _check(results, "_qk_normalize_('rms') is bit-identical to out-of-place",
           ret_q is q_inplace and torch.equal(ret_q, q_n), '')
    return results


def test_qk_l2_normalize_preserves_finite_dtype_limit_direction():
    """Finite rows whose L2 norm overflows must not become all zeros."""
    import torch.nn.functional as functional

    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    tolerances = {torch.float16: 2e-3, torch.bfloat16: 2e-3,
                  torch.float32: 1e-6, torch.float64: 1e-12}
    for dtype, tol in tolerances.items():
        normal = torch.randn(3, 7, dtype=dtype) * 0.7
        assert torch.equal(_qk_normalize(normal, 'l2'),
                           functional.normalize(normal, dim=-1))

        limit = torch.finfo(dtype).max
        x = torch.tensor([[limit, 0.0], [limit, limit],
                          [-limit, limit], [0.0, 0.0]], dtype=dtype)
        expected = torch.tensor(
            [[1.0, 0.0], [inv_sqrt2, inv_sqrt2],
             [-inv_sqrt2, inv_sqrt2], [0.0, 0.0]], dtype=dtype)
        y = _qk_normalize(x, 'l2')
        assert bool(torch.isfinite(y).all())
        torch.testing.assert_close(y, expected, atol=tol, rtol=tol)

        y_inplace = x.clone()
        assert _qk_normalize_(y_inplace, 'l2') is y_inplace
        torch.testing.assert_close(y_inplace, expected, atol=tol, rtol=tol)

        normal_row = normal[:1, :2]
        mixed = torch.cat(
            [normal_row, x[:1],
             torch.tensor([[float('nan'), 1.0]], dtype=dtype)])
        mixed_y = _qk_normalize(mixed, 'l2')
        assert torch.equal(mixed_y[:1],
                           functional.normalize(normal_row, dim=-1))
        assert bool(torch.isfinite(mixed_y[1]).all())
        assert bool(mixed_y[1].abs().sum() > 0)
        mixed_inplace = mixed.clone()
        assert _qk_normalize_(mixed_inplace, 'l2') is mixed_inplace
        torch.testing.assert_close(mixed_inplace, mixed_y, equal_nan=True)


def test_rwkv7_key_norm_preserves_finite_overflow(monkeypatch):
    from types import SimpleNamespace

    import ops_rwkv_backend
    from ops_rwkv import Rwkv7Mixer

    cfg = SimpleNamespace(
        d_model=16, rwkv_head_dim=4, rwkv_decay_lora=4,
        rwkv_a_lora=4, rwkv_gate_lora=4)
    mixer = Rwkv7Mixer(cfg).eval()
    with torch.no_grad():
        mixer.k_k.fill_(1e30)
    captured = {}
    real_forward = ops_rwkv_backend.rwkv7_forward

    def capture_forward(r, w, k, v, kk, a, **kwargs):
        captured['kk'] = kk.detach().clone()
        return real_forward(r, w, k, v, kk, a, **kwargs)

    monkeypatch.setattr(ops_rwkv_backend, 'rwkv7_forward', capture_forward)
    x = torch.randn(1, 3, 16)
    with torch.no_grad():
        mixer.forward_functional(x, None, None)
    kk = captured['kk']
    assert torch.isfinite(kk).all()
    norms = kk.norm(dim=-1)
    torch.testing.assert_close(
        norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_kda_runner_qk_norm_preserves_finite_overflow():
    from run_decoding import KDAAttnDecoding
    from run_quality import KDAAttn

    torch.manual_seed(0)
    x = torch.randn(1, 3, 16)
    modules = [KDAAttnDecoding(16, H=2, K=4, V=4),
               KDAAttn(16, H=2, K=4, V=4)]
    for module in modules:
        module.eval()
        with torch.no_grad():
            module.q.weight.mul_(1e25)
            module.k.weight.mul_(1e25)
        with torch.no_grad():
            y = module(x)
        assert torch.isfinite(y).all()
        assert y.abs().max() > 0


def main():
    all_results = []
    for fn in (test_rmsnorm_shape_polymorphism, test_rmsnorm_inplace_and_affine,
               test_rmsnorm_dtype_and_autograd, test_rmsnorm_validation,
               test_rmsnorm_overflow_finite_inputs,
               test_qk_normalize_delegates_to_rmsnorm):
        all_results += fn()
    failed = [r for r in all_results if r['status'] != 'PASS']
    print(f"\n{len(all_results) - len(failed)}/{len(all_results)} checks passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
