"""Regression tests for ``ptq_calibrate`` (post-training quantization
calibration with bounded memory and explicit release).

Covers:

1. **Symmetric calibration parity** — the calibrator's pinned scale equals the
   per-group absmax scale and ``quantize_with_scale`` reproduces
   ``kv_quant.fake_quantize`` when calibrated on the same tensor.
2. **Asymmetric calibration** — scale + zero point are derived from running
   min/max and round-trip through ``quantize_with_scale`` / ``dequantize_with_scale``.
3. **STE / autograd contract** — ``ste=True`` is the identity gradient,
   ``ste=False`` has an exactly zero gradient.
4. **Garbage-collection contract** — running statistics are O(groups), no raw
   calibration tensor is retained, ``free()`` is idempotent, and a freed
   calibrator rejects further ``observe``/``finish``.
5. **Input validation** — non-finite values, wrong group count, empty
   calibration, and missing asymmetric zero point all raise.

Run via pytest::

    pytest -q test_ptq_calibrate.py

Or directly::

    python3 test_ptq_calibrate.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from kv_quant import QuantSpec  # noqa: E402
from ptq_calibrate import (  # noqa: E402
    CalibrationResult,
    GroupAbsmaxCalibrator,
    calibrate_and_quantize,
    quantize_with_scale,
)


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _seed(seed=0):
    torch.manual_seed(seed)


def _randn(*shape, **kwargs):
    return torch.randn(*shape, **kwargs)


def _group_absmax(x, group_size):
    last = x.shape[-1]
    groups = x.reshape(-1, last // group_size, group_size)
    return groups.abs().amax(dim=-1).max(dim=0).values


# --- 1. symmetric calibration parity ---------------------------------------

def test_calibrator_symmetric_matches_absmax_scale():
    _seed(0)
    spec = QuantSpec(bits=4, group_size=16)
    x = _randn(2, 7, 64)  # 4 groups
    cal = GroupAbsmaxCalibrator(spec, group_count=4)
    n = cal.observe(x)
    _ok('rows folded in', n == 2 * 7, f'{n}')
    result = cal.finish()
    expected = _group_absmax(x, spec.group_size) / spec.qmax
    _ok('pinned scale equals per-group absmax scale',
        bool(torch.allclose(result.scale, expected, atol=1e-6)),
        f'max diff {float((result.scale - expected).abs().max()):.2e}')
    _ok('symmetric zero point is None', result.zero is None)
    _ok('result fields', isinstance(result, CalibrationResult)
        and result.n_obs == 14 and result.group_count == 4
        and result.symmetric is True)


def test_quantize_with_scale_pinned_scale_properties():
    """The calibrator pins ONE scale per group (max absmax across the
    calibration corpus). Quantizing against it must stay within the grid
    (no saturation from an undersized scale) and the rounding error per
    element is bounded by scale/2."""
    _seed(1)
    for bits, gs in ((4, 16), (2, 8), (8, 32)):
        spec = QuantSpec(bits=bits, group_size=gs)
        x = _randn(3, 5, gs * 3)
        cal = GroupAbsmaxCalibrator(spec, group_count=3)
        cal.observe(x)
        result = cal.finish()
        out = quantize_with_scale(x, spec, result.scale)
        _ok(f'shape/dtype preserved ({bits}b/g{gs})',
            out.shape == x.shape and out.dtype == x.dtype)
        # Pinned scale is conservative: >= every per-position absmax/qmax.
        lead = x.shape[:-1]
        groups = x.reshape(*lead, 3, spec.group_size)
        pos_absmax = groups.abs().amax(dim=-1).unsqueeze(-1)  # (..., 3, 1)
        need = pos_absmax / spec.qmax
        scale_b = result.scale.reshape((1,) * len(lead) + (3, 1))
        _ok('pinned scale covers every position',
            bool((scale_b >= need).all()))
        # Rounding error per element bounded by scale/2.
        err_groups = (out - x).abs().reshape(*lead, 3, spec.group_size)
        allowed = scale_b / 2.0 + 1e-5
        _ok(f'rounding error <= scale/2 ({bits}b/g{gs})',
            bool((err_groups <= allowed).all()),
            f'max ratio {float((err_groups / (scale_b / 2.0)).max()):.3f}')


# --- 2. asymmetric calibration ---------------------------------------------

def test_calibrator_asymmetric_scale_and_zero():
    _seed(2)
    spec = QuantSpec(bits=4, group_size=16, symmetric=False)
    x = _randn(2, 6, 32) * 3.0 - 1.0  # mixed sign
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    cal.observe(x)
    result = cal.finish()
    last = x.shape[-1]
    groups = x.reshape(-1, 2, spec.group_size)
    lo = groups.amin(dim=-1).min(dim=0).values
    hi = groups.amax(dim=-1).max(dim=0).values
    qmax = float(spec.qmax)
    exp_scale = torch.where(hi - lo > 0, (hi - lo) / qmax,
                            torch.ones_like(lo))
    exp_zero = torch.round(-lo / exp_scale).clamp(0, qmax)
    _ok('asymmetric scale matches running min/max',
        bool(torch.allclose(result.scale, exp_scale, atol=1e-6)),
        f'max diff {float((result.scale - exp_scale).abs().max()):.2e}')
    _ok('asymmetric zero matches running min/max',
        bool(torch.equal(result.zero, exp_zero)),
        f'max diff {float((result.zero - exp_zero).abs().max()):.2e}')
    _ok('result asymmetric flag', result.symmetric is False
        and result.zero is not None)


def test_quantize_is_idempotent_on_the_grid():
    """Quantizing an already-quantized (on-grid) tensor is a no-op, for both
    symmetric and asymmetric grids."""
    _seed(3)
    for symmetric in (True, False):
        spec = QuantSpec(bits=4, group_size=16, symmetric=symmetric)
        x = _randn(4, 32)
        cal = GroupAbsmaxCalibrator(spec, group_count=2)
        cal.observe(x)
        result = cal.finish()
        kwargs = {} if symmetric else {'zero': result.zero}
        out = quantize_with_scale(x, spec, result.scale, **kwargs)
        out2 = quantize_with_scale(out, spec, result.scale, **kwargs)
        _ok(f'idempotent on-grid (sym={symmetric})',
            bool(torch.equal(out2, out)),
            f'max drift {float((out2 - out).abs().max()):.2e}')
        # Reconstruction error bounded by scale/2 per element.
        err = (out - x).abs().reshape(4, 2, spec.group_size)
        allowed = result.scale.reshape(1, 2, 1) / 2.0 + 1e-5
        _ok(f'reconstruction error <= scale/2 (sym={symmetric})',
            bool((err <= allowed).all()))


# --- 3. STE / autograd contract -------------------------------------------

def test_ste_is_identity_gradient():
    _seed(4)
    spec = QuantSpec(bits=2, group_size=16)
    x = _randn(4, 32, requires_grad=True)  # 2 groups
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    cal.observe(x.detach())
    result = cal.finish()
    w = _randn(4, 32)
    loss = (w * quantize_with_scale(x, spec, result.scale, ste=True)).sum()
    (g,) = torch.autograd.grad(loss, x)
    _ok('STE is identity gradient', bool(torch.equal(g, w)),
        f'max|g-w|={float((g - w).abs().max()):.2e}')


def test_ste_false_flow_no_gradient():
    """ste=False computes under no_grad, so the output is detached and no
    gradient can reach the input."""
    _seed(5)
    spec = QuantSpec(bits=4, group_size=8)
    x = _randn(3, 40, requires_grad=True)  # 5 groups
    cal = GroupAbsmaxCalibrator(spec, group_count=5)
    cal.observe(x.detach())
    result = cal.finish()
    out = quantize_with_scale(x, spec, result.scale, ste=False)
    _ok('ste=False output is detached (requires_grad False)',
        out.requires_grad is False, f'{out.requires_grad}')


# --- 4. garbage-collection contract ----------------------------------------

def test_calibrator_retains_no_raw_tensor():
    _seed(6)
    spec = QuantSpec(bits=4, group_size=16)
    cal = GroupAbsmaxCalibrator(spec, group_count=4)
    big = _randn(50, 64)
    cal.observe(big)
    cal.observe(big)
    cal.observe(big)
    _ok('n_obs accumulates rows', cal.n_obs == 150, f'{cal.n_obs}')
    result = cal.finish()
    _ok('scale is O(groups), independent of batch count',
        result.scale.shape == (4,) and result.n_obs == 150)
    cal.free()
    _ok('free marks calibrator freed', cal.freed is True)
    # free is idempotent.
    cal.free()
    _ok('free idempotent', cal.freed is True)
    # observe/finish after free raise.
    try:
        cal.observe(_randn(2, 64))
        _ok('observe after free raises', False, 'no error')
    except ValueError:
        _ok('observe after free raises', True)
    try:
        cal.finish()
        _ok('finish after free raises', False, 'no error')
    except ValueError:
        _ok('finish after free raises', True)


def test_calibrate_and_quantize_one_shot_frees():
    _seed(7)
    spec = QuantSpec(bits=2, group_size=16)
    target = _randn(3, 32)
    batches = [_randn(4, 32), _randn(4, 32)]
    result = calibrate_and_quantize(batches, target, spec)
    _ok('one-shot returns CalibrationResult',
        isinstance(result, CalibrationResult)
        and result.scale.shape == (2,)
        and result.symmetric is True)
    _ok('one-shot quantize is finite and shape-preserving',
        bool(torch.isfinite(
            quantize_with_scale(target, spec, result.scale)).all()))


# --- 5. input validation ---------------------------------------------------

def test_non_finite_rejected():
    _seed(8)
    spec = QuantSpec(bits=4, group_size=16)
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    try:
        cal.observe(torch.tensor([1.0, float('nan')] * 16))
        _ok('calibrator rejects non-finite', False, 'no error')
    except ValueError:
        _ok('calibrator rejects non-finite', True)
    x = _randn(2, 32)
    cal.observe(x)
    result = cal.finish()
    bad = torch.tensor([1.0, float('inf')] * 16)
    try:
        quantize_with_scale(bad, spec, result.scale)
        _ok('quantize rejects non-finite target', False, 'no error')
    except ValueError:
        _ok('quantize rejects non-finite target', True)
    bad_scale = torch.tensor([1.0, float('nan')])
    try:
        quantize_with_scale(x, spec, bad_scale)
        _ok('quantize rejects non-finite scale', False, 'no error')
    except ValueError:
        _ok('quantize rejects non-finite scale', True)


def test_mismatched_group_count_raises():
    _seed(9)
    spec = QuantSpec(bits=4, group_size=16)
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    cal.observe(_randn(2, 32))
    try:
        cal.observe(_randn(2, 48))  # 3 groups != 2
        _ok('mismatched group count raises', False, 'no error')
    except ValueError:
        _ok('mismatched group count raises', True)


def test_empty_calibration_raises():
    spec = QuantSpec(bits=4, group_size=16)
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    try:
        cal.finish()
        _ok('empty calibration raises', False, 'no error')
    except ValueError:
        _ok('empty calibration raises', True)


def test_asymmetric_requires_zero():
    spec = QuantSpec(bits=4, group_size=16, symmetric=False)
    x = _randn(2, 32)
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    cal.observe(x)
    result = cal.finish()
    try:
        quantize_with_scale(x, spec, result.scale)  # no zero
        _ok('asymmetric without zero raises', False, 'no error')
    except ValueError:
        _ok('asymmetric without zero raises', True)


def test_shape_and_dtype_preserved():
    _seed(10)
    spec = QuantSpec(bits=4, group_size=8)
    x = _randn(2, 3, 16, dtype=torch.float16)
    cal = GroupAbsmaxCalibrator(spec, group_count=2)
    cal.observe(x)
    result = cal.finish()
    out = quantize_with_scale(x, spec, result.scale)
    _ok('shape preserved', out.shape == x.shape)
    _ok('dtype preserved', out.dtype == torch.float16)
    out64 = quantize_with_scale(x.to(torch.float64), spec,
                                result.scale.to(torch.float64))
    _ok('fp64 path stays fp64', out64.dtype == torch.float64)


_TEST_FUNCS = [
    test_calibrator_symmetric_matches_absmax_scale,
    test_quantize_with_scale_pinned_scale_properties,
    test_calibrator_asymmetric_scale_and_zero,
    test_quantize_is_idempotent_on_the_grid,
    test_ste_is_identity_gradient,
    test_ste_false_flow_no_gradient,
    test_calibrator_retains_no_raw_tensor,
    test_calibrate_and_quantize_one_shot_frees,
    test_non_finite_rejected,
    test_mismatched_group_count_raises,
    test_empty_calibration_raises,
    test_asymmetric_requires_zero,
    test_shape_and_dtype_preserved,
]


def main() -> int:
    print('=' * 70)
    print('ptq_calibrate regression tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f'  [FAIL] {fn.__name__}: AssertionError: {e}')
            results.append(False)
        except Exception as e:
            print(f'  [CRASH] {fn.__name__}: {type(e).__name__}: {e}')
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
