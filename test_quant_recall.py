"""Regression tests for ``kv_quant.py`` and Exp 10 (``run_quant_recall``).

Covers four layers:

1. **Byte accounting** — hand-computed ``QuantSpec`` costs (payload +
   per-group scale/zero-point metadata), validation of the spec contract,
   and the ``kv_budget`` byte bridge (``quantized_row_bytes`` /
   ``op_read_bytes`` / ``layout_read_bytes``) including the retained-state
   vs per-query-read split and policy typo protection.
2. **Fake-quantize contract** — grouped symmetric/asymmetric error
   bounds, exact reconstruction of zero and constant groups (with
   non-divisible remainders), non-finite rejection, dtype preservation,
   and the straight-through gradient.
3. **Planted-recall probe** — the stepwise write path (one token per call
   with ``initial_state``) reproduces the full-sequence reference state;
   without quantization both routes recall planted orthonormal-key
   associations exactly; 8-bit state keeps exact recall while 2-bit loses
   it; recall is monotone in bits on the pinned seeds.
4. **Report self-consistency** — Exp 10 rows are strict JSON, byte costs
   match ``state_elements * bits_per_element / 8``, and the frontier
   summary names a cell actually present in the measured grid.

The tests are self-contained and never touch the real ``results/``
directory.

Run via pytest::

    pytest -q test_quant_recall.py

Or directly::

    python3 test_quant_recall.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import kv_budget  # noqa: E402
from kaggle_setup import sanitize_for_json  # noqa: E402
from kv_quant import (  # noqa: E402
    FP8Spec,
    QuantSpec,
    _fp8_round_magnitude,
    bits_per_element,
    bytes_per_element,
    compression_ratio,
    fake_quantize,
    quantization_error,
    quantized_row_bytes,
    spec_summary,
    storage_bytes,
)
from run_quant_recall import (  # noqa: E402
    _STATE_FN,
    build_quant_recall_rows,
    full_sequence_state,
    planted_batch,
    run_planted_recall,
)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


_MQAR_SPEC = {
    'softmax': {'n_kv_heads': 2, 'd_k': 4, 'd_v': 4},
    'kda': {'hv': 1, 'd_k': 8, 'd_v': 4},
    'mamba2': {'nheads': 2, 'headdim': 4, 'd_state': 8},
    'csa': {'m': 4, 'topk': 2, 'c': 8, 'c_i': 4},
    'hca': {'m2': 8, 'c': 8},
}


# --- 1. byte accounting ---------------------------------------------------

def test_quant_spec_accounting():
    print('\nTest: QuantSpec byte accounting (hand values)')
    spec = QuantSpec(bits=4, group_size=64)
    _ok('4b/g64 symmetric = 4.25 bits/element',
        abs(bits_per_element(spec) - 4.25) < 1e-12, f'{bits_per_element(spec)}')
    _ok('4b/g64 = 0.53125 bytes/element',
        abs(bytes_per_element(spec) - 0.53125) < 1e-12,
        f'{bytes_per_element(spec)}')
    _ok('compression vs fp16 = 16/4.25',
        abs(compression_ratio(spec) - 16.0 / 4.25) < 1e-12,
        f'{compression_ratio(spec):.4f}')
    _ok('storage_bytes(256, 4b/g64) = 136.0 B',
        abs(storage_bytes(256, spec) - 136.0) < 1e-12,
        f'{storage_bytes(256, spec)}')
    asym = QuantSpec(bits=4, group_size=64, symmetric=False)
    _ok('asymmetric zero_bits defaults to payload bits',
        asym.zero_bits == 4
        and abs(bits_per_element(asym) - (4.0 + 20.0 / 64.0)) < 1e-12,
        f'zero_bits={asym.zero_bits}, bpe={bits_per_element(asym)}')
    free = QuantSpec(bits=4, group_size=32, scale_bits=0)
    _ok('scale_bits=0 is the ideal payload-only bound',
        abs(bits_per_element(free) - 4.0) < 1e-12,
        f'{bits_per_element(free)}')
    _ok('group_size=1 stores one scale per element',
        abs(bits_per_element(QuantSpec(4, 1)) - 20.0) < 1e-12, '')
    for kwargs, label in [
        (dict(bits=1), 'bits<2 rejected'),
        (dict(bits=17), 'bits>16 rejected'),
        (dict(bits=4, group_size=0), 'group_size=0 rejected'),
        (dict(bits=4, zero_bits=8), 'symmetric zero_bits rejected'),
        (dict(bits=4, scale_bits=-1), 'negative scale_bits rejected'),
    ]:
        try:
            QuantSpec(**kwargs)
            _ok(label, False, 'no error')
        except ValueError:
            _ok(label, True)
    for bad_n in (-1, True, float('nan'), float('inf'), 10 ** 400):
        try:
            storage_bytes(bad_n, spec)
            _ok(f'element count {bad_n!r} rejected', False, 'no error')
        except ValueError:
            _ok(f'element count {bad_n!r} rejected', True)
    for bad_base in (True, 0.0, float('nan'), float('inf'), 10 ** 400):
        try:
            compression_ratio(spec, bad_base)
            _ok(f'baseline {bad_base!r} rejected', False, 'no error')
        except ValueError:
            _ok(f'baseline {bad_base!r} rejected', True)


def test_quantized_row_bytes_bridge():
    print('\nTest: kv_budget byte bridge')
    row = kv_budget.op_read_elements('kda', 16, _MQAR_SPEC)
    _ok('KDA state_elements = hv*d_k*d_v = 32',
        row['state_elements'] == 32, f"{row['state_elements']}")
    _ok('KDA reads the state twice (core = 64)',
        abs(row['core_read_elements'] - 64.0) < 1e-12,
        f"{row['core_read_elements']}")
    _ok('state is excluded from total_read_elements',
        abs(row['total_read_elements'] - row['core_read_elements']) < 1e-12, '')
    byte_row = dict(row)
    byte_row.update(quantized_row_bytes(row, {'state': QuantSpec(4, 64)}))
    _ok('state_bytes prices the retained state (32 * 0.53125 = 17)',
        abs(byte_row['state_bytes'] - 17.0) < 1e-12,
        f"{byte_row['state_bytes']}")
    _ok('read components keep the fp16 default',
        abs(byte_row['core_read_bytes'] - 128.0) < 1e-12
        and abs(byte_row['total_read_bytes'] - 128.0) < 1e-12,
        f"core={byte_row['core_read_bytes']}, "
        f"total={byte_row['total_read_bytes']}")
    _ok('policy map is reported per component',
        byte_row['quant_bytes_per_element']['state'] == 0.53125
        and byte_row['quant_bytes_per_element']['core'] == 2.0, '')
    try:
        quantized_row_bytes(row, {'bogus': QuantSpec(4, 64)})
        _ok('unknown policy stem rejected', False, 'no error')
    except ValueError:
        _ok('unknown policy stem rejected', True)
    # JSON integers are unbounded; an element count too large for float is a
    # malformed row, not an OverflowError from the byte pricing.
    for key in ('core_read_elements', 'state_elements'):
        bad = dict(row)
        bad[key] = 10 ** 400
        try:
            quantized_row_bytes(bad)
            _ok(f'oversized {key} rejected', False, 'no error')
        except ValueError:
            _ok(f'oversized {key} rejected', True)
    # A QuantSpec applies to every component; None = fp16 baseline.
    allq = quantized_row_bytes(row, QuantSpec(8, 64))
    baseq = quantized_row_bytes(row)
    _ok('single QuantSpec prices every component',
        abs(allq['core_read_bytes'] - 64.0 * bits_per_element(
            QuantSpec(8, 64)) / 8.0) < 1e-12, '')
    _ok('default policy is fp16 (2 B/element)',
        abs(baseq['core_read_bytes'] - 128.0) < 1e-12, '')
    # Layout-level bridge: stems are the routes.
    lay = kv_budget.layout_read_bytes(16, (3, 1, 1), _MQAR_SPEC,
                                      quant={'kda': QuantSpec(4, 64)})
    _ok('layout row carries per-route byte columns',
        all(k in lay for k in ('kda_read_bytes', 'csa_read_bytes',
                               'hca_read_bytes', 'total_read_bytes')),
        ', '.join(sorted(k for k in lay if k.endswith('_bytes'))))
    _ok('layout total = sum of route byte columns',
        abs(lay['total_read_bytes']
            - (lay['kda_read_bytes'] + lay['csa_read_bytes']
               + lay['hca_read_bytes'])) < 1e-9, '')
    _ok('route-neutral memory aggregate stays diagnostic (default fp16)',
        abs(lay['memory_read_bytes'] - 3 * 64.0 * 2.0) < 1e-9
        and abs(lay['total_read_bytes']
                - (lay['kda_read_bytes'] + lay['csa_read_bytes']
                   + lay['hca_read_bytes'])) < 1e-9,
        f"memory={lay['memory_read_bytes']} total={lay['total_read_bytes']}")
    lay_m = kv_budget.layout_read_bytes(
        16, (3, 1, 1), _MQAR_SPEC, quant={'mamba2': QuantSpec(4, 64)},
        memory_op='mamba2')
    _ok('Mamba-2 layout total excludes the memory aggregate',
        abs(lay_m['total_read_bytes']
            - (lay_m['mamba2_read_bytes'] + lay_m['csa_read_bytes']
               + lay_m['hca_read_bytes'])) < 1e-9
        and lay_m['kda_read_bytes'] == 0.0
        and abs(lay_m['memory_read_bytes'] - 3 * 128.0 * 2.0) < 1e-9,
        f'{lay_m}')
    try:
        kv_budget.layout_read_bytes(16, (3, 1, 1), _MQAR_SPEC,
                                    quant={'core': QuantSpec(4, 64)})
        _ok('layout policy rejects operator-level stems', False, 'no error')
    except ValueError:
        _ok('layout policy rejects operator-level stems', True)


# --- 2. fake-quantize contract --------------------------------------------

def test_fake_quantize_contract():
    print('\nTest: grouped fake-quantize contract')
    torch.manual_seed(11)
    x = torch.rand(4, 128) * 2.0 - 1.0
    sym = QuantSpec(bits=4, group_size=32, symmetric=True)
    err = (x - fake_quantize(x, sym)).abs().max().item()
    # Per-group bound |err| <= amax/(2*qmax); absmax(x) <= 1 -> 1/14.
    _ok('symmetric 4-bit max error <= 1/14 + eps',
        err <= 1.0 / 14.0 + 1e-6, f'{err:.3e}')
    asym = QuantSpec(bits=4, group_size=32, symmetric=False)
    err_a = (x - fake_quantize(x, asym)).abs().max().item()
    _ok('asymmetric 4-bit max error <= 1/15 + eps',
        err_a <= 1.0 / 15.0 + 1e-6, f'{err_a:.3e}')
    # Grid property: every dequantized value is an integer multiple of its
    # group scale.
    dq = fake_quantize(x, sym)
    scale = x.reshape(4, -1, 32).abs().amax(-1, keepdim=True).reshape(4, -1, 1) \
        / sym.qmax
    ratio = (dq.reshape(4, -1, 32) / scale).reshape(4, -1)
    _ok('dequantized values lie on the per-group grid',
        bool((ratio - ratio.round()).abs().max() < 1e-4),
        f'max grid residual={float((ratio - ratio.round()).abs().max()):.2e}')
    _ok('all-zero input quantizes to exact zero',
        bool((fake_quantize(torch.zeros(2, 40), sym) == 0).all()))
    # Constant groups reconstruct exactly (span == 0 path), including a
    # non-divisible remainder: 37 columns = two full groups of 16 plus a
    # 5-column remainder that the quantizer zero-pads internally.
    base = torch.randn(3, 3)
    const = base.repeat_interleave(16, dim=1)[:, :37]
    for spec, label in [(QuantSpec(4, 16), 'symmetric'),
                        (QuantSpec(4, 16, symmetric=False), 'asymmetric')]:
        dqc = fake_quantize(const, spec)
        _ok(f'constant groups ({label}) round-trip exactly',
            dqc.shape == const.shape
            and bool(torch.allclose(dqc, const, atol=1e-6)),
            f'max|err|={(dqc - const).abs().max().item():.2e}')
    # Remainder groups do not leak across group boundaries: a tensor whose
    # first 16 columns are constant 1.0 and last 5 are constant 0.0 must
    # stay exact under both quantizers.
    split = torch.cat([torch.full((1, 16), 1.0),
                       torch.full((1, 5), 0.0)], dim=1)
    for spec, label in [(QuantSpec(4, 16), 'symmetric'),
                        (QuantSpec(4, 16, symmetric=False), 'asymmetric')]:
        dqs = fake_quantize(split, spec)
        _ok(f'remainder handling keeps group boundaries ({label})',
            bool(torch.allclose(dqs, split, atol=1e-6)),
            f'max|err|={(dqs - split).abs().max().item():.2e}')
    bad = x.clone()
    bad[0, 0] = float('nan')
    try:
        fake_quantize(bad, sym)
        _ok('non-finite input rejected', False, 'no error')
    except ValueError:
        _ok('non-finite input rejected', True)
    g = x.clone().requires_grad_(True)
    fake_quantize(g, sym, ste=True).sum().backward()
    _ok('STE gradient is exactly 1',
        bool((g.grad == 1.0).all()), f'{float(g.grad.mean()):.3f}')
    g2 = x.clone().requires_grad_(True)
    fake_quantize(g2, sym, ste=False).sum().backward()
    _ok('ste=False exposes the zero-gradient quantizer',
        bool((g2.grad == 0.0).all()),
        f'max|grad|={float(g2.grad.abs().max()):.1e}')
    dq16 = fake_quantize(x.half(), sym)
    _ok('fp16 input keeps fp16 dtype', dq16.dtype == torch.float16,
        f'{dq16.dtype}')
    stats = quantization_error(x, sym)
    _ok('quantization_error matches direct measurement',
        abs(stats['max_abs_error'] - err) < 1e-12 and stats['snr_db'] > 0,
        f"snr={stats['snr_db']:.1f} dB")
    # Asymmetric groups whose values are all positive (or all negative)
    # must still use the full unsigned code grid: the zero-point clamp
    # must not push the far end past qmax and saturate it.
    asym = QuantSpec(bits=4, group_size=16, symmetric=False)
    onesided = torch.linspace(1.0, 4.5, 16)[None, :]
    for label, v in (('all-positive', onesided),
                     ('all-negative', -onesided)):
        dq_one = fake_quantize(v, asym, ste=False)
        span = (v.max().clamp(min=0) - v.min().clamp(max=0)).item()
        step = span / asym.qmax
        err_one = (v - dq_one).abs().max().item()
        _ok(f'asymmetric {label} groups span the full code grid',
            err_one <= step / 2 + 1e-6,
            f'max|err|={err_one:.3e}, half-step={step / 2:.3e}')


# --- 3. planted-recall probe ----------------------------------------------

def test_quantized_state_recall_probe():
    print('\nTest: quantized-state planted recall (both memory routes)')
    keys, values = planted_batch(16, 8, seed=0, device='cpu')
    for route in ('kda', 'mamba2'):
        step = run_planted_recall(route, keys, values, None)
        ref = full_sequence_state(route, keys, values)
        step_state = _STATE_FN[route](keys, values, None, False)
        dev = (step_state - ref).abs().max().item()
        _ok(f'{route}: stepwise write path == full-sequence state',
            dev < 1e-5, f'max|dS|={dev:.2e}')
        _ok(f'{route}: no-quant recall is exact',
            step['readback_recall'] == 1.0
            and step['max_read_error'] < 1e-4,
            f"recall={step['readback_recall']}, "
            f"err={step['max_read_error']:.2e}")
        q8 = run_planted_recall(route, keys, values, QuantSpec(8, 64))
        _ok(f'{route}: 8-bit state keeps exact recall',
            q8['readback_recall'] == 1.0,
            f"recall={q8['readback_recall']}, "
            f"err={q8['max_read_error']:.3f}")
        q2 = run_planted_recall(route, keys, values, QuantSpec(2, 64))
        _ok(f'{route}: 2-bit state loses recall',
            q2['readback_recall'] < q8['readback_recall'],
            f"2-bit={q2['readback_recall']} vs 8-bit="
            f"{q8['readback_recall']}")
        q4 = run_planted_recall(route, keys, values, QuantSpec(4, 16))
        q3 = run_planted_recall(route, keys, values, QuantSpec(3, 16))
        _ok(f'{route}: 4-bit state is exact on the pinned seed',
            q4['readback_recall'] == 1.0, f"{q4['readback_recall']}")
        _ok(f'{route}: recall is monotone in bits on the pinned seed',
            q2['readback_recall'] <= q3['readback_recall']
            <= q4['readback_recall'] <= q8['readback_recall'], '')


# --- 4. report self-consistency -------------------------------------------

def test_quant_recall_report_rows():
    print('\nTest: Exp 10 report rows (strict JSON, self-consistent)')
    rows = build_quant_recall_rows(
        device='cpu', seeds=(0,), bits_list=(2, 4), group_sizes=(64,),
        include_asymmetric=False)
    cells = [r for r in rows if r['row_kind'] == 'quant_recall']
    summaries = [r for r in rows if r['row_kind'] == 'frontier_summary']
    _ok('2 routes x 3 specs = 6 cells', len(cells) == 6, f'{len(cells)}')
    _ok('one frontier summary per route', len(summaries) == 2,
        f'{len(summaries)}')
    try:
        json.dumps(sanitize_for_json(rows), allow_nan=False)
        json_detail = 'strict JSON'
        json_ok = True
    except (ValueError, TypeError) as e:
        json_ok = False
        json_detail = repr(e)
    _ok('report round-trips as strict JSON', json_ok, json_detail)
    bytes_ok = True
    for r in cells:
        if r['quantized']:
            bpe = bits_per_element(QuantSpec(
                r['bits'], r['group_size'], symmetric=r['symmetric']))
        else:
            bpe = 16.0
        expected = r['state_elements'] * bpe / 8.0
        if abs(r['state_bytes'] - expected) > 1e-9:
            bytes_ok = False
    _ok('state_bytes == elements * bits_per_element / 8', bytes_ok, '')
    rec_ok = True
    for r in cells:
        if not (0.0 <= r['recall_min'] <= r['recall_mean'] <= r['recall_max']
                <= 1.0):
            rec_ok = False
        if r['exact_recall'] != (r['recall_min'] == 1.0):
            rec_ok = False
    _ok('recall aggregates are ordered and exact_recall is conservative',
        rec_ok, '')
    _ok('single-seed cells leave std/CI undefined instead of faking zero',
        all(r['recall_std'] is None and r['recall_ci95'] is None
            and r['n_seeds'] == 1 for r in cells), '')
    multi = build_quant_recall_rows(
        device='cpu', seeds=(0, 1, 2), bits_list=(2,), group_sizes=(64,),
        include_asymmetric=False)
    multi_cells = [r for r in multi if r['row_kind'] == 'quant_recall']
    _ok('multi-seed cells report sample std and a 95% CI',
        all(r['n_seeds'] == 3 and r['recall_std'] is not None
            and r['recall_ci95'] is not None
            and r['recall_ci95'][0] <= r['recall_mean']
            <= r['recall_ci95'][1] for r in multi_cells),
        f'{len(multi_cells)} cells')
    for s in summaries:
        cell = s['cheapest_exact_cell']
        exact = [r for r in cells
                 if r['route'] == s['route'] and r['exact_recall']]
        in_grid = (cell is not None
                   and any(r['bits'] == cell['bits']
                           and r['group_size'] == cell['group_size']
                           and r['quantized'] == cell['quantized']
                           for r in exact))
        _ok(f"{s['route']}: cheapest exact cell is in the measured grid",
            in_grid, f'{cell is not None}')
        if cell is not None and exact:
            cheapest = min(exact, key=lambda r: r['state_bytes'])
            _ok(f"{s['route']}: frontier picks the cheapest exact cell",
                abs(cell['state_bytes'] - cheapest['state_bytes']) < 1e-9, '')


# --- 5. block-scaled FP8 storage -----------------------------------------

def test_fp8_spec_and_grid():
    print('\nTest: FP8Spec accounting and FP8 grid')
    spec = FP8Spec()
    _ok('MXFP8 default = 8 + 8/32 = 8.25 bits/element',
        abs(bits_per_element(spec) - 8.25) < 1e-12,
        f'{bits_per_element(spec)}')
    _ok('MXFP8 default = 1.03125 bytes/element',
        abs(bytes_per_element(spec) - 1.03125) < 1e-12,
        f'{bytes_per_element(spec)}')
    _ok('MXFP8 compression vs fp16 = 16/8.25',
        abs(spec_summary(spec)['compression_vs_fp16'] - 16.0 / 8.25)
        < 1e-12, f"{spec_summary(spec)['compression_vs_fp16']:.4f}")
    _ok('storage_bytes(256, MXFP8) = 264 B',
        abs(storage_bytes(256, spec) - 264.0) < 1e-12,
        f'{storage_bytes(256, spec)}')
    fp16_scale = FP8Spec(scale_bits=16)
    _ok('fp16 scale = 8.5 bits/element',
        abs(bits_per_element(fp16_scale) - 8.5) < 1e-12,
        f'{bits_per_element(fp16_scale)}')
    for kwargs, label in [
        (dict(format='e3m4'), 'unknown format rejected'),
        (dict(format=['e4m3fn']), 'unhashable format rejected'),
        (dict(group_size=0), 'group_size=0 rejected'),
        (dict(scale_bits=-1), 'negative scale_bits rejected'),
        (dict(power_of_two=1), 'non-bool power_of_two rejected'),
    ]:
        try:
            FP8Spec(**kwargs)
            _ok(label, False, 'no error')
        except ValueError:
            _ok(label, True)

    e4 = torch.tensor([0.0, 1.0, 1.0625, 1.1875, 1.3125, 2.0 ** -9,
                       0.0029296875, 0.015625, 448.0, 449.0])
    e4_ref = torch.tensor([0.0, 1.0, 1.0, 1.25, 1.25, 2.0 ** -9,
                           2.0 ** -8, 0.015625, 448.0, 448.0])
    got4 = _fp8_round_magnitude(e4, 'e4m3fn')
    _ok('E4M3FN grid: RNE ties, subnormals, saturation',
        bool(torch.equal(got4, e4_ref)),
        f'{got4.tolist()}')
    e5 = torch.tensor([0.0, 1.0, 1.125, 1.375, 2.0 ** -16, 2.0 ** -17,
                       57344.0, 60000.0])
    e5_ref = torch.tensor([0.0, 1.0, 1.0, 1.5, 2.0 ** -16, 0.0,
                           57344.0, 57344.0])
    got5 = _fp8_round_magnitude(e5, 'e5m2')
    _ok('E5M2 grid: RNE ties, subnormals, finite saturation',
        bool(torch.equal(got5, e5_ref)), f'{got5.tolist()}')
    for fmt, dtype_name in (('e4m3fn', 'float8_e4m3fn'),
                            ('e5m2', 'float8_e5m2')):
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            continue
        grid = torch.linspace(0, 60000 if fmt == 'e5m2' else 500,
                              20001)
        native = grid.to(dtype).float()
        manual = _fp8_round_magnitude(grid, fmt)
        _ok(f'{fmt}: manual grid matches torch native cast',
            bool(torch.equal(manual, native)),
            f'maxdiff={(manual - native).abs().max().item():.2e}')


def test_fp8_fake_quantize_contract():
    print('\nTest: block-scaled FP8 fake-quantize contract')
    torch.manual_seed(17)
    x = torch.randn(4, 128)
    exact = FP8Spec(power_of_two=False)
    y = fake_quantize(x, exact, ste=False)
    err = (x - y).abs()
    # E4M3FN normal relative step is 2**-3, so half-ULP <= |x|/16; the
    # min_normal/subnormal branch has a fixed step of 2**-9.
    bound = x.abs() / 16.0 + 2.0 ** -9
    _ok('E4M3FN exact-scale error <= half-ULP bound',
        bool((err <= bound + 1e-6).all()),
        f'max|err|={float(err.max()):.3e}')
    stats = quantization_error(x, exact)
    _ok('E4M3FN fp8 SNR is positive', stats['snr_db'] > 0,
        f"snr={stats['snr_db']:.1f} dB")
    const = torch.full((2, 37), 0.7)
    dqc = fake_quantize(const, FP8Spec(power_of_two=False), ste=False)
    _ok('constant groups reconstruct exactly with an exact scale',
        bool(torch.allclose(dqc, const, atol=1e-6)),
        f'max|err|={(dqc - const).abs().max().item():.2e}')
    _ok('all-zero FP8 groups are exactly zero',
        bool((fake_quantize(torch.zeros(2, 40), FP8Spec()) == 0).all()))
    pow2 = FP8Spec()
    yp = fake_quantize(x, pow2, ste=False)
    _ok('power-of-two scale stays finite and bounded',
        bool(torch.isfinite(yp).all())
        and float(yp.abs().max()) <= pow2.max_finite, '')
    # Finite inputs near FLT_MAX must not dequantize to +-inf just because
    # the power-of-two E8M0 scale rounds up past the dtype range.
    extreme = torch.tensor([[3.4e38, -3.4e38, 0.0, 1.7e38]])
    for fmt in ('e4m3fn', 'e5m2'):
        dq_extreme = fake_quantize(extreme, FP8Spec(format=fmt), ste=False)
        _ok(f'extreme finite input stays finite ({fmt})',
            bool(torch.isfinite(dq_extreme).all()),
            f'{dq_extreme.tolist()}')
    bad = x.clone()
    bad[0, 0] = float('inf')
    try:
        fake_quantize(bad, FP8Spec())
        _ok('non-finite FP8 input rejected', False, 'no error')
    except ValueError:
        _ok('non-finite FP8 input rejected', True)
    g = x.clone().requires_grad_(True)
    fake_quantize(g, FP8Spec(), ste=True).sum().backward()
    _ok('FP8 STE gradient is exactly 1',
        bool((g.grad == 1.0).all()), f'{float(g.grad.mean()):.3f}')
    g2 = x.clone().requires_grad_(True)
    fake_quantize(g2, FP8Spec(), ste=False).sum().backward()
    _ok('FP8 ste=False exposes the zero-gradient quantizer',
        bool((g2.grad == 0.0).all()), '')
    dq16 = fake_quantize(x.half(), FP8Spec())
    _ok('FP8 path preserves fp16 input dtype',
        dq16.dtype == torch.float16, f'{dq16.dtype}')
    row = kv_budget.op_read_elements('kda', 16, _MQAR_SPEC)
    byte_row = quantized_row_bytes(row, {'state': FP8Spec()})
    _ok('FP8 byte bridge prices the KDA state at 1.03125 B/elem',
        abs(byte_row['state_bytes'] - 32 * 1.03125) < 1e-12,
        f"{byte_row['state_bytes']}")


def test_fp8_fp64_dynamic_range():
    print('\nTest: FP8 fp64 dynamic range')
    for value in (1e300, 1e-300, 5e-324):
        x = torch.tensor([value], dtype=torch.float64)
        dq = fake_quantize(x, FP8Spec(), ste=False)
        ratio = (dq / x).item()
        _ok(f'fp64 {value:.1e}: finite and inside the power-of-two band',
            bool(torch.isfinite(dq).all()) and 0.25 <= ratio <= 2.0,
            f'ratio={ratio:.3e}')


def test_fp8_state_recall_probe():
    print('\nTest: FP8 state on the planted-recall probe')
    keys, values = planted_batch(16, 8, seed=0, device='cpu')
    for route in ('kda', 'mamba2'):
        fp16 = run_planted_recall(route, keys, values, None)
        mx = run_planted_recall(route, keys, values, FP8Spec())
        i8 = run_planted_recall(route, keys, values, QuantSpec(8, 64))
        _ok(f'{route}: MXFP8 state keeps exact recall (pinned seed)',
            mx['readback_recall'] == 1.0,
            f"recall={mx['readback_recall']}, "
            f"err={mx['max_read_error']:.3f}")
        _ok(f'{route}: MXFP8 read-back error exceeds int8 (coarser mantissa)',
            mx['max_read_error'] > i8['max_read_error'] > 0.0,
            f"fp8={mx['max_read_error']:.3f}, "
            f"int8={i8['max_read_error']:.3f}")
        _ok(f'{route}: FP8 recall does not exceed the fp16 reference',
            mx['readback_recall'] <= fp16['readback_recall'], '')
    try:
        run_planted_recall('kda', keys, values, 'fp8')
        _ok('non-spec FP8 argument rejected', False, 'no error')
    except TypeError:
        _ok('non-spec FP8 argument rejected', True)


_TEST_FUNCS = [
    test_quant_spec_accounting,
    test_quantized_row_bytes_bridge,
    test_fake_quantize_contract,
    test_quantized_state_recall_probe,
    test_quant_recall_report_rows,
    test_fp8_spec_and_grid,
    test_fp8_fake_quantize_contract,
    test_fp8_fp64_dynamic_range,
    test_fp8_state_recall_probe,
]


def main() -> int:
    print('=' * 70)
    print('kv_quant / Exp 10 (quantized-state recall) Regression Tests')
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


def test_storage_bytes_charges_one_metadata_block_per_whole_or_partial_group():
    """Integer layouts pay ceil(n/group_size) scales, not a linear average."""
    spec = QuantSpec(2, 64, scale_bits=16)
    # 1 element: 2 payload bits + one 16-bit scale = 18 bits = 2.25 B
    assert abs(storage_bytes(1, spec) - 2.25) < 1e-12
    assert abs(storage_bytes(63, spec) - (63 * 2 + 16) / 8.0) < 1e-12
    assert abs(storage_bytes(64, spec) - 18.0) < 1e-12
    assert abs(storage_bytes(65, spec) - (65 * 2 + 2 * 16) / 8.0) < 1e-12
    assert abs(storage_bytes(256, spec) - 72.0) < 1e-12
    fp8 = FP8Spec()
    assert abs(storage_bytes(1, fp8) - 2.0) < 1e-12
    assert abs(storage_bytes(256, fp8) - 264.0) < 1e-12
    # Fractional counts keep the effective-rate convention.
    assert abs(storage_bytes(1.5, spec)
               - 1.5 * bytes_per_element(spec)) < 1e-12


def test_quantization_extremes_stay_finite_and_json_safe():
    import json
    x = torch.tensor([[-3.4e38, 3.4e38]], dtype=torch.float32)
    spec = QuantSpec(bits=8, group_size=2, symmetric=False)
    y = fake_quantize(x, spec, ste=False)
    assert bool(torch.isfinite(y).all())
    stats = quantization_error(x, spec)
    assert stats['max_abs_error'] is not None
    assert stats['rms_error'] is not None
    assert stats['signal_rms'] is not None
    json.dumps(stats, allow_nan=False)

    n = 5 * 10 ** 307
    assert storage_bytes(n, QuantSpec(bits=4, group_size=64)) == 2.65625e307
