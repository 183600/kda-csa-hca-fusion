"""Regression tests for ``results_schema.py`` and ``roofline.py`` (Exp 7).

The tests are self-contained and never touch the real ``results/``
directory: roundtrip tests write to a per-test temp directory.

Run via pytest::

    pytest -q test_roofline_schema.py

Or directly::

    python3 test_roofline_schema.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import results_schema  # noqa: E402
import roofline  # noqa: E402
from run_kv_cache import prefill_flops  # noqa: E402
from run_roofline import _account_row  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# --- results_schema -------------------------------------------------------

def test_schema_detects_legacy_and_current_versions():
    print("\nTest: results_schema version detection")
    bare = [{'op': 'x'}]
    _ok('bare array -> 0.0',
        results_schema.detect_version(bare) == '0.0'
        and results_schema.is_compatible('0.0'), 'legacy list')
    legacy = {'metadata': {'schema_version': 1}, 'results': [{'op': 'x'}]}
    _ok('legacy envelope integer version -> 1.0',
        results_schema.detect_version(legacy) == '1.0'
        and results_schema.is_compatible('1.0'), 'exp4-style envelope')
    current = results_schema.envelope([{'op': 'x'}], kind='k')
    _ok('current envelope version',
        results_schema.detect_version(current)
        == results_schema.CURRENT_SCHEMA_VERSION, 'versioned envelope')


def test_schema_rejects_non_mapping_metadata():
    print("\nTest: results_schema validates envelope metadata type")
    for bad in ([], ['note', 'n'], 0, '', 1.5, True):
        for build in (
                lambda: results_schema.envelope([], kind='k', metadata=bad),
                lambda: results_schema.normalize(
                    {'results': []}, kind='k', metadata=bad)):
            raised = False
            try:
                build()
            except results_schema.SchemaFormatError:
                raised = True
            _ok(f'non-mapping metadata {bad!r} raises', raised, '')
    # A Mapping (not just a dict) and None remain accepted.
    from collections import UserDict
    assert results_schema.envelope(
        [], kind='k', metadata=UserDict({'x': 1}))['metadata']['x'] == 1
    assert results_schema.envelope([], kind='k', metadata=None)['metadata']
    _ok('Mapping/None metadata accepted', True, '')


def test_schema_versioned_envelope_roundtrip():
    print("\nTest: results_schema write/load roundtrip")
    with tempfile.TemporaryDirectory(prefix='_schema_test_') as tmp:
        path = os.path.join(tmp, 'exp_test.json')
        rows = [{'op': 'softmax', 'T': 4, 'value': 1.25}]
        results_schema.write(path, rows, kind='test_kind',
                             metadata={'note': 'n', 'schema_version': '9.9'})
        meta, loaded = results_schema.load(path)
        with open(path, encoding='utf-8') as f:
            on_disk = json.load(f)
        _ok('rows roundtrip', loaded == rows, f'{loaded}')
        _ok('metadata roundtrip (note kept)',
            meta.get('note') == 'n', f'{meta}')
        _ok('stale metadata.schema_version stripped',
            'schema_version' not in meta, f'{meta}')
        _ok('envelope carries kind + top-level version',
            on_disk.get('kind') == 'test_kind'
            and on_disk.get('schema_version')
            == results_schema.CURRENT_SCHEMA_VERSION, f'{on_disk.keys()}')
        _ok('created_utc recorded',
            isinstance(on_disk.get('created_utc'), str)
            and on_disk['created_utc'], f"{on_disk.get('created_utc')}")


def test_schema_future_version_is_refused():
    print("\nTest: results_schema refuses future major versions")
    future = {'schema_version': '2.0', 'metadata': {}, 'results': []}
    raised = False
    try:
        results_schema.parse(future)
    except results_schema.SchemaVersionError:
        raised = True
    _ok('parse raises SchemaVersionError', raised, 'future major')
    with tempfile.TemporaryDirectory(prefix='_schema_test_') as tmp:
        path = os.path.join(tmp, 'future.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(future, f)
        meta, rows = results_schema.load_or_empty(path)
        _ok('load_or_empty degrades to empty', (meta, rows) == ({}, []),
            f'{meta}, {rows}')
    malformed = {'metadata': {}, 'results': 'not-a-list'}
    raised = False
    try:
        results_schema.parse(malformed)
    except results_schema.SchemaFormatError:
        raised = True
    _ok('non-list results raises SchemaFormatError', raised, 'malformed')


def test_load_or_empty_degrades_on_io_and_decode_failures():
    print("\nTest: load_or_empty tolerates unreadable and non-UTF-8 paths")
    with tempfile.TemporaryDirectory(prefix='_schema_test_') as tmp:
        directory = os.path.join(tmp, 'results_dir')
        os.mkdir(directory)
        meta, rows = results_schema.load_or_empty(directory)
        _ok('directory path degrades to empty', (meta, rows) == ({}, []),
            f'{meta}, {rows}')

        binary = os.path.join(tmp, 'binary.json')
        with open(binary, 'wb') as f:
            f.write(b'\xff\xfe{}')
        meta, rows = results_schema.load_or_empty(binary)
        _ok('non-UTF-8 file degrades to empty', (meta, rows) == ({}, []),
            f'{meta}, {rows}')


# --- roofline math --------------------------------------------------------

def test_roofline_core_math():
    print("\nTest: roofline core FLOPs / bytes")
    # Causal entries for T=4: 4*5/2 = 10; FLOPs = 2*entries*H*(K+V).
    sm = roofline.softmax_core_flops(4, 2, 8)
    _ok('softmax core FLOPs', abs(sm - 2 * 10 * 2 * 16) < 1e-9, f'{sm}')
    kda = roofline.kda_core_flops(4, 2, 8, 8)
    _ok('KDA recurrent core FLOPs',
        abs(kda - 6 * 4 * 2 * 8 * 8) < 1e-9, f'{kda}')
    kda_c = roofline.kda_core_flops(64, 1, 8, 8, chunked=True,
                                    chunk_size=64)
    _ok('KDA chunked core adds inter-chunk terms',
        abs(kda_c - (6 * 64 * 1 * 64 + 3 * 64 * 1 * 64 * 8
                     + 64 * 1 * 64 * 64)) < 1e-9, f'{kda_c}')
    # KDA bytes: activations 3K+2V+1 per token-head plus 2*T*H*K*V state,
    # all fp32.
    cb = roofline.core_bytes('kda', 4, heads=2, head_dim=8, value_dim=8)
    expected_cb = (4 * 2 * (3 * 8 + 2 * 8 + 1) + 2 * 4 * 2 * 8 * 8) * 4
    _ok('KDA core bytes include state traffic',
        abs(cb - expected_cb) < 1e-9, f'{cb} vs {expected_cb}')


def test_roofline_point_classification():
    print("\nTest: roofline point classification and ceiling")
    # ridge = 1e12 / (100e9) = 10 FLOP/byte.
    mem_pt = roofline.roofline_point(
        flops=1e9, nbytes=1e9, time_ms=20.0,
        peak_flops_per_s=1e12, peak_bw_gbps=100.0)
    _ok('memory-bound classification + ceiling',
        mem_pt['roofline_bound'] == 'memory'
        and abs(mem_pt['roofline_ceiling_ms'] - 10.0) < 1e-6
        and abs(mem_pt['pct_of_roofline'] - 50.0) < 1e-6
        and mem_pt['efficiency_label'] == 'near-roofline', f'{mem_pt}')
    cmp_pt = roofline.roofline_point(
        flops=2e9, nbytes=1e8, time_ms=4.0,
        peak_flops_per_s=1e12, peak_bw_gbps=100.0)
    _ok('compute-bound classification + ceiling',
        cmp_pt['roofline_bound'] == 'compute'
        and abs(cmp_pt['roofline_ceiling_ms'] - 2.0) < 1e-6, f'{cmp_pt}')
    no_peak = roofline.roofline_point(flops=1e6, nbytes=1e5, time_ms=1.0)
    _ok('achieved fields without machine peaks',
        no_peak['achieved_tflops'] is not None
        and 'roofline_bound' not in no_peak, f'{no_peak}')
    _ok('efficiency labels',
        roofline.efficiency_label(75) == 'near-roofline'
        and roofline.efficiency_label(20) == 'moderate'
        and roofline.efficiency_label(1) == 'overhead-bound'
        and roofline.efficiency_label(None) == 'unknown', '')


def test_roofline_device_presets():
    print("\nTest: roofline device preset lookup")
    a100 = roofline.spec_for_device('NVIDIA A100-SXM4-80GB')
    _ok('A100 matched by substring',
        a100 is not None and a100['peak_bw_gbps'] == 2039.0, f'{a100}')
    _ok('longest key wins (RTX 4090)',
        roofline.spec_for_device('NVIDIA GeForce RTX 4090')
        ['peak_tflops'] == 82.6, 'rtx 4090')
    _ok('unknown device -> None',
        roofline.spec_for_device('Mystery Accelerator 9000') is None, '')


def test_roofline_host_microbenchmarks_smoke():
    print("\nTest: on-host bandwidth / GEMM measurement smoke")
    bw = roofline.measure_copy_bandwidth_gbps('cpu', size_mb=8, repeats=2)
    _ok('copy bandwidth measured', bw is not None and math.isfinite(bw)
        and bw > 0, f'{bw} GB/s')
    gemm = roofline.measure_gemm_tflops('cpu', n=128, repeats=2)
    _ok('fp32 GEMM throughput measured',
        gemm is not None and math.isfinite(gemm) and gemm > 0,
        f'{gemm} TFLOP/s')


# --- Exp 7 join -----------------------------------------------------------

def test_account_row_matches_prefill_flops():
    print("\nTest: run_roofline row accounting matches run_kv_cache")
    csa_row = {
        'op': 'csa', 'T': 128, 'time_ms': 1.0, 'param_elements': 11200,
        'workload': {'d': 64, 'm': 8, 'topk': 4, 'sliding_window': 8,
                     'heads': 2, 'c': 16, 'dc': 32, 'nIh': 2, 'cI': 8},
    }
    acct = _account_row(csa_row)
    geo = dict(d=64, csa_m=8, csa_topk=4, csa_nh=2, csa_c=16, csa_dc=32,
               csa_nIh=2, csa_cI=8, csa_sliding_window=8)
    _ok('CSA FLOPs == prefill_flops at benchmark geometry',
        acct['flops'] == prefill_flops('csa', 128, **geo),
        f"{acct['flops']}")
    _ok('weights term included from param_elements',
        acct['weights_included'] and acct['bytes'] >= 11200 * 4,
        f"{acct['bytes']} bytes")
    # Missing geometry keys (old benchmark revisions) must raise, so
    # run_roofline can report the row as skipped instead of guessing
    # paper-scale defaults.
    raised = False
    try:
        _account_row({'op': 'hybrid', 'T': 128, 'time_ms': 1.0,
                      'param_elements': None,
                      'workload': {'n_layers': 5}})
    except KeyError:
        raised = True
    _ok('incomplete hybrid workload raises KeyError', raised, '')


def test_roofline_rejects_unrepresentable_geometry():
    """JSON-sized ints are malformed geometry, not ``OverflowError``."""
    huge = 10 ** 400
    cases = [
        ('softmax_core_flops', lambda: roofline.softmax_core_flops(
            huge, 2, 8)),
        ('softmax heads', lambda: roofline.softmax_core_flops(
            4, huge, 8)),
        ('kda_core_flops', lambda: roofline.kda_core_flops(
            huge, 2, 8, 8)),
        ('kda chunk_size', lambda: roofline.kda_core_flops(
            4, 2, 8, 8, chunked=True, chunk_size=huge)),
        ('ssd_core_flops', lambda: roofline.ssd_core_flops(
            huge, 2, 8, 4)),
        ('ssd chunk_size', lambda: roofline.ssd_core_flops(
            8, 2, 8, 4, chunked=True, chunk_size=huge)),
        ('arithmetic_intensity', lambda: roofline.arithmetic_intensity(
            huge, 1.0)),
        ('ridge_point', lambda: roofline.ridge_point(huge, 100.0)),
        ('roofline_point', lambda: roofline.roofline_point(
            flops=1.0, nbytes=1.0, time_ms=huge)),
        ('account_row hca', lambda: _account_row({
            'op': 'hca', 'T': huge, 'param_elements': None,
            'workload': {'d': 64, 'm2': 8, 'heads': 2, 'c': 16,
                         'dc': 32, 'sliding_window': 8}})),
    ]
    for name, call in cases:
        raised = False
        try:
            call()
        except ValueError:
            raised = True
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'{name} rejects an oversized geometry', False,
                f'{type(exc).__name__}: {exc}')
            continue
        _ok(f'{name} rejects an oversized geometry', raised, '')


_TEST_FUNCS = [
    test_schema_detects_legacy_and_current_versions,
    test_schema_rejects_non_mapping_metadata,
    test_schema_versioned_envelope_roundtrip,
    test_schema_future_version_is_refused,
    test_roofline_core_math,
    test_roofline_rejects_unrepresentable_geometry,
    test_roofline_point_classification,
    test_roofline_device_presets,
    test_roofline_host_microbenchmarks_smoke,
    test_account_row_matches_prefill_flops,
]


def main() -> int:
    print('=' * 70)
    print('results_schema / roofline Regression Tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: AssertionError: {e}")
            results.append(False)
        except Exception as e:
            print(f"  [CRASH] {fn.__name__}: {type(e).__name__}: {e}")
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
