"""Regression tests for the SSD-scan latency profiler and FLOPs model.

Covers:

* ``roofline.ssd_core_flops`` / ``ssd_core_flops_breakdown`` /
  ``core_bytes('ssd', ...)`` — analytic accounting for the Mamba-2 scan in
  both shipped modes (sequential recurrence and chunked SSD).
* ``latency_profile`` — the measured counterpart: seeded inputs, warmup +
  repeated timing with injectable clock/runner, per-row parity gating
  against the sequential reference, measured-vs-model optimal chunk size,
  and the optional JSON envelope.

The harness mechanics are tested with a FAKE clock and FAKE runner so the
assertions are exact and never wall-clock flaky; one end-to-end real-torch
profile at tiny T pins the integration (inputs, ops dispatch, parity,
schema write/read round-trip).

Run via pytest::

    pytest -q test_latency_profile.py

Or directly::

    python3 test_latency_profile.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import latency_profile as lp  # noqa: E402
import results_schema  # noqa: E402
import roofline  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# --- analytic model -------------------------------------------------------

def test_ssd_core_flops_sequential_formula():
    print('\nTest: ssd_core_flops sequential formula and validation')
    val = roofline.ssd_core_flops(8, 2, 16, 4)
    _ok('sequential = 6*T*H*D*N', abs(val - 6 * 8 * 2 * 16 * 4) < 1e-9, f'{val}')
    _ok('T=0 -> 0 FLOPs', roofline.ssd_core_flops(0, 2, 16, 4) == 0.0, '')
    for bad in ((-1, 2, 16, 4), (8, 0, 16, 4), (8, 2, 0, 4), (8, 2, 16, 0),
                (8.0, 2, 16, 4), (True, 2, 16, 4), (8, 2, 16, False)):
        raised = False
        try:
            roofline.ssd_core_flops(*bad)
        except (TypeError, ValueError):
            raised = True
        _ok(f'rejects geometry {bad!r}', raised, '')
    raised = False
    try:
        roofline.ssd_core_flops(8, 2, 16, 4, chunked=True, chunk_size=0)
    except ValueError:
        raised = True
    _ok('rejects chunk_size=0', raised, '')


def test_ssd_core_flops_breakdown_exact():
    print('\nTest: chunked breakdown hand-computed phases')
    b = roofline.ssd_core_flops_breakdown(4, 1, 2, 2, chunk_size=2)
    # NT=2, Tp=4, levels=1:
    #   intra = 2*4*2*1*(2+2) = 64; source = readout = 2*4*1*2*2 = 32;
    #   scan  = 2*1*4*1*2*2/2 = 16; total = 144.
    _ok('intra_chunk_attention = 64',
        abs(b['intra_chunk_attention'] - 64.0) < 1e-9, f"{b['intra_chunk_attention']}")
    _ok('chunk_source = 32', abs(b['chunk_source'] - 32.0) < 1e-9,
        f"{b['chunk_source']}")
    _ok('inter_chunk_readout = 32',
        abs(b['inter_chunk_readout'] - 32.0) < 1e-9, f"{b['inter_chunk_readout']}")
    _ok('assoc_scan = 16', abs(b['assoc_scan'] - 16.0) < 1e-9,
        f"{b['assoc_scan']}")
    _ok('total = 144', abs(b['total'] - 144.0) < 1e-9, f"{b['total']}")
    _ok('n_chunks / padded_len / levels', 
        (b['n_chunks'], b['padded_len'], b['scan_levels']) == (2, 4, 1),
        f"{b['n_chunks']},{b['padded_len']},{b['scan_levels']}")
    _ok('ssd_core_flops(chunked) == breakdown total',
        roofline.ssd_core_flops(4, 1, 2, 2, chunked=True, chunk_size=2)
        == b['total'], '')
    # One chunk over the whole sequence: scan level 0, global quadratic term.
    one = roofline.ssd_core_flops_breakdown(4, 1, 2, 2, chunk_size=4)
    _ok('C=T degenerates to a single chunk',
        one['n_chunks'] == 1 and one['scan_levels'] == 0
        and abs(one['intra_chunk_attention'] - 128.0) < 1e-9, f"{one}")
    zero = roofline.ssd_core_flops_breakdown(0, 1, 2, 2, chunk_size=4)
    _ok('T=0 breakdown is all zeros', zero['total'] == 0.0, f"{zero['total']}")


def test_ssd_core_flops_chunk_grid_and_model_best():
    print('\nTest: chunk-size grid, model argmin, tie-break')
    T, H, D, N = 32, 2, 8, 4
    candidates = (4, 8, 16, 32)
    costs = [roofline.ssd_core_flops(T, H, D, N, chunked=True, chunk_size=C)
             for C in candidates]
    best = lp.model_best_chunk_size(T, heads=H, headdim=D, d_state=N,
                                    candidates=candidates)
    _ok('model argmin attains the grid minimum',
        costs[candidates.index(best)] == min(costs), f'{best} {costs}')
    _ok('argmin is deterministic',
        best == lp.model_best_chunk_size(T, heads=H, headdim=D, d_state=N,
                                         candidates=candidates), '')
    # T=0 makes every chunk size tie at 0 FLOPs -> smallest candidate wins.
    _ok('tie-break picks the smaller chunk size',
        lp.model_best_chunk_size(0, heads=H, headdim=D, d_state=N,
                                 candidates=(8, 16)) == 8, '')
    raised = False
    try:
        lp.model_best_chunk_size(T, heads=H, headdim=D, d_state=N,
                                 candidates=())
    except ValueError:
        raised = True
    _ok('empty candidate set rejected', raised, '')
    # More chunks must monotonically shrink the associative-scan share.
    scan_lo = roofline.ssd_core_flops_breakdown(T, H, D, N,
                                                chunk_size=4)['assoc_scan']
    scan_hi = roofline.ssd_core_flops_breakdown(T, H, D, N,
                                                chunk_size=32)['assoc_scan']
    _ok('assoc_scan cost shrinks with chunk size', scan_lo > scan_hi >= 0.0,
        f'{scan_lo} > {scan_hi}')


def test_core_bytes_ssd_accounting():
    print('\nTest: core_bytes ssd accounting')
    seq = roofline.core_bytes('ssd', 4, heads=2, head_dim=8, d_state=4,
                              ngroups=1)
    # activation = 2*4*2*8 (x+y) + 4*2 (dt) + 2*4*1*4 (B+C) = 168
    # state      = 2*4*2*8*4 = 512 -> (168+512) * 4 bytes = 2720
    _ok('sequential ssd bytes hand-computed', abs(seq - 2720.0) < 1e-9,
        f'{seq}')
    ch = roofline.core_bytes('ssd', 4, heads=2, head_dim=8, d_state=4,
                             ngroups=1, chunked=True, chunk_size=2)
    # NT=2 -> state = 2*2*2*8*4 = 256 -> (168+256) * 4 = 1696
    _ok('chunked ssd state shrinks per chunk', abs(ch - 1696.0) < 1e-9,
        f'{ch}')
    groups = roofline.core_bytes('ssd', 4, heads=2, head_dim=8, d_state=4,
                                 ngroups=2)
    _ok('ngroups scales the B/C term', abs(groups - (2720.0 + 32.0 * 4)) < 1e-9,
        f'{groups}')
    raised = False
    try:
        roofline.core_bytes('ssd', 4, heads=2, head_dim=8)
    except ValueError:
        raised = True
    _ok('ssd without d_state rejected', raised, '')
    raised = False
    try:
        roofline.core_bytes('nope', 4, heads=2, head_dim=8)
    except ValueError:
        raised = True
    _ok('unknown op rejected', raised, '')


# --- run_roofline integration --------------------------------------------

def test_run_roofline_accounts_mamba_route():
    print('\nTest: run_roofline accounts every Mamba-2 route row')
    from run_kv_cache import kv_cache_elements, prefill_flops
    from run_roofline import _account_row, _hybrid_mamba2_geometry

    T = 128
    # mamba2_scan: core boundary, sequential recurrence by default.
    scan_w = {'heads': 4, 'headdim': 32, 'd_state': 32}
    acct = _account_row({'op': 'mamba2_scan', 'T': T, 'time_ms': 1.0,
                         'workload': scan_w})
    _ok('mamba2_scan boundary is core',
        acct['analytic_boundary'] == 'core', acct['analytic_boundary'])
    _ok('mamba2_scan flops use the SSD scan model',
        acct['flops'] == roofline.ssd_core_flops(T, 4, 32, 32), f"{acct['flops']}")
    _ok('mamba2_scan bytes use the SSD compulsory-traffic model',
        acct['bytes'] == roofline.core_bytes(
            'ssd', T, heads=4, head_dim=32, d_state=32, ngroups=1),
        f"{acct['bytes']}")
    _ok('mamba2_scan core rows carry no weight traffic',
        acct['weights_included'] is False, '')
    chunked_acct = _account_row({
        'op': 'mamba2_scan', 'T': T, 'time_ms': 1.0,
        'workload': {**scan_w, 'mamba_chunk_size': 32}})
    _ok('a recorded chunk knob switches the scan accounting to chunked',
        chunked_acct['flops'] == roofline.ssd_core_flops(
            T, 4, 32, 32, chunked=True, chunk_size=32)
        and chunked_acct['bytes'] < acct['bytes'],
        f"{chunked_acct['flops']} vs {acct['flops']}")
    # The dedicated benchmark row must take the same core-boundary path,
    # not fall through to the end-to-end run_kv_cache accounting.
    row_chunked = _account_row({
        'op': 'mamba2_chunked_scan', 'T': T, 'time_ms': 1.0,
        'workload': {**scan_w, 'mamba_chunk_size': 32}})
    _ok('mamba2_chunked_scan boundary is core',
        row_chunked['analytic_boundary'] == 'core',
        row_chunked['analytic_boundary'])
    _ok('mamba2_chunked_scan row prices the chunked decomposition',
        row_chunked['flops'] == chunked_acct['flops']
        and row_chunked['bytes'] == chunked_acct['bytes'],
        f"{row_chunked['flops']} vs {chunked_acct['flops']}")

    # mamba2_layer: end-to-end single mixer at the RECORDED realization.
    layer_w = {'d': 64, 'expand': 2, 'd_state': 16, 'd_conv': 4, 'headdim': 32}
    layer_params = 39000
    acct_l = _account_row({'op': 'mamba2_layer', 'T': T, 'time_ms': 1.0,
                           'param_elements': layer_params, 'workload': layer_w})
    geo_l = dict(d=64, mamba_expand=2, mamba_d_state=16, mamba_d_conv=4,
                 mamba_headdim=32)
    _ok('mamba2_layer flops == mixer prefill_flops at recorded geometry',
        acct_l['flops'] == float(prefill_flops('mamba2', T, **geo_l)),
        f"{acct_l['flops']}")
    _ok('mamba2_layer bytes include weights + fixed SSM state',
        acct_l['bytes'] == roofline.elements_to_bytes(
            2 * T * 64 + kv_cache_elements(
                'mamba2', T, mode='compressed_kv_only', **geo_l) + layer_params)
        and acct_l['weights_included'], f"{acct_l['bytes']}")

    # hybrid_mamba2: accounted at the recorded Mamba geometry (its schema
    # has n_memory and no heads_qk/n_kda, so the join must not fall back to
    # paper-scale defaults).
    hybrid_w = {
        'n_layers': 5, 'memory': 'mamba2', 'n_memory': 3, 'n_csa': 1,
        'n_hca': 1, 'd': 64, 'csa_c': 16, 'csa_dc': 32, 'csa_nh': 2,
        'csa_m': 8, 'csa_topk': 4, 'csa_sliding_window': 8,
        'csa_nIh': 2, 'csa_cI': 8, 'hca_c': 16,
        'hca_dc': 32, 'hca_nh': 2, 'hca_m2': 16, 'hca_sliding_window': 8,
        'mamba_expand': 2, 'mamba_d_state': 16, 'mamba_d_conv': 4,
        'mamba_headdim': 32,
    }
    acct_h = _account_row({'op': 'hybrid_mamba2', 'T': T, 'time_ms': 1.0,
                           'param_elements': 100000, 'workload': hybrid_w})
    geo_h = _hybrid_mamba2_geometry(hybrid_w)
    _ok('hybrid_mamba2 flops at the recorded geometry',
        acct_h['flops'] == float(prefill_flops('hybrid_mamba2', T, **geo_h)),
        f"{acct_h['flops']}")
    paper_scale = float(prefill_flops('hybrid_mamba2', T, **{
        k: v for k, v in geo_h.items() if not k.startswith('mamba_')}))
    _ok('defaults over-account (the geometry forwarding is load-bearing)',
        paper_scale > acct_h['flops'],
        f'{paper_scale} vs {acct_h["flops"]}')
    _ok('hybrid_mamba2 byte model includes the recorded Mamba state',
        acct_h['bytes'] == roofline.elements_to_bytes(
            2 * T * 64 + kv_cache_elements(
                'hybrid_mamba2', T, mode='compressed_kv_only', **geo_h)
            + 100000), f"{acct_h['bytes']}")


# --- harness mechanics (fake clock / fake runner) -------------------------

def test_time_call_fake_clock():
    print('\nTest: time_call mechanics with an injected clock')
    ticks = iter([0.0, 0.001, 10.0, 10.002, 20.0, 20.001])

    def fake_timer():
        return next(ticks)

    calls = {'n': 0}

    def fn():
        calls['n'] += 1

    stats = lp.time_call(fn, warmup=2, repeats=3, timer=fake_timer)
    _ok('warmup runs are executed but not timed', calls['n'] == 5,
        f"calls={calls['n']}")
    times = [1.0, 2.0, 1.0]
    _ok('median/min/max in ms',
        abs(stats['median_ms'] - statistics.median(times)) < 1e-9
        and abs(stats['min_ms'] - 1.0) < 1e-9
        and abs(stats['max_ms'] - 2.0) < 1e-9, f'{stats}')
    _ok('std matches statistics.stdev',
        abs(stats['std_ms'] - statistics.stdev(times)) < 1e-6,
        f"{stats['std_ms']}")
    _ok('per-repeat times recorded',
        all(abs(a - b) < 1e-9 for a, b in zip(stats['times_ms'], times)),
        f"{stats['times_ms']}")
    for kwargs in ({'repeats': 0}, {'warmup': -1}):
        raised = False
        try:
            lp.time_call(lambda: None, **kwargs)
        except ValueError:
            raised = True
        _ok(f'rejects {kwargs}', raised, '')


def test_time_call_synchronize_hook():
    print('\nTest: synchronize hook is called around every timed call')
    events = []
    lp.time_call(lambda: events.append('fn'), warmup=1, repeats=2,
                 timer=lambda: 0.0,
                 synchronize=lambda: events.append('sync'))
    # Warmup fn is untimed; each timed call is bracketed by one sync before
    # the clock read and one after the call.
    _ok('sync bracketed the timed calls',
        events == ['fn', 'sync', 'fn', 'sync', 'sync', 'fn', 'sync'],
        f'{events}')


def _fake_inputs(T=8):
    return {'x': torch.zeros(1, T, 1, 1), 'dt': torch.ones(1, T, 1)}


def _fake_runner(corrupt=None):
    """Deterministic fake: zeros for every mode; optional one-cell offset."""
    def runner(mode, inputs, chunk_size):
        x = inputs['x']
        y = torch.zeros_like(x)
        state = torch.zeros(1, 1, 1, 1)
        if corrupt is not None:
            c_mode, c_chunk, delta = corrupt
            if mode == c_mode and chunk_size == c_chunk:
                y = y + delta
        return y, state
    return runner


def _fake_timer_factory(step=0.001):
    box = {'t': 0.0}

    def timer():
        t = box['t']
        box['t'] += step
        return t
    return timer


def test_profile_fake_runner_and_rows():
    print('\nTest: profile_scan_modes row schema with fake runner')
    profile = lp.profile_scan_modes(
        (8, 16), chunk_sizes=(4, 8), geometry={'heads': 2, 'headdim': 4,
                                               'd_state': 2, 'ngroups': 1},
        warmup=1, repeats=2, timer=_fake_timer_factory(), runner=_fake_runner(),
        include_quadratic=True, max_quadratic_len=16)
    rows = profile['rows']
    # 2 lengths x (sequential + 2 chunked + quadratic) = 8 rows.
    _ok('row count covers every mode', len(rows) == 8, f'{len(rows)}')
    seq = [r for r in rows if r['mode'] == 'sequential']
    chunked = [r for r in rows if r['mode'] == 'chunked']
    quad = [r for r in rows if r['mode'] == 'quadratic']
    _ok('sequential/ chunked / quadratic rows present',
        len(seq) == 2 and len(chunked) == 4 and len(quad) == 2,
        f'{len(seq)},{len(chunked)},{len(quad)}')
    _ok('all parity gates pass on identical outputs',
        all(r['parity_ok'] for r in rows), '')
    _ok('sequential row parity anchor is 0.0',
        all(r['parity_max_abs'] == 0.0 for r in seq), '')
    row = next(r for r in chunked if r['T'] == 16 and r['chunk_size'] == 8)
    _ok('chunked row records NT and levels',
        row['n_chunks'] == 2 and row['scan_levels'] == 1,
        f"{row['n_chunks']},{row['scan_levels']}")
    expected_flops = roofline.ssd_core_flops(
        16, 2, 4, 2, chunked=True, chunk_size=8)
    _ok('analytic_flops matches roofline model',
        row['analytic_flops'] == expected_flops, '')
    _ok('achieved_gflops == flops / ms * 1e-6',
        abs(row['achieved_gflops'] - row['analytic_flops']
            / row['time_ms'] * 1e-6) < 1e-12,
        f"{row['achieved_gflops']}")
    _ok('model best comes from the candidate grid',
        profile['model_best_chunk_size'][16] in (4, 8), '')
    _ok('measured best comes from the candidate grid',
        profile['measured_best_chunk_size'][16] in (4, 8), '')
    _ok('quadratic flops collapse to the C=T model',
        quad[0]['analytic_flops'] == roofline.ssd_core_flops(
            8, 2, 4, 2, chunked=True, chunk_size=8), '')


def test_profile_parity_guard_detects_divergence():
    print('\nTest: parity gate flags a divergent chunked row')
    profile = lp.profile_scan_modes(
        (8,), chunk_sizes=(4, 8), geometry={'heads': 2, 'headdim': 4,
                                            'd_state': 2, 'ngroups': 1},
        warmup=0, repeats=1, timer=_fake_timer_factory(),
        runner=_fake_runner(corrupt=('chunked', 4, 0.5)))
    bad = next(r for r in profile['rows'] if r['chunk_size'] == 4)
    good = next(r for r in profile['rows'] if r['chunk_size'] == 8)
    _ok('divergent row fails the parity gate',
        not bad['parity_ok'] and abs(bad['parity_max_abs'] - 0.5) < 1e-12,
        f"{bad['parity_ok']},{bad['parity_max_abs']}")
    _ok('matching rows still pass', good['parity_ok'], '')
    _ok('recommend ignores the gated row only by timing',
        profile['measured_best_chunk_size'][8] in (4, 8), '')


def test_recommend_and_summarize_synthetic():
    print('\nTest: recommend / summarize on synthetic rows')
    rows = [
        {'T': 1, 'mode': 'sequential', 'chunk_size': None, 'time_ms': 10.0,
         'parity_ok': True},
        {'T': 1, 'mode': 'chunked', 'chunk_size': 4, 'time_ms': 2.0,
         'parity_ok': True},
        {'T': 1, 'mode': 'chunked', 'chunk_size': 2, 'time_ms': 2.0,
         'parity_ok': False},
    ]
    best = lp.recommend_chunk_sizes(rows)
    _ok('timing tie breaks to the smaller chunk', best == {1: 2}, f'{best}')
    profile = {'rows': rows, 'measured_best_chunk_size': best,
               'model_best_chunk_size': {1: 4}}
    summary = lp.summarize(profile)
    _ok('summary joins seq / best / model / parity',
        len(summary) == 1 and summary[0]['sequential_ms'] == 10.0
        and summary[0]['best_chunked_ms'] == 2.0
        and abs(summary[0]['speedup_vs_sequential'] - 5.0) < 1e-12
        and summary[0]['model_best_chunk_size'] == 4
        and summary[0]['parity_ok'] is False, f'{summary}')


# --- real integration -----------------------------------------------------

def test_real_profile_small_and_json_roundtrip():
    print('\nTest: real-torch profile at tiny T + JSON envelope round-trip')
    inputs = lp.make_scan_inputs(6, heads=2, headdim=4, d_state=2, seed=3)
    again = lp.make_scan_inputs(6, heads=2, headdim=4, d_state=2, seed=3)
    _ok('input factory is seed-reproducible',
        all(torch.equal(inputs[k], again[k]) for k in inputs), '')
    _ok('dt is strictly positive', bool((inputs['dt'] > 0).all()), '')
    profile = lp.profile_scan_modes(
        (24,), chunk_sizes=(8, 24), geometry={'heads': 2, 'headdim': 4,
                                              'd_state': 2, 'ngroups': 1},
        device='cpu', seed=3, warmup=1, repeats=2, include_quadratic=True,
        max_quadratic_len=24)
    rows = profile['rows']
    _ok('real rows are finite with positive timings',
        all(r['time_ms'] > 0.0 and r['finite'] for r in rows), '')
    _ok('real chunked/quadratic parity gates pass',
        all(r['parity_ok'] for r in rows),
        f"worst={max(r['parity_max_abs'] for r in rows):.2e}")
    _ok('real summary reports a positive speedup',
        all(s['speedup_vs_sequential'] and s['speedup_vs_sequential'] > 0
            for s in lp.summarize(profile)), '')
    with tempfile.TemporaryDirectory(prefix='_latency_profile_') as tmp:
        path = os.path.join(tmp, 'profile.json')
        rc = lp.main(['--lengths', '16', '--chunk-sizes', '8,16',
                      '--repeats', '1', '--warmup', '0',
                      '--heads', '2', '--headdim', '4', '--d-state', '2',
                      '--json', path])
        _ok('CLI returns 0', rc == 0, f'rc={rc}')
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        _ok('JSON carries the profile kind', raw['kind'] == lp.PROFILE_KIND,
            raw.get('kind'))
        _meta, parsed_rows, _v = results_schema.parse(raw)
        _ok('JSON parses as a results_schema envelope',
            len(parsed_rows) == len(profile['rows']) - 1 and all(
                'parity_ok' in r for r in parsed_rows), f'{len(parsed_rows)}')
        _ok('JSON metadata records measured vs model optimum',
            'measured_best_chunk_size' in raw['metadata']
            and 'model_best_chunk_size' in raw['metadata'], '')


def test_benchmark_chunked_scan_row_metadata():
    print('\nTest: Exp 2 chunked-scan row carries boundary metadata + parity')
    import run_benchmark as rb

    meta = {}
    fn = rb.bench_mamba2_chunked_scan(1, 130, 64, torch.device('cpu'),
                                      meta=meta)
    with torch.no_grad():
        y, state = fn()
    _ok('chunked row runs at tiny T', tuple(y.shape) == (1, 130, 4, 32)
        and tuple(state.shape) == (1, 4, 32, 32),
        f'y={tuple(y.shape)} s={tuple(state.shape)}')
    boundary = meta.get('scan_boundary') or {}
    _ok('row records the chunk decomposition',
        boundary.get('n_chunks') == 3 and boundary.get('padded_len') == 192
        and boundary.get('scan_levels') == 2,
        f"boundary={boundary}")
    _ok('row records a passing parity gate',
        meta.get('scan_parity_ok') is True
        and meta.get('scan_parity_max_abs', 1.0) <= 1e-4,
        f"parity={meta.get('scan_parity_max_abs')}")
    _ok('row keeps the core-boundary weight contract',
        meta.get('param_elements') == 0, str(meta.get('param_elements')))


_TEST_FUNCS = [
    test_ssd_core_flops_sequential_formula,
    test_ssd_core_flops_breakdown_exact,
    test_ssd_core_flops_chunk_grid_and_model_best,
    test_core_bytes_ssd_accounting,
    test_run_roofline_accounts_mamba_route,
    test_time_call_fake_clock,
    test_time_call_synchronize_hook,
    test_profile_fake_runner_and_rows,
    test_profile_parity_guard_detects_divergence,
    test_recommend_and_summarize_synthetic,
    test_real_profile_small_and_json_roundtrip,
    test_benchmark_chunked_scan_row_metadata,
]


def main() -> int:
    print('=' * 70)
    print('Mamba-2 SSD Latency Profiler / FLOPs-model Regression Tests')
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
