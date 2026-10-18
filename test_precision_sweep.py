"""Regression tests for the mixed-precision parity sweep harness.

Covers ``precision_sweep.py`` (fp64 oracle, seed sweep, dtype contract,
error budgets, autocast mode, CLI/JSON envelope) and the
``results_schema`` row contract registered for ``kind='precision_sweep'``.

The harness mechanics are tested against synthetic rows / injected fake
runners so the assertions are exact and fast; one real end-to-end sweep
at tiny T pins the integration (input factory, ops dispatch, contracts,
budget verdicts, JSON write/parse/validate round-trip).

Run via pytest::

    pytest -q test_precision_sweep.py

Or directly::

    python3 test_precision_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import precision_sweep as ps  # noqa: E402
import results_schema  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _synth_row(case='kda', path='rec', dtype='float16', mode='explicit',
               seq_len=64, seed=0, out_eps=1.0, state_eps=1.0, finite=True,
               out_ok=True, state_ok=True, error=None):
    row = {
        'case': case, 'path': path, 'dtype': dtype, 'mode': mode,
        'seed': seed, 'seq_len': seq_len, 'chunk_size': 16,
        'out_dtype': dtype, 'state_dtype': 'float32',
        'out_dtype_ok': out_ok, 'state_dtype_ok': state_ok,
        'finite_ok': finite,
        'out_abs_err': 0.0, 'out_rel_err': 0.0, 'out_eps_units': out_eps,
        'state_abs_err': 0.0, 'state_rel_err': 0.0, 'state_eps_units': state_eps,
        'oracle_out_scale': 1.0, 'oracle_state_scale': 1.0,
    }
    if error is not None:
        row['error'] = error
    return row


# --- dtype / budget arithmetic -------------------------------------------

def test_unit_roundoff_and_dtype_parsing():
    print('\nTest: unit roundoff table and dtype parsing')
    _ok('fp16 u = 2^-11', ps.unit_roundoff(torch.float16) == 2.0 ** -11, '')
    _ok('bf16 u = 2^-8', ps.unit_roundoff(torch.bfloat16) == 2.0 ** -8, '')
    _ok('fp32 u = 2^-24', ps.unit_roundoff('float32') == 2.0 ** -24, '')
    _ok('fp64 u = 2^-53', ps.unit_roundoff('fp64') == 2.0 ** -53, '')
    _ok('alias half -> fp16', ps.as_torch_dtype('half') == torch.float16, '')
    _ok('alias bf16 -> bfloat16',
        ps.as_torch_dtype('bf16') == torch.bfloat16, '')
    _ok('torch. prefix stripped',
        ps.as_torch_dtype('torch.float16') == torch.float16, '')
    for bad in ('float8', 3, None, object()):
        raised = False
        try:
            ps.as_torch_dtype(bad)
        except (TypeError, ValueError):
            raised = True
        _ok(f'rejects dtype {bad!r}', raised, '')


def test_error_budget_regimes():
    print('\nTest: error budget regimes (storage quantization vs gamma_T)')
    f = 8.0
    _ok('storage dtype budget is T-independent',
        ps.error_budget_eps_units(torch.float16, 64, factor=f) == f
        and ps.error_budget_eps_units(torch.bfloat16, 4096, factor=f) == f, '')
    _ok('compute dtype budget is accum_factor * T (gamma_T)',
        ps.error_budget_eps_units(torch.float32, 64, factor=f,
                                  accum_factor=2.0) == 128.0
        and ps.error_budget_eps_units(torch.float64, 256, factor=f,
                                      accum_factor=2.0) == 512.0, '')
    _ok('accum_factor unchanged by dtype-specific factor',
        ps.error_budget_eps_units(torch.float32, 100, factor=0.5,
                                  accum_factor=3.0) == 300.0, '')
    for bad in (0, -1, 1.5, True, '4'):
        raised = False
        try:
            ps.error_budget_eps_units(torch.float16, bad)
        except (TypeError, ValueError):
            raised = True
        _ok(f'rejects seq_len {bad!r}', raised, '')
    for bad in (0.0, -2.0, float('nan'), float('inf')):
        for kw in ({'factor': bad}, {'accum_factor': bad}):
            raised = False
            try:
                ps.error_budget_eps_units(torch.float16, 64, **kw)
            except ValueError:
                raised = True
            _ok(f'rejects {list(kw)[0]}={bad!r}', raised, '')


# --- input factory ---------------------------------------------------------

def test_build_inputs_seeded_and_invariants():
    print('\nTest: seeded fp64 input factory per case')
    for case in ps.CASES:
        a = ps.build_inputs(case, seed=3, seq_len=12)
        b = ps.build_inputs(case, seed=3, seq_len=12)
        c = ps.build_inputs(case, seed=4, seq_len=12)
        _ok(f'{case}: same seed reproduces bit-identical inputs',
            all(torch.equal(a[k], b[k]) for k in a), '')
        _ok(f'{case}: different seed changes the draw',
            any(not torch.equal(a[k], c[k]) for k in a), '')
        _ok(f'{case}: all inputs are fp64',
            all(v.dtype == torch.float64 for v in a.values()), '')
    kda = ps.build_inputs('kda', seed=0, seq_len=12)
    _ok('kda q/k are unit-norm before quantization',
        torch.allclose(kda['q'].norm(dim=-1), torch.ones(1, 12, 2, dtype=torch.float64))
        and torch.allclose(kda['k'].norm(dim=-1), torch.ones(1, 12, 2, dtype=torch.float64)),
        '')
    _ok('kda g <= 0 and beta in [0, 0.5)',
        bool((kda['g'] <= 0).all() and (kda['beta'] >= 0).all()
             and (kda['beta'] < 0.5).all()), '')
    mam = ps.build_inputs('mamba2', seed=0, seq_len=12)
    _ok('mamba2 dt > 0 and A < 0',
        bool((mam['dt'] > 0).all() and (mam['A'] < 0).all()), '')
    rw = ps.build_inputs('rwkv7', seed=0, seq_len=12)
    _ok('rwkv7 a in [0,1) and kk unit-norm',
        bool((rw['a'] >= 0).all() and (rw['a'] < 1).all()
             and torch.allclose(rw['kk'].norm(dim=-1),
                                torch.ones(1, 12, 2, dtype=torch.float64))),
        '')
    for bad_case in ('kda2', '', None):
        raised = False
        try:
            ps.build_inputs(bad_case, seed=0, seq_len=4)
        except (TypeError, ValueError):
            raised = True
        _ok(f'rejects case {bad_case!r}', raised, '')
    for bad_seed in (True, 0.5, '0'):
        raised = False
        try:
            ps.build_inputs('kda', seed=bad_seed, seq_len=4)
        except (TypeError, ValueError):
            raised = True
        _ok(f'rejects seed {bad_seed!r}', raised, '')
    raised = False
    try:
        ps.build_inputs('kda', seed=0, seq_len=0)
    except ValueError:
        raised = True
    _ok('rejects seq_len=0', raised, '')


# --- metrics ---------------------------------------------------------------

def test_compare_to_oracle_hand_computed():
    print('\nTest: compare_to_oracle in absolute and eps units')
    u = 2.0 ** -11
    oracle = torch.tensor([1.0, 2.0], dtype=torch.float64)
    test = oracle + u
    cmp = ps.compare_to_oracle(test, oracle)
    _ok('abs error = u', abs(cmp['abs_err'] - u) < 1e-18, f"{cmp['abs_err']}")
    _ok('rel error = u / max|oracle| = u/2',
        abs(cmp['rel_err'] - u / 2) < 1e-18, f"{cmp['rel_err']}")
    _ok('eps units = 0.5', abs(cmp['rel_err'] / u - 0.5) < 1e-12, '')
    _ok('oracle scale recorded', cmp['oracle_scale'] == 2.0, '')
    zero = torch.zeros(3, dtype=torch.float64)
    _ok('zero oracle + exact test -> rel 0',
        ps.compare_to_oracle(zero, zero)['rel_err'] == 0.0, '')
    _ok('zero oracle + tiny diff -> rel inf',
        ps.compare_to_oracle(torch.full((3,), 1e-12, dtype=torch.float64),
                             zero)['rel_err'] == float('inf'), '')
    raised = False
    try:
        ps.compare_to_oracle(torch.zeros(2), torch.zeros(3))
    except ValueError:
        raised = True
    _ok('shape mismatch raises', raised, '')


def test_evaluate_seed_real_row_contract():
    print('\nTest: real evaluate_seed row (fp16) passes dtype contract')
    row = ps.evaluate_seed('kda', 'chunk', torch.float16, seed=0,
                           seq_len=8, chunk_size=4)
    _ok('no injected error', 'error' not in row, row.get('error'))
    _ok('output dtype matches input dtype',
        row['out_dtype'] == 'float16' and row['out_dtype_ok'], row['out_dtype'])
    _ok('state stays compute-precision fp32',
        row['state_dtype'] == 'float32' and row['state_dtype_ok'],
        row['state_dtype'])
    _ok('finite', row['finite_ok'], '')
    _ok('fp16 error within ~2u',
        row['out_eps_units'] < 4.0, f"{row['out_eps_units']:.3f}u")
    row32 = ps.evaluate_seed('mamba2', 'chunk', torch.float32, seed=0,
                             seq_len=8, chunk_size=4)
    _ok('fp32 row keeps fp32 out/state',
        row32['out_dtype'] == 'float32' and row32['state_dtype'] == 'float32'
        and row32['out_dtype_ok'] and row32['state_dtype_ok'], '')
    raised = False
    try:
        ps.evaluate_seed('kda', 'rec', torch.float16, seed=0, seq_len=8,
                         mode='bogus')
    except ValueError:
        raised = True
    _ok('unknown mode rejected', raised, '')
    raised = False
    try:
        ps.evaluate_seed('kda', 'rec', torch.float32, seed=0, seq_len=8,
                         mode='autocast')
    except ValueError:
        raised = True
    _ok('autocast with fp32 target rejected', raised, '')


def test_injected_contract_violations_and_failures():
    print('\nTest: injected dtype/finite violations are caught per row')
    real_call = ps.call_case

    def wrong_out_dtype(case, path, inputs, *, chunk_size=16):
        out, state = real_call(case, path, inputs, chunk_size=chunk_size)
        return out.float() if out.dtype != torch.float32 else out, state

    with mock.patch.object(ps, 'call_case', wrong_out_dtype):
        row = ps.evaluate_seed('kda', 'rec', torch.float16, seed=0, seq_len=8)
    _ok('fp32 output for fp16 input flagged', row['out_dtype_ok'] is False,
        row['out_dtype'])

    def wrong_state_dtype(case, path, inputs, *, chunk_size=16):
        out, state = real_call(case, path, inputs, chunk_size=chunk_size)
        return out, state.to(inputs[list(inputs)[0]].dtype)

    with mock.patch.object(ps, 'call_case', wrong_state_dtype):
        row = ps.evaluate_seed('kda', 'rec', torch.float16, seed=0, seq_len=8)
    _ok('fp16 state for fp16 input flagged (contract is fp32)',
        row['state_dtype_ok'] is False, row['state_dtype'])

    def nan_out(case, path, inputs, *, chunk_size=16):
        out, state = real_call(case, path, inputs, chunk_size=chunk_size)
        out = out.clone()
        out.view(-1)[0] = float('nan')
        return out, state

    with mock.patch.object(ps, 'call_case', nan_out):
        row = ps.evaluate_seed('kda', 'rec', torch.float16, seed=0, seq_len=8)
    _ok('non-finite output flagged', row['finite_ok'] is False, '')

    def boom(*a, **kw):
        raise RuntimeError('injected')

    with mock.patch.object(ps, 'call_case', boom):
        row = ps.evaluate_seed('kda', 'rec', torch.float16, seed=0, seq_len=8)
    _ok('raising runner recorded as error row',
        'error' in row and 'injected' in row['error'], row.get('error'))
    with mock.patch.object(ps, 'call_case', boom):
        rows = ps.run_sweep(cases=['kda'], dtypes=[torch.float16], seeds=[0],
                            seq_len=8)
    summary = ps.summarize(rows)
    _ok('error rows fail the group verdict',
        all(r['verdict'] == 'fail' and r['n_ok'] == 0 for r in summary),
        f"{len(summary)} groups")
    _ok('failures() explains the error',
        all('errored' in line for line in ps.failures(summary)), '')


# --- seed-sweep aggregation -------------------------------------------------

def test_summarize_seed_spread_and_verdicts():
    print('\nTest: seed-sweep max/median/spread, worst_seed, budget verdict')
    rows = [_synth_row(seed=s, out_eps=e, state_eps=e / 2)
            for s, e in enumerate((2.0, 4.0, 8.0))]
    (agg,) = ps.summarize(rows, budget_factor=16.0)
    _ok('n_seeds/n_ok counted', (agg['n_seeds'], agg['n_ok']) == (3, 3), f'{agg}')
    _ok('max over seeds = 8', agg['out_eps_max'] == 8.0, '')
    _ok('median over seeds = 4', agg['out_eps_median'] == 4.0, '')
    _ok('spread = max/median = 2', agg['out_spread'] == 2.0, '')
    _ok('worst_seed points at the max', agg['worst_seed'] == 2, '')
    _ok('within 16u budget -> pass',
        agg['out_budget_ok'] and agg['verdict'] == 'pass', f"{agg['verdict']}")

    over = [r for r in rows]
    over[2] = _synth_row(seed=2, out_eps=17.0, state_eps=1.0)
    (agg_over,) = ps.summarize(over, budget_factor=16.0)
    _ok('max above budget -> fail with reason',
        agg_over['verdict'] == 'fail' and not agg_over['out_budget_ok'],
        '')
    _ok('state budget checked separately',
        agg_over['state_budget_ok'] is True, '')
    _ok('failures() names the over-budget group',
        any('out error' in line for line in ps.failures([agg_over])), '')

    # A dtype contract violation in ANY seed fails the group.
    bad = [_synth_row(seed=0), _synth_row(seed=1, out_ok=False)]
    (agg_bad,) = ps.summarize(bad)
    _ok('one bad seed fails the group contract flag',
        agg_bad['dtype_contract_ok'] is False
        and agg_bad['verdict'] == 'fail', '')

    # A NaN seed contributes no eps sample but poisons finiteness.
    nan = [_synth_row(seed=0), _synth_row(seed=1, finite=False)]
    (agg_nan,) = ps.summarize(nan)
    _ok('non-finite seed fails group and is excluded from max',
        agg_nan['finite_ok'] is False and agg_nan['out_eps_max'] == 1.0
        and agg_nan['verdict'] == 'fail', f"{agg_nan['out_eps_max']}")

    _ok('no-success group reports None stats and fails',
        ps.summarize([_synth_row(error='boom')])[0]['out_eps_max'] is None,
        '')
    for bad_rows in ('not a list', [42]):
        raised = False
        try:
            ps.summarize(bad_rows)
        except TypeError:
            raised = True
        _ok(f'rejects summary input {bad_rows!r}', raised, '')


def test_format_table_handles_none_and_verdicts():
    print('\nTest: CLI table formatting')
    rows = ps.summarize([_synth_row(seed=0), _synth_row(seed=1, error='x')])
    text = ps.format_table(rows)
    _ok('table has a header and one row per group',
        text.count('\n') >= 2 and 'verdict' in text, text.splitlines()[0])
    _ok('partially errored group shows 1/2 and fails',
        '1/2' in text and 'fail' in text, '')
    only_error = ps.format_table(ps.summarize([_synth_row(error='x')]))
    _ok('fully errored group renders n/a stats',
        'n/a' in only_error and 'fail' in only_error, '')
    reasons = ps.failures(ps.summarize([_synth_row(error='x')]))
    _ok('failures() renders missing stats without crashing',
        all('n/a' in line or 'errored' in line for line in reasons),
        '; '.join(reasons))


# --- real integration -------------------------------------------------------

def test_real_tiny_sweep_all_pass_and_error_ordering():
    print('\nTest: real tiny sweep (explicit fp32/fp16/bf16) all within budget')
    rows = ps.run_sweep(cases=list(ps.CASES),
                        dtypes=[torch.float32, torch.float16, torch.bfloat16],
                        seeds=[0, 1], seq_len=16, chunk_size=8)
    summary = ps.summarize(rows)
    n_case_paths = sum(len(paths) for paths in ps.CASES.values())
    _ok(f'{n_case_paths} case-paths x 3 dtypes = {n_case_paths * 3} groups',
        len(summary) == n_case_paths * 3, f'{len(summary)}')
    _ok('every real group passes its budget',
        all(r['verdict'] == 'pass' for r in summary),
        '; '.join(ps.failures(summary))[:300])
    _ok('every real row honors the dtype contract',
        all(r['dtype_contract_ok'] and r['finite_ok'] for r in summary), '')
    worst = {}
    for r in summary:
        worst[r['dtype']] = max(worst.get(r['dtype'], 0.0),
                                max(r['out_eps_max'], r['state_eps_max']))
    # Absolute relative error must order fp32 < fp16 < bf16 (u32 << u16 < u8);
    # eps units are NOT comparable across dtypes by design.
    rel = {}
    for r in rows:
        if 'error' in r:
            continue
        rel[r['dtype']] = max(rel.get(r['dtype'], 0.0),
                              r['out_rel_err'], r['state_rel_err'])
    _ok('absolute error orders fp32 < fp16 < bf16',
        rel['float32'] < rel['float16'] < rel['bfloat16'],
        f"fp32={rel['float32']:.2e} fp16={rel['float16']:.2e} "
        f"bf16={rel['bfloat16']:.2e}")


def test_real_autocast_mode_bf16():
    print('\nTest: real autocast-mode evaluation (bf16) keeps fp32 contract')
    row = ps.evaluate_seed('kda', 'chunk', torch.bfloat16, seed=0,
                           seq_len=16, chunk_size=8, mode='autocast')
    _ok('autocast row has no error', 'error' not in row, row.get('error'))
    _ok('input dtype fp32 preserved on output',
        row['out_dtype'] == 'float32' and row['out_dtype_ok'], row['out_dtype'])
    _ok('state fp32', row['state_dtype'] == 'float32'
        and row['state_dtype_ok'], row['state_dtype'])
    _ok('finite and within budget',
        row['finite_ok'] and row['out_eps_units'] < 16.0,
        f"{row['out_eps_units']:.2f}u")
    summary = ps.summarize([row])
    _ok('autocast group passes', summary[0]['verdict'] == 'pass', '')


def test_autocast_state_contract_all_paths():
    print('\nTest: regression — every core path returns fp32 state under autocast')
    # The state einsum is autocast-eligible, so ``quadratic_mamba2_scan``
    # must cast its final state back to compute precision like the
    # sequential and chunked views. Pin the shared contract on every path.
    for case, paths in ps.CASES.items():
        for path in paths:
            row = ps.evaluate_seed(case, path, torch.bfloat16, seed=0,
                                   seq_len=16, chunk_size=8, mode='autocast')
            _ok(f'{case}/{path}: autocast state is compute-precision fp32',
                'error' not in row and row['state_dtype'] == 'float32'
                and row['state_dtype_ok'],
                row.get('error') or row.get('state_dtype'))
    quad = ps.evaluate_seed('mamba2', 'quad', torch.bfloat16, seed=0,
                            seq_len=16, chunk_size=8, mode='autocast')
    _ok('quadratic state matches sequential under autocast',
        quad['state_eps_units'] < 4.0, f"{quad['state_eps_units']:.2f}u")


def test_cli_json_roundtrip_and_contract():
    print('\nTest: CLI exit code, JSON envelope and registered contract')
    with tempfile.TemporaryDirectory(prefix='_precision_sweep_') as tmp:
        path = os.path.join(tmp, 'sweep.json')
        rc = ps.main(['--cases', 'kda', '--dtypes', 'float32,float16',
                      '--seeds', '0', '--length', '16', '--json', path])
        _ok('CLI returns 0 on all-pass', rc == 0, f'rc={rc}')
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
        _ok('JSON kind tagged', raw['kind'] == ps.SWEEP_KIND, raw.get('kind'))
        _meta, rows, _version = results_schema.parse(raw)
        n_case_paths = len(ps.CASES['kda'])
        n_expected = n_case_paths * 2
        _ok(f'{n_expected} groups written ({n_case_paths} paths x 2 dtypes)',
            len(rows) == n_expected, f'{len(rows)}')
        _ok('metadata records the sweep configuration',
            raw['metadata'].get('seeds') == [0]
            and raw['metadata'].get('dtypes', {}).get('explicit')
            == ['float32', 'float16']
            and raw['metadata'].get('n_groups') == n_expected, '')
        violations = results_schema.validate_payload(raw)
        _ok('payload passes the registered row contract', violations == [],
            '; '.join(violations[:3]))

    # Force an over-budget failure: deterministic CLI fail path.
    rc = ps.main(['--cases', 'kda', '--dtypes', 'float16', '--seeds', '0',
                  '--length', '16', '--budget-factor', '0.001'])
    _ok('CLI returns 1 when a group is over budget', rc == 1, f'rc={rc}')


def test_precision_contract_rejects_inconsistent_rows():
    print('\nTest: results_schema contract catches inconsistent aggregates')
    good = ps.summarize([_synth_row(seed=s) for s in range(2)])[0]
    _ok("contract registered for 'precision_sweep'",
        'precision_sweep' in results_schema.ROW_CONTRACTS, '')
    _ok('valid aggregate row passes',
        results_schema.validate_rows('precision_sweep', [good]) == [], '')

    def _mutated(**kw):
        row = dict(good)
        row.update(kw)
        return row

    cases = [
        ('verdict pass with a False flag',
         _mutated(verdict='pass', out_budget_ok=False)),
        ('verdict pass with n_ok < n_seeds',
         _mutated(verdict='pass', n_ok=1)),
        ('verdict fail while every check passed',
         _mutated(verdict='fail')),
        ('unknown dtype', _mutated(dtype='float8')),
        ('unknown mode', _mutated(mode='fp8_autocast')),
        ('non-positive budget', _mutated(budget_eps=0.0)),
        ('max below median', _mutated(out_eps_max=0.1, out_eps_median=1.0)),
        ('n_ok above n_seeds', _mutated(n_ok=5)),
        ('worst_seed set with n_ok=0',
         _mutated(n_ok=0, worst_seed=3, out_eps_max=None,
                  out_eps_median=None, state_eps_max=None,
                  state_eps_median=None, finite_ok=False, verdict='fail')),
    ]
    for name, row in cases:
        _ok(f'rejects: {name}',
            results_schema.validate_rows('precision_sweep', [row]) != [], '')
    dup = [dict(good), dict(good)]
    _ok('rejects duplicate group keys',
        results_schema.validate_rows('precision_sweep', dup) != [], '')
    _ok('metadata n_groups mismatch rejected',
        results_schema.validate_rows('precision_sweep', [good],
                                     {'n_groups': 3}) != [], '')


_TEST_FUNCS = [
    test_unit_roundoff_and_dtype_parsing,
    test_error_budget_regimes,
    test_build_inputs_seeded_and_invariants,
    test_compare_to_oracle_hand_computed,
    test_evaluate_seed_real_row_contract,
    test_injected_contract_violations_and_failures,
    test_summarize_seed_spread_and_verdicts,
    test_format_table_handles_none_and_verdicts,
    test_real_tiny_sweep_all_pass_and_error_ordering,
    test_real_autocast_mode_bf16,
    test_autocast_state_contract_all_paths,
    test_cli_json_roundtrip_and_contract,
    test_precision_contract_rejects_inconsistent_rows,
]


def main() -> int:
    print('=' * 70)
    print('Mixed-Precision Parity Sweep Regression Tests')
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
