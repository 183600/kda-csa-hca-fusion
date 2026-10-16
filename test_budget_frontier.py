"""Regression tests for ``kv_budget.py`` and Exp 9 (``run_budget_frontier``).

Covers three layers:

1. **Budget formulas** — hand-computed read counts at tiny geometries,
   component additivity, and the two structural claims of the round:
   KDA's read budget is constant in T (delta-rule O(1) state) while the
   CSA lightning-indexer scan grows with T even with ``topk`` fixed.
2. **Delta-rule identities** — the KDA recurrence with unit-norm keys,
   ``beta=1`` and ``alpha=1`` (g=0) is an exact associative memory: the
   most recent write is read back exactly, and orthonormal keys make every
   stored pair exactly recallable with an O(1)-sized state.
3. **Decode sweep smoke** — a tiny ``run_decode_sweep`` call produces
   finite measured rows whose analytic budget ordering matches the top-k
   ordering.

The tests are self-contained and never touch the real ``results/``
directory.

Run via pytest::

    pytest -q test_budget_frontier.py

Or directly::

    python3 test_budget_frontier.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import kv_budget  # noqa: E402
from ops_kda import naive_recurrent_kda  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# --- budget formulas ------------------------------------------------------

def test_sliding_window_entries_hand_values():
    print('\nTest: sliding-window entry counts')
    _ok('T=1,w=2 -> 1 (own token)',
        kv_budget.sliding_window_entries(1, 2) == 1, '')
    _ok('T=4,w=2 -> 1+2+2+2=7',
        kv_budget.sliding_window_entries(4, 2) == 7, '')
    _ok('T=16,w=4 -> 1+2+3+13*4=58',
        kv_budget.sliding_window_entries(16, 4) == 58, '')
    _ok('T=0 -> 0', kv_budget.sliding_window_entries(0, 4) == 0, '')


def test_op_budget_hand_values_mqar_geometry():
    print('\nTest: per-op budget at the MQAR geometry (T=16)')
    # d_k=d_v=16, n_kv_heads=2: avg (T+1)/2 * 2 * 32 = 544.
    sm = kv_budget.softmax_read_elements(16, 2, 16, 16)
    _ok('softmax = 544', abs(sm - 544.0) < 1e-9, f'{sm}')
    _ok('kda state = 2*16*16 = 512',
        kv_budget.kda_state_elements(2, 16, 16) == 512, '')
    _ok('kda read = 2x state = 1024',
        abs(kv_budget.kda_read_elements(16, 2, 16, 16) - 1024.0) < 1e-9, '')
    # CSA: m=4, topk=2, c=32, c_i=16, n_idx_heads=2, sw=4, T=16.
    # selected pairs = 20 -> 20*32/16 = 40; window = 58*32/16 = 116;
    # indexer pairs = 24 -> 24*16*2/16 = 48.
    csa = kv_budget.csa_read_elements(
        16, m=4, topk=2, c=32, c_i=16, n_idx_heads=2, sliding_window=4)
    _ok('CSA selected = 40', abs(csa['selected_read_elements'] - 40.0) < 1e-9,
        f"{csa}")
    _ok('CSA window = 116',
        abs(csa['window_read_elements'] - 116.0) < 1e-9, f"{csa}")
    _ok('CSA indexer = 48',
        abs(csa['indexer_read_elements'] - 48.0) < 1e-9, f"{csa}")
    _ok('CSA total = 204',
        abs(csa['total_read_elements'] - 204.0) < 1e-9, f"{csa}")
    # HCA: m2=8, c=32, sw=4 -> dense 8*32/16=16, window 116, total 132.
    hca = kv_budget.hca_read_elements(16, m2=8, c=32, sliding_window=4)
    _ok('HCA total = 132',
        abs(hca['total_read_elements'] - 132.0) < 1e-9, f"{hca}")


def test_budget_scaling_claims():
    print('\nTest: budget scaling in T (the round\'s structural claims)')
    kda_small = kv_budget.kda_read_elements(64, 2, 16, 16)
    kda_large = kv_budget.kda_read_elements(65536, 2, 16, 16)
    _ok('KDA read budget is O(1) in T', kda_small == kda_large == 1024.0,
        f'{kda_small} vs {kda_large}')
    sm_small = kv_budget.softmax_read_elements(64, 2, 16, 16)
    sm_large = kv_budget.softmax_read_elements(65536, 2, 16, 16)
    _ok('softmax read budget grows ~T', sm_large > 500 * sm_small,
        f'{sm_small} -> {sm_large}')
    # With topk fixed at 2, the selected-gather term saturates while the
    # indexer scan keeps growing linearly: the indexer is the binding term.
    c1 = kv_budget.csa_read_elements(
        256, m=4, topk=2, c=32, c_i=16, n_idx_heads=2, sliding_window=4)
    c2 = kv_budget.csa_read_elements(
        4096, m=4, topk=2, c=32, c_i=16, n_idx_heads=2, sliding_window=4)
    _ok('CSA indexer dominates at long T',
        c2['indexer_read_elements'] > 10 * c2['selected_read_elements'],
        f"sel={c2['selected_read_elements']:.1f} "
        f"idx={c2['indexer_read_elements']:.1f}")
    _ok('CSA selected term saturates at topk*c=64',
        c2['selected_read_elements'] < 64.0 + 1e-9,
        f"{c2['selected_read_elements']}")
    # Dense (topk >= blocks) must equal the no-cap count.
    dense = kv_budget.csa_read_elements(
        64, m=4, topk=99, c=32, c_i=16, n_idx_heads=2, sliding_window=4)
    full = kv_budget.csa_read_elements(
        64, m=4, topk=64, c=32, c_i=16, n_idx_heads=2, sliding_window=4)
    _ok('topk above block count degenerates to dense',
        dense['selected_read_elements'] == full['selected_read_elements'], '')


def test_op_and_layout_dispatch():
    print('\nTest: op/layout dispatch and validation')
    spec = {
        'softmax': {'n_kv_heads': 2, 'd_k': 16, 'd_v': 16},
        'kda': {'hv': 2, 'd_k': 16, 'd_v': 16},
        'csa': {'m': 4, 'topk': 2, 'c': 32, 'c_i': 16, 'n_idx_heads': 2,
                'sliding_window': 4},
        'hca': {'m2': 8, 'c': 32, 'sliding_window': 4},
    }
    totals = {op: kv_budget.op_read_elements(op, 16, spec)
              ['total_read_elements'] for op in kv_budget._OP_FUNCS}
    _ok('op totals match the component functions',
        abs(totals['softmax'] - 544.0) < 1e-9
        and abs(totals['kda'] - 1024.0) < 1e-9
        and abs(totals['csa'] - 204.0) < 1e-9
        and abs(totals['hca'] - 132.0) < 1e-9, f'{totals}')
    lay = kv_budget.layout_read_elements(16, (3, 1, 1), spec)
    _ok('3:1:1 layout = 3*1024 + 204 + 132 = 3408',
        abs(lay['total_read_elements'] - 3408.0) < 1e-9, f'{lay}')
    _ok('KDA state size reported once', lay['kda_read_elements'] == 3072.0, '')
    raised = False
    try:
        kv_budget.op_read_elements('mystery', 16, spec)
    except ValueError:
        raised = True
    _ok('unknown op raises ValueError', raised, '')
    raised = False
    try:
        kv_budget.op_read_elements('csa', 16, {'softmax': spec['softmax']})
    except ValueError:
        raised = True
    _ok('missing geometry block raises ValueError', raised, '')
    raised = False
    try:
        kv_budget.layout_read_elements(16, (0, 0, 0), spec)
    except ValueError:
        raised = True
    _ok('empty layout raises ValueError', raised, '')


def test_spearman_rank_correlation():
    print('\nTest: Spearman rank correlation helper')
    _ok('identical order -> +1',
        abs(kv_budget.spearman([1, 2, 3], [4, 5, 6]) - 1.0) < 1e-12, '')
    _ok('reversed order -> -1',
        abs(kv_budget.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12, '')
    s = kv_budget.spearman([1, 1, 2], [5, 5, 1])
    _ok('ties use average ranks', abs(s + 1.0) < 1e-12, f'{s}')
    _ok('constant input -> None',
        kv_budget.spearman([1, 1, 1], [1, 2, 3]) is None, '')
    _ok('single pair -> None', kv_budget.spearman([1], [2]) is None, '')
    raised = False
    try:
        kv_budget.spearman([1, 2], [1])
    except ValueError:
        raised = True
    _ok('length mismatch raises', raised, '')


# --- delta-rule identities -------------------------------------------------

def test_delta_rule_recent_write_identity():
    print('\nTest: delta-rule exact read-back of the most recent write')
    torch.manual_seed(0)
    B, T, HV, K, V = 1, 6, 1, 8, 4
    q = F.normalize(torch.randn(B, T, HV, K), dim=-1)
    k = F.normalize(torch.randn(B, T, HV, K), dim=-1)
    v = torch.randn(B, T, HV, V)
    g = torch.zeros(B, T, HV, K)          # alpha = exp(0) = 1
    beta = torch.ones(B, T, HV)           # delta-rule learning rate = 1
    _o, S = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    readback = S[0, 0].t() @ k[0, -1, 0]
    _ok('S^T k_last == v_last (fp32 tolerance)',
        torch.allclose(readback, v[0, -1, 0], atol=1e-4),
        f'max|err|={(readback - v[0, -1, 0]).abs().max().item():.3e}')
    _ok('state element count is hv*K*V',
        S.numel() == kv_budget.kda_state_elements(HV, K, V),
        f'{S.numel()}')


def test_delta_rule_orthonormal_keys_exact_recall():
    print('\nTest: orthonormal keys are all exactly recallable (O(1) state)')
    torch.manual_seed(1)
    K = 8
    Q, _R = torch.linalg.qr(torch.randn(K, K))
    keys = Q.t()                                # rows orthonormal
    B, T, HV, V = 1, K, 1, 3
    k = keys.view(B, T, HV, K)
    v = torch.randn(B, T, HV, V)
    g = torch.zeros(B, T, HV, K)
    beta = torch.ones(B, T, HV)
    _o, S = naive_recurrent_kda(k, k, v, g, beta, output_final_state=True)
    S = S[0, 0]
    errs = []
    for i in range(K):
        read = S.t() @ k[0, i, 0]
        errs.append((read - v[0, i, 0]).abs().max().item())
    _ok('every stored pair recalled after T=K writes',
        max(errs) < 1e-4, f'max|err|={max(errs):.3e}')
    _o2, S2 = naive_recurrent_kda(
        k[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3],
        output_final_state=True)
    _ok('state size independent of T',
        S2.numel() == S.numel()
        == kv_budget.kda_state_elements(HV, K, V), '')


# --- Exp 9 builder / sweep -------------------------------------------------

def test_build_budget_curves_rows():
    print('\nTest: run_budget_frontier.build_budget_curves row shape')
    from run_budget_frontier import LAYOUTS, build_budget_curves, mqar_spec
    spec = mqar_spec()
    rows = build_budget_curves([16], {'mqar': spec})
    _ok('one row per op and layout',
        len(rows) == 4 + len(LAYOUTS), f'{len(rows)} rows')
    csa = next(r for r in rows if r['row_kind'] == 'op'
               and r['name'] == 'csa')
    _ok('CSA row exposes both terms',
        abs(csa['total_read_elements'] - 204.0) < 1e-9
        and csa['indexer_read_elements'] == 48.0, f'{csa}')
    lay = next(r for r in rows if r['row_kind'] == 'layout'
               and r['name'] == '3:1:1')
    _ok('3:1:1 row total = 3408',
        abs(lay['total_read_elements'] - 3408.0) < 1e-9, f'{lay}')


def test_decode_sweep_smoke():
    print('\nTest: decode top-k sweep smoke (tiny budget)')
    from run_budget_frontier import run_decode_sweep
    rows = run_decode_sweep(
        torch.device('cpu'), [0, 2], [16], n_decode=2, repeats=1)
    _ok('two measured rows, no errors',
        len(rows) == 2 and all('error' not in r for r in rows), f'{rows}')
    _ok('prefill latency measured and finite',
        all(math.isfinite(r['prefill_ms']) and r['prefill_ms'] > 0
            for r in rows), '')
    _ok('decode latency measured and finite',
        all(math.isfinite(r['mean_decode_ms_per_token'])
            and r['mean_decode_ms_per_token'] > 0 for r in rows), '')
    by_k = {r['csa_topk']: r['total_read_elements_per_query'] for r in rows}
    _ok('budget increases from k=0 to k=2 (selected gather added)',
        by_k[2] > by_k[0], f'{by_k}')


_TEST_FUNCS = [
    test_sliding_window_entries_hand_values,
    test_op_budget_hand_values_mqar_geometry,
    test_budget_scaling_claims,
    test_op_and_layout_dispatch,
    test_spearman_rank_correlation,
    test_delta_rule_recent_write_identity,
    test_delta_rule_orthonormal_keys_exact_recall,
    test_build_budget_curves_rows,
    test_decode_sweep_smoke,
]


def main() -> int:
    print('=' * 70)
    print('kv_budget / Exp 9 (budget frontier) Regression Tests')
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
