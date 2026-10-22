"""Regression tests for ``kv_budget.py`` and Exp 9 (``run_budget_frontier``).

Covers four layers:

1. **Budget formulas** — hand-computed read counts at tiny geometries,
   component additivity, and the two structural claims of the round:
   KDA's read budget is constant in T (delta-rule O(1) state) while the
   CSA lightning-indexer scan grows with T even with ``topk`` fixed.
2. **Prefill/decode unification** — the marginal (decode-position) budget
   ``instant_read_elements`` averages to the prefill budget
   ``op_read_elements``, the O(1) ``windowed_read_elements`` reproduces
   both, and the same identities hold for layouts.
3. **Delta-rule identities** — the KDA recurrence with unit-norm keys,
   ``beta=1`` and ``alpha=1`` (g=0) is an exact associative memory: the
   most recent write is read back exactly, and orthonormal keys make every
   stored pair exactly recallable with an O(1)-sized state.
4. **Decode sweep smoke** — a tiny ``run_decode_sweep`` call produces
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
    _ok('mamba2 state = 4*16*8 = 512',
        kv_budget.mamba2_state_elements(4, 16, 8) == 512, '')
    _ok('mamba2 read = 2x state = 1024',
        abs(kv_budget.mamba2_read_elements(16, 4, 16, 8) - 1024.0) < 1e-9, '')
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
        'mamba2': {'nheads': 4, 'headdim': 16, 'd_state': 8},
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
    mamba = kv_budget.op_read_elements('mamba2', 16, spec)
    _ok('Mamba-2 op row is O(1) and reports its state',
        abs(mamba['total_read_elements'] - 1024.0) < 1e-9
        and mamba['state_elements'] == 512, f'{mamba}')
    _ok('per-op Mamba-2 marginal equals the prefill average',
        abs(kv_budget.instant_read_elements('mamba2', 16, spec)
            ['total_read_elements'] - mamba['total_read_elements']) < 1e-9, '')
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
    for name, fn, kwargs in (
            ('CSA', kv_budget.csa_read_elements,
             dict(T=16, m=4, topk=2, c=32, c_i=16)),
            ('HCA', kv_budget.hca_read_elements,
             dict(T=16, m2=8, c=32))):
        for bad in (True, 1.0):
            raised = False
            try:
                fn(sliding_window=bad, **kwargs)
            except ValueError:
                raised = True
            _ok(f'{name} rejects sliding_window={bad!r}', raised, '')


def _mqar_spec():
    """The Exp 9 MQAR geometry used by the hand-value tests."""
    return {
        'softmax': {'n_kv_heads': 2, 'd_k': 16, 'd_v': 16},
        'kda': {'hv': 2, 'd_k': 16, 'd_v': 16},
        'mamba2': {'nheads': 4, 'headdim': 16, 'd_state': 8},
        'csa': {'m': 4, 'topk': 2, 'c': 32, 'c_i': 16, 'n_idx_heads': 2,
                'sliding_window': 4},
        'hca': {'m2': 8, 'c': 32, 'sliding_window': 4},
    }


def test_layout_memory_route_substitution():
    print('\nTest: layout budget can select the Mamba-2 memory route')
    spec = _mqar_spec()
    spec['mamba2'] = {'nheads': 2, 'headdim': 16, 'd_state': 8}
    kda = kv_budget.layout_read_elements(16, (3, 1, 1), spec)
    _ok('default layout route is KDA and unchanged',
        kda['memory_op'] == 'kda'
        and kda['kda_read_elements'] == 3072.0
        and kda['mamba2_read_elements'] == 0.0
        and abs(kda['total_read_elements'] - 3408.0) < 1e-9, f'{kda}')
    mamba = kv_budget.layout_read_elements(
        16, (3, 1, 1), spec, memory_op='mamba2')
    _ok('Mamba-2 route prices 3 * (2*256) + CSA + HCA',
        mamba['memory_op'] == 'mamba2'
        and mamba['n_memory'] == 3 and mamba['n_kda'] == 0
        and mamba['n_mamba2'] == 3
        and mamba['memory_read_elements'] == 1536.0
        and mamba['kda_read_elements'] == 0.0
        and abs(mamba['total_read_elements'] - 1872.0) < 1e-9, f'{mamba}')
    step = kv_budget.instant_layout_read_elements(
        (3, 1, 1), 16, spec, memory_op='mamba2')
    _ok('Mamba-2 marginal step = 1536 + 320 + 192',
        abs(step['total_read_elements'] - 2048.0) < 1e-9, f'{step}')
    win = kv_budget.windowed_layout_read_elements(
        (3, 1, 1), 4, 20, spec, memory_op='mamba2')
    direct = [kv_budget.instant_layout_read_elements(
        (3, 1, 1), t, spec, memory_op='mamba2')['total_read_elements']
        for t in range(4, 20)]
    _ok('Mamba-2 windowed view is the mean of marginal steps',
        abs(win['total_read_elements'] - sum(direct) / len(direct)) < 1e-9,
        f"{win['total_read_elements']} vs {sum(direct) / len(direct)}")
    _ok('route stays explicit on the windowed row',
        win['memory_op'] == 'mamba2' and win['n_mamba2'] == 3, f'{win}')
    raised = False
    try:
        kv_budget.layout_read_elements(16, (3, 1, 1), spec,
                                       memory_op='rwkv7')
    except ValueError:
        raised = True
    _ok('unknown memory route raises ValueError', raised, '')


def test_instant_budget_hand_values():
    print('\nTest: marginal (decode-position) budget hand values')
    spec = _mqar_spec()
    # t=0: just the query's own token; no compressed block is visible yet.
    sm0 = kv_budget.instant_read_elements('softmax', 0, spec)
    _ok('softmax t=0 reads own token: 2*32=64',
        abs(sm0['total_read_elements'] - 64.0) < 1e-9, f'{sm0}')
    kda5 = kv_budget.instant_read_elements('kda', 5, spec)
    _ok('KDA marginal is position-independent (2*state)',
        kda5['total_read_elements'] == 1024.0, f'{kda5}')
    mamba5 = kv_budget.instant_read_elements('mamba2', 5, spec)
    _ok('Mamba-2 marginal is position-independent (2*state)',
        mamba5['total_read_elements'] == 1024.0
        and mamba5['state_elements'] == 512, f'{mamba5}')
    # CSA t=16: blocks 0..3 visible (strictly preceding), window saturates
    # at 4 tokens, indexer scores 4 blocks x 16 x 2 heads.
    csa16 = kv_budget.instant_read_elements('csa', 16, spec)
    _ok('CSA t=16 selected = min(2,4)*32 = 64',
        abs(csa16['selected_read_elements'] - 64.0) < 1e-9, f'{csa16}')
    _ok('CSA t=16 window = 4*32 = 128',
        abs(csa16['window_read_elements'] - 128.0) < 1e-9, f'{csa16}')
    _ok('CSA t=16 indexer = 4*16*2 = 128',
        abs(csa16['indexer_read_elements'] - 128.0) < 1e-9, f'{csa16}')
    hca15 = kv_budget.instant_read_elements('hca', 15, spec)
    _ok('HCA t=15 dense = floor(15/8)*32 = 32',
        abs(hca15['dense_read_elements'] - 32.0) < 1e-9, f'{hca15}')
    _ok('HCA t=7 has no visible block yet',
        kv_budget.instant_read_elements('hca', 7, spec)['dense_read_elements']
        == 0.0, '')
    raised = False
    try:
        kv_budget.instant_read_elements('csa', -1, spec)
    except ValueError:
        raised = True
    _ok('negative position raises ValueError', raised, '')
    raised = False
    try:
        kv_budget.instant_read_elements('mystery', 4, spec)
    except ValueError:
        raised = True
    _ok('unknown op raises ValueError', raised, '')


def test_prefill_decode_unification_identity():
    print('\nTest: prefill average == mean of decode marginals (unification)')
    spec = _mqar_spec()
    keys = ('core_read_elements', 'indexer_read_elements',
            'total_read_elements')
    for op in kv_budget._OP_FUNCS:
        for T in (1, 2, 7, 16, 33):
            avg = kv_budget.op_read_elements(op, T, spec)
            insts = [kv_budget.instant_read_elements(op, t, spec)
                     for t in range(T)]
            bad = [k for k in keys
                   if abs(sum(i[k] for i in insts) - T * avg[k]) > 1e-9]
            _ok(f'{op} T={T}: mean marginal == averaged prefill',
                not bad, f'{bad}')
    # The averaged layout is the layer-sum of the averaged ops, and the
    # marginal layout is the layer-sum of the marginal ops.
    for ratio in ((3, 1, 1), (4, 1, 0), (1, 1, 3)):
        for T in (1, 16, 33):
            lavg = kv_budget.layout_read_elements(T, ratio, spec)
            tot = sum(kv_budget.instant_layout_read_elements(ratio, t, spec)
                      ['total_read_elements'] for t in range(T))
            _ok(f'layout {ratio} T={T}: prefilled total == marginal sum',
                abs(tot - T * lavg['total_read_elements']) < 1e-9,
                f'{tot} vs {T * lavg["total_read_elements"]}')


def test_windowed_budget_matches_instant_sums():
    print('\nTest: O(1) windowed budget == direct marginal mean')
    spec = _mqar_spec()
    keys = ('core_read_elements', 'indexer_read_elements',
            'total_read_elements')
    for op in kv_budget._OP_FUNCS:
        for (t0, t1) in ((0, 1), (0, 16), (5, 17), (16, 26)):
            win = kv_budget.windowed_read_elements(op, t0, t1, spec)
            direct = [kv_budget.instant_read_elements(op, t, spec)
                      for t in range(t0, t1)]
            bad = [k for k in keys
                   if abs(win[k] - sum(d[k] for d in direct) / (t1 - t0))
                   > 1e-9]
            _ok(f'{op} [{t0},{t1}): windowed == marginal mean', not bad,
                f'{bad}')
            if t0 == 0:
                avg = kv_budget.op_read_elements(op, t1, spec)
                _ok(f'{op} [0,{t1}): windowed == averaged prefill',
                    all(abs(win[k] - avg[k]) < 1e-9 for k in keys), '')
    ratio = (3, 1, 1)
    wl = kv_budget.windowed_layout_read_elements(ratio, 4, 20, spec)
    lay = kv_budget.layout_read_elements(20, ratio, spec)
    direct = [kv_budget.instant_layout_read_elements(ratio, t, spec)
              for t in range(4, 20)]
    _ok('layout [4,20): windowed component == marginal mean',
        all(abs(wl[k] - sum(d[k] for d in direct) / 16) < 1e-9
            for k in ('kda_read_elements', 'csa_read_elements',
                      'hca_read_elements', 'total_read_elements')), '')
    _ok('layout [0,20) windowed == averaged layout',
        all(abs(kv_budget.windowed_layout_read_elements(ratio, 0, 20, spec)[k]
                - lay[k]) < 1e-9
            for k in ('kda_read_elements', 'csa_read_elements',
                      'hca_read_elements', 'total_read_elements')), '')
    for args in ((5, 5), (9, 3)):
        raised = False
        try:
            kv_budget.windowed_read_elements('csa', *args, spec)
        except ValueError:
            raised = True
        _ok(f'windowed {args} raises ValueError', raised, '')


def test_decode_frontier_structural_claims():
    print('\nTest: marginal frontier structural claims (decode side)')
    spec = _mqar_spec()
    # Decode marginal is monotone non-decreasing in position for the
    # O(T)-reading operators, and flat for KDA.
    for op in ('softmax', 'csa', 'hca'):
        prev = -1.0
        mono = True
        for t in range(0, 64):
            cur = kv_budget.instant_read_elements(op, t, spec)[
                'total_read_elements']
            if cur < prev - 1e-12:
                mono = False
                break
            prev = cur
        _ok(f'{op} marginal budget is non-decreasing in t', mono, '')
    kda_vals = {kv_budget.instant_read_elements('kda', t, spec)
                ['total_read_elements'] for t in range(0, 64)}
    _ok('KDA marginal budget is constant in t (O(1) state)',
        kda_vals == {1024.0}, f'{kda_vals}')
    # The CSA indexer term is what makes decode cost grow even when the
    # top-k gather is capped: at t=64 the scan dwarfs the gather.
    csa64 = kv_budget.instant_read_elements('csa', 64, spec)
    _ok('CSA indexer dominates selected at t=64',
        csa64['indexer_read_elements'] > 2 * csa64['selected_read_elements'],
        f"{csa64}")


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

def test_parse_ratio_rejects_invalid_layouts():
    """A malformed layout label must be a skip, never a crash.

    The layout accounting rejects negative components and the empty
    layout; the parser has to reject them too, otherwise the join dies
    with an uncaught ValueError instead of dropping the row.
    """
    from run_budget_frontier import _parse_ratio
    _ok('valid ratio parsed', _parse_ratio('3:1:1') == (3, 1, 1),
        str(_parse_ratio('3:1:1')))
    _ok('negative component rejected', _parse_ratio('3:-1:1') is None,
        str(_parse_ratio('3:-1:1')))
    _ok('empty layout rejected', _parse_ratio('0:0:0') is None,
        str(_parse_ratio('0:0:0')))
    _ok('wrong arity rejected', _parse_ratio('3:1') is None,
        str(_parse_ratio('3:1')))
    _ok('non-numeric rejected', _parse_ratio('a:b:c') is None,
        str(_parse_ratio('a:b:c')))


def test_build_budget_curves_rows():
    print('\nTest: run_budget_frontier.build_budget_curves row shape')
    from run_budget_frontier import LAYOUTS, build_budget_curves, mqar_spec
    spec = mqar_spec()
    rows = build_budget_curves([16], {'mqar': spec})
    _ok('one row per op and layout',
        len(rows) == 5 + len(LAYOUTS), f'{len(rows)} rows')
    csa = next(r for r in rows if r['row_kind'] == 'op'
               and r['name'] == 'csa')
    _ok('CSA row exposes both terms',
        abs(csa['total_read_elements'] - 204.0) < 1e-9
        and csa['indexer_read_elements'] == 48.0, f'{csa}')
    mamba = next(r for r in rows if r['row_kind'] == 'op'
                 and r['name'] == 'mamba2')
    _ok('Mamba-2 row joins at the Exp 4 quality geometry',
        abs(mamba['total_read_elements'] - 2048.0) < 1e-9
        and mamba['state_elements'] == 1024, f'{mamba}')
    lay = next(r for r in rows if r['row_kind'] == 'layout'
               and r['name'] == '3:1:1')
    _ok('3:1:1 row total = 3408',
        abs(lay['total_read_elements'] - 3408.0) < 1e-9, f'{lay}')


def test_decode_budget_curve_and_sweep_join():
    print('\nTest: decode_budget_curve rows and decode-sweep budget join')
    from run_budget_frontier import (LAYOUTS, build_decode_budget_curves,
                                     mqar_spec, run_decode_sweep)
    spec = mqar_spec()
    rows = build_decode_budget_curves([16], {'mqar': spec})
    _ok('one marginal row per op and layout',
        len(rows) == 5 + len(LAYOUTS), f'{len(rows)} rows')
    csa = next(r for r in rows if r['row_kind'] == 'op'
               and r['name'] == 'csa')
    _ok('CSA marginal t=16 total = 320, indexer = 128',
        abs(csa['total_read_elements'] - 320.0) < 1e-9
        and abs(csa['indexer_read_elements'] - 128.0) < 1e-9, f'{csa}')
    mamba = next(r for r in rows if r['row_kind'] == 'op'
                 and r['name'] == 'mamba2')
    _ok('Mamba-2 marginal is the O(1) state read',
        abs(mamba['total_read_elements'] - 2048.0) < 1e-9, f'{mamba}')
    lay = next(r for r in rows if r['row_kind'] == 'layout'
               and r['name'] == '3:1:1')
    _ok('3:1:1 marginal t=16 total = 3584',
        abs(lay['total_read_elements'] - 3584.0) < 1e-9, f'{lay}')
    _ok('marginal row carries position t (not T)',
        lay['t'] == 16 and 'T' not in lay, '')
    sweep = run_decode_sweep(
        torch.device('cpu'), [2], [16], n_decode=2, repeats=1)
    row = sweep[0]
    # The decode geometry is ``run_decoding.HybridDecoding``'s, not the
    # MQAR probe geometry; mirror ``_decode_row``'s spec builder.
    from run_budget_frontier import _spec_from_hybrid_cfg
    from run_decoding import HybridDecoding
    decode_spec = _spec_from_hybrid_cfg(
        HybridDecoding(d_model=64, total_layers=5, csa_topk=2).cfg,
        softmax_geo={'n_kv_heads': 2, 'd_k': 16, 'd_v': 16})
    step = kv_budget.instant_layout_read_elements((3, 1, 1), 16,
                                                  decode_spec)
    win = kv_budget.windowed_layout_read_elements((3, 1, 1), 16, 18,
                                                  decode_spec)
    _ok('decode geometry marginal step = 3072 + 64 + 64',
        abs(row['decode_step_read_elements_per_query'] - 3200.0) < 1e-9,
        f"{row['decode_step_read_elements_per_query']}")
    _ok('decode_sweep joins the marginal (first-decode-step) budget',
        abs(row['decode_step_read_elements_per_query']
            - step['total_read_elements']) < 1e-9
        and abs(row['decode_step_csa_read_elements']
                - step['csa_read_elements']) < 1e-9, f'{row}')
    _ok('decode_sweep joins the measured decode-window budget',
        abs(row['decode_window_read_elements_per_query']
            - win['total_read_elements']) < 1e-9
        and row['decode_window_read_elements_per_query']
        >= row['decode_step_read_elements_per_query'], f'{row}')
    _ok('decode_sweep keeps the backward-compatible prefill average',
        abs(row['total_read_elements_per_query']
            - kv_budget.layout_read_elements(
                16, (3, 1, 1), decode_spec)['total_read_elements']) < 1e-9,
        f"{row['total_read_elements_per_query']}")


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
    _ok('decode rows keep the prefill/decode bootstrap CIs',
        all(r.get('decode_ci95_low_ms') is not None
            and r.get('decode_ci95_high_ms') is not None
            and r.get('prefill_ci95_low_ms') is not None
            and r.get('prefill_ci95_high_ms') is not None
            and r.get('decode_n') == r['repeats'] * r['n_decode']
            and r.get('prefill_n') == r['repeats'] for r in rows),
        f"{[(r.get('decode_n'), r.get('prefill_n')) for r in rows]}")
    _ok('CI bounds bracket the reported point medians',
        all(r['decode_ci95_low_ms'] <= r['median_decode_ms_per_token']
            <= r['decode_ci95_high_ms']
            and r['prefill_ci95_low_ms'] <= r['prefill_ms']
            <= r['prefill_ci95_high_ms'] for r in rows), '')
    by_k = {r['csa_topk']: r['total_read_elements_per_query'] for r in rows}
    _ok('budget increases from k=0 to k=2 (selected gather added)',
        by_k[2] > by_k[0], f'{by_k}')


# --- Exp 9 cross-source provenance audit --------------------------------

def _audit_record(source, commit='a' * 40, seeds=(42, 43, 44, 45, 46),
                  backends=None, readable=True, dirty=False):
    return {
        'source': source,
        'path': f'/tmp/{source}.json',
        'exists': readable,
        'readable': readable,
        'schema_version': '1.0' if readable else None,
        'n_rows': 1,
        'sha256': '0' * 64 if readable else None,
        'git_commit': commit,
        'git_dirty': dirty,
        'captured_at_utc': '2026-01-01T00:00:00+00:00',
        'seeds': list(seeds),
        'n_seeds': len(seeds),
        'backends': backends if backends is not None else {
            'kda_backend_resolved': 'reference',
            'mamba_backend_resolved': 'reference',
            'rwkv_backend_resolved': 'reference',
        },
        'provenance_source': 'metadata.provenance' if readable else None,
        'error': None if readable else 'missing',
        'rows': [],
    }


def test_exp9_source_audit_same_revision_is_comparable():
    print('\nTest: Exp 9 source audit accepts a common revision/seed set')
    from run_budget_frontier import audit_join_sources
    records = [_audit_record(s) for s in
               ('exp4_mqar', 'exp5_ablation', 'exp8_mechanism_ablation')]
    rows, summary = audit_join_sources(records)
    _ok('same commit + shared seeds -> comparable',
        summary['status'] == 'comparable', summary['status'])
    _ok('ranking is allowed', summary['comparable_for_ranking'] is True, '')
    _ok('shared seed set is recorded',
        summary['shared_seeds'] == [42, 43, 44, 45, 46], summary['shared_seeds'])
    _ok('one row per source plus a cross-source row',
        len(rows) == 4
        and rows[-1]['source'] == 'cross_source'
        and all(r['section'] == 'source_audit' for r in rows), '')
    _ok('source rows omit the raw result rows',
        all('rows' not in r for r in rows), '')


def test_exp9_source_audit_rejects_mixed_revisions():
    print('\nTest: Exp 9 source audit rejects mixed revisions')
    from run_budget_frontier import audit_join_sources
    records = [_audit_record('exp4_mqar', commit='a' * 40),
               _audit_record('exp5_ablation', commit='b' * 40),
               _audit_record('exp8_mechanism_ablation')]
    _rows, summary = audit_join_sources(records)
    _ok('mixed commits -> mixed_revisions',
        summary['status'] == 'mixed_revisions', summary['status'])
    _ok('mixed revisions are not comparable for ranking',
        summary['comparable_for_ranking'] is False, '')
    _ok('both commits are recorded',
        summary['git_commits'] == ['a' * 40, 'b' * 40], summary['git_commits'])
    _ok('warning names the mixed revisions',
        any('different git commits' in w for w in summary['warnings']), '')


def test_exp9_source_audit_rejects_missing_provenance_and_seed_mismatch():
    print('\nTest: Exp 9 source audit rejects missing provenance / seeds')
    from run_budget_frontier import audit_join_sources
    no_prov = [_audit_record('exp4_mqar'),
               _audit_record('exp5_ablation', commit=None),
               _audit_record('exp8_mechanism_ablation')]
    _rows, summary = audit_join_sources(no_prov)
    _ok('missing commit -> provenance_missing',
        summary['status'] == 'provenance_missing', summary['status'])
    _ok('missing provenance is named',
        summary['missing_provenance'] == ['exp5_ablation'],
        summary['missing_provenance'])
    _ok('ranking is blocked', summary['comparable_for_ranking'] is False, '')
    disjoint = [_audit_record('exp4_mqar', seeds=(42, 43)),
                _audit_record('exp5_ablation', seeds=(44, 45)),
                _audit_record('exp8_mechanism_ablation', seeds=(46, 47))]
    _rows, summary = audit_join_sources(disjoint)
    _ok('disjoint seeds -> no_shared_seeds',
        summary['status'] == 'no_shared_seeds', summary['status'])
    _ok('no shared seed is reported', summary['shared_seeds'] == [], '')


def test_exp9_source_audit_rejects_partial_seed_metadata():
    print('\nTest: Exp 9 source audit blocks partial seed evidence')
    from run_budget_frontier import audit_join_sources
    records = [_audit_record('exp4_mqar', seeds=(42, 43)),
               _audit_record('exp5_ablation', seeds=()),
               _audit_record('exp8_mechanism_ablation', seeds=(42, 43))]
    _rows, summary = audit_join_sources(records)
    _ok('a source without seeds -> partial_seeds',
        summary['status'] == 'partial_seeds', summary['status'])
    _ok('partial seed evidence blocks ranking',
        summary['comparable_for_ranking'] is False, '')
    _ok('seed status still records the overlap',
        summary['seed_status'] == 'partial'
        and summary['shared_seeds'] == [42, 43],
        f"{summary['seed_status']} {summary['shared_seeds']}")
    _ok('warning names the source without seeds',
        any('partial seed metadata' in w and 'exp5_ablation' in w
            for w in summary['warnings']), summary['warnings'])
    # Partial metadata with no actual intersection is still the stronger
    # no-shared-seed verdict.
    disjoint = [_audit_record('exp4_mqar', seeds=(42,)),
                _audit_record('exp5_ablation', seeds=()),
                _audit_record('exp8_mechanism_ablation', seeds=(43,))]
    _rows, summary = audit_join_sources(disjoint)
    _ok('partial + disjoint seeds -> no_shared_seeds',
        summary['status'] == 'no_shared_seeds', summary['status'])


def test_exp9_source_audit_flags_dirty_tree_and_backend_mix():
    print('\nTest: Exp 9 source audit distinguishes dirty tree/backend mix')
    from run_budget_frontier import audit_join_sources
    dirty = [_audit_record('exp4_mqar', dirty=True),
             _audit_record('exp5_ablation'),
             _audit_record('exp8_mechanism_ablation')]
    _rows, summary = audit_join_sources(dirty)
    _ok('same commit + dirty -> comparable_dirty_tree',
        summary['status'] == 'comparable_dirty_tree', summary['status'])
    _ok('dirty tree still permits ranking with a caveat',
        summary['comparable_for_ranking'] is True
        and any('dirty checkout' in w for w in summary['warnings']), '')
    mixed = [_audit_record('exp4_mqar'),
             _audit_record('exp5_ablation', backends={
                 'kda_backend_resolved': 'fla',
                 'mamba_backend_resolved': 'reference',
                 'rwkv_backend_resolved': 'reference'}),
             _audit_record('exp8_mechanism_ablation')]
    _rows, summary = audit_join_sources(mixed)
    _ok('backend mismatch -> mixed_backends',
        summary['status'] == 'mixed_backends', summary['status'])
    _ok('mismatching backend is recorded',
        len(summary['backend_mismatches']) == 1
        and summary['backend_mismatches'][0]['backend']
        == 'kda_backend_resolved', summary['backend_mismatches'])


def test_exp9_load_join_source_reads_exp5_sidecar():
    print('\nTest: Exp 9 loader reads legacy Exp 5 result + provenance sidecar')
    import hashlib
    import json
    import tempfile
    from run_budget_frontier import _load_join_source
    with tempfile.TemporaryDirectory() as td:
        result_path = os.path.join(td, 'exp5_ablation.json')
        prov_path = os.path.join(td, 'exp5_ablation_provenance.json')
        payload = [{'ratio': '3:1:1', 'mean_acc': 0.5}]
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        with open(prov_path, 'w', encoding='utf-8') as f:
            json.dump({'git_commit': 'c' * 40, 'git_dirty': False,
                       'seeds': [42, 43], 'kda_backend': 'reference'}, f)
        rec = _load_join_source('exp5_ablation', result_path,
                                provenance_path=prov_path)
        digest = hashlib.sha256(open(result_path, 'rb').read()).hexdigest()
        _ok('legacy bare array is readable',
            rec['readable'] is True and rec['n_rows'] == 1, '')
        _ok('legacy schema version is 0.0',
            rec['schema_version'] == '0.0', rec['schema_version'])
        _ok('sidecar supplies commit and seeds',
            rec['git_commit'] == 'c' * 40 and rec['seeds'] == [42, 43],
            f"{rec['git_commit']} {rec['seeds']}")
        _ok('file hash is recorded', rec['sha256'] == digest, '')


def test_exp9_quality_join_carries_source_provenance():
    print('\nTest: Exp 9 quality join annotates rows with source provenance')
    from run_budget_frontier import build_quality_join, mqar_spec
    rec = _audit_record('exp5_ablation')
    rec['rows'] = [{'ratio': '3:1:1', 'mean_acc': 0.5, 'ci95_acc': 0.05,
                    'n_seeds_ok': 5, 'mean_fwd_ms': 1.0}]
    rows, skipped = build_quality_join(mqar_spec(), source_records=[rec])
    _ok('one joined row, no skip', len(rows) == 1 and not skipped, f'{skipped}')
    row = rows[0]
    _ok('joined row carries commit/dirty/seeds/hash',
        row['source_git_commit'] == 'a' * 40
        and row['source_git_dirty'] is False
        and row['source_seeds'] == [42, 43, 44, 45, 46]
        and row['source_sha256'] == '0' * 64, f'{row}')


def test_exp9_quality_join_keeps_mamba2_rows():
    print('\nTest: Exp 9 quality join keeps the Mamba-2 memory route')
    from run_budget_frontier import build_quality_join, mqar_spec
    rec = _audit_record('exp4_mqar')
    rec['rows'] = [
        {'op': 'kda', 'mean_acc': 0.5, 'ci95_acc': 0.05,
         'n_seeds_ok': 5, 'mean_fwd_ms': 1.0},
        {'op': 'mamba2', 'mean_acc': 0.4, 'ci95_acc': 0.05,
         'n_seeds_ok': 5, 'mean_fwd_ms': 1.1},
    ]
    rows, skipped = build_quality_join(mqar_spec(), source_records=[rec])
    labels = {r['label'] for r in rows}
    _ok('both memory routes are joined', labels == {'kda', 'mamba2'},
        f'{labels} skipped={skipped}')
    mamba = next(r for r in rows if r['label'] == 'mamba2')
    _ok('Mamba-2 row is priced on the O(1) state read',
        abs(mamba['total_read_elements_per_query'] - 2048.0) < 1e-9,
        f'{mamba["total_read_elements_per_query"]}')


def test_exp9_ranking_summary_carries_comparability():
    print('\nTest: Exp 9 ranking summary carries the comparability verdict')
    from run_budget_frontier import (LAYOUTS, build_ranking_summary,
                                     mqar_spec, paper_spec)
    join_rows = []
    for i, ratio in enumerate(LAYOUTS):
        join_rows.append({
            'source': 'exp5_ablation',
            'label': ':'.join(str(x) for x in ratio),
            'mean_acc': 0.10 + 0.05 * i,
        })
    summary = {
        'status': 'mixed_revisions',
        'comparable_for_ranking': False,
        'git_commits': ['a' * 40, 'b' * 40],
        'shared_seeds': [42, 43],
    }
    rows = build_ranking_summary(
        join_rows, {'mqar': mqar_spec(), 'paper': paper_spec()},
        comparability=summary)
    _ok('one row per geometry and context view', len(rows) == 4, len(rows))
    _ok('every ranking row carries the source verdict',
        all(r['source_comparability'] == 'mixed_revisions'
            and r['comparable_across_sources'] is False
            and r['source_git_commits'] == ['a' * 40, 'b' * 40]
            for r in rows), '')


_TEST_FUNCS = [
    test_sliding_window_entries_hand_values,
    test_op_budget_hand_values_mqar_geometry,
    test_budget_scaling_claims,
    test_op_and_layout_dispatch,
    test_instant_budget_hand_values,
    test_layout_memory_route_substitution,
    test_prefill_decode_unification_identity,
    test_windowed_budget_matches_instant_sums,
    test_decode_frontier_structural_claims,
    test_spearman_rank_correlation,
    test_delta_rule_recent_write_identity,
    test_delta_rule_orthonormal_keys_exact_recall,
    test_parse_ratio_rejects_invalid_layouts,
    test_build_budget_curves_rows,
    test_decode_budget_curve_and_sweep_join,
    test_decode_sweep_smoke,
    test_exp9_source_audit_same_revision_is_comparable,
    test_exp9_source_audit_rejects_mixed_revisions,
    test_exp9_source_audit_rejects_missing_provenance_and_seed_mismatch,
    test_exp9_source_audit_rejects_partial_seed_metadata,
    test_exp9_source_audit_flags_dirty_tree_and_backend_mix,
    test_exp9_load_join_source_reads_exp5_sidecar,
    test_exp9_quality_join_carries_source_provenance,
    test_exp9_quality_join_keeps_mamba2_rows,
    test_exp9_ranking_summary_carries_comparability,
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
