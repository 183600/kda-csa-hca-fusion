"""Regression tests for ``cache_accounting.py`` (live KV-cache audit).

What is pinned here
-------------------
1. **Analytical decomposition == Exp 3.** The component-wise
   ``analytic_cache_components`` must sum exactly to the committed
   ``run_kv_cache.kv_cache_elements(op, T, mode='full_accounting')`` for
   every operator and a sweep of context lengths (including T=0 and
   partial blocks).
2. **Live == analytical, for every fill history.** A real
   ``run_decoding.HybridDecoding`` prefilled in bulk and token-by-token
   must retain exactly the analytical element counts (allocated AND valid
   prefix), including geometric spare capacity, ring buffers, partial
   tails and CSA overlap state — batch>1 included.
3. **Geometric capacity contract.** Both caches must round every
   allocation up to the geometric capacity regardless of fill history;
   ``ops_decoding_cache``'s local helper must equal
   ``run_kv_cache.geometric_capacity``.
4. **Autograd hygiene.** ``HybridDecoding`` must not thread KDA recurrent
   states without detaching, or it would retain one autograd graph per
   decoded token (O(N) memory growth without ``no_grad()``). The
   graph-growth probe must report a bounded node count; CSA/HCA cache
   tensors must be detached; the graph-node counter itself is pinned with
   a positive control.
5. **Paper-aligned byte model.** The DeepSeek-V4 §2.3.4 mixed storage
   policy (BF16 on RoPE dims, FP8 elsewhere; FP8 indexer) must compute
   exact per-element byte costs and pool totals, and reject unknown
   policies.
6. **Paper cache-block structure.** ``lcm(m, m')`` blocks yield exactly
   ``k1 = lcm/m`` CSA rows and ``k2 = lcm/m'`` HCA rows; a live bulk
   prefill at a block boundary must match.

Run via pytest::

    pytest -q test_cache_accounting.py

Or directly::

    python3 test_cache_accounting.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import cache_accounting  # noqa: E402
from ops_fused import HybridConfig  # noqa: E402
from run_kv_cache import (geometric_capacity, kv_cache_breakdown,
                          kv_cache_elements)  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _cfg(**over) -> HybridConfig:
    base = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8, kda_chunk_size=0,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=5,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=5,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=4, hca_rope_dim=4, kda_backend='reference',
    )
    base.update(over)
    return HybridConfig(**base)


def _decoder(cfg, device='cpu'):
    from run_decoding import HybridDecoding
    n_total = cfg.n_kda + cfg.n_csa + cfg.n_hca
    return HybridDecoding(cfg=cfg, total_layers=n_total).to(device).eval()


def _fill(decoder, cfg, T, *, batch=1, incremental=False, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, T, cfg.d_model) * 0.1
    with torch.no_grad():
        decoder.reset()
        if incremental:
            for t in range(T):
                decoder(x[:, t:t + 1])
        else:
            decoder(x)
    return x


def test_analytic_decomposition_matches_exp3():
    """Component sums == committed ``full_accounting`` totals."""
    print("\nTest: analytical decomposition vs Exp 3 accounting")
    cfg = _cfg()
    geo = cache_accounting.exp3_geometry(cfg)
    failures = []
    for T in (0, 1, 3, 4, 5, 8, 27, 31, 100):
        comps = cache_accounting.analytic_cache_components(cfg, T)
        sink = comps['sink_logits']['elements_allocated']
        for op in ('kda', 'csa', 'hca'):
            total = sum(c['elements_allocated'] for c in comps.values()
                        if c['op'] == op)
            if op == 'csa':
                total += cfg.n_csa * cfg.csa_nh
            elif op == 'hca':
                total += cfg.n_hca * cfg.hca_nh
            committed = kv_cache_elements(op, T, mode='full_accounting',
                                          **geo)
            if total != committed:
                failures.append((T, op, total, committed))
        # The hybrid total must equal the sum of the three op totals.
        hybrid = kv_cache_elements('hybrid_kch', T, mode='full_accounting',
                                   **geo)
        total_all = sum(c['elements_allocated'] for c in comps.values()
                        if c['op'] != 'sink') + sink
        if total_all != hybrid:
            failures.append((T, 'hybrid_kch', total_all, hybrid))
    _ok('analytic decomposition == Exp 3 full_accounting',
        not failures, f'failures={failures[:3]}')

    raised = 0
    try:
        cache_accounting.analytic_cache_components(cfg, -1)
    except ValueError:
        raised += 1
    try:
        cache_accounting.exp3_geometry(cfg, conv_ksize=3)['kda_conv_ksize']
    except Exception:
        raised += 1
    _ok('analytic validation (T<0 rejected)', raised == 1, f'raised={raised}')


def test_kv_cache_breakdown_reconciles_in_both_modes():
    """``state_components`` sum to ``kv_elements`` in BOTH modes."""
    print("\nTest: kv_cache_breakdown vs kv_cache_elements invariant")
    ops = ('softmax_gqa', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca',
           'hybrid_kch', 'hybrid_mamba2', 'hybrid_rwkv7')
    failures = []
    for T in (0, 1, 2, 3, 7, 8, 9, 16, 17, 31, 512):
        for op in ops:
            for mode in ('compressed_kv_only', 'full_accounting'):
                comps = kv_cache_breakdown(op, T, mode=mode)
                total = sum(v for k, v in comps.items()
                            if k != 'compressed_kv_only_total')
                committed = kv_cache_elements(op, T, mode=mode)
                if total != committed:
                    failures.append((T, op, mode, total, committed))
    _ok('components reconcile with kv_elements in both modes',
        not failures, f'failures={failures[:3]}')
    raised = 0
    try:
        kv_cache_breakdown('kda', 8, mode='bogus')
    except ValueError:
        raised += 1
    _ok('unknown mode rejected', raised == 1, f'raised={raised}')


def test_live_equals_analytic_for_every_fill_history():
    """Live tensors match the analytical model for bulk and incremental runs.

    The critical property: the retained capacity is a function of
    the completed-block count only, so both fill histories produce exactly
    the same counts.
    """
    print("\nTest: live cache accounting == analytical (fill histories)")
    cfg = _cfg()
    T = 27  # 6 CSA blocks + 3 tail, 3 HCA blocks + 3 tail, SW full (win=5)

    for label, incremental in (('bulk', False), ('incremental', True)):
        dec = _decoder(cfg)
        _fill(dec, cfg, T, incremental=incremental, seed=3)
        live = cache_accounting.live_cache_components(dec)
        analytic = cache_accounting.analytic_cache_components(cfg, T)
        bad = []
        for name, a in analytic.items():
            rec = live['components'].get(name)
            l_alloc = rec['elements_allocated_per_sequence'] if rec else 0.0
            l_valid = rec['elements_valid_per_sequence'] if rec else 0.0
            if (l_alloc, l_valid) != (a['elements_allocated'],
                                      a['elements_valid']):
                bad.append((name, l_alloc, l_valid, a['elements_allocated'],
                            a['elements_valid']))
        audit = cache_accounting.audit_cache_accounting(
            dec, cfg, T=T, live=live)
        _ok(f'live==analytic [{label}]',
            not bad and audit['consistent'] and audit['mismatches'] == [],
            f'bad={bad[:2]}, consistent={audit["consistent"]}, '
            f'mismatches={audit["mismatches"][:2]}')
        json.dumps(live, allow_nan=False)  # must be JSON-strict

    # Batch > 1: per-sequence figures must divide by B exactly.
    cfg_b = _cfg()
    dec_b = _decoder(cfg_b)
    _fill(dec_b, cfg_b, T, batch=2, seed=4)
    live_b = cache_accounting.live_cache_components(dec_b)
    audit_b = cache_accounting.audit_cache_accounting(dec_b, cfg_b, T=T,
                                                      live=live_b)
    _ok('live==analytic [batch=2]',
        live_b['batch_size'] == 2 and audit_b['consistent'],
        f"B={live_b['batch_size']}, consistent={audit_b['consistent']}")

    # Windows disabled: no SW components, analytical reports zero.
    cfg_nw = _cfg(csa_sliding_window=0, hca_sliding_window=0)
    dec_nw = _decoder(cfg_nw)
    _fill(dec_nw, cfg_nw, T, seed=5)
    audit_nw = cache_accounting.audit_cache_accounting(dec_nw, cfg_nw, T=T)
    _ok('live==analytic [win=0]',
        audit_nw['consistent']
        and audit_nw['per_component']['csa_sliding_window']
        ['live_allocated'] == 0
        and audit_nw['per_component']['hca_sliding_window']
        ['live_allocated'] == 0,
        f"consistent={audit_nw['consistent']}, "
        f"sw_live={audit_nw['per_component']['csa_sliding_window']}")

    # Op absent from the layout: CSA removed; audit needs explicit T.
    cfg_nc = _cfg(n_kda=1, n_csa=0, n_hca=1)
    dec_nc = _decoder(cfg_nc)
    _fill(dec_nc, cfg_nc, T, seed=6)
    audit_nc = cache_accounting.audit_cache_accounting(dec_nc, cfg_nc, T=T)
    _ok('live==analytic [no CSA]',
        audit_nc['consistent']
        and audit_nc['per_operator']['csa']['live_per_layer'] is None,
        f"consistent={audit_nc['consistent']}")

    # T mismatch and empty decoder must raise.
    raised = 0
    dec_mismatch = _decoder(cfg)
    _fill(dec_mismatch, cfg, T, seed=13)
    try:
        cache_accounting.audit_cache_accounting(dec_mismatch, cfg, T=T + 1)
    except ValueError:
        raised += 1
    try:
        cache_accounting.live_cache_components(_decoder(cfg))
    except ValueError:
        raised += 1
    _ok('audit validation (T mismatch / empty decoder)', raised == 2,
        f'raised={raised}')


def test_geometric_capacity_contract():
    """Compressed storage capacity is history-independent and geometric."""
    print("\nTest: geometric compressed-cache capacity contract")
    from ops_decoding_cache import _geometric_capacity

    n_range = list(range(0, 65))
    same = all(_geometric_capacity(n) == geometric_capacity(n)
               for n in n_range)
    _ok('helper == run_kv_cache.geometric_capacity', same,
        f'_geometric_capacity(63)={_geometric_capacity(63)}, '
        f'geometric_capacity(63)={geometric_capacity(63)}')

    cfg = _cfg()
    T = 100  # 25 CSA blocks, 12 HCA blocks
    for label, incremental in (('bulk', False), ('incremental', True)):
        dec = _decoder(cfg)
        _fill(dec, cfg, T, incremental=incremental, seed=7)
        csa = dec._csa_caches[0]
        hca = dec._hca_caches[0]
        expected_csa = geometric_capacity(T // cfg.csa_m)
        expected_hca = geometric_capacity(T // cfg.hca_m2)
        valid = (
            csa.n_blocks == T // cfg.csa_m
            and csa._C_comp.shape[1] == expected_csa
            and csa._K_IComp.shape[1] == expected_csa
            and hca.n_blocks == T // cfg.hca_m2
            and hca._C_comp.shape[1] == expected_hca
        )
        _ok(f'capacity rounded to geometric [{label}]', valid,
            f'csa_cap={csa._C_comp.shape[1]} (expected {expected_csa}), '
            f'hca_cap={hca._C_comp.shape[1]} (expected {expected_hca})')

    bad = 0
    for args in ((-1,), ('x',), (True,)):
        try:
            _geometric_capacity(*args)
        except (TypeError, ValueError):
            bad += 1
    _ok('_geometric_capacity validation', bad == 3, f'raised={bad}')


def test_run_kv_cache_validation():
    """Public cache-accounting helpers reject bool / non-int geometry."""
    print("\nTest: run_kv_cache helper validation")
    from run_kv_cache import (causal_block_entries,
                              causal_selected_entries)

    cases = [
        ('geometric_capacity(True)', lambda: geometric_capacity(True)),
        ('causal_block_entries(True, 4)',
         lambda: causal_block_entries(True, 4)),
        ('causal_block_entries(4, True)',
         lambda: causal_block_entries(4, True)),
        ('causal_selected_entries(True, 4, 2)',
         lambda: causal_selected_entries(True, 4, 2)),
        ('causal_selected_entries(4, 4, True)',
         lambda: causal_selected_entries(4, 4, True)),
        ('kv_cache_elements(T=-1)',
         lambda: kv_cache_elements('csa', -1)),
        ('kv_cache_elements(T=True)',
         lambda: kv_cache_elements('csa', True)),
        ('kv_cache_elements(T=1.5)',
         lambda: kv_cache_elements('csa', 1.5)),
    ]
    bad = 0
    for _name, fn in cases:
        try:
            fn()
        except (TypeError, ValueError):
            bad += 1
    _ok('all invalid geometry raises', bad == len(cases),
        f'raised={bad}/{len(cases)}')
    zero_cache = kv_cache_elements('csa', 0, mode='compressed_kv_only')
    _ok('T=0 remains valid',
        geometric_capacity(0) == 0
        and causal_block_entries(0, 4) == 0
        and causal_selected_entries(0, 4, 2) == 0
        and isinstance(zero_cache, int) and zero_cache >= 0,
        f'kv_cache_elements={zero_cache!r}')


def test_autograd_hygiene_and_bounded_graph():
    """Decode caches are detached; retained graph is BOUNDED across steps."""
    print("\nTest: decode autograd hygiene + bounded graph growth")
    from cache_accounting import _reachable_graph_nodes

    # Positive control for the graph-node counter itself.
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    z = y.sum()
    ctrl = _reachable_graph_nodes(z) > 0 and _reachable_graph_nodes(y) > 0
    _ok('graph-node counter positive control', ctrl,
        f'z nodes={_reachable_graph_nodes(z)}, y nodes='
        f'{_reachable_graph_nodes(y)}')
    _ok('detached tensor has zero reachable nodes',
        _reachable_graph_nodes(z.detach()) == 0, '')

    cfg = _cfg()

    # no_grad run: every cache tensor is graph-free.
    dec = _decoder(cfg)
    _fill(dec, cfg, 16, seed=8)
    hygiene = cache_accounting.autograd_hygiene(dec)
    _ok('no_grad decode cache is clean',
        hygiene['clean'] and hygiene['reachable_graph_nodes'] == 0,
        f"clean={hygiene['clean']}, "
        f"nodes={hygiene['reachable_graph_nodes']}")

    # Grad-enabled run: CSA/HCA caches must STILL be detached (they
    # self-detach); the KDA state is allowed a single-step graph, but the
    # probe must show it does not accumulate across steps.
    probe = cache_accounting.autograd_graph_probe(
        cfg, prefill_len=4, steps=(4, 12), seed=9)
    _ok('graph growth probe is bounded',
        probe['bounded'] and probe['growth_per_step'] == 0,
        f"nodes={probe['reachable_nodes']}, "
        f"growth/step={probe['growth_per_step']}")

    dec_g = _decoder(cfg)
    torch.manual_seed(10)
    xg = (torch.randn(1, 8, cfg.d_model) * 0.1).requires_grad_(True)
    with torch.enable_grad():
        dec_g.reset()
        dec_g(xg)
        for _ in range(3):
            dec_g(xg[:, :1])
        live = cache_accounting.live_cache_components(dec_g)
    cache_comps = {
        name: rec for name, rec in live['components'].items()
        if rec['pool'] != cache_accounting.PARAMETER
    }
    non_kda_bearing = [name for name, rec in cache_comps.items()
                       if not name.startswith('kda_') and rec['has_grad_fn']]
    kda_bearing = cache_comps['kda_recurrent_state']['has_grad_fn']
    _ok('CSA/HCA cache tensors stay detached under enable_grad',
        not non_kda_bearing and bool(kda_bearing),
        f'non_kda_graph_bearing={non_kda_bearing}, '
        f'kda_state_graph={kda_bearing}')

    raised = 0
    try:
        cache_accounting.autograd_graph_probe(cfg, steps=(0, 4))
    except ValueError:
        raised += 1
    try:
        cache_accounting.autograd_graph_probe(cfg, steps=(4,))
    except ValueError:
        raised += 1
    _ok('graph probe validation', raised == 2, f'raised={raised}')


def test_paper_aligned_bytes():
    """DeepSeek-V4 §2.3.4 mixed-storage bytes are exact and pooled."""
    print("\nTest: paper-aligned byte model (mixed FP8 / BF16 RoPE)")
    cfg = _cfg(csa_c=16, hca_c=16, csa_cI=4, csa_rope_dim=4, hca_rope_dim=4)
    comps = cache_accounting.analytic_cache_components(cfg, T=64)
    mixed = cache_accounting.paper_aligned_bytes(
        cfg, comps, policy='mixed_fp8')
    pure = cache_accounting.paper_aligned_bytes(
        cfg, comps, policy='pure_bf16')

    # Hand-computed: compressed CSA rows = geometric_capacity(16) * 16.
    n_rows = geometric_capacity(64 // cfg.csa_m)
    csa_elems = n_rows * cfg.csa_c
    per_entry = (2 * 4 + (16 - 4)) / 16  # = 1.25 bytes/element
    idx_elems = n_rows * cfg.csa_cI
    math_ok = (
        mixed['by_component']['csa_compressed_kv']['elements_allocated']
        == csa_elems
        and mixed['by_component']['csa_compressed_kv']['bytes_per_element']
        == per_entry
        and mixed['by_component']['csa_compressed_kv']['bytes']
        == csa_elems * per_entry
        and mixed['by_component']['csa_indexer_kv']['bytes_per_element']
        == 1.0
        and mixed['by_component']['csa_indexer_kv']['bytes'] == idx_elems
        and mixed['by_component']['kda_recurrent_state']['bytes_per_element']
        == 2.0
    )
    _ok('mixed-storage per-component arithmetic', math_ok,
        f"csa_per_entry={mixed['by_component']['csa_compressed_kv']['bytes_per_element']}, "
        f"idx_bytes={mixed['by_component']['csa_indexer_kv']['bytes']}")

    n_elements = sum(c['elements_allocated'] for c in comps.values())
    pools = mixed['bytes_by_pool']
    _ok('pool totals + pure-BF16 baseline',
        pure['total_bytes'] == 2.0 * n_elements
        and set(pools) == {cache_accounting.CLASSICAL_KV_CACHE,
                           cache_accounting.STATE_CACHE,
                           cache_accounting.PARAMETER}
        and 0.0 < mixed['ratio_vs_pure_bf16'] < 1.0,
        f"pure={pure['total_bytes']}, mixed={mixed['total_bytes']}, "
        f"ratio={mixed['ratio_vs_pure_bf16']:.4f}, pools={pools}")

    # Paper-scale geometry: c=512, rope_dim=64 -> 576/512 B per element.
    big = HybridConfig(d_model=64, n_heads_qk=2, n_heads_v=2,
                       head_dim_k=8, head_dim_v=8,
                       csa_c=512, csa_rope_dim=64, csa_nIh=1, csa_cI=32,
                       n_csa=1, n_kda=0, n_hca=0)
    big_comps = cache_accounting.analytic_cache_components(big, T=1024)
    big_mixed = cache_accounting.paper_aligned_bytes(
        big, big_comps, policy='mixed_fp8')
    _ok('paper-scale per-element bytes (c=512, RoPE=64)',
        big_mixed['by_component']['csa_compressed_kv']['bytes_per_element']
        == 576 / 512,
        f"per_element="
        f"{big_mixed['by_component']['csa_compressed_kv']['bytes_per_element']}")

    raised = 0
    try:
        cache_accounting.paper_aligned_bytes(cfg, comps, policy='fp4')
    except ValueError:
        raised += 1
    _ok('unknown byte policy rejected', raised == 1, f'raised={raised}')


def test_lcm_packing_and_full_report():
    """Paper lcm(m, m') blocks + one full JSON-strict audit report."""
    print("\nTest: lcm block packing + full audit report")
    cfg = _cfg()
    pack = cache_accounting.lcm_packing(cfg, T=24)
    pack_ok = (
        pack['lcm_tokens'] == math.lcm(cfg.csa_m, cfg.hca_m2) == 8
        and pack['k1_csa_rows_per_block'] == 2
        and pack['k2_hca_rows_per_block'] == 1
        and pack['complete_lcm_blocks'] == 3
        and pack['csa_blocks_expected'] == 6
        and pack['hca_blocks_expected'] == 3
    )
    dec = _decoder(cfg)
    _fill(dec, cfg, 24, seed=11)
    live_blocks = (dec._csa_caches[0].n_blocks, dec._hca_caches[0].n_blocks)
    _ok('lcm packing matches live block counts',
        pack_ok and live_blocks == (6, 3),
        f"pack={pack}, live_blocks={live_blocks}")

    report = cache_accounting.run_cache_audit(
        dec, cfg, T=24, with_graph_probe=True,
        graph_probe_kwargs={'prefill_len': 4, 'steps': (2, 6), 'seed': 12})
    strict = True
    try:
        json.dumps(report, allow_nan=False)
    except ValueError:
        strict = False
    expected_keys = {'context_len', 'batch_size', 'layout', 'components',
                     'totals', 'comparison', 'autograd_hygiene', 'bytes',
                     'paper_alignment', 'autograd_graph_probe'}
    _ok('full audit report is JSON-strict with all blocks',
        strict and expected_keys <= set(report)
        and report['comparison']['consistent']
        and report['paper_alignment']['lcm_packing'] == pack,
        f"strict={strict}, keys={sorted(report)}")

    # End-to-end model probe (tiny LM): audit attaches at T=context+decode.
    from train_lm_autodl import LMWithHybrid
    cfg_lm = _cfg(kda_chunk_size=4)
    model = LMWithHybrid(17, cfg_lm).eval()
    model_report = cache_accounting.run_cache_audit_for_model(
        model, context_len=16, decode_steps=1, with_graph_probe=False)
    _ok('run_cache_audit_for_model (LMWithHybrid)',
        model_report['context_len'] == 17
        and model_report['comparison']['consistent'],
        f"T={model_report['context_len']}, "
        f"consistent={model_report['comparison']['consistent']}")

    # The training-probe entry point must attach the audit and stay
    # JSON-serializable (decode_consistency.json contract).
    import train_lm_autodl
    probe = train_lm_autodl.run_decode_consistency_probe(
        model, 'cpu', seq_len=8, seed=21)
    json.dumps(probe, allow_nan=False)
    attached = probe.get('cache_audit')
    _ok('run_decode_consistency_probe attaches cache_audit',
        attached is not None and attached['context_len'] == 8
        and attached['comparison']['consistent'],
        f"keys={'cache_audit' in probe}, "
        f"T={attached['context_len'] if attached else None}")

    raised = 0
    try:
        cache_accounting.lcm_packing(cfg, T=-1)
    except ValueError:
        raised += 1
    try:
        cache_accounting.run_cache_audit_for_model(model, context_len=0)
    except ValueError:
        raised += 1
    _ok('report validation', raised == 2, f'raised={raised}')


_TEST_FUNCS = [
    test_analytic_decomposition_matches_exp3,
    test_kv_cache_breakdown_reconciles_in_both_modes,
    test_live_equals_analytic_for_every_fill_history,
    test_geometric_capacity_contract,
    test_run_kv_cache_validation,
    test_autograd_hygiene_and_bounded_graph,
    test_paper_aligned_bytes,
    test_lcm_packing_and_full_report,
]


def main() -> int:
    print('=' * 70)
    print('Live KV-cache accounting / autograd-hygiene Regression Tests')
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
