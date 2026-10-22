"""Regression tests for ``run_mechanism_ablation.py`` (Exp 8).

The tests never touch the real ``results/`` directory: the end-to-end smoke
writes to a per-test temp path.

Run via pytest::

    pytest -q test_mechanism_ablation.py

Or directly::

    python3 test_mechanism_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ops_csa  # noqa: E402
import run_mechanism_ablation as M  # noqa: E402
from ops_fused import CSAHybridLayer, HybridKCHAttention  # noqa: E402
from run_quality import make_kv_update_batch, make_mqar_batch  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _set_env(env):
    """Apply env overrides, returning a restore callable."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    HybridKCHAttention._stateful_warned = True
    CSAHybridLayer.reset_warned()

    def _restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return _restore


def test_variant_configs_valid_and_param_matched():
    """Every variant builds a valid config; param counts are identical.

    The clean-ablation claim of Exp 8 is that only mechanism switches
    change; if a variant silently changed the parameter count (e.g. by
    swapping an operator) the paired comparison would be confounded.
    """
    print("\nTest: Exp 8 variant configs valid and parameter-matched")
    counts = {}
    for v in M.VARIANTS:
        cfg = M.make_variant_cfg(v)
        ok_rope = ((cfg.csa_rope_dim is None) if not v['rope'] else
                   cfg.csa_rope_dim == 64)
        _ok(f"{v['name']}: rope switch applied",
            ok_rope and ((cfg.hca_rope_dim is None) if not v['rope']
                         else cfg.hca_rope_dim == 64),
            f"csa={cfg.csa_rope_dim}, hca={cfg.hca_rope_dim}")
        _ok(f"{v['name']}: topk switch applied",
            cfg.csa_topk == v['topk'], f"topk={cfg.csa_topk}")
        model = HybridKCHAttention(cfg, total_layers=sum(M.RATIO))
        counts[v['name']] = sum(p.numel() for p in model.parameters())
        n_csa = M.set_indexer_ste(model, v['use_ste'])
        _ok(f"{v['name']}: one CSA layer for the diagnostic",
            n_csa == 1, f'n_csa={n_csa}')
    _ok('all variants share the same parameter count',
        len(set(counts.values())) == 1, f'{counts}')
    _ok('dense variant selects all compressed blocks',
        next(v for v in M.VARIANTS if v['name'] == 'dense')['topk']
        == M._N_BLOCKS, f"n_blocks={M._N_BLOCKS}")


def test_indexer_topk_matches_operator():
    """``indexer_topk_from_layer`` reproduces the operator's top-k indices.

    The diagnostic recomputes the lightning-indexer ranking outside
    ``naive_csa``; if the two ever diverge (normalization, mask, anchor
    order), the reported gold-block recall would describe a different
    selection than the model actually uses. This is the pin.
    """
    print("\nTest: diagnostic indexer top-k matches naive_csa's selection")
    torch.manual_seed(0)
    device = torch.device('cpu')
    d_model = 32
    embed = nn.Embedding(M.VOCAB, d_model)
    cfg = M.make_variant_cfg(M.VARIANTS[0])
    model = HybridKCHAttention(cfg, total_layers=sum(M.RATIO))
    model.eval()
    x_emb, _target, _cue, _meta = make_mqar_batch(
        3, M.SEQ_LEN, 1, M.VOCAB, embed, device,
        generator=torch.Generator(device).manual_seed(7), return_meta=True)

    csa_layer = next(layer for layer in model.layers
                     if isinstance(layer, CSAHybridLayer))
    captured = {}
    orig = ops_csa.csa_lightning_indexer

    def _spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        idx = out[0] if isinstance(out, tuple) else out
        captured.setdefault('indices', idx.detach().clone())
        captured.setdefault('x', hook_holder.get('x'))
        return out

    hook_holder = {}

    def _hook(_module, args):
        hook_holder['x'] = args[0].detach()

    handle = csa_layer.register_forward_pre_hook(_hook)
    ops_csa.csa_lightning_indexer = _spy
    try:
        with torch.no_grad():
            model.reset_state()
            model(x_emb)
    finally:
        ops_csa.csa_lightning_indexer = orig
        handle.remove()

    _ok('operator selection captured', 'indices' in captured, 'spy')
    diag = M.indexer_topk_from_layer(csa_layer, captured['x'])
    _ok('diagnostic indices equal the operator indices',
        torch.equal(diag, captured['indices']),
        f"diag[0]={diag[0].tolist()[:3]} op={captured['indices'][0].tolist()[:3]}")

    # The strictly-preceding causal mask contract: selected non-padded
    # blocks are always < t // m.
    T = diag.shape[1]
    t_idx = torch.arange(T)
    valid = diag >= 0
    causal_bound = (t_idx // cfg.csa_m)[None, :, None]   # [1, T, 1]
    inside = (~valid) | (diag < causal_bound)
    _ok('diagnostic selections respect the strictly-preceding mask',
        bool(inside.all()), 'causal mask')

    # Score path contract: return_scores=True must not change the hard
    # indices and must return the full block distribution used by the
    # ranking metrics.
    diag_s, soft = M.indexer_topk_from_layer(csa_layer, captured['x'],
                                             return_scores=True)
    _ok('return_scores keeps the hard indices bit-identical',
        torch.equal(diag_s, diag), 'scores path')
    _ok('soft weights shape [B, T, n_blocks]',
        soft.shape[0] == diag.shape[0] and soft.shape[1] == T
        and soft.shape[2] == M._N_BLOCKS, f'shape={tuple(soft.shape)}')
    cue = M.SEQ_LEN - 1
    cue_row_sum = float(soft.detach()[:, cue, :].sum(-1)[0])
    _ok('cue-row block distribution is normalised',
        abs(cue_row_sum - 1.0) < 1e-6, f'sum={cue_row_sum}')
    n_available = min(cue // cfg.csa_m, M._N_BLOCKS)
    _ok('masked (future) blocks get exactly zero probability',
        bool((soft[:, cue, n_available:] == 0).all()), 'mask zeros')


def test_make_mqar_batch_meta_backward_compatible():
    """``return_meta`` is opt-in and reports the cued pair index."""
    print("\nTest: make_mqar_batch return_meta contract")
    device = torch.device('cpu')
    embed = nn.Embedding(M.VOCAB, 32)
    out3 = make_mqar_batch(4, M.SEQ_LEN, 1, M.VOCAB, embed, device,
                           generator=torch.Generator(device).manual_seed(1))
    _ok('default return is the historical 3-tuple',
        isinstance(out3, tuple) and len(out3) == 3, f'len={len(out3)}')
    x_emb, target, cue_pos, meta = make_mqar_batch(
        4, M.SEQ_LEN, 2, M.VOCAB, embed, device,
        generator=torch.Generator(device).manual_seed(1), return_meta=True)
    key_idx = meta['key_idx']
    _ok('key_idx shape and range',
        key_idx.shape == (4,) and int(key_idx.min()) >= 0
        and int(key_idx.max()) < 2, f'{key_idx.tolist()}')
    _ok('cue_pos is the last position',
        cue_pos == M.SEQ_LEN - 1, f'cue_pos={cue_pos}')
    # Same generator seed -> same batch content regardless of meta flag.
    x_emb2, target2, _ = make_mqar_batch(
        4, M.SEQ_LEN, 2, M.VOCAB, embed, device,
        generator=torch.Generator(device).manual_seed(1))
    _ok('meta flag does not change the batch',
        torch.equal(x_emb, x_emb2) and torch.equal(target, target2), '')


def test_make_kv_update_batch_targets_latest_value():
    """The kv_update label is the NEW value of the key repeated at the cue."""
    print("\nTest: make_kv_update_batch target contract")
    device = torch.device('cpu')
    embed = nn.Embedding(32, 32)
    for n_kv in (1, 2, 4):
        x_emb, target, cue_pos, meta = make_kv_update_batch(
            16, 32, n_kv, 32, embed, device,
            generator=torch.Generator(device).manual_seed(n_kv),
            return_meta=True)
        ids = meta['input_ids']
        j = meta['key_idx']
        b = ids.shape[0]
        keys = ids[:, 0:2 * n_kv:2]
        vals_old = ids[:, 1:2 * n_kv:2]
        keys2 = ids[:, 2 * n_kv:4 * n_kv:2]
        vals2 = ids[:, 2 * n_kv + 1:4 * n_kv:2]
        ok = True
        for r in range(b):
            cue = ids[r, cue_pos]
            if int(cue) != int(keys[r, j[r]]):
                ok = False
                break
            slot = int((keys2[r] == cue).nonzero()[0, 0])
            if int(target[r]) != int(vals2[r, slot]):
                ok = False
                break
            if int(target[r]) == int(vals_old[r, j[r]]):
                ok = False
                break
        _ok(f'n_kv={n_kv}: target is the cued key\'s newest value',
            ok, f'target={target.tolist()[:4]}')
    _ok('cue_pos is the last position',
        cue_pos == 31, f'cue_pos={cue_pos}')


def test_dense_variant_recall_is_one():
    """With topk = n_blocks the diagnostic must hit every gold block."""
    print("\nTest: dense variant gold recall == 1.0")
    torch.manual_seed(0)
    device = torch.device('cpu')
    embed = nn.Embedding(M.VOCAB, 32)
    dense = next(v for v in M.VARIANTS if v['name'] == 'dense')
    cfg = M.make_variant_cfg(dense)
    model = HybridKCHAttention(cfg, total_layers=sum(M.RATIO))
    model.eval()
    diag = M.diagnose_gold_recall(
        model, embed, device=device, m=cfg.csa_m,
        n_batches=2, batch=16, seed=3)
    _ok('recall is exactly 1.0',
        diag['gold_recall'] == 1.0, f"recall={diag['gold_recall']}")
    _ok('chance recall is also 1.0 for dense selection',
        diag['chance_gold_recall'] == 1.0,
        f"chance={diag['chance_gold_recall']}, "
        f"k={diag['gold_topk_effective']}/{diag['gold_n_available_blocks']}")
    # Threshold-free ranking metrics must be well-formed even when the
    # hard top-k recall is trivially 1.0 (the ranking can still be poor).
    _ok('gold ranking metrics well-formed',
        diag['gold_pr_auc'] is not None
        and 0.0 <= diag['gold_pr_auc'] <= 1.0
        and diag['gold_roc_auc'] is not None
        and 0.0 <= diag['gold_roc_auc'] <= 1.0
        and diag['gold_mrr'] is not None
        and 0.0 < diag['gold_mrr'] <= 1.0
        and diag['gold_top1_recall'] is not None
        and 0.0 <= diag['gold_top1_recall'] <= 1.0,
        f"pr_auc={diag['gold_pr_auc']} mrr={diag['gold_mrr']}")
    expected_chance_ap = (1 + 0.5 + 1 / 3) / 3
    _ok('ranking chance references use the exact random-AP expectation',
        abs(diag['chance_gold_pr_auc'] - expected_chance_ap) < 1e-9
        and abs(diag['chance_gold_mrr'] - expected_chance_ap) < 1e-9
        and abs(diag['chance_gold_roc_auc'] - 0.5) < 1e-9,
        f"chance_pr={diag['chance_gold_pr_auc']} "
        f"chance_mrr={diag['chance_gold_mrr']}")
    _ok('one-gold-block macro AP equals MRR (per-query metrics)',
        abs(diag['gold_pr_auc'] - diag['gold_mrr']) < 1e-12,
        f"ap={diag['gold_pr_auc']} mrr={diag['gold_mrr']}")


def test_sparse_variant_recall_is_below_one_or_none():
    """topk1 diagnostic: recall <= 1/3 chance-level bound is not assumed."""
    print("\nTest: sparse (k=1) diagnostic is well-formed")
    torch.manual_seed(0)
    device = torch.device('cpu')
    embed = nn.Embedding(M.VOCAB, 32)
    v = next(v for v in M.VARIANTS if v['name'] == 'topk1')
    cfg = M.make_variant_cfg(v)
    model = HybridKCHAttention(cfg, total_layers=sum(M.RATIO))
    model.eval()
    diag = M.diagnose_gold_recall(
        model, embed, device=device, m=cfg.csa_m,
        n_batches=1, batch=16, seed=3)
    _ok('recall in [0, 1]',
        diag['gold_recall'] is not None and 0.0 <= diag['gold_recall'] <= 1.0,
        f"recall={diag['gold_recall']}")
    _ok('chance recall == 1/3 for k=1 of 3 blocks',
        abs(diag['chance_gold_recall'] - 1 / 3) < 1e-9,
        f"chance={diag['chance_gold_recall']}")
    _ok('sparse probe reports threshold-free ranking metrics',
        diag['gold_pr_auc'] is not None
        and 0.0 <= diag['gold_pr_auc'] <= 1.0
        and diag['gold_mrr'] is not None
        and abs(diag['chance_gold_pr_auc']
                - (1 + 0.5 + 1 / 3) / 3) < 1e-9,
        f"pr_auc={diag['gold_pr_auc']} mrr={diag['gold_mrr']}")


def test_paired_stats_math():
    """Paired summary: mean / zero-variance / single-pair behavior."""
    print("\nTest: paired_stats math")
    zero_var = M.paired_stats([0.1, 0.1, 0.1, 0.1, 0.1])
    _ok('zero-variance differences: mean exact, t undefined',
        abs(zero_var['mean'] - 0.1) < 1e-12 and zero_var['t_stat'] is None
        and zero_var['ci95'] == 0.0 and zero_var['n_pairs'] == 5,
        f'{zero_var}')
    single = M.paired_stats([0.2])
    _ok('single pair: undefined CI and t',
        single['mean'] == 0.2 and single['ci95'] is None
        and single['t_stat'] is None, f'{single}')
    empty = M.paired_stats([])
    _ok('empty: all None',
        empty == {'n_pairs': 0, 'mean': None, 'ci95': None,
                  't_stat': None}, f'{empty}')
    # mean=1, std=0.5, n=3 -> t = 1 / (0.5/sqrt(3)) = 3.4641...
    stats = M.paired_stats([0.5, 1.0, 1.5])
    _ok('t-statistic matches the closed form',
        abs(stats['t_stat'] - (1.0 / (0.5 / (3 ** 0.5)))) < 1e-9,
        f"t={stats['t_stat']}")


def test_ablation_design_audit():
    """The audit distinguishes complete leave-one-out from partial subsets."""
    print("\nTest: Exp 8 ablation-design audit")
    full = M.ablation_design_audit(M.VARIANTS)
    _ok('full variant table is leave-one-out complete',
        full['status'] == 'leave_one_out_complete'
        and full['missing_single_factor_variants'] == [],
        f"status={full['status']}")
    _ok('every factor has a one-factor counterpart',
        all(f['covered'] for f in full['factors'].values()),
        str({k: v['one_factor_variants']
             for k, v in full['factors'].items()}))
    _ok('no_rope is the rope one-factor variant',
        full['factors']['rope']['one_factor_variants'] == ['no_rope'], '')
    _ok('dense and topk1 cover the topk factor',
        set(full['factors']['topk']['one_factor_variants'])
        == {'dense', 'topk1'}, '')
    _ok('frozen_indexer is the ste one-factor variant',
        full['factors']['ste']['one_factor_variants'] == ['frozen_indexer'], '')
    _ok('no_rope_dense is an interaction cell',
        full['interaction_variants'] ==
        {'no_rope_dense': ['rope', 'topk']},
        str(full['interaction_variants']))

    subset = [v for v in M.VARIANTS
              if v['name'] in ('baseline', 'no_rope')]
    partial = M.ablation_design_audit(subset, requested=['baseline',
                                                         'no_rope'])
    _ok('subset missing one-factor counterparts is flagged incomplete',
        partial['status'] == 'incomplete'
        and partial['missing_single_factor_variants'] == ['topk', 'ste'],
        f"status={partial['status']} "
        f"missing={partial['missing_single_factor_variants']}")
    _ok('subset records the selection provenance',
        'MECH_VARIANTS subset' in partial['selection'], '')

    no_base = [v for v in M.VARIANTS if v['name'] in ('no_rope', 'topk1')]
    _ok('variant set without baseline is flagged',
        M.ablation_design_audit(no_base)['status'] == 'no_baseline', '')

    rows = [{'variant': 'baseline', 'n_params': 100, 'n_layers': 5},
            {'variant': 'no_rope', 'n_params': 100, 'n_layers': 5}]
    audit = M.ablation_design_audit(subset, rows=rows)
    cv = audit['controlled_variables']
    _ok('observed parameter / depth parity cross-checked',
        cv['parameter_count_parity'] is True and cv['depth_parity'] is True,
        str(cv))
    bad = M.ablation_design_audit(
        subset, rows=[{'variant': 'baseline', 'n_params': 100,
                       'n_layers': 5},
                      {'variant': 'no_rope', 'n_params': 101,
                       'n_layers': 5}])
    _ok('parameter-count drift detected',
        bad['controlled_variables']['parameter_count_parity'] is False, '')

    def _controlled_rows(steps_b=200):
        return [
            {'variant': 'baseline', 'n_params': 100, 'n_layers': 5,
             'steps': 200, 'train_batch': 16, 'lr': 3e-3,
             'seeds': [42, 43],
             'per_seed': [{'seed': 42, 'final_acc': 0.2},
                          {'seed': 43, 'final_acc': 0.3}]},
            {'variant': 'no_rope', 'n_params': 100, 'n_layers': 5,
             'steps': steps_b, 'train_batch': 16, 'lr': 3e-3,
             'seeds': [42, 43],
             'per_seed': [{'seed': 42, 'final_acc': 0.1},
                          {'seed': 43, 'final_acc': 0.2}]},
        ]

    controlled = M.ablation_design_audit(
        subset, rows=_controlled_rows(),
        recipe={'steps': 200, 'train_batch': 16, 'lr': 3e-3})
    cv = controlled['controlled_variables']
    _ok('matching recipe and seed set verified as controlled',
        cv['controls_status'] == 'controlled'
        and all(v is True for v in cv['recipe_parity'].values())
        and cv['seed_alignment_ok'] is True
        and cv['common_requested_seeds'] == [42, 43], str(cv))

    drifted = M.ablation_design_audit(subset, rows=_controlled_rows(100))
    _ok('training-budget drift detected',
        drifted['controlled_variables']['controls_status']
        == 'recipe_mismatch'
        and drifted['controlled_variables']['recipe_parity']['steps']
        is False, str(drifted['controlled_variables']))

    unpaired_rows = _controlled_rows()
    unpaired_rows[1]['per_seed'] = unpaired_rows[1]['per_seed'][:1]
    unpaired = M.ablation_design_audit(subset, rows=unpaired_rows)
    _ok('failed/missing paired seed detected',
        unpaired['controlled_variables']['controls_status']
        == 'seed_mismatch'
        and unpaired['controlled_variables']['seed_alignment_ok'] is False,
        str(unpaired['controlled_variables']))


def test_exp8_smoke_envelope():
    """End-to-end smoke: tiny run writes a v1.0 envelope with the right kind."""
    print("\nTest: Exp 8 end-to-end smoke (1 seed x 1 step)")
    restore = _set_env({'MECH_SEEDS': '1', 'MECH_STEPS': '1',
                        'MECH_VARIANTS': 'baseline,no_rope',
                        'MECH_TRAIN_BATCH': '4'})
    saved_path = M.RESULTS_PATH
    try:
        with tempfile.TemporaryDirectory(prefix='_mech_test_') as tmp:
            M.RESULTS_PATH = os.path.join(tmp, 'exp8.json')
            # Smoke runs must never append a fake row to the campaign
            # ledger: keep the manifest inside the temp directory too.
            manifest_env = _set_env({
                'RUN_MANIFEST_PATH': os.path.join(tmp, 'run_manifest.jsonl')})
            try:
                rc = M.main()
            finally:
                manifest_env()
            _ok('main returns 0 for a finite tiny run', rc == 0, f'rc={rc}')
            with open(M.RESULTS_PATH, encoding='utf-8') as f:
                payload = json.load(f)
            _ok('envelope kind + schema version',
                payload.get('kind') == 'exp8_mechanism_ablation'
                and payload.get('schema_version') == '1.0',
                f"kind={payload.get('kind')}")
            rows = payload.get('results', [])
            _ok('one row per variant', len(rows) == 2,
                f'n_rows={len(rows)}')
            baseline = next(r for r in rows if r['variant'] == 'baseline')
            _ok('baseline row has finite acc and recall in [0, 1]',
                baseline['mean_acc'] is not None
                and baseline['mean_gold_recall'] is not None
                and 0 <= baseline['mean_gold_recall'] <= 1,
                f"acc={baseline['mean_acc']}, "
                f"recall={baseline['mean_gold_recall']}")
            _ok('baseline row carries aggregated gold ranking metrics',
                baseline.get('mean_gold_pr_auc') is not None
                and 0 <= baseline['mean_gold_pr_auc'] <= 1
                and baseline.get('mean_gold_mrr') is not None
                and 0 < baseline['mean_gold_mrr'] <= 1
                and baseline.get('chance_gold_pr_auc') is not None,
                f"pr_auc={baseline.get('mean_gold_pr_auc')}, "
                f"mrr={baseline.get('mean_gold_mrr')}")
            no_rope = next(r for r in rows if r['variant'] == 'no_rope')
            _ok('paired stats present for the non-baseline variant',
                no_rope['paired_vs_baseline_mean_acc_diff'] is not None, '')
            _ok('paired PR-AUC contrast present for the primary metric',
                no_rope.get('paired_vs_baseline_mean_gold_pr_auc_diff')
                is not None
                and no_rope.get('paired_vs_baseline_gold_pr_auc_n_pairs') == 1
                and no_rope.get('paired_pr_auc_significant_bonferroni')
                is False, f"{no_rope.get('paired_vs_baseline_mean_gold_pr_auc_diff')}")
            _ok('design metadata documents the diagnostic',
                'gold_recall_definition' in payload.get('metadata', {})
                and 'caveats' in payload.get('metadata', {}), '')
            audit = payload.get('metadata', {}).get('ablation_design', {})
            _ok('ablation-design audit embedded in metadata',
                audit.get('status') == 'incomplete'
                and audit.get('missing_single_factor_variants')
                == ['topk', 'ste']
                and audit.get('controlled_variables', {})
                .get('parameter_count_parity') is True,
                f"status={audit.get('status')} "
                f"missing={audit.get('missing_single_factor_variants')}")
    finally:
        M.RESULTS_PATH = saved_path
        restore()


def test_single_seed_aggregate_reports_undefined_std():
    """One surviving seed must not claim zero sample variance."""
    mean, std, ci = M._mean_ci([0.5])
    _ok('single value mean is defined', mean == 0.5, mean)
    _ok('single value std is undefined', std is None, std)
    _ok('single value CI is undefined', ci is None, ci)
    mean, std, ci = M._mean_ci([0.4, 0.6])
    _ok('two values keep a finite std', std is not None and std > 0, std)


_TEST_FUNCS = [
    test_variant_configs_valid_and_param_matched,
    test_indexer_topk_matches_operator,
    test_make_mqar_batch_meta_backward_compatible,
    test_make_kv_update_batch_targets_latest_value,
    test_dense_variant_recall_is_one,
    test_sparse_variant_recall_is_below_one_or_none,
    test_paired_stats_math,
    test_ablation_design_audit,
    test_exp8_smoke_envelope,
    test_single_seed_aggregate_reports_undefined_std,
]


def main() -> int:
    print('=' * 70)
    print('run_mechanism_ablation (Exp 8) Regression Tests')
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
