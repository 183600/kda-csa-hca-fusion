"""Experiment 5 — hybrid layout ablation (multi-seed, device-aware).

Sweeps over different KDA:CSA:HCA ratios in the fused hybrid block and
measures (a) end-to-end forward latency and (b) MQAR accuracy with the same
training recipe as Experiment 4. This quantifies the design trade-off:

  * more KDA layers  -> faster, lower memory, but weaker long-range recall
  * more CSA layers   -> better sparse recall, but slower (top-k selection)
  * more HCA layers   -> cheapest global context, but coarser compression

Kaggle / review-driven additions (address reviewer concerns):

  * **Multi-seed runs.** Each ratio is trained over ``n_seeds`` seeds with
    mean +/- CI95 reported. The original paper's single-seed table had
    3:1:1=0.078 vs chance=0.0625 — within noise. Multi-seed makes the
    comparison statistically meaningful (or honestly shows it is not).
  * **More training steps.** The original 25 steps was too short for reliable
    convergence. We use 100+ steps and report convergence curves.
  * **Controlled depth.** Every ratio now contains exactly five layers, so
    ratio effects are no longer confounded with 3/4/5/6-layer model depth.
    ``n_params`` is still reported because operator types have different
    parameter counts.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    configure_torch_for_device, parse_int_env, sanitize_for_json,
    write_json_atomic, write_results_json, capture_provenance,
    make_seeded_generator,
)
from ops_fused import HybridKCHAttention, HybridConfig
from ops_kda_backend import validate_kda_backend
from run_quality import (
    make_mqar_batch, MQARHead, _parse_nkv_list, _fmt_tstat, _t_crit_975,
    _bonferroni_crit_q, _build_param_groups, SMALL_MODEL_SPEC,
)

logger = logging.getLogger(__name__)

VOCAB = 16
SEQ_LEN = 16


def _make_cfg(d_model=32, ratio=(3, 1, 1)):
    if not (isinstance(ratio, tuple) and len(ratio) == 3):
        raise ValueError(
            f"ratio must be a 3-tuple (n_kda, n_csa, n_hca), got {ratio!r}")
    if not all(isinstance(n, int) and n >= 0 for n in ratio):
        raise ValueError(
            f"ratio components must be non-negative ints, got {ratio!r}")
    n_kda, n_csa, n_hca = ratio
    spec = SMALL_MODEL_SPEC
    return HybridConfig(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=spec['csa_m'], csa_topk=spec['csa_topk'],
        csa_nh=spec['csa_nh'], csa_c=spec['csa_c'],
        csa_dc=spec['csa_dc'], csa_nIh=spec['csa_nIh'],
        csa_cI=spec['csa_cI'],
        csa_sliding_window=spec['csa_sliding_window'],
        hca_m2=spec['hca_m2'], hca_nh=spec['hca_nh'],
        hca_c=spec['hca_c'], hca_dc=spec['hca_dc'],
        hca_sliding_window=spec['hca_sliding_window'],
        n_kda=n_kda, n_csa=n_csa, n_hca=n_hca,
        kda_backend=validate_kda_backend(
            os.environ.get('KDA_BACKEND', 'reference')
        ),
    )


def _eval_model(model, head, embed, seq_len, n_kv=1, device='cpu',
                n_batches=4, batch=64):
    was_training = {m: m.training for m in (model, head, embed)}
    try:
        model.eval()
        head.eval()
        embed.eval()
        correct, total = 0, 0
        losses = []
        eval_gen = make_seeded_generator(12345, device=device)
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, VOCAB, embed, device, generator=eval_gen)
                model.reset_state()
                h = model(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
        if total == 0 or not losses:
            return 0.0, 0.0
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def eval_layout(ratio, d_model=32, seq_len=SEQ_LEN, n_kv=1, steps=100, lr=3e-3, seed=42,
                device='cpu', eval_batches=4, eval_batch=64, train_batch=None):
    if isinstance(device, str):
        device = torch.device(device)
    torch.manual_seed(seed)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio)
    total = sum(ratio)
    model = HybridKCHAttention(cfg, total_layers=total).to(device)
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)

    if train_batch is None:
        train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    model.train()
    head.train()
    embed.train()

    losses = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(ratio={ratio}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
        opt.zero_grad()
        loss.backward()
        bad_grads = [p for p in params
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grads:
            raise RuntimeError(
                f"non-finite gradient at step {step} in {len(bad_grads)} "
                f"params (ratio={ratio}, seed={seed}); aborting this seed "
                f"to prevent NaN propagation into parameters")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

    final_acc, final_loss = _eval_model(
        model, head, embed, seq_len, n_kv, device,
        n_batches=eval_batches, batch=eval_batch,
    )

    lat_gen = make_seeded_generator(99, device=device)
    x = torch.randn(1, seq_len, d_model, device=device, generator=lat_gen) * 0.1
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.reset_state()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                model.reset_state()
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / 5 * 1e3
    finally:
        model.train(was_training)

    last_loss = losses[-1] if losses else 0.0
    mean_last10 = sum(losses[-10:]) / min(10, len(losses)) if losses else 0.0
    return {
        'ratio': f'{ratio[0]}:{ratio[1]}:{ratio[2]}',
        'n_kv': n_kv,
        'layout': layout,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'fwd_ms': fwd_ms,
        'n_params': n_params,
        'n_layers': total,
        'seed': seed,
        'steps': steps,
        'train_batch': train_batch,
        'last_train_loss': last_loss,
        'mean_last10_loss': mean_last10,
        'loss_curve': losses,
    }


def eval_layout_multi_seed(ratio, n_seeds=5, steps=100, device='cpu', **kw):
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    per_seed = []
    for s in seeds:
        t0 = time.time()
        try:
            r = eval_layout(ratio, seed=s, steps=steps, device=device, **kw)
            r['train_time_s'] = time.time() - t0
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"fwd={r['fwd_ms']:.2f}ms  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            ratio_str = f'{ratio[0]}:{ratio[1]}:{ratio[2]}' if isinstance(ratio, tuple) else str(ratio)
            per_seed.append({
                'ratio': ratio_str,
                'n_kv': kw.get('n_kv', 1),
                'layout': None,
                'final_acc': None,
                'final_loss': None,
                'fwd_ms': None,
                'n_params': None,
                'n_layers': None,
                'seed': s,
                'steps': steps,
                'train_batch': kw.get('train_batch'),
                'last_train_loss': None,
                'mean_last10_loss': None,
                'loss_curve': [],
                'train_time_s': time.time() - t0,
                'error': str(e),
            })
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        raise RuntimeError(f"all {n_seeds} seeds failed for ratio {ratio}; "
                           f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    fwds = [r['fwd_ms'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_fwd = sum(fwds) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        std_acc = math.sqrt(var_acc)
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
    else:
        std_acc = 0.0
        ci_acc = None

    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    return {
        'ratio': ok_per_seed[0]['ratio'],
        'n_kv': ok_per_seed[0]['n_kv'],
        'layout': ok_per_seed[0]['layout'],
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds': len(per_seed),
        'n_seeds_total': len(per_seed),
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        'mean_train_time_s': sum(r.get('train_time_s', 0.0) for r in ok_per_seed) / n,
    }


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 5: Hybrid Layout Ablation (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')
    n_seeds = parse_int_env('ABL_SEEDS', 7, min_value=1, logger=logger)
    steps = parse_int_env('ABL_STEPS', 100, min_value=1, logger=logger)
    n_kv_list = _parse_nkv_list('ABL_NKV', '1')
    for n_kv in n_kv_list:
        if 2 * n_kv > VOCAB:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"VOCAB={VOCAB}; reduce n_kv or increase VOCAB.")
        if 2 * n_kv >= SEQ_LEN:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< SEQ_LEN={SEQ_LEN} (need room for the cue token at the end).")
    logger.info(f'  n_seeds={n_seeds}, steps={steps}, n_kv={n_kv_list}')
    ratios = [
        (3, 1, 1),
        (4, 1, 0), (4, 0, 1),
        (2, 2, 1), (2, 1, 2),
        (1, 3, 1), (1, 1, 3),
    ]
    if any(sum(r) != 5 for r in ratios):
        raise AssertionError('ablation layouts must all have total depth 5')
    n_tests = len(ratios) * len(n_kv_list)
    alpha_corrected = 0.05 / n_tests
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=exact-beta-bisection)')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for r in ratios:
            logger.info(f'\n-- ratio KDA:CSA:HCA = {r[0]}:{r[1]}:{r[2]} '
                        f'(n_kv={n_kv}, {n_seeds} seeds) --')
            try:
                res = eval_layout_multi_seed(r, n_seeds=n_seeds, steps=steps,
                                             device=device, n_kv=n_kv)
                t_stat = res.get('t_stat_vs_chance')
                n_ok = res.get('n_seeds_ok', 0)
                if t_stat is not None and n_ok >= 2:
                    crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
                    res['t_crit_bonferroni'] = crit
                    res['significant_bonferroni'] = (
                        crit is not None and t_stat > crit
                    )
                else:
                    res['t_crit_bonferroni'] = None
                    res['significant_bonferroni'] = False
                all_results.append(res)
                logger.info(f"  layout={res['layout']}  n_params={res['n_params']}  n_layers={res['n_layers']}")
                _ci = res['ci95_acc']
                _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
                logger.info(f"  -> mean_acc={res['mean_acc']:.4f} +/- {_ci_str} "
                            f"(std={res['std_acc']:.4f}, t_vs_chance={_fmt_tstat(res['t_stat_vs_chance'], width=0, prec=2)}, "
                            f"sig_bonferroni={res['significant_bonferroni']})")
                logger.info(f"     mean_fwd={res['mean_fwd_ms']:.2f}ms")
            except Exception as e:
                import traceback as _tb
                logger.error(f"  ratio {r[0]}:{r[1]}:{r[2]} FAILED: {e}")
                _tb.print_exc()
                all_results.append({
                    'ratio': f'{r[0]}:{r[1]}:{r[2]}',
                    'n_kv': n_kv,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    'layout': None,
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_fwd_ms': None,
                    'n_params': None,
                    'n_layers': sum(r),
                    'chance_acc': 1.0 / VOCAB,
                    't_stat_vs_chance': None,
                    't_crit_bonferroni': None,
                    'significant_bonferroni': False,
                    'mean_train_time_s': None,
                    'per_seed': [],
                })

    print('\n' + '=' * 112)
    print(f"{'n_kv':>4} | {'ratio':>8} | {'layout':>22} | {'layers':>6} | {'params':>8} | "
          f"{'mean_acc':>10} | {'+/- CI95':>10} | {'t_vs_chance':>12} | {'fwd_ms':>8}")
    print('-' * 112)
    for r in all_results:
        if 'error' in r:
            print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {'(error)':>22} | {r['n_layers']:>6} | "
                  f"{'-':>8} | {'-':>10} | {'-':>10} | {'-':>12} | {'-':>8}   ERROR: {r['error']}")
            continue
        layout_str = r.get('layout', '') or ''
        n_params = r['n_params'] if r.get('n_params') is not None else 0
        mean_acc = r['mean_acc'] if r.get('mean_acc') is not None else 0.0
        ci95 = r['ci95_acc']
        fwd = r['mean_fwd_ms'] if r.get('mean_fwd_ms') is not None else 0.0
        ci_str = f"{ci95:>10.4f}" if ci95 is not None else f"{'n/a':>10}"
        print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {layout_str:>22} | {r['n_layers']:>6} | "
              f"{n_params:>8} | {mean_acc:>10.4f} | {ci_str} | "
              f"{_fmt_tstat(r.get('t_stat_vs_chance'), width=12, prec=2)} | {fwd:>8.2f}")
    print(f"{'':>4} | {'chance':>8} | {'':>22} | {'':>6} | {'':>8} | "
          f"{1.0/VOCAB:>10.4f} | {'':>10} | {'':>12} | {'':>8}")

    logger.info('\nControlled-depth design: all reported ratios contain 5 layers.')
    logger.info('  Parameter counts can still differ by operator type and are reported;')
    logger.info('  interpret ratio effects together with n_params and paired seed data.')

    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results
                        if 'error' not in r), default=0)
    _c = 1.0 / VOCAB
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and 0.5 * _c < r['mean_acc'] < 1.5 * _c]
    far_below = [r for r in all_results
                 if 'error' not in r
                 and r.get('mean_acc') is not None
                 and r['mean_acc'] <= 0.5 * _c]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0
                         and len(near_chance) < len(all_results) // 2
                         and len(far_below) == 0)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ratios with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
    logger.info(f'  ratios near chance (0.5x-1.5x): {len(near_chance)}/{len(all_results)}')
    logger.info(f'  ratios far below chance (<=0.5x): {len(far_below)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: The ablation results do NOT support strong structural\n'
            '  conclusions. Either the seed count is too low (<5), no layout\n'
            '  reaches Bonferroni significance, or most accuracies are near\n'
            '  chance. Treat the ranking as exploratory, not confirmatory.\n'
            '  To improve power: increase ABL_SEEDS (>=7), increase ABL_STEPS,\n'
            '  or use a simpler task where the signal is stronger.')
    logger.info('=' * 70)
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds
        r['csa_indexer_normalize_qk'] = True
        r['kda_backend'] = os.environ.get('KDA_BACKEND', 'reference')
        r['significance_scope'] = 'vs_chance_baseline_not_pairwise_between_layouts'

    os.makedirs('results', exist_ok=True)
    write_results_json(all_results, 'results/exp5_ablation.json',
                       logger=logger)
    try:
        write_results_json(capture_provenance(),
                           'results/exp5_ablation_provenance.json',
                           logger=logger)
    except Exception as e:
        logger.warning(f'failed to write provenance: {e}')
    logger.info('\nSaved: results/exp5_ablation.json')
    n_errors = sum(1 for r in all_results if 'error' in r)
    n_incomplete = sum(
        1 for r in all_results
        if 'error' not in r
        and r.get('n_seeds_ok', n_seeds) != r.get('n_seeds', n_seeds)
    )
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_results)} layouts errored and '
            f'{n_incomplete}/{len(all_results)} have incomplete seed sets. '
            'Returning non-zero to prevent survivor-only aggregates from '
            'being treated as a successful experiment.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
