"""Experiment 8 — CSA mechanism ablation: partial RoPE, top-k block
selection, and lightning-indexer learning (STE).

Motivation
----------
Experiments 4/5 answer *which operators* help on MQAR (softmax / KDA / CSA /
HCA / hybrid layouts). They do not answer *which mechanisms inside CSA* do
the work. Three design choices from DeepSeek-V4's CSA are cheap to toggle
and were never measured in this repository:

1. **Partial RoPE** (paper §2.3.3). ``csa_rope_dim`` / ``hca_rope_dim`` can
   be set to ``None`` to disable the relative-position rotation on the
   core-attention queries / compressed KV / sliding-window keys and the
   output inverse rotation — with zero parameter-count change.
2. **Top-k block selection sparsity.** ``csa_topk`` selects ``k`` of the
   ``n_blocks = seq_len / csa_m`` strictly-preceding compressed blocks per
   query. ``k=1`` is maximally sparse, ``k=n_blocks`` degenerates CSA to
   *dense* attention over compressed entries (the sparse branch adds an
   explicit gather, not extra parameters).
3. **Indexer learning.** The lightning indexer is trained through a
   straight-through estimator (STE) around the non-differentiable
   ``torch.topk``. ``CSAHybridLayer.use_ste = False`` freezes the indexer
   at its random initialization (a documented ablation hook in
   ``ops_fused``): selection still happens, but it is not learned.

Design
------
All variants share the SAME 3:1:1 hybrid layout, depth (5 layers), and
parameter count — only the three mechanism switches change. Every variant is
trained from scratch on the Exp 4 MQAR task over ``MECH_SEEDS`` seeds with
the same per-seed data streams, so variant differences are PAIRED by seed
and the primary statistic is a two-sided paired t-test of the per-seed
accuracy difference against the baseline (Bonferroni-corrected across the
compared variants). This is a stronger design than an independent-samples
comparison and directly follows the Exp 5 validity policy.

Mechanistic diagnostic: indexer gold-block recall
-------------------------------------------------
Accuracy alone cannot tell whether CSA contributes retrieval or the KDA
layers compensate when its selection is broken. We therefore instrument the
CSA sub-layer: a forward pre-hook captures the (normalized) input to the
single CSA layer in the 3:1:1 stack, and ``indexer_topk_from_layer``
recomputes the exact lightning-indexer top-k from that input and the
layer's own weights (same compression, same L2-normalized cosine ranking,
same strictly-preceding causal block mask as ``naive_csa``). For each eval
sequence we then check whether the compressed block containing the cued
key/value pair is among the selected top-k blocks at the cue position.
``gold_recall`` is the fraction of sequences this succeeds on;
``chance_gold_recall = min(k, n_available) / n_available`` is the
closed-form chance level under uniform selection. This isolates *selection
quality* from end-task accuracy.

Budget / configuration (environment variables)
----------------------------------------------
* ``MECH_SEEDS``       (default 5)   — seeds per variant.
* ``MECH_STEPS``       (default 200) — training steps per run (matches the
  Exp 4 MQAR default; the hybrid is under-trained at 100 steps).
* ``MECH_TRAIN_BATCH`` (default 16)  — training batch size.
* ``MECH_NKV``         (default '1') — comma-separated MQAR n_kv values.
* ``MECH_VARIANTS``    (default '')  — optional comma-separated subset of
  variant names (must include ``baseline`` for the paired comparisons).

Output
------
``results/exp8_mechanism_ablation.json`` — versioned envelope
(``kind='exp8_mechanism_ablation'``) with one row per (variant, n_kv) and
the full per-seed records, plus a metadata block documenting the design,
the diagnostic definition, and the statistical scope.
"""

from __future__ import annotations

import dataclasses
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
    capture_provenance,
    configure_torch_for_device,
    make_seeded_generator,
    parse_int_env,
)
from ops_csa import _causal_block_mask, csa_compress_kv, csa_lightning_indexer
from ops_fused import CSAHybridLayer, HybridKCHAttention
from results_schema import write as write_schema
from run_ablation import _eval_model, _make_cfg
from run_quality import (
    MQARHead,
    SMALL_MODEL_SPEC,
    _bonferroni_crit_q,
    _build_param_groups,
    _parse_nkv_list,
    _t_crit_975,
    make_mqar_batch,
)

logger = logging.getLogger(__name__)

VOCAB = 16
SEQ_LEN = 16
RATIO = (3, 1, 1)
RESULTS_PATH = os.path.join('results', 'exp8_mechanism_ablation.json')

# Number of compressed CSA blocks at the default geometry. ``csa_m`` must
# divide ``seq_len`` for the cue-token diagnostic to see a whole number of
# blocks; validated below.
_N_BLOCKS = SEQ_LEN // SMALL_MODEL_SPEC['csa_m']
if SEQ_LEN % SMALL_MODEL_SPEC['csa_m'] != 0:
    raise ValueError(
        f"SEQ_LEN={SEQ_LEN} must be divisible by csa_m="
        f"{SMALL_MODEL_SPEC['csa_m']} for the block definitions in Exp 8.")

# Variant table. ``topk`` is the CSA selection width; ``dense`` uses all
# compressed blocks (no sparse selection). ``rope`` toggles partial RoPE on
# BOTH CSA and HCA (the paired core-attention stack) — ``no_rope`` therefore
# isolates "partial RoPE in the CSA/HCA pair", not RoPE-in-CSA alone.
VARIANTS: tuple[dict, ...] = (
    {'name': 'baseline', 'rope': True, 'topk': SMALL_MODEL_SPEC['csa_topk'],
     'use_ste': True,
     'question': 'reference: RoPE on, sparse top-k, learned indexer'},
    {'name': 'no_rope', 'rope': False, 'topk': SMALL_MODEL_SPEC['csa_topk'],
     'use_ste': True,
     'question': 'does partial RoPE (CSA+HCA) help MQAR?'},
    {'name': 'topk1', 'rope': True, 'topk': 1, 'use_ste': True,
     'question': 'does extreme selection sparsity (k=1) hurt?'},
    {'name': 'dense', 'rope': True, 'topk': _N_BLOCKS, 'use_ste': True,
     'question': 'is sparse selection better than dense compressed attention?'},
    {'name': 'frozen_indexer', 'rope': True,
     'topk': SMALL_MODEL_SPEC['csa_topk'], 'use_ste': False,
     'question': 'does LEARNING the top-k selection (STE) matter?'},
    {'name': 'no_rope_dense', 'rope': False, 'topk': _N_BLOCKS,
     'use_ste': True,
     'question': 'interaction: dense attention without positional rotation'},
)
VARIANT_NAMES = tuple(v['name'] for v in VARIANTS)


def make_variant_cfg(variant: dict, d_model: int = 32):
    """Build the HybridConfig for one mechanism variant.

    Starts from the frozen Exp 5 config (``run_ablation._make_cfg``) so the
    hybrid geometry is identical to the layout ablation, then overrides ONLY
    the mechanism switches. ``dataclasses.replace`` re-runs
    ``HybridConfig.__post_init__``, so the override values are validated by
    the same guards as any other config.
    """
    cfg = _make_cfg(d_model, RATIO)
    rope_dim = 64 if variant['rope'] else None
    return dataclasses.replace(
        cfg,
        csa_topk=int(variant['topk']),
        csa_rope_dim=rope_dim,
        hca_rope_dim=rope_dim,
    )


def set_indexer_ste(model, use_ste: bool) -> int:
    """Set ``use_ste`` / ``indexer_is_trained`` on every CSA sub-layer.

    Returns the number of CSA layers touched so callers can assert the
    3:1:1 layout actually contains one.
    """
    n = 0
    for layer in model.layers:
        if isinstance(layer, CSAHybridLayer):
            layer.use_ste = bool(use_ste)
            layer.indexer_is_trained = bool(use_ste)
            n += 1
    return n


def indexer_topk_from_layer(layer: CSAHybridLayer, x: torch.Tensor) -> torch.Tensor:
    """Recompute the lightning-indexer top-k of ``layer`` on input ``x``.

    ``x`` must be the tensor that reaches ``CSAHybridLayer.forward`` — i.e.
    the post-norm hidden state — which is exactly what a forward pre-hook on
    the layer captures. The computation mirrors the indexer part of
    ``ops_csa.naive_csa`` (including the ``normalize_qk=True`` cosine
    ranking and the strictly-preceding causal block mask) from the layer's
    own weights, so it can be compared index-for-index against the operator
    (see ``test_mechanism_ablation.py``).

    Returns integer indices ``[B, T_padded, topk]`` (invalid slots ``-1``),
    matching ``csa_lightning_indexer``'s contract.
    """
    cfg = layer.cfg
    B, T, _d = x.shape
    m = cfg.csa_m
    pad = (-T) % m
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    T_padded = T + pad
    n_blocks = T_padded // m

    K_idx_raw = F.linear(x, layer.W_KV_idx.weight)
    Z_idx = F.linear(x, layer.W_Z_idx.weight)
    K_IComp = csa_compress_kv(K_idx_raw, Z_idx, layer.B_idx, m)
    cQ = F.linear(x, layer.W_DQ.weight)
    q_idx = F.linear(cQ, layer.W_IUQ.weight).view(
        B, T_padded, cfg.csa_nIh, cfg.csa_cI)
    w_idx = F.linear(x, layer.W_w.weight)
    cbm = _causal_block_mask(T_padded, n_blocks, m, x.device)
    return csa_lightning_indexer(
        q_idx, K_IComp, w_idx, cfg.csa_topk,
        scale=None,
        causal_block_mask=cbm,
        return_soft_weights=False,
        normalize_qk=True,
    )


def diagnose_gold_recall(model, embed, *, seq_len=SEQ_LEN, n_kv=1, device,
                         m=None, n_batches=2, batch=64, seed=42,
                         vocab=VOCAB):
    """Measure the CSA indexer's gold-block recall@k at the cue position.

    For each sequence: the cued KV pair occupies source positions
    ``2*j`` (key) and ``2*j+1`` (value); its gold compressed block is
    ``(2*j) // m``. A sequence is a hit iff that block appears among the
    top-k blocks selected for the cue token (the last position). Requires
    key and value to fall in the SAME block — otherwise the "gold block"
    is ambiguous and the diagnostic is skipped with a reason (rather than
    silently reporting a distorted number).

    Returns a dict with ``gold_recall`` (``None`` when skipped),
    ``n_samples``, ``gold_topk_effective``, ``gold_n_available_blocks``,
    ``chance_gold_recall`` and ``skipped_reason``.
    """
    if m is None:
        m = SMALL_MODEL_SPEC['csa_m']
    csa_layers = [layer for layer in model.layers
                  if isinstance(layer, CSAHybridLayer)]
    if len(csa_layers) != 1:
        return {
            'gold_recall': None, 'n_samples': 0,
            'gold_topk_effective': None, 'gold_n_available_blocks': None,
            'chance_gold_recall': None,
            'skipped_reason': (f'expected exactly 1 CSA layer in the '
                               f'{RATIO} layout, found {len(csa_layers)}'),
        }
    layer = csa_layers[0]
    captured: dict = {}

    def _capture(_module, args):
        captured['x'] = args[0].detach()

    handle = layer.register_forward_pre_hook(_capture)
    gen = make_seeded_generator(seed + 6_000_000, device=device)
    was_training = model.training
    hits = 0
    total = 0
    skipped_reason = None
    cue_pos = seq_len - 1
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, _target, cue_pos, meta = make_mqar_batch(
                    batch, seq_len, n_kv, vocab, embed, device,
                    generator=gen, return_meta=True)
                model.reset_state()
                model(x_emb)
                x_norm = captured.get('x')
                if x_norm is None:
                    skipped_reason = 'forward pre-hook captured no CSA input'
                    break
                key_idx = meta['key_idx']
                key_block = (2 * key_idx) // m
                value_block = (2 * key_idx + 1) // m
                if not torch.equal(key_block, value_block):
                    skipped_reason = (
                        'cued key and value land in different compressed '
                        'blocks; gold block undefined for this geometry')
                    break
                indices = indexer_topk_from_layer(layer, x_norm)
                selected = indices[:, cue_pos, :]
                hits += int((selected == key_block[:, None])
                            .any(dim=1).sum().item())
                total += int(key_block.numel())
    finally:
        handle.remove()
        model.train(was_training)

    if skipped_reason is not None:
        return {
            'gold_recall': None, 'n_samples': 0,
            'gold_topk_effective': None, 'gold_n_available_blocks': None,
            'chance_gold_recall': None,
            'skipped_reason': skipped_reason,
        }
    n_blocks = -(-seq_len // m)
    n_available = min(cue_pos // m, n_blocks)
    topk_effective = min(int(layer.cfg.csa_topk), n_available)
    return {
        'gold_recall': hits / total if total else None,
        'n_samples': total,
        'gold_topk_effective': topk_effective,
        'gold_n_available_blocks': n_available,
        'chance_gold_recall': (topk_effective / n_available
                               if n_available else None),
        'skipped_reason': None,
    }


def train_variant(variant: dict, *, seed, steps, device, seq_len=SEQ_LEN,
                  n_kv=1, d_model=32, eval_batches=4, eval_batch=64,
                  train_batch=None, diagnose=True, diag_batches=2,
                  diag_batch=64, lr=3e-3):
    """Train one variant for one seed and return its per-seed record.

    The training and evaluation protocol (optimizer, losses, RNG stream
    offsets, latency probe) mirrors ``run_ablation.eval_layout`` so the
    accuracy numbers are directly comparable to Exp 5 at the same budget.
    """
    if isinstance(device, str):
        device = torch.device(device)
    torch.manual_seed(seed)
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = make_variant_cfg(variant, d_model)
    total_layers = sum(RATIO)
    model = HybridKCHAttention(cfg, total_layers=total_layers).to(device)
    n_csa = set_indexer_ste(model, variant['use_ste'])
    if n_csa != 1:
        raise RuntimeError(
            f"variant {variant['name']}: expected 1 CSA layer, found {n_csa}")

    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]
    n_params = sum(p.numel() for p in model.parameters())

    if train_batch is None:
        train_batch = parse_int_env('MECH_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    batch_gen = make_seeded_generator(seed + 1_000_000, device=device)
    model.train()
    head.train()
    embed.train()
    losses = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device,
            generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(variant={variant['name']}, seed={seed})")
        opt.zero_grad()
        loss.backward()
        bad_grads = [p for p in params
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grads:
            raise RuntimeError(
                f"non-finite gradient at step {step} in {len(bad_grads)} "
                f"params (variant={variant['name']}, seed={seed})")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

    final_acc, final_loss = _eval_model(
        model, head, embed, seq_len, n_kv, device,
        n_batches=eval_batches, batch=eval_batch, seed=seed)

    # Latency probe: same protocol as Exp 5 (eval mode, single thread on
    # CPU) so mechanism-variant costs are comparable across the repo.
    lat_gen = make_seeded_generator(99, device=device)
    x = torch.randn(1, seq_len, d_model, device=device,
                    generator=lat_gen) * 0.1
    was_training = model.training
    model.eval()
    prev_threads = torch.get_num_threads()
    if device.type != 'cuda':
        torch.set_num_threads(1)
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
        if device.type != 'cuda':
            torch.set_num_threads(prev_threads)
        model.train(was_training)

    if diagnose:
        diag = diagnose_gold_recall(
            model, embed, seq_len=seq_len, n_kv=n_kv, device=device,
            m=cfg.csa_m, n_batches=diag_batches, batch=diag_batch, seed=seed)
    else:
        diag = {'gold_recall': None, 'n_samples': 0,
                'gold_topk_effective': None,
                'gold_n_available_blocks': None,
                'chance_gold_recall': None,
                'skipped_reason': 'diagnostic disabled by caller'}

    return {
        'variant': variant['name'],
        'n_kv': n_kv,
        'seed': seed,
        'steps': steps,
        'train_batch': train_batch,
        'rope': bool(variant['rope']),
        'topk': int(variant['topk']),
        'use_ste': bool(variant['use_ste']),
        'final_acc': final_acc,
        'final_loss': final_loss,
        'fwd_ms': fwd_ms,
        'n_params': n_params,
        'n_layers': total_layers,
        'last_train_loss': losses[-1] if losses else None,
        'mean_last10_loss': (sum(losses[-10:]) / min(10, len(losses))
                             if losses else None),
        'loss_curve': losses,
        'gold_recall': diag['gold_recall'],
        'gold_recall_n_samples': diag['n_samples'],
        'gold_topk_effective': diag['gold_topk_effective'],
        'gold_n_available_blocks': diag['gold_n_available_blocks'],
        'chance_gold_recall': diag['chance_gold_recall'],
        'gold_recall_skipped_reason': diag['skipped_reason'],
        'layout': model.layout_str(),
    }


def _mean_ci(values, label=''):
    """Return ``(mean, std, ci95)`` for a list of floats (empty -> Nones)."""
    n = len(values)
    if n == 0:
        return None, None, None
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    crit = _t_crit_975(n)
    ci = crit * std / math.sqrt(n) if crit is not None else None
    return mean, std, ci


def paired_stats(diffs):
    """Two-sided paired summary for a list of per-seed differences.

    Returns ``{'n_pairs', 'mean', 'ci95', 't_stat'}``; ``t_stat`` is
    ``None`` when the differences are degenerate (fewer than 2 pairs or
    zero variance), matching the repo-wide convention that an undefined
    statistic is ``None`` rather than a misleading 0.
    """
    n = len(diffs)
    if n == 0:
        return {'n_pairs': 0, 'mean': None, 'ci95': None, 't_stat': None}
    mean = sum(diffs) / n
    if n < 2:
        return {'n_pairs': n, 'mean': mean, 'ci95': None, 't_stat': None}
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    crit = _t_crit_975(n)
    ci = crit * std / math.sqrt(n) if crit is not None else None
    t_stat = mean / (std / math.sqrt(n)) if std > 0 else None
    return {'n_pairs': n, 'mean': mean, 'ci95': ci, 't_stat': t_stat}


def summarize_variant(per_seed, variant):
    """Aggregate per-seed records into the row stored in the result file."""
    ok = [r for r in per_seed
          if 'error' not in r
          and r.get('final_acc') is not None
          and math.isfinite(r['final_acc'])
          and r.get('final_loss') is not None
          and math.isfinite(r['final_loss'])]
    accs = [r['final_acc'] for r in ok]
    mean_acc, std_acc, ci_acc = _mean_ci(accs)
    mean_fwd, _std_fwd, _ = _mean_ci([r['fwd_ms'] for r in ok])

    recall_rows = [r for r in ok if r.get('gold_recall') is not None]
    recalls = [r['gold_recall'] for r in recall_rows]
    mean_recall, std_recall, ci_recall = _mean_ci(recalls)
    chance_recall = (recall_rows[0]['chance_gold_recall']
                     if recall_rows else None)
    if len(recalls) > 1 and std_recall and std_recall > 0 \
            and chance_recall is not None:
        recall_t = ((mean_recall - chance_recall)
                    / (std_recall / math.sqrt(len(recalls))))
    else:
        recall_t = None
    skip_reason = next((r['gold_recall_skipped_reason'] for r in per_seed
                        if r.get('gold_recall_skipped_reason')), None)

    chance = 1.0 / VOCAB
    if len(accs) > 1 and std_acc and std_acc > 0:
        t_vs_chance = (mean_acc - chance) / (std_acc / math.sqrt(len(accs)))
    else:
        t_vs_chance = None

    return {
        'variant': variant['name'],
        'question': variant['question'],
        'rope': bool(variant['rope']),
        'topk': int(variant['topk']),
        'use_ste': bool(variant['use_ste']),
        'n_kv': ok[0]['n_kv'] if ok else per_seed[0].get('n_kv'),
        'layout': ok[0]['layout'] if ok else None,
        'n_seeds_ok': len(ok),
        'n_seeds_total': len(per_seed),
        'seeds': [r['seed'] for r in per_seed],
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'chance_acc': chance,
        't_stat_vs_chance': t_vs_chance,
        'mean_gold_recall': mean_recall,
        'std_gold_recall': std_recall,
        'ci95_gold_recall': ci_recall,
        'n_gold_recall_seeds': len(recalls),
        'chance_gold_recall': chance_recall,
        'gold_recall_t_vs_chance': recall_t,
        'gold_recall_skipped_reason': skip_reason,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok[0]['n_params'] if ok else None,
        'n_layers': ok[0]['n_layers'] if ok else None,
        'mean_train_time_s': (sum(r.get('train_time_s', 0.0) for r in ok)
                              / len(ok)) if ok else None,
    }


def main():
    info = configure_torch_for_device()
    device = info.device
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info('=' * 70)
    logger.info('Experiment 8: CSA Mechanism Ablation '
                '(RoPE / top-k selection / indexer STE)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')

    n_seeds = parse_int_env('MECH_SEEDS', 5, min_value=1, logger=logger)
    steps = parse_int_env('MECH_STEPS', 200, min_value=1, logger=logger)
    train_batch = parse_int_env('MECH_TRAIN_BATCH', 16, min_value=1,
                                logger=logger)
    n_kv_list = _parse_nkv_list('MECH_NKV', '1')
    for n_kv in n_kv_list:
        if 2 * n_kv > VOCAB:
            raise ValueError(
                f'MECH_NKV includes n_kv={n_kv} but 2*n_kv exceeds '
                f'VOCAB={VOCAB}.')
        if 2 * n_kv >= SEQ_LEN:
            raise ValueError(
                f'MECH_NKV includes n_kv={n_kv} but 2*n_kv must be '
                f'< SEQ_LEN={SEQ_LEN}.')

    requested = [v.strip() for v in
                 os.environ.get('MECH_VARIANTS', '').split(',') if v.strip()]
    if requested:
        unknown = [name for name in requested if name not in VARIANT_NAMES]
        if unknown:
            raise ValueError(
                f'MECH_VARIANTS contains unknown names {unknown}; '
                f'valid names: {list(VARIANT_NAMES)}')
        variants = [v for v in VARIANTS if v['name'] in requested]
    else:
        variants = list(VARIANTS)
    has_baseline = any(v['name'] == 'baseline' for v in variants)
    logger.info(f'  n_seeds={n_seeds}, steps={steps}, n_kv={n_kv_list}, '
                f'train_batch={train_batch}')
    logger.info(f'  variants: {[v["name"] for v in variants]} '
                f'(baseline present: {has_baseline})')

    n_tests = max(len(variants) - 1, 0)
    alpha_corrected = 0.05 / n_tests if n_tests else None

    seeds = [42 + i for i in range(n_seeds)]
    all_rows = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}')
        logger.info(f'{"=" * 70}')
        baseline_acc = {}
        for variant in variants:
            logger.info(f'\n-- variant {variant["name"]} '
                        f'(rope={variant["rope"]}, topk={variant["topk"]}, '
                        f'ste={variant["use_ste"]}, n_seeds={n_seeds}) --')
            per_seed = []
            for seed in seeds:
                t0 = time.time()
                try:
                    record = train_variant(
                        variant, seed=seed, steps=steps, device=device,
                        n_kv=n_kv, train_batch=train_batch)
                    record['train_time_s'] = time.time() - t0
                    logger.info(
                        f'    seed {seed}: acc={record["final_acc"]:.4f}  '
                        f'loss={record["final_loss"]:.4f}  '
                        f'recall={_fmt_recall(record["gold_recall"])}  '
                        f'fwd={record["fwd_ms"]:.2f}ms  '
                        f'time={record["train_time_s"]:.1f}s')
                    per_seed.append(record)
                except Exception as e:
                    logger.warning(f'    seed {seed} FAILED: {e}')
                    per_seed.append({
                        'variant': variant['name'], 'n_kv': n_kv,
                        'seed': seed, 'steps': steps,
                        'train_batch': train_batch,
                        'rope': bool(variant['rope']),
                        'topk': int(variant['topk']),
                        'use_ste': bool(variant['use_ste']),
                        'final_acc': None, 'final_loss': None,
                        'fwd_ms': None, 'n_params': None, 'n_layers': None,
                        'loss_curve': [], 'gold_recall': None,
                        'error': str(e),
                        'train_time_s': time.time() - t0,
                    })
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            row = summarize_variant(per_seed, variant)
            if variant['name'] == 'baseline':
                baseline_acc = {r['seed']: r['final_acc'] for r in per_seed
                                if 'error' not in r
                                and r.get('final_acc') is not None
                                and math.isfinite(r['final_acc'])}
            all_rows.append(row)
            _ci = row['ci95_acc']
            _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
            logger.info(
                f"  -> mean_acc={_fmt_num(row['mean_acc'])} +/- {_ci_str}  "
                f"recall={_fmt_recall(row['mean_gold_recall'])} "
                f"(chance={_fmt_recall(row['chance_gold_recall'])})  "
                f"fwd={_fmt_num(row['mean_fwd_ms'])}ms")

        # Pairwise (paired-by-seed) comparisons against the baseline.
        for row in all_rows:
            if row['n_kv'] != n_kv:
                continue
            row['paired_vs_baseline_mean_acc_diff'] = None
            row['paired_vs_baseline_ci95'] = None
            row['paired_vs_baseline_t_stat'] = None
            row['paired_vs_baseline_n_pairs'] = 0
            row['paired_vs_baseline_t_crit_bonferroni'] = None
            row['paired_significant_bonferroni'] = False
            row['gold_recall_t_crit_bonferroni'] = None
            row['gold_recall_significant_bonferroni'] = False
            if row['variant'] == 'baseline':
                continue
            diffs = []
            for r in row['per_seed']:
                if 'error' in r or r.get('final_acc') is None:
                    continue
                base = baseline_acc.get(r['seed'])
                if base is None or not math.isfinite(base):
                    continue
                diffs.append(r['final_acc'] - base)
            stats = paired_stats(diffs)
            crit = None
            if stats['t_stat'] is not None and alpha_corrected is not None:
                # Two-sided paired test: halve the corrected alpha so the
                # one-sided quantile returned by ``_bonferroni_crit_q``
                # corresponds to a two-sided |t| threshold.
                crit = _bonferroni_crit_q(stats['n_pairs'],
                                          alpha=alpha_corrected / 2.0)
            row['paired_vs_baseline_mean_acc_diff'] = stats['mean']
            row['paired_vs_baseline_ci95'] = stats['ci95']
            row['paired_vs_baseline_t_stat'] = stats['t_stat']
            row['paired_vs_baseline_n_pairs'] = stats['n_pairs']
            row['paired_vs_baseline_t_crit_bonferroni'] = crit
            row['paired_significant_bonferroni'] = (
                crit is not None and stats['t_stat'] is not None
                and abs(stats['t_stat']) > crit)
            recall_t = row['gold_recall_t_vs_chance']
            recall_n = row['n_gold_recall_seeds']
            if recall_t is not None and alpha_corrected is not None:
                # One-sided (recall > chance) with the same Bonferroni
                # family as the paired accuracy tests.
                recall_crit = _bonferroni_crit_q(recall_n,
                                                 alpha=alpha_corrected)
                row['gold_recall_t_crit_bonferroni'] = recall_crit
                row['gold_recall_significant_bonferroni'] = (
                    recall_crit is not None and recall_t > recall_crit)

    ok_rows = [r for r in all_rows if r.get('mean_acc') is not None]
    min_seeds_ok = min((r['n_seeds_ok'] for r in all_rows), default=0)
    n_sig_paired = sum(1 for r in all_rows
                       if r.get('paired_significant_bonferroni'))
    n_sig_recall = sum(1 for r in all_rows
                       if r.get('gold_recall_significant_bonferroni'))
    n_compared = sum(1 for r in all_rows if r.get('variant') != 'baseline')
    conclusions_valid = bool(
        n_seeds >= 5 and min_seeds_ok >= 5 and len(ok_rows) == len(all_rows)
        and (n_sig_paired > 0 or n_sig_recall > 0))
    for row in all_rows:
        row['conclusions_valid'] = conclusions_valid
        row['n_seeds_requested'] = n_seeds

    _print_table(all_rows)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary:')
    logger.info(f'  seeds requested: {n_seeds} (min survived per variant: '
                f'{min_seeds_ok})')
    logger.info(f'  variants with significant paired acc diff: '
                f'{n_sig_paired}/{n_compared} '
                f'(Bonferroni alpha='
                f'{alpha_corrected if alpha_corrected is not None else "n/a"})')
    logger.info(f'  variants with recall above chance (Bonferroni): '
                f'{n_sig_recall}/{len(all_rows)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: Exp 8 results do NOT support strong mechanistic\n'
            '  conclusions (seed count <5, failed seeds, or no Bonferroni-\n'
            '  significant effect). Treat the ranking as exploratory.')
    logger.info('=' * 70)

    metadata = {
        'experiment': 'exp8_mechanism_ablation',
        'design': (
            'Paired-by-seed MQAR ablation of three CSA mechanisms inside '
            'the frozen 3:1:1 hybrid layout: partial RoPE (CSA+HCA), top-k '
            'block-selection width, and indexer STE learning. All variants '
            'share depth=5 and parameter count; only the mechanism switches '
            'change.'),
        'variants': [
            {'name': v['name'], 'rope': bool(v['rope']),
             'topk': int(v['topk']), 'use_ste': bool(v['use_ste']),
             'question': v['question']}
            for v in variants],
        'gold_recall_definition': (
            "fraction of sequences where the compressed block containing "
            "the cued key/value pair is among the top-k blocks the CSA "
            "lightning indexer selects for the cue token (last position); "
            "recomputed from a forward pre-hook on the single CSA layer "
            "with the same normalization / causal mask as naive_csa. "
            "chance_gold_recall = min(k, n_available_blocks) / "
            "n_available_blocks."),
        'paired_test': (
            'two-sided paired t over seeds vs baseline; Bonferroni alpha = '
            '0.05/n_compared_variants; undefined statistics are null.'),
        'n_seeds': n_seeds,
        'steps': steps,
        'train_batch': train_batch,
        'n_kv': n_kv_list,
        'seeds': seeds,
        'seq_len': SEQ_LEN,
        'ratio': list(RATIO),
        'csa_m': SMALL_MODEL_SPEC['csa_m'],
        'csa_indexer_normalize_qk': True,
        'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
        'conclusions_valid': conclusions_valid,
        'caveats': [
            'The hybrid contains one CSA layer; the recall diagnostic '
            'measures that layer only. KDA layers can still carry the task '
            'when CSA selection fails, so end-accuracy differences may '
            'understate the mechanism effect.',
            'MQAR training is budget-limited; a null result at this budget '
            'is not evidence that the mechanism is useless at scale.',
            'RoPE is toggled on CSA and HCA together; the no_rope variant '
            'isolates the paired core-attention stack, not CSA alone.',
        ],
        'provenance': capture_provenance(),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, all_rows, kind='exp8_mechanism_ablation',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')

    n_errors = sum(1 for r in all_rows if r.get('n_seeds_ok', 0) == 0)
    n_incomplete = sum(1 for r in all_rows
                       if 0 < r.get('n_seeds_ok', 0) < r.get('n_seeds_total'))
    if n_errors or n_incomplete:
        logger.error(
            f'\n{n_errors}/{len(all_rows)} variants errored and '
            f'{n_incomplete}/{len(all_rows)} have incomplete seed sets; '
            f'returning non-zero.')
        return 1
    return 0


def _fmt_num(value, prec=4):
    return 'n/a' if value is None else f'{value:.{prec}f}'


def _fmt_recall(value):
    return 'n/a' if value is None else f'{value:.3f}'


def _print_table(rows):
    print('\n' + '=' * 108)
    print(f"{'variant':>16} | {'rope':>4} | {'k':>2} | {'STE':>3} | "
          f"{'mean_acc':>8} | {'+/-CI95':>7} | {'recall':>6} | "
          f"{'chance':>6} | {'paired_d':>9} | {'sig':>3} | {'fwd_ms':>7}")
    print('-' * 108)
    for r in rows:
        track = r.get('variant', '?')
        paired = r.get('paired_vs_baseline_mean_acc_diff')
        sig_p = r.get('paired_significant_bonferroni')
        sig_r = r.get('gold_recall_significant_bonferroni')
        sig = 'both' if (sig_p and sig_r) else ('acc' if sig_p
                                                else ('rec' if sig_r else '-'))
        print(f"{track:>16} | {str(r.get('rope')):>4} | {r.get('topk'):>2} | "
              f"{str(r.get('use_ste')):>3} | "
              f"{_fmt_num(r.get('mean_acc')):>8} | "
              f"{('n/a' if r.get('ci95_acc') is None else format(r['ci95_acc'], '.4f')):>7} | "
              f"{_fmt_recall(r.get('mean_gold_recall')):>6} | "
              f"{_fmt_recall(r.get('chance_gold_recall')):>6} | "
              f"{('n/a' if paired is None else format(paired, '+.4f')):>9} | "
              f"{sig:>3} | {_fmt_num(r.get('mean_fwd_ms'), 2):>7}")


if __name__ == '__main__':
    sys.exit(main())
