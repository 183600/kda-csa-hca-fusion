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

The selected variant set is audited as an ablation design and the verdict
is recorded in the ``ablation_design`` metadata: a single-factor
(leave-one-out) claim for a factor requires the baseline plus a variant
that differs from it in EXACTLY that factor; ``no_rope_dense`` is an
interaction cell. A ``MECH_VARIANTS`` subset that drops a one-factor
counterpart is flagged (``status='incomplete'`` + the missing factors),
never silently treated as a complete ablation; the audit also cross-checks
the observed parameter-count / depth parity across variants.

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

``gold_recall`` is thresholded (top-k hit or miss) and hides how well the
gold block is RANKED. The diagnostic therefore also reports threshold-free
metrics over the indexer's full soft distribution. Because every sequence
is its own retrieval query with EXACTLY ONE relevant block, the primary
metrics are macro-averaged over sequences, never pooled across them:

  * ``gold_pr_auc`` -- mean per-query average precision, tie-averaged.
    For a single relevant item this is identically ``gold_mrr``
    (AP = 1/rank); both are reported because they are computed by
    independent code paths.
  * ``gold_roc_auc`` -- mean per-query tie-corrected ROC-AUC
    (probability a gold score beats a random non-gold score of the SAME
    query).
  * ``gold_mrr`` -- mean per-sequence reciprocal rank of the gold block.
  * ``gold_top1_recall`` -- fraction of sequences where the gold block is
    ranked first.

Pooling candidates from different queries into one ranking (the previous
behaviour) silently lets high-scoring queries define the negatives for
low-scoring ones, so ``gold_pr_auc_pooled`` / ``gold_roc_auc_pooled`` are
kept only as labelled secondary diagnostics.

Chance references for one relevant item among ``n = n_available``
candidates are ``chance_gold_pr_auc = chance_gold_mrr = H(n) / n``
(``expected_average_precision(1, n - 1)``) and
``chance_gold_roc_auc = 0.5``. ``H(n)/n`` is the exact expected AP of a
uniformly random ranking; the prevalence ``1/n`` is the area under the
smoothed chance PR curve and is NOT the expectation of finite-sample AP
(it understates chance for rare positives).

Budget / configuration (environment variables)
----------------------------------------------
* ``MECH_SEEDS``       (default 5)   — seeds per variant.
* ``MECH_STEPS``       (default 200) — training steps per run (matches the
  Exp 4 MQAR default; the hybrid is under-trained at 100 steps).
* ``MECH_TRAIN_BATCH`` (default 16)  — training batch size.
* ``MECH_NKV``         (default '1') — comma-separated MQAR n_kv values.
* ``MECH_VARIANTS``    (default '')  — optional comma-separated subset of
  variant names (must include ``baseline`` for the paired comparisons).
* ``MECH_ERROR_BATCHES`` (default 2)  — held-out bad-case batches per seed
  (0 disables the pass). Every seed also records a held-out validation
  curve, labelled overfitting / underfitting / healthy / saturated by
  ``results_analysis.diagnose_training_curves``.
* ``MECH_ERROR_BATCH``   (default 64) — samples per bad-case batch.
* ``MECH_ERROR_EXAMPLES`` (default 2) — decoded examples kept per category.

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
    append_run_manifest,
    capture_provenance,
    configure_torch_for_device,
    make_run_manifest_entry,
    make_seeded_generator,
    parse_int_env,
    set_global_seed,
)
from ops_csa import _causal_block_mask, csa_compress_kv, csa_lightning_indexer
from ops_fused import CSAHybridLayer, HybridKCHAttention
from results_analysis import (binary_ranking_metrics,
                              expected_average_precision)
from results_schema import write as write_schema
from run_ablation import (
    _aggregate_curve_diagnosis,
    _attach_curve_diagnostics,
    _compact_overfit_report,
    _eval_model,
    _make_cfg,
    _run_ablation_overfit_gate,
)
from train_diagnostics import (
    DEFAULT_COST_KEYS, DEFAULT_MAX_COST_RATIO, DEFAULT_MIN_EFFECT,
    ERROR_ANALYSIS_SEED_OFFSET, VAL_MONITOR_SEED_OFFSET, ValPoint,
    ValidationCurve, accuracy_cost_tradeoff, log_pooled_errors, log_tradeoff,
    pool_error_analyses, raise_non_finite, triage_gradient,
)
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
#
# Each variant carries a PRE-REGISTERED directional prediction and an
# explicit falsification criterion (experiment-design requirement: a
# mechanism claim must state in advance what would refute it, not just ask
# an open question). ``predicted_acc_sign`` fixes the expected sign of the
# paired accuracy difference vs the baseline for the automated
# ``prediction_status`` verdict ('negative' = variant expected WORSE than
# baseline; 'nonpositive' = expected not-better, a null-style claim that
# only a significant POSITIVE difference would falsify; None for the
# reference variant). The falsifiers are written so that a Bonferroni-
# significant paired contrast in the opposite direction contradicts the
# prediction; non-significance is always reported as 'not_supported'
# (inconclusive, not evidence of equivalence).
VARIANTS: tuple[dict, ...] = (
    {'name': 'baseline', 'rope': True, 'topk': SMALL_MODEL_SPEC['csa_topk'],
     'use_ste': True,
     'question': 'reference: RoPE on, sparse top-k, learned indexer',
     'prediction': ('Reference variant (Exp 5 default 3:1:1): anchors every '
                    'paired contrast. No directional prediction of its own.'),
     'falsified_if': ('not applicable: the reference cannot falsify itself; '
                      'a broken baseline shows up as a failed seed or a '
                      'failed overfit gate, not as a contrast.'),
     'predicted_acc_sign': None},
    {'name': 'no_rope', 'rope': False, 'topk': SMALL_MODEL_SPEC['csa_topk'],
     'use_ste': True,
     'question': 'does partial RoPE (CSA+HCA) help MQAR?',
     'prediction': ('partial RoPE encodes relative order, which MQAR '
                    'associative recall exploits: removing it degrades '
                    'end-task accuracy (mean_acc(no_rope) < '
                    'mean_acc(baseline)) and/or the CSA gold-block recall.'),
     'falsified_if': ('no_rope is significantly BETTER than baseline on the '
                      'paired Bonferroni-corrected contrast; positional '
                      'rotation would then hurt this task rather than help.'),
     'predicted_acc_sign': 'negative'},
    {'name': 'topk1', 'rope': True, 'topk': 1, 'use_ste': True,
     'question': 'does extreme selection sparsity (k=1) hurt?',
     'prediction': ('k=1 removes the indexer\'s margin for selection error '
                    '(no hedge when the gold block is ranked 2nd): accuracy '
                    'and gold_recall fall below baseline, recall near the '
                    'chance level min(1, n_available)/n_available.'),
     'falsified_if': ('topk1 significantly BEATS baseline on the paired '
                      'Bonferroni-corrected contrast; extreme sparsity '
                      'would then be an advantage, not a cost.'),
     'predicted_acc_sign': 'negative'},
    {'name': 'dense', 'rope': True, 'topk': _N_BLOCKS, 'use_ste': True,
     'question': 'is sparse selection better than dense compressed attention?',
     'prediction': ('dense compressed attention sees every block but MQAR '
                    'needs exactly one: no significant end-task gain over '
                    'the sparse baseline, with distractor dilution a '
                    'plausible cause of a drop.'),
     'falsified_if': ('dense is significantly BETTER than baseline on the '
                      'paired Bonferroni-corrected contrast; selection '
                      'would then be a bottleneck rather than a feature. '
                      'A non-significant difference is INCONCLUSIVE at this '
                      'seed budget, not equivalence.'),
     'predicted_acc_sign': 'nonpositive'},
    {'name': 'frozen_indexer', 'rope': True,
     'topk': SMALL_MODEL_SPEC['csa_topk'], 'use_ste': False,
     'question': 'does LEARNING the top-k selection (STE) matter?',
     'prediction': ('STE training improves selection quality: baseline > '
                    'frozen_indexer on accuracy and/or gold ranking '
                    '(gold_pr_auc), with the frozen indexer\'s gold_recall '
                    'not above chance.'),
     'falsified_if': ('frozen_indexer is not significantly worse than '
                      'baseline AND its gold_recall significantly exceeds '
                      'chance; indexer learning would then not be required '
                      'for selection quality.'),
     'predicted_acc_sign': 'negative'},
    {'name': 'no_rope_dense', 'rope': False, 'topk': _N_BLOCKS,
     'use_ste': True,
     'question': 'interaction: dense attention without positional rotation',
     'prediction': ('interaction of the two ablations: with no relative-'
                    'order signal the dense indexer cannot exploit block '
                    'geometry, so the variant does not beat baseline '
                    '(recall at or below the no_rope level).'),
     'falsified_if': ('no_rope_dense significantly BEATS baseline on the '
                      'paired Bonferroni-corrected contrast; the two '
                      'ablations would then be super-additive (each alone '
                      'hurts but together they help), contradicting both '
                      'single-factor predictions.'),
     'predicted_acc_sign': 'nonpositive'},
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


def _build_variant_model(variant: dict, d_model, device):
    """Construct (model, head, embed) exactly as ``train_variant`` does.

    Factored out so the overfit gate builds each mechanism variant through
    the same construction path (config, STE switches, init distribution)
    as the sweep it guards.
    """
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = make_variant_cfg(variant, d_model)
    model = HybridKCHAttention(cfg, total_layers=sum(RATIO)).to(device)
    n_csa = set_indexer_ste(model, variant['use_ste'])
    if n_csa != 1:
        raise RuntimeError(
            f"variant {variant['name']}: expected 1 CSA layer, found {n_csa}")
    return model, head, embed


def _compact_gate_record(record):
    """Verdict + headline numbers per gate cell (curves stay in provenance)."""
    return {
        'enabled': record['enabled'],
        'strict': record.get('strict'),
        'steps': record.get('steps'),
        'seed': record.get('seed'),
        'train_batch': record.get('train_batch'),
        'reports': [_compact_overfit_report(r)
                    for r in record.get('reports', [])],
    }


def indexer_topk_from_layer(layer: CSAHybridLayer, x: torch.Tensor,
                            return_scores: bool = False):
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

    ``return_scores=True`` additionally returns the full differentiable
    block distribution ``soft_weights`` ``[B, T_padded, n_blocks]`` (the
    softmax of the indexer logits over causally available blocks) so the
    caller can compute ranking metrics over ALL blocks, not just the hard
    top-k. Default False preserves the historical single-tensor return.
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
    if return_scores:
        idx, soft = csa_lightning_indexer(
            q_idx, K_IComp, w_idx, cfg.csa_topk,
            scale=None,
            causal_block_mask=cbm,
            return_soft_weights=True,
            normalize_qk=True,
        )
        return idx, soft
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

    ``gold_recall`` (top-k) is a THRESHOLDED metric: it says whether the
    gold block cleared the cut but not HOW WELL the indexer RANKS it. The
    same run also reports threshold-free RANKING metrics over the full
    block distribution (from the differentiable soft weights), because the
    gold block is a single rare positive among ``n_available`` blocks:

      * ``gold_pr_auc`` — mean per-query average precision (tie-averaged),
        the PRIMARY metric. With exactly one relevant block per sequence
        it equals ``gold_mrr``; both are computed independently.
      * ``gold_roc_auc`` — mean per-query tie-corrected ROC-AUC.
      * ``gold_mrr`` — mean per-sequence reciprocal rank of the gold block
        (average-rank tie handling); ``chance_gold_mrr`` = H(n)/n.
      * ``gold_top1_recall`` — fraction of sequences where the gold block
        is ranked first.
      * ``gold_pr_auc_pooled`` / ``gold_roc_auc_pooled`` — the old
        candidate-pooled values, kept only as labelled secondary
        diagnostics (they mix negatives across queries and are not
        query-level retrieval metrics).

    Returns a dict with ``gold_recall`` (``None`` when skipped),
    ``n_samples``, ``gold_topk_effective``, ``gold_n_available_blocks``,
    ``chance_gold_recall``, the ranking metrics above (all ``None`` when
    skipped), and ``skipped_reason``.
    """
    empty = {
        'gold_recall': None, 'n_samples': 0,
        'gold_topk_effective': None, 'gold_n_available_blocks': None,
        'chance_gold_recall': None,
        'gold_pr_auc': None, 'gold_roc_auc': None, 'gold_mrr': None,
        'gold_top1_recall': None,
        'gold_pr_auc_pooled': None, 'gold_roc_auc_pooled': None,
        'chance_gold_pr_auc': None, 'chance_gold_roc_auc': None,
        'chance_gold_mrr': None,
        'gold_n_ranking_queries': 0,
        'skipped_reason': None,
    }
    if m is None:
        m = SMALL_MODEL_SPEC['csa_m']
    csa_layers = [layer for layer in model.layers
                  if isinstance(layer, CSAHybridLayer)]
    if len(csa_layers) != 1:
        return {
            **empty,
            'skipped_reason': (f'expected exactly 1 CSA layer in the '
                               f'{RATIO} layout, found {len(csa_layers)}'),
        }
    layer = csa_layers[0]
    captured: dict = {}

    def _capture(_module, args):
        captured['x'] = args[0].detach()

    # Causally available blocks at the cue position: fixed by the geometry,
    # so compute once BEFORE the loop (the score slice needs it too).
    n_blocks = -(-seq_len // m)
    n_cue = seq_len - 1
    n_available = min(n_cue // m, n_blocks)

    handle = layer.register_forward_pre_hook(_capture)
    gen = make_seeded_generator(seed + 6_000_000, device=device)
    was_training = model.training
    hits = 0
    top1_hits = 0
    rr_sum = 0.0
    roc_sum = 0.0
    n_roc_queries = 0
    total = 0
    score_chunks, label_chunks = [], []
    skipped_reason = None
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
                indices, soft = indexer_topk_from_layer(
                    layer, x_norm, return_scores=True)
                selected = indices[:, cue_pos, :]
                hits += int((selected == key_block[:, None])
                            .any(dim=1).sum().item())
                total += int(key_block.numel())

                # Ranking metrics at the cue position over causally
                # available blocks only (later blocks are masked to
                # probability 0 and would trivially inflate ROC-AUC). The
                # n_available==0 guard is for degenerate geometries only
                # (seq_len <= block size); it cannot trigger for the real
                # MQAR specs.
                if n_available <= 0:
                    continue
                scores_cue = soft[:, cue_pos, :n_available]
                labels_cue = (
                    torch.arange(n_available, device=scores_cue.device)
                    [None, :] == key_block[:, None])
                gold_scores = scores_cue.gather(
                    1, key_block[:, None]).squeeze(1)
                greater = (scores_cue > gold_scores[:, None]).sum(dim=1)
                equal = (scores_cue == gold_scores[:, None]).sum(dim=1)
                # Average-rank tie handling (matches binary_ranking_metrics'
                # ROC-AUC convention).
                avg_rank = (1.0 + greater.to(scores_cue.dtype)
                            + (equal.to(scores_cue.dtype) - 1.0) / 2.0)
                rr_sum += float((1.0 / avg_rank).sum().item())
                # Query-level ROC-AUC for one positive among n_available
                # candidates: (# negatives below + 0.5 * ties) / n_neg.
                # With average-rank tie handling this is exactly
                # (n_available - avg_rank) / (n_available - 1). Queries
                # with n_available == 1 have no negative and are skipped
                # (ROC-AUC undefined), matching binary_ranking_metrics.
                if n_available > 1:
                    roc_sum += float(
                        ((n_available - avg_rank)
                         / (n_available - 1)).sum().item())
                    n_roc_queries += int(avg_rank.numel())
                top1_hits += int((scores_cue.argmax(dim=1)
                                  == key_block).sum().item())
                score_chunks.append(scores_cue.reshape(-1))
                label_chunks.append(labels_cue.reshape(-1))
    finally:
        handle.remove()
        model.train(was_training)

    if skipped_reason is not None:
        return {**empty, 'skipped_reason': skipped_reason}
    topk_effective = min(int(layer.cfg.csa_topk), n_available)
    pooled = (binary_ranking_metrics(torch.cat(score_chunks),
                                     torch.cat(label_chunks))
              if score_chunks else None)
    # One relevant block per query: macro AP is the mean reciprocal
    # (tie-averaged) rank, and the chance AP equals the chance MRR
    # H(n)/n. Both are computed from the same per-query average ranks so
    # they can never disagree.
    chance_ap = (expected_average_precision(1, n_available - 1)
                 if n_available else None)
    return {
        'gold_recall': hits / total if total else None,
        'n_samples': total,
        'gold_topk_effective': topk_effective,
        'gold_n_available_blocks': n_available,
        'gold_n_ranking_queries': total,
        'chance_gold_recall': (topk_effective / n_available
                               if n_available else None),
        'gold_pr_auc': (rr_sum / total) if total else None,
        'gold_roc_auc': (roc_sum / n_roc_queries
                         if n_roc_queries else None),
        'gold_mrr': (rr_sum / total) if total else None,
        'gold_top1_recall': (top1_hits / total) if total else None,
        'gold_pr_auc_pooled': (pooled or {}).get('pr_auc'),
        'gold_roc_auc_pooled': (pooled or {}).get('roc_auc'),
        'chance_gold_pr_auc': chance_ap,
        'chance_gold_roc_auc': 0.5 if n_available > 1 else None,
        'chance_gold_mrr': chance_ap,
        'skipped_reason': None,
    }


def train_variant(variant: dict, *, seed, steps, device, seq_len=SEQ_LEN,
                  n_kv=1, d_model=32, eval_batches=4, eval_batch=64,
                  train_batch=None, diagnose=True, diag_batches=2,
                  diag_batch=64, lr=3e-3, val_every=0, val_batches=2,
                  val_batch=32, error_batches=0, error_batch=64,
                  max_error_examples=2):
    """Train one variant for one seed and return its per-seed record.

    The training and evaluation protocol (optimizer, losses, RNG stream
    offsets, latency probe) mirrors ``run_ablation.eval_layout`` so the
    accuracy numbers are directly comparable to Exp 5 at the same budget.

    It now also mirrors Exp 5's *diagnostics*: the same held-out
    validation curve, the same staged NaN triage and the same bad-case
    pass. Without them a mechanism variant that scored 0.62 and one that
    scored 0.62 by saturating a 0.61 model were indistinguishable, so
    "removing RoPE did not help" could not be separated from "removing
    RoPE was never trained long enough to show anything".
    """
    if isinstance(device, str):
        device = torch.device(device)
    # Seed Python/NumPy/torch/CUDA + deterministic cuDNN/algorithm selection
    # so the paired-variant comparison is reproducible on the same machine.
    set_global_seed(seed, deterministic=True)
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

    # Held-out monitor on the same protocol (and the same stream offsets)
    # as run_ablation.eval_layout; additive, so final_acc is unchanged.
    def _val_eval(_step):
        acc, loss = _eval_model(
            model, head, embed, seq_len, n_kv, device,
            n_batches=int(val_batches or 2), batch=int(val_batch or 32),
            seed=seed, seed_offset=VAL_MONITOR_SEED_OFFSET)
        return ValPoint(loss=loss, acc=acc)

    curve = ValidationCurve(
        _val_eval, steps=steps,
        val_every=int(val_every or 0) or max(1, steps // 4),
        label=f'variant {variant["name"]} n_kv={n_kv} seed={seed}',
        ceiling=1.0)
    curve.record(0)

    losses = []
    train_accs = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device,
            generator=batch_gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                params=params, logits=logits,
                context=f'step={step}, variant={variant["name"]}, '
                        f'seed={seed}')
        opt.zero_grad()
        loss.backward()
        bad_grads = triage_gradient(params)
        clip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=float(loss.detach()),
                params=params,
                grads={f'p{i}': p.grad for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'step={step}, variant={variant["name"]}, '
                        f'seed={seed}')
        opt.step()
        losses.append(loss.item())
        train_accs.append(float((logits.argmax(-1) == target).float().mean()))
        if curve.enabled and (step + 1) % curve.val_every == 0:
            curve.record(step + 1)

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
                'gold_n_ranking_queries': 0,
                'chance_gold_recall': None,
                'gold_pr_auc': None, 'gold_roc_auc': None,
                'gold_mrr': None, 'gold_top1_recall': None,
                'gold_pr_auc_pooled': None, 'gold_roc_auc_pooled': None,
                'chance_gold_pr_auc': None, 'chance_gold_roc_auc': None,
                'chance_gold_mrr': None,
                'skipped_reason': 'diagnostic disabled by caller'}

    curve_block = _attach_curve_diagnostics(
        curve=curve, model=model, head=head, embed=embed,
        train_losses=losses, train_accs=train_accs,
        op_name=f'variant:{variant["name"]}', seq_len=seq_len, n_kv=n_kv,
        device=device, steps=steps, error_batches=error_batches,
        error_batch=error_batch, max_error_examples=max_error_examples,
        seed=seed, label=f'variant {variant["name"]} n_kv={n_kv}')

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
        'acc_curve': train_accs,
        # Additive results-analysis block (held-out curve + verdict +
        # generalization gap + bad cases); see run_ablation.eval_layout.
        **curve_block,
        'gold_recall': diag['gold_recall'],
        'gold_recall_n_samples': diag['n_samples'],
        'gold_topk_effective': diag['gold_topk_effective'],
        'gold_n_available_blocks': diag['gold_n_available_blocks'],
        'chance_gold_recall': diag['chance_gold_recall'],
        # Threshold-free ranking quality of the gold block (see
        # diagnose_gold_recall): macro AP is primary under the 1-of-N
        # extreme-imbalance geometry; pooled values are secondary.
        'gold_pr_auc': diag.get('gold_pr_auc'),
        'gold_roc_auc': diag.get('gold_roc_auc'),
        'gold_mrr': diag.get('gold_mrr'),
        'gold_top1_recall': diag.get('gold_top1_recall'),
        'gold_pr_auc_pooled': diag.get('gold_pr_auc_pooled'),
        'gold_roc_auc_pooled': diag.get('gold_roc_auc_pooled'),
        'chance_gold_pr_auc': diag.get('chance_gold_pr_auc'),
        'chance_gold_roc_auc': diag.get('chance_gold_roc_auc'),
        'chance_gold_mrr': diag.get('chance_gold_mrr'),
        'gold_n_ranking_queries': diag.get('gold_n_ranking_queries'),
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

    # Seed-aggregated ranking quality of the gold block (threshold-free).
    # The primary metrics are macro-averaged per query; chance references
    # are geometry constants (H(n)/n for AP and MRR, 0.5 for ROC-AUC) and
    # identical across seeds of a variant.
    def _agg_optional(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return _mean_ci(vals) + (len(vals),)

    (mean_gold_pr_auc, std_gold_pr_auc,
     ci95_gold_pr_auc, n_gold_ranking) = _agg_optional('gold_pr_auc')
    mean_gold_roc_auc = _agg_optional('gold_roc_auc')[0]
    mean_gold_mrr, std_gold_mrr, ci95_gold_mrr, _ = _agg_optional('gold_mrr')
    mean_gold_top1 = _agg_optional('gold_top1_recall')[0]
    mean_gold_pr_auc_pooled = _agg_optional('gold_pr_auc_pooled')[0]
    mean_gold_roc_auc_pooled = _agg_optional('gold_roc_auc_pooled')[0]
    chance_gold_pr_auc = next(
        (r['chance_gold_pr_auc'] for r in ok
         if r.get('chance_gold_pr_auc') is not None), None)
    chance_gold_roc_auc = next(
        (r['chance_gold_roc_auc'] for r in ok
         if r.get('chance_gold_roc_auc') is not None), None)
    chance_gold_mrr = next(
        (r['chance_gold_mrr'] for r in ok
         if r.get('chance_gold_mrr') is not None), None)

    chance = 1.0 / VOCAB
    if len(accs) > 1 and std_acc and std_acc > 0:
        t_vs_chance = (mean_acc - chance) / (std_acc / math.sqrt(len(accs)))
    else:
        t_vs_chance = None

    # Pooled bad-case rollup for this variant (None when the pass is
    # disabled): which mistake dominates the variant's errors, with the
    # worst seed's decoded examples. Same definition as Exp 4/5.
    error_aggregate = pool_error_analyses(ok)
    log_pooled_errors(error_aggregate, label=f'variant {variant["name"]}')

    return {
        'variant': variant['name'],
        'question': variant['question'],
        # Pre-registered directional prediction + falsification criterion
        # (fixed in code before the numbers were read); prediction_status
        # is filled by main() once the paired contrast is computed.
        'prediction': variant['prediction'],
        'falsified_if': variant['falsified_if'],
        'predicted_acc_sign': variant['predicted_acc_sign'],
        'prediction_status': None,
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
        # Threshold-free gold-block ranking (see diagnose_gold_recall).
        # Macro AP is the primary metric: 1 gold block among
        # n_available candidates is an extreme-imbalance retrieval
        # problem, where ROC-AUC is optimistic and recall@k hides
        # near-misses just below the top-k cut. Pooled values are kept as
        # labelled secondary diagnostics only.
        'mean_gold_pr_auc': mean_gold_pr_auc,
        'std_gold_pr_auc': std_gold_pr_auc,
        'ci95_gold_pr_auc': ci95_gold_pr_auc,
        'mean_gold_roc_auc': mean_gold_roc_auc,
        'mean_gold_mrr': mean_gold_mrr,
        'std_gold_mrr': std_gold_mrr,
        'ci95_gold_mrr': ci95_gold_mrr,
        'mean_gold_top1_recall': mean_gold_top1,
        'mean_gold_pr_auc_pooled': mean_gold_pr_auc_pooled,
        'mean_gold_roc_auc_pooled': mean_gold_roc_auc_pooled,
        'n_gold_ranking_seeds': n_gold_ranking,
        'chance_gold_pr_auc': chance_gold_pr_auc,
        'chance_gold_roc_auc': chance_gold_roc_auc,
        'chance_gold_mrr': chance_gold_mrr,
        'gold_recall_skipped_reason': skip_reason,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok[0]['n_params'] if ok else None,
        'n_layers': ok[0]['n_layers'] if ok else None,
        'mean_train_time_s': (sum(r.get('train_time_s', 0.0) for r in ok)
                              / len(ok)) if ok else None,
        # Sweep-level view of the held-out curves (same definition as
        # run_ablation.eval_layout_multi_seed), so a mechanism row carries
        # the same curve vocabulary as a layout row.
        **_aggregate_curve_diagnosis(ok),
        # Pooled bad-case rollup for this variant (None when disabled).
        'error_analysis': error_aggregate,
    }


def _variant_factor_diffs(variant, baseline):
    """Names of the mechanism factors in which ``variant`` differs.

    Factor order is fixed (rope, topk, ste) so the audit output is
    deterministic. A variant differing in exactly one factor is a
    single-factor (leave-one-out) cell; two or more is an interaction.
    """
    diffs = []
    if bool(variant['rope']) != bool(baseline['rope']):
        diffs.append('rope')
    if int(variant['topk']) != int(baseline['topk']):
        diffs.append('topk')
    if bool(variant['use_ste']) != bool(baseline['use_ste']):
        diffs.append('ste')
    return diffs


def ablation_design_audit(variants, rows=None, requested=None,
                          recipe=None):
    """Audit the selected variant set as a controlled ablation design.

    Exp 8 toggles three CSA mechanism factors — partial RoPE, the top-k
    sparse selection width, and indexer STE learning. A single-factor
    (leave-one-out) claim for a factor requires the baseline PLUS a variant
    that differs from the baseline in EXACTLY that factor; a variant that
    changes two or more factors is an interaction cell and cannot support a
    single-factor claim on its own. A ``MECH_VARIANTS`` subset can silently
    drop the one-factor counterpart of a factor while still producing a
    paired-vs-baseline table, so the coverage is audited and recorded.

    When ``rows`` (the summarized result rows) are given, the audit also
    cross-checks the controlled-variable claim empirically: the observed
    parameter counts, depths AND training recipe (steps, batch, lr) must
    match across variants (the mechanism switches must not have changed
    model size or training budget), and every variant must share the same
    successfully evaluated seed set. ``recipe`` may pass the expected
    recipe explicitly; otherwise it is inferred from the rows where
    available.

    Returns a JSON-safe dict with ``status`` one of
    ``'leave_one_out_complete'`` / ``'incomplete'`` / ``'no_baseline'``.
    ``controlled_variables['controls_status']`` separately reports
    ``'controlled'`` / ``'recipe_mismatch'`` / ``'seed_mismatch'`` /
    ``'partially_observed'``.
    """
    names = [v['name'] for v in variants]
    baseline = next((v for v in variants if v['name'] == 'baseline'), None)
    factor_questions = {
        'rope': 'partial RoPE on the CSA/HCA core-attention stack',
        'topk': 'CSA top-k selection width (sparse vs dense)',
        'ste': 'lightning-indexer STE learning',
    }
    one_factor: dict[str, list[str]] = {f: [] for f in factor_questions}
    interactions: dict[str, list[str]] = {}
    for v in variants:
        if baseline is None or v is baseline:
            continue
        diffs = _variant_factor_diffs(v, baseline)
        if len(diffs) == 1:
            one_factor[diffs[0]].append(v['name'])
        elif diffs:
            interactions[v['name']] = diffs

    def _level(v, factor):
        if factor == 'topk':
            return int(v['topk'])
        if factor == 'ste':
            return bool(v['use_ste'])
        return bool(v['rope'])

    factors = {
        f: {
            'question': factor_questions[f],
            'baseline_level': (None if baseline is None
                               else _level(baseline, f)),
            'levels_seen': sorted({_level(v, f) for v in variants}),
            'one_factor_variants': sorted(set(one_factor[f])),
            'covered': bool(one_factor[f]),
        }
        for f in factor_questions
    }
    missing = [f for f in factor_questions if not one_factor[f]]
    if baseline is None:
        status = 'no_baseline'
    elif not missing:
        status = 'leave_one_out_complete'
    else:
        status = 'incomplete'

    audit = {
        'status': status,
        'selected_variants': names,
        'selection': ('all pre-registered variants' if requested is None
                      else f'MECH_VARIANTS subset: {sorted(requested)}'),
        'baseline_present': baseline is not None,
        'factors': factors,
        'interaction_variants': interactions,
        'missing_single_factor_variants': missing,
        'note': (
            'A single-factor (leave-one-out) claim for a factor needs the '
            'baseline plus a variant differing from it in exactly that '
            'factor; interaction variants (>= 2 factors changed) support '
            'only interaction claims. Missing single-factor variants mean '
            'the corresponding mechanism claim is NOT supported by this '
            'run, regardless of any paired contrast that still computes.'),
    }
    if rows is not None:
        def _observed(row, field):
            value = row.get(field)
            if value is None:
                for seed_row in row.get('per_seed') or []:
                    value = seed_row.get(field)
                    if value is not None:
                        break
            return value

        recipe_fields = ('steps', 'train_batch', 'lr')
        observed_fields = ('n_params', 'n_layers') + recipe_fields
        n_params: dict = {}
        n_layers: dict = {}
        values: dict[str, dict[str, set]] = {}
        requested_seeds: dict[str, set] = {}
        successful_seeds: dict[str, set] = {}
        for r in rows:
            name = r.get('variant')
            if name is None:
                continue
            values.setdefault(name, {f: set() for f in observed_fields})
            requested_seeds.setdefault(name, set())
            successful_seeds.setdefault(name, set())
            for field in observed_fields:
                value = _observed(r, field)
                if value is not None:
                    values[name][field].add(value)
            for seed in r.get('seeds') or []:
                requested_seeds[name].add(seed)
            for seed_row in r.get('per_seed') or []:
                if ('error' in seed_row
                        or seed_row.get('final_acc') is None):
                    continue
                successful_seeds[name].add(seed_row.get('seed'))

        names_with_rows = [n for n in names if n in values]

        def _parity(field):
            per_variant = [values[n][field] for n in names_with_rows]
            if not per_variant or any(not v for v in per_variant):
                return None
            if any(len(v) != 1 for v in per_variant):
                return False
            return len({next(iter(v)) for v in per_variant}) <= 1

        recipe_parity = {f: _parity(f) for f in recipe_fields}
        expected_recipe = dict(recipe or {})
        recipe_matches_expected = {}
        for field in recipe_fields:
            if field not in expected_recipe:
                continue
            per_variant = [values[n][field] for n in names_with_rows]
            if not per_variant or any(len(v) != 1 for v in per_variant):
                recipe_matches_expected[field] = None
            else:
                recipe_matches_expected[field] = all(
                    next(iter(v)) == expected_recipe[field]
                    for v in per_variant)

        req_sets = [requested_seeds[n] for n in names_with_rows]
        if not req_sets or any(not s for s in req_sets):
            seed_alignment_ok = None
            common_seeds: set = set()
        else:
            common_seeds = set.intersection(*req_sets)
            seed_alignment_ok = (
                len(common_seeds) == len(req_sets[0])
                and all(s == req_sets[0] for s in req_sets)
                and all(successful_seeds[n] == req_sets[0]
                        for n in names_with_rows))

        mismatched_fields = [f for f, ok in recipe_parity.items()
                             if ok is False]
        if (any(v is False for v in recipe_matches_expected.values())
                or mismatched_fields):
            controls_status = 'recipe_mismatch'
        elif seed_alignment_ok is False:
            controls_status = 'seed_mismatch'
        elif (any(v is None for v in recipe_parity.values())
              or seed_alignment_ok is None):
            controls_status = 'partially_observed'
        else:
            controls_status = 'controlled'

        audit['controlled_variables'] = {
            'parameter_count_by_variant': {
                n: next(iter(values[n]['n_params']))
                for n in names_with_rows if values[n]['n_params']},
            'parameter_count_parity': _parity('n_params'),
            'depth_by_variant': {
                n: next(iter(values[n]['n_layers']))
                for n in names_with_rows if values[n]['n_layers']},
            'depth_parity': _parity('n_layers'),
            'recipe_fields': list(recipe_fields),
            'recipe_by_variant': {
                n: {f: sorted(values[n][f], key=repr)
                    for f in recipe_fields}
                for n in names_with_rows},
            'recipe_parity': recipe_parity,
            'expected_recipe': expected_recipe,
            'recipe_matches_expected': recipe_matches_expected,
            'requested_seeds_by_variant': {
                n: sorted(s for s in requested_seeds[n] if s is not None)
                for n in names_with_rows},
            'successful_seeds_by_variant': {
                n: sorted(s for s in successful_seeds[n] if s is not None)
                for n in names_with_rows},
            'common_requested_seeds': sorted(
                s for s in common_seeds if s is not None),
            'seed_alignment_ok': seed_alignment_ok,
            'controls_status': controls_status,
            'note': ('A single-factor mechanism claim also requires the '
                     'variants to share steps/batch/lr and the requested '
                     'seed set; otherwise the contrast mixes the mechanism '
                     'with a budget or sampling difference.'),
        }
    return audit


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
    # Results-analysis controls: the held-out validation curve is always
    # recorded (read-only), and the bad-case pass is ON by default with the
    # Exp 4 budget (2 batches x 64 held-out samples per seed, dedicated RNG
    # stream). Without it, "removing RoPE did not help" could not be
    # separated from "the errors are all association mix-ups either way".
    error_batches = parse_int_env('MECH_ERROR_BATCHES', 2, min_value=0,
                                  logger=logger)
    error_batch = parse_int_env('MECH_ERROR_BATCH', 64, min_value=1,
                                logger=logger)
    error_examples = parse_int_env('MECH_ERROR_EXAMPLES', 2, min_value=0,
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
    logger.info(f'  bad-case: {error_batches}x{error_batch} held-out samples '
                f'per seed (stream +{ERROR_ANALYSIS_SEED_OFFSET})')
    logger.info(f'  variants: {[v["name"] for v in variants]} '
                f'(baseline present: {has_baseline})')

    # Pre-sweep overfit gate: every selected variant must be able to drive
    # one fixed MQAR batch to near-100% train accuracy through its own
    # forward path (RoPE on/off, top-k width, STE on/off) before the paired
    # sweep burns compute on a broken gradient path.
    gate_cells = [
        (f'variant {v["name"]}',
         lambda v=v: _build_variant_model(v, 32, device))
        for v in variants
    ]
    gate_record, proceed = _run_ablation_overfit_gate(
        gate_cells, device, n_kv=n_kv_list[0], env_prefix='MECH')
    if not proceed:
        return 1

    # Shared Bonferroni family: one paired accuracy test vs baseline per
    # non-baseline variant (only when the baseline variant is present) PLUS
    # one paired gold-PR-AUC test vs baseline per non-baseline variant PLUS
    # one one-sided recall-vs-chance test per variant (baseline included).
    # All use alpha_corrected below, so all must be counted; each test is
    # repeated for every n_kv group.
    n_paired_tests = ((len(variants) - 1) * len(n_kv_list)
                      if has_baseline else 0)
    n_pr_auc_tests = n_paired_tests
    n_tests = (n_paired_tests + n_pr_auc_tests
               + len(variants) * len(n_kv_list))
    alpha_corrected = 0.05 / n_tests if n_tests else None

    seeds = [42 + i for i in range(n_seeds)]
    all_rows = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}')
        logger.info(f'{"=" * 70}')
        baseline_acc = {}
        baseline_pr_auc = {}
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
                        n_kv=n_kv, train_batch=train_batch,
                        error_batches=error_batches,
                        error_batch=error_batch,
                        max_error_examples=error_examples)
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
                        'last_train_loss': None, 'mean_last10_loss': None,
                        'loss_curve': [], 'acc_curve': [],
                        # Mirror the success-row diagnostic schema
                        # (held-out curve + bad cases) so
                        # pd.DataFrame(per_seed) has uniform columns.
                        'val_every': None,
                        'val_history': [],
                        'curve_diagnosis': None,
                        'generalization_gap': None,
                        'best_val': None,
                        'val_loss_first': None, 'val_loss_last': None,
                        'val_acc_first': None, 'val_acc_last': None,
                        'val_acc_peak': None,
                        'error_analysis': None,
                        'error_analysis_seed_offset':
                            ERROR_ANALYSIS_SEED_OFFSET,
                        'val_monitor_seed_offset': VAL_MONITOR_SEED_OFFSET,
                        'gold_recall': None,
                        # Mirror the success-row diagnostic schema so
                        # pd.DataFrame(per_seed) has uniform columns.
                        'gold_recall_n_samples': 0,
                        'gold_topk_effective': None,
                        'gold_n_available_blocks': None,
                        'chance_gold_recall': None,
                        'gold_pr_auc': None, 'gold_roc_auc': None,
                        'gold_mrr': None, 'gold_top1_recall': None,
                        'gold_pr_auc_pooled': None,
                        'gold_roc_auc_pooled': None,
                        'chance_gold_pr_auc': None,
                        'chance_gold_roc_auc': None,
                        'chance_gold_mrr': None,
                        'gold_n_ranking_queries': 0,
                        'gold_recall_skipped_reason': None,
                        'layout': None,
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
                baseline_pr_auc = {
                    r['seed']: r['gold_pr_auc'] for r in per_seed
                    if 'error' not in r
                    and r.get('gold_pr_auc') is not None
                    and math.isfinite(r['gold_pr_auc'])}
            all_rows.append(row)
            _ci = row['ci95_acc']
            _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
            logger.info(
                f"  -> mean_acc={_fmt_num(row['mean_acc'])} +/- {_ci_str}  "
                f"recall={_fmt_recall(row['mean_gold_recall'])} "
                f"(chance={_fmt_recall(row['chance_gold_recall'])})  "
                f"pr_auc={_fmt_recall(row['mean_gold_pr_auc'])} "
                f"(chance={_fmt_recall(row['chance_gold_pr_auc'])})  "
                f"mrr={_fmt_recall(row['mean_gold_mrr'])}  "
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
            row['paired_vs_baseline_mean_gold_pr_auc_diff'] = None
            row['paired_vs_baseline_gold_pr_auc_ci95'] = None
            row['paired_vs_baseline_gold_pr_auc_t_stat'] = None
            row['paired_vs_baseline_gold_pr_auc_n_pairs'] = 0
            row['paired_vs_baseline_gold_pr_auc_t_crit_bonferroni'] = None
            row['paired_pr_auc_significant_bonferroni'] = False
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
            # Paired contrast on the PRIMARY ranking metric. Accuracy alone
            # cannot show whether a variant improved the indexer's gold-block
            # ranking; gold_pr_auc is that direct measurement and is paired
            # by seed against the same baseline run.
            pr_auc_diffs = []
            for r in row['per_seed']:
                if 'error' in r or r.get('gold_pr_auc') is None:
                    continue
                base_pr = baseline_pr_auc.get(r['seed'])
                if base_pr is None or not math.isfinite(base_pr):
                    continue
                pr_auc_diffs.append(r['gold_pr_auc'] - base_pr)
            pr_stats = paired_stats(pr_auc_diffs)
            pr_crit = None
            if pr_stats['t_stat'] is not None and alpha_corrected is not None:
                pr_crit = _bonferroni_crit_q(pr_stats['n_pairs'],
                                             alpha=alpha_corrected / 2.0)
            row['paired_vs_baseline_mean_gold_pr_auc_diff'] = pr_stats['mean']
            row['paired_vs_baseline_gold_pr_auc_ci95'] = pr_stats['ci95']
            row['paired_vs_baseline_gold_pr_auc_t_stat'] = pr_stats['t_stat']
            row['paired_vs_baseline_gold_pr_auc_n_pairs'] = pr_stats['n_pairs']
            row['paired_vs_baseline_gold_pr_auc_t_crit_bonferroni'] = pr_crit
            row['paired_pr_auc_significant_bonferroni'] = (
                pr_crit is not None and pr_stats['t_stat'] is not None
                and abs(pr_stats['t_stat']) > pr_crit)
            # Verdict against the PRE-REGISTERED prediction: only a
            # Bonferroni-significant paired contrast can support or
            # contradict a directional claim; anything else is
            # 'not_supported' (inconclusive), never confirmation of the
            # null-style 'nonpositive' predictions.
            sign_pred = row.get('predicted_acc_sign')
            if sign_pred is None or stats['mean'] is None:
                row['prediction_status'] = None
            elif not row['paired_significant_bonferroni']:
                row['prediction_status'] = 'not_supported'
            elif sign_pred == 'negative':
                row['prediction_status'] = (
                    'supported' if stats['mean'] < 0 else 'contradicted')
            elif sign_pred == 'nonpositive':
                row['prediction_status'] = (
                    'contradicted' if stats['mean'] > 0 else 'supported')
            else:
                raise ValueError(
                    f'unknown predicted_acc_sign {sign_pred!r} for variant '
                    f'{row["variant"]!r}')
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

    # Accuracy-vs-cost screen for every non-baseline variant. The paired
    # test answers "is the accuracy difference real?"; it does not answer
    # "is the variant worth adopting?". A significant gain that costs twice
    # the parameters / train time / inference latency (or a sub-minimum
    # effect that no test separates from noise) must not be quoted as an
    # improvement. Significance comes from the row's OWN Bonferroni test.
    for n_kv in n_kv_list:
        base_row = next(
            (r for r in all_rows
             if r['n_kv'] == n_kv and r.get('variant') == 'baseline'), None)
        for row in all_rows:
            if row['n_kv'] != n_kv or row.get('variant') == 'baseline':
                continue
            if base_row is None or row.get('mean_acc') is None \
                    or base_row.get('mean_acc') is None:
                row['cost_tradeoff'] = None
                row['cost_tradeoff_note'] = (
                    'no baseline row / comparable accuracy on one side; '
                    'refusing to call this an improvement.')
                continue
            verdict = accuracy_cost_tradeoff(
                row, base_row,
                significant=(bool(row.get('paired_significant_bonferroni'))
                             if row.get('paired_significant_bonferroni')
                             is not None else None))
            row['cost_tradeoff'] = verdict
            log_tradeoff(verdict,
                         label=f"variant {row['variant']} (n_kv={n_kv})")
            row['cost_tradeoff_note'] = verdict.get('recommendation', '')

    ok_rows = [r for r in all_rows if r.get('mean_acc') is not None]
    min_seeds_ok = min((r['n_seeds_ok'] for r in all_rows), default=0)
    n_sig_paired = sum(1 for r in all_rows
                       if r.get('paired_significant_bonferroni'))
    n_sig_recall = sum(1 for r in all_rows
                       if r.get('gold_recall_significant_bonferroni'))
    n_sig_pr_auc = sum(1 for r in all_rows
                       if r.get('paired_pr_auc_significant_bonferroni'))
    n_compared = sum(1 for r in all_rows if r.get('variant') != 'baseline')
    n_hyp_supported = sum(1 for r in all_rows
                          if r.get('prediction_status') == 'supported')
    n_hyp_contradicted = sum(1 for r in all_rows
                             if r.get('prediction_status') == 'contradicted')
    n_hyp_not_supported = sum(1 for r in all_rows
                              if r.get('prediction_status') == 'not_supported')
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
    logger.info(f'  variants with significant paired gold-PR-AUC diff: '
                f'{n_sig_pr_auc}/{n_compared} (same Bonferroni family)')
    logger.info(f'  variants with recall above chance (Bonferroni): '
                f'{n_sig_recall}/{len(all_rows)}')
    logger.info(f'  pre-registered predictions: {n_hyp_supported} supported '
                f'/ {n_hyp_contradicted} contradicted / '
                f'{n_hyp_not_supported} not_supported (inconclusive at this '
                f'budget)')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: Exp 8 results do NOT support strong mechanistic\n'
            '  conclusions (seed count <5, failed seeds, or no Bonferroni-\n'
            '  significant effect). Treat the ranking as exploratory.')
    logger.info('=' * 70)

    # Ablation-design audit: verify the selected variant set actually
    # supports the single-factor claims it reports (baseline + exactly-one-
    # factor counterparts), and cross-check the controlled-variable claim
    # against the observed parameter counts / depths.
    design_audit = ablation_design_audit(
        variants, rows=all_rows,
        requested=requested if requested else None,
        recipe={'steps': steps, 'train_batch': train_batch, 'lr': 3e-3})
    if design_audit['status'] != 'leave_one_out_complete':
        logger.warning(
            '  WARNING: selected variant set is not a complete leave-one-out '
            'design (%s); missing single-factor variants: %s. The '
            'corresponding mechanism claims are NOT supported by this run.',
            design_audit['status'],
            design_audit['missing_single_factor_variants'])
    cv = design_audit.get('controlled_variables') or {}
    if cv.get('parameter_count_parity') is False:
        logger.warning(
            '  WARNING: observed parameter counts differ across variants: %s',
            cv.get('parameter_count_by_variant'))
    if cv.get('depth_parity') is False:
        logger.warning(
            '  WARNING: observed depths differ across variants: %s',
            cv.get('depth_by_variant'))
    if cv.get('controls_status') in ('recipe_mismatch', 'seed_mismatch'):
        logger.warning(
            '  WARNING: controlled-variable check failed (%s): '
            'recipe_parity=%s seed_alignment_ok=%s',
            cv.get('controls_status'), cv.get('recipe_parity'),
            cv.get('seed_alignment_ok'))

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
             'question': v['question'],
             # Pre-registered before the numbers were read (see the
             # VARIANTS table): directional prediction, falsification
             # criterion, and the automated outcome counts below.
             'prediction': v['prediction'],
             'falsified_if': v['falsified_if'],
             'predicted_acc_sign': v['predicted_acc_sign']}
            for v in variants],
        'ablation_design': design_audit,
        'prediction_status_definition': (
            'Per-row verdict of the pre-registered prediction against the '
            'paired Bonferroni-corrected accuracy contrast vs baseline: '
            'supported = significant AND matching the predicted sign; '
            'contradicted = significant AND opposite to the predicted sign; '
            'not_supported = not significant (INCONCLUSIVE at this seed '
            'budget, never evidence for the null-style nonpositive '
            'predictions); None = reference variant.'),
        'hypothesis_outcomes': {
            'supported': n_hyp_supported,
            'contradicted': n_hyp_contradicted,
            'not_supported': n_hyp_not_supported,
            'note': ('Counts over all reported (variant, n_kv) rows. A '
                     'prediction is only "supported" under a Bonferroni-'
                     'significant paired contrast in the predicted '
                     'direction; the two-sided design can contradict a '
                     'prediction but cannot confirm a null.'),
        },
        'gold_recall_definition': (
            "fraction of sequences where the compressed block containing "
            "the cued key/value pair is among the top-k blocks the CSA "
            "lightning indexer selects for the cue token (last position); "
            "recomputed from a forward pre-hook on the single CSA layer "
            "with the same normalization / causal mask as naive_csa. "
            "chance_gold_recall = min(k, n_available_blocks) / "
            "n_available_blocks."),
        'gold_ranking_definition': (
            "Threshold-free ranking quality of the gold block over the "
            "causally available compressed blocks at the cue position, "
            "from the indexer's full soft block distribution (not only the "
            "hard top-k): gold_pr_auc = mean per-query average precision "
            "(primary metric: one gold block among n_available candidates "
            "is an extreme-imbalance retrieval problem where ROC-AUC is "
            "optimistic; for one relevant block it equals gold_mrr), "
            "gold_roc_auc = mean per-query tie-corrected ROC-AUC, "
            "gold_mrr = mean per-sequence reciprocal rank of the gold "
            "block, gold_top1_recall = fraction ranked first. "
            "gold_pr_auc_pooled / gold_roc_auc_pooled pool candidates "
            "across queries and are secondary diagnostics only. Chance "
            "references: chance_gold_pr_auc = chance_gold_mrr = "
            "H(n_available)/n_available (the exact expected AP of a "
            "uniformly random ranking, NOT the 1/n_available prevalence), "
            "chance_gold_roc_auc = 0.5. recall@k is thresholded and hides "
            "near-misses just below the cut; the ranking metrics do not. "
            "gold_pr_auc is also tested as a paired-by-seed contrast against "
            "the baseline (paired_vs_baseline_mean_gold_pr_auc_diff)."),
        'paired_test': (
            'two-sided paired t over seeds vs baseline for both final '
            'accuracy and gold_pr_auc; one-sided recall t vs chance; all '
            'share one Bonferroni family of n_tests = (2*(n_variants - 1) + '
            'n_variants) * n_kv when the baseline variant is present; '
            'alpha = 0.05/n_tests; undefined statistics are null.'),
        'cost_tradeoff_policy': {
            'min_effect': DEFAULT_MIN_EFFECT,
            'max_cost_ratio': DEFAULT_MAX_COST_RATIO,
            'cost_keys': list(DEFAULT_COST_KEYS),
            'note': (
                'Each non-baseline row carries cost_tradeoff: a variant is '
                'only an adoptable improvement if its corrected paired test '
                'is significant AND the accuracy gain clears min_effect AND '
                'no cost axis exceeds max_cost_ratio. The accuracy '
                'difference and the cost ratios are reported separately so '
                'either can be quoted on its own.'),
        },
        'results_analysis': {
            'validation_curve': (
                'every seed records a held-out (loss, accuracy) curve every '
                'steps//4 on seed + MECH val-monitor stream, diagnosed as '
                'overfitting / underfitting / healthy / saturated by '
                'results_analysis.diagnose_training_curves; row-level '
                'rollups are curve_verdict / curve_verdict_counts.'),
            'bad_case': (
                'every seed classifies its held-out misclassifications into '
                'task-level categories on a dedicated error stream '
                f'(seed + {ERROR_ANALYSIS_SEED_OFFSET}); the row-level '
                'error_analysis block pools the counts and keeps the worst '
                "seed's decoded examples."),
            'seed_offsets': {
                'training': 1_000_000,
                'headline_eval': 5_000_000,
                'val_monitor': VAL_MONITOR_SEED_OFFSET,
                'error_analysis': ERROR_ANALYSIS_SEED_OFFSET,
            },
        },
        'n_seeds': n_seeds,
        'steps': steps,
        'train_batch': train_batch,
        'n_kv': n_kv_list,
        'seeds': seeds,
        'training': {
            'optimizer': 'AdamW',
            'lr': 3e-3,
            'weight_decay_decay_group': 0.01,
            'weight_decay_no_decay_group': 0.0,
            'grad_clip_norm': 1.0,
            'batch_gen_seed_offset': 1_000_000,
            'eval_gen_seed_offset': 5_000_000,
        },
        'seq_len': SEQ_LEN,
        'ratio': list(RATIO),
        'data': {
            'kind': 'synthetic_mqar',
            'vocab': VOCAB,
            'seq_len': SEQ_LEN,
            'n_kv': list(n_kv_list),
            'train_batch': int(train_batch),
            'layout': ('alternating (key, value) pairs, distractor-only '
                       'filler, final-token recall cue'),
            'generator': ('make_mqar_batch with per-stream seeded '
                          'torch.Generator: training=seed+1_000_000, '
                          'headline_eval=seed+5_000_000, '
                          f'val_monitor=seed+{VAL_MONITOR_SEED_OFFSET}, '
                          f'error_analysis=seed+{ERROR_ANALYSIS_SEED_OFFSET}'),
            'split': ('synthetic on-the-fly streams; no corpus file, so '
                      'the seed offsets + task geometry are the data '
                      'version'),
        },
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
    # Overfit-gate verdict: compact form in the metadata next to the rows,
    # full record (per-step curves, batch fingerprint) in provenance.
    metadata['overfit_sanity_check'] = _compact_gate_record(gate_record)
    metadata['provenance']['overfit_sanity_check'] = gate_record

    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, all_rows, kind='exp8_mechanism_ablation',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')

    n_errors = sum(1 for r in all_rows if r.get('n_seeds_ok', 0) == 0)
    n_incomplete = sum(1 for r in all_rows
                       if 0 < r.get('n_seeds_ok', 0) < r.get('n_seeds_total'))
    rc = 1 if (n_errors or n_incomplete) else 0
    append_run_manifest(make_run_manifest_entry(
        experiment='exp8_mechanism_ablation',
        seeds=metadata['seeds'],
        provenance=metadata['provenance'],
        hyperparameters=metadata['training'],
        data=metadata.get('data'),
        artifacts={
            'results': RESULTS_PATH,
            'overfit_gate_results': 'results/exp8_overfit_gate.json',
        },
        results={
            'n_variants': len(all_rows),
            'n_seeds': int(n_seeds),
            'significant_accuracy_variants': int(n_sig_paired),
            'significant_pr_auc_variants': int(n_sig_pr_auc),
            'significant_recall_variants': int(n_sig_recall),
            'errors': int(n_errors),
            'incomplete_seed_sets': int(n_incomplete),
            'return_code': rc,
        },
        status='completed' if not rc else 'partial',
    ), logger=logger)
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
    print('\n' + '=' * 130)
    print(f"{'variant':>16} | {'rope':>4} | {'k':>2} | {'STE':>3} | "
          f"{'mean_acc':>8} | {'+/-CI95':>7} | {'recall':>6} | "
          f"{'chance':>6} | {'pr_auc':>6} | {'d_pr_auc':>8} | "
          f"{'paired_d':>9} | {'sig':>4} | {'pred':>4} | {'fwd_ms':>7}")
    print('-' * 130)
    pred_tags = {'supported': 'ok', 'contradicted': 'X',
                 'not_supported': '~'}
    for r in rows:
        track = r.get('variant', '?')
        paired = r.get('paired_vs_baseline_mean_acc_diff')
        paired_pr = r.get('paired_vs_baseline_mean_gold_pr_auc_diff')
        sig_p = r.get('paired_significant_bonferroni')
        sig_r = r.get('gold_recall_significant_bonferroni')
        sig_pr = r.get('paired_pr_auc_significant_bonferroni')
        sig = '+'.join(tag for tag, on in (('acc', sig_p), ('pr', sig_pr),
                                           ('rec', sig_r)) if on) or '-'
        pred = pred_tags.get(r.get('prediction_status'), '-')
        print(f"{track:>16} | {str(r.get('rope')):>4} | {r.get('topk'):>2} | "
              f"{str(r.get('use_ste')):>3} | "
              f"{_fmt_num(r.get('mean_acc')):>8} | "
              f"{('n/a' if r.get('ci95_acc') is None else format(r['ci95_acc'], '.4f')):>7} | "
              f"{_fmt_recall(r.get('mean_gold_recall')):>6} | "
              f"{_fmt_recall(r.get('chance_gold_recall')):>6} | "
              f"{_fmt_recall(r.get('mean_gold_pr_auc')):>6} | "
              f"{('n/a' if paired_pr is None else format(paired_pr, '+.4f')):>8} | "
              f"{('n/a' if paired is None else format(paired, '+.4f')):>9} | "
              f"{sig:>4} | {pred:>4} | {_fmt_num(r.get('mean_fwd_ms'), 2):>7}")


if __name__ == '__main__':
    sys.exit(main())
