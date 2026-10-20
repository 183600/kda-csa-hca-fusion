"""Experiment 9 — block-size x layer-interleaving associative-recall reach.

Motivation
----------
Exp 4 (``run_quality``) measures single-operator MQAR accuracy; Exp 5
(``run_ablation``) crosses layer layouts at FIXED compression factors; Exp 8
(``run_mechanism_ablation``) probes the CSA indexer's gold-block recall but
only for the single-CSA-layer 3:1:1 layout. Nothing in the repository ties
the three design axes of the DeepSeek-V4 / Kimi-Linear hybrids together:

1. **Block size** — CSA's compression factor ``m`` and HCA's ``m2``
   (``m=4``, ``m'=128`` in DeepSeek-V4 §4.2.1).
2. **Associative recall** — the MQAR capability that separates linearly
   mixed attention from softmax attention (Kimi Linear Fig. 4; the whole
   reason the hybrid exists).
3. **Layer interleaving** — where the CSA/HCA retrieval layers sit in the
   stack (``3:1:1`` Kimi-style, DeepSeek-V4-Flash's SW-then-interleaved,
   ...).

This module makes the interaction *measurable and checkable* in three
layers, without retraining the world:

Structural reachability (analytic, exact)
    A planted (key, value) association at positions ``(p, p+1)`` queried at
    the final position ``cue = T-1`` is recoverable by a layer iff

    * it is in the layer's sliding window: ``cue - p <= win - 1``
      (``[t-win+1, t]`` inclusive, ``ops_csa._sliding_window_scores``), or
    * key and value fall in the SAME compressed block AND that block is
      strictly preceding: ``p // b == (p+1) // b`` and
      ``p // b < cue // b`` (DeepSeek-V4 Eq. 16 / ``_causal_block_mask``).

    A recurrent state layer (KDA / Mamba-2) is capacity-limited: it has no
    *guaranteed* exact-recall path, so it is reported separately and never
    counted towards the structural union. The analytic table lets us state
    what the block size alone buys (coverage: larger blocks merge the pair
    more often) and what it costs (localisation: the cue sees
    ``cue // b`` candidate blocks, so a fixed top-k selects a smaller
    fraction of a finer block partition).

Empirical indexer recall (training-free measurement)
    ``per_layer_indexer_recall`` generalises Exp 8's single-layer
    diagnostic to ANY layout with arbitrarily many CSA layers: it captures
    each CSA sub-layer's post-norm input with a forward pre-hook,
    recomputes its lightning-indexer top-k from the layer's own weights
    (same code path as Exp 8, ``run_mechanism_ablation.indexer_topk_from_layer``),
    and checks the gold block at the cue. Optionally it ALSO pins the
    recomputed selection against the indices the operator actually used
    (``capture_operator=True``), so the diagnostic can never silently
    describe a different selection than the model runs.

Block-size x interleaving sweep
    ``run_recall_report`` trains the hybrid stack on the distance-controlled
    long-range MQAR task (``run_quality.make_long_range_batch``) over a grid
    of (m, m2) budget pairs x layout presets, and reports, per row:

      * end-task accuracy (overall + distance buckets),
      * per-layer and union indexer gold-block recall,
      * the analytic structural summary and the compressed-entry budget
        ``n_csa/m + n_hca/m2`` entries per token.

    The fixed-set protocol is deliberately small (d_model=32, T<=32):
    the point is the *structure* of the recall-reachability trade-off and
    its regression pinning, not a state-of-the-art number.

    Statistical design: every (m, m2, layout) cell trains the SAME seed
    list (default 3 seeds: 42, 43, 44; ``RECALL_SEEDS`` overrides the
    count), so cells are seed-paired and each row reports mean +/- sample
    std and the 95% t-CI of end-task accuracy and indexer union recall.
    Single- or two-seed runs are flagged as exploratory in the report
    metadata: with fewer than 3 seeds, cell-to-cell differences must not
    be read as effects. The pre-registered hypotheses (``HYPOTHESES``) and
    their analytic/empirical outcomes (``design['hypothesis_checks']``,
    plus a per-row ``indexer_union_above_chance`` check on trained rows)
    are fixed before the grid runs, so the coverage/localisation trade-off
    is measured rather than assumed.

Output
------
``results/exp9_recall_reach.json`` (strict JSON) via the CLI::

    python3 recall_reach.py --json results/exp9_recall_reach.json
    python3 recall_reach.py --quick --json /tmp/recall.json
    python3 recall_reach.py --no-train --json /tmp/structural.json
    RECALL_SEEDS=5 python3 recall_reach.py --seeds 42,43,44,45,46

Tests: ``python3 test_recall_reach.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    capture_provenance,
    configure_torch_for_device,
    make_seeded_generator,
    parse_int_env,
    sanitize_for_json,
    set_global_seed,
    write_results_json,
)
from ops_fused import (  # noqa: E402
    CSAHybridLayer,
    HCAHybridLayer,
    HybridConfig,
    HybridKCHAttention,
)
from run_ablation import _run_ablation_overfit_gate  # noqa: E402
from run_quality import (  # noqa: E402
    MQARHead,
    SMALL_MODEL_SPEC,
    _build_param_groups,
    _t_crit_975,
    make_long_range_batch,
    make_mqar_batch,
)
from train_diagnostics import (  # noqa: E402
    ValPoint,
    ValidationCurve,
    raise_non_finite,
    triage_gradient,
)

logger = logging.getLogger(__name__)

VOCAB = 16
SEQ_LEN = 32
D_MODEL = 32
RESULTS_PATH = os.path.join('results', 'exp9_recall_reach.json')

MEMORY_KINDS = ('kda', 'mamba2')
BLOCK_KINDS = ('csa', 'hca')

# Fixed-depth layout presets: every preset has exactly 6 memory + 2 CSA +
# 2 HCA layers (so the compressed-entry budget and the parameter count are
# identical across interleavings — only the POSITION of the retrieval
# layers changes, exactly like Exp 5's 'position' preset).
LAYOUT_PRESETS: dict[str, tuple[str, ...]] = {
    'alternating': ('kda', 'kda', 'kda', 'csa', 'hca') * 2,
    'retrieval_back': ('kda',) * 6 + ('csa', 'csa', 'hca', 'hca'),
    'retrieval_front': ('csa', 'csa', 'hca', 'hca') + ('kda',) * 6,
    'spread': ('kda', 'csa', 'kda', 'hca', 'kda',
               'kda', 'kda', 'csa', 'kda', 'hca'),
}

# Equal-compressed-entry-budget (m, m2) pairs: 1/m + 1/m2 is constant within
# each family. Mirrors Exp 5's ABL_MATRIX=budget preset so the two
# experiments share the same budget grid, but at the recall level: the same
# per-token cache cost, split differently between light and heavy compression.
BUDGET_PAIRS: tuple[tuple[int, int], ...] = (
    (4, 8), (8, 4),          # 1/m + 1/m2 = 0.375
    (4, 4), (3, 6), (6, 3),  # 1/m + 1/m2 = 0.500
)


# ===========================================================================
# 1. Structural reachability (analytic)
# ===========================================================================

@dataclass(frozen=True)
class LayerSpec:
    """Geometry of one layer in the plan (interleaving-aware).

    ``block`` is the compression factor (``m`` for CSA, ``m2`` for HCA) and
    is ``None`` for recurrent memory layers; ``window`` is the layer's
    sliding-window size (0 disables that branch); ``topk`` is the CSA
    selection width (``None`` for HCA / memory).
    """

    layer_index: int
    kind: str
    block: int | None
    window: int
    topk: int | None

    def __post_init__(self):
        if self.kind not in BLOCK_KINDS + MEMORY_KINDS:
            raise ValueError(
                f"LayerSpec.kind={self.kind!r} must be one of "
                f"{BLOCK_KINDS + MEMORY_KINDS}.")
        if self.layer_index < 0:
            raise ValueError(
                f"LayerSpec.layer_index={self.layer_index} must be >= 0.")
        if self.kind in BLOCK_KINDS:
            if not isinstance(self.block, int) or self.block < 1:
                raise ValueError(
                    f"LayerSpec.block={self.block!r} must be a positive int "
                    f"for kind={self.kind!r}.")
        if self.kind == 'csa':
            if not isinstance(self.topk, int) or self.topk < 0:
                raise ValueError(
                    f"LayerSpec.topk={self.topk!r} must be a non-negative "
                    f"int for kind='csa'.")
        if not isinstance(self.window, int) or self.window < 0:
            raise ValueError(
                f"LayerSpec.window={self.window!r} must be a non-negative "
                f"int (0 disables the branch).")


def plan_from_config(cfg: HybridConfig, layout) -> list[LayerSpec]:
    """Build the analytic layer plan from a validated config + layout tuple.

    ``layout`` must already be resolved (no 'memory' tokens); pass
    ``model.layout`` or ``cfg.layout`` after construction. The geometry is
    read from the config, so the plan is exactly what the built stack uses
    (``csa_m`` / ``hca_m2`` / the two window sizes / ``csa_topk``).
    """
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f"plan_from_config: cfg must be a HybridConfig, got "
            f"{type(cfg).__name__}.")
    layout = tuple(layout)
    if len(layout) == 0:
        raise ValueError('plan_from_config: layout must be non-empty.')
    specs: list[LayerSpec] = []
    for i, kind in enumerate(layout):
        if kind in ('csa', 'hca'):
            specs.append(LayerSpec(
                layer_index=i, kind=kind,
                block=cfg.csa_m if kind == 'csa' else cfg.hca_m2,
                window=(cfg.csa_sliding_window if kind == 'csa'
                        else cfg.hca_sliding_window),
                topk=cfg.csa_topk if kind == 'csa' else None,
            ))
        elif kind in MEMORY_KINDS:
            specs.append(LayerSpec(
                layer_index=i, kind=kind, block=None, window=0, topk=None))
        else:
            raise ValueError(
                f"plan_from_config: unresolved layer kind {kind!r}; pass a "
                f"resolved layout (no 'memory' tokens).")
    return specs


def plan_from_model(model) -> list[LayerSpec]:
    """Analytic plan of a built ``HybridKCHAttention`` stack."""
    if not hasattr(model, 'layout') or not hasattr(model, 'cfg'):
        raise TypeError(
            'plan_from_model: model must expose .layout and .cfg '
            '(HybridKCHAttention / HybridAttentionStack).')
    return plan_from_config(model.cfg, model.layout)


def _validate_key_positions(key_positions, cue: int) -> list[int]:
    out: list[int] = []
    for p in key_positions:
        if not isinstance(p, int) or isinstance(p, bool):
            raise ValueError(
                f'key positions must be plain ints, got {p!r} '
                f'({type(p).__name__}).')
        if not 0 <= p <= cue - 1:
            raise ValueError(
                f'key position p={p} must satisfy 0 <= p <= cue-1={cue - 1} '
                f'(the value sits at p+1 and must precede the cue).')
        out.append(p)
    if not out:
        raise ValueError('key_positions must be non-empty.')
    return out


def structural_recall(plan, seq_len: int,
                      key_positions=None) -> dict:
    """Exact per-placement reachability of a (key, value) pair.

    The cue is always the final position ``cue = seq_len - 1`` and the
    value always sits at ``key_pos + 1``. For every placement and layer,
    the result records the two exact-recall paths (block compression and
    sliding window), whether a recurrent-state layer is present, and the
    per-layer details needed for the block-size interpretation:

    * ``block_hit``: key and value in the same compressed block AND that
      block strictly precedes the cue's own block.
    * ``window_hit``: key inside the query's sliding window
      (``[cue-win+1, cue]``).
    * ``exact``: ``block_hit or window_hit``.

    Summaries aggregate placements (``any_exact_fraction``) and report the
    CSA selection pressure: with ``a = cue // m`` visible compressed blocks
    and top-k ``k``, a uniformly-random indexer selects the gold block with
    probability ``min(k, a) / a``.
    """
    plan = list(plan)
    if not plan:
        raise ValueError('structural_recall: plan must be non-empty.')
    for spec in plan:
        if not isinstance(spec, LayerSpec):
            raise TypeError(
                f'structural_recall: plan entries must be LayerSpec, got '
                f'{type(spec).__name__}.')
    if not isinstance(seq_len, int) or seq_len < 3:
        raise ValueError(
            f'structural_recall: seq_len={seq_len!r} must be an int >= 3 '
            f'(key + value + cue).')
    cue = seq_len - 1
    positions = (_validate_key_positions(key_positions, cue)
                 if key_positions is not None
                 else list(range(0, cue - 1)))

    rows: list[dict] = []
    for p in positions:
        dist = cue - p
        layer_rows = []
        for spec in plan:
            row = {
                'layer_index': spec.layer_index,
                'kind': spec.kind,
                'block': spec.block,
                'window': spec.window,
                'topk': spec.topk,
                'block_hit': False,
                'window_hit': False,
                'exact': False,
                'state': spec.kind in MEMORY_KINDS,
            }
            if spec.kind in BLOCK_KINDS:
                b = spec.block
                block_hit = (p // b == (p + 1) // b) and (p // b < cue // b)
                window_hit = spec.window > 0 and dist <= spec.window - 1
                row['block_hit'] = bool(block_hit)
                row['window_hit'] = bool(window_hit)
                row['exact'] = bool(block_hit or window_hit)
            layer_rows.append(row)
        rows.append({
            'key_pos': p,
            'distance': dist,
            'layers': layer_rows,
            'n_exact_paths': sum(1 for r in layer_rows if r['exact']),
            'any_exact': any(r['exact'] for r in layer_rows),
            'any_state': any(r['state'] for r in layer_rows),
        })

    n = len(rows)
    per_layer: list[dict] = []
    for j, spec in enumerate(plan):
        n_exact = sum(1 for r in rows if r['layers'][j]['exact'])
        n_block = sum(1 for r in rows if r['layers'][j]['block_hit'])
        n_window = sum(1 for r in rows if r['layers'][j]['window_hit'])
        entry = {
            'layer_index': spec.layer_index,
            'kind': spec.kind,
            'block': spec.block,
            'window': spec.window,
            'topk': spec.topk,
            'exact_fraction': n_exact / n,
            'block_hit_fraction': n_block / n,
            'window_hit_fraction': n_window / n,
        }
        if spec.kind == 'csa':
            n_avail = cue // spec.block
            p_sel = (min(spec.topk, n_avail) / n_avail) if n_avail > 0 else 0.0
            entry['visible_blocks_at_cue'] = n_avail
            entry['selection_prob_given_visible'] = p_sel
            entry['selection_chance'] = (n_block / n) * p_sel
        per_layer.append(entry)

    by_distance: dict[str, dict] = {}
    for r in rows:
        key = str(r['distance'])
        slot = by_distance.setdefault(key, {'n': 0, 'any_exact': 0})
        slot['n'] += 1
        slot['any_exact'] += int(r['any_exact'])
    for slot in by_distance.values():
        slot['any_exact_fraction'] = slot['any_exact'] / slot['n']

    csa_chances = [e['selection_chance'] for e in per_layer
                   if e['kind'] == 'csa']
    union_random = 1.0
    for c in csa_chances:
        union_random *= (1.0 - c)
    union_random = 1.0 - union_random

    return {
        'seq_len': seq_len,
        'cue_pos': cue,
        'n_placements': n,
        'any_exact_fraction': sum(1 for r in rows if r['any_exact']) / n,
        'any_state_fraction': sum(1 for r in rows if r['any_state']) / n,
        'n_exact_paths_mean': (sum(r['n_exact_paths'] for r in rows) / n),
        'per_layer': per_layer,
        'csa_union_selection_chance_indep': union_random,
        'by_distance': dict(sorted(
            by_distance.items(), key=lambda kv: int(kv[0]))),
        'rows': rows,
    }


def structural_summary(plan, seq_len: int) -> dict:
    """The placement rows stripped out — the JSON-weight-friendly summary."""
    full = structural_recall(plan, seq_len)
    return {k: v for k, v in full.items() if k != 'rows'}


# ===========================================================================
# 2. Empirical per-layer indexer gold-block recall
# ===========================================================================

def per_layer_indexer_recall(model, x_emb: torch.Tensor, key_pos,
                             *, cue_pos: int | None = None,
                             capture_operator: bool = False) -> dict:
    """Gold-block recall of EVERY CSA layer in an arbitrary layout.

    Generalises ``run_mechanism_ablation.diagnose_gold_recall`` from the
    single-CSA 3:1:1 stack to any number of CSA layers. For each CSA
    sub-layer the post-norm input is captured with a forward pre-hook and
    its lightning-indexer top-k is recomputed from the layer's own weights
    via the SAME helper Exp 8 pins against the operator
    (``indexer_topk_from_layer``). A sample is *resolved* when the cued
    (key, value) pair lies in one compressed block AND that block strictly
    precedes the cue's block; ``recall`` is the hit rate among resolved
    samples and ``chance`` the closed-form uniform-selection level.

    With ``capture_operator=True`` the indices the operator actually used
    are recorded (one call per CSA layer, in layout order) and returned as
    ``operator_alignment`` — a built-in positive control that the
    recomputed selection is the one the model ran.

    Args:
        model: a built ``HybridKCHAttention`` / ``HybridAttentionStack``.
        x_emb: embedding input ``[B, T, d]``.
        key_pos: integer positions of the cued keys, ``[B]`` (the value is
            assumed at ``key_pos + 1``, matching MQAR / long-range tasks).
        cue_pos: readout position (default ``T-1``).
        capture_operator: pin recomputed vs operator selections.

    Returns:
        A strict-JSON dict; ``layers`` has one entry per CSA layer in
        layout order, plus the stack-level union recall (a sample is a
        union hit if ANY CSA layer selects its gold block).
    """
    if not isinstance(x_emb, torch.Tensor) or x_emb.dim() != 3:
        raise ValueError(
            f'per_layer_indexer_recall: x_emb must be [B, T, d], got '
            f'{getattr(x_emb, "shape", type(x_emb))}.')
    B, T, _d = x_emb.shape
    if T < 3:
        raise ValueError(
            f'per_layer_indexer_recall: T={T} must be >= 3 (key + value + cue).')
    if cue_pos is None:
        cue_pos = T - 1
    if not isinstance(cue_pos, int) or not 0 <= cue_pos < T:
        raise ValueError(
            f'per_layer_indexer_recall: cue_pos={cue_pos!r} out of range '
            f'[0, T={T}).')
    key_pos_t = torch.as_tensor(key_pos, dtype=torch.long)
    if key_pos_t.dim() != 1 or key_pos_t.shape[0] != B:
        raise ValueError(
            f'per_layer_indexer_recall: key_pos must be a [B={B}] tensor, '
            f'got shape {tuple(key_pos_t.shape)}.')
    if int(key_pos_t.min()) < 0 or int(key_pos_t.max()) > cue_pos - 1:
        raise ValueError(
            f'per_layer_indexer_recall: key_pos must lie in [0, cue_pos-1='
            f'{cue_pos - 1}] so that key and value (p+1) precede the cue; '
            f'got [{int(key_pos_t.min())}, {int(key_pos_t.max())}].')

    csa_global = [i for i, layer in enumerate(model.layers)
                  if isinstance(layer, CSAHybridLayer)]
    if not csa_global:
        return {
            'n_samples': B, 'cue_pos': cue_pos,
            'layers': [], 'union_recall': None, 'union_n_resolved': 0,
            'first_hit_layer_hist': {}, 'operator_alignment': None,
            'skipped_reason': 'layout contains no CSA layer',
        }

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for gi in csa_global:
        def _make_hook(idx):
            def _hook(_module, args):
                captured[idx] = args[0].detach()
            return _hook
        handles.append(model.layers[gi].register_forward_pre_hook(_make_hook(gi)))

    op_calls: list[torch.Tensor] = []
    orig_indexer = None
    import ops_csa as _ops_csa
    if capture_operator:
        orig_indexer = _ops_csa.csa_lightning_indexer

        def _spy(*args, **kwargs):
            out = orig_indexer(*args, **kwargs)
            idx = out[0] if isinstance(out, tuple) else out
            op_calls.append(idx.detach().clone())
            return out

        _ops_csa.csa_lightning_indexer = _spy

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model.reset_state()
            model(x_emb)
    finally:
        model.train(was_training)
        for h in handles:
            h.remove()
        if orig_indexer is not None:
            _ops_csa.csa_lightning_indexer = orig_indexer

    from run_mechanism_ablation import indexer_topk_from_layer

    layer_records: list[dict] = []
    hit_matrix = torch.zeros(len(csa_global), B, dtype=torch.bool)
    resolved_matrix = torch.zeros(len(csa_global), B, dtype=torch.bool)
    alignments: list[bool] = []

    for order, gi in enumerate(csa_global):
        layer = model.layers[gi]
        m = int(layer.cfg.csa_m)
        topk = int(layer.cfg.csa_topk)
        recomputed = indexer_topk_from_layer(layer, captured[gi])
        selected = recomputed[:, cue_pos, :]                     # [B, topk]
        if capture_operator:
            if order < len(op_calls):
                alignments.append(bool(torch.equal(
                    recomputed, op_calls[order])))
            else:
                alignments.append(False)

        key_block = key_pos_t // m
        value_block = (key_pos_t + 1) // m
        same_block = key_block == value_block
        causal = key_block < (cue_pos // m)
        resolved = same_block & causal

        hit = (selected == key_block[:, None]).any(dim=1) & resolved
        hit_matrix[order] = hit
        resolved_matrix[order] = resolved

        n_resolved = int(resolved.sum().item())
        n_avail = cue_pos // m
        if n_avail > 0 and n_resolved:
            chance = min(topk, n_avail) / n_avail
        else:
            chance = None
        layer_records.append({
            'layer_index': gi,
            'csa_order': order,
            'm': m,
            'topk': topk,
            'visible_blocks_at_cue': n_avail,
            'n_resolved': n_resolved,
            'same_block_fraction': float(same_block.float().mean().item()),
            'causal_fraction': float(causal.float().mean().item()),
            'recall': (float(hit.sum().item()) / n_resolved
                       if n_resolved else None),
            'chance': chance,
            'operator_alignment': (alignments[order] if capture_operator
                                   else None),
        })

    union_hit = hit_matrix.any(dim=0) & resolved_matrix.any(dim=0)
    union_resolved = resolved_matrix.any(dim=0)
    n_union_resolved = int(union_resolved.sum().item())
    first_hit_hist: dict[str, int] = {}
    for order in range(len(csa_global)):
        prior = hit_matrix[:order].any(dim=0) if order > 0 else None
        first = hit_matrix[order] if prior is None else (hit_matrix[order]
                                                         & ~prior)
        count = int((first & union_resolved).sum().item())
        if count:
            first_hit_hist[str(layer_records[order]['layer_index'])] = count

    return {
        'n_samples': B,
        'cue_pos': cue_pos,
        'layers': layer_records,
        'union_recall': (float(union_hit.sum().item()) / n_union_resolved
                         if n_union_resolved else None),
        'union_n_resolved': n_union_resolved,
        'first_hit_layer_hist': first_hit_hist,
        'operator_alignment': (all(alignments) if capture_operator else None),
        'skipped_reason': None,
    }


def hca_block_coverage(model, key_pos, *, cue_pos: int) -> dict:
    """Structural (no selection) coverage of every HCA layer.

    HCA attends DENSELY over all causally-visible heavy-compressed entries,
    so its exact-recall path is purely geometric: the pair must share an
    ``m2`` block and that block must strictly precede the cue's own
    (``_causal_block_mask``). HCA has no top-k selection to lose the gold
    entry — but larger ``m2`` means the entry mixes more tokens, so its
    fraction is the *upper bound* on what HCA can localise.
    """
    key_pos_t = torch.as_tensor(key_pos, dtype=torch.long)
    if key_pos_t.dim() != 1 or key_pos_t.numel() == 0:
        raise ValueError('hca_block_coverage: key_pos must be 1-D non-empty.')
    if not isinstance(cue_pos, int) or cue_pos < 0:
        raise ValueError('hca_block_coverage: cue_pos must be a non-negative '
                         'int.')
    n = int(key_pos_t.numel())
    rows = []
    for gi, layer in enumerate(model.layers):
        if not isinstance(layer, HCAHybridLayer):
            continue
        m2 = int(layer.cfg.hca_m2)
        same = key_pos_t // m2 == (key_pos_t + 1) // m2
        causal = key_pos_t // m2 < (cue_pos // m2)
        rows.append({
            'layer_index': gi,
            'm2': m2,
            'same_block_fraction': int(same.sum().item()) / n,
            'resolved_fraction': int((same & causal).sum().item()) / n,
        })
    any_resolved = 0
    if rows:
        union = torch.stack([
            (key_pos_t // r['m2'] == (key_pos_t + 1) // r['m2'])
            & (key_pos_t // r['m2'] < (cue_pos // r['m2']))
            for r in rows]).any(dim=0)
        any_resolved = int(union.sum().item())
    return {'layers': rows, 'any_resolved_fraction': any_resolved / n}


# ===========================================================================
# 3. Block-size x layer-interleaving sweep
# ===========================================================================

def make_grid_cfg(m: int, m2: int, layout, *, d_model: int = D_MODEL,
                  csa_topk: int | None = None) -> HybridConfig:
    """HybridConfig for one (block size, interleaving) cell.

    Uses the frozen ``run_quality.SMALL_MODEL_SPEC`` widths (so every cell
    shares the same sub-layer geometry and parameterisation) and overrides
    only the compression factors and the layout. ``kda_chunk_size=0``
    keeps the recurrent path, matching the short sequences used here.
    """
    spec = SMALL_MODEL_SPEC
    if csa_topk is None:
        csa_topk = spec['csa_topk']
    return HybridConfig(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        kda_chunk_size=0,
        csa_m=int(m), csa_topk=int(csa_topk),
        csa_nh=spec['csa_nh'], csa_c=spec['csa_c'],
        csa_dc=spec['csa_dc'], csa_nIh=spec['csa_nIh'],
        csa_cI=spec['csa_cI'],
        csa_sliding_window=spec['csa_sliding_window'],
        csa_rope_dim=spec['csa_rope_dim'],
        hca_m2=int(m2), hca_nh=spec['hca_nh'], hca_c=spec['hca_c'],
        hca_dc=spec['hca_dc'],
        hca_sliding_window=spec['hca_sliding_window'],
        hca_rope_dim=spec['hca_rope_dim'],
        rope_base=spec['rope_base'],
        layout=tuple(layout),
    )


def _build_recall_model(m: int, m2: int, layout, d_model, device):
    """Construct (model, head, embed) exactly as ``train_recall_model`` does.

    Factored out so the overfit gate builds the probe model through the
    same construction path (config, layout, init distribution) as the grid
    sweep it guards.
    """
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = make_grid_cfg(m, m2, layout, d_model=d_model)
    model = HybridKCHAttention(cfg, total_layers=len(cfg.layout)).to(device)
    return model, head, embed


def _build_recall_val_set(*, embed, seq_len: int, batch: int, vocab: int,
                          n_batches: int, seed: int, device) -> list[dict]:
    """Fixed held-out long-range batches for validation monitoring.

    Built ONCE from a dedicated generator, and stored as input ids (not
    embeddings) so every checkpoint re-embeds them with the current
    embedding weights. No global RNG draw; the same fixed set is used for
    every checkpoint, so the curve is comparable across steps.
    """
    gen = make_seeded_generator(int(seed) + 9_000_000, device=device)
    out = []
    with torch.no_grad():
        for _ in range(max(1, int(n_batches))):
            _x, target, cue_pos, meta = make_long_range_batch(
                int(batch), int(seq_len), 1, int(vocab), embed, device,
                generator=gen, return_meta=True)
            out.append({
                'input_ids': meta['input_ids'].detach().clone(),
                'target': target.detach().clone(),
                'cue_pos': int(cue_pos),
            })
    return out


def _recall_validation_point(model, embed, head,
                             val_set: list[dict]) -> ValPoint:
    """Mean held-out CE + accuracy on the fixed validation set.

    Read-only: no gradient, no optimizer step, and all three modules get
    their previous train/eval mode restored in ``finally``.
    """
    modes = (model.training, embed.training, head.training)
    model.eval()
    embed.eval()
    head.eval()
    loss_sum, correct, total = 0.0, 0, 0
    try:
        with torch.no_grad():
            for batch in val_set:
                model.reset_state()
                logits = head(model(embed(batch['input_ids'])),
                              batch['cue_pos'])
                loss_sum += float(F.cross_entropy(logits, batch['target']))
                correct += int(
                    (logits.argmax(dim=-1) == batch['target']).sum())
                total += int(batch['target'].numel())
    finally:
        model.train(modes[0])
        embed.train(modes[1])
        head.train(modes[2])
    n_batches = max(1, len(val_set))
    return ValPoint(loss=loss_sum / n_batches,
                    acc=(correct / total) if total else 0.0)


def train_recall_model(layout, m: int, m2: int, *, seed: int, steps: int,
                       seq_len: int = SEQ_LEN, d_model: int = D_MODEL,
                       train_batch: int = 16, lr: float = 3e-3,
                       vocab: int = VOCAB, device='cpu',
                       val_every: int = 0, val_batches: int = 1,
                       val_batch: int = 64, return_report: bool = False):
    """Train a hybrid stack on the distance-controlled long-range task.

    One (key, value) pair per sequence at a uniformly random position
    (``make_long_range_batch``), cue at the last position, distractor-only
    filler. Returns ``(model, embed, head, losses)``; the model is left in
    ``eval()`` mode.

    ``val_every > 0`` additionally records a held-out (loss, accuracy)
    curve on a FIXED batch built from a dedicated generator. The monitor is
    read-only and RNG-neutral, so enabling it cannot change the training
    loss curve or the final accuracy. With ``return_report=True`` the
    tuple gains a fifth element: the curve/diagnosis report.
    """
    if isinstance(device, str):
        device = torch.device(device)
    # Seed Python/NumPy/torch/CUDA + deterministic cuDNN/algorithm selection
    # so the reachability grid is reproducible on the same machine.
    set_global_seed(seed, deterministic=True)
    embed = nn.Embedding(vocab, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    cfg = make_grid_cfg(m, m2, layout, d_model=d_model)
    model = HybridKCHAttention(cfg, total_layers=len(cfg.layout)).to(device)
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]
    gen = make_seeded_generator(seed + 1_000_000, device=device)

    model.train()
    head.train()
    embed.train()
    losses: list[float] = []
    val_every = max(0, int(val_every or 0))
    val_curve = None
    if val_every > 0:
        val_set = _build_recall_val_set(
            embed=embed, seq_len=seq_len, batch=val_batch, vocab=vocab,
            n_batches=val_batches, seed=seed, device=device)
        val_curve = ValidationCurve(
            lambda _step: _recall_validation_point(
                model, embed, head, val_set),
            steps=int(steps), val_every=val_every,
            label=f'recall {layout} m={m} m2={m2} seed={seed}',
            ceiling=1.0)
        val_curve.record(0)
    for step in range(int(steps)):
        x_emb, target, cue_pos = make_long_range_batch(
            train_batch, seq_len, 1, vocab, embed, device, generator=gen)
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        context = (f'train_recall_model layout={layout} m={m} m2={m2} '
                   f'seed={seed} step={step}')
        if not torch.isfinite(loss):
            # Staged triage: embedded input -> loss -> logits -> params.
            # Without this, a diverged seed reports only "non-finite loss"
            # and names none of the possible causes.
            raise_non_finite(
                'loss', loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                context=context)
        opt.zero_grad()
        loss.backward()
        bad_grads = triage_gradient(params)
        clip_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=float(loss.detach()),
                inputs={'x_emb': x_emb}, params=params, logits=logits,
                grads={f'grad[{i}]': p.grad
                       for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm), context=context)
        opt.step()
        losses.append(float(loss.item()))
        if val_curve is not None and (step + 1) % val_every == 0:
            val_curve.record(step + 1)
    model.eval()
    head.eval()
    embed.eval()
    report = None
    if val_curve is not None:
        block = val_curve.finalize(losses)
        report = {
            'val_every': int(val_every),
            'val_batches': int(max(1, val_batches)),
            'val_batch': int(val_batch),
            'validation': block,
            'curve_diagnosis': block['curve_diagnosis'],
            'generalization_gap': block['generalization_gap'],
            'val_loss_regret_from_best': block.get(
                'val_loss_regret_from_best'),
            'early_stopping_summary': block.get('early_stopping_summary'),
            'validation_protocol': {
                'fixed_set': True,
                'read_only': True,
                'rng_neutral': True,
                'seed_offset': 9_000_000,
                'n_sequences': int(max(1, val_batches)) * int(val_batch),
            },
        }
        val_curve.log_verdict(block)
    if return_report:
        return model, embed, head, losses, report
    return model, embed, head, losses


def _recall_bucket(distance: int, win: int, mid_edge: int) -> str:
    if 2 <= distance <= win:
        return 'window'
    if win < distance <= mid_edge:
        return 'mid'
    return 'far'


def _recall_error_category(pred: int, target: int, ids, key_pos: int,
                           cue_pos: int) -> str:
    """Name a wrong single-token recall in terms of the copy it made.

    ``cue_repeat`` copies the cue key, ``recency_copy`` copies the token
    just before the cue (the classic recency shortcut), and
    ``distractor_copy`` copies some other filler token. Anything else is
    ``other``. ``correct`` is only returned defensively (callers pass
    wrong rows).
    """
    if pred == target:
        return 'correct'
    if pred == int(ids[cue_pos]):
        return 'cue_repeat'
    if cue_pos > 0 and pred == int(ids[cue_pos - 1]):
        return 'recency_copy'
    if bool((ids == pred).any()):
        return 'distractor_copy'
    return 'other'


def evaluate_recall(model, embed, head, *, seq_len: int = SEQ_LEN,
                    n_batches: int = 4, batch: int = 64, seed: int = 42,
                    vocab: int = VOCAB, device='cpu',
                    max_error_examples: int = 8) -> dict:
    """Accuracy + distance buckets on fresh long-range batches.

    Also returns the *probe batch* (embeddings + key positions + cue) so
    the caller can run the indexer diagnostic on the exact same samples the
    accuracy was measured on, and a bad-case block that names *what a
    wrong prediction copied* (cue key, previous token, filler) plus its
    distance. Error categories are counted over every wrong sample;
    ``examples`` keeps at most ``max_error_examples`` JSON-safe rows.
    """
    if isinstance(device, str):
        device = torch.device(device)
    gen = make_seeded_generator(seed + 5_000_000, device=device)
    correct = 0
    total = 0
    buckets = {'window': [0, 0], 'mid': [0, 0], 'far': [0, 0]}
    win = SMALL_MODEL_SPEC['csa_sliding_window']
    mid_edge = max(win + 1, seq_len // 2)
    probe = None
    error_total = 0
    category_counts: dict[str, int] = {}
    examples: list[dict] = []
    max_error_examples = max(0, int(max_error_examples))
    with torch.no_grad():
        for b in range(n_batches):
            x_emb, target, cue_pos, meta = make_long_range_batch(
                batch, seq_len, 1, vocab, embed, device, generator=gen,
                return_meta=True)
            model.reset_state()
            logits = head(model(x_emb), cue_pos)
            pred = logits.argmax(dim=-1)
            hit = (pred == target)
            correct += int(hit.sum().item())
            total += int(target.numel())
            dist = meta['distance']
            for name, (lo, hi) in (('window', (2, win)),
                                   ('mid', (win + 1, mid_edge)),
                                   ('far', (mid_edge + 1, seq_len))):
                m = (dist >= lo) & (dist <= hi)
                buckets[name][0] += int((hit & m).sum().item())
                buckets[name][1] += int(m.sum().item())
            ids = meta['input_ids']
            wrong = (~hit).nonzero(as_tuple=False).flatten().tolist()
            error_total += len(wrong)
            for j in wrong:
                key_pos = int(meta['key_pos'][j])
                distance = int(dist[j])
                pred_id = int(pred[j])
                target_id = int(target[j])
                category = _recall_error_category(
                    pred_id, target_id, ids[j], key_pos, cue_pos)
                category_counts[category] = category_counts.get(
                    category, 0) + 1
                if len(examples) < max_error_examples:
                    examples.append({
                        'batch': int(b),
                        'sample': int(j),
                        'distance': distance,
                        'distance_bucket': _recall_bucket(
                            distance, win, mid_edge),
                        'key_pos': key_pos,
                        'category': category,
                        'pred': pred_id,
                        'target': target_id,
                        'key_id': int(ids[j, key_pos]),
                        'pred_is_key': pred_id == int(ids[j, key_pos]),
                        'pred_is_previous': (
                            cue_pos > 0
                            and pred_id == int(ids[j, cue_pos - 1])),
                    })
            if b == 0:
                probe = {'x_emb': x_emb, 'key_pos': meta['key_pos'],
                         'cue_pos': cue_pos}
    out = {
        'accuracy': (correct / total) if total else None,
        'n_eval': total,
        'buckets': {k: {'correct': v[0], 'n': v[1],
                        'accuracy': (v[0] / v[1]) if v[1] else None}
                    for k, v in buckets.items()},
        'error_analysis': {
            'n_errors': error_total,
            'error_rate': ((error_total / total) if total else None),
            'categories': {
                k: {'count': c,
                    'fraction': (c / error_total) if error_total else None}
                for k, c in sorted(category_counts.items())},
            'examples': examples,
            'protocol': ('wrong predictions on the same fixed eval batches '
                         'used for accuracy; categories name the copied '
                         'token, not a hypothesised mechanism'),
        },
    }
    out['probe'] = probe
    return out


def _mean_std_ci95(values: list[float]
                   ) -> tuple[float | None, float | None, float | None]:
    """Sample mean, sample std (ddof=1) and 95% t-CI half-width.

    ``(None, None, None)`` for an empty list; std/CI are ``None`` with a
    single observation (a sample std of one point is undefined, not 0).
    """
    n = len(values)
    if n == 0:
        return None, None, None
    mean = sum(values) / n
    if n == 1:
        return mean, None, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    t = _t_crit_975(n)
    ci = (se * t) if t is not None else None
    return mean, std, ci


# ---------------------------------------------------------------------------
# Pre-registered hypotheses (fixed BEFORE the grid is run)
# ---------------------------------------------------------------------------
# The grid exists to *measure* the coverage/localisation trade-off, not to
# confirm an assumed monotonic story. Each hypothesis states its prediction
# and what would falsify it; ``_hypothesis_checks`` then evaluates the
# analytic part on the frozen structural tables and the empirical part on
# the trained rows. The naive "bigger block => more coverage" expectation is
# included ON PURPOSE: the analytic table refutes it at block boundaries
# (the strict-preceding mask `p // b < cue // b` interacts with the
# partition), which is exactly why the ablation is measured rather than
# assumed.
HYPOTHESES = {
    'window_reach_matches_geometry': {
        'hypothesis': (
            'The sliding-window path is a pure distance condition: a layer '
            'with window ``win`` recovers exactly the placements with '
            'cue - p <= win - 1.'),
        'prediction': (
            'Per block layer, ``window_hit_fraction`` equals the geometric '
            'fraction of placements inside the window, for every cell.'),
        'falsified_if': (
            'Any layer window_hit_fraction deviates from the geometry '
            '(window semantics differ from the documented [t-win+1, t] '
            'inclusive interval).'),
    },
    'far_placements_covered_by_blocks': {
        'hypothesis': (
            'The compressed-block path is distance-independent: far '
            'placements (beyond every sliding window) remain recoverable '
            'when the key/value pair shares a strictly preceding block.'),
        'prediction': (
            'For every cell, placements farther than the largest window '
            'have a strictly positive exact-recall fraction.'),
        'falsified_if': (
            'A cell has zero exact coverage beyond the windows (the sparse '
            'branch cannot reach far placements at all).'),
    },
    'block_coverage_not_monotone_in_block_size': {
        'hypothesis': (
            'NAIVE EXPECTATION under test: for a fixed layout, a larger '
            'compression block weakly increases block-path coverage '
            '(same-block probability grows).'),
        'prediction': (
            'All comparable (same layout/layer, increasing m) pairs show '
            'non-decreasing ``block_hit_fraction``.'),
        'falsified_if': (
            'Any comparable pair shows decreasing coverage as m grows: the '
            'strict-preceding mask at the query-block boundary dominates, '
            'so coverage must be measured per geometry, not assumed '
            'monotone in m.'),
    },
    'indexer_union_above_chance': {
        'hypothesis': (
            'The learned lightning indexer retrieves the gold block more '
            'often than uniform selection over the visible blocks, so the '
            'CSA path is content-addressed rather than a random choice.'),
        'prediction': (
            'Trained rows have ``indexer_union_recall_mean`` above the '
            'analytic uniform-selection union chance '
            '(``csa_union_selection_chance_indep``).'),
        'falsified_if': (
            'Union recall is at or below the analytic chance: the indexer '
            'adds no selection signal over picking uniformly at random.'),
    },
}


def _window_fraction_from_distances(structural: dict, window: int) -> float:
    """Geometric window-hit fraction from the per-distance placement table."""
    if window <= 0:
        return 0.0
    total = hit = 0
    for d_str, slot in structural['by_distance'].items():
        n = int(slot['n'])
        total += n
        if int(d_str) <= window - 1:
            hit += n
    return (hit / total) if total else 0.0


def _far_fraction_from_distances(structural: dict, max_window: int) -> float:
    """Exact-recall fraction for placements beyond every window."""
    total = hit = 0
    for d_str, slot in structural['by_distance'].items():
        if int(d_str) > max_window - 1:
            n = int(slot['n'])
            total += n
            hit += int(slot['any_exact'])
    return (hit / total) if total else 0.0


def _hypothesis_checks(rows: list[dict]) -> dict:
    """Evaluate the pre-registered hypotheses over the report rows."""
    # H1: window geometry, H2: far coverage, H3: m-monotonicity.
    window_dev = []
    far_fractions = []
    for r in rows:
        st = r['structural']
        max_win = max((e['window'] for e in st['per_layer']), default=0)
        for e in st['per_layer']:
            if e['kind'] not in BLOCK_KINDS:
                continue
            geom = _window_fraction_from_distances(st, e['window'])
            if abs(e['window_hit_fraction'] - geom) > 1e-12:
                window_dev.append({
                    'group': r['group'], 'layer_index': e['layer_index'],
                    'window': e['window'],
                    'window_hit_fraction': e['window_hit_fraction'],
                    'geometric_fraction': geom,
                })
        if max_win > 0:
            far_fractions.append({
                'group': r['group'],
                'far_exact_fraction': _far_fraction_from_distances(st, max_win),
            })
    # H3: per (layout, csa layer index) series over increasing block size.
    series: dict[tuple, list] = {}
    for r in rows:
        for e in r['structural']['per_layer']:
            if e['kind'] != 'csa':
                continue
            series.setdefault((r['layout'], e['layer_index']), {})[
                int(e['block'])] = e['block_hit_fraction']
    monotone_violations = []
    n_pairs = 0
    for key, by_block in series.items():
        blocks = sorted(by_block)
        for b1, b2 in zip(blocks, blocks[1:]):
            n_pairs += 1
            if by_block[b2] < by_block[b1] - 1e-12:
                monotone_violations.append({
                    'layout': key[0], 'layer_index': key[1],
                    'm_smaller': b1, 'coverage_smaller': by_block[b1],
                    'm_larger': b2, 'coverage_larger': by_block[b2],
                })
    # H4 (empirical): union recall above the analytic uniform-selection chance.
    above = below = no_data = 0
    per_row = []
    for r in rows:
        recall = r.get('indexer_union_recall_mean')
        chance = r['structural'].get('csa_union_selection_chance_indep')
        if recall is None or chance is None:
            no_data += 1
            continue
        is_above = bool(recall > chance + 1e-12)
        above += int(is_above)
        below += int(not is_above)
        per_row.append({
            'group': r['group'],
            'indexer_union_recall_mean': recall,
            'uniform_union_chance': chance,
            'above_chance': is_above,
        })
    far_min = (min(f['far_exact_fraction'] for f in far_fractions)
               if far_fractions else None)
    return {
        'window_reach_matches_geometry': {
            'n_deviations': len(window_dev),
            'deviations': window_dev[:8],
            'verdict': (not window_dev) if rows else None,
        },
        'far_placements_covered_by_blocks': {
            'n_cells': len(far_fractions),
            'min_far_exact_fraction': far_min,
            'verdict': (bool(far_min and far_min > 0.0)
                        if far_fractions else None),
        },
        'block_coverage_not_monotone_in_block_size': {
            'n_comparable_pairs': n_pairs,
            'n_monotonicity_violations': len(monotone_violations),
            'violation_examples': monotone_violations[:8],
            'verdict': (None if not n_pairs
                        else len(monotone_violations) == 0),
        },
        'indexer_union_above_chance': {
            'n_rows_above': above,
            'n_rows_at_or_below': below,
            'n_rows_no_data': no_data,
            'per_row': per_row,
            'verdict': (None if not (above + below)
                        else (True if below == 0
                              else (False if above == 0 else None))),
            'verdict_note': ('True = every trained cell above chance, '
                             'False = every trained cell at/below chance, '
                             'None = mixed cells (read per_row)'),
        },
    }


def run_recall_report(*, pairs=BUDGET_PAIRS, layouts=None,
                      steps: int = 80, seeds=(42, 43, 44), seq_len: int = SEQ_LEN,
                      d_model: int = D_MODEL, train_batch: int = 16,
                      eval_batches: int = 4, eval_batch: int = 64,
                      device='cpu', train: bool = True,
                      csa_topk: int | None = None,
                      val_every: int = 0, val_batches: int = 1,
                      val_batch: int = 64) -> dict:
    """The block-size x interleaving recall grid (the experiment body).

    For each (m, m2) budget pair x layout preset and each seed, trains the
    hybrid on the long-range MQAR task (unless ``train=False``) and records
    end-task accuracy, per-layer/union indexer gold-block recall on the
    first eval batch, the analytic structural summary, and the
    compressed-entry budget. Returns a strict-JSON dict with one row per
    cell plus the design metadata.

    Statistical design: all cells share the SAME ``seeds`` list, so cells
    are seed-paired; each row reports ``accuracy_mean`` /
    ``accuracy_std`` / ``accuracy_ci95`` and the indexer union-recall
    mean/std over the seeds, with the per-seed records retained. The
    default is 3 seeds; a shorter seed list is recorded as exploratory in
    ``design['statistical_design']['warning']``.
    """
    if layouts is None:
        layouts = list(LAYOUT_PRESETS)
    if isinstance(layouts, dict):
        layouts = list(layouts.items())
    else:
        layouts = [(name, LAYOUT_PRESETS[name]) if isinstance(name, str)
                   else tuple(name) for name in layouts]

    rows = []
    for m, m2 in pairs:
        for layout_name, layout in layouts:
            layout = tuple(layout)
            cfg = make_grid_cfg(m, m2, layout, d_model=d_model,
                                csa_topk=csa_topk)
            plan = plan_from_config(cfg, cfg.layout)
            structural = structural_summary(plan, seq_len)
            n_csa = sum(1 for k in layout if k == 'csa')
            n_hca = sum(1 for k in layout if k == 'hca')
            row = {
                'group': f'{layout_name}-m{m}-m2_{m2}',
                'layout': layout_name,
                'layout_sequence': '-'.join(layout),
                'm': int(m), 'm2': int(m2),
                'n_layers': len(layout), 'n_csa': n_csa, 'n_hca': n_hca,
                'n_memory': len(layout) - n_csa - n_hca,
                'compressed_entries_per_token': (n_csa / m + n_hca / m2),
                'structural': structural,
                'per_seed': [],
            }
            if train:
                for seed in seeds:
                    model, embed, head, losses, val_report = train_recall_model(
                        layout, m, m2, seed=seed, steps=steps,
                        seq_len=seq_len, d_model=d_model,
                        train_batch=train_batch, device=device,
                        val_every=val_every, val_batches=val_batches,
                        val_batch=val_batch, return_report=True)
                    ev = evaluate_recall(
                        model, embed, head, seq_len=seq_len,
                        n_batches=eval_batches, batch=eval_batch,
                        seed=seed, device=device)
                    probe = ev.pop('probe')
                    indexer = per_layer_indexer_recall(
                        model, probe['x_emb'], probe['key_pos'],
                        cue_pos=probe['cue_pos'])
                    hca = hca_block_coverage(
                        model, probe['key_pos'], cue_pos=probe['cue_pos'])
                    row['per_seed'].append({
                        'seed': int(seed),
                        'final_loss': losses[-1] if losses else None,
                        'mean_last10_loss': (
                            sum(losses[-10:]) / min(10, len(losses))
                            if losses else None),
                        # Full per-step curve (the run log, not just endpoints).
                        'loss_curve': [float(x) for x in losses],
                        # Held-out curve on a fixed validation set (None when
                        # val_every=0). Kept per seed so the row-level
                        # verdict can be audited against the raw points.
                        'train_validation': val_report,
                        'accuracy': ev['accuracy'],
                        'buckets': ev['buckets'],
                        'error_analysis': ev['error_analysis'],
                        'indexer_recall_per_layer': [
                            l['recall'] for l in indexer['layers']],
                        'indexer_chance_per_layer': [
                            l['chance'] for l in indexer['layers']],
                        'indexer_union_recall': indexer['union_recall'],
                        'hca_resolved_fraction': hca['any_resolved_fraction'],
                    })
                verdicts: dict = {}
                for s in row['per_seed']:
                    rep = s.get('train_validation') or {}
                    diag = rep.get('curve_diagnosis') or {}
                    verdict = diag.get('verdict', 'not_monitored')
                    verdicts[verdict] = verdicts.get(verdict, 0) + 1
                last_rep = row['per_seed'][-1].get('train_validation') or {}
                row['train_curve_verdicts'] = verdicts
                row['train_curve_diagnosis'] = last_rep.get(
                    'curve_diagnosis')
                row['train_val_loss_regret_from_best'] = last_rep.get(
                    'val_loss_regret_from_best')
                row['train_early_stopping_summary'] = last_rep.get(
                    'early_stopping_summary')
                accs = [s['accuracy'] for s in row['per_seed']
                        if s['accuracy'] is not None]
                mean, std, ci = _mean_std_ci95(accs)
                row['accuracy_mean'] = mean
                row['accuracy_std'] = std
                row['accuracy_ci95'] = ci
                row['accuracy_n_seeds'] = len(accs)
                union = [s['indexer_union_recall'] for s in row['per_seed']
                         if s['indexer_union_recall'] is not None]
                union_mean, union_std, union_ci = _mean_std_ci95(union)
                row['indexer_union_recall_mean'] = union_mean
                row['indexer_union_recall_std'] = union_std
                row['indexer_union_recall_ci95'] = union_ci
                chance = structural.get('csa_union_selection_chance_indep')
                row['hypothesis_checks'] = {
                    'indexer_union_above_chance': {
                        'prediction': ('union recall above the analytic '
                                       'uniform-selection union chance'),
                        'uniform_union_chance': chance,
                        'observed_union_recall': union_mean,
                        'verdict': (None if union_mean is None or chance is None
                                    else bool(union_mean > chance + 1e-12)),
                    },
                }
                hca = [s['hca_resolved_fraction'] for s in row['per_seed']
                       if s['hca_resolved_fraction'] is not None]
                row['hca_resolved_fraction_mean'] = _mean_std_ci95(hca)[0]
                # Pooled bad-case rollup: category mix over all seeds, with
                # examples from the weakest seed (the most diagnostic one).
                cat_totals: dict[str, int] = {}
                n_errors_total = 0
                n_seeds_with_errors = 0
                for s in row['per_seed']:
                    ea = s.get('error_analysis') or {}
                    n_errors_total += int(ea.get('n_errors') or 0)
                    if int(ea.get('n_errors') or 0) > 0:
                        n_seeds_with_errors += 1
                    for cat, slot in (ea.get('categories') or {}).items():
                        cat_totals[cat] = cat_totals.get(cat, 0) + int(
                            slot.get('count') or 0)
                worst = min(
                    row['per_seed'],
                    key=lambda s: (s.get('accuracy') is None,
                                   -(s.get('accuracy') or 0.0)))
                worst_ea = worst.get('error_analysis') or {}
                row['error_analysis'] = {
                    'n_errors_total': n_errors_total,
                    'n_seeds_with_errors': n_seeds_with_errors,
                    'category_totals': dict(sorted(cat_totals.items())),
                    'category_fractions': {
                        k: (v / n_errors_total) if n_errors_total else None
                        for k, v in sorted(cat_totals.items())},
                    'worst_seed': int(worst.get('seed'))
                    if worst.get('seed') is not None else None,
                    'representative_examples': worst_ea.get(
                        'examples', []),
                    'protocol': ('category mix pooled over seeds; examples '
                                 'from the lowest-accuracy seed'),
                }
            rows.append(row)

    return {
        'experiment': 'exp9_recall_reach',
        'design': {
            'question': ('how do compression block size and retrieval-layer '
                         'interleaving interact with associative recall?'),
            'task': 'long-range MQAR (one random-distance pair per sequence)',
            'seq_len': int(seq_len),
            'd_model': int(d_model),
            'steps': int(steps) if train else 0,
            'seeds': [int(s) for s in seeds],
            'n_seeds': len(seeds),
            'statistical_design': {
                'seeds_per_cell': len(seeds),
                'minimum_seeds_for_claims': 3,
                'shared_seed_list': True,
                'seed_policy': ('every (m, m2, layout) cell trains the SAME '
                                'seed list, so cells are seed-paired and '
                                'cell differences can be compared pairwise'),
                'reported_statistics': ('per-cell mean, sample std (ddof=1) '
                                        'and 95% t-CI of accuracy / indexer '
                                        'union recall over the seeds; '
                                        'per-seed records are retained'),
                'warning': (
                    f'only {len(seeds)} seed(s) in a trained run: results '
                    f'are exploratory; do not read cell-to-cell differences '
                    f'as effects at this seed budget'
                    if train and len(seeds) < 3 else None),
            },
            'pairs': [[int(a), int(b)] for a, b in pairs],
            'layouts': [name for name, _ in layouts],
            'csa_topk': (int(csa_topk) if csa_topk is not None
                         else SMALL_MODEL_SPEC['csa_topk']),
            'trained': bool(train),
            'training': {
                'optimizer': 'AdamW',
                'lr': 3e-3,
                'weight_decay_decay_group': 0.01,
                'weight_decay_no_decay_group': 0.0,
                'grad_clip_norm': 1.0,
                'train_batch': int(train_batch),
                'eval_batches': int(eval_batches),
                'eval_batch': int(eval_batch),
                'batch_gen_seed_offset': 1_000_000,
                'eval_gen_seed_offset': 5_000_000,
                'validation_monitor': {
                    'val_every': int(val_every),
                    'val_batches': int(val_batches),
                    'val_batch': int(val_batch),
                    'val_gen_seed_offset': 9_000_000,
                    'note': ('fixed held-out long-range batch scored every '
                             'val_every steps; read-only and RNG-neutral, '
                             'so it cannot change the training loss curve '
                             'or final accuracy'),
                },
            },
            'provenance': capture_provenance(),
            'note': ('structural = exact placement reachability at the cue; '
                     'indexer recall = gold compressed block among the '
                     'selected top-k on a real eval batch'),
            'hypotheses': HYPOTHESES,
            'hypothesis_checks': _hypothesis_checks(rows),
        },
        'rows': rows,
    }


# ===========================================================================
# 4. CLI
# ===========================================================================

def _print_table(report: dict) -> None:
    print(f"{'group':>34} | {'cost':>5} | {'acc mean±std':>12} | "
          f"{'struct':>7} | {'idx-union mean±std':>17} | layer recalls")
    for r in report['rows']:
        acc = r.get('accuracy_mean')
        acc_std = r.get('accuracy_std')
        if acc is None:
            acc_s = 'n/a'
        elif acc_std is None:
            acc_s = f'{acc:.3f}'
        else:
            acc_s = f'{acc:.3f}±{acc_std:.3f}'
        iu = r.get('indexer_union_recall_mean')
        iu_std = r.get('indexer_union_recall_std')
        if iu is None:
            iu_s = 'n/a'
        elif iu_std is None:
            iu_s = f'{iu:.3f}'
        else:
            iu_s = f'{iu:.3f}±{iu_std:.3f}'
        recalls = [s.get('indexer_recall_per_layer') for s in r['per_seed']]
        rec_s = ('n/a' if not recalls
                 else '/'.join('n/a' if v is None else f'{v:.2f}'
                               for v in recalls[0]))
        print(f"{r['group']:>34} | "
              f"{r['compressed_entries_per_token']:>5.3f} | "
              f"{acc_s:>12} | "
              f"{r['structural']['any_exact_fraction']:>7.3f} | "
              f"{iu_s:>17} | {rec_s}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Block-size x layer-interleaving associative-recall '
                    'reachability probe (Exp 9).')
    parser.add_argument('--json', default=None,
                        help=f'write the report to this path '
                             f'(default: {RESULTS_PATH})')
    parser.add_argument('--no-train', action='store_true',
                        help='skip training: structural reachability only')
    parser.add_argument('--quick', action='store_true',
                        help='small grid (2 pairs x 2 layouts, short run)')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--seeds', default=None,
                        help='comma-separated seeds (default: RECALL_SEEDS '
                             'count, 3 -> 42,43,44; >=3 seeds per cell is '
                             'the policy, fewer is flagged exploratory)')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN)
    parser.add_argument('--d-model', type=int, default=D_MODEL)
    parser.add_argument('--train-batch', type=int, default=16)
    parser.add_argument('--eval-batches', type=int, default=4)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--val-every', type=int, default=None,
                        help='held-out validation interval in training '
                             'steps (default: steps // 4; 0 disables)')
    parser.add_argument('--val-batches', type=int, default=None,
                        help='fixed held-out batches per validation check')
    parser.add_argument('--val-batch', type=int, default=None,
                        help='sequences per fixed held-out validation batch')
    parser.add_argument('--device', default=None)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    info = configure_torch_for_device()
    device = args.device or str(info.device)
    steps = args.steps if args.steps is not None else parse_int_env(
        'RECALL_STEPS', 80, min_value=1, logger=logger)
    if args.seeds is not None:
        seeds = tuple(int(s) for s in args.seeds.split(',') if s.strip())
        if not seeds:
            raise SystemExit('--seeds must contain at least one integer.')
    else:
        n_seeds = parse_int_env('RECALL_SEEDS', 3, min_value=1, logger=logger)
        seeds = tuple(42 + i for i in range(n_seeds))
    if not args.no_train and len(seeds) < 3:
        logger.warning(
            'only %d seed(s) requested: per-cell differences are '
            'exploratory at this seed budget (policy: >= 3)', len(seeds))

    if args.quick:
        pairs = ((4, 8), (8, 4))
        layouts = ['alternating', 'retrieval_back']
        steps = min(steps, 40)
    else:
        pairs = BUDGET_PAIRS
        layouts = list(LAYOUT_PRESETS)

    if args.no_train:
        gate_record = {'enabled': False, 'strict': None, 'reports': [],
                       'reason': 'structural-only run (--no-train)'}
    else:
        # Pre-sweep overfit gate on the first grid cell, through the exact
        # training construction path. The per-cell sweep surfaces
        # cell-level errors; this gate catches a globally broken recipe
        # before the grid burns compute.
        gate_m, gate_m2 = pairs[0]
        gate_layout_name = layouts[0]
        gate_layout = (LAYOUT_PRESETS[gate_layout_name]
                       if isinstance(gate_layout_name, str)
                       else tuple(gate_layout_name))
        gate_record, proceed = _run_ablation_overfit_gate(
            [(f'long-range m={gate_m} m2={gate_m2} '
              f'layout={gate_layout_name}',
              lambda: _build_recall_model(gate_m, gate_m2, gate_layout,
                                          args.d_model, device))],
            device, env_prefix='RECALL',
            make_task_batch=lambda b, embed, gen: make_long_range_batch(
                b, args.seq_len, 1, VOCAB, embed, device, generator=gen),
            seq_len=args.seq_len)
        if not proceed:
            return 1

    val_every = (0 if args.no_train else
                 (max(0, int(args.val_every))
                  if args.val_every is not None
                  else parse_int_env('RECALL_VAL_EVERY', max(1, steps // 4),
                                     min_value=0, logger=logger)))
    val_batches = (max(1, int(args.val_batches))
                   if args.val_batches is not None
                   else parse_int_env('RECALL_VAL_BATCHES', 1,
                                      min_value=1, logger=logger))
    val_batch = (max(1, int(args.val_batch)) if args.val_batch is not None
                 else parse_int_env('RECALL_VAL_BATCH', 64, min_value=1,
                                    logger=logger))
    report = run_recall_report(
        pairs=pairs, layouts=layouts, steps=steps, seeds=seeds,
        seq_len=args.seq_len, d_model=args.d_model,
        train_batch=args.train_batch, eval_batches=args.eval_batches,
        eval_batch=args.eval_batch, device=device,
        train=not args.no_train, val_every=val_every,
        val_batches=val_batches, val_batch=val_batch)

    # Attach the pre-sweep gate verdict so the report is a complete run
    # record (the full per-step curves + fixed-batch fingerprint).
    report['overfit_sanity_check'] = gate_record
    payload = sanitize_for_json(report)
    json.dumps(payload, allow_nan=False)
    out_path = args.json or RESULTS_PATH
    if not args.quiet:
        _print_table(report)
    write_results_json(payload, out_path, logger=logger)
    print(f'[recall-reach] wrote {out_path} '
          f'({"trained" if not args.no_train else "structural-only"})')
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    sys.exit(main())
