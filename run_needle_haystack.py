"""Experiment 12 — needle-in-a-haystack long-context retrieval with quantization-aware evaluation.

Motivation
----------
Existing experiments measure:
* Exp 4 (MQAR): synthetic associative recall with learnable embeddings
* Exp 9 (recall_reach): structural + empirical indexer recall on planted orthonormal keys
* Exp 10 (quant_recall): intrinsic state-quantization damage on planted associations
* Exp 11 (quant_lm): end-task perplexity with PTQ/QAT on synthetic corpus

None of them measure the canonical **needle-in-a-haystack** capability: a specific
fact (the "needle") is embedded in a long, mostly-irrelevant context (the
"haystack"), and the model must retrieve it when prompted. This is the standard
long-context benchmark used by the community (e.g., NIAH in LongBench, RULER,
NeedleInAHaystack from gkamradt).

This experiment adds that missing evaluation axis, with three key integrations:

1. **bf16-native evaluation** — uses the bfloat16 numerical stability patches
   from ``ops_kda_autograd.py`` and extends them to CSA/HCA for a unified
   low-precision decode path.

2. **Quantization-aware PTQ evaluation** — reuses ``kv_quant.QuantSpec`` and
   ``fake_quantize_`` to fake-quantize the recurrent state / compressed caches
   after every decode write, measuring the recall-cost curve on the *same*
   needle task (not a planted proxy).

3. **QAT recovery probe** — trains a small hybrid LM from scratch with 2-bit
   STE state writes (mirroring Exp 11 Part B) and evaluates needle retrieval
   on the QAT model vs the fp16 baseline, reporting a recovery fraction.

The evaluation uses a **deterministic, offline haystack generator** (no network,
no external tokenizer) so it runs in CI. The haystack is constructed from a
fixed vocabulary of distractor sentences, the needle FACT (the answer sentence)
is inserted at a controlled position, and the fixed QUESTION is appended as the
last tokens of the sequence. The model therefore has to retrieve the fact from
the haystack rather than answer from the local question context. Retrieval is
measured by **full-answer exact match** under greedy autoregressive decoding
(every token of the answer must match), not by the first character alone.

Every reported cell is produced by a model **trained from scratch** on a fixed
training budget and a disjoint seed band (``train_context_len`` /
``train_needle_pos``; evaluation uses the held-out ``[0, EVAL_SEED_RESERVE)``
band). An explicit **untrained control** row (random initialisation, same
architecture and evaluation sequences) is written for every base geometry so
the trained numbers can be read against their actual chance floor. Aggregate
rows carry seed-level standard deviations, 95% t-intervals and a paired
significance test of the trained model's gain over its untrained control at
the SAME inference precision (dtype and PTQ spec), so quantization damage is
not counted as a training gain or loss.

Baselines and hypotheses
------------------------
Following the repository's baseline ladder (Exp 4: rule reference -> linear ->
tuned softmax -> complex op), this experiment adds the missing simplest rung
for the needle task: a **symbolic lexical-scan rule** that locates the planted
fact by matching the answer TEMPLATE prefix in the context and copies the
following tokens. It uses no model weights, and an oracle-position variant
(copy from the known ``needle_pos``) is reported as the task-level copy
ceiling. Every trained row is annotated with machine-readable hypothesis
checks (``learned_retrieval``, ``trained_matches_rule``) fixed BEFORE the
numbers are read, so "the model retrieved the needle" cannot be claimed when
the model is no better than a grep, and a ``needle_summary`` row aggregates
the verdicts. The full pre-registered hypothesis table lives in
``NEEDLE_HYPOTHESES`` and is echoed in the JSON ``design`` block.

Output
------
``results/exp12_needle_haystack.json`` (versioned envelope via ``results_schema``)
with ``row_kind`` in {``needle_recall``, ``needle_untrained_control``,
``needle_rule_baseline``, ``needle_summary``, ``qat_needle``,
``qat_needle_summary``}.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    append_run_manifest,
    capture_provenance,
    configure_torch_for_device,
    grad_flow_report,
    make_run_manifest_entry,
    make_seeded_generator,
    parse_int_env,
    sanitize_for_json,
    set_global_seed,
    write_results_json,
)
from kv_quant import (
    QuantSpec,
    bits_per_element,
    compression_ratio,
    fake_quantize,
    fake_quantize_,
    spec_summary,
    storage_bytes,
)
from ops_fused import HybridConfig, HybridKCHAttention, _memory_layer_lookback
from multi_seed_stats import holm_adjust, summarize_paired
from run_decoding import HybridDecoding
from results_schema import write as write_schema
from train_diagnostics import (
    ValPoint,
    ValidationCurve,
    optimizer_update_report,
    raise_non_finite,
    snapshot_parameters,
    triage_gradient,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp12_needle_haystack.json')

# ---------------------------------------------------------------------------
# Haystack generation (deterministic, offline)
# ---------------------------------------------------------------------------

# Fixed vocabulary of distractor "sentences" (token sequences). Each is a
# sequence of token ids. We use a small vocab and compose the haystack by
# concatenating these with the needle inserted at a specific position.
# The vocab is shared with the model's embedding table.

# Distractor "words" — each is a few tokens. We'll use a char-level tokenizer
# compatible with train_lm_autodl's _FallbackCharTokenizer so the same
# embedding table works.
_HAYSTACK_VOCAB_CHARS = (
    "abcdefghijklmnopqrstuvwxyz .,!?;:'\"()-_"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# The needle: the FACT sentence is planted in the haystack and the QUESTION is
# appended at the very end. The answer is the fact sentence itself (the model
# must reproduce every one of its tokens). Keeping the question at the end is
# what makes this a retrieval task: answering only requires the local question
# context when the question is adjacent to the fact.
_NEEDLE_QUESTION = "What is the secret code? "
_NEEDLE_ANSWER = "The secret code is 42-alpha-7."
_NEEDLE_FACT = _NEEDLE_ANSWER

# Distractor sentences (char-level, will be tokenized by the char tokenizer)
_DISTRACTOR_SENTENCES = [
    "The quick brown fox jumps over the lazy dog. ",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
    "In a hole in the ground there lived a hobbit. ",
    "It was the best of times, it was the worst of times. ",
    "To be or not to be, that is the question. ",
    "All happy families are alike; each unhappy family is unhappy in its own way. ",
    "The sky above the port was the color of television, tuned to a dead channel. ",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. ",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendia was to remember that distant afternoon when his father took him to discover ice. ",
    "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. ",
]


# ---------------------------------------------------------------------------
# Simplest baseline: symbolic lexical-scan rule (no learned weights)
# ---------------------------------------------------------------------------
# Baseline ladder for the needle task (mirroring Exp 4's rule -> linear ->
# softmax ladder): before training ANY model, ask whether a pattern matcher
# can solve the task. The rule knows the answer TEMPLATE (the first
# ``_RULE_TEMPLATE_PREFIX_TOKENS`` characters of the fixed-format answer
# sentence are domain knowledge, not model weights), scans the context for
# the LAST occurrence of that template prefix, and copies the following
# ``n_answer`` tokens. The fact sentence is the only template match in the
# deterministic haystack, so a perfect matcher reproduces the answer exactly.
# A trained model that does not at least match this rule has not learned
# *retrieval* on the task it is being evaluated on. The rule records
# ``uses_answer_prefix_template=True`` so it can never be mistaken for a
# weight-free oracle of the full answer.
#
# The oracle-position variant (copy ``n_answer`` tokens from the known
# ``needle_pos``) uses ground-truth metadata and is reported only as the
# task-level copy ceiling: it shows that once a model LOCALISES the fact,
# reproducing the answer is pure copying, so any learned failure is a
# localisation failure rather than a generation failure.
_RULE_TEMPLATE_PREFIX_TOKENS = 6  # len("The se") for the fixed answer

# ---------------------------------------------------------------------------
# Pre-registered hypotheses (fixed BEFORE reading any numbers)
# ---------------------------------------------------------------------------
# These are echoed into the JSON ``design`` block and drive the per-row
# ``hypothesis_checks`` verdicts. A claim of the form "route X performs
# long-context retrieval" requires the learned_retrieval and
# trained_matches_rule checks to hold; otherwise the row only shows that a
# recurrent/compressed model ran, not that it retrieves.
NEEDLE_HYPOTHESES = OrderedDict([
    ('learned_retrieval', {
        'hypothesis': (
            'The needle answer is a verbatim copy of the planted fact, so a '
            'trained model that LOCATES the fact should beat the untrained '
            'initialisation floor on held-out sequences (localisation, not '
            'generation, is the difficulty).'),
        'prediction': (
            'Trained exact-match accuracy exceeds the same-precision '
            'untrained control by a positive seed-paired margin; the lexical '
            'template scan (simplest non-learned baseline) reproduces the '
            'answer at high exact match, confirming the task is lexically '
            'copy-solvable.'),
        'falsified_if': (
            'Trained accuracy is not above the untrained control, or the '
            'template rule cannot find the planted fact (the task is not '
            'lexically solvable / the generator is broken).'),
    }),
    ('trained_matches_rule', {
        'hypothesis': (
            'A "learned retrieval" claim is only meaningful if the model is '
            'at least as accurate as the simplest rule baseline: locate the '
            'fact by template match and copy the continuation.'),
        'prediction': (
            'Trained exact-match accuracy is >= the lexical-scan rule '
            'accuracy on the SAME held-out evaluation sequences.'),
        'falsified_if': (
            'A trained memory route scores below the rule baseline, i.e. '
            'the learned retrieval is weaker than grep-and-copy on the task '
            'it is evaluated on.'),
    }),
    ('distance_degradation', {
        'hypothesis': (
            'Long-context retrieval is a localisation problem: at a fixed '
            'context length, placing the fact farther from the question '
            'makes localisation harder for bounded-state / compressed '
            'memory routes.'),
        'prediction': (
            'Exact-match accuracy at the farthest needle position is not '
            'higher than at the nearest position (within the same '
            'context/route, unquantized rows).'),
        'falsified_if': (
            'Far needles are retrieved better than near ones (the distance '
            'axis does not measure localisation difficulty), or every op is '
            'at chance so the axis carries no signal.'),
    }),
])


# ---------------------------------------------------------------------------
# Data protocol: held-out evaluation vs training streams
# ---------------------------------------------------------------------------
#
# A needle/haystack sequence is a DETERMINISTIC function of its seed (plus
# the context length / needle position). Two sequences therefore collide
# exactly when their seeds do (for the same geometry), so the train/test
# disjointness guarantee is enforced at the seed level:
#
#   * evaluation draws seeds from the reserved held-out band
#     ``[0, EVAL_SEED_RESERVE)`` (the historical 0..N seed values, kept
#     unchanged so reported accuracies stay comparable);
#   * training draws seeds from ``[EVAL_SEED_RESERVE, 2**31)``, so a
#     training batch can never re-generate an evaluation sequence.
#
# ``audit_train_eval_seed_disjointness`` verifies the two sets after the
# fact and the QAT probe records the verdict with its results.

EVAL_SEED_RESERVE = 4096
TRAIN_SEED_STREAM = 1_000_000  # generator offset for the training seed draws
SEED_CEILING = 2 ** 31 - 1
# Fixed-batch overfit gate stream: disjoint from evaluation (no offset),
# training (TRAIN_SEED_STREAM) and validation (VAL_SEED_BASE).
OVERFIT_BATCH_SEED_OFFSET = 3_000_000

# Held-out validation-monitoring band. It sits between the evaluation band
# [0, EVAL_SEED_RESERVE) and the training band [EVAL_SEED_RESERVE, 2**31):
# the evaluation uses the low seeds, training draws from 4096 upward, and
# validation uses [2048, 4096), so monitoring can never inspect a training
# or evaluation sequence. The sweep checks that the requested evaluation
# band does not reach into this reservation.
VAL_SEED_BASE = EVAL_SEED_RESERVE // 2


def eval_batch_seeds(batch_index: int, batch_size: int) -> list[int]:
    """Seeds of evaluation batch ``batch_index`` (the held-out band).

    Deterministic and identical across configs, so every (memory_type,
    dtype, quantization) cell is scored on the SAME evaluation sequences
    (paired comparison) and none of them can have been trained on.
    """
    if batch_index < 0 or batch_size < 1:
        raise ValueError(
            f'eval_batch_seeds needs batch_index >= 0 and batch_size >= 1, '
            f'got batch_index={batch_index}, batch_size={batch_size}')
    lo = batch_index * batch_size
    hi = lo + batch_size
    if hi > EVAL_SEED_RESERVE:
        raise ValueError(
            f'the evaluation seed band [0, {EVAL_SEED_RESERVE}) is exhausted '
            f'(requested seeds [{lo}, {hi})); raise EVAL_SEED_RESERVE or '
            f'lower batch_size/n_batches')
    return list(range(lo, hi))


def _needle_val_seeds(seed: int, count: int) -> list[int]:
    """Deterministic validation seeds inside the reserved val band."""
    span = EVAL_SEED_RESERVE - VAL_SEED_BASE
    count = int(count)
    if count < 1:
        raise ValueError('count must be >= 1')
    if count > span:
        raise ValueError(
            f'validation batch of {count} exceeds the reserved band of '
            f'{span} seeds')
    base = VAL_SEED_BASE + ((int(seed) * 257) % (span - count + 1))
    return list(range(base, base + count))


def draw_train_seeds(generator, batch_size: int, *, device=None) -> list[int]:
    """Draw one training batch's data seeds OUTSIDE the eval band.

    Training seeds are random per step (fresh data every optimizer step),
    but restricted to ``[EVAL_SEED_RESERVE, 2**31)`` so no drawn seed can
    reproduce an evaluation sequence.

    ``make_seeded_generator`` may return ``None`` on a CUDA build that
    cannot construct a device-specific generator (it then seeds the global
    CUDA generator instead). Callers therefore pass the target ``device``
    so the fallback draw uses that device's global generator rather than
    dereferencing a missing generator.
    """
    if generator is not None:
        device = generator.device
    elif device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)
    seeds = torch.randint(EVAL_SEED_RESERVE, SEED_CEILING,
                          (int(batch_size),), generator=generator,
                          device=device)
    return seeds.cpu().tolist()


def audit_train_eval_seed_disjointness(train_seeds, eval_seeds,
                                        val_seeds=None) -> dict:
    """Independent check of train / validation / evaluation seed streams.

    Historical fields (``disjoint``, ``overlap_count``, ...) keep their
    train-vs-eval meaning. When ``val_seeds`` is supplied, the audit also
    checks the two remaining pairs and verifies that every stream lies in
    its declared band:

      * train: [EVAL_SEED_RESERVE, SEED_CEILING)
      * eval:  [0, EVAL_SEED_RESERVE), and in the three-way protocol also
               below ``VAL_SEED_BASE`` so it cannot enter the validation
               reservation;
      * val:   [VAL_SEED_BASE, EVAL_SEED_RESERVE)

    ``all_disjoint`` is the pairwise AND over every supplied stream and
    ``bands_ok`` is the membership AND, so a caller can require both
    instead of checking only train-vs-eval.
    """
    train = {int(s) for s in train_seeds}
    ev = {int(s) for s in eval_seeds}
    val = {int(s) for s in (val_seeds or [])}
    train_eval = sorted(train & ev)
    train_val = sorted(train & val)
    eval_val = sorted(ev & val)
    train_in_band = all(EVAL_SEED_RESERVE <= s < SEED_CEILING
                        for s in train)
    eval_in_band = all(0 <= s < EVAL_SEED_RESERVE for s in ev)
    val_in_band = all(VAL_SEED_BASE <= s < EVAL_SEED_RESERVE for s in val)
    eval_avoids_val = all(s < VAL_SEED_BASE for s in ev)
    all_disjoint = not (train_eval or train_val or eval_val)
    bands_ok = bool(train_in_band and eval_in_band
                    and (not val or (val_in_band and eval_avoids_val)))
    return {
        'train_seed_min': min(train) if train else None,
        'train_seed_max': max(train) if train else None,
        'n_train_seeds': len(train),
        'n_eval_seeds': len(ev),
        'n_val_seeds': len(val),
        'eval_seed_band': [0, EVAL_SEED_RESERVE],
        'val_seed_band': [VAL_SEED_BASE, EVAL_SEED_RESERVE],
        'train_seed_band_declared': [EVAL_SEED_RESERVE, SEED_CEILING],
        'overlap_count': len(train_eval),
        # With a validation stream supplied, ``disjoint`` means the full
        # three-way pairwise contract; otherwise it keeps the historical
        # train-vs-eval meaning.
        'disjoint': (all_disjoint if val_seeds is not None
                     else not train_eval),
        'overlap_examples': train_eval[:8],
        'total_overlap_count': (len(train_eval) + len(train_val)
                                + len(eval_val)),
        'train_val_overlap_count': len(train_val),
        'train_val_overlap_examples': train_val[:8],
        'eval_val_overlap_count': len(eval_val),
        'eval_val_overlap_examples': eval_val[:8],
        'disjoint_from_val': not (train_val or eval_val),
        'all_disjoint': all_disjoint,
        'train_in_band': bool(train_in_band),
        'eval_in_band': bool(eval_in_band),
        'val_in_band': bool(val_in_band),
        'eval_avoids_val_band': bool(eval_avoids_val),
        'bands_ok': bands_ok,
    }


class _NeedleCharTokenizer:
    """Character-level tokenizer matching train_lm_autodl's fallback.

    This ensures the needle/haystack tokens are compatible with the model's
    embedding table when we reuse the LM infrastructure.
    """
    def __init__(self):
        self.chars = sorted(list(set(_HAYSTACK_VOCAB_CHARS)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_token_id = 0  # 'a' -> 0, but we'll use a dedicated pad if needed

    def encode(self, text: str, truncation: bool = False, max_length: int | None = None) -> list[int]:
        ids = [self.char_to_id.get(ch, 0) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


def build_needle_haystack(
    tokenizer: _NeedleCharTokenizer,
    context_len: int,
    needle_pos: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a single needle-in-haystack sequence (canonical NIAH layout).

    The FACT sentence (the needle) is inserted at ``needle_pos``; the fixed
    QUESTION is appended as the final ``len(q_ids)`` tokens. The model must
    retrieve the fact from that position to answer, so ``needle_pos`` really
    controls the retrieval distance (the earlier Q&A-adjacent layout made the
    task answerable from local context alone).

    Args:
        tokenizer: character tokenizer
        context_len: total sequence length (haystack + fact + question)
        needle_pos: position (0-indexed) where the needle FACT starts.
        seed: RNG seed for distractor selection

    Returns:
        input_ids: [context_len] token ids
        target_ids: [1] the FIRST answer token (full-answer matching lives in
            the evaluator; this keeps the historical return shape)
        answer_start: ``context_len`` — the answer is generated AFTER the last
            question token (so the last context position predicts target 0)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Tokenize needle components
    q_ids = tokenizer.encode(_NEEDLE_QUESTION)
    fact_ids = tokenizer.encode(_NEEDLE_FACT)
    needle_len = len(fact_ids) + len(q_ids)

    haystack_len = context_len - needle_len
    if haystack_len < 0:
        raise ValueError(
            f"context_len={context_len} is shorter than the needle "
            f"(fact_len={len(fact_ids)} + question_len={len(q_ids)})")
    if needle_pos < 0 or needle_pos > haystack_len:
        raise ValueError(
            f"needle_pos={needle_pos} must be in [0, {haystack_len}] for "
            f"context_len={context_len} (haystack length excluding the "
            f"fact and the final question)")

    # Build haystack from distractor sentences - use RNG to shuffle and sample
    distractor_sentences = list(_DISTRACTOR_SENTENCES)
    # Shuffle distractors per seed
    perm = torch.randperm(len(distractor_sentences), generator=gen).tolist()
    distractor_sentences = [distractor_sentences[i] for i in perm]

    # We need context_len - needle_len tokens of haystack
    haystack_needed = context_len - needle_len
    # Repeat distractors to fill (using RNG to vary the repetition pattern)
    full_haystack = []
    while len(full_haystack) < haystack_needed:
        # Shuffle the distractor order each repetition
        perm = torch.randperm(len(distractor_sentences), generator=gen).tolist()
        for idx in perm:
            sent_ids = tokenizer.encode(distractor_sentences[idx])
            full_haystack.extend(sent_ids)
            if len(full_haystack) >= haystack_needed:
                break
    full_haystack = full_haystack[:haystack_needed]

    # Insert the FACT at needle_pos; append the QUESTION at the end.
    before = full_haystack[:needle_pos]
    after = full_haystack[needle_pos:]
    input_ids = before + fact_ids + after + q_ids
    input_ids = input_ids[:context_len]  # exact by construction

    # The model predicts the first answer token at the last context position,
    # then the remaining answer tokens autoregressively.
    answer_start = context_len
    target_id = fact_ids[0] if fact_ids else tokenizer.char_to_id.get('T', 0)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor([target_id], dtype=torch.long),
        answer_start,
    )


def build_needle_batch(
    batch_size: int,
    context_len: int,
    needle_pos: int,
    tokenizer: _NeedleCharTokenizer,
    seeds: list[int] | None = None,
    device: str | torch.device = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build a batch of needle-in-haystack sequences.

    Returns:
        input_ids: [batch, context_len]
        targets: [batch] (single target token per sequence)
        answer_positions: list of answer start positions
    """
    if seeds is None:
        seeds = list(range(batch_size))
    batch_input = []
    batch_target = []
    answer_positions = []
    for i in range(batch_size):
        inp, tgt, ans_pos = build_needle_haystack(tokenizer, context_len, needle_pos, seed=seeds[i])
        batch_input.append(inp)
        batch_target.append(tgt)
        answer_positions.append(ans_pos)
    return (
        torch.stack(batch_input).to(device),
        torch.cat(batch_target).to(device),
        answer_positions,
    )


# ---------------------------------------------------------------------------
# Model wrappers for needle evaluation
# ---------------------------------------------------------------------------

class NeedleHybridLM(nn.Module):
    """Tiny hybrid LM for needle-in-haystack evaluation.

    Wraps HybridKCHAttention + embedding + LM head. Uses the same parameterization
    as train_lm_autodl.LMWithHybrid but with a fixed small config for fast eval.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        memory_type: str = 'kda',
        n_kda: int = 2,
        n_csa: int = 1,
        n_hca: int = 1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        cfg = HybridConfig(
            d_model=d_model,
            n_heads_qk=2,
            n_heads_v=2,
            head_dim_k=32,
            head_dim_v=32,
            kda_chunk_size=0,  # recurrent path for all lengths
            csa_m=8, csa_topk=2, csa_nh=2, csa_c=32, csa_dc=64,
            csa_nIh=1, csa_cI=16,
            csa_sliding_window=16,
            hca_m2=16, hca_nh=2, hca_c=32, hca_dc=64,
            hca_sliding_window=16,
            rope_base=10000.0,
            csa_rope_dim=32, hca_rope_dim=32,
            n_kda=n_kda, n_csa=n_csa, n_hca=n_hca,
            memory_type=memory_type,
            layout=None,
        )
        self.hybrid = HybridKCHAttention(cfg, total_layers=n_layers)
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward (prefill). Returns logits [B, T, vocab]."""
        x = self.embed(input_ids)
        self.hybrid.reset_state()
        h = self.hybrid(x)
        h = self.norm_f(h)
        return self.lm_head(h)

    def reset_state(self):
        self.hybrid.reset_state()

    def forward_step(self, input_ids: torch.Tensor, quant_spec: QuantSpec | None = None) -> torch.Tensor:
        """Single-step forward for autoregressive decode with optional state quantization.

        Args:
            input_ids: [B, 1] single token
            quant_spec: if provided, fake-quantize the KDA state after the step

        Returns:
            logits: [B, vocab]
        """
        x = self.embed(input_ids)  # [B, 1, d]
        h = self.hybrid(x)  # stateful, updates internal _kda_state
        if quant_spec is not None:
            # Quantize the stacked KDA state after every write
            state = self.hybrid._kda_state
            if state is not None:
                # state: [n_layers, B, HV, K, V]
                # Flatten per layer and quantize
                n_layers = state.shape[0]
                quantized_layers = []
                for li in range(n_layers):
                    layer_state = state[li]  # [B, HV, K, V]
                    flat = layer_state.reshape(layer_state.shape[0], -1)
                    deq = fake_quantize(flat, quant_spec, ste=False).reshape_as(layer_state)
                    quantized_layers.append(deq)
                self.hybrid._kda_state = torch.stack(quantized_layers, dim=0)
        h = self.norm_f(h)
        return self.lm_head(h).squeeze(1)  # [B, vocab]


class NeedleStepwiseKDA(nn.Module):
    """Standalone stepwise KDA LM (mirroring run_quant_lm.StepwiseKDALM).

    Used for the QAT probe: trains from scratch with STE-quantized state writes.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, H: int = 2, K: int = 16, V: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # KDAAttn parameterization (from run_quality)
        from run_quality import KDAAttn
        self.kda = KDAAttn(d_model, H=H, K=K, V=V)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def reset_state(self):
        """Reset state for compatibility with evaluation loop."""
        pass  # Stepwise KDA doesn't maintain persistent state across calls

    @staticmethod
    def quantize_state(state: torch.Tensor, spec: QuantSpec, ste: bool) -> torch.Tensor:
        flat = state.reshape(state.shape[0], -1)
        return fake_quantize(flat, spec, ste=ste).reshape_as(state)

    def forward(self, input_ids: torch.Tensor, spec: QuantSpec | None = None, ste: bool = False) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f'StepwiseKDALM: expected [B, T] input_ids, got {tuple(input_ids.shape)}')
        B, T = input_ids.shape
        x = self.embed(input_ids)
        ksize = self.kda.short_conv.kernel_size[0]
        x_conv = self.kda.short_conv(
            F.pad(x.transpose(1, 2), (ksize - 1, 0))).transpose(1, 2)
        q = F.normalize(F.silu(self.kda.q(x_conv)).view(B, T, self.kda.H, self.kda.K), dim=-1)
        k = F.normalize(F.silu(self.kda.k(x_conv)).view(B, T, self.kda.H, self.kda.K), dim=-1)
        v = F.silu(self.kda.v(x_conv)).view(B, T, self.kda.H, self.kda.V)
        g = -F.softplus(
            self.kda.g_up(F.silu(self.kda.g_down(x_conv))).view(B, T, self.kda.H, self.kda.K)
        ) * self.kda.DECAY_SCALE
        beta = torch.sigmoid(self.kda.beta(x_conv)).view(B, T, self.kda.H)

        from ops_kda import naive_recurrent_kda
        state = None
        outs = []
        for t in range(T):
            o, state = naive_recurrent_kda(
                q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                g[:, t:t + 1], beta[:, t:t + 1],
                initial_state=state, output_final_state=True)
            if spec is not None:
                state = self.quantize_state(state, spec, ste)
            outs.append(self.kda.o(o[:, 0].reshape(B, -1)))
        return self.lm_head(torch.stack(outs, dim=1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _needle_error_category(pred_row, expected_row, context_row) -> dict:
    """Classify one wrong full-answer prediction (bad-case analysis).

    The task is verbatim copying, so a wrong answer is attributable to a
    concrete failure mode rather than to label ambiguity:

      * ``recency_copy`` — the model copied the final context tokens
        (a local shortcut instead of retrieving the planted fact);
      * ``wrong_from_start`` — the very first answer token was wrong
        (localisation never reached the fact);
      * ``partial_answer`` — the prefix matched but a later token diverged
        (fact found, continuation mis-copied).

    Returns the category plus enough detail to debug the sample.
    """
    n = int(expected_row.numel())
    diff = (pred_row != expected_row).nonzero(as_tuple=False)
    first_wrong = int(diff[0]) if diff.numel() else n
    recency = context_row[-n:] if context_row.numel() >= n else context_row
    if recency.numel() == n and torch.equal(pred_row, recency):
        category = 'recency_copy'
    elif first_wrong == 0:
        category = 'wrong_from_start'
    else:
        category = 'partial_answer'
    return {
        'category': category,
        'first_wrong_token': first_wrong,
        'matching_prefix_tokens': first_wrong,
        'n_answer_tokens': n,
    }


def evaluate_needle_recall(
    model: nn.Module,
    tokenizer: _NeedleCharTokenizer,
    context_len: int,
    needle_pos: int,
    batch_size: int = 8,
    n_batches: int = 4,
    device: str | torch.device = 'cpu',
    quant_spec: QuantSpec | None = None,
    dtype: torch.dtype = torch.float32,
    max_error_examples: int = 0,
) -> dict:
    """Evaluate needle retrieval by greedy decoding of the FULL answer.

    The metric is exact match: every token of ``_NEEDLE_ANSWER`` must be
    produced. The previous implementation compared only the first answer
    token, which is not the documented "answer token(s)" exact match and is
    easy to inflate (one frequent character can dominate). Token-level and
    first-token accuracies are still reported for continuity.

    With ``quant_spec`` the memory state is fake-quantized after every write
    during the prefill as well as during decoding (the PTQ contract), so the
    prefill runs token-by-token through ``forward_step``.
    """
    model.eval()
    model.to(device)
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)

    expected = tokenizer.encode(_NEEDLE_ANSWER)
    n_gen = len(expected)
    expected_t = torch.tensor(expected, dtype=torch.long, device=device)

    correct = 0
    total = 0
    token_correct = 0
    token_total = 0
    first_correct = 0
    error_counts = OrderedDict((c, 0) for c in
                               ('recency_copy', 'wrong_from_start',
                                'partial_answer'))
    error_examples: dict = OrderedDict((c, []) for c in error_counts)

    for b in range(n_batches):
        seeds = eval_batch_seeds(b, batch_size)
        input_ids, _targets, _answer_positions = build_needle_batch(
            batch_size, context_len, needle_pos, tokenizer, seeds=seeds,
            device=device)
        input_ids = input_ids.to(device)

        if hasattr(model, 'hybrid'):
            # --- Hybrid model: batch-level prefill + stateful decode -------
            decode_mode = 'greedy_autoregressive'
            model.hybrid.reset_state()
            if quant_spec is None:
                logits = model(input_ids)
                next_logits = logits[:, -1]
            else:
                # Quantize after every state write: no vectorized shortcut.
                next_logits = None
                for t in range(context_len):
                    next_logits = model.forward_step(
                        input_ids[:, t:t + 1], quant_spec)
            generated = []
            cur = next_logits.argmax(dim=-1)
            for _ in range(n_gen):
                generated.append(cur)
                cur = model.forward_step(
                    cur.unsqueeze(-1), quant_spec).argmax(dim=-1)
            preds = torch.stack(generated, dim=1)  # [B, n_gen]
        else:
            # --- Stepwise model has no persistent cross-call state: score the
            # full answer teacher-forced in ONE pass instead of re-running the
            # growing prefix n_gen times. Every answer token must still be
            # correct; only the previous-token source differs (gold vs
            # generated), which is recorded as ``decode_mode``.
            decode_mode = 'teacher_forced_full_answer'
            model.reset_state() if hasattr(model, 'reset_state') else None
            full = torch.cat(
                [input_ids,
                 expected_t[:-1].unsqueeze(0).expand(batch_size, -1)], dim=1)
            logits = model(full, spec=quant_spec, ste=False)
            start = context_len - 1
            preds = logits[:, start:start + n_gen].argmax(dim=-1)

        match = preds.eq(expected_t.unsqueeze(0))
        correct += int(match.all(dim=1).sum().item())
        token_correct += int(match.sum().item())
        token_total += batch_size * n_gen
        first_correct += int(match[:, 0].sum().item())
        total += batch_size
        if int(max_error_examples) > 0:
            row_seeds = eval_batch_seeds(b, batch_size)
            for bi in range(int(batch_size)):
                if bool(match[bi].all()):
                    continue
                info = _needle_error_category(
                    preds[bi].detach().cpu(), expected_t.detach().cpu(),
                    input_ids[bi].detach().cpu())
                cat = info['category']
                error_counts[cat] = error_counts.get(cat, 0) + 1
                bucket = error_examples.setdefault(cat, [])
                if len(bucket) < int(max_error_examples):
                    bucket.append({
                        'seed': int(row_seeds[bi]),
                        'context_len': int(context_len),
                        'needle_pos': int(needle_pos),
                        **info,
                        'expected_ids': expected_t.detach().cpu().tolist(),
                        'predicted_ids': preds[bi].detach().cpu().tolist(),
                        'expected_text': tokenizer.decode(
                            expected_t.detach().cpu().tolist()),
                        'predicted_text': tokenizer.decode(
                            preds[bi].detach().cpu().tolist()),
                        'context_tail_text': tokenizer.decode(
                            input_ids[bi, -n_gen:].detach().cpu().tolist()),
                    })

    accuracy = correct / total if total else 0.0
    out = {
        # Full-answer exact match (the headline metric).
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        # Secondary metrics kept for continuity / diagnostics.
        'first_token_accuracy': first_correct / total if total else 0.0,
        'token_accuracy': token_correct / token_total if token_total else 0.0,
        'n_answer_tokens': n_gen,
        'answer_match': 'full_answer_exact_match',
        'decode_mode': decode_mode,
        'context_len': context_len,
        'needle_pos': needle_pos,
        'answer_start': context_len,
    }
    if int(max_error_examples) > 0:
        n_wrong = int(total - correct)
        out['error_analysis'] = {
            'n_samples': int(total),
            'n_wrong': n_wrong,
            'error_rate': (n_wrong / total) if total else None,
            'categories': {
                c: {
                    'count': int(n),
                    'rate_of_wrong': (n / n_wrong) if n_wrong else 0.0,
                }
                for c, n in error_counts.items()
            },
            'examples': {c: v for c, v in error_examples.items() if v},
            'note': ('Wrong full-answer samples on held-out sequences; '
                     'recency_copy = copied the final context tokens, '
                     'wrong_from_start = first answer token wrong '
                     '(localisation failure), partial_answer = correct '
                     'tokens at the start of the answer.'),
        }
    return out


def evaluate_needle_recall_autoregressive(
    model: nn.Module,
    tokenizer: _NeedleCharTokenizer,
    context_len: int,
    needle_pos: int,
    batch_size: int = 8,
    n_batches: int = 4,
    device: str | torch.device = 'cpu',
    quant_spec: QuantSpec | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Backwards-compatible alias: the default evaluator is autoregressive."""
    return evaluate_needle_recall(
        model, tokenizer, context_len, needle_pos, batch_size=batch_size,
        n_batches=n_batches, device=device, quant_spec=quant_spec,
        dtype=dtype)


# ---------------------------------------------------------------------------
# Symbolic rule baselines (no learned weights)
# ---------------------------------------------------------------------------

def _last_subsequence(haystack: list[int], needle: list[int],
                      limit: int) -> int | None:
    """Index of the LAST occurrence of ``needle`` with ``start <= limit``.

    ``limit`` bounds the start index (``start + len(needle) - 1`` may still
    be the last position examined); returns ``None`` when there is no match.
    """
    n = len(needle)
    if n == 0 or limit < 0:
        return None
    last_start = min(limit, len(haystack) - n)
    for i in range(last_start, -1, -1):
        if haystack[i:i + n] == needle:
            return i
    return None


def rule_baseline_predict(
    input_ids: torch.Tensor,
    tokenizer: _NeedleCharTokenizer,
    n_gen: int,
    *,
    needle_pos: int | None = None,
    prefix_tokens: int = _RULE_TEMPLATE_PREFIX_TOKENS,
) -> dict:
    """Symbolic copy baselines for the needle task (no model weights).

    ``template``: scan the context for the LAST occurrence of the answer
    template prefix (the first ``prefix_tokens`` tokens of the fixed answer
    sentence), then copy the following ``n_gen`` tokens. If the full prefix
    is absent, shorter prefixes are tried (recorded per row); if no prefix
    matches at all, the recency heuristic (copy the final ``n_gen`` context
    tokens) is used. The context region searched ends ``n_gen`` tokens
    before the sequence end so the copy window stays inside the context.

    ``oracle``: copy ``n_gen`` tokens starting at the ground-truth
    ``needle_pos`` (only computed when ``needle_pos`` is given). This is a
    metadata ceiling, not a deployable baseline.

    Returns a dict with ``template_preds`` / ``oracle_preds`` ([B, n_gen]
    long tensors), ``template_starts``, and ``template_prefix_lens``.
    """
    if input_ids.dim() != 2:
        raise ValueError(
            'rule_baseline_predict expects [B, T] input_ids, got '
            f'{tuple(input_ids.shape)}')
    if n_gen < 1:
        raise ValueError(f'rule_baseline_predict: n_gen={n_gen} must be >= 1')
    rows = input_ids.tolist()
    B, T = input_ids.shape
    answer_ids = tokenizer.encode(_NEEDLE_ANSWER)
    max_prefix = max(1, min(int(prefix_tokens), len(answer_ids), n_gen))
    template_preds: list[list[int]] = []
    oracle_preds: list[list[int]] | None = [] if needle_pos is not None else None
    template_starts: list[int] = []
    template_prefix_lens: list[int] = []
    for b in range(B):
        row = rows[b]
        start = None
        used = 0
        for plen in range(max_prefix, 0, -1):
            start = _last_subsequence(row, answer_ids[:plen], T - n_gen)
            if start is not None:
                used = plen
                break
        if start is None:
            start = max(0, T - n_gen)  # recency fallback
            used = 0
        span = (row[start:start + n_gen] + [0] * n_gen)[:n_gen]
        template_preds.append(span)
        template_starts.append(start)
        template_prefix_lens.append(used)
        if oracle_preds is not None:
            op = max(0, min(int(needle_pos), max(0, T - n_gen)))
            orow = (row[op:op + n_gen] + [0] * n_gen)[:n_gen]
            oracle_preds.append(orow)
    out = {
        'template_preds': torch.tensor(template_preds, dtype=torch.long,
                                       device=input_ids.device),
        'template_starts': template_starts,
        'template_prefix_lens': template_prefix_lens,
    }
    if oracle_preds is not None:
        out['oracle_preds'] = torch.tensor(oracle_preds, dtype=torch.long,
                                           device=input_ids.device)
    return out


@torch.no_grad()
def evaluate_rule_baselines(
    tokenizer: _NeedleCharTokenizer,
    context_len: int,
    needle_pos: int,
    batch_size: int = 8,
    n_batches: int = 4,
    device: str | torch.device = 'cpu',
) -> dict:
    """Score the symbolic rule baselines on the SAME held-out eval batches.

    Uses ``eval_batch_seeds`` exactly like ``evaluate_needle_recall``, so
    the returned accuracies are paired with every trained/untrained model
    row of the same geometry.
    """
    expected = tokenizer.encode(_NEEDLE_ANSWER)
    n_gen = len(expected)
    expected_t = torch.tensor(expected, dtype=torch.long, device=device)
    template_correct = template_token_correct = oracle_correct = 0
    total = token_total = 0
    prefix_lens: set[int] = set()
    for b in range(n_batches):
        seeds = eval_batch_seeds(b, batch_size)
        input_ids, _, _ = build_needle_batch(
            batch_size, context_len, needle_pos, tokenizer, seeds=seeds,
            device=device)
        out = rule_baseline_predict(input_ids, tokenizer, n_gen,
                                    needle_pos=needle_pos)
        match = out['template_preds'].eq(expected_t.unsqueeze(0))
        template_correct += int(match.all(dim=1).sum().item())
        template_token_correct += int(match.sum().item())
        match_o = out['oracle_preds'].eq(expected_t.unsqueeze(0))
        oracle_correct += int(match_o.all(dim=1).sum().item())
        prefix_lens.update(int(p) for p in out['template_prefix_lens'])
        total += batch_size
        token_total += batch_size * n_gen
    return {
        'rule': 'lexical_template_scan',
        'template_prefix_tokens': _RULE_TEMPLATE_PREFIX_TOKENS,
        'template_prefix_lens_used': sorted(prefix_lens),
        'uses_model_weights': False,
        'uses_answer_prefix_template': True,
        'uses_oracle_position': False,
        'decode_mode': 'rule_copy',
        'accuracy': template_correct / total if total else 0.0,
        'token_accuracy': (template_token_correct / token_total
                           if token_total else 0.0),
        'correct': template_correct,
        'total': total,
        'oracle_position_accuracy': (oracle_correct / total
                                     if total else 0.0),
        'oracle_position_correct': oracle_correct,
        'n_answer_tokens': n_gen,
        'context_len': context_len,
        'needle_pos': needle_pos,
        'chance_uniform_exact_match': (1.0 / tokenizer.vocab_size) ** n_gen,
        'chance_first_token': 1.0 / tokenizer.vocab_size,
    }


def _needle_row_hypothesis_checks(
    *,
    mean_acc: float,
    untrained_mean: float,
    rule_acc: float | None,
    paired: dict | None,
    n_seeds: int,
) -> dict:
    """Machine-readable verdicts for one trained row (fixed rules)."""
    gain = (mean_acc - untrained_mean
            if mean_acc is not None and untrained_mean is not None else None)
    p_dir = None
    if isinstance(paired, dict):
        p_dir = (paired.get('t_test') or {}).get('p_value_directional')
    significant_positive = bool(
        gain is not None and gain > 0.0 and p_dir is not None
        and p_dir < 0.05)
    checks = OrderedDict()
    checks['learned_retrieval'] = {
        'prediction': ('trained accuracy > same-precision untrained control '
                       '(seed-paired)'),
        'observed_gain': gain,
        'n_seeds': int(n_seeds),
        'p_value_directional': p_dir,
        'significant_positive_gain_at_0.05': (significant_positive
                                              if n_seeds >= 2 else None),
        'evidence': ('inferential' if n_seeds >= 2 else
                     'descriptive (n<2: no paired test)'),
        'verdict': (significant_positive if n_seeds >= 2 else None),
    }
    if rule_acc is None or mean_acc is None:
        rule_verdict = None
        diff = None
    else:
        diff = mean_acc - rule_acc
        rule_verdict = bool(mean_acc >= rule_acc)
    checks['trained_matches_rule'] = {
        'prediction': 'trained accuracy >= lexical-scan rule accuracy',
        'rule_accuracy': rule_acc,
        'difference': diff,
        'verdict': rule_verdict,
    }
    return checks


def _needle_learned_retrieval_family_key(row: dict) -> tuple:
    """One hypothesis per trained model and evaluation geometry.

    The same trained weights are evaluated under several dtypes / PTQ specs
    and, when ``train_context_len`` is pinned, at several evaluation
    contexts. Those rows are repeated measurements of the SAME model, so
    counting each as independent evidence of ``learned_retrieval``
    overstates the family. The key keeps the architecture, training
    geometry and evaluation geometry distinct, while the precision cells
    collapse to one primary test per key.
    """
    return (
        row.get('memory_type'),
        row.get('n_layers'),
        row.get('n_kda'),
        row.get('n_csa'),
        row.get('n_hca'),
        row.get('train_context_len'),
        row.get('train_needle_pos'),
        row.get('context_len'),
        row.get('needle_pos'),
    )


def _needle_primary_precision_rank(row: dict) -> tuple:
    """Prefer the unquantized reference cell, then the widest format."""
    if not row.get('quantized'):
        dtype_rank = {'fp32': 0, 'fp16': 1, 'bf16': 2}.get(
            str(row.get('dtype')), 3)
        return (0, dtype_rank, 0, 0, '')
    return (
        1,
        int(row.get('bits') or 0),
        int(row.get('group_size') or 0),
        0 if row.get('symmetric') else 1,
        '',
    )


def _needle_learned_retrieval_family(rows: list[dict],
                                     alpha: float = 0.05) -> dict:
    """Holm-correct the learned-retrieval tests over primary precision cells.

    Every recall row keeps its raw directional p-value (descriptive, and
    comparable with earlier result files). The familywise block selects one
    pre-registered precision cell per (architecture, training geometry,
    evaluation geometry) and reports both raw and Holm-adjusted p-values,
    so the summary cannot present repeated dtype/PTQ measurements as
    independent support for the same trained model.
    """
    groups: dict[tuple, list[tuple]] = {}
    for row in rows:
        if row.get('row_kind') != 'needle_recall':
            continue
        learned = ((row.get('hypothesis_checks') or {})
                   .get('learned_retrieval') or {})
        p_value = learned.get('p_value_directional')
        if p_value is None:
            continue
        key = _needle_learned_retrieval_family_key(row)
        groups.setdefault(key, []).append(
            (_needle_primary_precision_rank(row), row, float(p_value)))

    primary = []
    for key, candidates in groups.items():
        candidates.sort(key=lambda item: item[0])
        _, row, p_value = candidates[0]
        primary.append((key, row, p_value, candidates))
    primary.sort(key=lambda item: str(item[0]))

    holm = holm_adjust([item[2] for item in primary], alpha=alpha)
    for item, adjusted, rejected in zip(primary, holm['adjusted'],
                                        holm['rejected']):
        _, primary_row, _, candidates = item
        for _, row, row_p in candidates:
            row['learned_retrieval_holm'] = {
                'selected_as_primary': row is primary_row,
                'family_size': holm['n_tests'],
                'raw_p_value_directional': row_p,
                'holm_adjusted_p': adjusted if row is primary_row else None,
                'significant_at_alpha': (bool(rejected)
                                         if row is primary_row else None),
                'alpha': float(alpha),
                'family_unit': ('one primary-precision row per architecture '
                                'x training geometry x evaluation geometry'),
            }

    tests = []
    for (key, row, p_value, _), adjusted, rejected in zip(
            primary, holm['adjusted'], holm['rejected']):
        tests.append({
            'memory_type': row.get('memory_type'),
            'n_layers': row.get('n_layers'),
            'n_kda': row.get('n_kda'),
            'n_csa': row.get('n_csa'),
            'n_hca': row.get('n_hca'),
            'train_context_len': row.get('train_context_len'),
            'train_needle_pos': row.get('train_needle_pos'),
            'context_len': row.get('context_len'),
            'needle_pos': row.get('needle_pos'),
            'dtype': row.get('dtype'),
            'quantized': bool(row.get('quantized')),
            'bits': row.get('bits'),
            'group_size': row.get('group_size'),
            'symmetric': row.get('symmetric'),
            'raw_p_value_directional': p_value,
            'holm_adjusted_p': adjusted,
            'reject_at_alpha': bool(rejected),
        })
    return {
        'method': 'holm_bonferroni',
        'alpha': float(alpha),
        'family_unit': ('one primary-precision row per architecture x '
                        'training geometry x evaluation geometry'),
        'n_candidate_rows': sum(len(c) for _, _, _, c in primary),
        'n_primary_tests': holm['n_tests'],
        'n_rejected': int(sum(holm['rejected'])),
        'tests': tests,
    }


def summarize_needle_hypotheses(rows: list[dict]) -> dict:
    """Aggregate the pre-registered verdicts into one summary row."""
    recall = [r for r in rows if r.get('row_kind') == 'needle_recall']
    rules = [r for r in rows if r.get('row_kind') == 'needle_rule_baseline']
    unquant = [r for r in recall if not r.get('quantized')]
    match_verdicts = []
    for r in unquant:
        v = ((r.get('hypothesis_checks') or {})
             .get('trained_matches_rule') or {}).get('verdict')
        if v is not None:
            match_verdicts.append(bool(v))
    learn_verdicts = []
    for r in recall:
        v = ((r.get('hypothesis_checks') or {})
             .get('learned_retrieval') or {}).get('verdict')
        if v is not None:
            learn_verdicts.append(bool(v))
    learn_family = _needle_learned_retrieval_family(recall)
    # Distance check: unquantized rows grouped by (context, route); compare
    # the farthest vs nearest needle position (descriptive, seed-averaged).
    # Rows are per dtype, so a position can appear several times: the
    # accuracy AT an extreme position is the mean over the rows at that
    # position (not an extreme of the (position, accuracy) tuples, which
    # would take the lowest accuracy at the nearest position and the
    # highest at the farthest one and bias the verdict).
    by_axis: dict[tuple, dict[int, list[float]]] = {}
    for r in unquant:
        key = (r.get('context_len'), r.get('memory_type'))
        pos = int(r.get('needle_pos', 0))
        by_axis.setdefault(key, {}).setdefault(pos, []).append(
            float(r.get('accuracy_mean', 0.0)))
    distance_effects = []
    for key, by_pos in sorted(by_axis.items(), key=lambda kv: str(kv[0])):
        if len(by_pos) < 2:
            # No distance axis without at least two distinct positions.
            continue
        near_pos = min(by_pos)
        far_pos = max(by_pos)
        near_vals = by_pos[near_pos]
        far_vals = by_pos[far_pos]
        near = sum(near_vals) / len(near_vals)
        far = sum(far_vals) / len(far_vals)
        distance_effects.append({
            'context_len': key[0],
            'memory_type': key[1],
            'near_accuracy': near,
            'far_accuracy': far,
            'far_minus_near': far - near,
        })
    distance_verdict = None
    if distance_effects:
        distance_verdict = all(d['far_minus_near'] <= 0.0
                               for d in distance_effects)
    return {
        'row_kind': 'needle_summary',
        'n_recall_rows': len(recall),
        'n_unquantized_recall_rows': len(unquant),
        'n_rule_baseline_rows': len(rules),
        'rule_baseline_accuracy_mean': (
            sum(r['accuracy'] for r in rules) / len(rules) if rules else None),
        'rule_baseline_accuracy_min': (
            min(r['accuracy'] for r in rules) if rules else None),
        'n_trained_rows_meeting_rule': sum(match_verdicts),
        'n_trained_rows_below_rule': (len(match_verdicts)
                                      - sum(match_verdicts)),
        'all_trained_meet_rule': (all(match_verdicts)
                                  if match_verdicts else None),
        'n_learned_retrieval_verdicts_available': len(learn_verdicts),
        'n_learned_retrieval_supported': sum(learn_verdicts),
        'n_learned_retrieval_supported_holm': learn_family['n_rejected'],
        'learned_retrieval_familywise': learn_family,
        'distance_effects': distance_effects,
        'distance_degradation_verdict': distance_verdict,
        'hypotheses': {k: v['prediction'] for k, v in NEEDLE_HYPOTHESES.items()},
        'note': ('Verdicts are descriptive aggregates over the rows of this '
                 'run; with a single seed the learned_retrieval verdict is '
                 'None (no paired test) and only the rule comparison and '
                 'distance effect are reported. The raw '
                 'n_learned_retrieval_supported count treats repeated '
                 'dtype/PTQ rows as separate verdicts; the '
                 'learned_retrieval_familywise block Holm-corrects one '
                 'primary-precision test per trained model x geometry and is '
                 'the familywise count.'),
    }


# ---------------------------------------------------------------------------
# Training (for QAT probe)
# ---------------------------------------------------------------------------

def _needle_validation_point(model: nn.Module, tokenizer: _NeedleCharTokenizer,
                             context_len: int, needle_pos: int,
                             seeds: list[int], device,
                             quant_spec: QuantSpec | None = None,
                             ste: bool = False) -> ValPoint:
    """Held-out (loss, exact-match) for one validation checkpoint.

    Uses the SAME teacher-forced full-answer objective as training but on
    held-out sequences from the reserved validation band. Read-only: no
    gradient, no optimizer step, no RNG draw; the caller's train/eval mode
    is restored.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            input_ids, _targets, answer_positions = build_needle_batch(
                len(seeds), context_len, needle_pos, tokenizer,
                seeds=seeds, device=device)
            answer_ids = tokenizer.encode(_NEEDLE_ANSWER)
            answer_t = torch.tensor(answer_ids, dtype=torch.long,
                                    device=device)
            answer_tail = answer_t[:-1].unsqueeze(0).expand(len(seeds), -1)
            full = torch.cat([input_ids, answer_tail], dim=1)
            start = answer_positions[0] - 1
            if any(p - 1 != start for p in answer_positions):
                raise RuntimeError(
                    'needle validation answer positions differ within a batch')
            if hasattr(model, 'hybrid'):
                model.hybrid.reset_state()
                logits = model(full)
            else:
                model.reset_state() if hasattr(model, 'reset_state') else None
                logits = model(full, spec=quant_spec, ste=ste)
            pred = logits[:, start:start + answer_t.numel(), :]
            target = answer_t.unsqueeze(0).expand(len(seeds), -1)
            loss = F.cross_entropy(
                pred.reshape(-1, pred.size(-1)), target.reshape(-1))
            acc = (pred.argmax(-1) == target).all(dim=1).float().mean()
            return ValPoint(loss=float(loss), acc=float(acc))
    finally:
        model.train(was_training)


def train_needle_model(
    model: nn.Module,
    tokenizer: _NeedleCharTokenizer,
    train_context_len: int,
    train_needle_pos: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: str | torch.device,
    seed: int,
    quant_spec: QuantSpec | None = None,
    ste: bool = False,
    val_every: int = 0,
    val_batches: int = 1,
) -> tuple[nn.Module, dict]:
    """Train a needle model (full-sequence or stepwise with optional QAT).

    ``val_every > 0`` additionally records a held-out (loss, exact-match)
    curve on sequences from the reserved validation band, every
    ``val_every`` steps plus step 0 and the final step. The curve never
    touches the optimizer or the training RNG, so enabling it cannot change
    the trained weights; it only makes over/underfitting visible.
    """
    set_global_seed(seed, deterministic=True)
    model.train()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = make_seeded_generator(seed + TRAIN_SEED_STREAM, device=device)

    answer_ids = tokenizer.encode(_NEEDLE_ANSWER)
    n_answer = len(answer_ids)
    answer_t = torch.tensor(answer_ids, dtype=torch.long, device=device)

    losses = []
    data_seeds: list[int] = []
    val_every = max(0, int(val_every or 0))
    val_batches = max(1, int(val_batches or 1))
    val_seeds = _needle_val_seeds(seed, val_batches * min(batch_size, 4))
    val_curve = None
    if val_every > 0:
        val_curve = ValidationCurve(
            lambda _step: _needle_validation_point(
                model, tokenizer, train_context_len, train_needle_pos,
                val_seeds, device, quant_spec=quant_spec, ste=ste),
            steps=int(steps), val_every=val_every,
            label=f'needle seed={seed} ctx={train_context_len}',
            ceiling=1.0)
        val_curve.record(0)
    for step in range(steps):
        # Draw fresh data seeds for every step from the seeded generator.
        # Training seeds live in [EVAL_SEED_RESERVE, 2**31) so no training
        # batch can reproduce a held-out evaluation sequence.
        step_seeds = draw_train_seeds(gen, batch_size, device=device)
        data_seeds.extend(step_seeds)
        # Build training batch
        input_ids, _targets, answer_positions = build_needle_batch(
            batch_size, train_context_len, train_needle_pos, tokenizer,
            seeds=step_seeds, device=device
        )
        # The question is the final context token; append the answer tokens
        # (except the last) so teacher forcing supervises EVERY answer token,
        # not just the first one. The earlier loss only scored the first
        # answer token, which cannot teach a model to reproduce the answer.
        answer_tail = answer_t[:-1].unsqueeze(0).expand(batch_size, -1)
        train_ids = torch.cat([input_ids, answer_tail], dim=1)
        start = answer_positions[0] - 1
        if any(p - 1 != start for p in answer_positions):
            raise RuntimeError('needle answer positions differ within a batch')
        if hasattr(model, 'hybrid'):
            model.hybrid.reset_state()
            logits = model(train_ids)
        else:
            model.reset_state() if hasattr(model, 'reset_state') else None
            logits = model(train_ids, spec=quant_spec, ste=ste)
        pred = logits[:, start:start + n_answer, :]
        target = answer_t.unsqueeze(0).expand(batch_size, -1)
        loss = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)), target.reshape(-1))
        if not torch.isfinite(loss):
            # Staged triage: data/token ids -> parameters -> logits ->
            # gradients, naming the first failing stage instead of a bare
            # "non-finite loss".
            raise_non_finite(
                'loss', loss=loss,
                token_ids={'train_ids': train_ids},
                id_range=(0, tokenizer.vocab_size),
                params=model.parameters(), logits=logits,
                context=f'needle training seed={seed} step={step}')
        opt.zero_grad()
        loss.backward()
        bad_grads = triage_gradient(model.parameters())
        clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=loss,
                token_ids={'train_ids': train_ids},
                id_range=(0, tokenizer.vocab_size),
                params=model.parameters(),
                grads={f'grad[{i}]': p.grad for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'needle training seed={seed} step={step}')
        opt.step()
        losses.append(float(loss.item()))
        if val_curve is not None and (step + 1) % val_every == 0:
            val_curve.record(step + 1)

    tail = losses[-min(10, len(losses)):]
    report = {
        'losses': losses,
        'final_loss': losses[-1] if losses else None,
        'initial_loss': losses[0] if losses else None,
        'min_loss': min(losses) if losses else None,
        'last10_mean_loss': (sum(tail) / len(tail)) if tail else None,
        'steps': steps,
        'batch_size': batch_size,
        'lr': lr,
        'train_context_len': train_context_len,
        'train_needle_pos': train_needle_pos,
        'n_answer_tokens': n_answer,
        # Every data seed used to build a training batch, so the caller can
        # independently audit disjointness from the held-out eval stream.
        'data_seeds': data_seeds,
    }
    if val_curve is not None:
        monitor = val_curve.finalize(losses)
        report.update({
            'val_every': int(val_every),
            'val_seeds': list(val_seeds),
            'val_history': monitor['val_history'],
            'curve_diagnosis': monitor['curve_diagnosis'],
            'generalization_gap': monitor['generalization_gap'],
            'best_val': monitor['best_val'],
            'val_loss_regret_from_best': monitor.get(
                'val_loss_regret_from_best'),
            'early_stopping_summary': monitor.get('early_stopping_summary'),
            'validation_protocol': {
                'seed_band': [VAL_SEED_BASE, EVAL_SEED_RESERVE],
                'n_sequences': len(val_seeds),
                'read_only': True,
                'disjoint_from_train': not (
                    set(val_seeds) & set(data_seeds)),
                'disjoint_from_eval': not (
                    set(val_seeds) & set(eval_batch_seeds(0, 1))),
                # Three-way audit: pairwise disjointness plus membership
                # in the declared train/eval/val seed bands. The old two
                # booleans above only compared val with one eval batch.
                'seed_audit': audit_train_eval_seed_disjointness(
                    data_seeds, eval_batch_seeds(0, 1),
                    val_seeds=val_seeds),
            },
        })
        val_curve.log_verdict(monitor)
    else:
        report.update({'val_every': 0, 'val_history': [],
                       'curve_diagnosis': None})
    return model, report


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

@dataclass
class NeedleConfig:
    context_len: int
    needle_pos: int
    memory_type: str  # 'kda', 'mamba2', 'rwkv7'
    n_layers: int = 4
    n_kda: int = 2
    n_csa: int = 1
    n_hca: int = 1
    dtype: str = 'fp32'  # 'fp32', 'fp16', 'bf16'
    quant_spec: QuantSpec | None = None


def make_default_configs() -> list[NeedleConfig]:
    """Default sweep: context lengths x needle positions x memory types x dtypes."""
    configs = []
    context_lens = [256, 512, 1024, 2048]
    needle_ratios = [0.1, 0.25, 0.5, 0.75]  # relative position
    memory_types = ['kda', 'mamba2', 'rwkv7']
    dtypes = ['fp32', 'fp16', 'bf16']

    for cl in context_lens:
        for nr in needle_ratios:
            needle_pos = int(cl * nr)
            for mt in memory_types:
                for dt in dtypes:
                    configs.append(NeedleConfig(
                        context_len=cl,
                        needle_pos=needle_pos,
                        memory_type=mt,
                        dtype=dt,
                    ))
    return configs


def make_quant_configs(base_configs: list[NeedleConfig], specs: list[QuantSpec]) -> list[NeedleConfig]:
    """Extend configs with quantization specs for PTQ sweep.

    The unquantized base row is KEPT for every config so the sweep has a
    same-seed fp16/fp32 anchor to read the PTQ damage off directly.
    """
    configs = []
    for base in base_configs:
        configs.append(base)
        # Only quantize recurrent memory types (KDA, Mamba-2, RWKV-7)
        if base.memory_type in ('kda', 'mamba2', 'rwkv7'):
            for spec in specs:
                configs.append(NeedleConfig(
                    context_len=base.context_len,
                    needle_pos=base.needle_pos,
                    memory_type=base.memory_type,
                    n_layers=base.n_layers,
                    n_kda=base.n_kda,
                    n_csa=base.n_csa,
                    n_hca=base.n_hca,
                    dtype=base.dtype,
                    quant_spec=spec,
                ))
    return configs


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

_DTYPE_BITS = {'fp32': 32.0, 'fp16': 16.0, 'bf16': 16.0}


def _arch_key(cfg: NeedleConfig) -> tuple:
    """Everything that determines the weights, minus the seed."""
    return (cfg.memory_type, cfg.n_layers, cfg.n_kda, cfg.n_csa, cfg.n_hca)


def _quant_spec_key(spec: QuantSpec | None) -> tuple | None:
    """Hashable identity of an inference quantizer (None = full precision)."""
    if spec is None:
        return None
    return (int(spec.bits), int(spec.group_size), bool(spec.symmetric))


def _build_needle_model(cfg: NeedleConfig, tokenizer: _NeedleCharTokenizer) -> NeedleHybridLM:
    return NeedleHybridLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=cfg.n_layers,
        memory_type=cfg.memory_type,
        n_kda=cfg.n_kda,
        n_csa=cfg.n_csa,
        n_hca=cfg.n_hca,
    )


# ---------------------------------------------------------------------------
# Pre-training fixed-batch overfit sanity gate
# ---------------------------------------------------------------------------
def _needle_fixed_batch(tokenizer, context_len, needle_pos, batch_size,
                        seed, device):
    """One deterministic training batch, held fixed across gate steps."""
    gen = make_seeded_generator(seed + OVERFIT_BATCH_SEED_OFFSET,
                                device=device)
    data_seeds = draw_train_seeds(gen, batch_size, device=device)
    input_ids, _targets, answer_positions = build_needle_batch(
        batch_size, context_len, needle_pos, tokenizer, seeds=data_seeds,
        device=device)
    answer_t = torch.tensor(tokenizer.encode(_NEEDLE_ANSWER),
                            dtype=torch.long, device=device)
    answer_tail = answer_t[:-1].unsqueeze(0).expand(batch_size, -1)
    train_ids = torch.cat([input_ids, answer_tail], dim=1)
    start = answer_positions[0] - 1
    if any(p - 1 != start for p in answer_positions):
        raise RuntimeError('needle answer positions differ within a batch')
    target = answer_t.unsqueeze(0).expand(batch_size, -1)
    return train_ids, start, target, data_seeds


def run_needle_overfit_sanity_check(
        *, tokenizer, memory_type='kda', context_len=64, needle_pos=8,
        batch_size=8, steps=100, lr=3e-3, seed=2026, device='cpu',
        grad_clip=1.0, acc_threshold=0.99, loss_ratio_threshold=0.5,
        log_every=0, stepwise=False, quant_spec=None, ste=False):
    """Overfit ONE fixed needle batch before the full sweep.

    Mirrors the Exp 4/5/8/11 gates: a FRESH model is trained on a single
    fixed batch with regularization disabled (``weight_decay=0``; the needle
    pipeline has no dropout or data augmentation). Passes when the tail
    exact-match accuracy is near 1.0 OR the tail loss is below
    ``loss_ratio_threshold`` times the head loss, and every gradient is
    finite. This is necessary but NOT sufficient: it proves the training
    loss is connected to the parameters; it cannot prove the eval stream,
    tokenizer alignment, or quantization path are correct.

    ``stepwise=True`` exercises the ``NeedleStepwiseKDA`` QAT path instead
    of the hybrid stack; ``quant_spec`` + ``ste=True`` then fake-quantize
    every state write through the straight-through estimator, so a broken
    STE gradient path fails here rather than silently producing random QAT
    rows.
    """
    if isinstance(device, str):
        device = torch.device(device)
    steps = int(steps)
    if steps < 10:
        raise ValueError(
            f'overfit check needs steps >= 10 to be meaningful, got {steps}')
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1, got {batch_size}')
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    seed_info = set_global_seed(seed, deterministic=True)
    cfg = NeedleConfig(context_len=int(context_len),
                       needle_pos=int(needle_pos),
                       memory_type=str(memory_type))
    if stepwise:
        model = NeedleStepwiseKDA(tokenizer.vocab_size).to(device)
    else:
        model = _build_needle_model(cfg, tokenizer).to(device)
    model.train()
    train_ids, start, target, data_seeds = _needle_fixed_batch(
        tokenizer, cfg.context_len, cfg.needle_pos, batch_size, seed, device)
    n_answer = int(target.size(1))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=0.0)

    losses, exact, token_acc, elapsed_s = [], [], [], []
    grad_flow = None
    update_report = None
    t0 = time.time()
    for step in range(steps):
        model.reset_state()
        if stepwise:
            logits = model(train_ids, spec=quant_spec, ste=ste)
        else:
            logits = model(train_ids)
        pred = logits[:, start:start + n_answer, :]
        loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                               target.reshape(-1))
        if loss is None or not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=(loss if loss is not None else float('nan')),
                token_ids={'train_ids': train_ids},
                id_range=(0, tokenizer.vocab_size),
                params=model.parameters(), logits=logits,
                context=f'needle overfit sanity check step={step} '
                        f'mem={memory_type}')
        with torch.no_grad():
            correct_tokens = (pred.argmax(-1) == target)
            exact_acc = float(correct_tokens.all(dim=1).float().mean())
            tok_acc = float(correct_tokens.float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            grad_flow = grad_flow_report(model.named_parameters())
        bad_grads = triage_gradient(model.parameters())
        clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=loss,
                token_ids={'train_ids': train_ids},
                id_range=(0, tokenizer.vocab_size),
                params=model.parameters(), logits=logits,
                grads={f'grad[{i}]': p.grad
                       for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'needle overfit sanity check step={step} '
                        f'mem={memory_type}')
        before_step = None
        if update_report is None:
            before_step = snapshot_parameters(model.named_parameters())
        opt.step()
        if update_report is None:
            update_report = optimizer_update_report(
                before_step, model.named_parameters(), opt, step=step)
        losses.append(float(loss.detach()))
        exact.append(exact_acc)
        token_acc.append(tok_acc)
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'[overfit-check] needle/{memory_type} step {step:4d} '
                  f'loss {losses[-1]:.4f} exact {exact_acc:.4f} '
                  f'token_acc {tok_acc:.4f} ({elapsed_s[-1]:.1f}s)',
                  flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    final_exact = sum(exact[-tail:]) / tail
    final_token_acc = sum(token_acc[-tail:]) / tail
    passed = ((final_exact >= acc_threshold)
              or (final_loss < loss_ratio_threshold * initial_loss))
    if update_report is not None and not update_report['ok']:
        passed = False
    import hashlib
    h = hashlib.sha256()
    for name, t in (('train_ids', train_ids), ('target', target)):
        h.update(f'{name}:{tuple(t.shape)}:'.encode())
        h.update(str(t.detach().to('cpu').reshape(-1).tolist()).encode())
    report = {
        'passed': bool(passed),
        'label': ('needle-stepwise-ste' if stepwise
                  else f'needle-{memory_type}'),
        'memory_type': str(memory_type),
        'stepwise': bool(stepwise),
        'ste': bool(ste),
        'quant_spec': (spec_summary(quant_spec)
                       if quant_spec is not None else None),
        'context_len': int(context_len),
        'needle_pos': int(needle_pos),
        'batch_size': batch_size,
        'steps': steps,
        'lr': float(lr),
        'weight_decay': 0.0,
        'dropout': 0.0,
        'data_augmentation': 'none',
        'grad_clip_norm': float(grad_clip),
        'seed': int(seed),
        'seed_info': seed_info,
        'fixed_batch_seed': int(seed + OVERFIT_BATCH_SEED_OFFSET),
        'fixed_batch_seeds': list(data_seeds),
        'fixed_batch_sha256': h.hexdigest(),
        'optimizer_updates': update_report,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': min(losses),
        'loss_reduction': ((initial_loss - final_loss) / initial_loss
                           if initial_loss > 0 else None),
        'final_exact_match': final_exact,
        'best_exact_match': max(exact),
        'final_train_token_acc': final_token_acc,
        'acc_threshold': float(acc_threshold),
        'loss_ratio_threshold': float(loss_ratio_threshold),
        'loss_curve': losses,
        'exact_match_curve': exact,
        'token_acc_curve': token_acc,
        'elapsed_s': elapsed_s,
        'grad_flow': grad_flow,
        'note': ('Necessary but NOT sufficient: passing proves the loss is '
                 'connected to the fixed-batch parameters; it cannot rule '
                 'out eval-stream / tokenizer-alignment / quantization-path '
                 'bugs.'),
    }
    if grad_flow and grad_flow['n_missing_grad']:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the fixed needle batch for "
              f"{memory_type} (e.g. "
              f"{grad_flow['missing_grad_names'][:3]})", flush=True)
    return report


def _compact_needle_gate_report(report):
    return {k: v for k, v in report.items()
            if k not in ('loss_curve', 'exact_match_curve',
                         'token_acc_curve', 'elapsed_s', 'seed_info')}


def _run_needle_overfit_gate(*, tokenizer, device, memory_types,
                             env_prefix='NEEDLE', gate_context_len=64,
                             gate_needle_pos=8, lr=3e-3,
                             include_stepwise=False, qat_spec=None):
    """Run the fixed-batch gate once per memory route; return (record, ok).

    When ``include_stepwise`` is true, one additional report trains the
    QAT/STE stepwise model on a fixed batch, so the quantized path gets
    the same pre-sweep gradient check as the hybrid routes.
    """
    enabled = parse_int_env(f'{env_prefix}_OVERFIT_CHECK', 1, min_value=0,
                            logger=logger) != 0
    strict = parse_int_env(f'{env_prefix}_OVERFIT_STRICT', 1, min_value=0,
                           logger=logger) != 0
    if not enabled:
        logger.info(f'  overfit sanity gate disabled '
                    f'({env_prefix}_OVERFIT_CHECK=0)')
        return {'enabled': False, 'strict': bool(strict), 'reports': []}, True
    steps = parse_int_env(f'{env_prefix}_OVERFIT_STEPS', 100, min_value=10,
                          logger=logger)
    seed = parse_int_env(f'{env_prefix}_OVERFIT_SEED', 2026, min_value=0,
                         logger=logger)
    batch = parse_int_env(f'{env_prefix}_OVERFIT_BATCH', 8, min_value=1,
                          logger=logger)
    routes = list(dict.fromkeys(memory_types or ['kda']))
    logger.info(f'  overfit sanity gate: {len(routes)} route(s), {steps} '
                f'steps on one fixed batch each (batch={batch}, '
                f'seed={seed}, strict={bool(strict)})')
    reports = []
    all_passed = True
    for memory_type in routes:
        try:
            report = run_needle_overfit_sanity_check(
                tokenizer=tokenizer, memory_type=memory_type,
                context_len=gate_context_len, needle_pos=gate_needle_pos,
                batch_size=batch, steps=steps, lr=lr, seed=seed,
                device=device, log_every=max(10, steps // 5))
        except Exception as e:
            logger.error(f'  [overfit] needle/{memory_type}: gate crashed: '
                         f'{e}')
            report = {'passed': False, 'label': f'needle-{memory_type}',
                      'memory_type': memory_type, 'error': str(e),
                      'steps': steps, 'seed': seed, 'batch_size': batch}
        reports.append(report)
        if report.get('passed'):
            logger.info(
                f"  [overfit] needle/{memory_type}: PASS "
                f"(exact={report['final_exact_match']:.4f}, "
                f"loss {report['initial_loss']:.4f}->"
                f"{report['final_loss']:.4f})")
        else:
            all_passed = False
            logger.error(
                f"  [overfit] needle/{memory_type}: FAIL "
                f"(exact={report.get('final_exact_match')}, "
                f"loss={report.get('final_loss')}, "
                f"error={report.get('error')})")
    if include_stepwise:
        if qat_spec is None:
            qat_spec = QuantSpec(bits=2, group_size=16)
        try:
            report = run_needle_overfit_sanity_check(
                tokenizer=tokenizer, memory_type='kda',
                context_len=gate_context_len, needle_pos=gate_needle_pos,
                batch_size=batch, steps=steps, lr=lr, seed=seed,
                device=device, log_every=max(10, steps // 5),
                stepwise=True, quant_spec=qat_spec, ste=True)
        except Exception as e:
            logger.error(
                f'  [overfit] needle-stepwise-ste: gate crashed: {e}')
            report = {'passed': False, 'label': 'needle-stepwise-ste',
                      'stepwise': True, 'ste': True, 'error': str(e),
                      'steps': steps, 'seed': seed, 'batch_size': batch}
        reports.append(report)
        if report.get('passed'):
            logger.info(
                f"  [overfit] needle-stepwise-ste: PASS "
                f"(exact={report['final_exact_match']:.4f}, "
                f"loss {report['initial_loss']:.4f}->"
                f"{report['final_loss']:.4f})")
        else:
            all_passed = False
            logger.error(
                f"  [overfit] needle-stepwise-ste: FAIL "
                f"(exact={report.get('final_exact_match')}, "
                f"loss={report.get('final_loss')}, "
                f"error={report.get('error')})")
    record = {'enabled': True, 'strict': bool(strict), 'steps': steps,
              'seed': seed, 'train_batch': batch,
              'context_len': int(gate_context_len),
              'needle_pos': int(gate_needle_pos),
              'stepwise_gate': bool(include_stepwise), 'reports': reports}
    if not all_passed and strict:
        logger.error(
            f'  {env_prefix}_OVERFIT_STRICT=1: refusing to start the needle '
            f'sweep on a recipe that cannot overfit a fixed batch. Fix the '
            f'training path, or set {env_prefix}_OVERFIT_STRICT=0 to record '
            f'the failure and continue (not recommended).')
        return record, False
    return record, True


def _mean_ci95(values: list[float]) -> list[float] | None:
    """Two-sided 95% t-interval of the mean (None for n < 2)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return [mean, mean]
    from run_quality import _t_crit_975
    crit = _t_crit_975(n)
    if crit is None:
        return None
    half = crit * sd / math.sqrt(n)
    return [mean - half, mean + half]


def _measure_needle_state(model: NeedleHybridLM,
                          tokenizer: _NeedleCharTokenizer,
                          device: str | torch.device) -> dict:
    """Measure the ACTUAL per-sequence streaming state of a trained model.

    This helper reads the live buffers after a forward pass:
    ``_kda_state`` (the stacked recurrent memory) plus each memory layer's
    short-conv / token-shift lookback.
    """
    model.eval()
    with torch.no_grad():
        ids, _, _ = build_needle_batch(
            1, 64, 0, tokenizer, seeds=[eval_batch_seeds(0, 1)[0]],
            device=device)
        model(ids)
        state = getattr(model.hybrid, '_kda_state', None)
        mem_elements = int(state.numel() // state.shape[1]) if state is not None else 0
        lookback_elements = 0
        for layer in model.hybrid.memory_layers:
            lb = _memory_layer_lookback(layer)
            if lb is not None and lb.numel():
                lookback_elements += int(lb.numel() // lb.shape[0])
    model.hybrid.reset_state()
    return {
        'memory_state_elements': mem_elements,
        'lookback_elements': lookback_elements,
        'total_state_elements': mem_elements + lookback_elements,
        'n_memory_layers': int(model.hybrid.n_memory_layers),
    }


def run_needle_sweep(
    configs: list[NeedleConfig],
    tokenizer: _NeedleCharTokenizer,
    seeds: tuple[int, ...] = (0, 1, 2),
    batch_size: int = 4,
    n_batches: int = 2,
    device: str | torch.device = 'cpu',
    train_context_len: int | None = None,
    train_needle_pos: int | None = None,
    train_steps: int = 100,
    train_batch_size: int = 4,
    train_lr: float = 3e-3,
    val_every: int | None = None,
    val_batches: int = 1,
    max_error_examples: int = 1,
) -> list[dict]:
    """Train + evaluate every needle config, with untrained controls.

    By default a separate model is trained for each EVALUATION CONTEXT LENGTH
    (at needle position ``ctx // 4``), so the context sweep measures retrieval
    and quantization damage rather than length extrapolation. Passing
    ``train_context_len`` pins one training geometry for every cell.

    Training is cached per ``(architecture, train_geometry, seed)``: dtype and
    quantization do not change the weights (PTQ/QAT is inference-time here;
    the QAT probe covers quantized training), so the 1000+ PTQ cells share a
    handful of trained models instead of evaluating random initialisations.
    Each cache entry is trained once from scratch on a disjoint seed band and
    its actual training loss/seed audit is recorded on every row that uses it.
    """
    if val_every is None:
        # One point at 0, then four evenly spaced checks: enough shape for
        # the over/underfitting diagnosis at the default 100-step budget.
        val_every = max(1, int(train_steps) // 4)
    val_every = max(0, int(val_every))
    max_error_examples = max(0, int(max_error_examples))
    rows = []
    trained: dict[tuple, tuple[dict, dict]] = {}
    untrained_acc: dict[tuple, dict] = {}
    state_meas: dict[tuple, dict] = {}
    rule_baselines: dict[tuple[int, int], dict] = {}

    for cfg in configs:
        logger.info(f'  Running: ctx={cfg.context_len}, pos={cfg.needle_pos}, '
                    f'mem={cfg.memory_type}, dtype={cfg.dtype}, '
                    f'quant={cfg.quant_spec is not None}')
        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        torch_dtype = dtype_map[cfg.dtype]
        eval_seed_band = eval_batch_seeds(0, batch_size * n_batches)
        if val_every > 0 and max(eval_seed_band) >= VAL_SEED_BASE:
            raise ValueError(
                f'evaluation band reaches {max(eval_seed_band)}, into the '
                f'validation reservation [{VAL_SEED_BASE}, '
                f'{EVAL_SEED_RESERVE}); lower batch_size*n_batches or '
                f'disable validation (val_every=0)')

        # Simplest rung of the baseline ladder: the weight-free lexical-scan
        # rule, scored on the SAME held-out sequences as every model row.
        rule_key = (cfg.context_len, cfg.needle_pos)
        if rule_key not in rule_baselines:
            rule_baselines[rule_key] = evaluate_rule_baselines(
                tokenizer, cfg.context_len, cfg.needle_pos,
                batch_size=batch_size, n_batches=n_batches, device=device)
        rule = rule_baselines[rule_key]

        train_ctx = train_context_len or cfg.context_len
        train_pos = (train_needle_pos if train_needle_pos is not None
                     else train_ctx // 4)
        per_seed_results = []
        control_results = []
        seed_train_reports = []
        for seed in seeds:
            key = (_arch_key(cfg), train_ctx, train_pos, seed)
            if key not in trained:
                # Re-seed before model construction so ``seed`` controls the
                # initialization AND the training data stream.
                set_global_seed(seed, deterministic=True)
                model = _build_needle_model(cfg, tokenizer)
                model, rep = train_needle_model(
                    model, tokenizer, train_ctx, train_pos,
                    train_steps, train_batch_size, train_lr, device, seed,
                    val_every=val_every, val_batches=val_batches)
                audit = audit_train_eval_seed_disjointness(
                    rep['data_seeds'], eval_seed_band,
                    val_seeds=rep.get('val_seeds'))
                if not (audit['all_disjoint'] and audit['bands_ok']):
                    raise RuntimeError(
                        'needle train/val/eval seed audit failed for '
                        f'{key}: train/eval={audit["overlap_examples"]}, '
                        f'train/val={audit["train_val_overlap_examples"]}, '
                        f'eval/val={audit["eval_val_overlap_examples"]}, '
                        f'bands_ok={audit["bands_ok"]}')
                rep['data_protocol'] = audit
                trained[key] = ({k: v.detach().clone()
                                 for k, v in model.state_dict().items()}, rep)

            state_dict, rep = trained[key]
            seed_train_reports.append(rep)
            model = _build_needle_model(cfg, tokenizer)
            model.load_state_dict(state_dict)
            res = evaluate_needle_recall(
                model, tokenizer,
                context_len=cfg.context_len,
                needle_pos=cfg.needle_pos,
                batch_size=batch_size,
                n_batches=n_batches,
                device=device,
                quant_spec=cfg.quant_spec,
                dtype=torch_dtype,
                max_error_examples=max_error_examples,
            )
            res['seed'] = seed
            per_seed_results.append(res)

            # Untrained control: the SAME architecture/seed with random
            # initialisation, scored on the SAME evaluation sequences AND
            # through the SAME inference precision (dtype + PTQ spec). Using
            # an unquantized control for a quantized row would confound
            # quantization damage with the training gain.
            ckey = (_arch_key(cfg), cfg.context_len, cfg.needle_pos,
                    cfg.dtype, _quant_spec_key(cfg.quant_spec), seed)
            if ckey not in untrained_acc:
                set_global_seed(seed, deterministic=True)
                ctrl = _build_needle_model(cfg, tokenizer)
                untrained_acc[ckey] = evaluate_needle_recall(
                    ctrl, tokenizer,
                    context_len=cfg.context_len,
                    needle_pos=cfg.needle_pos,
                    batch_size=batch_size,
                    n_batches=n_batches,
                    device=device,
                    quant_spec=cfg.quant_spec,
                    dtype=torch_dtype,
                )
            control_results.append(untrained_acc[ckey])

        # Aggregate trained accuracy
        accs = [r['accuracy'] for r in per_seed_results]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        sd_acc = (math.sqrt(sum((a - mean_acc) ** 2 for a in accs) / (len(accs) - 1))
                  if len(accs) > 1 else None)
        ctrl_accs = [r['accuracy'] for r in control_results]
        ctrl_mean = sum(ctrl_accs) / len(ctrl_accs) if ctrl_accs else 0.0
        diffs = [a - c for a, c in zip(accs, ctrl_accs)]
        paired = summarize_paired(diffs) if len(diffs) >= 2 else None

        # Training-curve / bad-case rollup. All per-seed reports come from
        # the SAME cached trained weights used for this row (quantization
        # is inference-time here), so the diagnosis applies to every quant
        # cell of the architecture without retraining.
        verdict_counts: dict = OrderedDict()
        for rep_i in seed_train_reports:
            v = (rep_i.get('curve_diagnosis') or {}).get('verdict')
            key = v if v is not None else 'not_monitored'
            verdict_counts[key] = verdict_counts.get(key, 0) + 1
        train_reports = []
        for seed_i, rep_i in zip(seeds, seed_train_reports):
            train_reports.append({
                'seed': int(seed_i),
                'initial_loss': rep_i.get('initial_loss'),
                'final_loss': rep_i.get('final_loss'),
                'min_loss': rep_i.get('min_loss'),
                'last10_mean_loss': rep_i.get('last10_mean_loss'),
                'val_every': rep_i.get('val_every', 0),
                'val_history': rep_i.get('val_history') or [],
                'curve_diagnosis': rep_i.get('curve_diagnosis'),
                'generalization_gap': rep_i.get('generalization_gap'),
                'best_val': rep_i.get('best_val'),
                'early_stopping_summary': rep_i.get(
                    'early_stopping_summary'),
                'validation_protocol': rep_i.get('validation_protocol'),
            })

        ea_rows = [r for r in per_seed_results if r.get('error_analysis')]
        error_aggregate = None
        if ea_rows:
            pooled: dict = OrderedDict()
            n_samples = n_wrong = 0
            for r in ea_rows:
                ea = r['error_analysis']
                n_samples += int(ea.get('n_samples', 0))
                n_wrong += int(ea.get('n_wrong', 0))
                for c, v in (ea.get('categories') or {}).items():
                    acc = pooled.setdefault(c, {'count': 0})
                    acc['count'] += int(v.get('count', 0))
            for v in pooled.values():
                v['rate_of_wrong'] = (v['count'] / n_wrong) if n_wrong else 0.0
            worst = min(ea_rows, key=lambda r: r.get('accuracy') or 0.0)
            error_aggregate = {
                'n_seeds': len(ea_rows),
                'n_samples': n_samples,
                'n_wrong': n_wrong,
                'error_rate': (n_wrong / n_samples) if n_samples else None,
                'pooled_categories': pooled,
                'representative_seed': worst.get('seed'),
                'representative_examples': worst['error_analysis'].get(
                    'examples'),
                'note': ('Wrong full-answer samples on held-out sequences; '
                         'examples come from the lowest-accuracy seed, '
                         'counts are pooled over seeds.'),
            }

        state = state_meas.get(_arch_key(cfg))
        if state is None:
            state = _measure_needle_state(model, tokenizer, device)
            state_meas[_arch_key(cfg)] = state

        total_elems = state['total_state_elements']
        dtype_bits = _DTYPE_BITS[cfg.dtype]
        fp16_bytes = total_elems * 2.0
        if cfg.quant_spec is not None:
            mem_bytes = storage_bytes(state['memory_state_elements'], cfg.quant_spec)
            look_bytes = state['lookback_elements'] * dtype_bits / 8.0
            state_bytes = mem_bytes + look_bytes
        else:
            mem_bytes = state['memory_state_elements'] * dtype_bits / 8.0
            look_bytes = state['lookback_elements'] * dtype_bits / 8.0
            state_bytes = mem_bytes + look_bytes
        effective_bits = (state_bytes * 8.0 / total_elems) if total_elems else None

        row = {
            'row_kind': 'needle_recall',
            'context_len': cfg.context_len,
            'needle_pos': cfg.needle_pos,
            'needle_ratio': cfg.needle_pos / cfg.context_len if cfg.context_len > 0 else 0,
            'memory_type': cfg.memory_type,
            'n_layers': cfg.n_layers,
            'n_kda': cfg.n_kda,
            'n_csa': cfg.n_csa,
            'n_hca': cfg.n_hca,
            'dtype': cfg.dtype,
            'dtype_bits': dtype_bits,
            'quantized': cfg.quant_spec is not None,
            'bits': cfg.quant_spec.bits if cfg.quant_spec else None,
            'group_size': cfg.quant_spec.group_size if cfg.quant_spec else None,
            'symmetric': cfg.quant_spec.symmetric if cfg.quant_spec else None,
            'bits_per_element': (bits_per_element(cfg.quant_spec)
                                 if cfg.quant_spec else dtype_bits),
            'effective_state_bits_per_element': effective_bits,
            'state_bytes': state_bytes,
            'memory_state_bytes': mem_bytes,
            'lookback_bytes': look_bytes,
            'state_elements': total_elems,
            'memory_state_elements': state['memory_state_elements'],
            'lookback_elements': state['lookback_elements'],
            'compression_vs_fp16': (fp16_bytes / state_bytes
                                    if state_bytes > 0 else None),
            # Training provenance: every row is a TRAINED model.
            'is_trained': True,
            'train_steps': rep['steps'],
            'train_context_len': rep['train_context_len'],
            'train_needle_pos': rep['train_needle_pos'],
            'train_batch_size': rep['batch_size'],
            'train_lr': rep['lr'],
            'train_initial_loss': rep['initial_loss'],
            'train_final_loss': rep['final_loss'],
            'train_min_loss': rep['min_loss'],
            'train_last10_mean_loss': rep['last10_mean_loss'],
            'train_data_protocol': rep['data_protocol'],
            # Held-out monitoring: curve shape (not just final loss),
            # early-stopping regret, and full per-seed val histories.
            'train_curve_verdicts': dict(verdict_counts),
            'train_curve_diagnosis': (
                seed_train_reports[-1].get('curve_diagnosis')
                if seed_train_reports else None),
            'train_generalization_gap': (
                seed_train_reports[-1].get('generalization_gap')
                if seed_train_reports else None),
            'train_val_regret_from_best': (
                seed_train_reports[-1].get('val_loss_regret_from_best')
                if seed_train_reports else None),
            'train_reports': train_reports,
            # Bad cases: pooled category mix + worst-seed examples.
            'error_analysis': error_aggregate,
            # Full-answer metric + dispersion/significance vs the control.
            'accuracy_mean': mean_acc,
            'accuracy_std': sd_acc,
            'accuracy_min': min(accs) if accs else 0.0,
            'accuracy_max': max(accs) if accs else 0.0,
            'accuracy_ci95': _mean_ci95(accs),
            'accuracy_per_seed': accs,
            'token_accuracy_mean': (sum(r['token_accuracy'] for r in per_seed_results)
                                    / len(per_seed_results)),
            'first_token_accuracy_mean': (sum(r['first_token_accuracy']
                                              for r in per_seed_results)
                                          / len(per_seed_results)),
            'untrained_accuracy_mean': ctrl_mean,
            'untrained_accuracy_per_seed': ctrl_accs,
            'untrained_accuracy_ci95': _mean_ci95(ctrl_accs),
            'untrained_control_quantized': bool(cfg.quant_spec is not None),
            'untrained_control_bits': (
                cfg.quant_spec.bits if cfg.quant_spec is not None else None),
            'untrained_control_group_size': (
                cfg.quant_spec.group_size
                if cfg.quant_spec is not None else None),
            'untrained_control_symmetric': (
                cfg.quant_spec.symmetric
                if cfg.quant_spec is not None else None),
            'accuracy_gain_over_untrained_mean': (
                sum(diffs) / len(diffs) if diffs else None),
            'accuracy_gain_paired': paired,
            # Simplest baseline: weight-free lexical-scan rule on the SAME
            # held-out sequences (+ oracle-position copy ceiling).
            'rule_baseline_accuracy': rule['accuracy'],
            'rule_baseline_token_accuracy': rule['token_accuracy'],
            'rule_baseline_oracle_position_accuracy': rule[
                'oracle_position_accuracy'],
            'rule_baseline_uses_answer_prefix_template': rule[
                'uses_answer_prefix_template'],
            'accuracy_gain_over_rule_mean': mean_acc - rule['accuracy'],
            'hypothesis_checks': _needle_row_hypothesis_checks(
                mean_acc=mean_acc,
                untrained_mean=ctrl_mean,
                rule_acc=rule['accuracy'],
                paired=paired,
                n_seeds=len(seeds)),
            'chance_uniform_exact_match': (1.0 / tokenizer.vocab_size) **
                                          per_seed_results[0]['n_answer_tokens'],
            'chance_first_token': 1.0 / tokenizer.vocab_size,
            'seeds': list(seeds),
            'per_seed': per_seed_results,
        }
        rows.append(row)

        # One full-precision untrained-control row per base geometry,
        # written once from the unquantized config (quantized recall rows
        # carry their own same-precision control fields).
        base_key = ('control_row', _arch_key(cfg), cfg.context_len,
                    cfg.needle_pos)
        if (cfg.quant_spec is None
                and not any(r.get('control_key') == base_key for r in rows)):
            rows.append({
                'row_kind': 'needle_untrained_control',
                'control_key': base_key,
                'context_len': cfg.context_len,
                'needle_pos': cfg.needle_pos,
                'memory_type': cfg.memory_type,
                'n_layers': cfg.n_layers,
                'n_kda': cfg.n_kda,
                'n_csa': cfg.n_csa,
                'n_hca': cfg.n_hca,
                'is_trained': False,
                'train_steps': 0,
                'accuracy_mean': ctrl_mean,
                'accuracy_std': (math.sqrt(sum((a - ctrl_mean) ** 2
                                               for a in ctrl_accs) /
                                           (len(ctrl_accs) - 1))
                                 if len(ctrl_accs) > 1 else None),
                'accuracy_min': min(ctrl_accs) if ctrl_accs else 0.0,
                'accuracy_max': max(ctrl_accs) if ctrl_accs else 0.0,
                'accuracy_ci95': _mean_ci95(ctrl_accs),
                'accuracy_per_seed': ctrl_accs,
                'chance_uniform_exact_match': (1.0 / tokenizer.vocab_size) **
                                              control_results[0]['n_answer_tokens'],
                'chance_first_token': 1.0 / tokenizer.vocab_size,
                'seeds': list(seeds),
                'per_seed': control_results,
                'untrained_control_quantized': False,
                'note': ('full-precision random-initialisation floor; every '
                         'reported needle_recall row is trained and its gain '
                         'over a SAME-precision control is tested with '
                         'summarize_paired (quantized rows carry that '
                         'control in their own untrained_accuracy fields)'),
            })

        # One rule-baseline row per geometry (independent of route, dtype
        # and quantization: the rule has no weights), written once.
        rule_row_key = ('rule_row', cfg.context_len, cfg.needle_pos)
        if not any(r.get('rule_key') == rule_row_key for r in rows):
            rows.append({
                'row_kind': 'needle_rule_baseline',
                'rule_key': rule_row_key,
                **rule,
                'is_trained': False,
                'note': ('Simplest rung of the baseline ladder: a weight-free '
                         'template lexical scan (uses a prefix of the fixed '
                         'answer sentence as domain knowledge) followed by '
                         'copying; oracle_position_accuracy copies from the '
                         'ground-truth needle_pos and is a metadata ceiling. '
                         'A trained row that does not reach rule accuracy has '
                         'not learned retrieval on this task.'),
            })

    rows.append(summarize_needle_hypotheses(rows))
    return rows


def run_qat_needle_probe(
    tokenizer: _NeedleCharTokenizer,
    seeds: tuple[int, ...] = (0, 1, 2),
    qat_spec: QuantSpec = QuantSpec(bits=2, group_size=16),
    steps: int = 100,
    qat_steps: int = 100,
    train_context_len: int = 256,
    train_needle_pos: int = 64,
    eval_context_len: int = 512,
    eval_needle_pos: int = 128,
    batch_size: int = 4,
    lr: float = 3e-3,
    device: str | torch.device = 'cpu',
) -> list[dict]:
    """QAT recovery probe for needle retrieval (mirroring Exp 11 Part B).

    Trains a stepwise KDA LM from the SAME initialization:

    1. fp16 state writes (baseline, ``steps`` optimizer steps)
    2. 2-bit STE state writes (QAT, ``qat_steps`` steps from the fp16 init)
    3. fp16 state writes continued for ``qat_steps`` from the SAME fp16
       init (equal-budget control)

    The third arm exists because arm 2 receives ``steps + qat_steps``
    optimizer steps in total while arm 1 stops at ``steps``. Comparing
    QAT against arm 1 would confound the quantizer with extra training, so
    the primary QAT contrast is against arm 3's PTQ accuracy (identical
    total budget, initialization, seed, data stream, optimizer and LR).
    """
    rows = []
    for seed in seeds:
        # Re-seed before model construction so the ``seed`` recorded in the
        # row actually controls the model initialization (same rationale as
        # ``run_needle_sweep``; without it the constructions consumed the
        # global RNG stream wherever it happened to be).
        set_global_seed(seed, deterministic=True)
        # Train fp16 baseline
        fp16_model = NeedleStepwiseKDA(tokenizer.vocab_size, d_model=64, H=2, K=16, V=16)
        fp16_model, fp16_rep = train_needle_model(
            fp16_model, tokenizer, train_context_len, train_needle_pos,
            steps, batch_size, lr, device, seed, quant_spec=None, ste=False)

        # Train QAT model from SAME initialization (re-seed)
        qat_model = NeedleStepwiseKDA(tokenizer.vocab_size, d_model=64, H=2, K=16, V=16)
        # Copy fp16 weights to qat_model
        qat_model.load_state_dict(fp16_model.state_dict())
        qat_model, qat_rep = train_needle_model(
            qat_model, tokenizer, train_context_len, train_needle_pos,
            qat_steps, batch_size, lr, device, seed, quant_spec=qat_spec, ste=True)

        # Equal-budget fp16 control: continue the SAME initialization for
        # exactly ``qat_steps`` more steps with the SAME seed/data stream.
        # Only the state quantizer differs from the QAT arm, so the
        # matched-budget contrast isolates QAT rather than extra training.
        fp16_continued_model = NeedleStepwiseKDA(
            tokenizer.vocab_size, d_model=64, H=2, K=16, V=16)
        fp16_continued_model.load_state_dict(fp16_model.state_dict())
        fp16_continued_model, fp16_cont_rep = train_needle_model(
            fp16_continued_model, tokenizer, train_context_len,
            train_needle_pos, qat_steps, batch_size, lr, device, seed,
            quant_spec=None, ste=False)

        # Evaluate every arm on the needle task (held-out seed band; the
        # SAME evaluation sequences are used for every model so the
        # fp16/PTQ/QAT comparison is paired).
        fp16_acc = evaluate_needle_recall(
            fp16_model, tokenizer, eval_context_len, eval_needle_pos,
            batch_size=8, n_batches=2, device=device, quant_spec=None, dtype=torch.float16)
        ptq_acc = evaluate_needle_recall(
            fp16_model, tokenizer, eval_context_len, eval_needle_pos,
            batch_size=8, n_batches=2, device=device, quant_spec=qat_spec, dtype=torch.float16)
        qat_acc = evaluate_needle_recall(
            qat_model, tokenizer, eval_context_len, eval_needle_pos,
            batch_size=8, n_batches=2, device=device, quant_spec=qat_spec, dtype=torch.float16)
        qat_fp16_acc = evaluate_needle_recall(
            qat_model, tokenizer, eval_context_len, eval_needle_pos,
            batch_size=8, n_batches=2, device=device, quant_spec=None, dtype=torch.float16)
        # The matched control must be scored through the SAME quantizer as
        # the QAT arm (both are 2-bit PTQ at inference); its unquantized
        # score separately shows what extra fp16 training alone bought.
        fp16_cont_acc = evaluate_needle_recall(
            fp16_continued_model, tokenizer, eval_context_len,
            eval_needle_pos, batch_size=8, n_batches=2, device=device,
            quant_spec=None, dtype=torch.float16)
        fp16_cont_ptq_acc = evaluate_needle_recall(
            fp16_continued_model, tokenizer, eval_context_len,
            eval_needle_pos, batch_size=8, n_batches=2, device=device,
            quant_spec=qat_spec, dtype=torch.float16)

        # Leakage audit: every sequence fed to any training run (fp16
        # baseline, QAT, equal-budget fp16 control) must be disjoint from
        # the evaluation sequences.
        eval_seeds = (eval_batch_seeds(0, 8) + eval_batch_seeds(1, 8))
        stream_audit = audit_train_eval_seed_disjointness(
            list(fp16_rep.get('data_seeds') or [])
            + list(qat_rep.get('data_seeds') or [])
            + list(fp16_cont_rep.get('data_seeds') or []),
            eval_seeds,
            val_seeds=(list(fp16_rep.get('val_seeds') or [])
                       + list(qat_rep.get('val_seeds') or [])
                       + list(fp16_cont_rep.get('val_seeds') or [])))
        if not (stream_audit['all_disjoint'] and stream_audit['bands_ok']):
            raise RuntimeError(
                'needle train/val/eval seed audit failed: overlapping or '
                f'out-of-band seeds {stream_audit["overlap_examples"]} / '
                f'{stream_audit["train_val_overlap_examples"]} / '
                f'{stream_audit["eval_val_overlap_examples"]} '
                f'(bands_ok={stream_audit["bands_ok"]}); refusing to report '
                'leaked numbers')

        # Recovery fraction: 1 - (QAT_damage / PTQ_damage)
        # where damage = (acc_fp16 - acc_quantized) / acc_fp16
        fp16_a = fp16_acc['accuracy']
        ptq_a = ptq_acc['accuracy']
        qat_a = qat_acc['accuracy']
        damage = (fp16_a - ptq_a) / fp16_a if fp16_a > 0 else 0
        residual = (fp16_a - qat_a) / fp16_a if fp16_a > 0 else 0
        recovery = 1.0 - residual / damage if damage > 0 else None

        fp16_cont_a = fp16_cont_acc['accuracy']
        fp16_cont_ptq_a = fp16_cont_ptq_acc['accuracy']
        # Same recovery definition, but against the equally trained fp16
        # control (the causal QAT contrast) instead of the shorter arm.
        matched_damage = ((fp16_cont_a - fp16_cont_ptq_a) / fp16_cont_a
                          if fp16_cont_a > 0 else 0.0)
        matched_residual = ((fp16_cont_a - qat_a) / fp16_cont_a
                            if fp16_cont_a > 0 else 0.0)
        recovery_matched = (1.0 - matched_residual / matched_damage
                            if matched_damage > 0 else None)
        rows.append({
            'row_kind': 'qat_needle',
            'seed': seed,
            'steps': steps,
            'qat_steps': qat_steps,
            # Total optimizer steps per arm. The QAT and fp16-continued
            # arms must match exactly for the primary contrast to isolate
            # the quantizer rather than the training budget.
            'total_steps_fp16': steps,
            'total_steps_qat': steps + qat_steps,
            'total_steps_fp16_continued': steps + qat_steps,
            'budget_matched': True,
            'fp16_accuracy': fp16_a,
            'ptq_accuracy': ptq_a,
            'qat_accuracy': qat_a,
            'qat_fp16_accuracy': qat_fp16_acc['accuracy'],
            'fp16_continued_accuracy': fp16_cont_a,
            'fp16_continued_ptq_accuracy': fp16_cont_ptq_a,
            # Primary QAT effect at matched budget: positive means the
            # quantized-write training beats the equally trained fp16
            # control under the same 2-bit inference quantizer.
            'qat_matched_gain': qat_a - fp16_cont_ptq_a,
            # What extra optimization alone buys under the same quantizer.
            'extra_training_gain': fp16_cont_ptq_a - ptq_a,
            'qat_gain_vs_ptq': qat_a - ptq_a,
            'recovery_fraction': recovery,
            'recovery_fraction_matched': recovery_matched,
            'train_context_len': train_context_len,
            'train_needle_pos': train_needle_pos,
            'eval_context_len': eval_context_len,
            'eval_needle_pos': eval_needle_pos,
            'spec': spec_summary(qat_spec),
            'data_protocol': stream_audit,
        })

    # Summary over per-seed rows only (the summary row is appended last).
    per_seed_rows = [r for r in rows if r['row_kind'] == 'qat_needle']

    def _moment(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return {'mean': None, 'std': None, 'ci95': None,
                    'min': None, 'n': 0}
        mean = sum(vals) / len(vals)
        std = (math.sqrt(sum((v - mean) ** 2 for v in vals)
                         / (len(vals) - 1)) if len(vals) > 1 else None)
        return {'mean': mean, 'std': std,
                'ci95': _mean_ci95(vals),
                'min': min(vals), 'n': len(vals)}

    def _paired(values):
        vals = [v for v in values if v is not None]
        return summarize_paired(vals) if len(vals) >= 2 else None

    fp16 = _moment(r['fp16_accuracy'] for r in per_seed_rows)
    ptq = _moment(r['ptq_accuracy'] for r in per_seed_rows)
    qat = _moment(r['qat_accuracy'] for r in per_seed_rows)
    fp16_cont = _moment(r['fp16_continued_accuracy']
                        for r in per_seed_rows)
    fp16_cont_ptq = _moment(r['fp16_continued_ptq_accuracy']
                            for r in per_seed_rows)
    matched_gain = _moment(r['qat_matched_gain'] for r in per_seed_rows)
    extra_gain = _moment(r['extra_training_gain'] for r in per_seed_rows)
    rec = _moment(r['recovery_fraction'] for r in per_seed_rows)
    rec_matched = _moment(r['recovery_fraction_matched']
                          for r in per_seed_rows)
    qat_vs_matched = _paired(r['qat_matched_gain'] for r in per_seed_rows)
    qat_vs_ptq = _paired(r['qat_gain_vs_ptq'] for r in per_seed_rows)

    rows.append({
        'row_kind': 'qat_needle_summary',
        'seeds': list(seeds),
        'n_seeds': len(per_seed_rows),
        # Historical fields (kept for downstream consumers).
        'fp16_accuracy_mean': fp16['mean'],
        'ptq_accuracy_mean': ptq['mean'],
        'qat_accuracy_mean': qat['mean'],
        'recovery_fraction_mean': rec['mean'],
        'recovery_fraction_min': rec['min'],
        'recovery_fraction_matched_mean': rec_matched['mean'],
        'recovery_fraction_matched_std': rec_matched['std'],
        'recovery_fraction_matched_ci95': rec_matched['ci95'],
        'recovery_fraction_matched_min': rec_matched['min'],
        # Mean +/- sample std + 95% t-CI for every arm.
        'fp16_accuracy_std': fp16['std'],
        'fp16_accuracy_ci95': fp16['ci95'],
        'ptq_accuracy_std': ptq['std'],
        'ptq_accuracy_ci95': ptq['ci95'],
        'qat_accuracy_std': qat['std'],
        'qat_accuracy_ci95': qat['ci95'],
        'fp16_continued_accuracy_mean': fp16_cont['mean'],
        'fp16_continued_accuracy_std': fp16_cont['std'],
        'fp16_continued_accuracy_ci95': fp16_cont['ci95'],
        'fp16_continued_ptq_accuracy_mean': fp16_cont_ptq['mean'],
        'fp16_continued_ptq_accuracy_std': fp16_cont_ptq['std'],
        'fp16_continued_ptq_accuracy_ci95': fp16_cont_ptq['ci95'],
        # Budget-matched QAT contrast (the primary claim).
        'qat_matched_gain_mean': matched_gain['mean'],
        'qat_matched_gain_std': matched_gain['std'],
        'qat_matched_gain_ci95': matched_gain['ci95'],
        'qat_vs_matched_control': qat_vs_matched,
        # Extra training alone (quantized inference on both arms).
        'extra_training_gain_mean': extra_gain['mean'],
        'extra_training_gain_std': extra_gain['std'],
        'extra_training_gain_ci95': extra_gain['ci95'],
        'qat_vs_ptq': qat_vs_ptq,
        'budget_control': {
            'total_steps_fp16': steps,
            'total_steps_qat': steps + qat_steps,
            'total_steps_fp16_continued': steps + qat_steps,
            'matched': True,
            'note': ('QAT and the fp16-continued control share the same '
                     'initialization, seed, data stream, optimizer, LR and '
                     'total optimizer-step budget; only the state-write '
                     'quantizer differs. ``qat_vs_matched_control`` and '
                     '``recovery_fraction_matched`` are the causal QAT '
                     'contrasts; ``qat_vs_ptq`` and the historical '
                     '``recovery_fraction`` use the shorter fp16 arm and '
                     'are descriptive only.'),
        },
        'spec': spec_summary(qat_spec),
        # Every per-seed row carries its own audit; the summary states the
        # conjunction so a consumer does not have to walk the rows.
        'train_eval_disjoint_all_seeds': all(
            r.get('data_protocol', {}).get('disjoint', False)
            for r in per_seed_rows),
        'eval_seed_band': [0, EVAL_SEED_RESERVE],
        'train_seed_band': [EVAL_SEED_RESERVE, SEED_CEILING],
    })
    return rows


def _print_needle_rows(rows: list[dict]):
    logger.info('\n-- Needle-in-haystack recall (full-answer exact match) --')
    header = (f"{'ctx':>5} | {'pos':>5} | {'mem':>7} | {'dtype':>5} | "
              f"{'state':>9} | {'acc mean':>8} | {'ci95':>15} | "
              f"{'ctrl':>6} | {'rule':>6} | {'gain':>6} | "
              f"{'>>rule':>6} | {'st':>5}")
    logger.info(header)
    logger.info('-' * len(header))
    for r in rows:
        if r['row_kind'] != 'needle_recall':
            continue
        state_str = (f"{r['bits']}b/g{r['group_size']}"
                     if r['quantized'] else r['dtype'])
        ci = r.get('accuracy_ci95')
        ci_str = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else 'n/a'
        st = r.get('accuracy_std')
        st_str = f'{st:.3f}' if st is not None else 'n/a'
        rule_acc = r.get('rule_baseline_accuracy')
        rule_str = f'{rule_acc:.3f}' if rule_acc is not None else 'n/a'
        meets = ((r.get('hypothesis_checks') or {})
                 .get('trained_matches_rule') or {}).get('verdict')
        meets_str = ('yes' if meets else 'no') if meets is not None else 'n/a'
        logger.info(
            f"{r['context_len']:>5} | {r['needle_pos']:>5} | {r['memory_type']:>7} | "
            f"{r['dtype']:>5} | {state_str:>9} | "
            f"{r['accuracy_mean']:>8.3f} | {ci_str:>15} | "
            f"{r['untrained_accuracy_mean']:>6.3f} | "
            f"{rule_str:>6} | "
            f"{r['accuracy_gain_over_untrained_mean']:>+6.3f} | "
            f"{r['accuracy_gain_over_rule_mean']:>+6.3f} | {st_str:>5} "
            f"[{meets_str}]")


def _print_qat_rows(rows: list[dict]):
    logger.info('\n-- QAT needle recovery --')
    for r in rows:
        if r['row_kind'] != 'qat_needle_summary':
            continue
        # ``recovery_fraction_mean`` / ``_min`` are None when no seed had a
        # positive PTQ damage baseline (e.g. an all-zero fp16 accuracy), so
        # format them defensively instead of crashing on ``None:.1%``.
        rec = r['recovery_fraction_mean']
        rec_min = r['recovery_fraction_min']
        rec_str = f'{rec:.1%}' if rec is not None else 'n/a'
        rec_min_str = f'{rec_min:.1%}' if rec_min is not None else 'n/a'
        logger.info(
            f"  fp16 {r['fp16_accuracy_mean']:.3f} "
            f"({r.get('n_seeds', 0)} seeds) | "
            f"PTQ {r['ptq_accuracy_mean']:.3f} | "
            f"QAT {r['qat_accuracy_mean']:.3f} | "
            f"fp16-cont {r.get('fp16_continued_accuracy_mean', float('nan')):.3f} | "
            f"fp16-cont-PTQ "
            f"{r.get('fp16_continued_ptq_accuracy_mean', float('nan')):.3f} | "
            f"recovery {rec_str} (min {rec_min_str})")
        matched = r.get('qat_matched_gain_mean')
        if matched is not None:
            ci = r.get('qat_matched_gain_ci95')
            ci_str = f' [{ci[0]:+.3f},{ci[1]:+.3f}]' if ci else ''
            logger.info(
                f"  budget-matched QAT gain {matched:+.3f}{ci_str} "
                f"(vs equally trained fp16 control)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Needle-in-a-haystack retrieval with quantization-aware evaluation')
    parser.add_argument('--json', default=RESULTS_PATH, help='output path')
    parser.add_argument('--quick', action='store_true', help='small sweep for CI')
    parser.add_argument('--no-qat', action='store_true', help='skip QAT probe')
    parser.add_argument('--seeds', default='0,1,2', help='comma-separated seeds')
    parser.add_argument('--device', default=None, help='device override')
    parser.add_argument('--context-lens', default=None, help='comma-separated context lengths')
    parser.add_argument('--memory-types', default=None, help='comma-separated memory types')
    parser.add_argument('--dtypes', default=None, help='comma-separated dtypes (fp32,fp16,bf16)')
    parser.add_argument('--quant-bits', default=None, help='quantization bits for PTQ')
    parser.add_argument('--quant-groups', default=None, help='quantization group sizes')
    parser.add_argument('--train-steps', type=int, default=None,
                        help='optimizer steps per trained model')
    parser.add_argument('--train-context-len', type=int, default=None,
                        help='training geometry: context length')
    parser.add_argument('--train-needle-pos', type=int, default=None,
                        help='training geometry: needle position')
    parser.add_argument('--train-batch-size', type=int, default=4)
    parser.add_argument('--train-lr', type=float, default=3e-3)
    parser.add_argument('--val-every', type=int, default=None,
                        help='held-out validation interval in optimizer '
                             'steps (default: train_steps // 4; 0 disables)')
    parser.add_argument('--val-batches', type=int, default=1,
                        help='held-out validation sequences per checkpoint')
    parser.add_argument('--max-error-examples', type=int, default=1,
                        help='wrong samples retained per bad-case category '
                             '(0 disables the per-row error report)')
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--eval-batches', type=int, default=None,
                        help='held-out evaluation batches per cell')
    args = parser.parse_args()

    info = configure_torch_for_device()
    device = args.device or str(info.device)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    logger.info('=' * 70)
    logger.info('Experiment 12: needle-in-a-haystack retrieval + quantization-aware evaluation')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')

    seeds = tuple(int(s) for s in args.seeds.split(',') if s.strip())
    tokenizer = _NeedleCharTokenizer()

    # Explicit CLI filters override the phase defaults in BOTH modes.
    context_lens = ([int(x) for x in args.context_lens.split(',')]
                    if args.context_lens else None)
    memory_types = args.memory_types.split(',') if args.memory_types else None
    dtypes = args.dtypes.split(',') if args.dtypes else None
    quant_bits = ([int(x) for x in args.quant_bits.split(',')]
                  if args.quant_bits else None)
    quant_groups = ([int(x) for x in args.quant_groups.split(',')]
                    if args.quant_groups else None)

    if args.quick:
        context_lens = context_lens or [256, 512]
        needle_ratios = [0.25, 0.75]
        memory_types = memory_types or ['kda', 'mamba2']
        dtypes = dtypes or ['fp32', 'bf16']
        quant_bits = quant_bits or [2, 4]
        quant_groups = quant_groups or [16]
        qat_seeds = (0,)
        qat_steps = 20
        train_steps = 100
        eval_batches = 2
    else:
        context_lens = context_lens or [256, 512, 1024, 2048]
        needle_ratios = [0.1, 0.25, 0.5, 0.75]
        memory_types = memory_types or ['kda', 'mamba2', 'rwkv7']
        dtypes = dtypes or ['fp32', 'fp16', 'bf16']
        quant_bits = quant_bits or [2, 3, 4]
        quant_groups = quant_groups or [16, 64]
        # QAT is a key A/B comparison: >=3 seeds are required for any
        # dispersion/paired summary (the quick profile stays at one seed
        # and is explicitly plumbing-only).
        qat_seeds = (0, 1, 2)
        qat_steps = 100
        train_steps = 100
        eval_batches = 4

    train_steps = args.train_steps or train_steps
    train_context_len = args.train_context_len
    train_needle_pos = args.train_needle_pos
    eval_batches = args.eval_batches or eval_batches

    # Build base configs. Skip (context_len, ratio) cells where the fact or
    # the final question would not fit.
    fact_len = len(tokenizer.encode(_NEEDLE_FACT))
    question_len = len(tokenizer.encode(_NEEDLE_QUESTION))
    base_configs = []
    for cl in context_lens:
        for nr in needle_ratios:
            needle_pos = int(cl * nr)
            haystack_len = cl - fact_len - question_len
            if haystack_len < 0 or needle_pos > haystack_len:
                logger.info(f'  skipping ctx={cl}, ratio={nr}: fact '
                            f'(len={fact_len}) + final question '
                            f'(len={question_len}) leaves a haystack of '
                            f'{haystack_len}, needle_pos={needle_pos}')
                continue
            for mt in memory_types:
                for dt in dtypes:
                    base_configs.append(NeedleConfig(
                        context_len=cl,
                        needle_pos=needle_pos,
                        memory_type=mt,
                        dtype=dt,
                    ))

    # PTQ sweep
    quant_specs = [QuantSpec(bits=b, group_size=g) for b in quant_bits for g in quant_groups]
    quant_specs.append(QuantSpec(bits=4, group_size=64, symmetric=False))  # asymmetric probe
    all_configs = make_quant_configs(base_configs, quant_specs)

    logger.info(f'  {len(base_configs)} base configs x {len(quant_specs)} quant specs = {len(all_configs)} total')
    logger.info(f'  seeds: {seeds}, eval batch_size: {args.eval_batch_size}, '
                f'eval n_batches: {eval_batches}')
    if train_context_len is None:
        train_geom = 'one model per eval context (pos=ctx//4)'
    else:
        pos = train_needle_pos if train_needle_pos is not None else train_context_len // 4
        train_geom = f'ctx={train_context_len}, pos={pos}'
    logger.info(f'  training: every cell uses a model trained from scratch '
                f'({train_steps} steps, {train_geom}); untrained controls '
                f'reported alongside')

    # Pre-training fixed-batch gate: every memory route that will be swept
    # must first drive one fixed batch to a collapsing loss with finite
    # gradients. A broken gradient path would otherwise only surface as
    # random-level retrieval accuracy across the whole sweep.
    gate_record, proceed = _run_needle_overfit_gate(
        tokenizer=tokenizer, device=device,
        memory_types=memory_types or ['kda', 'mamba2', 'rwkv7'],
        lr=args.train_lr, include_stepwise=not args.no_qat,
        qat_spec=QuantSpec(bits=2, group_size=16))
    if not proceed:
        return 1

    # Run sweep
    needle_rows = run_needle_sweep(
        all_configs, tokenizer, seeds=seeds,
        batch_size=args.eval_batch_size, n_batches=eval_batches,
        device=device,
        train_context_len=train_context_len,
        train_needle_pos=train_needle_pos,
        train_steps=train_steps,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        val_every=args.val_every,
        val_batches=args.val_batches,
        max_error_examples=args.max_error_examples)

    # QAT probe
    qat_rows = []
    if not args.no_qat:
        qat_spec = QuantSpec(bits=2, group_size=16)
        qat_rows = run_qat_needle_probe(
            tokenizer, seeds=qat_seeds, qat_spec=qat_spec,
            steps=100 if not args.quick else 20,
            qat_steps=qat_steps,
            device=device)

    # Combine and write
    all_rows = needle_rows + qat_rows
    payload = sanitize_for_json({
        'experiment': 'exp12_needle_haystack',
        'design': {
            'question': 'how does long-context needle retrieval degrade with context length, needle position, memory operator, precision, and state quantization?',
            'task': 'needle-in-a-haystack: retrieve a fixed Q&A pair embedded in distractor text',
            'haystack': 'deterministic distractor sentences from fixed vocabulary',
            'needle': f'Q: "{_NEEDLE_QUESTION.strip()}" A: "{_NEEDLE_ANSWER.strip()}"',
            'eval_mode': 'next-token prediction at answer position after full prefill',
            'seeds': list(seeds),
            'quantization': 'PTQ sweep over bits x group_size + asymmetric probe; QAT with STE on stepwise KDA',
            'bf16_support': 'uses ops_kda_autograd bf16 stabilizers; CSA/HCA run in bf16 via fp32 LayerNorm',
            # Baseline ladder (Exp 4 convention): the simplest rung is a
            # weight-free rule; the untrained control is the chance floor;
            # every trained row is compared against BOTH.
            'baselines': {
                'rule_lexical_scan': (
                    'weight-free template match of the fixed answer prefix '
                    'followed by copying the continuation (no answer-text '
                    'oracle; the prefix comes from the known answer format)'),
                'rule_oracle_position': (
                    'copies from the ground-truth needle_pos metadata; a '
                    'task-level copy ceiling, not a deployable baseline'),
                'untrained_control': (
                    'same architecture/seed with random initialisation, '
                    'scored on the same held-out evaluation sequences and '
                    'through the same inference precision (dtype + PTQ spec); '
                    'the standalone control row is the full-precision case'),
                'chance': (
                    'uniform-guess exact match (1/vocab)^n_answer_tokens and '
                    'first-token chance 1/vocab, both recorded per row'),
            },
            'hypotheses': NEEDLE_HYPOTHESES,
            'hypothesis_verdicts': (
                'per-row hypothesis_checks (learned_retrieval, '
                'trained_matches_rule) + one needle_summary row aggregating '
                'them, including the farthest-vs-nearest distance effect'),
            # Data protocol: sequences are deterministic in their seed, so
            # train/test disjointness is enforced (and audited) at the seed
            # level. The QAT probe trains on (256, 64) geometry and scores
            # on the disjoint (512, 128) evaluation geometry.
            'data_protocol': {
                'generator': 'build_needle_haystack (deterministic per seed)',
                'eval_seed_band': [0, EVAL_SEED_RESERVE],
                'train_seed_band': [EVAL_SEED_RESERVE, SEED_CEILING],
                'train_seed_stream_offset': TRAIN_SEED_STREAM,
                'test_sealed': False,
                'note': ('The evaluation band is reserved: every training '
                         'step draws its seeds strictly above it, and the '
                         'QAT probe records an independent disjointness '
                         'audit per seed. A second held-out geometry '
                         '(higher context / later needle) is used for the '
                         'QAT probe so memorising the training contexts '
                         'does not transfer.'),
            },
        },
        'provenance': capture_provenance(),
        'rows': all_rows,
    })
    # Gate verdict next to the rows; full per-step curves in provenance.
    payload['overfit_sanity_check'] = {
        'enabled': gate_record['enabled'],
        'strict': gate_record.get('strict'),
        'steps': gate_record.get('steps'),
        'seed': gate_record.get('seed'),
        'train_batch': gate_record.get('train_batch'),
        'context_len': gate_record.get('context_len'),
        'needle_pos': gate_record.get('needle_pos'),
        'reports': [_compact_needle_gate_report(r)
                    for r in gate_record.get('reports', [])],
    }
    payload['provenance']['overfit_sanity_check'] = gate_record
    json.dumps(payload, allow_nan=False)
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    write_results_json(payload, args.json, logger=logger)
    logger.info(f'\nSaved: {args.json}')

    # Campaign-level ledger entry: one strict-JSON line linking this run's
    # seeds / recipe / data version / gate verdict to its artifact, so the
    # experiment is reproducible without hand-written notes.
    gate_failures = sum(1 for r in gate_record.get('reports', [])
                        if not r.get('passed'))
    manifest_seeds = list(dict.fromkeys(
        [*seeds, *(qat_seeds if not args.no_qat else [])]))
    append_run_manifest(make_run_manifest_entry(
        experiment='exp12_needle_haystack',
        seeds=manifest_seeds,
        provenance=payload['provenance'],
        hyperparameters={
            'eval_batch_size': int(args.eval_batch_size),
            'eval_batches': int(eval_batches),
            'train_steps': int(train_steps),
            'train_context_len': (None if train_context_len is None
                                  else int(train_context_len)),
            'train_needle_pos': (None if train_needle_pos is None
                                 else int(train_needle_pos)),
            'train_batch_size': int(args.train_batch_size),
            'train_lr': float(args.train_lr),
            'val_every': (None if args.val_every is None
                          else int(args.val_every)),
            'val_batches': int(args.val_batches),
            'max_error_examples': int(args.max_error_examples),
            'quick': bool(args.quick),
            'qat_enabled': not bool(args.no_qat),
            'qat_steps': int(qat_steps),
            'qat_seeds': list(qat_seeds),
            'context_lens': list(context_lens),
            'needle_ratios': list(needle_ratios),
            'memory_types': list(memory_types),
            'dtypes': list(dtypes),
            'quant_bits': list(quant_bits),
            'quant_groups': list(quant_groups),
        },
        data={
            'generator': 'build_needle_haystack (deterministic per seed)',
            'tokenizer': '_NeedleCharTokenizer',
            'vocab_size': int(tokenizer.vocab_size),
            'eval_seed_band': [0, EVAL_SEED_RESERVE],
            'train_seed_band': [EVAL_SEED_RESERVE, SEED_CEILING],
            'train_seed_stream_offset': TRAIN_SEED_STREAM,
            'n_base_configs': len(base_configs),
            'n_quant_configs': len(all_configs),
        },
        artifacts={'results': args.json},
        results={
            'n_rows': len(all_rows),
            'n_needle_rows': len(needle_rows),
            'n_qat_rows': len(qat_rows),
            'overfit_gate_enabled': bool(gate_record['enabled']),
            'overfit_gate_failures': int(gate_failures),
        },
        status=('completed' if gate_failures == 0 else 'partial'),
    ), logger=logger)

    if not args.quick:
        _print_needle_rows(needle_rows)
        if qat_rows:
            _print_qat_rows(qat_rows)

    return 0


if __name__ == '__main__':
    sys.exit(main())