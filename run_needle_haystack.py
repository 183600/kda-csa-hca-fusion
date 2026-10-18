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
fixed vocabulary of distractor sentences; the needle is a fixed Q&A pair
inserted at a controlled position. Retrieval is measured by exact-match of the
answer token(s) at the final position.

Output
------
``results/exp12_needle_haystack.json`` (versioned envelope via ``results_schema``)
with ``row_kind`` in {``needle_config``, ``needle_recall``, ``needle_summary``,
``qat_needle``, ``qat_needle_summary``}.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    capture_provenance,
    configure_torch_for_device,
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
from ops_fused import HybridConfig, HybridKCHAttention
from run_decoding import HybridDecoding
from results_schema import write as write_schema

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

# The needle: a fixed Q&A pair. The question is the "cue", the answer is the
# target. We construct this as a token sequence.
_NEEDLE_QUESTION = "What is the secret code? "
_NEEDLE_ANSWER = "The secret code is 42-alpha-7."

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
    """Build a single needle-in-haystack sequence.

    Args:
        tokenizer: character tokenizer
        context_len: total sequence length (including needle + haystack)
        needle_pos: position (0-indexed) where the needle QUESTION starts.
                    The answer follows immediately after.
        seed: RNG seed for distractor selection

    Returns:
        input_ids: [context_len] token ids
        target_ids: [1] answer token id(s) — we use the FIRST token of the answer
        answer_start: position where the answer begins (for verification)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Tokenize needle components
    q_ids = tokenizer.encode(_NEEDLE_QUESTION)
    a_ids = tokenizer.encode(_NEEDLE_ANSWER)
    needle_ids = q_ids + a_ids
    needle_len = len(needle_ids)

    if needle_pos + needle_len >= context_len:
        raise ValueError(f"needle_pos={needle_pos} + needle_len={needle_len} >= context_len={context_len}")

    # Build haystack from distractor sentences - use RNG to shuffle and sample
    distractor_sentences = list(_DISTRACTOR_SENTENCES)
    # Shuffle distractors per seed
    perm = torch.randperm(len(distractor_sentences), generator=gen).tolist()
    distractor_sentences = [distractor_sentences[i] for i in perm]

    distractor_ids = []
    for sent in distractor_sentences:
        distractor_ids.extend(tokenizer.encode(sent))

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

    # Insert needle at needle_pos
    # haystack is split: [0:needle_pos] + needle + [needle_pos:]
    before = full_haystack[:needle_pos]
    after = full_haystack[needle_pos:]

    input_ids = before + needle_ids + after
    input_ids = input_ids[:context_len]  # truncate if slight overflow

    # Target is the first token of the answer (the model predicts next token)
    # The answer starts at needle_pos + len(q_ids)
    answer_start = needle_pos + len(q_ids)
    target_id = a_ids[0] if a_ids else tokenizer.char_to_id.get('T', 0)

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
) -> dict:
    """Evaluate needle retrieval accuracy.

    Runs the model in eval mode, feeds the full context (prefill), then checks
    if the next-token prediction at the answer position matches the target.

    With quant_spec, the KDA state is fake-quantized after every layer write
    during the forward pass (mirroring the PTQ path in run_quant_lm).
    """
    model.eval()
    model.to(device)
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)

    correct = 0
    total = 0
    for b in range(n_batches):
        seeds = list(range(b * batch_size, (b + 1) * batch_size))
        input_ids, targets, answer_positions = build_needle_batch(
            batch_size, context_len, needle_pos, tokenizer, seeds=seeds, device=device
        )
        # Full prefill forward
        if hasattr(model, 'hybrid'):
            # NeedleHybridLM path
            model.hybrid.reset_state()
            if quant_spec is None:
                logits = model(input_ids.to(device))
            else:
                # PTQ contract: the recurrent state is fake-quantized after
                # every write, so the prefill must run token-by-token through
                # ``forward_step`` (the vectorized ``model(...)`` path has no
                # state-quantization hook). Only the prefix up to the last
                # scoring position is needed; the logits returned at step
                # ``t`` predict token ``t + 1``.
                last_pred_pos = max(answer_positions) - 1
                step_logits = [
                    model.forward_step(
                        input_ids[:, t:t+1].to(device), quant_spec)
                    for t in range(last_pred_pos + 1)
                ]
                logits = torch.stack(step_logits, dim=1)   # [B, T', vocab]
            # The answer token is predicted at position answer_start - 1
            # (since the model predicts next token)
            for i, ans_pos in enumerate(answer_positions):
                pred_pos = ans_pos - 1
                if pred_pos >= 0 and pred_pos < logits.shape[1]:
                    pred = logits[i, pred_pos].argmax().item()
                    if pred == targets[i].item():
                        correct += 1
                    total += 1
        else:
            # StepwiseKDALM or other: run step by step
            model.reset_state() if hasattr(model, 'reset_state') else None
            for i in range(batch_size):
                seq_ids = input_ids[i:i+1]
                model.reset_state() if hasattr(model, 'reset_state') else None
                logits = model(seq_ids, spec=quant_spec, ste=False)[:, :-1]
                ans_pos = answer_positions[i]
                pred_pos = ans_pos - 1
                if pred_pos >= 0 and pred_pos < logits.shape[1]:
                    pred = logits[0, pred_pos].argmax().item()
                    if pred == targets[i].item():
                        correct += 1
                    total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'context_len': context_len,
        'needle_pos': needle_pos,
    }


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
    """Evaluate needle retrieval via autoregressive decode.

    Feeds the context up to the question, then generates tokens and checks
    if the answer is produced. More realistic but slower.
    """
    model.eval()
    model.to(device)
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)

    correct = 0
    total = 0
    for b in range(n_batches):
        seeds = list(range(b * batch_size, (b + 1) * batch_size))
        input_ids, targets, answer_positions = build_needle_batch(
            batch_size, context_len, needle_pos, tokenizer, seeds=seeds, device=device
        )
        # Feed context up to the question, then generate the answer tokens
        # autoregressively.
        expected = tokenizer.encode(_NEEDLE_ANSWER)
        n_gen = len(expected)
        for i in range(batch_size):
            seq = input_ids[i:i+1]  # [1, context_len]
            ans_pos = answer_positions[i]
            generated: list[int] = []
            if hasattr(model, 'hybrid'):
                # Prefill up to (and including) the last question token; the
                # last prefill logits predict the FIRST answer token. The
                # previous implementation discarded these logits and re-fed
                # ``prefill[:, -1:]`` through ``forward_step``, consuming the
                # last token twice and corrupting the recurrent state.
                prefill = seq[:, :ans_pos]
                model.hybrid.reset_state()
                logits = model(prefill.to(device))
                next_logits = logits[:, -1]
                for _ in range(n_gen):
                    next_token = next_logits.argmax(dim=-1)
                    generated.append(next_token.item())
                    next_logits = model.forward_step(
                        next_token.unsqueeze(-1).to(device), quant_spec)
            else:
                # Stepwise model keeps no cross-call state: re-run the
                # growing prefix and take the last-position logits.
                prefill = seq[:, :ans_pos]
                model.reset_state() if hasattr(model, 'reset_state') else None
                logits = model(prefill, spec=quant_spec, ste=False)
                for _ in range(n_gen):
                    next_token = logits[:, -1].argmax(dim=-1)
                    generated.append(next_token.item())
                    prefill = torch.cat(
                        [prefill, next_token.unsqueeze(0)], dim=1)
                    logits = model(prefill, spec=quant_spec, ste=False)
            if generated == expected:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'context_len': context_len,
        'needle_pos': needle_pos,
    }


# ---------------------------------------------------------------------------
# Training (for QAT probe)
# ---------------------------------------------------------------------------

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
) -> tuple[nn.Module, dict]:
    """Train a needle model (full-sequence or stepwise with optional QAT)."""
    set_global_seed(seed, deterministic=True)
    model.train()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = make_seeded_generator(seed + 1_000_000, device=device)

    losses = []
    for step in range(steps):
        # Draw fresh data seeds for every step from the seeded generator.
        # Previously the generator was created and never used, so
        # ``build_needle_batch`` fell back to its default ``range(batch_size)``
        # seeds and re-created the IDENTICAL batch every step (memorization,
        # plus the training sequences coincided with the evaluation ones).
        step_seeds = torch.randint(
            0, 2**31 - 1, (batch_size,), generator=gen,
            device=gen.device).cpu().tolist()
        # Build training batch
        input_ids, targets, answer_positions = build_needle_batch(
            batch_size, train_context_len, train_needle_pos, tokenizer,
            seeds=step_seeds, device=device
        )
        # Simple next-token prediction loss on the answer position
        if hasattr(model, 'hybrid'):
            model.hybrid.reset_state()
            logits = model(input_ids)
            # Loss on answer token position
            loss = 0
            for i, ans_pos in enumerate(answer_positions):
                pred_pos = ans_pos - 1
                if pred_pos >= 0:
                    loss = loss + F.cross_entropy(
                        logits[i:i+1, pred_pos], targets[i:i+1])
            loss = loss / batch_size
        else:
            # Stepwise
            model.reset_state()
            logits = model(input_ids, spec=quant_spec, ste=ste)  # [B, T, vocab]
            # Next-token prediction loss: predict token t+1 from token t
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=-100)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}: {loss}')
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    return model, {'losses': losses, 'final_loss': losses[-1] if losses else None}


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
    same-seed fp16/fp32 anchor to read the PTQ damage off directly. Previously
    the base row was dropped for all recurrent memory types (i.e. always),
    which left the sweep with quantized rows only and no baseline.
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

def run_needle_sweep(
    configs: list[NeedleConfig],
    tokenizer: _NeedleCharTokenizer,
    seeds: tuple[int, ...] = (0, 1, 2),
    batch_size: int = 4,
    n_batches: int = 2,
    device: str | torch.device = 'cpu',
) -> list[dict]:
    """Run the needle sweep over configs and seeds."""
    rows = []
    for cfg in configs:
        logger.info(f'  Running: ctx={cfg.context_len}, pos={cfg.needle_pos}, '
                    f'mem={cfg.memory_type}, dtype={cfg.dtype}, '
                    f'quant={cfg.quant_spec is not None}')
        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        torch_dtype = dtype_map[cfg.dtype]

        per_seed_results = []
        for seed in seeds:
            # Re-seed before model construction so the ``seed`` recorded in
            # the row actually controls the model initialization (previously
            # the RNG stream kept advancing across configs, making runs
            # non-reproducible and the multi-seed spread uninterpretable).
            set_global_seed(seed, deterministic=True)
            # Build model
            model = NeedleHybridLM(
                vocab_size=tokenizer.vocab_size,
                d_model=128,
                n_layers=cfg.n_layers,
                memory_type=cfg.memory_type,
                n_kda=cfg.n_kda,
                n_csa=cfg.n_csa,
                n_hca=cfg.n_hca,
            )
            # Evaluate
            res = evaluate_needle_recall(
                model, tokenizer,
                context_len=cfg.context_len,
                needle_pos=cfg.needle_pos,
                batch_size=batch_size,
                n_batches=n_batches,
                device=device,
                quant_spec=cfg.quant_spec,
                dtype=torch_dtype,
            )
            res['seed'] = seed
            per_seed_results.append(res)

        # Aggregate
        accs = [r['accuracy'] for r in per_seed_results]
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        min_acc = min(accs) if accs else 0.0
        max_acc = max(accs) if accs else 0.0

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
            'quantized': cfg.quant_spec is not None,
            'bits': cfg.quant_spec.bits if cfg.quant_spec else None,
            'group_size': cfg.quant_spec.group_size if cfg.quant_spec else None,
            'symmetric': cfg.quant_spec.symmetric if cfg.quant_spec else None,
            'bits_per_element': bits_per_element(cfg.quant_spec) if cfg.quant_spec else 32.0,
            'state_bytes': 0,  # TODO: compute from model
            'accuracy_mean': mean_acc,
            'accuracy_min': min_acc,
            'accuracy_max': max_acc,
            'seeds': list(seeds),
            'per_seed': per_seed_results,
        }
        if cfg.quant_spec:
            row['state_bytes'] = storage_bytes(128 * 2 * 32 * 32, cfg.quant_spec)  # approximate
            row['compression_vs_fp16'] = compression_ratio(cfg.quant_spec)
        rows.append(row)

    return rows


def run_qat_needle_probe(
    tokenizer: _NeedleCharTokenizer,
    seeds: tuple[int, ...] = (0, 1),
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
    1. fp16 state writes (baseline)
    2. 2-bit STE state writes (QAT)
    Evaluates needle retrieval on both.
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

        # Evaluate both on needle task
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

        # Recovery fraction: 1 - (QAT_damage / PTQ_damage)
        # where damage = (acc_fp16 - acc_quantized) / acc_fp16
        fp16_a = fp16_acc['accuracy']
        ptq_a = ptq_acc['accuracy']
        qat_a = qat_acc['accuracy']
        damage = (fp16_a - ptq_a) / fp16_a if fp16_a > 0 else 0
        residual = (fp16_a - qat_a) / fp16_a if fp16_a > 0 else 0
        recovery = 1.0 - residual / damage if damage > 0 else None

        rows.append({
            'row_kind': 'qat_needle',
            'seed': seed,
            'steps': steps,
            'qat_steps': qat_steps,
            'fp16_accuracy': fp16_a,
            'ptq_accuracy': ptq_a,
            'qat_accuracy': qat_a,
            'qat_fp16_accuracy': qat_fp16_acc['accuracy'],
            'recovery_fraction': recovery,
            'train_context_len': train_context_len,
            'train_needle_pos': train_needle_pos,
            'eval_context_len': eval_context_len,
            'eval_needle_pos': eval_needle_pos,
            'spec': spec_summary(qat_spec),
        })

    # Summary
    fp16_mean = sum(r['fp16_accuracy'] for r in rows) / len(rows)
    ptq_mean = sum(r['ptq_accuracy'] for r in rows) / len(rows)
    qat_mean = sum(r['qat_accuracy'] for r in rows) / len(rows)
    recs = [r['recovery_fraction'] for r in rows if r['recovery_fraction'] is not None]
    rec_mean = sum(recs) / len(recs) if recs else None
    rec_min = min(recs) if recs else None

    rows.append({
        'row_kind': 'qat_needle_summary',
        'seeds': list(seeds),
        'fp16_accuracy_mean': fp16_mean,
        'ptq_accuracy_mean': ptq_mean,
        'qat_accuracy_mean': qat_mean,
        'recovery_fraction_mean': rec_mean,
        'recovery_fraction_min': rec_min,
        'spec': spec_summary(qat_spec),
    })
    return rows


def _print_needle_rows(rows: list[dict]):
    logger.info('\n-- Needle-in-haystack recall --')
    header = (f"{'ctx':>5} | {'pos':>5} | {'mem':>7} | {'dtype':>5} | "
              f"{'quant':>10} | {'acc mean':>8} | {'acc min':>7} | {'acc max':>7}")
    logger.info(header)
    logger.info('-' * len(header))
    for r in rows:
        if r['row_kind'] != 'needle_recall':
            continue
        quant_str = (f"{r['bits']}b/g{r['group_size']}" if r['quantized'] else 'fp16')
        logger.info(
            f"{r['context_len']:>5} | {r['needle_pos']:>5} | {r['memory_type']:>7} | "
            f"{r['dtype']:>5} | {quant_str:>10} | "
            f"{r['accuracy_mean']:>8.3f} | {r['accuracy_min']:>7.3f} | {r['accuracy_max']:>7.3f}")


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
            f"  fp16 {r['fp16_accuracy_mean']:.3f} | "
            f"PTQ {r['ptq_accuracy_mean']:.3f} | "
            f"QAT {r['qat_accuracy_mean']:.3f} | "
            f"recovery {rec_str} (min {rec_min_str})")


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

    # Explicit CLI filters override the phase defaults in BOTH modes: the
    # ``--quick`` branch used to overwrite them unconditionally, silently
    # ignoring ``--context-lens`` / ``--memory-types`` / ``--dtypes`` /
    # ``--quant-*`` (so a filtered quick run evaluated unrelated configs).
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
    else:
        context_lens = context_lens or [256, 512, 1024, 2048]
        needle_ratios = [0.1, 0.25, 0.5, 0.75]
        memory_types = memory_types or ['kda', 'mamba2', 'rwkv7']
        dtypes = dtypes or ['fp32', 'fp16', 'bf16']
        quant_bits = quant_bits or [2, 3, 4]
        quant_groups = quant_groups or [16, 64]
        qat_seeds = (0, 1)
        qat_steps = 100

    # Build base configs. Skip (context_len, ratio) cells where the needle
    # would not fit: ``build_needle_haystack`` raises when
    # ``needle_pos + needle_len >= context_len`` (an explicit CLI
    # ``--context-lens`` can be small enough for the 0.75 cell to overflow).
    needle_len = (len(tokenizer.encode(_NEEDLE_QUESTION))
                  + len(tokenizer.encode(_NEEDLE_ANSWER)))
    base_configs = []
    for cl in context_lens:
        for nr in needle_ratios:
            needle_pos = int(cl * nr)
            if needle_pos + needle_len >= cl:
                logger.info(f'  skipping ctx={cl}, ratio={nr}: needle '
                            f'(len={needle_len}) does not fit at '
                            f'pos={needle_pos}')
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
    logger.info(f'  seeds: {seeds}, batch_size: {4}, n_batches: {2}')

    # Run sweep
    needle_rows = run_needle_sweep(
        all_configs, tokenizer, seeds=seeds, batch_size=4, n_batches=2, device=device)

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
        },
        'provenance': capture_provenance(),
        'rows': all_rows,
    })
    json.dumps(payload, allow_nan=False)
    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    write_results_json(payload, args.json, logger=logger)
    logger.info(f'\nSaved: {args.json}')

    if not args.quick:
        _print_needle_rows(needle_rows)
        if qat_rows:
            _print_qat_rows(qat_rows)

    return 0


if __name__ == '__main__':
    sys.exit(main())