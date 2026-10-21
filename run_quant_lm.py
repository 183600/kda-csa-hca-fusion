"""Experiment 11 — state-quantized decode perplexity + STE/QAT recovery.

Motivation
----------
Experiment 10 (``run_quant_recall``) measures intrinsic damage: with the
KDA/Mamba-2 state fake-quantized after every write, how often can the
route still decode a planted association? That probe uses planted
orthonormal keys and no training, so it says nothing about the quantity
language-model deployment is actually scored on: **held-out perplexity**.

This experiment closes the loop with an end-task **perplexity evaluation**
and a QAT (**autograd / straight-through**) recovery probe:

* **Part A — PTQ (post-training quantization) perplexity.** A tiny hybrid
  LM (``train_lm_autodl.LMWithHybrid``: 2 KDA + 1 CSA + 1 HCA, the repo's
  canonical layout, chunked-KDA prefill) is trained on the deterministic
  offline synthetic fallback corpus. Held-out token-level NLL is then
  scored through the REAL incremental decode path
  (``run_decoding.HybridDecoding``, one token per call), with the stacked
  KDA state fake-quantized after every write. The spec grid is exactly
  Exp 10's grid, so the two experiments share a byte axis.
* **Part B — QAT recovery.** A standalone single-layer KDA LM with an
  explicit per-token state loop (``run_quality.KDAAttn``'s parameterization,
  recurrence called through ``ops_kda.naive_recurrent_kda`` with
  ``initial_state``) is trained twice from the SAME initialization: once
  with fp16 state writes, once with 2-bit state writes through the
  straight-through estimator (``kv_quant.fake_quantize(..., ste=True)``).
  Scoring the fp16 model with PTQ state vs scoring the QAT model shows
  whether autograd through the quantizer recovers the perplexity damage —
  i.e. that the STE contract is usable for QAT, not just formula-correct.
  If ``QLM_STEPS != QLM_QAT_STEPS``, a third fp16 model is trained for
  exactly ``QLM_QAT_STEPS`` from the same seed/init/data stream so the
  primary QAT contrast isolates the quantizer instead of the extra
  optimization budget.

Reference points are reported, not hidden:

* ``prefill_ppl`` is the chunked full-sequence path
  (``train_lm_autodl.evaluate_lm_loss``); ``decode_fp16_ppl`` is the same
  model scored token-by-token with an fp16 state. The two paths agree only
  up to kernel/reduction order (this is the ``lm_consistency`` contract),
  so every quantization delta is measured **against decode_fp16_ppl** on
  the same path. The prefill/decode gap is recorded per seed.
* A cell is flagged ``ppl_preserving`` when the min-over-seeds ratio
  ``decode_ppl / decode_fp16_ppl`` stays within ``QLM_PPL_TOL`` (default
  1%); the tolerance is explicit and conservative (min over seeds), not
  an "exact" claim.

Honesty / scope
---------------
* Corpus: the repo's deterministic ``_synthetic_corpus`` (no network, no
  HuggingFace tokenizer). Absolute perplexities are toy-scale and NOT
  comparable to real-corpus LM numbers; the measured quantity is the
  RELATIVE damage of state quantization at fixed weights and path.
* Only the KDA recurrent state is quantized. The CSA/HCA incremental
  caches, the KDA short-conv lookback, activations, weights and logits
  are fp16/fp32; the byte axis prices only the quantized state (see
  ``kv_budget`` for the full component split).
* Grouping runs along a per-layer flattened state buffer
  (``state.reshape(n_layers, -1)``), the same contiguous-buffer convention
  as Exp 10.
* Part B's QAT trains from scratch (no pre-trained checkpoint), 1 layer,
  and on the synthetic corpus: it demonstrates the autograd/STE contract
  end-to-end, not a production QAT recipe. QAT does not always fully
  recover the PTQ damage (seed variance is reported).

Output
------
``results/exp11_quant_lm.json`` (versioned envelope via ``results_schema``)
with ``row_kind`` in {``ptq_per_seed``, ``ptq_ppl``, ``ptq_summary``,
``qat_per_seed``, ``qat_recovery``}; CLI::

    python3 run_quant_lm.py
    QLM_SEEDS=0 QLM_STEPS=50 QLM_QAT_STEPS=50 python3 run_quant_lm.py
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    append_run_manifest,
    capture_provenance,
    configure_torch_for_device,
    grad_flow_report,
    make_run_manifest_entry,
    make_seeded_generator,
    parse_int_env,
    seed_worker,
    set_global_seed,
)
from kv_quant import (  # noqa: E402
    FP16_BITS,
    QuantSpec,
    bits_per_element,
    compression_ratio,
    fake_quantize,
    spec_summary,
    storage_bytes,
)
from lm_consistency import build_decoding_clone  # noqa: E402
from multi_seed_stats import summarize_paired  # noqa: E402
from ops_fused import HybridConfig  # noqa: E402
from ppl_bootstrap import paired_bootstrap_ppl  # noqa: E402
from ops_kda import naive_recurrent_kda  # noqa: E402
from results_analysis import (  # noqa: E402
    LM_ERROR_CATEGORIES,
    lm_token_diagnostics,
    summarize_lm_diagnostics,
)
from results_schema import write as write_schema  # noqa: E402
from train_diagnostics import (  # noqa: E402
    ValPoint,
    ValidationCurve,
    raise_non_finite,
    triage_gradient,
)
from run_quant_recall import (  # noqa: E402
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZES,
    default_sweep_specs,
)
from run_quality import KDAAttn  # noqa: E402
from train_lm_autodl import (  # noqa: E402
    LMWithHybrid,
    TinyStoriesLM,
    _FallbackCharTokenizer,
    _synthetic_corpus,
    _texts_fingerprint,
    _distribution_shift,
    _filter_leaking_heldout,
    _length_outlier_audit,
    _probe_token_lengths,
    _text_length_stats,
    _window_overlap_audit,
    clean_lm_texts,
    analyze_lm_errors,
    evaluate_lm_loss,
    run_overfit_sanity_check as run_lm_overfit_sanity_check,
    split_clean_texts,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp11_quant_lm.json')

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 400
# The fp16 control and the QAT model MUST see the same training budget for
# the A/B to be an isolation of the quantizer; keep the two knobs equal by
# default (QLM_QAT_STEPS exists only so a smoke run can shrink both).
DEFAULT_QAT_STEPS = DEFAULT_STEPS
DEFAULT_SEQ_LEN = 64
DEFAULT_BATCH = 8
DEFAULT_EVAL_SEQS = 13
DEFAULT_CORPUS = 64
DEFAULT_TOLERANCE = 0.01

# 2-bit/g16 is where Exp 10 loses recall for KDA; it is the PTQ cell QAT
# is asked to recover.
QAT_SPEC = QuantSpec(bits=2, group_size=16)


# --------------------------------------------------------------------------
# Data + model construction (deterministic, offline)
# --------------------------------------------------------------------------
def build_corpus(seq_len: int = DEFAULT_SEQ_LEN,
                 n_texts: int = DEFAULT_CORPUS,
                 seed: int = 0) -> tuple:
    """Deterministic offline corpus -> (tokenizer, split_texts, split_report).

    Uses the repository's synthetic fallback corpus and char tokenizer, so
    the experiment needs no network. Cleaning + the leakage-audited
    train/val/test split are the same code paths the real LM pipeline uses.
    """
    tokenizer = _FallbackCharTokenizer()
    cleaning = clean_lm_texts(_synthetic_corpus(n_texts), tokenizer,
                              seq_len)
    clean = cleaning['texts']
    split, report = split_clean_texts(clean, val_frac=0.2, test_frac=0.1,
                                      seed=seed)
    split_texts = {name: [clean[i] for i in split[name]]
                   for name in ('train', 'val', 'test')}

    # Last line of defence: a held-out row whose truncated (seq_len+1)-token
    # model input already occurs in TRAIN is a sample the model saw during
    # training, even when the raw strings differ. Filter those rows BEFORE
    # any evaluation; the audit below records whether any residual overlap
    # remains (a fallback keeps a split whole rather than emptying it).
    leak_filter = _filter_leaking_heldout(
        split_texts['train'], split_texts['val'], split_texts['test'],
        tokenizer, seq_len)
    split_texts['val'] = leak_filter['val_texts']
    split_texts['test'] = leak_filter['test_texts']
    window_overlap = _window_overlap_audit(
        split_texts['train'], split_texts['val'], split_texts['test'],
        tokenizer, seq_len)

    # Distribution diagnostics are fitted on TRAIN only and merely
    # reported for val/test: text length is not a model feature, so acting
    # on these fences would change the evaluation population.
    split_stats = {
        name: _text_length_stats(texts, tokenizer, seq_len)
        for name, texts in split_texts.items()}
    train_lengths = _probe_token_lengths(
        split_texts['train'], tokenizer, seq_len)
    diagnostics = {
        'fit_on': 'train',
        'split_stats': split_stats,
        'length_outlier_audit': {
            name: _length_outlier_audit(
                train_lengths,
                _probe_token_lengths(split_texts[name], tokenizer, seq_len))
            for name in ('val', 'test')},
        'distribution_shift': {
            name: _distribution_shift(split_stats['train'],
                                      split_stats[name])
            for name in ('val', 'test')},
        'note': ('Token-length outlier fences and Cohen\'s d are fitted on '
                 'TRAIN and reported for val/test; they are shift warnings, '
                 'never transforms or selection criteria.'),
    }
    report['cleaning'] = cleaning['stats']
    report['window_leak_filter'] = {
        k: v for k, v in leak_filter.items()
        if k not in ('val_texts', 'test_texts')}
    report['window_overlap_audit'] = window_overlap
    report['distribution_diagnostics'] = diagnostics
    return tokenizer, split_texts, report


def small_hybrid_config() -> HybridConfig:
    """The tiny 2 KDA + 1 CSA + 1 HCA geometry used by Part A.

    ``kda_chunk_size=32`` keeps the chunked prefill path active at
    ``T=64`` while per-token decode automatically uses the recurrent path
    (``T < kda_chunk_size``), so the prefill/decode boundary is exercised.
    """
    return HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        kda_chunk_size=32,
        csa_m=8, csa_topk=2, csa_nh=2, csa_c=16, csa_dc=16,
        csa_nIh=1, csa_cI=8, csa_sliding_window=4,
        hca_m2=16, hca_nh=2, hca_c=16, hca_dc=16, hca_sliding_window=4,
        n_kda=2, n_csa=1, n_hca=1,
    )


def _cfg_summary(cfg: HybridConfig) -> dict:
    fields = ('d_model', 'n_heads_qk', 'n_heads_v', 'head_dim_k',
              'head_dim_v', 'kda_chunk_size', 'csa_m', 'csa_topk',
              'hca_m2', 'n_kda', 'n_csa', 'n_hca', 'memory_type')
    return {name: getattr(cfg, name) for name in fields}


def encode_windows(tokenizer, texts, seq_len: int, device) -> list:
    """Right-pad each text to ``seq_len + 1`` tokens (for NLL scoring)."""
    pad = getattr(tokenizer, 'pad_token_id', 0)
    windows = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=seq_len + 1)
        ids = ids + [pad] * (seq_len + 1 - len(ids))
        windows.append(torch.tensor([ids[:seq_len + 1]], dtype=torch.long,
                                    device=device))
    return windows


def pad_token_id(tokenizer) -> int:
    return int(getattr(tokenizer, 'pad_token_id', 0) or 0)


# --------------------------------------------------------------------------
# Training (deterministic; full train split, seeded loader)
# --------------------------------------------------------------------------
def _make_loader(tokenizer, texts, seq_len, batch_size, seed):
    dataset = TinyStoriesLM(tokenizer, seq_len=seq_len, texts=texts,
                            name='train')
    return torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True,
        generator=make_seeded_generator(int(seed)),
        worker_init_fn=seed_worker)


def train_hybrid_lm(*, seed: int, tokenizer, train_texts, steps: int,
                    seq_len: int, batch_size: int, device, lr: float = 3e-3,
                    val_every: int = 0, val_monitor=None) -> tuple:
    """Train the tiny hybrid LM for ``steps`` steps; deterministic per seed.

    ``val_every > 0`` with a ``val_monitor(model) -> ValPoint`` records
    a held-out (loss, token-accuracy) curve. The monitor is read-only and
    RNG-neutral, so enabling it cannot change the trained weights.
    """
    set_global_seed(seed, deterministic=True)
    cfg = small_hybrid_config()
    model = LMWithHybrid(len(tokenizer), cfg).to(device)
    loader = _make_loader(tokenizer, train_texts, seq_len, batch_size, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=0.0)
    first_loss = last_loss = None
    losses: list[float] = []
    model.train()
    val_curve = None
    if int(val_every or 0) > 0 and val_monitor is not None:
        val_curve = ValidationCurve(
            lambda _step: val_monitor(model), steps=int(steps),
            val_every=int(val_every), label=f'Part A seed={seed}',
            ceiling=1.0)
        val_curve.record(0)
    step = 0
    while step < int(steps):
        for batch in loader:
            if step >= int(steps):
                break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            model.hybrid.reset_state()
            out = model(input_ids, labels=labels)
            loss = out['loss']
            if loss is None or not torch.isfinite(loss):
                # Same staged triage as the other trainers: token ids ->
                # loss -> logits -> parameters. A bare "non-finite loss"
                # names no cause, so a single bad seed costs a debugging
                # cycle instead of pointing at the first failing stage.
                raise_non_finite(
                    'loss', loss=(loss if loss is not None
                                  else float('nan')),
                    token_ids={'input_ids': input_ids},
                    id_range=(0, len(tokenizer)),
                    params=model.parameters(),
                    logits=out.get('logits'),
                    context=f'train_hybrid_lm seed={seed} step={step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad_grads = triage_gradient(model.parameters())
            if bad_grads:
                raise_non_finite(
                    'gradient', loss=loss,
                    token_ids={'input_ids': input_ids},
                    id_range=(0, len(tokenizer)),
                    params=model.parameters(),
                    grads={f'grad[{i}]': p.grad
                           for i, p in enumerate(bad_grads)},
                    context=f'train_hybrid_lm seed={seed} step={step}')
            opt.step()
            last_loss = float(loss.detach())
            losses.append(last_loss)
            if first_loss is None:
                first_loss = last_loss
            step += 1
            if val_curve is not None and step % int(val_every) == 0:
                val_curve.record(step)
    report = {'cfg': cfg, 'loss_first': first_loss,
              'loss_last': last_loss, 'steps': int(steps), 'losses': losses}
    if val_curve is not None:
        block = val_curve.finalize(losses)
        report.update({
            'val_every': int(val_every),
            'validation': block,
            'curve_diagnosis': block['curve_diagnosis'],
            'generalization_gap': block['generalization_gap'],
            'early_stopping_summary': block.get('early_stopping_summary'),
        })
        val_curve.log_verdict(block)
    return model, report


def train_stepwise_lm(*, seed: int, tokenizer, train_texts, steps: int,
                      seq_len: int, batch_size: int, device,
                      spec: QuantSpec | None = None, ste: bool = False,
                      lr: float = 3e-3, val_every: int = 0,
                      val_monitor=None) -> tuple:
    """Train the standalone per-token KDA LM, optionally QAT (STE) at ``spec``.

    ``val_every > 0`` with a ``val_monitor(model) -> ValPoint`` records
    the held-out curve, so a QAT/PTQ arm that is still improving is not read
    as converged just because the final loss happened to be lower.
    """
    set_global_seed(seed, deterministic=True)
    model = StepwiseKDALM(len(tokenizer)).to(device)
    loader = _make_loader(tokenizer, train_texts, seq_len, batch_size, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr),
                            weight_decay=0.0)
    report = {'loss_first': None, 'loss_last': None,
              'grad_norm_first': None, 'grad_norm_last': None,
              'all_grads_finite': True, 'steps': int(steps),
              'losses': []}
    model.train()
    val_curve = None
    if int(val_every or 0) > 0 and val_monitor is not None:
        val_curve = ValidationCurve(
            lambda _step: val_monitor(model), steps=int(steps),
            val_every=int(val_every), label=f'Part B seed={seed}',
            ceiling=1.0)
        val_curve.record(0)
    step = 0
    while step < int(steps):
        for batch in loader:
            if step >= int(steps):
                break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, spec=spec, ste=ste)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=-100)
            if loss is None or not torch.isfinite(loss):
                raise_non_finite(
                    'loss', loss=(loss if loss is not None
                                  else float('nan')),
                    token_ids={'input_ids': input_ids},
                    id_range=(0, len(tokenizer)),
                    params=model.parameters(), logits=logits,
                    context=f'train_stepwise_lm seed={seed} step={step}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad_grads = triage_gradient(model.parameters())
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.detach().pow(2).sum())
            grad_norm = math.sqrt(grad_norm)
            if bad_grads or not math.isfinite(grad_norm):
                raise_non_finite(
                    'gradient', loss=loss,
                    token_ids={'input_ids': input_ids},
                    id_range=(0, len(tokenizer)),
                    params=model.parameters(), logits=logits,
                    grads={f'grad[{i}]': p.grad
                           for i, p in enumerate(bad_grads)},
                    grad_norm=grad_norm,
                    context=f'train_stepwise_lm seed={seed} step={step}')
            report['all_grads_finite'] = True
            opt.step()
            report['loss_last'] = float(loss.detach())
            report['losses'].append(report['loss_last'])
            report['grad_norm_last'] = grad_norm
            if report['loss_first'] is None:
                report['loss_first'] = report['loss_last']
                report['grad_norm_first'] = grad_norm
            step += 1
            if val_curve is not None and step % int(val_every) == 0:
                val_curve.record(step)
    if val_curve is not None:
        block = val_curve.finalize(report['losses'])
        report.update({
            'val_every': int(val_every),
            'validation': block,
            'curve_diagnosis': block['curve_diagnosis'],
            'generalization_gap': block['generalization_gap'],
            'early_stopping_summary': block.get('early_stopping_summary'),
        })
        val_curve.log_verdict(block)
    return model, report


class StepwiseKDALM(KDAAttn):
    """Single-layer KDA LM with an explicit per-token recurrent state loop.

    Reuses ``run_quality.KDAAttn``'s parameterization (projections, low-rank
    gate, short conv, output projection) but drives the delta-rule
    recurrence one token at a time through ``naive_recurrent_kda`` with
    ``initial_state``, so the state is VISIBLE and can be fake-quantized
    after every write — exactly the write path Exp 10 probes, but wrapped
    in a differentiable LM loss for QAT.
    """

    def __init__(self, vocab_size: int, d_model: int = 32, H: int = 2,
                 K: int = 16, V: int = 16):
        super().__init__(d_model, H=H, K=K, V=V)
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def quantize_state(state: torch.Tensor, spec: QuantSpec,
                       ste: bool) -> torch.Tensor:
        """Grouped fake-quant of one step's state (flat buffer per batch row)."""
        flat = state.reshape(state.shape[0], -1)
        return fake_quantize(flat, spec, ste=ste).reshape_as(state)

    def forward(self, input_ids: torch.Tensor,
                spec: QuantSpec | None = None,
                ste: bool = False) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f'StepwiseKDALM: expected [B, T] input_ids, got '
                f'{tuple(input_ids.shape)}')
        B, T = input_ids.shape
        x = self.embed(input_ids)
        ksize = self.short_conv.kernel_size[0]
        x_conv = self.short_conv(
            F.pad(x.transpose(1, 2), (ksize - 1, 0))).transpose(1, 2)
        q = F.normalize(F.silu(self.q(x_conv)).view(B, T, self.H, self.K),
                        dim=-1)
        k = F.normalize(F.silu(self.k(x_conv)).view(B, T, self.H, self.K),
                        dim=-1)
        v = F.silu(self.v(x_conv)).view(B, T, self.H, self.V)
        g = -F.softplus(
            self.g_up(F.silu(self.g_down(x_conv))).view(B, T, self.H, self.K)
        ) * self.DECAY_SCALE
        beta = torch.sigmoid(self.beta(x_conv)).view(B, T, self.H)
        state = None
        outs = []
        for t in range(T):
            o, state = naive_recurrent_kda(
                q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                g[:, t:t + 1], beta[:, t:t + 1],
                initial_state=state, output_final_state=True)
            if spec is not None:
                state = self.quantize_state(state, spec, ste)
            outs.append(self.o(o[:, 0].reshape(B, -1)))
        return self.lm_head(torch.stack(outs, dim=1))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _ppl_from_nll(total_nll: float, n_tokens: int) -> float:
    if n_tokens <= 0:
        raise ValueError('no supervised tokens: cannot compute perplexity')
    return math.exp(min(total_nll / n_tokens, 700.0))


def quantize_stacked_state(clone, spec: QuantSpec | None) -> tuple:
    """Fake-quantize ``clone.model._kda_state`` in place; return (state, snr).

    ``state`` is the (possibly quantized) stacked state and ``snr`` the
    signal-to-quantization-noise ratio of this write (dB), or ``inf`` when
    no quantization was requested. Groups run along the flattened per-layer
    buffer, mirroring Exp 10.
    """
    state = clone.model._kda_state
    if state is None:
        raise RuntimeError(
            'quantize_stacked_state: the decode clone has no KDA state yet; '
            'run at least one forward step first')
    if spec is None:
        return state, math.inf
    flat = state.reshape(state.shape[0], -1)
    deq = fake_quantize(flat, spec, ste=False)
    err = deq - flat
    snr_num = float(flat.detach().pow(2).sum())
    snr_den = float(err.pow(2).sum())
    snr = (10.0 * math.log10(snr_num / snr_den)
           if snr_den > 0.0 and snr_num > 0.0 else math.inf)
    clone.model._kda_state = deq.reshape_as(state)
    return clone.model._kda_state, snr


@torch.no_grad()
def evaluate_decode_ppl(model, clone, windows: list, pad_id: int,
                        spec: QuantSpec | None = None,
                        collect_diagnostics: bool = False,
                        top_k: int = 5, confidence: float = 0.5) -> dict:
    """Held-out NLL/perplexity through the incremental decode path.

    Scores token-by-token from an empty state; after every write the
    stacked KDA state is fake-quantized at ``spec`` (``None`` = fp16
    baseline). Returns the perplexity, token-weighted mean NLL, token
    count, per-position argmax (for top-1 agreement), the state element
    count, and ``per_window`` — the ``(nll_sum, n_tokens)`` sufficient
    statistic per scored window, in window order, so any two specs on the
    SAME windows can be compared with the paired unit-level bootstrap
    (``ppl_bootstrap.paired_bootstrap_ppl``, feeding ``build_ptq_rows``).
    State-write SNR is measured separately by :func:`_decode_state_snr`
    so this scoring path stays lean.
    """
    model.eval()
    total_nll, n_tokens = 0.0, 0
    argmaxes = []
    per_window = []
    raw_diagnostics = []
    state_elements = None
    for ids in windows:
        clone.reset()
        x = model.embed(ids)
        T = ids.shape[1]
        step_logits = []
        for t in range(T - 1):
            h = clone(x[:, t:t + 1])
            state, _ = quantize_stacked_state(clone, spec)
            if state_elements is None:
                state_elements = int(state.numel())
            elif state_elements != int(state.numel()):
                raise AssertionError(
                    'state element count changed across decode steps: '
                    f'{state_elements} vs {state.numel()}')
            step_logits.append(model.lm_head(model.norm_f(h)))
        logits = torch.cat(step_logits, dim=1)  # [1, T-1, V]
        targets = ids[:, 1:1 + logits.shape[1]]
        mask = targets.ne(pad_id) & targets.ne(-100)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            reduction='none')
        window_nll_sum = float((nll * mask.reshape(-1)).sum())
        window_n_tokens = int(mask.sum())
        total_nll += window_nll_sum
        n_tokens += window_n_tokens
        per_window.append({'nll_sum': window_nll_sum,
                           'n_tokens': window_n_tokens})
        argmaxes.append(logits.argmax(-1)[mask])
        if collect_diagnostics:
            # Same logits already materialized for NLL: classify each token
            # by what the model failed to use. ``-100`` on padded targets
            # keeps the diagnostic denominator identical to the PPL one.
            diag_labels = torch.where(mask, targets,
                                      torch.full_like(targets, -100))
            with torch.no_grad():
                raw_diagnostics.append(lm_token_diagnostics(
                    logits.detach(), diag_labels, ids[:, :-1],
                    top_k=top_k, confidence=confidence))
    out = {
        'ppl': _ppl_from_nll(total_nll, n_tokens),
        'mean_nll': total_nll / max(1, n_tokens),
        'n_tokens': n_tokens,
        'argmax': argmaxes,
        'state_elements': state_elements,
        'per_window': per_window,
    }
    if collect_diagnostics:
        out['token_diagnostics'] = summarize_lm_diagnostics(raw_diagnostics)
        out['raw_diagnostics'] = raw_diagnostics
    return out


def quantization_error_delta(base_raw: list, quant_raw: list,
                             spec: QuantSpec) -> dict:
    """Quantization-induced bad cases relative to the fp16 decode path.

    Both inputs are the ``raw_diagnostics`` lists returned by
    :func:`evaluate_decode_ppl` on the SAME windows. Aligning them
    token-by-token separates:

      * ``induced`` — fp16 top-1 correct, quantized top-1 wrong;
      * ``rescued`` — fp16 wrong, quantized correct.

    Categories/distance buckets are computed only over the induced tokens,
    so the report says WHICH failure mode the state writes create (near
    miss, confident error, copy/retrieval miss, novel token) instead of
    only how much PPL moved. Pure tensor bookkeeping; no RNG, no forwards.
    """
    if len(base_raw) != len(quant_raw):
        raise ValueError('base/quant diagnostics cover different windows: '
                         f'{len(base_raw)} vs {len(quant_raw)}')

    def _cat(raw, key):
        return torch.cat([d[key].reshape(-1) for d in raw])

    base_valid = _cat(base_raw, 'valid')
    quant_valid = _cat(quant_raw, 'valid')
    if base_valid.shape != quant_valid.shape:
        raise ValueError('base/quant diagnostics have different shapes')
    valid = base_valid & quant_valid
    base_correct = _cat(base_raw, 'correct')
    quant_correct = _cat(quant_raw, 'correct')
    induced = valid & base_correct & ~quant_correct
    rescued = valid & ~base_correct & quant_correct
    n_valid = int(valid.sum())
    n_fp16_correct = int((valid & base_correct).sum())
    n_fp16_wrong = int((valid & ~base_correct).sum())
    n_induced = int(induced.sum())
    n_rescued = int(rescued.sum())

    categories = [c for d in quant_raw for row in d['category'] for c in row]
    if len(categories) != int(valid.numel()):
        raise ValueError('category list is not aligned with token masks')
    cat_counts = {name: 0 for name in LM_ERROR_CATEGORIES}
    for cat, is_induced in zip(categories, induced.tolist()):
        if is_induced:
            if cat not in cat_counts:
                raise ValueError(f'unknown LM error category: {cat!r}')
            cat_counts[cat] += 1

    distance = _cat(quant_raw, 'last_seen_distance')
    seen = _cat(quant_raw, 'seen_earlier')
    buckets = []
    for lo, hi, label in ((1, 8, 'd01_08'), (9, 32, 'd09_32'),
                          (33, 128, 'd33_128'), (129, None, 'd129_plus')):
        mask = induced & seen & (distance >= lo)
        if hi is not None:
            mask = mask & (distance <= hi)
        n = int(mask.sum())
        buckets.append({
            'bucket': label, 'distance_range': [lo, hi], 'count': n,
            'share_of_induced': (n / n_induced) if n_induced else None,
        })
    n_absent = int((induced & ~seen).sum())
    buckets.append({
        'bucket': 'not_in_context', 'distance_range': None,
        'count': n_absent,
        'share_of_induced': ((n_absent / n_induced) if n_induced else None),
    })

    probs = _cat(quant_raw, 'top1_prob')[induced].float()
    return {
        'spec': spec_summary(spec),
        'n_tokens': n_valid,
        'n_fp16_correct': n_fp16_correct,
        'n_fp16_wrong': n_fp16_wrong,
        'n_induced_wrong': n_induced,
        'induced_rate_of_fp16_correct': (
            n_induced / n_fp16_correct if n_fp16_correct else None),
        'n_rescued_wrong': n_rescued,
        'rescued_rate_of_fp16_wrong': (
            n_rescued / n_fp16_wrong if n_fp16_wrong else None),
        'net_extra_wrong': n_induced - n_rescued,
        'categories_induced': {
            name: {
                'count': int(count),
                'share_of_induced': ((count / n_induced) if n_induced else None),
                'definition': LM_ERROR_CATEGORIES[name],
            }
            for name, count in cat_counts.items()
        },
        'distance_buckets_induced': buckets,
        'top1_prob_induced_mean': (
            float(probs.mean()) if probs.numel() else None),
        'note': ('Quantization-induced top-1 flips (fp16 correct -> '
                 'quantized wrong) on the same held-out windows; '
                 'read-only, no RNG impact.'),
    }


def _decode_state_snr(model, clone, windows, pad_id, spec):
    """Aggregate state-write SNR (dB) for one (seed, spec) cell.

    Separate pass from :func:`evaluate_decode_ppl`.
    """
    if spec is None:
        return math.inf
    num = den = 0.0
    for ids in windows:
        clone.reset()
        x = model.embed(ids)
        for t in range(ids.shape[1] - 1):
            with torch.no_grad():
                clone(x[:, t:t + 1])
                state = clone.model._kda_state
                flat = state.reshape(state.shape[0], -1)
                deq = fake_quantize(flat, spec, ste=False)
                num += float(flat.pow(2).sum())
                den += float((deq - flat).pow(2).sum())
        clone.reset()
    return 10.0 * math.log10(num / den) if den > 0.0 and num > 0.0 \
        else math.inf


def evaluate_stepwise_ppl(model, windows: list, pad_id: int,
                          spec: QuantSpec | None = None) -> dict:
    """Held-out perplexity + token accuracy of the stepwise KDA LM."""
    was_training = model.training
    model.eval()
    total_nll, n_tokens, n_correct = 0.0, 0, 0
    try:
        with torch.no_grad():
            for ids in windows:
                logits = model(ids, spec=spec, ste=False)[:, :-1]
                targets = ids[:, 1:1 + logits.shape[1]]
                mask = targets.ne(pad_id) & targets.ne(-100)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                    reduction='none')
                total_nll += float((nll * mask.reshape(-1)).sum())
                n_tokens += int(mask.sum())
                n_correct += int(
                    ((logits.argmax(-1) == targets) & mask).sum())
    finally:
        # A validation monitor must not leave the trainer in eval mode:
        # dropout / norm statistics would silently change on the next step.
        model.train(was_training)
    return {'ppl': _ppl_from_nll(total_nll, n_tokens),
            'mean_nll': total_nll / max(1, n_tokens), 'n_tokens': n_tokens,
            'token_accuracy': (n_correct / n_tokens) if n_tokens else None}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _aggregate(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {'mean': None, 'min': None, 'max': None}
    return {'mean': sum(values) / len(values), 'min': min(values),
            'max': max(values)}


def _moment(values: list) -> dict:
    """Mean, sample std, and 95% t-CI (plus min/max) over finite values."""
    vals = [v for v in values
            if v is not None and isinstance(v, (int, float))
            and math.isfinite(float(v))]
    if not vals:
        return {'mean': None, 'std': None, 'ci95': None,
                'min': None, 'max': None, 'n': 0}
    mean = sum(vals) / len(vals)
    std = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
           if len(vals) > 1 else None)
    ci = None
    if len(vals) > 1 and std is not None:
        from run_quality import _t_crit_975
        crit = _t_crit_975(len(vals))
        if crit is not None:
            half = crit * std / math.sqrt(len(vals))
            ci = [mean - half, mean + half]
    return {'mean': mean, 'std': std, 'ci95': ci,
            'min': min(vals), 'max': max(vals), 'n': len(vals)}


def _eval_spec_key(spec: QuantSpec | None) -> str:
    """Unique label for a spec in per-spec diagnostic dictionaries.

    Bits/group alone collide between the symmetric default and the
    asymmetric probe at the same (bits, group). The suffix keeps both
    cells in the JSON instead of silently overwriting one.
    """
    if spec is None:
        return 'fp16'
    key = f'{spec.bits}b/g{spec.group_size}'
    return key if spec.symmetric else f'{key}-asym'


def _spec_fields(spec: QuantSpec | None) -> dict:
    if spec is None:
        return {
            'quantized': False, 'bits': None, 'group_size': None,
            'symmetric': None, 'bits_per_element': FP16_BITS,
            'compression_vs_fp16': 1.0,
        }
    return {
        'quantized': True, 'bits': spec.bits, 'group_size': spec.group_size,
        'symmetric': spec.symmetric,
        'bits_per_element': bits_per_element(spec),
        'compression_vs_fp16': compression_ratio(spec),
    }


def _paired_vs_fp16(group: list, fp16_cells_by_seed: dict,
                    n_boot: int) -> dict | None:
    """Per-seed paired bootstrap of one spec vs the fp16 baseline.

    Every spec is scored on the SAME held-out windows as the fp16 cell of
    its seed (``per_window``, window-aligned), so the comparison is
    paired: the window-difficulty variance cancels and only the
    quantization effect (plus its interaction with window content)
    remains. ``ppl_bootstrap.paired_bootstrap_ppl`` supplies the CI for
    ``delta_mean_nll = quantized - fp16`` and the two-sided centered
    bootstrap p-value per seed.

    Returns a JSON-safe aggregate over seeds, or ``None`` when no cell
    carries ``per_window`` units (e.g. legacy/hand-built cells): the
    paired test is then simply not reported rather than faked.
    """
    records = []
    for cell in group:
        ref = fp16_cells_by_seed.get(cell['seed'])
        a_units = cell.get('per_window')
        b_units = ref.get('per_window') if ref else None
        if not a_units or not b_units:
            continue
        records.append(paired_bootstrap_ppl(
            a_units, b_units, n_boot=int(n_boot), seed=int(cell['seed'])))
    if not records:
        return None
    deltas = _aggregate([r['delta_mean_nll'] for r in records])
    p_values = [r['p_value'] for r in records if r['p_value'] is not None]
    n_sig = sum(1 for r in records if r['significant'])
    n_worse = sum(1 for r in records
                  if r['significant'] and r['worse_model'] == 'a')
    n_better = sum(1 for r in records
                   if r['significant'] and r['worse_model'] == 'b')
    keep = ('p_value', 'p_value_resolution', 'significant', 'worse_model',
            'delta_mean_nll', 'ci_delta_mean_nll', 'ppl_ratio_a_over_b',
            'ci_ppl_ratio', 'n_units', 'n_tokens_a')
    return {
        'n_seeds_tested': len(records),
        'n_seeds_significant': n_sig,
        'n_seeds_significant_worse': n_worse,
        'n_seeds_significant_better': n_better,
        'all_tested_seeds_significant_worse': bool(
            records and n_worse == len(records)),
        'delta_mean_nll_mean': deltas['mean'],
        'delta_mean_nll_min': deltas['min'],
        'delta_mean_nll_max': deltas['max'],
        'p_value_min': min(p_values) if p_values else None,
        'per_seed': [{k: r[k] for k in keep} for r in records],
        'alpha': records[0]['alpha'],
        'n_boot': records[0]['n_boot'],
        'direction': ('delta_mean_nll = mean_nll(quantized) - '
                      'mean_nll(fp16) per seed; positive => quantization '
                      'damage (higher NLL); p-value = two-sided centered '
                      'paired bootstrap over the shared windows'),
        'method': records[0]['method'],
    }


def build_ptq_rows(per_seed: list, specs: list, tolerance: float,
                   n_boot: int = 2000) -> tuple:
    """Aggregate Part A per-seed cells into rows (cells + summary)."""
    seeds = [cell['seed'] for cell in per_seed]
    # fp16 decode ppl per seed for ratio normalization.
    fp16_by_seed = {cell['seed']: cell['decode_ppl']
                    for cell in per_seed if cell['spec'] is None}
    fp16_cells_by_seed = {cell['seed']: cell
                          for cell in per_seed if cell['spec'] is None}
    rows = []
    cells = []
    for spec in specs:
        # Value equality (QuantSpec is a frozen dataclass): a caller may
        # pass an equal-but-distinct spec object.
        cells.append([c for c in per_seed if c['spec'] == spec])
    for spec, group in zip(specs, cells):
        if not group:
            continue
        ppl = _aggregate([c['decode_ppl'] for c in group])
        ratio = _aggregate([
            c['decode_ppl'] / fp16_by_seed[c['seed']] for c in group])
        top1 = _aggregate([c['top1_agreement'] for c in group])
        snr = _aggregate([c['state_snr_db'] for c in group])
        state_elements = group[0]['state_elements']
        row = {
            'row_kind': 'ptq_ppl',
            'model': 'hybrid_lm',
            'seeds': [int(s) for s in seeds],
            'state_elements': int(state_elements),
            'state_bytes': (
                storage_bytes(state_elements, spec) if spec is not None
                else state_elements * FP16_BITS / 8.0),
            'prefill_ppl_mean': _aggregate(
                [c['prefill_ppl'] for c in group])['mean'],
            'decode_ppl_mean': ppl['mean'],
            'decode_ppl_min': ppl['min'],
            'decode_ppl_max': ppl['max'],
            'ppl_ratio_mean': ratio['mean'],
            'ppl_ratio_min': ratio['min'],
            'ppl_ratio_max': ratio['max'],
            'top1_agreement_mean': top1['mean'],
            'top1_agreement_min': top1['min'],
            'state_snr_db_mean': snr['mean'],
            'ppl_preserving': bool(
                ratio['min'] is not None and ratio['min'] <= 1.0 + tolerance),
            'tolerance': float(tolerance),
            # Significance view of the same comparison the tolerance flag
            # summarizes: is the per-seed quantization damage larger than
            # the window-sampling noise of this held-out set? (None for the
            # fp16 reference itself, or when per-window units are
            # unavailable.)
            'paired_vs_fp16': (
                _paired_vs_fp16(group, fp16_cells_by_seed, n_boot)
                if spec is not None else None),
        }
        row.update(_spec_fields(spec))
        rows.append(row)

    quantized = [r for r in rows if r['quantized']]
    fp16_row = next(r for r in rows if not r['quantized'])
    preserving = [r for r in quantized if r['ppl_preserving']]
    worst = (max(quantized, key=lambda r: r['ppl_ratio_mean'])
             if quantized else None)
    significant_damage = [
        r for r in quantized
        if (r.get('paired_vs_fp16') or {}).get(
            'all_tested_seeds_significant_worse')]
    rows.append({
        'row_kind': 'ptq_summary',
        'model': 'hybrid_lm',
        'seeds': [int(s) for s in seeds],
        'tolerance': float(tolerance),
        'n_cells': len(quantized),
        'fp16_decode_ppl_mean': fp16_row['decode_ppl_mean'],
        'fp16_state_bytes': fp16_row['state_bytes'],
        'prefill_ppl_mean': fp16_row['prefill_ppl_mean'],
        'prefill_decode_ppl_ratio': (
            fp16_row['prefill_ppl_mean'] / fp16_row['decode_ppl_mean']),
        'cheapest_ppl_preserving_cell': (
            min(preserving, key=lambda r: r['state_bytes'])
            if preserving else None),
        'worst_cell': worst,
        'n_cells_significant_damage_all_seeds': len(significant_damage),
        'significant_damage_note': (
            'cells whose quantized-vs-fp16 paired bootstrap over the '
            'shared held-out windows was significant (alpha=0.05) and '
            'worse for EVERY tested seed; the tolerance flag '
            'ppl_preserving answers "is the ratio small", this answers '
            '"is it distinguishable from window-sampling noise" — a cell '
            'can pass one and fail the other at these corpus sizes'),
    })
    return rows


def build_ptq_test_rows(train_reports: list, specs: list,
                        tolerance: float) -> list:
    """Aggregate the confirmatory TEST pass into ``ptq_test_ppl`` rows.

    These rows exist to answer the reviewer question "was the final number
    ever computed on data that did not participate in model/spec
    selection?". They are NOT inputs to ``build_ptq_rows`` and are clearly
    marked ``used_for_selection=False``. Ratios are against the fp16 test
    decode on the same windows, matching the validation definition.
    """
    usable = [r['test_eval'] for r in train_reports
              if r.get('test_eval') and not r['test_eval'].get('skipped')
              and r['test_eval'].get('n_tokens')]
    if not usable:
        return []
    fp16_by_seed = {int(r['seed']): r['test_eval']['fp16']['ppl']
                    for r in train_reports
                    if r.get('test_eval')
                    and not r['test_eval'].get('skipped')}
    rows = []
    for spec in [None] + list(specs):
        label = (None if spec is None else _eval_spec_key(spec))
        ppls, ratios, spec_seeds = [], [], []
        for rep in train_reports:
            te = rep.get('test_eval')
            if not te or te.get('skipped') or not te.get('n_tokens'):
                continue
            if spec is None:
                ppl = te['fp16']['ppl']
            else:
                cell = te.get('quantized', {}).get(label)
                if cell is None:
                    continue
                ppl = cell['ppl']
            ppls.append(float(ppl))
            spec_seeds.append(int(rep['seed']))
            if fp16_by_seed[int(rep['seed'])]:
                ratios.append(float(ppl) / fp16_by_seed[int(rep['seed'])])
        if not ppls:
            continue
        agg = _aggregate(ppls)
        ratio = _aggregate(ratios)
        row = {
            'row_kind': 'ptq_test_ppl',
            'model': 'hybrid_lm',
            'eval_split': 'test',
            'used_for_selection': False,
            'seeds': spec_seeds,
            'n_windows': int(usable[0]['n_windows']),
            'n_tokens_per_seed': [int(u['n_tokens']) for u in usable],
            'prefill_ppl_mean': _aggregate(
                [float(u['prefill_ppl']) for u in usable
                 if u.get('prefill_ppl') is not None])['mean'],
            'decode_ppl_mean': agg['mean'],
            'decode_ppl_min': agg['min'],
            'decode_ppl_max': agg['max'],
            'ppl_ratio_mean': ratio['mean'],
            'ppl_ratio_min': ratio['min'],
            'ppl_ratio_max': ratio['max'],
            'ppl_preserving': bool(
                ratio['min'] is not None and ratio['min'] <= 1.0 + tolerance),
            'tolerance': float(tolerance),
            'note': ('Confirmatory held-out test pass on the SAME trained '
                     'models, scored after the validation protocol and '
                     'spec grid were frozen; never used for selection, '
                     'early stopping, or tolerance tuning.'),
        }
        state_elements = int(usable[0].get('state_elements') or 0)
        row['state_elements'] = state_elements
        row['state_bytes'] = (
            storage_bytes(state_elements, spec) if spec is not None
            else state_elements * FP16_BITS / 8.0)
        row.update(_spec_fields(spec))
        rows.append(row)
    return rows


def build_qat_rows(per_seed: list, spec: QuantSpec) -> list:
    """Aggregate Part B per-seed QAT cells + the recovery summary row.

    Historical fields keep their exact definitions. The matched-budget
    fields use the equally trained fp16 control: when
    ``qat_steps == steps`` that control IS the historical fp16 arm, so the
    two recoveries coincide; when the budgets differ, only
    ``recovery_fraction_matched`` isolates the quantizer.
    """
    rows = []
    for cell in per_seed:
        row = dict(cell)
        row['row_kind'] = 'qat_per_seed'
        row.pop('spec', None)
        rows.append(row)
    fp16 = _aggregate([c['fp16_ppl'] for c in per_seed])
    ptq = _aggregate([c['ptq_ppl'] for c in per_seed])
    qat = _aggregate([c['qat_ppl'] for c in per_seed])
    qat_fp16 = _aggregate([c['qat_model_fp16_state_ppl'] for c in per_seed])
    recovery = _aggregate([c['recovery_fraction'] for c in per_seed])
    fp16_m = _moment([c.get('fp16_matched_ppl') for c in per_seed])
    ptq_m = _moment([c.get('ptq_matched_ppl') for c in per_seed])
    rec_m = _moment([c.get('recovery_fraction_matched')
                     for c in per_seed])
    fp16_mom = _moment([c['fp16_ppl'] for c in per_seed])
    ptq_mom = _moment([c['ptq_ppl'] for c in per_seed])
    qat_mom = _moment([c['qat_ppl'] for c in per_seed])
    matched_diffs = [c['qat_minus_matched_log_ppl'] for c in per_seed
                     if c.get('qat_minus_matched_log_ppl') is not None]
    matched_paired = (summarize_paired(matched_diffs)
                      if len(matched_diffs) >= 2 else None)
    budgets_matched = all(
        bool(c.get('historical_budget_matched',
                   c.get('steps') == c.get('qat_steps')))
        for c in per_seed)
    row = {
        'row_kind': 'qat_recovery',
        'model': 'stepwise_kda_lm',
        'seeds': [int(c['seed']) for c in per_seed],
        'n_seeds': len(per_seed),
        'spec': spec_summary(spec),
        'steps': per_seed[0]['steps'],
        'qat_steps': per_seed[0]['qat_steps'],
        'fp16_ppl_mean': fp16['mean'], 'fp16_ppl_min': fp16['min'],
        'fp16_ppl_max': fp16['max'],
        'ptq_ppl_mean': ptq['mean'], 'ptq_ppl_min': ptq['min'],
        'ptq_ppl_max': ptq['max'],
        'qat_ppl_mean': qat['mean'], 'qat_ppl_min': qat['min'],
        'qat_ppl_max': qat['max'],
        'qat_model_fp16_state_ppl_mean': qat_fp16['mean'],
        'recovery_fraction_mean': recovery['mean'],
        'recovery_fraction_min': recovery['min'],
        'recovery_fraction_max': recovery['max'],
        # Mean +/- sample std / 95% t-CI (historical means kept above).
        'fp16_ppl_std': fp16_mom['std'],
        'fp16_ppl_ci95': fp16_mom['ci95'],
        'ptq_ppl_std': ptq_mom['std'],
        'ptq_ppl_ci95': ptq_mom['ci95'],
        'qat_ppl_std': qat_mom['std'],
        'qat_ppl_ci95': qat_mom['ci95'],
        'recovery_fraction_std': _moment(
            [c['recovery_fraction'] for c in per_seed])['std'],
        'recovery_fraction_ci95': _moment(
            [c['recovery_fraction'] for c in per_seed])['ci95'],
        # Matched-budget control (primary causal QAT contrast).
        'historical_budget_matched': budgets_matched,
        'fp16_matched_ppl_mean': fp16_m['mean'],
        'fp16_matched_ppl_std': fp16_m['std'],
        'fp16_matched_ppl_ci95': fp16_m['ci95'],
        'ptq_matched_ppl_mean': ptq_m['mean'],
        'ptq_matched_ppl_std': ptq_m['std'],
        'ptq_matched_ppl_ci95': ptq_m['ci95'],
        'recovery_fraction_matched_mean': rec_m['mean'],
        'recovery_fraction_matched_std': rec_m['std'],
        'recovery_fraction_matched_ci95': rec_m['ci95'],
        'qat_vs_matched_control': matched_paired,
        'budget_control': {
            'total_steps_fp16': int(per_seed[0]['steps']),
            'total_steps_qat': int(per_seed[0]['qat_steps']),
            'total_steps_fp16_matched': int(per_seed[0]['qat_steps']),
            'historical_arms_equal_budget': bool(budgets_matched),
            'note': ('When QLM_STEPS != QLM_QAT_STEPS the historical fp16 '
                     'arm is shorter than QAT; the matched control is a '
                     'fresh fp16 run at QLM_QAT_STEPS with the same seed/'
                     'init/data/optimizer/LR, so only the quantizer '
                     'differs. qat_vs_matched_control tests that contrast '
                     '(log-PPL difference, <0 favors QAT).'),
        },
        'grad_norm_first_mean': _aggregate(
            [c['grad_norm_first'] for c in per_seed])['mean'],
        'grad_norm_last_mean': _aggregate(
            [c['grad_norm_last'] for c in per_seed])['mean'],
        'all_grads_finite': bool(
            all(c['all_grads_finite'] for c in per_seed)),
        'recovery_definition': (
            '1 - residual/ptq_damage where residual = qat_ppl/fp16_ppl - 1 '
            'and damage = ptq_ppl/fp16_ppl - 1 per seed; None when the '
            'seed shows no PTQ damage (damage <= 0). The *_matched variant '
            'uses the equally trained fp16 control instead of the shorter '
            'fp16 arm.'),
    }
    rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Pre-training overfit sanity gates (one fixed batch per part)
# --------------------------------------------------------------------------
def _fixed_lm_batch(tokenizer, train_texts, seq_len, batch_size, seed):
    """One fixed (input_ids, labels) batch from the seeded training loader.

    Deterministic given ``seed``: the DataLoader shuffle uses the seeded
    generator + ``seed_worker`` and the dataset order is fixed. The gate
    trains on THIS single batch for all steps (data held constant).
    """
    loader = _make_loader(tokenizer, train_texts, seq_len, batch_size, seed)
    batch = next(iter(loader))
    return {'input_ids': batch['input_ids'].clone(),
            'labels': batch['labels'].clone()}


def _run_stepwise_ste_overfit_gate(batch, *, device, tokenizer, steps, seed,
                                   lr=3e-3, grad_clip=1.0,
                                   loss_ratio_threshold=0.5, log_every=0):
    """Overfit one fixed batch through the QAT (STE) path of StepwiseKDALM.

    ``train_lm_autodl.run_overfit_sanity_check`` cannot cover Part B:
    ``StepwiseKDALM`` is a single KDA layer with no ``hybrid`` attribute and
    a logits-only forward, so the straight-through gradient contract in the
    fake-quantized state write path needs its own gate. Trains a FRESH
    model on a fixed batch with regularization disabled and passes when
    every gradient is finite AND the tail-mean loss drops below
    ``loss_ratio_threshold`` times the head-mean loss.

    NECESSARY, NOT SUFFICIENT: passing proves the STE path propagates a
    usable gradient signal on a fixed batch; it does not prove the
    quantizer matches the deployment path.
    """
    import gc

    if isinstance(device, str):
        device = torch.device(device)
    steps = int(steps)
    if steps < 10:
        raise ValueError(
            f'overfit check needs steps >= 10 to be meaningful, got {steps}')
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    seed_info = set_global_seed(seed, deterministic=True)
    model = StepwiseKDALM(len(tokenizer)).to(device)
    model.train()
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses, accs, elapsed_s = [], [], []
    all_grads_finite = True
    grad_flow = None
    t0 = time.time()
    for step in range(steps):
        logits = model(input_ids, spec=QAT_SPEC, ste=True)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100)
        if loss is None or not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=(loss if loss is not None else float('nan')),
                token_ids={'input_ids': input_ids},
                id_range=(0, len(tokenizer)),
                params=model.parameters(), logits=logits,
                context=f'overfit sanity check (Part B STE) step={step}')
        with torch.no_grad():
            mask = labels.reshape(-1).ne(-100)
            correct = ((logits.argmax(-1).reshape(-1) == labels.reshape(-1))
                       & mask).sum()
            acc = float(correct) / max(1, int(mask.sum()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            # One-shot gradient-flow audit right after the first backward:
            # on the STE path a detached or unused tensor would silently
            # keep receiving no gradient while the loss still falls.
            grad_flow = grad_flow_report(model.named_parameters())
        bad_grads = {name: p.grad for name, p in model.named_parameters()
                     if p.grad is not None
                     and not bool(torch.isfinite(p.grad).all())}
        clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            all_grads_finite = False
            raise_non_finite(
                'gradient', loss=loss,
                token_ids={'input_ids': input_ids},
                id_range=(0, len(tokenizer)),
                params=model.parameters(), logits=logits, grads=bad_grads,
                grad_norm=float(clip_norm),
                context=f'overfit sanity check (Part B STE) step={step}')
        opt.step()
        losses.append(float(loss.detach()))
        accs.append(acc)
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'[overfit-check] partB-stepwise-ste step {step:4d} '
                  f'loss {losses[-1]:.4f} token_acc {acc:.4f} '
                  f'({elapsed_s[-1]:.1f}s)', flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail
    passed = (all_grads_finite
              and final_loss < loss_ratio_threshold * initial_loss)
    report = {
        'passed': bool(passed),
        'label': 'partB-stepwise-ste',
        'steps': steps,
        'lr': float(lr),
        'weight_decay': 0.0,
        'dropout': 0.0,
        'data_augmentation': 'none',
        'grad_clip_norm': float(grad_clip),
        'seed': int(seed),
        'seed_info': seed_info,
        'spec': spec_summary(QAT_SPEC),
        'ste': True,
        'all_grads_finite': bool(all_grads_finite),
        'grad_flow': grad_flow,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': min(losses),
        'loss_reduction': ((initial_loss - final_loss) / initial_loss
                           if initial_loss > 0 else None),
        'final_train_token_acc': final_acc,
        'loss_ratio_threshold': float(loss_ratio_threshold),
        'losses': losses,
        'accs': accs,
        'elapsed_s': elapsed_s,
        'note': ('Necessary but NOT sufficient: passing proves the STE path '
                 'propagates a usable gradient signal on a fixed batch; it '
                 'does not prove the quantizer matches the deployment path.'),
    }
    if grad_flow and grad_flow['n_missing_grad']:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the first backward "
              f"(e.g. {grad_flow['missing_grad_names'][:3]})", flush=True)
    del model, opt
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def _compact_qlm_gate_report(report):
    """Strip per-step curves for compact metadata embedding."""
    return {k: v for k, v in report.items()
            if k not in ('losses', 'accs', 'elapsed_s', 'seed_info')}


def run_overfit_gates(*, device, tokenizer, split_texts, seq_len,
                      batch_size, skip_qat, lr=3e-3):
    """Pre-training overfit gates for both QLM parts; (record, proceed).

    Environment controls:
      * ``QLM_OVERFIT_CHECK=1|0``  (default 1: run the gates)
      * ``QLM_OVERFIT_STRICT=1|0`` (default 1: refuse to start training
        when any gate cell fails)
      * ``QLM_OVERFIT_STEPS=N``    (default 100 optimizer steps per cell)
      * ``QLM_OVERFIT_SEED=N``     (default 2026: model + fixed batch)
      * ``QLM_OVERFIT_BATCH=N``    (default 8 sequences)
    """
    enabled = parse_int_env('QLM_OVERFIT_CHECK', 1, min_value=0,
                            logger=logger) != 0
    strict = parse_int_env('QLM_OVERFIT_STRICT', 1, min_value=0,
                           logger=logger) != 0
    if not enabled:
        logger.info('  overfit sanity gates disabled '
                    '(QLM_OVERFIT_CHECK=0)')
        return {'enabled': False, 'strict': bool(strict), 'reports': []}, True
    steps = parse_int_env('QLM_OVERFIT_STEPS', 100, min_value=10,
                          logger=logger)
    seed = parse_int_env('QLM_OVERFIT_SEED', 2026, min_value=0,
                         logger=logger)
    gate_batch = parse_int_env('QLM_OVERFIT_BATCH', 8, min_value=1,
                               logger=logger)
    batch = _fixed_lm_batch(tokenizer, split_texts['train'], seq_len,
                            gate_batch, seed)
    logger.info(f'  overfit sanity gates: {steps} steps on one fixed batch '
                f'each (batch={gate_batch}, seed={seed}, '
                f'strict={bool(strict)})')

    reports = []
    # Part A: hybrid LM through the canonical train_lm_autodl gate.
    try:
        rep_a = run_lm_overfit_sanity_check(
            lambda: LMWithHybrid(len(tokenizer), small_hybrid_config()),
            batch, device, steps=steps, lr=lr, seed=seed,
            log_every=max(10, steps // 5))
        rep_a['label'] = 'partA-hybrid-lm'
    except Exception as e:
        logger.error(f'  [overfit] partA-hybrid-lm: gate crashed: {e}')
        rep_a = {'passed': False, 'label': 'partA-hybrid-lm',
                 'error': str(e), 'steps': steps, 'seed': seed,
                 'train_batch': gate_batch}
    reports.append(rep_a)

    # Part B: STE QAT path (skipped with the rest of Part B).
    if not skip_qat:
        try:
            rep_b = _run_stepwise_ste_overfit_gate(
                batch, device=device, tokenizer=tokenizer, steps=steps,
                seed=seed, lr=lr, log_every=max(10, steps // 5))
        except Exception as e:
            logger.error(f'  [overfit] partB-stepwise-ste: gate crashed: {e}')
            rep_b = {'passed': False, 'label': 'partB-stepwise-ste',
                     'error': str(e), 'steps': steps, 'seed': seed,
                     'train_batch': gate_batch}
        reports.append(rep_b)

    all_passed = True
    for rep in reports:
        if rep.get('passed'):
            logger.info(
                f"  [overfit] {rep['label']}: PASS "
                f"(loss {rep.get('initial_loss'):.4f}->"
                f"{rep.get('final_loss'):.4f})")
        else:
            all_passed = False
            logger.error(
                f"  [overfit] {rep['label']}: FAIL "
                f"(loss={rep.get('final_loss')}, "
                f"error={rep.get('error')})")
    record = {'enabled': True, 'strict': bool(strict), 'steps': steps,
              'seed': seed, 'train_batch': gate_batch, 'reports': reports}
    if not all_passed and strict:
        logger.error(
            '  QLM_OVERFIT_STRICT=1: refusing to start training on a '
            'recipe that cannot overfit a fixed batch. Fix the training '
            'path, or set QLM_OVERFIT_STRICT=0 to record the failure and '
            'continue (not recommended).')
        return record, False
    return record, True


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------
def run_part_a(*, device, tokenizer, split_texts, seeds, specs, steps,
               seq_len, batch_size, eval_seqs, tolerance, lr=3e-3,
               val_every=0, error_batches=0, max_error_examples=2,
               test_seqs=0) -> tuple:
    """Train + PTQ-score the hybrid LM; return (per_seed_cells, train_report).

    ``val_every`` records the held-out curve during training and
    ``error_batches`` runs the token-level bad-case analysis on the
    validation split, both read-only and off by default so direct callers
    (and the unit tests) keep the historical training path exactly.

    ``test_seqs > 0`` additionally scores the sealed TEST split with the
    already-trained model after every validation-based decision has been
    made. The test windows are never used for the spec grid, the tolerance
    flag, the validation curve, or the bad-case analysis; they are stored
    in the train report as a confirmatory read-only measurement. Keeping
    the default at 0 preserves the historical direct-call contract.
    """
    pad = pad_token_id(tokenizer)
    val_windows = encode_windows(tokenizer, split_texts['val'][:eval_seqs],
                                 seq_len, device)
    if not val_windows:
        raise RuntimeError('Part A: the val split produced no windows')
    val_dataset = TinyStoriesLM(tokenizer, seq_len=seq_len,
                                texts=split_texts['val'][:eval_seqs],
                                name='val', split='val')
    per_seed, train_reports = [], []
    for seed in seeds:
        def _val_monitor(model, _dataset=val_dataset):
            rep = evaluate_lm_loss(
                model, _dataset, batch_size=1, device=device,
                return_accuracy=True)
            if rep['loss'] is None:
                raise RuntimeError('Part A validation produced no tokens')
            return ValPoint(
                loss=rep['loss'],
                acc=(rep['token_accuracy']
                     if rep.get('token_accuracy') is not None else 0.0))

        model, report = train_hybrid_lm(
            seed=seed, tokenizer=tokenizer, train_texts=split_texts['train'],
            steps=steps, seq_len=seq_len, batch_size=batch_size,
            device=device, lr=lr, val_every=val_every,
            val_monitor=_val_monitor)
        if int(error_batches or 0) > 0:
            # Token-level bad cases on the SAME held-out split used for the
            # curve: near-miss / confident-wrong / copyable-missed / novel
            # categories plus decoded examples, so a PPL change can be
            # attributed to a failure mode rather than to a scalar.
            report['error_analysis'] = analyze_lm_errors(
                model, val_dataset, batch_size=1, device=device,
                max_batches=int(error_batches), seed=seed,
                tokenizer=tokenizer,
                max_examples=int(max_error_examples))
        prefill = evaluate_lm_loss(model, val_dataset, batch_size=1,
                                   device=device)
        if prefill['perplexity'] is None:
            raise RuntimeError('Part A: prefill evaluation had no tokens')
        clone = build_decoding_clone(model, device=device)
        diagnose_quant = int(error_batches or 0) > 0
        base = evaluate_decode_ppl(
            model, clone, val_windows, pad, spec=None,
            collect_diagnostics=diagnose_quant)
        quant_errors = {}
        cells = [None] + list(specs)
        for spec in cells:
            scored = (base if spec is None
                      else evaluate_decode_ppl(
                          model, clone, val_windows, pad, spec=spec,
                          collect_diagnostics=diagnose_quant))
            if spec is not None and diagnose_quant:
                label = _eval_spec_key(spec)
                quant_errors[label] = quantization_error_delta(
                    base['raw_diagnostics'], scored['raw_diagnostics'], spec)
            snr = math.inf if spec is None else _decode_state_snr(
                model, clone, val_windows, pad, spec)
            agreement = 1.0
            if spec is not None:
                a = torch.cat(base['argmax'])
                b = torch.cat(scored['argmax'])
                agreement = float((a == b).to(torch.float32).mean())
            per_seed.append({
                'seed': int(seed),
                'spec': spec,
                'prefill_ppl': float(prefill['perplexity']),
                'decode_ppl': scored['ppl'],
                'mean_nll': scored['mean_nll'],
                'n_tokens': scored['n_tokens'],
                'top1_agreement': agreement,
                'state_elements': scored['state_elements'],
                'state_snr_db': snr,
                # Per-window (nll_sum, n_tokens) sufficient statistic: the
                # inputs of the paired quantized-vs-fp16 bootstrap in
                # build_ptq_rows. Same window order for every spec, so the
                # pairing is exact by construction.
                'per_window': scored['per_window'],
            })
        if quant_errors:
            report['quantization_error_analysis'] = quant_errors
            for label, delta in quant_errors.items():
                logger.info(
                    '  [quant-bad-case] %s: %d/%d fp16-correct tokens '
                    'flipped (%s), rescued %d, top categories %s',
                    label, delta['n_induced_wrong'], delta['n_fp16_correct'],
                    ('n/a' if delta['induced_rate_of_fp16_correct'] is None
                     else f"{delta['induced_rate_of_fp16_correct']:.2%}"),
                    delta['n_rescued_wrong'],
                    ', '.join(
                        f'{name}={v["count"]}'
                        for name, v in delta['categories_induced'].items()
                        if v['count']))
        report['seed'] = int(seed)
        if int(test_seqs or 0) > 0:
            # Confirmatory TEST pass: by this point the spec grid, the
            # tolerance and every validation decision are fixed. Opening
            # test here cannot influence selection (and the returned cells
            # keep their historical validation semantics).
            test_texts = list(split_texts.get('test', []))[:int(test_seqs)]
            test_windows = encode_windows(tokenizer, test_texts, seq_len,
                                         device)
            if test_windows:
                test_dataset = TinyStoriesLM(
                    tokenizer, seq_len=seq_len, texts=test_texts,
                    name='test', split='test')
                test_prefill = evaluate_lm_loss(
                    model, test_dataset, batch_size=1, device=device)
                test_base = evaluate_decode_ppl(
                    model, clone, test_windows, pad, spec=None)
                test_quant = {}
                for spec in specs:
                    label = _eval_spec_key(spec)
                    scored_test = evaluate_decode_ppl(
                        model, clone, test_windows, pad, spec=spec)
                    test_quant[label] = {
                        'ppl': scored_test['ppl'],
                        'mean_nll': scored_test['mean_nll'],
                        'n_tokens': scored_test['n_tokens'],
                    }
                report['test_eval'] = {
                    'split': 'test',
                    'used_for_selection': False,
                    'n_windows': len(test_windows),
                    'n_tokens': test_base['n_tokens'],
                    'prefill_ppl': test_prefill.get('perplexity'),
                    'state_elements': test_base.get('state_elements'),
                    'fp16': {
                        'ppl': test_base['ppl'],
                        'mean_nll': test_base['mean_nll'],
                        'n_tokens': test_base['n_tokens'],
                    },
                    'quantized': test_quant,
                }
                logger.info(
                    'Part A seed %d: sealed TEST decode fp16 ppl %.4f '
                    '(n_tokens=%d; not used for selection)',
                    seed, test_base['ppl'], test_base['n_tokens'])
            else:
                report['test_eval'] = {
                    'split': 'test', 'used_for_selection': False,
                    'skipped': True, 'reason': 'no test windows',
                    'n_windows': 0, 'quantized': {},
                }
        train_reports.append(report)
        logger.info(
            'Part A seed %d: prefill ppl %.4f, decode fp16 ppl %.4f '
            '(train loss %.4f -> %.4f)', seed, prefill['perplexity'],
            base['ppl'], report['loss_first'], report['loss_last'])
    return per_seed, train_reports


def run_part_b(*, device, tokenizer, split_texts, seeds, qat_steps,
               seq_len, batch_size, eval_seqs, spec, steps, lr=3e-3,
               val_every=0) -> list:
    """fp16-vs-QAT A/B training + scoring for the stepwise KDA LM."""
    pad = pad_token_id(tokenizer)
    val_windows = encode_windows(tokenizer, split_texts['val'][:eval_seqs],
                                 seq_len, device)
    if not val_windows:
        raise RuntimeError('Part B: the val split produced no windows')

    def _val_monitor(model):
        rep = evaluate_stepwise_ppl(model, val_windows, pad)
        return ValPoint(
            loss=rep['mean_nll'],
            acc=(rep['token_accuracy']
                 if rep.get('token_accuracy') is not None else 0.0))

    per_seed = []
    for seed in seeds:
        fp16_model, fp16_rep = train_stepwise_lm(
            seed=seed, tokenizer=tokenizer, train_texts=split_texts['train'],
            steps=steps, seq_len=seq_len, batch_size=batch_size,
            device=device, spec=None, ste=False, lr=lr,
            val_every=val_every, val_monitor=_val_monitor)
        qat_model, qat_rep = train_stepwise_lm(
            seed=seed, tokenizer=tokenizer, train_texts=split_texts['train'],
            steps=qat_steps, seq_len=seq_len, batch_size=batch_size,
            device=device, spec=spec, ste=True, lr=lr,
            val_every=val_every, val_monitor=_val_monitor)
        fp16_ppl = evaluate_stepwise_ppl(fp16_model, val_windows, pad)['ppl']
        ptq_ppl = evaluate_stepwise_ppl(fp16_model, val_windows, pad,
                                        spec=spec)['ppl']
        qat_ppl = evaluate_stepwise_ppl(qat_model, val_windows, pad,
                                        spec=spec)['ppl']
        qat_fp16 = evaluate_stepwise_ppl(qat_model, val_windows, pad)['ppl']

        # Budget-matched fp16 control. QAT trains for ``qat_steps`` while
        # the historical fp16 arm trains for ``steps``; when those differ,
        # the QAT-vs-fp16 contrast is confounded with extra optimization.
        # Reuse the fp16 arm only when the budgets already match; otherwise
        # train a fresh control for exactly ``qat_steps`` with the same
        # seed/init/data/optimizer/LR (the only difference from QAT is the
        # state quantizer).
        historical_budget_matched = int(qat_steps) == int(steps)
        matched_reused = historical_budget_matched
        if matched_reused:
            fp16_matched_rep = fp16_rep
            fp16_matched_ppl = fp16_ppl
            ptq_matched_ppl = ptq_ppl
        else:
            fp16_matched_model, fp16_matched_rep = train_stepwise_lm(
                seed=seed, tokenizer=tokenizer,
                train_texts=split_texts['train'], steps=qat_steps,
                seq_len=seq_len, batch_size=batch_size, device=device,
                spec=None, ste=False, lr=lr, val_every=val_every,
                val_monitor=_val_monitor)
            fp16_matched_ppl = evaluate_stepwise_ppl(
                fp16_matched_model, val_windows, pad)['ppl']
            ptq_matched_ppl = evaluate_stepwise_ppl(
                fp16_matched_model, val_windows, pad, spec=spec)['ppl']

        damage = ptq_ppl / fp16_ppl - 1.0
        residual = qat_ppl / fp16_ppl - 1.0
        recovery = (1.0 - residual / damage) if damage > 0.0 else None
        matched_damage = ptq_matched_ppl / fp16_matched_ppl - 1.0
        matched_residual = qat_ppl / fp16_matched_ppl - 1.0
        recovery_matched = (
            1.0 - matched_residual / matched_damage
            if matched_damage > 0.0 else None)
        per_seed.append({
            'seed': int(seed),
            'steps': int(steps),
            'qat_steps': int(qat_steps),
            'total_steps_fp16': int(steps),
            'total_steps_qat': int(qat_steps),
            'total_steps_fp16_matched': int(qat_steps),
            'historical_budget_matched': bool(historical_budget_matched),
            'matched_control_reused': bool(matched_reused),
            'fp16_ppl': fp16_ppl,
            'ptq_ppl': ptq_ppl,
            'qat_ppl': qat_ppl,
            'qat_model_fp16_state_ppl': qat_fp16,
            'fp16_matched_ppl': fp16_matched_ppl,
            'ptq_matched_ppl': ptq_matched_ppl,
            'recovery_fraction': recovery,
            'recovery_fraction_matched': recovery_matched,
            # Log-PPL difference at matched budget: <0 means QAT has lower
            # perplexity than the equally trained fp16 control.
            'qat_minus_matched_log_ppl': (
                math.log(qat_ppl) - math.log(fp16_matched_ppl)
                if qat_ppl > 0.0 and fp16_matched_ppl > 0.0 else None),
            'extra_steps_vs_fp16': int(qat_steps) - int(steps),
            # Persist the held-out curve for each arm (None when disabled):
            # a QAT arm may end with lower train loss while still improving
            # or already overfitting on validation, which the endpoint PPL
            # contrast alone cannot reveal.
            'fp16_validation': fp16_rep.get('validation'),
            'qat_validation': qat_rep.get('validation'),
            'fp16_matched_validation': fp16_matched_rep.get('validation'),
            'grad_norm_first': qat_rep['grad_norm_first'],
            'grad_norm_last': qat_rep['grad_norm_last'],
            'all_grads_finite': bool(
                qat_rep['all_grads_finite']
                and fp16_rep['all_grads_finite']
                and fp16_matched_rep['all_grads_finite']),
        })
        logger.info(
            'Part B seed %d: fp16 %.4f | PTQ %s %.4f | QAT %.4f '
            '| fp16-matched %.4f (PTQ %.4f) | recovery %s '
            '(matched %s)', seed, fp16_ppl,
            spec_summary(spec)['bits'], ptq_ppl, qat_ppl,
            fp16_matched_ppl, ptq_matched_ppl,
            'n/a' if recovery is None else f'{recovery:.2%}',
            'n/a' if recovery_matched is None
            else f'{recovery_matched:.2%}')
    return per_seed


def _ints(raw, name, min_value=1):
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as e:
            raise ValueError(f'{name}: {part!r} is not an int') from e
        if value < min_value:
            raise ValueError(f'{name}: {value} < {min_value}')
        values.append(value)
    if not values:
        raise ValueError(f'{name}: empty list')
    return tuple(dict.fromkeys(values))


def _print_ptq(rows):
    logger.info('\n-- Part A: held-out decode perplexity vs state bits '
                '(mean [min..max] over seeds) --')
    header = (f"{'cell':>12} | {'bits/el':>7} | {'bytes':>7} | "
              f"{'xfp16':>6} | ppl (min..max) | ratio | top1 | snr dB")
    logger.info(header)
    logger.info('-' * len(header))
    for row in rows:
        if row['row_kind'] != 'ptq_ppl':
            continue
        label = ('fp16' if not row['quantized']
                 else f"{row['bits']}b/g{row['group_size']}")
        logger.info(
            f"{label:>12} | {row['bits_per_element']:>7.2f} | "
            f"{row['state_bytes']:>7.1f} | "
            f"{row['compression_vs_fp16']:>5.2f}x | "
            f"{row['decode_ppl_mean']:.4f} "
            f"[{row['decode_ppl_min']:.4f}..{row['decode_ppl_max']:.4f}] | "
            f"{row['ppl_ratio_mean']:.4f} | "
            f"{row['top1_agreement_mean']:.3f} | "
            f"{row['state_snr_db_mean']:.1f}")
        paired = row.get('paired_vs_fp16')
        if paired:
            _d = paired['delta_mean_nll_mean']
            _p = paired['p_value_min']
            logger.info(
                f"{'':>12}   paired vs fp16: "
                f"{paired['n_seeds_significant_worse']}/"
                f"{paired['n_seeds_tested']} seeds significant worse, "
                f"delta NLL mean "
                f"{'n/a' if _d is None else f'{_d:+.4f}'} "
                f"(p_min {'n/a' if _p is None else f'{_p:.4g}'})")


def _print_qat(rows):
    logger.info('\n-- Part B: QAT recovery (2-bit state writes, STE) --')
    for row in rows:
        if row['row_kind'] != 'qat_recovery':
            continue
        rec = row.get('recovery_fraction_mean')
        rec_str = f'{rec:.2%}' if rec is not None else 'n/a'
        rec_m = row.get('recovery_fraction_matched_mean')
        rec_m_str = f'{rec_m:.2%}' if rec_m is not None else 'n/a'
        logger.info(
            f"  fp16 {row['fp16_ppl_mean']:.4f} | PTQ {row['ptq_ppl_mean']:.4f} "
            f"| QAT {row['qat_ppl_mean']:.4f} | "
            f"fp16-matched {row.get('fp16_matched_ppl_mean') or float('nan'):.4f} "
            f"| recovery {rec_str} (matched {rec_m_str}) | "
            f"budgets_equal={row.get('historical_budget_matched')}")


def main():
    info = configure_torch_for_device()
    device = info.device
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info('=' * 70)
    logger.info('Experiment 11: state-quantized decode perplexity + QAT '
                'recovery')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')

    seeds = _ints(os.environ.get(
        'QLM_SEEDS', ','.join(map(str, DEFAULT_SEEDS))), 'QLM_SEEDS', 0)
    steps = int(os.environ.get('QLM_STEPS', DEFAULT_STEPS))
    qat_steps = int(os.environ.get('QLM_QAT_STEPS', DEFAULT_QAT_STEPS))
    seq_len = int(os.environ.get('QLM_SEQ_LEN', DEFAULT_SEQ_LEN))
    batch_size = int(os.environ.get('QLM_BATCH', DEFAULT_BATCH))
    eval_seqs = int(os.environ.get('QLM_EVAL_SEQS', DEFAULT_EVAL_SEQS))
    n_corpus = int(os.environ.get('QLM_CORPUS', DEFAULT_CORPUS))
    skip_qat = os.environ.get('QLM_SKIP_QAT', '0').strip().lower() in (
        '1', 'true', 'yes', 'y', 'on')
    tolerance = float(os.environ.get('QLM_PPL_TOL', DEFAULT_TOLERANCE))
    if min(steps, qat_steps, seq_len, batch_size, eval_seqs, n_corpus) < 1:
        raise ValueError('QLM_* step/geometry knobs must be >= 1')
    bits = _ints(os.environ.get(
        'QLM_BITS', ','.join(map(str, DEFAULT_BITS))), 'QLM_BITS', 2)
    groups = _ints(os.environ.get(
        'QLM_GROUP_SIZES', ','.join(map(str, DEFAULT_GROUP_SIZES))),
        'QLM_GROUP_SIZES', 1)
    specs = [s for s in default_sweep_specs(bits, groups,
                                            include_asymmetric=True)
             if s is not None]
    # Bootstrap replicates for the paired quantized-vs-fp16 test over the
    # shared held-out windows (pure Python, no extra forwards).
    pp_boot = parse_int_env('QLM_PPL_BOOT', 2000, min_value=1)
    # Held-out curve + bad-case analysis (read-only; cannot change weights).
    # Default: four evenly spaced validation checks per arm.
    val_every = parse_int_env('QLM_VAL_EVERY', max(1, steps // 4),
                              min_value=0, logger=logger)
    error_batches = parse_int_env(
        'QLM_ERROR_BATCHES', max(1, min(int(eval_seqs), 4)), min_value=0,
        logger=logger)
    max_error_examples = parse_int_env('QLM_ERROR_EXAMPLES', 2,
                                       min_value=0, logger=logger)
    logger.info(f'  seeds: {list(seeds)}, steps: {steps}, '
                f'qat_steps: {qat_steps}, seq_len: {seq_len}, '
                f'batch: {batch_size}, eval_seqs: {eval_seqs}')
    logger.info(f'  validation monitor: every {val_every} steps; bad-case '
                f'batches: {error_batches}')
    logger.info(f'  specs: {len(specs)} quantized cells + fp16 '
                f'(paired bootstrap n_boot={pp_boot})')

    tokenizer, split_texts, split_report = build_corpus(
        seq_len=seq_len, n_texts=n_corpus, seed=0)
    logger.info(f'  corpus: synthetic fallback, '
                f'splits={ {k: len(v) for k, v in split_texts.items()} }')

    # Pre-training overfit sanity gates: each part must be able to drive a
    # fixed small batch to a collapsing loss (Part B additionally requires
    # finite STE gradients) before the seed sweep burns compute.
    gate_record, proceed = run_overfit_gates(
        device=device, tokenizer=tokenizer, split_texts=split_texts,
        seq_len=seq_len, batch_size=batch_size, skip_qat=skip_qat)
    if not proceed:
        return 1

    ptq_per_seed, train_reports = run_part_a(
        device=device, tokenizer=tokenizer, split_texts=split_texts,
        seeds=seeds, specs=specs, steps=steps, seq_len=seq_len,
        batch_size=batch_size, eval_seqs=eval_seqs, tolerance=tolerance,
        val_every=val_every, error_batches=error_batches,
        max_error_examples=max_error_examples,
        # Confirmatory only: test is opened after every validation-based
        # decision in this run has been made.
        test_seqs=eval_seqs)
    ptq_rows = build_ptq_rows(ptq_per_seed, [None] + specs, tolerance,
                              n_boot=pp_boot)
    ptq_test_rows = build_ptq_test_rows(train_reports, specs, tolerance)
    _print_ptq(ptq_rows)
    if ptq_test_rows:
        logger.info('\n-- Part A: confirmatory TEST decode perplexity '
                    '(never used for selection) --')
        for row in ptq_test_rows:
            label = ('fp16' if not row['quantized']
                     else f"{row['bits']}b/g{row['group_size']}")
            logger.info(
                f"  {label:>12} | test decode ppl {row['decode_ppl_mean']:.4f} "
                f"[{row['decode_ppl_min']:.4f}..{row['decode_ppl_max']:.4f}] "
                f"| ratio vs test fp16 {row['ppl_ratio_mean']:.4f} | "
                f"n_tokens/seed {row['n_tokens_per_seed']}")

    qat_rows: list = []
    if not skip_qat:
        qat_per_seed = run_part_b(
            device=device, tokenizer=tokenizer, split_texts=split_texts,
            seeds=seeds, qat_steps=qat_steps, seq_len=seq_len,
            batch_size=batch_size, eval_seqs=eval_seqs, spec=QAT_SPEC,
            steps=steps, val_every=val_every)
        qat_rows = build_qat_rows(qat_per_seed, QAT_SPEC)
        _print_qat(qat_rows)

    plt_rows = []
    for cell in ptq_per_seed:
        row = {k: v for k, v in cell.items() if k != 'spec'}
        row['row_kind'] = 'ptq_per_seed'
        row.update(_spec_fields(cell['spec']))
        plt_rows.append(row)

    metadata = {
        'experiment': 'exp11_quant_lm',
        'design': (
            'End-task perplexity evaluation of state quantization: Part A '
            'trains the tiny hybrid LM (2 KDA + 1 CSA + 1 HCA) on the '
            'deterministic offline synthetic corpus and scores held-out '
            'token NLL token-by-token through run_decoding.HybridDecoding '
            'with the stacked KDA state fake-quantized after every write '
            '(Exp 10 grid, flat per-layer grouping); Part B trains a '
            'standalone per-token KDA LM twice from the same init '
            '(fp16 vs 2-bit STE state writes) to measure whether QAT '
            'recovers the PTQ perplexity damage through the quantizer. '
            'When QLM_STEPS != QLM_QAT_STEPS, an additional fp16 control '
            'is trained for exactly QLM_QAT_STEPS from the same '
            'seed/init/data/optimizer/LR, and the matched-budget recovery '
            'is reported so extra optimization is not credited to QAT.'),
        'model': _cfg_summary(small_hybrid_config()),
        'corpus': {
            'source': 'synthetic fallback (deterministic, offline; no '
                      'network / HuggingFace tokenizer)',
            'tokenizer': '_FallbackCharTokenizer (printable ASCII)',
            'split': split_report,
        },
        'protocol': {
            'decode_path': 'run_decoding.HybridDecoding, one token per '
                           'call from an empty state (prefill chunk path '
                           'only used for the prefill_ppl reference)',
            'quantized_tensor': 'clone.model._kda_state (stacked KDA '
                                'state), grouped along the flattened '
                                'per-layer buffer',
            'delta_normalization': 'ppl_ratio = decode_ppl(spec) / '
                                   'decode_ppl(fp16) on the SAME path and '
                                   'seed; min-over-seeds for the '
                                   'ppl_preserving flag',
            'paired_bootstrap': (
                'quantized vs fp16 scored on the SAME held-out windows, '
                'so the paired unit-level bootstrap '
                '(ppl_bootstrap.paired_bootstrap_ppl) separates the '
                'quantization effect from window-sampling noise: CI for '
                'delta_mean_nll = quantized - fp16 and a two-sided '
                'centered bootstrap p-value, per seed; the seed-level '
                'aggregate is in each ptq_ppl row as paired_vs_fp16. '
                f'n_boot={pp_boot} (env QLM_PPL_BOOT)'),
            'tolerance': tolerance,
            'validation_monitor': {
                'val_every': int(val_every),
                'bad_case_batches': int(error_batches),
                'bad_case_examples_per_category': int(max_error_examples),
                'note': ('Held-out (loss, token-accuracy) curve during '
                         'training and token-level bad-case categories on '
                         'the same validation split; read-only and '
                         'RNG-neutral, so enabling them cannot change a '
                         'reported PPL.'),
            },
            'test_split': {
                'used': bool(ptq_test_rows),
                'eval_sequences': int(eval_seqs),
                'note': ('Part A scores the sealed test split once per '
                         'seed after the spec grid, tolerance and '
                         'validation protocol are frozen. The resulting '
                         'ptq_test_ppl rows are marked '
                         'used_for_selection=False and do not feed back '
                         'into any row used for selection.'),
            },
            'qat': 'straight-through fake_quantize (ste=True) on every '
                   'state write during training; inference scoring uses '
                   'ste=False on the same spec. The primary QAT contrast '
                   'is against the equally trained fp16 control '
                   '(recovery_fraction_matched / qat_vs_matched_control), '
                   'never against a shorter-budget fp16 arm',
        },
        'train_reports': [
            {k: v for k, v in rep.items() if k != 'cfg'}
            for rep in train_reports],
        'seeds': [int(s) for s in seeds],
        'seed_scheme': (
            'set_global_seed(seed, deterministic=True) at the top of both '
            'train_hybrid_lm and train_stepwise_lm; the training DataLoader '
            'is seeded with generator=make_seeded_generator(seed) and '
            'worker_init_fn=seed_worker; the corpus build uses seed=0'),
        'steps': steps,
        'qat_steps': qat_steps,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'eval_sequences': eval_seqs,
        'specs': [spec_summary(s) for s in specs],
        'qat_spec': spec_summary(QAT_SPEC),
        'caveats': [
            'Toy corpus and model: absolute perplexities are not '
            'comparable to real-corpus LM numbers; the measured quantity '
            'is the relative damage of state quantization at fixed '
            'weights and path.',
            'Only the KDA recurrent state is quantized; CSA/HCA '
            'incremental caches, short-conv lookback, activations, '
            'weights and logits stay fp16/fp32.',
            'The fp16 prefill path (chunked KDA) and the fp16 decode path '
            '(recurrent KDA) differ by kernel/reduction order; all '
            'quantization deltas are measured against decode fp16 on the '
            'same path.',
            'Part B demonstrates the autograd/STE contract end-to-end on '
            'a 1-layer from-scratch model; it is not a production QAT '
            'recipe and recovery is seed-dependent.',
            'ppl_preserving is a tolerance flag (min-over-seeds ratio <= '
            f'1 + {tolerance:.2%}), not an exactness claim.',
        ],
        'provenance': capture_provenance(),
    }
    # Overfit-gate verdict: compact form in the metadata next to the rows,
    # full record (per-step curves, batch fingerprint) in provenance.
    metadata['overfit_sanity_check'] = {
        'enabled': gate_record['enabled'],
        'strict': gate_record.get('strict'),
        'steps': gate_record.get('steps'),
        'seed': gate_record.get('seed'),
        'train_batch': gate_record.get('train_batch'),
        'reports': [_compact_qlm_gate_report(r)
                    for r in gate_record.get('reports', [])],
    }
    metadata['provenance']['overfit_sanity_check'] = gate_record

    rows = plt_rows + ptq_rows + qat_rows + ptq_test_rows
    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, rows, kind='exp11_quant_lm',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')

    # Campaign-level ledger entry: seeds / recipe / corpus fingerprints /
    # gate verdict / artifact path in one strict-JSON line.
    gate_failures = sum(1 for r in gate_record.get('reports', [])
                        if not r.get('passed'))
    append_run_manifest(make_run_manifest_entry(
        experiment='exp11_quant_lm',
        seeds=list(seeds),
        provenance=metadata['provenance'],
        hyperparameters={
            'steps': int(steps),
            'qat_steps': int(qat_steps),
            'seq_len': int(seq_len),
            'batch_size': int(batch_size),
            'eval_sequences': int(eval_seqs),
            'n_corpus_texts': int(n_corpus),
            'lr': 3e-3,
            'tolerance': float(tolerance),
            'val_every': int(val_every),
            'error_batches': int(error_batches),
            'max_error_examples': int(max_error_examples),
            'qat_enabled': not bool(skip_qat),
            'specs': [spec_summary(s) for s in specs],
            'qat_spec': spec_summary(QAT_SPEC),
        },
        data={
            'source': metadata['corpus']['source'],
            'split': split_report,
            'sizes': {name: len(texts)
                      for name, texts in split_texts.items()},
            'fingerprints': {
                name: _texts_fingerprint(texts)
                for name, texts in split_texts.items()},
        },
        artifacts={'results': RESULTS_PATH},
        results={
            'n_rows': len(rows),
            'n_ptq_rows': len(ptq_rows),
            'n_ptq_test_rows': len(ptq_test_rows),
            'n_qat_rows': len(qat_rows),
            'n_specs': len(specs),
            'n_seeds': len(seeds),
            'overfit_gate_enabled': bool(gate_record['enabled']),
            'overfit_gate_failures': int(gate_failures),
        },
        status=('completed' if gate_failures == 0 else 'partial'),
    ), logger=logger)
    return 0


if __name__ == '__main__':
    sys.exit(main())
