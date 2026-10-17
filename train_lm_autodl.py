"""
LM training for the KDA+CSA+HCA hybrid, built on ``ops_fused``.

The model uses ``HybridKCHAttention`` from ``ops_fused.py``, which already
implements:
- KDA with proper unit-norm q/k, g clamp, chunked path, conv lookback
- CSA with overlapped 2-branch compression, STE for indexer, sink logits
- HCA with heavy compression + dense + sliding window
- Decoding caches (ops_decoding_cache) for efficient long-context

Defaults: 2000 steps, d_model=256, 5 layers (3:1:1), seq 1024 — sized for a
small single-GPU run.

Data protocol (leakage-safe): the corpus is cleaned (empty / exact-
duplicate / unsupervised documents dropped) and then partitioned into
train / validation / test BEFORE any model runs. Training only touches
train; validation drives model selection and early stopping; the test
split stays sealed and is opened exactly once at the end, on the frozen
val-selected checkpoint. See ``clean_lm_texts`` / ``split_clean_texts``
and ``SealedTestSet``.

Results analysis (what the run record answers beyond "final loss"):
``training_curve`` + per-check validation history feed a streaming
curve-shape verdict (overfitting / underfitting / healthy / diverged) and
a final ``curve_diagnosis`` with next actions; ``error_analysis`` is a
token-level bad-case report on the held-out validation split (near-miss /
confident blind spot / copyable-missed / novel-token categories plus
accuracy by position and by copy distance, with decoded examples); and
``efficiency`` records params, throughput, evaluation latency and peak
memory next to the quality numbers, so a gain can never be claimed
without its cost.
"""
import os
import sys
import time
import argparse
import gc
import hashlib
import datetime
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
except ImportError:  # tqdm is preinstalled on Kaggle/AutoDL but is
    # optional for a minimal local run — fall back to a no-op wrapper.
    def tqdm(*args, **kwargs):
        class _NoOpBar:
            def __init__(self, *a, **k):
                self.total = k.get("total", None)
                self.n = 0
            def update(self, n=1):
                self.n += n
            def close(self):
                pass
            def set_description(self, *a, **k):
                pass
            def set_postfix(self, *a, **k):
                # Called every optimizer step (``pbar.set_postfix``); must
                # exist or the no-tqdm fallback crashes the training loop.
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                self.close()
        return _NoOpBar(*args, **kwargs)

# import repo modules
from ops_fused import HybridConfig, HybridKCHAttention
from ops_kda_backend import validate_kda_backend
from kaggle_setup import (configure_torch_for_device, set_global_seed,
                          seed_worker, capture_provenance,
                          write_json_atomic, write_results_json,
                          split_train_val_test, audit_split)
from results_analysis import (LM_ERROR_CATEGORIES, diagnose_nonfinite,
                              diagnose_training_curves, lm_token_diagnostics,
                              summarize_lm_diagnostics)
from ppl_bootstrap import bootstrap_ppl_ci, throughput_summary

# tokenizer / dataset handling (reuse TinyStories if available)
try:
    from transformers import AutoTokenizer
    from datasets import load_dataset
    HAS_HF = True
except Exception:
    # ImportError on hosts without the HF stack; also catches transient
    # import-time failures (e.g. a half-installed datasets). A bare
    # ``except:`` here would also swallow KeyboardInterrupt / SystemExit
    # during interpreter shutdown, so scope to Exception.
    HAS_HF = False


class _FallbackCharTokenizer:
    """Minimal char-level tokenizer used when ``transformers`` is absent.

    Keeps the training pipeline runnable with ZERO network / HuggingFace
    dependencies (e.g. a bare Kaggle CPU session without the optional
    ``pip install transformers datasets`` step, or an offline machine):
    the dataset falls back to synthetic text and this tokenizer maps
    printable ASCII characters to ids. Quality is obviously worse than a
    real BPE tokenizer — it exists only so the code path executes
    end-to-end; install ``transformers`` for the real TinyStories runs.
    """

    _SPECIALS = ["<pad>", "<unk>"]

    def __init__(self):
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        self._itos = self._SPECIALS + chars
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.pad_token_id = 0
        self.unk_token_id = 1

    def __len__(self):
        return len(self._itos)

    def encode(self, text, truncation=False, max_length=None):
        ids = [self._stoi.get(ch, self.unk_token_id) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        """Inverse of :meth:`encode` (bad-case analysis needs readable text).

        Analysis-only: training never calls this, so adding it cannot change
        any numeric result.
        """
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and 0 <= i < len(self._SPECIALS):
                continue
            out.append(self._itos[i] if 0 <= i < len(self._itos) else '')
        return ''.join(out)

def _synthetic_corpus(n=1000):
    """Deterministic synthetic fallback corpus (unique texts).

    The texts are unique (index suffix) so the cleaning step's exact-
    duplicate removal and the subsequent train/val/test split stay
    meaningful even on the no-network path: a corpus that repeated the
    same string N times would collapse to a single story after dedup and
    make a three-way split degenerate.
    """
    return [f"Once upon a time, story {i}. " * 40 for i in range(int(n))]


def _load_raw_texts(max_samples, split="train"):
    """Load the raw text corpus (HF TinyStories or the synthetic fallback).

    Returns ``(texts, data_source)``. Cleaning and splitting happen
    AFTER this call and BEFORE any model sees a token (see
    ``clean_lm_texts`` / ``split_clean_texts``).
    """
    if HAS_HF:
        try:
            ds = load_dataset("roneneldan/TinyStories", split=split)
            if max_samples:
                ds = ds.select(range(min(len(ds), int(max_samples))))
            return list(ds["text"]), "hf:roneneldan/TinyStories"
        except Exception as e:
            print(f"HF load failed {e}, using synthetic")
            return (_synthetic_corpus(),
                    "synthetic:once_upon_a_time (HF load failed)")
    return _synthetic_corpus(), "synthetic:once_upon_a_time (no transformers)"


class TinyStoriesLM(Dataset):
    def __init__(self, tokenizer, seq_len=1024, max_samples=20000, split="train",
                 texts=None, name=None):
        """Token-window dataset over a list of raw stories.

        When ``texts`` is None the corpus is loaded from HuggingFace (or
        the synthetic fallback), capped at ``max_samples``. The training
        entry point instead loads, cleans and partitions the corpus
        explicitly and passes the pre-sliced ``texts`` of ONE split here,
        so no dataset object ever spans the train/val/test boundary.
        """
        self.seq_len = seq_len
        self.split = split
        if texts is None:
            texts, source = _load_raw_texts(max_samples, split)
            self.max_samples = max_samples
        else:
            texts = list(texts)
            source = f"provided:{name or split}"
            self.max_samples = len(texts)
        self.texts = texts
        self.data_source = source
        self.tokenizer = tokenizer
        # Content fingerprint so a run record can identify WHICH data was
        # trained on. Uses a bounded, deterministic sample (first
        # ``n_probe`` texts truncated per-text); the sample is enough to
        # detect a swapped split / stale cache.
        self.data_fingerprint = _texts_fingerprint(self.texts)

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.seq_len + 1)
        real_len = min(len(tokens), self.seq_len + 1)
        if real_len < self.seq_len + 1:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - real_len)
        else:
            tokens = tokens[:self.seq_len + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        if real_len <= 1:
            labels[:] = -100
        elif real_len - 1 < labels.numel():
            labels[real_len - 1:] = -100
        return {"input_ids": input_ids, "labels": labels}

# --------------------------------------------------------------------------
# Data cleaning + leakage-safe splitting
# --------------------------------------------------------------------------
#
# Order of operations in ``main`` (matters for leakage):
#
#   1. load the raw corpus;
#   2. clean it (drop empty / exactly duplicated / unsupervised texts);
#   3. draw the deterministic train/val/test split;
#   4. verify the split (``audit_split`` + cross-split text-digest check);
#   5. train on TRAIN only, monitor/select on VAL only;
#   6. after the model is frozen, open the sealed TEST set exactly once.
#
# No statistic used to transform features may be fitted on val/test. This
# pipeline fits no preprocessing transform at all (the tokenizer is the
# pretrained GPT-2 tokenizer, not fitted here), but the split is still
# drawn first so that any future fitted transform (e.g. a learned
# vocabulary or token-length normalizer) is forced to use train indices
# only. Per-split summary statistics below are computed for REPORTING
# distribution shift, never fed back into training or model selection.

def _text_digests(texts):
    """Content digests of a list of texts (exact-duplicate identity)."""
    return {hashlib.sha256(str(t).encode('utf-8', 'replace')).digest()
            for t in texts}


def clean_lm_texts(texts, tokenizer, seq_len, *, dedup=True, min_tokens=2):
    """Clean the raw corpus BEFORE any split is drawn.

    Cleaning steps (all recorded in the returned stats):

      * drop empty / whitespace-only documents;
      * drop exact duplicates (content hash) so the same story can never
        end up on both sides of the split — dedup BEFORE splitting is
        what makes the cross-split disjointness guarantee possible;
      * drop documents that tokenize to fewer than ``min_tokens`` ids:
        they produce no supervised label at all (every target is masked
        with -100), i.e. they are pure label noise for a next-token
        objective;
      * drop documents the tokenizer cannot encode (counted separately).

    Returns ``{'texts': [...], 'stats': {...}}``. The stats are
    JSON-serializable and go straight into the run record.
    """
    texts = list(texts)
    clean, seen = [], set()
    n_empty = n_duplicates = n_too_short = n_tokenize_errors = 0
    for text in texts:
        s = str(text)
        if not s.strip():
            n_empty += 1
            continue
        if dedup:
            key = hashlib.sha256(s.encode('utf-8', 'replace')).digest()
            if key in seen:
                n_duplicates += 1
                continue
            seen.add(key)
        try:
            ids = tokenizer.encode(s, truncation=True,
                                   max_length=seq_len + 1)
        except Exception:
            n_tokenize_errors += 1
            continue
        if len(ids) < int(min_tokens):
            n_too_short += 1
            continue
        clean.append(s)
    n_raw = len(texts)
    stats = {
        'n_raw': n_raw,
        'n_kept': len(clean),
        'n_empty': n_empty,
        'n_duplicates': n_duplicates,
        'n_too_short_to_supervise': n_too_short,
        'n_tokenize_errors': n_tokenize_errors,
        'dedup': bool(dedup),
        'min_tokens': int(min_tokens),
        'kept_fraction': (len(clean) / n_raw) if n_raw else 0.0,
    }
    return {'texts': clean, 'stats': stats}


def split_clean_texts(texts, *, val_frac=0.1, test_frac=0.1, seed=0):
    """Draw and audit the train/val/test split over CLEANED texts.

    The split itself is deterministic (see
    ``kaggle_setup.split_train_val_test``). On top of the structural
    audit this verifies that the three text partitions share no exact
    duplicate — a defense-in-depth check against a future refactor that
    moves dedup after the split.

    Returns ``(split, report)`` where ``split`` maps split names to index
    lists and ``report`` is the JSON-serializable audit + overlap report.
    Raises ``RuntimeError`` on any leakage violation.
    """
    split = split_train_val_test(len(texts), val_frac=val_frac,
                                 test_frac=test_frac, seed=seed)
    audit = audit_split(split, len(texts))
    by_split = {k: [texts[i] for i in split[k]]
                for k in ('train', 'val', 'test')}
    digests = {k: _text_digests(v) for k, v in by_split.items()}
    overlaps = {
        'train_val': len(digests['train'] & digests['val']),
        'train_test': len(digests['train'] & digests['test']),
        'val_test': len(digests['val'] & digests['test']),
    }
    report = {
        'seed': int(seed),
        'val_frac': float(val_frac),
        'test_frac': float(test_frac),
        'audit': audit,
        'exact_text_overlaps': overlaps,
    }
    problems = list(audit['problems'])
    for pair, n_overlap in overlaps.items():
        if n_overlap:
            problems.append(
                f'{pair}: {n_overlap} identical text(s) shared across splits')
    if problems:
        raise RuntimeError(
            'train/val/test split failed its leakage audit: '
            + '; '.join(problems))
    return split, report


def _text_length_stats(texts, tokenizer, seq_len, *, n_probe=256):
    """Descriptive token-length stats for ONE split (reporting only).

    Computed on a bounded, evenly spaced probe so it stays cheap on a
    20k-story corpus. This is a distribution-shift DIAGNOSTIC: it is
    reported in the run record and never used to fit a transformation or
    to select a model.
    """
    n = len(texts)
    if n == 0:
        return {'n': 0, 'n_probe': 0, 'mean_tokens': None,
                'std_tokens': None, 'min_tokens': None, 'max_tokens': None,
                'cap_frac': None, 'unk_rate': None}
    if n <= n_probe:
        probe = list(texts)
    else:
        stride = max(1, n // n_probe)
        probe = [texts[i] for i in range(0, n, stride)][:n_probe]
    lengths, unk, total = [], 0, 0
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    for text in probe:
        try:
            ids = tokenizer.encode(text, truncation=True,
                                   max_length=seq_len + 1)
        except Exception:
            continue
        lengths.append(len(ids))
        if unk_id is not None:
            unk += sum(1 for i in ids if i == unk_id)
            total += len(ids)
    if not lengths:
        return {'n': n, 'n_probe': 0, 'mean_tokens': None,
                'std_tokens': None, 'min_tokens': None, 'max_tokens': None,
                'cap_frac': None, 'unk_rate': None}
    mean = sum(lengths) / len(lengths)
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return {
        'n': n,
        'n_probe': len(lengths),
        'mean_tokens': mean,
        'std_tokens': math.sqrt(var),
        'min_tokens': min(lengths),
        'max_tokens': max(lengths),
        # Proxy for "truncated at the model context": a text encoded with
        # max_length=seq_len+1 that comes back at the cap may be longer.
        'cap_frac': sum(1 for x in lengths if x >= seq_len + 1) / len(lengths),
        'unk_rate': (unk / total) if total else 0.0,
    }


def _distribution_shift(train_stats, other_stats, *, threshold=0.5):
    """Standardized mean difference (Cohen's d) of token lengths vs train.

    Reports (does not act): a large |d| is a warning that the held-out
    split is not exchangeable with train, which would make the reported
    held-out loss hard to interpret. The threshold only sets the
    ``flag`` field.
    """
    m_t, m_o = train_stats.get('mean_tokens'), other_stats.get('mean_tokens')
    if m_t is None or m_o is None:
        return {'reference': 'train', 'cohen_d': None, 'abs_cohen_d': None,
                'flag': None, 'threshold': float(threshold)}
    s_t = train_stats.get('std_tokens') or 0.0
    s_o = other_stats.get('std_tokens') or 0.0
    pooled = math.sqrt((s_t ** 2 + s_o ** 2) / 2.0)
    d = (m_o - m_t) / pooled if pooled > 0 else 0.0
    return {
        'reference': 'train',
        'cohen_d': d,
        'abs_cohen_d': abs(d),
        'flag': abs(d) >= threshold,
        'threshold': float(threshold),
    }


class SealedTestSet:
    """One-shot accessor for the held-out test split.

    The test set must stay untouched until the final model (selected on
    validation) is frozen, then be evaluated exactly once: each extra
    look turns it into an implicit training signal. ``open()`` enforces
    the contract mechanically — a second call raises instead of quietly
    re-evaluating.
    """

    def __init__(self, factory):
        if not callable(factory):
            raise TypeError('SealedTestSet needs a zero-arg factory')
        self._factory = factory
        self._opened = False

    @property
    def opened(self):
        return self._opened

    def open(self):
        if self._opened:
            raise RuntimeError(
                'the test split has already been opened; it must be '
                'evaluated exactly once, after validation-based model '
                'selection and early stopping are finished')
        self._opened = True
        return self._factory()


@torch.no_grad()
def evaluate_lm_loss(model, dataset, *, batch_size=1, device='cpu',
                     max_batches=None, use_amp=False, autocast_dtype=None,
                     return_units=False):
    """Token-weighted mean cross-entropy loss of ``model`` on ``dataset``.

    Deterministic, read-only evaluation used for validation monitoring,
    early stopping and the single final test evaluation:

      * ``shuffle=False``: the same items in the same order every call,
        so a validation-loss curve across checkpoints is comparable and
        the result is reproducible;
      * ``torch.no_grad()`` + ``model.eval()``: no gradient, no weight
        update, no optimizer step — evaluation can never fit anything;
      * the DataLoader is given a DEDICATED generator: PyTorch draws one
        base-seed number per iterator construction even with
        ``shuffle=False``/``num_workers=0``, and without this it would
        come from the global torch RNG and perturb the training stream.
        The dedicated generator is created fresh from a fixed seed each
        call, so evaluation is deterministic AND RNG-neutral;
      * ``max_batches`` caps the token budget; the mean is weighted by
        the number of UNMASKED tokens, so a capped pass is still an
        unbiased token-level estimate of the full split.

    ``return_units=True`` additionally attaches ``per_unit`` — one
    ``{index, nll_sum, n_tokens}`` record per scored dataset item, in
    dataset order — the sufficient statistic for the unit-level
    bootstrap inference in ``ppl_bootstrap`` (CI / paired tests for
    perplexity comparisons). In that mode the reported ``loss`` is
    rebuilt by summing the unit NLL sums (per-token cross-entropy from
    ``out['logits']``, ignore_index=-100), so the point estimate and the
    bootstrap inputs are the SAME arithmetic; the two paths agree within
    float rounding. Batches with no supervised tokens contribute no
    units (they also contributed nothing before).

    Returns a JSON-serializable report. ``loss`` / ``perplexity`` are
    ``None`` when the dataset has no supervised tokens.
    """
    if isinstance(device, str):
        device = torch.device(device)
    eval_gen = torch.Generator()
    eval_gen.manual_seed(0)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                        num_workers=0, generator=eval_gen)
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    n_batches = n_sequences = 0
    per_unit: list[dict] = []
    unit_offset = 0
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            n_tok = int((labels != -100).sum())
            if n_tok == 0:
                unit_offset += int(labels.shape[0])
                continue
            model.hybrid.reset_state()
            with torch.autocast(device_type=device.type,
                                dtype=autocast_dtype or torch.float32,
                                enabled=bool(use_amp)):
                out = model(input_ids, labels=labels)
            loss = out["loss"]
            if loss is None or not torch.isfinite(loss):
                # Held-out evaluation diverged: name the first failing stage
                # (data ids -> loss -> parameters) so the refusal to report
                # a metric is actionable.
                diag = diagnose_nonfinite(
                    loss=(loss if loss is not None else float('nan')),
                    token_ids={'input_ids': input_ids},
                    id_range=(0, int(getattr(
                        getattr(model, 'embed', None),
                        'num_embeddings', input_ids.max().item() + 1))),
                    params=model.parameters(),
                    amp_enabled=bool(use_amp),
                    context='held-out evaluation')
                raise RuntimeError(
                    f'non-finite held-out LM loss ({loss}) during '
                    f'evaluation; refusing to report a corrupted metric. '
                    f'Diagnosis [{diag["verdict"]}]: {diag["summary"]}')
            if return_units:
                # Per-item NLL sums for the unit-level bootstrap. The
                # logits are recomputed through the SAME ignore_index
                # cross-entropy the model uses internally; positions with
                # label -100 contribute exactly 0 to both the sum and
                # (via the mask) the token count.
                logits = out["logits"]
                flat_logits = logits.reshape(-1, logits.size(-1))
                tok_nll = F.cross_entropy(
                    flat_logits.float(), labels.reshape(-1),
                    ignore_index=-100, reduction='none').view(labels.shape)
                row_sums = tok_nll.sum(dim=1)
                row_counts = (labels != -100).sum(dim=1)
                for b in range(int(labels.shape[0])):
                    per_unit.append({
                        'index': unit_offset + b,
                        'nll_sum': float(row_sums[b]),
                        'n_tokens': int(row_counts[b]),
                    })
                # Rebuild the total from the units so the point estimate
                # and the bootstrap inputs are the same arithmetic.
                total_nll += float(row_sums.sum())
            else:
                total_nll += float(loss.detach()) * n_tok
            total_tokens += n_tok
            unit_offset += int(labels.shape[0])
            n_batches += 1
            n_sequences += int(input_ids.shape[0])
            if max_batches is not None and n_batches >= int(max_batches):
                break
    finally:
        model.train(was_training)
    mean_loss = (total_nll / total_tokens) if total_tokens else None
    report = {
        'loss': mean_loss,
        'perplexity': (math.exp(min(mean_loss, 700.0))
                       if mean_loss is not None else None),
        'n_batches': n_batches,
        'n_sequences': n_sequences,
        'n_tokens': total_tokens,
        'max_batches': (None if max_batches is None else int(max_batches)),
        'batch_size': int(batch_size),
        # Read-only contract: spelled out so an artifact reader can verify
        # the metric's provenance.
        'shuffle': False,
        'grad_enabled': False,
    }
    if return_units:
        report['per_unit'] = per_unit
        report['per_unit_contract'] = (
            'one record per scored dataset item in dataset order: '
            '{index, nll_sum, n_tokens}; consumed by '
            'ppl_bootstrap.bootstrap_ppl_ci / paired_bootstrap_ppl; '
            'records with n_tokens == 0 are dropped by normalize_units')
    return report


def _decode_tokens(tokenizer, ids):
    """Best-effort token-id -> text for bad-case example display.

    Returns ``None`` when the tokenizer has no usable ``decode``; the example
    record then still carries the raw ids.
    """
    if tokenizer is None or not hasattr(tokenizer, 'decode'):
        return None
    try:
        return tokenizer.decode([int(i) for i in ids],
                                skip_special_tokens=False)
    except Exception:
        return None


def _peak_memory_bytes():
    """Best-effort peak memory of the current process.

    CUDA runs report the allocator high-water mark. CPU runs report OS RSS
    (``ru_maxrss``), which is an UPPER BOUND on the model's footprint: it
    includes the interpreter, imported libraries and data. The measurement
    is labelled in the run record so it is never compared as if it were a
    pure model-size number.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux/most Unices report KiB; macOS reports bytes.
        return rss if sys.platform == 'darwin' else rss * 1024
    except Exception:
        return None


@torch.no_grad()
def analyze_lm_errors(model, dataset, *, batch_size=1, device='cpu',
                      max_batches=2, seed=0, tokenizer=None, max_examples=3,
                      top_k=5, confidence=0.5, context_tokens=24):
    """Bad-case analysis: inspect mispredicted tokens on a held-out split.

    This is the LM-scale counterpart of ``run_quality.analyze_errors``:
    natural text has no exact task labels, so every wrong next-token
    prediction is categorized by WHAT INFORMATION the model failed to use
    (see ``results_analysis.LM_ERROR_CATEGORIES``):

      * ``near_miss`` — target in the top-k, top-1 disagreed;
      * ``confident_wrong`` — wrong top-1 with high confidence (bias/blind
        spot);
      * ``copyable_missed`` — target DID appear earlier in the context but
        ranked outside the top-k (retrieval/copy failure — the actionable
        class for a long-context memory architecture);
      * ``novel_token`` — target never appeared in the context (intrinsic
        prediction difficulty).

    The report also slices accuracy by relative position (cold start vs
    steady state) and by distance to the target's last earlier occurrence
    (1-8 / 9-32 / 33-128 / 129+ tokens, plus never-seen), which turns "the
    model is 30% wrong" into "it copies at distance 8 but not at 128" — the
    evidence an architecture change needs.

    Contract (same discipline as ``evaluate_lm_loss``): read-only
    (``no_grad`` + eval mode restored), deterministic (fixed generator;
    ``shuffle=False``), and RNG-neutral (dedicated DataLoader generator, no
    global RNG draws), so running it cannot change training or the sealed
    test metric. It must be called on a held-out split (validation), never
    on the sealed test set or on data the model memorized.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if int(max_batches) < 0:
        raise ValueError('max_batches must be >= 0')
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False,
                        num_workers=0, generator=gen)
    was_training = model.training
    model.eval()
    diags = []
    examples = OrderedDict((name, []) for name in LM_ERROR_CATEGORIES)
    n_batches = n_nonfinite = 0
    try:
        for batch in loader:
            if max_batches is not None and n_batches >= int(max_batches):
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if labels.numel() == 0 or not (labels != -100).any():
                continue
            model.hybrid.reset_state()
            logits = model(input_ids)["logits"]
            if not torch.isfinite(logits).all():
                n_nonfinite += 1
                continue
            d = lm_token_diagnostics(logits, labels, input_ids,
                                     top_k=top_k, confidence=confidence)
            diags.append(d)
            sample_base = n_batches * int(batch_size)
            for bi in range(int(input_ids.shape[0])):
                for t, cat in enumerate(d['category'][bi]):
                    if cat in ('correct', 'ignored'):
                        continue
                    bucket = examples[cat]
                    if len(bucket) >= int(max_examples):
                        continue
                    lo = max(0, t - int(context_tokens))
                    ctx_ids = [int(v) for v in input_ids[bi, lo:t].tolist()]
                    bucket.append({
                        'sample': sample_base + bi,
                        'position': int(t),
                        'seq_len': int(input_ids.shape[1]),
                        'target': int(labels[bi, t]),
                        'pred': int(d['pred'][bi, t]),
                        'target_rank': int(d['target_rank'][bi, t]),
                        'top1_prob': float(d['top1_prob'][bi, t]),
                        'repeat_distance': (
                            int(d['last_seen_distance'][bi, t])
                            if bool(d['seen_earlier'][bi, t]) else None),
                        'context_token_ids': ctx_ids,
                        'context_text': _decode_tokens(tokenizer, ctx_ids),
                        'target_text': _decode_tokens(
                            tokenizer, [int(labels[bi, t])]),
                        'pred_text': _decode_tokens(
                            tokenizer, [int(d['pred'][bi, t])]),
                    })
                    if len(bucket) > 0 and all(
                            len(examples[c]) >= int(max_examples)
                            for c in LM_ERROR_CATEGORIES
                            if c not in ('correct', 'ignored')):
                        break
            n_batches += 1
            if max_batches is not None and n_batches >= int(max_batches):
                break
    finally:
        if hasattr(model, 'hybrid') and hasattr(model.hybrid, 'reset_state'):
            model.hybrid.reset_state()
        model.train(was_training)

    report = summarize_lm_diagnostics(diags)
    report.update({
        'seed': int(seed),
        'generator_seed': int(seed),
        'split': getattr(dataset, 'split', 'held-out'),
        'n_batches_analyzed': int(n_batches),
        'n_batches_skipped_nonfinite': int(n_nonfinite),
        'batch_size': int(batch_size),
        'stream': ('held-out split, fixed order (shuffle=False, dedicated '
                   'DataLoader generator); never the sealed test set'),
        'top_k': int(top_k),
        'confidence': float(confidence),
        'examples': {k: v for k, v in examples.items() if v},
    })
    if n_nonfinite:
        report['note'] += (' Non-finite logits were found in '
                           f'{n_nonfinite} batch(es); run '
                           'diagnose_nonfinite() on that batch before '
                           'trusting any metric.')
    if diags:
        acc = report['token_accuracy']
        print(f"[bad-case] LM held-out ({report['split']}): "
              f"{report['n_wrong']}/{report['n_tokens']} tokens wrong "
              f"({100.0 * report['n_wrong'] / max(1, report['n_tokens']):.1f}%), "
              f"token_acc={acc:.4f}")
        mix = {c: v['count'] for c, v in report['categories'].items()
               if v['count']}
        print(f"[bad-case] categories: {mix}")
        print('[bad-case] accuracy by position quartile: ' + ', '.join(
            f"{b['quartile']}={b['accuracy']:.3f}(n={b['n']})"
            for b in report['accuracy_by_position_quartile']))
        print('[bad-case] accuracy by repeat distance: ' + ', '.join(
            f"{b['bucket']}="
            f"{'n/a' if b['accuracy'] is None else format(b['accuracy'], '.3f')}"
            f"(n={b['n']})"
            for b in report['accuracy_by_repeat_distance']))
        shown = 0
        for cat, bucket in report['examples'].items():
            if shown >= 3:
                break
            ex = bucket[0]
            ctx = ex['context_text']
            ctx_show = (f'...{ctx[-120:]}' if isinstance(ctx, str) and
                        len(ctx) > 120 else ctx)
            print(f"[bad-case] e.g. {cat}: pos={ex['position']} "
                  f"target={ex['target_text']!r} (id {ex['target']}) "
                  f"pred={ex['pred_text']!r} (id {ex['pred']}) "
                  f"rank={ex['target_rank']} top1p={ex['top1_prob']:.3f} "
                  f"repeat_distance={ex['repeat_distance']}")
            print(f"[bad-case]   context={ctx_show!r}"
                  if ctx_show is not None
                  else f"[bad-case]   context_ids={ex['context_token_ids']}")
            shown += 1
    else:
        print(f"[bad-case] no analyzable held-out batches "
              f"(analyzed={n_batches}, skipped_nonfinite={n_nonfinite})")
    return report


class LMWithHybrid(nn.Module):
    def __init__(self, vocab_size, cfg: HybridConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        # Total depth: the EXPLICIT layout is authoritative when set (the
        # eight-group matrix pins 10-layer layouts whose length is NOT
        # expressible as n_kda+n_csa+n_hca — using the ratio fields there
        # would silently build a 5-layer model instead of the configured
        # one). Ratio mode derives depth from the ratio fields.
        if cfg.layout is not None:
            total_layers = len(cfg.layout)
        else:
            total_layers = cfg.n_kda + cfg.n_csa + cfg.n_hca
        self.hybrid = HybridKCHAttention(cfg, total_layers=total_layers)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x = self.hybrid(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

def count_params(m):
    seen = set()
    total = 0
    for p in m.parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def _texts_fingerprint(texts, n_probe=64, max_chars_per_text=4096):
    """Bounded content fingerprint of the training texts (data version id).

    Hashes a deterministic sample of at most ``n_probe`` evenly spaced
    texts (each truncated to ``max_chars_per_text``) plus the corpus size,
    so a run record can detect a swapped split / stale dataset cache
    without hashing gigabytes on every launch.
    """
    n = len(texts)
    h = hashlib.sha256()
    h.update(
        f"n={n};probe={min(n, n_probe)};max_chars={max_chars_per_text}".encode())
    if n:
        stride = max(1, n // max(1, n_probe))
        seen = 0
        for i in range(0, n, stride):
            if seen >= n_probe:
                break
            h.update(str(texts[i])[:max_chars_per_text].encode(
                'utf-8', 'replace'))
            seen += 1
        # Always include the last sample so a truncation of the corpus (e.g.
        # a shorter stale cache) changes the fingerprint even if the stride
        # sample happens to coincide.
        h.update(str(texts[-1])[:max_chars_per_text].encode(
            'utf-8', 'replace'))
    return h.hexdigest()


def _batch_fingerprint(batch):
    """sha256 over the fixed overfit batch (ids + labels + shapes)."""
    h = hashlib.sha256()
    for name in ("input_ids", "labels"):
        t = batch[name].detach().to('cpu')
        h.update(f"{name}:{tuple(t.shape)}:".encode())
        h.update(str(t.contiguous().view(-1).tolist()).encode())
    return h.hexdigest()


def run_overfit_sanity_check(model_factory, batch, device, *, steps=100,
                             lr=3e-3, seed=42, grad_clip=1.0,
                             loss_ratio_threshold=0.5, log_every=10):
    """Overfit ONE fixed batch as a gradient-flow sanity gate.

    Runs before full training and trains a FRESH model (``model_factory()``)
    on a single fixed batch with all regularization disabled
    (``weight_decay=0``; the hybrid stack has no dropout and this pipeline
    applies no data augmentation — raw token windows only). If a model
    cannot drive the loss down on a handful of examples, something is
    fundamentally broken (gradients not flowing, detached loss, wrong
    labels, optimizer not stepping), and the gate catches it before a
    full 2000-step run.

    Pass criterion: the mean loss over the last ``tail`` steps must be
    strictly below ``loss_ratio_threshold`` times the mean loss over the
    first ``warmup`` steps (default: at least a 50% reduction). The token
    accuracy trajectory is reported for inspection but is informative
    only: this is a NECESSARY, NOT SUFFICIENT check — data-augmentation,
    feature-alignment and label-mapping bugs can still pass it, so it must
    be complemented by inspecting intermediate results.

    Returns a JSON-serializable report dict with the loss/accuracy curves.
    """
    if isinstance(device, str):
        device = torch.device(device)
    steps = int(steps)
    if steps < 10:
        # A handful of steps is dominated by early-loss noise (the loss can
        # even rise for a few steps before collapsing); requiring >= 10
        # avoids a misleading FAIL on a healthy model.
        raise ValueError(
            f"overfit check needs steps >= 10 to be meaningful, got {steps}")
    warmup = max(1, min(5, steps // 10))
    tail = max(1, min(5, steps // 10))

    # Deterministic, regularization-free recipe. Re-seeding here makes the
    # probe model's init independent of any RNG consumption before the call.
    seed_info = set_global_seed(seed, deterministic=True)
    model = model_factory().to(device)
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    losses, accs, elapsed_s = [], [], []
    t0 = time.time()
    for step in range(steps):
        model.hybrid.reset_state()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                f"overfit sanity check: non-finite LM loss at step {step} "
                f"({loss}); aborting before full training.")
        with torch.no_grad():
            mask = labels.ne(-100)
            correct = ((out["logits"].argmax(-1) == labels) & mask).sum()
            acc = float(correct) / max(1, int(mask.sum()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.detach()))
        accs.append(acc)
        elapsed_s.append(time.time() - t0)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"[overfit-check] step {step:4d} loss {losses[-1]:.4f} "
                  f"token_acc {acc:.4f} ({elapsed_s[-1]:.1f}s)", flush=True)

    initial_loss = sum(losses[:warmup]) / warmup
    final_loss = sum(losses[-tail:]) / tail
    best_loss = min(losses)
    loss_reduction = (initial_loss - final_loss) / initial_loss
    final_acc = sum(accs[-tail:]) / tail
    passed = final_loss < loss_ratio_threshold * initial_loss
    report = {
        "passed": bool(passed),
        "steps": steps,
        "lr": float(lr),
        "weight_decay": 0.0,
        "grad_clip_norm": float(grad_clip),
        "dropout": 0.0,
        "data_augmentation": "none",
        "seed": int(seed),
        "seed_info": seed_info,
        "fixed_batch_sha256": _batch_fingerprint(batch),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "loss_reduction": loss_reduction,
        "loss_ratio_threshold": float(loss_ratio_threshold),
        "final_train_token_acc": final_acc,
        "losses": losses,
        "accs": accs,
        "elapsed_s": elapsed_s,
        "note": ("Necessary but NOT sufficient: passing proves gradients flow "
                 "and the loss responds on a fixed batch; it cannot rule out "
                 "data-augmentation / feature-alignment / label-mapping bugs."),
    }
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run_decode_consistency_probe(model, device, seq_len, output_dir=None,
                                 seed=1234):
    """Prefill-decode consistency + decode-roofline probe of the trained LM.

    A trained checkpoint is only useful for generation if token-by-token
    decoding (KDA recurrent state + per-layer CSA/HCA incremental caches,
    ``run_decoding.HybridDecoding``) reproduces the full-sequence training
    forward. This runs both paths through the same weights on one synthetic
    batch (random token ids — the check is about numerical path equivalence,
    not language quality) and writes a JSON report next to the checkpoint.
    See ``lm_consistency`` for the accounting conventions.
    """
    import lm_consistency
    import cache_accounting

    T = max(8, min(int(seq_len), 64))
    prefill_len = max(4, T // 2)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randint(0, model.embed.num_embeddings, (1, T),
                        generator=gen).to(device)
    report = lm_consistency.run_lm_consistency_check(
        model, ids, prefill_len=prefill_len, device=device)
    # Reconcile the clone's live decode caches against the analytical
    # accounting (component-wise, both modelled and capacity-grown totals).
    report['cache_audit'] = cache_accounting.run_cache_audit_for_model(
        model, context_len=prefill_len, decode_steps=T - prefill_len,
        device=device, seed=seed)
    # Provenance so a saved report identifies the code/env/seed that made it.
    report['seed'] = int(seed)
    report['provenance'] = capture_provenance()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decode_consistency.json')
        write_json_atomic(report, path, indent=2, allow_nan=False)
        report['report_path'] = path
    return report


def _print_consistency_report(report, title=''):
    roof = report.get('roofline') or {}
    print(f"[consistency]{' ' + title if title else ''} "
          f"max|dlogit|={report['max_abs_logit_diff']:.3e} "
          f"(decode-only {report['decode_max_abs_logit_diff']:.3e}), "
          f"top1_agreement={report['top1_agreement']:.4f}, "
          f"consistent={report['consistent']} "
          f"(tol={report['tolerance']:.1e}), "
          f"diagnosis={report.get('diagnosis', '?')}")
    selection = report.get('selection') or {}
    if selection.get('agreement') is not None:
        print(f"[consistency] CSA top-k selection agreement: "
              f"{selection['agreement']:.4f} "
              f"(decode {selection['decode_agreement']:.4f}, "
              f"{selection.get('n_compared', 0)} positions compared)")
        if report.get('diagnosis') == 'topk_selection_divergence':
            print("[consistency] NOTE: selections differ while exact ties "
                  "cannot split the paths (ascending-index tie-break); "
                  "inspect the recorded near-tie mismatches")
    if roof:
        print(f"[roofline] decode at T={roof['context_len']}: "
              f"AI={roof['arithmetic_intensity_flops_per_byte']:.3f} "
              f"FLOPs/byte, bound={roof.get('roofline_bound', '?')}, "
              f"decode={report['decode_ms_per_token']:.3f} ms/token, "
              f"{roof['bytes_per_token']:.0f} B/token")
    audit = report.get('cache_audit') or {}
    comp = audit.get('comparison')
    if comp:
        b = audit['bytes']
        print(f"[cache-audit] live==analytical at T={comp['context_len']}: "
              f"consistent={comp['consistent']} "
              f"(autograd clean={audit['autograd_hygiene']['clean']}, "
              f"bytes[{b['policy']}]={b['total_bytes']:.0f} vs "
              f"bf16={b['pure_bf16_total_bytes']:.0f})")
    if report.get('report_path'):
        print(f"[consistency] report saved: {report['report_path']}")


def run_checkpoint_consistency(path, device, seed=1234):
    """Load a ``train_lm_autodl`` checkpoint and probe prefill-decode."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        # torch < 2.0 has no ``weights_only`` kwarg.
        ckpt = torch.load(path, map_location='cpu')
    cfg = HybridConfig(**ckpt['cfg'])
    model = LMWithHybrid(int(ckpt['vocab']), cfg)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return run_decode_consistency_probe(
        model, device, seq_len=64, output_dir=os.path.dirname(path) or '.',
        seed=seed)

def _parse_lm_layout(spec: str, repeat: int = 2) -> tuple[str, ...]:
    """Parse the --layout / LM_LAYOUT spec into a concrete layout tuple.

    Two forms (mirroring ``run_ablation``'s experiment matrix so the LM
    and the MQAR ablation speak the same layout language):

      * a single preset letter ``A``/``B``/``C``/``D``/``E`` — the 5-layer
        matrix unit from ``run_ablation.MATRIX_UNIT``, repeated ``repeat``
        times (default 2 -> the 10-layer formal configuration);
      * a comma-separated kind list — ``memory`` tokens allowed (resolved
        via the memory route) alongside the concrete kinds.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"--layout={spec!r} must be a non-empty string.")
    spec = spec.strip()
    if spec.upper() in ('A', 'B', 'C', 'D', 'E') and len(spec) == 1:
        if not isinstance(repeat, int) or repeat < 1:
            raise ValueError(
                f"layout repeat={repeat!r} must be a positive int.")
        from run_ablation import MATRIX_UNIT
        return tuple(MATRIX_UNIT[spec.upper()] * repeat)
    kinds = [k.strip().lower() for k in spec.split(',') if k.strip()]
    if not kinds:
        raise ValueError(
            f"--layout={spec!r} parsed to an empty kind list.")
    valid = ('memory', 'kda', 'mamba2', 'rwkv7', 'csa', 'hca')
    for i, k in enumerate(kinds):
        if k not in valid:
            raise ValueError(
                f"--layout kind {i}={k!r} is not one of {valid}.")
    return tuple(kinds)


def _lm_rwkv_dims() -> dict:
    """Env-tunable RWKV-7 geometry overrides for the LM entry (optional).

    Reuses the SAME env knobs as run_ablation / run_quality
    (RWKV_HEAD_DIM / RWKV_DECAY_LORA / RWKV_A_LORA / RWKV_GATE_LORA)
    so one environment pins one geometry across all entries. Only set
    knobs are returned — unset fields keep the HybridConfig defaults
    (the main-table geometry: head_dim=64, loras 32/32/64).
    """
    out = {}
    for env, field in (('RWKV_HEAD_DIM', 'rwkv_head_dim'),
                       ('RWKV_DECAY_LORA', 'rwkv_decay_lora'),
                       ('RWKV_A_LORA', 'rwkv_a_lora'),
                       ('RWKV_GATE_LORA', 'rwkv_gate_lora')):
        raw = os.environ.get(env)
        if raw is None or not raw.strip():
            continue
        try:
            out[field] = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{env}={raw!r} must be an int (got {exc}).") from exc
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--autodl", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lm")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model init and DataLoader shuffle.")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of the cleaned corpus held out as the "
                             "validation split (model selection + early "
                             "stopping). Default 0.1.")
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of the cleaned corpus sealed as the "
                             "test split (opened exactly once, after the "
                             "model is frozen). Default 0.1.")
    parser.add_argument("--val_every", type=int, default=100,
                        help="Run a validation pass every N optimizer steps "
                             "(default 100); a validation pass also runs at "
                             "step 0 and after the final step.")
    parser.add_argument("--eval_max_batches", type=int, default=64,
                        help="Cap the number of batches per validation/test "
                             "pass for a bounded compute budget (default 64; "
                             "0 = evaluate the full split).")
    parser.add_argument("--ppl_bootstrap_nboot", type=int, default=2000,
                        help="Bootstrap replicates for the unit-level "
                             "perplexity CI on the sealed test split "
                             "(default 2000; 0 disables the CI). The "
                             "bootstrap resamples whole evaluation "
                             "sequences, reusing the SAME single test pass "
                             "(no extra forward), so the CI adds no compute "
                             "beyond pure-Python resampling.")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop training after this many consecutive "
                             "validation checks without improvement "
                             "(default 5; 0 disables early stopping). The "
                             "best validation checkpoint is always kept and "
                             "used for the single final test evaluation.")
    parser.add_argument("--error_analysis_batches", type=int, default=2,
                        help="Held-out validation batches inspected by the "
                             "token-level bad-case analysis at the "
                             "val-selected checkpoint (default 2; 0 "
                             "disables). Runs ONLY on the validation split — "
                             "never on train or the sealed test set.")
    parser.add_argument("--error_analysis_examples", type=int, default=3,
                        help="Max decoded examples stored/printed per error "
                             "category (default 3).")
    parser.add_argument("--consistency_check", action="store_true",
                        help="Load --checkpoint, run the prefill-decode "
                             "consistency + decode-roofline probe, then exit "
                             "without training.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --consistency_check "
                             "(default: <output_dir>/final_lm.pt).")
    parser.add_argument("--skip_consistency_check", action="store_true",
                        help="Skip the automatic prefill-decode consistency "
                             "probe run after training finishes.")
    parser.add_argument("--skip_overfit_check", action="store_true",
                        help="Skip the small-batch overfit sanity gate before "
                             "training (NOT recommended: the gate catches "
                             "broken gradient flow / loss wiring cheaply).")
    parser.add_argument("--overfit_steps", type=int, default=100,
                        help="Optimizer steps for the fixed-batch overfit "
                             "sanity gate (default: 100).")
    parser.add_argument("--overfit_lr", type=float, default=3e-3,
                        help="LR for the overfit gate. Deliberately higher "
                             "than the 3e-4 training LR so a healthy model "
                             "memorizes the batch quickly (default: 3e-3).")
    parser.add_argument("--overfit_loss_ratio", type=float, default=0.5,
                        help="Gate passes when the tail-mean loss is strictly "
                             "below this fraction of the head-mean loss "
                             "(default: 0.5).")
    parser.add_argument("--overfit_check_only", action="store_true",
                        help="Run ONLY the overfit sanity gate (no full "
                             "training); useful as a fast smoke test.")
    # --- memory route / explicit layout (the LM entry joins the ablation
    # framework's memory-route matrix: same KDA / Mamba-2 / RWKV-7
    # operators, same layout language, same env knobs). ---
    parser.add_argument("--memory", choices=("kda", "mamba2", "rwkv7"),
                        default=None,
                        help="Recurrent memory operator filling the memory "
                             "slots (default: env LM_MEMORY or 'kda'). The "
                             "CSA/HCA layers and all other fields are "
                             "shared, so this single knob is the "
                             "KDA-replacement ablation at LM scale.")
    parser.add_argument("--layout", type=str, default=None,
                        help="Explicit hybrid layout. Either a matrix "
                             "preset letter A/B/C/D/E (the 5-layer units "
                             "from run_ablation.MATRIX_UNIT, repeated "
                             "--layout_repeat times; default repeat 2 -> "
                             "the 10-layer formal setting), or a "
                             "comma-separated kind list "
                             "(memory/kda/mamba2/rwkv7/csa/hca). Default: "
                             "env LM_LAYOUT, or the historical "
                             "n_kda:n_csa:n_hca ratio mode.")
    parser.add_argument("--layout_repeat", type=int, default=None,
                        help="Repetitions of the A/B/C/D preset unit "
                             "(default: env LM_LAYOUT_REPEAT or 2 = the "
                             "10-layer formal configuration). Ignored for "
                             "comma-separated layouts.")
    args = parser.parse_args()

    # All three splits are mandatory: without validation there is no
    # leakage-safe model selection / early stopping, and without a sealed
    # test split the reported generalization number is not held out.
    if not 0.0 < args.val_frac < 1.0 or not 0.0 < args.test_frac < 1.0:
        raise SystemExit("--val_frac and --test_frac must be in (0, 1)")
    if args.val_frac + args.test_frac >= 1.0:
        raise SystemExit("--val_frac + --test_frac must leave a positive "
                         "train fraction")
    if args.val_every < 0:
        raise SystemExit("--val_every must be >= 0")
    if args.eval_max_batches < 0:
        raise SystemExit("--eval_max_batches must be >= 0")
    if args.early_stop_patience < 0:
        raise SystemExit("--early_stop_patience must be >= 0")
    if args.error_analysis_batches < 0:
        raise SystemExit("--error_analysis_batches must be >= 0")
    if args.error_analysis_examples < 0:
        raise SystemExit("--error_analysis_examples must be >= 0")
    if args.ppl_bootstrap_nboot < 0:
        raise SystemExit("--ppl_bootstrap_nboot must be >= 0 (0 disables)")

    is_kaggle = args.kaggle or os.path.exists("/kaggle")
    info = configure_torch_for_device()
    device = info.device
    print(f"Device: {device}, is_kaggle={is_kaggle}")

    # Seed Python random / NumPy / torch / CUDA together; deterministic
    # cuDNN + algorithm selection for same-machine reproducibility.
    seed_info = set_global_seed(args.seed, deterministic=True)
    # Provenance (git commit / dirty flag / torch+cuda versions / device) is
    # captured ONCE at run start and embedded in every artifact this script
    # writes (checkpoints, run record, consistency report).
    provenance = capture_provenance()

    if args.consistency_check:
        ckpt_path = args.checkpoint or os.path.join(
            args.output_dir, "final_lm.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        report = run_checkpoint_consistency(ckpt_path, device, seed=args.seed)
        _print_consistency_report(report, title=os.path.basename(ckpt_path))
        return

    if HAS_HF:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        print("[train_lm_autodl] transformers not installed; using the "
              "char-level fallback tokenizer over synthetic text "
              "(install `transformers datasets` for real TinyStories runs)")
        tokenizer = _FallbackCharTokenizer()
    if tokenizer is None:
        raise RuntimeError("Need transformers tokenizer")
    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    kda_backend = validate_kda_backend(
        os.environ.get('KDA_BACKEND', 'reference')
    )
    # Memory route + explicit layout (the KDA→Mamba-2/RWKV-7 replacement
    # knob at LM scale). Validated here so a typo fails before the model
    # is built; the RWKV geometry env knobs are shared with the other
    # entries (see _lm_rwkv_dims).
    from ops_mamba_backend import validate_mamba_backend
    from ops_rwkv_backend import validate_rwkv_backend
    mamba_backend = validate_mamba_backend(
        os.environ.get('MAMBA_BACKEND', 'reference'))
    rwkv_backend = validate_rwkv_backend(
        os.environ.get('RWKV_BACKEND', 'reference'))
    memory_route = args.memory or os.environ.get('LM_MEMORY', 'kda')
    if memory_route not in ('kda', 'mamba2', 'rwkv7'):
        raise ValueError(
            f"memory route {memory_route!r} must be one of "
            f"('kda', 'mamba2', 'rwkv7').")
    layout_spec = args.layout or os.environ.get('LM_LAYOUT')
    layout_repeat = (args.layout_repeat if args.layout_repeat is not None
                     else int(os.environ.get('LM_LAYOUT_REPEAT', '2') or 2))
    route_kw = dict(
        memory_type=memory_route,
        mamba_backend=mamba_backend,
        rwkv_backend=rwkv_backend,
    )
    route_kw.update(_lm_rwkv_dims())
    if layout_spec is not None:
        route_kw['layout'] = _parse_lm_layout(layout_spec, layout_repeat)
    if is_kaggle:
        cfg = HybridConfig(d_model=256, n_heads_qk=2, n_heads_v=2, head_dim_k=32, head_dim_v=32,
                           csa_m=8, csa_topk=4, hca_m2=16, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64, kda_backend=kda_backend, **route_kw)
        batch_size = 1
        grad_accum = 8
        seq_len = 512
        max_steps = 500
    else:
        cfg = HybridConfig(d_model=512, n_heads_qk=4, n_heads_v=4, head_dim_k=64, head_dim_v=64,
                           csa_m=16, csa_topk=8, hca_m2=64, n_kda=3, n_csa=1, n_hca=1,
                           kda_chunk_size=64, kda_backend=kda_backend, **route_kw)
        batch_size = 2
        grad_accum = 4
        seq_len = 1024
        max_steps = 2000

    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.seq_len is not None:
        seq_len = args.seq_len
    if args.max_steps is not None:
        max_steps = args.max_steps

    if cfg.layout is not None:
        layout_str = '-'.join(k.upper() for k in cfg.layout)
        print(f"Hybrid layout: explicit {len(cfg.layout)} layers "
              f"[{layout_str}], d={cfg.d_model}, "
              f"memory={cfg.memory_type}")
    else:
        print(f"Hybrid layout: {cfg.n_kda}:{cfg.n_csa}:{cfg.n_hca} "
              f"(memory={cfg.memory_type}), d={cfg.d_model}")
    print(f"Training profile: batch_size={batch_size}, grad_accum={grad_accum}, "
          f"seq_len={seq_len}, optimizer_steps={max_steps}, seed={args.seed}")

    # ------------------------------------------------------------------
    # Data: load -> clean -> split (in this order) -> audit.
    #
    # The corpus budget is scaled so the TRAIN split keeps its row budget
    # (20000 local / 5000 Kaggle): holding out 10% val + 10% test must
    # not silently shrink the training set.
    # ------------------------------------------------------------------
    train_budget = 20000 if not is_kaggle else 5000
    train_frac = 1.0 - args.val_frac - args.test_frac
    corpus_budget = int(math.ceil(train_budget / train_frac))
    raw_texts, data_source = _load_raw_texts(corpus_budget, split="train")
    cleaning = clean_lm_texts(raw_texts, tokenizer, seq_len)
    clean_texts = cleaning['texts']
    if len(clean_texts) < 10:
        raise SystemExit(
            f"only {len(clean_texts)} usable documents after cleaning "
            f"(raw={len(raw_texts)}); refusing to train on a degenerate "
            f"corpus")
    split, split_report = split_clean_texts(
        clean_texts, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed)
    train_texts = [clean_texts[i] for i in split['train']]
    val_texts = [clean_texts[i] for i in split['val']]
    test_indices = list(split['test'])

    # Per-split length/unk statistics. Descriptive distribution-shift
    # diagnostics ONLY; they are never fitted or applied to features. The
    # test split's stats are computed with its single evaluation at the
    # very end (see below) — not here.
    stats_train = _text_length_stats(train_texts, tokenizer, seq_len)
    stats_val = _text_length_stats(val_texts, tokenizer, seq_len)
    shift_val = _distribution_shift(stats_train, stats_val)

    ds = TinyStoriesLM(tokenizer, seq_len=seq_len, texts=train_texts,
                       split="train", name="train")
    val_ds = TinyStoriesLM(tokenizer, seq_len=seq_len, texts=val_texts,
                           split="validation", name="val")
    # The test split stays unopened until the model is frozen. Only a
    # factory (plus the index list) exists during training.
    sealed_test = SealedTestSet(
        lambda: TinyStoriesLM(
            tokenizer, seq_len=seq_len,
            texts=[clean_texts[i] for i in test_indices],
            split="test", name="test"))

    print(f"Data: {data_source}, raw={len(raw_texts)}, "
          f"cleaned={len(clean_texts)} "
          f"(empty={cleaning['stats']['n_empty']}, "
          f"dup={cleaning['stats']['n_duplicates']}, "
          f"too_short={cleaning['stats']['n_too_short_to_supervise']})")
    print(f"Split (seed={args.seed}): train={len(train_texts)} "
          f"val={len(val_texts)} test={len(test_indices)} "
          f"[val/test sealed until final evaluation]")
    print(f"Fingerprints: train={ds.data_fingerprint[:16]}... "
          f"val={val_ds.data_fingerprint[:16]}...")
    if shift_val['flag']:
        print(f"[data] WARNING: val token-length distribution differs from "
              f"train (Cohen's d={shift_val['cohen_d']:.3f}); the held-out "
              f"loss may not be exchangeable with train")
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed + 1_000_000)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=loader_gen, worker_init_fn=seed_worker)

    # ------------------------------------------------------------------
    # Overfit sanity gate (before any full training): train a FRESH copy of
    # the model on ONE fixed batch with regularization off. A model that
    # cannot memorize a handful of examples has a fundamental bug
    # (gradients not flowing, detached loss, wrong labels, optimizer not
    # stepping), caught before the full 500-2000 step run. This is necessary
    # but NOT sufficient — augmentation/alignment/label-map bugs can still
    # pass — so it complements, never replaces, intermediate-result checks.
    # ------------------------------------------------------------------
    overfit_report = None
    overfit_batch = None
    if not args.skip_overfit_check:
        # Separate generator + loader: the probe batch is fixed across runs
        # and consuming it does NOT advance the training loader's stream, so
        # training stays bit-identical to a run with the gate skipped.
        probe_gen = torch.Generator()
        probe_gen.manual_seed(args.seed + 2_000_000)
        probe_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, generator=probe_gen)
        overfit_batch = next(iter(probe_loader))
        print(f"[overfit-check] training {args.overfit_steps} steps on one "
              f"fixed batch of {int(overfit_batch['input_ids'].shape[0])} "
              f"sequence(s) (weight_decay=0, no dropout/augmentation)...")
        overfit_report = run_overfit_sanity_check(
            lambda: LMWithHybrid(vocab_size, cfg), overfit_batch, device,
            steps=args.overfit_steps, lr=args.overfit_lr, seed=args.seed,
            loss_ratio_threshold=args.overfit_loss_ratio)
        print(f"[overfit-check] result: {'PASS' if overfit_report['passed'] else 'FAIL'} "
              f"loss {overfit_report['initial_loss']:.3f} -> "
              f"{overfit_report['final_loss']:.3f} "
              f"(best {overfit_report['best_loss']:.3f}), "
              f"token_acc -> {overfit_report['final_train_token_acc']:.3f}")
        if not overfit_report['passed']:
            raise SystemExit(
                "[overfit-check] FAILED: the model could not overfit a single "
                f"fixed batch in {args.overfit_steps} steps "
                f"(loss {overfit_report['initial_loss']:.3f} -> "
                f"{overfit_report['final_loss']:.3f}; required < "
                f"{args.overfit_loss_ratio:.2f}x the initial loss). This "
                "usually means gradients are not flowing, the loss is "
                "mis-wired, or labels are wrong — refusing to start the full "
                "run. Debug with --overfit_check_only, or bypass (NOT "
                "recommended) with --skip_overfit_check.")
        if args.overfit_check_only:
            print("[overfit-check] --overfit_check_only set; exiting without "
                  "full training.")
            return

    # Re-seed right before the real model is constructed so its init is
    # identical to a run where the gate never ran.
    set_global_seed(args.seed, deterministic=True)
    model = LMWithHybrid(vocab_size, cfg).to(device)
    params_count = count_params(model)
    print(f"Params: {params_count/1e6:.2f}M")
    print(f"Layout: {model.hybrid.layout_str()}")

    no_decay_ids = set()
    for submod in model.modules():
        if isinstance(submod, (nn.Embedding, nn.LayerNorm)):
            for p in submod.parameters(recurse=False):
                no_decay_ids.add(id(p))
    _POSITIONAL_BIAS_SUFFIXES = ('Ba', 'Bb', 'B_idx', 'B_pos')
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            continue
        seen.add(id(p))
        leaf_name = name.rsplit('.', 1)[-1]
        if (id(p) in no_decay_ids or p.ndim <= 1
                or leaf_name in _POSITIONAL_BIAS_SUFFIXES):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4)

    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif use_amp:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and autocast_dtype == torch.float16))
    print(f"AMP enabled={use_amp}, dtype={autocast_dtype}, grad_scaler={scaler.is_enabled()}")

    # Hyperparameters are captured ONCE here and embedded in every
    # checkpoint / run record so no artifact is ambiguous about its recipe.
    hyperparams = {
        'optimizer': 'AdamW',
        'lr': 3e-4,
        'weight_decay_decay_group': 0.1,
        'weight_decay_no_decay_group': 0.0,
        'grad_clip_norm': 1.0,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'effective_batch_sequences': batch_size * grad_accum,
        'seq_len': seq_len,
        'max_optimizer_steps': max_steps,
        'amp_enabled': use_amp,
        'autocast_dtype': str(autocast_dtype),
        'kda_backend': kda_backend,
        'model_config': dict(cfg.__dict__),
        # Held-out protocol (recorded with the recipe so a reader knows
        # exactly how model selection and the final number were obtained).
        'val_every': int(args.val_every),
        'eval_max_batches': int(args.eval_max_batches),
        'early_stop_patience': int(args.early_stop_patience),
        'model_selection': 'min token-weighted validation loss',
        'error_analysis_batches': int(args.error_analysis_batches),
        'error_analysis_examples': int(args.error_analysis_examples),
    }
    data_record = {
        'source': data_source,
        'split': 'train (carved from the raw train split after cleaning)',
        'cleaning': cleaning['stats'],
        'raw_corpus_budget': corpus_budget,
        'train_row_budget': train_budget,
        'split_protocol': {
            'order': ('load -> clean (dedup) -> split -> audit -> train; '
                      'test stays sealed until the model is frozen'),
            'seed': args.seed,
            'val_frac': float(args.val_frac),
            'test_frac': float(args.test_frac),
            'chronological': False,
            'group_ids': None,
            'note': ('Story-level deterministic split drawn with a '
                     'dedicated random.Random(seed), so the global RNG '
                     'streams driving model init / training batches are '
                     'untouched. No fitted preprocessing transform exists '
                     'in this pipeline; were one added (e.g. a tokenizer '
                     'or length normalizer), it must be fitted on train '
                     'indices only.'),
        },
        'splits': {
            'train': {
                'n': len(train_texts),
                'content_fingerprint_sha256': ds.data_fingerprint,
                'token_length_stats': stats_train,
            },
            'val': {
                'n': len(val_texts),
                'content_fingerprint_sha256': val_ds.data_fingerprint,
                'token_length_stats': stats_val,
            },
            'test': {
                'n': len(test_indices),
                'content_fingerprint_sha256': None,
                'token_length_stats': None,
                'sealed_until_final_evaluation': True,
            },
        },
        'split_audit': split_report,
        'distribution_shift': {
            'val_vs_train': shift_val,
            'test_vs_train': None,
        },
        'tokenizer': type(tokenizer).__name__,
        'vocab_size': vocab_size,
        'content_fingerprint_sha256': ds.data_fingerprint,
        'fingerprint_scheme': ('sha256 of <=64 evenly spaced texts truncated '
                               'to 4096 chars + corpus size'),
    }
    created_utc = datetime.datetime.now(
        datetime.timezone.utc).isoformat()

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    it = iter(loader)
    start = time.time()
    recent_step_losses = []
    # Full per-optimizer-step training curve (recorded in checkpoints and in
    # the run record; "don't log experiments by hand" — the files ARE the log).
    curve_steps, curve_losses, curve_elapsed_s = [], [], []
    optimizer_step = 0
    micro_step = 0
    # Cost telemetry (efficiency-vs-accuracy): tokens actually pushed through
    # the model, so throughput is measured, not inferred from the recipe.
    train_tokens = 0
    # Per-step total gradient norm (pre-clip). Recorded, never kept: it is
    # the "did the optimizer stay healthy" signal that loss alone hides.
    grad_norms = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _checkpoint_payload(extra=None):
        """Self-describing checkpoint: weights + recipe + provenance + curve.

        Recording the hyperparameters, git revision (and dirty flag), seed
        and the loss history INSIDE each checkpoint means even a crashed run's
        partial checkpoint can be traced back to the exact code/data/recipe
        that produced it. ``extra`` adds per-checkpoint metadata (e.g. the
        validation loss that selected it).
        """
        payload = {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "vocab": vocab_size,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "seed": args.seed,
            "seed_info": seed_info,
            "created_utc": created_utc,
            "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
            "hyperparams": hyperparams,
            "data": data_record,
            "loss_history": list(curve_losses),
        }
        if extra:
            payload.update(extra)
        return payload

    # ------------------------------------------------------------------
    # Validation-based model selection + early stopping.
    #
    # The validation split is used ONLY here and (optionally) to stop
    # training; the test split is untouched until after this block. The
    # evaluation is read-only (no optimizer step, no shuffling, no RNG),
    # so enabling/disabling validation cannot change the training math.
    # ------------------------------------------------------------------
    eval_batch_cap = (None if args.eval_max_batches == 0
                      else int(args.eval_max_batches))
    best_val = {'loss': float('inf'), 'step': None, 'state': None}
    val_history = []
    checks_without_improvement = 0
    stopped_early = False

    def _run_validation(step):
        nonlocal checks_without_improvement, stopped_early
        t0 = time.time()
        report = evaluate_lm_loss(
            model, val_ds, batch_size=batch_size, device=device,
            max_batches=eval_batch_cap, use_amp=use_amp,
            autocast_dtype=autocast_dtype)
        if report['loss'] is None:
            raise RuntimeError(
                f'validation split produced no supervised tokens at step '
                f'{step}; cannot select a model')
        val_history.append({
            'step': int(step),
            'loss': report['loss'],
            'perplexity': report['perplexity'],
            'n_batches': report['n_batches'],
            'n_sequences': report['n_sequences'],
            'n_tokens': report['n_tokens'],
            'elapsed_s': time.time() - t0,
        })
        # Streaming curve-shape verdict: the over/underfitting diagnosis is
        # recomputed at every validation check (including the point just
        # appended), so a decaying-train / rising-val run is visible DURING
        # training, not only in the final artifact. It is read-only and does
        # not influence the loop (early stopping keeps its own
        # improves/patience rule).
        _curve_verdict = diagnose_training_curves(
            curve_losses, [h['loss'] for h in val_history])['verdict']
        val_history[-1]['curve_verdict'] = _curve_verdict
        print(f"[monitor] step {step}: curve verdict={_curve_verdict}")
        if report['loss'] < best_val['loss']:
            best_val['loss'] = report['loss']
            best_val['step'] = int(step)
            best_val['state'] = {
                k: v.detach().to('cpu').clone()
                for k, v in model.state_dict().items()
            }
            checks_without_improvement = 0
            torch.save(
                _checkpoint_payload({
                    'selected_by': 'min_validation_loss',
                    'val_loss': report['loss'],
                    'val_perplexity': report['perplexity'],
                    'val_step': int(step),
                }),
                os.path.join(args.output_dir, 'best_val_lm.pt'))
            print(f"[val] step {step}: loss {report['loss']:.4f}, "
                  f"ppl {report['perplexity']:.2f} (new best -> "
                  f"best_val_lm.pt)")
        else:
            checks_without_improvement += 1
            print(f"[val] step {step}: loss {report['loss']:.4f} "
                  f"(best {best_val['loss']:.4f} @ step {best_val['step']}; "
                  f"{checks_without_improvement} check(s) without "
                  f"improvement)")
            if (args.early_stop_patience
                    and checks_without_improvement >= args.early_stop_patience):
                stopped_early = True
                print(f"[val] early stopping: no improvement for "
                      f"{checks_without_improvement} consecutive validation "
                      f"checks (patience {args.early_stop_patience}); "
                      f"the best validation checkpoint is kept")

    _run_validation(0)

    pbar = tqdm(total=max_steps, desc="optimizer steps")
    optimizer.zero_grad(set_to_none=True)
    while optimizer_step < max_steps:
        accum_loss = 0.0
        accum_batches = 0
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                try:
                    batch = next(it)
                except StopIteration:
                    break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            train_tokens += int(input_ids.numel())

            model.hybrid.reset_state()

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=use_amp):
                out = model(input_ids, labels=labels)
                raw_loss = out["loss"]
            if raw_loss is None or not torch.isfinite(raw_loss):
                # Systematic NaN/Inf triage (results_analysis.diagnose_nonfinite):
                # token-id range (data), parameter finiteness, AMP scaler state,
                # and — when AMP is on — a same-batch fp32 forward comparison,
                # which separates a mixed-precision overflow from a math bug.
                # The report is embedded in the exception so the failure names
                # its first failing stage.
                fp32_loss = None
                if use_amp and raw_loss is not None:
                    try:
                        with torch.no_grad():
                            model.hybrid.reset_state()
                            fp32_loss = model(input_ids,
                                              labels=labels)["loss"]
                        model.hybrid.reset_state()
                    except Exception:
                        fp32_loss = None
                diag = diagnose_nonfinite(
                    loss=(raw_loss if raw_loss is not None
                          else float('nan')),
                    token_ids={'input_ids': input_ids},
                    id_range=(0, vocab_size),
                    params=model.parameters(),
                    logits=out.get("logits"),
                    amp_enabled=bool(use_amp),
                    amp_scaler=(scaler if scaler.is_enabled() else None),
                    fp32_forward_loss=fp32_loss,
                    context=(f'optimizer_step={optimizer_step}, '
                             f'micro_step={micro_step}'))
                raise RuntimeError(
                    f"non-finite LM loss at optimizer_step={optimizer_step}, "
                    f"micro_step={micro_step}: {raw_loss}; diagnosis "
                    f"[{diag['verdict']}]: {diag['summary']}")
            scaled_loss = raw_loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accum_loss += float(raw_loss.detach().item())
            accum_batches += 1
            micro_step += 1

        if accum_batches == 0:
            break

        if accum_batches < grad_accum:
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        # clip_grad_norm_ RETURNS the total norm BEFORE clipping: keeping it
        # turns "loss was fine" into "gradients stayed bounded", and a
        # non-finite norm is the gradient-explosion stage of the NaN triage
        # (checked BEFORE the optimizer writes anything into the weights).
        grad_norm = torch.nn.utils.clip_grad_norm_(decay + no_decay, 1.0)
        if not torch.isfinite(grad_norm):
            bad_grads = {name: p.grad for name, p in model.named_parameters()
                         if p.grad is not None
                         and not torch.isfinite(p.grad).all()}
            diag = diagnose_nonfinite(
                loss=accum_loss / max(1, accum_batches),
                token_ids={'input_ids': input_ids},
                id_range=(0, vocab_size),
                params=model.parameters(),
                logits=out.get("logits"),
                grads=bad_grads or None,
                grad_norm=float('nan'),
                amp_enabled=bool(use_amp),
                amp_scaler=(scaler if scaler.is_enabled() else None),
                context=(f'optimizer_step={optimizer_step}, non-finite '
                         f'grad norm before the step'))
            raise RuntimeError(
                f"non-finite gradient norm at optimizer_step={optimizer_step} "
                f"({grad_norm}); diagnosis [{diag['verdict']}]: "
                f"{diag['summary']}")
        grad_norms.append(grad_norm.detach())
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_step += 1
        step_loss = accum_loss / max(1, accum_batches)
        recent_step_losses.append(step_loss)
        curve_steps.append(optimizer_step)
        curve_losses.append(step_loss)
        curve_elapsed_s.append(time.time() - start)
        pbar.update(1)
        pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        if optimizer_step % 10 == 0:
            mean_recent = sum(recent_step_losses) / len(recent_step_losses)
            print(f"step {optimizer_step} loss {mean_recent:.4f}")
            recent_step_losses.clear()

        if optimizer_step % 500 == 0:
            torch.save(_checkpoint_payload(),
                       os.path.join(args.output_dir, f"step_{optimizer_step}.pt"))

        if args.val_every and optimizer_step % args.val_every == 0:
            _run_validation(optimizer_step)
            if stopped_early:
                break

    pbar.close()
    elapsed = time.time() - start
    # Capture the TRAINING peak before evaluation/analysis forwards run, so
    # the recorded footprint is the training recipe's, not the diagnostics'.
    peak_train_memory = _peak_memory_bytes()
    if not stopped_early and (not args.val_every
                              or optimizer_step % args.val_every != 0):
        _run_validation(optimizer_step)
    print(f"Finished in {elapsed/60:.2f} min")
    torch.save(_checkpoint_payload(),
               os.path.join(args.output_dir, "final_lm.pt"))

    # ------------------------------------------------------------------
    # The model is now frozen: selection (and any early stopping) used
    # validation only. Load the val-selected checkpoint and open the
    # sealed test split EXACTLY ONCE for the final held-out evaluation.
    # Nothing downstream may use this number to change the model: that is
    # what keeps the test set a test set.
    # ------------------------------------------------------------------
    if best_val['state'] is None:
        raise RuntimeError(
            'no validation checkpoint was recorded; refusing to open the '
            'test split')
    model.load_state_dict(best_val['state'])
    model.to(device)
    model.eval()
    # Bad-case analysis on the VALIDATION split (held out from training, but
    # the legitimate diagnostic set): print and record which tokens the
    # val-selected model misses and why. Runs before the sealed test set is
    # opened, is read-only/RNG-inert, and never inspects test samples.
    error_analysis = None
    if int(args.error_analysis_batches or 0) > 0:
        print(f"[bad-case] analyzing up to {int(args.error_analysis_batches)} "
              f"held-out validation batch(es) at the val-selected checkpoint "
              f"(step {best_val['step']})...")
        error_analysis = analyze_lm_errors(
            model, val_ds, batch_size=batch_size, device=device,
            max_batches=int(args.error_analysis_batches), seed=args.seed,
            tokenizer=tokenizer,
            max_examples=int(args.error_analysis_examples))
    test_dataset = sealed_test.open()
    stats_test = _text_length_stats(test_dataset.texts, tokenizer, seq_len)
    shift_test = _distribution_shift(stats_train, stats_test)
    test_t0 = time.time()
    test_report = evaluate_lm_loss(
        model, test_dataset, batch_size=batch_size, device=device,
        max_batches=eval_batch_cap, use_amp=use_amp,
        autocast_dtype=autocast_dtype, return_units=True)
    test_report['elapsed_s'] = time.time() - test_t0
    test_units = test_report.pop('per_unit', [])
    test_report.pop('per_unit_contract', None)
    # The evaluation pass is a pure-prefill workload: record its throughput
    # next to the quality number, from the SAME single pass (no second
    # forward). End-to-end loop timing, same-machine comparisons only.
    test_report['prefill_throughput'] = throughput_summary(
        n_tokens=test_report['n_tokens'],
        elapsed_s=test_report['elapsed_s'],
        n_sequences=test_report['n_sequences'])
    # Unit-level bootstrap CI for the final perplexity: quantifies the
    # sampling uncertainty of THIS held-out corpus (resampling whole
    # sequences, not tokens). Cheap (pure-Python, same single pass).
    n_boot = int(getattr(args, 'ppl_bootstrap_nboot', 0) or 0)
    if test_units and n_boot > 0:
        test_report['ppl_bootstrap'] = bootstrap_ppl_ci(
            test_units, n_boot=n_boot, seed=args.seed)
    else:
        test_report['ppl_bootstrap'] = {
            'skipped': True,
            'reason': ('--ppl_bootstrap_nboot=0' if n_boot <= 0
                       else 'no supervised test tokens'),
        }
    model.train()
    _ci = (test_report['ppl_bootstrap'].get('ci_perplexity')
           if not test_report['ppl_bootstrap'].get('skipped') else None)
    _tps = test_report['prefill_throughput']['tokens_per_second']
    print(f"[test] single evaluation of the sealed test split at the "
          f"val-selected checkpoint (step {best_val['step']}): loss "
          f"{test_report['loss']:.4f}, ppl {test_report['perplexity']:.2f} "
          f"{'' if _ci is None else f'[{_ci[0]:.2f}, {_ci[1]:.2f}] '}"
          f"over {test_report['n_tokens']} tokens, "
          f"prefill {_tps:,.0f} tok/s")
    data_record['splits']['test']['content_fingerprint_sha256'] = \
        test_dataset.data_fingerprint
    data_record['splits']['test']['token_length_stats'] = stats_test
    data_record['splits']['test']['sealed_until_final_evaluation'] = False
    data_record['distribution_shift']['test_vs_train'] = shift_test

    # Persist the full run record (code version + hyperparameters + seed +
    # data version + training curve + final result) next to the checkpoint.
    # The file is the experiment log — no manual bookkeeping.
    mean_last10 = (sum(curve_losses[-10:]) / min(10, len(curve_losses))
                   if curve_losses else None)

    # Curve-shape diagnosis: overfitting / underfitting is a property of the
    # monitored train-vs-validation curves, not of the final numbers. The
    # verdict and its recommendations are recorded so a reader of the run
    # record sees whether the best validation checkpoint (used above) was
    # picked because of overfitting, and what to try next.
    curve_diagnosis = diagnose_training_curves(
        curve_losses, [h['loss'] for h in val_history])
    print(f"[diagnosis] train/val curves: {curve_diagnosis['verdict']}")
    for _rec in curve_diagnosis['recommendations']:
        print(f"[diagnosis]   - {_rec}")

    # Efficiency-vs-accuracy record: the recipe's measured cost next to its
    # quality, so a future "improvement" can be checked against params,
    # throughput, evaluation latency and memory in the SAME artifact — a
    # +0.5% loss improvement that triples step time is visible here instead
    # of being silently rewarded in a leaderboard of final numbers.
    val_eval_samples = [h for h in val_history
                        if h.get('elapsed_s') and h.get('n_sequences')]
    efficiency = {
        'n_params': int(params_count),
        'train_tokens': int(train_tokens),
        'train_seconds': float(elapsed),
        'train_tokens_per_second': (
            float(train_tokens / elapsed) if elapsed > 0 else None),
        'optimizer_steps': int(optimizer_step),
        'seconds_per_optimizer_step': (
            float(elapsed / optimizer_step) if optimizer_step else None),
        'peak_memory_bytes': peak_train_memory,
        'peak_memory_measurement': (
            'cuda_max_memory_allocated' if torch.cuda.is_available()
            else 'process_ru_maxrss_upper_bound'),
        'validation_eval': {
            'n_checks': len(val_history),
            'mean_forward_ms_per_sequence': (
                float(1000.0 * sum(h['elapsed_s'] / h['n_sequences']
                                   for h in val_eval_samples)
                      / len(val_eval_samples))
                if val_eval_samples else None),
        },
        'test_eval_ms_per_sequence': (
            float(1000.0 * test_report['elapsed_s'] / test_report['n_sequences'])
            if test_report.get('elapsed_s') and test_report['n_sequences']
            else None),
        'note': ('Cost is measured for THIS recipe (hardware-dependent, so '
                 'compare runs on the same machine). Peak memory on CPU is '
                 'the process RSS upper bound, not the model footprint. '
                 'Quality claims must be reported together with this block.'),
    }
    _tps = efficiency['train_tokens_per_second']
    print(f"[efficiency] params={params_count/1e6:.2f}M, "
          f"train={'n/a' if _tps is None else format(_tps, ',.0f')} tok/s, "
          f"peak_mem={efficiency['peak_memory_bytes']} B")

    # Optimizer-health record: the pre-clip gradient norm distribution is the
    # last stage of the NaN/explosion triage and the first sign of an LR
    # that is too large (long tail of huge norms even while the loss looks
    # finite). Kept as a compact summary, not a per-step series.
    gradient_norms = None
    if grad_norms:
        gn = torch.stack(grad_norms).float()
        gradient_norms = {
            'n_steps': int(gn.numel()),
            'max': float(gn.max()),
            'mean': float(gn.mean()),
            'last': float(gn[-1]),
            'clip_norm': 1.0,
            'note': ('norm BEFORE clipping (clip_grad_norm_ return value); '
                     'a max much larger than clip_norm means the raw '
                     'gradients were exploding and clipping is masking it.'),
        }
        print(f"[gradients] pre-clip norm max={gradient_norms['max']:.3f}, "
              f"mean={gradient_norms['mean']:.3f} "
              f"(clip at {gradient_norms['clip_norm']})")
    run_record = {
        "kind": "train_lm_autodl",
        "created_utc": created_utc,
        "elapsed_s": elapsed,
        "seed": args.seed,
        "seed_info": seed_info,
        "provenance": provenance,
        "hyperparameters": hyperparams,
        "data": data_record,
        "overfit_sanity_check": overfit_report,
        "training_curve": {
            "optimizer_step": curve_steps,
            "loss": curve_losses,
            "elapsed_s": curve_elapsed_s,
        },
        # Shape-based verdict over the train and validation curves
        # (overfitting / underfitting / healthy / diverged) with the
        # numeric evidence and next actions.
        "curve_diagnosis": curve_diagnosis,
        # Token-level bad cases on the held-out validation split: category
        # mix (near-miss / confident wrong / copyable missed / novel) plus
        # accuracy by position and by copy distance, with decoded examples.
        # None when disabled via --error_analysis_batches 0.
        "error_analysis": error_analysis,
        # Measured cost of this recipe (params / throughput / eval latency /
        # peak memory) so quality and efficiency stay reported together.
        "efficiency": efficiency,
        # Pre-clip gradient-norm summary: optimizer-health evidence that the
        # loss curve alone cannot provide.
        "gradient_norms": gradient_norms,
        "validation": {
            "criterion": "min token-weighted validation loss",
            "history": val_history,
            "best": {
                "loss": best_val['loss'] if best_val['step'] is not None
                        else None,
                "step": best_val['step'],
                "checkpoint": os.path.join(args.output_dir, "best_val_lm.pt"),
            },
            "early_stopping": {
                "enabled": bool(args.early_stop_patience),
                "patience": int(args.early_stop_patience),
                "stopped_early": bool(stopped_early),
                "checks_without_improvement": int(checks_without_improvement),
            },
        },
        "test": {
            # The single evaluation of the sealed test split, on the model
            # selected by validation. Recorded under its own key so it can
            # never be confused with a training or validation metric.
            "evaluated_once": True,
            "selection": "min validation loss",
            "selected_step": best_val['step'],
            "selected_checkpoint": os.path.join(args.output_dir,
                                                "best_val_lm.pt"),
            "report": test_report,
            "note": ("The test split was opened exactly once, after model "
                     "selection and early stopping were finished; its "
                     "metric must not be used for any further tuning."),
        },
        "final": {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "last_loss": curve_losses[-1] if curve_losses else None,
            "mean_last10_loss": mean_last10,
            "checkpoint": os.path.join(args.output_dir, "final_lm.pt"),
            "checkpoint_best_val": os.path.join(args.output_dir,
                                                "best_val_lm.pt"),
            "completed": optimizer_step >= max_steps,
            "stopped_early": bool(stopped_early),
        },
    }
    record_path = os.path.join(args.output_dir, "run_record.json")
    write_results_json(run_record, record_path)
    print(f"[run-record] wrote {record_path}")

    if not args.skip_consistency_check:
        # Probing the val-selected weights (the model currently in memory
        # for the test evaluation), so the audit matches the checkpoint
        # the test number was produced with.
        report = run_decode_consistency_probe(
            model, device, seq_len, args.output_dir, seed=args.seed)
        _print_consistency_report(report, title="best_val_lm.pt")

if __name__ == "__main__":
    main()
