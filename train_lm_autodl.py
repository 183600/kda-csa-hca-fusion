"""
LM training for the KDA+CSA+HCA hybrid, built on ``ops_fused``.

Architecture contrast (``--arch``): ``hybrid`` (default) trains the
recurrent/retrieval stack below; ``softmax`` trains a MATCHED dense causal
attention baseline (same width/depth, data, optimizer, token budget,
weight tying and embedding scale) through the same training loop. The two
share ``build_lm_model`` so the overfit gate, the real run and checkpoint
reload cannot silently build different architectures. ``LM_HYPOTHESES``
pre-registers the comparison (quality at matched budget + decode-state
efficiency) and every run record embeds it before the sealed test split is
opened.

The hybrid model uses ``HybridKCHAttention`` from ``ops_fused.py``, which
already implements:
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
import json
import math
import cmath
import dataclasses
import numbers
import operator
import unicodedata
from collections import Counter, OrderedDict
import numpy as np
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
                          split_train_val_test, audit_split,
                          grad_flow_report, make_run_manifest_entry,
                          append_run_manifest)
from results_analysis import (LM_ERROR_CATEGORIES, diagnose_nonfinite,
                              diagnose_training_curves, lm_token_diagnostics,
                              summarize_lm_diagnostics)
from train_diagnostics import (
    accuracy_regime,
    optimizer_update_report,
    raise_non_finite,
    snapshot_parameters,
    triage_gradient,
)
from ppl_bootstrap import bootstrap_ppl_ci, throughput_summary
from near_duplicates import (
    DEFAULT_NEAR_DUP_MAX_CANDIDATES,
    DEFAULT_NEAR_DUP_NUM_PERM,
    DEFAULT_NEAR_DUP_SHINGLE_SIZE,
    DEFAULT_NEAR_DUP_THRESHOLD,
    near_duplicate_clusters,
    near_duplicate_pairs,
    summarize_cross_split,
)
import grad_checkpointing as grad_ckpt

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
                 texts=None, name=None, prefetch_cache=False, prefetch_cap=1024):
        """Token-window dataset over a list of raw stories.

        When ``prefetch_cache`` is set, each sample's token ids are encoded
        once and reused on later ``__getitem__`` calls (prefetch), cutting
        repeated ``tokenizer.encode`` work in the training hot path. A
        correctness self-check re-encodes every 64th cache hit and asserts
        equality with the cached ids, so a stale cache can never silently
        change the training inputs.

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
        # Prefetch cache: idx -> encoded token ids (tuple). Populated lazily
        # on first access; capped by ``prefetch_cap`` (0 = unbounded). Only
        # active when ``prefetch_cache`` is on; defaults off so existing
        # training runs are bit-identical.
        self.prefetch_cache = bool(prefetch_cache)
        self.prefetch_cap = int(prefetch_cap)
        if self.prefetch_cap < 0:
            raise ValueError(
                f"prefetch_cap must be >= 0 (0 = unbounded), got "
                f"{self.prefetch_cap}.")
        self._prefetch_store: dict[int, tuple[int, ...]] = {}
        self._prefetch_hits = 0

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.prefetch_cache and idx in self._prefetch_store:
            tokens = list(self._prefetch_store[idx])
            self._prefetch_hits += 1
            # Correctness self-check: a cached encoding must stay identical
            # to a fresh encode of the same text (tokenizer is a pure
            # function; this guards against a future stateful tokenizer or a
            # stale cache silently changing training inputs).
            if self._prefetch_hits % 64 == 0:
                fresh = self.tokenizer.encode(
                    text, truncation=True, max_length=self.seq_len + 1)
                if list(fresh) != tokens:
                    # Explicit raise so ``python -O`` cannot strip the
                    # stale-cache check and train on wrong tokens.
                    raise RuntimeError(
                        f"prefetch cache mismatch at idx={idx}")
        else:
            tokens = self.tokenizer.encode(
                text, truncation=True, max_length=self.seq_len + 1)
            if self.prefetch_cache and self.prefetch_cap > 0 \
                    and len(self._prefetch_store) < self.prefetch_cap:
                self._prefetch_store[idx] = tuple(tokens)
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

def _normalized_text(text):
    """Canonical text identity for duplicate detection.

    Exact-string dedup misses the same document stored with different
    whitespace / Unicode normalization (common when a corpus is merged
    from CSV, JSON and HTML sources). NFKC plus whitespace collapsing
    makes those variants hash to one identity WITHOUT lowercasing, so
    genuinely different casing is still treated as different text.
    """
    return ' '.join(unicodedata.normalize('NFKC', str(text)).split())


def _text_digests(texts):
    """Content digests of a list of texts (exact-duplicate identity)."""
    return {hashlib.sha256(str(t).encode('utf-8', 'replace')).digest()
            for t in texts}


def _normalized_text_digests(texts):
    """Content digests after whitespace/Unicode normalization."""
    return {hashlib.sha256(_normalized_text(t).encode('utf-8', 'replace')).digest()
            for t in texts}


def _is_nonfinite_document(text):
    """True for a null / non-finite VALUE (not a legitimate string).

    A raw corpus assembled from a dataframe or a JSON feed can carry
    ``None`` / ``NaN`` / ``Inf`` entries. ``str(value)`` would turn those
    into the strings ``'None'`` / ``'nan'`` / ``'inf'`` - i.e. pure label
    noise that no cleaning step downstream would ever notice. Strings and
    finite numbers are NOT non-finite documents (``5`` is legitimate text).
    """
    if text is None:
        return True
    if isinstance(text, bool):
        return False
    # Every Python / NumPy integer is finite, including values whose
    # ``float()`` conversion would overflow (``10**400`` is a legitimate
    # text document, not a missing value).
    if isinstance(text, numbers.Integral):
        return False
    # ``numbers.Real`` covers Python / NumPy scalar floats, which are the
    # usual carriers of a missing/infinite value in a dataframe column.
    # Comparing / stringifying a NumPy ``inf`` does not reveal that it is
    # non-finite (``inf == inf``), so the old ``(int, float)`` check
    # silently kept it as the literal text "inf".
    if isinstance(text, numbers.Real):
        try:
            return not math.isfinite(float(text))
        except OverflowError:
            # A finite value too large for float is still finite.
            return False
        except (ValueError, TypeError):
            return True
    # Complex infinities/NaNs need ``cmath``; ``x != x`` only catches NaN.
    if isinstance(text, numbers.Complex):
        try:
            return not cmath.isfinite(text)
        except (OverflowError, ValueError, TypeError):
            return False
    # Decimal and similar numeric types expose an exact ``is_finite``.
    is_finite = getattr(text, 'is_finite', None)
    if callable(is_finite):
        try:
            return not bool(is_finite())
        except Exception:
            pass
    # NaN-like objects whose ``x != x`` (e.g. object floats) are caught by
    # the comparison; guard against types whose __ne__ raises.
    try:
        return bool(text != text)
    except Exception:
        return False


def _validated_token_ids(ids):
    """Exact non-negative integer token ids from a tokenizer result.

    A tokenizer that returns ``None``, a string, a float id, or a negative
    id would otherwise crash a later ``len()`` / be silently truncated by
    ``np.asarray(..., dtype=uint64)``. Those are tokenizer contract
    violations, not documents, so callers count them as tokenize errors.
    Objects implementing ``__index__`` (NumPy / torch integer scalars) are
    accepted; floats and bools are not.
    """
    if isinstance(ids, (str, bytes)):
        raise ValueError('tokenizer returned text instead of token ids')
    try:
        iterator = iter(ids)
    except TypeError:
        raise ValueError(
            f'tokenizer returned a non-iterable {type(ids).__name__}') from None
    out = []
    for token in iterator:
        if isinstance(token, bool):
            raise ValueError('token id is a bool')
        if isinstance(token, numbers.Integral):
            token_id = int(token)
        else:
            try:
                token_id = operator.index(token)
            except TypeError:
                raise ValueError(
                    f'token id is not an exact integer: {token!r}') from None
            token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f'negative token id: {token_id}')
        out.append(token_id)
    return out


def clean_lm_texts(texts, tokenizer, seq_len, *, dedup=True, min_tokens=2,
                   dedup_windows=False, dedup_near=False,
                   near_dup_threshold=DEFAULT_NEAR_DUP_THRESHOLD,
                   near_dup_shingle_size=DEFAULT_NEAR_DUP_SHINGLE_SIZE,
                   near_dup_num_perm=DEFAULT_NEAR_DUP_NUM_PERM,
                   near_dup_max_candidates=DEFAULT_NEAR_DUP_MAX_CANDIDATES):
    """Clean the raw corpus BEFORE any split is drawn.

    Cleaning steps (all recorded in the returned stats):

      * drop null / non-finite documents (``None``, NaN, +-Inf) instead of
        stringifying them into ``'None'``/``'nan'`` label noise;
      * drop empty / whitespace-only documents;
      * drop duplicates (content hash) so the same story can never end
        up on both sides of the split — dedup BEFORE splitting is what
        makes the cross-split disjointness guarantee possible. A second
        text whose NFKC-normalized, whitespace-collapsed form matches an
        already-kept text is also dropped (``n_normalized_duplicates``);
        exact-string-only dedup would leak the same document when sources
        differ only in whitespace or Unicode compatibility forms;
      * count documents whose MODEL WINDOW (the ``seq_len + 1`` tokens
        ``TinyStoriesLM`` actually consumes) was already seen: two
        different strings sharing that token prefix are the same training
        sample, and if one landed in train and the other in val/test the
        shared sample would be memorized rather than generalized. With
        ``dedup_windows=True`` they are dropped; by default they are only
        counted (``n_window_duplicates``) so the reference corpus
        composition is unchanged for a short-``seq_len`` corpus where
        many distinct documents share a prefix;
      * with ``dedup_near=True``, drop NEAR-duplicate documents as well:
        two windows whose token n-gram Jaccard is at least
        ``near_dup_threshold`` are the same story with an edit, not two
        independent samples. The duplicate clusters are computed on the
        cleaned corpus BEFORE the split, so no near-twin can straddle the
        train/eval boundary. The default is off for backward
        compatibility; the training entry point enables the cross-split
        AUDIT by default (see ``near_duplicate_leakage_audit``) and the
        pre-split drop only with ``--dedup_near``;
      * drop documents that tokenize to fewer than ``min_tokens`` ids:
        they produce no supervised label at all (every target is masked
        with -100), i.e. they are pure label noise for a next-token
        objective;
      * drop documents the tokenizer cannot encode (counted separately).

    Returns ``{'texts': [...], 'stats': {...}}``. The stats are
    JSON-serializable and go straight into the run record.
    """
    texts = list(texts)
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    min_tokens = _exact_int_arg('min_tokens', min_tokens, minimum=0)
    clean, seen_texts, seen_normalized, seen_windows = [], set(), set(), set()
    clean_ids = [] if dedup_near else None
    n_nonfinite = n_empty = n_duplicates = 0
    n_normalized_duplicates = 0
    n_window_duplicates = n_too_short = n_tokenize_errors = 0
    for text in texts:
        if _is_nonfinite_document(text):
            n_nonfinite += 1
            continue
        s = str(text)
        if not s.strip():
            n_empty += 1
            continue
        key = hashlib.sha256(s.encode('utf-8', 'replace')).digest()
        if dedup:
            normalized_key = hashlib.sha256(
                _normalized_text(s).encode('utf-8', 'replace')).digest()
            if key in seen_texts or normalized_key in seen_normalized:
                n_duplicates += 1
                if key not in seen_texts:
                    n_normalized_duplicates += 1
                continue
            seen_texts.add(key)
            seen_normalized.add(normalized_key)
        try:
            ids = _validated_token_ids(tokenizer.encode(
                s, truncation=True, max_length=seq_len + 1))
        except Exception:
            n_tokenize_errors += 1
            continue
        if len(ids) < int(min_tokens):
            n_too_short += 1
            continue
        # Identity of the actual model input: the truncated token window.
        # Catches whitespace/unicode variants and long documents collapsed
        # onto the same context by truncation. Always counted (leak
        # diagnostic); only dropped when ``dedup_windows`` opts in.
        window_key = tuple(ids)
        if window_key in seen_windows:
            n_window_duplicates += 1
            if dedup_windows:
                continue
        else:
            seen_windows.add(window_key)
        clean.append(s)
        if clean_ids is not None:
            # Compact representation: a 20k-document corpus at seq_len=1024
            # would otherwise hold ~20M Python ints (hundreds of MB) just
            # for the near-duplicate sketch.
            clean_ids.append(np.asarray(ids, dtype=np.uint64))

    near_dup = {
        'dedup_near': bool(dedup_near),
        'near_dup_threshold': float(near_dup_threshold),
        'near_dup_shingle_size': int(near_dup_shingle_size),
        'near_dup_num_perm': int(near_dup_num_perm),
        'near_dup_max_candidates': int(near_dup_max_candidates),
        'n_near_duplicates': 0,
        'n_near_duplicate_pairs': 0,
        'n_near_duplicate_clusters': 0,
        'largest_near_duplicate_cluster': 0,
        'near_duplicate_candidates_truncated': False,
    }
    if dedup_near:
        # Run the detector even for an empty / single-document corpus so the
        # detector's parameter contract is enforced regardless of corpus
        # size (an invalid threshold must not be silently recorded).
        near = near_duplicate_pairs(
            clean_ids, threshold=near_dup_threshold,
            shingle_size=near_dup_shingle_size,
            num_perm=near_dup_num_perm,
            max_candidate_pairs=near_dup_max_candidates)
        clusters = near_duplicate_clusters(near['pairs'], len(clean))
        # Keep the first (corpus-order) representative of every cluster;
        # dropping the rest removes the redundant near-twin without
        # looking at the split assignment.
        drop = set()
        for cluster in clusters:
            drop.update(cluster[1:])
        if drop:
            clean = [text for i, text in enumerate(clean) if i not in drop]
        near_dup.update({
            'n_near_duplicates': len(drop),
            'n_near_duplicate_pairs': int(near['n_pairs']),
            'n_near_duplicate_clusters': len(clusters),
            'largest_near_duplicate_cluster': (
                max((len(c) for c in clusters), default=0)),
            'near_duplicate_candidates_truncated':
                bool(near['candidates_truncated']),
        })
    n_raw = len(texts)
    stats = {
        'n_raw': n_raw,
        'n_kept': len(clean),
        'n_empty': n_empty,
        'n_nonfinite': n_nonfinite,
        'n_duplicates': n_duplicates,
        'n_normalized_duplicates': n_normalized_duplicates,
        'n_window_duplicates': n_window_duplicates,
        'n_too_short_to_supervise': n_too_short,
        'n_tokenize_errors': n_tokenize_errors,
        'dedup': bool(dedup),
        'dedup_windows': bool(dedup_windows),
        'min_tokens': int(min_tokens),
        'kept_fraction': (len(clean) / n_raw) if n_raw else 0.0,
        **near_dup,
    }
    return {'texts': clean, 'stats': stats}


def split_clean_texts(texts, *, val_frac=0.1, test_frac=0.1, seed=0,
                      group_ids=None, chronological=False,
                      tokenizer=None, seq_len=None,
                      near_dup_audit=False,
                      near_dup_threshold=DEFAULT_NEAR_DUP_THRESHOLD,
                      near_dup_shingle_size=DEFAULT_NEAR_DUP_SHINGLE_SIZE,
                      near_dup_num_perm=DEFAULT_NEAR_DUP_NUM_PERM,
                      near_dup_max_candidates=DEFAULT_NEAR_DUP_MAX_CANDIDATES,
                      strict_near_dup=False):
    """Draw and audit the train/val/test split over CLEANED texts.

    The split itself is deterministic (see
    ``kaggle_setup.split_train_val_test``). It supports the two
    structured-data regimes that a plain i.i.d. shuffle would leak on:

      * ``chronological=True`` — time series / ordered corpora: earlier
        documents form train, the next block validation, the most recent
        block test. Nothing is shuffled, so no future document is ever
        used to fit or select anything that predicts the past;
      * ``group_ids`` — one label per document (user / author / session /
        source id): all documents of a group land on the SAME side, so
        no group straddles train/val/test.

    On top of the structural audit this verifies that the three text
    partitions share no exact OR normalized duplicate (same content after
    NFKC + whitespace collapsing) — a defense-in-depth check against a
    future refactor that moves dedup after the split.

    ``near_dup_audit=True`` adds the token n-gram near-duplicate audit (see
    :func:`near_duplicate_leakage_audit`). It reports cross-split pairs in
    ``report['near_duplicate_overlaps']`` but does NOT drop rows; dropping
    must happen before the split (``clean_lm_texts(..., dedup_near=True)``).
    ``strict_near_dup=True`` forces the audit and raises when any
    cross-split near-duplicate pair remains.

    Returns ``(split, report)`` where ``split`` maps split names to index
    lists and ``report`` is the JSON-serializable audit + overlap report.
    Raises ``RuntimeError`` on any leakage violation.
    """
    if strict_near_dup:
        near_dup_audit = True
    if near_dup_audit and (tokenizer is None or seq_len is None):
        raise ValueError(
            'split_clean_texts(near_dup_audit=True) needs the tokenizer and '
            'seq_len used by the model window (got tokenizer=None or '
            'seq_len=None)')
    if group_ids is not None and len(group_ids) != len(texts):
        raise ValueError(
            f'group_ids has length {len(group_ids)} but there are '
            f'{len(texts)} texts')
    split = split_train_val_test(len(texts), val_frac=val_frac,
                                 test_frac=test_frac, seed=seed,
                                 group_ids=group_ids,
                                 chronological=chronological)
    audit = audit_split(split, len(texts), group_ids=group_ids,
                        chronological=chronological)
    by_split = {k: [texts[i] for i in split[k]]
                for k in ('train', 'val', 'test')}
    digests = {k: _text_digests(v) for k, v in by_split.items()}
    normalized_digests = {k: _normalized_text_digests(v)
                          for k, v in by_split.items()}
    overlaps = {
        'train_val': len(digests['train'] & digests['val']),
        'train_test': len(digests['train'] & digests['test']),
        'val_test': len(digests['val'] & digests['test']),
    }
    normalized_overlaps = {
        'train_val': len(normalized_digests['train']
                         & normalized_digests['val']),
        'train_test': len(normalized_digests['train']
                          & normalized_digests['test']),
        'val_test': len(normalized_digests['val']
                        & normalized_digests['test']),
    }
    # The audit verifies the ordering itself: raw row order without
    # groups, group first-appearance order when groups are declared.
    order_verified = audit['chronological_order_verified']
    report = {
        'seed': int(seed),
        'val_frac': float(val_frac),
        'test_frac': float(test_frac),
        'chronological': bool(chronological),
        'group_aware': group_ids is not None,
        'n_groups': (len(set(map(str, group_ids)))
                     if group_ids is not None else None),
        'chronological_order_verified': order_verified,
        'audit': audit,
        'exact_text_overlaps': overlaps,
        'normalized_text_overlaps': normalized_overlaps,
        'near_duplicate_overlaps': None,
    }
    if near_dup_audit:
        near = near_duplicate_leakage_audit(
            by_split['train'], by_split['val'], by_split['test'],
            tokenizer, seq_len, threshold=near_dup_threshold,
            shingle_size=near_dup_shingle_size, num_perm=near_dup_num_perm,
            max_candidate_pairs=near_dup_max_candidates)
        report['near_duplicate_overlaps'] = near
        if strict_near_dup and near['detector']['candidates_truncated']:
            raise RuntimeError(
                'train/val/test split cannot certify near-duplicate '
                'safety: the candidate budget was exhausted '
                f"({near['detector']['n_candidate_pairs']} candidate "
                'pairs); raise near_dup_max_candidates')
        if strict_near_dup and near['n_cross_split_pairs']:
            problems = list(audit['problems'])
            problems.append(
                f"{near['n_cross_split_pairs']} cross-split "
                f"near-duplicate pair(s) at Jaccard >= "
                f"{near_dup_threshold}: {near['by_split_pair']}")
            raise RuntimeError(
                'train/val/test split failed its near-duplicate leakage '
                'audit: ' + '; '.join(problems))
    problems = list(audit['problems'])
    for pair, n_overlap in overlaps.items():
        if n_overlap:
            problems.append(
                f'{pair}: {n_overlap} identical text(s) shared across splits')
    for pair, n_overlap in normalized_overlaps.items():
        if n_overlap:
            problems.append(
                f'{pair}: {n_overlap} normalized-duplicate text(s) shared '
                'across splits')
    if problems:
        raise RuntimeError(
            'train/val/test split failed its leakage audit: '
            + '; '.join(problems))
    return split, report


def _window_digest(text, tokenizer, seq_len):
    """Identity of the exact model input window (seq_len + 1 tokens)."""
    try:
        ids = _validated_token_ids(tokenizer.encode(
            text, truncation=True, max_length=seq_len + 1))
    except Exception:
        return None
    h = hashlib.sha256()
    for token_id in ids:
        h.update(str(token_id).encode('ascii'))
        h.update(b',')
    return h.digest()


def _window_overlap_audit(train_texts, val_texts, test_texts, tokenizer,
                          seq_len):
    """Count identical MODEL-INPUT windows shared across split boundaries.

    Exact-string dedup in :func:`clean_lm_texts` cannot see this leak:
    two different documents can truncate to the same ``seq_len + 1``
    token window, and ``TinyStoriesLM`` then trains on / evaluates the
    SAME sample. The global count is already recorded during cleaning;
    this audit additionally says WHICH pair of splits shares windows.
    With ``--dedup_windows`` the pre-split cleaning removes them all and
    every count here is zero.
    """
    def _digests(texts):
        out = set()
        for text in texts:
            d = _window_digest(text, tokenizer, seq_len)
            if d is not None:
                out.add(d)
        return out

    train = _digests(train_texts)
    val = _digests(val_texts)
    test = _digests(test_texts)
    overlaps = {
        'train_val': len(train & val),
        'train_test': len(train & test),
        'val_test': len(val & test),
    }
    total = sum(overlaps.values())
    return {
        'train_val': overlaps['train_val'],
        'train_test': overlaps['train_test'],
        'val_test': overlaps['val_test'],
        'total_cross_split': total,
        'any': bool(total),
        'n_train_windows': len(train),
        'n_val_windows': len(val),
        'n_test_windows': len(test),
        'note': ('Windows are the exact seq_len+1 token inputs the model '
                 'sees after truncation. Any non-zero count is a '
                 'sample-level leak; use --dedup_windows to remove them '
                 'before splitting, or --strict_window_dedup to refuse to '
                 'train on a corpus that has them.'),
    }


def _filter_leaking_heldout(train_texts, val_texts, test_texts, tokenizer,
                             seq_len, *, min_keep=1):
    """Drop held-out rows whose model window already exists on the train side.

    The exact-string and optional window cleaning happen BEFORE the split,
    so this is the last line of defence: after the split, any held-out
    document whose truncated ``seq_len+1`` token window is identical to a
    TRAIN window would be a sample the model trained on. Train is
    authoritative here, exactly the direction a leakage filter must use.
    Test rows are additionally checked against the kept validation rows.

    Refuses to apply a drop that would empty a held-out split (the corpus
    then has no non-leaking held-out document at this ``seq_len``); in
    that case the rows are kept, ``applied`` is False, and the caller's
    window-overlap audit reports the residual leak. Returns the filtered
    lists plus before/after counts so the run record can show exactly what
    was removed and what was kept.
    """
    def _digests(texts):
        out = set()
        for text in texts:
            d = _window_digest(text, tokenizer, seq_len)
            if d is not None:
                out.add(d)
        return out

    train_keys = _digests(train_texts)

    def _split(texts, forbidden):
        kept, dropped = [], 0
        for text in texts:
            d = _window_digest(text, tokenizer, seq_len)
            if d is not None and (d in train_keys or d in forbidden):
                dropped += 1
            else:
                kept.append(text)
        return kept, dropped

    val_kept, val_dropped = _split(val_texts, set())
    val_fallback = bool(val_dropped and len(val_kept) < int(min_keep))
    if val_fallback:
        val_kept, val_dropped = list(val_texts), 0
    val_keys = _digests(val_kept)
    test_kept, test_dropped = _split(test_texts, val_keys)
    test_fallback = bool(test_dropped and len(test_kept) < int(min_keep))
    if test_fallback:
        test_kept, test_dropped = list(test_texts), 0
    applied = bool(val_dropped or test_dropped)
    return {
        'applied': applied,
        'train_authoritative': True,
        'min_keep': int(min_keep),
        'val_texts': val_kept,
        'test_texts': test_kept,
        'val_dropped': int(val_dropped),
        'test_dropped': int(test_dropped),
        'val_fallback_all_leaking': val_fallback,
        'test_fallback_all_leaking': test_fallback,
        'n_val_before': len(val_texts),
        'n_test_before': len(test_texts),
        'n_val_after': len(val_kept),
        'n_test_after': len(test_kept),
        'note': ('Held-out rows whose exact model-input window also occurs '
                 'in TRAIN are dropped before any validation/test '
                 'measurement; test rows are also checked against the kept '
                 'validation windows. A split is kept whole when dropping '
                 'would empty it, and window_overlap_audit then reports the '
                 'residual leak.'),
    }


def near_duplicate_leakage_audit(
        train_texts, val_texts, test_texts, tokenizer, seq_len, *,
        threshold=DEFAULT_NEAR_DUP_THRESHOLD,
        shingle_size=DEFAULT_NEAR_DUP_SHINGLE_SIZE,
        num_perm=DEFAULT_NEAR_DUP_NUM_PERM,
        max_candidate_pairs=DEFAULT_NEAR_DUP_MAX_CANDIDATES):
    """Audit NEAR-duplicate leakage across the three already-drawn splits.

    Exact-string and exact-window dedup cannot see a document that was
    stored again with a few edited words.  Such a near-twin is a leak when
    one copy is in train and the other in val/test: the held-out loss then
    measures memorisation of a near-copy instead of generalisation.  This
    function compares the truncated model-input windows with token
    n-gram Jaccard (MinHash candidate retrieval + exact verification, see
    :mod:`near_duplicates`) and reports only pairs that STRADDLE a split
    boundary.  Within-split redundancy is counted separately.

    The audit is structural: it reads the text/token content but computes
    no model metric and makes no model-selection decision, so it can run
    before training with the test split still sealed.  It never drops a
    row; dropping must happen before the split (``clean_lm_texts(...,
    dedup_near=True)``) or be refused by the caller's strict gate.
    """
    train_texts = list(train_texts)
    val_texts = list(val_texts)
    test_texts = list(test_texts)
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    if not callable(getattr(tokenizer, 'encode', None)):
        raise ValueError('tokenizer must provide a callable encode()')
    combined = train_texts + val_texts + test_texts
    labels = (['train'] * len(train_texts)
              + ['val'] * len(val_texts)
              + ['test'] * len(test_texts))
    token_ids = []
    n_tokenize_errors = 0
    for text in combined:
        try:
            ids = _validated_token_ids(tokenizer.encode(
                text, truncation=True, max_length=seq_len + 1))
            token_ids.append(np.asarray(ids, dtype=np.uint64))
        except Exception:
            n_tokenize_errors += 1
            token_ids.append(None)
    detector = near_duplicate_pairs(
        token_ids, threshold=threshold, shingle_size=shingle_size,
        num_perm=num_perm, max_candidate_pairs=max_candidate_pairs)
    report = summarize_cross_split(detector['pairs'], labels)
    report.update({
        'enabled': True,
        'splits': {'train': len(train_texts),
                   'val': len(val_texts),
                   'test': len(test_texts)},
        'n_tokenize_errors': int(n_tokenize_errors),
        'detector': {
            'method': 'token-n-gram jaccard (minhash + exact verification)',
            'threshold': float(threshold),
            'shingle_size': int(shingle_size),
            'num_perm': int(num_perm),
            'band_rows': int(detector['band_rows']),
            'seed': int(detector['seed']),
            'n_documents': int(detector['n_documents']),
            'n_nonempty_documents': int(detector['n_nonempty_documents']),
            'n_candidate_pairs': int(detector['n_candidate_pairs']),
            'candidates_truncated': bool(detector['candidates_truncated']),
            'exact_verification': bool(detector['exact_verification']),
        },
        'flag': bool(report['n_cross_split_pairs'] > 0),
    })
    return report


def _evenly_spaced_probe_indices(n, n_probe):
    """At most ``n_probe`` strictly increasing indices spanning ``[0, n)``.

    Unlike a fixed ``stride = n // n_probe`` walk, this always reaches the
    LAST text and spreads the requested budget over the whole split.  The
    old stride-then-truncate selection silently stopped early whenever
    ``n // n_probe`` rounded down (e.g. n=300 / 256 kept texts 0..255 and
    n=1000 / 256 kept texts 0..765), so the diagnostics under-sampled the
    corpus tail and could miss a length shift there.
    """
    n = int(n)
    n_probe = int(n_probe)
    if n <= 0 or n_probe <= 0:
        return []
    if n <= n_probe:
        return list(range(n))
    if n_probe == 1:
        return [0]
    return [round(i * (n - 1) / (n_probe - 1)) for i in range(n_probe)]


def _text_length_stats(texts, tokenizer, seq_len, *, n_probe=256):
    """Descriptive token-length stats for ONE split (reporting only).

    Computed on a bounded, evenly spaced probe so it stays cheap on a
    20k-story corpus. This is a distribution-shift DIAGNOSTIC: it is
    reported in the run record and never used to fit a transformation or
    to select a model.
    """
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    n_probe = _exact_int_arg('n_probe', n_probe, minimum=0)
    n = len(texts)
    if n == 0:
        return {'n': 0, 'n_probe': 0, 'mean_tokens': None,
                'std_tokens': None, 'min_tokens': None, 'max_tokens': None,
                'cap_frac': None, 'unk_rate': None}
    probe = [texts[i] for i in _evenly_spaced_probe_indices(n, n_probe)]
    lengths, unk, total = [], 0, 0
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    for text in probe:
        try:
            ids = _validated_token_ids(tokenizer.encode(
                text, truncation=True, max_length=seq_len + 1))
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


def _probe_token_lengths(texts, tokenizer, seq_len, *, n_probe=256):
    """Bounded token-length probe used by the train-only outlier audit.

    Same deterministic, evenly spaced selection as
    :func:`_text_length_stats`, returned as plain lengths so robust
    quantiles can be fitted on TRAIN and applied to val/test without ever
    pooling the held-out splits into the statistic.
    """
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    n_probe = _exact_int_arg('n_probe', n_probe, minimum=0)
    n = len(texts)
    if n == 0:
        return []
    probe = [texts[i] for i in _evenly_spaced_probe_indices(n, n_probe)]
    lengths = []
    for text in probe:
        try:
            ids = _validated_token_ids(tokenizer.encode(
                text, truncation=True, max_length=seq_len + 1))
        except Exception:
            continue
        lengths.append(len(ids))
    return lengths


def _quantile_type7(sorted_values, q):
    """Linear-interpolation quantile (same convention as ppl_bootstrap)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError('cannot take a quantile of an empty sample')
    q_value = _finite_float_or_none(q)
    if q_value is None or not 0.0 <= q_value <= 1.0:
        raise ValueError(f'quantile must be a finite number in [0, 1], got {q!r}')
    if n == 1:
        return float(sorted_values[0])
    pos = q_value * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return (float(sorted_values[lo]) * (1.0 - frac)
            + float(sorted_values[hi]) * frac)


def _length_outlier_audit(train_lengths, heldout_lengths, *,
                          iqr_multiplier=1.5):
    """Robust outlier count for ONE held-out split against TRAIN fences.

    The Tukey fences ``[Q1 - k*IQR, Q3 + k*IQR]`` are fitted on the TRAIN
    token-length probe ONLY, then applied unchanged to the held-out probe.
    Pooling train+val+test to pick the fences would let the held-out
    distribution move the decision boundary (and is exactly the
    normalization-on-all-data mistake this pipeline avoids).

    Malformed lengths (non-finite, negative, bool, non-numeric) are
    counted and skipped rather than silently entering the quantiles and
    making the recorded fences NaN/inf. A non-finite multiplier is a
    caller error and raises ``ValueError``.
    """
    multiplier = _finite_float_or_none(iqr_multiplier)
    if multiplier is None or multiplier < 0.0:
        raise ValueError(
            f'iqr_multiplier must be a finite number >= 0, got '
            f'{iqr_multiplier!r}')

    try:
        train_values = list(train_lengths) if train_lengths is not None else []
    except TypeError:
        raise ValueError('train_lengths must be a sequence of lengths') from None
    try:
        heldout_values = (list(heldout_lengths)
                          if heldout_lengths is not None else [])
    except TypeError:
        raise ValueError(
            'heldout_lengths must be a sequence of lengths') from None

    def _finite_lengths(values):
        finite, invalid = [], 0
        for value in values:
            number = _finite_float_or_none(value)
            if number is None or number < 0.0:
                invalid += 1
                continue
            finite.append(number)
        return finite, invalid

    train_values, n_invalid_train = _finite_lengths(train_values)
    heldout_values, n_invalid_heldout = _finite_lengths(heldout_values)
    if not train_values:
        return {'fit_on': 'train', 'fences': None, 'n_probe': 0,
                'n_outliers': None, 'outlier_fraction': None,
                'flag': None, 'iqr_multiplier': multiplier,
                'n_invalid_train': n_invalid_train,
                'n_invalid_heldout': n_invalid_heldout,
                'note': 'no finite non-negative train lengths'}
    ordered = sorted(train_values)
    q1 = _quantile_type7(ordered, 0.25)
    q3 = _quantile_type7(ordered, 0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    fences = (q1, q3, iqr, lower, upper)
    if not all(math.isfinite(value) for value in fences):
        return {'fit_on': 'train', 'fences': None, 'n_probe': 0,
                'n_outliers': None, 'outlier_fraction': None,
                'flag': None, 'iqr_multiplier': multiplier,
                'n_invalid_train': n_invalid_train,
                'n_invalid_heldout': n_invalid_heldout,
                'note': ('token-length fences are outside the finite float '
                         'range')}
    n = len(heldout_values)
    n_out = sum(1 for x in heldout_values if x < lower or x > upper)
    frac = (n_out / n) if n else None
    return {
        'fit_on': 'train',
        'fences': {'q1': q1, 'q3': q3, 'iqr': iqr,
                   'lower': lower, 'upper': upper},
        'iqr_multiplier': multiplier,
        'n_probe': n,
        'n_outliers': n_out,
        'outlier_fraction': frac,
        'n_invalid_train': n_invalid_train,
        'n_invalid_heldout': n_invalid_heldout,
        # Report-only: text length is not a fitted feature and clipping
        # held-out documents would change the evaluation population. The
        # flag is a distribution-shift warning, not a gate.
        'flag': bool(frac is not None and frac >= 0.05),
    }


def _finite_float_or_none(value):
    """Finite float for a real scalar, else ``None`` (bools are not numbers)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _distribution_shift(train_stats, other_stats, *, threshold=0.5):
    """Standardized mean difference (Cohen's d) of token lengths vs train.

    Reports (does not act): a large |d| is a warning that the held-out
    split is not exchangeable with train, which would make the reported
    held-out loss hard to interpret. The threshold only sets the
    ``flag`` field.

    Zero-variance handling matters: if both splits have the same token
    length for every probed document (``pooled == 0``), the old formula
    returned ``d = 0.0`` even when the two means were completely
    different, silently reporting "no shift".  A constant train length of
    3 and a constant held-out length of 300 is the strongest possible
    shift.  This function now reports ``cohen_d=None`` for that case
    (the standardized effect is undefined, not zero), keeps the raw
    ``mean_difference``, and sets ``flag=True`` whenever the means differ.

    Malformed statistics (non-mapping input, non-finite means, negative or
    non-finite SDs) are reported as ``flag=None`` with a ``note`` instead
    of crashing the data-record assembly or smuggling a NaN into JSON.
    """
    threshold_value = _finite_float_or_none(threshold)
    if threshold_value is None or threshold_value < 0.0:
        raise ValueError(
            f'threshold must be a finite number >= 0, got {threshold!r}')

    def _invalid(note):
        return {'reference': 'train', 'cohen_d': None, 'abs_cohen_d': None,
                'mean_difference': None, 'zero_pooled_variance': None,
                'flag': None, 'threshold': threshold_value, 'note': note}

    try:
        m_t_raw = train_stats.get('mean_tokens')
        m_o_raw = other_stats.get('mean_tokens')
        s_t_raw = train_stats.get('std_tokens')
        s_o_raw = other_stats.get('std_tokens')
    except AttributeError:
        return _invalid('length statistics must be mappings')
    m_t = _finite_float_or_none(m_t_raw)
    m_o = _finite_float_or_none(m_o_raw)
    if m_t is None or m_o is None:
        return _invalid('missing or non-finite mean token length')
    s_t = 0.0 if s_t_raw is None else _finite_float_or_none(s_t_raw)
    s_o = 0.0 if s_o_raw is None else _finite_float_or_none(s_o_raw)
    if s_t is None or s_o is None or s_t < 0.0 or s_o < 0.0:
        return _invalid('invalid standard deviation')
    mean_difference = m_o - m_t
    mean_difference_finite = math.isfinite(mean_difference)
    # ``s ** 2`` raises OverflowError on CPython for finite inputs near the
    # float limit, and even when it returns a value the intermediate sum can
    # overflow. Keep the historical expression for ordinary inputs (so the
    # reported values are unchanged) and fall back to scaled arithmetic only
    # when that expression cannot produce a finite pooled SD.
    try:
        pooled = math.sqrt((s_t ** 2 + s_o ** 2) / 2.0)
    except OverflowError:
        pooled = math.inf
    if not math.isfinite(pooled):
        scale = max(s_t, s_o)
        if scale == 0.0:
            pooled = 0.0
        else:
            pooled = scale * (
                math.hypot(s_t / scale, s_o / scale) / math.sqrt(2.0))
    elif pooled == 0.0 and (s_t > 0.0 or s_o > 0.0):
        # Squaring subnormal-sized SDs underflows to zero even though the
        # pooled SD (and hence d) is nonzero. Re-evaluate in scaled units;
        # do not report a constant split when the inputs vary.
        scale = max(s_t, s_o)
        pooled = scale * (
            math.hypot(s_t / scale, s_o / scale) / math.sqrt(2.0))
    zero_pooled = pooled == 0.0
    if zero_pooled:
        # The standardized effect is undefined when both splits are
        # constant. Do not encode a real separation as d=0.
        cohen_d = 0.0 if mean_difference == 0.0 else None
        flag = mean_difference != 0.0
    else:
        ratio = (mean_difference / pooled) if mean_difference_finite else None
        if ratio is not None and math.isfinite(ratio):
            cohen_d = ratio
            flag = abs(ratio) >= threshold_value
        else:
            # Both the difference and/or the pooled SD can be outside the
            # representable float range even though all inputs are finite.
            # A common scaling leaves the standardized effect unchanged.
            scale = max(abs(m_t), abs(m_o), s_t, s_o, 1.0)
            scaled_pooled = math.hypot(s_t / scale, s_o / scale) / math.sqrt(
                2.0)
            if scaled_pooled > 0.0:
                scaled_ratio = ((m_o / scale) - (m_t / scale)) / scaled_pooled
            else:
                scaled_ratio = math.inf
            if math.isfinite(scaled_ratio):
                cohen_d = scaled_ratio
                flag = abs(scaled_ratio) >= threshold_value
            else:
                cohen_d = None
                flag = True
    if not mean_difference_finite:
        # +-max float values can differ by more than the float range can
        # represent. ``null`` is the JSON-safe statement of that fact.
        mean_difference = None
    return {
        'reference': 'train',
        'cohen_d': cohen_d,
        'abs_cohen_d': abs(cohen_d) if cohen_d is not None else None,
        'mean_difference': mean_difference,
        'zero_pooled_variance': bool(zero_pooled),
        'flag': bool(flag),
        'threshold': threshold_value,
        'note': ('zero within-split variance: d is undefined, not zero; '
                 'flag is set from the raw mean difference'
                 if zero_pooled and cohen_d is None else
                 'Cohen d on the bounded token-length probe'),
    }


def _exact_int_arg(name, value, *, minimum):
    """Exact integer argument (bool / float would silently change geometry)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an exact int, got {value!r}')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    return value


def _finite_unit_interval_arg(name, value):
    """Finite real argument in [0, 1] (bool is not a valid threshold)."""
    number = _finite_float_or_none(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise ValueError(
            f'{name} must be a finite number in [0, 1], got {value!r}')
    return number


def _probe_token_documents(texts, tokenizer, seq_len, *, n_probe):
    """Per-document token counts over a bounded, evenly spaced probe.

    Returns ``(documents, stats)``: ``documents`` is a list of
    ``collections.Counter`` objects (one per successfully tokenized,
    non-empty probe document) and ``stats`` is a JSON-serializable record
    of how the probe was obtained.  The probe is the SAME deterministic
    evenly spaced selection used by :func:`_text_length_stats`, so the
    token audit and the length audit describe the same documents and the
    audit cost is bounded on a large corpus.

    A document the tokenizer cannot encode (or that yields an invalid
    negative id) is counted in ``n_tokenize_errors`` instead of aborting
    the audit: one bad row must not hide the distribution of the rest.
    """
    n_probe = _exact_int_arg('n_probe', n_probe, minimum=1)
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    texts = list(texts)
    n = len(texts)
    indices = _evenly_spaced_probe_indices(n, n_probe)
    documents: list[Counter] = []
    n_errors = n_empty = n_tokens = 0
    for i in indices:
        text = texts[i]
        if _is_nonfinite_document(text):
            n_errors += 1
            continue
        try:
            ids = _validated_token_ids(tokenizer.encode(
                str(text), truncation=True, max_length=seq_len + 1))
            tokens = list(ids)
        except Exception:
            n_errors += 1
            continue
        if not tokens:
            n_empty += 1
            continue
        documents.append(Counter(tokens))
        n_tokens += len(tokens)
    return documents, {
        'n_texts': n,
        'n_probe_requested': int(n_probe),
        'n_documents_probed': len(indices),
        'n_documents_scored': len(documents),
        'n_empty_documents': n_empty,
        'n_tokenize_errors': n_errors,
        'n_tokens': n_tokens,
    }


def _decode_token_id(tokenizer, token_id, *, max_chars=40):
    """Best-effort readable form of one token id (``None`` if unsupported)."""
    decode = getattr(tokenizer, 'decode', None)
    if not callable(decode):
        return None
    try:
        text = decode([int(token_id)], skip_special_tokens=False)
    except TypeError:
        try:
            text = decode([int(token_id)])
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    # A JSON record must not smuggle raw NUL / newlines into the examples;
    # escape them so the field is inspectable in a one-line log.
    text = text.replace('\x00', '\\x00').replace('\r', '\\r') \
               .replace('\n', '\\n')
    return text[:int(max_chars)]


def _unigram_shift_stats(fit_documents, eval_documents, *,
                         tokenizer=None, max_examples=0):
    """Unigram distribution comparison between two document samples.

    Both samples are collections of per-document token ``Counter`` objects,
    so the unit of sampling stays the DOCUMENT (tokens inside a document are
    correlated).  Returns ``None`` when either sample has no tokens.

    Metrics (all bounded, JSON-safe):

    * ``unseen_token_mass`` — fraction of held-out TOKENS whose id never
      occurs in the fit sample.  This is the share of the evaluation the
      model's training vocabulary/embedding never saw.
    * ``unseen_type_fraction`` — fraction of held-out token TYPES that are
      absent from the fit sample.
    * ``js_divergence_bits`` — Jensen-Shannon divergence (base 2, in
      [0, 1]) between the two unigram distributions over the union of
      observed ids.  Plug-in JS is biased upward by finite samples; the
      caller compares it with a within-train null computed the same way.
    * ``total_variation`` — half the L1 distance between the two unigram
      distributions, in [0, 1].
    """
    max_examples = _exact_int_arg('max_examples', max_examples, minimum=0)
    if not fit_documents or not eval_documents:
        return None
    fit_counts: Counter = Counter()
    for document in fit_documents:
        fit_counts.update(document)
    eval_counts: Counter = Counter()
    for document in eval_documents:
        eval_counts.update(document)
    fit_total = sum(fit_counts.values())
    eval_total = sum(eval_counts.values())
    if fit_total <= 0 or eval_total <= 0:
        return None

    unseen = [token_id for token_id in eval_counts
              if token_id not in fit_counts]
    unseen_mass = (sum(eval_counts[token_id] for token_id in unseen)
                   / eval_total)
    unseen_type_fraction = len(unseen) / max(len(eval_counts), 1)

    js = 0.0
    total_variation = 0.0
    # Sorted, not raw set order: the JS sum is a float reduction, so a
    # deterministic order makes repeated audits bit-identical.
    for token_id in sorted(set(fit_counts) | set(eval_counts)):
        p = fit_counts.get(token_id, 0) / fit_total
        q = eval_counts.get(token_id, 0) / eval_total
        midpoint = 0.5 * (p + q)
        if p > 0.0:
            js += 0.5 * p * math.log2(p / midpoint)
        if q > 0.0:
            js += 0.5 * q * math.log2(q / midpoint)
        total_variation += abs(p - q)
    # Guard the bounds against floating-point round-off (the exact values
    # are in [0, 1]); a value outside them would be an invalid metric.
    js = min(1.0, max(0.0, js))
    total_variation = min(1.0, max(0.0, 0.5 * total_variation))

    examples = []
    ordered_unseen = sorted(unseen,
                            key=lambda t: (-eval_counts[t], t))
    for token_id in ordered_unseen[:max_examples]:
        examples.append({
            'token_id': int(token_id),
            'count': int(eval_counts[token_id]),
            'decoded': (_decode_token_id(tokenizer, token_id)
                        if tokenizer is not None else None),
        })
    return {
        'n_fit_documents': len(fit_documents),
        'n_eval_documents': len(eval_documents),
        'n_fit_tokens': int(fit_total),
        'n_eval_tokens': int(eval_total),
        'n_fit_types': len(fit_counts),
        'n_eval_types': len(eval_counts),
        'n_unseen_types': len(unseen),
        'unseen_type_fraction': float(unseen_type_fraction),
        'unseen_token_mass': float(unseen_mass),
        'js_divergence_bits': float(js),
        'total_variation': float(total_variation),
        'unseen_examples': examples,
        # ``max_examples=0`` (the null fold) deliberately omits examples;
        # that is not a truncated list, so do not report it as one.
        'examples_truncated': bool(max_examples > 0
                                   and len(unseen) > len(examples)),
    }


def _stable_document_order(n_documents, *, seed):
    """Deterministic document order derived only from ``(seed, index)``.

    The null calibration must not use the global RNG: the audit runs inside
    the training entry point and must not perturb the model / batch RNG
    streams.  Sorting by a keyed SHA-256 digest gives a reproducible,
    seed-specific permutation with no library RNG state at all.
    """
    n = _exact_int_arg('n_documents', n_documents, minimum=0)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an exact int, got {seed!r}')
    seed = int(seed)
    return sorted(
        range(n),
        key=lambda i: hashlib.sha256(
            f'{seed}\x00{i}'.encode('ascii')).digest())


def token_distribution_audit(
        train_texts, heldout_texts, tokenizer, seq_len, *,
        n_probe=256, js_threshold=0.05, unseen_mass_threshold=0.01,
        max_examples=10, seed=0):
    """Audit held-out token-distribution shift against the TRAIN probe.

    Token-length shift (:func:`_distribution_shift`) is only one nuisance
    dimension.  Two splits can have the same length distribution while the
    held-out split contains a different vocabulary / topic / source mix;
    the held-out perplexity then partly measures domain mismatch and the
    two numbers are not exchangeable.  This audit tests that directly on
    token unigrams, fitting every reference statistic on TRAIN only (the
    pipeline's standing contract) and never inspecting a model metric.

    Why a null calibration
    ----------------------
    A plug-in comparison always finds *some* held-out tokens unseen in
    train: with finite samples, even a perfectly i.i.d. split produces
    unseen rare words, and the amount grows as the fit sample shrinks.
    Reporting that raw number as "distribution shift" would flag every
    split.  The audit therefore also computes the SAME metrics on two
    disjoint folds of the TRAIN probe (document-level folds, preferring an
    evaluation fold sized to the held-out probe and falling back to a
    balanced train half-split), giving an empirical i.i.d. baseline.  The
    decision uses the EXCESS ``max(0, observed - null)``.  If the train
    probe is too small for a null, ``flag`` is ``None``: the raw numbers
    are still recorded, but the audit refuses to call them a shift.

    The default thresholds are pre-specified warnings (5% excess JS,
    1% excess unseen-token mass), not gates; no row is dropped and no
    model-selection decision is derived from them.  In the LM entry point
    ``test_vs_train`` is computed only after the sealed test split is
    opened for its single final evaluation.

    Returns a strictly JSON-serializable dict with ``observed``,
    ``null_reference``, ``excess``, ``flag`` and the probe provenance.
    """
    n_probe = _exact_int_arg('n_probe', n_probe, minimum=1)
    max_examples = _exact_int_arg('max_examples', max_examples, minimum=0)
    js_threshold = _finite_unit_interval_arg('js_threshold', js_threshold)
    unseen_mass_threshold = _finite_unit_interval_arg(
        'unseen_mass_threshold', unseen_mass_threshold)
    seq_len = _exact_int_arg('seq_len', seq_len, minimum=1)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an exact int, got {seed!r}')
    seed = int(seed)
    if not callable(getattr(tokenizer, 'encode', None)):
        raise ValueError('tokenizer must provide a callable encode()')

    train_documents, train_probe = _probe_token_documents(
        list(train_texts), tokenizer, seq_len, n_probe=n_probe)
    heldout_documents, heldout_probe = _probe_token_documents(
        list(heldout_texts), tokenizer, seq_len, n_probe=n_probe)
    report = {
        'available': bool(train_documents and heldout_documents),
        'fit_on': 'train probe only',
        'reference': 'train',
        'train_probe': train_probe,
        'heldout_probe': heldout_probe,
        'js_threshold': js_threshold,
        'unseen_mass_threshold': unseen_mass_threshold,
        'max_examples': max_examples,
        'seed': seed,
        'observed': None,
        'null_reference': None,
        'excess': None,
        'flag': None,
        'flag_reasons': [],
        'no_rows_dropped': True,
        'note': ('Descriptive distribution-shift warning fitted on TRAIN; '
                 'the flag compares the held-out probe with a within-train '
                 'document-level null and is never used for model '
                 'selection or as a data transform.'),
    }
    if not report['available']:
        report['reason'] = (
            'no non-empty tokenized documents in the train or held-out '
            'probe')
        return report
    observed = _unigram_shift_stats(
        train_documents, heldout_documents, tokenizer=tokenizer,
        max_examples=max_examples)
    if observed is None:
        report['available'] = False
        report['reason'] = 'the token probe contained no token ids'
        return report
    report['observed'] = observed

    # Within-train null: two disjoint document folds, sized so the null
    # evaluation fold matches the held-out probe's document count.  When
    # the train probe is too small for two matched folds, fall back to a
    # balanced half-split; if even that is not possible, no null is
    # reported and the audit refuses to flag.
    order = _stable_document_order(len(train_documents), seed=seed)
    n_eval = min(len(heldout_documents), len(train_documents))
    null = None
    if n_eval >= 2 and len(train_documents) >= 2 * n_eval:
        null_fit_idx = order[n_eval:2 * n_eval]
        null_eval_idx = order[:n_eval]
    else:
        n_eval = len(train_documents) // 2
        if n_eval >= 2 and len(train_documents) - n_eval >= 2:
            null_fit_idx = order[n_eval:]
            null_eval_idx = order[:n_eval]
        else:
            null_fit_idx = null_eval_idx = None
    if null_fit_idx is not None:
        null = _unigram_shift_stats(
            [train_documents[i] for i in null_fit_idx],
            [train_documents[i] for i in null_eval_idx],
            tokenizer=None, max_examples=0)
    report['null_reference'] = null
    if null is None:
        report['flag_reasons'] = ['null_calibration_unavailable']
        report['excess'] = None
        return report

    excess_unseen = max(
        0.0, observed['unseen_token_mass'] - null['unseen_token_mass'])
    excess_js = max(
        0.0, observed['js_divergence_bits'] - null['js_divergence_bits'])
    reasons = []
    if excess_unseen > unseen_mass_threshold:
        reasons.append('unseen_token_mass')
    if excess_js > js_threshold:
        reasons.append('js_divergence_bits')
    report['excess'] = {
        'unseen_token_mass': float(excess_unseen),
        'js_divergence_bits': float(excess_js),
        'definition': 'max(0, observed - within-train null)',
    }
    report['flag'] = bool(reasons)
    report['flag_reasons'] = reasons
    return report


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
                     return_units=False, return_accuracy=False):
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
    total_correct = 0
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
            if return_accuracy:
                # Optional token accuracy on the SAME masked positions:
                # turns the loss curve into a (loss, accuracy) pair the
                # held-out monitor can diagnose (e.g. saturated accuracy
                # while loss still falls).
                valid_mask = labels != -100
                total_correct += int(
                    ((out['logits'].argmax(dim=-1) == labels)
                     & valid_mask).sum())
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
        'token_accuracy': ((total_correct / total_tokens)
                           if return_accuracy and total_tokens else None),
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


def _lm_error_examples_saturated(examples, max_examples) -> bool:
    """Whether every error category already reached its example cap.

    ``examples`` maps a category name to its collected examples; the
    ``'correct'`` / ``'ignored'`` pseudo-categories are excluded because
    they never collect examples.
    """
    return all(len(examples[c]) >= int(max_examples)
               for c in LM_ERROR_CATEGORIES
               if c not in ('correct', 'ignored'))


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
    # Counts EVERY batch pulled from the loader (including the ones that are
    # skipped below).  ``n_batches`` counts only the ANALYZED batches, so it
    # cannot be used to locate a sample in the dataset: a skipped batch would
    # shift every later row index down by one batch.
    loader_batch_index = 0
    try:
        for batch in loader:
            if max_batches is not None and n_batches >= int(max_batches):
                break
            # Row offset of this batch's first sample in the dataset
            # (``shuffle=False``, so the loader is a plain sequential slice).
            batch_row_base = loader_batch_index * int(batch_size)
            loader_batch_index += 1
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
            sample_base = batch_row_base
            saturated = False
            for bi in range(int(input_ids.shape[0])):
                if saturated:
                    break
                for t, cat in enumerate(d['category'][bi]):
                    if cat in ('correct', 'ignored'):
                        continue
                    bucket = examples[cat]
                    if len(bucket) >= int(max_examples):
                        continue
                    lo = max(0, t + 1 - int(context_tokens))
                    ctx_ids = [int(v) for v in input_ids[bi, lo:t + 1].tolist()]
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
                    if _lm_error_examples_saturated(examples, max_examples):
                        # Every error category has reached its cap: stop
                        # collecting examples for the rest of the stream.
                        saturated = True
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
            f"{b['quartile']}="
            f"{'n/a' if b['accuracy'] is None else format(b['accuracy'], '.3f')}"
            f"(n={b['n']})"
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


def _sample_token(logits, *, temperature=1.0, top_k=None, top_p=None):
    """Sample a single token id from logits with temperature / top-k / top-p.

    ``temperature <= 0`` selects greedy (argmax). ``top_k`` keeps only the
    ``k`` most likely tokens; ``top_p`` keeps the smallest nucleus whose
    cumulative probability mass is at least ``top_p``. Returns a Python int.
    Deterministic for a fixed torch RNG state (caller seeds via
    ``torch.manual_seed``).
    """
    logits = logits.float().reshape(-1)
    vocab = int(logits.shape[0])
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    if top_k is not None and 0 < int(top_k) < vocab:
        k = int(top_k)
        top_values, top_indices = torch.topk(logits, k)
        masked = torch.full_like(logits, float('-inf'))
        masked[top_indices] = top_values
        logits = masked
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs_sorted, dim=-1)
        sorted_remove = cum_probs > float(top_p)
        # Keep the first token that pushes cumulative mass over top_p (the
        # standard nucleus rule); drop everything after it.
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_remove,
                                                  float('-inf'))
        logits = logits.new_full((vocab,), float('-inf'))
        logits.scatter_(0, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    if temperature is not None and temperature <= 0:
        return int(torch.argmax(probs).item())
    return int(torch.multinomial(probs, num_samples=1).item())


class LMWithHybrid(nn.Module):
    """Causal LM wrapper: token embedding -> hybrid stack -> final norm -> head.

    The token embedding and the output head are WEIGHT-TIED
    (``lm_head.weight is embed.weight``), so the embedding matrix plays two
    roles: the input representation and the unembedding projection.

    ``embed_scale`` is the GPT-2-style embedding scaling factor. The raw
    embedding lookup is initialised with ``std=0.02``; fed straight into the
    first hybrid layer that gives the value/gate projections pre-activations
    of magnitude ~0.02, which under-trains those paths early in training.
    Multiplying the looked-up embedding by ``sqrt(d_model)`` restores a
    healthy input scale (~0.16-0.32) for the CSA/HCA branches (which do NOT
    renormalise their inputs) and for the value/gate paths, while leaving
    the KDA ``q``/``k`` branches (L2-normalised) essentially unchanged.

    Hardware-aware notes (why this is a fused scalar multiply, not a
    separate op or a re-init):

      * ONE vectorised kernel: the scale is a Python ``float`` broadcast
        over the ``[B, T, d]`` gather output — no second materialised
        tensor, no extra allocation, no per-forward host->device copy.
      * ZERO new parameters / buffers: the scale is a plain attribute, so
        ``state_dict`` keys, ``count_params`` and the reproducibility
        contract are all unchanged.
      * Autocast / dtype-safe: a Python-float scalar never promotes the
        tensor dtype, so bf16/fp16 autocast behaviour is identical to the
        unscaled path.

    The scale is applied on EVERY forward — training, validation and the
    pure-prefill evaluation path all call :meth:`forward` — so prefill
    throughput/quality measurements reflect the scaled embeddings.

    ``ckpt_policy`` selects the gradient-checkpointing policy
    (``grad_checkpointing.CHECKPOINT_POLICIES``) applied to the hybrid
    attention stack during the forward. ``'none'`` (the default) keeps the
    legacy stateful ``self.hybrid(x)`` call bit-for-bit; the other policies
    route the stack through ``grad_checkpointing.forward_with_policy``,
    which wraps the selected layers in ``torch.utils.checkpoint`` (pure
    functional recompute — the RoPE'd CSA/HCA layers and the KDA/Mamba-2/
    RWKV-7 layers recompute identically because their forwards are
    side-effect-free w.r.t. module state when the recurrent state is
    threaded externally, which the training loop guarantees via
    ``model.hybrid.reset_state()`` before every micro-step).
    """

    def __init__(self, vocab_size, cfg: HybridConfig, embed_scale=None,
                 ckpt_policy=None):
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
        # Embedding scaling factor (GPT-2 style). ``None`` selects the
        # standard ``sqrt(d_model)`` default; pass ``1.0`` to recover the
        # legacy unscaled behaviour exactly.
        if embed_scale is None:
            embed_scale = math.sqrt(cfg.d_model)
        if (not isinstance(embed_scale, (int, float))
                or not math.isfinite(embed_scale) or embed_scale <= 0):
            raise ValueError(
                f"embed_scale must be a positive finite number, got "
                f"{embed_scale!r}.")
        self.embed_scale = float(embed_scale)
        # Gradient-checkpointing policy for the hybrid stack. Validated via
        # ``grad_checkpointing.parse_policy`` so a typo fails at
        # construction, not mid-training.
        self.ckpt_policy = grad_ckpt.parse_policy(
            'none' if ckpt_policy is None else ckpt_policy)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids) * self.embed_scale
        if self.ckpt_policy != 'none':
            # Pure-functional checkpointed stack forward (see
            # ``grad_checkpointing.forward_with_policy``): identical
            # residual/norm plumbing and value parity with the stateful
            # forward, but the selected layers' activations are recomputed
            # during backward instead of retained. The training loop resets
            # the stack state before every micro-step, so the externally
            # threaded state is always None here.
            x, _, _ = grad_ckpt.forward_with_policy(
                self.hybrid, x, policy=self.ckpt_policy)
        else:
            x = self.hybrid(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids, *, max_new_tokens=32, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None, seed=None):
        """Autoregressive next-token generation from the trained LM.

        Generation is correctness-first: every step re-runs a full forward
        over the *entire* prefix (prompt + tokens generated so far) with the
        recurrent state reset, takes the logits at the last position, and
        samples the next token.

        Why recompute the full prefix: ``LMWithHybrid``'s KDA layers carry a
        recurrent state across forward calls, but the CSA/HCA layers are
        stateless *inside* the stack — their incremental decode state lives
        in ``ops_decoding_cache`` and is not threaded through
        ``LMWithHybrid.forward``. Recomputing the full prefix is therefore
        the only path that lets *all three* operator types see the complete
        history on every generated token. It is O(n^2) in generated length —
        fine for short generations and a correct reference; an
        incremental-cache path can be layered on top later through
        ``run_decoding.HybridDecoding`` (the same path used by
        ``lm_consistency``).

        Returns a list of generated token-id lists (one per batch item),
        *excluding* the input prompt. Sampling is deterministic when ``seed``
        is set.
        """
        was_training = self.training
        self.eval()
        device = self.embed.weight.device
        input_ids = input_ids.to(device)
        B, T = input_ids.shape
        if seed is not None:
            torch.manual_seed(int(seed))
        sequences = [list(input_ids[b].tolist()) for b in range(B)]
        finished = [False] * B
        try:
            for _ in range(int(max_new_tokens)):
                any_active = False
                logits_all = [None] * B
                for b in range(B):
                    if finished[b]:
                        continue
                    any_active = True
                    prefix = torch.tensor([sequences[b]], device=device,
                                          dtype=torch.long)
                    self.hybrid.reset_state()
                    out = self(prefix)
                    logits_all[b] = out["logits"][0, -1]
                if not any_active:
                    break
                for b in range(B):
                    if finished[b] or logits_all[b] is None:
                        continue
                    next_id = _sample_token(
                        logits_all[b], temperature=temperature,
                        top_k=top_k, top_p=top_p)
                    sequences[b].append(int(next_id))
                    if eos_token_id is not None \
                            and int(next_id) == int(eos_token_id):
                        finished[b] = True
        finally:
            self.train(was_training)
            self.hybrid.reset_state()
        return [seq[T:] for seq in sequences]


def _fallback_tokenizer_payload(tokenizer=None) -> dict:
    """Serialise a fallback char tokenizer into a JSON-friendly dict."""
    tok = tokenizer if tokenizer is not None else _FallbackCharTokenizer()
    return {
        'type': 'fallback_char',
        'itos': list(tok._itos),
        'pad_token_id': int(tok.pad_token_id),
        'unk_token_id': int(tok.unk_token_id),
    }


def _tokenizer_payload(tokenizer=None) -> dict:
    """Describe the tokenizer a checkpoint was trained with.

    Export must rebuild the SAME vocabulary, not whatever tokenizer is
    installed on the exporting host: a fallback-char run exported on a host
    with ``transformers`` would otherwise advertise GPT-2 ids for a
    97-token embedding. The payload is a plain JSON-safe dict so it can be
    stored inside the checkpoint (``weights_only=True``-compatible).
    """
    if tokenizer is None:
        return _fallback_tokenizer_payload()
    if isinstance(tokenizer, _FallbackCharTokenizer) \
            or hasattr(tokenizer, '_itos'):
        # Any char-tokenizer with an id->char table can be represented
        # exactly; do not silently replace a custom vocabulary with the
        # default fallback.
        return _fallback_tokenizer_payload(tokenizer)
    name = getattr(tokenizer, 'name_or_path', None)
    if not isinstance(name, str) or not name.strip():
        # No reconstructable name (e.g. an in-memory custom tokenizer);
        # the fallback is the only framework-free representation.
        return _fallback_tokenizer_payload()
    return {
        'type': 'hf',
        'name': name,
        'pad_token_id': getattr(tokenizer, 'pad_token_id', None),
        'unk_token_id': getattr(tokenizer, 'unk_token_id', None),
    }


def _tokenizer_from_payload(payload) -> object:
    """Rebuild the tokenizer described by a checkpoint/export payload."""
    if not isinstance(payload, dict):
        payload = {}
    if payload.get('type') == 'fallback_char':
        tok = _FallbackCharTokenizer()
        tok._itos = list(payload['itos'])
        tok._stoi = {c: i for i, c in enumerate(tok._itos)}
        tok.pad_token_id = int(payload.get('pad_token_id', 0))
        tok.unk_token_id = int(payload.get('unk_token_id', 1))
        return tok
    tok_name = payload.get('name', 'gpt2')
    if HAS_HF:
        try:
            return AutoTokenizer.from_pretrained(tok_name)
        except Exception:
            pass
    return _FallbackCharTokenizer()


def _export_lm_bundle(ckpt_path, export_dir, generation_defaults=None,
                      device='cpu'):
    """Package a trained LM checkpoint into a self-contained deployment bundle.

    Writes, into ``export_dir``:

      * ``config.json``        - :class:`HybridConfig` (``dataclasses.asdict``)
                                 needed to rebuild the model
      * ``pytorch_model.bin``  - model ``state_dict``
      * ``tokenizer.json``     - fallback-char ``itos`` (or an HF name marker)
      * ``generation_config.json`` - sampling defaults
                                 (temperature/top_k/top_p/eos/max_new_tokens + vocab)
      * ``provenance.json``    - environment/git/torch provenance

    Returns the export directory. The bundle is deliberately framework-lean:
    rebuilding needs only this repo plus ``torch`` (and ``transformers`` for
    the optional HF tokenizer path), so a trained model can be shipped and
    reloaded for inference without the training script.
    """
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = HybridConfig(**ckpt['cfg'])
    arch = ckpt.get('arch', 'hybrid')
    vocab = int(ckpt['vocab'])
    model = build_lm_model(
        arch, vocab, cfg,
        d_model=ckpt.get('model_d_model') if arch == 'softmax' else None)
    model.load_state_dict(ckpt['model'])
    model.eval()

    os.makedirs(export_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(export_dir, 'pytorch_model.bin'))
    write_json_atomic(dataclasses.asdict(cfg),
                      os.path.join(export_dir, 'config.json'),
                      indent=2, allow_nan=False)

    # Prefer the tokenizer recorded at TRAINING time. Only legacy
    # checkpoints (saved before this field existed) fall back to the
    # host's tokenizer, preserving their previous export behaviour.
    tok_payload = ckpt.get('tokenizer')
    if not isinstance(tok_payload, dict):
        if HAS_HF:
            try:
                _hf_tok = AutoTokenizer.from_pretrained("gpt2")
                tok_payload = _tokenizer_payload(_hf_tok)
            except Exception:
                tok_payload = _fallback_tokenizer_payload()
        else:
            tok_payload = _fallback_tokenizer_payload()
    write_json_atomic(tok_payload,
                      os.path.join(export_dir, 'tokenizer.json'),
                      indent=2, allow_nan=False)

    gen_cfg = {
        'temperature': 1.0, 'top_k': None, 'top_p': None,
        'eos_token_id': None, 'max_new_tokens': 32, 'vocab': vocab,
        # The bundle must be rebuildable without the training checkpoint:
        # record the architecture discriminator and (for the parameter-
        # matched dense baseline) its width alongside the weights.
        'arch': arch,
        'model_d_model': int(model.embed.embedding_dim),
    }
    if generation_defaults:
        for key in ('temperature', 'top_k', 'top_p', 'eos_token_id',
                    'max_new_tokens'):
            if key in generation_defaults:
                gen_cfg[key] = generation_defaults[key]
    write_json_atomic(gen_cfg,
                      os.path.join(export_dir, 'generation_config.json'),
                      indent=2, allow_nan=False)
    write_json_atomic(capture_provenance(),
                      os.path.join(export_dir, 'provenance.json'),
                      indent=2, allow_nan=False)
    return export_dir


def load_lm_for_inference(export_dir, device='cpu'):
    """Reload a deployment bundle produced by :func:`_export_lm_bundle`.

    Returns ``(model, tokenizer, generation_config)`` with the model in
    ``eval()`` on ``device``, ready for :meth:`LMWithHybrid.generate`.
    """
    with open(os.path.join(export_dir, 'config.json')) as f:
        cfg = HybridConfig(**json.load(f))
    with open(os.path.join(export_dir, 'tokenizer.json')) as f:
        tok_payload = json.load(f)
    tok = _tokenizer_from_payload(tok_payload)
    if getattr(tok, "pad_token", None) is None \
            and hasattr(tok, "eos_token"):
        tok.pad_token = tok.eos_token
    with open(os.path.join(export_dir, 'generation_config.json')) as f:
        gen_config = json.load(f)
    vocab = int(gen_config.get('vocab', len(tok)))
    arch = gen_config.get('arch', 'hybrid')
    d_model = (gen_config.get('model_d_model')
               if arch in ('softmax', 'dense') else None)
    with open(os.path.join(export_dir, 'pytorch_model.bin'), 'rb') as f:
        state_dict = torch.load(f, map_location='cpu', weights_only=True)
    model = build_lm_model(arch, vocab, cfg, d_model=d_model)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, tok, gen_config


class _DenseSoftmaxStack(nn.Module):
    """Dense causal-softmax stack with the interface the LM loop expects.

    The training loop resets recurrent state before every micro-step
    (``model.hybrid.reset_state()``) and prints the layout
    (``model.hybrid.layout_str()``); this wrapper provides both so the
    softmax baseline is a drop-in architecture for the SAME loop without
    special-casing call sites. Attention is stateless, so ``reset_state``
    is a no-op.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int = 4):
        super().__init__()
        from run_quality import ResidualAttnLayer, SoftmaxAttn
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}")
        head_dim = d_model // n_heads
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList([
            ResidualAttnLayer(
                SoftmaxAttn(d_model, H=n_heads, K=head_dim, V=head_dim),
                d_model)
            for _ in range(self.n_layers)
        ])

    def reset_state(self):
        return None

    def layout_str(self):
        return '-'.join(['SOFTMAX'] * self.n_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SoftmaxBaselineLM(nn.Module):
    """Dense causal-softmax LM: the simplest real token-mixing baseline.

    Matched to ``LMWithHybrid`` on embedding/head tying, embedding scale,
    final norm and position-less sequence mixing, so a hybrid-vs-baseline
    comparison changes ONLY the mixing architecture (the scientific
    contrast), not the tokenizer, data, optimizer, budget or parameter
    tying. Softmax attention is stateless: decode cost grows with context
    (KV cache), unlike the recurrent hybrid's constant-size state — that
    is exactly one of the pre-registered predictions
    (``LM_HYPOTHESES['decode_state_efficiency']``).
    """

    def __init__(self, vocab_size, cfg: HybridConfig, n_heads: int = 4,
                 embed_scale=None, d_model=None):
        super().__init__()
        if cfg.layout is not None:
            total_layers = len(cfg.layout)
        else:
            total_layers = cfg.n_kda + cfg.n_csa + cfg.n_hca
        # Default: matched WIDTH (cfg.d_model). An explicit ``d_model``
        # supports the optional parameter-matched control (the dense
        # baseline is otherwise larger than the hybrid at equal width).
        self.d_model = int(d_model if d_model is not None else cfg.d_model)
        self.embed = nn.Embedding(vocab_size, self.d_model)
        self.hybrid = _DenseSoftmaxStack(self.d_model, total_layers,
                                         n_heads=n_heads)
        self.norm_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if embed_scale is None:
            embed_scale = math.sqrt(self.d_model)
        if (not isinstance(embed_scale, (int, float))
                or not math.isfinite(embed_scale) or embed_scale <= 0):
            raise ValueError(
                f"embed_scale must be a positive finite number, got "
                f"{embed_scale!r}.")
        self.embed_scale = float(embed_scale)
        # The baseline has no recurrent stack to checkpoint, so keep the
        # attribute for parity with LMWithHybrid (and to make the recorded
        # recipe explicit).
        self.ckpt_policy = 'none'

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids) * self.embed_scale
        x = self.hybrid(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids, *, max_new_tokens=32, temperature=1.0,
                 top_k=None, top_p=None, eos_token_id=None, seed=None):
        """Autoregressive generation, same contract as the hybrid LM.

        Export bundles advertise ``generate`` for every architecture, so the
        dense baseline must implement it too; the implementation is shared
        (fully recomputed prefix, state reset, training mode restored).
        """
        return LMWithHybrid.generate(
            self, input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token_id=eos_token_id, seed=seed)


def build_lm_model(arch, vocab_size, cfg: HybridConfig, ckpt_policy=None,
                   d_model=None):
    """Construct the pre-registered LM architecture.

    ``hybrid`` is the KDA/CSA/HCA attention stack; ``softmax`` is the
    matched dense causal-attention baseline. Keeping the choice in ONE
    factory guarantees the overfit gate, the real model and checkpoint
    reload all build the same architecture.
    """
    if arch in ('hybrid', 'kda', 'mamba2', 'rwkv7'):
        if d_model is not None:
            raise ValueError(
                "d_model override applies only to the softmax baseline")
        return LMWithHybrid(vocab_size, cfg, ckpt_policy=ckpt_policy)
    if arch in ('softmax', 'dense'):
        if ckpt_policy not in (None, 'none'):
            raise ValueError(
                f"arch={arch!r} has no recurrent stack to checkpoint; "
                f"got ckpt_policy={ckpt_policy!r}.")
        return SoftmaxBaselineLM(vocab_size, cfg, d_model=d_model)
    raise ValueError(
        f"unknown arch={arch!r}; expected 'hybrid' or 'softmax'.")


def _param_matched_softmax_width(vocab_size, cfg: HybridConfig,
                                 target_params, n_heads=4):
    """Smallest-width dense baseline closest to ``target_params``.

    Width (``d_model``) is the only free baseline dimension that preserves
    the matched depth/data/optimizer/budget protocol. Parameter counts are
    monotone in width, so a binary search over ``d_model = k * n_heads``
    finds the closest match; both the requested target and the achieved
    count are recorded in the run record. Returns ``(d_model, n_params)``.
    """
    if not isinstance(target_params, int) or target_params <= 0:
        raise ValueError(
            f"target_params must be a positive int, got {target_params!r}")

    def build(d):
        return SoftmaxBaselineLM(vocab_size, cfg, n_heads=n_heads,
                                 d_model=d)

    def n_params(d):
        return count_params(build(d))

    k_lo, k_hi = 1, max(4, cfg.d_model // n_heads * 4)
    while n_params(k_hi * n_heads) < target_params:
        k_hi *= 2
    while k_lo < k_hi:
        k_mid = (k_lo + k_hi) // 2
        if n_params(k_mid * n_heads) < target_params:
            k_lo = k_mid + 1
        else:
            k_hi = k_mid
    candidates = [k for k in (k_lo - 1, k_lo, k_lo + 1) if k >= 1]
    best_k = min(candidates,
                 key=lambda k: abs(n_params(k * n_heads) - target_params))
    d = best_k * n_heads
    return d, n_params(d)


# Pre-registered LM-scale hypotheses. These are recorded in every run
# record BEFORE the test split is opened, so the comparison cannot be
# re-framed after seeing the numbers. Verdicts are computed by the
# multi-seed analysis over matched-budget runs (hybrid vs softmax), not
# asserted per single run.
LM_HYPOTHESES = OrderedDict([
    ('matched_budget_quality', {
        'hypothesis': (
            'At matched width/depth, step/token budget, data and optimizer '
            'settings, the KDA/CSA/HCA hybrid matches the dense '
            'causal-softmax LM on held-out token loss.'),
        'prediction': (
            'Paired-seed test perplexity difference (hybrid - softmax) is '
            'within the paired 95% CI of zero, or the hybrid is better. '
            'By default width/depth are matched and the dense baseline is '
            'typically the LARGER model (a hybrid win is then conservative); '
            'parameter counts are always recorded, and '
            '--match_baseline_params shrinks the baseline width to match '
            'them for a strict equal-parameter control.'),
        'falsified_if': (
            'The hybrid is worse than the softmax baseline by more than the '
            'paired 95% CI at the matched budget.'),
        'status_rule': ('paired test perplexity over seeds, matched budget '
                        '(hybrid vs softmax)'),
    }),
    ('decode_state_efficiency', {
        'hypothesis': (
            'The recurrent hybrid decodes with constant-size state, whereas '
            'dense softmax attention grows a KV cache with context.'),
        'prediction': (
            'Hybrid decode bytes/token are flat in context length and lower '
            'than the softmax baseline at long context.'),
        'falsified_if': (
            'Hybrid decode bytes/token grow with context or are not lower '
            'than the softmax baseline at the longest measured context.'),
        'status_rule': ('measured decode cache bytes/token vs context '
                        'length (exp3_kv_cache / decode consistency)'),
    }),
])


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

    Each sampled text is framed with its index and byte length before
    hashing. Without the length prefix, concatenating the raw strings made
    the fingerprint ambiguous at text boundaries: ``["ab", "c"]`` and
    ``["a", "bc"]`` hashed the same byte stream and therefore received the
    same data-version id even though they are different corpora.
    """
    n_probe = _exact_int_arg('n_probe', n_probe, minimum=0)
    max_chars_per_text = _exact_int_arg(
        'max_chars_per_text', max_chars_per_text, minimum=1)
    n = len(texts)
    indices = _evenly_spaced_probe_indices(n, n_probe)
    h = hashlib.sha256()
    h.update(
        f"n={n};probe={len(indices)};max_chars={max_chars_per_text}".encode())
    for i in indices:
        chunk = str(texts[i])[:max_chars_per_text].encode(
            'utf-8', 'replace')
        # Explicit framing: index, byte length, then the bytes. The index
        # also distinguishes a reordered probe from the original one.
        h.update(b'\x00i=')
        h.update(str(int(i)).encode('ascii'))
        h.update(b';len=')
        h.update(str(len(chunk)).encode('ascii'))
        h.update(b'\x00')
        h.update(chunk)
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
    grad_flow = None
    update_report = None
    t0 = time.time()
    for step in range(steps):
        model.hybrid.reset_state()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if loss is None or not torch.isfinite(loss):
            raise_non_finite(
                'loss', loss=(loss if loss is not None else float('nan')),
                token_ids={'input_ids': input_ids},
                id_range=(0, int(getattr(model.embed, 'num_embeddings',
                                         input_ids.max().item() + 1))),
                params=model.parameters(), logits=out.get("logits"),
                context=f'overfit sanity check step={step} seed={seed}')
        with torch.no_grad():
            mask = labels.ne(-100)
            correct = ((out["logits"].argmax(-1) == labels) & mask).sum()
            acc = float(correct) / max(1, int(mask.sum()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_flow is None:
            # One-shot gradient-flow audit on the first backward: turns the
            # pass/fail verdict into a diagnostic for a detached / dead
            # branch (a model can still memorize through the layers that DO
            # receive gradients while another operator silently gets none).
            grad_flow = grad_flow_report(model.named_parameters())
        bad_grads = triage_gradient(model.parameters())
        clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   grad_clip)
        if bad_grads or not torch.isfinite(clip_norm):
            raise_non_finite(
                'gradient', loss=loss,
                token_ids={'input_ids': input_ids},
                id_range=(0, int(getattr(model.embed, 'num_embeddings',
                                         input_ids.max().item() + 1))),
                params=model.parameters(), logits=out.get("logits"),
                grads={f'grad[{i}]': p.grad
                       for i, p in enumerate(bad_grads)},
                grad_norm=float(clip_norm),
                context=f'overfit sanity check step={step} seed={seed}')
        before_step = None
        if update_report is None:
            before_step = snapshot_parameters(model.named_parameters())
        opt.step()
        if update_report is None:
            update_report = optimizer_update_report(
                before_step, model.named_parameters(), opt, step=step)
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
    if update_report is not None and not update_report['ok']:
        passed = False
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
        # Gradient-flow audit (first step): which trainable tensors the
        # loss actually reached. A missing / zero gradient names a detached
        # or dead branch that the loss-based pass criterion could mask.
        "grad_flow": grad_flow,
        # Optimizer coverage/update audit (first step): catches a parameter
        # group that omits a trainable tensor, or a param with a non-zero
        # gradient that the step never moved.
        "optimizer_updates": update_report,
        "note": ("Necessary but NOT sufficient: passing proves gradients flow "
                 "and the loss responds on a fixed batch; it cannot rule out "
                 "data-augmentation / feature-alignment / label-mapping bugs."),
    }
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if grad_flow and grad_flow["n_missing_grad"]:
        print(f"[overfit-check] WARNING: {grad_flow['n_missing_grad']}/"
              f"{grad_flow['n_trainable_tensors']} trainable tensor(s) "
              f"received NO gradient on the fixed batch "
              f"(e.g. {grad_flow['missing_grad_names'][:3]}); a detached or "
              f"unused branch can hide behind a passing loss curve.",
              flush=True)
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
    arch = ckpt.get('arch', 'hybrid')
    if arch == 'softmax':
        raise SystemExit(
            "checkpoint arch='softmax' is stateless: the prefill-decode "
            "consistency probe applies only to the recurrent hybrid.")
    model = build_lm_model(
        arch, int(ckpt['vocab']), cfg,
        d_model=ckpt.get('model_d_model') if arch == 'softmax' else None)
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


def _accuracy_regime_for_history(val_history, *, ceiling=1.0):
    """Accuracy regime with ``val_acc_best_step`` in optimizer steps.

    ``train_diagnostics.accuracy_regime`` works on bare accuracy sequences
    and therefore reports the peak's INDEX. Records built from a
    validation history must map that index back through the same filtered
    points and store the actual step, as ``ValidationCurve`` does.
    """
    points = [h for h in val_history
              if h.get('token_accuracy') is not None]
    regime = accuracy_regime(
        [h['token_accuracy'] for h in points], ceiling=ceiling)
    if regime is not None:
        regime['val_acc_best_step'] = int(
            points[int(regime['val_acc_best_step'])]['step'])
    return regime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--autodl", action="store_true")
    parser.add_argument("--arch", choices=("hybrid", "softmax"),
                        default="hybrid",
                        help="LM architecture: 'hybrid' (KDA/CSA/HCA stack) "
                             "or 'softmax' (matched dense causal-attention "
                             "baseline; the pre-registered control). "
                             "Default: hybrid.")
    parser.add_argument("--match_baseline_params", action="store_true",
                        help="Only with --arch softmax: choose the baseline "
                             "width so its parameter count matches the "
                             "hybrid's at the same depth (default: matched "
                             "WIDTH/depth instead; the dense baseline is "
                             "then larger, which is conservative for a "
                             "hybrid win). The target/achieved parameter "
                             "counts and widths are recorded.")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Cap the raw corpus at this many documents BEFORE "
                             "cleaning/splitting (0 = the recipe default "
                             "budget). Useful for smoke runs; the cap is "
                             "recorded in the data record.")
    parser.add_argument("--prefetch_cache", action="store_true",
                        help="Enable the token-id prefetch cache in "
                             "TinyStoriesLM: encode each sample once and "
                             "reuse the cached ids on later __getitem__ "
                             "calls (prefetch), cutting repeated tokenizer "
                             "encode work in the training hot path. Off by "
                             "default so existing runs are bit-identical; a "
                             "correctness self-check re-encodes every 64th "
                             "cache hit and asserts equality with the cache.")
    parser.add_argument("--prefetch_cap", type=int, default=1024,
                        help="Max number of samples held in the prefetch "
                             "cache (0 = unbounded). Default 1024.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lm")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Append-only run-manifest (JSONL) path. Default: "
                             "env RUN_MANIFEST_PATH, else "
                             "results/run_manifest.jsonl. One line per run "
                             "links code revision, hyperparameters, seeds, "
                             "data fingerprints, artifacts and final "
                             "results — the machine-readable experiment log.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model init and DataLoader shuffle.")
    # --- text generation (the trained LM can also produce text) -----------
    parser.add_argument("--generate", action="store_true",
                        help="Load --checkpoint and generate text from --prompt "
                             "instead of training. Runs a single generation "
                             "pass and exits.")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to continue with --generate.")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Number of tokens to generate with --generate.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for --generate (<=0 = greedy).")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k cutoff for --generate sampling.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus (top-p) cutoff for --generate sampling.")
    parser.add_argument("--eos_token_id", type=int, default=None,
                        help="Token id that stops --generate early when sampled.")
    parser.add_argument("--generate_seed", type=int, default=None,
                        help="RNG seed for --generate sampling (deterministic).")
    parser.add_argument("--generate_max_context", type=int, default=512,
                        help="Max prompt tokens fed to --generate (truncated).")
    # --- deployment: package a trained checkpoint into a self-contained
    # bundle that can be reloaded and used to generate outside this script. --
    parser.add_argument("--export", action="store_true",
                        help="Load --checkpoint and write a self-contained "
                             "deployment bundle (config + weights + tokenizer "
                             "desc + generation config + provenance) to "
                             "--export_dir, then exit without training.")
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Destination directory for --export (default: "
                             "--output_dir).")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of the cleaned corpus held out as the "
                             "validation split (model selection + early "
                             "stopping). Default 0.1.")
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of the cleaned corpus sealed as the "
                             "test split (opened exactly once, after the "
                             "model is frozen). Default 0.1.")
    parser.add_argument("--chronological_split", action="store_true",
                        help="Split WITHOUT shuffling: the earliest documents "
                             "form train, the next block validation, the most "
                             "recent block test. Use this for time-ordered "
                             "corpora (prevent future->past leakage). "
                             "Default: off (i.i.d. story split via a seeded "
                             "shuffle).")
    parser.add_argument("--dedup_windows", action="store_true",
                        help="Additionally drop documents whose truncated "
                             "seq_len+1 token window was already seen (the "
                             "sample TinyStoriesLM actually trains on), not "
                             "just exact string duplicates. Off by default: "
                             "with a small seq_len many distinct long "
                             "documents share a prefix, so enable it only "
                             "when prefix repetition is genuine leakage.")
    parser.add_argument("--strict_window_dedup", action="store_true",
                        help="Refuse to train when the audited train/val/"
                             "test splits share any identical truncated "
                             "model-input window. Pair with "
                             "--dedup_windows for a corpus where distinct "
                             "documents commonly share a prefix.")
    parser.add_argument("--dedup_near", action="store_true",
                        help="Drop NEAR-duplicate documents (token n-gram "
                             "Jaccard >= --near_dup_threshold) BEFORE the "
                             "split. Exact/normalized dedup cannot see a "
                             "story stored again with a few words changed; "
                             "such a near-twin is a leak when the two copies "
                             "land on different sides of train/eval. Off by "
                             "default for backward compatibility; the "
                             "cross-split near-duplicate AUDIT is always on "
                             "unless --skip_near_dup_audit is passed.")
    parser.add_argument("--near_dup_threshold", type=float,
                        default=DEFAULT_NEAR_DUP_THRESHOLD,
                        help="Pre-specified token n-gram Jaccard threshold "
                             "for near-duplicate detection (default 0.8). "
                             "Must be in (0, 1]; do not tune it on the "
                             "observed held-out pairs.")
    parser.add_argument("--near_dup_shingle", type=int,
                        default=DEFAULT_NEAR_DUP_SHINGLE_SIZE,
                        help="Token n-gram width for near-duplicate "
                             "detection (default 5).")
    parser.add_argument("--near_dup_max_candidates", type=int,
                        default=DEFAULT_NEAR_DUP_MAX_CANDIDATES,
                        help="Cap on LSH candidate pairs re-scored by the "
                             "near-duplicate audit (default 200000). When "
                             "the cap is hit the audit reports a lower "
                             "bound, and --strict_near_dup refuses to "
                             "certify the split.")
    parser.add_argument("--strict_near_dup", action="store_true",
                        help="Refuse to train when the audited train/val/"
                             "test splits share any near-duplicate pair at "
                             "Jaccard >= --near_dup_threshold. Pair with "
                             "--dedup_near to remove them before the split.")
    parser.add_argument("--skip_near_dup_audit", action="store_true",
                        help="Skip the (default-on) cross-split "
                             "near-duplicate audit. Not recommended: without "
                             "it the leakage-safe split guarantee covers "
                             "only exact duplicates and identical truncated "
                             "windows.")
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
    parser.add_argument("--kda_decay_init", choices=("default", "log_uniform"),
                        default=None,
                        help="KDA per-channel log-decay initialization: "
                             "'default' (historical zero bias) or "
                             "'log_uniform' (RWKV-style multi-timescale "
                             "spectrum; zero extra parameters). Default: "
                             "HybridConfig's 'default'.")
    parser.add_argument("--kda_decay_half_life_min", type=float, default=None,
                        help="Fastest channel half-life in tokens for "
                             "--kda_decay_init log_uniform (default 1.0).")
    parser.add_argument("--kda_decay_half_life_max", type=float, default=None,
                        help="Slowest channel half-life in tokens for "
                             "--kda_decay_init log_uniform (default 1024.0).")
    parser.add_argument("--ckpt_policy",
                        choices=grad_ckpt.CHECKPOINT_POLICIES, default="none",
                        help="Gradient-checkpointing policy for the hybrid "
                             "stack: 'none' (default; legacy stateful "
                             "forward, no recompute), 'all' (checkpoint every "
                             "layer), 'memory' (only the kda/mamba2/rwkv7 "
                             "layers), 'retrieval' (only the csa/hca "
                             "layers). Trades one extra recompute forward "
                             "per checkpointed layer for lower retained-"
                             "activation memory; outputs and parameter "
                             "gradients are value-identical to 'none' (see "
                             "test_lm_ckpt_policy.py).")
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
    if args.max_docs < 0:
        raise SystemExit("--max_docs must be >= 0")

    arch = args.arch
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

    if args.generate:
        ckpt_path = args.checkpoint or os.path.join(
            args.output_dir, "final_lm.pt")
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        if not args.prompt:
            raise SystemExit("--generate requires --prompt")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg = HybridConfig(**ckpt['cfg'])
        arch = ckpt.get('arch', 'hybrid')
        vocab = int(ckpt['vocab'])
        model = build_lm_model(
            arch, vocab, cfg,
            d_model=ckpt.get('model_d_model') if arch == 'softmax' else None)
        model.load_state_dict(ckpt['model'])
        model.to(device).eval()
        tok_payload = ckpt.get('tokenizer')
        if isinstance(tok_payload, dict):
            # Rebuild the vocabulary the model was TRAINED with, not the
            # host's installed tokenizer.
            tokenizer = _tokenizer_from_payload(tok_payload)
        elif HAS_HF:
            try:
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception:
                tokenizer = _FallbackCharTokenizer()
        else:
            tokenizer = _FallbackCharTokenizer()
        if getattr(tokenizer, "pad_token", None) is None \
                and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        prompt_ids = tokenizer.encode(args.prompt, truncation=True,
                                      max_length=args.generate_max_context)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                 device=device)
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=args.eos_token_id,
            seed=args.generate_seed,
        )[0]
        full = list(prompt_ids) + gen_ids
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full, skip_special_tokens=True)
        result = {
            'checkpoint': ckpt_path,
            'prompt': args.prompt,
            'prompt_ids': prompt_ids,
            'generated_ids': gen_ids,
            'generated_text': gen_text,
            'full_text': full_text,
            'max_new_tokens': int(args.max_new_tokens),
            'temperature': float(args.temperature),
            'top_k': (None if args.top_k is None else int(args.top_k)),
            'top_p': (None if args.top_p is None else float(args.top_p)),
            'eos_token_id': (
                None if args.eos_token_id is None else int(args.eos_token_id)),
            'generate_seed': (
                None if args.generate_seed is None else int(args.generate_seed)),
            'provenance': provenance,
        }
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, 'generation.json')
        write_json_atomic(result, out_path, indent=2, allow_nan=False)
        print(f"[generate] prompt={args.prompt!r}")
        print(f"[generate] generated={gen_text!r}")
        print(f"[generate] full={full_text!r}")
        print(f"[generate] saved={out_path}")
        return

    if args.export:
        ckpt_path = args.checkpoint or os.path.join(
            args.output_dir, "final_lm.pt")
        export_dir = args.export_dir or args.output_dir
        generation_defaults = {
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'eos_token_id': args.eos_token_id,
            'max_new_tokens': args.max_new_tokens,
        }
        _export_lm_bundle(ckpt_path, export_dir,
                          generation_defaults=generation_defaults,
                          device=device)
        print(f"[export] bundle written to {export_dir}")
        # Optional smoke generation from the reloaded bundle to prove the
        # deployment artifact round-trips end to end.
        if args.prompt:
            model, tok, gen_cfg = load_lm_for_inference(
                export_dir, device=device)
            prompt_ids = tok.encode(args.prompt, truncation=True,
                                    max_length=args.generate_max_context)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                     device=device)
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=int(gen_cfg.get('max_new_tokens',
                                               args.max_new_tokens)),
                temperature=float(gen_cfg.get('temperature',
                                              args.temperature)),
                top_k=(None if gen_cfg.get('top_k') is None
                       else int(gen_cfg['top_k'])),
                top_p=(None if gen_cfg.get('top_p') is None
                       else float(gen_cfg['top_p'])),
                eos_token_id=(None if gen_cfg.get('eos_token_id') is None
                              else int(gen_cfg['eos_token_id'])),
                seed=args.generate_seed,
            )[0]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            print(f"[export] prompt={args.prompt!r}")
            print(f"[export] generated={gen_text!r}")
        return

    if HAS_HF:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            print(f"[train_lm_autodl] GPT-2 tokenizer load failed ({e}); "
                  "using the char-level fallback tokenizer over synthetic "
                  "text")
            tokenizer = _FallbackCharTokenizer()
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
    if args.kda_decay_init is not None:
        route_kw['kda_decay_init'] = args.kda_decay_init
    if args.kda_decay_half_life_min is not None:
        route_kw['kda_decay_half_life_min'] = args.kda_decay_half_life_min
    if args.kda_decay_half_life_max is not None:
        route_kw['kda_decay_half_life_max'] = args.kda_decay_half_life_max
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

    # Optional parameter-matched control: the dense baseline is LARGER than
    # the hybrid at equal width, so by default we match width/depth and
    # report the asymmetry. This flag instead shrinks the baseline's width
    # until the parameter counts match (depth/data/budget unchanged).
    baseline_d_model = None
    param_match = None
    if args.match_baseline_params and arch != 'softmax':
        raise SystemExit(
            "--match_baseline_params requires --arch softmax")
    if args.match_baseline_params:
        target_params = count_params(LMWithHybrid(vocab_size, cfg))
        baseline_d_model, achieved = _param_matched_softmax_width(
            vocab_size, cfg, target_params)
        param_match = {
            'enabled': True,
            'target_model': 'hybrid',
            'target_params': int(target_params),
            'achieved_params': int(achieved),
            'relative_error': (abs(achieved - target_params)
                               / max(1, target_params)),
            'hybrid_d_model': int(cfg.d_model),
            'baseline_d_model': int(baseline_d_model),
            'depth': ('cfg.layout' if cfg.layout is not None
                      else 'n_kda+n_csa+n_hca'),
        }
        print(f"Param-matched baseline: d_model={baseline_d_model} "
              f"({achieved/1e6:.2f}M) vs hybrid {target_params/1e6:.2f}M")

    if cfg.layout is not None:
        layout_str = '-'.join(k.upper() for k in cfg.layout)
        print(f"Hybrid layout: explicit {len(cfg.layout)} layers "
              f"[{layout_str}], d={cfg.d_model}, "
              f"memory={cfg.memory_type}")
    else:
        print(f"Hybrid layout: {cfg.n_kda}:{cfg.n_csa}:{cfg.n_hca} "
              f"(memory={cfg.memory_type}), d={cfg.d_model}")
    print(f"Training profile: batch_size={batch_size}, grad_accum={grad_accum}, "
          f"seq_len={seq_len}, optimizer_steps={max_steps}, seed={args.seed}, "
          f"ckpt_policy={args.ckpt_policy}")

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
    if args.max_docs:
        corpus_budget = min(corpus_budget, int(args.max_docs))
    raw_texts, data_source = _load_raw_texts(corpus_budget, split="train")
    cleaning = clean_lm_texts(
        raw_texts, tokenizer, seq_len,
        dedup_windows=bool(args.dedup_windows),
        dedup_near=bool(args.dedup_near),
        near_dup_threshold=args.near_dup_threshold,
        near_dup_shingle_size=args.near_dup_shingle,
        near_dup_max_candidates=args.near_dup_max_candidates)
    clean_texts = cleaning['texts']
    if len(clean_texts) < 10:
        raise SystemExit(
            f"only {len(clean_texts)} usable documents after cleaning "
            f"(raw={len(raw_texts)}); refusing to train on a degenerate "
            f"corpus")
    split, split_report = split_clean_texts(
        clean_texts, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, chronological=bool(args.chronological_split))
    train_texts = [clean_texts[i] for i in split['train']]
    val_texts = [clean_texts[i] for i in split['val']]
    test_texts = [clean_texts[i] for i in split['test']]
    # Train is authoritative: drop held-out rows whose model-input window
    # already exists in train (and test rows that duplicate kept val).
    leak_filter = _filter_leaking_heldout(
        train_texts, val_texts, test_texts, tokenizer, seq_len)
    val_texts = leak_filter['val_texts']
    test_texts = leak_filter['test_texts']
    if leak_filter['applied']:
        print(f"[data] window-leak filter: dropped "
              f"{leak_filter['val_dropped']} val / "
              f"{leak_filter['test_dropped']} test row(s) whose "
              f"truncated window already appeared in train; kept "
              f"val={len(val_texts)} test={len(test_texts)}")
    window_audit = _window_overlap_audit(
        train_texts, val_texts, test_texts, tokenizer, seq_len)
    if window_audit['any']:
        print(f"[data] WARNING: {window_audit['total_cross_split']} "
              f"identical model-input window(s) still shared across splits "
              f"(train/val={window_audit['train_val']}, "
              f"train/test={window_audit['train_test']}, "
              f"val/test={window_audit['val_test']}); "
              f"use --dedup_windows to remove them before the split")
        if args.strict_window_dedup:
            raise SystemExit(
                "refusing to train: cross-split model-window leakage "
                f"detected ({window_audit['total_cross_split']} duplicate "
                "window(s)); rerun with --dedup_windows or relax with "
                "--strict_window_dedup off")

    # Near-duplicate leakage audit. Exact/normalized/window audits above
    # catch identical content; this catches the same story with edits.
    # It reads only text/token content (no model metric), so it can run
    # while the test split is still sealed for training.
    if args.strict_near_dup and args.skip_near_dup_audit:
        raise SystemExit(
            "--strict_near_dup requires the near-duplicate audit; remove "
            "--skip_near_dup_audit")
    near_audit = None
    if not args.skip_near_dup_audit:
        near_audit = near_duplicate_leakage_audit(
            train_texts, val_texts, test_texts, tokenizer, seq_len,
            threshold=args.near_dup_threshold,
            shingle_size=args.near_dup_shingle,
            max_candidate_pairs=args.near_dup_max_candidates)
        truncated = bool(
            near_audit['detector']['candidates_truncated'])
        if truncated:
            print("[data] WARNING: near-duplicate candidate budget was "
                  "exhausted; the reported cross-split pair count is a "
                  "lower bound (raise --near_dup_max_candidates for a "
                  "full audit)")
            if args.strict_near_dup:
                raise SystemExit(
                    "refusing to certify the split: the near-duplicate "
                    "candidate budget was exhausted "
                    f"({near_audit['detector']['n_candidate_pairs']} "
                    "candidate pairs), so the audit cannot prove the "
                    "absence of cross-split near duplicates; raise "
                    "--near_dup_max_candidates")
        if near_audit['n_cross_split_pairs']:
            print(f"[data] WARNING: {near_audit['n_cross_split_pairs']} "
                  f"cross-split near-duplicate pair(s) at token "
                  f"Jaccard >= {args.near_dup_threshold} "
                  f"({near_audit['by_split_pair']}); a held-out near-twin "
                  f"of a train document inflates the held-out metric. "
                  f"Use --dedup_near to remove near duplicates before the "
                  f"split, or --strict_near_dup to refuse to train.")
            if args.strict_near_dup:
                raise SystemExit(
                    "refusing to train: cross-split near-duplicate leakage "
                    f"detected ({near_audit['n_cross_split_pairs']} pair(s) "
                    f"at Jaccard >= {args.near_dup_threshold}); rerun with "
                    "--dedup_near or relax --strict_near_dup")

    # Per-split length/unk statistics. Descriptive distribution-shift
    # diagnostics ONLY; they are never fitted or applied to features. The
    # test split's stats are computed with its single evaluation at the
    # very end (see below) — not here.
    stats_train = _text_length_stats(train_texts, tokenizer, seq_len)
    stats_val = _text_length_stats(val_texts, tokenizer, seq_len)
    shift_val = _distribution_shift(stats_train, stats_val)
    # Robust outlier fences are fitted on TRAIN lengths only, then applied
    # to the held-out probes. This is a diagnostic, not a transform.
    train_lengths = _probe_token_lengths(train_texts, tokenizer, seq_len)
    val_lengths = _probe_token_lengths(val_texts, tokenizer, seq_len)
    outlier_val = _length_outlier_audit(train_lengths, val_lengths)
    # Vocabulary/domain shift: length balance is necessary, not sufficient.
    # All reference statistics are fitted on TRAIN; the null folds are
    # drawn from TRAIN, so this never looks at the sealed test split.
    token_shift_val = token_distribution_audit(
        train_texts, val_texts, tokenizer, seq_len, seed=args.seed)

    ds = TinyStoriesLM(tokenizer, seq_len=seq_len, texts=train_texts,
                       split="train", name="train",
                       prefetch_cache=args.prefetch_cache,
                       prefetch_cap=args.prefetch_cap)
    val_ds = TinyStoriesLM(tokenizer, seq_len=seq_len, texts=val_texts,
                           split="validation", name="val")
    # The test split stays unopened until the model is frozen. Only a
    # factory (plus the index list) exists during training.
    sealed_test = SealedTestSet(
        lambda: TinyStoriesLM(
            tokenizer, seq_len=seq_len, texts=test_texts,
            split="test", name="test"))

    print(f"Data: {data_source}, raw={len(raw_texts)}, "
          f"cleaned={len(clean_texts)} "
          f"(empty={cleaning['stats']['n_empty']}, "
          f"nonfinite={cleaning['stats']['n_nonfinite']}, "
          f"dup={cleaning['stats']['n_duplicates']}, "
          f"window_dup={cleaning['stats']['n_window_duplicates']}, "
          f"near_dup={cleaning['stats']['n_near_duplicates']}, "
          f"too_short={cleaning['stats']['n_too_short_to_supervise']})")
    print(f"Split (seed={args.seed}, "
          f"chronological={bool(args.chronological_split)}): "
          f"train={len(train_texts)} "
          f"val={len(val_texts)} test={len(test_texts)} "
          f"[val/test sealed until final evaluation]")
    print(f"Fingerprints: train={ds.data_fingerprint[:16]}... "
          f"val={val_ds.data_fingerprint[:16]}...")
    if shift_val['flag']:
        _d = shift_val['cohen_d']
        _d_text = ('undefined (zero within-split variance)'
                   if _d is None else f'{_d:.3f}')
        _mean_diff = shift_val['mean_difference']
        _mean_text = ('unrepresentable (outside float range)'
                      if _mean_diff is None else f'{_mean_diff:.3f}')
        print(f"[data] WARNING: val token-length distribution differs from "
              f"train (Cohen's d={_d_text}, mean difference={_mean_text}); "
              f"the held-out loss may not be exchangeable with train")
    if token_shift_val['flag']:
        _excess = token_shift_val['excess'] or {}
        print(f"[data] WARNING: val token distribution differs from train "
              f"beyond the within-train null "
              f"({token_shift_val['flag_reasons']}): excess unseen-token "
              f"mass={_excess.get('unseen_token_mass', 0.0):.4f}, "
              f"excess JS={_excess.get('js_divergence_bits', 0.0):.4f} bits")
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
            lambda: build_lm_model(arch, vocab_size, cfg,
                                   ckpt_policy=args.ckpt_policy,
                                   d_model=baseline_d_model),
            overfit_batch, device,
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
    model = build_lm_model(arch, vocab_size, cfg,
                           ckpt_policy=args.ckpt_policy,
                           d_model=baseline_d_model).to(device)
    if baseline_d_model is not None and model.embed.embedding_dim \
            != baseline_d_model:
        raise RuntimeError('baseline width override was not applied')
    params_count = count_params(model)
    print(f"Params: {params_count/1e6:.2f}M")
    print(f"Arch: {arch}")
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
        'arch': arch,
        'model_d_model': int(model.embed.embedding_dim),
        'param_match': param_match,
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
        'ckpt_policy': model.ckpt_policy,
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
            'chronological': bool(args.chronological_split),
            'group_ids': None,
            'chronological_order_verified':
                split_report['chronological_order_verified'],
            'note': ('Story-level deterministic split drawn with a '
                     'dedicated random.Random(seed), so the global RNG '
                     'streams driving model init / training batches are '
                     'untouched. With --chronological_split the corpus '
                     'order is preserved (past -> train, next -> val, '
                     'most recent -> test) so no future document leaks '
                     'into fitting or selection. Cleaning before the split '
                     'also removes exact/normalized duplicates; documents '
                     'whose truncated model window repeats are counted '
                     '(n_window_duplicates) and dropped only with '
                     '--dedup_windows; near-duplicates (token n-gram '
                     'Jaccard >= near_dup_threshold) are audited across '
                     'splits and dropped before the split only with '
                     '--dedup_near. No fitted preprocessing '
                     'transform exists in this pipeline; were one added '
                     '(e.g. a tokenizer or length normalizer), it must '
                     'be fitted on train indices only.'),
        },
        'near_dup_policy': {
            'pre_split_dedup': bool(args.dedup_near),
            'cross_split_audit': bool(not args.skip_near_dup_audit),
            'strict_gate': bool(args.strict_near_dup),
            'threshold': float(args.near_dup_threshold),
            'shingle_size': int(args.near_dup_shingle),
            'max_candidate_pairs': int(args.near_dup_max_candidates),
            'scope': ('truncated seq_len+1 model-input token windows; '
                      'token n-gram Jaccard'),
        },
        'near_duplicate_audit': near_audit,
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
                'n': len(test_texts),
                'content_fingerprint_sha256': None,
                'token_length_stats': None,
                'sealed_until_final_evaluation': True,
            },
        },
        'split_audit': split_report,
        'window_overlap_audit': window_audit,
        'window_leak_filter': {
            'applied': leak_filter['applied'],
            'train_authoritative': leak_filter['train_authoritative'],
            'val_dropped': leak_filter['val_dropped'],
            'test_dropped': leak_filter['test_dropped'],
            'val_fallback_all_leaking': leak_filter[
                'val_fallback_all_leaking'],
            'test_fallback_all_leaking': leak_filter[
                'test_fallback_all_leaking'],
            'n_val_before': leak_filter['n_val_before'],
            'n_test_before': leak_filter['n_test_before'],
            'n_val_after': leak_filter['n_val_after'],
            'n_test_after': leak_filter['n_test_after'],
            'note': leak_filter['note'],
        },
        'distribution_shift': {
            'val_vs_train': shift_val,
            'test_vs_train': None,
            'length_outliers': {
                'iqr_multiplier': 1.5,
                'fences_fit_on': 'train only',
                'val_vs_train': outlier_val,
                'test_vs_train': None,
            },
            'token_distribution': {
                'fit_on': 'train probe only',
                'null_calibration': 'within-train document folds',
                'val_vs_train': token_shift_val,
                'test_vs_train': None,
            },
        },
        'tokenizer': type(tokenizer).__name__,
        'vocab_size': vocab_size,
        'content_fingerprint_sha256': ds.data_fingerprint,
        'fingerprint_scheme': ('sha256 of <=64 evenly spaced texts truncated '
                               'to 4096 chars + corpus size, each framed by '
                               'index and byte length'),
        'prefetch_cache': bool(args.prefetch_cache),
        'prefetch_cap': int(args.prefetch_cap),
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
            "arch": arch,
            # Actual model width (differs from cfg.d_model only for the
            # parameter-matched softmax control).
            "model_d_model": int(model.embed.embedding_dim),
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
            "tokenizer": _tokenizer_payload(tokenizer),
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
            autocast_dtype=autocast_dtype, return_accuracy=True)
        if report['loss'] is None:
            raise RuntimeError(
                f'validation split produced no supervised tokens at step '
                f'{step}; cannot select a model')
        val_history.append({
            'step': int(step),
            'loss': report['loss'],
            'perplexity': report['perplexity'],
            # Token accuracy comes from the SAME held-out forward pass (no
            # extra compute): a loss that stops falling at an accuracy
            # ceiling is saturation, which the loss curve alone would
            # mislabel as overfitting. ``.get`` keeps legacy/monkeypatched
            # evaluators that predate ``return_accuracy`` working.
            'token_accuracy': report.get('token_accuracy'),
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
    test_lengths = _probe_token_lengths(test_dataset.texts, tokenizer,
                                        seq_len)
    outlier_test = _length_outlier_audit(train_lengths, test_lengths)
    # The test split is now open for its single final evaluation, so the
    # same vocabulary-shift audit can be recorded without peeking earlier.
    token_shift_test = token_distribution_audit(
        train_texts, test_dataset.texts, tokenizer, seq_len, seed=args.seed)
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
    if token_shift_test['flag']:
        _test_excess = token_shift_test['excess'] or {}
        print(f"[test] WARNING: the sealed test split's token distribution "
              f"differs from train beyond the within-train null "
              f"({token_shift_test['flag_reasons']}): excess unseen-token "
              f"mass={_test_excess.get('unseen_token_mass', 0.0):.4f}, "
              f"excess JS={_test_excess.get('js_divergence_bits', 0.0):.4f} "
              f"bits; the test perplexity partly reflects domain mismatch")
    data_record['splits']['test']['content_fingerprint_sha256'] = \
        test_dataset.data_fingerprint
    data_record['splits']['test']['token_length_stats'] = stats_test
    data_record['splits']['test']['sealed_until_final_evaluation'] = False
    data_record['distribution_shift']['test_vs_train'] = shift_test
    data_record['distribution_shift']['length_outliers']['test_vs_train'] = \
        outlier_test
    data_record['distribution_shift']['token_distribution'][
        'test_vs_train'] = token_shift_test

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
    # Accuracy-level regime: distinguishes a flat validation loss at the
    # task ceiling (converged) from a flat loss caused by overfitting.
    acc_regime = _accuracy_regime_for_history(val_history, ceiling=1.0)
    if acc_regime is not None:
        curve_diagnosis['accuracy_regime'] = acc_regime
    # Early-stopping accounting from the monitored curve: the model that is
    # finally evaluated is the best-validation checkpoint, so the gap
    # between the last check and that best point quantifies what stopping
    # at the budget endpoint would have cost.
    if val_history:
        best_check = min(val_history, key=lambda h: h['loss'])
        final_check = val_history[-1]
        curve_diagnosis['early_stopping_summary'] = {
            'selected_step': int(best_check['step']),
            'final_step': int(final_check['step']),
            'steps_saved': int(final_check['step'] - best_check['step']),
            'val_loss_at_selection': float(best_check['loss']),
            'val_loss_at_final': float(final_check['loss']),
            'val_loss_regret': float(final_check['loss']
                                     - best_check['loss']),
            'selected_by': 'min validation loss',
        }
        if best_check.get('token_accuracy') is not None:
            peak_acc = max(h['token_accuracy'] for h in val_history
                           if h.get('token_accuracy') is not None)
            curve_diagnosis['early_stopping_summary'].update({
                'val_acc_at_selection': float(best_check['token_accuracy']),
                'val_acc_at_final': (
                    float(final_check['token_accuracy'])
                    if final_check.get('token_accuracy') is not None
                    else None),
                # Accuracy left on the table because selection follows
                # validation LOSS, not accuracy: peak accuracy minus the
                # accuracy at the loss-selected checkpoint.
                'val_acc_regret_from_peak': float(
                    peak_acc - best_check['token_accuracy']),
            })
    print(f"[diagnosis] train/val curves: {curve_diagnosis['verdict']}")
    if acc_regime is not None:
        print(f"[diagnosis] held-out token accuracy: "
              f"{acc_regime['val_acc_head_mean']:.4f} -> "
              f"{acc_regime['val_acc_tail_mean']:.4f} "
              f"(peak {acc_regime['val_acc_peak']:.4f}, "
              f"saturated={acc_regime['saturated']})")
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
        "arch": arch,
        "created_utc": created_utc,
        "elapsed_s": elapsed,
        "seed": args.seed,
        # Pre-registered before the sealed test split is opened.
        "hypotheses": LM_HYPOTHESES,
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

    # Append this run to the shared experiment manifest (append-only JSONL
    # ledger): the compact index that makes every run traceable — code
    # revision, hyperparameters (full + recipe sha256), seed(s), data
    # fingerprints, artifact paths and final results, plus a training-curve
    # digest. Recording is automatic; nothing is tracked by hand.
    consistency_path = None
    if not args.skip_consistency_check and arch == 'hybrid':
        # Probing the val-selected weights (the model currently in memory
        # for the test evaluation), so the audit matches the checkpoint
        # the test number was produced with. The stateless softmax
        # baseline has no recurrent state to reconcile, so the probe is
        # not applicable (recorded as null).
        report = run_decode_consistency_probe(
            model, device, seq_len, args.output_dir, seed=args.seed)
        _print_consistency_report(report, title="best_val_lm.pt")
        consistency_path = os.path.join(args.output_dir,
                                        "decode_consistency.json")
    elif arch != 'hybrid' and not args.skip_consistency_check:
        print("[consistency] skipped: stateless softmax baseline has no "
              "prefill/decode state to reconcile")
    manifest_entry = make_run_manifest_entry(
        experiment='train_lm_autodl',
        seeds=int(args.seed),
        provenance=provenance,
        hyperparameters=hyperparams,
        data={
            'source': data_source,
            'tokenizer': type(tokenizer).__name__,
            'vocab_size': vocab_size,
            'cleaning': cleaning['stats'],
            'split_protocol': data_record['split_protocol'],
            'split_audit': split_report,
            'near_dup_policy': data_record['near_dup_policy'],
            'near_duplicate_audit': data_record['near_duplicate_audit'],
            'splits': {
                split: {
                    'n': rec['n'],
                    'content_fingerprint_sha256':
                        rec['content_fingerprint_sha256'],
                }
                for split, rec in data_record['splits'].items()
            },
        },
        artifacts={
            'run_record': record_path,
            'checkpoint_best_val': os.path.join(args.output_dir,
                                                "best_val_lm.pt"),
            'checkpoint_final': os.path.join(args.output_dir, "final_lm.pt"),
            'decode_consistency': consistency_path,
        },
        results={
            'optimizer_step': int(optimizer_step),
            'completed': bool(optimizer_step >= max_steps),
            'stopped_early': bool(stopped_early),
            'selected_step': best_val['step'],
            'val_best_loss': best_val['loss'],
            'final_train_loss': (curve_losses[-1] if curve_losses
                                 else None),
            'test_loss': test_report['loss'],
            'test_perplexity': test_report['perplexity'],
            'curve_diagnosis': curve_diagnosis['verdict'],
            'training_curve_sha256': hashlib.sha256(
                json.dumps([curve_steps, curve_losses],
                           separators=(',', ':')).encode()).hexdigest(),
        },
    )
    manifest_path = append_run_manifest(manifest_entry, args.manifest)
    print(f"[manifest] appended run to {manifest_path}")

if __name__ == "__main__":
    main()
