"""Deterministic near-duplicate detection for text-corpus leakage audits.

Exact-string / normalized-whitespace deduplication (implemented in
``train_lm_autodl.clean_lm_texts``) cannot see a document that was stored
again with a few words changed, a truncated sentence, or a different
numbering template.  That kind of near duplicate is still a leak when the
two copies land on different sides of a train/eval split: the model can
memorise one copy and score well on its near-twin, so the held-out metric
is no longer an out-of-sample estimate.

This module implements a small, dependency-light detector:

* the unit of identity is the **truncated model-input token window**
  (``seq_len + 1`` token ids), i.e. the same sequence the LM actually
  trains on / is evaluated on;
* documents are compared by the Jaccard similarity of their sets of
  contiguous token ``shingle_size``-grams;
* a MinHash sketch finds candidate pairs quickly and an exact second pass
  over the 64-bit shingle hashes verifies the reported similarity, so the
  LSH candidates are a recall aid rather than the metric;
* every hash, permutation and tie-break is derived from an explicit
  integer seed with no NumPy / Python global-RNG state.  Repeated calls on
  the same input therefore return identical pairs and cannot perturb the
  training / evaluation RNG streams.

The detector is intentionally conservative about what it *claims*:

* ``threshold`` is a pre-specified similarity, not tuned on the observed
  pairs; the reported ``jaccard`` is the exact shingle-set Jaccard;
* a candidate cap bounds the worst-case work on a heavily templated
  corpus, and the report says when that cap was hit
  (``candidates_truncated`` means the pair list is a lower bound);
* the exact verification compares 64-bit shingle hashes, so two distinct
  shingles could in principle collide (probability ~2**-64 per pair of
  shingles; not a correctness issue at corpus scale).

The module exposes four building blocks:

``minhash_signature``       -- the MinHash sketch for one token sequence.
``near_duplicate_pairs``     -- candidate retrieval + exact verification.
``near_duplicate_clusters``  -- connected components of the reported pairs,
                               used to drop all but one representative.
``summarize_cross_split``    -- split-aware counts/examples for the audit.
"""
from __future__ import annotations

import math
import numbers
from typing import Iterable, Sequence

import numpy as np

# Pre-specified defaults.  A 5-gram shingle is long enough to be specific
# to a phrase (not just a shared word) and short enough that a one-word
# edit still leaves most shingles intact.  0.8 Jaccard means "four fifths
# of the token 5-grams are shared"; this is the standard near-duplicate
# range and is fixed before looking at any corpus.
DEFAULT_NEAR_DUP_THRESHOLD = 0.8
DEFAULT_NEAR_DUP_SHINGLE_SIZE = 5
DEFAULT_NEAR_DUP_NUM_PERM = 64
DEFAULT_NEAR_DUP_BAND_ROWS = 4
DEFAULT_NEAR_DUP_MAX_CANDIDATES = 200_000

_MASK64 = 0xFFFFFFFFFFFFFFFF
_U64 = np.uint64

# SplitMix64 / Fibonacci constants.  All arithmetic below is explicitly
# uint64, so overflow is the intended wraparound of the mixing function.
_C1 = _U64(0x9E3779B97F4A7C15)
_C2 = _U64(0xBF58476D1CE4E5B9)
_C3 = _U64(0x94D049BB133111EB)
_HASH_BASE = _U64(0x9E3779B185EBCA87)
_PERM_A_SALT = _U64(0xD6E8FEB86659FD93)
_PERM_B_SALT = _U64(0xA24BAED4963EE407)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser (uint64 wraparound on purpose)."""
    x = np.asarray(x, dtype=_U64) + _C1
    x = (x ^ (x >> _U64(30))) * _C2
    x = (x ^ (x >> _U64(27))) * _C3
    return x ^ (x >> _U64(31))


def _require_int(name: str, value, *, minimum: int) -> int:
    """Exact integer geometry (bool and floats are not dimensions).

    ``int(5.9) == 5`` / ``int(True) == 1`` would silently run a different
    detector than the caller asked for, so reject them up front.  NumPy
    integer scalars are accepted (``numbers.Integral``), mirroring the
    token-id representation used by the caller.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an exact int, got {value!r}')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    return value


def _check_parameters(*, threshold: float, shingle_size: int, num_perm: int,
                      band_rows: int,
                      max_candidate_pairs: int) -> None:
    if (isinstance(threshold, bool)
            or not isinstance(threshold, numbers.Real)
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) <= 1.0):
        raise ValueError(
            f'near-duplicate threshold must be a finite number in (0, 1], '
            f'got {threshold!r}')
    shingle_size = _require_int('shingle_size', shingle_size, minimum=1)
    num_perm = _require_int('num_perm', num_perm, minimum=2)
    band_rows = _require_int('band_rows', band_rows, minimum=1)
    _require_int('max_candidate_pairs', max_candidate_pairs, minimum=1)
    if num_perm % band_rows != 0:
        raise ValueError(
            f'num_perm={num_perm} must be divisible by band_rows={band_rows}')
    if band_rows > num_perm:
        raise ValueError(
            f'band_rows={band_rows} must be <= num_perm={num_perm}')


def _as_ids(token_ids: Sequence[int] | None) -> np.ndarray:
    """Tokens as a uint64 array; empty for ``None`` / empty sequences."""
    if token_ids is None:
        return np.empty(0, dtype=_U64)
    if isinstance(token_ids, np.ndarray):
        return np.asarray(token_ids, dtype=_U64).reshape(-1)
    return np.asarray(list(token_ids), dtype=_U64).reshape(-1)


def _effective_shingle_size(n_tokens: int, shingle_size: int) -> int:
    """Use the whole short sequence as one shingle (never an empty shingle)."""
    n = int(n_tokens)
    if n <= 0:
        return 0
    return min(int(shingle_size), n)


def _shingle_hashes(token_ids: Sequence[int] | None, *,
                    shingle_size: int, seed: int) -> np.ndarray:
    """Sorted unique 64-bit hashes of the token n-gram shingles.

    The rolling polynomial hash is computed in uint64 arithmetic (wraps by
    design), then finalised with SplitMix64 so the bits are well mixed;
    sorting + ``np.unique`` gives the set representation used both by the
    MinHash minima and by the exact Jaccard verification.
    """
    ids = _as_ids(token_ids)
    n = int(ids.size)
    s = _effective_shingle_size(n, shingle_size)
    if s == 0:
        return np.empty(0, dtype=_U64)
    m = n - s + 1
    h = np.zeros(m, dtype=_U64)
    for j in range(s):
        # The position-dependent term prevents the same token id at
        # different shingle offsets from hashing identically.
        h = h * _HASH_BASE + ids[j:j + m] + _U64(j + 1)
    h = _splitmix64(h ^ _U64(int(seed) & _MASK64))
    return np.unique(h)


def minhash_signature(token_ids: Sequence[int] | None, *,
                      num_perm: int = DEFAULT_NEAR_DUP_NUM_PERM,
                      shingle_size: int = DEFAULT_NEAR_DUP_SHINGLE_SIZE,
                      seed: int = 0) -> np.ndarray | None:
    """MinHash sketch of one token sequence, or ``None`` for an empty one.

    Each of the ``num_perm`` rows is the minimum of an affine permutation
    ``a_i * h + b_i (mod 2**64)`` over the shingle hashes.  ``a_i`` is
    forced odd, so it is a bijection on the uint64 value space (a
    permutation of the hash universe), which is the MinHash requirement;
    the constants are derived deterministically from ``seed`` and ``i``.
    """
    num_perm = _require_int('num_perm', num_perm, minimum=2)
    shingle_size = _require_int('shingle_size', shingle_size, minimum=1)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an exact int, got {seed!r}')
    hashes = _shingle_hashes(token_ids, shingle_size=shingle_size, seed=seed)
    if hashes.size == 0:
        return None
    out = np.empty(int(num_perm), dtype=_U64)
    seed_u = _U64(int(seed) & _MASK64)
    # A 16-row chunk keeps the temporary ``(perm, shingle)`` matrix small
    # even for a very long document, while still doing the heavy min()
    # reduction inside NumPy.
    chunk = 16
    for start in range(0, int(num_perm), chunk):
        stop = min(start + chunk, int(num_perm))
        idx = np.arange(start, stop, dtype=_U64)
        a = _splitmix64(idx + seed_u + _PERM_A_SALT) | _U64(1)
        b = _splitmix64(idx + seed_u + _PERM_B_SALT)
        vals = a[:, None] * hashes[None, :] + b[:, None]
        out[start:stop] = vals.min(axis=1)
    return out


def near_duplicate_pairs(
        token_ids: Iterable[Sequence[int] | None], *,
        threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        shingle_size: int = DEFAULT_NEAR_DUP_SHINGLE_SIZE,
        num_perm: int = DEFAULT_NEAR_DUP_NUM_PERM,
        band_rows: int = DEFAULT_NEAR_DUP_BAND_ROWS,
        seed: int = 0,
        max_candidate_pairs: int = DEFAULT_NEAR_DUP_MAX_CANDIDATES,
) -> dict:
    """Find pairs of token sequences with shingle Jaccard >= ``threshold``.

    Parameters mirror :func:`minhash_signature`; ``max_candidate_pairs``
    bounds the LSH verification work on a corpus where a shared template
    puts many documents into the same band bucket.

    Returns a JSON-serializable report::

        {'threshold': 0.8,
         'shingle_size': 5,
         'num_perm': 64,
         'band_rows': 4,
         'n_documents': 120,
         'n_nonempty_documents': 118,
         'n_band_buckets': 64,
         'n_candidate_pairs': 7,
         'candidates_truncated': False,
         'n_pairs': 2,
         'pairs': [[3, 41, 0.94], ...],   # sorted by (i, j)
         'exact_verification': True,
         ...}

    ``pairs`` are index pairs into the input sequence, sorted
    deterministically.  Only pairs at or above the threshold are returned;
    ``n_candidate_pairs`` counts the LSH candidates *before* exact
    verification so the report distinguishes "no near duplicates" from
    "the candidate cap hid them".
    """
    _check_parameters(
        threshold=threshold, shingle_size=shingle_size, num_perm=num_perm,
        band_rows=band_rows, max_candidate_pairs=max_candidate_pairs)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f'seed must be an exact int, got {seed!r}')
    token_ids = list(token_ids)
    n_docs = len(token_ids)
    signatures: list[np.ndarray | None] = [
        minhash_signature(ids, num_perm=num_perm,
                          shingle_size=shingle_size, seed=seed)
        for ids in token_ids
    ]
    n_nonempty = sum(s is not None for s in signatures)

    # LSH: split the signature into independent bands of ``band_rows``
    # rows.  Two docs with Jaccard J collide in one band with probability
    # J**band_rows, so with 16 bands of 4 rows the candidate probability
    # at J=0.8 is 1 - (1 - 0.8**4)**16 ~= 0.9997.
    n_bands = int(num_perm) // int(band_rows)
    buckets: dict[tuple, list[int]] = {}
    for doc_idx, sig in enumerate(signatures):
        if sig is None:
            continue
        for band in range(n_bands):
            lo = band * int(band_rows)
            key = (band,) + tuple(int(v) for v in sig[lo:lo + int(band_rows)])
            buckets.setdefault(key, []).append(doc_idx)

    candidates: set[tuple[int, int]] = set()
    truncated = False
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for pos, i in enumerate(bucket):
            for j in bucket[pos + 1:]:
                pair = (i, j) if i < j else (j, i)
                candidates.add(pair)
                if len(candidates) >= int(max_candidate_pairs):
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break

    # Exact second pass.  Cache the sorted shingle-hash sets of the docs
    # that actually occur in a candidate pair: only those need verifying,
    # and recomputing the rolling hash here keeps the sketch pass memory
    # bounded for a large corpus.
    shingle_cache: dict[int, np.ndarray] = {}

    def _hashes(idx: int) -> np.ndarray:
        cached = shingle_cache.get(idx)
        if cached is None:
            cached = _shingle_hashes(token_ids[idx],
                                     shingle_size=shingle_size, seed=seed)
            shingle_cache[idx] = cached
        return cached

    pairs: list[list] = []
    for i, j in sorted(candidates):
        a, b = _hashes(i), _hashes(j)
        if a.size == 0 or b.size == 0:
            continue
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = a.size + b.size - inter
        jaccard = (float(inter) / float(union)) if union else 0.0
        if jaccard >= float(threshold):
            pairs.append([int(i), int(j), jaccard])
    pairs.sort(key=lambda row: (row[0], row[1]))

    return {
        'threshold': float(threshold),
        'shingle_size': int(shingle_size),
        'num_perm': int(num_perm),
        'band_rows': int(band_rows),
        'seed': int(seed),
        'n_documents': n_docs,
        'n_nonempty_documents': int(n_nonempty),
        'n_band_buckets': len(buckets),
        'n_candidate_pairs': int(len(candidates)),
        'max_candidate_pairs': int(max_candidate_pairs),
        'candidates_truncated': bool(truncated),
        'n_pairs': len(pairs),
        'pairs': pairs,
        'exact_verification': True,
        'note': ('Token n-gram Jaccard over the truncated model-input '
                 'window; LSH retrieves candidates and every reported pair '
                 'is re-scored exactly. When candidates_truncated is True '
                 'the pair list is a lower bound.'),
    }


def _pair_indices(row, where: str) -> tuple[int, int]:
    """Extract exact integer indices from a reported pair row.

    ``int(0.9) == 0`` would silently connect the wrong documents, so a
    float / bool / string index is a malformed row rather than something
    to coerce.
    """
    if isinstance(row, (str, bytes)):
        raise ValueError(f'{where}: malformed near-duplicate pair: {row!r}')
    try:
        fields = len(row)
    except TypeError:
        raise ValueError(
            f'{where}: malformed near-duplicate pair: {row!r}') from None
    if fields < 2:
        raise ValueError(f'{where}: malformed near-duplicate pair: {row!r}')
    out = []
    for pos in range(2):
        value = row[pos]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f'{where}: pair index #{pos} must be an exact int, got '
                f'{value!r} in row {row!r}')
        out.append(int(value))
    return out[0], out[1]


def near_duplicate_clusters(pairs: Iterable[Sequence], n_documents: int,
                            *, keep: str = 'first') -> list[list[int]]:
    """Connected components of the near-duplicate graph.

    ``pairs`` may be the ``pairs`` list from :func:`near_duplicate_pairs`
    (``[i, j, jaccard]`` rows) or plain ``(i, j)`` tuples.  Returns only
    components with at least two documents, each sorted ascending and the
    list sorted by its first member.

    ``keep='first'`` documents the intended policy: keep the first
    (lowest-index, i.e. corpus-order) representative of a component and
    drop the rest.  Transitive closure is deliberate: if A~B and B~C but
    A is not itself similar to C, dropping C is the conservative choice
    because B could have leaked A-like content into C.
    """
    if keep != 'first':
        raise ValueError(f"keep must be 'first', got {keep!r}")
    n = _require_int('n_documents', n_documents, minimum=0)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Smaller root wins, so the component's representative is the
        # lowest document index independent of pair ordering.
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    for row in pairs:
        i, j = _pair_indices(row, 'near_duplicate_clusters')
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f'near-duplicate pair {(i, j)} out of range for n={n}')
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [sorted(members) for members in groups.values()
                if len(members) > 1]
    clusters.sort(key=lambda members: members[0])
    return clusters


def summarize_cross_split(pairs: Iterable[Sequence],
                          split_labels: Sequence[str], *,
                          max_examples: int = 10) -> dict:
    """Count near-duplicate pairs that straddle two split labels.

    ``split_labels`` is one label per document (for example
    ``['train', ..., 'val', ..., 'test', ...]`` combined in the same order
    as the ``token_ids`` passed to :func:`near_duplicate_pairs`).
    ``max_examples`` caps the stored example rows; the counts are never
    truncated.

    The caller is responsible for the ordering: a pair ``(i, j)`` is
    compared using ``split_labels[i]`` / ``split_labels[j]``.
    """
    max_examples = _require_int('max_examples', max_examples, minimum=0)
    labels = [str(x) for x in split_labels]
    n = len(labels)
    by_pair: dict[str, int] = {}
    within = 0
    involved: set[int] = set()
    max_jaccard = None
    examples: list[dict] = []
    for row in pairs:
        i, j = _pair_indices(row, 'summarize_cross_split')
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f'near-duplicate pair {(i, j)} out of range for n={n}')
        jaccard = None
        if len(row) >= 3:
            raw = row[2]
            if raw is not None:
                if (isinstance(raw, bool)
                        or not isinstance(raw, numbers.Real)
                        or not math.isfinite(float(raw))):
                    raise ValueError(
                        'summarize_cross_split: pair jaccard must be a '
                        f'finite number or None, got {raw!r} in row {row!r}')
                jaccard = float(raw)
        li, lj = labels[i], labels[j]
        if li == lj:
            within += 1
            continue
        key = '|'.join(sorted((li, lj)))
        by_pair[key] = by_pair.get(key, 0) + 1
        involved.add(i)
        involved.add(j)
        if jaccard is not None:
            max_jaccard = (jaccard if max_jaccard is None
                           else max(max_jaccard, jaccard))
        if len(examples) < max_examples:
            examples.append({
                'i': i, 'j': j,
                'splits': [li, lj],
                'jaccard': jaccard,
            })
    total = sum(by_pair.values())
    return {
        'n_cross_split_pairs': int(total),
        'n_within_split_pairs': int(within),
        'n_documents_involved': len(involved),
        'by_split_pair': dict(sorted(by_pair.items())),
        'max_cross_split_jaccard': max_jaccard,
        'examples': examples,
        'examples_truncated': bool(total > len(examples)),
        'note': ('A cross-split near-duplicate pair means one document\'s '
                 'near-twin is on the other side of the train/eval '
                 'boundary. Structural audit only: no model metric is '
                 'computed here.'),
    }
