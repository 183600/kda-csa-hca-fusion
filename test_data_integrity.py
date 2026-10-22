"""Regression tests for near-duplicate leakage detection and data provenance.

These target the data-processing contract rather than a model: two texts that
are the same story with an edit must be found before/around the split, the
detector must be deterministic and RNG-neutral, and the corpus fingerprint
must not be ambiguous at text boundaries.
"""
from __future__ import annotations

import json
import random

import numpy as np

from near_duplicates import (
    near_duplicate_clusters,
    near_duplicate_pairs,
    summarize_cross_split,
)
from train_lm_autodl import (
    _FallbackCharTokenizer,
    _is_nonfinite_document,
    _texts_fingerprint,
    clean_lm_texts,
    near_duplicate_leakage_audit,
    split_clean_texts,
)


def _random_doc(rng, n=300, vocab=50):
    return [rng.randrange(vocab) for _ in range(n)]


def test_near_duplicate_pairs_detects_single_edit_and_skips_unrelated():
    rng = random.Random(7)
    base = _random_doc(rng)
    near = list(base)
    near[150] = (near[150] + 1) % 50
    unrelated = _random_doc(rng)

    report = near_duplicate_pairs([base, near, unrelated], threshold=0.8)
    assert report['exact_verification'] is True
    found = {(int(i), int(j)): float(jac)
             for i, j, jac in report['pairs']}
    assert (0, 1) in found
    assert not any(2 in pair for pair in found)
    assert found[(0, 1)] >= 0.8
    # The exact second pass, not the LSH candidate, produces the number.
    assert found[(0, 1)] < 1.0
    json.dumps(report, allow_nan=False)


def test_near_duplicate_pairs_respects_threshold_and_empty_docs():
    rng = random.Random(11)
    base = _random_doc(rng)
    # Half of the shingles are shared, far below the 0.8 default.
    half = base[:150] + _random_doc(rng)
    report = near_duplicate_pairs([base, half], threshold=0.8)
    assert report['n_pairs'] == 0

    short = near_duplicate_pairs(
        [[], None, [1, 2, 3], [1, 2, 3]], threshold=1.0)
    assert short['n_pairs'] == 1
    assert short['pairs'][0][:2] == [2, 3]
    assert short['pairs'][0][2] == 1.0


def test_near_duplicate_pairs_is_deterministic_and_rng_neutral():
    rng = random.Random(3)
    docs = [_random_doc(rng) for _ in range(6)]
    docs.append(list(docs[0]))
    docs.append(list(docs[1]))
    py_state = random.getstate()
    np_state = np.random.get_state()
    first = near_duplicate_pairs(docs, threshold=0.8)
    second = near_duplicate_pairs(docs, threshold=0.8)
    assert first == second
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]


def test_near_duplicate_clusters_keep_first_representative():
    pairs = [[0, 1, 0.9], [2, 3, 0.95], [1, 2, 0.85]]
    clusters = near_duplicate_clusters(pairs, 5)
    assert clusters == [[0, 1, 2, 3]]
    # The representative policy documented by the module: first member.
    assert clusters[0][0] == 0


def test_clean_lm_texts_drops_near_duplicate_before_split():
    tok = _FallbackCharTokenizer()
    base = ' '.join(f'w{i}' for i in range(120))
    near = base.replace('w20', 'w999')
    other = ' '.join(f'z{i}' for i in range(120))

    default = clean_lm_texts([base, near, other], tok, seq_len=1024)
    assert default['texts'] == [base, near, other]
    assert default['stats']['n_near_duplicates'] == 0
    assert default['stats']['dedup_near'] is False

    cleaned = clean_lm_texts([base, near, other], tok, seq_len=1024,
                             dedup_near=True)
    assert cleaned['texts'] == [base, other]
    stats = cleaned['stats']
    assert stats['dedup_near'] is True
    assert stats['n_near_duplicates'] == 1
    assert stats['n_near_duplicate_pairs'] >= 1
    assert stats['n_near_duplicate_clusters'] == 1
    assert stats['largest_near_duplicate_cluster'] == 2
    # The surviving corpus has no exact/normalized/window duplicates; the
    # near pass only removes the edited twin.
    assert stats['n_duplicates'] == 0
    assert stats['n_window_duplicates'] == 0


def test_split_clean_texts_near_duplicate_audit_flags_cross_split_pair():
    tok = _FallbackCharTokenizer()
    base = ' '.join(f'w{i}' for i in range(120))
    near = base.replace('w20', 'w999')
    texts = [base] + [' '.join(f'q{i}_{j}' for i in range(80))
                      for j in range(8)] + [near]
    split, report = split_clean_texts(
        texts, val_frac=0.1, test_frac=0.1, seed=0, chronological=True,
        tokenizer=tok, seq_len=1024, near_dup_audit=True)
    assert split['train'][0] == 0
    assert split['test'][-1] == len(texts) - 1
    near_report = report['near_duplicate_overlaps']
    assert near_report['n_cross_split_pairs'] == 1
    assert near_report['by_split_pair'] == {'test|train': 1}
    assert near_report['detector']['threshold'] == 0.8
    assert near_report['examples'][0]['splits'] == ['train', 'test']
    json.dumps(report, allow_nan=False)


def test_split_clean_texts_strict_near_duplicate_gate_refuses_leak():
    tok = _FallbackCharTokenizer()
    base = ' '.join(f'w{i}' for i in range(120))
    near = base.replace('w20', 'w999')
    texts = [base] + [' '.join(f'q{i}_{j}' for i in range(80))
                      for j in range(8)] + [near]
    try:
        split_clean_texts(
            texts, val_frac=0.1, test_frac=0.1, seed=0, chronological=True,
            tokenizer=tok, seq_len=1024, strict_near_dup=True)
    except RuntimeError as exc:
        assert 'near-duplicate' in str(exc)
    else:  # pragma: no cover - a leak must never pass the strict gate
        raise AssertionError('strict near-duplicate gate did not raise')


def test_near_duplicate_leakage_audit_reports_within_and_cross_split():
    tok = _FallbackCharTokenizer()
    base = ' '.join(f'w{i}' for i in range(120))
    near = base.replace('w20', 'w999')
    # Both copies in train: within-split redundancy, not a cross-split leak.
    report = near_duplicate_leakage_audit(
        [base, near], ['unrelated ' * 30], ['other ' * 30],
        tok, seq_len=1024)
    assert report['n_cross_split_pairs'] == 0
    assert report['n_within_split_pairs'] == 1
    assert report['flag'] is False


def test_cross_split_summary_counts_only_boundary_pairs():
    report = summarize_cross_split(
        [[0, 1, 0.9], [1, 2, 0.85], [2, 3, 0.8]],
        ['train', 'train', 'val', 'test'])
    assert report['n_cross_split_pairs'] == 2
    assert report['n_within_split_pairs'] == 1
    assert report['by_split_pair'] == {'train|val': 1, 'test|val': 1}
    assert report['n_documents_involved'] == 3


def test_near_duplicate_candidate_cap_is_reported_not_silent():
    rng = random.Random(5)
    base = _random_doc(rng, n=80)
    docs = [list(base) for _ in range(20)]
    for i, doc in enumerate(docs):
        doc[i] = (doc[i] + 1) % 50
    report = near_duplicate_pairs(docs, threshold=0.8,
                                  max_candidate_pairs=7)
    assert report['candidates_truncated'] is True
    assert report['n_candidate_pairs'] == 7
    assert report['max_candidate_pairs'] == 7


def test_near_duplicate_parameter_validation():
    for kwargs in ({'threshold': 0.0}, {'threshold': 1.5},
                   {'shingle_size': 0}, {'num_perm': 3},
                   {'band_rows': 0}, {'max_candidate_pairs': 0}):
        try:
            near_duplicate_pairs([[1, 2, 3]], **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - every invalid value must be rejected
            raise AssertionError(f'invalid near-dup parameter accepted: {kwargs}')


def test_numpy_infinite_scalars_are_dropped_not_stringified():
    # A dataframe/JSON feed commonly carries NumPy scalars.  ``inf == inf``,
    # so the old ``x != x`` fallback only caught NaN, and ``np.float32(inf)``
    # survived as the literal text "inf".
    for value in (np.float32('inf'), np.float32('-inf'),
                  np.float64('inf'), np.float16('nan')):
        assert _is_nonfinite_document(value) is True
    assert _is_nonfinite_document(np.float32(1.5)) is False
    assert _is_nonfinite_document(np.int64(3)) is False

    tok = _FallbackCharTokenizer()
    cleaned = clean_lm_texts(
        [np.float32('inf'), np.float32('nan'), 'a real document'],
        tok, seq_len=1024)
    assert cleaned['texts'] == ['a real document']
    assert cleaned['stats']['n_nonfinite'] == 2


def test_texts_fingerprint_is_unambiguous_at_text_boundaries():
    # Both corpora concatenate to the same byte stream "abc".  A fingerprint
    # without per-text length framing would assign them the same data id.
    assert _texts_fingerprint(['ab', 'c']) != _texts_fingerprint(['a', 'bc'])
    assert _texts_fingerprint(['ab', 'c']) == _texts_fingerprint(['ab', 'c'])
    # Order is part of the corpus identity.
    assert _texts_fingerprint(['ab', 'c']) != _texts_fingerprint(['c', 'ab'])
