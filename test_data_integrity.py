"""Regression tests for near-duplicate leakage detection and data provenance.

These target the data-processing contract rather than a model: two texts that
are the same story with an edit must be found before/around the split, the
detector must be deterministic and RNG-neutral, and the corpus fingerprint
must not be ambiguous at text boundaries.
"""
from __future__ import annotations

import json
import random
from decimal import Decimal
from fractions import Fraction

import numpy as np

from near_duplicates import (
    cluster_labels,
    near_duplicate_cluster_labels,
    near_duplicate_clusters,
    near_duplicate_pairs,
    summarize_cross_split,
)
from train_lm_autodl import (
    _FallbackCharTokenizer,
    _is_nonfinite_document,
    _evenly_spaced_probe_indices,
    _near_duplicate_cluster_report,
    _probe_token_lengths,
    _text_length_stats,
    _window_digest,
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

    # Coerced types are rejected instead of silently truncating:
    # int(5.9) would run a different detector than requested, and
    # int(True) would accept an accidental flag.
    for kwargs in ({'shingle_size': 5.9}, {'num_perm': 64.5},
                   {'band_rows': True}, {'max_candidate_pairs': 1.5},
                   {'seed': 1.5}, {'threshold': True},
                   {'threshold': float('nan')}):
        try:
            near_duplicate_pairs([[1, 2, 3]], **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(
                f'coerced near-dup parameter accepted: {kwargs}')

    # A non-integer document index must not be coerced to the wrong row.
    for bad_pairs in ([[0.5, 1]], [[True, 1]], [['0', 1]],
                      [{0: 1, 1: 0}], [{0, 1}]):
        try:
            near_duplicate_clusters(bad_pairs, 2)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'malformed pair accepted: {bad_pairs}')
    try:
        summarize_cross_split([[0, 1, float('nan')]],
                              ['train', 'val'])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('non-finite pair jaccard accepted')


def test_numpy_infinite_scalars_are_dropped_not_stringified():
    # A dataframe/JSON feed commonly carries NumPy scalars.  ``inf == inf``,
    # so the old ``x != x`` fallback only caught NaN, and ``np.float32(inf)``
    # survived as the literal text "inf".
    for value in (np.float32('inf'), np.float32('-inf'),
                  np.float64('inf'), np.float16('nan')):
        assert _is_nonfinite_document(value) is True
    assert _is_nonfinite_document(np.float32(1.5)) is False
    assert _is_nonfinite_document(np.int64(3)) is False


def test_nonfinite_classification_handles_huge_ints_and_decimals():
    # A finite integer too large for float() is data, not a missing value.
    assert _is_nonfinite_document(10 ** 400) is False
    cleaned = clean_lm_texts([10 ** 400, 'a real document'],
                             _FallbackCharTokenizer(), seq_len=1024)
    assert len(cleaned['texts']) == 2
    assert cleaned['stats']['n_nonfinite'] == 0

    # Decimal infinities / signalling NaNs are not caught by ``x != x``.
    for value in (Decimal('Infinity'), Decimal('-Infinity'),
                  Decimal('NaN'), Decimal('sNaN')):
        assert _is_nonfinite_document(value) is True
    assert _is_nonfinite_document(Decimal('1.5')) is False
    for value in (complex(0, float('inf')), complex(float('nan'), 0)):
        assert _is_nonfinite_document(value) is True
    assert _is_nonfinite_document(complex(1, 2)) is False
    # Finite extended-precision values must not be dropped just because
    # converting them to a Python float overflows / rounds to inf.
    assert _is_nonfinite_document(np.longdouble('1e4000')) is False
    assert _is_nonfinite_document(Fraction(10 ** 400, 1)) is False
    assert _is_nonfinite_document(np.longdouble('inf')) is True


def test_near_duplicate_token_ids_are_not_silently_coerced():
    for bad in ([1.0, 2.0], [True, 2], [-1, 2], [2 ** 64, 1],
                {1: 'x'}, {1, 2}, bytearray(b'\x01\x02'),
                memoryview(b'\x01\x02'),
                'not ids', np.array([1.0, 2.0]), np.array([True, False])):
        try:
            near_duplicate_pairs([[1, 2, 3, 4], bad], threshold=1.0)
        except ValueError:
            pass
        else:  # pragma: no cover - every malformed id must be rejected
            raise AssertionError(f'malformed token ids accepted: {bad!r}')

    # NumPy / torch integer scalars are exact ids and remain accepted.
    ok = near_duplicate_pairs(
        [[1, 2, 3, 4], [np.int64(1), np.uint64(2), np.int64(3), 4]],
        threshold=1.0)
    assert ok['pairs'] == [[0, 1, 1.0]]

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


def test_near_duplicate_candidate_cap_exact_boundary_is_complete():
    # Exactly ``max_candidate_pairs`` distinct candidates is a complete
    # audit: nothing was dropped. Only a pair that cannot be retained
    # makes the report a lower bound.
    two = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    exact_one = near_duplicate_pairs(two, max_candidate_pairs=1)
    assert exact_one['n_candidate_pairs'] == 1
    assert exact_one['candidates_truncated'] is False

    three = [[1, 2, 3, 4, 5, 6]] * 3
    exact_three = near_duplicate_pairs(three, max_candidate_pairs=3)
    assert exact_three['n_candidate_pairs'] == 3
    assert exact_three['candidates_truncated'] is False

    over = near_duplicate_pairs(three, max_candidate_pairs=2)
    assert over['n_candidate_pairs'] == 2
    assert over['candidates_truncated'] is True


class _BadIdsTokenizer:
    """Tokenizer whose encode() returns a malformed id sequence."""
    pad_token_id = 0
    unk_token_id = 0

    def __init__(self, result):
        self.result = result

    def encode(self, text, truncation=False, max_length=None):
        return self.result


def test_malformed_token_ids_are_reported_not_crashed_or_coerced():
    valid = 'a real document ' * 5
    for bad in (None, 1.5, 'not ids', [1.0, 2.0], [-1, 2], [True, 2],
                object()):
        tok = _BadIdsTokenizer(bad)
        cleaned = clean_lm_texts([valid], tok, seq_len=1024)
        assert cleaned['texts'] == []
        assert cleaned['stats']['n_tokenize_errors'] == 1
        assert _text_length_stats([valid], tok, 1024)['n_probe'] == 0
        assert _probe_token_lengths([valid], tok, 1024) == []
        assert _window_digest(valid, tok, 1024) is None
        audit = near_duplicate_leakage_audit([valid], ['b'], ['c'], tok,
                                             seq_len=1024)
        assert audit['n_tokenize_errors'] == 3

    # Integer-like scalars (NumPy / torch) are real token ids; the
    # uint64 cast in the near-duplicate path must not see floats.
    import numpy as np
    tok = _BadIdsTokenizer(np.array([1, 2, 3], dtype=np.int64))
    cleaned = clean_lm_texts([valid], tok, seq_len=1024)
    assert cleaned['texts'] == [valid]
    assert cleaned['stats']['n_tokenize_errors'] == 0
    assert _probe_token_lengths([valid], tok, 1024) == [3]


def test_near_duplicate_parameter_and_jaccard_overflow_is_controlled():
    for threshold in (10 ** 400, Fraction(10 ** 400, 1), Fraction(1, 10 ** 400)):
        try:
            near_duplicate_pairs([[1, 2, 3]], threshold=threshold)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'huge threshold accepted: {threshold!r}')
    for jaccard in (10 ** 400, Fraction(10 ** 400, 1), -0.1, 1.1):
        try:
            summarize_cross_split([[0, 1, jaccard]], ['train', 'val'])
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid jaccard accepted: {jaccard!r}')
    # Tiny positive fractions are valid thresholds (not coerced away).
    report = near_duplicate_pairs([[1, 2, 3, 4]], threshold=Fraction(1, 2))
    assert report['threshold'] == 0.5


def test_clean_near_dup_parameters_are_validated_even_on_tiny_corpus():
    tok = _FallbackCharTokenizer()
    for kwargs in ({'near_dup_threshold': 0.0},
                   {'near_dup_threshold': float('nan')},
                   {'near_dup_shingle_size': 0},
                   {'near_dup_num_perm': 3},
                   {'near_dup_max_candidates': 0}):
        for texts in ([], ['one document']):
            try:
                clean_lm_texts(texts, tok, seq_len=1024, dedup_near=True,
                               **kwargs)
            except ValueError:
                pass
            else:  # pragma: no cover - every invalid value must be rejected
                raise AssertionError(
                    f'invalid near-dup parameter accepted on {texts!r}: '
                    f'{kwargs}')


def test_near_duplicate_audit_accepts_iterators_and_validates_window_args():
    tok = _FallbackCharTokenizer()
    report = near_duplicate_leakage_audit(
        (text for text in ['a' * 20]),
        (text for text in ['b' * 20]), [], tok, seq_len=64)
    assert report['splits'] == {'train': 1, 'val': 1, 'test': 0}
    assert report['n_tokenize_errors'] == 0
    json.dumps(report, allow_nan=False)

    for bad_seq_len in (0, -1, 1.5, True):
        try:
            near_duplicate_leakage_audit(['a'], ['b'], [], tok, bad_seq_len)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(
                f'invalid audit seq_len accepted: {bad_seq_len!r}')
    try:
        near_duplicate_leakage_audit(['a'], ['b'], [], object(), 64)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('tokenizer without encode() was accepted')


def test_data_probe_geometry_requires_exact_integers():
    tok = _FallbackCharTokenizer()
    for bad_seq_len in (0, 1.5, True):
        try:
            clean_lm_texts(['a real document'], tok, bad_seq_len)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(
                f'invalid cleaning seq_len accepted: {bad_seq_len!r}')
    for bad_min_tokens in (2.5, True):
        try:
            clean_lm_texts(['a real document'], tok, 1024,
                           min_tokens=bad_min_tokens)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(
                f'invalid min_tokens accepted: {bad_min_tokens!r}')
    for kwargs in ({'n_probe': 1.5}, {'n_probe': -1},
                   {'max_chars_per_text': 0},
                   {'max_chars_per_text': True}):
        try:
            _texts_fingerprint(['a', 'b'], **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid fingerprint geometry accepted: {kwargs}')


def test_token_ids_beyond_uint64_are_rejected_before_the_hash_cast():
    class _HugeIdTokenizer:
        pad_token_id = 0
        unk_token_id = 0

        def encode(self, text, truncation=False, max_length=None):
            return [2 ** 64, 1]

    tok = _HugeIdTokenizer()
    cleaned = clean_lm_texts(['a long document', 'another long document'],
                             tok, seq_len=64, dedup_near=True)
    assert cleaned['texts'] == []
    assert cleaned['stats']['n_tokenize_errors'] == 2
    audit = near_duplicate_leakage_audit(['a'], ['b'], [], tok, 64)
    assert audit['n_tokenize_errors'] == 2


def test_single_probe_uses_the_corpus_tail_and_fingerprint_needs_content():
    assert _evenly_spaced_probe_indices(10, 1) == [9]
    # A one-text fingerprint must still see a change in the last document.
    assert _texts_fingerprint(['a', 'b'], n_probe=1) != _texts_fingerprint(
        ['a', 'c'], n_probe=1)
    for bad_probe in (0, -1):
        try:
            _texts_fingerprint(['a', 'b'], n_probe=bad_probe)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'content-free probe accepted: {bad_probe}')


def test_length_and_fingerprint_probes_accept_text_iterators():
    tok = _FallbackCharTokenizer()
    texts = ['abc', 'defgh']
    stats = _text_length_stats((t for t in texts), tok, 64)
    assert stats['n'] == 2 and stats['n_probe'] == 2
    assert _probe_token_lengths((t for t in texts), tok, 64) == [3, 5]
    assert _texts_fingerprint((t for t in texts)) == _texts_fingerprint(texts)


def test_unigram_counts_must_be_exact_non_negative_integers():
    from collections import Counter
    from train_lm_autodl import _unigram_shift_stats
    for fit, evl in (([Counter({1: 1.5})], [Counter({2: 2.5})]),
                     ([Counter({1: float('nan')})], [Counter({2: 1})]),
                     ([Counter({1: -1})], [Counter({2: 1})]),
                     ([Counter({1: True})], [Counter({2: 1})])):
        try:
            _unigram_shift_stats(fit, evl, max_examples=2)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'malformed counters accepted: {fit!r}')


def test_cluster_labels_align_to_documents_and_mark_singletons():
    pairs = [[0, 2, 0.9], [2, 4, 0.85]]
    assert cluster_labels(pairs, 6) == [0, 1, 0, 3, 0, 5]
    assert cluster_labels([], 3) == [0, 1, 2]
    for bad_n in (-1, True, 1.5):
        try:
            cluster_labels([], bad_n)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid n_documents accepted: {bad_n!r}')
    try:
        cluster_labels([[0, 9, 0.9]], 3)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('out-of-range pair accepted')


def test_near_duplicate_cluster_labels_report_is_json_safe():
    base = list(range(100))
    near = base.copy()
    near[50] = 999
    other = list(range(1000, 1100))
    report = near_duplicate_cluster_labels(
        [base, near, other], threshold=0.8, shingle_size=5)
    assert report['cluster_ids'] == [0, 0, 2]
    assert report['n_documents'] == 3
    assert report['n_clusters'] == 2
    assert report['n_redundant_documents'] == 1
    assert report['cluster_sizes']['max'] == 2
    assert report['candidates_truncated'] is False
    assert report['labels_are_document_indices'] is True
    json.dumps(report, allow_nan=False)


def test_heldout_cluster_report_aligns_texts_to_bootstrap_labels():
    tok = _FallbackCharTokenizer()
    base = ' '.join(f'w{i}' for i in range(120))
    near = base.replace('w20', 'w999')
    other = ' '.join(f'z{i}' for i in range(120))
    report = _near_duplicate_cluster_report(
        [base, near, other], tok, 1024, threshold=0.8,
        shingle_size=5, num_perm=64, max_candidate_pairs=1000)
    assert report['cluster_ids'][0] == report['cluster_ids'][1]
    assert report['cluster_ids'][2] == 2
    assert report['n_redundant_documents'] == 1
    assert report['n_tokenize_errors'] == 0
    json.dumps(report, allow_nan=False)
