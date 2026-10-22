"""Regression tests for the held-out token-distribution shift audit.

The audit exists to catch the case where train and held-out splits have the
same *length* profile but a different vocabulary / topic / source mix, which
would make a perplexity comparison partly a domain-mismatch measurement.  The
tests pin the two properties that make the diagnostic trustworthy:

* the reference distribution is fitted on TRAIN only and the null calibration
  is drawn from TRAIN, so no held-out or sealed-test row can move the
  decision boundary;
* the raw plug-in numbers are NOT the decision: an i.i.d. split with rare
  tokens naturally has unseen held-out tokens, so only the excess over a
  within-train document-level null is allowed to raise the flag.

Everything is deterministic and RNG-neutral, and every record must survive
``json.dumps(..., allow_nan=False)``.
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

import numpy as np

from train_lm_autodl import (
    _FallbackCharTokenizer,
    _distribution_shift,
    _length_outlier_audit,
    _quantile_type7,
    token_distribution_audit,
)


class _WordTokenizer:
    """Word-level tokenizer with an unbounded id space (rare-word tests)."""

    pad_token_id = 0
    unk_token_id = 1

    def encode(self, text, truncation=False, max_length=None):
        ids = [int(word.split('_')[1]) for word in str(text).split()]
        if truncation and max_length is not None:
            ids = ids[:int(max_length)]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(f'w_{int(i)}' for i in ids)


class _RaisingTokenizer:
    pad_token_id = 0
    unk_token_id = 1

    def encode(self, text, truncation=False, max_length=None):
        raise RuntimeError('synthetic tokenizer failure')


class _FloatIdTokenizer:
    pad_token_id = 0
    unk_token_id = 1

    def encode(self, text, truncation=False, max_length=None):
        # ``int(1.9) == 1`` would silently count the wrong token.
        return [1.0, 2.0]


class _NegativeIdTokenizer:
    pad_token_id = 0
    unk_token_id = 1

    def encode(self, text, truncation=False, max_length=None):
        return [-1, 2]


def _zipf_documents(rng, *, n_docs, n_words, vocab, exponent=0.8):
    return [
        ' '.join(
            f"w_{min(int(rng.paretovariate(exponent)), int(vocab) - 1)}"
            for _ in range(int(n_words)))
        for _ in range(int(n_docs))
    ]


def test_disjoint_vocabularies_are_flagged():
    tok = _FallbackCharTokenizer()
    train = ['a' * 50 for _ in range(20)]
    heldout = ['b' * 50 for _ in range(10)]

    report = token_distribution_audit(train, heldout, tok, 64)
    assert report['available'] is True
    observed = report['observed']
    assert observed['unseen_token_mass'] == 1.0
    assert observed['js_divergence_bits'] == 1.0
    assert report['null_reference']['unseen_token_mass'] == 0.0
    assert report['excess']['unseen_token_mass'] == 1.0
    assert report['flag'] is True
    assert 'unseen_token_mass' in report['flag_reasons']
    json.dumps(report, allow_nan=False)


def test_iid_rare_tokens_do_not_flag_above_the_null():
    # A large vocabulary makes the raw plug-in "unseen held-out tokens"
    # count strictly positive even though train and val are drawn from the
    # SAME generator.  The within-train null must absorb that finite-sample
    # baseline so the audit does not flag every split.
    tok = _WordTokenizer()
    rng = random.Random(1)
    train = _zipf_documents(rng, n_docs=120, n_words=32, vocab=20_000)
    heldout = _zipf_documents(rng, n_docs=40, n_words=32, vocab=20_000)

    report = token_distribution_audit(train, heldout, tok, 256)
    assert report['available'] is True
    assert report['observed']['unseen_token_mass'] > 0.0
    assert report['null_reference']['unseen_token_mass'] > 0.0
    assert report['null_reference']['unseen_examples'] == []
    assert report['null_reference']['examples_truncated'] is False
    assert report['excess']['unseen_token_mass'] == 0.0
    assert report['flag'] is False
    assert report['flag_reasons'] == []
    json.dumps(report, allow_nan=False)


def test_domain_mixture_shift_is_detected():
    tok = _WordTokenizer()
    rng = random.Random(2)
    train = [' '.join(f'w_{rng.randrange(0, 500)}' for _ in range(40))
             for _ in range(300)]
    # Only 10% of held-out TOKENS come from a foreign vocabulary; the
    # within-train null is ~0.5%, so the excess is the signal.
    heldout = [
        ' '.join(
            f'w_{rng.randrange(0, 500) if rng.random() > 0.1 else rng.randrange(500, 2000)}'
            for _ in range(40))
        for _ in range(60)
    ]

    report = token_distribution_audit(train, heldout, tok, 256)
    assert report['observed']['unseen_token_mass'] > 0.05
    assert report['excess']['unseen_token_mass'] > 0.05
    assert report['flag'] is True
    assert 'unseen_token_mass' in report['flag_reasons']
    # Examples are the most frequent unseen ids, deterministically ordered.
    examples = report['observed']['unseen_examples']
    assert 0 < len(examples) <= 10
    assert [row['count'] for row in examples] == sorted(
        (row['count'] for row in examples), reverse=True)
    assert all(row['decoded'] for row in examples)
    json.dumps(report, allow_nan=False)


def test_audit_is_deterministic_and_rng_neutral():
    tok = _WordTokenizer()
    rng = random.Random(3)
    train = _zipf_documents(rng, n_docs=80, n_words=24, vocab=5000)
    heldout = _zipf_documents(rng, n_docs=30, n_words=24, vocab=5000)
    py_state = random.getstate()
    np_state = np.random.get_state()

    first = token_distribution_audit(train, heldout, tok, 128)
    second = token_distribution_audit(train, heldout, tok, 128)

    assert first == second
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]


def test_tiny_train_probe_reports_no_null_instead_of_flagging():
    tok = _FallbackCharTokenizer()
    report = token_distribution_audit(
        ['a' * 20, 'b' * 20], ['c' * 20, 'd' * 20], tok, 64)
    assert report['available'] is True
    assert report['observed'] is not None
    assert report['null_reference'] is None
    assert report['excess'] is None
    assert report['flag'] is None
    assert report['flag_reasons'] == ['null_calibration_unavailable']
    json.dumps(report, allow_nan=False)


def test_empty_or_untokenizable_probes_are_reported_not_crashed():
    tok = _FallbackCharTokenizer()
    no_train = token_distribution_audit([], ['a' * 20], tok, 64)
    assert no_train['available'] is False
    assert 'reason' in no_train
    json.dumps(no_train, allow_nan=False)

    no_heldout = token_distribution_audit(['a' * 20], [], tok, 64)
    assert no_heldout['available'] is False
    json.dumps(no_heldout, allow_nan=False)

    for bad_tokenizer in (_RaisingTokenizer(), _FloatIdTokenizer(),
                          _NegativeIdTokenizer()):
        broken = token_distribution_audit(
            ['a' * 20] * 4, ['b' * 20] * 4, bad_tokenizer, 64)
        assert broken['available'] is False
        assert broken['train_probe']['n_tokenize_errors'] == 4
        assert broken['heldout_probe']['n_tokenize_errors'] == 4
        json.dumps(broken, allow_nan=False)


def test_probe_accounting_and_example_cap():
    tok = _FallbackCharTokenizer()
    train = ['abcdefghij' * 5 for _ in range(30)]
    heldout = ['abcXYZ' * 5 for _ in range(10)]
    report = token_distribution_audit(
        train, heldout, tok, 64, n_probe=7, max_examples=2)

    assert report['train_probe']['n_documents_probed'] == 7
    assert report['heldout_probe']['n_documents_probed'] == 7
    assert report['train_probe']['n_documents_scored'] == 7
    assert report['observed']['unseen_examples'][:2]
    assert len(report['observed']['unseen_examples']) <= 2
    assert report['observed']['examples_truncated'] is True
    json.dumps(report, allow_nan=False)


def test_token_audit_parameter_validation():
    tok = _FallbackCharTokenizer()
    texts = ['a' * 10, 'b' * 10]
    for kwargs in (
            {'n_probe': 0}, {'n_probe': 2.5}, {'n_probe': True},
            {'max_examples': -1}, {'max_examples': 1.5},
            {'js_threshold': -0.1}, {'js_threshold': 1.1},
            {'js_threshold': float('nan')}, {'js_threshold': True},
            {'unseen_mass_threshold': -0.1},
            {'unseen_mass_threshold': float('inf')},
            {'seed': 1.5}, {'seed': True},
    ):
        try:
            token_distribution_audit(texts, texts, tok, 64, **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - every invalid value must be rejected
            raise AssertionError(f'invalid token-audit parameter accepted: {kwargs}')

    for bad_seq_len in (0, -1, 1.5, True):
        try:
            token_distribution_audit(texts, texts, tok, bad_seq_len)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid seq_len accepted: {bad_seq_len!r}')

    try:
        token_distribution_audit(texts, texts, object(), 64)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('tokenizer without encode() was accepted')


def test_length_shift_zero_variance_does_not_hide_a_separation():
    # Both splits constant: the pooled SD is zero, so Cohen's d is
    # undefined.  Equal means are a genuine d=0; different means are the
    # strongest possible shift and must still flag.
    same = _distribution_shift({'mean_tokens': 10, 'std_tokens': 0},
                               {'mean_tokens': 10, 'std_tokens': 0})
    assert same['cohen_d'] == 0.0
    assert same['zero_pooled_variance'] is True
    assert same['flag'] is False
    json.dumps(same, allow_nan=False)

    separated = _distribution_shift({'mean_tokens': 3, 'std_tokens': 0},
                                    {'mean_tokens': 300, 'std_tokens': 0})
    assert separated['cohen_d'] is None
    assert separated['abs_cohen_d'] is None
    assert separated['mean_difference'] == 297.0
    assert separated['zero_pooled_variance'] is True
    assert separated['flag'] is True
    json.dumps(separated, allow_nan=False)


def test_length_shift_normal_and_missing_cases_remain_json_safe():
    normal = _distribution_shift({'mean_tokens': 100, 'std_tokens': 10},
                                 {'mean_tokens': 200, 'std_tokens': 10})
    assert normal['cohen_d'] == 10.0
    assert normal['flag'] is True
    json.dumps(normal, allow_nan=False)

    missing = _distribution_shift({'mean_tokens': None, 'std_tokens': 1},
                                  {'mean_tokens': 10, 'std_tokens': 1})
    assert missing['cohen_d'] is None
    assert missing['flag'] is None
    json.dumps(missing, allow_nan=False)

    nonfinite = _distribution_shift(
        {'mean_tokens': float('inf'), 'std_tokens': 1},
        {'mean_tokens': 10, 'std_tokens': 1})
    assert nonfinite['cohen_d'] is None
    assert nonfinite['flag'] is None
    json.dumps(nonfinite, allow_nan=False)


def test_length_shift_extreme_finite_values_are_json_safe():
    # Finite SDs near the float limit used to raise OverflowError while
    # squaring, hiding a genuine shift instead of reporting it.
    shifted = _distribution_shift(
        {'mean_tokens': 0.0, 'std_tokens': 1e308},
        {'mean_tokens': 1e308, 'std_tokens': 1e308})
    assert shifted['cohen_d'] == 1.0
    assert shifted['flag'] is True
    assert shifted['mean_difference'] == 1e308
    json.dumps(shifted, allow_nan=False)

    # The raw difference of two finite near-limit values can itself exceed
    # the float range; it must not smuggle inf into the record.
    overflow = _distribution_shift(
        {'mean_tokens': -1e308, 'std_tokens': 1.0},
        {'mean_tokens': 1e308, 'std_tokens': 1.0})
    assert overflow['mean_difference'] is None
    assert overflow['flag'] is True
    assert overflow['cohen_d'] is None
    json.dumps(overflow, allow_nan=False)


def test_length_outlier_audit_skips_malformed_lengths_and_rejects_bad_k():
    report = _length_outlier_audit(
        [1, 2, 3, 4, float('inf'), float('nan'), None, 'bad', True],
        [1, float('nan'), float('inf'), None, -3])
    assert report['n_invalid_train'] == 5
    assert report['n_invalid_heldout'] == 4
    assert report['n_probe'] == 1
    assert report['fences']['q1'] == 1.75
    assert report['fences']['q3'] == 3.25
    json.dumps(report, allow_nan=False)

    assert _length_outlier_audit([], [1])['fences'] is None
    assert _length_outlier_audit([float('nan')], [1])['fences'] is None
    # Finite near-limit lengths can overflow the upper Tukey fence; it
    # must be reported as unavailable instead of inf.
    extreme = _length_outlier_audit([1e308, 1.4e308, 1.7e308], [1])
    assert extreme['fences'] is None
    assert extreme['n_outliers'] is None
    json.dumps(extreme, allow_nan=False)

    for bad_k in (-0.1, float('nan'), float('inf'), True, '1.5'):
        try:
            _length_outlier_audit([1, 2, 3], [1], iqr_multiplier=bad_k)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid iqr_multiplier accepted: {bad_k!r}')


def test_length_shift_does_not_treat_subnormal_sd_as_zero_variance():
    report = _distribution_shift(
        {'mean_tokens': 0.0, 'std_tokens': 1e-200},
        {'mean_tokens': 1e-300, 'std_tokens': 1e-200})
    # Squaring 1e-200 underflows, but the pooled SD is still nonzero and
    # d = 1e-100 is far below the 0.5 warning threshold.
    assert report['zero_pooled_variance'] is False
    assert report['cohen_d'] == 1e-100
    assert report['flag'] is False
    json.dumps(report, allow_nan=False)


def test_quantile_argument_is_bounded_and_finite():
    for q in (0.0, 0.25, 0.5, 1.0):
        assert 1.0 <= _quantile_type7([1.0, 2.0, 3.0, 4.0], q) <= 4.0
    for q in (-0.1, 1.5, float('nan'), float('inf'), True, '0.5'):
        try:
            _quantile_type7([1.0, 2.0, 3.0, 4.0], q)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid quantile accepted: {q!r}')


def test_quantile_sample_values_must_be_finite_and_representable():
    for values in ([10 ** 400], [10 ** 400, 2 * 10 ** 400],
                   [float('nan'), 1.0, 2.0],
                   [float('inf'), 1.0, 2.0]):
        try:
            _quantile_type7(values, 0.5)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f'invalid sample accepted: {values!r}')


def test_nonzero_statistics_below_float_range_are_not_flattened_to_zero():
    tiny = Fraction(1, 10 ** 400)
    # A true finite difference/SD that underflows to 0.0 must not become a
    # confident "no shift" / zero-variance result.
    report = _distribution_shift(
        {'mean_tokens': 0, 'std_tokens': tiny},
        {'mean_tokens': tiny, 'std_tokens': tiny})
    assert report['flag'] is None
    assert report['zero_pooled_variance'] is None
    json.dumps(report, allow_nan=False)
    try:
        _quantile_type7([tiny, 2 * tiny], 0.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError('underflowing sample values were accepted')
