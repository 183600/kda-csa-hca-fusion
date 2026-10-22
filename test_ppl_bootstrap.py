"""Regression tests for unit-level perplexity bootstrap inference.

``ppl_bootstrap.py`` decides whether a held-out perplexity difference
between two LM evaluations is larger than the sampling noise of the
corpus, so its exact behaviour is pinned here:

* token-weighted point statistics (the estimand is the CORPUS mean NLL,
  not the mean of per-unit perplexities);
* CI determinism, global-RNG isolation (resampling must never perturb
  training seed streams), and alpha-nesting of the percentile interval;
* the paired test: identical inputs give exactly p=1, a clear
  heterogeneous shift is detected with the CI excluding zero, swapping
  the models negates the delta but not the p-value;
* the alignment contract (paired test refuses mismatched corpora);
* the ``train_lm_autodl.evaluate_lm_loss(return_units=True)`` integration:
  unit sums must recombine into the SAME point estimate the canonical
  path reports, and the unit list must be batch-composition invariant.

How to run::

    pytest -q test_ppl_bootstrap.py            # from the repo root

The file is intentionally standalone (no conftest dependency);
``pyproject.toml`` sets ``testpaths = ["."]`` so a bare ``pytest`` run
collects it.
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ppl_bootstrap as P  # noqa: E402


# ---------------------------------------------------------------------------
# Point statistics + CI mechanics
# ---------------------------------------------------------------------------

def test_unit_stats_token_weighting():
    # (nll_sum, n_tokens): the corpus mean is 18/8, NOT the uniform mean
    # over units ((2 + 2.5 + 3 + 2) / 4).
    units = [(4.0, 2), (5.0, 2), (3.0, 1), {'nll_sum': 6.0, 'n_tokens': 3}]
    stats = P.unit_nll_stats(units)
    assert stats['n_units'] == 4
    assert stats['n_tokens'] == 8
    assert stats['total_nll'] == 18.0
    assert stats['mean_nll'] == pytest.approx(18.0 / 8.0, rel=0, abs=1e-15)
    assert stats['perplexity'] == pytest.approx(math.exp(18.0 / 8.0),
                                                rel=1e-12)

    # Zero-token units are dropped: they carry no information and would
    # otherwise allow tokenless bootstrap replicates.
    stats = P.unit_nll_stats([(0.0, 0), (81.0, 9), (0.0, 1)])
    assert stats['n_units'] == 2 and stats['n_tokens'] == 10
    assert stats['mean_nll'] == pytest.approx(8.1, rel=0, abs=1e-15)


def test_unit_normalization_validation():
    with pytest.raises(ValueError):
        P.unit_nll_stats([])
    with pytest.raises(ValueError):
        P.unit_nll_stats([(1.0, 0), (2.0, 0)])  # no informative units
    with pytest.raises(ValueError):
        P.unit_nll_stats([(1.0, -1)])
    with pytest.raises(ValueError):
        P.unit_nll_stats([(float('nan'), 1)])
    with pytest.raises(ValueError):
        P.unit_nll_stats([(float('inf'), 1)])
    with pytest.raises(ValueError):
        P.unit_nll_stats([(1.0,)])  # not a pair
    with pytest.raises(ValueError):
        P.unit_nll_stats([{'nll_sum': 1.0}])  # missing n_tokens
    # Token counts are exact integers: a fractional/boolean count must
    # not be silently truncated into a different corpus size.
    for bad_tokens in (True, 1.9, '2'):
        with pytest.raises(ValueError):
            P.unit_nll_stats([(1.0, bad_tokens)])
    with pytest.raises(ValueError):
        P.unit_nll_stats([(True, 1)])  # bool nll_sum


def test_bootstrap_seed_must_be_an_exact_int():
    """Float / boolean seeds used to be silently truncated by ``int(seed)``."""
    units = [(4.0, 2), (5.0, 2), (3.0, 1)]
    for bad in (True, 1.5, '1'):
        with pytest.raises(ValueError):
            P.bootstrap_ppl_ci(units, n_boot=10, seed=bad)
        with pytest.raises(ValueError):
            P.paired_bootstrap_ppl(units, units, n_boot=10, seed=bad)


def test_bootstrap_ci_deterministic_rng_isolated_and_nested():
    units = [(4.0, 2), (5.0, 2), (3.0, 1), (6.0, 3), (2.0, 2)]
    state = random.getstate()
    first = P.bootstrap_ppl_ci(units, alpha=0.05, n_boot=1000, seed=11)
    assert random.getstate() == state, 'bootstrap leaked into global RNG'
    second = P.bootstrap_ppl_ci(units, alpha=0.05, n_boot=1000, seed=11)
    assert first == second, 'same seed must give a bit-identical report'
    other = P.bootstrap_ppl_ci(units, alpha=0.05, n_boot=1000, seed=12)
    assert other['ci_mean_nll'] != first['ci_mean_nll']

    wide = P.bootstrap_ppl_ci(units, alpha=0.10, n_boot=1000, seed=11)
    narrow = P.bootstrap_ppl_ci(units, alpha=0.02, n_boot=1000, seed=11)
    # Same seed => same resample set => the wider-alpha interval is the
    # nested one.
    assert wide['ci_mean_nll'][0] >= narrow['ci_mean_nll'][0]
    assert wide['ci_mean_nll'][1] <= narrow['ci_mean_nll'][1]
    assert first['ci_perplexity'] == [
        math.exp(first['ci_mean_nll'][0]),
        math.exp(first['ci_mean_nll'][1])]
    assert first['degenerate'] is False

    with pytest.raises(ValueError):
        P.bootstrap_ppl_ci(units, n_boot=0)
    with pytest.raises(ValueError):
        P.bootstrap_ppl_ci(units, alpha=1.0)
    for bad_n_boot in (True, 1.5, '10'):
        with pytest.raises(ValueError):
            P.bootstrap_ppl_ci(units, n_boot=bad_n_boot)
    for bad_alpha in (True, float('nan'), '0.05'):
        with pytest.raises(ValueError):
            P.bootstrap_ppl_ci(units, alpha=bad_alpha)

    single = P.bootstrap_ppl_ci([(2.0, 4)], n_boot=50, seed=0)
    assert single['degenerate'] is True
    assert single['ci_mean_nll'][0] == single['ci_mean_nll'][1]


# ---------------------------------------------------------------------------
# Paired bootstrap test
# ---------------------------------------------------------------------------

def _units_from_means(means, tokens=20):
    return [(m * tokens, tokens) for m in means]


def test_paired_bootstrap_identical_models_give_p_one():
    a = _units_from_means([1.5 + 0.1 * i for i in range(12)])
    rep = P.paired_bootstrap_ppl(a, a, n_boot=500, seed=5)
    assert rep['p_value'] == 1.0
    assert rep['significant'] is False and rep['worse_model'] is None
    assert rep['delta_mean_nll'] == 0.0
    assert rep['ci_delta_mean_nll'] == [0.0, 0.0]
    assert rep['ppl_ratio_a_over_b'] == 1.0
    assert rep['ci_ppl_ratio'] == [1.0, 1.0]


def test_paired_bootstrap_detects_heterogeneous_shift():
    b = _units_from_means([1.5 + 0.1 * i for i in range(12)])
    a = _units_from_means(
        [1.5 + 0.1 * i + 1.0 + 0.05 * (i % 3) for i in range(12)])
    rep = P.paired_bootstrap_ppl(a, b, n_boot=2000, seed=5)
    assert rep['significant'] is True
    assert rep['worse_model'] == 'a'
    assert rep['p_value'] < 0.05
    assert rep['ci_delta_mean_nll'][0] > 0.0
    assert rep['ci_ppl_ratio'][0] > 1.0
    assert rep['ppl_ratio_a_over_b'] == pytest.approx(
        math.exp(rep['delta_mean_nll']), rel=1e-12)

    # Swapping the models negates the delta and the direction, but the
    # two-sided centered test is symmetric: same p-value.
    swapped = P.paired_bootstrap_ppl(b, a, n_boot=2000, seed=5)
    assert swapped['p_value'] == rep['p_value']
    assert swapped['delta_mean_nll'] == pytest.approx(
        -rep['delta_mean_nll'], rel=0, abs=1e-15)
    assert swapped['worse_model'] == 'b'


def test_paired_bootstrap_alignment_and_degenerate_contracts():
    a = _units_from_means([2.0, 2.1, 2.2])
    b = _units_from_means([2.5, 2.6, 2.7])
    rep = P.paired_bootstrap_ppl(a, b, n_boot=200, seed=0)
    assert rep['n_units'] == 3 and rep['n_pairs'] == 3
    assert rep['direction'].startswith('delta_mean_nll')

    with pytest.raises(ValueError):
        P.paired_bootstrap_ppl(a, b[:-1], n_boot=200)  # length mismatch
    # A raw length mismatch must be rejected even when independently
    # dropping zero-token units would make the filtered lengths equal.
    a_with_zero = [(2.0, 4), (99.0, 0), (2.2, 4)]
    b_short = [(2.5, 4), (2.6, 4)]
    with pytest.raises(ValueError):
        P.paired_bootstrap_ppl(a_with_zero, b_short, n_boot=200)
    mismatched = [(nll, tok + 1) for nll, tok in b]
    with pytest.raises(ValueError):
        P.paired_bootstrap_ppl(a, mismatched, n_boot=200)
    # Override is explicit; the comparison still runs.
    lax = P.paired_bootstrap_ppl(a, mismatched, n_boot=200,
                                 check_alignment=False)
    assert lax['n_pairs'] == 3

    one = P.paired_bootstrap_ppl([(2.0, 4)], [(3.0, 4)], n_boot=200, seed=0)
    assert one['degenerate'] is True and one['p_value'] is None
    assert one['significant'] is None and one['worse_model'] is None


def test_paired_bootstrap_drops_zero_token_units_of_BOTH_models():
    """The paired test must keep the two unit lists index-aligned.

    A zero-token unit carries no information, so it is dropped — but the
    two models must drop the SAME positions. Dropping them per model let
    the lists drift: with A's zero-token unit at index 1 and B's at index
    2, A's third unit was paired with B's second one, and the pairwise
    token-count check could not detect it (the counts still matched).
    """
    # index:      0          1         2          3
    a = [(10.0, 4), (5.0, 0), (12.0, 6), (14.0, 10)]
    b = [(11.0, 4), (13.0, 6), (7.0, 0), (15.0, 10)]
    rep = P.paired_bootstrap_ppl(a, b, n_boot=200, seed=0)
    # A's zero-token unit is at index 1, B's at index 2: the UNION {1, 2}
    # is dropped, so indices 0 and 3 survive and stay paired with each
    # other.  Dropping per model kept three units and paired A[2] with
    # B[1] - a different corpus window - which the pairwise token-count
    # check cannot see (the counts still matched).
    assert rep['n_units'] == 2 and rep['n_pairs'] == 2
    assert rep['mean_nll_a'] == pytest.approx(24.0 / 14.0)
    assert rep['mean_nll_b'] == pytest.approx(26.0 / 14.0)
    assert rep['n_tokens_a'] == rep['n_tokens_b'] == 14

    # A zero-token unit at the SAME index for both models is dropped too.
    a2 = [(10.0, 10), (5.0, 0), (12.0, 10)]
    b2 = [(11.0, 10), (6.0, 0), (7.0, 10)]
    rep2 = P.paired_bootstrap_ppl(a2, b2, n_boot=200, seed=0)
    assert rep2['n_units'] == 2
    assert rep2['mean_nll_a'] == pytest.approx(22.0 / 20.0)
    assert rep2['mean_nll_b'] == pytest.approx(18.0 / 20.0)


def test_throughput_summary_contract():
    ts = P.throughput_summary(n_tokens=1000, elapsed_s=2.0, n_sequences=10)
    assert ts['tokens_per_second'] == pytest.approx(500.0, rel=1e-12)
    assert ts['sequences_per_second'] == pytest.approx(5.0, rel=1e-12)
    assert ts['ms_per_sequence'] == pytest.approx(200.0, rel=1e-12)
    assert ts['phase'] == 'prefill'
    assert 'end-to-end' in ts['note']
    with pytest.raises(ValueError):
        P.throughput_summary(n_tokens=10, elapsed_s=0.0)
    with pytest.raises(ValueError):
        P.throughput_summary(n_tokens=10, elapsed_s=float('inf'))
    with pytest.raises(ValueError):
        P.throughput_summary(n_tokens=-1, elapsed_s=1.0)
    with pytest.raises(ValueError):
        P.throughput_summary(n_tokens=1, elapsed_s=1.0, n_sequences=-2)
    for bad_tokens in (True, 1.5, '10'):
        with pytest.raises(ValueError):
            P.throughput_summary(n_tokens=bad_tokens, elapsed_s=1.0)
    for bad_elapsed in (True, '1.0'):
        with pytest.raises(ValueError):
            P.throughput_summary(n_tokens=10, elapsed_s=bad_elapsed)
    for bad_sequences in (True, 1.5, '2'):
        with pytest.raises(ValueError):
            P.throughput_summary(
                n_tokens=10, elapsed_s=1.0, n_sequences=bad_sequences)


# ---------------------------------------------------------------------------
# evaluate_lm_loss(return_units=True) integration
# ---------------------------------------------------------------------------

class _WindowDataset:
    """Tiny in-memory LM dataset: fixed-length windows, first label masked."""

    def __init__(self, n_windows=6, seq_len=12, vocab=13, seed=3):
        import torch
        gen = torch.Generator().manual_seed(seed)
        self.rows = []
        for _ in range(n_windows):
            ids = torch.randint(1, vocab, (seq_len,), generator=gen)
            labels = ids.clone()
            labels[0] = -100
            self.rows.append({'input_ids': ids, 'labels': labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _tiny_lm(device):
    import torch
    from ops_fused import HybridConfig
    from train_lm_autodl import LMWithHybrid
    torch.manual_seed(2027)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=4,
        csa_m=4, csa_topk=100, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=8, hca_rope_dim=8,
        kda_backend='reference',
    )
    return LMWithHybrid(13, cfg).to(device).eval()


def test_evaluate_lm_loss_return_units_matches_point_estimate(device='cpu'):
    from train_lm_autodl import evaluate_lm_loss

    model = _tiny_lm(device)
    ds = _WindowDataset()
    base = evaluate_lm_loss(model, ds, batch_size=3, device=device)
    with_units = evaluate_lm_loss(model, ds, batch_size=3, device=device,
                                  return_units=True)

    # Identical point estimate on the canonical and unit-collecting paths.
    assert with_units['loss'] == pytest.approx(base['loss'], rel=1e-6)
    assert with_units['perplexity'] == pytest.approx(base['perplexity'],
                                                     rel=1e-6)
    assert with_units['n_tokens'] == base['n_tokens']

    units = with_units['per_unit']
    assert len(units) == len(ds)
    assert [u['index'] for u in units] == list(range(len(ds)))
    assert all(math.isfinite(u['nll_sum']) and u['n_tokens'] > 0
               for u in units)
    assert sum(u['n_tokens'] for u in units) == base['n_tokens']
    assert sum(u['nll_sum'] for u in units) == pytest.approx(
        base['loss'] * base['n_tokens'], rel=1e-6)

    stats = P.unit_nll_stats(units)
    assert stats['mean_nll'] == pytest.approx(base['loss'], rel=1e-6)

    # The report must be strict-JSON safe (no NaN/Inf smuggled in) and the
    # per_unit records must be accepted verbatim by the bootstrap.
    import json
    json.dumps(with_units, allow_nan=False)
    ci = P.bootstrap_ppl_ci(units, n_boot=200, seed=0)
    assert ci['n_units'] == len(ds)
    assert ci['ci_perplexity'][0] <= ci['ci_perplexity'][1]
    assert with_units['shuffle'] is False
    assert with_units['grad_enabled'] is False


def test_evaluate_lm_loss_units_are_batch_composition_invariant(device='cpu'):
    from train_lm_autodl import evaluate_lm_loss

    model = _tiny_lm(device)
    ds = _WindowDataset()
    u1 = evaluate_lm_loss(model, ds, batch_size=1, device=device,
                          return_units=True)['per_unit']
    u2 = evaluate_lm_loss(model, ds, batch_size=2, device=device,
                          return_units=True)['per_unit']

    # Same number of units, same order, same token counts; the NLL sums
    # may differ only in float-associativity noise (per-sequence forward
    # results are batch-independent by construction).
    assert len(u1) == len(u2)
    assert [u['n_tokens'] for u in u1] == [u['n_tokens'] for u in u2]
    rep = P.paired_bootstrap_ppl(u1, u2, n_boot=200, seed=0)
    assert abs(rep['delta_mean_nll']) < 1e-5

    # Zero-token evaluation must still be handled: no supervised tokens ->
    # no units, point estimate None (the dataset contract upstream uses
    # -100 to mask). An empty dataset yields zero batches.
    empty = evaluate_lm_loss(model, _WindowDataset(n_windows=0),
                             batch_size=2, device=device, return_units=True)
    assert empty['loss'] is None and empty['perplexity'] is None
    assert empty['per_unit'] == [] and empty['n_tokens'] == 0


def test_finite_unit_sums_cannot_overflow_into_non_json_reports():
    """Finite inputs can still overflow a plain Python sum.

    The module promises strict-JSON-safe reports, so a pooled total that
    becomes inf must raise instead of leaking inf into the point stats.
    """
    import json
    with pytest.raises(ValueError):
        P.unit_nll_stats([(1e308, 1), (1e308, 1)])
    with pytest.raises(ValueError):
        P.bootstrap_ppl_ci([(1e308, 1), (1e308, 1)], n_boot=10, seed=0)
    with pytest.raises(ValueError):
        P.paired_bootstrap_ppl([(1e308, 1), (1e308, 1)],
                               [(1e308, 1), (1e308, 1)], n_boot=10, seed=0)

    # A finite full-sample total can still overflow a resampled sum when
    # the same large unit is drawn repeatedly. The scaled resampling must
    # keep the report finite and JSON-serializable.
    units = [(1.5e308, 1), (1e307, 1), (1e307, 1)]
    ci = P.bootstrap_ppl_ci(units, n_boot=100, seed=0)
    assert all(math.isfinite(x) for x in ci['ci_mean_nll'])
    json.dumps(ci, allow_nan=False)
    # The PAIRED path draws the same shared indices; its replicate sums can
    # overflow in exactly the same way, and a common overflow in both
    # models used to produce ``inf - inf = nan`` even when the point
    # estimate is finite. The scaled resampling must keep both the CI and
    # the p-value report strict-JSON-safe.
    paired = P.paired_bootstrap_ppl(units, units, n_boot=100, seed=0)
    assert paired['delta_mean_nll'] == 0.0
    assert paired['ci_delta_mean_nll'] == [0.0, 0.0]
    assert paired['p_value'] == 1.0
    assert all(math.isfinite(x) for x in paired['ci_delta_mean_nll'])
    json.dumps(paired, allow_nan=False)
    other = [(1.5e308 - 1e300, 1), (1e307, 1), (1e307, 1)]
    paired_diff = P.paired_bootstrap_ppl(units, other, n_boot=100, seed=0)
    assert all(math.isfinite(x) for x in paired_diff['ci_delta_mean_nll'])
    assert math.isfinite(paired_diff['p_value'])
    json.dumps(paired_diff, allow_nan=False)
