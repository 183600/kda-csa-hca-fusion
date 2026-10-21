"""Regression tests for ``ppl_calibration`` — the simulation-based
validation of the perplexity inference machinery.

What is pinned here
-------------------
1. **Simulation contract.** The synthetic corpora are deterministic under
   a local ``random.Random`` seed, paired corpora are unit-aligned
   (identical per-unit token counts, as ``paired_bootstrap_ppl``
   requires), malformed scenarios are refused, and the process-global
   RNG stream is never touched (mirroring ``ppl_bootstrap``'s isolation
   contract).
2. **Calibration verdicts hold at the committed protocol.** The full
   scenario grid (coverage at 40/12 units, null size control, power at
   two effect sizes) must clear its pre-registered tolerances, the report
   must be strict-JSON safe, row arithmetic must be self-consistent
   (``coverage == n_covered / n_reps``, verdicts must match their bounds)
   and the same call must reproduce the same rows.
3. **Decode-vs-prefill corpus-NLL parity (decoding cache x perplexity).**
   A tiny trained-geometry hybrid LM is scored on the SAME windows twice:
   once through the full-sequence prefill path, once token-by-token
   through the incremental decoding caches (``HybridDecoding``). The
   per-window ``(nll_sum, n_tokens)`` units must be aligned, per-token
   NLLs must agree to fp tolerance, and the paired bootstrap of the
   corpus NLL must find no practical difference (effect-size gate, not
   the p-value).

Run via pytest::

    pytest -q test_ppl_calibration.py

Or directly::

    python3 test_ppl_calibration.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ppl_bootstrap as P  # noqa: E402
import ppl_calibration as PC  # noqa: E402

# Reduced study settings for the always-run test path (the committed
# protocol lives in run_ppl_calibration.py): enough replicates for the
# verdicts to be stable under the pre-registered scenario tolerances while
# keeping the module in the low-seconds range.
TEST_REPS = 150
TEST_NBOOT = 250

VOCAB = 13
SEQ_LEN = 16
N_WINDOWS = 8


def test_simulation_contract_and_rng_isolation():
    # Deterministic under a local seed.
    a1 = PC.simulate_units(7, random.Random(5))
    a2 = PC.simulate_units(7, random.Random(5))
    assert a1 == a2
    assert all(isinstance(nll, float) and nll > 0.0 for nll, _ in a1)
    assert all(isinstance(tok, int) and tok >= 1 for _, tok in a1)

    # Paired corpora are unit-aligned (the paired test's contract).
    ua, ub = PC.simulate_paired_units(
        9, random.Random(6), delta=0.1, noise_sd=0.2)
    assert len(ua) == len(ub) == 9
    assert [t for _, t in ua] == [t for _, t in ub]

    # Malformed scenarios are refused instead of silently mis-measured.
    for bad in ({'n_units': 1, 'alpha': 0.1, 'tol': 0.05},
                {'n_units': 8, 'alpha': 0.1, 'tol': -1.0}):
        try:
            PC.coverage_study([bad], reps=2, n_boot=2)
        except ValueError:
            pass
        else:
            raise AssertionError(f'coverage_study accepted {bad!r}')
    try:
        PC.paired_study([{'n_units': 8, 'alpha': 0.1, 'delta': 0.2,
                          'check': 'size', 'tol': 0.05}], reps=2, n_boot=2)
    except ValueError:
        pass
    else:
        raise AssertionError('a size-control scenario with delta > 0 '
                             'must be refused')

    # The process-global RNG stream is never advanced.
    state = random.getstate()
    PC.simulate_units(4, random.Random(1))
    PC.simulate_paired_units(4, random.Random(2), delta=0.05)
    assert random.getstate() == state

    # Same seed -> identical study rows (local streams only).
    scen = [{'n_units': 12, 'alpha': 0.10, 'tol': 0.08}]
    _ = PC.coverage_study(scen, reps=20, n_boot=40, seed=3)
    rows_a = PC.coverage_study(scen, reps=20, n_boot=40, seed=3)
    rows_b = PC.coverage_study(scen, reps=20, n_boot=40, seed=3)
    assert rows_a == rows_b


def test_calibration_report_is_calibrated_and_self_consistent():
    report = PC.calibration_report(
        reps=TEST_REPS, n_boot=TEST_NBOOT, seed=0)

    assert report['all_pass'] is True, report['rows']
    assert report['n_failed'] == 0
    # The committed grid: 4 coverage + 4 paired rows.
    sections = [r['section'] for r in report['rows']]
    assert sections.count('coverage') == 4
    assert sections.count('paired') == 4

    seen = set()
    for r in report['rows']:
        if r['section'] == 'coverage':
            assert r['n_covered'] == int(round(
                r['coverage'] * r['n_reps']))
            assert abs(r['coverage'] - r['n_covered'] / r['n_reps']) < 1e-12
            assert abs(r['bias'] - (r['coverage'] - r['nominal'])) < 1e-12
            assert abs(r['nominal'] - (1.0 - r['alpha'])) < 1e-12
            assert (r['verdict'] == 'calibrated') == (
                abs(r['bias']) <= r['tol'])
            # Real calibration evidence, not a vacuous all-pass: coverage
            # must be within [0, 1] AND within a sane distance of nominal.
            assert 0.0 <= r['coverage'] <= 1.0
            assert abs(r['bias']) <= 0.12
            key = ('coverage', r['n_units'], r['alpha'])
        else:
            assert r['n_significant'] == int(round(
                r['rejection_rate'] * r['n_reps']))
            assert abs(r['rejection_rate']
                       - r['n_significant'] / r['n_reps']) < 1e-12
            assert 0.0 <= r['rejection_rate'] <= 1.0
            assert abs(r['se'] - (r['rejection_rate']
                                  * (1 - r['rejection_rate'])
                                  / r['n_reps']) ** 0.5) < 1e-12
            if r['check'] == 'size':
                assert r['delta'] == 0.0
                assert (r['meets']) == (r['rejection_rate'] <= r['threshold'])
                assert (r['verdict'] == 'within_bound') == r['meets']
                # Null control must not be grossly liberal.
                assert r['rejection_rate'] <= r['alpha'] + 0.10
            else:
                assert r['delta'] > 0.0
                assert (r['meets']) == (r['rejection_rate'] >= r['threshold'])
                assert (r['verdict'] == 'met') == r['meets']
                # A power scenario that never rejects would be vacuous.
                assert r['rejection_rate'] >= 0.4
            key = ('paired', r['n_units'], r['alpha'], r['delta'])
        assert key not in seen, f'duplicate row key {key!r}'
        seen.add(key)

    # Strict-JSON safe for the result envelope.
    json.dumps(report, allow_nan=False)

    # Determinism of the full report at the same settings.
    again = PC.calibration_report(
        reps=TEST_REPS, n_boot=TEST_NBOOT, seed=0)
    assert again['rows'] == report['rows']


def _valid_exp15_rows():
    """Minimal conforming Exp 15 payload (one row per section)."""
    return [
        {'section': 'coverage', 'n_units': 40, 'alpha': 0.10, 'n_boot': 500,
         'n_reps': 400, 'n_covered': 360, 'coverage': 0.9, 'nominal': 0.9,
         'se': (0.9 * 0.1 / 400) ** 0.5, 'tol': 0.05, 'bias': 0.0,
         'verdict': 'calibrated', 'note': ''},
        {'section': 'paired', 'check': 'size', 'n_units': 40, 'alpha': 0.10,
         'delta': 0.0, 'noise_sd': 0.2, 'n_boot': 500, 'n_reps': 400,
         'n_significant': 40, 'rejection_rate': 0.1,
         'se': (0.1 * 0.9 / 400) ** 0.5, 'threshold': 0.16, 'tol': 0.06,
         'verdict': 'within_bound', 'meets': True, 'note': ''},
        {'section': 'paired', 'check': 'power', 'n_units': 40, 'alpha': 0.10,
         'delta': 0.10, 'noise_sd': 0.2, 'n_boot': 500, 'n_reps': 400,
         'n_significant': 240, 'rejection_rate': 0.6,
         'se': (0.6 * 0.4 / 400) ** 0.5, 'threshold': 0.55, 'min_power': 0.55,
         'verdict': 'met', 'meets': True, 'note': ''},
        {'section': 'decode_prefill', 'n_windows': 16, 'seq_len': 24,
         'n_tokens': 368, 'max_window_nll_diff': 1e-6,
         'prefill_ppl': 13.0, 'decode_ppl': 13.0, 'delta_mean_nll': 1e-8,
         'ci_delta_mean_nll': [-2e-8, 3e-8], 'ppl_ratio': 1.0 + 1e-8,
         'ci_ppl_ratio': [math.exp(-2e-8), math.exp(3e-8)], 'p_value': 0.5,
         'significant': False, 'window_tol_nll': 1e-4,
         'effect_tol_nll': 1e-6, 'effect_tol_ratio': 1e-5, 'parity_ok': True,
         'effect_ok': True, 'finite_ok': True, 'note': ''},
    ]


def test_exp15_row_contract_reports_overflow_instead_of_crashing():
    """An overflowing CI bound must be a CONTRACT VIOLATION, not a crash.

    ``math.exp`` raises ``OverflowError`` above ~709.78. The validator
    used to call it unguarded, so a row whose ``ci_delta_mean_nll`` bound
    overflowed aborted the whole validation with an unexpected exception
    instead of reporting the violation.
    """
    import copy

    from results_schema import ROW_CONTRACTS

    contract = ROW_CONTRACTS['exp15_ppl_calibration']
    rows = copy.deepcopy(_valid_exp15_rows())
    row = rows[3]
    row['ci_delta_mean_nll'] = [700.0, 900.0]
    row['ci_ppl_ratio'] = [math.exp(700.0), math.exp(700.0)]
    violations = contract(rows, {})
    assert any('ci_ppl_ratio' in v for v in violations), violations


def test_exp15_row_contract_rejects_tampering():
    import copy

    from results_schema import ROW_CONTRACTS

    contract = ROW_CONTRACTS['exp15_ppl_calibration']
    rows = _valid_exp15_rows()
    assert contract(rows, {}) == [], contract(rows, {})

    def _mutations():
        yield 'coverage rate not derived from counts', \
            lambda r: r[0].__setitem__('coverage', 0.5)
        yield 'coverage verdict contradicts bias', \
            lambda r: r[0].__setitem__('verdict', 'biased')
        yield 'coverage tol not the pre-registered value', \
            lambda r: r[0].__setitem__('tol', 0.2)
        yield 'paired size tol not the pre-registered value', \
            lambda r: r[1].__setitem__('tol', 0.2)
        yield 'paired size with non-zero delta', \
            lambda r: r[1].__setitem__('delta', 0.1)
        yield 'paired meets contradicts rate vs threshold', \
            lambda r: r[1].__setitem__('meets', False)
        yield 'paired power min_power not the pre-registered floor', \
            lambda r: (r[2].__setitem__('min_power', 0.2),
                       r[2].__setitem__('threshold', 0.2))
        yield 'paired rate not derived from counts', \
            lambda r: r[1].__setitem__('rejection_rate', 0.5)
        yield 'paired threshold not alpha + recorded tol', \
            lambda r: r[1].__setitem__('threshold', 0.01)
        yield 'decode_prefill effect_ok flipped', \
            lambda r: r[3].__setitem__('effect_ok', False)
        yield 'decode_prefill ci_ppl_ratio not exponentiated ci', \
            lambda r: r[3].__setitem__('ci_ppl_ratio', [1.0, 1.0])
        yield 'decode_prefill p_value out of range', \
            lambda r: r[3].__setitem__('p_value', 1.5)
        yield 'duplicate coverage row key', \
            lambda r: r.append(copy.deepcopy(r[0]))

    for name, mutate in _mutations():
        tampered = copy.deepcopy(rows)
        mutate(tampered)
        assert contract(tampered, {}), (
            f'tampered payload accepted: {name}')


def _tiny_hybrid_lm(device):
    """Random-weight hybrid LM (KDA + CSA + HCA) at test geometry."""
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
    return LMWithHybrid(VOCAB, cfg).to(device).eval()


def _window_units_prefill(model, ids):
    """Per-window ``(nll_sum, n_tokens)`` from one full-sequence forward."""
    model.hybrid.reset_state()
    tower = model.hybrid(model.embed(ids))
    logits = model.lm_head(model.norm_f(tower))[:, :-1]
    targets = ids[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                          reduction='none').view(targets.shape)
    return (float(nll.sum()), int(nll.numel()))


def _window_units_decode(clone, model, ids):
    """Per-window ``(nll_sum, n_tokens)`` through the incremental caches."""
    clone.reset()
    pieces = [clone(model.embed(ids[:, t:t + 1]))
              for t in range(ids.shape[1] - 1)]
    tower = torch.cat(pieces, dim=1)
    logits = model.lm_head(model.norm_f(tower))
    targets = ids[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                          reduction='none').view(targets.shape)
    return (float(nll.sum()), int(nll.numel()))


def test_decode_prefill_perplexity_parity(device='cpu'):
    from lm_consistency import build_decoding_clone

    model = _tiny_hybrid_lm(device)
    clone = build_decoding_clone(model, device=device)
    gen = torch.Generator(device='cpu').manual_seed(3)
    windows = [torch.randint(1, VOCAB, (1, SEQ_LEN), generator=gen).to(device)
               for _ in range(N_WINDOWS)]

    pre_units, dec_units = [], []
    max_window_nll_diff = 0.0
    with torch.no_grad():
        for ids in windows:
            pre_sum, pre_tok = _window_units_prefill(model, ids)
            dec_sum, dec_tok = _window_units_decode(clone, model, ids)
            assert pre_tok == dec_tok == SEQ_LEN - 1
            pre_units.append({'nll_sum': pre_sum, 'n_tokens': pre_tok})
            dec_units.append({'nll_sum': dec_sum, 'n_tokens': dec_tok})
            max_window_nll_diff = max(max_window_nll_diff,
                                      abs(pre_sum - dec_sum))

    # Corpus-level prefill == incremental decode on the same metric.
    pre_stat = P.unit_nll_stats(pre_units)
    dec_stat = P.unit_nll_stats(dec_units)
    assert abs(pre_stat['mean_nll'] - dec_stat['mean_nll']) < 1e-6
    assert max_window_nll_diff < 1e-4, (
        f'decode/prefill per-window NLL sums diverged by '
        f'{max_window_nll_diff}')

    paired = P.paired_bootstrap_ppl(dec_units, pre_units,
                                    n_boot=1000, seed=0)
    # Effect-size gate: the paths must be metrically indistinguishable.
    assert abs(paired['delta_mean_nll']) < 1e-6
    assert abs(paired['ppl_ratio_a_over_b'] - 1.0) < 1e-5
    assert paired['p_value'] is not None
    assert 0.0 <= paired['p_value'] <= 1.0
    assert paired['n_units'] == N_WINDOWS
    json.dumps({'pre': pre_units, 'dec': dec_units, 'paired': paired},
               allow_nan=False)


def main() -> int:
    test_simulation_contract_and_rng_isolation()
    test_calibration_report_is_calibrated_and_self_consistent()
    test_decode_prefill_perplexity_parity()
    print('\nAll ppl-calibration tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
