"""Experiment 15 — calibrated perplexity inference for the decode cache.

Motivation
----------
Two claims recur across this repository's LM results, and both are
statistical:

1. *"Perplexity differences smaller than the corpus-noise width are not
   decisions"* — quantified by ``ppl_bootstrap``'s percentile CIs and
   paired tests (used by ``run_quant_lm``'s quantization-damage flags and
   ``train_lm_autodl``'s sealed-test report). Whether that inference is
   actually calibrated (CIs covering at the nominal rate, null tests not
   firing on identical models) was never measured.
2. *"Incremental decoding equals prefill"* — established token-by-token
   and logit-by-logit by the cache regression tests, but not at the
   level of the reported corpus metric (the token-weighted NLL /
   perplexity), which is what the LM tables actually contain.

This experiment closes both gaps in one artifact:

* ``coverage`` / ``paired`` rows: :mod:`ppl_calibration` simulations over
  synthetic corpora with a KNOWN population mean NLL. Coverage must stay
  within each scenario's pre-registered tolerance; the paired test's
  null size must stay within ``alpha + tol`` (a slightly wider tolerance
  is pre-registered for the 12-unit small-corpus regime, where
  percentile inference is known to be mildly liberal) and its power at
  the two pre-registered effect sizes must clear its floor.
* ``decode_prefill`` row: a tiny hybrid LM (KDA + CSA + HCA with the
  real incremental caches via ``run_decoding.HybridDecoding``) scores
  the SAME windows through the full-sequence prefill path and through
  token-by-token decoding. Per-window ``(nll_sum, n_tokens)`` units are
  passed to the shipped paired bootstrap; the row records the max
  per-window NLL divergence, the corpus-PPL delta and the paired CI. The
  gate is an EFFECT-SIZE bound (|delta| and |PPL ratio - 1| below fp
  tolerance), not a p-value: a paired test may legitimately flag a
  systematic fp-order difference of ~1e-8 as "significant", which is
  exactly why the repo also carries tolerance flags next to its
  p-values.

Parity gate
-----------
Any calibration verdict failure or any decode/prefill effect failure
refuses to write the result file — the artifact is a certificate, not a
log.

Outputs
-------
``results/exp15_ppl_calibration.json`` (schema kind
``exp15_ppl_calibration``) + a provenance side file. Environment
overrides: ``PPL_CAL_REPS`` (simulation replicates, default 400),
``PPL_CAL_NBOOT`` (bootstrap replicates per simulation, default 500),
``PPL_CAL_SEED`` (base seed, default 0); the defaults are the committed
protocol.

Run::

    python3 run_ppl_calibration.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ppl_bootstrap as P  # noqa: E402
import ppl_calibration as PC  # noqa: E402
from kaggle_setup import (  # noqa: E402
    append_run_manifest, capture_provenance, make_run_manifest_entry,
    set_global_seed, write_json_atomic,
)
from results_schema import (  # noqa: E402
    SchemaError, write_checked as write_results_checked,
)

# Committed simulation protocol (see ppl_calibration.default_scenarios).
EXP_REPS = 400
EXP_NBOOT = 500
EXP_SEED = 0
# Fixed model-construction draw, independent of the calibration seed. The
# same model is used for both sides of the decode-vs-prefill parity probe.
MODEL_CONSTRUCTION_SEED = 2027

# Decode-vs-prefill geometry: small enough for a CPU CI child process,
# large enough that the incremental caches complete several compressed
# blocks per window and that the paired test has 16 units to resample.
DP_VOCAB = 13
DP_SEQ_LEN = 24
DP_WINDOWS = 16
DP_NBOOT = 4000
# Effect-size gate (fp32 corpus-NLL drift must be below this; the paired
# p-value is reported but intentionally NOT gated).
DP_EFFECT_TOL_NLL = 1e-6
DP_EFFECT_TOL_RATIO = 1e-5
DP_WINDOW_TOL_NLL = 1e-4


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _tiny_hybrid_lm(device):
    from ops_fused import HybridConfig
    from train_lm_autodl import LMWithHybrid

    torch.manual_seed(MODEL_CONSTRUCTION_SEED)
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
    return LMWithHybrid(DP_VOCAB, cfg).to(device).eval()


def bench_decode_prefill(device='cpu', *, seq_len: int = DP_SEQ_LEN,
                         n_windows: int = DP_WINDOWS) -> list:
    """Corpus-NLL parity of incremental decode vs full-sequence prefill.

    Scores ``n_windows`` random windows through both paths, builds the
    per-window sufficient statistics, runs the shipped paired bootstrap,
    and returns one JSON-safe ``decode_prefill`` row. The prefill path and
    the decode path use the exact same logits->NLL arithmetic (next-token
    shift, no masking) so the only difference is the computation path:
    chunked full-sequence forward vs token-by-token with the CSA/HCA/KDA
    incremental caches.
    """
    from lm_consistency import build_decoding_clone

    # The parity gate uses a PAIRED bootstrap over windows: fewer than two
    # windows makes the paired p-value undefined, and a one-token window has
    # no next-token targets at all. Reject those degenerate schedules up
    # front instead of failing later in ``math.isfinite(None)`` / the
    # paired bootstrap's empty-informative-unit guard.
    for name, value, minimum in (('seq_len', seq_len, 2),
                                 ('n_windows', n_windows, 2)):
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < minimum:
            raise ValueError(
                f'bench_decode_prefill: {name}={value!r} must be an int '
                f'>= {minimum}.')

    model = _tiny_hybrid_lm(device)
    clone = build_decoding_clone(model, device=device)
    gen = torch.Generator(device='cpu').manual_seed(3)
    pre_units, dec_units = [], []
    max_window_diff = 0.0
    n_tokens = 0
    with torch.no_grad():
        for _ in range(int(n_windows)):
            ids = torch.randint(1, DP_VOCAB, (1, int(seq_len)),
                                generator=gen).to(device)
            model.hybrid.reset_state()
            tower = model.hybrid(model.embed(ids))
            logits = model.lm_head(model.norm_f(tower))[:, :-1]
            targets = ids[:, 1:]

            def _nll_sum(logit_rows):
                nll = F.cross_entropy(
                    logit_rows.reshape(-1, DP_VOCAB),
                    targets.reshape(-1), reduction='none')
                return float(nll.sum()), int(nll.numel())

            pre_sum, pre_tok = _nll_sum(logits)

            clone.reset()
            pieces = [clone(model.embed(ids[:, t:t + 1]))
                      for t in range(ids.shape[1] - 1)]
            dec_logits = model.lm_head(model.norm_f(
                torch.cat(pieces, dim=1)))
            dec_sum, dec_tok = _nll_sum(dec_logits)

            if pre_tok != dec_tok:
                raise AssertionError(
                    f'decode/prefill token count mismatch: {pre_tok} vs '
                    f'{dec_tok}')
            max_window_diff = max(max_window_diff, abs(pre_sum - dec_sum))
            n_tokens += pre_tok
            pre_units.append({'nll_sum': pre_sum, 'n_tokens': pre_tok})
            dec_units.append({'nll_sum': dec_sum, 'n_tokens': dec_tok})

    paired = P.paired_bootstrap_ppl(
        dec_units, pre_units, n_boot=DP_NBOOT, seed=0)
    pre_stat = P.unit_nll_stats(pre_units)
    dec_stat = P.unit_nll_stats(dec_units)
    delta = paired['delta_mean_nll']
    ratio = paired['ppl_ratio_a_over_b']
    parity_ok = max_window_diff <= DP_WINDOW_TOL_NLL
    effect_ok = (abs(delta) <= DP_EFFECT_TOL_NLL
                 and abs(ratio - 1.0) <= DP_EFFECT_TOL_RATIO)
    finite_values = [
        pre_stat['mean_nll'], dec_stat['mean_nll'], delta,
        paired['ci_delta_mean_nll'][0], paired['ci_delta_mean_nll'][1],
        ratio]
    if paired['p_value'] is not None:
        finite_values.append(paired['p_value'])
    finite_ok = all(math.isfinite(v) for v in finite_values)
    return [{
        'section': 'decode_prefill',
        'n_windows': int(n_windows),
        'seq_len': int(seq_len),
        'n_tokens': int(n_tokens),
        'max_window_nll_diff': max_window_diff,
        'prefill_ppl': pre_stat['perplexity'],
        'decode_ppl': dec_stat['perplexity'],
        'delta_mean_nll': delta,
        'ci_delta_mean_nll': list(paired['ci_delta_mean_nll']),
        'ppl_ratio': ratio,
        'ci_ppl_ratio': list(paired['ci_ppl_ratio']),
        'p_value': paired['p_value'],
        'significant': bool(paired['significant']),
        'window_tol_nll': DP_WINDOW_TOL_NLL,
        'effect_tol_nll': DP_EFFECT_TOL_NLL,
        'effect_tol_ratio': DP_EFFECT_TOL_RATIO,
        'parity_ok': bool(parity_ok),
        'effect_ok': bool(effect_ok),
        'finite_ok': bool(finite_ok),
        'note': ('per-window (nll_sum, n_tokens) units of the same windows '
                 'through full-sequence prefill and token-by-token decode; '
                 'gate is the effect size, the p-value is reported only'),
    }]


def main() -> int:
    reps = _env_int('PPL_CAL_REPS', EXP_REPS)
    n_boot = _env_int('PPL_CAL_NBOOT', EXP_NBOOT)
    seed = _env_int('PPL_CAL_SEED', EXP_SEED)
    # Seed every RNG family, not just torch's global generator: the
    # calibration code uses local random.Random streams. ``_tiny_hybrid_lm``
    # then fixes the model-construction draw to MODEL_CONSTRUCTION_SEED, the
    # value recorded in both the result metadata and the run manifest.
    set_global_seed(seed, deterministic=True)

    print(f'[exp15] calibration grid (reps={reps}, n_boot={n_boot}, '
          f'seed={seed})...')
    report = PC.calibration_report(reps=reps, n_boot=n_boot, seed=seed)
    rows = list(report['rows'])
    for r in rows:
        if r['section'] == 'coverage':
            print(f"  coverage n_units={r['n_units']:>2} "
                  f"alpha={r['alpha']:.2f}: {r['coverage']:.3f} "
                  f"(nominal {r['nominal']:.2f}, bias {r['bias']:+.3f} "
                  f"+/-{r['tol']:.2f}) -> {r['verdict']}")
        else:
            print(f"  paired  {r['check']:>5} n_units={r['n_units']:>2} "
                  f"delta={r['delta']:.2f}: rate={r['rejection_rate']:.3f} "
                  f"(bound {r['threshold']:.2f}) -> {r['verdict']}")

    print('[exp15] decode-vs-prefill corpus-NLL parity...')
    dp_rows = bench_decode_prefill()
    rows += dp_rows
    dp = dp_rows[0]
    print(f"  windows={dp['n_windows']} T={dp['seq_len']} "
          f"max_window_diff={dp['max_window_nll_diff']:.3e} "
          f"(tol {dp['window_tol_nll']:.0e})")
    print(f"  prefill_ppl={dp['prefill_ppl']:.6f} "
          f"decode_ppl={dp['decode_ppl']:.6f} "
          f"delta_nll={dp['delta_mean_nll']:+.3e} "
          f"ratio={dp['ppl_ratio']:.9f} (p={dp['p_value']:.4f}, "
          f"significant={dp['significant']})")
    print(f"  parity_ok={dp['parity_ok']} effect_ok={dp['effect_ok']} "
          f"finite_ok={dp['finite_ok']}")

    if not report['all_pass'] or not (
            dp['parity_ok'] and dp['effect_ok'] and dp['finite_ok']):
        print('[exp15] ERROR: calibration or decode/prefill parity gate '
              'failed; refusing to write a result file')
        return 1

    os.makedirs('results', exist_ok=True)
    metadata = {
        'device': 'cpu',
        'simulation': report['settings'],
        'scenarios': PC.default_scenarios(),
        'method': report['method'],
        'decode_prefill_protocol': {
            'vocab': DP_VOCAB, 'seq_len': DP_SEQ_LEN,
            'n_windows': DP_WINDOWS, 'n_boot': DP_NBOOT,
            'window_tol_nll': DP_WINDOW_TOL_NLL,
            'effect_tol_nll': DP_EFFECT_TOL_NLL,
            'effect_tol_ratio': DP_EFFECT_TOL_RATIO,
        },
        'seeds': {
            'value': seed,
            'scheme': ('set_global_seed(seed) covers Python/NumPy/torch/'
                       'CUDA; ppl_calibration uses local random.Random '
                       f'streams from (seed={seed}, scenario, replicate); '
                       f'torch.manual_seed({MODEL_CONSTRUCTION_SEED}) for '
                       'model construction, Generator(3) for windows'),
        },
        'provenance': capture_provenance(),
    }
    try:
        write_results_checked(
            'results/exp15_ppl_calibration.json', rows,
            kind='exp15_ppl_calibration', metadata=metadata)
    except SchemaError as e:
        print(f'[exp15] ERROR: refusing to write a contract-violating '
              f'result file: {e}')
        return 1
    print('\nSaved: results/exp15_ppl_calibration.json '
          '(schema kind=exp15_ppl_calibration)')
    append_run_manifest(make_run_manifest_entry(
        experiment='exp15_ppl_calibration',
        seeds=seed,
        provenance=metadata['provenance'],
        hyperparameters={
            'simulation': report['settings'],
            'scenarios': PC.default_scenarios(),
            'method': report['method'],
            'decode_prefill_protocol': metadata['decode_prefill_protocol'],
            'model_construction_seed': MODEL_CONSTRUCTION_SEED,
            'window_generator_seed': 3,
        },
        data={
            'kind': 'synthetic_calibration_and_decode_parity',
            'scenarios': PC.default_scenarios(),
            'decode_prefill_protocol': metadata['decode_prefill_protocol'],
            'corpora': 'synthetic NLL draws with known population mean',
        },
        artifacts={
            'results': 'results/exp15_ppl_calibration.json',
            'provenance': (
                'results/exp15_ppl_calibration_provenance.json'),
        },
        results={
            'all_pass': bool(report['all_pass']),
            'n_rows': len(rows),
            'decode_prefill_parity_ok': bool(dp['parity_ok']),
            'decode_prefill_effect_ok': bool(dp['effect_ok']),
        },
    ))
    try:
        prov = capture_provenance()
        prov['seeds'] = metadata['seeds']
        write_json_atomic(
            prov, 'results/exp15_ppl_calibration_provenance.json',
            indent=2, allow_nan=False)
    except Exception as e:  # pragma: no cover - side file only
        print(f'[exp15] WARNING: failed to write provenance: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
