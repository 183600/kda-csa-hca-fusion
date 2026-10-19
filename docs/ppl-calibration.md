# Calibrating the perplexity inference machinery (Exp 15)

*Why the paired bootstrap behind Exp 11's perplexity claims now ships a
certificate: pre-registered coverage/size/power thresholds checked by
simulation, plus a decode-vs-prefill corpus-NLL parity gate.*

## The problem

`ppl_bootstrap.py` turns two models' per-unit NLLs into the paired
bootstrap confidence intervals and p-values that Exp 11 (`run_quant_lm`,
`run_quant_recall` verdicts) and the decode/quantization narratives rest
on. Nothing in the pipeline, however, checked the *statistical* claim the
machinery makes: that a nominal 90%/95% interval actually covers at ~90%/
95%, that the paired test rejects at ~`alpha` under the null, and that it
has enough power to detect the effect sizes the results talk about. A
subtle bug in resampling, centering or the p-value orientation would
have kept every downstream number plausible-looking.

There is a second, quieter coupling: per-token decode and batched prefill
must agree on corpus NLL — the perplexity numbers are computed through
the same scoring path that the cache-based decode uses, so a
decode-vs-prefill scoring mismatch would contaminate exactly the metric
the bootstrap intervals describe.

## What was built

`ppl_calibration.py` — a simulation harness that drives the REAL
inference machinery (`lm_consistency.PairedBootstrap` /
`ppl_bootstrap` primitives) against synthetic per-unit NLLs with known
ground truth:

* **coverage**: units are valid with probability `1 - alpha`; the
  fraction of simulated datasets whose CI covers the true mean NLL is
  counted. A CI would be rejected if its coverage deviates from nominal
  by more than the pre-registered tolerance (0.05 moderate, 0.08 small);
* **size control**: `delta = 0` paired data; the rejection rate must stay
  within `alpha + tol` (tol 0.06 moderate, 0.10 small);
* **power**: `delta > 0` with pre-registered floors (0.55 at
  `delta = 0.10`, 0.90 at `delta = 0.20`);
* **decode/prefill parity**: `lm_consistency.build_decoding_clone` gives
  a prefill LM and its decode-mode clone; per-window corpus NLLs are
  scored both ways and compared with a paired bootstrap
  (`run_ppl_calibration.py` gate: per-window max |diff| ≤ 1e-4 nats and
  recorded effect tolerances on the mean/ratio BEFORE the certificate is
  archived).

Simulation uses only `random()`/`randint` through a locally seeded
`random.Random` (version-stable across Python builds, global RNG
untouched), and paired units share their difficulty draw so the paired
structure is real. All thresholds are pre-registered in
`default_scenarios()` and mirrored by the `exp15_ppl_calibration` row
contract in `results_schema.py`: a tampered artifact (widened tolerance,
flipped verdict, threshold that no longer equals `alpha + tol` or the
recorded floor, non-derived rate) is refused by the schema check.

## Measured certificate (Exp 15, committed grid, `reps=400, n_boot=500`)

| section | scenario | measured | bound/floor | verdict |
|---|---|---|---|---|
| coverage | `n=40, alpha=0.10` | 0.868 (bias −0.032) | ±0.05 | calibrated |
| coverage | `n=40, alpha=0.05` | 0.915 (bias −0.035) | ±0.05 | calibrated |
| coverage | `n=12, alpha=0.10` | 0.873 (bias −0.027) | ±0.08 | calibrated |
| coverage | `n=12, alpha=0.05` | 0.887 (bias −0.062) | ±0.08 | calibrated |
| size | `n=40, alpha=0.10` | 0.087 | ≤0.16 | within bound |
| size | `n=12, alpha=0.10` | 0.172 | ≤0.20 | within bound |
| power | `delta=0.10` | 0.615 | ≥0.55 | met |
| power | `delta=0.20` | 0.988 | ≥0.90 | met |

decode vs prefill: 16 windows / 368 tokens, max per-window
`|ΔNLL| = 7.6e-06` nats (tol 1e-4), `Δmean NLL = +2.1e-08`,
`PPL ratio = 1.000000021`, bootstrap `p = 0.54` (not significant),
`parity_ok = effect_ok = finite_ok = True`.

Interpretation: the inference path is honest at the scales results use
(40 units is the Exp 11 contrast size); small-sample behaviour is
slightly conservative on coverage and mildly liberal on size at 12 units
— documented, bounded by the recorded 0.20 tolerance, and visible in the
artifact rather than hidden.

## Verification

* `test_ppl_calibration.py` — RNG isolation/determinism contract, the
  committed report staying calibrated + self-consistent, decode/prefill
  parity, and the schema contract rejecting 11 distinct tampering modes.
* `run_ppl_calibration.py` — Exp 15: parity gate first, then the
  certificate is written as a `results_schema` envelope with a
  provenance side file.
* `results_schema.py` — `exp15_ppl_calibration` row contract: every rate
  re-derived from its counts, every bound re-derived from its
  pre-registered tolerance/floor, duplicate rows rejected.

## Limitations

* Coverage-estimate resolution is `1/reps` (0.0025 at the committed
  grid); the tolerances absorb this plus the documented small-sample
  undercoverage, they are not a statement about any other simulator
  geometry.
* The power floors are calibrated to the simulator's noise geometry
  (`noise_sd`, triangular unit difficulty); they are evidence about the
  inference path, not a promise for the real LM's effect-size
  distribution.
* Decode/prefill parity uses a tiny random-weight hybrid LM for speed;
  it pins the scoring machinery, not the trained checkpoint's numerics.
