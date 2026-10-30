# Long-context RoPE phase precision

*Why the sin/cos tables are built from a float64 phase even when the model
runs in bf16/fp32, and how the fix is pinned.*

## The problem

`ops_rope.partial_rope` rotates with `angle = position * inv_freq`, where
`inv_freq_j = base ** (-2j / rope_dim)`. The table was previously computed
entirely in the storage/compute dtype (fp32 for fp32/bf16/fp16 inputs). That
is fine at short context, but the absolute quantization of a float32 angle
grows with the position:

| position | float32 phase error (rad) | bf16 ulps in cos |
|---:|---:|---:|
| 1e3 | < 1e-7 | 0 |
| 1e5 | ~1e-4 | 0-1 |
| 1e6 | up to ~0.04 (measured 0.039 on a 10k-position scan) | up to ~5 |

The phase error is not a storage-dtype effect: it happens before the table
is rounded to fp32/bf16. At million-token contexts it is larger than a bf16
ulp and thousands of fp32 ulps, so it changes the relative-position signal
the core attention relies on (`R(s*m - t)`).

## The fix

`_build_cos_sin` now evaluates both the phase product and `cos`/`sin` in
float64, then rounds once into the historical table dtype:

```
phase = float64(position) * float64(inv_freq)
cos   = phase.cos().to(table_dtype)
sin   = phase.sin().to(table_dtype)
```

Only the returned tables keep the old dtype, so cache/downstream memory is
unchanged; the float64 transient is built in bounded row chunks
(`_TRIG_CHUNK_ROWS = 16384`). For fp64 inputs the computation is unchanged
(bit-identical). For fp32/bf16/fp16 inputs the rounded table now equals the
float64 phase table rounded once, instead of the float32 phase table rounded.

On this CPU host, building the 64-dim table for 1M positions takes ~0.35 s
versus ~0.25 s for the old fp32 path (1.37x); the table is cached and reused
across the 2-3 rotations of a CSA/HCA layer, so this is not on the steady
decode path.

## Verification (`checks/csa.py::test_partial_rope_long_context_phase`)

The test is a numerical equivalence certificate, not a self-comparison:

1. Build an absolute reference from the float64 phase, rounded once to the
   storage dtype.
2. Recover the implementation's tables by rotating basis vectors; assert the
   fp32 tables agree within `1e-6` and the bf16 tables within one bf16 ulp.
   The pre-fix code exceeded both bounds at positions near 1e6 (0.009 and
   2.5 ulp respectively on the pinned draw).
3. Assert grouped rotation equals per-token rotation bit-for-bit for a
   64-position long-context vector in both fp32 and bf16. This is the
   prefill-versus-incremental-decode equivalence contract, and it exercises
   both the short (value-keyed) and long (identity-keyed) table caches.

## Relation to statistical equivalence testing

`docs/equivalence-testing.md` describes TOST for seed-paired *experiment*
differences. This document uses "equivalence" in the numerical sense: a
pre-specified absolute margin derived from the storage dtype (`1 ulp`) and
checked against an independent higher-precision oracle. Do not report a
TOST p-value for a deterministic operator comparison; do not use an
observed-error-derived tolerance for a confirmatory scientific claim.
