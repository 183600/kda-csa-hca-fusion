# Backend resolution provenance: requested ≠ resolved

*Why every result file answers "which kernels actually ran?" — and how
the optional-backend probes are kept honest.*

## The problem

Three operators can swap their PyTorch reference implementation for an
optional accelerated package:

| operator | config / env | optional package | kernels |
|---|---|---|---|
| KDA | `kda_backend` / `KDA_BACKEND` | `flash-linear-attention` | Triton `chunk_kda` / `fused_recurrent_kda` |
| Mamba-2 | `mamba_backend` / `MAMBA_BACKEND` | `mamba-ssm` | Triton SSD (`Mamba2`) |
| RWKV-7 | `rwkv_backend` / `RWKV_BACKEND` | `flash-linear-attention` (`fla.ops.rwkv7`) | `chunk_rwkv7` / `fused_mul_recurrent_rwkv7` |

With `'auto'`, dispatch uses the accelerated kernel **only when CUDA is
available and the package imports**; otherwise it silently runs the
reference path and records a marker on the mixer
(`effective_backend = 'reference (auto fallback)'`).

A result file that recorded only the **requested** value — e.g.
`"mamba_backend": "auto"` — would leave two runs with the same command and
the same result filename meaning opposite things:

* a T4 Kaggle box: mamba_ssm Triton kernels actually ran;
* a CPU box (or a CUDA box with a broken optional install): the pure-PyTorch
  reference ran.

For retrieval-recall metrics (MQAR accuracy, CSA indexer gold-block recall,
recall-reach sweeps) that difference is exactly the kind of silent factor
that makes tables incomparable across machines. Provenance must name the
resolved backend, not the intent.

## What is recorded

`kaggle_setup.capture_provenance()` merges the
`backend_resolution_provenance()` block. Every runner that calls
`capture_provenance()` (all of them, including the recall experiments and
`make_run_manifest_entry`) picks these fields up automatically:

| field | meaning |
|---|---|
| `mamba_backend`, `rwkv_backend` | requested value (env var; `kda_backend` is the same kind of field) |
| `kda_backend_resolved`, `mamba_backend_resolved`, `rwkv_backend_resolved` | what `resolve_*_backend()` says dispatch will actually use |
| `fla_available`, `fla_rwkv7_available`, `mamba_ssm_available` | package importable (distinguishes "absent" from "installed but broken") |
| `mamba_ssm_version` | installed wheel version |

Rules:

* **Never raises.** Each probe is wrapped; failure degrades to `None`. A
  provenance capture is not allowed to take down an experiment.
* **Independent.** A broken Mamba probe cannot erase the KDA/RWKV record.
* **Strict JSON.** Only `str` / `bool` / `None` values are emitted.
* On a CPU-only box `*_resolved` is always `'reference'` even when the
  package is installed — that is the honest answer to "what ran".

## The probe contract (sibling symmetry)

All three availability probes share one policy:

```
probe()  ->  never raises; broken install == not available
explicit backend='<accelerated>'  ->  raises the actionable import error
backend='auto'  ->  accelerates when possible, falls back with a RuntimeWarning
```

* `ops_kda_backend.fla_available()` catches every import-chain failure and
  returns `False`; an explicit `'fla'` request still gets an
  `ImportError` that names the original exception.
* `mamba2_official_available()` catches every import-chain failure and
  returns `False`; an explicit `'official'` request still gets an
  `ImportError` that names the original exception (Triton/torch mismatch,
  half-extracted wheel, ...) instead of leaking a deep Triton traceback.
* `rwkv7_fla_available()` catches every import-chain failure; a failed
  import is cached as a definite failure.

`ops_rwkv_backend._load_fla_rwkv7()` attempts the import once and caches
the outcome (`_FLA_PROBED`). Any failure stores an actionable
`ImportError` (with the original exception as `__cause__`) before setting
`_FLA_PROBED`, so a broken install is a definite, repeatable "not
available" and dispatch either falls back (`'auto'`) or fails loudly
(`'fla'`).

The `'auto'` fallback surface is aligned with
`ops_kda_backend.kda_forward`: `ImportError` / `ValueError` / `TypeError` /
`NotImplementedError` / `AssertionError` / `RuntimeError`, with CUDA
out-of-memory **re-raised** (a resource failure must stay loud) and a
`RuntimeWarning` for every actual fallback.

## Tests

`test_backend_probe_provenance.py` (CPU-only, no optional packages needed):

* broken `fla.ops.rwkv7` → both probes `False`, cached error present, no
  `None` callables, explicit `'fla'` raises, `'auto'`/`'reference'` work;
* fake healthy `fla.ops.rwkv7` → adapter passes log-decay through and forms
  `a = -kk`, `b = kk * a`; `T>1` uses the chunked kernel, `T==1` the
  recurrent one; `'auto'` dispatches under mocked CUDA;
* fake kernel raising `TypeError` → warning + reference output; raising
  `RuntimeError('CUDA out of memory ...')` → re-raised;
* broken `mamba_ssm` → probe `False` twice, explicit load raises
  `ImportError` with the original error as `__cause__`;
* fake `mamba_ssm` → kwargs filtered against the constructor signature with
  a warning; `transfer_reference_state` maps `dt_proj.bias → dt_bias` and
  lists fused-layout parameters it cannot source;
* `capture_provenance()` carries all fields, agrees with the
  `resolve_*_backend` helpers, stays strict-JSON, and survives broken
  optional installs;
* HCA single-branch / CSA overlapped **recompression seams**: blocks
  completed from a partial accumulator plus later chunks match the
  full-sequence compression and the token-by-token path for every chunking
  (block bookkeeping exact; values within fp32 softmax round-off), not
  just for token-by-token fills.

## Limitations

* ``build_mamba2_mixer`` / ``build_rwkv7_mixer`` fall back at BUILD
  time on a narrower exception surface; a drifted constructor raising
  `TypeError` would crash `'auto'` instead of falling back (dispatch-time
  fallbacks are aligned).
* The run manifest records provenance per run, but no figure/analysis code
  groups result rows by resolved backend; a recall table mixing
  `reference` and `mamba_ssm` rows is detectable but not flagged.
