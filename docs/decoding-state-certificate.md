# Decoding-state certificate (Exp 6 companion)

The repository's central efficiency claim is that the memory operators
(KDA, Mamba-2, RWKV-7) carry a **constant-size decoding state**: the
recurrent state is a sufficient statistic of the prefix, so the
per-request footprint does not grow with context. Softmax attention grows
linearly (KV cache), and CSA/HCA grow with the compressed-block count
(~`T/m` + fixed window). The claim is backed by the analytical accounting
(`run_kv_cache.py`, `cache_accounting.py`) and per-op tests; the Exp 6
certificate additionally MEASURES it across all operators. For a
counterfactual proof that each state family is actually read during
incremental decode, see `docs/decode-state-ablation.md`.

## What Exp 6 records

Every `run_decoding.py` row carries three fields, measured from the
live tensors **after** the full prefill + decode loop and **outside** the
timed region (so they cannot perturb the latency numbers):

| Field | Meaning |
|---|---|
| `decode_state_bytes` | Logically live state bytes (valid prefix + accumulator) |
| `decode_state_allocated_bytes` | Physical storage, including geometric cache slack and the full ring buffer |
| `decode_state_breakdown` | Per-component `{allocated_bytes, valid_bytes}` |

The run envelope includes `metadata.decode_state_certificate`: per operator,
the valid bytes at each `prefill_len`, a `classification`
(`constant` only when the total AND every observed component are
identical at every length, else `growing`), `max_over_min` (the observed
growth ratio), `growing_components`, the per-component series, and the
`total_valid_flat_but_components_grow` cancellation flag. This turns the
claim into a per-run measurement with an auditable trail rather than an
assertion, and makes a flat total with offsetting components visible
instead of silently certifying it as constant.

## Uniform accounting surface

The certificate reads one API across the three kinds of decode state:

* `CSADecodingCache.nbytes` / `.nbytes_valid` / `.nbytes_breakdown()`
* `HCADecodingCache.nbytes` / `.nbytes_valid` / `.nbytes_breakdown()`
* `Mamba2DecodingState.nbytes` / `.nbytes_breakdown()`
* per-wrapper `state_nbytes()` on the `run_decoding` model classes
  (`SoftmaxAttnDecoding`, `KDAAttnDecoding`, `Mamba2AttnDecoding`,
  `Rwkv7AttnDecoding`, `CSAAttnDecoding`, `HCAAttnDecoding`,
  `HybridDecoding`), all returning
  `{'components', 'allocated_bytes', 'valid_bytes'}` with plain ints so
  the rows are JSON-safe under `allow_nan=False`.

The cache component names (`compressed_kv`, `indexer_kv`,
`sliding_window`, `partial_tail`, `overlap`) match
`cache_accounting.COMPONENT_SPECS` / `run_kv_cache.full_accounting`, so
the live numbers can be reconciled one-to-one with Exp 3's analytical
model. The allocated capacity follows the same geometric rule
(`next pow2 >= n_blocks`), which is what makes that reconciliation exact.

## Invariants pinned by tests

`test_decode_state_certificate.py` checks 18 properties, among them:

* the cache breakdown equals an independent walk over the cache's
  tensors AND the analytic per-component formula (fp32 bytes);
* a fresh cache retains the fixed ring buffer but has zero valid bytes
  (`allocated >= valid` always);
* KDA / Mamba-2 / RWKV-7 state bytes are **identical** after T=8 and
  T=32, while softmax scales exactly with T and CSA/HCA grow strictly
  sub-linearly (below the softmax factor);
* `bench_decoding` rows carry JSON-safe certificate fields and
  `_decode_state_certificate` classifies constant vs growing correctly
  while excluding error rows;
* the certificate is component-aware: a flat total caused by offsetting
  components is classified as growing. At prefill lengths 30/60/120 an
  HCA cache can hold 2560 B at every point while compressed KV grows
  512 → 1024 → 2048 B and the partial tail empties 1536 → 1024 → 0 B.
  Rows expose `growing_components`,
  `total_valid_flat_but_components_grow`, and the per-component series
  so the cancellation is auditable; legacy rows without a breakdown keep
  the total-only fallback.

`results_schema`'s `exp6_decoding` contract additionally rejects
malformed certificate fields on write (`decode_state_bytes` must be a
positive int, `decode_state_allocated_bytes >= decode_state_bytes`,
`decode_state_breakdown` must be an object).

## How to regenerate

```bash
DECODE_PREFILL_LENS='128,512,1024,2048' DECODE_N_TOKENS=20 \
    python3 run_decoding.py
```

The certificate table is printed at the end of the run and stored in
`results/exp6_decoding.json` (`metadata.decode_state_certificate`).
