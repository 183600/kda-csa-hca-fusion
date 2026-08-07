# Correctness Audit — KDA/CSA/HCA fused decoder

Scope: `ops_kda.py`, `ops_csa.py`, `ops_hca.py`, `ops_fused.py`,
`ops_decoding_cache.py`, `ops_kda_backend.py`, and the full regression
suite in `run_correctness.py`.

## Method

1. Read every kernel implementation and the regression suite end-to-end.
2. Ran the entire suite: **all ~70 test functions pass** (no FAIL, no CRASH).
3. Ran independent stress probes (hand-written references sharing no code
   with the kernels) to catch bugs the suite's fixed seeds may miss:
   - KDA chunk-vs-recurrent over 30 random seeds x 5 chunk sizes x 4
     head/config combos (incl. nondivisible T): worst `max_diff ≈ 1.5e-8`.
   - CSA incremental decode-cache vs `naive_csa` on the full sequence:
     `max_diff ≈ 1.8e-7` (fp32).
   - CSA overlapped compression vs a manual block reference over 50 seeds.

## Verdict

**No new, still-present correctness bug was found.** The CHANGED/entire
code under audit is numerically sound for the configurations exercised.

Notable verified-correct points:
- `naive_chunk_kda` == `naive_recurrent_kda` (exact to ~1e-8), including
  GVA and nondivisible `T` (chunk pads then slices, validated).
- `naive_csa` vs incremental `CSADecodingCache` agree; sliding-window ring
  buffer matches an unfold-based rebuild; causal block mask is applied
  correctly; empty ring slots are zeroed so no garbage leaks.
- Fused `naive_cse`/`naive_hca` projection layouts and GQA pooling are
  internally consistent with the cache walkers and bench counts
  (`test_prefill_flops_*`, `test_kda_gate_matches_hybrid`).
- STE (`topk_columns` vs `full_softmax`) is forward-equivalent and
  backward-distinct as documented; gradient contracts verified.

## Caveats (verified, but already documented as artifacts — not new bugs)

1. Sparse index decode uses `torch.topk`; when `topk` < number of valid
   blocks and indices tie, the exact selected set / ordering can differ
   from a dense top-k of the same logits. This is explicitly described in
   the code and the suite deliberately uses "select-all"
   (`topk=100`) to sidestep it. Impact is nil at exhaustive-k; at small
   sampling top-k it only permutes among tied blocks.

2. The SW ring-buffer path stores the cumulative buffer as
   `buffer.view(B, 2*win, ...)`; the zeroing guard inserted at
   update-time prevents uninitialized `torch.empty` slots from being
   read.

Neither changes reference numerics in any continuous setting; both are
documented trade-offs rather than latent defects.

## Recommendation

No code change is required for correctness. The suite already pins the
only two known artifacts to tolerable levels (5e-5 `/T`, cache `1e-5`).