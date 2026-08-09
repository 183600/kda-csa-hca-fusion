# Correctness Audit — KDA/CSA/HCA fused decoder

Scope: `ops_kda.py`, `ops_csa.py`, `ops_hca.py`, `ops_fused.py`,
`ops_decoding_cache.py`, `ops_kda_backend.py`, and the full regression
suite in `run_correctness.py`.

## Method

1. Read every kernel implementation and the regression suite end-to-end.
2. Ran the entire suite: **all ~70 test functions pass** (no FAIL, no CRASH).
   Re-confirmed on 2026-08-09 (torch 2.6.0+cpu, Python 3.11.15):
   **251/251 checks pass**, `results/exp1_correctness.json` regenerated.
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

## Paper cross-check (2026-08-09, against committed PDFs)

Re-verified every operator against the two source papers added to the repo
(`2510.26692v2.pdf` = Kimi Linear / KDA, `2606.19348v1.pdf` = DeepSeek-V4 /
CSA + HCA). All formulas match; no discrepancy found.

| Operator | Paper formula | Code location | Match |
|---|---|---|---|
| KDA state update | Eq. 1: `S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`, `o_t = S_tᵀ q_t`, fine-grained diagonalized gate `α_t = exp(g)` | `ops_kda.py:248-251` | ✓ |
| CSA projections | Eq. 9-10: `C_a/C_b = H·W_aKV/W_bKV`, `Z_a/Z_b = H·W_aZ/W_bZ` | `ops_csa.py:815-842` | ✓ |
| CSA overlapped compression | Eq. 11-12: block i fuses a-branch block i + b-branch block i−1, softmax over 2m, `-inf` pad at i=0 | `ops_csa.py:252-278` | ✓ |
| CSA lightning indexer | Eq. 13-17: `c_Q=H·W_DQ`, `q_I=c_Q·W_IUQ`, `w_I=H·W_w`, score `Σ_h w·ReLU(q_h·K_IComp)`, top-k | `ops_csa.py:849-874, 503-508` | ✓ |
| CSA shared-KV MQA | Eq. 18-19: queries share latent `c_Q`, MQA over selected compressed entries | `ops_csa.py:897, 1074-1135` | ✓ |
| CSA/HCA sink | Eq. 27: `exp(sink_h)` added to softmax *denominator only*, no value term | `ops_csa.py:1078-1118`, `ops_hca.py:148-173` | ✓ |
| CSA/HCA sliding window | §2.3.3: uncompressed recent-`win` branch concatenated with compressed | `ops_csa.py:1138-1187`, `ops_hca.py:179-194` | ✓ |
| HCA compression | Eq. 22-23: single branch, **no overlap**, softmax over m′ | `ops_csa.py:203-208`, `ops_hca.py:132-135` | ✓ |
| HCA dense MQA | Eq. 24-26: low-rank `c_Q`, dense MQA over all compressed blocks | `ops_hca.py:140-176` | ✓ |

Deliberate deviations, all documented in code/README (not bugs):
- L2-normalized q / C_comp with `scale=1.0` instead of the paper's RMSNorm +
  partial RoPE + FP8/FP4 (reference simplification, README "Limitations").
- Grouped output projection is present in `ops_fused` (paper §2.3.1); the raw
  per-head output `[B,T,nh·c]` is returned to it by `naive_csa`/`naive_hca`.
- STE (`topk_columns`/`full_softmax`) is the reference stand-in for the
  paper's contrastive indexer auxiliary loss (`aux_contrastive` reserved).
- Indexer `ReLU` head-weighted scoring matches Eq. 16 exactly (scale is the
  only addition, auto-selected 1.0 under `normalize_qk`).

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