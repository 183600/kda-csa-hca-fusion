# Unit-level bootstrap inference for perplexity evaluation

`ppl_bootstrap.py` turns a held-out perplexity point estimate into an
interval and turns a two-model comparison into a paired test. It is pure
Python (no numpy/scipy), deterministic under a fixed seed, and never
touches the global RNG stream.

## Why units, not tokens

A corpus perplexity is `exp(sum(NLL) / sum(tokens))`. Resampling raw
tokens as if they were independent understates the variance of that mean:
tokens inside one sequence are correlated, so the effective sample size
is closer to the number of sequences than to the number of tokens.
`ppl_bootstrap` therefore resamples **evaluation units** (one scored
sequence/window) with replacement and re-pools the tokens of the
resampled units. A unit is fully described by the sufficient statistic
`(nll_sum, n_tokens)`.

Similarly, comparing two models by checking whether their individual
confidence intervals overlap is not a test: the dominant variance term
(corpus difficulty) is shared and cancels in the paired view.
`paired_bootstrap_ppl` draws **one** index vector per replicate and
applies it to both models, which is the correct paired statistic.

## API

```python
import ppl_bootstrap as pb

# 1. Collect units with the canonical evaluator (same single test pass,
#    no extra forwards):
report = evaluate_lm_loss(model, test_dataset, batch_size=1,
                          device=device, return_units=True)
units = report['per_unit']            # [{index, nll_sum, n_tokens}, ...]

# 2. CI for one model's corpus perplexity (resamples whole sequences):
ci = pb.bootstrap_ppl_ci(units, alpha=0.05, n_boot=2000, seed=42)
# ci['ci_perplexity'] = [lo, hi]

# 3. Paired comparison of two models scored on the SAME units:
a = evaluate_lm_loss(model_a, test_dataset, return_units=True)['per_unit']
b = evaluate_lm_loss(model_b, test_dataset, return_units=True)['per_unit']
rep = pb.paired_bootstrap_ppl(a, b, alpha=0.05, n_boot=2000, seed=42)
# rep['delta_mean_nll']        a - b  (positive => a is worse)
# rep['ci_delta_mean_nll']     percentile CI
# rep['ppl_ratio_a_over_b']    exp(delta) with rep['ci_ppl_ratio']
# rep['p_value']               two-sided centered bootstrap test
# rep['significant'], rep['worse_model'] in {'a', 'b', None}

# 4. Prefill throughput of the evaluation pass (same measured loop):
pb.throughput_summary(n_tokens=report['n_tokens'],
                      elapsed_s=elapsed, n_sequences=report['n_sequences'])
```

## Statistical contract

* **Estimand** — token-weighted corpus mean NLL / perplexity, not the
  mean of per-unit perplexities (that would over-weight short units).
* **CI** — percentile bootstrap (type-7 interpolation). Because `exp` is
  monotone, the PPL interval is exactly the exponentiated NLL interval.
* **p-value** — two-sided centered bootstrap hypothesis test (Davison &
  Hinkley 1997, §4.4): model A's per-unit NLL sums are shifted by a
  token-proportional constant so the two pooled means coincide, then the
  fraction of centered paired replicates deviating at least as far as the
  observed delta is counted (`+1` correction). Identical inputs give
  `p = 1` exactly; the resolution is `1 / (n_boot + 1)`.
* **Direction** — `delta_mean_nll = mean_nll(a) - mean_nll(b)`; positive
  means model A has higher NLL, i.e. A is worse.
* **Alignment** — paired calls refuse mismatched corpora (different unit
  counts, or different per-unit token counts): those are not paired
  samples. The check can be overridden explicitly.
* **Degenerate cases** — units with zero supervised tokens are dropped;
  with fewer than two informative units the CI degenerates and the
  p-value is reported as `None` instead of fabricated.

## Where it is used

* `train_lm_autodl.py` — the single sealed-test evaluation collects
  per-unit NLLs in the same pass and reports
  `test_report['ppl_bootstrap']` (CI for the final perplexity) plus
  `test_report['prefill_throughput']` (tokens/s of the evaluation loop).
  `--ppl_bootstrap_nboot` controls the replicate count (0 disables).
* `run_quant_lm.py` — each quantized spec is scored on the same held-out
  windows as its fp16 baseline, so `build_ptq_rows` adds a
  `paired_vs_fp16` block: per-seed paired CI / p-value for the
  quantization damage, and a summary count of cells whose damage is
  significant for **every** seed. Replicates via `QLM_PPL_BOOT`
  (default 2000).

## Caveats

* The bootstrap quantifies **corpus-sampling** uncertainty only. It does
  not model training-seed variance — use `multi_seed_stats` on per-seed
  numbers for that, and treat the two axes as composing (bootstrap the
  per-seed evaluations, then a t/permutation test across seeds).
* Percentile intervals only (no BCa), and p-values are coarse at small
  unit counts (again: resolution `1 / (n_boot + 1)`).
* Small held-out corpora (the Kaggle/AutoDL regime) make the intervals
  wide on purpose: that width is the honest report of what the split can
  support, not a defect of the method.

## Cluster-robust evaluation (near-duplicate / grouped units)

The unit bootstrap assumes the scored units are independent draws from the
corpus. A held-out split that still contains near-duplicates violates that
assumption: two edited copies of the same story are correlated, so
resampling them separately understates the corpus variance and makes the
CI (and the paired p-value) optimistic. This is the inference-side
counterpart of the data-integrity audit: detection makes the redundancy
visible, cluster resampling makes the uncertainty honest.

`near_duplicates.near_duplicate_cluster_labels(token_ids, ...)` returns one
label per document, aligned to document order:

* a singleton document keeps its own index;
* every member of a near-duplicate component shares the component's lowest
  document index;
* `candidates_truncated` is reported, so a capped search is never mistaken
  for a complete clustering.

Both bootstrap entry points accept `cluster_ids=`:

```python
from near_duplicates import near_duplicate_cluster_labels
import ppl_bootstrap as pb

clusters = near_duplicate_cluster_labels(test_token_windows)
ci = pb.bootstrap_ppl_ci(units, cluster_ids=clusters['cluster_ids'])
rep = pb.paired_bootstrap_ppl(
    units_a, units_b, cluster_ids=clusters['cluster_ids'])
```

With `cluster_ids`, each replicate draws `n_clusters` clusters **with
replacement** and pools every unit of each selected cluster. The paired
test uses the same cluster draw for both models, so the pairing is
preserved. Point estimates (`mean_nll`, `perplexity`, `delta_mean_nll`,
`ppl_ratio_a_over_b`) are unchanged; only the interval and the decision are
computed at the cluster level. The report adds:

| Field | Meaning |
|---|---|
| `n_clusters` | number of independent clusters resampled |
| `resampling_unit` | `cluster` or `evaluation_unit` |
| `cluster_design.effective_n_units` | Kish effective sample size `(Σmᵢ)² / Σmᵢ²` |
| `cluster_design.design_effect` | `n_units / effective_n_units` |
| `cluster_design.cluster_sizes` | min / max / mean / largest share |

The design effect is **size-only** and assumes perfect within-cluster
correlation. It is a conservative description of the redundancy, not an
ICC estimated from the outcome. With all-singleton labels the cluster draw
is identical to the historical unit draw, so the report is bit-identical
to `cluster_ids=None` (apart from the added labels/summary).

`train_lm_autodl.py` computes the test-split cluster labels from the
truncated `seq_len + 1` token windows with the same threshold and shingle
size as the near-duplicate audit. This is a data-only step performed
before the sealed test split is opened for its single evaluation, and the
labels are recorded under `splits.test.near_duplicate_clusters`. The final
`ppl_bootstrap` block is therefore cluster-aware whenever the audit is
enabled; skipping the near-duplicate audit (`--skip_near_dup_audit`) falls
back to the unit-level bootstrap.

### Caveats

* Cluster-aware resampling repairs dependence inside the **detected**
  near-duplicate clusters. Lexical Jaccard does not detect true paraphrase,
  so semantic redundancy remains a lower-bound problem (see
  [data-integrity.md](data-integrity.md)).
* Dependence across clusters (same author, source, template or document
  family) is not inferred from text similarity. Pass explicit `group_ids`
  as `cluster_ids` when that structure is known.
* If the candidate cap is exhausted, the labels are a lower bound and the
  recorded `candidates_truncated` flag says so. Raise the cap or report the
  interval as a lower-bound uncertainty statement.
