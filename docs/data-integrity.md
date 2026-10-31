# Data integrity: near-duplicate leakage and split provenance

The experiment suite distinguishes four kinds of train/eval contamination:

| Kind | Example | Where it is handled |
|---|---|---|
| Exact duplicate | the same story stored twice | `clean_lm_texts` content hash |
| Normalized duplicate | same text, different whitespace / Unicode form | `clean_lm_texts` NFKC + whitespace hash |
| Identical model window | different long documents that truncate to the same `seq_len + 1` token prefix | `clean_lm_texts` window counter, optional `dedup_windows`, `_window_overlap_audit`, `_filter_leaking_heldout` |
| **Near duplicate** | same story with a few words changed, re-numbered, or lightly paraphrased | **`near_duplicates.py` + `near_duplicate_leakage_audit`** |

The first three were already enforced. This document covers the fourth, plus
the fingerprint fix that makes the recorded data identity unambiguous.

## Why near duplicates matter

A held-out split is only an out-of-sample estimate if the evaluation
documents are not near-copies of training documents. If document A is in
train and its edited twin B is in validation/test, a model can memorise A
and score well on B. The resulting perplexity/accuracy is then a
**memorisation** measurement, not a generalisation measurement, and the
bias is in the optimistic direction. Exact and whitespace dedup cannot see
this because A and B are different strings and different hashes.

Near-duplicate detection is therefore a data-validity gate, not a
performance optimisation. It is deliberately kept separate from model
selection: the audit reads text/token content but computes no model metric,
so it can run before training while the test split is still sealed.

## Detection method

The unit of identity is the **truncated model-input window**, exactly the
`seq_len + 1` token ids that `TinyStoriesLM` feeds to the model. Two
documents are compared by the Jaccard similarity of their sets of
contiguous token `shingle_size`-grams:

```
J(A, B) = |shingles(A) ∩ shingles(B)| / |shingles(A) ∪ shingles(B)|
```

The default is `shingle_size=5` and `threshold=0.8` (four fifths of the
5-grams shared), fixed **before** looking at the corpus. The threshold is
part of the data protocol, not a knob tuned against held-out pairs.

Candidate retrieval uses MinHash + LSH:

1. each token sequence is reduced to a 64-row MinHash signature;
2. the signature is split into 16 bands of 4 rows; documents sharing a
   band become candidates (at `J=0.8` the candidate probability is
   `1 - (1 - 0.8^4)^16 ≈ 0.9997`);
3. **every candidate pair is re-scored exactly** on the sorted 64-bit
   shingle hashes. LSH only controls recall; the reported Jaccard is the
   exact set overlap (up to a negligible 64-bit hash collision).

All hashing, permutations and tie-breaks are derived from an explicit
integer seed with no Python/NumPy/torch global-RNG use. Repeated calls on
the same corpus return identical pairs and cannot perturb the training or
evaluation RNG streams.

The worst case is bounded by `max_candidate_pairs` (default 200,000;
`--near_dup_max_candidates` in the LM entry point). On a
heavily templated corpus the LSH buckets can be large; when the cap is hit
`candidates_truncated=True` is recorded and the reported pair list is a
**lower bound**. That distinction matters: "no near duplicates found" and
"the candidate budget ran out" are different scientific statements, and
the run record keeps them separate.

## Policies in the LM entry point

`train_lm_autodl.py` records three explicit policies in
`data.near_dup_policy`:

| Field | Default | Meaning |
|---|---|---|
| `pre_split_dedup` | `False` (`--dedup_near`) | drop every non-representative member of a near-duplicate cluster **before** the split; the first (corpus-order) document is kept |
| `cross_split_audit` | `True` (disable with `--skip_near_dup_audit`) | report near-duplicate pairs that straddle train/val/test |
| `strict_gate` | `False` (`--strict_near_dup`) | refuse to train when the cross-split audit finds any pair, **or when the candidate cap was exhausted** (an incomplete audit cannot certify absence) |

The audit is on by default because a leakage-safe split is otherwise only
safe against the three exact/window cases. The pre-split **drop** remains
opt-in to preserve the historical corpus composition: dropping near
duplicates changes the training set, so it must be an explicit,
reported decision. `--dedup_near` and `--strict_near_dup` compose:```bash
# Audit only: report cross-split near-twin pairs, keep the corpus as-is.
python train_lm_autodl.py

# Leakage-hardened: remove near duplicates before the split, then gate.
python train_lm_autodl.py --dedup_near --strict_near_dup
```

For a direct API call, `clean_lm_texts(..., dedup_near=True)` returns the
stats `n_near_duplicates`, `n_near_duplicate_pairs`,
`n_near_duplicate_clusters`, `largest_near_duplicate_cluster` and
`near_duplicate_candidates_truncated`.
`split_clean_texts(..., tokenizer=..., seq_len=..., near_dup_audit=True)`
adds `report['near_duplicate_overlaps']`; passing `strict_near_dup=True`
forces the audit and raises on any cross-split pair.

The recorded audit block contains:

```text
enabled, flag, n_cross_split_pairs, n_within_split_pairs,
by_split_pair, n_documents_involved, max_cross_split_jaccard,
examples, examples_truncated, n_tokenize_errors,
detector: {method, threshold, shingle_size, num_perm, band_rows, seed,
           n_documents, n_nonempty_documents, n_candidate_pairs,
           candidates_truncated, exact_verification}
```

`n_within_split_pairs` is redundancy inside one split, not leakage; it is
reported so a reader can see that the detector fired without treating
redundancy as contamination. `examples` is capped, but every count is a
full count (unless `candidates_truncated=True`).

## Stratified splitting (covariate balance)

The seeded shuffle is unbiased in expectation, but a small held-out split
can still draw an unrepresentative covariate mix. If val/test are shorter
or longer than train, the held-out loss partly measures that mismatch, and
the split-to-split variance inflates the seed budget. The split helpers now
accept a pre-specified `strata` sequence (one label per row):

```python
from kaggle_setup import split_train_val_test

strata = [...]  # one pre-specified covariate label per row
split = split_train_val_test(n, val_frac=0.1, test_frac=0.1,
                             seed=0, strata=strata)
```

`split_clean_texts(..., strata=...)` forwards the same argument, and the LM
entry point exposes it as `--stratify_by length --stratify_bins 4`.

### What the strata are

A stratum is a **covariate label fixed before the split**, not a target
label, a loss, or a metric. The built-in `length` scheme bins each
document by the number of tokens in its truncated model window. The
boundaries are fixed
fractions of the window (`cap/8, cap/4, cap/2` for four bins, where
`cap = seq_len + 1`), not corpus quantiles: a quantile boundary would
depend on the held-out rows, while a fixed boundary is a property of the
model geometry that can be declared before looking at the data. Documents
that fail tokenization go to `len_bin_unknown` instead of aborting the
split.

### Allocation rule

Inside each stratum the rows are shuffled with a dedicated RNG seeded by
`SHA-256(seed, stratum_label)` and cut with Hamilton rounding at the global
train/val/test fractions. The RNG is local, so the split still cannot
perturb Python/NumPy/torch training streams, and the result is independent
of row order. If a requested split would remain empty (possible when every
stratum is smaller than the number of splits), one row is moved from the
largest bucket, and the audit reports the affected stratum/split cell as
empty instead of hiding the imbalance.

`audit_split(..., strata=...)` adds a descriptive block:

```text
stratified, strata: {
  n_strata, stratum_sizes, counts: {label: {train, val, test}},
  global_fractions, max_abs_fraction_deviation,
  n_missing_stratum_split_cells, note
}
```

It is deliberately **not** a pass/fail leakage gate: a stratum with one row
cannot populate three splits, so the audit reports the imbalance instead of
rejecting an otherwise valid split. `ok` continues to mean the structural
invariants (disjoint, complete, non-empty) hold. The post-split
window-leak filter can remove held-out rows after this audit; the LM run
record therefore also stores `split_protocol.post_filter_strata_balance`,
the same balance block computed on the rows that are actually evaluated.

### Reporting and limitations

Report `stratified_by`, the bin boundaries/counts and the per-stratum split
counts next to the split seed. A stratified run and an unstratified run are
different data protocols; compare them as a protocol sensitivity check, not
as two views of one number. Stratification balances only the chosen
covariate — it does not fix a domain shift in vocabulary, style or topic
(use the unigram audit above), and it cannot repair near-duplicate leakage.
Choose the covariate and bins before inspecting held-out metrics; changing
them after seeing a shift or a score is a selection effect. `strata` is
mutually exclusive with `group_ids` and `chronological` in the current
implementation, because those modes already impose a structural split and
mixing the rules would make the audit ambiguous.

## Fingerprint framing

`TinyStoriesLM.data_fingerprint` is the corpus version id embedded in the
run record, checkpoints and manifest. It hashes at most 64 evenly spaced
texts, but each text is now framed with its probe index and **byte
length** before hashing. Without that framing, concatenating raw strings
made the fingerprint ambiguous at text boundaries: `["ab", "c"]` and
`["a", "bc"]` produced the same byte stream and therefore the same data
id even though they are different corpora. The index also makes a
reordered probe distinguishable.

This is a fingerprint, not a full-corpus digest: it is intentionally
bounded so launching a run does not hash gigabytes. The split sizes, the
cleaning stats and the per-split near-duplicate audit are stored next to
it to make a swapped or partially stale split visible.

## Recommended reporting

For a paper/report, state:

* the cleaning configuration (exact/normalized/window/near policies and
  threshold);
* whether `--dedup_near` was used and, if so, how many documents it
  removed (`cleaning.n_near_duplicates`);
* the cross-split audit result (`near_duplicate_audit`), including
  `candidates_truncated`;
* if the threshold is varied as a sensitivity analysis, report both runs.
  Do **not** choose the threshold that makes the held-out number look
  best;
* whether the split was stratified (`--stratify_by`), the pre-specified
  bins/boundaries and the per-stratum counts from `split_audit.strata`;
* whether held-out inference was cluster-aware. `ppl_bootstrap` accepts
  `cluster_ids` so near-duplicate documents are resampled as one unit
  instead of independent samples; report `n_clusters`, the design effect
  and `candidates_truncated` from the final bootstrap block. See
  [ppl-bootstrap.md](ppl-bootstrap.md#cluster-robust-evaluation-near-duplicate--grouped-units).

Near-duplicate dedup changes the evaluation population. A run with
`--dedup_near` and a run without it are different experiments; compare
them as a data-protocol sensitivity check, not as two views of the same
number.

## Limitations

* Token n-gram Jaccard is a lexical similarity. A true paraphrase with
  almost no shared 5-grams is not detected. The audit is a lower bound on
  semantic duplication, not a semantic deduplicator.
* The detector compares the truncated window only. Two long documents
  that diverge before `seq_len` are not near duplicates from the model's
  point of view, because the model never sees the shared tail.
* Candidate retrieval is a probabilistic recall aid. The exact second
  pass removes false positives, but the LSH candidate set can miss a pair
  whose true Jaccard is just below the banding sweet spot. At the default
  `J=0.8`, the design candidate probability is ~0.9997; lower thresholds
  need more bands/rows.
* The candidate cap is a real truncation. On a corpus with a shared
  template, raise it or report the audit as a lower bound.
* The audit does not repair a split and does not impute missing data. It
  only makes the leakage visible; the remedy is to drop near duplicates
  before splitting, or to keep them and report the contamination.

## Held-out vocabulary / domain shift (token unigrams)

Near-duplicate leakage is an *overlap* failure. The complementary
*exchangeability* failure is a held-out split whose token distribution is
not representative of train: a different topic, source, language variety or
annotation template. Then held-out perplexity partly measures domain
mismatch, and a comparison between two models can flip because one happened
to fit the shifted vocabulary better. Length statistics cannot see this —
two splits can have identical length histograms and disjoint vocabularies.

`train_lm_autodl.token_distribution_audit` compares the token-unigram
distributions of the train probe and a held-out probe. It is fitted on
**train only**, is deterministic and RNG-neutral, and computes no model
metric, so it can run before the sealed test split is opened. In the LM
entry point the `test_vs_train` block is filled only after that split's
single final evaluation; `run_quant_lm.build_corpus` records both val and
test alongside its existing offline length diagnostics.

### Metrics

On the same bounded, evenly spaced document probe used by the length
diagnostics:

| Field | Meaning |
|---|---|
| `unseen_token_mass` | fraction of held-out tokens whose id never occurs in the train probe |
| `unseen_type_fraction` | fraction of held-out token types absent from train |
| `js_divergence_bits` | Jensen-Shannon divergence (base 2) between the unigram distributions over the union of observed ids |
| `total_variation` | half the L1 distance between the same two distributions |
| `unseen_examples` | up to `max_examples` most frequent unseen ids, with a best-effort decoded form |

### Why the raw number is not the decision

A plug-in comparison always finds *some* unseen held-out tokens: with finite
samples, even a perfectly i.i.d. split contains rare words that the train
sample missed, and the amount grows as the fit sample shrinks. Reporting
that raw count as "distribution shift" would flag every split.

The audit therefore computes the **same metrics on two disjoint document
folds of the train probe**, preferring an evaluation fold sized to the
held-out probe and falling back to a balanced train half-split when two
matched folds are not available. That is an empirical i.i.d. null built
from the *same* corpus, tokenizer and probe mechanism. (If the train probe
is too small for either construction, no null is reported.) The decision
uses

```
excess = max(0, observed - null_reference)
```

and sets `flag=True` only when the excess unseen-token mass exceeds
`unseen_mass_threshold` (default 1%) or the excess JS exceeds
`js_threshold` (default 0.05 bits). The defaults are pre-specified warning
levels, not gates: no row is dropped and no model-selection decision is
derived from them. If the train probe is too small for two disjoint folds,
`null_reference` is `None` and `flag` is `None` — the raw metrics are still
recorded, but the audit refuses to call them a shift.

The null is deliberately conservative: the observed fit uses the entire
train probe while the null fit uses one fold, so finite-sample unseen mass
in the null is at least as large as in the observed comparison. A moderate
domain mixture still clears the bar (a 10% foreign-vocabulary mixture is
flagged in the regression suite), which is the intended sensitivity.

### Record layout

In the LM run record (`train_lm_autodl.py`):

```text
distribution_shift.token_distribution = {
  fit_on: "train probe only",
  null_calibration: "within-train document folds",
  val_vs_train: {
    available, reference: "train", fit_on,
    train_probe: {n_texts, n_probe_requested, n_documents_probed,
                  n_documents_scored, n_empty_documents,
                  n_tokenize_errors, n_tokens},
    heldout_probe: {...same...},
    observed: {n_fit_documents, n_eval_documents, n_fit_tokens,
               n_eval_tokens, n_fit_types, n_eval_types, n_unseen_types,
               unseen_type_fraction, unseen_token_mass,
               js_divergence_bits, total_variation,
               unseen_examples, examples_truncated},
    null_reference: {...same metrics, no examples...} | null,
    excess: {unseen_token_mass, js_divergence_bits, definition} | null,
    flag: true | false | null,
    flag_reasons: [...], js_threshold, unseen_mass_threshold,
    seed, max_examples, no_rows_dropped: true
  },
  test_vs_train: {...same, filled after the sealed test opens...}
}
```

`run_quant_lm.build_corpus` records the same blocks under
`distribution_diagnostics.token_distribution_shift` for its val/test splits.

### Recommended reporting

Report the flag, the observed excess, and the probe sizes. A flagged split
should be described as a distribution-shift caveat on the held-out number,
not silently re-split after seeing the result — changing the split after
looking at the metric is a selection effect. The honest remedies, fixed
before the next run, are: stratify the split on the shifted variable,
restrict claims to the vocabulary both splits cover, or report the shift as
part of the result.

### Limitations

* Unigram overlap is lexical. A topic shift expressed with the same common
  words is not detected; rare-word and domain shifts are.
* Tokenization is the model's tokenizer, so subword vocabularies hide some
  word-level novelty (a new word composed of known subwords is not unseen).
* The probe is bounded and evenly spaced; a shift confined to a small part
  of the corpus can be missed. `n_probe` is recorded.
* The null calibrates finite-sample bias, not every possible confound. It
  is a warning diagnostic, not a formal test with a p-value.
* The audit reads text only; it cannot tell whether a shift is harmful for a
  particular model, only that the held-out population differs.
