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
  best.

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
