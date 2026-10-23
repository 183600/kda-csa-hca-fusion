# Reproducibility: code identity and provenance verification

This document describes the reproducibility contract implemented by
`kaggle_setup.capture_provenance()` and the `reproducibility.py` verifier. It
complements [backend-provenance.md](backend-provenance.md), which records
*which accelerator backend* ran; this document records *which code and
numerical runtime* produced a result and how to check it later.

## Why a commit hash is not enough

A result file that stores only `git_commit` has two reproducibility gaps:

1. **Dirty checkouts.** Two runs can share `HEAD` while executing different
   uncommitted edits. `git_dirty: true` says "not the committed code" but not
   *which* uncommitted code, so the runs cannot be distinguished after the
   fact.
2. **Untracked code.** A new helper module imported by the runner is invisible
   to `git rev-parse HEAD` and may not appear in `git status` if it is
   accidentally ignored.

The provenance block therefore adds `source_tree_sha256`, a content digest of
the working-tree code/config files. The digest includes tracked files plus
untracked-not-ignored files, and frames each repo-relative path before its
content, so a rename is a different identity even when the bytes are
unchanged. It covers `.py`, `.toml`, `.txt`, `.sh`, `.cfg`, `.ini`, `.yml`,
`.yaml` and the conventional `Dockerfile` / `Makefile` names. It is **code
identity**, not a data fingerprint: corpus versions are recorded separately
by each runner's data-version block.

The remaining provenance fields support replay and drift diagnosis:

| Field | Meaning |
|---|---|
| `git_commit`, `git_branch`, `git_describe` | committed revision |
| `git_dirty`, `git_dirty_files` | whether the worktree differed from HEAD, and how many paths |
| `source_tree_sha256`, `source_file_count`, `source_missing_count` | working-tree code identity |
| `command_line` | argv with credential-bearing flag values redacted |
| `working_directory`, `python_executable` | launch context |
| `package_versions` | exact installed versions (torch, NumPy, SciPy, einops, matplotlib, transformers, datasets, tqdm, pytest, optional backends) |
| `python_version`, `torch_version`, `cuda_version`, `numpy_version`, `scipy_version`, ... | convenience top-level copies of the runtime identity |
| `torch_num_threads`, `num_threads`, `determinism` | thread count and deterministic-algorithm / cuDNN flags in effect when provenance was captured |
| backend fields | requested vs. resolved KDA / Mamba-2 / RWKV backends (see [backend-provenance.md](backend-provenance.md)) |

`PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` remain recorded because they
cannot be changed after interpreter / cuBLAS initialization.

## Verifying a result

```bash
# Human-readable per-field report.
python reproducibility.py results/exp4_mqar.json

# Machine-readable report; one object per file.
python reproducibility.py --json results/exp4_mqar.json results/exp5_ablation.json

# Print the identity that a NEW result would record, without running one.
python reproducibility.py --identity
```

Exit codes are part of the contract:

| Code | Meaning |
|---:|---|
| `0` | every checked result matches this checkout |
| `1` | at least one code, numerical-runtime, content-integrity or pre-registration field drifted |
| `2` | provenance or a design binding is absent, unreadable, or cannot be verified |

The verifier separates three severities:

* **blocking** — `git_commit`, `source_tree_sha256`, Python, torch, NumPy and
  SciPy versions. A mismatch means the recorded numbers came from a different
  implementation or numerical stack.
* **advisory** — platform, CUDA/device name, matplotlib/einops versions and
  thread count. These differ legitimately between CPU and GPU hosts; the
  report warns without failing.
* **unverifiable** — a blocking field is missing from the current report, or
  the record carries no code identity at all. Missing evidence is never
  treated as agreement.

A dirty run that has a source digest is verifiable by content: if the digest
matches, the exact uncommitted code is present. A dirty run **without** a
source digest is reported as `unverifiable`, because the commit hash cannot
reconstruct it.

The report also carries a `design:` line for multi-seed results. The runtime
audit embeds the canonical fingerprint and a snapshot of the pre-registered
plan it was evaluated against; the verifier re-hashes the snapshot and
compares it with the current registry. A post-hoc change to the SESOI, seed
budget, required controls, analysis family, or decision rule is
`registry_drift` (exit `1`); an edited snapshot is `snapshot_mismatch` (exit
`1`); a legacy audit without a fingerprint is `unbound` (warning only).
See [preregistration-binding.md](preregistration-binding.md).

## Result content integrity

Provenance answers *which code and runtime produced the numbers*. It does not
answer whether the numbers on disk are still the ones that run wrote: a
post-processing script (or a hand edit) can change a metric without touching
the provenance block, and the provenance comparison would still pass.

Every envelope written by `results_schema.write` / `write_checked` therefore
carries a self-describing content digest:

```json
"integrity": {
  "algorithm": "sha256",
  "scope": "sha256 of the canonical JSON envelope (sorted keys, compact separators, UTF-8) with the integrity field itself removed",
  "value": "<64 hex characters>"
}
```

The digest covers every other field, including `metadata.provenance` and the
per-seed `results` rows. It is computed over **canonical JSON** (sorted keys,
compact separators, strict RFC 8259 numbers), so pretty-printing, key
reordering, or re-saving a file does not invalidate it; changing a value
does. The block is computed last, after `created_utc` is fixed, so it
identifies the exact artifact rather than only its rows.

`reproducibility.verify_record` now includes an `integrity` sub-report and
folds a mismatch into the top-level verdict:

| Integrity status | Meaning | Effect on the verdict |
|---|---|---|
| `ok` | digest matches the content | none |
| `missing` | legacy file with no digest | warning only; provenance still checked |
| `mismatch` | at least one field was edited after writing | **`drift`** (exit `1`) |
| `malformed` / `unsupported` | digest present but unreadable or uses an unknown algorithm | `unverifiable` (exit `2`) |

`results_schema.py` also exposes the check directly:

```bash
# Report and gate integrity for result files.
python results_schema.py --check-integrity results/*.json

# Same, but treat a legacy file without a digest as a failure.
python results_schema.py --require-integrity results/*.json
```

`--check-integrity` gates only the digest, so it also works for the
campaign summary (which is not an envelope and has no `results` list).
Schema validation stays a separate request via `--check` / `--check-rows`;
combine them to gate both.

Because a mismatch proves the content differs from the archived artifact,
it is reported as drift rather than as a warning. This is deliberately
stronger than the advisory platform checks: a changed metric is not a
legitimate consequence of rerunning on another host. The campaign summary
(`results/summary.json`) uses the same block, so its pass/fail list and
timings are covered too.

## Cross-source comparability

`run_budget_frontier.audit_join_sources()` joins measured rows from Exp 4,
Exp 5 and Exp 8. It now compares `source_tree_sha256` in addition to the git
commit and seed set:

| Status | Ranking allowed? | Meaning |
|---|---|---|
| `comparable` | yes | one commit, one source digest, shared seeds, consistent backends |
| `comparable_dirty_tree` | yes, with caveat | one commit and one source digest, but at least one source was dirty |
| `mixed_source_trees` | **no** | same commit, but the source digests differ: the dirty checkouts contained different code |
| `mixed_revisions` | no | different commits |
| `mixed_backends` | no | different resolved kernels |

Joined rows carry `source_git_commit`, `source_git_dirty`,
`source_tree_sha256`, `source_seeds` and `source_sha256`, so the verdict is
visible at row level rather than only in the run log. The cross-source audit
remains conservative: a dirty source with no digest yields a warning even
when its commit matches.

## Scope and limitations

* The digest identifies code/config content. It does not hash datasets,
  generated result files, PDFs or figures; use the data-version block and the
  per-file `sha256` in Exp 9 for those. The envelope `integrity` digest does
  cover a result file, but it is tamper evidence, not a signature: it detects
  accidental or naive edits, not an adversary who rewrites both the content
  and the stored digest.
* The digest is a working-tree digest. It intentionally changes when a source
  file is edited, renamed, added or deleted; reproducing a historical run
  therefore means restoring the same tree, not merely checking out the same
  commit.
* Exact reproducibility across CPU/GPU architectures is not claimed. Floating
  point reduction order can differ; `capture_provenance` records the runtime
  so such comparisons are at least attributable.
* Recording a command line is not a sandbox. Credential-bearing flags named in
  `_SECRET_ARG_FLAGS` are redacted, but secrets passed through environment
  variables or non-standard flags remain the caller's responsibility.

## Tests

`test_source_identity.py` covers the digest contract (determinism, content /
path / rename sensitivity, non-source exclusion, no-git fallback), the
commit/branch/dirty record, credential redaction, verifier statuses, and the
Exp 9 `mixed_source_trees` verdict. `test_result_integrity.py` covers the
envelope content digest (formatting neutrality, row/metadata/provenance
tampering, legacy `missing`, `malformed`/`unsupported` statuses, and the
`--check-integrity` CLI).

```bash
pytest -q test_source_identity.py test_result_integrity.py
```
