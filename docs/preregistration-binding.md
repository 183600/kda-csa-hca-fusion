# Pre-registration binding: from a plan registry to a checkable result

`experiment_design.py` keeps the confirmatory design (SESOI, seed budget,
required controls, analysis family, decision/stopping rules) in a committed
registry, and `audit_run()` labels each completed run
`confirmatory` / `screening` / `incomplete_attrition` / ... . That is
necessary but not sufficient for a pre-registration claim: the registry is
code, so it can be edited after the run. A result file that only says
`experiment_design.status = "confirmatory"` cannot show *which* plan
definition produced that label.

The binding closes that gap with two additions.

## 1. A canonical plan fingerprint

`experiment_design.plan_sha256(plan_or_id)` hashes the canonical JSON of one
registered plan (sorted keys, compact separators, strict numbers, UTF-8).
Key order and pretty-printing do not matter; any semantic change to the plan
does. The scope string stored next to every digest is
`PLAN_FINGERPRINT_SCOPE` and is versioned: a change to the hashing contract
requires a new scope, and a record with an unrecognized scope is reported as
unverifiable rather than compared against a different algorithm.

## 2. The audit carries the fingerprint and a snapshot

`audit_run()` now returns, in addition to its verdict:

| Field | Meaning |
|---|---|
| `plan_id` | the registered plan that was audited |
| `plan_sha256` | canonical digest of that plan definition |
| `plan_fingerprint_scope` | the documented digest contract |
| `plan_definition` | the full plan snapshot the verdict was computed from |

Every runner that already embeds the audit (`run_quality`, `run_ablation`
matrix/ratio, `run_mechanism_ablation`) therefore writes the binding into its
result file automatically. The ratio sweep's sidecar
`provenance.experiment_design` block is supported as well as the usual
`metadata.experiment_design`.

The prospective manifest written by `run_all.py` before the experiments now
also records each row's `plan_sha256` and, when provenance capture succeeds,
the `metadata.provenance` block. That binds the pre-registration itself to
the working tree and runtime that wrote it. `experiment_design` stays
stdlib-only: provenance is an explicit optional argument, not an internal
import, so planning still works before torch is installed.

## Verification

`reproducibility.compare_design_binding(record)` extracts the audit, re-hashes
the snapshot, and compares the recorded digest with the current registry:

| Status | Meaning | Effect on `verify_record` |
|---|---|---|
| `match` | snapshot and current registry both equal the recorded digest | none |
| `snapshot_mismatch` | the audit's snapshot does not hash to `plan_sha256` | **drift** |
| `registry_drift` | the current registered plan differs from the audited one | **drift** |
| `unbound` | legacy audit with no valid `plan_sha256` | warning only |
| `missing` | no design audit in the record | warning only |
| `unknown_plan` | the audited plan id is no longer registered | unverifiable |
| `registry_unavailable` / `malformed` | the registry/hash contract cannot resolve the record | unverifiable |

`python reproducibility.py results/exp4_mqar.json` prints a `design:` line.
Exit codes are unchanged: `1` for any drift (including registry drift),
`2` when provenance or the design binding cannot be verified, `0` otherwise.
Legacy results remain readable: they are reported as `unbound`/`missing` with
a warning instead of being retroactively declared invalid.

## What this does and does not prove

It proves that the plan snapshot embedded in a result hashes to the recorded
digest, and that the *current* registry still produces that digest. It does
not prove the registry was frozen before data collection: that ordering comes
from the committed history and from `run_all.py` writing the manifest before
the experiments. The snapshot makes the claim auditable without requiring the
historical checkout, which is the practical improvement over a bare status
string.

The binding covers the *analysis plan* (SESOI, seed budget, controls, family,
decision rule). It does not cover the numerical runtime or the working-tree
source content; those are recorded separately by `capture_provenance()` and
checked by the code/runtime layers of the same verifier. See
[reproducibility.md](reproducibility.md) and
[experiment-design.md](experiment-design.md).

## Tests

`test_design_binding.py` covers canonicalization and content sensitivity, the
snapshot embedded by every registered plan, manifest provenance, both result
layouts, tampered snapshots, post-hoc registry changes, legacy `unbound`
audits, unknown plans, and scope changes.

```bash
pytest -q test_design_binding.py
```
