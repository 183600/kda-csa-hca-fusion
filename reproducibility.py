#!/usr/bin/env python3
"""Verify that a recorded result was produced by the current code/environment.

Every experiment embeds ``capture_provenance()`` in its result envelope. This
module turns that block into an actionable check instead of a passive record:

* **Code identity** — the recorded ``git_commit`` must match, and when a
  ``source_tree_sha256`` is present the working-tree content must match too.
  The digest is what makes a dirty-checkout run verifiable: two runs at the
  same commit but with different uncommitted edits are different experiments.
* **Runtime identity** — Python, torch, NumPy and SciPy versions must match.
  A different numerical stack can change the last digits of a metric; the
  report distinguishes that from a code change.
* **Advisory identity** — platform, CUDA/device, plotting-library version and
  thread count are reported as warnings rather than hard failures (they are
  expected to differ across CPU/GPU reruns).
* **Result integrity** — when the file carries an ``integrity`` digest
  (written by ``results_schema``), the verifier recomputes it over the
  canonical payload. A mismatch means a number, seed, or metadata field was
  edited after the run, which is a different (and more serious) problem than
  environment drift. Legacy files without a digest are still checked for
  provenance and reported as ``missing`` integrity.
* **Pre-registration binding** — multi-seed results embed the plan
  fingerprint and the plan snapshot that ``audit_run`` evaluated them
  against. The verifier re-hashes the snapshot and compares it with the
  current registry. A changed SESOI, seed budget, required control, or
  decision rule after the run is reported as ``registry_drift``; an edited
  snapshot is ``snapshot_mismatch``. Legacy audits without a fingerprint are
  reported as ``unbound`` and remain verifiable (with a warning), so the
  check is backward compatible.

Usage::

    # Check one or more result files against the current checkout.
    python reproducibility.py results/exp4_mqar.json

    # Machine-readable report; exit code follows the same contract.
    python reproducibility.py --json results/*.json

    # Print the identity of the current working tree (what a new result
    # would record) without running an experiment.
    python reproducibility.py --identity

Exit codes: ``0`` all checked results reproducible; ``1`` at least one code,
runtime, content-integrity or pre-registration drift; ``2`` provenance or a
design binding was absent/unverifiable.

The comparison is intentionally conservative: missing evidence is reported
as unverifiable rather than treated as agreement.
"""
from __future__ import annotations

import argparse
import json
import sys

from kaggle_setup import capture_provenance, capture_source_identity

#: Fields that identify the code that executed.
CODE_FIELDS = ('git_commit', 'source_tree_sha256')
#: Package versions that identify the numerical runtime.
RUNTIME_PACKAGES = ('python', 'torch', 'numpy', 'scipy')
#: Reported as warnings; CPU/GPU reruns legitimately differ here.
ADVISORY_FIELDS = ('git_dirty', 'platform', 'cuda_version', 'device_name',
                   'matplotlib_version', 'einops_version',
                   'torch_num_threads')

_PROVENANCE_MARKERS = (
    'git_commit', 'source_tree_sha256', 'torch_version', 'python_version',
    'captured_at_utc',
)


def _looks_like_provenance(value) -> bool:
    return isinstance(value, dict) and any(k in value for k in _PROVENANCE_MARKERS)


def extract_provenance(record) -> dict:
    """Find the provenance block in a result payload.

    Supports the repository's envelopes: top-level ``provenance``, Exp 4/5/8
    ``metadata.provenance``, ``run_all``'s summary ``env`` block, and a bare
    provenance dict. Returns ``{}`` when no recognizable block exists.
    """
    if not isinstance(record, dict):
        return {}
    for key in ('provenance', 'env'):
        value = record.get(key)
        if _looks_like_provenance(value):
            return value
    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        value = metadata.get('provenance')
        if _looks_like_provenance(value):
            return value
    if _looks_like_provenance(record):
        return record
    return {}


def _version_of(provenance: dict, name: str):
    """Read ``<name>_version`` or ``package_versions[name]`` from a block."""
    value = provenance.get(f'{name}_version')
    if value is None:
        versions = provenance.get('package_versions')
        if isinstance(versions, dict):
            value = versions.get(name)
    return value


def _same_value(left, right) -> bool:
    """JSON-scalar equality that treats ``None`` as a value, not a wildcard."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        return left == right
    return str(left) == str(right)


def _check(field: str, recorded, current, *, severity: str) -> dict:
    if recorded is None and current is None:
        # Both sides report the same absence (e.g. scipy not installed).
        status = 'match'
    elif recorded is None:
        status = 'not_recorded'
    elif current is None:
        status = 'missing_current'
    elif _same_value(recorded, current):
        status = 'match'
    else:
        status = 'mismatch'
    return {
        'field': field,
        'recorded': recorded,
        'current': current,
        'status': status,
        'severity': severity,
        'blocking': severity == 'blocking',
    }


def _check_version(recorded_block: dict, current_block: dict, name: str,
                   *, severity: str) -> dict:
    recorded = _version_of(recorded_block, name)
    current = _version_of(current_block, name)
    return _check(f'{name}_version', recorded, current, severity=severity)


def compare_provenance(recorded, current=None) -> dict:
    """Compare a recorded provenance block with the current environment.

    Returns a strict-JSON report. ``status`` is one of:

    * ``reproducible`` — every blocking field matches (advisory differences
      may still be present and are listed as warnings);
    * ``drift`` — at least one code/runtime field differs from the record;
    * ``unverifiable`` — the record lacks identity evidence, or a blocking
      field cannot be read from the current environment;
    * ``no_provenance`` — ``recorded`` has no recognizable provenance block.

    ``current`` defaults to a fresh :func:`capture_provenance`.
    """
    recorded = recorded if isinstance(recorded, dict) else {}
    if not _looks_like_provenance(recorded):
        return {
            'status': 'no_provenance',
            'reproducible': False,
            'n_blocking_mismatches': 0,
            'n_blocking_missing': 0,
            'n_advisory_mismatches': 0,
            'checks': {},
            'warnings': ['no provenance block was found in the record'],
        }
    if current is None:
        current = capture_provenance()
    current = current if isinstance(current, dict) else {}

    checks: dict[str, dict] = {}
    for field in CODE_FIELDS:
        checks[field] = _check(field, recorded.get(field),
                               current.get(field), severity='blocking')
    for package in RUNTIME_PACKAGES:
        checks[f'{package}_version'] = _check_version(
            recorded, current, package, severity='blocking')
    for field in ADVISORY_FIELDS:
        if field.endswith('_version'):
            checks[field] = _check_version(
                recorded, current, field.removesuffix('_version'),
                severity='advisory')
        else:
            checks[field] = _check(field, recorded.get(field),
                                   current.get(field), severity='advisory')

    warnings = []
    for name, check in checks.items():
        if check['status'] == 'mismatch':
            if check['severity'] == 'blocking':
                warnings.append(f'{name} differs from the recorded run')
            else:
                warnings.append(
                    f'{name} differs (advisory; expected across hosts)')
        elif check['status'] == 'missing_current':
            warnings.append(
                f'{name} is missing from the current environment report')
        elif check['status'] == 'not_recorded' and check['severity'] == 'blocking':
            warnings.append(
                f'{name} was not recorded; this part of the identity cannot '
                'be verified')

    blocking_mismatch = sum(
        1 for c in checks.values()
        if c['severity'] == 'blocking' and c['status'] == 'mismatch')
    blocking_missing = sum(
        1 for c in checks.values()
        if c['severity'] == 'blocking' and c['status'] == 'missing_current')
    # Python/torch/NumPy identify the numerical stack. A legacy record that
    # omits one of them cannot be called reproducible, while an omitted
    # optional SciPy (the repository has an exact fallback path) is only a
    # warning.
    required_runtime_missing = [
        f'{package}_version' for package in ('python', 'torch', 'numpy')
        if checks[f'{package}_version']['status'] == 'not_recorded']
    if required_runtime_missing:
        warnings.append(
            'the record omits required runtime version(s): '
            + ', '.join(required_runtime_missing))
    advisory_mismatch = sum(
        1 for c in checks.values()
        if c['severity'] == 'advisory' and c['status'] == 'mismatch')

    # A dirty run without a source digest cannot be tied to exact code. The
    # commit hash alone does not identify uncommitted edits.
    dirty_without_digest = (
        recorded.get('git_dirty') is True
        and not isinstance(recorded.get('source_tree_sha256'), str))
    no_code_identity = (
        not recorded.get('git_commit')
        and not isinstance(recorded.get('source_tree_sha256'), str))
    if no_code_identity:
        warnings.append('the record carries no git commit or source digest')
    if dirty_without_digest:
        warnings.append(
            'the run was recorded from a dirty tree without a source digest; '
            'the exact uncommitted code cannot be recovered')

    if blocking_mismatch:
        status = 'drift'
    elif (blocking_missing or required_runtime_missing
          or no_code_identity or dirty_without_digest):
        status = 'unverifiable'
    else:
        status = 'reproducible'

    return {
        'status': status,
        'reproducible': status == 'reproducible',
        'n_blocking_mismatches': blocking_mismatch,
        'n_blocking_missing': blocking_missing,
        'n_required_runtime_missing': len(required_runtime_missing),
        'n_advisory_mismatches': advisory_mismatch,
        'checks': checks,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# Pre-registration binding
# ---------------------------------------------------------------------------

#: Keys that identify a runtime design audit (``experiment_design.audit_run``)
#: rather than an arbitrary metadata mapping that happens to contain
#: ``plan_id``.
_DESIGN_AUDIT_MARKERS = ('plan_sha256', 'plan_definition', 'planning',
                         'checks')


def extract_design_audit(record) -> dict:
    """Find the pre-registered design audit embedded in a result payload.

    Runners attach ``experiment_design.audit_run(...)`` under
    ``metadata.experiment_design``; the ratio sweep also writes it into its
    sidecar ``provenance.experiment_design`` block. Both layouts are
    supported, plus a top-level block for ad-hoc records. Returns ``{}``
    when no recognizable audit exists (a legacy result written before the
    binding existed).
    """
    if not isinstance(record, dict):
        return {}
    candidates = []
    metadata = record.get('metadata')
    if isinstance(metadata, dict):
        candidates.append(metadata.get('experiment_design'))
    candidates.append(record.get('experiment_design'))
    provenance = record.get('provenance')
    if isinstance(provenance, dict):
        candidates.append(provenance.get('experiment_design'))
    for value in candidates:
        if isinstance(value, dict) and 'plan_id' in value and any(
                marker in value for marker in _DESIGN_AUDIT_MARKERS):
            return value
    return {}


_HEX_DIGEST_LEN = 64


def _is_sha256_hex(value) -> bool:
    if not isinstance(value, str) or len(value) != _HEX_DIGEST_LEN:
        return False
    return all(ch in '0123456789abcdef' for ch in value.lower())


def compare_design_binding(record) -> dict:
    """Check the run's embedded plan fingerprint against the registry.

    A runtime audit records the plan it was evaluated against. The audit
    carries both the fingerprint and a snapshot of the plan definition, so
    this check can distinguish:

    * ``match`` — the recorded plan is byte-identical (canonically) to the
      currently registered plan;
    * ``registry_drift`` — the result was audited against a DIFFERENT plan
      definition than the one in this checkout (e.g. the SESOI, seed
      budget, required controls or decision rule changed after the run);
    * ``snapshot_mismatch`` — the audit's own plan snapshot does not hash to
      its recorded ``plan_sha256``, so the audit block was edited after the
      fact;
    * ``unbound`` / ``missing`` — a legacy audit with no fingerprint /
      no audit at all. These are warnings only: historical files predate
      the binding and cannot be retroactively bound;
    * ``unknown_plan`` / ``registry_unavailable`` / ``malformed`` — the
      registry cannot currently resolve the plan, so the binding is
      unverifiable rather than falsely reported as matching.

    Drift matters scientifically: a result labelled ``confirmatory`` under
    one SESOI/family is not the pre-registered experiment if the registry's
    SESOI/family changed later. The audit snapshot preserves the original
    definition even when the current registry no longer matches.
    """
    audit = extract_design_audit(record)
    report = {
        'status': 'missing',
        'plan_id': None,
        'recorded_sha256': None,
        'snapshot_sha256': None,
        'current_sha256': None,
        'scope': None,
        'has_snapshot': False,
        'warnings': [],
        'note': ('the design fingerprint binds a result to the exact '
                 'pre-registered plan definition it was audited against'),
    }
    if not audit:
        report['warnings'].append(
            'no pre-registered design audit was found (legacy result); the '
            'run cannot be tied to a registered plan')
        return report

    plan_id = audit.get('plan_id')
    recorded = audit.get('plan_sha256')
    snapshot = audit.get('plan_definition')
    scope = audit.get('plan_fingerprint_scope')
    report['plan_id'] = plan_id if isinstance(plan_id, str) else None
    report['recorded_sha256'] = (
        recorded if isinstance(recorded, str) else None)
    report['has_snapshot'] = isinstance(snapshot, dict)
    report['scope'] = scope if isinstance(scope, str) else None

    # Import here so a result file can still be checked for provenance when
    # the planning module is unavailable (e.g. a partial install).
    try:
        from experiment_design import (PLANS, PLAN_FINGERPRINT_SCOPE,
                                        plan_sha256)
    except Exception as exc:  # pragma: no cover - defensive import guard
        report['status'] = 'registry_unavailable'
        report['warnings'].append(
            f'the pre-registered design registry is unavailable '
            f'({type(exc).__name__}: {exc}); the binding cannot be checked')
        return report

    if not isinstance(plan_id, str) or plan_id not in PLANS:
        report['status'] = 'unknown_plan'
        report['warnings'].append(
            f'the audited plan {plan_id!r} is not registered in this '
            'checkout; the run cannot be tied to a current plan')
        return report

    if recorded is None:
        report['status'] = 'unbound'
        report['warnings'].append(
            'the design audit carries no plan_sha256 (legacy result); the '
            'pre-registration cannot be verified')
        return report
    if not _is_sha256_hex(recorded):
        report['status'] = 'malformed'
        report['warnings'].append(
            'the design audit carries a malformed plan_sha256; the '
            'pre-registration cannot be verified')
        return report
    recorded = recorded.lower()
    report['recorded_sha256'] = recorded

    if scope is not None and scope != PLAN_FINGERPRINT_SCOPE:
        report['status'] = 'malformed'
        report['warnings'].append(
            'the recorded plan fingerprint uses an unrecognized scope; the '
            'digest algorithm may have changed, so the binding is not '
            'comparable')
        return report

    if isinstance(snapshot, dict):
        try:
            report['snapshot_sha256'] = plan_sha256(snapshot)
        except Exception as exc:
            report['status'] = 'malformed'
            report['warnings'].append(
                f'the embedded plan snapshot is not hashable '
                f'({type(exc).__name__}: {exc})')
            return report
        if report['snapshot_sha256'] != recorded:
            report['status'] = 'snapshot_mismatch'
            report['warnings'].append(
                'the embedded plan snapshot does not hash to the recorded '
                'plan_sha256: the audit block was edited after the run')
            return report

    try:
        report['current_sha256'] = plan_sha256(plan_id)
    except Exception as exc:
        report['status'] = 'registry_unavailable'
        report['warnings'].append(
            f'the registered plan could not be hashed '
            f'({type(exc).__name__}: {exc})')
        return report

    if report['current_sha256'] != recorded:
        report['status'] = 'registry_drift'
        report['warnings'].append(
            'the registered plan definition changed after this run: the '
            'recorded SESOI/seed budget/controls/decision rule no longer '
            'matches the current registry')
    else:
        report['status'] = 'match'
    return report


def verify_record(record, current=None) -> dict:
    """Extract the provenance block from ``record`` and compare it.

    The report also carries an ``integrity`` sub-report and a ``design``
    sub-report. A digest mismatch is treated as ``drift`` even when the code
    and runtime match: the file no longer contains the numbers the recorded
    run wrote. A design-fingerprint mismatch is likewise drift, because the
    result is no longer tied to the registered analysis plan. A missing
    digest or a legacy unbound audit is a warning only, while an
    unreadable/unsupported digest or an unresolvable plan degrades a
    would-be ``reproducible`` verdict to ``unverifiable``.
    """
    report = compare_provenance(extract_provenance(record), current=current)
    try:
        from results_schema import verify_integrity
        integrity = verify_integrity(record)
    except Exception as exc:  # pragma: no cover - defensive import guard
        integrity = {
            'status': 'unavailable',
            'algorithm': None,
            'recorded': None,
            'observed': None,
            'scope': None,
            'error': f'{type(exc).__name__}: {exc}',
        }
    report['integrity'] = integrity
    status = integrity.get('status')
    if status == 'mismatch':
        report['status'] = 'drift'
        report['reproducible'] = False
        report['n_integrity_mismatches'] = 1
        report['warnings'].append(
            'the result content digest does not match: at least one field '
            'was edited after the file was written')
    elif status in ('malformed', 'unsupported', 'unavailable'):
        report['n_integrity_mismatches'] = 0
        if report['status'] == 'reproducible':
            report['status'] = 'unverifiable'
            report['reproducible'] = False
        report['warnings'].append(
            f'the result integrity digest is {status}; the file content '
            'cannot be verified')
    elif status == 'missing':
        report['n_integrity_mismatches'] = 0
        report['warnings'].append(
            'the record carries no content digest (legacy file); its '
            'numbers cannot be checked for post-hoc edits')
    else:
        report['n_integrity_mismatches'] = 0

    design = compare_design_binding(record)
    report['design'] = design
    report['n_design_mismatches'] = 0
    if design['status'] in ('registry_drift', 'snapshot_mismatch'):
        report['n_design_mismatches'] = 1
        report['status'] = 'drift'
        report['reproducible'] = False
        report['warnings'].append(
            'the result is not bound to the pre-registered analysis plan: '
            'the recorded plan fingerprint differs from the current '
            'registry or from the audit snapshot')
    elif design['status'] in ('unverifiable', 'unknown_plan',
                              'registry_unavailable', 'malformed'):
        if report['status'] == 'reproducible':
            report['status'] = 'unverifiable'
            report['reproducible'] = False
        report['warnings'].append(
            f'the pre-registered design binding is {design["status"]}; the '
            'plan behind this result cannot be verified')
    else:
        # ``missing`` / ``unbound`` are legacy-only states and must not
        # invalidate otherwise-verifiable historical results.
        report['warnings'].extend(design.get('warnings') or [])
    return report


def _format_report(path: str, report: dict) -> str:
    lines = [f'{path}: {report["status"].upper()}']
    integrity = report.get('integrity')
    if isinstance(integrity, dict):
        lines.append(f'  integrity: {integrity.get("status", "unknown")}')
    design = report.get('design')
    if isinstance(design, dict):
        plan = design.get('plan_id') or '-'
        lines.append(
            f'  design:    {design.get("status", "unknown")} (plan={plan})')
    for name, check in report.get('checks', {}).items():
        if check['status'] == 'match':
            continue
        lines.append(
            f'  {name:<28} {check["status"]:<15} '
            f'recorded={check["recorded"]!r} current={check["current"]!r}')
    for warning in report.get('warnings', []):
        lines.append(f'  warning: {warning}')
    return '\n'.join(lines)


def _load_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Verify recorded experiments against the current checkout.')
    parser.add_argument(
        'paths', nargs='*', help='result JSON file(s) to verify')
    parser.add_argument(
        '--json', action='store_true',
        help='emit one machine-readable report object per file')
    parser.add_argument(
        '--identity', action='store_true',
        help='print the current source/environment identity and exit')
    args = parser.parse_args(argv)

    if args.identity:
        payload = {
            'source': capture_source_identity(),
            'environment': capture_provenance(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not args.paths:
        parser.error('provide at least one result JSON path (or --identity)')

    current = capture_provenance()
    reports = {}
    saw_drift = False
    saw_unverifiable = False
    for path in args.paths:
        try:
            record = _load_json(path)
        except (OSError, ValueError) as exc:
            reports[path] = {
                'status': 'unreadable',
                'reproducible': False,
                'error': f'{type(exc).__name__}: {exc}',
                'checks': {},
                'warnings': [],
            }
            saw_unverifiable = True
            continue
        report = verify_record(record, current=current)
        reports[path] = report
        if report['status'] == 'drift':
            saw_drift = True
        elif report['status'] in ('unverifiable', 'no_provenance'):
            saw_unverifiable = True

    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=False))
    else:
        for path, report in reports.items():
            print(_format_report(path, report))
        n_ok = sum(1 for r in reports.values()
                   if r['status'] == 'reproducible')
        print(f'\n{n_ok}/{len(reports)} result(s) reproducible against this '
              f'checkout')

    if saw_drift:
        return 1
    if saw_unverifiable:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
