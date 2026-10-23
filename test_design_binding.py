"""Tests for the pre-registration -> result binding.

The runtime design audit records the plan it was evaluated against. These
tests cover the fingerprint itself, its embedding in ``audit_run`` and in the
prospective design manifest, and the verifier's ability to distinguish a
still-matching plan from a post-hoc registry/snapshot edit.

How to run::

    pytest -q test_design_binding.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import experiment_design as D  # noqa: E402
from reproducibility import (  # noqa: E402
    compare_design_binding,
    verify_record,
)


def _levels(plan):
    """Observed factor levels for a plan (the seed block is a formula)."""
    return {
        name: list(cfg['levels'])
        for name, cfg in plan.get('factors', {}).items()
        if isinstance(cfg.get('levels'), list)
    }


def _audit(plan_id: str, n_seeds: int | None = None):
    plan = D.get_plan(plan_id)
    if n_seeds is None:
        n_seeds = max(2, int(plan.get('planned_seeds', 2)))
    return D.audit_run(
        plan_id,
        seeds=[42 + i for i in range(int(n_seeds))],
        levels=_levels(plan),
    )


def test_plan_sha256_is_canonical_and_content_sensitive():
    plan = D.get_plan('exp4_mqar')
    digest = D.plan_sha256(plan)
    assert digest == D.plan_sha256('exp4_mqar')
    # Key order is not part of the identity.
    reordered = dict(reversed(list(plan.items())))
    assert D.plan_sha256(reordered) == digest
    # Any semantic change is.
    changed = copy.deepcopy(plan)
    changed['sesoi'] = float(changed['sesoi']) + 0.01
    assert D.plan_sha256(changed) != digest
    changed = copy.deepcopy(plan)
    changed['required_levels'] = {}
    assert D.plan_sha256(changed) != digest


def test_plan_sha256_rejects_unknown_or_non_finite_plans():
    with pytest.raises(D.DesignError):
        D.plan_sha256('not_a_registered_plan')
    with pytest.raises(D.DesignError):
        D.plan_sha256({'sesoi': float('nan')})
    with pytest.raises(D.DesignError):
        D.plan_sha256(['not', 'a', 'dict'])


@pytest.mark.parametrize('plan_id', D.PLAN_IDS)
def test_every_audit_embeds_matching_fingerprint_and_snapshot(plan_id):
    audit = _audit(plan_id)
    plan = D.get_plan(plan_id)
    assert audit['plan_sha256'] == D.plan_sha256(plan)
    assert audit['plan_fingerprint_scope'] == D.PLAN_FINGERPRINT_SCOPE
    assert audit['plan_definition'] == plan
    # The audit must remain strict JSON: it is written into every result file.
    json.dumps(audit, allow_nan=False)
    report = compare_design_binding({'metadata': {'experiment_design': audit}})
    assert report['status'] == 'match'
    assert report['current_sha256'] == audit['plan_sha256']
    assert report['snapshot_sha256'] == audit['plan_sha256']


def test_design_manifest_binds_each_plan_and_provenance(tmp_path):
    provenance = {
        'git_commit': 'a' * 40,
        'git_dirty': False,
        'source_tree_sha256': 'b' * 64,
        'torch_version': '2.0.0',
    }
    manifest = D.design_manifest(
        seeds_by_plan={'exp4_mqar': 3}, provenance=provenance)
    assert manifest['metadata']['provenance'] == provenance
    assert manifest['metadata']['plan_fingerprint_scope'] == \
        D.PLAN_FINGERPRINT_SCOPE
    for row in manifest['results']:
        assert row['plan_sha256'] == D.plan_sha256(row['plan_id'])
        assert row['plan_fingerprint_scope'] == D.PLAN_FINGERPRINT_SCOPE

    path = tmp_path / 'experiment_design.json'
    D.write_design_manifest(str(path), manifest=manifest)
    with path.open(encoding='utf-8') as fh:
        loaded = json.load(fh)
    assert loaded['metadata']['provenance'] == provenance
    assert loaded['results'] == manifest['results']


def test_design_manifest_rejects_non_strict_provenance():
    with pytest.raises(D.DesignError):
        D.design_manifest(provenance=['not', 'a', 'dict'])
    with pytest.raises(D.DesignError):
        D.design_manifest(provenance={'loss': float('nan')})


def test_binding_extracts_ratio_sidecar_provenance():
    audit = _audit('exp5_ratio', n_seeds=4)
    # run_ablation stores the ratio audit inside its sidecar provenance block
    # rather than under metadata.experiment_design. Both the sidecar file
    # itself (a bare provenance dict) and an envelope that nests it must work.
    sidecar = {'git_commit': 'c' * 40, 'experiment_design': audit}
    report = compare_design_binding(sidecar)
    assert report['status'] == 'match'
    assert report['plan_id'] == 'exp5_ratio'

    nested = {'provenance': sidecar}
    assert compare_design_binding(nested)['status'] == 'match'


def test_snapshot_tamper_is_flagged_as_drift():
    audit = _audit('exp4_mqar')
    tampered = copy.deepcopy(audit)
    tampered['plan_definition']['sesoi'] = float(
        tampered['plan_definition']['sesoi']) + 0.01
    report = compare_design_binding(
        {'metadata': {'experiment_design': tampered}})
    assert report['status'] == 'snapshot_mismatch'
    assert report['snapshot_sha256'] != report['recorded_sha256']

    verified = verify_record({'metadata': {'experiment_design': tampered}})
    assert verified['status'] == 'drift'
    assert verified['reproducible'] is False
    assert verified['n_design_mismatches'] == 1
    assert verified['design']['status'] == 'snapshot_mismatch'


def test_registry_drift_after_plan_change_is_flagged_as_drift():
    # Simulate a result audited against a plan whose SESOI was later edited in
    # the registry. The snapshot is internally consistent (it hashes to the
    # recorded digest), so only the registry comparison can catch it.
    old_plan = copy.deepcopy(D.get_plan('exp4_mqar'))
    old_plan['sesoi'] = float(old_plan['sesoi']) + 0.01
    audit = _audit('exp4_mqar')
    audit['plan_definition'] = old_plan
    audit['plan_sha256'] = D.plan_sha256(old_plan)

    report = compare_design_binding({'metadata': {'experiment_design': audit}})
    assert report['status'] == 'registry_drift'
    assert report['snapshot_sha256'] == report['recorded_sha256']
    assert report['current_sha256'] != report['recorded_sha256']

    verified = verify_record({'metadata': {'experiment_design': audit}})
    assert verified['status'] == 'drift'
    assert verified['n_design_mismatches'] == 1


def test_legacy_audit_without_fingerprint_is_a_warning_only():
    legacy = _audit('exp4_mqar')
    for key in ('plan_sha256', 'plan_definition', 'plan_fingerprint_scope'):
        legacy.pop(key)
    report = compare_design_binding(
        {'metadata': {'experiment_design': legacy}})
    assert report['status'] == 'unbound'
    assert report['recorded_sha256'] is None

    verified = verify_record({'metadata': {'experiment_design': legacy}})
    assert verified['design']['status'] == 'unbound'
    assert verified['status'] != 'drift'
    assert any('legacy' in warning for warning in verified['warnings'])


def test_matching_design_and_provenance_stay_reproducible():
    # End-to-end happy path: a current-format envelope (provenance + content
    # digest) with a matching plan must still verify as reproducible. This
    # guards against the new layer turning every valid result into a warning.
    from kaggle_setup import capture_provenance
    from results_schema import normalize

    audit = _audit('exp4_mqar')
    record = normalize(
        {'results': []},
        kind='exp4_mqar',
        metadata={
            'provenance': capture_provenance(),
            'experiment_design': audit,
        },
    )
    report = verify_record(record)
    assert report['status'] == 'reproducible'
    assert report['design']['status'] == 'match'
    assert report['integrity']['status'] == 'ok'
    assert report['n_design_mismatches'] == 0
    assert report['warnings'] == []


def test_missing_audit_is_reported_but_not_drift():
    report = compare_design_binding({'results': []})
    assert report['status'] == 'missing'
    verified = verify_record({'results': []})
    assert verified['design']['status'] == 'missing'
    assert verified['status'] != 'drift'


def test_unknown_plan_is_unverifiable():
    audit = _audit('exp4_mqar')
    audit['plan_id'] = 'plan_removed_from_the_registry'
    report = compare_design_binding({'metadata': {'experiment_design': audit}})
    assert report['status'] == 'unknown_plan'
    verified = verify_record({'metadata': {'experiment_design': audit}})
    assert verified['status'] in ('unverifiable', 'no_provenance')
    assert verified['reproducible'] is False
    assert verified['design']['status'] == 'unknown_plan'


def test_malformed_fingerprint_is_not_silently_accepted():
    audit = _audit('exp4_mqar')
    audit['plan_sha256'] = 'not-a-digest'
    report = compare_design_binding({'metadata': {'experiment_design': audit}})
    assert report['status'] == 'malformed'
    verified = verify_record({'metadata': {'experiment_design': audit}})
    assert verified['reproducible'] is False
    assert verified['design']['status'] == 'malformed'


def test_scope_change_is_unverifiable():
    audit = _audit('exp4_mqar')
    audit['plan_fingerprint_scope'] = 'some-future-scope-v2'
    report = compare_design_binding({'metadata': {'experiment_design': audit}})
    assert report['status'] == 'malformed'
    verified = verify_record({'metadata': {'experiment_design': audit}})
    assert verified['status'] in ('unverifiable', 'no_provenance')
    assert verified['reproducible'] is False
