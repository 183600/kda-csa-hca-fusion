"""Tests for tamper-evident result envelopes.

The provenance layer answers *which code and runtime* produced a result.
These tests cover the second half of the contract: the result file also
carries a canonical content digest, so a post-hoc edit to a metric, a seed,
or the provenance block itself is detectable instead of being silently
accepted as a reproducible record.

Run with::

    pytest -q test_result_integrity.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402

import results_schema as rs  # noqa: E402
from kaggle_setup import capture_provenance, write_json_atomic  # noqa: E402
from reproducibility import verify_record  # noqa: E402


def _write(tmp_path, rows=None, *, metadata=None, name='result.json'):
    rows = rows if rows is not None else [{'op': 'softmax', 'value': 1.25}]
    path = tmp_path / name
    rs.write(str(path), rows, kind='test_kind', metadata=metadata)
    with open(path, encoding='utf-8') as fh:
        return path, json.load(fh)


def _tampered(payload, mutate):
    out = copy.deepcopy(payload)
    mutate(out)
    return out


def test_written_envelope_has_verifiable_integrity(tmp_path):
    _path, payload = _write(tmp_path)
    assert payload['integrity']['algorithm'] == rs.INTEGRITY_ALGORITHM
    assert len(payload['integrity']['value']) == 64
    assert rs.verify_integrity(payload)['status'] == 'ok'

    # The digest is over canonical content, not over disk formatting: key
    # order and indentation are irrelevant, so re-serializing a result does
    # not manufacture a false tamper alarm.
    reordered = {k: payload[k] for k in reversed(list(payload))}
    assert rs.verify_integrity(reordered)['status'] == 'ok'


def test_integrity_detects_row_metadata_and_provenance_edits(tmp_path):
    _path, payload = _write(tmp_path, metadata={'note': 'original'})

    row_edit = _tampered(
        payload, lambda p: p['results'][0].__setitem__('value', 9.99))
    metadata_edit = _tampered(
        payload, lambda p: p['metadata'].__setitem__('note', 'edited'))
    provenance_edit = _tampered(
        payload, lambda p: p.setdefault('metadata', {}).__setitem__(
            'provenance', {'git_commit': 'f' * 40}))

    for edited in (row_edit, metadata_edit, provenance_edit):
        report = rs.verify_integrity(edited)
        assert report['status'] == 'mismatch'
        assert report['recorded'] != report['observed']


def test_legacy_missing_and_malformed_integrity_are_distinguished():
    legacy = {'schema_version': '1.0', 'kind': 'k', 'metadata': {},
              'results': []}
    assert rs.verify_integrity(legacy)['status'] == 'missing'
    assert rs.verify_integrity([])['status'] == 'missing'

    unsupported = copy.deepcopy(legacy)
    unsupported['integrity'] = {'algorithm': 'md5', 'value': '0' * 32,
                                'scope': 'x'}
    assert rs.verify_integrity(unsupported)['status'] == 'unsupported'

    malformed = copy.deepcopy(legacy)
    malformed['integrity'] = {'algorithm': 'sha256', 'value': 'not-a-digest'}
    assert rs.verify_integrity(malformed)['status'] == 'malformed'

    not_a_mapping = copy.deepcopy(legacy)
    not_a_mapping['integrity'] = 'sha256:abc'
    assert rs.verify_integrity(not_a_mapping)['status'] == 'malformed'


def test_compute_integrity_does_not_mutate_payload(tmp_path):
    _path, payload = _write(tmp_path)
    before = copy.deepcopy(payload)
    block = rs.compute_integrity(payload)
    assert payload == before
    assert block['algorithm'] == rs.INTEGRITY_ALGORITHM


def test_cli_checks_integrity_and_can_require_it(tmp_path, capsys):
    good_path, good = _write(tmp_path, name='good.json')
    assert rs.main(['--check-integrity', str(good_path)]) == 0

    tampered = _tampered(
        good, lambda p: p['results'][0].__setitem__('value', 2.0))
    bad_path = tmp_path / 'bad.json'
    write_json_atomic(tampered, str(bad_path), allow_nan=False)
    assert rs.main(['--check-integrity', str(bad_path)]) == 1

    legacy_path = tmp_path / 'legacy.json'
    write_json_atomic({'results': []}, str(legacy_path), allow_nan=False)
    # Missing evidence is reported, not treated as proof of tampering ...
    assert rs.main(['--check-integrity', str(legacy_path)]) == 0
    # ... unless the caller explicitly requires a digest.
    assert rs.main(['--require-integrity', str(legacy_path)]) == 1
    capsys.readouterr()


def test_cli_integrity_is_independent_of_envelope_shape(tmp_path, capsys):
    # A campaign summary is a dict with ``runs``/``env``, not an envelope
    # with a ``results`` list. Its digest is still checkable without
    # requiring the caller to also pass the schema gate.
    summary = rs.attach_integrity({
        'runs': [{'name': 'exp1', 'status': 'ok'}],
        'env': {'git_commit': 'a' * 40},
    })
    path = tmp_path / 'summary.json'
    write_json_atomic(summary, str(path), allow_nan=False)
    assert rs.main(['--check-integrity', str(path)]) == 0
    # Schema gating remains a separate, explicit request.
    assert rs.main(['--check', str(path)]) == 1
    capsys.readouterr()


def test_verifier_turns_content_tampering_into_drift(tmp_path):
    provenance = capture_provenance()
    path, payload = _write(tmp_path, metadata={'provenance': provenance})

    ok = verify_record(payload, current=provenance)
    assert ok['integrity']['status'] == 'ok'
    assert ok['status'] == 'reproducible'

    tampered = _tampered(
        payload, lambda p: p['results'][0].__setitem__('value', 9.99))
    drift = verify_record(tampered, current=provenance)
    assert drift['integrity']['status'] == 'mismatch'
    assert drift['status'] == 'drift'
    assert drift['reproducible'] is False
    assert drift['n_integrity_mismatches'] == 1
    assert any('digest' in w for w in drift['warnings'])

    # A legacy record keeps the provenance verdict but warns that its
    # numbers cannot be checked for post-hoc edits.
    legacy = copy.deepcopy(payload)
    del legacy['integrity']
    legacy_report = verify_record(legacy, current=provenance)
    assert legacy_report['integrity']['status'] == 'missing'
    assert legacy_report['status'] == 'reproducible'
    assert any('content digest' in w for w in legacy_report['warnings'])


def test_atomic_writer_preserves_sorted_newline_convention(tmp_path):
    path = tmp_path / 'nested' / 'report.json'
    path.parent.mkdir()
    write_json_atomic({'b': 2, 'a': 1}, str(path), sort_keys=True,
                      trailing_newline=True)
    text = path.read_text(encoding='utf-8')
    assert text.endswith('\n')
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {'a': 1, 'b': 2}
    # No temporary file is left behind after a successful atomic replace.
    assert [p.name for p in path.parent.iterdir()] == ['report.json']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
