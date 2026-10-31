"""Tests for source-tree identity and the provenance verifier.

These cover the reproducibility layer added around ``capture_provenance``:

* the working-tree source digest is deterministic, content/path sensitive,
  and independent of non-source data files;
* ``capture_source_identity`` records commit/branch/dirty state;
* the verifier distinguishes code drift from runtime/advisory drift;
* Exp 9's cross-source audit rejects two dirty runs whose code differs even
  when they share a git commit.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    _is_source_file,
    _redact_command_line,
    capture_provenance,
    capture_source_identity,
    source_tree_sha256,
)
from reproducibility import (  # noqa: E402
    compare_provenance,
    extract_provenance,
    main as reproducibility_main,
)
from run_budget_frontier import (  # noqa: E402
    _load_join_source,
    audit_join_sources,
)

GIT = shutil.which('git')


def _git(root, *args):
    env = dict(os.environ)
    env.update({
        'GIT_AUTHOR_NAME': 'test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
    })
    return subprocess.run(
        ['git', '-C', str(root), *args], check=True, capture_output=True,
        text=True, env=env,
    ).stdout.strip()


def _make_repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    _git(root, 'init', '-q')
    (root / 'a.py').write_text('VALUE = 1\n', encoding='utf-8')
    (root / 'requirements.txt').write_text('torch>=2\n', encoding='utf-8')
    _git(root, 'add', 'a.py', 'requirements.txt')
    _git(root, 'commit', '-q', '-m', 'initial')
    return root


@pytest.mark.skipif(GIT is None, reason='git is not installed')
def test_source_tree_digest_is_content_and_path_sensitive(tmp_path):
    root = _make_repo(tmp_path)
    first = source_tree_sha256(str(root))
    assert isinstance(first, str) and len(first) == 64
    assert source_tree_sha256(str(root)) == first

    # Content change.
    (root / 'a.py').write_text('VALUE = 2\n', encoding='utf-8')
    changed = source_tree_sha256(str(root))
    assert changed != first
    (root / 'a.py').write_text('VALUE = 1\n', encoding='utf-8')
    assert source_tree_sha256(str(root)) == first

    # Untracked source files are part of the identity.
    (root / 'helper.py').write_text('X = 1\n', encoding='utf-8')
    with_helper = source_tree_sha256(str(root))
    assert with_helper != first
    (root / 'helper.py').unlink()
    assert source_tree_sha256(str(root)) == first

    # A rename changes the framed path even when the bytes are unchanged.
    (root / 'a.py').rename(root / 'renamed.py')
    assert source_tree_sha256(str(root)) != first

    # Non-source payloads are not code identity.
    (root / 'data.bin').write_bytes(b'\x00\x01')
    renamed_digest = source_tree_sha256(str(root))
    (root / 'data.bin').unlink()
    assert source_tree_sha256(str(root)) == renamed_digest



def test_reference_papers_are_not_code_identity(tmp_path):
    """arXiv reference text is an experiment input, not executable source.

    ``.txt`` is in the source suffix allowlist for dependency manifests, so
    the paper payloads used to enter ``source_tree_sha256``. A paper edit or
    a newly added preprint then looked like code drift to the verifier. The
    digest now excludes the arXiv filename form while keeping ordinary text
    manifests in scope.
    """
    assert _is_source_file('2510.26692v2.txt') is False
    assert _is_source_file('2606.19348v1.txt') is False
    assert _is_source_file('references/2510.26692v2.txt') is False
    assert _is_source_file('requirements-ci.txt') is True
    assert _is_source_file('constraints.txt') is True
    assert _is_source_file('notes.txt') is True

    root = tmp_path / 'plain'
    root.mkdir()
    (root / 'module.py').write_text('X = 1\n', encoding='utf-8')
    (root / 'requirements.txt').write_text('torch>=2\n', encoding='utf-8')
    (root / '2510.26692v2.txt').write_text('paper v1\n', encoding='utf-8')
    (root / 'notes.txt').write_text('config notes\n', encoding='utf-8')

    before = source_tree_sha256(str(root))
    identity = capture_source_identity(str(root))
    # module.py + requirements.txt + notes.txt; the paper is excluded.
    assert identity['source_file_count'] == 3

    (root / '2510.26692v2.txt').write_text('paper v2\n', encoding='utf-8')
    assert source_tree_sha256(str(root)) == before

    (root / 'notes.txt').write_text('changed config notes\n', encoding='utf-8')
    assert source_tree_sha256(str(root)) != before


@pytest.mark.skipif(GIT is None, reason='git is not installed')
def test_capture_source_identity_records_commit_and_dirty_state(tmp_path):
    root = _make_repo(tmp_path)
    clean = capture_source_identity(str(root))
    expected_commit = _git(root, 'rev-parse', 'HEAD')
    assert clean['git_commit'] == expected_commit
    assert clean['git_branch'] in ('master', 'main')
    assert clean['git_dirty'] is False
    assert clean['git_dirty_files'] == 0
    assert clean['source_missing_count'] == 0
    assert clean['source_file_count'] >= 2

    (root / 'a.py').write_text('VALUE = 3\n', encoding='utf-8')
    dirty = capture_source_identity(str(root))
    assert dirty['git_commit'] == expected_commit
    assert dirty['git_dirty'] is True
    assert dirty['git_dirty_files'] == 1
    assert dirty['source_tree_sha256'] != clean['source_tree_sha256']


def test_source_tree_digest_falls_back_without_git(tmp_path):
    root = tmp_path / 'plain'
    root.mkdir()
    (root / 'module.py').write_text('X = 1\n', encoding='utf-8')
    digest = source_tree_sha256(str(root))
    assert isinstance(digest, str) and len(digest) == 64
    identity = capture_source_identity(str(root))
    assert identity['git_commit'] is None
    assert identity['git_dirty'] is None
    assert identity['source_tree_sha256'] == digest


def test_redact_command_line_hides_credential_values():
    argv = ['run.py', '--api-key', 'sekret', '--token=abc',
            '--steps', '3', '--password=xyz']
    assert _redact_command_line(argv) == [
        'run.py', '--api-key', '<redacted>', '--token=<redacted>',
        '--steps', '3', '--password=<redacted>',
    ]


def test_capture_provenance_carries_source_identity_and_is_strict_json():
    provenance = capture_provenance()
    assert isinstance(provenance['source_tree_sha256'], str)
    assert len(provenance['source_tree_sha256']) == 64
    assert provenance['source_file_count'] >= 1
    assert isinstance(provenance['command_line'], list)
    assert isinstance(provenance['package_versions'], dict)
    assert isinstance(provenance['determinism'], dict)
    assert isinstance(provenance['torch_num_threads'], int)
    json.dumps(provenance, allow_nan=False, sort_keys=True)


def _provenance(**overrides):
    base = {
        'git_commit': 'a' * 40,
        'git_dirty': False,
        'source_tree_sha256': 'b' * 64,
        'python_version': '3.12.0',
        'torch_version': '2.2.0',
        'numpy_version': '1.26.0',
        'scipy_version': '1.11.0',
        'platform': 'Linux-test',
        'cuda_version': None,
        'device_name': 'cpu',
        'matplotlib_version': '3.8.0',
        'einops_version': '0.7.0',
        'torch_num_threads': 4,
    }
    base.update(overrides)
    return base


def test_compare_provenance_matches_and_detects_code_drift():
    recorded = _provenance()
    assert compare_provenance(recorded, recorded)['status'] == 'reproducible'

    code_drift = dict(recorded, source_tree_sha256='c' * 64)
    report = compare_provenance(recorded, code_drift)
    assert report['status'] == 'drift'
    assert report['n_blocking_mismatches'] >= 1
    assert any('source_tree_sha256' in w for w in report['warnings'])

    runtime_drift = dict(recorded, torch_version='2.3.0')
    report = compare_provenance(recorded, runtime_drift)
    assert report['status'] == 'drift'
    assert report['checks']['torch_version']['status'] == 'mismatch'


def test_compare_provenance_separates_advisory_drift_from_code_drift():
    recorded = _provenance()
    other_host = dict(recorded, platform='Darwin-test', device_name='cuda:0',
                      matplotlib_version='3.9.0')
    report = compare_provenance(recorded, other_host)
    assert report['status'] == 'reproducible'
    assert report['n_advisory_mismatches'] >= 1
    assert any('advisory' in w for w in report['warnings'])


def test_compare_provenance_marks_dirty_without_digest_unverifiable():
    recorded = _provenance(git_dirty=True, source_tree_sha256=None)
    report = compare_provenance(recorded, _provenance(git_dirty=True))
    assert report['status'] == 'unverifiable'
    assert any('source digest' in w for w in report['warnings'])


def test_compare_provenance_rejects_missing_code_identity():
    recorded = {
        'python_version': '3.12.0',
        'torch_version': '2.2.0',
        'numpy_version': '1.26.0',
    }
    report = compare_provenance(recorded, recorded)
    assert report['status'] == 'unverifiable'
    assert report['status'] != 'reproducible'


def test_extract_provenance_handles_repository_envelopes():
    prov = _provenance()
    assert extract_provenance({'metadata': {'provenance': prov}}) == prov
    assert extract_provenance({'provenance': prov}) == prov
    assert extract_provenance({'env': prov}) == prov
    assert extract_provenance(prov) == prov
    assert extract_provenance({'results': []}) == {}
    assert extract_provenance([]) == {}


def test_verifier_cli_returns_two_for_missing_provenance(tmp_path, capsys):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps({'results': []}), encoding='utf-8')
    code = reproducibility_main(['--json', str(path)])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload[str(path)]['status'] == 'no_provenance'


def test_verifier_cli_returns_two_for_unreadable_file(tmp_path, capsys):
    path = tmp_path / 'broken.json'
    path.write_text('{not json', encoding='utf-8')
    code = reproducibility_main(['--json', str(path)])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload[str(path)]['status'] == 'unreadable'


def _audit_record(source, *, commit='a' * 40, dirty=True, digest='d' * 64):
    return {
        'source': source,
        'path': f'/tmp/{source}.json',
        'exists': True,
        'readable': True,
        'schema_version': '1.0',
        'n_rows': 1,
        'sha256': '0' * 64,
        'git_commit': commit,
        'git_dirty': dirty,
        'source_tree_sha256': digest,
        'captured_at_utc': '2026-01-01T00:00:00+00:00',
        'seeds': [42, 43],
        'n_seeds': 2,
        'backends': {
            'kda_backend_resolved': 'reference',
            'mamba_backend_resolved': 'reference',
            'rwkv_backend_resolved': 'reference',
        },
        'provenance_source': 'metadata.provenance',
        'error': None,
        'rows': [],
    }


def test_audit_accepts_identical_dirty_source_trees():
    records = [_audit_record(s) for s in
               ('exp4_mqar', 'exp5_ablation', 'exp8_mechanism_ablation')]
    _rows, summary = audit_join_sources(records)
    assert summary['status'] == 'comparable_dirty_tree'
    assert summary['comparable_for_ranking'] is True
    assert summary['source_tree_sha256s'] == ['d' * 64]


def test_audit_rejects_different_dirty_source_trees_at_same_commit():
    records = [
        _audit_record('exp4_mqar', digest='d' * 64),
        _audit_record('exp5_ablation', digest='e' * 64),
        _audit_record('exp8_mechanism_ablation', digest='d' * 64),
    ]
    _rows, summary = audit_join_sources(records)
    assert summary['status'] == 'mixed_source_trees'
    assert summary['comparable_for_ranking'] is False
    assert summary['source_tree_sha256s'] == ['d' * 64, 'e' * 64]
    assert any('different code' in w for w in summary['warnings'])


def test_audit_warns_when_dirty_sources_have_no_digest():
    records = [
        _audit_record('exp4_mqar', digest=None),
        _audit_record('exp5_ablation', digest=None),
        _audit_record('exp8_mechanism_ablation', digest=None),
    ]
    _rows, summary = audit_join_sources(records)
    assert summary['status'] == 'comparable_dirty_tree'
    assert summary['missing_source_tree'] == [
        'exp4_mqar', 'exp5_ablation', 'exp8_mechanism_ablation']
    assert any('no source-tree digest' in w for w in summary['warnings'])


def test_load_join_source_records_source_digest_from_sidecar(tmp_path):
    result_path = tmp_path / 'exp5_ablation.json'
    prov_path = tmp_path / 'exp5_ablation_provenance.json'
    result_path.write_text(json.dumps([{'ratio': '3:1:1',
                                        'mean_acc': 0.5}]),
                           encoding='utf-8')
    prov_path.write_text(json.dumps({
        'git_commit': 'c' * 40,
        'git_dirty': True,
        'source_tree_sha256': 'f' * 64,
        'seeds': [42],
    }), encoding='utf-8')
    rec = _load_join_source('exp5_ablation', str(result_path),
                            provenance_path=str(prov_path))
    assert rec['readable'] is True
    assert rec['source_tree_sha256'] == 'f' * 64
    assert rec['git_dirty'] is True
