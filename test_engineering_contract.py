"""Tests for the declared-environment contract.

These are metadata checks, not numerical tests: they make a silent edit to
``pyproject.toml``, ``requirements.txt`` or ``requirements-ci.txt`` fail loudly
instead of changing the environment used by a future run.  They also pin the
registered pre-registration plans to real runner files and result paths.
"""
from __future__ import annotations

import json
import pathlib

import dependency_contract
import experiment_design
import timing_design

ROOT = pathlib.Path(__file__).resolve().parent

PYPROJECT = """\
[project]
name = "contract-demo"
requires-python = ">=3.10,<3.14"
dependencies = ["demo-dep>=1,<3"]
[project.optional-dependencies]
dev = ["pytest>=7,<10"]
[tool.setuptools]
py-modules = ["demo"]
packages = ["pkg"]
"""

REQUIREMENTS = "demo-dep>=1,<3\npytest>=7,<10\n"
CI_REQUIREMENTS = "demo-dep==2.0\npytest==9.0.2\n"


def _write_project(root: pathlib.Path) -> pathlib.Path:
    (root / 'pyproject.toml').write_text(PYPROJECT, encoding='utf-8')
    (root / 'requirements.txt').write_text(REQUIREMENTS, encoding='utf-8')
    (root / 'requirements-ci.txt').write_text(
        CI_REQUIREMENTS, encoding='utf-8')
    (root / 'demo.py').write_text('VALUE = 1\n', encoding='utf-8')
    (root / 'test_demo.py').write_text('', encoding='utf-8')
    (root / 'conftest.py').write_text('', encoding='utf-8')
    package = root / 'pkg'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    return root


def test_repository_environment_contract_is_clean():
    report = dependency_contract.check_contract(ROOT)
    assert report['ok'], report['errors']
    assert report['checks']['module_manifest']['n_expected'] > 0
    assert report['checks']['package_manifest']['missing'] == []


def test_contract_accepts_a_complete_synthetic_project(tmp_path):
    report = dependency_contract.check_contract(_write_project(tmp_path))
    assert report['ok'], report['errors']


def test_contract_reports_an_unpackaged_module(tmp_path):
    root = _write_project(tmp_path)
    (root / 'orphan.py').write_text('VALUE = 2\n', encoding='utf-8')
    report = dependency_contract.check_contract(root)
    assert not report['ok']
    assert 'orphan' in report['checks']['module_manifest']['missing']
    assert any('orphan' in error for error in report['errors'])


def test_contract_reports_a_requirements_mirror_drift(tmp_path):
    root = _write_project(tmp_path)
    (root / 'requirements.txt').write_text(
        'demo-dep>=1,<2\npytest>=7,<10\n', encoding='utf-8')
    report = dependency_contract.check_contract(root)
    assert not report['ok']
    assert report['checks']['requirements_mirror']['mismatch']


def test_contract_reports_a_ci_pin_outside_the_pyproject_bound(tmp_path):
    root = _write_project(tmp_path)
    (root / 'requirements-ci.txt').write_text(
        'demo-dep==3.0\npytest==9.0.2\n', encoding='utf-8')
    report = dependency_contract.check_contract(root)
    assert not report['ok']
    assert report['checks']['ci_pins']['errors']


def test_contract_cli_is_machine_readable_and_uses_exit_codes(
        tmp_path, capsys):
    root = _write_project(tmp_path)
    assert dependency_contract.main(['--root', str(root), '--json']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['ok'] is True
    (root / 'orphan.py').write_text('VALUE = 2\n', encoding='utf-8')
    assert dependency_contract.main(['--root', str(root), '--json']) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload['ok'] is False


def test_registered_plans_point_at_real_runners_and_result_paths():
    pyproject = dependency_contract._load_toml(ROOT / 'pyproject.toml')
    installable = set(
        pyproject['tool']['setuptools'].get('py-modules') or [])
    artifacts = {}

    assert experiment_design.validate_plans() == []
    for plan_id, plan in experiment_design.PLANS.items():
        assert plan['plan_id'] == plan_id
        runner = str(plan['runner']).split()[0]
        assert (ROOT / runner).is_file(), f'{plan_id}: missing {runner}'
        assert pathlib.Path(runner).stem in installable, (
            f'{plan_id}: {runner} is not installable')
        assert plan['result_files']
        for path in plan['result_files']:
            assert path.startswith('results/'), (plan_id, path)
            assert not pathlib.PurePosixPath(path).is_absolute()
            assert path not in artifacts, (
                f'{path} is claimed by {artifacts.get(path)} and {plan_id}')
            artifacts[path] = plan_id

    timing_errors = timing_design.validate_plans()
    assert not any(timing_errors.values()), timing_errors
    for plan_id, plan in timing_design.TIMING_PLANS.items():
        assert plan['plan_id'] == plan_id
        runner = str(plan['runner']).split()[0]
        assert (ROOT / runner).is_file(), f'{plan_id}: missing {runner}'
        assert pathlib.Path(runner).stem in installable, (
            f'{plan_id}: {runner} is not installable')
        assert plan['result_files']
        for path in plan['result_files']:
            assert path.startswith('results/'), (plan_id, path)
            assert path not in artifacts, (
                f'{path} is claimed by {artifacts.get(path)} and {plan_id}')
            artifacts[path] = plan_id


def test_run_all_snapshots_the_contract_into_the_campaign_summary():
    import run_all

    block = run_all._check_environment_contract()
    assert block['status'] == 'ok', block
    assert block['report']['ok'] is True
    json.dumps(block, allow_nan=False)


# ---------------------------------------------------------------------------
# CI gate: the documented environment-contract layer must actually run
# ---------------------------------------------------------------------------

def test_ci_gate_environment_contract_layer_passes_on_this_repository():
    import ci_gate

    before = list(ci_gate.FAILURES)
    try:
        ci_gate.check_environment_contract()
        assert ci_gate.FAILURES == before
    finally:
        ci_gate.FAILURES[:] = before


def test_ci_gate_environment_contract_layer_fails_on_drift(
        tmp_path, monkeypatch):
    import ci_gate

    root = _write_project(tmp_path)
    (root / 'orphan.py').write_text('VALUE = 2\n', encoding='utf-8')
    monkeypatch.setattr(ci_gate, 'ROOT', root)
    before = list(ci_gate.FAILURES)
    try:
        ci_gate.check_environment_contract()
        assert any('orphan' in failure for failure in ci_gate.FAILURES), (
            ci_gate.FAILURES)
    finally:
        ci_gate.FAILURES[:] = before


# ---------------------------------------------------------------------------
# Environment lock: exact installed closure + declaration identity
# ---------------------------------------------------------------------------

def test_environment_lock_is_deterministic_and_self_verifying(tmp_path):
    lock_a = dependency_contract.build_environment_lock(ROOT)
    lock_b = dependency_contract.build_environment_lock(ROOT)
    assert lock_a['kind'] == dependency_contract.LOCK_KIND
    assert lock_a['lock_sha256'] == lock_b['lock_sha256']
    assert lock_a['installed_packages']
    assert set(lock_a['declarations']) == {
        'pyproject.toml', 'requirements.txt', 'requirements-ci.txt'}

    path = tmp_path / 'environment_lock.json'
    written = dependency_contract.write_environment_lock(path, ROOT)
    on_disk = json.loads(path.read_text(encoding='utf-8'))
    assert on_disk == written
    report = dependency_contract.verify_environment_lock(on_disk, ROOT)
    assert report['ok'] is True, report
    assert report['status'] == 'ok'
    assert report['integrity']['status'] == 'ok'
    assert report['python']['status'] == 'match'


def test_environment_lock_detects_declaration_and_version_drift(tmp_path):
    root = _write_project(tmp_path)
    lock = dependency_contract.build_environment_lock(root)
    assert dependency_contract.verify_environment_lock(lock, root)['ok']

    (root / 'requirements.txt').write_text(
        'demo-dep>=1,<2\npytest>=7,<10\n', encoding='utf-8')
    report = dependency_contract.verify_environment_lock(lock, root)
    assert report['status'] == 'drift'
    assert report['declarations']['requirements.txt']['status'] == 'mismatch'
    assert report['ok'] is False

    lock = dependency_contract.build_environment_lock(ROOT)
    installed = lock['installed_packages']
    name = next(name for name, version in installed.items()
                if isinstance(version, str))
    installed[name] = '0.0.0-not-installed'
    lock['lock_sha256'] = dependency_contract.environment_lock_sha256(lock)
    report = dependency_contract.verify_environment_lock(lock, ROOT)
    assert report['status'] == 'drift'
    assert any(row['name'] == name
               for row in report['packages']['drift'])


def test_environment_lock_cli_roundtrip_and_tamper_detection(
        tmp_path, capsys):
    path = tmp_path / 'environment_lock.json'
    assert dependency_contract.main(
        ['--root', str(ROOT), '--write-lock', str(path), '--json']) == 0
    written = json.loads(capsys.readouterr().out)
    assert written['ok'] is True
    assert written['lock_sha256']

    assert dependency_contract.main(
        ['--root', str(ROOT), '--check-lock', str(path), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'ok'

    lock = json.loads(path.read_text(encoding='utf-8'))
    lock['python']['version'] = '0.0.0'
    path.write_text(json.dumps(lock), encoding='utf-8')
    assert dependency_contract.main(
        ['--root', str(ROOT), '--check-lock', str(path), '--json']) == 1
    tampered = json.loads(capsys.readouterr().out)
    assert tampered['status'] == 'drift'
    assert tampered['integrity']['status'] == 'mismatch'

    path.write_text('{not json', encoding='utf-8')
    assert dependency_contract.main(
        ['--root', str(ROOT), '--check-lock', str(path), '--json']) == 2


def test_environment_lock_rejects_malformed_shapes_and_duplicate_drift():
    lock = dependency_contract.build_environment_lock(ROOT)
    lock['installed_packages'] = []
    lock['lock_sha256'] = dependency_contract.environment_lock_sha256(lock)
    report = dependency_contract.verify_environment_lock(lock, ROOT)
    assert report['status'] == 'unverifiable'
    assert report['ok'] is False
    assert 'installed_packages' in report['errors'][0]

    lock = dependency_contract.build_environment_lock(ROOT)
    installed = lock['installed_packages']
    name = next(name for name, version in installed.items()
                if isinstance(version, str))
    installed[name] = [installed[name], '0.0.0-extra-provider']
    lock['lock_sha256'] = dependency_contract.environment_lock_sha256(lock)
    report = dependency_contract.verify_environment_lock(lock, ROOT)
    assert report['status'] == 'drift'
    assert any(row['name'] == name
               for row in report['packages']['drift'])


def test_run_all_writes_a_campaign_environment_lock(tmp_path, monkeypatch):
    import run_all

    monkeypatch.chdir(tmp_path)
    block = run_all._write_environment_lock()
    assert block['status'] == 'ok', block
    path = tmp_path / 'results' / 'environment_lock.json'
    assert path.is_file()
    lock = json.loads(path.read_text(encoding='utf-8'))
    assert lock['lock_sha256'] == block['lock_sha256']
    assert lock['lock_sha256'] in path.read_text(encoding='utf-8')
    report = dependency_contract.verify_environment_lock(lock, ROOT)
    assert report['status'] == 'ok', report
