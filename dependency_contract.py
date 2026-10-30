#!/usr/bin/env python3
"""Check that the declared Python environment is internally consistent.

The reproducibility contract has three declarations:

* ``pyproject.toml`` is the canonical source of dependency bounds and of the
  installable module/package manifest;
* ``requirements.txt`` mirrors the runtime and ``dev`` requirements so the
  ``pip install -r`` workflow creates the same environment as
  ``pip install -e .[dev]``;
* ``requirements-ci.txt`` pins the exact versions used by the CI gate.

A silent edit to one file but not the others changes the environment without
changing any experiment code, which is exactly the kind of drift the result
provenance cannot diagnose after the fact.  This module checks the three
declarations without importing torch or running an experiment.  It is used by
``ci_gate.py`` and by ``test_engineering_contract.py``.

The checks are static metadata checks.  They do not resolve dependencies over
the network; use a generated lock file for bit-exact replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent


class ContractUnavailable(RuntimeError):
    """Raised when the metadata parsers are unavailable."""


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ContractUnavailable(
                'dependency_contract requires tomllib (Python >= 3.11) or '
                'tomli (installed with pytest on Python 3.10)') from exc
    with path.open('rb') as fh:
        return tomllib.load(fh)


def _packaging():
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.utils import canonicalize_name
        from packaging.version import Version
    except ModuleNotFoundError as exc:
        raise ContractUnavailable(
            'dependency_contract requires the packaging distribution '
            '(a dependency of pytest and pip)') from exc
    return Requirement, SpecifierSet, canonicalize_name, Version


def _requirement_text(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith('#'):
        return None
    if text.startswith('-') or '://' in text:
        return None
    if ' #' in text:
        text = text.split(' #', 1)[0].strip()
    return text or None


def _parse_requirements(path: pathlib.Path, Requirement) -> list:
    entries = []
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(),
                                  start=1):
        text = _requirement_text(line)
        if text is None:
            continue
        try:
            entries.append(Requirement(text))
        except Exception as exc:
            raise ValueError(
                f'{path.name}:{lineno}: invalid requirement {text!r}: {exc}'
            ) from None
    return entries


def _merge_requirements(requirements, canonicalize_name) -> dict:
    merged = {}
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        merged.setdefault(name, requirement)
    return merged


def _spec_key(requirement) -> tuple:
    return tuple(sorted(str(spec) for spec in requirement.specifier))


def _marker_key(requirement) -> str | None:
    return None if requirement.marker is None else str(requirement.marker)


def _check_modules(base: pathlib.Path, setuptools: dict) -> dict:
    listed = setuptools.get('py-modules') or []
    expected = {
        path.stem for path in base.glob('*.py')
        if not path.name.startswith('test_') and path.name != 'conftest.py'
    }
    listed_set = set(listed)
    missing = sorted(expected - listed_set)
    unknown = sorted(listed_set - expected)
    return {
        'ok': not missing and not unknown,
        'n_expected': len(expected),
        'n_listed': len(listed_set),
        'missing': missing,
        'unknown': unknown,
    }


def _check_packages(base: pathlib.Path, setuptools: dict) -> dict:
    listed = set(setuptools.get('packages') or [])
    expected = {
        path.name for path in base.iterdir()
        if path.is_dir() and path.name.isidentifier()
        and (path / '__init__.py').is_file()
    }
    missing = sorted(expected - listed)
    unknown = sorted(listed - expected)
    return {
        'ok': not missing and not unknown,
        'n_expected': len(expected),
        'n_listed': len(listed),
        'missing': missing,
        'unknown': unknown,
    }


# --------------------------------------------------------------------------
# Environment lock: exact installed versions + declaration identity
# --------------------------------------------------------------------------
#
# The contract above proves that the three declarations agree.  It does not
# prove that a later rerun installs the same numerical stack.  The lock adds
# the missing half: a canonical, self-checking snapshot of the declarations
# and the exact installed distributions on the machine that produced a run.
# Verify it on the replay host before running the campaign; a lock is a
# host-specific artifact, not a universal requirements file.

LOCK_VERSION = '1.0'
LOCK_KIND = 'python_environment_lock'
LOCK_DIGEST_SCOPE = (
    'sha256 of the canonical JSON lock (sorted keys, compact separators, '
    'UTF-8) with the lock_sha256 field itself removed')
_DECLARATION_FILES = ('pyproject.toml', 'requirements.txt',
                      'requirements-ci.txt')


def _canonical_lock_bytes(lock: dict) -> bytes:
    body = {k: v for k, v in lock.items() if k != 'lock_sha256'}
    return json.dumps(body, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False,
                      default=str).encode('utf-8')


def environment_lock_sha256(lock: dict) -> str:
    """Canonical content digest of an environment lock.

    The digest covers every field except ``lock_sha256`` itself, so editing a
    version or declaration hash after the lock was written changes the
    digest.  It is tamper evidence, not a signature.
    """
    if not isinstance(lock, dict):
        raise TypeError(
            f'environment_lock_sha256 expects a dict, found '
            f'{type(lock).__name__}')
    return hashlib.sha256(_canonical_lock_bytes(lock)).hexdigest()


def _file_digest(path: pathlib.Path) -> dict:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {'present': False, 'sha256': None, 'n_bytes': None,
                'error': type(exc).__name__}
    return {
        'present': True,
        'sha256': hashlib.sha256(data).hexdigest(),
        'n_bytes': len(data),
    }


def _installed_distributions(canonicalize_name) -> dict[str, str | list[str]]:
    """Best-effort exact versions of every installed distribution.

    Names are PEP 503-normalized.  If the same normalized name is provided by
    more than one distribution with different versions, the value is a sorted
    list rather than silently picking one.  Scanning failures are recorded by
    omission; lock creation must not fail just because one metadata entry is
    malformed.
    """
    try:
        from importlib import metadata
        distributions = list(metadata.distributions())
    except Exception:
        return {}
    found: dict[str, set[str]] = {}
    for dist in distributions:
        try:
            raw_name = dist.metadata['Name']
            version = str(dist.version)
        except Exception:
            continue
        if not raw_name:
            continue
        name = canonicalize_name(str(raw_name))
        found.setdefault(name, set()).add(version)
    installed: dict[str, str | list[str]] = {}
    for name in sorted(found):
        versions = sorted(found[name])
        installed[name] = versions[0] if len(versions) == 1 else versions
    return installed


def _declared_package_table(project: dict, Requirement,
                            canonicalize_name) -> dict[str, dict]:
    declared: dict[str, dict] = {}
    groups = [('runtime', project.get('dependencies') or [])]
    for extra, raw_requirements in sorted(
            (project.get('optional-dependencies') or {}).items()):
        groups.append((f'extra:{extra}', raw_requirements or []))
    for group, raw_requirements in groups:
        for raw in raw_requirements:
            try:
                requirement = Requirement(raw)
            except Exception as exc:
                raise ValueError(
                    f'pyproject.toml dependency {raw!r} is invalid: {exc}'
                ) from None
            name = canonicalize_name(requirement.name)
            row = declared.setdefault(
                name, {'specifiers': [], 'groups': [], 'ci_pin': None})
            specifier = str(requirement)
            if specifier not in row['specifiers']:
                row['specifiers'].append(specifier)
            if group not in row['groups']:
                row['groups'].append(group)
    for row in declared.values():
        row['specifiers'].sort()
        row['groups'].sort()
    return declared


def build_environment_lock(root: str | pathlib.Path | None = None) -> dict:
    """Build a deterministic, strict-JSON environment lock.

    The lock records:

    * SHA-256 and size of each declaration file;
    * the declared requirement specifiers and CI exact pins;
    * Python implementation/version/``requires-python``;
    * OS system and machine;
    * every installed distribution's exact version (the transitive closure);
    * a canonical ``lock_sha256`` over all of the above.

    It never resolves packages over the network and never imports torch.  The
    same declarations plus the same installed distributions produce the same
    digest, so a lock can be compared without a timestamp.
    """
    base = pathlib.Path(root) if root is not None else ROOT
    Requirement, _SpecifierSet, canonicalize_name, _Version = _packaging()
    pyproject_path = base / 'pyproject.toml'
    if not pyproject_path.is_file():
        raise FileNotFoundError(f'missing declaration file: {pyproject_path}')
    data = _load_toml(pyproject_path)
    project = data.get('project') or {}

    declarations = {}
    for name in _DECLARATION_FILES:
        path = base / name
        declarations[name] = _file_digest(path)

    declared = _declared_package_table(project, Requirement,
                                       canonicalize_name)
    ci_path = base / 'requirements-ci.txt'
    if ci_path.is_file():
        for requirement in _parse_requirements(ci_path, Requirement):
            name = canonicalize_name(requirement.name)
            specifiers = list(requirement.specifier)
            if len(specifiers) == 1 and specifiers[0].operator == '==':
                declared.setdefault(
                    name, {'specifiers': [], 'groups': [], 'ci_pin': None}
                )['ci_pin'] = specifiers[0].version

    installed = _installed_distributions(canonicalize_name)
    # Declared packages that are not installed are still part of the lock:
    # recording the absence is what lets the replay host detect that a
    # dependency appeared (or disappeared) later.
    for name in declared:
        installed.setdefault(name, None)
    installed = {name: installed[name] for name in sorted(installed)}

    lock = {
        'lock_version': LOCK_VERSION,
        'kind': LOCK_KIND,
        'digest_scope': LOCK_DIGEST_SCOPE,
        'python': {
            'implementation': platform.python_implementation(),
            'version': platform.python_version(),
            'requires_python': project.get('requires-python'),
        },
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
        },
        'declarations': declarations,
        'declared_packages': {name: declared[name]
                              for name in sorted(declared)},
        'installed_packages': installed,
        'lock_sha256': None,
    }
    lock['lock_sha256'] = environment_lock_sha256(lock)
    return lock


def write_environment_lock(path: str | pathlib.Path,
                           root: str | pathlib.Path | None = None) -> dict:
    """Build and atomically write an environment lock; return the lock."""
    lock = build_environment_lock(root)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f'.{target.name}.', suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(lock, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write('\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return lock


def _same_installed(left, right) -> bool:
    """Exact equality, including duplicate-provider list shape.

    A single version and a list of versions are not interchangeable: the
    former means one provider, the latter means the same normalized name was
    provided by multiple distributions.  Treating them as equal would hide a
    real change in the installed closure.
    """
    return left == right


def verify_environment_lock(lock, root: str | pathlib.Path | None = None) -> dict:
    """Compare a recorded lock with the current environment.

    Status contract (mirrors ``reproducibility.py``):

    * ``ok`` — the lock digest is intact (or the legacy digest is missing, a
      warning) and every declaration hash, Python version, declared specifier,
      CI pin and installed distribution matches;
    * ``drift`` — at least one recorded value differs, or the lock content was
      edited after it was written;
    * ``unverifiable`` — the current environment cannot be inspected, or the
      lock has no usable digest/version metadata;
    * ``malformed`` — the payload is not an environment-lock mapping.

    A lock is host-specific because it records the installed transitive
    closure.  Comparing a CPU lock with a CUDA host is expected to drift.
    """
    report = {
        'status': 'malformed',
        'ok': False,
        'integrity': {'status': 'unverifiable', 'recorded': None,
                      'observed': None},
        'declarations': {},
        'python': {},
        'platform': {},
        'packages': {'n_recorded': 0, 'n_current': 0, 'drift': []},
        'errors': [],
        'warnings': [],
    }
    if not isinstance(lock, dict):
        report['errors'].append(
            f'environment lock must be an object, found '
            f'{type(lock).__name__}')
        return report
    if lock.get('kind') != LOCK_KIND:
        report['errors'].append(
            f"unrecognized lock kind: {lock.get('kind')!r}")
        return report
    try:
        recorded_major = int(str(lock.get('lock_version')).split('.')[0])
        current_major = int(str(LOCK_VERSION).split('.')[0])
    except (TypeError, ValueError):
        report['errors'].append(
            f"unrecognized lock_version: {lock.get('lock_version')!r}")
        return report
    if recorded_major != current_major:
        report['errors'].append(
            f"lock_version {lock.get('lock_version')!r} is not supported by "
            f'this reader ({LOCK_VERSION})')
        report['status'] = 'unverifiable'
        return report
    for field in ('python', 'platform', 'declarations', 'declared_packages',
                  'installed_packages'):
        value = lock.get(field)
        if value is not None and not isinstance(value, dict):
            report['errors'].append(
                f'lock field {field!r} must be an object, found '
                f'{type(value).__name__}')
            report['status'] = 'unverifiable'
            return report

    integrity = report['integrity']
    recorded_digest = lock.get('lock_sha256')
    integrity['recorded'] = recorded_digest
    if recorded_digest is None:
        integrity['status'] = 'missing'
        report['warnings'].append(
            'the lock carries no lock_sha256 (legacy lock); its integrity '
            'cannot be checked, but the recorded environment is still '
            'compared')
    elif not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        integrity['status'] = 'malformed'
        report['errors'].append('the lock carries a malformed lock_sha256')
        report['status'] = 'unverifiable'
        return report
    else:
        try:
            observed = environment_lock_sha256(lock)
        except Exception as exc:
            integrity['status'] = 'malformed'
            report['errors'].append(
                f'the lock cannot be canonicalized: '
                f'{type(exc).__name__}: {exc}')
            report['status'] = 'unverifiable'
            return report
        integrity['observed'] = observed
        if observed != recorded_digest.lower():
            integrity['status'] = 'mismatch'
            report['errors'].append(
                'the lock content does not match its lock_sha256: at least '
                'one field was edited after it was written')
            report['status'] = 'drift'
            report['ok'] = False
            return report
        integrity['status'] = 'ok'

    try:
        current = build_environment_lock(root)
    except (ContractUnavailable, OSError, ValueError, KeyError) as exc:
        report['status'] = 'unverifiable'
        report['errors'].append(
            f'the current environment cannot be inspected: '
            f'{type(exc).__name__}: {exc}')
        return report

    drift = False
    recorded_declarations = lock.get('declarations') or {}
    current_declarations = current.get('declarations') or {}
    for name in sorted(set(recorded_declarations) | set(current_declarations)):
        left = recorded_declarations.get(name) or {}
        right = current_declarations.get(name) or {}
        if not isinstance(left, dict) or not isinstance(right, dict):
            same = False
        else:
            same = (left.get('sha256') == right.get('sha256')
                    and left.get('present') == right.get('present'))
        report['declarations'][name] = {
            'recorded_sha256': (left.get('sha256')
                                if isinstance(left, dict) else None),
            'current_sha256': (right.get('sha256')
                               if isinstance(right, dict) else None),
            'status': 'match' if same else 'mismatch',
        }
        if not same:
            drift = True

    recorded_python = lock.get('python') or {}
    current_python = current.get('python') or {}
    python_same = (
        recorded_python.get('version') == current_python.get('version')
        and recorded_python.get('implementation')
        == current_python.get('implementation'))
    report['python'] = {
        'recorded': recorded_python.get('version'),
        'current': current_python.get('version'),
        'implementation_recorded': recorded_python.get('implementation'),
        'implementation_current': current_python.get('implementation'),
        'status': 'match' if python_same else 'mismatch',
    }
    if not python_same:
        drift = True

    recorded_platform = lock.get('platform') or {}
    current_platform = current.get('platform') or {}
    platform_same = (
        recorded_platform.get('system') == current_platform.get('system')
        and recorded_platform.get('machine')
        == current_platform.get('machine'))
    report['platform'] = {
        'recorded': dict(recorded_platform),
        'current': dict(current_platform),
        'status': 'match' if platform_same else 'mismatch',
    }
    if not platform_same:
        drift = True

    recorded_declared = lock.get('declared_packages') or {}
    current_declared = current.get('declared_packages') or {}
    package_drift = []
    for name in sorted(set(recorded_declared) | set(current_declared)):
        left = recorded_declared.get(name) or {}
        right = current_declared.get(name) or {}
        same = (isinstance(left, dict) and isinstance(right, dict)
                and left.get('specifiers') == right.get('specifiers')
                and left.get('ci_pin') == right.get('ci_pin')
                and left.get('groups') == right.get('groups'))
        if not same:
            package_drift.append({
                'name': name,
                'kind': 'declared_specifier',
                'recorded': left,
                'current': right,
            })
            drift = True

    recorded_installed = lock.get('installed_packages') or {}
    current_installed = current.get('installed_packages') or {}
    report['packages']['n_recorded'] = len(recorded_installed)
    report['packages']['n_current'] = len(current_installed)
    for name in sorted(set(recorded_installed) | set(current_installed)):
        left = recorded_installed.get(name)
        right = current_installed.get(name)
        if _same_installed(left, right):
            continue
        if name not in current_installed:
            kind = 'missing'
        elif name not in recorded_installed:
            kind = 'unexpected'
        else:
            kind = 'version'
        package_drift.append({
            'name': name,
            'kind': kind,
            'recorded': left,
            'current': right,
        })
        drift = True
    report['packages']['drift'] = package_drift

    if drift:
        report['status'] = 'drift'
        report['errors'].append(
            f'environment drift: {len(package_drift)} package difference(s), '
            f'{sum(1 for row in report["declarations"].values() if row["status"] != "match")} '
            'declaration difference(s)'
            + ('' if report['python']['status'] == 'match'
               else ', Python identity differs')
            + ('' if report['platform']['status'] == 'match'
               else ', platform differs'))
        return report

    report['status'] = 'ok'
    report['ok'] = True
    return report


def check_contract(root: str | pathlib.Path | None = None) -> dict:
    """Return a strict-JSON report for the environment declaration.

    ``root`` defaults to the directory containing this module.  The report's
    ``ok`` is true only when every check passes.  Individual ``checks`` carry
    the offending names so a failure is actionable rather than only a boolean.
    """
    base = pathlib.Path(root) if root is not None else ROOT
    Requirement, SpecifierSet, canonicalize_name, Version = _packaging()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, dict] = {}

    pyproject_path = base / 'pyproject.toml'
    requirements_path = base / 'requirements.txt'
    ci_path = base / 'requirements-ci.txt'
    for path in (pyproject_path, requirements_path, ci_path):
        if not path.is_file():
            errors.append(f'missing declaration file: {path.name}')

    data: dict[str, Any] = {}
    if pyproject_path.is_file():
        try:
            data = _load_toml(pyproject_path)
        except (OSError, ValueError) as exc:
            errors.append(f'cannot parse {pyproject_path.name}: {exc}')

    project = data.get('project') or {}
    setuptools = (data.get('tool') or {}).get('setuptools') or {}

    checks['module_manifest'] = _check_modules(base, setuptools)
    if not checks['module_manifest']['ok']:
        if checks['module_manifest']['missing']:
            errors.append(
                'pyproject.toml py-modules omits existing module(s): '
                + ', '.join(checks['module_manifest']['missing']))
        if checks['module_manifest']['unknown']:
            errors.append(
                'pyproject.toml py-modules names missing module(s): '
                + ', '.join(checks['module_manifest']['unknown']))

    checks['package_manifest'] = _check_packages(base, setuptools)
    if not checks['package_manifest']['ok']:
        if checks['package_manifest']['missing']:
            errors.append(
                'pyproject.toml packages omits existing package(s): '
                + ', '.join(checks['package_manifest']['missing']))
        if checks['package_manifest']['unknown']:
            errors.append(
                'pyproject.toml packages names missing package(s): '
                + ', '.join(checks['package_manifest']['unknown']))

    declared_requirements = []
    for raw in project.get('dependencies') or []:
        try:
            declared_requirements.append(Requirement(raw))
        except Exception as exc:
            errors.append(f'pyproject.toml dependency {raw!r} is invalid: {exc}')
    extras = project.get('optional-dependencies') or {}
    for raw in extras.get('dev') or []:
        try:
            declared_requirements.append(Requirement(raw))
        except Exception as exc:
            errors.append(
                f'pyproject.toml dev dependency {raw!r} is invalid: {exc}')
    declared = _merge_requirements(declared_requirements, canonicalize_name)

    installed_requirements = []
    if requirements_path.is_file():
        try:
            installed_requirements = _parse_requirements(
                requirements_path, Requirement)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    mirrored = {}
    for requirement in installed_requirements:
        name = canonicalize_name(requirement.name)
        if name in mirrored:
            errors.append(
                f'requirements.txt lists {name} more than once')
        mirrored[name] = requirement

    mirror_missing = sorted(set(declared) - set(mirrored))
    mirror_unknown = sorted(set(mirrored) - set(declared))
    mirror_mismatch = []
    for name in sorted(set(declared) & set(mirrored)):
        left = declared[name]
        right = mirrored[name]
        if (_spec_key(left) != _spec_key(right)
                or _marker_key(left) != _marker_key(right)):
            mirror_mismatch.append({
                'name': name,
                'pyproject': str(left),
                'requirements': str(right),
            })
    checks['requirements_mirror'] = {
        'ok': not mirror_missing and not mirror_unknown and not mirror_mismatch,
        'missing': mirror_missing,
        'unknown': mirror_unknown,
        'mismatch': mirror_mismatch,
    }
    if mirror_missing:
        errors.append(
            'requirements.txt omits pyproject dependency/ies: '
            + ', '.join(mirror_missing))
    if mirror_unknown:
        errors.append(
            'requirements.txt lists package(s) absent from pyproject: '
            + ', '.join(mirror_unknown))
    for row in mirror_mismatch:
        errors.append(
            f'requirements.txt {row["name"]} specifier differs from '
            f'pyproject.toml: {row["requirements"]!r} vs '
            f'{row["pyproject"]!r}')

    ci_requirements = []
    if ci_path.is_file():
        try:
            ci_requirements = _parse_requirements(ci_path, Requirement)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    ci_errors = []
    for requirement in ci_requirements:
        name = canonicalize_name(requirement.name)
        specifiers = list(requirement.specifier)
        if name not in declared:
            ci_errors.append(
                f'{name} is pinned in requirements-ci.txt but absent from '
                'pyproject.toml')
            continue
        if len(specifiers) != 1 or specifiers[0].operator != '==':
            ci_errors.append(
                f'{name} must use one exact == pin in requirements-ci.txt')
            continue
        try:
            version = Version(specifiers[0].version)
        except Exception as exc:
            ci_errors.append(f'{name} has an invalid pinned version: {exc}')
            continue
        allowed = declared[name]
        if not allowed.specifier.contains(version, prereleases=True):
            ci_errors.append(
                f'{name}=={version} violates the pyproject.toml bound '
                f'{allowed.specifier or "(unbounded)"}')
        if name in mirrored and not mirrored[name].specifier.contains(
                version, prereleases=True):
            ci_errors.append(
                f'{name}=={version} violates the requirements.txt bound '
                f'{mirrored[name].specifier or "(unbounded)"}')
    checks['ci_pins'] = {
        'ok': not ci_errors,
        'n_pins': len(ci_requirements),
        'errors': ci_errors,
    }
    errors.extend(ci_errors)

    requires_python = project.get('requires-python')
    python_ok = True
    if requires_python:
        try:
            specifier = SpecifierSet(requires_python)
            current = Version(
                f'{sys.version_info.major}.{sys.version_info.minor}.'
                f'{sys.version_info.micro}')
            python_ok = specifier.contains(current, prereleases=True)
            if not python_ok:
                errors.append(
                    f'running Python {current} is outside '
                    f'project.requires-python {requires_python!r}')
        except Exception as exc:
            python_ok = False
            errors.append(f'invalid project.requires-python: {exc}')
    checks['requires_python'] = {
        'ok': python_ok,
        'declared': requires_python,
        'current': (f'{sys.version_info.major}.{sys.version_info.minor}.'
                    f'{sys.version_info.micro}'),
    }

    return {
        'ok': not errors,
        'root': str(base),
        'errors': errors,
        'warnings': warnings,
        'checks': checks,
    }


def _print_lock_report(report: dict) -> None:
    status = 'PASS' if report.get('ok') else report.get('status', 'FAIL')
    print(f'[dependency-contract] lock {status.upper()}')
    integrity = report.get('integrity') or {}
    print(f"  integrity              {integrity.get('status', 'unknown')}")
    python = report.get('python') or {}
    print(f"  python                 {python.get('status', 'unknown')}")
    platform_row = report.get('platform') or {}
    print(f"  platform               {platform_row.get('status', 'unknown')}")
    packages = report.get('packages') or {}
    print(f"  packages               recorded={packages.get('n_recorded')} "
          f"current={packages.get('n_current')} "
          f"drift={len(packages.get('drift') or [])}")
    for name, row in (report.get('declarations') or {}).items():
        print(f"  {name:<22} {row.get('status', 'unknown')}")
    for row in (packages.get('drift') or [])[:20]:
        print(f"  drift: {row.get('name')} ({row.get('kind')}): "
              f"recorded={row.get('recorded')!r} "
              f"current={row.get('current')!r}")
    for error in report.get('errors') or []:
        print(f'  error: {error}')
    for warning in report.get('warnings') or []:
        print(f'  warning: {warning}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Check pyproject.toml / requirements.txt / '
                    'requirements-ci.txt consistency, and write or verify '
                    'an exact environment lock.')
    parser.add_argument('--root', default=None,
                        help='repository root (default: this file\'s directory)')
    parser.add_argument('--json', action='store_true',
                        help='print the report as strict JSON')
    lock_group = parser.add_mutually_exclusive_group()
    lock_group.add_argument(
        '--write-lock', metavar='PATH',
        help='write a deterministic environment lock to PATH and exit')
    lock_group.add_argument(
        '--check-lock', metavar='PATH',
        help='verify an environment lock against the current environment')
    args = parser.parse_args(argv)

    if args.write_lock:
        try:
            lock = write_environment_lock(args.write_lock, args.root)
        except (ContractUnavailable, OSError, ValueError, KeyError) as exc:
            print(f'[dependency-contract] cannot write lock: {exc}')
            return 2
        payload = {
            'ok': True,
            'path': str(args.write_lock),
            'lock_sha256': lock['lock_sha256'],
            'n_packages': len(lock.get('installed_packages') or {}),
            'declarations': {
                name: row.get('sha256')
                for name, row in (lock.get('declarations') or {}).items()
            },
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
        else:
            print(f"[dependency-contract] lock written: {args.write_lock}")
            print(f"  lock_sha256            {payload['lock_sha256']}")
            print(f"  packages               {payload['n_packages']}")
            for name, digest in payload['declarations'].items():
                print(f"  {name:<22} {digest}")
        return 0

    if args.check_lock:
        try:
            with open(args.check_lock, encoding='utf-8') as fh:
                lock = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f'[dependency-contract] cannot read lock: {exc}')
            return 2
        try:
            report = verify_environment_lock(lock, args.root)
        except ContractUnavailable as exc:
            print(f'[dependency-contract] unavailable: {exc}')
            return 2
        if args.json:
            print(json.dumps(report, sort_keys=True, allow_nan=False))
        else:
            _print_lock_report(report)
        if report.get('status') == 'ok':
            return 0
        if report.get('status') == 'drift':
            return 1
        return 2

    try:
        report = check_contract(args.root)
    except ContractUnavailable as exc:
        print(f'[dependency-contract] unavailable: {exc}')
        return 2
    if args.json:
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    else:
        status = 'PASS' if report['ok'] else 'FAIL'
        print(f'[dependency-contract] {status}')
        for name, check in report['checks'].items():
            mark = 'ok' if check.get('ok') else 'FAIL'
            print(f'  {name:<22} {mark}')
        for error in report['errors']:
            print(f'  error: {error}')
        for warning in report['warnings']:
            print(f'  warning: {warning}')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
