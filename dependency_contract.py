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
import json
import pathlib
import sys
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Check pyproject.toml / requirements.txt / '
                    'requirements-ci.txt consistency.')
    parser.add_argument('--root', default=None,
                        help='repository root (default: this file\'s directory)')
    parser.add_argument('--json', action='store_true',
                        help='print the report as strict JSON')
    args = parser.parse_args(argv)
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
