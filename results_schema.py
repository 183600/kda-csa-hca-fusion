"""Versioned result-file envelope shared by every experiment in this repo.

Motivation
----------
Historically the ``results/*.json`` files used two incompatible top-level
shapes (README "Known schema inconsistency"):

* a **bare array** ``[{row}, ...]`` — Exp 1/2/3/5/6 plus ``summary.json``;
* an **envelope object** ``{"metadata": {...}, "results": [...]}`` —
  Exp 4 only, with an informal integer ``metadata.schema_version``.

Downstream consumers (``make_figures.load``, ``run_roofline``, ad-hoc
analysis scripts) had to special-case both shapes, and nothing recorded
*which* schema revision produced a file, so a future format change could
silently corrupt analysis. This module makes the envelope the single
canonical on-disk shape, with an explicit semantic version:

    {
      "schema_version": "1.0",
      "kind": "exp7_roofline",          # which experiment produced this
      "created_utc": "2026-...Z",
      "metadata": {...},                 # provenance / caveats
      "results": [{...}, ...]            # the per-row records
    }

Backward compatibility
----------------------
``parse`` accepts and classifies three historical shapes:

===============  ==============================================  =======
on-disk shape    example                                         version
===============  ==============================================  =======
bare array       ``results/exp2_benchmark.json``                 ``0.0``
legacy envelope  ``results/exp4_mqar.json``                      ``1``
versioned env.   files written by this module                    ``1.0``
===============  ==============================================  =======

* Legacy ``0.0`` files are migrated in-memory by :func:`parse` /
  :func:`load` (they simply lack a version); they are never rewritten
  implicitly. Use :func:`write` to persist a versioned copy.
* A file whose MAJOR version is newer than :data:`CURRENT_SCHEMA_VERSION`
  raises :class:`SchemaVersionError` from the strict :func:`load` instead
  of being silently misread. :func:`load_or_empty` degrades to ``({}, [])``
  (for figure code that must never crash the pipeline) and prints a loud
  stderr diagnostic.

CLI
---
``python results_schema.py results/exp2_benchmark.json`` prints a
version report per file. ``--check`` exits non-zero if any file is
incompatible (future major version) or malformed, so a CI job can gate on
it. With no arguments it scans ``results/*.json``.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import sanitize_for_json, write_json_atomic  # noqa: E402

CURRENT_SCHEMA_VERSION = "1.0"


class SchemaError(ValueError):
    """Base class for result-schema violations."""


class SchemaVersionError(SchemaError):
    """The file's schema version is incompatible with this reader."""


class SchemaFormatError(SchemaError):
    """The file has neither a bare-array nor an envelope shape."""


def parse_version(value) -> tuple[int, int]:
    """Parse ``"1.0"`` / ``"1"`` / ``1`` into a ``(major, minor)`` tuple.

    Accepts the informal integer versions used historically by
    ``run_quality.py`` (``metadata.schema_version: 1``). Raises
    :class:`SchemaVersionError` on anything that cannot be interpreted as
    a non-negative numeric version.
    """
    if isinstance(value, bool) or value is None:
        raise SchemaVersionError(f'unrecognized schema_version: {value!r}')
    if isinstance(value, int):
        if value < 0:
            raise SchemaVersionError(f'unrecognized schema_version: {value!r}')
        return (value, 0)
    if isinstance(value, float):
        # JSON permits NaN/Infinity literals; math.isfinite guards both the
        # int(value) conversions (NaN -> ValueError, inf -> OverflowError)
        # and any non-numeric edge, so the documented SchemaVersionError
        # contract holds.
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise SchemaVersionError(f'unrecognized schema_version: {value!r}')
        return (int(value), 0)
    if isinstance(value, str):
        parts = value.strip().split('.')
        try:
            nums = [int(p) for p in parts]
        except ValueError as e:
            raise SchemaVersionError(
                f'unrecognized schema_version: {value!r}') from e
        if not nums or any(n < 0 for n in nums):
            raise SchemaVersionError(f'unrecognized schema_version: {value!r}')
        return (nums[0], nums[1] if len(nums) > 1 else 0)
    raise SchemaVersionError(f'unrecognized schema_version: {value!r}')


def _version_str(value) -> str:
    major, minor = parse_version(value)
    return f'{major}.{minor}'


def detect_version(data) -> str:
    """Return the schema version string of a parsed JSON payload.

    * bare array -> ``"0.0"`` (legacy, pre-versioning);
    * dict envelope -> top-level ``schema_version`` if present, else
      ``metadata.schema_version`` (Exp 4's informal integer), else
      ``"0.0"``.
    """
    if isinstance(data, list):
        return '0.0'
    if isinstance(data, dict):
        if 'schema_version' in data:
            return _version_str(data['schema_version'])
        meta = data.get('metadata')
        if isinstance(meta, dict) and 'schema_version' in meta:
            return _version_str(meta['schema_version'])
        return '0.0'
    raise SchemaFormatError(
        f'result payload must be a list or dict, got {type(data).__name__}')


def is_compatible(version) -> bool:
    """True iff ``version`` can be read by this implementation.

    Compatible = same major version as :data:`CURRENT_SCHEMA_VERSION`, or
    the legacy ``0.x`` series (pre-versioning files are always readable;
    they are migrated in memory).
    """
    major, _ = parse_version(version)
    cur_major, _ = parse_version(CURRENT_SCHEMA_VERSION)
    return major == 0 or major == cur_major


def parse(data) -> tuple[dict, list, str]:
    """Split a payload into ``(metadata, rows, version)``.

    Raises :class:`SchemaVersionError` when the payload declares a FUTURE
    major version (unknown format — refusing to guess), and
    :class:`SchemaFormatError` when the payload is neither a list nor an
    envelope dict containing a ``results`` list.
    """
    if isinstance(data, list):
        return {}, data, '0.0'
    if isinstance(data, dict):
        version = detect_version(data)
        major, _ = parse_version(version)
        cur_major, _ = parse_version(CURRENT_SCHEMA_VERSION)
        if major > cur_major:
            raise SchemaVersionError(
                f'result file declares schema_version={version}, newer than '
                f'this reader supports ({CURRENT_SCHEMA_VERSION}); refusing '
                f'to parse it with possibly-incompatible semantics.')
        rows = data.get('results')
        if not isinstance(rows, list):
            raise SchemaFormatError(
                "envelope dict must contain a 'results' list, found "
                f'{type(rows).__name__}')
        meta = data.get('metadata') or {}
        if not isinstance(meta, dict):
            raise SchemaFormatError(
                f"envelope 'metadata' must be a dict, found "
                f'{type(meta).__name__}')
        return meta, rows, version
    raise SchemaFormatError(
        f'result payload must be a list or dict, got {type(data).__name__}')


def envelope(rows: list, *, kind: str, metadata: dict | None = None,
             created_utc: str | None = None) -> dict:
    """Build a current-version envelope around ``rows``.

    ``kind`` identifies the producing experiment (e.g. ``"exp7_roofline"``)
    so tooling can validate that a file holds what its name claims.
    """
    meta = dict(metadata) if metadata else {}
    meta.setdefault('created_utc', created_utc or _dt.datetime.now(
        _dt.timezone.utc).isoformat())
    return {
        'schema_version': CURRENT_SCHEMA_VERSION,
        'kind': kind,
        'created_utc': meta['created_utc'],
        'metadata': meta,
        'results': rows,
    }


def normalize(data, *, kind: str, metadata: dict | None = None) -> dict:
    """Migrate any accepted payload to a current-version envelope (pure).

    Legacy metadata (e.g. Exp 4's provenance block) is preserved; a
    call-site ``metadata`` argument overrides same-named legacy keys.
    """
    legacy_meta, rows, _version = parse(data)
    merged = dict(legacy_meta)
    if metadata:
        merged.update(metadata)
    merged.pop('schema_version', None)
    return envelope(rows, kind=kind, metadata=merged)


def load(path: str) -> tuple[dict, list]:
    """Strict load: ``(metadata, rows)``.

    Raises FileNotFoundError / JSONDecodeError / SchemaError. Use
    :func:`load_or_empty` when a missing or incompatible file must not
    abort the caller.
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    meta, rows, _version = parse(data)
    return meta, rows


def load_or_empty(path: str, *, context: str = '') -> tuple[dict, list]:
    """Forgiving load: ``({}, [])`` on any failure, with stderr diagnostics.

    Mirrors the historical ``make_figures.load`` contract (a broken file
    must not crash the figure pipeline) while still surfacing the reason.
    An incompatible FUTURE schema version is reported loudly because
    silently returning no data for a newer file risks silent data loss.
    """
    label = f'{context}: ' if context else ''
    try:
        return load(path)
    except FileNotFoundError:
        print(f'[results_schema] {label}{path} not found; skipping',
              file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f'[results_schema] {label}{path} is malformed: {e}; skipping',
              file=sys.stderr)
    except SchemaError as e:
        print(f'[results_schema] {label}{path}: {e}', file=sys.stderr)
    return {}, []


def write(path: str, rows: list, *, kind: str, metadata: dict | None = None,
          indent: int = 2) -> None:
    """Atomically write ``rows`` as a current-version envelope.

    Non-finite floats are sanitized to ``null`` first (strict RFC 8259
    output, consistent with every other writer in this repo).
    """
    payload = normalize({'results': rows}, kind=kind, metadata=metadata)
    write_json_atomic(sanitize_for_json(payload), path, indent=indent,
                      allow_nan=False)


def _describe(path: str) -> tuple[str, int, str]:
    """Return ``(version, n_rows, status)`` for the CLI report."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return '?', 0, 'missing'
    except json.JSONDecodeError as e:
        return '?', 0, f'malformed ({e})'
    try:
        version = detect_version(data)
        _meta, rows, _ = parse(data)
    except SchemaError as e:
        try:
            version = detect_version(data)
        except SchemaError:
            version = '?'
        return version, 0, f'incompatible ({e})'
    status = 'ok' if is_compatible(version) else 'INCOMPATIBLE'
    return version, len(rows), status


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = '--check' in argv
    argv = [a for a in argv if a != '--check']
    paths = argv or sorted(glob.glob(os.path.join('results', '*.json')))
    if not paths:
        print('[results_schema] no result files found')
        return 0
    bad = 0
    for p in paths:
        version, n_rows, status = _describe(p)
        print(f'{version:>5}  {n_rows:>6} rows  {status:<12}  {p}')
        if status != 'ok':
            bad += 1
    if check and bad:
        print(f'[results_schema] {bad} file(s) failed the schema check')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
