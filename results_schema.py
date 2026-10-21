"""Versioned result-file envelope shared by every experiment in this repo.

Motivation
----------
``results/*.json`` files appear in two top-level shapes
(README "Known schema inconsistency"):

* a **bare array** ``[{row}, ...]`` — Exp 1/2/3/5/6 plus ``summary.json``;
* an **envelope object** ``{"metadata": {...}, "results": [...]}`` —
  Exp 4 only, with an informal integer ``metadata.schema_version``.

Downstream consumers (``make_figures.load``, ``run_roofline``, ad-hoc
analysis scripts) would otherwise have to special-case both shapes, and
without a recorded schema revision a format change could silently
corrupt analysis. This module defines the envelope as the single
canonical on-disk shape, with an explicit semantic version:

    {
      "schema_version": "1.0",
      "kind": "exp7_roofline",          # which experiment produced this
      "created_utc": "2026-...Z",
      "metadata": {...},                 # provenance / caveats
      "results": [{...}, ...]            # the per-row records
    }

Per-kind row contracts
----------------------
Structural shape (envelope + version) says nothing about whether the
rows make sense. A partial matrix (7 of 8 groups), a contrast whose
``mean_diff`` disagrees with the per-seed rows, or an Exp 6 row whose
``uses_incremental_cache`` flag contradicts its operator are all
schema-valid yet silently corrupt downstream analysis. :func:`validate_rows`
therefore applies a per-``kind`` *contract* (registered in
:data:`ROW_CONTRACTS`) that checks:

* required fields and their types;
* uniqueness of the natural key ((group, n_kv), (op, prefill_len), ...);
* internal arithmetic (``mean_acc`` recomputed from ``per_seed``);
* run-level completeness (the group set recorded in the envelope's
  ``metadata.expected_groups`` must all be present);
* metadata invariants (Exp 6's hybrid rows MUST be flagged as using the
  incremental cache — the caches are wired in every hybrid stack
  regardless of the memory route).

:func:`write_checked` validates before writing and raises
:class:`RowContractError`, so a producer that violates the contract
fails the experiment loudly instead of persisting a misleading result
file. ``--check-rows`` extends
the CLI ``--check`` mode to freshly written kind-tagged files; unknown
kinds (and legacy version-0 files without a ``kind``) pass untouched.

Backward compatibility
----------------------
``parse`` accepts and classifies three on-disk shapes:

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
it; ``--check-rows`` additionally enforces the per-kind row contracts on
kind-tagged files. With no arguments it scans ``results/*.json``.
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


class RowContractError(SchemaError):
    """The payload's rows violate the registered contract for their kind."""


def parse_version(value) -> tuple[int, int]:
    """Parse ``"1.0"`` / ``"1"`` / ``1`` into a ``(major, minor)`` tuple.

    Accepts the informal integer versions written by ``run_quality.py``
    (``metadata.schema_version: 1``). Raises
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


# ---------------------------------------------------------------------------
# Per-kind row contracts
# ---------------------------------------------------------------------------
# A contract is a pure ``fn(rows, metadata) -> list[str]`` returning
# human-readable violations (empty = conforming). Register one per
# ``kind`` with the decorator; unknown kinds are left unvalidated so a
# new experiment can start writing envelopes before its contract exists.

ROW_CONTRACTS: dict = {}

# Hybrid rows record the RESOLVED layout as a hyphen-joined uppercase
# string (e.g. 'KDA-KDA-KDA-CSA-HCA'; see ops_fused.HybridAttentionStack
# .layout_str and LAYOUT_KINDS).
_LAYOUT_TOKENS = frozenset({'KDA', 'MAMBA2', 'RWKV7', 'CSA', 'HCA',
                            'LINEAR', 'SOFTMAX'})
_MEMORY_TOKENS = frozenset({'KDA', 'MAMBA2', 'RWKV7'})
_EXP6_HYBRID_OPS = frozenset({'hybrid', 'hybrid_mamba2', 'hybrid_rwkv7'})
_EXP6_CACHE_OPS = frozenset({'csa', 'hca'}) | _EXP6_HYBRID_OPS
_EXP6_PREFILL_BUILDS = frozenset({
    'batched_populate_layer_by_layer',
    'batched_append_from_naive_output',
    'native_operator_state',
})
# ``precision_sweep.py`` aggregate rows.
_PRECISION_MODES = frozenset({'explicit', 'autocast'})
_PRECISION_DTYPES = frozenset({'float16', 'bfloat16', 'float32', 'float64'})
_PRECISION_VERDICTS = frozenset({'pass', 'fail'})


def register_row_contract(kind: str):
    """Decorator registering ``fn(rows, metadata) -> list[str]`` for ``kind``."""
    def _decorator(fn):
        ROW_CONTRACTS[kind] = fn
        return fn
    return _decorator


def _finite_number(value) -> bool:
    """True for a real (non-bool) finite int/float."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _ci_ratio_matches(ci, ci_ratio) -> bool:
    """Whether ``ci_ratio`` is the exponentiation of ``ci``.

    ``math.exp`` raises ``OverflowError`` above ~709.78, so the
    exponentiation is computed defensively: a CI bound that overflows
    makes the row a CONTRACT VIOLATION (it cannot be the exponentiation
    of the reported bound), never an exception that aborts the validator.
    """
    for bound, ratio in zip(ci, ci_ratio):
        try:
            want = math.exp(float(bound))
        except OverflowError:
            return False
        tol = 1e-9 * max(1.0, abs(float(ratio)))
        if abs(float(ratio) - want) > tol:
            return False
    return True


def _recomputed_mean(per_seed: list) -> tuple[float | None, int]:
    """Mean over the per-seed rows that qualify for aggregation.

    Mirrors ``run_ablation.eval_group_multi_seed``'s ``ok_per_seed``
    filter: a seed contributes iff it has no ``error`` and BOTH
    ``final_acc`` and ``final_loss`` are finite.
    """
    accs = []
    for s in per_seed:
        if not isinstance(s, dict) or 'error' in s:
            continue
        acc, loss = s.get('final_acc'), s.get('final_loss')
        if _finite_number(acc) and _finite_number(loss):
            accs.append(float(acc))
    if not accs:
        return None, 0
    return sum(accs) / len(accs), len(accs)


@register_row_contract('exp5b_ablation_matrix')
def _validate_matrix_rows(rows: list, metadata: dict) -> list[str]:
    """Eight-group matrix: unique (group, n_kv), self-consistent aggregates.

    Also enforces run-level completeness when the writer recorded
    ``metadata['expected_groups']`` (the group ids the preset promised):
    every expected group id must appear in the rows. A crash that writes
    a survivor-only matrix is exactly the failure this catches.
    """
    out: list[str] = []
    seen: set = set()
    present: set = set()
    meta = metadata if isinstance(metadata, dict) else {}
    expected = meta.get('expected_groups')
    total_layers = meta.get('total_layers')
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        group, memory, n_kv = row.get('group'), row.get('memory'), row.get('n_kv')
        if not isinstance(group, str) or not group:
            out.append(f"{where}: 'group' must be a non-empty string, got {group!r}")
        if not isinstance(memory, str) or not memory:
            out.append(f"{where}: 'memory' must be a non-empty string, got {memory!r}")
        if not _positive_int(n_kv):
            out.append(f"{where}: 'n_kv' must be a positive int, got {n_kv!r}")
        if isinstance(group, str) and _positive_int(n_kv):
            key = (group, n_kv)
            if key in seen:
                out.append(f'{where}: duplicate (group, n_kv)={key!r}')
            seen.add(key)
            present.add(group)
        if 'error' in row:
            # A failed group still carries its identity fields; the numeric
            # checks below only apply to successful rows.
            continue
        per_seed = row.get('per_seed')
        if not isinstance(per_seed, list):
            out.append(f"{where}: 'per_seed' must be a list, got {type(per_seed).__name__}")
            continue
        recomputed, n_finite = _recomputed_mean(per_seed)
        mean = row.get('mean_acc')
        if not _finite_number(mean):
            out.append(f"{where}: 'mean_acc' must be finite, got {mean!r}")
        elif recomputed is not None:
            tol = 1e-9 * max(1.0, abs(float(mean)))
            if abs(recomputed - float(mean)) > tol:
                out.append(
                    f"{where}: mean_acc={mean!r} != mean(per_seed final_acc)="
                    f'{recomputed!r}')
        n_ok = row.get('n_seeds_ok')
        if isinstance(n_ok, int) and not isinstance(n_ok, bool) \
                and n_ok != n_finite:
            out.append(
                f'{where}: n_seeds_ok={n_ok} but {n_finite} finite '
                f'per-seed final_acc value(s)')
        layout = row.get('layout')
        if not isinstance(layout, str) or not layout:
            out.append(
                f'{where}: layout must be the non-empty resolved layout '
                f'string (e.g. KDA-KDA-KDA-CSA-HCA), got {layout!r}')
        else:
            tokens = layout.split('-')
            bad_tokens = sorted({t for t in tokens if t not in _LAYOUT_TOKENS})
            if bad_tokens:
                out.append(
                    f'{where}: layout has unknown token(s) {bad_tokens}; '
                    f'expected tokens from {sorted(_LAYOUT_TOKENS)}')
            elif isinstance(memory, str):
                stray_memory = sorted(
                    {t.lower() for t in tokens
                     if t in _MEMORY_TOKENS and t.lower() != memory})
                if stray_memory:
                    out.append(
                        f'{where}: layout carries memory token(s) '
                        f'{stray_memory} but memory={memory!r} '
                        f'(one memory route per stack)')
        n_params, n_layers = row.get('n_params'), row.get('n_layers')
        if not _positive_int(n_params):
            out.append(f"{where}: 'n_params' must be a positive int, got {n_params!r}")
        if not _positive_int(n_layers):
            out.append(f"{where}: 'n_layers' must be a positive int, got {n_layers!r}")
        elif total_layers is not None and n_layers != total_layers:
            out.append(
                f'{where}: n_layers={n_layers} != metadata.total_layers='
                f'{total_layers} (the design is depth-controlled)')
    if isinstance(expected, list) and expected:
        missing = sorted(set(expected) - present)
        if missing:
            out.append(
                f'matrix incomplete: expected group(s) {missing} are absent '
                f'from the {len(present)} group(s) present')
    return out


@register_row_contract('exp5b_ablation_matrix_contrasts')
def _validate_contrast_rows(rows: list, metadata: dict) -> list[str]:
    """Paired contrasts: finite effects, >= 2 common seeds, unique keys."""
    out: list[str] = []
    seen: set = set()
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        n_kv = row.get('n_kv')
        if not _positive_int(n_kv):
            out.append(f"{where}: 'n_kv' must be a positive int, got {n_kv!r}")
        for key in ('memory', 'contrast'):
            if not isinstance(row.get(key), str) or not row.get(key):
                out.append(f"{where}: '{key}' must be a non-empty string, "
                           f'got {row.get(key)!r}')
        if not _finite_number(row.get('mean_diff')):
            out.append(f"{where}: 'mean_diff' must be finite, got {row.get('mean_diff')!r}")
        t_stat = row.get('t_stat')
        if t_stat is not None and not _finite_number(t_stat):
            out.append(f"{where}: 't_stat' must be null or finite, got {t_stat!r}")
        n_common = row.get('n_common_seeds')
        if not (isinstance(n_common, int) and not isinstance(n_common, bool)
                and n_common >= 2):
            out.append(
                f"{where}: 'n_common_seeds' must be >= 2 for a paired "
                f'contrast, got {n_common!r}')
        if row.get('metric') != 'final_acc':
            out.append(f"{where}: metric must be 'final_acc', got {row.get('metric')!r}")
        if row.get('direction') != 'higher_is_better':
            out.append(
                f"{where}: direction must be 'higher_is_better', got "
                f"{row.get('direction')!r}")
        key = (n_kv, row.get('memory'), row.get('contrast'))
        if all(part is not None for part in key):
            if key in seen:
                out.append(f'{where}: duplicate contrast key {key!r}')
            seen.add(key)
    return out


@register_row_contract('exp6_decoding')
def _validate_exp6_rows(rows: list, metadata: dict) -> list[str]:
    """Incremental-decoding benchmark: complete cells + honest cache flags.

    The cache flags are not cosmetic: every ``hybrid*`` row runs the
    CSA/HCA incremental caches in each sub-layer (regardless of memory
    route), and every standalone ``csa``/``hca`` row routes prefill
    through the batched cache population. A row that says otherwise is a
    metadata bug that would mislead the fairness discussion.
    """
    out: list[str] = []
    seen: set = set()
    meta = metadata if isinstance(metadata, dict) else {}
    prefill_lens = meta.get('prefill_lens')
    ops_expected = meta.get('ops')
    cells: set = set()
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        op, plen = row.get('op'), row.get('prefill_len')
        if not isinstance(op, str) or not op:
            out.append(f"{where}: 'op' must be a non-empty string, got {op!r}")
        if not _positive_int(plen):
            out.append(f"{where}: 'prefill_len' must be a positive int, got {plen!r}")
        if not _positive_int(row.get('n_layers')):
            out.append(f"{where}: 'n_layers' must be a positive int, "
                       f"got {row.get('n_layers')!r}")
        if isinstance(op, str) and _positive_int(plen):
            key = (op, plen)
            if key in seen:
                out.append(f'{where}: duplicate (op, prefill_len)={key!r}')
            seen.add(key)
            cells.add(key)
        if 'error' in row:
            continue
        for key in ('prefill_ms', 'mean_decode_ms_per_token',
                    'median_decode_ms_per_token'):
            value = row.get(key)
            if not (_finite_number(value) and float(value) > 0):
                out.append(f'{where}: {key!r} must be finite and > 0, got {value!r}')
        for key in ('n_decode', 'repeats'):
            if not _positive_int(row.get(key)):
                out.append(f"{where}: '{key}' must be a positive int, "
                           f"got {row.get(key)!r}")
        peak = row.get('peak_mem_MB')
        if peak is not None and not (_finite_number(peak) and float(peak) >= 0):
            out.append(f"{where}: 'peak_mem_MB' must be null or finite >= 0, "
                       f'got {peak!r}')
        # Decode-state certificate fields (optional for legacy rows, but
        # checked when present): valid bytes must be a positive int,
        # allocated >= valid, and the breakdown must be a mapping.
        state_valid = row.get('decode_state_bytes')
        state_alloc = row.get('decode_state_allocated_bytes')
        for key, value in (('decode_state_bytes', state_valid),
                           ('decode_state_allocated_bytes', state_alloc)):
            if value is not None and not _positive_int(value):
                out.append(f"{where}: {key!r} must be null or a positive int, "
                           f'got {value!r}')
        if _positive_int(state_valid) and _positive_int(state_alloc) \
                and state_alloc < state_valid:
            out.append(
                f"{where}: decode_state_allocated_bytes={state_alloc} < "
                f'decode_state_bytes={state_valid}')
        breakdown = row.get('decode_state_breakdown')
        if breakdown is not None and not isinstance(breakdown, dict):
            out.append(f"{where}: 'decode_state_breakdown' must be an object, "
                       f'got {type(breakdown).__name__}')
        flag = row.get('uses_incremental_cache')
        build = row.get('prefill_cache_build')
        if not isinstance(flag, bool):
            out.append(f"{where}: 'uses_incremental_cache' must be a bool, "
                       f'got {flag!r}')
        if build not in _EXP6_PREFILL_BUILDS:
            out.append(f"{where}: 'prefill_cache_build'={build!r} not in "
                       f'{sorted(_EXP6_PREFILL_BUILDS)}')
        if isinstance(op, str) and isinstance(flag, bool):
            if op in _EXP6_CACHE_OPS and not flag:
                out.append(
                    f'{where}: op={op!r} runs the incremental cache but '
                    f'uses_incremental_cache=False')
            if op not in _EXP6_CACHE_OPS and flag:
                out.append(
                    f'{where}: op={op!r} carries fixed native state but '
                    f'uses_incremental_cache=True')
            if op in _EXP6_HYBRID_OPS \
                    and build != 'batched_populate_layer_by_layer':
                out.append(
                    f'{where}: hybrid op {op!r} populates per-layer caches '
                    f'during prefill but prefill_cache_build={build!r}')
            if op in ('csa', 'hca') \
                    and build != 'batched_append_from_naive_output':
                out.append(
                    f'{where}: standalone {op!r} prefill builds the cache '
                    f'from naive projections but prefill_cache_build={build!r}')
            if op not in _EXP6_CACHE_OPS and build != 'native_operator_state':
                out.append(
                    f'{where}: op {op!r} uses its native state path but '
                    f'prefill_cache_build={build!r}')
    if isinstance(prefill_lens, list) and prefill_lens:
        stray = sorted({plen for _op, plen in cells} - set(prefill_lens))
        if stray:
            out.append(f'prefill_len value(s) {stray} not in metadata.prefill_lens')
    if isinstance(ops_expected, list) and ops_expected:
        stray_ops = sorted({op for op, _plen in cells} - set(ops_expected))
        if stray_ops:
            out.append(f'op value(s) {stray_ops} not in metadata.ops')
        missing_cells = sorted(
            (op, plen) for op in ops_expected for plen in (prefill_lens or [])
            if (op, plen) not in cells
        )
        if missing_cells:
            out.append(f'missing (op, prefill_len) cell(s): {missing_cells[:6]}')
    return out


@register_row_contract('precision_sweep')
def _validate_precision_sweep_rows(rows: list, metadata: dict) -> list[str]:
    """Mixed-precision sweep aggregates: self-consistent verdicts only.

    Every group is one (case, path, dtype, mode) cell with per-seed
    statistics. The contract checks the arithmetic a downstream reader
    relies on (max >= median, n_ok <= n_seeds, worst_seed present iff
    there is a finite seed) and, most importantly, that ``verdict`` is
    CONSISTENT with the recorded flags: a 'pass' whose budget flag is
    False (or whose n_ok < n_seeds) is a producer bug that would let a
    broken precision claim be archived.
    """
    out: list[str] = []
    seen: set = set()
    meta = metadata if isinstance(metadata, dict) else {}
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        for key in ('case', 'path'):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                out.append(f"{where}: '{key}' must be a non-empty string, got {value!r}")
        dtype, mode = row.get('dtype'), row.get('mode')
        if dtype not in _PRECISION_DTYPES:
            out.append(f"{where}: 'dtype'={dtype!r} not in "
                       f'{sorted(_PRECISION_DTYPES)}')
        if mode not in _PRECISION_MODES:
            out.append(f"{where}: 'mode'={mode!r} not in "
                       f'{sorted(_PRECISION_MODES)}')
        if not _positive_int(row.get('seq_len')):
            out.append(f"{where}: 'seq_len' must be a positive int, "
                       f"got {row.get('seq_len')!r}")
        n_seeds, n_ok = row.get('n_seeds'), row.get('n_ok')
        if not _positive_int(n_seeds):
            out.append(f"{where}: 'n_seeds' must be a positive int, "
                       f"got {n_seeds!r}")
        if not isinstance(n_ok, int) or isinstance(n_ok, bool) \
                or n_ok < 0 or (isinstance(n_seeds, int) and n_ok > n_seeds):
            out.append(f"{where}: 'n_ok'={n_ok!r} must be an int in "
                       f'[0, n_seeds={n_seeds!r}]')
        budget = row.get('budget_eps')
        if not (_finite_number(budget) and float(budget) > 0):
            out.append(f"{where}: 'budget_eps' must be finite and > 0, "
                       f'got {budget!r}')
        for key in ('out_eps_max', 'out_eps_median', 'state_eps_max',
                    'state_eps_median'):
            value = row.get(key)
            if value is not None and not (_finite_number(value)
                                          and float(value) >= 0):
                out.append(f'{where}: {key!r} must be null or finite >= 0, '
                           f'got {value!r}')
        for hi, lo in (('out_eps_max', 'out_eps_median'),
                       ('state_eps_max', 'state_eps_median')):
            if _finite_number(row.get(hi)) and _finite_number(row.get(lo)) \
                    and float(row[hi]) + 1e-12 < float(row[lo]):
                out.append(f'{where}: {hi!r} < {lo!r}')
        seed = row.get('worst_seed')
        if seed is not None and (isinstance(seed, bool)
                                 or not isinstance(seed, int) or seed < 0):
            out.append(f"{where}: 'worst_seed' must be null or a "
                       f'non-negative int, got {seed!r}')
        if n_ok == 0 and seed is not None:
            out.append(f'{where}: worst_seed set but n_ok=0 (no finite seed)')
        flags = {}
        for key in ('out_budget_ok', 'state_budget_ok', 'dtype_contract_ok',
                    'finite_ok'):
            value = row.get(key)
            flags[key] = value
            if not isinstance(value, bool):
                out.append(f"{where}: {key!r} must be a bool, got {value!r}")
        verdict = row.get('verdict')
        if verdict not in _PRECISION_VERDICTS:
            out.append(f"{where}: 'verdict'={verdict!r} not in "
                       f'{sorted(_PRECISION_VERDICTS)}')
        if verdict == 'pass':
            if n_ok != n_seeds:
                out.append(f'{where}: verdict=pass but n_ok={n_ok!r} != '
                           f'n_seeds={n_seeds!r}')
            if all(isinstance(v, bool) for v in flags.values()) \
                    and not all(flags.values()):
                out.append(f'{where}: verdict=pass but flags {flags!r} '
                           f'are not all True')
            if row.get('out_eps_max') is None or row.get('state_eps_max') is None:
                out.append(f'{where}: verdict=pass requires finite '
                           f'out_eps_max/state_eps_max')
        elif verdict == 'fail' and n_ok == n_seeds \
                and all(isinstance(v, bool) and v for v in flags.values()) \
                and row.get('out_eps_max') is not None \
                and row.get('state_eps_max') is not None:
            out.append(f'{where}: verdict=fail but every recorded check '
                       f'passed')
        if isinstance(dtype, str) and mode in _PRECISION_MODES \
                and isinstance(row.get('case'), str) \
                and isinstance(row.get('path'), str):
            key = (row['case'], row['path'], dtype, mode)
            if key in seen:
                out.append(f'{where}: duplicate group key {key!r}')
            seen.add(key)
    expected = meta.get('n_groups')
    if _positive_int(expected) and expected != len(rows):
        out.append(f'metadata.n_groups={expected} but payload has '
                   f'{len(rows)} row(s)')
    return out


# ``run_indexer_selector.py`` (Exp 14) row sections. The experiment file
# is a cost + safety record for ``ops_csa._canonical_topk_indices``: the
# parity gate must be CLEAN (any mismatch or crash refuses archival) and
# every timing row must be finite, positive, and tagged with the dispatch
# branch it exercised, so a future edit that silently changes the
# selection kernel fails the contract instead of archiving plausible
# numbers.
_EXP14_SECTIONS = frozenset({
    'parity', 'selector_latency', 'indexer_latency', 'sink_guard'})
_EXP14_DISPATCH = frozenset({'sort', 'boundary', 'empty'})
_EXP14_TIMING_FIELDS = {
    'selector_latency': ('shipped_ms', 'historical_sort_ms',
                         'torch_topk_ms'),
    'indexer_latency': ('indexer_shipped_ms', 'indexer_historical_sort_ms',
                        'step_shipped_ms', 'step_historical_sort_ms',
                        'selector_shipped_ms', 'selector_historical_sort_ms'),
}


@register_row_contract('exp14_indexer_selector')
def _validate_exp14_rows(rows: list, metadata: dict) -> list[str]:
    """Indexer-selector rows: clean parity gate + finite positive timings.

    Checks, per section:

    * ``parity``: positive ``n_cases`` and EXACTLY zero ``mismatches`` /
      ``crashes`` -- a non-zero parity row means the shipped selector
      disagrees with the reference or crashed, exactly the failure the
      experiment exists to catch;
    * ``selector_latency`` / ``indexer_latency``: positive integer
      geometry, a ``dispatch`` tag in {'sort', 'boundary', 'empty'},
      finite positive medians, and ``speedup == historical / shipped``
      (the two recorded medians must reproduce the recorded ratio);
    * ``sink_guard``: ``finite`` is True and the recorded max-abs values
      are finite and >= 0 (the NaN / degenerate-config guard).
    """
    out: list[str] = []
    seen: set = set()
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        section = row.get('section')
        if section not in _EXP14_SECTIONS:
            out.append(f'{where}: section={section!r} not in '
                       f'{sorted(_EXP14_SECTIONS)}')
            continue
        if section == 'parity':
            case = row.get('case')
            if not isinstance(case, str) or not case:
                out.append(
                    f"{where}: 'case' must be a non-empty string, got {case!r}")
            if not _positive_int(row.get('n_cases')):
                out.append(f"{where}: 'n_cases' must be a positive int, "
                           f"got {row.get('n_cases')!r}")
            for key in ('mismatches', 'crashes'):
                value = row.get(key)
                if value != 0:
                    out.append(f'{where}: {key}={value!r} must be exactly 0 '
                               f'(the parity gate is the experiment)')
            key = ('parity', case)
        elif section in ('selector_latency', 'indexer_latency'):
            for fld in ('n_blocks', 'topk'):
                if not _positive_int(row.get(fld)):
                    out.append(f"{where}: {fld!r} must be a positive int, "
                               f"got {row.get(fld)!r}")
            if section == 'indexer_latency' \
                    and not _positive_int(row.get('T')):
                out.append(f"{where}: 'T' must be a positive int, "
                           f"got {row.get('T')!r}")
            if section == 'selector_latency' \
                    and not _positive_int(row.get('rows')):
                out.append(f"{where}: 'rows' must be a positive int, "
                           f"got {row.get('rows')!r}")
            dispatch = row.get('dispatch')
            if dispatch not in _EXP14_DISPATCH:
                out.append(f'{where}: dispatch={dispatch!r} not in '
                           f'{sorted(_EXP14_DISPATCH)}')
            for fld in _EXP14_TIMING_FIELDS[section]:
                value = row.get(fld)
                if not (_finite_number(value) and float(value) > 0):
                    out.append(f"{where}: {fld!r} must be finite and > 0, "
                               f'got {value!r}')
            for sp, (hist, shipped) in {
                'speedup_vs_sort': ('historical_sort_ms', 'shipped_ms'),
                'indexer_speedup': ('indexer_historical_sort_ms',
                                    'indexer_shipped_ms'),
                'step_speedup': ('step_historical_sort_ms',
                                 'step_shipped_ms'),
            }.items():
                if sp not in row:
                    continue
                value = row.get(sp)
                if not (_finite_number(value) and float(value) > 0):
                    out.append(f"{where}: {sp!r} must be finite and > 0, "
                               f'got {value!r}')
                a, b = row.get(hist), row.get(shipped)
                if _finite_number(a) and _finite_number(b) and float(b) > 0:
                    want = float(a) / float(b)
                    if abs(float(value) - want) > 1e-6 * max(1.0, want):
                        out.append(f'{where}: {sp}={value!r} != '
                                   f'{hist}/{shipped}={want!r}')
            key = ('selector_latency', row.get('rows'), row.get('n_blocks')) \
                if section == 'selector_latency' \
                else ('indexer_latency', row.get('T'))
        else:  # sink_guard
            case = row.get('case')
            if not isinstance(case, str) or not case:
                out.append(
                    f"{where}: 'case' must be a non-empty string, got {case!r}")
            if row.get('finite') is not True:
                out.append(f"{where}: 'finite' must be True, got "
                           f"{row.get('finite')!r}")
            for fld in ('max_abs_diff_vs_naive', 'output_max_abs'):
                value = row.get(fld)
                if not (_finite_number(value) and float(value) >= 0):
                    out.append(f"{where}: {fld!r} must be finite and >= 0, "
                               f'got {value!r}')
            key = ('sink_guard', case)
        if key in seen:
            out.append(f'{where}: duplicate key {key!r}')
        seen.add(key)
    return out


# ``run_ppl_calibration.py`` (Exp 15) row sections: the calibration
# certificate for the perplexity inference machinery. The contract pins
# the ARITHMETIC of every rate (counts -> rate -> bias/verdict) and of
# every pre-registered bound, so a future edit that widens a scenario
# tolerance or ships an uncalibrated report cannot be silently archived
# as a green artifact. The decode/prefill row's pass flags must match the
# recorded measurements and tolerances (the effect-size gate, not the
# p-value, is the contract).
_EXP15_SECTIONS = frozenset({'coverage', 'paired', 'decode_prefill'})
_EXP15_PAIRED_CHECKS = frozenset({'size', 'power'})


@register_row_contract('exp15_ppl_calibration')
def _validate_exp15_rows(rows: list, metadata: dict) -> list[str]:
    """Calibration rows: self-consistent rates, bounds and verdicts.

    Checks, per section:

    * ``coverage``: positive unit/replicate counts, ``alpha`` in (0, 1),
      ``n_covered`` in ``[0, n_reps]`` with ``coverage``, ``nominal``,
      ``se`` and ``bias`` reproduced from the counts, positive ``tol``,
      and a ``{'calibrated', 'biased'}`` verdict consistent with
      ``|bias| <= tol``;
    * ``paired``: ``{'size', 'power'}`` check tag (size requires
      ``delta == 0``, power ``delta > 0``), ``n_significant`` in
      ``[0, n_reps]``, ``rejection_rate`` / ``se`` reproduced from the
      counts, a ``threshold`` in (0, 1], and ``meets`` / ``verdict``
      consistent with the one-sided comparison the check demands;
    * ``decode_prefill``: positive geometry/token counts, finite
      non-negative per-window divergence, positive finite PPLs, an
      ascending finite CI, ``p_value`` in [0, 1], and ``parity_ok`` /
      ``effect_ok`` flags exactly reproducing the recorded tolerances
      (the p-value must NOT gate the pass flags).
    """
    out: list[str] = []
    seen: set = set()
    for i, row in enumerate(rows):
        where = f'row[{i}]'
        if not isinstance(row, dict):
            out.append(f'{where}: expected an object, got {type(row).__name__}')
            continue
        section = row.get('section')
        if section not in _EXP15_SECTIONS:
            out.append(f'{where}: section={section!r} not in '
                       f'{sorted(_EXP15_SECTIONS)}')
            continue
        if section == 'coverage':
            if not _positive_int(row.get('n_units')):
                out.append(f"{where}: 'n_units' must be a positive int, "
                           f"got {row.get('n_units')!r}")
            alpha = row.get('alpha')
            if not (_finite_number(alpha) and 0.0 < float(alpha) < 1.0):
                out.append(f"{where}: 'alpha' must be in (0, 1), got {alpha!r}")
            for fld in ('n_boot', 'n_reps'):
                if not _positive_int(row.get(fld)):
                    out.append(f"{where}: {fld!r} must be a positive int, "
                               f"got {row.get(fld)!r}")
            n_reps = row.get('n_reps')
            n_covered = row.get('n_covered')
            if not (_positive_int(n_reps)
                    and isinstance(n_covered, int)
                    and 0 <= n_covered <= int(n_reps)):
                out.append(f"{where}: 'n_covered' must be an int in "
                           f"[0, n_reps], got {n_covered!r}")
            coverage = row.get('coverage')
            nominal = row.get('nominal')
            bias = row.get('bias')
            se = row.get('se')
            tol = row.get('tol')
            if _positive_int(row.get('n_reps')) and \
                    isinstance(n_covered, int) and 0 <= n_covered <= n_reps:
                want = n_covered / n_reps
                if not (_finite_number(coverage)
                        and abs(float(coverage) - want) < 1e-9):
                    out.append(f"{where}: coverage={coverage!r} != "
                               f'n_covered/n_reps={want!r}')
            if _finite_number(alpha) and _finite_number(coverage):
                if not (_finite_number(nominal)
                        and abs(float(nominal) - (1.0 - float(alpha))) < 1e-12):
                    out.append(f"{where}: nominal={nominal!r} != 1-alpha")
                if not (_finite_number(bias)
                        and abs(float(bias) - (float(coverage)
                                               - (1.0 - float(alpha))))
                        < 1e-9):
                    out.append(f"{where}: bias={bias!r} != coverage - nominal")
            if (_finite_number(coverage) and _positive_int(n_reps)):
                want_se = math.sqrt(max(0.0, float(coverage)
                                        * (1.0 - float(coverage))
                                        / int(n_reps)))
                if not (_finite_number(se)
                        and abs(float(se) - want_se) < 1e-9):
                    out.append(f"{where}: se={se!r} != binomial SE "
                               f'{want_se!r}')
            if not (_finite_number(tol) and float(tol) > 0.0):
                out.append(f"{where}: 'tol' must be finite and > 0, "
                           f"got {tol!r}")
            elif _positive_int(row.get('n_units')):
                want_tol = 0.05 if int(row['n_units']) >= 32 else 0.08
                if abs(float(tol) - want_tol) > 1e-12:
                    out.append(f'{where}: tol={tol!r} != pre-registered '
                               f'{want_tol!r} for n_units='
                               f"{row['n_units']!r}")
            verdict = row.get('verdict')
            if verdict not in ('calibrated', 'biased'):
                out.append(f"{where}: verdict={verdict!r} not in "
                           f"['calibrated', 'biased']")
            elif _finite_number(bias) and _finite_number(tol):
                want = ('calibrated'
                        if abs(float(bias)) <= float(tol) else 'biased')
                if verdict != want:
                    out.append(f'{where}: verdict={verdict!r} contradicts '
                               f'bias={bias!r} vs tol={tol!r}')
            key = ('coverage', row.get('n_units'), row.get('alpha'))
        elif section == 'paired':
            check = row.get('check')
            if check not in _EXP15_PAIRED_CHECKS:
                out.append(f'{where}: check={check!r} not in '
                           f'{sorted(_EXP15_PAIRED_CHECKS)}')
            if not _positive_int(row.get('n_units')):
                out.append(f"{where}: 'n_units' must be a positive int, "
                           f"got {row.get('n_units')!r}")
            alpha = row.get('alpha')
            if not (_finite_number(alpha) and 0.0 < float(alpha) < 1.0):
                out.append(f"{where}: 'alpha' must be in (0, 1), got {alpha!r}")
            delta = row.get('delta')
            if not (_finite_number(delta) and float(delta) >= 0.0):
                out.append(f"{where}: 'delta' must be finite and >= 0, "
                           f"got {delta!r}")
            elif check == 'size' and float(delta) != 0.0:
                out.append(f'{where}: size control requires delta == 0, '
                           f'got {delta!r}')
            elif check == 'power' and float(delta) <= 0.0:
                out.append(f'{where}: power scenario requires delta > 0, '
                           f'got {delta!r}')
            noise = row.get('noise_sd')
            if not (_finite_number(noise) and float(noise) > 0.0):
                out.append(f"{where}: 'noise_sd' must be finite and > 0, "
                           f"got {noise!r}")
            for fld in ('n_boot', 'n_reps'):
                if not _positive_int(row.get(fld)):
                    out.append(f"{where}: {fld!r} must be a positive int, "
                               f"got {row.get(fld)!r}")
            n_reps = row.get('n_reps')
            n_sig = row.get('n_significant')
            if not (_positive_int(n_reps) and isinstance(n_sig, int)
                    and 0 <= n_sig <= int(n_reps)):
                out.append(f"{where}: 'n_significant' must be an int in "
                           f"[0, n_reps], got {n_sig!r}")
            rate = row.get('rejection_rate')
            if _positive_int(n_reps) and isinstance(n_sig, int) \
                    and 0 <= n_sig <= n_reps:
                want = n_sig / n_reps
                if not (_finite_number(rate)
                        and abs(float(rate) - want) < 1e-9):
                    out.append(f'{where}: rejection_rate={rate!r} != '
                               f'n_significant/n_reps={want!r}')
                want_se = math.sqrt(max(0.0, want * (1.0 - want) / n_reps))
                if not (_finite_number(row.get('se'))
                        and abs(float(row['se']) - want_se) < 1e-9):
                    out.append(f"{where}: se={row.get('se')!r} != "
                               f'binomial SE {want_se!r}')
            threshold = row.get('threshold')
            if not (_finite_number(threshold)
                    and 0.0 < float(threshold) <= 1.0):
                out.append(f"{where}: 'threshold' must be in (0, 1], "
                           f"got {threshold!r}")
            # The pre-registered bound must be RECORDED next to the rate
            # and must reproduce the threshold exactly: a future edit that
            # quietly widens a tolerance changes the artifact and fails
            # here instead of yielding a greener certificate.
            if check == 'size':
                tol = row.get('tol')
                if not (_finite_number(tol) and float(tol) > 0.0):
                    out.append(f"{where}: size control needs a finite "
                               f"positive 'tol', got {tol!r}")
                else:
                    if _positive_int(row.get('n_units')):
                        want_tol = (0.06 if int(row['n_units']) >= 32
                                    else 0.10)
                        if abs(float(tol) - want_tol) > 1e-12:
                            out.append(f'{where}: size tol={tol!r} != '
                                       f'pre-registered {want_tol!r} for '
                                       f"n_units={row['n_units']!r}")
                    if _finite_number(alpha) and _finite_number(threshold) \
                            and not (abs(float(threshold)
                                         - (float(alpha) + float(tol)))
                                     < 1e-12):
                        out.append(f'{where}: threshold={threshold!r} != '
                                   f'alpha + tol='
                                   f'{float(alpha) + float(tol)!r}')
            else:
                floor = row.get('min_power')
                if not (_finite_number(floor)
                        and 0.0 < float(floor) <= 1.0):
                    out.append(f"{where}: power scenario needs 'min_power' "
                               f"in (0, 1], got {floor!r}")
                else:
                    if _finite_number(delta):
                        want_floor = (0.55 if float(delta) <= 0.15 else 0.90)
                        if abs(float(floor) - want_floor) > 1e-12:
                            out.append(f'{where}: min_power={floor!r} != '
                                       f'pre-registered {want_floor!r} '
                                       f'for delta={delta!r}')
                    if _finite_number(threshold) \
                            and not (abs(float(threshold) - float(floor))
                                     < 1e-12):
                        out.append(f'{where}: threshold={threshold!r} != '
                                   f'min_power={floor!r}')
            meets = row.get('meets')
            if not isinstance(meets, bool):
                out.append(f"{where}: 'meets' must be a bool, got {meets!r}")
            elif _finite_number(rate) and _finite_number(threshold):
                want = (float(rate) <= float(threshold) if check == 'size'
                        else float(rate) >= float(threshold))
                if meets != want:
                    out.append(f'{where}: meets={meets!r} contradicts '
                               f'rate={rate!r} vs threshold={threshold!r}')
            verdict = row.get('verdict')
            allowed = ('within_bound', 'liberal') if check == 'size' \
                else ('met', 'below_floor')
            if verdict not in allowed:
                out.append(f'{where}: verdict={verdict!r} not in '
                           f'{sorted(allowed)}')
            elif isinstance(meets, bool):
                want = allowed[0] if meets else allowed[1]
                if verdict != want:
                    out.append(f'{where}: verdict={verdict!r} contradicts '
                               f'meets={meets!r}')
            key = ('paired', row.get('n_units'), row.get('alpha'),
                   row.get('delta'), check)
        else:  # decode_prefill
            for fld in ('n_windows', 'seq_len', 'n_tokens'):
                if not _positive_int(row.get(fld)):
                    out.append(f"{where}: {fld!r} must be a positive int, "
                               f"got {row.get(fld)!r}")
            for fld in ('max_window_nll_diff', 'window_tol_nll'):
                if not (_finite_number(row.get(fld))
                        and float(row[fld]) >= 0.0):
                    out.append(f"{where}: {fld!r} must be finite and >= 0, "
                               f"got {row.get(fld)!r}")
            for fld in ('prefill_ppl', 'decode_ppl', 'ppl_ratio'):
                if not (_finite_number(row.get(fld))
                        and float(row[fld]) > 0.0):
                    out.append(f"{where}: {fld!r} must be finite and > 0, "
                               f"got {row.get(fld)!r}")
            if not _finite_number(row.get('delta_mean_nll')):
                out.append(f"{where}: 'delta_mean_nll' must be finite, "
                           f"got {row.get('delta_mean_nll')!r}")
            ci = row.get('ci_delta_mean_nll')
            if not (isinstance(ci, list) and len(ci) == 2
                    and all(_finite_number(v) for v in ci)
                    and float(ci[0]) <= float(ci[1])):
                out.append(f"{where}: 'ci_delta_mean_nll' must be an "
                           f'ascending finite pair, got {ci!r}')
            ci_ratio = row.get('ci_ppl_ratio')
            if not (isinstance(ci, list) and len(ci) == 2
                    and all(_finite_number(v) for v in ci)):
                pass  # already reported above
            elif not (isinstance(ci_ratio, list) and len(ci_ratio) == 2
                      and all(_finite_number(v) and float(v) > 0.0
                              for v in ci_ratio)
                      and _ci_ratio_matches(ci, ci_ratio)):
                out.append(f"{where}: 'ci_ppl_ratio' must be the exponentiated "
                           f'ci_delta_mean_nll, got {ci_ratio!r} for {ci!r}')
            p_value = row.get('p_value')
            if not (_finite_number(p_value) and 0.0 <= float(p_value) <= 1.0):
                out.append(f"{where}: 'p_value' must be in [0, 1], "
                           f"got {p_value!r}")
            if not isinstance(row.get('significant'), bool):
                out.append(f"{where}: 'significant' must be a bool, got "
                           f"{row.get('significant')!r}")
            for fld in ('parity_ok', 'effect_ok', 'finite_ok'):
                if not isinstance(row.get(fld), bool):
                    out.append(f"{where}: {fld!r} must be a bool, got "
                               f"{row.get(fld)!r}")
            wtol = row.get('window_tol_nll')
            if (isinstance(row.get('parity_ok'), bool)
                    and _finite_number(wtol)
                    and _finite_number(row.get('max_window_nll_diff'))):
                want = (float(row['max_window_nll_diff']) <= float(wtol))
                if row['parity_ok'] != want:
                    out.append(f'{where}: parity_ok contradicts '
                               f'max_window_nll_diff vs window_tol_nll')
            etol_nll = row.get('effect_tol_nll')
            etol_ratio = row.get('effect_tol_ratio')
            for fld, tol_val in (('effect_tol_nll', etol_nll),
                                 ('effect_tol_ratio', etol_ratio)):
                if not (_finite_number(tol_val) and float(tol_val) > 0.0):
                    out.append(f"{where}: {fld!r} must be finite and > 0, "
                               f'got {tol_val!r}')
            if (isinstance(row.get('effect_ok'), bool)
                    and _finite_number(etol_nll) and _finite_number(etol_ratio)
                    and _finite_number(row.get('delta_mean_nll'))
                    and _finite_number(row.get('ppl_ratio'))):
                want = (abs(float(row['delta_mean_nll']))
                        <= float(etol_nll)
                        and abs(float(row['ppl_ratio']) - 1.0)
                        <= float(etol_ratio))
                if row['effect_ok'] != want:
                    out.append(f'{where}: effect_ok contradicts delta / '
                               f'ppl_ratio vs the recorded tolerances')
            key = ('decode_prefill', row.get('n_windows'), row.get('seq_len'))
        if key in seen:
            out.append(f'{where}: duplicate key {key!r}')
        seen.add(key)
    return out


def validate_rows(kind: str, rows, metadata: dict | None = None) -> list[str]:
    """Return the contract violations for ``kind``'s rows ([] = conforming).

    Unknown kinds and unregistered contracts return an empty list, so a
    new experiment can write a versioned envelope (and pass ``--check``)
    before its row contract exists.
    """
    if not isinstance(rows, list):
        return [f'{kind}: rows must be a list, got {type(rows).__name__}']
    contract = ROW_CONTRACTS.get(kind)
    if contract is None:
        return []
    return contract(rows, metadata if isinstance(metadata, dict) else {})


def validate_payload(data) -> list[str]:
    """Contract violations of a parsed payload ([] when inapplicable).

    Applies only to dict envelopes carrying a registered top-level
    ``kind``; bare legacy arrays and unregistered kinds pass untouched.
    """
    if not isinstance(data, dict):
        return []
    kind = data.get('kind')
    if not isinstance(kind, str) or kind not in ROW_CONTRACTS:
        return []
    meta, rows, _version = parse(data)
    return validate_rows(kind, rows, meta)


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

    Mirrors the ``make_figures.load`` contract (a broken file must not
    crash the figure pipeline) while still surfacing the reason.
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


def write_checked(path: str, rows: list, *, kind: str,
                  metadata: dict | None = None, indent: int = 2) -> None:
    """:func:`write` after :func:`validate_rows`; raises on violations.

    A producer bug that would persist a misleading result file (partial
    matrix, arithmetic-inconsistent aggregate, contradictory Exp 6 cache
    flags) fails the experiment loudly instead of being silently
    archived. Unknown kinds pass (see :func:`validate_rows`).
    """
    violations = validate_rows(kind, rows, metadata)
    if violations:
        shown = '; '.join(violations[:5])
        more = f' (+{len(violations) - 5} more)' if len(violations) > 5 else ''
        raise RowContractError(
            f'{path}: {len(violations)} row-contract violation(s) for '
            f'kind={kind!r}: {shown}{more}')
    write(path, rows, kind=kind, metadata=metadata, indent=indent)


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


def _row_violations(path: str) -> list[str]:
    """Contract violations of a kind-tagged file ([] when inapplicable)."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    try:
        return validate_payload(data)
    except SchemaError:
        return []


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = '--check' in argv or '--check-rows' in argv
    check_rows = '--check-rows' in argv
    argv = [a for a in argv if a not in ('--check', '--check-rows')]
    paths = argv or sorted(glob.glob(os.path.join('results', '*.json')))
    if not paths:
        print('[results_schema] no result files found')
        return 0
    bad = 0
    for p in paths:
        version, n_rows, status = _describe(p)
        violations: list[str] = []
        if check_rows and status == 'ok':
            violations = _row_violations(p)
            if violations:
                status = f'contract ({len(violations)})'
        print(f'{version:>5}  {n_rows:>6} rows  {status:<12}  {p}')
        for v in violations[:5]:
            print(f'          - {v}')
        if status != 'ok':
            bad += 1
    if check and bad:
        mode = 'schema+row-contract' if check_rows else 'schema'
        print(f'[results_schema] {bad} file(s) failed the {mode} check')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
