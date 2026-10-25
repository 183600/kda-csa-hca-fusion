"""Recall-per-byte diagnostics for the associative-recall ratio sweep.

Why this module?
----------------
Exp 5 (``run_ablation``) already reports MQAR accuracy together with
parameters, analytic FLOPs, forward latency and training time.  All of
those answer "what does this KDA:CSA:HCA allocation cost on the measured
prefill/training path", but none of them prices the phase where a hybrid
long-context model is actually memory-bound: **token-by-token decode**.
Two layouts can have identical parameter counts and similar short-sequence
latency while moving very different amounts of recurrent state + compressed
cache per decoded token.  A one-point MQAR gain that triples decode traffic
is a different operating point, not an unconditional improvement.

This module adds the missing axis without duplicating the canonical byte
model:

* ``decode_bandwidth.decode_traffic_breakdown`` remains the single source
  of truth for per-sequence/per-token read and write bytes.
  :func:`decode_traffic_fields` only renames/validates its output into
  JSON-safe row fields.
* :func:`chance_corrected_recall` turns MQAR accuracy into skill above the
  ``1/VOCAB`` chance rate, so a memoryless control contributes exactly 0
  and layouts are comparable across task difficulties.
* :func:`recall_per_megabyte` divides that skill by the decode traffic,
  the natural "associative recall per MB moved" efficiency for a
  memory-bound kernel.
* :func:`annotate_recall_efficiency` marks the accuracy-vs-traffic Pareto
  frontier (accuracy maximise, decode bytes minimise) and records, per
  row, the explicit dominator.  No row is silently dropped: malformed or
  byte-less rows are listed under ``skipped``.

The byte model is a **compulsory-traffic lower bound** (weights are not
included here; this is the per-token state/cache term).  It is an analysis
of the reference geometry, not a claim about a production kernel.  Use
``roofline.measure_copy_bandwidth_gbps`` to turn bytes into a bandwidth
bound when a wall-clock comparison is needed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence

#: Row field written by :func:`decode_traffic_fields` and consumed by the
#: Pareto/efficiency annotation.  Kept as one constant so producers and
#: consumers cannot disagree about the cost axis name.
DEFAULT_TRAFFIC_KEY = 'decode_bytes_per_token'

#: Decimal megabytes, matching how throughput is quoted elsewhere in the
#: repository (GB/s uses 1e9, not 2**30).
_BYTES_PER_MB = 1_000_000.0


def _finite_or_none(value):
    """Coerce a value to a finite float, or ``None`` when impossible.

    Booleans are rejected explicitly: ``True`` is an int in Python and
    would otherwise be silently priced as one byte.  Strings/bytes are
    rejected too: a JSON string (even a numeric-looking one) is a
    malformed measurement, and coercing it would let stringified output
    silently enter the efficiency frontier.
    """
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _ledger_sequence_length(value):
    """Validate a ledger ``T`` as a non-negative integer (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f'decode_traffic_fields: T={value!r} must be a non-negative '
            f'int or None.')
    if value < 0:
        raise ValueError(
            f'decode_traffic_fields: T={value!r} must be >= 0.')
    return value


def _ledger_op_map(traffic: Mapping, key: str) -> Mapping:
    """Return a validated per-operator byte map (missing -> empty)."""
    value = traffic.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f'decode_traffic_fields: {key} must be a mapping, got '
            f'{type(value).__name__}.')
    return value


def chance_corrected_recall(accuracy, chance):
    """MQAR skill above chance: ``(acc - chance) / (1 - chance)``.

    Returns ``None`` for a missing/non-finite input, an accuracy outside
    ``[0, 1]``, or a chance rate outside ``[0, 1)`` (chance == 1 leaves no
    headroom, so no finite skill scale exists).  A model at chance maps to
    exactly 0; a perfect model maps to 1.  Negative values are preserved
    on purpose: below-chance performance is evidence, not a missing value.
    """
    acc = _finite_or_none(accuracy)
    base = _finite_or_none(chance)
    if acc is None or base is None:
        return None
    if not (0.0 <= acc <= 1.0) or not (0.0 <= base < 1.0):
        return None
    return (acc - base) / (1.0 - base)


def _per_megabyte(value, nbytes):
    """Divide by the byte count in decimal MB, keeping the result JSON-safe.

    A positive subnormal byte count can underflow to zero when expressed in
    MB (``5e-324 / 1e6``), and a tiny non-zero denominator can overflow the
    quotient to ``+inf``.  Neither is a meaningful efficiency, so both map
    to ``None`` instead of raising ``ZeroDivisionError`` or emitting a
    non-finite value.
    """
    megabytes = nbytes / _BYTES_PER_MB
    if megabytes <= 0.0:
        return None
    result = value / megabytes
    return result if math.isfinite(result) else None


def recall_per_megabyte(accuracy, chance, bytes_per_token):
    """Chance-corrected recall per decimal MB of per-token decode traffic.

    ``bytes_per_token`` must be finite and strictly positive; a zero (or
    missing) denominator has no meaningful efficiency, so the result is
    ``None`` rather than ``+inf``.  The sign follows the skill term: a
    below-chance layout gets a negative efficiency.
    """
    skill = chance_corrected_recall(accuracy, chance)
    nbytes = _finite_or_none(bytes_per_token)
    if skill is None or nbytes is None or nbytes <= 0.0:
        return None
    return _per_megabyte(skill, nbytes)


def decode_traffic_fields(traffic: Mapping) -> dict:
    """Validate a ``decode_traffic_breakdown`` result into row fields.

    Only the scalar fields a ratio-sweep row needs are copied; the full
    per-operator breakdown is preserved under ``decode_bytes_by_op`` so a
    reader can see whether the traffic is recurrent-state or CSA/HCA
    cache dominated.  Raises ``TypeError``/``ValueError`` on a malformed
    ledger rather than writing a row that looks priced but is not.
    """
    if not isinstance(traffic, Mapping):
        raise TypeError(
            f'decode_traffic_fields: traffic must be a mapping, got '
            f'{type(traffic).__name__}.')
    required = ('read_bytes_per_token', 'write_bytes_per_token',
                'total_bytes_per_token')
    values = {}
    for key in required:
        number = _finite_or_none(traffic.get(key))
        if number is None or number < 0.0:
            raise ValueError(
                f'decode_traffic_fields: {key}={traffic.get(key)!r} must '
                f'be a finite non-negative number.')
        values[key] = number
    total = values['total_bytes_per_token']
    # The canonical model defines total = read + write.  Re-check the
    # identity here so a future ledger change cannot make the new cost
    # axis quietly inconsistent with the numbers it was built from.
    if not math.isclose(total,
                        values['read_bytes_per_token']
                        + values['write_bytes_per_token'],
                        rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            'decode_traffic_fields: total_bytes_per_token does not equal '
            'read + write; refusing to price an inconsistent ledger.')
    read_by_op = _ledger_op_map(traffic, 'read_bytes_by_op')
    write_by_op = _ledger_op_map(traffic, 'write_bytes_by_op')
    by_op = {}
    for op in ('memory', 'csa', 'hca'):
        read_raw = read_by_op.get(op)
        write_raw = write_by_op.get(op)
        read = _finite_or_none(read_raw)
        write = _finite_or_none(write_raw)
        for label, raw, number in (
                ('read', read_raw, read), ('write', write_raw, write)):
            if raw is not None and (number is None or number < 0.0):
                raise ValueError(
                    f'decode_traffic_fields: {op} {label} bytes per token '
                    f'({raw!r}) must be a finite non-negative number.')
        by_op[op] = {
            'read_bytes_per_token': read,
            'write_bytes_per_token': write,
        }
    layout_raw = traffic.get('layout')
    if layout_raw is None:
        layout = None
    else:
        if (isinstance(layout_raw, (str, bytes))
                or not isinstance(layout_raw, Sequence)):
            raise TypeError(
                f'decode_traffic_fields: layout must be a sequence of '
                f'layer counts, got {type(layout_raw).__name__}.')
        layout = list(layout_raw)
        for index, count in enumerate(layout):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(
                    f'decode_traffic_fields: layout[{index}] must be an '
                    f'int layer count, got {type(count).__name__}.')
            if count < 0:
                raise ValueError(
                    f'decode_traffic_fields: layout[{index}]={count} must '
                    f'be >= 0.')
    memory_route = traffic.get('memory_route')
    if memory_route is not None and not isinstance(memory_route, str):
        raise TypeError(
            'decode_traffic_fields: memory_route must be a string or '
            f'None, got {type(memory_route).__name__}.')
    fields = {
        'decode_T': _ledger_sequence_length(traffic.get('T')),
        'decode_read_bytes_per_token': values['read_bytes_per_token'],
        'decode_write_bytes_per_token': values['write_bytes_per_token'],
        DEFAULT_TRAFFIC_KEY: total,
        'decode_bytes_by_op': by_op,
        'decode_memory_route': memory_route,
        'decode_layout': layout,
        'decode_traffic_scope': (
            'canonical per-sequence per-token state/cache bytes '
            '(compulsory lower bound; tower weights excluded)'),
    }
    return fields


def _unique_keys(rows: Sequence, name_key: str) -> list[str]:
    """Stable unique key per row (duplicates get ``name#2``/``name#3``)."""
    seen: dict = {}
    used: set = set()
    keys: list[str] = []
    for index, row in enumerate(rows):
        name = (row.get(name_key)
                if isinstance(row, Mapping) else None)
        if name is None:
            name = f'row#{index}'
        try:
            hash(name)
        except TypeError:
            name = repr(name)
        name = str(name)
        count = seen.get(name, 0) + 1
        key = name if count == 1 else f'{name}#{count}'
        while key in used:
            count += 1
            key = f'{name}#{count}'
        seen[name] = count
        used.add(key)
        keys.append(key)
    return keys


def annotate_recall_efficiency(rows: Sequence[Mapping], *,
                               accuracy_key: str = 'mean_acc',
                               chance_key: str = 'chance_acc',
                               bytes_key: str = DEFAULT_TRAFFIC_KEY,
                               name_key: str = 'ratio') -> dict:
    """Attach chance-corrected recall-per-MB and a Pareto frontier.

    A row is Pareto-optimal when no other row has accuracy >= its own and
    decode bytes <= its own, with at least one strict inequality.  The
    comparison uses raw accuracy (chance is constant inside a ratio-sweep
    cell, so the dominance order is identical to chance-corrected skill),
    while every row also records the normalized skill and recall/MB so the
    efficiency number itself is interpretable.

    Each mapping row gets ``recall_efficiency`` = ``{optimal,
    dominated_by, chance_corrected_recall, recall_per_megabyte,
    accuracy_per_megabyte, decode_bytes_per_token}``.  Rows missing a
    finite accuracy or traffic value are listed in ``skipped`` and get
    ``recall_efficiency=None`` instead of being silently dropped.

    Returns a JSON-safe report:
    ``{'accuracy_key', 'chance_key', 'bytes_key', 'frontier',
    'dominated_by', 'points', 'skipped'}``.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(
            'annotate_recall_efficiency: rows must be a sequence of '
            f'mappings, got {type(rows).__name__}.')
    keys = _unique_keys(rows, name_key)
    points: dict = {}
    skipped: list[str] = []
    for row, key in zip(rows, keys):
        if not isinstance(row, Mapping):
            skipped.append(key)
            continue
        acc = _finite_or_none(row.get(accuracy_key))
        base = _finite_or_none(row.get(chance_key))
        nbytes = _finite_or_none(row.get(bytes_key))
        skill = chance_corrected_recall(acc, base)
        if acc is None or skill is None or nbytes is None or nbytes <= 0.0:
            skipped.append(key)
            if isinstance(row, MutableMapping):
                row['recall_efficiency'] = None
            continue
        points[key] = {
            'accuracy': acc,
            'chance': base,
            'decode_bytes_per_token': nbytes,
            'chance_corrected_recall': skill,
            'recall_per_megabyte': recall_per_megabyte(acc, base, nbytes),
            'accuracy_per_megabyte': _per_megabyte(acc, nbytes),
        }

    names = list(points)
    dominated_by: dict = {}
    for candidate in names:
        dominators = []
        for other in names:
            if other == candidate:
                continue
            acc_ge = (points[other]['accuracy']
                      >= points[candidate]['accuracy'])
            bytes_le = (points[other]['decode_bytes_per_token']
                        <= points[candidate]['decode_bytes_per_token'])
            strict = (points[other]['accuracy']
                      > points[candidate]['accuracy']
                      or points[other]['decode_bytes_per_token']
                      < points[candidate]['decode_bytes_per_token'])
            if acc_ge and bytes_le and strict:
                dominators.append(other)
        if dominators:
            # Name the closest dominator: highest accuracy, then cheapest.
            dominators.sort(
                key=lambda n: (-points[n]['accuracy'],
                               points[n]['decode_bytes_per_token']))
            dominated_by[candidate] = dominators[0]

    frontier = [n for n in names if n not in dominated_by]
    for row, key in zip(rows, keys):
        if not isinstance(row, MutableMapping):
            continue
        point = points.get(key)
        if point is None:
            row['recall_efficiency'] = None
            continue
        row['recall_efficiency'] = {
            'optimal': key in frontier,
            'dominated_by': dominated_by.get(key),
            'chance_corrected_recall': point['chance_corrected_recall'],
            'recall_per_megabyte': point['recall_per_megabyte'],
            'accuracy_per_megabyte': point['accuracy_per_megabyte'],
            'decode_bytes_per_token': point['decode_bytes_per_token'],
        }

    return {
        'accuracy_key': accuracy_key,
        'chance_key': chance_key,
        'bytes_key': bytes_key,
        'frontier': frontier,
        'dominated_by': dominated_by,
        'points': points,
        'skipped': skipped,
    }
