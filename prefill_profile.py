"""Chunked-prefill throughput and numerical-drift profiler.

Why this module?
----------------
The decoding wrappers in :mod:`run_decoding` support *continuation
chunks*: after a first prefill, the remaining prompt can be fed in
several calls instead of one.  That is exactly what a serving engine
does for chunked prefill (interleave a long prompt with decode work), but
the repository only pinned its *correctness* (chunk-schedule invariance)
and only benchmarked a one-shot prefill followed by single-token decode.

This module closes that gap with a small, dependency-light profiler:

* ``expand_prefill_schedule`` — deterministic schedule construction
  (``full``, a uniform chunk size, or an explicit list);
* ``run_prefill_schedule`` — feed a prompt through any decoder wrapper
  using a schedule, with a cache reset before every independent run;
* ``drift_metrics`` — fp64-accumulated max-abs / relative-L2 / cosine
  drift against the one-shot reference, with explicit non-finite
  detection (a fast schedule must never silently change the answer);
* ``time_prefill_schedule`` — warmup + repeated wall-clock timing with
  median/min/max and tokens/s;
* ``profile_prefill_schedules`` — the combined experiment row: throughput
  AND drift for every schedule, so speed cannot be traded for correctness
  without the report showing it.

The numerical-stability contract
--------------------------------
Chunked prefill is only useful if it is *numerically* the same model as
one-shot prefill.  The decoding cache already implements the same
strictly-preceding block mask and orthogonal RoPE at absolute positions;
changing the chunk boundaries changes only the order in which independent
token projections and cache appends are dispatched.  The profiler makes
that claim falsifiable:

* all drift reductions run in float64 (output tensors are cast, not the
  model), so a fp32 report does not hide a 1e-6 signal in fp32 rounding;
* a row whose output contains NaN/Inf is marked ``finite=False`` and
  fails even if the numeric drift fields are undefined;
* a row whose output dtype differs from the reference is marked
  ``dtype_match=False`` and fails: a schedule may not change the model's
  precision, even when the resulting numeric drift is tiny;
* the pass criterion is ``dtype_match and max_abs <= atol + rtol *
  max|reference|`` (the same mixed absolute/relative form as
  ``torch.testing.assert_close``);
* ``bitwise_equal`` is reported separately so exact schedule invariance
  (the CPU case) is visible rather than rounded away.

Usage::

    python3 prefill_profile.py --model hybrid --total-len 256
    python3 prefill_profile.py --model csa --total-len 128 \
        --schedules full,64,32,16,1 --json results/prefill_profile.json
    python3 prefill_profile.py --model hca --total-len 32 \
        --schedules full,8:8:8:8,1 --repeats 5 --warmup 2

Wall-clock numbers are machine-, torch-build- and thread-count-specific;
never compare them across hosts.  The one invariant that IS portable is
the drift column: every schedule should agree with the one-shot reference
inside the configured tolerance.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from kaggle_setup import (capture_provenance, set_global_seed,
                          write_json_atomic)  # noqa: E402

__all__ = [
    'PREFILL_PROFILE_KIND',
    'drift_metrics',
    'expand_prefill_schedule',
    'profile_prefill_schedules',
    'run_prefill_schedule',
    'schedule_label',
    'time_prefill_schedule',
]

#: ``results_schema``-style kind for the optional JSON artifact.  The
#: profiler is a diagnostic tool, not a committed experiment, so this kind
#: is intentionally NOT registered in ``results_schema``.
PREFILL_PROFILE_KIND = 'chunked_prefill_profile'

#: Default drift tolerances (fp32-oriented; fp64 only becomes stricter).
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4

#: Default CLI schedules: one-shot plus a geometric chunk sweep.
DEFAULT_SCHEDULES = ('full', '64', '32', '16', '8', '1')

#: A uniform schedule is materialized as one list entry per chunk; reject
#: counts that would exhaust memory before the first forward runs.
MAX_SCHEDULE_CHUNKS = 1_000_000


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------
def _positive_int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{what} must be an int, got {value!r}')
    if value < 1:
        raise ValueError(f'{what} must be >= 1, got {value}')
    return value


def _nonnegative_tolerance(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a real number, got {value!r}')
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be finite and >= 0, got {value!r}')
    return value


def expand_prefill_schedule(total_len: int, chunks):
    """Expand ``chunks`` into a list of chunk lengths summing to ``total_len``.

    Accepted forms
    --------------
    * ``'full'`` / ``'one-shot'`` -> ``[total_len]``;
    * an ``int`` ``N`` -> uniform chunks ``[N, N, ..., remainder]``;
    * a ``':'``-separated string ``'4:3:3'`` -> the literal list;
    * a sequence of positive ints -> the literal list.

    A literal schedule MUST sum to ``total_len`` exactly.  Silently
    appending the remainder would make two different schedules share a
    label while running different work; rejecting is the safer contract
    for a benchmark.

    ``total_len == 0`` returns ``[]`` for every form.
    """
    if isinstance(total_len, bool) or not isinstance(total_len, int):
        raise TypeError(f'total_len must be an int, got {total_len!r}')
    if total_len < 0:
        raise ValueError(f'total_len must be >= 0, got {total_len}')
    if total_len == 0:
        return []

    if isinstance(chunks, str):
        text = chunks.strip()
        if not text:
            raise ValueError('prefill schedule string must not be empty')
        low = text.lower()
        if low in ('full', 'one-shot', 'oneshot'):
            return [total_len]
        if ':' in text:
            parts = [p.strip() for p in text.split(':')]
            if any(not p for p in parts):
                raise ValueError(
                    f'prefill schedule {chunks!r} has an empty component')
            try:
                values = [int(p) for p in parts]
            except ValueError as exc:
                raise ValueError(
                    f'prefill schedule {chunks!r} must contain ints') from exc
            return _literal_schedule(total_len, values, chunks)
        try:
            size = int(text)
        except ValueError as exc:
            raise ValueError(
                f'prefill schedule {chunks!r} must be "full", an int, or '
                f'a colon-separated int list') from exc
        return _uniform_schedule(total_len, _positive_int(
            size, f'prefill chunk size {chunks!r}'))

    if isinstance(chunks, int) and not isinstance(chunks, bool):
        return _uniform_schedule(total_len, _positive_int(
            chunks, 'prefill chunk size'))

    try:
        values = list(chunks)
    except TypeError as exc:
        raise TypeError(
            f'prefill schedule must be "full", an int, or a sequence of '
            f'ints; got {type(chunks).__name__}') from exc
    return _literal_schedule(total_len, values, chunks)


def _uniform_schedule(total_len: int, size: int) -> list[int]:
    n_chunks = total_len // size + (1 if total_len % size else 0)
    if n_chunks > MAX_SCHEDULE_CHUNKS:
        raise ValueError(
            f'uniform prefill schedule would have {n_chunks} chunks '
            f'(limit {MAX_SCHEDULE_CHUNKS}); use a larger chunk size or '
            f'an explicit schedule')
    return [size] * (total_len // size) + (
        [total_len % size] if total_len % size else [])


def _literal_schedule(total_len: int, values, original) -> list[int]:
    if not values:
        raise ValueError('prefill schedule must not be empty')
    out = [_positive_int(v, f'prefill chunk in {original!r}')
           for v in values]
    if sum(out) != total_len:
        raise ValueError(
            f'prefill schedule {original!r} sums to {sum(out)}, expected '
            f'total_len={total_len}')
    return out


def schedule_label(chunks) -> str:
    """Compact, deterministic label for a schedule spec or list."""
    if isinstance(chunks, str):
        text = chunks.strip()
        low = text.lower()
        if low in ('full', 'one-shot', 'oneshot'):
            return 'full'
        if ':' in text:
            return text
        try:
            return str(int(text))
        except ValueError:
            return text
    if isinstance(chunks, int) and not isinstance(chunks, bool):
        return str(chunks)
    try:
        values = list(chunks)
    except TypeError:
        return str(chunks)
    if not values:
        return 'empty'
    if len(set(values)) == 1:
        return str(values[0])
    return ':'.join(str(v) for v in values)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def _require_reset(model) -> None:
    reset = getattr(model, 'reset', None)
    if not callable(reset):
        raise TypeError(
            'prefill profiler requires a model with a callable reset() so '
            'independent runs do not inherit cache/state from the previous '
            f'run (got {type(model).__name__})')


def run_prefill_schedule(model, x: torch.Tensor, chunks, *,
                         reset: bool = True,
                         schedule: list[int] | None = None):
    """Run ``model`` over ``x`` using a chunk schedule; return the output.

    ``chunks`` is expanded against ``x.shape[1]`` (see
    :func:`expand_prefill_schedule`).  ``schedule`` may be passed instead
    when the caller already expanded it (avoids re-parsing for every
    timing repeat).  A cache reset happens before the first chunk; each
    call is made under ``torch.no_grad`` because chunked prefill is an
    inference path.
    """
    if x.dim() != 3:
        raise ValueError(f'x must be [B, T, d], got {tuple(x.shape)}')
    total_len = x.shape[1]
    if schedule is None:
        schedule = expand_prefill_schedule(total_len, chunks)
    else:
        try:
            schedule = list(schedule)
        except TypeError as exc:
            raise TypeError(
                'expanded schedule must be a sequence of positive ints, '
                f'got {type(schedule).__name__}') from exc
        if total_len == 0:
            if schedule:
                raise ValueError(
                    f'empty input requires an empty schedule, got '
                    f'{schedule!r}')
        else:
            # Reuse the same positivity/sum contract as a user-supplied
            # literal schedule; otherwise a negative entry (e.g. [-1, T+1])
            # silently slices weirdly instead of failing.
            schedule = _literal_schedule(total_len, schedule, schedule)
    _require_reset(model)
    if reset:
        model.reset()
    if total_len == 0:
        with torch.no_grad():
            return model(x)
    outs = []
    pos = 0
    with torch.no_grad():
        for size in schedule:
            outs.append(model(x[:, pos:pos + size]))
            pos += size
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# Numerical drift
# ---------------------------------------------------------------------------
def _finite_float(value) -> float | None:
    value = float(value)
    return value if value == value and value not in (float('inf'),
                                                     float('-inf')) else None


def drift_metrics(reference: torch.Tensor, candidate: torch.Tensor, *,
                  dtype: torch.dtype = torch.float64) -> dict:
    """Compare two same-shaped tensors; report fp64-accumulated drift.

    Returns a JSON-safe dict.  If either tensor contains NaN/Inf, or the
    shapes differ, ``finite`` is False and the numeric fields are None
    (undefined) rather than a misleading NaN.  Empty tensors are treated
    as exactly equal (zero drift, cosine 1.0).

    ``dtype_match`` is reported independently of the drift: two schedules
    can agree numerically while returning different dtypes (e.g. one path
    silently promotes fp32 to fp64).  That is a split contract, not a
    rounding difference, so callers that require schedule invariance must
    check it as well.
    """
    if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor):
        raise TypeError('drift_metrics expects two torch.Tensor inputs')
    dtype_match = reference.dtype == candidate.dtype
    if reference.shape != candidate.shape:
        return {
            'shape_match': False,
            'dtype_match': dtype_match,
            'finite': False,
            'bitwise_equal': False,
            'max_abs': None,
            'mean_abs': None,
            'rms': None,
            'relative_l2': None,
            'cosine': None,
            'reference_norm': None,
            'candidate_norm': None,
            'n_elements': int(reference.numel()),
        }
    ref = reference.detach().to(dtype=dtype)
    cand = candidate.detach().to(dtype=dtype)
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(cand).all())
    if ref.numel() == 0:
        return {
            'shape_match': True,
            'dtype_match': dtype_match,
            'finite': finite,
            'bitwise_equal': True,
            'max_abs': 0.0,
            'mean_abs': 0.0,
            'rms': 0.0,
            'relative_l2': 0.0,
            'cosine': 1.0,
            'reference_norm': 0.0,
            'candidate_norm': 0.0,
            'n_elements': 0,
        }
    if not finite:
        return {
            'shape_match': True,
            'dtype_match': dtype_match,
            'finite': False,
            'bitwise_equal': False,
            'max_abs': None,
            'mean_abs': None,
            'rms': None,
            'relative_l2': None,
            'cosine': None,
            'reference_norm': None,
            'candidate_norm': None,
            'n_elements': int(ref.numel()),
        }
    diff = cand - ref
    abs_diff = diff.abs()
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    rms = torch.sqrt(torch.mean(diff * diff))
    if not bool(torch.isfinite(rms)):
        # ``diff**2`` can overflow even when the RMS itself is
        # representable (e.g. a 2e200 difference in fp64). Scale before
        # squaring; the ordinary in-range value above is unchanged bit for
        # bit.
        scale = abs_diff.max()
        if bool(torch.isfinite(scale)) and float(scale) > 0.0:
            rms = torch.sqrt(torch.mean((diff / scale) ** 2)) * scale

    def _safe_norm(t):
        norm = torch.linalg.vector_norm(t)
        if bool(torch.isfinite(norm)):
            return norm
        scale = float(t.abs().max()) if t.numel() else 0.0
        if scale == 0.0:
            return torch.zeros((), dtype=t.dtype)
        return torch.linalg.vector_norm(t / scale) * scale

    ref_norm = _safe_norm(ref)
    cand_norm = _safe_norm(cand)
    diff_norm = _safe_norm(diff)
    tiny = torch.finfo(dtype).tiny
    relative_l2 = diff_norm / ref_norm.clamp_min(tiny)
    if ref_norm <= tiny or cand_norm <= tiny:
        # Cosine is defined as 1.0 for two zero vectors and 0.0 when only
        # one is zero; mirror ``F.cosine_similarity``'s eps convention.
        cosine = 1.0 if (ref_norm <= tiny and cand_norm <= tiny) else 0.0
    else:
        dot = torch.dot(ref.reshape(-1), cand.reshape(-1))
        denominator = ref_norm * cand_norm
        raw = dot / denominator
        if not bool(torch.isfinite(raw)):
            # Squared magnitudes near the float limit overflow both the
            # dot product and the norms. Recompute the cosine after
            # dividing by a common scale; the ratio is unchanged.
            scale = max(float(ref.abs().max()), float(cand.abs().max()))
            if scale == 0.0:
                raw = torch.ones((), dtype=ref.dtype)
            else:
                rs = ref / scale
                cs = cand / scale
                raw = (torch.dot(rs.reshape(-1), cs.reshape(-1))
                       / (torch.linalg.vector_norm(rs)
                          * torch.linalg.vector_norm(cs)))
        cosine = float(torch.clamp(raw, -1.0, 1.0))
    return {
        'shape_match': True,
        'dtype_match': dtype_match,
        'finite': True,
        'bitwise_equal': bool(torch.equal(reference, candidate)),
        'max_abs': _finite_float(max_abs),
        'mean_abs': _finite_float(mean_abs),
        'rms': _finite_float(rms),
        'relative_l2': _finite_float(relative_l2),
        'cosine': _finite_float(cosine),
        'reference_norm': _finite_float(ref_norm),
        'candidate_norm': _finite_float(cand_norm),
        'n_elements': int(ref.numel()),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def time_prefill_schedule(model, x: torch.Tensor, chunks, *,
                          repeats: int = 3, warmup: int = 1,
                          return_output: bool = False,
                          schedule: list[int] | None = None) -> dict:
    """Time a full scheduled prefill (cache reset before every trial).

    Returns median/min/max latency, tokens/s, chunk count and the expanded
    schedule.  ``warmup`` untimed runs absorb allocator/autotune cost.
    ``return_output=True`` also returns the output of the LAST timed run
    (for drift comparison without an extra forward).
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f'repeats must be an int >= 1, got {repeats!r}')
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f'warmup must be an int >= 0, got {warmup!r}')
    total_len = x.shape[1]
    if schedule is None:
        schedule = expand_prefill_schedule(total_len, chunks)
    else:
        schedule = list(schedule)
    _require_reset(model)
    device = x.device

    def _one() -> torch.Tensor:
        model.reset()
        return run_prefill_schedule(
            model, x, chunks, reset=False, schedule=schedule)

    for _ in range(warmup):
        _one()
    _sync(device)

    times_ms = []
    out = None
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = _one()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    median_ms = statistics.median(times_ms)
    row = {
        'schedule': schedule_label(schedule),
        'n_chunks': len(schedule),
        'chunk_sizes': list(schedule),
        'prefill_len': int(total_len),
        'latency_ms': float(median_ms),
        'latency_min_ms': float(min(times_ms)),
        'latency_max_ms': float(max(times_ms)),
        'tokens_per_s': (float(total_len) / (median_ms / 1e3)
                         if median_ms > 0 else None),
        'repeats': repeats,
        'warmup': warmup,
    }
    if return_output:
        return row, out
    return row


# ---------------------------------------------------------------------------
# Combined profile
# ---------------------------------------------------------------------------
def profile_prefill_schedules(model, x: torch.Tensor, schedules, *,
                              repeats: int = 3, warmup: int = 1,
                              atol: float = DEFAULT_ATOL,
                              rtol: float = DEFAULT_RTOL,
                              drift_dtype: torch.dtype = torch.float64) -> dict:
    """Throughput + numerical drift for every schedule.

    The one-shot (``[T]``) schedule is the reference.  Every row carries
    both its timing and its drift against that reference; a row passes iff
    the output dtype matches the reference, the output is finite, and it
    satisfies ``max_abs <= atol + rtol * max|reference|``.  The dtype
    clause keeps a schedule that silently changes precision from hiding
    behind a tiny numeric drift.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f'repeats must be an int >= 1, got {repeats!r}')
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f'warmup must be an int >= 0, got {warmup!r}')
    if x.dim() != 3:
        raise ValueError(f'x must be [B, T, d], got {tuple(x.shape)}')
    atol = _nonnegative_tolerance(atol, 'atol')
    rtol = _nonnegative_tolerance(rtol, 'rtol')
    total_len = x.shape[1]
    schedules = list(schedules)
    if not schedules:
        raise ValueError('schedules must not be empty')
    expanded = [expand_prefill_schedule(total_len, s) for s in schedules]

    # An empty prompt already has a well-defined one-shot schedule: no
    # chunks at all. ``[0]`` is not a valid literal schedule (chunks must
    # be positive and must not be passed for an empty input), so use the
    # empty list that ``expand_prefill_schedule(0, ...)`` produces.
    ref_schedule = [] if total_len == 0 else [total_len]
    ref_row, reference = time_prefill_schedule(
        model, x, [], repeats=repeats, warmup=warmup,
        return_output=True, schedule=ref_schedule)
    ref_row['schedule'] = 'full'
    ref_row['is_reference'] = True
    ref_row['drift'] = drift_metrics(
        reference, reference, dtype=drift_dtype)
    # The reference is a row too: a non-finite reference output cannot
    # pass, even when it is the only schedule profiled. The remaining
    # numeric fields are undefined in that case (see ``drift_metrics``).
    ref_row['passed'] = bool(
        ref_row['drift']['finite'] and ref_row['drift']['dtype_match'])
    ref_row['tolerance'] = {
        'atol': float(atol), 'rtol': float(rtol),
        'criterion': 'dtype_match and max_abs <= atol + rtol * max|reference|'}

    rows = [ref_row]
    seen = set()
    if reference.numel() and ref_row['drift']['finite']:
        ref_scale = float(reference.detach().to(dtype=drift_dtype).abs().max())
    elif reference.numel():
        ref_scale = None
    else:
        ref_scale = 0.0
    tol_abs = (None if ref_scale is None
               else float(atol) + float(rtol) * ref_scale)
    for spec, sched in zip(schedules, expanded):
        key = tuple(sched)
        if key == tuple(ref_schedule) or key in seen:
            # A second identical schedule adds no information (and would
            # produce a misleading duplicate row).
            continue
        seen.add(key)
        row, out = time_prefill_schedule(
            model, x, [], repeats=repeats, warmup=warmup,
            return_output=True, schedule=sched)
        drift = drift_metrics(reference, out, dtype=drift_dtype)
        row['is_reference'] = False
        row['drift'] = drift
        row['tolerance'] = {
            'atol': float(atol), 'rtol': float(rtol),
            'criterion': 'dtype_match and max_abs <= atol + rtol * max|reference|'}
        row['passed'] = bool(
            drift['dtype_match'] and tol_abs is not None and drift['finite']
            and drift['max_abs'] is not None
            and drift['max_abs'] <= tol_abs)
        row['_requested_schedule'] = schedule_label(spec)
        rows.append(row)

    return {
        'kind': PREFILL_PROFILE_KIND,
        'prefill_len': int(total_len),
        'd_model': int(x.shape[2]),
        'model': type(model).__name__,
        'device': str(x.device),
        'dtype': str(x.dtype),
        'atol': float(atol),
        'rtol': float(rtol),
        'max_abs_tolerance': tol_abs,
        'reference_max_abs': ref_scale,
        'tolerance': {
            'atol': float(atol), 'rtol': float(rtol),
            'criterion': 'dtype_match and max_abs <= atol + rtol * max|reference|'},
        'all_passed': bool(all(r['passed'] for r in rows)),
        'rows': rows,
        'note': (
            'Latency is machine/thread-specific; drift vs the one-shot '
            'reference is the portable result.'),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_MODEL_CHOICES = ('csa', 'hca', 'hybrid', 'hybrid_mamba2', 'hybrid_rwkv7')


def _build_model(name: str, d_model: int):
    from run_decoding import (CSAAttnDecoding, HCAAttnDecoding,
                              HybridDecoding)
    if name == 'csa':
        return CSAAttnDecoding(d_model, topk=100)
    if name == 'hca':
        return HCAAttnDecoding(d_model)
    if name == 'hybrid':
        return HybridDecoding(d_model, total_layers=5, csa_topk=100)
    if name == 'hybrid_mamba2':
        return HybridDecoding(
            d_model, total_layers=5, csa_topk=100,
            memory_type='mamba2')
    if name == 'hybrid_rwkv7':
        return HybridDecoding(
            d_model, total_layers=5, csa_topk=100,
            memory_type='rwkv7')
    raise ValueError(f'unknown model {name!r}; choose from {_MODEL_CHOICES}')


def _resolve_device(text: str) -> torch.device:
    text = (text or 'auto').strip().lower()
    if text == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(text)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')
    return device


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Profile chunked-prefill throughput and numerical drift.')
    parser.add_argument('--model', choices=_MODEL_CHOICES, default='hybrid')
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--total-len', type=int, default=256)
    parser.add_argument(
        '--schedules', default=','.join(DEFAULT_SCHEDULES),
        help='comma-separated schedule specs ("full", an int, or "a:b:c"); '
             f'default {",".join(DEFAULT_SCHEDULES)}')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--atol', type=float, default=DEFAULT_ATOL)
    parser.add_argument('--rtol', type=float, default=DEFAULT_RTOL)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=0,
                        help='CPU intra-op threads (0 = leave unchanged)')
    parser.add_argument('--json', default=None,
                        help='optional JSON output path')
    args = parser.parse_args(argv)

    if args.total_len < 1:
        parser.error('--total-len must be >= 1')
    if args.d_model < 1:
        parser.error('--d-model must be >= 1')
    if args.repeats < 1:
        parser.error('--repeats must be >= 1')
    if args.warmup < 0:
        parser.error('--warmup must be >= 0')
    if args.atol < 0 or args.rtol < 0:
        parser.error('--atol/--rtol must be >= 0')

    device = _resolve_device(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    elif device.type == 'cpu':
        torch.set_num_threads(1)

    schedules = [s.strip() for s in args.schedules.split(',') if s.strip()]
    if not schedules:
        parser.error('--schedules must contain at least one spec')
    for spec in schedules:
        try:
            expand_prefill_schedule(args.total_len, spec)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

    seed_info = set_global_seed(args.seed, deterministic=True)
    torch.manual_seed(args.seed)
    model = _build_model(args.model, args.d_model).to(device).eval()
    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    x = (torch.randn(1, args.total_len, args.d_model, generator=gen) * 0.1
         ).to(device=device)

    report = profile_prefill_schedules(
        model, x, schedules, repeats=args.repeats, warmup=args.warmup,
        atol=args.atol, rtol=args.rtol)
    report['seed'] = args.seed
    report['seed_info'] = seed_info
    report['torch_version'] = torch.__version__
    report['num_threads'] = torch.get_num_threads()
    report['provenance'] = capture_provenance()

    print('=' * 78)
    print(f"Chunked-prefill profile: {args.model} on {device} "
          f"(T={args.total_len}, d_model={args.d_model})")
    print('=' * 78)
    print(f"{'schedule':>12} {'chunks':>7} {'latency(ms)':>12} "
          f"{'tok/s':>10} {'max|drift|':>12} {'rel-L2':>10} {'pass':>6}")
    print('-' * 78)
    for row in report['rows']:
        drift = row['drift']
        max_abs = ('n/a' if drift['max_abs'] is None
                   else f"{drift['max_abs']:.3e}")
        rel = ('n/a' if drift['relative_l2'] is None
               else f"{drift['relative_l2']:.3e}")
        tok_s = ('n/a' if row['tokens_per_s'] is None
                 else f"{row['tokens_per_s']:.0f}")
        print(f"{row['schedule']:>12} {row['n_chunks']:>7} "
              f"{row['latency_ms']:>12.3f} {tok_s:>10} {max_abs:>12} "
              f"{rel:>10} {str(row['passed']):>6}")
    print('-' * 78)
    ref_max = report['reference_max_abs']
    ref_max_text = 'n/a' if ref_max is None else f'{ref_max:.3e}'
    print(f"Tolerance: {report['tolerance']['criterion']} "
          f"(atol={report['atol']:.1e}, rtol={report['rtol']:.1e}, "
          f"max|ref|={ref_max_text})")
    print('Latency is host/thread-specific; only the drift column is portable.')

    if args.json:
        path = os.path.abspath(args.json)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        write_json_atomic(report, path, indent=2, sort_keys=True,
                          allow_nan=False, trailing_newline=True)
        print(f'Saved: {path}')

    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
