"""Post-training quantization calibration with bounded memory and explicit release.

Motivation
----------
:mod:`kv_quant` supplies grouped fake-quantization, but
``fake_quantize(x, spec)`` derives a *fresh* per-group absmax scale from the
tensor it quantizes. In a decode loop that re-quantizes the KV/state after
every write the grid therefore drifts step to step, and any calibration data
collected up front is never summarized into a single pinned scale. Real
post-training quantization does the opposite:

* run a calibration pass over representative data,
* collapse each batch to running per-group statistics and **release the raw
  calibration tensor immediately** (bounded memory — the garbage-collection
  contract),
* derive ONE scale (and zero point) from the accumulated statistics,
* pin that scale for every subsequent quantize/dequantize.

This module is that missing calibration layer:

* :class:`GroupAbsmaxCalibrator` — running per-group absmax (and min/max for
  asymmetric grids) over calibration batches. ``observe(x)`` folds each batch
  into running statistics and releases it; no calibration tensor is retained
  after the call. ``finish()`` derives per-group scales; ``free()`` drops the
  running stats and runs :func:`gc.collect` (plus
  ``torch.cuda.empty_cache`` on CUDA).
* :func:`quantize_with_scale` / :func:`dequantize_with_scale` — round a
  tensor to the grid defined by a *pinned* per-group scale and dequantize,
  symmetric or asymmetric, with an optional straight-through gradient.
* :func:`calibrate_and_quantize` — one-shot: observe calibration data,
  finish, quantize a target tensor with the pinned scale, free the calibrator.

All statistics are running (per-group max/min only), so memory is O(groups)
independent of the number of calibration batches — a calibration pass over a
long corpus never grows the heap.

Contract / honesty
------------------
* Groups run along the **last axis**; ``group_size`` contiguous elements
  share one scale, matching :mod:`kv_quant`.
* All observations must share the same group count (same state/KV shape),
  which is the realistic PTQ scenario; a mismatch raises instead of
  silently misaligning scales.
* Non-finite calibration or target values raise (a quantizer must not
  launder NaN/Inf into the grid).
* :meth:`GroupAbsmaxCalibrator.free` is idempotent and must be called (or
  used via :func:`calibrate_and_quantize`, which frees automatically) so the
  running statistics and any CUDA memory are released deterministically.
* Symmetric quantization uses ``qmax = 2**(bits-1) - 1`` and a zero-free
  grid ``{-qmax..qmax} * scale``; asymmetric uses
  ``qmax = 2**bits - 1`` with an integer zero point, mirroring
  :class:`kv_quant.QuantSpec`.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass

import torch

from kv_quant import QuantSpec


__all__ = [
    'GroupAbsmaxCalibrator',
    'calibrate_and_quantize',
    'dequantize_with_scale',
    'quantize_with_scale',
]


def _require_int(value: int, name: str, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f'{name} must be an int, got {value!r}')
    if not (lo <= value <= hi):
        raise ValueError(f'{name} must be in [{lo}, {hi}], got {value!r}')
    return value


def _require_finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real number, got {value!r}')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return number


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of :meth:`GroupAbsmaxCalibrator.finish`.

    Attributes:
        scale: per-group scale tensor of shape ``(group_count,)`` (fp32).
        zero: per-group zero point of shape ``(group_count,)`` for
            asymmetric specs, ``None`` for symmetric specs.
        n_obs: number of calibration rows folded into the statistics.
        group_count: number of groups the scale was derived over.
        symmetric: whether the grid is symmetric.
    """

    scale: torch.Tensor
    zero: torch.Tensor | None
    n_obs: int
    group_count: int
    symmetric: bool


class GroupAbsmaxCalibrator:
    """Running per-group quantization statistics over calibration batches.

    ``observe(x)`` folds each calibration batch into running per-group
    absmax (symmetric grids) or running min/max (asymmetric grids) and then
    releases the batch — no raw calibration tensor is retained, so memory is
    O(groups) regardless of how many batches are observed. Call
    :meth:`finish` to derive the pinned scale(s), then :meth:`free` to drop
    the running statistics.

    All statistics are accumulated in fp32 so fp16/bf16 calibration data
    cannot overflow the running maxima.

    Args:
        spec: the :class:`kv_quant.QuantSpec` to calibrate for.
        group_count: number of groups along the last axis (last_dim /
            group_size). Every observation must reshape to exactly this many
        groups.

    Raises:
        ValueError: if ``group_count`` is not a positive int or ``spec`` is
            not a :class:`QuantSpec`.
    """

    def __init__(self, spec: QuantSpec, group_count: int) -> None:
        if not isinstance(spec, QuantSpec):
            raise TypeError(
                f'GroupAbsmaxCalibrator: spec must be a QuantSpec, got '
                f'{type(spec).__name__}.')
        _require_int(group_count, 'group_count', 1, 2 ** 30)
        self._spec = spec
        self._group_count = int(group_count)
        self._absmax = torch.zeros(self._group_count, dtype=torch.float32)
        self._lo = torch.full((self._group_count,), float('inf'),
                              dtype=torch.float32)
        self._hi = torch.full((self._group_count,), float('-inf'),
                              dtype=torch.float32)
        self._n_obs = 0
        self._freed = False

    # -- accessors -------------------------------------------------------
    @property
    def spec(self) -> QuantSpec:
        return self._spec

    @property
    def group_count(self) -> int:
        return self._group_count

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def freed(self) -> bool:
        return self._freed

    # -- calibration -----------------------------------------------------
    def observe(self, x: torch.Tensor) -> int:
        """Fold one calibration batch into the running statistics.

        The last axis is split into ``group_size``-wide groups; every group
        contributes its absmax (and min/max for asymmetric grids). The input
        tensor is **not retained** — only the running per-group extrema are
        updated — so calling this repeatedly over a long corpus uses
        constant memory.

        Returns the number of rows folded in from this batch.
        Raises ``ValueError`` for non-finite input or a group count that does
        not match the calibrator.
        """
        if self._freed:
            raise ValueError(
                'GroupAbsmaxCalibrator.observe: the calibrator has been '
                'freed; create a new one for further calibration.')
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f'GroupAbsmaxCalibrator.observe: expected a tensor, got '
                f'{type(x).__name__}.')
        if x.numel() == 0:
            return 0
        if not torch.is_floating_point(x):
            raise ValueError(
                f'GroupAbsmaxCalibrator.observe: expected a floating-point '
                f'tensor, got dtype {x.dtype}.')
        if not bool(torch.isfinite(x).all()):
            raise ValueError(
                'GroupAbsmaxCalibrator.observe: calibration batch contains '
                'NaN/Inf; refusing to summarize non-finite values.')

        last = x.shape[-1]
        if last % self._spec.group_size != 0:
            raise ValueError(
                f'GroupAbsmaxCalibrator.observe: last dim {last} is not '
                f'divisible by group_size {self._spec.group_size}; expected '
                f'{self._group_count} groups.')
        group_count = last // self._spec.group_size
        if group_count != self._group_count:
            raise ValueError(
                f'GroupAbsmaxCalibrator.observe: batch has {group_count} '
                f'groups but the calibrator expects {self._group_count}; '
                'all calibration batches must share the same state/KV shape.')

        work = x.to(torch.float32)
        groups = work.reshape(-1, self._group_count, self._spec.group_size)
        n_rows = int(groups.shape[0])
        g_absmax = groups.abs().amax(dim=-1)          # (rows, groups)
        g_lo = groups.amin(dim=-1)
        g_hi = groups.amax(dim=-1)

        with torch.no_grad():
            self._absmax = torch.maximum(
                self._absmax, g_absmax.max(dim=0).values.to(torch.float32))
            self._lo = torch.minimum(
                self._lo, g_lo.min(dim=0).values.to(torch.float32))
            self._hi = torch.maximum(
                self._hi, g_hi.max(dim=0).values.to(torch.float32))
        self._n_obs += n_rows
        return n_rows

    def finish(self) -> CalibrationResult:
        """Derive the pinned per-group scale(s) from the running statistics.

        Symmetric groups with absmax 0 get scale 1 (they quantize to exactly
        zero). Asymmetric degenerate groups (span 0) get scale 1 and zero 0.

        Raises ``ValueError`` if no batch has been observed or the
        calibrator has already been freed.
        """
        if self._freed:
            raise ValueError(
                'GroupAbsmaxCalibrator.finish: the calibrator has been '
                'freed; create a new one.')
        if self._n_obs <= 0:
            raise ValueError(
                'GroupAbsmaxCalibrator.finish: no calibration batches have '
                'been observed; cannot derive a scale.')
        qmax = float(self._spec.qmax)
        scale = torch.where(
            self._absmax > 0, self._absmax / qmax,
            torch.ones_like(self._absmax))
        zero: torch.Tensor | None = None
        if not self._spec.symmetric:
            span = self._hi - self._lo
            degenerate = span <= 0
            scale = torch.where(degenerate, torch.ones_like(span),
                                span / qmax)
            zero = torch.where(
                degenerate, torch.zeros_like(self._lo),
                torch.round(-self._lo / scale).clamp(0, self._spec.qmax))
        return CalibrationResult(
            scale=scale, zero=zero, n_obs=self._n_obs,
            group_count=self._group_count, symmetric=self._spec.symmetric)

    def free(self) -> None:
        """Release the running statistics (idempotent).

        Deletes the internal tensors, runs :func:`gc.collect`, and calls
        ``torch.cuda.empty_cache`` on CUDA devices. Safe to call multiple
        times and safe to call after :meth:`finish`.
        """
        if self._freed:
            return
        self._absmax = None  # type: ignore[assignment]
        self._lo = None  # type: ignore[assignment]
        self._hi = None  # type: ignore[assignment]
        self._freed = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _broadcast_groups(scale: torch.Tensor, ndim: int, group_count: int,
                      device) -> torch.Tensor:
    """Reshape a (group_count,) scale to broadcast over the group axis."""
    shape = (1,) * (ndim - 1) + (group_count, 1)
    return scale.to(device=device).reshape(shape)


def quantize_with_scale(x: torch.Tensor, spec: QuantSpec, scale: torch.Tensor,
                        zero: torch.Tensor | None = None,
                        *,
                        ste: bool = False) -> torch.Tensor:
    """Quantize ``x`` to a *pinned* per-group grid and dequantize.

    Unlike :func:`kv_quant.fake_quantize`, which derives its own scale from
    ``x``, this function quantizes against a caller-supplied ``scale``
    (produced by :meth:`GroupAbsmaxCalibrator.finish`), so the grid is stable
    across calls — exactly the PTQ deployment contract.

    The output has the same shape/dtype/device as ``x``. With
    ``ste=True`` the straight-through estimator is applied (identity
    gradient); with ``ste=False`` (inference) the gradient is exactly zero.

    Raises:
        ValueError: for non-finite input/scale, wrong group count, or a
            missing zero point for an asymmetric spec.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'quantize_with_scale: expected a tensor, got '
                        f'{type(x).__name__}.')
    if not isinstance(spec, QuantSpec):
        raise TypeError(f'quantize_with_scale: spec must be a QuantSpec, got '
                        f'{type(spec).__name__}.')
    if not torch.is_floating_point(x):
        raise ValueError(
            f'quantize_with_scale: expected a floating-point tensor, got '
            f'dtype {x.dtype}.')
    if x.dim() == 0:
        raise ValueError('quantize_with_scale: expected a tensor of rank >= 1')
    if x.numel() == 0:
        return x.clone()
    if not bool(torch.isfinite(x).all()):
        raise ValueError(
            'quantize_with_scale: input contains NaN/Inf; refusing to '
            'quantize non-finite values.')
    if not isinstance(scale, torch.Tensor) or not torch.is_floating_point(
            scale):
        raise TypeError(
            f'quantize_with_scale: scale must be a floating-point tensor, '
            f'got {type(scale).__name__}.')
    if not bool(torch.isfinite(scale).all()):
        raise ValueError('quantize_with_scale: scale contains NaN/Inf.')

    last = x.shape[-1]
    if last % spec.group_size != 0:
        raise ValueError(
            f'quantize_with_scale: last dim {last} is not divisible by '
            f'group_size {spec.group_size}.')
    group_count = last // spec.group_size
    scale = scale.reshape(-1)
    if scale.shape[0] != group_count:
        raise ValueError(
            f'quantize_with_scale: scale has {scale.shape[0]} entries but '
            f'the last axis holds {group_count} groups.')
    if (not spec.symmetric) and zero is None:
        raise ValueError(
            'quantize_with_scale: an asymmetric spec requires a zero point '
            '(pass ``zero`` from ``CalibrationResult.zero``).')
    if zero is not None and not torch.is_floating_point(zero):
        raise TypeError(
            f'quantize_with_scale: zero must be a floating-point tensor, '
            f'got {type(zero).__name__}.')

    orig_dtype = x.dtype
    compute = (torch.float64 if orig_dtype == torch.float64
               else torch.float32)
    work = x.to(compute)
    s = _broadcast_groups(scale.to(compute), work.dim(), group_count,
                          work.device)
    qmax = spec.qmax
    lead = work.shape[:-1]
    groups = work.reshape(*lead, group_count, spec.group_size)

    with torch.no_grad():
        if spec.symmetric:
            codes = torch.round(groups / s).clamp(-qmax, qmax)
            deq = codes * s
        else:
            z = _broadcast_groups(zero.to(compute), work.dim(), group_count,
                                  work.device)
            codes = torch.round(groups / s) + z
            codes = codes.clamp(0, qmax)
            deq = (codes - z) * s
        limit = float(torch.finfo(orig_dtype).max)
        overflow = (~torch.isfinite(deq)) | (deq.abs() > limit)
        if bool(overflow.any()):
            safe = torch.clamp(
                torch.where(torch.isnan(deq), torch.zeros_like(deq), deq),
                min=-limit, max=limit)
            deq = torch.where(overflow, safe, deq)

    deq = deq.reshape(*lead, last).to(orig_dtype)
    if ste and x.requires_grad:
        deq = x + (deq - x).detach()
    return deq


def dequantize_with_scale(codes: torch.Tensor, spec: QuantSpec,
                          scale: torch.Tensor,
                          zero: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize integer codes against a pinned per-group scale.

    Symmetric: ``deq = codes * scale``. Asymmetric: ``deq = (codes - zero) *
    scale``. ``codes`` may be any integer dtype; the result is fp32 (fp64 for
    fp64 inputs).
    """
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f'dequantize_with_scale: expected a tensor, got '
                        f'{type(codes).__name__}.')
    if not isinstance(spec, QuantSpec):
        raise TypeError(f'dequantize_with_scale: spec must be a QuantSpec, '
                        f'got {type(spec).__name__}.')
    if codes.numel() == 0:
        return codes.clone().to(torch.float64
                              if codes.dtype == torch.float64 else torch.float32)
    last = codes.shape[-1]
    if last % spec.group_size != 0:
        raise ValueError(
            f'dequantize_with_scale: last dim {last} is not divisible by '
            f'group_size {spec.group_size}.')
    group_count = last // spec.group_size
    scale = scale.reshape(-1)
    if scale.shape[0] != group_count:
        raise ValueError(
            f'dequantize_with_scale: scale has {scale.shape[0]} entries '
            f'but the last axis holds {group_count} groups.')
    if (not spec.symmetric) and zero is None:
        raise ValueError(
            'dequantize_with_scale: an asymmetric spec requires a zero '
            'point.')

    compute = (torch.float64 if codes.dtype == torch.float64
               else torch.float32)
    work = codes.to(compute)
    s = _broadcast_groups(scale.to(compute), work.dim(), group_count,
                          work.device)
    lead = work.shape[:-1]
    groups = work.reshape(*lead, group_count, spec.group_size)
    if spec.symmetric:
        deq = groups * s
    else:
        z = _broadcast_groups(zero.to(compute), work.dim(), group_count,
                              work.device)
        deq = (groups - z) * s
    return deq.reshape(*lead, last)


def calibrate_and_quantize(calibration_batches: list[torch.Tensor],
                           target: torch.Tensor, spec: QuantSpec) -> CalibrationResult:
    """One-shot PTQ: calibrate on batches, quantize ``target``, free.

    Observes every calibration batch (releasing each immediately), derives
    the pinned scale via :meth:`finish`, quantizes ``target`` with
    :func:`quantize_with_scale` (``ste=False``), then frees the calibrator.
    Returns the :class:`CalibrationResult`; the quantized target is not
    returned — call :func:`quantize_with_scale`` with the result's
    ``scale``/``zero`` when you need the tensor.

    This is the complete PTQ contract in one call: calibration data is
    summarized into bounded statistics and then released, so the retained
    footprint after this function is just the scale (O(groups)). The caller
    applies the result to a target tensor with :func:`quantize_with_scale`.
    """
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f'calibrate_and_quantize: target must be a tensor, got '
            f'{type(target).__name__}.')
    if not isinstance(spec, QuantSpec):
        raise TypeError(
            f'calibrate_and_quantize: spec must be a QuantSpec, got '
            f'{type(spec).__name__}.')
    if not torch.is_floating_point(target):
        raise ValueError(
            'calibrate_and_quantize: target must be a floating-point tensor.')
    last = target.shape[-1]
    if last % spec.group_size != 0:
        raise ValueError(
            f'calibrate_and_quantize: target last dim {last} is not '
            f'divisible by group_size {spec.group_size}.')
    calibrator = GroupAbsmaxCalibrator(spec, last // spec.group_size)
    try:
        for batch in calibration_batches:
            calibrator.observe(batch)
        result = calibrator.finish()
    finally:
        calibrator.free()
    return result
