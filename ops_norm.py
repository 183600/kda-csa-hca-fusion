"""Shape-polymorphic RMSNorm reference implementation.

The fused operators in this repository implement the DeepSeek-V4 §2.3.3
normalization form as a private helper (``ops_csa._qk_normalize``).  That
helper is exactly right for its call sites, but it only normalizes the last
axis and cannot be reused without copying the formula.  This module exposes
the same arithmetic as a small, dependency-free reference primitive whose
semantics are pinned by tests in ``run_correctness.py``:

    rms_norm(x) = x / sqrt(mean(x**2) + eps)

The reduction spans any positive number of trailing axes (the default is the
last axis, matching the operator call sites), and arbitrary leading axes are
supported.  An optional affine transform (``weight`` / ``bias``) is applied
after normalization, mirroring ``torch.nn.RMSNorm``.

The implementation deliberately uses ``torch.rsqrt`` (as the operators do)
instead of ``1 / torch.sqrt`` so that calling it from ``_qk_normalize`` is
bit-identical to the previous inline expression.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

RMS_NORM_EPS = 1e-6
"""Default epsilon, matching the paper-form normalization in ``ops_csa``."""

__all__ = ["RMS_NORM_EPS", "rms_norm", "rms_norm_"]


def _resolve_normalized_shape(
    x: torch.Tensor,
    normalized_shape: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Validate ``x`` and return the trailing shape used by the reduction."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"rms_norm: x must be a torch.Tensor, got {type(x)!r}")
    if not x.is_floating_point():
        raise TypeError(
            f"rms_norm: x must be floating point, got dtype={x.dtype}")
    if x.dim() == 0:
        raise ValueError("rms_norm: x must have at least one dimension")

    if normalized_shape is None:
        shape = (int(x.shape[-1]),)
    elif isinstance(normalized_shape, int):
        shape = (normalized_shape,)
    else:
        try:
            shape = tuple(normalized_shape)
        except TypeError as exc:
            raise TypeError(
                "rms_norm: normalized_shape must be an int, a sequence of "
                f"ints, or None, got {normalized_shape!r}") from exc

    if not shape:
        raise ValueError("rms_norm: normalized_shape must not be empty")
    if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0
           for d in shape):
        raise ValueError(
            "rms_norm: normalized_shape entries must be positive ints, got "
            f"{shape}")
    if len(shape) > x.dim():
        raise ValueError(
            f"rms_norm: normalized_shape={shape} has more dimensions than "
            f"x (rank {x.dim()})")
    trailing = tuple(int(d) for d in x.shape[-len(shape):])
    if trailing != shape:
        raise ValueError(
            f"rms_norm: x.shape[-{len(shape)}:]={trailing} does not match "
            f"normalized_shape={shape}")
    return shape


def _check_eps(eps: float) -> float:
    """Reject eps values that would silently produce NaNs/negative variances."""
    try:
        value = float(eps)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"rms_norm: eps must be a real number, got {eps!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"rms_norm: eps must be finite and non-negative, got {eps!r}")
    return value


def _check_affine(
    x: torch.Tensor,
    name: str,
    value: torch.Tensor | None,
    normalized_shape: tuple[int, ...],
) -> None:
    """Validate an optional weight/bias against the normalized trailing shape."""
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"rms_norm: {name} must be a torch.Tensor or None, got "
            f"{type(value)!r}")
    if tuple(value.shape) != normalized_shape:
        raise ValueError(
            f"rms_norm: {name}.shape={tuple(value.shape)} must equal "
            f"normalized_shape={normalized_shape}")
    if value.device != x.device:
        raise ValueError(
            f"rms_norm: {name} is on {value.device} but x is on {x.device}")
    if value.dtype != x.dtype:
        raise ValueError(
            f"rms_norm: {name}.dtype={value.dtype} must match x.dtype={x.dtype}")


def _reduction_dims(x: torch.Tensor, normalized_shape: tuple[int, ...]):
    """Return the reduction dims, using the scalar form for the common case."""
    first = x.dim() - len(normalized_shape)
    if len(normalized_shape) == 1:
        return first
    return tuple(range(first, x.dim()))


def rms_norm(
    x: torch.Tensor,
    eps: float = RMS_NORM_EPS,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    normalized_shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """Reference RMSNorm over the trailing ``normalized_shape`` axes.

    Args:
        x: Floating-point tensor with arbitrary leading axes.  Empty leading
            axes (e.g. shape ``(0, D)``) are supported and produce an empty
            output of the same shape.
        eps: Stabilizer added to the mean square *before* the reciprocal
            square root (the paper/operator convention), finite and >= 0.
        weight: Optional multiplicative gain with shape ``normalized_shape``.
        bias: Optional additive offset with shape ``normalized_shape``.
        normalized_shape: Trailing shape to normalize.  ``None`` (default)
            normalizes only the last axis; an int is promoted to a 1-tuple.

    Returns:
        A tensor with the same shape, dtype, and device as ``x``.  The
        computation itself is performed in ``x.dtype`` (no hidden upcast), so
        float64 inputs give a float64 reference.

    Raises:
        TypeError: for non-tensor / non-floating-point inputs or non-real eps.
        ValueError: for invalid shapes, a non-positive normalized axis, a
            negative/non-finite eps, or mismatched affine parameters.
    """
    shape = _resolve_normalized_shape(x, normalized_shape)
    eps = _check_eps(eps)
    _check_affine(x, "weight", weight, shape)
    _check_affine(x, "bias", bias, shape)

    dims = _reduction_dims(x, shape)
    mean_square = x.pow(2).mean(dim=dims, keepdim=True)
    y = x * torch.rsqrt(mean_square + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm_(
    x: torch.Tensor,
    eps: float = RMS_NORM_EPS,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    normalized_shape: int | Sequence[int] | None = None,
) -> torch.Tensor:
    """In-place :func:`rms_norm`, returning ``x``.

    The caller must own ``x`` (never a view of cache storage or an input that
    is still read elsewhere).  Autograd is not supported: in-place writes
    invalidate the saved tensors used by the backward of the mean square, so
    use :func:`rms_norm` when gradients are required.  The elementwise chain
    is identical to :func:`rms_norm`, making the result bit-identical.
    """
    shape = _resolve_normalized_shape(x, normalized_shape)
    eps = _check_eps(eps)
    _check_affine(x, "weight", weight, shape)
    _check_affine(x, "bias", bias, shape)
    if x.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(
            "rms_norm_ does not support autograd; use rms_norm instead")

    dims = _reduction_dims(x, shape)
    mean_square = x.pow(2).mean(dim=dims, keepdim=True)
    x.mul_(torch.rsqrt(mean_square + eps))
    if weight is not None:
        x.mul_(weight)
    if bias is not None:
        x.add_(bias)
    return x
