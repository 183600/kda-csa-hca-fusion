"""Grouped low-bit KV quantization primitives + byte accounting.

Motivation
----------
Every cost in this repository is priced in KV-side *elements*
(``kv_budget`` / ``run_kv_cache``): fp16/fp32 scalars. Real long-context
serving stores that state **quantized** (int8 / 4-bit grouped) to trade
recall for bytes, and nothing in the repo could express that trade. This
module adds the missing second axis:

* :class:`QuantSpec` — a validated integer storage contract: payload bits
  per element, group size, symmetric/asymmetric, metadata bits.
* :class:`FP8Spec` — a block-scaled FP8 contract (E4M3FN/E5M2 payload,
  shared E8M0 or fp16 scale), the format real FP8 state caches use.
* :func:`fake_quantize` — exact grouped round-to-grid + dequantize with a
  straight-through gradient, so a decoder's quantize/dequantize kernels can
  be simulated in the full-precision graph without low-bit kernels.
* :func:`fake_quantize_` — the in-place, inference-only variant for
  re-quantizing a state/cache buffer after every decode write (same
  numerics, no second tensor, rejects autograd-tracked inputs).
* :func:`quantization_error` — observed max/RMS error and SNR (dB).
* byte accounting — :func:`bits_per_element` / :func:`bytes_per_element` /
  :func:`storage_bytes` / :func:`compression_ratio`, charging the
  per-group scale (and optional zero point) metadata exactly as a real
  cache layout would.
* :func:`quantized_row_bytes` — converts any element row produced by
  ``kv_budget`` (operator-level or layout-level) into component-wise byte
  columns under a component→:class:`QuantSpec` / :class:`FP8Spec` policy.

Contract / honesty
------------------
* ``bits`` are **payload** bits per element. ``group_size`` elements share
  one scale (and one zero point when asymmetric). Metadata is charged per
  group: ``(scale_bits + zero_bits) / group_size`` extra bits per element.
  ``scale_bits=0`` / ``zero_bits=0`` give the idealized "free metadata"
  lower bound; the defaults charge fp16 scales and same-width zero points.
* Grouping is along the **last tensor axis** (the flattened KV/state inner
  dimension), matching grouped per-channel quantization layouts where
  ``group_size`` contiguous elements share a scale.
* Symmetric quantization uses ``qmax = 2**(bits-1) - 1`` and an absmax
  scale, so the grid is ``{-qmax..qmax} * scale``. Asymmetric uses
  ``qmax = 2**bits - 1``, a min/max scale and an integer zero point.
* ``fake_quantize`` computes scales in fp32 under ``no_grad`` and applies
  the straight-through estimator by default: gradients w.r.t. the input
  are exactly 1 almost everywhere (pinned by tests). This is the standard
  QAT contract; inference-only callers can pass ``ste=False``.
* Non-finite inputs are rejected (a quantizer must never silently turn
  NaN/Inf into a plausible-looking grid value), mirroring the repo-wide
  non-finite handling of the KDA operators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

FP16_BITS = 16.0
FP16_BYTES_PER_ELEMENT = FP16_BITS / 8.0
FP8_BITS = 8.0

_FP8_FORMATS = {
    'e4m3fn': {
        'ebits': 4,
        'mbits': 3,
        'max_finite': 448.0,
        'min_normal': 2.0 ** -6,
    },
    'e5m2': {
        'ebits': 5,
        'mbits': 2,
        'max_finite': 57344.0,
        'min_normal': 2.0 ** -14,
    },
}


def _require_int(value, name: str, lo: int, hi: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f'{name} must be an int, got {value!r}')
    if not (lo <= value <= hi):
        raise ValueError(f'{name} must be in [{lo}, {hi}], got {value!r}')
    return value


@dataclass(frozen=True)
class QuantSpec:
    """Storage contract for one grouped-quantized tensor.

    Args:
        bits: payload bits per element (int in ``[2, 16]``).
        group_size: elements sharing one scale / zero point (``>= 1``).
        symmetric: ``True`` → absmax-scaled signed grid
            ``{-qmax..qmax} * scale`` with ``qmax = 2**(bits-1) - 1``;
            ``False`` → min/max-scaled unsigned grid with a zero point.
        scale_bits: bits charged per stored scale (default 16, i.e. fp16).
        zero_bits: bits charged per stored zero point. Meaningful only for
            ``symmetric=False``; defaults to ``bits`` when omitted. A value
            of ``0`` means "not stored" for either mode.
    """

    bits: int
    group_size: int = 64
    symmetric: bool = True
    scale_bits: int = 16
    zero_bits: int | None = None

    def __post_init__(self) -> None:
        _require_int(self.bits, 'QuantSpec.bits', 2, 16)
        _require_int(self.group_size, 'QuantSpec.group_size', 1, 2 ** 30)
        _require_int(self.scale_bits, 'QuantSpec.scale_bits', 0, 64)
        if not isinstance(self.symmetric, bool):
            raise ValueError(
                f'QuantSpec.symmetric must be a bool, got {self.symmetric!r}')
        if self.zero_bits is not None:
            _require_int(self.zero_bits, 'QuantSpec.zero_bits', 0, 64)
        if self.symmetric and self.zero_bits not in (None, 0):
            raise ValueError(
                'QuantSpec: symmetric quantization has no zero point, so '
                f'zero_bits must be None or 0, got {self.zero_bits!r}.')
        if not self.symmetric and self.zero_bits is None:
            # Default: store the zero point at the same width as the codes.
            object.__setattr__(self, 'zero_bits', self.bits)

    @property
    def qmax(self) -> int:
        """Largest quantized code magnitude (symmetric) / code (asymmetric)."""
        if self.symmetric:
            return 2 ** (self.bits - 1) - 1
        return 2 ** self.bits - 1

    @property
    def metadata_bits(self) -> float:
        """Metadata bits charged per element (scale + optional zero point)."""
        return (self.scale_bits + (self.zero_bits or 0)) / self.group_size


@dataclass(frozen=True)
class FP8Spec:
    """Block-scaled 8-bit floating-point storage contract.

    This is the FP8 counterpart of :class:`QuantSpec`: the payload is a
    real IEEE-style FP8 element (E4M3FN or E5M2), not an integer code, and
    one scale is shared by a contiguous block of ``group_size`` elements.
    The default mirrors MXFP8 — 32-element blocks with an 8-bit (E8M0,
    power-of-two) shared scale — so the storage cost includes the scale
    actually needed to decode the block.

    Args:
        format: ``'e4m3fn'`` (4 exponent bits, 3 mantissa bits, finite
            range ±448, NaN at the two all-ones codes, no Inf) or
            ``'e5m2'`` (5 exponent bits, 2 mantissa bits, finite range
            ±57344, Inf/NaN at exponent 31).
        group_size: elements sharing one scale (``>= 1``).
        scale_bits: bits charged per stored scale (default 8 = E8M0;
            use 16 for an fp16 scale).
        power_of_two: ``True`` snaps the per-block scale up to the next
            power of two (the MXFP8/E8M0 contract); ``False`` uses the
            exact absmax scale. Power-of-two scales can never overflow the
            FP8 format, while exact scales use the full mantissa range.
    """

    format: str = 'e4m3fn'
    group_size: int = 32
    scale_bits: int = 8
    power_of_two: bool = True

    def __post_init__(self) -> None:
        if self.format not in _FP8_FORMATS:
            raise ValueError(
                f'FP8Spec.format must be one of {sorted(_FP8_FORMATS)}, '
                f'got {self.format!r}')
        _require_int(self.group_size, 'FP8Spec.group_size', 1, 2 ** 30)
        _require_int(self.scale_bits, 'FP8Spec.scale_bits', 0, 64)
        if not isinstance(self.power_of_two, bool):
            raise ValueError(
                f'FP8Spec.power_of_two must be a bool, got '
                f'{self.power_of_two!r}')

    @property
    def payload_bits(self) -> int:
        return 8

    @property
    def max_finite(self) -> float:
        return _FP8_FORMATS[self.format]['max_finite']

    @property
    def min_normal(self) -> float:
        return _FP8_FORMATS[self.format]['min_normal']

    @property
    def metadata_bits(self) -> float:
        """Metadata bits charged per element (the shared block scale)."""
        return self.scale_bits / self.group_size


StorageSpec = QuantSpec | FP8Spec


def _require_storage_spec(spec, name: str) -> StorageSpec:
    if not isinstance(spec, (QuantSpec, FP8Spec)):
        raise TypeError(
            f'{name}: expected QuantSpec or FP8Spec, got {spec!r}')
    return spec


def bits_per_element(spec: StorageSpec) -> float:
    """Payload + metadata bits per stored element."""
    _require_storage_spec(spec, 'bits_per_element')
    if isinstance(spec, FP8Spec):
        return FP8_BITS + spec.metadata_bits
    return float(spec.bits) + spec.metadata_bits


def bytes_per_element(spec: StorageSpec) -> float:
    """Effective bytes per stored element (payload + metadata)."""
    return bits_per_element(spec) / 8.0


def storage_bytes(n_elements, spec: StorageSpec) -> float:
    """Bytes required to store ``n_elements`` under ``spec`` (fractional ok)."""
    if (not isinstance(n_elements, (int, float))
            or isinstance(n_elements, bool)
            or not math.isfinite(n_elements)):
        raise ValueError(
            f'storage_bytes: n_elements must be a finite number, got '
            f'{n_elements!r}')
    if n_elements < 0:
        raise ValueError(
            f'storage_bytes: n_elements must be >= 0, got {n_elements!r}')
    return float(n_elements) * bytes_per_element(spec)


def compression_ratio(spec: StorageSpec,
                      baseline_bits: float = FP16_BITS) -> float:
    """Byte reduction factor vs an un-quantized ``baseline_bits`` layout.

    ``> 1`` means quantized storage is smaller. Undefined (raises) for a
    non-positive baseline.
    """
    if (not isinstance(baseline_bits, (int, float))
            or isinstance(baseline_bits, bool)
            or not math.isfinite(baseline_bits) or baseline_bits <= 0):
        raise ValueError(
            f'compression_ratio: baseline_bits must be finite and positive, '
            f'got {baseline_bits!r}')
    return float(baseline_bits) / bits_per_element(spec)


def spec_summary(spec: StorageSpec,
                 baseline_bits: float = FP16_BITS) -> dict:
    """Flat, JSON-safe description of a spec (for provenance metadata)."""
    _require_storage_spec(spec, 'spec_summary')
    if isinstance(spec, FP8Spec):
        return {
            'format': spec.format,
            'bits': spec.payload_bits,
            'group_size': spec.group_size,
            'symmetric': True,
            'scale_bits': spec.scale_bits,
            'zero_bits': 0,
            'power_of_two_scale': spec.power_of_two,
            'bits_per_element': bits_per_element(spec),
            'bytes_per_element': bytes_per_element(spec),
            'compression_vs_fp16': compression_ratio(spec, baseline_bits),
        }
    return {
        'bits': spec.bits,
        'group_size': spec.group_size,
        'symmetric': spec.symmetric,
        'scale_bits': spec.scale_bits,
        'zero_bits': spec.zero_bits,
        'bits_per_element': bits_per_element(spec),
        'bytes_per_element': bytes_per_element(spec),
        'compression_vs_fp16': compression_ratio(spec, baseline_bits),
    }


def _pad_last_dim(x: torch.Tensor, group_size: int):
    """Pad the last dim up to a multiple of ``group_size``; return (padded, n)."""
    n = x.shape[-1]
    pad = (-n) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, n


def _fp8_round_magnitude(values: torch.Tensor, fmt: str) -> torch.Tensor:
    """Round non-negative magnitudes to the finite FP8 grid (ties to even).

    The result saturates at ``max_finite`` instead of overflowing to Inf
    (E5M2) or NaN (E4M3FN): a quantizer that silently emits non-finite
    values is not a storage contract. Subnormal magnitudes are rounded on
    the format's subnormal step, so the grid is exact at both ends.
    """
    meta = _FP8_FORMATS[fmt]
    max_finite = meta['max_finite']
    min_normal = meta['min_normal']
    mbits = meta['mbits']
    values = values.clamp(max=max_finite)
    exp = torch.floor(torch.log2(values.clamp(min=min_normal)))
    normal_step = torch.exp2(exp - mbits)
    normal = torch.round(values / normal_step) * normal_step
    subnormal_step = min_normal / (2.0 ** mbits)
    subnormal = torch.round(values / subnormal_step) * subnormal_step
    return torch.where(values < min_normal, subnormal, normal).clamp(
        max=max_finite)


def _fp8_group_scale(groups: torch.Tensor, spec: FP8Spec) -> torch.Tensor:
    with torch.no_grad():
        amax = groups.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / spec.max_finite,
                            torch.ones_like(amax))
        if spec.power_of_two:
            finfo = torch.finfo(groups.dtype)
            if groups.dtype == torch.float64:
                min_scale = finfo.tiny * finfo.eps
                exp_min, exp_max = -1074.0, 1023.0
            else:
                min_scale = finfo.tiny
                exp_min, exp_max = -126.0, 127.0
            exp = torch.ceil(torch.log2(scale.clamp(min=min_scale)))
            exp = exp.clamp(exp_min, exp_max)
            scale = torch.exp2(exp)
    return scale


def _fake_quantize_fp8(x: torch.Tensor, spec: FP8Spec,
                       *, ste: bool) -> torch.Tensor:
    orig_dtype = x.dtype
    compute_dtype = (torch.float64 if orig_dtype == torch.float64
                     else torch.float32)
    work = x.to(compute_dtype)
    padded, n = _pad_last_dim(work, spec.group_size)
    lead = padded.shape[:-1]
    groups = padded.reshape(*lead, -1, spec.group_size)
    scale = _fp8_group_scale(groups, spec)
    magnitude = groups.abs() / scale
    codes = _fp8_round_magnitude(magnitude, spec.format)
    deq = torch.sign(groups) * codes * scale
    deq = deq.reshape(*lead, -1)[..., :n].to(orig_dtype)
    if ste and x.requires_grad:
        deq = x + (deq - x).detach()
    return deq


def fake_quantize(x: torch.Tensor, spec: StorageSpec,
                  *, ste: bool = True) -> torch.Tensor:
    """Quantize ``x`` to ``spec``'s grid and dequantize, in full precision.

    Groups run along the last axis; the grid is per-group. The output has
    the same shape/dtype/device as ``x``. With ``ste=True`` (default) the
    straight-through gradient is applied, so autograd sees
    ``d(output)/d(input) == 1`` almost everywhere.

    ``spec`` may be a :class:`QuantSpec` (integer round-to-grid) or an
    :class:`FP8Spec` (block-scaled E4M3FN/E5M2 round-to-nearest-even).

    Raises:
        ValueError: if ``x`` is not a floating-point tensor or contains
            non-finite values (a quantizer must not launder NaN/Inf into
            the grid).
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'fake_quantize: expected a tensor, got {type(x)!r}')
    if not torch.is_floating_point(x):
        raise ValueError(
            f'fake_quantize: expected a floating-point tensor, got dtype '
            f'{x.dtype}')
    if x.dim() == 0:
        raise ValueError('fake_quantize: expected a tensor of rank >= 1')
    _require_storage_spec(spec, 'fake_quantize')
    if x.numel() == 0:
        return x.clone()
    if not bool(torch.isfinite(x).all()):
        raise ValueError(
            'fake_quantize: input contains NaN/Inf; refusing to quantize '
            'non-finite values (they would map to an arbitrary grid point).')
    if isinstance(spec, FP8Spec):
        return _fake_quantize_fp8(x, spec, ste=ste)

    orig_dtype = x.dtype
    compute_dtype = (torch.float64 if orig_dtype == torch.float64
                     else torch.float32)
    work = x.to(compute_dtype)
    padded, n = _pad_last_dim(work, spec.group_size)
    lead = padded.shape[:-1]
    groups = padded.reshape(*lead, -1, spec.group_size)
    qmax = spec.qmax

    if spec.symmetric:
        # absmax scale per group. Zero groups get scale 1 so the codes and
        # the reconstruction are exactly zero.
        with torch.no_grad():
            amax = groups.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(amax > 0, amax / qmax,
                                torch.ones_like(amax))
        codes = torch.round(groups / scale).clamp(-qmax, qmax)
        deq = codes * scale
    else:
        # min/max scale + integer zero point per group. The representable
        # range must include zero: the unsigned code grid is ``[0, qmax]``
        # and the zero point is clamped into that range, so a group whose
        # values are all positive (or all negative) would otherwise have
        # its far end pushed past the grid and saturated to a constant.
        # Constant groups reconstruct their value exactly.
        with torch.no_grad():
            zero_t = torch.zeros_like(groups[..., :1])
            lo = torch.minimum(groups.amin(dim=-1, keepdim=True), zero_t)
            hi = torch.maximum(groups.amax(dim=-1, keepdim=True), zero_t)
            span = hi - lo
            degenerate = span <= 0
            scale = torch.where(degenerate, torch.ones_like(span),
                                span / qmax)
            zero = torch.where(
                degenerate, torch.zeros_like(lo),
                torch.round(-lo / scale).clamp(0, qmax))
        codes = torch.round(groups / scale) + zero
        codes = codes.clamp(0, qmax)
        deq = (codes - zero) * scale
        deq = torch.where(degenerate, hi, deq)

    deq = deq.reshape(*lead, -1)[..., :n].to(orig_dtype)
    if ste and x.requires_grad:
        # Straight-through: forward value from the grid, identity gradient.
        deq = x + (deq - x).detach()
    return deq


def fake_quantize_(x: torch.Tensor, spec: StorageSpec) -> torch.Tensor:
    """In-place :func:`fake_quantize` for inference-time cache writes.

    Overwrites ``x`` with its quantize→dequantize image and returns ``x``,
    so a decode loop can re-quantize a state buffer after every write
    without allocating (and retaining) a second tensor:

        state = fake_quantize_(state, spec)   # inference, no grad

    Contract / honesty:

    * **Inference-only.** ``x.requires_grad=True`` is rejected: in-place
      writes to an autograd-tracked tensor either error or silently corrupt
      the graph, and the STE (QAT) path must go through the functional
      :func:`fake_quantize`, which returns a new tensor and keeps gradients.
    * The quantizer itself is computed out-of-place and then copied back
      under ``torch.no_grad()``, so the numerical result is EXACTLY
      :func:`fake_quantize` with ``ste=False`` (pinned by tests).
    * Non-finite inputs raise (via :func:`fake_quantize`) instead of
      writing a plausible-looking grid value into the buffer.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'fake_quantize_: expected a tensor, got {type(x)!r}')
    if x.requires_grad:
        raise ValueError(
            'fake_quantize_: x.requires_grad=True — the in-place quantizer '
            'is inference-only. Use fake_quantize(x, spec, ste=True) for '
            'QAT / straight-through gradients.')
    deq = fake_quantize(x, spec, ste=False)
    with torch.no_grad():
        x.copy_(deq)
    return x


def quantization_error(x: torch.Tensor, spec: StorageSpec) -> dict:
    """Observed error of ``fake_quantize(x, spec)`` vs ``x``.

    Returns ``max_abs_error``, ``rms_error``, ``signal_rms`` and ``snr_db``
    (``inf`` for an exactly representable input). Errors are measured in
    the input's compute precision (fp32/fp64).
    """
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        raise ValueError('quantization_error: x must be a non-empty tensor')
    deq = fake_quantize(x, spec, ste=False)
    compute_dtype = (torch.float64 if x.dtype == torch.float64
                     else torch.float32)
    xf, df = x.to(compute_dtype), deq.to(compute_dtype)
    err = xf - df
    rms_err = float(err.pow(2).mean().sqrt())
    signal_rms = float(xf.pow(2).mean().sqrt())
    if rms_err == 0.0:
        snr_db = math.inf
    elif signal_rms == 0.0:
        snr_db = -math.inf
    else:
        snr_db = 20.0 * math.log10(signal_rms / rms_err)
    return {
        'max_abs_error': float(err.abs().max()),
        'rms_error': rms_err,
        'signal_rms': signal_rms,
        'snr_db': snr_db,
        'bits_per_element': bits_per_element(spec),
    }


# --------------------------------------------------------------------------
# Budget-row byte accounting (the kv_budget bridge)
# --------------------------------------------------------------------------

_READ_SUFFIX = '_read_elements'
_STATE_KEY = 'state_elements'
_TOTAL_KEY = 'total_read_elements'


# ``core_read_elements`` in a kv_budget operator row already includes its
# per-branch read components (CSA: selected + window; HCA: dense + window).
# They are exposed separately for diagnosis, but must not be added to the
# aggregate a second time when converting elements to bytes.
_CORE_SUBCOMPONENT_STEMS = frozenset({'selected', 'window', 'dense'})


def _row_component_stems(row: dict) -> list[str]:
    """Stems of the per-component element fields of a kv_budget row.

    ``foo_read_elements`` → ``foo``; ``state_elements`` → ``state``.
    The aggregate ``total_read_elements`` is not a component.
    """
    stems = []
    for key in row:
        if key == _TOTAL_KEY:
            continue
        if key.endswith(_READ_SUFFIX):
            stems.append(key[:-len(_READ_SUFFIX)])
        elif key == _STATE_KEY:
            stems.append('state')
    return sorted(set(stems))


def _normalize_policy(row: dict, quant) -> dict:
    """Resolve ``quant`` into a ``{stem: QuantSpec | None}`` policy.

    ``quant`` may be ``None`` (fp16 baseline for every component), a single
    :class:`QuantSpec` / :class:`FP8Spec` (applied to every component), or a
    mapping from row field stem (e.g. ``'core'``, ``'indexer'``,
    ``'state'``, ``'kda'``) to a spec/``None``. A ``'*'`` key sets the
    default for
    stems not listed. Stems that do not exist in ``row`` raise
    ``ValueError`` (typo protection: applying a policy to a field the row
    does not carry would silently not happen).
    """
    stems = _row_component_stems(row)
    if quant is None:
        return {stem: None for stem in stems}
    if isinstance(quant, (QuantSpec, FP8Spec)):
        return {stem: quant for stem in stems}
    if not isinstance(quant, dict):
        raise TypeError(
            f'quantized_row_bytes: quant must be None, a QuantSpec, an '
            f'FP8Spec or a dict, got {type(quant)!r}')
    unknown = sorted(set(quant) - set(stems) - {'*'})
    if unknown:
        raise ValueError(
            f'quantized_row_bytes: policy names unknown component(s) '
            f'{unknown}; this row has {stems}.')
    default = quant.get('*')
    if default is not None and not isinstance(default, (QuantSpec, FP8Spec)):
        raise TypeError(
            f'quantized_row_bytes: policy["*"] must be a QuantSpec, an '
            f'FP8Spec or None, got {default!r}')
    resolved = {}
    for stem in stems:
        spec = quant.get(stem, default)
        if spec is not None and not isinstance(spec, (QuantSpec, FP8Spec)):
            raise TypeError(
                f'quantized_row_bytes: policy[{stem!r}] must be a '
                f'QuantSpec, an FP8Spec or None, got {spec!r}')
        resolved[stem] = spec
    return resolved


def quantized_row_bytes(row: dict, quant=None) -> dict:
    """Component-wise byte columns for a ``kv_budget`` element row.

    ``row`` is any element row produced by ``kv_budget`` (operator-level
    ``op_read_elements`` or layout-level ``layout_read_elements``). Every
    ``<stem>_read_elements`` field becomes ``<stem>_read_bytes`` and the
    KDA ``state_elements`` becomes ``state_bytes``.

    The per-query aggregate ``total_read_bytes`` sums the *read* components
    only: the retained state is storage, not a per-query read, matching
    ``total_read_elements`` (which counts reads; the KDA state size is
    exposed separately as ``state_elements``). For operator rows whose
    ``core_read_elements`` already aggregates branch reads, the branch
    columns (``selected`` / ``window`` / ``dense``) stay diagnostic and are
    not added to the total again. The returned dict also carries
    ``quant_bytes_per_element`` (stem → bytes/element) so a reader can see
    which policy priced which component.

    ``quant`` follows :func:`_normalize_policy`: ``None`` prices every
    component at fp16 (2.0 bytes/element); a :class:`QuantSpec` or
    :class:`FP8Spec` applies to all; a mapping prices per stem with an
    optional ``'*'`` fallback.
    """
    if not isinstance(row, dict):
        raise TypeError(f'quantized_row_bytes: row must be a dict, got '
                        f'{type(row)!r}')
    stems = _row_component_stems(row)
    if not stems:
        raise ValueError(
            'quantized_row_bytes: row carries no element fields (expected '
            f'keys like "<op>_read_elements" or "state_elements"); got '
            f'{sorted(row)}.')
    policy = _normalize_policy(row, quant)
    out: dict = {}
    bytes_per_element_map: dict = {}
    total = 0.0
    for stem in stems:
        spec = policy[stem]
        bpe = (FP16_BYTES_PER_ELEMENT if spec is None
               else bytes_per_element(spec))
        bytes_per_element_map[stem] = bpe
        if stem == 'state':
            value = row.get(_STATE_KEY)
            key = 'state_bytes'
        else:
            value = row.get(f'{stem}{_READ_SUFFIX}')
            key = f'{stem}_read_bytes'
        if value is None:
            out[key] = None
            continue
        b = float(value) * bpe
        out[key] = b
        if stem != 'state' and not (
                'core' in stems and stem in _CORE_SUBCOMPONENT_STEMS):
            # The retained state is storage, not a per-query read; it is
            # exposed as state_bytes but never folded into total. Branch
            # components are already part of ``core_read_elements``.
            total += b
    out[_TOTAL_KEY.replace('_elements', '_bytes')] = total
    out['quant_bytes_per_element'] = bytes_per_element_map
    return out
