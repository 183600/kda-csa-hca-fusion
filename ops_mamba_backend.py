"""Pluggable Mamba-2 backends: reference vs. the official ``mamba_ssm`` package.

Mirrors the KDA backend adapter (``ops_kda_backend.py``): the repository's
pure-PyTorch reference path (``ops_mamba.Mamba2Mixer``) stays the default so
reference experiments remain dependency-free, and the optional official
implementation (``mamba_ssm``'s Triton-backed ``Mamba2`` mixer) can be
selected without changing the model's surrounding structure.

Backend contract
----------------
* ``reference`` — always available, runs on CPU and CUDA. This is the
  correctness baseline.
* ``official`` — requires the optional ``mamba-ssm`` package AND CUDA
  tensors. Requesting it explicitly when either is missing raises an
  actionable error: an explicitly requested accelerated backend must never
  silently fall back to the reference path and keep reporting accelerated
  numbers (the same rule ``ops_kda_backend`` applies to ``kda_backend='fla'``).
* ``auto`` — official when CUDA is available and the package imports, else
  the reference path. Allowed to fall back (with the resolved backend
  recorded on the mixer's ``effective_backend`` attribute).

Weight transfer
---------------
The reference mixer and the official ``Mamba2`` share only part of their
parameter layout: ``A_log``, ``D``, ``out_proj`` and the dt bias are
shape-compatible (``dt_proj.bias`` maps onto the official ``dt_bias`` on
revisions that use it). The official module, however, fuses
``[z, x, B, C, dt]`` into ``in_proj`` and convolves ``xBC`` jointly, while
the reference keeps the Mamba-1-style ``x_proj`` + ``dt_proj`` head and an
x-only short conv — so a full state_dict copy is impossible for those
tensors. :func:`transfer_reference_state` copies the compatible tensors
and raises a clear ``ValueError`` listing the structurally incompatible
fused-layout parameters instead of silently producing wrong outputs.

Tested against ``mamba-ssm >= 2.2`` (the revision that ships the Mamba-2
SSD kernels). The adapter filters kwargs against the installed signature
so nearby revisions remain importable, mirroring the FLA adapter's
compatibility approach.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import torch
import torch.nn as nn

_VALID_BACKENDS = {"reference", "official", "auto"}

_MAMBA2_CLS: Any = None
_INFERENCE_PARAMS_CLS: Any = None
_IMPORT_ERROR: Exception | None = None


def validate_mamba_backend(backend: str) -> str:
    """Validate and return a Mamba-2 backend name."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"mamba_backend={backend!r} must be one of "
            f"{sorted(_VALID_BACKENDS)}.")
    return backend


def _load_official_module():
    """Import the official Mamba-2 mixer + InferenceParams lazily once.

    Failure contract: an *absent* package raises the install-hint
    ``ImportError``; a package that is installed but BROKEN (e.g. its
    Triton dependency fails at import time with ``RuntimeError``, or a
    half-extracted wheel raises ``OSError``) is wrapped into an equally
    actionable ``ImportError`` that names the original exception. Both
    failures are surfaced by an explicit ``backend='official'`` request;
    the availability probe catches them and reports "not available" so
    ``backend='auto'`` can fall back (see
    :func:`mamba2_official_available`).
    """
    global _MAMBA2_CLS, _INFERENCE_PARAMS_CLS, _IMPORT_ERROR
    if _MAMBA2_CLS is not None:
        return _MAMBA2_CLS, _INFERENCE_PARAMS_CLS
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
        try:
            # Canonical location across released mamba-ssm revisions
            # (2.0.x - 2.2.x and current main): a plain dataclass in
            # ``mamba_ssm.utils.generation``.
            from mamba_ssm.utils.generation import InferenceParams
        except ImportError:  # pragma: no cover - revision drift fallback
            # Older / relocated copies (the package root never re-exports
            # it in released revisions; keep both paths importable).
            try:
                from mamba_ssm.ops.triton.layer import (  # type: ignore
                    InferenceParams)
            except ImportError:
                from mamba_ssm.utils import InferenceParams  # type: ignore
    except ImportError as exc:
        _IMPORT_ERROR = exc
        raise ImportError(
            "Mamba-2 backend 'official' requires the optional dependency "
            "'mamba-ssm' (pip install -e '.[mamba]'). Install it, or use "
            "mamba_backend='reference' / 'auto'."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - broken installs raise anything
        _IMPORT_ERROR = exc
        raise ImportError(
            "Mamba-2 backend 'official' found 'mamba-ssm' installed but "
            "its import chain failed "
            f"({type(exc).__name__}: {exc}); the installation looks "
            "broken (Triton/torch version mismatch or an incomplete "
            "wheel). Reinstall mamba-ssm, or use mamba_backend="
            "'reference' / 'auto'."
        ) from exc
    _MAMBA2_CLS = _Mamba2
    _INFERENCE_PARAMS_CLS = InferenceParams
    return _MAMBA2_CLS, _INFERENCE_PARAMS_CLS


def mamba2_official_available() -> bool:
    """Return whether the official Mamba-2 module can be imported.

    Availability PROBE: never raises (mirrors
    ``ops_kda_backend.fla_available`` / ``ops_rwkv_backend.
    rwkv7_fla_available``). A broken install reads as unavailable so
    ``resolve_mamba_backend('auto')`` stays honest and can fall back;
    an explicit ``backend='official'`` request still surfaces the
    actionable ``ImportError`` from :func:`_load_official_module`.
    """
    try:
        _load_official_module()
    except Exception:  # noqa: BLE001 - a probe must never raise
        return False
    return True


def resolve_mamba_backend(requested: str) -> str:
    """Resolve the backend that WOULD be used, for provenance recording.

    Returns ``'official'`` only when the request is compatible with the
    environment (CUDA present + package importable); otherwise
    ``'reference'``. Unlike :func:`build_official_mamba2_mixer` this never
    raises, so experiment runners can record intent + resolution separately.
    """
    validate_mamba_backend(requested)
    if requested == 'reference':
        return 'reference'
    if torch.cuda.is_available() and mamba2_official_available():
        return 'official'
    return 'reference'


def _build_official_mamba2(cfg) -> nn.Module:
    """Construct the official ``mamba_ssm`` Mamba2 mixer from a config.

    Keyword arguments are filtered against the installed constructor's
    signature (mirroring ``ops_kda_backend._supported_kwargs``), so only
    arguments the installed ``mamba_ssm`` revision accepts are passed.
    """
    Mamba2, _ = _load_official_module()
    from ops_mamba import resolve_mamba_dt_rank
    kwargs: dict[str, Any] = {
        'd_model': cfg.d_model,
        'd_state': int(getattr(cfg, 'mamba_d_state', 64)),
        'd_conv': int(getattr(cfg, 'mamba_d_conv', 4)),
        'expand': int(getattr(cfg, 'mamba_expand', 2)),
        'headdim': int(getattr(cfg, 'mamba_headdim', 64)),
        'ngroups': int(getattr(cfg, 'mamba_ngroups', 1)),
        'dt_rank': resolve_mamba_dt_rank(
            getattr(cfg, 'mamba_dt_rank', 'auto'), cfg.d_model),
        # The wrapper manages inference state itself; layer_idx=0 keeps the
        # official module's cache indexing deterministic.
        'layer_idx': 0,
    }
    try:
        signature = inspect.signature(Mamba2.__init__)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()):
        dropped = [k for k in kwargs if k not in signature.parameters]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            warnings.warn(
                f"The installed mamba_ssm Mamba2 does not accept {dropped}; "
                f"those settings fall back to the package defaults, so the "
                f"official backend's geometry may differ from the reference "
                f"config. Pin mamba-ssm >= 2.2 for an exact match.",
                RuntimeWarning, stacklevel=3)
    return Mamba2(**kwargs)


def transfer_reference_state(official_module: nn.Module,
                             reference_module: nn.Module) -> None:
    """Copy reference-mixer weights into an official ``Mamba2`` module.

    Handles both ``dt`` parameterizations found across ``mamba_ssm``
    revisions:

      * ``dt_proj.weight`` + ``dt_proj.bias`` (matches the reference), and
      * ``dt_proj.weight`` + separate ``dt_bias`` (older layout): the
        reference's ``dt_proj.bias`` maps onto ``dt_bias``.

    Raises ``ValueError`` listing any official parameter that could not be
    sourced from the reference state dict. The released ``mamba_ssm``
    Mamba2 modules fuse ``[z, x, B, C, dt]`` into ``in_proj`` and convolve
    ``xBC`` jointly, so their ``in_proj`` / ``conv1d`` / ``norm`` tensors
    have no counterpart in the Mamba-1-style reference head and are NOT
    transferable; only ``A_log`` / ``D`` / ``out_proj`` / the dt bias are.
    """
    ref_sd = reference_module.state_dict()
    target_sd = official_module.state_dict()
    transferred, missing = [], []
    with torch.no_grad():
        for name, t in target_sd.items():
            if name in ref_sd and ref_sd[name].shape == t.shape:
                t.copy_(ref_sd[name].to(t.dtype))
                transferred.append(name)
            elif name == 'dt_bias' and 'dt_proj.bias' in ref_sd \
                    and ref_sd['dt_proj.bias'].shape == t.shape:
                t.copy_(ref_sd['dt_proj.bias'].to(t.dtype))
                transferred.append(name)
            else:
                missing.append(name)
    if missing:
        raise ValueError(
            f"transfer_reference_state: parameter(s) {missing} cannot be "
            f"sourced from the reference mixer: the installed mamba_ssm "
            f"Mamba2 fuses [z, x, B, C, dt] into in_proj and convolves xBC "
            f"jointly, so only A_log / D / out_proj / the dt bias are "
            f"shape-compatible. Use mamba_backend='reference' for "
            f"weight-compatible runs.")
    return None


class OfficialMamba2Mixer(nn.Module):
    """Mamba-2 mixer backed by the official ``mamba_ssm`` implementation.

    Implements the SAME memory-layer contract as ``ops_mamba.Mamba2Mixer``
    (``forward`` / ``forward_functional`` / ``prefill`` / ``step`` /
    ``reset_conv_state``) by driving the official module through its
    ``InferenceParams`` state cache:

      * ``forward_functional(x, state, conv_state)`` builds a per-call
        ``InferenceParams``, seeds ``key_value_memory_dict[0]`` with the
        incoming (cloned) states, runs the official module with
        ``seqlen_offset=0`` (T > 1: the parallel prefill path, which writes
        the final states back into the cache) or ``seqlen_offset>0``
        (T == 1: the recurrent step path, which updates the cache in
        place), and returns the new states.
      * SSM state ``[B, nheads, headdim, d_state]`` (fp32) matches the
        reference contract. The conv state is the OFFICIAL layout
        ``[B, d_inner + 2 * ngroups * d_state, d_conv]`` (input dtype):
        the official short conv runs over ``xBC`` jointly and keeps
        ``d_conv`` time steps, so it is NOT interchangeable with the
        reference's x-only ``[B, d_inner, d_conv - 1]`` buffer.
      * The official parallel path starts from a FRESH prefix and ignores
        any incoming state, so chunked prefill continuation is not
        supported: ``forward_functional`` rejects ``T > 1`` with a
        non-``None`` state up front (feed the whole prefix in one call,
        then decode with ``step``). The reference backend has no such
        restriction.

    The official kernels are CUDA-only. Feeding CPU tensors raises a clear
    ``RuntimeError`` (use the reference backend on CPU).
    """

    def __init__(self, cfg):
        super().__init__()
        from ops_mamba import validate_mamba2_geometry
        geo = validate_mamba2_geometry(
            cfg.d_model,
            int(getattr(cfg, 'mamba_d_state', 64)),
            int(getattr(cfg, 'mamba_d_conv', 4)),
            int(getattr(cfg, 'mamba_expand', 2)),
            int(getattr(cfg, 'mamba_headdim', 64)),
            int(getattr(cfg, 'mamba_ngroups', 1)),
            getattr(cfg, 'mamba_dt_rank', 'auto'))
        self.d_model = cfg.d_model
        self.d_state = int(getattr(cfg, 'mamba_d_state', 64))
        self.d_conv = int(getattr(cfg, 'mamba_d_conv', 4))
        self.ngroups = int(getattr(cfg, 'mamba_ngroups', 1))
        self.d_inner = geo['d_inner']
        self.nheads = geo['nheads']
        # ``validate_mamba2_geometry`` returns the derived sizes, not the
        # configured headdim; read it from the config exactly like the
        # reference mixer does (the geometry check already proved it
        # divides d_inner).
        self.headdim = int(getattr(cfg, 'mamba_headdim', 64))
        # Official conv channels: xBC = x + the B/C projections.
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.backend = 'official'
        self.effective_backend = 'official'
        # Provenance counterpart of the reference mixer's ``scan_mode``:
        # the official kernels always run their own internal chunked SSD,
        # and the reference-only ``mamba_chunk_size`` knob does not apply
        # (HybridConfig rejects that combination for an explicit request).
        self.scan_mode = 'official-internal'
        self.mamba = _build_official_mamba2(cfg)
        # Runtime conv-state buffer for the stateful forward() wrapper.
        self.register_buffer('_conv_lookback', None, persistent=False)

    # -- helpers --------------------------------------------------------
    def _require_cuda(self, x: torch.Tensor) -> None:
        if not x.is_cuda:
            raise RuntimeError(
                "Mamba-2 backend 'official' requires CUDA tensors (the "
                "mamba_ssm Triton kernels have no CPU path). Use "
                "mamba_backend='reference' for CPU experiments.")

    def _make_inference_params(self, batch: int, seqlen: int,
                               device: torch.device):
        _, InferenceParams = _load_official_module()
        ip = InferenceParams(max_seqlen=max(seqlen, 1),
                             max_batch_size=batch)
        ip.seqlen_offset = 0
        return ip

    # -- state contract -------------------------------------------------
    def reset_conv_state(self) -> None:
        self._conv_lookback = None

    def state_nbytes(self, batch: int = 1) -> dict:
        # Official conv state: xBC channels over the full d_conv window.
        ssm_elems = batch * self.nheads * self.headdim * self.d_state
        conv_elems = batch * self.conv_dim * self.d_conv
        return {
            'ssm_state_bytes': ssm_elems * 4,
            'conv_state_bytes': conv_elems * 4,
            'total_bytes': (ssm_elems + conv_elems) * 4,
            'ssm_state_elements': ssm_elems,
            'conv_state_elements': conv_elems,
        }

    # -- forward paths ----------------------------------------------------
    def forward(self, x: torch.Tensor,
                state: torch.Tensor | None = None):
        o, new_state, new_conv = self.forward_functional(
            x, state, self._conv_lookback)
        self._conv_lookback = new_conv
        return o, new_state

    def forward_functional(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        conv_state: torch.Tensor | None,
        *,
        detach_lookback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._require_cuda(x)
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"OfficialMamba2Mixer: input d_model={d} does not match the "
                f"configured d_model={self.d_model}.")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if T == 0:
            if state is not None and state.shape[0] == B:
                new_state = state.to(device=x.device, dtype=compute_dtype)
            else:
                new_state = torch.zeros(
                    B, self.nheads, self.headdim, self.d_state,
                    device=x.device, dtype=compute_dtype)
            new_conv = conv_state
            if new_conv is not None:
                if new_conv.shape[0] != B:
                    # A different batch cannot be resumed; the reference
                    # Mixer / KDAHybridLayer contract nulls the lookback
                    # (the T > 1 path below also starts fresh on a batch
                    # change) instead of returning a state of the wrong
                    # batch size.
                    new_conv = None
                else:
                    new_conv = new_conv.to(device=x.device, dtype=x.dtype)
                    if detach_lookback:
                        new_conv = new_conv.detach()
                    new_conv = new_conv.clone()
            return x.new_zeros(B, 0, d), new_state, new_conv

        if T > 1 and (state is not None or conv_state is not None):
            raise ValueError(
                "OfficialMamba2Mixer: the official mamba_ssm parallel "
                "prefill path always starts from a fresh prefix and "
                "ignores incoming conv/SSM states, so a T > 1 call cannot "
                "resume from saved state. Feed the whole prefix in one "
                "prefill() call and continue with step(), or use "
                "mamba_backend='reference' for chunked prefill.")
        ip = self._make_inference_params(B, T, x.device)
        # Seed the official state cache. The official module reads and
        # updates these tensors IN PLACE through its InferenceParams
        # protocol (``key_value_memory_dict[layer_idx] = (conv, ssm)``),
        # so seed with clones of the caller's tensors to avoid mutating
        # state the caller still owns.
        if conv_state is not None and tuple(conv_state.shape) != (
                B, self.conv_dim, self.d_conv):
            if conv_state.shape[0] != B:
                # A different batch cannot be resumed; start fresh (the
                # official module would allocate the same zeros itself).
                conv_state = None
            else:
                raise ValueError(
                    f"OfficialMamba2Mixer: conv_state shape "
                    f"{tuple(conv_state.shape)} is not the official conv "
                    f"layout {(B, self.conv_dim, self.d_conv)} "
                    f"(xBC channels over d_conv steps). A reference-backend "
                    f"conv state [B, {self.d_inner}, {self.d_conv - 1}] "
                    f"cannot be resumed on the official backend — the two "
                    f"conv layouts are not interchangeable.")
        conv_seed = (conv_state.clone()
                     if conv_state is not None else
                     torch.zeros(B, self.conv_dim, self.d_conv,
                                 device=x.device, dtype=x.dtype))
        if state is not None and state.shape[0] == B:
            # ``to(copy=True)`` = cast-if-needed plus unconditional copy,
            # the same contents as ``clone().to()`` with one temporary less.
            ssm_seed = state.to(compute_dtype, copy=True)
        else:
            ssm_seed = torch.zeros(
                B, self.nheads, self.headdim, self.d_state,
                device=x.device, dtype=compute_dtype)
        ip.key_value_memory_dict[0] = (conv_seed, ssm_seed)
        # seqlen_offset > 0 selects the official recurrent STEP path;
        # seqlen_offset == 0 runs the parallel path and copies the final
        # states back into the cache.
        ip.seqlen_offset = 1 if T == 1 else 0
        out = self.mamba(x, inference_params=ip)
        new_conv, new_state = ip.key_value_memory_dict[0]
        # Same single-shot cast+copy as ``to().clone()`` above.
        new_state = new_state.to(compute_dtype, copy=True)
        new_conv = new_conv.to(x.dtype)
        if detach_lookback:
            new_conv = new_conv.detach()
        new_conv = new_conv.clone()
        return out, new_state, new_conv

    # -- unified inference protocol ---------------------------------------
    def prefill(self, x: torch.Tensor):
        out, ssm_state, conv_state = self.forward_functional(
            x, None, None, detach_lookback=True)
        return out, {
            'ssm_state': ssm_state.detach() if ssm_state is not None else None,
            'conv_state': conv_state,
        }

    def step(self, x_t: torch.Tensor, state: dict):
        if state is not None and not isinstance(state, dict):
            raise TypeError(
                f"OfficialMamba2Mixer.step: state must be the dict returned "
                f"by prefill/step (or None), got {type(state).__name__}.")
        if x_t.dim() != 3 or x_t.shape[1] != 1:
            raise ValueError(
                f"OfficialMamba2Mixer.step: x_t must be [B, 1, d_model], "
                f"got shape {tuple(x_t.shape)}.")
        state = state or {}
        out, ssm_state, conv_state = self.forward_functional(
            x_t, state.get('ssm_state'), state.get('conv_state'),
            detach_lookback=True)
        return out, {
            'ssm_state': ssm_state.detach() if ssm_state is not None else None,
            'conv_state': conv_state,
        }


def build_official_mamba2_mixer(cfg):
    """Build the official-backend Mamba-2 mixer, erroring when unavailable.

    Raises:
        ImportError: ``mamba-ssm`` is not installed.
        RuntimeError: the package imports but CUDA is unavailable (the
            official kernels are CUDA-only; an explicitly requested
            accelerated backend must fail loudly, never fall back).
    """
    if not mamba2_official_available():
        # Re-raise with the canonical message (includes install hint).
        _load_official_module()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Mamba-2 backend 'official' requires CUDA, but "
            "torch.cuda.is_available() is False. Use "
            "mamba_backend='reference' for CPU experiments or run on a "
            "CUDA device.")
    return OfficialMamba2Mixer(cfg)


__all__ = [
    "OfficialMamba2Mixer",
    "build_official_mamba2_mixer",
    "mamba2_official_available",
    "resolve_mamba_backend",
    "transfer_reference_state",
    "validate_mamba_backend",
]
