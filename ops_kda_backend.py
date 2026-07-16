"""Pluggable KDA backends for the hybrid research model.

The repository keeps its PyTorch implementation as the default reference path.
When the optional ``flash-linear-attention`` package is installed, callers can
select the FLA Triton implementation without changing the model's projection
or gate parameterization.

Important contract:
    ``q`` and ``k`` are already L2-normalized, ``g`` is already a log-space
    decay gate, and ``beta`` is already post-sigmoid.  The FLA call therefore
    disables its optional q/k normalization, gate activation, and beta sigmoid
    paths.  This avoids silently applying any activation twice.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import torch


_VALID_BACKENDS = {"reference", "fla", "auto"}
_fla_import_warning_emitted = False


def validate_kda_backend(backend: str) -> str:
    """Validate and return a KDA backend name."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"kda_backend={backend!r} must be one of "
            f"{sorted(_VALID_BACKENDS)}.")
    return backend


def _load_fla_ops():
    """Import FLA lazily so it remains an optional dependency."""
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ImportError as exc:
        raise ImportError(
            "KDA backend 'fla' requires the optional dependency "
            "'flash-linear-attention'. Install it with "
            "pip install -e '.[fla]' or choose kda_backend='reference'."
        ) from exc
    return chunk_kda, fused_recurrent_kda


def fla_available() -> bool:
    """Return whether FLA can be imported without changing global state."""
    try:
        _load_fla_ops()
    except ImportError:
        return False
    return True


def _supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep the adapter compatible with nearby FLA API revisions.

    FLA has renamed a few optional KDA keyword arguments over time.  The
    required tensors are passed by name, while optional arguments are filtered
    against the installed function signature.  This keeps the reference repo
    importable with an older/newer FLA release without vendoring FLA itself.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Most Python wrappers expose a signature.  If a compiled wrapper does
        # not, pass the conservative core API only.
        core = {"q", "k", "v", "g", "beta", "scale",
                "initial_state", "output_final_state"}
        return {k: v for k, v in kwargs.items() if k in core}

    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD
           for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _call_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    use_chunk: bool,
    g_clamp_min: float,
):
    chunk_kda, fused_recurrent_kda = _load_fla_ops()
    fn = chunk_kda if use_chunk else fused_recurrent_kda
    if g_clamp_min > -float('inf'):
        # Match the repository reference contract before handing the already
        # activated log-space gate to FLA. Without this, FLA and reference
        # diverge for pathological gates below -10.
        g = g.clamp(min=float(g_clamp_min))

    if isinstance(scale, torch.Tensor):
        scale_value = float(scale.detach().item())
    elif scale is None:
        scale_value = None
    else:
        scale_value = float(scale)

    kwargs: dict[str, Any] = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": scale_value,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        # The caller has already performed these transformations.
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": False,
        "chunk_size": chunk_size,
    }
    result = fn(**_supported_kwargs(fn, kwargs))
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            "The installed FLA KDA operator returned an unexpected result; "
            "expected (output, final_state).")
    return result[0], result[1]


def kda_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    use_chunk: bool = False,
    g_clamp_min: float = -10.0,
    backend: str = "reference",
):
    """Run KDA through the selected reference or optional FLA backend.

    ``backend='reference'`` is bit-for-bit compatible with the existing
    repository dispatch. ``backend='fla'`` requires FLA and raises an
    actionable ImportError when it is not installed. ``backend='auto'`` uses
    FLA only for CUDA tensors when it is importable; unsupported FLA argument
    combinations fall back to the reference implementation. ``g_clamp_min``
    is applied before both backends so their gate contract matches.
    """
    global _fla_import_warning_emitted
    backend = validate_kda_backend(backend)

    use_fla = backend == "fla"
    if backend == "auto":
        use_fla = bool(q.is_cuda and fla_available())

    if use_fla and not q.is_cuda:
        raise RuntimeError(
            "KDA backend 'fla' requires CUDA tensors; use "
            "kda_backend='reference' for CPU experiments.")

    if use_fla:
        if backend == "auto" and not q.is_cuda:
            use_fla = False
        else:
            try:
                return _call_fla(
                    q, k, v, g, beta,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    chunk_size=chunk_size,
                    use_chunk=use_chunk,
                    g_clamp_min=g_clamp_min,
                )
            except (ImportError, ValueError, NotImplementedError) as exc:
                if backend == "fla":
                    raise
                warnings.warn(
                    f"kda_backend='auto' could not use FLA ({type(exc).__name__}: "
                    f"{exc}); falling back to the reference implementation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _fla_import_warning_emitted = True
                use_fla = False

    if backend == "auto" and not use_fla and q.is_cuda and not _fla_import_warning_emitted:
        warnings.warn(
            "kda_backend='auto' could not use FLA; falling back to the "
            "repository reference implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        _fla_import_warning_emitted = True

    # Lazy import avoids an import cycle and preserves the original public
    # ops_kda module as the single source of truth for correctness tests.
    from ops_kda import naive_chunk_kda, naive_recurrent_kda

    if use_chunk:
        return naive_chunk_kda(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            g_clamp_min=g_clamp_min,
        )
    return naive_recurrent_kda(
        q, k, v, g, beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        g_clamp_min=g_clamp_min,
    )


__all__ = ["fla_available", "kda_forward", "validate_kda_backend"]
