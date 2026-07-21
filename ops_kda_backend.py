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
import math
import warnings
from typing import Any

import torch


_VALID_BACKENDS = {"reference", "fla", "auto"}
_fla_import_warning_emitted = False
_FLA_OPS: tuple[Any, Any] | None = None


def validate_kda_backend(backend: str) -> str:
    """Validate and return a KDA backend name."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"kda_backend={backend!r} must be one of "
            f"{sorted(_VALID_BACKENDS)}.")
    return backend


def _load_fla_ops():
    """Import FLA lazily once so the decode hot path has no import lookup."""
    global _FLA_OPS
    if _FLA_OPS is not None:
        return _FLA_OPS
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ImportError as exc:
        raise ImportError(
            "KDA backend 'fla' requires the optional dependency "
            "'flash-linear-attention'. Install it with "
            "pip install -e '.[fla]' or choose kda_backend='reference'."
        ) from exc
    _FLA_OPS = (chunk_kda, fused_recurrent_kda)
    return _FLA_OPS


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
    # g_clamp_min is already applied once in kda_forward before dispatch;
    # do NOT apply it a second time here to avoid double-clamping.
    # We pass the original g_clamp_min to the FLA kernel.  Because g has
    # already been clamped to [g_clamp_min, inf) in kda_forward, the FLA
    # kernel's internal clamp with the same bound is a strict no-op, so
    # there is no double-clamping side effect.  Passing -inf as a sentinel
    # is unsafe because the FLA Triton kernel may compute exp(g_clamp_min),
    # yielding 0.0 or NaN and corrupting the delta-rule state update.
    # Preserve the original ``scale`` value (including torch.Tensor) so that
    # gradients can flow back and device placement is retained.  Only convert
    # plain Python numbers when needed; do NOT detach/item() a tensor scale.
    if isinstance(scale, torch.Tensor):
        scale_value = scale
    else:
        scale_value = scale

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
        # Pass the original g_clamp_min so that the FLA kernel uses the
        # exact same clamp bound as the caller.  Because g has already been
        # clamped to [g_clamp_min, inf) in kda_forward, the FLA kernel's
        # internal clamp with the same bound is a strict no-op, matching
        # the reference path without risking exp(-inf) -> 0/NaN corruption.
        "g_clamp_min": g_clamp_min,
    }
    result = fn(**_supported_kwargs(fn, kwargs))
    # Match the repository reference contract: outputs retain the caller's
    # value dtype, while recurrent state stays in compute precision (fp32 for
    # fp16/bf16 inputs, fp64 for fp64 inputs). Without this normalization,
    # FLA may return a low-precision state and long decode sessions accumulate
    # avoidable quantization error compared with the reference path.
    if output_final_state:
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(
                "The installed FLA KDA operator returned an unexpected result; "
                "expected (output, final_state).")
        output, final_state = result[0], result[1]
    else:
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        final_state = None
    output = output.to(dtype=v.dtype)
    if output_final_state and final_state is not None:
        state_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
        final_state = final_state.to(dtype=state_dtype)
    return output, final_state


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
    combinations (including assertion-based checks in older FLA releases)
    fall back to the reference implementation. ``g_clamp_min``
    is applied before both backends so their gate contract matches.
    """
    global _fla_import_warning_emitted
    backend = validate_kda_backend(backend)
    if not isinstance(g_clamp_min, (int, float)) or isinstance(g_clamp_min, bool):
        raise TypeError(f"g_clamp_min must be a real number, got {g_clamp_min!r}")
    if math.isnan(float(g_clamp_min)):
        raise ValueError("g_clamp_min must not be NaN")

    # Apply g_clamp_min exactly once here, before any backend dispatch.
    # The downstream functions (_call_fla, naive_recurrent_kda,
    # naive_chunk_kda) must NOT apply their own clamp to avoid
    # double-clamping.  We pass the original g_clamp_min to the downstream
    # functions; because g has already been clamped to [g_clamp_min, inf)
    # here, the downstream internal clamp with the same bound is a strict
    # no-op.  Do NOT pass -inf as a sentinel: the FLA Triton kernel may
    # compute exp(g_clamp_min), yielding 0.0 or NaN and corrupting the
    # delta-rule state update.
    if g_clamp_min > -float('inf'):
        g = g.clamp(min=float(g_clamp_min))
    downstream_clamp = g_clamp_min

    use_fla = backend == "fla"
    if backend == "auto":
        use_fla = bool(q.is_cuda and fla_available())

    if use_fla:
        if not q.is_cuda:
            raise RuntimeError(
                "KDA backend 'fla' requires CUDA tensors; use "
                "kda_backend='reference' for CPU experiments.")
        try:
            return _call_fla(
                q, k, v, g, beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                use_chunk=use_chunk,
                g_clamp_min=downstream_clamp,
            )
        except (ImportError, ValueError, NotImplementedError, AssertionError) as exc:
            if backend == "fla":
                raise
            if not _fla_import_warning_emitted:
                warnings.warn(
                    f"kda_backend='auto' could not use FLA ({type(exc).__name__}: "
                    f"{exc}); falling back to the reference implementation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _fla_import_warning_emitted = True
            # Fall back to the reference implementation for this call only.
            # The global warning flag suppresses duplicate log spam but must
            # NOT permanently disable FLA for subsequent calls, otherwise a
            # single transient edge case would silently degrade every later
            # KDA computation in the process to the slow Python reference.
            use_fla = False

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
            g_clamp_min=downstream_clamp,
        )
    return naive_recurrent_kda(
        q, k, v, g, beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        g_clamp_min=downstream_clamp,
    )


__all__ = ["fla_available", "kda_forward", "validate_kda_backend"]
