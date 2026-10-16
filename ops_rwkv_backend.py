"""RWKV-7 backend selection, lazy import, and the FLA adapter.

Mirrors ``ops_kda_backend`` / ``ops_mamba_backend``:

  * :func:`validate_rwkv_backend` — eager validation (a typo fails at
    config creation, before any parameters are allocated); the optional
    ``fla`` import stays lazy until a forward pass actually needs it.
  * :func:`rwkv7_fla_available` / :func:`resolve_rwkv_backend` —
    availability probe + never-raising provenance helper.
  * :func:`rwkv7_forward` — the dispatch point between the reference
    scan (``ops_rwkv.rwkv7_reference_scan``) and the accelerated FLA
    kernels.

**Decay-contract conversion (the double-transform hazard).** The mixer
always emits the *log-decay* (``log_w = -0.6065306597126334 *
sigmoid(w_lora(xw))``, so ``decay = exp(log_w)``). The two upstream
conventions differ:

  * ``fla.ops.rwkv7.chunk_rwkv7`` / ``fused_mul_recurrent_rwkv7`` take
    ``log_w`` (``gk``) directly -> **pass-through, no transform**.
  * the BlinkDL x070 ``RWKV7_OP`` takes a raw pre-activation and
    applies ``exp(-exp(w))`` INSIDE the op -> a raw-w adapter must feed
    it ``w_raw = log(-log_w)``-style pre-activations, never the
    log-decay.

This module's FLA adapter consumes log-decay and passes it through
untouched; any future raw-w backend adapter MUST document its
conversion in one place (here). ``rwkv7_reference_scan`` also consumes
log-decay and applies ``exp()`` exactly once.

**State layout.** The fla kernels use ``[B, H, K, V]`` (rows = key dim,
columns = value dim) — the same layout as the reference scan in
``ops_rwkv.py`` — so the adapter transposes nothing. (The BlinkDL demo
uses the transposed ``[B, H, V, K]`` layout; if a future adapter binds
to THAT convention it must transpose both the initial and the final
state, and say so here.)

**Environment co-existence caveat.** The repository's KDA route also
uses a flash-linear-attention backend (``ops_kda_backend``) that
imports the ``fla`` namespace. RWKV-FLA requires ``fla.ops.rwkv7``.
The two MAY be the same install (recent ``flash-linear-attention``
revisions ship both ``fla.ops.delta_rule``-family KDA ops and
``fla.ops.rwkv7``), but if the pinned KDA version predates
``fla.ops.rwkv7`` the two requirements conflict: isolate the RWKV
experiments in their own environment with both features verified —
never overwrite-install over the KDA pin (the experiment-design
requirement). :func:`rwkv7_fla_available` therefore checks for the
``fla.ops.rwkv7`` subpackage specifically, not merely ``import fla``.
"""

from __future__ import annotations

import warnings

import torch

from ops_rwkv import rwkv7_reference_scan

_VALID_BACKENDS = {"reference", "fla", "auto"}

_FLA_CHUNK_FN = None
_FLA_RECURRENT_FN = None
_FLA_IMPORT_ERROR: Exception | None = None
_FLA_PROBED = False


def validate_rwkv_backend(backend: str) -> str:
    """Validate and return an RWKV-7 backend name."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"rwkv_backend={backend!r} must be one of "
            f"{sorted(_VALID_BACKENDS)}.")
    return backend


def _load_fla_rwkv7():
    """Import the fla RWKV-7 kernels lazily once.

    Checks for ``fla.ops.rwkv7`` specifically: an installed
    flash-linear-attention that predates the RWKV-7 ops would satisfy
    ``import fla`` but cannot serve this backend, and silently mapping
    it to the reference path would misreport provenance.
    """
    global _FLA_CHUNK_FN, _FLA_RECURRENT_FN, _FLA_IMPORT_ERROR, _FLA_PROBED
    if _FLA_PROBED:
        if _FLA_IMPORT_ERROR is not None:
            raise ImportError(_FLA_IMPORT_ERROR)
        return _FLA_CHUNK_FN, _FLA_RECURRENT_FN
    _FLA_PROBED = True
    try:
        from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
    except ImportError as exc:
        _FLA_IMPORT_ERROR = ImportError(
            "The RWKV-7 'fla' backend requires flash-linear-attention "
            "with fla.ops.rwkv7 (pip install flash-linear-attention, or "
            "the RWKV-FLA fork). Note the KDA route may pin an older "
            "'fla' without rwkv7 ops — if so, isolate the RWKV "
            "environment instead of overwriting the KDA pin. "
            f"(import error: {exc})"
        )
        raise _FLA_IMPORT_ERROR from exc
    _FLA_CHUNK_FN = chunk_rwkv7
    _FLA_RECURRENT_FN = fused_mul_recurrent_rwkv7
    return _FLA_CHUNK_FN, _FLA_RECURRENT_FN


def rwkv7_fla_available() -> bool:
    """Return whether the fla RWKV-7 kernels can be imported."""
    try:
        _load_fla_rwkv7()
    except ImportError:
        return False
    return True


def resolve_rwkv_backend(requested: str) -> str:
    """Resolve the backend that WOULD be used, for provenance recording.

    Returns ``'fla'`` only when the request is compatible with the
    environment (CUDA present + kernels importable); otherwise
    ``'reference'``. Never raises, so experiment runners can record
    intent + resolution separately (mirrors
    ``ops_mamba_backend.resolve_mamba_backend``).
    """
    validate_rwkv_backend(requested)
    if requested == 'reference':
        return 'reference'
    if torch.cuda.is_available() and rwkv7_fla_available():
        return 'fla'
    return 'reference'


def _fla_rwkv7_scan(
    r: torch.Tensor,
    log_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Accelerated recurrence via ``fla.ops.rwkv7``.

    Convention mapping (documented ONCE, here — see the module
    docstring):

    * the fla kernels' ``gk`` / ``log_w`` argument IS the log-decay this
      module receives -> **passed through unchanged** (the kernel
      applies ``exp()`` internally exactly once);
    * the fla kernels' low-rank update operands are ``a_op = -kk`` and
      ``b_op = kk * a`` while this module's contract passes ``kk`` and
      ``a`` separately -> the adapter forms ``a_op`` / ``b_op``
      itself, so a caller can never double-negate;
    * the fla state layout ``[B, H, K, V]`` matches the reference scan
      -> no transposes;
    * ``chunk_rwkv7`` (the differentiable training path) is used for
      ``T > 1``; ``fused_mul_recurrent_rwkv7`` (faster single-step) for
      ``T == 1`` — the same split the fla model layer itself uses.

    The fla kernels return the final state in fp32 and the output in
    the input dtype, matching the reference scan's contract.
    """
    chunk_fn, recurrent_fn = _load_fla_rwkv7()
    if r.shape[1] == 1:
        # Decode step: the fused recurrent kernel is faster for T == 1
        # (no chunk-parallel overhead). Same math, same state contract.
        o, state = recurrent_fn(
            r=r, w=log_w, k=k, v=v, kk=kk, a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return o, state
    o, state = chunk_fn(
        r=r, log_w=log_w, k=k, v=v,
        a=-kk, b=kk * a,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    return o, state


def rwkv7_forward(
    r: torch.Tensor,
    log_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    backend: str = "reference",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch the RWKV-7 recurrence (the KDA ``kda_forward`` analogue).

    Args and returns match :func:`ops_rwkv.rwkv7_reference_scan`
    exactly; ``backend`` selects the implementation. ``'fla'`` errors
    loudly when unavailable (an explicitly requested accelerated
    backend must never silently fall back — mirrors
    ``ops_mamba_backend``); ``'auto'`` resolves to fla when possible
    and falls back with a warning; ``'reference'`` always uses the
    in-module scan.
    """
    backend = validate_rwkv_backend(backend)
    if backend == 'reference':
        return rwkv7_reference_scan(
            r, log_w, k, v, kk, a,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state)
    if backend == 'fla':
        return _fla_rwkv7_scan(
            r, log_w, k, v, kk, a,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state)
    # backend == 'auto'
    if torch.cuda.is_available() and rwkv7_fla_available():
        try:
            return _fla_rwkv7_scan(
                r, log_w, k, v, kk, a,
                scale=scale, initial_state=initial_state,
                output_final_state=output_final_state)
        except (ImportError, RuntimeError):
            # Importable but broken at runtime (e.g. a triton mismatch):
            # fall back with a warning so the resolution is visible.
            warnings.warn(
                "rwkv7_forward: backend 'auto' fell back to the reference "
                "scan (the fla kernels failed at runtime). The resolved "
                "backend must still be recorded via "
                "resolve_rwkv_backend / effective_backend.",
                stacklevel=2)
    return rwkv7_reference_scan(
        r, log_w, k, v, kk, a,
        scale=scale, initial_state=initial_state,
        output_final_state=output_final_state)
