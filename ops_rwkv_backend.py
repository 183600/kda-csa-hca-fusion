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

**Keyword spelling across FLA revisions.** The chunked kernel names the
decay argument differently depending on the installed distribution:
``RWKV-Vibe/rwkv-fla`` exposes ``log_w=`` (pass-through) next to a raw
``w=`` that it converts internally, while mainline
``flash-linear-attention`` names the log-decay ``w=`` and has no
``log_w`` at all. The adapter therefore reads the installed signature
(:func:`_chunk_decay_param`) and forwards the log-decay under whichever
keyword that revision expects — forwarding ``w=`` to the fork would
convert the decay a second time, and forwarding ``log_w=`` to mainline
leaves its required ``w`` unbound.

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

import inspect
import sys
import warnings
from typing import Any

import torch

from ops_rwkv import rwkv7_reference_scan

_VALID_BACKENDS = {"reference", "fla", "auto"}

_FLA_CHUNK_FN = None
_FLA_RECURRENT_FN = None
_FLA_IMPORT_ERROR: Exception | None = None
_FLA_PROBED = False
_FLA_MODULE = None

# ``inspect.signature`` is introspection-heavy and the accepted keyword set
# of an installed FLA operator cannot change while the function object is
# alive, so the decay-keyword classification is computed once per function
# (a decode loop dispatches through here on every token).
_DECAY_PARAM_CACHE: dict[Any, str] = {}


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

    Outcome caching contract: the first import ATTEMPT is the only one
    (``_FLA_PROBED``). Whether it succeeds OR fails, the outcome is
    cached so repeated probes / dispatches never re-import a broken
    package. A failed attempt ALWAYS stores an actionable
    ``ImportError`` in ``_FLA_IMPORT_ERROR`` before re-raising —
    including non-``ImportError`` failures such as a Triton version
    mismatch raising ``RuntimeError`` at import time — so a broken
    install is a definite, repeatable "not available".
    """
    global _FLA_CHUNK_FN, _FLA_RECURRENT_FN, _FLA_IMPORT_ERROR, _FLA_PROBED, _FLA_MODULE
    # A cached outcome is only valid for the SAME fla.ops.rwkv7 module
    # object. Tests (and environments) inject a different module under that
    # name, which must trigger a fresh probe instead of reusing a stale
    # cached answer — otherwise a broken install could be cached as
    # "available" (or vice versa) across injected modules.
    if _FLA_PROBED:
        current_module = sys.modules.get('fla.ops.rwkv7')
        if current_module is not _FLA_MODULE:
            _FLA_PROBED = False
            _FLA_CHUNK_FN = None
            _FLA_RECURRENT_FN = None
            _FLA_IMPORT_ERROR = None
            _FLA_MODULE = None
        else:
            if _FLA_IMPORT_ERROR is not None:
                raise ImportError(_FLA_IMPORT_ERROR)
            return _FLA_CHUNK_FN, _FLA_RECURRENT_FN
    try:
        from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
    except Exception as exc:  # noqa: BLE001 - broken installs raise anything
        _FLA_IMPORT_ERROR = ImportError(
            "The RWKV-7 'fla' backend requires flash-linear-attention "
            "with fla.ops.rwkv7 (pip install flash-linear-attention, or "
            "the RWKV-FLA fork). Note the KDA route may pin an older "
            "'fla' without rwkv7 ops — if so, isolate the RWKV "
            "environment instead of overwriting the KDA pin. "
            f"(import error: {type(exc).__name__}: {exc})"
        )
        _FLA_PROBED = True
        _FLA_MODULE = sys.modules.get('fla.ops.rwkv7')
        raise _FLA_IMPORT_ERROR from exc
    _FLA_CHUNK_FN = chunk_rwkv7
    _FLA_RECURRENT_FN = fused_mul_recurrent_rwkv7
    _FLA_PROBED = True
    _FLA_MODULE = sys.modules.get('fla.ops.rwkv7')
    return _FLA_CHUNK_FN, _FLA_RECURRENT_FN


def rwkv7_fla_available() -> bool:
    """Return whether the fla RWKV-7 kernels can be imported.

    Availability PROBE: never raises (mirrors
    ``ops_kda_backend.fla_available``). A *broken* installation (import
    chain failing with ``RuntimeError`` from a Triton mismatch,
    ``OSError`` from a half-extracted wheel, ...) reads as "not
    available" so ``'auto'`` can fall back and provenance helpers stay
    honest; an explicitly requested ``'fla'`` dispatch still surfaces
    the actionable cached ``ImportError`` from
    :func:`_load_fla_rwkv7`.
    """
    try:
        _load_fla_rwkv7()
    except Exception:  # noqa: BLE001 - a probe must never raise
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


def _chunk_decay_param(chunk_fn) -> str:
    """Return the keyword that carries the log-decay into ``chunk_fn``.

    The chunked RWKV-7 kernel has two upstream spellings for the decay:

    * ``RWKV-Vibe/rwkv-fla`` (the fork this repo cites for the
      parameterization) — ``chunk_rwkv7(r, k, v, a, b, w=None, log_w=None)``.
      ``log_w`` carries the log-decay and is forwarded **unchanged**; ``w``
      is a RAW pre-activation that the kernel converts internally
      (``log_w = -exp(w)``).
    * mainline ``flash-linear-attention`` (the version ``pyproject.toml``
      pins) — ``chunk_rwkv7(r, w, k, v, a, b)`` where ``w`` IS the
      log-decay (``gk=w``, forwarded unchanged).

    Both spellings expect the SAME log-decay from this module, so the
    keyword is chosen from the INSTALLED signature: forwarding ``w=`` to
    the fork would apply the ``-exp()`` conversion a second time and
    silently change the decay; forwarding ``log_w=`` to mainline is
    swallowed by its ``**kwargs`` and leaves the required ``w`` unbound.
    """
    cached = _DECAY_PARAM_CACHE.get(chunk_fn)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(chunk_fn).parameters
    except (TypeError, ValueError):
        params = {}
    if 'log_w' in params:
        param = 'log_w'
    elif 'w' in params:
        param = 'w'
    else:
        # Signature not introspectable (a compiled wrapper): keep the
        # documented fork contract.  A mainline kernel then raises
        # TypeError, which the 'auto' path surfaces as a fallback.
        param = 'log_w'
    _DECAY_PARAM_CACHE[chunk_fn] = param
    return param


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
      ``T > 1``; ``fused_mul_recurrent_rwkv7`` (the single-step kernel)
      for ``T == 1`` — the same split the fla model layer itself uses.

    The fla kernels return the final state in fp32 and the output in
    the input dtype, matching the reference scan's contract.
    """
    chunk_fn, recurrent_fn = _load_fla_rwkv7()
    if r.shape[1] == 1:
        # Decode step: the fused recurrent kernel is selected for T == 1.
        # Same math, same state contract.
        o, state = recurrent_fn(
            r=r, w=log_w, k=k, v=v, kk=kk, a=a,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return o, state
    # The decay keyword differs between the two upstream spellings; both
    # receive the log-decay unchanged (see :func:`_chunk_decay_param`).
    chunk_kwargs = dict(
        r=r, k=k, v=v,
        a=-kk, b=kk * a,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    chunk_kwargs[_chunk_decay_param(chunk_fn)] = log_w
    o, state = chunk_fn(**chunk_kwargs)
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
    and falls back with a ``RuntimeWarning`` naming the cause;
    ``'reference'`` always uses the in-module scan. The 'auto' fallback
    surface mirrors ``ops_kda_backend.kda_forward``: ``ImportError`` /
    ``ValueError`` / ``TypeError`` / ``NotImplementedError`` /
    ``AssertionError`` / ``RuntimeError``, but NOT CUDA out-of-memory
    (re-raised: a resource failure must stay loud instead of silently
    re-running the reference scan).
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
        except (ImportError, ValueError, TypeError, NotImplementedError,
                AssertionError, RuntimeError) as exc:
            # API drift (renamed/missing kwargs -> TypeError/ValueError)
            # and runtime kernel failures (a Triton compile/launch failure
            # -> RuntimeError) both justify the documented 'auto' fallback
            # — the same exception surface ops_kda_backend uses. CUDA
            # out-of-memory does NOT: it is a resource failure, and
            # silently re-running the workload on the Python reference
            # loop would hide it.
            if isinstance(exc, RuntimeError) and \
                    'out of memory' in str(exc).lower():
                raise
            # Fall back with a warning so the resolution is visible.
            warnings.warn(
                f"rwkv7_forward: backend 'auto' fell back to the reference "
                f"scan ({type(exc).__name__}: {exc}). The resolved backend "
                f"must still be recorded via resolve_rwkv_backend / "
                f"effective_backend.",
                RuntimeWarning, stacklevel=2)
    return rwkv7_reference_scan(
        r, log_w, k, v, kk, a,
        scale=scale, initial_state=initial_state,
        output_final_state=output_final_state)
