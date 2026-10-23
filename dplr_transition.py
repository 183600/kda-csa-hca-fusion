"""Explicit diagonal-plus-low-rank (DPLR) transition algebra for KDA/RWKV-7.

Both gated linear-attention recurrences in this repository advance a
key-by-value state through a **diagonal-plus-low-rank** transition:

KDA (per step)::

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
        = [Diag(alpha_t) - beta_t k_t (alpha_t * k_t)^T] S_{t-1}
          + beta_t k_t v_t^T
        = [Diag(d_t) + U_t V_t^T] S_{t-1} + b_t

RWKV-7 (per step)::

    S_t = [Diag(w_t) + kka_t kk_t^T] S_{t-1} + k_t v_t^T

so both are the same algebraic object with rank ``r = 1`` per step.  A
*chunk* of ``C`` steps composes those maps into another DPLR map with
``r <= C`` (or ``r <= K`` after exact rank compression)::

    M_chunk = M_C ... M_1 = Diag(prod_c d_c) + L R^T

This module makes that structure explicit and auditable.  It is a
correctness-reference companion to ``ops_kda``: the chunked KDA path
already builds the same map implicitly through the delta-rule/wY
triangular solve, and :func:`kda_dplr_certificate` checks the two
constructions against each other and against a plain per-step loop.

Why it matters
--------------
* **Correctness.**  The wY identity ``M = Diag(gexp_last) - kexp^T w`` is
  load-bearing for the chunk and associative-scan paths, but the existing
  tests pin it only through end-to-end outputs.  The certificate here
  isolates the transition itself, so a sign/order regression is reported
  as a transition error rather than as a diffuse output drift.
* **Cost.**  A DPLR factor pair costs ``K r`` scalars instead of ``K^2``
  and applying it costs ``O(K V + K r V)`` instead of ``O(K^2 V)``.  The
  composition rule below doubles the rank exactly (``r1 + r2``) and can
  compress back to ``K`` without approximation, which is what a kernel
  that keeps transitions factored would need.
* **Regression testing.**  :func:`kda_dplr_certificate` returns a
  JSON-safe record (errors, ranks, conditioning) that the repo's
  correctness suite runs over a fixed random matrix, so a refactor that
  changes the implicit transition cannot pass silently.

Composition rule (exact, non-commutative)::

    M_2 M_1 = [Diag(d_2) + U_2 V_2^T][Diag(d_1) + U_1 V_1^T]
            = Diag(d_2 * d_1)
              + [d_2 * U_1 + U_2 (V_2^T U_1),  U_2] [V_1, d_1 * V_2]^T

The rank of the composition is ``r_1 + r_2``; :func:`compose` compresses
it back to at most ``K`` with a QR factorisation (exact up to fp
round-off) when requested.

All functions are pure PyTorch, CPU/GPU agnostic and do not mutate their
inputs.
"""

from __future__ import annotations

import torch

__all__ = [
    'compose',
    'dense',
    'effective_rank',
    'from_kda_step',
    'from_rwkv7_step',
    'kda_dplr_certificate',
    'kda_chunk_factors',
    'rwkv7_chunk_factors',
    'scan_work_plan',
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_factor(diag: torch.Tensor, left: torch.Tensor,
                  right: torch.Tensor, name: str) -> None:
    """Validate one DPLR factor triple ``(diag, left, right)``."""
    for tag, t in (('diag', diag), ('left', left), ('right', right)):
        if not torch.is_tensor(t):
            raise TypeError(f"{name}: {tag} must be a torch.Tensor, got "
                            f"{type(t).__name__}.")
        if not t.is_floating_point():
            raise TypeError(f"{name}: {tag} must be a floating-point tensor, "
                            f"got {t.dtype}.")
    if diag.dim() < 1:
        raise ValueError(f"{name}: diag must have at least one dimension.")
    k = diag.shape[-1]
    if left.shape[-2] != k or right.shape[-2] != k:
        raise ValueError(
            f"{name}: expected left/right with shape [..., {k}, r], got "
            f"{tuple(left.shape)} and {tuple(right.shape)}.")
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(
            f"{name}: left/right rank mismatch: {left.shape[-1]} vs "
            f"{right.shape[-1]}.")
    if left.shape[:-2] != diag.shape[:-1] or right.shape[:-2] != diag.shape[:-1]:
        raise ValueError(
            f"{name}: diag {tuple(diag.shape)} and left "
            f"{tuple(left.shape)} / right {tuple(right.shape)} have "
            f"different leading dimensions.")
    if left.device != diag.device or right.device != diag.device:
        raise ValueError(f"{name}: diag/left/right must share a device.")


def _compress(diag: torch.Tensor, left: torch.Tensor,
              right: torch.Tensor) -> tuple:
    """Exact rank compression of ``left @ right^T`` to at most ``K`` columns.

    ``left = Q R`` (reduced QR) gives ``left right^T = Q (R right^T)`` so
    the factor pair ``(Q, right R^T)`` represents the identical matrix.
    No singular values are discarded: when ``rank > K`` the pair is merely
    re-expressed in a ``K``-dimensional basis.
    """
    k = diag.shape[-1]
    rank = left.shape[-1]
    if rank <= k:
        return diag, left, right
    q, r = torch.linalg.qr(left, mode='reduced')
    # q: [..., K, K], r: [..., K, rank]; q @ (r @ right^T) == left @ right^T.
    new_right = right @ r.transpose(-1, -2)
    return diag, q, new_right


def effective_rank(left: torch.Tensor, right: torch.Tensor,
                   *, rtol: float | None = None) -> int:
    """Numerical rank of the low-rank part ``left @ right^T``.

    ``rtol`` defaults to ``max(shape) * eps`` of the tensor dtype, the
    usual SVD cut-off.  The returned rank is an ``int`` and therefore
    JSON-safe.  A zero-width factor has rank 0.
    """
    if left.shape[-1] == 0:
        return 0
    if rtol is None:
        eps = torch.finfo(left.dtype).eps
        rtol = max(left.shape[-1], left.shape[-2]) * eps
    matrix = left @ right.transpose(-1, -2)
    sv = torch.linalg.svdvals(matrix)
    if sv.numel() == 0 or not torch.isfinite(sv).all():
        return -1
    cutoff = float(rtol) * sv[..., :1]
    return int((sv > cutoff).sum(dim=-1).max().item())


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------
def dense(diag: torch.Tensor, left: torch.Tensor,
          right: torch.Tensor) -> torch.Tensor:
    """Materialise ``Diag(diag) + left @ right^T`` (``[..., K, K]``)."""
    _check_factor(diag, left, right, 'dense')
    return torch.diag_embed(diag) + left @ right.transpose(-1, -2)


def compose(diag1: torch.Tensor, left1: torch.Tensor, right1: torch.Tensor,
            diag2: torch.Tensor, left2: torch.Tensor,
            right2: torch.Tensor, *, compress: bool = True) -> tuple:
    """Compose two DPLR maps: return the factors of ``M_2 @ M_1``.

    Args:
        diag1/left1/right1: factors of the map applied **first**.
        diag2/left2/right2: factors of the map applied **second**.
        compress: when True (default) an exact QR compression caps the
            returned rank at ``K``.  When False the rank is
            ``r1 + r2`` and the caller may compress once at the end (the
            per-chunk builder does this to avoid one QR per step).

    Returns:
        ``(diag, left, right)`` with ``diag = diag2 * diag1`` and rank
        ``r1 + r2`` (or ``<= K`` after compression).
    """
    _check_factor(diag1, left1, right1, 'compose(first)')
    _check_factor(diag2, left2, right2, 'compose(second)')
    if diag1.shape[:-1] != diag2.shape[:-1]:
        raise ValueError(
            "compose: the two maps must have identical leading dimensions, "
            f"got {tuple(diag1.shape)} and {tuple(diag2.shape)}.")
    # Cross term V_2^T U_1: [..., r2, r1].
    cross = right2.transpose(-1, -2) @ left1
    new_left = torch.cat(
        [diag2.unsqueeze(-1) * left1 + left2 @ cross, left2], dim=-1)
    new_right = torch.cat(
        [right1, diag1.unsqueeze(-1) * right2], dim=-1)
    new_diag = diag1 * diag2
    if compress:
        return _compress(new_diag, new_left, new_right)
    return new_diag, new_left, new_right


def from_kda_step(k: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor) -> tuple:
    """Rank-1 DPLR factors of one KDA step.

    ``M_t = Diag(exp(g_t)) - beta_t k_t (exp(g_t) * k_t)^T``.  The
    source term ``beta_t k_t v_t^T`` is not part of the transition and is
    returned separately by the caller when needed.

    Args:
        k: ``[..., K]`` key vector.
        g: ``[..., K]`` log-decay (``alpha = exp(g)``).
        beta: ``[...]`` delta-rule learning rate.
    """
    for tag, t in (('k', k), ('g', g), ('beta', beta)):
        if not torch.is_tensor(t):
            raise TypeError(f"from_kda_step: {tag} must be a tensor.")
    if k.shape != g.shape:
        raise ValueError(
            f"from_kda_step: k/g shape mismatch {tuple(k.shape)} vs "
            f"{tuple(g.shape)}.")
    if beta.shape != k.shape[:-1]:
        raise ValueError(
            f"from_kda_step: beta must have shape {tuple(k.shape[:-1])}, "
            f"got {tuple(beta.shape)}.")
    alpha = g.exp()
    diag = alpha
    left = (-beta.unsqueeze(-1) * k).unsqueeze(-1)   # [..., K, 1]
    # NOTE: the rank direction is alpha * k (the gate is applied before the
    # delta-rule correction), not k.  Swapping it would silently change the
    # state decay of the correction.
    right = (alpha * k).unsqueeze(-1)                # [..., K, 1]
    return diag, left, right


def from_rwkv7_step(w: torch.Tensor, kk: torch.Tensor,
                    kka: torch.Tensor) -> tuple:
    """Rank-1 DPLR factors of one RWKV-7 step.

    ``M_t = Diag(w_t) + kka_t kk_t^T`` (the ``k_t v_t^T`` source is not
    part of the transition).  ``kka`` already carries the sign
    convention of :func:`ops_rwkv.rwkv7_reference_scan`.
    """
    for tag, t in (('w', w), ('kk', kk), ('kka', kka)):
        if not torch.is_tensor(t):
            raise TypeError(f"from_rwkv7_step: {tag} must be a tensor.")
    if w.shape != kk.shape or w.shape != kka.shape:
        raise ValueError(
            "from_rwkv7_step: w/kk/kka must have identical shapes, got "
            f"{tuple(w.shape)}, {tuple(kk.shape)}, {tuple(kka.shape)}.")
    return w, kka.unsqueeze(-1), kk.unsqueeze(-1)


def _reshape_chunks(x: torch.Tensor, chunk_size: int, pad_value: float,
                    time_dim: int) -> tuple:
    """Pad the time axis and reshape to ``[..., NT, C, D]``.

    ``time_dim`` must be given explicitly (KDA's ``k`` is ``[B, T, H, K]``
    so its time axis is ``-3``, while ``beta`` is ``[B, T, H]`` and uses
    ``-2``).  Padding appends ``pad`` identity steps at the end.
    """
    axis = time_dim if time_dim >= 0 else x.dim() + time_dim
    if not 0 <= axis < x.dim():
        raise ValueError(
            f"_reshape_chunks: time_dim={time_dim} is out of range for a "
            f"{x.dim()}-D tensor.")
    t = x.shape[axis]
    pad = (-t) % chunk_size
    if pad:
        shape = list(x.shape)
        shape[axis] = pad
        filler = torch.full(shape, pad_value, dtype=x.dtype, device=x.device)
        x = torch.cat([x, filler], dim=axis)
    nt = (t + pad) // chunk_size
    shape = list(x.shape)
    shape[axis:axis + 1] = [nt, chunk_size]
    return x.reshape(shape), nt


def kda_chunk_factors(k: torch.Tensor, g: torch.Tensor,
                      beta: torch.Tensor, chunk_size: int) -> tuple:
    """Compose per-step KDA factors into per-chunk DPLR factors.

    Args:
        k: ``[B, T, H, K]`` keys.
        g: ``[B, T, H, K]`` log-decays.
        beta: ``[B, T, H]`` learning rates.
        chunk_size: positive chunk length ``C``.  A non-divisible ``T``
            is right-padded with identity steps (``g=0, beta=0``).

    Returns:
        ``(diag, left, right)`` with shapes ``[B, H, NT, K]``,
        ``[B, H, NT, K, r]`` and ``[B, H, NT, K, r]``.  The chunk map is
        ``M_n = M_{(n+1)C} ... M_{nC+1}``.
    """
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size < 1):
        raise ValueError(
            f"kda_chunk_factors: chunk_size={chunk_size!r} must be a "
            f"positive int.")
    if k.shape != g.shape or k.dim() != 4:
        raise ValueError(
            f"kda_chunk_factors: k/g must share shape [B, T, H, K], got "
            f"{tuple(k.shape)} and {tuple(g.shape)}.")
    if beta.shape != k.shape[:3]:
        raise ValueError(
            f"kda_chunk_factors: beta must have shape {tuple(k.shape[:3])}, "
            f"got {tuple(beta.shape)}.")
    kc, nt = _reshape_chunks(k, chunk_size, 0.0, time_dim=-3)
    gc, _ = _reshape_chunks(g, chunk_size, 0.0, time_dim=-3)
    bc, _ = _reshape_chunks(beta, chunk_size, 0.0, time_dim=-2)
    if nt == 0:
        # Empty sequence: there is no chunk to compose.  ``torch.stack``
        # rejects an empty list, so return the naturally shaped empty
        # factors (zero chunks, zero-width rank) instead.  ``kc`` is
        # ``[B, NT, C, H, K]`` here, so the documented factor layout
        # ``[B, H, NT, K, r]`` is assembled from the batch/head/dim axes.
        lead = (kc.shape[0], kc.shape[-2], 0)
        dim = kc.shape[-1]
        return (kc.new_empty(*lead, dim),
                kc.new_empty(*lead, dim, 0),
                kc.new_empty(*lead, dim, 0))
    diag_list, left_list, right_list = [], [], []
    for n in range(nt):
        diag = None
        left = None
        right = None
        for c in range(chunk_size):
            d_c, l_c, r_c = from_kda_step(kc[:, n, c], gc[:, n, c],
                                          bc[:, n, c])
            if diag is None:
                diag, left, right = d_c, l_c, r_c
            else:
                # Current map is M_{c-1} ... M_0; the new map is M_c, so
                # the composed map is M_c @ current.
                diag, left, right = compose(
                    diag, left, right, d_c, l_c, r_c, compress=False)
        diag, left, right = _compress(diag, left, right)
        diag_list.append(diag)
        left_list.append(left)
        right_list.append(right)
    return (torch.stack(diag_list, dim=2),
            torch.stack(left_list, dim=2),
            torch.stack(right_list, dim=2))


def rwkv7_chunk_factors(w: torch.Tensor, kk: torch.Tensor,
                        kka: torch.Tensor, chunk_size: int) -> tuple:
    """Compose per-step RWKV-7 factors into per-chunk DPLR factors.

    Args are ``[..., T, K]``; the time axis is ``-2``.  Same padding and
    composition order as :func:`kda_chunk_factors`.
    """
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size < 1):
        raise ValueError(
            f"rwkv7_chunk_factors: chunk_size={chunk_size!r} must be a "
            f"positive int.")
    wc, nt = _reshape_chunks(w, chunk_size, 1.0, time_dim=-2)
    kkc, _ = _reshape_chunks(kk, chunk_size, 0.0, time_dim=-2)
    kkac, _ = _reshape_chunks(kka, chunk_size, 0.0, time_dim=-2)
    if nt == 0:
        # Empty sequence: no chunk map to compose (see the KDA counterpart).
        lead = wc.shape[:-3] + (0,)
        dim = wc.shape[-1]
        return (wc.new_empty(*lead, dim),
                wc.new_empty(*lead, dim, 0),
                wc.new_empty(*lead, dim, 0))
    diag_list, left_list, right_list = [], [], []
    for n in range(nt):
        diag = None
        left = None
        right = None
        for c in range(chunk_size):
            d_c, l_c, r_c = from_rwkv7_step(wc[..., n, c, :],
                                            kkc[..., n, c, :],
                                            kkac[..., n, c, :])
            if diag is None:
                diag, left, right = d_c, l_c, r_c
            else:
                diag, left, right = compose(
                    diag, left, right, d_c, l_c, r_c, compress=False)
        diag, left, right = _compress(diag, left, right)
        diag_list.append(diag)
        left_list.append(left)
        right_list.append(right)
    # ``diag`` is ``[..., K]`` (one fewer axis than ``left``/``right``), so
    # the new chunk axis sits at ``-2`` for the diagonal but at ``-3`` for
    # the rank factors: stacking the latter with ``-2`` would interleave
    # the chunk axis between K and the rank axis, yielding
    # ``[..., K, NT, r]`` instead of the documented ``[..., NT, K, r]``.
    return (torch.stack(diag_list, dim=-2),
            torch.stack(left_list, dim=-3),
            torch.stack(right_list, dim=-3))


# ---------------------------------------------------------------------------
# Two-phase block-scan work model (producer / scanner / consumer)
# ---------------------------------------------------------------------------
def _scan_compositions(n: int) -> int:
    """Map compositions performed by the Hillis-Steele doubling core."""
    total = 0
    off = 1
    while off < n:
        total += n - off
        off *= 2
    return total


def scan_work_plan(n_chunks: int, block_size: int, *, dim: int = 128,
                   value_dim: int = 64, chunk_size: int = 64) -> dict:
    """Cost model for the blocked two-phase (specialised) chunk scan.

    The reference implementation in ``ops_kda._blocked_chunk_states``
    splits the scan into three specialised roles, the CPU analogue of
    warp specialisation:

    * **producer** — build one block's composed affine map from its
      per-chunk DPLR factors (independent across blocks, so parallel);
    * **scanner** — Hillis-Steele prefix over the block maps only;
    * **consumer** — replay each block from its scanned incoming state
      and emit the per-chunk states (sequential in the block order).

    Only the transition work is modelled: the intra-chunk readout
    (``C_n S_n + a_n``) is independent of the state-resolution strategy
    and cancels when comparing full and blocked scans.  MAC counts are
    dominant-term estimates, not a latency prediction.

    Args:
        n_chunks: number of chunks ``NT >= 1``.
        block_size: chunk count per block (``>= 1``; the caller treats 1
            as the full scan).
        dim: key/state dimension ``K``.
        value_dim: value dimension ``V``.
        chunk_size: token count per chunk ``C``.

    Returns:
        JSON-safe dict with composition counts, transition storage and a
        producer/consumer balance diagnostic.  ``producer_hidden`` is
        True when one chunk's producer work fits inside its consumer
        work, i.e. a specialised producer warp could run ahead of the
        sequential state chain.
    """
    for name, value, low in (('n_chunks', n_chunks, 1),
                             ('block_size', block_size, 1),
                             ('dim', dim, 1),
                             ('value_dim', value_dim, 1),
                             ('chunk_size', chunk_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(
                f"scan_work_plan: {name}={value!r} must be an int >= {low}.")
    n_blocks = (n_chunks + block_size - 1) // block_size
    within = n_chunks - n_blocks          # producer compositions
    scanner = _scan_compositions(n_blocks)
    dense = _scan_compositions(n_chunks)
    total = within + scanner
    k, v, c = dim, value_dim, chunk_size
    producer = k * k * c + k * c * v      # kexp^T w and kexp^T u
    consumer = k * k * v                  # M_n @ S_n plus the source add
    return {
        'n_chunks': int(n_chunks),
        'block_size': int(block_size),
        'n_blocks': int(n_blocks),
        'producer_compositions': int(within),
        'scanner_compositions': int(scanner),
        'scan_compositions': int(total),
        'full_scan_compositions': int(dense),
        'composition_savings_factor': (
            float(dense / total) if total else 1.0),
        'dense_transition_scalars': int(n_chunks * k * k),
        'blocked_transition_scalars': int(n_blocks * k * k),
        'transition_memory_ratio': (
            float(n_blocks / n_chunks) if n_chunks else 1.0),
        'producer_macs_per_chunk': int(producer),
        'consumer_macs_per_chunk': int(consumer),
        'producer_hidden_by_consumer': bool(producer <= consumer),
        'serial_transition_macs': int(
            n_chunks * (producer + consumer) + scanner * k ** 3),
        'specialized_transition_macs': int(
            n_chunks * max(producer, consumer) + scanner * k ** 3),
    }


# ---------------------------------------------------------------------------
# Certificate: the chunked wY map vs the per-step DPLR product
# ---------------------------------------------------------------------------
def _default_tol(dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return 1e-10
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2
    return 1e-5


def kda_dplr_certificate(k: torch.Tensor, g: torch.Tensor,
                         beta: torch.Tensor, v: torch.Tensor,
                         chunk_size: int) -> dict:
    """Audit the KDA chunk transition against its per-step DPLR product.

    Three independent constructions are compared on the same inputs:

    1. ``per-step``: apply ``Diag(exp(g_t)) + U_t V_t^T`` and the source
       ``beta_t k_t v_t^T`` one token at a time;
    2. ``dplr chunk``: compose the per-step factors with
       :func:`kda_chunk_factors` and densify;
    3. ``wY chunk``: the implicit map built by ``ops_kda``'s triangular
       solve, ``Diag(gexp_last) - kexp^T w`` with source ``kexp^T u``.

    Args:
        k: ``[B, T, H, K]`` keys.
        g: ``[B, T, H, K]`` log-decays (no clamp is applied here).
        beta: ``[B, T, H]`` learning rates.
        v: ``[B, T, H, V]`` values (drives the source term).
        chunk_size: positive chunk length.

    Returns:
        JSON-safe dict with absolute/relative errors, the numerical rank
        of every low-rank part (max over chunks), the rank bound, and
        ``passed`` against a dtype-derived tolerance.  ``not_estimable``
        is returned instead of raising when the inputs are non-finite.
    """
    for tag, t in (('k', k), ('g', g), ('beta', beta), ('v', v)):
        if not torch.is_tensor(t):
            raise TypeError(f"kda_dplr_certificate: {tag} must be a tensor.")
    if k.dim() != 4 or g.shape != k.shape or v.dim() != 4:
        raise ValueError(
            "kda_dplr_certificate: expected k/g [B,T,H,K] and v "
            f"[B,T,H,V]; got {tuple(k.shape)}, {tuple(g.shape)}, "
            f"{tuple(v.shape)}.")
    if beta.shape != (k.shape[0], k.shape[1], k.shape[2]):
        raise ValueError(
            f"kda_dplr_certificate: beta must be [B,T,H]="
            f"{tuple(beta.shape[:3])}, got {tuple(beta.shape)}.")
    if v.shape[:3] != k.shape[:3]:
        raise ValueError(
            "kda_dplr_certificate: v must share B,T,H with k, got "
            f"{tuple(v.shape)} vs {tuple(k.shape)}.")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or chunk_size < 1):
        raise ValueError(
            f"kda_dplr_certificate: chunk_size={chunk_size!r} must be a "
            f"positive int.")
    if not all(torch.isfinite(t).all() for t in (k, g, beta, v)):
        return {'not_estimable': True, 'reason': 'non-finite inputs',
                'passed': False}

    from ops_kda import _chunk_kda_prepare  # lazy: avoids an import cycle

    b, t, h, dim = k.shape
    val_dim = v.shape[-1]
    dtype = k.dtype
    # ``_chunk_kda_prepare`` builds the wY transform.  q is irrelevant to
    # the transition/source, so a zero tensor keeps the certificate focused;
    # g_clamp_min=-inf disables the clamp so the comparison uses raw g.
    q = torch.zeros_like(k)
    prep = _chunk_kda_prepare(
        q, k, v, g, beta, None, None, False, chunk_size, float('-inf'),
        None, 'kda_dplr_certificate')
    if prep['early']:
        return {'not_estimable': True, 'reason': 'T == 0', 'passed': False}
    gg, ggx, w, kk, u, nt = (
        prep['g'], prep['gexp'], prep['w'], prep['k'], prep['u'],
        prep['NT'])
    kexp = torch.exp(gg[..., -1:, :] - gg) * kk
    m_wy = torch.diag_embed(ggx[..., -1, :]) - \
        kexp.transpose(-1, -2) @ w
    b_wy = kexp.transpose(-1, -2) @ u

    d_chunk, l_chunk, r_chunk = kda_chunk_factors(k, g, beta, chunk_size)
    m_dplr = dense(d_chunk, l_chunk, r_chunk)

    # Per-step state recurrence (independent of the chunk machinery).
    kc, _ = _reshape_chunks(k, chunk_size, 0.0, time_dim=-3)
    gc, _ = _reshape_chunks(g, chunk_size, 0.0, time_dim=-3)
    bc, _ = _reshape_chunks(beta, chunk_size, 0.0, time_dim=-2)
    vc, _ = _reshape_chunks(v, chunk_size, 0.0, time_dim=-3)
    # Deterministic, non-random initial state: the certificate must not
    # perturb the caller's global RNG stream.
    idx = torch.arange(dim * val_dim, dtype=dtype, device=k.device)
    s0 = torch.sin(idx * 0.7).reshape(1, 1, dim, val_dim) * 0.1
    s_step = s0
    for n in range(nt):
        for c in range(chunk_size):
            d_c, l_c, r_c = from_kda_step(kc[:, n, c], gc[:, n, c],
                                          bc[:, n, c])
            # b_c = beta_c k_c v_c^T
            source = bc[:, n, c].unsqueeze(-1).unsqueeze(-1) * \
                kc[:, n, c].unsqueeze(-1) * vc[:, n, c].unsqueeze(-2)
            s_step = d_c.unsqueeze(-1) * s_step + \
                l_c @ (r_c.transpose(-1, -2) @ s_step) + source

    # Chunked recurrence driven by the explicit DPLR maps.
    s_chunk = s0
    for n in range(nt):
        s_chunk = m_dplr[:, :, n] @ s_chunk + b_wy[:, :, n]

    # Chunked recurrence driven by the implicit wY maps.
    s_wy = s0
    for n in range(nt):
        s_wy = m_wy[:, :, n] @ s_wy + b_wy[:, :, n]

    def _max_abs(x: torch.Tensor) -> float:
        return float(x.abs().max().item()) if x.numel() else 0.0

    map_err = _max_abs(m_dplr - m_wy)
    step_err = _max_abs(s_step - s_chunk)
    wy_err = _max_abs(s_step - s_wy)
    scale = max(_max_abs(s_step), _max_abs(m_wy), 1.0)
    tol = _default_tol(dtype)
    rank_dplr = max(effective_rank(l_chunk[:, :, n], r_chunk[:, :, n])
                    for n in range(nt))
    rank_wy = max(effective_rank(
        -kexp[:, :, n].transpose(-1, -2),
        w[:, :, n].transpose(-1, -2)) for n in range(nt))
    return {
        'n_chunks': int(nt),
        'chunk_size': int(chunk_size),
        'n_tokens': int(t),
        'dim': int(dim),
        'value_dim': int(val_dim),
        'max_abs_map_error': map_err,
        'max_abs_state_error': step_err,
        'max_abs_wy_state_error': wy_err,
        'max_rel_error': max(map_err, step_err) / scale,
        'rank_max': int(rank_dplr),
        'wy_rank_max': int(rank_wy),
        'rank_bound': int(min(chunk_size, dim)),
        'tolerance': tol,
        'passed': bool(map_err <= tol * max(1.0, _max_abs(m_wy))
                       and step_err <= tol * max(1.0, _max_abs(s_step))
                       and wy_err <= tol * max(1.0, _max_abs(s_step))),
    }


# ---------------------------------------------------------------------------
# CLI: a small, reproducible certificate table
# ---------------------------------------------------------------------------
def _main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Audit the DPLR structure of KDA/RWKV-7 transitions.')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--value-dim', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', choices=('float32', 'float64'),
                        default='float64')
    parser.add_argument('--plan-chunks', type=int, default=0,
                        help='print the two-phase scan work plan for this '
                             'many chunks instead of the certificate')
    parser.add_argument('--plan-block', type=int, default=8)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.dim < 1 or args.chunk_size < 1 or args.tokens < 1:
        parser.error('--dim/--chunk-size/--tokens must be positive')

    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    if args.plan_chunks:
        plan = scan_work_plan(
            args.plan_chunks, args.plan_block, dim=args.dim,
            value_dim=args.value_dim, chunk_size=args.chunk_size)
        if args.json:
            print(json.dumps(plan, sort_keys=True))
        else:
            print(f"two-phase block scan work plan (NT={args.plan_chunks}, "
                  f"B={args.plan_block}, K={args.dim}, C={args.chunk_size})")
            for key, value in plan.items():
                print(f"  {key:30s} {value}")
        return 0
    torch.manual_seed(args.seed)
    b, t, h, dim, vdim = 1, args.tokens, args.heads, args.dim, args.value_dim
    k = torch.nn.functional.normalize(torch.randn(b, t, h, dim, dtype=dtype),
                                      dim=-1)
    g = -torch.rand(b, t, h, dim, dtype=dtype) * 0.1
    beta = torch.rand(b, t, h, dtype=dtype) * 0.2
    v = torch.randn(b, t, h, vdim, dtype=dtype) * 0.1
    cert = kda_dplr_certificate(k, g, beta, v, args.chunk_size)
    if args.json:
        print(json.dumps(cert, sort_keys=True))
    else:
        print(f"KDA DPLR certificate (dim={dim}, C={args.chunk_size}, "
              f"T={t}, dtype={args.dtype})")
        for key in ('n_chunks', 'max_abs_map_error', 'max_abs_state_error',
                    'max_abs_wy_state_error', 'max_rel_error', 'rank_max',
                    'rank_bound', 'tolerance', 'passed'):
            print(f"  {key:24s} {cert[key]}")
    return 0 if cert.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(_main())
