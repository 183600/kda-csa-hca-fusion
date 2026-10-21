"""Partial Rotary Positional Embedding (RoPE) for CSA / HCA.

Implements the partial RoPE described in DeepSeek-V4 (arXiv:2606.19348v1,
§2.3.3 "Partial Rotary Positional Embedding"):

    "For each query vector and KV entry vector used in CSA and HCA, we
    apply RoPE to its last 64 dimensions. Since the KV entries serve as
    both attention keys and values, the naive core attention outputs
    {o_t,i} will carry absolute position embeddings, derived from the
    weighted sum of KV entries. As a countermeasure, we also apply RoPE
    with position [minus the query position] on the last 64 dimensions of
    each o_t,i. In this way, the output of the core attention will also
    carry relative position embeddings — the contribution of each KV
    entry to the core attention outputs will also be related to the
    distance between the query and the KV entry."

Design notes (interpretation choices the paper leaves open)
-----------------------------------------------------------

1. **Which dims rotate.** Only the LAST ``rope_dim`` dimensions of each
   per-head vector (``rope_dim = 64`` in the paper). ``effective_rope_dim``
   clamps the request to the largest even value that fits the head
   dimension, so tiny test geometries (e.g. ``c = 8``) still exercise the
   RoPE path with ``rope_dim = 8``.

2. **Position of a compressed KV entry.** A compressed entry aggregates a
   window of source tokens, so no single "true" position exists. We
   assign the *block anchor* position ``s * m`` (the first source-token
   position of block ``s``; for CSA's overlapped two-branch compression
   this is also the boundary between the b-branch and a-branch source
   windows). This keeps every position in TOKEN units, consistent with
   the query positions and the sliding-window key positions, so the
   relative phase (t − s·m) is meaningful.

3. **Output inverse rotation.** The paper's phrasing ("RoPE with
   position −i") is interpreted, per its stated intent, as rotating the
   core-attention output by the NEGATED QUERY POSITION (−t): the
   contribution of entry s to query t becomes R(s·m − t)·v_s, a function
   of the query–entry distance only.

4. **Pairing convention.** GPT-J-style adjacent pairing *within the rope
   slice*: dims (d0, d1), (d2, d3), ... of the last ``rope_dim`` dims form
   rotation pairs, with inv_freq_j = base^(−2j / rope_dim). Any
   self-consistent convention would work; this one is used everywhere
   (queries, compressed keys/values, sliding-window keys, and the output
   inverse rotation).

The rotation is applied via :func:`partial_rope`; the inverse rotation is
the same function with negated positions.

5. **Cache tensor ownership.** The angle/position caches hold ORDINARY
   tensors even when populated from inside ``torch.inference_mode()`` (see
   ``ops_cache.normal_tensor_scope``). Inference tensors carry no version
   counter and cannot be saved for backward, so caching one would crash the
   next long-context inference call (identity-cache key reads
   ``positions._version``) or the next grad-enabled call. Regression tests:
   ``test_rope_inference_mode.py`` and ``test_cache_inference_mode.py``.
"""

from __future__ import annotations

import collections as _collections
import weakref as _weakref

import torch

from ops_cache import normal_tensor_scope


def effective_rope_dim(requested: int | None, head_dim: int) -> int:
    """Clamp a requested rope width to a usable even value.

    Returns the effective rope dimension: the largest even integer that is
    ``<= min(requested, head_dim)``. Returns 0 (RoPE disabled) when
    ``requested`` is None / <= 0 or ``head_dim`` < 2.

    Paper-faithful default: ``effective_rope_dim(64, c)`` — for the repo's
    default CSA/HCA head dim (``c = 64``) this is exactly 64; for the
    small MQAR/decoding geometries (``c = 8/16/32``) it clamps to the head
    dim so the RoPE code path is still exercised.
    """
    if requested is None:
        return 0
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ValueError(
            f"rope_dim={requested!r} must be an int (or None to disable)")
    if requested <= 0:
        return 0
    if head_dim < 2:
        return 0
    r = min(requested, head_dim)
    return r - (r % 2)


def _angle_dtype(x: torch.Tensor) -> torch.dtype:
    """Precision for the sin/cos tables (mirrors the repo compute_dtype)."""
    return torch.float64 if x.dtype == torch.float64 else torch.float32


# Backwards-compatible local name; the shared implementation lives in
# ``ops_cache`` so every operator uses the same cache-ownership rule.
_normal_tensor_scope = normal_tensor_scope


# ``inv_freq = base ** (-arange(0, rope_dim, 2) / rope_dim)`` depends only on
# (rope_dim, base, device, dtype) — never on the data. The tensors are treated
# as immutable by all callers.
_INV_FREQ_CACHE: dict = {}


def _inv_freq(rope_dim: int, base: float, device, dtype: torch.dtype) -> torch.Tensor:
    """Cached ``base ** (-arange(0, rope_dim, 2) / rope_dim)`` table."""
    key = (rope_dim, float(base), str(device), dtype)
    inv_freq = _INV_FREQ_CACHE.get(key)
    if inv_freq is None:
        with _normal_tensor_scope():
            inv_freq = base ** (
                -torch.arange(0, rope_dim, 2, dtype=dtype, device=device)
                / float(rope_dim)
            )
        _INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


# ``(cos, sin)`` tables for INTEGER position vectors, keyed by the exact
# position values plus (rope_dim, base, device, angle_dtype, position dtype).
# The decode path calls ``partial_rope`` several times per token with the
# same one-element position vector (query + sliding-window key), so the
# value key is tiny there and repeats across calls. The tables are pure
# functions of the key; entries are treated as immutable (only
# ``view``/read by consumers).
#
# Only SHORT position vectors are value-keyed (``_TABLE_CACHE_MAX_POS``):
# building the key materializes ``tuple(positions.tolist())``, which for a
# full-sequence call would be an O(P) host copy (and a P-int key) just to
# look up a table that is almost never reused at that size. Longer vectors
# take the uncached branch in :func:`partial_rope`, which is textually the
# same computation (bit-identical tables). The cache is an LRU bounded by
# total element count, so the resident tables stay small regardless of the
# requested sequence lengths.
_TABLE_CACHE: '_collections.OrderedDict' = _collections.OrderedDict()
_TABLE_CACHE_MAX_ENTRIES = 128
_TABLE_CACHE_MAX_ELEMS = 1 << 22
_TABLE_CACHE_ELEMS = 0
_TABLE_CACHE_MAX_POS = 8

# Integer position dtypes eligible for the table cache (float positions can
# carry signed zeros / NaN payloads that a value-keyed cache would merge).
_INT_POS_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)

# Element floor for materializing the broadcast ``(cos, sin)`` tables in
# ``partial_rope`` (see :func:`_materialize_pair_table`). Below this size the
# table is passed as a stride-0 broadcast view.
_ROPE_MATERIALIZE_MIN_ELEMS = 1024

# Identity/version-keyed cache for LONG position vectors (full-sequence
# ``token_positions``): the value-keyed ``_TABLE_CACHE`` above only covers
# short vectors. Within one CSA/HCA forward the same ``tpos`` tensor is
# passed to 2-3 ``partial_rope`` calls (query, sliding-window keys, negated
# output rotation), and its ``(cos, sin)`` tables are a pure function of the
# position bits plus the rope parameters. The key carries the object id,
# ``data_ptr`` and ``_version`` (an in-place update invalidates the entry);
# a weakref to the positions tensor guards against id reuse after the
# caller's tensor is garbage collected. Consumers treat positions as
# read-only, matching the existing direct computation.
_IDENTITY_TABLE_CACHE: dict = {}


# Shape-keyed cache for the position vectors built by ``token_positions`` /
# ``block_positions``. Within one CSA/HCA forward the same token-position
# vector is consumed by 2-3 ``partial_rope`` calls (query, sliding-window
# keys, negated output rotation); consumers only read / negate it, so the
# cached tensor is shared and must be treated as immutable.
_POSITION_VEC_CACHE: dict = {}


def _materialize_pair_table(
    tbl: torch.Tensor,
    target_shape,
    dim: int,
) -> torch.Tensor:
    """Expand a cached ``[P, r/2]`` table for a rotated pair slice.

    ``target_shape`` is the shape of the pair slice ``x[..., 0::2]``: ``P``
    along ``dim``, ``r/2`` last. Broadcasting the per-position table across
    an inner axis (the head axis of ``[B, T, nh, c]`` queries / outputs) adds
    a stride-0 axis directly above the feature axis, which drops the
    elementwise kernels onto a non-vectorized scalar loop. The inner
    broadcast axes are therefore materialized with ``cat``. A leading
    stride-0 axis (batch) is left broadcast: it collapses into the outer
    loop there and stays on the vectorized path. Both layouts present the
    same per-position values to the products and sums.
    """
    view_shape = [1] * len(target_shape)
    view_shape[dim] = tbl.shape[0]
    view_shape[-1] = tbl.shape[-1]
    out = tbl.view(view_shape)
    # Materialize every inner axis (between the leading axis and the
    # feature axis) that the source table does not already cover, in ONE
    # ``expand`` + ``contiguous`` copy. This fills the exact same element
    # slots as the former per-axis ``cat([out] * n)`` cascade with the
    # same broadcast values, without the repeated full-tensor copies.
    expand_shape = list(view_shape)
    needs_materialize = False
    for ax in range(1, len(target_shape) - 1):
        n = target_shape[ax]
        if ax != dim and n != 1:
            expand_shape[ax] = n
            needs_materialize = True
    if not needs_materialize:
        return out
    return out.expand(expand_shape).contiguous()


def _cos_sin_cached(
    positions: torch.Tensor,
    rope_dim: int,
    base: float,
    angle_dtype: torch.dtype,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``(cos, sin)`` tables for an integer ``positions`` vector.

    Only integer position dtypes (the ones every in-repo caller uses:
    ``torch.arange`` / ``torch.tensor([t])``) take this path; float
    positions — where ``-0.0`` vs ``+0.0`` or NaN payloads could make a
    value-keyed cache ambiguous — keep the direct computation. The two
    computations are otherwise textually identical, so a cache hit returns
    the exact bits the uncached path would produce.
    """
    global _TABLE_CACHE_ELEMS
    key = (rope_dim, float(base), str(device), angle_dtype,
           positions.dtype, tuple(positions.tolist()))
    entry = _TABLE_CACHE.get(key)
    if entry is not None:
        _TABLE_CACHE.move_to_end(key)
        return entry
    # Build the cached tables as ORDINARY tensors (see
    # :func:`_normal_tensor_scope`): a table created during an
    # inference-mode warm-up stays usable for every later autograd call.
    with _normal_tensor_scope():
        pos_f = positions.to(device=device, dtype=angle_dtype)
        angles = (pos_f[:, None]
                  * _inv_freq(rope_dim, base, device, angle_dtype)[None, :])
        cos = angles.cos()
        sin = angles.sin()
    entry_elems = cos.numel() + sin.numel()
    if entry_elems <= _TABLE_CACHE_MAX_ELEMS:
        while _TABLE_CACHE and (
                len(_TABLE_CACHE) >= _TABLE_CACHE_MAX_ENTRIES
                or _TABLE_CACHE_ELEMS + entry_elems > _TABLE_CACHE_MAX_ELEMS):
            _old, (_old_cos, _old_sin) = _TABLE_CACHE.popitem(last=False)
            _TABLE_CACHE_ELEMS -= _old_cos.numel() + _old_sin.numel()
        _TABLE_CACHE[key] = (cos, sin)
        _TABLE_CACHE_ELEMS += entry_elems
    return cos, sin


def _cos_sin_by_identity(
    positions: torch.Tensor,
    rope_dim: int,
    base: float,
    angle_dtype: torch.dtype,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity/version-keyed ``(cos, sin)`` tables (long position vectors).

    Computes exactly the same tables as the direct branch of
    :func:`partial_rope` (same expressions, same order), and returns the
    cached pair when the same tensor object with the same contents is
    passed again (the common case: one layer forward reuses ``tpos``).
    """
    # Inference tensors have NO version counter, so they cannot be keyed (and
    # are created by callers inside ``torch.inference_mode()`` for every
    # long-context forward). Compute their tables directly, still as
    # ordinary tensors so the result can feed a later autograd pass.
    cacheable = not positions.is_inference()
    if cacheable:
        key = (id(positions), positions.data_ptr(), positions._version,
               tuple(positions.shape), positions.dtype, rope_dim, float(base),
               str(device), angle_dtype)
        entry = _IDENTITY_TABLE_CACHE.get(key)
        if entry is not None:
            ref, cos, sin = entry
            if ref() is positions:
                return cos, sin
            _IDENTITY_TABLE_CACHE.pop(key, None)
    with _normal_tensor_scope():
        pos_f = positions.to(device=device, dtype=angle_dtype)
        angles = (pos_f[:, None]
                  * _inv_freq(rope_dim, base, device, angle_dtype)[None, :])
        cos = angles.cos()
        sin = angles.sin()
    if cacheable and cos.numel() + sin.numel() <= _TABLE_CACHE_MAX_ELEMS:
        if len(_IDENTITY_TABLE_CACHE) >= 8:
            _IDENTITY_TABLE_CACHE.clear()
        _IDENTITY_TABLE_CACHE[key] = (_weakref.ref(positions), cos, sin)
    return cos, sin


def partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    base: float = 10000.0,
    dim: int = 1,
) -> torch.Tensor:
    """Rotate the LAST ``rope_dim`` feature dims of ``x`` at given positions.

    Parameters
    ----------
    x : torch.Tensor
        ``[..., P, D]`` along ``dim`` — e.g. queries ``[B, T, nh, c]`` use
        ``dim=1`` (the sequence axis is T); compressed KV ``[B, n_blocks, c]``
        use ``dim=1`` as well.  ``dim=1`` is the default because it is the
        sequence axis of every layout this repository uses; the previous
        default ``-2`` resolved to the HEAD axis for the 4-D query layout
        and silently rotated the heads instead of the tokens.
    positions : torch.Tensor
        1-D tensor of length ``P = x.shape[dim]`` with the position of each
        row along the sequence axis. May be negative (negative positions
        implement the INVERSE rotation used for the core-attention output).
    rope_dim : int
        Number of trailing FEATURE dims (last axis) to rotate (already
        clamped by :func:`effective_rope_dim`; values <= 0 disable).
    base : float
        RoPE frequency base (default 10000.0).
    dim : int
        The sequence axis along which ``positions`` applies (default 1;
        must not be the last axis, which is the feature axis).  A 2-D
        ``[P, D]`` tensor must pass ``dim=0`` explicitly.

    Returns
    -------
    torch.Tensor
        Same shape / dtype as ``x`` with the last ``rope_dim`` feature dims
        rotated. The rotation is orthogonal (norm-preserving), so L2 norms
        computed before/after agree up to floating-point rounding.
    """
    if rope_dim is None or rope_dim <= 0:
        return x
    nd = x.dim()
    if not -nd <= dim < nd:
        raise ValueError(
            f"partial_rope: dim={dim} is out of range for a {nd}-D tensor "
            f"(valid range [{-nd}, {nd - 1}])")
    if dim == -1 or dim == nd - 1:
        raise ValueError(
            f"partial_rope: dim={dim} cannot be the feature (last) axis; "
            f"pass the sequence axis explicitly (dim=1 for the "
            f"[B, T, ...] layouts, dim=0 for a 2-D [P, D] tensor)")
    if dim < 0:
        dim = nd + dim
    D = x.shape[-1]
    if rope_dim > D:
        raise ValueError(
            f"partial_rope: rope_dim={rope_dim} exceeds the feature dim D={D}")
    P = x.shape[dim]
    if positions.numel() != P:
        raise ValueError(
            f"partial_rope: positions has {positions.numel()} entries but axis "
            f"dim={dim} of x has P={P}.")
    if P == 0:
        return x
    if rope_dim % 2 != 0:
        raise ValueError(
            f"partial_rope: rope_dim={rope_dim} must be even.")
    if not isinstance(base, (int, float)) or base <= 0:
        raise ValueError(f"partial_rope: base={base!r} must be positive.")
    device = x.device
    angle_dtype = _angle_dtype(x)
    if (positions.dtype in _INT_POS_DTYPES and not positions.requires_grad
            and positions.numel() <= _TABLE_CACHE_MAX_POS):
        # Exact-value table cache (see ``_cos_sin_cached``); the uncached
        # branch below performs the identical computation. Only short
        # position vectors are keyed by value: the key construction is
        # O(P) on the host.
        cos, sin = _cos_sin_cached(
            positions, rope_dim, base, angle_dtype, device)
    elif (positions.dtype in _INT_POS_DTYPES and not positions.requires_grad):
        # Long integer vectors: identity/version-keyed cache instead of the
        # O(P) value key (see ``_cos_sin_by_identity``). Same expressions as
        # the direct branch below, so values are bit-identical.
        cos, sin = _cos_sin_by_identity(
            positions, rope_dim, base, angle_dtype, device)
    else:
        # inv_freq_j = base^(-2j/rope_dim), j = 0 .. rope_dim/2 - 1
        # (see ``_inv_freq``).
        inv_freq = _inv_freq(rope_dim, base, device, angle_dtype)  # [rope_dim/2]
        # ``.to`` returns the tensor itself when dtype AND device already
        # match, so this guard is value-identical.
        if positions.dtype != angle_dtype or positions.device != device:
            positions = positions.to(device=device, dtype=angle_dtype)
        angles = positions[:, None] * inv_freq[None, :]      # [P, rope_dim/2]
        cos = angles.cos()
        sin = angles.sin()
    if angle_dtype != x.dtype:
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
    # Broadcast cos/sin so that the position axis lands on ``dim`` and the
    # pair axis lands on the last axis: shape [1, ..., P, ..., 1, r/2].
    bc_shape = [1] * x.dim()
    bc_shape[dim] = P
    bc_shape[-1] = rope_dim // 2

    x_rot = x[..., D - rope_dim:]                            # [..., P, r]
    x1 = x_rot[..., 0::2]                                    # [..., P, r/2]
    x2 = x_rot[..., 1::2]                                    # [..., P, r/2]
    if x1.numel() >= _ROPE_MATERIALIZE_MIN_ELEMS:
        # Broadcast-heavy path (per-head q / out rotations, and every long
        # sequence): dense tables + contiguous pair copies (identical
        # per-element math, see :func:`_materialize_pair_table`). ``o1`` /
        # ``o2`` are plain expressions, so this path is valid with and
        # without autograd.
        cos = _materialize_pair_table(cos, x1.shape, dim)
        sin = _materialize_pair_table(sin, x1.shape, dim)
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.empty(x.shape, dtype=x.dtype, device=device)
        if rope_dim < D:
            out[..., : D - rope_dim] = x[..., : D - rope_dim]
        out[..., D - rope_dim::2] = o1
        out[..., D - rope_dim + 1::2] = o2
        return out
    cos = cos.view(bc_shape)
    sin = sin.view(bc_shape)
    if (x.requires_grad or positions.requires_grad) and torch.is_grad_enabled():
        # Differentiable path: ``out=`` kernels do not record autograd, so
        # build the rotated pairs as plain expressions (identical values).
        # Also required when only ``positions`` carries a gradient (the
        # inference path's ``out=`` kernel would reject it outright).
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        out = torch.empty(x.shape, dtype=x.dtype, device=device)
        if rope_dim < D:
            out[..., : D - rope_dim] = x[..., : D - rope_dim]
        out[..., D - rope_dim::2] = o1
        out[..., D - rope_dim + 1::2] = o2
        return out
    # Inference path. Write the rotated pairs directly into the output slices
    # with ``out=`` kernels: the per-element operations (and their order) are
    # unchanged.
    out = torch.empty(x.shape, dtype=x.dtype, device=device)
    if rope_dim < D:
        out[..., : D - rope_dim] = x[..., : D - rope_dim]
    o1 = out[..., D - rope_dim::2]                           # [..., P, r/2]
    o2 = out[..., D - rope_dim + 1::2]                       # [..., P, r/2]
    tmp = torch.empty(x1.shape, dtype=x.dtype, device=device)
    torch.mul(x2, sin, out=tmp)
    torch.mul(x1, cos, out=o1)
    o1.sub_(tmp)
    torch.mul(x2, cos, out=tmp)
    torch.mul(x1, sin, out=o2)
    o2.add_(tmp)
    return out


def token_positions(T: int, device, dtype=torch.long) -> torch.Tensor:
    """Positions 0..T-1 for query / sliding-window tokens.

    The returned tensor is cached and shared; callers must treat it as
    read-only (all in-repo consumers do).
    """
    key = ('tok', T, str(device), str(dtype))
    pos = _POSITION_VEC_CACHE.get(key)
    if pos is None:
        # Ordinary tensor even when the call happens under
        # ``torch.inference_mode()`` (see :func:`_normal_tensor_scope`):
        # cached inference tensors have no version counter, which the
        # long-context identity-table cache key requires.
        with _normal_tensor_scope():
            pos = torch.arange(T, device=device, dtype=dtype)
        if len(_POSITION_VEC_CACHE) >= 64:
            _POSITION_VEC_CACHE.clear()
        _POSITION_VEC_CACHE[key] = pos
    return pos


def block_positions(n_blocks: int, m: int, device,
                    dtype=torch.long) -> torch.Tensor:
    """Anchor positions ``s * m`` of the ``n_blocks`` compressed entries.

    The returned tensor is cached and shared; callers must treat it as
    read-only (all in-repo consumers do).
    """
    key = ('blk', n_blocks, m, str(device), str(dtype))
    pos = _POSITION_VEC_CACHE.get(key)
    if pos is None:
        # Ordinary tensor even under inference mode (see
        # :func:`_normal_tensor_scope`).
        with _normal_tensor_scope():
            pos = torch.arange(n_blocks, device=device, dtype=dtype) * m
        if len(_POSITION_VEC_CACHE) >= 64:
            _POSITION_VEC_CACHE.clear()
        _POSITION_VEC_CACHE[key] = pos
    return pos


def inverse_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    base: float = 10000.0,
    dim: int = 1,
) -> torch.Tensor:
    """Inverse-rotate at the given positions (rotate by the negated angle).

    Used for the core-attention OUTPUT: the output rows are rotated by
    ``-positions`` so that each KV entry's contribution carries the
    RELATIVE position (entry position − query position), matching the
    DeepSeek-V4 §2.3.3 countermeasure for absolute-position leakage.
    """
    neg = -positions if positions.is_floating_point() else -positions.to(torch.long)
    return partial_rope(x, neg, rope_dim, base, dim=dim)
