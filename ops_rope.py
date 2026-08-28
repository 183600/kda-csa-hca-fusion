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
"""

from __future__ import annotations

import torch


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


def partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    base: float = 10000.0,
    dim: int = -2,
) -> torch.Tensor:
    """Rotate the LAST ``rope_dim`` feature dims of ``x`` at given positions.

    Parameters
    ----------
    x : torch.Tensor
        ``[..., P, D]`` along ``dim`` — e.g. queries ``[B, T, nh, c]`` use
        ``dim=1`` (the sequence axis is T); compressed KV ``[B, n_blocks, c]``
        use ``dim=1`` (== ``dim=-2``).
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
        The sequence axis along which ``positions`` applies (default -2;
        must not be the last axis, which is the feature axis).

    Returns
    -------
    torch.Tensor
        Same shape / dtype as ``x`` with the last ``rope_dim`` feature dims
        rotated. The rotation is orthogonal (norm-preserving), so L2 norms
        computed before/after agree up to floating-point rounding.
    """
    if rope_dim is None or rope_dim <= 0:
        return x
    if dim == -1 or dim == x.dim() - 1:
        raise ValueError(
            f"partial_rope: dim={dim} cannot be the feature (last) axis")
    if dim < 0:
        dim = x.dim() + dim
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
    # inv_freq_j = base^(-2j/rope_dim), j = 0 .. rope_dim/2 - 1
    inv_freq = base ** (
        -torch.arange(0, rope_dim, 2, dtype=angle_dtype, device=device)
        / float(rope_dim)
    )                                                        # [rope_dim/2]
    pos = positions.to(device=device, dtype=angle_dtype)     # [P]
    angles = pos[:, None] * inv_freq[None, :]                # [P, rope_dim/2]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    # Broadcast cos/sin so that the position axis lands on ``dim`` and the
    # pair axis lands on the last axis: shape [1, ..., P, ..., 1, r/2].
    bc_shape = [1] * x.dim()
    bc_shape[dim] = P
    bc_shape[-1] = rope_dim // 2
    cos = cos.view(bc_shape)
    sin = sin.view(bc_shape)

    x_rot = x[..., D - rope_dim:]                            # [..., P, r]
    x1 = x_rot[..., 0::2]                                    # [..., P, r/2]
    x2 = x_rot[..., 1::2]                                    # [..., P, r/2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    rotated = torch.stack([o1, o2], dim=-1).flatten(-2)      # [..., P, r]
    if rope_dim == D:
        return rotated
    return torch.cat([x[..., : D - rope_dim], rotated], dim=-1)


def token_positions(T: int, device, dtype=torch.long) -> torch.Tensor:
    """Positions 0..T-1 for query / sliding-window tokens."""
    return torch.arange(T, device=device, dtype=dtype)


def block_positions(n_blocks: int, m: int, device,
                    dtype=torch.long) -> torch.Tensor:
    """Anchor positions ``s * m`` of the ``n_blocks`` compressed entries."""
    return torch.arange(n_blocks, device=device, dtype=dtype) * m


def inverse_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    base: float = 10000.0,
    dim: int = -2,
) -> torch.Tensor:
    """Inverse-rotate at the given positions (rotate by the negated angle).

    Used for the core-attention OUTPUT: the output rows are rotated by
    ``-positions`` so that each KV entry's contribution carries the
    RELATIVE position (entry position − query position), matching the
    DeepSeek-V4 §2.3.3 countermeasure for absolute-position leakage.
    """
    return partial_rope(x, -positions.to(torch.long), rope_dim, base, dim=dim)
