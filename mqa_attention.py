"""Dense multi-query / grouped-query attention reference (Exp 14).

Why this operator?
------------------
The softmax rungs of the repo's baseline ladder are full multi-head
attention: ``run_quality.SoftmaxAttn`` and
``run_decoding.SoftmaxAttnDecoding`` allocate one K/V head per query head,
so their decode KV cache grows as ``n_heads``. The KDA operator has its own
sharing scheme in the other direction (GVA: q/k shared across ``G = HV/H``
value heads), but neither ladder rung covers *dense multi-query attention*
(Shazeer 2019) or its grouped middle ground (GQA, Ainslie et al. 2023),
where ``n_kv_heads <= n_heads`` query heads share one K/V head.

That omission matters for this repository's central trade-off: the hybrid
memory routes (KDA / Mamba-2 / RWKV-7) claim an ``O(1)``-in-context decode
state, and the cheapest way to make a *softmax* baseline competitive on
decode traffic is exactly KV-head sharing. Exp 14 needs a trustworthy
reference for that comparison, so this module implements the operator and
pins its two defining identities:

* **Grouped execution == expanded MHA.** The primary forward groups query
  heads with an einsum over ``[B, T, G, Hkv, D]``; :meth:`MQAAttn.to_mha`
  expands the K/V projection rows by ``repeat_interleave`` into a full
  MHA layer. The two must agree (this is the classic GQA indexing bug
  detector).
* **Incremental decode == full-sequence last position.** The append-only
  cache path (used for token-by-token decode) must reproduce the
  full-sequence forward.

Accounting
----------
:func:`attention_parameter_elements`, :func:`attention_prefill_flops`,
:func:`kv_state_elements_per_token` and :func:`kv_bytes_reduction` price the
operator. Head sharing leaves prefill FLOPs unchanged at fixed
``(n_heads, head_dim)`` (every query still attends over the same entries)
but divides the decode KV traffic by ``n_heads / n_kv_heads``; keeping both
facts in one module is what makes the Exp 14 roofline table honest.

Honesty notes
-------------
* This is a **reference** implementation: the forward materializes the
  ``[B, T, T]`` score matrix, has no dropout, and is not meant to be fast.
* The mask is applied on every call (including a trivial one-row decode
  mask) so there is exactly ONE code path; no cache-shape special case can
  drift from the full-sequence path.
* Follows the repo's fp32 accounting (``itemsize=4``) unless overridden.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

FP32_BYTES = 4


def _require_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}={value!r} must be a positive int.")
    return value


@dataclass(frozen=True)
class MQACache:
    """Append-only K/V cache for incremental decode.

    Shapes: ``k``/``v`` are ``[B, S, n_kv_heads, head_dim]`` with ``S``
    the number of cached source positions (``0`` only before the first
    appended token; every real call appends at least one token).
    """

    k: torch.Tensor
    v: torch.Tensor

    @property
    def seen(self) -> int:
        return int(self.k.shape[1])

    @property
    def batch(self) -> int:
        return int(self.k.shape[0])

    @property
    def device(self):
        return self.k.device

    @property
    def dtype(self):
        return self.k.dtype


def expand_kv_heads(k: torch.Tensor, v: torch.Tensor, *, n_heads: int,
                    n_kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand shared K/V heads to full query-head count (the MHA oracle).

    ``k``/``v`` have shape ``[..., n_kv_heads, head_dim]``; the returned
    tensors have shape ``[..., n_heads, head_dim]`` where head ``h`` copies
    the K/V head ``h // group`` (``group = n_heads // n_kv_heads``) — the
    same indexing used for the weight expansion in :meth:`MQAAttn.to_mha`.
    This materialized expansion is the independent oracle the grouped
    einsum path is checked against in the tests.
    """
    n_heads = _require_positive_int('n_heads', n_heads)
    n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must be divisible by n_kv_heads="
            f"{n_kv_heads}.")
    group = n_heads // n_kv_heads
    if k.shape[-2] != n_kv_heads or v.shape[-2] != n_kv_heads:
        raise ValueError(
            f"k/v head axes must be {n_kv_heads}; got {tuple(k.shape)} / "
            f"{tuple(v.shape)}.")
    return (k.repeat_interleave(group, dim=-2),
            v.repeat_interleave(group, dim=-2))


class MQAAttn(nn.Module):
    """Causal reference attention with ``n_kv_heads`` shared K/V heads.

    ``n_kv_heads == n_heads`` is plain MHA, ``n_kv_heads == 1`` is MQA, and
    any divisor in between is GQA. ``head_dim`` is shared by Q and K/V, so
    the layer geometry is exactly ``(d_model, n_heads, n_kv_heads,
    head_dim)``; ``n_heads * head_dim`` may differ from ``d_model`` (the
    projections carry the dimension change, as in the rest of the repo).
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 n_kv_heads: int | None = None, head_dim: int = 8,
                 *, bias: bool = False):
        super().__init__()
        self.d_model = _require_positive_int('d_model', d_model)
        self.n_heads = _require_positive_int('n_heads', n_heads)
        if n_kv_heads is None:
            n_kv_heads = n_heads
        self.n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads="
                f"{self.n_kv_heads} (query heads per shared K/V head must "
                f"be a whole number).")
        self.head_dim = _require_positive_int('head_dim', head_dim)
        if not isinstance(bias, bool):
            raise TypeError(f"bias={bias!r} must be a bool.")
        self.use_bias = bias
        self.q = nn.Linear(d_model, self.n_heads * head_dim, bias=bias)
        self.k = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=bias)
        self.v = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=bias)
        self.o = nn.Linear(self.n_heads * head_dim, d_model, bias=bias)
        self.scale = head_dim ** -0.5

    @property
    def group_size(self) -> int:
        """Query heads sharing each K/V head (``n_heads // n_kv_heads``)."""
        return self.n_heads // self.n_kv_heads

    def forward_incremental(self, x: torch.Tensor,
                            cache: MQACache | None = None,
                            ) -> tuple[torch.Tensor, MQACache]:
        """Causal forward, optionally appending to an ``MQACache``.

        With ``cache=None`` this is the full-sequence prefill/training
        forward. With a cache it appends ``T`` new tokens and lets them
        attend to the cached prefix plus each other (causal within the
        new block), returning the updated cache. One code path for both
        roles — the mask is always applied.
        """
        if x.dim() != 3:
            raise ValueError(
                f"MQAAttn: x must be [B, T, d_model]; got shape "
                f"{tuple(x.shape)}.")
        B, T, d = x.shape
        if d != self.d_model:
            raise ValueError(
                f"MQAAttn: x last dim {d} != d_model {self.d_model}.")
        if T < 1:
            raise ValueError(f"MQAAttn: T must be >= 1, got {T}.")
        q = self.q(x).view(B, T, self.n_heads, self.head_dim)
        k_new = self.k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v_new = self.v(x).view(B, T, self.n_kv_heads, self.head_dim)
        if cache is None:
            k_all, v_all = k_new, v_new
            seen = 0
        else:
            if not isinstance(cache, MQACache):
                raise TypeError(
                    f"MQAAttn: cache must be an MQACache or None, got "
                    f"{type(cache).__name__}.")
            if cache.batch != B:
                raise ValueError(
                    f"MQAAttn: cache batch {cache.batch} != x batch {B}.")
            if cache.k.shape[-2:] != (self.n_kv_heads, self.head_dim):
                raise ValueError(
                    f"MQAAttn: cache K head shape {tuple(cache.k.shape[-2:])} "
                    f"!= ({self.n_kv_heads}, {self.head_dim}).")
            k_all = torch.cat([cache.k, k_new], dim=1)
            v_all = torch.cat([cache.v, v_new], dim=1)
            seen = cache.seen
        S = k_all.shape[1]
        # Grouped scores: [B, G, Hkv, T, S]. Query heads are grouped
        # CONTIGUOUSLY (the standard GQA layout): head ``h*G + g`` uses
        # K/V head ``h``, so `view(Hkv, G, D)` + repeat_interleave in
        # to_mha() describes the same sharing. Scores are indexed
        # ``bghts`` with h the K/V-head axis and g the within-group axis.
        q_grouped = q.view(B, T, self.n_kv_heads, self.group_size,
                           self.head_dim)
        scores = torch.einsum('bthgd,bshd->bghts', q_grouped, k_all)
        scores = scores * self.scale
        # Strictly causal: query at absolute position seen+t sees s <= seen+t.
        src = torch.arange(S, device=x.device)
        tgt = seen + torch.arange(T, device=x.device)
        blocked = src.unsqueeze(0) > tgt.unsqueeze(1)
        scores = scores.masked_fill(
            blocked.view(1, 1, 1, T, S), float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum('bghts,bshd->bthgd', probs, v_all)
        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.o(out), MQACache(k=k_all, v=v_all)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence causal forward (training / quality probes)."""
        return self.forward_incremental(x, None)[0]

    def to_mha(self) -> 'MQAAttn':
        """Return the equivalent full-MHA layer with expanded K/V weights.

        Writes the grouped layer's behavior into an ``n_kv_heads ==
        n_heads`` layer: q/o weights are copied, and each K/V output block
        of ``head_dim`` rows is repeated ``group_size`` times so expanded
        head ``h`` carries the weight of source head ``h // group``.
        This is the equivalence certificate used by the tests.
        """
        mha = MQAAttn(self.d_model, self.n_heads, self.n_heads,
                      self.head_dim, bias=self.use_bias).to(
            self.q.weight.device, self.q.weight.dtype)
        group = self.group_size
        with torch.no_grad():
            mha.q.weight.copy_(self.q.weight)
            mha.o.weight.copy_(self.o.weight)
            for src, dst in ((self.k, mha.k), (self.v, mha.v)):
                expanded = src.weight.view(
                    self.n_kv_heads, self.head_dim, self.d_model
                ).repeat_interleave(group, dim=0).reshape(
                    self.n_heads * self.head_dim, self.d_model)
                dst.weight.copy_(expanded)
                if self.use_bias:
                    dst.bias.copy_(src.bias.view(
                        self.n_kv_heads, self.head_dim
                    ).repeat_interleave(group, dim=0).reshape(-1))
            if self.use_bias:
                mha.q.bias.copy_(self.q.bias)
                mha.o.bias.copy_(self.o.bias)
        return mha


def attention_parameter_elements(d_model: int, n_heads: int,
                                 n_kv_heads: int, head_dim: int,
                                 *, bias: bool = False) -> int:
    """Number of trainable elements in an :class:`MQAAttn` layer."""
    d_model = _require_positive_int('d_model', d_model)
    n_heads = _require_positive_int('n_heads', n_heads)
    n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
    head_dim = _require_positive_int('head_dim', head_dim)
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must be divisible by n_kv_heads="
            f"{n_kv_heads}.")
    total = (d_model * n_heads * head_dim            # q
             + 2 * d_model * n_kv_heads * head_dim   # k, v
             + n_heads * head_dim * d_model)         # o
    if bias:
        total += n_heads * head_dim + n_kv_heads * head_dim
        total += n_kv_heads * head_dim + d_model
    return int(total)


def attention_prefill_flops(T: int, d_model: int, n_heads: int,
                            n_kv_heads: int, head_dim: int) -> float:
    """Forward FLOPs (2 * MACs) for a causal :class:`MQAAttn` forward.

    Core: ``2 * (T(T+1)/2) * n_heads * (head_dim + head_dim)`` — head
    sharing does NOT change the query-side work at fixed ``n_heads``.
    Projections: q (``d -> n_heads*head_dim``), k/v (``d ->
    n_kv_heads*head_dim``), o (``n_heads*head_dim -> d``). Same convention
    as ``run_kv_cache.prefill_flops``.
    """
    if isinstance(T, bool) or not isinstance(T, int) or T < 0:
        raise ValueError(f"T={T!r} must be a non-negative int.")
    d_model = _require_positive_int('d_model', d_model)
    n_heads = _require_positive_int('n_heads', n_heads)
    n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
    head_dim = _require_positive_int('head_dim', head_dim)
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must be divisible by n_kv_heads="
            f"{n_kv_heads}.")
    entries = T * (T + 1) // 2
    core = 2.0 * entries * n_heads * (head_dim + head_dim)
    proj = 2.0 * T * d_model * (n_heads * head_dim
                                + 2 * n_kv_heads * head_dim)
    out_proj = 2.0 * T * (n_heads * head_dim) * d_model
    return core + proj + out_proj


def kv_state_elements_per_token(n_kv_heads: int, head_dim: int) -> int:
    """K + V elements appended to the decode cache by one new token."""
    n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
    head_dim = _require_positive_int('head_dim', head_dim)
    return 2 * n_kv_heads * head_dim


def attention_kv_read_elements(T: int, n_kv_heads: int, head_dim: int,
                               *, batch: int = 1) -> int:
    """Compulsory decode KV reads for one new token at context ``T``.

    Every cached K/V entry is read once (attention over the whole prefix);
    with a single query token there is no reuse to exploit across the
    sequence, so reads = ``batch * T * kv_state_elements_per_token``.
    """
    if isinstance(T, bool) or not isinstance(T, int) or T < 0:
        raise ValueError(f"T={T!r} must be a non-negative int.")
    batch = _require_positive_int('batch', batch)
    return int(batch * T * kv_state_elements_per_token(n_kv_heads, head_dim))


def kv_bytes_reduction(n_heads: int, n_kv_heads: int) -> float:
    """KV-cache traffic / parameter reduction factor ``n_heads / n_kv_heads``.

    Applies to both the retained cache size and the per-token decode read
    (weights and activations, not just the cache). FLOPs are unchanged.
    """
    n_heads = _require_positive_int('n_heads', n_heads)
    n_kv_heads = _require_positive_int('n_kv_heads', n_kv_heads)
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads={n_heads} must be divisible by n_kv_heads="
            f"{n_kv_heads}.")
    return float(n_heads // n_kv_heads)
