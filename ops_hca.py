"""Heavily Compressed Attention (HCA) — naive PyTorch reference.

Implements the HCA operator from DeepSeek-V4 (arXiv:2606.19348v1, §2.3.2):

    1. Heavier KV compression: every ``m'`` (>> m) consecutive KV entries are
       consolidated into one (no overlap, single branch) — Eq. (20)–(23).
    2. Dense (not sparse) shared-KV MQA over the compressed entries.
    3. A small sliding-window branch keeps local fine-grained dependencies.
    4. Optional attention sink (learnable per-head logit in the softmax denom).

HCA trades recall granularity for extreme compression, complementing CSA's
sparse selection: where CSA keeps ``k`` of ``n/m`` entries, HCA keeps *all*
``n/m'`` heavily-compressed entries, with ``m'`` typically an order of
magnitude larger than ``m``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ops_csa import csa_compress_kv, _causal_block_mask


def naive_hca(
    H: torch.Tensor,               # [B, T, d]
    W_KV: torch.Tensor,            # [d, c]
    W_Z: torch.Tensor,             # [d, c]
    B_pos: torch.Tensor,           # [m2, c]
    W_DQ: torch.Tensor,            # [d, dc]
    W_UQ: torch.Tensor,            # [dc, c*nh]
    *,
    m2: int,                       # heavy compression factor (m' in the paper)
    nh: int,
    c: int,
    dc: int,
    scale: float | None = None,
    sliding_window: int = 0,
    sink_logits: torch.Tensor | None = None,    # [nh]
) -> torch.Tensor:
    """Full HCA forward (heavy compression + dense MQA + optional SW + sink).

    ``T`` does NOT need to be divisible by ``m2``: the function right-pads
    the sequence with zeros up to the next multiple of ``m2`` and trims the
    output back to the original length, mirroring the contract of
    ``naive_chunk_kda`` and ``naive_csa``. Real tokens keep their original
    positions; only the last partial block contains padding zeros, and the
    causal block mask ensures no real token attends to it.
    """
    B_, T, d = H.shape
    if scale is None:
        scale = c ** -0.5
    device = H.device
    # Degenerate case: empty sequence. Without this guard the downstream
    # ``csa_compress_kv`` would raise a cryptic broadcasting error
    # (``Expected size 0 but got size 1``) because n_blocks=0 makes the
    # [B, n_blocks, m2, c] reshape collapse against [m2, c] positional bias.
    # Return a zero-shaped output matching the contract. Mirrors the guard
    # in ``ops_csa.py::naive_csa``.
    if T == 0:
        return torch.zeros(B_, 0, nh * c, dtype=H.dtype, device=device)
    # Right-pad T up to a multiple of m2 so callers don't have to. Real
    # tokens keep their original positions; only the last partial block
    # contains padding zeros, and no real token attends to it (causal block
    # mask). This removes a footgun where direct callers (without the
    # external padding done by ``HybridKCHAttention`` or ``HCAAttn``) would
    # hit a bare ``AssertionError`` with no message.
    original_T = T
    pad = (-T) % m2
    if pad:
        H = F.pad(H, (0, 0, 0, pad))
        T = T + pad
    n_blocks = T // m2

    # --- 1. Heavy KV compression (single branch, no overlap) ---
    C = H @ W_KV                                                   # [B, T, c]
    Z = H @ W_Z                                                    # [B, T, c]
    C_comp = csa_compress_kv(C, Z, B_pos, m2)                     # [B, n_blocks, c] in compute_dtype
    C_comp_n = F.normalize(C_comp, dim=-1)

    # --- 2. Dense shared-KV MQA ---
    # Dtype consistency: ``C_comp`` is returned by ``csa_compress_kv`` in
    # ``compute_dtype`` (fp32 for fp16 inputs, fp64 for fp64 inputs). The
    # downstream ``scores`` einsum mixes ``q`` (H.dtype) with ``C_comp_n``
    # (compute_dtype); ``torch.einsum`` does NOT auto-promote mixed dtypes and
    # raises ``RuntimeError`` for fp16/bf16 inputs. We cast ``q`` to
    # ``compute_dtype`` before normalization so the entire attention core runs
    # in one consistent precision. Mirrors the fix in ``ops_csa.py::naive_csa``.
    compute_dtype = torch.float64 if H.dtype == torch.float64 else torch.float
    cQ = H @ W_DQ                                                  # [B, T, dc]
    q = (cQ @ W_UQ).view(B_, T, nh, c).to(compute_dtype)           # [B, T, nh, c]
    q = F.normalize(q, dim=-1)

    # Causal block mask: query t attends ONLY to blocks strictly before
    # floor(t/m2) — i.e. blocks that contain only past tokens. The block
    # containing t is NOT included because its compressed representation
    # aggregates all m2 tokens in the block (including t itself and any
    # later tokens in the same block), so attending to it would leak future
    # information. The sliding-window branch handles intra-block and
    # near-context attention separately.
    cbm = _causal_block_mask(T, n_blocks, m2, device)

    # Precompute all (B, T, n_blocks) attention logits at once (fully vectorized).
    scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * scale
    scores = scores.masked_fill(~cbm[None, None], float('-inf'))
    if sink_logits is not None:
        # Attention sink: a per-head constant added to the denominator.
        # Numerically stable logsumexp approach — keep sink_logits in
        # log space (never exp it, which could overflow to inf when the
        # learnable parameter grows during training and makes denom=inf,
        # p=0/inf=nan or inf/inf=nan). Mirrors the fix in ops_csa.py.
        #
        # The sink MUST be shifted by -row_max along with the scores:
        #   p_i = exp(s_i - M) / (sum_j exp(s_j - M) + exp(sink - M))
        # Without the shift the sink is over-weighted by exp(M), a
        # systematic ~13% bias in the default c=64 config. See the
        # detailed comment in ops_csa.py::naive_csa for the algebra.
        log_sink = sink_logits.view(1, nh, 1, 1).to(scores)  # [1, nh, 1, 1]
        # scores already carries -inf at causally-masked slots, so
        # logsumexp/exp naturally yield 0 there; fully-masked rows also
        # collapse to p=0 (logaddexp(-inf, log_sink) = log_sink,
        # exp(-inf - log_sink) = 0) — PROVIDED log_sink is finite. If
        # log_sink is also -inf (e.g. sink_logits diverged to -inf
        # during training), then (shifted - log_denom) = (-inf - (-inf))
        # = NaN. The all_masked guard below zeros out such rows so the
        # downstream einsum produces 0 instead of NaN. Mirrors the guard
        # in the ``else`` branch and in ``ops_csa.py::naive_csa``.
        row_max = scores.amax(-1, keepdim=True).clamp(min=0)        # [B, nh, T, 1]
        shifted = scores - row_max                                  # [B, nh, T, n_blocks]
        shifted_sink = log_sink - row_max                           # [B, nh, T, 1]
        lse = torch.logsumexp(shifted, dim=-1, keepdim=True)
        log_denom = torch.logaddexp(lse, shifted_sink)              # [B, nh, T, 1]
        p = (shifted - log_denom).exp()                            # [B, nh, T, n_blocks]
        # NaN guard: zero out rows where every block is causally masked
        # (e.g. t < m2). Without this, a -inf log_sink would produce NaN
        # via (-inf - (-inf)) = NaN, and the einsum would propagate it.
        all_masked = torch.isinf(scores).all(-1, keepdim=True)   # [B, nh, T, 1]
        p = p.masked_fill(all_masked, 0.0)
    else:
        # Rows with no valid block to attend to (e.g. t < m2 under the
        # causal block mask) are entirely -inf; softmax over them would
        # yield NaN. Detect such rows and force their weights to 0 so the
        # contribution is 0 instead of NaN.
        all_masked = torch.isinf(scores).all(-1, keepdim=True)   # [B, nh, T, 1]
        safe = scores.masked_fill(all_masked, 0.0)
        p = torch.softmax(safe, dim=-1)
        p = p.masked_fill(all_masked, 0.0)
    out = torch.einsum('b h t n, b n d -> b t h d', p, C_comp_n)   # [B, T, nh, c]

    # --- 3. Sliding window branch (uncompressed local KV) ---
    if sliding_window > 0:
        win = sliding_window
        # Position-relative quantities depend only on T (not on batch), so we
        # build them once outside the batch loop instead of recreating them
        # per-batch like the original reference implementation did.
        i_t = torch.arange(T, device=device)
        # dist[t, n] = n - t; query t attends to positions with -win < n-t <= 0
        # i.e. the causal window [t-win+1, t] (past + current only).
        dist = i_t[None, :] - i_t[:, None]                       # [T, T]
        win_mask = (dist <= 0) & (dist > -win)                   # [T, T]
        # Fully vectorized batched shared-KV MQA over the local window.
        # q:        [B, T, nh, c]
        # C_local:  [B, T, c]
        # scores:   [B, nh, T, T]  (b h t n)
        # win_mask broadcasts from [1, 1, T, T] -> [B, nh, T, T].
        # NOTE: every query t always has itself in the window (dist=0 satisfies
        # the mask for win >= 1), so no row is fully -inf and softmax is NaN-free.
        #
        # Dtype: cast C to compute_dtype so the SW branch matches the dense
        # branch's precision. Without this, the SW softmax runs in H.dtype
        # (e.g. fp16) while the dense softmax ran in compute_dtype (fp32) —
        # an asymmetric precision loss that silently degrades the SW branch's
        # contribution for fp16 inputs. Mirrors the fix in ops_csa.py.
        C_local = F.normalize(C.to(compute_dtype), dim=-1)              # [B, T, c]
        scores = torch.einsum('b t h d, b n d -> b h t n', q, C_local) * scale
        scores = scores.masked_fill(~win_mask[None, None], float('-inf'))
        p = torch.softmax(scores, dim=-1)
        sw_out = torch.einsum('b h t n, b n d -> b t h d', p, C_local)
        out = out + sw_out

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    # Trim the padded SUFFIX off the SEQUENCE axis (dim=1) so the output
    # matches the input's original T (right-padding added zeros at the end,
    # which never affect real-token outputs thanks to the causal block mask).
    return out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
