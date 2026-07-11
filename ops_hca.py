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
    W_KV: torch.Tensor,            # [c, d]   (nn.Linear.weight layout: [out, in])
    W_Z: torch.Tensor,             # [c, d]
    B_pos: torch.Tensor,           # [m2, c]
    W_DQ: torch.Tensor,            # [dc, d]
    W_UQ: torch.Tensor,            # [c*nh, dc]
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

    **Weight layout** (P0 API fix): all ``W_*`` tensors follow the
    ``nn.Linear.weight`` convention — shape ``[out_features, in_features]``.
    Internally we use ``F.linear(x, W)`` (which computes ``x @ W.T``) instead
    of the previous ``x @ W`` form that required callers to pass
    ``self.W_KV.weight.T``. Callers now pass ``self.W_KV.weight`` directly.

    ``T`` does NOT need to be divisible by ``m2``: the function right-pads
    the sequence with zeros up to the next multiple of ``m2`` and trims the
    output back to the original length, mirroring the contract of
    ``naive_chunk_kda`` and ``naive_csa``. Real tokens keep their original
    positions; only the last partial block contains padding zeros, and the
    causal block mask ensures no real token attends to it.
    """
    B_, T, d = H.shape
    # Validate structural params early so a caller passing m2=0, nh=0, etc.
    # gets a clear ValueError instead of a cryptic ZeroDivisionError or
    # IndexError deep inside the operator. Mirrors naive_csa's validation.
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1``. ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic crashes these guards are specifically meant to prevent.
    # ``raise ValueError`` is the standard exception for invalid user input (the previous
    # tests in ``run_correctness.py::test_csa_hca_input_validation`` expect
    # (they now catch ``ValueError``; the test was updated to accept both ValueError and AssertionError for backward compatibility with any external callers that may still catch AssertionError).
    if m2 < 1:
        raise ValueError(f"heavy compression factor m2={m2} must be >= 1")
    if nh < 1:
        raise ValueError(f"nh={nh} must be >= 1")
    if c < 1:
        raise ValueError(f"c={c} must be >= 1")
    if dc < 1:
        raise ValueError(f"dc={dc} must be >= 1")
    # ``sliding_window`` is gated by ``if sliding_window > 0`` below, so a
    # negative value silently skips the SW branch (looking like the caller
    # intentionally disabled it). A negative window is never a meaningful
    # configuration — reject it so the caller learns about the typo instead
    # of getting a model with no local-attention branch. Mirrors the
    # validation added to ``naive_csa``.
    if sliding_window < 0:
        raise ValueError(
            f"sliding_window={sliding_window} must be >= 0 "
            f"(0 disables the branch)")
    # Cosine-attention scale: when both ``q`` and ``C_comp`` are L2-normalized
    # (see ``F.normalize`` calls below), their dot product is already a cosine
    # similarity in ``[-1, 1]``. The previous default ``scale = c ** -0.5``
    # further shrinks the scores into a narrow band, making softmax over the
    # compressed blocks nearly uniform — effectively turning dense attention
    # into average pooling. Standard cosine-attention uses ``τ = 1``. The extra
    # ``1/sqrt(c)`` was a leftover from un-normalized softmax-attention.
    if scale is None:
        scale = 1.0
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
    # hit a bare ``ValueError`` with no message.
    original_T = T
    pad = (-T) % m2
    if pad:
        H = F.pad(H, (0, 0, 0, pad))
        T = T + pad
    n_blocks = T // m2

    # --- 1. Heavy KV compression (single branch, no overlap) ---
    # P0 API fix: use F.linear with W in nn.Linear.weight layout [out, in].
    C = F.linear(H, W_KV)                                          # [B, T, c]
    Z = F.linear(H, W_Z)                                           # [B, T, c]
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
    cQ = F.linear(H, W_DQ)                                         # [B, T, dc]
    q = F.linear(cQ, W_UQ).view(B_, T, nh, c).to(compute_dtype)    # [B, T, nh, c]
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
        # P5 fix — TRUE O(T·win) sliding-window attention (was O(T²)).
        #
        # The previous implementation built a full ``[T, T]`` boolean mask
        # (``win_mask``) and a full ``[B, nh, T, T]`` attention-scores tensor,
        # then masked every entry outside the window to ``-inf`` before
        # softmax. Even though only ``win`` entries per row were non-trivial,
        # the dense matmul (``einsum('bthd,bnd->bhtn')``) and the dense
        # softmax both did ``O(T²·nh·c)`` work — the window size ``win`` had
        # NO effect on the compute cost. At ``T=2048`` this allocated and
        # filled a 4M-entry scores tensor per call regardless of ``win``,
        # defeating the whole purpose of a local-attention mechanism.
        #
        # The fix uses a banded / windowed-gather approach: left-pad
        # ``C_local`` with ``win-1`` zero columns, use ``unfold`` to extract
        # per-query windows of shape ``[B, T, win, c]``, compute scores ONLY
        # over the ``win`` entries (``[B, T, nh, win]``), mask the left-edge
        # padding slots to ``-inf``, softmax, and weighted-sum over the
        # ``win`` dimension. This is ``O(T·win·nh·c)`` — the window size now
        # genuinely controls the cost.
        #
        # Numerically identical to the old dense+mask approach (verified by
        # ``test_hca_sliding_window_causality`` in run_correctness.py):
        # softmax over the ``win`` non-masked entries of a row is the same
        # whether the masked entries are materialized as ``-inf`` in a
        # ``[T,T]`` tensor or absent from a ``[T,win]`` tensor.
        #
        # Dtype: cast C to compute_dtype so the SW branch matches the dense
        # branch's precision. Without this, the SW softmax runs in H.dtype
        # (e.g. fp16) while the dense softmax ran in compute_dtype (fp32) —
        # an asymmetric precision loss that silently degrades the SW branch's
        # contribution for fp16 inputs. Mirrors the fix in ops_csa.py.
        C_local = F.normalize(C.to(compute_dtype), dim=-1)              # [B, T, c]
        # Left-pad the key dimension with (win-1) zero columns. After padding,
        # padded position ``p`` maps to original position ``p - (win-1)``.
        # The window for query ``t`` is padded positions ``[t, t+win-1]``,
        # which map to original positions ``[t-win+1, t]`` — the causal
        # window. For ``t < win-1`` the first ``win-1-t`` entries are
        # zero-padding (masked below).
        C_padded = F.pad(C_local, (0, 0, win - 1, 0))            # [B, T+win-1, c]
        # unfold(dim=1, size=win, step=1) extracts T overlapping windows of
        # length win. Result shape ``[B, T, c, win]``; permute to
        # ``[B, T, win, c]`` for the einsum.
        C_windows = C_padded.unfold(1, win, 1).permute(0, 1, 3, 2)  # [B, T, win, c]
        # Validity mask: window slot j for query t is a real (non-padding)
        # position iff ``t - win + 1 + j >= 0``, i.e. ``j >= win-1-t``.
        # For t >= win-1 every slot is valid. Shape ``[T, win]``.
        # NOTE: every query t always has itself in the window (slot j=win-1
        # maps to original position t, always valid), so no row is fully
        # -inf and softmax is NaN-free.
        _j = torch.arange(win, device=device)
        _t = torch.arange(T, device=device)
        valid_mask = _j[None, :] >= (win - 1 - _t[:, None])      # [T, win]
        scores = torch.einsum('b t h d, b t w d -> b t h w', q, C_windows) * scale
        scores = scores.masked_fill(~valid_mask[None, :, None, :], float('-inf'))
        p = torch.softmax(scores, dim=-1)                        # [B, T, nh, win]
        sw_out = torch.einsum('b t h w, b t w d -> b t h d', p, C_windows)
        out = out + sw_out

    # Return the raw per-head core-attention output [B, T, nh, c] flattened to
    # [B, T, nh*c]; the caller performs the grouped output projection.
    # Trim the padded SUFFIX off the SEQUENCE axis (dim=1) so the output
    # matches the input's original T (right-padding added zeros at the end,
    # which never affect real-token outputs thanks to the causal block mask).
    return out.reshape(B_, T, nh * c).to(H.dtype)[:, :original_T]
