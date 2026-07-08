"""Method analysis module.

This module provides the theoretical backing that the reviewer flagged as
missing from the original paper:

  1. **3:1:1 ratio rationale.** A capacity/recall/global-context budget
     argument for why 3 KDA : 1 CSA : 1 HCA is a principled default, not just
     "inherited from Kimi Linear".

  2. **Headwise fusion sketch.** A small prototype of *headwise* fusion
     (mixing KDA and compressed-attention heads within a single layer) as a
     concrete future-work artifact, with a forward pass that can be benchmarked.

  3. **Complete CSA / HCA formulas.** The full mathematical formulas for both
     operators, in one place, for reproducibility.

  4. **Overlap-compression causality proof.** A short formal argument for why
     the two-branch overlapped compression cannot leak future tokens.

This module is importable (``from method_analysis import ...``) and also
runnable as a script to print a human-readable summary.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device
from ops_csa import csa_compress_kv, _causal_block_mask
from ops_kda import naive_recurrent_kda


# ---------------------------------------------------------------------------
# 1. 3:1:1 ratio rationale
# ---------------------------------------------------------------------------

RATIONALE_3_1_1 = """
Why 3:1:1? A budget argument.
================================

The tribrid interleaves three operators with complementary cost/ability
profiles. Let the per-unit budget be B layers (we use B=5). We want to
allocate layers to maximize a composite objective:

    J = alpha * (recall capacity) + beta * (global context) + gamma * (cheap mixing)

subject to:
  - recall capacity   is provided ONLY by CSA (top-k sparse retrieval);
  - global context    is provided cheapest by HCA (dense over heavily compressed);
  - cheap mixing      is provided by KDA (O(1) state, linear time);
  - at least 1 CSA layer is needed for any non-trivial recall;
  - at least 1 HCA layer is needed for any non-trivial global context;
  - KDA layers have a *capacity limit*: stacking >4 without a recall layer
    causes interference (empirically observed in the ablation, and consistent
    with the Kimi Linear finding that 3:1 KDA:MLA is a sweet spot).

Let n_kda, n_csa, n_hca be the allocation with n_kda+n_csa+n_hca = B.

Claim: (n_kda, n_csa, n_hca) = (3, 1, 1) is Pareto-optimal for B=5.

Proof sketch (by enumeration of the feasible frontier, B=5, n_csa>=1, n_hca>=1):

  * (5,0,0): infeasible (violates n_csa>=1, n_hca>=1). Without the
    feasibility constraints it would be "no recall, no global context",
    which is dominated.
  * (4,1,0): infeasible (violates n_hca>=1). The closest FEASIBLE
    allocation is (3,1,1) (5L, all three operators present). Adding a 4th
    KDA layer without a global-context branch (i.e. comparing (4,1,0) at
    5L vs (3,1,1) at 5L) loses HCA's global context for one more
    finite-state layer — a dominated trade.
    Note: our ablation includes 4:1:1 (4 KDA + 1 CSA + 1 HCA = 6L), which
    is NOT an equal-budget comparison to 3:1:1 (5L). The 4:1:1 result
    *underperforms* 3:1:1, but this is confounded by depth (6L vs 5L at a
    fixed step budget leaves 4:1:1 under-trained). The clean equal-budget
    comparison (4,1,0) vs (3,1,1) is not in the ablation set; the
    theoretical argument above is what supports the claim.
  * (4,0,1): infeasible (violates n_csa>=1). Closest feasible is (3,1,1):
    one fewer KDA layer buys the recall branch. Dominated.
  * (3,1,1): recall + global context + 3 cheap mixing layers. This is the
    minimal feasible allocation that has all three capabilities.
  * (2,2,1) or (2,1,2): more recall/global but fewer cheap mixing layers.
    These trade O(1)-state KDA layers for O(T/m)-state CSA/HCA layers,
    increasing KV cache without a proven quality gain at small scale.
  * (1,2,2), (1,1,3), (1,3,1), etc.: dominated — KDA is the cheapest
    layer, removing it inflates cost. The remaining allocations on the
    frontier (n_kda<3, n_csa+n_hca>=4) trade more cheap layers for more
    expensive ones without a quality gain at small scale.

So (3,1,1) is the knee of the Pareto frontier: it is the allocation with the
*most* KDA layers (cheapest) subject to having at least one recall (CSA) and
one global-context (HCA) layer. This is exactly the "3:1" logic of Kimi Linear
(3 cheap + 1 expensive), extended with a second expensive-but-different
operator.

The argument does NOT prove 3:1:1 is globally optimal at production scale —
that requires the large-scale ablation we flag as future work. But it does
show 3:1:1 is the principled default given the design constraints.
"""


def print_rationale():
    print(RATIONALE_3_1_1)


# ---------------------------------------------------------------------------
# 2. Headwise fusion sketch
# ---------------------------------------------------------------------------

@dataclass
class HeadwiseConfig:
    """Configuration for a single headwise-fused layer.

    Instead of dedicating whole layers to one operator, headwise fusion splits
    the H heads of a single layer into three groups:
      - H_kda heads use the KDA delta recurrence;
      - H_csa heads use CSA compressed sparse attention;
      - H_hca heads use HCA heavily-compressed dense attention.
    All three run in parallel on the same hidden state, and their outputs are
    concatenated along the head dimension.
    """
    d_model: int = 256
    H_total: int = 6
    H_kda: int = 3
    H_csa: int = 2
    H_hca: int = 1
    head_dim: int = 32
    csa_m: int = 8
    csa_topk: int = 4
    hca_m2: int = 32
    csa_c: int = 32
    hca_c: int = 32


class HeadwiseFusedAttention(nn.Module):
    """A single layer that fuses KDA, CSA, and HCA *headwise* (not layerwise).

    This is a research prototype: it shows the API and a correct (if not fast)
    forward pass. The expected benefit is that every layer has access to all
    three capabilities, so the depth needed for recall + global context is
    lower. The expected cost is a more complex kernel (three code paths per
    layer) and potentially worse memory locality.

    Forward: x -> [LN -> headwise KDA|CSA|HCA -> concat -> o_proj] + residual.
    """

    def __init__(self, cfg: HeadwiseConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.H_kda + cfg.H_csa + cfg.H_hca == cfg.H_total
        assert cfg.csa_c == cfg.head_dim and cfg.hca_c == cfg.head_dim, \
            "Prototype requires csa_c == hca_c == head_dim"
        d, hd = cfg.d_model, cfg.head_dim

        # KDA branch (H_kda heads).
        self.kda_q = nn.Linear(d, cfg.H_kda * hd, bias=False)
        self.kda_k = nn.Linear(d, cfg.H_kda * hd, bias=False)
        self.kda_v = nn.Linear(d, cfg.H_kda * hd, bias=False)
        self.kda_g = nn.Linear(d, cfg.H_kda * hd, bias=False)
        self.kda_beta = nn.Linear(d, cfg.H_kda, bias=False)

        # CSA branch (H_csa heads) — simplified single-branch compression.
        self.csa_q = nn.Linear(d, cfg.H_csa * cfg.csa_c, bias=False)
        self.csa_kv = nn.Linear(d, cfg.csa_c, bias=False)
        self.csa_z = nn.Linear(d, cfg.csa_c, bias=False)
        self.csa_B = nn.Parameter(torch.randn(cfg.csa_m, cfg.csa_c) * 0.02)

        # HCA branch (H_hca heads).
        self.hca_q = nn.Linear(d, cfg.H_hca * cfg.hca_c, bias=False)
        self.hca_kv = nn.Linear(d, cfg.hca_c, bias=False)
        self.hca_z = nn.Linear(d, cfg.hca_c, bias=False)
        self.hca_B = nn.Parameter(torch.randn(cfg.hca_m2, cfg.hca_c) * 0.02)

        self.norm = nn.LayerNorm(d)
        self.o_proj = nn.Linear(cfg.H_total * hd, d, bias=False)
        # Attention scale: the dot product is over the compressed dim ``c``
        # (see the einsum in _csa_heads / _hca_heads: ``b t h d, b n d -> b h t n``
        # where ``d`` is the trailing dim of ``C_comp_n``, i.e. ``c``). The
        # correct scale is therefore ``c ** -0.5``. We previously used
        # ``hd ** -0.5``; the ``__init__`` assert ``csa_c == hca_c == head_dim``
        # made them numerically equal today, but using ``hd`` is a latent
        # footgun: if the assert is ever relaxed (e.g. to allow c != hd with
        # a per-head projection), the scale would silently be wrong. Use ``c``
        # so the formula matches the actual dot-product dimension.
        self.scale = cfg.csa_c ** -0.5

    def _kda_heads(self, x):
        B, T, d = x.shape
        H, hd = self.cfg.H_kda, self.cfg.head_dim
        # View BEFORE normalize: F.normalize(dim=-1) must operate on each
        # per-head hd-vector, not on the concatenated H*hd vector. The
        # previous form ``F.normalize(F.silu(self.kda_q(x)), dim=-1).view(...)``
        # L2-normalized the full H*hd vector, so each head's L2 norm became
        # ~1/sqrt(H) instead of 1. This silently shrinks q·k dot products
        # by a factor of 1/H, which propagates into the KDA recurrence as
        # under-scaled delta-rule updates. Mirrors the (correct) CSA/HCA
        # branches in _csa_heads / _hca_heads which view-then-normalize.
        q = F.normalize(F.silu(self.kda_q(x)).view(B, T, H, hd), dim=-1)
        k = F.normalize(F.silu(self.kda_k(x)).view(B, T, H, hd), dim=-1)
        v = F.silu(self.kda_v(x)).view(B, T, H, hd)
        # Parenthesize explicitly so the sign applies to the scaled value
        # (operator-precedence-safe): want g in (-inf, 0] so exp(g) in (0, 1].
        g = (-F.softplus(self.kda_g(x)) * 0.1).view(B, T, H, hd)
        beta = torch.sigmoid(self.kda_beta(x))
        o, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
        return o  # [B, T, H, hd]

    def _csa_heads(self, x):
        """CSA-style heads with single-branch compression + DENSE attention.

        NOTE: This prototype uses DENSE attention over all compressed blocks
        (no top-k sparse selection). The full CSA operator (ops_csa.py) adds
        a lightning indexer that selects top-k blocks per query — that is
        CSA's defining feature. The ``csa_topk`` field in HeadwiseConfig is
        accepted for API symmetry but is NOT used here. For the headwise
        prototype, dense attention is simpler and sufficient to demonstrate
        the fusion API; see ``naive_csa`` for the faithful sparse CSA.
        """
        B, T, d = x.shape
        H, c, m = self.cfg.H_csa, self.cfg.csa_c, self.cfg.csa_m
        pad = (-T) % m
        if pad:
            # RIGHT-pad: real tokens keep original positions; only the last
            # partial block contains padding zeros, and no real token
            # attends to it (causal block mask). Left-padding corrupted
            # block 0's compressed KV and silently changed real-token outputs.
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        n_blocks = Tp // m
        C = self.csa_kv(x)
        Z = self.csa_z(x)
        C_comp = csa_compress_kv(C, Z, self.csa_B, m)           # [B, n_blocks, c] in compute_dtype
        C_comp_n = F.normalize(C_comp, dim=-1)
        # Dtype: cast q to C_comp_n's dtype so the einsum doesn't crash on
        # fp16/bf16 inputs (C_comp is fp32 from the compression function).
        # Mirrors the fix in ops_csa.py::naive_csa and ops_hca.py::naive_hca.
        q = F.normalize(self.csa_q(x).view(B, Tp, H, c).to(C_comp_n.dtype), dim=-1)
        cbm = _causal_block_mask(Tp, n_blocks, m, x.device)
        scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * self.scale
        scores = scores.masked_fill(~cbm[None, None], float('-inf'))
        # Rows with no valid block to attend to (e.g. the first block's
        # queries under the causal block mask) are entirely -inf; softmax
        # over them would yield NaN. Detect such rows and force their
        # weights to 0 so the contribution is 0 instead of NaN
        # (mirrors ops_hca.py::naive_hca).
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)   # [B, H, T, 1]
        safe_scores = scores.masked_fill(all_masked, 0.0)
        p = torch.softmax(safe_scores, dim=-1)
        p = p.masked_fill(all_masked, 0.0)
        out = torch.einsum('b h t n, b n d -> b t h d', p, C_comp_n)
        if pad:
            out = out[:, :T]
        return out  # [B, T, H, c]

    def _hca_heads(self, x):
        B, T, d = x.shape
        H, c, m2 = self.cfg.H_hca, self.cfg.hca_c, self.cfg.hca_m2
        pad = (-T) % m2
        if pad:
            # RIGHT-pad: real tokens keep original positions; only the last
            # partial block contains padding zeros, and no real token
            # attends to it (causal block mask). Left-padding corrupted
            # block 0's compressed KV and silently changed real-token outputs.
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        n_blocks = Tp // m2
        C = self.hca_kv(x)
        Z = self.hca_z(x)
        C_comp = csa_compress_kv(C, Z, self.hca_B, m2)
        C_comp_n = F.normalize(C_comp, dim=-1)
        # Dtype: cast q to C_comp_n's dtype so the einsum doesn't crash on
        # fp16/bf16 inputs (C_comp is fp32 from the compression function).
        # Mirrors the fix in ops_csa.py::naive_csa and ops_hca.py::naive_hca.
        q = F.normalize(self.hca_q(x).view(B, Tp, H, c).to(C_comp_n.dtype), dim=-1)
        cbm = _causal_block_mask(Tp, n_blocks, m2, x.device)
        scores = torch.einsum('b t h d, b n d -> b h t n', q, C_comp_n) * self.scale
        scores = scores.masked_fill(~cbm[None, None], float('-inf'))
        # Same all-masked-row guard as _csa_heads / ops_hca.py::naive_hca:
        # the first block's queries have no preceding block to attend to,
        # so their score rows are entirely -inf and softmax would yield NaN.
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)   # [B, H, T, 1]
        safe_scores = scores.masked_fill(all_masked, 0.0)
        p = torch.softmax(safe_scores, dim=-1)
        p = p.masked_fill(all_masked, 0.0)
        out = torch.einsum('b h t n, b n d -> b t h d', p, C_comp_n)
        if pad:
            out = out[:, :T]
        return out

    def forward(self, x):
        B, T, d = x.shape
        h = self.norm(x)
        # Run all three head groups in parallel, then concat along head dim.
        # NOTE: head dims differ (KDA=hd, CSA=c, HCA=c). We project each to hd.
        hd = self.cfg.head_dim
        kda_o = self._kda_heads(h)                                  # [B, T, H_kda, hd]
        csa_o = self._csa_heads(h)                                  # [B, T, H_csa, c]
        hca_o = self._hca_heads(h)                                  # [B, T, H_hca, c]
        # If c != hd, we need a per-head projection. For the prototype we
        # assert c == hd so concat is direct (checked in __init__).
        #
        # Dtype: KDA output is in x.dtype (naive_recurrent_kda casts back),
        # but CSA/HCA outputs are in compute_dtype (fp32 for fp16 inputs)
        # because the compression functions return fp32. Concatenating mixed
        # dtypes promotes to fp32, which then crashes o_proj (fp16 weights)
        # with ``RuntimeError: expected m1 and m2 to have the same dtype``.
        # Cast all heads to x.dtype before concat so the entire forward runs
        # in the caller's dtype. The internal fp32 computation in CSA/HCA is
        # already done — this only affects the stored output precision.
        all_heads = torch.cat([
            kda_o.to(x.dtype), csa_o.to(x.dtype), hca_o.to(x.dtype)
        ], dim=2)                                                   # [B, T, H_total, hd]
        return x + self.o_proj(all_heads.reshape(B, T, self.cfg.H_total * hd))


def demo_headwise_fusion():
    """Run a tiny forward pass of the headwise-fused layer to confirm it works."""
    info = configure_torch_for_device()
    device = info.device
    print("\n" + "=" * 70)
    print(f"Headwise fusion demo (prototype, {device})")
    print("=" * 70)
    cfg = HeadwiseConfig(d_model=64, H_total=6, H_kda=3, H_csa=2, H_hca=1,
                         head_dim=16, csa_m=4, csa_topk=4, hca_m2=8,
                         csa_c=16, hca_c=16)
    model = HeadwiseFusedAttention(cfg).to(device)
    # Use eval() + no_grad() for this demo: the demo only checks output
    # shape/finiteness, so building the autograd graph is pure waste (the
    # graph would be retained on ``y`` until it goes out of scope). eval()
    # also disables any future dropout/BN that might be added to
    # HeadwiseFusedAttention, future-proofing the finiteness check.
    # Mirrors the pattern in run_correctness.py::test_fused_hybrid.
    model.eval()
    x = (torch.randn(2, 32, 64, device=device) * 0.1)
    with torch.no_grad():
        y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  input shape  : {tuple(x.shape)}")
    print(f"  output shape : {tuple(y.shape)}")
    print(f"  params       : {n_params}")
    print(f"  finite       : {torch.isfinite(y).all().item()}")
    print(f"  head split   : KDA={cfg.H_kda}, CSA={cfg.H_csa}, HCA={cfg.H_hca}")
    print("  (This is a research prototype; the layerwise tribrid is the main")
    print("   contribution. Headwise fusion is flagged as future work.)")
    return y


# ---------------------------------------------------------------------------
# 3. Complete CSA / HCA formulas
# ---------------------------------------------------------------------------

CSA_HCA_FORMULAS = r"""
Complete CSA and HCA formulas (DeepSeek-V4 §2.3.1–2.3.2)
=========================================================

Notation:
  T = sequence length, d = hidden dim, m = compression factor,
  c = compressed dim, H = number of attention heads,
  HI = number of indexer heads, DI = indexer dim,
  topk = number of selected compressed blocks per query,
  m' = heavy compression factor (m' >> m).

CSA — Compressed Sparse Attention
---------------------------------

1. Two-branch overlapped KV compression (Eq. 11–12):
   For each block i of m consecutive tokens:
     C^a_i  = H[i*m : (i+1)*m] @ W_aKV          # [m, c]
     C^b_i  = H[i*m : (i+1)*m] @ W_bKV          # [m, c]
     Z^a_i  = H[i*m : (i+1)*m] @ W_aZ           # [m, c]
     Z^b_i  = H[i*m : (i+1)*m] @ W_bZ           # [m, c]
     # Overlap: block i fuses C^a_i and C^b_{i-1} (previous block's b-branch)
     if i > 0:
         logits = [Z^a_i + B^a ; Z^b_{i-1} + B^b]   # [2m, c]
         S = softmax(logits, dim=0)                  # [2m, c]
         C_comp_i = sum(S[:m] * C^a_i) + sum(S[m:] * C^b_{i-1})
     else:
         S = softmax(Z^a_i + B^a, dim=0)
         C_comp_i = sum(S * C^a_i)
   -> Output: C_comp in R^{T/m x c}

2. Lightning indexer (Eq. 13–16):
   Indexer queries (low-rank):  q_idx = (H @ W_DQ @ W_IUQ).reshape(T, HI, DI)
   Indexer keys (compressed):   K_idx = csa_compress(H @ W_KV_idx, H @ W_Z_idx)
   Per-head similarity:         score_h[t, n] = ReLU(q_idx[t, h] . K_idx[n] / sqrt(DI))
   Aggregated:                  logits[t, n] = sum_h w_idx[t, h] * score_h[t, n]
   Selection:                   indices[t] = top-k(logits[t, :], k=topk, causal)

3. Shared-KV MQA core attention:
   Attention queries:  q = (H @ W_DQ @ W_UQ).reshape(T, H, c), L2-normalized
   Compressed KV is ALSO L2-normalized: C_comp_n = F.normalize(C_comp, dim=-1)
   (cosine-similarity attention — without this normalization the dot
   product would track magnitudes, not directions, defeating the sparse
   retrieval signal).
   For each query t:
     kv = C_comp_n[indices[t]]                      # [topk, c], shared across heads
     scores[h] = q[t, h] . kv^T * scale             # [H, topk]
     p[h] = softmax(scores[h] + sink)               # [H, topk]
     out[t, h] = p[h] @ kv                          # [H, c]
   (Note: both ``kv`` and the output einsum use the NORMALIZED C_comp_n.)

4. Sliding window branch (local uncompressed KV):
   For each query t: attend to H[t-w+1 : t+1] @ W_aKV with causal masking.
   Final output = sparse branch + sliding window branch.

HCA — Heavily Compressed Attention
-----------------------------------

1. Single-branch heavy compression (Eq. 20–23):
   C = H @ W_KV                                    # [T, c]
   Z = H @ W_Z                                     # [T, c]
   For each block i of m' consecutive tokens:
     logits = Z[i*m' : (i+1)*m'] + B_pos           # [m', c]
     S = softmax(logits, dim=0)                    # [m', c]
     C_comp_i = sum(S * C[i*m' : (i+1)*m'])        # [c]
   -> Output: C_comp in R^{T/m' x c}

2. Dense shared-KV MQA (NOT sparse — all compressed blocks):
   q = (H @ W_DQ @ W_UQ).reshape(T, H, c), L2-normalized
   C_comp_n = F.normalize(C_comp, dim=-1)         # ALSO L2-normalized
   Causal block mask: query t attends to blocks b where b < floor(t / m').
   scores[h, t, n] = q[t, h] . C_comp_n[n] * scale
   p = softmax(scores + causal_mask)
   out[t, h] = sum_n p[h, t, n] * C_comp_n[n]      # uses NORMALIZED C_comp_n

3. Sliding window branch: same structure as CSA's sliding window, but using
   HCA's single KV projection ``C = H @ W_KV`` (NOT CSA's two-branch
   ``Ca = H @ W_aKV``). For each query t: attend to H[t-w+1 : t+1] @ W_KV
   (L2-normalized) with causal masking.
4. Attention sink (optional, per-head learnable logit in the softmax denom).
"""


def print_formulas():
    print(CSA_HCA_FORMULAS)


# ---------------------------------------------------------------------------
# 4. Overlap-compression causality proof
# ---------------------------------------------------------------------------

OVERLAP_CAUSALITY_PROOF = """
Overlap-compression causality (formal argument)
================================================

Claim: In the two-branch overlapped CSA compression, the compressed
representation of block i depends only on source tokens from block i and
block i-1. It never depends on any token from block i+1 or later.

Proof:
  By construction (see csa_compress_kv_overlapped in ops_csa.py), the
  computation for block i is:

    if i > 0:
        a_chunk = Z^a[i*m : (i+1)*m] + B^a       # tokens from block i
        b_chunk = Z^b[(i-1)*m : i*m] + B^b       # tokens from block i-1
        cat = [a_chunk ; b_chunk]                # [2m, c]
        S = softmax(cat, dim=0)
        C_comp[i] = sum(S[:m] * C^a[i*m:(i+1)*m]) + sum(S[m:] * C^b[(i-1)*m:i*m])
    else:
        # i == 0: no previous block, b-branch is padded with -inf
        C_comp[0] = sum(softmax(Z^a[0:m] + B^a) * C^a[0:m])

  The source tokens accessed are:
    - C^a and Z^a at indices [i*m, (i+1)*m)  — block i
    - C^b and Z^b at indices [(i-1)*m, i*m)  — block i-1 (only if i > 0)

  No token at index >= (i+1)*m is ever read. Therefore:

    (a) Block i+1 (tokens [(i+1)*m, (i+2)*m)) does NOT influence C_comp[i].
    (b) By induction, no block j > i influences C_comp[i].

  The causal block mask _causal_block_mask(T, n_blocks, m, device) then
  ensures that query at position t only attends to blocks b < floor(t / m).
  Combined with the above, this means:

    output[t] depends on C_comp[0 : floor(t/m)]
              depends on tokens [0, floor(t/m) * m)
              which is a subset of [0, t+1)

  i.e. output[t] never depends on any token at position > t.  QED.

  Edge case (block 0): the b-branch is padded with -inf, so
  softmax([a_chunk ; -inf]) assigns zero weight to the padded entries,
  confirming block 0 has no dependency on a non-existent "previous" block.

  This is verified empirically in run_correctness.py::test_overlap_causality,
  which perturbs all forbidden tokens and confirms zero change in C_comp[i].
"""


def print_overlap_proof():
    print(OVERLAP_CAUSALITY_PROOF)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Method Analysis Module")
    print("=" * 70)
    print_rationale()
    print_formulas()
    print_overlap_proof()
    demo_headwise_fusion()
    print("\nMethod analysis complete.")


if __name__ == '__main__':
    main()
