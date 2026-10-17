"""Regression tests for deterministic CSA indexer top-k tie-breaking.

What is pinned here
-------------------
``torch.topk`` leaves the order of equal values unspecified, and on CPU
its tie choice depends on the tensor's SIZE: ``torch.zeros(3)`` and
``torch.zeros(4)`` (one larger entry) pick different runner-up positions.
CSA's batched prefill scores ``n_blocks_full`` entries (including
``-inf``-masked future blocks) while the incremental decode path scores
only the completed ``n_blocks_inc`` ones, so tied ReLU scores require a
size-independent selection rule for the incremental output to match the
full-prefill output.

``ops_csa._canonical_topk_indices`` resolves ties on ascending block
index (a pure function of the score vector). These tests pin:

* the canonical tie rule (smallest indices win) and the ``-inf -> -1``
  slot contract;
* size independence across many block counts;
* indexer-level agreement between a masked full tensor and its valid
  prefix (the exact prefill-vs-decode geometry);
* end-to-end ``CSADecodingCache`` == ``naive_csa`` with a genuinely
  SPARSE ``topk < n_blocks`` and every score tied at 0.

Run via pytest::

    pytest -q test_csa_tiebreak.py

Or directly::

    python3 test_csa_tiebreak.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ops_csa import (  # noqa: E402
    _canonical_topk_indices, csa_lightning_indexer, naive_csa,
)
from ops_decoding_cache import CSADecodingCache  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _make_csa_params(d, c, c_I, m, nh, nIh, dc, device, seed=0):
    """Random CSA parameters mirroring run_correctness's decoding scaffold."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rp(out, inp):
        return nn.Linear(inp, out, bias=False).weight.detach().to(device)

    return dict(
        W_aKV=rp(c, d), W_bKV=rp(c, d), W_aZ=rp(c, d), W_bZ=rp(c, d),
        Ba=torch.randn(m, c, generator=g, device=device) * 0.02,
        Bb=torch.randn(m, c, generator=g, device=device) * 0.02,
        W_DQ=rp(dc, d), W_UQ=rp(c * nh, dc), W_IUQ=rp(c_I * nIh, dc),
        W_w=rp(nIh, d), W_KV_idx=rp(c_I, d), W_Z_idx=rp(c_I, d),
        B_idx=torch.randn(m, c_I, generator=g, device=device) * 0.02,
    )


def _project_csa(H, p, c, c_I, dc, nh, nIh):
    """Compute the 6 CSA projections + queries from H (mirrors naive_csa)."""
    combined_weight = torch.cat(
        [p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
         p['W_KV_idx'], p['W_Z_idx']], dim=0)
    combined_out = F.linear(H, combined_weight)
    Ca, Cb, Za, Zb, K_idx_raw, Z_idx = combined_out.split(
        [c, c, c, c, c_I, c_I], dim=-1)
    cQ = F.linear(H, p['W_DQ'])
    q = F.linear(cQ, p['W_UQ']).view(*H.shape[:2], nh, c)
    q_idx = F.linear(cQ, p['W_IUQ']).view(*H.shape[:2], nIh, c_I)
    w_idx = F.linear(H, p['W_w'])
    return Ca, Cb, Za, Zb, K_idx_raw, Z_idx, q, q_idx, w_idx


def test_canonical_topk_tie_rule_and_size_independence():
    """The selector is size-independent and ties go to the smallest index."""
    # [1, 0, 0, 0] -> top-2 must be blocks {0, 1}; top-3 {0, 1, 2}.
    scores = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sel2 = _canonical_topk_indices(scores, 2)
    sel3 = _canonical_topk_indices(scores, 3)
    _ok(
        'canonical tie-break picks smallest indices',
        sel2.tolist() == [[0, 1]] and sel3.tolist() == [[0, 1, 2]],
        f'top2={sel2.tolist()}, top3={sel3.tolist()}')

    # ``-inf`` selections (fewer finite scores than topk) stay -1.
    masked = torch.tensor([[2.0, 0.0, float('-inf'), float('-inf')]])
    selm = _canonical_topk_indices(masked, 3)
    _ok(
        'canonical keeps the -inf -> -1 contract',
        selm.tolist() == [[0, 1, -1]], f'{selm.tolist()}')

    # Size independence: one large score + zero ties, over a range of n.
    for n in (3, 4, 5, 8, 16, 33, 64, 65, 127):
        row = torch.zeros(1, n)
        row[0, 0] = 5.0
        for k in (1, 2, 3, min(n, 7)):
            sel = _canonical_topk_indices(row, k)
            _ok(
                f'canonical size-independent n={n} k={k}',
                sel.tolist() == [list(range(min(k, n)))],
                f'{sel.tolist()}')


def test_indexer_selection_agrees_full_vs_prefix_under_ties():
    """Masked full score tensor and its valid prefix select identically.

    Two tie-heavy configurations:
    (n_blocks=6, n_valid=4, topk=2) — scores ``[1,0,0,0]`` pick the
    runner-up from a 3-way zero tie; and (6, 5, 3) — scores
    ``[1,0,0,0,2]`` add a second tie slot. The canonical selector must
    return exactly the same index tensor for the masked full tensor and
    the unmasked prefix.
    """
    B, T, HI, DI = 1, 12, 2, 4
    q_idx = torch.zeros(B, T, HI, DI)
    q_idx[..., 0] = 1.0
    w = torch.ones(B, T, HI)
    cases = (
        # (k_rows, n_blocks_full, n_valid, topk)
        ([1.0, 0.0, 0.0, 0.0, -1.0, -2.0], 6, 4, 2),
        ([1.0, 0.0, 0.0, 0.0, 2.0, -3.0], 6, 5, 3),
    )
    for k_rows, n_full, n_valid, topk in cases:
        k_idx = torch.zeros(1, n_full, DI)
        for i, v in enumerate(k_rows):
            k_idx[0, i, 0] = v
        cbm = torch.zeros(T, n_full, dtype=torch.bool)
        cbm[:, :n_valid] = True
        idx_full = csa_lightning_indexer(q_idx, k_idx, w, topk,
                                         causal_block_mask=cbm)
        cbm_inc = torch.ones(T, n_valid, dtype=torch.bool)
        idx_inc = csa_lightning_indexer(q_idx, k_idx[:, :n_valid], w, topk,
                                        causal_block_mask=cbm_inc)
        _ok(
            f'prefill/decode-style selections agree under ties '
            f'(n={n_full}, valid={n_valid}, topk={topk})',
            torch.equal(idx_full, idx_inc),
            f'full={idx_full[0, 0].tolist()} vs '
            f'inc={idx_inc[0, 0].tolist()}')


def test_csa_decoding_cache_sparse_topk_ties(device='cpu'):
    """Incremental decode == full ``naive_csa`` with sparse topk + ties.

    The indexer weights (``W_IUQ``, ``W_w``) are zeroed so every valid
    block has the identical score 0 (maximal ties), and
    ``topk=2 < n_blocks=4`` forces a genuinely sparse selection; the
    canonical tie-break is what keeps the full and incremental paths on
    the same tied blocks.
    """
    torch.manual_seed(42)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win = 2, 4
    p = _make_csa_params(d, c, c_I, m, nh, nIh, dc, device, seed=1)
    p['W_IUQ'] = torch.zeros_like(p['W_IUQ'])
    p['W_w'] = torch.zeros_like(p['W_w'])
    T = 17
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1

    o_full = naive_csa(
        H, p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
        p['Ba'], p['Bb'], p['W_DQ'], p['W_UQ'], p['W_IUQ'],
        p['W_w'], p['W_KV_idx'], p['W_Z_idx'], p['B_idx'],
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=win, sink_logits=None, use_ste=False,
        normalize_qk=True,
    )

    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    outs = []
    for t in range(T):
        H_t = H[:, t:t + 1]
        proj = _project_csa(H_t, p, c, c_I, dc, nh, nIh)
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = proj
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx,
            p['Ba'], p['Bb'], p['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, q_idx, w_idx,
            topk=topk, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=None, use_ste=False,
        )
        outs.append(o_t.reshape(B, 1, nh * c))
    o_inc = torch.cat(outs, dim=1).to(o_full.dtype)

    max_diff = (o_full - o_inc).abs().max().item()
    _ok(
        'csa_decode_sparse_topk_ties_matches_full',
        max_diff < 1e-5,
        f'max_diff={max_diff:.6e} (tol=1e-5), T={T}, m={m}, win={win}, '
        f'topk={topk} < n_blocks={T // m} (all scores tied at 0)')


if __name__ == '__main__':
    test_canonical_topk_tie_rule_and_size_independence()
    test_indexer_selection_agrees_full_vs_prefix_under_ties()
    test_csa_decoding_cache_sparse_topk_ties()
    print('all CSA tie-break tests passed')
