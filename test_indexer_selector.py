"""Regression tests for the lightning-indexer top-k selector
(``ops_csa._canonical_topk_indices``): path parity + NaN handling.

What is pinned here
-------------------
1. **The two selection kernels are interchangeable.** The selector
   dispatches between one stable descending sort (small arrays / dense
   selections) and the ``O(n)`` boundary path (``kthvalue`` + masks) as a
   pure COST heuristic. The grid below provably exercises BOTH branches
   and compares every result against an independent reference (stable
   descending sort over ``NaN -> -inf`` scores with ``-inf`` selections
   reported as ``-1``): random scores, maximal ties, ``-inf`` runs, NaN
   sprinkles, all-NaN rows, fp32 and fp64, ``k`` = 0 / > ``n``.
2. **NaN scores are unselectable, never a crash.** A NaN loses both the
   ``>`` and the ``==`` comparison, so a NaN at the boundary leaves a row
   with fewer than ``s`` selectable entries. The hazard shapes below
   return well-formed ``[..., s]`` tensors that match the reference, and
   NaN slots surface as ``-1``.
3. **The end-to-end contract is unchanged.** ``CSADecodingCache``
   stepwise decode equals full ``naive_csa`` with sink + sparse top-k +
   fully tied scores; the sink-only degenerate configuration (``win=0``,
   ``topk=0``, no completed block) is all-masked on both paths; and
   NaN-poisoned indexer weights leave both paths agreeing, finite, and
   with all-``-1`` selections.

Run via pytest::

    pytest -q test_indexer_selector.py

Or directly::

    python3 test_indexer_selector.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ops_csa import (  # noqa: E402
    _TOPK_SORT_MAX_ELEMS, _TOPK_TIE_REUSE_MIN_ELEMS,
    _boundary_selected_indices, _canonical_topk_indices,
    csa_lightning_indexer, naive_csa,
)
from ops_decoding_cache import CSADecodingCache  # noqa: E402
from run_indexer_selector import (  # noqa: E402
    _timeit_blocked_samples, _timeit_samples,
)
from test_csa_tiebreak import (  # noqa: E402
    _make_csa_params, _project_csa,
)

NAN = float('nan')
NINF = float('-inf')


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _reference(scores, k):
    """Independent canonical rule: stable sort over ``NaN -> -inf``.

    Slots that end up at ``-inf`` are reported as ``-1`` (the documented
    "invalid slot" contract). This never runs a kthvalue/boundary pass, so
    agreeing with it over the grid below is real evidence that the shipped
    boundary path implements the same rule as the sort path.
    """
    n = scores.shape[-1]
    s = min(k, n)
    if s == 0:
        return scores.new_zeros(*scores.shape[:-1], 0, dtype=torch.long)
    safe = scores.masked_fill(torch.isnan(scores), NINF)
    order = torch.sort(
        safe, dim=-1, descending=True, stable=True).indices[..., :s]
    values = safe.gather(-1, order)
    # Only -inf is the invalid-slot sentinel. ``+inf`` is a valid score
    # (``_nan_safe_softmax`` puts all mass on it), so ``isinf`` here would
    # wrongly reject the selector's correct +inf handling.
    return order.masked_fill(torch.isneginf(values), -1)


def _would_use_boundary(shape, k):
    """Replicate the cost dispatch to count which branch a case tests."""
    n = shape[-1]
    s = min(k, n)
    if s == 0 or 2 * s >= n:
        return False
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows * n > _TOPK_SORT_MAX_ELEMS


def _score_patterns(n, seed):
    """Score rows covering ties / inf / NaN hazards."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    rows.append(torch.randint(0, 3, (n,)).float() - 1.0)          # heavy ties
    rows.append(torch.zeros(n))                                  # all tied
    rows.append(torch.randn(n, generator=g) * 3)                 # generic
    r = torch.randn(n, generator=g)
    r[torch.randperm(n, generator=g)[:max(1, n // 5)]] = NINF    # -inf run
    rows.append(r)
    r = torch.randn(n, generator=g)
    r[torch.randperm(n, generator=g)[:max(1, n // 7)]] = NAN     # sparse NaN
    rows.append(r)
    rows.append(torch.full((n,), NAN))                           # all NaN
    rows.append(torch.full((n,), NINF))                          # all -inf
    r = torch.zeros(n)                                           # boundary tie
    r[: max(1, n // 2)] = 1.0
    rows.append(r)
    r = torch.randn(n, generator=g)                              # +inf valid
    r[torch.randperm(n, generator=g)[:max(1, n // 9)]] = float('inf')
    rows.append(r)
    return rows


def test_selector_grid_matches_reference_on_both_paths():
    """Every grid cell matches the reference, on both dispatch branches."""
    torch.manual_seed(0)
    shapes = [
        (1, 3), (1, 7), (2, 5), (4, 4), (1, 16),
        (1, 256), (3, 512), (2, 1024), (1, 2048), (1, 4096), (4, 4096),
        (2, 1), (5, 2), (3, 33),
    ]
    ks = (0, 1, 2, 3, 7, 256, 5000)
    n_cases = n_boundary = n_sort_sparse = 0
    for shape in shapes:
        n = shape[-1]
        for k in ks:
            s = min(k, n)
            for i, row in enumerate(_score_patterns(n, seed=1000 + n)):
                scores = row.unsqueeze(0).repeat(shape[0], 1)
                ref = _reference(scores, k)
                got = _canonical_topk_indices(scores, k)
                n_cases += 1
                if _would_use_boundary(shape, k):
                    n_boundary += 1
                elif s > 0 and 2 * s < n:
                    n_sort_sparse += 1
                if not torch.equal(ref, got):
                    raise AssertionError(
                        f'path mismatch shape={shape} k={k} pattern={i}\n'
                        f'scores={scores.tolist()}\nref={ref.tolist()}\n'
                        f'got={got.tolist()}')
    # Both branches must actually be exercised, else this test could pass
    # while never running the boundary path.
    _ok(
        'grid exercises both dispatch branches',
        n_boundary > 0 and n_sort_sparse > 0,
        f'{n_cases} cases, {n_boundary} boundary, '
        f'{n_sort_sparse} sparse-sort')
    _ok(
        'grid matches the independent reference exactly',
        True,
        f'{n_cases} cases across shapes={len(shapes)}, ks={len(ks)}')

    # fp64 must follow the same rule (kthvalue / isnan dtype support).
    r = torch.randn(1, 3000, dtype=torch.float64)
    r[0, 5] = NAN
    r[0, 7] = NINF
    got64 = _canonical_topk_indices(r, 9)
    _ok(
        'fp64 selection matches the reference',
        torch.equal(got64, _reference(r, 9)) and got64.dtype == torch.long,
        f'dtype={got64.dtype}, sel={got64[0].tolist()}')


def test_nan_scores_are_unselectable_not_a_crash():
    """NaN hazard shapes are well-formed and reference-exact."""
    # Multi-row sparse (boundary path): the NaN at the boundary loses both
    # comparisons, so the row has fewer than ``s`` selectable entries.
    crash = torch.tensor([
        [NAN, 5.0, 4.0, 3.0, 2.0, NAN],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    for k in (1, 2, 5):
        got = _canonical_topk_indices(crash, k)
        _ok(
            f'boundary-path NaN row matches the reference (k={k})',
            torch.equal(got, _reference(crash, k))
            and got.shape == (2, min(k, 6)),
            f'{got.tolist()}')

    # Single-row NaN: the sort path must never select the NaN slot.
    single = torch.tensor([[NAN, NAN, 1.0]])
    got1 = _canonical_topk_indices(single, 1)
    _ok(
        'single-row NaN slot is never selected',
        got1.tolist() == [[2]] and torch.equal(got1, _reference(single, 1)),
        f'{got1.tolist()} (finite block index 2 must win)')

    # All-NaN rows: every slot invalid.
    allnan = torch.full((3, 40), NAN)
    got2 = _canonical_topk_indices(allnan, 5)
    _ok(
        'all-NaN rows select nothing',
        got2.tolist() == [[-1] * 5] * 3,
        f'{got2.tolist()}')

    # General invariant: the count of returned valid slots equals
    # min(k, #non-NaN scores) per row, and no returned index exceeds n-1.
    for shape, k in (((2, 9), 3), ((5, 5000), 7), ((1, 8192), 4)):
        rows = []
        for i in range(shape[0]):
            r = torch.randn(shape[-1])
            r[:: 11 + i] = NAN
            r[:: 13 + i] = NINF
            rows.append(r)
        scores = torch.stack(rows)
        got = _canonical_topk_indices(scores, k)
        finite = (~torch.isnan(scores)).sum(-1)
        want_valid = torch.clamp(finite, max=min(k, shape[-1]))
        valid = (got >= 0).sum(-1)
        _ok(
            f'valid-slot count == min(k, #selectable) shape={shape}',
            torch.equal(valid, want_valid)
            and bool(got.max() <= shape[-1] - 1),
            f'valid={valid.tolist()} want={want_valid.tolist()}')


def _full_row_boundary_reference(blk, s):
    """Pre-specialization boundary kernel (full-row tie scan), for parity."""
    n = blk.shape[-1]
    top_s, top_i = torch.topk(blk, s, dim=-1, sorted=True)
    boundary = top_s[..., -1:]
    above = top_s > boundary
    n_above = above.sum(dim=-1, keepdim=True, dtype=torch.int32)
    ar = torch.arange(n, device=blk.device)
    above_idx = torch.sort(torch.where(above, top_i, n), dim=-1).values
    tie_idx = torch.topk(
        torch.where(blk == boundary, ar, n),
        s, dim=-1, largest=False, sorted=True).values
    slots = torch.arange(s, device=blk.device)
    tie_slot = (slots - n_above).clamp_(min=0)
    return torch.where(
        slots >= n_above, tie_idx.gather(-1, tie_slot), above_idx)


def test_tie_reuse_fast_path_matches_full_row_scan():
    """The tie-reuse specialization is index-identical to the full scan.

    The fast path fires when every boundary tie already sits in the top-s
    block (unique-boundary continuous scores, +inf ties that all fit) and
    must fall back for extra boundary ties (heavy ReLU ties, all -inf) and
    NaN boundaries. The full-row kernel is the independent reference, and
    the final selector is checked against the canonical sort reference too.
    """
    n = _TOPK_TIE_REUSE_MIN_ELEMS
    torch.manual_seed(0)
    r = torch.randn(2, n)
    unique = r
    heavy = torch.zeros(2, n)
    heavy[:, : n // 2] = 1.0
    posinf = torch.randn(1, n)
    posinf[0, 5] = float('inf')
    nan_boundary = torch.randn(1, n)
    nan_boundary[0, 7] = NAN
    cases = (
        ('unique boundary', unique),
        ('heavy boundary ties', heavy),
        ('+inf boundary', posinf),
        ('NaN boundary', nan_boundary),
        ('all -inf', torch.full((1, n), NINF)),
    )
    for name, scores in cases:
        for k in (1, 3, 16, 100):
            s = min(k, n)
            got = _boundary_selected_indices(scores, s)
            want = _full_row_boundary_reference(scores, s)
            _ok(
                f'tie-reuse parity [{name}, k={k}]',
                torch.equal(got, want),
                f'got={got[0].tolist()[:8]}... want={want[0].tolist()[:8]}...')
            final = _canonical_topk_indices(scores, k)
            ref = _reference(scores, k)
            _ok(
                f'canonical parity [{name}, k={k}]',
                torch.equal(final, ref),
                f'got={final[0].tolist()[:8]}... ref={ref[0].tolist()[:8]}...')


def test_indexer_nan_weights_multi_row_no_crash():
    """Prefill-shaped indexer calls with NaN logits select no blocks."""
    B, T, HI, DI = 2, 8, 2, 4
    n_blocks = 6
    q_idx = torch.randn(B, T, HI, DI)
    k_idx = torch.randn(B, n_blocks, DI)
    # NaN indexer head weights poison every [B, T, n_blocks] logit.
    w_idx = torch.full((B, T, HI), NAN)
    cbm = torch.ones(T, n_blocks, dtype=torch.bool)
    idx = csa_lightning_indexer(q_idx, k_idx, w_idx, 3,
                                causal_block_mask=cbm)
    _ok(
        'NaN-poisoned prefill logits select no blocks (no crash)',
        idx.shape == (B, T, 3) and bool((idx == -1).all()),
        f'shape={tuple(idx.shape)}, all -1={bool((idx == -1).all())}')

    # Same, at a geometry that provably runs the boundary path:
    # sparse top-k and rows*n above the sort-path element budget.
    B2, T2, n2, k2 = 1, 600, 2100, 3
    shape = (B2, T2, n2)
    assert _would_use_boundary(shape, k2), 'test must hit the boundary path'
    q2 = torch.randn(B2, T2, HI, DI)
    k2t = torch.randn(B2, n2, DI)
    w2 = torch.full((B2, T2, HI), NAN)
    idx2 = csa_lightning_indexer(q2, k2t, w2, k2)
    _ok(
        'NaN-poisoned boundary-path prefill selects no blocks (no crash)',
        idx2.shape == (B2, T2, k2) and bool((idx2 == -1).all()),
        f'shape={tuple(idx2.shape)}, rows*n={T2 * n2} > '
        f'_TOPK_SORT_MAX_ELEMS={_TOPK_SORT_MAX_ELEMS}, '
        f'all -1={bool((idx2 == -1).all())}')


def _run_cache_vs_naive(p, n_blocks, topk, win, sink, T, poison_nan=False,
                        device='cpu'):
    """Stepwise ``CSADecodingCache`` vs full ``naive_csa`` (module-level
    scaffold mirroring ``test_csa_tiebreak``'s decode test)."""
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    B = 1
    H = torch.randn(B, T, d, device=device) * 0.1
    if poison_nan:
        p = dict(p)
        p['W_w'] = torch.full_like(p['W_w'], NAN)
    o_full = naive_csa(
        H, p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
        p['Ba'], p['Bb'], p['W_DQ'], p['W_UQ'], p['W_IUQ'],
        p['W_w'], p['W_KV_idx'], p['W_Z_idx'], p['B_idx'],
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=win, sink_logits=sink, use_ste=False,
        normalize_qk=True,
    )
    cache = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    outs = []
    for t in range(T):
        H_t = H[:, t:t + 1]
        proj = _project_csa(H_t, p, c, c_I, dc, nh, nIh)
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = proj
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx, p['Ba'], p['Bb'], p['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, q_idx, w_idx,
            topk=topk, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=sink, use_ste=False,
        )
        outs.append(o_t.reshape(B, 1, nh * c))
    o_inc = torch.cat(outs, dim=1).to(o_full.dtype)
    return o_full, o_inc


def test_cache_decode_with_sink_sparse_ties_and_nan_weights():
    """End-to-end: sink + sparse tied top-k, degenerate sink-only, and
    NaN-poisoned indexer weights all agree with ``naive_csa``."""
    torch.manual_seed(7)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    sink = torch.randn(nh) * 0.5

    # (a) Sink + sparse top-k + fully tied scores (maximal canonical
    # tie-break pressure through the selector on both paths).
    p = _make_csa_params(d, c, c_I, m, nh, nIh, dc, 'cpu', seed=11)
    p['W_IUQ'] = torch.zeros_like(p['W_IUQ'])
    p['W_w'] = torch.zeros_like(p['W_w'])
    o_full, o_inc = _run_cache_vs_naive(
        p, n_blocks=4, topk=2, win=4, sink=sink, T=17)
    diff = (o_full - o_inc).abs().max().item()
    _ok(
        'cache decode == naive (sink, sparse tied top-k)',
        diff < 1e-5,
        f'max_diff={diff:.6e}, topk=2 < n_blocks=4, all indexer scores '
        f'tied at 0')

    # (b) Degenerate sink-only configuration: no window, no top-k, no
    # completed block -> the sparse and SW branches are both empty and the
    # output must be the (finite) all-masked result on BOTH paths.
    o_full, o_inc = _run_cache_vs_naive(
        p, n_blocks=4, topk=0, win=0, sink=sink, T=3)
    diff = (o_full - o_inc).abs().max().item()
    _ok(
        'cache decode == naive (sink-only, win=0, topk=0)',
        diff < 1e-6 and bool(torch.isfinite(o_full).all()),
        f'max_diff={diff:.6e}, |full|max={o_full.abs().max().item():.3e}')

    # (c) NaN-poisoned indexer weights: every logit is NaN, so the
    # selector must return all-``-1`` and the sparse branch contributes
    # nothing on BOTH the full-prefill and the incremental paths; the pin
    # here is cross-path agreement + finiteness.
    o_full, o_inc = _run_cache_vs_naive(
        p, n_blocks=2, topk=2, win=0, sink=None, T=9, poison_nan=True)
    diff = (o_full - o_inc).abs().max().item()
    _ok(
        'cache decode == naive (NaN indexer weights, no crash)',
        diff < 1e-6 and bool(torch.isfinite(o_full).all()),
        f'max_diff={diff:.6e}, NaN weights -> all-(-1) selection, '
        f'outputs finite')


def test_timing_samples_feed_the_ci_fields():
    """The blocked timer must return every timed call, not only a median.

    The experiment attaches bootstrap CIs to the latency rows; that is only
    auditable if the raw samples are exposed. This pins the sample count
    (``reps * rounds`` for the A/B path, ``reps`` for the single-variant
    path) and that the point medians are finite.
    """
    print('\nTest: blocked timing exposes samples for the bootstrap CIs')
    calls = {'a': 0, 'b': 0}

    def fn_a():
        calls['a'] += 1

    def fn_b():
        calls['b'] += 1

    med_a, med_b, samples_a, samples_b = _timeit_blocked_samples(
        fn_a, fn_b, warmup=0, reps=2, rounds=3)
    _ok('blocked timer keeps reps*rounds samples per variant',
        len(samples_a) == 6 and len(samples_b) == 6,
        f'{len(samples_a)} / {len(samples_b)}')
    _ok('blocked timer runs both variants in every round',
        calls == {'a': 6, 'b': 6}, f'{calls}')
    _ok('blocked medians and samples are finite',
        all(math.isfinite(v) and v >= 0.0
            for v in (med_a, med_b, *samples_a, *samples_b)),
        f'{med_a}, {med_b}')

    med, samples = _timeit_samples(fn_a, warmup=1, reps=4)
    _ok('single-variant timer keeps every post-warmup sample',
        len(samples) == 4 and all(math.isfinite(v) for v in samples)
        and math.isfinite(med), f'{len(samples)} samples')


def main() -> int:
    test_selector_grid_matches_reference_on_both_paths()
    test_nan_scores_are_unselectable_not_a_crash()
    test_tie_reuse_fast_path_matches_full_row_scan()
    test_indexer_nan_weights_multi_row_no_crash()
    test_cache_decode_with_sink_sparse_ties_and_nan_weights()
    test_timing_samples_feed_the_ci_fields()
    print('\nAll indexer-selector tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
