"""Experiment 14 — lightning-indexer top-k selector: path parity and
per-token decode cost at long context.

Motivation
----------
CSA's lightning indexer (DeepSeek-V4 Eq. 13-17) scores every completed
compressed block per query and keeps the top-k. ``_canonical_topk_indices``
makes that selection a pure function of the score vector (ties break on
ascending block index), so prefill and incremental decode agree exactly.
Two interchangeable kernels implement that rule: a stable descending sort
(whose output *is* the canonical rule) and an ``O(n)`` boundary path
(``kthvalue`` + comparison masks) for multi-row sparse scores.

``ops_csa._canonical_topk_indices`` picks its kernel by COST only
(both kernels implement the identical canonical rule; see
``test_indexer_selector.py``):

* stable sort when the selection is dense (``2*s >= n``) or the whole
  score array is small (``rows * n <= _TOPK_SORT_MAX_ELEMS``);
* the ``O(n)`` boundary path otherwise, including sparse decode with
  more blocks than the element budget.

What this experiment measures
-----------------------------
1. ``parity``: the shipped selector against an independent sort-based
   reference over a grid that provably exercises BOTH kernels, including
   the NaN hazard shapes. Rows must show zero mismatches and zero crashes
   BEFORE any timing row is trusted.
2. ``selector_latency``: per-call median latency of the shipped selector
   vs the stable-sort kernel vs ``torch.topk`` over a ``rows x n_blocks``
   grid.
3. ``indexer_latency``: end-to-end ``csa_lightning_indexer`` median
   latency at decode geometry with the shipped selector vs the sort
   kernel swapped in (temporarily, restored in a ``finally``), plus
   the full ``CSADecodingCache.forward_step`` step for share context.
4. ``sink_guard``: the sink-only degenerate configuration (no window, no
   top-k, no completed block) and NaN-poisoned indexer weights must stay
   finite and agree with ``naive_csa`` — the selector must never turn a
   NaN into a real block index.

Outputs
-------
``results/exp14_indexer_selector.json`` (schema kind
``exp14_indexer_selector``) + a provenance side file. Environment
overrides: ``SELECTOR_TOPK`` (default 16) and ``SELECTOR_DECODE_LENS``
(comma-separated token counts, default 16384,65536); the defaults are the
committed protocol.

Run::

    python3 run_indexer_selector.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ops_csa import (  # noqa: E402
    _TOPK_SORT_MAX_ELEMS, _canonical_topk_indices, csa_lightning_indexer,
    naive_csa,
)
from ops_decoding_cache import CSADecodingCache, _block_valid_mask  # noqa: E402
import ops_csa  # noqa: E402
from results_schema import (  # noqa: E402
    SchemaError, write_checked as write_results_checked,
)
from kaggle_setup import capture_provenance, write_json_atomic  # noqa: E402

NAN = float('nan')
NINF = float('-inf')

# Timing protocol: warmup + median over repetitions (repo-wide convention;
# see run_decoding's "median over repeats").
WARMUP = 3
REPS = 7

# Selector grid (rows = queries scored per call: rows=1 is B=1 decode).
SELECTOR_GRID = (
    # (rows, n_blocks)
    (1, 256), (1, 1024), (1, 2048), (1, 4096), (1, 16384), (1, 65536),
    (8, 256), (8, 1024), (8, 2048), (8, 4096), (8, 16384),
)

# Decode geometry: m=16 blocks of 16 tokens (T = 16 * n_blocks).
DECODE_GEOMETRY = dict(c=64, c_I=32, m=16, nh=4, nIh=2, win=64)
DECODE_LENS = (16384, 65536)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_list(name: str, default: tuple) -> tuple:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return tuple(int(x.strip()) for x in raw.split(',') if x.strip())
    except ValueError:
        return default


def _timeit(fn, warmup: int = WARMUP, reps: int = REPS) -> float:
    """Median wall-clock seconds per call (float, CPU-pinned by caller)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _timeit_blocked(fn_a, fn_b, warmup: int = WARMUP, reps: int = REPS,
                    rounds: int = 2) -> tuple:
    """Drift-resistant medians for two variants, measured in homogeneous
    blocks with alternating block order (A-B then B-A by default).

    Call-by-call interleaving would let each variant disturb the other's
    allocator/cache state; separate blocks in ABBA order instead cancel
    first-order drift while each variant runs in its own steady state.
    Returns ``(median_a, median_b)`` (median of the per-round medians).
    """
    med_a, med_b = [], []
    for rnd in range(rounds):
        pairs = ((fn_a, med_a), (fn_b, med_b))
        if rnd % 2:
            pairs = (pairs[1], pairs[0])
        for fn, out in pairs:
            for _ in range(warmup):
                fn()
            samples = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                samples.append(time.perf_counter() - t0)
            out.append(statistics.median(samples))
    return statistics.median(med_a), statistics.median(med_b)


def _historical_sort_selector(scores: torch.Tensor, k: int) -> torch.Tensor:
    """The pre-change kernel for the decode shape (ops_csa history).

    One stable descending sort + ``-inf -> -1`` slot masking — the exact
    expression the single-row fast path ran before the boundary-path
    dispatch replaced it.
    """
    s = min(k, scores.shape[-1])
    if s == 0:
        return scores.new_zeros(*scores.shape[:-1], 0, dtype=torch.long)
    order = torch.sort(
        scores, dim=-1, descending=True, stable=True).indices[..., :s]
    values = scores.gather(-1, order)
    return order.masked_fill(torch.isinf(values), -1)


def _reference_selector(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Independent canonical rule: stable sort over ``NaN -> -inf``."""
    n = scores.shape[-1]
    s = min(k, n)
    if s == 0:
        return scores.new_zeros(*scores.shape[:-1], 0, dtype=torch.long)
    safe = scores.masked_fill(torch.isnan(scores), NINF)
    order = torch.sort(
        safe, dim=-1, descending=True, stable=True).indices[..., :s]
    values = safe.gather(-1, order)
    return order.masked_fill(torch.isneginf(values), -1)


def _dispatch_of(shape, k) -> str:
    n = shape[-1]
    s = min(k, n)
    if s == 0:
        return 'empty'
    rows = 1
    for d in shape[:-1]:
        rows *= d
    if 2 * s >= n or rows * n <= _TOPK_SORT_MAX_ELEMS:
        return 'sort'
    return 'boundary'


def bench_parity(topk: int) -> list:
    """Selector vs reference over the dispatch-crossing grid + NaN hazards."""
    rows = []
    g = torch.Generator().manual_seed(0)
    n_checked = n_boundary = n_sort = 0
    for shape in ((1, 7), (2, 5), (1, 256), (3, 512), (2, 1024),
                  (1, 2048), (1, 4096), (4, 4096), (3, 33)):
        n = shape[-1]
        for k in (0, 1, 3, topk, n):
            for pat in ('random', 'ties', 'inf', 'nan'):
                base = torch.randn(n, generator=g) * 3
                if pat == 'ties':
                    base = (torch.randint(0, 3, (n,), generator=g)
                            .float() - 1.0)
                elif pat == 'inf':
                    base[torch.randperm(n, generator=g)[: max(1, n // 5)]] = NINF
                elif pat == 'nan':
                    base[torch.randperm(n, generator=g)[: max(1, n // 7)]] = NAN
                scores = base.unsqueeze(0).repeat(shape[0], 1)
                ref = _reference_selector(scores, k)
                got = _canonical_topk_indices(scores, k)
                n_checked += 1
                disp = _dispatch_of(shape, k)
                n_boundary += disp == 'boundary'
                n_sort += disp == 'sort'
                if not torch.equal(ref, got):
                    raise AssertionError(
                        f'selector path mismatch shape={shape} k={k} '
                        f'pat={pat}')
    rows.append({
        'section': 'parity', 'case': 'canonical_grid',
        'n_cases': n_checked, 'mismatches': 0, 'crashes': 0,
        'boundary_cases': n_boundary, 'sort_cases': n_sort,
        'note': ('shipped selector == sort-based reference; both dispatch '
                 'branches exercised'),
    })
    # NaN hazard shapes must select without raising and match the reference.
    hazards = [
        ('boundary_nan_crash',
         torch.tensor([[NAN, 5.0, 4.0, 3.0, 2.0, NAN],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 2),
        ('single_row_nan_slot', torch.tensor([[NAN, NAN, 1.0]]), 1),
        ('all_nan_rows', torch.full((3, 40), NAN), 5),
    ]
    n_haz = n_crash = n_bad = 0
    for name, scores, k in hazards:
        n_haz += 1
        try:
            got = _canonical_topk_indices(scores, k)
        except RuntimeError:
            n_crash += 1
            continue
        if not torch.equal(got, _reference_selector(scores, k)):
            n_bad += 1
    rows.append({
        'section': 'parity', 'case': 'nan_hazards',
        'n_cases': n_haz, 'mismatches': n_bad, 'crashes': n_crash,
        'note': 'NaN hazard shapes return well-formed reference-exact rows',
    })
    return rows


def bench_selector_latency(topk: int) -> list:
    """Per-call median latency: shipped vs historical-sort vs torch.topk."""
    rows = []
    for r, n in SELECTOR_GRID:
        if topk > n:
            continue
        scores = torch.randn(r, n)
        s = min(topk, n)
        t_shipped, t_hist = _timeit_blocked(
            lambda: _canonical_topk_indices(scores, topk),
            lambda: _historical_sort_selector(scores, topk))
        t_topk = _timeit(lambda: torch.topk(scores, s, dim=-1).indices)
        rows.append({
            'section': 'selector_latency',
            'rows': r, 'n_blocks': n, 'topk': topk,
            'dispatch': _dispatch_of((r, n), topk),
            'shipped_ms': t_shipped * 1e3,
            'historical_sort_ms': t_hist * 1e3,
            'torch_topk_ms': t_topk * 1e3,
            'speedup_vs_sort': t_hist / t_shipped,
        })
    return rows


def _build_cache(T: int, geom: dict, device='cpu', dtype=torch.float32):
    """Populate a CSADecodingCache to T tokens with bulk chunks (fast)."""
    B = 1
    cache = CSADecodingCache(B, geom['c'], geom['c_I'], geom['m'],
                             geom['win'], device, dtype)
    cache._normalize_qk = True
    g = torch.Generator(device=device).manual_seed(7)
    step = 512
    for s0 in range(0, T, step):
        n = min(s0 + step, T) - s0
        Ca = torch.randn(B, n, geom['c'], generator=g)
        Cb = torch.randn(B, n, geom['c'], generator=g)
        Za = torch.randn(B, n, geom['c'], generator=g)
        Zb = torch.randn(B, n, geom['c'], generator=g)
        Ki = torch.randn(B, n, geom['c_I'], generator=g)
        Zi = torch.randn(B, n, geom['c_I'], generator=g)
        cache.append_step(
            Ca, Cb, Za, Zb, Ki, Zi,
            torch.zeros(geom['m'], geom['c']),
            torch.zeros(geom['m'], geom['c']),
            torch.zeros(geom['m'], geom['c_I']))
        cache.forward_step(
            torch.randn(B, 1, geom['nh'], geom['c'], generator=g),
            torch.randn(B, 1, geom['nIh'], geom['c_I'], generator=g),
            torch.randn(B, 1, geom['nIh'], generator=g),
            topk=16, nh=geom['nh'], nIh=geom['nIh'],
            sink_logits=None, use_ste=False, normalize_qk=True)
    return cache


def bench_indexer_latency(topk: int, geom: dict, lengths) -> list:
    """End-to-end indexer at decode geometry: shipped vs historical sort."""
    rows = []

    def _with_historical(fn):
        """Run ``fn`` with the pre-change selector swapped in (restored)."""
        def _wrapped():
            prev = ops_csa._canonical_topk_indices
            ops_csa._canonical_topk_indices = _historical_sort_selector
            try:
                fn()
            finally:
                ops_csa._canonical_topk_indices = prev
        return _wrapped

    for T in lengths:
        cache = _build_cache(T, geom)
        nb = cache.n_blocks
        if nb == 0:
            continue
        q_idx = torch.randn(1, 1, geom['nIh'], geom['c_I'])
        w_idx = torch.randn(1, 1, geom['nIh'])
        K = cache.K_IComp
        K32 = K.to(torch.float32)
        qn = F.normalize(q_idx, dim=-1)
        Kn = F.normalize(K32, dim=-1)
        cbm = _block_valid_mask(
            nb, (cache.n_tokens_seen - 1) // geom['m'],
            torch.device('cpu'))[None, :]

        def idx_call():
            with torch.no_grad():
                csa_lightning_indexer(qn, Kn, w_idx, topk, scale=1.0,
                                      causal_block_mask=cbm,
                                      normalize_qk=False)

        t_shipped, t_hist = _timeit_blocked(
            idx_call, _with_historical(idx_call))

        q = torch.randn(1, 1, geom['nh'], geom['c'])
        sink = torch.zeros(geom['nh'])

        def step_call():
            with torch.no_grad():
                cache.forward_step(
                    q, q_idx, w_idx, topk=topk, nh=geom['nh'],
                    nIh=geom['nIh'], sink_logits=sink, use_ste=False,
                    normalize_qk=True)

        t_step, t_step_hist = _timeit_blocked(
            step_call, _with_historical(step_call), warmup=2, reps=5)
        # Selector-only medians on the same geometry (rows=1 synthetic).
        scores = torch.randn(1, nb)
        t_sel, t_sel_hist = _timeit_blocked(
            lambda: _canonical_topk_indices(scores, topk),
            lambda: _historical_sort_selector(scores, topk))
        rows.append({
            'section': 'indexer_latency', 'T': T, 'n_blocks': nb,
            'topk': topk, 'dispatch': _dispatch_of((1, nb), topk),
            'indexer_shipped_ms': t_shipped * 1e3,
            'indexer_historical_sort_ms': t_hist * 1e3,
            'indexer_speedup': t_hist / t_shipped,
            'step_shipped_ms': t_step * 1e3,
            'step_historical_sort_ms': t_step_hist * 1e3,
            'step_speedup': t_step_hist / t_step,
            'selector_shipped_ms': t_sel * 1e3,
            'selector_historical_sort_ms': t_sel_hist * 1e3,
            'selector_share_of_step': t_sel / t_step,
        })
    return rows


def _make_csa_params(d, c, c_I, m, nh, nIh, dc, device, seed=0):
    import torch.nn as nn

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


def bench_sink_guard() -> list:
    """Sink-only degenerate config + NaN-poisoned weights stay finite.

    Mirrors ``test_indexer_selector.py``'s end-to-end cases so the
    experiment file carries its own evidence next to the timings.
    """
    rows = []
    torch.manual_seed(3)
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    sink = torch.randn(nh) * 0.5
    p = _make_csa_params(d, c, c_I, m, nh, nIh, dc, 'cpu', seed=11)
    p['W_IUQ'] = torch.zeros_like(p['W_IUQ'])
    p['W_w'] = torch.zeros_like(p['W_w'])
    B = 1
    T = 3
    H = torch.randn(B, T, d) * 0.1
    o_full = naive_csa(
        H, p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
        p['Ba'], p['Bb'], p['W_DQ'], p['W_UQ'], p['W_IUQ'],
        p['W_w'], p['W_KV_idx'], p['W_Z_idx'], p['B_idx'],
        m=m, topk=0, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=0, sink_logits=sink, use_ste=False,
        normalize_qk=True,
    )
    cache = CSADecodingCache(B, c, c_I, m, 0, 'cpu', torch.float32)
    outs = []
    for t in range(T):
        proj = _project_csa(H[:, t:t + 1], p, c, c_I, dc, nh, nIh)
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = proj
        cache.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx, p['Ba'], p['Bb'], p['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache.forward_step(
            q_n, q_idx, w_idx, topk=0, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=sink, use_ste=False)
        outs.append(o_t.reshape(B, 1, nh * c))
    o_inc = torch.cat(outs, dim=1)
    diff = (o_full - o_inc).abs().max().item()
    rows.append({
        'section': 'sink_guard', 'case': 'sink_only_degenerate',
        'topk': 0, 'win': 0, 'n_blocks': 0,
        'max_abs_diff_vs_naive': diff,
        'output_max_abs': o_full.abs().max().item(),
        'finite': bool(torch.isfinite(o_full).all()
                       and torch.isfinite(o_inc).all()),
    })
    # NaN-poisoned indexer weights: selection all -1, outputs finite.
    p2 = dict(p)
    p2['W_w'] = torch.full_like(p['W_w'], NAN)
    H2 = torch.randn(B, 9, d) * 0.1
    o_full2 = naive_csa(
        H2, p2['W_aKV'], p2['W_bKV'], p2['W_aZ'], p2['W_bZ'],
        p2['Ba'], p2['Bb'], p2['W_DQ'], p2['W_UQ'], p2['W_IUQ'],
        p2['W_w'], p2['W_KV_idx'], p2['W_Z_idx'], p2['B_idx'],
        m=m, topk=2, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=0, sink_logits=None, use_ste=False,
        normalize_qk=True,
    )
    cache2 = CSADecodingCache(B, c, c_I, m, 0, 'cpu', torch.float32)
    outs2 = []
    for t in range(9):
        proj = _project_csa(H2[:, t:t + 1], p2, c, c_I, dc, nh, nIh)
        Ca, Cb, Za, Zb, K_idx, Z_idx, q, q_idx, w_idx = proj
        cache2.append_step(
            Ca, Cb, Za, Zb, K_idx, Z_idx, p2['Ba'], p2['Bb'], p2['B_idx'])
        q_n = F.normalize(q.to(torch.float), dim=-1)
        o_t = cache2.forward_step(
            q_n, q_idx, w_idx, topk=2, nh=nh, nIh=nIh, scale=1.0,
            sink_logits=None, use_ste=False)
        outs2.append(o_t.reshape(B, 1, nh * c))
    o_inc2 = torch.cat(outs2, dim=1)
    diff2 = (o_full2 - o_inc2).abs().max().item()
    rows.append({
        'section': 'sink_guard', 'case': 'nan_indexer_weights',
        'topk': 2, 'win': 0, 'n_blocks': 2,
        'max_abs_diff_vs_naive': diff2,
        'output_max_abs': o_full2.abs().max().item(),
        'finite': bool(torch.isfinite(o_full2).all()
                       and torch.isfinite(o_inc2).all()),
    })
    return rows


def main() -> int:
    topk = _env_int('SELECTOR_TOPK', 16)
    lengths = _env_list('SELECTOR_DECODE_LENS', DECODE_LENS)
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    results = []
    print('[exp14] parity gate (shipped vs sort-based reference)...')
    results += bench_parity(topk)
    for r in results:
        print(f"  {r['case']}: {r['n_cases']} cases, "
              f"{r['mismatches']} mismatches, {r['crashes']} crashes "
              f"(boundary={r.get('boundary_cases')}, "
              f"sort={r.get('sort_cases')})")

    print('[exp14] selector latency grid...')
    results += bench_selector_latency(topk)
    for r in results:
        if r['section'] != 'selector_latency':
            continue
        print(f"  rows={r['rows']:>2} n={r['n_blocks']:>6} "
              f"({r['dispatch']:>8}): shipped={r['shipped_ms']:8.4f}ms "
              f"sort={r['historical_sort_ms']:9.4f}ms "
              f"topk={r['torch_topk_ms']:8.4f}ms "
              f"speedup={r['speedup_vs_sort']:6.2f}x")

    print('[exp14] end-to-end decode indexer...')
    idx_rows = bench_indexer_latency(topk, DECODE_GEOMETRY, lengths)
    results += idx_rows
    for r in idx_rows:
        print(f"  T={r['T']:>6} n_blocks={r['n_blocks']:>5} "
              f"({r['dispatch']:>8}): indexer {r['indexer_shipped_ms']:.4f}"
              f"ms vs {r['indexer_historical_sort_ms']:.4f}ms "
              f"({r['indexer_speedup']:.2f}x), step "
              f"{r['step_shipped_ms']:.4f}ms vs "
              f"{r['step_historical_sort_ms']:.4f}ms "
              f"({r['step_speedup']:.2f}x)")

    print('[exp14] sink guard...')
    guard = bench_sink_guard()
    results += guard
    for r in guard:
        print(f"  {r['case']}: diff={r['max_abs_diff_vs_naive']:.2e} "
              f"|out|max={r['output_max_abs']:.3e} finite={r['finite']}")

    if not all(r.get('finite', True) and
               r.get('mismatches', 0) == 0 and r.get('crashes', 0) == 0
               for r in results):
        print('[exp14] ERROR: parity/guard checks failed; refusing to '
              'write a result file')
        return 1

    device = torch.device('cpu')
    os.makedirs('results', exist_ok=True)
    metadata = {
        'device': str(device),
        'topk': topk,
        'selector_grid': [list(p) for p in SELECTOR_GRID],
        'decode_lengths': list(lengths),
        'decode_geometry': dict(DECODE_GEOMETRY),
        'topk_sort_max_elems': _TOPK_SORT_MAX_ELEMS,
        'timing_protocol': {
            'warmup': WARMUP, 'reps': REPS,
            'statistic': 'median wall-clock seconds per call',
        },
        'seeds': {
            'value': 0,
            'scheme': ('torch.manual_seed(0); seeded torch.Generator(0/7) '
                       'for score tensors and cache fill; see '
                       'run_indexer_selector.main'),
        },
        'provenance': capture_provenance(),
    }
    try:
        write_results_checked(
            'results/exp14_indexer_selector.json', results,
            kind='exp14_indexer_selector', metadata=metadata)
    except SchemaError as e:
        print(f'[exp14] ERROR: refusing to write a contract-violating '
              f'result file: {e}')
        return 1
    print('\nSaved: results/exp14_indexer_selector.json '
          '(schema kind=exp14_indexer_selector)')
    try:
        prov = capture_provenance()
        prov['seeds'] = metadata['seeds']
        write_json_atomic(
            prov, 'results/exp14_indexer_selector_provenance.json',
            indent=2, allow_nan=False)
    except Exception as e:  # pragma: no cover - side file only
        print(f'[exp14] WARNING: failed to write provenance: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
