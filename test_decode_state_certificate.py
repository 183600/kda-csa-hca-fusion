"""Decode-state certificate: live byte accounting and the constant-state claim.

What is pinned here
-------------------
The repository's central efficiency claim is that the memory operators
(KDA, Mamba-2, RWKV-7) carry a **constant-size decoding state** — the
SSD-style recurrent state is a sufficient statistic of the prefix, so the
per-request footprint does not grow with context — while softmax attention
grows linearly and CSA/HCA grow with the compressed-block count (~T/m).

Three things are verified against the LIVE tensors (not the analytic
model):

1. ``CSADecodingCache`` / ``HCADecodingCache`` expose an exact accounting
   surface (``nbytes``, ``nbytes_valid``, ``nbytes_breakdown``) that
   agrees with an independent walk over their tensors and with the
   analytic per-component formula implied by the geometric block
   capacity ``next_pow2(n_blocks)``.
2. The Exp 6 decode wrappers' ``state_nbytes()`` shows the claimed
   scaling: exactly constant for KDA / Mamba-2 / RWKV-7, exactly linear
   for the softmax KV cache, strictly sub-linear for CSA/HCA.
3. ``run_decoding.bench_decoding`` records the certificate fields on
   every row (JSON-safe) and ``_decode_state_certificate`` classifies the
   per-op growth from the measured bytes.

The analytical twin (``run_kv_cache.full_accounting`` /
``cache_accounting.analytic_cache_components``) is pinned elsewhere; the
value here is that a silent regression in the cache allocator or the
decode wrappers (e.g. accidental per-token state retention) turns the
constant-state claim red.

Run via pytest::

    pytest -q test_decode_state_certificate.py

Or directly::

    python3 test_decode_state_certificate.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_decoding_cache import (  # noqa: E402
    CSADecodingCache, HCADecodingCache,
)


def _ok(name, cond, detail=''):
    """Emit the repo's list-of-dict status contract (conftest interprets it)."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    return {'name': name, 'status': status, 'detail': detail}


def test_decoding_cache_nbytes_accounting(device='cpu'):
    """CSA/HCA caches: live accounting == tensor walk == analytic formula."""
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    win, B, T = 6, 2, 11
    esize = 4  # fp32
    results = []

    # ---- CSA: feed zeros (the accounting only cares about shapes) -------
    csa = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    for _ in range(T):
        z_c = torch.zeros(B, 1, c, device=device)
        z_i = torch.zeros(B, 1, c_I, device=device)
        csa.append_step(z_c, z_c, z_c, z_c, z_i, z_i,
                        torch.zeros(m, c, device=device),
                        torch.zeros(m, c, device=device),
                        torch.zeros(m, c_I, device=device))
    bd_csa = csa.nbytes_breakdown()

    def _walk_csa():
        alloc = valid = 0

        def add(tensor, valid_elems=None):
            nonlocal alloc, valid
            if tensor is None:
                return
            alloc += tensor.numel() * tensor.element_size()
            elems = tensor.numel() if valid_elems is None else valid_elems
            valid += elems * tensor.element_size()

        add(csa._C_comp, csa.n_blocks * c * B)
        add(csa._K_IComp, csa.n_blocks * c_I * B)
        if csa.sw_buffer is not None:
            add(csa.sw_buffer._buf, csa.sw_buffer.n_valid * c * B)
        for group in (csa._acc_Ca, csa._acc_Cb, csa._acc_Za, csa._acc_Zb,
                      csa._acc_K_idx, csa._acc_Z_idx):
            for tensor in group:
                add(tensor)
        add(csa._prev_Cb)
        add(csa._prev_Zb)
        return alloc, valid

    walk_csa = _walk_csa()
    cap = 1 << (csa.n_blocks - 1).bit_length() if csa.n_blocks else 0
    exp_compressed_alloc = cap * c * B * esize
    exp_compressed_valid = csa.n_blocks * c * B * esize
    exp_indexer_alloc = cap * c_I * B * esize
    exp_indexer_valid = csa.n_blocks * c_I * B * esize
    exp_sw_alloc = win * c * B * esize
    exp_sw_valid = min(T, win) * c * B * esize
    exp_partial = (T % m) * (4 * c + 2 * c_I) * B * esize
    exp_overlap = 2 * m * c * B * esize if csa.n_blocks else 0
    csa_components_ok = (
        bd_csa['compressed_kv_allocated_bytes'] == exp_compressed_alloc
        and bd_csa['compressed_kv_valid_bytes'] == exp_compressed_valid
        and bd_csa['indexer_kv_allocated_bytes'] == exp_indexer_alloc
        and bd_csa['indexer_kv_valid_bytes'] == exp_indexer_valid
        and bd_csa['sliding_window_allocated_bytes'] == exp_sw_alloc
        and bd_csa['sliding_window_valid_bytes'] == exp_sw_valid
        and bd_csa['partial_tail_allocated_bytes'] == exp_partial
        and bd_csa['overlap_allocated_bytes'] == exp_overlap)
    results.append(_ok(
        'csa nbytes accounting matches analytic components',
        csa_components_ok,
        f'compressed={bd_csa["compressed_kv_allocated_bytes"]} '
        f'(exp {exp_compressed_alloc}), '
        f'indexer={bd_csa["indexer_kv_allocated_bytes"]} '
        f'(exp {exp_indexer_alloc}), '
        f'sw={bd_csa["sliding_window_allocated_bytes"]} (exp {exp_sw_alloc}), '
        f'partial={bd_csa["partial_tail_allocated_bytes"]}'
        f' (exp {exp_partial}), '
        f'overlap={bd_csa["overlap_allocated_bytes"]} (exp {exp_overlap})'))
    results.append(_ok(
        'csa nbytes totals match the tensor walk',
        (bd_csa['total_allocated_bytes'], bd_csa['total_valid_bytes'])
        == walk_csa and csa.nbytes == walk_csa[0]
        and csa.nbytes_valid == walk_csa[1],
        f'breakdown={bd_csa["total_allocated_bytes"]}/'
        f'{bd_csa["total_valid_bytes"]}, walk={walk_csa}, '
        f'nbytes={csa.nbytes}/{csa.nbytes_valid}'))

    # A fresh cache still RETAINS the fixed ring buffer (allocated) but has
    # zero valid state — allocated >= valid always.
    fresh_csa = CSADecodingCache(B, c, c_I, m, win, device, torch.float32)
    results.append(_ok(
        'csa nbytes: allocated >= valid (fresh cache keeps ring buffer)',
        fresh_csa.nbytes >= fresh_csa.nbytes_valid == 0
        and fresh_csa.nbytes == win * c * B * esize,
        f'allocated={fresh_csa.nbytes}, valid={fresh_csa.nbytes_valid}'))

    # ---- HCA (m2 = 4) ----------------------------------------------------
    m2 = 4
    hca = HCADecodingCache(B, c, m2, win, device, torch.float32)
    for _ in range(T):
        z_c = torch.zeros(B, 1, c, device=device)
        hca.append_step(z_c, z_c, torch.zeros(m2, c, device=device))
    bd_hca = hca.nbytes_breakdown()
    cap_h = 1 << (hca.n_blocks - 1).bit_length() if hca.n_blocks else 0
    exp_h_alloc = cap_h * c * B * esize
    exp_h_valid = hca.n_blocks * c * B * esize
    exp_h_sw_alloc = win * c * B * esize
    exp_h_sw_valid = min(T, win) * c * B * esize
    exp_h_partial = (T % m2) * 2 * c * B * esize
    hca_components_ok = (
        bd_hca['compressed_kv_allocated_bytes'] == exp_h_alloc
        and bd_hca['compressed_kv_valid_bytes'] == exp_h_valid
        and bd_hca['sliding_window_allocated_bytes'] == exp_h_sw_alloc
        and bd_hca['sliding_window_valid_bytes'] == exp_h_sw_valid
        and bd_hca['partial_tail_allocated_bytes'] == exp_h_partial)
    results.append(_ok(
        'hca nbytes accounting matches analytic components',
        hca_components_ok,
        f'compressed={bd_hca["compressed_kv_allocated_bytes"]} (exp {exp_h_alloc}), '
        f'sw={bd_hca["sliding_window_allocated_bytes"]} (exp {exp_h_sw_alloc}), '
        f'partial={bd_hca["partial_tail_allocated_bytes"]} (exp {exp_h_partial})'))

    # JSON-safety: the breakdown must serialize under strict parsers.
    json_ok = True
    try:
        json.dumps(bd_csa, allow_nan=False)
        json.dumps(bd_hca, allow_nan=False)
    except (TypeError, ValueError):
        json_ok = False
    results.append(_ok('nbytes_breakdown is JSON-safe', json_ok, ''))
    return results


def test_decode_state_constant_vs_growing(device='cpu'):
    """KDA/Mamba-2/RWKV-7 state bytes constant; softmax linear; CSA/HCA sub-linear."""
    from run_decoding import (CSAAttnDecoding, HCAAttnDecoding,
                              KDAAttnDecoding, Mamba2AttnDecoding,
                              Rwkv7AttnDecoding, SoftmaxAttnDecoding)

    # d_model must be >= RWKV-7's default LoRA widths (32/32/64) so the
    # decode-scale factory does not reject the config (64 is its benchmark
    # geometry as well).
    d = 64
    lens = (8, 32)
    torch.manual_seed(0)

    def state_after(model, T):
        model.eval()
        model.reset()
        x = torch.randn(1, T, d, device=device) * 0.1
        with torch.no_grad():
            model(x)
        return model.state_nbytes()

    results = []
    for name, factory in (
            ('kda', lambda: KDAAttnDecoding(d)),
            ('mamba2', lambda: Mamba2AttnDecoding(d)),
            ('rwkv7', lambda: Rwkv7AttnDecoding(d))):
        model = factory()
        s_small = state_after(model, lens[0])
        s_big = state_after(model, lens[1])
        results.append(_ok(
            f'{name} decode state is constant in context length',
            s_small['valid_bytes'] == s_big['valid_bytes']
            and s_small['allocated_bytes'] == s_big['allocated_bytes']
            and s_small['valid_bytes'] > 0,
            f'T={lens[0]}: {s_small["valid_bytes"]}B, '
            f'T={lens[1]}: {s_big["valid_bytes"]}B'))

    softmax = SoftmaxAttnDecoding(d)
    s_small = state_after(softmax, lens[0])
    s_big = state_after(softmax, lens[1])
    results.append(_ok(
        'softmax KV state grows exactly linearly with context',
        s_big['valid_bytes'] == (lens[1] // lens[0]) * s_small['valid_bytes']
        and s_big['valid_bytes'] > s_small['valid_bytes'],
        f'T={lens[0]}: {s_small["valid_bytes"]}B, '
        f'T={lens[1]}: {s_big["valid_bytes"]}B'))

    # CSA/HCA at m/m2 = 4: the compressed-block term grows ~T/m, strictly
    # positive but well below the softmax linear factor of 4.
    for name, factory in (
            ('csa', lambda: CSAAttnDecoding(d, m=4, topk=2, nh=2, c=8,
                                            dc=8, nIh=1, c_I=4,
                                            sliding_window=4, rope_dim=None)),
            ('hca', lambda: HCAAttnDecoding(d, m2=4, nh=2, c=8, dc=8,
                                            sliding_window=4,
                                            rope_dim=None))):
        model = factory()
        s_small = state_after(model, lens[0])
        s_big = state_after(model, lens[1])
        results.append(_ok(
            f'{name} decode state grows sub-linearly (compressed cache)',
            s_small['valid_bytes'] > 0
            and s_big['valid_bytes'] > s_small['valid_bytes']
            and s_big['valid_bytes']
            < (lens[1] // lens[0]) * s_small['valid_bytes'],
            f'T={lens[0]}: {s_small["valid_bytes"]}B, '
            f'T={lens[1]}: {s_big["valid_bytes"]}B'))
    return results


def test_bench_decoding_records_state_bytes(device='cpu'):
    """Exp 6 rows carry the certificate fields and the classifier works."""
    from run_decoding import (KDAAttnDecoding, _decode_state_certificate,
                              bench_decoding)

    d = 16
    torch.manual_seed(0)
    r = bench_decoding(KDAAttnDecoding(d), d, prefill_len=8, n_decode=2,
                       device=torch.device(device), repeats=1)
    results = [_ok(
        'bench_decoding records live decode-state bytes',
        isinstance(r.get('decode_state_bytes'), int)
        and r['decode_state_bytes'] > 0
        and isinstance(r.get('decode_state_allocated_bytes'), int)
        and r['decode_state_allocated_bytes'] >= r['decode_state_bytes']
        and isinstance(r.get('decode_state_breakdown'), dict)
        and 'recurrent_state' in r['decode_state_breakdown'],
        f'valid={r.get("decode_state_bytes")}B, '
        f'allocated={r.get("decode_state_allocated_bytes")}B, '
        f'components={sorted((r.get("decode_state_breakdown") or {}).keys())}')]
    json_ok = True
    try:
        json.dumps(r, allow_nan=False)
    except (TypeError, ValueError):
        json_ok = False
    results.append(_ok('exp6 row with state bytes is JSON-safe', json_ok, ''))

    # Synthetic certificate: identical bytes across prefill lengths =>
    # constant; growing bytes => growing; error rows are excluded.
    rows = [
        {'op': 'kda', 'prefill_len': 8, 'decode_state_bytes': 1000},
        {'op': 'kda', 'prefill_len': 32, 'decode_state_bytes': 1000},
        {'op': 'softmax', 'prefill_len': 8, 'decode_state_bytes': 2000},
        {'op': 'softmax', 'prefill_len': 32, 'decode_state_bytes': 8000},
        {'op': 'broken', 'prefill_len': 8, 'error': 'OOM'},
    ]
    cert = _decode_state_certificate(rows, [8, 32])
    results.append(_ok(
        'decode-state certificate classifies constant vs growing',
        cert.get('kda', {}).get('classification') == 'constant'
        and cert.get('softmax', {}).get('classification') == 'growing'
        and cert['softmax']['max_over_min'] == 4.0
        and 'broken' not in cert,
        f'kda={cert.get("kda", {}).get("classification")}, '
        f'softmax={cert.get("softmax", {}).get("classification")}'))
    return results


def test_certificate_rejects_component_cancellation():
    """A flat total is not a constant state when components cancel.

    Synthetic HCA-shaped rows at prefill lengths 30/60/120: compressed KV
    grows 512 -> 1024 -> 2048 B while the partial tail empties
    1536 -> 1024 -> 0 B, so the total stays exactly 2560 B. The old
    total-only rule would call this constant; the component-aware
    certificate must call it growing and expose the cancellation.
    """
    from run_decoding import _decode_state_certificate

    lengths = [30, 60, 120]
    rows = []
    for p, compressed, tail in zip(lengths, (512, 1024, 2048),
                                   (1536, 1024, 0)):
        rows.append({
            'op': 'hca', 'prefill_len': p, 'decode_state_bytes': 2560,
            'decode_state_breakdown': {
                'hca_compressed_kv': {'allocated_bytes': compressed,
                                      'valid_bytes': compressed},
                'hca_sliding_window': {'allocated_bytes': 512,
                                       'valid_bytes': 512},
                'hca_partial_tail': {'allocated_bytes': tail,
                                     'valid_bytes': tail},
            },
        })
    rows += [
        {'op': 'kda', 'prefill_len': p, 'decode_state_bytes': 1100,
         'decode_state_breakdown': {
             'recurrent_state': {'allocated_bytes': 1000, 'valid_bytes': 1000},
             'conv_lookback': {'allocated_bytes': 100, 'valid_bytes': 100},
         }} for p in lengths
    ] + [
        # Legacy row pair without a breakdown: total-only fallback.
        {'op': 'legacy_constant', 'prefill_len': p, 'decode_state_bytes': 64}
        for p in lengths
    ]
    cert = _decode_state_certificate(rows, lengths)
    hca = cert.get('hca', {})
    kda = cert.get('kda', {})
    legacy = cert.get('legacy_constant', {})
    results = [
        _ok('component cancellation is classified as growing',
            hca.get('classification') == 'growing'
            and hca.get('total_valid_flat') is True
            and hca.get('total_valid_flat_but_components_grow') is True,
            f"class={hca.get('classification')}, "
            f"total_flat={hca.get('total_valid_flat')}, "
            f"growing={hca.get('growing_components')}"),
        _ok('certificate names both offsetting components',
            hca.get('growing_components')
            == ['hca_compressed_kv', 'hca_partial_tail']
            and hca['component_valid_bytes_by_prefill_len'][
                'hca_compressed_kv'] == {'30': 512, '60': 1024,
                                         '120': 2048},
            f"growing={hca.get('growing_components')}"),
        _ok('genuinely constant components stay constant',
            kda.get('classification') == 'constant'
            and kda.get('growing_components') == []
            and kda.get('total_valid_flat') is True,
            f"class={kda.get('classification')}"),
        _ok('legacy rows without a breakdown use the total-only fallback',
            legacy.get('classification') == 'constant'
            and legacy.get('growing_components') == [],
            f"class={legacy.get('classification')}"),
    ]
    return results


if __name__ == '__main__':
    all_rows = []
    for fn in (test_decoding_cache_nbytes_accounting,
               test_decode_state_constant_vs_growing,
               test_bench_decoding_records_state_bytes,
               test_certificate_rejects_component_cancellation):
        print(f'== {fn.__name__} ==')
        all_rows += fn()
    failed = [r for r in all_rows if r['status'] != 'PASS']
    if failed:
        for r in failed:
            print(f"FAILED: {r['name']}: {r['detail']}")
        sys.exit(1)
    print(f'all {len(all_rows)} decode-state certificate checks passed')
