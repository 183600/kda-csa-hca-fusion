"""Regression tests for the CSA compression primitives."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ops_csa import csa_compress_kv, csa_compress_kv_overlapped  # noqa: E402


def test_overlapped_zero_length_matches_single_branch_dtype():
    """T=0 must return the compute dtype (fp32) like the T>0 path."""
    B, T, m, c = 2, 0, 4, 8
    dtype = torch.float16
    Ca = torch.randn(B, T, c, dtype=dtype)
    Cb = torch.randn(B, T, c, dtype=dtype)
    Za = torch.randn(B, T, c, dtype=dtype)
    Zb = torch.randn(B, T, c, dtype=dtype)
    Ba = torch.randn(m, c, dtype=dtype)
    Bb = torch.randn(m, c, dtype=dtype)

    out = csa_compress_kv_overlapped(Ca, Cb, Za, Zb, Ba, Bb, m)
    assert out.shape == (B, 0, c)
    assert out.dtype == torch.float32
    single = csa_compress_kv(Ca, Za, Ba, m)
    assert single.dtype == out.dtype == torch.float32


def test_compress_chunk_tiles_match_forced_full_path():
    """Every tile size reproduces the full-buffer compression."""
    torch.manual_seed(7)
    B, T, m, c = 2, 64, 4, 8
    n_blocks = T // m
    C = torch.randn(B, T, c)
    Z = torch.randn(B, T, c)
    bias = torch.randn(m, c) * 0.1
    Ca = torch.randn(B, T, c)
    Cb = torch.randn(B, T, c)
    Za = torch.randn(B, T, c)
    Zb = torch.randn(B, T, c)
    Ba = torch.randn(m, c) * 0.1
    Bb = torch.randn(m, c) * 0.1

    with torch.no_grad():
        ref_s = csa_compress_kv(C, Z, bias, m, compress_chunk=n_blocks)
        ref_o = csa_compress_kv_overlapped(
            Ca, Cb, Za, Zb, Ba, Bb, m, compress_chunk=n_blocks)
        for tile in (1, 3, 5, n_blocks):
            got_s = csa_compress_kv(C, Z, bias, m, compress_chunk=tile)
            got_o = csa_compress_kv_overlapped(
                Ca, Cb, Za, Zb, Ba, Bb, m, compress_chunk=tile)
            assert torch.allclose(got_s, ref_s, atol=1e-6, rtol=0)
            assert torch.allclose(got_o, ref_o, atol=1e-6, rtol=0)


def test_compress_chunk_rejects_invalid_values():
    """``compress_chunk`` follows the int-geometry contract."""
    B, T, m, c = 1, 8, 2, 4
    C = torch.randn(B, T, c)
    Z = torch.randn(B, T, c)
    bias = torch.randn(m, c)
    Ca = torch.randn(B, T, c)
    Cb = torch.randn(B, T, c)
    Za = torch.randn(B, T, c)
    Zb = torch.randn(B, T, c)
    Ba = torch.randn(m, c)
    Bb = torch.randn(m, c)
    for bad in (True, 1.5, -1):
        for fn, args in (
                (csa_compress_kv, (C, Z, bias, m)),
                (csa_compress_kv_overlapped,
                 (Ca, Cb, Za, Zb, Ba, Bb, m))):
            try:
                fn(*args, compress_chunk=bad)
            except ValueError:
                continue
            raise AssertionError(f'{fn.__name__} accepted {bad!r}')



def test_naive_hca_compress_chunk_end_to_end():
    """``naive_hca`` forwards the compression tile size end to end."""
    from ops_hca import naive_hca

    torch.manual_seed(11)
    B, T, d = 1, 64, 16
    c, dc, nh, m2 = 8, 8, 2, 8

    def w(out, inp):
        return torch.randn(out, inp) * 0.2

    args = (
        torch.randn(B, T, d) * 0.1,
        w(c, d), w(c, d), torch.randn(m2, c) * 0.1,
        w(dc, d), w(c * nh, dc),
    )
    kw = dict(m2=m2, nh=nh, c=c, dc=dc, sliding_window=4,
              sink_logits=torch.zeros(nh), rope_dim=None)
    with torch.no_grad():
        full = naive_hca(*args, **kw, compress_chunk=T + 1)
        for chunk in (1, 5):
            got = naive_hca(*args, **kw, compress_chunk=chunk)
            assert torch.allclose(got, full, atol=1e-6, rtol=0)
    for bad in (True, 1.5, -1):
        try:
            naive_hca(*args, **kw, compress_chunk=bad)
        except ValueError:
            continue
        raise AssertionError(f'naive_hca accepted compress_chunk={bad!r}')
