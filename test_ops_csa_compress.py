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
