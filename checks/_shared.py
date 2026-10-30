from __future__ import annotations

import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, sanitize_for_json, write_json_atomic
from ops_kda import (
    naive_recurrent_kda, naive_chunk_kda, scripted_chunk_kda,
    compiled_recurrent_kda, per_channel_log_decay_bias,
    assoc_scan_chunk_kda,
)
from dplr_transition import kda_dplr_certificate
from ops_csa import csa_compress_kv_overlapped, csa_lightning_indexer, _causal_block_mask, naive_csa
from ops_hca import naive_hca
from ops_fused import (HybridAttentionStack, HybridKCHAttention,
                       HybridConfig, KDAHybridLayer, build_mixer)
from ops_kda_backend import fla_available, kda_forward, validate_kda_backend
from ops_mamba import (Mamba2Mixer, build_mamba2_mixer,
                        validate_mamba2_geometry)
from ops_mamba_backend import (mamba2_official_available,
                                resolve_mamba_backend)
from ops_rwkv import (Rwkv7Mixer, build_rwkv7_mixer,
                      validate_rwkv7_geometry, rwkv7_reference_scan)
from ops_rwkv_backend import (resolve_rwkv_backend,
                              rwkv7_fla_available)
from ops_decoding_cache import Mamba2DecodingState

logger = logging.getLogger(__name__)


TOL = {
    # Default tolerance for "outputs match a reference" checks.
    'match': {
        torch.float32: 1e-5,
        torch.float64: 1e-9,
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
    },
    # Tolerance for "outputs are bit-identical across two code paths" checks
    # (e.g. chunk vs recurrent KDA — same math, different reduction order).
    'bit_match': {
        torch.float32: 1e-6,
        torch.float64: 1e-10,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
    },
    # Tolerance for "value is effectively zero" checks.
    'zero': 1e-12,
}


def _ok(name: str, cond: bool, detail: str = '') -> dict:
    status = 'PASS' if cond else 'FAIL'
    msg = f"  [{status}] {name}: {detail}"
    if cond:
        logger.info(msg)
    else:
        logger.error(msg)
    return {'name': name, 'status': status, 'detail': detail}


def _none_or_norm(g):
    """Return ``'None'`` if ``g is None`` else the L1 norm as a float.

    Used by ``test_csa_indexer_ste_gradient`` to render the per-parameter
    gradient in a compact form for the failure detail message. Returning
    the L1 norm (instead of the full tensor) keeps log lines short even
    when the test runs at large ``c`` / ``c_I``.
    """
    if g is None:
        return 'None'
    return float(g.abs().sum().item())


def _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=0):
    """Build random CSA parameters for the decoding cache tests."""
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


def _make_hca_decoding_params(d, c, m2, nh, dc, device, seed=0):
    """Build random HCA parameters for the decoding cache tests."""
    import torch.nn as nn
    g = torch.Generator(device=device).manual_seed(seed)
    def rp(out, inp):
        return nn.Linear(inp, out, bias=False).weight.detach().to(device)
    return dict(
        W_KV=rp(c, d), W_Z=rp(c, d),
        B_pos=torch.randn(m2, c, generator=g, device=device) * 0.02,
        W_DQ=rp(dc, d), W_UQ=rp(c * nh, dc),
    )


def _project_csa_for_cache(H, p, c, c_I, dc, nh, nIh):
    """Compute the 6 CSA projections + queries from H (mirrors naive_csa)."""
    import torch.nn.functional as F
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


def _project_hca_for_cache(H, p, c, dc, nh):
    """Compute the HCA projections + queries from H (mirrors naive_hca)."""
    import torch.nn.functional as F
    C = F.linear(H, p['W_KV'])
    Z = F.linear(H, p['W_Z'])
    cQ = F.linear(H, p['W_DQ'])
    q = F.linear(cQ, p['W_UQ']).view(*H.shape[:2], nh, c)
    return C, Z, q


def _run_safe(fn, device):
    """Run one test function with exception isolation.

    Without this wrapper, a single test crash (e.g. an unexpected
    ``RuntimeError`` from a dtype mismatch, or an ``ImportError``)
    propagates up through ``main()`` and discards every subsequent
    test result. The JSON report is never written, so the user has
    no way to see which tests passed before the crash. Wrapping each
    test lets the rest of the suite continue and produces a full
    report with the crashed test marked as FAIL.
    """
    try:
        return fn(device)
    except Exception as e:
        logger.exception(f"Test {fn.__name__} crashed")
        return [_ok(fn.__name__, False,
                    f'CRASH: {type(e).__name__}: {e}')]


def _mamba2_small_cfg(**overrides) -> HybridConfig:
    """Small Mamba-2-friendly config (d_model=32, headdim=16 -> 4 heads)."""
    base = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        memory_type='mamba2',
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=16, mamba_backend='reference',
    )
    base.update(overrides)
    return HybridConfig(**base)


def _rwkv7_small_cfg(**overrides) -> HybridConfig:
    """Small RWKV-7-friendly config (d_model=32, head_dim=16 -> H=2)."""
    base = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        memory_type='rwkv7',
        rwkv_head_dim=16, rwkv_decay_lora=4, rwkv_a_lora=4,
        rwkv_gate_lora=8, rwkv_backend='reference',
    )
    base.update(overrides)
    return HybridConfig(**base)


__all__ = [n for n in globals() if not n.startswith('__')]
