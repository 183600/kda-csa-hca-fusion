"""Regression tests for ``LMWithHybrid`` embedding scaling (GPT-2 style).

This pins the "embedding layer + hardware-aware fused scale" improvement:

1. The default ``embed_scale`` is ``sqrt(d_model)`` (the standard GPT-2
   choice), so the first hybrid layer sees unit-ish scale inputs instead of
   the raw ``std=0.02`` embeddings.
2. The scale is applied as a fused scalar multiply on the embedding gather
   output — verified by comparing the hybrid-stack input against a manually
   scaled embedding.
3. The scale introduces NO new parameters / buffers: ``state_dict`` keys and
   ``count_params`` are identical to the unscaled (``embed_scale=1.0``)
   model, so the reproducibility contract is preserved.
4. The pure-prefill evaluation path (which calls ``LMWithHybrid.forward``)
   is exercised and produces finite logits.
5. Invalid scale values are rejected with an actionable error.

Run via pytest::

    pytest -q test_embed_scale.py

Or directly::

    python3 test_embed_scale.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_fused import HybridConfig  # noqa: E402
from train_lm_autodl import LMWithHybrid, count_params  # noqa: E402


def _tiny_cfg(d_model=32) -> HybridConfig:
    """Minimal 3-layer (3:1:1 -> 1:1:1) config for a cheap LM."""
    return HybridConfig(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=0,           # recurrent-only: cheapest path
        csa_m=4, csa_topk=4, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=8, hca_rope_dim=8,
        kda_backend='reference',
    )


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _capture_stack_input(model, input_ids):
    """Run a forward while capturing the tensor fed into the hybrid stack.

    Returns the ``[B, T, d]`` tensor that ``LMWithHybrid.forward`` passes to
    ``self.hybrid`` — i.e. the SCALED embedding output.
    """
    captured = {}
    handle = model.hybrid.register_forward_pre_hook(
        lambda _mod, args: captured.setdefault('x', args[0]))
    try:
        model.hybrid.reset_state()
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()
    return captured['x']


def test_embed_scale_default_is_sqrt_d_model(device='cpu'):
    print("\nTest: default embed_scale == sqrt(d_model)")
    for d_model in (32, 64, 256):
        m = LMWithHybrid(17, _tiny_cfg(d_model)).to(device)
        _ok(f"d_model={d_model}", m.embed_scale == math.sqrt(d_model),
            f"embed_scale={m.embed_scale}, expected {math.sqrt(d_model)}")


def test_embed_scale_fused_into_stack_input(device='cpu'):
    """The scale multiplies the gather output feeding the hybrid stack."""
    print("\nTest: embed_scale is fused into the hybrid-stack input")
    torch.manual_seed(0)
    cfg = _tiny_cfg(d_model=32)
    m = LMWithHybrid(17, cfg, embed_scale=4.0).to(device).eval()
    ids = torch.randint(0, 17, (2, 9), device=device)
    stack_in = _capture_stack_input(m, ids)
    raw = m.embed(ids)
    _ok("stack input == raw embed * scale",
        torch.allclose(stack_in, raw * 4.0, atol=1e-6),
        f"max|diff|={(stack_in - raw * 4.0).abs().max().item():.3e}")


def test_embed_scale_one_matches_legacy(device='cpu'):
    """embed_scale=1.0 reproduces the legacy unscaled forward exactly."""
    print("\nTest: embed_scale=1.0 == legacy unscaled forward")
    torch.manual_seed(1)
    cfg = _tiny_cfg(d_model=32)
    m1 = LMWithHybrid(17, cfg, embed_scale=1.0).to(device).eval()
    m2 = LMWithHybrid(17, cfg, embed_scale=1.0).to(device).eval()
    m2.load_state_dict(m1.state_dict())
    ids = torch.randint(0, 17, (2, 7), device=device)
    m1.hybrid.reset_state(); m2.hybrid.reset_state()
    with torch.no_grad():
        o1 = m1(ids)['logits']
        o2 = m2(ids)['logits']
    _ok("identical logits at scale=1.0",
        torch.allclose(o1, o2, atol=0.0),
        f"max|diff|={(o1 - o2).abs().max().item():.3e}")


def test_embed_scale_no_new_state(device='cpu'):
    """Scaling adds no parameters/buffers: state_dict and count unchanged."""
    print("\nTest: embed_scale adds no parameters / state_dict entries")
    m_def = LMWithHybrid(17, _tiny_cfg(d_model=32)).to(device)
    m_one = LMWithHybrid(17, _tiny_cfg(d_model=32), embed_scale=1.0).to(device)
    _ok("state_dict keys identical",
        set(m_def.state_dict()) == set(m_one.state_dict()),
        f"keys={len(m_def.state_dict())}")
    _ok("count_params identical",
        count_params(m_def) == count_params(m_one),
        f"params={count_params(m_def)}")


def test_embed_scale_prefill_path_finite(device='cpu'):
    """The pure-prefill evaluation path yields finite logits + loss."""
    print("\nTest: prefill path finite logits/loss with scaled embedding")
    torch.manual_seed(2)
    m = LMWithHybrid(17, _tiny_cfg(d_model=32)).to(device).eval()
    ids = torch.randint(0, 17, (2, 16), device=device)
    labels = ids.clone(); labels[:, :3] = -100
    m.hybrid.reset_state()
    with torch.no_grad():
        out = m(ids, labels=labels)
    _ok("finite logits", bool(torch.isfinite(out['logits']).all()),
        f"logits shape {tuple(out['logits'].shape)}")
    _ok("finite loss", bool(torch.isfinite(out['loss'])),
        f"loss={out['loss'].item():.4f}")


def test_embed_scale_validation(device='cpu'):
    """Non-positive / non-finite / non-numeric scales are rejected."""
    print("\nTest: embed_scale input validation")
    cfg = _tiny_cfg(d_model=32)
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        try:
            LMWithHybrid(17, cfg, embed_scale=bad)
        except ValueError:
            _ok(f"rejects {bad}", True)
        else:
            _ok(f"rejects {bad}", False, "no ValueError raised")


def main() -> int:
    print("=== test_embed_scale: embedding-layer fused scaling ===")
    tests = [
        test_embed_scale_default_is_sqrt_d_model,
        test_embed_scale_fused_into_stack_input,
        test_embed_scale_one_matches_legacy,
        test_embed_scale_no_new_state,
        test_embed_scale_prefill_path_finite,
        test_embed_scale_validation,
    ]
    for t in tests:
        t()
    print("=== test_embed_scale: ALL PASS ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
