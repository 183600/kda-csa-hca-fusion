"""Regression tests: partial RoPE caches vs ``torch.inference_mode()``.

What is pinned here
-------------------
``ops_rope`` caches its angle tables (``inv_freq``, value-keyed ``cos/sin``,
identity-keyed ``cos/sin``) and the position vectors returned by
``token_positions`` / ``block_positions`` in module-level dictionaries. All
of those objects used to be created under the caller's ambient mode, so a
long-context inference warm-up poisoned the caches in two ways:

1. Tensors created inside ``torch.inference_mode()`` are *inference tensors*:
   they have NO version counter. The long-context identity-table cache key
   reads ``positions._version``, so a position vector of length
   ``P > _TABLE_CACHE_MAX_POS (=8)`` created under inference mode raised
   ``RuntimeError: Inference tensors do not track version counter`` — i.e.
   inference-mode prefill/decoding crashed exactly on the long sequences
   RoPE exists for.
2. Tables cached as inference tensors cannot participate in a later
   grad-enabled call: reuse raised
   ``RuntimeError: Inference tensors cannot be saved for backward``.
3. ``partial_rope``'s ``dim`` validation must run in the right order. The
   feature-axis check ("``dim`` cannot be the feature (last) axis") ran
   BEFORE the negative-index wraparound, so a dim below ``-nd`` (e.g.
   ``dim=-5`` on a 4-D tensor) slipped through, wrapped to ``-1`` and
   silently resolved to the feature axis - rotating the heads, and failing
   with the confusing ``positions has N entries but axis dim=-1 ... has
   P=D`` instead of naming the bad dim. Out-of-range dims are now rejected
   up front, and in-range negative dims wrap to the same axis as their
   positive form.

``ops_rope._normal_tensor_scope`` now builds every cached object in
``torch.inference_mode(False)``, so the cache always holds ordinary tensors.
These tests verify the long-context inference path, the warm-up -> autograd
reuse path, end-to-end ``naive_csa`` / ``naive_hca`` output parity between
inference mode and normal mode, and the ``dim`` argument validation.

Run via pytest::

    pytest -q test_rope_inference_mode.py

Or directly::

    python3 test_rope_inference_mode.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import ops_rope as rope_mod  # noqa: E402
from ops_rope import (  # noqa: E402
    block_positions,
    inverse_partial_rope,
    partial_rope,
    token_positions,
)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _ref_rope(x, positions, rope_dim, base):
    """Cache-free float64 reference for ``dim=1`` inputs ``[B, T, H, D]``."""
    D = x.shape[-1]
    inv = base ** (-torch.arange(
        0, rope_dim, 2, dtype=torch.float64, device=x.device) / rope_dim)
    ang = positions.to(
        device=x.device, dtype=torch.float64)[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    xr = x.to(torch.float64)[..., D - rope_dim:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    o2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    out = x.to(torch.float64).clone()
    out[..., D - rope_dim::2] = o1
    out[..., D - rope_dim + 1::2] = o2
    return out


def test_position_vectors_are_ordinary_under_inference(device='cpu'):
    """Cached position vectors must not be inference tensors."""
    long_T = 29  # > _TABLE_CACHE_MAX_POS, and unused by other tests
    rope_mod._POSITION_VEC_CACHE.pop(
        ('tok', long_T, str(device), str(torch.long)), None)
    rope_mod._POSITION_VEC_CACHE.pop(
        ('blk', 5, 16, str(device), str(torch.long)), None)
    with torch.inference_mode():
        pos = token_positions(long_T, device)
        blk = block_positions(5, 16, device)
    _ok('token_positions is ordinary under inference_mode',
        not pos.is_inference(), f'T={long_T}')
    _ok('block_positions is ordinary under inference_mode',
        not blk.is_inference(), 'n_blocks=5, m=16')


def test_long_context_rope_in_inference_mode(device='cpu'):
    """Long-context (P > 8) RoPE runs under inference mode and matches ref.

    Pre-fix this raised ``RuntimeError: Inference tensors do not track
    version counter`` from the identity-table cache key.
    """
    torch.manual_seed(2026)
    T, rope_dim, base = 29, 8, 10000.0
    x = torch.randn(2, T, 2, 12, dtype=torch.float64,
                    device=device) * 0.1
    # Force the position vector to be materialised inside inference mode.
    rope_mod._POSITION_VEC_CACHE.pop(
        ('tok', T, str(device), str(torch.long)), None)
    with torch.inference_mode():
        pos = token_positions(T, device)
        y = partial_rope(x, pos, rope_dim, base, dim=1)
        y_inv = inverse_partial_rope(y, pos, rope_dim, base, dim=1)
    ref = _ref_rope(x, pos, rope_dim, base)
    err = (y.to(torch.float64) - ref).abs().max().item()
    inv_err = (y_inv.to(torch.float64) - x).abs().max().item()
    _ok('long-context inference-mode partial_rope is finite',
        bool(torch.isfinite(y).all()))
    _ok('long-context inference-mode partial_rope matches reference',
        err < 1e-12, f'T={T}, max_err={err:.2e}')
    _ok('long-context inference-mode inverse rotation round-trips',
        inv_err < 1e-12, f'max_err={inv_err:.2e}')


def test_inference_warmup_does_not_poison_autograd(device='cpu'):
    """Tables built during an inference warm-up stay usable in autograd.

    Uses a unique ``(rope_dim, base)`` key so the tables are guaranteed to be
    BUILT during the inference call, then reuses the same key with
    ``requires_grad`` inputs. Pre-fix the second call raised
    ``RuntimeError: Inference tensors cannot be saved for backward``.
    """
    torch.manual_seed(4321)
    rope_dim, base, T = 10, 4321.0, 5  # T <= 8 -> value-keyed table cache
    x = torch.randn(1, T, 2, 14, dtype=torch.float64, device=device) * 0.1
    with torch.inference_mode():
        pos = token_positions(T, device)
        y_warm = partial_rope(x, pos, rope_dim, base, dim=1)

    # The cached tables for this key must be ordinary tensors.
    inv = rope_mod._INV_FREQ_CACHE.get((rope_dim, base, str(device),
                                        torch.float64))
    tables = [v for k, v in rope_mod._TABLE_CACHE.items()
              if k[:4] == (rope_dim, base, str(device), torch.float64)]
    _ok('inv_freq cached after inference warm-up is ordinary',
        inv is not None and not inv.is_inference())
    _ok('value-keyed cos/sin tables cached after inference warm-up are '
        'ordinary',
        bool(tables) and all(not c.is_inference() and not s.is_inference()
                             for c, s in tables),
        f'n_entries={len(tables)}')

    ref = _ref_rope(x, pos, rope_dim, base)
    warm_err = (y_warm.to(torch.float64) - ref).abs().max().item()
    _ok('inference warm-up matches cache-free reference',
        warm_err < 1e-12, f'max_err={warm_err:.2e}')

    x_grad = x.clone().requires_grad_(True)
    y_grad = partial_rope(x_grad, pos, rope_dim, base, dim=1)
    y_grad.square().sum().backward()
    x_ref = x.clone().requires_grad_(True)
    _ref_rope(x_ref, pos, rope_dim, base).square().sum().backward()
    grad_ok = (x_grad.grad is not None
               and bool(torch.isfinite(x_grad.grad).all())
               and torch.allclose(x_grad.grad, x_ref.grad,
                                  atol=1e-10, rtol=1e-10))
    _ok('autograd survives inference warm-up with the same cache key',
        grad_ok, f'rope_dim={rope_dim}, base={base}')


def test_naive_csa_hca_inference_mode_long_rope(device='cpu'):
    """Full CSA/HCA operators with RoPE and T > 8 under inference mode.

    Matches the normal-mode output (atol 1e-5); pre-fix both calls crashed
    on the long token-position vector.
    """
    from ops_csa import naive_csa
    from ops_hca import naive_hca
    from run_correctness import (
        _make_csa_decoding_params, _make_hca_decoding_params,
    )

    # --- CSA (T=17 > 8, so token_positions takes the identity-cache path) ---
    d, c, c_I, m, nh, nIh, dc = 32, 8, 4, 4, 2, 1, 8
    topk, win, rope_dim = 100, 4, 8
    B, T = 1, 17
    p = _make_csa_decoding_params(d, c, c_I, m, nh, nIh, dc, device, seed=11)
    torch.manual_seed(42)
    H = torch.randn(B, T, d, device=device) * 0.1
    kwargs = dict(
        m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=win, sink_logits=None, use_ste=False,
        normalize_qk=True, rope_dim=rope_dim,
    )
    args = (
        H, p['W_aKV'], p['W_bKV'], p['W_aZ'], p['W_bZ'],
        p['Ba'], p['Bb'], p['W_DQ'], p['W_UQ'], p['W_IUQ'],
        p['W_w'], p['W_KV_idx'], p['W_Z_idx'], p['B_idx'],
    )
    with torch.inference_mode():
        o_inf = naive_csa(*args, **kwargs)
    o_norm = naive_csa(*args, **kwargs)
    csa_err = (o_inf - o_norm).abs().max().item()
    _ok('naive_csa inference_mode matches normal (RoPE, T>8)',
        csa_err < 1e-5, f'T={T}, rope_dim={rope_dim}, err={csa_err:.2e}')

    # --- HCA ---
    d, c, m2, nh, dc = 32, 8, 4, 2, 8
    p = _make_hca_decoding_params(d, c, m2, nh, dc, device, seed=12)
    torch.manual_seed(43)
    H = torch.randn(B, T, d, device=device) * 0.1
    hca_args = (H, p['W_KV'], p['W_Z'], p['B_pos'], p['W_DQ'], p['W_UQ'])
    hca_kwargs = dict(
        m2=m2, nh=nh, c=c, dc=dc, sliding_window=win,
        sink_logits=None, rope_dim=rope_dim,
    )
    with torch.inference_mode():
        o_inf = naive_hca(*hca_args, **hca_kwargs)
    o_norm = naive_hca(*hca_args, **hca_kwargs)
    hca_err = (o_inf - o_norm).abs().max().item()
    _ok('naive_hca inference_mode matches normal (RoPE, T>8)',
        hca_err < 1e-5, f'T={T}, rope_dim={rope_dim}, err={hca_err:.2e}')


def test_default_dim_is_the_sequence_axis(device='cpu'):
    """The default ``dim`` must rotate the SEQUENCE, not the heads.

    The 4-D query layout is ``[B, T, nh, c]``; its sequence axis is 1.
    The previous default (``-2``) resolved to axis 2 - the HEAD axis -
    and silently rotated the heads with token positions instead of the
    tokens.
    """
    torch.manual_seed(7)
    x = torch.randn(1, 4, 4, 8, dtype=torch.float64, device=device)
    pos = torch.arange(4, dtype=torch.long, device=device)
    default = partial_rope(x, pos, 8, 10000.0)
    explicit = partial_rope(x, pos, 8, 10000.0, dim=1)
    _ok('default dim equals the sequence axis (dim=1)',
        torch.equal(default, explicit), '')
    # Position 0 is the identity rotation, so token 0 is untouched.
    _ok('position 0 is the identity rotation',
        torch.allclose(default[0, 0], x[0, 0]), '')
    # With the sequence axis, the rotation does not depend on the head
    # index: reversing the head order commutes with the rotation.  With
    # the head axis it would not (each head would be rotated by its own
    # index as a position).
    flipped = partial_rope(x[:, :, [3, 2, 1, 0], :], pos, 8, 10000.0)
    _ok('the rotation does not depend on the head index',
        torch.allclose(flipped, default[:, :, [3, 2, 1, 0], :]), '')


def test_dim_validation_rejects_feature_and_out_of_range(device='cpu'):
    """``dim`` validation must reject the feature axis and out-of-range dims.

    Pre-fix a dim below ``-nd`` (``dim=-5`` on a 4-D tensor) escaped the
    feature-axis check, wrapped to ``-1`` and resolved to the feature axis
    via ``x.shape[-1]``, so it either rotated the heads when the positions
    length happened to match D, or failed with
    ``positions has N entries but axis dim=-1 ... has P=D``.
    """
    torch.manual_seed(11)
    x = torch.randn(2, 5, 3, 8, dtype=torch.float64, device=device)
    pos_seq = torch.arange(5, dtype=torch.long, device=device)
    pos_bat = torch.arange(2, dtype=torch.long, device=device)

    def _raises(dim, positions, needle):
        try:
            partial_rope(x, positions, 4, 10000.0, dim=dim)
        except ValueError as exc:
            _ok(f'dim={dim} rejected', needle in str(exc), str(exc))
            return
        _ok(f'dim={dim} rejected', False, 'no ValueError raised')

    _raises(-1, pos_seq, 'cannot be the feature (last) axis')
    _raises(3, pos_seq, 'cannot be the feature (last) axis')
    _raises(-5, pos_bat, 'out of range for a 4-D tensor')
    _raises(4, pos_bat, 'out of range for a 4-D tensor')
    # A 2-D [P, D] layout must pass dim=0 explicitly (documented contract).
    y2 = torch.randn(5, 8, dtype=torch.float64, device=device)
    try:
        partial_rope(y2, pos_seq, 4, 10000.0, dim=-1)
        _ok('2-D dim=-1 rejected', False, 'no ValueError raised')
    except ValueError as exc:
        _ok('2-D dim=-1 rejected',
            'cannot be the feature (last) axis' in str(exc), str(exc))

    # In-range negative dims wrap to the same axis as their positive form.
    for neg, pos_axis, positions in ((-4, 0, pos_bat), (-3, 1, pos_seq)):
        a = partial_rope(x, positions, 4, 10000.0, dim=neg)
        b = partial_rope(x, positions, 4, 10000.0, dim=pos_axis)
        _ok(f'dim={neg} equals dim={pos_axis}', torch.equal(a, b),
            f'maxdiff={(a - b).abs().max().item():.2e}')


if __name__ == '__main__':
    test_position_vectors_are_ordinary_under_inference()
    test_long_context_rope_in_inference_mode()
    test_inference_warmup_does_not_poison_autograd()
    test_default_dim_is_the_sequence_axis()
    test_dim_validation_rejects_feature_and_out_of_range()
    test_naive_csa_hca_inference_mode_long_rope()
    print('all RoPE inference-mode regression tests passed')
