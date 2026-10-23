"""Regression test: stable softplus inverse in ``per_channel_log_decay_bias``.

``ops_kda.per_channel_log_decay_bias`` maps a desired half-life to the
gate bias ``u`` via the log-softplus inverse ``log(expm1(target))``.
``expm1`` overflows to ``inf`` once ``target * log2(e) > ~709`` (fp64), so
small ``decay_scale`` / short ``min_half_life`` produced ``inf`` bias —
i.e. the gate jumped from "almost no decay" straight to ``alpha = 0``
(complete forgetting).  The stable form ``log(-expm1(-target))`` is exact
over the whole finite range.

Run via pytest::

    pytest -q test_kda_decay_bias_stability.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_kda import per_channel_log_decay_bias  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'


def _recovered_half_lives(bias, scale):
    softplus = torch.nn.functional.softplus(bias.double())
    return math.log(2.0) / (softplus * scale)


def test_large_target_stays_finite():
    scale, hl_min, hl_max, K = 1e-4, 1.0, 4.0, 4
    bias = per_channel_log_decay_bias(
        1, K, scale, min_half_life=hl_min, max_half_life=hl_max)
    _ok('small decay_scale emits finite bias',
        bool(torch.isfinite(bias).all()), f'bias={bias.tolist()}')
    got = _recovered_half_lives(bias, scale)
    want = hl_min * (hl_max / hl_min) ** (
        torch.arange(K, dtype=torch.float64) / (K - 1))
    _ok('recovered half-lives match the requested spectrum',
        bool(torch.allclose(got, want, rtol=1e-5, atol=1e-6)),
        f'got={got.tolist()} want={want.tolist()}')


def test_default_spectrum_unchanged():
    scale, hl_min, hl_max, K = 0.1, 2.0, 1024.0, 8
    bias = per_channel_log_decay_bias(
        2, K, scale, min_half_life=hl_min, max_half_life=hl_max)
    _ok('default range emits finite bias',
        bool(torch.isfinite(bias).all()), f'max={bias.abs().max().item():.4f}')
    got = _recovered_half_lives(bias.view(2, K)[0], scale)
    want = hl_min * (hl_max / hl_min) ** (
        torch.arange(K, dtype=torch.float64) / (K - 1))
    _ok('default half-lives match the requested spectrum',
        bool(torch.allclose(got, want, rtol=1e-4, atol=1e-5)),
        f'first={got[0].item():.4f} last={got[-1].item():.4f}')


def test_huge_target_is_not_truncated():
    bias = per_channel_log_decay_bias(
        1, 1, 1e-6, min_half_life=1.0, max_half_life=1.0)
    _ok('1e-6 decay_scale emits a finite, huge bias',
        bool(torch.isfinite(bias).all()) and bias.item() > 600.0,
        f'bias={bias.tolist()}')
    hl = _recovered_half_lives(bias, 1e-6)
    _ok('huge bias recovers a 1-token half-life',
        bool(abs(hl.item() - 1.0) < 1e-6), f'half_life={hl.item():.12f}')


def test_extreme_half_life_range_uses_log_space():
    """``max/min`` above the float range must not poison the bias."""
    scale, hl_min, hl_max, K = 1e-10, 1e-10, 1e308, 4
    bias = per_channel_log_decay_bias(
        1, K, scale, min_half_life=hl_min, max_half_life=hl_max)
    _ok('finite bias when max/min overflows',
        bool(torch.isfinite(bias).all()), f'bias={bias.tolist()}')
    got = _recovered_half_lives(bias, scale)
    frac = torch.arange(K, dtype=torch.float64) / (K - 1)
    want = torch.exp(
        torch.log(torch.tensor(hl_min, dtype=torch.float64))
        + frac * (torch.log(torch.tensor(hl_max, dtype=torch.float64))
                  - torch.log(torch.tensor(hl_min, dtype=torch.float64))))
    _ok('log-space fallback recovers the requested spectrum',
        bool(torch.allclose(got, want, rtol=1e-3, atol=0.0)),
        f'got={got.tolist()} want={want.tolist()}')


def test_unrepresentable_half_life_range_is_rejected():
    """A bias outside the returned float32 range is a malformed profile."""
    for kwargs in (
        dict(decay_scale=1e-300, min_half_life=1.0, max_half_life=4.0),
        dict(decay_scale=0.1, min_half_life=1e-300, max_half_life=1e300),
    ):
        try:
            per_channel_log_decay_bias(1, 4, **kwargs)
        except ValueError:
            _ok(f'rejects unrepresentable profile {kwargs}', True)
        else:
            _ok(f'rejects unrepresentable profile {kwargs}', False,
                'no error raised')


if __name__ == '__main__':
    test_large_target_stays_finite()
    test_default_spectrum_unchanged()
    test_huge_target_is_not_truncated()
    test_extreme_half_life_range_uses_log_space()
    test_unrepresentable_half_life_range_is_rejected()
    print('all KDA decay-bias stability tests passed')
