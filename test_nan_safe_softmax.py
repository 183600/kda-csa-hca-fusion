"""Regression tests for the NaN-safe softmax used by CSA / HCA.

What is pinned here
-------------------
``ops_csa._nan_safe_softmax`` (re-exported by ``ops_hca``) is the last line
of defence before a fully-masked attention row turns into a ``NaN`` that
propagates through the whole layer output.  Its contract is:

* a row whose entries are **all** ``-inf`` is *fully masked* and must return
  **zeros**;
* a row with **finite** entries must equal ``torch.softmax`` exactly;
* a row containing ``+inf`` entries must stay **finite**: the softmax limit
  puts all the mass on the ``+inf`` entries (``1/k`` each).  ``torch.softmax``
  returns ``NaN`` there.

The third point is the one that regressed: the fully-masked predicate used
``isinf()``, which also matches ``+inf``.  An all-``+inf`` row was therefore
classified as "fully masked" and silently zeroed (the mass belongs on the
``+inf`` entries), while a row mixing ``+inf`` and finite entries was *not*
detected and produced ``NaN`` — in a function whose whole purpose is to not
produce ``NaN``.

The decoding-cache read paths (``ops_decoding_cache``) share the same
contract: they are the fast path behind ``naive_csa`` / ``naive_hca`` and
must not diverge from the reference chain on a ``+inf`` / ``-inf`` row.

Run via pytest::

    pytest -q test_nan_safe_softmax.py

Or directly::

    python3 test_nan_safe_softmax.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_csa import _nan_safe_softmax  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def test_fully_masked_rows_return_zeros():
    """A row of all ``-inf`` returns zeros (the documented contract)."""
    logits = torch.tensor([[[float('-inf'), float('-inf'), float('-inf')]]])
    out = _nan_safe_softmax(logits, dim=-1)
    _ok('all -inf row -> zeros',
        torch.equal(out, torch.zeros_like(out)), f'{out.tolist()}')
    _ok('all -inf row stays finite', bool(torch.isfinite(out).all()), '')


def test_finite_rows_match_torch_softmax():
    """Finite rows are bit-identical to ``torch.softmax``."""
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 8, dtype=torch.float64)
    out = _nan_safe_softmax(logits, dim=-1)
    _ok('finite rows == torch.softmax',
        torch.equal(out, torch.softmax(logits, dim=-1)), '')


def test_positive_infinity_rows_stay_finite():
    """``+inf`` rows keep the softmax limit instead of NaN / all-zeros."""
    all_pos = torch.tensor([[[float('inf'), float('inf')]]])
    out = _nan_safe_softmax(all_pos, dim=-1)
    _ok('all +inf row -> uniform over the +inf entries',
        torch.allclose(out, torch.full_like(out, 0.5)), f'{out.tolist()}')

    mixed = torch.tensor([[[float('inf'), 0.0, 1.0]]])
    out = _nan_safe_softmax(mixed, dim=-1)
    _ok('mixed +inf row -> all mass on the +inf entry',
        torch.allclose(out, torch.tensor([[[1.0, 0.0, 0.0]]])),
        f'{out.tolist()}')

    two_pos = torch.tensor([[[float('inf'), float('inf'), 1.0, -1.0]]])
    out = _nan_safe_softmax(two_pos, dim=-1)
    _ok('two +inf entries split the mass evenly',
        torch.allclose(out, torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])),
        f'{out.tolist()}')

    _ok('no NaN for +inf rows', bool(torch.isfinite(out).all()), '')


def test_masked_rows_and_pos_inf_coexist():
    """A batch that mixes a masked row and a ``+inf`` row is handled."""
    logits = torch.tensor([
        [[float('-inf'), float('-inf')]],
        [[float('inf'), 0.0]],
    ])
    out = _nan_safe_softmax(logits, dim=-1)
    _ok('masked row zeroed', torch.equal(out[0], torch.zeros(1, 2)),
        f'{out[0].tolist()}')
    _ok('+inf row normalized', torch.allclose(out[1], torch.tensor([[1.0, 0.0]])),
        f'{out[1].tolist()}')


def test_explicit_all_masked_mask_is_honoured():
    """A caller-supplied fully-masked mask still zeroes those rows."""
    logits = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    mask = torch.tensor([[[True], [False]]])
    out = _nan_safe_softmax(logits, dim=-1, all_masked_mask=mask)
    _ok('supplied mask zeroes its row',
        torch.equal(out[0, 0], torch.zeros(2)), f'{out[0, 0].tolist()}')
    _ok('other row untouched',
        torch.allclose(out[0, 1], torch.softmax(logits[0, 1], dim=-1)),
        f'{out[0, 1].tolist()}')


# --- decoding-cache read paths -------------------------------------------
#
# The chunked / per-token decode reads used to run a bare
# ``torch.softmax`` (NaN on a ``+inf`` row) and the HCA read used
# ``torch.isinf(...).all()`` as its fully-masked predicate (``isinf``
# also matches ``+inf``).  Both are the regression pinned above, so the
# cache paths are held to the same contract as the reference chain.

def test_joint_softmax_chunk_matches_the_nan_safe_contract():
    """``_joint_softmax_chunk`` follows the ``+inf`` / ``-inf`` contract."""
    from ops_decoding_cache import _joint_softmax_chunk
    logits = torch.tensor([[[[1.0, float('inf'), 2.0, 3.0],
                             [float('inf'), float('inf'), 0.0, 1.0],
                             [1.0, 5.0, -1.0, 2.0],
                             [2.0, -1.0, 0.0, 4.0]]]])
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    valid[..., 2, 1:] = False                      # partially masked row
    valid[..., 3, :] = False                       # fully masked row
    out = _joint_softmax_chunk(logits, valid, None)
    masked_logits = logits.masked_fill(~valid, float('-inf'))
    ref = _nan_safe_softmax(
        masked_logits, dim=-1, all_masked_mask=~valid.any(-1, keepdim=True))
    _ok('cache chunk softmax == reference NaN-safe softmax',
        torch.equal(out, ref), f'{out.tolist()}')
    _ok('cache chunk softmax stays finite on +inf rows',
        bool(torch.isfinite(out).all()), f'{out.tolist()}')
    _ok('+inf row puts all its mass on the +inf entry',
        torch.allclose(out[0, 0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0])),
        f'{out[0, 0, 0].tolist()}')
    _ok('two +inf entries split the mass evenly',
        torch.allclose(out[0, 0, 1], torch.tensor([0.5, 0.5, 0.0, 0.0])),
        f'{out[0, 0, 1].tolist()}')
    _ok('partially masked row keeps softmax on its valid slot',
        torch.allclose(out[0, 0, 2], torch.tensor([1.0, 0.0, 0.0, 0.0])),
        f'{out[0, 0, 2].tolist()}')
    _ok('fully-masked row is zeroed',
        torch.equal(out[0, 0, 3], torch.zeros(4)), f'{out[0, 0, 3].tolist()}')


def test_sink_softmax_positive_infinity_rows_stay_finite():
    """The sink branch follows the same ``+inf`` limit as the no-sink path."""
    from ops_csa import _sink_softmax
    logits = torch.tensor([[
        [[float('inf'), 0.0, 1.0]],
        [[float('inf'), float('inf'), float('-inf')]],
        [[float('-inf'), float('-inf'), float('-inf')]],
    ]])
    log_sink = torch.tensor([[[[0.7]]]])
    out = _sink_softmax(logits, log_sink)
    _ok('mixed +inf row -> all mass on the +inf entry',
        torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])),
        f'{out[0, 0, 0].tolist()}')
    _ok('two +inf entries split the mass evenly (sink gets none)',
        torch.allclose(out[0, 1, 0], torch.tensor([0.5, 0.5, 0.0])),
        f'{out[0, 1, 0].tolist()}')
    _ok('all -inf row is zeroed despite the sink',
        torch.equal(out[0, 2, 0], torch.zeros(3)),
        f'{out[0, 2, 0].tolist()}')
    _ok('sink softmax stays finite', bool(torch.isfinite(out).all()), '')

    # All-masked rows must not poison the backward pass with 0 * inf.
    x = torch.tensor([
        [[[float('-inf'), float('-inf')]]],
        [[[0.2, 0.3]]],
    ], requires_grad=True)
    y = _sink_softmax(x, torch.tensor([[[[0.7]]]]))
    grad = torch.autograd.grad(y.sum(), x)[0]
    _ok('all-masked sink row has a finite gradient',
        bool(torch.isfinite(grad).all()), f'{grad.tolist()}')


def _overflow_reference_sink(sink):
    """CSA/HCA reference outputs for finite inputs whose scores overflow."""
    from ops_csa import naive_csa
    from ops_hca import naive_hca
    torch.manual_seed(11)

    def pos(*shape):
        return torch.rand(*shape) * 0.2 + 0.05

    B, T, d = 1, 8, 8
    m, topk, nh, nIh, c, c_I, dc = 4, 2, 2, 2, 8, 4, 8
    H = pos(B, T, d)
    csa = naive_csa(
        H, pos(c, d), pos(c, d), pos(c, d), pos(c, d),
        pos(m, c), pos(m, c), pos(dc, d), pos(c * nh, dc),
        pos(c_I * nIh, dc), pos(nIh, d), pos(c_I, d), pos(c_I, d),
        pos(m, c_I), m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
        sliding_window=0, sink_logits=sink, normalize_qk=True,
        scale=1e39, fuse_projections=False)
    m2, nh2, c2, dc2 = 4, 2, 8, 8
    H2 = pos(B, T, d)
    hca = naive_hca(
        H2, pos(c2, d), pos(c2, d), pos(m2, c2), pos(dc2, d),
        pos(c2 * nh2, dc2), m2=m2, nh=nh2, c=c2, dc=dc2,
        sliding_window=0, sink_logits=sink, scale=1e39)
    return csa, hca


def test_reference_sink_keeps_overflowed_scores_finite():
    """``naive_csa`` / ``naive_hca`` sink paths must not emit NaN on +inf."""
    sink = torch.tensor([0.5, -0.3])
    csa, hca = _overflow_reference_sink(sink)
    _ok('CSA reference sink output stays finite under score overflow',
        bool(torch.isfinite(csa).all()),
        f'nan_count={(~torch.isfinite(csa)).sum().item()}')
    _ok('HCA reference sink output stays finite under score overflow',
        bool(torch.isfinite(hca).all()),
        f'nan_count={(~torch.isfinite(hca)).sum().item()}')


def _hca_overflow_decode(sink):
    """HCA decode with scores that overflow to ``+inf`` (no q/k norm)."""
    from ops_decoding_cache import HCADecodingCache
    torch.manual_seed(3)
    B, c, m2, win, nh = 1, 4, 4, 2, 1
    cache = HCADecodingCache(
        B, c, m2, win, torch.device('cpu'), torch.float32)
    cache._normalize_qk = False
    outs = []
    with torch.no_grad():
        for _t in range(6):
            # ~1e19 stored values x ~2e19 query values = ~2e38+ -> +inf
            # scores in fp32; the state itself stays finite.
            cache.append_step(
                torch.randn(B, 1, c) * 1e19, torch.randn(B, 1, c) * 0.1,
                torch.zeros(1, c))
            q = torch.randn(B, 1, nh, c) * 2e19
            outs.append(cache.forward_step(
                q, nh=nh, scale=1.0, sink_logits=sink, normalize_qk=False))
    return torch.cat(outs, dim=1)


def test_hca_decode_step_keeps_overflowed_scores_finite():
    """The HCA decode read must not turn ``+inf`` scores into NaN.

    Both chains mirror ``naive_hca``: an overflowed ``+inf`` score keeps
    the softmax limit instead of ``NaN``, and the sink cannot receive mass
    from a row whose valid scores are all ``+inf``.
    """
    out = _hca_overflow_decode(None)
    _ok('sink-less HCA decode output stays finite under score overflow',
        bool(torch.isfinite(out).all()),
        f'nan_count={(~torch.isfinite(out)).sum().item()} '
        f'shape={tuple(out.shape)}')
    out_sink = _hca_overflow_decode(torch.tensor([0.5]))
    _ok('sink HCA decode output stays finite under score overflow',
        bool(torch.isfinite(out_sink).all()),
        f'nan_count={(~torch.isfinite(out_sink)).sum().item()} '
        f'shape={tuple(out_sink.shape)}')
    _ok('sink HCA decode keeps its shape and dtype',
        out_sink.shape == out.shape and out_sink.dtype == out.dtype,
        f'{tuple(out_sink.shape)} {out_sink.dtype}')


def test_reference_operators_reject_oversized_scale():
    """A JSON-sized int temperature is a ValueError, not OverflowError."""
    from ops_csa import naive_csa
    from ops_hca import naive_hca
    torch.manual_seed(13)
    B, T, d = 1, 4, 8
    m, topk, nh, nIh, c, c_I, dc = 4, 2, 2, 2, 8, 4, 8
    H = torch.rand(B, T, d)
    csa_args = (
        H, torch.rand(c, d), torch.rand(c, d), torch.rand(c, d),
        torch.rand(c, d), torch.rand(m, c), torch.rand(m, c),
        torch.rand(dc, d), torch.rand(c * nh, dc),
        torch.rand(c_I * nIh, dc), torch.rand(nIh, d), torch.rand(c_I, d),
        torch.rand(c_I, d), torch.rand(m, c_I))
    m2, nh2, c2, dc2 = 4, 2, 8, 8
    hca_args = (
        torch.rand(B, T, d), torch.rand(c2, d), torch.rand(c2, d),
        torch.rand(m2, c2), torch.rand(dc2, d),
        torch.rand(c2 * nh2, dc2))
    huge = 10 ** 400
    try:
        naive_csa(*csa_args, m=m, topk=topk, nh=nh, nIh=nIh, c=c, c_I=c_I,
                  dc=dc, scale=huge)
    except ValueError:
        _ok('naive_csa rejects an oversized scale', True, '')
    else:
        _ok('naive_csa rejects an oversized scale', False, 'no ValueError')
    try:
        naive_hca(*hca_args, m2=m2, nh=nh2, c=c2, dc=dc2, scale=huge)
    except ValueError:
        _ok('naive_hca rejects an oversized scale', True, '')
    else:
        _ok('naive_hca rejects an oversized scale', False, 'no ValueError')


_TEST_FUNCS = [
    test_fully_masked_rows_return_zeros,
    test_finite_rows_match_torch_softmax,
    test_positive_infinity_rows_stay_finite,
    test_masked_rows_and_pos_inf_coexist,
    test_explicit_all_masked_mask_is_honoured,
    test_sink_softmax_positive_infinity_rows_stay_finite,
    test_reference_sink_keeps_overflowed_scores_finite,
    test_reference_operators_reject_oversized_scale,
    test_joint_softmax_chunk_matches_the_nan_safe_contract,
    test_hca_decode_step_keeps_overflowed_scores_finite,
]


if __name__ == '__main__':
    for _fn in _TEST_FUNCS:
        _fn()
    print('all NaN-safe softmax tests passed')
