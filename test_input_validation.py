"""Regression tests for the public operators' input-validation contract.

What is pinned here
-------------------
The repository validates its public entry points explicitly (``raise
ValueError``) instead of relying on ``assert``: ``assert`` statements are
silently stripped under ``python -O`` / ``PYTHONOPTIMIZE=1``, which would
turn an actionable contract error into an opaque failure from deep inside
a recurrence.

Three contracts are covered:

* ``ops_gated_attention.GatedDeltaRule`` rejects a head / key-dim /
  value-dim mismatch with a ``ValueError`` (and keeps rejecting it under
  ``python -O``);
* ``online_softmax.online_softmax_state`` requires a BOOLEAN mask. An
  integer mask would make ``~mask`` a bitwise inversion (``1 -> -2``,
  ``0 -> -1``) instead of a logical negation;
* ``ops_decoding_cache.CSADecodingCache.forward_step`` validates the
  indexer operands' shapes and devices, like ``forward_chunk`` already
  does, so a caller gets a clear error instead of a cryptic einsum
  failure deep inside the indexer.

Run via pytest::

    pytest -q test_input_validation.py

Or directly::

    python3 test_input_validation.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from online_softmax import online_softmax_state  # noqa: E402
from ops_decoding_cache import (  # noqa: E402
    CSADecodingCache, HCADecodingCache, _SlidingWindowRingBuffer,
)
from ops_gated_attention import (  # noqa: E402
    GatedDeltaRule, gated_delta_rule_step,
)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def test_gated_delta_rule_rejects_geometry_mismatch():
    """A head / key-dim / value-dim mismatch is a ValueError."""
    layer = GatedDeltaRule(num_heads=2, key_dim=4, value_dim=5)
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 5)
    layer(q, k, v)  # the matching geometry works

    for name, args in (
            ('head count', (torch.randn(1, 3, 3, 4), torch.randn(1, 3, 3, 4),
                            torch.randn(1, 3, 3, 5))),
            ('key dim', (torch.randn(1, 3, 2, 7), torch.randn(1, 3, 2, 7),
                         torch.randn(1, 3, 2, 5))),
            ('value dim', (torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4),
                           torch.randn(1, 3, 2, 6))),
    ):
        try:
            layer(*args)
            _ok(f'{name} mismatch rejected', False, 'no error raised')
        except ValueError:
            _ok(f'{name} mismatch rejected', True, '')
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'{name} mismatch rejected', False,
                f'{type(exc).__name__}: {exc}')


def test_gated_delta_rule_step_rejects_unknown_gate_type():
    """The functional step must not treat every label as ``per_head``."""
    S = torch.zeros(1, 2, 4, 5)
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 5)
    gate = torch.ones(2)
    beta = torch.full((2,), 0.1)
    out, state = gated_delta_rule_step(
        S, q, k, v, gate, beta, gate_type='per_head')
    assert out.shape == (1, 2, 5) and state.shape == S.shape
    try:
        gated_delta_rule_step(
            S, q, k, v, gate, beta, gate_type='bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown gate_type was silently accepted')


def test_gated_delta_rule_check_survives_python_optimized():
    """The geometry check must survive ``python -O`` (no bare asserts)."""
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, %r)
        import torch
        from ops_gated_attention import GatedDeltaRule
        layer = GatedDeltaRule(num_heads=2, key_dim=4, value_dim=5)
        q = torch.randn(1, 3, 2, 7)
        k = torch.randn(1, 3, 2, 7)
        v = torch.randn(1, 3, 2, 5)
        try:
            layer(q, k, v)
            print('NO_ERROR')
        except ValueError:
            print('VALUE_ERROR')
        except Exception as exc:
            print('OTHER:' + type(exc).__name__)
    ''') % HERE
    r = subprocess.run([sys.executable, '-O', '-c', code],
                       capture_output=True, text=True)
    _ok('python -O still rejects the geometry mismatch',
        r.stdout.strip() == 'VALUE_ERROR',
        f'stdout={r.stdout.strip()!r} stderr={r.stderr.strip()[:120]!r}')


def test_online_softmax_rejects_non_boolean_mask():
    """An integer mask must be rejected, not bit-inverted."""
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    bool_mask = torch.ones(2, 3, dtype=torch.bool)
    online_softmax_state(q, k, v, mask=bool_mask)  # the contract works
    for bad in (torch.ones(2, 3), torch.tensor([[1, 0, 1], [1, 1, 0]])):
        try:
            online_softmax_state(q, k, v, mask=bad)
            _ok(f'{bad.dtype} mask rejected', False, 'no error raised')
        except ValueError:
            _ok(f'{bad.dtype} mask rejected', True, '')
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'{bad.dtype} mask rejected', False,
                f'{type(exc).__name__}: {exc}')


def test_decoding_cache_to_rejects_non_float_storage_dtype():
    """``.to(dtype=...)`` follows the constructor's floating-point contract."""
    B, c, c_I, m, win = 1, 4, 4, 2, 4
    csa = CSADecodingCache(B, c, c_I, m, win, 'cpu', torch.float32)
    hca = HCADecodingCache(B, c, m, win, 'cpu', torch.float32)
    ring = _SlidingWindowRingBuffer(B, win, c, 'cpu', torch.float32)
    for obj in (csa, hca, ring):
        for bad in (torch.int32, torch.bool):
            try:
                obj.to(dtype=bad)
            except TypeError:
                pass
            else:
                raise AssertionError(
                    f'{type(obj).__name__}.to accepted {bad}')
        assert obj.dtype == torch.float32
    csa.to(dtype=torch.float64)
    hca.to(dtype=torch.float64)
    ring.to(dtype=torch.float64)
    assert (csa.dtype, hca.dtype, ring.dtype) == (torch.float64,) * 3


def test_csa_decode_step_validates_indexer_operands():
    """``forward_step`` checks q_idx / w_idx like ``forward_chunk`` does."""
    B, c, c_I, m, win, nh, nIh = 1, 4, 4, 2, 4, 2, 2
    cache = CSADecodingCache(B, c, c_I, m, win, 'cpu', torch.float32)
    q = torch.randn(B, 1, nh, c)
    q_idx = torch.randn(B, 1, nIh, c_I)
    w_idx = torch.randn(B, 1, nIh)

    bad_cases = (
        ('q_idx missing the sequence axis', q_idx.view(B, nIh, c_I), w_idx),
        ('w_idx missing the sequence axis', q_idx, w_idx.view(B, nIh)),
        ('w_idx with the wrong head count', q_idx,
         torch.randn(B, 1, nIh + 1)),
    )
    for name, bad_q_idx, bad_w_idx in bad_cases:
        try:
            cache.forward_step(q, bad_q_idx, bad_w_idx, topk=2, nh=nh,
                               nIh=nIh, scale=1.0)
            _ok(f'{name} rejected', False, 'no error raised')
        except ValueError:
            _ok(f'{name} rejected', True, '')
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'{name} rejected', False, f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    test_gated_delta_rule_rejects_geometry_mismatch()
    test_gated_delta_rule_step_rejects_unknown_gate_type()
    test_gated_delta_rule_check_survives_python_optimized()
    test_online_softmax_rejects_non_boolean_mask()
    test_decoding_cache_to_rejects_non_float_storage_dtype()
    test_csa_decode_step_validates_indexer_operands()
    print('all input-validation tests passed')
