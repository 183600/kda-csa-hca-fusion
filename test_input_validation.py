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
from types import SimpleNamespace

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


def test_kda_chunk_size_requires_an_exact_positive_int():
    """A bool / float / string chunk size must fail before slicing."""
    from ops_kda import naive_chunk_kda

    B, T, H, K, HV, V = 1, 3, 1, 2, 1, 2
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV, K)
    beta = torch.rand(B, T, HV)
    naive_chunk_kda(q, k, v, g, beta, chunk_size=2)  # valid contract

    for bad in (True, 1.5, '2'):
        try:
            naive_chunk_kda(q, k, v, g, beta, chunk_size=bad)
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'KDA chunk_size {bad!r} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'KDA chunk_size {bad!r} rejected', False,
                'no error raised')
    try:
        naive_chunk_kda(q, k, v, g, beta, chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError('KDA chunk_size=0 was silently accepted')


def test_hybrid_config_rejects_coerced_and_nonfinite_fields():
    """Numeric/typed config fields fail at construction, not deep in torch.

    ``init_std`` is multiplied into every positional bias; a negative value
    used to fail inside ``torch.randn`` at stack-build time, while NaN
    silently poisoned the biases. ``kda_chunk_size``/``kda_decay_scale``
    accepted ``True`` as 1/1.0, and the string-membership fields raised a
    bare ``TypeError`` for a list. All are actionable ``ValueError``/config
    errors now, and the valid values still work.
    """
    from ops_fused import HybridAttentionStack, HybridConfig

    base = dict(d_model=8, n_heads_qk=1, n_heads_v=1, head_dim_k=2,
                head_dim_v=2)
    HybridConfig(**base, init_std=0.0, kda_chunk_size=0)
    cases = (
        {'init_std': float('nan')},
        {'init_std': float('inf')},
        {'init_std': -0.1},
        {'init_std': True},
        {'kda_chunk_size': True},
        {'kda_decay_scale': True},
        {'memory_type': []},
        {'csa_qk_norm_mode': []},
        {'hca_qk_norm_mode': {}},
    )
    for patch in cases:
        try:
            HybridConfig(**base, **patch)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'HybridConfig {patch} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'HybridConfig {patch} rejected', False, 'no error raised')

    try:
        HybridAttentionStack(HybridConfig(**base), total_layers=True)
    except ValueError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f'total_layers=True raised {type(exc).__name__}: {exc}')
    else:
        raise AssertionError('total_layers=True was silently accepted')


def test_kda_g_clamp_rejects_positive_infinity_and_nan():
    """A positive-infinity lower bound is never a meaningful gate clamp.

    ``g.clamp(min=+inf)`` makes every log-decay ``+inf`` (state becomes
    non-finite), so both the operators and the backend adapter must reject
    it before the clamp. ``-inf`` (disabled) and finite bounds still work.
    """
    from ops_kda import naive_chunk_kda, naive_recurrent_kda
    from ops_kda_backend import kda_forward

    B, T, H, K, HV, V = 1, 3, 1, 2, 1, 2
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV, K)
    beta = torch.rand(B, T, HV)
    negative_inf = -float('inf')
    for fn, kwargs in (
            (naive_recurrent_kda, {}),
            (naive_chunk_kda, {'chunk_size': 2}),
            (kda_forward, {'use_chunk': False})):
        for bad in (float('inf'), float('nan'), True):
            try:
                fn(q, k, v, g, beta, g_clamp_min=bad, **kwargs)
            except (TypeError, ValueError):
                pass
            except Exception as exc:  # noqa: BLE001 - the point is the type
                _ok(f'{fn.__name__} g_clamp_min={bad!r} rejected', False,
                    f'{type(exc).__name__}: {exc}')
            else:
                _ok(f'{fn.__name__} g_clamp_min={bad!r} rejected', False,
                    'no error raised')
        with torch.no_grad():
            fn(q, k, v, g, beta, g_clamp_min=negative_inf, **kwargs)


def test_rope_rejects_invalid_dims_bases_and_positions():
    """RoPE parameter errors fail before they can produce NaN tables.

    ``base=NaN/inf`` used to reach ``_inv_freq`` and turn every rotated
    channel into NaN; a float/bool ``rope_dim`` or ``dim`` reached slicing
    with an opaque error. Positions are also required to be a 1-D integer
    vector, and the cached position builders require exact non-negative
    lengths.
    """
    from ops_rope import block_positions, partial_rope, token_positions

    x = torch.randn(1, 4, 8)
    pos = torch.arange(4)
    partial_rope(x, pos, 4)  # valid contract still works
    for kwargs in (
            {'rope_dim': 4, 'base': float('nan')},
            {'rope_dim': 4, 'base': float('inf')},
            {'rope_dim': 4, 'base': True},
            {'rope_dim': 4.0},
            {'rope_dim': True},
            {'rope_dim': 4, 'dim': True}):
        try:
            partial_rope(x, pos, **kwargs)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'partial_rope {kwargs} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'partial_rope {kwargs} rejected', False, 'no error raised')

    for fn, args in (
            (token_positions, (-1, 'cpu')),
            (token_positions, (True, 'cpu')),
            (block_positions, (-1, 2, 'cpu')),
            (block_positions, (2, 0, 'cpu'))):
        try:
            fn(*args)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            _ok(f'{fn.__name__}{args} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'{fn.__name__}{args} rejected', False, 'no error raised')


def test_decoding_cache_geometry_requires_exact_ints():
    """Cache constructors reject bool/float geometry and non-finite bases.

    ``B/c/m/win`` feed ``torch.zeros`` shapes and ring slicing; a bool or a
    float used to survive the ``< 1`` checks and fail later with an opaque
    torch error. ``rope_base=NaN/inf`` silently produced NaN position
    tables. Both cache classes and the ring buffer now validate the
    constructor contract, while valid geometry and the ``-inf``/finite
    clamp-like sentinels used elsewhere are unaffected.
    """
    from ops_decoding_cache import (
        CSADecodingCache, HCADecodingCache, _SlidingWindowRingBuffer,
    )

    CSADecodingCache(1, 4, 4, 2, 4, 'cpu', torch.float32)
    HCADecodingCache(1, 4, 2, 4, 'cpu', torch.float32)
    _SlidingWindowRingBuffer(1, 4, 4, 'cpu', torch.float32)

    for ctor, good in (
            (CSADecodingCache, (1, 4, 4, 2, 4, 'cpu', torch.float32)),
            (HCADecodingCache, (1, 4, 2, 4, 'cpu', torch.float32))):
        for bad in ((True,) + good[1:], (1.5,) + good[1:],
                    ('1',) + good[1:]):
            try:
                ctor(*bad)
            except ValueError:
                pass
            except Exception as exc:  # noqa: BLE001 - the point is the type
                _ok(f'{ctor.__name__} {bad} rejected', False,
                    f'{type(exc).__name__}: {exc}')
            else:
                _ok(f'{ctor.__name__} {bad} rejected', False,
                    'no error raised')
        for base in (float('nan'), float('inf'), True):
            if ctor is CSADecodingCache:
                call = (1, 4, 4, 2, 4, 'cpu', torch.float32)
            else:
                call = (1, 4, 2, 4, 'cpu', torch.float32)
            try:
                ctor(*call, rope_base=base)
            except ValueError:
                pass
            except Exception as exc:  # noqa: BLE001
                _ok(f'{ctor.__name__} rope_base={base!r} rejected', False,
                    f'{type(exc).__name__}: {exc}')
            else:
                _ok(f'{ctor.__name__} rope_base={base!r} rejected', False,
                    'no error raised')

    for args in ((True, 2, 4, 'cpu', torch.float32),
                 (1, 2.0, 4, 'cpu', torch.float32),
                 (1, 2, 0, 'cpu', torch.float32)):
        try:
            _SlidingWindowRingBuffer(*args)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            _ok(f'ring buffer {args} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'ring buffer {args} rejected', False, 'no error raised')


def test_online_softmax_chunk_sizes_require_exact_ints():
    """A bool/float tile size must not silently become a different schedule."""
    from online_softmax import online_softmax_attention

    q = torch.randn(2, 2, 3, 4)
    k = torch.randn(2, 1, 5, 4)
    v = torch.randn(2, 1, 5, 4)
    online_softmax_attention(q, k, v, key_chunk=2, query_chunk=2)
    for patch in ({'key_chunk': True}, {'key_chunk': 1.5},
                  {'query_chunk': True}, {'query_chunk': '2'}):
        try:
            online_softmax_attention(q, k, v, **patch)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'online softmax {patch} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'online softmax {patch} rejected', False,
                'no error raised')


def test_gate_factory_and_gated_delta_geometry_reject_coercions():
    """Head widths are exact positive ints; gate std is finite and >= 0."""
    from ops_gated_attention import GatedDeltaRule, create_gate

    create_gate('per_channel', 2, 4)
    create_gate('none', 0, 0)  # 'none' ignores the unused geometry
    for kwargs in (
            {'num_heads': True}, {'num_heads': 2.0}, {'gate_dim': 0},
            {'init_std': -1.0}, {'init_std': float('nan')},
            {'init_value': float('inf')}):
        call = {'num_heads': 2, 'gate_dim': 4}
        call.update(kwargs)
        try:
            create_gate('per_channel', **call)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'create_gate {kwargs} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'create_gate {kwargs} rejected', False, 'no error raised')

    for args in ((True, 2, 2), (1, 0, 2), (1, 2, -1), (1, 2.0, 2)):
        try:
            GatedDeltaRule(*args)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            _ok(f'GatedDeltaRule {args} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'GatedDeltaRule {args} rejected', False, 'no error raised')


def test_backend_validators_reject_non_string_values():
    """Unhashable / non-string backend names are ValueError, not TypeError.

    ``backend not in {...}`` raises ``TypeError: unhashable type`` for a
    list/dict, bypassing the documented validation contract. The three
    adapters (KDA, Mamba-2, RWKV-7) must reject every non-string value
    with the same actionable ``ValueError`` as an unknown string.
    """
    from ops_kda_backend import validate_kda_backend
    from ops_mamba_backend import validate_mamba_backend
    from ops_rwkv_backend import validate_rwkv_backend

    validators = (
        ('KDA', validate_kda_backend, 'reference'),
        ('Mamba-2', validate_mamba_backend, 'reference'),
        ('RWKV-7', validate_rwkv_backend, 'reference'),
    )
    for name, validate, good in validators:
        assert validate(good) == good
        for bad in (['reference'], {'backend': 'reference'}, None, 1, True):
            try:
                validate(bad)
            except ValueError:
                pass
            except Exception as exc:  # noqa: BLE001 - the point is the type
                _ok(f'{name} backend {bad!r} rejected', False,
                    f'{type(exc).__name__}: {exc}')
            else:
                _ok(f'{name} backend {bad!r} rejected', False,
                    'no error raised')


def test_memory_mixers_reject_coerced_integer_config_fields():
    """RWKV-7 / Mamba-2 config ints are validated, not ``int()``-truncated.

    The constructors used to cast optional geometry fields with ``int(...)``,
    so ``rwkv_head_dim=1.5`` silently became 1 (a different model), a
    numeric string was accepted, and ``True`` became 1. The algebra
    validators are strict; the constructors must hand them the raw values.
    """
    from ops_mamba import Mamba2Mixer
    from ops_mamba_backend import OfficialMamba2Mixer
    from ops_rwkv import Rwkv7Mixer

    rwkv_ok = SimpleNamespace(d_model=64, rwkv_head_dim=16)
    Rwkv7Mixer(rwkv_ok)  # the valid contract still works
    mamba_ok = SimpleNamespace(
        d_model=8, mamba_d_state=4, mamba_d_conv=2, mamba_expand=2,
        mamba_headdim=8, mamba_ngroups=1)
    Mamba2Mixer(mamba_ok)

    cases = (
        ('RWKV head_dim float', Rwkv7Mixer,
         SimpleNamespace(d_model=64, rwkv_head_dim=1.5)),
        ('RWKV head_dim string', Rwkv7Mixer,
         SimpleNamespace(d_model=64, rwkv_head_dim='16')),
        ('Mamba d_state float', Mamba2Mixer,
         SimpleNamespace(d_model=8, mamba_d_state=1.5)),
        ('Mamba d_conv string', Mamba2Mixer,
         SimpleNamespace(d_model=8, mamba_d_conv='4')),
        ('official Mamba d_state float', OfficialMamba2Mixer,
         SimpleNamespace(d_model=8, mamba_d_state=1.5)),
    )
    for name, ctor, cfg in cases:
        try:
            ctor(cfg)
            _ok(f'{name} rejected', False, 'no error raised')
        except ValueError:
            _ok(f'{name} rejected', True, '')
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'{name} rejected', False, f'{type(exc).__name__}: {exc}')

    try:
        Rwkv7Mixer(SimpleNamespace(d_model=64, rwkv_head_dim=16),
                   rwkv_layer_idx=True)
    except ValueError:
        pass
    else:
        raise AssertionError('boolean rwkv_layer_idx was silently accepted')


if __name__ == '__main__':
    test_gated_delta_rule_rejects_geometry_mismatch()
    test_gated_delta_rule_step_rejects_unknown_gate_type()
    test_gated_delta_rule_check_survives_python_optimized()
    test_online_softmax_rejects_non_boolean_mask()
    test_decoding_cache_to_rejects_non_float_storage_dtype()
    test_csa_decode_step_validates_indexer_operands()
    test_kda_chunk_size_requires_an_exact_positive_int()
    test_kda_g_clamp_rejects_positive_infinity_and_nan()
    test_rope_rejects_invalid_dims_bases_and_positions()
    test_decoding_cache_geometry_requires_exact_ints()
    test_online_softmax_chunk_sizes_require_exact_ints()
    test_gate_factory_and_gated_delta_geometry_reject_coercions()
    test_backend_validators_reject_non_string_values()
    test_memory_mixers_reject_coerced_integer_config_fields()
    test_hybrid_config_rejects_coerced_and_nonfinite_fields()
    print('all input-validation tests passed')
