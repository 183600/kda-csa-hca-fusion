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
        {'n_kda': '3'},
        {'n_csa': 1.5},
        {'n_hca': True},
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
    from ops_kda_autograd import custom_recurrent_kda
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
            (custom_recurrent_kda, {}),
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


def test_gated_delta_rule_rejects_nonfinite_init_and_clamps():
    """GatedDeltaRule parameters are validated before they poison the state.

    ``g_init``/``beta_init`` are copied into nn.Parameters; a NaN there made
    every forward output NaN. ``g_clamp_max=+inf`` lets ``exp(g)`` overflow
    instead of reporting the invalid bound, and a reversed clamp range
    silently pins the gate to one endpoint.
    """
    from ops_gated_attention import GatedDeltaRule

    GatedDeltaRule(2, 4, 5)  # valid defaults still construct
    cases = (
        {'g_init': float('nan')},
        {'beta_init': float('inf')},
        {'g_clamp_min': float('nan')},
        {'g_clamp_min': float('inf')},
        {'g_clamp_max': float('inf')},
        {'g_clamp_min': 1.0, 'g_clamp_max': -1.0},
    )
    for kwargs in cases:
        try:
            GatedDeltaRule(2, 4, 5, **kwargs)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'GatedDeltaRule {kwargs} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'GatedDeltaRule {kwargs} rejected', False,
                'no error raised')
    # -inf is the documented "disable the lower clamp" sentinel.
    layer = GatedDeltaRule(2, 4, 5, g_clamp_min=float('-inf'))
    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 5)
    o, _ = layer(q, k, v)
    _ok('finite -inf lower clamp still runs', torch.isfinite(o).all(), '')


def test_benchmark_chunk_shape_guard_survives_python_optimized():
    """The chunked-KDA trim check must not be a bare ``assert``.

    ``python -O`` strips ``assert`` statements, which would silently accept
    a chunk-path regression that returns the padded length instead of T and
    corrupt every benchmark row. The guard is an explicit RuntimeError.
    """
    import run_benchmark as rb

    original = rb.kda_forward

    def _wrong_shape(*args, **kwargs):
        return torch.zeros(1, 4, 1, 2), torch.zeros(1, 1, 2, 2)

    rb.kda_forward = _wrong_shape
    try:
        fn = rb.bench_kda_chunk(1, 5, 1, 2, 2, 'cpu')
        try:
            fn()
        except RuntimeError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok('chunk shape guard rejected the padded shape', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok('chunk shape guard rejected the padded shape', False,
                'no error raised')
    finally:
        rb.kda_forward = original


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


def test_t0_memory_mixers_reject_wrong_full_state_geometry():
    """T=0 must not bypass the [B, H, K, V] state contract.

    These branches only checked ``state.shape[0] == B`` and echoed a
    same-batch tensor back, so a state from a different geometry was
    accepted on an empty call and could re-enter a later decode step.
    """
    from ops_fused import KDAHybridLayer
    from ops_mamba import Mamba2Mixer
    from ops_rwkv import Rwkv7Mixer

    kda_cfg = SimpleNamespace(
        d_model=8, n_heads_qk=2, head_dim_k=3, head_dim_v=4, n_heads_v=2,
        kda_decay_init='default', kda_decay_scale=0.1, kda_chunk_size=0,
        kda_backend='reference', kda_recompute_attn=False,
        kda_assoc_scan=False, n_kda=1, n_csa=0, n_hca=0, layout=None)
    kda = KDAHybridLayer(kda_cfg)
    try:
        kda.forward_functional(torch.zeros(1, 0, 8),
                               torch.zeros(1, 2, 5, 5), None)
    except ValueError:
        pass
    else:
        raise AssertionError('KDA T=0 accepted a wrong K/V state shape')

    mamba_cfg = SimpleNamespace(
        d_model=8, mamba_d_state=4, mamba_d_conv=2, mamba_expand=2,
        mamba_headdim=8, mamba_ngroups=1)
    mamba = Mamba2Mixer(mamba_cfg)
    try:
        mamba.forward_functional(
            torch.zeros(1, 0, 8),
            torch.zeros(1, mamba.nheads, mamba.headdim, mamba.d_state + 1),
            None)
    except ValueError:
        pass
    else:
        raise AssertionError('Mamba-2 T=0 accepted a wrong state shape')

    rwkv = Rwkv7Mixer(SimpleNamespace(d_model=64, rwkv_head_dim=16))
    try:
        rwkv.forward_functional(
            torch.zeros(1, 0, 64),
            torch.zeros(1, rwkv.n_heads, rwkv.head_dim, rwkv.head_dim + 1),
            None)
    except ValueError:
        pass
    else:
        raise AssertionError('RWKV-7 T=0 accepted a wrong state shape')

    # The short-conv / shift state is part of the same per-layer contract:
    # an empty chunk must not echo a stale geometry back to the caller.
    x_kda = torch.zeros(1, 0, 8)
    kda_lb = kda.short_conv.kernel_size[0] - 1
    for bad in (torch.zeros(1, kda_lb + 1, 8),
                torch.zeros(1, kda_lb, 7)):
        try:
            kda.forward_functional(x_kda, None, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f'KDA T=0 accepted a wrong lookback shape {tuple(bad.shape)}')
    kda.forward_functional(x_kda, None, torch.zeros(1, kda_lb, 8))
    try:
        kda.forward_functional(torch.randn(1, 1, 8), None,
                               torch.zeros(1, kda_lb + 1, 8))
    except ValueError:
        pass
    else:
        raise AssertionError('KDA T=1 accepted a wrong lookback shape')

    x_mamba = torch.zeros(1, 0, 8)
    mamba_conv = mamba.d_conv - 1
    for bad in (torch.zeros(1, mamba.d_inner, mamba_conv + 1),
                torch.zeros(1, mamba.d_inner + 1, mamba_conv)):
        try:
            mamba.forward_functional(x_mamba, None, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f'Mamba-2 T=0 accepted a wrong conv state {tuple(bad.shape)}')
    mamba.forward_functional(
        x_mamba, None, torch.zeros(1, mamba.d_inner, mamba_conv))
    try:
        mamba.forward_functional(
            torch.randn(1, 1, 8), None,
            torch.zeros(1, mamba.d_inner, mamba_conv + 1))
    except ValueError:
        pass
    else:
        raise AssertionError('Mamba-2 T=1 accepted a wrong conv state')

    x_rwkv = torch.zeros(1, 0, 64)
    for bad in (torch.zeros(1, 2, 64), torch.zeros(1, 1, 63)):
        try:
            rwkv.forward_functional(x_rwkv, None, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f'RWKV-7 T=0 accepted a wrong shift state {tuple(bad.shape)}')
    rwkv.forward_functional(x_rwkv, None, torch.zeros(1, 1, 64))


def test_csa_hca_geometry_requires_exact_ints():
    """Bool/float geometry must not silently mean a different operator.

    ``m=True`` used to mean ``m=1`` (one token per compressed block) and a
    float chunk factor reached ``view``/``topk`` with an opaque error. The
    public CSA/HCA operators now share the repository's exact-int contract;
    valid integer geometry is unchanged.
    """
    from ops_csa import (
        csa_compress_kv, csa_compress_kv_overlapped,
        csa_lightning_indexer, naive_csa,
    )
    from ops_hca import naive_hca

    B, T, c, c_I = 1, 4, 3, 2
    C = torch.randn(B, T, c)
    Z = torch.randn(B, T, c)
    B_pos = torch.randn(2, c)
    for bad in (True, 1.0, '2', None):
        for fn, args in (
                (csa_compress_kv, (C, Z, B_pos, bad)),
                (csa_compress_kv_overlapped,
                 (C, C, Z, Z, B_pos, B_pos, bad)),
        ):
            try:
                fn(*args)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(
                    f'{fn.__name__} accepted m={bad!r}')
    # The valid integer path still works (T must be divisible by m).
    csa_compress_kv(C, Z, B_pos, 2)
    csa_compress_kv_overlapped(C, C, Z, Z, B_pos, B_pos, 2)

    q_idx = torch.randn(1, 2, 1, c_I)
    k_idx = torch.randn(1, 2, c_I)
    w_idx = torch.randn(1, 2, 1)
    for bad in (True, 1.0, '1', None):
        try:
            csa_lightning_indexer(q_idx, k_idx, w_idx, topk=bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(
                f'csa_lightning_indexer accepted topk={bad!r}')
    csa_lightning_indexer(q_idx, k_idx, w_idx, topk=1)

    H = torch.zeros(1, 3, 4)
    for name, fn, kwargs in (
            ('naive_csa', naive_csa,
             dict(m=2, topk=1, nh=1, nIh=1, c=2, c_I=2, dc=2)),
            ('naive_hca', naive_hca,
             dict(m2=2, nh=1, c=2, dc=2)),
    ):
        key = 'm' if name == 'naive_csa' else 'm2'
        for bad in (True, 2.0, '2'):
            patch = dict(kwargs)
            patch[key] = bad
            try:
                fn(H, *(None,) * (13 if name == 'naive_csa' else 5),
                   **patch)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f'{name} accepted {key}={bad!r}')


def test_csa_hca_validate_rope_base_even_when_rope_is_disabled():
    """A disabled rotation must not excuse an invalid RoPE base.

    ``naive_csa``/``naive_hca`` only reached the downstream ``partial_rope``
    validator when ``rope_dim`` was active. With ``rope_dim=0`` a NaN/Inf
    base was accepted, and a string/bool raised a bare TypeError. The cache
    constructors validate the base unconditionally, so the direct operators
    must too.
    """
    from ops_csa import naive_csa
    from ops_hca import naive_hca

    c, d, c_I, m, nh, nIh, dc = 4, 8, 2, 2, 1, 1, 4
    W = torch.randn(c, d)
    W_idx = torch.randn(c_I, d)
    B_pos = torch.randn(m, c)
    B_idx = torch.randn(m, c_I)
    W_DQ = torch.randn(dc, d)
    W_UQ = torch.randn(c * nh, dc)
    W_IUQ = torch.randn(c_I * nIh, dc)
    W_w = torch.randn(nIh, d)
    H = torch.randn(1, 3, d)
    for bad in (float('nan'), float('inf'), '10000', True):
        try:
            naive_csa(
                H, W, W, W, W, B_pos, B_pos, W_DQ, W_UQ, W_IUQ, W_w,
                W_idx, W_idx, B_idx,
                m=m, topk=1, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
                rope_dim=0, rope_base=bad)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'naive_csa rope_base={bad!r} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'naive_csa rope_base={bad!r} rejected', False,
                'no error raised')
        try:
            naive_hca(
                H, W, W, B_pos, W_DQ, W_UQ,
                m2=m, nh=nh, c=c, dc=dc,
                rope_dim=0, rope_base=bad)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - the point is the type
            _ok(f'naive_hca rope_base={bad!r} rejected', False,
                f'{type(exc).__name__}: {exc}')
        else:
            _ok(f'naive_hca rope_base={bad!r} rejected', False,
                'no error raised')
    # The valid disabled-RoPE path (and an explicit valid base) still works.
    naive_csa(H, W, W, W, W, B_pos, B_pos, W_DQ, W_UQ, W_IUQ, W_w,
              W_idx, W_idx, B_idx,
              m=m, topk=1, nh=nh, nIh=nIh, c=c, c_I=c_I, dc=dc,
              rope_dim=0, rope_base=10000.0)
    naive_hca(H, W, W, B_pos, W_DQ, W_UQ,
              m2=m, nh=nh, c=c, dc=dc, rope_dim=0, rope_base=10000.0)


if __name__ == '__main__':
    test_gated_delta_rule_rejects_geometry_mismatch()
    test_gated_delta_rule_step_rejects_unknown_gate_type()
    test_t0_memory_mixers_reject_wrong_full_state_geometry()
    test_csa_hca_geometry_requires_exact_ints()
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
    test_gated_delta_rule_rejects_nonfinite_init_and_clamps()
    test_benchmark_chunk_shape_guard_survives_python_optimized()
    test_backend_validators_reject_non_string_values()
    test_memory_mixers_reject_coerced_integer_config_fields()
    test_hybrid_config_rejects_coerced_and_nonfinite_fields()
    test_csa_hca_validate_rope_base_even_when_rope_is_disabled()
    print('all input-validation tests passed')
