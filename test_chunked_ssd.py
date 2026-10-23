"""Regression tests for the chunked-SSD Mamba-2 path.

``ops_mamba.chunked_mamba2_scan`` is a third evaluation of the Mamba-2 (SSD)
operator, numerically independent from both existing paths:

* ``naive_mamba2_scan`` — sequential scalar recurrence, and
* ``quadratic_mamba2_scan`` — global O(T²) masked-attention dual form.

It splits the sequence into chunks, evaluates each chunk's intra-chunk
outputs with the masked quadratic form restricted to the chunk, and resolves
the inter-chunk recurrence ``S_{n+1} = a_n S_n + b_n`` (scalar per-head
transition ``a_n``, matrix source ``b_n``) with a Hillis-Steele associative
prefix scan over the chunk transitions. Every exponent is anchored at the
state before its chunk starts (log-space gating), so no intra-chunk
exponent differs two large cumulative log-decays.

The tests here pin:

1. **Hand-computed values** — a two-step case with the expected outputs
   written in closed form.
2. **Forward parity** — the chunked path agrees with the sequential
   recurrence and the global dual form at fp64 machine precision, across
   group counts, initial states, chunk sizes (including C=1, C>T, and
   non-divisible T), and at fp32 round-off.
3. **Chunk-size invariance** — every chunk size produces the same values
   (the decomposition is exact; only rounding differs).
4. **Log-space gating stability** — long sequences whose cumulative
   log-decay is huge (up to ~8e4) stay finite and match the sequential
   recurrence; the chunk anchor keeps each exponent bounded.
5. **Contract parity** — dtype passthrough (fp16/fp32/fp64), compute-
   precision states, T=0, ``output_final_state=False``, and identical
   rejection of malformed inputs by the shared validator.
6. **The scan core** — closed-form match against a hand-rolled Python
   prefix loop, the ``ceil(log2 NT)`` depth contract, exact ``S0``
   passthrough for chunk 0, and TorchScript parity.
7. **Gradient parity** — every input gradient (x, dt, B, C, A, D, S0)
   matches the sequential path's graph.
8. **Mixer integration** — ``Mamba2Mixer`` with ``mamba_chunk_size > 0``
   equals the default sequential mixer on the same weights, records its
   ``scan_mode``, and the config validates the knob.

Run via pytest::

    pytest -q test_chunked_ssd.py

Or directly::

    python3 test_chunked_ssd.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_mamba import (  # noqa: E402
    _assoc_scan_chunk_transitions,
    _get_scripted_chunk_transition_scan,
    chunked_mamba2_scan,
    mamba2_scan_boundary_metadata,
    naive_mamba2_scan,
    quadratic_mamba2_scan,
)

FP64_TOL = 1e-11
FP32_TOL = 2e-4
GRAD_TOL = 1e-10


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _rand_inputs(batch, seqlen, nheads, headdim, groups, d_state,
                 dtype=torch.float64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype,
                    generator=gen) * 0.5
    dt = torch.rand(batch, seqlen, nheads, dtype=dtype, generator=gen) * 0.1 \
        + 1e-3
    Bm = torch.randn(batch, seqlen, groups, d_state, dtype=dtype,
                     generator=gen) * 0.5
    Cm = torch.randn(batch, seqlen, groups, d_state, dtype=dtype,
                     generator=gen) * 0.5
    A = -torch.exp(torch.log(torch.arange(
        1, nheads + 1, dtype=dtype)))
    D = torch.linspace(0.5, 1.0, nheads, dtype=dtype)
    return x, dt, Bm, Cm, A, D


def _chunked(x, dt, Bm, Cm, A, D, **kwargs):
    out = chunked_mamba2_scan(x, dt, Bm, Cm, A, D, **kwargs)
    assert isinstance(out, tuple) and len(out) == 2
    return out


# --- hand-computed values -------------------------------------------------

def test_chunked_hand_computed_values():
    print('\nTest: chunked SSD against hand-computed closed forms')
    # B=H=D=N=1, x=[1,1], dt=1, A=-1, B=C=1, D=0, S0=0, one chunk of 2:
    #   y_0 = 1,   y_1 = 1 + exp(-1);   S_2 = 1 + exp(-1).
    x = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    dt = torch.ones(1, 2, 1, dtype=torch.float64)
    Bm = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    Cm = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    A = torch.tensor([-1.0], dtype=torch.float64)
    D = torch.zeros(1, dtype=torch.float64)
    for chunk_size in (1, 2, 5):
        y_ch, s_ch = _chunked(x, dt, Bm, Cm, A, D, chunk_size=chunk_size)
        expect_y = torch.tensor([[[[1.0]], [[1.0 + math.exp(-1.0)]]]],
                                dtype=torch.float64)
        expect_s = torch.tensor([[[[1.0 + math.exp(-1.0)]]]],
                                dtype=torch.float64)
        _ok(f'closed-form output [C={chunk_size}]',
            (y_ch - expect_y).abs().max().item() < 1e-12,
            f'maxdiff={(y_ch - expect_y).abs().max().item():.2e}')
        _ok(f'closed-form final state [C={chunk_size}]',
            (s_ch - expect_s).abs().max().item() < 1e-12,
            f'maxdiff={(s_ch - expect_s).abs().max().item():.2e}')


# --- operator boundary metadata -------------------------------------------

def test_chunked_boundary_metadata():
    print('\nTest: chunk-boundary metadata + exact checkpoint resume')
    cases = [
        (0, 4, 0, 0, 0, (), ()),
        (1, 4, 1, 4, 0, (0,), (1,)),
        (8, 4, 2, 8, 1, (0, 4), (4, 4)),
        (9, 4, 3, 12, 2, (0, 4, 8), (4, 4, 1)),
        (13, 3, 5, 15, 3, (0, 3, 6, 9, 12), (3, 3, 3, 3, 1)),
        (64, 64, 1, 64, 0, (0,), (64,)),
    ]
    for T, C, n, padded, levels, starts, lengths in cases:
        md = mamba2_scan_boundary_metadata(T, chunk_size=C)
        label = f'T={T} C={C}'
        _ok(f'boundary metadata [chunk starts {label}]',
            md['n_chunks'] == n and md['padded_len'] == padded
            and md['pad'] == padded - T and md['scan_levels'] == levels,
            f"n={md['n_chunks']} padded={md['padded_len']} "
            f"levels={md['scan_levels']}")
        _ok(f'boundary metadata [chunk starts/lengths {label}]',
            md['chunk_starts'] == starts
            and md['chunk_lengths'] == lengths
            and md['last_chunk_len'] == (lengths[-1] if lengths else 0),
            f"starts={md['chunk_starts']} lengths={md['chunk_lengths']}")

    # The analytic model and the operator must agree on the boundary.
    import roofline
    for T in (0, 1, 5, 13, 64, 65, 127):
        for C in (1, 3, 8, 16, 64):
            md = mamba2_scan_boundary_metadata(T, chunk_size=C)
            br = roofline.ssd_core_flops_breakdown(
                T, 4, 8, 8, chunk_size=C)
            _ok(f'boundary matches roofline [T={T} C={C}]',
                md['n_chunks'] == br['n_chunks']
                and md['padded_len'] == br['padded_len']
                and md['scan_levels'] == br['scan_levels'],
                f"op=({md['n_chunks']},{md['padded_len']},"
                f"{md['scan_levels']}) model=({br['n_chunks']},"
                f"{br['padded_len']},{br['scan_levels']})")

    for bad in (-1, 1.5, True, None, 10 ** 400):
        try:
            mamba2_scan_boundary_metadata(bad, chunk_size=4)
        except ValueError:
            pass
        else:
            _ok(f'seqlen={bad!r} rejected', False, 'no error raised')
    for bad in (0, -1, 1.5, True, None, 10 ** 400):
        try:
            mamba2_scan_boundary_metadata(8, chunk_size=bad)
        except ValueError:
            pass
        else:
            _ok(f'chunk_size={bad!r} rejected', False, 'no error raised')
    _ok('invalid metadata arguments rejected', True, '-1/1.5/True/None')

    # A chunk start is an exact checkpoint: prefix scan -> carry the state
    # -> suffix scan reproduces the full call to fp round-off. This is the
    # property that makes the metadata useful for chunked prefill instead
    # of merely descriptive.
    T, C = 37, 8
    x, dt, Bm, Cm, A, D = _rand_inputs(2, T, 4, 5, 2, 6, seed=91)
    S0 = torch.randn(2, 4, 5, 6, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(92)) * 0.1
    y_full, s_full = _chunked(x, dt, Bm, Cm, A, D, initial_state=S0,
                              chunk_size=C)
    md = mamba2_scan_boundary_metadata(T, chunk_size=C)
    for start in md['chunk_starts'][1:]:
        y_pre, s_pre = _chunked(x[:, :start], dt[:, :start],
                                Bm[:, :start], Cm[:, :start], A, D,
                                initial_state=S0, chunk_size=C)
        y_suf, s_suf = _chunked(x[:, start:], dt[:, start:],
                                Bm[:, start:], Cm[:, start:], A, D,
                                initial_state=s_pre, chunk_size=C)
        y_diff = max((y_pre - y_full[:, :start]).abs().max().item(),
                     (y_suf - y_full[:, start:]).abs().max().item())
        _ok(f'checkpoint resume output [start={start}]',
            y_diff < 1e-12, f'maxdiff={y_diff:.2e}')
        _ok(f'checkpoint resume final state [start={start}]',
            (s_suf - s_full).abs().max().item() < 1e-12,
            f'maxdiff={(s_suf - s_full).abs().max().item():.2e}')


# --- forward parity -------------------------------------------------------

def test_chunked_forward_parity():
    print('\nTest: chunked == sequential == global dual (forward)')
    batch, nheads, headdim, d_state = 2, 4, 6, 5
    for seqlen in (7, 13, 64):
        for groups in (1, 2, 4):
            x, dt, Bm, Cm, A, D = _rand_inputs(
                batch, seqlen, nheads, headdim, groups, d_state,
                seed=100 + groups + seqlen)
            S0 = (torch.randn(batch, nheads, headdim, d_state,
                              dtype=torch.float64,
                              generator=torch.Generator().manual_seed(
                                  200 + groups)) * 0.1)
            y_rec, s_rec = naive_mamba2_scan(
                x, dt, Bm, Cm, A, D, initial_state=S0)
            y_dual, s_dual = quadratic_mamba2_scan(
                x, dt, Bm, Cm, A, D, initial_state=S0)
            for chunk_size in (1, 3, 16, seqlen, seqlen + 5):
                y_ch, s_ch = _chunked(
                    x, dt, Bm, Cm, A, D, initial_state=S0,
                    chunk_size=chunk_size)
                dy = max((y_ch - y_rec).abs().max().item(),
                         (y_ch - y_dual).abs().max().item())
                ds = max((s_ch - s_rec).abs().max().item(),
                         (s_ch - s_dual).abs().max().item())
                label = f'T={seqlen} groups={groups} C={chunk_size}'
                _ok(f'outputs match [fwd {label}]', dy < FP64_TOL,
                    f'maxdiff={dy:.2e}')
                _ok(f'final states match [fwd {label}]', ds < FP64_TOL,
                    f'maxdiff={ds:.2e}')

    # fp32 parity against the sequential recurrence.
    x, dt, Bm, Cm, A, D = _rand_inputs(
        2, 41, 4, 8, 2, 7, dtype=torch.float32, seed=555)
    y_rec, s_rec = naive_mamba2_scan(x, dt, Bm, Cm, A, D)
    y_ch, s_ch = _chunked(x, dt, Bm, Cm, A, D, chunk_size=16)
    _ok('fp32 outputs match',
        (y_ch - y_rec).abs().max().item() < FP32_TOL,
        f'maxdiff={(y_ch - y_rec).abs().max().item():.2e}')
    _ok('fp32 states match',
        (s_ch - s_rec).abs().max().item() < FP32_TOL,
        f'maxdiff={(s_ch - s_rec).abs().max().item():.2e}')


def test_chunked_chunk_size_invariance():
    print('\nTest: the decomposition is invariant to chunk_size')
    x, dt, Bm, Cm, A, D = _rand_inputs(
        2, 97, 4, 8, 2, 6, seed=31)
    S0 = torch.randn(2, 4, 8, 6, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(32)) * 0.1
    y_ref, s_ref = _chunked(x, dt, Bm, Cm, A, D, initial_state=S0,
                            chunk_size=1)
    for chunk_size in (2, 3, 7, 16, 31, 64, 96, 128):
        y_ch, s_ch = _chunked(x, dt, Bm, Cm, A, D, initial_state=S0,
                              chunk_size=chunk_size)
        dy = (y_ch - y_ref).abs().max().item()
        ds = (s_ch - s_ref).abs().max().item()
        _ok(f'C=1 == C={chunk_size}', max(dy, ds) < FP64_TOL,
            f'y={dy:.2e} s={ds:.2e}')


# --- log-space gating stability -------------------------------------------

def test_chunked_log_space_gating_stability():
    print('\nTest: log-space gating stays stable for huge cumulative decay')
    torch.manual_seed(77)
    # Per-step decay is small but T is large: the cumulative log-decay
    # reaches ~-4096 while every CHUNK-local exponent stays bounded by the
    # chunk's own decay budget (64 * dt * |A| <= 128).
    batch, seqlen, nheads, headdim, d_state = 1, 4096, 2, 8, 8
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float64) * 0.5
    dt = torch.full((batch, seqlen, nheads), 0.01, dtype=torch.float64)
    Bm = torch.randn(batch, seqlen, 1, d_state, dtype=torch.float64) * 0.5
    Cm = torch.randn(batch, seqlen, 1, d_state, dtype=torch.float64) * 0.5
    A = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    D = torch.linspace(0.5, 1.0, nheads, dtype=torch.float64)
    y_rec, s_rec = naive_mamba2_scan(x, dt, Bm, Cm, A, D)
    y_ch, s_ch = _chunked(x, dt, Bm, Cm, A, D, chunk_size=64)
    _ok('fp64 huge-decay outputs finite and matching',
        bool(torch.isfinite(y_ch).all())
        and (y_ch - y_rec).abs().max().item() < 1e-11,
        f'maxdiff={(y_ch - y_rec).abs().max().item():.2e}')
    _ok('fp64 huge-decay final state matching',
        (s_ch - s_rec).abs().max().item() < 1e-11,
        f'maxdiff={(s_ch - s_rec).abs().max().item():.2e}')

    # fp32: no cancellation blow-up (exponents are anchored per chunk).
    y_ch32, s_ch32 = _chunked(
        x.float(), dt.float(), Bm.float(), Cm.float(), A.float(), D.float(),
        chunk_size=64)
    y_rec32, s_rec32 = naive_mamba2_scan(
        x.float(), dt.float(), Bm.float(), Cm.float(), A.float(), D.float())
    _ok('fp32 huge-decay outputs finite and matching',
        bool(torch.isfinite(y_ch32).all())
        and (y_ch32 - y_rec32).abs().max().item() < FP32_TOL,
        f'maxdiff={(y_ch32 - y_rec32).abs().max().item():.2e}')

    # Very strong decay (G down to ~-8e4): the sequential state underflows
    # to zero; the chunked path must stay finite and do the same.
    dt_strong = torch.ones(batch, seqlen, nheads, dtype=torch.float64)
    A_strong = torch.tensor([-10.0, -20.0], dtype=torch.float64)
    y_strong, s_strong = _chunked(
        x, dt_strong, Bm, Cm, A_strong, D, chunk_size=128)
    y_rec_s, s_rec_s = naive_mamba2_scan(x, dt_strong, Bm, Cm, A_strong, D)
    _ok('extreme decay finite and matching',
        bool(torch.isfinite(y_strong).all())
        and bool(torch.isfinite(s_strong).all())
        and (y_strong - y_rec_s).abs().max().item() < 1e-12,
        f'maxdiff={(y_strong - y_rec_s).abs().max().item():.2e}')


# --- contract parity ------------------------------------------------------

def test_chunked_contract_parity():
    print('\nTest: shared input/output/dtype contract')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 9, 4, 5, 2, 3
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=7)

    # T=0: zero-length output, state passthrough/zeros.
    y_ch, s_ch = _chunked(x[:, :0], dt[:, :0], Bm[:, :0], Cm[:, :0], A, D,
                          chunk_size=4)
    _ok('T=0 output shape [B,0,H,D]',
        tuple(y_ch.shape) == (batch, 0, nheads, headdim), f'{tuple(y_ch.shape)}')
    S0 = torch.randn(batch, nheads, headdim, d_state, dtype=torch.float64)
    y_ch, s_ch = _chunked(x[:, :0], dt[:, :0], Bm[:, :0], Cm[:, :0], A, D,
                          initial_state=S0, chunk_size=4)
    _ok('T=0 with initial state passes it through',
        torch.equal(s_ch, S0), '')

    # output_final_state=False.
    y_ch, s_ch = _chunked(x, dt, Bm, Cm, A, D, output_final_state=False,
                          chunk_size=4)
    _ok('output_final_state=False returns None', s_ch is None, '')

    # fp32 / fp16 dtype passthrough with compute-precision states.
    for dtype, state_dtype in ((torch.float32, torch.float32),
                               (torch.float16, torch.float32)):
        xd, dtd, Bd, Cd, Ad, Dd = _rand_inputs(
            batch, seqlen, nheads, headdim, groups, d_state,
            dtype=dtype, seed=7)
        y_ch, s_ch = _chunked(xd, dtd, Bd, Cd, Ad, Dd, chunk_size=4)
        _ok(f'{dtype} output dtype preserved, states {state_dtype}',
            y_ch.dtype == dtype and s_ch.dtype == state_dtype,
            f'y={y_ch.dtype} s={s_ch.dtype}')


def _raises_all(x, dt, Bm, Cm, A, D, label, **kwargs):
    raised = []
    for fn in (naive_mamba2_scan, quadratic_mamba2_scan):
        try:
            fn(x, dt, Bm, Cm, A, D, **kwargs)
        except ValueError:
            raised.append(True)
        except Exception as e:
            print(f'    {fn.__name__}: unexpected {type(e).__name__}: {e}')
            raised.append(False)
        else:
            raised.append(False)
    try:
        chunked_mamba2_scan(x, dt, Bm, Cm, A, D, chunk_size=4, **kwargs)
    except ValueError:
        raised.append(True)
    except Exception as e:
        print(f'    chunked_mamba2_scan: unexpected {type(e).__name__}: {e}')
        raised.append(False)
    else:
        raised.append(False)
    _ok(f'all reject {label}', all(raised), f'{raised}')


def test_chunked_validation_rejections():
    print('\nTest: malformed inputs rejected identically')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 6, 4, 3, 2, 4
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=3)
    S0 = torch.randn(batch, nheads, headdim, d_state, dtype=torch.float64)

    _raises_all(x[0], dt, Bm, Cm, A, D, 'x rank != 4')
    _raises_all(x, dt[:, :, :nheads - 1], Bm, Cm, A, D,
                'dt head-count mismatch')
    _raises_all(x, dt, Bm[:1], Cm[:1], A, D, 'B/C batch mismatch')
    _raises_all(x, dt, Bm[:, :seqlen - 1], Cm[:, :seqlen - 1], A, D,
                'B/C seqlen mismatch')
    _raises_all(x, dt, Bm, Cm[:, :, :, :d_state - 1], A, D,
                'C shape != B shape')
    B3 = torch.randn(batch, seqlen, 3, d_state, dtype=torch.float64)
    _raises_all(x, dt, B3, B3, A, D, 'nheads % ngroups != 0')
    _raises_all(x, dt, Bm, Cm, A[:-1], D, 'A length mismatch')
    _raises_all(x, dt, Bm, Cm, A, D[:-1], 'D length mismatch')
    _raises_all(x, dt, Bm, Cm, A, D, 'initial_state rank',
                initial_state=S0[0])
    _raises_all(x, dt, Bm, Cm, A, D, 'initial_state shape',
                initial_state=S0[:, :, :, :d_state - 1])
    # chunk_size contract (chunked path only).
    for bad in (0, -1, 1.5, True, None, 10 ** 400):
        try:
            chunked_mamba2_scan(x, dt, Bm, Cm, A, D, chunk_size=bad)
        except ValueError:
            pass
        else:
            _ok(f'chunk_size={bad!r} rejected', False, 'no error raised')
    _ok('all bad chunk_size values rejected', True,
        '0,-1,1.5,True,None,10**400')


# --- scan core ------------------------------------------------------------

def _reference_scalar_scan(a, b, S0):
    """Textbook sequential reference for the scalar-affine prefix scan."""
    BH, NT = a.shape
    S = S0
    S_in = []
    for n in range(NT):
        S_in.append(S)
        S = a[:, n].unsqueeze(-1).unsqueeze(-1) * S + b[:, n]
    return torch.stack(S_in, dim=1), S


def test_assoc_scan_core_depth_and_scripted():
    print('\nTest: chunk-transition scan core (closed form, depth, script)')
    torch.manual_seed(79)
    for NT in (1, 2, 3, 5, 16, 17):
        for BH, Dd, Nd in ((2, 4, 3), (3, 2, 5)):
            a = torch.rand(BH, NT, dtype=torch.float64) * 0.5 + 0.5
            b = torch.randn(BH, NT, Dd, Nd, dtype=torch.float64) * 0.2
            S0 = torch.randn(BH, Dd, Nd, dtype=torch.float64) * 0.2
            S_in, S_end, levels = _assoc_scan_chunk_transitions(a, b, S0)
            ref_in, ref_end = _reference_scalar_scan(a, b, S0)
            expected_levels = 0 if NT == 1 else math.ceil(math.log2(NT))
            assert levels == expected_levels, (NT, levels)
            assert torch.equal(S_in[:, 0], S0), 'chunk 0 must see S0 exactly'
            _ok(f'scan matches loop [NT={NT} BH={BH}]',
                max((S_in - ref_in).abs().max().item(),
                    (S_end - ref_end).abs().max().item()) < 1e-12,
                f'maxdiff={max((S_in - ref_in).abs().max().item(), (S_end - ref_end).abs().max().item()):.2e}')
            _ok(f'log depth [NT={NT}]', levels == expected_levels,
                f'levels={levels}')
    scripted = _get_scripted_chunk_transition_scan()
    if scripted is not None:
        eager = _assoc_scan_chunk_transitions(a, b, S0)
        out = scripted(a, b, S0)
        _ok('scripted core is bit-identical to eager',
            torch.equal(out[0], eager[0]) and torch.equal(out[1], eager[1])
            and out[2] == eager[2], '')
    else:  # pragma: no cover - torch.jit always available here
        print('  [SKIP] torch.jit.script unavailable; eager core only')


# --- gradient parity ------------------------------------------------------

def test_chunked_gradient_parity():
    print('\nTest: chunked == sequential (gradients)')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 23, 4, 5, 2, 4
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=42)
    S0 = (torch.randn(batch, nheads, headdim, d_state,
                      dtype=torch.float64,
                      generator=torch.Generator().manual_seed(43)) * 0.1)
    base = {'x': x, 'dt': dt, 'B': Bm, 'C': Cm, 'A': A, 'D': D, 'S0': S0}
    names = list(base)

    def _leaves():
        return {k: v.clone().requires_grad_(True) for k, v in base.items()}

    def _loss(y, s):
        wy = torch.linspace(0.5, 1.5, y.numel(),
                            dtype=torch.float64).reshape(y.shape)
        ws = torch.linspace(1.5, 0.5, s.numel(),
                            dtype=torch.float64).reshape(s.shape)
        return (y * wy).sum() + (s * ws).sum()

    grads = []
    for fn, kw in ((naive_mamba2_scan, {}),
                   (chunked_mamba2_scan, {'chunk_size': 7})):
        leaf = _leaves()
        y, s = fn(leaf['x'], leaf['dt'], leaf['B'], leaf['C'], leaf['A'],
                  leaf['D'], initial_state=leaf['S0'], **kw)
        _loss(y, s).backward()
        grads.append(leaf)

    for name in names:
        g0, g1 = grads[0][name].grad, grads[1][name].grad
        finite = bool(torch.isfinite(g1).all())
        nonzero = bool(g1.abs().max().item() > 0)
        diff = (g0 - g1).abs().max().item()
        _ok(f'grad[{name}] matches and is finite/nonzero',
            finite and nonzero and diff < GRAD_TOL,
            f'maxdiff={diff:.2e} finite={finite} nonzero={nonzero}')


# --- mixer integration ----------------------------------------------------

def _mamba_cfg(**overrides):
    from ops_fused import HybridConfig
    kwargs = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16, memory_type='mamba2',
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=16, mamba_backend='reference',
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def test_mixer_chunked_scan_integration():
    print('\nTest: Mamba2Mixer honours mamba_chunk_size end to end')
    from ops_mamba import Mamba2Mixer
    torch.manual_seed(0)
    cfg_seq = _mamba_cfg()
    seq = Mamba2Mixer(cfg_seq).double().eval()
    _ok('default mixer records sequential scan_mode',
        seq.scan_mode == 'sequential', seq.scan_mode)
    for chunk_size in (1, 7, 64):
        cfg_ch = _mamba_cfg(mamba_chunk_size=chunk_size)
        chunked = Mamba2Mixer(cfg_ch).double().eval()
        chunked.load_state_dict(seq.state_dict())
        _ok(f'chunked mixer records scan_mode [C={chunk_size}]',
            chunked.scan_mode == 'chunked', chunked.scan_mode)
        x = torch.randn(2, 17, 32, dtype=torch.float64)
        with torch.no_grad():
            out_seq, s_seq, conv_seq = seq.forward_functional(x, None, None)
            out_ch, s_ch, conv_ch = chunked.forward_functional(x, None, None)
        diff = (out_seq - out_ch).abs().max().item()
        sdiff = (s_seq - s_ch).abs().max().item()
        _ok(f'mixer output invariant to the scan [C={chunk_size}]',
            diff < 1e-9, f'maxdiff={diff:.2e}')
        _ok(f'mixer state invariant to the scan [C={chunk_size}]',
            sdiff < 1e-9, f'maxdiff={sdiff:.2e}')
        _ok(f'conv state scan-independent [C={chunk_size}]',
            torch.equal(conv_seq, conv_ch), '')

    # Config validation: bad values and the official-backend contradiction.
    for bad in (-1, 1.5, True, 'x'):
        try:
            _mamba_cfg(mamba_chunk_size=bad)
        except (TypeError, ValueError):
            continue
        _ok(f'HybridConfig rejects mamba_chunk_size={bad!r}', False,
            'no error')
    _ok('HybridConfig rejects bad mamba_chunk_size', True, '-1,1.5,True,x')
    try:
        _mamba_cfg(mamba_chunk_size=16, mamba_backend='official')
    except ValueError:
        _ok('HybridConfig rejects official + chunked scan', True, '')
    else:
        _ok('HybridConfig rejects official + chunked scan', False,
            'no error')
    _ok('chunked config accepted with auto/reference',
        _mamba_cfg(mamba_chunk_size=16, mamba_backend='auto').mamba_chunk_size
        == 16, '')


_TEST_FUNCS = [
    test_chunked_hand_computed_values,
    test_chunked_forward_parity,
    test_chunked_chunk_size_invariance,
    test_chunked_log_space_gating_stability,
    test_chunked_contract_parity,
    test_chunked_validation_rejections,
    test_assoc_scan_core_depth_and_scripted,
    test_chunked_gradient_parity,
    test_mixer_chunked_scan_integration,
]


def main() -> int:
    print('=' * 70)
    print('Chunked SSD (associative-scan Mamba-2 path) Regression Tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f'  [FAIL] {fn.__name__}: AssertionError: {e}')
            results.append(False)
        except Exception as e:
            print(f'  [CRASH] {fn.__name__}: {type(e).__name__}: {e}')
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
