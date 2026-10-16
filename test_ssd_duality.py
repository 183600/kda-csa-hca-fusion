"""SSD-duality regression tests for the Mamba-2 reference scan.

``ops_mamba.quadratic_mamba2_scan`` evaluates the state-space-duality dual
form of the recurrence implemented by ``ops_mamba.naive_mamba2_scan``: the
same operator written as causally-masked quadratic attention with a
semi-separable pairwise decay (arXiv:2405.21060 Sections 3-4), computed
from cumulative log-decays instead of a sequential state recurrence.

The two code paths share only the input contract (one validator), so their
agreement is an independent correctness oracle. This matters because the
other Mamba-2 correctness chains cannot catch a shared semantic error:

* the prefill/step/segmented-prefill tests all run the SAME recurrence, and
* the official ``mamba_ssm`` backend parity test self-skips on CPU boxes.

The tests here pin, to fp64 machine precision:

1. **Hand-computed values** — a two-step case and an initial-state decay
   case where the expected outputs are written down in closed form.
2. **Forward parity** — outputs and final states agree across group
   counts (1/2/4), with and without a nonzero initial state, at T=13 and
   T=64.
3. **Contract parity** — dtype passthrough (fp32/fp16 inputs, fp32
   states), the T=0 edge case, ``output_final_state=False``, and identical
   rejection of malformed inputs by the shared validator.
4. **Gradient parity** — every input gradient (x, dt, B, C, A, D, S0)
   agrees between the two computational graphs.

Run via pytest::

    pytest -q test_ssd_duality.py

Or directly::

    python3 test_ssd_duality.py
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import ops_mamba  # noqa: E402
from ops_mamba import naive_mamba2_scan, quadratic_mamba2_scan  # noqa: E402


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


def _scan_pair(x, dt, Bm, Cm, A, D, **kwargs):
    y_rec, s_rec = naive_mamba2_scan(x, dt, Bm, Cm, A, D, **kwargs)
    y_dual, s_dual = quadratic_mamba2_scan(x, dt, Bm, Cm, A, D, **kwargs)
    return y_rec, s_rec, y_dual, s_dual


# --- hand-computed values -------------------------------------------------

def test_ssd_hand_computed_values():
    print('\nTest: SSD dual form against hand-computed closed forms')
    # Case 1: B=H=D=N=1, x=[1,1], dt=1, A=-1, B=C=1, D=0, S0=0.
    #   S_0 = 1 (after token 0), y_0 = 1
    #   S_1 = exp(-1)*S_0 + 1 = 1 + exp(-1), y_1 = 1 + exp(-1)
    x = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    dt = torch.ones(1, 2, 1, dtype=torch.float64)
    Bm = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    Cm = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    A = torch.tensor([-1.0], dtype=torch.float64)
    D = torch.zeros(1, dtype=torch.float64)
    _, s_rec, y_dual, s_dual = _scan_pair(x, dt, Bm, Cm, A, D)
    expect_y = torch.tensor([[[[1.0]], [[1.0 + math.exp(-1.0)]]]],
                            dtype=torch.float64)
    expect_s = torch.tensor([[[[1.0 + math.exp(-1.0)]]]], dtype=torch.float64)
    _ok('dual matches closed-form output',
        (y_dual - expect_y).abs().max().item() < 1e-12,
        f'maxdiff={(y_dual - expect_y).abs().max().item():.2e}')
    _ok('dual matches closed-form final state',
        (s_dual - expect_s).abs().max().item() < 1e-12,
        f'maxdiff={(s_dual - expect_s).abs().max().item():.2e}')
    _ok('recurrent matches the same closed form',
        (s_rec - expect_s).abs().max().item() < 1e-12, '')

    # Case 2: initial-state decay with x=0: y_t = exp(-(t+1)) * S0.
    T = 5
    x0 = torch.zeros(1, T, 1, 1, dtype=torch.float64)
    dt1 = torch.ones(1, T, 1, dtype=torch.float64)
    B1 = torch.ones(1, T, 1, 1, dtype=torch.float64)
    C1 = torch.ones(1, T, 1, 1, dtype=torch.float64)
    S0 = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    _, s_rec0, y_dual0, s_dual0 = _scan_pair(
        x0, dt1, B1, C1, A, D, initial_state=S0)
    expect_decay = torch.tensor(
        [[[[math.exp(-(t + 1))]] for t in range(T)]], dtype=torch.float64)
    _ok('initial-state term decays as exp(sum dt*A) in both views',
        (y_dual0 - expect_decay).abs().max().item() < 1e-12
        and (s_dual0 - math.exp(-T)).abs().max().item() < 1e-12,
        f'y={(y_dual0 - expect_decay).abs().max().item():.2e} '
        f's={(s_dual0 - math.exp(-T)).abs().max().item():.2e}')


# --- forward parity -------------------------------------------------------

def test_ssd_forward_parity():
    print('\nTest: recurrent scan == dual form (forward)')
    batch, seqlen, nheads, headdim, d_state = 2, 13, 4, 6, 5
    for groups in (1, 2, 4):
        for with_init in (False, True):
            x, dt, Bm, Cm, A, D = _rand_inputs(
                batch, seqlen, nheads, headdim, groups, d_state,
                seed=10 + groups)
            S0 = (torch.randn(batch, nheads, headdim, d_state,
                              dtype=torch.float64,
                              generator=torch.Generator().manual_seed(
                                  20 + groups)) * 0.1
                  if with_init else None)
            y_rec, s_rec, y_dual, s_dual = _scan_pair(
                x, dt, Bm, Cm, A, D, initial_state=S0)
            y_diff = (y_rec - y_dual).abs().max().item()
            s_diff = (s_rec - s_dual).abs().max().item()
            label = f'groups={groups} init={with_init}'
            _ok(f'outputs identical [{label}]', y_diff < 1e-12,
                f'maxdiff={y_diff:.2e}')
            _ok(f'final states identical [{label}]', s_diff < 1e-12,
                f'maxdiff={s_diff:.2e}')

    # Longer sequence: checks the cumulative-sum formulation does not
    # accumulate a drift relative to the recurrence.
    x, dt, Bm, Cm, A, D = _rand_inputs(
        2, 64, 4, 8, 2, 7, seed=99)
    y_rec, s_rec, y_dual, s_dual = _scan_pair(x, dt, Bm, Cm, A, D)
    _ok('T=64 outputs identical',
        (y_rec - y_dual).abs().max().item() < 1e-11,
        f'maxdiff={(y_rec - y_dual).abs().max().item():.2e}')
    _ok('T=64 final states identical',
        (s_rec - s_dual).abs().max().item() < 1e-11,
        f'maxdiff={(s_rec - s_dual).abs().max().item():.2e}')


# --- contract parity ------------------------------------------------------

def test_ssd_contract_parity():
    print('\nTest: shared input/output/dtype contract')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 9, 4, 5, 2, 3
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=7)

    # T=0: zero-length output, state passthrough/zeros.
    x0 = x[:, :0]
    y_rec, s_rec, y_dual, s_dual = _scan_pair(
        x0, dt[:, :0], Bm[:, :0], Cm[:, :0], A, D)
    _ok('T=0 output shape [B,0,H,D]',
        tuple(y_rec.shape) == (batch, 0, nheads, headdim)
        and torch.equal(y_rec, y_dual), f'{tuple(y_rec.shape)}')
    _ok('T=0 states identical', torch.equal(s_rec, s_dual), '')
    S0 = torch.randn(batch, nheads, headdim, d_state, dtype=torch.float64)
    y_rec, s_rec, y_dual, s_dual = _scan_pair(
        x0, dt[:, :0], Bm[:, :0], Cm[:, :0], A, D, initial_state=S0)
    _ok('T=0 with initial state passes it through in both views',
        torch.equal(s_rec, S0) and torch.equal(s_rec, s_dual), '')

    # output_final_state=False contract.
    y_rec, s_rec, y_dual, s_dual = _scan_pair(
        x, dt, Bm, Cm, A, D, output_final_state=False)
    _ok('output_final_state=False returns None from both',
        s_rec is None and s_dual is None
        and (y_rec - y_dual).abs().max().item() < 1e-12, '')

    # fp32: compute stays fp32, parity at fp32 round-off level.
    x32, dt32, B32, C32, A32, D32 = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state,
        dtype=torch.float32, seed=7)
    y_rec, s_rec, y_dual, s_dual = _scan_pair(
        x32, dt32, B32, C32, A32, D32)
    _ok('fp32 output dtype preserved',
        y_rec.dtype == torch.float32 and y_dual.dtype == torch.float32, '')
    _ok('fp32 states in compute precision',
        s_rec.dtype == torch.float32 and s_dual.dtype == torch.float32, '')

    # fp16: output fp16, states computed/stored in fp32.
    x16, dt16, B16, C16, A16, D16 = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state,
        dtype=torch.float16, seed=7)
    y_rec, s_rec, y_dual, s_dual = _scan_pair(
        x16, dt16, B16, C16, A16, D16)
    _ok('fp16 output dtype preserved, states fp32',
        y_rec.dtype == torch.float16 and y_dual.dtype == torch.float16
        and s_rec.dtype == torch.float32 and s_dual.dtype == torch.float32,
        f'y={y_rec.dtype} s={s_rec.dtype}')


def _raises_both(x, dt, Bm, Cm, A, D, label, **kwargs):
    raised = [False, False]
    for i, fn in enumerate((naive_mamba2_scan, quadratic_mamba2_scan)):
        try:
            fn(x, dt, Bm, Cm, A, D, **kwargs)
        except ValueError:
            raised[i] = True
        except Exception as e:
            print(f'    {fn.__name__}: unexpected {type(e).__name__}: {e}')
    _ok(f'both reject {label}', all(raised), f'recurrent={raised[0]} '
        f'dual={raised[1]}')


def test_ssd_validation_parity():
    print('\nTest: both scans reject the same malformed inputs')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 6, 4, 3, 2, 4
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=3)
    S0 = torch.randn(batch, nheads, headdim, d_state, dtype=torch.float64)

    _raises_both(x[0], dt, Bm, Cm, A, D, 'x rank != 4')
    _raises_both(x, dt[:, :, : nheads - 1], Bm, Cm, A, D,
                 'dt head-count mismatch')
    _raises_both(x, dt, Bm[:1], Cm[:1], A, D, 'B/C batch mismatch')
    _raises_both(x, dt, Bm[:, :seqlen - 1], Cm[:, :seqlen - 1], A, D,
                 'B/C seqlen mismatch')
    _raises_both(x, dt, Bm, Cm[:, :, :, :d_state - 1], A, D,
                 'C shape != B shape')
    # nheads=4 with groups=3: heads are not divisible by groups.
    B3 = torch.randn(batch, seqlen, 3, d_state, dtype=torch.float64)
    _raises_both(x, dt, B3, B3, A, D, 'nheads % ngroups != 0')
    _raises_both(x, dt, Bm, Cm, A[:-1], D, 'A length mismatch')
    _raises_both(x, dt, Bm, Cm, A, D[:-1], 'D length mismatch')
    _raises_both(x, dt, Bm, Cm, A, D, 'initial_state rank',
                 initial_state=S0[0])
    _raises_both(x, dt, Bm, Cm, A, D, 'initial_state shape',
                 initial_state=S0[:, :, :, :d_state - 1])


# --- gradient parity ------------------------------------------------------

def test_ssd_gradient_parity():
    print('\nTest: recurrent scan == dual form (gradients)')
    batch, seqlen, nheads, headdim, groups, d_state = 2, 11, 4, 6, 2, 5
    x, dt, Bm, Cm, A, D = _rand_inputs(
        batch, seqlen, nheads, headdim, groups, d_state, seed=42)
    S0 = (torch.randn(batch, nheads, headdim, d_state, dtype=torch.float64,
                      generator=torch.Generator().manual_seed(43)) * 0.1)
    base = {'x': x, 'dt': dt, 'B': Bm, 'C': Cm, 'A': A, 'D': D, 'S0': S0}
    names = list(base)

    def _leaves():
        return {k: v.clone().requires_grad_(True) for k, v in base.items()}

    def _loss(y, s):
        # Deterministic projections (no RNG): same weights for both paths.
        wy = torch.linspace(0.5, 1.5, y.numel(),
                            dtype=torch.float64).reshape(y.shape)
        ws = torch.linspace(1.5, 0.5, s.numel(),
                            dtype=torch.float64).reshape(s.shape)
        return (y * wy).sum() + (s * ws).sum()

    grads = []
    for fn in (naive_mamba2_scan, quadratic_mamba2_scan):
        leaf = _leaves()
        y, s = fn(leaf['x'], leaf['dt'], leaf['B'], leaf['C'], leaf['A'],
                  leaf['D'], initial_state=leaf['S0'])
        _loss(y, s).backward()
        grads.append(leaf)

    for name in names:
        a, b = grads[0][name].grad, grads[1][name].grad
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        nonzero = bool(a.abs().max().item() > 0)
        diff = (a - b).abs().max().item()
        _ok(f'grad[{name}] matches and is finite/nonzero',
            finite and nonzero and diff < 1e-10,
            f'maxdiff={diff:.2e} finite={finite} nonzero={nonzero}')


# --- end-to-end mixer integration -----------------------------------------

def test_mixer_forward_pinned_by_dual_form():
    print('\nTest: the full Mamba2Mixer forward is invariant to the scan view')
    from ops_fused import HybridConfig
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16, memory_type='mamba2',
        mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        mamba_headdim=16, mamba_backend='reference')
    torch.manual_seed(0)
    mixer = ops_mamba.Mamba2Mixer(cfg).double().eval()
    x = torch.randn(2, 17, 32, dtype=torch.float64)
    with torch.no_grad():
        out_ref, s_ref, conv_ref = mixer.forward_functional(x, None, None)
        # Swap ONLY the scan inside the module's namespace: the projections,
        # conv, gating, and state plumbing stay untouched, so any output
        # difference would mean the two scan views disagree somewhere the
        # raw-scan parity test cannot see (e.g. dtype/layout mismatches).
        orig = ops_mamba.naive_mamba2_scan
        ops_mamba.naive_mamba2_scan = ops_mamba.quadratic_mamba2_scan
        try:
            out_dual, s_dual, conv_dual = mixer.forward_functional(
                x, None, None)
        finally:
            ops_mamba.naive_mamba2_scan = orig
    _ok('mixer output invariant to the scan view',
        (out_ref - out_dual).abs().max().item() < 1e-9,
        f'maxdiff={(out_ref - out_dual).abs().max().item():.2e}')
    _ok('mixer SSM state invariant to the scan view',
        (s_ref - s_dual).abs().max().item() < 1e-9,
        f'maxdiff={(s_ref - s_dual).abs().max().item():.2e}')
    _ok('mixer conv state identical (scan-independent)',
        torch.equal(conv_ref, conv_dual), '')


_TEST_FUNCS = [
    test_ssd_hand_computed_values,
    test_ssd_forward_parity,
    test_ssd_contract_parity,
    test_ssd_validation_parity,
    test_ssd_gradient_parity,
    test_mixer_forward_pinned_by_dual_form,
]


def main() -> int:
    print('=' * 70)
    print('SSD duality (Mamba-2 dual-form oracle) Regression Tests')
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
