"""Regression tests for the custom-autograd KDA recurrence.

Covers ``ops_kda_autograd.custom_recurrent_kda``:

* forward parity — bit-identical to ``ops_kda.naive_recurrent_kda``
  (fp32/fp64, GVA on/off, with/without ``initial_state``, T = 0/1);
* backward parity — gradients match the autograd path through
  ``naive_recurrent_kda`` to float round-off, plus
  ``torch.autograd.gradcheck`` on the raw Function in fp64;
* log-decay clamp semantics — ``g_clamp_min`` is applied before the
  custom Function, so clamped channels get exactly zero gradient;
* gradient-precision sweep — the analytic backward's fp64-oracle error
  across fp32/fp16/bf16 stays within ``precision_sweep``'s dtype budgets
  and is on par with the autograd path (no precision loss from the
  hand-written adjoint);
* memory footprint — the custom node's autograd graph has O(1) nodes and
  saves an order of magnitude fewer bytes than the per-step graph of
  ``naive_recurrent_kda``;
* integration — the ``custom`` path registered in ``precision_sweep``
  honors the dtype contract and error budgets in explicit/autocast modes.

Run via pytest::

    pytest -q test_kda_custom_autograd.py

Or directly::

    python3 test_kda_custom_autograd.py
"""
from __future__ import annotations

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import precision_sweep as ps  # noqa: E402
from ops_kda import naive_recurrent_kda  # noqa: E402
from ops_kda_autograd import custom_recurrent_kda  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _kda_inputs(B, T, H, K, V, *, G=1, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    HV = H * G
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    v = torch.randn(B, T, HV, V, dtype=dtype) * 0.3
    g = -torch.rand(B, T, HV, K, dtype=dtype) * 0.1
    beta = torch.rand(B, T, HV, dtype=dtype) * 0.3
    return q, k, v, g, beta


def _grads(fn, dtype, *, B=2, T=17, H=2, K=4, V=5, G=1, use_init=True,
           final_state_loss=True, seed=3, g_values=None, **kwargs):
    q, k, v, g, beta = _kda_inputs(B, T, H, K, V, G=G, dtype=dtype, seed=seed)
    if g_values is not None:
        g = g_values.to(dtype)
    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
    g.requires_grad_(); beta.requires_grad_()
    init = None
    if use_init:
        init = (torch.randn(B, H * G, K, V, dtype=dtype) * 0.1)
        init.requires_grad_()
    o, s = fn(q, k, v, g, beta, initial_state=init,
              output_final_state=True, **kwargs)
    loss = (o * 1.3).sum()
    if final_state_loss:
        loss = loss + (s * 0.2).sum()
    loss.backward()
    grads = [q.grad, k.grad, v.grad, g.grad, beta.grad]
    if init is not None:
        grads.append(init.grad)
    return o.detach(), grads


# ---------------------------------------------------------------------------
# Forward / backward parity
# ---------------------------------------------------------------------------
def test_forward_bit_identical_to_naive():
    print('\nTest: custom forward is bit-identical to naive_recurrent_kda')
    for dtype, tol in ((torch.float64, 0.0), (torch.float32, 0.0)):
        for G in (1, 2):
            for use_init in (False, True):
                q, k, v, g, beta = _kda_inputs(2, 13, 2, 4, 5, G=G,
                                               dtype=dtype, seed=G)
                init = (torch.randn(2, 2 * G, 4, 5, dtype=dtype) * 0.1
                        if use_init else None)
                o1, s1 = naive_recurrent_kda(
                    q, k, v, g, beta, initial_state=init,
                    output_final_state=True)
                o2, s2 = custom_recurrent_kda(
                    q, k, v, g, beta, initial_state=init,
                    output_final_state=True)
                diff = max((o1 - o2).abs().max().item(),
                           (s1 - s2).abs().max().item())
                _ok(f'{dtype} G={G} init={use_init}: exact forward parity',
                    diff <= tol, f'max diff {diff:.2e}')
    # T = 0 and T = 1 edge shapes
    for T in (0, 1):
        q, k, v, g, beta = _kda_inputs(1, T, 1, 3, 2, dtype=torch.float32)
        o1, s1 = naive_recurrent_kda(q, k, v, g, beta,
                                     output_final_state=True)
        o2, s2 = custom_recurrent_kda(q, k, v, g, beta,
                                      output_final_state=True)
        _ok(f'T={T}: forward/state match', torch.equal(o1, o2)
            and torch.equal(s1, s2), f'{tuple(o2.shape)}')
    # no-grad delegation equals the reference path
    q, k, v, g, beta = _kda_inputs(1, 9, 2, 3, 3, dtype=torch.float32)
    with torch.no_grad():
        o1, s1 = custom_recurrent_kda(q, k, v, g, beta,
                                      output_final_state=True)
    o2, s2 = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    _ok('no_grad delegation matches reference',
        torch.equal(o1, o2) and torch.equal(s1, s2), '')
    # output_final_state=False returns None state and matches
    o1, s1 = custom_recurrent_kda(q, k, v, g, beta)
    o2, _ = naive_recurrent_kda(q, k, v, g, beta)
    _ok('output_final_state=False', s1 is None and torch.equal(o1, o2), '')


def test_backward_gradients_match_autograd():
    print('\nTest: custom backward matches autograd through the naive path')
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
        for G in (1, 2):
            for use_init in (True, False):
                _, ga = _grads(naive_recurrent_kda, dtype, G=G,
                               use_init=use_init)
                _, gb = _grads(custom_recurrent_kda, dtype, G=G,
                               use_init=use_init)
                names = ('dq', 'dk', 'dv', 'dg', 'dbeta') + \
                    (('dinit',) if use_init else ())
                for name, a, b in zip(names, ga, gb):
                    diff = (a - b).abs().max().item()
                    scale = a.abs().max().item()
                    _ok(f'{dtype} G={G} init={use_init} {name}: parity',
                        diff <= tol * max(1.0, scale),
                        f'max diff {diff:.2e} scale {scale:.2e}')
    # loss on the output alone (final state output unused)
    _, ga = _grads(naive_recurrent_kda, torch.float64,
                   final_state_loss=False)
    _, gb = _grads(custom_recurrent_kda, torch.float64,
                   final_state_loss=False)
    diffs = [float((a - b).abs().max()) for a, b in zip(ga, gb)]
    _ok('output-only loss: parity', max(diffs) < 1e-12, f'max {max(diffs):.2e}')


def test_gradcheck_small_fp64():
    print('\nTest: torch.autograd.gradcheck on the custom Function (fp64)')
    B, T, H, K, V = 1, 4, 2, 3, 2
    q, k, v, g, beta = _kda_inputs(B, T, H, K, V, dtype=torch.float64, seed=5)
    init = torch.randn(B, H, K, V, dtype=torch.float64) * 0.1
    for t in (q, k, v, g, beta, init):
        t.requires_grad_()

    def f_with_init(q, k, v, g, beta, init):
        return custom_recurrent_kda(q, k, v, g, beta, initial_state=init,
                                    output_final_state=True)

    def f_no_init(q, k, v, g, beta):
        return custom_recurrent_kda(q, k, v, g, beta,
                                    output_final_state=True)

    _ok('gradcheck with initial_state',
        torch.autograd.gradcheck(f_with_init,
                                 (q, k, v, g, beta, init),
                                 eps=1e-6, atol=1e-6, rtol=1e-4), '')
    _ok('gradcheck without initial_state',
        torch.autograd.gradcheck(f_no_init, (q, k, v, g, beta),
                                 eps=1e-6, atol=1e-6, rtol=1e-4), '')


# ---------------------------------------------------------------------------
# log-decay clamp semantics
# ---------------------------------------------------------------------------
def test_log_decay_clamp_gradient_semantics():
    print('\nTest: log-decay clamp semantics (zero grad on clamped channels)')
    B, T, H, K, V = 2, 9, 2, 4, 3
    q, k, v, g, beta = _kda_inputs(B, T, H, K, V, dtype=torch.float32, seed=7)
    g = -torch.rand(B, T, H, K) * 30.0  # most channels below any clamp
    for clamp in (-1.0, -5.0, -10.0, float('-inf')):
        _, ga = _grads(naive_recurrent_kda, torch.float32, B=B, T=T, H=H,
                       K=K, V=V, seed=7, g_values=g, g_clamp_min=clamp)
        o2, gb = _grads(custom_recurrent_kda, torch.float32, B=B, T=T, H=H,
                        K=K, V=V, seed=7, g_values=g, g_clamp_min=clamp)
        diffs = [float((a - b).abs().max()) for a, b in zip(ga, gb)]
        _ok(f'clamp={clamp}: gradient parity', max(diffs) == 0.0,
            f'max {max(diffs):.2e}')
        if clamp > float('-inf'):
            mask = g < clamp
            _ok(f'clamp={clamp}: dg exactly 0 on clamped channels',
                bool((gb[3][mask] == 0).all().item()),
                f'{int(mask.sum().item())}/{g.numel()} clamped')


# ---------------------------------------------------------------------------
# dtype contract / precision sweep
# ---------------------------------------------------------------------------
def test_dtype_contract_and_state_dtype():
    print('\nTest: dtype contract (out dtype preserved, state compute-precision)')
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        q, k, v, g, beta = _kda_inputs(1, 8, 2, 3, 3, dtype=dtype, seed=11)
        o, s = custom_recurrent_kda(q, k, v, g, beta,
                                    output_final_state=True)
        expected_state = (torch.float32 if dtype != torch.float64
                          else torch.float64)
        _ok(f'{dtype}: out dtype preserved and finite',
            o.dtype == dtype and bool(torch.isfinite(o).all()), str(o.dtype))
        _ok(f'{dtype}: state is {expected_state}',
            s.dtype == expected_state, str(s.dtype))
    q, k, v, g, beta = _kda_inputs(1, 8, 2, 3, 3, dtype=torch.float32, seed=1)
    _, s = custom_recurrent_kda(q, k, v, g, beta, output_final_state=True,
                                state_dtype=torch.float16)
    _ok('state_dtype override honored', s.dtype == torch.float16, str(s.dtype))


def test_gradient_precision_sweep_against_fp64_oracle():
    print('\nTest: gradient-precision sweep vs fp64 oracle (dtype budgets)')
    T = 64
    seed = 0
    base = ps.build_inputs('kda', seed, T)  # fp64, normalized q/k

    def grads_for(fn, dtype):
        t = {key: val.to(dtype).clone().requires_grad_()
             for key, val in base.items()}
        o, s = fn(**t, output_final_state=True)
        loss = (o * 1.3).sum() + (s * 0.2).sum()
        return [x.detach().to(torch.float64) for x in
                torch.autograd.grad(loss, [t['q'], t['k'], t['v'],
                                           t['g'], t['beta']])]

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        u = ps.unit_roundoff(dtype)
        budget = ps.error_budget_eps_units(dtype, T)
        oracle = grads_for(custom_recurrent_kda, torch.float64)
        custom = grads_for(custom_recurrent_kda, dtype)
        autograd_ref = grads_for(naive_recurrent_kda, dtype)
        rel_custom = max(float((a - b).abs().max() / a.abs().max())
                         for a, b in zip(custom, oracle))
        rel_auto = max(float((a - b).abs().max() / a.abs().max())
                       for a, b in zip(autograd_ref, oracle))
        _ok(f'{ps.dtype_name(dtype)}: custom grad error within budget',
            rel_custom / u <= budget,
            f'{rel_custom / u:.2f}u <= {budget:.1f}u (rel {rel_custom:.2e})')
        _ok(f'{ps.dtype_name(dtype)}: on par with autograd path',
            rel_custom <= rel_auto * 1.05 + 1e-15,
            f'custom {rel_custom:.2e} vs autograd {rel_auto:.2e}')


# ---------------------------------------------------------------------------
# Memory footprint: O(1) graph node vs O(T) per-step graph
# ---------------------------------------------------------------------------
def _graph_nodes(root):
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        for nxt, _ in getattr(node, 'next_functions', ()) or ():
            stack.append(nxt)
    return len(seen)


def _saved_bytes(fn):
    total = [0]

    def pack(tensor):
        total[0] += int(tensor.untyped_storage().nbytes())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total[0]


def test_autograd_graph_and_memory_footprint():
    print('\nTest: O(1) custom graph node vs O(T) naive graph')
    B, T, H, K, V = 1, 64, 2, 4, 4
    q, k, v, g, beta = _kda_inputs(B, T, H, K, V, dtype=torch.float32, seed=2)

    def run(fn):
        return fn(q.clone().requires_grad_(), k.clone().requires_grad_(),
                  v.clone().requires_grad_(), g.clone().requires_grad_(),
                  beta.clone().requires_grad_(),
                  output_final_state=True)[0]

    o_naive = run(naive_recurrent_kda)
    o_custom = run(custom_recurrent_kda)
    n_naive = _graph_nodes(o_naive.grad_fn)
    n_custom = _graph_nodes(o_custom.grad_fn)
    _ok('custom graph has O(1) nodes', n_custom <= 16, f'{n_custom} nodes')
    _ok('naive graph scales with T', n_naive >= 4 * T,
        f'{n_naive} nodes (>= {4 * T})')
    _ok('custom graph is far smaller', n_custom * 10 < n_naive,
        f'{n_custom} vs {n_naive}')

    saved_naive = _saved_bytes(lambda: run(naive_recurrent_kda))
    saved_custom = _saved_bytes(lambda: run(custom_recurrent_kda))
    _ok('custom retains less activation memory',
        saved_custom * 3 < saved_naive,
        f'{saved_custom / 1024:.0f} KiB vs {saved_naive / 1024:.0f} KiB '
        f'({saved_naive / max(saved_custom, 1):.1f}x lower)')


# ---------------------------------------------------------------------------
# Integration with the mixed-precision sweep
# ---------------------------------------------------------------------------
def test_precision_sweep_custom_path_registered():
    print('\nTest: precision_sweep custom path (contract + budget + parity)')
    _ok("'custom' registered in CASES['kda']",
        'custom' in ps.CASES['kda'], str(ps.CASES['kda']))
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        row_custom = ps.evaluate_seed('kda', 'custom', dtype, seed=0,
                                      seq_len=16, chunk_size=8)
        row_rec = ps.evaluate_seed('kda', 'rec', dtype, seed=0,
                                   seq_len=16, chunk_size=8)
        _ok(f'{ps.dtype_name(dtype)}: no harness error',
            'error' not in row_custom and 'error' not in row_rec,
            row_custom.get('error', ''))
        _ok(f'{ps.dtype_name(dtype)}: within budget and dtype contract',
            row_custom['out_eps_units'] <= ps.error_budget_eps_units(dtype, 16)
            and row_custom['out_dtype_ok'] and row_custom['state_dtype_ok'],
            f"out {row_custom['out_eps_units']:.2f}u, "
            f"out_dtype {row_custom['out_dtype']}, "
            f"state_dtype {row_custom['state_dtype']}")
        _ok(f'{ps.dtype_name(dtype)}: forward parity custom vs rec',
            row_custom['out_abs_err'] == row_rec['out_abs_err']
            and row_custom['state_abs_err'] == row_rec['state_abs_err'],
            'bit-identical forward')
    row_auto = ps.evaluate_seed('kda', 'custom', torch.bfloat16, seed=0,
                                seq_len=16, chunk_size=8, mode='autocast')
    _ok('autocast bf16: fp32 contract kept',
        'error' not in row_auto and row_auto['out_dtype'] == 'float32'
        and row_auto['state_dtype'] == 'float32',
        row_auto.get('error') or row_auto['state_dtype'])


def test_module_documents_first_order_scope():
    print('\nTest: module docstring documents first-order-only scope')
    import ops_kda_autograd as mod
    doc = mod.__doc__ or ''
    _ok('docstring mentions double-backward limitation',
        'double-backward' in doc, '')
    _ok('forward loop source is retrievable for TorchScript',
        inspect.getsource(mod._custom_kda_forward_loop).startswith('def'))


_TEST_FUNCS = [
    test_forward_bit_identical_to_naive,
    test_backward_gradients_match_autograd,
    test_gradcheck_small_fp64,
    test_log_decay_clamp_gradient_semantics,
    test_dtype_contract_and_state_dtype,
    test_gradient_precision_sweep_against_fp64_oracle,
    test_autograd_graph_and_memory_footprint,
    test_precision_sweep_custom_path_registered,
    test_module_documents_first_order_scope,
]


def main() -> int:
    print('=' * 70)
    print('Custom-Autograd KDA Recurrence Regression Tests')
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
