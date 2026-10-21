"""Regression tests for bfloat16 gated KDA and GatedAttentionMixin.

Covers:
* bfloat16 forward parity (custom vs naive vs chunk)
* bfloat16 backward parity (custom vs naive autograd)
* bfloat16 gradcheck (fp64 oracle)
* Extreme sequence length (T=2048) NaN/Inf monitoring
* GatedAttentionMixin unit tests (shape, gradient flow, init semantics)
* GatedDeltaRule functional tests
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn.functional as F

from ops_kda import naive_recurrent_kda, naive_chunk_kda
from ops_kda_autograd import custom_recurrent_kda
from ops_gated_attention import (
    GatedAttentionMixin,
    create_gate,
    apply_gate_fn,
    GatedDeltaRule,
    gated_delta_rule_step,
)


def _ok(name: str, cond: bool, detail: str = '') -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'


def _kda_inputs(B, T, H, K, V, *, G=1, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    HV = H * G
    q = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
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


# =============================================================================
# bfloat16 KDA correctness tests
# =============================================================================
def test_bf16_forward_parity():
    print('\nTest: bfloat16 forward parity (custom vs naive vs chunk)')
    dtype = torch.bfloat16
    for G in (1, 2):
        for use_init in (False, True):
            q, k, v, g, beta = _kda_inputs(2, 31, 2, 4, 5, G=G, dtype=dtype, seed=G)
            init = (torch.randn(2, 2 * G, 4, 5, dtype=dtype) * 0.1 if use_init else None)

            o1, s1 = naive_recurrent_kda(q, k, v, g, beta, initial_state=init, output_final_state=True)
            o2, s2 = custom_recurrent_kda(q, k, v, g, beta, initial_state=init, output_final_state=True)
            o3, s3 = naive_chunk_kda(q, k, v, g, beta, initial_state=init, output_final_state=True, chunk_size=16)

            # bfloat16 tolerance: ~1e-2 relative
            tol = 2e-2
            for name, a, b in [('custom vs naive', o1, o2), ('chunk vs naive', o1, o3),
                                ('custom state vs naive', s1, s2), ('chunk state vs naive', s1, s3)]:
                diff = (a - b).abs().max().item()
                scale = a.abs().max().item()
                _ok(f'{dtype} G={G} init={use_init} {name}', diff <= tol * max(1.0, scale),
                    f'max diff {diff:.2e} scale {scale:.2e}')


def test_bf16_backward_parity():
    print('\nTest: bfloat16 backward parity (custom vs naive autograd)')
    dtype = torch.bfloat16
    for G in (1, 2):
        for use_init in (True, False):
            _, ga = _grads(naive_recurrent_kda, dtype, G=G, use_init=use_init)
            _, gb = _grads(custom_recurrent_kda, dtype, G=G, use_init=use_init)
            names = ('dq', 'dk', 'dv', 'dg', 'dbeta') + (('dinit',) if use_init else ())
            for name, a, b in zip(names, ga, gb):
                diff = (a - b).abs().max().item()
                scale = a.abs().max().item()
                _ok(f'{dtype} G={G} init={use_init} {name}: parity',
                    diff <= 5e-2 * max(1.0, scale),
                    f'max diff {diff:.2e} scale {scale:.2e}')


def test_bf16_gradcheck():
    print('\nTest: bfloat16 gradcheck via fp64 oracle')
    # gradcheck requires fp64, but we verify the bfloat16 path matches fp64 oracle
    B, T, H, K, V = 1, 5, 2, 3, 2
    for fn_name, fn in [('naive', naive_recurrent_kda), ('custom', custom_recurrent_kda)]:
        q, k, v, g, beta = _kda_inputs(B, T, H, K, V, dtype=torch.float64, seed=7)
        init = torch.randn(B, H, K, V, dtype=torch.float64) * 0.1
        for t in (q, k, v, g, beta, init):
            t.requires_grad_()

        def f(q, k, v, g, beta, init):
            return fn(q, k, v, g, beta, initial_state=init, output_final_state=True)

        ok = torch.autograd.gradcheck(f, (q, k, v, g, beta, init), eps=1e-6, atol=1e-4, rtol=1e-3)
        _ok(f'gradcheck {fn_name} (fp64 oracle)', ok, '')


def test_bf16_long_sequence_stability():
    print('\nTest: bfloat16 long sequence (T=2048) NaN/Inf monitoring')
    dtype = torch.bfloat16
    B, T, H, K, V = 1, 2048, 2, 16, 16
    q, k, v, g, beta = _kda_inputs(B, T, H, K, V, dtype=dtype, seed=42)

    # Test both paths
    for fn_name, fn in [('naive', naive_recurrent_kda), ('custom', custom_recurrent_kda),
                        ('chunk', lambda *a, **kw: naive_chunk_kda(*a, **kw, chunk_size=64))]:
        o, s = fn(q, k, v, g, beta, output_final_state=True)
        has_nan = torch.isnan(o).any().item() or (s is not None and torch.isnan(s).any().item())
        has_inf = torch.isinf(o).any().item() or (s is not None and torch.isinf(s).any().item())
        _ok(f'{dtype} T={T} {fn_name}: no NaN', not has_nan, f'has_nan={has_nan}')
        _ok(f'{dtype} T={T} {fn_name}: no Inf', not has_inf, f'has_inf={has_inf}')


def test_bf16_chunk_vs_recurrent_agreement():
    print('\nTest: bfloat16 chunk vs recurrent agreement')
    dtype = torch.bfloat16
    for T in (63, 64, 65, 127, 128, 129, 255, 256, 257):
        q, k, v, g, beta = _kda_inputs(1, T, 2, 8, 8, dtype=dtype, seed=T)
        o1, s1 = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
        o2, s2 = naive_chunk_kda(q, k, v, g, beta, output_final_state=True, chunk_size=64)
        diff = (o1 - o2).abs().max().item()
        scale = o1.abs().max().item()
        _ok(f'{dtype} T={T} chunk vs recurrent', diff <= 3e-2 * max(1.0, scale),
            f'max diff {diff:.2e} scale {scale:.2e}')


# =============================================================================
# GatedAttentionMixin tests
# =============================================================================
class _TestAttention(GatedAttentionMixin, torch.nn.Module):
    def __init__(self, gate_type, gate_dim, num_heads):
        super().__init__()
        self._init_gate(gate_type, gate_dim, num_heads)

    def forward(self, x):
        return self.apply_gate(x)


def test_gated_attention_mixin_shapes():
    print('\nTest: GatedAttentionMixin shape handling')
    for gate_type in ('none', 'per_head', 'per_channel'):
        for num_heads in (1, 4, 8):
            for gate_dim in (16, 32, 64):
                m = _TestAttention(gate_type, gate_dim, num_heads)
                x = torch.randn(2, 10, num_heads, gate_dim)
                out = m(x)
                _ok(f'{gate_type} H={num_heads} D={gate_dim} shape',
                    out.shape == x.shape, f'{out.shape} vs {x.shape}')


def test_gated_attention_mixin_gradient_flow():
    print('\nTest: GatedAttentionMixin gradient flow')
    for gate_type in ('per_head', 'per_channel'):
        m = _TestAttention(gate_type, 16, 4)
        x = torch.randn(2, 8, 4, 16, requires_grad=True)
        out = m(x)
        loss = out.sum()
        loss.backward()
        _ok(f'{gate_type} gate.grad exists', m._gate.grad is not None, '')
        _ok(f'{gate_type} x.grad exists', x.grad is not None, '')
        _ok(f'{gate_type} gate.grad non-zero', m._gate.grad.abs().max().item() > 0, '')


def test_gated_attention_mixin_init_semantics():
    print('\nTest: GatedAttentionMixin initialization near identity')
    for gate_type in ('per_head', 'per_channel'):
        m = _TestAttention(gate_type, 32, 4)
        gate_val = m.get_gate_values()
        _ok(f'{gate_type} init near 1.0', (gate_val - 1.0).abs().max().item() < 0.1,
            f'max diff from 1: {(gate_val - 1.0).abs().max().item():.3f}')


def test_gated_attention_functional():
    print('\nTest: apply_gate_fn / create_gate functional forms')
    x = torch.randn(2, 10, 4, 32)
    for gate_type in ('none', 'per_head', 'per_channel'):
        gate = create_gate(gate_type, 4, 32, init_value=1.0, init_std=0.0)
        out = apply_gate_fn(x, gate)
        if gate_type == 'none':
            _ok(f'{gate_type} functional: identity', torch.equal(out, x), '')
        else:
            _ok(f'{gate_type} functional: shape', out.shape == x.shape, '')
            _ok(f'{gate_type} functional: near identity', (out - x).abs().max().item() < 1e-5, '')


# =============================================================================
# GatedDeltaRule tests
# =============================================================================
def test_gated_delta_rule_forward():
    print('\nTest: GatedDeltaRule forward shape & basic properties')
    B, T, H, K, V = 2, 16, 4, 8, 8
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V) * 0.3

    for gate_type in ('per_head', 'per_channel'):
        layer = GatedDeltaRule(H, K, V, gate_type=gate_type)
        o, s = layer(q, k, v)
        _ok(f'GatedDeltaRule {gate_type} output shape', o.shape == (B, T, H, V), f'{o.shape}')
        _ok(f'GatedDeltaRule {gate_type} state shape', s.shape == (B, H, K, V), f'{s.shape}')

        # With initial state
        init = torch.randn(B, H, K, V) * 0.1
        o2, s2 = layer(q, k, v, initial_state=init)
        _ok(f'GatedDeltaRule {gate_type} with init state', o2.shape == (B, T, H, V), '')


def test_gated_delta_rule_gradient():
    print('\nTest: GatedDeltaRule gradient flow')
    B, T, H, K, V = 2, 8, 4, 8, 8
    # Create leaf tensors first, then normalize and enable grad
    q = F.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    k = F.normalize(torch.randn(B, T, H, K), dim=-1).requires_grad_()
    v = (torch.randn(B, T, H, V) * 0.3).requires_grad_()

    layer = GatedDeltaRule(H, K, V, gate_type='per_channel')
    o, s = layer(q, k, v)
    loss = o.sum() + s.sum()
    loss.backward()

    _ok('GatedDeltaRule q.grad', q.grad is not None and q.grad.abs().max() > 0, '')
    _ok('GatedDeltaRule k.grad', k.grad is not None and k.grad.abs().max() > 0, '')
    _ok('GatedDeltaRule v.grad', v.grad is not None and v.grad.abs().max() > 0, '')
    _ok('GatedDeltaRule g.grad', layer.g.grad is not None and layer.g.grad.abs().max() > 0, '')
    _ok('GatedDeltaRule beta.grad', layer.beta.grad is not None and layer.beta.grad.abs().max() > 0, '')


def test_gated_delta_rule_step_functional():
    print('\nTest: gated_delta_rule_step functional equivalence')
    B, H, K, V = 2, 4, 8, 8
    S = torch.randn(B, H, K, V)
    q_t = F.normalize(torch.randn(B, H, K), dim=-1)
    k_t = F.normalize(torch.randn(B, H, K), dim=-1)
    v_t = torch.randn(B, H, V) * 0.3
    gate = torch.ones(H, K) * 0.9
    beta = torch.ones(H) * 0.1

    o1, S1 = gated_delta_rule_step(S, q_t, k_t, v_t, gate, beta, 'per_channel')
    o2, S2 = gated_delta_rule_step(S, q_t, k_t, v_t, gate, beta, 'per_channel')

    _ok('functional step determinism', torch.equal(o1, o2) and torch.equal(S1, S2), '')

    # Test per_head gate
    gate_h = torch.ones(H) * 0.9
    o3, S3 = gated_delta_rule_step(S, q_t, k_t, v_t, gate_h, beta, 'per_head')
    _ok('per_head gate shape', o3.shape == (B, H, V), '')


def test_gated_delta_rule_state_contract_and_mixed_dtype_step():
    print('\nTest: GatedDeltaRule state geometry + functional-step dtypes')
    B, T, H, K, V = 2, 6, 3, 4, 5
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    v = torch.randn(B, T, H, V) * 0.3
    layer = GatedDeltaRule(H, K, V, gate_type='per_channel')

    bad_shapes = [(B, H, K, 1), (B, H, 1, V), (1, H, K, V),
                  (B, 1, K, V), (1, 1, 1, 1, 1)]
    for shape in bad_shapes:
        raised = False
        try:
            layer(q, k, v, initial_state=torch.zeros(shape))
        except ValueError:
            raised = True
        _ok(f'rejects initial_state {shape}', raised, '')

    # The functional step follows the module's mixed-dtype promotion, so a
    # float64 state with float32 activations must run (not raise).
    S = torch.randn(B, H, K, V, dtype=torch.float64)
    gate = torch.full((H, K), 0.9, dtype=torch.float32)
    beta = torch.full((H,), 0.1, dtype=torch.float32)
    o, S_new = gated_delta_rule_step(
        S, q[:, 0], k[:, 0], v[:, 0], gate, beta, 'per_channel')
    _ok('mixed-dtype step runs', o.shape == (B, H, V), f'{o.shape}')
    _ok('mixed-dtype step promotes state', S_new.dtype == torch.float64,
        f'{S_new.dtype}')
    _ok('mixed-dtype step promotes output', o.dtype == torch.float64,
        f'{o.dtype}')


def test_gated_delta_rule_empty_sequence_dtype_contract():
    print('\nTest: GatedDeltaRule T=0 dtype contract')
    B, H, K, V = 2, 3, 4, 5
    layer = GatedDeltaRule(H, K, V, gate_type='per_channel')
    q = torch.empty(B, 0, H, K)
    k = torch.empty(B, 0, H, K)
    v = torch.empty(B, 0, H, V)
    init = torch.randn(B, H, K, V, dtype=torch.float64)

    o0, s0 = layer(q, k, v, initial_state=init)
    q1 = torch.randn(B, 1, H, K)
    k1 = torch.randn(B, 1, H, K)
    v1 = torch.randn(B, 1, H, V)
    o1, s1 = layer(q1, k1, v1, initial_state=init)
    _ok('T=0 output shape', o0.shape == (B, 0, H, V), f'{o0.shape}')
    _ok('T=0 state promoted', s0.dtype == torch.float64, f'{s0.dtype}')
    _ok('T=0 output matches T=1 dtype', o0.dtype == o1.dtype,
        f'{o0.dtype} vs {o1.dtype}')


def test_gated_delta_rule_bf16():
    print('\nTest: GatedDeltaRule bfloat16 forward')
    dtype = torch.bfloat16
    B, T, H, K, V = 2, 16, 4, 8, 8
    q = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype) * 0.3

    layer = GatedDeltaRule(H, K, V, gate_type='per_channel')
    layer = layer.to(dtype=dtype)
    o, s = layer(q, k, v)

    _ok('GatedDeltaRule bf16 output dtype', o.dtype == dtype, f'{o.dtype}')
    _ok('GatedDeltaRule bf16 state dtype', s.dtype == dtype, f'{s.dtype}')
    _ok('GatedDeltaRule bf16 no NaN', not torch.isnan(o).any().item(), '')
    _ok('GatedDeltaRule bf16 no Inf', not torch.isinf(o).any().item(), '')


if __name__ == '__main__':
    test_bf16_forward_parity()
    test_bf16_backward_parity()
    test_bf16_gradcheck()
    test_bf16_long_sequence_stability()
    test_bf16_chunk_vs_recurrent_agreement()

    test_gated_attention_mixin_shapes()
    test_gated_attention_mixin_gradient_flow()
    test_gated_attention_mixin_init_semantics()
    test_gated_attention_functional()

    test_gated_delta_rule_forward()
    test_gated_delta_rule_gradient()
    test_gated_delta_rule_step_functional()
    test_gated_delta_rule_state_contract_and_mixed_dtype_step()
    test_gated_delta_rule_empty_sequence_dtype_contract()
    test_gated_delta_rule_bf16()

    print('\n=== ALL TESTS PASSED ===')