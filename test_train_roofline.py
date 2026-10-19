"""Regression tests for gradient-checkpointing policies + Exp 14 runner.

What is pinned here
-------------------
``grad_checkpointing`` re-implements the hybrid stack forward as a pure
function with selectable per-layer checkpointing. The contracts that must
never drift:

* **policy semantics** — which layer kinds each policy checkpoints;
* **value parity** — ``policy='none'`` reproduces
  ``HybridAttentionStack.forward_functional`` exactly (outputs, stacked
  state, lookbacks) and every policy reproduces the same outputs (the
  forward is deterministic; checkpointing may only change what autograd
  RETAINS);
* **gradient parity** — parameter gradients after backward match the
  uncheckpointed baseline for every policy and every memory route;
* **bytes contract** — the ``saved_tensors_hooks`` probe measures fewer
  retained bytes under checkpointing than without, on every route;
* **overhead model** — the analytic recompute factor ``(3n + c) / 3n``;
* **runner payload** — Exp 14's pure report builder is JSON-safe and its
  roofline rows carry the expected fields.

Run via pytest::

    pytest -q test_train_roofline.py

Or directly::

    python3 test_train_roofline.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import grad_checkpointing as gc  # noqa: E402
from ops_fused import HybridAttentionStack  # noqa: E402
from run_decode_bandwidth import build_base_cfg  # noqa: E402

TOL = 1e-6
T = 16
BATCH = 2


def tiny_cfg(memory_type: str = 'kda'):
    cfg = build_base_cfg(32)
    if memory_type != 'kda':
        cfg = dataclasses.replace(cfg, memory_type=memory_type)
    return cfg


def build_stack(cfg, seed=0):
    torch.manual_seed(seed)
    return HybridAttentionStack(cfg, total_layers=5)


# ---------------------------------------------------------------------------
# policy semantics
# ---------------------------------------------------------------------------

def test_parse_policy():
    for policy in gc.CHECKPOINT_POLICIES:
        assert gc.parse_policy(policy) == policy
    with pytest.raises(ValueError):
        gc.parse_policy('sometimes')
    with pytest.raises(TypeError):
        gc.parse_policy(None)


def test_should_checkpoint_matrix():
    layout = ['kda', 'kda', 'kda', 'csa', 'hca']
    assert gc.checkpoint_overhead_model(layout, 'none')['n_layers_checked'] == 0
    assert gc.checkpoint_overhead_model(layout, 'all')['n_layers_checked'] == 5
    assert gc.checkpoint_overhead_model(
        layout, 'memory')['n_layers_checked'] == 3
    assert gc.checkpoint_overhead_model(
        layout, 'retrieval')['n_layers_checked'] == 2
    assert gc.should_checkpoint('kda', 'memory')
    assert gc.should_checkpoint('mamba2', 'memory')
    assert not gc.should_checkpoint('csa', 'memory')
    assert gc.should_checkpoint('csa', 'retrieval')
    assert not gc.should_checkpoint('kda', 'retrieval')
    with pytest.raises(ValueError):
        gc.should_checkpoint('kda', 'bogus')


def test_overhead_model_factors():
    layout = ['kda', 'kda', 'kda', 'csa', 'hca']
    none = gc.checkpoint_overhead_model(layout, 'none')
    assert none['extra_forward_fraction'] == 0.0
    assert none['training_flops_relative_to_unchecked'] == pytest.approx(1.0)
    all_ = gc.checkpoint_overhead_model(layout, 'all')
    assert all_['extra_forward_fraction'] == pytest.approx(1.0)
    assert all_['training_flops_relative_to_unchecked'] == pytest.approx(4 / 3)
    mem = gc.checkpoint_overhead_model(layout, 'memory')
    assert mem['training_flops_relative_to_unchecked'] == pytest.approx(
        (15 + 3) / 15)
    ret = gc.checkpoint_overhead_model(layout, 'retrieval')
    assert ret['training_flops_relative_to_unchecked'] == pytest.approx(
        (15 + 2) / 15)


def test_minimum_retained_bytes():
    assert gc.minimum_retained_bytes(5, 2, 16, 32) == 5 * 2 * 16 * 32 * 4
    assert gc.minimum_retained_bytes(0, 2, 16, 32) == 0
    with pytest.raises(ValueError):
        gc.minimum_retained_bytes(5, 2, 16, 32, itemsize=0)
    with pytest.raises(ValueError):
        gc.minimum_retained_bytes(-1, 2, 16, 32)


# ---------------------------------------------------------------------------
# probe mechanics
# ---------------------------------------------------------------------------

def test_probe_counts_and_tags_saved_tensors():
    probe = gc.SavedBytesProbe()
    lin = nn.Linear(4, 4)
    x = torch.randn(2, 3, 4, requires_grad=True)
    with probe.measure():
        with probe.tagged('first'):
            out1 = lin(x)
        with probe.tagged('second'):
            y = lin(x)
    assert out1 is not None
    by_tag = probe.bytes_by_tag()
    assert set(by_tag) == {'first', 'second'}
    assert by_tag['first'] == by_tag['second']
    # At least the input activation is saved on each call.
    assert by_tag['first'] >= 2 * 3 * 4 * 4
    assert probe.total_tensors() >= 2
    assert probe.total_bytes() == by_tag['first'] + by_tag['second']
    y.sum().backward()
    # Backward traffic outside the measure() block is not counted.
    assert probe.total_bytes() == by_tag['first'] + by_tag['second']
    probe.reset()
    assert probe.total_bytes() == 0 and probe.bytes_by_tag() == {}


def test_probe_unattributed_tag():
    probe = gc.SavedBytesProbe()
    lin = nn.Linear(3, 3)
    x = torch.randn(1, 2, 3, requires_grad=True)
    with probe.measure():
        out = lin(x)
    # Without a tag the saves land in the unattributed bucket (>= the
    # activation bytes; torch may additionally save the weight).
    assert probe.bytes_by_tag()['__unattributed__'] >= 1 * 2 * 3 * 4
    del out


# ---------------------------------------------------------------------------
# value / gradient parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('policy', gc.CHECKPOINT_POLICIES)
def test_value_parity_vs_forward_functional(policy):
    cfg = tiny_cfg()
    stack = build_stack(cfg)
    x = torch.randn(BATCH, T, cfg.d_model)
    ref_out, ref_state, ref_lbs = stack.forward_functional(
        x, None, [None] * stack.n_memory_layers)
    out, state, lbs = gc.forward_with_policy(stack, x, policy=policy)
    assert torch.equal(out, ref_out)
    assert torch.equal(state, ref_state)
    assert len(lbs) == len(ref_lbs)
    for a, b in zip(lbs, ref_lbs):
        assert torch.equal(a, b)


@pytest.mark.parametrize('memory_type', ['kda', 'mamba2', 'rwkv7'])
@pytest.mark.parametrize('policy', gc.CHECKPOINT_POLICIES)
def test_gradient_parity_all_routes(policy, memory_type):
    cfg = tiny_cfg(memory_type)
    stack = build_stack(cfg)
    x = torch.randn(BATCH, T, cfg.d_model)
    stack.zero_grad(set_to_none=True)
    out0, _, _ = stack.forward_functional(
        x, None, [None] * stack.n_memory_layers)
    out0.pow(2).mean().backward()
    grads0 = {name: p.grad.detach().clone()
              for name, p in stack.named_parameters()}
    stack.zero_grad(set_to_none=True)
    out, _, _ = gc.forward_with_policy(stack, x, policy=policy)
    out.pow(2).mean().backward()
    for name, p in stack.named_parameters():
        assert p.grad is not None, name
        diff = (p.grad - grads0[name]).abs().max().item()
        assert diff < TOL, (policy, memory_type, name, diff)


@pytest.mark.parametrize('memory_type', ['kda', 'mamba2', 'rwkv7'])
def test_checkpointing_reduces_retained_bytes(memory_type):
    cfg = tiny_cfg(memory_type)
    stack = build_stack(cfg)
    x = torch.randn(BATCH, T, cfg.d_model)
    saved = {}
    for policy in gc.CHECKPOINT_POLICIES:
        probe = gc.SavedBytesProbe()
        with probe.measure():
            out, _, _ = gc.forward_with_policy(stack, x, policy=policy,
                                               probe=probe)
        saved[policy] = probe.total_bytes()
        del out
    assert saved['all'] < saved['none']
    assert saved['memory'] < saved['none']
    assert saved['retrieval'] < saved['none']
    assert saved['all'] < saved['memory']
    floor = gc.minimum_retained_bytes(5, BATCH, T, cfg.d_model)
    assert saved['all'] >= floor
    assert saved['none'] > floor


def test_forward_with_policy_state_threading():
    cfg = tiny_cfg()
    stack = build_stack(cfg)
    x1 = torch.randn(BATCH, T, cfg.d_model)
    _, state, lookbacks = stack.forward_functional(
        x1, None, [None] * stack.n_memory_layers)
    x2 = torch.randn(BATCH, T, cfg.d_model)
    out_ref, state_ref, _ = stack.forward_functional(
        x2, state.detach(), [lb.detach() for lb in lookbacks])
    out, state2, _ = gc.forward_with_policy(
        stack, x2, policy='memory', state=state.detach(),
        lookbacks=[lb.detach() for lb in lookbacks])
    assert torch.equal(out, out_ref)
    assert torch.equal(state2, state_ref)
    with pytest.raises(ValueError):
        gc.forward_with_policy(stack, x2, state=state[:1].detach())
    with pytest.raises(ValueError):
        gc.forward_with_policy(stack, x2, lookbacks=[None])


# ---------------------------------------------------------------------------
# runner payload
# ---------------------------------------------------------------------------

def test_quality_probe_sample_stats_and_seed_pairing():
    """The MQA quality probe reports sample std/CI and paired contrasts."""
    import statistics
    from run_train_roofline import quality_probe

    rows = quality_probe(
        seeds=(0, 1, 2), device='cpu', d_model=32, n_heads=4, head_dim=8,
        seq_len=8, vocab=4, steps=1, lr=3e-3, batch=4,
        eval_batches=1, eval_batch=4)
    by_variant = {row['variant']: row for row in rows}
    assert set(by_variant) == {'mha', 'gqa', 'mqa'}
    assert by_variant['mha']['paired_vs_mha'] is None
    for name, row in by_variant.items():
        assert row['n_seeds'] == 3
        assert len(row['accs']) == len(row['seeds']) == 3
        assert row['std_acc'] == pytest.approx(
            statistics.stdev(row['accs']))
        assert row['std_acc'] != pytest.approx(
            statistics.pstdev(row['accs'])) or len(set(row['accs'])) == 1
        assert row['min_acc'] <= row['mean_acc'] <= row['max_acc']
        assert row['ci95_acc'] is not None
        lo, hi = row['ci95_acc']
        assert lo <= row['mean_acc'] <= hi
        if name != 'mha':
            paired = row['paired_vs_mha']
            assert paired is not None and paired['n_pairs'] == 3
            assert paired['mean_diff'] == pytest.approx(
                row['mean_acc'] - by_variant['mha']['mean_acc'])
            p_raw = paired['t_test']['p_value_two_sided']
            if p_raw is not None:
                assert paired['bonferroni_family_size'] == 2
                assert paired['p_value_bonferroni'] == pytest.approx(
                    min(1.0, 2.0 * p_raw))
                assert paired['significant_bonferroni'] is (
                    paired['p_value_bonferroni'] < 0.05)


def test_runner_report_json_safe():
    from run_train_roofline import KIND, build_report
    report = build_report(
        T=T, batch=1, d_model=32, seed=0, policies=('none', 'all'),
        itemsize=4, device='cpu',
        peaks={'peak_bw_gbps': 100.0, 'peak_tflops': 1.0},
        peak_source='explicit', quality_seeds=(0,), quality_steps=1,
        quality_lr=3e-3, skip_quality=True, timing_reps=1)
    assert report['kind'] == KIND
    assert [r['policy'] for r in report['checkpoint_policies']] == \
        ['none', 'all']
    for row in report['checkpoint_policies']:
        assert row['output_max_abs_diff_vs_baseline'] < TOL
        assert row['grad_max_abs_diff_vs_baseline'] < TOL
        assert row['saved_activation_bytes'] > 0
        assert row['fwd_bwd_ms_min'] > 0
    roofline_rows = {r['policy']: r for r in report['training_roofline']}
    assert roofline_rows['none']['arithmetic_intensity_flops_per_byte'] \
        < roofline_rows['all']['arithmetic_intensity_flops_per_byte']
    for row in roofline_rows.values():
        assert row['roofline_bound'] in ('compute', 'memory')
        assert row['roofline_ceiling_ms'] > 0
        assert 0.0 <= row['activation_share'] <= 1.0
    mqa_rows = {r['variant']: r for r in report['mqa_variants']}
    assert set(mqa_rows) == {'mha', 'gqa', 'mqa'}
    assert mqa_rows['mha']['kv_reduction_vs_mha'] == 1.0
    assert mqa_rows['gqa']['kv_reduction_vs_mha'] == 2.0
    assert mqa_rows['mqa']['kv_reduction_vs_mha'] == 4.0
    assert report['quality_probe']['status'] == 'skipped'
    assert len(report['hybrid_decode_reference']) == 3
    json.dumps(report, allow_nan=False)
    # The written artifact must be a versioned results_schema envelope.
    import results_schema
    from kaggle_setup import sanitize_for_json
    from run_train_roofline import report_envelope
    env = report_envelope(report)
    assert env['kind'] == KIND
    assert results_schema.validate_payload(env) == []
    _, rows, version = results_schema.parse(env)
    assert len(rows) == 2
    assert results_schema.is_compatible(version)
    assert env['metadata']['quality_probe']['status'] == 'skipped'
    json.dumps(sanitize_for_json(env), allow_nan=False)


def test_runner_rejects_bad_geometry():
    from run_train_roofline import build_report
    with pytest.raises(ValueError):
        build_report(
            T=T, batch=1, d_model=30, seed=0, policies=('none',),
            itemsize=4, device='cpu', peaks={}, peak_source='none',
            quality_seeds=(0,), quality_steps=1, quality_lr=3e-3,
            skip_quality=True, timing_reps=1)
    with pytest.raises(ValueError):
        build_report(
            T=0, batch=1, d_model=32, seed=0, policies=('none',),
            itemsize=4, device='cpu', peaks={}, peak_source='none',
            quality_seeds=(0,), quality_steps=1, quality_lr=3e-3,
            skip_quality=True, timing_reps=1)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
