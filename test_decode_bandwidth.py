"""Regression tests for the decode-step memory-traffic ledger (Exp 13).

What is pinned here
-------------------
``decode_bandwidth`` answers "where does decode bandwidth go, and when
does the dominant consumer switch from the shared weights to the
per-sequence state?" across the KDA / Mamba-2 / RWKV-7 memory routes.
These tests lock the pieces that must never drift:

* the weight count measured from the REAL ``HybridAttentionStack``
  modules (not a re-derived geometry formula), including the
  ``sum(by_kind) == total`` identity;
* the per-operator read split against the canonical
  ``lm_consistency`` / ``kv_budget`` decode byte model, and the
  route-invariance of the CSA / HCA terms;
* the batch-scaling identities of the ledger
  (``total = weights + B * per-sequence``, shares sum to 1, weight bytes
  batch-independent, itemsize linearity, optional weight term);
* the weights<->state dominance switch batch, checked against the manual
  ``weight_elements / state_elements_per_token`` ratio and against the
  ~0.5 state share at the switch point;
* the route comparison table (all three routes at a shared geometry,
  per-route state traffic distinct, deterministic, explicit layouts and
  unknown routes rejected);
* input validation and the ``bandwidth_bound_ms`` unit contract.

Run via pytest::

    pytest -q test_decode_bandwidth.py

Or directly::

    python3 test_decode_bandwidth.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402

import decode_bandwidth as db  # noqa: E402
import kv_budget  # noqa: E402
import lm_consistency  # noqa: E402
from kv_quant import FP8Spec  # noqa: E402
from ops_fused import HybridAttentionStack, HybridConfig  # noqa: E402


def small_cfg(**overrides) -> HybridConfig:
    """A d_model=32 geometry valid for all three memory routes at once."""
    kw = dict(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=8,
        csa_cI=4, csa_nIh=1, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=8,
        hca_sliding_window=4,
        mamba_d_state=16, mamba_headdim=8, mamba_expand=2,
        rwkv_head_dim=8, rwkv_decay_lora=8, rwkv_a_lora=8,
        rwkv_gate_lora=16,
    )
    kw.update(overrides)
    return HybridConfig(**kw)


T = 128


# ---------------------------------------------------------------------------
# weights measured from the real modules
# ---------------------------------------------------------------------------

def test_parameter_elements_match_live_stack():
    torch.manual_seed(21)
    cfg = small_cfg()
    params = db.attention_tower_parameter_elements(cfg)
    ref_stack = HybridAttentionStack(cfg, total_layers=5)
    ref_total = int(sum(p.numel() for p in ref_stack.parameters()))
    assert params['total'] == ref_total
    assert sum(params['by_kind'].values()) == params['total']
    assert set(params['by_kind']) == {'kda', 'csa', 'hca', 'norm'}
    assert params['layout'] == ['kda', 'kda', 'kda', 'csa', 'hca']


def test_unit_layout_ratio_and_explicit():
    cfg = small_cfg()
    assert db.unit_layout(cfg) == ('kda', 'kda', 'kda', 'csa', 'hca')
    explicit = small_cfg(layout=('memory',) * 3 + ('csa', 'hca'))
    assert db.unit_layout(explicit) == ('kda', 'kda', 'kda', 'csa', 'hca')


def test_parameter_count_does_not_perturb_rng():
    torch.manual_seed(3)
    before = torch.random.get_rng_state().clone()
    db.attention_tower_parameter_elements(small_cfg())
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_weight_bytes_scale_with_itemsize():
    cfg = small_cfg()
    w4 = db.attention_tower_weight_bytes(cfg, itemsize=4)
    w8 = db.attention_tower_weight_bytes(cfg, itemsize=8)
    assert w8['bytes'] == 2 * w4['bytes']
    assert w4['bytes_by_kind']['kda'] == 4 * w4['by_kind']['kda']
    assert sum(w4['bytes_by_kind'].values()) == w4['bytes']


# ---------------------------------------------------------------------------
# per-operator split against the canonical byte model
# ---------------------------------------------------------------------------

def test_split_matches_canonical_model():
    cfg = small_cfg()
    elems = lm_consistency.decode_elements_per_token(cfg, T)
    bd = db.decode_traffic_breakdown(cfg, T)
    read_sum = sum(bd['read_elements_by_op'].values())
    assert abs(read_sum - elems['read_elements_per_token']) < 1e-6
    assert bd['write_elements_by_op']['memory'] \
        == elems['write_elements_per_token']
    assert bd['write_elements_by_op']['csa'] == 0.0
    assert bd['write_elements_by_op']['hca'] == 0.0
    assert bd['total_bytes_per_token'] \
        == bd['total_elements_per_token'] * bd['itemsize']


def test_split_csa_hca_match_kv_budget_rows():
    cfg = small_cfg()
    bd = db.decode_traffic_breakdown(cfg, T)
    spec = lm_consistency.decode_read_spec(cfg)
    csa = kv_budget.op_read_elements('csa', T, spec)['total_read_elements']
    hca = kv_budget.op_read_elements('hca', T, spec)['total_read_elements']
    assert abs(bd['read_elements_by_op']['csa'] - csa) < 1e-9
    assert abs(bd['read_elements_by_op']['hca'] - hca) < 1e-9
    # The memory term is the remainder: it must be strictly positive here.
    assert bd['read_elements_by_op']['memory'] > 0.0


def test_split_memory_term_survives_long_context():
    # At long context the CSA/HCA terms are O(T^2) while the recurrent
    # state read is O(1). Deriving the memory term by subtraction loses it
    # to floating-point cancellation; the canonical component must survive.
    for route in db.MEMORY_ROUTES:
        cfg = small_cfg(memory_type=route)
        elems = lm_consistency.decode_elements_per_token(cfg, 10 ** 30)
        bd = db.decode_traffic_breakdown(cfg, 10 ** 30)
        memory = bd['read_elements_by_op']['memory']
        assert memory == elems['memory_read_elements_per_token'] > 0.0
        assert sum(bd['read_elements_by_op'].values()) == pytest.approx(
            elems['read_elements_per_token'], rel=1e-12)


# ---------------------------------------------------------------------------
# batch-scaling identities
# ---------------------------------------------------------------------------

def test_ledger_batch_identities():
    cfg = small_cfg()
    bd = db.decode_traffic_breakdown(cfg, T)
    led1 = db.decode_step_ledger(cfg, T, batch=1)
    led8 = db.decode_step_ledger(cfg, T, batch=8)
    assert led8['total_bytes_per_step'] == (
        led8['weight_bytes_per_step'] + 8 * bd['total_bytes_per_token'])
    assert led1['total_bytes_per_step'] == (
        led1['weight_bytes_per_step'] + bd['total_bytes_per_token'])
    for led in (led1, led8):
        assert abs(led['weights_share'] + led['state_share'] - 1.0) < 1e-12
    assert led1['weight_bytes_per_step'] == led8['weight_bytes_per_step']
    assert led8['state_share'] > led1['state_share']
    assert led8['total_bytes_per_step'] > led1['total_bytes_per_step']
    # Per-sequence per-token metric shrinks with batch but never below
    # the state-only term.
    assert led8['bytes_per_sequence_per_token'] \
        < led1['bytes_per_sequence_per_token']
    assert led8['bytes_per_sequence_per_token'] >= \
        led8['state_bytes_per_sequence_per_token']


def test_ledger_itemsize_and_weight_toggle():
    cfg = small_cfg()
    led4 = db.decode_step_ledger(cfg, T, batch=8, itemsize=4)
    led8 = db.decode_step_ledger(cfg, T, batch=8, itemsize=8)
    assert led8['total_bytes_per_step'] == 2 * led4['total_bytes_per_step']
    now = db.decode_step_ledger(cfg, T, batch=4, include_weights=False)
    assert now['weight_bytes_per_step'] == 0
    assert now['weight_elements'] == 0
    assert now['total_bytes_per_step'] \
        == 4 * db.decode_traffic_breakdown(cfg, T)['total_bytes_per_token']


def test_switch_batch_definition_and_share():
    cfg = small_cfg()
    elems = lm_consistency.decode_elements_per_token(cfg, T)
    params = db.attention_tower_parameter_elements(cfg)
    switch = db.weights_state_switch_batch(cfg, T)
    assert switch is not None
    assert abs(switch - params['total'] / elems['total_elements_per_token']) \
        < 1e-9
    b_near = max(1, int(round(switch)))
    led = db.decode_step_ledger(cfg, T, batch=b_near)
    share = led['state_share']
    assert abs(share - 0.5) < 0.1, f'B={b_near}, share={share}'
    assert led['weights_state_switch_batch'] == pytest.approx(switch)

    fp8_led = db.decode_step_ledger(
        cfg, T, batch=8, storage={'memory': FP8Spec()},
        weight_itemsize=2)
    fp8_switch = db.weights_state_switch_batch(
        cfg, T, storage={'memory': FP8Spec()}, weight_itemsize=2)
    assert fp8_led['weights_state_switch_batch'] == pytest.approx(fp8_switch)


# ---------------------------------------------------------------------------
# route comparison table
# ---------------------------------------------------------------------------

def test_route_table_all_routes_and_route_invariance():
    cfg = small_cfg()
    rows = db.route_switch_table(cfg, T, batches=(1, 16))
    assert [r['route'] for r in rows] == ['kda', 'mamba2', 'rwkv7']
    assert all(r['status'] == 'ok' for r in rows), [r.get('reason')
                                                    for r in rows]
    csa = {r['read_bytes_by_op_per_token']['csa'] for r in rows}
    hca = {r['read_bytes_by_op_per_token']['hca'] for r in rows}
    assert len(csa) == 1 and len(hca) == 1
    states = {r['state_bytes_per_token'] for r in rows}
    assert len(states) == 3, states
    weights = {r['weight_bytes_per_step'] for r in rows}
    assert len(weights) == 3, weights
    for row in rows:
        assert set(row['by_batch']) == {'1', '16'}
        assert row['weights_state_switch_batch'] > 0
        assert row['by_batch']['16']['state_share'] \
            > row['by_batch']['1']['state_share']
        assert abs(sum(row['read_bytes_by_op_per_token'].values())
                   - row['state_read_bytes_per_token']) < 1e-6


def test_route_table_deterministic_and_isolated():
    cfg = small_cfg()
    rows = db.route_switch_table(cfg, T, batches=(2, 8))
    rows2 = db.route_switch_table(cfg, T, batches=(2, 8))
    assert rows == rows2
    # The base config must not be mutated by the table build.
    assert cfg.memory_type == 'kda' and cfg.layout is None


def test_route_table_unavailable_route_recorded():
    # A route that cannot build at this geometry must appear as an
    # 'unavailable' row (with the reason) instead of aborting the table or
    # being silently dropped. mamba_headdim=5 does not divide d_inner=64,
    # which mamba2's geometry validator rejects — while the base KDA
    # config itself stays valid.
    cfg = small_cfg(mamba_headdim=5)
    rows = db.route_switch_table(cfg, T, batches=(1,))
    by_route = {r['route']: r for r in rows}
    assert by_route['kda']['status'] == 'ok'
    assert by_route['rwkv7']['status'] == 'ok'
    assert by_route['mamba2']['status'] == 'unavailable'
    assert by_route['mamba2']['reason']
    # Unknown routes are a caller bug, not a data condition: raise.
    with pytest.raises(ValueError):
        db.route_switch_table(cfg, T, routes=('mamba3',))


# ---------------------------------------------------------------------------
# validation / helpers
# ---------------------------------------------------------------------------

def test_validation_errors():
    cfg = small_cfg()
    with pytest.raises(ValueError):
        db.decode_step_ledger(cfg, T, batch=0)
    with pytest.raises(TypeError):
        db.decode_step_ledger(cfg, T, batch=True)
    # JSON integers are unbounded: one outside the float range is a malformed
    # geometry, not an OverflowError leaking out of the byte arithmetic.
    with pytest.raises(ValueError):
        db.decode_step_ledger(cfg, T, batch=10 ** 400)
    with pytest.raises(ValueError):
        db.decode_step_ledger(cfg, T, batch=10 ** 308)
    with pytest.raises(ValueError):
        db.route_switch_table(cfg, T, batches=(10 ** 400,))
    with pytest.raises(ValueError):
        db.decode_step_ledger(cfg, T, itemsize=0)
    with pytest.raises(ValueError):
        db.decode_traffic_breakdown(cfg, 0)
    with pytest.raises(TypeError):
        db.decode_traffic_breakdown('not-a-cfg', T)
    explicit = small_cfg(layout=('memory',) * 3 + ('csa', 'hca'))
    with pytest.raises(ValueError):
        db.route_switch_table(explicit, T)
    with pytest.raises(ValueError):
        db.route_switch_table(cfg, T, batches=())
    with pytest.raises(ValueError):
        db.bandwidth_bound_ms(1e6, 0.0)
    with pytest.raises(ValueError):
        db.bandwidth_bound_ms(-1.0, 100.0)
    with pytest.raises(ValueError):
        db.bandwidth_bound_ms(1e308, 1e-308)
    for bad in (True, float('nan'), float('inf'), 10 ** 400):
        with pytest.raises(ValueError):
            db.bandwidth_bound_ms(bad, 100.0)
        with pytest.raises(ValueError):
            db.bandwidth_bound_ms(1e6, bad)
    with pytest.raises(ValueError):
        db.attention_tower_weight_bytes(cfg, itemsize=0)
    with pytest.raises(ValueError):
        db.attention_tower_weight_bytes(cfg, itemsize=10 ** 400)


def test_bandwidth_bound_units():
    assert db.bandwidth_bound_ms(1e9, 1000.0) == pytest.approx(1.0)
    assert db.bandwidth_bound_ms(0, 100.0) == 0.0
    assert db.bandwidth_bound_ms(4e9, 500.0) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# per-operator storage policy (FP8 recurrent state)
# ---------------------------------------------------------------------------

def test_fp8_state_storage_pricing():
    cfg = small_cfg()
    fp8 = FP8Spec()
    base = db.decode_traffic_breakdown(cfg, T)
    opt = db.decode_traffic_breakdown(cfg, T,
                                      storage={'memory': fp8})
    assert opt['bytes_per_element_by_op'] == {
        'memory': 1.03125, 'csa': 4.0, 'hca': 4.0}
    assert opt['read_bytes_by_op']['csa'] == base['read_bytes_by_op']['csa']
    assert opt['read_bytes_by_op']['hca'] == base['read_bytes_by_op']['hca']
    assert opt['read_bytes_by_op']['memory'] == pytest.approx(
        base['read_elements_by_op']['memory'] * 1.03125)
    assert opt['total_bytes_per_token'] < base['total_bytes_per_token']
    assert opt['read_bytes_per_token'] + opt['write_bytes_per_token'] \
        == pytest.approx(opt['total_bytes_per_token'])

    led = db.decode_step_ledger(cfg, T, batch=8,
                                storage={'memory': fp8},
                                weight_itemsize=2)
    fp8_state = 8 * opt['total_bytes_per_token']
    assert led['state_bytes_per_step'] == pytest.approx(fp8_state)
    assert led['weight_bytes_per_step'] \
        == db.attention_tower_weight_bytes(cfg, itemsize=2)['bytes']
    assert led['total_bytes_per_step'] == pytest.approx(
        led['weight_bytes_per_step'] + fp8_state)

    old_switch = db.weights_state_switch_batch(cfg, T)
    new_switch = db.weights_state_switch_batch(
        cfg, T, storage={'memory': fp8}, weight_itemsize=2)
    manual = (db.attention_tower_weight_bytes(cfg, itemsize=2)['bytes']
              / opt['total_bytes_per_token'])
    assert new_switch == pytest.approx(manual)
    # Shrinking the state moves the switch to a larger batch; with fp16
    # weights the switch is smaller than the fp32-weight/fp8-state one but
    # still larger than the all-fp32 element-ratio baseline.
    fp32_switch = db.weights_state_switch_batch(
        cfg, T, storage={'memory': fp8})
    assert fp32_switch > new_switch > old_switch

    rows = db.route_switch_table(cfg, T, batches=(1, 16),
                                 storage={'memory': fp8})
    assert all(r['status'] == 'ok' for r in rows)
    base_rows = db.route_switch_table(cfg, T, batches=(1, 16))
    for row, base_row in zip(rows, base_rows):
        assert row['state_bytes_per_token'] \
            < base_row['state_bytes_per_token']
        assert row['weights_state_switch_batch'] \
            > base_row['weights_state_switch_batch']
        assert row['bytes_per_element_by_op']['memory'] == 1.03125

    summary = db.storage_summary({'memory': fp8})
    assert summary['bytes_per_element_by_op']['memory'] == 1.03125
    assert summary['spec']['memory']['format'] == 'e4m3fn'
    with pytest.raises(ValueError):
        db.decode_traffic_breakdown(cfg, T, storage={'mamba': fp8})
    with pytest.raises(TypeError):
        db.decode_traffic_breakdown(cfg, T, storage={'memory': 'fp8'})
    with pytest.raises(TypeError):
        db.decode_traffic_breakdown(cfg, T, storage={'memory': 10 ** 400})
    with pytest.raises(ValueError):
        db.decode_traffic_breakdown(cfg, T, storage={'memory': 1e308})


# ---------------------------------------------------------------------------
# runner payload (pure builder, no I/O)
# ---------------------------------------------------------------------------

def test_runner_report_is_json_safe():
    import json

    from run_decode_bandwidth import KIND, build_report
    report = build_report(T=256, batches=(1, 8), routes=db.MEMORY_ROUTES,
                          d_model=32, itemsize=4, bw_gbps=100.0,
                          bw_source='explicit')
    assert report['kind'] == KIND
    assert report['config']['layout_unit'] == ['kda', 'kda', 'kda',
                                               'csa', 'hca']
    assert [r['route'] for r in report['routes']] == ['kda', 'mamba2',
                                                      'rwkv7']
    assert all(r['status'] == 'ok' for r in report['routes'])
    for row in report['routes']:
        assert set(row['by_batch']) == {'1', '8'}
        for cell in row['by_batch'].values():
            assert cell['bandwidth_bound_ms'] >= 0.0
    json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
