"""Regression tests for the TTFT storage-budget ledger (Exp 17).

What is pinned here
-------------------
``ttft_budget`` composes two lower bounds:

    TTFT >= max(prefill compute, prefill memory) + first-decode memory

and prices both with a per-operator storage policy. These tests lock the
contracts that would otherwise drift silently:

* prefill FLOPs equal the canonical route-aware ``run_kv_cache`` row (KDA
  and Mamba-2);
* storage-only FP8 changes bytes but never the FLOP count, and the TTFT
  speedup appears only when the weighted phase is memory-bound;
* a long prompt at a realistic ridge point is compute-bound, so shrinking
  weights barely moves TTFT (the negative result that motivates the
  separate ``compute_speedup`` assumption);
* the first decode step uses the *marginal* cache budget at position
  ``T``, not the prefill average, so long contexts pay more than the
  steady-state ledger reports;
* batch moves the prefill across the compute/memory ridge;
* input validation and JSON-strict output.

Run via pytest::

    pytest -q test_ttft_budget.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402

import decode_bandwidth as db  # noqa: E402
import kv_quant  # noqa: E402
from ops_fused import HybridConfig  # noqa: E402
from run_decode_bandwidth import build_base_cfg  # noqa: E402
from run_kv_cache import prefill_flops  # noqa: E402
from ttft_budget import (SCENARIOS, first_decode_elements,  # noqa: E402
                         layout_prefill_flops, prefill_state_elements,
                         resolve_scenario, ttft_ledger)


def small_cfg(**overrides) -> HybridConfig:
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


PEAK = 67.0
BW = 3350.0


# ---------------------------------------------------------------------------
# FLOP model
# ---------------------------------------------------------------------------

def test_prefill_flops_match_canonical_hybrid_rows():
    import lm_consistency
    for route, op in (('kda', 'hybrid_kch'), ('mamba2', 'hybrid_mamba2'),
                      ('rwkv7', 'hybrid_rwkv7')):
        cfg = small_cfg(memory_type=route)
        geo = lm_consistency.lm_tower_geometry(cfg)
        expected = float(prefill_flops(op, 128, **geo))
        assert layout_prefill_flops(cfg, 128) == pytest.approx(expected)
        assert layout_prefill_flops(cfg, 128, batch=4) == pytest.approx(
            4.0 * expected)


def test_state_elements_are_layout_aware():
    cfg = small_cfg()
    row = prefill_state_elements(cfg, 129)
    assert row['total'] == pytest.approx(sum(row['by_op'].values()))
    assert set(row['by_op']) == {'memory', 'csa', 'hca'}
    assert row['by_op']['memory'] > 0
    assert row['by_op']['csa'] > 0
    assert row['by_op']['hca'] > 0


def test_explicit_layout_with_mamba_route():
    cfg = small_cfg(memory_type='mamba2',
                    layout=('memory', 'memory', 'csa', 'hca'))
    assert set(cfg.layout) == {'mamba2', 'csa', 'hca'}
    led = ttft_ledger(cfg, 64, peak_tflops=PEAK, bw_gbps=BW)
    assert led['memory_route'] == 'mamba2'
    assert led['layout_counts'] == [2, 1, 1]

    # Mamba-2's SSD step reads the recurrent state twice (decayed update
    # and C @ S^T readout), matching kv_budget.instant_read_elements; the
    # first-decode budget must not charge it only once.
    import lm_consistency
    from run_kv_cache import kv_cache_elements
    geo = lm_consistency.lm_tower_geometry(cfg)
    one_state = float(kv_cache_elements(
        'mamba2', 1, mode='compressed_kv_only', **geo))
    row = first_decode_elements(cfg, 64)
    assert row['read_elements_by_op']['memory'] == pytest.approx(
        2 * 2 * one_state)


# ---------------------------------------------------------------------------
# storage-only FP8: bytes change, FLOPs do not
# ---------------------------------------------------------------------------

def test_fp8_storage_changes_bytes_not_flops():
    cfg = small_cfg()
    base = ttft_ledger(cfg, 16, peak_tflops=PEAK, bw_gbps=BW)
    weights = ttft_ledger(cfg, 16, peak_tflops=PEAK, bw_gbps=BW,
                          weight_storage=kv_quant.FP8Spec())
    assert weights['prefill']['flops'] == base['prefill']['flops']
    assert weights['weights']['bytes'] < base['weights']['bytes']
    assert (weights['weights']['bytes_per_element']
            == pytest.approx(kv_quant.bytes_per_element(kv_quant.FP8Spec())))
    assert (weights['storage']['state']['bytes_per_element_by_op']
            == base['storage']['state']['bytes_per_element_by_op'])


def test_short_memory_bound_prompt_gets_a_large_weight_fp8_speedup():
    cfg = build_base_cfg(256)
    base = ttft_ledger(cfg, 1, peak_tflops=PEAK, bw_gbps=BW)
    fp8 = ttft_ledger(cfg, 1, peak_tflops=PEAK, bw_gbps=BW,
                      **resolve_scenario('fp8-weights'))
    assert base['prefill']['bound'] == 'memory'
    assert (base['ttft_lower_bound_ms'] / fp8['ttft_lower_bound_ms']
            > 1.5)


def test_long_compute_bound_prompt_is_storage_insensitive():
    cfg = small_cfg()
    base = ttft_ledger(cfg, 4096, peak_tflops=PEAK, bw_gbps=BW)
    fp8 = ttft_ledger(cfg, 4096, peak_tflops=PEAK, bw_gbps=BW,
                      **resolve_scenario('fp8-weights-state'))
    assert base['prefill']['bound'] == 'compute'
    assert (base['ttft_lower_bound_ms'] / fp8['ttft_lower_bound_ms']
            < 1.01)


def test_fp8_compute_assumption_is_explicit_and_effective():
    cfg = small_cfg()
    base = ttft_ledger(cfg, 4096, peak_tflops=PEAK, bw_gbps=BW)
    fast = ttft_ledger(cfg, 4096, peak_tflops=PEAK, bw_gbps=BW,
                       **resolve_scenario(
                           'fp8-weights-state-fp8-compute'))
    assert fast['machine']['compute_speedup'] == 2.0
    assert fast['prefill']['compute_ms'] == pytest.approx(
        base['prefill']['compute_ms'] / 2.0)
    assert (base['ttft_lower_bound_ms'] / fast['ttft_lower_bound_ms']
            > 1.9)


def test_batch_moves_the_compute_memory_boundary():
    cfg = build_base_cfg(256)
    small_batch = ttft_ledger(cfg, 4, batch=1, peak_tflops=PEAK, bw_gbps=BW)
    large_batch = ttft_ledger(cfg, 4, batch=8, peak_tflops=PEAK, bw_gbps=BW)
    assert small_batch['prefill']['bound'] == 'memory'
    assert large_batch['prefill']['bound'] == 'compute'


# ---------------------------------------------------------------------------
# first-decode marginal budget
# ---------------------------------------------------------------------------

def test_batch_scales_prefill_state_writes():
    cfg = small_cfg()
    one = ttft_ledger(cfg, 64, batch=1, peak_tflops=PEAK, bw_gbps=BW)
    eight = ttft_ledger(cfg, 64, batch=8, peak_tflops=PEAK, bw_gbps=BW)
    assert (eight['prefill']['bytes']['state']
            == pytest.approx(8.0 * one['prefill']['bytes']['state']))
    for op, elements in one['prefill']['state_elements_by_op'].items():
        assert (eight['prefill']['state_elements_by_op'][op]
                == pytest.approx(8.0 * elements))
    assert eight['prefill']['bytes']['total'] == pytest.approx(
        eight['weights']['bytes'] + eight['prefill']['bytes']['activations']
        + eight['prefill']['bytes']['state'])


def test_first_decode_reads_are_marginal_not_average():
    cfg = small_cfg()
    row = first_decode_elements(cfg, 4096)
    ratio = (row['read_elements'] / row['average_read_elements'])
    assert ratio > 1.5
    assert (row['read_elements_by_op']['csa']
            > row['average_read_elements_by_op']['csa'])
    led = ttft_ledger(cfg, 4096, peak_tflops=PEAK, bw_gbps=BW)
    assert led['first_decode']['instant_vs_average_reads'] == pytest.approx(
        ratio)


def test_first_decode_memory_includes_one_weight_stream():
    cfg = small_cfg()
    led = ttft_ledger(cfg, 128, peak_tflops=PEAK, bw_gbps=BW)
    assert led['first_decode']['bytes']['total'] == pytest.approx(
        led['weights']['bytes'] + led['first_decode']['bytes']['state'])
    assert led['prefill_share'] + led['first_decode_share'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# validation / JSON strictness
# ---------------------------------------------------------------------------

def test_invalid_inputs_raise():
    cfg = small_cfg()
    with pytest.raises(ValueError):
        ttft_ledger(cfg, 0, peak_tflops=PEAK, bw_gbps=BW)
    with pytest.raises(ValueError):
        ttft_ledger(cfg, 8, batch=0, peak_tflops=PEAK, bw_gbps=BW)
    with pytest.raises(ValueError):
        ttft_ledger(cfg, 8, peak_tflops=0.0, bw_gbps=BW)
    with pytest.raises(ValueError):
        ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=-1.0)
    with pytest.raises(ValueError):
        ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=BW,
                    state_mode='guess')
    with pytest.raises(TypeError):
        ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=BW,
                    weight_storage='fp8')
    with pytest.raises(TypeError):
        ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=BW,
                    weight_storage={'memory': kv_quant.FP8Spec()})
    with pytest.raises(ValueError):
        resolve_scenario('int4')
    huge = 10 ** 400
    for bad_call in (
            lambda: ttft_ledger(cfg, huge, peak_tflops=PEAK, bw_gbps=BW),
            lambda: ttft_ledger(cfg, 8, batch=huge,
                                peak_tflops=PEAK, bw_gbps=BW),
            lambda: ttft_ledger(cfg, 8, peak_tflops=huge, bw_gbps=BW),
            lambda: ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=huge),
            lambda: ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=BW,
                                compute_speedup=huge),
            lambda: ttft_ledger(cfg, 8, peak_tflops=PEAK, bw_gbps=BW,
                                weight_elements=huge),
            lambda: layout_prefill_flops(cfg, 8, batch=huge)):
        with pytest.raises(ValueError):
            bad_call()


def test_report_is_strict_json():
    cfg = small_cfg()
    led = ttft_ledger(cfg, 64, peak_tflops=PEAK, bw_gbps=BW,
                      **resolve_scenario('fp8-weights-state'))
    json.dumps(led, allow_nan=False)
    assert set(SCENARIOS) >= {'fp16', 'fp8-weights',
                              'fp8-weights-state'}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
