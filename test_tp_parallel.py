"""Tensor-parallel correctness + latency-model regression tests.

These pin the two claims of :mod:`tp_parallel`:

1. **Head sharding is numerically transparent.**  Every rank's partial
   output sums to the unsharded layer output (KDA also reassembles the
   recurrent state), a whole 3:1:1 hybrid stack produces the same MQAR
   representations under TP=2, and ``kda_stack_forward_tp`` streams
   uneven chunks while preserving the KDA incremental state and conv
   lookbacks.  This is why MQAR accuracy measured on the unsharded tower
   transfers to a TP deployment (and to a stateful decode).
2. **The latency ledger is self-consistent.**  The replicated + sharded
   FLOP split equals the canonical ``run_kv_cache.prefill_flops``, the
   TP=1 traffic equals the canonical decode accounting, KDA state traffic
   divides by the degree while CSA/HCA shared-KV traffic is replicated,
   and the ring all-reduce / sweep arithmetic is well-formed.

How to run::

    pytest -q test_tp_parallel.py
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decode_bandwidth  # noqa: E402
import tp_parallel as tp  # noqa: E402
from ops_fused import (  # noqa: E402
    CSAHybridLayer,
    HCAHybridLayer,
    HybridAttentionStack,
    HybridConfig,
    HybridKCHAttention,
    KDAHybridLayer,
)


@pytest.fixture
def device(request):
    try:
        return request.config.getoption('--device')
    except (ValueError, KeyError):
        return 'cpu'


def _kda_cfg(chunk_size: int) -> HybridConfig:
    return HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=4,
        head_dim_k=16, head_dim_v=16,
        kda_chunk_size=chunk_size,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=32, csa_dc=64,
        csa_nIh=2, csa_cI=16, csa_sliding_window=4,
        hca_m2=8, hca_nh=2, hca_c=32, hca_dc=64, hca_sliding_window=4,
    )


@pytest.mark.parametrize('chunk_size', [0, 16])
def test_kda_head_shard_equivalence(chunk_size, device):
    """KDA partial outputs/states reassemble to the unsharded result.

    Covers both the recurrent path (``chunk_size=0``) and the chunkwise
    prefill path, and a GVA geometry (H=2, HV=4) where a rank boundary
    falls inside a q/k head group.
    """
    torch.manual_seed(11)
    cfg = _kda_cfg(chunk_size)
    layer = KDAHybridLayer(cfg).to(device).eval()
    x = torch.randn(2, 32, 32, device=device)
    with torch.no_grad():
        full, full_state, _ = layer.forward_functional(x, None, None)
        parts, states = [], []
        for rank in range(2):
            out = tp.kda_layer_forward_tp(layer, x, None, rank, 2)
            parts.append(out['output'])
            states.append(out['state'])
        out_diff = (torch.stack(parts).sum(dim=0) - full).abs().max().item()
        state_diff = (torch.cat(states, dim=1) - full_state).abs().max().item()
    assert out_diff < 1e-5, f'out max|diff|={out_diff:.2e}'
    assert state_diff < 1e-5, f'state max|diff|={state_diff:.2e}'


def _assert_close(name, a, b, tol=1e-5):
    assert a.shape == b.shape, f'{name}: shape {tuple(a.shape)} != {tuple(b.shape)}'
    diff = (a.float() - b.float()).abs().max().item()
    assert diff < tol, f'{name}: max|diff|={diff:.2e} (tol {tol})'


def _assert_lookbacks_close(name, got, want):
    assert len(got) == len(want), f'{name}: {len(got)} != {len(want)} entries'
    for idx, (a, b) in enumerate(zip(got, want)):
        if a is None or b is None:
            assert a is None and b is None, \
                f'{name}[{idx}]: one side is None ({a is None} vs {b is None})'
        else:
            _assert_close(f'{name}[{idx}]', a, b)


@pytest.mark.parametrize('schedule', ([1, 3, 4, 3], [5, 6], [11],
                                      [0, 4, 0, 7]))
@pytest.mark.parametrize('tp_size', (2, 4))
def test_kda_streaming_tp_matches_full_sequence(schedule, tp_size, device):
    """Streaming TP carries the KDA incremental rule across chunk calls.

    A KDA-only stack is fed in ``schedule``-sized chunks.  Both the
    unsharded ``forward_functional`` path and the simulated TP forward
    must reproduce the one-shot full-sequence output, recurrent state and
    short-conv lookbacks.  The zero-length chunks also pin the T=0
    state-preserving contract.
    """
    torch.manual_seed(17)
    cfg = dataclasses.replace(_kda_cfg(16), n_kda=3, n_csa=0, n_hca=0)
    model = HybridAttentionStack(cfg, total_layers=3).to(device).eval()
    n_layers = len(model.layout)
    x = torch.randn(2, 11, cfg.d_model, device=device) * 0.5
    assert sum(schedule) == x.shape[1]

    with torch.no_grad():
        full, full_state, full_lbs = model.forward_functional(
            x, None, [None] * n_layers)

        outs, state, lbs, pos = [], None, [None] * n_layers, 0
        for length in schedule:
            y, state, lbs = model.forward_functional(
                x[:, pos:pos + length], state, lbs)
            outs.append(y)
            pos += length
        streamed = torch.cat(outs, dim=1)
        _assert_close('unsharded streaming output', streamed, full)
        _assert_close('unsharded streaming state', state, full_state)
        _assert_lookbacks_close('unsharded streaming lookbacks', lbs, full_lbs)

        outs, state, lbs, pos = [], None, [None] * n_layers, 0
        for length in schedule:
            res = tp.kda_stack_forward_tp(
                model, x[:, pos:pos + length], tp_size,
                kda_state=state, conv_lookbacks=lbs)
            outs.append(res['output'])
            state, lbs = res['kda_state'], res['conv_lookbacks']
            pos += length
        streamed_tp = torch.cat(outs, dim=1)

    _assert_close('TP streaming output', streamed_tp, full)
    _assert_close('TP streaming state', state, full_state)
    _assert_lookbacks_close('TP streaming lookbacks', lbs, full_lbs)
    assert state.shape == (n_layers, 2, cfg.n_heads_v,
                           cfg.head_dim_k, cfg.head_dim_v)


def test_kda_streaming_tp_contract_and_backend(device):
    """Validation, backend provenance, and the no-all-reduce state map."""
    torch.manual_seed(19)
    cfg = dataclasses.replace(_kda_cfg(16), n_kda=2, n_csa=0, n_hca=0)
    model = HybridAttentionStack(cfg, total_layers=2).to(device).eval()
    x = torch.randn(2, 7, cfg.d_model, device=device) * 0.5
    with torch.no_grad():
        out = tp.kda_stack_forward_tp(model, x, 2)
    assert out['backend'] == 'reference'
    assert out['kda_state'].shape == (
        2, 2, cfg.n_heads_v, cfg.head_dim_k, cfg.head_dim_v)
    assert all(lb is not None for lb in out['conv_lookbacks'])

    with torch.no_grad():
        forced = tp.kda_stack_forward_tp(
            model, x, 2, kda_state=out['kda_state'],
            conv_lookbacks=out['conv_lookbacks'], backend='reference')
    assert forced['backend'] == 'reference'

    with pytest.raises(ValueError):
        tp.kda_stack_forward_tp(model, x, 2, backend='not-a-backend')
    with pytest.raises(ValueError):
        tp.kda_stack_forward_tp(
            model, x, 2,
            kda_state=torch.zeros(2, 2, cfg.n_heads_v - 1,
                                  cfg.head_dim_k, cfg.head_dim_v))
    with pytest.raises(ValueError):
        tp.kda_stack_forward_tp(model, x, 2, conv_lookbacks=[None])

    hybrid = HybridAttentionStack(_kda_cfg(16), total_layers=5).to(
        device).eval()
    with pytest.raises(NotImplementedError):
        tp.kda_stack_forward_tp(hybrid, x, 2)


def test_csa_hca_head_shard_equivalence(device):
    """CSA/HCA partial outputs sum to the unsharded output."""
    cfg = tp.mqar_probe_config()
    torch.manual_seed(12)
    x = torch.randn(2, 24, cfg.d_model, device=device)
    for cls, kind in ((CSAHybridLayer, 'csa'), (HCAHybridLayer, 'hca')):
        layer = cls(cfg).to(device).eval()
        with torch.no_grad():
            full = layer(x)[0]
            parts = [tp.layer_forward_tp(layer, kind, x, rank, 2)['output']
                     for rank in range(2)]
            diff = (torch.stack(parts).sum(dim=0) - full).abs().max().item()
        assert diff < 1e-5, f'{kind} max|diff|={diff:.2e}'


def test_tp_stack_mqar_invariance(device):
    """A TP=2 hybrid stack preserves MQAR probe representations."""
    from torch.nn import Embedding
    from run_quality import make_mqar_batch

    cfg = tp.mqar_probe_config()
    torch.manual_seed(13)
    model = HybridKCHAttention(cfg, total_layers=5).to(device).eval()
    vocab = 32
    embed = Embedding(vocab, vocab).to(device)
    with torch.no_grad():
        embed.weight.copy_(torch.eye(vocab, device=device))
    gen = torch.Generator(device=device).manual_seed(0)
    x_emb, _, _ = make_mqar_batch(4, 32, 2, vocab, embed,
                                  device=device, generator=gen)
    with torch.no_grad():
        y = model(x_emb)
        y_tp = tp.hybrid_stack_forward_tp(model, x_emb, 2)['output']
    diff = (y - y_tp).abs().max().item()
    assert diff < 1e-5, f'MQAR TP max|diff|={diff:.2e}'


def test_tp_plan_ledger_and_validation():
    """Plan identities, canonical TP=1 parameter count, invalid degrees."""
    cfg = tp.mqar_probe_config()
    for degree in (1, 2):
        plan = tp.tp_plan(cfg, degree, total_layers=5)
        full = plan['full_parameter_elements']
        assert full == (plan['sharded_parameter_elements']
                        + plan['replicated_parameter_elements']
                        + plan['norm_parameter_elements'])
        assert plan['per_rank_parameter_elements'] == (
            plan['sharded_parameter_elements'] // degree
            + plan['replicated_parameter_elements']
            + plan['norm_parameter_elements'])
        assert plan['weight_shard_fraction'] <= 1.0
    plan1 = tp.tp_plan(cfg, 1)
    canonical = decode_bandwidth.attention_tower_parameter_elements(cfg)['total']
    assert plan1['full_parameter_elements'] == canonical
    assert tp.tp_plan(cfg, 2)['weight_shard_fraction'] < 1.0

    with pytest.raises(ValueError):
        tp.tp_plan(cfg, 4)           # head axes are 2
    with pytest.raises(ValueError):
        tp.head_ranges(3, 2)
    with pytest.raises(ValueError):
        tp.head_ranges(4, 0)
    mamba = dataclasses.replace(
        cfg, memory_type='mamba2', mamba_headdim=16,
        mamba_d_state=16, mamba_expand=2)
    with pytest.raises(NotImplementedError):
        tp.tp_plan(mamba, 1)


def test_tp_flops_split_matches_canonical():
    """Replicated + sharded FLOPs equal ``prefill_flops`` exactly."""
    cfg = tp.adjacent_probe_config()
    for kind in ('kda', 'csa', 'hca'):
        for T in (1, 16, 64, 257):
            split = tp.layer_flops_split(cfg, kind, T)
            assert split['replicated'] >= 0
            assert split['sharded'] >= 0
            assert split['total'] == pytest.approx(
                split['canonical_prefill_flops'], rel=1e-12, abs=1e-9)


def test_tp_decode_traffic_sharding():
    """KDA state divides by TP; shared CSA/HCA KV stays replicated."""
    cfg = tp.adjacent_probe_config()
    full = tp.tp_decode_traffic(cfg, 512, 1)
    shard = tp.tp_decode_traffic(cfg, 512, 4)
    assert full['state_bytes_per_token_per_rank'] == \
        full['full_state_bytes_per_token']
    assert shard['write_bytes_by_op_per_rank']['memory'] == pytest.approx(
        full['write_bytes_by_op_per_rank']['memory'] / 4)
    for op in ('csa', 'hca'):
        assert shard['read_bytes_by_op_per_rank'][op] == \
            full['read_bytes_by_op_per_rank'][op]
    assert shard['kv_replicated_bytes_per_token'] == \
        full['kv_replicated_bytes_per_token']
    assert shard['state_bytes_per_token_per_rank'] < \
        full['state_bytes_per_token_per_rank']


def test_tp_latency_sweep_invariants():
    """Ring all-reduce arithmetic + sweep monotonicity/contracts."""
    hw = tp.HardwareModel(peak_tflops=50.0, peak_bw_gbps=1000.0,
                          link_bw_gbps=100.0, allreduce_latency_us=5.0)
    assert tp.ring_allreduce_ms(1024, 1, hw) == 0.0
    assert tp.ring_allreduce_ms(0, 8, hw) == 0.0
    r2 = tp.ring_allreduce_ms(1 << 20, 2, hw)
    r4 = tp.ring_allreduce_ms(1 << 20, 4, hw)
    assert 0 < r2 < r4

    cfg = tp.mqar_probe_config()
    sweep = tp.tp_sweep(cfg, (1, 2), prefill_len=32, decode_batch=8,
                        total_layers=5, hardware=hw)
    assert sweep['prefill_optimum_tp'] in (1, 2)
    assert sweep['decode_optimum_tp'] in (1, 2)
    row1, row2 = sweep['rows']
    assert row1['tp_size'] == 1 and row2['tp_size'] == 2
    assert row1['prefill']['comm_ms'] == 0.0
    assert row1['decode']['comm_ms'] == 0.0
    assert row1['speedup_vs_tp1']['prefill'] == pytest.approx(1.0)
    assert row2['speedup_vs_tp1']['prefill'] > 0
    assert row2['decode']['state_bytes_per_sequence'] < \
        row1['decode']['state_bytes_per_sequence']
    assert row2['prefill']['full_flops'] > 0
    for row in sweep['rows']:
        for phase in ('prefill', 'decode'):
            for key in ('total_ms', 'core_ms', 'compute_ms', 'memory_ms',
                        'comm_ms', 'comm_share'):
                value = row[phase][key]
                assert math.isfinite(value) and value >= 0
        assert row['decode']['tokens_per_s'] > 0
        assert row['prefill']['bound'] in (
            'compute', 'memory', 'communication')
        assert row['decode']['bound'] in (
            'compute', 'memory', 'communication')

    with pytest.raises(ValueError):
        tp.tp_latency(cfg, tp_size=0, prefill_len=32)
    with pytest.raises(ValueError):
        tp.HardwareModel(peak_tflops=-1.0)


def test_hardware_from_gpu_cpu_fallback(monkeypatch):
    """CPU / no-CUDA returns the nominal HardwareModel, not a crash."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    hw = tp.HardwareModel.from_gpu('auto')
    assert hw.gpu_anchored is False
    assert hw.peak_tflops == pytest.approx(60.0)
    assert hw.peak_bw_gbps == pytest.approx(3000.0)
    assert hw.link_bw_gbps == pytest.approx(300.0)
    for value in dataclasses.asdict(hw).values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(value)


def test_hardware_from_gpu_monkeypatched_gpu(monkeypatch):
    """A real CUDA device anchors peak bandwidth and sets gpu_anchored."""

    class FakeProps:
        # Attribute names must match torch's ``_CudaDeviceProperties``.
        memory_clock_rate = 8000  # kHz
        memory_bus_width = 4096  # bits

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: FakeProps())
    hw = tp.HardwareModel.from_gpu('auto')
    assert hw.gpu_anchored is True
    expected = 8000 * 1e3 * 4096 / 8.0 / 1e9
    assert hw.peak_bw_gbps == pytest.approx(expected)
    # peak_tflops / link are left nominal (require kernel measurement).
    assert hw.peak_tflops == pytest.approx(60.0)
    assert hw.link_bw_gbps == pytest.approx(300.0)

    # A device that does not report the clock / bus width cannot anchor the
    # model, so the nominal bandwidth is kept and the provenance flag stays
    # False (the anchor must not be claimed when it did not happen).
    class BareProps:
        name = 'FakeGPU'

    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: BareProps())
    hw_bare = tp.HardwareModel.from_gpu('auto')
    assert hw_bare.gpu_anchored is False
    assert hw_bare.peak_bw_gbps == pytest.approx(3000.0)

    # Explicit nominal mode never touches torch.cuda.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    hw_nominal = tp.HardwareModel.from_gpu('nominal')
    assert hw_nominal.gpu_anchored is False
    assert hw_nominal.peak_bw_gbps == pytest.approx(3000.0)

    with pytest.raises(ValueError):
        tp.HardwareModel.from_gpu('not-a-mode')


def test_tp_decode_traffic_sliding_window_limits_reads():
    """A smaller CSA/HCA sliding window caps replicated decode reads under TP.

    The sliding-window branch only admits the most recent ``window`` tokens,
    so shrinking the window reduces the per-rank replicated CSA/HCA read
    traffic; KDA memory reads (head-scaled, divided by TP) are unaffected by
    the CSA/HCA window knob.
    """
    T = 512
    base = tp.adjacent_probe_config()
    small = dataclasses.replace(base, csa_sliding_window=8,
                                hca_sliding_window=8)
    large = dataclasses.replace(base, csa_sliding_window=T,
                                hca_sliding_window=T)
    for tp_size in (1, 4):
        s = tp.tp_decode_traffic(small, T, tp_size)
        l = tp.tp_decode_traffic(large, T, tp_size)
        assert s['read_bytes_by_op_per_rank']['csa'] < \
            l['read_bytes_by_op_per_rank']['csa']
        assert s['read_bytes_by_op_per_rank']['hca'] < \
            l['read_bytes_by_op_per_rank']['hca']
        assert s['read_bytes_by_op_per_rank']['memory'] == \
            l['read_bytes_by_op_per_rank']['memory']
        assert s['state_bytes_per_token_per_rank'] < \
            l['state_bytes_per_token_per_rank']


def test_tp_sweep_hardware_param():
    """tp_sweep accepts a hardware= anchor and rows reflect it."""
    hw_a = tp.HardwareModel(peak_tflops=50.0, peak_bw_gbps=1000.0,
                            link_bw_gbps=100.0, allreduce_latency_us=5.0)
    hw_b = tp.HardwareModel(peak_tflops=100.0, peak_bw_gbps=2000.0,
                            link_bw_gbps=200.0, allreduce_latency_us=5.0)
    cfg = tp.mqar_probe_config()
    sa = tp.tp_sweep(cfg, (1, 2), prefill_len=32, decode_batch=8,
                     total_layers=5, hardware=hw_a)
    sb = tp.tp_sweep(cfg, (1, 2), prefill_len=32, decode_batch=8,
                     total_layers=5, hardware=hw_b)
    assert sa['hardware']['peak_tflops'] == 50.0
    assert sb['hardware']['peak_tflops'] == 100.0
    assert sb['rows'][0]['prefill']['compute_ms'] <= \
        sa['rows'][0]['prefill']['compute_ms']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_kda_tp_decay_init_under_autocast(device):
    """The TP KDA path keeps the fp32 decay prior from promoting the gate."""
    cfg = dataclasses.replace(
        _kda_cfg(0), kda_decay_init='log_uniform')
    layer = KDAHybridLayer(cfg).to(device).eval()
    x = torch.randn(2, 5, cfg.d_model, device=device)
    dtype = torch.bfloat16
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        out = tp.kda_layer_forward_tp(layer, x, None, 0, 2)
    assert out['output'].dtype == dtype
    assert torch.isfinite(out['output']).all()
    assert torch.isfinite(out['state']).all()
