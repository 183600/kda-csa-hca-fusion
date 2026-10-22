"""Regression tests for ``prefill_profile.py`` (chunked-prefill profiler).

What is pinned here
-------------------
1. Schedule construction is deterministic and strict (uniform chunks,
   literal lists, ``full``, exact-sum validation, bad-input rejection).
2. ``run_prefill_schedule`` reproduces the one-shot output for CSA, HCA
   and the hybrid stack across ``full`` / uniform / token-by-token /
   ragged schedules.
3. The continuation-chunk path uses ONE batched projection call per
   chunk (the throughput change in ``run_decoding``), while still
   matching the token-by-token continuation output.
4. ``drift_metrics`` is fp64-accurate, detects non-finite outputs, and
   reports bitwise equality separately.
5. ``profile_prefill_schedules`` emits JSON-safe throughput + drift rows
   and FAILS a schedule whose output drifts (a fast-but-wrong schedule
   cannot pass silently).

Run via pytest::

    pytest -q test_prefill_profile.py

Or directly::

    python3 test_prefill_profile.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest  # noqa: E402
import torch  # noqa: E402

from prefill_profile import (  # noqa: E402
    drift_metrics,
    expand_prefill_schedule,
    profile_prefill_schedules,
    run_prefill_schedule,
    schedule_label,
    time_prefill_schedule,
)
from run_decoding import (  # noqa: E402
    CSAAttnDecoding,
    HCAAttnDecoding,
    HybridDecoding,
)


def _tiny_csa(d_model=16):
    return CSAAttnDecoding(
        d_model, m=4, topk=100, nh=2, c=4, dc=8, nIh=1, c_I=4,
        sliding_window=4, rope_dim=4).eval()


def _tiny_hca(d_model=16):
    return HCAAttnDecoding(
        d_model, m2=4, nh=2, c=4, dc=8, sliding_window=4,
        rope_dim=4).eval()


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def test_expand_prefill_schedule_forms():
    assert expand_prefill_schedule(10, 4) == [4, 4, 2]
    assert expand_prefill_schedule(10, 'full') == [10]
    assert expand_prefill_schedule(10, 'one-shot') == [10]
    assert expand_prefill_schedule(10, '4:3:3') == [4, 3, 3]
    assert expand_prefill_schedule(10, [4, 3, 3]) == [4, 3, 3]
    assert expand_prefill_schedule(8, 4) == [4, 4]
    assert expand_prefill_schedule(0, 'full') == []
    assert expand_prefill_schedule(0, []) == []


def test_expand_prefill_schedule_rejects_ambiguous_inputs():
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, [4, 3])  # does not sum to T
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, '4:3')  # does not sum to T
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, 0)
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, [0, 10])
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, [])
    with pytest.raises(ValueError):
        expand_prefill_schedule(-1, 'full')
    with pytest.raises(TypeError):
        expand_prefill_schedule(10, True)
    with pytest.raises(ValueError):
        expand_prefill_schedule(10, 'nonsense')


def test_schedule_label_is_stable():
    assert schedule_label('full') == 'full'
    assert schedule_label('8') == '8'
    assert schedule_label(8) == '8'
    assert schedule_label([4, 4, 2]) == '4:4:2'
    assert schedule_label([4, 4, 4]) == '4'
    assert schedule_label('4:3:3') == '4:3:3'


# ---------------------------------------------------------------------------
# Execution equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('name,factory', [
    ('csa', _tiny_csa),
    ('hca', _tiny_hca),
    ('hybrid', lambda d_model=16: HybridDecoding(
        d_model, total_layers=5, csa_topk=100).eval()),
])
def test_chunk_schedules_match_one_shot(name, factory):
    torch.manual_seed(11)
    d_model, length = 16 if name != 'hybrid' else 32, 10
    x = torch.randn(1, length, d_model) * 0.1
    model = factory(d_model=d_model) if name == 'hybrid' else factory()
    with torch.no_grad():
        reference = run_prefill_schedule(model, x, 'full')
        for schedule in (length, 3, 1, [4, 3, 3]):
            out = run_prefill_schedule(model, x, schedule)
            assert out.shape == reference.shape
            drift = drift_metrics(reference, out)
            assert drift['finite']
            assert drift['max_abs'] is not None
            assert drift['max_abs'] < 1e-5, (
                f'{name} schedule={schedule} drifted by '
                f"{drift['max_abs']:.3e}")


def test_continuation_chunk_uses_one_batched_projection():
    """The chunked-prefill continuation path projects the chunk once."""
    torch.manual_seed(23)
    x = torch.randn(1, 10, 16) * 0.1
    for name, model in (('csa', _tiny_csa()), ('hca', _tiny_hca())):
        calls = []
        original = model._project

        def wrapped(H, _orig=original, _calls=calls):
            _calls.append(int(H.shape[1]))
            return _orig(H)

        model._project = wrapped
        with torch.no_grad():
            model(x[:, :5])
            calls.clear()
            y_chunk = model(x[:, 5:])
        assert calls == [5], f'{name}: projection calls {calls}'
        model._project = original

        with torch.no_grad():
            model.reset()
            model(x[:, :5])
            y_tokens = torch.cat(
                [model(x[:, i:i + 1]) for i in range(5, 10)], dim=1)
        drift = drift_metrics(y_tokens, y_chunk)
        assert drift['finite'] and drift['max_abs'] is not None
        assert drift['max_abs'] < 1e-5, f'{name}: {drift["max_abs"]:.3e}'


def test_run_prefill_schedule_validates():
    x = torch.randn(1, 6, 8)
    model = _tiny_hca(d_model=8)
    with pytest.raises(ValueError):
        run_prefill_schedule(model, x, [4, 1])  # wrong sum
    with pytest.raises(ValueError):
        run_prefill_schedule(model, x, [-1, 7])  # negative, sum matches
    with pytest.raises(ValueError):
        run_prefill_schedule(model, x, [0, 6])  # zero-sized chunk
    with pytest.raises(TypeError):
        run_prefill_schedule(model, x, [3.0, 3.0])  # non-int chunk
    with pytest.raises(ValueError):
        run_prefill_schedule(model, torch.zeros(1, 0, 8), 'full',
                             schedule=[1])  # non-empty schedule, empty input
    with pytest.raises(ValueError):
        run_prefill_schedule(model, torch.randn(1, 6), 'full')  # rank
    with pytest.raises(TypeError):
        run_prefill_schedule(object(), x, 'full')  # no reset()


# ---------------------------------------------------------------------------
# Drift metric
# ---------------------------------------------------------------------------
def test_drift_metrics_exact_and_scaled():
    ref = torch.tensor([1.0, -2.0, 3.0])
    same = drift_metrics(ref, ref.clone())
    assert same['finite'] and same['bitwise_equal']
    assert same['max_abs'] == 0.0 and same['relative_l2'] == 0.0
    assert same['cosine'] == pytest.approx(1.0)

    shifted = drift_metrics(ref, ref + 0.01)
    assert shifted['max_abs'] == pytest.approx(0.01)
    assert shifted['relative_l2'] > 0

    doubled = drift_metrics(ref, ref * 2)
    assert doubled['relative_l2'] == pytest.approx(1.0, rel=1e-12)
    assert doubled['cosine'] == pytest.approx(1.0, abs=1e-12)


def test_drift_metrics_nonfinite_and_shape_mismatch():
    ref = torch.ones(2, 3)
    bad = ref.clone()
    bad[0, 0] = float('nan')
    drift = drift_metrics(ref, bad)
    assert drift['finite'] is False
    assert drift['max_abs'] is None

    mismatched = drift_metrics(ref, torch.ones(2, 4))
    assert mismatched['shape_match'] is False
    assert mismatched['finite'] is False

    empty = drift_metrics(torch.zeros(0), torch.zeros(0))
    assert empty['finite'] and empty['cosine'] == 1.0
    assert empty['n_elements'] == 0


def test_drift_metrics_flags_dtype_mismatch():
    # Same values, different precision: numeric drift is zero but the
    # schedule contract is still violated, so dtype_match must be False.
    ref = torch.tensor([1.0, 2.0], dtype=torch.float32)
    drift = drift_metrics(ref, ref.to(torch.float64))
    assert drift['dtype_match'] is False
    assert drift['finite'] is True
    assert drift['max_abs'] == 0.0
    same = drift_metrics(ref, ref.clone())
    assert same['dtype_match'] is True


def test_drift_metrics_uses_fp64_accumulation():
    # 1 + 1e-12 rounds back to 1.0 in fp32 but is representable in fp64.
    # The metric path must cast to fp64 before reducing, not silently
    # quantize the perturbation away.
    ref = torch.tensor([1.0, 2.0], dtype=torch.float64)
    cand = ref + 1e-12
    drift = drift_metrics(ref, cand)
    assert drift['finite'] and drift['max_abs'] is not None
    assert drift['max_abs'] > 0.0
    assert drift['max_abs'] == pytest.approx(1e-12, rel=1e-6)
    assert drift['relative_l2'] is not None


# ---------------------------------------------------------------------------
# Combined profile
# ---------------------------------------------------------------------------
def test_profile_emits_throughput_and_drift_rows():
    torch.manual_seed(31)
    model = _tiny_hca()
    x = torch.randn(1, 10, 16) * 0.1
    report = profile_prefill_schedules(
        model, x, ['full', '4', '1'], repeats=2, warmup=1)
    assert report['kind'] == 'chunked_prefill_profile'
    assert report['all_passed'] is True
    assert len(report['rows']) == 3
    assert report['rows'][0]['is_reference'] is True
    labels = {row['schedule'] for row in report['rows']}
    assert labels == {'full', '4:4:2', '1'}
    for row in report['rows']:
        assert row['latency_ms'] >= 0.0
        assert row['n_chunks'] >= 1
        assert row['tokens_per_s'] is None or row['tokens_per_s'] > 0.0
        assert row['drift']['finite'] is True
        assert row['passed'] is True
    json.dumps(report, allow_nan=False)


def test_profile_handles_empty_prefill():
    # ``expand_prefill_schedule(0, ...)`` and ``run_prefill_schedule``
    # both support an empty prompt; the reference row must use the same
    # empty schedule instead of the invalid literal ``[0]``.
    model = _tiny_hca()
    x = torch.zeros(1, 0, 16)
    report = profile_prefill_schedules(
        model, x, ['full', '4'], repeats=1, warmup=0)
    assert report['prefill_len'] == 0
    assert len(report['rows']) == 1
    row = report['rows'][0]
    assert row['is_reference'] is True
    assert row['n_chunks'] == 0
    assert row['chunk_sizes'] == []
    assert row['drift']['finite'] is True
    assert row['passed'] is True
    assert report['all_passed'] is True


def test_profile_rejects_drifting_schedule():
    class _ChunkSensitive(torch.nn.Module):
        def reset(self):
            pass

        def forward(self, x):
            # Single-token calls deliberately corrupt the output, so an
            # aggressive schedule must be flagged as non-equivalent.
            return x + 0.01 if x.shape[1] == 1 else x

    x = torch.randn(1, 4, 8)
    report = profile_prefill_schedules(
        _ChunkSensitive(), x, ['full', '1'], repeats=1, warmup=0)
    assert report['all_passed'] is False
    bad = [r for r in report['rows'] if not r['is_reference']][0]
    assert bad['schedule'] == '1'
    assert bad['passed'] is False
    assert bad['drift']['max_abs'] == pytest.approx(0.01, rel=1e-5)


def test_profile_rejects_dtype_changing_schedule():
    class _DtypeSwitch(torch.nn.Module):
        def reset(self):
            pass

        def forward(self, x):
            # Token-by-token calls return fp64 with the same values, so a
            # numeric-only drift check would pass.
            out = x + 0.0
            return out.to(torch.float64) if x.shape[1] == 1 else out

    x = torch.randn(1, 4, 8)
    report = profile_prefill_schedules(
        _DtypeSwitch(), x, ['full', '1'], repeats=1, warmup=0)
    assert report['all_passed'] is False
    bad = [r for r in report['rows'] if not r['is_reference']][0]
    assert bad['schedule'] == '1'
    assert bad['passed'] is False
    assert bad['drift']['dtype_match'] is False
    assert bad['drift']['max_abs'] == 0.0


def test_timing_returns_output_of_last_run():
    model = _tiny_hca()
    x = torch.randn(1, 8, 16) * 0.1
    row, out = time_prefill_schedule(
        model, x, '4', repeats=2, warmup=1, return_output=True)
    assert row['n_chunks'] == 2
    assert out.shape == (1, 8, 16)
    ref = run_prefill_schedule(model, x, 'full')
    assert drift_metrics(ref, out)['max_abs'] < 1e-5


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_drift_metrics_rms_survives_squaring_overflow():
    # 2e200 is finite in fp64, but squaring it overflows. The RMS is still
    # representable and must be reported, not silently dropped.
    ref = torch.tensor([1e200], dtype=torch.float64)
    cand = torch.tensor([-1e200], dtype=torch.float64)
    drift = drift_metrics(ref, cand)
    assert drift['finite'] is True
    assert drift['max_abs'] == pytest.approx(2e200, rel=1e-12)
    assert drift['rms'] == pytest.approx(2e200, rel=1e-12)
    import json
    json.dumps(drift, allow_nan=False)


def test_drift_metrics_huge_finite_tensors_are_json_safe():
    import json
    a = torch.tensor([[1e154, 1e154]], dtype=torch.float64)
    drift = drift_metrics(a, a.clone())
    assert drift['cosine'] is not None
    assert abs(drift['cosine'] - 1.0) < 1e-12
    assert drift['relative_l2'] == 0.0
    json.dumps(drift, allow_nan=False)
    doubled = drift_metrics(a, a * 2)
    assert doubled['relative_l2'] == 1.0
    json.dumps(doubled, allow_nan=False)
