"""Regression tests for ``lm_consistency.py`` (LM prefill-decode + roofline).

These pin the pretraining-LM decode path:

1. ``LMWithHybrid`` full-sequence (prefill) logits == token-by-token decode
   through ``run_decoding.HybridDecoding`` (KDA recurrent state + per-layer
   CSA/HCA incremental caches), with the clone carrying the exact tower
   weights.
2. The decode-phase roofline accounting: KDA compulsory state bytes, the
   memory-bound classification at an explicit ridge point, CSA's O(T)
   indexer scan growth, and input validation.

Run via pytest::

    pytest -q test_lm_consistency.py

Or directly::

    python3 test_lm_consistency.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import lm_consistency  # noqa: E402
from ops_fused import HybridConfig  # noqa: E402
from run_kv_cache import prefill_flops  # noqa: E402
from train_lm_autodl import LMWithHybrid  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def test_lm_prefill_decode_consistency(device='cpu'):
    """Training-time prefill == cached incremental decode at the LM level.

    ``train_lm_autodl.LMWithHybrid`` is TRAINED with a full-sequence forward;
    generation from the trained checkpoint runs token-by-token through
    ``run_decoding.HybridDecoding`` (KDA recurrent state + one CSA/HCA
    incremental cache per sub-layer). This pins the two paths together
    end-to-end: same weights, same geometry, logits agree to fp32 tolerance
    and top-1 predictions match exactly.

    Geometry notes:
      * ``kda_chunk_size=4`` with ``T=24 > 4`` makes the training forward run
        the CHUNKED KDA path while each decode step (T=1) falls back to the
        RECURRENT path — so the check also exercises the chunk/recurrent
        boundary inside the full LM stack, not just CSA/HCA cache threading.
      * ``csa_topk=100`` selects every visible compressed block, sidestepping
        ``torch.topk`` tie-breaking differences between the batched prefill
        indexer and the per-token decode indexer (README fairness note #4).
        Sparse top-k SELECTION equality is a separate property and is NOT
        claimed by this test.
    """
    print("\nTest: LM prefill vs cached incremental decode consistency")
    from run_decoding import HybridDecoding

    torch.manual_seed(2027)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=4,
        csa_m=4, csa_topk=100, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=8, hca_rope_dim=8,
        kda_backend='reference',
    )
    model = LMWithHybrid(17, cfg).to(device).eval()
    T, prefill_len = 24, 16
    ids = torch.randint(0, 17, (1, T),
                        generator=torch.Generator().manual_seed(7)).to(device)

    report = lm_consistency.run_lm_consistency_check(
        model, ids, prefill_len=prefill_len, device=device)

    # The clone must carry the LM tower's exact weights (path equivalence,
    # not weight re-initialization).
    clone = lm_consistency.build_decoding_clone(model)
    src = model.hybrid.state_dict()
    dst = clone.model.state_dict()
    weights_equal = (set(src) == set(dst)
                     and all(torch.equal(src[k], dst[k]) for k in src))

    # Passing both cfg and a CONFLICTING csa_topk must raise, not silently
    # pick one (the clone path relies on cfg being authoritative).
    conflict_raised = False
    try:
        HybridDecoding(cfg=cfg, csa_topk=cfg.csa_topk + 1)
    except ValueError:
        conflict_raised = True

    # A report destined for JSON must not smuggle NaN/Inf (strict dumps).
    json_ok = True
    try:
        json.dumps(report, allow_nan=False)
    except ValueError:
        json_ok = False

    # start beyond the sequence must raise rather than silently returning an
    # empty comparison.
    edge_raised = False
    try:
        lm_consistency.consistency_metrics(
            torch.zeros(1, 3, 5), torch.zeros(1, 3, 5), start=3)
    except ValueError:
        edge_raised = True

    _ok('lm_prefill_decode_consistency max logit diff',
        report['max_abs_logit_diff'] < 1e-5,
        f"max_diff={report['max_abs_logit_diff']:.3e} (tol=1e-5), "
        f"decode-only={report['decode_max_abs_logit_diff']:.3e}")
    _ok('lm_prefill_decode_consistency top1 agreement',
        report['top1_agreement'] == 1.0
        and report['decode_top1_agreement'] == 1.0,
        f"all={report['top1_agreement']:.4f}, "
        f"decode-only={report['decode_top1_agreement']:.4f}, "
        f"consistent={report['consistent']}")
    _ok('lm_decode_clone_carries_exact_weights',
        weights_equal and conflict_raised,
        f'weights_equal={weights_equal}, conflict_raised={conflict_raised}')
    _ok('lm_consistency_report_is_json_strict',
        json_ok and edge_raised,
        f'strict_dumps_ok={json_ok}, start_oob_raises={edge_raised}')


def test_zero_decode_report_is_defined(device='cpu'):
    """``prefill_len == T`` (including the default T=1) must not crash."""
    torch.manual_seed(2028)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=4,
        n_kda=1, n_csa=0, n_hca=0,
        kda_backend='reference',
    )
    model = LMWithHybrid(17, cfg).to(device).eval()
    ids = torch.randint(0, 17, (1, 1),
                        generator=torch.Generator().manual_seed(8)).to(device)
    report = lm_consistency.run_lm_consistency_check(model, ids, device=device)
    _ok('zero-decode report has no decode positions',
        report['n_decode'] == 0 and report['prefill_len'] == 1, '')
    _ok('zero-decode metrics are undefined, not fabricated',
        report['decode_max_abs_logit_diff'] is None
        and report['decode_top1_agreement'] is None, '')
    _ok('zero-decode report stays strict-JSON safe',
        json.dumps(report, allow_nan=False) is not None, '')


def test_selection_diagnostic(device='cpu'):
    """The probe must separate top-k tie artifacts from real state bugs.

    1. Synthetic capture records through ``lm_consistency._summarize_selection``
       pin the accounting exactly: per-region totals, agreements, and the
       mismatch entries (layer/position/region/full/inc sets).
    2. End-to-end on a select-all indexer (``csa_topk=100`` > visible blocks):
       selection sets cannot disagree, so the report must show agreement 1.0
       and diagnosis ``'consistent'`` — never a false top-k diagnosis.
    3. When drift IS observed, ``diagnosis`` is ``'topk_selection_divergence'``
       IFF the selections actually disagree — the report must not blame cache
       state (``'state_or_math_divergence'``) for a tie-breaking artifact.
    """
    print("\nTest: prefill/decode top-k selection diagnostic")
    from run_decoding import HybridDecoding  # noqa: F401  (import sanity)

    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8, kda_chunk_size=4,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        n_kda=0, n_csa=1, n_hca=0,
    )

    # --- 1: synthetic capture (deterministic planted mismatch). ---
    def row(pairs):
        return torch.tensor(pairs, dtype=torch.long).unsqueeze(0)

    lm_full = row([[0, 1], [1, 2], [2, 3], [3, 0]])          # [1, 4, 2]
    clone_pre = row([[0, 1], [1, 2]])                        # [1, 2, 2]
    inc0 = row([[2, 3]])                                     # t=2 matches
    inc1 = row([[0, 1]])                                     # t=3 MISMATCH
    summary = lm_consistency._summarize_selection(
        {'full': [lm_full, clone_pre], 'inc': [inc0, inc1]},
        cfg, prefill_len=2, seq_len=4)
    mm = summary['mismatches']
    syn_ok = (
        summary['n_compared'] == 4
        and summary['agreement'] == 0.75
        and summary['decode_agreement'] == 0.5
        and len(mm) == 1 and mm[0] == {'layer': 0, 'position': 3,
                                       'region': 'decode',
                                       'full': [0, 3], 'inc': [0, 1]})
    _ok('selection_summary synthetic counts', syn_ok,
        f"agreement={summary['agreement']}, "
        f"decode={summary['decode_agreement']}, mismatches={mm}")

    short = lm_consistency._summarize_selection(
        {'full': [lm_full, clone_pre], 'inc': [inc0, inc1]},
        cfg, prefill_len=1, seq_len=4)
    _ok('selection_summary needs vectorized prefill',
        short['agreement'] is None and 'prefill_len >= 2' in short['note'],
        f"agreement={short['agreement']}, note={short['note'][:60]!r}")

    # --- 2: end-to-end select-all control (no ties to break). ---
    torch.manual_seed(31)
    cfg_dense = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8, kda_chunk_size=4,
        csa_m=4, csa_topk=100, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=8, hca_rope_dim=8, kda_backend='reference',
    )
    model = LMWithHybrid(17, cfg_dense).to(device).eval()
    ids = torch.randint(0, 17, (1, 24),
                        generator=torch.Generator().manual_seed(41)).to(device)
    report = lm_consistency.run_lm_consistency_check(
        model, ids, prefill_len=16, device=device)
    sel = report['selection']
    dense_ok = (sel['agreement'] == 1.0 and sel['decode_agreement'] == 1.0
                and sel['mismatches'] == []
                and report['diagnosis'] == 'consistent')
    _ok('selection diagnostic select-all control', dense_ok,
        f"agreement={sel['agreement']}, diagnosis={report['diagnosis']}")

    # --- 3: diagnosis rule — never blame state when selections differ. ---
    diag = report.get('diagnosis')
    rule_ok = (
        (report['consistent'] and diag == 'consistent')
        or (not report['consistent'] and diag != 'consistent'
            and diag in ('topk_selection_divergence',
                         'state_or_math_divergence'))
    ) and ((diag == 'topk_selection_divergence')
           == (sel['agreement'] is not None and sel['agreement'] < 1.0))
    _ok('selection diagnostic verdict rule', rule_ok,
        f"consistent={report['consistent']}, diagnosis={diag}, "
        f"agreement={sel['agreement']}")


def test_decode_roofline_accounting(device='cpu'):
    """Analytic decode-phase roofline: KDA state bytes, AI, bound, scaling.

    1. KDA-only geometry: bytes/token must equal the compulsory state
       traffic ``3 * HV * K * V`` elements (delta-term read + output read +
       state WRITE) times ``itemsize`` — the exact decode analogue of the
       Exp 3/7 byte model, so the decode roofline can never silently diverge
       from the state accounting used everywhere else in the repo.
    2. Explicit peaks pin the ridge point: a tiny-geometry decode must
       classify as MEMORY-bound, and the reported AI must equal
       ``flops_per_token_amortized / bytes_per_token`` exactly.
    3. CSA's lightning-indexer scan is O(T): per-token byte reads must grow
       from a short to a long context.
    4. Input validation: T < 1 is rejected.
    """
    print("\nTest: decode-phase roofline accounting (KDA/CSA)")

    # --- 1+2: KDA-only geometry, hand-computed state bytes. ---
    cfg_kda = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=4,  # keep the T=16 prefill truly on the chunk path
        n_kda=1, n_csa=0, n_hca=0,
    )
    T = 16
    point = lm_consistency.decode_roofline(
        cfg_kda, T, device='cpu', itemsize=4,
        measured_decode_ms_per_token=2.0,
        peak_tflops=1.0, peak_bw_gbps=100.0)
    state = cfg_kda.n_heads_v * cfg_kda.head_dim_k * cfg_kda.head_dim_v
    expected_bytes = 3 * state * 4
    flops = float(prefill_flops(
        'hybrid_kch', T, **lm_consistency.lm_tower_geometry(cfg_kda)))
    ridge = 1e12 / (100.0 * 1e9)  # 10 FLOPs/byte
    ai_ok = abs(point['arithmetic_intensity_flops_per_byte']
                - flops / T / expected_bytes) < 1e-9
    _ok('decode_roofline kda_state_bytes',
        point['bytes_per_token'] == expected_bytes
        and point['elements_per_token'] == 3 * state,
        f"bytes={point['bytes_per_token']} (expected {expected_bytes}), "
        f"elements={point['elements_per_token']}")
    _ok('decode_roofline memory_bound_classification',
        point['roofline_bound'] == 'memory'
        and point['arithmetic_intensity_flops_per_byte'] < ridge
        and ai_ok
        and abs(point['achieved_bw_gbps']
                - expected_bytes / 2.0e-3 / 1e9) < 1e-9,
        f"AI={point['arithmetic_intensity_flops_per_byte']:.4f} "
        f"(ridge={ridge}), bound={point['roofline_bound']}, "
        f"achieved_bw={point['achieved_bw_gbps']:.3f} GB/s")

    # --- 3: CSA indexer-scan growth with context length. ---
    cfg_csa = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        n_kda=0, n_csa=1, n_hca=0,
    )
    short = lm_consistency.decode_elements_per_token(cfg_csa, 16)
    long = lm_consistency.decode_elements_per_token(cfg_csa, 256)
    _ok('decode_roofline csa_reads_grow_with_context',
        long['read_elements_per_token'] > short['read_elements_per_token'],
        f"reads@T=16={short['read_elements_per_token']:.1f} < "
        f"reads@T=256={long['read_elements_per_token']:.1f}")

    # --- 4: validation. ---
    raised = False
    try:
        lm_consistency.decode_elements_per_token(cfg_kda, 0)
    except ValueError:
        raised = True
    _ok('decode_roofline rejects T<1', raised, '')


def test_decode_elements_reject_unrepresentable_memory_geometry():
    """Huge SSM state geometry must fail as ``ValueError``, not raw overflow.

    ``decode_elements_per_token`` multiplies unbounded config integers into
    the compulsory read/write totals. For an extreme but structurally valid
    Mamba-2 / RWKV-7 geometry, the old code did ``reads += memory_reads``
    with a Python int too large for a float and leaked the interpreter's
    ``OverflowError`` instead of the module's documented validation error.
    """
    print("\nTest: decode elements reject unrepresentable memory geometry")

    cfg_mamba = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        mamba_d_state=8, mamba_headdim=4, mamba_expand=2,
        n_kda=1, n_csa=0, n_hca=0, memory_type='mamba2')
    mamba = lm_consistency.decode_elements_per_token(cfg_mamba, 16)
    _ok('normal Mamba-2 decode elements stay finite',
        mamba['memory_read_elements_per_token'] == 512.0
        and mamba['memory_write_elements_per_token'] == 512.0
        and mamba['total_elements_per_token'] == 1024.0,
        f'{mamba}')

    cfg_rwkv = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        rwkv_head_dim=8, rwkv_gate_lora=8,
        rwkv_decay_lora=8, rwkv_a_lora=8,
        n_kda=1, n_csa=0, n_hca=0, memory_type='rwkv7')
    rwkv = lm_consistency.decode_elements_per_token(cfg_rwkv, 16)
    _ok('normal RWKV-7 decode elements stay finite',
        rwkv['memory_read_elements_per_token'] == 256.0
        and rwkv['memory_write_elements_per_token'] == 256.0
        and rwkv['total_elements_per_token'] == 512.0,
        f'{rwkv}')

    huge = 10 ** 200
    cases = [
        ('Mamba-2', HybridConfig(
            d_model=huge, n_heads_qk=2, n_heads_v=2,
            head_dim_k=16, head_dim_v=16,
            mamba_d_state=huge, mamba_headdim=1, mamba_expand=1,
            n_kda=1, n_csa=0, n_hca=0, memory_type='mamba2')),
        ('RWKV-7', HybridConfig(
            d_model=huge, n_heads_qk=2, n_heads_v=2,
            head_dim_k=16, head_dim_v=16, rwkv_head_dim=huge,
            n_kda=1, n_csa=0, n_hca=0, memory_type='rwkv7')),
    ]
    for name, cfg in cases:
        detail = ''
        try:
            lm_consistency.decode_elements_per_token(cfg, 16)
            detail = 'no exception'
        except OverflowError as exc:
            detail = f'raw OverflowError: {exc}'
        except ValueError as exc:
            detail = str(exc)
        _ok(f'{name} huge state rejected as ValueError',
            detail.startswith('decode_elements_per_token:')
            and 'unrepresentable as a finite float' in detail,
            detail)


def test_consistency_metrics_start_validation():
    """``start`` must be a non-negative exact int.

    A negative ``start`` used to slice from the END of the sequence
    (silently comparing the wrong positions), and ``True`` / ``1.0`` were
    accepted as position 1. Both are caller bugs, not valid comparisons.
    """
    a = torch.zeros(1, 3, 5)
    b = torch.zeros(1, 3, 5)
    cases = [(-1, ValueError), (True, TypeError), (1.0, TypeError)]
    for bad, exc in cases:
        raised = False
        try:
            lm_consistency.consistency_metrics(a, b, start=bad)
        except exc:
            raised = True
        _ok(f'consistency_metrics rejects start={bad!r}',
            raised, f'expected {exc.__name__}')
    ok = lm_consistency.consistency_metrics(a, b, start=0)
    _ok('consistency_metrics still accepts start=0',
        ok['n_positions'] == 3 and ok['consistent'], f'{ok}')


def test_decode_roofline_host_peaks_fallback():
    """``host_peaks`` measures on-host and labels the source; explicit wins."""
    print("\nTest: decode roofline host peaks resolution")
    explicit = lm_consistency.host_peaks(
        'cpu', peak_tflops=3.5, peak_bw_gbps=42.0)
    _ok('explicit peaks win with source label',
        explicit['peak_tflops'] == 3.5 and explicit['peak_bw_gbps'] == 42.0
        and explicit['peak_sources']['peak_tflops'] == 'explicit'
        and explicit['peak_sources']['peak_bw_gbps'] == 'explicit',
        f"{explicit['peak_sources']}")


_TEST_FUNCS = [
    test_lm_prefill_decode_consistency,
    test_consistency_metrics_start_validation,
    test_selection_diagnostic,
    test_decode_roofline_accounting,
    test_decode_elements_reject_unrepresentable_memory_geometry,
    test_decode_roofline_host_peaks_fallback,
]


def main() -> int:
    print('=' * 70)
    print('LM prefill-decode consistency / decode-roofline Regression Tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: AssertionError: {e}")
            results.append(False)
        except Exception as e:
            print(f"  [CRASH] {fn.__name__}: {type(e).__name__}: {e}")
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
