"""Experiment 1 — correctness verification.

This module is a thin CLI/scheduler.  The individual checks live in
the :mod:`checks` package, grouped by subsystem (KDA, CSA/HCA, hybrid,
FLOPs/cache accounting, decoding cache, Mamba-2, RWKV-7).  Public and
private test names are re-exported here for compatibility with the
existing pytest collection and ``from run_correctness import ...`` users.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (append_run_manifest, capture_provenance,
                          configure_torch_for_device, make_run_manifest_entry,
                          sanitize_for_json, write_json_atomic)

from checks._shared import *  # noqa: F401,F403
from checks.kda import *  # noqa: F401,F403
from checks.csa import *  # noqa: F401,F403
from checks.hca import *  # noqa: F401,F403
from checks.hybrid import *  # noqa: F401,F403
from checks.flops import *  # noqa: F401,F403
from checks.decoding import *  # noqa: F401,F403
from checks.mamba import *  # noqa: F401,F403
from checks.rwkv import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

def main():
    info = configure_torch_for_device()
    device = info.device
    # fp64 tests run on the detected device. On T4 fp64 is slow but correct,
    # and these tests are tiny (T<=128). Keeping them on-device verifies the
    # GPU code path end-to-end.
    logger.info('=' * 70)
    logger.info(f'Experiment 1: Correctness Verification ({device})')
    logger.info('=' * 70)
    all_results = []
    all_results += _run_safe(test_kda_chunk_vs_recurrent, device)
    all_results += _run_safe(test_kda_backend_reference_dispatch, device)
    all_results += _run_safe(test_kda_gva, device)
    all_results += _run_safe(test_kda_chunk_gva, device)
    # Review-fix 1.1-a regression: compiled_recurrent_kda(fullgraph=True)
    # must still compile after review-fix 1.1 added the non-finite-output
    # check (which is data-dependent branching and, unguarded, breaks
    # torch.compile tracing). Has a real one-time torch.compile cost
    # (several seconds even for tiny T on CPU) — marked 'slow' in
    # conftest.py so `pytest -m "not slow"` can skip it during fast loops.
    all_results += _run_safe(test_compiled_recurrent_kda_fullgraph, device)
    # Review-fix 1.2 regression: scripted_chunk_kda (both use_script=True
    # and False) agrees with naive_chunk_kda / naive_recurrent_kda now
    # that the two share _chunk_kda_prepare/_chunk_kda_finalize instead of
    # hand-duplicated setup/tail code.
    all_results += _run_safe(test_scripted_chunk_kda_matches_naive, device)
    all_results += _run_safe(test_csa_causality, device)
    all_results += _run_safe(test_hca_forward_smoke, device)
    all_results += _run_safe(test_fused_hybrid, device)
    # New reviewer-driven checks.
    all_results += _run_safe(test_overlap_causality, device)
    all_results += _run_safe(test_kda_gradient, device)
    # P0 regression: g-clamp prevents catastrophic state decay.
    all_results += _run_safe(test_kda_g_clamp, device)
    # RWKV-inspired per-channel (multi-timescale) KDA decay initialization.
    all_results += _run_safe(test_kda_per_channel_decay_init, device)
    # Review-fix 1.1 regression: non-finite KDA output triggers a warning
    # instead of silently propagating (see ops_kda._warn_if_nonfinite).
    all_results += _run_safe(test_kda_unnormalized_input_warns, device)
    # Regression test for chunk-vs-recurrent gradient agreement (fp64).
    all_results += _run_safe(test_kda_chunk_vs_recurrent_gradient, device)
    # DPLR transition certificate (chunk map vs per-step product) and
    # the work-efficient two-phase scan.
    all_results += _run_safe(test_kda_dplr_transition_certificate, device)
    all_results += _run_safe(test_csa_indexer_validity, device)
    # Boundary regression: compressed blocks become visible when windows close.
    all_results += _run_safe(test_causal_block_mask_window_close, device)
    # P0 regression: w_idx=None branch of csa_lightning_indexer.
    all_results += _run_safe(test_csa_indexer_w_idx_none, device)
    # P0 regression: topk=0 guard pushed down into csa_lightning_indexer.
    all_results += _run_safe(test_csa_indexer_topk_zero, device)
    # P0-4 regression: STE must make the CSA indexer parameters trainable.
    all_results += _run_safe(test_csa_indexer_ste_gradient, device)
    # Review-fix 1.3 regression: fuse_projections=True/False forward
    # value is identical; only the use_ste=False gradient contract
    # (None vs zero-but-non-None) differs, as documented in naive_csa.
    all_results += _run_safe(test_csa_fuse_projections_grad_contract, device)
    # P0-2 regression: ste_mode='full_softmax' must be a distinct branch
    # (not silently aliased to 'topk_columns').
    all_results += _run_safe(test_csa_indexer_ste_full_softmax, device)
    # P0-3 regression: detach_lookback=False enables cross-chunk BPTT.
    all_results += _run_safe(test_kda_cross_chunk_bptt, device)
    # P1-2 regression: KDA state preserves fp32 precision for fp16/bf16 inputs.
    all_results += _run_safe(test_kda_state_dtype_preservation, device)
    all_results += _run_safe(test_hca_sliding_window_causality, device)
    all_results += _run_safe(test_csa_full_pipeline_causality, device)
    # Regression tests for bugs found during code review.
    all_results += _run_safe(test_hybrid_padding_no_crash, device)
    all_results += _run_safe(test_hybrid_state_buffer_registration, device)
    all_results += _run_safe(test_bench_hybrid_no_grad_inference, device)
    all_results += _run_safe(test_hybrid_per_layer_kda_state, device)
    all_results += _run_safe(test_csa_hca_right_padding_correctness, device)
    all_results += _run_safe(test_csa_hca_sink_numerical_correctness, device)
    # New tests for dtype consistency and chunk padding edge cases.
    all_results += _run_safe(test_kda_chunk_nondivisible_T, device)
    all_results += _run_safe(test_csa_hca_fp16_dtype_consistency, device)
    all_results += _run_safe(test_kda_initial_state_dtype_mismatch, device)
    # Regression test for hybrid backward gradient flow.
    all_results += _run_safe(test_hybrid_backward_produces_grads, device)
    # Additional edge-case tests for broader coverage.
    all_results += _run_safe(test_csa_hca_bf16_dtype_consistency, device)
    all_results += _run_safe(test_csa_hca_no_sink_no_sliding_window, device)
    all_results += _run_safe(test_csa_topk_edge_cases, device)
    all_results += _run_safe(test_kda_single_token_decode, device)
    all_results += _run_safe(test_csa_hca_extreme_sink_values, device)
    all_results += _run_safe(test_hybrid_no_kda_layout, device)
    # Regression tests for n_csa=0 / n_hca=0 layouts (ablation sweep coverage).
    all_results += _run_safe(test_hybrid_no_csa_layout, device)
    all_results += _run_safe(test_hybrid_no_hca_layout, device)
    # Regression tests for the internal-padding + topk=0 fixes.
    all_results += _run_safe(test_csa_hca_non_divisible_T, device)
    all_results += _run_safe(test_csa_topk_zero, device)
    all_results += _run_safe(test_hca_T_smaller_than_m2, device)
    # Regression test for weight-decay parameter grouping.
    all_results += _run_safe(test_weight_decay_param_groups, device)
    # Regression test for T=0 (empty sequence) edge case.
    all_results += _run_safe(test_csa_hca_zero_length_sequence, device)
    # Regression test for csa_compress_kv_overlapped T=0 (direct call).
    all_results += _run_safe(test_csa_compress_kv_overlapped_zero_length, device)
    # Inference overlapped compression avoids the full-size shifted pads.
    all_results += _run_safe(test_overlapped_compress_no_pad_parity, device)
    # Block-tiled compression: auto/explicit tiles are value-equivalent and
    # bound the logits transient at long context.
    all_results += _run_safe(test_compress_chunk_tiling_parity, device)
    # Regression test for missing input validation (c_I=0, nIh=0,
    # sliding_window<0, topk<0, total_layers<0).
    all_results += _run_safe(test_csa_hca_input_validation, device)
    # Regression tests for KDA state dtype/batch-size handling and
    # prefill_flops head-count correctness (recently fixed).
    all_results += _run_safe(test_hybrid_kda_state_dtype_mismatch, device)
    all_results += _run_safe(test_hybrid_kda_state_batch_size_change, device)
    all_results += _run_safe(test_prefill_flops_head_count, device)
    all_results += _run_safe(test_prefill_flops_causal_block_entries, device)
    # P1-4 regression: ceil block count at non-divisible T.
    all_results += _run_safe(test_kv_cache_ceil_block_count, device)
    all_results += _run_safe(test_geometric_decode_cache_capacity, device)
    all_results += _run_safe(test_kv_cache_full_accounting_runtime_state, device)
    all_results += _run_safe(test_prefill_flops_softmax_gqa_projections, device)
    all_results += _run_safe(test_prefill_flops_kch_output_projections, device)
    # Exp 17: the distance buckets and the recall-conditioned FLOPs ledger
    # must agree with the structural reachability contract.
    all_results += _run_safe(
        test_recall_distance_buckets_match_structural_window, device)
    all_results += _run_safe(test_recall_flops_audit, device)
    all_results += _run_safe(test_standalone_kda_gate_matches_hybrid, device)
    all_results += _run_safe(test_decoding_batch_size_change, device)
    # Incremental decoding cache tests (ops_decoding_cache.py).
    # These verify the new CSADecodingCache / HCADecodingCache that
    # close the Exp 6 scope gap (CSA / HCA decode latency).
    all_results += _run_safe(test_csa_decoding_cache_correctness, device)
    all_results += _run_safe(test_hca_decoding_cache_correctness, device)
    all_results += _run_safe(test_csa_decoding_cache_compressed_blocks_match, device)
    all_results += _run_safe(test_csa_decoding_cache_sliding_window_ring_buffer, device)
    all_results += _run_safe(test_csa_decoding_cache_indexer_dynamic_update, device)
    all_results += _run_safe(test_csa_decoding_cache_prefill_then_decode, device)
    all_results += _run_safe(test_hca_decoding_cache_compressed_blocks_match, device)
    all_results += _run_safe(test_decoding_cache_reset_clears_state, device)
    # Integrated Exp6 hybrid decode-cache path (KDA state + CSA/HCA caches).
    all_results += _run_safe(test_hybrid_decoding_cache_matches_full_sequence, device)
    # The tensor-parallel streaming oracle must carry the same KDA state,
    # conv lookbacks, and CSA/HCA caches and reproduce the unsharded decoder.
    all_results += _run_safe(test_hybrid_streaming_tp_matches_unsharded, device)
    # The same integration on the Mamba-2 route (SSM state + conv
    # lookbacks + CSA/HCA caches).
    all_results += _run_safe(test_hybrid_decoding_mamba2_route, device)
    # MQAR task construction shared by Exp 4 / Exp 5 (run_quality/run_ablation):
    # filler noise must never leak key/value/answer tokens.
    all_results += _run_safe(test_mqar_noise_excludes_keys_and_values, device)
    all_results += _run_safe(test_ranking_chance_reference_is_exact, device)

    # Partial RoPE (DeepSeek-V4 §2.3.3) — paper-faithful positional
    # embedding for CSA/HCA: unit properties, module wiring, and
    # incremental-decode equivalence with the full-sequence operators.
    all_results += _run_safe(test_partial_rope_properties, device)
    all_results += _run_safe(test_partial_rope_long_context_phase, device)
    all_results += _run_safe(test_csa_hca_rope_changes_output, device)
    all_results += _run_safe(test_csa_hca_rope_prefill_vs_cache, device)
    all_results += _run_safe(test_csa_hca_rope_module_matches_naive, device)
    # qk_norm_mode='rms' — the paper §2.3.3 RMSNorm normalization form.
    all_results += _run_safe(test_csa_hca_qk_norm_mode_rms, device)

    # Mamba-2 (SSD) reference mixer + the switchable-memory hybrid stack.
    # Covers the framework's real correctness risks: decode equivalence,
    # segmented prefill, causality, conv boundaries, state independence,
    # weight roundtrip, layout validation, and the KDA back-compat
    # contract. The official-backend parity check self-skips (with a
    # recorded note) when the optional mamba-ssm extra / CUDA is absent.
    all_results += _run_safe(test_mamba2_geometry_validation, device)
    all_results += _run_safe(test_mamba2_forward_smoke, device)
    all_results += _run_safe(test_mamba2_prefill_step_matches_full, device)
    all_results += _run_safe(test_mamba2_segmented_prefill, device)
    all_results += _run_safe(test_mamba2_causality, device)
    all_results += _run_safe(test_mamba2_conv_boundary_cases, device)
    all_results += _run_safe(test_mamba2_gradient, device)
    all_results += _run_safe(test_mamba2_state_save_load, device)
    all_results += _run_safe(test_mamba2_state_request_independence, device)
    all_results += _run_safe(test_hybrid_stack_memory_routes, device)
    all_results += _run_safe(test_mamba2_official_backend_parity, device)
    all_results += _run_safe(test_mamba2_backend_selection_contract, device)

    # RWKV-7 (Goose) reference mixer — the third memory route. Covers the
    # same correctness risks as the Mamba-2 block plus the RWKV-specific
    # ones: the scan's algebraic state-layout/decay-orientation contract,
    # the non-zero-initial-state gradient path (cross-chunk BPTT), the
    # depth-dependent init, and the live cache accounting (matrix +
    # shift states, actual dtypes). The FLA backend parity check
    # self-skips (with a recorded note) when fla.ops.rwkv7 / CUDA is
    # absent.
    all_results += _run_safe(test_rwkv7_scan_reference_properties, device)
    all_results += _run_safe(test_rwkv7_geometry_validation, device)
    all_results += _run_safe(test_rwkv7_forward_smoke, device)
    all_results += _run_safe(test_rwkv7_prefill_step_matches_full, device)
    all_results += _run_safe(test_rwkv7_segmented_prefill, device)
    all_results += _run_safe(test_rwkv7_causality, device)
    all_results += _run_safe(test_rwkv7_shift_boundary_cases, device)
    all_results += _run_safe(test_rwkv7_gradient, device)
    all_results += _run_safe(test_rwkv7_state_save_load, device)
    all_results += _run_safe(test_rwkv7_state_request_independence, device)
    all_results += _run_safe(test_rwkv7_nonzero_initial_state, device)
    all_results += _run_safe(test_hybrid_stack_rwkv7_route, device)
    all_results += _run_safe(test_hybrid_decoding_rwkv7_route, device)
    all_results += _run_safe(test_rwkv7_backend_selection_contract, device)
    all_results += _run_safe(test_rwkv7_fla_backend_parity, device)
    all_results += _run_safe(test_rwkv7_cache_live_accounting, device)
    # FP8 block-scaled state storage priced through the decode ledger.
    all_results += _run_safe(test_fp8_state_storage_ledger, device)
    # Exp 5 ratio sweep: accuracy-vs-decode-traffic efficiency/Pareto wiring.
    all_results += _run_safe(test_recall_efficiency_ratio_sweep, device)
    all_results += _run_safe(test_online_softmax_shared_kv_tiling, device)
    # Planning-layer contract: a pre-registered 2x2 interaction is a
    # four-cell contrast and cannot borrow the main-effect seed budget.
    all_results += _run_safe(test_interaction_planning_is_four_cell_budgeted,
                             device)

    passed = sum(r['status'] == 'PASS' for r in all_results)
    logger.info('-' * 70)
    logger.info(f'Total: {passed}/{len(all_results)} passed')

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a NaN/Inf slipped into a test's
    # ``detail`` field (e.g. a KDA recurrence overflow producing a non-finite
    # ``o_diff = (o_rec - o_chk).abs().max().item()``, or a NaN-propagating
    # sink test), Python's default json.dump would emit literal ``NaN``/
    # ``Infinity`` tokens that are INVALID JSON per RFC 8259 and break strict
    # parsers (JS ``JSON.parse``, jq, pandas with ``orient='records'``). The
    # writes go through ``kaggle_setup.write_json_atomic`` (temp file + fsync
    # + os.replace), so a process kill or disk-full mid-write leaves the
    # target file untouched rather than a truncated partial JSON document;
    # see ``write_json_atomic``'s docstring for the full rationale.
    try:
        write_json_atomic(all_results, 'results/exp1_correctness.json',
                          indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in results; sanitizing to null: {e}')
        write_json_atomic(sanitize_for_json(all_results),
                          'results/exp1_correctness.json',
                          indent=2, allow_nan=False)
    logger.info('Saved: results/exp1_correctness.json')
    # Bind the raw check list to the campaign ledger. The file itself is a
    # bare array for backward compatibility, so the manifest artifact digest
    # is its byte-level identity.
    try:
        provenance = capture_provenance()
        append_run_manifest(make_run_manifest_entry(
            experiment='exp1_correctness',
            seeds=[],
            provenance=provenance,
            hyperparameters={
                'device': str(device),
                'n_checks': len(all_results),
                'runner': 'run_correctness.main',
            },
            data={
                'kind': 'operator_regression_suite',
                'randomness': ('deterministic per-check seeds; no corpus '
                               'file'),
            },
            artifacts={'results': 'results/exp1_correctness.json'},
            results={
                'passed': int(passed),
                'total': int(len(all_results)),
            },
            status=('completed' if passed == len(all_results)
                    else 'partial'),
        ), logger=logger)
    except Exception as exc:
        logger.warning(
            f'could not append the correctness run to the manifest: '
            f'{type(exc).__name__}: {exc}')
    return 0 if passed == len(all_results) else 1


if __name__ == '__main__':
    sys.exit(main())
