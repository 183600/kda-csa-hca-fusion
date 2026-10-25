"""Prefill-decode consistency + decode-phase roofline for the pretraining LM.

Why this module?
----------------
``train_lm_autodl.LMWithHybrid`` is trained with a full-sequence (prefill)
forward through ``ops_fused.HybridKCHAttention``. Serving that CPU/GPU-trained
checkpoint requires token-by-token decoding through
``run_decoding.HybridDecoding``, which threads the KDA recurrent state plus one
CSA/HCA incremental cache per sub-layer. If those two paths disagree, the model
generates different tokens than it scored during pretraining — a silent
correctness bug that no training-loss curve can reveal. This module makes the
two paths comparable and quantifies the gap:

* :func:`build_decoding_clone` — a ``HybridDecoding`` tower sharing the LM
  tower's geometry AND weights, so prefill vs incremental execution is the
  only difference between the two logit tensors;
* :func:`prefill_decode_tower_outputs` / :func:`consistency_metrics` —
  max/mean logit drift, top-1 agreement, and per-position diagnostics;
* :func:`decode_roofline` — the decode phase's analytic work
  (FLOPs amortized from the Exp 3 closed form ``run_kv_cache.prefill_flops``,
  compulsory bytes from the ``kv_budget`` read model plus the KDA state write)
  combined with the measured per-token wall clock into Exp 7 roofline
  coordinates (``roofline.roofline_point``). Exp 7 only ever placed PREFILL
  rows (from Exp 2) on the roofline; this adds the decode phase, which is the
  phase where the O(1)-state operators are supposed to win.

Caveats (mirroring the repo's accounting-honesty policy)
--------------------------------------------------------
* The byte model is a compulsory-traffic lower bound (KV-side reads + KDA
  state write). Compressed-block/SW buffer WRITES are amortized and excluded;
  weights are steady-state and excluded; library-internal traffic is out of
  scope (same caveat as ``run_roofline``).
* The FLOP model is the *amortized* per-token share of the Exp 3 prefill
  closed form, ``prefill_flops(T) / T`` — smooth in ``T`` and consistent by
  construction with the committed Exp 3 accounting. It is not a
  decode-kernel microbenchmark.
* The measured time comes from the naive Python reference on the host (CPU
  CI or GPU), so achieved bandwidth/tflops sit far below the roofline for
  the same reason Exp 7 rows do. The value is the bound CLASSIFICATION and
  the prefill/decode asymmetry, not the absolute efficiency.
"""
from __future__ import annotations

import contextlib
import functools
import time

import torch

import kv_budget
import roofline
from ops_fused import HybridConfig
from run_kv_cache import prefill_flops

DEFAULT_TOLERANCE = 1e-4

# Attention-tower submodules of the pretraining LM (duck-typed so this
# module never has to import train_lm_autodl, which keeps the import graph
# acyclic and lets tests use any equivalent model object).
_TOWER_ATTRS = ('embed', 'hybrid', 'norm_f', 'lm_head')


def _sync(device: str) -> None:
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _preset_for_device(device: str) -> dict | None:
    """Spec-sheet fallback when on-host peak measurement is unavailable."""
    name = ''
    try:
        if str(device).startswith('cuda'):
            name = torch.cuda.get_device_name(device)
    except Exception:
        name = ''
    return roofline.spec_for_device(name)


def host_peaks(device: str = 'cpu', *, peak_tflops: float | None = None,
               peak_bw_gbps: float | None = None) -> dict:
    """Resolve the machine roofline anchors (TFLOP/s fp32, GB/s).

    Priority per anchor: explicit argument -> on-host measurement
    (:mod:`roofline`) -> spec-sheet preset. ``None`` when everything is
    unavailable; ``roofline.roofline_point`` then skips the ceiling fields.
    """
    sources: dict[str, str] = {}
    if peak_tflops is None:
        measured = roofline.measure_gemm_tflops(device)
        if measured is not None:
            peak_tflops, sources['peak_tflops'] = measured, 'measured_on_host'
        else:
            spec = _preset_for_device(device)
            if spec:
                peak_tflops = spec['peak_tflops']
                sources['peak_tflops'] = f"preset:{spec['preset_key']}"
    else:
        sources['peak_tflops'] = 'explicit'
    if peak_bw_gbps is None:
        measured = roofline.measure_copy_bandwidth_gbps(device)
        if measured is not None:
            peak_bw_gbps, sources['peak_bw_gbps'] = measured, 'measured_on_host'
        else:
            spec = _preset_for_device(device)
            if spec:
                peak_bw_gbps = spec['peak_bw_gbps']
                sources['peak_bw_gbps'] = f"preset:{spec['preset_key']}"
    else:
        sources['peak_bw_gbps'] = 'explicit'
    return {'peak_tflops': peak_tflops, 'peak_bw_gbps': peak_bw_gbps,
            'peak_sources': sources}


def lm_tower_geometry(cfg: HybridConfig) -> dict:
    """Map a :class:`HybridConfig` onto ``run_kv_cache.prefill_flops`` kwargs.

    Mirrors ``run_roofline._hybrid_geometry`` but sourced from the config
    instead of an Exp 2 ``workload`` record. ``kda_conv_ksize`` is the
    depthwise short-conv kernel size hardcoded in
    ``ops_fused.KDAHybridLayer`` (``nn.Conv1d(..., kernel_size=3)``).
    The hybrid layer counts and the memory-route geometry are
    EXPLICIT-LAYOUT aware (an explicit 10-layer matrix layout overrides
    the n_kda:n_csa:n_hca ratio fields), and the mamba2 / rwkv7 geometry
    keys let ``prefill_flops`` evaluate the route-aware hybrid row.
    """
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'lm_tower_geometry: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    if cfg.layout is not None:
        n_mem = sum(1 for k in cfg.layout if k in ('kda', 'mamba2', 'rwkv7'))
        n_csa = sum(1 for k in cfg.layout if k == 'csa')
        n_hca = sum(1 for k in cfg.layout if k == 'hca')
    else:
        n_mem, n_csa, n_hca = cfg.n_kda, cfg.n_csa, cfg.n_hca
    return dict(
        d=cfg.d_model, H=cfg.n_heads_qk, K=cfg.head_dim_k, V=cfg.head_dim_v,
        kda_hv=cfg.n_heads_v, kda_k=cfg.head_dim_k, kda_v=cfg.head_dim_v,
        kda_chunk_size=cfg.kda_chunk_size, kda_conv_ksize=3,
        csa_m=cfg.csa_m, csa_topk=cfg.csa_topk, csa_nh=cfg.csa_nh,
        csa_c=cfg.csa_c, csa_dc=cfg.csa_dc, csa_nIh=cfg.csa_nIh,
        csa_cI=cfg.csa_cI, csa_sliding_window=cfg.csa_sliding_window,
        hca_m2=cfg.hca_m2, hca_nh=cfg.hca_nh, hca_c=cfg.hca_c,
        hca_dc=cfg.hca_dc, hca_sliding_window=cfg.hca_sliding_window,
        hybrid_n_kda=n_mem, hybrid_n_csa=n_csa,
        hybrid_n_hca=n_hca,
        mamba_d_state=cfg.mamba_d_state, mamba_d_conv=cfg.mamba_d_conv,
        mamba_expand=cfg.mamba_expand, mamba_headdim=cfg.mamba_headdim,
        mamba_ngroups=cfg.mamba_ngroups, mamba_dt_rank=cfg.mamba_dt_rank,
        rwkv_head_dim=cfg.rwkv_head_dim, rwkv_decay_lora=cfg.rwkv_decay_lora,
        rwkv_a_lora=cfg.rwkv_a_lora, rwkv_gate_lora=cfg.rwkv_gate_lora,
    )


def _lm_memory_route(cfg: HybridConfig) -> str:
    """The concrete memory route of an LM tower (layout-aware)."""
    if cfg.layout is not None:
        return next(
            (k for k in cfg.layout if k in ('kda', 'mamba2', 'rwkv7')),
            cfg.memory_type)
    return cfg.memory_type


def _n_csa_layers(cfg: HybridConfig) -> int:
    """Number of CSA layers in the RESOLVED stack (layout-aware)."""
    if cfg.layout is not None:
        return sum(1 for k in cfg.layout if k == 'csa')
    return cfg.n_csa


def decode_read_spec(cfg: HybridConfig) -> dict:
    """Per-operator geometry blocks consumed by ``kv_budget.op_read_elements``."""
    return {
        'kda': {'hv': cfg.n_heads_v, 'd_k': cfg.head_dim_k,
                'd_v': cfg.head_dim_v},
        'csa': {'m': cfg.csa_m, 'topk': cfg.csa_topk, 'c': cfg.csa_c,
                'c_i': cfg.csa_cI, 'n_idx_heads': cfg.csa_nIh,
                'sliding_window': cfg.csa_sliding_window},
        'hca': {'m2': cfg.hca_m2, 'c': cfg.hca_c,
                'sliding_window': cfg.hca_sliding_window},
    }


def decode_elements_per_token(cfg: HybridConfig, T: int) -> dict:
    """Compulsory KV-side elements moved per decoded token at context ``T``.

    Read side: ``kv_budget.op_read_elements`` per layer (average over the
    ``T`` queries, paper-exact causal masks), summed over the layout.
    Write side: the KDA recurrent state write (one ``hv*d_k*d_v`` per KDA
    layer per step). Compressed-block and sliding-window WRITES are
    amortized over ``m``/``m2``/``win`` steps and excluded; they are
    second-order at the geometries these reports target.
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 1:
        raise ValueError(
            f'decode_elements_per_token: T must be a positive int, got {T!r}')
    spec = decode_read_spec(cfg)
    reads = 0.0
    writes = 0.0
    memory_reads = 0.0
    memory_writes = 0.0
    route = _lm_memory_route(cfg)
    if cfg.layout is not None:
        n_mem = sum(1 for k in cfg.layout if k in ('kda', 'mamba2', 'rwkv7'))
        n_csa = sum(1 for k in cfg.layout if k == 'csa')
        n_hca = sum(1 for k in cfg.layout if k == 'hca')
    else:
        n_mem, n_csa, n_hca = cfg.n_kda, cfg.n_csa, cfg.n_hca
    if route == 'kda' and n_mem > 0:
        kda = kv_budget.op_read_elements('kda', T, spec)
        memory_reads = n_mem * kda['total_read_elements']
        memory_writes = n_mem * (kda['state_elements'] or 0)
        reads += memory_reads
        writes += memory_writes
    elif route == 'mamba2' and n_mem > 0:
        # Mamba-2: the SSM state is fully read and rewritten per step
        # (nheads * headdim * d_state per layer); the conv state
        # (d_inner * (d_conv - 1)) is read too but is second-order and
        # amortized — excluded, mirroring the KDA write-side convention.
        d_inner = cfg.mamba_expand * cfg.d_model
        mamba_heads = (d_inner // cfg.mamba_headdim
                       if cfg.mamba_headdim > 0 else 1)
        ssm = mamba_heads * cfg.mamba_headdim * cfg.mamba_d_state
        memory_reads = n_mem * ssm
        memory_writes = n_mem * ssm
        reads += memory_reads
        writes += memory_writes
    elif route == 'rwkv7' and n_mem > 0:
        # RWKV-7: the matrix state is fully read and rewritten per step
        # (H * K * V per layer); the token-shift state (d) is read too
        # but is second-order — excluded, mirroring the KDA convention.
        rwkv_heads = (cfg.d_model // cfg.rwkv_head_dim
                      if cfg.rwkv_head_dim > 0 else 1)
        matrix = rwkv_heads * cfg.rwkv_head_dim * cfg.rwkv_head_dim
        memory_reads = n_mem * matrix
        memory_writes = n_mem * matrix
        reads += memory_reads
        writes += memory_writes
    if n_csa > 0:
        csa = kv_budget.op_read_elements('csa', T, spec)
        reads += n_csa * csa['total_read_elements']
    if n_hca > 0:
        hca = kv_budget.op_read_elements('hca', T, spec)
        reads += n_hca * hca['total_read_elements']
    return {'read_elements_per_token': reads,
            'write_elements_per_token': float(writes),
            'total_elements_per_token': reads + writes,
            'memory_read_elements_per_token': float(memory_reads),
            'memory_write_elements_per_token': float(memory_writes),
            'memory_route': route,
            'layout': [n_mem, n_csa, n_hca]}


def decode_roofline(cfg: HybridConfig, T: int, *, device: str = 'cpu',
                    itemsize: int = 4,
                    measured_decode_ms_per_token: float | None = None,
                    peak_tflops: float | None = None,
                    peak_bw_gbps: float | None = None) -> dict:
    """Decode-phase roofline point for one LM tower at context length ``T``.

    ``flops_per_token`` is the amortized share of the Exp 3 end-to-end
    prefill closed form (input + grouped-output projections + core), i.e.
    ``prefill_flops('hybrid_kch', T) / T``; ``bytes_per_token`` is the
    :func:`decode_elements_per_token` compulsory model. When a measured
    per-token decode latency is supplied the achieved-throughput fields and
    the roofline ceiling/efficiency fields are populated (otherwise only the
    analytic coordinates are returned).
    """
    if not isinstance(T, int) or isinstance(T, bool) or T < 1:
        raise ValueError(f'decode_roofline: T must be a positive int, got {T!r}')
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'decode_roofline: cfg must be a HybridConfig, got '
            f'{type(cfg).__name__}.')
    geo = lm_tower_geometry(cfg)
    hybrid_op = {'kda': 'hybrid_kch', 'mamba2': 'hybrid_mamba2',
                 'rwkv7': 'hybrid_rwkv7'}[_lm_memory_route(cfg)]
    flops_total = float(prefill_flops(hybrid_op, T, **geo))
    flops_per_token = flops_total / T
    elems = decode_elements_per_token(cfg, T)
    bytes_per_token = elems['total_elements_per_token'] * itemsize
    peaks = host_peaks(device, peak_tflops=peak_tflops,
                       peak_bw_gbps=peak_bw_gbps)
    peak_flops = (peaks['peak_tflops'] or 0.0) * 1e12 or None
    point = roofline.roofline_point(
        flops=flops_per_token, nbytes=bytes_per_token,
        time_ms=measured_decode_ms_per_token or 0.0,
        peak_flops_per_s=peak_flops, peak_bw_gbps=peaks['peak_bw_gbps'])
    point.update({
        'context_len': T,
        'layout': elems['layout'],
        'flops_total_prefill': flops_total,
        'flops_per_token_amortized': flops_per_token,
        'bytes_per_token': bytes_per_token,
        'elements_per_token': elems['total_elements_per_token'],
        'read_elements_per_token': elems['read_elements_per_token'],
        'write_elements_per_token': elems['write_elements_per_token'],
        'itemsize': itemsize,
        'measured_decode_ms_per_token': measured_decode_ms_per_token,
        'peak_tflops': peaks['peak_tflops'],
        'peak_bw_gbps': peaks['peak_bw_gbps'],
        'peak_sources': peaks['peak_sources'],
        'flops_model': ('amortized Exp 3 closed form: '
                        'run_kv_cache.prefill_flops(hybrid_kch, T) / T'),
        'byte_model': ('compulsory KV reads (kv_budget, per layer, averaged '
                       'over the T prefill queries) + KDA recurrent-state '
                       'write; compressed/SW writes amortized and excluded; '
                       'weights steady-state and excluded'),
    })
    return point


def _tower(model) -> tuple[HybridConfig, object]:
    """Validate the duck-typed LM interface and return (cfg, hybrid tower)."""
    missing = [a for a in _TOWER_ATTRS if not hasattr(model, a)]
    if missing:
        raise TypeError(
            f'lm_consistency: model is missing attribute(s) {missing}; '
            f'expected an LMWithHybrid-style object with '
            f'{list(_TOWER_ATTRS)}.')
    hybrid = model.hybrid
    cfg = getattr(model, 'cfg', None)
    if cfg is None:
        cfg = getattr(hybrid, 'cfg', None)
    if not isinstance(cfg, HybridConfig):
        raise TypeError(
            f'lm_consistency: could not find a HybridConfig on the model '
            f'(model.cfg={getattr(model, "cfg", None)!r}, '
            f'hybrid.cfg={getattr(hybrid, "cfg", None)!r}).')
    return cfg, hybrid


@contextlib.contextmanager
def _capture_csa_indexer_selections():
    """Record every ``csa_lightning_indexer`` call's indices during the block.

    Temporarily wraps the module-level symbol in BOTH ``ops_csa`` (batched
    prefill path) and ``ops_decoding_cache`` (per-token decode path).
    Yields ``{'full': [...], 'inc': [...]}`` lists of detached index tensors.
    Diagnostics-only: the patch is process-global, so do not run another
    forward concurrently with the block.
    """
    import ops_csa
    import ops_decoding_cache

    full: list[torch.Tensor] = []
    inc: list[torch.Tensor] = []
    orig_full = ops_csa.csa_lightning_indexer
    orig_inc = ops_decoding_cache.csa_lightning_indexer

    def _wrap(sink, orig):
        @functools.wraps(orig)
        def wrapped(q_idx, k_idx, w_idx, topk, *args, **kwargs):
            out = orig(q_idx, k_idx, w_idx, topk, *args, **kwargs)
            idx = out[0] if isinstance(out, tuple) else out
            sink.append(idx.detach().clone())
            return out
        return wrapped

    ops_csa.csa_lightning_indexer = _wrap(full, orig_full)
    ops_decoding_cache.csa_lightning_indexer = _wrap(inc, orig_inc)
    try:
        yield {'full': full, 'inc': inc}
    finally:
        ops_csa.csa_lightning_indexer = orig_full
        ops_decoding_cache.csa_lightning_indexer = orig_inc


def _selected_sets(idx_row: torch.Tensor) -> set:
    return {int(i) for i in idx_row.reshape(-1) if int(i) >= 0}


def _summarize_selection(captured: dict, cfg: HybridConfig, *,
                         prefill_len: int, seq_len: int,
                         max_mismatches: int = 10) -> dict:
    """Per-position top-k selection agreement between the two paths.

    Call order contract (relied upon here):
      * ``full`` records are emitted layer-by-layer by the vectorized
        prefill forwards, so the first ``n_csa`` records belong to the LM
        full-sequence forward and the next ``n_csa`` to the clone's prefill
        chunk (which reuses ``naive_csa`` for its output);
      * ``inc`` records are emitted one indexer call per CSA layer per
        decoded token, in step-major order.
    """
    n = _n_csa_layers(cfg)
    full_records = captured['full']
    inc_records = captured['inc']
    base = {
        'n_csa_layers': n,
        'n_prefill_positions': prefill_len,
        'n_decode_positions': seq_len - prefill_len,
        'agreement': None,
        'decode_agreement': None,
        'mismatches': [],
        'note': '',
    }
    if n < 1:
        base['note'] = 'no CSA layers'
        return base
    if prefill_len < 2:
        base['note'] = ('selection diagnostic needs prefill_len >= 2 '
                        '(the clone prefill branch is vectorized only for '
                        'chunks longer than one token)')
        return base
    if len(full_records) < 2 * n or len(inc_records) < n:
        base['note'] = (f'unexpected capture counts: full={len(full_records)} '
                        f'(need {2 * n}), inc={len(inc_records)}')
        return base
    lm_full = full_records[:n]
    clone_pre = full_records[n:2 * n]
    inc_by_layer = [[] for _ in range(n)]
    for k, idx in enumerate(inc_records):
        inc_by_layer[k % n].append(idx)

    mismatches: list[dict] = []
    counts = {'prefill': [0, 0], 'decode': [0, 0]}      # [total, match]

    def _compare(layer: int, position: int, a: torch.Tensor, b: torch.Tensor,
                 region: str) -> None:
        sa, sb = _selected_sets(a), _selected_sets(b)
        counts[region][0] += 1
        if sa == sb:
            counts[region][1] += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append({'layer': layer, 'position': position,
                               'region': region, 'full': sorted(sa),
                               'inc': sorted(sb)})

    for layer in range(n):
        full_idx = lm_full[layer][0]                    # [T, topk]
        pre_idx = clone_pre[layer][0]                   # [prefill_len, topk]
        for t in range(min(prefill_len, full_idx.shape[0])):
            _compare(layer, t, full_idx[t], pre_idx[t], 'prefill')
        for step, idx in enumerate(inc_by_layer[layer]):
            t = prefill_len + step
            if t >= full_idx.shape[0]:
                break
            _compare(layer, t, full_idx[t], idx[0, 0], 'decode')
    n_total = counts['prefill'][0] + counts['decode'][0]
    n_match = counts['prefill'][1] + counts['decode'][1]
    base.update({
        'agreement': n_match / n_total if n_total else None,
        'decode_agreement': (
            counts['decode'][1] / counts['decode'][0]
            if counts['decode'][0] > 0 else None),
        'n_compared': n_total,
        'mismatches': mismatches,
        'note': ('set equality of selected block indices per (layer, query '
                 'position); the indexer breaks score ties on ascending '
                 'block index (canonical, size-independent), so a mismatch '
                 'indicates genuine state/math divergence rather than '
                 'torch.topk tie-breaking'),
    })
    return base


def build_decoding_clone(model, device=None):
    """A ``HybridDecoding`` tower sharing ``model``'s geometry and weights.

    Runs on CPU until ``device`` is given (the weights are copied from
    whatever device ``model`` lives on, so a CUDA-trained LM can be probed
    on CPU without duplicating its GPU footprint). The caller owns
    ``clone.reset()`` / cache lifecycle.
    """
    from run_decoding import HybridDecoding

    _cfg, hybrid = _tower(model)
    # Pass the tower's memory route explicitly: HybridDecoding validates
    # that a caller-supplied config's memory_type matches (a KDA-defaulted
    # clone of an mamba2 / rwkv7 tower would raise, and silently running
    # the wrong memory operator would be worse).
    clone = HybridDecoding(cfg=hybrid.cfg,
                           total_layers=hybrid.total_layers,
                           memory_type=hybrid.cfg.memory_type)
    clone.model.load_state_dict(hybrid.state_dict())
    clone.eval()
    if device is not None:
        clone = clone.to(device)
    return clone


def prefill_decode_tower_outputs(model, input_ids, prefill_len: int | None = None,
                                 device=None, *,
                                 capture_selection: bool = False) -> dict:
    """Run the full-sequence and incremental paths through the SAME weights.

    Returns the attention-tower outputs (pre-norm, pre-head) for both paths
    plus their wall-clock timings, so the caller can apply ``norm_f`` /
    ``lm_head`` exactly once and compare like with like:

    * ``tower_full``: ``model.hybrid(embed(input_ids))`` — the training-time
      prefill forward (KDA chunk path if ``T >= kda_chunk_size``);
    * ``tower_inc``: prefill of the first ``prefill_len`` tokens through the
      cache-enabled clone, then one token per call (KDA recurrent path +
      CSA/HCA incremental caches).

    With ``capture_selection=True`` the CSA lightning-indexer index tensors
    of BOTH paths are recorded (via a temporary module-level patch of
    ``csa_lightning_indexer``) and returned under ``selection_records`` so
    the caller can separate selection divergence from genuine state/math
    divergence. Exact score ties cannot split the paths (the indexer
    breaks ties on ascending block index), so a mismatch here indicates
    near-tied logits with differing fp ordering or a real selection bug.
    """
    cfg, hybrid = _tower(model)
    T = int(input_ids.shape[1])
    if T < 1:
        raise ValueError(
            f'prefill_decode_tower_outputs: need T >= 1, got {T}.')
    if prefill_len is None:
        prefill_len = max(1, T // 2)
    if not 1 <= prefill_len <= T:
        raise ValueError(
            f'prefill_decode_tower_outputs: prefill_len={prefill_len} must '
            f'be in [1, T={T}].')

    dev = input_ids.device if device is None else device
    was_training = model.training
    model.eval()
    clone = build_decoding_clone(model, device=dev)
    capture_cm = (_capture_csa_indexer_selections() if capture_selection
                  else contextlib.nullcontext(None))
    try:
        with capture_cm as captured, torch.no_grad():
            model.hybrid.reset_state()
            x = model.embed(input_ids)
            _sync(str(dev))
            start = _now()
            tower_full = model.hybrid(x)
            _sync(str(dev))
            prefill_full_ms = _elapsed_ms(start)

            clone.reset()
            _sync(str(dev))
            start = _now()
            tower_pre = clone(x[:, :prefill_len])
            _sync(str(dev))
            prefill_inc_ms = _elapsed_ms(start)

            pieces = [tower_pre]
            _sync(str(dev))
            start = _now()
            for t in range(prefill_len, T):
                pieces.append(clone(x[:, t:t + 1]))
            _sync(str(dev))
            decode_ms = _elapsed_ms(start)
            tower_inc = torch.cat(pieces, dim=1)
    finally:
        model.train(was_training)

    n_decode = T - prefill_len
    out = {
        'cfg': cfg,
        'prefill_len': prefill_len,
        'n_decode': n_decode,
        'seq_len': T,
        'tower_full': tower_full,
        'tower_inc': tower_inc,
        'prefill_full_ms': prefill_full_ms,
        'prefill_inc_ms': prefill_inc_ms,
        'decode_ms_total': decode_ms,
        'decode_ms_per_token': (decode_ms / n_decode) if n_decode else None,
    }
    if capture_selection and captured is not None:
        out['selection'] = _summarize_selection(
            captured, cfg, prefill_len=prefill_len, seq_len=T)
    return out


def _now() -> float:
    return time.perf_counter()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1e3


def consistency_metrics(logits_full: torch.Tensor, logits_inc: torch.Tensor,
                        *, start: int = 0, tolerance: float = DEFAULT_TOLERANCE
                        ) -> dict:
    """Logit-space drift between the full-sequence and incremental paths.

    ``start`` skips leading positions (e.g. pass ``prefill_len`` to score
    only the decoded token positions). Top-1 agreement is computed per
    position and averaged over batch. ``consistent`` is the tolerance
    verdict on the max abs drift.
    """
    if logits_full.shape != logits_inc.shape:
        raise ValueError(
            f'consistency_metrics: shape mismatch '
            f'{tuple(logits_full.shape)} vs {tuple(logits_inc.shape)}.')
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError(
            f'consistency_metrics: start must be an int, got {start!r}.')
    if start < 0:
        raise ValueError(
            f'consistency_metrics: start={start} must be >= 0.')
    a = logits_full[:, start:]
    b = logits_inc[:, start:]
    if a.numel() == 0:
        raise ValueError(
            f'consistency_metrics: start={start} leaves no positions to '
            f'compare (T={logits_full.shape[1]}).')
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    top1 = (a.argmax(dim=-1) == b.argmax(dim=-1)).to(torch.float32).mean()
    return {
        'n_positions': int(a.shape[1]),
        'max_abs_logit_diff': max_abs,
        'mean_abs_logit_diff': float(diff.mean().item()),
        'top1_agreement': float(top1.item()),
        'tolerance': float(tolerance),
        'consistent': max_abs < tolerance,
    }


def run_lm_consistency_check(model, input_ids, *, prefill_len: int | None = None,
                             device=None,
                             tolerance: float = DEFAULT_TOLERANCE,
                             with_roofline: bool = True,
                             peak_tflops: float | None = None,
                             peak_bw_gbps: float | None = None) -> dict:
    """One-shot prefill-decode consistency + decode-roofline report.

    The model object must expose ``embed`` / ``hybrid`` / ``norm_f`` /
    ``lm_head`` (``train_lm_autodl.LMWithHybrid`` satisfies this). The
    report is a plain dict of Python scalars (JSON-safe) covering:
    all-position and decode-only logit drift, top-1 agreement, measured
    prefill/decode timings, and — when ``with_roofline`` — the
    :func:`decode_roofline` block built from the same context length.

    When the config has CSA layers the report also carries a ``selection``
    block (per-position top-k selection agreement) and a ``diagnosis``
    verdict:

    * ``'consistent'``: drift is within tolerance;
    * ``'topk_selection_divergence'``: the causal logit drift coincides
      with prefill/decode indexer selections differing. Exact ties no
      longer split the two paths (the indexer breaks ties on ascending
      block index via ``ops_csa._canonical_topk_indices``), so this now
      points at NEAR-tied logits whose fp ordering differs between the
      batched and per-token tables — still typically an artifact of a
      sparse, untrained/loosely-trained indexer, NOT a cache state bug,
      but worth inspecting the recorded mismatches;
    * ``'state_or_math_divergence'``: drift above tolerance while the
      selections still agree — investigate the incremental caches.
    """
    cfg, _hybrid = _tower(model)
    out = prefill_decode_tower_outputs(
        model, input_ids, prefill_len=prefill_len, device=device,
        capture_selection=_n_csa_layers(cfg) > 0)
    with torch.no_grad():
        logits_full = model.lm_head(model.norm_f(out['tower_full']))
        logits_inc = model.lm_head(model.norm_f(out['tower_inc']))

    all_pos = consistency_metrics(logits_full, logits_inc,
                                  start=0, tolerance=tolerance)
    if out['n_decode'] > 0:
        decode_pos = consistency_metrics(logits_full, logits_inc,
                                         start=out['prefill_len'],
                                         tolerance=tolerance)
    else:
        # ``prefill_len == T`` is a valid input to
        # ``prefill_decode_tower_outputs`` (it reports
        # ``decode_ms_per_token=None``); there are no decode-only positions
        # to compare, so report the metric as undefined rather than calling
        # ``consistency_metrics`` with an empty slice.
        decode_pos = {'max_abs_logit_diff': None, 'top1_agreement': None}
    dev = str(input_ids.device if device is None else device)
    report = {
        'seq_len': out['seq_len'],
        'prefill_len': out['prefill_len'],
        'n_decode': out['n_decode'],
        'device': dev,
        'vocab_size': int(logits_full.shape[-1]),
        'max_abs_logit_diff': all_pos['max_abs_logit_diff'],
        'mean_abs_logit_diff': all_pos['mean_abs_logit_diff'],
        'top1_agreement': all_pos['top1_agreement'],
        'decode_max_abs_logit_diff': decode_pos['max_abs_logit_diff'],
        'decode_top1_agreement': decode_pos['top1_agreement'],
        'tolerance': float(tolerance),
        'consistent': all_pos['consistent'],
    }
    report['prefill_full_ms'] = out['prefill_full_ms']
    report['prefill_inc_ms'] = out['prefill_inc_ms']
    report['decode_ms_per_token'] = out['decode_ms_per_token']
    if 'selection' in out:
        selection = out['selection']
        report['selection'] = selection
        if report['consistent']:
            report['diagnosis'] = 'consistent'
        elif (selection.get('agreement') is not None
              and selection['agreement'] < 1.0):
            report['diagnosis'] = 'topk_selection_divergence'
        else:
            report['diagnosis'] = 'state_or_math_divergence'
    else:
        report['diagnosis'] = ('consistent' if report['consistent']
                               else 'state_or_math_divergence')
    if with_roofline:
        report['roofline'] = decode_roofline(
            cfg, out['seq_len'], device=dev,
            measured_decode_ms_per_token=out['decode_ms_per_token'],
            peak_tflops=peak_tflops, peak_bw_gbps=peak_bw_gbps)
    return report
