"""Experiment 9 — KV read-budget frontier.

Motivation
----------
Experiments 4/5/8 measure MQAR quality of the operators and hybrid layouts;
Experiments 3/7 account for retained state and roofline distance. None of
them answers the *budget* question that drives the top-k knob and the hybrid
ratio: **how many KV-side elements does one query read, and which term
dominates as context grows?**

This experiment unifies three questions under one currency (KV-side
elements read per query per layer; see ``kv_budget`` for the exact
contract). The currency has TWO views that ``kv_budget`` keeps unified:

* the **prefill average** over positions ``0..T-1`` (``op_read_elements`` /
  ``layout_read_elements``), and
* the **marginal / decode-step** view at a single position ``t``
  (``instant_read_elements`` / ``instant_layout_read_elements``), whose
  rolling mean is exactly the prefill average and whose value at
  ``t = T`` is the budget of the first token decoded after a ``T``-token
  prefill. ``windowed_*`` gives the O(1) mean over any decode window.

Questions:

1. **Analytic budget curves** over context length for the four core
   operators and the seven Exp 5 hybrid layouts, in two geometries (the
   small MQAR probe and the paper-scale ``run_kv_cache.DEFAULTS``). The
   curves expose the two terms of CSA: the O(topk) selected gather AND the
   O(T) lightning-indexer scan, which the reference implementation performs
   over every visible compressed block before selecting the top-k. They also
   expose the delta-rule extreme: KDA's state read is O(1) in T. A second
   `decode_budget_curve` section reports the MARGINAL (per-position) view
   for the same operators and layouts: the true per-token budget during
   decoding, not the prefill average (the decode average understates the
   cost of late positions).
2. **Measured top-k decode sweep**: per-token decode latency of the fused
   3:1:1 hybrid as ``csa_topk`` varies (0 = sparse branch off, then 1/2/4/8),
   joined with BOTH analytic views at the same geometry: the prefill
   average (backward-compatible ``total_read_elements_per_query``) and the
   decode-window budget over the actually decoded positions. This tests
   whether shrinking the top-k budget actually buys latency in the
   reference implementation.
3. **Quality-budget join**: if Exp 4/5/8 result files exist, each measured
   configuration is annotated with its analytic budget at the MQAR context
   (T=16), and the Exp 5 layout ranking is compared against the budget
   ranking at T=16 and at a long-context extrapolation (T=8192, paper
   geometry) in both views. Exp 5 controls depth but not budget; the join
   makes that distinction explicit and exploratory (Spearman over 7
   layouts).

Honesty / scope
---------------
* The budget model is an analytical read count, not profiled traffic:
  library-internal materialization (unfold, temporaries) is out of scope.
* The indexer scan is charged on every query because the reference
  implementation scores all visible blocks before top-k (including
  ``topk=0``).
* The decode-sweep geometry is intentionally smaller than the Exp 4/5/8
  quality geometry (see ``run_decoding.HybridDecoding``); budgets are
  reported per geometry and never silently mixed.
* The long-context ranking is an extrapolation of the same layout budgets;
  the measured accuracies are T=16 only. Spearman over 7 layouts is
  exploratory, not confirmatory.

Environment variables
---------------------
* ``BUDGET_T_GRID``         (default '16,64,256,1024,4096,16384').
* ``BUDGET_TOPK``           (default '0,1,2,4,8')  decode sweep values.
* ``BUDGET_PREFILL_LENS``   (default '256,512').
* ``BUDGET_DECODE_TOKENS``  (default 10).
* ``BUDGET_REPEATS``        (default 2).

Output
------
``results/exp9_budget_frontier.json`` — v1.0 envelope
(``kind='exp9_budget_frontier'``) with a flat row list; every row carries
``section`` in {'budget_curve', 'decode_budget_curve', 'decode_sweep',
'quality_join', 'ranking_summary', 'source_audit'}. The ``source_audit``
rows record each join source's file hash, git commit/dirty flag, seeds
and resolved backends, plus a cross-source comparability verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (
    capture_provenance,
    configure_torch_for_device,
    parse_int_env,
    set_global_seed,
)
from kv_budget import (
    instant_layout_read_elements,
    instant_read_elements,
    layout_read_elements,
    op_read_elements,
    spearman,
    windowed_layout_read_elements,
)
from results_schema import detect_version as detect_schema_version
from results_schema import load as load_schema
from results_schema import parse as parse_schema
from results_schema import write as write_schema
from run_ablation import _make_cfg
from run_decoding import HybridDecoding, bench_decoding
from run_kv_cache import DEFAULTS as PAPER_DEFAULTS
from run_kv_cache import GQA_H, GQA_HEAD_DIM
from run_quality import SoftmaxAttn

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp9_budget_frontier.json')
MQAR_SEQ_LEN = 16
DECODE_D_MODEL = 64
DECODE_TOTAL_LAYERS = 5
DEFAULT_T_GRID = '16,64,256,1024,4096,16384'
DEFAULT_TOPK = '0,1,2,4,8'
DEFAULT_PREFILL_LENS = '256,512'
DEFAULT_DECODE_TOKENS = 10
DEFAULT_REPEATS = 2
# Seed for the measured decode sweep's model construction / benchmark inputs
# (recorded in the result metadata so the latency rows are reproducible).
DECODE_SWEEP_SEED = 0
LAYOUTS = (
    (3, 1, 1),
    (4, 1, 0), (4, 0, 1),
    (2, 2, 1), (2, 1, 2),
    (1, 3, 1), (1, 1, 3),
)
JOIN_SOURCES = (
    ('exp4_mqar', os.path.join('results', 'exp4_mqar.json')),
    ('exp5_ablation', os.path.join('results', 'exp5_ablation.json')),
    ('exp8_mechanism_ablation',
     os.path.join('results', 'exp8_mechanism_ablation.json')),
)
# Exp 5 writes its result file as a bare array (legacy schema 0.0), so its
# provenance/seeds live in a sidecar written by the same run.
SOURCE_PROVENANCE_PATHS = {
    'exp5_ablation': os.path.join(
        'results', 'exp5_ablation_provenance.json'),
}


def _parse_int_list(raw, name, min_value=0):
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            raise ValueError(
                f'{name} contains a non-integer entry: {part!r}')
    if not values:
        raise ValueError(f'{name} has no integer entries: {raw!r}')
    if any(v < min_value for v in values):
        raise ValueError(
            f'{name} entries must be >= {min_value}, got {values}')
    return values


def _spec_from_hybrid_cfg(cfg, *, softmax_geo):
    """Build a kv_budget geometry dict from a ``HybridConfig``."""
    return {
        'softmax': dict(softmax_geo),
        'kda': {'hv': cfg.n_heads_v, 'd_k': cfg.head_dim_k,
                'd_v': cfg.head_dim_v},
        'csa': {'m': cfg.csa_m, 'topk': cfg.csa_topk, 'c': cfg.csa_c,
                'c_i': cfg.csa_cI, 'n_idx_heads': cfg.csa_nIh,
                'sliding_window': cfg.csa_sliding_window},
        'hca': {'m2': cfg.hca_m2, 'c': cfg.hca_c,
                'sliding_window': cfg.hca_sliding_window},
    }


def mqar_spec():
    """MQAR probe geometry (Exp 4/5/8): d_model=32 3:1:1 hybrid + softmax."""
    cfg = _make_cfg(32, (3, 1, 1))
    softmax = SoftmaxAttn(8)
    return _spec_from_hybrid_cfg(
        cfg, softmax_geo={'n_kv_heads': softmax.H, 'd_k': softmax.K,
                          'd_v': softmax.V})


def paper_spec():
    """Paper-scale geometry (Exp 3 defaults): H=8, K=V=128, c=128."""
    p = PAPER_DEFAULTS
    return {
        'softmax': {'n_kv_heads': GQA_H, 'd_k': GQA_HEAD_DIM,
                    'd_v': GQA_HEAD_DIM},
        'kda': {'hv': p['kda_hv'], 'd_k': p['kda_k'], 'd_v': p['kda_v']},
        'csa': {'m': p['csa_m'], 'topk': p['csa_topk'], 'c': p['csa_c'],
                'c_i': p['csa_cI'], 'n_idx_heads': p['csa_nIh'],
                'sliding_window': p['csa_sliding_window']},
        'hca': {'m2': p['hca_m2'], 'c': p['hca_c'],
                'sliding_window': p['hca_sliding_window']},
    }


def build_budget_curves(t_grid, specs):
    """Analytic budget rows for every (geometry, T, op/layout) cell."""
    rows = []
    for geom, spec in specs.items():
        for T in t_grid:
            for op in ('softmax', 'kda', 'csa', 'hca'):
                r = op_read_elements(op, T, spec)
                rows.append({
                    'section': 'budget_curve',
                    'row_kind': 'op',
                    'geometry': geom,
                    'name': op,
                    **r,
                })
            for ratio in LAYOUTS:
                r = layout_read_elements(T, ratio, spec)
                rows.append({
                    'section': 'budget_curve',
                    'row_kind': 'layout',
                    'geometry': geom,
                    'name': f'{ratio[0]}:{ratio[1]}:{ratio[2]}',
                    'ratio': list(ratio),
                    **r,
                })
    return rows


def build_decode_budget_curves(t_grid, specs):
    """Marginal budget rows — the decode-side view of the frontier.

    Row ``t`` is the budget charged to the query at position ``t`` (i.e.
    the first token decoded after a ``t``-token prefill), NOT the prefill
    average over ``0..t-1``. The two views agree under averaging by
    construction (``kv_budget`` pins the identity); at long context the
    marginal view shows the true per-token decode cost, which the prefill
    average understates.
    """
    rows = []
    for geom, spec in specs.items():
        for t in t_grid:
            for op in ('softmax', 'kda', 'csa', 'hca'):
                r = instant_read_elements(op, t, spec)
                rows.append({
                    'section': 'decode_budget_curve',
                    'row_kind': 'op',
                    'geometry': geom,
                    'name': op,
                    **r,
                })
            for ratio in LAYOUTS:
                r = instant_layout_read_elements(ratio, t, spec)
                rows.append({
                    'section': 'decode_budget_curve',
                    'row_kind': 'layout',
                    'geometry': geom,
                    'name': f'{ratio[0]}:{ratio[1]}:{ratio[2]}',
                    'ratio': list(ratio),
                    **r,
                })
    return rows


def _decode_row(cfg, bench, topk, prefill_len, n_decode, repeats):
    softmax_geo = {'n_kv_heads': 2, 'd_k': 16, 'd_v': 16}
    spec = _spec_from_hybrid_cfg(cfg, softmax_geo=softmax_geo)
    ratio = (cfg.n_kda, cfg.n_csa, cfg.n_hca)
    budget = layout_read_elements(prefill_len, ratio, spec)
    csa_op = op_read_elements('csa', prefill_len, spec)
    # Decode-side (marginal) budgets: the first decoded token sits at
    # position prefill_len, and the n_decode measured tokens span the
    # window [prefill_len, prefill_len + n_decode).
    step = instant_layout_read_elements(ratio, prefill_len, spec)
    window = windowed_layout_read_elements(
        ratio, prefill_len, prefill_len + n_decode, spec)
    return {
        'section': 'decode_sweep',
        'csa_topk': topk,
        'prefill_len': prefill_len,
        'n_decode': n_decode,
        'repeats': repeats,
        'layout': f'{cfg.n_kda}:{cfg.n_csa}:{cfg.n_hca}',
        'prefill_ms': bench['prefill_ms'],
        'mean_decode_ms_per_token': bench['mean_decode_ms_per_token'],
        'median_decode_ms_per_token': bench['median_decode_ms_per_token'],
        'peak_mem_MB': bench['peak_mem_MB'],
        'budget_context_T': prefill_len,
        'total_read_elements_per_query': budget['total_read_elements'],
        'csa_indexer_read_elements_per_query':
            csa_op['indexer_read_elements'],
        'csa_core_read_elements_per_query':
            csa_op['core_read_elements'],
        'decode_step_read_elements_per_query':
            step['total_read_elements'],
        'decode_window_read_elements_per_query':
            window['total_read_elements'],
        'decode_step_kda_read_elements': step['kda_read_elements'],
        'decode_step_csa_read_elements': step['csa_read_elements'],
        'decode_step_hca_read_elements': step['hca_read_elements'],
    }


def run_decode_sweep(device, topk_values, prefill_lens, n_decode, repeats):
    """Measure decode latency of the fused hybrid vs the top-k budget."""
    rows = []
    for prefill_len in prefill_lens:
        for topk in topk_values:
            try:
                set_global_seed(DECODE_SWEEP_SEED, deterministic=True)
                model = HybridDecoding(
                    d_model=DECODE_D_MODEL, total_layers=DECODE_TOTAL_LAYERS,
                    csa_topk=topk)
                bench = bench_decoding(
                    model, DECODE_D_MODEL, prefill_len, n_decode, device,
                    repeats=repeats)
                rows.append(_decode_row(
                    model.cfg, bench, topk, prefill_len, n_decode, repeats))
                logger.info(
                    f'  decode sweep topk={topk} L={prefill_len}: '
                    f'prefill={bench["prefill_ms"]:.2f}ms '
                    f'decode/token={bench["mean_decode_ms_per_token"]:.3f}ms')
            except Exception as e:
                logger.warning(
                    f'  decode sweep topk={topk} L={prefill_len} FAILED: {e}')
                rows.append({
                    'section': 'decode_sweep',
                    'csa_topk': topk,
                    'prefill_len': prefill_len,
                    'n_decode': n_decode,
                    'repeats': repeats,
                    'error': str(e),
                })
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return rows


def _parse_ratio(label):
    """Parse a ``'n_kda:n_csa:n_hca'`` layout label.

    Returns ``None`` for anything that is not a valid layout, INCLUDING a
    negative component: the layout accounting rejects those, and a
    malformed label must be a documented skip (the row is dropped) rather
    than an uncaught ``ValueError`` that aborts the whole join.
    """
    try:
        parts = [int(x) for x in str(label).replace(' ', '').split(':')]
    except (TypeError, ValueError):
        return None
    if len(parts) != 3:
        return None
    if any(n < 0 for n in parts):
        return None
    if sum(parts) == 0:
        return None
    return tuple(parts)


def _file_sha256(path):
    """SHA-256 of a result file, so a join records the exact bytes used."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_join_source(source, path, provenance_path=None):
    """Load one Exp 4/5/8 result file plus its provenance for the join.

    Returns a record with the rows and the comparability fields. Missing
    or malformed sources are recorded (``readable=False`` + ``error``)
    instead of aborting the experiment: Exp 9 must still run when one
    upstream experiment has not been produced. Exp 5's bare-array result
    has no metadata, so its sidecar provenance file is the fallback.
    """
    if provenance_path is None:
        provenance_path = SOURCE_PROVENANCE_PATHS.get(source)
    rec = {
        'source': source,
        'path': path,
        'exists': os.path.exists(path),
        'readable': False,
        'schema_version': None,
        'n_rows': 0,
        'sha256': None,
        'git_commit': None,
        'git_dirty': None,
        'captured_at_utc': None,
        'seeds': [],
        'n_seeds': 0,
        'backends': {},
        'provenance_source': None,
        'error': None,
        'rows': [],
    }
    if not rec['exists']:
        rec['error'] = f'{path} not found; join skipped (run the experiment)'
        return rec
    try:
        rec['sha256'] = _file_sha256(path)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        rec['schema_version'] = detect_schema_version(data)
        meta, rows, _version = parse_schema(data)
    except Exception as e:
        rec['error'] = f'{path} unreadable: {type(e).__name__}: {e}'
        return rec
    if not isinstance(rows, list):
        rec['error'] = f'{path} did not yield a row list'
        return rec
    rec['rows'] = rows
    rec['n_rows'] = len(rows)
    rec['readable'] = True

    prov = meta.get('provenance')
    if isinstance(prov, dict) and prov:
        rec['provenance_source'] = 'metadata.provenance'
    else:
        prov = {}
    seeds = meta.get('seeds')
    sidecar_meta = {}
    if provenance_path and os.path.exists(provenance_path):
        try:
            with open(provenance_path, encoding='utf-8') as f:
                sidecar_meta = json.load(f)
        except Exception as e:
            rec['error'] = (f'{provenance_path} unreadable: '
                            f'{type(e).__name__}: {e}')
            sidecar_meta = {}
    if isinstance(sidecar_meta, dict):
        sidecar_prov = sidecar_meta.get('provenance')
        if not isinstance(sidecar_prov, dict) or not sidecar_prov:
            sidecar_prov = sidecar_meta
        if not prov and isinstance(sidecar_prov, dict):
            prov = sidecar_prov
            if rec['provenance_source'] is None:
                rec['provenance_source'] = 'sidecar'
        if not isinstance(seeds, list) and isinstance(
                sidecar_meta.get('seeds'), list):
            seeds = sidecar_meta['seeds']

    if isinstance(seeds, list):
        rec['seeds'] = sorted({int(s) for s in seeds
                               if isinstance(s, int)
                               and not isinstance(s, bool)})
    rec['n_seeds'] = len(rec['seeds'])
    rec['git_commit'] = prov.get('git_commit')
    rec['git_dirty'] = prov.get('git_dirty')
    rec['captured_at_utc'] = prov.get('captured_at_utc')
    rec['backends'] = {
        key: prov.get(key) or prov.get(key.replace('_resolved', ''))
        for key in ('kda_backend_resolved', 'mamba_backend_resolved',
                    'rwkv_backend_resolved')
        if prov.get(key) or prov.get(key.replace('_resolved', ''))
    }
    return rec


def _source_audit_row(rec):
    """The on-disk summary of one join source (rows omitted)."""
    row = {k: v for k, v in rec.items() if k != 'rows'}
    row['section'] = 'source_audit'
    return row


def audit_join_sources(source_records):
    """Check whether Exp 4/5/8 rows may be compared in one ranking.

    The join is only interpretable when the sources ran the same code
    revision, overlap on seeds, and agree on the resolved memory
    backends. This function never changes the measured numbers; it
    records the evidence and returns a status used to annotate (and, for
    mixed revisions, caveat) the ranking summary.
    """
    records = list(source_records)
    present = [r for r in records if r.get('readable')]
    missing = [r['source'] for r in records if not r.get('readable')]
    missing_prov = [r['source'] for r in present
                    if not r.get('git_commit')]
    commits = sorted({r['git_commit'] for r in present
                      if r.get('git_commit')})
    dirty = [r['source'] for r in present if r.get('git_dirty') is True]
    seed_sets = [set(r['seeds']) for r in present if r.get('seeds')]
    if seed_sets:
        shared = set.intersection(*seed_sets)
        shared_seeds = sorted(shared)
        seed_status = ('shared' if shared_seeds else 'disjoint')
        if len(seed_sets) != len(present):
            seed_status = 'partial'
    else:
        shared_seeds = []
        seed_status = 'missing'

    backend_keys = sorted({key for r in present
                           for key in (r.get('backends') or {})})
    backend_mismatches = []
    for key in backend_keys:
        values = {r['source']: (r.get('backends') or {})[key]
                  for r in present if key in (r.get('backends') or {})}
        if len(set(values.values())) > 1:
            backend_mismatches.append({
                'backend': key,
                'values': values,
            })
    backends_consistent = not backend_mismatches

    if not present:
        status = 'no_sources'
    elif missing or missing_prov:
        status = 'provenance_missing'
    elif len(commits) > 1:
        status = 'mixed_revisions'
    elif not shared_seeds:
        status = 'no_shared_seeds'
    elif not backends_consistent:
        status = 'mixed_backends'
    elif dirty:
        status = 'comparable_dirty_tree'
    else:
        status = 'comparable'
    comparable_for_ranking = status in (
        'comparable', 'comparable_dirty_tree')

    warnings = []
    if status == 'mixed_revisions':
        warnings.append(
            'join sources were produced at different git commits '
            f'{commits}; the cross-source ranking mixes code revisions')
    if status == 'provenance_missing':
        warnings.append(
            'join provenance is incomplete (missing sources '
            f'{missing}; missing git commit {missing_prov})')
    if status == 'no_shared_seeds':
        warnings.append(
            'join sources share no seed; seed-level differences are '
            'confounded with the source experiment')
    if status == 'mixed_backends':
        warnings.append(
            'join sources resolved different memory backends: '
            f'{backend_mismatches}')
    if status == 'comparable_dirty_tree':
        warnings.append(
            'join sources share a git commit but were produced from a '
            f'dirty checkout ({dirty}); the exact code revision is not '
            'recoverable from the commit alone')

    summary = {
        'status': status,
        'comparable_for_ranking': comparable_for_ranking,
        'n_sources_requested': len(records),
        'n_sources_readable': len(present),
        'missing_sources': missing,
        'missing_provenance': missing_prov,
        'git_commits': commits,
        'dirty_sources': dirty,
        'seed_status': seed_status,
        'shared_seeds': shared_seeds,
        'n_shared_seeds': len(shared_seeds),
        'backends_consistent': backends_consistent,
        'backend_mismatches': backend_mismatches,
        'warnings': warnings,
    }
    audit_rows = [_source_audit_row(r) for r in records]
    audit_rows.append({
        'section': 'source_audit',
        'source': 'cross_source',
        **summary,
    })
    return audit_rows, summary


def build_quality_join(spec, context_T=MQAR_SEQ_LEN, source_records=None):
    """Join existing Exp 4/5/8 measured rows with their analytic budget.

    Returns ``(rows, skipped)`` where ``skipped`` lists per-source reasons;
    a missing source is a documented skip, never a silent zero. Each
    joined row carries the source's git commit, dirty flag, seed list and
    file hash, so a mixed-revision join is visible at row level rather
    than only in the run log.
    """
    if source_records is None:
        source_records = [_load_join_source(source, path)
                          for source, path in JOIN_SOURCES]
    rows = []
    skipped = []
    for rec in source_records:
        source = rec['source']
        if not rec.get('readable'):
            skipped.append(rec.get('error')
                           or f"{rec.get('path')} unreadable; join skipped")
            continue
        for r in rec['rows']:
            if not isinstance(r, dict):
                continue
            acc = r.get('mean_acc')
            if acc is None or not isinstance(acc, (int, float)) \
                    or not math.isfinite(acc):
                continue
            budget = None
            label = None
            if source == 'exp4_mqar':
                op = r.get('op')
                if op in ('softmax', 'kda', 'csa', 'hca'):
                    budget = op_read_elements(op, context_T,
                                              spec)['total_read_elements']
                    label = str(op)
            elif source == 'exp5_ablation':
                ratio = _parse_ratio(r.get('ratio'))
                if ratio is not None and sum(ratio) > 0:
                    budget = layout_read_elements(
                        context_T, ratio, spec)['total_read_elements']
                    label = ':'.join(str(x) for x in ratio)
            else:  # exp8
                topk = r.get('topk')
                variant = r.get('variant')
                if isinstance(topk, int) and variant is not None:
                    local = {k: dict(v) if isinstance(v, dict) else v
                             for k, v in spec.items()}
                    local['csa']['topk'] = topk
                    budget = layout_read_elements(
                        context_T, (3, 1, 1), local)['total_read_elements']
                    label = f'{variant} (k={topk})'
            if budget is None:
                continue
            rows.append({
                'section': 'quality_join',
                'source': source,
                'label': label,
                'context_T': context_T,
                'mean_acc': float(acc),
                'ci95_acc': r.get('ci95_acc'),
                'n_seeds_ok': r.get('n_seeds_ok'),
                'mean_fwd_ms': r.get('mean_fwd_ms'),
                'total_read_elements_per_query': budget,
                'source_git_commit': rec.get('git_commit'),
                'source_git_dirty': rec.get('git_dirty'),
                'source_seeds': list(rec.get('seeds') or []),
                'source_sha256': rec.get('sha256'),
                'source_schema_version': rec.get('schema_version'),
            })
    return rows, skipped


def build_ranking_summary(join_rows, specs, aT=MQAR_SEQ_LEN, bT=8192,
                          comparability=None):
    """Compare the Exp 5 layout accuracy ranking with budget rankings.

    Only ``exp5_ablation`` rows are used (one row per layout, its own
    controlled-depth measurement). The comparison at ``aT`` is the measured
    context; at ``bT`` it extrapolates the same layouts' budgets to long
    context in each geometry. Over 7 layouts this is exploratory.

    ``comparability`` is the summary from :func:`audit_join_sources`. It
    is copied onto every ranking row so a consumer cannot read the
    Spearman coefficient without also seeing whether the source runs are
    on a common revision/seed set.
    """
    cmp = comparability if isinstance(comparability, dict) else {}
    source_fields = {
        'source_comparability': cmp.get('status'),
        'source_git_commits': cmp.get('git_commits'),
        'source_shared_seeds': cmp.get('shared_seeds'),
        'comparable_across_sources': cmp.get('comparable_for_ranking'),
    }
    exp5 = [r for r in join_rows if r.get('source') == 'exp5_ablation']
    if len(exp5) < 3:
        return [{
            'section': 'ranking_summary',
            'skipped_reason': (
                'fewer than 3 exp5_ablation rows were joined; ranking '
                'summary skipped'),
            **source_fields,
        }]
    labels = [r['label'] for r in exp5]
    accs = [r['mean_acc'] for r in exp5]

    def _order_and_spearman(budgets):
        order = [lbl for _b, lbl in
                 sorted(zip(budgets, labels), key=lambda t: t[0])]
        return order, spearman(budgets, accs)

    out = []
    for geom, spec in specs.items():
        ratios = [_parse_ratio(lbl) for lbl in labels]
        budgets = [layout_read_elements(aT, r, spec)['total_read_elements']
                   for r in ratios]
        order, rho = _order_and_spearman(budgets)
        # Decode-step view: budget of the first token decoded after a
        # aT-token prefill (t = aT), not the aT-prefill average.
        dec_budgets = [instant_layout_read_elements(r, aT, spec)
                       ['total_read_elements'] for r in ratios]
        dec_order, dec_rho = _order_and_spearman(dec_budgets)
        out.append({
            'section': 'ranking_summary',
            'geometry': geom,
            'context_T': aT,
            'decode_budget_position': aT,
            'n_layouts': len(labels),
            'labels': labels,
            'layout_order_by_budget_ascending': order,
            'spearman_acc_vs_budget': rho,
            'layout_order_by_decode_budget_ascending': dec_order,
            'spearman_acc_vs_decode_budget': dec_rho,
            **source_fields,
        })
        long_budgets = [layout_read_elements(bT, r, spec)
                        ['total_read_elements'] for r in ratios]
        long_order, long_rho = _order_and_spearman(long_budgets)
        long_dec_budgets = [instant_layout_read_elements(r, bT, spec)
                            ['total_read_elements'] for r in ratios]
        long_dec_order, long_dec_rho = _order_and_spearman(long_dec_budgets)
        out.append({
            'section': 'ranking_summary',
            'geometry': geom,
            'context_T': bT,
            'decode_budget_position': bT,
            'n_layouts': len(labels),
            'labels': labels,
            'layout_order_by_budget_ascending': long_order,
            'spearman_acc_vs_budget': long_rho,
            'layout_order_by_decode_budget_ascending': long_dec_order,
            'spearman_acc_vs_decode_budget': long_dec_rho,
            'note': 'long-context budget is an extrapolation of the same '
                    'layouts; accuracies were measured at T=16 only',
            'interpretation_caveat': (
                'A positive rho here does NOT establish that budget buys '
                'accuracy: the long-context budget ordering is largely set '
                'by operator composition (CSA-heavy layouts carry the O(T) '
                'indexer term), and composition also drives the measured '
                'short-context accuracy. The rank correlation detects the '
                'composition confound, not a causal budget effect.'),
            **source_fields,
        })
    return out


def _print_curves(rows, t_grid):
    print('\n' + '=' * 96)
    print('Analytic read budget (avg elements per query per layer)')
    print('=' * 96)
    for geom in ('mqar', 'paper'):
        print(f'\ngeometry={geom}')
        header = f"{'kind':>8} | " + ' | '.join(
            f'T={T:>6}' for T in t_grid)
        print(header)
        print('-' * len(header))
        for name in ('softmax', 'kda', 'csa', 'hca'):
            vals = []
            for T in t_grid:
                r = next(r for r in rows if r['section'] == 'budget_curve'
                         and r['geometry'] == geom and r['row_kind'] == 'op'
                         and r['name'] == name and r['T'] == T)
                vals.append(f"{r['total_read_elements']:>8.1f}")
            print(f'{name:>8} | ' + ' | '.join(vals))
        for ratio in LAYOUTS:
            label = f'{ratio[0]}:{ratio[1]}:{ratio[2]}'
            vals = []
            for T in t_grid:
                r = next(r for r in rows if r['section'] == 'budget_curve'
                         and r['geometry'] == geom
                         and r['row_kind'] == 'layout'
                         and r['name'] == label and r['T'] == T)
                vals.append(f"{r['total_read_elements']:>8.1f}")
            print(f'{label:>8} | ' + ' | '.join(vals))


def _print_decode_frontier(rows, t_grid):
    """Marginal (decode-step) budget: one query at position t, not averaged."""
    print('\n' + '=' * 96)
    print('Decode-step marginal budget (elements read by ONE query at t)')
    print('=' * 96)
    for geom in ('mqar', 'paper'):
        print(f'\ngeometry={geom}')
        header = f"{'kind':>8} | " + ' | '.join(
            f't={t:>6}' for t in t_grid)
        print(header)
        print('-' * len(header))
        for name in ('softmax', 'kda', 'csa', 'hca'):
            vals = []
            for t in t_grid:
                r = next(r for r in rows
                         if r['section'] == 'decode_budget_curve'
                         and r['geometry'] == geom and r['row_kind'] == 'op'
                         and r['name'] == name and r['t'] == t)
                vals.append(f"{r['total_read_elements']:>8.1f}")
            print(f'{name:>8} | ' + ' | '.join(vals))


def main():
    info = configure_torch_for_device()
    device = info.device
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info('=' * 70)
    logger.info('Experiment 9: KV Read-Budget Frontier '
                '(top-k budget / delta-rule O(1) / hybrid ratio)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')

    t_grid = _parse_int_list(
        os.environ.get('BUDGET_T_GRID', DEFAULT_T_GRID),
        'BUDGET_T_GRID', min_value=1)
    topk_values = _parse_int_list(
        os.environ.get('BUDGET_TOPK', DEFAULT_TOPK),
        'BUDGET_TOPK', min_value=0)
    prefill_lens = _parse_int_list(
        os.environ.get('BUDGET_PREFILL_LENS', DEFAULT_PREFILL_LENS),
        'BUDGET_PREFILL_LENS', min_value=1)
    n_decode = parse_int_env(
        'BUDGET_DECODE_TOKENS', DEFAULT_DECODE_TOKENS, min_value=1,
        logger=logger)
    repeats = parse_int_env(
        'BUDGET_REPEATS', DEFAULT_REPEATS, min_value=1, logger=logger)
    logger.info(f'  T grid: {t_grid}')
    logger.info(f'  decode: topk={topk_values}, prefill={prefill_lens}, '
                f'n_decode={n_decode}, repeats={repeats}')

    specs = {'mqar': mqar_spec(), 'paper': paper_spec()}
    rows = build_budget_curves(t_grid, specs)
    _print_curves(rows, t_grid)
    decode_curve_rows = build_decode_budget_curves(t_grid, specs)
    rows.extend(decode_curve_rows)
    _print_decode_frontier(decode_curve_rows, t_grid)

    logger.info('\n-- decode top-k sweep (fused 3:1:1 hybrid) --')
    decode_rows = run_decode_sweep(
        device, topk_values, prefill_lens, n_decode, repeats)
    rows.extend(decode_rows)

    source_records = [_load_join_source(source, path)
                      for source, path in JOIN_SOURCES]
    source_audit_rows, source_audit = audit_join_sources(source_records)
    rows.extend(source_audit_rows)
    join_rows, join_skipped = build_quality_join(
        specs['mqar'], source_records=source_records)
    rows.extend(join_rows)
    for reason in join_skipped:
        logger.info(f'  join skipped: {reason}')
    if source_audit['status'] not in ('comparable',
                                      'comparable_dirty_tree'):
        logger.warning('  quality join is NOT comparable across sources: '
                       f"{source_audit['status']} ({'; '.join(source_audit['warnings'])})")
    rows.extend(build_ranking_summary(
        join_rows, specs, comparability=source_audit))

    metadata = {
        'experiment': 'exp9_budget_frontier',
        'design': (
            'One currency for top-k budget / delta-rule state / hybrid '
            'ratio: analytical KV-side elements read per query per layer, '
            'in two unified views. budget_curve rows are the prefill '
            'average over positions 0..T-1; decode_budget_curve rows are '
            'the marginal view at position t (the first token decoded '
            'after a t-token prefill); the rolling mean of the marginal '
            'view is exactly the prefill average (pinned by '
            'test_budget_frontier). Curves cover the 4 core operators and '
            'the 7 Exp 5 layouts in MQAR and paper geometries; a measured '
            'decode-latency sweep over csa_topk joins BOTH views; Exp 4/5/8 '
            'rows (if present) are joined with their measured-context '
            'budget, and the layout ranking is reported under both views.'),
        'prefill_decode_unification': (
            'kv_budget.instant_read_elements(op, t) charges exactly the '
            'elements read by the query at position t under the same causal '
            'conventions as the averaged model; mean_{t<T} instant == '
            'op_read_elements(op, T) by construction (identity tests in '
            'test_budget_frontier). A decode sweep that '
            'measures per-token latency should therefore be compared '
            'against the decode_window_* fields (mean marginal over the '
            'actually decoded positions [prefill_len, prefill_len + n)) '
            'rather than the prefill average, which mixes in cheap early '
            'positions. The prefill average is kept in '
            'total_read_elements_per_query for backward compatibility.'),
        'currency_definition': (
            'element = one scalar of the KV-side representation (compressed '
            'shared-KV vector counts once; window token contributes c; KDA '
            'state contributes hv*d_k*d_v). Causal masks match the '
            'implementations (inclusive own-token for softmax/window, '
            'strictly preceding compressed blocks). KDA reads its state '
            'twice per step (delta masked read + output read). CSA charges '
            'the lightning-indexer scan over ALL visible compressed blocks '
            '(it runs before top-k in the reference implementation, '
            'including topk=0). Query-side projections excluded.'),
        'model_scope': (
            'analytical read counts, NOT profiled traffic; library-internal '
            'materialization (unfold, temporaries) is out of scope, as in '
            'the Exp 7 compulsory-traffic caveat.'),
        'geometries': specs,
        'decode_geometry': (
            'run_decoding.HybridDecoding defaults (d_model=64, c=8, dc=8, '
            'cI=4, nIh=1, m=4, m2=4, sw=4, 3:1:1); decode numbers must not '
            'be compared head-to-head with Exp 4/5/8 quality geometry.'),
        't_grid': t_grid,
        'decode_sweep': {
            'csa_topk': topk_values,
            'prefill_lens': prefill_lens,
            'n_decode': n_decode,
            'repeats': repeats,
        },
        'join_sources': [name for name, _ in JOIN_SOURCES],
        'join_skipped': join_skipped,
        'source_audit': source_audit,
        'caveats': [
            'The budget is an analytical lower bound on KV-side reads; '
            'matching latency to budget is confounded by Python-loop '
            'overhead in the reference implementation.',
            'Exp 5 layouts are depth-controlled, not budget-controlled; '
            'the ranking summary is exploratory (7 layouts, Spearman). '
            'Correlations between the measured T=16 accuracy and the '
            'long-context budget ranking confound operator composition '
            'with budget: they do not identify a causal budget effect.',
            'Long-context layout budgets (T>16) are extrapolations; no '
            'quality was measured at those contexts.',
            'topk=0 still charges the indexer scan because the reference '
            'operator scores all blocks before skipping the gather; a '
            'production engine could skip selection entirely.',
            'The decode-step (marginal) view is the same analytical model '
            'evaluated at one position; it excludes Python-loop and '
            'cache-management overhead, so it is the right shape of the '
            'decode cost, not a latency prediction.',
            'Cross-experiment join provenance is audited in source_audit. '
            'The ranking is only marked comparable when every source has '
            'provenance, one shared git commit, a non-empty shared seed '
            'set, and consistent resolved backends; a dirty checkout is '
            'reported separately because the commit alone then does not '
            'pin the exact code.',
        ],
        'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
        'seed': DECODE_SWEEP_SEED,
        'seed_scheme': (
            'set_global_seed(0, deterministic=True) before every decode-sweep '
            'model construction; analytical budget rows are seed-free'),
        'provenance': capture_provenance(),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, rows, kind='exp9_budget_frontier',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')

    errors = sum(1 for r in decode_rows if 'error' in r)
    if (not any(r['section'] == 'budget_curve' for r in rows)
            or not any(r['section'] == 'decode_budget_curve' for r in rows)
            or errors):
        logger.error(
            f'\nexp9 incomplete: {errors} decode-sweep cells errored; '
            f'returning non-zero.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
