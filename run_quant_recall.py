"""Experiment 10 — quantized-state recall / cost probe for both memory routes.

Motivation
----------
The Exp 5 eight-group matrix crosses the memory route (KDA vs Mamba-2)
with the layer layout; Exp 8/9 measure CSA indexer recall and structural
recall reach; Exp 9 (`kv_budget`) prices KV reads in fp16 *elements*. None
of them can answer the serving question the `quantization` axis raises:

    At a given per-state byte budget, does the recurrent route still
    recall the associations it stored?

This probe puts the two recurrent routes of the eight-group matrix on a
**recall-cost curve**:

* **Recall.** Orthonormal keys are the exact-recall best case of the
  delta-rule / SSD state (pinned for KDA by
  ``test_budget_frontier.test_delta_rule_orthonormal_keys_exact_recall``).
  The probe writes ``T`` (key, value) pairs through the route's real
  reference operator — ``ops_kda.naive_recurrent_kda`` or
  ``ops_mamba2_scan`` — one token at a time, **fake-quantizing the state
  after every write** (``kv_quant.fake_quantize``), then reads every key
  back from the final state and decodes the nearest stored value. With no
  quantization the read-back error is fp32 round-off (~1e-6) and every
  read is exact; with a low-bit grid the accumulated quantization noise
  eventually flips the nearest-value decode. That flip rate is the
  measured recall.
* **Cost.** ``kv_quant.bits_per_element`` prices the state storage
  including the per-group scale (and zero-point) metadata, so the x-axis
  is real bytes, not "nominal bits". ``state_bytes`` uses the operator's
  actual state element count (``hv * d_k * d_v`` for KDA, the same count
  for the probe's Mamba-2 geometry).

The output is a small explicit **route x bits matrix** (the design of this
repo's random-inspiration words): every (route, bits, group_size) cell
× seeds, plus a frontier summary naming the cheapest *exact-recall* cell
per route.

Honesty / scope
---------------
* Single-operator probe, **no training**: weights are irrelevant because
  the probe feeds the state recurrences directly with planted keys/values.
  It measures state-quantization damage, not end-task quality.
* Orthonormal keys are the best case for exact recall; real keys are
  correlated and the recall frontier at 2-3 bits will be worse.
* The state is quantized as a **flattened contiguous buffer** (groups of
  ``group_size`` adjacent elements share a scale), matching a state cache
  kept as one contiguous quantized buffer. Other layouts (per-head-dim
  groups) give different low-bit numbers.
* Only the recurrent *state* is quantized here. Activations, logits,
  weights, and the CSA/HCA compressed-KV caches are out of scope; the
  ``kv_budget`` byte bridge (``op_read_bytes`` / ``layout_read_bytes``)
  is what generalizes the accounting to those components.
* Multi-seed aggregation reports mean, sample std, 95% t-CI, min and max
  recall; "exact" means min-over-seeds == 1.0 (conservative). Three seeds
  keep the probe fast; the 2-bit cells are the noisy ones and are
  reported with their full spread.
* Nearest-value decisions are made from **fp64 distances** even when the
  recurrence itself runs in fp32. The fp32 distance matrix can round two
  candidates to the same float and let tensor order (not the state) decide
  the recall, which is exactly the numerical-stability failure a quantized
  state is most exposed to. The relative decision margin
  ``(d_wrong - d_correct) / (d_wrong + d_correct)`` and an ambiguous-tie
  count are reported next to recall so a cell that only barely decodes is
  not read as a comfortable pass.
* **Iso-budget frontier.** ``frontier_summary`` answers "what is the
  cheapest exact cell"; it does not answer the benchmark-design question
  "at a fixed state-byte budget, which measured cell has the best
  worst-seed recall?" ``iso_budget_frontier`` rows answer that under the
  same conservative min-over-seeds objective, so the two routes can be
  compared at matched storage cost instead of at unmatched byte counts.

Output
------
``results/exp10_quant_recall.json`` (versioned envelope via
``results_schema``) with ``row_kind`` in {``quant_recall``,
``frontier_summary``, ``iso_budget_frontier``}; CLI::

    python3 run_quant_recall.py
    QUANT_BITS=2,3,4 QUANT_SEEDS=0,1 python3 run_quant_recall.py
    QUANT_BUDGETS=18,34,64 python3 run_quant_recall.py
"""

from __future__ import annotations

import logging
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    append_run_manifest,
    capture_provenance,
    configure_torch_for_device,
    make_run_manifest_entry,
    make_seeded_generator,
)
from kv_quant import (  # noqa: E402
    FP16_BITS,
    FP8Spec,
    QuantSpec,
    bits_per_element,
    compression_ratio,
    fake_quantize,
    spec_summary,
    storage_bytes,
)
from ops_kda import naive_recurrent_kda  # noqa: E402
from ops_mamba import naive_mamba2_scan  # noqa: E402
from results_schema import write as write_schema  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp10_quant_recall.json')

ROUTES = ('kda', 'mamba2')
DEFAULT_BITS = (2, 3, 4, 6, 8)
DEFAULT_GROUP_SIZES = (16, 64)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_N_KEYS = 16
DEFAULT_VALUE_DIM = 8
ASYMMETRIC_PROBE = QuantSpec(bits=4, group_size=64, symmetric=False)


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------
def planted_batch(n_keys: int, value_dim: int, seed: int, device='cpu'):
    """One planted-recall episode: orthonormal keys + random values.

    Keys are the rows of a random orthogonal matrix (QR), so the state can
    hold all ``n_keys`` associations exactly at full precision. Values are
    i.i.d. standard normal; decoding is nearest-value in ``R**value_dim``.
    Returned tensors are ``keys [n_keys, n_keys]`` (unit-norm rows) and
    ``values [n_keys, value_dim]``.
    """
    gen = make_seeded_generator(seed, device)
    raw = torch.randn(n_keys, n_keys, device=device, generator=gen)
    q, _r = torch.linalg.qr(raw)
    keys = q.t().contiguous()
    values = torch.randn(n_keys, value_dim, device=device, generator=gen)
    return keys, values


# --------------------------------------------------------------------------
# Route-level planted write/read (quantize after every write)
# --------------------------------------------------------------------------
def _kda_state(keys: torch.Tensor, values: torch.Tensor, spec,
               quantize_after_write: bool):
    """Write pairs through ``naive_recurrent_kda``; return the final state.

    With ``quantize_after_write=True`` the state is fake-quantized after
    every one-token step (a decoder storing the state quantized).
    """
    T, K = keys.shape
    V = values.shape[-1]
    q = k = keys.view(1, T, 1, K)
    v = values.view(1, T, 1, V)
    g = torch.zeros(1, T, 1, K, device=keys.device, dtype=keys.dtype)
    beta = torch.ones(1, T, 1, device=keys.device, dtype=keys.dtype)
    S = None
    for t in range(T):
        _, S = naive_recurrent_kda(
            q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
            g[:, t:t + 1], beta[:, t:t + 1],
            initial_state=S, output_final_state=True)
        if quantize_after_write and spec is not None:
            S = fake_quantize(S.reshape(-1), spec).reshape(S.shape)
    return S


def _mamba2_state(keys: torch.Tensor, values: torch.Tensor, spec,
                  quantize_after_write: bool):
    """Write pairs through ``naive_mamba2_scan``; return the final state.

    Geometry: ``d_state = n_keys`` (the key dimension), ``headdim =
    value_dim`` (the value dimension), ``ngroups = nheads = 1``, decay
    ``A = 0`` (no forgetting), ``dt = 1``, skip ``D = 0``. Then the state
    after all writes is ``S[i, j] = sum_t values[t, i] * keys[t, j]`` and
    reading with ``C = keys[i]`` returns ``values[i]`` exactly when keys
    are orthonormal.
    """
    T, K = keys.shape
    V = values.shape[-1]
    x = values.view(1, T, 1, V)
    dt = torch.ones(1, T, 1, device=keys.device, dtype=keys.dtype)
    B = keys.view(1, T, 1, K)
    C = B.clone()
    A = torch.zeros(1, device=keys.device, dtype=keys.dtype)
    D = torch.zeros(1, device=keys.device, dtype=keys.dtype)
    S = None
    for t in range(T):
        _, S = naive_mamba2_scan(
            x[:, t:t + 1], dt[:, t:t + 1], B[:, t:t + 1],
            C[:, t:t + 1], A, D,
            initial_state=S, output_final_state=True)
        if quantize_after_write and spec is not None:
            S = fake_quantize(S.reshape(-1), spec).reshape(S.shape)
    return S


_STATE_FN = {'kda': _kda_state, 'mamba2': _mamba2_state}


def read_back(route: str, keys: torch.Tensor, state: torch.Tensor):
    """Read every planted key from the final state.

    KDA: ``read_i = S^T k_i`` with ``S [1, 1, K, V]``.
    Mamba-2: ``read_i = C_i . S`` with ``S [1, 1, V, K]``.
    Returns ``reads [T, value_dim]``.
    """
    T = keys.shape[0]
    if route == 'kda':
        return torch.stack([state[0, 0].t() @ keys[i] for i in range(T)])
    if route == 'mamba2':
        return torch.stack([state[0, 0] @ keys[i] for i in range(T)])
    raise ValueError(f'read_back: unknown route {route!r}; expected one of '
                     f'{list(ROUTES)}.')


#: Relative margin below which a nearest-value decision is treated as a
#: numerical tie (both distances differ by less than this fraction of their
#: sum). Such a decision can flip between two mathematically equivalent
#: code paths, so it must not be counted as a stable recall.
DECISION_TIE_TOL = 1e-9


def _decode_recall(reads: torch.Tensor, values: torch.Tensor):
    """Nearest-value decode: recall, max error and decision margin.

    Distances are always evaluated in fp64. Casting an fp64 unsaturated /
    quantized state down to fp32 before ``cdist`` can collapse two
    candidates onto the same float; the argmin then follows tensor order
    rather than the storage grid, conflating a real recall loss with a
    round-off tie. The relative margin
    ``(d_wrong - d_correct) / (d_wrong + d_correct)`` is in ``[-1, 1]``
    and is reported so near-boundary cells are visible.
    """
    reads64 = reads.detach().to(torch.float64)
    values64 = values.detach().to(torch.float64)
    dist = torch.cdist(reads64, values64)
    pred = dist.argmin(dim=-1)
    target = torch.arange(values.shape[0], device=values.device)
    correct = pred == target
    recall = float(correct.to(torch.float64).mean())
    max_err = float((reads64 - values64).abs().max())
    if dist.shape[0] < 2:
        return recall, max_err, {
            'readback_min_margin': None,
            'readback_mean_margin': 0.0,
            'n_ambiguous_decisions': 0,
        }
    target_col = target[:, None]
    correct_dist = dist.gather(1, target_col).squeeze(1)
    wrong_dist = dist.masked_fill(
        torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device),
        float('inf')).min(dim=-1).values
    denom = (correct_dist + wrong_dist).clamp_min(
        torch.finfo(torch.float64).tiny)
    margin = (wrong_dist - correct_dist) / denom
    correct_margin = margin[correct]
    return recall, max_err, {
        'readback_min_margin': (float(correct_margin.min())
                                if correct_margin.numel() else None),
        'readback_mean_margin': float(margin.mean()),
        'n_ambiguous_decisions': int(
            (margin.abs() <= DECISION_TIE_TOL).sum().item()),
    }


def run_planted_recall(route: str, keys: torch.Tensor, values: torch.Tensor,
                       spec: QuantSpec | FP8Spec | None = None) -> dict:
    """One planted-recall episode for one route and quantization spec.

    ``spec=None`` is the full-precision reference (no quantization).
    ``FP8Spec`` exercises the block-scaled FP8 grid through the same
    stepwise quantize-after-write path.
    """
    if route not in _STATE_FN:
        raise ValueError(f'run_planted_recall: unknown route {route!r}; '
                         f'expected one of {list(ROUTES)}.')
    if spec is not None and not isinstance(spec, (QuantSpec, FP8Spec)):
        raise TypeError(f'run_planted_recall: spec must be None, a '
                        f'QuantSpec or an FP8Spec, got {spec!r}')
    state = _STATE_FN[route](keys, values, spec,
                             quantize_after_write=spec is not None)
    reads = read_back(route, keys, state)
    recall, max_err, margins = _decode_recall(reads, values)
    return {
        'readback_recall': recall,
        'max_read_error': max_err,
        'state_elements': int(state.numel()),
        **margins,
    }


def full_sequence_state(route: str, keys: torch.Tensor,
                        values: torch.Tensor) -> torch.Tensor:
    """Single-call (full-sequence) reference state for the cross-check.

    The stepwise write path with ``spec=None`` must reproduce this state to
    fp32 round-off; the test suite pins it.
    """
    T, K = keys.shape
    V = values.shape[-1]
    if route == 'kda':
        q = k = keys.view(1, T, 1, K)
        v = values.view(1, T, 1, V)
        g = torch.zeros(1, T, 1, K, device=keys.device, dtype=keys.dtype)
        beta = torch.ones(1, T, 1, device=keys.device, dtype=keys.dtype)
        _, S = naive_recurrent_kda(q, k, v, g, beta,
                                   output_final_state=True)
        return S
    if route == 'mamba2':
        x = values.view(1, T, 1, V)
        dt = torch.ones(1, T, 1, device=keys.device, dtype=keys.dtype)
        B = keys.view(1, T, 1, K)
        C = B.clone()
        A = torch.zeros(1, device=keys.device, dtype=keys.dtype)
        D = torch.zeros(1, device=keys.device, dtype=keys.dtype)
        _, S = naive_mamba2_scan(x, dt, B, C, A, D,
                                 output_final_state=True)
        return S
    raise ValueError(f'full_sequence_state: unknown route {route!r}.')


# --------------------------------------------------------------------------
# Sweep construction
# --------------------------------------------------------------------------
def default_sweep_specs(bits_list=DEFAULT_BITS,
                        group_sizes=DEFAULT_GROUP_SIZES,
                        include_asymmetric=True) -> list:
    """``[None]`` (fp16 reference) + one spec per (bits, group) + asym probe."""
    specs = [None]
    for bits in bits_list:
        for group in group_sizes:
            specs.append(QuantSpec(bits=bits, group_size=group))
    if include_asymmetric:
        specs.append(ASYMMETRIC_PROBE)
    return specs


def _aggregate(values: list) -> dict:
    """Mean/min/max plus sample std and a two-sided 95% t-interval.

    An empty value list yields an explicit empty aggregate (``n = 0`` and
    ``None`` statistics) instead of a ``ZeroDivisionError`` from the mean
    and a ``ValueError`` from ``min()`` on an empty sequence.
    """
    n = len(values)
    if n == 0:
        return {'mean': None, 'std': None, 'ci95': None,
                'min': None, 'max': None, 'n': 0}
    mean = sum(values) / n
    std = None
    ci95 = None
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        from run_quality import _t_crit_975
        crit = _t_crit_975(n)
        if crit is not None:
            half = crit * (std / math.sqrt(n))
            ci95 = [mean - half, mean + half]
    return {'mean': mean, 'std': std, 'ci95': ci95,
            'min': min(values), 'max': max(values), 'n': n}


def build_quant_recall_rows(device='cpu', seeds=DEFAULT_SEEDS,
                            bits_list=DEFAULT_BITS,
                            group_sizes=DEFAULT_GROUP_SIZES,
                            n_keys=DEFAULT_N_KEYS,
                            value_dim=DEFAULT_VALUE_DIM,
                            routes=ROUTES,
                            include_asymmetric=True,
                            budgets=None) -> list:
    """Run the probe matrix and return flat, JSON-safe rows.

    One row per (route, spec) cell aggregating over seeds, a
    ``frontier_summary`` row per route (cheapest exact-recall cell), and
    one ``iso_budget_frontier`` row per (route, byte cap). ``budgets=None``
    uses every measured cell's ``state_bytes`` as a cap (plus fp16), so the
    frontier never depends on an unmeasured/interpolated cost.
    """
    rows: list = []
    seeds = list(seeds)
    if not seeds:
        raise ValueError(
            'build_quant_recall_rows: seeds must be a non-empty iterable '
            '(the per-cell aggregate is taken over the seeds).')
    for route in routes:
        for spec in default_sweep_specs(bits_list, group_sizes,
                                        include_asymmetric):
            recalls, max_errs, state_elements = [], [], None
            margins, ambiguous = [], []
            for seed in seeds:
                keys, values = planted_batch(n_keys, value_dim, seed, device)
                out = run_planted_recall(route, keys, values, spec)
                recalls.append(out['readback_recall'])
                max_errs.append(out['max_read_error'])
                margins.append(out['readback_min_margin'])
                ambiguous.append(out['n_ambiguous_decisions'])
                if state_elements is None:
                    state_elements = out['state_elements']
                elif state_elements != out['state_elements']:
                    raise AssertionError(
                        'state element count changed across seeds; the '
                        'probe geometry must be fixed')
            rec = _aggregate(recalls)
            errmax = _aggregate(max_errs)
            finite_margins = [m for m in margins if m is not None]
            if spec is None:
                row = {
                    'row_kind': 'quant_recall',
                    'route': route,
                    'quantized': False,
                    'bits': None,
                    'group_size': None,
                    'symmetric': None,
                    'bits_per_element': FP16_BITS,
                    'state_bytes': state_elements * FP16_BITS / 8.0,
                    'compression_vs_fp16': 1.0,
                }
            else:
                row = {
                    'row_kind': 'quant_recall',
                    'route': route,
                    'quantized': True,
                    'bits': spec.bits,
                    'group_size': spec.group_size,
                    'symmetric': spec.symmetric,
                    'bits_per_element': bits_per_element(spec),
                    'state_bytes': storage_bytes(state_elements, spec),
                    'compression_vs_fp16': compression_ratio(spec),
                }
            row.update({
                'n_keys': int(n_keys),
                'value_dim': int(value_dim),
                'context_T': int(n_keys),
                'seeds': [int(s) for s in seeds],
                'state_elements': int(state_elements),
                'fp16_state_bytes': state_elements * FP16_BITS / 8.0,
                'recall_mean': rec['mean'],
                'recall_std': rec['std'],
                'recall_ci95': rec['ci95'],
                'recall_min': rec['min'],
                'recall_max': rec['max'],
                'n_seeds': rec['n'],
                'max_read_error_mean': errmax['mean'],
                'max_read_error_max': errmax['max'],
                'decision_margin_min': (min(finite_margins)
                                        if finite_margins else None),
                'decision_margin_mean': (
                    sum(finite_margins) / len(finite_margins)
                    if finite_margins else None),
                'n_ambiguous_decisions': int(sum(ambiguous)),
                'decision_tie_tolerance': DECISION_TIE_TOL,
                'exact_recall': bool(rec['min'] == 1.0),
            })
            rows.append(row)
    rows.extend(build_frontier_summary(rows, routes))
    rows.extend(build_iso_budget_frontier(rows, routes, budgets=budgets))
    return rows


def build_frontier_summary(rows: list, routes=ROUTES) -> list:
    """Cheapest exact-recall cell per route (conservative min-over-seeds)."""
    out = []
    for route in routes:
        cells = [r for r in rows
                 if r['row_kind'] == 'quant_recall' and r['route'] == route]
        if not cells:
            continue
        fp16 = next(r for r in cells if not r['quantized'])
        exact = [r for r in cells if r['exact_recall']]
        any_cell = [r for r in cells if r['quantized']]
        out.append({
            'row_kind': 'frontier_summary',
            'route': route,
            'fp16_state_bytes': fp16['fp16_state_bytes'],
            'n_cells': len(cells),
            'cheapest_exact_cell': (
                min(exact, key=lambda r: r['state_bytes'])
                if exact else None),
            'cheapest_cell': (
                min(any_cell, key=lambda r: r['state_bytes'])
                if any_cell else None),
        })
    return out


def _validate_budgets(budgets) -> list:
    """Positive finite byte caps, deduplicated and sorted."""
    if isinstance(budgets, (int, float)) and not isinstance(budgets, bool):
        budgets = (budgets,)
    out = []
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise ValueError(
                f'build_iso_budget_frontier: budget={budget!r} must be a '
                f'positive finite number of bytes.')
        try:
            value = float(budget)
        except OverflowError:
            raise ValueError(
                f'build_iso_budget_frontier: budget={budget!r} is not '
                f'representable as a float.') from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f'build_iso_budget_frontier: budget={budget!r} must be a '
                f'positive finite number of bytes.')
        out.append(value)
    if not out:
        raise ValueError(
            'build_iso_budget_frontier: budgets must be non-empty.')
    return sorted(set(out))


def build_iso_budget_frontier(rows, routes=ROUTES, budgets=None) -> list:
    """Best measured cell at or below each state-byte budget.

    This is the *budget-constrained* companion to
    :func:`build_frontier_summary` (cheapest exact cell). For every cap it
    picks the feasible cell with the highest worst-seed recall
    (``recall_min``), breaking ties by mean recall and then by cheaper
    storage. ``budgets=None`` uses the measured caps themselves (plus the
    fp16 state size), so no cap is an unmeasured interpolation.

    The strict ``<=`` comparison is intentional: a cell whose metadata
    pushes it one byte over the cap is not feasible, and reporting it as
    such is the point of a budget frontier. Each row records the selected
    cell and the headroom so a reader can see whether the choice is tight.
    """
    cells = [r for r in rows if r.get('row_kind') == 'quant_recall']
    if not cells:
        return []
    if budgets is None:
        budgets = sorted({float(r['state_bytes']) for r in cells})
    caps = _validate_budgets(budgets)
    out = []
    for route in routes:
        route_cells = [r for r in cells if r['route'] == route]
        if not route_cells:
            continue
        fp16 = next((r for r in route_cells if not r['quantized']), None)
        for cap in caps:
            feasible = [r for r in route_cells
                        if float(r['state_bytes']) <= cap]
            row = {
                'row_kind': 'iso_budget_frontier',
                'route': route,
                'budget_bytes': cap,
                'fp16_state_bytes': (
                    fp16['fp16_state_bytes'] if fp16 is not None else None),
                'covered': bool(feasible),
                'n_feasible': len(feasible),
            }
            if not feasible:
                row.update({
                    'selected': None,
                    'selected_state_bytes': None,
                    'selected_recall_min': None,
                    'selected_recall_mean': None,
                    'selected_exact_recall': None,
                    'selected_compression_vs_fp16': None,
                    'budget_headroom_bytes': None,
                    'budget_utilization': None,
                })
            else:
                best = max(
                    feasible,
                    key=lambda r: (float(r['recall_min']),
                                   float(r['recall_mean']),
                                   -float(r['state_bytes'])))
                headroom = cap - float(best['state_bytes'])
                row.update({
                    'selected': {
                        'quantized': best['quantized'],
                        'bits': best['bits'],
                        'group_size': best['group_size'],
                        'symmetric': best['symmetric'],
                        'state_bytes': best['state_bytes'],
                        'recall_min': best['recall_min'],
                        'recall_mean': best['recall_mean'],
                        'exact_recall': best['exact_recall'],
                    },
                    'selected_state_bytes': best['state_bytes'],
                    'selected_recall_min': best['recall_min'],
                    'selected_recall_mean': best['recall_mean'],
                    'selected_exact_recall': best['exact_recall'],
                    'selected_compression_vs_fp16':
                        best['compression_vs_fp16'],
                    'budget_headroom_bytes': headroom,
                    'budget_utilization': (
                        float(best['state_bytes']) / cap),
                })
            out.append(row)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _cell_label(row) -> str:
    if not row['quantized']:
        return 'fp16      '
    sym = 'sym' if row['symmetric'] else 'asym'
    return f"{row['bits']}b/g{row['group_size']:<3d}{sym}"


def _print_rows(rows: list) -> None:
    logger.info('\n-- planted-recall x state quantization '
                '(mean [min..max] over seeds) --')
    header = (f"{'route':>7} | {'cell':>12} | {'bits/el':>7} | "
              f"{'bytes':>7} | {'x_fp16':>6} | recall mean+-sd "
              f"[min..max] | max|err|")
    logger.info(header)
    logger.info('-' * len(header))
    for row in rows:
        if row['row_kind'] != 'quant_recall':
            continue
        logger.info(
            f"{row['route']:>7} | {_cell_label(row):>12} | "
            f"{row['bits_per_element']:>7.2f} | {row['state_bytes']:>7.1f} | "
            f"{row['compression_vs_fp16']:>5.2f}x | "
            f"{row['recall_mean']:.3f}+-"
            f"{('n/a' if row['recall_std'] is None else format(row['recall_std'], '.3f'))} "
            f"[{row['recall_min']:.3f}..{row['recall_max']:.3f}] | "
            f"{row['max_read_error_mean']:.3f}")


def _print_frontier(rows: list) -> None:
    logger.info('\n-- recall-cost frontier (cheapest exact-recall cell) --')
    for row in rows:
        if row['row_kind'] != 'frontier_summary':
            continue
        cell = row['cheapest_exact_cell']
        if cell is None:
            logger.info(f"  {row['route']:>7}: no exact-recall cell in the "
                        f"grid (fp16 state = "
                        f"{row['fp16_state_bytes']:.1f} B)")
            continue
        logger.info(
            f"  {row['route']:>7}: {_cell_label(cell)} = "
            f"{cell['bits_per_element']:.2f} bits/element, "
            f"{cell['state_bytes']:.1f} B "
            f"({cell['compression_vs_fp16']:.2f}x smaller than fp16), "
            f"recall min {cell['recall_min']:.3f}")
        cheapest = row['cheapest_cell']
        if cheapest is not None and cheapest is not cell:
            logger.info(
                f"           cheapest cell overall: "
                f"{_cell_label(cheapest)} = "
                f"{cheapest['bits_per_element']:.2f} bits/el, "
                f"recall min {cheapest['recall_min']:.3f}")


def _print_iso_budget(rows: list) -> None:
    logger.info('\n-- iso-byte budget frontier (best worst-seed recall at '
                'or below the cap) --')
    for row in rows:
        if row['row_kind'] != 'iso_budget_frontier':
            continue
        if not row['covered']:
            logger.info(
                f"  {row['route']:>7} cap={row['budget_bytes']:>7.1f} B: "
                f"no measured cell fits")
            continue
        selected = row['selected']
        label = ('fp16' if not selected['quantized'] else
                 f"{selected['bits']}b/g{selected['group_size']}"
                 f"{'sym' if selected['symmetric'] else 'asym'}")
        logger.info(
            f"  {row['route']:>7} cap={row['budget_bytes']:>7.1f} B -> "
            f"{label:>12} {selected['state_bytes']:>7.1f} B "
            f"(headroom {row['budget_headroom_bytes']:.1f} B), "
            f"recall min {selected['recall_min']:.3f} "
            f"mean {selected['recall_mean']:.3f}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_float_list(raw, name, min_value=0.0):
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as e:
            raise ValueError(f'{name}: {part!r} is not a number') from e
        if not math.isfinite(value) or value <= min_value:
            raise ValueError(
                f'{name}: {value!r} must be a finite number > {min_value}')
        values.append(value)
    if not values:
        raise ValueError(f'{name}: empty list')
    return tuple(dict.fromkeys(values))


def _parse_int_list(raw, name, min_value=0):
    values = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as e:
            raise ValueError(f'{name}: {part!r} is not an int') from e
        if value < min_value:
            raise ValueError(f'{name}: {value} < {min_value}')
        values.append(value)
    if not values:
        raise ValueError(f'{name}: empty list')
    return tuple(dict.fromkeys(values))


def main():
    info = configure_torch_for_device()
    device = info.device
    logging.basicConfig(
        level=logging.INFO, format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    logger.info('=' * 70)
    logger.info('Experiment 10: quantized-state recall / cost probe '
                '(KDA vs Mamba-2)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')

    bits_list = _parse_int_list(os.environ.get(
        'QUANT_BITS', ','.join(map(str, DEFAULT_BITS))), 'QUANT_BITS', 2)
    group_sizes = _parse_int_list(os.environ.get(
        'QUANT_GROUP_SIZES', ','.join(map(str, DEFAULT_GROUP_SIZES))),
        'QUANT_GROUP_SIZES', 1)
    seeds = _parse_int_list(os.environ.get(
        'QUANT_SEEDS', ','.join(map(str, DEFAULT_SEEDS))), 'QUANT_SEEDS', 0)
    n_keys = _parse_int_list(os.environ.get('QUANT_N_KEYS',
                                            str(DEFAULT_N_KEYS)),
                             'QUANT_N_KEYS', 2)[0]
    value_dim = _parse_int_list(os.environ.get('QUANT_VALUE_DIM',
                                               str(DEFAULT_VALUE_DIM)),
                                'QUANT_VALUE_DIM', 1)[0]
    budgets_env = os.environ.get('QUANT_BUDGETS')
    budgets = (_parse_float_list(budgets_env, 'QUANT_BUDGETS')
               if budgets_env is not None and budgets_env.strip() else None)
    logger.info(f'  routes: {list(ROUTES)}')
    logger.info(f'  bits: {list(bits_list)}, groups: {list(group_sizes)}, '
                f'seeds: {list(seeds)}')
    logger.info(f'  geometry: n_keys={n_keys}, value_dim={value_dim}')
    logger.info(f'  iso-byte caps: '
                f'{list(budgets) if budgets is not None else "measured"}')

    rows = build_quant_recall_rows(
        device=device, seeds=seeds, bits_list=bits_list,
        group_sizes=group_sizes, n_keys=n_keys, value_dim=value_dim,
        budgets=budgets)
    _print_rows(rows)
    _print_frontier(rows)
    _print_iso_budget(rows)

    specs_provenance = [spec_summary(QuantSpec(bits=b, group_size=g))
                        for b in bits_list for g in group_sizes]
    specs_provenance.append(spec_summary(ASYMMETRIC_PROBE))
    metadata = {
        'experiment': 'exp10_quant_recall',
        'design': (
            'Planted associative recall through the two recurrent memory '
            'routes of the Exp 5 eight-group matrix (KDA / Mamba-2) with '
            'the state fake-quantized after every write; recall-cost curve '
            '= nearest-value decode accuracy vs state bytes (payload + '
            'group metadata). Crosses the matrix memory-route axis with a '
            'state-bits axis; seeds are aggregated to mean, sample std, '
            '95% t-CI, min/max. Nearest-value decisions use fp64 distances '
            'and report a relative decision margin; iso_budget_frontier '
            'selects the best worst-seed recall at each measured byte cap '
            'so routes can be compared at matched storage cost.'),
        'probe_geometry': {
            'n_keys': n_keys,
            'value_dim': value_dim,
            'context_T': n_keys,
            'keys': 'rows of a random orthogonal matrix (exact-recall best '
                    'case for both recurrences)',
            'values': 'i.i.d. standard normal, nearest-value decode',
            'kda': 'q=k=keys, g=0, beta=1, T=1 decode steps with '
                   'initial_state; state quantized after every write',
            'mamba2': 'd_state=n_keys, headdim=value_dim, ngroups=nheads=1, '
                      'dt=1, A=0 (no decay), D=0; state quantized after '
                      'every write',
            'quantization': 'flattened contiguous state buffer; groups of '
                            'group_size adjacent elements share one scale '
                            '(and zero point for asymmetric)',
        },
        'cost_currency': (
            'state_bytes = state_elements * bits_per_element(spec); '
            'bits_per_element charges payload bits plus '
            '(scale_bits + zero_bits)/group_size metadata bits. '
            'compression_vs_fp16 = 16 / bits_per_element.'),
        'decision_currency': (
            'Nearest-value distances are computed in fp64 even when the '
            'recurrence runs in fp32. The relative margin is '
            '(d_wrong - d_correct) / (d_wrong + d_correct); a decision '
            f'with |margin| <= {DECISION_TIE_TOL} is counted as an '
            'ambiguous numerical tie and is not evidence of a stable '
            'recall.'),
        'iso_budget_frontier': (
            'For each byte cap, the measured cell (including the fp16 '
            'reference) with the highest worst-seed recall among cells '
            'whose state_bytes <= cap, ties broken by mean recall then '
            'cheaper storage. Caps default to the measured state_bytes '
            'values, so no interpolated cost is used.'),
        'iso_budget_caps_bytes': (
            list(budgets) if budgets is not None else None),
        'specs': specs_provenance,
        'caveats': [
            'Single-operator probe, no training and no model weights: it '
            'measures state-quantization damage to stored associations, '
            'not end-task quality.',
            'Orthonormal keys are the best case for exact recall; real '
            'keys are correlated, so low-bit cells are optimistic.',
            'Only the recurrent state is quantized; activations, weights, '
            'and the CSA/HCA compressed-KV caches are out of scope (use '
            'kv_budget.op_read_bytes / layout_read_bytes for those '
            'components).',
            'Three seeds keep the probe fast; "exact" is the conservative '
            'min-over-seeds == 1.0 criterion, and 2-bit cells carry the '
            'visible seed variance (sample std, 95% t-CI and min..max '
            'reported). With only three seeds the CI is wide and is a '
            'descriptive uncertainty statement, not a population guarantee.',
            'The iso-budget frontier optimizes the observed worst-seed '
            'recall at the measured caps; it is a selection over a finite '
            'grid, not a proof that a cell remains exact on unseen keys, '
            'and the caps are not hardware-enforced quotas.',
        ],
        'provenance': capture_provenance(),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, rows, kind='exp10_quant_recall',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')
    append_run_manifest(make_run_manifest_entry(
        experiment='exp10_quant_recall',
        seeds=list(seeds),
        provenance=metadata['provenance'],
        hyperparameters={
            'routes': list(ROUTES),
            'bits': list(bits_list),
            'group_sizes': list(group_sizes),
            'probe_geometry': metadata['probe_geometry'],
            'specs': specs_provenance,
            'iso_budget_caps_bytes': (
                list(budgets) if budgets is not None else 'measured'),
            'decision_tie_tolerance': DECISION_TIE_TOL,
        },
        data={
            'kind': 'synthetic_planted_associative_recall',
            'n_keys': n_keys,
            'value_dim': value_dim,
            'context_T': n_keys,
        },
        artifacts={'results': RESULTS_PATH},
        results={
            'n_rows': len(rows),
            'n_seeds': len(seeds),
            'routes': list(ROUTES),
        },
    ), logger=logger)
    return 0


if __name__ == '__main__':
    sys.exit(main())
