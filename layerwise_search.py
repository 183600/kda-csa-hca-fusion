"""Experiment 12 — greedy layerwise architecture search (GLAS).

Motivation
----------
The repository ships three hand-designed layerwise layouts (the 3:1:1
ratio layout, Exp 5's ``position`` preset, Exp 9's interleaving presets) and
measures them, but it has no SEARCH over the layerwise placement of the
three operators. The design question behind both reference papers is
layerwise: Kimi Linear pretrains a layerwise hybrid of KDA and MLA, and
DeepSeek-V4 interleaves CSA / HCA at fixed depth. Which layer gets KDA,
which gets CSA, which gets HCA — and how sparse should the CSA top-k
selection be — is currently chosen by hand.

This module makes that choice searchable under the repository's TWO
resource currencies, both of which already have exact accounting:

* **decode-state bytes** (Exp 3 / Exp 6; ``cache_accounting``) — the
  per-request state retained after a prefix of ``T`` tokens. The memory
  route (KDA) is O(1) in ``T``; CSA / HCA grow with the compressed-block
  count (about ``T/m`` and ``T/m2`` rows per layer), which is exactly the
  growing-vs-constant certificate of ``docs/decoding-state-certificate.md``.
* **KV read elements per decode step** (Exp 9b; ``kv_budget``) — the
  marginal budget of the first token decoded after a ``T``-token prefill,
  which is where the CSA top-k selection cost lives (the indexer scans
  EVERY visible compressed block before selecting ``topk``).

The search objective is a trained quality probe (default: the
distance-controlled long-range MQAR task, single (key, value) pair, cue at
the last position — the same probe Exp 9 uses for its block-size grid).
Candidates are evaluated under FIXED seeds, so differences between
candidates are paired by seed, mirroring Exp 5/8's validity policy.

Search operator (why greedy, why layerwise)
-------------------------------------------
The exact decode-state cost of a layout is **order-invariant** (state is a
per-layer sum), and so is the analytic read budget. Only the trained probe
can distinguish ``kda,kda,kda,csa,hca`` from ``csa,kda,kda,kda,hca``.
The search therefore operates on the actual ``HybridConfig.layout`` tuple —
the same object the stack builds from — with three deterministic,
single-edit neighborhoods:

1. **substitution** — change one slot's kind (``2 * depth`` candidates);
2. **adjacent swap** — exchange two neighboring slots of different kinds
   (the layerwise-order edit; cost-neutral by construction, so any score
   difference is an empirical placement effect);
3. **top-k move** — change ``csa_topk`` over a fixed grid (state-neutral,
   read-budget-relevant; only generated when the layout has a CSA layer).

Greedy best-improvement with strict ``min_delta`` accepts one candidate per
round and stops when no neighbor improves, so the trace is reproducible
from ``(start, objective, seeds, budget)`` alone. Every evaluated candidate
is retained in the result (not just the accepted path), which makes the
quality-vs-cost **Pareto frontier** of the explored neighborhood available
for analysis even when the search rejects a move.

Honesty / scope
---------------
* The probe model is the frozen ``run_quality.SMALL_MODEL_SPEC`` geometry
  (``d_model=32``): the point is the search MACHINERY and the reproducible
  cost bookkeeping, not a state-of-the-art architecture. Any conclusion
  drawn from a short search run is exploratory.
* The analytic cost is an element-count model converted at ``dtype_bytes``
  (default 4) per element; it does not model per-component mixed
  precision, allocator fragmentation, or the batch dimension (state is
  reported per sequence). See ``cache_accounting`` for the authoritative
  component taxonomy.
* ``state_constant`` compares allocated bytes at ``T`` and ``2T``: a
  heuristic two-point version of Exp 6's certificate (a memory-only layout
  is exactly constant; any layout with a CSA/HCA layer grows).
* Only the reference operators are searched. ``memory_type`` stays ``kda``
  (the probe geometry of ``recall_reach.make_grid_cfg``); extending the
  space to Mamba-2 / RWKV-7 is a matter of providing another
  ``candidate_config``/objective pair.

Output
------
``results/exp12_layerwise_search.json`` via the CLI::

    python3 layerwise_search.py --quick
    python3 layerwise_search.py --depth 5 --steps 150 --seeds 42,43 \\
        --budget-mult 1.25 --json results/exp12_layerwise_search.json

Tests: ``python3 test_layerwise_search.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_accounting import (  # noqa: E402
    COMPONENT_SPECS,
    PARAMETER,
    analytic_cache_components,
)
from kaggle_setup import (  # noqa: E402
    capture_provenance,
    configure_torch_for_device,
    parse_int_env,
    sanitize_for_json,
    write_results_json,
)
from kv_budget import (  # noqa: E402
    instant_layout_read_elements,
    layout_read_elements,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join('results', 'exp12_layerwise_search.json')

#: Layer kinds the search can place. Mirrors the three operators of the
#: repository title; the MQAR probe builder (``recall_reach.make_grid_cfg``)
#: pins ``memory_type='kda'``.
SEARCH_KINDS: tuple[str, ...] = ('kda', 'csa', 'hca')

#: Default top-k grid for the state-neutral, read-budget-relevant move.
DEFAULT_TOPK_GRID: tuple[int, ...] = (0, 1, 2, 4, 8)

#: The reference 3:1:1 layout the budget defaults are derived from.
DEFAULT_START: tuple[str, ...] = ('kda', 'kda', 'kda', 'csa', 'hca')


# ===========================================================================
# 1. Candidate layouts
# ===========================================================================

@dataclass(frozen=True)
class Candidate:
    """One search point: a layerwise layout + CSA top-k.

    ``layout`` is an exact, resolved per-layer kind tuple (no ``'memory'``
    tokens) — the same object ``HybridConfig.layout`` accepts. ``csa_topk``
    is carried alongside the layout because it is state-neutral and
    otherwise would force a separate search axis; it is ignored (but still
    recorded) when the layout contains no CSA layer.
    """

    layout: tuple[str, ...]
    csa_topk: int = 2

    def __post_init__(self) -> None:
        layout = tuple(self.layout)
        object.__setattr__(self, 'layout', layout)
        if not layout:
            raise ValueError('Candidate.layout must be non-empty.')
        for i, kind in enumerate(layout):
            if kind not in SEARCH_KINDS:
                raise ValueError(
                    f'Candidate.layout[{i}]={kind!r} must be one of '
                    f'{list(SEARCH_KINDS)}.')
        if not isinstance(self.csa_topk, int) or isinstance(self.csa_topk, bool):
            raise TypeError(
                f'Candidate.csa_topk={self.csa_topk!r} must be an int.')
        if self.csa_topk < 0:
            raise ValueError(
                f'Candidate.csa_topk={self.csa_topk} must be >= 0.')

    @property
    def counts(self) -> dict[str, int]:
        """Per-kind layer counts (``{'kda': .., 'csa': .., 'hca': ..}``)."""
        return {kind: self.layout.count(kind) for kind in SEARCH_KINDS}

    @property
    def label(self) -> str:
        """Stable, human-readable identity used for sorting and dedup."""
        return f"{','.join(self.layout)}/topk={self.csa_topk}"


def candidate_from_ratio(ratio: Sequence[int], *,
                         kinds: Sequence[str] = SEARCH_KINDS,
                         csa_topk: int = 2) -> Candidate:
    """Canonical layout from per-kind counts (kinds in ``kinds`` order)."""
    if len(ratio) != len(kinds):
        raise ValueError(
            f'candidate_from_ratio: got {len(ratio)} counts for '
            f'{len(kinds)} kinds {list(kinds)}.')
    layout: list[str] = []
    for kind, n in zip(kinds, ratio):
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(
                f'candidate_from_ratio: count for {kind!r} must be a '
                f'non-negative int, got {n!r}.')
        layout.extend([kind] * n)
    return Candidate(tuple(layout), csa_topk=csa_topk)


def layout_neighbors(candidate: Candidate, *,
                     kinds: Sequence[str] = SEARCH_KINDS,
                     topk_grid: Sequence[int] = DEFAULT_TOPK_GRID,
                     include_swaps: bool = True) -> list[Candidate]:
    """Deterministic single-edit neighborhood of ``candidate``.

    Order is fixed (and therefore the search is reproducible):

    1. substitutions, slot index ascending, new kind in ``kinds`` order;
    2. adjacent swaps, slot index ascending (only for unequal neighbors);
    3. top-k moves, ``topk_grid`` ascending (only when the layout has CSA).

    Duplicates (e.g. two swaps producing the same layout) are removed while
    preserving the first occurrence.
    """
    if not isinstance(candidate, Candidate):
        raise TypeError(
            f'layout_neighbors: candidate must be a Candidate, got '
            f'{type(candidate).__name__}.')
    if not isinstance(topk_grid, Sequence) or isinstance(
            topk_grid, (str, bytes)):
        raise TypeError('layout_neighbors: topk_grid must be a sequence.')
    out: list[Candidate] = []
    seen: set[Candidate] = set()

    def push(cand: Candidate) -> None:
        if cand != candidate and cand not in seen:
            seen.add(cand)
            out.append(cand)

    for i in range(len(candidate.layout)):
        for kind in kinds:
            if kind != candidate.layout[i]:
                layout = list(candidate.layout)
                layout[i] = kind
                push(Candidate(tuple(layout), csa_topk=candidate.csa_topk))
    if include_swaps:
        for i in range(len(candidate.layout) - 1):
            if candidate.layout[i] != candidate.layout[i + 1]:
                layout = list(candidate.layout)
                layout[i], layout[i + 1] = layout[i + 1], layout[i]
                push(Candidate(tuple(layout), csa_topk=candidate.csa_topk))
    if 'csa' in candidate.layout:
        for k in topk_grid:
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                raise ValueError(
                    f'layout_neighbors: topk_grid entries must be '
                    f'non-negative ints, got {k!r}.')
            push(Candidate(candidate.layout, csa_topk=k))
    return out


# ===========================================================================
# 2. Exact resource accounting
# ===========================================================================

def _validate_T(T: int, fn_name: str) -> None:
    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        raise ValueError(f'{fn_name}: T must be a non-negative int, got {T!r}')


def candidate_config(candidate: Candidate, *, m: int, m2: int,
                     d_model: int = 32):
    """Build the probe ``HybridConfig`` for one candidate.

    Delegates to ``recall_reach.make_grid_cfg`` so the searched candidates
    and the Exp 9 MQAR probe share ONE geometry (sub-layer widths,
    compression factors, sliding windows, config validation). The import is
    local: ``recall_reach`` pulls in the training stack, which the pure
    cost/neighborhood tests do not need.
    """
    from recall_reach import make_grid_cfg

    if not isinstance(candidate, Candidate):
        raise TypeError(
            f'candidate_config: candidate must be a Candidate, got '
            f'{type(candidate).__name__}.')
    return make_grid_cfg(int(m), int(m2), candidate.layout,
                         d_model=int(d_model),
                         csa_topk=int(candidate.csa_topk))


def _state_bytes_at(cfg, T: int, dtype_bytes: int
                    ) -> tuple[dict, dict, int, int]:
    """Per-sequence bytes at ``T``, split into state and parameters.

    Returns ``(allocated_breakdown, valid_breakdown, parameter_allocated,
    parameter_valid)``. Components in the ``PARAMETER`` pool (the attention
    sinks) are parameters shared by every request, so they are reported
    separately and excluded from the per-request state totals — the
    decode-state budget is about what one request retains, matching
    Exp 6's certificate semantics.
    """
    comps = analytic_cache_components(cfg, T)
    alloc: dict[str, int] = {}
    valid: dict[str, int] = {}
    param_alloc = param_valid = 0
    for name, comp in comps.items():
        nbytes_alloc = int(comp['elements_allocated']) * dtype_bytes
        nbytes_valid = int(comp['elements_valid']) * dtype_bytes
        if COMPONENT_SPECS[name][1] == PARAMETER:
            param_alloc += nbytes_alloc
            param_valid += nbytes_valid
            continue
        alloc[name] = nbytes_alloc
        valid[name] = nbytes_valid
    return alloc, valid, param_alloc, param_valid


def _bytes_by_op(breakdown: dict[str, int]) -> dict[str, int]:
    """Group a component breakdown by ``COMPONENT_SPECS`` op."""
    by_op: dict[str, int] = {}
    for name, nbytes in breakdown.items():
        op = COMPONENT_SPECS[name][0]
        by_op[op] = by_op.get(op, 0) + int(nbytes)
    return {op: by_op[op] for op in sorted(by_op)}


def budget_spec(cfg) -> dict:
    """``kv_budget`` geometry dict for a ``HybridConfig``.

    Mirrors ``run_budget_frontier._spec_from_hybrid_cfg`` (the committed
    Exp 9b builder) so the search's read budget and the budget-frontier
    experiment cannot disagree on the per-operator geometry.
    """
    return {
        'softmax': {'n_kv_heads': cfg.n_heads_qk, 'd_k': cfg.head_dim_k,
                    'd_v': cfg.head_dim_v},
        'kda': {'hv': cfg.n_heads_v, 'd_k': cfg.head_dim_k,
                'd_v': cfg.head_dim_v},
        'csa': {'m': cfg.csa_m, 'topk': cfg.csa_topk, 'c': cfg.csa_c,
                'c_i': cfg.csa_cI, 'n_idx_heads': cfg.csa_nIh,
                'sliding_window': cfg.csa_sliding_window},
        'hca': {'m2': cfg.hca_m2, 'c': cfg.hca_c,
                'sliding_window': cfg.hca_sliding_window},
    }


def read_budget(cfg, T: int) -> dict:
    """Exact KV-side read elements per query at context ``T``.

    Two unified views from ``kv_budget`` (see its module docstring for the
    currency contract): the prefill average over positions ``0..T-1`` and
    the MARGINAL budget of the first token decoded after the ``T``-token
    prefill (the decode step where the CSA indexer scans all visible
    compressed blocks before selecting top-k).
    """
    _validate_T(T, 'read_budget')
    spec = budget_spec(cfg)
    counts = (cfg.layout.count('kda') if cfg.layout is not None else cfg.n_kda,
              cfg.layout.count('csa') if cfg.layout is not None else cfg.n_csa,
              cfg.layout.count('hca') if cfg.layout is not None else cfg.n_hca)
    avg = layout_read_elements(T, counts, spec)
    marginal = instant_layout_read_elements(counts, T, spec)
    return {
        'prefill_avg_read_elements': float(avg['total_read_elements']),
        'decode_marginal_read_elements': float(
            marginal['total_read_elements']),
        'decode_marginal_by_op': {
            'kda': float(marginal['kda_read_elements']),
            'csa': float(marginal['csa_read_elements']),
            'hca': float(marginal['hca_read_elements']),
        },
    }


def cost_report(candidate: Candidate, *, m: int, m2: int, T: int,
                d_model: int = 32, dtype_bytes: int = 4,
                check_constant: bool = True) -> dict:
    """Full resource report for one candidate at context ``T``.

    Returns plain floats/ints (JSON-safe). ``state_bytes_allocated`` /
    ``state_bytes_valid`` are per-sequence sums over Exp 3's component
    taxonomy; ``state_bytes_by_op`` groups them by operator. When
    ``check_constant`` and ``T > 0``, ``state_constant`` compares the
    allocated total at ``T`` and ``2T`` (a two-point certificate: exactly
    True for a memory-only layout, False for any layout with CSA/HCA).
    """
    if not isinstance(candidate, Candidate):
        raise TypeError(
            f'cost_report: candidate must be a Candidate, got '
            f'{type(candidate).__name__}.')
    _validate_T(T, 'cost_report')
    if not isinstance(dtype_bytes, int) or dtype_bytes < 1:
        raise ValueError(
            f'cost_report: dtype_bytes must be a positive int, got '
            f'{dtype_bytes!r}.')
    cfg = candidate_config(candidate, m=m, m2=m2, d_model=d_model)
    alloc, valid, param_alloc, param_valid = _state_bytes_at(
        cfg, T, dtype_bytes)
    report = {
        'layout': list(candidate.layout),
        'label': candidate.label,
        'csa_topk': int(candidate.csa_topk),
        'counts': candidate.counts,
        'T': int(T),
        'state_bytes_allocated': int(sum(alloc.values())),
        'state_bytes_valid': int(sum(valid.values())),
        'parameter_bytes': int(param_alloc),
        'parameter_bytes_valid': int(param_valid),
        'state_bytes_by_op': _bytes_by_op(alloc),
        'state_bytes_breakdown': alloc,
        **read_budget(cfg, T),
    }
    if check_constant and T > 0:
        alloc2, _, _, _ = _state_bytes_at(cfg, 2 * T, dtype_bytes)
        total2 = int(sum(alloc2.values()))
        report['state_bytes_allocated_at_2T'] = total2
        report['state_constant'] = (total2 == report['state_bytes_allocated'])
    else:
        report['state_bytes_allocated_at_2T'] = report[
            'state_bytes_allocated']
        report['state_constant'] = None
    return report


# ===========================================================================
# 3. Budgeted search problem
# ===========================================================================

@dataclass
class SearchProblem:
    """A candidate evaluator + exact cost model + optional resource caps.

    ``evaluate`` returns ``{'score': float | None, 'metrics': {...}}``
    (higher score is better). Exceptions raised by ``evaluate`` are caught
    and recorded as ``error`` (score ``None``): one broken candidate must
    not kill a multi-hour search. ``cost`` returns the :func:`cost_report`
    dict. ``feasible`` reports the budget verdict only — an errored but
    in-budget candidate is still recorded, just never selected.
    """

    evaluate: Callable[[Candidate], dict]
    cost: Callable[[Candidate], dict]
    max_state_bytes: int | None = None
    max_decode_read_elements: float | None = None
    metadata: dict = field(default_factory=dict)

    def record(self, candidate: Candidate) -> dict:
        cost = self.cost(candidate)
        try:
            ev = self.evaluate(candidate)
            if not isinstance(ev, dict):
                raise TypeError(
                    f'evaluate() must return a dict, got '
                    f'{type(ev).__name__}.')
            score = ev.get('score')
            metrics = ev.get('metrics', {}) or {}
            error = None
        except Exception as exc:  # noqa: BLE001 — search robustness
            score, metrics = None, {}
            error = f'{type(exc).__name__}: {exc}'
        record = {
            'layout': list(candidate.layout),
            'label': candidate.label,
            'csa_topk': int(candidate.csa_topk),
            'score': None if score is None else float(score),
            'metrics': dict(metrics),
            'error': error,
            **cost,
        }
        record['feasible'] = self.feasible(record)
        return record

    def feasible(self, record: dict) -> bool:
        if self.max_state_bytes is not None and \
                record['state_bytes_allocated'] > self.max_state_bytes:
            return False
        if self.max_decode_read_elements is not None and \
                record['decode_marginal_read_elements'] > \
                self.max_decode_read_elements:
            return False
        return True


def pareto_frontier(records: Iterable[dict], *,
                    score_key: str = 'score',
                    cost_keys: Sequence[str] = ('state_bytes_allocated',
                                                'decode_marginal_read_elements')
                    ) -> list[dict]:
    """Non-dominated records (max score, min every cost), deterministic.

    A record is dominated when another record has ``score >= `` and every
    cost ``<=`` with at least one strict inequality. Records with
    ``score is None`` are excluded. The frontier is sorted by the cost
    tuple then by label, so the output is stable across runs.
    """
    pool = [r for r in records if r.get(score_key) is not None]
    for key in cost_keys:
        pool = [r for r in pool if r.get(key) is not None]
    frontier: list[dict] = []
    for r in pool:
        dominated = False
        for other in pool:
            if other is r:
                continue
            ge_score = other[score_key] >= r[score_key]
            le_cost = all(other[k] <= r[k] for k in cost_keys)
            strictly = (other[score_key] > r[score_key]
                        or any(other[k] < r[k] for k in cost_keys))
            if ge_score and le_cost and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(r)
    frontier.sort(key=lambda r: (
        tuple(float(r[k]) for k in cost_keys), -float(r[score_key]),
        str(r.get('label', ''))))
    return frontier


def greedy_search(problem: SearchProblem, start: Candidate, *,
                  max_rounds: int = 3,
                  topk_grid: Sequence[int] = DEFAULT_TOPK_GRID,
                  include_swaps: bool = True,
                  min_delta: float = 0.0,
                  logger: logging.Logger | None = None) -> dict:
    """Greedy best-improvement layerwise search.

    Each round evaluates every not-yet-seen neighbor of the current point
    (deterministic order), keeps the best budget-feasible candidate with a
    non-``None`` score, and accepts it only when it beats the current score
    by more than ``min_delta``. Ties are broken by lower state bytes, then
    lower decode reads, then label (so the result is a pure function of the
    problem). Stops early when no neighbor improves. Every evaluated
    candidate is returned in ``records`` (with the frontier in ``pareto``),
    not just the accepted path.

    The reported ``best`` closes the quality plateau: among the feasible
    candidates within ``min_delta`` of the greedy endpoint's score, the
    cheapest one wins (so a state-neutral top-k reduction is recommended
    even though the strict-improvement rule would not move to it). The raw
    endpoint is kept as ``greedy_endpoint`` for the trace.
    """
    if not isinstance(problem, SearchProblem):
        raise TypeError(
            f'greedy_search: problem must be a SearchProblem, got '
            f'{type(problem).__name__}.')
    if not isinstance(start, Candidate):
        raise TypeError(
            f'greedy_search: start must be a Candidate, got '
            f'{type(start).__name__}.')
    if not isinstance(max_rounds, int) or isinstance(max_rounds, bool) or \
            max_rounds < 0:
        raise ValueError(
            f'greedy_search: max_rounds must be a non-negative int, got '
            f'{max_rounds!r}.')

    records: dict[Candidate, dict] = {}

    def get(cand: Candidate) -> dict:
        if cand not in records:
            records[cand] = problem.record(cand)
        return records[cand]

    current = start
    current_record = get(current)
    rounds: list[dict] = []
    for round_index in range(1, max_rounds + 1):
        neighbors = layout_neighbors(current, topk_grid=topk_grid,
                                     include_swaps=include_swaps)
        fresh = [(cand, get(cand)) for cand in neighbors]
        selectable = [(cand, rec) for cand, rec in fresh
                      if rec['feasible'] and rec['score'] is not None]
        if not selectable:
            rounds.append({
                'round': round_index,
                'from': current.label,
                'accepted': False,
                'reason': 'no feasible improving neighbor evaluated',
                'candidates': [rec['label'] for _, rec in fresh],
            })
            break
        best_cand, best_rec = min(selectable, key=lambda pair: (
            -pair[1]['score'], pair[1]['state_bytes_allocated'],
            pair[1]['decode_marginal_read_elements'], pair[1]['label']))
        current_score = current_record['score']
        improved = (current_score is None
                    or best_rec['score'] > current_score + float(min_delta))
        rounds.append({
            'round': round_index,
            'from': current.label,
            'from_score': current_score,
            'best_neighbor': best_rec['label'],
            'best_neighbor_score': best_rec['score'],
            'best_neighbor_state_bytes': best_rec['state_bytes_allocated'],
            'best_neighbor_decode_reads': best_rec[
                'decode_marginal_read_elements'],
            'accepted': bool(improved),
            'reason': ('improved' if improved
                       else 'no strict improvement over current'),
            'candidates': [rec['label'] for _, rec in fresh],
        })
        if logger is not None:
            logger.info(
                'round %d: %s -> %s (%s, %.4f -> %.4f)', round_index,
                current.label, best_rec['label'],
                'accept' if improved else 'stop',
                -1.0 if current_score is None else current_score,
                best_rec['score'])
        if not improved:
            break
        current, current_record = best_cand, best_rec

    # Plateau closure: the greedy endpoint may be quality-tied with a
    # resource-cheaper candidate that the strict-improvement rule refused
    # to move to (e.g. the state-neutral top-k move). Report the cheapest
    # feasible candidate within ``min_delta`` of the endpoint's score, so
    # the recommended configuration is never Pareto-dominated on its own
    # plateau. The raw endpoint stays available as ``greedy_endpoint``.
    endpoint = current_record
    plateau = [r for r in records.values()
               if r['feasible'] and r['score'] is not None
               and endpoint['score'] is not None
               and r['score'] >= endpoint['score'] - float(min_delta)]
    best = endpoint
    if plateau:
        best = min(plateau, key=lambda r: (
            r['state_bytes_allocated'], r['decode_marginal_read_elements'],
            r['label']))
    ordered = sorted(records.values(),
                     key=lambda r: (r['state_bytes_allocated'], r['label']))
    return {
        'start': get(start),
        'best': best,
        'greedy_endpoint': endpoint,
        'rounds': rounds,
        'records': ordered,
        'pareto': pareto_frontier(ordered),
    }


# ===========================================================================
# 4. MQAR objective (the trained probe)
# ===========================================================================

def make_mqar_objective(*, m: int, m2: int, seeds: Sequence[int] = (42,),
                        steps: int = 100, seq_len: int = 16,
                        d_model: int = 32, train_batch: int = 16,
                        eval_batches: int = 4, eval_batch: int = 64,
                        device='cpu', logger: logging.Logger | None = None,
                        ) -> Callable[[Candidate], dict]:
    """Return an ``evaluate(candidate)`` closure for the MQAR probe.

    Trains one probe per candidate per seed through Exp 9's
    ``recall_reach.train_recall_model`` (same data stream per seed for
    every candidate, so comparisons are paired) and returns the mean
    accuracy as the score. Per-seed accuracies, losses' final value, and
    the geometry are recorded in ``metrics`` for the trace.
    """
    from recall_reach import evaluate_recall, train_recall_model

    if not seeds:
        raise ValueError('make_mqar_objective: seeds must be non-empty.')

    def evaluate(candidate: Candidate) -> dict:
        accuracies: list[float | None] = []
        final_losses: list[float | None] = []
        for seed in seeds:
            model, embed, head, losses = train_recall_model(
                candidate.layout, int(m), int(m2), seed=int(seed),
                steps=int(steps), seq_len=int(seq_len), d_model=int(d_model),
                train_batch=int(train_batch), device=device)
            result = evaluate_recall(
                model, embed, head, seq_len=int(seq_len),
                n_batches=int(eval_batches), batch=int(eval_batch),
                seed=int(seed), device=device)
            accuracies.append(result['accuracy'])
            final_losses.append(losses[-1] if losses else None)
            if logger is not None:
                logger.info('  probe %s seed=%d acc=%s',
                            candidate.label, seed, result['accuracy'])
        finite = [a for a in accuracies if a is not None]
        score = sum(finite) / len(finite) if finite else None
        return {
            'score': score,
            'metrics': {
                'accuracy': score,
                'per_seed_accuracy': [None if a is None else float(a)
                                      for a in accuracies],
                'seeds': [int(s) for s in seeds],
                'final_loss': [None if v is None else float(v)
                               for v in final_losses],
                'steps': int(steps),
                'seq_len': int(seq_len),
                'm': int(m),
                'm2': int(m2),
                'probe': 'mqar-long-range-cue-last',
            },
        }

    return evaluate


# ===========================================================================
# 5. CLI
# ===========================================================================

def _parse_layout(raw: str, *, field_name: str = 'layout') -> tuple[str, ...]:
    kinds = tuple(part.strip().lower() for part in str(raw).split(',')
                  if part.strip())
    if not kinds:
        raise SystemExit(f'{field_name} must contain at least one kind.')
    for kind in kinds:
        if kind not in SEARCH_KINDS:
            raise SystemExit(
                f'{field_name} contains {kind!r}; allowed: '
                f'{list(SEARCH_KINDS)}.')
    return kinds


def _parse_int_tuple(raw: str, *, field_name: str,
                     min_value: int = 0) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(raw).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(f'{field_name} contains non-integer {part!r}.')
        if value < min_value:
            raise SystemExit(
                f'{field_name} entries must be >= {min_value}, got {value}.')
        out.append(value)
    if not out:
        raise SystemExit(f'{field_name} has no entries: {raw!r}.')
    return tuple(out)


def _fmt_bytes(n: int) -> str:
    for unit, scale in (('GiB', 1 << 30), ('MiB', 1 << 20), ('KiB', 1 << 10)):
        if n >= scale:
            return f'{n / scale:.2f} {unit}'
    return f'{n} B'


def _print_report(report: dict) -> None:
    cfg = report['config']
    print('=' * 100)
    print('Layerwise architecture search (Exp 12) — greedy best-improvement')
    print(f"  depth={cfg['depth']} start={cfg['start']} "
          f"m={cfg['m']} m2={cfg['m2']} T={cfg['budget_T']} "
          f"steps={cfg['steps']} seeds={cfg['seeds']}")
    if cfg['max_state_bytes'] is not None:
        print(f"  state budget: <= {cfg['max_state_bytes']} B "
              f"({_fmt_bytes(cfg['max_state_bytes'])})")
    else:
        print('  state budget: none (free exploration)')
    if cfg['max_decode_read_elements'] is not None:
        print(f"  decode read cap: <= {cfg['max_decode_read_elements']:.1f} "
              f"elements/token")
    print('-' * 100)
    print(f"{'layout':<28}{'topk':>5}{'state B':>12}{'const':>7}"
          f"{'reads/tok':>12}{'ok':>4}{'score':>9}")
    for rec in report['records']:
        score = rec['score']
        print(f"{','.join(rec['layout']):<28}{rec['csa_topk']:>5}"
              f"{rec['state_bytes_allocated']:>12}"
              f"{str(rec['state_constant']):>7}"
              f"{rec['decode_marginal_read_elements']:>12.1f}"
              f"{str(rec['feasible']):>5}"
              f"{('  n/a' if score is None else f'{score:>9.4f}')}")
    print('-' * 100)
    best = report['best']
    print(f"BEST: {best['label']}  score={best['score']}  "
          f"state={_fmt_bytes(best['state_bytes_allocated'])}  "
          f"reads/token={best['decode_marginal_read_elements']:.1f}")
    print(f"PARETO FRONT ({len(report['pareto'])} points):")
    for rec in report['pareto']:
        print(f"  {rec['label']:<30} score={rec['score']:.4f}  "
              f"state={_fmt_bytes(rec['state_bytes_allocated'])}  "
              f"reads={rec['decode_marginal_read_elements']:.1f}")
    print('=' * 100)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Greedy layerwise architecture search over KDA/CSA/HCA '
                    'placements under exact decode-state and read budgets '
                    '(Exp 12).')
    parser.add_argument('--json', default=None,
                        help=f'output path (default: {RESULTS_PATH})')
    parser.add_argument('--start', default=','.join(DEFAULT_START),
                        help='starting layerwise layout (comma-separated '
                             f'kinds; default: {",".join(DEFAULT_START)})')
    parser.add_argument('--depth', type=int, default=None,
                        help='total layers (default: len(--start); when set '
                             'with --start, the layout is truncated/cycled)')
    parser.add_argument('--m', type=int, default=4, help='CSA block size')
    parser.add_argument('--m2', type=int, default=8, help='HCA block size')
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--steps', type=int, default=None,
                        help='probe training steps per candidate')
    parser.add_argument('--seeds', default=None,
                        help='comma-separated probe seeds (default: 42)')
    parser.add_argument('--train-batch', type=int, default=16)
    parser.add_argument('--eval-batches', type=int, default=4)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--budget-T', type=int, default=4096,
                        help='context length for the decode-state budget')
    parser.add_argument('--budget-mult', type=float, default=None,
                        help='cap = ceil(mult * start state bytes); '
                             'default: no cap')
    parser.add_argument('--max-state-bytes', type=int, default=None,
                        help='explicit per-request decode-state cap (bytes)')
    parser.add_argument('--max-decode-read-elements', type=float, default=None,
                        help='optional cap on the marginal per-token reads')
    parser.add_argument('--topk-grid', default=','.join(
                            str(k) for k in DEFAULT_TOPK_GRID))
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.0,
                        help='minimum score improvement to accept a move')
    parser.add_argument('--no-swaps', action='store_true',
                        help='disable adjacent-swap (ordering) moves')
    parser.add_argument('--device', default=None)
    parser.add_argument('--quick', action='store_true',
                        help='small search: depth 3, 60 steps, 1 round')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    info = configure_torch_for_device()
    device = args.device or str(info.device)

    start_layout = _parse_layout(args.start, field_name='--start')
    if args.depth is not None:
        if args.depth < 1:
            raise SystemExit('--depth must be >= 1.')
        # Cycle the parsed pattern (or truncate it) to the requested depth.
        start_layout = tuple(start_layout[i % len(start_layout)]
                             for i in range(args.depth))
    topk_grid = _parse_int_tuple(args.topk_grid, field_name='--topk-grid')
    steps = args.steps if args.steps is not None else parse_int_env(
        'SEARCH_STEPS', 100, min_value=1, logger=logger)
    if args.seeds is not None:
        seeds = _parse_int_tuple(args.seeds, field_name='--seeds',
                                 min_value=0)
    else:
        seeds = (42,)
    start_topk = 2
    if 'csa' not in start_layout:
        start_topk = 0
    start = Candidate(start_layout, csa_topk=start_topk)

    if args.quick:
        # A 3-layer start that exercises every kind and every move type.
        if args.start == ','.join(DEFAULT_START):
            start_layout = ('kda', 'csa', 'hca')
        else:
            start_layout = tuple(start_layout[i % len(start_layout)]
                                 for i in range(3))
        start = Candidate(start_layout, csa_topk=start_topk)
        steps = min(steps, 60)
        args.max_rounds = min(args.max_rounds, 1)

    def cost_fn(candidate):
        return cost_report(candidate, m=args.m, m2=args.m2,
                           T=args.budget_T, d_model=args.d_model)

    reference_bytes = cost_fn(start)['state_bytes_allocated']
    max_state_bytes = args.max_state_bytes
    if max_state_bytes is None and args.budget_mult is not None:
        if args.budget_mult <= 0:
            raise SystemExit('--budget-mult must be > 0.')
        max_state_bytes = int(math.ceil(args.budget_mult * reference_bytes))

    evaluate = make_mqar_objective(
        m=args.m, m2=args.m2, seeds=seeds, steps=steps,
        seq_len=args.seq_len, d_model=args.d_model,
        train_batch=args.train_batch, eval_batches=args.eval_batches,
        eval_batch=args.eval_batch, device=device, logger=logger)
    problem = SearchProblem(
        evaluate=evaluate, cost=cost_fn,
        max_state_bytes=max_state_bytes,
        max_decode_read_elements=args.max_decode_read_elements,
        metadata={'probe': 'mqar-long-range-cue-last'})

    result = greedy_search(
        problem, start, max_rounds=args.max_rounds, topk_grid=topk_grid,
        include_swaps=not args.no_swaps, min_delta=args.min_delta,
        logger=logger)

    report = {
        'kind': 'exp12_layerwise_search',
        'method': 'greedy best-improvement over single-edit layout '
                  'neighbors (substitution / adjacent swap / top-k), '
                  'paired-seed MQAR probe, exact Exp 3+9b cost model',
        'config': {
            'start': start.label,
            'depth': len(start.layout),
            'm': int(args.m),
            'm2': int(args.m2),
            'd_model': int(args.d_model),
            'seq_len': int(args.seq_len),
            'steps': int(steps),
            'seeds': [int(s) for s in seeds],
            'train_batch': int(args.train_batch),
            'eval_batches': int(args.eval_batches),
            'eval_batch': int(args.eval_batch),
            'topk_grid': [int(k) for k in topk_grid],
            'include_swaps': not args.no_swaps,
            'max_rounds': int(args.max_rounds),
            'min_delta': float(args.min_delta),
            'budget_T': int(args.budget_T),
            'dtype_bytes': 4,
            'reference_state_bytes': int(reference_bytes),
            'max_state_bytes': (None if max_state_bytes is None
                                else int(max_state_bytes)),
            'max_decode_read_elements': args.max_decode_read_elements,
        },
        'start': result['start'],
        'best': result['best'],
        'rounds': result['rounds'],
        'records': result['records'],
        'pareto': result['pareto'],
        'provenance': capture_provenance(),
    }
    payload = sanitize_for_json(report)
    json.dumps(payload, allow_nan=False)
    if not args.quiet:
        _print_report(payload)
    out_path = args.json or RESULTS_PATH
    write_results_json(payload, out_path, logger=logger)
    print(f'[layerwise-search] wrote {out_path} '
          f'({len(result["records"])} candidates evaluated)')
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    sys.exit(main())
