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

Statistical design
------------------
Every candidate is evaluated on the SAME seed list (default 3 seeds: 42,
43, 44; ``SEARCH_SEEDS`` overrides the count), so candidate-vs-candidate
differences are seed-paired. The objective reports mean +/- sample std /
95% CI, and the greedy acceptance rule additionally requires an
improvement to be *reproducible across seeds*: when both records carry
per-seed accuracies, a move is accepted only if the seed-paired one-sided
t-test on the differences clears alpha=0.05 (a unanimous zero-variance
sweep needs ``n >= 6`` for the exact sign-flip bound to clear it). The
raw score rule stays as ``improvement`` in each round record; the final
``best_vs_start`` block reports the paired test of the recommended
candidate against the start. With a single seed the paired test is
undefined and the search falls back to the raw rule — the report flags
that as exploratory.

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
* ``state_constant`` is a component-aware, context-ratio certificate:
  allocated bytes are swept at ``T, 2T, 4T`` and the layout is constant
  only when *every component* is flat. A two-point total-only check is
  unsound: HCA at ``T=14, m2=4`` has 1536 B at both 14 and 28 (compressed
  KV doubles while the partial tail empties), so component cancellation
  can mask unbounded growth. The report keeps the legacy two-point flag
  as ``state_constant_two_point`` for audit and adds the full certificate.
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
    append_run_manifest,
    capture_provenance,
    configure_torch_for_device,
    make_run_manifest_entry,
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


#: Context-length ratios for the state certificate's ratio sweep. The
#: sweep must contain more than the two points ``{1, 2}``: a total-only
#: comparison can be flat at two lengths while individual components
#: (compressed blocks growing, partial tail emptying) cancel exactly.
_STATE_CERTIFICATE_RATIOS = (1, 2, 4)


def _loglog_slope(xs: Sequence[int], ys: Sequence[int]):
    """Least-squares slope of ``log(y) ~ log(x)`` (``None`` if undefined).

    Used as the certificate's growth exponent: 0 for constant state,
    between 0 and 1 for sub-linear growth, ~1 for linear growth. Only
    positive points enter the fit; fewer than two points or a degenerate
    x-range yields ``None`` rather than a fabricated number.
    """
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)
           if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    den = sum((x - mx) ** 2 for x in lx)
    if den <= 0.0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den
    return 0.0 if abs(slope) < 1e-12 else slope


def cost_report(candidate: Candidate, *, m: int, m2: int, T: int,
                d_model: int = 32, dtype_bytes: int = 4,
                check_constant: bool = True) -> dict:
    """Full resource report for one candidate at context ``T``.

    Returns plain floats/ints (JSON-safe). ``state_bytes_allocated`` /
    ``state_bytes_valid`` are per-sequence sums over Exp 3's component
    taxonomy; ``state_bytes_by_op`` groups them by operator.

    When ``check_constant`` and ``T > 0``, the report carries a
    **component-aware state certificate** over a context-ratio sweep
    ``T, 2T, 4T``:

    * ``state_constant`` is True only when every component's allocated
      bytes are identical at all swept lengths (hence the total is flat
      for the right reason, not because components cancel);
    * ``state_growth_certificate`` records the per-length totals, the
      per-component series, the growing components, the log-log growth
      exponent, and ``total_flat_but_components_grow`` (the cancellation
      case a total-only check would misclassify);
    * ``state_constant_two_point`` preserves the old ``T`` vs ``2T``
      total-only verdict for audit; it is NOT the certificate.
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
        lengths = [T * r for r in _STATE_CERTIFICATE_RATIOS]
        snapshots = [_state_bytes_at(cfg, length, dtype_bytes)[0]
                     for length in lengths]
        names = sorted({name for snap in snapshots for name in snap})
        component_bytes = {
            name: [int(snap.get(name, 0)) for snap in snapshots]
            for name in names
        }
        totals = [int(sum(snap.values())) for snap in snapshots]
        growing = sorted(
            name for name, series in component_bytes.items()
            if len(set(series)) > 1)
        total_flat = len(set(totals)) == 1
        report['state_bytes_allocated_at_2T'] = totals[1]
        report['state_constant_two_point'] = (
            totals[1] == totals[0])
        report['state_constant'] = bool(total_flat and not growing)
        report['state_growth_certificate'] = {
            'lengths': lengths,
            'total_bytes': totals,
            'component_bytes': component_bytes,
            'growing_components': growing,
            'component_growth_exponents': {
                name: _loglog_slope(lengths, component_bytes[name])
                for name in growing
            },
            'total_flat': total_flat,
            'total_flat_but_components_grow': bool(total_flat and growing),
            'growth_exponent': _loglog_slope(lengths, totals),
            'max_over_min': (
                max(totals) / min(totals) if min(totals) > 0 else None),
        }
    else:
        report['state_bytes_allocated_at_2T'] = report[
            'state_bytes_allocated']
        report['state_constant_two_point'] = None
        report['state_constant'] = None
        report['state_growth_certificate'] = None
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


def _paired_seed_diffs(rec_a: dict, rec_b: dict):
    """Common-seed paired accuracy differences ``a - b`` from metrics.

    Returns ``(seeds, diffs)`` when BOTH records carry aligned
    ``metrics['seeds']`` / ``metrics['per_seed_accuracy']`` lists with at
    least one common seed, else ``None``. The objective evaluates every
    candidate on the same seed list, so the common-seed intersection is
    the paired sample.
    """
    ma = rec_a.get('metrics') or {}
    mb = rec_b.get('metrics') or {}
    seeds_a = ma.get('seeds')
    seeds_b = mb.get('seeds')
    acc_a = ma.get('per_seed_accuracy')
    acc_b = mb.get('per_seed_accuracy')
    if not (isinstance(seeds_a, list) and isinstance(seeds_b, list)
            and isinstance(acc_a, list) and isinstance(acc_b, list)):
        return None
    if len(seeds_a) != len(acc_a) or len(seeds_b) != len(acc_b):
        return None
    map_a = {int(s): a for s, a in zip(seeds_a, acc_a) if a is not None}
    map_b = {int(s): a for s, a in zip(seeds_b, acc_b) if a is not None}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return None
    return common, [float(map_a[s]) - float(map_b[s]) for s in common]


def _paired_support(diffs: list[float], *, alpha: float = 0.05):
    """Seed-paired one-sided support test for ``mean(diffs) > 0``.

    Reuses ``run_quality``'s ``_paired_one_sided_t`` (one implementation
    of the paired statistic across the repository). Zero-variance
    unanimous sweeps count only when the exact one-sided sign-flip
    p-value ``2^{-(n-1)}`` clears ``alpha`` (n >= 6 at alpha=0.05) — the
    same evidence rule Exp 5 applies. Returns ``(stats, supported)``.
    """
    from run_quality import _bonferroni_crit_q, _paired_one_sided_t

    stats = _paired_one_sided_t(diffs)
    n = int(stats['n_pairs'])
    crit = _bonferroni_crit_q(n, alpha=alpha) if n >= 2 else None
    supported = bool(crit is not None and stats['t_stat'] is not None
                     and stats['t_stat'] > crit)
    if stats['t_stat'] is None and stats['all_positive'] and n >= 2:
        supported = bool(2.0 ** (-(n - 1)) <= alpha)
    return stats, supported


def greedy_search(problem: SearchProblem, start: Candidate, *,
                  max_rounds: int = 3,
                  topk_grid: Sequence[int] = DEFAULT_TOPK_GRID,
                  include_swaps: bool = True,
                  min_delta: float = 0.0,
                  require_supported: bool = True,
                  alpha: float = 0.05,
                  selection_correction: bool = False,
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

    With ``require_supported`` (default) the raw improvement is ALSO
    checked against the seed-paired one-sided t-test at ``alpha`` whenever
    both records carry per-seed metrics: a move whose mean gain is not
    reproducible across seeds is recorded (``improvement=True``,
    ``improvement_supported=False``) but NOT accepted. Problems whose
    metrics lack per-seed accuracies (injected synthetic landscapes) fall
    back to the raw rule, as does a bootstrap move from ``score=None``.

    ``selection_correction`` (default False, exposed as
    ``--selection-correction``) applies a Bonferroni correction over the
    whole set of feasible neighbors that could have been selected in that
    round. Testing only the argmax candidate at the uncorrected alpha is
    a winner's-curse error: the selected candidate is the best of many
    correlated comparisons, so its p-value is not valid as if it were
    pre-registered. With the flag on, the acceptance test uses
    ``alpha / n_selectable`` and the round record carries both alphas;
    ``best_vs_start`` is similarly corrected over all evaluated feasible
    candidates. The correction is off by default because the search is
    documented as exploratory and at 3 seeds it is extremely conservative.

    The reported ``best`` closes the quality plateau: among the feasible
    candidates whose score is no higher than the greedy endpoint's and no
    more than ``min_delta`` below it, a strictly resource-cheaper one wins
    (so a state-neutral top-k reduction is recommended even though the
    strict-improvement rule would not move to it); without a strictly
    cheaper tie the endpoint itself is reported, so an unsupported
    noise-sized score bump can never promote a candidate. The raw endpoint is kept as ``greedy_endpoint`` for the
    trace. The returned ``best_vs_start`` block is the paired test of the
    recommended candidate against the start (``None`` when the metrics do
    not carry per-seed data).
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
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(
            f'greedy_search: alpha must be in (0, 1), got {alpha!r}.')

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
        paired = None
        supported = None
        family_size = len(selectable)
        alpha_move = (float(alpha) / family_size
                      if (selection_correction and family_size)
                      else float(alpha))
        if improved and current_score is not None and require_supported:
            pair = _paired_seed_diffs(best_rec, current_record)
            if pair is not None:
                pair_stats, supported = _paired_support(
                    pair[1], alpha=alpha_move)
                paired = {
                    'seeds': [int(s) for s in pair[0]],
                    'selection_correction': bool(selection_correction),
                    'selection_family_size': family_size,
                    'alpha_uncorrected': float(alpha),
                    'alpha_used': alpha_move,
                    **pair_stats,
                }
        accepted = bool(improved and supported is not False)
        if not improved:
            reason = 'no strict improvement over current'
        elif accepted:
            reason = ('improved' if paired is None
                      else 'improved (seed-paired supported)')
        else:
            reason = (f'mean score improved but not reproducible across '
                      f'seeds (paired one-sided t, '
                      f'alpha={alpha_move:g}'
                      + (f' = {float(alpha):g}/{family_size} selection-'
                         f'corrected' if selection_correction else '')
                      + ')')
        rounds.append({
            'round': round_index,
            'from': current.label,
            'from_score': current_score,
            'best_neighbor': best_rec['label'],
            'best_neighbor_score': best_rec['score'],
            'best_neighbor_state_bytes': best_rec['state_bytes_allocated'],
            'best_neighbor_decode_reads': best_rec[
                'decode_marginal_read_elements'],
            'improvement': bool(improved),
            'improvement_supported': supported,
            'selection_correction': bool(selection_correction),
            'selection_family_size': family_size,
            'alpha_uncorrected': float(alpha),
            'alpha_used': alpha_move,
            'paired': paired,
            'accepted': accepted,
            'reason': reason,
            'candidates': [rec['label'] for _, rec in fresh],
        })
        if logger is not None:
            logger.info(
                'round %d: %s -> %s (%s, %.4f -> %.4f)', round_index,
                current.label, best_rec['label'],
                'accept' if accepted else 'stop',
                -1.0 if current_score is None else current_score,
                best_rec['score'])
        if not accepted:
            break
        current, current_record = best_cand, best_rec

    # Plateau closure: the greedy endpoint may be quality-tied with a
    # resource-cheaper candidate that the strict-improvement rule refused
    # to move to (e.g. the state-neutral top-k move). Report the cheapest
    # feasible candidate within ``min_delta`` of the endpoint's score, so
    # the recommended configuration is never Pareto-dominated on its own
    # plateau. The raw endpoint stays available as ``greedy_endpoint``.
    # The closure only ever recommends a STRICTLY resource-cheaper tie:
    # with equal resources the endpoint itself wins, so an unsupported
    # (noise-sized) score bump can never promote a non-endpoint candidate.
    # The plateau is bounded ABOVE by the endpoint's score: a neighbour
    # with a higher raw mean that failed the paired-support test is not a
    # "tie", so it must not be promoted merely because it is cheaper.
    endpoint = current_record
    plateau = [r for r in records.values()
               if r['feasible'] and r['score'] is not None
               and endpoint['score'] is not None
               and endpoint['score'] - float(min_delta)
               <= r['score'] <= endpoint['score']]
    best = endpoint
    if plateau:
        candidate = min(plateau, key=lambda r: (
            r['state_bytes_allocated'], r['decode_marginal_read_elements'],
            r['label']))
        endpoint_cost = (endpoint['state_bytes_allocated'],
                         endpoint['decode_marginal_read_elements'])
        candidate_cost = (candidate['state_bytes_allocated'],
                          candidate['decode_marginal_read_elements'])
        if candidate_cost < endpoint_cost:
            best = candidate
    ordered = sorted(records.values(),
                     key=lambda r: (r['state_bytes_allocated'], r['label']))
    best_vs_start = None
    pair = _paired_seed_diffs(best, get(start))
    if pair is not None:
        final_family_size = max(1, sum(
            1 for r in records.values()
            if r['feasible'] and r['score'] is not None))
        alpha_final = (float(alpha) / final_family_size
                       if selection_correction else float(alpha))
        pair_stats, supported = _paired_support(pair[1], alpha=alpha_final)
        best_vs_start = {
            'test': ('seed-paired one-sided t on per-seed accuracy '
                     '(best - start); H1: best > start'),
            'alpha': alpha_final,
            'alpha_uncorrected': float(alpha),
            'selection_correction': bool(selection_correction),
            'selection_family_size': final_family_size,
            'seeds': [int(s) for s in pair[0]],
            'significant': bool(supported),
            **pair_stats,
        }
    return {
        'start': get(start),
        'best': best,
        'greedy_endpoint': endpoint,
        'rounds': rounds,
        'records': ordered,
        'pareto': pareto_frontier(ordered),
        'best_vs_start': best_vs_start,
        'acceptance_policy': {
            'raw_rule': 'best feasible neighbor score > current + min_delta',
            'min_delta': float(min_delta),
            'require_supported': bool(require_supported),
            'alpha': float(alpha),
            'selection_correction': bool(selection_correction),
            'selection_correction_method': (
                'bonferroni over all feasible scored contenders in the '
                'round (and over all evaluated candidates for '
                'best_vs_start)'
                if selection_correction else
                'none: the selected argmax is tested at the uncorrected '
                'alpha, so its p-value is selection-biased and the search '
                'must be treated as exploratory'),
            'seed_support_rule': (
                'seed-paired one-sided t on per-candidate common-seed '
                'accuracies; zero-variance unanimous sweeps need '
                '2^{-(n-1)} <= alpha; falls back to the raw rule when '
                'per-seed metrics are absent'),
        },
    }


# ===========================================================================
# 4. MQAR objective (the trained probe)
# ===========================================================================

def make_mqar_objective(*, m: int, m2: int, seeds: Sequence[int] = (42, 43, 44),
                        steps: int = 100, seq_len: int = 16,
                        d_model: int = 32, train_batch: int = 16,
                        eval_batches: int = 4, eval_batch: int = 64,
                        device='cpu', logger: logging.Logger | None = None,
                        ) -> Callable[[Candidate], dict]:
    """Return an ``evaluate(candidate)`` closure for the MQAR probe.

    Trains one probe per candidate per seed through Exp 9's
    ``recall_reach.train_recall_model`` (same data stream per seed for
    every candidate, so comparisons are paired) and returns the mean
    accuracy as the score. Per-seed accuracies, the mean's sample std /
    standard error / 95% t-CI, losses' final value, and the geometry are
    recorded in ``metrics`` for the trace — the greedy acceptance rule in
    :func:`greedy_search` reads the per-seed pairings from there.
    """
    from recall_reach import evaluate_recall, train_recall_model
    from run_quality import _t_crit_975

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
        if len(finite) >= 2 and score is not None:
            var = sum((a - score) ** 2 for a in finite) / (len(finite) - 1)
            score_std = math.sqrt(var)
            score_se = score_std / math.sqrt(len(finite))
            t_crit = _t_crit_975(len(finite))
            score_ci95 = (score_se * t_crit) if t_crit is not None else None
        else:
            score_std = score_se = score_ci95 = None
        return {
            'score': score,
            'metrics': {
                'accuracy': score,
                'accuracy_std': score_std,
                'accuracy_se': score_se,
                'accuracy_ci95': score_ci95,
                'n_seeds': len(finite),
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


def _fmt_score(rec: dict) -> str:
    """``mean±std`` when the record carries a seed spread, else the mean."""
    score = rec.get('score')
    if score is None:
        return 'n/a'
    std = (rec.get('metrics') or {}).get('accuracy_std')
    return f'{score:.4f}' if std is None else f'{score:.4f}±{std:.4f}'


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
          f"{'reads/tok':>12}{'ok':>4}{'score mean±std':>19}")
    for rec in report['records']:
        print(f"{','.join(rec['layout']):<28}{rec['csa_topk']:>5}"
              f"{rec['state_bytes_allocated']:>12}"
              f"{str(rec['state_constant']):>7}"
              f"{rec['decode_marginal_read_elements']:>12.1f}"
              f"{str(rec['feasible']):>5}"
              f"{('  n/a' if rec['score'] is None else f'{_fmt_score(rec):>19}')}")
    print('-' * 100)
    best = report['best']
    print(f"BEST: {best['label']}  score={_fmt_score(best)}  "
          f"state={_fmt_bytes(best['state_bytes_allocated'])}  "
          f"reads/token={best['decode_marginal_read_elements']:.1f}")
    bvs = report.get('best_vs_start')
    if bvs is not None:
        t_stat = bvs['t_stat']
        t_s = 'n/a' if t_stat is None else f'{t_stat:.3f}'
        print(f"BEST vs START (seed-paired): mean_diff={bvs['mean_diff']:+.4f}  "
              f"t={t_s}  n={bvs['n_pairs']}  "
              f"significant={bvs['significant']}")
    else:
        print('BEST vs START: no per-seed metrics (paired test unavailable)')
    print(f"PARETO FRONT ({len(report['pareto'])} points):")
    for rec in report['pareto']:
        print(f"  {rec['label']:<30} score={_fmt_score(rec)}  "
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
                        help='comma-separated probe seeds (default: '
                             'SEARCH_SEEDS count, 3 -> 42,43,44; >=3 seeds '
                             'is the policy, fewer is flagged exploratory)')
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
    parser.add_argument('--selection-correction', action='store_true',
                        help='Bonferroni-correct the paired acceptance test '
                             'over all feasible contenders in each round '
                             '(default: off, exploratory)')
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
        n_seeds = parse_int_env('SEARCH_SEEDS', 3, min_value=1,
                                logger=logger)
        seeds = tuple(42 + i for i in range(n_seeds))
    if len(seeds) < 3:
        logger.warning(
            'only %d probe seed(s): the seed-paired acceptance test is '
            'undefined and candidate differences are exploratory '
            '(policy: >= 3)', len(seeds))
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

    # Pre-search overfit gate through the exact probe construction path.
    # A greedy search trains many candidates; a broken recipe must be caught
    # once, before the compute is spent.
    from recall_reach import (
        VOCAB,
        _build_recall_model,
        make_long_range_batch,
    )
    from run_ablation import _run_ablation_overfit_gate

    gate_record, proceed = _run_ablation_overfit_gate(
        [(f'search start {start.label}',
          lambda: _build_recall_model(args.m, args.m2, start.layout,
                                      args.d_model, device))],
        device, env_prefix='SEARCH',
        make_task_batch=lambda b, embed, gen: make_long_range_batch(
            b, args.seq_len, 1, VOCAB, embed, device, generator=gen),
        seq_len=args.seq_len)
    if not proceed:
        return 1

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
        selection_correction=args.selection_correction,
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
            'n_seeds': len(seeds),
            'statistical_design': {
                'seeds_per_candidate': len(seeds),
                'minimum_seeds_for_claims': 3,
                'shared_seed_list': True,
                'seed_policy': ('every candidate is evaluated on the SAME '
                                'seed list, so candidate differences are '
                                'seed-paired'),
                'reported_statistics': ('per-candidate mean, sample std '
                                        '(ddof=1), standard error and 95% '
                                        't-CI of MQAR accuracy; per-seed '
                                        'accuracies retained'),
                'selection_correction': (
                    'bonferroni over feasible contenders'
                    if args.selection_correction else
                    'none (uncorrected argmax test; exploratory)'),
                'warning': (
                    f'only {len(seeds)} probe seed(s): the seed-paired '
                    f'acceptance test is undefined and candidate '
                    f'differences are exploratory at this seed budget'
                    if len(seeds) < 3 else None),
            },
            'train_batch': int(args.train_batch),
            'eval_batches': int(args.eval_batches),
            'eval_batch': int(args.eval_batch),
            'topk_grid': [int(k) for k in topk_grid],
            'include_swaps': not args.no_swaps,
            'max_rounds': int(args.max_rounds),
            'min_delta': float(args.min_delta),
            'selection_correction': bool(args.selection_correction),
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
        'best_vs_start': result['best_vs_start'],
        'acceptance_policy': result['acceptance_policy'],
        'overfit_sanity_check': gate_record,
        'provenance': capture_provenance(),
    }
    payload = sanitize_for_json(report)
    json.dumps(payload, allow_nan=False)
    if not args.quiet:
        _print_report(payload)
    out_path = args.json or RESULTS_PATH
    write_results_json(payload, out_path, logger=logger)

    # Campaign-level ledger entry: seed list, full search recipe, data
    # version, gate verdict and best candidate in one strict-JSON line.
    gate_failures = sum(1 for r in gate_record.get('reports', [])
                        if not r.get('passed'))
    best = result['best']
    append_run_manifest(make_run_manifest_entry(
        experiment='exp12_layerwise_search',
        seeds=list(seeds),
        provenance=report['provenance'],
        hyperparameters=report['config'],
        data={
            'probe': 'mqar-long-range-cue-last',
            'generator': 'run_quality.make_long_range_batch',
            'vocab': int(VOCAB),
            'seq_len': int(args.seq_len),
            'd_model': int(args.d_model),
            'shared_seed_list': True,
        },
        artifacts={'results': out_path},
        results={
            'n_candidates': len(result['records']),
            'n_rounds': len(result['rounds']),
            'best_label': best.get('label'),
            'best_score': best.get('score'),
            'best_state_bytes': best.get('state_bytes_allocated'),
            'overfit_gate_enabled': bool(gate_record.get('enabled')),
            'overfit_gate_failures': int(gate_failures),
        },
        status=('completed' if gate_failures == 0 else 'partial'),
    ), logger=logger)

    print(f'[layerwise-search] wrote {out_path} '
          f'({len(result["records"])} candidates evaluated)')
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    sys.exit(main())
