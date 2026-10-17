# Greedy layerwise architecture search (Exp 12)

`layerwise_search.py` searches the **layerwise placement** of the three
operators (KDA / CSA / HCA) under the repository's two exact resource
currencies. It complements — rather than replaces — the hand-designed
layouts measured by Exp 5 (`position` preset), Exp 9 (`recall_reach`
interleaving presets) and Exp 9b (`run_budget_frontier` ratio budgets).

## Search space

A search point is a `layerwise_search.Candidate`:

* `layout` — an exact, resolved per-layer kind tuple
  (`('kda', 'csa', 'hca', ...)`), accepted directly as
  `HybridConfig.layout`; and
* `csa_topk` — the CSA top-k block-selection width, carried alongside the
  layout because it is state-neutral and otherwise would need a separate
  search axis (it is ignored when the layout has no CSA layer).

Three deterministic single-edit neighborhoods are generated per round:

| Move | Count | Effect on state bytes | Rationale |
|---|---|---|---|
| substitution | `2 · depth` | changes the operator mix | the layerwise-kind edit |
| adjacent swap | ≤ `depth − 1` | **none** (sum is order-invariant) | isolates the empirical placement effect |
| top-k move | `len(grid) − 1` (if CSA) | **none** | trades decode reads against quality |

The neighborhood order is fixed, so the whole search is reproducible from
`(start, objective, seeds, budget)` alone.

## Cost model (exact, analytic)

* **Decode-state bytes** — `cache_accounting.analytic_cache_components`
  (the same component taxonomy Exp 3 reconciles against the live decode
  caches), summed over the two cache pools and reported per operator.
  Components in the `PARAMETER` pool (attention sinks) are shared by all
  requests and are reported separately. A memory-only layout is exactly
  constant in `T`; any layout with a CSA/HCA layer grows with the
  compressed-block count — the two-point `state_constant` flag is the
  Exp 6 certificate reduced to the search's needs.
* **KV read budget** — `kv_budget` (Exp 9b), both the prefill average and
  the marginal decode step after a `T`-token prefill. The CSA term is
  where the top-k selection cost lives: the indexer scans **every** visible
  compressed block before selecting `topk`.

Costs are per sequence, converted at `dtype_bytes` (default 4) per element;
mixed precision, allocator slack and the batch dimension are out of scope
(see `cache_accounting` for the authoritative model).

## Objective

The default objective is the distance-controlled long-range MQAR probe of
Exp 9 (`recall_reach.train_recall_model` / `evaluate_recall`): one planted
`(key, value)` pair per sequence, cue at the last position, fixed
`run_quality.SMALL_MODEL_SPEC` geometry. Every candidate is trained from
scratch under the **same seeds**, so candidate differences are paired by
seed (Exp 5/8's validity policy) and a broken candidate is recorded, not
fatal.

## Greedy search

`greedy_search` is best-improvement: each round evaluates every unseen
neighbor, accepts the best budget-feasible candidate whose score exceeds
the current one by more than `min_delta`, and stops when no neighbor
improves. Tie-breaks are (lower state bytes, lower reads, label), so ties
are broken deterministically toward the cheaper configuration. **Every**
evaluated candidate is retained in `records`, and the quality-vs-cost
`pareto` frontier of the explored neighborhood is reported even when a
move is rejected.

After the loop, the reported `best` closes the quality plateau: among
feasible candidates within `min_delta` of the greedy endpoint's score, the
cheapest one (state bytes, then reads, then label) is recommended. A
resource-only improvement (equal score, lower reads — e.g. a state-neutral
top-k reduction) is therefore always recommended over the endpoint it ties
with, while the raw endpoint remains visible as `greedy_endpoint`.

This greediness is myopic by design: because a substitution changes the
state cost in one step, a tight budget can make a quality-improving move
infeasible even when a two-step detour (drop one retrieval layer, add
another) would fit. The trace makes that visible; a broader beam search is
left for future work.

## How to run

```bash
# quick smoke (3 layers, 60 steps, 1 round)
python3 layerwise_search.py --quick

# the reference-shaped run: 3:1:1 start, capped at 1.25x its own
# decode-state footprint at T=4096, paired seeds
python3 layerwise_search.py --depth 5 --steps 120 --seeds 42,43 \
    --budget-mult 1.25 --json results/exp12_layerwise_search.json

# free exploration (no state cap), then read the Pareto front
python3 layerwise_search.py --depth 5 --steps 120 --max-rounds 3
```

Environment fallback for the step count: `SEARCH_STEPS`.

Output: `results/exp12_layerwise_search.json` (strict JSON via
`kaggle_setup.write_results_json`) with the full search config, the start
and best records, the per-round trace, every evaluated candidate, the
Pareto frontier, and provenance. The probe is the frozen small geometry,
so results are **exploratory** — the deliverable is the search machinery
and the reproducible accounting, not an architecture claim.

## Tests

`test_layerwise_search.py` (18 tests) pins the neighborhood exactness and
determinism, the cost model against an independent walk over
`analytic_cache_components` and against the committed Exp 9b geometry
builder, `O(1)`-vs-growing state behavior, read-budget monotonicity in
`csa_topk`, the search loop on injected landscapes (target recovery,
budget exclusion, error records, reproducibility, Pareto correctness), and
the build/forward + objective integration:

```bash
pytest -q test_layerwise_search.py
```
