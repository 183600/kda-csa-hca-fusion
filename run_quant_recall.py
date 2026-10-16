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
* Multi-seed aggregation reports mean/min/max recall; "exact" means
  min-over-seeds == 1.0 (conservative). Three seeds keep the probe fast;
  the 2-bit cells are the noisy ones and are reported with min/max.

Output
------
``results/exp10_quant_recall.json`` (versioned envelope via
``results_schema``) with ``row_kind`` in {``quant_recall``,
``frontier_summary``}; CLI::

    python3 run_quant_recall.py
    QUANT_BITS=2,3,4 QUANT_SEEDS=0,1 python3 run_quant_recall.py
"""

from __future__ import annotations

import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    capture_provenance,
    configure_torch_for_device,
    make_seeded_generator,
)
from kv_quant import (  # noqa: E402
    FP16_BITS,
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


def _decode_recall(reads: torch.Tensor, values: torch.Tensor):
    """Nearest-value decode: recall fraction and max read-back error."""
    dist = torch.cdist(reads.float(), values.float())
    pred = dist.argmin(dim=-1)
    target = torch.arange(values.shape[0], device=values.device)
    recall = float((pred == target).float().mean())
    max_err = float((reads - values).abs().max())
    return recall, max_err


def run_planted_recall(route: str, keys: torch.Tensor, values: torch.Tensor,
                       spec: QuantSpec | None = None) -> dict:
    """One planted-recall episode for one route and quantization spec.

    ``spec=None`` is the full-precision reference (no quantization).
    """
    if route not in _STATE_FN:
        raise ValueError(f'run_planted_recall: unknown route {route!r}; '
                         f'expected one of {list(ROUTES)}.')
    if spec is not None and not isinstance(spec, QuantSpec):
        raise TypeError(f'run_planted_recall: spec must be None or a '
                        f'QuantSpec, got {spec!r}')
    state = _STATE_FN[route](keys, values, spec,
                             quantize_after_write=spec is not None)
    reads = read_back(route, keys, state)
    recall, max_err = _decode_recall(reads, values)
    return {
        'readback_recall': recall,
        'max_read_error': max_err,
        'state_elements': int(state.numel()),
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
    return {'mean': sum(values) / len(values), 'min': min(values),
            'max': max(values)}


def build_quant_recall_rows(device='cpu', seeds=DEFAULT_SEEDS,
                            bits_list=DEFAULT_BITS,
                            group_sizes=DEFAULT_GROUP_SIZES,
                            n_keys=DEFAULT_N_KEYS,
                            value_dim=DEFAULT_VALUE_DIM,
                            routes=ROUTES,
                            include_asymmetric=True) -> list:
    """Run the probe matrix and return flat, JSON-safe rows.

    One row per (route, spec) cell aggregating over seeds, plus a
    ``frontier_summary`` row per route (cheapest exact-recall cell).
    """
    rows: list = []
    for route in routes:
        for spec in default_sweep_specs(bits_list, group_sizes,
                                        include_asymmetric):
            recalls, max_errs, state_elements = [], [], None
            for seed in seeds:
                keys, values = planted_batch(n_keys, value_dim, seed, device)
                out = run_planted_recall(route, keys, values, spec)
                recalls.append(out['readback_recall'])
                max_errs.append(out['max_read_error'])
                if state_elements is None:
                    state_elements = out['state_elements']
                elif state_elements != out['state_elements']:
                    raise AssertionError(
                        'state element count changed across seeds; the '
                        'probe geometry must be fixed')
            rec = _aggregate(recalls)
            errmax = _aggregate(max_errs)
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
                'recall_min': rec['min'],
                'recall_max': rec['max'],
                'max_read_error_mean': errmax['mean'],
                'max_read_error_max': errmax['max'],
                'exact_recall': bool(rec['min'] == 1.0),
            })
            rows.append(row)
    rows.extend(build_frontier_summary(rows, routes))
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
              f"{'bytes':>7} | {'x_fp16':>6} | recall (min..max) | "
              f"max|err|")
    logger.info(header)
    logger.info('-' * len(header))
    for row in rows:
        if row['row_kind'] != 'quant_recall':
            continue
        logger.info(
            f"{row['route']:>7} | {_cell_label(row):>12} | "
            f"{row['bits_per_element']:>7.2f} | {row['state_bytes']:>7.1f} | "
            f"{row['compression_vs_fp16']:>5.2f}x | "
            f"{row['recall_mean']:.3f} "
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
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
    logger.info(f'  routes: {list(ROUTES)}')
    logger.info(f'  bits: {list(bits_list)}, groups: {list(group_sizes)}, '
                f'seeds: {list(seeds)}')
    logger.info(f'  geometry: n_keys={n_keys}, value_dim={value_dim}')

    rows = build_quant_recall_rows(
        device=device, seeds=seeds, bits_list=bits_list,
        group_sizes=group_sizes, n_keys=n_keys, value_dim=value_dim)
    _print_rows(rows)
    _print_frontier(rows)

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
            'state-bits axis; seeds are aggregated mean/min/max.'),
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
            'visible seed variance (min..max reported).',
        ],
        'provenance': capture_provenance(),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH) or '.', exist_ok=True)
    write_schema(RESULTS_PATH, rows, kind='exp10_quant_recall',
                 metadata=metadata)
    logger.info(f'\nSaved: {RESULTS_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
