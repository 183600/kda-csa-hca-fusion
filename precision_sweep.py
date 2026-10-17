#!/usr/bin/env python3
"""Mixed-precision parity sweep for the memory-route cores.

Why a separate harness?
-----------------------
Every correctness test in this repository pins the operators in fp32 (and
``run_quality.py`` trains them in fp32), while ``train_lm_autodl.py``
deploys them under bf16/fp16 autocast. The gap between "validated" and
"deployed" is exactly where reduced-precision regressions live:

* a single-seed smoke test cannot distinguish legitimate input/output
  quantization from a real algorithm bug, and it can pass by seed luck —
  the same tolerance check may fail on a different random draw;
* comparing a reduced-precision path against another reduced-precision
  path measures disagreement between two approximations, not precision
  loss;
* an error number without a *scale* ("max abs diff = 3e-3") is not
  falsifiable: fp16's unit roundoff is ~150x bf16's, so the same absolute
  error means very different things per dtype.

This module supplies the missing evaluation harness:

1. **fp64 oracle.** For every case the same operator is also run in fp64
   on the *identical* inputs (built in fp64, then rounded down for the
   tested run). The reported error is precision loss relative to the
   exact semantics, on a per-dtype interpretable scale: units of the
   dtype's unit roundoff ``u`` (``out_eps_units``).
2. **Seed sweep.** ``n_seeds`` independent input draws per
   (case, path, dtype); the aggregate reports max **and** median error,
   the spread (max/median), and the ``worst_seed`` so any budget failure
   is reproducible with a single number.
3. **Dtype contract.** On every row: the output must come back in the
   INPUT dtype (``ops_*`` contract), the returned state in compute
   precision (fp32 for fp16/bf16/fp32 inputs, fp64 for fp64). A harness
   that only looks at error would miss a silent upcast that hides
   precision loss from the caller.
4. **Two deployment modes.**
   * ``explicit``: inputs rounded to the tested dtype (weights/activations
     stored at that dtype; the ops upcast to fp32 internally).
   * ``autocast``: fp32 storage, ``torch.autocast`` downcasts eligible
     ops — the actual ``train_lm_autodl.py`` AMP path.
5. **Explicit error budgets.** ``error_budget_eps_units`` gives a named
   envelope in ``u`` units per dtype class: storage dtypes are
   quantization-limited (T-independent, because the ops compute in fp32;
   gate = ``factor * u``), while compute dtypes accumulate round-off over
   the T-step recurrence (gate = ``accum_factor * T * u``, the textbook
   ``gamma_T`` bound with slack for cancellation-sensitive paths). Budgets
   are smoke-level regression gates, not numerical-error theorems — the
   module says so in its JSON metadata.

The optional ``--json`` envelope (``results_schema`` kind
``'precision_sweep'``) records the aggregate rows plus the sweep
configuration, so a run is reproducible and diffable.

Usage::

    python3 precision_sweep.py                          # all cases, 3 dtypes, 3 seeds
    python3 precision_sweep.py --cases kda --length 256
    python3 precision_sweep.py --mode both --json results/precision_sweep.json
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from ops_kda import (assoc_scan_chunk_kda, naive_chunk_kda,  # noqa: E402
                     naive_recurrent_kda)
from ops_mamba import (chunked_mamba2_scan, naive_mamba2_scan,  # noqa: E402
                       quadratic_mamba2_scan)
from ops_rwkv import rwkv7_reference_scan  # noqa: E402

#: ``results_schema`` kind for the optional JSON output.
SWEEP_KIND = 'precision_sweep'

#: Unit roundoff (half a ulp at 1.0; IEEE binary formats).
UNIT_ROUNDOFF = {
    torch.float16: 2.0 ** -11,
    torch.bfloat16: 2.0 ** -8,
    torch.float32: 2.0 ** -24,
    torch.float64: 2.0 ** -53,
}

#: Dtypes the ops STORE at; internal compute is upcast to fp32 (see
#: ``ops_kda._compute_dtype`` and the twin helpers in ops_mamba/ops_rwkv).
STORAGE_DTYPES = (torch.float16, torch.bfloat16)

#: Storage-dtype verdict factor: allowed error = factor * u (T-independent;
#: the ops compute in fp32, so only quantization touches the dtype grid).
DEFAULT_BUDGET_FACTOR = 8.0

#: Compute-dtype verdict factor: allowed error = accum_factor * T * u (the
#: textbook linear round-off bound ``gamma_T = T*u`` with slack for
#: cancellation-sensitive draws; the quadratic SSD state measured 0.87*T*u
#: at T=1024, so a factor of 2 covers the observed envelope).
DEFAULT_ACCUM_FACTOR = 2.0

#: Default number of independent seeds per (case, path, dtype).
DEFAULT_N_SEEDS = 3
DEFAULT_SEEDS = tuple(range(DEFAULT_N_SEEDS))

#: Case registry: family -> numerically independent paths of the same op.
CASES = {
    'kda': ('rec', 'chunk', 'assoc'),
    'mamba2': ('naive', 'quad', 'chunk'),
    'rwkv7': ('rwkv',),
}

#: Per-family generator offset so families never share the same draw
#: (mirrors the +N_000_000 stream offsets used by run_quality/run_ablation).
_SEED_OFFSETS = {'kda': 0, 'mamba2': 10_000, 'rwkv7': 20_000}

#: Fixed tiny core geometry (batch 1, 2 heads, 8-wide head) — the sweep is
#: about dtype, not capacity, so the numbers stay cheap on CPU.
_GEOMETRY = {
    'kda': {'batch': 1, 'heads': 2, 'key_dim': 8, 'value_dim': 8},
    'mamba2': {'batch': 1, 'heads': 2, 'headdim': 8, 'd_state': 4,
               'ngroups': 1},
    'rwkv7': {'batch': 1, 'heads': 2, 'key_dim': 8, 'value_dim': 8},
}

MODES = ('explicit', 'autocast')


# ---------------------------------------------------------------------------
# Dtype / budget arithmetic
# ---------------------------------------------------------------------------
def dtype_name(dtype) -> str:
    """Short name of a torch dtype ('float16', not 'torch.float16')."""
    return str(as_torch_dtype(dtype)).replace('torch.', '')


def unit_roundoff(dtype) -> float:
    """Unit roundoff of a supported dtype (accepts torch.dtype or str)."""
    dt = as_torch_dtype(dtype)
    return UNIT_ROUNDOFF[dt]


def as_torch_dtype(dtype) -> torch.dtype:
    """Normalize ``dtype`` (``torch.dtype`` or short/long string) or raise."""
    if isinstance(dtype, torch.dtype):
        dt = dtype
    elif isinstance(dtype, str):
        key = dtype.strip().lower()
        key = key[len('torch.'):] if key.startswith('torch.') else key
        aliases = {
            'fp16': 'float16', 'half': 'float16',
            'bf16': 'bfloat16', 'bfloat16': 'bfloat16',
            'fp32': 'float32', 'float': 'float32', 'float32': 'float32',
            'fp64': 'float64', 'double': 'float64', 'float64': 'float64',
        }
        name = aliases.get(key, key)
        table = {dtype_name(d): d for d in UNIT_ROUNDOFF}
        if name not in table:
            raise ValueError(
                f'unsupported dtype {dtype!r}; expected one of '
                f'{sorted(table)} (or aliases fp16/bf16/fp32/fp64).')
        dt = table[name]
    else:
        raise TypeError(
            f'dtype must be a torch.dtype or str, got {type(dtype).__name__}.')
    if dt not in UNIT_ROUNDOFF:
        raise ValueError(f'unsupported dtype {dt}; not in {sorted(UNIT_ROUNDOFF)}.')
    return dt


def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name}={value!r} must be a positive int.')
    return value


def error_budget_eps_units(dtype, seq_len: int, *,
                           factor: float = DEFAULT_BUDGET_FACTOR,
                           accum_factor: float = DEFAULT_ACCUM_FACTOR) -> float:
    """Allowed worst-case relative error, in units of the dtype's ``u``.

    Two regimes, matching the ops' compute-dtype contract:

    * **storage dtypes** (fp16/bf16): the ops upcast inputs to fp32 and
      only round the output back, so the error is dominated by input and
      output quantization on the dtype grid — a T-independent envelope of
      a few ``u`` (the observed worst case on the shipped cores is ~2u),
      gated by ``factor``.
    * **compute dtypes** (fp32/fp64): round-off accumulates along the
      T-step recurrence and, on the cancellation-sensitive quadratic/
      chunked SSD paths, grows roughly linearly in T (measured 0.87*T*u
      for the state at T=1024). The gate is the textbook ``gamma_T = T*u``
      bound times ``accum_factor``.

    The factors are *regression gates* with deliberate headroom, not
    theorems: they must tolerate the observed accumulation while still
    catching order-of-magnitude precision regressions (e.g. an op that
    silently starts accumulating in the storage dtype).
    """
    dt = as_torch_dtype(dtype)
    _require_positive_int(seq_len, 'seq_len')
    for name, value in (('factor', factor), ('accum_factor', accum_factor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(
                f'{name}={value!r} must be a positive finite number.')
    if dt in STORAGE_DTYPES:
        return float(factor)
    return float(accum_factor) * float(seq_len)


# ---------------------------------------------------------------------------
# Seeded inputs (fp64 base; CPU generator -> reproducible on CPU and CUDA)
# ---------------------------------------------------------------------------
def build_inputs(case: str, seed: int, seq_len: int, *, device='cpu') -> dict:
    """Build one fp64 input draw for ``case`` at ``seed``.

    The draw is made in fp64 on a private CPU generator (so CPU and CUDA
    materializations are bit-identical) and then moved to ``device``.
    Tests cast the result down with ``.to(dtype)``; the oracle keeps the
    fp64 tensors, which makes "the tested input" EXACTLY the oracle input
    rounded to the dtype grid.
    """
    if case not in CASES:
        raise ValueError(f'unknown case {case!r}; expected {sorted(CASES)}.')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f'seed={seed!r} must be an int.')
    _require_positive_int(seq_len, 'seq_len')
    device = torch.device(device)
    gen = torch.Generator(device='cpu').manual_seed(
        (int(seed) + _SEED_OFFSETS[case]) & 0xFFFFFFFF)
    g64 = dict(device='cpu', dtype=torch.float64, generator=gen)

    if case == 'kda':
        g = _GEOMETRY['kda']
        B, T, H, K, V = g['batch'], seq_len, g['heads'], g['key_dim'], g['value_dim']
        q = torch.randn(B, T, H, K, **g64)
        k = torch.randn(B, T, H, K, **g64)
        return {
            'q': torch.nn.functional.normalize(q, dim=-1).to(device),
            'k': torch.nn.functional.normalize(k, dim=-1).to(device),
            'v': (torch.randn(B, T, H, V, **g64) * 0.5).to(device),
            'g': (-torch.rand(B, T, H, K, **g64) * 0.1).to(device),
            'beta': (torch.rand(B, T, H, **g64) * 0.5).to(device),
        }
    if case == 'mamba2':
        g = _GEOMETRY['mamba2']
        B, T, H, D, N = (g['batch'], seq_len, g['heads'], g['headdim'],
                         g['d_state'])
        G = g['ngroups']
        return {
            'x': (torch.randn(B, T, H, D, **g64) * 0.5).to(device),
            'dt': torch.nn.functional.softplus(
                torch.randn(B, T, H, **g64) * 0.3).to(device),
            'B': torch.randn(B, T, G, N, **g64).to(device),
            'C': torch.randn(B, T, G, N, **g64).to(device),
            'A': (-torch.exp(torch.randn(H, **g64))).to(device),
            'D': torch.randn(H, **g64).to(device),
        }
    # rwkv7
    g = _GEOMETRY['rwkv7']
    B, T, H, K, V = g['batch'], seq_len, g['heads'], g['key_dim'], g['value_dim']
    kk = torch.nn.functional.normalize(torch.randn(B, T, H, K, **g64), dim=-1)
    return {
        'r': torch.randn(B, T, H, K, **g64).to(device),
        'log_w': (-torch.rand(B, T, H, K, **g64)).to(device),
        'k': torch.randn(B, T, H, K, **g64).to(device),
        'v': (torch.randn(B, T, H, V, **g64) * 0.5).to(device),
        'kk': kk.to(device),
        'a': torch.rand(B, T, H, K, **g64).to(device),
    }


def call_case(case: str, path: str, inputs: dict, *, chunk_size: int = 16):
    """Run one (case, path) with ``inputs``; returns ``(out, final_state)``.

    ``output_final_state=True`` is forced so the state contract is always
    observable. The inputs are passed as-is (the caller has already cast
    them to the dtype under test); no internal dtype transformation is
    hidden here.
    """
    if case not in CASES:
        raise ValueError(f'unknown case {case!r}; expected {sorted(CASES)}.')
    if path not in CASES[case]:
        raise ValueError(
            f'unknown path {path!r} for case {case!r}; expected '
            f'{CASES[case]}.')
    _require_positive_int(chunk_size, 'chunk_size')
    if case == 'kda':
        fn = {'rec': naive_recurrent_kda,
              'chunk': lambda **kw: naive_chunk_kda(**kw, chunk_size=chunk_size),
              'assoc': lambda **kw: assoc_scan_chunk_kda(**kw,
                                                        chunk_size=chunk_size),
              }[path]
        return fn(**inputs, output_final_state=True)
    if case == 'mamba2':
        fn = {'naive': naive_mamba2_scan,
              'quad': quadratic_mamba2_scan,
              'chunk': lambda **kw: chunked_mamba2_scan(**kw,
                                                       chunk_size=chunk_size),
              }[path]
        return fn(**inputs, output_final_state=True)
    return rwkv7_reference_scan(**inputs, output_final_state=True)


# ---------------------------------------------------------------------------
# Metrics and one evaluation
# ---------------------------------------------------------------------------
def compare_to_oracle(test, oracle) -> dict:
    """Max-abs / relative error of ``test`` against the fp64 ``oracle``."""
    if test.shape != oracle.shape:
        raise ValueError(
            f'compare_to_oracle: shape mismatch {tuple(test.shape)} vs '
            f'{tuple(oracle.shape)}.')
    t = test.detach().to(torch.float64)
    r = oracle.detach().to(torch.float64)
    diff = (t - r).abs().max().item()
    scale = r.abs().max().item()
    if scale > 0.0:
        rel = diff / scale
    else:
        rel = 0.0 if diff == 0.0 else float('inf')
    return {'abs_err': diff, 'rel_err': rel, 'oracle_scale': scale}


def _expected_state_dtype(dtype: torch.dtype) -> torch.dtype:
    """Compute-precision state contract: fp64 only for fp64 inputs."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def evaluate_seed(case: str, path: str, dtype, seed: int, *,
                  seq_len: int = 64, device='cpu', chunk_size: int = 16,
                  mode: str = 'explicit') -> dict:
    """Run one (case, path, dtype, seed) and return a single result row.

    The fp64 oracle runs the SAME (case, path); the tested run sees the
    oracle inputs rounded to ``dtype`` (``explicit``) or kept fp32 under
    ``torch.autocast`` (``autocast``). Any exception is captured into the
    row as ``error`` instead of aborting the sweep (mirrors the per-seed
    error handling in ``run_quality.train_multi_seed``).
    """
    dt = as_torch_dtype(dtype)
    if mode not in MODES:
        raise ValueError(f'mode={mode!r} must be one of {MODES}.')
    if mode == 'autocast' and dt not in STORAGE_DTYPES:
        raise ValueError(
            f'autocast mode only supports storage dtypes '
            f'{[dtype_name(d) for d in STORAGE_DTYPES]}, got {dt}.')
    _require_positive_int(seq_len, 'seq_len')

    row = {
        'case': case, 'path': path, 'dtype': dtype_name(dt), 'mode': mode,
        'seed': int(seed), 'seq_len': int(seq_len),
        'chunk_size': int(chunk_size),
    }

    base = build_inputs(case, seed, seq_len, device=device)
    try:
        oracle_out, oracle_state = call_case(case, path, base,
                                             chunk_size=chunk_size)
    except Exception as e:  # harness bug/wrong geometry: record, don't abort
        row.update({'error': f'oracle: {type(e).__name__}: {e}'})
        return row

    if mode == 'explicit':
        test_inputs = {k: v.to(dt) for k, v in base.items()}
        expected_out_dtype = dt
        try:
            out, state = call_case(case, path, test_inputs,
                                   chunk_size=chunk_size)
        except Exception as e:
            row.update({'error': f'test: {type(e).__name__}: {e}'})
            return row
    else:  # autocast: fp32 storage, torch decides what to downcast
        test_inputs = {k: v.to(torch.float32) for k, v in base.items()}
        expected_out_dtype = torch.float32
        try:
            device_type = torch.device(device).type
            with torch.autocast(device_type=device_type, dtype=dt):
                out, state = call_case(case, path, test_inputs,
                                       chunk_size=chunk_size)
        except Exception as e:
            row.update({'error': f'autocast: {type(e).__name__}: {e}'})
            return row

    u = UNIT_ROUNDOFF[dt]
    is_finite = bool(torch.isfinite(out).all().item()
                     and torch.isfinite(state).all().item())
    out_cmp = compare_to_oracle(out, oracle_out)
    state_cmp = compare_to_oracle(state, oracle_state)
    row.update({
        'out_dtype': dtype_name(out.dtype),
        'state_dtype': dtype_name(state.dtype),
        'out_dtype_ok': out.dtype == expected_out_dtype,
        'state_dtype_ok': state.dtype == _expected_state_dtype(
            dt if mode == 'explicit' else torch.float32),
        'finite_ok': is_finite,
        'out_abs_err': out_cmp['abs_err'],
        'out_rel_err': out_cmp['rel_err'],
        'out_eps_units': out_cmp['rel_err'] / u,
        'state_abs_err': state_cmp['abs_err'],
        'state_rel_err': state_cmp['rel_err'],
        'state_eps_units': state_cmp['rel_err'] / u,
        'oracle_out_scale': out_cmp['oracle_scale'],
        'oracle_state_scale': state_cmp['oracle_scale'],
    })
    return row


def run_sweep(*, cases=None, dtypes, seeds, seq_len: int = 64,
              device='cpu', chunk_size: int = 16,
              mode: str = 'explicit') -> list:
    """Evaluate the (case, path) x dtype x seed grid into flat result rows."""
    selected = list(CASES) if cases is None else list(cases)
    for case in selected:
        if case not in CASES:
            raise ValueError(f'unknown case {case!r}; expected {sorted(CASES)}.')
    dts = [as_torch_dtype(d) for d in dtypes]
    if not dts:
        raise ValueError('dtypes must be a non-empty iterable.')
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError('seeds must be a non-empty iterable.')
    rows = []
    for case in selected:
        for path in CASES[case]:
            for dt in dts:
                for seed in seed_list:
                    rows.append(evaluate_seed(
                        case, path, dt, seed, seq_len=seq_len, device=device,
                        chunk_size=chunk_size, mode=mode))
    return rows


# ---------------------------------------------------------------------------
# Seed-sweep aggregation and verdicts
# ---------------------------------------------------------------------------
def _median(values: list):
    return statistics.median(values) if values else None


def summarize(rows: list, *,
              budget_factor: float = DEFAULT_BUDGET_FACTOR,
              accum_factor: float = DEFAULT_ACCUM_FACTOR) -> list:
    """Aggregate per-seed rows into one verdict row per (case, path, dtype, mode).

    For every group: max/median ``out_eps_units`` over the seeds, the
    spread (max/median, how much a single seed could have fooled a
    one-shot gate), the seed producing the maximum, the budget in the
    same units, and the contract flags. ``verdict`` is ``'pass'`` only if
    every seed is finite, dtype-contract-clean, and within both budgets.
    Pure over its input (so tests can feed synthetic rows).
    """
    if not isinstance(rows, list):
        raise TypeError(f'rows must be a list, got {type(rows).__name__}.')
    groups: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError('every row must be a dict.')
        key = (row.get('case'), row.get('path'), row.get('dtype'), row.get('mode'))
        groups.setdefault(key, []).append(row)

    out_rows = []
    for (case, path, dtype, mode), group in groups.items():
        dt = as_torch_dtype(dtype)
        seq_len = _require_positive_int(int(group[0].get('seq_len', 64)),
                                        'seq_len')
        budget = error_budget_eps_units(dt, seq_len, factor=budget_factor,
                                        accum_factor=accum_factor)
        ok_rows = [r for r in group if 'error' not in r]
        out_eps = [float(r['out_eps_units']) for r in ok_rows
                   if r.get('finite_ok')]
        state_eps = [float(r['state_eps_units']) for r in ok_rows
                     if r.get('finite_ok')]
        out_max = max(out_eps) if out_eps else None
        out_med = _median(out_eps)
        state_max = max(state_eps) if state_eps else None
        state_med = _median(state_eps)
        worst_seed = None
        if out_eps:
            worst_seed = int(max((r for r in ok_rows if r.get('finite_ok')),
                                 key=lambda r: r['out_eps_units'])['seed'])
        spread = (out_max / out_med) if out_max and out_med and out_med > 0 else None
        contract_ok = all(
            r.get('out_dtype_ok') and r.get('state_dtype_ok') for r in group)
        finite_ok = bool(ok_rows) and len(ok_rows) == len(group) and all(
            r.get('finite_ok') for r in group)
        out_budget_ok = out_max is not None and out_max <= budget
        state_budget_ok = state_max is not None and state_max <= budget
        verdict = 'pass' if (len(ok_rows) == len(group) and contract_ok
                             and finite_ok and out_budget_ok
                             and state_budget_ok) else 'fail'
        out_rows.append({
            'case': case, 'path': path, 'dtype': dtype_name(dt), 'mode': mode,
            'seq_len': seq_len,
            'n_seeds': len(group), 'n_ok': len(ok_rows),
            'out_eps_max': out_max, 'out_eps_median': out_med,
            'state_eps_max': state_max, 'state_eps_median': state_med,
            'out_spread': spread, 'worst_seed': worst_seed,
            'budget_eps': budget,
            'out_budget_ok': out_budget_ok,
            'state_budget_ok': state_budget_ok,
            'dtype_contract_ok': contract_ok,
            'finite_ok': finite_ok,
            'verdict': verdict,
        })
    return out_rows


def _eps_text(value) -> str:
    return 'n/a' if value is None else f'{value:.1f}u'


def failures(summary_rows: list) -> list:
    """The failing aggregate rows, with a one-line human reason each."""
    out = []
    for row in summary_rows:
        if row.get('verdict') == 'pass':
            continue
        reasons = []
        if row.get('n_ok', 0) != row.get('n_seeds'):
            reasons.append(f"{row['n_seeds'] - row['n_ok']} seed(s) errored")
        if not row.get('finite_ok'):
            reasons.append('non-finite output/state')
        if not row.get('dtype_contract_ok'):
            reasons.append('dtype contract violated')
        if not row.get('out_budget_ok'):
            reasons.append(
                f"out error {_eps_text(row['out_eps_max'])} > "
                f"{row['budget_eps']:.1f}u")
        if not row.get('state_budget_ok'):
            reasons.append(
                f"state error {_eps_text(row['state_eps_max'])} > "
                f"{row['budget_eps']:.1f}u")
        label = f"{row['case']}/{row['path']}/{row['dtype']}/{row['mode']}"
        out.append(f"{label}: {'; '.join(reasons) or 'failed'}")
    return out


def format_table(summary_rows: list) -> str:
    """Fixed-width text table for the CLI (pure; tested on synthetic rows)."""
    header = (f"{'case':>7} {'path':>6} {'dtype':>9} {'mode':>9} "
              f"{'seeds':>5} {'out_eps_max':>11} {'median':>8} "
              f"{'budget':>7} {'verdict':>7}")
    lines = [header, '-' * len(header)]
    for r in summary_rows:
        out_max = f"{r['out_eps_max']:.2f}" if r['out_eps_max'] is not None else 'n/a'
        out_med = f"{r['out_eps_median']:.2f}" if r['out_eps_median'] is not None else 'n/a'
        lines.append(
            f"{str(r['case']):>7} {str(r['path']):>6} {str(r['dtype']):>9} "
            f"{str(r['mode']):>9} {str(r['n_ok']) + '/' + str(r['n_seeds']):>5} "
            f"{out_max:>11} {out_med:>8} {r['budget_eps']:>7.1f} "
            f"{str(r['verdict']):>7}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_csv_list(raw: str, name: str) -> list:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        raise ValueError(f'--{name} must not be empty.')
    return parts


def _parse_int_csv(raw: str, name: str) -> list:
    try:
        return [int(p) for p in _parse_csv_list(raw, name)]
    except ValueError as e:
        raise ValueError(f'--{name} must be a comma list of ints: {e}') from e


def default_autocast_dtypes(device: str) -> tuple:
    """CPU AMP is bf16 in practice; CUDA supports both half and bf16.

    CPU ``float16`` autocast is emulated (observed ~4x the bf16 error at
    the same length) and is not a production path, so it is excluded by
    default and only run when explicitly requested via ``--autocast-dtypes``.
    """
    return (torch.bfloat16,) if torch.device(device).type == 'cpu' \
        else (torch.float16, torch.bfloat16)


def _write_json(path: str, summary_rows: list, metadata: dict) -> None:
    from results_schema import write_checked
    write_checked(path, summary_rows, kind=SWEEP_KIND, metadata=metadata)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Mixed-precision parity sweep for the memory-route cores '
                    '(fp64 oracle, seed sweep, dtype contract, error budgets).')
    parser.add_argument('--cases', default='all',
                        help="comma list or 'all' (default). "
                             f"Available: {','.join(CASES)}")
    parser.add_argument('--dtypes', default='float32,float16,bfloat16',
                        help='explicit-mode dtypes (default: %(default)s)')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)),
                        help='comma list of seed indices (default: %(default)s)')
    parser.add_argument('--length', type=int, default=64,
                        help='sequence length per case (default: %(default)s)')
    parser.add_argument('--chunk-size', type=int, default=16,
                        help='chunk size for the chunked paths (default: %(default)s)')
    parser.add_argument('--mode', choices=('explicit', 'autocast', 'both'),
                        default='explicit',
                        help="'explicit' rounds inputs to --dtypes; "
                             "'autocast' keeps fp32 storage and lets "
                             "torch.autocast downcast; 'both' runs both")
    parser.add_argument('--autocast-dtypes', default=None,
                        help='autocast target dtypes (default: bfloat16 on '
                             'CPU, float16,bfloat16 on CUDA)')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'),
                        help='device (default: %(default)s)')
    parser.add_argument('--budget-factor', type=float,
                        default=DEFAULT_BUDGET_FACTOR,
                        help='storage-dtype error budget = factor * u '
                             '(default: %(default)s)')
    parser.add_argument('--accum-factor', type=float,
                        default=DEFAULT_ACCUM_FACTOR,
                        help='compute-dtype error budget = accum_factor * T * u '
                             '(textbook gamma_T bound; default: %(default)s)')
    parser.add_argument('--json', default=None,
                        help='write the aggregate rows as a versioned '
                             f'results_schema envelope (kind={SWEEP_KIND!r})')
    args = parser.parse_args(argv)

    cases = list(CASES) if args.cases.strip().lower() == 'all' \
        else _parse_csv_list(args.cases, 'cases')
    explicit_dtypes = [as_torch_dtype(d) for d in
                       _parse_csv_list(args.dtypes, 'dtypes')]
    seeds = _parse_int_csv(args.seeds, 'seeds')
    _require_positive_int(args.length, 'length')
    _require_positive_int(args.chunk_size, 'chunk_size')

    runs = []  # (mode, [dtypes])
    if args.mode in ('explicit', 'both'):
        runs.append(('explicit', explicit_dtypes))
    if args.mode in ('autocast', 'both'):
        if args.autocast_dtypes:
            ac_dtypes = [as_torch_dtype(d) for d in
                         _parse_csv_list(args.autocast_dtypes, 'autocast-dtypes')]
        else:
            ac_dtypes = list(default_autocast_dtypes(args.device))
        runs.append(('autocast', ac_dtypes))

    all_rows = []
    for mode, dtypes in runs:
        print(f'[precision_sweep] mode={mode} dtypes='
              f'{",".join(dtype_name(d) for d in dtypes)} cases={cases} '
              f'seeds={seeds} T={args.length}')
        all_rows.extend(run_sweep(
            cases=cases, dtypes=dtypes, seeds=seeds, seq_len=args.length,
            device=args.device, chunk_size=args.chunk_size, mode=mode))

    summary_rows = summarize(all_rows, budget_factor=args.budget_factor,
                             accum_factor=args.accum_factor)
    print()
    print(format_table(summary_rows))
    print('budget = factor*u for fp16/bf16 (quantization envelope), '
          'accum_factor*T*u for fp32/fp64 (round-off envelope)')

    if args.json:
        metadata = {
            'budget_factor': float(args.budget_factor),
            'accum_factor': float(args.accum_factor),
            'seq_len': int(args.length),
            'chunk_size': int(args.chunk_size),
            'seeds': seeds,
            'device': args.device,
            'modes': [m for m, _ in runs],
            'dtypes': {m: [dtype_name(d) for d in dts] for m, dts in runs},
            'n_evaluations': len(all_rows),
            'n_groups': len(summary_rows),
            'oracle': 'fp64 run of the same (case, path) on the same '
                      'generator draw; the tested input is that draw rounded '
                      'to the tested dtype (explicit) or the fp32 cast '
                      '(autocast)',
            'warnings': [
                'budgets are smoke-level regression envelopes (factor x unit '
                'roundoff), not numerical-error theorems',
                'autocast numbers depend on the torch build/kernels and are '
                'not portable across hosts',
                'CPU fp16 autocast is emulated and excluded by default',
            ],
        }
        _write_json(args.json, summary_rows, metadata)
        print(f'\n[precision_sweep] wrote {args.json}')

    bad = failures(summary_rows)
    if bad:
        print(f'\n[precision_sweep] FAILED: {len(bad)} group(s) out of budget:')
        for line in bad:
            print(f'  - {line}')
        return 1
    print('\n[precision_sweep] PASS: all groups within budget.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
