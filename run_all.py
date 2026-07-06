"""Single-entry-point runner for all experiments — Kaggle notebook friendly.

Usage on Kaggle
---------------
1. Upload this whole ``experiments/`` directory as a Kaggle Dataset (or clone
   the repo into the notebook working directory).
2. In a notebook cell::

     !pip install -q einops matplotlib
     import sys; sys.path.insert(0, '/kaggle/input/<your-dataset-name>/experiments')
     %run /kaggle/input/<your-dataset-name>/experiments/run_all.py

   Or, to run individual experiments::

     from run_all import run_all
     run_all(seeds=5, steps=200)

3. All results are written to ``results/`` and figures to ``figures/``
   relative to the current working directory. On Kaggle, because
   ``/kaggle/input`` is read-only, outputs are redirected to
   ``/kaggle/working/results`` and ``/kaggle/working/figures``.

What this runner does
---------------------
  * Installs the minimal deps (einops) if missing.
  * Calls ``setup_kaggle()`` to install the CUDA torch wheel on Kaggle T4.
  * Prints an environment summary.
  * Runs all six experiments + method analysis + figure generation.
  * Saves a combined ``results/summary.json`` with pass/fail and key numbers.

Environment knobs (set before importing / via ``os.environ``):
  * ``MQAR_SEEDS``      (default 5)   — seeds for the MQAR experiment.
  * ``MQAR_STEPS``      (default 200) — training steps for non-softmax ops.
  * ``MQAR_SOFTMAX_STEPS`` (default 500) — extra steps for the softmax baseline.
  * ``ABL_SEEDS``       (default 5)   — seeds for the ablation.
  * ``ABL_STEPS``       (default 100) — training steps for the ablation.
  * ``SKIP_SLOW``       (default 0)   — if "1", skip CSA-heavy experiments on CPU.

This script is the reproducibility anchor referenced by the paper's
"Reproducibility" section.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

# Ensure the experiments directory is on the path when run as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _ensure_deps():
    """Install einops if missing (Kaggle's default env may not have it)."""
    try:
        import einops  # noqa: F401
    except ImportError:
        print('[run_all] installing einops...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'einops'])
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print('[run_all] installing matplotlib...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib'])


def _setup():
    """Probe environment, install CUDA torch on Kaggle if needed."""
    from kaggle_setup import setup_kaggle, print_env_summary, configure_torch_for_device
    setup_kaggle(verbose=True)
    info = print_env_summary()
    return info


def _run(name, fn):
    """Run one experiment with timing and error capture."""
    print('\n' + '#' * 70)
    print(f'# {name}')
    print('#' * 70)
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f'\n[{name}] OK ({dt:.1f}s)')
        return {'name': name, 'status': 'ok', 'time_s': dt}
    except Exception as e:
        dt = time.time() - t0
        print(f'\n[{name}] FAILED ({dt:.1f}s): {e}')
        traceback.print_exc()
        return {'name': name, 'status': 'fail', 'time_s': dt, 'error': str(e)}


def run_all(seeds=None, steps=None):
    """Run every experiment in sequence.

    ``seeds`` and ``steps`` override the environment variables if given.
    """
    if seeds is not None:
        os.environ['MQAR_SEEDS'] = str(seeds)
        os.environ['ABL_SEEDS'] = str(seeds)
    if steps is not None:
        os.environ['MQAR_STEPS'] = str(steps)
        # Ablation (Exp 5) sweeps multiple KDA:CSA:HCA ratios, each trained
        # across ABL_SEEDS seeds. Total cost is n_ratios * n_seeds * steps,
        # far larger than the single-track MQAR run, so halve the per-run
        # step count (floored at 50 — the ablation doc's convergence minimum)
        # to keep wall-clock tractable without regressing to the old 25-step
        # under-trained regime.
        os.environ['ABL_STEPS'] = str(max(50, steps // 2))

    _ensure_deps()
    info = _setup()

    # Choose a writable output directory.
    # On Kaggle the script lives under /kaggle/input/... which is a read-only
    # mount, so writing results/figures there fails with OSError [Errno 30].
    # Fall back to /kaggle/working (Kaggle's writable output dir) in that case.
    if os.access(HERE, os.W_OK):
        out_root = HERE
    else:
        out_root = os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working')
        os.makedirs(out_root, exist_ok=True)
        # HERE is already on sys.path (module load), so imports still work
        # after we chdir away from it.
        print(f'[run_all] script dir is read-only; writing outputs to {out_root}')
    os.chdir(out_root)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    summary = {'env': repr(info), 'runs': []}

    # Import after deps are installed.
    import run_correctness
    import run_kv_cache
    import run_benchmark
    import run_quality
    import run_ablation
    import run_decoding
    import method_analysis
    import make_figures

    skip_slow = os.environ.get('SKIP_SLOW', '0') == '1'
    is_cpu = not info.has_gpu

    # 1. Correctness — always run (fast, ~seconds).
    summary['runs'].append(_run('exp1_correctness', run_correctness.main))

    # 2. KV cache analysis — pure arithmetic, always run.
    summary['runs'].append(_run('exp3_kv_cache', run_kv_cache.main))

    # 3. Method analysis (formulas + headwise demo) — always run.
    summary['runs'].append(_run('method_analysis', method_analysis.main))

    # 4. Latency benchmark — on CPU the CSA/HCA Python loops are slow at T=2048.
    #    Skip the largest lengths on CPU if SKIP_SLOW is set.
    if skip_slow and is_cpu:
        print('\n[run_all] SKIP_SLOW=1 on CPU: truncating benchmark lengths.')
        # run_benchmark.main() reads BENCH_LENGTHS (comma-separated) and
        # falls back to the full sweep {128,256,512,1024,2048} when unset.
        os.environ['BENCH_LENGTHS'] = '128,256,512'
        summary['runs'].append(_run('exp2_benchmark', run_benchmark.main))
    else:
        summary['runs'].append(_run('exp2_benchmark', run_benchmark.main))

    # 5. MQAR quality — multi-seed. On CPU with CSA this is the slowest.
    if skip_slow and is_cpu:
        print('\n[run_all] SKIP_SLOW=1 on CPU: reducing MQAR to 3 seeds / 100 steps.')
        # NOTE: use direct assignment, NOT ``setdefault``. The earlier block at
        # the top of ``run_all`` already set ``MQAR_SEEDS`` / ``MQAR_STEPS`` via
        # direct assignment from the ``seeds`` / ``steps`` parameters, so
        # ``setdefault`` here is a no-op and the reduction never happens —
        # the log message lies and the full 5-seed / 200-step run is launched,
        # defeating the whole point of SKIP_SLOW on CPU.
        os.environ['MQAR_SEEDS'] = '3'
        os.environ['MQAR_STEPS'] = '100'
        os.environ['MQAR_SOFTMAX_STEPS'] = '200'
    summary['runs'].append(_run('exp4_mqar', run_quality.main))

    # 6. Ablation — multi-seed.
    if skip_slow and is_cpu:
        # Same direct-assignment fix as above (``setdefault`` is a no-op
        # because ``ABL_SEEDS`` / ``ABL_STEPS`` were already set above).
        os.environ['ABL_SEEDS'] = '3'
        os.environ['ABL_STEPS'] = '50'
    summary['runs'].append(_run('exp5_ablation', run_ablation.main))

    # 7. Decoding latency — fast (only softmax + KDA).
    summary['runs'].append(_run('exp6_decoding', run_decoding.main))

    # 8. Figures — generate from whatever results exist.
    def _make_figs():
        try:
            make_figures.main()
        except Exception as e:
            # Figures are best-effort; a missing result file shouldn't fail the run.
            print(f'[make_figures] partial: {e}')
    summary['runs'].append(_run('make_figures', _make_figs))

    # Final summary.
    n_ok = sum(1 for r in summary['runs'] if r['status'] == 'ok')
    n_fail = sum(1 for r in summary['runs'] if r['status'] == 'fail')
    total_t = sum(r['time_s'] for r in summary['runs'])
    summary['n_ok'] = n_ok
    summary['n_fail'] = n_fail
    summary['total_time_s'] = total_t

    print('\n' + '=' * 70)
    print('Run-all summary')
    print('=' * 70)
    for r in summary['runs']:
        print(f"  {r['status'].upper():>4}  {r['name']:<24}  {r['time_s']:>8.1f}s")
    print('-' * 70)
    print(f'  {n_ok} ok, {n_fail} failed, total {total_t:.1f}s')

    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSaved: results/summary.json')


if __name__ == '__main__':
    run_all()
