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
    """Probe environment; on Kaggle+GPU verify CUDA is available.

    P0-3 fix: the previous version called ``setup_kaggle()`` which
    installed the CUDA wheel IN-PROCESS. As documented in
    ``kaggle_setup.setup_kaggle``'s new docstring, that install does
    NOT take effect in the current process (libtorch.so is pinned in
    memory until the process exits), so the first Kaggle run silently
    used CPU.

    ``setup_kaggle()`` now ONLY VERIFIES CUDA availability (it raises
    ``RuntimeError`` if Kaggle+GPU is detected but
    ``torch.cuda.is_available()`` is False). The actual wheel install
    must be done in a separate bootstrap step via
    ``kaggle_setup.bootstrap_kaggle_cuda()`` followed by a kernel
    restart.

    ``SKIP_CUDA_CHECK=1`` bypasses the guard for users who intentionally
    want to run on CPU on a GPU machine (e.g. for debugging).
    """
    from kaggle_setup import setup_kaggle, print_env_summary
    if os.environ.get('SKIP_CUDA_CHECK', '0') == '1':
        print('[run_all] SKIP_CUDA_CHECK=1: bypassing CUDA availability guard.')
    else:
        setup_kaggle(verbose=True)
    info = print_env_summary()
    return info


def _run(name, fn):
    """Run one experiment with timing and error capture.

    Contract for ``fn``'s return value (the P0-2 fix):

    * ``None`` or ``0``  -> success.
    * non-zero int / non-None truthy value -> failure (recorded as
      ``status='fail'`` with the return value in ``error``).

    The previous implementation ignored the return value entirely, so
    ``run_correctness.main()`` — which returns ``1`` when any test fails
    — was silently recorded as ``status='ok'``. Combined with the
    figure-generation swallow (see ``_make_figs`` below), the runner
    could report 8/8 OK on a run that actually had correctness failures
    AND a malformed MQAR JSON. This made the green summary unreliable.

    We deliberately keep the contract permissive (None/0 == success) so
    that existing experiment ``main()`` functions that implicitly return
    ``None`` continue to be treated as success; only callers that
    explicitly opt into the return-code protocol (currently just
    ``run_correctness.main``) are affected.
    """
    print('\n' + '#' * 70)
    print(f'# {name}')
    print('#' * 70)
    t0 = time.time()
    try:
        rc = fn()
    except Exception as e:
        dt = time.time() - t0
        print(f'\n[{name}] FAILED ({dt:.1f}s): {e}')
        traceback.print_exc()
        return {'name': name, 'status': 'fail', 'time_s': dt, 'error': str(e)}
    dt = time.time() - t0
    # Honor the explicit return-code contract. A non-zero / non-None
    # return value signals failure even when no exception was raised.
    if rc is not None and rc != 0:
        msg = f'{name} returned non-zero status: {rc!r}'
        print(f'\n[{name}] FAILED ({dt:.1f}s): {msg}')
        return {'name': name, 'status': 'fail', 'time_s': dt, 'error': msg,
                'return_code': str(rc)}
    print(f'\n[{name}] OK ({dt:.1f}s)')
    return {'name': name, 'status': 'ok', 'time_s': dt}


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dump(allow_nan=False)
    succeeds. Mirrors the helper in run_kv_cache.py / run_decoding.py.

    A single experiment crash that leaves a NaN in the summary (e.g. a
    ``time_s=float('nan')`` from a clock glitch) used to make the entire
    ``summary.json`` write raise ``ValueError: Out of range float values are
    not JSON compliant``, dropping the whole summary on the floor. The
    summary fields are normally finite, but the defensive guard is cheap.

    Delegates to the centralized ``sanitize_for_json`` helper in
    kaggle_setup.py (was a local copy; the wrapper is kept here so
    run_all.py's _run / summary code path that calls ``_sanitize(summary)``
    below continues to work without touching the call sites).
    """
    from kaggle_setup import sanitize_for_json
    return sanitize_for_json(obj)


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
    # Save the caller's CWD so we can restore it in the finally block below.
    # ``os.chdir`` is a process-global side effect: if a notebook calls
    # ``run_all()`` and then writes files relative to their original CWD,
    # those files would silently land in ``out_root`` instead. Restoring
    # the CWD on exit (including on exception) makes run_all() behave as a
    # well-behaved library function rather than a process mutator.
    _orig_cwd = os.getcwd()
    os.chdir(out_root)
    try:
        os.makedirs('results', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        # Tell make_figures.py where to read results and write figures. On Kaggle
        # this is /kaggle/working/{results,figures}, NOT the read-only
        # /kaggle/input/... directory where this script lives. Without these env
        # vars, make_figures.py reads from _ROOT/results (read-only, possibly
        # stale) and tries to write to _ROOT/figures (raising OSError [Errno 30]).
        # On a normal clone, out_root==HERE so the env vars match the defaults
        # already used by make_figures.py and the behavior is unchanged.
        os.environ['RESULTS_DIR'] = os.path.join(out_root, 'results')
        os.environ['FIGURES_DIR'] = os.path.join(out_root, 'figures')

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
            # P4 fix: do NOT reduce ABL_SEEDS below 5 on CPU. The previous
            # override (ABL_SEEDS=3) made the Bonferroni-corrected t-test
            # essentially unachievable (n=3 -> 2 dof -> critical t ≈ 12.9 at
            # corrected alpha=0.007), guaranteeing significant_bonferroni=False
            # for EVERY layout regardless of the true effect size. With 3 seeds
            # the experiment cannot support any structural conclusion, so the
            # "skip slow" shortcut was silently invalidating the entire
            # ablation. We keep the step reduction (50 steps is enough to show
            # the trend) but preserve the seed count at the default (7) so the
            # statistical test retains adequate power. If CPU runtime is a
            # concern, reduce the number of RATIOS or n_kv values instead.
            os.environ['ABL_SEEDS'] = os.environ.get('ABL_SEEDS', '7')
            os.environ['ABL_STEPS'] = '50'
        summary['runs'].append(_run('exp5_ablation', run_ablation.main))

        # 7. Decoding latency — fast (only softmax + KDA).
        summary['runs'].append(_run('exp6_decoding', run_decoding.main))

        # 8. Figures — generate from whatever results exist.
        # The P0-2 fix: the previous ``_make_figs`` swallowed EVERY
        # exception (including programming errors like NameError,
        # AttributeError, KeyError from a refactor, or a malformed-JSON
        # ``json.JSONDecodeError`` that should have been caught upstream
        # but wasn't). The outer ``_run`` therefore ALWAYS recorded
        # ``status='ok'``, so a broken ``make_figures.main`` was
        # invisible in the run-all summary — the user saw 8/8 green.
        #
        # We now distinguish two failure modes:
        #
        # * ``FileNotFoundError`` / ``json.JSONDecodeError``: a result
        #   file is missing or malformed. ``make_figures.load`` already
        #   degrades gracefully for individual figures (returns ``[]``
        #   and logs a skip), so a propagated instance of these means
        #   the figure step as a whole could not even enumerate inputs.
        #   Treat as a soft warning: print and continue, but mark the
        #   step failed in the summary so the user knows the figures
        #   are incomplete.
        #
        # * Any other exception: a programming error. Re-raise so
        #   ``_run`` records ``status='fail'`` with the full traceback
        #   in the summary. The green-report bug is fixed.
        def _make_figs():
            try:
                make_figures.main()
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Soft failure: a result file is missing or malformed.
                # ``make_figures.load`` handles per-figure skips, but a
                # top-level FileNotFoundError means the whole results
                # dir is unreachable. Print a warning and return a
                # non-zero status so ``_run`` records it as a failure
                # (the figure step is incomplete, not "ok").
                print(f'[make_figures] incomplete: {e}')
                traceback.print_exc()
                return 1
            # Any other exception propagates to ``_run``'s except block
            # and is recorded as status='fail'. No more silent swallow.
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
            json.dump(_sanitize(summary), f, indent=2, allow_nan=False)
        print('\nSaved: results/summary.json')

        # P0-2 fix: return the summary AND a non-zero exit code when any
        # run failed. The previous version returned ``None`` implicitly
        # and the ``if __name__ == '__main__'`` block called
        # ``run_all()`` without ``sys.exit``, so even a fully-red
        # summary exited 0 — CI gates that check ``$?`` would pass.
        # We now return the summary dict so programmatic callers
        # (notebook, downstream scripts) can inspect ``n_fail``, and
        # the ``__main__`` block maps ``n_fail > 0`` to ``sys.exit(1)``.
        return summary

    finally:
        # Restore the caller's CWD (saved before os.chdir above) so
        # run_all() does not leave the process in out_root on return.
        # This runs on both clean exit and exception, so a notebook
        # caller never finds itself unexpectedly in /kaggle/working.
        os.chdir(_orig_cwd)


if __name__ == '__main__':
    _summary = run_all()
    # P0-2 fix: propagate failure to the shell. Without this, CI that
    # gates on ``$?`` would pass even when every experiment failed.
    if _summary is not None and _summary.get('n_fail', 0) > 0:
        sys.exit(1)
    sys.exit(0)
