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
  * ``MQAR_SOFTMAX_STEPS`` (default: same as MQAR_STEPS) — optional explicitly labelled softmax long-training sensitivity run.
  * ``ABL_SEEDS``       (default 7)   — seeds for the ablation (run_all(seeds=...) overrides).
  * ``ABL_STEPS``       (default 100) — training steps for the ablation.
  * ``SKIP_SLOW``       (default 0)   — if "1", skip CSA-heavy experiments on CPU.

This script is the reproducibility anchor referenced by the paper's
"Reproducibility" section.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from importlib import metadata as importlib_metadata

# Ensure the experiments directory is on the path when run as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _version_tuple(raw: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a package version for bound checks."""
    parts = re.findall(r"\d+", raw)
    return tuple(int(x) for x in parts[:3]) or (0,)


def _version_in_range(raw: str, lower: tuple[int, ...], upper: tuple[int, ...]) -> bool:
    value = _version_tuple(raw)
    return value >= lower and value < upper


def _ensure_deps():
    """Ensure runtime packages exist within the repository's tested bounds.

    Checking only whether packages import was insufficient:
    Kaggle images can already contain an incompatible version, and the runner
    would silently produce different numerical/figure behavior than the
    pinned environment. Torch is never replaced in-process; an incompatible
    torch version fails loudly with an install/restart instruction.
    """
    bounded = {
        'einops': ('einops>=0.6,<0.9', (0, 6), (0, 9)),
        'matplotlib': ('matplotlib>=3.5,<3.11', (3, 5), (3, 11)),
        'numpy': ('numpy>=1.21,<2.3', (1, 21), (2, 3)),
    }
    for package, (spec, lower, upper) in bounded.items():
        try:
            installed = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            installed = None
        if installed is not None and _version_in_range(installed, lower, upper):
            continue
        if installed is not None:
            if package in sys.modules:
                raise RuntimeError(
                    f'{package}=={installed} is already loaded but outside the '
                    f'tested range {spec}. Install the supported version and '
                    'restart the Python process before running experiments.')
            print(f'[run_all] {package}=={installed} is outside the tested range; '
                  f'installing {spec}...')
        else:
            print(f'[run_all] installing {spec}...')
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', spec,
        ])
        try:
            installed_after = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f'{package} installation completed but metadata is still missing') from exc
        if not _version_in_range(installed_after, lower, upper):
            raise RuntimeError(
                f'{package}=={installed_after} remains outside the tested range '
                f'after installing {spec}')

    # SciPy is optional: run_quality/run_ablation have a documented exact
    # fallback for the required t-distribution calculations. Do not introduce
    # a new network dependency on Kaggle just because SciPy is absent; only
    # reject an installed SciPy version that is outside the tested range.
    try:
        scipy_version = importlib_metadata.version('scipy')
    except importlib_metadata.PackageNotFoundError:
        print('[run_all] scipy is not installed; using the repository statistical fallback.')
    else:
        if not _version_in_range(scipy_version, (1, 7), (1, 15)):
            raise RuntimeError(
                f'scipy=={scipy_version} is outside the tested range >=1.7,<1.15. '
                'Install a supported version or remove it to use the exact fallback.')

    # Do not silently run experiments against an untested torch release. In
    # particular, SDPA/kernel-selection changes can alter the benchmark and
    # quality numbers. If torch is absent, importing kaggle_setup below will
    # produce its own dependency error; report a clearer message here.
    try:
        torch_version = importlib_metadata.version('torch')
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'torch is not installed; install the project dependencies before '
            'running run_all.py') from exc
    if not _version_in_range(torch_version, (2, 2), (2, 7)):
        raise RuntimeError(
            f'torch=={torch_version} is outside the tested range >=2.2,<2.7. '
            'Install a supported torch build and restart the process before '
            'running experiments.')


def _setup():
    """Probe environment; on Kaggle+GPU verify CUDA is available.

    ``setup_kaggle()`` ONLY VERIFIES CUDA availability (it raises
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


# Result files written by each experiment's ``main()``. When an experiment
# fails, its previous run's JSON must not survive to be plotted as if it were
# fresh output (see the figure step below).
_EXPERIMENT_OUTPUTS = {
    'exp1_correctness': ['results/exp1_correctness.json'],
    'exp2_benchmark': ['results/exp2_benchmark.json',
                       'results/exp2_benchmark_provenance.json'],
    'exp3_kv_cache': ['results/exp3_kv_cache.json'],
    'exp4_mqar': ['results/exp4_mqar.json'],
    'exp5_ablation': ['results/exp5_ablation.json',
                      'results/exp5_ablation_provenance.json'],
    'exp6_decoding': ['results/exp6_decoding.json'],
}


def _purge_stale_outputs(name):
    """Remove result JSONs that a failed run would leave behind as stale."""
    for rel in _EXPERIMENT_OUTPUTS.get(name, ()):
        try:
            if os.path.exists(rel):
                os.remove(rel)
        except OSError as e:
            print(f'[run_all] WARNING: could not remove stale {rel}: {e}')


def _run(name, fn):
    """Run one experiment with timing and error capture.

    Contract for ``fn``'s return value:

    * ``None`` or ``0``  -> success.
    * non-zero int / non-None truthy value -> failure (recorded as
      ``status='fail'`` with the return value in ``error``).

    The contract is deliberately permissive (None/0 == success) so
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
        _purge_stale_outputs(name)
        return {'name': name, 'status': 'fail', 'time_s': dt, 'error': str(e)}
    dt = time.time() - t0
    # Honor the explicit return-code contract. A non-zero / non-None
    # return value signals failure even when no exception was raised.
    if rc is not None and rc != 0:
        msg = f'{name} returned non-zero status: {rc!r}'
        print(f'\n[{name}] FAILED ({dt:.1f}s): {msg}')
        _purge_stale_outputs(name)
        return {'name': name, 'status': 'fail', 'time_s': dt, 'error': msg,
                'return_code': str(rc)}
    print(f'\n[{name}] OK ({dt:.1f}s)')
    return {'name': name, 'status': 'ok', 'time_s': dt}


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so json.dump(allow_nan=False)
    succeeds. Mirrors the helper in run_kv_cache.py / run_decoding.py.

    A single experiment crash that leaves a NaN in the summary could
    otherwise make the entire ``summary.json`` write raise
    ``ValueError: Out of range float values are not JSON compliant``,
    dropping the whole summary on the floor. The summary fields are
    normally finite, but the defensive guard is cheap.

    Delegates to the centralized ``sanitize_for_json`` helper in
    kaggle_setup.py (the wrapper is kept here so run_all.py's _run /
    summary code path that calls ``_sanitize(summary)`` below continues
    to work without touching the call sites).
    """
    from kaggle_setup import sanitize_for_json
    return sanitize_for_json(obj)


def run_all(seeds=None, steps=None):
    """Run every experiment in sequence.

    ``seeds`` and ``steps`` override the environment variables if given.
    """
    # Save the env vars we are about to override so they can be
    # restored in the ``finally`` block below. Snapshot the original
    # values and restore them on exit (including on exception), making
    # ``run_all()`` behave as a well-behaved library function rather
    # than a process mutator. Mirrors the CWD-restore pattern already
    # in place.
    _env_keys = ('MQAR_SEEDS', 'ABL_SEEDS', 'MQAR_STEPS', 'ABL_STEPS',
                 'MQAR_SOFTMAX_STEPS', 'BENCH_LENGTHS', 'RESULTS_DIR',
                 'FIGURES_DIR', 'SKIP_SLOW')
    # Snapshot the precise pre-call state of each env var: distinguish
    # "unset" from "set to a value" so the finally block can restore the
    # exact state.
    _orig_env = {k: (os.environ[k] if k in os.environ else None) for k in _env_keys}
    # The env-var mutations below live INSIDE the try-finally block so
    # a failure in ``_ensure_deps()`` / ``_setup()`` does not leak the
    # mutated env vars to the caller's process. We snapshot the
    # originals here (so the finally block can restore them) but DEFER
    # the actual mutations until we are inside the try block (after
    # ``_ensure_deps`` / ``_setup`` succeed). The mutations now live
    # just after ``os.chdir(out_root)`` below.

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
        # Apply the seeds/steps env-var mutations INSIDE the try block
        # (after _ensure_deps/_setup succeeded) so a failure in those
        # functions does not leak the mutations to the caller's
        # process. The finally block at the end of this function
        # restores the originals.
        if seeds is not None:
            os.environ['MQAR_SEEDS'] = str(seeds)
            os.environ['ABL_SEEDS'] = str(seeds)
        if steps is not None:
            os.environ['MQAR_STEPS'] = str(steps)
            # Ablation (Exp 5) sweeps multiple KDA:CSA:HCA ratios, each trained
            # across ABL_SEEDS seeds. Keep the per-run step count synchronized
            # with the MQAR run so the cross-experiment comparison remains fair
            # and the caller's explicit ``steps`` budget is honored uniformly.
            os.environ['ABL_STEPS'] = str(steps)

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

        # Store structured provenance rather than only ``repr(EnvInfo)`` so
        # the summary records git commit, torch/CUDA versions, thread count,
        # and the selected KDA backend. This matters when comparing a
        # reference run with a later FLA run.
        from kaggle_setup import capture_provenance
        summary = {'env': capture_provenance(), 'runs': []}

        # Import after deps are installed.
        import run_correctness
        import run_kv_cache
        import run_benchmark
        import run_quality
        import run_ablation
        import run_decoding
        import method_analysis
        import make_figures

        skip_slow = os.environ.get('SKIP_SLOW', '0').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
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
            # SKIP_SLOW only TRUNCATES (filters out lengths > 512),
            # never EXPANDS. Parse the user's list (or the default
            # {128,256,512,1024,2048} if unset), filter to <= 512, and
            # re-join. If the user's list is already all <= 512, it is
            # unchanged.
            _bl_raw = os.environ.get('BENCH_LENGTHS', '128,256,512,1024,2048')
            try:
                _bl_vals = [int(x.strip()) for x in _bl_raw.split(',')]
            except ValueError:
                _bl_vals = [128, 256, 512, 1024, 2048]
            _bl_truncated = [str(v) for v in _bl_vals if v <= 512]
            if not _bl_truncated:
                # User's list had nothing <= 512; fall back to the safe set.
                _bl_truncated = ['128', '256', '512']
            os.environ['BENCH_LENGTHS'] = ','.join(_bl_truncated)
            summary['runs'].append(_run('exp2_benchmark', run_benchmark.main))
        else:
            summary['runs'].append(_run('exp2_benchmark', run_benchmark.main))

        # 5. MQAR quality — multi-seed. On CPU with CSA this is the slowest.
        if skip_slow and is_cpu:
            print('\n[run_all] SKIP_SLOW=1 on CPU: reducing MQAR to <=100 steps, '
                  'seeds held at >=5 for a valid Bonferroni test.')
            # SKIP_SLOW only TRUNCATES the step budget, never EXPANDS.
            # Use ``min(user_value, safe_ceiling)`` so a user who already
            # set a smaller value keeps it.
            try:
                _mqar_steps = int(os.environ.get('MQAR_STEPS', '200'))
            except ValueError:
                _mqar_steps = 200
            os.environ['MQAR_STEPS'] = str(min(_mqar_steps, 100))
            # Keep the seed count at the default (5) even under SKIP_SLOW.
            # Below 5 seeds the Bonferroni-corrected t-test is essentially
            # unachievable and the experiment cannot support any structural
            # conclusion (the same policy run_ablation applies to ABL_SEEDS
            # below). Preserve/raise the seed count to at least 5 even if the
            # caller passed ``run_all(seeds=3)`` or exported ``MQAR_SEEDS=3``.
            try:
                _mqar_seeds = int(os.environ.get('MQAR_SEEDS', '5'))
            except ValueError:
                _mqar_seeds = 5
            os.environ['MQAR_SEEDS'] = str(max(5, _mqar_seeds))
            # Keep the primary comparison fair even in the reduced CPU run.
            # A longer softmax-only sensitivity run must be requested
            # explicitly by the caller, not silently enabled here.
            #
            # Only override MQAR_SOFTMAX_STEPS if the caller did NOT
            # explicitly set it. Honor the explicit request and only fall
            # back to ``MQAR_STEPS`` (the reduced budget) when the caller
            # left it unset.
            if 'MQAR_SOFTMAX_STEPS' not in os.environ:
                os.environ['MQAR_SOFTMAX_STEPS'] = os.environ['MQAR_STEPS']
        summary['runs'].append(_run('exp4_mqar', run_quality.main))

        # 6. Ablation — multi-seed.
        if skip_slow and is_cpu:
            # Do NOT reduce ABL_SEEDS below 5 on CPU. With fewer than 5
            # seeds the Bonferroni-corrected t-test is essentially
            # unachievable and the experiment cannot support any
            # structural conclusion. We keep the step reduction (50
            # steps is enough to show the trend) but preserve the seed
            # count at the default (7) so the statistical test retains
            # adequate power. Preserve/raise the seed count to at least
            # 5 even if the caller passed ``run_all(seeds=3)`` or
            # exported ``ABL_SEEDS=3``; below 5 seeds the run is
            # knowingly underpowered.
            try:
                _abl_seeds = int(os.environ.get('ABL_SEEDS', '7'))
            except ValueError:
                _abl_seeds = 7
            os.environ['ABL_SEEDS'] = str(max(5, _abl_seeds))
            # Use ``min(user_value, 50)`` so a user who explicitly set
            # ``ABL_STEPS=10`` keeps their faster config. 50 is the
            # ceiling (the documented minimum for showing the ablation
            # trend); below that the user is explicitly opting into an
            # under-trained run.
            try:
                _abl_steps = int(os.environ.get('ABL_STEPS', '100'))
            except ValueError:
                _abl_steps = 100
            os.environ['ABL_STEPS'] = str(min(_abl_steps, 50))
        summary['runs'].append(_run('exp5_ablation', run_ablation.main))

        # 7. Decoding latency — fast (only softmax + KDA).
        summary['runs'].append(_run('exp6_decoding', run_decoding.main))

        # 8. Figures — generate from whatever results exist.
        # Two failure modes:
        #
        # * ``FileNotFoundError`` / ``json.JSONDecodeError``: a result
        #   file is missing or malformed. ``make_figures.load`` degrades
        #   gracefully for individual figures (returns ``[]`` and logs a
        #   skip), so ``make_figures.main()`` never propagates these —
        #   the expected figure outputs are verified below instead.
        #   Treat as a soft warning: print and continue, but mark the
        #   step failed in the summary so the user knows the figures
        #   are incomplete.
        #
        # * Any other exception: a programming error. Re-raise so
        #   ``_run`` records ``status='fail'`` with the full traceback
        #   in the summary.
        def _make_figs():
            try:
                rc = make_figures.main()
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Soft failure: a result file is missing or malformed.
                # Print a warning and return a non-zero status so ``_run``
                # records it as a failure (the figure step is incomplete,
                # not "ok").
                print(f'[make_figures] incomplete: {e}')
                traceback.print_exc()
                return 1
            # Any other exception propagates to ``_run``'s except block
            # and is recorded as status='fail'.
            if rc:
                return rc
            # ``make_figures.main()`` skips (rather than crashes on) a
            # result file that is missing or has empty/malformed data
            # (``load()`` returns ``[]``), so it returns ``None`` even when
            # figures were silently skipped. Verify the expected figure
            # outputs — mirroring ``main()``'s own ``os.path.exists`` guards
            # on the result files — so an incomplete figure set is still
            # recorded as a failure in the summary.
            res_dir = make_figures._RESULTS_DIR
            fig_dir = make_figures._FIGURES_DIR
            expected = []
            if os.path.exists(os.path.join(res_dir, 'exp2_benchmark.json')):
                expected.append('fig_benchmark.pdf')
            if os.path.exists(os.path.join(res_dir, 'exp3_kv_cache.json')):
                expected += ['fig_kv_cache.pdf', 'fig_flops.pdf']
            if os.path.exists(os.path.join(res_dir, 'exp4_mqar.json')):
                expected.append('fig_mqar_nkv1.pdf')
            if os.path.exists(os.path.join(res_dir, 'exp5_ablation.json')):
                expected.append('fig_ablation_nkv1.pdf')
            if os.path.exists(os.path.join(res_dir, 'exp6_decoding.json')):
                expected.append('fig_decoding.pdf')
            expected.append('fig_architecture.pdf')
            missing = [f for f in expected
                       if not os.path.exists(os.path.join(fig_dir, f))]
            if missing:
                print('[make_figures] incomplete; missing figures: '
                      + ', '.join(missing))
                return 1
            return 0
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

        # Use the shared atomic JSON writer (temp file + fsync +
        # os.replace) so a process kill or disk-full mid-write leaves the
        # target file as the OLD version (or absent) rather than a
        # truncated partial JSON document. See
        # kaggle_setup.write_json_atomic's docstring.
        from kaggle_setup import write_json_atomic
        write_json_atomic(_sanitize(summary), 'results/summary.json',
                          indent=2, allow_nan=False)
        print('\nSaved: results/summary.json')

        # Return the summary AND a non-zero exit code when any run
        # failed. Return the summary dict so programmatic callers
        # (notebook, downstream scripts) can inspect ``n_fail``, and
        # the ``__main__`` block maps ``n_fail > 0`` to ``sys.exit(1)``.
        return summary

    finally:
        # Restore the caller's CWD (saved before os.chdir above) so
        # run_all() does not leave the process in out_root on return.
        # This runs on both clean exit and exception, so a notebook
        # caller never finds itself unexpectedly in /kaggle/working.
        os.chdir(_orig_cwd)
        # Restore the env vars we overrode so a notebook caller does
        # not find ``MQAR_SEEDS=5`` etc. permanently set after
        # run_all() returns. Matches the CWD-restore contract:
        # run_all() is a library function, not a process mutator.
        #
        # The snapshot recorded ``None`` for vars that were UNSET
        # before the call (distinct from vars that were set to the
        # literal string ``"None"``). Restoring the precise pre-call
        # state means an internally-injected var (e.g.
        # ``MQAR_SOFTMAX_STEPS`` added by the SKIP_SLOW branch when
        # the caller had not set it) is popped, while a var the
        # caller explicitly set is restored to its original value.
        for _k, _v in _orig_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v


if __name__ == '__main__':
    _summary = run_all()
    # Propagate failure to the shell so CI gates that check ``$?``
    # exit non-zero when at least one experiment failed.
    if _summary is not None and _summary.get('n_fail', 0) > 0:
        sys.exit(1)
    sys.exit(0)
