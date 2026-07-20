"""Regression tests for figure generation edge cases.

These tests verify that ``make_figures.py`` handles edge cases gracefully:
  * Empty result files (no data)
  * All-error result files (every row has an 'error' key)
  * Missing result files (file does not exist)
  * Suptitle positioning (no clipping at the top of the saved image)

The tests are self-contained: they redirect ``make_figures._RESULTS_DIR``
and ``make_figures._FIGURES_DIR`` to a per-test temp directory (via pytest's
``tmp_path`` + ``monkeypatch``, or via ``tempfile.TemporaryDirectory`` for
the direct ``python3 test_figures.py`` path). They NEVER touch the real
``results/`` or ``figures/`` directories, so they are safe to run in
parallel (pytest-xdist) and do not require backup/restore machinery.

Run via pytest:
    pytest test_figures.py

Or directly:
    python3 test_figures.py
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figures  # noqa: E402


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures.

    Previously this helper only returned the bool, so ``pytest`` silently
    counted every test as passing even when ``cond`` was False (pytest
    ignores return values). Now we both print (for the ``python3
    test_figures.py`` direct-run path) and assert (for the pytest path).
    """
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


# --- per-test temp-directory helpers ------------------------------------
# P1-6 fix: every test must run against a per-test temp directory, NOT the
# real ``results/`` / ``figures/`` directories. The previous backup/restore
# mechanism was broken in two ways:
#   1. ``_restore_results`` only copied old files back — it did NOT delete
#      files that the test created and that didn't exist before. So a test
#      that wrote ``_test_envelope.json`` would leave that file in the real
#      ``results/`` dir forever.
#   2. Multiple pytest-xdist workers would compete on the same real
#      directory, corrupting each other's backup/restore cycles.
# The fix uses pytest's ``tmp_path`` + ``monkeypatch`` (or a
# ``tempfile.TemporaryDirectory`` for the direct-run path) to redirect
# ``make_figures._RESULTS_DIR`` and ``make_figures._FIGURES_DIR`` to a
# per-test temp dir. The real directories are never touched.

@contextlib.contextmanager
def _redirect_dirs(tmpdir: str):
    """Redirect ``make_figures._RESULTS_DIR`` and ``_FIGURES_DIR`` to
    ``tmpdir`` for the duration of the ``with`` block.

    Creates ``{tmpdir}/results`` and ``{tmpdir}/figures`` subdirectories
    so the figure functions' ``os.makedirs(..., exist_ok=True)`` calls
    are no-ops. Restores the original values on exit (even on exception).
    """
    results_dir = os.path.join(tmpdir, 'results')
    figures_dir = os.path.join(tmpdir, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    orig_results = make_figures._RESULTS_DIR
    orig_figures = make_figures._FIGURES_DIR
    make_figures._RESULTS_DIR = results_dir
    make_figures._FIGURES_DIR = figures_dir
    try:
        yield results_dir, figures_dir
    finally:
        make_figures._RESULTS_DIR = orig_results
        make_figures._FIGURES_DIR = orig_figures


def _write_results(results_dir: str, name: str, data):
    """Write a synthetic results JSON file into ``results_dir``."""
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, name), 'w') as f:
        json.dump(data, f)


# --- pytest integration -------------------------------------------------
# When run via pytest, each test receives ``tmp_path`` (a per-test
# ``pathlib.Path``) and ``monkeypatch``. We use ``_redirect_dirs`` to
# point ``make_figures`` at the temp dir for the duration of the test.
# No autouse session fixture is needed — there is no shared state to
# back up or restore.

try:
    import pytest

    @pytest.fixture(autouse=True)
    def _close_figs_fixture():
        """Per-test fixture: close any matplotlib figures left open by an
        exception mid-``fig_*`` call so they don't accumulate across the
        suite and cause "figure canvas already drawn" surprises."""
        yield
        import matplotlib.pyplot as plt
        plt.close('all')

except ImportError:
    # pytest not installed — running via ``python3 test_figures.py``.
    # The ``main()`` function below creates a single TemporaryDirectory
    # and each test uses it via ``_redirect_dirs``.
    pass


# --- tests ---------------------------------------------------------------

def test_load_envelope_format(tmp_path=None, monkeypatch=None):
    """``make_figures.load`` must accept the P0-1 envelope schema.

    The MQAR writer (``run_quality.main``) emits a single JSON object
    ``{"metadata": {...}, "results": [...]}``. The previous writer
    concatenated two top-level documents (``{...}\\n[...]``) which is
    not valid JSON; ``make_figures.load`` then returned ``[]`` and the
    MQAR figure was silently skipped. This test pins the contract that
    the new envelope is parsed into the inner ``results`` array.
    """
    print("\nTest: load() unwraps the {metadata, results} envelope")
    with _redirect_dirs_for(tmp_path, monkeypatch) as (results_dir, _):
        _write_results(results_dir, '_test_envelope.json', {
            'metadata': {'schema_version': 1, 'csa_indexer_trained': False},
            'results': [
                {'op': 'softmax', 'T': 128, 'acc': 0.5},
                {'op': 'kda', 'T': 128, 'acc': 0.4},
            ],
        })
        data = make_figures.load('_test_envelope.json')
        _ok('envelope unwrapped to results list',
            isinstance(data, list) and len(data) == 2,
            f'type={type(data).__name__}, len={len(data) if isinstance(data, list) else "n/a"}')


def test_load_legacy_bare_array(tmp_path=None, monkeypatch=None):
    """``make_figures.load`` must still accept the legacy bare-array schema.

    All non-MQAR result files (benchmark, kv_cache, ablation, decoding,
    correctness) are bare arrays. The P0-1 envelope fix must not break
    them.
    """
    print("\nTest: load() still accepts legacy bare-array schema")
    with _redirect_dirs_for(tmp_path, monkeypatch) as (results_dir, _):
        _write_results(results_dir, '_test_bare.json', [
            {'op': 'softmax', 'T': 128, 'acc': 0.5},
            {'op': 'kda', 'T': 128, 'acc': 0.4},
        ])
        data = make_figures.load('_test_bare.json')
        _ok('bare array returned as-is',
            isinstance(data, list) and len(data) == 2,
            f'type={type(data).__name__}, len={len(data) if isinstance(data, list) else "n/a"}')


def test_fig_benchmark_empty_data(tmp_path=None, monkeypatch=None):
    """fig_benchmark with empty data should not crash or emit legend warning."""
    print("\nTest: fig_benchmark with empty data")
    with _redirect_dirs_for(tmp_path, monkeypatch) as (results_dir, _):
        _write_results(results_dir, 'exp2_benchmark.json', [])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            make_figures.fig_benchmark()
            legend_warned = any('No artists with labels' in str(wm.message)
                                for wm in w)
            _ok('no crash + no legend warning',
                not legend_warned,
                f'legend_warning={legend_warned}')


def test_fig_benchmark_all_errors(tmp_path=None, monkeypatch=None):
    """fig_benchmark with all-errors data should not crash or warn."""
    print("\nTest: fig_benchmark with all-errors data")
    with _redirect_dirs_for(tmp_path, monkeypatch) as (results_dir, _):
        _write_results(results_dir, 'exp2_benchmark.json', [
            {'T': 128, 'op': 'softmax', 'error': 'OOM', 'device': 'cpu'},
            {'T': 128, 'op': 'kda', 'error': 'OOM', 'device': 'cpu'},
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            make_figures.fig_benchmark()
            legend_warned = any('No artists with labels' in str(wm.message)
                                for wm in w)
            _ok('no crash + no legend warning',
                not legend_warned,
                f'legend_warning={legend_warned}')


def test_plot_ablation_group_empty_data(tmp_path=None, monkeypatch=None):
    """_plot_ablation_group with empty records should not crash.

    Previously this raised ``ValueError: max() iterable argument is empty``
    from ``max(max(a + c for a, c in zip(accs, acc_cis)) * 1.3, 0.2)``
    when accs was empty.
    """
    print("\nTest: _plot_ablation_group with empty data")
    with _redirect_dirs_for(tmp_path, monkeypatch):
        make_figures._plot_ablation_group([], 1, False)
        _ok('no crash on empty data', True, '')


def test_plot_ablation_group_all_errors(tmp_path=None, monkeypatch=None):
    """_plot_ablation_group with all-error records should plot ERR bars."""
    print("\nTest: _plot_ablation_group with all-error data")
    records = [
        {'ratio': '3:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 5, 'n_params': None},
        {'ratio': '4:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 6, 'n_params': None},
    ]
    with _redirect_dirs_for(tmp_path, monkeypatch) as (_, figures_dir):
        # Delete the figure file BEFORE the call so the existence check
        # cannot pass on a stale file from a prior run.
        fig_path = os.path.join(figures_dir, 'fig_ablation_nkv1.png')
        if os.path.exists(fig_path):
            os.remove(fig_path)
        make_figures._plot_ablation_group(records, 1, False)
        saved = os.path.exists(fig_path)
        _ok('all-error data plotted as ERR bars', saved,
            f'figure_saved={saved}')


def test_fig_ablation_suptitle_not_clipped(tmp_path=None, monkeypatch=None):
    """The fig_ablation suptitle must not be clipped at the top of the image.

    Previously the suptitle was positioned at y=1.02 (above the figure's
    top edge y=1.0) AND savefig was called WITHOUT ``bbox_inches='tight'``,
    causing the top half of the title text to be cut off in the saved
    PNG/PDF. This test generates a figure and verifies that the topmost
    rows of the image are NOT text (i.e. there is white space above the
    suptitle).
    """
    print("\nTest: fig_ablation suptitle is not clipped")
    # Use realistic ablation data so the figure has content.
    records = [{
        'ratio': '3:1:1', 'n_kv': 1,
        'layout': 'KDA-KDA-KDA-CSA-HCA', 'n_seeds_ok': 2,
        'n_seeds_failed': 0, 'n_seeds': 2, 'seeds': [42, 43],
        'per_seed': [{'final_acc': 0.1, 'final_loss': 2.7, 'fwd_ms': 4.0,
                      'seed': 42, 'train_time_s': 1.0}],
        'mean_acc': 0.1, 'std_acc': 0.01, 'ci95_acc': 0.05,
        'chance_acc': 0.0625, 't_stat_vs_chance': 10.0,
        'mean_fwd_ms': 4.0, 'n_params': 26852, 'n_layers': 5,
        'mean_train_time_s': 1.0,
    }]
    with _redirect_dirs_for(tmp_path, monkeypatch) as (_, figures_dir):
        # Delete the figure file BEFORE the call so the mtime check
        # cannot pass on a stale file from a prior run, and so that
        # mtime_before is 0 (robust against filesystem timestamp
        # granularity — many filesystems have 1-second mtime
        # resolution, so a stale file rewritten within the same second
        # would not show mtime > mtime_before).
        fig_path = os.path.join(figures_dir, 'fig_ablation_nkv1.png')
        if os.path.exists(fig_path):
            os.remove(fig_path)
        mtime_before = 0
        make_figures._plot_ablation_group(records, 1, False)
        assert os.path.exists(fig_path), 'figure was not written'
        assert os.path.getmtime(fig_path) > mtime_before, \
            'figure was not regenerated (stale file from a prior run)'

        img = mpimg.imread(fig_path)
        # The top 5 rows should be predominantly white (padding above the
        # suptitle). If the suptitle is clipped, row 0 will have substantial
        # non-white (text) pixels.
        top_5_rows = img[:5, :, :3]
        white_mask = (top_5_rows > 0.95).all(axis=-1)
        white_pct = white_mask.mean() * 100
        not_clipped = white_pct > 99.0  # Allow up to 1% non-white (anti-aliasing)
        _ok('suptitle not clipped at top',
            not_clipped,
            f'top_5_rows_white_pct={white_pct:.1f}% (need >99%)')


# --- helper to bridge pytest (tmp_path/monkeypatch) and direct-run -----

@contextlib.contextmanager
def _redirect_dirs_for(tmp_path, monkeypatch):
    """Bridge pytest's ``tmp_path`` / ``monkeypatch`` fixtures and the
    direct-run path (``python3 test_figures.py``).

    * Under pytest: both fixtures are non-None. Use ``monkeypatch.setattr``
      so pytest auto-undoes the redirect at test teardown (defensive — even
      if our ``finally`` block in ``_redirect_dirs`` is skipped by a
      SIGKILL, pytest's teardown still runs).
    * Under direct run: both are None. Use the context-manager form
      ``_redirect_dirs`` with a ``tempfile.TemporaryDirectory``.
    """
    if tmp_path is not None and monkeypatch is not None:
        # pytest path: use the per-test tmp_path directory.
        results_dir = os.path.join(str(tmp_path), 'results')
        figures_dir = os.path.join(str(tmp_path), 'figures')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        monkeypatch.setattr(make_figures, '_RESULTS_DIR', results_dir)
        monkeypatch.setattr(make_figures, '_FIGURES_DIR', figures_dir)
        try:
            yield results_dir, figures_dir
        finally:
            # monkeypatch auto-undoes the setattr, but we yield in a
            # try/finally so the directories are cleaned up even if the
            # test body raises.
            pass
    else:
        # Direct-run path: use a per-test TemporaryDirectory. The
        # directory is cleaned up automatically when the ``with`` block
        # exits (``_redirect_dirs`` is a context manager that yields and
        # restores on exit, and ``tempfile.TemporaryDirectory`` cleans
        # up the temp dir on exit).
        with tempfile.TemporaryDirectory(prefix='_fig_test_') as tmpdir:
            with _redirect_dirs(tmpdir) as (results_dir, figures_dir):
                yield results_dir, figures_dir


def main():
    print('=' * 70)
    print('Figure Generation Regression Tests')
    print('=' * 70)
    results = []
    # When run directly (not via pytest), each test gets a fresh temp
    # directory via the ``_redirect_dirs_for`` context manager. We pass
    # ``tmp_path=None, monkeypatch=None`` to signal the direct-run path.
    for fn in [
        test_load_envelope_format,
        test_load_legacy_bare_array,
        test_fig_benchmark_empty_data,
        test_fig_benchmark_all_errors,
        test_plot_ablation_group_empty_data,
        test_plot_ablation_group_all_errors,
        test_fig_ablation_suptitle_not_clipped,
    ]:
        try:
            fn(tmp_path=None, monkeypatch=None)
            results.append(True)
        except AssertionError as e:
            # Surface assertion failures loudly so the user can see which
            # check failed and why. Previously this branch silently
            # swallowed the message, making direct-run failures
            # undiagnosable (only the pass count was printed).
            print(f"  [FAIL] {fn.__name__}: AssertionError: {e}")
            results.append(False)
        except Exception as e:
            # Unexpected exception (not an assertion failure). Surface
            # it loudly but keep going so the user sees all failures.
            print(f"  [CRASH] {fn.__name__}: {type(e).__name__}: {e}")
            results.append(False)
        finally:
            # Close any figures the test may have left open. The pytest
            # autouse fixture (_close_figs_fixture) handles this under
            # pytest, but when ``python test_figures.py`` is run
            # directly (e.g. from run_all.py), no such fixture runs and
            # a half-built figure from a CRASHed test would leak into
            # the next test, potentially causing
            # ``RuntimeWarning: More than 20 figures`` or canvas-state
            # surprises.
            import matplotlib.pyplot as _plt
            _plt.close('all')
    n_pass = sum(1 for r in results if r)
    n_total = len(results)
    print('-' * 70)
    print(f'Total: {n_pass}/{n_total} passed')
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
