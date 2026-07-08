"""Regression tests for figure generation edge cases.

These tests verify that ``make_figures.py`` handles edge cases gracefully:
  * Empty result files (no data)
  * All-error result files (every row has an 'error' key)
  * Missing result files (file does not exist)
  * Suptitle positioning (no clipping at the top of the saved image)

The tests are designed to be self-contained: they create temporary result
files, run the figure functions, and assert properties via ``assert``
(raising AssertionError on failure). They do NOT depend on the actual
experiment results.

Run via pytest:
    pytest test_figures.py

Or directly:
    python3 test_figures.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figures  # noqa: E402

RESULTS_DIR = os.path.join(HERE, 'results')
FIGURES_DIR = os.path.join(HERE, 'figures')

# Per-session backup directory. Use mkdtemp so concurrent test runs (e.g.
# CI matrix) do not collide on a shared global path. The previous
# ``BACKUP_DIR = os.path.join(tempfile.gettempdir(), '_fig_test_backups')``
# was a fixed global path; two parallel test runs would overwrite each
# other's backups.
_BACKUP_DIR = tempfile.mkdtemp(prefix='_fig_test_backups_')


def _backup_results():
    """Back up all existing result files AND figures to a temp dir.

    Previously only ``results/`` was backed up, but the tests also write
    into ``figures/`` (via ``make_figures.fig_*``). After a test run the
    real figures were silently overwritten by synthetic test output.
    """
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    for src_dir, name in [(RESULTS_DIR, 'results'), (FIGURES_DIR, 'figures')]:
        if not os.path.isdir(src_dir):
            continue
        dst = os.path.join(_BACKUP_DIR, name)
        os.makedirs(dst, exist_ok=True)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(dst, fname))


def _restore_results():
    """Restore result files AND figures from the backup."""
    for dst_dir, name in [(RESULTS_DIR, 'results'), (FIGURES_DIR, 'figures')]:
        src_dir = os.path.join(_BACKUP_DIR, name)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(dst_dir, fname))


def _write_results(name, data):
    """Write a synthetic results JSON file into the real results/ dir.

    The test must run ``make_figures.fig_*`` against the same directory
    ``make_figures.load`` reads from (``_RESULTS_DIR`` inside
    ``make_figures.py``, which equals ``HERE/results``). So we have to
    write into the real ``results/`` dir — but the autouse fixture below
    backs it up and restores it after the session.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), 'w') as f:
        json.dump(data, f)


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


# --- pytest integration ---------------------------------------------------
# When run via pytest (``pytest test_figures.py``), the test_* functions
# are picked up by name. Without this autouse fixture, pytest would NOT
# call ``main()``, so the backup/restore in ``main()`` would never run,
# and the synthetic test data written by ``_write_results`` would
# silently OVERWRITE the real ``results/exp2_benchmark.json`` (and
# ``figures/fig_*.png/pdf``) — corrupting committed experiment data.

try:
    import pytest

    @pytest.fixture(autouse=True, scope='session')
    def _backup_restore_fixture():
        """Session-scoped autouse fixture: back up real results/figures
        before any test runs, and restore them after the session ends."""
        _backup_results()
        yield
        _restore_results()
        # Best-effort cleanup of the backup dir.
        try:
            shutil.rmtree(_BACKUP_DIR)
        except OSError:
            pass

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
    # The ``main()`` function below handles backup/restore manually.
    pass


# --- tests ---------------------------------------------------------------

def test_fig_benchmark_empty_data():
    """fig_benchmark with empty data should not crash or emit legend warning."""
    print("\nTest: fig_benchmark with empty data")
    _write_results('exp2_benchmark.json', [])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Let exceptions propagate — pytest (or main's try/finally) will
        # surface the traceback. The previous bare ``except Exception``
        # swallowed real programming errors (NameError, AttributeError
        # from a refactor) and only printed them, making debugging
        # nearly impossible.
        make_figures.fig_benchmark()
        legend_warned = any('No artists with labels' in str(wm.message)
                            for wm in w)
        _ok('no crash + no legend warning',
            not legend_warned,
            f'legend_warning={legend_warned}')


def test_fig_benchmark_all_errors():
    """fig_benchmark with all-error data should not crash or warn."""
    print("\nTest: fig_benchmark with all-error data")
    _write_results('exp2_benchmark.json', [
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


def test_plot_ablation_group_empty_data():
    """_plot_ablation_group with empty records should not crash.

    Previously this raised ``ValueError: max() iterable argument is empty``
    from ``max(max(a + c for a, c in zip(accs, acc_cis)) * 1.3, 0.2)``
    when accs was empty.
    """
    print("\nTest: _plot_ablation_group with empty data")
    make_figures._plot_ablation_group([], 1, False)
    _ok('no crash on empty data', True, '')


def test_plot_ablation_group_all_errors():
    """_plot_ablation_group with all-error records should plot ERR bars."""
    print("\nTest: _plot_ablation_group with all-error data")
    records = [
        {'ratio': '3:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 5, 'n_params': None},
        {'ratio': '4:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 6, 'n_params': None},
    ]
    # Delete the figure file BEFORE the call so the existence check
    # cannot pass on a stale file from a prior run (the previous test
    # silently passed because the committed ``figures/fig_ablation_nkv1.png``
    # already existed, regardless of whether the function actually wrote it).
    fig_path = os.path.join(FIGURES_DIR, 'fig_ablation_nkv1.png')
    if os.path.exists(fig_path):
        os.remove(fig_path)
    make_figures._plot_ablation_group(records, 1, False)
    saved = os.path.exists(fig_path)
    _ok('all-error data plotted as ERR bars', saved,
        f'figure_saved={saved}')


def test_fig_ablation_suptitle_not_clipped():
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
    # Capture mtime BEFORE the call so we can assert the figure was
    # actually rewritten (not just read from a stale prior-run file).
    fig_path = os.path.join(FIGURES_DIR, 'fig_ablation_nkv1.png')
    mtime_before = os.path.getmtime(fig_path) if os.path.exists(fig_path) else 0
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


def main():
    print('=' * 70)
    print('Figure Generation Regression Tests')
    print('=' * 70)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    _backup_results()
    results = []
    try:
        # Each test function now asserts internally; we still capture
        # bool returns for the summary printout (and to keep running
        # subsequent tests even if one asserts).
        for fn in [
            test_fig_benchmark_empty_data,
            test_fig_benchmark_all_errors,
            test_plot_ablation_group_empty_data,
            test_plot_ablation_group_all_errors,
            test_fig_ablation_suptitle_not_clipped,
        ]:
            try:
                fn()
                results.append(True)
            except AssertionError:
                results.append(False)
            except Exception as e:
                # Unexpected exception (not an assertion failure). Surface
                # it loudly but keep going so the user sees all failures.
                print(f"  [CRASH] {fn.__name__}: {type(e).__name__}: {e}")
                results.append(False)
    finally:
        _restore_results()
        try:
            shutil.rmtree(_BACKUP_DIR)
        except OSError:
            pass
    n_pass = sum(1 for r in results if r)
    n_total = len(results)
    print('-' * 70)
    print(f'Total: {n_pass}/{n_total} passed')
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
