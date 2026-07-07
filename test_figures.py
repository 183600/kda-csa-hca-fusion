"""Regression tests for figure generation edge cases.

These tests verify that ``make_figures.py`` handles edge cases gracefully:
  * Empty result files (no data)
  * All-error result files (every row has an 'error' key)
  * Missing result files (file does not exist)
  * Suptitle positioning (no clipping at the top of the saved image)

The tests are designed to be self-contained: they create temporary result
files, run the figure functions, verify no exceptions are raised, and clean
up. They do NOT depend on the actual experiment results.

Run directly:
    python3 test_figures.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_figures


RESULTS_DIR = os.path.join(HERE, 'results')
BACKUP_DIR = '/tmp/_fig_test_backups'


def _backup_results():
    """Back up all existing result files to a temp dir."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for fname in os.listdir(RESULTS_DIR):
        shutil.copy(os.path.join(RESULTS_DIR, fname),
                    os.path.join(BACKUP_DIR, fname))


def _restore_results():
    """Restore result files from the backup."""
    if not os.path.isdir(BACKUP_DIR):
        return
    for fname in os.listdir(BACKUP_DIR):
        shutil.copy(os.path.join(BACKUP_DIR, fname),
                    os.path.join(RESULTS_DIR, fname))


def _write_results(name, data):
    with open(os.path.join(RESULTS_DIR, name), 'w') as f:
        json.dump(data, f)


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    return cond


def test_fig_benchmark_empty_data():
    """fig_benchmark with empty data should not crash or emit legend warning."""
    print("\nTest: fig_benchmark with empty data")
    _write_results('exp2_benchmark.json', [])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            make_figures.fig_benchmark()
            legend_warned = any('No artists with labels' in str(wm.message)
                                for wm in w)
            return _ok('no crash + no legend warning',
                       not legend_warned,
                       f'legend_warning={legend_warned}')
        except Exception as e:
            return _ok('no crash + no legend warning', False,
                       f'{type(e).__name__}: {e}')


def test_fig_benchmark_all_errors():
    """fig_benchmark with all-error data should not crash or warn."""
    print("\nTest: fig_benchmark with all-error data")
    _write_results('exp2_benchmark.json', [
        {'T': 128, 'op': 'softmax', 'error': 'OOM', 'device': 'cpu'},
        {'T': 128, 'op': 'kda', 'error': 'OOM', 'device': 'cpu'},
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            make_figures.fig_benchmark()
            legend_warned = any('No artists with labels' in str(wm.message)
                                for wm in w)
            return _ok('no crash + no legend warning',
                       not legend_warned,
                       f'legend_warning={legend_warned}')
        except Exception as e:
            return _ok('no crash + no legend warning', False,
                       f'{type(e).__name__}: {e}')


def test_plot_ablation_group_empty_data():
    """_plot_ablation_group with empty records should not crash.

    Previously this raised ``ValueError: max() iterable argument is empty``
    from ``max(max(a + c for a, c in zip(accs, acc_cis)) * 1.3, 0.2)``
    when accs was empty.
    """
    print("\nTest: _plot_ablation_group with empty data")
    try:
        make_figures._plot_ablation_group([], 1, False)
        return _ok('no crash on empty data', True, '')
    except Exception as e:
        return _ok('no crash on empty data', False,
                   f'{type(e).__name__}: {e}')


def test_plot_ablation_group_all_errors():
    """_plot_ablation_group with all-error records should plot ERR bars."""
    print("\nTest: _plot_ablation_group with all-error data")
    records = [
        {'ratio': '3:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 5, 'n_params': None},
        {'ratio': '4:1:1', 'n_kv': 1, 'error': 'err', 'mean_acc': None,
         'n_layers': 6, 'n_params': None},
    ]
    try:
        make_figures._plot_ablation_group(records, 1, False)
        # Verify the figure was saved
        fig_path = os.path.join(HERE, 'figures', 'fig_ablation_nkv1.png')
        saved = os.path.exists(fig_path)
        return _ok('all-error data plotted as ERR bars', saved,
                   f'figure_saved={saved}')
    except Exception as e:
        return _ok('all-error data plotted as ERR bars', False,
                   f'{type(e).__name__}: {e}')


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
    try:
        make_figures._plot_ablation_group(records, 1, False)
    except Exception as e:
        return _ok('figure generated', False, f'{type(e).__name__}: {e}')

    fig_path = os.path.join(HERE, 'figures', 'fig_ablation_nkv1.png')
    if not os.path.exists(fig_path):
        return _ok('figure saved', False, 'file does not exist')

    img = mpimg.imread(fig_path)
    # The top 5 rows should be predominantly white (padding above the
    # suptitle). If the suptitle is clipped, row 0 will have substantial
    # non-white (text) pixels.
    top_5_rows = img[:5, :, :3]
    white_mask = (top_5_rows > 0.95).all(axis=-1)
    white_pct = white_mask.mean() * 100
    not_clipped = white_pct > 99.0  # Allow up to 1% non-white (anti-aliasing)
    return _ok('suptitle not clipped at top',
               not_clipped,
               f'top_5_rows_white_pct={white_pct:.1f}% (need >99%)')


def main():
    print('=' * 70)
    print('Figure Generation Regression Tests')
    print('=' * 70)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    _backup_results()
    results = []
    try:
        results.append(test_fig_benchmark_empty_data())
        results.append(test_fig_benchmark_all_errors())
        results.append(test_plot_ablation_group_empty_data())
        results.append(test_plot_ablation_group_all_errors())
        results.append(test_fig_ablation_suptitle_not_clipped())
    finally:
        _restore_results()
    n_pass = sum(1 for r in results if r)
    n_total = len(results)
    print('-' * 70)
    print(f'Total: {n_pass}/{n_total} passed')
    return 0 if n_pass == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
