"""pytest configuration for the kda-csa-hca-fusion repository.

The repository's regression tests live in ``run_correctness.py`` and use a
custom runner (``_ok(name, cond, detail)`` + ``main()``) that emits a
structured ``results/exp1_correctness.json``. The test functions themselves
use the standard ``test_*`` naming convention and take an optional
``device='cpu'`` argument, so they are also discoverable by pytest.

This conftest:

* Adds the repository root to ``sys.path`` so ``import ops_kda`` etc. work
  without ``sys.path.insert`` hacks (also handled by ``pip install -e .``).
* Registers a ``device`` fixture so ``pytest`` can pass ``'cpu'`` (or
  ``'cuda'``) to test functions that accept a ``device`` argument.
* Marks the long-running tests (``test_csa_full_pipeline_causality``,
  ``test_prefill_flops_*`` at T>=1024, etc.) as ``slow`` so they can be
  skipped with ``pytest -m "not slow"`` during fast CI loops.

Usage::

    # Run all tests with the custom runner (canonical, emits JSON):
    python run_correctness.py

    # Run a subset with pytest (faster iteration, parallelizable):
    pytest -q run_correctness.py::test_kda_chunk_vs_recurrent
    pytest -q -k "kda" run_correctness.py
    pytest -q -m "not slow" run_correctness.py test_figures.py

    # Run on GPU (if available):
    pytest -q run_correctness.py --device cuda
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the repository root is on sys.path so ``import ops_kda`` works
# even when pytest is invoked from a different working directory. This
# mirrors the ``sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))``
# pattern used at the top of every script, but centralizes it so test
# modules don't need to repeat the hack.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def pytest_addoption(parser):
    """Add a ``--device`` flag so tests can be run on CPU or CUDA."""
    parser.addoption(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run tests on (default: cpu).',
    )


@pytest.fixture
def device(request):
    """Fixture yielding the device string selected via ``--device``.

    Test functions that take a ``device`` parameter will receive this
    fixture's value automatically. Functions that don't take ``device``
    are unaffected.
    """
    return request.config.getoption('--device')


# Tests that are known to be slow (long sequences, multi-seed sweeps, or
# full-pipeline causality checks that perturb every source position). Mark
# them so ``pytest -m "not slow"`` can skip them during fast CI loops.
_SLOW_TESTS = {
    'test_csa_full_pipeline_causality',
    'test_prefill_flops_causal_block_entries',
    'test_kv_cache_ceil_block_count',
    'test_csa_hca_extreme_sink_values',
    'test_hca_sliding_window_causality',
}


def pytest_collection_modifyitems(config, items):
    """Auto-mark slow tests and set the device argument default.

    Test functions in ``run_correctness.py`` take an optional
    ``device='cpu'`` argument. When pytest collects them, it sees the
    parameter and tries to fixture-inject it. We mark slow tests with the
    ``slow`` marker so they can be skipped with ``-m "not slow"``.
    """
    for item in items:
        # Mark slow tests by function name.
        for slow_name in _SLOW_TESTS:
            if item.name.startswith(slow_name):
                item.add_marker(pytest.mark.slow)
                break


# ----------------------------------------------------------------------------
# P0-1 fix: convert "list-of-dict result" returns into real pytest failures.
# ----------------------------------------------------------------------------
# The test functions in ``run_correctness.py`` follow a custom-runner pattern:
# each ``test_*`` function returns a list of ``_ok(name, cond, detail)`` dicts
# instead of using ``assert``. The custom ``main()`` runner in that file
# aggregates these dicts and writes a structured ``exp1_correctness.json``.
#
# The problem: pytest IGNORES non-None return values from test functions.
# A ``cond=False`` recorded via ``_ok(name, False, ...)`` would therefore
# be silently marked as PASS by pytest — a "false green" that hides real
# regressions when contributors run ``pytest -q run_correctness.py``.
#
# This hook runs after each test function returns. If the return value is a
# list of dicts containing a ``status`` field, we scan for any non-PASS
# entries and convert them into an ``AssertionError`` so pytest reports the
# test as failed (with a useful message listing every failed sub-check).
#
# The custom ``main()`` runner is unaffected because it reads the same list
# directly. Both protocols now agree: a FAIL in any ``_ok`` is a test failure
# in both the JSON report AND pytest's exit code.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    res = outcome.get_result()
    # Only inspect list-of-dict returns (the ``_ok`` pattern). Anything else
    # (None, scalar, etc.) is left to pytest's default behaviour.
    if isinstance(res, list) and res and all(
            isinstance(r, dict) and 'status' in r for r in res):
        failures = [r for r in res if r.get('status') != 'PASS']
        if failures:
            msgs = '\n'.join(
                f"  - [{r.get('status','?')}] {r.get('name','?')}: "
                f"{r.get('detail','')}" for r in failures)
            raise AssertionError(
                f"{len(failures)} check(s) failed in {item.name}:\n{msgs}")
