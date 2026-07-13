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
    # torch.compile has a real one-time compilation cost (observed
    # 5-15s even for a tiny T=8 recurrence on CPU) that is unrelated to
    # the algorithm itself; mark slow so fast CI loops can skip it.
    'test_compiled_recurrent_kda_fullgraph',
}


def pytest_collection_modifyitems(config, items):
    """Auto-mark slow tests and skip CUDA tests when CUDA is unavailable.

    Test functions in ``run_correctness.py`` take an optional
    ``device='cpu'`` argument. When pytest collects them, it sees the
    parameter and tries to fixture-inject it. We mark slow tests with the
    ``slow`` marker so they can be skipped with ``-m "not slow"``.

    P1-6 fix (Batch-3): when the user passes ``--device cuda`` but CUDA is
    unavailable, previously the tests would run with ``device='cuda'`` and
    crash deep inside torch with a cryptic ``RuntimeError: CUDA is not
    available``. Now we detect this up-front and skip the CUDA-only tests
    with a clear ``skip`` reason instead of letting them crash.
    """
    device = config.getoption('--device')
    if device == 'cuda':
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if not cuda_available:
            skip_cuda = pytest.mark.skip(
                reason='--device cuda requested but torch.cuda.is_available() '
                       'is False; run on CPU (the default) or fix your CUDA '
                       'environment before re-running with --device cuda.')
            for item in items:
                # Only skip tests that actually take a ``device`` argument
                # (functions without ``device`` are device-agnostic and
                # should still run).
                if 'device' in getattr(item, 'fixturenames', ()):
                    item.add_marker(skip_cuda)
            return  # Skip the slow-test marking; the items are already skipped.
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
# The previous implementation used a ``pytest_runtest_call`` hookwrapper and
# read ``outcome.get_result()``. That DOES NOT WORK: for ``pytest_runtest_call``
# the outcome's result is the return value of ``item.runtest()``, which is
# always ``None`` (the test function's return value is discarded by pytest's
# default ``pytest_pyfunc_call`` implementation BEFORE ``runtest`` returns).
# Verified: a test returning ``[{'status': 'FAIL'}]`` was still reported as
# ``1 passed`` with only a ``PytestReturnNotNoneWarning``.
#
# The fix: replace the default ``pytest_pyfunc_call`` implementation with our
# own that (a) calls the test function, (b) captures the return value, (c)
# inspects it for the ``_ok`` list-of-dict pattern, and (d) raises
# ``AssertionError`` if any sub-check has ``status != 'PASS'``. The
# ``tryfirst=True`` ensures our hook runs before pytest's default
# implementation, and returning ``True`` stops the default from running.
# Returning ``None`` (for async / generator tests we do not handle) lets the
# default implementation run as usual.
@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Capture test function return values and convert list-of-dict FAIL
    entries into real AssertionError failures.

    Replaces the broken ``pytest_runtest_call`` hookwrapper (which could not
    see the test function's return value because ``outcome.get_result()``
    returns ``None`` for non-raising tests).

    Only handles plain synchronous functions. Async tests, generator tests,
    and any other exotic test types fall through to pytest's default
    implementation (return ``None`` to yield to the next hook).
    """
    import inspect as _inspect
    testfunction = pyfuncitem.obj
    # Let the default impl handle async / generator test functions — replacing
    # them would require duplicating pytest's async-runner logic, which is
    # fragile and outside the scope of this fix.
    if _inspect.iscoroutinefunction(testfunction) or \
       _inspect.isasyncgenfunction(testfunction) or \
       _inspect.isgeneratorfunction(testfunction):
        return None  # fall through to default impl
    funcargs = pyfuncitem.funcargs
    # Filter funcargs to only the parameters the test function actually
    # accepts, so we don't pass unexpected kwargs (which would raise
    # TypeError). Mirrors how pytest's default impl resolves arguments via
    # ``pyfuncitem._fixtureinfo.argnames`` but uses the public
    # ``inspect.signature`` API instead of the private ``_fixtureinfo``.
    try:
        sig = _inspect.signature(testfunction)
        accepted = set(sig.parameters.keys())
        testargs = {k: v for k, v in funcargs.items() if k in accepted}
    except (ValueError, TypeError):
        # Builtins or C-implemented functions may not have a signature;
        # fall back to passing all funcargs (the default impl's behaviour).
        testargs = dict(funcargs)
    res = testfunction(**testargs)
    # Inspect the return value (the ``_ok`` pattern from run_correctness.py).
    # Only list-of-dict-with-'status' returns are interpreted; everything
    # else (None, scalar, etc.) is left to pytest's default behaviour (which
    # is to issue a ``PytestReturnNotNoneWarning`` for non-None returns).
    if isinstance(res, list) and res and all(
            isinstance(r, dict) and 'status' in r for r in res):
        failures = [r for r in res if r.get('status') != 'PASS']
        if failures:
            msgs = '\n'.join(
                f"  - [{r.get('status','?')}] {r.get('name','?')}: "
                f"{r.get('detail','')}" for r in failures)
            raise AssertionError(
                f"{len(failures)} check(s) failed in {pyfuncitem.name}:\n{msgs}")
    return True  # stop the default impl from running
