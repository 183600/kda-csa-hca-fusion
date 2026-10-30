"""Single-entry-point runner for all experiments — Kaggle notebook friendly.

Usage on Kaggle
---------------
1. Upload the repository root (this file sits at the top level, next to
   ``ops_*.py``) as a Kaggle Dataset — or clone the repo into the notebook
   working directory.
2. In a notebook cell::

     !pip install -q einops matplotlib
     import sys; sys.path.insert(0, '/kaggle/input/<your-dataset-name>')
     %run /kaggle/input/<your-dataset-name>/run_all.py

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
  * Calls ``setup_kaggle()`` to VERIFY CUDA availability on Kaggle T4 (the
    CUDA wheel install itself lives in ``kaggle_setup.bootstrap_kaggle_cuda``
    and requires a kernel restart — see that module's docstring).
  * Prints an environment summary.
  * Runs all experiments (Exp 1-9) + method analysis + figure generation.
  * Saves a combined ``results/summary.json`` with pass/fail and key numbers.

Environment knobs (set before importing / via ``os.environ``):
  * ``MQAR_SEEDS``      (default 5)   — seeds for the MQAR experiment.
  * ``MQAR_STEPS``      (default 200) — training steps for non-softmax ops.
  * ``MQAR_SOFTMAX_STEPS`` (default: same as MQAR_STEPS) — optional explicitly labelled softmax long-training sensitivity run.
  * ``ABL_SEEDS``       (default 7)   — seeds for the ablation (run_all(seeds=...) overrides).
  * ``ABL_STEPS``       (default 100) — training steps for the ablation.
  * ``MECH_SEEDS``      (default 5)   — seeds for the mechanism ablation (Exp 8).
  * ``MECH_STEPS``      (default 200) — training steps for Exp 8.
  * ``MECH_VARIANTS``   (default "")  — optional comma-separated Exp 8 variant subset.
  * ``BUDGET_T_GRID``   (default '16,64,256,1024,4096,16384') — Exp 9 context grid.
  * ``BUDGET_TOPK``     (default '0,1,2,4,8') — Exp 9 decode top-k sweep values.
  * ``BUDGET_PREFILL_LENS`` (default '256,512') — Exp 9 decode prefill lengths.
  * ``BUDGET_DECODE_TOKENS`` (default 10) — Exp 9 decode tokens per cell.
  * ``BUDGET_REPEATS``  (default 2) — Exp 9 decode timing repeats.
  * ``STATS_CAL_REPS``  (default 300) — Exp 16 calibration replicates.
  * ``STATS_CAL_NBOOT`` (default 400) — Exp 16 bootstrap replicates.
  * ``STATS_CAL_SEED``  (default 0)   — Exp 16 base seed.
  * ``SKIP_SLOW``       (default 0)   — if "1", skip CSA-heavy experiments on CPU.

This script is the reproducibility anchor referenced by the paper's
"Reproducibility" section.
"""

from __future__ import annotations

import glob
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

import experiment_design  # noqa: E402  (needs HERE on sys.path)
import timing_design  # noqa: E402  (needs HERE on sys.path)


# ---------------------------------------------------------------------------
# Run profiles (framework design: smoke / screening / formal tiers).
#
# RUN_ALL_PROFILE selects the budget tier for a full pipeline run:
#
#   * ``smoke``  — pipeline VALIDATION only: tiny seeds/steps/lengths so
#     forward, backward, caches, result files, and figures all get
#     exercised end-to-end in minutes. Numbers from a smoke run are NOT
#     comparable to anything (the Bonferroni machinery is knowingly
#     underpowered at 2 seeds) — the run only proves the plumbing works.
#     Also executes a 1-seed, 5-layer, 5-step pass of the eight-group
#     matrix (exp5b smoke) so the ABL_MODE=matrix path is CI-covered.
#   * ``screen`` (default) — the small-scale screening tier.
#   * ``full``   — the formal-validation tier: seed counts raised to 7 so
#     the multi-seed statistics stabilise, and the timing repeat counts
#     raised to the pre-registered confirmatory budget
#     (``timing_design.TIMING_PLANS``). Training step budgets are NOT
#     silently raised (scale those explicitly via MQAR_STEPS / ABL_STEPS /
#     ... once the screening curves say where the plateau sits).
#
# Application rule: a preset only sets variables the caller did NOT
# export — an explicit env var always wins over the profile, so
# ``RUN_ALL_PROFILE=smoke MQAR_SEEDS=5`` really runs 5 seeds.
# ---------------------------------------------------------------------------
_RUN_PROFILES: dict[str, dict[str, str] | None] = {
    'smoke': {
        'MQAR_SEEDS': '2', 'MQAR_STEPS': '30',
        'ABL_SEEDS': '2', 'ABL_STEPS': '30',
        'MECH_SEEDS': '2', 'MECH_STEPS': '30',
        'RECALL_SEEDS': '2', 'RECALL_STEPS': '20',
        'BENCH_LENGTHS': '128,256', 'BENCH_REPEATS': '3',
        'BUDGET_PREFILL_LENS': '256', 'BUDGET_DECODE_TOKENS': '3',
        'BUDGET_REPEATS': '1', 'SKIP_SLOW': '1',
        # Quantization experiments (Exp 10 / Exp 11): keep the plumbing
        # (every spec/mode exercised) but shrink the geometry and the
        # training budgets to the smallest values that still exercise
        # multi-seed aggregation and both recovery metrics.
        'QUANT_SEEDS': '1', 'QUANT_BITS': '2,4',
        'QUANT_GROUP_SIZES': '16,64', 'QUANT_N_KEYS': '8',
        'QUANT_VALUE_DIM': '4',
        'QLM_SEEDS': '1', 'QLM_STEPS': '20', 'QLM_QAT_STEPS': '20',
        'QLM_SEQ_LEN': '32', 'QLM_BATCH': '4', 'QLM_EVAL_SEQS': '2',
        'QLM_CORPUS': '32',
    },
    'screen': None,   # default — no preset overrides
    'full': {
        'MQAR_SEEDS': '7', 'ABL_SEEDS': '7', 'MECH_SEEDS': '7',
        'RECALL_SEEDS': '7',
        # The timing plans' confirmatory repeat budget (timing_design.py).
        # Screening defaults (5 / 3) remain for smoke and screen profiles.
        'BENCH_REPEATS': '15', 'DECODE_REPEATS': '15',
    },
}


def _apply_run_profile() -> str:
    """Apply the RUN_ALL_PROFILE preset to unset env vars; return the name."""
    profile = os.environ.get('RUN_ALL_PROFILE', 'screen').strip().lower()
    if profile not in _RUN_PROFILES:
        raise ValueError(
            f"RUN_ALL_PROFILE={profile!r} must be one of "
            f"{sorted(_RUN_PROFILES)}.")
    preset = _RUN_PROFILES[profile]
    applied = []
    if preset:
        for key, value in preset.items():
            if key not in os.environ:
                os.environ[key] = value
                applied.append(key)
    if applied:
        print(f'[run_all] profile {profile!r}: set '
              f'{", ".join(sorted(applied))} (unset-vars only)')
    else:
        print(f'[run_all] profile {profile!r}: no overrides needed')
    return profile


def _env_seed_list(name: str, default: int) -> list[int]:
    """Read a seed COUNT env var the way the runners do, then expand it.

    The runners use ``parse_int_env`` (warn + default on a malformed value),
    so the design manifest must use the same fallback; otherwise the plan
    would record a budget the experiment did not use.
    """
    raw = os.environ.get(name, str(default)).strip()
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = int(default)
    # The runners call parse_int_env(..., min_value=1): a zero/negative
    # seed count falls back to the default rather than running zero seeds.
    if count < 1:
        count = int(default)
    return [experiment_design.DEFAULT_SEED_BASE + i for i in range(count)]


def _env_count(name: str, default: int) -> int:
    """Read an env-var count with the runners' fallback semantics."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    return value if value >= 1 else int(default)


def _raise_seed_floor(name: str, default: int, minimum: int = 5) -> int:
    """Raise a seed-count env var to ``minimum`` and return the new count.

    ``parse_int_env`` falls back to ``default`` on a malformed value, so use
    the same fallback before applying the floor.  The CPU ``SKIP_SLOW`` path
    uses this to keep the multi-seed tests usable; applying it BEFORE the
    design manifest (and before provenance capture) makes the recorded seed
    budget match the budget the runners actually use.
    """
    raw = os.environ.get(name, str(default)).strip()
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = int(default)
    if count < int(minimum):
        count = int(minimum)
    os.environ[name] = str(count)
    return count


def _write_experiment_design_manifest(profile: str) -> dict:
    """Write the prospective design manifest; return a compact summary block.

    This runs BEFORE the experiments, so the seed budget, SESOI and required
    n are on disk independently of the numbers. A stale registry is reported
    but does not block the pipeline: the manifest still records what the
    run is about to do, and each runner attaches its own runtime audit.
    """
    design_seeds = {
        'exp4_mqar': _env_seed_list('MQAR_SEEDS', 5),
        'exp5_ratio': _env_seed_list('ABL_SEEDS', 7),
        'exp5b_matrix': _env_seed_list('ABL_SEEDS', 7),
        'exp8_mechanism': _env_seed_list('MECH_SEEDS', 5),
        'exp9_recall': _env_seed_list('RECALL_SEEDS', 3),
    }
    design_errors = experiment_design.validate_plans()
    if design_errors:
        print('[run_all] WARNING: the pre-registered design registry is '
              'inconsistent; the manifest is still written so the run is not '
              'blocked, but fix these before citing it:')
        for error in design_errors:
            print(f'  - {error}')
    # Bind the pre-registration to the tree/runtime that writes it. Imported
    # lazily because experiment_design itself is stdlib-only; if provenance
    # capture fails the manifest is still written (planning must not depend
    # on torch), but the failure is visible in the run log.
    try:
        from kaggle_setup import capture_provenance
        design_provenance = capture_provenance()
    except Exception as exc:  # pragma: no cover - defensive
        design_provenance = None
        print(f'[run_all] WARNING: could not capture design provenance '
              f'({type(exc).__name__}: {exc}); the design manifest will not '
              f'be bound to a source revision')
    manifest = experiment_design.design_manifest(
        seeds_by_plan=design_seeds, provenance=design_provenance)
    experiment_design.write_design_manifest(
        'results/experiment_design.json', manifest=manifest)
    for row in manifest['results']:
        plan = row['planning']
        eq = plan.get('equivalence') or {}
        inter = plan.get('interaction') or {}
        print(f"[design] {row['plan_id']}: n={row['n_seeds']} "
              f"required={plan['required_seeds']} "
              f"power={plan['planned_power']} "
              f"status={row['status']} "
              f"eq_n={eq.get('required_seeds_at_zero', '-')} "
              f"eq_status={eq.get('status', '-')} "
              f"int_n={inter.get('required_seeds', '-')} "
              f"int_status={inter.get('status', '-')} "
              f"plan_sha={row['plan_sha256'][:12]}")
    return {
        'path': 'results/experiment_design.json',
        'run_profile': profile,
        'validation_errors': design_errors,
        'provenance_captured': design_provenance is not None,
        'plans': {
            row['plan_id']: {
                'n_seeds': row['n_seeds'],
                'status': row['status'],
                'plan_sha256': row['plan_sha256'],
                'required_seeds': row['planning']['required_seeds'],
                'planned_power': row['planning']['planned_power'],
                'equivalence_required_seeds': (
                    (row['planning'].get('equivalence') or {}).get(
                        'required_seeds_at_zero')),
                'equivalence_status': row['planning'].get(
                    'equivalence_status'),
                # Interaction claims are a separate budget: surface it so a
                # main-effect superiority budget cannot be read as a
                # transfer/complementarity budget.
                'interaction_required_seeds': (
                    (row['planning'].get('interaction') or {}).get(
                        'required_seeds')),
                'interaction_status': row['planning'].get(
                    'interaction_status'),
            }
            for row in manifest['results']
        },
    }


def _write_timing_design_manifest(profile: str) -> dict:
    """Write the prospective latency measurement designs BEFORE the runs.

    The repeat budget is read from the same env vars the runners use, so the
    manifest records the measurement actually about to happen. A stale or
    under-budget registry is reported but does not block the pipeline; each
    runner attaches its own runtime audit (balanced order, warmup, achieved
    CV) to its result metadata.
    """
    repeats_by_plan = {
        'exp2_benchmark': _env_count('BENCH_REPEATS', 5),
        'exp6_decoding': _env_count('DECODE_REPEATS', 3),
    }
    design_errors = timing_design.validate_plans()
    bad_plans = {plan_id: messages for plan_id, messages
                 in design_errors.items() if messages}
    if bad_plans:
        print('[run_all] WARNING: the pre-registered timing design registry '
              'is inconsistent; the manifest is still written so the run is '
              'not blocked, but fix these before citing it:')
        for plan_id, messages in bad_plans.items():
            for message in messages:
                print(f'  - {plan_id}: {message}')
    try:
        from kaggle_setup import capture_provenance
        design_provenance = capture_provenance()
    except Exception as exc:  # pragma: no cover - defensive
        design_provenance = None
        print(f'[run_all] WARNING: could not capture timing-design provenance '
              f'({type(exc).__name__}: {exc}); the manifest will not be bound '
              f'to a source revision')
    manifest = timing_design.design_manifest(
        repeats_by_plan=repeats_by_plan, provenance=design_provenance)
    timing_design.write_design_manifest(
        'results/timing_design.json', manifest=manifest)
    for row in manifest['results']:
        planning = row['planning']
        print(f"[timing-design] {row['plan_id']}: n={row['n_repeats']} "
              f"family={planning['family_size']} "
              f"need={planning['required_repeats']} "
              f"power={planning['planned_power']} "
              f"status={row['status']} "
              f"plan_sha={row['plan_sha256'][:12]}")
    return {
        'path': 'results/timing_design.json',
        'run_profile': profile,
        'validation_errors': design_errors,
        'provenance_captured': design_provenance is not None,
        'plans': {
            row['plan_id']: {
                'n_repeats': row['n_repeats'],
                'status': row['status'],
                'plan_sha256': row['plan_sha256'],
                'required_repeats': row['planning']['required_repeats'],
                'planned_power': row['planning']['planned_power'],
                'planned_mde_ratio': row['planning']['planned_mde_ratio'],
            }
            for row in manifest['results']
        },
    }


def _version_tuple(raw: str) -> tuple[int, ...]:
    """Parse the numeric prefix of a package version for bound checks."""
    parts = re.findall(r"\d+", raw)
    return tuple(int(x) for x in parts[:3]) or (0,)


def _version_in_range(raw: str, lower: tuple[int, ...], upper: tuple[int, ...]) -> bool:
    value = _version_tuple(raw)
    return value >= lower and value < upper


def _ensure_deps():
    """Ensure runtime packages exist; never fight a preinstalled environment.

    Kaggle notebook images ship recent, GPU-compatible builds of
    torch / numpy / matplotlib, and an in-place downgrade of numpy can
    break the preinstalled torch stack, so this function never
    hard-rejects a preinstalled package.

    Policy:
      * A MISSING package is pip-installed (within the tested bounds),
        because absent dependencies are always fatal.
      * An INSTALLED package outside the tested bounds only triggers a
        WARNING: the repository's pure-PyTorch operator code does not
        depend on version-sensitive APIs for CORRECTNESS. The only
        expected variance is the SDPA kernel selected inside the softmax
        baseline (torch releases differ here); benchmark comparisons
        are read with that caveat. Set ``STRICT_DEPS=1`` to enable the
        hard version gate for bit-reproducible reruns.
    """
    strict = os.environ.get('STRICT_DEPS', '0') == '1'
    # Advisory tested bounds. Upper bounds track the newest releases the
    # full suite (run_correctness + all experiments + figures) is validated
    # against, so Kaggle / AutoDL images with recent preinstalled stacks
    # run WARNING-FREE.
    bounded = {
        'einops': ('einops>=0.6,<0.9', (0, 6), (0, 9)),
        'matplotlib': ('matplotlib>=3.5,<3.12', (3, 5), (3, 12)),
        'numpy': ('numpy>=1.21,<2.6', (1, 21), (2, 6)),
    }
    for package, (spec, lower, upper) in bounded.items():
        try:
            installed = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            installed = None
        if installed is not None and _version_in_range(installed, lower, upper):
            continue
        if installed is not None:
            if strict:
                raise RuntimeError(
                    f'{package}=={installed} is outside the tested range {spec} '
                    'and STRICT_DEPS=1 is set. Install the supported version '
                    'and restart the Python process before running experiments.')
            print(f'[run_all] WARNING: {package}=={installed} is outside the '
                  f'historically-tested range {spec}; continuing anyway. '
                  '(Set STRICT_DEPS=1 to enforce the bounds.)')
            continue
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
    # fallback for the required t-distribution calculations. Do not add a
    # network dependency on Kaggle when SciPy is absent; an out-of-range
    # installed SciPy only warns (see _ensure_deps docstring).
    try:
        scipy_version = importlib_metadata.version('scipy')
    except importlib_metadata.PackageNotFoundError:
        print('[run_all] scipy is not installed; using the repository statistical fallback.')
    else:
        if not _version_in_range(scipy_version, (1, 7), (1, 19)):
            if strict:
                raise RuntimeError(
                    f'scipy=={scipy_version} is outside the tested range >=1.7,<1.19 '
                    'and STRICT_DEPS=1 is set. Install a supported version or '
                    'remove it to use the exact fallback.')
            print(f'[run_all] WARNING: scipy=={scipy_version} is outside the '
                  'tested range >=1.7,<1.19; continuing with it (the exact '
                  'fallback is used only when scipy is absent).')

    # torch: only the LOWER bound is a hard requirement (the operator code
    # uses no APIs newer than torch 2.2). The advisory upper bound below
    # matches requirements.txt / pyproject.toml (``torch>=2.2,<3``); releases
    # inside that range are tested and must NOT trigger a warning. A torch
    # NEWER than the tested range may still change SDPA kernel selection
    # and therefore the softmax-baseline comparison; warn but proceed. If
    # torch is absent, importing kaggle_setup below will produce its own
    # dependency error.
    try:
        torch_version = importlib_metadata.version('torch')
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            'torch is not installed; install the project dependencies before '
            'running run_all.py') from exc
    if not _version_in_range(torch_version, (2, 2), (99, 0)):
        raise RuntimeError(
            f'torch=={torch_version} is older than the required torch>=2.2. '
            'Upgrade torch and restart the process before running experiments.')
    if not _version_in_range(torch_version, (2, 2), (3, 0)):
        if strict:
            raise RuntimeError(
                f'torch=={torch_version} is outside the tested range >=2.2,<3 '
                'and STRICT_DEPS=1 is set. Install a supported torch build and '
                'restart the process before running experiments.')
        print(f'[run_all] WARNING: torch=={torch_version} is newer than the '
              'validated >=2.2,<3 range; correctness tests still '
              'run, but softmax-baseline benchmark numbers may deviate from '
              'the committed historical results (SDPA kernel-selection '
              'changes across torch releases).')


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
# fresh output (see the figure step below). Every file an experiment can
# write must be listed here, including provenance side files and the
# auxiliary outputs (``exp4b`` quality-task rows, the ``exp5b`` matrix
# smoke) — a missing entry would leave a stale file to be misread next to
# this run's fresh ones.
_EXPERIMENT_OUTPUTS = {
    'exp1_correctness': ['results/exp1_correctness.json'],
    'exp2_benchmark': ['results/exp2_benchmark.json',
                       'results/exp2_benchmark_provenance.json',
                       'results/exp2_benchmark_raw.json'],
    'exp3_kv_cache': ['results/exp3_kv_cache.json'],
    'exp4_mqar': ['results/exp4_mqar.json',
                  'results/exp4b_quality_tasks.json',
                  'results/exp4_overfit_gate.json'],
    'exp5_ablation': ['results/exp5_ablation.json',
                      'results/exp5_ablation_provenance.json'],
    'exp5b_ablation_matrix_smoke': [
        'results/exp5b_ablation_matrix.json',
        'results/exp5b_ablation_matrix_contrasts.json',
        'results/exp5b_ablation_matrix_provenance.json'],
    'exp6_decoding': ['results/exp6_decoding.json',
                      'results/exp6_decoding_provenance.json',
                      'results/exp6_decoding_raw.json'],
    'exp14_indexer_selector': ['results/exp14_indexer_selector.json',
                               'results/exp14_indexer_selector_provenance.json'],
    'exp15_ppl_calibration': ['results/exp15_ppl_calibration.json',
                              'results/exp15_ppl_calibration_provenance.json'],
    'exp16_paired_stats_calibration': [
        'results/exp16_paired_stats_calibration.json',
        'results/exp16_paired_stats_calibration_provenance.json'],
    'exp7_roofline': ['results/exp7_roofline.json'],
    'exp8_mechanism_ablation': ['results/exp8_mechanism_ablation.json'],
    'exp9_recall': ['results/exp9_recall_reach.json'],
    'exp9_budget_frontier': ['results/exp9_budget_frontier.json'],
    'exp10_quant_recall': ['results/exp10_quant_recall.json'],
    'exp11_quant_lm': ['results/exp11_quant_lm.json'],
    # LM-scale architecture contrast (smoke profile): raw run records in
    # the checkpoint tree, hence listed as raw side files below.
    'exp12_lm_hybrid_smoke': [
        'checkpoints_lm/smoke_hybrid/run_record.json'],
    'exp12_lm_softmax_smoke': [
        'checkpoints_lm/smoke_softmax/run_record.json'],
    # Prospective design manifest written before the experiments start.
    'experiment_design': ['results/experiment_design.json'],
    'timing_design': ['results/timing_design.json'],
}

# Result files listed in ``_EXPERIMENT_OUTPUTS`` that are NOT results_schema
# envelopes: provenance side files (plain ``capture_provenance()`` dicts)
# and Exp 4's overfit-gate report payload (a raw report dict with no
# ``results`` list). The schema integrity gate below skips exactly these;
# anything else an experiment writes MUST parse as a compatible
# ``results_schema`` payload or the gate fails the pipeline.
_RAW_SIDE_FILES = {
    'results/exp2_benchmark_provenance.json',
    'results/exp2_benchmark_raw.json',
    'results/exp5_ablation_provenance.json',
    'results/exp5b_ablation_matrix_provenance.json',
    'results/exp6_decoding_provenance.json',
    'results/exp6_decoding_raw.json',
    'results/exp14_indexer_selector_provenance.json',
    'results/exp15_ppl_calibration_provenance.json',
    'results/exp16_paired_stats_calibration_provenance.json',
    'results/exp4_overfit_gate.json',
    'results/exp9_recall_reach.json',
    'results/experiment_design.json',
    'results/timing_design.json',
    'checkpoints_lm/smoke_hybrid/run_record.json',
    'checkpoints_lm/smoke_softmax/run_record.json',
}


# Figures each experiment's ``main()`` feeds. Purged alongside the stale
# result JSONs on failure: otherwise a failed run leaves last session's
# PDFs in ``figures/`` and they can be mistaken for fresh output (the
# figure step only regenerates figures whose result files exist).
# Glob patterns cover make_figures.py's per-n_kv suffixed names
# (``fig_mqar_nkv2.pdf``) together with the legacy unsuffixed name.
_EXPERIMENT_FIGURES = {
    'exp2_benchmark': ['figures/fig_benchmark.pdf'],
    'exp3_kv_cache': ['figures/fig_kv_cache.pdf', 'figures/fig_flops.pdf'],
    'exp4_mqar': ['figures/fig_mqar*.pdf'],
    'exp5_ablation': ['figures/fig_ablation*.pdf'],
    'exp6_decoding': ['figures/fig_decoding.pdf'],
    'exp7_roofline': ['figures/fig_roofline.pdf'],
    'exp8_mechanism_ablation': ['figures/fig_mechanism_ablation*.pdf'],
    'exp9_budget_frontier': ['figures/fig_budget_frontier.pdf'],
}


def _purge_stale_outputs(name):
    """Remove result JSONs / figures that a failed run would leave stale."""
    for rel in _EXPERIMENT_OUTPUTS.get(name, ()):
        try:
            if os.path.exists(rel):
                os.remove(rel)
        except OSError as e:
            print(f'[run_all] WARNING: could not remove stale {rel}: {e}')
    for pattern in _EXPERIMENT_FIGURES.get(name, ()):
        for rel in glob.glob(pattern):
            try:
                os.remove(rel)
            except OSError as e:
                print(f'[run_all] WARNING: could not remove stale {rel}: {e}')


def _check_results_schema():
    """Schema integrity gate over the result files present in ``results/``.

    Wired into the pipeline here because ``ci_gate.py`` is frozen: every
    schema-managed file named in ``_EXPERIMENT_OUTPUTS`` (i.e. everything
    except ``_RAW_SIDE_FILES``) must parse as a compatible
    ``results_schema`` payload, AND — for kind-tagged files with a
    registered row contract (exp5b matrix + contrasts, exp6 decoding) —
    its rows must satisfy that contract: complete group sets, aggregates
    consistent with the per-seed rows, honest incremental-cache flags.
    Returns 0 on success and a non-zero exit code otherwise, honoring
    ``_run``'s return-code contract.
    """
    import results_schema
    candidates: list[str] = []
    seen: set[str] = set()
    for outputs in _EXPERIMENT_OUTPUTS.values():
        for rel in outputs:
            if rel in _RAW_SIDE_FILES or rel in seen:
                continue
            seen.add(rel)
            candidates.append(rel)
    present = [p for p in candidates if os.path.exists(p)]
    if not present:
        print('[results_schema] no schema-managed result files present')
        return 1
    missing = [p for p in candidates if p not in present]
    if missing:
        # A missing file belongs to an experiment that did not write its
        # output (its own step is already recorded as failed above, or the
        # run profile legitimately skipped it, e.g. exp5b in non-smoke
        # profiles). Report it for visibility without double-counting.
        print('[results_schema] not written this run (skipped): '
              + ', '.join(missing))
    return results_schema.main(['--check', '--check-rows'] + present)


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


def _run_train_lm_smoke(arch):
    """Pipeline smoke for ``train_lm_autodl`` (tiny budget, one arch).

    The full LM runs are launched separately (they need a real corpus and
    a GPU to be meaningful); this exercises the SAME entry point end to
    end — data fallback, tokenizer, overfit gate skip, training loop,
    validation selection, sealed-test evaluation, run record and manifest
    — for both the hybrid and its matched softmax baseline.
    """
    import train_lm_autodl
    out_dir = os.path.join('checkpoints_lm', f'smoke_{arch}')
    argv = [
        'train_lm_autodl.py', '--arch', arch, '--kaggle',
        '--max_docs', '64', '--max_steps', '2', '--seq_len', '32',
        '--batch_size', '2', '--val_every', '1', '--eval_max_batches', '1',
        '--early_stop_patience', '0', '--error_analysis_batches', '0',
        '--ppl_bootstrap_nboot', '0', '--skip_overfit_check',
        '--skip_consistency_check', '--output_dir', out_dir,
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = train_lm_autodl.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(
                f'train_lm_autodl({arch}) exited with {exc}') from exc
        rc = exc.code
    finally:
        sys.argv = old_argv
    return rc


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
                 'FIGURES_DIR', 'SKIP_SLOW', 'MECH_SEEDS', 'MECH_STEPS',
                 'MECH_TRAIN_BATCH', 'MECH_NKV', 'MECH_VARIANTS',
                 'BUDGET_T_GRID', 'BUDGET_TOPK', 'BUDGET_PREFILL_LENS',
                 'BUDGET_DECODE_TOKENS', 'BUDGET_REPEATS',
                 # Run-profile + matrix-mode variables: run_all itself may
                 # set these (profile presets, the matrix smoke pass), so
                 # the finally block has to restore the caller's state.
                 'RUN_ALL_PROFILE', 'BENCH_REPEATS', 'DECODE_REPEATS',
                 'MQAR_OPS',
                 'MQAR_TASKS', 'MAMBA_BACKEND', 'ABL_MODE', 'ABL_MATRIX',
                 'ABL_TOTAL_LAYERS', 'ABL_MEMORY', 'ABL_D_MODEL',
                 'ABL_TRAIN_BATCH',
                 # Quantization experiments (Exp 10 / Exp 11): the smoke
                 # profile and SKIP_SLOW CPU path truncate these.
                 'QUANT_BITS', 'QUANT_GROUP_SIZES', 'QUANT_SEEDS',
                 'QUANT_N_KEYS', 'QUANT_VALUE_DIM',
                 'QLM_SEEDS', 'QLM_STEPS', 'QLM_QAT_STEPS', 'QLM_SEQ_LEN',
                 'QLM_BATCH', 'QLM_EVAL_SEQS', 'QLM_CORPUS', 'QLM_SKIP_QAT',
                 'QLM_PPL_TOL', 'QLM_BITS', 'QLM_GROUP_SIZES')
    # Snapshot the precise pre-call state of each env var: distinguish
    # "unset" from "set to a value" so the finally block can restore the
    # exact state.
    _orig_env = {k: (os.environ[k] if k in os.environ else None) for k in _env_keys}
    # The env-var mutations below live INSIDE the try-finally block so
    # a failure in ``_ensure_deps()`` / ``_setup()`` does not leak the
    # mutated env vars to the caller's process. We snapshot the
    # originals here (so the finally block can restore them) but DEFER
    # the actual mutations until we are inside the try block (after
    # ``_ensure_deps`` / ``_setup`` succeed). The mutations run just
    # after ``os.chdir(out_root)`` below.

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
        # Apply the smoke / screen / full budget tier BEFORE any experiment
        # reads its env knobs (unset-vars only; explicit user requests win).
        # This lives INSIDE the try so an unknown RUN_ALL_PROFILE value
        # still restores the caller's CWD / env via the finally block
        # instead of leaking the ``os.chdir(out_root)`` above.
        profile = _apply_run_profile()
        # Apply the seeds/steps env-var mutations INSIDE the try block
        # (after _ensure_deps/_setup succeeded) so a failure in those
        # functions does not leak the mutations to the caller's
        # process. The finally block at the end of this function
        # restores the originals.
        if seeds is not None:
            os.environ['MQAR_SEEDS'] = str(seeds)
            os.environ['ABL_SEEDS'] = str(seeds)
            os.environ['MECH_SEEDS'] = str(seeds)
        if steps is not None:
            os.environ['MQAR_STEPS'] = str(steps)
            # Ablation (Exp 5) sweeps multiple KDA:CSA:HCA ratios, each trained
            # across ABL_SEEDS seeds. Keep the per-run step count synchronized
            # with the MQAR run so the cross-experiment comparison remains fair
            # and the caller's explicit ``steps`` budget is honored uniformly.
            os.environ['ABL_STEPS'] = str(steps)
            # The mechanism ablation (Exp 8) trains several variants on the
            # same MQAR task; synchronize its budget for the same reason.
            os.environ['MECH_STEPS'] = str(steps)

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

        # Resolve the CPU/SKIP_SLOW policy now and apply its seed floor
        # BEFORE provenance capture and the design manifest. The runners
        # apply the same floor just before they start; doing it here keeps
        # the recorded budget (and the prospective design manifest) equal to
        # the budget the experiments actually run instead of the pre-floor
        # smoke value.
        skip_slow = os.environ.get('SKIP_SLOW', '0').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        is_cpu = not info.has_gpu
        if skip_slow and is_cpu:
            for _seed_var, _seed_default in (('MQAR_SEEDS', 5),
                                             ('ABL_SEEDS', 7),
                                             ('MECH_SEEDS', 5),
                                             ('RECALL_SEEDS', 3)):
                _raise_seed_floor(_seed_var, _seed_default)

        # Store structured provenance rather than only ``repr(EnvInfo)`` so
        # the summary records git commit, torch/CUDA versions, thread count,
        # and the selected KDA backend. This matters when comparing a
        # reference run with a later FLA run.
        from kaggle_setup import capture_provenance
        summary = {
            'env': capture_provenance(),
            'runs': [],
            'run_profile': profile,
        }

        # Pre-registered design manifest. Written BEFORE the experiments so
        # the seed budget, SESOI and required n are on disk independently of
        # the numbers (a plan cannot be edited after seeing the outcome and
        # still be pre-registered). Each runner also attaches its own
        # runtime audit to its result metadata.
        try:
            summary['experiment_design'] = \
                _write_experiment_design_manifest(profile)
        except Exception as exc:  # planning must never block the experiments
            print(f'[run_all] WARNING: could not write the experiment '
                  f'design manifest: {exc}')
            summary['experiment_design'] = {'error': str(exc)}

        try:
            summary['timing_design'] = \
                _write_timing_design_manifest(profile)
        except Exception as exc:  # planning must never block the experiments
            print(f'[run_all] WARNING: could not write the timing design '
                  f'manifest: {exc}')
            summary['timing_design'] = {'error': str(exc)}

        # Import after deps are installed.
        import run_correctness
        import run_kv_cache
        import run_benchmark
        import run_quality
        import run_ablation
        import recall_reach
        import run_mechanism_ablation
        import run_decoding
        import run_indexer_selector
        import run_ppl_calibration
        import run_stats_calibration
        import run_roofline
        import run_budget_frontier
        import test_roofline_schema
        import test_mechanism_ablation
        import test_budget_frontier
        import test_ssd_duality
        import test_quant_recall
        import test_quant_lm
        import test_indexer_selector
        import test_ppl_calibration
        import test_multi_seed_calibration
        import run_quant_recall
        import run_quant_lm
        import method_analysis
        import make_figures

        # 1. Correctness — always run (fast, ~seconds).
        summary['runs'].append(_run('exp1_correctness', run_correctness.main))

        # 1b. Schema/roofline unit tests — the results_schema / roofline
        #     modules carry their own test module; run it here so the full
        #     pipeline covers it.
        summary['runs'].append(
            _run('test_roofline_schema', test_roofline_schema.main))

        # 1c. Mechanism-ablation unit tests — Exp 8's diagnostic and
        #     paired-statistics helpers carry their own test module.
        summary['runs'].append(
            _run('test_mechanism_ablation', test_mechanism_ablation.main))

        # 1d. Budget-frontier unit tests — the kv_budget formulas and the
        #     delta-rule O(1) identities carry their own test module.
        summary['runs'].append(
            _run('test_budget_frontier', test_budget_frontier.main))

        # 1e. SSD-duality unit tests — the Mamba-2 reference scan pinned
        #     against the independent dual (masked-attention) form, with
        #     forward/gradient/contract parity.
        summary['runs'].append(
            _run('test_ssd_duality', test_ssd_duality.main))

        # 1f. Quantized-state codec tests (Exp 10) — kv_quant codec
        #     contract, grouped packing, and report aggregation.
        summary['runs'].append(
            _run('test_quant_recall', test_quant_recall.main))

        # 1g. State-quantized LM tests (Exp 11) — in-place
        #     ``fake_quantize_`` contract, STE gradient identity, stepwise
        #     KDA QAT smoke, and report-row aggregation.
        summary['runs'].append(
            _run('test_quant_lm', test_quant_lm.main))

        # 1h. Lightning-indexer selector tests (Exp 14) — the canonical
        #     top-k selector's two kernels (sort / O(n) boundary) must be
        #     interchangeable, NaN scores must be unselectable without
        #     crashing, and the decode/sink end-to-end contracts must hold.
        summary['runs'].append(
            _run('test_indexer_selector', test_indexer_selector.main))

        # 1i. PPL-machinery calibration tests (Exp 15) — the simulation
        #     engine's RNG-isolation/determinism contract, the committed
        #     calibration report staying calibrated, decode/prefill
        #     perplexity parity, and the schema contract rejecting
        #     tampered certificates.
        summary['runs'].append(
            _run('test_ppl_calibration', test_ppl_calibration.main))

        # 2. KV cache analysis — pure arithmetic, always run.
        summary['runs'].append(_run('exp3_kv_cache', run_kv_cache.main))

        # 3. Method analysis (formulas + headwise demo) — always run.
        summary['runs'].append(_run('method_analysis', method_analysis.main))

        # 4. Latency benchmark — SKIP_SLOW truncates the largest lengths on
        #    CPU (the CSA/HCA per-token Python loops at T=2048).
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

        # 4b. Roofline analysis — pure post-processing over the Exp 2 rows
        #     just written (analytic FLOPs/bytes + on-host peak
        #     microbenchmarks). Always run: cheap and CPU-friendly.
        summary['runs'].append(_run('exp7_roofline', run_roofline.main))

        # 5. MQAR quality — multi-seed.
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
            # ``ABL_STEPS=10`` keeps their chosen budget. 50 is the
            # ceiling (the documented minimum for showing the ablation
            # trend); below that the user is explicitly opting into an
            # under-trained run.
            try:
                _abl_steps = int(os.environ.get('ABL_STEPS', '100'))
            except ValueError:
                _abl_steps = 100
            os.environ['ABL_STEPS'] = str(min(_abl_steps, 50))
        summary['runs'].append(_run('exp5_ablation', run_ablation.main))

        # 5b. Matrix smoke — ONLY in the smoke profile. A 1-seed,
        #     5-layer, 5-step pass over all TWELVE groups (A/B/C/D x
        #     kda/mamba2/rwkv7) proves the ABL_MODE=matrix plumbing
        #     (explicit layouts, ALL THREE memory routes — including the
        #     RWKV-7 route's geometry clamping, state stacking, and result
        #     rows — and the contrast side file) without any meaningful
        #     training budget. Numbers from this step are
        #     pipeline-validation only.
        if profile == 'smoke':
            _matrix_saved = {k: os.environ.get(k) for k in
                             ('ABL_MODE', 'ABL_MATRIX', 'ABL_SEEDS',
                              'ABL_STEPS', 'ABL_TOTAL_LAYERS', 'ABL_MEMORY',
                              'ABL_TRAIN_BATCH', 'ABL_D_MODEL',
                              'RWKV_HEAD_DIM')}
            os.environ.update(
                ABL_MODE='matrix', ABL_MATRIX='main8', ABL_SEEDS='1',
                ABL_STEPS='5', ABL_TOTAL_LAYERS='5',
                ABL_MEMORY='kda,mamba2,rwkv7', ABL_TRAIN_BATCH='8')
            try:
                summary['runs'].append(
                    _run('exp5b_ablation_matrix_smoke', run_ablation.main))
            finally:
                for k, v in _matrix_saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            # LM-scale architecture contrast plumbing: the hybrid and its
            # matched dense-softmax baseline both run through the SAME
            # entry point (2 optimizer steps; pipeline validation only).
            summary['runs'].append(
                _run('exp12_lm_hybrid_smoke',
                     lambda: _run_train_lm_smoke('hybrid')))
            summary['runs'].append(
                _run('exp12_lm_softmax_smoke',
                     lambda: _run_train_lm_smoke('softmax')))

        # 6b. Mechanism ablation — multi-seed. Same SKIP_SLOW policy as
        #     Exp 5: keep the seed count >= 5 for a usable paired test,
        #     truncate the step budget only.
        if skip_slow and is_cpu:
            try:
                _mech_seeds = int(os.environ.get('MECH_SEEDS', '5'))
            except ValueError:
                _mech_seeds = 5
            os.environ['MECH_SEEDS'] = str(max(5, _mech_seeds))
            try:
                _mech_steps = int(os.environ.get('MECH_STEPS', '200'))
            except ValueError:
                _mech_steps = 200
            os.environ['MECH_STEPS'] = str(min(_mech_steps, 50))
        summary['runs'].append(
            _run('exp8_mechanism_ablation', run_mechanism_ablation.main))

        # 6c. Block-size x interleaving associative-recall reach (Exp 9).
        #     Structural tables are analytic; the trained grid uses the
        #     registered seed list (RECALL_SEEDS) and is bound to the
        #     exp9_recall plan by audit_run. Same SKIP_SLOW policy as the
        #     other multi-seed grids: keep the seed count usable (>= 5) and
        #     truncate only the step budget.
        if skip_slow and is_cpu:
            try:
                _recall_seeds = int(os.environ.get('RECALL_SEEDS', '3'))
            except ValueError:
                _recall_seeds = 3
            os.environ['RECALL_SEEDS'] = str(max(5, _recall_seeds))
            try:
                _recall_steps = int(os.environ.get('RECALL_STEPS', '80'))
            except ValueError:
                _recall_steps = 80
            os.environ['RECALL_STEPS'] = str(min(_recall_steps, 50))
        summary['runs'].append(
            _run('exp9_recall', recall_reach.main))

        # 7. Decoding latency.
        summary['runs'].append(_run('exp6_decoding', run_decoding.main))

        # 7a. Lightning-indexer selector (Exp 14) — parity gate + per-token
        #     selection cost. The selector's O(n) boundary kernel replaces
        #     the historical stable sort for sparse long-context decode
        #     (both kernels are canonical; parity is checked before any
        #     timing row is trusted). CPU-cheap (~seconds, no training).
        summary['runs'].append(
            _run('exp14_indexer_selector', run_indexer_selector.main))

        # 7a-bis. PPL-machinery calibration (Exp 15) — simulation-based
        #     calibration certificate for the ppl_bootstrap inference path
        #     (coverage / size control / power at the pre-registered
        #     grid) plus the paired decode-vs-prefill corpus-NLL parity
        #     check. Pure Python, CPU-cheap (~10s); the parity gate runs
        #     before anything is archived.
        summary['runs'].append(
            _run('exp15_ppl_calibration', run_ppl_calibration.main))

        # 7a-ter. Multi-seed inference calibration (Exp 16) — null size,
        #     CI coverage and power of the paired t/sign/permutation panel
        #     used by Exp 4/5/8. Pure Python, CPU-cheap (~2 s); the gate
        #     refuses to archive a certificate with a failed gated row.
        summary['runs'].append(
            _run('exp16_paired_stats_calibration',
                 run_stats_calibration.main))

        # 7b. Budget frontier — analytic curves + measured top-k decode
        #     sweep + joins over Exp 4/5/8 results. On CPU with SKIP_SLOW,
        #     truncate the measured sweep (prefill, decode tokens, repeats);
        #     the analytic curves are always kept.
        if skip_slow and is_cpu:
            try:
                _bp = [int(x.strip()) for x in
                       os.environ.get('BUDGET_PREFILL_LENS',
                                      '256,512').split(',') if x.strip()]
            except ValueError:
                _bp = [256, 512]
            _bp = [v for v in _bp if v <= 256] or [256]
            os.environ['BUDGET_PREFILL_LENS'] = ','.join(str(v) for v in _bp)
            try:
                _bd = int(os.environ.get('BUDGET_DECODE_TOKENS', '10'))
            except ValueError:
                _bd = 10
            os.environ['BUDGET_DECODE_TOKENS'] = str(max(1, min(_bd, 5)))
            try:
                _br = int(os.environ.get('BUDGET_REPEATS', '2'))
            except ValueError:
                _br = 2
            os.environ['BUDGET_REPEATS'] = str(max(1, min(_br, 1)))
        summary['runs'].append(
            _run('exp9_budget_frontier', run_budget_frontier.main))

        # 7c. Quantized-state recall / cost probe (Exp 10) — pure-torch
        #     numeric sweep, CPU-friendly. Under SKIP_SLOW on CPU only the
        #     seed count and probe geometry are truncated (never expanded);
        #     the runner's own defaults are the committed protocol.
        if skip_slow and is_cpu:
            try:
                _qs = [int(x) for x in os.environ.get(
                    'QUANT_SEEDS', '0,1,2').split(',') if x.strip()]
            except ValueError:
                _qs = [0, 1, 2]
            _qs = _qs[:1] or [0]
            os.environ['QUANT_SEEDS'] = ','.join(str(s) for s in _qs)
            try:
                _qnk = int(os.environ.get('QUANT_N_KEYS', '16'))
            except ValueError:
                _qnk = 16
            os.environ['QUANT_N_KEYS'] = str(max(1, min(_qnk, 8)))
        summary['runs'].append(
            _run('exp10_quant_recall', run_quant_recall.main))

        # 7d. State-quantized decode perplexity + QAT recovery (Exp 11) —
        #     LM training dominates the cost. Under SKIP_SLOW on CPU
        #     truncate every budget (one seed, <=50 train/QAT steps,
        #     short sequences, few eval sequences); the fp16 baseline and
        #     the QAT comparison always run so both metrics are covered.
        if skip_slow and is_cpu:
            try:
                _qlm_seeds = [int(x) for x in os.environ.get(
                    'QLM_SEEDS', '0,1,2').split(',') if x.strip()]
            except ValueError:
                _qlm_seeds = [0, 1, 2]
            _qlm_seeds = _qlm_seeds[:1] or [0]
            os.environ['QLM_SEEDS'] = ','.join(str(s) for s in _qlm_seeds)
            for _key, _default, _cap in (
                    ('QLM_STEPS', 400, 50), ('QLM_QAT_STEPS', 400, 50),
                    ('QLM_SEQ_LEN', 64, 32), ('QLM_BATCH', 8, 4),
                    ('QLM_EVAL_SEQS', 13, 4), ('QLM_CORPUS', 64, 32)):
                try:
                    _val = int(os.environ.get(_key, str(_default)))
                except ValueError:
                    _val = _default
                os.environ[_key] = str(max(1, min(_val, _cap)))
        summary['runs'].append(
            _run('exp11_quant_lm', run_quant_lm.main))

        # 7e. Schema integrity gate — every schema-managed result file
        #     written above must parse as a compatible ``results_schema``
        #     payload. This is the pipeline's structural CI check for the
        #     results contract (``ci_gate.py`` itself is frozen).
        summary['runs'].append(
            _run('results_schema_check', _check_results_schema))

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
            def _fig_group_names(rel_name: str, base: str) -> list[str]:
                """Figure filenames produced for one result file's n_kv groups.

                Mirrors ``make_figures.fig_mqar`` / ``fig_ablation``: when the
                result contains exactly one ``n_kv`` group, both the legacy
                ``<base>.pdf`` and ``<base>_nkv<n>.pdf`` are written; with
                several groups only ``<base>_nkv<n>.pdf`` per group is
                written. If the file yields no rows (malformed or all-error),
                no figure is written at all and the legacy name is expected so
                the orchestrator records the incomplete figure set as a
                failure.
                """
                groups = sorted({r.get('n_kv', 1)
                                 for r in make_figures.load(rel_name)})
                if not groups:
                    return [f'{base}.pdf']
                names = [f'{base}_nkv{n}.pdf' for n in groups]
                if len(groups) == 1:
                    names.append(f'{base}.pdf')
                return names
            expected = []
            if os.path.exists(os.path.join(res_dir, 'exp2_benchmark.json')):
                expected.append('fig_benchmark.pdf')
            if os.path.exists(os.path.join(res_dir, 'exp3_kv_cache.json')):
                expected += ['fig_kv_cache.pdf', 'fig_flops.pdf']
            if os.path.exists(os.path.join(res_dir, 'exp4_mqar.json')):
                expected += _fig_group_names('exp4_mqar.json', 'fig_mqar')
            if os.path.exists(os.path.join(res_dir, 'exp5_ablation.json')):
                expected += _fig_group_names('exp5_ablation.json', 'fig_ablation')
            if os.path.exists(os.path.join(res_dir, 'exp6_decoding.json')):
                expected.append('fig_decoding.pdf')
            if os.path.exists(os.path.join(res_dir, 'exp7_roofline.json')):
                expected.append('fig_roofline.pdf')
            if os.path.exists(os.path.join(
                    res_dir, 'exp8_mechanism_ablation.json')):
                expected += _fig_group_names('exp8_mechanism_ablation.json',
                                             'fig_mechanism_ablation')
            if os.path.exists(os.path.join(
                    res_dir, 'exp9_budget_frontier.json')):
                expected.append('fig_budget_frontier.pdf')
            expected.append('fig_architecture.pdf')
            missing = [f for f in expected
                       if not os.path.exists(os.path.join(fig_dir, f))]
            if missing:
                print('[make_figures] incomplete; missing figures: '
                      + ', '.join(missing))
                return 1
            return 0
        summary['runs'].append(_run('make_figures', _make_figs))

        # Final summary. The seed-budget block is captured HERE (not at
        # startup) because the run_profile and the SKIP_SLOW CPU path may
        # raise the seed counts mid-run (e.g. smoke's MQAR_SEEDS=2 is
        # floored to 5 on CPU) — the summary must record what actually ran.
        # All multi-seed experiments use the same scheme: base seed 42,
        # per-run seed = 42 + i.
        summary['seeds'] = {
            'MQAR_SEEDS': os.environ.get('MQAR_SEEDS'),
            'ABL_SEEDS': os.environ.get('ABL_SEEDS'),
            'MECH_SEEDS': os.environ.get('MECH_SEEDS'),
            'RECALL_SEEDS': os.environ.get('RECALL_SEEDS'),
            'QUANT_SEEDS': os.environ.get('QUANT_SEEDS'),
            'QLM_SEEDS': os.environ.get('QLM_SEEDS'),
            'base_seed': 42,
            'scheme': ('per-run seed = base_seed + i (MQAR/ABL/MECH); '
                       'benchmark input RNG is seeded per-(op, T) via '
                       'crc32; decoding/roofline use seed 0; the '
                       'quantization runners use their explicit seed lists '
                       '(QUANT_SEEDS/QLM_SEEDS). See each result file\'s '
                       'provenance/metadata.'),
        }
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
        #
        # The campaign summary is also an archived claim: it lists which
        # experiments passed and how long they took. Attach the same
        # canonical content digest used by result envelopes so an edited
        # summary is detectable by ``reproducibility.py``/``results_schema``.
        from kaggle_setup import write_json_atomic
        from results_schema import attach_integrity
        summary = attach_integrity(_sanitize(summary))
        write_json_atomic(summary, 'results/summary.json',
                          indent=2, allow_nan=False)
        print('\nSaved: results/summary.json')

        # Final campaign entry: bind the synthesis (summary + prospective
        # design manifest + generated figures) into the append-only ledger.
        # Per-experiment entries already hash their own result files; this
        # line makes the campaign-level claims checkable too.
        try:
            from kaggle_setup import (append_run_manifest,
                                      make_run_manifest_entry)
            campaign_artifacts = {
                'summary': 'results/summary.json',
                'experiment_design': 'results/experiment_design.json',
                'timing_design': 'results/timing_design.json',
            }
            for fig_path in sorted(glob.glob('figures/*')):
                if os.path.isfile(fig_path):
                    campaign_artifacts[
                        f'figure:{os.path.basename(fig_path)}'] = fig_path
            append_run_manifest(make_run_manifest_entry(
                experiment='run_all_campaign',
                seeds=[],
                provenance=summary.get('env'),
                hyperparameters={
                    'run_profile': summary.get('run_profile'),
                    'seeds': summary.get('seeds'),
                },
                data={
                    'kind': 'campaign_directory',
                    'n_runs': len(summary.get('runs') or []),
                },
                artifacts=campaign_artifacts,
                results={
                    'n_ok': summary.get('n_ok'),
                    'n_fail': summary.get('n_fail'),
                    'total_time_s': summary.get('total_time_s'),
                },
                status=('completed' if not summary.get('n_fail')
                        else 'partial'),
            ))
            print('Saved: results/run_manifest.jsonl (campaign entry)')
        except Exception as exc:
            # The ledger is bookkeeping; a failure to append the campaign
            # line must not invalidate the experiment outputs themselves.
            print(f'[run_all] WARNING: could not append campaign manifest '
                  f'entry: {type(exc).__name__}: {exc}')

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
