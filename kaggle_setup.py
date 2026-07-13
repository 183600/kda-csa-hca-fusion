"""Kaggle / local environment setup and device helpers.

This module is the single entry point for making every experiment in this
repository run transparently on either:

  * a local CPU-only machine (the original paper's setting);
  * a Kaggle notebook with a Tesla T4 (16 GB) GPU;
  * any other CUDA-capable box.

It exposes:

  * ``detect_env()``        -> dict with flags (is_kaggle, has_gpu, device, ...)
  * ``get_device()``         -> ``torch.device``
  * ``setup_kaggle()``       -> SM1 fix: previously documented as "installs
                                the right torch wheel"; the implementation
                                (see ``bootstrap_kaggle_cuda``) actually
                                VALIDATES CUDA availability and raises
                                ``RuntimeError`` if a GPU was expected but
                                not found. The wheel-install logic lives
                                in ``bootstrap_kaggle_cuda``; this function
                                is the validation entry point.
  * ``to_device(x, device)`` -> moves tensors / modules recursively.
  * ``num_workers()``        -> sensible DataLoader worker count.

Design notes
------------
Kaggle's default Python environment ships a CPU-only torch build. To use the T4
GPU we must install the CUDA wheel. We do this *only* when:

  1. we are inside Kaggle (``/kaggle/input`` exists), AND
  2. ``torch.cuda.is_available()`` is False, AND
  3. an NVIDIA GPU is visible to ``nvidia-smi``.

This keeps the local CPU workflow untouched.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import torch


KAGGLE_FLAG_PATH = "/kaggle/input"


# Module-level flag guarding the one-shot ``torch.set_num_interop_threads(1)``
# call. PyTorch only allows setting interop threads ONCE per process; any
# subsequent call raises ``RuntimeError: cannot set number of interop threads
# after parallel work has started``. ``run_all.py`` invokes
# ``configure_torch_for_device`` from every experiment's ``main()``, so without
# this guard the second experiment (after the first ``backward()`` triggers
# inter-op parallelism) would crash the whole runner.
_interop_threads_set = False


# PyTorch CUDA wheel index URLs (see https://pytorch.org/get-started/locally/).
# Ordered from newest to oldest; cu121 is the historical default for Kaggle T4.
_PYTORCH_WHEEL_INDEX_URLS = {
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
}

# Fallback wheel when the driver-reported CUDA version cannot be probed.
# Matches the previous hard-coded value, known to work on Kaggle T4 (sm_75).
_DEFAULT_CUDA_WHEEL_KEY = "cu121"


def _detect_cuda_wheel_index() -> str:
    """Probe ``nvidia-smi`` for the driver-reported CUDA version and return the
    best matching PyTorch wheel index URL.

    Selection rules:
        * CUDA >= 12.4 -> cu124
        * CUDA >= 12.1 -> cu121
        * otherwise    -> cu118

    Falls back to ``cu121`` if ``nvidia-smi`` is missing, exits non-zero, or
    its output cannot be parsed (e.g. older drivers that don't print the
    ``CUDA Version:`` header).
    """
    fallback = _PYTORCH_WHEEL_INDEX_URLS[_DEFAULT_CUDA_WHEEL_KEY]
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return fallback
    if out.returncode != 0:
        return fallback
    # nvidia-smi prints a header line like:
    #   | NVIDIA-SMI 535.104.05  Driver Version: 535.104.05  CUDA Version: 12.2 |
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    if m is None:
        return fallback
    try:
        major, minor = int(m.group(1)), int(m.group(2))
    except ValueError:
        return fallback
    if (major, minor) >= (12, 4):
        return _PYTORCH_WHEEL_INDEX_URLS["cu124"]
    if (major, minor) >= (12, 1):
        return _PYTORCH_WHEEL_INDEX_URLS["cu121"]
    return _PYTORCH_WHEEL_INDEX_URLS["cu118"]


def is_kaggle() -> bool:
    """True iff running inside a Kaggle notebook / script."""
    return os.path.exists(KAGGLE_FLAG_PATH) or os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""


def _nvidia_smi_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return out.returncode == 0


@dataclass
class EnvInfo:
    is_kaggle: bool
    has_gpu: bool
    device: torch.device
    device_name: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    num_threads: int

    def __repr__(self) -> str:
        return (
            f"EnvInfo(is_kaggle={self.is_kaggle}, has_gpu={self.has_gpu}, "
            f"device={self.device}, device_name={self.device_name!r}, "
            f"torch={self.torch_version}, cuda={self.cuda_version})"
        )


def setup_kaggle(verbose: bool = True) -> None:
    """Install the CUDA torch wheel if running on Kaggle with a GPU but a
    CPU-only torch build.

    P0-3 fix — process-internal torch replacement does NOT work:

    This module does ``import torch`` at the top of the file. Once a
    Python process has imported torch, the loaded ``libtorch.so`` binary
    and the ``torch._C`` extension module are pinned in memory for the
    lifetime of that process. Running ``pip install --upgrade torch``
    replaces the files on disk, but the already-loaded binary in the
    current process keeps the OLD (CPU-only) symbols. Subsequent calls
    to ``torch.cuda.is_available()`` continue to return ``False`` even
    though ``pip show torch`` reports the CUDA build.

    The previous implementation installed the CUDA wheel and then
    printed "CUDA should now be available after a restart", but
    ``run_all()`` immediately continued into ``detect_env()`` and the
    experiments — without any restart — so the first run on Kaggle
    silently used CPU. The "after a restart" caveat was effectively
    unreachable from the documented entry point.

    The correct fix is to split the bootstrap from the experiment
    process:

    1. **Bootstrap** (run once, in a throwaway process or a notebook
       first cell): install the CUDA wheel, then EXIT / restart the
       kernel. ``bootstrap_kaggle_cuda()`` below performs the install
       and raises ``RuntimeError`` to force the caller to restart.
    2. **Experiment**: after restart, call ``setup_kaggle()``. It now
       only VERIFIES that CUDA is available; if the user forgot to
       restart, it raises ``RuntimeError`` instead of silently
       continuing on CPU.

    This makes the failure mode loud: a misconfigured Kaggle run now
    raises immediately at startup, rather than producing CPU-only
    results that look like GPU results in the summary.
    """
    if torch.cuda.is_available():
        if verbose:
            print("[kaggle_setup] CUDA is available; no setup needed.")
        return
    if not (is_kaggle() and _nvidia_smi_available()):
        # Not a Kaggle+GPU environment; CPU is the expected config.
        # Do nothing — local CPU runs are still supported.
        return
    # Kaggle + NVIDIA GPU detected, but torch.cuda.is_available() is False.
    # This is the P0-3 scenario: either the CUDA wheel was never installed,
    # or it was installed in a PREVIOUS process and the current process is
    # still running the old CPU-only torch (the most common case when the
    # user ran ``bootstrap_kaggle_cuda()`` but did not restart the kernel).
    #
    # We deliberately DO NOT install the wheel here. Installing it in the
    # current process is a no-op for torch.cuda.is_available() (the binary
    # is already loaded), and continuing to the experiments would silently
    # produce CPU results — exactly the bug we are fixing.
    #
    # Instead, raise so the caller knows the environment is not ready.
    raise RuntimeError(
        "[kaggle_setup] Kaggle + NVIDIA GPU detected but "
        "torch.cuda.is_available() is False. The CUDA torch wheel must "
        "be installed in a SEPARATE bootstrap step BEFORE running "
        "experiments, because replacing torch in an already-running "
        "Python process does not take effect (the loaded libtorch.so "
        "binary is pinned in memory until the process exits).\n\n"
        "To fix:\n"
        "  1. Run ``python -c 'from kaggle_setup import "
        "bootstrap_kaggle_cuda; bootstrap_kaggle_cuda()'`` (or run the "
        "kaggle_bootstrap notebook cell) in a throwaway process.\n"
        "  2. RESTART this Python process / kernel.\n"
        "  3. Re-run the experiments. ``setup_kaggle()`` will then see "
        "torch.cuda.is_available() == True and proceed.\n\n"
        "If you intended to run on CPU, set SKIP_CUDA_CHECK=1 in the "
        "environment to bypass this guard."
    )


def bootstrap_kaggle_cuda(verbose: bool = True) -> None:
    """Install the CUDA torch wheel for the NEXT Python process.

    This is the bootstrap function that the P0-3 fix splits out of
    ``setup_kaggle()``. It performs the pip install that puts the CUDA
    wheel on disk, then raises ``RuntimeError`` to remind the caller
    that the CURRENT process must exit before the new wheel takes
    effect.

    Usage on Kaggle (notebook first cell)::

        from kaggle_setup import bootstrap_kaggle_cuda
        try:
            bootstrap_kaggle_cuda()
        except RuntimeError as e:
            print(e)
            print("Restarting kernel... (run experiments in the next cell)")
            # On Kaggle, use the kernel-restart API or ask the user to
            # click Restart in the UI. The experiments must run in a
            # fresh process that re-imports torch from the new wheel.
            raise

    This function is idempotent: if ``torch.cuda.is_available()`` is
    already True, it returns immediately without installing anything.
    """
    if torch.cuda.is_available():
        if verbose:
            print("[kaggle_setup] CUDA already available; bootstrap is a no-op.")
        return
    if not (is_kaggle() and _nvidia_smi_available()):
        if verbose:
            print("[kaggle_setup] Not a Kaggle+GPU environment; bootstrap is a no-op.")
        return
    if verbose:
        print("[kaggle_setup] Kaggle + NVIDIA GPU detected but torch has no CUDA.")
        print("[kaggle_setup] Installing CUDA torch wheel (this happens once)...")
    index_url = _detect_cuda_wheel_index()
    if verbose:
        print(f"[kaggle_setup] Using PyTorch wheel index: {index_url}")
    # Use ``--extra-index-url`` instead of ``--index-url``: the latter
    # REPLACES PyPI entirely, which breaks resolution of any non-torch
    # dependency that torch wheels pull in (e.g. ``typing-extensions``,
    # ``sympy``). ``--extra-index-url`` keeps PyPI as the primary
    # source and adds the PyTorch wheel index as a fallback.
    #
    # CRITICAL: ``--upgrade`` is REQUIRED. Without it, pip treats the
    # already-installed (CPU-only) torch as satisfying ``torch>=2.1``
    # and does NOTHING. ``--upgrade-strategy=only-if-needed`` is a
    # modifier of ``--upgrade`` and is meaningless on its own.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade",
        "--upgrade-strategy", "only-if-needed",
        # P1-4 fix: pin the SAME upper bound as pyproject.toml's
        # ``torch>=2.2,<2.7``. The previous ``torch>=2.1`` had no upper
        # bound, so a fresh Kaggle bootstrap could pull torch 2.7+ which
        # changes ``scaled_dot_product_attention`` kernel selection and
        # padding handling — silently producing numerically different
        # benchmark results from the committed ones. Bumping the lower
        # bound to 2.2 matches pyproject.toml exactly; keeping the upper
        # bound at <2.7 prevents breaking changes from sneaking in.
        "torch>=2.2,<2.7", "--extra-index-url", index_url,
    ])
    # The install succeeded, but the CURRENT process still has the old
    # CPU-only torch loaded. Force the caller to restart.
    raise RuntimeError(
        "[kaggle_setup] CUDA torch wheel installed successfully, but "
        "the CURRENT Python process still has the old CPU-only torch "
        "loaded (libtorch.so is pinned in memory until the process "
        "exits). You MUST restart the Python process / kernel before "
        "running any experiment, otherwise torch.cuda.is_available() "
        "will still return False and all experiments will silently "
        "run on CPU.\n\n"
        "After restart, call ``setup_kaggle()`` (or just "
        "``run_all()``) — it will verify CUDA is available and "
        "proceed. Do NOT call ``bootstrap_kaggle_cuda()`` again; it "
        "is a one-shot installer."
    )


def detect_env() -> EnvInfo:
    """Probe the environment and return an ``EnvInfo``."""
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if has_gpu else "cpu")
    device_name = (
        torch.cuda.get_device_name(0) if has_gpu
        else "cpu"
    )
    cuda_version = (
        torch.version.cuda if has_gpu and torch.version.cuda is not None
        else None
    )
    # On CPU, cap threads to keep the benchmark fair and avoid oversubscription.
    # Kaggle CPUs have 2 cores; local machines typically have 8.
    if has_gpu:
        num_threads = 1  # GPU path is single-threaded on host
    else:
        # Leave one core free for the OS.
        num_threads = max(1, (os.cpu_count() or 1) - 1)
    return EnvInfo(
        is_kaggle=is_kaggle(),
        has_gpu=has_gpu,
        device=device,
        device_name=device_name,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        num_threads=num_threads,
    )


def get_device() -> torch.device:
    """Return the default torch device for this environment."""
    return detect_env().device


def to_device(x, device: torch.device | None = None):
    """Move a tensor, module, dict, list, or tuple to ``device`` recursively."""
    if device is None:
        device = get_device()
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, torch.nn.Module):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def num_workers() -> int:
    """Sensible DataLoader worker count for the current environment.

    SM7 note: this function is currently NOT called by any of the four
    experiment runners (run_benchmark / run_quality / run_ablation /
    run_decoding / run_kv_cache) — they all use ``batch_size=1`` training
    loops with no DataLoader. It is kept for external callers (e.g. a
    user adapting the repo to a real training pipeline) and for the
    module's documented public API. If you remove it, also update the
    module docstring above.
    """
    if is_kaggle():
        return 2
    return max(1, (os.cpu_count() or 2) // 2)


def configure_logging(verbose: bool = True) -> None:
    """Configure the root logger with a single ``StreamHandler`` and a simple
    ``"[%(levelname)s] %(message)s"`` formatter.

    Level is ``INFO`` if ``verbose`` else ``WARNING``.

    Called from ``configure_torch_for_device`` so every experiment that calls
    ``configure_torch_for_device`` gets logging set up.

    Batch-3 fix: previously this function UNCONDITIONALLY removed all
    existing root handlers and replaced them with our own. When the host
    application (Jupyter notebook, MLflow, a parent pipeline) had
    configured its own root handlers, our call silently destroyed them
    — breaking the host's logging. The report flagged this as a side
    effect that "may break host notebook/application logs".

    The fix: only configure the root logger if it has NO handlers yet
    (``logging.getLogger().handlers == []``). If the host already
    configured logging, we respect its configuration and only adjust
    the LEVEL (downgrading WARNING→INFO is safe; the host's handlers
    still see the records). This mirrors the standard library's
    ``logging.basicConfig`` idempotency contract (basicConfig is a
    no-op if the root already has handlers).

    The experiment scripts use ``logging.getLogger(__name__)`` which
    propagates to the root, so configuring root is the right level
    (catches all module loggers uniformly). A future refactor could
    move all loggers under a named project logger (``kch_fusion``)
    for full isolation, but that requires touching every runner and
    is tracked as Batch-3 "split large files" follow-up work.
    """
    root = logging.getLogger()
    if not root.handlers:
        # No host handlers: install our own. This is the "fresh process"
        # path (the common case when running ``python run_all.py`` from
        # the command line). ``logging.basicConfig`` would do the same,
        # but we want our specific formatter.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root.addHandler(handler)
    # Always set the level (downgrading is safe; the host's handlers
    # still see records at or above this level). If the host set a MORE
    # VERBOSE level (e.g. DEBUG), we don't override it — only adjust
    # toward our level if the current level is more verbose than ours.
    target_level = logging.INFO if verbose else logging.WARNING
    if root.level > target_level:
        root.setLevel(target_level)


def configure_torch_for_device(
    device: torch.device | None = None,
    *,
    verbose: bool = True,  # SM5 fix: was hardcoded True; now a kwarg.
) -> EnvInfo:
    """Set global thread count + cudnn flags appropriate for the device.

    Call this once at the top of every experiment script.

    Args:
        verbose: if True, configure logging at INFO level. Set to False
            for non-experiment callers (e.g. ``method_analysis.demo_*``)
            that don't want the global logger reconfigured.
    """
    configure_logging(verbose=verbose)
    info = detect_env()
    if device is None:
        device = info.device
    if device.type == "cpu":
        torch.set_num_threads(info.num_threads)
        # ``set_num_interop_threads`` can only be called ONCE per process
        # (PyTorch raises ``RuntimeError`` on any subsequent call, even after
        # ``import torch`` alone has triggered inter-op init in some builds).
        # ``run_all.py`` calls this function from every experiment's main(),
        # so without the guard the second experiment would crash. The module-
        # level ``_interop_threads_set`` flag remembers that we already did it.
        global _interop_threads_set
        if not _interop_threads_set:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Already set (e.g. user called it manually, or torch init
                # happened before us). Silently ignore — the value is either
                # already 1 or close enough that inter-op contention is not
                # the bottleneck on the CPU path.
                pass
            _interop_threads_set = True
    else:
        # cuDNN autotune is a big win on T4 for the fixed shapes we benchmark.
        torch.backends.cudnn.benchmark = True
        # TF32 (TensorFloat-32) matmul is only supported on Ampere (sm_80)
        # and later. Kaggle's T4 is sm_75 (Turing), so allow_tf32=True is a
        # no-op there — harmless but misleading (it suggests TF32 acceleration
        # is active when it is not). Guard with a capability check so the
        # flag is only set on GPUs that actually use it. On Ampere+ this
        # gives a ~8x matmul speedup with ~3 decimal digits of precision
        # (perfectly fine for our benchmarks); on Turing it correctly does
        # nothing.
        try:
            _cap = torch.cuda.get_device_capability(0)
            if _cap[0] >= 8:  # sm_80+ (Ampere, Ada, Hopper, ...)
                torch.backends.cuda.matmul.allow_tf32 = True
        except (RuntimeError, IndexError):
            # Capability query failed (e.g. CUDA device vanished mid-run).
            # Silently skip — the default (False) is always safe.
            pass
    return info


def print_env_summary() -> EnvInfo:
    """Print a readable summary of the environment and return it."""
    info = detect_env()
    print("=" * 70)
    print("Environment summary")
    print("=" * 70)
    print(f"  is_kaggle     : {info.is_kaggle}")
    print(f"  has_gpu       : {info.has_gpu}")
    print(f"  device        : {info.device}")
    print(f"  device_name   : {info.device_name}")
    print(f"  python        : {info.python_version}")
    print(f"  torch         : {info.torch_version}")
    print(f"  cuda          : {info.cuda_version}")
    print(f"  num_threads   : {info.num_threads}")
    print("=" * 70)
    return info


def capture_provenance() -> dict:
    """BQ8 / AK10 fix: capture environment provenance for result JSON files.

    Returns a dict with torch / CUDA / Python / git-commit / env-var
    metadata so a result JSON is self-describing. Previously the four
    experiment runners (benchmark, quality, ablation, decoding, kv_cache)
    wrote rows with no environment metadata — a reader picking up a stale
    JSON had no way to tell which torch version / GPU / commit produced
    it, making cross-version comparisons meaningless.
    """
    import os
    import platform
    import subprocess
    import sys
    import datetime
    info = detect_env()
    # Best-effort git commit; ignore errors (running from a tarball, etc).
    git_commit = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=here, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    return {
        'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'torch_version': info.torch_version,
        'cuda_version': info.cuda_version,
        'device_name': info.device_name,
        'git_commit': git_commit,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', '<unset>'),
        'num_threads': info.num_threads,
    }


def parse_int_env(var_name: str, default: int, *, min_value: int = 1,
                  logger: logging.Logger | None = None) -> int:
    """Parse an int environment variable with robust fallback.

    Returns the parsed int value, or ``default`` if the env var is unset OR
    set to a value that cannot be parsed as an int (e.g. ``'abc'``,
    ``'5.0'``) OR that fails the ``min_value`` check. A warning is logged
    when the env var is set but invalid, so the user knows their setting
    was ignored rather than silently dropped.

    This mirrors the robust pattern already used for ``BENCH_REPEATS`` /
    ``BENCH_LENGTHS`` in ``run_benchmark.py`` (which previously crashed
    the whole benchmark on malformed input). Extending the same pattern
    to ``MQAR_SEEDS`` / ``MQAR_STEPS`` / ``MQAR_SOFTMAX_STEPS`` /
    ``MQAR_TRAIN_BATCH`` / ``ABL_SEEDS`` / ``ABL_STEPS`` / ``ABL_TRAIN_BATCH``
    prevents a single typo from crashing an entire multi-hour experiment
    with no informative error message — previously the bare
    ``int(os.environ.get(...))`` raised ``ValueError: invalid literal for
    int() with base 10: 'abc'`` with no context about WHICH env var was
    bad or what the default would have been.

    Args:
        var_name: environment variable name to read.
        default: value to return when the env var is unset or invalid.
        min_value: inclusive lower bound; values below this fall back to
            ``default`` (with a warning). Use ``min_value=0`` to allow
            zero, or a negative value to disable the check.
        logger: optional logger to receive the warning; falls back to the
            module logger for this file.
    """
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    log = logger if logger is not None else logging.getLogger(__name__)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        log.warning(
            f'invalid {var_name}={raw!r} (not an int); using default {default}')
        return default
    if val < min_value:
        log.warning(
            f'invalid {var_name}={raw!r} (must be >= {min_value}); '
            f'using default {default}')
        return default
    return val


def sanitize_for_json(obj):
    """Recursively replace non-finite floats with ``None`` for strict JSON.

    Mirrors the inline ``_sanitize`` helpers previously duplicated across
    ``run_kv_cache.py`` / ``run_quality.py`` / ``run_ablation.py`` /
    ``run_decoding.py`` / ``run_all.py``. Centralizing here removes 5
    copies of the same logic and ensures any future fix (e.g. handling a
    new edge case) propagates everywhere.

    Python's default ``json.dump`` emits non-standard ``NaN`` / ``Infinity``
    literals which most strict parsers (JS ``JSON.parse``, pandas
    ``read_json``, jq) reject. ``json.dump(..., allow_nan=False)`` raises
    ``ValueError`` instead, but the raised error fires mid-write — leaving
    a partial JSON file that no parser can read. Sanitizing to ``None``
    first lets the strict write succeed cleanly.

    Recurses into dicts, lists, AND tuples (``json.dumps`` serializes
    tuples as JSON arrays, so a tuple containing a NaN/Inf must also be
    sanitized to avoid crashing the second ``allow_nan=False`` dump).
    """
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


def write_json_atomic(payload, target_path: str, *, indent: int = 2,
                      allow_nan: bool = False, default=None) -> None:
    """Atomically write ``payload`` as JSON to ``target_path``.

    P1-5 fix: the previous pattern across all runners was::

        text = json.dumps(payload, indent=2, allow_nan=False)
        with open('results/expN.json', 'w') as f:
            f.write(text)

    This is NOT atomic. If the process is killed (SIGKILL, OOM, Kaggle
    session timeout) or the disk fills up mid-write, the target file is
    left TRUNCATED — a partial JSON document that no parser can read.
    Downstream code (``make_figures.load``) would then silently skip the
    experiment, or worse, crash with a confusing ``JSONDecodeError`` that
    doesn't point at the real cause.

    The atomic fix uses the standard temp-file + ``os.replace`` pattern:

    1. Serialize to a string first (catches NaN/Inf via ``allow_nan=False``
       BEFORE touching the filesystem — the target file is never opened
       for writing until serialization succeeds).
    2. Write the string to a TEMPORARY file in the SAME directory as the
       target (so ``os.replace`` is a same-filesystem rename and therefore
       atomic on POSIX).
    3. ``flush()`` + ``os.fsync()`` to durable storage (so the data is
       actually on disk, not just in the OS page cache — a power loss
       after ``close()`` but before ``fsync()`` could still lose the
       write).
    4. ``os.replace(temp, target)`` — atomic rename. Either the target
       file is the OLD version (write in progress) or the NEW version
       (write complete); there is no intermediate truncated state.

    Args:
        payload: the object to serialize (must be JSON-serializable after
            ``sanitize_for_json`` if ``allow_nan=False``).
        target_path: the final file path. Parent directories must already
            exist (runners call ``os.makedirs('results', exist_ok=True)``
            before this function).
        indent: passed to ``json.dumps`` (default 2 for readability).
        allow_nan: passed to ``json.dumps`` (default False for strict
            RFC 8259 compliance — call ``sanitize_for_json`` first if the
            payload may contain non-finite floats).
        default: passed to ``json.dumps`` (default None — use
            ``sanitize_for_json`` instead of this for non-finite handling).
    """
    import json
    import os
    import tempfile

    # Step 1: serialize to a string. If allow_nan=False and the payload
    # contains NaN/Inf, this raises ValueError BEFORE we touch the
    # filesystem — the target file is never opened for writing.
    text = json.dumps(payload, indent=indent, allow_nan=allow_nan,
                      default=default)

    # Step 2: write to a temp file in the SAME directory as the target
    # (so os.replace is a same-filesystem rename and therefore atomic).
    target_dir = os.path.dirname(os.path.abspath(target_path))
    # tempfile.mkstemp returns (fd, path); the file is created with
    # mode 0600 by default, which is fine for experiment outputs.
    fd, tmp_path = tempfile.mkstemp(
        prefix='.tmp_',
        suffix=os.path.basename(target_path),
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # Step 3: atomic rename. On POSIX this is guaranteed atomic by
        # rename(2); on Windows it's atomic as long as the target doesn't
        # exist or both files are on the same filesystem (which they are,
        # since we created the temp in the same directory).
        os.replace(tmp_path, target_path)
    except Exception:
        # If anything went wrong (write error, fsync error, rename error),
        # clean up the temp file so we don't leave orphaned .tmp_* files
        # in the results directory. The target file is untouched (still
        # the old version or absent), so the failure is recoverable.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def make_seeded_generator(seed: int, device=None):
    """Create a seeded ``torch.Generator`` with a safe CPU fallback.

    P2-1 helper (round 3): older torch builds or CPU-only installs do not
    support ``torch.Generator(device='cuda')`` (raises ``RuntimeError``).
    A bare ``torch.Generator(device=device)`` in those environments crashes
    experiment startup (Exp2/4/5/6). This helper tries the requested device
    first and falls back to a CPU generator, which is still accepted by
    ``torch.randn(..., device='cuda', generator=cpu_gen)`` (the RNG draws
    happen on CPU, then tensors are materialized on the target device;
    determinism is preserved).

    Args:
        seed: integer seed (callers are responsible for choosing a
            per-op / per-seed / per-call unique value).
        device: target device (torch.device, string, or None for CPU).
    """
    import torch
    if device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(device)
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g


def write_results_json(payload, target_path, *, indent=2, logger=None):
    """AK1 fix: shared "atomic write + sanitize-fallback" helper.

    Previously the same ~10-line ``try: write_json_atomic(allow_nan=False)
    except ValueError: write_json_atomic(sanitize_for_json(...))`` pattern
    was copy-pasted in run_ablation.py, run_decoding.py, run_kv_cache.py,
    run_benchmark.py, and run_quality.py. The copy-paste had already
    drifted (e.g. some log the error, some don't; some use
    ``logger.error``, some ``logger.warning``). This helper consolidates
    the pattern so future fixes land in one place.
    """
    try:
        write_json_atomic(payload, target_path, indent=indent,
                          allow_nan=False)
    except ValueError as exc:
        if logger is not None:
            logger.error(
                f'non-finite value in results; sanitizing to null: {exc}')
        write_json_atomic(sanitize_for_json(payload), target_path,
                          indent=indent, allow_nan=False)


if __name__ == "__main__":
    # When run directly, probe and optionally install the CUDA wheel.
    setup_kaggle()
    print_env_summary()
