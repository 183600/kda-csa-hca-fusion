"""Kaggle / local environment setup and device helpers.

This module is the single entry point for making every experiment in this
repository run transparently on either:

  * a local CPU-only machine (the original paper's setting);
  * a Kaggle notebook with a Tesla T4 (16 GB) GPU;
  * any other CUDA-capable box.

It exposes:

  * ``detect_env()``        -> dict with flags (is_kaggle, has_gpu, device, ...)
  * ``get_device()``         -> ``torch.device``
  * ``setup_kaggle()``       -> validates CUDA availability and raises
                                ``RuntimeError`` if a GPU was expected but
                                not found. The wheel-install logic lives
                                in ``bootstrap_kaggle_cuda``; this function
                                is the validation entry point.
  * ``to_device(x, device)`` -> moves tensors / modules recursively.
  * ``num_workers()``        -> sensible DataLoader worker count.
  * ``set_global_seed(seed)`` -> seeds Python/NumPy/torch/CUDA (and
                                optionally cuDNN determinism) from one seed.
  * ``seed_worker``          -> deterministic DataLoader ``worker_init_fn``.
  * ``capture_provenance()`` -> environment + git metadata for run records,
    including requested vs. resolved accelerator backends
    (``backend_resolution_provenance()``).
  * ``write_json_atomic()``  -> atomic, strict-JSON result writer.

Design notes
------------
Kaggle's default Python environment ships a CPU-only torch build. To use the T4
GPU we must install the CUDA wheel. We do this *only* when:

  1. we are inside Kaggle (``/kaggle/input`` exists), AND
  2. ``torch.cuda.is_available()`` is False, AND
  3. an NVIDIA GPU is visible to ``nvidia-smi``.

This keeps the local CPU workflow untouched. On current Kaggle GPU images
torch already ships WITH CUDA, so the bootstrap is normally a no-op; it is
the rescue path for the (rarer) CPU-only-torch images.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
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
# Ordered from newest to oldest. cu121 covers drivers that report CUDA
# 12.1-12.3 (older Kaggle T4 images).
# cu126/cu128 are required for torch >= 2.7 wheels (the last cu124 wheel is
# torch 2.6.0), which matters when the preinstalled CPU-only torch on an
# older Kaggle image is newer than 2.6 and the bootstrap must replace it.
_PYTORCH_WHEEL_INDEX_URLS = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
}

# Fallback wheel when the driver-reported CUDA version cannot be probed.
# Known to work on Kaggle T4 (sm_75): torch 2.6.0 ships cu118/cu124/cu126
# wheels, and the driver on current GPU images is >= 12.4.
_DEFAULT_CUDA_WHEEL_KEY = "cu124"

# Per-index EXACT torch pins for the CUDA bootstrap. The pin must satisfy TWO
# constraints simultaneously:
#
#   (a) An exact pin + ``--force-reinstall`` is REQUIRED to guarantee the
#       CPU-only build is actually REPLACED: with a range like
#       ``torch>=2.2,<2.7``, an image that already ships CPU-only torch 2.6.0
#       satisfies the requirement and pip would silently keep the CPU build.
#
#   (b) The pinned version MUST publish a ``+cuXXX`` wheel on the SELECTED
#       index. Because the install uses ``--extra-index-url`` (PyPI stays the
#       primary index), a pin with no wheel on the CUDA index makes pip fall
#       back to PyPI — whose Linux torch wheels are CPU-ONLY — so the install
#       "succeeds" while torch still has no CUDA, and the restart loop below
#       never converges. Verified against download.pytorch.org/whl/<idx>/torch/:
#         cu118: 2.6.0+cu118 exists
#         cu121: last release is 2.5.1+cu121 (2.6.0 has NO cu121 wheel)
#         cu124: 2.6.0+cu124 exists
#         cu126: 2.6.0+cu126 exists
#         cu128: wheels start at 2.7.0+cu128 (2.6.0 has NO cu128 wheel)
#       All pinned versions support sm_75 (Kaggle T4).
_BOOTSTRAP_TORCH_PINS = {
    "cu118": "torch==2.6.0",
    "cu121": "torch==2.5.1",
    "cu124": "torch==2.6.0",
    "cu126": "torch==2.6.0",
    "cu128": "torch==2.7.1",
}


def _detect_cuda_wheel_key() -> str:
    """Probe ``nvidia-smi`` for the driver-reported CUDA version and return the
    best matching PyTorch wheel index KEY (``cu118`` .. ``cu128``).

    Selection rules (newest wheel index whose runtime does not exceed the
    driver; a NEWER driver with an OLDER runtime is always compatible):

        * CUDA >= 12.8 -> cu128
        * CUDA >= 12.6 -> cu126
        * CUDA >= 12.4 -> cu124
        * CUDA >= 12.1 -> cu121
        * otherwise    -> cu118

    Falls back to ``cu124`` if ``nvidia-smi`` is missing, exits non-zero, or
    its output cannot be parsed (e.g. older drivers that don't print the
    ``CUDA Version:`` header).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return _DEFAULT_CUDA_WHEEL_KEY
    if out.returncode != 0:
        return _DEFAULT_CUDA_WHEEL_KEY
    # nvidia-smi prints a header line like:
    #   | NVIDIA-SMI 535.104.05  Driver Version: 535.104.05  CUDA Version: 12.2 |
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out.stdout)
    if m is None:
        return _DEFAULT_CUDA_WHEEL_KEY
    try:
        major, minor = int(m.group(1)), int(m.group(2))
    except ValueError:
        return _DEFAULT_CUDA_WHEEL_KEY
    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (12, 4):
        return "cu124"
    if (major, minor) >= (12, 1):
        return "cu121"
    return "cu118"


def _detect_cuda_wheel_index() -> str:
    """Return the PyTorch wheel index URL for the detected CUDA runtime."""
    return _PYTORCH_WHEEL_INDEX_URLS[_detect_cuda_wheel_key()]


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

    Process-internal torch replacement does NOT work:

    This module does ``import torch`` at the top of the file. Once a
    Python process has imported torch, the loaded ``libtorch.so`` binary
    and the ``torch._C`` extension module are pinned in memory for the
    lifetime of that process. Running ``pip install --upgrade torch``
    replaces the files on disk, but the already-loaded binary in the
    current process keeps the OLD (CPU-only) symbols. Subsequent calls
    to ``torch.cuda.is_available()`` continue to return ``False`` even
    though ``pip show torch`` reports the CUDA build.

    The bootstrap is therefore split from the experiment process:

    1. **Bootstrap** (run once, in a throwaway process or a notebook
       first cell): install the CUDA wheel, then EXIT / restart the
       kernel. ``bootstrap_kaggle_cuda()`` below performs the install
       and raises ``RuntimeError`` to force the caller to restart.
    2. **Experiment**: after restart, call ``setup_kaggle()``. It only
       VERIFIES that CUDA is available; if the user forgot to restart,
       it raises ``RuntimeError`` instead of silently continuing on CPU.

    This makes the failure mode loud: a misconfigured Kaggle run raises
    immediately at startup, rather than producing CPU-only results
    that look like GPU results in the summary.
    """
    if torch.cuda.is_available():
        if verbose:
            print("[kaggle_setup] CUDA is available; no setup needed.")
        return
    if not (is_kaggle() and _nvidia_smi_available()):
        # Not a Kaggle+GPU environment; CPU is the expected config.
        # Do nothing — local CPU runs are still supported.
        return
    # Honor ``SKIP_CUDA_CHECK=1`` so direct callers of ``setup_kaggle()``
    # (e.g. ``python kaggle_setup.py`` from the command line, or a notebook
    # cell that calls ``setup_kaggle()`` directly without going through
    # ``run_all._setup()``) can bypass the CUDA-availability guard on a
    # GPU machine to run CPU-only experiments. The README documents
    # ``SKIP_CUDA_CHECK=1`` as the official bypass.
    if os.environ.get('SKIP_CUDA_CHECK', '0') == '1':
        if verbose:
            print("[kaggle_setup] SKIP_CUDA_CHECK=1: bypassing CUDA "
                  "availability guard.")
        return
    # Kaggle + NVIDIA GPU detected, but torch.cuda.is_available() is False.
    # Either the CUDA wheel was never installed, or it was installed in a
    # previous process and the current process is still running the old
    # CPU-only torch (the most common case when the user ran
    # ``bootstrap_kaggle_cuda()`` but did not restart the kernel).
    #
    # We deliberately DO NOT install the wheel here. Installing it in the
    # current process is a no-op for torch.cuda.is_available() (the binary
    # is already loaded), and continuing to the experiments would silently
    # produce CPU results.
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

    This bootstrap function is split out of ``setup_kaggle()``. It performs
    the pip install that puts the CUDA
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
    index_key = _detect_cuda_wheel_key()
    index_url = _PYTORCH_WHEEL_INDEX_URLS[index_key]
    # Per-index exact pin (see the _BOOTSTRAP_TORCH_PINS comment block): the
    # pinned version must actually publish a +cuXXX wheel on the SELECTED
    # index, otherwise pip silently falls back to PyPI's CPU-only build of
    # the same version and the bootstrap reports success while CUDA stays
    # unavailable (restart loop that never converges).
    torch_pin = _BOOTSTRAP_TORCH_PINS.get(
        index_key, _BOOTSTRAP_TORCH_PINS[_DEFAULT_CUDA_WHEEL_KEY])
    if verbose:
        print(f"[kaggle_setup] Using PyTorch wheel index: {index_url}")
        print(f"[kaggle_setup] Installing {torch_pin} (pin matched to the "
              f"{index_key} index).")
    # Use ``--extra-index-url`` instead of ``--index-url``: the latter
    # REPLACES PyPI entirely, which breaks resolution of any non-torch
    # dependency that torch wheels pull in (e.g. ``typing-extensions``,
    # ``sympy``). ``--extra-index-url`` keeps PyPI as the primary
    # source and adds the PyTorch wheel index as a fallback.
    #
    # CRITICAL: ``--force-reinstall`` with an EXACT version pin
    # (``torch_pin``) is REQUIRED. Without it, pip treats the
    # already-installed (CPU-only) torch as satisfying the requirement and
    # does NOTHING — including when the installed CPU version numerically
    # matches the pin. A range pin (e.g. ``torch>=2.2,<2.7``) has the same
    # hole: an image that already ships CPU-only torch 2.6.0 satisfies it
    # and the CUDA wheel is never installed.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--force-reinstall",
        "--upgrade-strategy", "only-if-needed",
        torch_pin, "--extra-index-url", index_url,
    ])
    # Post-install sanity check: the wheel that landed on disk must be a
    # CUDA build (version carries a ``+cuXXX`` local tag). Query pip in a
    # FRESH subprocess — the current process already imported the old
    # CPU-only torch, and in-process importlib.metadata may serve cached
    # distribution metadata. If the tag is missing, pip fell back to the
    # CPU-only PyPI wheel (pin has no wheel on the selected index) and
    # telling the user to restart would loop forever.
    show = subprocess.run(
        [sys.executable, "-m", "pip", "show", "torch"],
        capture_output=True, text=True, check=False, timeout=120,
    )
    m = re.search(r"^Version:\s*(\S+)\s*$", show.stdout, re.M)
    installed_version = m.group(1) if m else "unknown"
    if not re.search(r"\+cu\d+", installed_version):
        raise RuntimeError(
            f"[kaggle_setup] The bootstrap installed torch=={installed_version} "
            f"from the {index_key} index, but it is NOT a CUDA build (the "
            f"version string has no '+cuXXX' local tag — pip fell back to the "
            f"CPU-only PyPI wheel). This means the exact pin "
            f"{torch_pin!r} has no matching wheel on {index_url}. "
            "The kernel restart below would NOT help; please report this "
            "so the _BOOTSTRAP_TORCH_PINS entry for this index can be "
            "updated to a version that actually publishes a CUDA wheel "
            "there."
        )
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

    Only configure the root logger if it has NO handlers yet
    (``logging.getLogger().handlers == []``). If the host already configured
    logging, we respect its configuration and only adjust the LEVEL
    (downgrading WARNING→INFO is safe; the host's handlers still see the
    records). This mirrors the standard library's ``logging.basicConfig``
    idempotency contract (basicConfig is a no-op if the root already has
    handlers).

    The experiment scripts use ``logging.getLogger(__name__)`` which
    propagates to the root, so configuring root is the right level
    (catches all module loggers uniformly).
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


def _enable_cudnn_autotune_if_safe() -> bool:
    """Enable cuDNN autotune unless determinism was already requested.

    ``torch.backends.cudnn.benchmark = True`` lets cuDNN time candidate
    kernels and choose one per input shape. That choice is not guaranteed
    to be stable run-to-run, so it must not clobber
    ``set_global_seed(deterministic=True)`` (which sets
    ``cudnn.deterministic=True`` / ``benchmark=False``) or an explicit
    ``torch.use_deterministic_algorithms(True)``. Returns whether autotune
    was enabled.
    """
    try:
        if torch.are_deterministic_algorithms_enabled():
            return False
        if torch.backends.cudnn.deterministic:
            return False
        torch.backends.cudnn.benchmark = True
        return True
    except Exception:
        # Never let a provenance/performance convenience break a run.
        return False


def configure_torch_for_device(
    device: torch.device | None = None,
    *,
    verbose: bool = True,
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
    # Apply ``torch.set_num_threads`` on BOTH CPU and GPU branches so the
    # reported value matches the actual value and host-side work is bounded.
    torch.set_num_threads(info.num_threads)
    if device.type == "cpu":
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
        # Enable cuDNN autotune for the fixed input shapes used by the
        # benchmarks, but NEVER override an already-requested deterministic
        # configuration: benchmark mode times candidate kernels and can
        # pick different ones between runs, undoing the same-machine
        # reproducibility guarantee that set_global_seed(deterministic=True)
        # established when it was called before this function.
        _enable_cudnn_autotune_if_safe()
        # TF32 (TensorFloat-32) matmul is only supported on Ampere (sm_80)
        # and later. Kaggle's T4 is sm_75 (Turing), so allow_tf32=True is a
        # no-op there — harmless but misleading (it suggests TF32 acceleration
        # is active when it is not). Guard with a capability check so the
        # flag is only set on GPUs that actually use it: on Ampere+ this
        # enables TF32 matmul (reduced mantissa precision); on Turing it
        # correctly does nothing.
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


def backend_resolution_provenance() -> dict:
    """Requested vs. RESOLVED accelerator backends, best-effort.

    A recall/quality table must be attributable to the kernels that
    actually ran, not the ones the environment requested: with
    ``KDA_BACKEND=auto`` / ``MAMBA_BACKEND=auto`` / ``RWKV_BACKEND=auto``
    a CPU-only box (or one with a broken optional package) silently
    produces reference-implementation numbers while a GPU box runs the
    Triton kernels — same command, same result filename, different
    experiment. The ``*_backend`` keys record the env request; the
    ``*_backend_resolved`` keys record what dispatch will actually do
    (the same ``resolve_*_backend`` helpers the runners use); the
    ``*_available`` booleans distinguish "package absent" from "package
    installed but unusable"; the version keys pin the optional wheels.

    Every probe here runs through the never-raising availability checks,
    and any unexpected failure degrades to ``None`` — provenance capture
    must never be the reason an experiment dies. All values are strict
    JSON (str / bool / None).
    """
    import os
    from importlib import metadata as _importlib_metadata

    requested_kda = os.environ.get('KDA_BACKEND', 'reference')
    requested_mamba = os.environ.get('MAMBA_BACKEND', 'reference')
    requested_rwkv = os.environ.get('RWKV_BACKEND', 'reference')
    out = {
        'mamba_backend': requested_mamba,
        'rwkv_backend': requested_rwkv,
        'kda_backend_resolved': None,
        'mamba_backend_resolved': None,
        'rwkv_backend_resolved': None,
        'fla_available': None,
        'fla_rwkv7_available': None,
        'mamba_ssm_available': None,
        'mamba_ssm_version': None,
    }
    try:
        out['mamba_ssm_version'] = _importlib_metadata.version('mamba-ssm')
    except Exception:
        pass
    # Each backend is probed independently: a failure in one resolver must
    # not erase another backend's record.
    try:
        from ops_kda_backend import fla_available, resolve_kda_backend
        try:
            out['fla_available'] = bool(fla_available())
        except Exception:
            pass
        try:
            out['kda_backend_resolved'] = resolve_kda_backend(requested_kda)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from ops_mamba_backend import (mamba2_official_available,
                                       resolve_mamba_backend)
        try:
            out['mamba_ssm_available'] = bool(mamba2_official_available())
        except Exception:
            pass
        try:
            out['mamba_backend_resolved'] = resolve_mamba_backend(
                requested_mamba)
        except Exception:
            pass
    except Exception:
        pass
    try:
        from ops_rwkv_backend import (resolve_rwkv_backend,
                                      rwkv7_fla_available)
        try:
            out['fla_rwkv7_available'] = bool(rwkv7_fla_available())
        except Exception:
            pass
        try:
            out['rwkv_backend_resolved'] = resolve_rwkv_backend(
                requested_rwkv)
        except Exception:
            pass
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Source-tree identity (reproducibility)
# --------------------------------------------------------------------------
#
# A git commit hash identifies the committed code, but it says nothing about
# a run started from a dirty checkout: two runs can share the same HEAD and
# still execute different code. ``git_dirty`` alone cannot distinguish
# "uncommitted README edit" from "uncommitted operator change". The source
# digest below closes that gap by hashing the working-tree content of every
# code/config file (tracked AND untracked-but-not-ignored, so a new helper
# module is part of the identity too). Paths are framed before content so a
# rename is visible and concatenation is unambiguous.
_SOURCE_FILE_SUFFIXES = frozenset({
    '.py', '.toml', '.txt', '.sh', '.cfg', '.ini', '.yml', '.yaml',
})
_SOURCE_FILE_NAMES = frozenset({'Dockerfile', 'Makefile', 'makefile'})
_SOURCE_TREE_DIGEST_VERSION = b'kda-csa-hca-source-tree-v1\x00'

# Arguments that may carry credentials; their values are redacted before the
# command line is stored in a result file.
_SECRET_ARG_FLAGS = frozenset({
    '--token', '--api-key', '--api_key', '--apikey', '--password',
    '--secret', '--access-token', '--hf-token',
})


def _source_tree_root(root=None) -> str:
    """Return the repository root for ``root`` (or this module's directory)."""
    base = os.path.abspath(
        root if root is not None else os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], cwd=base,
            stderr=subprocess.DEVNULL,
        )
        resolved = out.decode('utf-8', 'replace').strip()
        if resolved:
            return os.path.abspath(resolved)
    except Exception:
        pass
    return base


def _is_source_file(rel_path: str) -> bool:
    """Whether a repo-relative path is part of the code/config identity."""
    name = rel_path.rsplit('/', 1)[-1]
    if name in _SOURCE_FILE_NAMES:
        return True
    dot = name.rfind('.')
    return dot >= 0 and name[dot:].lower() in _SOURCE_FILE_SUFFIXES


def _source_files(root: str) -> list[str]:
    """Repo-relative code/config files, tracked or untracked-but-not-ignored."""
    try:
        raw = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others',
             '--exclude-standard'],
            cwd=root, stderr=subprocess.DEVNULL,
        )
    except Exception:
        raw = None
    if raw is not None:
        files = []
        for chunk in raw.split(b'\0'):
            if not chunk:
                continue
            rel = chunk.decode('utf-8', 'surrogateescape').replace('\\', '/')
            if _is_source_file(rel):
                files.append(rel)
        return sorted(set(files))

    # Git is unavailable (tarball / installed sdist). Walk the tree, skipping
    # generated / cache / hidden directories. The suffix filter keeps the
    # digest scoped to code and configuration rather than data or figures.
    skip_dirs = {
        '__pycache__', 'results', 'figures', 'checkpoints_lm', 'wt103_cache',
        'node_modules', 'venv', '.venv', 'build', 'dist',
    }
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in skip_dirs and not d.startswith('.'))
        for filename in sorted(filenames):
            rel = os.path.relpath(
                os.path.join(dirpath, filename), root).replace(os.sep, '/')
            if _is_source_file(rel):
                files.append(rel)
    return sorted(set(files))


def _source_tree_digest(root=None) -> tuple[str | None, int, int]:
    """Return ``(sha256, n_read, n_missing)`` for the working-tree sources."""
    import hashlib

    root = _source_tree_root(root)
    files = _source_files(root)
    if not files:
        return None, 0, 0
    digest = hashlib.sha256()
    digest.update(_SOURCE_TREE_DIGEST_VERSION)
    n_read = 0
    n_missing = 0
    for rel in files:
        digest.update(rel.encode('utf-8', 'surrogateescape'))
        digest.update(b'\0')
        try:
            with open(os.path.join(root, *rel.split('/')), 'rb') as f:
                data = f.read()
        except OSError:
            # A tracked-but-deleted (or unreadable) file is still part of the
            # identity; mark it so the digest cannot silently equal the
            # pre-deletion tree.
            digest.update(b'MISSING\0')
            n_missing += 1
            continue
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
        n_read += 1
    return digest.hexdigest(), n_read, n_missing


def source_tree_sha256(root=None) -> str | None:
    """Stable SHA-256 of the working-tree code/config files.

    Covers ``.py`` / ``.toml`` / ``.txt`` / ``.sh`` / ``.cfg`` / ``.ini`` /
    ``.yml`` / ``.yaml`` files (tracked plus untracked non-ignored) and the
    conventional ``Dockerfile`` / ``Makefile`` names. This is code identity,
    not a data fingerprint: corpus files are recorded separately by each
    runner's data-version block. Returns ``None`` when no source files are
    found (e.g. a broken package install).
    """
    return _source_tree_digest(root)[0]


def _package_versions(names) -> dict:
    """Best-effort exact versions for the named distributions."""
    from importlib import metadata

    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


def _redact_command_line(argv=None) -> list[str]:
    """Return ``argv`` with credential-bearing flag values redacted."""
    args = list(sys.argv if argv is None else argv)
    out: list[str] = []
    redact_next = False
    for arg in args:
        text = str(arg)
        if redact_next:
            out.append('<redacted>')
            redact_next = False
            continue
        name = text.split('=', 1)[0].lower()
        if name in _SECRET_ARG_FLAGS:
            if '=' in text:
                out.append(f'{text.split("=", 1)[0]}=<redacted>')
            else:
                out.append(text)
                redact_next = True
            continue
        out.append(text)
    return out


def capture_source_identity(root=None) -> dict:
    """Capture the working tree's code identity, including dirty content.

    Unlike a bare ``git rev-parse HEAD``, this records the *content* that a
    run actually executed: the commit, branch, dirty flag/count, and a
    ``source_tree_sha256`` over the working tree. Two dirty runs that share a
    commit but contain different edits therefore receive different identities
    instead of being indistinguishable.

    Every field is best-effort: a tarball without git still gets a source
    digest, and missing git metadata is recorded as ``None`` rather than
    failing the run.
    """
    root = _source_tree_root(root)
    identity = {
        'git_commit': None,
        'git_branch': None,
        'git_describe': None,
        'git_dirty': None,
        'git_dirty_files': None,
        'source_tree_sha256': None,
        'source_file_count': 0,
        'source_missing_count': 0,
    }

    def _git_text(args):
        try:
            return subprocess.check_output(
                ['git'] + list(args), cwd=root,
                stderr=subprocess.DEVNULL,
            ).decode('utf-8', 'replace').strip()
        except Exception:
            return None

    commit = _git_text(['rev-parse', 'HEAD'])
    if commit:
        identity['git_commit'] = commit
    branch = _git_text(['rev-parse', '--abbrev-ref', 'HEAD'])
    if branch:
        identity['git_branch'] = branch
    describe = _git_text(['describe', '--tags', '--always', '--dirty'])
    if describe:
        identity['git_describe'] = describe
    status = _git_text(['status', '--porcelain=v1'])
    if status is not None:
        dirty_lines = [line for line in status.splitlines() if line.strip()]
        identity['git_dirty'] = bool(dirty_lines)
        identity['git_dirty_files'] = len(dirty_lines)

    digest, n_read, n_missing = _source_tree_digest(root)
    identity['source_tree_sha256'] = digest
    identity['source_file_count'] = n_read
    identity['source_missing_count'] = n_missing
    return identity


def capture_provenance() -> dict:
    """Capture environment provenance for result JSON files.

    Returns a dict with torch / CUDA / Python / git-commit / env-var
    metadata so a result JSON is self-describing. Without this, a reader
    picking up a stale JSON has no way to tell which torch version / GPU
    / commit produced it, making cross-version comparisons meaningless.

    ``git_dirty`` records whether the working tree had uncommitted changes
    when the run started: a bare commit hash is not enough to reproduce a
    run whose code differs from HEAD, and silently reporting only the hash
    would make such a run look reproducible when it is not.
    ``source_tree_sha256`` closes the remaining gap: it digests the actual
    working-tree code/config content, so two dirty runs at the same commit
    are distinguishable. The invocation (redacted argv + cwd), exact
    dependency versions, and current determinism/thread settings are
    recorded alongside.

    Also merges :func:`backend_resolution_provenance`: requested backend
    env vars alone are misleading when ``'auto'`` silently resolves to
    the reference path, so the resolved backends / optional-package
    availability / versions are recorded alongside.
    """
    import datetime
    import platform
    import sys
    try:
        from importlib import metadata as _importlib_metadata
        fla_version = _importlib_metadata.version('flash-linear-attention')
    except Exception:
        fla_version = None
    try:
        import numpy as _np
        numpy_version = _np.__version__
    except Exception:
        numpy_version = None
    info = detect_env()
    try:
        source = capture_source_identity()
    except Exception:
        # Provenance capture must never be the reason an experiment dies.
        source = {}
    try:
        backend_info = backend_resolution_provenance()
    except Exception:
        # Even the never-raising helper is guarded here: provenance must
        # never take a run down, whatever happens in an exotic env.
        backend_info = {}
    try:
        package_versions = _package_versions((
            'torch', 'numpy', 'scipy', 'matplotlib', 'einops',
            'transformers', 'datasets', 'tqdm', 'pytest',
            'flash-linear-attention', 'mamba-ssm',
        ))
    except Exception:
        package_versions = {}
    return {
        'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python_version': sys.version.split()[0],
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'torch_version': info.torch_version,
        'cuda_version': info.cuda_version,
        'numpy_version': numpy_version,
        'scipy_version': package_versions.get('scipy'),
        'matplotlib_version': package_versions.get('matplotlib'),
        'einops_version': package_versions.get('einops'),
        'device_name': info.device_name,
        'git_commit': source.get('git_commit'),
        'git_branch': source.get('git_branch'),
        'git_describe': source.get('git_describe'),
        'git_dirty': source.get('git_dirty'),
        'git_dirty_files': source.get('git_dirty_files'),
        'source_tree_sha256': source.get('source_tree_sha256'),
        'source_file_count': source.get('source_file_count'),
        'source_missing_count': source.get('source_missing_count'),
        # Exact command line (credential flags redacted) and working
        # directory: the hyperparameter block alone cannot reconstruct how
        # a run was launched.
        'command_line': _redact_command_line(),
        'working_directory': os.getcwd(),
        'kda_backend': os.environ.get('KDA_BACKEND', 'reference'),
        'flash_linear_attention_version': fla_version,
        'package_versions': package_versions,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', '<unset>'),
        # Deterministic-cuBLAS workspace config (set by
        # ``set_global_seed(deterministic=True)``; ``<unset>`` when the run
        # did not request deterministic algorithms). Recorded so a reader
        # can tell whether cuBLAS GEMMs were pinned to deterministic
        # algorithms for this run.
        'cublas_workspace_config': os.environ.get(
            'CUBLAS_WORKSPACE_CONFIG', '<unset>'),
        'num_threads': info.num_threads,
        'torch_num_threads': _torch_num_threads(),
        'determinism': {
            'torch_deterministic_algorithms': bool(
                torch.are_deterministic_algorithms_enabled()),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        },
        **backend_info,
    }


def _torch_num_threads() -> int | None:
    """Actual intra-op thread count (``None`` if torch cannot report it)."""
    try:
        return int(torch.get_num_threads())
    except Exception:
        return None


def set_global_seed(seed: int, *, deterministic: bool = False) -> dict:
    """Seed every RNG that can affect a training run from one integer.

    Seeds Python ``random``, NumPy (when importable), PyTorch's global CPU
    generator and every CUDA device generator. Call this once per training
    entry point (and again before any model construction that must be
    reproducible independently of earlier RNG consumption).

    ``deterministic=True`` additionally switches cuDNN to deterministic
    kernels and disables the autotuner
    (``torch.backends.cudnn.deterministic = True`` /
    ``benchmark = False``) and asks PyTorch for deterministic algorithm
    implementations (``torch.use_deterministic_algorithms(True,
    warn_only=True)`` — ``warn_only`` avoids hard failures for the few ops
    that have no deterministic implementation). This makes runs on the
    same machine reproducible run-to-run. It cannot make runs
    bit-identical ACROSS hardware architectures; different GPUs/CPU
    kernels legitimately differ in floating-point reduction order.

    Note ``PYTHONHASHSEED`` cannot be changed after interpreter start; set
    it in the environment before launching if string-hash order matters.
    ``capture_provenance`` records its value so runs can be compared.

    ``deterministic=True`` also exports ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    when the variable is unset. PyTorch's deterministic-algorithms mode
    requires this workspace configuration for cuBLAS GEMMs; without it,
    those GEMMs either raise a ``RuntimeError`` (strict mode) or warn and
    fall back to a nondeterministic algorithm (``warn_only=True``, the mode
    used here). The variable must be set BEFORE the first cuBLAS handle is
    created, which is why this function is the right place: every training
    entry point calls it before constructing models or running GPU work.
    A user-supplied value is never overwritten.

    Returns a JSON-serializable dict describing what was seeded, suitable
    for embedding in a run record.
    """
    import random as _random

    seed = int(seed)
    _random.seed(seed)

    # Export the cuBLAS deterministic workspace config BEFORE any CUDA /
    # cuBLAS work in this process (see the docstring). Set even when
    # ``torch.cuda.is_available()`` is False: a later CUDA initialization
    # (e.g. a DataLoader worker or a lazily-built CUDA tensor) would still
    # read it, and the variable is harmless on CPU-only hosts.
    cublas_workspace_config = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if deterministic and cublas_workspace_config is None:
        cublas_workspace_config = ':4096:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_workspace_config

    numpy_seeded = False
    try:
        import numpy as _np
        _np.random.seed(seed % (2 ** 32))
        numpy_seeded = True
    except Exception:
        # NumPy is optional in minimal environments; Python+torch seeding
        # is still complete for this repo's code paths.
        pass

    torch.manual_seed(seed)
    cuda_seeded = False
    try:
        if torch.cuda.is_available():
            # torch.manual_seed already seeds all devices in modern torch,
            # but this is explicit and harmless on old builds.
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
    except Exception:
        pass

    cudnn_deterministic = False
    deterministic_algorithms = False
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn_deterministic = True
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_algorithms = True
        except Exception:
            # Very old torch builds lack the kwarg; cudnn.deterministic above
            # still covers the convolution/cuDNN paths.
            pass

    return {
        'seed': seed,
        'python_random': True,
        'numpy': numpy_seeded,
        'torch': True,
        'cuda': cuda_seeded,
        'deterministic_requested': bool(deterministic),
        'cudnn_deterministic': cudnn_deterministic,
        'deterministic_algorithms': deterministic_algorithms,
        'cublas_workspace_config': cublas_workspace_config,
    }


def seed_worker(worker_id: int) -> None:
    """Deterministic ``DataLoader(worker_init_fn=...)`` for Python/NumPy/torch.

    A DataLoader with ``num_workers > 0`` does NOT inherit the parent
    process's Python/NumPy RNG state in its workers, and PyTorch only seeds
    the torch RNG in each worker. Passing this function as
    ``worker_init_fn`` derives one seed per worker from
    ``torch.initial_seed()`` — which is itself deterministic only when the
    DataLoader is constructed with a seeded base ``generator=`` — and
    applies it to ``random`` and NumPy too.
    """
    import random as _random

    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    _random.seed(worker_seed)
    try:
        import numpy as _np
        _np.random.seed(worker_seed)
    except Exception:
        pass


def parse_int_env(var_name: str, default: int, *, min_value: int = 1,
                  logger: logging.Logger | None = None) -> int:
    """Parse an int environment variable with robust fallback.

    Returns the parsed int value, or ``default`` if the env var is unset OR
    set to a value that cannot be parsed as an int (e.g. ``'abc'``,
    ``'5.0'``) OR that fails the ``min_value`` check. A warning is logged
    when the env var is set but invalid, so the user knows their setting
    was ignored rather than silently dropped.

    This mirrors the robust pattern already used for ``BENCH_REPEATS`` /
    ``BENCH_LENGTHS`` in ``run_benchmark.py``. Extending the same pattern
    to ``MQAR_SEEDS`` / ``MQAR_STEPS`` / ``MQAR_SOFTMAX_STEPS`` /
    ``MQAR_TRAIN_BATCH`` / ``ABL_SEEDS`` / ``ABL_STEPS`` / ``ABL_TRAIN_BATCH``
    surfaces an invalid setting with a warning naming the offending env var
    and the default that was used instead.

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
    if val > sys.maxsize:
        # Python ints are unbounded, but every count from this helper ends
        # up in a range/loop/tensor shape; silently accepting a value that
        # cannot be indexed would turn a typo into a hang or OverflowError.
        log.warning(
            f'invalid {var_name}={raw!r} (larger than sys.maxsize); '
            f'using default {default}')
        return default
    return val


def sanitize_for_json(obj):
    """Recursively replace non-finite floats with ``None`` for strict JSON.

    Mirrors the inline ``_sanitize`` helpers duplicated across
    ``run_kv_cache.py`` / ``run_quality.py`` / ``run_ablation.py`` /
    ``run_decoding.py`` / ``run_all.py``. Centralizing here removes 5
    copies of the same logic and ensures any future change (e.g. handling
    a new edge case) propagates everywhere.

    Python's default ``json.dump`` emits non-standard ``NaN`` / ``Infinity``
    literals which most strict parsers (JS ``JSON.parse``, pandas
    ``read_json``, jq) reject. ``json.dump(..., allow_nan=False)`` raises
    ``ValueError`` instead, but the raised error fires mid-write — leaving
    a partial JSON file that no parser can read. Sanitizing to ``None``
    first lets the strict write succeed cleanly.

    Recurses into dicts, lists, AND tuples (``json.dumps`` serializes
    tuples as JSON arrays, so a tuple containing a NaN/Inf must also be
    sanitized to keep the strict ``allow_nan=False`` write clean).
    """
    import math
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if hasattr(obj, 'ndim') and obj.ndim == 0:
        return sanitize_for_json(obj.item())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(x) for x in obj]
    return obj


def write_json_atomic(payload, target_path: str, *, indent: int = 2,
                      allow_nan: bool = False, default=None,
                      sort_keys: bool = False,
                      trailing_newline: bool = False) -> None:
    """Atomically write ``payload`` as JSON to ``target_path``.

    Uses the standard temp-file + ``os.replace`` pattern:

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
        sort_keys: passed to ``json.dumps`` (default False). Results whose
            consumers used sorted output can keep byte-stable formatting.
        trailing_newline: append ``"\\n"`` before the atomic rename. Some
            existing CLI reports ended with a newline; this preserves that
            convention without a second, non-atomic write.
    """
    import json
    import os
    import tempfile

    # Step 1: serialize to a string. If allow_nan=False and the payload
    # contains NaN/Inf, this raises ValueError BEFORE we touch the
    # filesystem — the target file is never opened for writing.
    text = json.dumps(payload, indent=indent, allow_nan=allow_nan,
                      default=default, sort_keys=sort_keys)
    if trailing_newline:
        text += '\n'

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
    """Create a seeded generator compatible with the requested device.

    CUDA random ops require a CUDA generator; passing a CPU generator to
    ``torch.randn(..., device='cuda', generator=...)`` raises
    ``Expected a 'cuda' device type for generator but found 'cpu'``.

    For an old CUDA build that cannot construct a device-specific generator,
    seed the global CUDA generator and return ``None``. Callers already pass
    this value as ``generator=...``; ``None`` selects the correctly-typed global
    CUDA generator and preserves deterministic execution within that process.
    CPU callers still receive an independent CPU generator.

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
    except Exception as exc:
        if device.type == 'cuda':
            # A CPU generator cannot be passed to CUDA random ops. Fall back
            # to the device's global generator rather than returning an
            # incompatible CPU object.
            torch.cuda.manual_seed(int(seed))
            warnings.warn(
                "CUDA device-specific torch.Generator is unavailable; "
                "using the seeded global CUDA generator. Per-generator RNG "
                f"isolation is unavailable ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        g = torch.Generator()
        g.manual_seed(int(seed))
        return g


def grad_flow_report(named_parameters, *, zero_tol: float = 0.0,
                     max_names: int = 10) -> dict:
    """Audit which trainable tensors received a gradient on the last backward.

    Call AFTER ``loss.backward()`` and BEFORE ``optimizer.step()`` (which
    zeroes/consumes nothing, but ``zero_grad(set_to_none=True)`` clears the
    gradients). The overfit sanity gates use this to turn their pass/fail
    verdict into an actionable diagnostic: a model can still drive a fixed
    batch loss down through the layers that DO get gradients while a
    detached / dead branch silently receives none — exactly the class of
    "gradients not flowing" bug the fixed-batch gate exists to catch.

    ``named_parameters`` is any iterable of ``(name, Parameter)`` pairs
    (e.g. ``model.named_parameters()``).

    ``zero_tol`` is the gradient norm at or below which a tensor counts as
    "zero gradient" (default 0.0: only exactly-zero norms). Returns a
    JSON-serializable dict:

      * ``n_trainable_tensors`` / ``n_with_grad`` / ``n_missing_grad`` /
        ``n_zero_grad`` counts,
      * ``missing_grad_names`` / ``zero_grad_names`` (capped at
        ``max_names``; the caps are recorded so a truncated list is not
        mistaken for the full set),
      * ``global_grad_norm`` (L2 over all parameter gradients).
    """
    import math

    if isinstance(zero_tol, bool) or not isinstance(zero_tol, (int, float)):
        raise ValueError(f'zero_tol must be a real number, got {zero_tol!r}')
    try:
        zero_tol = float(zero_tol)
    except OverflowError:
        raise ValueError(
            f'zero_tol={zero_tol!r} must be representable as a finite float.'
        ) from None
    if not math.isfinite(zero_tol) or zero_tol < 0.0:
        raise ValueError(f'zero_tol must be finite and >= 0, got {zero_tol!r}')
    if isinstance(max_names, bool) or not isinstance(max_names, int):
        raise ValueError(f'max_names must be an exact int, got {max_names!r}')
    if max_names < 0:
        raise ValueError(f'max_names must be >= 0, got {max_names}')

    n_total = 0
    n_missing = 0
    n_zero = 0
    missing: list[str] = []
    zero: list[str] = []
    norms: list[float] = []
    nonfinite_norm = False
    for name, p in named_parameters:
        if not getattr(p, 'requires_grad', False):
            continue
        n_total += 1
        g = p.grad
        if g is None:
            n_missing += 1
            if len(missing) < max_names:
                missing.append(str(name))
            continue
        # fp64 accumulation avoids overflowing a finite fp64 gradient when
        # it is narrowed to fp32, and keeps the squared sum representable
        # below. ``abs`` handles complex gradients while staying real.
        magnitude = g.detach().abs().to(torch.float64)
        if magnitude.numel() == 0:
            norm = 0.0
        else:
            peak = float(magnitude.max())
            if not math.isfinite(peak):
                norm = peak
            elif peak == 0.0:
                norm = 0.0
            else:
                # torch's norm squares in the tensor dtype, so a finite
                # 1e200 gradient overflows to inf; scale before squaring.
                norm = peak * float((magnitude / peak).norm())
        norms.append(norm)
        if not math.isfinite(norm):
            nonfinite_norm = True
        if norm <= zero_tol:
            n_zero += 1
            if len(zero) < max_names:
                zero.append(str(name))
    if nonfinite_norm:
        global_grad_norm = None
    elif not norms:
        global_grad_norm = 0.0
    else:
        scale = max(norms)
        if scale == 0.0:
            global_grad_norm = 0.0
        else:
            global_grad_norm = scale * math.sqrt(math.fsum(
                (norm / scale) ** 2 for norm in norms))
            if not math.isfinite(global_grad_norm):
                global_grad_norm = None
    return {
        'n_trainable_tensors': n_total,
        'n_with_grad': n_total - n_missing,
        'n_missing_grad': n_missing,
        'n_zero_grad': n_zero,
        'zero_tol': float(zero_tol),
        'missing_grad_names': missing,
        'zero_grad_names': zero,
        'names_truncated_at': int(max_names),
        'global_grad_norm': global_grad_norm,
        'note': ('Missing-gradient tensors were not reached by the loss '
                 '(detached or unused branch); zero-gradient tensors are '
                 'reached but contribute nothing on this batch. Both are '
                 'diagnostic evidence for the fixed-batch gate.'),
    }


# --------------------------------------------------------------------------
# Leakage-safe train / validation / test splitting
# --------------------------------------------------------------------------
#
# Scientific-methodology contract enforced by these helpers:
#
#   * The split is drawn BEFORE any fitted preprocessing. Any statistic
#     used to transform features (mean/std for standardization, min/max
#     for scaling, median for imputation, ...) must be computed on the
#     TRAIN indices only and then applied to val/test. These helpers only
#     partition indices, so there is nothing to fit here; the contract is
#     written down so callers apply it downstream.
#   * ``chronological=True`` partitions by position (past -> train,
#     then val, then the most recent data as test) and NEVER shuffles.
#     Within-training batch shuffling of the train partition is still
#     allowed and expected (it does not move information across splits).
#   * ``group_ids=...`` keeps every group entirely inside one split:
#     no individual/entity may straddle train/val/test, otherwise
#     group-level information (user style, recording session, ...)
#     leaks across the boundary.
#   * The split is deterministic for a given ``(n_samples, fractions,
#     seed, group_ids, chronological)`` and uses its own
#     ``random.Random(seed)`` instance, so drawing a split never perturbs
#     the global Python/NumPy/torch RNG streams that drive model init or
#     training batches.
#   * ``audit_split`` is the independent verifier: disjointness,
#     completeness, size sanity and group integrity. Callers should run
#     it on the returned assignment and abort when it fails.


def largest_remainder_sizes(n_units: int, fractions) -> list:
    """Allocate ``n_units`` units proportionally to ``fractions``.

    Uses the largest-remainder (Hamilton) method: each bucket gets the
    floor of its exact share, then remaining units go to the buckets with
    the largest fractional parts (ties broken by bucket order). The
    returned list sums to exactly ``n_units`` and is deterministic, so a
    split's sizes are reproducible without any RNG.

    ``fractions`` are normalized to sum to 1 first, so callers may pass
    any non-negative weights.
    """
    import math
    if isinstance(n_units, bool) or not isinstance(n_units, int):
        raise ValueError(f'n_units must be an exact int, got {n_units!r}')
    if n_units < 0:
        raise ValueError(f'n_units must be >= 0, got {n_units}')
    try:
        float(n_units)
    except OverflowError:
        raise ValueError(
            f'n_units={n_units!r} must be representable as a finite float.'
        ) from None
    try:
        fractions = [float(f) for f in fractions]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'fractions must contain finite real numbers, got {fractions!r}'
        ) from None
    if not fractions:
        raise ValueError('fractions must be non-empty')
    if any(not math.isfinite(f) for f in fractions):
        raise ValueError(f'fractions must be finite, got {fractions}')
    if any(f < 0 for f in fractions):
        raise ValueError(f'fractions must be non-negative, got {fractions}')
    total = sum(fractions)
    if not math.isfinite(total):
        # The weights are finite but their sum overflows. Scale by the
        # largest weight first so the normalized shares stay meaningful.
        scale = max(fractions)
        if scale <= 0:
            raise ValueError(
                f'fractions must have a positive sum, got {fractions}')
        scaled = [f / scale for f in fractions]
        total = math.fsum(scaled)
        fractions = [f / total for f in scaled]
    else:
        if total <= 0:
            raise ValueError(
                f'fractions must have a positive sum, got {fractions}')
        fractions = [f / total for f in fractions]
    raw = [n_units * f for f in fractions]
    sizes = [int(math.floor(x)) for x in raw]
    remainder = n_units - sum(sizes)
    order = sorted(range(len(fractions)),
                   key=lambda i: (-(raw[i] - sizes[i]), i))
    for i in order[:remainder]:
        sizes[i] += 1
    return sizes


def _partition_group_order(order, members, targets, active, buckets):
    """Assign whole groups to consecutive active buckets with balanced cuts.

    The group order is either the seeded shuffle (i.i.d. groups) or the
    first-appearance order (chronological groups). In both cases groups
    must stay whole, so the only legal partitions are contiguous cuts of
    this order. The old greedy walk could overshoot a bucket with one
    large group and then leave the following bucket EMPTY (e.g. group
    sizes 60/30/10 with 10% val + 10% test produced 100/0/0). Here the
    cuts are chosen to minimise the target-relative squared deviation,
    subject to every requested bucket receiving at least one group when
    there are enough groups. Fewer than ``len(active)`` groups cannot be
    partitioned into non-empty requested buckets; the caller's audit then
    rejects the split instead of silently training on an empty split.
    """
    import bisect
    n_groups = len(order)
    k = len(active)
    result = {b: [] for b in buckets}
    if k == 0 or n_groups == 0:
        return result
    if k == 1:
        for g in order:
            result[buckets[active[0]]].extend(members[g])
        return result
    if n_groups < k:
        # Not enough indivisible groups for non-empty requested buckets.
        # Assign one group per bucket in order, then the rest to the last
        # bucket; audit_split will flag the empty bucket(s).
        for pos, g in enumerate(order):
            result[buckets[active[min(pos, k - 1)]]].extend(members[g])
        return result

    sizes = [len(members[g]) for g in order]
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + size)
    total = prefix[-1]

    def _seg_error(size, target):
        # Relative squared deviation: a 2-row error matters much more on a
        # 10-row test set than on a 1000-row train set.
        return ((size - target) ** 2) / float(max(target, 1))

    if k == 2:
        desired = targets[active[0]]
        best = None
        for c in range(1, n_groups):
            err = (_seg_error(prefix[c], desired)
                   + _seg_error(total - prefix[c],
                                targets[active[1]]))
            if best is None or err < best[0]:
                best = (err, c)
        cut = best[1]
        bounds = [0, cut, n_groups]
    else:
        # k == 3 (train/val/test). Search the first cut exhaustively and
        # the second around its target cumulative size.
        desired1 = targets[active[0]]
        desired2 = desired1 + targets[active[1]]
        best = None
        for c1 in range(1, n_groups - 1):
            want = desired2
            start = c1 + 1
            j0 = bisect.bisect_left(prefix, want, start, n_groups)
            for c2 in {j0 - 1, j0, j0 + 1}:
                if not (start <= c2 <= n_groups - 1):
                    continue
                err = (_seg_error(prefix[c1], targets[active[0]])
                       + _seg_error(prefix[c2] - prefix[c1],
                                    targets[active[1]])
                       + _seg_error(total - prefix[c2],
                                    targets[active[2]]))
                if best is None or err < best[0]:
                    best = (err, c1, c2)
        if best is None:
            # Degenerate fallback (should be unreachable for n_groups>=3).
            bounds = [0, 1, n_groups]
        else:
            bounds = [0, best[1], best[2], n_groups]

    for bi, bucket in enumerate(active):
        for g in order[bounds[bi]:bounds[bi + 1]]:
            result[buckets[bucket]].extend(members[g])
    return result


def _validate_strata(strata, n_samples: int):
    """Canonicalize per-row stratum labels (``None`` means unstratified)."""
    if strata is None:
        return None
    if isinstance(strata, (str, bytes)):
        raise ValueError(
            'strata must be a sequence of labels, not a single string')
    try:
        labels = [str(label) for label in strata]
    except TypeError:
        raise ValueError(
            'strata must be a sequence of per-row labels') from None
    if len(labels) != n_samples:
        raise ValueError(
            f'strata has length {len(labels)} but n_samples={n_samples}')
    return labels


def _stratum_seed(seed: int, label: str) -> int:
    """Deterministic per-stratum RNG seed independent of Python hashing."""
    import hashlib
    payload = f'{int(seed)}\x00{label}'.encode('utf-8', 'replace')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def _stratified_partition(strata, *, train_frac: float, val_frac: float,
                          test_frac: float, active, buckets, seed: int):
    """Allocate rows to splits within each stratum, then repair empties.

    Each stratum is shuffled with its own SHA-256-derived RNG seed and cut
    with Hamilton rounding at the GLOBAL split fractions. This keeps every
    sufficiently large stratum represented in every requested split while
    leaving the global row budget approximately unchanged. A requested
    split that is still empty (possible when every stratum is smaller than
    the number of splits) receives one row from the largest bucket; the
    caller's strata audit reports the affected cell as missing rather than
    hiding the imbalance.
    """
    import random
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(strata):
        groups.setdefault(label, []).append(i)
    fractions = [train_frac, val_frac, test_frac]
    result = {bucket: [] for bucket in buckets}
    for label in sorted(groups):
        rows = groups[label]
        random.Random(_stratum_seed(seed, label)).shuffle(rows)
        sizes = largest_remainder_sizes(len(rows), fractions)
        cursor = 0
        for idx in range(3):
            result[buckets[idx]].extend(rows[cursor:cursor + sizes[idx]])
            cursor += sizes[idx]
    n_repairs = 0
    for idx in active:
        key = buckets[idx]
        if result[key]:
            continue
        donors = [bucket for bucket in buckets if len(result[bucket]) > 1]
        if not donors:
            donors = [bucket for bucket in buckets if result[bucket]]
        if not donors:
            break
        donor = max(donors,
                    key=lambda bucket: (len(result[bucket]),
                                        -buckets.index(bucket)))
        result[key].append(result[donor].pop())
        n_repairs += 1
    for key in result:
        result[key].sort()
    return result, n_repairs


def split_train_val_test(n_samples: int, *, val_frac: float = 0.1,
                         test_frac: float = 0.1, seed: int = 0,
                         group_ids=None, chronological: bool = False,
                         strata=None) -> dict:
    """Partition ``range(n_samples)`` into disjoint train / val / test sets.

    Args:
        n_samples: number of rows (or, with ``group_ids``, rows total).
        val_frac: target fraction of rows for validation.
        test_frac: target fraction for the sealed test set.
        seed: deterministic seed for the split. A dedicated
            ``random.Random(seed)`` instance is used, so the global
            Python/NumPy/torch RNG streams are NOT advanced.
        group_ids: optional length-``n_samples`` sequence of group labels
            (user id, document id, recording session, ...). All rows of a
            group land in the SAME split. With groups, ``val_frac`` /
            ``test_frac`` are targets at row granularity; the achieved
            sizes may deviate because whole groups are indivisible.
        chronological: if True, do not shuffle; earlier rows form the
            train set, then val, then the most recent rows form the test
            set. Use this for time series / any data with meaningful
            order, so the model never sees the future of the data it is
            validating or testing on.
        strata: optional length-``n_samples`` sequence of PRE-SPECIFIED
            covariate labels (e.g. fixed token-length bins). The split is
            drawn independently inside every stratum at the same global
            fractions, which keeps the held-out splits representative of
            the covariate and reduces split-to-split variance. Stratifying
            uses only this covariate, never a label, model metric or
            held-out score. Mutually exclusive with ``chronological`` and
            ``group_ids``: ordered or indivisible-group data already has a
            structural split, and silently mixing the two rules would make
            the audit ambiguous.

    Returns:
        ``{'train': [...], 'val': [...], 'test': [...]}`` with sorted
        Python-int index lists. The lists are pairwise disjoint and their
        union is exactly ``range(n_samples)``.
    """
    import math
    import random
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError(
            f'n_samples must be an exact int, got {n_samples!r}')
    n = n_samples
    if n < 1:
        raise ValueError(f'n_samples must be >= 1, got {n}')
    if n > sys.maxsize:
        raise ValueError(
            f'n_samples={n!r} is too large to enumerate as a split.')
    for name, value in (('val_frac', val_frac), ('test_frac', test_frac)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a real number, got {value!r}')
        try:
            number = float(value)
        except OverflowError:
            raise ValueError(
                f'{name}={value!r} must be representable as a finite float.'
            ) from None
        if not math.isfinite(number):
            raise ValueError(f'{name}={value!r} must be finite.')
    val_frac = float(val_frac)
    test_frac = float(test_frac)
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f'val_frac must be in [0, 1), got {val_frac}')
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f'test_frac must be in [0, 1), got {test_frac}')
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0.0:
        raise ValueError(
            f'val_frac + test_frac must leave a positive train fraction, '
            f'got val={val_frac}, test={test_frac}')

    strata_labels = _validate_strata(strata, n)
    if strata_labels is not None and chronological:
        raise ValueError(
            'strata cannot be combined with chronological=True: a '
            'time-ordered split already fixes every row\'s position')
    if strata_labels is not None and group_ids is not None:
        raise ValueError(
            'strata cannot be combined with group_ids: group-aware and '
            'stratified allocation rules would conflict; use one of them')

    if group_ids is not None:
        if len(group_ids) != n:
            raise ValueError(
                f'group_ids has length {len(group_ids)} but n_samples={n}')
        # Stable order of first appearance; chronological mode treats it
        # as the natural (time) order.
        order, seen = [], set()
        members = {}
        for i, g in enumerate(group_ids):
            g = str(g)
            if g not in seen:
                seen.add(g)
                order.append(g)
                members[g] = []
            members[g].append(i)
    else:
        order = list(range(n))
        members = None

    if not chronological:
        random.Random(int(seed)).shuffle(order)

    # Target sizes in row units.
    target_train, target_val, target_test = largest_remainder_sizes(
        n, [train_frac, val_frac, test_frac])
    targets = [target_train, target_val, target_test]
    buckets = ['train', 'val', 'test']

    # Requested buckets come from the FRACTIONS, not the rounded targets:
    # train is always requested (train_frac > 0), val/test only when their
    # fraction is positive. Using ``target > 0`` here silently dropped a
    # requested split on tiny data (e.g. n=3 with default 10% val/test
    # rounded both targets to 0 and put every row in train), even though
    # ``audit_split`` rightly rejects an empty split.
    active = [0]
    if val_frac > 0:
        active.append(1)
    if test_frac > 0:
        active.append(2)
    if len(active) > n:
        raise ValueError(
            f'cannot make {len(active)} non-empty splits from '
            f'n_samples={n}; use more rows or set smaller val/test '
            f'fractions (got val={val_frac}, test={test_frac})')

    # Hamilton rounding can still give a requested bucket zero units when
    # n is small. Move single units from the largest bucket until every
    # requested split has at least one row (possible because n >= active).
    for idx in active:
        while targets[idx] == 0:
            donors = [j for j in range(3)
                      if j != idx and targets[j] > 0]
            if not donors:
                raise ValueError(
                    f'cannot allocate a non-empty {buckets[idx]} split '
                    f'from n_samples={n}')
            donor = max(donors, key=lambda j: targets[j])
            targets[donor] -= 1
            targets[idx] += 1

    result = {'train': [], 'val': [], 'test': []}
    if strata_labels is not None:
        result, _ = _stratified_partition(
            strata_labels, train_frac=train_frac, val_frac=val_frac,
            test_frac=test_frac, active=active, buckets=buckets,
            seed=int(seed))
        return result
    if members is not None:
        # Whole-group partition. The order is the seeded shuffle for
        # i.i.d. groups or first-appearance order for chronological data;
        # either way, cut it into contiguous whole-group blocks.
        result = _partition_group_order(
            order, members, targets, active, buckets)
        for key in result:
            result[key].sort()
        return result

    counts = [0, 0, 0]
    bucket_ptr = 0
    for unit in order:
        while (bucket_ptr < len(active) - 1
               and counts[active[bucket_ptr]] >=
               targets[active[bucket_ptr]]):
            bucket_ptr += 1
        idx = active[bucket_ptr]
        result[buckets[idx]].append(unit)
        counts[idx] += 1
    for key in result:
        result[key].sort()
    return result


def _strata_balance_from_index_sets(index_sets, strata_labels):
    """Covariate-balance block from already-validated per-split indices."""
    keys = ('train', 'val', 'test')
    labels = sorted(set(strata_labels))
    counts = {label: {key: 0 for key in keys} for label in labels}
    n_assigned = sum(len(index_sets[key]) for key in keys)
    for key in keys:
        for i in index_sets[key]:
            if isinstance(i, bool) or not isinstance(i, int):
                continue
            if 0 <= i < len(strata_labels):
                counts[strata_labels[i]][key] += 1
    stratum_sizes = {
        label: sum(counts[label].values()) for label in labels}
    global_fractions = {
        key: (len(index_sets[key]) / n_assigned) if n_assigned else 0.0
        for key in keys}
    max_deviation = 0.0
    missing_cells = 0
    for label in labels:
        size = stratum_sizes[label]
        for key in keys:
            if size:
                max_deviation = max(
                    max_deviation,
                    abs(counts[label][key] / size - global_fractions[key]))
            if index_sets[key] and counts[label][key] == 0:
                missing_cells += 1
    return {
        'n_strata': len(labels),
        'stratum_sizes': stratum_sizes,
        'counts': counts,
        'global_fractions': global_fractions,
        'max_abs_fraction_deviation': max_deviation,
        'n_missing_stratum_split_cells': missing_cells,
        'note': ('descriptive covariate balance, not a leakage gate; '
                 'a stratum smaller than the number of splits cannot '
                 'populate every split'),
    }


def strata_balance_report(split, strata, n_samples: int):
    """Public covariate-balance report for any (possibly filtered) split.

    Unlike :func:`audit_split`, this does not require the split to be
    complete: it is also used AFTER the post-split window-leak filter,
    when some held-out rows have been removed. ``n_samples`` is the
    original cleaned-corpus size that ``strata`` indexes into.
    """
    strata_labels = _validate_strata(strata, n_samples)
    if strata_labels is None:
        return None
    index_sets = {}
    for key in ('train', 'val', 'test'):
        index_sets[key] = [
            i for i in (split.get(key) or [])
            if not isinstance(i, bool) and isinstance(i, int)
            and 0 <= i < n_samples]
    return _strata_balance_from_index_sets(index_sets, strata_labels)


def audit_split(split: dict, n_samples: int, group_ids=None,
                chronological: bool = False, strata=None) -> dict:
    """Independently verify a train/val/test assignment has no leakage.

    Checks, in order:

      * the three keys exist and hold integer indices in range;
      * no index is duplicated WITHIN a split, and no index appears in
        two different splits (cross-split leakage);
      * every index of ``range(n_samples)`` appears exactly once
        (silent row loss would bias the evaluation);
      * train/val/test are non-empty;
      * if ``group_ids`` is given, no group straddles two splits;
      * if ``chronological`` is set, the split is ordered past -> future
        (without groups: every train row precedes every val row, which
        precedes every test row; with groups: first-appearance order of
        the groups follows the same train -> val -> test order). This
        catches a future refactor that shuffles a time-ordered split
        while the caller believes it is chronological.

    ``strata`` adds a DESCRIPTIVE covariate-balance block: per-stratum
    split counts, the global split fractions, and the largest absolute
    deviation from those fractions. It does not change ``ok``: a stratum
    smaller than the number of requested splits cannot populate all of
    them, so balance is reported rather than turned into a pass/fail gate.

    Returns a JSON-serializable report with ``ok`` plus the individual
    booleans and a ``problems`` list. Callers should abort on
    ``ok == False`` instead of training on a compromised split.
    """
    import collections.abc
    if not isinstance(split, collections.abc.Mapping):
        raise TypeError(
            f'audit_split: split must be a mapping, got {type(split).__name__}')
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError(
            f'audit_split: n_samples must be an exact int, got {n_samples!r}')
    n = n_samples
    if n < 0:
        raise ValueError(f'audit_split: n_samples must be >= 0, got {n}')
    if n > sys.maxsize:
        raise ValueError(
            f'audit_split: n_samples={n!r} is too large to audit.')
    strata_labels = _validate_strata(strata, n)
    problems = []
    expected_keys = {'train', 'val', 'test'}
    if set(split) != expected_keys:
        problems.append(
            f'split keys must be {sorted(expected_keys)}, got {sorted(split)}')
    index_sets = {}
    for key in ('train', 'val', 'test'):
        raw_idx = list(split.get(key) or [])
        idx = []
        malformed = False
        for i in raw_idx:
            if isinstance(i, bool) or not isinstance(i, int):
                malformed = True
                continue
            idx.append(i)
        if malformed:
            problems.append(f'{key}: non-integer index present')
        index_sets[key] = idx
        if len(set(idx)) != len(idx):
            problems.append(f'{key}: duplicate indices within the split')
        if idx and (min(idx) < 0 or max(idx) >= n):
            problems.append(f'{key}: index out of range [0, {n})')
    paired_disjoint = True
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = set(index_sets[a]) & set(index_sets[b])
        if overlap:
            paired_disjoint = False
            problems.append(
                f'{a}/{b} overlap on {len(overlap)} row(s), '
                f'first={sorted(overlap)[0]}')
    union = set()
    for idx in index_sets.values():
        union.update(idx)
    # Every valid index is in [0, n), so the union size is the number of
    # assigned rows. Comparing sizes avoids materialising ``set(range(n))``
    # for a huge (but still auditable) sample count.
    complete = len(union) == n
    if not complete:
        missing = n - len(union)
        problems.append(f'{missing} row(s) not assigned to any split')
    non_empty = all(len(v) > 0 for v in index_sets.values())
    if not non_empty:
        empty = [k for k, v in index_sets.items() if not v]
        problems.append(f'empty split(s): {empty}')
    groups_ok = True
    if group_ids is not None:
        if len(group_ids) != n:
            groups_ok = False
            problems.append('group_ids length does not match n_samples')
        else:
            group_to_split = {}
            for key in ('train', 'val', 'test'):
                for i in index_sets[key]:
                    g = str(group_ids[i])
                    prev = group_to_split.get(g)
                    if prev is not None and prev != key:
                        groups_ok = False
                        problems.append(
                            f'group {g!r} straddles {prev} and {key}')
                        break
                    group_to_split[g] = key
    chronological_ok = None
    if chronological:
        if group_ids is not None and len(group_ids) == n:
            # Group order is defined by first appearance (the natural time
            # order of a grouped stream). Whole groups may interleave in
            # row space, so compare group blocks, not min/max row indices.
            first_seen = {}
            for i, g in enumerate(group_ids):
                g = str(g)
                if g not in first_seen:
                    first_seen[g] = i
            blocks = {}
            for key in ('train', 'val', 'test'):
                blocks[key] = [first_seen[str(group_ids[i])]
                               for i in index_sets[key]]
            chronological_ok = bool(
                blocks['train'] and blocks['val'] and blocks['test']
                and max(blocks['train']) < min(blocks['val'])
                and max(blocks['val']) < min(blocks['test']))
        elif group_ids is None:
            chronological_ok = bool(
                index_sets['train'] and index_sets['val']
                and index_sets['test']
                and max(index_sets['train']) < min(index_sets['val'])
                and max(index_sets['val']) < min(index_sets['test']))
        if chronological_ok is False:
            problems.append(
                'chronological split is not ordered past -> future '
                '(train before val before test)')
        elif chronological_ok is None:
            problems.append(
                'chronological audit requested but group_ids length does '
                'not match n_samples')
    strata_report = None
    if strata_labels is not None:
        strata_report = _strata_balance_from_index_sets(
            index_sets, strata_labels)
    return {
        'ok': not problems,
        'pairwise_disjoint': paired_disjoint,
        'complete': complete,
        'non_empty': non_empty,
        'group_integrity': groups_ok,
        'chronological': bool(chronological),
        'chronological_order_verified': chronological_ok,
        'stratified': strata_labels is not None,
        'strata': strata_report,
        'n_samples': n,
        'sizes': {k: len(v) for k, v in index_sets.items()},
        'problems': problems,
    }


def file_sha256(path: str, *, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase SHA-256 of ``path``'s exact bytes.

    Used to bind a run-manifest entry to the artifacts it produced. The
    hash is over the bytes on disk (not a canonical re-serialization), so
    pretty-printing a result file *does* change its artifact digest; that
    is intentional for a byte-level archive check. The envelope
    ``integrity`` digest is the formatting-neutral complement.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(int(chunk_size)), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_bytes(entry: dict) -> bytes:
    """Canonical JSON bytes of a manifest entry without its own digest."""
    import json

    body = {k: v for k, v in entry.items() if k != 'entry_sha256'}
    text = json.dumps(sanitize_for_json(body), sort_keys=True,
                      separators=(',', ':'), allow_nan=False, default=str)
    return text.encode('utf-8')


def manifest_entry_sha256(entry: dict) -> str:
    """Content digest of one run-manifest entry.

    The digest covers every field except ``entry_sha256`` itself, including
    the artifact digests and provenance block, so editing a seed, a metric,
    or an artifact path after the line was appended is detectable. Like the
    envelope integrity digest it is tamper evidence, not a signature.
    """
    import hashlib

    if not isinstance(entry, dict):
        raise TypeError(
            f'manifest_entry_sha256 expects a dict, found '
            f'{type(entry).__name__}')
    return hashlib.sha256(_canonical_manifest_bytes(entry)).hexdigest()


def _artifact_digests(artifacts) -> dict:
    """Hash every existing file named in a role -> path mapping."""
    digests = {}
    if not isinstance(artifacts, dict):
        return digests
    for role, path in artifacts.items():
        if not isinstance(path, str) or not path:
            continue
        try:
            if not os.path.isfile(path):
                continue
            digests[str(role)] = {
                'path': path,
                'sha256': file_sha256(path),
                'n_bytes': os.path.getsize(path),
            }
        except OSError:
            # A missing/unreadable artifact is recorded by its absence;
            # hashing must never be the reason a run fails to log itself.
            continue
    return digests


def write_results_json(payload, target_path, *, indent=2, logger=None):
    """Shared "atomic write + sanitize-fallback" helper.

    Consolidates the ``try: write_json_atomic(allow_nan=False)
    except ValueError: write_json_atomic(sanitize_for_json(...))`` pattern
    so future changes land in one place.

    Mapping payloads are sealed with ``results_schema.attach_integrity``
    before writing, so a legacy dict-style record (provenance sidecar, run
    record, overfit-gate report, ...) carries the same canonical content
    digest as a schema envelope. Bare lists stay bare for backward
    compatibility; their campaign-level identity is provided by the
    run-manifest artifact digest.
    """
    if isinstance(payload, dict):
        try:
            from results_schema import attach_integrity
            payload = attach_integrity(payload)
        except Exception as exc:  # pragma: no cover - defensive
            if logger is not None:
                logger.warning(
                    f'could not attach content integrity to '
                    f'{target_path}: {type(exc).__name__}: {exc}')
    try:
        write_json_atomic(payload, target_path, indent=indent,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        # ``ValueError`` is raised by ``json.dumps(allow_nan=False)`` when
        # the payload contains NaN/Inf floats. ``TypeError`` is raised when
        # the payload contains a non-JSON-serializable type (e.g. a
        # ``torch.Tensor``, a ``numpy.float32`` scalar that is NOT a
        # subclass of ``float``, or a custom object). Catching both and
        # falling back to ``sanitize_for_json`` (which recursively walks
        # containers and replaces non-finite floats with ``None``) at
        # least lets the write succeed; any remaining non-serializable
        # objects are passed through ``json.dumps(default=str)`` below for
        # a best-effort string representation.
        if logger is not None:
            logger.error(
                f'non-finite or non-serializable value in results; '
                f'sanitizing to null / string fallback: {exc}')
        # First try sanitizing (handles NaN/Inf and recurses into
        # containers). If a non-serializable type remains, fall back to
        # ``default=str`` so the write still succeeds (the offending
        # value becomes its ``str()`` representation, which is enough
        # for debugging — the alternative is losing ALL results).
        def _reseal(value):
            # Sanitizing changes the content, so the earlier digest is
            # stale. Recompute it over exactly what is about to be written.
            if isinstance(value, dict):
                try:
                    from results_schema import attach_integrity
                    return attach_integrity(value)
                except Exception:  # pragma: no cover - defensive
                    return value
            return value

        try:
            write_json_atomic(_reseal(sanitize_for_json(payload)),
                              target_path, indent=indent,
                              allow_nan=False)
        except (TypeError, ValueError):
            write_json_atomic(_reseal(sanitize_for_json(payload)),
                              target_path, indent=indent,
                              allow_nan=False, default=str)


# --------------------------------------------------------------------------
# Experiment tracking: append-only run manifest (the machine-readable log)
# --------------------------------------------------------------------------
#
# Every experiment already writes a self-describing result file
# (provenance + hyperparameters + seeds + data version + curves + final
# results). What those per-experiment files cannot answer is "which runs
# exist and where are their artifacts?" across a whole campaign. The run
# manifest is that index: one strict-JSON line per completed run, appended
# to a single ledger, linking the run's identity (experiment, seeds, code
# revision, recipe hash, data fingerprints) to its artifacts and headline
# results. Recording is automatic — no hand-written experiment log.

#: Environment override for the default manifest location (used by tests
#: and by launchers that keep experiment outputs on a scratch volume).
RUN_MANIFEST_PATH_ENV = 'RUN_MANIFEST_PATH'
DEFAULT_RUN_MANIFEST_PATH = os.path.join('results', 'run_manifest.jsonl')


def recipe_sha256(hyperparameters) -> str:
    """Stable sha256 identity of a hyperparameter/config mapping.

    Two runs with the same recipe hash are the SAME configuration; a
    different hash means some hyperparameter changed. Canonicalized as
    sorted-key compact JSON over the NaN-sanitized mapping, so dict
    insertion order and NaN/Inf values cannot change the identity.
    """
    import hashlib
    import json

    canonical = json.dumps(sanitize_for_json(hyperparameters or {}),
                           sort_keys=True, separators=(',', ':'),
                           allow_nan=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def make_run_manifest_entry(*, experiment: str, seeds, provenance=None,
                            hyperparameters=None, data=None, artifacts=None,
                            results=None, status: str = 'completed',
                            manifest_version: int = 1) -> dict:
    """Build one run-manifest entry (JSON-serializable, RFC-8259 strict).

    Args:
        experiment: stable experiment name (e.g. ``'train_lm_autodl'``).
        seeds: the run's random seed(s) — int, or list of ints for
            multi-seed sweeps. Required: a run without its seed recorded
            is not reproducible.
        provenance: ``capture_provenance()`` dict; captured here when None
            so the entry always carries the code revision / env.
        hyperparameters: the recipe (full mapping is stored; the compact
            ``recipe_sha256`` identity is added alongside).
        data: data-version block (corpus fingerprints, split sizes, ...).
        artifacts: mapping of role -> path so every artifact is findable.
            Existing files are also hashed into ``artifact_digests`` (role
            -> path / sha256 / n_bytes), so a later edit is detectable.
        results: final numbers (loss/perplexity/accuracy/verdict, ...).
        status: ``'completed'`` / ``'failed'`` / ``'partial'``.
        manifest_version: schema version for forward compatibility.

    The returned entry is sealed with ``entry_sha256`` over the canonical
    JSON of every other field. ``reproducibility.py --manifest`` recomputes
    it and re-hashes the artifacts to distinguish an untouched ledger from
    a post-hoc edit or a replaced output.
    """
    import datetime

    if seeds is None:
        raise ValueError('run manifest entry requires the run seed(s)')
    if provenance is None:
        provenance = capture_provenance()
    hyperparameters = sanitize_for_json(hyperparameters or {})
    artifacts = sanitize_for_json(artifacts or {})
    entry = {
        'manifest_version': int(manifest_version),
        'experiment': str(experiment),
        'created_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'status': str(status),
        'seeds': seeds,
        'recipe_sha256': recipe_sha256(hyperparameters),
        'hyperparameters': hyperparameters,
        'data': sanitize_for_json(data or {}),
        'artifacts': artifacts,
        # Byte-level identity of every artifact that existed when the entry
        # was created. ``artifacts`` remains the backward-compatible
        # role -> path mapping; this block adds the digest and size so a
        # later edit/replacement is detectable.
        'artifact_digests': _artifact_digests(artifacts),
        'results': sanitize_for_json(results or {}),
        'provenance': provenance,
    }
    # Seal the line last, after every field is fixed. A reader can recompute
    # the digest over the canonical entry and detect post-hoc edits.
    entry['entry_sha256'] = manifest_entry_sha256(entry)
    return entry


def append_run_manifest(entry, path: str | None = None,
                        logger=None) -> str:
    """Append one run-manifest entry as a strict-JSON line; return the path.

    The line is written with a single ``write`` in append mode (atomic for
    lines up to ``PIPE_BUF`` on POSIX), then flushed and fsynced so the
    ledger survives a crash immediately after the call. Path resolution:
    explicit ``path`` argument, else ``$RUN_MANIFEST_PATH``, else
    ``results/run_manifest.jsonl``. Parent directories are created.
    """
    import json

    if path is None:
        path = os.environ.get(RUN_MANIFEST_PATH_ENV,
                              DEFAULT_RUN_MANIFEST_PATH)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(sanitize_for_json(entry), sort_keys=True,
                      allow_nan=False)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()
        os.fsync(f.fileno())
    if logger is not None:
        logger.info(f'  [manifest] appended {entry.get("experiment", "?")} '
                    f'run to {path}')
    return path


if __name__ == "__main__":
    # When run directly, probe and optionally install the CUDA wheel.
    setup_kaggle()
    print_env_summary()
