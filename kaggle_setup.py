"""Kaggle / local environment setup and device helpers.

This module is the single entry point for making every experiment in this
repository run transparently on either:

  * a local CPU-only machine (the original paper's setting);
  * a Kaggle notebook with a Tesla T4 (16 GB) GPU;
  * any other CUDA-capable box.

It exposes:

  * ``detect_env()``        -> dict with flags (is_kaggle, has_gpu, device, ...)
  * ``get_device()``         -> ``torch.device``
  * ``setup_kaggle()``       -> call once at the top of a notebook; installs the
                                right torch wheel if CUDA is present but
                                ``torch.cuda.is_available()`` is False.
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

    Idempotent: if ``torch.cuda.is_available()`` is already True, do nothing.
    """
    if torch.cuda.is_available():
        return
    if not (is_kaggle() and _nvidia_smi_available()):
        return
    if verbose:
        print("[kaggle_setup] Kaggle + NVIDIA GPU detected but torch has no CUDA.")
        print("[kaggle_setup] Installing CUDA torch wheel (this happens once)...")
    # Pick the wheel that matches the driver-reported CUDA version. Falls back
    # to cu121 (the historical Kaggle T4 wheel) if probing fails.
    index_url = _detect_cuda_wheel_index()
    if verbose:
        print(f"[kaggle_setup] Using PyTorch wheel index: {index_url}")
    # The upper bound was previously pinned to <2.6, which prevents installation
    # on environments with newer torch (2.6+). Remove the upper bound and rely
    # on ``--upgrade-strategy=only-if-needed`` to avoid unnecessary upgrades.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade-strategy", "only-if-needed",
        "torch>=2.1", "--index-url", index_url,
    ])
    if verbose:
        print("[kaggle_setup] Done. CUDA should now be available after a restart "
              "of the Python process if it was imported before.")


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
    """Sensible DataLoader worker count for the current environment."""
    if is_kaggle():
        return 2
    return max(1, (os.cpu_count() or 2) // 2)


def configure_logging(verbose: bool = True) -> None:
    """Configure the root logger with a single ``StreamHandler`` and a simple
    ``"[%(levelname)s] %(message)s"`` formatter.

    Level is ``INFO`` if ``verbose`` else ``WARNING``. Idempotent: repeated
    calls reset the root logger's handlers so multiple experiments in one
    process do not stack duplicate handlers.

    Called from ``configure_torch_for_device`` so every experiment that calls
    ``configure_torch_for_device`` gets logging set up.
    """
    root = logging.getLogger()
    # Remove any pre-existing handlers so repeated calls do not duplicate
    # output (e.g. when run_all.py invokes several experiment mains in turn).
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO if verbose else logging.WARNING)


def configure_torch_for_device(device: torch.device | None = None) -> EnvInfo:
    """Set global thread count + cudnn flags appropriate for the device.

    Call this once at the top of every experiment script.
    """
    configure_logging(verbose=True)
    info = detect_env()
    if device is None:
        device = info.device
    if device.type == "cpu":
        torch.set_num_threads(info.num_threads)
        torch.set_num_interop_threads(1)
    else:
        # cuDNN autotune is a big win on T4 for the fixed shapes we benchmark.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
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


if __name__ == "__main__":
    # When run directly, probe and optionally install the CUDA wheel.
    setup_kaggle()
    print_env_summary()
