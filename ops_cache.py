"""Shared ownership rules for module-level tensor caches.

Tensors created inside ``torch.inference_mode()`` are *inference tensors*:
they carry no version counter and cannot be saved for backward. A cache that
is populated during an inference warm-up and later read by a grad-enabled
call therefore either raises

* ``RuntimeError: Inference tensors do not track version counter`` (when the
  cache key reads ``_version``), or
* ``RuntimeError: Inference tensors cannot be saved for backward`` (when the
  cached tensor enters an autograd-saved operation), or
* ``RuntimeError: Inplace update to inference tensor outside InferenceMode``
  (when a cached scratch is written by a later ``no_grad`` call).

Every module-level cache that can be filled under inference mode must build
its entries in :func:`normal_tensor_scope`. The cached values are unchanged
(``torch.inference_mode(False)`` only affects tensor ownership metadata), and
cache hits remain zero-copy because the stored tensor is already ordinary.

Regression oracle: ``test_cache_inference_mode.py``.
"""

from __future__ import annotations

import torch


def normal_tensor_scope():
    """Return a context manager that creates ORDINARY tensors.

    ``torch.inference_mode(False)`` exits inference mode for the duration of
    the scope even when nested inside ``torch.inference_mode()``. Use it
    around the construction of every tensor that is written to a module-level
    cache; do not wrap cache *hits* (an ordinary tensor read under inference
    mode stays ordinary and needs no clone).
    """
    return torch.inference_mode(False)
