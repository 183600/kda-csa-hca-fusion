"""Reproducibility / experiment-tracking regression tests.

Covers the shared RNG helpers in ``kaggle_setup``
(:func:`set_global_seed`, :func:`seed_worker`,
:func:`make_seeded_generator`), the pre-training overfit sanity gate in
``train_lm_autodl``, and the seed-recording contract of the experiment
runners.

How to run::

    pytest -q test_reproducibility.py          # from the repo root

The file is intentionally standalone (no conftest dependency);
``pyproject.toml`` sets ``testpaths = ["."]`` so any bare ``pytest`` run
collects it.
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import (  # noqa: E402
    make_seeded_generator,
    seed_worker,
    set_global_seed,
)
from ops_fused import HybridConfig  # noqa: E402
from train_lm_autodl import LMWithHybrid, run_overfit_sanity_check  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_determinism():
    """Keep ``set_global_seed(deterministic=True)`` side effects test-local."""
    cudnn_det = torch.backends.cudnn.deterministic
    cudnn_bench = torch.backends.cudnn.benchmark
    det_algos = torch.are_deterministic_algorithms_enabled()
    yield
    torch.backends.cudnn.deterministic = cudnn_det
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.use_deterministic_algorithms(det_algos)


def _draw_all_rngs():
    values = [
        torch.randn(4).tolist(),
        [random.random() for _ in range(3)],
        torch.randint(0, 1000, (4,)).tolist(),
    ]
    try:
        import numpy as np
        values.append(np.random.rand(4).tolist())
    except ImportError:
        pass
    return values


def test_set_global_seed_is_reproducible_and_seed_sensitive():
    set_global_seed(1234)
    first = _draw_all_rngs()
    set_global_seed(1234)
    assert _draw_all_rngs() == first
    set_global_seed(1235)
    assert _draw_all_rngs() != first


def test_set_global_seed_reports_what_was_seeded():
    info = set_global_seed(7, deterministic=True)
    assert info['seed'] == 7
    assert info['python_random'] and info['torch']
    assert info['deterministic_requested'] is True
    assert info['cudnn_deterministic'] is True
    assert info['deterministic_algorithms'] is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()
    json.dumps(info, allow_nan=False)


def test_seed_worker_derives_from_torch_initial_seed():
    torch.manual_seed(321)
    set_global_seed(999)  # scramble the python/numpy RNG state first
    torch.manual_seed(321)  # the torch seed a DataLoader worker would set
    seed_worker(0)
    assert random.random() == random.Random(321).random()
    try:
        import numpy as np
    except ImportError:
        return
    assert (np.random.rand(3).tolist()
            == np.random.RandomState(321).rand(3).tolist())


def test_make_seeded_generator_is_reproducible():
    def sample(seed):
        gen = make_seeded_generator(seed)
        return torch.randn(8, generator=gen).tolist()

    assert sample(11) == sample(11)
    assert sample(11) != sample(12)


def _tiny_cfg() -> HybridConfig:
    return HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8, kda_chunk_size=0,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=5,
        hca_m2=8, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=5,
        n_kda=1, n_csa=1, n_hca=1,
        csa_rope_dim=4, hca_rope_dim=4, kda_backend='reference',
    )


def _fixed_batch(vocab_size=33, B=2, T=24, seed=1):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (B, T), generator=gen)
    return {'input_ids': ids, 'labels': ids.clone()}


def _overfit(steps, lr, seed=5):
    return run_overfit_sanity_check(
        lambda: LMWithHybrid(33, _tiny_cfg()), _fixed_batch(), 'cpu',
        steps=steps, lr=lr, seed=seed, log_every=0)


def test_overfit_check_rejects_too_few_steps():
    with pytest.raises(ValueError):
        _overfit(steps=9, lr=3e-3)


def test_overfit_check_detects_frozen_training():
    # lr=0 freezes the weights: the loss is flat and the gate must FAIL
    # rather than let a broken optimizer (or detached loss) start a long run.
    report = _overfit(steps=15, lr=0.0)
    assert report['passed'] is False
    # Frozen weights => identical token accuracy on the identical batch.
    assert report['accs'][0] == report['accs'][-1]


def test_overfit_check_passes_on_healthy_training():
    report = _overfit(steps=40, lr=3e-3)
    assert report['passed'] is True
    assert report['losses'][-1] < report['losses'][0]
    assert report['final_loss'] < report['loss_ratio_threshold'] * \
        report['initial_loss']
    # The report is the tracked artifact for this gate; it must serialize.
    json.dumps(report, allow_nan=False)


def test_cache_audit_records_seed():
    from cache_accounting import run_cache_audit
    from run_decoding import HybridDecoding

    torch.manual_seed(3)
    decoder = HybridDecoding(cfg=_tiny_cfg(), total_layers=3).eval()
    x = torch.randn(1, 8, decoder.cfg.d_model) * 0.1
    with torch.no_grad():
        decoder.reset()
        decoder(x)
    report = run_cache_audit(decoder, decoder.cfg, T=8, seed=7)
    assert report['seed'] == 7
    json.dumps(report, allow_nan=False)
