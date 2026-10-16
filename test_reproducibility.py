"""Reproducibility / experiment-tracking regression tests.

Covers the shared RNG helpers in ``kaggle_setup``
(:func:`set_global_seed`, :func:`seed_worker`,
:func:`make_seeded_generator`), the pre-training overfit sanity gates in
``train_lm_autodl`` (LM) and ``run_quality`` (classification / MQAR), and
the seed-recording contract of the experiment runners.

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
from run_quality import (  # noqa: E402
    train_one,
    run_overfit_sanity_check as run_mqar_overfit_sanity_check,
)
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


# ---------------------------------------------------------------------------
# CUDA deterministic-algorithm environment (set_global_seed)
# ---------------------------------------------------------------------------

def test_set_global_seed_exports_cublas_workspace_config():
    """deterministic=True must export CUBLAS_WORKSPACE_CONFIG.

    PyTorch's ``use_deterministic_algorithms`` requires the config for
    deterministic cuBLAS GEMMs; without it, GEMMs warn and fall back to a
    nondeterministic algorithm (or raise, depending on strictness).
    """
    saved = os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    try:
        info = set_global_seed(11, deterministic=True)
        assert info['cublas_workspace_config'] == ':4096:8'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        # A second call must not clobber an already-present value.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        info2 = set_global_seed(11, deterministic=True)
        assert info2['cublas_workspace_config'] == ':16:8'
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'
    finally:
        if saved is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = saved


# ---------------------------------------------------------------------------
# Classification (MQAR) overfit sanity gate in run_quality
# ---------------------------------------------------------------------------

def test_mqar_overfit_check_rejects_too_few_steps():
    with pytest.raises(ValueError):
        run_mqar_overfit_sanity_check('kda', steps=9)


def test_mqar_overfit_check_passes_on_healthy_training():
    report = run_mqar_overfit_sanity_check(
        'kda', steps=200, lr=3e-3, seed=2026, train_batch=32)
    assert report['passed'] is True
    # The canonical MQAR config is chosen so every operator memorizes the
    # fixed batch completely; require near-100% train accuracy.
    assert report['final_train_acc'] >= 0.99
    assert report['final_loss'] < report['initial_loss'] * 0.1
    # The report is the tracked artifact for this gate; it must serialize.
    json.dumps(report, allow_nan=False)


def test_mqar_overfit_check_detects_frozen_training():
    # lr=0 freezes the weights: the loss is flat and the gate must FAIL
    # rather than let a broken optimizer (or detached loss) start the sweep.
    report = run_mqar_overfit_sanity_check('kda', steps=15, lr=0.0)
    assert report['passed'] is False
    # Frozen weights => identical token accuracy on the identical batch.
    assert report['acc_curve'][0] == report['acc_curve'][-1]


def test_mqar_overfit_check_is_deterministic():
    kw = dict(steps=12, lr=3e-3, seed=123)
    first = run_mqar_overfit_sanity_check('csa', **kw)
    second = run_mqar_overfit_sanity_check('csa', **kw)
    assert first['fixed_batch_sha256'] == second['fixed_batch_sha256']
    assert first['loss_curve'] == second['loss_curve']
    assert first['acc_curve'] == second['acc_curve']


def test_mqar_overfit_gate_does_not_perturb_training_results():
    # The gate re-seeds internally and train_one re-seeds per op/seed, so a
    # gate run must leave train_one's curves bit-identical. This guards the
    # "gate is diagnostic-only" contract the sweep relies on.
    kwargs = dict(seed=7, steps=15, device='cpu')
    before = train_one('kda', **kwargs)
    run_mqar_overfit_sanity_check('kda', steps=10, seed=999)
    after = train_one('kda', **kwargs)
    assert before['loss_curve'] == after['loss_curve']
    assert before['acc_curve'] == after['acc_curve']
    assert before['final_acc'] == after['final_acc']


def test_lm_overfit_gate_does_not_perturb_model_init():
    # train_lm_autodl re-seeds right before constructing the real model so
    # its init is identical to a run where the gate never ran. Guard that
    # contract: construct the model with the same seeding recipe used by
    # main(), with and without a preceding gate run, and compare weights.
    def construct():
        set_global_seed(42, deterministic=True)
        return LMWithHybrid(33, _tiny_cfg()).state_dict()

    baseline = {k: v.clone() for k, v in construct().items()}
    run_overfit_sanity_check(
        lambda: LMWithHybrid(33, _tiny_cfg()), _fixed_batch(), 'cpu',
        steps=10, lr=3e-3, seed=5, log_every=0)
    after = construct()
    assert baseline.keys() == after.keys()
    assert all(torch.equal(baseline[k], after[k]) for k in baseline)
