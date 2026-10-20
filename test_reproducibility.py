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
    append_run_manifest,
    audit_split,
    grad_flow_report,
    make_run_manifest_entry,
    make_seeded_generator,
    recipe_sha256,
    seed_worker,
    set_global_seed,
    split_train_val_test,
)
from ops_fused import HybridConfig  # noqa: E402
from run_ablation import (  # noqa: E402
    _build_layout_model,
    _run_ablation_overfit_gate,
    eval_layout,
)
from run_ablation import (  # noqa: E402
    run_overfit_sanity_check as run_hybrid_overfit_sanity_check,
)
from run_quality import (  # noqa: E402
    make_mqar_batch,
    train_one,
    run_overfit_sanity_check as run_mqar_overfit_sanity_check,
)
import train_lm_autodl  # noqa: E402
from train_lm_autodl import (  # noqa: E402
    LMWithHybrid,
    SealedTestSet,
    TinyStoriesLM,
    _FallbackCharTokenizer,
    clean_lm_texts,
    evaluate_lm_loss,
    run_overfit_sanity_check,
    split_clean_texts,
)


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


def test_cudnn_autotune_never_overrides_deterministic_settings():
    import kaggle_setup

    # A deterministic request made before device configuration must survive:
    # benchmark=True lets cuDNN pick kernels by timing, which can differ
    # between otherwise identical runs.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert kaggle_setup._enable_cudnn_autotune_if_safe() is False
    assert torch.backends.cudnn.benchmark is False

    # When determinism was not requested, the historical autotune path is
    # still enabled.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    assert kaggle_setup._enable_cudnn_autotune_if_safe() is True
    assert torch.backends.cudnn.benchmark is True


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


# ---------------------------------------------------------------------------
# Needle-sweep (Exp 12) fixed-batch overfit gate
# ---------------------------------------------------------------------------

def _needle_gate(steps, *, lr=3e-3, seed=123, batch=4):
    import run_needle_haystack as nh

    return nh.run_needle_overfit_sanity_check(
        tokenizer=nh._NeedleCharTokenizer(), memory_type='kda',
        context_len=64, needle_pos=8, batch_size=batch, steps=steps,
        lr=lr, seed=seed, device='cpu', log_every=0)


def test_needle_overfit_check_rejects_too_few_steps():
    with pytest.raises(ValueError):
        _needle_gate(steps=9)


def test_needle_overfit_check_detects_frozen_training():
    report = _needle_gate(steps=12, lr=0.0)
    assert report['passed'] is False
    assert report['loss_curve'][0] == report['loss_curve'][-1]
    assert report['exact_match_curve'][0] == \
        report['exact_match_curve'][-1]


def test_needle_overfit_check_is_deterministic_and_reports_grad_flow():
    first = _needle_gate(steps=12, seed=321)
    second = _needle_gate(steps=12, seed=321)
    assert first['fixed_batch_sha256'] == second['fixed_batch_sha256']
    assert first['loss_curve'] == second['loss_curve']
    assert first['exact_match_curve'] == second['exact_match_curve']
    assert first['seed_info']['torch'] is True
    assert first['grad_flow']['n_missing_grad'] == 0
    assert first['weight_decay'] == 0.0
    assert first['data_augmentation'] == 'none'
    json.dumps(first, allow_nan=False)


def test_needle_stepwise_ste_overfit_gate_covers_qat_path():
    import run_needle_haystack as nh

    report = nh.run_needle_overfit_sanity_check(
        tokenizer=nh._NeedleCharTokenizer(), memory_type='kda',
        context_len=64, needle_pos=8, batch_size=4, steps=12,
        lr=3e-3, seed=123, device='cpu', log_every=0,
        stepwise=True, quant_spec=nh.QuantSpec(bits=2, group_size=16),
        ste=True)
    assert report['label'] == 'needle-stepwise-ste'
    assert report['stepwise'] is True and report['ste'] is True
    assert report['quant_spec']['bits'] == 2
    assert report['grad_flow']['n_missing_grad'] == 0
    assert isinstance(report['passed'], bool)
    json.dumps(report, allow_nan=False)


def test_needle_overfit_gate_can_be_disabled(monkeypatch):
    import run_needle_haystack as nh

    monkeypatch.setenv('NEEDLE_OVERFIT_CHECK', '0')
    record, proceed = nh._run_needle_overfit_gate(
        tokenizer=nh._NeedleCharTokenizer(), device='cpu',
        memory_types=['kda'])
    assert proceed is True
    assert record['enabled'] is False
    assert record['reports'] == []


# ---------------------------------------------------------------------------
# Hybrid-sweep (Exp 5 / 8 / 9) classification overfit gate in run_ablation
# ---------------------------------------------------------------------------

def _hybrid_gate(steps, *, lr=3e-3, seed=123, batch=8):
    return run_hybrid_overfit_sanity_check(
        lambda: _build_layout_model((3, 1, 1), 32, 'cpu'),
        lambda b, embed, gen: make_mqar_batch(
            b, 16, 1, 16, embed, 'cpu', generator=gen),
        label='layout KDA:CSA:HCA=3:1:1', steps=steps, lr=lr, seed=seed,
        train_batch=batch, device='cpu')


def test_hybrid_overfit_check_rejects_too_few_steps():
    with pytest.raises(ValueError):
        _hybrid_gate(steps=9)


def test_hybrid_overfit_check_passes_on_healthy_training():
    report = _hybrid_gate(steps=25)
    assert report['passed'] is True
    # The hybrid layout contains only one retrieval layer, so the gate
    # relies on the fixed batch being fully memorized.
    assert report['final_train_acc'] >= 0.99
    # The report is the tracked artifact for this gate; it must serialize.
    json.dumps(report, allow_nan=False)


def test_hybrid_overfit_check_detects_frozen_training():
    # lr=0 freezes the weights: the loss is flat and the gate must FAIL
    # rather than let a broken optimizer (or a detached CSA/HCA path) start
    # the multi-seed sweep.
    report = _hybrid_gate(steps=15, lr=0.0)
    assert report['passed'] is False
    assert report['acc_curve'][0] == report['acc_curve'][-1]


def test_hybrid_overfit_check_is_deterministic():
    first = _hybrid_gate(steps=12, seed=321)
    second = _hybrid_gate(steps=12, seed=321)
    assert first['fixed_batch_sha256'] == second['fixed_batch_sha256']
    assert first['loss_curve'] == second['loss_curve']
    assert first['acc_curve'] == second['acc_curve']


def test_hybrid_overfit_gate_does_not_perturb_layout_results():
    # The gate re-seeds internally and eval_layout re-seeds per cell, so a
    # gate run must leave the sweep's curves bit-identical. This guards the
    # "gate is diagnostic-only" contract the runners rely on.
    kwargs = dict(seed=7, steps=5, device='cpu')
    before = eval_layout((3, 1, 1), **kwargs)
    _hybrid_gate(steps=10)
    after = eval_layout((3, 1, 1), **kwargs)
    assert before['loss_curve'] == after['loss_curve']
    assert before['final_acc'] == after['final_acc']


def test_hybrid_overfit_gate_can_be_disabled(monkeypatch):
    def _unexpected_build():
        raise AssertionError(
            'gate builder must not run when the gate is disabled')

    monkeypatch.setenv('TESTABL_OVERFIT_CHECK', '0')
    record, proceed = _run_ablation_overfit_gate(
        [('never-run', _unexpected_build)], 'cpu', env_prefix='TESTABL')
    assert proceed is True
    assert record['enabled'] is False
    assert record['reports'] == []


# ---------------------------------------------------------------------------
# Leakage-safe train/val/test splitting and data cleaning
# ---------------------------------------------------------------------------

def test_split_train_val_test_is_disjoint_complete_and_deterministic():
    split = split_train_val_test(100, val_frac=0.2, test_frac=0.1, seed=3)
    assert {k: len(v) for k, v in split.items()} == {
        'train': 70, 'val': 20, 'test': 10}
    report = audit_split(split, 100)
    assert report['ok'], report['problems']
    assert report['pairwise_disjoint'] and report['complete']
    assert report['non_empty']
    # Deterministic for the same seed, different for another seed.
    again = split_train_val_test(100, val_frac=0.2, test_frac=0.1, seed=3)
    assert split == again
    other = split_train_val_test(100, val_frac=0.2, test_frac=0.1, seed=4)
    assert other != split
    # Drawing a split must not touch the global torch RNG stream.
    state = torch.get_rng_state()
    try:
        split_train_val_test(50, seed=11)
        assert torch.equal(state, torch.get_rng_state())
    finally:
        torch.set_rng_state(state)


def test_split_chronological_never_shuffles():
    split = split_train_val_test(100, val_frac=0.2, test_frac=0.1, seed=0,
                                 chronological=True)
    # Past -> train, next block -> val, most recent block -> test.
    assert split['train'] == list(range(70))
    assert split['val'] == list(range(70, 90))
    assert split['test'] == list(range(90, 100))
    # Changing the seed must not reorder a chronological split.
    assert split_train_val_test(100, val_frac=0.2, test_frac=0.1, seed=99,
                                chronological=True) == split


def test_split_keeps_groups_whole():
    # 12 groups of varying sizes, interleaved in the row order.
    group_sizes = [1, 5, 2, 8, 3, 1, 4, 2, 6, 1, 3, 9]
    group_ids = []
    for g, size in enumerate(group_sizes):
        group_ids.extend([f'user{g}'] * size)
    n = len(group_ids)
    split = split_train_val_test(n, val_frac=0.2, test_frac=0.2, seed=5,
                                 group_ids=group_ids)
    report = audit_split(split, n, group_ids=group_ids)
    assert report['ok'], report['problems']
    assert report['group_integrity']
    # Explicit re-check: every group lives entirely on one side.
    for g in set(group_ids):
        members = {i for i, gg in enumerate(group_ids) if gg == g}
        assert all(members <= set(split[k]) for k in ('train', 'val', 'test')
                   if members & set(split[k]))


def test_audit_split_detects_overlap_and_missing_rows():
    bad = {'train': [0, 1, 2, 2], 'val': [2, 3], 'test': [4]}
    report = audit_split(bad, 6)
    assert not report['ok']
    assert not report['pairwise_disjoint']
    assert any('overlap' in p for p in report['problems'])
    assert any('duplicate' in p for p in report['problems'])
    assert any('not assigned' in p for p in report['problems'])
    # A group straddling two splits is flagged too.
    straddling = {'train': [0, 1], 'val': [2], 'test': [3]}
    report = audit_split(straddling, 4, group_ids=['a', 'a', 'b', 'b'])
    assert not report['ok'] and not report['group_integrity']


def test_clean_lm_texts_drops_empty_duplicates_and_unlearnable():
    tok = _FallbackCharTokenizer()
    texts = ['hello world', 'hello world', '   ', '', 'x', 'abc def']
    result = clean_lm_texts(texts, tok, seq_len=16)
    assert result['texts'] == ['hello world', 'abc def']
    stats = result['stats']
    assert stats['n_raw'] == 6
    assert stats['n_kept'] == 2
    assert stats['n_empty'] == 2
    assert stats['n_duplicates'] == 1
    assert stats['n_too_short_to_supervise'] == 1
    json.dumps(stats, allow_nan=False)


def test_split_clean_texts_has_no_cross_split_overlap():
    tok = _FallbackCharTokenizer()
    texts = [f'unique story number {i}' for i in range(200)]
    # Inject exact duplicates in the raw list; cleaning must remove them
    # BEFORE the split, otherwise the same text could land on two sides.
    texts.extend(texts[:50])
    cleaned = clean_lm_texts(texts, tok, seq_len=32)['texts']
    assert len(cleaned) == 200
    split, report = split_clean_texts(cleaned, val_frac=0.2, test_frac=0.2,
                                      seed=1)
    assert report['audit']['ok'], report['audit']['problems']
    assert report['exact_text_overlaps'] == {
        'train_val': 0, 'train_test': 0, 'val_test': 0}
    # The three text partitions are disjoint by construction.
    by_split = {k: {cleaned[i] for i in split[k]} for k in split}
    assert not (by_split['train'] & by_split['val'])
    assert not (by_split['train'] & by_split['test'])
    assert not (by_split['val'] & by_split['test'])


def test_clean_lm_texts_drops_null_and_nonfinite_documents():
    tok = _FallbackCharTokenizer()
    texts = [
        None, float('nan'), float('inf'), float('-inf'),  # label noise
        '', '   ',                                        # empty
        12345,                                            # legitimate number
        'a real story about a cat',
    ]
    result = clean_lm_texts(texts, tok, seq_len=32)
    stats = result['stats']
    assert stats['n_nonfinite'] == 4
    assert stats['n_empty'] == 2
    assert result['texts'] == ['12345', 'a real story about a cat']
    # A finite number is text, not a null value.
    assert '12345' in result['texts']
    json.dumps(stats, allow_nan=False)


def test_clean_lm_texts_counts_and_optionally_drops_window_duplicates():
    """Distinct strings that truncate to the same model input are duplicates.

    ``TinyStoriesLM`` only ever sees the first ``seq_len + 1`` tokens, so
    two long documents sharing that prefix are the SAME training sample;
    keeping both would let one split's sample be memorized from the other.
    The drop is opt-in (``dedup_windows``) because with a short ``seq_len``
    many genuinely distinct documents share a prefix; the count is always
    reported as a leak diagnostic.
    """
    tok = _FallbackCharTokenizer()
    seq_len = 8
    a = 'x' * 20 + ' tail one'
    b = 'x' * 20 + ' tail two'   # same first 9 tokens as ``a``
    # Default: reported, but the reference corpus composition is kept.
    diag = clean_lm_texts([a, b], tok, seq_len=seq_len)
    assert diag['texts'] == [a, b]
    assert diag['stats']['n_window_duplicates'] == 1
    assert diag['stats']['dedup_windows'] is False
    # Opt-in: the shared training sample is removed before the split.
    result = clean_lm_texts([a, b], tok, seq_len=seq_len, dedup_windows=True)
    assert result['texts'] == [a]
    stats = result['stats']
    assert stats['n_duplicates'] == 0
    assert stats['n_window_duplicates'] == 1
    assert stats['dedup_windows'] is True
    # Exact duplicates are still reported under ``n_duplicates``.
    exact = clean_lm_texts([a, a], tok, seq_len=seq_len, dedup_windows=True)
    assert exact['stats']['n_duplicates'] == 1
    assert exact['stats']['n_window_duplicates'] == 0


def test_split_clean_texts_chronological_is_ordered_and_group_aware():
    tok = _FallbackCharTokenizer()
    texts = [f'unique story number {i}' for i in range(100)]
    split, report = split_clean_texts(texts, val_frac=0.2, test_frac=0.1,
                                      seed=7, chronological=True)
    # Past -> train, next block -> val, most recent block -> test.
    assert split['train'] == list(range(70))
    assert split['val'] == list(range(70, 90))
    assert split['test'] == list(range(90, 100))
    assert report['chronological'] is True
    assert report['chronological_order_verified'] is True
    assert report['group_aware'] is False
    assert report['audit']['ok'], report['audit']['problems']
    # The seed cannot reorder a chronological split.
    again, _ = split_clean_texts(texts, val_frac=0.2, test_frac=0.1, seed=99,
                                 chronological=True)
    assert again == split


def test_split_clean_texts_groups_stay_on_one_side():
    tok = _FallbackCharTokenizer()
    # 20 users, 4 interleaved documents each (group != row order).
    group_ids = [f'user{i % 20}' for i in range(80)]
    texts = [f'story {i} by {g}' for i, g in enumerate(group_ids)]
    split, report = split_clean_texts(texts, val_frac=0.25, test_frac=0.25,
                                      seed=3, group_ids=group_ids)
    assert report['group_aware'] is True
    assert report['n_groups'] == 20
    assert report['audit']['ok'], report['audit']['problems']
    assert report['audit']['group_integrity'] is True
    side = {}
    for name in ('train', 'val', 'test'):
        for i in split[name]:
            g = group_ids[i]
            assert g not in side or side[g] == name
            side[g] = name
    assert len(side) == 20
    # A mismatched group list is rejected before any split is drawn.
    with pytest.raises(ValueError):
        split_clean_texts(texts, group_ids=group_ids[:-1])


def test_needle_train_eval_seed_streams_are_reserved_and_disjoint():
    from run_needle_haystack import (
        EVAL_SEED_RESERVE,
        SEED_CEILING,
        audit_train_eval_seed_disjointness,
        draw_train_seeds,
        eval_batch_seeds,
    )
    # The held-out evaluation band keeps its historical 0..N values ...
    assert eval_batch_seeds(0, 4) == [0, 1, 2, 3]
    assert eval_batch_seeds(2, 4) == [8, 9, 10, 11]
    # ... and cannot silently overflow into the training band.
    with pytest.raises(ValueError):
        eval_batch_seeds(EVAL_SEED_RESERVE, 1)
    # Training seeds are drawn strictly outside the evaluation band.
    gen = torch.Generator().manual_seed(0)
    train_seeds = draw_train_seeds(gen, 512)
    assert train_seeds and all(EVAL_SEED_RESERVE <= s < SEED_CEILING
                               for s in train_seeds)
    # Same generator state -> same draw; next step -> fresh data.
    gen2 = torch.Generator().manual_seed(0)
    assert draw_train_seeds(gen2, 512) == train_seeds
    assert draw_train_seeds(gen2, 512) != train_seeds
    eval_seeds = eval_batch_seeds(0, 8) + eval_batch_seeds(1, 8)
    audit = audit_train_eval_seed_disjointness(train_seeds, eval_seeds)
    assert audit['disjoint'] is True
    assert audit['overlap_count'] == 0
    assert audit['eval_seed_band'] == [0, EVAL_SEED_RESERVE]
    # The audit actually detects a leak instead of rubber-stamping.
    leaked = audit_train_eval_seed_disjointness([5, 100], [5, 6])
    assert leaked['disjoint'] is False
    assert leaked['overlap_examples'] == [5]


def test_sealed_test_set_opens_exactly_once():
    calls = []

    def factory():
        calls.append(1)
        return 'test-dataset'

    sealed = SealedTestSet(factory)
    assert sealed.opened is False
    assert sealed.open() == 'test-dataset'
    assert sealed.opened is True
    with pytest.raises(RuntimeError):
        sealed.open()
    assert calls == [1]


def test_lm_dataset_masks_padding_labels():
    tok = _FallbackCharTokenizer()
    ds = TinyStoriesLM(tok, seq_len=16, texts=['abc'], name='train')
    item = ds[0]
    assert item['input_ids'].shape == (16,)
    assert item['labels'].shape == (16,)
    # 'abc' -> 3 real tokens, so exactly 2 supervised next-token targets;
    # everything after (padding) is masked out of the loss.
    supervised = (item['labels'] != -100)
    assert int(supervised.sum()) == 2
    assert bool(supervised[:2].all()) and bool((~supervised[2:]).all())
    # Labels never contain an unsupervised input token id by accident.
    assert int(item['labels'][-1]) == -100


def _tiny_lm_dataset(tokenizer, seq_len=8, n=8):
    texts = [f'hello world token {i}' for i in range(n)]
    return TinyStoriesLM(tokenizer, seq_len=seq_len, texts=texts, name='test')


def test_evaluate_lm_loss_is_read_only_rng_neutral_and_deterministic():
    torch.manual_seed(0)
    tok = _FallbackCharTokenizer()
    model = LMWithHybrid(len(tok), _tiny_cfg())
    dataset = _tiny_lm_dataset(tok)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng_state = torch.get_rng_state()
    try:
        first = evaluate_lm_loss(model, dataset, batch_size=4, device='cpu',
                                 max_batches=2)
        assert torch.equal(rng_state, torch.get_rng_state())
    finally:
        torch.set_rng_state(rng_state)
    # Read-only: no optimizer step / no weight update can happen here.
    assert all(torch.equal(before[k], model.state_dict()[k])
               for k in before)
    assert first['shuffle'] is False and first['grad_enabled'] is False
    assert first['n_tokens'] > 0
    assert first['loss'] is not None and first['loss'] == first['loss']
    assert first['perplexity'] is not None and first['perplexity'] > 0
    # Deterministic across calls (fixed order, fixed generator).
    second = evaluate_lm_loss(model, dataset, batch_size=4, device='cpu',
                              max_batches=2)
    assert first == second
    # The model's train/eval flag is restored.
    assert model.training is True


def test_main_requires_nonzero_val_and_test_fractions(monkeypatch):
    for flag in ('--val_frac', '--test_frac'):
        monkeypatch.setattr(sys, 'argv',
                            ['train_lm_autodl.py', flag, '0'])
        with pytest.raises(SystemExit):
            train_lm_autodl.main()


def test_main_early_stops_on_validation_and_tests_once(monkeypatch, tmp_path):
    # Full training entry point on the offline path, with the validation
    # loss forced constant: after the (unbeatable) step-0 check, patience
    # must stop training, the step-0 checkpoint must be kept, and the
    # sealed test split must still be evaluated exactly once on it.
    import json as _json

    def fake_eval(model, dataset, **kwargs):
        return {'loss': 5.0, 'perplexity': 148.4, 'n_batches': 1,
                'n_sequences': 2, 'n_tokens': 30, 'max_batches': 1,
                'batch_size': 2, 'shuffle': False, 'grad_enabled': False}

    monkeypatch.setattr(train_lm_autodl, 'evaluate_lm_loss', fake_eval)
    monkeypatch.setattr(
        train_lm_autodl, '_load_raw_texts',
        lambda max_samples, split='train': (
            [f'unique story number {i} for training' for i in range(60)],
            'synthetic:test'))
    monkeypatch.setattr(sys, 'argv', [
        'train_lm_autodl.py', '--kaggle', '--max_steps', '10',
        '--seq_len', '8', '--batch_size', '2', '--val_every', '1',
        '--eval_max_batches', '1', '--early_stop_patience', '2',
        '--skip_overfit_check', '--skip_consistency_check',
        '--output_dir', str(tmp_path), '--seed', '3',
        '--manifest', str(tmp_path / 'run_manifest.jsonl'),
    ])
    train_lm_autodl.main()

    record = _json.loads((tmp_path / 'run_record.json').read_text())
    assert record['final']['stopped_early'] is True
    assert record['final']['completed'] is False
    assert record['final']['optimizer_step'] == 2
    assert record['validation']['early_stopping']['stopped_early'] is True
    assert record['validation']['best']['step'] == 0
    assert record['test']['evaluated_once'] is True
    assert record['test']['selected_step'] == 0
    assert record['test']['report']['loss'] == 5.0
    # The split itself was audited and the cleaning step ran before it.
    assert record['data']['split_audit']['audit']['ok'] is True
    assert record['data']['cleaning']['n_kept'] == 60
    # The split protocol records the two new structural-data regimes.
    assert record['data']['split_protocol']['chronological'] is False
    assert record['data']['split_protocol']['group_ids'] is None
    assert record['data']['split_audit']['chronological'] is False
    assert 'n_window_duplicates' in record['data']['cleaning']
    assert record['data']['splits']['test']['sealed_until_final_evaluation'] \
        is False
    # The run also landed in the manifest ledger with seed / status / curve
    # digest, linking the record to its artifacts.
    lines = (tmp_path / 'run_manifest.jsonl').read_text().splitlines()
    assert len(lines) == 1
    entry = _json.loads(lines[0])
    assert entry['experiment'] == 'train_lm_autodl'
    assert entry['status'] == 'completed'
    assert entry['seeds'] == 3
    assert entry['results']['stopped_early'] is True
    assert entry['results']['selected_step'] == 0
    assert entry['results']['test_loss'] == 5.0
    assert len(entry['results']['training_curve_sha256']) == 64
    assert entry['recipe_sha256'] == recipe_sha256(entry['hyperparameters'])


def test_main_chronological_split_stays_past_to_future(monkeypatch,
                                                       tmp_path):
    # Same offline entry point, but with --chronological_split: the split
    # must not shuffle, must be recorded as ordered, and the sealed test
    # evaluation still happens exactly once.
    import json as _json

    def fake_eval(model, dataset, **kwargs):
        return {'loss': 5.0, 'perplexity': 148.4, 'n_batches': 1,
                'n_sequences': 2, 'n_tokens': 30, 'max_batches': 1,
                'batch_size': 2, 'shuffle': False, 'grad_enabled': False}

    monkeypatch.setattr(train_lm_autodl, 'evaluate_lm_loss', fake_eval)
    monkeypatch.setattr(
        train_lm_autodl, '_load_raw_texts',
        lambda max_samples, split='train': (
            [f'unique chronological story {i}' for i in range(60)],
            'synthetic:test'))
    monkeypatch.setattr(sys, 'argv', [
        'train_lm_autodl.py', '--kaggle', '--max_steps', '10',
        '--seq_len', '8', '--batch_size', '2', '--val_every', '1',
        '--eval_max_batches', '1', '--early_stop_patience', '2',
        '--skip_overfit_check', '--skip_consistency_check',
        '--chronological_split',
        '--output_dir', str(tmp_path), '--seed', '3',
        '--manifest', str(tmp_path / 'run_manifest.jsonl'),
    ])
    train_lm_autodl.main()

    record = _json.loads((tmp_path / 'run_record.json').read_text())
    protocol = record['data']['split_protocol']
    assert protocol['chronological'] is True
    audit = record['data']['split_audit']
    assert audit['chronological'] is True
    assert audit['chronological_order_verified'] is True
    assert audit['audit']['ok'], audit['audit']['problems']
    assert record['test']['evaluated_once'] is True


# ---------------------------------------------------------------------------
# Run manifest (experiment ledger) and gradient-flow audit
# ---------------------------------------------------------------------------

def test_recipe_sha256_ignores_key_order_and_nan():
    a = {'lr': 0.1, 'batch': 32, 'drop': float('nan')}
    b = {'drop': float('nan'), 'batch': 32, 'lr': 0.1}
    assert recipe_sha256(a) == recipe_sha256(b)
    assert recipe_sha256(a) != recipe_sha256({'lr': 0.1, 'batch': 33})
    assert len(recipe_sha256({})) == 64


def test_manifest_entry_requires_seeds_and_carries_identity():
    with pytest.raises(ValueError):
        make_run_manifest_entry(experiment='x', seeds=None)
    entry = make_run_manifest_entry(
        experiment='unit', seeds=[1, 2], hyperparameters={'lr': 0.1})
    assert entry['experiment'] == 'unit'
    assert entry['seeds'] == [1, 2]
    assert entry['manifest_version'] == 1
    assert entry['recipe_sha256'] == recipe_sha256({'lr': 0.1})
    # Strict JSON: no NaN/Inf leaks into the ledger.
    json.dumps(entry, allow_nan=False)


def test_manifest_append_is_strict_jsonl_and_env_resolved(monkeypatch,
                                                          tmp_path):
    path = tmp_path / 'nested' / 'run_manifest.jsonl'
    e1 = make_run_manifest_entry(experiment='a', seeds=1,
                                 hyperparameters={'lr': 0.1,
                                                  'bad': float('inf')})
    e2 = make_run_manifest_entry(experiment='b', seeds=2)
    assert append_run_manifest(e1, str(path)) == str(path)
    assert append_run_manifest(e2, str(path)) == str(path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    entries = [json.loads(line) for line in lines]
    assert [e['experiment'] for e in entries] == ['a', 'b']
    # Non-finite hyperparameters are sanitized to null, not written raw.
    assert entries[0]['hyperparameters']['bad'] is None
    # Env fallback is used when no explicit path is given.
    env_path = tmp_path / 'env_manifest.jsonl'
    monkeypatch.setenv('RUN_MANIFEST_PATH', str(env_path))
    append_run_manifest(e1)
    assert json.loads(
        env_path.read_text().splitlines()[0])['experiment'] == 'a'


def test_grad_flow_report_distinguishes_missing_and_zero_grads():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    names = list(model.named_parameters())

    before = grad_flow_report(names)
    assert before['n_trainable_tensors'] == 4
    assert before['n_missing_grad'] == 4
    assert before['n_zero_grad'] == 0
    assert before['global_grad_norm'] == 0.0
    assert len(before['missing_grad_names']) == 4

    loss = model(torch.arange(12).reshape(3, 4).float()).sum()
    loss.backward()
    after = grad_flow_report(names)
    assert after['n_missing_grad'] == 0
    assert after['n_zero_grad'] == 0
    assert after['global_grad_norm'] > 0.0
    json.dumps(after, allow_nan=False)

    # A detached branch keeps its parameters grad-less while the rest flow.
    for p in model.parameters():
        p.grad = None
    model[2].weight.grad = torch.zeros_like(model[2].weight)
    mixed = grad_flow_report(names)
    assert mixed['n_missing_grad'] == 3
    assert mixed['n_zero_grad'] == 1
    assert mixed['zero_grad_names'] == ['2.weight']
    assert mixed['names_truncated_at'] == 10


def test_ablation_manifest_entry_resolves_env_path(monkeypatch, tmp_path):
    # The ablation runner must land in the same shared ledger with the mode
    # / recipe / seed information, so run_quality + run_ablation + LM runs
    # form one campaign index.
    import run_ablation

    manifest = tmp_path / 'run_manifest.jsonl'
    monkeypatch.setenv('RUN_MANIFEST_PATH', str(manifest))
    run_ablation._append_ablation_manifest('ratio', 0, 3, 5, [1, 2])
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry['experiment'] == 'exp5_ratio_ablation'
    assert entry['seeds'] == [42, 43, 44]
    assert entry['status'] == 'completed'
    assert entry['results']['mode'] == 'ratio'
    assert entry['results']['n_kv'] == [1, 2]
    assert entry['results']['return_code'] == 0
    assert entry['artifacts']['results'] == 'results/exp5_ablation.json'


def test_train_lm_main_is_deterministic_across_runs(monkeypatch, tmp_path):
    # End-to-end determinism contract: two full main() runs with the same
    # seed must produce bit-identical training curves, checkpoints, and
    # manifest curve digests (catches unseeded ops / RNG-order leaks).
    def fake_eval(model, dataset, **kwargs):
        return {'loss': 5.0, 'perplexity': 148.4, 'n_batches': 1,
                'n_sequences': 2, 'n_tokens': 30, 'max_batches': 1,
                'batch_size': 2, 'shuffle': False, 'grad_enabled': False}

    monkeypatch.setattr(train_lm_autodl, 'evaluate_lm_loss', fake_eval)
    monkeypatch.setattr(
        train_lm_autodl, '_load_raw_texts',
        lambda max_samples, split='train': (
            [f'unique story number {i} for training' for i in range(60)],
            'synthetic:test'))

    def run_once(tag):
        out = tmp_path / tag
        manifest = out / 'run_manifest.jsonl'
        monkeypatch.setattr(sys, 'argv', [
            'train_lm_autodl.py', '--kaggle', '--max_steps', '4',
            '--seq_len', '8', '--batch_size', '2', '--val_every', '2',
            '--eval_max_batches', '1', '--early_stop_patience', '0',
            '--ppl_bootstrap_nboot', '0',
            '--skip_overfit_check', '--skip_consistency_check',
            '--output_dir', str(out), '--seed', '11',
            '--manifest', str(manifest),
        ])
        train_lm_autodl.main()
        record = json.loads((out / 'run_record.json').read_text())
        entry = json.loads(manifest.read_text().splitlines()[-1])
        return record, entry

    rec_a, man_a = run_once('run_a')
    rec_b, man_b = run_once('run_b')

    def _deterministic_bits(rec):
        # Wall-clock fields (elapsed_s / timing blocks) and run-local
        # artifact paths differ by construction; everything else must match
        # bit-for-bit.
        def strip(obj):
            if isinstance(obj, dict):
                return {k: strip(v) for k, v in obj.items()
                        if k not in ('timing', 'elapsed_s', 'tokens_per_second',
                                     'sequences_per_second', 'ms_per_sequence')}
            if isinstance(obj, list):
                return [strip(v) for v in obj]
            return obj

        return strip({
            'optimizer_step': rec['training_curve']['optimizer_step'],
            'train_loss': rec['training_curve']['loss'],
            'val_history': rec['validation']['history'],
            'best': {k: v for k, v in rec['validation']['best'].items()
                     if k != 'checkpoint'},
            'last_loss': rec['final']['last_loss'],
            'test_report': rec['test']['report'],
        })

    assert _deterministic_bits(rec_a) == _deterministic_bits(rec_b)
    assert (man_a['results']['training_curve_sha256']
            == man_b['results']['training_curve_sha256'])
