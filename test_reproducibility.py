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
    audit_split,
    make_seeded_generator,
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
    assert record['data']['splits']['test']['sealed_until_final_evaluation'] \
        is False
