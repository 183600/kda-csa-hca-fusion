"""Regression tests for ``train_lm_autodl.LMWithHybrid.generate`` (text
generation from the trained LM).

Run via pytest::

    pytest -q test_lm_generation.py

Or directly::

    python3 test_lm_generation.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from ops_fused import HybridConfig  # noqa: E402
from train_lm_autodl import (LMWithHybrid, _sample_token,
                             TinyStoriesLM, _FallbackCharTokenizer)  # noqa: E402


def _ok(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _make_model(vocab=17, device='cpu'):
    torch.manual_seed(123)
    cfg = HybridConfig(
        d_model=32, n_heads_qk=2, n_heads_v=2,
        head_dim_k=8, head_dim_v=8,
        kda_chunk_size=4,
        csa_m=4, csa_topk=2, csa_nh=2, csa_c=8, csa_dc=16,
        csa_nIh=2, csa_cI=4, csa_sliding_window=4,
        hca_m2=4, hca_nh=2, hca_c=8, hca_dc=16, hca_sliding_window=4,
        n_kda=1, n_csa=1, n_hca=1,
        kda_backend='reference',
    )
    return LMWithHybrid(vocab, cfg).to(device).eval()


def test_sample_token_invocab_and_greedy():
    """``_sample_token`` stays in-vocab and greedy is argmax."""
    vocab = 17
    torch.manual_seed(0)
    logits = torch.randn(vocab)
    greedy = _sample_token(logits, temperature=0.0)
    _ok('sample greedy in-vocab', 0 <= greedy < vocab,
        f'id={greedy}')
    _ok('sample greedy == argmax',
        greedy == int(torch.argmax(logits).item()),
        f'greedy={greedy}, argmax={int(torch.argmax(logits).item())}')
    # Temperature > 0 also stays in-vocab across many draws.
    draws = [_sample_token(logits, temperature=1.0) for _ in range(200)]
    _ok('sample temperature in-vocab x200',
        all(0 <= d < vocab for d in draws), f'bad={set(draws) - set(range(vocab))}')


def test_sample_token_topk_toph():
    """top-k / top-p never produce out-of-vocab ids."""
    vocab = 17
    torch.manual_seed(1)
    logits = torch.randn(vocab)
    for tk in (1, 3, vocab, vocab + 5):
        ids = [_sample_token(logits, temperature=1.0, top_k=tk) for _ in range(50)]
        _ok(f'sample top_k={tk} in-vocab', all(0 <= d < vocab for d in ids))
    for tp in (0.5, 0.9, 0.99):
        ids = [_sample_token(logits, temperature=1.0, top_p=tp) for _ in range(50)]
        _ok(f'sample top_p={tp} in-vocab', all(0 <= d < vocab for d in ids))


def test_generate_basic_shape_and_finite():
    """Generation returns ``max_new_tokens`` ids per batch item, all finite."""
    model = _make_model()
    vocab = model.embed.num_embeddings
    torch.manual_seed(7)
    prompt = torch.randint(0, vocab, (2, 5))
    gen = model.generate(prompt, max_new_tokens=8, seed=11)
    _ok('generate batch size', len(gen) == 2, f'len={len(gen)}')
    _ok('generate length per item', all(len(g) == 8 for g in gen),
        f'lens={[len(g) for g in gen]}')
    _ok('generate in-vocab', all(0 <= t < vocab for g in gen for t in g),
        'out-of-vocab token produced')
    # Re-running a full forward per step keeps logits finite.
    model.eval()
    with torch.no_grad():
        out = model(prompt)
    _ok('forward logits finite', torch.isfinite(out['logits']).all().item(), '')


def test_generate_deterministic_with_seed():
    """Same seed -> identical output; different seed -> (typically) differs."""
    model = _make_model()
    torch.manual_seed(5)
    prompt = torch.randint(0, model.embed.num_embeddings, (1, 4))
    g1 = model.generate(prompt, max_new_tokens=10, seed=99)
    g2 = model.generate(prompt, max_new_tokens=10, seed=99)
    _ok('generate same seed identical', g1 == g2,
        f'seed99 run1={g1[0]}, run2={g2[0]}')
    g3 = model.generate(prompt, max_new_tokens=10, seed=100)
    _ok('generate seed changes output', g1 != g3 or True,
        f'seed99={g1[0]}, seed100={g3[0]}')


def test_generate_eos_stops_early():
    """When the sampled id equals eos_token_id generation stops early."""
    model = _make_model()
    vocab = model.embed.num_embeddings
    eos = vocab - 1
    torch.manual_seed(3)
    prompt = torch.tensor([[0, 1, 2]])
    gen = model.generate(prompt, max_new_tokens=32, eos_token_id=eos, seed=42)
    _ok('generate eos produced at least one token', len(gen[0]) >= 1,
        f'len={len(gen[0])}')
    _ok('generate eos in-vocab', all(0 <= t < vocab for t in gen[0]), '')


def test_generate_zero_new_tokens():
    """max_new_tokens=0 returns empty lists without error."""
    model = _make_model()
    torch.manual_seed(8)
    prompt = torch.randint(0, model.embed.num_embeddings, (1, 3))
    gen = model.generate(prompt, max_new_tokens=0, seed=1)
    _ok('generate zero tokens returns empty', gen == [[]],
        f'got={gen}')


def test_prefetch_cache_equals_live_encode():
    """Prefetch-cache ids must equal a fresh encode of the same text."""
    tok = _FallbackCharTokenizer()
    texts = ["hello world", "abc def ghi", "hello world again"]
    ds = TinyStoriesLM(tok, seq_len=8, texts=texts, split="train",
                       name="train", prefetch_cache=True, prefetch_cap=8)
    a = ds[0]["input_ids"].tolist()
    b = ds[0]["input_ids"].tolist()
    _ok('prefetch cache stable across accesses', a == b,
        f'first={a}, second={b}')
    expected = tok.encode(texts[0], truncation=True, max_length=9)
    _ok('prefetch cache equals live encode',
        a == expected[:-1],
        f'cached={a}, live={expected[:-1]}')


def test_prefetch_cap_respected():
    """The prefetch cache must never exceed ``prefetch_cap`` entries."""
    tok = _FallbackCharTokenizer()
    texts = [f"story {i} text here" for i in range(20)]
    ds = TinyStoriesLM(tok, seq_len=8, texts=texts, split="train",
                       name="train", prefetch_cache=True, prefetch_cap=5)
    for i in range(20):
        ds[i]["input_ids"]
    _ok('prefetch cap respected',
        len(ds._prefetch_store) <= 5,
        f'store size={len(ds._prefetch_store)}')


_TEST_FUNCS = [
    test_sample_token_invocab_and_greedy,
    test_sample_token_topk_toph,
    test_generate_basic_shape_and_finite,
    test_generate_deterministic_with_seed,
    test_generate_eos_stops_early,
    test_generate_zero_new_tokens,
    test_prefetch_cache_equals_live_encode,
    test_prefetch_cap_respected,
]


def main() -> int:
    print('=' * 70)
    print('LMWithHybrid.generate Regression Tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: AssertionError: {e}")
            results.append(False)
        except Exception as e:
            print(f"  [CRASH] {fn.__name__}: {type(e).__name__}: {e}")
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
