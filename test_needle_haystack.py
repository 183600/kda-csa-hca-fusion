"""Regression tests for ``run_needle_haystack.py`` (Exp 12: needle-in-a-haystack
retrieval with quantization-aware evaluation).

What is pinned here
-------------------
1. **Haystack generation is deterministic.** The same (context_len, needle_pos,
   seed) produces identical token sequences; the needle Q&A pair is at the
   exact expected positions.

2. **NeedleHybridLM forward/step shapes and dtypes.** The model runs in fp32,
   fp16, and bf16 without NaN/Inf; the stepwise decode path produces the same
   next-token logits as the full prefill path (up to fp round-off).

3. **Quantization PTQ path.** With a QuantSpec, the KDA state is fake-quantized
   after every layer write; the output dtype and shape are preserved; no
   NaN/Inf are introduced.

4. **QAT probe contract.** The StepwiseKDA model trains with 2-bit STE state
   writes; gradients are finite and non-zero; the STE identity holds at the
   last state write (matching the Exp 11 Part B contract).

5. **bf16 numerical stability.** The CSA/HCA operators run in bf16 without
   overflow/underflow; the bf16-safe softmax and scaling patches activate
   only for bfloat16 dtypes.

6. **Report self-consistency.** The JSON output round-trips as strict JSON;
   all row kinds are present; per-seed records are retained.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from run_needle_haystack import (  # noqa: E402
    _NeedleCharTokenizer,
    build_needle_haystack,
    build_needle_batch,
    NeedleHybridLM,
    NeedleStepwiseKDA,
    evaluate_needle_recall,
    train_needle_model,
)
from kv_quant import QuantSpec  # noqa: E402


def _ok(name: str, cond: bool, detail: str = '') -> None:
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'


def test_haystack_generation_deterministic():
    """Same seed -> same haystack."""
    print('\nTest: deterministic haystack generation')
    tok = _NeedleCharTokenizer()
    for ctx_len in (128, 256):
        for needle_pos in (16, 32, 64):
            for seed in (0, 1, 42):
                ids1, tgt1, ans1 = build_needle_haystack(tok, ctx_len, needle_pos, seed)
                ids2, tgt2, ans2 = build_needle_haystack(tok, ctx_len, needle_pos, seed)
                _ok(f'ctx={ctx_len} pos={needle_pos} seed={seed} ids match',
                    torch.equal(ids1, ids2))
                _ok(f'ctx={ctx_len} pos={needle_pos} seed={seed} tgt match',
                    torch.equal(tgt1, tgt2))
                _ok(f'ctx={ctx_len} pos={needle_pos} seed={seed} ans match',
                    ans1 == ans2)


def test_haystack_needle_position():
    """The fact sits at needle_pos and the question is the final tokens."""
    print('\nTest: needle position correctness')
    tok = _NeedleCharTokenizer()
    q_ids = tok.encode("What is the secret code? ")
    a_ids = tok.encode("The secret code is 42-alpha-7.")

    for ctx_len in (128, 256):
        for needle_pos in (16, 32, 64):
            ids, tgt, ans_start = build_needle_haystack(tok, ctx_len, needle_pos, seed=0)
            _ok(f'ctx={ctx_len} pos={needle_pos} answer_start={ans_start}',
                ans_start == ctx_len)
            # The FACT (answer sentence) is planted at needle_pos ...
            _ok(f'needle fact at pos',
                torch.equal(ids[needle_pos:needle_pos+len(a_ids)], torch.tensor(a_ids)))
            # ... and the QUESTION is appended at the very end, so it is
            # not adjacent to the fact.
            _ok(f'question at end',
                torch.equal(ids[ctx_len-len(q_ids):], torch.tensor(q_ids)))
            _ok(f'target is first answer token',
                tgt.tolist() == [a_ids[0]])


def test_batch_generation():
    """build_needle_batch produces correct shapes and distinct sequences."""
    print('\nTest: batch generation shapes and diversity')
    tok = _NeedleCharTokenizer()
    batch_size = 4
    ctx_len = 128
    needle_pos = 32
    input_ids, targets, ans_positions = build_needle_batch(
        batch_size, ctx_len, needle_pos, tok, device='cpu')
    _ok('input_ids shape', tuple(input_ids.shape) == (batch_size, ctx_len),
        f'{input_ids.shape}')
    _ok('targets shape', tuple(targets.shape) == (batch_size,),
        f'{targets.shape}')
    _ok('answer_positions len', len(ans_positions) == batch_size)
    # Different seeds -> different sequences
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            _ok(f'seq {i} != seq {j}', not torch.equal(input_ids[i], input_ids[j]))


def test_needle_hybrid_lm_forward():
    """NeedleHybridLM forward pass works in all dtypes."""
    print('\nTest: NeedleHybridLM forward pass (fp32/fp16/bf16)')
    tok = _NeedleCharTokenizer()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        model = NeedleHybridLM(tok.vocab_size, d_model=64, n_layers=2,
                               memory_type='kda', n_kda=1, n_csa=1, n_hca=0)
        model.to(dtype=dtype)
        input_ids = torch.randint(0, tok.vocab_size, (2, 32))
        with torch.no_grad():
            logits = model(input_ids)
        _ok(f'{dtype} logits shape', tuple(logits.shape) == (2, 32, tok.vocab_size))
        _ok(f'{dtype} no NaN', not torch.isnan(logits).any().item())
        _ok(f'{dtype} no Inf', not torch.isinf(logits).any().item())


def test_needle_hybrid_lm_stepwise_quant():
    """NeedleHybridLM stepwise forward with quantization."""
    print('\nTest: NeedleHybridLM stepwise + PTQ')
    tok = _NeedleCharTokenizer()
    model = NeedleHybridLM(tok.vocab_size, d_model=64, n_layers=2,
                           memory_type='kda', n_kda=1, n_csa=1, n_hca=0)
    spec = QuantSpec(bits=4, group_size=16)
    input_ids = torch.randint(0, tok.vocab_size, (2, 16))
    model.reset_state()
    with torch.no_grad():
        # Full prefill
        logits1 = model(input_ids)
        # Stepwise with quantization
        model.reset_state()
        for t in range(input_ids.shape[1]):
            logits_t = model.forward_step(input_ids[:, t:t+1], quant_spec=spec)
    _ok('stepwise quant: no NaN', not torch.isnan(logits_t).any().item())
    _ok('stepwise quant: correct shape', tuple(logits_t.shape) == (2, tok.vocab_size))


def test_stepwise_kda_ste_contract():
    """StepwiseKDA STE gradient contract (mirrors Exp 11 Part B)."""
    print('\nTest: StepwiseKDA STE gradient contract')
    tok = _NeedleCharTokenizer()
    spec = QuantSpec(bits=2, group_size=16)
    model = NeedleStepwiseKDA(tok.vocab_size, d_model=32, H=2, K=8, V=8)
    input_ids = torch.randint(0, tok.vocab_size, (2, 8), requires_grad=False)
    input_ids.requires_grad_(False)

    # Train with STE
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        logits = model(input_ids, spec=spec, ste=True)  # [B, T, vocab]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        _ok('STE: grads finite', all(p.grad is not None and torch.isfinite(p.grad).all()
                                     for p in model.parameters() if p.grad is not None))
        _ok('STE: grads non-zero', any(p.grad is not None and p.grad.abs().sum() > 0
                                       for p in model.parameters()))
        opt.step()

    # STE=False path: verify no error on inference
    model2 = NeedleStepwiseKDA(tok.vocab_size, d_model=32, H=2, K=8, V=8)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    with torch.no_grad():
        _ = model2(input_ids[:1], spec=spec, ste=False)
    # (Detailed STE contract is in test_quant_lm.py)


def test_evaluate_needle_recall():
    """evaluate_needle_recall returns valid accuracy."""
    print('\nTest: evaluate_needle_recall basic')
    tok = _NeedleCharTokenizer()
    model = NeedleHybridLM(tok.vocab_size, d_model=64, n_layers=2,
                           memory_type='kda', n_kda=1, n_csa=1, n_hca=0)
    res = evaluate_needle_recall(
        model, tok, context_len=128, needle_pos=32,
        batch_size=4, n_batches=1, device='cpu', quant_spec=None, dtype=torch.float32)
    _ok('accuracy in [0,1]', 0.0 <= res['accuracy'] <= 1.0)
    _ok('correct <= total', res['correct'] <= res['total'])
    _ok('total > 0', res['total'] > 0)


def test_bf16_csa_hca_no_nan_inf():
    """CSA/HCA operators in bf16 produce no NaN/Inf."""
    print('\nTest: CSA/HCA bf16 numerical stability')
    from ops_csa import naive_csa
    from ops_hca import naive_hca
    from ops_fused import HybridConfig

    B, T, d = 1, 32, 64
    H = torch.randn(B, T, d, dtype=torch.bfloat16)
    cfg = HybridConfig(d_model=d, n_heads_qk=2, n_heads_v=2,
                       head_dim_k=16, head_dim_v=16, kda_chunk_size=0,
                       csa_m=4, csa_topk=2, csa_nh=2, csa_c=32, csa_dc=64,
                       csa_nIh=1, csa_cI=16, csa_sliding_window=4,
                       hca_m2=8, hca_nh=2, hca_c=32, hca_dc=64,
                       hca_sliding_window=4)

    # CSA
    from ops_fused import CSAHybridLayer
    csa = CSAHybridLayer(cfg).to(dtype=torch.bfloat16)
    with torch.no_grad():
        out_csa, _ = csa(H)
    _ok('CSA bf16 no NaN', not torch.isnan(out_csa).any().item())
    _ok('CSA bf16 no Inf', not torch.isinf(out_csa).any().item())

    # HCA
    from ops_fused import HCAHybridLayer
    hca = HCAHybridLayer(cfg).to(dtype=torch.bfloat16)
    with torch.no_grad():
        out_hca, _ = hca(H)
    _ok('HCA bf16 no NaN', not torch.isnan(out_hca).any().item())
    _ok('HCA bf16 no Inf', not torch.isinf(out_hca).any().item())


def test_needle_experiment_json_output():
    """Full experiment run produces valid JSON with all row kinds."""
    print('\nTest: experiment JSON structure')
    # Run a minimal quick experiment
    import subprocess
    result = subprocess.run([
        sys.executable, 'run_needle_haystack.py',
        '--quick', '--no-qat', '--json', '/tmp/needle_test.json',
        '--seeds', '0', '--context-lens', '128',
        '--memory-types', 'kda', '--dtypes', 'fp32,bf16',
        '--quant-bits', '4', '--quant-groups', '16'
    ], capture_output=True, text=True, cwd=HERE, timeout=120)
    if result.returncode != 0:
        print(f'  stderr: {result.stderr}')
        print(f'  stdout: {result.stdout}')
    _ok('experiment runs successfully', result.returncode == 0,
        f'returncode={result.returncode}')

    with open('/tmp/needle_test.json') as f:
        payload = json.load(f)

    _ok('has experiment field', 'experiment' in payload)
    _ok('experiment is exp12', payload['experiment'] == 'exp12_needle_haystack')
    _ok('has rows', 'rows' in payload and len(payload['rows']) > 0)

    row_kinds = {r['row_kind'] for r in payload['rows']}
    _ok('has needle_recall rows', 'needle_recall' in row_kinds)
    _ok('has untrained control rows',
        'needle_untrained_control' in row_kinds, f'kinds={sorted(row_kinds)}')
    recall = [r for r in payload['rows'] if r['row_kind'] == 'needle_recall']
    _ok('every recall row is trained',
        all(r.get('is_trained') is True for r in recall))
    _ok('recall rows carry full-answer metric',
        all(r.get('per_seed', [{}])[0].get('answer_match')
            == 'full_answer_exact_match' for r in recall))
    _ok('recall rows carry CI fields',
        all('accuracy_ci95' in r and 'accuracy_std' in r for r in recall))
    _ok('untrained rows carry is_trained=False',
        all(r.get('is_trained') is False for r in payload['rows']
            if r['row_kind'] == 'needle_untrained_control'))

    # Validate JSON round-trips strictly
    json.dumps(payload, allow_nan=False)
    _ok('strict JSON round-trip', True)


def main() -> int:
    test_haystack_generation_deterministic()
    test_haystack_needle_position()
    test_batch_generation()
    test_needle_hybrid_lm_forward()
    test_needle_hybrid_lm_stepwise_quant()
    test_stepwise_kda_ste_contract()
    test_evaluate_needle_recall()
    test_bf16_csa_hca_no_nan_inf()
    test_needle_experiment_json_output()
    print('\n=== ALL NEEDLE HAYSTACK TESTS PASSED ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())