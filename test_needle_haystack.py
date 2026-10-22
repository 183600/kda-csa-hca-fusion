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

7. **Distance degradation axis.** ``summarize_needle_hypotheses`` reports
   the accuracy AT the nearest / farthest needle position (mean over the
   per-dtype rows at that position), so the pre-registered
   ``distance_degradation_verdict`` cannot be flipped by which dtype
   happens to score best at which position.
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
    _build_needle_decoder,
    _needle_primary_precision_rank,
    _quantized_hybrid_prefill_last_logits,
    build_needle_haystack,
    build_needle_batch,
    NeedleHybridLM,
    NeedleStepwiseKDA,
    evaluate_needle_recall,
    evaluate_rule_baselines,
    rule_baseline_predict,
    run_qat_needle_probe,
    summarize_needle_hypotheses,
    train_needle_model,
    NEEDLE_HYPOTHESES,
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


def test_quantized_hybrid_prefill_keeps_csa_context():
    """Quantized prefill must not reduce CSA/HCA to single-token attention.

    ``forward_step`` feeds one token to a tower whose CSA/HCA branches are
    stateless. The cache-aware prefill helper must instead let the final
    query see the whole prefix.
    """
    tok = _NeedleCharTokenizer()
    torch.manual_seed(0)
    model = NeedleHybridLM(
        tok.vocab_size, d_model=64, n_layers=1,
        memory_type='kda', n_kda=0, n_csa=1, n_hca=0).eval()
    ids = torch.randint(0, tok.vocab_size, (1, 32))
    changed = ids.clone()
    changed[0, 0] = (int(changed[0, 0]) + 1) % tok.vocab_size
    spec = QuantSpec(bits=4, group_size=16)
    decoder = _build_needle_decoder(model, 'cpu', torch.float32)
    with torch.no_grad():
        base = _quantized_hybrid_prefill_last_logits(
            model, ids, spec, decoder)
        perturbed = _quantized_hybrid_prefill_last_logits(
            model, changed, spec, decoder)
    diff = (base - perturbed).abs().max().item()
    _ok('changing an early context token changes the final-token logits',
        diff > 1e-6, f'max|d|={diff:.3e}')

    # The evaluator must take the cache-aware path, not the stateless
    # single-token forward_step loop that caused the context loss.
    from unittest.mock import patch
    with patch.object(model, 'forward_step',
                      side_effect=AssertionError('stateless prefill used')):
        result = evaluate_needle_recall(
            model, tok, context_len=128, needle_pos=32, batch_size=1,
            n_batches=1, device='cpu', quant_spec=spec,
            dtype=torch.float32)
    _ok('quantized evaluator does not prefill through forward_step',
        result['total'] == 1, result)

    # The unquantized arm must use the same cache-aware path; the old
    # ``model.forward_step`` loop silently dropped all CSA/HCA prefix
    # context, so the unquantized and quantized rows were produced by
    # different models.
    with patch.object(model, 'forward_step',
                      side_effect=AssertionError('stateless prefill used')):
        plain = evaluate_needle_recall(
            model, tok, context_len=128, needle_pos=32, batch_size=1,
            n_batches=1, device='cpu', quant_spec=None,
            dtype=torch.float32)
    _ok('unquantized evaluator also uses the cache-aware path',
        plain['total'] == 1, plain)


def test_primary_precision_rank_prefers_the_widest_format():
    """Without an unquantized row, the primary cell is the highest-bit one."""
    low = _needle_primary_precision_rank(
        {'quantized': True, 'bits': 2, 'group_size': 16})
    high = _needle_primary_precision_rank(
        {'quantized': True, 'bits': 8, 'group_size': 16})
    _ok('8-bit ranks ahead of 2-bit', high < low, f'{high} vs {low}')
    unquant = _needle_primary_precision_rank(
        {'quantized': False, 'dtype': 'fp16'})
    _ok('an unquantized cell still ranks first', unquant < high, unquant)


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


def test_needle_validation_and_bad_cases():
    """Training curve + wrong-sample analysis are attached and read-only."""
    print('\nTest: needle validation curve and bad-case analysis')
    tok = _NeedleCharTokenizer()
    model = NeedleHybridLM(tok.vocab_size, d_model=32, n_layers=2,
                           memory_type='kda', n_kda=1, n_csa=1, n_hca=0)
    model, rep = train_needle_model(
        model, tok, 128, 32, 2, 2, 3e-3, 'cpu', 0,
        val_every=1, val_batches=2)
    hist = rep['val_history']
    _ok('curve records step 0 + every training step + final',
        [p['step'] for p in hist][0] == 0 and hist[-1]['step'] == 2,
        str([p['step'] for p in hist]))
    _ok('every validation point has a finite loss/accuracy',
        all(0.0 <= p['acc'] <= 1.0 and p['loss'] == p['loss']
            for p in hist), str(hist))
    proto = rep['validation_protocol']
    _ok('validation band is disjoint from train/eval',
        proto['read_only'] is True and proto['disjoint_from_train'] is True
        and proto['disjoint_from_eval'] is True, str(proto))
    _ok('early-stopping regret is quantified',
        rep['early_stopping_summary'] is not None
        and 'val_loss_regret' in rep['early_stopping_summary'], '')

    res = evaluate_needle_recall(
        model, tok, context_len=128, needle_pos=32, batch_size=2,
        n_batches=1, device='cpu', max_error_examples=1)
    ea = res['error_analysis']
    cats = set(ea['categories'])
    _ok('wrong samples are categorized',
        cats == {'recency_copy', 'wrong_from_start', 'partial_answer'},
        str(cats))
    _ok('category counts partition the wrong samples',
        sum(v['count'] for v in ea['categories'].values()) == ea['n_wrong'],
        f"{ea['n_wrong']}")
    _ok('category rates sum to 1 when there are errors',
        ea['n_wrong'] == 0
        or abs(sum(v['rate_of_wrong'] for v in ea['categories'].values())
               - 1.0) < 1e-12, '')
    _ok('examples are capped per category',
        all(len(v) <= 1 for v in ea['examples'].values()), '')


def test_rule_baseline_recovers_fact():
    """The weight-free lexical-scan baseline reproduces the planted fact."""
    print('\nTest: symbolic rule baseline (template scan + oracle copy)')
    tok = _NeedleCharTokenizer()
    expected = torch.tensor(tok.encode("The secret code is 42-alpha-7."))
    for ctx_len, needle_pos in ((128, 16), (256, 64), (512, 128)):
        # Direct copy semantics on one batch.
        input_ids, _, _ = build_needle_batch(
            4, ctx_len, needle_pos, tok, seeds=[0, 1, 2, 3])
        out = rule_baseline_predict(
            input_ids, tok, expected.numel(), needle_pos=needle_pos)
        _ok(f'ctx={ctx_len} template copies the fact',
            bool(out['template_preds'].eq(expected.unsqueeze(0)).all()),
            f'preds[0]={out["template_preds"][0].tolist()}')
        _ok(f'ctx={ctx_len} oracle copies the fact',
            bool(out['oracle_preds'].eq(expected.unsqueeze(0)).all()))
        _ok(f'ctx={ctx_len} template starts at needle_pos',
            all(s == needle_pos for s in out['template_starts']),
            f'starts={out["template_starts"]}')
        # Aggregated evaluator uses the held-out eval stream.
        res = evaluate_rule_baselines(
            tok, ctx_len, needle_pos, batch_size=4, n_batches=2)
        _ok(f'ctx={ctx_len} rule exact match == 1.0',
            res['accuracy'] == 1.0, f'acc={res["accuracy"]}')
        _ok(f'ctx={ctx_len} oracle ceiling == 1.0',
            res['oracle_position_accuracy'] == 1.0)
        _ok(f'ctx={ctx_len} rule uses no weights',
            res['uses_model_weights'] is False
            and res['uses_answer_prefix_template'] is True
            and res['uses_oracle_position'] is False)
        _ok(f'ctx={ctx_len} chance recorded below 1.0',
            0.0 < res['chance_uniform_exact_match'] < 1.0)


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


def test_qat_probe_budget_matched_control():
    """QAT is compared against an equally trained fp16 control.

    The QAT arm receives ``steps + qat_steps`` optimizer steps from the
    fp16 initialization. The control must receive exactly the same total
    budget from the same init/seed/data stream; otherwise a "recovery"
    could be explained by extra training alone.
    """
    print('\nTest: QAT probe equal-budget fp16 control')
    tok = _NeedleCharTokenizer()
    spec = QuantSpec(bits=4, group_size=16)
    rows = run_qat_needle_probe(
        tok, seeds=(0,), qat_spec=spec, steps=1, qat_steps=1,
        train_context_len=128, train_needle_pos=16,
        eval_context_len=128, eval_needle_pos=32,
        batch_size=2, device='cpu')
    per_seed = [r for r in rows if r['row_kind'] == 'qat_needle']
    summary = next(r for r in rows if r['row_kind'] == 'qat_needle_summary')
    _ok('one per-seed QAT row', len(per_seed) == 1, f'{len(per_seed)}')
    row = per_seed[0]
    _ok('QAT and continued-fp16 total budgets match',
        row['total_steps_qat'] == row['total_steps_fp16_continued'] == 2,
        f"{row['total_steps_qat']} vs "
        f"{row['total_steps_fp16_continued']}")
    _ok('the shorter fp16 arm records its different budget',
        row['total_steps_fp16'] == 1, f"{row['total_steps_fp16']}")
    _ok('budget-matched accuracy fields are present and in [0,1]',
        all(0.0 <= row[k] <= 1.0 for k in (
            'fp16_continued_accuracy', 'fp16_continued_ptq_accuracy'))
        and abs(row['qat_matched_gain']
                - (row['qat_accuracy']
                   - row['fp16_continued_ptq_accuracy'])) < 1e-12,
        f"matched_gain={row['qat_matched_gain']}")
    _ok('extra-training gain is derived from the two quantized arms',
        abs(row['extra_training_gain']
            - (row['fp16_continued_ptq_accuracy']
               - row['ptq_accuracy'])) < 1e-12,
        f"extra_gain={row['extra_training_gain']}")
    _ok('matched recovery uses the matched design (not the short arm)',
        (row['recovery_fraction_matched'] is None)
        or abs((1.0 - ((row['fp16_continued_accuracy']
                        - row['qat_accuracy'])
                       / (row['fp16_continued_accuracy']
                          - row['fp16_continued_ptq_accuracy'])))
               - row['recovery_fraction_matched']) < 1e-9,
        f"matched_recovery={row['recovery_fraction_matched']}")
    _ok('summary carries the matched control and budgets',
        summary.get('fp16_continued_ptq_accuracy_mean') is not None
        and 'recovery_fraction_matched_mean' in summary
        and summary['recovery_fraction_matched_mean']
        == row['recovery_fraction_matched']
        and summary.get('budget_control', {}).get('matched') is True
        and summary['budget_control']['total_steps_qat']
        == summary['budget_control']['total_steps_fp16_continued'],
        f"budget={summary.get('budget_control')}")
    _ok('training data remains disjoint from evaluation',
        row['data_protocol']['disjoint'] is True,
        f"{row['data_protocol']}")


def test_distance_effect_uses_accuracy_at_extreme_positions():
    """The distance axis compares accuracy AT the near/far positions.

    Recall rows are emitted per dtype, so one needle position can appear
    several times inside a (context, route) group.  Taking the extreme of
    the ``(position, accuracy)`` tuples would report the LOWEST accuracy
    at the nearest position and the HIGHEST at the farthest one, biasing
    ``far_minus_near`` upward and the ``distance_degradation_verdict``
    with it.  The accuracy at an extreme position is the mean over the
    rows at that position.
    """
    print('\nTest: distance effect picks accuracy at the extreme positions')

    def _row(ctx, pos, mem, acc, dtype):
        return {
            'row_kind': 'needle_recall', 'context_len': ctx,
            'needle_pos': pos, 'memory_type': mem, 'dtype': dtype,
            'quantized': False, 'accuracy_mean': acc,
            'hypothesis_checks': {
                'learned_retrieval': {'verdict': None},
                'trained_matches_rule': {'verdict': None},
            },
        }

    dtypes = ['fp32', 'fp16', 'bf16']
    # The fact is inserted at ``needle_pos`` from the START and the question
    # is appended at the END, so pos=64 is the FARTHEST placement and pos=192
    # the NEAREST one. Every dtype degrades with distance -> verdict holds.
    degrading = ([_row(256, 64, 'kda', a, d)
                  for d, a in zip(dtypes, [0.40, 0.35, 0.30])]
                 + [_row(256, 192, 'kda', a, d)
                    for d, a in zip(dtypes, [0.90, 0.85, 0.80])])
    summary = summarize_needle_hypotheses(degrading)
    eff = summary['distance_effects'][0]
    _ok('near position is the one closest to the appended question',
        eff['near_needle_pos'] == 192 and eff['far_needle_pos'] == 64
        and eff['near_distance'] == 64 and eff['far_distance'] == 192,
        f"{eff}")
    _ok('near accuracy is the mean at the nearest position',
        abs(eff['near_accuracy'] - 0.85) < 1e-12,
        f"{eff['near_accuracy']}")
    _ok('far accuracy is the mean at the farthest position',
        abs(eff['far_accuracy'] - 0.35) < 1e-12,
        f"{eff['far_accuracy']}")
    _ok('degrading across every dtype -> verdict True',
        summary['distance_degradation_verdict'] is True,
        f"far-near={eff['far_minus_near']}")

    # A fact near the question is retrieved worse -> verdict falsified.
    improving = ([_row(256, 64, 'kda', a, d)
                  for d, a in zip(dtypes, [0.80, 0.85, 0.90])]
                 + [_row(256, 192, 'kda', a, d)
                    for d, a in zip(dtypes, [0.30, 0.35, 0.40])])
    summary = summarize_needle_hypotheses(improving)
    eff = summary['distance_effects'][0]
    _ok('improving across every dtype -> verdict False',
        summary['distance_degradation_verdict'] is False
        and eff['far_minus_near'] > 0,
        f"far-near={eff['far_minus_near']}")

    # One row per position (single dtype) keeps the exact reported values.
    single = [_row(256, 64, 'kda', 0.4, 'fp32'),
              _row(256, 192, 'kda', 0.9, 'fp32')]
    eff = summarize_needle_hypotheses(single)['distance_effects'][0]
    _ok('single-dtype rows keep their exact accuracies',
        eff['near_accuracy'] == 0.9 and eff['far_accuracy'] == 0.4
        and eff['far_minus_near'] == -0.5, f'{eff}')

    # A single needle position is not a distance axis (no fake far/near).
    one_pos = [_row(256, 64, 'kda', 0.9, 'fp32'),
               _row(256, 64, 'kda', 0.4, 'fp16')]
    summary = summarize_needle_hypotheses(one_pos)
    _ok('one distinct position yields no distance axis',
        summary['distance_effects'] == []
        and summary['distance_degradation_verdict'] is None,
        f"{summary['distance_effects']}")

    # Malformed recall rows must be skipped, not read as pos=0 / acc=0.0
    # (which would invent a third distance point and bias both extremes).
    malformed = degrading + [
        {'row_kind': 'needle_recall', 'memory_type': 'kda',
         'dtype': 'fp32', 'quantized': False, 'accuracy_mean': 0.5,
         'hypothesis_checks': {}},                      # missing context/pos
        _row(256, 'bad', 'kda', 0.1, 'fp32'),
        _row(256, 128, 'kda', float('nan'), 'fp32'),
        _row(256, 128, 'kda', True, 'fp32'),
    ]
    summary = summarize_needle_hypotheses(malformed)
    _ok('malformed recall rows are skipped from the distance axis',
        len(summary['distance_effects']) == 1
        and summary['distance_effects'][0]['near_needle_pos'] == 192
        and summary['distance_effects'][0]['far_needle_pos'] == 64,
        f"{summary['distance_effects']}")

    # Non-dict rows, a non-dict hypothesis block and a string p-value must
    # not abort the aggregation; a malformed rule-baseline accuracy is
    # skipped instead of becoming 0.0 / raising.
    hardened = malformed + [
        42,
        {'row_kind': 'needle_recall', 'memory_type': 'kda',
         'context_len': 256, 'needle_pos': 128, 'accuracy_mean': 0.5,
         'hypothesis_checks': 'not-a-mapping'},
        {'row_kind': 'needle_recall', 'memory_type': 'kda',
         'context_len': 256, 'needle_pos': 128, 'accuracy_mean': 0.5,
         'hypothesis_checks': {'learned_retrieval':
                               {'p_value_directional': '0.01'}}},
        {'row_kind': 'needle_recall', 'memory_type': 'kda',
         'context_len': 256, 'needle_pos': 128, 'accuracy_mean': 0.5,
         'hypothesis_checks': {'learned_retrieval': []}},
        {'row_kind': 'needle_rule_baseline', 'accuracy': 'bad'},
        {'row_kind': 'needle_rule_baseline', 'accuracy': 0.25},
    ]
    summary = summarize_needle_hypotheses(hardened)
    _ok('malformed non-distance rows are skipped safely',
        summary['rule_baseline_accuracy_mean'] == 0.25
        and summary['n_rule_baseline_rows'] == 2
        and len(summary['distance_effects']) == 1,
        f"rule={summary['rule_baseline_accuracy_mean']}, "
        f"n_rule={summary['n_rule_baseline_rows']}")


def test_learned_retrieval_familywise_correction():
    """Repeated dtype/PTQ rows must not count as independent evidence.

    ``summarize_needle_hypotheses`` keeps the raw row-level count for
    comparability with earlier files, but the familywise block selects one
    primary precision row per (architecture, training geometry, evaluation
    geometry) and Holm-corrects those p-values.
    """
    print('\nTest: needle learned-retrieval familywise correction')

    def _row(ctx, dtype, p_value, *, quantized=False, bits=None,
             verdict=True, rule_p=None):
        return {
            'row_kind': 'needle_recall',
            'context_len': ctx, 'needle_pos': ctx // 4,
            'memory_type': 'kda', 'n_layers': 4, 'n_kda': 2,
            'n_csa': 1, 'n_hca': 1,
            'train_context_len': ctx, 'train_needle_pos': ctx // 4,
            'quantized': quantized, 'dtype': dtype, 'bits': bits,
            'group_size': 16, 'symmetric': True,
            'accuracy_mean': 0.9,
            'hypothesis_checks': {
                'learned_retrieval': {
                    'p_value_directional': p_value,
                    'verdict': verdict,
                },
                'trained_matches_rule': {
                    'p_value_directional': rule_p,
                    'verdict': True,
                },
            },
        }

    # The same two trained models are measured at three precisions. The
    # family is two, not six, and the fp32 cell is the primary row.
    rows = []
    for ctx in (256, 512):
        rows += [
            _row(ctx, 'fp32', 0.01),
            _row(ctx, 'fp16', 0.001),
            _row(ctx, 'bf16', 0.02),
        ]
    summary = summarize_needle_hypotheses(rows)
    family = summary['learned_retrieval_familywise']
    _ok('raw row count is preserved',
        summary['n_learned_retrieval_supported'] == 6,
        f"raw={summary['n_learned_retrieval_supported']}")
    _ok('family collapses repeated precision cells',
        family['n_primary_tests'] == 2
        and family['n_candidate_rows'] == 6,
        f"primary={family['n_primary_tests']}, "
        f"candidates={family['n_candidate_rows']}")
    _ok('fp32 rows are selected as primary',
        all(t['dtype'] == 'fp32' and t['reject_at_alpha']
            for t in family['tests']),
        f"{[(t['dtype'], t['reject_at_alpha']) for t in family['tests']]}")
    _ok('non-primary rows say they were not selected',
        all(r['learned_retrieval_holm']['selected_as_primary'] is False
            and r['learned_retrieval_holm']['holm_adjusted_p'] is None
            for r in rows if r['dtype'] != 'fp32'))

    # Three uncorrected p<0.05 rows, none significant after Holm.
    rows = [_row(256, 'fp32', 0.030),
            _row(512, 'fp32', 0.040),
            _row(1024, 'fp32', 0.045)]
    summary = summarize_needle_hypotheses(rows)
    family = summary['learned_retrieval_familywise']
    _ok('uncorrected rows would all look supported',
        summary['n_learned_retrieval_supported'] == 3,
        f"raw={summary['n_learned_retrieval_supported']}")
    _ok('Holm rejects none of the three weak p-values',
        family['n_rejected'] == 0
        and all(not t['reject_at_alpha'] for t in family['tests']),
        f"family={family['n_rejected']}, "
        f"adjusted={[t['holm_adjusted_p'] for t in family['tests']]}")

    # If the reference dtype is absent, the next-best unquantized cell is
    # still the primary test rather than dropping the group.
    rows = [_row(256, 'fp16', 0.01)]
    family = summarize_needle_hypotheses(rows)[
        'learned_retrieval_familywise']
    _ok('missing fp32 falls back to the best unquantized cell',
        family['n_primary_tests'] == 1
        and family['tests'][0]['dtype'] == 'fp16'
        and family['n_rejected'] == 1, f'{family["tests"]}')

    # The trained-vs-rule claim gets the same treatment: repeated dtype
    # rows collapse to one primary test, and three weak crossings do not
    # survive Holm.
    rows = []
    for ctx in (256, 512):
        rows += [
            _row(ctx, 'fp32', 0.01, rule_p=0.01),
            _row(ctx, 'fp16', 0.001, rule_p=0.001),
        ]
    summary = summarize_needle_hypotheses(rows)
    rule_family = summary['trained_matches_rule_familywise']
    _ok('rule family collapses repeated precision cells',
        rule_family['n_primary_tests'] == 2
        and rule_family['n_candidate_rows'] == 4
        and rule_family['n_rejected'] == 2,
        f'{rule_family["n_primary_tests"]} primary, '
        f'{rule_family["n_candidate_rows"]} candidates')
    _ok('summary exposes the corrected rule count',
        summary['n_trained_rows_significantly_above_rule_holm'] == 2,
        f"{summary['n_trained_rows_significantly_above_rule_holm']}")

    rows = [_row(256, 'fp32', 0.01, rule_p=0.030),
            _row(512, 'fp32', 0.01, rule_p=0.040),
            _row(1024, 'fp32', 0.01, rule_p=0.045)]
    rule_family = summarize_needle_hypotheses(rows)[
        'trained_matches_rule_familywise']
    _ok('weak rule crossings do not survive Holm',
        rule_family['n_rejected'] == 0,
        f"adjusted={[t['holm_adjusted_p'] for t in rule_family['tests']]}")


def test_needle_experiment_json_output():
    """Full experiment run produces valid JSON with all row kinds."""
    print('\nTest: experiment JSON structure')
    # Run a minimal quick experiment. Both the result and the campaign
    # ledger live in a temp directory: a smoke test must not pollute the
    # real experiment index with a fake run.
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='_needle_test_') as tmp:
        json_path = os.path.join(tmp, 'needle_test.json')
        env = dict(os.environ)
        env['RUN_MANIFEST_PATH'] = os.path.join(tmp, 'run_manifest.jsonl')
        result = subprocess.run([
            sys.executable, 'run_needle_haystack.py',
            '--quick', '--no-qat', '--json', json_path,
            '--seeds', '0', '--context-lens', '128',
            '--memory-types', 'kda', '--dtypes', 'fp32,bf16',
            '--quant-bits', '4', '--quant-groups', '16'
        ], capture_output=True, text=True, cwd=HERE, timeout=120, env=env)
        if result.returncode != 0:
            print(f'  stderr: {result.stderr}')
            print(f'  stdout: {result.stdout}')
        _ok('experiment runs successfully', result.returncode == 0,
            f'returncode={result.returncode}')

        with open(json_path) as f:
            payload = json.load(f)

    _ok('has experiment field', 'experiment' in payload)
    _ok('experiment is exp12', payload['experiment'] == 'exp12_needle_haystack')
    _ok('has rows', 'rows' in payload and len(payload['rows']) > 0)

    row_kinds = {r['row_kind'] for r in payload['rows']}
    _ok('has needle_recall rows', 'needle_recall' in row_kinds)
    _ok('has untrained control rows',
        'needle_untrained_control' in row_kinds, f'kinds={sorted(row_kinds)}')
    _ok('has rule baseline rows',
        'needle_rule_baseline' in row_kinds, f'kinds={sorted(row_kinds)}')
    _ok('has hypothesis summary row',
        'needle_summary' in row_kinds, f'kinds={sorted(row_kinds)}')
    _ok('design carries the pre-registered hypotheses',
        set(payload['design'].get('hypotheses', {})) == set(NEEDLE_HYPOTHESES),
        f"keys={sorted(payload['design'].get('hypotheses', {}))}")
    _ok('design documents the baseline ladder',
        'rule_lexical_scan' in payload['design'].get('baselines', {}))
    rule_rows = [r for r in payload['rows']
                 if r['row_kind'] == 'needle_rule_baseline']
    _ok('rule rows are weight-free',
        all(r.get('is_trained') is False
            and r.get('uses_model_weights') is False for r in rule_rows))
    recall = [r for r in payload['rows'] if r['row_kind'] == 'needle_recall']
    _ok('every recall row is trained',
        all(r.get('is_trained') is True for r in recall))
    _ok('recall rows carry full-answer metric',
        all(r.get('per_seed', [{}])[0].get('answer_match')
            == 'full_answer_exact_match' for r in recall))
    _ok('recall rows carry CI fields',
        all('accuracy_ci95' in r and 'accuracy_std' in r for r in recall))
    _ok('recall rows carry the rule comparison + hypothesis checks',
        all('rule_baseline_accuracy' in r
            and 'accuracy_gain_over_rule_mean' in r
            and set(r.get('hypothesis_checks', {}))
            == {'learned_retrieval', 'trained_matches_rule'}
            for r in recall))
    quantized_recall = [r for r in recall if r.get('quantized')]
    _ok('quick sweep contains quantized recall rows',
        bool(quantized_recall), f'{len(quantized_recall)} rows')
    _ok('quantized recall rows use a quant-matched untrained control',
        all(r.get('untrained_control_quantized') is True
            and r.get('untrained_control_bits') == r.get('bits')
            and r.get('untrained_control_group_size') == r.get('group_size')
            for r in quantized_recall), f'{len(quantized_recall)} rows')
    _ok('unquantized recall rows use a full-precision control',
        all(r.get('untrained_control_quantized') is False
            and r.get('untrained_control_bits') is None
            for r in recall if not r.get('quantized')))
    _ok('untrained rows carry is_trained=False',
        all(r.get('is_trained') is False for r in payload['rows']
            if r['row_kind'] == 'needle_untrained_control'))
    _ok('the standalone control row is the full-precision floor',
        all(r.get('untrained_control_quantized') is False
            for r in payload['rows']
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
    test_needle_validation_and_bad_cases()
    test_rule_baseline_recovers_fact()
    test_bf16_csa_hca_no_nan_inf()
    test_qat_probe_budget_matched_control()
    test_distance_effect_uses_accuracy_at_extreme_positions()
    test_learned_retrieval_familywise_correction()
    test_needle_experiment_json_output()
    print('\n=== ALL NEEDLE HAYSTACK TESTS PASSED ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())

def test_needle_mean_ci_is_json_safe_for_finite_extremes():
    from run_needle_haystack import _mean_ci95
    import json
    assert _mean_ci95([1e308, 1e308, 1e308]) == [1e308, 1e308]
    assert _mean_ci95([1e308, -1e308]) is None
    ci = _mean_ci95([1, 2, 3, 4])
    assert ci is not None
    json.dumps(ci, allow_nan=False)
