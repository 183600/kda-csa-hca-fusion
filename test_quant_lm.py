"""Regression tests for Exp 11 (``run_quant_lm``) and its autograd contracts.

Covers four layers:

1. **In-place inference quantizer** (``kv_quant.fake_quantize_``) — exact
   equality with the functional path, idempotence on the grid, dtype/shape
   preservation, non-finite rejection, and the inference-only guard
   (autograd-tracked inputs are refused).
2. **STE / autograd contract** — the straight-through quantizer is the
   identity on the backward pass (single application, composite linear
   map, asymmetric + tail-padded groups), ``ste=False`` has zero gradient,
   and a multi-step KDA recurrence trained through 2-bit state writes
   produces finite, non-zero gradients with the exact STE identity at the
   last write. This is the QAT contract Exp 11 Part B exercises.
3. **Stepwise KDA LM + QAT smoke** — forward shape/dtype, finite loss,
   finite non-zero gradients through 2-bit STE state writes, and
   run-to-run determinism under a fixed seed.
4. **Report self-consistency** — ``build_ptq_rows`` / ``build_qat_rows``
   aggregation (mean/min/max, min-over-seeds tolerance flag, recovery
   fraction), correct frontier pick, and validation of the committed
   ``results/exp11_quant_lm.json`` when present (schema v1.0, byte
   self-consistency, required row kinds).

The tests never write ``results/``; the committed artifact is only read.

Run via pytest::

    pytest -q test_quant_lm.py

Or directly::

    python3 test_quant_lm.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from kaggle_setup import sanitize_for_json  # noqa: E402
from kv_quant import (  # noqa: E402
    QuantSpec,
    bits_per_element,
    fake_quantize,
    fake_quantize_,
)
from ops_kda import naive_recurrent_kda  # noqa: E402
from results_schema import load as load_schema  # noqa: E402
from run_quant_lm import (  # noqa: E402
    QAT_SPEC,
    StepwiseKDALM,
    build_corpus,
    build_ptq_rows,
    build_qat_rows,
    run_part_b,
)


def _ok(name, cond, detail=''):
    """Print a PASS/FAIL line AND assert, so pytest counts failures."""
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}: {detail}")
    assert cond, f'{name}: {detail}'
    return cond


def _torch_seed(seed=0):
    torch.manual_seed(seed)


def _randn(*shape, **kwargs):
    return torch.randn(*shape, **kwargs)


# --- 1. in-place inference quantizer --------------------------------------

def test_fake_quantize_inplace_contract():
    print('\nTest: fake_quantize_ in-place inference contract')
    spec = QuantSpec(bits=4, group_size=16)
    x = _randn(3, 5, 33)
    ref = fake_quantize(x, spec, ste=False)
    y = x.clone()
    returned = fake_quantize_(y, spec)
    _ok('returns the same tensor object (in-place)',
        returned is y, f'{returned is y}')
    _ok('exactly equals fake_quantize(ste=False)',
        bool(torch.equal(y, ref)), '')
    _ok('keeps shape', y.shape == x.shape, f'{tuple(y.shape)}')

    # Idempotence: re-quantizing grid values is exact for symmetric
    # (asymmetric min/max re-scaling is exact only up to fp32 round-off).
    y2 = y.clone()
    fake_quantize_(y2, spec)
    _ok('symmetric idempotent', bool(torch.equal(y2, y)), '')
    asym = QuantSpec(bits=4, group_size=64, symmetric=False)
    a = fake_quantize(_randn(4, 40), asym, ste=False)
    first = fake_quantize(a, asym, ste=False)
    second = fake_quantize(first, asym, ste=False)
    _ok('asymmetric re-quantization is stable',
        float((first - second).abs().max()) <= 1e-6,
        f'max drift {float((first - second).abs().max()):.2e}')

    # dtype preservation (fp16 input stays fp16).
    h = torch.randn(4, 32, dtype=torch.float16)
    fake_quantize_(h, QuantSpec(bits=4, group_size=8))
    _ok('fp16 dtype preserved', h.dtype == torch.float16, f'{h.dtype}')

    # Empty tensor is a no-op.
    e = torch.empty(0)
    fake_quantize_(e, spec)
    _ok('empty tensor no-op', e.numel() == 0, '')

    # Non-finite rejection (never launder NaN/Inf into the grid).
    bad = torch.tensor([1.0, float('nan')])
    try:
        fake_quantize_(bad, spec)
        _ok('non-finite rejected', False, 'no error')
    except ValueError:
        _ok('non-finite rejected', True)

    # Inference-only: autograd-tracked inputs are refused.
    try:
        fake_quantize_(torch.randn(3, requires_grad=True), spec)
        _ok('requires_grad input rejected', False, 'no error')
    except ValueError:
        _ok('requires_grad input rejected', True)


# --- 2. STE / autograd contract -------------------------------------------

def test_ste_autograd_contract():
    print('\nTest: straight-through gradient contract (QAT support)')
    _torch_seed(0)
    for spec in (QuantSpec(bits=4, group_size=16),
                 QuantSpec(bits=2, group_size=64),
                 QuantSpec(bits=4, group_size=64, symmetric=False),
                 QuantSpec(bits=3, group_size=7)):  # 7 -> tail padding
        x = _randn(4, 33, requires_grad=True)
        w = _randn(4, 33)
        loss = (w * fake_quantize(x, spec, ste=True)).sum()
        (g,) = torch.autograd.grad(loss, x)
        _ok(f'STE is identity ({spec.bits}b/g{spec.group_size}'
            f'{"a" if not spec.symmetric else "s"})',
            bool(torch.equal(g, w)), f'max|g-w|={float((g - w).abs().max()):.2e}')

    # Composite linear map: d/dx w . fq(A x) == d/dx w . (A x) exactly,
    # because the STE contributes an identity Jacobian at the quantizer.
    x = _randn(2, 8, requires_grad=True)
    A = _randn(5, 8)
    w = _randn(5)
    spec = QuantSpec(bits=2, group_size=16)
    g_quant = torch.autograd.grad(
        (w * fake_quantize(x @ A.t(), spec, ste=True)).sum(), x)[0]
    g_surrogate = torch.autograd.grad((w * (x @ A.t())).sum(), x)[0]
    _ok('composite map gradient equals the unquantized surrogate',
        bool(torch.equal(g_quant, g_surrogate)),
        f'max diff {float((g_quant - g_surrogate).abs().max()):.2e}')

    # ste=False (inference) has an exact-zero gradient (round/clamp are
    # flat almost everywhere; scales are computed under no_grad).
    x = _randn(3, 40, requires_grad=True)
    loss = fake_quantize(x, QuantSpec(bits=4, group_size=8),
                         ste=False).pow(2).sum()
    (g,) = torch.autograd.grad(loss, x)
    _ok('ste=False gradient is exactly zero',
        bool((g == 0).all()), f'max|g|={float(g.abs().max()):.2e}')

    # Multi-step recurrence with 2-bit STE state writes: gradients flow
    # through the whole chain, are finite and non-zero, and the LAST write
    # sees the exact STE identity (d loss / d pre-quant state == w).
    _torch_seed(1)
    B, T, H, K, V = 1, 3, 1, 4, 3
    q = _randn(B, T, H, K, requires_grad=True)
    k = _randn(B, T, H, K, requires_grad=True)
    v = _randn(B, T, H, V, requires_grad=True)
    g = _randn(B, T, H, K, requires_grad=True)
    beta_pre = _randn(B, T, H, requires_grad=True)
    beta = torch.sigmoid(beta_pre)
    spec = QuantSpec(bits=2, group_size=8)
    state = None
    pre = None
    outs = []
    for t in range(T):
        o, state = naive_recurrent_kda(
            q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1],
            beta[:, t:t + 1], initial_state=state, output_final_state=True)
        pre = state
        pre.retain_grad()
        outs.append(o)
        state = fake_quantize(state.reshape(B, -1), spec,
                              ste=True).reshape(B, H, K, V)
    w = _randn(B, H * K * V)
    loss = (w * state.reshape(B, -1)).sum()
    # ``q`` only enters the recurrence through the outputs (the state update
    # has no q term), so the state-only loss above leaves q disconnected:
    # autograd may then legitimately report ``q.grad is None`` (dead-branch
    # elision — whether it does depends on the TorchScript executor state,
    # making the check order-dependent). Give q a real path by adding the
    # EARLIER steps' outputs to the loss; the LAST step's output is excluded
    # because it depends on ``pre`` and would perturb the exact STE identity
    # asserted immediately below.
    for o_t in outs[:-1]:
        loss = loss + o_t.sum()
    loss.backward()
    _ok('last state write carries the exact STE identity',
        bool(torch.equal(pre.grad.reshape(B, -1), w)),
        f'max|pre.grad-w|={float((pre.grad.reshape(B, -1) - w).abs().max()):.2e}')
    _ok('value gradient flows through the quantized recurrence',
        v.grad is not None and bool(torch.isfinite(v.grad).all())
        and float(v.grad.abs().sum()) > 0.0,
        f'|v.grad|_1={0.0 if v.grad is None else float(v.grad.abs().sum()):.4f}')
    for name, tensor in (('q', q), ('k', k), ('g', g), ('beta', beta_pre)):
        grad_ok = (tensor.grad is not None
                   and bool(torch.isfinite(tensor.grad).all()))
        _ok(f'{name} gradient finite', grad_ok, '')


# --- 3. stepwise KDA LM + QAT smoke ---------------------------------------

def _qat_train_losses(seed, steps=3, qat=True):
    _torch_seed(seed)
    model = StepwiseKDALM(vocab_size=16, d_model=8, H=1, K=4, V=4)
    ids = torch.randint(2, 16, (2, 6))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        logits = model(ids, spec=QAT_SPEC if qat else None, ste=qat)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return model, losses


def test_stepwise_kda_lm_qat_smoke():
    print('\nTest: stepwise KDA LM forward + 2-bit STE QAT smoke')
    model, losses = _qat_train_losses(seed=0, steps=1, qat=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randint(2, 16, (2, 6)))
    _ok('forward logits shape [B, T, vocab]',
        tuple(logits.shape) == (2, 6, 16), f'{tuple(logits.shape)}')
    _ok('forward logits finite', bool(torch.isfinite(logits).all()), '')
    _ok('unquantized loss finite', math.isfinite(losses[0]),
        f'{losses[0]:.4f}')

    model, losses = _qat_train_losses(seed=0, steps=3, qat=True)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = any(float(g.abs().sum()) > 0.0 for g in grads)
    _ok('2-bit QAT loss finite on every step',
        all(math.isfinite(v) for v in losses), f'{losses}')
    _ok('all QAT parameter gradients finite', finite, '')
    _ok('QAT gradients are non-zero (autograd reaches the state writes)',
        nonzero, '')

    _, losses_a = _qat_train_losses(seed=7, steps=3, qat=True)
    _, losses_b = _qat_train_losses(seed=7, steps=3, qat=True)
    _ok('QAT training is deterministic under a fixed seed',
        losses_a == losses_b, f'{losses_a} vs {losses_b}')


def test_part_b_matched_budget_control():
    """Unequal QLM_STEPS / QLM_QAT_STEPS gets an equal-budget control."""
    print('\nTest: Part B budget-matched fp16 control')
    tok, splits, _ = build_corpus(seq_len=8, n_texts=8, seed=0)
    rows = run_part_b(
        device='cpu', tokenizer=tok, split_texts=splits, seeds=(0,),
        qat_steps=2, seq_len=8, batch_size=2, eval_seqs=1,
        spec=QAT_SPEC, steps=1, lr=3e-3)
    row = rows[0]
    _ok('unequal historical budgets are flagged and matched',
        row['historical_budget_matched'] is False
        and row['matched_control_reused'] is False
        and row['total_steps_qat'] == row['total_steps_fp16_matched'] == 2
        and row['total_steps_fp16'] == 1, f"{row['total_steps_qat']}")
    _ok('the matched control is separately trained and scored',
        math.isfinite(row['fp16_matched_ppl'])
        and math.isfinite(row['ptq_matched_ppl'])
        and row['fp16_matched_ppl'] != row['fp16_ppl'], '')
    _ok('matched recovery is derived from the matched damage',
        (row['recovery_fraction_matched'] is None)
        or abs((1.0 - (row['qat_ppl'] / row['fp16_matched_ppl'] - 1.0)
                / (row['ptq_matched_ppl'] / row['fp16_matched_ppl'] - 1.0))
               - row['recovery_fraction_matched']) < 1e-9, '')
    _ok('matched log-PPL difference is finite',
        row['qat_minus_matched_log_ppl'] is not None
        and math.isfinite(row['qat_minus_matched_log_ppl']), '')

    rows_equal = run_part_b(
        device='cpu', tokenizer=tok, split_texts=splits, seeds=(0,),
        qat_steps=1, seq_len=8, batch_size=2, eval_seqs=1,
        spec=QAT_SPEC, steps=1, lr=3e-3)
    eq = rows_equal[0]
    _ok('equal budgets reuse the historical fp16 arm',
        eq['historical_budget_matched'] is True
        and eq['matched_control_reused'] is True
        and eq['fp16_matched_ppl'] == eq['fp16_ppl']
        and eq['ptq_matched_ppl'] == eq['ptq_ppl'], '')


# --- 4. report self-consistency -------------------------------------------

def _fake_window_units(total_nll, counts, profile):
    """Window units whose token-weighted mean reproduces the cell's NLL."""
    assert abs(sum(profile) - 1.0) < 1e-12
    return [{'nll_sum': total_nll * w, 'n_tokens': c}
            for w, c in zip(profile, counts)]


_FP16_PROFILE = [0.24, 0.30, 0.22, 0.24]
_QUANT_PROFILE = [0.20, 0.36, 0.20, 0.24]


def _fake_cells():
    """Minimal per-seed cells accepted by the row builders."""
    spec = QuantSpec(bits=2, group_size=16)
    cells = []
    for seed, fp16_ppl, quant_ppl in ((0, 2.0, 2.1), (1, 3.0, 3.3)):
        cells.append({
            'seed': seed, 'spec': None, 'prefill_ppl': fp16_ppl * 1.001,
            'decode_ppl': fp16_ppl, 'mean_nll': math.log(fp16_ppl),
            'n_tokens': 100, 'top1_agreement': 1.0,
            'state_elements': 512, 'state_snr_db': math.inf,
            'per_window': _fake_window_units(math.log(fp16_ppl) * 100,
                                             [25, 25, 25, 25],
                                             _FP16_PROFILE),
        })
        cells.append({
            'seed': seed, 'spec': spec, 'prefill_ppl': fp16_ppl * 1.001,
            'decode_ppl': quant_ppl, 'mean_nll': math.log(quant_ppl),
            'n_tokens': 100, 'top1_agreement': 0.9,
            'state_elements': 512, 'state_snr_db': 10.0,
            'per_window': _fake_window_units(math.log(quant_ppl) * 100,
                                             [25, 25, 25, 25],
                                             _QUANT_PROFILE),
        })
    return cells


def test_quant_lm_report_rows():
    print('\nTest: Exp 11 row aggregation + committed artifact')
    spec = QuantSpec(bits=2, group_size=16)
    cells = _fake_cells()
    rows = build_ptq_rows(cells, [None, spec], tolerance=0.01)
    ptq = [r for r in rows if r['row_kind'] == 'ptq_ppl']
    summary = [r for r in rows if r['row_kind'] == 'ptq_summary']
    _ok('one aggregated row per spec', len(ptq) == 2, f'{len(ptq)}')
    _ok('one summary row', len(summary) == 1, f'{len(summary)}')
    q = next(r for r in ptq if r['quantized'])
    ratios = [2.1 / 2.0, 3.3 / 3.0]
    _ok('ppl_ratio aggregates are correct',
        abs(q['ppl_ratio_mean'] - sum(ratios) / 2) < 1e-12
        and abs(q['ppl_ratio_min'] - min(ratios)) < 1e-12
        and abs(q['ppl_ratio_max'] - max(ratios)) < 1e-12,
        f"{q['ppl_ratio_mean']:.6f}")
    _ok('ppl_preserving uses the conservative min-over-seeds ratio',
        q['ppl_preserving'] == (min(ratios) <= 1.01),
        f"{q['ppl_preserving']}")
    _ok('state_bytes matches elements * bits_per_element / 8',
        abs(q['state_bytes']
            - 512 * bits_per_element(spec) / 8.0) < 1e-9,
        f"{q['state_bytes']:.3f}")
    s = summary[0]
    _ok('summary names the cheapest preserving cell from the grid',
        (s['cheapest_ppl_preserving_cell'] is None
         and not q['ppl_preserving'])
        or (s['cheapest_ppl_preserving_cell'] is not None
            and abs(s['cheapest_ppl_preserving_cell']['state_bytes']
                    - q['state_bytes']) < 1e-9),
        '')
    _ok('prefill/decode reference ratio is recorded',
        math.isfinite(s['prefill_decode_ppl_ratio'])
        and s['prefill_decode_ppl_ratio'] > 0.0,
        f"{s['prefill_decode_ppl_ratio']:.6f}")

    # Paired quantized-vs-fp16 bootstrap (per-window units are supplied by
    # _fake_cells, so the paired path must be populated and self-consistent).
    paired = q['paired_vs_fp16']
    _ok('paired_vs_fp16 is populated when per-window units exist',
        paired is not None and paired['n_seeds_tested'] == 2,
        f"{None if paired is None else paired['n_seeds_tested']}")
    _ok('paired deltas are positive (quantized has higher NLL) for both '
        'seeds',
        paired is not None
        and all(r['delta_mean_nll'] > 0.0 for r in paired['per_seed'])
        and paired['delta_mean_nll_min'] > 0.0,
        '' if paired is None else
        f"{[round(r['delta_mean_nll'], 4) for r in paired['per_seed']]}")
    _ok('paired CI bounds are ordered and the ratio CI is positive',
        paired is not None and all(
            r['ci_delta_mean_nll'][0] <= r['ci_delta_mean_nll'][1]
            and 0.0 < r['ci_ppl_ratio'][0] <= r['ci_ppl_ratio'][1]
            for r in paired['per_seed']),
        '')
    _ok('fp16 reference row carries no paired comparison',
        next(r for r in ptq if not r['quantized'])['paired_vs_fp16'] is None,
        '')
    _ok('summary counts cells whose damage is significant for all seeds',
        isinstance(s['n_cells_significant_damage_all_seeds'], int)
        and 0 <= s['n_cells_significant_damage_all_seeds'] <= len(ptq),
        f"{s['n_cells_significant_damage_all_seeds']}")

    qat_cells = [
        {'seed': 0, 'steps': 10, 'qat_steps': 20, 'fp16_ppl': 2.0,
         'ptq_ppl': 4.0, 'qat_ppl': 2.5, 'qat_model_fp16_state_ppl': 2.1,
         'recovery_fraction': 0.75, 'grad_norm_first': 1.0,
         'grad_norm_last': 0.5, 'all_grads_finite': True,
         'fp16_matched_ppl': 1.9, 'ptq_matched_ppl': 3.8,
         'recovery_fraction_matched': 0.7,
         'qat_minus_matched_log_ppl': math.log(2.5 / 1.9),
         'historical_budget_matched': False, 'matched_control_reused': False},
        {'seed': 1, 'steps': 10, 'qat_steps': 20, 'fp16_ppl': 3.0,
         'ptq_ppl': 3.0, 'qat_ppl': 3.1, 'qat_model_fp16_state_ppl': 3.0,
         'recovery_fraction': None, 'grad_norm_first': 1.0,
         'grad_norm_last': 0.5, 'all_grads_finite': True,
         'fp16_matched_ppl': 2.9, 'ptq_matched_ppl': 3.2,
         'recovery_fraction_matched': 0.5,
         'qat_minus_matched_log_ppl': math.log(3.1 / 2.9),
         'historical_budget_matched': False, 'matched_control_reused': False},
    ]
    qrows = build_qat_rows(qat_cells, spec)
    agg = next(r for r in qrows if r['row_kind'] == 'qat_recovery')
    _ok('QAT recovery aggregates ignore missing fractions',
        abs(agg['recovery_fraction_mean'] - 0.75) < 1e-12
        and abs(agg['recovery_fraction_min'] - 0.75) < 1e-12,
        f"{agg['recovery_fraction_mean']}")
    _ok('QAT summary carries mean +/- std + CI',
        agg.get('qat_ppl_std') is not None
        and agg.get('qat_ppl_ci95') is not None
        and agg.get('recovery_fraction_matched_std') is not None, '')
    _ok('matched-budget control is aggregated and flagged',
        abs(agg['fp16_matched_ppl_mean'] - 2.4) < 1e-12
        and agg['budget_control']['historical_arms_equal_budget'] is False
        and agg['budget_control']['total_steps_qat'] == 20
        and agg['budget_control']['total_steps_fp16_matched'] == 20, '')
    paired = agg.get('qat_vs_matched_control')
    _ok('QAT vs matched control has a paired multi-seed test',
        paired is not None and paired['n_pairs'] == 2
        and paired['mean_diff'] is not None
        and paired['t_test']['p_value_two_sided'] is not None,
        f"{None if paired is None else paired['mean_diff']}")
    try:
        json.dumps(sanitize_for_json(qrows), allow_nan=False)
        qat_json_ok = True
    except (ValueError, TypeError):
        qat_json_ok = False
    _ok('QAT rows round-trip as strict JSON', qat_json_ok, '')

    artifact = os.path.join(HERE, 'results', 'exp11_quant_lm.json')
    if not os.path.exists(artifact):
        print('  [SKIP] committed results/exp11_quant_lm.json not present yet')
        return
    with open(artifact, encoding='utf-8') as f:
        payload = json.load(f)
    _ok('artifact schema kind', payload.get('kind') == 'exp11_quant_lm',
        f"{payload.get('kind')}")
    _ok('artifact schema version', payload.get('schema_version') == '1.0',
        f"{payload.get('schema_version')}")
    _meta, committed = load_schema(artifact)
    kinds = {r['row_kind'] for r in committed}
    _ok('artifact carries all row kinds',
        {'ptq_per_seed', 'ptq_ppl', 'ptq_summary', 'qat_per_seed',
         'qat_recovery'} <= kinds, f'{sorted(kinds)}')
    bytes_ok = True
    for r in committed:
        if r['row_kind'] != 'ptq_ppl':
            continue
        if r['quantized']:
            bpe = bits_per_element(QuantSpec(
                r['bits'], r['group_size'], symmetric=r['symmetric']))
        else:
            bpe = 16.0
        if abs(r['state_bytes'] - r['state_elements'] * bpe / 8.0) > 1e-9:
            bytes_ok = False
    _ok('artifact byte columns are self-consistent', bytes_ok, '')
    fp16 = next(r for r in committed
                if r['row_kind'] == 'ptq_ppl' and not r['quantized'])
    _ok('fp16 reference row is unique and ordered',
        fp16['decode_ppl_min'] <= fp16['decode_ppl_mean']
        <= fp16['decode_ppl_max']
        and abs(fp16['ppl_ratio_mean'] - 1.0) < 1e-12, '')


_TEST_FUNCS = [
    test_fake_quantize_inplace_contract,
    test_ste_autograd_contract,
    test_stepwise_kda_lm_qat_smoke,
    test_part_b_matched_budget_control,
    test_quant_lm_report_rows,
]


def main() -> int:
    print('=' * 70)
    print('Exp 11 (state-quantized perplexity / QAT) Regression Tests')
    print('=' * 70)
    results = []
    for fn in _TEST_FUNCS:
        try:
            fn()
            results.append(True)
        except AssertionError as e:
            print(f'  [FAIL] {fn.__name__}: AssertionError: {e}')
            results.append(False)
        except Exception as e:
            print(f'  [CRASH] {fn.__name__}: {type(e).__name__}: {e}')
            results.append(False)
    n_pass = sum(1 for r in results if r)
    print('-' * 70)
    print(f'Total: {n_pass}/{len(results)} passed')
    return 0 if n_pass == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
