"""Experiment 5 — hybrid layout ablation (multi-seed, device-aware).

Sweeps over different KDA:CSA:HCA ratios in the fused hybrid block and
measures (a) end-to-end forward latency and (b) MQAR accuracy with the same
training recipe as Experiment 4. This quantifies the design trade-off:

  * more KDA layers  -> faster, lower memory, but weaker long-range recall
  * more CSA layers   -> better sparse recall, but slower (top-k selection)
  * more HCA layers   -> cheapest global context, but coarser compression

Kaggle / review-driven additions (address reviewer concerns):

  * **Multi-seed runs.** Each ratio is trained over ``n_seeds`` seeds with
    mean +/- CI95 reported. The original paper's single-seed table had
    3:1:1=0.078 vs chance=0.0625 — within noise. Multi-seed makes the
    comparison statistically meaningful (or honestly shows it is not).
  * **More training steps.** The original 25 steps was far too few for the
    deeper 4:1:1 (6-layer) model to converge, which likely explains its
    anomalously low 0.031 score. We use 100+ steps and report convergence
    curves.
  * **Controlled parameter count.** We report ``n_params`` per ratio and add
    a note when ratios have different depths (e.g. 4:1:1 = 6 layers vs
    3:1:1 = 5 layers), so the comparison is honest about depth confounds.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggle_setup import configure_torch_for_device, parse_int_env, sanitize_for_json, write_json_atomic
from ops_fused import HybridKCHAttention, HybridConfig
from run_quality import (
    make_mqar_batch, MQARHead, _parse_nkv_list, _fmt_tstat, _t_crit_975,
    _bonferroni_crit_q, _build_param_groups, SMALL_MODEL_SPEC,
)

logger = logging.getLogger(__name__)

# Vocab and seq_len are task hyperparameters. Lifted to a single module-level
# constant so the chance baseline (1/VOCAB) is computed in exactly ONE place
# and stays consistent with the vocab passed to make_mqar_batch / nn.Embedding.
# The previous code hardcoded ``1/16`` in three independent sites (the chance
# baseline in eval_layout_multi_seed, the summary-table chance row, and the
# vocab arg to make_mqar_batch / nn.Embedding), which would silently lie if
# any one site were ever changed.
VOCAB = 16
SEQ_LEN = 16


def _make_cfg(d_model=32, ratio=(3, 1, 1)):
    # Validate ratio so a typo (e.g. ``ratio=(3, 1)`` or ``ratio=(-1, 1, 1)``)
    # produces a clear error instead of an opaque ValueError from the
    # tuple-unpack or a silently-empty KDA layout (``['kda'] * -1 == []``).
    # NOTE: use ``raise ValueError`` (NOT ``assert``) so the checks
    # survive ``python -O`` / ``PYTHONOPTIMIZE=1`` — ``assert`` statements
    # are silently stripped under optimization, which would re-expose the
    # cryptic ValueError / silent-empty-layout this guard is meant to
    # prevent. Mirrors the convention established in ops_kda.py.
    if not (isinstance(ratio, tuple) and len(ratio) == 3):
        raise ValueError(
            f"ratio must be a 3-tuple (n_kda, n_csa, n_hca), got {ratio!r}")
    if not all(isinstance(n, int) and n >= 0 for n in ratio):
        raise ValueError(
            f"ratio components must be non-negative ints, got {ratio!r}")
    n_kda, n_csa, n_hca = ratio
    # Use the shared small-model spec so Experiment 5 (ablation) tests the
    # SAME CSA/HCA sub-layer widths as Experiment 4 (standalone MQAR).
    # Previously this used csa_c=16, csa_cI=8, csa_dc=32, hca_c=16,
    # hca_dc=32 — HALF the width of run_quality.CSAAttn / HCAAttn — making
    # cross-experiment comparisons silently confounded. The shared spec is
    # defined in run_quality.py (imported above) so there is exactly ONE
    # place to tune the small-model architecture.
    spec = SMALL_MODEL_SPEC
    # HCA's defining feature is *heavy* compression: m2 should be >> m so the
    # HCA branch produces far fewer compressed blocks than CSA, trading recall
    # granularity for global context. The shared spec already enforces
    # m2=8 > m=4. With seq_len=16 and m=4, n_blocks_CSA=4; setting m2=8 gives
    # n_blocks_HCA=2, exercising the "heavier compression" regime while
    # staying within the small ablation budget.
    return HybridConfig(
        d_model=d_model, n_heads_qk=2, n_heads_v=2,
        head_dim_k=16, head_dim_v=16,
        csa_m=spec['csa_m'], csa_topk=spec['csa_topk'],
        csa_nh=spec['csa_nh'], csa_c=spec['csa_c'],
        csa_dc=spec['csa_dc'], csa_nIh=spec['csa_nIh'],
        csa_cI=spec['csa_cI'],
        csa_sliding_window=spec['csa_sliding_window'],
        hca_m2=spec['hca_m2'], hca_nh=spec['hca_nh'],
        hca_c=spec['hca_c'], hca_dc=spec['hca_dc'],
        hca_sliding_window=spec['hca_sliding_window'],
        n_kda=n_kda, n_csa=n_csa, n_hca=n_hca,
    )


def _eval_model(model, head, embed, seq_len, n_kv=1, device='cpu',
                n_batches=4, batch=64):
    # Save the prior train/eval state so we can restore it after eval —
    # a latent footgun if a caller ever invokes _eval_model mid-training
    # (e.g. for periodic validation). Mirrors the fix in run_quality.py.
    was_training = {m: m.training for m in (model, head, embed)}
    try:
        model.eval()
        head.eval()
        embed.eval()  # nn.Embedding has no dropout/batchnorm so this is a no-op,
                      # but we set it for symmetry with model/head so future
                      # additions (e.g. embedding dropout) do not silently stay
                      # in train mode during evaluation. Mirrors run_quality.py.
        correct, total = 0, 0
        losses = []
        # Fixed seed for the eval generator so every ratio sees the SAME eval
        # batches (apples-to-apples comparison at eval time too, not just train).
        # Different ratios consume different numbers of RNG draws during model
        # init (different parameter counts), so using the global RNG would desync
        # eval batches across ratios. Mirrors run_quality.py::_eval_model.
        eval_gen = torch.Generator(device=device)
        eval_gen.manual_seed(12345)
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, VOCAB, embed, device, generator=eval_gen)
                model.reset_state()  # independent eval batch
                h = model(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
        # Guard against n_batches=0 (or batch=0): without this the function
        # raises ZeroDivisionError on ``correct / total`` and
        # ``sum([]) / len([])``. Returns 0.0 for both metrics so the caller
        # gets a finite (if meaningless) value rather than a crash. Mirrors
        # run_quality.py::_eval_model.
        if total == 0 or not losses:
            return 0.0, 0.0
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def eval_layout(ratio, d_model=32, seq_len=SEQ_LEN, n_kv=1, steps=100, lr=3e-3, seed=42,
                device='cpu', eval_batches=4, eval_batch=64, train_batch=None):
    # Coerce string device -> torch.device for notebook callers. Mirrors
    # run_quality.py::train_one / train_multi_seed.
    if isinstance(device, str):
        device = torch.device(device)
    torch.manual_seed(seed)
    # Create embed and head BEFORE the ratio-specific model so their initial
    # weights are IDENTICAL across ratios for a given seed. Different ratios
    # have different parameter counts, so creating the model first would
    # consume a different number of RNG draws and desync the downstream
    # embed/head init — a silent confound in the multi-seed CI. (The previous
    # order was model -> head -> embed, which left both embed and head
    # ratio-dependent.) Mirrors the fix in run_quality.py::train_one.
    embed = nn.Embedding(VOCAB, d_model).to(device)
    head = MQARHead(d_model, VOCAB).to(device)
    cfg = _make_cfg(d_model, ratio)
    total = sum(ratio)
    model = HybridKCHAttention(cfg, total_layers=total).to(device)
    # Build parameter groups with proper weight-decay exclusion: embeddings,
    # biases, and LayerNorm parameters are NOT weight-decayed (standard ML
    # practice). Mirrors the fix in run_quality.py::train_one.
    param_groups = _build_param_groups(model, head, embed, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    layout = model.layout_str()
    n_params = sum(p.numel() for p in model.parameters())

    # Separate generator for batch generation so the per-step batches are
    # IDENTICAL across ratios for a given seed. Different ratios have different
    # parameter counts, so they consume a different number of RNG draws during
    # model init. Without a separate generator for batch generation, the same
    # seed would produce different training data per ratio — a silent confound
    # in the multi-seed CI that undermines the apples-to-apples comparison.
    # Mirrors run_quality.py::train_one.
    batch_gen = torch.Generator(device=device)
    batch_gen.manual_seed(seed + 1)  # offset so it does not collide with
                                      # the seed used for model init.

    # Configurable training batch size (default 16, overridable via the
    # ABL_TRAIN_BATCH env var for memory-constrained or GPU runs). Mirrors
    # run_quality.py's MQAR_TRAIN_BATCH. Robust env var parsing: a malformed
    # ``ABL_TRAIN_BATCH=abc`` (or ``=0``, which would crash on the first
    # batch with ZeroDivisionError) previously crashed the whole experiment
    # with no informative error. ``parse_int_env`` logs a warning and falls
    # back to the default, matching the robustness pattern already used for
    # BENCH_REPEATS in run_benchmark.py.
    if train_batch is None:
        train_batch = parse_int_env('ABL_TRAIN_BATCH', 16, min_value=1,
                                    logger=logger)

    # Set train mode ONCE before the loop, not per step. The previous code
    # called model.train()/head.train() inside the loop, which is a redundant
    # O(steps) no-op once the modules are already in train mode. Mirrors
    # run_quality.py::train_one.
    model.train()
    head.train()
    embed.train()

    losses = []
    for step in range(steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, VOCAB, embed, device, generator=batch_gen)
        # Each MQAR batch is independent — clear KDA recurrent state so
        # samples from the previous batch don't leak in.
        model.reset_state()
        h = model(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        # NaN/Inf guard: mirrors run_quality.py::train_one. A divergent step
        # would otherwise propagate NaN into all parameters via backward +
        # opt.step, and the final eval would still return a finite (bogus)
        # accuracy, silently corrupting the aggregate mean/CI. Raise here so
        # the per-seed try/except in eval_layout_multi_seed catches it.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(ratio={ratio}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
        opt.zero_grad()
        loss.backward()
        # Guard against NaN/Inf gradients BEFORE clip+step. clip_grad_norm_
        # computes total_norm = NaN when any grad is NaN, and the
        # ``total_norm > max_norm`` comparison is False for NaN, so no
        # clipping happens and the NaN grads pass through to opt.step(),
        # corrupting all parameters in one step. The next iteration's
        # forward would then produce a NaN loss caught by the guard
        # above, but the seed is already lost without a clear root cause.
        bad_grads = [p for p in params
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad_grads:
            raise RuntimeError(
                f"non-finite gradient at step {step} in {len(bad_grads)} "
                f"params (ratio={ratio}, seed={seed}); aborting this seed "
                f"to prevent NaN propagation into parameters")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())

    # Final eval on multiple batches
    final_acc, final_loss = _eval_model(
        model, head, embed, seq_len, n_kv, device,
        n_batches=eval_batches, batch=eval_batch,
    )

    # Forward latency (on the actual device).
    # Use a dedicated seeded generator (NOT the global RNG) so the latency
    # input is identical across ratios and seeds. The global RNG state
    # differs per ratio because different ratios have different parameter
    # counts (different ``nn.Linear`` init RNG draws during
    # ``HybridKCHAttention.__init__``), so the previous code's
    # ``torch.randn(...)`` produced different inputs across ratios,
    # confounding latency comparisons.
    lat_gen = torch.Generator(device=device)
    lat_gen.manual_seed(99)
    x = torch.randn(1, seq_len, d_model, device=device, generator=lat_gen) * 0.1
    # Switch to eval mode so any future Dropout/BN-style stochasticity
    # does not contaminate the latency measurement. Today the model has
    # only LayerNorm (eval is a no-op), but the guard future-proofs the
    # measurement.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.reset_state()
            model(x)  # warmup
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                model.reset_state()
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t0) / 5 * 1e3
    finally:
        model.train(was_training)

    # Guard against steps=0 (would crash on losses[-1] / sum([])/0 below).
    last_loss = losses[-1] if losses else 0.0
    mean_last10 = sum(losses[-10:]) / min(10, len(losses)) if losses else 0.0
    return {
        'ratio': f'{ratio[0]}:{ratio[1]}:{ratio[2]}',
        'n_kv': n_kv,
        'layout': layout,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'fwd_ms': fwd_ms,
        'n_params': n_params,
        'n_layers': total,
        'seed': seed,
        'steps': steps,
        'train_batch': train_batch,
        'last_train_loss': last_loss,
        'mean_last10_loss': mean_last10,
        # Full per-step loss curve so the figure / paper can plot convergence
        # trajectories. The docstring promised "convergence curves" but the
        # previous version discarded the list and saved only the last value.
        'loss_curve': losses,
    }


def eval_layout_multi_seed(ratio, n_seeds=5, steps=100, device='cpu', **kw):
    # Coerce string device -> torch.device for notebook callers. Mirrors
    # run_quality.py::train_multi_seed.
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    per_seed = []
    for s in seeds:
        t0 = time.time()
        # Per-seed try/except: one divergent seed (NaN loss, OOM, etc.) should
        # not crash the whole ratio. We log and skip; the aggregate stats are
        # computed over whichever seeds succeeded (and n in the CI formula
        # shrinks accordingly).
        try:
            r = eval_layout(ratio, seed=s, steps=steps, device=device, **kw)
            r['train_time_s'] = time.time() - t0
            # Log BEFORE appending so that if the formatter raises (e.g. on an
            # unexpected None field) the except branch appends a single clean
            # stub, not a duplicate of the half-built entry. Mirrors the fix
            # in run_quality.py::train_multi_seed.
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"fwd={r['fwd_ms']:.2f}ms  time={r['train_time_s']:.1f}s")
            per_seed.append(r)
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            per_seed.append({
                'seed': s, 'error': str(e),
                'final_acc': None, 'final_loss': None, 'fwd_ms': None,
            })
        # On GPU, clear the CUDA cache between seeds so the allocator does
        # not accumulate freed-but-unreleased blocks across 35 trainings
        # (5 ratios x 7 seeds x model+opt state), which could cause an
        # avoidable OOM on a constrained GPU.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Filter out failed seeds before computing aggregate stats.
    # A seed that diverged to NaN loss is caught by eval_layout's NaN guard
    # and recorded with an 'error' key — but defensively reject any seed
    # whose final_acc/final_loss is None or non-finite, mirroring
    # run_quality.py::train_multi_seed. NaN propagates through sum()/std()
    # and would silently corrupt the aggregate mean/CI otherwise.
    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        # All seeds failed: propagate the first error so the outer per-ratio
        # try/except can record a stub result.
        raise RuntimeError(f"all {n_seeds} seeds failed for ratio {ratio}; "
                           f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    fwds = [r['fwd_ms'] for r in ok_per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_fwd = sum(fwds) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        std_acc = math.sqrt(var_acc)
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
    else:
        # With one surviving seed, uncertainty is maximal — NOT zero.
        # Returning ``ci_acc = 0.0`` (the previous value) implies perfect
        # precision and misleads downstream figure generation and the
        # ``significant_bonferroni`` flag. Mirror ``t_stat=None`` (line
        # below) which already marks the test as undefined for n=1.
        std_acc = 0.0
        ci_acc = None

    # One-sample t-test vs chance: tests whether mean_acc differs from the
    # chance level (1/VOCAB here). The t-statistic is only defined when n > 1
    # and the sample standard deviation is strictly positive; otherwise we
    # return None (the test is not computable, not "infinitely significant").
    chance = 1.0 / VOCAB
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    return {
        'ratio': ok_per_seed[0]['ratio'],
        'n_kv': ok_per_seed[0]['n_kv'],
        'layout': ok_per_seed[0]['layout'],
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds': len(per_seed),
        'n_seeds_total': len(per_seed),
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat,
        'mean_fwd_ms': mean_fwd,
        'n_params': ok_per_seed[0]['n_params'],
        'n_layers': ok_per_seed[0]['n_layers'],
        'mean_train_time_s': sum(r.get('train_time_s', 0.0) for r in ok_per_seed) / n,
    }


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 5: Hybrid Layout Ablation (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device: {device}')
    # Robust env var parsing: a single malformed value (e.g. ``ABL_SEEDS=abc``)
    # previously crashed the whole multi-seed sweep with a bare
    # ``ValueError: invalid literal for int()``. ``parse_int_env`` logs a
    # warning and falls back to the default, matching the robustness pattern
    # already used for BENCH_REPEATS / BENCH_LENGTHS in run_benchmark.py.
    # P4 fix — increase the default seed count from 5 to 7 and warn loudly
    # when the sample size is too small to support structural conclusions.
    #
    # The issue identified that the ablation was running with only 3 seeds
    # (because run_all.py overrides ABL_SEEDS=3 under SKIP_SLOW on CPU),
    # that most accuracies were near the chance level (0.0625), and that
    # ALL layouts had significant_bonferroni=False. With n=3 a one-sample
    # t-test has only 2 degrees of freedom — the Bonferroni-corrected
    # critical t-value at alpha=0.05/7≈0.0071 is ~12.9, which is essentially
    # unachievable in practice. The experiment as configured CANNOT reach
    # significance regardless of the underlying effect size.
    #
    # Raising the default to 7 (6 dof) brings the corrected critical value
    # down to ~4.9, which is achievable for a real effect. We also add an
    # explicit ``conclusions_valid`` flag to the JSON and a prominent
    # warning in the log so downstream consumers (figures, reports) do not
    # draw strong structural conclusions from underpowered experiments.
    n_seeds = parse_int_env('ABL_SEEDS', 7, min_value=1, logger=logger)
    steps = parse_int_env('ABL_STEPS', 100, min_value=1, logger=logger)
    n_kv_list = _parse_nkv_list('ABL_NKV', '1')
    # Pre-validate n_kv against VOCAB and SEQ_LEN so the user gets a clear
    # error message instead of an opaque AssertionError from deep inside
    # ``make_mqar_batch`` during the first training step (after spending
    # time on model init for every ratio). Mirrors run_quality.py::main.
    for n_kv in n_kv_list:
        if 2 * n_kv > VOCAB:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"VOCAB={VOCAB}; reduce n_kv or increase VOCAB.")
        if 2 * n_kv >= SEQ_LEN:
            raise ValueError(
                f"ABL_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< SEQ_LEN={SEQ_LEN} (need room for the cue token at the end).")
    logger.info(f'  n_seeds={n_seeds}, steps={steps}, n_kv={n_kv_list}')
    ratios = [(3, 1, 1), (4, 1, 1), (2, 1, 1), (1, 1, 1), (3, 0, 1), (3, 1, 0), (0, 1, 1)]
    # Bonferroni correction: we run len(ratios) * len(n_kv_list) one-sample
    # t-tests vs chance at alpha=0.05 each. Without correction the
    # family-wise false-positive rate inflates to ~1-(1-0.05)^(n_tests)
    # (~30% for 7 tests, ~66% for 21). We compute the corrected alpha and
    # the corresponding t-critical value, then flag each result as
    # significant_bonferroni iff |t_stat| exceeds the corrected critical
    # value (with n-1 dof). The raw t_stat and uncorrected interpretation
    # are preserved in the JSON for transparency.
    n_tests = len(ratios) * len(n_kv_list)
    alpha_corrected = 0.05 / n_tests
    # P0-3 fix: use the module-level ``_bonferroni_crit_q`` from run_quality,
    # which has a proper Cornish-Fisher fallback when scipy is unavailable
    # (the previous nested implementation returned None and silently zeroed
    # out all significance conclusions). We just detect scipy availability
    # for the log line below.
    try:
        from scipy.stats import t as _t_dist  # noqa: F401
        bonferroni_available = True
    except ImportError:
        bonferroni_available = False
    logger.info(f'  {n_tests} one-sample t-tests vs chance; '
                f'Bonferroni-corrected alpha={alpha_corrected:.4f} '
                f'(scipy={bonferroni_available}; fallback=Cornish-Fisher)')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for r in ratios:
            logger.info(f'\n-- ratio KDA:CSA:HCA = {r[0]}:{r[1]}:{r[2]} '
                        f'(n_kv={n_kv}, {n_seeds} seeds) --')
            # Per-ratio try/except so ONE failing ratio (OOM, divergence,
            # assertion) does not crash the entire sweep and lose ALL the
            # other ratios' results. The error is logged and recorded as a
            # stub result with status='error' so the JSON file is always
            # written and downstream figure generation can skip the missing
            # ratio gracefully.
            try:
                res = eval_layout_multi_seed(r, n_seeds=n_seeds, steps=steps,
                                             device=device, n_kv=n_kv)
                # Bonferroni significance flag: t_stat exceeds the corrected
                # critical value for this n. Stored alongside the raw t_stat so
                # downstream consumers can show both the uncorrected and the
                # corrected interpretation.
                t_stat = res.get('t_stat_vs_chance')
                n_ok = res.get('n_seeds_ok', 0)
                if t_stat is not None and n_ok >= 2:
                    crit = _bonferroni_crit_q(n_ok, alpha=alpha_corrected)
                    res['t_crit_bonferroni'] = crit
                    # ``crit`` is None when scipy is unavailable (see
                    # ``_bonferroni_crit``). ``t_stat > None`` raises
                    # ``TypeError: '>' not supported between instances of
                    # 'float' and 'NoneType'`` in Python 3, so guard the
                    # comparison: when the critical value cannot be
                    # computed we conservatively report ``False`` (not
                    # significant), matching the previous ``inf``-based
                    # behaviour without the JSON-serialization crash.
                    #
                    # P0-3 fix: use a ONE-SIDED test (``t_stat > crit``) instead
                    # of ``abs(t_stat) > crit``. The research question is "does
                    # this layout learn the task ABOVE chance", which is
                    # directional. The previous two-sided test flagged a layout
                    # as "significant" even when its accuracy was significantly
                    # BELOW chance (large negative t_stat), which is the
                    # opposite of what "this layout works" means. A
                    # below-chance result indicates the model is systematically
                    # wrong, NOT that the layout "works". The Bonferroni-corrected
                    # critical value ``_bonferroni_crit`` is already the
                    # upper-tail quantile, so the one-sided comparison is the
                    # correct use of that quantile.
                    res['significant_bonferroni'] = (
                        crit is not None and t_stat > crit
                    )
                else:
                    res['t_crit_bonferroni'] = None
                    res['significant_bonferroni'] = False
                all_results.append(res)
                logger.info(f"  layout={res['layout']}  n_params={res['n_params']}  n_layers={res['n_layers']}")
                # ci95_acc is None when only one seed survived (see
                # eval_layout_multi_seed). Formatting None directly raises
                # ``TypeError: unsupported format string passed to NoneType``.
                # Fall back to 'n/a' for the log line; the JSON summary table
                # (line below) already handles None via ``r.get(...) or 0.0``.
                _ci = res['ci95_acc']
                _ci_str = f'{_ci:.4f}' if _ci is not None else 'n/a'
                logger.info(f"  -> mean_acc={res['mean_acc']:.4f} +/- {_ci_str} "
                            f"(std={res['std_acc']:.4f}, t_vs_chance={_fmt_tstat(res['t_stat_vs_chance'], width=0, prec=2)}, "
                            f"sig_bonferroni={res['significant_bonferroni']})")
                logger.info(f"     mean_fwd={res['mean_fwd_ms']:.2f}ms")
            except Exception as e:
                import traceback as _tb
                logger.error(f"  ratio {r[0]}:{r[1]}:{r[2]} FAILED: {e}")
                _tb.print_exc()
                # Error stub MUST include every key that success rows carry
                # so downstream JSON consumers do not KeyError on error rows.
                # Mirrors the fix in run_quality.py::main.
                all_results.append({
                    'ratio': f'{r[0]}:{r[1]}:{r[2]}',
                    'n_kv': n_kv,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    # 'layout' is present on success rows (line 359), so include
                    # it on error rows too for schema consistency. Without it,
                    # ``r['layout']`` KeyError on error rows for strict-schema
                    # consumers (e.g. pandas with explicit dtype, or downstream
                    # figure scripts).
                    'layout': None,
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_fwd_ms': None,
                    'n_params': None,
                    'n_layers': sum(r),
                    'chance_acc': 1.0 / VOCAB,
                    't_stat_vs_chance': None,
                    't_crit_bonferroni': None,
                    'significant_bonferroni': False,
                    'mean_train_time_s': None,
                    'per_seed': [],
                })

    # Summary table (grouped by n_kv). Header is 112 chars wide:
    # 4+3+8+3+22+3+6+3+8+3+10+3+10+3+12+3+8 = 112. Use the same width for rules.
    print('\n' + '=' * 112)
    print(f"{'n_kv':>4} | {'ratio':>8} | {'layout':>22} | {'layers':>6} | {'params':>8} | "
          f"{'mean_acc':>10} | {'+/- CI95':>10} | {'t_vs_chance':>12} | {'fwd_ms':>8}")
    print('-' * 112)
    for r in all_results:
        # Skip error rows in the summary table (they have null fields).
        if 'error' in r:
            print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {'(error)':>22} | {r['n_layers']:>6} | "
                  f"{'-':>8} | {'-':>10} | {'-':>10} | {'-':>12} | {'-':>8}   ERROR: {r['error']}")
            continue
        layout_str = r.get('layout', '') or ''
        # Use explicit ``is not None`` checks (not ``or 0.0``): ``or`` would
        # silently coalesce a legitimate 0.0 (e.g. an all-zero mean_acc)
        # into the fallback, and more importantly would render a None CI
        # (single-seed case) as ``0.0000`` instead of 'n/a' — implying
        # perfect precision when in fact the uncertainty is maximal.
        n_params = r['n_params'] if r.get('n_params') is not None else 0
        mean_acc = r['mean_acc'] if r.get('mean_acc') is not None else 0.0
        ci95 = r['ci95_acc']
        fwd = r['mean_fwd_ms'] if r.get('mean_fwd_ms') is not None else 0.0
        ci_str = f"{ci95:>10.4f}" if ci95 is not None else f"{'n/a':>10}"
        print(f"{r['n_kv']:>4} | {r['ratio']:>8} | {layout_str:>22} | {r['n_layers']:>6} | "
              f"{n_params:>8} | {mean_acc:>10.4f} | {ci_str} | "
              f"{_fmt_tstat(r.get('t_stat_vs_chance'), width=12, prec=2)} | {fwd:>8.2f}")
    print(f"{'':>4} | {'chance':>8} | {'':>22} | {'':>6} | {'':>8} | "
          f"{1.0/VOCAB:>10.4f} | {'':>10} | {'':>12} | {'':>8}")

    # Honest note about depth confound
    logger.info('\nNote on the 4:1:1 anomaly:')
    logger.info('  4:1:1 has 6 layers vs 3:1:1 has 5 layers. At a fixed step budget')
    logger.info('  the deeper model needs more steps to converge, so a low 4:1:1')
    logger.info('  score reflects under-training, not necessarily a worse structure.')
    logger.info('  See the per-seed trajectories in the JSON for convergence evidence.')

    # P4 fix — statistical-validity summary. The issue flagged that the
    # ablation had (a) only 3 seeds, (b) accuracies near chance, and
    # (c) ALL significant_bonferroni=False. We now compute an explicit
    # ``conclusions_valid`` flag that downstream figures/reports can check
    # before drawing structural conclusions, and emit a prominent warning
    # so the limitation is impossible to miss.
    n_any_sig = sum(1 for r in all_results if r.get('significant_bonferroni'))
    min_seeds_ok = min((r.get('n_seeds_ok', 0) for r in all_results
                        if 'error' not in r), default=0)
    # A result is "near chance" if mean_acc < 1.5x the chance level.
    near_chance = [r for r in all_results
                   if 'error' not in r
                   and r.get('mean_acc') is not None
                   and r['mean_acc'] < 1.5 * (1.0 / VOCAB)]
    conclusions_valid = (n_seeds >= 5 and min_seeds_ok >= 5
                         and n_any_sig > 0 and len(near_chance) < len(all_results) // 2)
    logger.info('\n' + '=' * 70)
    logger.info('Statistical validity summary (P4 fix):')
    logger.info(f'  seeds requested: {n_seeds}  (min survived: {min_seeds_ok})')
    logger.info(f'  ratios with significant_bonferroni=True: {n_any_sig}/{len(all_results)}')
    logger.info(f'  ratios near chance (<1.5x): {len(near_chance)}/{len(all_results)}')
    logger.info(f'  conclusions_valid: {conclusions_valid}')
    if not conclusions_valid:
        logger.warning(
            '  WARNING: The ablation results do NOT support strong structural\n'
            '  conclusions. Either the seed count is too low (<5), no layout\n'
            '  reaches Bonferroni significance, or most accuracies are near\n'
            '  chance. Treat the ranking as exploratory, not confirmatory.\n'
            '  To improve power: increase ABL_SEEDS (>=7), increase ABL_STEPS,\n'
            '  or use a simpler task where the signal is stronger.')
    logger.info('=' * 70)
    # Attach the validity flag to every result record so downstream
    # consumers (make_figures, reports) can check it without recomputing.
    for r in all_results:
        r['conclusions_valid'] = conclusions_valid
        r['n_seeds_requested'] = n_seeds

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a divergent seed slipped past
    # the NaN guard and the per_seed filter, Python's default json.dump
    # would emit literal ``NaN``/``Infinity`` tokens, which are INVALID JSON
    # per RFC 8259 and cause strict parsers (js, jq, pandas with
    # ``orient='records'``) to reject the whole file. With allow_nan=False
    # the call raises ValueError instead — surfacing the corruption loudly
    # rather than shipping a broken file. Mirrors run_quality.py::main.
    #
    # CRITICAL: serialize to a STRING first (json.dumps), then write the
    # string to the file. The previous pattern called json.dump directly
    # on the file object inside a try/except — when the first dump raised
    # ValueError mid-write (on encountering a NaN), the file was left
    # with a PARTIAL JSON document. The fallback json.dump then APPENDED
    # to the partial content, producing invalid JSON (two concatenated
    # fragments) that no parser could read. Serializing to a string first
    # guarantees atomicity: either the complete JSON is written or nothing
    # is. Mirrors the fix in run_quality.py::main.
    # P1-5 fix: use the shared atomic JSON writer (temp file + fsync +
    # os.replace) so a process kill or disk-full mid-write leaves the
    # target file as the OLD version (or absent) rather than a truncated
    # partial JSON document. See kaggle_setup.write_json_atomic's docstring
    # for the full rationale.
    try:
        write_json_atomic(all_results, 'results/exp5_ablation.json',
                          indent=2, allow_nan=False)
    except ValueError as e:
        logger.error(f'non-finite value in results; sanitizing to null: {e}')
        write_json_atomic(sanitize_for_json(all_results),
                          'results/exp5_ablation.json',
                          indent=2, allow_nan=False)
    logger.info('\nSaved: results/exp5_ablation.json')
    # P0-2 fix: return non-zero if any layout's training crashed
    # (``'error' in r``), so ``run_all._run`` records the experiment as
    # ``status='fail'`` instead of silently treating a partial run as success.
    # Mirrors the fix in run_quality.py::main and run_benchmark.py::main.
    n_errors = sum(1 for r in all_results if 'error' in r)
    if n_errors:
        logger.error(
            f'\n[P0-2] {n_errors}/{len(all_results)} layouts errored out. '
            f'Returning non-zero so run_all records this experiment as failed.')
        return 1
    return 0


if __name__ == '__main__':
    main()
