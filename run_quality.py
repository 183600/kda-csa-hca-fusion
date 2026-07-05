"""Experiment 4 — synthetic quality proxy (Multi-Query Associative Recall).

MQAR (arXiv:2302.13071) is the canonical associative-recall probe used to
evaluate linear-attention variants: the model must recall one of ``n_kv``
key-value pairs presented earlier in the sequence when cued by its key.
KDA's delta rule + fine-grained gating is designed exactly for this kind of
in-context associative memory; CSA's sparse retrieval should also help; HCA's
heavy compression trades recall granularity for global context.

We train a tiny single-layer classifier on top of each attention operator for
a fixed number of steps and report final accuracy. Absolute accuracy is not
meaningful at this scale, but the *relative ranking* across operators is.

Kaggle / review-driven additions (address reviewer concerns):

  * **Multi-seed runs with confidence intervals.** Every operator is trained
    over ``n_seeds`` (default 5) seeds; we report mean +/- std and a
    one-sample t-test vs the chance baseline. This addresses the concern
    that the original single-seed numbers were within random noise of
    chance.
  * **Softmax baseline convergence.** The softmax-attention baseline is given
    more steps (``softmax_steps``) so it actually converges — the original
    paper's softmax only reached 10.2% (barely above 6.25% chance), making the
    comparison meaningless. A converged softmax is the *right* upper bound.
  * **Device awareness.** Runs on GPU (Kaggle T4) when available, falling back
    to CPU. The T4 makes the CSA per-token Python loop bearable.
  * **Controlled evaluation.** A larger final eval batch (256) and multiple
    eval batches for a more stable accuracy estimate.
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

from kaggle_setup import configure_torch_for_device, get_device, to_device

logger = logging.getLogger(__name__)


def _fmt_tstat(t, width=12, prec=2):
    """Format a t-statistic for display, None-safe.

    The one-sample t-test vs chance is undefined for n == 1 or when the
    sample standard deviation is zero; in those cases ``t_stat`` is None
    and we render ``"n/a"`` instead of a number.
    """
    if t is None:
        return 'n/a'.rjust(width)
    return f'{t:.{prec}f}'.rjust(width)


# Module-level cache for scipy.stats.t.ppf.
#   None  -> not yet probed
#   False -> scipy unavailable, use the fallback table
#   <fn>  -> scipy.stats.t.ppf callable
_T_PP = None


def _t_crit_975(n):
    """Two-sided 95% critical value of the t-distribution with ``n-1`` dof.

    Returns ``t`` such that ``P(-t < T < t) = 0.95`` where ``T`` follows a
    Student's t distribution with ``n - 1`` degrees of freedom (i.e. the
    one-sided 97.5% quantile, ``scipy.stats.t.ppf(0.975, n - 1)``).

    Resolution order:
      1. ``scipy.stats.t.ppf`` when scipy is importable (exact for any n).
      2. A hardcoded table covering ``n = 2..30`` (df = 1..29). This fixes
         the n = 11..30 range where the previous table fell back to the
         normal approximation 1.96 and lost up to ~8% accuracy.
      3. For ``n > 30`` the normal approximation ``1.96`` (error < 1%).
      4. For ``n < 2`` the CI is undefined; returns ``0.0``.

    The scipy availability check runs once and is cached in ``_T_PP`` so
    repeated calls are essentially free.
    """
    global _T_PP
    if n < 2:
        return 0.0
    if _T_PP is None:
        try:
            from scipy.stats import t as _t_dist
            _T_PP = _t_dist.ppf
        except ImportError:
            _T_PP = False
    if _T_PP:
        return _T_PP(0.975, n - 1)
    # Hardcoded two-sided 95% critical values, n = 2..30 (df = 1..29).
    _TABLE = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
        22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    }
    return _TABLE.get(n, 1.96)


def make_mqar_batch(batch: int, seq_len: int, n_kv: int, vocab: int,
                    embed: nn.Embedding, device=None):
    """Build an MQAR batch with *learnable* embeddings.

    Layout per sequence (length ``seq_len``):
      positions [0, 2*n_kv): alternating (key, value) tokens
      positions [2*n_kv, seq_len-1): random noise tokens
      position  seq_len-1: a cue key (repeat of one of the keys)
    Target = the value that was paired with the cued key.

    Fully vectorized: all ``batch`` sequences are constructed in parallel
    via a single argsort-based per-row permutation and advanced indexing,
    avoiding the per-sample Python loop and ``.item()`` calls that broke
    async kernel launches.
    """
    if device is None:
        device = embed.weight.device

    # Guard against silent shape corruption: if 2*n_kv exceeds vocab, the
    # argsort slice returns fewer than 2*n_kv ids and keys/vals end up with
    # mismatched shapes ([batch, vocab//2] vs the expected [batch, n_kv]).
    # If 2*n_kv >= seq_len, the cue at position seq_len-1 would overwrite a
    # KV pair (or fall inside the KV region), making the task trivially
    # solvable or unsolvable. Both used to fail silently.
    assert 2 * n_kv <= vocab, (
        f"2*n_kv={2*n_kv} must be <= vocab={vocab} (need 2*n_kv distinct ids)")
    assert 2 * n_kv < seq_len, (
        f"2*n_kv={2*n_kv} must be < seq_len={seq_len} (need room for cue token)")

    # Random noise base (covers noise positions; KV positions overwritten below).
    x = torch.randint(0, vocab, (batch, seq_len), device=device)
    cue_pos = seq_len - 1

    # One uniformly-random permutation of [0, vocab) per sequence, obtained
    # by argsort of iid uniform keys (ties have measure zero in float32).
    # Slicing the first 2*n_kv columns yields 2*n_kv distinct ids per row.
    pair_ids = torch.rand(batch, vocab, device=device).argsort(dim=-1)[:, :2 * n_kv]
    keys = pair_ids[:, 0::2]   # [batch, n_kv]
    vals = pair_ids[:, 1::2]   # [batch, n_kv]

    # Place (key, value) pairs at positions [0, 2*n_kv) via broadcast
    # advanced indexing: x[b_idx, even_pos] has shape [batch, n_kv].
    b_idx = torch.arange(batch, device=device).unsqueeze(-1)          # [batch, 1]
    even_pos = torch.arange(0, 2 * n_kv, 2, device=device)            # [n_kv]
    odd_pos = torch.arange(1, 2 * n_kv, 2, device=device)             # [n_kv]
    x[b_idx, even_pos] = keys
    x[b_idx, odd_pos] = vals

    # Vectorized cue selection: pick one key index per sequence in parallel.
    j = torch.randint(0, n_kv, (batch,), device=device)               # [batch]
    row = torch.arange(batch, device=device)                          # [batch]
    x[:, cue_pos] = keys[row, j]
    target = vals[row, j]

    # Apply embedding (gradients flow to the embedding table).
    return embed(x), target, cue_pos


class MQARHead(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor, cue_pos: int):
        return self.fc(self.norm(x[:, cue_pos]))


class ResidualAttnLayer(nn.Module):
    """Generic residual attention layer: x = x + Attn(LN(x))."""

    def __init__(self, attn: nn.Module, d_model: int):
        super().__init__()
        self.attn = attn
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.attn(self.norm(x))


class SoftmaxAttn(nn.Module):
    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V
        self.scale = K ** -0.5

    def forward(self, x):
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.H, self.K)
        k = self.k(x).view(B, T, self.H, self.K)
        v = self.v(x).view(B, T, self.H, self.V)
        s = torch.einsum('bthk,bshk->bhts', q, k) * self.scale
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        s = s.masked_fill(mask, float('-inf'))
        p = torch.softmax(s, dim=-1)
        out = torch.einsum('bhts,bshv->bthv', p, v)
        return self.o(out.reshape(B, T, self.H * self.V))


class KDAAttn(nn.Module):
    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        self.g = nn.Linear(d_model, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        self.H, self.K, self.V = H, K, V

    def forward(self, x):
        from ops_kda import naive_recurrent_kda
        B, T, d = x.shape
        q = F.normalize(F.silu(self.q(x)), dim=-1).view(B, T, self.H, self.K)
        k = F.normalize(F.silu(self.k(x)), dim=-1).view(B, T, self.H, self.K)
        v = F.silu(self.v(x)).view(B, T, self.H, self.V)
        g = -F.softplus(self.g(x)).view(B, T, self.H, self.K) * 0.1
        beta = torch.sigmoid(self.beta(x))
        out, _ = naive_recurrent_kda(q, k, v, g, beta, output_final_state=False)
        return self.o(out.reshape(B, T, self.H * self.V))


class CSAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        c, dc = 32, 64
        m, nh, nIh, cI, topk = 4, 2, 2, 16, 4
        self.m, self.topk, self.nh, self.nIh, self.c, self.cI, self.dc = \
            m, topk, nh, nIh, c, cI, dc
        self.W_aKV = nn.Linear(d_model, c, bias=False)
        self.W_bKV = nn.Linear(d_model, c, bias=False)
        self.W_aZ = nn.Linear(d_model, c, bias=False)
        self.W_bZ = nn.Linear(d_model, c, bias=False)
        self.Ba = nn.Parameter(torch.randn(m, c) * 0.02)
        self.Bb = nn.Parameter(torch.randn(m, c) * 0.02)
        self.W_DQ = nn.Linear(d_model, dc, bias=False)
        self.W_UQ = nn.Linear(dc, c * nh, bias=False)
        self.W_IUQ = nn.Linear(dc, cI * nIh, bias=False)
        self.W_w = nn.Linear(d_model, nIh, bias=False)
        self.W_KV_idx = nn.Linear(d_model, cI, bias=False)
        self.W_Z_idx = nn.Linear(d_model, cI, bias=False)
        self.B_idx = nn.Parameter(torch.randn(m, cI) * 0.02)
        self.sink = nn.Parameter(torch.zeros(nh))
        self.o = nn.Linear(c * nh, d_model, bias=False)

    def forward(self, x):
        from ops_csa import naive_csa
        T = x.shape[1]
        pad = (-T) % self.m
        if pad:
            x = F.pad(x, (0, 0, pad, 0))
        o = naive_csa(
            x, self.W_aKV.weight.T, self.W_bKV.weight.T,
            self.W_aZ.weight.T, self.W_bZ.weight.T, self.Ba, self.Bb,
            self.W_DQ.weight.T, self.W_UQ.weight.T, self.W_IUQ.weight.T,
            self.W_w.weight.T, self.W_KV_idx.weight.T, self.W_Z_idx.weight.T,
            self.B_idx,
            m=self.m, topk=self.topk, nh=self.nh, nIh=self.nIh,
            c=self.c, c_I=self.cI, dc=self.dc,
            sliding_window=4, sink_logits=self.sink,
        )
        if pad:
            # Trim the padded prefix off the SEQUENCE axis (dim=1).
            # `o[pad:]` slices dim=0 (batch) which both crashes for B<=pad
            # and silently corrupts results for B>pad.
            o = o[:, pad:]
        return self.o(o)


class HCAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        c, dc = 32, 64
        m2, nh = 4, 2
        self.m2, self.nh, self.c, self.dc = m2, nh, c, dc
        self.W_KV = nn.Linear(d_model, c, bias=False)
        self.W_Z = nn.Linear(d_model, c, bias=False)
        self.B_pos = nn.Parameter(torch.randn(m2, c) * 0.02)
        self.W_DQ = nn.Linear(d_model, dc, bias=False)
        self.W_UQ = nn.Linear(dc, c * nh, bias=False)
        self.sink = nn.Parameter(torch.zeros(nh))
        self.o = nn.Linear(c * nh, d_model, bias=False)

    def forward(self, x):
        from ops_hca import naive_hca
        T = x.shape[1]
        pad = (-T) % self.m2
        if pad:
            x = F.pad(x, (0, 0, pad, 0))
        o = naive_hca(x, self.W_KV.weight.T, self.W_Z.weight.T, self.B_pos,
                      self.W_DQ.weight.T, self.W_UQ.weight.T,
                      m2=self.m2, nh=self.nh, c=self.c, dc=self.dc,
                      sliding_window=4, sink_logits=self.sink)
        if pad:
            # Trim the padded prefix off the SEQUENCE axis (dim=1).
            # `o[pad:]` slices dim=0 (batch) which both crashes for B<=pad
            # and silently corrupts results for B>pad.
            o = o[:, pad:]
        return self.o(o)


def _eval_model(layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=4, batch=64):
    """Evaluate accuracy over multiple fresh batches for a stable estimate."""
    layer.eval()
    head.eval()
    correct, total = 0, 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x_emb, target, cue_pos = make_mqar_batch(batch, seq_len, n_kv, vocab, embed, device)
            h = layer(x_emb)
            logits = head(h, cue_pos)
            correct += (logits.argmax(-1) == target).sum().item()
            total += target.numel()
            losses.append(F.cross_entropy(logits, target).item())
    return correct / total, sum(losses) / len(losses)


def train_one(op_name, d_model=32, seq_len=16, n_kv=1, vocab=16,
              steps=100, lr=3e-3, seed=42, device='cpu',
              softmax_steps=None, eval_batches=4, eval_batch=64):
    """Train a single operator on MQAR for ``steps`` steps.

    ``softmax_steps`` lets the softmax baseline train longer to reach
    convergence (the original 100 steps left softmax at ~10%, barely above the
    6.25% chance — a meaningless upper bound).
    """
    if softmax_steps is None:
        softmax_steps = steps
    actual_steps = softmax_steps if op_name == 'softmax' else steps

    torch.manual_seed(seed)
    factories = {
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'csa':     lambda: CSAAttn(d_model),
        'hca':     lambda: HCAAttn(d_model),
    }
    embed = nn.Embedding(vocab, d_model).to(device)
    layer = ResidualAttnLayer(factories[op_name](), d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    params = list(embed.parameters()) + list(layer.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    losses, accs = [], []
    train_batch = 32
    for step in range(actual_steps):
        x_emb, target, cue_pos = make_mqar_batch(train_batch, seq_len, n_kv, vocab, embed, device)
        layer.train()
        head.train()
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(-1) == target).float().mean().item()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        accs.append(acc)

    # Final eval on multiple fresh batches for a stable accuracy estimate.
    final_acc, final_loss = _eval_model(
        layer, head, embed, seq_len, n_kv, vocab, device,
        n_batches=eval_batches, batch=eval_batch,
    )
    chance = 1.0 / vocab
    return {
        'op': op_name,
        'n_kv': n_kv,
        'final_acc': final_acc,
        'final_loss': final_loss,
        'chance_acc': chance,
        'last_train_acc': accs[-1],
        'mean_last10_loss': sum(losses[-10:]) / min(10, len(losses)),
        'mean_last10_acc': sum(accs[-10:]) / min(10, len(accs)),
        'steps': actual_steps,
        'seed': seed,
    }


def train_multi_seed(op_name, n_seeds=5, steps=100, softmax_steps=300,
                     device='cpu', **kw):
    """Train ``op_name`` over ``n_seeds`` seeds.

    Returns a dict with per-seed results plus aggregate stats (mean, std,
    95% CI half-width via t-distribution).
    """
    seeds = [42 + i for i in range(n_seeds)]
    per_seed = []
    for s in seeds:
        t0 = time.time()
        r = train_one(op_name, seed=s, steps=steps, device=device,
                      softmax_steps=softmax_steps, **kw)
        r['train_time_s'] = time.time() - t0
        per_seed.append(r)
        logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                    f"steps={r['steps']}  time={r['train_time_s']:.1f}s")

    accs = [r['final_acc'] for r in per_seed]
    losses = [r['final_loss'] for r in per_seed]
    n = len(accs)
    mean_acc = sum(accs) / n
    mean_loss = sum(losses) / n
    if n > 1:
        var_acc = sum((a - mean_acc) ** 2 for a in accs) / (n - 1)
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / (n - 1)
        std_acc = math.sqrt(var_acc)
        std_loss = math.sqrt(var_loss)
        # 95% CI half-width using t-distribution with n-1 dof.
        t = _t_crit_975(n)
        ci_acc = t * std_acc / math.sqrt(n)
        ci_loss = t * std_loss / math.sqrt(n)
    else:
        std_acc = std_loss = 0.0
        ci_acc = ci_loss = 0.0

    # One-sample t-test vs chance: tests whether mean_acc differs from the
    # chance level. The t-statistic is only defined when n > 1 and the
    # sample standard deviation is strictly positive; otherwise we return
    # None (the test is not computable, not "infinitely significant").
    chance = per_seed[0]['chance_acc']
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    return {
        'op': op_name,
        'n_kv': per_seed[0]['n_kv'],
        'n_seeds': n,
        'seeds': seeds,
        'per_seed': per_seed,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'ci95_acc': ci_acc,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'ci95_loss': ci_loss,
        'chance_acc': chance,
        't_stat_vs_chance': t_stat,
        'mean_train_time_s': sum(r['train_time_s'] for r in per_seed) / n,
    }


def _parse_nkv_list(env_var, default='1'):
    """Parse a comma-separated list of n_kv values from an env var.

    ``MQAR_NKV=1``       -> [1]           (backward compatible)
    ``MQAR_NKV=1,2,4``   -> [1, 2, 4]
    """
    raw = os.environ.get(env_var, default)
    try:
        vals = [int(x.strip()) for x in raw.split(',') if x.strip()]
    except ValueError:
        raise ValueError(
            f'{env_var}={raw!r} must be comma-separated ints, e.g. "1,2,4"')
    if not vals:
        vals = [int(default)]
    return vals


def main():
    info = configure_torch_for_device()
    device = info.device
    logger.info('=' * 70)
    logger.info('Experiment 4: MQAR Synthetic Quality Probe (multi-seed)')
    logger.info('=' * 70)
    logger.info(f'  device        : {device}')
    n_kv_list = _parse_nkv_list('MQAR_NKV', '1')
    logger.info(f'  vocab=16, seq_len=16, n_kv={n_kv_list}')
    logger.info(f'  chance accuracy = {1/16:.4f} (independent of n_kv; target is 1-of-vocab)')
    n_seeds = int(os.environ.get('MQAR_SEEDS', '5'))
    steps = int(os.environ.get('MQAR_STEPS', '200'))
    # Softmax gets more steps to actually converge (original paper's 100 left
    # it at ~10%, barely above 6.25% chance — a useless upper bound).
    softmax_steps = int(os.environ.get('MQAR_SOFTMAX_STEPS', '500'))
    logger.info(f'  n_seeds       : {n_seeds}')
    logger.info(f'  steps         : {steps}  (softmax: {softmax_steps})')

    all_results = []
    for n_kv in n_kv_list:
        logger.info(f'\n{"=" * 70}')
        logger.info(f'  n_kv = {n_kv}   (harder: {n_kv} KV pairs to disambiguate)')
        logger.info(f'{"=" * 70}')
        for op in ['softmax', 'kda', 'csa', 'hca']:
            logger.info(f'\nTraining {op} (n_kv={n_kv}, {n_seeds} seeds)...')
            r = train_multi_seed(op, n_seeds=n_seeds, steps=steps,
                                 softmax_steps=softmax_steps, device=device,
                                 n_kv=n_kv)
            all_results.append(r)
            logger.info(f"  -> mean_acc={r['mean_acc']:.4f} +/- {r['ci95_acc']:.4f} "
                        f"(std={r['std_acc']:.4f}, t_vs_chance={_fmt_tstat(r['t_stat_vs_chance'], width=0, prec=2)})")

    # Summary table (grouped by n_kv)
    print('\n' + '=' * 80)
    print(f"{'n_kv':>4} | {'op':>10} | {'mean_acc':>10} | {'+/- CI95':>10} | "
          f"{'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 80)
    for r in all_results:
        print(f"{r['n_kv']:>4} | {r['op']:>10} | {r['mean_acc']:>10.4f} | "
              f"{r['ci95_acc']:>10.4f} | {r['std_acc']:>8.4f} | "
              f"{_fmt_tstat(r['t_stat_vs_chance'], width=12, prec=2)} | {r['mean_loss']:>10.4f}")
    print(f"{'':>4} | {'chance':>10} | {1/16:>10.4f} |")

    os.makedirs('results', exist_ok=True)
    with open('results/exp4_mqar.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info('\nSaved: results/exp4_mqar.json')


if __name__ == '__main__':
    main()
