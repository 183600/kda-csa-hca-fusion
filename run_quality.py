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

from kaggle_setup import configure_torch_for_device
from ops_kda import naive_recurrent_kda
from ops_csa import naive_csa
from ops_hca import naive_hca

logger = logging.getLogger(__name__)


def _build_param_groups(*modules, weight_decay=0.01):
    """Build AdamW parameter groups with proper weight-decay exclusion.

    Standard ML practice: embeddings, biases, and LayerNorm parameters should
    NOT be weight-decayed. Weight decay on these can hurt training quality
    (e.g. shrinking the embedding table towards zero degrades representation
    quality; decaying LayerNorm affine params breaks normalization statistics).

    Grouping rules:
      * Parameters from ``nn.Embedding`` modules -> no decay
      * Parameters from ``nn.LayerNorm`` modules -> no decay
      * 1-D parameters (biases, e.g. ``nn.Conv1d.bias``, ``nn.Linear.bias``)
        -> no decay
      * All other parameters (e.g. ``nn.Linear.weight``, ``nn.Conv1d.weight``,
        ``nn.Parameter`` tensors of ndim >= 2) -> decay

    Note: we check the MODULE TYPE (not the parameter name) to identify
    embeddings and LayerNorms, because ``named_parameters()`` on a submodule
    returns names RELATIVE to that submodule (e.g. just ``weight``, not
    ``embed.weight``). A name-based heuristic would miss the embedding table.

    Returns a list of param groups suitable for ``torch.optim.AdamW``.
    """
    no_decay = []
    decay = []
    for module in modules:
        # Collect the ids of parameters that belong to Embedding / LayerNorm
        # submodules. We walk the module tree once so that nested modules
        # (e.g. ``layer.norm`` inside ``ResidualAttnLayer``) are correctly
        # identified by their module type, not by a fragile name match.
        no_decay_ids = set()
        for submod in module.modules():
            if isinstance(submod, (nn.Embedding, nn.LayerNorm)):
                for p in submod.parameters(recurse=False):
                    no_decay_ids.add(id(p))
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # Exclude: Embedding/LayerNorm params (by module type), 1-D params
            # (biases), and any nn.Parameter with ndim < 2 (e.g. the CSA/HCA
            # positional biases Ba/Bb/B_idx/B_pos and the sink logits).
            is_no_decay = (
                id(p) in no_decay_ids
                or p.ndim <= 1
            )
            if is_no_decay:
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


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
      3. For ``n > 30`` the normal approximation ``1.96`` (relative error
         ~4% at n=31, drops below 1% only around n≈100, and below 0.1% past
         n≈400 — so callers using small samples should ensure scipy is
         available or extend the table).
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
                    embed: nn.Embedding, device=None, generator=None):
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

    ``generator`` (optional ``torch.Generator``) makes the batch generation
    reproducible and INDEPENDENT of the global RNG. This matters for
    multi-operator comparisons: each operator has a different parameter
    count, so it consumes a different number of global-RNG draws during
    init. Without a dedicated generator, the *same* seed produced
    *different* training batches across operators — a silent confound in
    the multi-seed CI. Pass a per-seed generator to guarantee that, for a
    given seed, all operators see the same training data.
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
    x = torch.randint(0, vocab, (batch, seq_len), device=device, generator=generator)
    cue_pos = seq_len - 1

    # One uniformly-random permutation of [0, vocab) per sequence, obtained
    # by argsort of iid uniform keys (ties have measure zero in float32).
    # Slicing the first 2*n_kv columns yields 2*n_kv distinct ids per row.
    pair_ids = torch.rand(batch, vocab, device=device, generator=generator).argsort(dim=-1)[:, :2 * n_kv]
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
    j = torch.randint(0, n_kv, (batch,), device=device, generator=generator)  # [batch]
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
        # Lazily-built causal mask cache. T is not known at __init__ (it
        # depends on the input), so we build the mask on the first forward
        # and cache it. Subsequent forwards with the same T reuse the cached
        # mask instead of allocating a new [T, T] tensor every call. The
        # cache is keyed by (T, device) to handle seq-len or device changes.
        # Registered as a non-persistent buffer so .to(device) moves it and
        # state_dict skips it (it is derived, not learned).
        self.register_buffer('_causal_mask', None, persistent=False)
        self._mask_key = None  # (T, device) the cached mask was built for

    def _get_causal_mask(self, T, device):
        """Return the cached [T, T] strictly-upper-triangular bool mask,
        rebuilding it only when T or device changes."""
        key = (T, str(device))
        if self._mask_key != key or self._causal_mask is None:
            self._causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
            )
            self._mask_key = key
        return self._causal_mask

    def forward(self, x):
        B, T, d = x.shape
        q = self.q(x).view(B, T, self.H, self.K)
        k = self.k(x).view(B, T, self.H, self.K)
        v = self.v(x).view(B, T, self.H, self.V)
        s = torch.einsum('bthk,bshk->bhts', q, k) * self.scale
        mask = self._get_causal_mask(T, x.device)
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
        # ``naive_csa`` handles non-divisible T via internal right-padding
        # and trims its output back to the original T, so we no longer need
        # to pad/trim here. The external padding was originally added to fix
        # a LEFT-padding bug, but that fix now lives inside ``naive_csa``
        # itself (see ``test_csa_hca_non_divisible_T``). Removing the
        # redundant padding keeps this wrapper thin and avoids doing the
        # pad/trim twice.
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
        return self.o(o)


class HCAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        c, dc = 32, 64
        # HCA's defining feature is *heavy* compression: m2 should be >> m so
        # the HCA branch produces far fewer compressed blocks than CSA, trading
        # recall granularity for global context. The previous value m2=4 was
        # equal to CSA's m=4 (see CSAAttn above), which made HCA behave
        # identically to CSA-without-indexer and silently defeated the purpose
        # of including HCA in the MQAR comparison. With seq_len=16 and CSA
        # m=4 (n_blocks_CSA=4), setting m2=8 gives n_blocks_HCA=2, exercising
        # the heavier-compression regime. Mirrors the rationale already
        # documented in run_ablation.py::_make_cfg.
        m2, nh = 8, 2
        self.m2, self.nh, self.c, self.dc = m2, nh, c, dc
        self.W_KV = nn.Linear(d_model, c, bias=False)
        self.W_Z = nn.Linear(d_model, c, bias=False)
        self.B_pos = nn.Parameter(torch.randn(m2, c) * 0.02)
        self.W_DQ = nn.Linear(d_model, dc, bias=False)
        self.W_UQ = nn.Linear(dc, c * nh, bias=False)
        self.sink = nn.Parameter(torch.zeros(nh))
        self.o = nn.Linear(c * nh, d_model, bias=False)

    def forward(self, x):
        # ``naive_hca`` handles non-divisible T via internal right-padding
        # and trims its output back to the original T, so we no longer need
        # to pad/trim here. See the comment in ``CSAAttn.forward`` for the
        # full rationale.
        o = naive_hca(x, self.W_KV.weight.T, self.W_Z.weight.T, self.B_pos,
                      self.W_DQ.weight.T, self.W_UQ.weight.T,
                      m2=self.m2, nh=self.nh, c=self.c, dc=self.dc,
                      sliding_window=4, sink_logits=self.sink)
        return self.o(o)


def _eval_model(layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=4, batch=64):
    """Evaluate accuracy over multiple fresh batches for a stable estimate.

    Uses a dedicated ``torch.Generator`` for batch generation so the eval
    batches are reproducible and independent of any global-RNG state left
    over by training. This also makes the eval pass deterministic across
    re-runs, which is useful for debugging.

    Restores each module's train/eval mode after evaluation so a future
    caller that evaluates mid-training does not silently resume training
    in eval mode (no dropout, BN using running stats).
    """
    # Save the prior train/eval state so we can restore it after eval —
    # a latent footgun if a caller ever invokes _eval_model mid-training.
    was_training = {m: m.training for m in (layer, head, embed)}
    try:
        layer.eval()
        head.eval()
        embed.eval()  # nn.Embedding has no dropout/batchnorm so this is a no-op,
                      # but we set it for symmetry with layer/head so future
                      # additions (e.g. embedding dropout) do not silently stay
                      # in train mode during evaluation.
        # Fixed seed for the eval generator so every operator sees the SAME eval
        # batches (apples-to-apples comparison at eval time too, not just train).
        eval_gen = torch.Generator(device=device)
        eval_gen.manual_seed(12345)
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_mqar_batch(
                    batch, seq_len, n_kv, vocab, embed, device, generator=eval_gen)
                h = layer(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
        # Guard against n_batches=0 (or batch=0): without this the function
        # raises ZeroDivisionError on ``correct / total`` and
        # ``sum([]) / len([])``. Returns 0.0 for both metrics so the caller
        # gets a finite (if meaningless) value rather than a crash. Mirrors
        # the steps=0 guard in train_one.
        if total == 0 or not losses:
            return 0.0, 0.0
        return correct / total, sum(losses) / len(losses)
    finally:
        for m, was in was_training.items():
            m.train(was)


def train_one(op_name, d_model=32, seq_len=16, n_kv=1, vocab=16,
              steps=100, lr=3e-3, seed=42, device='cpu',
              softmax_steps=None, eval_batches=4, eval_batch=64,
              train_batch=None):
    """Train a single operator on MQAR for ``steps`` steps.

    ``softmax_steps`` lets the softmax baseline train longer to reach
    convergence (the original 100 steps left softmax at ~10%, barely above the
    6.25% chance — a meaningless upper bound).

    ``train_batch`` overrides the per-step training batch size (default 32,
    previously a hardcoded local magic number — now overridable for
    memory-constrained or GPU runs via the ``MQAR_TRAIN_BATCH`` env var).

    RNG isolation: the model is initialized with ``torch.manual_seed(seed)``,
    but a SEPARATE per-step generator (also seeded with ``seed``) drives
    ``make_mqar_batch``. This is critical for the multi-seed comparison: the
    different operators have different parameter counts, so they consume a
    different number of RNG draws during init. Without a separate generator
    for batch generation, the *same* seed produced *different* training
    batches across operators, undermining the apples-to-apples comparison
    that multi-seed CI is meant to strengthen.

    ``device`` may be passed as a string ('cpu'/'cuda') or a
    ``torch.device``; string inputs are coerced to ``torch.device`` so
    callers from notebooks don't hit ``AttributeError`` on ``device.type``.
    """
    # Coerce string device (e.g. 'cpu', 'cuda') to torch.device so callers
    # passing a string from a notebook don't hit AttributeError on
    # ``device.type`` inside this function. main() already passes a real
    # torch.device, but train_one is a public function and the coercion is
    # cheap and idempotent.
    if isinstance(device, str):
        device = torch.device(device)
    if softmax_steps is None:
        softmax_steps = steps
    if train_batch is None:
        train_batch = int(os.environ.get('MQAR_TRAIN_BATCH', '32'))
    actual_steps = softmax_steps if op_name == 'softmax' else steps

    torch.manual_seed(seed)
    factories = {
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'csa':     lambda: CSAAttn(d_model),
        'hca':     lambda: HCAAttn(d_model),
    }
    # Create embed and head BEFORE the operator-specific layer so their
    # initial weights are IDENTICAL across operators for a given seed.
    # Different operators have different parameter counts, so creating the
    # layer first would consume a different number of RNG draws and desync
    # the downstream head init — a silent confound in the multi-seed CI.
    # (The previous order was embed -> layer -> head, which left the head's
    # init operator-dependent.)
    embed = nn.Embedding(vocab, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    layer = ResidualAttnLayer(factories[op_name](), d_model).to(device)
    # Build parameter groups with proper weight-decay exclusion: embeddings,
    # biases, and LayerNorm parameters are NOT weight-decayed (standard ML
    # practice — decaying these can hurt training quality). The previous
    # version applied weight decay uniformly to all parameters.
    param_groups = _build_param_groups(embed, layer, head, weight_decay=0.01)
    opt = torch.optim.AdamW(param_groups, lr=lr)
    params = [p for g in param_groups for p in g['params']]

    # Separate generator for batch generation so the per-step batches are
    # IDENTICAL across operators for a given seed (the model init consumed a
    # different number of RNG draws per operator, which would otherwise
    # desync the global RNG and produce different training data per op).
    batch_gen = torch.Generator(device=device)
    batch_gen.manual_seed(seed + 1)  # offset so it does not collide with
                                      # the seed used for model init.

    losses, accs = [], []
    # Set train mode ONCE before the loop, not per step (the previous code
    # called layer.train()/head.train() inside the loop, which is a redundant
    # O(steps) no-op once the modules are already in train mode).
    layer.train()
    head.train()
    embed.train()
    for step in range(actual_steps):
        x_emb, target, cue_pos = make_mqar_batch(
            train_batch, seq_len, n_kv, vocab, embed, device, generator=batch_gen)
        h = layer(x_emb)
        logits = head(h, cue_pos)
        loss = F.cross_entropy(logits, target)
        # NaN/Inf guard: if a single step diverges (e.g. AdamW + lr=3e-3
        # instability, or a single Inf in KDA state from a bad beta/g
        # combination), backward() would propagate NaN into ALL parameter
        # grads, opt.step() would write NaN into ALL parameters, and the
        # rest of training would silently produce NaN logits — the loss
        # curve would look like [..., 2.4, 2.5, NaN, NaN, NaN, ...] with
        # no error raised. Worse, the final eval would still return a
        # *finite* accuracy (argmax on NaN is undefined but deterministic,
        # often returning 0), so the divergent seed would NOT be caught by
        # the per-seed try/except in train_multi_seed and would silently
        # corrupt the aggregate mean/CI. Raise here so the per-seed
        # try/except catches it as a failed seed.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()} "
                f"(op={op_name}, seed={seed}); aborting this seed to "
                f"prevent silent NaN propagation into aggregate stats")
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
    # Guard against steps=0 (would crash on accs[-1] / sum([])/0 below).
    # Returns a stub with the eval result and zeros for the training
    # trajectory fields. ``actual_steps`` is reported as 0 so downstream
    # consumers can detect the degenerate case.
    if not losses:
        return {
            'op': op_name,
            'n_kv': n_kv,
            'final_acc': final_acc,
            'final_loss': final_loss,
            'chance_acc': chance,
            'last_train_acc': 0.0,
            'mean_last10_loss': 0.0,
            'mean_last10_acc': 0.0,
            'steps': actual_steps,
            'seed': seed,
            'train_batch': train_batch,
        }
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
        'train_batch': train_batch,
    }


def train_multi_seed(op_name, n_seeds=5, steps=100, softmax_steps=500,
                     device='cpu', **kw):
    """Train ``op_name`` over ``n_seeds`` seeds.

    Returns a dict with per-seed results plus aggregate stats (mean, std,
    95% CI half-width via t-distribution).

    Per-seed error handling: a single divergent seed (NaN loss, OOM, etc.)
    is caught and recorded as a stub entry in ``per_seed`` rather than
    crashing the whole operator's multi-seed run. Aggregate stats are
    computed over the surviving (successful) seeds. If ALL seeds fail, the
    function raises a ``RuntimeError`` so the caller can record a stub
    result for the operator (mirrors ``run_ablation.py::eval_layout_multi_seed``).
    Previously, a single seed failure crashed the entire experiment, losing
    all other operators' results — inconsistent with the ablation runner's
    more robust pattern.

    ``device`` may be passed as a string or a ``torch.device``; string
    inputs are coerced for notebook callers.
    """
    # Coerce string device -> torch.device so callers passing 'cpu'/'cuda'
    # from a notebook don't hit AttributeError on ``device.type`` below.
    if isinstance(device, str):
        device = torch.device(device)
    seeds = [42 + i for i in range(n_seeds)]
    per_seed = []
    for s in seeds:
        t0 = time.time()
        # Per-seed try/except: one divergent seed should not crash the
        # whole operator. We log and record a stub; the aggregate stats
        # are computed over whichever seeds succeeded.
        try:
            r = train_one(op_name, seed=s, steps=steps, device=device,
                          softmax_steps=softmax_steps, **kw)
            r['train_time_s'] = time.time() - t0
            per_seed.append(r)
            logger.info(f"    seed {s}: acc={r['final_acc']:.4f}  loss={r['final_loss']:.4f}  "
                        f"steps={r['steps']}  time={r['train_time_s']:.1f}s")
        except Exception as e:
            logger.warning(f"    seed {s} FAILED: {e}")
            per_seed.append({
                'seed': s, 'error': str(e),
                'final_acc': None, 'final_loss': None,
            })
        # On GPU, clear the CUDA cache between seeds so the allocator does
        # not accumulate freed-but-unreleased blocks across seeds.
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Filter out failed seeds before computing aggregate stats.
    # A seed that diverged to NaN loss is caught by train_one's NaN guard
    # and recorded with an 'error' key — but a seed that produced a NaN
    # final_acc/final_loss via a path the guard did not cover (e.g. eval
    # on NaN params from a step that crashed before the guard fired) would
    # survive the 'error' filter and silently corrupt the aggregate mean
    # (NaN propagates through sum()/std()). Defensively reject any seed
    # whose final_acc or final_loss is None or non-finite.
    ok_per_seed = [
        r for r in per_seed
        if 'error' not in r
        and r.get('final_acc') is not None
        and math.isfinite(r['final_acc'])
        and r.get('final_loss') is not None
        and math.isfinite(r['final_loss'])
    ]
    if not ok_per_seed:
        # All seeds failed: propagate the error so the caller can record
        # a stub result for this operator.
        raise RuntimeError(
            f"all {n_seeds} seeds failed for op '{op_name}'; "
            f"first error: {per_seed[0].get('error')}")

    accs = [r['final_acc'] for r in ok_per_seed]
    losses = [r['final_loss'] for r in ok_per_seed]
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
    chance = ok_per_seed[0]['chance_acc']
    if n > 1 and std_acc > 0:
        t_stat = (mean_acc - chance) / (std_acc / math.sqrt(n))
    else:
        t_stat = None

    return {
        'op': op_name,
        'n_kv': ok_per_seed[0]['n_kv'],
        # Report the REQUESTED seed count (len(per_seed)), not the surviving
        # count (n), so the figure title and downstream consumers reflect
        # how many seeds were actually run. ``n_seeds_ok`` / ``n_seeds_failed``
        # carry the surviving / failed breakdown. Mirrors run_ablation.py's
        # convention (which already used len(per_seed) here).
        'n_seeds': len(per_seed),
        'n_seeds_ok': n,
        'n_seeds_failed': len(per_seed) - n,
        'n_seeds_total': len(per_seed),
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
        'mean_train_time_s': sum(r.get('train_time_s', 0.0)
                                 for r in ok_per_seed) / n,
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
    # These are the task hyperparameters. Lifted to named constants so the
    # chance-accuracy log line below uses the SAME value as the per-seed
    # ``chance = 1.0 / vocab`` computation in ``train_one`` (the previous
    # code hardcoded 1/16 in three print/log lines, which would have silently
    # lied if vocab were ever changed).
    vocab = 16
    seq_len = 16
    chance = 1.0 / vocab
    n_kv_list = _parse_nkv_list('MQAR_NKV', '1')
    # Pre-validate n_kv against vocab and seq_len so the user gets a clear
    # error message instead of an opaque AssertionError from deep inside
    # ``make_mqar_batch`` during training.
    for n_kv in n_kv_list:
        if 2 * n_kv > vocab:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} exceeds "
                f"vocab={vocab}; reduce n_kv or increase vocab.")
        if 2 * n_kv >= seq_len:
            raise ValueError(
                f"MQAR_NKV includes n_kv={n_kv} but 2*n_kv={2*n_kv} must be "
                f"< seq_len={seq_len} (need room for the cue token at the end).")
    logger.info(f'  vocab={vocab}, seq_len={seq_len}, n_kv={n_kv_list}')
    logger.info(f'  chance accuracy = {chance:.4f} (independent of n_kv; target is 1-of-vocab)')
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
            # Per-operator try/except so ONE failing operator (all seeds
            # diverged / OOMed) does not crash the whole experiment and
            # lose the other operators' results. The error is logged and
            # recorded as a stub result so the JSON file is always written
            # and downstream figure generation can skip the missing op
            # gracefully (mirrors run_ablation.py's per-ratio try/except).
            try:
                r = train_multi_seed(op, n_seeds=n_seeds, steps=steps,
                                     softmax_steps=softmax_steps, device=device,
                                     n_kv=n_kv)
                all_results.append(r)
                logger.info(f"  -> mean_acc={r['mean_acc']:.4f} +/- {r['ci95_acc']:.4f} "
                            f"(std={r['std_acc']:.4f}, t_vs_chance={_fmt_tstat(r['t_stat_vs_chance'], width=0, prec=2)})")
            except Exception as e:
                import traceback as _tb
                logger.error(f"  op '{op}' FAILED: {e}")
                _tb.print_exc()
                # Error stub MUST include every key that success rows
                # carry, set to None/empty/zero as appropriate, so
                # downstream JSON consumers (make_figures.py, pandas
                # DataFrames, etc.) iterating over fields do not KeyError
                # on error rows. The success-row schema is defined by
                # train_multi_seed's return dict — mirror it exactly.
                all_results.append({
                    'op': op,
                    'n_kv': n_kv,
                    'n_seeds': n_seeds,
                    'n_seeds_ok': 0,
                    'n_seeds_failed': n_seeds,
                    'n_seeds_total': n_seeds,
                    'seeds': [],
                    'error': str(e),
                    'mean_acc': None,
                    'ci95_acc': None,
                    'std_acc': None,
                    'mean_loss': None,
                    'std_loss': None,
                    'ci95_loss': None,
                    'chance_acc': 1.0 / vocab,
                    't_stat_vs_chance': None,
                    'mean_train_time_s': None,
                    'per_seed': [],
                })

    # Summary table (grouped by n_kv)
    print('\n' + '=' * 80)
    print(f"{'n_kv':>4} | {'op':>10} | {'mean_acc':>10} | {'+/- CI95':>10} | "
          f"{'std':>8} | {'t_vs_chance':>12} | {'mean_loss':>10}")
    print('-' * 80)
    for r in all_results:
        # Error rows have None for all numeric fields; render them as
        # dashes instead of crashing on ``f'{None:.4f}'``.
        if 'error' in r:
            err_msg = (r.get('error') or '')[:40]
            print(f"{r['n_kv']:>4} | {r['op']:>10} | {'-':>10} | "
                  f"{'-':>10} | {'-':>8} | {'-':>12} | {'-':>10}   ERROR: {err_msg}")
            continue
        print(f"{r['n_kv']:>4} | {r['op']:>10} | {r['mean_acc']:>10.4f} | "
              f"{r['ci95_acc']:>10.4f} | {r['std_acc']:>8.4f} | "
              f"{_fmt_tstat(r['t_stat_vs_chance'], width=12, prec=2)} | {r['mean_loss']:>10.4f}")
    # Chance row: fill ALL columns so the table renders as a clean grid
    # (the previous version only filled 3 of 7 columns, leaving the right
    # side of the row ragged with no trailing pipe separators).
    print(f"{'':>4} | {'chance':>10} | {chance:>10.4f} | "
          f"{'':>10} | {'':>8} | {'':>12} | {'':>10}")

    os.makedirs('results', exist_ok=True)
    # Write strict JSON (allow_nan=False): if a divergent seed slipped
    # past the NaN guard and the per_seed filter, Python's default
    # json.dump would emit literal ``NaN``/``Infinity`` tokens, which are
    # INVALID JSON per RFC 8259 and cause strict parsers (js, jq, pandas
    # with ``orient='records'``) to reject the whole file. With
    # allow_nan=False the call raises ValueError instead — surfacing the
    # corruption loudly rather than shipping a broken file.
    with open('results/exp4_mqar.json', 'w') as f:
        try:
            json.dump(all_results, f, indent=2, allow_nan=False)
        except ValueError as e:
            # Fall back to replacing non-finite values with null so the
            # file is still written (downstream code handles None), and
            # log the corruption loudly.
            logger.error(f'non-finite value in results; sanitizing to null: {e}')
            import math as _math
            def _sanitize(o):
                if isinstance(o, float) and not _math.isfinite(o):
                    return None
                if isinstance(o, dict):
                    return {k: _sanitize(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_sanitize(x) for x in o]
                return o
            json.dump(_sanitize(all_results), f, indent=2, allow_nan=False)
    logger.info('\nSaved: results/exp4_mqar.json')


if __name__ == '__main__':
    main()
