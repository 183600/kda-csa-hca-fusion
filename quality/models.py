"""run_quality split module: models."""

from __future__ import annotations

from ._shared import *  # noqa: F401,F403

from .tasks import (  # noqa: F401
    SMALL_MODEL_SPEC,
    SYNTHETIC_TASKS,
)


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


class NoMixAttn(nn.Module):
    """Trivial memoryless baseline: an identity token mixer.

    Hypothesis (baseline ladder, experiment-design requirement): a
    multi-query associative-recall target is a function of the cued key
    AND of the earlier (key, value) pair, so a model with NO token mixing
    cannot recover it and must stay at chance. This operator makes that
    prediction testable with the SAME training recipe as every complex
    mixer: wrapped in ``ResidualAttnLayer`` it computes
    ``x + LayerNorm(x)`` per position (a position-wise MLP with no
    information flow across time), and the head reads only the cue
    position.

    Role in the comparison
    ----------------------
    * ``linear`` is the simplest rung of the baseline ladder. A complex
      operator that does not beat it is not doing anything useful.
    * It is NOT a strawman: it gets the same data streams, optimizer, and
      budget as every other operator, and the tuned-baseline protocol
      gives it its own learning-rate sweep, so "beats the simple
      baseline" means beating a properly tuned simple model.
    * On tasks where it DOES beat chance (e.g. if the target is predictable
      from the current token alone), the task is confounded and the
      operator ranking on it is not evidence about memory mechanisms.

    Zero parameters and zero inter-token communication by construction.
    """

    def forward(self, x):
        return x


class KDAAttn(nn.Module):
    # Named default for the per-channel log-decay scale, so all KDA modules
    # (KDAHybridLayer, KDAAttn, KDAAttnDecoding, _kda_heads) share the same
    # value via HybridConfig.kda_decay_scale (the fused model) or this
    # attribute (the standalone KDAAttn).
    # Mirrors run_decoding.KDAAttnDecoding.decay_scale.
    DECAY_SCALE = 0.1

    def __init__(self, d_model, H=2, K=16, V=16):
        super().__init__()
        self.q = nn.Linear(d_model, H * K, bias=False)
        self.k = nn.Linear(d_model, H * K, bias=False)
        self.v = nn.Linear(d_model, H * V, bias=False)
        # Match KDAHybridLayer's low-rank gate parameterization: d -> K -> H*K,
        # so standalone Exp4 KDA shares the hybrid KDA layer's operator
        # boundary and the Exp3 FLOPs formula.
        self.g_down = nn.Linear(d_model, K, bias=False)
        self.g_up = nn.Linear(K, H * K, bias=False)
        self.beta = nn.Linear(d_model, H, bias=False)
        self.o = nn.Linear(H * V, d_model, bias=False)
        # Causal depthwise short-conv (kernel=3) matching KDAHybridLayer
        # (ops_fused.py:307) and KDAAttnDecoding (run_decoding.py:283), so
        # cross-experiment KDA comparisons and the run_kv_cache
        # ``short_conv`` FLOPs term (line 340) see the same operator.
        # Exp 4 runs on full sequences (not streaming), so we left-pad with
        # zeros each call (causal, no future leakage) and do NOT carry a
        # persistent ``_conv_lookback`` buffer (unlike KDAHybridLayer /
        # KDAAttnDecoding which need it for autoregressive decode).
        self.short_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=0,
            groups=d_model, bias=True,
        )
        # Keep quality experiments on the reference backend by default so
        # historical accuracy numbers remain comparable. Set KDA_BACKEND=fla
        # (after installing the optional [fla] extra) only for an explicitly
        # accelerated run; KDA_BACKEND=auto uses FLA on CUDA when available.
        self.kda_backend = validate_kda_backend(
            os.environ.get('KDA_BACKEND', 'reference')
        )
        self.H, self.K, self.V = H, K, V

    def forward(self, x):
        B, T, d = x.shape
        # Apply the causal depthwise short-conv before the q/k/v/g/beta
        # projections, mirroring KDAHybridLayer.forward (ops_fused.py:501)
        # and KDAAttnDecoding.forward
        # (run_decoding.py:326). Left-pad by (ksize-1)=2 zeros so position
        # t sees {t-2, t-1, t} and NEVER t+1 (future leakage). Conv1d
        # padding=0; the F.pad does the causal left-pad explicitly.
        ksize = self.short_conv.kernel_size[0]
        x_conv = self.short_conv(
            F.pad(x.transpose(1, 2), (ksize - 1, 0))
        ).transpose(1, 2)
        # View BEFORE normalize: L2 normalization must operate on each
        # per-head K-dim vector, not on the concatenated H*K vector —
        # otherwise each head's L2 norm becomes ~1/sqrt(H) instead of 1 and
        # the q.k dot products shrink by a factor of 1/H. Mirrors
        # ops_fused.py::KDAHybridLayer.
        q = _l2_normalize(
            F.silu(self.q(x_conv)).view(B, T, self.H, self.K))
        k = _l2_normalize(
            F.silu(self.k(x_conv)).view(B, T, self.H, self.K))
        v = F.silu(self.v(x_conv)).view(B, T, self.H, self.V)
        # Log-space gate: low-rank down/up with a softplus-style decay,
        # matching ops_fused.KDAHybridLayer and run_kv_cache.prefill_flops.
        # Uses the named DECAY_SCALE constant so all KDA instantiations agree.
        g = -F.softplus(self.g_up(self.g_down(x_conv))).view(
            B, T, self.H, self.K) * self.DECAY_SCALE
        beta = torch.sigmoid(self.beta(x_conv))
        out, _ = kda_forward(
            q, k, v, g, beta,
            output_final_state=False,
            use_chunk=False,
            backend=self.kda_backend,
        )
        return self.o(out.reshape(B, T, self.H * self.V))


class Mamba2Attn(nn.Module):
    """Standalone single-layer Mamba-2 operator for the quality probes.

    Wraps the reference :class:`ops_mamba.Mamba2Mixer` (via the backend
    factory, so ``MAMBA_BACKEND=official`` selects the mamba_ssm kernels
    where available). The geometry knobs are sized for the tiny quality
    models and are env-overridable: ``MAMBA2_D_STATE`` (default 16),
    ``MAMBA2_HEADDIM`` (default 16; clamped down to the largest divisor
    of ``expand * d_model``), ``MAMBA2_EXPAND`` (default 2),
    ``MAMBA2_D_CONV`` (default 4).

    Like :class:`KDAAttn`, the operator is STATELESS per call: the MQAR
    training loop feeds an independent random batch each step, so any
    conv context carried across calls would be cross-batch leakage.
    ``forward`` therefore always starts from a fresh conv state and a
    fresh SSM state.
    """

    def __init__(self, d_model):
        super().__init__()
        expand = parse_int_env('MAMBA2_EXPAND', 2, min_value=1, logger=logger)
        d_state = parse_int_env('MAMBA2_D_STATE', 16, min_value=1,
                                logger=logger)
        headdim = parse_int_env('MAMBA2_HEADDIM', 16, min_value=1,
                                logger=logger)
        d_conv = parse_int_env('MAMBA2_D_CONV', 4, min_value=2, logger=logger)
        d_inner = expand * d_model
        # Clamp headdim down to the largest divisor of d_inner (a
        # misconfigured headdim would otherwise abort the whole sweep with
        # a geometry error; the env knobs are convenience, not precision
        # instruments).
        headdim_eff = min(headdim, d_inner)
        while d_inner % headdim_eff != 0:
            headdim_eff //= 2
        cfg = HybridConfig(
            d_model=d_model, memory_type='mamba2',
            mamba_d_state=d_state, mamba_d_conv=d_conv,
            mamba_expand=expand, mamba_headdim=headdim_eff,
            mamba_backend=os.environ.get('MAMBA_BACKEND', 'reference'),
        )
        self.mixer = build_mamba2_mixer(cfg)

    def forward(self, x):
        out, _, _ = self.mixer.forward_functional(x, None, None)
        return out


class Rwkv7Attn(nn.Module):
    """Standalone single-layer RWKV-7 operator for the quality probes.

    Wraps the reference :class:`ops_rwkv.Rwkv7Mixer` (via the backend
    factory, so ``RWKV_BACKEND=fla`` selects the flash-linear-attention
    kernels where available). The geometry knobs are sized for the tiny
    quality models and are env-overridable: ``RWKV_HEAD_DIM`` (default
    16; clamped down to the largest divisor of ``d_model``),
    ``RWKV_DECAY_LORA`` / ``RWKV_A_LORA`` (default 4),
    ``RWKV_GATE_LORA`` (default 8) — the same env knobs run_ablation's
    matrix uses, so the standalone probes and the matrix stacks test
    the same mixer geometry at a given d_model.

    Like :class:`Mamba2Attn`, the operator is STATELESS per call: the
    MQAR training loop feeds an independent random batch each step, so
    any shift state carried across calls would be cross-batch leakage.
    ``forward`` therefore always starts from a fresh shift state and a
    fresh matrix state.
    """

    def __init__(self, d_model):
        super().__init__()
        head_dim = parse_int_env('RWKV_HEAD_DIM', 16, min_value=1,
                                 logger=logger)
        decay_lora = parse_int_env('RWKV_DECAY_LORA', 4, min_value=1,
                                   logger=logger)
        a_lora = parse_int_env('RWKV_A_LORA', 4, min_value=1, logger=logger)
        gate_lora = parse_int_env('RWKV_GATE_LORA', 8, min_value=1,
                                  logger=logger)
        head_dim_eff = min(head_dim, d_model)
        while d_model % head_dim_eff != 0:
            head_dim_eff //= 2
        cfg = HybridConfig(
            d_model=d_model, memory_type='rwkv7',
            rwkv_head_dim=head_dim_eff,
            rwkv_decay_lora=min(decay_lora, d_model),
            rwkv_a_lora=min(a_lora, d_model),
            rwkv_gate_lora=min(gate_lora, d_model),
            rwkv_backend=os.environ.get('RWKV_BACKEND', 'reference'),
        )
        # A standalone single-layer probe: this layer is RWKV layer 0 of
        # 1 (the depth-dependent init constants degenerate gracefully —
        # see Rwkv7Mixer._initialize_weights).
        self.mixer = build_rwkv7_mixer(cfg, rwkv_layer_idx=0,
                                        n_rwkv_layers=1)

    @property
    def effective_backend(self):
        return getattr(self.mixer, 'effective_backend', 'reference')

    def forward(self, x):
        out, _, _ = self.mixer.forward_functional(x, None, None)
        return out


class CSAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Use the shared small-model spec so Experiment 4 (standalone MQAR)
        # and Experiment 5 (ablation) test the SAME CSA sub-layer widths.
        spec = SMALL_MODEL_SPEC
        c, dc = spec['csa_c'], spec['csa_dc']
        m, nh, nIh, cI, topk = (
            spec['csa_m'], spec['csa_nh'], spec['csa_nIh'],
            spec['csa_cI'], spec['csa_topk'],
        )
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
        # and trims its output back to the original T, so no external
        # pad/trim is needed (see ``test_csa_hca_non_divisible_T``). Keeping
        # this wrapper thin avoids doing the pad/trim twice.
        o = naive_csa(
            x, self.W_aKV.weight, self.W_bKV.weight,
            self.W_aZ.weight, self.W_bZ.weight, self.Ba, self.Bb,
            self.W_DQ.weight, self.W_UQ.weight, self.W_IUQ.weight,
            self.W_w.weight, self.W_KV_idx.weight, self.W_Z_idx.weight,
            self.B_idx,
            m=self.m, topk=self.topk, nh=self.nh, nIh=self.nIh,
            c=self.c, c_I=self.cI, dc=self.dc,
            sliding_window=SMALL_MODEL_SPEC['csa_sliding_window'],
            sink_logits=self.sink,
            # Use cosine-style indexer scoring for the quality experiment so
            # top-k selection is not confounded by q_idx / K_idx vector norms.
            normalize_qk=True,
            # Partial RoPE (DeepSeek-V4 §2.3.3), paper-faithful default.
            rope_dim=SMALL_MODEL_SPEC.get('csa_rope_dim', 64),
            rope_base=SMALL_MODEL_SPEC.get('rope_base', 10000.0),
        )
        return self.o(o)


class HCAAttn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Use the shared small-model spec (mirrors CSAAttn). HCA's defining
        # feature is *heavy* compression: m2 >> m so the HCA branch produces
        # far fewer compressed blocks than CSA, trading recall granularity
        # for global context. With seq_len=16 and CSA m=4 (n_blocks_CSA=4),
        # setting m2=8 gives n_blocks_HCA=2, exercising the heavier-
        # compression regime.
        spec = SMALL_MODEL_SPEC
        c, dc = spec['hca_c'], spec['hca_dc']
        m2, nh = spec['hca_m2'], spec['hca_nh']
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
        # and trims its output back to the original T, so no pad/trim is
        # needed here. See the comment in ``CSAAttn.forward`` for the
        # full rationale.
        o = naive_hca(x, self.W_KV.weight, self.W_Z.weight, self.B_pos,
                      self.W_DQ.weight, self.W_UQ.weight,
                      m2=self.m2, nh=self.nh, c=self.c, dc=self.dc,
                      sliding_window=SMALL_MODEL_SPEC['hca_sliding_window'],
                      sink_logits=self.sink,
                      # Partial RoPE (DeepSeek-V4 §2.3.3), paper-faithful
                      # default.
                      rope_dim=SMALL_MODEL_SPEC.get('hca_rope_dim', 64),
                      rope_base=SMALL_MODEL_SPEC.get('rope_base', 10000.0))
        return self.o(o)


def _eval_model(layer, head, embed, seq_len, n_kv, vocab, device,
                n_batches=4, batch=64, seed=42, task='mqar',
                gen_offset=5_000_000, collect_predictions=False):
    """Evaluate accuracy over multiple fresh batches for a stable estimate.

    ``task`` selects the synthetic-task generator from ``SYNTHETIC_TASKS``
    (default ``'mqar'`` keeps the historical behaviour).

    Uses a dedicated ``torch.Generator`` for batch generation so the eval
    batches are reproducible and independent of any global-RNG state left
    over by training. This also makes the eval pass deterministic across
    re-runs, which is useful for debugging.

    The eval generator is derived from the training ``seed`` (not a fixed
    constant), so each seed is evaluated on FRESH eval batches. This makes
    the multi-seed ``std_acc`` / CI / t-test in ``train_multi_seed`` capture
    BOTH training stochasticity and eval-sampling variance — i.e. the full
    variance of repeating the experiment from scratch. A fixed eval seed
    would measure every seed on the SAME eval batches, excluding
    eval-sampling variance from the CI and overstating the significance of
    the operator comparisons. Because the
    generator depends only on ``seed``, all operators for a given seed still
    see the SAME eval batches (apples-to-apples), and the far ``+ 5_000_000``
    offset keeps the eval stream clear of every seed's ``+ 1_000_000``
    training stream (mirrors ``run_ablation.py::_eval_model``).

    ``gen_offset`` selects which seed offset defines this eval stream. The
    default (5_000_000) is the headline eval the reported accuracy comes
    from. The held-out validation-monitoring curve uses
    ``_VAL_MONITOR_SEED_OFFSET`` (6_000_000) instead, so monitored examples
    are never the examples the headline metric was computed on, and (with
    the error-analysis stream at 7_000_000) monitoring cannot leak into the
    reported result.

    ``collect_predictions`` (default False, historical 2-tuple return)
    additionally returns the concatenated per-sample ``pred`` / ``target``
    and per-class ``scores`` tensors (on CPU) so the caller can compute
    confusion-matrix metrics (macro/micro/weighted F1, per-class
    precision/recall, PR-AUC / ROC-AUC one-vs-rest) without a second
    forward pass. The accumulated tensors are detached, so collecting them
    can never change gradients or global RNG state.

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
        # Fresh eval batches per seed (see the docstring for the statistical
        # rationale): seed+5_000_000 is far from every seed's training stream
        # at seed+1_000_000 and from run_ablation's own eval offset usage.
        eval_gen = make_seeded_generator(seed + gen_offset, device=device)
        make_batch = SYNTHETIC_TASKS[task]['make']
        correct, total = 0, 0
        losses = []
        pred_chunks, target_chunks, score_chunks = [], [], []
        with torch.no_grad():
            for _ in range(n_batches):
                x_emb, target, cue_pos = make_batch(
                    batch, seq_len, n_kv, vocab, embed, device,
                    generator=eval_gen)
                h = layer(x_emb)
                logits = head(h, cue_pos)
                correct += (logits.argmax(-1) == target).sum().item()
                total += target.numel()
                losses.append(F.cross_entropy(logits, target).item())
                if collect_predictions:
                    pred_chunks.append(logits.argmax(-1).detach().to('cpu'))
                    target_chunks.append(target.detach().to('cpu'))
                    score_chunks.append(logits.detach().float().to('cpu'))
        # Guard against n_batches=0 (or batch=0): without this the function
        # raises ZeroDivisionError on ``correct / total`` and
        # ``sum([]) / len([])``. Returns 0.0 for both metrics so the caller
        # gets a finite (if meaningless) value rather than a crash. Mirrors
        # the steps=0 guard in train_one.
        if total == 0 or not losses:
            if collect_predictions:
                return 0.0, 0.0, {
                    'pred': torch.zeros(0, dtype=torch.long),
                    'target': torch.zeros(0, dtype=torch.long),
                    'scores': torch.zeros(0, 0),
                }
            return 0.0, 0.0
        acc = correct / total
        mean_loss = sum(losses) / len(losses)
        if collect_predictions:
            return acc, mean_loss, {
                'pred': torch.cat(pred_chunks),
                'target': torch.cat(target_chunks),
                'scores': torch.cat(score_chunks),
            }
        return acc, mean_loss
    finally:
        for m, was in was_training.items():
            m.train(was)


_MEMORYLESS_OPS = frozenset({'linear'})


def _op_factory(op_name, d_model):
    """Return the attention-operator factory for a quality-probe op.

    Lifted out of ``train_one`` so ``run_overfit_sanity_check`` constructs
    the operator through the IDENTICAL code path (same classes, same
    ``d_model``). The factory dictionary itself consumes no RNG draws, so
    routing both callers through this function does not change the seeded
    initialization stream.
    """
    factories = {
        # Simplest rung of the baseline ladder: no token mixing at all.
        # Any operator that cannot beat this on a memory task is not
        # demonstrating a memory mechanism.
        'linear':  lambda: NoMixAttn(),
        'softmax': lambda: SoftmaxAttn(d_model),
        'kda':     lambda: KDAAttn(d_model),
        'mamba2':  lambda: Mamba2Attn(d_model),
        'rwkv7':   lambda: Rwkv7Attn(d_model),
        'csa':     lambda: CSAAttn(d_model),
        'hca':     lambda: HCAAttn(d_model),
    }
    if op_name not in factories:
        raise ValueError(
            f"op_name={op_name!r} must be one of {sorted(factories)}.")
    return factories[op_name]


def _build_task_model(op_name, d_model, vocab, device):
    """Build ``(embed, head, layer)`` in the seeded construction order.

    Order matters for reproducibility: embed and head are created BEFORE
    the operator-specific layer so their initial weights are IDENTICAL
    across operators for a given seed. Different operators consume a
    different number of RNG draws during init, so creating the layer first
    would desync the downstream head init — a silent confound in the
    multi-seed CI. Callers must invoke this after ``set_global_seed(seed)``
    and must not reorder the three constructions.

    Shared by ``train_one`` and ``run_overfit_sanity_check`` so the gate
    validates the exact model that the full experiment trains.
    """
    embed = nn.Embedding(vocab, d_model).to(device)
    head = MQARHead(d_model, vocab).to(device)
    layer = ResidualAttnLayer(_op_factory(op_name, d_model)(), d_model).to(device)
    return embed, head, layer


__all__ = [
    'MQARHead',
    'ResidualAttnLayer',
    'SoftmaxAttn',
    'NoMixAttn',
    'KDAAttn',
    'Mamba2Attn',
    'Rwkv7Attn',
    'CSAAttn',
    'HCAAttn',
    '_eval_model',
    '_MEMORYLESS_OPS',
    '_op_factory',
    '_build_task_model',
]
