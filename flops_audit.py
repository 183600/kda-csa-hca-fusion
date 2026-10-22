"""Measured GEMM-work audit of the reference operators (Exp 16).

Why this module?
----------------
``run_kv_cache.prefill_flops`` is the repository's FLOP accounting for
Exp 3 and Exp 14.  It is analytic, closed-form, and follows the papers'
idealized causal counts (DeepSeek-V4, Kimi Linear §6.3 Eq 13).  The README
explicitly flags the gap: the numbers are **not** measured from a real
forward pass and should be cross-checked with a FLOP counter.  This module
is that cross-check, and it answers three questions the analytic ledger
cannot:

1. **Which path actually runs?**  ``prefill_flops('kda')`` always adds the
   chunkwise inter-chunk terms, but ``KDAHybridLayer`` only dispatches to
   the chunk path when ``T >= kda_chunk_size``; shorter sequences (every
   MQAR probe in Exp 4/5/9/11) run the step-by-step recurrent KDA, so the
   analytic number overcounts the executed work there.
2. **What does the reference implementation actually execute?**  The
   reference chunk kernel *materializes* the intra-chunk matrices
   ``A`` / ``Aqk`` and solves a unit-lower-triangular system — neither is
   itemized in Eq 13.  At the default ``C=64`` the measured GEMM work of
   the reference KDA layer is ~1.33x the canonical count.  The audit
   reports that "reference tax" instead of hiding it.
3. **What do the gates / softmaxes cost?**  The analytic ledger counts
   matmuls only.  The audit additionally records a **special-operation
   ledger** (``exp`` / ``softplus`` / ``silu`` / ``sigmoid`` for the KDA
   gated linear attention, ``softmax`` / ``logsumexp`` for the CSA/HCA
   joint softmax, ``sin`` / ``cos`` for partial RoPE, ``l2_norm`` for the
   cosine-attention normalization).  These are not FLOPs in the GEMM
   sense, but they are throughput-relevant on real hardware.

What is counted
---------------
* ``mm`` / ``addmm`` / ``bmm`` / ``baddbmm``: ``2 * MACs`` FLOPs, read off
  the dispatched operand shapes (broadcasted ``einsum`` always decomposes
  to these, so no equation parser is needed).
* ``convolution``: ``2 * N * C_out * prod(out_spatial) * (C_in / groups) *
  prod(kernel)`` FLOPs.
* ``linalg_solve_triangular``: the reference chunk path solves
  ``(I - A) X = A`` with ``X`` as an ``N x N`` right-hand side.  The audit
  uses the textbook forward-substitution count ``N^2 (N - 1) / 2`` MACs
  per system (this matched the dispatch-level counter in the regression
  tests); LAPACK's blocked TRSM has the same asymptotics with a smaller
  constant, so the audit is conservative on the reference's behalf.
* special ops: element counts, one per op invocation, keyed by a readable
  label (not converted to FLOPs).

The exact reference model
-------------------------
``reference_gemm_macs(kind, cfg, T)`` re-derives the *executed* GEMM work
from the implementation's matmul shapes, not from the paper.  Regression
tests pin ``reference_gemm_macs == measured`` to the MAC for the default
KDA / CSA / HCA paths, so the model cannot drift from the operators it
describes.  ``canonical_flops`` returns the ideal analytic number, so the
gap between the two is always visible.

Scope and honesty
-----------------
* CPU/fp32 reference path, fresh state, ``batch`` sequences, default
  flags: ``kda_recompute_attn=False`` / ``kda_assoc_scan=False``.  The
  associative-scan variant adds batched state-combine matmuls that this
  model does not itemize (the audit still measures them; the exact-model
  function raises instead of guessing).
* The KDA recurrent path implements two of its three per-step contractions
  as elementwise broadcast-reduce (`(k_i * S).sum(-2)` and the
  ``addcmul_`` outer product).  Their MACs are real work but not GEMM
  dispatches; ``kda_broadcast_encoded_macs`` reports them separately.
* HCA with a single compressed block (``n_blocks == 1``) lets PyTorch fold
  the value contraction into a mul+sum, so the measured GEMM can fall
  short of the model by one ``T * c * nh`` term there.  Tests use
  ``n_blocks >= 2``; the audit reports the raw difference either way.

Usage::

    python3 flops_audit.py --config probe --seq-lens 32,64,128,256
    python3 -m pytest -q test_flops_audit.py
"""
from __future__ import annotations

import argparse
import dataclasses
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ops_fused import HybridConfig, build_mixer
from ops_kda import _attn_block_rows, _recompute_block_rows
from run_kv_cache import prefill_flops

__all__ = [
    'GemmLedger',
    'audit_layer',
    'canonical_flops',
    'hca_reference_gemm_macs',
    'geometry_kwargs',
    'kda_broadcast_encoded_macs',
    'kda_canonical_terms',
    'kda_executed_flops',
    'kda_path',
    'kda_reference_gemm_macs',
    'measure_forward',
    'probe_config',
    'reference_gemm_macs',
]

SUPPORTED_KINDS = ('kda', 'mamba2', 'csa', 'hca')

#: Readable labels for the non-matmul ops the ledger tracks.  These are
#: element counts, NOT FLOPs: the point is to expose the gated-attention /
#: softmax / RoPE normalization mass that the analytic ledger ignores.
SPECIAL_OPS = {
    'aten::exp': 'exp',
    'aten::exp_': 'exp',
    'aten::sigmoid': 'sigmoid',
    'aten::sigmoid_': 'sigmoid',
    'aten::tanh': 'tanh',
    'aten::tanh_': 'tanh',
    'aten::softplus': 'softplus',
    'aten::silu': 'silu',
    'aten::silu_': 'silu',
    'aten::sin': 'sin',
    'aten::cos': 'cos',
    'aten::rsqrt': 'rsqrt',
    'aten::rsqrt_': 'rsqrt',
    'aten::sqrt': 'sqrt',
    'aten::sqrt_': 'sqrt',
    'aten::_softmax': 'softmax',
    'aten::log_softmax': 'log_softmax',
    'aten::logsumexp': 'logsumexp',
    'aten::linalg_vector_norm': 'l2_norm',
}

_MATMUL_OPS = ('aten::mm', 'aten::addmm', 'aten::bmm', 'aten::baddbmm')


@dataclasses.dataclass(frozen=True)
class GemmLedger:
    """Measured work of one forward pass.

    ``macs_by_op`` uses the dispatch op name as the key; ``trsm_macs`` is
    the estimated triangular-solve work; ``special_ops`` maps the readable
    labels above to element counts.
    """

    macs_by_op: dict
    trsm_macs: int = 0
    trsm_calls: int = 0
    special_ops: dict = dataclasses.field(default_factory=dict)

    @property
    def matmul_macs(self) -> int:
        return sum(self.macs_by_op.values())

    @property
    def gemm_macs(self) -> int:
        return self.matmul_macs + self.trsm_macs

    @property
    def gemm_flops(self) -> int:
        return 2 * self.gemm_macs

    @property
    def special_total(self) -> int:
        return sum(self.special_ops.values())

    def as_dict(self) -> dict:
        return {
            'macs_by_op': dict(self.macs_by_op),
            'matmul_macs': self.matmul_macs,
            'trsm_macs': self.trsm_macs,
            'trsm_calls': self.trsm_calls,
            'gemm_macs': self.gemm_macs,
            'gemm_flops': self.gemm_flops,
            'special_ops': dict(self.special_ops),
        }


class _GemmAuditMode(TorchDispatchMode):
    """TorchDispatchMode that counts matmul/conv/solve and special ops."""

    def __init__(self) -> None:
        super().__init__()
        self.macs_by_op: dict = {}
        self.trsm_macs = 0
        self.trsm_calls = 0
        self.special_ops: dict = {}

    def _count_macs(self, name: str, args) -> int | None:
        if name == 'aten::mm':
            a, b = args[0], args[1]
            return a.shape[0] * a.shape[1] * b.shape[1]
        if name == 'aten::addmm':
            a, b = args[1], args[2]
            return a.shape[0] * a.shape[1] * b.shape[1]
        if name in ('aten::bmm', 'aten::baddbmm'):
            a, b = args[-2], args[-1]
            return a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
        if name == 'aten::convolution':
            inp, weight = args[0], args[1]
            stride, padding, dilation = args[3], args[4], args[5]
            out_spatial = 1
            for i in range(2, inp.dim()):
                k = weight.shape[i]
                out_spatial *= (
                    inp.shape[i] + 2 * padding[i - 2]
                    - dilation[i - 2] * (k - 1) - 1
                ) // stride[i - 2] + 1
            return (inp.shape[0] * weight.shape[0] * out_spatial
                    * weight.shape[1] * math.prod(weight.shape[2:]))
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        output = func(*args, **(kwargs or {}))
        if name in _MATMUL_OPS or name == 'aten::convolution':
            macs = self._count_macs(name, args)
            if macs is not None:
                self.macs_by_op[name] = self.macs_by_op.get(name, 0) + macs
        elif name == 'aten::linalg_solve_triangular':
            a = args[1]
            n = a.shape[-1]
            systems = a.numel() // (n * n)
            self.trsm_macs += systems * n * n * (n - 1) // 2
            self.trsm_calls += 1
        label = SPECIAL_OPS.get(name)
        if label is not None:
            self.special_ops[label] = (
                self.special_ops.get(label, 0) + args[0].numel())
        return output

    def ledger(self) -> GemmLedger:
        return GemmLedger(
            macs_by_op=dict(self.macs_by_op),
            trsm_macs=self.trsm_macs,
            trsm_calls=self.trsm_calls,
            special_ops=dict(self.special_ops),
        )


def measure_forward(module, x: torch.Tensor) -> tuple:
    """Run ``module(x)`` under the audit counter.

    The module is evaluated (no dropout paths exist, but ``eval`` keeps the
    contract explicit) with autograd disabled, matching how the reference
    operators are priced.  Returns ``(module_output, GemmLedger)``.
    """
    mode = _GemmAuditMode()
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad(), mode:
            out = module(x)
    finally:
        module.train(was_training)
    return out, mode.ledger()


def geometry_kwargs(cfg) -> dict:
    """``HybridConfig`` -> ``run_kv_cache.prefill_flops`` kwargs.

    Mirrors ``run_train_roofline.hybrid_geometry_kwargs`` (regression-tested
    against it) so this module does not import the whole Exp 14 runner.
    """
    return dict(
        d=cfg.d_model, H=cfg.n_heads_qk, K=cfg.head_dim_k, V=cfg.head_dim_v,
        kda_hv=cfg.n_heads_v, kda_k=cfg.head_dim_k, kda_v=cfg.head_dim_v,
        kda_conv_ksize=3, kda_chunk_size=cfg.kda_chunk_size,
        csa_m=cfg.csa_m, csa_topk=cfg.csa_topk, csa_c=cfg.csa_c,
        csa_dc=cfg.csa_dc, csa_cI=cfg.csa_cI, csa_nIh=cfg.csa_nIh,
        csa_nh=cfg.csa_nh, csa_sliding_window=cfg.csa_sliding_window,
        hca_m2=cfg.hca_m2, hca_c=cfg.hca_c, hca_dc=cfg.hca_dc,
        hca_nh=cfg.hca_nh, hca_sliding_window=cfg.hca_sliding_window,
        mamba_d_state=cfg.mamba_d_state, mamba_d_conv=cfg.mamba_d_conv,
        mamba_expand=cfg.mamba_expand, mamba_headdim=cfg.mamba_headdim,
        mamba_ngroups=cfg.mamba_ngroups, mamba_dt_rank=cfg.mamba_dt_rank,
        rwkv_head_dim=cfg.rwkv_head_dim, rwkv_decay_lora=cfg.rwkv_decay_lora,
        rwkv_a_lora=cfg.rwkv_a_lora, rwkv_gate_lora=cfg.rwkv_gate_lora,
    )


def canonical_flops(kind: str, cfg, T: int, batch: int = 1) -> float:
    """Ideal analytic prefill FLOPs (``run_kv_cache`` convention)."""
    _validate(kind, T, batch)
    return float(batch) * float(prefill_flops(kind, T, **geometry_kwargs(cfg)))


def _validate(kind: str, T: int, batch: int = 1) -> None:
    if kind not in SUPPORTED_KINDS:
        raise ValueError(
            f'flops_audit: kind={kind!r} must be one of {SUPPORTED_KINDS}.')
    if isinstance(T, bool) or not isinstance(T, int):
        raise TypeError(f'flops_audit: T={T!r} must be an int.')
    if T < 0:
        raise ValueError(f'flops_audit: T={T} must be >= 0.')
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError(f'flops_audit: batch={batch!r} must be an int.')
    if batch < 1:
        raise ValueError(f'flops_audit: batch={batch} must be >= 1.')


def kda_path(cfg, T: int) -> str:
    """Which KDA core path ``KDAHybridLayer`` dispatches to at length ``T``."""
    if cfg.kda_chunk_size < 1 or T < cfg.kda_chunk_size:
        return 'recurrent'
    return 'chunk'


def _prepare_entries(block_rows: int, BT: int) -> int:
    """Entries the ``A`` / ``Aqk`` build writes per chunk (row blocks)."""
    total = 0
    for c0 in range(0, BT, block_rows):
        c1 = min(BT, c0 + block_rows)
        total += (c1 - c0) * c1
    return total


def _recompute_entries(block_rows: int, BT: int) -> int:
    """Entries the recompute variant rebuilds per chunk (column blocks)."""
    total = 0
    for j0 in range(0, BT, block_rows):
        j1 = min(BT, j0 + block_rows)
        total += (BT - j0) * (j1 - j0)
    return total


def kda_canonical_terms(cfg, T: int, batch: int = 1) -> dict:
    """Term-by-term decomposition of the canonical KDA prefill FLOPs.

    Mirrors the ``prefill_flops('kda')`` branches; ``sum(values)`` equals
    ``canonical_flops('kda', ...)`` (pinned by the regression suite).
    """
    d, H, K, V = cfg.d_model, cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v
    HV = cfg.n_heads_v
    b = float(batch)
    proj = b * 2 * T * (
        d * (2 * H * K + HV * V + K + HV) + K * HV * K)
    short_conv = b * 2 * T * d * 3
    recurrent = b * 2 * 3 * T * HV * K * V
    c = cfg.kda_chunk_size
    chunk_inter = b * 3 * T * HV * c * K
    chunk_a = b * T * HV * c * c
    out_proj = b * 2 * T * HV * V * d
    return {
        'proj': proj,
        'short_conv': short_conv,
        'recurrent': recurrent,
        'chunk_inter': chunk_inter,
        'chunk_a': chunk_a,
        'out_proj': out_proj,
    }


def kda_executed_flops(cfg, T: int, batch: int = 1) -> float:
    """Canonical KDA FLOPs restricted to the path that actually executes.

    The canonical formula always adds the chunkwise inter-chunk terms, but
    the layer only runs the chunk path for ``T >= kda_chunk_size``.  For
    shorter sequences those terms are dropped; the recurrent term stays
    (it is the work the recurrent path performs, up to the two
    broadcast-encoded contractions reported by
    :func:`kda_broadcast_encoded_macs`).
    """
    _validate('kda', T, batch)
    terms = kda_canonical_terms(cfg, T, batch)
    if kda_path(cfg, T) == 'recurrent':
        terms['chunk_inter'] = 0.0
        terms['chunk_a'] = 0.0
    return float(sum(terms.values()))


def kda_broadcast_encoded_macs(cfg, T: int, batch: int = 1) -> int:
    """Recurrent-path MACs the reference encodes as elementwise broadcast-reduce.

    ``naive_recurrent_kda`` computes ``(k_i * S).sum(-2)`` and the
    ``addcmul_`` outer product instead of GEMMs: real work, no ``bmm``
    dispatch.  Zero on the chunk path (all chunk-path contractions are
    matmuls).
    """
    _validate('kda', T, batch)
    if kda_path(cfg, T) != 'recurrent':
        return 0
    return batch * 2 * T * cfg.n_heads_v * cfg.head_dim_k * cfg.head_dim_v


def kda_reference_gemm_macs(cfg, T: int, batch: int = 1) -> int:
    """Exact GEMM+TRSM MACs executed by ``KDAHybridLayer`` at length ``T``."""
    _validate('kda', T, batch)
    if cfg.kda_assoc_scan:
        raise NotImplementedError(
            'flops_audit: the associative-scan KDA variant adds batched '
            'state-combine matmuls that this exact model does not itemize. '
            'Call measure_forward() to measure it instead.')
    d, H, K, V = cfg.d_model, cfg.n_heads_qk, cfg.head_dim_k, cfg.head_dim_v
    HV, B, BT = cfg.n_heads_v, batch, cfg.kda_chunk_size
    proj = B * T * (d * (2 * H * K + HV * V + K + HV) + K * HV * K)
    conv = B * T * d * 3
    out_proj = B * T * HV * V * d
    if kda_path(cfg, T) == 'recurrent':
        return proj + conv + out_proj + B * T * HV * K * V
    Tp = T + (-T) % BT
    NT = Tp // BT
    block_rows = _attn_block_rows(B, HV, NT, BT, K)
    entries = _prepare_entries(block_rows, BT)
    a_macs = B * HV * NT * entries * K
    if cfg.kda_recompute_attn:
        recompute_rows = _recompute_block_rows(B, HV, NT, BT, K)
        aqk_macs = B * HV * NT * _recompute_entries(recompute_rows, BT) * K
    else:
        aqk_macs = a_macs
    trsm = B * HV * NT * BT * BT * (BT - 1) // 2
    w_macs = B * HV * NT * BT * BT * K
    u_macs = B * HV * NT * BT * BT * V
    ws_macs = B * HV * NT * BT * K * V
    qs_macs = B * HV * NT * BT * K * V
    aqv_macs = B * HV * NT * BT * BT * V
    ksv_macs = B * HV * NT * K * BT * V
    return (proj + conv + out_proj + a_macs + aqk_macs + trsm + w_macs
            + u_macs + ws_macs + qs_macs + aqv_macs + ksv_macs)


def mamba2_path(cfg) -> str:
    """Which Mamba-2 core path ``Mamba2Mixer`` dispatches to."""
    _validate('mamba2', 0, 1)
    return 'chunked' if int(getattr(cfg, 'mamba_chunk_size', 0)) > 0 \
        else 'sequential'


def mamba2_canonical_terms(cfg, T: int, batch: int = 1) -> dict:
    """Term-by-term decomposition of the canonical Mamba-2 prefill FLOPs.

    Mirrors ``prefill_flops('mamba2')``; ``sum(values)`` equals
    ``canonical_flops('mamba2', ...)``. The ``scan`` term charges the three
    state-sized MACs the ideal recurrence performs (decayed update,
    ``x (x) B`` outer product, ``C @ S^T`` readout); only the readout
    dispatches as a GEMM in the reference implementation.
    """
    _validate('mamba2', T, batch)
    d = cfg.d_model
    d_inner = cfg.mamba_expand * d
    headdim = cfg.mamba_headdim
    d_state = cfg.mamba_d_state
    ngroups = cfg.mamba_ngroups
    nheads = d_inner // headdim
    dt_rank_cfg = cfg.mamba_dt_rank
    dt_rank = (max(1, (d + 15) // 16)
               if isinstance(dt_rank_cfg, str) else int(dt_rank_cfg))
    b = float(batch)
    return {
        'in_proj': b * 2 * T * d * 2 * d_inner,
        'conv': b * 2 * T * d_inner * cfg.mamba_d_conv,
        'x_proj': b * 2 * T * d_inner * (dt_rank + 2 * ngroups * d_state),
        'dt_proj': b * 2 * T * dt_rank * nheads,
        'scan': b * 2 * 3 * T * nheads * headdim * d_state,
        'out_proj': b * 2 * T * d_inner * d,
    }


def mamba2_broadcast_encoded_macs(cfg, T: int, batch: int = 1) -> int:
    """State-sized MACs the sequential reference encodes elementwise.

    The decayed update ``decay * S + x_dt`` and the ``x (x) B`` outer
    product are broadcast elementwise kernels, not GEMMs. Two state-sized
    MACs per token are therefore missing from the dispatch ledger; the
    readout ``C @ S^T`` is the one scan contraction that does dispatch.
    """
    _validate('mamba2', T, batch)
    if mamba2_path(cfg) != 'sequential':
        raise NotImplementedError(
            'flops_audit: the chunked Mamba-2 scan has a different '
            'broadcast/GEMM split; call measure_forward() instead.')
    nheads = (cfg.mamba_expand * cfg.d_model) // cfg.mamba_headdim
    return batch * 2 * T * nheads * cfg.mamba_headdim * cfg.mamba_d_state


def mamba2_executed_flops(cfg, T: int, batch: int = 1) -> float:
    """Exact FLOPs executed by the sequential reference Mamba-2 layer.

    ``2 * reference_gemm_macs + 2 * broadcast_encoded_macs``. This equals
    the canonical ``prefill_flops('mamba2')`` for the sequential path
    (pinned by the regression suite), which is why the accounting gap is
    the broadcast encoding rather than a missing model term.
    """
    _validate('mamba2', T, batch)
    return (2.0 * mamba2_reference_gemm_macs(cfg, T, batch)
            + 2.0 * mamba2_broadcast_encoded_macs(cfg, T, batch))


def mamba2_reference_gemm_macs(cfg, T: int, batch: int = 1) -> int:
    """Exact GEMM+conv MACs executed by the reference ``Mamba2Mixer``.

    Mirrors ``ops_mamba.Mamba2Mixer.forward_functional`` on the default
    sequential scan: the four projections, the depthwise causal short conv,
    and the per-step ``baddbmm`` readout ``C @ S^T``.  The state update
    (``decay * S + dt * x (x) B``) and the output gate are elementwise, so
    they are *not* in this GEMM ledger (the same convention as KDA's
    broadcast-encoded term).

    The chunked SSD path and a non-reference backend have a different
    operator boundary, so they raise ``NotImplementedError`` instead of
    being silently priced with the sequential model.
    """
    _validate('mamba2', T, batch)
    if int(getattr(cfg, 'mamba_chunk_size', 0)) != 0:
        raise NotImplementedError(
            'flops_audit: the chunked Mamba-2 scan has a different '
            'quadratic-intra-chunk / associative-transition ledger; call '
            'measure_forward() to measure it instead.')
    backend = str(getattr(cfg, 'mamba_backend', 'reference'))
    if backend != 'reference':
        raise NotImplementedError(
            f'flops_audit: mamba_backend={backend!r} is not the reference '
            f'implementation; call measure_forward() instead.')
    d = cfg.d_model
    d_inner = cfg.mamba_expand * d
    headdim = cfg.mamba_headdim
    d_state = cfg.mamba_d_state
    ngroups = cfg.mamba_ngroups
    nheads = d_inner // headdim
    dt_rank_cfg = cfg.mamba_dt_rank
    if isinstance(dt_rank_cfg, str):
        dt_rank = max(1, (d + 15) // 16)
    else:
        dt_rank = int(dt_rank_cfg)
    B, T = batch, T
    in_proj = B * T * d * (2 * d_inner)
    conv = B * T * d_inner * cfg.mamba_d_conv
    x_proj = B * T * d_inner * (dt_rank + 2 * ngroups * d_state)
    dt_proj = B * T * dt_rank * nheads
    scan_readout = B * T * nheads * headdim * d_state
    out_proj = B * T * d_inner * d
    return (in_proj + conv + x_proj + dt_proj + scan_readout + out_proj)


def csa_reference_gemm_macs(cfg, T: int, batch: int = 1) -> int:
    """Exact GEMM MACs executed by ``CSAHybridLayer`` at length ``T``.

    Mirrors ``ops_csa.naive_csa``: per-token projections on the padded
    sequence, the lightning-indexer scoring over ALL compressed blocks
    (dense, then masked — not the ideal causal count), the sparse core
    over a fixed ``topk``-wide gather, and the dense sliding window.
    """
    _validate('csa', T, batch)
    d, c, dc = cfg.d_model, cfg.csa_c, cfg.csa_dc
    nh, nIh, cI = cfg.csa_nh, cfg.csa_nIh, cfg.csa_cI
    m, topk, sw, B = cfg.csa_m, cfg.csa_topk, cfg.csa_sliding_window, batch
    Tp = T + (-T) % m
    nb = Tp // m
    proj = B * (Tp * d * (4 * c + 2 * cI + dc)
                + Tp * dc * (c * nh + cI * nIh)
                + Tp * d * nIh + T * nh * c * d)
    indexer = B * (Tp * nb * cI * nIh + Tp * nb * nIh)
    core = B * 2 * Tp * topk * c * nh if nb > 0 else 0
    window = B * 2 * Tp * sw * c * nh if sw > 0 else 0
    return proj + indexer + core + window


def hca_reference_gemm_macs(cfg, T: int, batch: int = 1) -> int:
    """Exact GEMM MACs executed by ``HCAHybridLayer`` at length ``T``.

    Mirrors ``ops_hca.naive_hca``: per-token projections on the padded
    sequence, dense shared-KV attention over ALL compressed blocks, and
    the dense sliding window.  With ``n_blocks == 1`` PyTorch may fold the
    value contraction into a mul+sum (one ``Tp * c * nh`` term short);
    tests use ``n_blocks >= 2``.
    """
    _validate('hca', T, batch)
    d, c, dc = cfg.d_model, cfg.hca_c, cfg.hca_dc
    nh, m2, sw, B = cfg.hca_nh, cfg.hca_m2, cfg.hca_sliding_window, batch
    Tp = T + (-T) % m2
    nb = Tp // m2
    proj = B * (Tp * d * (2 * c + dc) + Tp * dc * c * nh + T * nh * c * d)
    core = B * 2 * Tp * nb * c * nh if nb > 0 else 0
    window = B * 2 * Tp * sw * c * nh if sw > 0 else 0
    return proj + core + window


def reference_gemm_macs(kind: str, cfg, T: int, batch: int = 1) -> int:
    """Exact executed GEMM+TRSM MACs for the reference layer ``kind``."""
    if kind == 'kda':
        return kda_reference_gemm_macs(cfg, T, batch)
    if kind == 'mamba2':
        return mamba2_reference_gemm_macs(cfg, T, batch)
    if kind == 'csa':
        return csa_reference_gemm_macs(cfg, T, batch)
    if kind == 'hca':
        return hca_reference_gemm_macs(cfg, T, batch)
    _validate(kind, T, batch)
    raise AssertionError('unreachable')


def probe_config(**overrides) -> HybridConfig:
    """Default audit geometry (d_model=64, 3:1:1-style operator shapes)."""
    base = dict(
        d_model=64, n_heads_qk=2, n_heads_v=4, head_dim_k=32, head_dim_v=32,
        kda_chunk_size=64, csa_m=16, csa_topk=4, csa_nh=2, csa_c=16,
        csa_dc=16, csa_nIh=2, csa_cI=8, csa_sliding_window=16,
        hca_m2=32, hca_nh=2, hca_c=16, hca_dc=16, hca_sliding_window=16,
    )
    base.update(overrides)
    return HybridConfig(**base)


def audit_layer(kind: str, cfg, T: int, batch: int = 1, *,
                device: str = 'cpu', seed: int = 0) -> dict:
    """Measure one reference layer and compare against both ledgers.

    Returns a dict with the measured ledger, the exact reference model,
    the canonical analytic number, the KDA path (when relevant), and the
    special-op counts.  No assertions here: the regression suite owns the
    contracts.
    """
    _validate(kind, T, batch)
    torch.manual_seed(seed)
    layer = build_mixer(kind, cfg).to(device).eval()
    x = torch.randn(batch, T, cfg.d_model, device=device)
    _, ledger = measure_forward(layer, x)
    try:
        model_macs = reference_gemm_macs(kind, cfg, T, batch)
    except NotImplementedError:
        model_macs = None
    result = {
        'kind': kind,
        'T': T,
        'batch': batch,
        'ledger': ledger,
        'canonical_flops': canonical_flops(kind, cfg, T, batch),
        'reference_model_macs': model_macs,
        'reference_gemm_macs': ledger.gemm_macs,
        'reference_gemm_flops': ledger.gemm_flops,
    }
    if kind == 'kda':
        result['path'] = kda_path(cfg, T)
        result['executed_flops'] = kda_executed_flops(cfg, T, batch)
        result['broadcast_encoded_macs'] = kda_broadcast_encoded_macs(
            cfg, T, batch)
    elif kind == 'mamba2':
        result['path'] = mamba2_path(cfg)
        if result['path'] == 'sequential':
            result['canonical_terms'] = mamba2_canonical_terms(
                cfg, T, batch)
            result['executed_flops'] = mamba2_executed_flops(cfg, T, batch)
            result['broadcast_encoded_macs'] = (
                mamba2_broadcast_encoded_macs(cfg, T, batch))
    if model_macs is not None:
        result['model_matches_measurement'] = (
            model_macs == ledger.gemm_macs)
        result['model_minus_measured_macs'] = model_macs - ledger.gemm_macs
    else:
        result['model_matches_measurement'] = None
        result['model_minus_measured_macs'] = None
    canon = result['canonical_flops']
    result['reference_tax_vs_canonical'] = (
        ledger.gemm_flops / canon if canon else float('nan'))
    return result


def _format_table(rows: list) -> str:
    lines = []
    header = (f"{'kind':>4} {'T':>7} {'path':>9} {'canonical':>12} "
              f"{'ref GEMM':>12} {'ref/canon':>9} {'special':>9}")
    lines.append(header)
    lines.append('-' * len(header))
    for row in rows:
        special = sum(row['ledger'].special_ops.values())
        lines.append(
            f"{row['kind']:>4} {row['T']:>7} {row.get('path', '-'):>9} "
            f"{row['canonical_flops'] / 1e6:>10.3f}M "
            f"{row['reference_gemm_flops'] / 1e6:>10.3f}M "
            f"{row['reference_tax_vs_canonical']:>9.3f} "
            f"{special:>9d}")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Measured GEMM-work audit of the reference operators.')
    parser.add_argument('--kinds', default='kda,mamba2,csa,hca',
                        help='comma-separated layer kinds (default: all)')
    parser.add_argument('--seq-lens', default='32,64,128,256',
                        help='comma-separated sequence lengths')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    kinds = [k.strip() for k in args.kinds.split(',') if k.strip()]
    lens = [int(t) for t in args.seq_lens.split(',') if t.strip()]
    cfg = probe_config()
    rows = []
    for T in lens:
        for kind in kinds:
            rows.append(audit_layer(kind, cfg, T, batch=args.batch,
                                    device=args.device, seed=args.seed))
    print(_format_table(rows))
    print()
    for row in rows:
        if row['kind'] not in ('kda', 'mamba2'):
            continue
        print(f"{row['kind'].upper()} T={row['T']:>4} ({row['path']}): "
              f"canonical={row['canonical_flops'] / 1e6:.3f}M "
              f"executed={row['executed_flops'] / 1e6:.3f}M "
              f"ref_gemm={row['reference_gemm_flops'] / 1e6:.3f}M "
              f"broadcast_encoded="
              f"{2 * row['broadcast_encoded_macs'] / 1e6:.3f}M")
    print()
    for row in rows:
        if row['ledger'].special_ops:
            parts = ', '.join(f'{k}={v}' for k, v in
                              sorted(row['ledger'].special_ops.items()))
            print(f"specials {row['kind']:>4} T={row['T']:>4}: {parts}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
