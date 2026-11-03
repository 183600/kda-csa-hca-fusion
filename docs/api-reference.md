# API Reference

This document provides a quick reference for the public API of the key modules.
It is intended as a companion to the module docstings, not a replacement.

## Table of Contents

- [KDA (Kimi Delta Attention)](#kda)
- [CSA (Compressed Sparse Attention)](#csa)
- [HCA (Heavily Compressed Attention)](#hca)
- [Mamba-2](#mamba-2)
- [RWKV-7](#rwkv-7)
- [Fused Hybrid Stack](#fused-hybrid-stack)
- [Decoding State](#decoding-state)

---

## KDA

`ops_kda.py` — Kimi-style Delta Attention (recurrent delta-rule with per-channel log-decay).

### Core Functions

#### `naive_recurrent_kda`

```python
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    output_final_state: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    g_clamp_min: float = -10.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

Compute KDA output via the per-step recurrent loop.

**Input Shapes:**
- `q`, `k`: `[B, T, H, K]` (L2-normalized)
- `v`: `[B, T, HV, V]`
- `g`: `[B, T, HV, K]` (log-space per-channel gate)
- `beta`: `[B, T, HV]` (post-sigmoid delta learning rate)
- `initial_state`: `[B, HV, K, V]` or `None`

**Returns:**
- `o`: `[B, T, HV, V]` — output tensor
- `s`: `[B, HV, K, V]` or `None` — final recurrent state (when `output_final_state=True`)

**Constraints:**
- All inputs must share the same device, dtype, and `(B, T)` dimensions.
- `HV` must be divisible by `H` (GVA factor).
- `q`, `k` must be L2-normalized along the last dimension.

#### `naive_chunk_kda`

```python
def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    output_final_state: bool = False,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    g_clamp_min: float = -10.0,
    use_script: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

Chunkwise-parallel variant of KDA. Processes the sequence in `chunk_size` blocks,
matching the recurrent path to fp tolerance.

**Additional Parameters:**
- `chunk_size`: Minimum T to trigger chunk mode. `T < chunk_size` falls back to
  the recurrent path. `chunk_size <= 0` forces always recurrent.

#### `compiled_recurrent_kda`

```python
def compiled_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    output_final_state: bool = False,
    fullgraph: bool = False,
    dynamic: bool = True,
    mode: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

`torch.compile` wrapper around `naive_recurrent_kda`. Cached per signature.
Use `fullgraph=True` for graph breaks to fail loudly.

#### `per_channel_log_decay_bias`

```python
def per_channel_log_decay_bias(
    n_heads: int,
    head_dim: int,
    decay_scale: float,
    *,
    min_half_life: float = 1.0,
    max_half_life: float = 1024.0,
) -> torch.Tensor:
```

Build a per-channel log-decay bias for multi-timescale initialization.
Returns a `[n_heads * head_dim]` tensor suitable for adding to the gate
pre-activation.

---

## CSA

`ops_csa.py` — DeepSeek-V4 Compressed Sparse Attention.

### Core Functions

#### `naive_csa`

```python
def naive_csa(
    H: torch.Tensor,
    W_aKV: torch.Tensor,
    W_bKV: torch.Tensor,
    W_aZ: torch.Tensor,
    W_bZ: torch.Tensor,
    W_KV_idx: torch.Tensor,
    W_Z_idx: torch.Tensor,
    Ba: torch.Tensor,
    Bb: torch.Tensor,
    B_idx: torch.Tensor,
    W_DQ: torch.Tensor,
    W_UQ: torch.Tensor,
    W_IUQ: torch.Tensor,
    W_w: torch.Tensor,
    c: int,
    dc: int,
    c_I: int,
    nIh: int,
    n_blocks: int,
    topk: int,
    m: int,
    sliding_window: int = 0,
    rope_dim: int = 0,
    rope_base: float = 10000.0,
    qk_norm_mode: str = 'l2',
    rope_scale: int = 1,
    rope_type: str = 'linear',
) -> torch.Tensor:
```

Compute CSA attention: compress KV pairs, select top-k blocks, attend.

**Parameters:**
- `H`: `[B, T, D]` input hidden states
- `n_blocks`: number of compressed blocks (T / m)
- `topk`: number of blocks to retrieve per query
- `m`: compression factor (block size)
- `sliding_window`: optional sliding window for recent KV mixing

**Returns:**
- `[B, T, D]` output

#### `csa_lightning_indexer`

```python
def csa_lightning_indexer(
    scores: torch.Tensor,
    topk: int,
    csa_indexer_chunk: int = 0,
    tie_break: str = 'stable_sort',
) -> torch.Tensor:
```

Select top-k blocks per query with tie-breaking support.

**Parameters:**
- `scores`: `[B, T, n_blocks]` attention scores
- `topk`: number of blocks to select
- `csa_indexer_chunk`: for long-context tiling (0 = disabled)
- `tie_break`: `'stable_sort'` or `'first'` or `'last'`

**Returns:**
- `[B, T, topk]` indices of selected blocks

---

## HCA

`ops_hca.py` — DeepSeek-V4 Heavily Compressed Attention.

### Core Functions

#### `naive_hca`

```python
def naive_hca(
    H: torch.Tensor,
    W_KV: torch.Tensor,
    W_Z: torch.Tensor,
    B_pos: torch.Tensor,
    W_DQ: torch.Tensor,
    W_UQ: torch.Tensor,
    c: int,
    dc: int,
    m2: int,
    nh: int,
    n_blocks: int,
    sliding_window: int = 0,
    rope_dim: int = 0,
    rope_base: float = 10000.0,
    qk_norm_mode: str = 'l2',
) -> torch.Tensor:
```

Compute HCA attention: heavy compression + dense MQA over compressed entries.

**Parameters:**
- `H`: `[B, T, D]` input hidden states
- `n_blocks`: number of compressed blocks
- `sliding_window`: optional sliding window for recent context

**Returns:**
- `[B, T, D]` output

---

## Mamba-2

`ops_mamba.py` — SSD mixer from Mamba-2 (arXiv:2405.21060).

### Core Functions

#### `Mamba2Mixer`

```python
class Mamba2Mixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dt_rank: str | int = "auto",
        backend: str = "reference",
    ):
```

Mamba-2 SSD mixer layer.

**Parameters:**
- `d_model`: model dimension (must be divisible by `headdim * ngroups`)
- `d_state`: state dimension per head
- `d_conv`: convolution width
- `expand`: expansion factor for hidden dimension
- `headdim`: head dimension (`d_model // (n_heads * ngroups)`)
- `ngroups`: number of groups sharing B/C matrices
- `dt_rank`: rank of the dt projection (int or "auto")
- `backend`: `'reference'`, `'official'`, or `'auto'`

**Forward Pass:**
```python
x = torch.randn(B, T, d_model)
mixer = Mamba2Mixer(d_model, d_state=64)
y = mixer(x)  # [B, T, d_model]
```

---

## RWKV-7

`ops_rwkv.py` — RWKV-7 "Goose" mixer.

### Core Functions

#### `Rwkv7Mixer`

```python
class Rwkv7Mixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int = 64,
        decay_lora: int = 32,
        a_lora: int = 32,
        gate_lora: int = 64,
        backend: str = "reference",
    ):
```

RWKV-7 TimeMix layer (no First).

**Parameters:**
- `d_model`: model dimension
- `head_dim`: head dimension
- `decay_lora`, `a_lora`, `gate_lora`: LoRA widths for branches
- `backend`: `'reference'` or `'fla'`

**Forward Pass:**
```python
x = torch.randn(B, T, d_model)
rwkv = Rwkv7Mixer(d_model, head_dim=64)
y = rwkv(x)  # [B, T, d_model]
```

---

## Fused Hybrid Stack

`ops_fused.py` — Hybrid attention stack (KDA + CSA + HCA).

### HybridConfig

```python
@dataclass
class HybridConfig:
    d_model: int = 256
    n_heads_qk: int = 4         # H
    n_heads_v: int = 4          # HV (GVA factor = HV // H)
    head_dim_k: int = 64        # K
    head_dim_v: int = 64        # V
    csa_m: int = 16             # CSA compression factor
    csa_topk: int = 8
    csa_c: int = 64
    csa_dc: int = 128
    csa_nh: int = 4
    csa_cI: int = 32
    csa_nIh: int = 2
    csa_sliding_window: int = 16
    hca_m2: int = 64            # HCA compression
    hca_c: int = 64
    hca_dc: int = 128
    hca_nh: int = 4
    hca_sliding_window: int = 16
    memory_type: str = "kda"      # "kda", "mamba2", "rwkv7"
    n_kda: int = 3
    n_csa: int = 1
    n_hca: int = 1
    layout: Optional[tuple[str, ...]] = None
    # ... plus all memory-specific params
```

### HybridAttentionStack

```python
class HybridAttentionStack(nn.Module):
    def __init__(self, cfg: HybridConfig, total_layers: int = 1):
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[list] = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
```

The main hybrid stack.

**Forward Pass:**
```python
from ops_fused import HybridConfig, HybridAttentionStack

cfg = HybridConfig(
    d_model=64, n_heads_qk=2, n_heads_v=2,
    head_dim_k=16, head_dim_v=16,
    memory_type='kda',  # or 'mamba2', 'rwkv7'
)
model = HybridAttentionStack(cfg, total_layers=4).eval()

x = torch.randn(2, 128, 64)
with torch.no_grad():
    y = model(x)  # [2, 128, 64]
```

**Decoding:**
```python
# Incremental decoding with state
y, new_state = model(x, initial_state=prev_state, return_state=True)
```

---

## Decoding State

`ops_decoding_cache.py` — Decoding cache for streaming inference.

### CSADecodingCache

```python
class CSADecodingCache:
    def __init__(
        self,
        n_blocks: int,
        c: int,
        block_size: int,
        n_heads: int,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
    def update(
        self,
        cS: torch.Tensor,  # compressed states from prefill
        cZ: torch.Tensor,
    ) -> torch.Tensor:
    def clear(self):
```

### HCADecodingCache

Similar interface for HCA decoding state.

### Mamba2DecodingState

```python
class Mamba2DecodingState:
    def __init__(
        self,
        d_state: int,
        d_conv: int,
        n_heads: int,
        headdim: int,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
```

Manages Mamba-2 state for incremental decoding.

---

## Quick Reference: Import Paths

```python
# KDA core functions
from ops_kda import (
    naive_recurrent_kda,
    naive_chunk_kda,
    compiled_recurrent_kda,
    scripted_chunk_kda,
    per_channel_log_decay_bias,
)

# Backend adapter
from ops_kda_backend import kda_forward, validate_kda_backend

# CSA
from ops_csa import naive_csa, csa_lightning_indexer

# HCA
from ops_hca import naive_hca

# Mamba-2
from ops_mamba import Mamba2Mixer, build_mamba2_mixer

# RWKV-7
from ops_rwkv import Rwkv7Mixer, build_rwkv7_mixer

# Fused stack
from ops_fused import HybridConfig, HybridAttentionStack, HybridKCHAttention

# Decoding cache
from ops_decoding_cache import CSADecodingCache, HCADecodingCache, Mamba2DecodingState
```

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `KDA_NONFINITE_WARN` | `1` | Control non-finite output warnings in KDA |
| `KDA_BACKEND` | `reference` | Default KDA backend selection |
| `MAMBA_BACKEND` | `reference` | Default Mamba-2 backend selection |
| `RWKV_BACKEND` | `reference` | Default RWKV-7 backend selection |