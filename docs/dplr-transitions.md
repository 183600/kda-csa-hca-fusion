# Diagonal-plus-low-rank transitions: certificate and block scan

*Why the KDA/RWKV-7 state update is one algebraic object, how the chunked
path's implicit transition is audited against it, and how a two-phase
(producer / scanner / consumer) block scan makes the associative-scan
path work-efficient.*

## The problem

Every gated linear-attention layer in this repository advances a
key-by-value state with a rank-1 correction to a diagonal decay:

```
KDA:     S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
RWKV-7:  S_t = (Diag(w_t) + kka_t kk_t^T) S_{t-1} + k_t v_t^T
```

Both are **diagonal-plus-low-rank (DPLR)** maps

```
S_t = [Diag(d_t) + U_t V_t^T] S_{t-1} + b_t,        U_t, V_t in R^{K x r}
```

with `r = 1` per step. KDA writes `d_t = exp(g_t)`, `U_t = -beta_t k_t`,
`V_t = exp(g_t) * k_t`; RWKV-7 writes `d_t = w_t`, `U_t = kka_t`,
`V_t = kk_t`. The chunked KDA path composes a whole chunk of these maps
into `Diag(gexp_last) - kexp^T w` through the delta-rule/wY triangular
solve. That identity is load-bearing for `naive_chunk_kda`,
`assoc_scan_chunk_kda`, the compiled chunk path and the decoding paths,
but until now the tests only pinned it through end-to-end outputs: a
sign or ordering bug in `w` would surface as diffuse output drift, not as
a transition-level failure.

## The algebra

`dplr_transition.py` exposes the exact, non-commutative composition rule:

```
M_2 M_1 = [Diag(d_2) + U_2 V_2^T] [Diag(d_1) + U_1 V_1^T]
        = Diag(d_2 d_1)
          + [d_2 U_1 + U_2 (V_2^T U_1),  U_2] [V_1, d_1 V_2]^T
```

The rank of the composition is exactly `r_1 + r_2`, so a `C`-token chunk
is `Diag(...)` plus a rank-`min(C, K)` factor pair. When the rank exceeds
`K`, `compose` applies a reduced QR to `left` and re-expresses the pair in
a `K`-dimensional basis. **Nothing is truncated**: the map is unchanged up
to fp round-off, which `test_compose_compresses_without_changing_the_map`
checks against the dense product.

This is also why a DPLR-aware kernel is attractive: a factor pair stores
`2 K r` scalars instead of `K^2`, and applying it costs
`O(K V + K r V)` instead of `O(K^2 V)`. The certificate reports the
numerical rank so a caller can decide whether the factored form is worth
keeping.

## Certificate: three constructions, one map

`kda_dplr_certificate(k, g, beta, v, chunk_size)` builds the same chunk
transition three ways and reports the disagreement:

| construction | what it computes |
|---|---|
| per-step | `S <- Diag(exp(g_t)) S + U_t V_t^T S + beta_t k_t v_t^T`, one token at a time |
| DPLR chunk | `kda_chunk_factors` composes the per-step factors, then `dense(...)` |
| wY chunk | the production formula `Diag(gexp_last) - kexp^T w`, source `kexp^T u` |

The returned record is JSON-safe and includes `max_abs_map_error`,
`max_abs_state_error`, `max_abs_wy_state_error`, `max_rel_error`,
`rank_max`, `wy_rank_max`, `rank_bound = min(chunk_size, K)`,
`tolerance`, and `passed`. Non-finite inputs return
`{'not_estimable': True, ...}` rather than raising, matching the
repository's JSON contracts.

The certificate uses `g_clamp_min=-inf` internally so it compares raw
gates; it does not modify the caller's tensors and does not touch the
global RNG stream (the initial state is a deterministic sine pattern).

`run_correctness.py::test_kda_dplr_transition_certificate` runs the
certificate over a fixed matrix of seeds, dtypes (fp32/fp64),
divisible/non-divisible `T`, and a strong-decay regime, then checks that
`rank_max == wy_rank_max` and that both respect `min(C, K)`. This is the
regression layer: a change to the wY solve, the gate convention or the
per-step factor ordering now fails with a transition-level error and the
offending seed.

```
python3 dplr_transition.py --dim 32 --chunk-size 16 --tokens 64
python3 dplr_transition.py --plan-chunks 128 --plan-block 8
```

## Two-phase block scan

`assoc_scan_chunk_kda` resolves the per-chunk affine recurrence
`S_{n+1} = M_n S_n + b_n` with a Hillis-Steele prefix scan. That does
`NT log2(NT)` dense `K x K` compositions and keeps all `NT` maps (and the
doubling levels) alive. The blocked mode
(`scan_block_size >= 2`) splits the work into three specialised roles:

1. **producer** — compose each block's map sequentially from its DPLR
   chunk factors (`NT - NT/b` compositions; independent across blocks);
2. **scanner** — Hillis-Steele over the `NT/b` block maps only
   (`(NT/b) log2(NT/b)` compositions);
3. **consumer** — replay each block from its scanned incoming state,
   rebuild the per-chunk maps from the factors and emit the per-chunk
   states (`NT` applications, which the full scan also pays).

The persistent set of dense `K x K` maps is `ceil(NT/b)` instead of `NT`
(one block's maps are transient), and the composition count drops from
about `NT log2(NT)` to `NT + (NT/b) log2(NT/b)`. For `NT = 128`, `b = 8`:

| quantity | full scan | blocked (`b=8`) |
|---|---:|---:|
| compositions | 769 | 161 |
| composition work | 1.0x | **0.21x** |
| persistent transition scalars | `128 K^2` | `16 K^2` (**0.125x**) |

Measured on this CPU host (4 threads, fp32, `K=128`, `V=64`, `C=64`;
median of 3, end-to-end `assoc_scan_chunk_kda`):

| `T` | `NT` | full | blocked | speedup |
|---:|---:|---:|---:|---:|
| 2048 | 32 | 0.079 s | 0.056 s | 1.41x |
| 8192 | 128 | 0.363 s | 0.242 s | 1.50x |
| 16384 | 256 | 0.916 s | 0.541 s | 1.69x |

The CPU loop over blocks is overhead that a fused kernel would remove;
these numbers are evidence that the reduced composition count is real,
not a latency model.

Parity is fp round-off only: `1e-8`/`1e-9` in fp32 and `1e-16`/`1e-17`
in fp64 versus the full scan, with gradient parity at the same level
(`test_blocked_scan_gradient_parity_and_backend_passthrough`). The API is
opt-in (`scan_block_size=0` keeps the historical scan) and is also
reachable through `kda_forward(..., assoc_scan=True, scan_block_size=8)`.

### The "warp specialisation" reading

The three roles above are the CPU analogue of a warp-specialised kernel:
producer warps build block maps, scanner warps run the prefix, consumer
warps carry the state and write outputs. `scan_work_plan` exposes the
trade-off without pretending to be a GPU model:

```
python3 dplr_transition.py --plan-chunks 128 --plan-block 8
```

It returns composition counts, transition storage, and per-chunk producer
vs consumer MAC estimates. `producer_hidden_by_consumer` is the condition
for a producer warp to run ahead of the sequential state chain: with
`C=16` the producer fits inside the consumer; with `C=64` it does not,
so the producer is the bottleneck and a wider block or a leaner factor
build is the lever.

## Limitations

* Rank grows by `r_1 + r_2` per composition and saturates at `K`; the
  QR compression is exact but costs `O(K^2 r)`. The block scan therefore
  keeps dense block maps rather than an ever-growing factor pair.
* The blocked scan changes the association order of the chunk prefix.
  It is not bit-identical to the full scan and must not be used to
  justify a bitwise-reproducibility claim.
* `scan_work_plan` is a dominant-term MAC model. It has no cache,
  launch-overhead or warp-occupancy term; the measured table above is the
  authoritative CPU answer.
* The certificate covers KDA's transition and source. RWKV-7's step
  factors are covered algebraically
  (`test_step_factors_match_the_documented_recurrences`), but RWKV-7 has
  no chunked production path yet, so there is no end-to-end RWKV
  certificate.

## Code map

* `dplr_transition.py` — `compose`, `dense`, `from_kda_step`,
  `from_rwkv7_step`, `kda_chunk_factors`, `kda_dplr_certificate`,
  `scan_work_plan`, CLI.
* `ops_kda._blocked_chunk_states` — two-phase block scan over the
  prepared KDA factors; `assoc_scan_chunk_kda(scan_block_size=...)`
  selects it; `ops_kda_backend.kda_forward` passes the knob through.
* `test_dplr_transition.py` — algebra, composition ranks, per-step and
  chunk identities, certificate matrix, blocked-scan parity/gradients,
  GVA/initial-state, work-plan consistency.
* `run_correctness.py::test_kda_dplr_transition_certificate` — the CI
  regression entry point.
