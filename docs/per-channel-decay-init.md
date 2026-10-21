# Per-channel log-decay initialization (`kda_decay_init`)

## Motivation

KDA's forget gate is already per-channel,

```
g_t = -softplus(g_up(g_down(x_conv_t))) * kda_decay_scale
```

but at initialization both projections are zero-mean, so every channel
starts with the *same* expected per-step log-decay
(`softplus(0) * 0.1 ~= 0.069`, a half-life of about 10 tokens). RWKV-7
instead writes a deterministic per-channel decay ramp into its decay-branch
bias, giving the recurrent state channels that forget at different rates from
the first step. That multi-timescale prior is the part of RWKV's initialization
recipe that KDA was missing.

`kda_decay_init='log_uniform'` adds an RWKV-style, zero-parameter bias to the
KDA gate pre-activation. It is an **initialization prior**, not a learned
parameter: the low-rank gate still trains normally, and the bias is a
non-persistent buffer rebuilt from the config on checkpoint reload.

## Construction

For channel `c` in a head, the target half-life is log-spaced between
`kda_decay_half_life_min` and `kda_decay_half_life_max` tokens:

```
T_half(c) = T_min * (T_max / T_min) ** (c / (K - 1))
lambda(c) = ln(2) / T_half(c)
bias(c)   = softplus^{-1}(lambda(c) / kda_decay_scale)
```

At initialization `u = 0 + bias(c)`, so the expected log-decay is exactly
`-lambda(c)` and the expected per-step retention is `exp(-lambda(c))`.
The spectrum repeats identically across value heads. `head_dim == 1` gets the
fast endpoint (no span to interpolate).

`softplus^{-1}` is evaluated in float64 (`log(expm1(lambda / scale))`) and
cast once to float32. The result is a `[n_heads_v * head_dim_k]` buffer.

## API

```python
HybridConfig(
    kda_decay_init="log_uniform",   # or "default" (historical zero bias)
    kda_decay_half_life_min=1.0,    # fastest channel, tokens
    kda_decay_half_life_max=1024.0, # slowest channel, tokens
)
```

The LM entry exposes the same knobs:

```bash
python3 train_lm_autodl.py \
  --kda_decay_init log_uniform \
  --kda_decay_half_life_min 1 \
  --kda_decay_half_life_max 1024
```

The values are stored in `HybridConfig`, hence in every checkpoint and run
record. Old checkpoints do not contain these keys; they reconstruct the
historical `default` (all-zero bias), so reload is numerically unchanged.

## Why this is zero-cost

* One `[HV*K]` non-persistent buffer; no parameters, no optimizer state, no
  `state_dict` keys.
* The add is skipped entirely under the default profile, so legacy numerics
  and latency are unchanged.
* Under `log_uniform`, the add is a single broadcast per forward (same
  `[B,T,HV*K]` tensor, no extra matmul).
* Tensor-parallel KDA slices the same buffer along the value-head axis, so
  the sharded and unsharded paths stay identical.

## Interaction with embedding scaling

`LMWithHybrid` scales embeddings by `sqrt(d_model)`, which raises the
variance of `x_conv` and therefore of `g_up(g_down(x_conv))`. The decay
spectrum is an additive offset on the pre-activation, not a rescaling of it:
the *expected* half-life of channel `c` stays at its configured value, while
the input-dependent spread of the gate is preserved. This makes the profile a
clean control for experiments that vary `embed_scale`.

## Experiment design

A matched-budget comparison should hold data, depth, width, optimizer, seed
and `embed_scale` fixed and toggle only:

```
kda_decay_init in {"default", "log_uniform"}
```

Pre-registered expectation: the multi-timescale prior helps long-range
retrieval (MQAR / needle) more than short-context language modelling, because
only the former rewards slow state channels. Falsified if paired-seed gains
are inside the 95% CI of zero on both task families. Report the realized
half-life spectrum and the trained gate spectrum (not just the init) so a
collapsed spectrum is visible.

## Limitations

The bias is frozen: if training drives `g_up` away from the prior, the
spectrum is no longer enforced. That is intentional (it is an init, not a
regularizer). A future variant could anneal or softly regularize the gate
toward the spectrum; that would require a new loss term and is outside this
change.

## Tests

`run_correctness.py::test_kda_per_channel_decay_init` pins the spectrum
math, head repetition, forward injection, zero parameters / state-dict keys,
the default zero-bias path, and config validation.
