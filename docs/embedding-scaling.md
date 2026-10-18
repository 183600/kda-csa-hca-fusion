# Embedding scaling (`LMWithHybrid.embed_scale`)

## Motivation

`LMWithHybrid` initialises its token embedding with `std=0.02`. Fed raw into
the first hybrid layer, that gives the value / gate projections pre-activations
of magnitude ~0.02, which under-trains those paths early in training, and it
leaves the CSA / HCA branches (which do **not** renormalise their inputs) with
a very small operating range. The KDA `q`/`k` branches are L2-normalised, so
they are largely insensitive to the input scale.

The standard remedy (GPT-2, *Language Models are Unsupervised Multitask
Learners*, §3.1) is to multiply the looked-up embedding by `sqrt(d_model)`:

```
x = embed(input_ids) * sqrt(d_model)
```

With `d_model=256` this maps the `std=0.02` embeddings to `std≈0.32`, a
healthy input scale for the first layer.

## API

```python
LMWithHybrid(vocab_size, cfg, embed_scale=None)
```

* `embed_scale=None` (default) → `sqrt(cfg.d_model)` (the standard choice).
* `embed_scale=1.0` → recovers the legacy unscaled behaviour **exactly**.
* Any positive finite float is accepted; anything else raises `ValueError`.

The scale is a plain Python `float` attribute (`model.embed_scale`).

## Why this is a hardware-aware fused multiply

* **One vectorised kernel.** A Python-float scalar broadcast over the
  `[B, T, d]` gather output — no second materialised tensor, no extra
  allocation, no per-forward host→device copy.
* **Zero new parameters / buffers.** `state_dict` keys, `count_params` and
  the reproducibility contract are all unchanged (pinned by
  `test_embed_scale.py`).
* **Autocast / dtype-safe.** A Python-float scalar never promotes the tensor
  dtype, so bf16/fp16 autocast behaviour is identical to the unscaled path.

## Prefill path

The scale is applied inside `LMWithHybrid.forward`, so **every** consumer —
training, validation, and the pure-prefill evaluation pass (which records
`prefill_throughput` in the run record) — sees the scaled embeddings. There is
no separate prefill code path to keep in sync.

## Reproducibility / baseline note

Because the default is `sqrt(d_model)`, runs that do not pass `embed_scale`
will differ numerically from pre-scaled runs. To reproduce an unscaled
baseline, construct the model with `embed_scale=1.0`. The scale is **not**
stored in the checkpoint `state_dict` (it is a plain attribute), so when
comparing a scaled model against a stored checkpoint you must construct the
model with the same `embed_scale` that produced the checkpoint.

## Tests

`test_embed_scale.py` pins:

1. the default is `sqrt(d_model)` for several `d_model`;
2. the scale is fused into the hybrid-stack input (`stack_input == embed*scale`);
3. `embed_scale=1.0` reproduces the legacy unscaled forward bit-exactly;
4. `state_dict` keys and `count_params` are unchanged;
5. the pure-prefill path yields finite logits / loss;
6. invalid scales (0, negative, NaN, inf) are rejected.

```bash
pytest -q test_embed_scale.py
# or
python3 test_embed_scale.py
```
