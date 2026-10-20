# Shape-polymorphic RMSNorm reference (`ops_norm`)

## Motivation

DeepSeek-V4 §2.3.3 specifies an RMSNorm on each query head and on the single
compressed-KV head. The fused operators implemented that formula inline in
`ops_csa._qk_normalize`:

```python
x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
```

The inline form is correct for the operator call sites, but it is not a
reusable *reference*: it only normalizes the last axis, has no affine
parameters, and the same formula would have to be copied by any new consumer
(e.g. a future LM norm or a test harness). `ops_norm.py` extracts the
primitive and pins its semantics.

## Formula and conventions

```
rms_norm(x) = x / sqrt(mean(x**2) + eps)
```

* `eps` is added **inside** the square root (paper convention), default
  `RMS_NORM_EPS = 1e-6`.
* The implementation uses `torch.rsqrt`, exactly like the operators, so
  `_qk_normalize(x, 'rms')` remains **bit-identical** to the old inline
  expression.
* The reduction spans the trailing `normalized_shape` axes. `None` (default)
  means the last axis; an `int` is promoted to a 1-tuple, so
  `normalized_shape=4` and `(4,)` are identical.
* `weight` / `bias` are optional and must have exactly
  `normalized_shape`; they are applied after normalization
  (`y * weight + bias`), mirroring `torch.nn.RMSNorm`.
* Computation stays in `x.dtype`: there is no hidden upcast. Pass a
  `float64` tensor for a high-precision reference.
* The zero vector maps to zero for `eps > 0` (the stabilizer sits inside the
  denominator, so `0 * rsqrt(eps) == 0`).

## Shape polymorphism

Arbitrary leading axes are supported and the operation is a pure per-row map,
so regrouping leading axes cannot change values:

```python
y_4d = rms_norm(x.reshape(2, 3, 5, 7))
y_3d = rms_norm(x.reshape(6, 5, 7))
assert torch.equal(y_4d.reshape(6, 5, 7), y_3d)
```

Empty leading axes (e.g. `(0, D)`) pass through, which matters for the
operators' `T == 0` branches. Non-contiguous inputs are accepted.

## API

```python
from ops_norm import RMS_NORM_EPS, rms_norm, rms_norm_

rms_norm(x, eps=1e-6, weight=None, bias=None, normalized_shape=None)
rms_norm_(x, ...)  # in-place, returns x, no autograd
```

`rms_norm_` requires sole ownership of `x` (never a cache view or a tensor
read elsewhere) and raises if autograd is enabled and `x.requires_grad`.
The elementwise chain is identical to `rms_norm`, so results are bit-equal.

Validation is explicit: non-floating-point input, negative/NaN/non-numeric
`eps`, empty or mismatched `normalized_shape`, zero-size normalized axes, and
mismatched affine shape/dtype/device all raise rather than silently producing
NaNs or broadcasting surprises.

## Integration

`ops_csa._qk_normalize` / `_qk_normalize_` now delegate the `'rms'` branch to
`ops_norm.rms_norm` / `rms_norm_` with `RMS_NORM_EPS`. Every consumer
(`ops_hca`, `ops_fused`, `ops_decoding_cache`, `run_decoding`) automatically
uses the shared primitive. The `'l2'` branch is unchanged.

## Tests

`test_ops_norm.py` pins:

1. exact agreement with the inline formula over ranks 1–4;
2. leading-axis regrouping, empty leading axes, non-contiguous inputs;
3. multi-axis `normalized_shape` and affine weight/bias;
4. bit-exact in-place equivalence;
5. dtype preservation (fp32/fp64/bf16) and zero-vector handling;
6. `torch.autograd.gradcheck` plus finite weight/bias gradients;
7. every validation path;
8. `_qk_normalize('rms')` bit-equality with the primitive and inline formula.

```bash
pytest -q test_ops_norm.py
# or
python3 test_ops_norm.py
```

The existing end-to-end check (`test_csa_hca_qk_norm_mode_rms` in
`run_correctness.py`) continues to compare `naive_hca` against a manual
RMSNorm pipeline, so the delegation is also covered by `ci_gate.py`.
