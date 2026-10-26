"""Counterfactual state ablation + causality audit for hybrid decode.

The incremental decode path threads three state families through a stack:
the recurrent memory state (KDA / Mamba-2 / RWKV-7), the memory layer's
short-conv / token-shift lookback, and one CSA/HCA cache per retrieval
layer.  The equivalence tests compare the cached path against a
full-sequence forward, but an equivalence test is **vacuous** if the
integration silently bypasses a state family (a cache that is never read
still produces a matching output when the suffix is short).  This module
adds the missing negative controls:

* ``decode_state_ablation`` runs the same weights and the same token
  stream in three arms per state family.  ``full`` is the ordinary
  incremental decoder; ``at_prefill`` drops the family once after the
  prefill (a counterfactual branch); ``every_step`` drops it before each
  decoded token (a stateless decoder).  A family is load-bearing iff the
  suffix output changes beyond ``tol``, so a rename / refactor that
  disconnects a cache is caught instead of hiding behind an exact match.
* ``decode_causality_audit`` checks that the prefix is unchanged when
  future tokens change, compares the incrementally produced prefix with a
  one-shot forward, and requires the perturbation to actually change the
  suffix (the non-vacuous control).

Both functions are measurement helpers for tests and experiments; they do
not train, benchmark, or modify weights.  ``test_tp_parallel`` drives them
on the 3:1:1 KDA hybrid geometry and reuses the future-perturbation
contract on the streaming TP reference.
"""

from __future__ import annotations

import math

import torch

DECODE_STATE_PARTS = ('memory_state', 'memory_lookback',
                      'csa_cache', 'hca_cache')
DROP_MODES = ('at_prefill', 'every_step')


def _validate_decoder(decoder):
    if not callable(getattr(decoder, 'reset', None)) \
            or not callable(getattr(decoder, 'forward', None)):
        raise TypeError(
            'decode_state_ablation: decoder must expose reset() and '
            'forward(x) (run_decoding.HybridDecoding is the reference).')
    model = getattr(decoder, 'model', None)
    if model is None or not hasattr(model, 'memory_layers'):
        raise TypeError(
            'decode_state_ablation: decoder.model must be a hybrid stack '
            'with a memory_layers list.')
    return model


def _validate_inputs(decoder, x):
    _validate_decoder(decoder)
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f'decode_state_ablation: x must be a torch.Tensor, got '
            f'{type(x).__name__}.')
    if x.dim() != 3:
        raise ValueError(
            f'decode_state_ablation: x must be [B, T, d], got '
            f'{tuple(x.shape)}.')
    if x.shape[1] < 2:
        raise ValueError(
            'decode_state_ablation: at least two tokens are needed for a '
            f'prefill + decode audit, got T={x.shape[1]}.')


def _validate_tol(tol, caller: str):
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError(
            f'{caller}: tol must be a real number, got {tol!r}.')
    try:
        value = float(tol)
    except OverflowError:
        raise ValueError(
            f'{caller}: tol must be representable as a finite float, '
            f'got {tol!r}.') from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f'{caller}: tol must be finite and >= 0, got {tol!r}.')
    return value


def available_state_parts(decoder) -> dict:
    """Which state families the decoder actually owns."""
    model = _validate_decoder(decoder)
    return {
        'memory_state': len(model.memory_layers) > 0,
        'memory_lookback': len(model.memory_layers) > 0,
        'csa_cache': len(getattr(decoder, '_csa_caches', ())) > 0,
        'hca_cache': len(getattr(decoder, '_hca_caches', ())) > 0,
    }


def _resolve_parts(decoder, parts):
    available = available_state_parts(decoder)
    if parts is None:
        parts = tuple(p for p in DECODE_STATE_PARTS if available[p])
    if isinstance(parts, str):
        parts = (parts,)
    parts = tuple(parts)
    if not parts:
        raise ValueError('decode_state_ablation: parts must be non-empty.')
    unknown = [p for p in parts if p not in DECODE_STATE_PARTS]
    if unknown:
        raise ValueError(
            f'decode_state_ablation: unknown part(s) {unknown}; choose '
            f'from {list(DECODE_STATE_PARTS)}.')
    missing = [p for p in parts if not available[p]]
    if missing:
        raise ValueError(
            f'decode_state_ablation: decoder does not own {missing}; '
            f'available parts are '
            f'{[p for p in DECODE_STATE_PARTS if available[p]]}.')
    return parts


def drop_decode_state(decoder, part: str) -> None:
    """Drop one state family in place (counterfactual / negative control).

    ``memory_state`` clears ``decoder.model._kda_state`` (the stacked
    recurrent state, regardless of the memory route);
    ``memory_lookback`` calls every memory layer's ``reset_conv_state``;
    ``csa_cache`` / ``hca_cache`` replace the per-layer cache objects with
    ``None`` so the next step allocates a fresh (empty) cache.
    """
    if part not in DECODE_STATE_PARTS:
        raise ValueError(
            f'drop_decode_state: part={part!r} must be one of '
            f'{list(DECODE_STATE_PARTS)}.')
    available = available_state_parts(decoder)
    if not available[part]:
        raise ValueError(
            f'drop_decode_state: decoder has no {part!r} state to drop.')
    if part == 'memory_state':
        decoder.model._kda_state = None
    elif part == 'memory_lookback':
        for layer in decoder.model.memory_layers:
            layer.reset_conv_state()
    elif part == 'csa_cache':
        decoder._csa_caches = [None for _ in decoder._csa_caches]
    else:
        decoder._hca_caches = [None for _ in decoder._hca_caches]


def _decode(decoder, x, prefill_len, part=None, mode=None):
    decoder.reset()
    pieces = []
    with torch.no_grad():
        pieces.append(decoder(x[:, :prefill_len]))
        if part is not None and mode == 'at_prefill':
            drop_decode_state(decoder, part)
        for t in range(prefill_len, x.shape[1]):
            if part is not None and mode == 'every_step':
                drop_decode_state(decoder, part)
            pieces.append(decoder(x[:, t:t + 1]))
    return torch.cat(pieces, dim=1)


def _suffix_metrics(reference, observed, start, tol):
    # Compute in float64: casting a finite float64 value (e.g. 1e39) to
    # float32 would turn it into inf and hide a real divergence.
    ref = reference[:, start:].detach().double()
    got = observed[:, start:].detach().double()
    raw_diff = ref - got
    overflowed = not bool(torch.isfinite(raw_diff).all())
    if overflowed:
        # The subtraction can overflow even when both suffixes are finite
        # (e.g. 1e308 vs -1e308). Normalize by the larger magnitude so the
        # per-step mask and relative-L2 remain computable; fields whose
        # original-unit value still exceeds the float range report None.
        scale = max(float(ref.abs().max()), float(got.abs().max()))
        ref_scaled = ref / scale
        got_scaled = got / scale
        diff = (ref_scaled - got_scaled).abs() * scale
    else:
        diff = raw_diff.abs()
    ref_norm = float(torch.linalg.vector_norm(ref))
    diff_norm = float(torch.linalg.vector_norm(raw_diff))
    if math.isfinite(ref_norm) and math.isfinite(diff_norm):
        if ref_norm > 0.0:
            rel_l2 = diff_norm / ref_norm
        else:
            # 0/0 is undefined, and a nonzero difference over a zero
            # reference has no finite relative scale; None is the
            # JSON-safe statement.
            rel_l2 = 0.0 if diff_norm == 0.0 else None
    else:
        # Squaring can overflow the norm even when every element and the
        # elementwise difference are finite (e.g. 1e200 in fp64). The
        # elementwise-difference path above already scales; the
        # norm-overflow case needs the same treatment or a well-defined
        # relative L2 would be reported as None.
        scale = max(float(ref.abs().max()), float(got.abs().max()))
        if scale == 0.0:
            rel_l2 = 0.0
        else:
            scaled_diff_norm = float(torch.linalg.vector_norm(
                (ref / scale) - (got / scale)))
            scaled_ref_norm = float(torch.linalg.vector_norm(ref / scale))
            if scaled_ref_norm > 0.0:
                rel_l2 = scaled_diff_norm / scaled_ref_norm
            else:
                rel_l2 = 0.0 if scaled_diff_norm == 0.0 else None
    if rel_l2 is not None and not math.isfinite(rel_l2):
        rel_l2 = None
    per_step = diff.amax(dim=(0, 2))
    # A NaN survives ``amax`` as NaN, and ``NaN > tol`` is False, so a
    # state family whose removal poisons the suffix with NaN would be
    # certified as NOT load-bearing. A non-finite suffix element is a
    # divergence (and certainly not evidence that the state was unused).
    nonfinite_step = (~torch.isfinite(diff)).any(dim=(0, 2))
    hits = ((per_step > tol) | nonfinite_step).nonzero()
    prefix_diff = (reference[:, :start].detach().double()
                   - observed[:, :start].detach().double()).abs()

    def _finite(value):
        return value if value is not None and math.isfinite(value) else None

    return {
        'max_abs_diff': (_finite(float(diff.max()))
                         if diff.numel() else 0.0),
        'mean_abs_diff': (_finite(float(diff.mean()))
                          if diff.numel() else 0.0),
        'rel_l2': rel_l2,
        'first_divergence_step': int(hits[0]) if hits.numel() else None,
        'prefix_max_abs_diff': (_finite(float(prefix_diff.max()))
                                if prefix_diff.numel() else 0.0),
    }


def decode_state_ablation(decoder, x, *, prefill_len, parts=None,
                          modes=DROP_MODES, tol=1e-5) -> dict:
    """Run the full decoder and one counterfactual arm per state family.

    Args:
        decoder: ``run_decoding.HybridDecoding``-compatible incremental
            model (``reset()`` + stateful ``forward(x_chunk)``).
        x: ``[B, T, d]`` token stream. All arms see the same weights and
            the same stream; the only difference is the dropped state.
        prefill_len: number of leading tokens decoded before the
            counterfactual drop. Must satisfy ``0 < prefill_len < T``.
        parts: state families to ablate (default: all available).
        modes: ``'at_prefill'``, ``'every_step'``, or both.
        tol: absolute per-element threshold used to declare the first
            divergent decode step.

    Returns a JSON-safe dict with the schedule plus one row per
    ``(part, mode)``. ``load_bearing`` is true iff a suffix element
    exceeds ``tol``; ``prefix_max_abs_diff`` must stay zero because the
    drop happens after the prefill.
    """
    _validate_inputs(decoder, x)
    if isinstance(prefill_len, bool) or not isinstance(prefill_len, int):
        raise TypeError(
            f'decode_state_ablation: prefill_len must be an int, got '
            f'{prefill_len!r}.')
    if not 0 < prefill_len < x.shape[1]:
        raise ValueError(
            f'decode_state_ablation: need 0 < prefill_len < T, got '
            f'prefill_len={prefill_len}, T={x.shape[1]}.')
    _validate_tol(tol, 'decode_state_ablation')
    if isinstance(modes, str):
        modes = (modes,)
    modes = tuple(modes)
    if not modes:
        raise ValueError('decode_state_ablation: modes must be non-empty.')
    unknown_modes = [m for m in modes if m not in DROP_MODES]
    if unknown_modes:
        raise ValueError(
            f'decode_state_ablation: unknown mode(s) {unknown_modes}; '
            f'choose from {list(DROP_MODES)}.')
    parts = _resolve_parts(decoder, parts)
    reference = _decode(decoder, x, prefill_len)
    arms = []
    for part in parts:
        for mode in modes:
            observed = _decode(decoder, x, prefill_len, part, mode)
            row = _suffix_metrics(reference, observed, prefill_len, tol)
            row['part'] = part
            row['mode'] = mode
            row['load_bearing'] = row['first_divergence_step'] is not None
            arms.append(row)
    return {
        'prefill_len': prefill_len,
        'n_decode_tokens': x.shape[1] - prefill_len,
        'n_tokens': x.shape[1],
        'parts': list(parts),
        'modes': list(modes),
        'tol': tol,
        'arms': arms,
    }


def _resolve_prefix_chunks(cut, prefix_chunks):
    if prefix_chunks is None:
        return (cut,)
    chunks = tuple(prefix_chunks)
    if not chunks or any(isinstance(c, bool) or not isinstance(c, int)
                         or c < 1 for c in chunks):
        raise ValueError(
            'decode_causality_audit: prefix_chunks must be a non-empty '
            f'sequence of positive ints, got {prefix_chunks!r}.')
    if sum(chunks) != cut:
        raise ValueError(
            f'decode_causality_audit: prefix_chunks sum to {sum(chunks)} '
            f'but cut={cut}.')
    return chunks


def decode_causality_audit(decoder, x, *, cut=None, prefix_chunks=None,
                           seed=0, tol=1e-5) -> dict:
    """Audit causal prefix reuse and future-token independence.

    Three checks share the same weights and stream:

    1. ``prefix_unchanged``: replacing every token at/after ``cut`` with
       fresh noise must not change any output at ``t < cut``.
    2. ``future_perturbation_nonvacuous``: the same replacement must
       change at least one suffix element (otherwise the probe is
       vacuous and proves nothing).
    3. ``incremental_prefix_matches_one_shot``: a prefix decoded as
       ``prefix_chunks`` (default: one prefill chunk) followed by
       token-by-token output must match the one-shot causal forward on
       ``t < cut``.

    ``prefix_chunks`` can split the prefix into a prefill plus a
    continuation chunk to exercise ``forward_chunk`` rather than only the
    first-prefill path.
    """
    _validate_inputs(decoder, x)
    B, T, _ = x.shape
    if cut is None:
        cut = T // 2
    if isinstance(cut, bool) or not isinstance(cut, int):
        raise TypeError(
            f'decode_causality_audit: cut must be an int, got {cut!r}.')
    if not 0 < cut < T:
        raise ValueError(
            f'decode_causality_audit: need 0 < cut < T, got cut={cut}, '
            f'T={T}.')
    _validate_tol(tol, 'decode_causality_audit')
    chunks = _resolve_prefix_chunks(cut, prefix_chunks)

    with torch.no_grad():
        decoder.reset()
        reference = decoder(x)
        perturbed = x.clone()
        generator = torch.Generator(device=x.device).manual_seed(seed)
        perturbed[:, cut:] = torch.randn(
            perturbed[:, cut:].shape, generator=generator,
            device=x.device, dtype=x.dtype)
        decoder.reset()
        changed = decoder(perturbed)

        decoder.reset()
        pieces = []
        start = 0
        for length in chunks:
            pieces.append(decoder(x[:, start:start + length]))
            start += length
        for t in range(cut, T):
            pieces.append(decoder(x[:, t:t + 1]))
        incremental = torch.cat(pieces, dim=1)

    # Compute the audit differences in float64 for the same reason as
    # ``_suffix_metrics``: a finite float64 output above the float32 range
    # must not be rounded to inf in the difference. A non-finite difference
    # cannot certify a causal match; for the future perturbation it counts
    # as non-vacuous (a poisoned suffix is still a changed suffix), never as
    # a silent no-op from ``NaN > tol`` being False.
    prefix_diff = (reference[:, :cut].detach().double()
                   - changed[:, :cut].detach().double()).abs().max()
    suffix_diff = (reference[:, cut:].detach().double()
                   - changed[:, cut:].detach().double()).abs().max()
    incremental_diff = (reference[:, :cut].detach().double()
                        - incremental[:, :cut].detach().double()).abs().max()
    prefix_value = float(prefix_diff)
    suffix_value = float(suffix_diff)
    incremental_value = float(incremental_diff)
    return {
        'n_tokens': T,
        'cut': cut,
        'prefix_chunks': list(chunks),
        'seed': seed,
        'tol': tol,
        'prefix_max_abs_diff': (prefix_value
                                if math.isfinite(prefix_value) else None),
        'future_perturbation_suffix_max_abs_diff': (
            suffix_value if math.isfinite(suffix_value) else None),
        'incremental_prefix_max_abs_diff': (
            incremental_value if math.isfinite(incremental_value) else None),
        'prefix_unchanged': bool(math.isfinite(prefix_value)
                                 and prefix_value <= tol),
        'future_perturbation_nonvacuous': bool(
            not math.isfinite(suffix_value) or suffix_value > tol),
        'incremental_prefix_matches_one_shot': bool(
            math.isfinite(incremental_value) and incremental_value <= tol),
    }
