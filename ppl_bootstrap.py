"""Unit-level bootstrap inference for held-out perplexity evaluation.

Why this module?
----------------
``train_lm_autodl.evaluate_lm_loss`` reports a point estimate: the
token-weighted mean NLL, exponentiated to perplexity. That is enough to
rank runs, but not enough to answer the question a reader of the results
table actually asks: *is the difference between two variants larger than
the sampling noise of this evaluation corpus?* The standard failure modes
are:

* comparing two models by checking whether their confidence intervals
  overlap — two CIs can overlap while the paired difference is clearly
  significant, because the dominant variance term (corpus difficulty) is
  shared and cancels in the paired view;
* resampling raw tokens — tokens inside one sequence are correlated, so
  token-level resampling understates the variance of the corpus mean
  (the effective sample size is closer to the number of sequences than
  to the number of tokens);
* re-using the same evaluation set to pick between recipes and then
  reporting a point estimate as if it were a population quantity.

This module implements the standard resampling machinery (Efron's
percentile bootstrap; Koehn's paired evaluation variant; Davison &
Hinkley §4.4 for the centered hypothesis test) at the level of evaluation
UNITS — one unit is one scored sequence/window, summarized by the
sufficient statistic ``(nll_sum, n_tokens)``:

* :func:`bootstrap_ppl_ci` — percentile CI for one model's corpus
  perplexity, resampling whole units and re-pooling their tokens;
* :func:`paired_bootstrap_ppl` — paired comparison of two models scored
  on the SAME units: CI for the mean-NLL difference and the PPL ratio,
  plus a two-sided centered bootstrap p-value;
* :func:`throughput_summary` — the matching cost metric: prefill
  tokens/s measured over the same evaluation loop that produced the
  perplexity (evaluation is a pure-prefill workload);
* :func:`normalize_units` / :func:`unit_nll_stats` — validation and
  point statistics shared by all of the above.

The primitives are pure Python (no numpy/scipy) and draw from a LOCAL
``random.Random(seed)`` instance: the same seed gives a bit-identical
report, and the global RNG stream is never touched (the same
reproducibility/isolation contract as ``multi_seed_stats``).

What the bootstrap does and does not cover
------------------------------------------
* It quantifies SAMPLING uncertainty over the evaluated corpus. It does
  NOT model training-seed/optimization variance — pair those with
  ``multi_seed_stats`` on per-seed numbers; the two axes compose (a
  per-seed bootstrap CI, then a t/permutation test across seeds).
* Percentile CIs are reported as-is (no BCa correction), and the p-value
  resolution is ``1 / (n_boot + 1)``; with fewer than ~2 units no
  resampling-based uncertainty can be estimated at all (the report marks
  that as ``degenerate`` instead of pretending).
* The CI/PPL mapping is exact: ``exp`` is monotone, so the PPL interval
  is the exponentiated mean-NLL interval (no extra approximation).

Intended use in this repo: the LM evaluation path runs on Kaggle /
AutoDL images where the held-out corpus is a small split, so two
recipes often differ by less than one corpus's sampling noise. Scoring
both variants on the same units and reporting the paired CI is what
keeps a "new best perplexity" claim honest. (Not to be confused with
``kaggle_setup.bootstrap_kaggle_cuda``, which bootstraps the *CUDA
environment*; this module bootstraps a *statistic*.)
"""
from __future__ import annotations

import math
import random

__all__ = [
    'normalize_units',
    'unit_nll_stats',
    'bootstrap_ppl_ci',
    'paired_bootstrap_ppl',
    'throughput_summary',
    'MAX_LOG_PPL',
]

# exp() overflow guard shared with ``train_lm_autodl.evaluate_lm_loss``:
# a mean NLL above this is reported as exp(700) rather than inf so the
# report stays strict-JSON serializable.
MAX_LOG_PPL = 700.0


def _ppl_from_mean_nll(mean_nll: float) -> float:
    return math.exp(min(float(mean_nll), MAX_LOG_PPL))


def _validate_boot_args(alpha: float, n_boot: int,
                        seed: int | None = None) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) \
            or not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f'n_boot must be >= 1, got {n_boot!r}')
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, int)):
        raise ValueError(f'seed must be an int, got {seed!r}')


def _parse_units(units, label: str = 'units') -> list[tuple[float, int]]:
    """Validate evaluation units and return ``[(nll_sum, n_tokens), ...]``.

    Zero-token units are KEPT here.  A paired comparison must drop the
    zero-token positions of BOTH models (their UNION) so the two lists
    keep their index alignment; dropping them per model would shift the
    remaining units against each other.  :func:`normalize_units` drops
    them for the single-model statistics.
    """
    if units is None:
        raise ValueError(f'{label} must be an iterable of '
                         '(nll_sum, n_tokens) pairs or dicts, got None')
    out: list[tuple[float, int]] = []
    for i, unit in enumerate(units):
        if isinstance(unit, dict):
            try:
                nll_sum = unit['nll_sum']
                n_tokens = unit['n_tokens']
            except KeyError as exc:
                raise ValueError(
                    f'unit {i}: missing key {exc} (expected a dict with '
                    f"'nll_sum' and 'n_tokens')") from None
        else:
            try:
                nll_sum, n_tokens = unit
            except (TypeError, ValueError):
                raise ValueError(
                    f'unit {i}: expected an (nll_sum, n_tokens) pair or a '
                    f'dict, got {unit!r}') from None
        if isinstance(nll_sum, bool) or not isinstance(nll_sum, (int, float)):
            raise ValueError(
                f'unit {i}: nll_sum must be a real number, got {nll_sum!r}')
        nll_sum = float(nll_sum)
        if not math.isfinite(nll_sum):
            raise ValueError(
                f'unit {i}: nll_sum is not finite ({nll_sum!r}); refusing '
                f'to bootstrap a corrupted metric')
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int):
            raise ValueError(
                f'unit {i}: n_tokens must be an int, got {n_tokens!r}')
        if n_tokens < 0:
            raise ValueError(f'unit {i}: n_tokens must be >= 0, '
                             f'got {n_tokens}')
        out.append((nll_sum, n_tokens))
    return out


def normalize_units(units) -> list[tuple[float, int]]:
    """Validate evaluation units and return ``[(nll_sum, n_tokens), ...]``.

    Accepts an iterable of ``(nll_sum, n_tokens)`` pairs/tuples/lists or
    of dicts with ``'nll_sum'`` / ``'n_tokens'`` keys (the shape
    ``evaluate_lm_loss(..., return_units=True)`` emits as ``per_unit``).

    Contract:
      * unit order is preserved — two ``evaluate_lm_loss`` calls on the
        same dataset with the same batch size yield unit-aligned lists,
        which is what makes the paired test pair CORRECT units;
      * ``nll_sum`` must be finite (a non-finite value means the
        evaluation already refused to report a metric) and
        ``n_tokens >= 0``;
      * units with zero supervised tokens are DROPPED: they carry no
        information for any corpus-mean statistic and resampling only
        them would produce tokenless replicates (division by zero);
      * at least one informative unit must remain.
    """
    out = _parse_units(units, 'units')
    informative = [(nll, tok) for nll, tok in out if tok > 0]
    if not informative:
        raise ValueError('no informative evaluation units (all missing or '
                         'zero supervised tokens)')
    return informative


def unit_nll_stats(units) -> dict:
    """Point statistics of the corpus-level NLL/perplexity.

    Token-weighted by construction: the estimand is
    ``exp(sum(nll_sum) / sum(n_tokens))`` (the corpus perplexity), NOT the
    mean of per-unit perplexities, which would over-weight short units.
    """
    norm = normalize_units(units)
    total_nll = sum(nll for nll, _ in norm)
    total_tokens = sum(tok for _, tok in norm)
    mean_nll = total_nll / total_tokens
    return {
        'n_units': len(norm),
        'n_tokens': total_tokens,
        'total_nll': total_nll,
        'mean_nll': mean_nll,
        'perplexity': _ppl_from_mean_nll(mean_nll),
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile (type 7; numpy's default convention).

    Deterministic and dependency-free. ``sorted_values`` must be sorted and
    non-empty; ``q`` in [0, 1].
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _resample_means(sums: list[float], toks: list[int], n_boot: int,
                    seed: int) -> list[float]:
    """Bootstrap the token-weighted mean over units.

    Each replicate draws ``n`` unit indices WITH replacement (shared
    indices across models when used for a paired test) and pools the
    resampled units' tokens. Returns ``n_boot`` replicate means in draw
    order; the caller sorts a copy for percentiles.
    """
    rng = random.Random(int(seed))
    n = len(sums)
    reps: list[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        t = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += sums[i]
            t += toks[i]
        reps.append(s / t)
    return reps


def bootstrap_ppl_ci(units, *, alpha: float = 0.05, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Percentile bootstrap CI for one model's corpus perplexity.

    Resamples evaluation units (sequences/windows) with replacement,
    re-pools their tokens and recomputes ``exp(mean NLL)`` per replicate.
    Because ``exp`` is monotone, the perplexity interval is exactly the
    exponentiated mean-NLL interval — no separate approximation.

    With a single informative unit the interval degenerates to the point
    estimate and ``degenerate`` is set: there is no resampling evidence to
    estimate corpus uncertainty from. JSON-serializable report.
    """
    _validate_boot_args(alpha, n_boot, seed)
    norm = normalize_units(units)
    stats = unit_nll_stats(norm)
    n = len(norm)
    sums = [nll for nll, _ in norm]
    toks = [tok for _, tok in norm]
    reps = _resample_means(sums, toks, int(n_boot), seed)
    ordered = sorted(reps)
    lo = _percentile(ordered, float(alpha) / 2.0)
    hi = _percentile(ordered, 1.0 - float(alpha) / 2.0)
    return {
        'n_units': stats['n_units'],
        'n_tokens': stats['n_tokens'],
        'mean_nll': stats['mean_nll'],
        'perplexity': stats['perplexity'],
        'ci_mean_nll': [lo, hi],
        'ci_perplexity': [_ppl_from_mean_nll(lo), _ppl_from_mean_nll(hi)],
        'alpha': float(alpha),
        'n_boot': int(n_boot),
        'seed': int(seed),
        'degenerate': n < 2,
        'method': ('percentile bootstrap over evaluation units '
                   '(token-weighted pooling; unit = scored sequence)'),
    }


def paired_bootstrap_ppl(units_a, units_b, *, alpha: float = 0.05,
                         n_boot: int = 10000, seed: int = 0,
                         check_alignment: bool = True) -> dict:
    """Paired bootstrap comparison of two models on the same units.

    ``units_a`` / ``units_b`` are the per-unit ``(nll_sum, n_tokens)``
    lists of model A and model B evaluated on the SAME corpus in the SAME
    order (e.g. two ``evaluate_lm_loss(..., return_units=True)`` reports,
    or a quantized vs fp16 decode scoring over the same windows). Every
    bootstrap replicate draws ONE index vector and applies it to both
    models, so the corpus-difficulty variance cancels — that pairing is
    the whole point.

    Reported quantities (all JSON-safe):

    * ``delta_mean_nll = mean_nll(a) - mean_nll(b)`` — positive means
      model A has HIGHER NLL, i.e. A is worse;
    * ``ci_delta_mean_nll`` — percentile CI from the uncentered paired
      replicates;
    * ``ppl_ratio_a_over_b = exp(delta_mean_nll)`` and its CI (monotone
      transform of the delta interval; >1 means A is worse);
    * ``p_value`` — two-sided CENTERED bootstrap hypothesis test
      (Davison & Hinkley §4.4): model A's per-unit NLL sums are shifted
      by a token-proportional constant so the two pooled means coincide
      exactly, the shared-index resampling is repeated, and the p-value
      is the fraction of centered replicates whose absolute delta is at
      least the observed absolute delta (``+1`` correction). Identical
      inputs give exactly ``p = 1.0``;
    * ``significant`` / ``worse_model`` — decision at ``alpha`` and the
      direction; both are ``None`` (``p_value`` too) when fewer than two
      informative units remain, since no resampling distribution exists.

    ``check_alignment`` (default True) verifies that unit token counts
    match pairwise — a mismatch means the two evaluations did not see the
    same data and the pairing would be wrong.
    """
    _validate_boot_args(alpha, n_boot, seed)
    raw_a = _parse_units(units_a, 'units_a')
    raw_b = _parse_units(units_b, 'units_b')
    if len(raw_a) == len(raw_b):
        # Same corpus in the same order: drop the zero-token positions of
        # BOTH models (their UNION).  Dropping them per model would let
        # the two lists drift out of index alignment — with A's zero-token
        # unit at index 1 and B's at index 2, A's third unit would be
        # paired with B's second one, and ``check_alignment`` cannot see
        # it because the token counts still match pairwise.
        keep = [i for i, (ua, ub) in enumerate(zip(raw_a, raw_b))
                if ua[1] > 0 and ub[1] > 0]
        a = [raw_a[i] for i in keep]
        b = [raw_b[i] for i in keep]
        if not a:
            raise ValueError(
                'no informative evaluation units (all missing or zero '
                'supervised tokens) for either model')
    else:
        # Different raw unit counts: the two evaluations did not see the
        # same corpus, so no alignment can be established. Do NOT filter
        # zero-token units per model here: that can make the filtered
        # lengths coincide by accident and pair unrelated units.
        raise ValueError(
            f'paired bootstrap needs the same number of raw units for both '
            f'models, got {len(raw_a)} vs {len(raw_b)}; score both models '
            f'on the same corpus with the same batch size/max_batches')
    if len(a) != len(b):
        raise ValueError(
            f'paired bootstrap needs the same number of informative units '
            f'for both models, got {len(a)} vs {len(b)}; score both models '
            f'on the same corpus with the same batch size/max_batches')
    if check_alignment:
        for i, (ua, ub) in enumerate(zip(a, b)):
            if ua[1] != ub[1]:
                raise ValueError(
                    f'unit {i}: token counts differ ({ua[1]} vs {ub[1]}); '
                    f'the two evaluations did not see the same data, so '
                    f'pairing would be invalid (pass check_alignment=False '
                    f'to override at your own risk)')

    stats_a = unit_nll_stats(a)
    stats_b = unit_nll_stats(b)
    delta = stats_a['mean_nll'] - stats_b['mean_nll']
    total_tokens = stats_a['n_tokens']
    # Center model A in SUM space: subtract shift * n_tokens_i / total so
    # that sum(centered_a) == sum(b) exactly. This is the token-weighted
    # analogue of subtracting the mean per-unit difference.
    shift = stats_a['total_nll'] - stats_b['total_nll']
    centered_a = [(nll - shift * (tok / total_tokens), tok)
                  for nll, tok in a]

    rng = random.Random(int(seed))
    n = len(a)
    a_sums = [nll for nll, _ in a]
    b_sums = [nll for nll, _ in b]
    toks = [tok for _, tok in a]
    c_sums = [nll for nll, _ in centered_a]
    reps: list[float] = []
    hits = 0
    abs_delta = abs(delta)
    for _ in range(int(n_boot)):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sb = sc = 0.0
        t = 0
        for i in idx:
            sa += a_sums[i]
            sb += b_sums[i]
            sc += c_sums[i]
            t += toks[i]
        mean_a = sa / t
        mean_b = sb / t
        reps.append(mean_a - mean_b)
        if abs((sc / t) - mean_b) >= abs_delta:
            hits += 1
    ordered = sorted(reps)
    lo = _percentile(ordered, float(alpha) / 2.0)
    hi = _percentile(ordered, 1.0 - float(alpha) / 2.0)
    if n < 2:
        p_value = None
        significant = None
        worse = None
    else:
        p_value = (hits + 1) / (int(n_boot) + 1)
        significant = bool(p_value <= float(alpha))
        worse = ('a' if delta > 0 else 'b') if significant else None
    return {
        'n_units': n,
        'n_pairs': n,
        'n_tokens_a': stats_a['n_tokens'],
        'n_tokens_b': stats_b['n_tokens'],
        'mean_nll_a': stats_a['mean_nll'],
        'mean_nll_b': stats_b['mean_nll'],
        'perplexity_a': stats_a['perplexity'],
        'perplexity_b': stats_b['perplexity'],
        'delta_mean_nll': delta,
        'ci_delta_mean_nll': [lo, hi],
        'ppl_ratio_a_over_b': _ppl_from_mean_nll(delta),
        'ci_ppl_ratio': [_ppl_from_mean_nll(lo), _ppl_from_mean_nll(hi)],
        'p_value': p_value,
        'p_value_resolution': 1.0 / (int(n_boot) + 1),
        'significant': significant,
        'worse_model': worse,
        'alpha': float(alpha),
        'n_boot': int(n_boot),
        'seed': int(seed),
        'degenerate': n < 2,
        'direction': ('delta_mean_nll = mean_nll(model_a) - '
                      'mean_nll(model_b); positive => model_a is worse'),
        'method': ('paired percentile bootstrap over shared evaluation '
                   'units + two-sided centered bootstrap hypothesis test '
                   '(Davison & Hinkley 1997, §4.4)'),
    }


def throughput_summary(*, n_tokens: int, elapsed_s: float,
                       n_sequences: int | None = None,
                       phase: str = 'prefill') -> dict:
    """Measured evaluation throughput, labelled with its measurement scope.

    Evaluation of an LM is a pure-prefill workload (full-sequence forward
    passes), so ``tokens_per_second`` here IS the prefill throughput of
    the evaluated model on this host. It is an END-TO-END number: data
    loading/collation and Python-loop overhead are included, it is not a
    kernel microbenchmark, and it is only comparable across runs on the
    same machine (the repo's accounting-honesty policy).
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int):
        raise ValueError(f'n_tokens must be an int, got {n_tokens!r}')
    if n_tokens < 0:
        raise ValueError(f'n_tokens must be >= 0, got {n_tokens}')
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)):
        raise ValueError(f'elapsed_s must be a real number, got {elapsed_s!r}')
    elapsed = float(elapsed_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(f'elapsed_s must be a positive finite number, '
                         f'got {elapsed_s!r}')
    if n_sequences is not None:
        if isinstance(n_sequences, bool) or not isinstance(n_sequences, int):
            raise ValueError(
                f'n_sequences must be an int, got {n_sequences!r}')
        if n_sequences < 0:
            raise ValueError(f'n_sequences must be >= 0, got {n_sequences}')
    return {
        'phase': phase,
        'n_tokens': n_tokens,
        'n_sequences': n_sequences,
        'elapsed_s': elapsed,
        'tokens_per_second': n_tokens / elapsed,
        'sequences_per_second': (
            (n_sequences / elapsed) if n_sequences is not None else None),
        'ms_per_sequence': (
            (1000.0 * elapsed / n_sequences) if n_sequences else None),
        'note': ('end-to-end evaluation-loop timing (data loading + '
                 'full-sequence forward); compare only same-machine runs, '
                 'and do not read it as a kernel benchmark'),
    }
