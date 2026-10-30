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
  point statistics shared by all of the above;
* :func:`cluster_design_report` — size-only summary of a cluster
  assignment (Kish effective sample size / design effect) for corpora
  whose units are not independent.

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
* Units are assumed independent unless ``cluster_ids`` is supplied. With
  near-duplicate / grouped units, pass the labels from
  ``near_duplicates.near_duplicate_cluster_labels`` so whole clusters are
  resampled; otherwise the interval is too narrow.

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
import numbers
import random
import sys

__all__ = [
    'normalize_units',
    'unit_nll_stats',
    'bootstrap_ppl_ci',
    'paired_bootstrap_ppl',
    'cluster_design_report',
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
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    try:
        alpha_value = float(alpha)
    except OverflowError:
        raise ValueError(
            f'alpha must be in (0, 1), got {alpha!r}') from None
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f'n_boot must be >= 1, got {n_boot!r}')
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, int)):
        raise ValueError(f'seed must be an int, got {seed!r}')


def _divide_by_count(value: float, count: int) -> float:
    """Divide by a token count that may exceed the float range.

    A pooled or resampled token count can be a Python int beyond
    ``float``'s maximum even when every individual unit is representable.
    Scale by a power of two so the quotient stays representable instead of
    letting ``int`` -> ``float`` conversion raise ``OverflowError``.
    """
    try:
        return value / count
    except OverflowError:
        shift = count.bit_length() - 53
        scaled = count >> shift
        return math.ldexp(value / scaled, -shift)


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
        try:
            nll_sum = float(nll_sum)
        except OverflowError:
            # A JSON int beyond the float range is not a finite sum.
            raise ValueError(
                f'unit {i}: nll_sum={nll_sum!r} is not representable as a '
                f'finite float; refusing to bootstrap a corrupted metric'
            ) from None
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
        try:
            float(n_tokens)
        except OverflowError:
            # The pooled corpus mean divides by the token count, which
            # cannot be represented for an unbounded JSON integer.
            raise ValueError(
                f'unit {i}: n_tokens={n_tokens!r} must be representable as '
                f'a finite float') from None
        out.append((nll_sum, n_tokens))
    return out


def _parse_cluster_ids(cluster_ids, n_units: int,
                       label: str = 'cluster_ids') -> list[int | str] | None:
    """Validate one cluster label per evaluation unit.

    Cluster ids identify the independent sampling unit when several
    evaluation units are not independent (for example, a cluster of
    near-duplicate documents). Labels must be exact integers or strings:
    silently coercing ``0.9`` to ``0`` or ``True`` to ``1`` would merge
    unrelated units, the same failure mode the split helpers reject.
    """
    if cluster_ids is None:
        return None
    ids = list(cluster_ids)
    if len(ids) != int(n_units):
        raise ValueError(
            f'{label} has length {len(ids)} but there are {n_units} '
            f'evaluation units; pass one label per scored unit, in the '
            f'same order as the units')
    out: list[int | str] = []
    for i, value in enumerate(ids):
        if value is None or isinstance(value, bool):
            raise ValueError(
                f'{label}[{i}] must be an exact int or str, got {value!r}')
        if isinstance(value, numbers.Integral):
            out.append(int(value))
        elif isinstance(value, str):
            out.append(value)
        else:
            raise ValueError(
                f'{label}[{i}] must be an exact int or str, got {value!r}')
    return out


def _cluster_groups(cluster_ids: list[int | str]) -> list[list[int]]:
    """Unit indices grouped by cluster, in first-appearance order."""
    groups: dict[int | str, list[int]] = {}
    for i, label in enumerate(cluster_ids):
        groups.setdefault(label, []).append(i)
    return list(groups.values())


def _kish_design(groups: list[list[int]], n_units: int) -> dict:
    """Size-only design summary for a cluster-resampled estimate.

    ``effective_n_units = (sum m_i)^2 / sum m_i^2`` is Kish's effective
    sample size for unequal cluster sizes. It assumes perfectly correlated
    units inside a cluster, so it is a conservative lower bound on the
    information the corpus carries; the actual within-cluster correlation
    is not estimated from the outcome. For singleton clusters it equals
    ``n_units`` and the design effect is 1.
    """
    sizes = [len(group) for group in groups]
    total = sum(sizes)
    sum_sq = sum(m * m for m in sizes)
    effective = ((total * total) / sum_sq) if sum_sq else 0.0
    if n_units <= 0:
        effective = 0.0
    design_effect = (n_units / effective) if effective > 0.0 else None
    return {
        'n_units': int(n_units),
        'n_clusters': len(groups),
        'n_singleton_clusters': sum(1 for m in sizes if m == 1),
        'n_multi_unit_clusters': sum(1 for m in sizes if m > 1),
        'cluster_sizes': {
            'min': (min(sizes) if sizes else None),
            'max': (max(sizes) if sizes else None),
            'mean': ((total / len(sizes)) if sizes else None),
            'largest_share': ((max(sizes) / total) if total else None),
        },
        'effective_n_units': effective,
        'design_effect': design_effect,
        'method': ('Kish effective sample size over cluster sizes; assumes '
                   'perfect within-cluster correlation (conservative, not '
                   'estimated from the outcome)'),
    }


def cluster_design_report(cluster_ids) -> dict:
    """Describe the clustering of evaluation units without any outcome.

    ``cluster_ids`` is one label per unit. The report is deterministic,
    reads no metric, and can therefore be recorded before or independently
    of a model evaluation. It is the same block embedded in
    :func:`bootstrap_ppl_ci` / :func:`paired_bootstrap_ppl` when
    ``cluster_ids`` is supplied.
    """
    if cluster_ids is None:
        raise ValueError('cluster_ids must be an iterable of one label per '
                         'unit, got None')
    ids = list(cluster_ids)
    parsed = _parse_cluster_ids(ids, len(ids))
    return _kish_design(_cluster_groups(parsed), len(parsed))


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
    if not math.isfinite(total_nll):
        raise ValueError(
            'total nll_sum is not finite after pooling finite units; the '
            'report would contain inf and cannot be strict-JSON safe')
    mean_nll = _divide_by_count(total_nll, total_tokens)
    if not math.isfinite(mean_nll):
        raise ValueError('mean_nll is not finite after pooling')
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
    # Resampling with replacement can pick one large unit repeatedly, so a
    # direct sum can overflow even when the full-sample total is finite.
    # Rescale by the largest |sum| first and multiply the (bounded) ratio
    # back at the end; this keeps the token-weighted mean finite without
    # changing the ordinary-range estimate beyond round-off.
    sum_scale = max((abs(x) for x in sums), default=0.0)
    if sum_scale == 0.0:
        scaled = [0.0] * n
    else:
        scaled = [x / sum_scale for x in sums]
    reps: list[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        t = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += scaled[i]
            t += toks[i]
        reps.append((_divide_by_count(s, t) * sum_scale)
                    if sum_scale else 0.0)
    return reps


def _resample_cluster_means(sums: list[float], toks: list[int],
                            groups: list[list[int]], n_boot: int,
                            seed: int) -> list[float]:
    """Bootstrap the token-weighted mean by resampling whole clusters.

    A replicate draws ``len(groups)`` clusters WITH replacement and pools
    every unit of each selected cluster. Clusters are the independent
    sampling units; drawing individual units inside a correlated cluster
    would understate the corpus variance.
    """
    rng = random.Random(int(seed))
    n_clusters = len(groups)
    sum_scale = max((abs(x) for x in sums), default=0.0)
    if sum_scale == 0.0:
        scaled = [0.0] * len(sums)
    else:
        scaled = [x / sum_scale for x in sums]
    reps: list[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        t = 0
        for _ in range(n_clusters):
            group = groups[rng.randrange(n_clusters)]
            for i in group:
                s += scaled[i]
                t += toks[i]
        reps.append((_divide_by_count(s, t) * sum_scale)
                    if sum_scale else 0.0)
    return reps


def bootstrap_ppl_ci(units, *, alpha: float = 0.05, n_boot: int = 10000,
                     seed: int = 0, cluster_ids=None) -> dict:
    """Percentile bootstrap CI for one model's corpus perplexity.

    Resamples evaluation units (sequences/windows) with replacement,
    re-pools their tokens and recomputes ``exp(mean NLL)`` per replicate.
    Because ``exp`` is monotone, the perplexity interval is exactly the
    exponentiated mean-NLL interval — no separate approximation.

    ``cluster_ids`` (optional) is one label per unit. When supplied, whole
    clusters are resampled instead of individual units: documents in one
    near-duplicate cluster are not independent observations, so resampling
    them separately would make the interval too narrow. The report then
    also carries ``n_clusters`` and a ``cluster_design`` block with the
    conservative Kish effective sample size. ``cluster_ids=None`` keeps
    the historical unit-level behaviour.

    With a single informative unit (or a single cluster) the interval
    degenerates to the point estimate and ``degenerate`` is set: there is
    no resampling evidence to estimate corpus uncertainty from.
    JSON-serializable report.
    """
    _validate_boot_args(alpha, n_boot, seed)
    raw = _parse_units(units, 'units')
    ids = _parse_cluster_ids(cluster_ids, len(raw))
    if ids is None:
        norm = normalize_units(raw)
        groups = None
    else:
        # Drop the zero-token units of the single model, keeping the
        # cluster labels aligned with the surviving units.
        keep = [i for i, (_, tok) in enumerate(raw) if tok > 0]
        norm = [raw[i] for i in keep]
        if not norm:
            raise ValueError(
                'no informative evaluation units (all missing or zero '
                'supervised tokens)')
        ids = [ids[i] for i in keep]
        groups = _cluster_groups(ids)
    stats = unit_nll_stats(norm)
    n = len(norm)
    sums = [nll for nll, _ in norm]
    toks = [tok for _, tok in norm]
    if groups is None:
        reps = _resample_means(sums, toks, int(n_boot), seed)
    else:
        reps = _resample_cluster_means(sums, toks, groups, int(n_boot), seed)
    ordered = sorted(reps)
    lo = _percentile(ordered, float(alpha) / 2.0)
    hi = _percentile(ordered, 1.0 - float(alpha) / 2.0)
    report = {
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
        'resampling_unit': ('evaluation_unit' if groups is None else 'cluster'),
        'method': ('percentile bootstrap over evaluation units '
                   '(token-weighted pooling; unit = scored sequence)'),
    }
    if groups is not None:
        report.update({
            'n_clusters': len(groups),
            'degenerate': n < 2 or len(groups) < 2,
            'cluster_design': _kish_design(groups, n),
            'method': ('percentile cluster bootstrap over evaluation units '
                       '(token-weighted pooling; whole near-duplicate / '
                       'group clusters resampled with replacement)'),
        })
    return report


def paired_bootstrap_ppl(units_a, units_b, *, alpha: float = 0.05,
                         n_boot: int = 10000, seed: int = 0,
                         check_alignment: bool = True,
                         cluster_ids=None) -> dict:
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

    ``cluster_ids`` (optional) is one label per raw unit. When supplied,
    every replicate resamples whole clusters (the SAME cluster draw for
    both models), because near-duplicate documents inside one cluster are
    not independent evaluation samples. The point estimates are unchanged;
    only the uncertainty/decision is computed at the cluster level.
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
    ids = _parse_cluster_ids(cluster_ids, len(raw_a))
    groups = None
    if ids is not None:
        ids = [ids[i] for i in keep]
        groups = _cluster_groups(ids)
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
    delta_finite = math.isfinite(delta)
    shift_finite = math.isfinite(shift)
    centered_a = ([(nll - shift * _divide_by_count(float(tok), total_tokens),
                    tok)
                   for nll, tok in a] if shift_finite
                  else [(0.0, tok) for _, tok in a])

    rng = random.Random(int(seed))
    n = len(a)
    a_sums = [nll for nll, _ in a]
    b_sums = [nll for nll, _ in b]
    toks = [tok for _, tok in a]
    c_sums = [nll for nll, _ in centered_a]
    # A resample can draw the same large unit ``n`` times, so a finite
    # full-sample total can still overflow an individual replicate sum
    # (the single-model path in ``_resample_means`` already guards this).
    # Below the safe bound keep the historical raw arithmetic bit-for-bit;
    # above it rescale every sum by the same factor and multiply the
    # replicate difference back once. The token-weighted mean and the
    # centered-hit comparison are scale-invariant, so this changes only the
    # overflow case (where the raw path produced ``inf``/``nan``).
    abs_delta = abs(delta)
    max_unit = max((abs(x) for x in a_sums + b_sums + c_sums), default=0.0)
    resample_bound = max_unit * n
    scaled = (not math.isfinite(resample_bound)
              or resample_bound > sys.float_info.max * 0.5)
    if scaled and max_unit > 0.0:
        a_work = [x / max_unit for x in a_sums]
        b_work = [x / max_unit for x in b_sums]
        c_work = [x / max_unit for x in c_sums]
        scaled_abs_delta = abs_delta / max_unit
    else:
        a_work, b_work, c_work = a_sums, b_sums, c_sums
        scaled_abs_delta = abs_delta
    reps: list[float] = []
    hits = 0

    def _draw_indices() -> list[int]:
        if groups is None:
            return [rng.randrange(n) for _ in range(n)]
        n_clusters = len(groups)
        drawn: list[int] = []
        for _ in range(n_clusters):
            drawn.extend(groups[rng.randrange(n_clusters)])
        return drawn

    for _ in range(int(n_boot)):
        idx = _draw_indices()
        sa = sb = sc = 0.0
        t = 0
        for i in idx:
            sa += a_work[i]
            sb += b_work[i]
            sc += c_work[i]
            t += toks[i]
        mean_a = _divide_by_count(sa, t)
        mean_b = _divide_by_count(sb, t)
        reps.append((mean_a - mean_b) * max_unit if scaled else
                    (mean_a - mean_b))
        hit_gap = _divide_by_count(sc, t) - mean_b
        if abs(hit_gap) >= scaled_abs_delta:
            hits += 1
    # A finite difference can still produce an unrepresentable replicate
    # (e.g. the scaled path multiplies back beyond the float range), and
    # an inf/nan endpoint would break the strict-JSON contract.
    ordered = sorted(x for x in reps if math.isfinite(x))
    if ordered:
        lo = _percentile(ordered, float(alpha) / 2.0)
        hi = _percentile(ordered, 1.0 - float(alpha) / 2.0)
    else:
        lo = hi = None
    if n < 2 or (groups is not None and len(groups) < 2) \
            or not (delta_finite and shift_finite):
        # Without a representable observed shift / centered resample there
        # is no valid centered hypothesis test to report.
        p_value = None
        significant = None
        worse = None
    else:
        p_value = (hits + 1) / (int(n_boot) + 1)
        significant = bool(p_value <= float(alpha))
        worse = ('a' if delta > 0 else 'b') if significant else None
    report = {
        'n_units': n,
        'n_pairs': n,
        'n_tokens_a': stats_a['n_tokens'],
        'n_tokens_b': stats_b['n_tokens'],
        'mean_nll_a': stats_a['mean_nll'],
        'mean_nll_b': stats_b['mean_nll'],
        'perplexity_a': stats_a['perplexity'],
        'perplexity_b': stats_b['perplexity'],
        'delta_mean_nll': delta if delta_finite else None,
        'ci_delta_mean_nll': [lo, hi],
        'ppl_ratio_a_over_b': (_ppl_from_mean_nll(delta)
                               if delta_finite else None),
        'ci_ppl_ratio': [_ppl_from_mean_nll(x) if x is not None else None
                         for x in (lo, hi)],
        'p_value': p_value,
        'p_value_resolution': 1.0 / (int(n_boot) + 1),
        'significant': significant,
        'worse_model': worse,
        'alpha': float(alpha),
        'n_boot': int(n_boot),
        'seed': int(seed),
        'degenerate': (n < 2 or (groups is not None and len(groups) < 2)),
        'resampling_unit': ('evaluation_unit' if groups is None else 'cluster'),
        'direction': ('delta_mean_nll = mean_nll(model_a) - '
                      'mean_nll(model_b); positive => model_a is worse'),
        'method': ('paired percentile bootstrap over shared evaluation '
                   'units + two-sided centered bootstrap hypothesis test '
                   '(Davison & Hinkley 1997, §4.4)'),
    }
    if groups is not None:
        report.update({
            'n_clusters': len(groups),
            'cluster_design': _kish_design(groups, n),
            'method': ('paired percentile cluster bootstrap over shared '
                       'evaluation units + two-sided centered bootstrap '
                       'hypothesis test (Davison & Hinkley 1997, §4.4)'),
        })
    return report


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
    try:
        elapsed = float(elapsed_s)
    except OverflowError:
        raise ValueError(f'elapsed_s must be a positive finite number, '
                         f'got {elapsed_s!r}') from None
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(f'elapsed_s must be a positive finite number, '
                         f'got {elapsed_s!r}')
    if n_sequences is not None:
        if isinstance(n_sequences, bool) or not isinstance(n_sequences, int):
            raise ValueError(
                f'n_sequences must be an int, got {n_sequences!r}')
        if n_sequences < 0:
            raise ValueError(f'n_sequences must be >= 0, got {n_sequences}')
    def _finite(value):
        return value if value is not None and math.isfinite(value) else None

    try:
        tokens_per_second = _finite(n_tokens / elapsed)
    except OverflowError:
        tokens_per_second = None
    sequences_per_second = None
    if n_sequences is not None:
        try:
            sequences_per_second = _finite(n_sequences / elapsed)
        except OverflowError:
            sequences_per_second = None
    ms_per_sequence = None
    if n_sequences:
        try:
            ms_per_sequence = _finite(1000.0 * elapsed / n_sequences)
        except OverflowError:
            ms_per_sequence = None
    return {
        'phase': phase,
        'n_tokens': n_tokens,
        'n_sequences': n_sequences,
        'elapsed_s': elapsed,
        'tokens_per_second': tokens_per_second,
        'sequences_per_second': sequences_per_second,
        'ms_per_sequence': ms_per_sequence,
        'note': ('end-to-end evaluation-loop timing (data loading + '
                 'full-sequence forward); compare only same-machine runs, '
                 'and do not read it as a kernel benchmark'),
    }
