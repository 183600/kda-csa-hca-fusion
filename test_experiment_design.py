"""Tests for the pre-registered experiment-design and planning module.

The test names follow ``test_*`` so pytest can collect them directly::

    pytest -q test_experiment_design.py

They pin the properties the planning layer is supposed to guarantee:

* every committed plan is internally consistent and its declared
  confirmatory seed budget reaches its own target power;
* planning uses the multiplicity-corrected per-test alpha, not the raw
  family-wise alpha;
* a run is labelled confirmatory only when it reaches the required seed
  count, contains the required controls, AND the analyzed population is the
  complete requested seed block (dropped seeds are attrition, not a pass);
* a pilot SD is extracted from per-seed rows (or from reported
  ``sd_diff`` rows) and never promotes the pilot itself to confirmatory;
* the manifest writer produces strict JSON that can be read back.
"""
from __future__ import annotations

import json
import math

import pytest

import experiment_design as D


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_is_internally_consistent():
    errors = D.validate_plans()
    assert errors == [], errors
    assert set(D.PLANS) == set(D.PLAN_IDS)
    for plan_id, plan in D.PLANS.items():
        assert plan['plan_id'] == plan_id
        # The confirmatory budget must be above the screening budget;
        # otherwise the "screening vs confirmatory" labels are noise.
        assert plan['planned_seeds'] > plan['screening_seeds'] >= 2
        # Every plan must name at least one result artifact.
        assert plan['result_files']
        # A returned plan is a copy: mutating it cannot corrupt the registry.
        copy = D.get_plan(plan_id)
        copy['planned_seeds'] = -1
        assert D.PLANS[plan_id]['planned_seeds'] > 1
    with pytest.raises(D.DesignError):
        D.get_plan('not_a_plan')


def test_validate_plan_rejects_stale_or_incomplete_designs():
    base = D.get_plan('exp4_mqar')
    # 10 is above the screening budget but still below the required 11.
    changed = dict(base, planned_seeds=10)
    errors = D.validate_plan(changed)
    assert any('below the prospective required_seeds' in e for e in errors)
    too_small = dict(base, planned_seeds=2)
    assert any('screening_seeds must be below' in e
               for e in D.validate_plan(too_small))
    missing_control = dict(base, required_levels={'operator': ['softmax']})
    # A required level absent from the factor itself is a registry error.
    missing_control['required_levels'] = {'operator': ['not-an-op']}
    assert any('not in the factor' in e
               for e in D.validate_plan(missing_control))
    bad_factor = dict(base, factors={'operator': {'role': 'treatment',
                                                  'levels': [1, 1]}})
    assert any('duplicate levels' in e for e in D.validate_plan(bad_factor))
    assert D.validate_plan('not a dict')


# ---------------------------------------------------------------------------
# Prospective planning
# ---------------------------------------------------------------------------

def test_planning_uses_corrected_alpha_and_mde():
    plan = D.get_plan('exp4_mqar')
    rep = D.planning_report(plan, seeds=plan['planned_seeds'])
    assert rep['status'] == 'confirmatory_ready'
    assert rep['required_seeds'] is not None
    assert plan['planned_seeds'] >= rep['required_seeds']
    assert rep['planned_power'] >= rep['target_power'] - 1e-9
    assert rep['alpha_per_test'] == pytest.approx(
        rep['alpha_family_wise'] / rep['family_size'])
    # The MDE is reported in endpoint units using the same normal model:
    # (z_crit + z_power) * SD / sqrt(n), with z_crit at the corrected alpha.
    from statistics import NormalDist
    z_crit = NormalDist().inv_cdf(1.0 - rep['alpha_per_test'])
    z_power = NormalDist().inv_cdf(rep['target_power'])
    expected_mde = ((z_crit + z_power) * rep['assumed_sd']
                    / math.sqrt(rep['seeds']))
    assert rep['planned_mde'] == pytest.approx(expected_mde)
    assert rep['planned_mde'] > 0

    # A larger family needs a larger n at the same SESOI and target power.
    small = D.planning_report(plan, seeds=5, family_size=1)
    large = D.planning_report(plan, seeds=5, family_size=20)
    assert large['required_seeds'] >= small['required_seeds']
    assert large['planned_power'] < small['planned_power']

    # The default screening budget is explicitly below the confirmatory n.
    screening = D.planning_report(plan, seeds=plan['screening_seeds'])
    assert screening['status'] == 'underpowered_at_sesoi'
    assert screening['planned_power'] < plan['target_power']


def test_normal_upper_quantile_mirrors_small_tails():
    """Planning must not lose tail precision before 1-tail rounds to 1."""
    assert D._normal_upper_quantile(1e-8) == pytest.approx(
        5.61200124417479, abs=1e-12)
    assert D._normal_upper_quantile(1e-12) == pytest.approx(
        7.034483825301132, abs=1e-12)


def test_two_sided_planning_applies_bonferroni_once():
    # exp5_ratio is a two-sided family: alpha_per_test is the one-sided TAIL
    # probability (alpha/(2*family)).  multi_seed_stats takes a test level,
    # so passing the tail alpha directly would halve it a second time and
    # overstate both the MDE and the required seed count.
    plan = D.get_plan('exp5_ratio')
    rep = D.planning_report(plan)
    assert rep['alternative'] == 'two-sided'
    assert rep['alpha_per_test'] == pytest.approx(
        rep['alpha_family_wise'] / (2.0 * rep['family_size']))
    from statistics import NormalDist
    z_crit = NormalDist().inv_cdf(1.0 - rep['alpha_per_test'])
    z_power = NormalDist().inv_cdf(rep['target_power'])
    expected_mde = ((z_crit + z_power) * rep['assumed_sd']
                    / math.sqrt(rep['seeds']))
    assert rep['planned_mde'] == pytest.approx(expected_mde)
    expected_required = math.ceil(
        ((z_crit + z_power) / rep['standardized_effect_dz']) ** 2)
    assert rep['required_seeds'] == max(2, expected_required)
    assert plan['planned_seeds'] >= rep['required_seeds']
    ncp = rep['standardized_effect_dz'] * math.sqrt(rep['seeds'])
    expected_power = ((1.0 - NormalDist().cdf(z_crit - ncp))
                      + NormalDist().cdf(-z_crit - ncp))
    assert rep['planned_power'] == pytest.approx(expected_power)


def test_lower_tail_planning_aims_the_sesoi_downward():
    # SESOI is a magnitude.  Under a 'less' alternative the effect being
    # powered is -SESOI, so the planned power must match the 'greater'
    # mirror image instead of collapsing to ~0.
    plan = D.get_plan('exp4_mqar')
    greater = D.planning_report(plan, alternative='greater')
    less = D.planning_report(plan, alternative='less')
    assert less['alpha_per_test'] == pytest.approx(greater['alpha_per_test'])
    assert less['required_seeds'] == greater['required_seeds']
    assert less['planned_mde'] == pytest.approx(greater['planned_mde'])
    assert less['planned_power'] == pytest.approx(greater['planned_power'])
    assert less['status'] == 'confirmatory_ready'


def test_planning_rejects_invalid_overrides():
    plan = D.get_plan('exp4_mqar')
    with pytest.raises(D.DesignError):
        D.planning_report(plan, seeds=-1)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, family_size=0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, alpha=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, sd=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, sesoi=0.0)
    with pytest.raises(D.DesignError):
        D.planning_report(plan, alternative='either')
    # n < 2 leaves the inferential quantities undefined, not zero.
    tiny = D.planning_report(plan, seeds=1)
    assert tiny['planned_power'] is None
    assert tiny['planned_mde'] is None


# ---------------------------------------------------------------------------
# Equivalence (TOST) planning
# ---------------------------------------------------------------------------

def test_equivalence_planning_uses_the_tost_doubled_tail():
    """TOST n uses z_{1-beta/2}, its own family, and the TOST margin."""
    from statistics import NormalDist
    rep = D.planning_report('exp4_mqar')
    eq = rep['equivalence']
    assert eq is not None
    assert eq['margin'] == 0.05
    assert eq['family_size'] == 21
    assert eq['alpha_per_test'] == pytest.approx(0.05 / 21)
    assert eq['ci_level'] == pytest.approx(1.0 - 2.0 * 0.05 / 21)
    z_crit = NormalDist().inv_cdf(1.0 - eq['alpha_per_test'])
    z_power = NormalDist().inv_cdf(0.5 * (1.0 + eq['target_power']))
    expected = math.ceil(((z_crit + z_power) * rep['assumed_sd']
                          / eq['margin']) ** 2)
    assert eq['required_seeds_at_zero'] == max(2, expected)
    # At a true difference of zero the forward power is
    # 2*Phi(margin/se - z) - 1.
    se = rep['assumed_sd'] / math.sqrt(eq['seeds'])
    expected_power = 2.0 * NormalDist().cdf(
        eq['margin'] / se - z_crit) - 1.0
    assert eq['planned_power_at_zero'] == pytest.approx(
        max(0.0, min(1.0, expected_power)))
    # The superiority budget is not an equivalence budget.
    assert rep['required_seeds'] < eq['required_seeds_at_zero']
    assert eq['supports_equivalence_at_planned_n'] is False
    assert eq['status'] == 'equivalence_underpowered'
    assert eq['claim_allowed'] is False
    # The (1-2*alpha) interval is wider than the margin at this n, so no
    # observed mean could yield a verdict.
    assert eq['minimum_supportable_margin_at_n'] > eq['margin']

    # The forward helper agrees, and at the required n it reaches target.
    assert D.equivalence_power(
        eq['required_seeds_at_zero'], eq['margin'], rep['assumed_sd'],
        alpha=eq['alpha_family_wise'], family_size=eq['family_size']) \
        >= eq['target_power'] - 1e-9
    # A true difference at the margin cannot be declared equivalent; the
    # power is at most the per-test alpha.
    assert D.equivalence_power(
        eq['required_seeds_at_zero'], eq['margin'], rep['assumed_sd'],
        alpha=eq['alpha_family_wise'], family_size=eq['family_size'],
        true_diff=eq['margin']) <= eq['alpha_per_test'] + 1e-9
    # A zero-variance vector is equivalent exactly when it is inside the
    # strict margin.
    assert D.equivalence_power(5, 0.05, 0.0) == 1.0
    assert D.equivalence_power(5, 0.05, 0.0, true_diff=0.05) == 0.0
    assert D.equivalence_power(5, 0.05, 0.0, true_diff=0.06) == 0.0


def test_json_sized_seed_counts_keep_planning_json_safe():
    huge = 10 ** 400
    beyond = 10 ** 1000
    assert D.equivalence_power(huge, 0.05, 0.1) == 1.0
    assert D.equivalence_power(beyond, 0.05, 0.1) == 1.0
    eq = D.equivalence_plan(margin=0.05, sd=0.1, seeds=huge)
    assert eq['status'] == 'equivalence_supported'
    assert eq['planned_power_at_zero'] == 1.0
    assert 0.0 < eq['minimum_supportable_margin_at_n'] < 1e-150
    json.dumps(eq, allow_nan=False)
    rep = D.planning_report('exp4_mqar', seeds=huge)
    assert rep['equivalence']['status'] == 'equivalence_supported'
    json.dumps(rep, allow_nan=False)
    inter = D.planning_report('exp5b_matrix', seeds=huge)['interaction']
    assert inter['status'] == 'confirmatory_ready'
    assert inter['planned_power'] == 1.0
    json.dumps(inter, allow_nan=False)


def test_json_sized_seed_count_is_rejected_before_enumeration():
    huge = 10 ** 400
    with pytest.raises(D.DesignError, match='too large to enumerate'):
        D.audit_run('exp4_mqar', seeds=huge)
    with pytest.raises(D.DesignError, match='too large to enumerate'):
        D.design_manifest(seeds_by_plan={'exp4_mqar': huge})


def test_equivalence_margin_must_not_exceed_the_sesoi():
    base = D.get_plan('exp4_mqar')
    wide = dict(base, equivalence=dict(base['equivalence'], margin=0.20))
    errors = D.validate_plan(wide)
    assert any('must not exceed the superiority sesoi' in e for e in errors)

    # A plan that CLAIMS a powered equivalence test must actually reach its
    # own TOST requirement.
    claimed = dict(base, equivalence=dict(base['equivalence'],
                                          required_for_claim=True))
    errors = D.validate_plan(claimed)
    assert any('equivalence.required_for_claim' in e for e in errors)
    assert any('required_seeds_at_zero' in e for e in errors)

    # At the actual TOST requirement the same claim validates.
    supported = dict(base, planned_seeds=68,
                     equivalence=dict(base['equivalence'],
                                      required_for_claim=True))
    assert D.validate_plan(supported) == []


def test_equivalence_helpers_reject_malformed_inputs():
    with pytest.raises(D.DesignError):
        D.equivalence_seeds_required(0.0, 0.1)
    with pytest.raises(D.DesignError):
        D.equivalence_seeds_required(0.05, -0.1)
    with pytest.raises(D.DesignError):
        D.equivalence_seeds_required(0.05, 0.1, family_size=0)
    with pytest.raises(D.DesignError):
        D.equivalence_seeds_required(0.05, 0.1, alpha=0.6)
    with pytest.raises(D.DesignError):
        D.equivalence_power(1.5, 0.05, 0.1)
    with pytest.raises(D.DesignError):
        D.equivalence_power(5, 0.05, 0.1, true_diff=float('nan'))
    with pytest.raises(D.DesignError):
        D.equivalence_plan(margin=0.05, sd=0.1, seeds=True)
    with pytest.raises(D.DesignError):
        D.equivalence_plan(margin=0.05, sd=0.1, seeds=5,
                           required_for_claim='yes')
    assert D.equivalence_power(1, 0.05, 0.1) is None


def test_audit_only_allows_equivalence_when_the_budget_supports_it():
    levels = {'operator': ['linear', 'softmax', 'kda']}
    unpowered = D.audit_run('exp4_mqar', seeds=[42 + i for i in range(11)],
                            levels=levels, family_size=5)
    assert unpowered['status'] == 'confirmatory'
    assert unpowered['equivalence_claim_allowed'] is False
    assert 'practical_equivalence_at_margin' not in \
        unpowered['claims_allowed']
    assert any('TOST margin' in w for w in unpowered['design_warnings'])
    assert unpowered['equivalence']['margin'] == 0.05

    powered = D.audit_run('exp4_mqar', seeds=[42 + i for i in range(68)],
                          levels=levels, family_size=5)
    assert powered['equivalence_claim_allowed'] is True
    assert 'practical_equivalence_at_margin' in powered['claims_allowed']
    assert powered['design_warnings'] == []


def test_mechanism_plan_makes_no_equivalence_claim():
    rep = D.planning_report('exp8_mechanism')
    assert rep['equivalence'] is None
    assert rep['equivalence_status'] is None
    audit = D.audit_run('exp8_mechanism', seeds=[42 + i for i in range(14)],
                        levels={'variant': ['baseline', 'no_rope']},
                        family_size=5)
    assert audit['equivalence'] is None
    assert audit['equivalence_claim_allowed'] is None
    assert 'practical_equivalence_at_margin' not in audit['claims_allowed']


def test_mechanism_plan_requires_every_single_factor_arm():
    """Exp 8's prospective audit must not accept a smaller ablation.

    Each one-factor variant names a separate mechanism contrast. Dropping
    one and keeping the remaining rows does not make the run a smaller
    confirmatory ablation; it makes the corresponding mechanism claim
    exploratory because its control is absent.
    """
    arms = ['baseline', 'no_rope', 'topk1', 'dense', 'frozen_indexer']
    complete = D.audit_run(
        'exp8_mechanism', seeds=[42 + i for i in range(14)],
        levels={'variant': arms}, family_size=6)
    assert complete['status'] == 'confirmatory'
    assert complete['missing_required_levels'] == []

    subset = D.audit_run(
        'exp8_mechanism', seeds=[42 + i for i in range(14)],
        levels={'variant': ['baseline', 'no_rope']}, family_size=5)
    assert subset['status'] == 'exploratory_missing_controls'
    assert set(subset['missing_required_levels']) == {
        'variant:topk1', 'variant:dense', 'variant:frozen_indexer'}


def test_runner_equivalence_margins_match_the_registry():
    """The pre-registered TOST margin must equal the runner's default.

    The runners' margins are module constants (overridable only at run
    start). If one is changed without updating the registry, the planning
    report silently describes a different test from the one that runs.
    Parse the literal assignments instead of importing the torch-heavy
    runner modules, so this check works in the stdlib-only planning
    environment.
    """
    import ast
    import pathlib

    def constant(path, name):
        tree = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
        for node in tree.body:
            targets = (node.targets if isinstance(node, ast.Assign) else [])
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        raise AssertionError(f'{name} not found as a literal in {path}')

    assert constant('run_quality.py', 'PAIRWISE_EQUIVALENCE_MARGIN') == \
        D.PLANS['exp4_mqar']['equivalence']['margin']
    assert constant('run_ablation.py', 'CONTRAST_EQUIVALENCE_MARGIN') == \
        D.PLANS['exp5_ratio']['equivalence']['margin']
    assert D.PLANS['exp5_ratio']['equivalence']['margin'] == \
        D.PLANS['exp5b_matrix']['equivalence']['margin'], (
            'run_ablation uses ONE margin for both the ratio and matrix '
            'modes; the two plans must register the same value')


def test_pilot_reports_the_separate_equivalence_requirement():
    report = D.pilot_report('exp4_mqar', _exp4_rows(), family_size=5,
                            sd=0.10)
    assert report['required_seeds_equivalence'] is not None
    assert report['required_seeds_equivalence'] > \
        report['required_seeds_pooled_sd']
    assert report['recommended_seeds'] == \
        report['required_seeds_pooled_sd']
    assert 'equivalence' in report['equivalence_recommendation']
    # A supplied SD has no observed max, so the conservative companion is
    # undefined rather than silently reusing the point estimate.
    assert report['required_seeds_equivalence_conservative'] is None
    # An extracted pilot carries the max per-contrast SD and its
    # conservative TOST requirement.
    observed = D.pilot_report('exp4_mqar', _exp4_rows(), family_size=5)
    assert observed['required_seeds_equivalence_conservative'] >= \
        observed['required_seeds_equivalence']
    assert 'conservative max SD' in observed['equivalence_recommendation']


# ---------------------------------------------------------------------------
# Runtime audit
# ---------------------------------------------------------------------------

def test_audit_labels_screening_confirmatory_and_invalid_runs():
    plan = D.get_plan('exp4_mqar')
    screening = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['screening_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=1)
    assert screening['status'] == 'screening'
    assert screening['claims_allowed'] == [
        'descriptive', 'exploratory_inference_with_caveat']

    confirmatory = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['planned_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=plan['planned_family_size'])
    assert confirmatory['status'] == 'confirmatory'
    assert confirmatory['checks']['family_size_at_least_planned'] is True
    assert 'paired_inference_at_planned_alpha' in \
        confirmatory['claims_allowed']

    # Reaching the required n only because the family shrank is not the
    # pre-registered experiment; it is a confirmatory test of a subset.
    reduced = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(plan['planned_seeds'])],
        levels={'operator': ['linear', 'softmax', 'kda']}, family_size=1)
    assert reduced['status'] == 'reduced_family'
    assert reduced['family_reduced'] is True
    assert reduced['claims_allowed'] == [
        'descriptive', 'paired_inference_for_tested_subset']

    # A family LARGER than the pre-registered one is also not the
    # pre-registered experiment, even when the seed budget is ample at the
    # larger corrected alpha: the extra comparisons were not planned.
    expanded = D.audit_run(
        'exp4_mqar', seeds=[42 + i for i in range(100)],
        levels={'operator': ['linear', 'softmax', 'kda']},
        family_size=plan['planned_family_size'] + 1)
    assert expanded['status'] == 'screening'
    assert expanded['family_expanded'] is True
    assert expanded['family_reduced'] is False
    assert expanded['checks']['family_size_at_most_planned'] is False
    assert expanded['claims_allowed'] == [
        'descriptive', 'exploratory_inference_with_caveat']
    assert any('exceeds the pre-registered planned_family_size' in w
               for w in expanded['design_warnings'])

    missing_control = D.audit_run(
        'exp4_mqar', seeds=[42, 43, 44],
        levels={'operator': ['softmax', 'kda']}, family_size=1)
    assert missing_control['status'] == 'exploratory_missing_controls'
    assert 'operator:linear' in missing_control['missing_required_levels']
    assert missing_control['claims_allowed'] == ['descriptive_only']

    duplicate = D.audit_run('exp4_mqar', seeds=[42, 42],
                            levels={'operator': ['linear', 'softmax']})
    assert duplicate['status'] == 'invalid'
    assert duplicate['claims_allowed'] == []
    assert duplicate['checks']['seeds_unique'] is False

    one_seed = D.audit_run('exp4_mqar', seeds=[42],
                           levels={'operator': ['linear', 'softmax']})
    assert one_seed['status'] == 'invalid'


def test_audit_does_not_require_unobserved_levels_to_fail():
    # A caller that does not pass a factor cannot have it silently pass OR
    # silently fail: the check is recorded as not_evaluated.
    audit = D.audit_run('exp8_mechanism', seeds=[42, 43, 44, 45, 46],
                        family_size=4)
    assert audit['level_checks']['variant']['status'] == 'not_evaluated'
    assert audit['checks']['required_levels_present'] is True
    assert audit['status'] in ('screening', 'confirmatory')


def test_audit_reports_attrition_instead_of_confirmatory():
    """Survivors of a failed seed block are not the pre-registered run."""
    plan = D.get_plan('exp4_mqar')
    requested = [42 + i for i in range(plan['planned_seeds'])]
    levels = {'operator': ['linear', 'softmax', 'kda']}
    family = plan['planned_family_size']

    # Omitting analyzed_seeds keeps the historical requested-block audit.
    prospective = D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                             family_size=family)
    assert prospective['status'] == 'confirmatory'
    assert prospective['analysis_population']['provided'] is False
    assert prospective['n_analyzed'] is None
    assert prospective['planning_at_analyzed_n'] is None

    # Two missing seeds downgrade the claim even though the surviving n
    # still clears the old screening budget.
    analyzed = requested[:-2]
    attrition = D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                            family_size=family, analyzed_seeds=analyzed)
    assert attrition['status'] == 'incomplete_attrition'
    assert attrition['n_seeds'] == len(requested)
    assert attrition['n_analyzed'] == len(analyzed)
    assert attrition['analysis_population']['missing_seeds'] == requested[-2:]
    assert attrition['analysis_population']['n_missing'] == 2
    assert attrition['analysis_population']['attrition_rate'] == 2 / 11
    assert attrition['checks']['complete_case_population'] is False
    assert attrition['claims_allowed'] == [
        'descriptive', 'complete_case_paired_inference_only']
    # Realized power is reported at the analyzed n, separately from the
    # planned power at the requested n.
    assert attrition['planning_at_analyzed_n']['seeds'] == len(analyzed)
    assert attrition['planning']['seeds'] == len(requested)

    # A complete analyzed block is still confirmatory.
    complete = D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                           family_size=family, analyzed_seeds=requested)
    assert complete['status'] == 'confirmatory'
    assert complete['analysis_population']['complete_case'] is True
    assert complete['analysis_population']['missing_seeds'] == []

    # Fewer than two analyzed seeds cannot support paired inference.
    empty = D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                        family_size=family, analyzed_seeds=[42])
    assert empty['status'] == 'incomplete_attrition'
    assert empty['claims_allowed'] == []
    assert empty['checks']['analyzed_seeds_valid'] is False

    # Seeds outside the requested block are reported, not silently added.
    extra = D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                        family_size=family,
                        analyzed_seeds=requested[:-1] + [999])
    assert extra['status'] == 'incomplete_attrition'
    assert extra['analysis_population']['unexpected_seeds'] == [999]
    assert extra['checks']['complete_case_population'] is False

    # An int is a seed COUNT for ``seeds`` but ambiguous for the analyzed
    # population; require the explicit list instead of inventing 42..46.
    with pytest.raises(D.DesignError):
        D.audit_run('exp4_mqar', seeds=requested, levels=levels,
                    analyzed_seeds=5)


def test_common_analyzed_seeds_intersects_survivors_conservatively():
    def row(seeds, **kwargs):
        return {'op': 'kda',
                'per_seed': [{'seed': s, 'final_acc': 0.5, **kwargs}
                             for s in seeds]}

    assert D.common_analyzed_seeds([row([42, 43, 44]),
                                    row([42, 43, 44])]) == [42, 43, 44]
    # A success row may carry an explicit ``error: null`` after a JSON
    # round-trip; only a non-null error means the row is unusable.
    assert D.common_analyzed_seeds(
        [dict(row([42, 43]), error=None)]) == [42, 43]

    # A seed that failed in one row is not part of the complete-case block.
    failed = row([42, 43, 44])
    failed['per_seed'][1]['error'] = 'diverged'
    assert D.common_analyzed_seeds([row([42, 43, 44]), failed]) == [42, 44]

    # A row-level error means no population can be certified.
    assert D.common_analyzed_seeds([row([42, 43]), {'error': 'OOM'}]) == []
    # A missing/malformed per_seed list is conservative too.
    assert D.common_analyzed_seeds([{'op': 'kda'}]) == []
    assert D.common_analyzed_seeds([{'op': 'kda', 'per_seed': 5}]) == []
    # Optional finite-metric filtering keeps non-finite survivors out.
    bad = row([42, 43])
    bad['per_seed'][0]['final_acc'] = float('nan')
    assert D.common_analyzed_seeds(
        [bad, row([42, 43])], finite_keys=('final_acc',)) == [43]
    assert D.common_analyzed_seeds([]) == []
    with pytest.raises(D.DesignError):
        D.common_analyzed_seeds({'op': 'kda'})
    with pytest.raises(D.DesignError):
        D.common_analyzed_seeds(5)


# ---------------------------------------------------------------------------
# Pilot -> power -> confirmatory
# ---------------------------------------------------------------------------

def _exp4_rows(n_seeds=5):
    seeds = [42 + i for i in range(n_seeds)]
    base = [0.50, 0.52, 0.48, 0.51, 0.49][:n_seeds]
    diffs = [0.10, 0.12, 0.08, 0.15, 0.09][:n_seeds]
    rows = []
    for op, deltas in (('softmax', [0.0] * n_seeds), ('kda', diffs)):
        rows.append({
            'op': op, 'task': 'mqar', 'n_kv': 1,
            'per_seed': [
                {'seed': seed, 'final_acc': b + d}
                for seed, b, d in zip(seeds, base, deltas)
            ],
        })
    return rows


def test_pilot_sd_from_per_seed_rows_and_recommendation():
    info = D.pilot_sd_from_rows(_exp4_rows(), group_key='op',
                                reference='softmax',
                                cell_keys=['task', 'n_kv'])
    expected = [0.10, 0.12, 0.08, 0.15, 0.09]
    mean = sum(expected) / len(expected)
    expected_sd = math.sqrt(sum((d - mean) ** 2 for d in expected) / 4)
    assert info['method'] == 'per_seed_paired_differences'
    assert info['n_pairs'] == 1
    assert info['n_differences'] == 5
    assert info['sd_pooled'] == pytest.approx(expected_sd)
    assert info['sd'] == pytest.approx(expected_sd)
    assert info['sd_max'] == pytest.approx(expected_sd)

    report = D.pilot_report('exp4_mqar', _exp4_rows())
    assert report['status'] in ('pilot_reaches_required_n',
                                'pilot_is_screening_only')
    assert report['pilot_is_confirmatory'] is False
    assert report['pilot_seeds'] == 5
    assert report['observed_sd']['method'] == 'per_seed_paired_differences'
    assert report['required_seeds_pooled_sd'] >= 2
    assert report['recommended_seeds'] >= report['required_seeds_pooled_sd']


def test_pilot_report_with_a_large_sd_recommends_more_seeds():
    report = D.pilot_report('exp4_mqar', _exp4_rows(), family_size=5,
                            sd=0.25)
    assert report['observed_sd']['method'] == 'supplied_sd'
    assert report['required_seeds_pooled_sd'] > \
        D.PLANS['exp4_mqar']['screening_seeds']
    assert report['recommended_seeds'] >= report['required_seeds_pooled_sd']
    assert report['pilot_is_confirmatory'] is False


def test_pilot_uses_reported_contrast_sd_rows():
    rows = [
        {'contrast': 'B-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 0.10, 'mean_diff': 0.05},
        {'contrast': 'C-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 0.20, 'mean_diff': -0.05},
    ]
    info = D.pilot_sd_from_rows(rows, group_key='contrast',
                                reference='A', cell_keys=['n_kv'])
    assert info['method'] == 'reported_sd_diff'
    assert info['sd_pooled'] == pytest.approx(0.15)
    assert info['sd_max'] == pytest.approx(0.20)
    report = D.pilot_report('exp5b_matrix', rows, family_size=2)
    assert report['required_seeds_conservative'] >= \
        report['required_seeds_pooled_sd']


def test_pilot_can_exclude_the_memoryless_control():
    rows = _exp4_rows()
    rows.append({
        'op': 'linear', 'task': 'mqar', 'n_kv': 1,
        'per_seed': [{'seed': 42 + i, 'final_acc': 0.05 + 0.04 * i}
                     for i in range(5)],
    })
    all_pairs = D.pilot_sd_from_rows(
        rows, group_key='op', reference='softmax', cell_keys=['task', 'n_kv'])
    no_linear = D.pilot_sd_from_rows(
        rows, group_key='op', reference='softmax',
        cell_keys=['task', 'n_kv'], exclude=['linear'])
    assert all_pairs['n_pairs'] == 2
    assert no_linear['n_pairs'] == 1
    assert no_linear['sd_max'] < all_pairs['sd_max']


def test_pilot_with_zero_variance_is_not_a_license_for_two_seeds():
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.5},
                      {'seed': 43, 'final_acc': 0.5}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.6},
                      {'seed': 43, 'final_acc': 0.6}]},
    ]
    report = D.pilot_report('exp4_mqar', rows)
    assert report['status'] == 'degenerate_zero_sd'
    assert report['pilot_is_confirmatory'] is False


def test_pilot_without_two_common_seeds_is_insufficient():
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.5}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.6}]},
    ]
    assert D.pilot_sd_from_rows(rows, group_key='op', reference='softmax',
                               cell_keys=['task', 'n_kv']) is None
    report = D.pilot_report('exp4_mqar', rows)
    assert report['status'] == 'insufficient_pilot_data'
    assert report['observed_sd'] is None


def test_pilot_sd_from_rows_keeps_extreme_finite_differences_json_safe():
    """Finite 1e308 differences square past the float range.

    The pooled within-pair SD is still representable (1.41e308), so the
    extraction must not turn it into ``inf`` and must remain strict-JSON
    serializable; ordinary pilots keep their historical statistics.
    """
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.0},
                      {'seed': 43, 'final_acc': 0.0}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 1e308},
                      {'seed': 43, 'final_acc': -1e308}]},
    ]
    info = D.pilot_sd_from_rows(rows, group_key='op', reference='softmax',
                                cell_keys=['task', 'n_kv'])
    assert info is not None
    assert math.isfinite(info['sd_pooled'])
    assert math.isfinite(info['sd_max'])
    json.dumps(info, allow_nan=False)

    # Mixed hashable seed types must not abort the extraction on sort().
    mixed = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.0},
                      {'seed': '43', 'final_acc': 0.0}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.1},
                      {'seed': '43', 'final_acc': 0.2}]},
    ]
    info = D.pilot_sd_from_rows(mixed, group_key='op', reference='softmax',
                                cell_keys=['task', 'n_kv'])
    assert info is not None and info['n_differences'] == 2


def test_pilot_sd_skips_unrepresentable_paired_differences():
    # 1e308 - (-1e308) overflows; the contrast cannot enter a finite SD.
    rows = [
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 1e308},
                      {'seed': 43, 'final_acc': 1e308}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': -1e308},
                      {'seed': 43, 'final_acc': -1e308}]},
    ]
    assert D.pilot_sd_from_rows(rows, group_key='op', reference='softmax',
                                cell_keys=['task', 'n_kv']) is None


def test_pilot_reported_sd_median_does_not_overflow():
    rows = [
        {'contrast': 'B-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 1.7e308},
        {'contrast': 'C-A', 'n_kv': 1, 'n_common_seeds': 5,
         'sd_diff': 1.7e308},
    ]
    info = D.pilot_sd_from_rows(rows, group_key='contrast', reference='A',
                                cell_keys=['n_kv'])
    assert info['sd_median'] == 1.7e308
    assert info['sd_pooled'] == 1.7e308
    json.dumps(info, allow_nan=False)


# ---------------------------------------------------------------------------
# Manifest + CLI
# ---------------------------------------------------------------------------

def test_manifest_roundtrip_and_cli(tmp_path):
    manifest = D.design_manifest()
    assert manifest['schema_version'] == '1.0'
    assert manifest['kind'] == 'experiment_design'
    assert {row['plan_id'] for row in manifest['results']} == set(D.PLANS)
    # Strict JSON: nothing in the plan/planning path may be NaN or Inf.
    encoded = json.dumps(manifest, allow_nan=False)
    assert json.loads(encoded)['kind'] == 'experiment_design'

    path = tmp_path / 'design.json'
    D.write_design_manifest(str(path), manifest=manifest)
    with open(path, encoding='utf-8') as fh:
        loaded = json.load(fh)
    assert loaded['results'] == manifest['results']

    # The CLI check path validates the committed registry.
    assert D.main(['--check']) == 0
    # The CLI writes the same envelope.
    out = tmp_path / 'cli_design.json'
    assert D.main(['--plan', 'exp4_mqar', '--json', str(out)]) == 0
    written = json.loads(out.read_text(encoding='utf-8'))
    assert written['kind'] == 'experiment_design'
    assert len(written['results']) == 1
    assert written['results'][0]['plan_id'] == 'exp4_mqar'

    # Pilot CLI reads a result file and emits a planner report.
    pilot_path = tmp_path / 'exp4.json'
    pilot_path.write_text(json.dumps({'metadata': {},
                                      'results': _exp4_rows()}),
                          encoding='utf-8')
    rc = D.main(['--plan', 'exp4_mqar', '--pilot', str(pilot_path)])
    assert rc == 0


def test_manifest_seed_override_is_recorded():
    manifest = D.design_manifest(['exp4_mqar'], seeds_by_plan={
        'exp4_mqar': [42, 43, 44]})
    row = manifest['results'][0]
    assert row['seeds'] == [42, 43, 44]
    assert row['n_seeds'] == 3
    # Three seeds are below the confirmatory budget and stay labelled so.
    assert row['status'] == 'underpowered_at_sesoi'


def test_run_all_wires_the_design_manifest(monkeypatch, tmp_path):
    # run_all is importable without torch (the heavy imports live inside
    # run_all()); the design manifest must be registered as a raw side file
    # and its seed reader must agree with parse_int_env's fallback.
    import run_all
    assert 'results/experiment_design.json' in \
        run_all._EXPERIMENT_OUTPUTS['experiment_design']
    assert 'results/experiment_design.json' in run_all._RAW_SIDE_FILES
    monkeypatch.setenv('MQAR_SEEDS', '3')
    assert run_all._env_seed_list('MQAR_SEEDS', 5) == [42, 43, 44]
    monkeypatch.setenv('MQAR_SEEDS', 'not-an-int')
    assert run_all._env_seed_list('MQAR_SEEDS', 5) == [42, 43, 44, 45, 46]

    # The pre-run writer records the actual budget before the experiments.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MQAR_SEEDS', '3')
    monkeypatch.setenv('ABL_SEEDS', '4')
    monkeypatch.setenv('MECH_SEEDS', '2')
    block = run_all._write_experiment_design_manifest('smoke')
    assert block['path'] == 'results/experiment_design.json'
    assert block['validation_errors'] == []
    assert block['plans']['exp4_mqar']['n_seeds'] == 3
    assert block['plans']['exp5_ratio']['n_seeds'] == 4
    assert block['plans']['exp8_mechanism']['n_seeds'] == 2
    # The pre-run summary surfaces the separate TOST budget, so a reader
    # cannot mistake a superiority budget for an equivalence one.
    assert block['plans']['exp4_mqar']['equivalence_status'] == \
        'equivalence_underpowered'
    assert block['plans']['exp4_mqar']['equivalence_required_seeds'] >= 11
    assert block['plans']['exp8_mechanism']['equivalence_status'] is None
    written = json.loads(
        (tmp_path / 'results' / 'experiment_design.json').read_text(
            encoding='utf-8'))
    assert written['kind'] == 'experiment_design'
    assert {row['plan_id']: row['n_seeds']
            for row in written['results']}['exp4_mqar'] == 3


def test_cpu_skip_slow_seed_floor_is_recorded(monkeypatch, tmp_path):
    # The smoke profile writes 2 seeds; the CPU SKIP_SLOW path raises that
    # floor to 5 just before each runner. The manifest must record the
    # budget that will actually run, not the pre-floor smoke value.
    import run_all
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SKIP_SLOW', '1')
    monkeypatch.setenv('MQAR_SEEDS', '2')
    monkeypatch.setenv('ABL_SEEDS', '2')
    monkeypatch.setenv('MECH_SEEDS', '2')
    monkeypatch.setenv('RECALL_SEEDS', '2')
    for var, default in (('MQAR_SEEDS', 5), ('ABL_SEEDS', 7),
                         ('MECH_SEEDS', 5), ('RECALL_SEEDS', 3)):
        assert run_all._raise_seed_floor(var, default) == 5
    # A malformed value falls back to the default and is then floored.
    monkeypatch.setenv('MQAR_SEEDS', 'not-an-int')
    assert run_all._raise_seed_floor('MQAR_SEEDS', 5) == 5
    block = run_all._write_experiment_design_manifest('smoke')
    assert {plan: row['n_seeds'] for plan, row in block['plans'].items()} \
        == {'exp4_mqar': 5, 'exp5_ratio': 5, 'exp5b_matrix': 5,
            'exp8_mechanism': 5, 'exp9_recall': 5}


def test_malformed_plan_and_audit_inputs_raise_design_errors():
    """Malformed registry/audit inputs must not escape as raw TypeError.

    These are configuration errors, so the public entry points report a
    DesignError (or a validation string) instead of an opaque crash from a
    dict lookup / set key.
    """
    base = D.get_plan('exp4_mqar')
    for factor in (None, [], 'x'):
        plan = dict(base, factors={'operator': factor})
        errors = D.validate_plan(plan)
        assert isinstance(errors, list) and errors
        assert any('not a dict' in e for e in errors)

    for levels in (5, 'softmax', ('softmax',)):
        with pytest.raises(D.DesignError):
            D.audit_run('exp4_mqar', seeds=[42, 43], levels=levels)
    bad_levels = {'operator': 3}
    with pytest.raises(D.DesignError):
        D.audit_run('exp4_mqar', seeds=[42, 43], levels=bad_levels)
    ok = D.audit_run('exp4_mqar', seeds=[42, 43],
                    levels={'operator': ['linear', 'softmax']})
    assert ok['level_checks']['operator']['status'] == 'present'

    for key, value in (('alpha', '0.05'), ('sesoi', None),
                       ('assumed_sd', True), ('target_power', '0.8'),
                       ('planned_family_size', True)):
        plan = dict(base, **{key: value})
        with pytest.raises(D.DesignError):
            D.planning_report(plan)


def test_pilot_extraction_skips_unhashable_cells_groups_and_seeds():
    """A malformed pilot row is skipped, not allowed to abort extraction."""
    rows = [
        {'op': 'softmax', 'task': ['mqa'], 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.2},
                      {'seed': 43, 'final_acc': 0.2},
                      {'seed': 44, 'final_acc': 0.2}]},
        {'op': 'kda', 'task': ['mqa'], 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.3},
                      {'seed': 43, 'final_acc': 0.3},
                      {'seed': 44, 'final_acc': 0.3}]},
        {'op': ['softmax'], 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': 42, 'final_acc': 0.9},
                      {'seed': 43, 'final_acc': 0.9}]},
        {'op': 'kda', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': ['42'], 'final_acc': 0.9},
                      {'seed': 43, 'final_acc': 0.9}]},
        {'op': 'softmax', 'task': 'mqar', 'n_kv': 1,
         'per_seed': [{'seed': [42], 'final_acc': 0.2},
                      {'seed': 43, 'final_acc': 0.2}]},
    ]
    assert D.pilot_sd_from_rows(rows, group_key='op', reference='softmax',
                                cell_keys=['task', 'n_kv']) is None
    # Non-iterable per_seed and non-exact/infinite n_common_seeds are
    # malformed rows too; they must be skipped rather than truncating a
    # float count (which would invent seed pairs) or raising.
    malformed = [
        {'op': 'kda', 'per_seed': 5},
        {'op': 'kda', 'per_seed': {'seed': 42, 'final_acc': 0.5}},
        {'contrast': 'B-A', 'n_kv': 1,
         'n_common_seeds': float('inf'), 'sd_diff': 0.1},
        {'contrast': 'B-A', 'n_kv': 1, 'n_common_seeds': 2.9,
         'sd_diff': 0.1},
        {'contrast': 'B-A', 'n_kv': 1, 'n_common_seeds': True,
         'sd_diff': 0.1},
    ]
    assert D.pilot_sd_from_rows(
        malformed, group_key='op', reference='softmax',
        cell_keys=['n_kv']) is None


def test_planning_mde_and_minimum_margin_are_json_safe_at_float_limit():
    report = D.planning_report('exp4_mqar', seeds=2, sd=1e308)
    assert report['planned_mde'] is None
    assert report['equivalence']['minimum_supportable_margin_at_n'] is None
    json.dumps(report, allow_nan=False)
    assert D._is_number(10 ** 400) is False
    assert D._clean_float(10 ** 400) is None

    # A finite SD can overflow the interaction's sqrt(2) inflation. That
    # must produce a JSON-safe not-estimable record, not a crash.
    inter = D.planning_report(
        'exp5b_matrix', seeds=2, sd=1.5e308)['interaction']
    assert inter['sd_interaction'] is None
    assert inter['required_seeds'] is None
    assert inter['planned_power'] is None
    assert inter['status'] == 'not_estimable'
    assert 'overflowed' in inter['caveat']
    json.dumps(inter, allow_nan=False)


def test_interaction_planning_uses_the_four_cell_variance_budget():
    """A 2x2 interaction is planned at sqrt(2) x the main-effect SD."""
    for plan_id, expected_family in (('exp5b_matrix', 24),
                                     ('exp8_mechanism', 6)):
        rep = D.planning_report(plan_id)
        inter = rep['interaction']
        assert inter is not None
        assert inter['family_size'] == expected_family
        assert inter['alternative'] == 'two-sided'
        assert inter['sd_inflation'] == D.INTERACTION_SD_INFLATION
        assert abs(inter['sd_interaction']
                   - rep['assumed_sd'] * inter['sd_inflation']) < 1e-12
        # The four-cell contrast is strictly less powerful than the
        # two-cell main effect at the same n and SESOI: same budget must
        # not silently cover the interaction.
        assert inter['required_seeds'] > rep['required_seeds']
        assert inter['planned_power'] < rep['planned_power']
        assert inter['supports_interaction_at_planned_n'] is False
        assert inter['status'] == 'underpowered_at_sesoi'
        assert inter['shortfall_seeds'] == (
            inter['required_seeds'] - inter['seeds'])
        json.dumps(rep, allow_nan=False)

    assert D.planning_report('exp4_mqar')['interaction'] is None
    assert D.planning_report('exp4_mqar')['interaction_status'] is None
    assert D.planning_report('exp4_mqar')['interaction_claim_allowed'] is None
    assert D.validate_plans() == []


def test_interaction_audit_labels_without_upgrading_the_main_status():
    levels = {'memory': ['kda', 'mamba2', 'rwkv7'],
              'layout_unit': ['A', 'B', 'C', 'D']}
    rep = D.planning_report('exp5b_matrix')
    unpowered = D.audit_run(
        'exp5b_matrix', seeds=[42 + i for i in range(14)],
        levels=levels, family_size=24)
    # The main-effect design is confirmatory; the interaction claim is not.
    assert unpowered['status'] == 'confirmatory'
    assert unpowered['interaction_claim_allowed'] is False
    assert unpowered['checks']['interaction_budget_supports_claim'] is False
    assert 'interaction_claim_at_sesoi' not in unpowered['claims_allowed']
    assert any('interaction' in w and 'descriptive' in w
               for w in unpowered['design_warnings'])

    enough = rep['interaction']['required_seeds']
    powered = D.audit_run(
        'exp5b_matrix', seeds=[42 + i for i in range(enough)],
        levels=levels, family_size=24)
    assert powered['status'] == 'confirmatory'
    assert powered['interaction_claim_allowed'] is True
    assert 'interaction_claim_at_sesoi' in powered['claims_allowed']
    assert not any('interaction' in w for w in powered['design_warnings'])

    # A plan that *requires* the interaction cannot keep a budget that
    # cannot power it.
    stale = D.get_plan('exp8_mechanism')
    stale['interaction']['required_for_claim'] = True
    errors = D.validate_plan(stale)
    assert any('interaction.required_for_claim is set' in e for e in errors)
    # The four-cell variance can never be below the two-cell one.
    bad = D.get_plan('exp8_mechanism')
    bad['interaction']['sd_inflation'] = 0.5
    assert any('sd_inflation must be >= 1.0' in e
               for e in D.validate_plan(bad))


def test_pilot_reports_the_separate_interaction_budget():
    rows = []
    for group, base in (('KDA-A', 0.5), ('KDA-D', 0.7)):
        rows.append({'group': group, 'n_kv': 1,
                     'per_seed': [
                         {'seed': seed, 'final_acc': acc}
                         for seed, acc in zip(
                             (42, 43, 44),
                             (base, base + 0.05, base - 0.05))]})
    rep = D.pilot_report('exp5b_matrix', rows)
    assert rep['required_seeds_interaction'] is not None
    assert rep['required_seeds_interaction'] >= rep['required_seeds_pooled_sd']
    assert 'four-cell interaction' in rep['interaction_recommendation']
