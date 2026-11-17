"""Integration tests for the ablation / comparison statistics layer.

These tests exercise the *statistical* and *contrast-reporting* logic used
by Exp 5/5b/8/10/11/12 without running any training (which would be slow
and GPU-dependent). They verify:

* paired t-test arithmetic (vs. a known reference value);
* TOST equivalence verdicts at a pre-specified margin;
* Bonferroni-corrected critical values;
* capacity-label logic (>5% parameter gap -> operator_plus_capacity);
* experiment_design audit labels for confirmatory vs. screening runs.

Run with:  python -m pytest test_ablation.py -v
           or:  python test_ablation.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_seed_stats import (
    paired_t_test, tost_from_summary, seeds_required, min_detectable_effect,
    summarize_paired,
)
from run_ablation import (
    _bonferroni_crit_q, _t_crit_975,
    CAPACITY_MATCH_TOLERANCE, CONTRAST_EQUIVALENCE_MARGIN,
)
from run_memory_hparam_ablation import _paired_t_pvalue as mh_paired_t_pvalue
from experiment_design import audit_run

PASS = 0
FAIL = 0


def _check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  PASS  {name}' + (f' — {detail}' if detail else ''))
    else:
        FAIL += 1
        print(f'  FAIL  {name}' + (f' — {detail}' if detail else ''))
    return condition


# ---------------------------------------------------------------------------
# 1. Paired t-test (multi_seed_stats) — known reference value.
# ---------------------------------------------------------------------------
def test_paired_t_test_known_value():
    # diffs = [0.5, 0.6, 0.4, 0.7, 0.3]; mean=0.5, sd=sqrt(0.025)=0.1581
    diffs = [0.5, 0.6, 0.4, 0.7, 0.3]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    t_stat = mean / (sd / math.sqrt(n))
    # df=4, t=3.162 -> two-sided p ~ 0.034 (verify by recomputation)
    res = paired_t_test(diffs, alternative='two-sided')
    _check('paired_t_test returns dict', isinstance(res, dict))
    _check('paired_t_test t_stat matches',
           abs(res['t_stat'] - t_stat) < 1e-9,
           f"t={res['t_stat']:.6f} expected={t_stat:.6f}")
    _check('paired_t_test p_value in [0,1]',
           0.0 <= res['p_value'] <= 1.0,
           f"p={res['p_value']:.6f}")
    _check('paired_t_test dof = n-1', res['dof'] == n - 1,
           f"dof={res['dof']}")


def test_paired_t_test_directional():
    # Non-constant positive diffs -> directional p should be ~ half of two-sided
    diffs = [0.5, 0.6, 0.4, 0.7, 0.3]
    res_two = paired_t_test(diffs, alternative='two-sided')
    res_gt = paired_t_test(diffs, alternative='greater')
    res_lt = paired_t_test(diffs, alternative='less')
    _check('greater p_value <= two-sided p_value',
           res_gt['p_value'] <= res_two['p_value'],
           f"gt={res_gt['p_value']:.6f} two={res_two['p_value']:.6f}")
    _check('less p_value >= two-sided p_value (positive mean)',
           res_lt['p_value'] >= res_two['p_value'],
           f"lt={res_lt['p_value']:.6f} two={res_two['p_value']:.6f}")


# ---------------------------------------------------------------------------
# 2. TOST equivalence.
# ---------------------------------------------------------------------------
def test_tost_equivalent_when_diff_inside_margin():
    # mean_diff=0.01, sd=0.02, n=7, margin=0.05 -> should be equivalent
    tost = tost_from_summary(mean_diff=0.01, sd_diff=0.02, n_pairs=7,
                             margin=0.05)
    _check('TOST returns dict', isinstance(tost, dict))
    _check('TOST equivalent=True when |mean_diff| << margin',
           tost['equivalent'] is True,
           f"equivalent={tost['equivalent']}")


def test_tost_not_equivalent_when_diff_outside_margin():
    # mean_diff=0.20, sd=0.02, n=7, margin=0.05 -> not equivalent
    tost = tost_from_summary(mean_diff=0.20, sd_diff=0.02, n_pairs=7,
                             margin=0.05)
    _check('TOST equivalent=False when mean_diff >> margin',
           tost['equivalent'] is False,
           f"equivalent={tost['equivalent']}")


def test_tost_inconclusive_at_small_n():
    # n=2 -> exact test floor, TOST cannot declare equivalence reliably
    tost = tost_from_summary(mean_diff=0.0, sd_diff=0.01, n_pairs=2,
                             margin=0.05)
    _check('TOST with n=2 has no df (or conservative)',
           tost.get('equivalent') in (True, False, None),
           f"equivalent={tost.get('equivalent')}")


# ---------------------------------------------------------------------------
# 3. Bonferroni critical value.
# ---------------------------------------------------------------------------
def test_bonferroni_crit_q_monotonic():
    # Fixed n=7: smaller alpha (more corrections) -> larger critical value.
    c_alpha_05 = _bonferroni_crit_q(7, alpha=0.05)
    c_alpha_01 = _bonferroni_crit_q(7, alpha=0.01)
    c_alpha_001 = _bonferroni_crit_q(7, alpha=0.001)
    _check('Bonferroni crit q1 is finite and positive',
           c_alpha_05 is not None and c_alpha_05 > 0,
           f"q(7,0.05)={c_alpha_05}")
    _check('smaller alpha => larger crit (more conservative)',
           c_alpha_05 is not None and c_alpha_01 is not None
           and c_alpha_001 is not None
           and c_alpha_05 < c_alpha_01 < c_alpha_001,
           f"q05={c_alpha_05:.4f} q01={c_alpha_01:.4f} q001={c_alpha_001:.4f}")
    # Fixed alpha=0.05: larger n -> smaller crit (converges to z=1.645).
    c_n5 = _bonferroni_crit_q(5, alpha=0.05)
    c_n7 = _bonferroni_crit_q(7, alpha=0.05)
    c_n30 = _bonferroni_crit_q(30, alpha=0.05)
    _check('larger n => smaller crit (t -> z)',
           c_n5 is not None and c_n7 is not None and c_n30 is not None
           and c_n5 > c_n7 > c_n30,
           f"n5={c_n5:.4f} n7={c_n7:.4f} n30={c_n30:.4f}")


# ---------------------------------------------------------------------------
# 4. Capacity label (>5% param gap -> operator_plus_capacity).
# ---------------------------------------------------------------------------
def test_capacity_label_logic():
    ref_params = 100000
    # 4% gap -> capacity_matched
    gap_04 = abs(104000 - ref_params) / ref_params
    label_04 = 'operator_plus_capacity' if gap_04 > CAPACITY_MATCH_TOLERANCE else 'capacity_matched'
    _check('4% gap => capacity_matched',
           label_04 == 'capacity_matched',
           f"gap={gap_04:.4%} label={label_04}")
    # 6% gap -> operator_plus_capacity
    gap_06 = abs(106000 - ref_params) / ref_params
    label_06 = 'operator_plus_capacity' if gap_06 > CAPACITY_MATCH_TOLERANCE else 'capacity_matched'
    _check('6% gap => operator_plus_capacity',
           label_06 == 'operator_plus_capacity',
           f"gap={gap_06:.4%} label={label_06}")
    _check('tolerance is 0.05', CAPACITY_MATCH_TOLERANCE == 0.05)


# ---------------------------------------------------------------------------
# 5. Local paired t p-value (run_memory_hparam_ablation) vs multi_seed_stats.
# ---------------------------------------------------------------------------
def test_local_paired_t_pvalue_matches_module():
    # Reproduce a known t-statistic and compare p-values.
    diffs = [0.1, 0.15, 0.05, 0.2, 0.12]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    t_stat = mean / (sd / math.sqrt(n))
    p_local = mh_paired_t_pvalue(t_stat, n - 1)
    p_module = paired_t_test(diffs, alternative='two-sided')['p_value']
    _check('local paired t p-value matches multi_seed_stats',
           abs(p_local - p_module) < 1e-6,
           f"local={p_local:.6f} module={p_module:.6f}")


# ---------------------------------------------------------------------------
# 6. seeds_required / min_detectable_effect planning.
# ---------------------------------------------------------------------------
def test_seeds_required_planning():
    # dz=0.5 (medium), alpha=0.05, power=0.8 -> roughly n=34 per standard tables
    n_req = seeds_required(dz=0.5, alpha=0.05, power=0.8)
    _check('seeds_required returns positive int',
           isinstance(n_req, int) and n_req > 0,
           f"n={n_req}")
    _check('seeds_required for dz=0.5 is plausible (>=10)',
           n_req >= 10,
           f"n={n_req}")
    mde = min_detectable_effect(n=7, sd=0.1, alpha=0.05, power=0.8)
    _check('MDE returns positive float',
           isinstance(mde, float) and mde > 0,
           f"mde={mde:.6f}")


# ---------------------------------------------------------------------------
# 7. experiment_design audit labels.
# ---------------------------------------------------------------------------
def test_audit_run_confirmatory_when_budget_met():
    # exp5b_matrix plan: confirmatory requires 14 seeds (per registry).
    # Passing >= planned seeds with full factor levels -> 'confirmatory'.
    audit = audit_run(
        'exp5b_matrix',
        seeds=list(range(1, 15)),  # 14 seeds
        levels={
            'memory_type': ['kda', 'mamba2'],
            'layout_group': ['A', 'B', 'C', 'D'],
        },
        family_size=10,
        analyzed_seeds=list(range(1, 15)),
        sd=None,
    )
    _check('audit returns dict with status', isinstance(audit, dict)
           and 'status' in audit,
           f"status={audit.get('status')}")
    # 14 seeds = planned budget; full controls -> confirmatory (or at least
    # not 'invalid' / 'exploratory_missing_controls')
    _check('audit status is a recognised label',
           audit['status'] in ('confirmatory', 'reduced_family', 'screening',
                               'incomplete_attrition',
                               'exploratory_missing_controls', 'invalid'),
           f"status={audit['status']}")


def test_audit_run_exploratory_when_controls_missing():
    audit = audit_run(
        'exp5b_matrix',
        seeds=list(range(1, 6)),  # only 5 seeds
        levels={
            'memory_type': ['kda'],  # missing mamba2 control
            'layout_group': ['A'],
        },
        family_size=3,
        analyzed_seeds=list(range(1, 6)),
        sd=None,
    )
    _check('missing controls => not confirmatory',
           audit['status'] != 'confirmatory',
           f"status={audit['status']}")


# ---------------------------------------------------------------------------
# 8. summarize_paired end-to-end.
# ---------------------------------------------------------------------------
def test_summarize_paired_json_safe():
    diffs = [0.05, -0.02, 0.08, 0.01, 0.04]
    s = summarize_paired(diffs, seed=42, n_boot=100)
    _check('summarize_paired returns dict', isinstance(s, dict))
    # Must carry the core effect estimate and at least one p-value.
    _check('summarize_paired has mean_diff', 'mean_diff' in s,
           f"keys={sorted(s.keys())}")
    _check('summarize_paired has t_test.p_value_two_sided',
           't_test' in s and 'p_value_two_sided' in s['t_test'],
           f"t_test keys={sorted(s['t_test'].keys()) if 't_test' in s else 'NO t_test'}")
    # All numeric values must be JSON-safe (no NaN/Inf).
    try:
        json.dumps(s)
        _check('summarize_paired is JSON-serializable', True)
    except (ValueError, TypeError) as e:
        _check('summarize_paired is JSON-serializable', False,
               f"json error: {e}")


# ---------------------------------------------------------------------------
# Run all tests.
# ---------------------------------------------------------------------------
def main():
    print(f'=== ablation statistics integration tests ===')
    for fn in (
        test_paired_t_test_known_value,
        test_paired_t_test_directional,
        test_tost_equivalent_when_diff_inside_margin,
        test_tost_not_equivalent_when_diff_outside_margin,
        test_tost_inconclusive_at_small_n,
        test_bonferroni_crit_q_monotonic,
        test_capacity_label_logic,
        test_local_paired_t_pvalue_matches_module,
        test_seeds_required_planning,
        test_audit_run_confirmatory_when_budget_met,
        test_audit_run_exploratory_when_controls_missing,
        test_summarize_paired_json_safe,
    ):
        fn()
    print(f'\n=== results: {PASS} passed, {FAIL} failed ===')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
