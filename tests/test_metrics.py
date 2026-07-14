"""Tests for biasscope.metrics, with hand-computed expected values.

Fair case (two groups, identical confusion matrices):
    Group A (n=10): y_true = [1,1,1,1,1,0,0,0,0,0]
                    y_pred = [1,1,1,1,0,0,0,0,0,1]
        TP=4, FN=1, FP=1, TN=4
        positive_rate = (TP+FP)/n = 5/10 = 0.5
        TPR = TP/(TP+FN) = 4/5 = 0.8
    Group B: identical to Group A -> same stats.
    => demographic_parity_diff = 0, equal_opportunity_diff = 0,
       disparate_impact_ratio = 1.0

Biased case (two groups, clearly unequal treatment):
    Group A (privileged, n=12): y_true = [1]*6 + [0]*6
                                 y_pred = [1]*6 + [0]*6  (perfect predictions)
        TP=6, FN=0, FP=0, TN=6
        positive_rate = 6/12 = 0.5
        TPR = 6/6 = 1.0
    Group B (unprivileged, n=8): y_true = [1]*4 + [0]*4
                                  y_pred = [1,1,0,0, 0,0,0,0]
        TP=2, FN=2, FP=0, TN=4
        positive_rate = 2/8 = 0.25
        TPR = 2/4 = 0.5
    Group A is the reference (larger n=12).
    => demographic_parity_diff (B - A) = 0.25 - 0.5 = -0.25
    => equal_opportunity_diff (B - A) = 0.5 - 1.0 = -0.5
    => disparate_impact_ratio = rate(B)/rate(A) = 0.25/0.5 = 0.5

Three-group case (adds Group C, n=5) to exercise pairwise-vs-reference and
max-pairwise-gap logic for >2 groups:
    Group C (n=5): y_true = [1,1,1,0,0]
                   y_pred = [1,0,0,0,0]
        TP=1, FN=2, FP=0, TN=2
        positive_rate = 1/5 = 0.2
        TPR = 1/3 = 0.33333...
    Reference is still Group A (n=12, largest).
    => vs A: demographic_parity_diff = 0.2 - 0.5 = -0.3
             equal_opportunity_diff = 1/3 - 1.0 = -2/3
             disparate_impact_ratio = 0.2 / 0.5 = 0.4
    Max pairwise gap across {A:0.5, B:0.25, C:0.2}:
        max demographic parity gap = 0.5 - 0.2 = 0.3
        max equal opportunity gap (TPRs {1.0, 0.5, 1/3}) = 1.0 - 1/3 = 2/3
        worst-case disparate impact ratio = min/max = 0.2/0.5 = 0.4
"""

import math

import pandas as pd
import pytest

from biasscope.metrics import (
    compute_fairness_metrics,
    compute_group_stats,
    demographic_parity_diff,
    disparate_impact_ratio,
    equal_opportunity_diff,
)


def _fair_df() -> pd.DataFrame:
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    rows = []
    for g in ("A", "B"):
        for yt, yp in zip(y_true, y_pred):
            rows.append({"y_true": yt, "y_pred": yp, "group": g})
    return pd.DataFrame(rows)


def _biased_df() -> pd.DataFrame:
    rows = []
    # Group A: n=12, perfect predictions.
    a_true = [1] * 6 + [0] * 6
    a_pred = [1] * 6 + [0] * 6
    for yt, yp in zip(a_true, a_pred):
        rows.append({"y_true": yt, "y_pred": yp, "group": "A"})
    # Group B: n=8, model under-predicts positives among actual positives.
    b_true = [1, 1, 1, 1, 0, 0, 0, 0]
    b_pred = [1, 1, 0, 0, 0, 0, 0, 0]
    for yt, yp in zip(b_true, b_pred):
        rows.append({"y_true": yt, "y_pred": yp, "group": "B"})
    return pd.DataFrame(rows)


def _three_group_df() -> pd.DataFrame:
    df = _biased_df()
    c_true = [1, 1, 1, 0, 0]
    c_pred = [1, 0, 0, 0, 0]
    rows = [{"y_true": yt, "y_pred": yp, "group": "C"} for yt, yp in zip(c_true, c_pred)]
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def test_group_stats_fair_case():
    stats = compute_group_stats(_fair_df(), "y_true", "y_pred", "group")
    for g in ("A", "B"):
        s = stats[g]
        assert s.n == 10
        assert s.tp == 4
        assert s.fp == 1
        assert s.tn == 4
        assert s.fn == 1
        assert s.positive_rate == pytest.approx(0.5)
        assert s.tpr == pytest.approx(0.8)


def test_group_stats_biased_case():
    stats = compute_group_stats(_biased_df(), "y_true", "y_pred", "group")
    a, b = stats["A"], stats["B"]
    assert (a.tp, a.fp, a.tn, a.fn) == (6, 0, 6, 0)
    assert a.positive_rate == pytest.approx(0.5)
    assert a.tpr == pytest.approx(1.0)

    assert (b.tp, b.fp, b.tn, b.fn) == (2, 0, 4, 2)
    assert b.positive_rate == pytest.approx(0.25)
    assert b.tpr == pytest.approx(0.5)


def test_demographic_parity_diff_fair():
    assert demographic_parity_diff(0.5, 0.5) == pytest.approx(0.0)


def test_demographic_parity_diff_biased():
    assert demographic_parity_diff(0.25, 0.5) == pytest.approx(-0.25)


def test_equal_opportunity_diff_fair():
    assert equal_opportunity_diff(0.8, 0.8) == pytest.approx(0.0)


def test_equal_opportunity_diff_biased():
    assert equal_opportunity_diff(0.5, 1.0) == pytest.approx(-0.5)


def test_equal_opportunity_diff_none_when_undefined():
    assert equal_opportunity_diff(None, 1.0) is None
    assert equal_opportunity_diff(0.5, None) is None


def test_disparate_impact_ratio_fair():
    assert disparate_impact_ratio(0.5, 0.5) == pytest.approx(1.0)


def test_disparate_impact_ratio_biased():
    assert disparate_impact_ratio(0.25, 0.5) == pytest.approx(0.5)


def test_disparate_impact_ratio_zero_privileged_rate():
    assert disparate_impact_ratio(0.1, 0.0) is None


def test_compute_fairness_metrics_fair_case():
    report = compute_fairness_metrics(_fair_df(), "y_true", "y_pred", "group")
    # Both groups tie on size (n=10 each); reference is whichever groupby
    # picks first among ties via max() -- either is valid since they're equal.
    assert report.reference_group in ("A", "B")
    other = "B" if report.reference_group == "A" else "A"
    pw = report.pairwise[other]
    assert pw.demographic_parity_diff == pytest.approx(0.0)
    assert pw.equal_opportunity_diff == pytest.approx(0.0)
    assert pw.disparate_impact_ratio == pytest.approx(1.0)
    assert report.max_demographic_parity_gap == pytest.approx(0.0)
    assert report.max_equal_opportunity_gap == pytest.approx(0.0)
    assert report.worst_case_disparate_impact_ratio == pytest.approx(1.0)


def test_compute_fairness_metrics_biased_case():
    report = compute_fairness_metrics(_biased_df(), "y_true", "y_pred", "group")
    assert report.reference_group == "A"  # A has n=12 > B's n=8
    pw = report.pairwise["B"]
    assert pw.demographic_parity_diff == pytest.approx(-0.25)
    assert pw.equal_opportunity_diff == pytest.approx(-0.5)
    assert pw.disparate_impact_ratio == pytest.approx(0.5)
    assert report.max_demographic_parity_gap == pytest.approx(0.25)
    assert report.max_equal_opportunity_gap == pytest.approx(0.5)
    assert report.worst_case_disparate_impact_ratio == pytest.approx(0.5)


def test_compute_fairness_metrics_three_groups():
    report = compute_fairness_metrics(_three_group_df(), "y_true", "y_pred", "group")
    assert report.reference_group == "A"

    pw_b = report.pairwise["B"]
    assert pw_b.demographic_parity_diff == pytest.approx(-0.25)
    assert pw_b.disparate_impact_ratio == pytest.approx(0.5)

    pw_c = report.pairwise["C"]
    assert pw_c.demographic_parity_diff == pytest.approx(-0.3)
    assert pw_c.equal_opportunity_diff == pytest.approx(1 / 3 - 1.0)
    assert pw_c.disparate_impact_ratio == pytest.approx(0.4)

    assert report.max_demographic_parity_gap == pytest.approx(0.3)
    assert report.max_equal_opportunity_gap == pytest.approx(1.0 - 1 / 3)
    assert report.worst_case_disparate_impact_ratio == pytest.approx(0.4)


def test_compute_fairness_metrics_requires_two_groups():
    df = _fair_df()
    df_single_group = df[df["group"] == "A"]
    with pytest.raises(ValueError):
        compute_fairness_metrics(df_single_group, "y_true", "y_pred", "group")


def test_compute_fairness_metrics_explicit_reference_group():
    report = compute_fairness_metrics(
        _biased_df(), "y_true", "y_pred", "group", reference_group="B"
    )
    assert report.reference_group == "B"
    pw_a = report.pairwise["A"]
    # Now A is compared against B: diff = rate(A) - rate(B) = 0.5 - 0.25 = 0.25
    assert pw_a.demographic_parity_diff == pytest.approx(0.25)
    assert pw_a.disparate_impact_ratio == pytest.approx(2.0)


def test_validate_binary_columns_rejects_non_binary():
    df = pd.DataFrame({"y_true": [1, 2, 0], "y_pred": [1, 0, 0], "group": ["A", "A", "B"]})
    with pytest.raises(ValueError):
        compute_group_stats(df, "y_true", "y_pred", "group")


def test_validate_binary_columns_missing_column():
    df = pd.DataFrame({"y_true": [1, 0], "y_pred": [1, 0], "group": ["A", "B"]})
    with pytest.raises(ValueError):
        compute_group_stats(df, "y_true", "does_not_exist", "group")
