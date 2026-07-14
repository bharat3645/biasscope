"""Tests for biasscope.grading threshold boundaries."""

import pytest

from biasscope.grading import (
    explain_grade,
    grade_demographic_parity_diff,
    grade_disparate_impact_ratio,
    grade_equal_opportunity_diff,
    overall_grade,
    worst_grade,
)


# --- disparate impact ratio ---------------------------------------------

@pytest.mark.parametrize(
    "ratio,expected",
    [
        (1.0, "A"),
        (0.95, "A"),
        (0.9499, "B"),
        (0.90, "B"),
        (0.8999, "C"),
        (0.80, "C"),
        (0.7999, "D"),
        (0.70, "D"),
        (0.6999, "F"),
        (0.0, "F"),
    ],
)
def test_grade_disparate_impact_ratio_boundaries(ratio, expected):
    assert grade_disparate_impact_ratio(ratio) == expected


def test_grade_disparate_impact_ratio_none_is_f():
    assert grade_disparate_impact_ratio(None) == "F"


def test_grade_disparate_impact_ratio_symmetric_above_one():
    # ratio of 2.0 (unprivileged group favored 2x) should fold to 0.5 -> F,
    # same severity as its reciprocal.
    assert grade_disparate_impact_ratio(2.0) == grade_disparate_impact_ratio(0.5)
    assert grade_disparate_impact_ratio(1.0 / 0.95) == "A"


# --- demographic parity difference --------------------------------------

@pytest.mark.parametrize(
    "diff,expected",
    [
        (0.0, "A"),
        (0.05, "A"),
        (-0.05, "A"),
        (0.0501, "B"),
        (0.10, "B"),
        (0.1001, "C"),
        (0.20, "C"),
        (0.2001, "D"),
        (0.30, "D"),
        (0.3001, "F"),
        (-0.3001, "F"),
    ],
)
def test_grade_demographic_parity_diff_boundaries(diff, expected):
    assert grade_demographic_parity_diff(diff) == expected


def test_grade_demographic_parity_diff_none_is_f():
    assert grade_demographic_parity_diff(None) == "F"


# --- equal opportunity difference ---------------------------------------

@pytest.mark.parametrize(
    "diff,expected",
    [
        (0.0, "A"),
        (0.05, "A"),
        (0.0501, "B"),
        (0.10, "B"),
        (0.1001, "C"),
        (0.20, "C"),
        (0.2001, "D"),
        (0.30, "D"),
        (0.3001, "F"),
    ],
)
def test_grade_equal_opportunity_diff_boundaries(diff, expected):
    assert grade_equal_opportunity_diff(diff) == expected


def test_grade_equal_opportunity_diff_none_is_f():
    assert grade_equal_opportunity_diff(None) == "F"


# --- overall / worst grade -----------------------------------------------

def test_worst_grade_picks_lowest():
    assert worst_grade(["A", "B", "F", "C"]) == "F"
    assert worst_grade(["A", "A", "A"]) == "A"
    assert worst_grade(["B", "C"]) == "C"


def test_worst_grade_requires_nonempty():
    with pytest.raises(ValueError):
        worst_grade([])


def test_overall_grade_is_worst_grade():
    assert overall_grade(["A", "D", "B"]) == "D"


def test_explain_grade_known_grades():
    for g in ("A", "B", "C", "D", "F"):
        text = explain_grade(g)
        assert isinstance(text, str)
        assert len(text) > 0


def test_explain_grade_unknown_raises():
    with pytest.raises(ValueError):
        explain_grade("Z")
