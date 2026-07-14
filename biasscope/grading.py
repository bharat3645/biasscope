"""Pure, testable grading functions: map fairness metrics -> letter grades.

Thresholds are documented, simple, and intentionally conservative. They are
biasscope's own editorial judgment calls, informed by common regulatory
rules of thumb (e.g. the US EEOC "four-fifths rule" for disparate impact),
NOT a legal standard. Reasonable people (and reasonable regulators) could
draw the lines differently -- see README "Limitations" section.

Grade scale used throughout: A (best) > B > C > D > F (worst).
"""

from __future__ import annotations

from typing import Iterable, Optional

GRADE_ORDER = ["A", "B", "C", "D", "F"]
_GRADE_RANK = {g: i for i, g in enumerate(GRADE_ORDER)}  # 0 = best (A), 4 = worst (F)


def _rank(grade: str) -> int:
    return _GRADE_RANK[grade]


def worst_grade(grades: Iterable[str]) -> str:
    """Return the worst (lowest) grade in a collection of letter grades."""
    grades = list(grades)
    if not grades:
        raise ValueError("worst_grade requires at least one grade")
    return max(grades, key=_rank)


def grade_disparate_impact_ratio(ratio: Optional[float]) -> str:
    """Grade a disparate impact ratio (unprivileged rate / privileged rate).

    Thresholds (ratio closer to 1.0 is fairer):
        ratio >= 0.95  -> A  (essentially parity)
        ratio >= 0.90  -> B  (minor gap)
        ratio >= 0.80  -> C  (at the edge of the four-fifths rule)
        ratio >= 0.70  -> D  (fails four-fifths rule, moderately)
        ratio <  0.70  -> F  (fails four-fifths rule, severely)

    A ratio > 1.0 (unprivileged group favored) is folded symmetrically:
    we grade on how far the ratio is from 1.0 in either direction, since
    a large imbalance in *either* direction indicates disparate treatment.
    None (undefined, e.g. privileged group has zero positive rate) grades F
    because the ratio cannot be verified as safe.
    """
    if ratio is None:
        return "F"
    # Fold ratios > 1 back into the same [0, 1] scale for symmetric grading.
    normalized = ratio if ratio <= 1.0 else (1.0 / ratio if ratio > 0 else 0.0)

    if normalized >= 0.95:
        return "A"
    if normalized >= 0.90:
        return "B"
    if normalized >= 0.80:
        return "C"
    if normalized >= 0.70:
        return "D"
    return "F"


def grade_demographic_parity_diff(diff: Optional[float]) -> str:
    """Grade a demographic parity difference (positive-rate gap between groups).

    Graded on the absolute difference in positive-prediction rate:
        |diff| <= 0.05  -> A
        |diff| <= 0.10  -> B
        |diff| <= 0.20  -> C
        |diff| <= 0.30  -> D
        |diff| >  0.30  -> F
    """
    if diff is None:
        return "F"
    abs_diff = abs(diff)
    if abs_diff <= 0.05:
        return "A"
    if abs_diff <= 0.10:
        return "B"
    if abs_diff <= 0.20:
        return "C"
    if abs_diff <= 0.30:
        return "D"
    return "F"


def grade_equal_opportunity_diff(diff: Optional[float]) -> str:
    """Grade an equal opportunity difference (true-positive-rate gap).

    Uses the same bucket boundaries as demographic parity difference, since
    both are rate-differences on a 0-1 scale; equal opportunity is generally
    considered at least as important because it is measured only among
    actual positives (e.g. truly-qualified applicants):
        |diff| <= 0.05  -> A
        |diff| <= 0.10  -> B
        |diff| <= 0.20  -> C
        |diff| <= 0.30  -> D
        |diff| >  0.30  -> F
    None (undefined -- e.g. a group has no actual positives) grades F because
    equal opportunity cannot be verified for that group.
    """
    if diff is None:
        return "F"
    abs_diff = abs(diff)
    if abs_diff <= 0.05:
        return "A"
    if abs_diff <= 0.10:
        return "B"
    if abs_diff <= 0.20:
        return "C"
    if abs_diff <= 0.30:
        return "D"
    return "F"


def overall_grade(grades: Iterable[str]) -> str:
    """Overall grade for a set of metric grades: the worst one wins.

    Rationale: a report card should not let one great metric mask a
    dangerously bad one. Fairness failures on any single axis are still
    failures.
    """
    return worst_grade(grades)


GRADE_EXPLANATIONS = {
    "A": "No meaningful disparity was detected on this metric.",
    "B": "A small disparity was detected. Likely low risk, but worth monitoring over time.",
    "C": "A moderate disparity was detected, at or near common regulatory thresholds "
    "(e.g. the four-fifths rule). This warrants investigation before shipping.",
    "D": "A significant disparity was detected. This model likely treats groups "
    "unequally on this metric and needs remediation or a documented justification "
    "before deployment.",
    "F": "A severe disparity was detected (or the metric could not be verified as safe). "
    "Shipping this model as-is carries meaningful fairness and regulatory risk.",
}


def explain_grade(grade: str) -> str:
    """Plain-English explanation text for a given letter grade."""
    if grade not in GRADE_EXPLANATIONS:
        raise ValueError(f"Unknown grade '{grade}'")
    return GRADE_EXPLANATIONS[grade]
