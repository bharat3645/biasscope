"""biasscope: a lightweight, transparent ML fairness report-card generator.

biasscope computes standard fairness metrics (demographic parity difference,
equal opportunity difference, disparate impact ratio, per-group confusion
matrices) directly with pandas/numpy and turns them into a plain-English
report card with letter grades, aimed at product managers, compliance
reviewers, and engineers who are not fairness specialists.
"""

__version__ = "0.1.0"

from biasscope.metrics import compute_fairness_metrics, compute_group_stats
from biasscope.grading import (
    grade_disparate_impact_ratio,
    grade_demographic_parity_diff,
    grade_equal_opportunity_diff,
    overall_grade,
)
from biasscope.report import generate_report, report_to_markdown

__all__ = [
    "compute_fairness_metrics",
    "compute_group_stats",
    "grade_disparate_impact_ratio",
    "grade_demographic_parity_diff",
    "grade_equal_opportunity_diff",
    "overall_grade",
    "generate_report",
    "report_to_markdown",
]
