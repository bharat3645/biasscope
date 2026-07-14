"""Report-card generation: turns raw fairness metrics into a Markdown report.

The report is meant to be read by a product manager, compliance reviewer,
or non-specialist engineer -- not just an ML practitioner. Every number is
paired with a letter grade and a plain-English explanation.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from biasscope.grading import (
    explain_grade,
    grade_demographic_parity_diff,
    grade_disparate_impact_ratio,
    grade_equal_opportunity_diff,
    overall_grade,
)
from biasscope.metrics import AttributeFairnessReport, compute_fairness_metrics


@dataclass
class GroupRow:
    group: str
    n: int
    positive_rate: float
    tpr: Optional[float]
    confusion_matrix: Dict[str, int]
    is_reference: bool
    demographic_parity_diff: Optional[float] = None
    demographic_parity_grade: Optional[str] = None
    equal_opportunity_diff: Optional[float] = None
    equal_opportunity_grade: Optional[str] = None
    disparate_impact_ratio: Optional[float] = None
    disparate_impact_grade: Optional[str] = None


@dataclass
class AttributeReportCard:
    protected_col: str
    reference_group: str
    rows: List[GroupRow]
    max_demographic_parity_gap: float
    max_demographic_parity_grade: str
    max_equal_opportunity_gap: Optional[float]
    max_equal_opportunity_grade: str
    worst_case_disparate_impact_ratio: Optional[float]
    worst_case_disparate_impact_grade: str
    attribute_overall_grade: str


@dataclass
class Report:
    generated_at: str
    n_rows: int
    y_true_col: str
    y_pred_col: str
    attributes: List[AttributeReportCard]
    overall_grade: str


def build_attribute_report_card(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    protected_col: str,
    reference_group: Optional[str] = None,
) -> AttributeReportCard:
    fm: AttributeFairnessReport = compute_fairness_metrics(
        df, y_true_col, y_pred_col, protected_col, reference_group=reference_group
    )

    rows: List[GroupRow] = []
    for name, stats in fm.group_stats.items():
        is_ref = name == fm.reference_group
        row = GroupRow(
            group=name,
            n=stats.n,
            positive_rate=stats.positive_rate,
            tpr=stats.tpr,
            confusion_matrix=stats.confusion_matrix(),
            is_reference=is_ref,
        )
        if not is_ref and name in fm.pairwise:
            pw = fm.pairwise[name]
            row.demographic_parity_diff = pw.demographic_parity_diff
            row.demographic_parity_grade = grade_demographic_parity_diff(
                pw.demographic_parity_diff
            )
            row.equal_opportunity_diff = pw.equal_opportunity_diff
            row.equal_opportunity_grade = grade_equal_opportunity_diff(
                pw.equal_opportunity_diff
            )
            row.disparate_impact_ratio = pw.disparate_impact_ratio
            row.disparate_impact_grade = grade_disparate_impact_ratio(
                pw.disparate_impact_ratio
            )
        rows.append(row)

    # Order: reference group first, then others by name for determinism.
    rows.sort(key=lambda r: (not r.is_reference, r.group))

    max_dp_grade = grade_demographic_parity_diff(fm.max_demographic_parity_gap)
    max_eo_grade = grade_equal_opportunity_diff(fm.max_equal_opportunity_gap)
    worst_di_grade = grade_disparate_impact_ratio(fm.worst_case_disparate_impact_ratio)

    attr_overall = overall_grade([max_dp_grade, max_eo_grade, worst_di_grade])

    return AttributeReportCard(
        protected_col=protected_col,
        reference_group=fm.reference_group,
        rows=rows,
        max_demographic_parity_gap=fm.max_demographic_parity_gap,
        max_demographic_parity_grade=max_dp_grade,
        max_equal_opportunity_gap=fm.max_equal_opportunity_gap,
        max_equal_opportunity_grade=max_eo_grade,
        worst_case_disparate_impact_ratio=fm.worst_case_disparate_impact_ratio,
        worst_case_disparate_impact_grade=worst_di_grade,
        attribute_overall_grade=attr_overall,
    )


def generate_report(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    protected_cols: Sequence[str],
    reference_groups: Optional[Dict[str, str]] = None,
) -> Report:
    """Compute a full multi-attribute fairness report card.

    reference_groups: optional {protected_col: group_name} to override the
    default "largest group is reference" behavior per attribute.
    """
    reference_groups = reference_groups or {}
    attributes = [
        build_attribute_report_card(
            df,
            y_true_col,
            y_pred_col,
            col,
            reference_group=reference_groups.get(col),
        )
        for col in protected_cols
    ]
    overall = overall_grade([a.attribute_overall_grade for a in attributes])
    return Report(
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        n_rows=len(df),
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        attributes=attributes,
        overall_grade=overall,
    )


def _fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.1%}"


def _fmt_ratio(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.3f}"


def _fmt_diff(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:+.1%}"


LIMITATIONS_TEXT = """## What these grades do NOT tell you

- **Fairness metrics can conflict with each other.** It is mathematically
  impossible, except in special cases, to simultaneously satisfy demographic
  parity, equal opportunity, and calibration across groups when base rates
  differ. A high grade on one metric may come at the cost of another --
  read all metrics together, not in isolation.
- **This is not a legal compliance certification.** Grades are based on
  editorial thresholds inspired by common regulatory heuristics (e.g. the
  US EEOC four-fifths rule, discussions around the EU AI Act's requirements
  for high-risk AI systems). They do not constitute legal advice, and
  passing this report card does not mean a model is legally compliant in
  any jurisdiction.
- **Statistical fairness is not the same as substantive fairness.** A model
  can pass every metric here and still cause harm through proxy variables,
  feedback loops, or downstream decisions that this report cannot see.
- **Small sample sizes make these metrics noisy.** A gap computed on a
  handful of examples per group is not statistically reliable; treat grades
  for small groups with proportional skepticism.
- **This tool only sees what's in the CSV.** It cannot assess data quality,
  label bias in the ground truth itself, or whether the protected attribute
  columns were captured/inferred appropriately.
- **Domain judgment is still required.** A human familiar with the use
  case, the affected population, and applicable law must make the final
  call on whether to ship.
"""


def report_to_markdown(report: Report) -> str:
    lines: List[str] = []
    lines.append("# biasscope Fairness Report Card")
    lines.append("")
    lines.append(f"Generated: {report.generated_at}")
    lines.append("")
    lines.append(f"Rows analyzed: {report.n_rows}")
    lines.append(f"Prediction column: `{report.y_pred_col}`  |  Ground-truth column: `{report.y_true_col}`")
    lines.append("")
    lines.append(f"## Overall grade: {report.overall_grade}")
    lines.append("")
    lines.append(explain_grade(report.overall_grade))
    lines.append("")
    lines.append(
        "*The overall grade is the worst grade across every protected attribute and "
        "metric below -- a report card does not let one strong metric hide a severe "
        "disparity elsewhere.*"
    )
    lines.append("")

    for attr in report.attributes:
        lines.append(f"## Protected attribute: `{attr.protected_col}`")
        lines.append("")
        lines.append(f"Reference (largest) group: **{attr.reference_group}**")
        lines.append(f"Attribute grade: **{attr.attribute_overall_grade}**")
        lines.append("")

        lines.append("### Per-group summary")
        lines.append("")
        lines.append("| Group | n | Positive rate | TPR (recall) | TP | FP | TN | FN |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in attr.rows:
            cm = row.confusion_matrix
            label = f"{row.group} (reference)" if row.is_reference else row.group
            lines.append(
                f"| {label} | {row.n} | {_fmt_pct(row.positive_rate)} | "
                f"{_fmt_pct(row.tpr)} | {cm['TP']} | {cm['FP']} | {cm['TN']} | {cm['FN']} |"
            )
        lines.append("")

        lines.append("### Pairwise fairness metrics (vs. reference group)")
        lines.append("")
        lines.append(
            "| Group | Demographic parity diff | Grade | Equal opportunity diff | "
            "Grade | Disparate impact ratio | Grade |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in attr.rows:
            if row.is_reference:
                continue
            lines.append(
                f"| {row.group} | {_fmt_diff(row.demographic_parity_diff)} | "
                f"{row.demographic_parity_grade} | {_fmt_diff(row.equal_opportunity_diff)} | "
                f"{row.equal_opportunity_grade} | {_fmt_ratio(row.disparate_impact_ratio)} | "
                f"{row.disparate_impact_grade} |"
            )
        lines.append("")

        lines.append("### Worst-case gap across all groups (max-pairwise-gap)")
        lines.append("")
        lines.append(
            f"- Max demographic parity gap: {_fmt_pct(attr.max_demographic_parity_gap)} "
            f"-> grade {attr.max_demographic_parity_grade}"
        )
        lines.append(
            f"- Max equal opportunity gap: {_fmt_pct(attr.max_equal_opportunity_gap)} "
            f"-> grade {attr.max_equal_opportunity_grade}"
        )
        lines.append(
            f"- Worst-case disparate impact ratio: "
            f"{_fmt_ratio(attr.worst_case_disparate_impact_ratio)} -> grade "
            f"{attr.worst_case_disparate_impact_grade}"
        )
        lines.append("")

        lines.append("### What this means")
        lines.append("")
        lines.append(
            f"- **Demographic parity** ({attr.max_demographic_parity_grade}): "
            + explain_grade(attr.max_demographic_parity_grade)
        )
        lines.append(
            f"- **Equal opportunity** ({attr.max_equal_opportunity_grade}): "
            + explain_grade(attr.max_equal_opportunity_grade)
        )
        lines.append(
            f"- **Disparate impact (four-fifths rule)** "
            f"({attr.worst_case_disparate_impact_grade}): "
            + explain_grade(attr.worst_case_disparate_impact_grade)
        )
        lines.append("")

    lines.append(LIMITATIONS_TEXT)

    return "\n".join(lines)
