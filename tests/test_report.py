"""Tests for biasscope.report: report assembly and Markdown rendering."""

import pandas as pd
import pytest

from biasscope.report import generate_report, report_to_markdown


def _biased_df() -> pd.DataFrame:
    rows = []
    a_true = [1] * 6 + [0] * 6
    a_pred = [1] * 6 + [0] * 6
    for yt, yp in zip(a_true, a_pred):
        rows.append({"y_true": yt, "y_pred": yp, "gender": "A", "race": "X"})
    b_true = [1, 1, 1, 1, 0, 0, 0, 0]
    b_pred = [1, 1, 0, 0, 0, 0, 0, 0]
    for yt, yp in zip(b_true, b_pred):
        rows.append({"y_true": yt, "y_pred": yp, "gender": "B", "race": "Y"})
    return pd.DataFrame(rows)


def _fair_df() -> pd.DataFrame:
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    rows = []
    for g in ("A", "B"):
        for yt, yp in zip(y_true, y_pred):
            rows.append({"y_true": yt, "y_pred": yp, "gender": g})
    return pd.DataFrame(rows)


def test_generate_report_biased_case_overall_grade_is_f():
    report = generate_report(_biased_df(), "y_true", "y_pred", ["gender"])
    assert report.n_rows == 20
    attr = report.attributes[0]
    assert attr.protected_col == "gender"
    assert attr.reference_group == "A"
    # disparate impact ratio 0.5 -> F, so attribute + overall grade should be F
    assert attr.worst_case_disparate_impact_grade == "F"
    assert attr.attribute_overall_grade == "F"
    assert report.overall_grade == "F"


def test_generate_report_fair_case_overall_grade_is_a():
    report = generate_report(_fair_df(), "y_true", "y_pred", ["gender"])
    attr = report.attributes[0]
    assert attr.max_demographic_parity_grade == "A"
    assert attr.max_equal_opportunity_grade == "A"
    assert attr.worst_case_disparate_impact_grade == "A"
    assert attr.attribute_overall_grade == "A"
    assert report.overall_grade == "A"


def test_generate_report_multiple_attributes_overall_is_worst():
    df = _biased_df()
    report = generate_report(df, "y_true", "y_pred", ["gender", "race"])
    assert len(report.attributes) == 2
    # Both gender and race split identically along the same rows here, so
    # both attributes should independently reflect the same biased result,
    # and overall should be the worst of both (F).
    assert report.overall_grade == "F"


def test_report_to_markdown_contains_key_sections():
    report = generate_report(_biased_df(), "y_true", "y_pred", ["gender"])
    md = report_to_markdown(report)
    assert "# biasscope Fairness Report Card" in md
    assert "Overall grade" in md
    assert "gender" in md
    assert "Demographic parity diff" in md
    assert "Disparate impact ratio" in md
    assert "What these grades do NOT tell you" in md
    assert "not a legal compliance certification" in md


def test_report_to_markdown_is_deterministic():
    report = generate_report(_fair_df(), "y_true", "y_pred", ["gender"])
    md1 = report_to_markdown(report)
    # regenerate with a fresh report object (generated_at will differ, so
    # strip that line before comparing the rest for determinism)
    report2 = generate_report(_fair_df(), "y_true", "y_pred", ["gender"])
    md2 = report_to_markdown(report2)

    def strip_timestamp(md):
        return "\n".join(l for l in md.splitlines() if not l.startswith("Generated:"))

    assert strip_timestamp(md1) == strip_timestamp(md2)
