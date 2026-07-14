"""Optional narrative layer on top of a biasscope Report.

By default (no ANTHROPIC_API_KEY and no llm_client passed in), narrate_report
returns a deterministic, fully templated narrative -- no network calls, no
API key required, same output every time for the same report. This keeps
biasscope usable offline and in CI.

If an llm_client is supplied (or the ANTHROPIC_API_KEY environment variable
is set and the `anthropic` package is installed), narrate_report will try to
produce a richer, model-written narrative summary of the report instead.
That code path is best-effort: on any failure it silently falls back to the
deterministic template so the function never crashes a pipeline over an
optional feature.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from biasscope.report import Report


def _deterministic_narrative(report: Report) -> str:
    """Template-based narrative summary, no external dependencies."""
    lines = []
    lines.append(
        f"This model's overall biasscope grade is {report.overall_grade}, based on "
        f"{report.n_rows} rows across {len(report.attributes)} protected attribute(s)."
    )

    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    worst_attrs = [
        a for a in report.attributes
        if grade_rank[a.attribute_overall_grade] == max(
            grade_rank[x.attribute_overall_grade] for x in report.attributes
        )
    ]

    if report.overall_grade in ("A", "B"):
        lines.append(
            "No protected attribute showed a severe disparity. That said, a good "
            "grade here is not a substitute for a human fairness review, especially "
            "for higher-stakes deployments."
        )
    else:
        attr_names = ", ".join(f"`{a.protected_col}`" for a in worst_attrs)
        lines.append(
            f"The weakest results were on {attr_names}, which should be investigated "
            "before this model ships. Consider whether the disparity comes from the "
            "training data, the labels, the features used, or the decision threshold."
        )

    for a in report.attributes:
        lines.append(
            f"- `{a.protected_col}` (reference group: {a.reference_group}): "
            f"overall grade {a.attribute_overall_grade}. "
            f"Max demographic parity gap {a.max_demographic_parity_gap:.1%} "
            f"(grade {a.max_demographic_parity_grade}); "
            f"max equal opportunity gap "
            f"{'N/A' if a.max_equal_opportunity_gap is None else f'{a.max_equal_opportunity_gap:.1%}'} "
            f"(grade {a.max_equal_opportunity_grade}); "
            f"worst-case disparate impact ratio "
            f"{'N/A' if a.worst_case_disparate_impact_ratio is None else f'{a.worst_case_disparate_impact_ratio:.3f}'} "
            f"(grade {a.worst_case_disparate_impact_grade})."
        )

    lines.append(
        "Remember: these grades reflect statistical disparities in this dataset only. "
        "They are not a legal compliance certification, and fairness metrics can "
        "trade off against one another -- see the report's limitations section."
    )
    return "\n".join(lines)


def _anthropic_narrative(report: Report, llm_client: Any) -> Optional[str]:
    """Best-effort real LLM narrative using an Anthropic client.

    llm_client is expected to expose the `messages.create(...)` interface of
    the official `anthropic` Python SDK (anthropic.Anthropic()). This path is
    intentionally isolated and defensive: any exception here is swallowed by
    the caller, which falls back to the deterministic template.
    """
    from biasscope.report import report_to_markdown

    report_md = report_to_markdown(report)
    prompt = (
        "You are helping a product manager understand an ML fairness report card. "
        "Write a concise (150-250 word) plain-English narrative summary of the "
        "following report. Call out the overall grade, the weakest protected "
        "attribute, and one concrete next step. Do not invent numbers that are not "
        "in the report. Do not claim legal compliance.\n\n"
        f"{report_md}"
    )
    response = llm_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    # anthropic SDK response shape: response.content is a list of blocks with .text
    try:
        return "".join(block.text for block in response.content)
    except AttributeError:
        return str(response)


def narrate_report(report: Report, llm_client: Optional[Any] = None) -> str:
    """Produce a plain-English narrative for a biasscope Report.

    - If llm_client is given, try to use it for a richer narrative.
    - Else, if ANTHROPIC_API_KEY is set and the `anthropic` package is
      installed, construct a client and try that.
    - In all other cases (or if the above raises for any reason), fall back
      to a deterministic, templated narrative that requires no network
      access and no API key.
    """
    client = llm_client
    if client is None and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except Exception:
            client = None

    if client is not None:
        try:
            result = _anthropic_narrative(report, client)
            if result:
                return result
        except Exception:
            pass  # fall through to deterministic narrative

    return _deterministic_narrative(report)
