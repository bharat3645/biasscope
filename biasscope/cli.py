"""biasscope CLI: python -m biasscope audit data.csv --y-true y_true --y-pred y_pred --protected gender race"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from biasscope.narrate import narrate_report
from biasscope.report import generate_report, report_to_markdown


def _load_config(config_path: Optional[str]) -> Dict:
    if not config_path:
        return {}
    path = Path(config_path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "Reading a YAML config requires PyYAML. Install it with "
                "`pip install pyyaml`, or use a JSON config instead."
            ) from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biasscope",
        description="Generate a plain-English ML fairness report card from a CSV of predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="Audit a CSV of predictions/labels/protected attributes."
    )
    audit.add_argument("csv_path", help="Path to input CSV file.")
    audit.add_argument("--y-true", dest="y_true", help="Ground-truth label column name.")
    audit.add_argument("--y-pred", dest="y_pred", help="Model prediction column name.")
    audit.add_argument(
        "--protected",
        dest="protected",
        nargs="+",
        help="One or more protected-attribute column names.",
    )
    audit.add_argument(
        "--config",
        dest="config",
        help="Optional YAML/JSON config file with y_true/y_pred/protected/reference_groups keys. "
        "CLI flags override config values.",
    )
    audit.add_argument(
        "--reference",
        dest="reference",
        nargs="*",
        default=[],
        metavar="COL=GROUP",
        help="Force a reference group per attribute, e.g. --reference gender=male race=white",
    )
    audit.add_argument(
        "-o", "--output", dest="output", help="Write the Markdown report to this file instead of stdout."
    )
    audit.add_argument(
        "--narrate",
        action="store_true",
        help="Also print a plain-English narrative summary (deterministic unless "
        "ANTHROPIC_API_KEY is set).",
    )
    audit.add_argument(
        "--json",
        dest="json_out",
        help="Also write raw computed metrics as JSON to this path.",
    )
    return parser


def _parse_reference_overrides(pairs) -> Dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--reference expects COL=GROUP, got '{pair}'")
        col, group = pair.split("=", 1)
        out[col] = group
    return out


def _report_to_dict(report) -> dict:
    def attr_to_dict(a):
        return {
            "protected_col": a.protected_col,
            "reference_group": a.reference_group,
            "attribute_overall_grade": a.attribute_overall_grade,
            "max_demographic_parity_gap": a.max_demographic_parity_gap,
            "max_demographic_parity_grade": a.max_demographic_parity_grade,
            "max_equal_opportunity_gap": a.max_equal_opportunity_gap,
            "max_equal_opportunity_grade": a.max_equal_opportunity_grade,
            "worst_case_disparate_impact_ratio": a.worst_case_disparate_impact_ratio,
            "worst_case_disparate_impact_grade": a.worst_case_disparate_impact_grade,
            "groups": [
                {
                    "group": r.group,
                    "n": r.n,
                    "positive_rate": r.positive_rate,
                    "tpr": r.tpr,
                    "confusion_matrix": r.confusion_matrix,
                    "is_reference": r.is_reference,
                    "demographic_parity_diff": r.demographic_parity_diff,
                    "demographic_parity_grade": r.demographic_parity_grade,
                    "equal_opportunity_diff": r.equal_opportunity_diff,
                    "equal_opportunity_grade": r.equal_opportunity_grade,
                    "disparate_impact_ratio": r.disparate_impact_ratio,
                    "disparate_impact_grade": r.disparate_impact_grade,
                }
                for r in a.rows
            ],
        }

    return {
        "generated_at": report.generated_at,
        "n_rows": report.n_rows,
        "y_true_col": report.y_true_col,
        "y_pred_col": report.y_pred_col,
        "overall_grade": report.overall_grade,
        "attributes": [attr_to_dict(a) for a in report.attributes],
    }


def run_audit(args: argparse.Namespace) -> int:
    config = _load_config(args.config)

    y_true = args.y_true or config.get("y_true")
    y_pred = args.y_pred or config.get("y_pred")
    protected = args.protected or config.get("protected")
    reference_groups = dict(config.get("reference_groups", {}))
    reference_groups.update(_parse_reference_overrides(args.reference))

    if not y_true or not y_pred or not protected:
        raise SystemExit(
            "Must specify --y-true, --y-pred, and --protected (via flags or --config)."
        )

    df = pd.read_csv(args.csv_path)
    report = generate_report(df, y_true, y_pred, protected, reference_groups=reference_groups)
    markdown = report_to_markdown(report)

    if args.narrate:
        markdown += "\n\n## Narrative summary\n\n" + narrate_report(report)

    if args.output:
        Path(args.output).write_text(markdown)
        print(f"Report written to {args.output}")
    else:
        print(markdown)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(_report_to_dict(report), indent=2))
        print(f"Raw metrics written to {args.json_out}", file=sys.stderr)

    return 0


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "audit":
        return run_audit(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
