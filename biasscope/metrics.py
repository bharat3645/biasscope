"""Core fairness-metric math for biasscope.

Every metric here is implemented directly with pandas/numpy so the
computation is fully auditable -- no hidden logic inside a heavyweight
fairness library. Definitions:

Demographic parity difference
    difference in the *positive-prediction rate* -- P(y_pred = 1) -- between
    a group and a reference group. 0 means the groups get positive
    predictions at the same rate.

Equal opportunity difference
    difference in the *true-positive rate* (recall), i.e. P(y_pred = 1 |
    y_true = 1), between a group and a reference group. This only looks at
    people who are actually in the positive class, so it isolates whether
    the model is equally good at *finding* qualified/positive members of
    each group.

Disparate impact ratio
    ratio of positive-prediction rates: rate(unprivileged) / rate(privileged).
    A ratio below 0.8 trips the so-called "four-fifths rule", a heuristic
    that originated in US EEOC hiring-discrimination guidance. It is a
    common regulatory rule of thumb, not a legal guarantee of fairness.

Per-group confusion matrix
    raw TP / FP / TN / FN counts for each group, for full transparency.

For a protected attribute with more than two groups, biasscope treats the
largest group as the reference ("privileged") group and reports each other
group's metrics relative to it, plus a max-pairwise-gap that captures the
worst-case disparity across *all* group pairs (not just vs. the reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GroupStats:
    group: str
    n: int
    n_positive_actual: int  # count of y_true == 1
    tp: int
    fp: int
    tn: int
    fn: int
    positive_rate: float  # P(y_pred == 1)
    tpr: Optional[float]  # P(y_pred == 1 | y_true == 1), None if no actual positives
    fpr: Optional[float]  # P(y_pred == 1 | y_true == 0), None if no actual negatives

    def confusion_matrix(self) -> Dict[str, int]:
        return {"TP": self.tp, "FP": self.fp, "TN": self.tn, "FN": self.fn}


def compute_group_stats(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, protected_col: str
) -> Dict[str, GroupStats]:
    """Compute confusion-matrix-derived stats for every group in protected_col.

    y_true and y_pred columns are expected to be binary (0/1 or True/False).
    """
    _validate_binary_columns(df, y_true_col, y_pred_col)

    stats: Dict[str, GroupStats] = {}
    for group_val, sub in df.groupby(protected_col):
        y_true = sub[y_true_col].astype(int).to_numpy()
        y_pred = sub[y_pred_col].astype(int).to_numpy()

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        n = len(sub)
        n_pos_actual = tp + fn
        n_neg_actual = tn + fp

        positive_rate = (tp + fp) / n if n > 0 else float("nan")
        tpr = tp / n_pos_actual if n_pos_actual > 0 else None
        fpr = fp / n_neg_actual if n_neg_actual > 0 else None

        stats[str(group_val)] = GroupStats(
            group=str(group_val),
            n=n,
            n_positive_actual=n_pos_actual,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            positive_rate=positive_rate,
            tpr=tpr,
            fpr=fpr,
        )
    return stats


def _validate_binary_columns(df: pd.DataFrame, *cols: str) -> None:
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        uniques = set(pd.unique(df[col].dropna()))
        if not uniques.issubset({0, 1, True, False}):
            raise ValueError(
                f"Column '{col}' must be binary (0/1 or True/False); "
                f"found values: {sorted(uniques, key=str)}"
            )


def _pick_reference_group(stats: Dict[str, GroupStats]) -> str:
    """Reference/"privileged" group defaults to the largest group by count.

    This is a simplifying, documented assumption: biasscope does not know
    which group is socially/historically privileged, so it uses group size
    as a practical stand-in. Callers who know the true privileged group
    can pass reference_group explicitly to compute_fairness_metrics.
    """
    return max(stats.values(), key=lambda s: s.n).group


@dataclass
class PairwiseMetrics:
    group: str
    reference_group: str
    demographic_parity_diff: float
    equal_opportunity_diff: Optional[float]
    disparate_impact_ratio: Optional[float]


@dataclass
class AttributeFairnessReport:
    protected_col: str
    reference_group: str
    group_stats: Dict[str, GroupStats]
    pairwise: Dict[str, PairwiseMetrics]
    max_demographic_parity_gap: float
    max_equal_opportunity_gap: Optional[float]
    worst_case_disparate_impact_ratio: Optional[float]


def demographic_parity_diff(rate_group: float, rate_reference: float) -> float:
    """positive_rate(group) - positive_rate(reference)."""
    return rate_group - rate_reference


def equal_opportunity_diff(
    tpr_group: Optional[float], tpr_reference: Optional[float]
) -> Optional[float]:
    """tpr(group) - tpr(reference), or None if either is undefined."""
    if tpr_group is None or tpr_reference is None:
        return None
    return tpr_group - tpr_reference


def disparate_impact_ratio(
    rate_unprivileged: float, rate_privileged: float
) -> Optional[float]:
    """rate(unprivileged) / rate(privileged). None if privileged rate is 0."""
    if rate_privileged == 0:
        return None
    return rate_unprivileged / rate_privileged


def compute_fairness_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    protected_col: str,
    reference_group: Optional[str] = None,
) -> AttributeFairnessReport:
    """Compute the full fairness report for one protected attribute column.

    If reference_group is not given, the largest group is used as the
    reference ("privileged") group -- see _pick_reference_group.
    """
    group_stats = compute_group_stats(df, y_true_col, y_pred_col, protected_col)
    if len(group_stats) < 2:
        raise ValueError(
            f"Protected attribute '{protected_col}' must have at least 2 groups; "
            f"found {len(group_stats)}"
        )

    ref = reference_group if reference_group is not None else _pick_reference_group(group_stats)
    if ref not in group_stats:
        raise ValueError(f"reference_group '{ref}' not found among groups {list(group_stats)}")
    ref_stats = group_stats[ref]

    pairwise: Dict[str, PairwiseMetrics] = {}
    for name, s in group_stats.items():
        if name == ref:
            continue
        dp_diff = demographic_parity_diff(s.positive_rate, ref_stats.positive_rate)
        eo_diff = equal_opportunity_diff(s.tpr, ref_stats.tpr)
        di_ratio = disparate_impact_ratio(s.positive_rate, ref_stats.positive_rate)
        pairwise[name] = PairwiseMetrics(
            group=name,
            reference_group=ref,
            demographic_parity_diff=dp_diff,
            equal_opportunity_diff=eo_diff,
            disparate_impact_ratio=di_ratio,
        )

    all_rates = [s.positive_rate for s in group_stats.values()]
    max_dp_gap = max(all_rates) - min(all_rates)

    all_tprs = [s.tpr for s in group_stats.values() if s.tpr is not None]
    max_eo_gap = (max(all_tprs) - min(all_tprs)) if len(all_tprs) >= 2 else None

    worst_di_ratio = (min(all_rates) / max(all_rates)) if max(all_rates) > 0 else None

    return AttributeFairnessReport(
        protected_col=protected_col,
        reference_group=ref,
        group_stats=group_stats,
        pairwise=pairwise,
        max_demographic_parity_gap=max_dp_gap,
        max_equal_opportunity_gap=max_eo_gap,
        worst_case_disparate_impact_ratio=worst_di_ratio,
    )
