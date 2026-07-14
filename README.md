# biasscope

**biasscope** turns raw ML fairness metrics into a plain-English report card
that a product manager, compliance reviewer, or non-specialist engineer can
read and act on *before* a model ships.

## The problem

Fairness toolkits like [Fairlearn](https://fairlearn.org/) and
[Aequitas](https://github.com/dssg/aequitas) compute the right numbers, but
their output is built for people who already know what "equalized odds"
means. In practice, the person deciding whether to ship a model is often a
PM, a compliance reviewer, or an engineer who did not build the model and
does not have time to learn fairness-ML vocabulary. They need a report that
says, in plain language: *is this okay to ship, and why not, in terms I can
act on.*

biasscope is not a replacement for Fairlearn or Aequitas. It trades their
statistical completeness (many metrics, confidence intervals, mitigation
algorithms, multi-metric dashboards) for something narrower and more
immediately usable: a small, transparent set of metrics, computed directly
with pandas/numpy (no black-box dependency), turned into letter grades and a
Markdown report card with explanations, a limitations section, and no
required setup beyond a CSV file.

| | Fairlearn / Aequitas | biasscope |
|---|---|---|
| Audience | ML practitioners | PMs, compliance reviewers, generalist engineers |
| Metric coverage | Broad (dozens of metrics, mitigation algorithms) | Focused (4 core metrics + confusion matrices) |
| Output | Python objects / dashboards | A letter-graded Markdown report card |
| Dependencies | Fairlearn/Aequitas + their transitive deps | pandas + numpy only (core path) |
| Implementation | Library internals | Every formula is ~10 lines of pandas/numpy you can read yourself |

## Regulatory context (not legal advice)

Interest in this kind of tooling is rising because AI regulation is
tightening: the **EU AI Act** imposes bias-testing and documentation
obligations on providers of "high-risk" AI systems, and in the US, the
**EEOC four-fifths rule** (a selection rate for one group below 80% of the
rate for the highest-scoring group is treated as evidence of adverse
impact) has been a long-standing enforcement heuristic in employment
contexts. biasscope's disparate impact ratio check is directly inspired by
the four-fifths rule.

**biasscope is not a compliance tool and does not provide legal advice or a
compliance guarantee.** It is a diagnostic aid. Whether a given disparity is
legally or ethically acceptable depends on jurisdiction, use case, and facts
that no automated tool can fully evaluate. Always involve legal/compliance
expertise for real deployment decisions.

## Install

```bash
git clone https://github.com/bharat3645/biasscope.git
cd biasscope
pip install -r requirements.txt
```

Core functionality (metrics, grading, Markdown report, CLI) only requires
`pandas` and `numpy`. `anthropic` is optional and only used by the optional
LLM narration layer.

## Usage

Your CSV needs, at minimum: a ground-truth label column, a prediction
column, and one or more protected-attribute columns. Both label and
prediction columns must be binary (0/1 or True/False).

```bash
python -m biasscope audit data.csv \
    --y-true y_true \
    --y-pred y_pred \
    --protected gender race \
    --narrate \
    -o report.md
```

You can also drive it from a JSON/YAML config instead of flags (see
`config.example.json`):

```bash
python -m biasscope audit data.csv --config config.example.json
```

### Example

A synthetic 300-row loan-approval dataset is included as `sample_data.csv`
(columns: `y_true`, `y_pred`, `race`, `gender`), simulating a model that
approves qualified applicants in group A at 90% and in group B at only 55%.
Running:

```bash
python -m biasscope audit sample_data.csv --y-true y_true --y-pred y_pred --protected race gender --narrate
```

produces a report card whose `race` section looks like:

```
## Protected attribute: `race`

Reference (largest) group: **A**
Attribute grade: **F**

### Per-group summary

| Group | n | Positive rate | TPR (recall) | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|
| A (reference) | 200 | 53.0% | 85.1% | 86 | 20 | 79 | 15 |
| B | 100 | 30.0% | 48.9% | 23 | 7 | 46 | 24 |

### Pairwise fairness metrics (vs. reference group)

| Group | Demographic parity diff | Grade | Equal opportunity diff | Grade | Disparate impact ratio | Grade |
|---|---|---|---|---|---|---|
| B | -23.0% | D | -36.2% | F | 0.566 | F |
```

...with an overall grade of **F**, and plain-English explanations for every
number. The `gender` attribute in the same run comes out much closer to
parity (grade C, driven mainly by a moderate equal-opportunity gap) --
showing how the tool surfaces which *specific* attribute and *specific*
metric need attention, rather than a single opaque pass/fail.

### Python API

```python
import pandas as pd
from biasscope import generate_report, report_to_markdown
from biasscope.narrate import narrate_report

df = pd.read_csv("data.csv")
report = generate_report(df, y_true_col="y_true", y_pred_col="y_pred", protected_cols=["gender", "race"])
print(report_to_markdown(report))
print(narrate_report(report))  # deterministic template unless ANTHROPIC_API_KEY is set
```

## Metric definitions

Let `y_pred` be the model's binary prediction and `y_true` the ground-truth
binary label. For a protected attribute with groups `g1, g2, ...`, biasscope
picks the **largest group as the reference ("privileged") group** by
default (you can override this with `--reference COL=GROUP` or the
`reference_groups` config key).

- **Demographic parity difference** = `P(y_pred=1 | group) - P(y_pred=1 | reference)`
  -- the gap in how often each group receives a positive prediction,
  regardless of ground truth.
- **Equal opportunity difference** = `P(y_pred=1 | y_true=1, group) - P(y_pred=1 | y_true=1, reference)`
  -- the gap in true-positive rate (recall), computed only among people who
  are actually positive. This isolates whether the model is equally good at
  *finding* qualified/positive members of each group.
- **Disparate impact ratio** = `P(y_pred=1 | unprivileged group) / P(y_pred=1 | privileged group)`
  -- a ratio below **0.8** trips the four-fifths rule heuristic (flagged in
  the report). A ratio of 1.0 means perfect parity in selection rate.
- **Per-group confusion matrix**: raw TP / FP / TN / FN counts per group,
  included for full transparency and to let you recompute anything yourself.

For protected attributes with **more than two groups**, biasscope reports
each non-reference group's metrics against the reference group *and* a
**max-pairwise-gap**: the largest demographic-parity gap and largest
equal-opportunity gap across *any* pair of groups (not just vs. the
reference), plus the worst-case disparate impact ratio (`min positive rate /
max positive rate` across all groups). This ensures a bad disparity between
two *non-reference* groups cannot hide just because neither is the
reference.

## Grading thresholds

Grades are assigned by pure, independently-testable functions in
`biasscope/grading.py`:

- **Disparate impact ratio**: >=0.95 -> A, >=0.90 -> B, >=0.80 -> C, >=0.70 -> D, <0.70 -> F
  (ratios above 1.0 are folded symmetrically, since favoring either group
  strongly is still a disparity).
- **Demographic parity difference** and **equal opportunity difference**
  (graded on absolute value): <=5pp -> A, <=10pp -> B, <=20pp -> C, <=30pp -> D, >30pp -> F.
- **Overall grade** (per attribute, and across the whole report) is the
  *worst* of its component grades -- one great metric cannot mask a severe
  disparity on another.

These thresholds are biasscope's own editorial judgment call, informed by
the four-fifths rule and common practice, not a legal standard.

## Limitations

- **Fairness metrics can conflict with each other.** It is mathematically
  impossible in general to simultaneously satisfy demographic parity, equal
  opportunity, and calibration across groups when base rates differ between
  groups. Read every metric in context, not in isolation.
- **This is not a legal compliance certification.** See "Regulatory
  context" above.
- **Statistical fairness is not the same as substantive fairness.** A model
  can pass every metric here and still cause harm through proxy variables,
  feedback loops, or downstream decisions this report cannot see.
- **Small sample sizes make these metrics noisy.** Treat grades computed on
  a handful of examples per group with proportional skepticism.
- **biasscope only sees what's in the CSV.** It cannot assess data quality,
  label bias in the ground truth itself, or whether protected attributes
  were captured/inferred appropriately.
- **Domain judgment is still required.** A human familiar with the use
  case, the affected population, and applicable law must make the final
  call on whether to ship.

Every generated report repeats a version of this list under "What these
grades do NOT tell you," so it travels with the artifact, not just this
README.

## Optional LLM narration

`biasscope.narrate.narrate_report(report, llm_client=None)` adds a short
plain-English narrative summary on top of the structured report.

- With no `llm_client` and no `ANTHROPIC_API_KEY` environment variable set,
  it returns a fully deterministic, templated narrative -- no network
  calls, no API key, same output every time for the same report. This is
  the default, and what CI / offline usage will always get.
- If `ANTHROPIC_API_KEY` is set (and the `anthropic` package is installed),
  or you pass your own `llm_client` (an `anthropic.Anthropic()` instance),
  biasscope will attempt a real model-generated narrative instead. This
  path is best-effort: any failure (network, auth, quota) is caught and it
  silently falls back to the deterministic narrative so a missing/invalid
  key never breaks a pipeline.

```bash
export ANTHROPIC_API_KEY=your-key-here   # optional; never commit real keys
python -m biasscope audit data.csv --y-true y_true --y-pred y_pred --protected race --narrate
```

## Testing

```bash
pytest tests/ -v
```

The test suite constructs small synthetic DataFrames with **hand-computed
expected values** (worked out in the test file's docstrings) for a clearly
fair case, a clearly biased case, and a three-group case, plus exhaustive
boundary tests for every grading threshold. All formulas are cross-checked
against the confusion-matrix counts by hand before being asserted.

## License

MIT -- see [LICENSE](LICENSE).
