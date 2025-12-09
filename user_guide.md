id: 69384caf472b65d9d1c9a3d6_user_guide
summary: Data Quality Evaluator User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: AI-Driven Data Quality Scorecard — Codelab User Guide

## 0. Why this app matters and what you will learn
Duration: 3:00

This codelab guides you through QuLab, a Streamlit app that turns raw datasets into data-quality evidence for AI model training. You’ll act as a Quantitative Analyst ensuring a simulated corporate bond dataset is fit for purpose.

What you will learn:
- How to quickly scan a dataset to spot immediate risks (structure, types, ranges)
- How to enforce business policies with rule-based validity checks
- How to measure completeness and detect statistical outliers using IQR and Z-score concepts
- How to verify time-series integrity and identify duplicate records
- How to summarize health with a weighted data-quality scorecard
- How to investigate specific rule breaches and statistical anomalies
- How to apply AI-based anomaly detection (Isolation Forest) to uncover subtle issues
- How to configure alert thresholds to monitor data quality
- How to generate a management-ready impact report

Why this matters:
- Poor data quality degrades model training, inflates risk, and weakens governance
- A structured workflow enables faster go/no-go decisions and audit-ready evidence
- Combining rules, statistics, and AI yields broader coverage than any single method

<aside class="positive">
Use QuLab as a repeatable pre-modeling checklist to prevent bad data from entering your training pipeline. It gives you traceable metrics, visual diagnostics, and actionable thresholds.
</aside>


## 1. Getting started and navigation
Duration: 1:30

- Launch the app and locate the sidebar:
  - The logo and a Navigation select box let you jump between steps.
  - A Reset Application button clears your current session.
- The main area shows the current step with instructions and outputs.
- Follow steps in order (1 through 10) for a complete workflow.

Tip: QuLab uses a simulated corporate bond dataset with intentional quirks (gaps and duplicates) to mirror real-world conditions.


## 2. Data Overview
Duration: 3:00

Goal: Load a fresh dataset and perform a quick sanity check.

How to use:
- Open “1. Data Overview”.
- Click “Load Corporate Bond Data”.
- Review:
  - Head of the dataset
  - Schema info (types, non-null counts)
  - Descriptive statistics for numerical and categorical fields

What to look for:
- Missing values in key features like `Volume`, `Rating`, `Risk_Score`, `Index_Value`, `Coupon_Rate`
- Suspicious ranges (e.g., negative values, impossible dates)
- Categories that don’t match expectations

<aside class="positive">
Early scanning helps you tailor the next checks. If ranges or categories look off, tighten rule thresholds in the next step.
</aside>


## 3. Validity Checks
Duration: 4:00

Goal: Apply rule-based checks that reflect business and regulatory expectations.

How to use:
- Open “2. Validity Checks”.
- In “Configure Validity Rules”, keep the default rules or adjust them:
  - Coupon Rate within bounds
  - Risk Score non-negative
  - Rating within an allowed set
  - Volume non-negative
  - Maturity Date after trade Date
- Click “Run Validity Checks” and review the results table.

Concepts:
- Validity rules translate policy into pass/fail gates. Example constraints:
  $$ 0.01 \leq \text{Coupon Rate} \leq 0.10,\quad \text{Risk Score} \geq 0.0,\quad \text{Volume} \geq 0.0,\quad \text{Maturity Date} > \text{Date} $$

Interpreting results:
- Focus on rules with higher breach percentages.
- Decide if breaches reflect real business scenarios (e.g., rare instruments) or data defects.

<aside class="negative">
Avoid ignoring breaches because “they’re rare.” Even small violations can skew models or violate policy, especially in regulated environments.
</aside>


## 4. Completeness & Consistency (Outliers)
Duration: 5:00

Goal: Quantify missingness and detect outliers in model-relevant features.

How to use:
- Open “3. Completeness & Consistency”.
- Select columns for Completeness Check (missing data).
- Select numerical columns for Outlier Detection (consistency).
- Choose the method: “IQR” (default) or “Z-Score”.
- Adjust “IQR Multiplier (k)” to tune sensitivity.
- Click “Run Statistical Checks” to see:
  - Completeness results
  - Consistency (Outlier) results
  - Distribution plots with outlier bounds

Concepts:
- IQR method flags values far from the middle 50%:
  $$ x < Q1 - k \cdot IQR \quad \text{or} \quad x > Q3 + k \cdot IQR $$
  where $IQR = Q3 - Q1$ and $k$ is your tolerance.
- Z-Score method uses standardized distance from the mean (typically threshold 3).

Interpretation:
- “WARN” indicates low-level issues; “FAIL” calls for action (imputation, capping, or investigation).
- Higher $k$ focuses on extreme outliers—useful in finance where spikes may be genuine events.

<aside class="positive">
Use distribution plots to decide between capping outliers, transforming data (e.g., log), or leaving them as legitimate signals.
</aside>


## 5. Timeliness & Duplication
Duration: 3:00

Goal: Ensure time-series integrity and record uniqueness.

How to use:
- Open “4. Timeliness & Duplication”.
- Confirm `Date` as the date column.
- Select unique identifier columns for duplicate checks (default `Bond_ID`, `Date`).
- Click “Run Timeliness & Duplicate Checks”.
- Review:
  - Missing dates per bond (time gaps)
  - Duplicate records based on selected keys
- Inspect “Sample Duplicate Records” and “Sample Missing Dates” sections if present.

Why it matters:
- Gaps distort rolling features, stress testing, and backtests.
- Duplicates inflate counts and bias summary statistics.

Remediation:
- For gaps: source missing records or adjust model windows.
- For duplicates: deduplicate with clear business logic (e.g., last write wins).


## 6. Data Quality Scorecard
Duration: 4:00

Goal: Translate multiple checks into a single, weighted decision metric.

How to use:
- Open “5. Data Quality Scorecard”.
- Set weights for each dimension (Completeness, Validity, Consistency, Timeliness, Uniqueness).
- Click “Generate Data Quality Scorecard”.
- Review:
  - Table with dimension scores and the Overall Score
  - Bar chart with an overlaid Overall Score

Concepts:
- Dimension scores convert breach percentages into a 0–1 scale:
  $$ s_i = 1 - \frac{\text{Breach Percentage}_i}{100} $$
- The overall score is a weighted average:
  $$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$

Interpretation:
- Align weights to business priorities (e.g., raise Validity for regulated use cases).
- Use the Overall Score for go/no-go decisions and to track improvements over time.


## 7. Investigate Validity Breaches
Duration: 3:00

Goal: Drill down into specific rule violations to decide remediation.

How to use:
- Open “6. Investigate Validity Breaches”.
- Review the “Breach Summary”.
- Select a breached rule from the dropdown.
- Choose how many samples to view and click “Show Sample Records”.

What to look for:
- Recurrent issues tied to specific bonds, dates, or data sources
- Edge cases that should be allowed versus clear errors to fix

Action ideas:
- Fix upstream data creation or ingestion logic
- Filter bad records from training sets
- Document justified exceptions with clear rationale


## 8. Investigate Statistical Anomalies
Duration: 4:00

Goal: Visualize missingness and outlier distributions for informed pre-processing.

How to use:
- Open “7. Investigate Statistical Anomalies”.
- View the “Statistical Issues Summary”.
- Inspect the “Missingness Matrix” for patterns (systematic vs. random).
- Choose a numerical column and click “Plot Outlier Distribution”.

Interpretation:
- Use plots to confirm whether flagged outliers are noise or signal.
- Consider imputation strategies (mean/median/forward-fill), capping, or transformations.

<aside class="positive">
A structured view of missingness and outliers prevents ad-hoc fixes. Apply rules consistently and document changes to preserve reproducibility.
</aside>


## 9. AI Anomaly Detection
Duration: 5:00

Goal: Use an unsupervised model to catch subtle, multi-feature anomalies.

How to use:
- Open “8. AI Anomaly Detection”.
- Select features (e.g., `Index_Value`, `Volume`, `Coupon_Rate`, `Risk_Score`).
- Set the “Contamination Rate” (expected anomaly percentage).
- Click “Run AI Anomaly Detection”.
- Review:
  - Anomaly Count and Percentage
  - Sample Detected Anomalies table
  - Anomaly Score Distribution plot
  - Anomaly Scores Over Time plot

Concepts:
- Isolation Forest isolates points that behave differently from the majority.
- It’s robust for multi-dimensional irregularities that rules or simple thresholds miss.

Interpretation:
- Investigate clusters of anomalies by date or bond.
- Balance false positives with contamination tuning; start small and adjust pragmatically.

<aside class="negative">
Do not treat AI anomaly flags as absolute truth. Validate with domain experts, especially before taking corrective action that drops data.
</aside>


## 10. Alerts Configuration & Status
Duration: 3:00

Goal: Turn your analysis into ongoing guardrails with thresholds.

How to use:
- Open “9. Alerts Configuration & Status”.
- Set thresholds for:
  - Overall Score
  - Completeness Score
  - Validity Score
  - AI Anomaly Percentage
- Click “Check for Critical Alerts” to see any triggered alerts.

Best practices:
- Align thresholds to risk appetite and regulatory commitments.
- Start slightly conservative, then calibrate as you observe production behavior.

Outcome:
- A clear list of current alerts you can act on or escalate.


## 11. Data Quality Impact Report
Duration: 3:00

Goal: Produce a concise, shareable summary for stakeholders.

How to use:
- Open “10. Data Quality Impact Report”.
- Click “Generate Impact Report”.
- Review:
  - Overall Score and dimension scores
  - Validity and statistical summaries
  - AI anomalies overview
  - Critical alerts
  - Report table for export

Use cases:
- Executive updates and go/no-go sign-offs
- Audit evidence and governance meetings
- Backlog creation for remediation tasks

<aside class="positive">
The report consolidates narrative and numbers, enabling fast decisions. Use it to track improvements across data refreshes.
</aside>


## 12. Putting it all together: A repeatable workflow
Duration: 2:00

- Load data and scan it for obvious risks
- Enforce business validity rules
- Measure completeness and outliers; visualize distributions
- Check timeliness and duplicates for time-series integrity
- Weight and compute the overall data-quality score
- Investigate specific issues with targeted drill-downs
- Layer on AI anomaly detection for subtle patterns
- Configure alerts to monitor ongoing runs
- Generate and share the impact report

Recommended next steps:
- Document decisions for each issue (fix upstream, filter, or justify)
- Track score trends over time to quantify improvements
- Integrate thresholds and reports into your operational runbook

You now have an end-to-end, decision-focused data-quality workflow that blends rules, statistics, and AI—purpose-built for model readiness in finance.
