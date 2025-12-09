# QuLab: AI‑Driven Data Quality Scorecard (Streamlit Lab)

A hands-on Streamlit application that guides you through an end-to-end data quality workflow for time-series financial data. You will play the role of a Quantitative Analyst validating a synthetic corporate bond dataset before using it to train AI models. The app simulates realistic data issues (gaps, duplicates, outliers), lets you configure business rules and statistical checks, produces a weighted scorecard, detects anomalies with AI, and generates a management-ready impact report.

## Features

- End-to-end data quality pipeline for time-series data:
  - Data generation of a realistic corporate bond panel with intentional issues
  - Rule-based validity checks (ranges, categories, non-negativity, date logic)
  - Completeness and outlier analysis (IQR or Z-Score) with visualizations
  - Timeliness and duplicate detection across IDs and dates
  - Weighted data quality scorecard across dimensions (completeness, validity, consistency, timeliness, uniqueness)
  - Drilldowns for rule breaches and statistical anomalies
  - AI anomaly detection using Isolation Forest (PyOD) with visual diagnostics
  - Threshold-based alerts for proactive monitoring
  - Consolidated impact report for stakeholders
- Interactive, narrative-driven UI designed for a Quant/DataOps workflow
- Session state management with one-click reset
- Query-parameter navigation to deep-link to specific pages

## Getting Started

### Prerequisites

- Python 3.9–3.11
- pip (or conda/mamba)
- macOS, Linux, or Windows

Note: The app uses Streamlit’s query parameter API. Use a recent Streamlit version.

### Installation

1) Clone the repository:
```bash
git clone https://github.com/your-org/quilab.git
cd quilab
```

2) (Recommended) Create and activate a virtual environment:
```bash
# Using venv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3) Install dependencies:
```bash
pip install -r requirements.txt
```

If you don’t have a requirements.txt yet, create one with the following:
```text
streamlit>=1.30
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
pyod>=1.1.3
missingno>=0.5.2
```

## Usage

1) Launch the app:
```bash
streamlit run app.py
```

2) Navigate through the workflow using the sidebar:
- 1. Data Overview
  - Generate and load the synthetic corporate bond dataset
  - Preview the data head, schema, and descriptive stats
  - A CSV (corporate_bond_data.csv) is created in the project root
- 2. Validity Checks
  - Configure business rules (coupon range, risk score min, allowed ratings, non-negative volume, maturity date after trade date)
  - Run checks and review breach counts/percentages
- 3. Completeness & Consistency
  - Select columns for missingness analysis and outlier detection
  - Choose IQR or Z-Score method and tune thresholds (IQR multiplier)
  - View results, outlier counts/percentages, and distributions with bounds
- 4. Timeliness & Duplication
  - Detect gaps in daily frequency per Bond_ID
  - Check duplicates by chosen identifier columns (e.g., Bond_ID + Date)
- 5. Data Quality Scorecard
  - Set weights per dimension (completeness, validity, consistency, timeliness, uniqueness)
  - Generate a consolidated scorecard with an overall score
  - Visualize dimension scores against the overall score
- 6. Investigate Validity Breaches
  - See sorted breach summary
  - Sample concrete records for any breached rule
- 7. Investigate Statistical Anomalies
  - Review a combined summary of missingness and outliers
  - Visualize missingness matrix
  - Plot outlier distributions with detection bounds
- 8. AI Anomaly Detection
  - Select features and contamination rate
  - Run Isolation Forest to score anomalies
  - View anomaly samples and score distributions over time
- 9. Alerts Configuration & Status
  - Configure thresholds for overall score, completeness, validity, and AI anomaly percentage
  - Check and view any triggered alerts
- 10. Data Quality Impact Report
  - Generate a consolidated, management-ready report as markdown plus a tabular summary

3) Optional:
- Use “Reset Application” in the sidebar to clear session state and restart
- Deep-link to a specific page with query params, e.g.:
  - http://localhost:8501/?page=5.%20Data%20Quality%20Scorecard

## Project Structure

```text
.
├── app.py                               # Streamlit entrypoint and router
├── utils.py                             # Data generation, checks, scoring, AI, reporting utilities
├── application_pages/
│   ├── __init__.py
│   ├── page_1_data_overview.py          # Load and preview dataset
│   ├── page_2_validity_checks.py        # Rule-based validity checks
│   ├── page_3_completeness_consistency.py# Missingness and outlier analysis
│   ├── page_4_timeliness_duplication.py # Timeliness (gaps) & duplicates
│   ├── page_5_scorecard.py              # Weighted scorecard
│   ├── page_6_investigate_validity.py   # Drilldown on rule breaches
│   ├── page_7_investigate_statistical.py# Drilldown on statistical issues
│   ├── page_8_ai_anomaly.py             # Isolation Forest (PyOD) anomalies
│   ├── page_9_alerts.py                 # Threshold-based alerts
│   └── page_10_report.py                # Impact report generation
├── corporate_bond_data.csv              # Generated dataset (created at runtime)
└── README.md
```

Notes:
- The dataset is generated by utils.generate_and_load_corporate_bond_data and saved as corporate_bond_data.csv in the project root.
- All page-to-page state is managed with Streamlit’s session_state.

## Technology Stack

- Framework: Streamlit
- Data: pandas, numpy
- Visualization: matplotlib, seaborn, missingno
- ML/AI: scikit-learn (preprocessing), PyOD (Isolation Forest)
- Python standard library: datetime, io, etc.

## Contributing

Contributions are welcome! To propose changes:

1) Fork the repository and create a feature branch:
```bash
git checkout -b feature/your-feature
```

2) Make your changes. Please:
- Keep functions modular and documented (docstrings)
- Add comments for non-obvious logic
- Prefer vectorized pandas operations where possible
- Maintain consistent naming for session_state keys

3) Test locally:
```bash
streamlit run app.py
```

4) Commit and push:
```bash
git commit -m "Add your message"
git push origin feature/your-feature
```

5) Open a pull request describing:
- The problem addressed
- The approach taken
- Any UI changes and their impact
- Testing steps

For bug reports or feature requests, please open an issue with steps to reproduce and screenshots/logs when helpful.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

- Maintainer: Your Name
- Email: your.email@example.com
- Issues: https://github.com/your-org/quilab/issues
- Website: https://www.quantuniversity.com/ (brand referenced in the sidebar logo)

## Troubleshooting

- ModuleNotFoundError (pyod, missingno, etc.):
  - Ensure you installed all dependencies from requirements.txt
- Streamlit query params warnings:
  - Use a recent Streamlit version (>= 1.30 recommended)
- PyOD/scikit-learn install issues:
  - Upgrade pip and wheel: pip install --upgrade pip wheel
  - On Apple Silicon/Windows, ensure you’re using a supported Python version and prebuilt wheels

If issues persist, please open a GitHub issue with your OS, Python version, and the full error log.