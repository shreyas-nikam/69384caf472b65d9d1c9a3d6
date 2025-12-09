
# Streamlit Application Specification: AI-Driven Data Quality Scorecard for Regulated Finance

## 1. Application Overview

**Narrative-Aligned Overview:**

This Streamlit application takes on the persona of a **Quantitative Analyst** working in a DataOps team at a leading financial institution. The narrative guides the user through the critical process of establishing a robust, continuous data quality management framework for AI training datasets in a highly regulated environment. The story unfolds as the Quantitative Analyst identifies, quantifies, and addresses data quality issues across various dimensions, culminating in a comprehensive data quality scorecard and an impact report for stakeholders.

The user will experience the journey as a sequence of decisions and insights, mirroring the realistic workflow of a Quantitative Analyst. They will:
1.  **Load and explore** a simulated corporate bond dataset, immediately identifying initial data quality concerns.
2.  **Define and apply rule-based checks** to ensure data validity against business and regulatory requirements.
3.  **Perform statistical analyses** to assess completeness and consistency, detecting missing values and outliers.
4.  **Evaluate timeliness and uniqueness** to ensure data integrity for time-series models.
5.  **Aggregate these findings** into a weighted data quality score, providing a holistic view of data health.
6.  **Investigate specific breaches and anomalies** to understand their root causes and potential impact on AI models.
7.  **Employ AI-based anomaly detection** to uncover subtle, complex data quality deviations that traditional methods might miss.
8.  **Configure an alerting system** to monitor critical data quality thresholds proactively.
9.  **Generate a comprehensive impact report** to communicate findings, risks, and recommendations to senior management and data owners.

**Real-world Problem:**

The core problem addressed is the imperative for financial institutions to ensure the integrity and reliability of data used to train AI models. In regulated environments, poor data quality can lead to biased AI outputs, significant compliance breaches, financial losses, and reputational damage. The application demonstrates how to move from reactive issue-fixing to a proactive, continuous data quality management framework that underpins trustworthy and compliant AI.

**How the Streamlit App Helps:**

The Streamlit app acts as an interactive "Data Quality Workbench." It allows the Quantitative Analyst persona to:
*   **Systematically apply** various data quality checks (rule-based, statistical, AI-based) through intuitive interactive components.
*   **Visualize results** immediately, turning complex data issues into actionable insights.
*   **Dynamically adjust parameters** (e.g., outlier thresholds, dimension weights, alert thresholds) to observe their impact on data quality assessments and scores.
*   **Experience the iterative process** of data quality analysis, investigation, and reporting.
*   **Quantify the impact** of data quality on AI model reliability and regulatory compliance.

The app avoids direct explanations of concepts, instead *showing* how each check and metric directly supports the persona's goal of ensuring AI model integrity and compliance in a financial setting.

**Learning Goals (Applied Skills):**

By interacting with this application, the persona (and user) will gain applied skills in:
*   **Configuring Data Quality Checks:** Ability to define and apply rule-based, statistical, and AI-based data quality checks to financial time-series data.
*   **Interpreting Data Quality Metrics:** Skill in analyzing completeness, validity, consistency, timeliness, and uniqueness metrics within a financial context.
*   **Quantifying Data Health:** Competence in developing and interpreting a weighted data quality scorecard and overall score.
*   **Identifying and Investigating Anomalies:** Proficiency in drilling down into specific data breaches, missingness patterns, statistical outliers, and AI-detected anomalies.
*   **Applying AI for Data Quality:** Understanding how unsupervised AI models (like Isolation Forest) can augment traditional data quality checks to detect subtle issues.
*   **Setting Proactive Alerts:** Capability to define and monitor critical thresholds for data quality scores and anomaly rates.
*   **Generating Actionable Reports:** Experience in compiling comprehensive data quality impact reports for various stakeholders, including recommendations for remediation.
*   **Understanding Business Impact:** Appreciation for the direct links between data quality, AI model performance, and regulatory compliance in financial services.

## 2. User Interface Requirements

The UI will guide the Quantitative Analyst through a structured, chronological narrative, with each section building upon the previous one. Navigation will be sequential, reinforcing the story flow.

#### Layout & Navigation Structure

The application will feature a multi-page layout, with a sidebar for navigation to allow users to revisit previous steps, and a main content area for the current step's narrative, inputs, and outputs. Each page/section will correspond to a logical step in the Quantitative Analyst's data quality assessment process.

*   **Sidebar:**
    *   Application Title: "AI-Driven Data Quality Scorecard"
    *   Page Links:
        *   "1. Data Overview"
        *   "2. Validity Checks"
        *   "3. Completeness & Consistency"
        *   "4. Timeliness & Duplication"
        *   "5. Data Quality Scorecard"
        *   "6. Investigate Validity Breaches"
        *   "7. Investigate Statistical Anomalies"
        *   "8. AI Anomaly Detection"
        *   "9. Alerts Configuration & Status"
        *   "10. Data Quality Impact Report"
    *   (Optional) "Reset Application" button: Clears session state and restarts.

*   **Main Content Area (Per Page):**
    *   **Header:** `st.header(f"{Page Number}. {Page Title}")`
    *   **Narrative Introduction:** `st.markdown()` explaining the persona's current task and its relevance.
    *   **Input Widgets:** Configurable parameters for the current step.
    *   **Action Button(s):** To trigger calculations or analysis for the current step.
    *   **Results & Visualizations:** Dynamic display of tables, charts, and summaries.
    *   **Narrative Conclusion/Insight:** `st.markdown()` interpreting the results in the persona's context and setting up the next step.

#### Input Widgets and Controls

All widgets will be contextualized within the persona's workflow.

**Page 1: Data Overview**
*   **`st.button("Load Corporate Bond Data")`**: Triggers the generation and loading of `corporate_bond_data.csv`. This represents the persona's initial step of getting data into their environment.
    *   *Purpose:* Initialize the data for analysis.
    *   *Action:* Simulate receiving a new dataset for AI model training.

**Page 2: Validity Checks**
*   **`st.expander("Configure Validity Rules")`**:
    *   **`st.checkbox("Enable Coupon Rate Range Check", value=True)`**: Toggle for rule.
        *   **`st.number_input("Min Coupon Rate", value=0.01, step=0.001, format="%.3f")`**: Input for $R_{min}$.
        *   **`st.number_input("Max Coupon Rate", value=0.10, step=0.001, format="%.3f")`**: Input for $R_{max}$.
        *   *Purpose:* Define acceptable range for `Coupon_Rate`.
        *   *Action:* Persona setting a business rule: "$0.01 \leq \text{Coupon Rate} \leq 0.10$".
    *   **`st.checkbox("Enable Risk Score Positive Check", value=True)`**: Toggle for rule.
        *   **`st.number_input("Min Risk Score", value=0.0, step=0.1)`**: Input for $S_{min}$.
        *   *Purpose:* Ensure `Risk_Score` is non-negative.
        *   *Action:* Persona setting a business rule: "$\text{Risk Score} \geq 0.0$".
    *   **`st.checkbox("Enable Rating Categories Check", value=True)`**: Toggle for rule.
        *   **`st.text_input("Allowed Ratings (comma-separated)", value='AAA,AA,A,BBB,BB,B,CCC,CC,C,D')`**: Input for allowed categorical values.
        *   *Purpose:* Define allowed `Rating` categories.
        *   *Action:* Persona ensuring `Rating` aligns with accepted financial standards.
    *   **`st.checkbox("Enable Volume Non-Negative Check", value=True)`**: Toggle for rule.
        *   **`st.number_input("Min Volume", value=0.0, step=0.1)`**: Input for $V_{min}$.
        *   *Purpose:* Ensure `Volume` is non-negative.
        *   *Action:* Persona setting a business rule: "$\text{Volume} \geq 0.0$".
    *   **`st.checkbox("Enable Maturity Date Future Check", value=True)`**: Toggle for rule.
        *   *Purpose:* Ensure `Maturity_Date` is after `Date`.
        *   *Action:* Persona setting a temporal business rule: "$\text{Maturity Date} > \text{Date}$".
*   **`st.button("Run Validity Checks")`**: Triggers `check_validity_rules` with configured rules.
    *   *Purpose:* Execute the defined business rules against the dataset.
    *   *Action:* Persona applying compliance rules to the data.

**Page 3: Completeness & Consistency**
*   **`st.multiselect("Select Columns for Completeness Check", default=['Volume', 'Rating', 'Risk_Score', 'Index_Value', 'Coupon_Rate'])`**: Columns to check for missing values.
    *   *Purpose:* Specify which columns are critical for completeness.
    *   *Action:* Persona identifying key features for AI models that must not have missing data.
*   **`st.multiselect("Select Numerical Columns for Outlier Detection", default=['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score'])`**: Numerical columns for outlier detection.
    *   *Purpose:* Specify which numerical features need consistency checks.
    *   *Action:* Persona selecting numerical features prone to errors or extreme values.
*   **`st.radio("Outlier Detection Method", options=['IQR', 'Z-Score'], index=0)`**: Chooses method.
    *   *Purpose:* Allows persona to select different statistical approaches for outlier detection.
    *   *Action:* Persona deciding on the appropriate statistical rigor.
*   **`st.slider("IQR Multiplier (k)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)`**: Multiplier $k$ for IQR-based outlier detection (conditional on 'IQR' method).
    *   *Purpose:* Adjust the sensitivity of outlier detection.
    *   *Action:* Persona fine-tuning the definition of an "extreme" outlier.
*   **`st.button("Run Statistical Checks")`**: Triggers `check_completeness_and_consistency`.
    *   *Purpose:* Execute statistical checks.
    *   *Action:* Persona initiating the statistical analysis of data quality.

**Page 4: Timeliness & Duplication**
*   **`st.text_input("Date Column Name", value='Date')`**: Column containing date information for timeliness.
    *   *Purpose:* Identify the primary time-series column.
    *   *Action:* Persona specifying the time anchor for data.
*   **`st.multiselect("Unique Identifier Columns for Duplicates", default=['Bond_ID', 'Date'])`**: Columns forming a unique record key.
    *   *Purpose:* Define the combination of columns that should uniquely identify each record.
    *   *Action:* Persona specifying the granularity for duplicate detection.
*   **`st.button("Run Timeliness & Duplicate Checks")`**: Triggers `check_timeliness_and_duplicates`.
    *   *Purpose:* Execute checks for date gaps and duplicate records.
    *   *Action:* Persona ensuring temporal integrity and non-redundancy.

**Page 5: Data Quality Scorecard**
*   **`st.expander("Configure Dimension Weights")`**:
    *   **`st.slider("Completeness Weight", min_value=0.0, max_value=1.0, value=0.25, step=0.05, key='w_completeness')`**: Weight for completeness score.
    *   **`st.slider("Validity Weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key='w_validity')`**: Weight for validity score.
    *   **`st.slider("Consistency Weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05, key='w_consistency')`**: Weight for consistency score.
    *   **`st.slider("Timeliness Weight", min_value=0.0, max_value=1.0, value=0.10, step=0.05, key='w_timeliness')`**: Weight for timeliness score.
    *   **`st.slider("Uniqueness Weight", min_value=0.0, max_value=1.0, value=0.10, step=0.05, key='w_uniqueness')`**: Weight for uniqueness score.
    *   *Purpose:* Allow persona to assign importance to each data quality dimension.
    *   *Action:* Persona aligning the scoring system with business criticality and regulatory focus.
*   **`st.button("Generate Data Quality Scorecard")`**: Triggers `DataQualityScorer.generate_scorecard`.
    *   *Purpose:* Compute and display the aggregated data quality scores.
    *   *Action:* Persona consolidating all data quality findings into a single, interpretable metric.

**Page 6: Investigate Validity Breaches**
*   **`st.selectbox("Select Breached Rule for Sample Records", options=list(rule_breaches.keys()) if 'rule_breaches' in st.session_state else [])`**: Dynamically populated dropdown with rules that have breaches.
    *   *Purpose:* Allow persona to select a specific rule for deeper investigation.
    *   *Action:* Persona pinpointing a specific type of business rule violation.
*   **`st.number_input("Number of Sample Records", min_value=1, max_value=50, value=10)`**: Number of sample records to display.
    *   *Purpose:* Control the size of the sample for detailed review.
    *   *Action:* Persona reviewing individual problematic records.
*   **`st.button("Show Sample Records")`**: Triggers `get_breach_sample`.
    *   *Purpose:* Display specific records violating the selected rule.
    *   *Action:* Persona examining concrete examples of data flaws.

**Page 7: Investigate Statistical Anomalies**
*   **`st.selectbox("Select Numerical Column to Plot Outliers", options=st.session_state.get('consistency_check_columns', []))`**: Dynamically populated with numerical columns checked for consistency.
    *   *Purpose:* Allow persona to visualize outliers for a specific column.
    *   *Action:* Persona choosing a feature to understand its outlier distribution.
*   **`st.button("Plot Outlier Distribution")`**: Triggers `plot_outlier_distribution`.
    *   *Purpose:* Generate a histogram highlighting outliers for the selected column.
    *   *Action:* Persona visualizing the impact of outliers on data distribution.

**Page 8: AI Anomaly Detection**
*   **`st.multiselect("Features for AI Anomaly Detection", default=st.session_state.get('ai_features', ['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score']))`**: Numerical features to feed into the AI model.
    *   *Purpose:* Select relevant features for complex anomaly detection.
    *   *Action:* Persona defining the input space for the AI model.
*   **`st.slider("Contamination Rate (e.g., % of expected anomalies)", min_value=0.001, max_value=0.10, value=0.015, step=0.001, format="%.3f")`**: Expected proportion of anomalies in the dataset.
    *   *Purpose:* Parameter for Isolation Forest, guiding the model on how many anomalies to expect.
    *   *Action:* Persona providing a prior belief on the prevalence of anomalies.
*   **`st.button("Run AI Anomaly Detection")`**: Triggers `apply_ai_anomaly_detection`.
    *   *Purpose:* Execute the AI-based anomaly detection.
    *   *Action:* Persona leveraging advanced AI to find subtle data shifts.

**Page 9: Alerts Configuration & Status**
*   **`st.number_input("Overall Score Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01)`**: Minimum acceptable overall data quality score.
    *   *Purpose:* Define the lower bound for the overall data quality.
    *   *Action:* Persona setting a critical target for aggregate data health.
*   **`st.number_input("Completeness Score Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01)`**: Minimum acceptable completeness score.
    *   *Purpose:* Define the lower bound for data completeness.
    *   *Action:* Persona ensuring that key features have sufficient data.
*   **`st.number_input("Validity Score Threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)`**: Minimum acceptable validity score.
    *   *Purpose:* Define the lower bound for rule adherence.
    *   *Action:* Persona ensuring strict compliance with business rules.
*   **`st.number_input("AI Anomaly Percentage Threshold", min_value=0.0, max_value=10.0, value=3.0, step=0.1)`**: Maximum acceptable percentage of AI-detected anomalies.
    *   *Purpose:* Define the upper bound for the proportion of complex anomalies.
    *   *Action:* Persona determining the tolerance for subtle data deviations.
*   **`st.button("Check for Critical Alerts")`**: Triggers `check_for_critical_alerts`.
    *   *Purpose:* Evaluate current data quality against defined thresholds.
    *   *Action:* Persona proactively monitoring for data degradation.

**Page 10: Data Quality Impact Report**
*   **`st.button("Generate Impact Report")`**: Triggers `generate_data_quality_impact_report`.
    *   *Purpose:* Compile all findings into a structured, human-readable report.
    *   *Action:* Persona preparing a comprehensive briefing for stakeholders.

#### Visualization Components

All visualizations will be rendered using `matplotlib.pyplot` and `seaborn` (for custom plots) or `st.dataframe` for tabular data.

*   **Initial Data Overview**:
    *   `st.dataframe(df.head())`: Display of the first few rows.
    *   `st.code(df.info().to_string())`: Summary of DataFrame information.
    *   `st.dataframe(df.describe(include='all'))`: Descriptive statistics.
*   **Rule-Based Validity Check Results**:
    *   `st.dataframe(rule_check_results)`: Table summarizing rules, columns, breach counts, breach percentages, and status.
*   **Sample Breached Records**:
    *   `st.dataframe(sample_breaches)`: Table displaying sample records for a selected breached rule.
*   **Completeness Check Results**:
    *   `st.dataframe(completeness_results)`: Table summarizing missing data (`Column`, `Missing_Count`, `Missing_Percentage`, `Status`).
*   **Consistency Check (Outliers) Results**:
    *   `st.dataframe(consistency_results)`: Table summarizing outlier detection (`Column`, `Outlier_Count`, `Outlier_Percentage`, `Status`).
*   **Consistency Check Visualizations**:
    *   `st.pyplot(fig)`: Matplotlib figures for histograms and box plots for numerical consistency check columns, clearly indicating outlier boundaries ($Q1 - k \cdot IQR$ and $Q3 + k \cdot IQR$).
*   **Timeliness and Duplicate Check Results**:
    *   `st.dataframe(timeliness_duplicate_results)`: Table summarizing date gaps and duplicate records.
*   **Sample Duplicate Records/Missing Dates**:
    *   `st.dataframe(td_breaches['duplicate_records'].head())` (if duplicates exist).
    *   `st.dataframe(td_breaches['date_timeliness_gaps'].head())` (if date gaps exist).
*   **Data Quality Scorecard**:
    *   `st.dataframe(data_quality_scorecard)`: Tabular scorecard presenting `Dimension`, `Average_Breach_Percentage`, `Score`, and `Overall Score`.
*   **Dimension Scores Bar Chart**:
    *   `st.pyplot(fig)`: Bar plot visualizing individual data quality dimension scores (0-1 scale) with a horizontal line representing the overall score.
*   **Statistical Issues Summary**:
    *   `st.dataframe(statistical_issues_summary)`: Combined table summarizing missing values and outliers across columns, sorted by percentage.
*   **Outlier Distribution Plot**:
    *   `st.pyplot(fig)`: Histogram showing the distribution of a selected numerical column, with outliers highlighted.
*   **Missingness Matrix**:
    *   `st.pyplot(fig)`: `missingno.matrix` visualization to display patterns of missing data.
*   **AI Anomaly Score Distribution**:
    *   `st.pyplot(fig)`: Histogram of anomaly scores from the Isolation Forest model, distinguishing between normal and anomalous scores.
*   **Anomaly Scores Over Time**:
    *   `st.pyplot(fig)`: Line plot showing anomaly scores over time (for a sampled subset), highlighting detected anomalies.
*   **Alerts Display**:
    *   `st.error()` or `st.warning()`: Clear display of any triggered critical data quality alerts.
*   **Data Quality Impact Report**:
    *   `st.markdown(report_string, unsafe_allow_html=True)`: Formatted text-based summary report.

#### Interactive Elements & Feedback Mechanisms

*   **Buttons:** `st.button()` for advancing the story, triggering calculations, and showing results. Each button click will update the Streamlit state and display new components.
*   **Dynamic Content:** Select boxes and other widgets will be dynamically populated based on previous calculations (e.g., `rule_breaches` for selecting a rule to investigate).
*   **Loading Indicators:** `st.spinner()` will be used for longer-running computations (e.g., AI anomaly detection, initial data generation) to provide user feedback.
*   **Status Messages:** `st.success()`, `st.info()`, `st.warning()`, `st.error()` will communicate the outcome of operations (e.g., "Data Loaded Successfully!", "No critical alerts detected.").
*   **Expander/Collapsible Sections:** `st.expander()` will be used for configuration settings (e.g., rule definitions, dimension weights) to keep the UI clean.

## 3. Additional Requirements

#### Annotations & Tooltips

Contextual explanations will be provided using `st.markdown()` and `st.info()` blocks, drawing heavily from the notebook's narrative and explanations. Tooltips (`help` parameter in widgets) will offer brief, on-demand context.

**Examples:**
*   For the `IQR Multiplier (k)` slider: `help="Adjust the sensitivity of outlier detection. A higher k (e.g., 3.0) focuses on more extreme outliers, crucial in finance where significant deviations might be legitimate events rather than errors."`
*   After displaying `completeness_results`: `st.info("As a Quantitative Analyst, missing data means incomplete information for models, potentially leading to biased training or reduced prediction accuracy. Consider imputation strategies for critical columns.")`
*   For the $S_{overall}$ formula:
    `$$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$`
    `Where $N$ is the number of data quality dimensions, $s_i$ is the normalized score for dimension $i$ (from 0 to 1), and $w_i$ is the weight assigned to dimension $i$.`

#### State Management Requirements

Extensive use of `st.session_state` is critical to maintain the application's narrative flow and prevent data loss.
*   **Persistent Data:** The main DataFrame (`df_bonds`) and all intermediate results (`rule_check_results`, `completeness_results`, `statistical_outliers`, `data_quality_scorecard`, `df_bonds_anomalies`, `current_alerts`, etc.) will be stored in `st.session_state` after each computation.
*   **Input Preservation:** All input widget values (sliders, text inputs, multiselects) will be stored in `st.session_state` to ensure they persist across page navigations and re-runs.
*   **Dependency Management:** If an upstream parameter is changed, dependent downstream outputs will be cleared or marked as "stale," prompting the user to re-run the relevant checks. For simplicity, changing an input might trigger a full re-run of the current section's logic, automatically updating `st.session_state`.
*   **Initialization:** `st.session_state` will be initialized with default values or empty containers at the very beginning of the application if not already present, preventing `KeyError` issues.

## 4. Notebook Content and Code Requirements

All relevant code from the Jupyter Notebook will be extracted and integrated into the Streamlit application. Each major function will be called by an appropriate Streamlit interactive component. Markdown explanations will be rendered as narrative blocks within the UI.

**Mapping Notebook Sections to Streamlit Implementation:**

**Page 1: Data Overview**
*   **Narrative:** `st.markdown()` for "Introduction: The Imperative of Data Quality..." and "1. Setting the Stage..."
*   **Code:**
    ```python
    # In a utility file or at the top of the main app.py
    import pandas as pd
    import numpy as np
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from pyod.models.iforest import IsolationForest
    import missingno as msno

    # --- Data Generation and Loading Functions ---
    def generate_and_load_corporate_bond_data(start_date='2010-01-01', end_date='2019-12-31', num_bonds=10, filename='corporate_bond_data.csv'):
        # ... (full function as provided in notebook) ...
        df = pd.concat(all_data).reset_index(drop=True)
        duplicate_rows = df.sample(n=5, random_state=42)
        df = pd.concat([df, duplicate_rows]).sort_values(by='Date').reset_index(drop=True)
        bond_to_gap = df[df['Bond_ID'] == 'Bond_A'].copy()
        gap_dates = pd.to_datetime(['2015-03-10', '2015-03-11', '2015-03-12'])
        df = df[~((df['Bond_ID'] == 'Bond_A') & (df['Date'].isin(gap_dates)))]
        df.to_csv(filename, index=False)
        return pd.read_csv(filename, parse_dates=['Date', 'Maturity_Date'])

    def load_data(filepath='corporate_bond_data.csv'):
        df = pd.read_csv(filepath, parse_dates=['Date', 'Maturity_Date'])
        return df

    # Streamlit Page Logic:
    # if st.button("Load Corporate Bond Data"):
    #     with st.spinner("Generating and loading data..."):
    #         st.session_state['df_bonds'] = generate_and_load_corporate_bond_data()
    #     st.success("Data loaded successfully!")
    # if 'df_bonds' in st.session_state:
    #     st.dataframe(st.session_state['df_bonds'].head())
    #     buffer = StringIO()
    #     st.session_state['df_bonds'].info(buf=buffer)
    #     st.code(buffer.getvalue())
    #     st.dataframe(st.session_state['df_bonds'].describe(include='all'))
    ```

**Page 2: Validity Checks**
*   **Narrative:** `st.markdown()` for "2. Implementing Rule-Based Checks..." and explanation of results.
*   **Code:**
    ```python
    # --- Rule-Based Validity Function ---
    def check_validity_rules(df, rules):
        # ... (full function as provided in notebook) ...
        return pd.DataFrame(results), breached_records

    # Streamlit Page Logic:
    # # ... (widgets for rule configuration) ...
    # if st.button("Run Validity Checks"):
    #     # Build `validation_rules` dict from widget inputs
    #     st.session_state['rule_check_results'], st.session_state['rule_breaches'] = \
    #         check_validity_rules(st.session_state['df_bonds'], validation_rules)
    # st.dataframe(st.session_state.get('rule_check_results'))
    # if 'risk_score_positive' in st.session_state.get('rule_breaches', {}):
    #     st.dataframe(st.session_state['rule_breaches']['risk_score_positive'].head())
    ```

**Page 3: Completeness & Consistency**
*   **Narrative:** `st.markdown()` for "3. Implementing Statistical Checks..." and interpretation of IQR formula.
    *   **LaTeX:**
        `The formula for IQR-based outlier detection for a data point $x$ in a distribution is:`
        `$$ x < Q1 - k \cdot IQR \quad \text{or} \quad x > Q3 + k \cdot IQR $$`
        `where $Q1$ is the 25th percentile, $Q3$ is the 75th percentile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a chosen multiplier (e.g., $k=3.0$).`
*   **Code:**
    ```python
    # --- Completeness and Consistency Function ---
    def check_completeness_and_consistency(df, completeness_cols, consistency_cols, outlier_method='iqr', iqr_multiplier=1.5):
        # ... (full function as provided in notebook) ...
        # Ensure plt.figure(fig.number) is called before st.pyplot(fig) to re-activate
        return pd.DataFrame(completeness_results), pd.DataFrame(consistency_results), outlier_records, visualizations

    # Streamlit Page Logic:
    # # ... (widgets for column selection, outlier method, iqr_multiplier) ...
    # if st.button("Run Statistical Checks"):
    #     (st.session_state['completeness_results'], st.session_state['consistency_results'],
    #      st.session_state['statistical_outliers'], st.session_state['consistency_plots']) = \
    #         check_completeness_and_consistency(st.session_state['df_bonds'], ...)
    # st.dataframe(st.session_state.get('completeness_results'))
    # st.dataframe(st.session_state.get('consistency_results'))
    # if 'Volume' in st.session_state.get('statistical_outliers', {}):
    #     st.dataframe(st.session_state['statistical_outliers']['Volume'].head())
    # for col, fig in st.session_state.get('consistency_plots', {}).items():
    #     st.pyplot(fig)
    ```

**Page 4: Timeliness & Duplication**
*   **Narrative:** `st.markdown()` for "4. Addressing Timeliness and Duplication..." and interpretation of results.
*   **Code:**
    ```python
    # --- Timeliness and Duplicates Function ---
    def check_timeliness_and_duplicates(df, date_col='Date', id_cols=['Bond_ID', 'Date']):
        # ... (full function as provided in notebook) ...
        return pd.DataFrame(results), breached_records

    # Streamlit Page Logic:
    # # ... (widgets for date_col, id_cols) ...
    # if st.button("Run Timeliness & Duplicate Checks"):
    #     (st.session_state['timeliness_duplicate_results'], st.session_state['td_breaches']) = \
    #         check_timeliness_and_duplicates(st.session_state['df_bonds'], date_col, id_cols)
    # st.dataframe(st.session_state.get('timeliness_duplicate_results'))
    # if 'duplicate_records' in st.session_state.get('td_breaches', {}):
    #     st.dataframe(st.session_state['td_breaches']['duplicate_records'].head())
    # if 'date_timeliness_gaps' in st.session_state.get('td_breaches', {}):
    #     st.dataframe(st.session_state['td_breaches']['date_timeliness_gaps'].head())
    ```

**Page 5: Data Quality Scorecard**
*   **Narrative:** `st.markdown()` for "5. Developing a Data Quality Scoring Framework" and explanation of overall score formula.
    *   **LaTeX:**
        `The overall data quality score $S_{overall}$ for a dataset is calculated as a weighted average of individual dimension scores $s_i$:`
        `$$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$`
        `Where:`
        `- $N$ is the number of data quality dimensions.`
        `- $s_i$ is the normalized score for dimension $i$ (ranging from 0 to 1).`
        `- $w_i$ is the weight assigned to dimension $i$.`
        `For each dimension, the score is typically derived from the breach percentage: $s_i = 1 - \frac{\text{Breach Percentage}_i}{100}$.`
*   **Code:**
    ```python
    # --- Data Quality Scorer Class ---
    class DataQualityScorer:
        def __init__(self, weights=None):
            # ... (full class as provided in notebook) ...
        def _calculate_dimension_score(self, breach_percentage):
            return max(0.0, 1.0 - (breach_percentage / 100.0))
        def generate_scorecard(self, completeness_df, validity_df, consistency_df, timeliness_duplicate_df):
            # ... (full method as provided in notebook) ...
            return scorecard_df, dimension_scores

    # Streamlit Page Logic:
    # # ... (widgets for weights) ...
    # if st.button("Generate Data Quality Scorecard"):
    #     custom_weights = {k: st.session_state[f'w_{k}'] for k in ['completeness', 'validity', 'consistency', 'timeliness', 'uniqueness']}
    #     dq_scorer = DataQualityScorer(weights=custom_weights)
    #     (st.session_state['data_quality_scorecard'], st.session_state['dimension_scores_dict']) = \
    #         dq_scorer.generate_scorecard(st.session_state['completeness_results'], st.session_state['rule_check_results'],
    #                                      st.session_state['consistency_results'], st.session_state['timeliness_duplicate_results'])
    # st.dataframe(st.session_state.get('data_quality_scorecard'))
    # fig_scorecard, ax_scorecard = plt.subplots(figsize=(10, 6))
    # sns.barplot(x=list(st.session_state['dimension_scores_dict'].keys()), y=list(st.session_state['dimension_scores_dict'].values()), ax=ax_scorecard, palette='viridis')
    # # ... (add title, labels, axhline as in notebook) ...
    # st.pyplot(fig_scorecard)
    ```

**Page 6: Investigate Validity Breaches**
*   **Narrative:** `st.markdown()` for "6. Investigating Rule-Based Breaches..." and impact.
*   **Code:**
    ```python
    # --- Breach Summary Functions ---
    def summarize_rule_breaches(rule_check_results, breached_records):
        # ... (full function as provided in notebook) ...
        return breached_summary
    def get_breach_sample(breached_records, rule_name, n=5):
        # ... (full function as provided in notebook) ...
        return pd.DataFrame()

    # Streamlit Page Logic:
    # breached_rules_summary = summarize_rule_breaches(st.session_state.get('rule_check_results'), st.session_state.get('rule_breaches'))
    # st.dataframe(breached_rules_summary)
    # # ... (widgets for selectbox and number_input for sample) ...
    # if st.button("Show Sample Records"):
    #     sample_breaches = get_breach_sample(st.session_state['rule_breaches'], selected_rule_name, n=num_samples)
    #     st.dataframe(sample_breaches)
    ```

**Page 7: Investigate Statistical Anomalies**
*   **Narrative:** `st.markdown()` for "7. Investigating Statistical Anomalies..." and impact.
*   **Code:**
    ```python
    # --- Statistical Issue Summary and Plotting Functions ---
    def summarize_statistical_issues(completeness_df, consistency_df):
        # ... (full function as provided in notebook) ...
        return combined_summary
    def plot_outlier_distribution(df, outlier_records, column_name):
        # ... (full function as provided in notebook) ...
        # Ensure plt.figure(fig.number) is called before st.pyplot(fig) to re-activate
        plt.show() # This ensures the figure is closed for Streamlit rendering

    # Streamlit Page Logic:
    # statistical_issues_summary = summarize_statistical_issues(st.session_state.get('completeness_results'), st.session_state.get('consistency_results'))
    # st.dataframe(statistical_issues_summary)
    # # ... (widget for selectbox for outlier column) ...
    # if st.button("Plot Outlier Distribution"):
    #     fig_outlier_dist, ax_outlier_dist = plt.subplots(figsize=(10, 6))
    #     plot_outlier_distribution(st.session_state['df_bonds'], st.session_state['statistical_outliers'], selected_outlier_col)
    #     st.pyplot(fig_outlier_dist)
    # st.markdown("\n--- Visualizing Missing Data Patterns ---")
    # fig_missingno, ax_missingno = plt.subplots(figsize=(12, 7))
    # msno.matrix(st.session_state['df_bonds'], ax=ax_missingno, sparkline=False)
    # st.pyplot(fig_missingno)
    ```

**Page 8: AI Anomaly Detection**
*   **Narrative:** `st.markdown()` for "8. Advanced AI-Based Anomaly Detection..." and impact.
*   **Code:**
    ```python
    # --- AI Anomaly Detection Functions ---
    def apply_ai_anomaly_detection(df, features_for_anomaly_detection, contamination_rate=0.01, random_state=42):
        # ... (full function as provided in notebook) ...
        return df_with_anomalies, model, anomaly_summary
    def plot_anomaly_scores(df_with_anomalies, anomaly_score_col='anomaly_score', is_anomaly_col='is_anomaly'):
        # ... (full function as provided in notebook) ...
        # Ensure plt.figure(fig.number) is called before st.pyplot(fig) to re-activate
        plt.show() # Close figures for Streamlit rendering

    # Streamlit Page Logic:
    # # ... (widgets for features, contamination_rate) ...
    # if st.button("Run AI Anomaly Detection"):
    #     (st.session_state['df_bonds_anomalies'], st.session_state['if_model'],
    #      st.session_state['anomaly_summary_dict']) = \
    #         apply_ai_anomaly_detection(st.session_state['df_bonds'], ai_features, contamination_rate)
    # st.info(f"Anomaly Count: {st.session_state['anomaly_summary_dict']['Anomaly_Count']}")
    # st.dataframe(st.session_state['df_bonds_anomalies'][st.session_state['df_bonds_anomalies']['is_anomaly']].head())
    # fig_anomaly_scores_hist, ax_anomaly_scores_hist = plt.subplots(figsize=(12, 6))
    # # Call plot_anomaly_scores which handles figure creation internally
    # plot_anomaly_scores(st.session_state['df_bonds_anomalies'])
    # # st.pyplot() calls are handled internally by plot_anomaly_scores now.
    ```

**Page 9: Alerts Configuration & Status**
*   **Narrative:** `st.markdown()` for "9. Developing a Continuous Monitoring System..." and impact.
*   **Code:**
    ```python
    # --- Alerting Function ---
    def check_for_critical_alerts(dimension_scores, data_quality_scorecard, anomaly_summary,
                                  overall_score_threshold=0.75,
                                  completeness_score_threshold=0.80,
                                  validity_score_threshold=0.80,
                                  ai_anomaly_percentage_threshold=2.0):
        # ... (full function as provided in notebook) ...
        return alerts

    # Streamlit Page Logic:
    # # ... (widgets for thresholds) ...
    # if st.button("Check for Critical Alerts"):
    #     st.session_state['current_alerts'] = check_for_critical_alerts(
    #         st.session_state['dimension_scores_dict'], st.session_state['data_quality_scorecard'],
    #         st.session_state['anomaly_summary_dict'], overall_score_threshold, completeness_score_threshold,
    #         validity_score_threshold, ai_anomaly_percentage_threshold
    #     )
    # if st.session_state.get('current_alerts'):
    #     for alert in st.session_state['current_alerts']:
    #         st.error(alert)
    # else:
    #     st.success("No critical data quality alerts detected.")
    ```

**Page 10: Data Quality Impact Report**
*   **Narrative:** `st.markdown()` for "10. Generating the Data Quality Impact Report" and impact.
*   **Code:**
    ```python
    # --- Report Generation Function ---
    def generate_data_quality_impact_report(data_quality_scorecard, rule_check_results, statistical_issues_summary, anomaly_summary, current_alerts):
        # ... (full function as provided in notebook) ...
        return report_df, "\n".join(report_summary_str)

    # Streamlit Page Logic:
    # if st.button("Generate Impact Report"):
    #     report_df, report_string = generate_data_quality_impact_report(
    #         st.session_state['data_quality_scorecard'],
    #         summarize_rule_breaches(st.session_state['rule_check_results'], st.session_state['rule_breaches']),
    #         summarize_statistical_issues(st.session_state['completeness_results'], st.session_state['consistency_results']),
    #         st.session_state['anomaly_summary_dict'],
    #         st.session_state['current_alerts']
    #     )
    #     st.markdown(report_string, unsafe_allow_html=True)
    ```

Each `matplotlib.pyplot` figure generated by the backend functions will be explicitly rendered using `st.pyplot(fig_object)`. To avoid overlapping plots or memory issues, `plt.close(fig_object)` will be called after `st.pyplot()` in the Streamlit code, or the plotting functions modified to accept an `ax` object or return the `fig` object directly for Streamlit to handle. The current notebook code for plotting functions directly calls `plt.show()`; this will need to be changed to `return fig` and then `st.pyplot(fig)` in Streamlit, followed by `plt.close(fig)`.

