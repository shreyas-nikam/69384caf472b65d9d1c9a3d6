
# Streamlit Application Specification: AI-Powered Data Quality for Financial Bonds

## 1. Application Overview

As a Quantitative Analyst in a DataOps team, my primary responsibility is to ensure the integrity and reliability of data feeding our AI models in a highly regulated financial environment. Unreliable data can lead to catastrophic consequences: inaccurate risk assessments, non-compliant reporting, and significant financial losses. This Streamlit application will guide me through a comprehensive, story-driven workflow to proactively assess, monitor, and report on the data quality of a simulated corporate bond dataset.

The application will demonstrate a realistic scenario where I, as the Quantitative Analyst, am tasked with establishing a robust data quality scoring system. I will interact with the system to uncover hidden data quality issues using a multi-faceted approach, including rule-based validation, statistical anomaly detection, and advanced AI-driven techniques. The app will then help me aggregate these findings into a unified data quality score, define critical alerts, and generate a comprehensive impact report for stakeholders.

The core problem addressed is the imperative for **trustworthy and compliant AI models** in finance, which relies entirely on **high-quality, auditable data**. This application provides an interactive sandbox to understand and apply advanced data quality management principles.

**Learning Goals (Applied Skills for the Persona):**
By interacting with this application, I will gain practical experience in:
-   **Configuring and applying rule-based data validation** based on financial domain expertise.
-   **Performing statistical checks** for data completeness and consistency, including outlier detection.
-   **Identifying and quantifying timeliness and uniqueness issues** in time-series financial data.
-   **Developing a weighted data quality scoring framework** to provide a holistic view of data health.
-   **Leveraging AI-based unsupervised anomaly detection** to uncover subtle data deviations.
-   **Establishing critical thresholds** for continuous data quality monitoring and alert generation.
-   **Translating complex data quality metrics into actionable business insights** through comprehensive reporting.
-   **Understanding the iterative process** of data quality assessment and its direct impact on AI model robustness and regulatory compliance.

## 2. User Interface Requirements

The UI will follow a chronological narrative structure, presented as sequential sections on a single Streamlit page, ensuring a smooth, story-driven workflow. Each section will correspond to a logical step in the Quantitative Analyst's data quality assessment process. Navigation will be implicit through completion of tasks, with results from one step informing the next.

### Layout & Navigation Structure

The application will use a single page, with `st.container` elements for each major section, potentially wrapped in `st.expander` elements that allow the user to focus on the current step while keeping previous results accessible. `st.session_state` will be extensively used to manage data and analysis results across sections.

**Overall Page Title:** "AI-Powered Data Quality for Financial Bonds: A Quantitative Analyst's Workflow"

**Page Sections (Chronological Flow):**

#### **Step 1: Setting the Stage: Data Loading & Initial Exploration**
-   **Narrative:** `st.markdown("## 1. Setting the Stage: Loading and Initial Exploration of Financial Bond Data\n\nAs a Quantitative Analyst in a DataOps team, my first step is always to get a clear picture of the raw data. This involves loading it and performing initial sanity checks to identify immediate red flags. We'll use a simulated dataset, `corporate_bond_data.csv`, mimicking real-world financial data with inherent quality challenges.")`
-   **Interactive Elements:**
    -   `st.button("Load Corporate Bond Data", key="load_data")`: Initiates the data generation and loading process.
-   **Outputs:**
    -   `st.success("Simulated data loaded successfully!")` (upon completion).
    -   `st.subheader("First 5 rows of the dataset:")`
    -   `st.dataframe(df_bonds.head())`
    -   `st.subheader("Dataset Info:")`
    -   `st.text(df_bonds.info(verbose=True, buf=StringIO()))` (captures info() output).
    -   `st.subheader("Basic Descriptive Statistics:")`
    -   `st.dataframe(df_bonds.describe(include='all'))`
-   **Annotations:** `st.info("The initial data load and descriptive statistics reveal a dataset of several thousand bond records. The 'df.info()' output already hints at potential issues, particularly non-null counts that are less than the total number of entries for some columns, indicating missing values. This quick scan is crucial for understanding the scale of the data and gives a preliminary sense of where quality issues might lie.")`

#### **Step 2: Implementing Rule-Based Checks: Ensuring Validity**
-   **Narrative:** `st.markdown("## 2. Implementing Rule-Based Checks: Ensuring Validity\n\nIn a regulated financial environment, many data quality checks are based on predefined business rules and domain knowledge. These 'rule-based' checks are fundamental for ensuring data **validity**. Failing these rules can lead to incorrect model assumptions or misreporting to regulatory bodies.")`
-   **Input Widgets:**
    -   **Expandable Configuration for Rules:** `st.expander("Configure Validity Rules")`
        -   `st.text_area("Validation Rules (JSON format)", value=json.dumps(validation_rules, indent=2), height=200, key="validity_rules_config")`: Allows the persona to review and potentially modify rule definitions.
        -   `st.button("Apply and Run Validity Checks", key="run_validity_checks")`
-   **Outputs:**
    -   `st.subheader("Rule-Based Validity Check Results:")`
    -   `st.dataframe(rule_check_results)`
    -   `st.subheader("Sample of Breached Records:")`
    -   `st.selectbox("Select Rule to View Sample Breaches", options=list(rule_breaches.keys()), key="select_validity_breach")` (Only if `rule_breaches` is not empty).
    -   `st.dataframe(rule_breaches[st.session_state.select_validity_breach].head(10))` (If a rule is selected).
-   **Annotations:** `st.info("The rule-based checks quickly identify critical validity issues. For instance, a negative 'Risk_Score' or a 'Coupon_Rate' outside the expected 1% to 10% band are direct violations of business logic. As a QA, I know these data points could lead to flawed risk calculations or non-compliant financial reporting if used in AI models.")`
    -   `st.markdown("For example, the rule for `Coupon_Rate` checks if its value $x$ satisfies $0.01 \le x \le 0.10$. Similarly, `Risk_Score` must satisfy $x \ge 0.0$.")`

#### **Step 3: Implementing Statistical Checks: Completeness and Consistency**
-   **Narrative:** `st.markdown("## 3. Implementing Statistical Checks: Completeness and Consistency\n\nBeyond explicit rules, data quality also encompasses **completeness** (are all expected values present?) and **consistency** (are values coherent and free from anomalies?). Statistical methods are excellent for identifying these issues, especially outliers that might signal erroneous data entry or unusual market events.")`
-   **Input Widgets:**
    -   `st.subheader("Outlier Detection Configuration:")`
    -   `st.radio("Outlier Method", options=['IQR', 'Z-Score'], index=0, key="outlier_method")`
    -   `st.slider("IQR Multiplier (k)", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="iqr_multiplier")` (Only visible if 'IQR' method is selected).
    -   `st.button("Run Statistical Checks", key="run_statistical_checks")`
-   **Outputs:**
    -   `st.subheader("Completeness Check Results (Missing Values):")`
    -   `st.dataframe(completeness_results)`
    -   `st.subheader("Consistency Check Results (Outliers):")`
    -   `st.dataframe(consistency_results)`
    -   `st.subheader("Visualizing Missing Data Patterns:")`
    -   `st.pyplot(missingno.matrix(df_bonds, figsize=(12, 7), sparkline=False))`
    -   `st.subheader("Outlier Distributions:")`
    -   `st.selectbox("Select Column to Visualize Outliers", options=consistency_check_columns, key="select_outlier_column")` (Only if `consistency_check_columns` are not empty).
    -   `st.pyplot(consistency_plots[st.session_state.select_outlier_column])` (Displays both histogram and boxplot for selected column).
-   **Annotations:** `st.info("The completeness check confirms missing values in critical columns like 'Volume' and 'Rating'. Missing data means incomplete information for models, potentially leading to biased training. Consistency checks highlight extreme outliers (using $k=3.0$ for IQR) in numerical features. These deviations must be carefully handled for model robustness.")`
    -   `st.markdown("The formula for IQR-based outlier detection for a data point $x$ in a distribution is: $$ x < Q1 - k \cdot IQR \\ \text{or} \\ x > Q3 + k \cdot IQR $$ where $Q1$ is the 25th percentile, $Q3$ is the 75th percentile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is the chosen multiplier.")`

#### **Step 4: Addressing Timeliness and Duplication: Enhancing Data Reliability**
-   **Narrative:** `st.markdown("## 4. Addressing Timeliness and Duplication: Enhancing Data Reliability\n\nBeyond validity, completeness, and consistency, other crucial data quality dimensions for financial AI are **timeliness** (is the data current and complete for expected periods?) and the absence of **duplicates**. Late or duplicated data can skew analyses and lead to incorrect signals for AI models.")`
-   **Input Widgets:**
    -   `st.multiselect("Unique Identifier Columns for Duplicates Check", options=df_bonds.columns.tolist(), default=['Bond_ID', 'Date'], key="id_cols_for_duplicates")`
    -   `st.button("Run Timeliness & Duplicate Checks", key="run_timeliness_duplicates")`
-   **Outputs:**
    -   `st.subheader("Timeliness and Duplicate Check Results:")`
    -   `st.dataframe(timeliness_duplicate_results)`
    -   `st.subheader("Sample of Records Identified as Duplicates:")`
    -   `st.dataframe(td_breaches.get('duplicate_records', pd.DataFrame()).head(10))`
    -   `st.subheader("Sample of Missing Dates (Timeliness Gaps):")`
    -   `st.dataframe(td_breaches.get('date_timeliness_gaps', pd.DataFrame()).head(10))`
-   **Annotations:** `st.info("Date gaps for specific bonds are critical for time-series AI models, as they expect continuous data. Duplicate records artificially inflate data counts and bias statistical measures. Addressing these ensures our AI models are trained on clean, non-redundant, and properly sequenced historical data.")`

#### **Step 5: Developing a Data Quality Scoring Framework**
-   **Narrative:** `st.markdown("## 5. Developing a Data Quality Scoring Framework\n\nWith individual data quality dimensions defined and checked, my next task is to aggregate these findings into a unified, interpretable Data Quality Score. This score provides a high-level overview, allowing me to quickly assess the overall health of the dataset and communicate it effectively to stakeholders. The framework uses a weighted average approach.")`
-   **Input Widgets:**
    -   `st.subheader("Configure Dimension Weights:")`
    -   `col1, col2, col3, col4, col5 = st.columns(5)`
    -   `col1.slider("Completeness", 0.0, 1.0, 0.25, 0.01, key="weight_completeness")`
    -   `col2.slider("Validity", 0.0, 1.0, 0.35, 0.01, key="weight_validity")`
    -   `col3.slider("Consistency", 0.0, 1.0, 0.20, 0.01, key="weight_consistency")`
    -   `col4.slider("Timeliness", 0.0, 1.0, 0.10, 0.01, key="weight_timeliness")`
    -   `col5.slider("Uniqueness", 0.0, 1.0, 0.10, 0.01, key="weight_uniqueness")`
    -   `st.button("Calculate Data Quality Scorecard", key="calculate_dq_score")`
-   **Outputs:**
    -   `st.subheader("Data Quality Scorecard:")`
    -   `st.dataframe(data_quality_scorecard)`
    -   `st.subheader("Visualizing Dimension Scores:")`
    -   `st.pyplot(dimension_scores_plot)` (Bar chart with overall score line).
-   **Annotations:** `st.info("The Data Quality Scorecard provides a consolidated view of our dataset's health. The individual dimension scores and an overall weighted score of $S_{overall}=X.XX$ offer an immediate, quantified assessment. This is invaluable for a QA, telling me which areas (e.g., 'Validity' or 'Completeness') require priority attention.")`
    -   `st.markdown("The overall data quality score $S_{overall}$ for a dataset is calculated as a weighted average of individual dimension scores $s_i$: $$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$ Where $s_i$ is the normalized score for dimension $i$ (from $0$ to $1$) and $w_i$ is its assigned weight.")`

#### **Step 6: Investigating Rule-Based Breaches and Statistical Anomalies**
-   **Narrative:** `st.markdown("## 6. Investigating Breaches (Rule-Based & Statistical)\n\nA low Validity or Consistency score from our scorecard signals significant issues. As a Quantitative Analyst, I need to pinpoint exactly *which* rules are most frequently breached and what specific data points or statistical issues are causing these violations. This granular insight allows me to understand the potential impact on our AI models and initiate targeted data cleaning or process improvements.")`
-   **Outputs:**
    -   `st.subheader("Summary of Most Breached Rules:")`
    -   `st.dataframe(breached_rules_summary)` (Sorted by Breach_Percentage).
    -   `st.selectbox("Select a Breached Rule to View Sample Records", options=breached_rules_summary['Rule'].tolist(), key="select_breach_rule_sample")`
    -   `st.dataframe(sample_breaches_for_selected_rule)`
    -   `st.subheader("Summary of Top Statistical Data Quality Issues (Missing Values & Outliers):")`
    -   `st.dataframe(statistical_issues_summary)`
    -   `st.subheader("Visualizing Data Issues:")`
    -   `st.pyplot(msno.matrix(df_bonds, figsize=(12, 7), sparkline=False))` (Re-display for context).
    -   `st.selectbox("Re-select Column to Plot Outliers", options=consistency_check_columns, key="select_outlier_column_replot")`
    -   `st.pyplot(consistency_plots[st.session_state.select_outlier_column_replot])` (Re-display histogram and boxplot for selected column).
-   **Annotations:** `st.info("The summary highlights critical violations, such as negative 'Risk_Scores' or out-of-range 'Coupon_Rates', which could drastically distort risk assessments. Similarly, high missingness in 'Volume' or extreme outliers in 'Index_Value' require immediate attention. These issues directly impact AI model accuracy and regulatory compliance.")`

#### **Step 7: Advanced AI-Based Anomaly Detection for Subtle Shifts**
-   **Narrative:** `st.markdown("## 7. Advanced AI-Based Anomaly Detection for Subtle Shifts\n\nWhile rule-based and statistical checks catch many issues, some data quality problems are more subtle, manifesting as complex deviations that a simple threshold might miss. This is where AI-based anomaly detection algorithms, like Isolation Forest, become invaluable. As a QA, I use these to identify data points that are 'different' from the norm, even if they don't explicitly violate a rule.")`
-   **Input Widgets:**
    -   `st.multiselect("Numerical Features for AI Anomaly Detection", options=df_bonds.select_dtypes(include=np.number).columns.tolist(), default=['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score'], key="ai_features_selection")`
    -   `st.slider("Assumed Contamination Rate (proportion of outliers in data)", min_value=0.001, max_value=0.1, value=0.015, step=0.001, key="contamination_rate_slider")`
    -   `st.button("Run AI Anomaly Detection (Isolation Forest)", key="run_ai_anomaly_detection")`
-   **Outputs:**
    -   `st.subheader("AI-Based Anomaly Detection Summary:")`
    -   `st.write(f"Total Records: {anomaly_summary_dict['Total_Records']}")`
    -   `st.write(f"Anomaly Count: {anomaly_summary_dict['Anomaly_Count']}")`
    -   `st.write(f"Anomaly Percentage: {anomaly_summary_dict['Anomaly_Percentage']:.2f}%")`
    -   `st.subheader("Sample of AI-Detected Anomalous Records:")`
    -   `st.dataframe(df_bonds_anomalies[df_bonds_anomalies['is_anomaly']].head(10))`
    -   `st.subheader("Distribution of Anomaly Scores:")`
    -   `st.pyplot(anomaly_score_hist_plot)`
    -   `st.subheader("Anomaly Scores Over Time (Sampled Data):")`
    -   `st.pyplot(anomaly_score_time_plot)`
-   **Annotations:** `st.info("The Isolation Forest model identified a set of data points as anomalous. These aren't necessarily 'errors' but complex deviations across features. For a QA, these could signal new market regimes, unusual trading activities, or early warnings of data corruption. Investigating these helps refine data preprocessing and design more robust AI models.")`

#### **Step 8: Developing a Continuous Monitoring System: Alerts for Critical Deviations**
-   **Narrative:** `st.markdown("## 8. Developing a Continuous Monitoring System: Alerts for Critical Deviations\n\nOur data quality scoring system can now identify issues. The next crucial step is to integrate these insights into a continuous monitoring system that generates alerts for critical deviations. This enables proactive intervention before poor data quality impacts downstream AI applications or regulatory reporting. As a QA, I define the thresholds that trigger these alerts.")`
-   **Input Widgets:**
    -   `st.subheader("Configure Alert Thresholds:")`
    -   `st.slider("Overall DQ Score Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01, key="overall_dq_threshold")`
    -   `st.slider("Completeness Score Threshold", min_value=0.0, max_value=1.0, value=0.75, step=0.01, key="completeness_dq_threshold")`
    -   `st.slider("Validity Score Threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01, key="validity_dq_threshold")`
    -   `st.slider("AI Anomaly Percentage Threshold", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="ai_anomaly_percent_threshold")`
    -   `st.button("Check for Critical Alerts", key="check_alerts_button")`
-   **Outputs:**
    -   `st.subheader("Current Data Quality Alerts:")`
    -   (Conditional display based on `current_alerts` list)
        -   `st.warning(f"- {alert}")` for each alert.
        -   `st.success("No critical data quality alerts detected. Data quality is within acceptable limits.")`
-   **Annotations:** `st.info("This alert system is the operational heartbeat of our DataOps framework. Critical alerts, like an overall score below $0.70$ or validity below $0.85$, demand immediate attention as they pose direct threats to AI model reliability and regulatory compliance. Defining these explicit thresholds ensures focus on the most impactful issues.")`

#### **Step 9: Generating the Data Quality Impact Report**
-   **Narrative:** `st.markdown("## 9. Generating the Data Quality Impact Report\n\nThe final step in our data quality workflow is to consolidate all findings into a comprehensive Data Quality Impact Report. This report is crucial for communicating the current state of data quality, highlighting specific issues, explaining their potential impact on AI models and business operations, and recommending remediation strategies. As a QA, I'm responsible for translating complex data quality metrics into clear, actionable business insights for management and data owners.")`
-   **Interactive Elements:**
    -   `st.button("Generate Data Quality Impact Report", key="generate_report_button")`
-   **Outputs:**
    -   `st.subheader("Data Quality Impact Report:")`
    -   `st.markdown(report_summary_str)` (Displays the full formatted report).
    -   `st.download_button("Download Report as Text", data=report_summary_str, file_name="Data_Quality_Impact_Report.txt", mime="text/plain")`
-   **Annotations:** `st.info("This report is the culmination of our efforts. It provides a clear, structured narrative on the data's state, crucial for AI model training. I use this to inform stakeholders, prioritize action (e.g., address negative 'Risk_Scores' or missing 'Volume' data), and drive continuous improvement in our data pipelines, ensuring trustworthy AI analytics.")`

### Input Widgets and Controls (Detailed)

-   **`st.button("Load Corporate Bond Data", key="load_data")`**
    -   **Purpose in Story:** Persona initiates the data loading process.
    -   **Action:** Simulates fetching and parsing a new dataset for analysis.
    -   **Parameters:** None directly visible to user.
-   **`st.text_area("Validation Rules (JSON format)", value=json.dumps(validation_rules, indent=2), height=200, key="validity_rules_config")`**
    -   **Purpose in Story:** Persona reviews or customizes the business rules.
    -   **Action:** Allows dynamic configuration of validation logic, reflecting real-world policy adjustments.
    -   **Parameters:**
        -   `value`: Default JSON representation of `validation_rules`.
        -   `height`: 200 pixels.
        -   `key`: "validity_rules_config".
-   **`st.button("Apply and Run Validity Checks", key="run_validity_checks")`**
    -   **Purpose in Story:** Persona commits to the defined rules and runs the checks.
    -   **Action:** Triggers `check_validity_rules` with the (potentially updated) rules.
-   **`st.selectbox("Select Rule to View Sample Breaches", options=list(rule_breaches.keys()), key="select_validity_breach")`**
    -   **Purpose in Story:** Persona drills down into specific rule violations.
    -   **Action:** Selects a rule to view problematic records for targeted investigation.
    -   **Parameters:** `options`: List of rule names with breaches.
-   **`st.radio("Outlier Method", options=['IQR', 'Z-Score'], index=0, key="outlier_method")`**
    -   **Purpose in Story:** Persona chooses a statistical method for outlier detection.
    -   **Action:** Decides which mathematical approach (IQR or Z-score) to use, reflecting different statistical assumptions.
    -   **Parameters:** `options`: ['IQR', 'Z-Score'], `index`: 0 (IQR by default).
-   **`st.slider("IQR Multiplier (k)", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="iqr_multiplier")`**
    -   **Purpose in Story:** Persona adjusts the sensitivity of outlier detection.
    -   **Action:** Controls the $k$ multiplier in the IQR formula $Q1 - k \cdot IQR$ and $Q3 + k \cdot IQR$, allowing for detection of mild or extreme outliers. Default value of $3.0$ aligns with detecting *extreme* outliers.
    -   **Parameters:** `min_value`: 1.0, `max_value`: 5.0, `value`: 3.0, `step`: 0.1.
-   **`st.button("Run Statistical Checks", key="run_statistical_checks")`**
    -   **Purpose in Story:** Persona executes the configured statistical checks.
    -   **Action:** Triggers `check_completeness_and_consistency`.
-   **`st.multiselect("Unique Identifier Columns for Duplicates Check", options=df_bonds.columns.tolist(), default=['Bond_ID', 'Date'], key="id_cols_for_duplicates")`**
    -   **Purpose in Story:** Persona defines what constitutes a unique record.
    -   **Action:** Selects key columns to identify composite duplicates, reflecting domain knowledge of unique identifiers.
    -   **Parameters:** `options`: All DataFrame columns. `default`: `['Bond_ID', 'Date']`.
-   **`st.button("Run Timeliness & Duplicate Checks", key="run_timeliness_duplicates")`**
    -   **Purpose in Story:** Persona runs checks for time-series gaps and duplicate entries.
    -   **Action:** Triggers `check_timeliness_and_duplicates`.
-   **`st.slider("Completeness Weight", 0.0, 1.0, 0.25, 0.01, key="weight_completeness")` (and similar for Validity, Consistency, Timeliness, Uniqueness)**
    -   **Purpose in Story:** Persona prioritizes different data quality dimensions.
    -   **Action:** Assigns criticality weights to each dimension, influencing the overall data quality score according to business context (e.g., compliance, model sensitivity). Weights are normalized internally.
    -   **Parameters:** `min_value`: 0.0, `max_value`: 1.0, `value`: various defaults, `step`: 0.01.
-   **`st.button("Calculate Data Quality Scorecard", key="calculate_dq_score")`**
    -   **Purpose in Story:** Persona aggregates all quality metrics into a single scorecard.
    -   **Action:** Triggers `DataQualityScorer.generate_scorecard` with current weights.
-   **`st.multiselect("Numerical Features for AI Anomaly Detection", options=df_bonds.select_dtypes(include=np.number).columns.tolist(), default=['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score'], key="ai_features_selection")`**
    -   **Purpose in Story:** Persona selects features relevant for advanced anomaly detection.
    -   **Action:** Defines the input space for the AI model, crucial for focusing the anomaly detection on key financial metrics.
    -   **Parameters:** `options`: Numerical DataFrame columns. `default`: Core numerical columns.
-   **`st.slider("Assumed Contamination Rate (proportion of outliers in data)", min_value=0.001, max_value=0.1, value=0.015, step=0.001, key="contamination_rate_slider")`**
    -   **Purpose in Story:** Persona provides a prior estimate for the proportion of anomalies.
    -   **Action:** Informs the Isolation Forest model about the expected percentage of anomalies, which affects its sensitivity.
    -   **Parameters:** `min_value`: 0.001, `max_value`: 0.1, `value`: 0.015, `step`: 0.001.
-   **`st.button("Run AI Anomaly Detection (Isolation Forest)", key="run_ai_anomaly_detection")`**
    -   **Purpose in Story:** Persona initiates the AI-driven anomaly detection.
    -   **Action:** Triggers `apply_ai_anomaly_detection`.
-   **`st.slider("Overall DQ Score Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01, key="overall_dq_threshold")` (and similar for Completeness, Validity, AI Anomaly Percentage)**
    -   **Purpose in Story:** Persona sets critical boundaries for data quality.
    -   **Action:** Defines the operational thresholds for triggering alerts, directly influencing proactive interventions.
    -   **Parameters:** `min_value`: 0.0 (or 0.0 for percentage), `max_value`: 1.0 (or 10.0 for percentage), `value`: various defaults, `step`: 0.01 (or 0.1 for percentage).
-   **`st.button("Check for Critical Alerts", key="check_alerts_button")`**
    -   **Purpose in Story:** Persona evaluates the current data quality against defined alert thresholds.
    -   **Action:** Triggers `check_for_critical_alerts`.
-   **`st.button("Generate Data Quality Impact Report", key="generate_report_button")`**
    -   **Purpose in Story:** Persona compiles all findings into a stakeholder-ready report.
    -   **Action:** Triggers `generate_data_quality_impact_report`.

### Visualization Components

All visualizations are generated using `matplotlib.pyplot` and `seaborn` as per the notebook and displayed using `st.pyplot()`.

-   **`st.dataframe()` displays:**
    -   `df_bonds.head()`: Initial data preview.
    -   `df_bonds.describe(include='all')`: Basic statistics.
    -   `rule_check_results`: Summary of validity breaches.
    -   `rule_breaches[selected_rule].head(10)`: Sample records for a selected breached rule.
    -   `completeness_results`: Missing value summary.
    -   `consistency_results`: Outlier summary.
    -   `timeliness_duplicate_results`: Timeliness and duplicate check summary.
    -   `td_breaches['duplicate_records'].head(10)`: Sample duplicate records.
    -   `td_breaches['date_timeliness_gaps'].head(10)`: Sample missing dates.
    -   `data_quality_scorecard`: Consolidated data quality scores.
    -   `breached_rules_summary`: Top rule breaches.
    -   `statistical_issues_summary`: Top statistical issues.
    -   `df_bonds_anomalies[df_bonds_anomalies['is_anomaly']].head(10)`: Sample AI-detected anomalies.
-   **`missingno.matrix(df_bonds, figsize=(12, 7), sparkline=False)`**: Visualizes patterns of missing data.
-   **`consistency_plots[col]` (histograms and boxplots):** Generated for numerical columns to visualize distributions and outliers.
    -   Example for 'Volume':
        -   Histogram: `sns.histplot(numeric_data, kde=True)` with IQR bounds as `axvline`.
        -   Boxplot: `sns.boxplot(y=numeric_data)`.
-   **Data Quality Dimension Scores Bar Chart:**
    -   Bar chart: `sns.barplot(x=list(dimension_scores_dict.keys()), y=list(dimension_scores_dict.values()))`.
    -   Overall score line: `plt.axhline(y=overall_score, color='red', linestyle='--')`.
-   **Anomaly Score Distribution Histogram:**
    -   Histogram: `sns.histplot(df_with_anomalies['anomaly_score'], kde=True, color='blue')`
    -   Highlighted anomalies: `sns.histplot(df_with_anomalies[df_with_anomalies['is_anomaly']]['anomaly_score'], kde=True, color='red', alpha=0.6)`.
-   **Anomaly Scores Over Time Line Plot:**
    -   Line plot: `sns.lineplot(x='Date', y='anomaly_score', hue='is_anomaly', data=df_bonds_anomalies.sort_values('Date').sample(...))`.

### Interactive Elements & Feedback Mechanisms

-   **Buttons:** Trigger specific analysis steps (`Load Data`, `Apply Checks`, `Calculate Score`, `Run Anomaly Detection`, `Check Alerts`, `Generate Report`).
-   **Sliders/Radio Buttons/Multi-selects/Text Areas:** Allow dynamic adjustment of parameters (rule definitions, IQR multiplier, weights, thresholds, features for AI), directly impacting results.
-   **Conditional Rendering:** Sections are revealed or expanded as the user progresses through the story. For example, statistical checks only appear after rule-based checks are run.
-   **Success/Warning/Info Messages:** `st.success()`, `st.warning()`, `st.info()` will provide immediate feedback on actions (e.g., data loaded, alerts triggered, no issues found).
-   **Dropdowns for Samples:** Allow granular exploration of breached/anomalous records.
-   **Download Button:** For the final impact report, allowing the persona to export their findings.

## 3. Additional Requirements

### Annotations & Tooltips

-   All output tables and charts will be accompanied by `st.info()` or `st.markdown()` blocks, explaining the significance of the results *in the context of the Quantitative Analyst's role* and the financial scenario. These will tie back to model accuracy, regulatory compliance, and business impact.
-   Key interactive widgets will have tooltips (if supported by Streamlit or simulated with `st.sidebar.markdown`) to explain their purpose and effect on the analysis in a persona-aligned way (e.g., "Adjust the IQR multiplier to detect more extreme or milder outliers, impacting data cleaning strategy for AI models").

### State Management Requirements

-   All user inputs from widgets (sliders, text areas, select boxes) must be stored in `st.session_state` to ensure that their values persist across page reruns and user interactions.
-   Crucial intermediate data (`df_bonds`, `rule_check_results`, `rule_breaches`, `completeness_results`, `consistency_results`, `statistical_outliers`, `timeliness_duplicate_results`, `td_breaches`, `data_quality_scorecard`, `dimension_scores_dict`, `df_bonds_anomalies`, `anomaly_summary_dict`, `current_alerts`) must be stored in `st.session_state` after their respective computation steps. This allows subsequent functions to access the most current state of the analysis without re-computation (unless an upstream parameter changes).
-   The application must gracefully handle scenarios where a required `st.session_state` variable is not yet populated (e.g., show a placeholder or a prompt to run a previous step).

## 4. Notebook Content and Code Requirements

The Streamlit application will directly integrate the Python functions and logic from the provided Jupyter Notebook. Each logical step in the notebook will correspond to a section in the Streamlit app.

**Global Imports (Streamlit app start):**
```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import json
from datetime import timedelta, datetime
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IsolationForest
from io import StringIO # For df.info()
```

**Helper Functions from Notebook (to be defined in app):**

1.  **`generate_and_load_corporate_bond_data`:**
    -   **Integration:** Called by `st.button("Load Corporate Bond Data")`. The returned `df` will be stored in `st.session_state.df_bonds`.
    -   **Code Stub:**
        ```python
        def generate_and_load_corporate_bond_data(start_date='2010-01-01', end_date='2019-12-31', num_bonds=10, filename='corporate_bond_data.csv'):
            dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
            all_data = []
            for i in range(num_bonds):
                bond_id = f'Bond_{chr(65 + i)}'
                df_bond = pd.DataFrame({'Date': dates})
                df_bond['Bond_ID'] = bond_id
                base_value = 100 + i * 5
                df_bond['Index_Value'] = base_value + np.cumsum(np.random.normal(0, 0.5, len(dates)))
                df_bond['Volume'] = np.random.randint(1000, 100000, len(dates))
                df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.05)), 'Volume'] = np.nan
                df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.02)), 'Volume'] = 0
                df_bond.loc[np.random.choice(df_bond.index, 5), 'Volume'] = np.random.randint(500000, 1000000, 5)
                df_bond['Coupon_Rate'] = np.random.uniform(0.02, 0.08, len(dates))
                df_bond.loc[np.random.choice(df_bond.index, 10), 'Coupon_Rate'] = np.random.uniform(-0.01, 0.15, 10)
                ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
                df_bond['Rating'] = np.random.choice(ratings, len(dates))
                df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.03)), 'Rating'] = np.nan
                df_bond.loc[np.random.choice(df_bond.index, 5), 'Rating'] = 'Invalid_Rating'
                df_bond['Maturity_Date'] = pd.to_datetime(start_date) + timedelta(days=np.random.randint(365*2, 365*10, len(dates)))
                df_bond.loc[np.random.choice(df_bond.index, 5), 'Maturity_Date'] = pd.to_datetime(start_date) - timedelta(days=np.random.randint(100, 500, 5))
                df_bond['Risk_Score'] = np.random.uniform(1.0, 7.0, len(dates))
                df_bond.loc[np.random.choice(df_bond.index, 8), 'Risk_Score'] = np.random.uniform(-5.0, 15.0, 8)
                df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.01)), 'Risk_Score'] = np.nan
                regions = ['North America', 'Europe', 'Asia', 'South America']
                df_bond['Issuer_Region'] = np.random.choice(regions, len(dates))
                df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.02)), 'Issuer_Region'] = 'Unknown'
                all_data.append(df_bond)
            df = pd.concat(all_data).reset_index(drop=True)
            duplicate_rows = df.sample(n=5, random_state=42)
            df = pd.concat([df, duplicate_rows]).sort_values(by='Date').reset_index(drop=True)
            bond_to_gap = df[df['Bond_ID'] == 'Bond_A'].copy()
            gap_dates = pd.to_datetime(['2015-03-10', '2015-03-11', '2015-03-12'])
            df = df[~((df['Bond_ID'] == 'Bond_A') & (df['Date'].isin(gap_dates)))]
            # df.to_csv(filename, index=False) # Not writing to file for Streamlit
            return df.copy() # return a copy to avoid SettingWithCopyWarning
        def load_data(filepath='corporate_bond_data.csv'): # Not used in Streamlit directly, as data is generated
            df = pd.read_csv(filepath, parse_dates=['Date', 'Maturity_Date'])
            return df
        ```

2.  **`check_validity_rules`:**
    -   **Integration:** Called by `st.button("Apply and Run Validity Checks")`. Takes `st.session_state.df_bonds` and JSON-parsed `st.session_state.validity_rules_config`. Returns `rule_check_results` and `rule_breaches` to `st.session_state`.
    -   **Code Stub:**
        ```python
        def check_validity_rules(df, rules):
            results = []
            breached_records = {}
            for rule_name, params in rules.items():
                column = params['column']
                check_type = params['check_type']
                breach_mask = pd.Series(False, index=df.index)
                if column not in df.columns:
                    results.append({'Rule': rule_name, 'Column': column, 'Breaches': 0, 'Breach_Percentage': 0.0, 'Status': 'Column Not Found'})
                    continue
                if check_type == 'range':
                    lower_bound = params['lower_bound']
                    upper_bound = params['upper_bound']
                    breach_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                elif check_type == 'min_value':
                    min_value = params['min_value']
                    breach_mask = (df[column] < min_value)
                elif check_type == 'max_value':
                    max_value = params['max_value']
                    breach_mask = (df[column] > max_value)
                elif check_type == 'allowed_values':
                    allowed_values = set(params['allowed_values'])
                    breach_mask = ~df[column].isin(allowed_values) & df[column].notna()
                elif check_type == 'future_date':
                    reference_date_column = params['reference_date_column']
                    breach_mask = df[column] < df[reference_date_column]
                else:
                    results.append({'Rule': rule_name, 'Column': column, 'Breaches': 0, 'Breach_Percentage': 0.0, 'Status': f'Unknown check_type {check_type}'})
                    continue
                breaches_count = breach_mask.sum()
                breach_percentage = (breaches_count / len(df)) * 100 if len(df) > 0 else 0.0
                results.append({
                    'Rule': rule_name,
                    'Column': column,
                    'Breaches': breaches_count,
                    'Breach_Percentage': round(breach_percentage, 2),
                    'Status': 'Breached' if breaches_count > 0 else 'Passed'
                })
                if breaches_count > 0:
                    breached_records[rule_name] = df[breach_mask].copy()
            return pd.DataFrame(results), breached_records
        ```

3.  **`check_completeness_and_consistency`:**
    -   **Integration:** Called by `st.button("Run Statistical Checks")`. Takes `st.session_state.df_bonds`, selected `outlier_method` and `iqr_multiplier`. Returns `completeness_results`, `consistency_results`, `statistical_outliers`, `consistency_plots` to `st.session_state`.
    -   **Code Stub:**
        ```python
        def check_completeness_and_consistency(df, completeness_cols, consistency_cols, outlier_method='iqr', iqr_multiplier=1.5):
            completeness_results = []
            for col in completeness_cols:
                if col not in df.columns:
                    completeness_results.append({'Column': col, 'Missing_Count': 0, 'Missing_Percentage': 0.0, 'Status': 'Column Not Found'})
                    continue
                missing_count = df[col].isnull().sum()
                total_count = len(df)
                missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0.0
                completeness_results.append({
                    'Column': col, 'Missing_Count': missing_count, 'Missing_Percentage': round(missing_percentage, 2),
                    'Status': 'Incomplete' if missing_count > 0 else 'Complete'
                })
            consistency_results = []
            outlier_records = {}
            visualizations = {}
            for col in consistency_cols:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    consistency_results.append({'Column': col, 'Outlier_Count': 0, 'Outlier_Percentage': 0.0, 'Status': 'Skipped (Non-numeric/Not Found)'})
                    continue
                numeric_data = df[col].dropna()
                if len(numeric_data) == 0:
                    consistency_results.append({'Column': col, 'Outlier_Count': 0, 'Outlier_Percentage': 0.0, 'Status': 'No Numeric Data'})
                    continue
                outlier_mask = pd.Series(False, index=df.index)
                fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # Create figure here
                if outlier_method.lower() == 'iqr':
                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    sns.histplot(numeric_data, kde=True, ax=axes[0])
                    axes[0].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
                    axes[0].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
                    axes[0].set_title(f'Distribution of {col} with IQR Bounds')
                    axes[0].legend()
                    sns.boxplot(y=numeric_data, ax=axes[1])
                    axes[1].set_title(f'Box Plot of {col}')
                elif outlier_method.lower() == 'z_score':
                    mean = numeric_data.mean()
                    std_dev = numeric_data.std()
                    z_scores = np.abs((df[col] - mean) / std_dev)
                    threshold = 3 # Common threshold for z-score
                    outlier_mask = (z_scores > threshold)
                    sns.histplot(numeric_data, kde=True, ax=axes[0])
                    axes[0].set_title(f'Distribution of {col}')
                    sns.scatterplot(x=df.index, y=df[col], hue=outlier_mask, ax=axes[1])
                    axes[1].set_title(f'Z-Score Outliers in {col}')
                else:
                    plt.close(fig) # Close unused figure
                    consistency_results.append({'Column': col, 'Outlier_Count': 0, 'Outlier_Percentage': 0.0, 'Status': f'Unknown outlier_method {outlier_method}'})
                    continue
                plt.tight_layout()
                visualizations[col] = fig # Store the figure object
                outlier_count = outlier_mask.sum()
                outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0.0
                consistency_results.append({
                    'Column': col, 'Outlier_Count': outlier_count, 'Outlier_Percentage': round(outlier_percentage, 2),
                    'Status': 'Outliers Detected' if outlier_count > 0 else 'No Outliers'
                })
                if outlier_count > 0:
                    outlier_records[col] = df[outlier_mask].copy()
            return pd.DataFrame(completeness_results), pd.DataFrame(consistency_results), outlier_records, visualizations
        ```

4.  **`check_timeliness_and_duplicates`:**
    -   **Integration:** Called by `st.button("Run Timeliness & Duplicate Checks")`. Takes `st.session_state.df_bonds` and `st.session_state.id_cols_for_duplicates`. Returns `timeliness_duplicate_results` and `td_breaches` to `st.session_state`.
    -   **Code Stub:**
        ```python
        def check_timeliness_and_duplicates(df, date_col='Date', id_cols=['Bond_ID', 'Date']):
            results = []
            breached_records = {}
            df_sorted = df.sort_values(by=[id_cols[0], date_col])
            date_gaps_count = 0
            date_gaps_df = pd.DataFrame(columns=['Bond_ID', 'Missing_Date'])
            for bond_id in df_sorted[id_cols[0]].unique():
                bond_df = df_sorted[df_sorted[id_cols[0]] == bond_id].copy()
                if len(bond_df) > 1:
                    expected_dates = pd.date_range(start=bond_df[date_col].min(), end=bond_df[date_col].max(), freq='D')
                    actual_dates = pd.Index(bond_df[date_col].unique())
                    missing_dates = expected_dates.difference(actual_dates)
                    if len(missing_dates) > 0:
                        date_gaps_count += len(missing_dates)
                        temp_df = pd.DataFrame({'Bond_ID': bond_id, 'Missing_Date': missing_dates})
                        date_gaps_df = pd.concat([date_gaps_df, temp_df], ignore_index=True)
            total_expected_days_across_bonds = 0
            for bond_id in df_sorted[id_cols[0]].unique():
                bond_df = df_sorted[df_sorted[id_cols[0]] == bond_id].copy()
                if len(bond_df) > 1:
                    total_expected_days_across_bonds += (bond_df[date_col].max() - bond_df[date_col].min()).days + 1
            date_gap_percentage = (date_gaps_count / total_expected_days_across_bonds) * 100 if total_expected_days_across_bonds > 0 else 0.0
            results.append({
                'Rule': 'date_timeliness_gaps', 'Column': date_col, 'Breaches': date_gaps_count,
                'Breach_Percentage': round(date_gap_percentage, 2), 'Status': 'Gaps Detected' if date_gaps_count > 0 else 'No Gaps'
            })
            if date_gaps_count > 0:
                breached_records['date_timeliness_gaps'] = date_gaps_df
            duplicate_mask = df.duplicated(subset=id_cols, keep=False)
            duplicate_count = duplicate_mask.sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0.0
            results.append({
                'Rule': 'duplicate_records', 'Column': ', '.join(id_cols), 'Breaches': duplicate_count,
                'Breach_Percentage': round(duplicate_percentage, 2), 'Status': 'Duplicates Detected' if duplicate_count > 0 else 'No Duplicates'
            })
            if duplicate_count > 0:
                breached_records['duplicate_records'] = df[duplicate_mask].sort_values(by=id_cols).copy()
            return pd.DataFrame(results), breached_records
        ```

5.  **`DataQualityScorer` class:**
    -   **Integration:** Instantiated with weights from `st.session_state` sliders. `generate_scorecard` method called by `st.button("Calculate Data Quality Scorecard")`. Returns `data_quality_scorecard` and `dimension_scores_dict` to `st.session_state`.
    -   **Code Stub:**
        ```python
        class DataQualityScorer:
            def __init__(self, weights=None):
                self.weights = weights if weights is not None else {
                    'completeness': 0.25, 'validity': 0.30, 'consistency': 0.20,
                    'timeliness': 0.15, 'uniqueness': 0.10
                }
                total_weight = sum(self.weights.values())
                if total_weight == 0:
                    raise ValueError("Total weights cannot be zero.")
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
            def _calculate_dimension_score(self, breach_percentage):
                return max(0.0, 1.0 - (breach_percentage / 100.0))
            def generate_scorecard(self, completeness_df, validity_df, consistency_df, timeliness_duplicate_df):
                scorecard_data = []
                dimension_scores = {}
                avg_missing_percentage = completeness_df['Missing_Percentage'].mean() if not completeness_df.empty else 0
                dimension_scores['completeness'] = self._calculate_dimension_score(avg_missing_percentage)
                scorecard_data.append({'Dimension': 'Completeness', 'Average_Breach_Percentage': round(avg_missing_percentage, 2), 'Score': round(dimension_scores['completeness'], 2)})
                avg_validity_breach_percentage = validity_df['Breach_Percentage'].mean() if not validity_df.empty else 0
                dimension_scores['validity'] = self._calculate_dimension_score(avg_validity_breach_percentage)
                scorecard_data.append({'Dimension': 'Validity', 'Average_Breach_Percentage': round(avg_validity_breach_percentage, 2), 'Score': round(dimension_scores['validity'], 2)})
                avg_consistency_breach_percentage = consistency_df['Outlier_Percentage'].mean() if not consistency_df.empty else 0
                dimension_scores['consistency'] = self._calculate_dimension_score(avg_consistency_breach_percentage)
                scorecard_data.append({'Dimension': 'Consistency', 'Average_Breach_Percentage': round(avg_consistency_breach_percentage, 2), 'Score': round(dimension_scores['consistency'], 2)})
                timeliness_breach = timeliness_duplicate_df[timeliness_duplicate_df['Rule'] == 'date_timeliness_gaps']
                avg_timeliness_breach_percentage = timeliness_breach['Breach_Percentage'].iloc[0] if not timeliness_breach.empty else 0
                dimension_scores['timeliness'] = self._calculate_dimension_score(avg_timeliness_breach_percentage)
                scorecard_data.append({'Dimension': 'Timeliness', 'Average_Breach_Percentage': round(avg_timeliness_breach_percentage, 2), 'Score': round(dimension_scores['timeliness'], 2)})
                uniqueness_breach = timeliness_duplicate_df[timeliness_duplicate_df['Rule'] == 'duplicate_records']
                avg_uniqueness_breach_percentage = uniqueness_breach['Breach_Percentage'].iloc[0] if not uniqueness_breach.empty else 0
                dimension_scores['uniqueness'] = self._calculate_dimension_score(avg_uniqueness_breach_percentage)
                scorecard_data.append({'Dimension': 'Uniqueness', 'Average_Breach_Percentage': round(avg_uniqueness_breach_percentage, 2), 'Score': round(dimension_scores['uniqueness'], 2)})
                overall_score = sum(dimension_scores[dim] * self.weights.get(dim, 0) for dim in dimension_scores)
                scorecard_df = pd.DataFrame(scorecard_data)
                scorecard_df.loc['Overall'] = ['Overall Score', np.nan, round(overall_score, 2)]
                return scorecard_df, dimension_scores
        ```

6.  **`summarize_rule_breaches` & `get_breach_sample`:**
    -   **Integration:** `summarize_rule_breaches` called to populate `st.session_state.breached_rules_summary`. `get_breach_sample` called to display samples based on `st.session_state.select_breach_rule_sample`.
    -   **Code Stub:**
        ```python
        def summarize_rule_breaches(rule_check_results):
            breached_summary = rule_check_results[rule_check_results['Breaches'] > 0].sort_values(by='Breach_Percentage', ascending=False)
            return breached_summary
        def get_breach_sample(breached_records, rule_name, n=5):
            if rule_name in breached_records:
                return breached_records[rule_name].head(n)
            return pd.DataFrame()
        ```

7.  **`summarize_statistical_issues` & `plot_outlier_distribution`:**
    -   **Integration:** `summarize_statistical_issues` called to populate `st.session_state.statistical_issues_summary`. `plot_outlier_distribution` used to display plots.
    -   **Code Stub:**
        ```python
        def summarize_statistical_issues(completeness_df, consistency_df):
            completeness_summary = completeness_df[completeness_df['Missing_Count'] > 0].rename(columns={'Missing_Count': 'Count', 'Missing_Percentage': 'Percentage', 'Status': 'Issue_Type'})
            completeness_summary['Issue'] = 'Missing Values'
            completeness_summary = completeness_summary[['Column', 'Issue', 'Count', 'Percentage']]
            consistency_summary = consistency_df[consistency_df['Outlier_Count'] > 0].rename(columns={'Outlier_Count': 'Count', 'Outlier_Percentage': 'Percentage', 'Status': 'Issue_Type'})
            consistency_summary['Issue'] = 'Outliers'
            consistency_summary = consistency_summary[['Column', 'Issue', 'Count', 'Percentage']]
            combined_summary = pd.concat([completeness_summary, consistency_summary]).sort_values(by='Percentage', ascending=False)
            return combined_summary
        def plot_outlier_distribution(df, outlier_records, column_name):
            if column_name in outlier_records:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[column_name].dropna(), kde=True, color='skyblue', label='All Data', ax=ax)
                sns.histplot(outlier_records[column_name][column_name].dropna(), kde=True, color='red', label='Outliers', alpha=0.6, ax=ax)
                ax.set_title(f'Distribution of {column_name} with Outliers Highlighted')
                ax.set_xlabel(column_name)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                return fig
            return None
        ```

8.  **`apply_ai_anomaly_detection` & `plot_anomaly_scores`:**
    -   **Integration:** `apply_ai_anomaly_detection` called by `st.button("Run AI Anomaly Detection")`. Returns `df_bonds_anomalies`, `if_model`, `anomaly_summary_dict` to `st.session_state`. `plot_anomaly_scores` used to display the related plots.
    -   **Code Stub:**
        ```python
        def apply_ai_anomaly_detection(df, features_for_anomaly_detection, contamination_rate=0.01, random_state=42):
            df_processed = df[features_for_anomaly_detection].copy()
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_processed)
            model = IsolationForest(contamination=contamination_rate, random_state=random_state, n_estimators=100, behaviour='new')
            model.fit(scaled_features)
            df_with_anomalies = df.copy()
            df_with_anomalies['anomaly_score'] = model.decision_function(scaled_features)
            df_with_anomalies['is_anomaly'] = model.predict(scaled_features)
            df_with_anomalies['is_anomaly'] = df_with_anomalies['is_anomaly'] == -1
            anomaly_count = df_with_anomalies['is_anomaly'].sum()
            total_records = len(df_with_anomalies)
            anomaly_percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0.0
            anomaly_summary = {
                'Anomaly_Count': anomaly_count, 'Anomaly_Percentage': round(anomaly_percentage, 2),
                'Total_Records': total_records
            }
            return df_with_anomalies, model, anomaly_summary
        def plot_anomaly_scores(df_with_anomalies, anomaly_score_col='anomaly_score', is_anomaly_col='is_anomaly'):
            fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
            sns.histplot(df_with_anomalies[anomaly_score_col], kde=True, color='blue', label='Normal Scores', ax=ax_hist)
            sns.histplot(df_with_anomalies[df_with_anomalies[is_anomaly_col]][anomaly_score_col], kde=True, color='red', label='Anomalous Scores', alpha=0.6, ax=ax_hist)
            ax_hist.set_title('Distribution of Anomaly Scores from Isolation Forest')
            ax_hist.set_xlabel('Anomaly Score (Lower is More Anomalous)')
            ax_hist.set_ylabel('Frequency')
            ax_hist.legend()
            ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
            
            fig_time, ax_time = plt.subplots(figsize=(12, 6))
            sampled_df = df_with_anomalies.sort_values('Date').sample(n=min(len(df_with_anomalies), 1000), random_state=42)
            sns.lineplot(x='Date', y='anomaly_score', hue='is_anomaly', data=sampled_df, 
                         palette={True: 'red', False: 'blue'}, style='is_anomaly', 
                         markers=True, dashes=False, size='is_anomaly', sizes=(50, 10), ax=ax_time)
            ax_time.set_title('Anomaly Scores Over Time (Sampled Data)')
            ax_time.set_xlabel('Date')
            ax_time.set_ylabel('Anomaly Score')
            ax_time.grid(True, linestyle='--', alpha=0.6)
            
            return fig_hist, fig_time
        ```

9.  **`check_for_critical_alerts`:**
    -   **Integration:** Called by `st.button("Check for Critical Alerts")`. Takes `st.session_state` values. Returns `current_alerts` list to `st.session_state`.
    -   **Code Stub:**
        ```python
        def check_for_critical_alerts(dimension_scores, data_quality_scorecard, anomaly_summary,
                                      overall_score_threshold=0.75, completeness_score_threshold=0.80,
                                      validity_score_threshold=0.80, ai_anomaly_percentage_threshold=2.0):
            alerts = []
            overall_score = data_quality_scorecard.loc['Overall', 'Score']
            if overall_score < overall_score_threshold:
                alerts.append(f"CRITICAL ALERT: Overall Data Quality Score ({overall_score:.2f}) is below threshold ({overall_score_threshold:.2f}).")
            if dimension_scores.get('completeness', 1.0) < completeness_score_threshold:
                alerts.append(f"ALERT: Completeness Score ({dimension_scores.get('completeness', 1.0):.2f}) is below threshold ({completeness_score_threshold:.2f}).")
            if dimension_scores.get('validity', 1.0) < validity_score_threshold:
                alerts.append(f"CRITICAL ALERT: Validity Score ({dimension_scores.get('validity', 1.0):.2f}) is below threshold ({validity_score_threshold:.2f}).")
            if anomaly_summary['Anomaly_Percentage'] > ai_anomaly_percentage_threshold:
                alerts.append(f"ALERT: AI-detected anomaly percentage ({anomaly_summary['Anomaly_Percentage']:.2f}%) exceeds threshold ({ai_anomaly_percentage_threshold:.2f}%).")
            return alerts
        ```

10. **`generate_data_quality_impact_report`:**
    -   **Integration:** Called by `st.button("Generate Data Quality Impact Report")`. Takes `st.session_state` values. Returns `report_summary_str` to `st.session_state`.
    -   **Code Stub:**
        ```python
        def generate_data_quality_impact_report(data_quality_scorecard, rule_check_results, statistical_issues_summary, anomaly_summary, current_alerts):
            report_summary_str = []
            report_summary_str.append(f"--- Data Quality Impact Report ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
            report_summary_str.append("\n## Executive Summary")
            overall_score = data_quality_scorecard.loc['Overall', 'Score']
            report_summary_str.append(f"The current overall data quality score is: {overall_score:.2f} (out of 1.0).")
            if current_alerts:
                report_summary_str.append(f"**{len(current_alerts)} Critical Alerts have been triggered:**")
                for alert in current_alerts:
                    report_summary_str.append(f"- {alert}")
            else:
                report_summary_str.append("No critical data quality alerts were triggered. Data quality is within acceptable parameters.")
            report_summary_str.append("\n## Detailed Data Quality Scorecard")
            report_summary_str.append(data_quality_scorecard.to_string())
            report_summary_str.append("\n## Key Breaches and Anomalies")
            breached_rules_summary = rule_check_results[rule_check_results['Breaches'] > 0].sort_values(by='Breach_Percentage', ascending=False)
            if not breached_rules_summary.empty:
                report_summary_str.append("\n### Rule-Based Validity Issues (Top 5 by percentage):")
                report_summary_str.append(breached_rules_summary.head(5).to_string())
                report_summary_str.append("\n*Impact:* Direct violations of business logic and regulatory requirements. Can lead to flawed financial calculations, incorrect risk assessments, and non-compliant reporting by AI models.")
            if not statistical_issues_summary.empty:
                report_summary_str.append("\n### Statistical Data Issues (Top 5 by percentage):")
                report_summary_str.append(statistical_issues_summary.head(5).to_string())
                report_summary_str.append("\n*Impact:* Missing values can introduce bias and reduce AI model accuracy. Outliers can lead to models overfitting to noise, producing unstable predictions or false positives in fraud detection.")
            if anomaly_summary['Anomaly_Count'] > 0:
                report_summary_str.append("\n### AI-Detected Complex Anomalies:")
                report_summary_str.append(f"Isolation Forest identified {anomaly_summary['Anomaly_Count']} records ({anomaly_summary['Anomaly_Percentage']:.2f}%) as anomalous.")
                report_summary_str.append("\n*Impact:* These subtle deviations, though not always explicit errors, can signal evolving data patterns, potential data drift, or emerging risks. AI models trained without addressing these might become less robust over time.")
            report_summary_str.append("\n## Recommendations")
            report_summary_str.append("- Prioritize remediation for rules with high breach percentages (e.g., negative Risk_Scores, out-of-range Coupon_Rates).")
            report_summary_str.append("- Implement data imputation strategies for critical columns with high missingness (e.g., Volume, Rating).")
            report_summary_str.append("- Investigate the root cause of AI-detected anomalies to distinguish between data errors and genuine market events.")
            report_summary_str.append("- Establish data validation checkpoints upstream in the data pipeline to prevent issues from propagating.")
            
            # For Streamlit, we primarily return the string.
            return "\n".join(report_summary_str)
        ```

**Initial Data & Rule Definitions (within the Streamlit app's main script, often in `if 'df_bonds' not in st.session_state:` blocks):**
```python
# Default validation rules for initial load, as defined in notebook
default_validation_rules = {
    'coupon_rate_range': {'column': 'Coupon_Rate', 'check_type': 'range', 'lower_bound': 0.01, 'upper_bound': 0.10},
    'risk_score_positive': {'column': 'Risk_Score', 'check_type': 'min_value', 'min_value': 0.0},
    'rating_categories': {'column': 'Rating', 'check_type': 'allowed_values', 'allowed_values': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']},
    'volume_non_negative': {'column': 'Volume', 'check_type': 'min_value', 'min_value': 0.0},
    'maturity_date_future': {'column': 'Maturity_Date', 'check_type': 'future_date', 'reference_date_column': 'Date'}
}
completeness_check_columns = ['Volume', 'Rating', 'Risk_Score', 'Index_Value', 'Coupon_Rate']
consistency_check_columns = ['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score']
id_columns_for_duplicates = ['Bond_ID', 'Date']
```
All markdown explanations from the notebook will be incorporated as `st.markdown` or `st.info` within the corresponding Streamlit sections to maintain the narrative flow and provide context for the persona's actions. The application will ensure that calculations are performed sequentially, reflecting the persona's step-by-step decision-making process.

