
# Building a Data Quality Scoring System for AI in Regulated Finance

## Introduction: The Imperative of Data Quality in Financial AI

As a Quantitative Analyst in a DataOps team at a leading financial institution, I'm acutely aware that the integrity of our AI models hinges entirely on the quality of the data we feed them. In a regulated environment, biased or unreliable AI outputs aren't just suboptimal; they carry significant compliance, financial, and reputational risks. My current mandate is to establish a robust data quality scoring system for our AI training datasets, ensuring our models remain trustworthy and compliant.

This notebook will walk through my process:
1.  **Understanding our data** and identifying potential quality pitfalls.
2.  **Defining and quantifying key data quality dimensions** using rule-based and statistical checks.
3.  **Developing a holistic scoring system** to provide a clear, actionable overview of data quality.
4.  **Implementing AI-based anomaly detection** to catch subtle, complex data quality deviations.
5.  **Generating an impact report** to inform stakeholders and drive remediation efforts.

Our goal is to move beyond reactive issue-fixing to a proactive, continuous data quality management framework that underpins reliable AI.

## 1. Setting the Stage: Loading and Initial Exploration of Financial Bond Data

Our AI models, particularly those involved in risk assessment and portfolio optimization, rely heavily on financial time-series data, such as corporate bond indices. Before we can build any models, my first step is always to get a clear picture of the raw data. This involves loading it and performing initial sanity checks to identify immediate red flags.

For this lab, we will use a simulated dataset, `corporate_bond_data.csv`, designed to mimic real-world financial data with inherent data quality challenges such as missing values, outliers, and inconsistencies, which are common in regulated financial environments.

### Code Cell (Function Definition)

```python
# Function to generate and load a simulated corporate bond dataset with injected quality issues
# This function will create a CSV file and load it into a pandas DataFrame.
# It will simulate: missing values, outliers, invalid ranges, inconsistent categories, duplicate rows, and date gaps.
import pandas as pd
import numpy as np
from datetime import timedelta

def generate_and_load_corporate_bond_data(start_date='2010-01-01', end_date='2019-12-31', num_bonds=10, filename='corporate_bond_data.csv'):
    """
    Generates a simulated corporate bond dataset with various data quality issues.
    """
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
    
    all_data = []
    for i in range(num_bonds):
        bond_id = f'Bond_{chr(65 + i)}'
        
        df_bond = pd.DataFrame({'Date': dates})
        df_bond['Bond_ID'] = bond_id
        
        # Index_Value (simulated trend with noise)
        base_value = 100 + i * 5
        df_bond['Index_Value'] = base_value + np.cumsum(np.random.normal(0, 0.5, len(dates)))
        
        # Volume (with missing values, zeros, and outliers)
        df_bond['Volume'] = np.random.randint(1000, 100000, len(dates))
        df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.05)), 'Volume'] = np.nan # 5% missing
        df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.02)), 'Volume'] = 0 # 2% zero
        df_bond.loc[np.random.choice(df_bond.index, 5), 'Volume'] = np.random.randint(500000, 1000000, 5) # Outliers
        
        # Coupon_Rate (with invalid ranges)
        df_bond['Coupon_Rate'] = np.random.uniform(0.02, 0.08, len(dates))
        df_bond.loc[np.random.choice(df_bond.index, 10), 'Coupon_Rate'] = np.random.uniform(-0.01, 0.15, 10) # Invalid range
        
        # Rating (with missing values and inconsistent categories)
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
        df_bond['Rating'] = np.random.choice(ratings, len(dates))
        df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.03)), 'Rating'] = np.nan # 3% missing
        df_bond.loc[np.random.choice(df_bond.index, 5), 'Rating'] = 'Invalid_Rating' # Inconsistent category
        
        # Maturity_Date (future dates, with some past dates for inconsistency)
        df_bond['Maturity_Date'] = pd.to_datetime(start_date) + timedelta(days=np.random.randint(365*2, 365*10, len(dates)))
        df_bond.loc[np.random.choice(df_bond.index, 5), 'Maturity_Date'] = pd.to_datetime(start_date) - timedelta(days=np.random.randint(100, 500, 5)) # Past dates
        
        # Risk_Score (with outliers and negative values)
        df_bond['Risk_Score'] = np.random.uniform(1.0, 7.0, len(dates))
        df_bond.loc[np.random.choice(df_bond.index, 8), 'Risk_Score'] = np.random.uniform(-5.0, 15.0, 8) # Outliers/invalid range
        df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.01)), 'Risk_Score'] = np.nan # 1% missing
        
        # Issuer_Region (with 'Unknown' values)
        regions = ['North America', 'Europe', 'Asia', 'South America']
        df_bond['Issuer_Region'] = np.random.choice(regions, len(dates))
        df_bond.loc[np.random.choice(df_bond.index, int(len(df_bond) * 0.02)), 'Issuer_Region'] = 'Unknown' # Inconsistent/placeholder

        all_data.append(df_bond)
    
    df = pd.concat(all_data).reset_index(drop=True)
    
    # Simulate some full duplicate rows
    duplicate_rows = df.sample(n=5, random_state=42)
    df = pd.concat([df, duplicate_rows]).sort_values(by='Date').reset_index(drop=True)

    # Simulate date gaps (for one bond)
    bond_to_gap = df[df['Bond_ID'] == 'Bond_A'].copy()
    gap_dates = pd.to_datetime(['2015-03-10', '2015-03-11', '2015-03-12'])
    df = df[~((df['Bond_ID'] == 'Bond_A') & (df['Date'].isin(gap_dates)))]

    df.to_csv(filename, index=False)
    print(f"Simulated data saved to {filename}")
    return pd.read_csv(filename, parse_dates=['Date', 'Maturity_Date'])

def load_data(filepath='corporate_bond_data.csv'):
    """Loads the corporate bond data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=['Date', 'Maturity_Date'])
    return df
```

### Code Cell (Function Execution)

```python
# Generate and load the simulated dataset
bond_data_filepath = 'corporate_bond_data.csv'
df_bonds = generate_and_load_corporate_bond_data(filename=bond_data_filepath)

# Display the first few rows and basic information
print("First 5 rows of the dataset:")
print(df_bonds.head())
print("\nDataset Info:")
df_bonds.info()
print("\nBasic Statistics:")
print(df_bonds.describe(include='all'))
```

### Markdown Cell (Explanation of Execution)

The initial data load and descriptive statistics reveal a dataset of several thousand bond records, spanning multiple years. We can immediately see the column names: `Date`, `Bond_ID`, `Index_Value`, `Volume`, `Coupon_Rate`, `Rating`, `Maturity_Date`, `Risk_Score`, and `Issuer_Region`. The `df.info()` output already hints at potential issues, particularly non-null counts that are less than the total number of entries for some columns like `Volume`, `Rating`, and `Risk_Score`, indicating missing values. This quick scan is crucial for a Quantitative Analyst; it helps in understanding the scale of the data and gives a preliminary sense of where the quality issues might lie before diving into detailed checks.

## 2. Implementing Rule-Based Checks: Ensuring Validity

In a regulated financial environment, many data quality checks are based on predefined business rules and domain knowledge. These "rule-based" checks are fundamental for ensuring data **validity**. For example, a bond's coupon rate must be positive and within a reasonable range, and a risk score cannot be negative. Failing these rules can lead to incorrect model assumptions or, worse, misreporting to regulatory bodies.

### Code Cell (Function Definition)

```python
import pandas as pd

def check_validity_rules(df, rules):
    """
    Applies a set of rule-based validity checks to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        rules (dict): A dictionary where keys are rule names and values are dictionaries
                      specifying the 'column', 'check_type', and 'threshold' or 'allowed_values'.
                      
                      Example rules:
                      {
                          'coupon_rate_range': {'column': 'Coupon_Rate', 'check_type': 'range', 'lower_bound': 0.00, 'upper_bound': 0.10},
                          'risk_score_positive': {'column': 'Risk_Score', 'check_type': 'min_value', 'min_value': 0.0},
                          'rating_categories': {'column': 'Rating', 'check_type': 'allowed_values', 'allowed_values': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']}
                          'volume_non_negative': {'column': 'Volume', 'check_type': 'min_value', 'min_value': 0.0},
                          'maturity_date_future': {'column': 'Maturity_Date', 'check_type': 'future_date', 'reference_date_column': 'Date'}
                      }
    Returns:
        pd.DataFrame: A DataFrame summarizing the results of each check (count of breaches, percentage of breaches).
        dict: A dictionary containing DataFrames of breached records for each rule.
    """
    results = []
    breached_records = {}

    for rule_name, params in rules.items():
        column = params['column']
        check_type = params['check_type']
        
        breach_mask = pd.Series(False, index=df.index)

        if column not in df.columns:
            print(f"Warning: Column '{column}' for rule '{rule_name}' not found in DataFrame. Skipping.")
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
            breach_mask = ~df[column].isin(allowed_values) & df[column].notna() # Exclude NaN from 'invalid category'
        elif check_type == 'future_date':
            # Check if Maturity_Date is before the corresponding Date
            reference_date_column = params['reference_date_column']
            breach_mask = df[column] < df[reference_date_column]
        else:
            print(f"Warning: Unknown check_type '{check_type}' for rule '{rule_name}'. Skipping.")
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
            breached_records[rule_name] = df[breach_mask]

    return pd.DataFrame(results), breached_records
```

### Code Cell (Function Execution)

```python
# Define the specific rule-based checks for our financial data
validation_rules = {
    'coupon_rate_range': {'column': 'Coupon_Rate', 'check_type': 'range', 'lower_bound': 0.01, 'upper_bound': 0.10},
    'risk_score_positive': {'column': 'Risk_Score', 'check_type': 'min_value', 'min_value': 0.0},
    'rating_categories': {'column': 'Rating', 'check_type': 'allowed_values', 'allowed_values': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']},
    'volume_non_negative': {'column': 'Volume', 'check_type': 'min_value', 'min_value': 0.0},
    'maturity_date_future': {'column': 'Maturity_Date', 'check_type': 'future_date', 'reference_date_column': 'Date'}
}

# Execute the rule-based checks
rule_check_results, rule_breaches = check_validity_rules(df_bonds, validation_rules)

print("Rule-Based Validity Check Results:")
print(rule_check_results)

print("\nSample of records breaching 'risk_score_positive' rule:")
if 'risk_score_positive' in rule_breaches:
    print(rule_breaches['risk_score_positive'].head())
```

### Markdown Cell (Explanation of Execution)

The rule-based checks quickly identified several critical validity issues. For instance, the `risk_score_positive` rule detected instances where bond risk scores were negative, which is unacceptable for our models. Similarly, `coupon_rate_range` caught rates outside the expected 1% to 10% band, and `maturity_date_future` flagged bonds whose maturity date had already passed the current `Date` record. These breaches are direct violations of business logic and regulatory expectations. As a Quantitative Analyst, this tells me that these data points, if used in AI models, could lead to flawed risk calculations, incorrect valuation, and potentially non-compliant financial reporting. These issues must be addressed before the data can be considered "valid" for downstream AI applications.

## 3. Implementing Statistical Checks: Completeness and Consistency

Beyond explicit rules, data quality also encompasses **completeness** (are all expected values present?) and **consistency** (are values coherent and free from anomalies?). Statistical methods are excellent for identifying these issues, especially outliers that might signal erroneous data entry or unusual market events that warrant further investigation.

### Code Cell (Function Definition)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_completeness_and_consistency(df, completeness_cols, consistency_cols, outlier_method='iqr', iqr_multiplier=1.5):
    """
    Performs completeness checks (missing values) and consistency checks (outlier detection).

    Args:
        df (pd.DataFrame): The input DataFrame.
        completeness_cols (list): List of columns to check for missing values.
        consistency_cols (list): List of numerical columns to check for outliers.
        outlier_method (str): Method for outlier detection ('iqr' or 'z_score').
        iqr_multiplier (float): Multiplier for IQR method (e.g., 1.5 for mild outliers, 3 for extreme).

    Returns:
        tuple:
            - pd.DataFrame: Summary of completeness checks.
            - pd.DataFrame: Summary of consistency checks (outliers).
            - dict: DataFrames of records identified as outliers for each column.
            - dict: Visualizations (histograms and boxplots) for consistency checks.
    """
    completeness_results = []
    for col in completeness_cols:
        if col not in df.columns:
            completeness_results.append({'Column': col, 'Missing_Count': 0, 'Missing_Percentage': 0.0, 'Status': 'Column Not Found'})
            continue
        missing_count = df[col].isnull().sum()
        total_count = len(df)
        missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0.0
        completeness_results.append({
            'Column': col,
            'Missing_Count': missing_count,
            'Missing_Percentage': round(missing_percentage, 2),
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

        if outlier_method == 'iqr':
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Plotting for IQR
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(numeric_data, kde=True, ax=axes[0])
            axes[0].axvline(lower_bound, color='r', linestyle='--', label='Lower Bound')
            axes[0].axvline(upper_bound, color='r', linestyle='--', label='Upper Bound')
            axes[0].set_title(f'Distribution of {col} with IQR Bounds')
            axes[0].legend()
            
            sns.boxplot(y=numeric_data, ax=axes[1])
            axes[1].set_title(f'Box Plot of {col}')
            plt.tight_layout()
            visualizations[col] = fig

        elif outlier_method == 'z_score':
            mean = numeric_data.mean()
            std_dev = numeric_data.std()
            z_scores = np.abs((df[col] - mean) / std_dev)
            threshold = 3 # Common threshold for z-score
            outlier_mask = (z_scores > threshold)
            
            # Plotting for Z-score
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(numeric_data, kde=True, ax=axes[0])
            axes[0].set_title(f'Distribution of {col}')
            
            sns.scatterplot(x=df.index, y=df[col], hue=outlier_mask, ax=axes[1])
            axes[1].set_title(f'Z-Score Outliers in {col}')
            plt.tight_layout()
            visualizations[col] = fig
        else:
            print(f"Warning: Unknown outlier_method '{outlier_method}' for column '{col}'. Skipping.")
            continue

        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0.0
        
        consistency_results.append({
            'Column': col,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': round(outlier_percentage, 2),
            'Status': 'Outliers Detected' if outlier_count > 0 else 'No Outliers'
        })
        
        if outlier_count > 0:
            outlier_records[col] = df[outlier_mask]

    return pd.DataFrame(completeness_results), pd.DataFrame(consistency_results), outlier_records, visualizations
```

### Code Cell (Function Execution)

```python
# Define columns for completeness and consistency checks
completeness_check_columns = ['Volume', 'Rating', 'Risk_Score', 'Index_Value', 'Coupon_Rate']
consistency_check_columns = ['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score']

# Execute the completeness and consistency checks using IQR for outliers
completeness_results, consistency_results, statistical_outliers, consistency_plots = \
    check_completeness_and_consistency(df_bonds, completeness_check_columns, consistency_check_columns, outlier_method='iqr', iqr_multiplier=3.0) # Using 3.0 for extreme outliers

print("Completeness Check Results:")
print(completeness_results)

print("\nConsistency Check (Outliers) Results:")
print(consistency_results)

print("\nSample of records identified as outliers in 'Volume':")
if 'Volume' in statistical_outliers:
    print(statistical_outliers['Volume'].head())

# Display the generated plots
for col, fig in consistency_plots.items():
    print(f"\nVisualizations for {col}:")
    plt.figure(fig.number) # Reactivate the figure to display it
    plt.show()
```

### Markdown Cell (Explanation of Execution)

The completeness check confirms my initial suspicion from `df.info()`. Columns like `Volume`, `Rating`, and `Risk_Score` have significant percentages of missing values. For a Quantitative Analyst, missing data means incomplete information for models, potentially leading to biased training, reduced prediction accuracy, or even model crashes if not handled properly.

The consistency checks, using the Interquartile Range (IQR) method with a multiplier of $3.0$ for identifying *extreme* outliers, revealed unusual values in `Index_Value`, `Volume`, `Coupon_Rate`, and `Risk_Score`. The box plots clearly highlight these data points falling outside the $Q1 - 3 \cdot IQR$ and $Q3 + 3 \cdot IQR$ boundaries.

The formula for IQR-based outlier detection for a data point $x$ in a distribution is:
$$ x < Q1 - k \cdot IQR \quad \text{or} \quad x > Q3 + k \cdot IQR $$
where $Q1$ is the 25th percentile, $Q3$ is the 75th percentile, $IQR = Q3 - Q1$ is the interquartile range, and $k$ is a chosen multiplier (here, $k=3.0$).

These outliers represent significant deviations from the norm. In finance, such anomalies could be erroneous data entries, system glitches, or legitimate but extreme market events. Distinguishing these is critical for model robustness. Training an AI model on data with unhandled outliers can lead to models that overfit to noise or make highly inaccurate predictions on new, normal data.

## 4. Addressing Timeliness and Duplication: Enhancing Data Reliability

Beyond validity, completeness, and consistency, other crucial data quality dimensions for financial AI are **timeliness** (is the data current and complete for expected periods?) and the absence of **duplicates**. Late or duplicated data can skew analyses, inflate datasets, and lead to incorrect signals for AI models, especially in high-frequency trading or risk monitoring.

### Code Cell (Function Definition)

```python
import pandas as pd
from datetime import timedelta

def check_timeliness_and_duplicates(df, date_col='Date', id_cols=['Bond_ID', 'Date']):
    """
    Checks for date gaps (timeliness) and duplicate records.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        id_cols (list): List of columns that uniquely identify a record (for duplicate check).

    Returns:
        tuple:
            - pd.DataFrame: Summary of timeliness and duplicate checks.
            - dict: DataFrames of breached records (date gaps, duplicates).
    """
    results = []
    breached_records = {}

    # 1. Timeliness Check (Date Gaps)
    # Group by Bond_ID and check for gaps in the Date sequence
    df_sorted = df.sort_values(by=[id_cols[0], date_col]) # Sort by Bond_ID and Date
    
    date_gaps_count = 0
    date_gaps_df = pd.DataFrame()
    
    for bond_id in df_sorted[id_cols[0]].unique():
        bond_df = df_sorted[df_sorted[id_cols[0]] == bond_id].copy()
        if len(bond_df) > 1:
            expected_dates = pd.date_range(start=bond_df[date_col].min(), end=bond_df[date_col].max(), freq='D')
            actual_dates = pd.Index(bond_df[date_col].unique())
            missing_dates = expected_dates.difference(actual_dates)
            
            if len(missing_dates) > 0:
                date_gaps_count += len(missing_dates)
                # For reporting, we can list the missing dates for this bond
                for missing_date in missing_dates:
                    date_gaps_df = pd.concat([date_gaps_df, pd.DataFrame([{'Bond_ID': bond_id, 'Missing_Date': missing_date}])])

    total_expected_days_across_bonds = 0
    for bond_id in df_sorted[id_cols[0]].unique():
        bond_df = df_sorted[df_sorted[id_cols[0]] == bond_id].copy()
        if len(bond_df) > 1:
            total_expected_days_across_bonds += (bond_df[date_col].max() - bond_df[date_col].min()).days + 1

    date_gap_percentage = (date_gaps_count / total_expected_days_across_bonds) * 100 if total_expected_days_across_bonds > 0 else 0.0

    results.append({
        'Rule': 'date_timeliness_gaps',
        'Column': date_col,
        'Breaches': date_gaps_count,
        'Breach_Percentage': round(date_gap_percentage, 2),
        'Status': 'Gaps Detected' if date_gaps_count > 0 else 'No Gaps'
    })
    if date_gaps_count > 0:
        breached_records['date_timeliness_gaps'] = date_gaps_df

    # 2. Duplicate Records Check
    duplicate_mask = df.duplicated(subset=id_cols, keep=False) # Mark all duplicates, not just subsequent ones
    duplicate_count = duplicate_mask.sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0.0

    results.append({
        'Rule': 'duplicate_records',
        'Column': ', '.join(id_cols),
        'Breaches': duplicate_count,
        'Breach_Percentage': round(duplicate_percentage, 2),
        'Status': 'Duplicates Detected' if duplicate_count > 0 else 'No Duplicates'
    })
    if duplicate_count > 0:
        breached_records['duplicate_records'] = df[duplicate_mask].sort_values(by=id_cols)
        
    return pd.DataFrame(results), breached_records
```

### Code Cell (Function Execution)

```python
# Define the unique identifier columns for duplicate checks (Bond_ID and Date should be unique per record)
id_columns_for_duplicates = ['Bond_ID', 'Date']

# Execute the timeliness and duplicate checks
timeliness_duplicate_results, td_breaches = check_timeliness_and_duplicates(df_bonds, date_col='Date', id_cols=id_columns_for_duplicates)

print("Timeliness and Duplicate Check Results:")
print(timeliness_duplicate_results)

print("\nSample of records identified as duplicates:")
if 'duplicate_records' in td_breaches:
    print(td_breaches['duplicate_records'].head())

print("\nMissing dates for 'Bond_A':")
if 'date_timeliness_gaps' in td_breaches:
    print(td_breaches['date_timeliness_gaps'].head())
```

### Markdown Cell (Explanation of Execution)

The timeliness check successfully identified gaps in our daily time series for specific bonds, as evidenced by missing dates. This is critical for time-series-dependent AI models that expect regular, continuous data. A Quantitative Analyst knows that missing observations can lead to incorrect feature engineering, biased trend analysis, or even failure of models designed for consistent periodic data.

The duplicate check revealed multiple instances where entire records were identical based on `Bond_ID` and `Date`. Duplicates artificially inflate data counts, bias statistical measures (e.g., averages, variances), and can lead to models learning redundant patterns, reducing their generalization capability and efficiency. This could also misrepresent the true activity for a given bond on a given date. Addressing these issues ensures that our AI models are trained on clean, non-redundant, and properly sequenced historical data, fostering greater trust in their outputs.

## 5. Developing a Data Quality Scoring Framework

With individual data quality dimensions defined and checked, my next task is to aggregate these findings into a unified, interpretable Data Quality Score. This score provides a high-level overview, allowing me to quickly assess the overall health of the dataset and communicate it effectively to stakeholders and senior management. The framework will use a weighted average approach, where each dimension (completeness, validity, consistency, timeliness, and uniqueness) contributes based on its defined importance.

The overall data quality score $S_{overall}$ for a dataset is calculated as a weighted average of individual dimension scores $s_i$:
$$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$
Where:
- $N$ is the number of data quality dimensions.
- $s_i$ is the normalized score for dimension $i$ (ranging from 0 to 1). A score of 1 indicates perfect quality for that dimension, and 0 indicates the worst quality.
- $w_i$ is the weight assigned to dimension $i$, reflecting its business criticality.

For each dimension, the score is typically derived from the breach percentage: $s_i = 1 - \frac{\text{Breach Percentage}_i}{100}$.

### Code Cell (Function Definition)

```python
import pandas as pd

class DataQualityScorer:
    """
    Calculates a weighted average data quality score based on various dimension results.
    """
    def __init__(self, weights=None):
        """
        Initializes the DataQualityScorer with optional weights for dimensions.
        Default weights are set to reflect typical importance in finance.
        """
        self.weights = weights if weights is not None else {
            'completeness': 0.25,
            'validity': 0.30,
            'consistency': 0.20,
            'timeliness': 0.15,
            'uniqueness': 0.10
        }
        # Ensure weights sum to 1 for easier interpretation, or normalize later
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            raise ValueError("Total weights cannot be zero.")
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _calculate_dimension_score(self, breach_percentage):
        """
        Calculates a dimension score from breach percentage.
        Score = 1 - (breach_percentage / 100)
        """
        return max(0.0, 1.0 - (breach_percentage / 100.0))

    def generate_scorecard(self, completeness_df, validity_df, timeliness_duplicate_df):
        """
        Generates a comprehensive data quality scorecard with an overall weighted score.

        Args:
            completeness_df (pd.DataFrame): Results from completeness checks.
            validity_df (pd.DataFrame): Results from validity (rule-based) checks.
            timeliness_duplicate_df (pd.DataFrame): Results from timeliness and duplicate checks.

        Returns:
            pd.DataFrame: A detailed scorecard including individual dimension scores and an overall score.
            dict: Individual dimension scores.
        """
        scorecard_data = []
        dimension_scores = {}

        # Completeness Score (average across all checked completeness columns)
        avg_missing_percentage = completeness_df['Missing_Percentage'].mean() if not completeness_df.empty else 0
        dimension_scores['completeness'] = self._calculate_dimension_score(avg_missing_percentage)
        scorecard_data.append({'Dimension': 'Completeness', 'Average_Breach_Percentage': round(avg_missing_percentage, 2), 'Score': round(dimension_scores['completeness'], 2)})

        # Validity Score (average across all rule-based validity checks)
        avg_validity_breach_percentage = validity_df['Breach_Percentage'].mean() if not validity_df.empty else 0
        dimension_scores['validity'] = self._calculate_dimension_score(avg_validity_breach_percentage)
        scorecard_data.append({'Dimension': 'Validity', 'Average_Breach_Percentage': round(avg_validity_breach_percentage, 2), 'Score': round(dimension_scores['validity'], 2)})
        
        # Consistency Score (average across all outlier checks)
        # Note: consistency_df is generated from `check_completeness_and_consistency` (section 3)
        # We need to pass it explicitly from the notebook or calculate here from the global results
        # For simplicity, let's assume `consistency_results` DataFrame from previous step is accessible
        # If not, it needs to be an input to this method. For now, let's include it in the arguments.
        # This will be passed as consistency_df in the function call.
        avg_consistency_breach_percentage = 0 # Placeholder, will be updated in execution block
        dimension_scores['consistency'] = self._calculate_dimension_score(avg_consistency_breach_percentage) # Placeholder
        
        # Timeliness Score (from date_timeliness_gaps rule)
        timeliness_breach = timeliness_duplicate_df[timeliness_duplicate_df['Rule'] == 'date_timeliness_gaps']
        avg_timeliness_breach_percentage = timeliness_breach['Breach_Percentage'].iloc[0] if not timeliness_breach.empty else 0
        dimension_scores['timeliness'] = self._calculate_dimension_score(avg_timeliness_breach_percentage)
        scorecard_data.append({'Dimension': 'Timeliness', 'Average_Breach_Percentage': round(avg_timeliness_breach_percentage, 2), 'Score': round(dimension_scores['timeliness'], 2)})

        # Uniqueness Score (from duplicate_records rule)
        uniqueness_breach = timeliness_duplicate_df[timeliness_duplicate_df['Rule'] == 'duplicate_records']
        avg_uniqueness_breach_percentage = uniqueness_breach['Breach_Percentage'].iloc[0] if not uniqueness_breach.empty else 0
        dimension_scores['uniqueness'] = self._calculate_dimension_score(avg_uniqueness_breach_percentage)
        scorecard_data.append({'Dimension': 'Uniqueness', 'Average_Breach_Percentage': round(avg_uniqueness_breach_percentage, 2), 'Score': round(dimension_scores['uniqueness'], 2)})
        
        # Calculate consistency score properly
        avg_consistency_breach_percentage = consistency_results['Outlier_Percentage'].mean() if not consistency_results.empty else 0
        dimension_scores['consistency'] = self._calculate_dimension_score(avg_consistency_breach_percentage)
        scorecard_data.insert(2, {'Dimension': 'Consistency', 'Average_Breach_Percentage': round(avg_consistency_breach_percentage, 2), 'Score': round(dimension_scores['consistency'], 2)})

        # Calculate Overall Weighted Score
        overall_score = sum(dimension_scores[dim] * self.weights.get(dim, 0) for dim in dimension_scores)
        
        scorecard_df = pd.DataFrame(scorecard_data)
        scorecard_df.loc['Overall'] = ['Overall Score', np.nan, round(overall_score, 2)]

        return scorecard_df, dimension_scores
```

### Code Cell (Function Execution)

```python
# Instantiate the DataQualityScorer with custom weights if desired
# Example weights emphasizing validity and completeness more in a regulated financial setting
custom_weights = {
    'completeness': 0.25, # Critical for model inputs
    'validity': 0.35,     # Business rules and regulatory compliance
    'consistency': 0.20,  # Outliers can skew models
    'timeliness': 0.10,   # Important for time-sensitive models
    'uniqueness': 0.10    # Prevents data inflation and bias
}
dq_scorer = DataQualityScorer(weights=custom_weights)

# Generate the data quality scorecard
# Ensure that `consistency_results` from Section 3 is passed correctly
data_quality_scorecard, dimension_scores_dict = \
    dq_scorer.generate_scorecard(completeness_results, rule_check_results, timeliness_duplicate_results)

print("--- Data Quality Scorecard ---")
print(data_quality_scorecard.to_string())

# Visualize dimension scores
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=list(dimension_scores_dict.keys()), y=list(dimension_scores_dict.values()), palette='viridis')
plt.title('Data Quality Dimension Scores (0-1 Scale)')
plt.ylabel('Score')
plt.xlabel('Data Quality Dimension')
plt.ylim(0, 1)
plt.axhline(y=data_quality_scorecard.loc['Overall', 'Score'], color='red', linestyle='--', label=f"Overall Score: {data_quality_scorecard.loc['Overall', 'Score']:.2f}")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

### Markdown Cell (Explanation of Execution)

The Data Quality Scorecard provides a consolidated view of our dataset's health. The individual dimension scores (Completeness, Validity, Consistency, Timeliness, Uniqueness) are now visible, along with an overall weighted score. For a Quantitative Analyst, this scorecard is invaluable. Instead of sifting through multiple reports, I get an immediate, quantified assessment. For example, a low "Validity" score tells me that critical business rules are being violated, requiring immediate attention. A moderate "Completeness" score suggests imputation strategies might be needed.

The bar chart visually reinforces these scores, making it easy to identify the weakest links in our data quality chain. The overall score, at $X.XX$ (e.g., $0.78$), gives a single metric for management to track data quality trends over time. This metric forms the basis for our internal SLAs (Service Level Agreements) for data providers and helps prioritize remediation efforts. It’s a powerful tool for driving data governance and ensuring that our AI initiatives are built on a solid foundation of reliable data.

## 6. Investigating Rule-Based Breaches and Their Impact

A low Validity score from our scorecard signals significant issues with adherence to business rules. As a Quantitative Analyst, I need to pinpoint exactly *which* rules are most frequently breached and what specific data points are causing these violations. This granular insight allows me to understand the potential impact on our AI models and initiate targeted data cleaning or process improvements.

### Code Cell (Function Definition)

```python
import pandas as pd

def summarize_rule_breaches(rule_check_results, breached_records):
    """
    Summarizes rule-based breaches, identifying the most critical ones.

    Args:
        rule_check_results (pd.DataFrame): DataFrame summarizing all rule checks.
        breached_records (dict): Dictionary of DataFrames containing records that breached each rule.

    Returns:
        pd.DataFrame: A summary of rules with breaches, sorted by breach percentage.
    """
    breached_summary = rule_check_results[rule_check_results['Breaches'] > 0].sort_values(by='Breach_Percentage', ascending=False)
    return breached_summary

def get_breach_sample(breached_records, rule_name, n=5):
    """
    Retrieves a sample of records for a specific breached rule.

    Args:
        breached_records (dict): Dictionary of DataFrames containing records that breached each rule.
        rule_name (str): The name of the rule to retrieve samples for.
        n (int): Number of sample records to return.

    Returns:
        pd.DataFrame: Sample of breached records.
    """
    if rule_name in breached_records:
        return breached_records[rule_name].head(n)
    return pd.DataFrame()
```

### Code Cell (Function Execution)

```python
# Summarize the rule-based breaches
breached_rules_summary = summarize_rule_breaches(rule_check_results, rule_breaches)

print("--- Summary of Breached Rules ---")
print(breached_rules_summary.to_string())

# Select the rule with the highest breach percentage for detailed inspection
most_breached_rule_name = breached_rules_summary['Rule'].iloc[0] if not breached_rules_summary.empty else None

if most_breached_rule_name:
    print(f"\n--- Sample records for the most breached rule: '{most_breached_rule_name}' ---")
    sample_breaches = get_breach_sample(rule_breaches, most_breached_rule_name, n=10)
    print(sample_breaches.to_string())
else:
    print("\nNo rule-based breaches detected.")
```

### Markdown Cell (Explanation of Execution)

The summary of breached rules clearly highlights which validity constraints are most frequently violated in our `corporate_bond_data`. For example, if `risk_score_positive` shows a high breach percentage, it means a significant number of bond records have invalid negative risk scores.

For a Quantitative Analyst, this is critical:
*   **Model Accuracy:** Negative risk scores would drastically distort risk assessments, potentially leading to underestimation of portfolio risk or incorrect investment decisions. An AI model trained on such data would learn to associate invalid risk profiles, producing unreliable predictions.
*   **Regulatory Compliance:** Financial regulations often stipulate strict requirements for data validity (e.g., all values must be non-negative). Breaches directly impact our ability to comply with these mandates.

The sample records for the most breached rule provide concrete examples, enabling me to communicate specific data issues to data stewards for root cause analysis and remediation. This deep dive directly informs data cleansing strategies or necessitates discussions with data providers about upstream data generation processes.

## 7. Investigating Statistical Anomalies and Their Impact

Our statistical checks for completeness and consistency (outliers) surfaced additional data quality concerns. A significant number of outliers or missing values can subtly yet profoundly impact AI model performance. As a Quantitative Analyst, I need to understand the distribution of these anomalies and their potential effect on model training and inference.

### Code Cell (Function Definition)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_statistical_issues(completeness_df, consistency_df):
    """
    Summarizes completeness and consistency issues.

    Args:
        completeness_df (pd.DataFrame): Results from completeness checks.
        consistency_df (pd.DataFrame): Results from consistency (outlier) checks.

    Returns:
        pd.DataFrame: Combined summary of missing data and outliers, sorted by percentage.
    """
    completeness_summary = completeness_df[completeness_df['Missing_Count'] > 0].rename(columns={'Missing_Count': 'Count', 'Missing_Percentage': 'Percentage', 'Status': 'Issue_Type'})
    completeness_summary['Issue'] = 'Missing Values'
    completeness_summary = completeness_summary[['Column', 'Issue', 'Count', 'Percentage']]

    consistency_summary = consistency_df[consistency_df['Outlier_Count'] > 0].rename(columns={'Outlier_Count': 'Count', 'Outlier_Percentage': 'Percentage', 'Status': 'Issue_Type'})
    consistency_summary['Issue'] = 'Outliers'
    consistency_summary = consistency_summary[['Column', 'Issue', 'Count', 'Percentage']]

    combined_summary = pd.concat([completeness_summary, consistency_summary]).sort_values(by='Percentage', ascending=False)
    return combined_summary

def plot_outlier_distribution(df, outlier_records, column_name):
    """
    Visualizes the distribution of a column, highlighting outliers.
    """
    if column_name in outlier_records:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column_name].dropna(), kde=True, color='skyblue', label='All Data')
        sns.histplot(outlier_records[column_name][column_name].dropna(), kde=True, color='red', label='Outliers', alpha=0.6)
        plt.title(f'Distribution of {column_name} with Outliers Highlighted')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print(f"No outliers detected for column '{column_name}'.")
```

### Code Cell (Function Execution)

```python
# Summarize all statistical issues (missing values and outliers)
statistical_issues_summary = summarize_statistical_issues(completeness_results, consistency_results)

print("--- Summary of Statistical Data Quality Issues ---")
print(statistical_issues_summary.to_string())

# Focus on a column with significant outliers, e.g., 'Volume'
plot_outlier_distribution(df_bonds, statistical_outliers, 'Volume')

# For missing values, visualize the missingness pattern across columns
import missingno as msno
print("\n--- Visualizing Missing Data Patterns ---")
msno.matrix(df_bonds, figsize=(12, 7), sparkline=False)
plt.title('Missingness Matrix of Bond Data')
plt.show()
```

### Markdown Cell (Explanation of Execution)

The summary of statistical issues and the visualizations provide a deeper understanding of our data's completeness and consistency. The `missingno.matrix` plot clearly shows patterns of missingness across columns, which can indicate systematic issues. For example, if `Volume` and `Risk_Score` often go missing together, it might suggest a specific data source or collection process failure.

The distribution plot for `Volume` with outliers highlighted vividly demonstrates where the unusual values lie. For a Quantitative Analyst, this directly informs data preprocessing:
*   **Missing Values:** High percentages of missing values in `Volume` or `Rating` mean we cannot simply drop these records without losing valuable information. Imputation strategies (mean, median, mode, or more advanced model-based imputation) become necessary. The choice of imputation method heavily influences downstream AI model performance and should be carefully considered to avoid introducing bias.
*   **Outliers:** The identified outliers in `Index_Value` or `Volume` could be critical errors or genuine extreme events. Distinguishing these is crucial. If they are errors, they need correction or removal. If they are legitimate extreme events, they might contain valuable information for fraud detection or market shock models, but require careful handling (e.g., Winsorization, robust scaling) to prevent them from dominating the model's learning process.

By visualizing and quantifying these statistical anomalies, I can make informed decisions about how to clean and prepare the data for AI, ensuring that our models are robust and less prone to statistical bias.

## 8. Advanced AI-Based Anomaly Detection for Subtle Shifts

While rule-based and statistical checks catch many issues, some data quality problems are more subtle. These often manifest as complex deviations or shifts in patterns that a simple threshold or statistical bound might miss. This is where AI-based anomaly detection algorithms, like Isolation Forest, become invaluable. As a Quantitative Analyst, I use these to identify data points that are "different" from the norm, even if they don't explicitly violate a rule or fall outside a simple range. This is especially important for continuous monitoring of live data feeds where new, unforeseen patterns of data degradation can emerge.

### Code Cell (Function Definition)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

def apply_ai_anomaly_detection(df, features_for_anomaly_detection, contamination_rate=0.01, random_state=42):
    """
    Applies Isolation Forest for unsupervised anomaly detection on specified features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features_for_anomaly_detection (list): List of numerical columns to use for anomaly detection.
        contamination_rate (float): The proportion of outliers in the dataset.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple:
            - pd.DataFrame: The original DataFrame with 'anomaly_score' and 'is_anomaly' columns added.
            - pyod.models.iforest.IsolationForest: The trained Isolation Forest model.
            - dict: Summary statistics of anomalies.
    """
    # Prepare data: handle missing values and scale numerical features
    df_processed = df[features_for_anomaly_detection].copy()
    
    # Impute missing values with median for anomaly detection, as Isolation Forest doesn't handle NaNs directly
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed)
    
    # Train Isolation Forest model
    model = IsolationForest(contamination=contamination_rate, random_state=random_state, n_estimators=100, behaviour='new') # n_estimators for robustness, behaviour='new' for recent PyOD versions
    model.fit(scaled_features)
    
    # Predict anomaly scores and labels
    df_with_anomalies = df.copy()
    df_with_anomalies['anomaly_score'] = model.decision_function(scaled_features) # Lower score indicates more anomalous
    df_with_anomalies['is_anomaly'] = model.predict(scaled_features) # 1 for inliers, -1 for outliers
    
    # Convert PyOD's -1/1 labels to True/False for clarity
    df_with_anomalies['is_anomaly'] = df_with_anomalies['is_anomaly'] == -1
    
    anomaly_count = df_with_anomalies['is_anomaly'].sum()
    total_records = len(df_with_anomalies)
    anomaly_percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0.0

    anomaly_summary = {
        'Anomaly_Count': anomaly_count,
        'Anomaly_Percentage': round(anomaly_percentage, 2),
        'Total_Records': total_records
    }
    
    return df_with_anomalies, model, anomaly_summary

def plot_anomaly_scores(df_with_anomalies, anomaly_score_col='anomaly_score', is_anomaly_col='is_anomaly'):
    """
    Plots anomaly scores, highlighting detected anomalies.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df_with_anomalies[anomaly_score_col], kde=True, color='blue', label='Normal Scores')
    sns.histplot(df_with_anomalies[df_with_anomalies[is_anomaly_col]][anomaly_score_col], kde=True, color='red', label='Anomalous Scores', alpha=0.6)
    plt.title('Distribution of Anomaly Scores from Isolation Forest')
    plt.xlabel('Anomaly Score (Lower is More Anomalous)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(12, 6))
    # Plotting anomaly scores over time for a specific bond, or overall index
    sns.lineplot(x='Date', y='anomaly_score', hue='is_anomaly', data=df_with_anomalies.sort_values('Date').sample(n=min(len(df_with_anomalies), 1000), random_state=42), palette={True: 'red', False: 'blue'}, style='is_anomaly', markers=True, dashes=False, size='is_anomaly', sizes=(50, 10))
    plt.title('Anomaly Scores Over Time (Sampled Data)')
    plt.xlabel('Date')
    plt.ylabel('Anomaly Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

```

### Code Cell (Function Execution)

```python
# Define numerical features for AI anomaly detection. Categorical features would need encoding.
ai_features = ['Index_Value', 'Volume', 'Coupon_Rate', 'Risk_Score']

# Execute AI-based anomaly detection
df_bonds_anomalies, if_model, anomaly_summary_dict = \
    apply_ai_anomaly_detection(df_bonds, ai_features, contamination_rate=0.015) # Assuming 1.5% of data are anomalies

print("--- AI-Based Anomaly Detection Summary ---")
print(f"Total Records: {anomaly_summary_dict['Total_Records']}")
print(f"Anomaly Count: {anomaly_summary_dict['Anomaly_Count']}")
print(f"Anomaly Percentage: {anomaly_summary_dict['Anomaly_Percentage']}%")

print("\nSample of AI-detected anomalous records:")
print(df_bonds_anomalies[df_bonds_anomalies['is_anomaly']].head().to_string())

# Plot anomaly scores
plot_anomaly_scores(df_bonds_anomalies)
```

### Markdown Cell (Explanation of Execution)

The AI-based anomaly detection using Isolation Forest has identified a set of data points as anomalous, which might have slipped past our simpler rule-based and statistical checks. The anomaly score distribution plot shows a clear separation, with lower scores indicating higher anomaly likelihood. The time-series plot of anomaly scores helps visualize when these complex deviations occur.

For a Quantitative Analyst, these AI-detected anomalies are particularly interesting:
*   **Subtle Shifts:** These aren't necessarily "errors" in the traditional sense but rather data points that deviate significantly from learned normal patterns across multiple features. They could represent new market regimes, unusual trading activities, or even early warning signs of data corruption that doesn't trigger simple thresholds.
*   **Actionable Insights:** Further investigation into these records (e.g., sample of records with `is_anomaly=True`) can uncover hidden data quality issues or reveal genuine, but rare, market events that our AI models need to either be robust against or specifically trained to detect. This enhances our understanding of data behavior under stress.
*   **Robust AI:** By identifying and understanding these complex anomalies, we can refine our data preprocessing, potentially re-engineer features, or design more robust AI models that are less susceptible to learning from these deviations, leading to more reliable predictions in dynamic financial markets. This proactive detection prevents AI models from making critical decisions based on potentially misleading or uncharacteristic data.

## 9. Developing a Continuous Monitoring System: Alerts for Critical Deviations

Our data quality scoring system is now capable of identifying issues through rule-based, statistical, and AI-based methods. The next crucial step for a DataOps team is to integrate these insights into a continuous monitoring system that generates alerts for critical deviations. This enables proactive intervention before poor data quality impacts downstream AI applications or regulatory reporting. As a Quantitative Analyst, I define the thresholds that trigger these alerts, balancing sensitivity with the need to avoid alert fatigue.

### Code Cell (Function Definition)

```python
import pandas as pd

def check_for_critical_alerts(dimension_scores, data_quality_scorecard, anomaly_summary,
                              overall_score_threshold=0.75,
                              completeness_score_threshold=0.80,
                              validity_score_threshold=0.80,
                              ai_anomaly_percentage_threshold=2.0):
    """
    Checks if any data quality metrics fall below critical thresholds or exceed anomaly limits.

    Args:
        dimension_scores (dict): Dictionary of individual dimension scores.
        data_quality_scorecard (pd.DataFrame): The full data quality scorecard.
        anomaly_summary (dict): Summary from AI-based anomaly detection.
        overall_score_threshold (float): Minimum acceptable overall data quality score.
        completeness_score_threshold (float): Minimum acceptable completeness score.
        validity_score_threshold (float): Minimum acceptable validity score.
        ai_anomaly_percentage_threshold (float): Maximum acceptable AI-detected anomaly percentage.

    Returns:
        list: A list of critical alerts generated.
    """
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

### Code Cell (Function Execution)

```python
# Define critical thresholds for alerting based on regulatory and business impact
critical_overall_dq_threshold = 0.70
critical_completeness_dq_threshold = 0.75
critical_validity_dq_threshold = 0.85
critical_ai_anomaly_percent = 3.0 # Max 3% of data can be AI-detected anomalies without an alert

# Execute the alert check
current_alerts = check_for_critical_alerts(
    dimension_scores_dict,
    data_quality_scorecard,
    anomaly_summary_dict,
    overall_score_threshold=critical_overall_dq_threshold,
    completeness_score_threshold=critical_completeness_dq_threshold,
    validity_score_threshold=critical_validity_dq_threshold,
    ai_anomaly_percentage_threshold=critical_ai_anomaly_percent
)

print("--- Current Data Quality Alerts ---")
if current_alerts:
    for alert in current_alerts:
        print(f"- {alert}")
else:
    print("No critical data quality alerts detected. Data quality is within acceptable limits.")
```

### Markdown Cell (Explanation of Execution)

The alert system provides an immediate flag when data quality deviates from predefined acceptable levels. For a Quantitative Analyst, this is the operational heartbeat of a DataOps framework. Critical alerts, like a low overall score or a breach in validity, demand immediate attention as they pose direct threats to AI model reliability and regulatory compliance. Less critical alerts, such as a slight dip in completeness or a higher-than-usual percentage of AI-detected anomalies, signal trends that need monitoring and potential proactive investigation before they escalate.

By setting explicit thresholds, such as $S_{overall} < 0.70$ or $s_{validity} < 0.85$, we ensure that the team focuses on the most impactful issues. This mechanism helps prevent alert fatigue while ensuring that no significant data quality degradation goes unnoticed, thereby safeguarding our AI pipelines and maintaining organizational trust. It transforms raw data quality metrics into actionable insights for the DataOps team.

## 10. Generating the Data Quality Impact Report

The final step in our data quality workflow is to consolidate all findings into a comprehensive Data Quality Impact Report. This report is crucial for communicating the current state of data quality, highlighting specific issues, explaining their potential impact on AI models and business operations, and recommending remediation strategies. As a Quantitative Analyst, I'm responsible for translating complex data quality metrics into clear, actionable business insights for management and data owners.

### Code Cell (Function Definition)

```python
import pandas as pd
from datetime import datetime

def generate_data_quality_impact_report(data_quality_scorecard, rule_check_results, statistical_issues_summary, anomaly_summary, current_alerts):
    """
    Generates a comprehensive Data Quality Impact Report.

    Args:
        data_quality_scorecard (pd.DataFrame): The full data quality scorecard.
        rule_check_results (pd.DataFrame): Results from rule-based validity checks.
        statistical_issues_summary (pd.DataFrame): Combined summary of missing data and outliers.
        anomaly_summary (dict): Summary from AI-based anomaly detection.
        current_alerts (list): List of triggered critical alerts.

    Returns:
        pd.DataFrame: A structured report DataFrame.
        str: A formatted string summary of the report.
    """
    report_sections = []
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
    
    # Rule-Based Validity Issues
    breached_rules_summary = rule_check_results[rule_check_results['Breaches'] > 0].sort_values(by='Breach_Percentage', ascending=False)
    if not breached_rules_summary.empty:
        report_summary_str.append("\n### Rule-Based Validity Issues (Top 5 by percentage):")
        report_summary_str.append(breached_rules_summary.head(5).to_string())
        report_summary_str.append("\n*Impact:* Direct violations of business logic and regulatory requirements. Can lead to flawed financial calculations, incorrect risk assessments, and non-compliant reporting by AI models.")
    
    # Statistical Issues (Completeness & Consistency)
    if not statistical_issues_summary.empty:
        report_summary_str.append("\n### Statistical Data Issues (Top 5 by percentage):")
        report_summary_str.append(statistical_issues_summary.head(5).to_string())
        report_summary_str.append("\n*Impact:* Missing values can introduce bias and reduce AI model accuracy. Outliers can lead to models overfitting to noise, producing unstable predictions or false positives in fraud detection.")
    
    # AI-Based Anomalies
    if anomaly_summary['Anomaly_Count'] > 0:
        report_summary_str.append("\n### AI-Detected Complex Anomalies:")
        report_summary_str.append(f"Isolation Forest identified {anomaly_summary['Anomaly_Count']} records ({anomaly_summary['Anomaly_Percentage']:.2f}%) as anomalous.")
        report_summary_str.append("\n*Impact:* These subtle deviations, though not always explicit errors, can signal evolving data patterns, potential data drift, or emerging risks. AI models trained without addressing these might become less robust over time.")
    
    report_summary_str.append("\n## Recommendations")
    report_summary_str.append("- Prioritize remediation for rules with high breach percentages (e.g., negative Risk_Scores, out-of-range Coupon_Rates).")
    report_summary_str.append("- Implement data imputation strategies for critical columns with high missingness (e.g., Volume, Rating).")
    report_summary_str.append("- Investigate the root cause of AI-detected anomalies to distinguish between data errors and genuine market events.")
    report_summary_str.append("- Establish data validation checkpoints upstream in the data pipeline to prevent issues from propagating.")

    # A more structured DataFrame for programmatic use might be more complex
    # For this deliverable, a detailed string output is often more directly usable for reports
    # but we can return a simple df for placeholder
    report_df = pd.DataFrame([{"Report_Summary": "\n".join(report_summary_str)}])
    
    return report_df, "\n".join(report_summary_str)
```

### Code Cell (Function Execution)

```python
# Generate the final Data Quality Impact Report
dq_report_df, dq_report_string = generate_data_quality_impact_report(
    data_quality_scorecard,
    rule_check_results,
    statistical_issues_summary,
    anomaly_summary_dict,
    current_alerts
)

print(dq_report_string)

# The DataFrame version could be saved to a file if needed, e.g., for further automated processing
# dq_report_df.to_csv('data_quality_impact_report.csv', index=False)
```

### Markdown Cell (Explanation of Execution)

This Data Quality Impact Report is the culmination of our efforts. It presents a clear, structured narrative on the current state of our `corporate_bond_data`, crucial for AI model training. As a Quantitative Analyst, I use this report to:

*   **Inform Stakeholders:** Provide a digestible summary for Risk Managers, Data Governance teams, and senior leadership, highlighting the overall data health and any critical alerts.
*   **Prioritize Action:** The detailed sections on rule-based, statistical, and AI-detected anomalies, along with their specific impacts, guide data engineers and data stewards in prioritizing remediation efforts. For instance, knowing that `Risk_Score` has a high percentage of negative values (validity breach) and `Volume` has many missing entries (completeness issue) dictates immediate data cleaning and potentially an update to data ingestion processes.
*   **Drive Continuous Improvement:** The recommendations section outlines concrete steps for improving data quality upstream. This fosters a culture of proactive data governance and ensures that our AI models, which underpin critical financial decisions, are always trained on trustworthy and compliant data. This report directly supports maintaining organizational trust in AI analytics in a highly regulated environment.

## Conclusion: Trustworthy AI in Regulated Finance

Through this notebook, we've established a comprehensive data quality scoring system crucial for any DataOps team operating in a regulated financial environment. As a Quantitative Analyst, I've demonstrated how to transition from raw data to actionable insights by:

1.  **Systematically identifying critical data quality dimensions** relevant to financial AI.
2.  **Quantifying these dimensions** using a combination of rule-based validation for **validity**, statistical methods for **completeness** and **consistency**, and checks for **timeliness** and **uniqueness**.
3.  **Integrating an AI-based anomaly detection algorithm (Isolation Forest)** to uncover subtle, multi-dimensional data quality shifts that traditional methods might miss.
4.  **Developing a weighted scoring framework** to provide a holistic and easily interpretable Data Quality Score.
5.  **Implementing an alerting mechanism** to proactively flag critical deviations.
6.  **Generating a detailed Data Quality Impact Report** to inform stakeholders and drive targeted remediation.

By embedding these rigorous data quality controls into our AI pipelines, we ensure that our predictive models are built on a foundation of clean, reliable, and compliant data. This proactive approach not only mitigates significant financial and reputational risks but also fosters a higher level of trust in the AI-driven decisions made across the organization. This framework is essential for achieving reliable, auditable, and scalable AI operations in the complex landscape of regulated finance.
