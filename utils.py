# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest as IsolationForest
import missingno as msno


# -----------------------------
# Data Generation and Loading
# -----------------------------
def generate_and_load_corporate_bond_data(start_date='2010-01-01', end_date='2019-12-31', num_bonds=10, filename='corporate_bond_data.csv', random_state=42):
    '''
    Generate a simulated corporate bond panel dataset suitable for time-series model training/testing.
    Includes intentional duplicates and date gaps to reflect real-world data quality issues.
    '''
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    ratings_scale = ['AAA','AA','A','BBB','BB','B','CCC','CC','C','D']

    all_data = []
    for i in range(num_bonds):
        bond_id = f'Bond_{chr(65+i)}'
        n = len(dates)
        # Simulate index value as a noisy random walk around 100
        steps = rng.normal(loc=0.02, scale=0.5, size=n)
        index_value = 100 + np.cumsum(steps)
        # Volume: log-normal like positive values with occasional spikes
        volume = np.abs(rng.lognormal(mean=10.0, sigma=0.5, size=n))
        # Coupon rate: stable around 3%-7% with small noise
        coupon_rate = np.clip(rng.normal(loc=0.05, scale=0.01, size=n), 0.0, 0.15)
        # Risk score: 0 to 1 with mild drift
        risk_score = np.clip(rng.beta(a=2, b=5, size=n) + rng.normal(0, 0.02, size=n), 0.0, 1.0)

        # Map risk score to rating buckets
        rating_idx = np.clip((9 - np.floor(risk_score*10)).astype(int), 0, 9)
        rating = [ratings_scale[idx] for idx in rating_idx]

        # Maturity date 2-7 years after current date
        maturity_offsets = rng.integers(low=720, high=2555, size=n)  # days
        maturity_date = pd.to_datetime(dates) + pd.to_timedelta(maturity_offsets, unit='D')

        df_bond = pd.DataFrame({
            'Bond_ID': bond_id,
            'Date': dates,
            'Index_Value': index_value,
            'Volume': volume,
            'Coupon_Rate': coupon_rate,
            'Risk_Score': risk_score,
            'Rating': rating,
            'Maturity_Date': maturity_date
        })

        # Inject some missingness
        for col in ['Volume','Rating','Risk_Score','Index_Value','Coupon_Rate']:
            mask = rng.random(n) < 0.01  # ~1% missing
            df_bond.loc[mask, col] = np.nan

        all_data.append(df_bond)

    df = pd.concat(all_data).reset_index(drop=True)

    # Add intentional duplicates (5 random rows duplicated)
    duplicate_rows = df.sample(n=5, random_state=random_state)
    df = pd.concat([df, duplicate_rows]).sort_values(by='Date').reset_index(drop=True)

    # Add intentional timeliness gaps for Bond_A
    gap_dates = pd.to_datetime(['2015-03-10', '2015-03-11', '2015-03-12'])
    df = df[~((df['Bond_ID'] == 'Bond_A') & (df['Date'].isin(gap_dates)))]

    df.to_csv(filename, index=False)
    return pd.read_csv(filename, parse_dates=['Date','Maturity_Date'])


def load_data(filepath='corporate_bond_data.csv'):
    df = pd.read_csv(filepath, parse_dates=['Date','Maturity_Date'])
    return df


# --------------------------------
# Rule-Based Validity Checks
# --------------------------------
def check_validity_rules(df: pd.DataFrame, rules: dict):
    '''
    Apply business validity rules to the dataset.
    rules expects a dict with structure like:
    {
      'coupon_rate_range': {'enabled': True, 'min': 0.01, 'max': 0.10},
      'risk_score_positive': {'enabled': True, 'min': 0.0},
      'rating_categories': {'enabled': True, 'allowed': ['AAA','AA',...]},
      'volume_non_negative': {'enabled': True, 'min': 0.0},
      'maturity_future': {'enabled': True}
    }
    Returns: (results_df, breached_records_dict)
    '''
    results = []
    breached_records = {}
    total = len(df)

    # Coupon rate range
    if rules.get('coupon_rate_range', {}).get('enabled', False):
        rmin = rules['coupon_rate_range'].get('min', 0.01)
        rmax = rules['coupon_rate_range'].get('max', 0.10)
        mask = ~(df['Coupon_Rate'].between(rmin, rmax, inclusive='both'))
        mask = mask | df['Coupon_Rate'].isna()
        breaches = df[mask]
        count = len(breaches)
        pct = 100.0 * count / total if total else 0.0
        results.append({'Rule': 'Coupon Rate Range', 'Column': 'Coupon_Rate', 'Breach_Count': count, 'Breach_Percentage': pct, 'Status': 'PASS' if count==0 else 'FAIL'})
        if count:
            breached_records['coupon_rate_range'] = breaches

    # Risk score non-negative (>= min)
    if rules.get('risk_score_positive', {}).get('enabled', False):
        smin = rules['risk_score_positive'].get('min', 0.0)
        mask = (df['Risk_Score'] < smin) | df['Risk_Score'].isna()
        breaches = df[mask]
        count = len(breaches)
        pct = 100.0 * count / total if total else 0.0
        results.append({'Rule': 'Risk Score Non-Negative', 'Column': 'Risk_Score', 'Breach_Count': count, 'Breach_Percentage': pct, 'Status': 'PASS' if count==0 else 'FAIL'})
        if count:
            breached_records['risk_score_positive'] = breaches

    # Rating categories allowed
    if rules.get('rating_categories', {}).get('enabled', False):
        allowed = set(rules['rating_categories'].get('allowed', []))
        mask = ~(df['Rating'].isin(allowed)) | df['Rating'].isna()
        breaches = df[mask]
        count = len(breaches)
        pct = 100.0 * count / total if total else 0.0
        results.append({'Rule': 'Rating Category Allowed', 'Column': 'Rating', 'Breach_Count': count, 'Breach_Percentage': pct, 'Status': 'PASS' if count==0 else 'FAIL'})
        if count:
            breached_records['rating_categories'] = breaches

    # Volume non-negative
    if rules.get('volume_non_negative', {}).get('enabled', False):
        vmin = rules['volume_non_negative'].get('min', 0.0)
        mask = (df['Volume'] < vmin) | df['Volume'].isna()
        breaches = df[mask]
        count = len(breaches)
        pct = 100.0 * count / total if total else 0.0
        results.append({'Rule': 'Volume Non-Negative', 'Column': 'Volume', 'Breach_Count': count, 'Breach_Percentage': pct, 'Status': 'PASS' if count==0 else 'FAIL'})
        if count:
            breached_records['volume_non_negative'] = breaches

    # Maturity date after trade date
    if rules.get('maturity_future', {}).get('enabled', False):
        mask = (df['Maturity_Date'] <= df['Date']) | df['Maturity_Date'].isna() | df['Date'].isna()
        breaches = df[mask]
        count = len(breaches)
        pct = 100.0 * count / total if total else 0.0
        results.append({'Rule': 'Maturity Date After Trade Date', 'Column': 'Maturity_Date', 'Breach_Count': count, 'Breach_Percentage': pct, 'Status': 'PASS' if count==0 else 'FAIL'})
        if count:
            breached_records['maturity_future'] = breaches

    results_df = pd.DataFrame(results)
    return results_df, breached_records


# ---------------------------------------------
# Completeness and Consistency (Statistical)
# ---------------------------------------------
def check_completeness_and_consistency(df: pd.DataFrame, completeness_cols, consistency_cols, outlier_method='IQR', iqr_multiplier=3.0):
    '''
    Computes missingness for selected columns and outliers for selected numerical columns.
    Returns: completeness_df, consistency_df, outlier_records_dict, figures_dict
    '''
    completeness_results = []
    for col in completeness_cols:
        missing_count = int(df[col].isna().sum())
        pct = 100.0 * missing_count / len(df) if len(df) else 0.0
        status = 'PASS' if missing_count == 0 else 'WARN' if pct < 1.0 else 'FAIL'
        completeness_results.append({'Column': col, 'Missing_Count': missing_count, 'Missing_Percentage': pct, 'Status': status})
    completeness_df = pd.DataFrame(completeness_results)

    consistency_results = []
    outlier_records = {}
    figures = {}

    for col in consistency_cols:
        series = df[col].dropna()
        if series.empty:
            consistency_results.append({'Column': col, 'Outlier_Count': 0, 'Outlier_Percentage': 0.0, 'Status': 'PASS'})
            continue

        if outlier_method.upper() == 'IQR':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        else:
            mean = series.mean()
            std = series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0
            z = (df[col] - mean) / std
            z_threshold = 3.0
            mask = (z.abs() > z_threshold)
            lower = mean - z_threshold*std
            upper = mean + z_threshold*std

        mask = mask & df[col].notna()
        outliers_df = df[mask][['Bond_ID','Date',col]].copy()
        count = len(outliers_df)
        pct = 100.0 * count / len(df[col].dropna()) if len(df[col].dropna()) else 0.0
        status = 'PASS' if count == 0 else 'WARN' if pct < 1.0 else 'FAIL'
        consistency_results.append({'Column': col, 'Outlier_Count': count, 'Outlier_Percentage': pct, 'Status': status})
        if count:
            outlier_records[col] = outliers_df

        # Visualization: histogram with boundaries and boxplot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df[col], bins=40, kde=True, ax=axes[0], color='#4C78A8')
        axes[0].axvline(lower, color='red', linestyle='--', label='Lower Bound')
        axes[0].axvline(upper, color='red', linestyle='--', label='Upper Bound')
        axes[0].set_title(f'Distribution of {col}')
        axes[0].legend()
        sns.boxplot(x=df[col], ax=axes[1], color='#72B7B2')
        axes[1].axvline(lower, color='red', linestyle='--')
        axes[1].axvline(upper, color='red', linestyle='--')
        axes[1].set_title(f'Boxplot with Outlier Bounds')
        fig.tight_layout()
        figures[col] = fig

    consistency_df = pd.DataFrame(consistency_results)
    return completeness_df, consistency_df, outlier_records, figures


# ---------------------------------------------
# Timeliness and Duplicates
# ---------------------------------------------
def check_timeliness_and_duplicates(df: pd.DataFrame, date_col='Date', id_cols=None):
    '''
    Checks for gaps in the daily timeline per Bond_ID and duplicate records by id_cols.
    Returns: results_df, breached_records_dict
    '''
    if id_cols is None:
        id_cols = ['Bond_ID','Date']

    results = []
    breaches = {}

    # Timeliness: detect missing dates per Bond_ID
    gaps_list = []
    for bond_id, grp in df.groupby('Bond_ID'):
        grp_sorted = grp.sort_values(by=date_col)
        full_range = pd.date_range(start=grp_sorted[date_col].min(), end=grp_sorted[date_col].max(), freq='D')
        missing = full_range.difference(grp_sorted[date_col].unique())
        for d in missing:
            gaps_list.append({'Bond_ID': bond_id, 'Missing_Date': pd.to_datetime(d)})
    gaps_df = pd.DataFrame(gaps_list)
    gap_count = len(gaps_df)
    gap_pct = 100.0 * gap_count / (df['Bond_ID'].nunique() * df.groupby('Bond_ID')[date_col].apply(lambda s: s.nunique()).mean()) if len(df) else 0.0
    results.append({'Metric': 'Timeliness Missing Dates', 'Count': gap_count, 'Percentage': gap_pct, 'Status': 'PASS' if gap_count==0 else 'FAIL'})
    if gap_count:
        breaches['date_timeliness_gaps'] = gaps_df

    # Duplicates: detect duplicates by id_cols
    dup_mask = df.duplicated(subset=id_cols, keep=False)
    dup_df = df[dup_mask].sort_values(by=id_cols)
    dup_count = len(dup_df)
    dup_pct = 100.0 * dup_count / len(df) if len(df) else 0.0
    results.append({'Metric': 'Duplicate Records', 'Count': dup_count, 'Percentage': dup_pct, 'Status': 'PASS' if dup_count==0 else 'FAIL'})
    if dup_count:
        breaches['duplicate_records'] = dup_df

    results_df = pd.DataFrame(results)
    return results_df, breaches


# ---------------------------------------------
# Data Quality Scorecard
# ---------------------------------------------
class DataQualityScorer:
    def __init__(self, weights=None):
        default_weights = {
            'completeness': 0.25,
            'validity': 0.35,
            'consistency': 0.20,
            'timeliness': 0.10,
            'uniqueness': 0.10
        }
        self.weights = weights if weights is not None else default_weights

    @staticmethod
    def _calculate_dimension_score(breach_percentage: float) -> float:
        return max(0.0, 1.0 - (breach_percentage / 100.0))

    def generate_scorecard(self, completeness_df: pd.DataFrame, validity_df: pd.DataFrame, consistency_df: pd.DataFrame, timeliness_duplicate_df: pd.DataFrame):
        # Compute average breach percentage per dimension
        comp_pct = float(completeness_df['Missing_Percentage'].mean()) if completeness_df is not None and not completeness_df.empty else 0.0
        valid_pct = float(validity_df['Breach_Percentage'].mean()) if validity_df is not None and not validity_df.empty else 0.0
        cons_pct = float(consistency_df['Outlier_Percentage'].mean()) if consistency_df is not None and not consistency_df.empty else 0.0
        time_pct = 0.0
        uniq_pct = 0.0
        if timeliness_duplicate_df is not None and not timeliness_duplicate_df.empty:
            row_time = timeliness_duplicate_df[timeliness_duplicate_df['Metric']=='Timeliness Missing Dates']
            row_dup = timeliness_duplicate_df[timeliness_duplicate_df['Metric']=='Duplicate Records']
            time_pct = float(row_time['Percentage'].values[0]) if not row_time.empty else 0.0
            uniq_pct = float(row_dup['Percentage'].values[0]) if not row_dup.empty else 0.0

        scores = {
            'completeness': self._calculate_dimension_score(comp_pct),
            'validity': self._calculate_dimension_score(valid_pct),
            'consistency': self._calculate_dimension_score(cons_pct),
            'timeliness': self._calculate_dimension_score(time_pct),
            'uniqueness': self._calculate_dimension_score(uniq_pct),
        }

        # Weighted overall
        numer = sum(scores[k] * self.weights.get(k, 0.0) for k in scores)
        denom = sum(self.weights.get(k, 0.0) for k in scores)
        overall = numer / denom if denom else 0.0

        scorecard_df = pd.DataFrame({
            'Dimension': list(scores.keys()),
            'Average_Breach_Percentage': [comp_pct, valid_pct, cons_pct, time_pct, uniq_pct],
            'Score': [scores['completeness'], scores['validity'], scores['consistency'], scores['timeliness'], scores['uniqueness']]
        })
        scorecard_df['Overall_Score'] = overall

        return scorecard_df, scores


# ---------------------------------------------
# Breach Summaries and Sampling
# ---------------------------------------------
def summarize_rule_breaches(rule_check_results: pd.DataFrame, breached_records: dict):
    if rule_check_results is None or rule_check_results.empty:
        return pd.DataFrame(columns=['Rule','Column','Breach_Count','Breach_Percentage','Status'])
    return rule_check_results.sort_values(by='Breach_Percentage', ascending=False).reset_index(drop=True)


def get_breach_sample(breached_records: dict, rule_name: str, n: int = 5):
    if not breached_records or rule_name not in breached_records:
        return pd.DataFrame()
    df_rule = breached_records[rule_name]
    n = min(n, len(df_rule))
    return df_rule.sample(n=n, random_state=42).sort_values(by=['Bond_ID','Date'])


# ---------------------------------------------
# Statistical Issue Summary and Plotting
# ---------------------------------------------
def summarize_statistical_issues(completeness_df: pd.DataFrame, consistency_df: pd.DataFrame):
    parts = []
    if completeness_df is not None and not completeness_df.empty:
        tmp = completeness_df[['Column','Missing_Count','Missing_Percentage','Status']].copy()
        tmp.rename(columns={'Missing_Count':'Count','Missing_Percentage':'Percentage'}, inplace=True)
        tmp['Type'] = 'Missing'
        parts.append(tmp)
    if consistency_df is not None and not consistency_df.empty:
        tmp = consistency_df[['Column','Outlier_Count','Outlier_Percentage','Status']].copy()
        tmp.rename(columns={'Outlier_Count':'Count','Outlier_Percentage':'Percentage'}, inplace=True)
        tmp['Type'] = 'Outliers'
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(columns=['Type','Column','Count','Percentage','Status'])
    combined = pd.concat(parts)
    combined = combined[['Type','Column','Count','Percentage','Status']].sort_values(by='Percentage', ascending=False).reset_index(drop=True)
    return combined


def plot_outlier_distribution(df: pd.DataFrame, outlier_records: dict, column_name: str, method: str = 'IQR', iqr_multiplier: float = 3.0):
    series = df[column_name].dropna()
    if series.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'No data available for {column_name}', ha='center', va='center')
        ax.axis('off')
        return fig

    if method.upper() == 'IQR':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
    else:
        mean = series.mean()
        std = series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0
        z_threshold = 3.0
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, bins=40, kde=True, ax=ax, color='#4C78A8')
    ax.axvline(lower, color='red', linestyle='--', label='Lower Bound')
    ax.axvline(upper, color='red', linestyle='--', label='Upper Bound')
    ax.set_title(f'Outlier Distribution: {column_name}')
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------
# AI Anomaly Detection
# ---------------------------------------------
def apply_ai_anomaly_detection(df: pd.DataFrame, features_for_anomaly_detection, contamination_rate: float = 0.015, random_state: int = 42):
    '''
    Apply Isolation Forest (PyOD) for anomaly detection on selected features.
    Returns: df_with_anomalies, model, anomaly_summary_dict
    '''
    X = df[features_for_anomaly_detection].copy()

    # Handle missing values by simple imputation (median), realistic for anomaly detection pre-processing
    X_imputed = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    model = IsolationForest(contamination=contamination_rate, random_state=random_state)
    model.fit(X_scaled)

    # PyOD: 1 for outliers, 0 for inliers
    labels = model.predict(X_scaled)
    scores = model.decision_scores_

    df_out = df.copy()
    df_out['anomaly_score'] = scores
    df_out['is_anomaly'] = labels.astype(int) == 1

    anomaly_count = int((df_out['is_anomaly']).sum())
    total = len(df_out)
    anomaly_pct = 100.0 * anomaly_count / total if total else 0.0
    summary = {
        'Anomaly_Count': anomaly_count,
        'Total_Records': total,
        'Anomaly_Percentage': anomaly_pct
    }

    return df_out, model, summary


def plot_anomaly_scores(df_with_anomalies: pd.DataFrame, anomaly_score_col: str = 'anomaly_score', is_anomaly_col: str = 'is_anomaly'):
    # Histogram of anomaly scores
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df_with_anomalies, x=anomaly_score_col, hue=is_anomaly_col, bins=50, kde=True, ax=ax1, palette='Set1', stat='count')
    ax1.set_title('Anomaly Score Distribution')
    ax1.legend(title='Is Anomaly')
    fig1.tight_layout()

    # Time series of anomaly score (sampled for readability)
    df_plot = df_with_anomalies.copy()
    if 'Date' in df_plot.columns:
        df_plot = df_plot.sort_values(by='Date')
        if len(df_plot) > 1000:
            df_plot = df_plot.iloc[::max(1, len(df_plot)//1000), :]
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    if 'Date' in df_plot.columns:
        ax2.plot(df_plot['Date'], df_plot[anomaly_score_col], color='#4C78A8', label='Anomaly Score')
        ax2.scatter(df_plot.loc[df_plot[is_anomaly_col], 'Date'], df_plot.loc[df_plot[is_anomaly_col], anomaly_score_col], color='red', s=10, label='Detected Anomaly')
        ax2.set_xlabel('Date')
    else:
        ax2.plot(np.arange(len(df_plot)), df_plot[anomaly_score_col], color='#4C78A8', label='Anomaly Score')
        ax2.scatter(np.where(df_plot[is_anomaly_col])[0], df_plot.loc[df_plot[is_anomaly_col], anomaly_score_col], color='red', s=10, label='Detected Anomaly')
        ax2.set_xlabel('Record Index')
    ax2.set_ylabel('Score')
    ax2.set_title('Anomaly Scores Over Time')
    ax2.legend()
    fig2.tight_layout()

    return fig1, fig2


# ---------------------------------------------
# Alerts and Reporting
# ---------------------------------------------
def check_for_critical_alerts(dimension_scores: dict, data_quality_scorecard: pd.DataFrame, anomaly_summary: dict,
                              overall_score_threshold: float = 0.75,
                              completeness_score_threshold: float = 0.80,
                              validity_score_threshold: float = 0.80,
                              ai_anomaly_percentage_threshold: float = 2.0):
    alerts = []

    if dimension_scores is None or data_quality_scorecard is None:
        alerts.append('Scorecard not available. Please generate the scorecard before checking alerts.')
        return alerts

    overall_score = float(data_quality_scorecard['Overall_Score'].iloc[0]) if not data_quality_scorecard.empty else 0.0
    if overall_score < overall_score_threshold:
        alerts.append(f'Overall data quality score {overall_score:.2f} is below threshold {overall_score_threshold:.2f}.')

    if dimension_scores.get('completeness', 1.0) < completeness_score_threshold:
        alerts.append(f'Completeness score {dimension_scores.get("completeness", 0.0):.2f} is below threshold {completeness_score_threshold:.2f}.')

    if dimension_scores.get('validity', 1.0) < validity_score_threshold:
        alerts.append(f'Validity score {dimension_scores.get("validity", 0.0):.2f} is below threshold {validity_score_threshold:.2f}.')

    if anomaly_summary is not None and anomaly_summary.get('Anomaly_Percentage', 0.0) > ai_anomaly_percentage_threshold:
        alerts.append(f'AI-detected anomaly percentage {anomaly_summary.get("Anomaly_Percentage", 0.0):.2f}% exceeds threshold {ai_anomaly_percentage_threshold:.2f}%.')

    return alerts


def generate_data_quality_impact_report(data_quality_scorecard: pd.DataFrame, rule_check_results: pd.DataFrame,
                                        statistical_issues_summary: pd.DataFrame, anomaly_summary: dict, current_alerts: list):
    '''
    Compose a text-based impact report consolidating the scorecard, breaches, statistical issues, AI anomalies, and alerts.
    Returns: report_table_df, report_string
    '''
    lines = []
    lines.append('# Data Quality Impact Report')
    lines.append('')

    # Overall score
    if data_quality_scorecard is not None and not data_quality_scorecard.empty:
        overall = float(data_quality_scorecard['Overall_Score'].iloc[0])
        lines.append(f'- Overall Data Quality Score: {overall:.3f}')
    else:
        lines.append('- Overall Data Quality Score: N/A')

    # Dimension scores
    if data_quality_scorecard is not None and not data_quality_scorecard.empty:
        for _, row in data_quality_scorecard.iterrows():
            lines.append(f"  - {row['Dimension'].title()} Score: {row['Score']:.3f} (Avg Breach: {row['Average_Breach_Percentage']:.2f}%)")

    # Validity breaches summary
    lines.append('')
    lines.append('## Validity Breaches Summary')
    if rule_check_results is not None and not rule_check_results.empty:
        for _, r in rule_check_results.iterrows():
            lines.append(f"- {r['Rule']}: Breaches={int(r['Breach_Count'])}, Percentage={r['Breach_Percentage']:.2f}% (Status: {r['Status']})")
    else:
        lines.append('- No validity check results available.')

    # Statistical issues summary
    lines.append('')
    lines.append('## Statistical Issues Summary')
    if statistical_issues_summary is not None and not statistical_issues_summary.empty:
        for _, s in statistical_issues_summary.head(20).iterrows():
            lines.append(f"- {s['Type']} - {s['Column']}: Count={int(s['Count'])}, Percentage={s['Percentage']:.2f}% (Status: {s['Status']})")
    else:
        lines.append('- No statistical issues computed.')

    # AI anomalies summary
    lines.append('')
    lines.append('## AI-Detected Anomalies')
    if anomaly_summary is not None and len(anomaly_summary)>0:
        lines.append(f"- Anomaly Count: {anomaly_summary.get('Anomaly_Count', 0)}")
        lines.append(f"- Anomaly Percentage: {anomaly_summary.get('Anomaly_Percentage', 0.0):.2f}%")
    else:
        lines.append('- AI anomaly detection not run.')

    # Alerts
    lines.append('')
    lines.append('## Critical Alerts')
    if current_alerts:
        for a in current_alerts:
            lines.append(f'- ALERT: {a}')
    else:
        lines.append('- No critical alerts triggered based on current thresholds.')

    report_string = '\n'.join(lines)

    # Tabular summarized view for export
    rows = []
    if data_quality_scorecard is not None and not data_quality_scorecard.empty:
        for _, row in data_quality_scorecard.iterrows():
            rows.append({'Section': 'Scorecard', 'Item': row['Dimension'], 'Metric': 'Score', 'Value': f"{row['Score']:.3f}"})
            rows.append({'Section': 'Scorecard', 'Item': row['Dimension'], 'Metric': 'Avg Breach %', 'Value': f"{row['Average_Breach_Percentage']:.2f}"})
        rows.append({'Section': 'Scorecard', 'Item': 'Overall', 'Metric': 'Overall Score', 'Value': f"{float(data_quality_scorecard['Overall_Score'].iloc[0]):.3f}"})

    if rule_check_results is not None and not rule_check_results.empty:
        for _, r in rule_check_results.iterrows():
            rows.append({'Section': 'Validity', 'Item': r['Rule'], 'Metric': 'Breach %', 'Value': f"{r['Breach_Percentage']:.2f}"})

    if statistical_issues_summary is not None and not statistical_issues_summary.empty:
        for _, s in statistical_issues_summary.iterrows():
            rows.append({'Section': 'Statistical', 'Item': f"{s['Type']} - {s['Column']}", 'Metric': 'Percentage', 'Value': f"{s['Percentage']:.2f}"})

    if anomaly_summary is not None and len(anomaly_summary)>0:
        rows.append({'Section': 'AI', 'Item': 'Anomalies', 'Metric': 'Percentage', 'Value': f"{anomaly_summary.get('Anomaly_Percentage', 0.0):.2f}"})

    if current_alerts:
        for a in current_alerts:
            rows.append({'Section': 'Alerts', 'Item': 'Critical', 'Metric': 'Message', 'Value': a})

    report_df = pd.DataFrame(rows)
    return report_df, report_string
