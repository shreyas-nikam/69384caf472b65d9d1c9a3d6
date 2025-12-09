# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from utils import generate_data_quality_impact_report, summarize_rule_breaches, summarize_statistical_issues


def main():
    st.header('10. Data Quality Impact Report')

    st.markdown('''
Your final task is to communicate findings in a concise, actionable format for senior management and data owners. The report aligns outcomes with decisions and next steps.

- What you are trying to achieve: Provide transparency on data readiness and quantify residual risk.
- How this helps your workflow: A structured summary enables sign-off, remediation tracking, and audit evidence.
- How the method supports the action: The app compiles key metrics and alerts into a coherent narrative so stakeholders can act quickly.
''')

    if 'data_quality_scorecard' not in st.session_state:
        st.warning('Please generate the scorecard in \"5. Data Quality Scorecard\" for a complete report. You can still generate a partial report.')

    if st.button('Generate Impact Report'):
        rule_summary = summarize_rule_breaches(st.session_state.get('rule_check_results', pd.DataFrame()), st.session_state.get('rule_breaches', {}))
        stat_summary = summarize_statistical_issues(st.session_state.get('completeness_results', pd.DataFrame()), st.session_state.get('consistency_results', pd.DataFrame()))
        report_df, report_str = generate_data_quality_impact_report(
            st.session_state.get('data_quality_scorecard', pd.DataFrame()),
            rule_summary,
            stat_summary,
            st.session_state.get('anomaly_summary_dict', {}),
            st.session_state.get('current_alerts', [])
        )
        st.session_state['impact_report_df'] = report_df
        st.session_state['impact_report_str'] = report_str
        st.success('Report generated below. You can copy and share it with stakeholders.')

    if 'impact_report_str' in st.session_state:
        st.markdown(st.session_state['impact_report_str'])

    if 'impact_report_df' in st.session_state:
        st.subheader('Report Table (for export)')
        st.dataframe(st.session_state['impact_report_df'])
