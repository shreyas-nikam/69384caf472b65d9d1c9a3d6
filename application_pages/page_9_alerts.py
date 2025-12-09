# -*- coding: utf-8 -*-
import streamlit as st
from utils import check_for_critical_alerts


def main():
    st.header('9. Alerts Configuration & Status')

    st.markdown('''
At this point, you want proactive monitoring. Define thresholds that matter for your business, then check whether the latest run should trigger action.

- What you are trying to achieve: Convert analysis into continuous guardrails that alert you before model quality degrades.
- How this helps your workflow: Threshold-based alerts create predictable operating procedures with clear escalation paths.
- How the method supports the action: The app compares your current scores and anomaly rates to these thresholds and reports any breaches.
''')

    if 'data_quality_scorecard' not in st.session_state or 'dimension_scores_dict' not in st.session_state or 'anomaly_summary_dict' not in st.session_state:
        st.warning('Some inputs may be missing (scorecard or AI anomalies). You can still set thresholds and check alerts based on available items.')

    overall_thr = st.number_input('Overall Score Threshold', min_value=0.0, max_value=1.0, value=0.70, step=0.01)
    comp_thr = st.number_input('Completeness Score Threshold', min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    valid_thr = st.number_input('Validity Score Threshold', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    ai_thr = st.number_input('AI Anomaly Percentage Threshold', min_value=0.0, max_value=10.0, value=3.0, step=0.1)

    if st.button('Check for Critical Alerts'):
        alerts = check_for_critical_alerts(
            st.session_state.get('dimension_scores_dict', {}),
            st.session_state.get('data_quality_scorecard'),
            st.session_state.get('anomaly_summary_dict', {}),
            overall_score_threshold=float(overall_thr),
            completeness_score_threshold=float(comp_thr),
            validity_score_threshold=float(valid_thr),
            ai_anomaly_percentage_threshold=float(ai_thr)
        )
        st.session_state['current_alerts'] = alerts

    if st.session_state.get('current_alerts'):
        for alert in st.session_state['current_alerts']:
            st.error(alert)
    else:
        st.success('No critical data quality alerts detected.')
