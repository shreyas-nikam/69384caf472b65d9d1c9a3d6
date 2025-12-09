# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from utils import summarize_rule_breaches, get_breach_sample


def main():
    st.header('6. Investigate Validity Breaches')

    st.markdown('''
Now that you know which rules were violated, you need to review concrete records to explain causes and decide remediation (fix upstream, filter, or override with justification).

- What you are trying to achieve: Trace specific examples to understand whether breaches are genuine business scenarios or data defects.
- How this helps your workflow: Evidence-based remediation decisions reduce rework and improve trust with governance teams.
- How the method supports the action: Drill-down tables show the actual rows that triggered alerts, so stakeholders can validate corrections.
''')

    if 'rule_check_results' not in st.session_state or 'rule_breaches' not in st.session_state:
        st.error('Validity results are not available. Please run checks in \"2. Validity Checks\".')
        return

    breached_rules_summary = summarize_rule_breaches(st.session_state['rule_check_results'], st.session_state['rule_breaches'])
    st.subheader('Breach Summary')
    st.dataframe(breached_rules_summary)

    options = list(st.session_state['rule_breaches'].keys()) if 'rule_breaches' in st.session_state else []
    selected_rule = st.selectbox('Select Breached Rule for Sample Records', options=options, index=0 if options else 0)
    num_samples = st.number_input('Number of Sample Records', min_value=1, max_value=50, value=10)

    if st.button('Show Sample Records') and selected_rule:
        sample_df = get_breach_sample(st.session_state['rule_breaches'], selected_rule, n=int(num_samples))
        if sample_df is not None and not sample_df.empty:
            st.dataframe(sample_df)
        else:
            st.info('No records available for the selected rule (it may have been resolved or filtered).')
