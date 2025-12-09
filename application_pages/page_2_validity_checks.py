# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from utils import check_validity_rules


def main():
    st.header('2. Validity Checks')

    st.markdown('''
In your role, you translate policy and trading rules into data checks that prevent downstream model surprises. Here, you configure and apply rule-based checks that reflect business and regulatory expectations.

- What you are trying to achieve: Ensure each field respects the firm\'s rulebook before it can influence a model.
- How this helps your workflow: You triage obvious errors early (e.g., negative volumes, invalid ratings) so later statistical and AI steps focus on subtler issues.
- How the method supports the action: The rules below encode accepted ranges and sets; breaches instantly quantify where the dataset defies policy.
''')

    if 'df_bonds' not in st.session_state:
        st.error('Dataset not loaded yet. Go to \"1. Data Overview\" and load the corporate bond data.')
        return

    with st.expander('Configure Validity Rules', expanded=True):
        enable_coupon = st.checkbox('Enable Coupon Rate Range Check', value=True)
        min_coupon = st.number_input('Min Coupon Rate', value=0.01, step=0.001, format='%.3f')
        max_coupon = st.number_input('Max Coupon Rate', value=0.10, step=0.001, format='%.3f')

        enable_risk = st.checkbox('Enable Risk Score Positive Check', value=True)
        min_risk = st.number_input('Min Risk Score', value=0.0, step=0.1)

        enable_rating = st.checkbox('Enable Rating Categories Check', value=True)
        allowed_ratings_str = st.text_input('Allowed Ratings (comma-separated)', value='AAA,AA,A,BBB,BB,B,CCC,CC,C,D')
        allowed_ratings = [r.strip() for r in allowed_ratings_str.split(',') if r.strip()]

        enable_volume = st.checkbox('Enable Volume Non-Negative Check', value=True)
        min_volume = st.number_input('Min Volume', value=0.0, step=0.1)

        enable_maturity = st.checkbox('Enable Maturity Date Future Check', value=True)

    if st.button('Run Validity Checks'):
        validation_rules = {
            'coupon_rate_range': {'enabled': enable_coupon, 'min': min_coupon, 'max': max_coupon},
            'risk_score_positive': {'enabled': enable_risk, 'min': min_risk},
            'rating_categories': {'enabled': enable_rating, 'allowed': allowed_ratings},
            'volume_non_negative': {'enabled': enable_volume, 'min': min_volume},
            'maturity_future': {'enabled': enable_maturity}
        }
        res_df, breaches = check_validity_rules(st.session_state['df_bonds'], validation_rules)
        st.session_state['rule_check_results'] = res_df
        st.session_state['rule_breaches'] = breaches
        st.success('Validity checks completed. Review the table below and proceed to investigate breaches later if required.')

    if 'rule_check_results' in st.session_state:
        st.dataframe(st.session_state['rule_check_results'])

    st.markdown(r'$$ 0.01 \leq \text{Coupon Rate} \leq 0.10, \quad \text{Risk Score} \geq 0.0, \quad \text{Volume} \geq 0.0, \quad \text{Maturity Date} > \text{Date} $$')

    st.info('Interpreting this as a Quant: each inequality represents a gate you set before any modeling. Breaches indicate where operational data creation or ingestion failed your policy controls.')
