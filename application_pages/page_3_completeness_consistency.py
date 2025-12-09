# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import check_completeness_and_consistency


def main():
    st.header('3. Completeness & Consistency')

    st.markdown('''
Now you quantify missingness and detect extreme values that can distort model training. You focus on features your models actually use, so your checks align with production risk.

- What you are trying to achieve: Ensure critical inputs are present and values behave as expected.
- How this helps your workflow: You decide if imputation or capping is needed, and where.
- How the method supports the action: Statistical thresholds (IQR or Z-Score) turn your tolerance into measurable criteria for outliers.
''')

    if 'df_bonds' not in st.session_state:
        st.error('Dataset not loaded yet. Go to \"1. Data Overview\" and load the corporate bond data.')
        return

    default_completeness = ['Volume','Rating','Risk_Score','Index_Value','Coupon_Rate']
    default_consistency = ['Index_Value','Volume','Coupon_Rate','Risk_Score']

    completeness_cols = st.multiselect('Select Columns for Completeness Check', options=list(st.session_state['df_bonds'].columns), default=default_completeness)
    consistency_cols = st.multiselect('Select Numerical Columns for Outlier Detection', options=['Index_Value','Volume','Coupon_Rate','Risk_Score'], default=default_consistency)
    method = st.radio('Outlier Detection Method', options=['IQR','Z-Score'], index=0)
    iqr_k = st.slider('IQR Multiplier (k)', min_value=1.0, max_value=5.0, value=3.0, step=0.1, help='Adjust the sensitivity of outlier detection. A higher k (e.g., 3.0) focuses on more extreme outliers, crucial in finance where significant deviations might be legitimate events rather than errors.')

    if st.button('Run Statistical Checks'):
        comp_df, cons_df, outliers, figs = check_completeness_and_consistency(st.session_state['df_bonds'], completeness_cols, consistency_cols, outlier_method=method, iqr_multiplier=iqr_k)
        st.session_state['completeness_results'] = comp_df
        st.session_state['consistency_results'] = cons_df
        st.session_state['statistical_outliers'] = outliers
        st.session_state['consistency_plots'] = figs
        st.session_state['consistency_check_columns'] = consistency_cols
        st.session_state['outlier_method'] = method
        st.session_state['iqr_multiplier'] = iqr_k
        st.success('Statistical checks completed. Review results below.')

    if 'completeness_results' in st.session_state:
        st.subheader('Completeness Results')
        st.dataframe(st.session_state['completeness_results'])
        st.info('As a Quantitative Analyst, missing data means incomplete information for models, potentially leading to biased training or reduced prediction accuracy. Consider imputation strategies for critical columns.')

    if 'consistency_results' in st.session_state:
        st.subheader('Consistency (Outlier) Results')
        st.dataframe(st.session_state['consistency_results'])

    if 'consistency_plots' in st.session_state:
        st.subheader('Distributions with Outlier Bounds')
        for col, fig in st.session_state['consistency_plots'].items():
            st.pyplot(fig)
            plt.close(fig)

    st.markdown(r'The formula for IQR-based outlier detection for a data point $x$ is:')
    st.markdown(r'$$ x < Q1 - k \cdot IQR \quad \text{or} \quad x > Q3 + k \cdot IQR $$')
    st.markdown(r'where $Q1$ is the 25th percentile, $Q3$ is the 75th percentile, $IQR = Q3 - Q1$, and $k$ is your chosen multiplier.')
