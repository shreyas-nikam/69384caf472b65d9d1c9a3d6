# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from utils import summarize_statistical_issues, plot_outlier_distribution


def main():
    st.header('7. Investigate Statistical Anomalies')

    st.markdown('''
You now explore statistical anomalies flagged earlier. Visual confirmation helps you decide whether to cap, transform, or exclude data before modeling.

- What you are trying to achieve: Understand the shape and extremity of distributions behind outlier flags.
- How this helps your workflow: Choose appropriate pre-processing that preserves signal while removing noise.
- How the method supports the action: Histograms and missingness profiles translate abstract metrics into tangible patterns.
''')

    if 'df_bonds' not in st.session_state or 'completeness_results' not in st.session_state or 'consistency_results' not in st.session_state:
        st.error('Required results are not available. Please run \"3. Completeness & Consistency\" first.')
        return

    summary = summarize_statistical_issues(st.session_state['completeness_results'], st.session_state['consistency_results'])
    st.subheader('Statistical Issues Summary')
    st.dataframe(summary)

    st.subheader('Missingness Matrix')
    fig_m, ax_m = plt.subplots(figsize=(12, 6))
    msno.matrix(st.session_state['df_bonds'], ax=ax_m, sparkline=False)
    st.pyplot(fig_m)
    plt.close(fig_m)

    cols = st.session_state.get('consistency_check_columns', [])
    selected_col = st.selectbox('Select Numerical Column to Plot Outliers', options=cols, index=0 if cols else 0)

    if st.button('Plot Outlier Distribution') and selected_col:
        method = st.session_state.get('outlier_method', 'IQR')
        iqr_k = st.session_state.get('iqr_multiplier', 3.0)
        fig = plot_outlier_distribution(st.session_state['df_bonds'], st.session_state.get('statistical_outliers', {}), selected_col, method=method, iqr_multiplier=iqr_k)
        st.pyplot(fig)
        plt.close(fig)
