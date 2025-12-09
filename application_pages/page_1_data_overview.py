# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import StringIO
from utils import generate_and_load_corporate_bond_data


def main():
    st.header('1. Data Overview')

    st.markdown('''
You are the Quantitative Analyst in the DataOps team. A new corporate bond dataset has just arrived for training a risk model. Your goal on this page is to bring the data into your workbench and quickly assess if anything looks off before you proceed.

- What you are trying to achieve: Establish a clean starting point by loading a fresh dataset and scanning its structure and basic stats.
- How this helps your workflow: Early visibility into types, ranges, and suspicious values helps you decide which checks to prioritize on the next pages.
- How the method supports the action: The system simulates a realistic panel dataset across bonds and dates, including intentional issues like duplicates and gaps to mirror operational realities.
''')

    if st.button('Load Corporate Bond Data'):
        with st.spinner('Generating and loading data...'):
            df_loaded = generate_and_load_corporate_bond_data()
            st.session_state['df_bonds'] = df_loaded
        st.success('Data loaded successfully! The dataset includes realistic quirks (e.g., duplicate records and a few date gaps) to exercise your checks.')

    if 'df_bonds' in st.session_state:
        st.subheader('Quick Scan')
        st.dataframe(st.session_state['df_bonds'].head())
        buffer = StringIO()
        st.session_state['df_bonds'].info(buf=buffer)
        st.code(buffer.getvalue())
        st.dataframe(st.session_state['df_bonds'].describe(include='all'))

        st.info('Tip: If you notice unusual ranges or unexpected categories at this stage, you can tailor the rules and thresholds in the next steps to focus on these risks.')

    else:
        st.warning('Click \"Load Corporate Bond Data\" to begin your analysis.')
