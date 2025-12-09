# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from utils import check_timeliness_and_duplicates


def main():
    st.header('4. Timeliness & Duplication')

    st.markdown('''
Your models rely on orderly time series without unexpected gaps or accidental record duplication. Here you verify the temporal integrity and uniqueness required for robust backtests and training.

- What you are trying to achieve: Confirm the dataset is complete across time and free from duplicate keys.
- How this helps your workflow: Prevents false patterns created by duplicated rows and missed trading days.
- How the method supports the action: Automated checks scan all bonds and dates, surfacing gaps and collisions you need to remediate.
''')

    if 'df_bonds' not in st.session_state:
        st.error('Dataset not loaded yet. Go to \"1. Data Overview\" and load the corporate bond data.')
        return

    date_col = st.text_input('Date Column Name', value='Date')
    id_cols = st.multiselect('Unique Identifier Columns for Duplicates', options=list(st.session_state['df_bonds'].columns), default=['Bond_ID','Date'])

    if st.button('Run Timeliness & Duplicate Checks'):
        td_df, td_breaches = check_timeliness_and_duplicates(st.session_state['df_bonds'], date_col=date_col, id_cols=id_cols)
        st.session_state['timeliness_duplicate_results'] = td_df
        st.session_state['td_breaches'] = td_breaches
        st.success('Timeliness and duplication checks completed.')

    if 'timeliness_duplicate_results' in st.session_state:
        st.dataframe(st.session_state['timeliness_duplicate_results'])

        if 'duplicate_records' in st.session_state.get('td_breaches', {}):
            st.subheader('Sample Duplicate Records')
            st.dataframe(st.session_state['td_breaches']['duplicate_records'].head())
        if 'date_timeliness_gaps' in st.session_state.get('td_breaches', {}):
            st.subheader('Sample Missing Dates')
            st.dataframe(st.session_state['td_breaches']['date_timeliness_gaps'].head())
