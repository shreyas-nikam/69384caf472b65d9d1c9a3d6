# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DataQualityScorer


def main():
    st.header('5. Data Quality Scorecard')

    st.markdown('''
You now need a single view that summarizes the health of your dataset so you can decide if it\'s ready for model training. Configure the importance of each dimension and generate a transparent scorecard.

- What you are trying to achieve: Turn many checks into a concise, weighted decision metric.
- How this helps your workflow: Score-driven triage helps you justify go/no-go decisions and document rationale for auditors.
- How the method supports the action: Weighted averages reflect business priorities (e.g., validity may count more than timeliness depending on the use case).
''')

    if not all(k in st.session_state for k in ['rule_check_results','completeness_results','consistency_results','timeliness_duplicate_results']):
        st.warning('Some upstream checks are missing. You can still configure weights, but generate the scorecard after running earlier steps for full coverage.')

    with st.expander('Configure Dimension Weights', expanded=True):
        w_completeness = st.slider('Completeness Weight', min_value=0.0, max_value=1.0, value=0.25, step=0.05, key='w_completeness')
        w_validity = st.slider('Validity Weight', min_value=0.0, max_value=1.0, value=0.35, step=0.05, key='w_validity')
        w_consistency = st.slider('Consistency Weight', min_value=0.0, max_value=1.0, value=0.20, step=0.05, key='w_consistency')
        w_timeliness = st.slider('Timeliness Weight', min_value=0.0, max_value=1.0, value=0.10, step=0.05, key='w_timeliness')
        w_uniqueness = st.slider('Uniqueness Weight', min_value=0.0, max_value=1.0, value=0.10, step=0.05, key='w_uniqueness')

    if st.button('Generate Data Quality Scorecard'):
        custom_weights = {
            'completeness': st.session_state.get('w_completeness', 0.25),
            'validity': st.session_state.get('w_validity', 0.35),
            'consistency': st.session_state.get('w_consistency', 0.20),
            'timeliness': st.session_state.get('w_timeliness', 0.10),
            'uniqueness': st.session_state.get('w_uniqueness', 0.10)
        }
        dq = DataQualityScorer(weights=custom_weights)
        scorecard_df, scores_dict = dq.generate_scorecard(
            st.session_state.get('completeness_results', pd.DataFrame()),
            st.session_state.get('rule_check_results', pd.DataFrame()),
            st.session_state.get('consistency_results', pd.DataFrame()),
            st.session_state.get('timeliness_duplicate_results', pd.DataFrame())
        )
        st.session_state['data_quality_scorecard'] = scorecard_df
        st.session_state['dimension_scores_dict'] = scores_dict
        st.success('Scorecard generated. Review the table and chart below.')

    if 'data_quality_scorecard' in st.session_state:
        st.dataframe(st.session_state['data_quality_scorecard'])

    if 'dimension_scores_dict' in st.session_state and st.session_state['dimension_scores_dict']:
        fig, ax = plt.subplots(figsize=(10,6))
        dims = list(st.session_state['dimension_scores_dict'].keys())
        vals = list(st.session_state['dimension_scores_dict'].values())
        sns.barplot(x=dims, y=vals, ax=ax, palette='viridis')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Data Quality Dimension Scores')
        if 'data_quality_scorecard' in st.session_state:
            overall = float(st.session_state['data_quality_scorecard']['Overall_Score'].iloc[0])
            ax.axhline(overall, color='red', linestyle='--', label='Overall Score')
            ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(r'The overall data quality score $S_{overall}$ for a dataset is calculated as a weighted average of individual dimension scores $s_i$:')
    st.markdown(r'$$ S_{overall} = \frac{\sum_{i=1}^{N} s_i \cdot w_i}{\sum_{i=1}^{N} w_i} $$')
    st.markdown(r'For each dimension, the score is derived from breach percentage as $s_i = 1 - \frac{\text{Breach Percentage}_i}{100}$.')
