# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import apply_ai_anomaly_detection, plot_anomaly_scores


def main():
    st.header('8. AI Anomaly Detection')

    st.markdown('''
Traditional checks can miss complex data quality shifts. Here you apply an unsupervised model to highlight subtle anomalies across multiple features.

- What you are trying to achieve: Catch non-obvious, multi-variable irregularities before they flow into model training.
- How this helps your workflow: AI-backed flags provide an additional layer of assurance when rule-based logic looks clean but the data story still feels off.
- How the method supports the action: Isolation Forest scores isolate points that behave differently from the majority, guiding deeper review.
''')

    if 'df_bonds' not in st.session_state:
        st.error('Dataset not loaded yet. Go to \"1. Data Overview\" and load the corporate bond data.')
        return

    default_features = st.session_state.get('ai_features', ['Index_Value','Volume','Coupon_Rate','Risk_Score'])
    features = st.multiselect('Features for AI Anomaly Detection', options=['Index_Value','Volume','Coupon_Rate','Risk_Score'], default=default_features)
    contamination = st.slider('Contamination Rate (e.g., % of expected anomalies)', min_value=0.001, max_value=0.10, value=0.015, step=0.001, format='%.3f')

    if st.button('Run AI Anomaly Detection'):
        df_anom, model, summary = apply_ai_anomaly_detection(st.session_state['df_bonds'], features, contamination_rate=float(contamination))
        st.session_state['df_bonds_anomalies'] = df_anom
        st.session_state['if_model'] = model
        st.session_state['anomaly_summary_dict'] = summary
        st.success('AI anomaly detection completed.')

    if 'anomaly_summary_dict' in st.session_state:
        st.info(f"Anomaly Count: {st.session_state['anomaly_summary_dict'].get('Anomaly_Count', 0)} | Percentage: {st.session_state['anomaly_summary_dict'].get('Anomaly_Percentage', 0.0):.2f}%")

    if 'df_bonds_anomalies' in st.session_state:
        st.subheader('Sample Detected Anomalies')
        st.dataframe(st.session_state['df_bonds_anomalies'][st.session_state['df_bonds_anomalies']['is_anomaly']].head())

        fig1, fig2 = plot_anomaly_scores(st.session_state['df_bonds_anomalies'])
        st.pyplot(fig1)
        plt.close(fig1)
        st.pyplot(fig2)
        plt.close(fig2)
