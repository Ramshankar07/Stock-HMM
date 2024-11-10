"""One-day ahead forecast visualization page"""

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

### Functions ###

@st.cache_data
def compute_forecast_stats(forecast: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics for 1-day-ahead stock price forecasts."""
    df_ahead = pd.DataFrame(
        columns=['AAPL', 'TSLA', 'NVDA'],
        data=forecast[0,:,:].T
    )
    df_stats = pd.DataFrame(
        index=['AAPL', 'TSLA', 'NVDA'],
        columns=[
            'Mean',
            'Std',
            '5%-percentile',
            '25%-percentile',
            'Median',
            '75%-percentile',
            '95%-percentile'
        ],
        data=[[
            df_ahead['AAPL'].mean(),
            df_ahead['AAPL'].std(),
            df_ahead['AAPL'].quantile(.05),
            df_ahead['AAPL'].quantile(.25),
            df_ahead['AAPL'].median(),
            df_ahead['AAPL'].quantile(.75),
            df_ahead['AAPL'].quantile(.95)
        ], [
            df_ahead['TSLA'].mean(),
            df_ahead['TSLA'].std(),
            df_ahead['TSLA'].quantile(.05),
            df_ahead['TSLA'].quantile(.25),
            df_ahead['TSLA'].median(),
            df_ahead['TSLA'].quantile(.75),
            df_ahead['TSLA'].quantile(.95)
        ], [
            df_ahead['NVDA'].mean(),
            df_ahead['NVDA'].std(),
            df_ahead['NVDA'].quantile(.05),
            df_ahead['NVDA'].quantile(.25),
            df_ahead['NVDA'].median(),
            df_ahead['NVDA'].quantile(.75),
            df_ahead['NVDA'].quantile(.95)
        ]]
    )
    return df_ahead, df_stats

@st.cache_data
def make_one_day_head_forecast_histogram(df: pd.DataFrame, ticker: str
    ) -> go.Figure:
    """
    Create a histogram of 1-day-ahead forecast values for a given stock.
    """
    fig = px.histogram(df[ticker], x=ticker, histnorm='percent', nbins=15)
    fig.update_layout(
        title=f"{ticker} Price Distribution",
        xaxis_title="Predicted Price ($)",
        yaxis_title="Percentage (%)"
    )
    return fig

######

price_forecast = st.session_state.price_forecast

df_ahead, df_stats = compute_forecast_stats(price_forecast)

st.markdown("## One-Day-Ahead Price Distribution")
st.markdown("These histograms show the distribution of predicted prices for tomorrow based on our model:")

col1, col2, col3 = st.columns(3)

fig1 = make_one_day_head_forecast_histogram(df_ahead, 'AAPL')
fig2 = make_one_day_head_forecast_histogram(df_ahead, 'TSLA')
fig3 = make_one_day_head_forecast_histogram(df_ahead, 'NVDA')

with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("## Statistical Summary")
st.markdown("Key statistics for tomorrow's predicted prices (in USD):")
st.write(df_stats.style.format(precision=2))