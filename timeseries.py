"""Time series visualization page"""

import plotly.express as px
import streamlit as st

df = st.session_state.data

st.markdown("## Historical Stock Prices")
st.markdown("These charts show the historical price movements for each stock:")

# Create figure for AAPL
fig1 = px.line(df.reset_index(), x='Date', y='AAPL')
fig1.update_layout(
    title="Apple (AAPL) Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    showlegend=False
)

# Create figure for TSLA
fig2 = px.line(df.reset_index(), x='Date', y='TSLA')
fig2.update_layout(
    title="Tesla (TSLA) Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    showlegend=False
)

# Create figure for NVDA
fig3 = px.line(df.reset_index(), x='Date', y='NVDA')
fig3.update_layout(
    title="NVIDIA (NVDA) Stock Price History",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    showlegend=False
)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

if 'data' not in st.session_state:
    st.session_state['data'] = df