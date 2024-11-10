"""Many-day ahead forecast visualization page"""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

### Function ###

@st.cache_data
def make_many_day_forecast_plot(df: pd.DataFrame, forecast: np.ndarray,
    ticker: str, horizon: int) -> go.Figure:
    """
    Generate a many-day-ahead forecast plot with historical data and forecast
    quantiles.
    """
    df.index.names = ['Date']
    forecast_uni = np.squeeze(forecast[:,df.columns == ticker,:])
    
    # Historical data (last 30 trading days)
    df_history = df[[ticker]].iloc[-30:].reset_index()
    last_date = df.index[-1]
    last_value = df[ticker].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(horizon + 1)]
    
    # Calculate forecast quantiles
    dfs_quantile = [pd.DataFrame({
        'Date': future_dates,
        ticker: [last_value] + list(np.quantile(forecast_uni, q=q, axis=1))
    }) for q in [.05, .20, .35, .65, .80, .95]]
    
    # Create the plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df_history['Date'],
        y=df_history[ticker],
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast quantiles with custom colors and styles
    quantile_colors = ['rgba(255,0,0,0.3)', 'rgba(255,165,0,0.3)', 'rgba(255,255,0,0.3)']
    quantile_names = ['95% Interval', '80% Interval', '65% Interval']
    
    for i, (lower_df, upper_df) in enumerate(zip(
        dfs_quantile[:3], dfs_quantile[-3:][::-1]
    )):
        fig.add_trace(go.Scatter(
            x=lower_df['Date'],
            y=lower_df[ticker],
            line=dict(color=quantile_colors[i], width=0),
            name=quantile_names[i],
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=upper_df['Date'],
            y=upper_df[ticker],
            fill='tonexty',
            fillcolor=quantile_colors[i],
            line=dict(color=quantile_colors[i], width=0),
            name=quantile_names[i]
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    return fig

######

st.markdown("""
## Multi-Day Price Forecasts

The colored bands represent different confidence intervals for our predictions:
- The **red** band shows the 90% confidence interval (between 5th and 95th percentiles)
- The **orange** band shows the 60% confidence interval (between 20th and 80th percentiles)
- The **yellow** band shows the 30% confidence interval (between 35th and 65th percentiles)

Wider bands indicate more uncertainty in the prediction.
""")

df = st.session_state.data
price_forecast = st.session_state.price_forecast
horizon = st.session_state.horizon

fig1 = make_many_day_forecast_plot(df, price_forecast, 'AAPL', horizon)
fig2 = make_many_day_forecast_plot(df, price_forecast, 'TSLA', horizon)
fig3 = make_many_day_forecast_plot(df, price_forecast, 'NVDA', horizon)

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)