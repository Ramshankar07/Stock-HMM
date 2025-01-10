# Stock Price Forecaster
Deployed Link: https://stock-hmm-analysis.streamlit.app/
This is a Streamlit application designed to forecast stock prices using a Hidden Markov Model (HMM) based on historical stock price data. The application enables users to view 1-day and multi-day forecasts for several major tech stocks, including:
- Apple Inc. (AAPL)
- Tesla Inc. (TSLA)
- NVIDIA Corporation (NVDA)
The HMM models the multivariate (3-dimensional) time series of stock prices, capturing the opening price, closing price, and volume. The model is trained using maximum likelihood estimation to compute the parameters from historical observations. The number of hidden states (a key hyperparameter) is tuned using the Akaike Information Criterion (AIC) to avoid overfitting.
# Features:
- Stock Data Visualization: Users can input a stock ticker symbol and select a date range to visualize historical stock data.
- 1-Day and Multi-Day Forecasts: The app provides probabilistic forecasts for future stock prices.
- Monte Carlo Sampling: Forecasts are generated through Monte Carlo sampling, allowing for robust prediction intervals.
- Technical Indicators: Users can explore technical indicators such as Moving Averages (SMA/EMA), Bollinger Bands, and more.
