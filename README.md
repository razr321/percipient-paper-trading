# Percipient Digital Assets -- Paper Trading Dashboard

Live paper trading dashboard built with Streamlit, simulating four quantitative crypto strategies with $10,000 starting capital.

## Strategies

1. **Kalman Pairs (BTC/ETH)** -- Mean-reversion on the BTC/ETH log-price spread z-score
2. **Dual Momentum** -- Cross-sectional long/short: rank 20 tokens by 30-day return, long top 5, short bottom 5
3. **Residual Momentum** -- Factor-neutral: token return minus BTC return, long top 5 residual winners
4. **On-Chain Momentum** -- Taker buy ratio signals: long when ratio > 0.55, short when < 0.45

## Features

- Live price data from Binance Vision API
- Quarter-Kelly position sizing with 30% max allocation per strategy
- Full fee modeling (0.075% taker + 0.02% slippage + funding rates)
- Auto-refresh every 5 minutes
- Persistent state across sessions via JSON
- Interactive Plotly charts (equity curve, drawdown, rolling Sharpe)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

Push this directory to a GitHub repo and connect it to [share.streamlit.io](https://share.streamlit.io).
