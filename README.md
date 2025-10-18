# Sector ETF Aanlyzer (SEA) 📊

**Proprietary Multi-Factor & Rotational Analytics Platform for Institutional Asset Allocation.**

## Overview

This Streamlit application provides a robust, enterprise-grade interface for cross-sectional analysis of US equity sector ETFs (S&P 500 Select Sector SPDRs). It features real-time performance metrics, risk-adjusted return analytics, factor exposure mapping, and a quantitative strategy backtester.

## Key Features

* **Real-Time Monitoring:** Price points and 24-hour return shifts.
* **Quantitative Audit:** Comprehensive institutional metrics including Sharpe Ratio, Sortino Ratio, Max Drawdown, Beta, and Annualized Alpha.
* **Tactical Backtester:** Simulates dynamic sector allocation strategies based on momentum and trend filtering.
* **Factor Mapping:** Maps current sector positioning to systemic style factors (Growth/Value, Cyclicality).
* **External Sentiment Feed:** Aggregates top relevant headlines via NewsAPI (requires API key setup).

## Project Structure

* `streamlit_app.py`: The main application and UI/UX layer.
* `src/data.py`: Handles all data fetching (`yfinance`), preprocessing, and factor/metric calculations.
* `src/metrics.py`: Defines core financial performance and risk calculations.
* `src/plots.py`: Contains functions to generate all interactive Plotly visualizations (Scatter, Cumulative Return).

## Deployment Instructions (Streamlit Cloud)

To deploy this application on Streamlit Cloud, follow these steps:

1.  **Set up Repository:** Push this entire directory to a new public GitHub repository.
2.  **API Key (Optional):** If you wish to enable the News Feed, set your `NEWSAPI_KEY` as a secret in your Streamlit Cloud app settings.
    * **Filename:** `.streamlit/secrets.toml`
    * **Content:** `newsapi_key = "YOUR_NEWSAPI_KEY"` (Use the Streamlit UI to set secrets securely, do not commit this file).
3.  **Deploy:** Link the repository to your Streamlit Cloud workspace, ensuring the main file path is set to `streamlit_app.py`.