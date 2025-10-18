# src/metrics.py
import pandas as pd
import numpy as np

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily pct returns (drop initial NaNs)."""
    return prices.pct_change().dropna(how="all")

def rolling_cum_return(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling cumulative return over 'window' trading days (approx).
    Returns DataFrame aligned with original index; early rows NaN.
    """
    daily = pct_returns(prices)
    # compute rolling product of (1 + r) then -1
    roll = (1 + daily).rolling(window=window).apply(lambda x: x.prod() - 1, raw=True)
    return roll

def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Cumulative returns from first available date (i.e., growth of $1)."""
    daily = pct_returns(prices)
    return (1 + daily).cumprod()

def annualized_vol(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    return daily_returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(daily_returns: pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> pd.Series:
    rf_daily = risk_free_rate / periods_per_year
    excess = daily_returns - rf_daily
    # avoid division by zero
    denom = excess.std()
    denom[denom == 0] = np.nan
    return (excess.mean() / denom) * np.sqrt(periods_per_year)
