# src/data.py
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple

# --- Static Data for Factor Exposure Analysis ---
FACTOR_EXPOSURES = {
    "XLK": {"Style": "Growth", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLF": {"Style": "Value", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLE": {"Style": "Value", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLY": {"Style": "Growth", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLV": {"Style": "Blend", "Sensitivity": "Low Beta", "Cyclicality": "Defensive"},
    "XLU": {"Style": "Value", "Sensitivity": "Low Beta", "Cyclicality": "Defensive"},
    "XLB": {"Style": "Value", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLI": {"Style": "Blend", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLP": {"Style": "Value", "Sensitivity": "Low Beta", "Cyclicality": "Defensive"},
    "XLC": {"Style": "Growth", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
    "XLRE": {"Style": "Value", "Sensitivity": "Low Beta", "Cyclicality": "Defensive"},
    "GLD": {"Style": "Safe Haven", "Sensitivity": "Low Beta", "Cyclicality": "Non-Correlated"},
    "SLV": {"Style": "Safe Haven", "Sensitivity": "Medium Beta", "Cyclicality": "Non-Correlated"},
    "SPY": {"Style": "Benchmark", "Sensitivity": "1.0", "Cyclicality": "Market"},
    "QQQ": {"Style": "Growth", "Sensitivity": "High Beta", "Cyclicality": "Cyclical"},
}

# --- ETF Mapping ---
ETF_INFO = {
    "XLK": ("Technology", "Tracks technology sector performance."),
    "XLF": ("Financials", "Tracks financial sector performance."),
    "XLE": ("Energy", "Tracks energy sector performance."),
    "XLV": ("Health Care", "Tracks healthcare sector performance."),
    "XLY": ("Consumer Discretionary", "Tracks consumer discretionary sector."),
    "XLP": ("Consumer Staples", "Tracks consumer staples sector performance."),
    "XLI": ("Industrials", "Tracks industrial sector performance."),
    "XLB": ("Materials", "Tracks materials sector performance."),
    "XLRE": ("Real Estate", "Tracks real estate sector performance (added for completeness)."),
    "XLU": ("Utilities", "Tracks utilities sector performance."),
    "XLC": ("Communication Services", "Tracks communication services performance."),
    "GLD": ("Gold", "Tracks the price of gold."),
    "SLV": ("Silver", "Tracks the price of silver."),
    "SPY": ("S&P 500 (Benchmark)", "Primary benchmark for the US stock market."),
    "QQQ": ("Nasdaq 100", "Tracks the largest non-financial companies on the Nasdaq."),
}

def _extract_adjclose_from_multi(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Handles yfinance multi-index output (for multiple tickers) to extract the adjusted price.
    Since yfinance defaults to auto_adjust=True, we prioritize 'Close'.
    """
    cols = raw.columns
    
    # Guard against non-MultiIndex data frames mistakenly passed here
    if not isinstance(cols, pd.MultiIndex):
        # Fallback to the general single-ticker logic if possible
        if "Close" in raw.columns:
            return raw[["Close"]].rename(columns={"Close": tickers[0]}) if len(tickers) == 1 else raw[tickers]
        return raw 

    # 1. Try 'Close' (Expected when auto_adjust=True)
    if "Close" in cols.get_level_values(1):
        try:
            return raw.xs("Close", axis=1, level=1)
        except Exception:
            # Fall through to the next check if extraction fails unexpectedly
            pass

    # 2. Fallback to 'Adj Close' (Expected when auto_adjust=False)
    if "Adj Close" in cols.get_level_values(1):
        try:
            return raw.xs("Adj Close", axis=1, level=1)
        except Exception:
            # Fall through to the next check if extraction fails unexpectedly
            pass

    # 3. Last resort: pick first numeric column per ticker
    out = pd.DataFrame(index=raw.index)
    for t in tickers:
        try:
            # Select by ticker (Level 0)
            sub = raw.loc[:, t] 
            for c in sub.columns:
                if pd.api.types.is_numeric_dtype(sub[c]):
                    out[t] = sub[c]
                    break
        except Exception:
            continue
            
    return out

def fetch_prices(tickers: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    """
    Return a DataFrame of adjusted prices, robustly handling yfinance output.
    Uses auto_adjust=True to simplify to 'Close' price.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    tickers = [t.upper() for t in tickers if isinstance(t, str)]
    if not tickers:
        raise ValueError("Ticker list is empty or contains invalid types.")

    # Use auto_adjust=True to get adjusted prices under the 'Close' column
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True, group_by="ticker")
    
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for tickers: {', '.join(tickers)}. Check network or ticker list.")
        
    if isinstance(raw.columns, pd.MultiIndex):
        prices = _extract_adjclose_from_multi(raw, tickers)
    else:
        # Single Ticker download or non-MultiIndex output
        available = list(raw.columns)
        
        # 1. Prefer 'Close' (Correct column name when auto_adjust=True)
        if "Close" in available: 
            # If multiple tickers were requested but only one column came back, 
            # we assume the first ticker was the target.
            prices = raw[["Close"]].copy()
            prices.columns = [tickers[0]]
            
        # 2. Fallback to 'Adj Close' (for older yfinance/auto_adjust=False)
        elif "Adj Close" in available:
            prices = raw[["Adj Close"]].copy()
            prices.columns = [tickers[0]]

        # 3. Last resort: check if ticker names are columns (uncommon with auto_adjust=True)
        elif all(t in available for t in tickers):
            prices = raw[tickers].copy()
            
        else:
            # Final fallback, try to extract any relevant column if structure is unexpected
            prices = pd.DataFrame()
            for t in tickers:
                if t in raw.columns:
                    prices[t] = raw[t]

    present = [c for c in tickers if c in prices.columns]
    prices = prices[present]
    
    if prices.empty:
        raise RuntimeError(f"Data extraction failed for all requested tickers: {', '.join(tickers)}")

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices.dropna(how='all')


def calculate_rrg_components(prices: pd.DataFrame, benchmark_symbol: str = "SPY") -> pd.DataFrame:
    """
    Calculates Relative Strength (RS) Ratio (Long-term) and Momentum (Short-term) for RRG.
    
    NOTE: This function returns a cross-section (snapshot) of the RRG components for the latest date.
    The index is set to the calculation date to allow the external code to use .normalize().
    The 'Symbol' column holds the ticker names.
    """
    if benchmark_symbol not in prices.columns or prices.shape[0] < 60:
        return pd.DataFrame()

    benchmark = prices[benchmark_symbol]
    
    # Calculate daily Relative Strength (RS) for all non-benchmark assets
    rs_df = prices.drop(columns=[benchmark_symbol], errors='ignore').div(benchmark, axis=0)

    # RS-Ratio (X-axis): RS relative to its 60-day SMA (Long-term trend)
    rs_60d_sma = rs_df.rolling(window=60).mean()
    # RS-Momentum (Y-axis): RS relative to its 5-day SMA (Short-term momentum)
    rs_5d_sma = rs_df.rolling(window=5).mean()

    # Calculate RRG components based on latest data point
    latest_rs = rs_df.iloc[-1]
    latest_rs_60d_sma = rs_60d_sma.iloc[-1]
    latest_rs_5d_sma = rs_5d_sma.iloc[-1]

    rs_ratio = (latest_rs / latest_rs_60d_sma) 
    rs_momentum = (latest_rs / latest_rs_5d_sma) 
    
    # Get the latest *valid* date directly from the input prices DataFrame's index.
    if not prices.empty:
        # Get the last date and normalize it to remove time component
        calculation_date = prices.index[-1].normalize()
    else:
        return pd.DataFrame()


    rrg_df = pd.DataFrame({
        'Symbol': rs_ratio.index,
        'Sector': [ETF_INFO.get(t, ('N/A',''))[0] for t in rs_ratio.index],
        'RS_Ratio': rs_ratio,
        'RS_Momentum': rs_momentum
    }).dropna()
    
    # CRITICAL FIX: Explicitly cast the index to DatetimeIndex to prevent inference errors 
    # when the Streamlit app concatenates the results.
    rrg_df.index = pd.DatetimeIndex([calculation_date] * len(rrg_df))
    
    return rrg_df
