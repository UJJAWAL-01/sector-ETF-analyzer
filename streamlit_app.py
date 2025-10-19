# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# Import custom modules
try:
    from streamlit_autorefresh import st_autorefresh
    HAVE_AUTORE = True
except ImportError:
    HAVE_AUTORE = False

# Import local data/util modules (assuming src/data.py and src/plots.py are available)
try:
    import src.data as data
    import src.plots as plots
except ImportError as e:
    # Fail gracefully: show message but allow program flow so we can inspect / test
    st.error("Fatal Error: Core analytic modules ('src.data', 'src.plots') are inaccessible. "
             "Check repository structure or PYTHONPATH.")
    st.write(f"ImportError: {e}")
    # Stop early to avoid cascading exceptions in the UI (keep but explicit)
    st.stop()


# --- 1. Enterprise Theming and Global Configuration ---
st.set_page_config(
    page_title="Sector ETF Analyzer (SEA)", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:mangumochie95@gmail.com',
        'Report a bug': "mailto:mangumochie95@gmail.com",
        'About': "# Strategic Sector Alpha Generator\n\n**Proprietary Sector Rotation Analytics Platform.**\n\n*Version 1.2.0 - Q4 2024*"
    }
)

# Use a clean, descriptive title
st.title("Sector ETF Analyzer (SEA) 📊")
st.caption("Proprietary Multi-Factor & Rotational Analytics Platform.")


# Custom Streamlit Theme (Enhanced Dark Mode - based on provided styling)
st.markdown(
    """
    <style>
    /* Global Background and Text */
    .stApp {
        background-color: #0d1117; /* Deep Dark BG */
        color: #c9d1d9; /* Light Text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Headers & Subheaders (Use strong color for focus) */
    h1, h2, h3 {
        color: #58a6ff; /* A professional blue for emphasis */
        border-bottom: 1px solid #21262d;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    /* Buttons & Controls */
    .stButton>button {
        background-color: #21262d; 
        color: #c9d1d9; 
        border: 1px solid #30363d;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #30363d; 
        border-color: #58aaee;
    }
    /* DataFrames */
    .stDataFrame {
        font-size: 14px;
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    /* Metrics Boxes - Enhanced Visual Separation */
    [data-testid="stMetric"] {
        background-color: #161b22; 
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricLabel"] {
        font-weight: bold;
        color: #8b949e; 
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------
# SIDEBAR: DATA INGESTION & PARAMETERIZATION
# ----------------------------
st.sidebar.header("Data Ingestion & Parameterization ⚙️")

# Use technical term for the list
st.sidebar.markdown("**Asset Universe Configuration**")
DEFAULT_TICKERS = list(data.ETF_INFO.keys())
DEFAULT_SECTOR_ETFS = [t for t in DEFAULT_TICKERS if t.startswith("XL")]

selected_tickers = st.sidebar.multiselect(
    "Select Component ETF Constituents",
    options=DEFAULT_TICKERS,
    default=DEFAULT_SECTOR_ETFS,
    help="Define the investment universe (sectors, indices, and commodities) for cross-sectional analysis."
)

SYMBOLS = [t for t in selected_tickers if t in data.ETF_INFO]
BENCHMARK_SYMBOL = "SPY"
if BENCHMARK_SYMBOL not in SYMBOLS:
    SYMBOLS.append(BENCHMARK_SYMBOL)

if not SYMBOLS:
    st.error("Configuration Error: The asset universe must contain at least one valid ticker.")
    st.stop()

# Use technical term for the API key
st.sidebar.markdown("---")
st.sidebar.markdown("**External Data Access**")
_api_key = None
try:
    _api_key = st.secrets.get("newsapi_key", "")
except Exception:
    _api_key = ""

if not _api_key:
    tmp_key = st.sidebar.text_input("Key", type="password")
    if tmp_key:
        _api_key = tmp_key

# Use technical term for the date range
st.sidebar.markdown("---")
st.sidebar.markdown("**Historical Lookback Period**")
start_date = st.sidebar.date_input("Inception Date for Time Series", value=datetime.now().date() - timedelta(days=365*2), 
                                   help="Sets the start date for all price and performance calculations (e.g., Sharpe, RRG).")

# Auto-refresh toggle (technical language)
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Enable Auto-Synchronization (15 min interval)", value=False)
if auto_refresh:
    if HAVE_AUTORE:
        st.sidebar.info("Auto-Synchronization Protocol: Active.")
        st_autorefresh(interval=15 * 60 * 1000, key="data_autorefresh")
    else:
        st.sidebar.warning("`streamlit-autorefresh` dependency not satisfied.")

manual_refresh = st.sidebar.button("🔁 Execute Data Refresh & Cache Invalidation")

# ----------------------------
# CACHING / DATA FETCH 
# ----------------------------
@st.cache_data(ttl=60 * 30)
def load_all_data(tickers, start):
    """Fetch and calculate all core data structures."""
    try:
        # Note: data.py might still calculate RRG, but we ignore the output here
        prices = data.fetch_prices(tickers, start=start.strftime('%Y-%m-%d'))
        if prices.empty:
            raise RuntimeError("Data Ingestion Error: Zero data points returned for the specified time frame.")
        
        returns = prices.pct_change().dropna(how="all")
        cumulative = (1 + returns).cumprod().fillna(1.0)
        
        # We still call the RRG function from data.py to maintain the structure, 
        # but we don't return the rrg_df, effectively removing it from the app flow.
        # Assuming calculate_rrg_components is in data.py and returns a DataFrame
        data.calculate_rrg_components(prices, benchmark_symbol=BENCHMARK_SYMBOL)
        
        # Only return prices, returns, cumulative
        return prices, returns, cumulative
    except Exception as e:
        st.error(f"Time Series Ingestion Failure: {e}. Validate connectivity and ticker configuration.")
        empty_df = pd.DataFrame()
        # Return empty dataframes (no rrg_df anymore)
        return empty_df, empty_df, empty_df

@st.cache_data(ttl=60 * 60)
def get_sector_news_cached(sector_keyword, api_key):
    """Fetch top headlines for a sector via NewsAPI (cached)."""
    if not api_key: return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"'{sector_keyword}' sector", 
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": api_key
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            articles = r.json().get("articles", [])
            return [{"title": a["title"], "url": a["url"], "source": a["source"]["name"]} for a in articles]
        else:
            st.warning(f"External Feed Failure (NewsAPI): Status code {r.status_code}. Verify API quota.")
            return []
    except Exception as e:
        st.error(f"News API Connection Disruption: {e}")
        return []
    return []


# If manual refresh requested, clear caches to force refetch
if manual_refresh:
    try:
        load_all_data.clear()
        get_sector_news_cached.clear() 
        st.sidebar.success("Analytic Engine Caches invalidated. Executing data reload...")
    except Exception:
        pass

# Load all data (Receiving 3 dataframes now: prices, returns, cumulative)
prices, returns, cumulative = load_all_data(SYMBOLS, start=start_date)

if prices.empty:
    st.stop()

available_symbols = [s for s in SYMBOLS if s in prices.columns]
if not available_symbols:
    st.error("Data Validation Failure: No price series available after filtering.")
    st.stop()


# ----------------------------
# Institutional Analytics Calculation (Moved here as it's needed globally)
# ----------------------------
@st.cache_data(ttl=60 * 30)
def calculate_analytics(returns_df, cumulative_df, benchmark_symbol):
    """Calculates all risk/return metrics."""
    
    def calculate_volatility(returns_df, annualize=True):
        return returns_df.std() * (np.sqrt(252) if annualize else 1) * 100

    def calculate_max_drawdown(cum_df):
        roll_max = cum_df.cummax()
        drawdown = (cum_df / roll_max) - 1
        return drawdown.min() * 100

    def calculate_sharpe_ratio(returns_df, rf=0.02):
        rf_daily = rf / 252
        excess = returns_df.sub(rf_daily)
        denom = excess.std()
        denom[denom == 0] = np.nan
        return (excess.mean() / denom) * np.sqrt(252)

    def calculate_sortino_ratio(returns_df, rf=0.02):
        rf_daily = rf / 252
        excess = returns_df.sub(rf_daily)
        downside_returns = returns_df[returns_df < rf_daily].fillna(0)
        
        downside_std = downside_returns.std()
        downside_std[downside_std == 0] = np.nan
        
        annualized_excess_mean = excess.mean() * 252
        annualized_downside_std = downside_std * np.sqrt(252)

        return annualized_excess_mean / annualized_downside_std

    def calculate_beta_alpha(returns_df, benchmark_series):
        betas = {}; alphas = {}
        for col in returns_df.columns:
            try:
                common_data = pd.concat([returns_df[col], benchmark_series], axis=1).dropna()
                if common_data.empty: raise ValueError("No common data points.")
                
                asset_returns = common_data.iloc[:, 0]
                bench_returns = common_data.iloc[:, 1]
                
                cov = np.cov(asset_returns, bench_returns)[0, 1]
                bench_var = bench_returns.var()
                
                beta = cov / bench_var if bench_var != 0 else np.nan
                alpha = (asset_returns.mean() * 252) - (beta * bench_returns.mean() * 252)
            except Exception:
                beta = np.nan
                alpha = np.nan
            betas[col] = beta
            alphas[col] = alpha
        return pd.Series(betas), pd.Series(alphas)

    # --- Calculations ---
    volatility = calculate_volatility(returns_df)
    max_dd = calculate_max_drawdown(cumulative_df)
    sharpe = calculate_sharpe_ratio(returns_df)
    sortino = calculate_sortino_ratio(returns_df)

    benchmark = returns_df[benchmark_symbol] if benchmark_symbol in returns_df.columns else None
    
    if benchmark is not None:
        beta, alpha = calculate_beta_alpha(returns_df, benchmark)
    else:
        beta = pd.Series(np.nan, index=returns_df.columns)
        alpha = pd.Series(np.nan, index=returns_df.columns)

    analytics_df = pd.DataFrame({
        "Symbol": returns_df.columns,
        "Sector": [data.ETF_INFO.get(s, (s,''))[0] for s in returns_df.columns],
        "Volatility (%)": volatility.round(2),
        "Max Drawdown (%)": max_dd.round(2),
        "Sharpe": sharpe.round(2),
        "Sortino": sortino.round(2),
        "Beta (vs SPY)": beta.round(2),
        "Alpha (Annualized)": alpha.round(3),
    }).set_index("Symbol")
    
    return analytics_df, volatility, sharpe

# Run analytics once
analytics_df, volatility_series, sharpe_series = calculate_analytics(returns, cumulative, BENCHMARK_SYMBOL)

# ----------------------------
# DASHBOARD LAYOUT (Updated tab structure)
# ----------------------------

tab1, tab2 = st.tabs(["🎯 Rotational Momentum & Performance", "📰 Sentiment & Factor Exposure"])

with tab1:
    # --- Live Snapshot: Use a more technical title and structure ---
    st.header("Real-Time Price & Performance Monitoring 📈")
    
    # Use a maximum of 6 columns for better spacing on large monitors
    metrics_cols = st.columns(min(len(available_symbols), 6))

    for i, sym in enumerate(available_symbols):
        curr = prices[sym].iloc[-1] if len(prices[sym]) >= 1 else np.nan
        prev = prices[sym].iloc[-2] if len(prices[sym]) >= 2 else np.nan
        
        pct = ((curr - prev) / prev * 100) if (pd.notna(curr) and pd.notna(prev) and prev != 0) else np.nan
        
        sector_name = data.ETF_INFO.get(sym, (sym, ''))[0]
        label = f"{sector_name} ({sym})" 
        
        with metrics_cols[i % len(metrics_cols)]:
            if pd.notna(curr):
                delta_str = f"{pct:.2f}%" if pd.notna(pct) else "N/A"
                delta_color = "normal" if pct >= 0 else "inverse"
                st.metric(label=label, value=f"${curr:.2f}", delta=delta_str, delta_color=delta_color)
            else:
                st.metric(label=label, value="Data Gap", delta="")

    st.markdown("---")


    # --- Institutional Analytics (Moved here) ---
    st.header("Quantitative Risk and Performance Assessment")
    st.markdown("Comprehensive evaluation of risk-adjusted returns (Sharpe, Sortino) and systemic market exposure (Beta, Alpha).")

    # --- Metric Definitions (Improved formatting) ---
    with st.expander("❓ Key Metric Definitions & Interpretation", expanded=False):
        st.markdown("""
            | Metric | Definition | Allocation Interpretation |
            | :--- | :--- | :--- |
            | **Volatility (%)** | Annualized standard deviation of daily returns. | Lower is indicative of superior stability and risk control. |
            | **Max Drawdown (%)** | Largest peak-to-trough decline (capital at risk). | Measures catastrophic risk potential; minimized exposure is preferred. |
            | **Sharpe Ratio** | Excess return per unit of total volatility. | **Primary efficiency metric.** Higher values denote better risk compensation. |
            | **Sortino Ratio**| Excess return per unit of **downside** volatility. | Superior measure to Sharpe, focusing on detrimental risk (negative returns). |
            | **Beta (vs SPY)** | Sensitivity of asset returns to the S&P 500 benchmark. | $\gt 1.0$ suggests **pro-cyclical/aggressive** exposure; $\lt 1.0$ is **defensive**. |
            | **Alpha (Annualized)** | Idiosyncratic return not explained by market exposure (CAPM). | Measures sector/security-specific value generation; positive values confirm superior selection. |
        """)

    # Format the output dataframe for presentation (technical output)
    analytics_df_formatted = analytics_df.copy()
    for col in analytics_df.columns:
        if "Return" in col or "Drawdown" in col or "Volatility" in col:
            analytics_df_formatted[col] = analytics_df_formatted[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        elif "Sharpe" in col or "Sortino" in col or "Beta" in col:
            analytics_df_formatted[col] = analytics_df_formatted[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        elif "Alpha" in col:
            analytics_df_formatted[col] = analytics_df_formatted[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    st.dataframe(analytics_df_formatted, use_container_width=True)

    st.markdown("---")

    # --- Performance Rankings ---
    st.subheader("Time-Series Performance Ranking Matrix")
    st.markdown("Quantitative assessment of sector momentum and risk-adjusted efficiency across varying time horizons.")

    windows = {"1W":5, "1M":21, "3M":63, "6M":126}
    metrics = pd.DataFrame(index=returns.columns)

    for name, w in windows.items():
        try:
            roll = (1 + returns).rolling(window=w).apply(lambda x: x.prod() - 1, raw=True).iloc[-1]
        except Exception:
            # Fallback (keeping original fallback logic)
            roll = returns.tail(w).mean()
        metrics[name] = roll

    # Fetch pre-calculated Volatility and Sharpe from the main analytics run
    metrics["Volatility"] = volatility_series / 100
    metrics["Sharpe"] = sharpe_series

    col_rank_a, col_rank_b = st.columns([1, 4])
    with col_rank_a:
        rank_metric = st.selectbox("Key Ranking Metric", ["3M", "6M", "Sharpe", "Volatility"], index=0, 
                                   help="Select the primary metric for ordinal ranking.")
    
    if rank_metric not in metrics.columns:
        st.error(f"Metric {rank_metric} is non-existent.")
    else:
        ascending = (rank_metric == "Volatility") 
        
        ranked = metrics.sort_values(by=rank_metric, ascending=ascending).reset_index()
        symbol_col_name = ranked.columns[0]
        if symbol_col_name != 'Symbol':
            ranked.rename(columns={symbol_col_name: 'Symbol'}, inplace=True)
        
        ranked["Rank"] = np.arange(1, len(ranked) + 1)
        ranked["Sector"] = ranked["Symbol"].map(lambda x: data.ETF_INFO.get(x, (x, ''))[0])
        
        display_df = ranked[["Rank", "Sector", "Symbol", "1W", "1M", "3M", "6M", "Volatility", "Sharpe"]]
        display_df_formatted = display_df.copy()
        
        for c in ["1W", "1M","3M","6M", "Volatility"]:
             display_df_formatted[c] = (display_df_formatted[c] * 100).apply(lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A")
        
        display_df_formatted["Sharpe"] = display_df_formatted["Sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
        
        with col_rank_b:
             st.dataframe(display_df_formatted.set_index("Rank"), use_container_width=True)

    st.markdown("---")

    # --- Interactive Scatter Plot for Risk/Reward ---
    st.subheader("Efficiency Frontier Mapping: Sharpe vs. Volatility")
    st.markdown("Visualized risk-adjusted return (Y-axis) against absolute risk (X-axis). The **Top-Left Quadrant** represents optimal risk efficiency.")

    # Prepare data for scatter plot
    scatter_data = pd.DataFrame({
        'Symbol': volatility_series.index,
        'Sector': [data.ETF_INFO.get(s, (s,''))[0] for s in volatility_series.index],
        'Volatility': volatility_series,
        'Sharpe': sharpe_series,
    })

    # Calculate 3M Return for color scale
    windows = {"3M": 63}
    metrics_3m = pd.DataFrame(index=returns.columns)
    for name, w in windows.items():
        try:
            roll = (1 + returns).rolling(window=w).apply(lambda x: x.prod() - 1, raw=True).iloc[-1]
        except Exception:
            roll = returns.tail(w).mean()
        metrics_3m[name] = roll

    scatter_data = scatter_data.merge(metrics_3m[['3M']], left_on='Symbol', right_index=True)
    scatter_data['3M_Return'] = scatter_data['3M'] * 100
    scatter_data.drop(columns=['3M'], inplace=True)
    scatter_data.dropna(subset=['Volatility', 'Sharpe', '3M_Return'], inplace=True)


    if not scatter_data.empty:
        fig_scatter = plots.plot_sharpe_volatility_scatter(scatter_data)
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("Insufficient data to compute and map the Efficiency Frontier scatter plot.")
    
    st.markdown("---")

    # --- Cumulative Performance ---
    st.subheader("Historical Trajectory: Cumulative Price Return")
    st.markdown("Visual representation of capital growth relative to initial price point over the entire time series.")

    preferred_defaults = ["XLK", BENCHMARK_SYMBOL, "GLD"]
    dynamic_default = [sym for sym in preferred_defaults if sym in available_symbols]

    sel = st.multiselect("Select Assets for Trajectory Plotting", options=available_symbols, default=dynamic_default, 
                         help="Select the specific symbols to overlay on the cumulative return chart.")

    if sel:
        fig_cum = plots.plot_cumulative(cumulative[sel].dropna(), data.ETF_INFO, title="Cumulative Performance Trajectory")
        st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": True})
    else:
        st.info("A minimum of one asset is required to render the cumulative trajectory plot.")

    st.markdown("---")

    # --- Strategy Backtesting (Moved here for better flow) ---
    st.subheader("Dynamic Sector Allocation Strategy Backtester 🧪")
    st.markdown("Simulate a tactical rotation strategy based on quantifiable momentum signals and trend filtering.")

    # Helper function (Keep logic as is)
    def run_strategy(prices_df, returns_df, top_n, rebalance_window, indicator_type):
        # ... [Keep run_strategy logic exactly as is] ...
        sim_df = returns_df.copy()
        weights = pd.DataFrame(0.0, index=sim_df.index, columns=sim_df.columns)
        lookback_mom = 63  
        lookback_sma = 200 
        start_offset = max(lookback_mom, lookback_sma)
        
        if len(sim_df.index) < start_offset: return pd.Series(), pd.Series() 

        for i in range(start_offset, len(sim_df.index), rebalance_window):
            date = sim_df.index[i]
            
            lookback_prices_full = prices_df.loc[:date].iloc[-start_offset-1:]
            
            if lookback_prices_full.shape[0] < start_offset + 1: continue
                
            sma_200 = lookback_prices_full.iloc[-lookback_sma-1:-1].mean()
            current_prices = lookback_prices_full.iloc[-1]
            trend_filter = (current_prices > sma_200)

            momentum_start_price = lookback_prices_full.iloc[-lookback_mom-1]
            momentum_end_price = lookback_prices_full.iloc[-1]
            lookback_momentum = (momentum_end_price / momentum_start_price) - 1

            tradable_momentum = lookback_momentum.drop(BENCHMARK_SYMBOL, errors='ignore')
            
            if indicator_type == "Dual-Momentum (SMA 200 Filter)":
                tradable_trend_filter = trend_filter.drop(BENCHMARK_SYMBOL, errors='ignore')
                filtered_momentum = tradable_momentum[tradable_trend_filter]
                
                if not filtered_momentum.empty:
                    selected_tickers = filtered_momentum.sort_values(ascending=False).head(top_n).index
                else:
                    selected_tickers = []
            else: 
                selected_tickers = tradable_momentum.sort_values(ascending=False).head(top_n).index
            
            end_index = min(i + rebalance_window, len(sim_df.index))
            weights.loc[sim_df.index[i]:sim_df.index[end_index-1], selected_tickers] = 1.0 / top_n

        strategy_returns = (weights * sim_df).sum(axis=1)
        strategy_cumulative = (1 + strategy_returns).cumprod().fillna(1.0)
        
        benchmark_cum = cumulative[BENCHMARK_SYMBOL].loc[strategy_cumulative.index[0]:].copy()

        if not strategy_cumulative.empty and strategy_cumulative.iloc[0] != 0:
            strategy_cumulative = strategy_cumulative / strategy_cumulative.iloc[0] * benchmark_cum.iloc[0]
        
        return strategy_cumulative, strategy_returns

    # Backtesting UI (Cleaned up layout and terminology)
    tradable_symbols = [s for s in available_symbols if s.startswith("XL")]

    col_strat_a, col_strat_b, col_strat_c = st.columns([1, 1, 1])

    with col_strat_a:
        top_n = st.number_input("Portfolio Concentration (Top N Assets)", min_value=1, max_value=len(tradable_symbols), value=3, step=1, 
                                 help="The number of highest-ranking sectors held in the tactical portfolio.")
    with col_strat_b:
        rebalance_freq_options = ["Monthly (21 trading days)", "Quarterly (63 trading days)"]
        rebalance_freq = st.selectbox("Reallocation Cadence", rebalance_freq_options, index=0, 
                                     help="Defines the interval for re-ranking and tactical portfolio adjustment.")
        rebalance_window = 21 if "Monthly" in rebalance_freq else 63
    with col_strat_c:
        indicator_type = st.selectbox("Allocation Signal Type", 
                                      ["3-Month Relative Momentum (63-Day)", "Dual-Momentum (SMA 200 Trend Filter)"], 
                                      index=1, 
                                      help="The core quantitative metric used to determine sector selection.")

    if st.button("▶️ Initiate Strategy Simulation & Audit"):
        if top_n > len(tradable_symbols):
            st.error(f"Input Error: Portfolio concentration ({top_n}) exceeds the tradable universe size.")
        else:
            try:
                strat_prices = prices[tradable_symbols + [BENCHMARK_SYMBOL]].dropna(how='all').fillna(method='ffill')
                strat_returns = strat_prices.pct_change().dropna()
                
                strategy_cumulative, strategy_returns = run_strategy(
                    prices_df=strat_prices, 
                    returns_df=strat_returns, 
                    top_n=top_n, 
                    rebalance_window=rebalance_window,
                    indicator_type=indicator_type 
                )
                
                if strategy_cumulative.empty:
                    st.warning("Simulation Warning: Insufficient historical data for the defined lookback period.")
                else:
                    st.subheader("Strategy Performance Audit: Trajectory and Metrics")
                    
                    final_cum = pd.concat({
                        "Tactical Rotation Strategy": strategy_cumulative, 
                        BENCHMARK_SYMBOL: cumulative[BENCHMARK_SYMBOL]
                    }, axis=1).dropna()
                    
                    fig_strat = go.Figure()
                    fig_strat.add_trace(go.Scatter(x=final_cum.index, y=(final_cum["Tactical Rotation Strategy"] - 1.0), mode="lines", name="Tactical Rotation Strategy", line=dict(color='#FFA500', width=3))) # Brighter Orange
                    fig_strat.add_trace(go.Scatter(x=final_cum.index, y=(final_cum[BENCHMARK_SYMBOL] - 1.0), mode="lines", name=f"Passive Benchmark ({BENCHMARK_SYMBOL})", line=dict(color='#1e90ff', dash='dash'))) # DodgerBlue & Dashed
                    
                    fig_strat.update_layout(
                        title=f"Strategy Cumulative Performance vs. Benchmark ({indicator_type})",
                        xaxis_title="Simulation Period", 
                        yaxis_title="Normalized Cumulative Growth (%)", 
                        height=500,
                        yaxis=dict(tickformat=".0%"), 
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
                    
                    # --- KPI Metrics (Professionalized) ---
                    strat_final_return = (strategy_cumulative.iloc[-1] / strategy_cumulative.iloc[0]) - 1
                    strat_vol = strategy_returns.std() * np.sqrt(252)
                    strat_max_dd = (strategy_cumulative / strategy_cumulative.cummax() - 1.0).min()
                    
                    bm_row = analytics_df.loc[BENCHMARK_SYMBOL]
                    bm_final_return = cumulative[BENCHMARK_SYMBOL].iloc[-1] - 1.0
                    bm_vol = bm_row["Volatility (%)"] / 100
                    bm_max_dd = bm_row["Max Drawdown (%)"] / 100

                    kpi_cols = st.columns(3)
                    
                    with kpi_cols[0]:
                        st.metric("Total Compounded Return", 
                                     f"{strat_final_return:.2%}", 
                                     delta=f"Alpha vs BM: {(strat_final_return - bm_final_return):.2%}")
                    
                    with kpi_cols[1]:
                        st.metric("Annualized Volatility (Risk)", 
                                     f"{strat_vol:.2%}", 
                                     delta=f"Delta vs BM: {(strat_vol - bm_vol):.2%}",
                                     delta_color="inverse") 
            
                    with kpi_cols[2]:
                        st.metric("Maximum Capital Drawdown", 
                                     f"{strat_max_dd:.2%}", 
                                     delta=f"Delta vs BM: {(strat_max_dd - bm_max_dd):.2%}",
                                     delta_color="inverse") 
        
            except Exception as e:
                st.error(f"Backtesting Protocol Failure: {e}. Review data integrity and lookback parameters.")


with tab2: # Previously tab3
    # --- Factor Exposure Analysis ---
    st.header("Proprietary Factor Exposure & Style Drift Analysis")
    st.markdown("Mapping current sector positioning to systemic market factors (Growth/Value, Sensitivity, Cyclicality) for macro-economic alignment.")
    
    factor_data = {t: data.FACTOR_EXPOSURES.get(t, {"Style": "N/A", "Sensitivity": "N/A", "Cyclicality": "N/A"}) 
                   for t in available_symbols}
    factor_df = pd.DataFrame.from_dict(factor_data, orient='index')
    factor_df.index.name = "Symbol"
    factor_df.insert(0, "Sector", factor_df.index.map(lambda t: data.ETF_INFO.get(t, (t,''))[0]))

    st.dataframe(factor_df, use_container_width=True)
    
    st.markdown("---")

    # --- Sector News (Sentiment Analysis Proxy) ---
    st.header("Curated External Sentiment Feeds 📰")
    st.markdown("Aggregated top headlines providing qualitative sentiment insight into selected sector dynamics.")
    
    if not _api_key:
        st.warning("NewsAPI Authentication Required: Please input the API key in the sidebar configuration.")
    else:
        news_sector_options = {k: v[0] for k, v in data.ETF_INFO.items() if k.startswith("XL")}
        
        sector_choice = st.selectbox("Select Sector for Feed Aggregation", 
                                     options=list(news_sector_options.values()), 
                                     index=0,
                                     help="Retrieves top 5 high-impact headlines via NewsAPI for granular sector sentiment tracking.")
        
        news_items = get_sector_news_cached(sector_choice, _api_key)
        
        if news_items:
            for n in news_items:
                st.markdown(f"**[{n['title']}]({n['url']})** — *{n['source']}*")
        else:
            st.info(f"No recent, relevant news items found for '{sector_choice}' or external feed access quota has been reached.")

st.markdown("---")
st.caption(f"System Status: Data Pipeline Last Synchronized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC-5)")
