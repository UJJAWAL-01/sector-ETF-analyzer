# src/plots.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data import ETF_INFO

# Define RRG Quadrant Colors
RRG_COLORS = {
    'Leading': '#10b981',  # Green
    'Weakening': '#f59e0b', # Yellow/Orange
    'Lagging': '#ef4444',   # Red
    'Improving': '#3b82f6', # Blue
}

def plot_rrg(rrg_df: pd.DataFrame):
    """
    Generates an interactive Relative Rotation Graph (RRG) showing historical trails.
    The input DataFrame should contain columns like 'TICKER_RS' and 'TICKER_Mom'.
    """
    if rrg_df.empty:
        return go.Figure().update_layout(title="RRG Chart: No Data")
        
    # Get the latest point for the main scatter plot markers
    latest_points = rrg_df.iloc[-1].to_frame().T
    
    # Get all unique tickers from the column names (e.g., 'XLK_RS' -> 'XLK')
    tickers = sorted(list(set(c.split('_')[0] for c in rrg_df.columns)))

    fig = go.Figure()
    
    # 1. Draw the center lines (quadrants)
    fig.add_shape(type="line", x0=0, y0=-5, x1=0, y1=5, line=dict(color="#6b7280", width=1))
    fig.add_shape(type="line", x0=-5, y0=0, x1=5, y1=0, line=dict(color="#6b7280", width=1))
    
    # Determine the boundaries based on the data range
    x_max = rrg_df.filter(like='_RS').abs().max().max() * 1.1 or 2
    y_max = rrg_df.filter(like='_Mom').abs().max().max() * 1.1 or 2
    max_range = max(x_max, y_max, 2)
    
    # Define Quadrant Labels (for annotation)
    annotations = [
        # Leading (RS+, Mom+)
        {'x': max_range * 0.9, 'y': max_range * 0.9, 'text': 'Leading', 'showarrow': False, 'font': {'color': RRG_COLORS['Leading'], 'size': 14}},
        # Weakening (RS+, Mom-)
        {'x': max_range * 0.9, 'y': -max_range * 0.9, 'text': 'Weakening', 'showarrow': False, 'font': {'color': RRG_COLORS['Weakening'], 'size': 14}},
        # Lagging (RS-, Mom-)
        {'x': -max_range * 0.9, 'y': -max_range * 0.9, 'text': 'Lagging', 'showarrow': False, 'font': {'color': RRG_COLORS['Lagging'], 'size': 14}},
        # Improving (RS-, Mom+)
        {'x': -max_range * 0.9, 'y': max_range * 0.9, 'text': 'Improving', 'showarrow': False, 'font': {'color': RRG_COLORS['Improving'], 'size': 14}},
    ]
    
    fig.update_layout(annotations=annotations)


    # Function to determine quadrant color based on Z-scores
    def get_quadrant_color(rs, mom):
        if rs >= 0 and mom >= 0: return RRG_COLORS['Leading']
        if rs >= 0 and mom < 0: return RRG_COLORS['Weakening']
        if rs < 0 and mom < 0: return RRG_COLORS['Lagging']
        if rs < 0 and mom >= 0: return RRG_COLORS['Improving']
        return '#ffffff'

    # 2. Draw Trails (The path of the asset)
    for ticker in tickers:
        rs_col = f"{ticker}_RS"
        mom_col = f"{ticker}_Mom"
        
        # Get the full trail data for the ticker
        trail_df = rrg_df[[rs_col, mom_col]].copy().rename(columns={rs_col: 'RS', mom_col: 'Mom'})
        trail_df['Date'] = trail_df.index.strftime('%Y-%m-%d')
        
        # Determine the latest quadrant color for the scatter marker
        latest_rs = latest_points.get(rs_col, pd.Series([0])).iloc[0]
        latest_mom = latest_points.get(mom_col, pd.Series([0])).iloc[0]
        quadrant_color = get_quadrant_color(latest_rs, latest_mom)
        
        # Add the trail (line trace)
        fig.add_trace(go.Scatter(
            x=trail_df['RS'],
            y=trail_df['Mom'],
            mode='lines',
            line=dict(color=quadrant_color, width=1.5),
            name=f"{ticker} Trail",
            hoverinfo='skip', # Hide hover for the trail itself
            opacity=0.5,
            showlegend=False
        ))
        
        # Add the latest point (scatter marker)
        fig.add_trace(go.Scatter(
            x=[latest_rs],
            y=[latest_mom],
            mode='markers+text',
            marker=dict(size=12, color=quadrant_color, line=dict(width=1, color='white')),
            name=f"{ticker} ({ETF_INFO.get(ticker, (ticker, ''))[0]})",
            text=[ticker],
            textposition="top center",
            textfont=dict(size=10, color='white'),
            customdata=[[f"RS: {latest_rs:.2f}", f"Mom: {latest_mom:.2f}"]],
            hovertemplate=
                f'<b>{ticker} ({ETF_INFO.get(ticker, (ticker, ""))}[0])</b><br>' +
                'RS: %{x:.2f}<br>' +
                'Momentum: %{y:.2f}<extra></extra>'
        ))

    # 3. Final layout configuration
    fig.update_layout(
        title='Relative Rotation Graph (RRG)',
        xaxis=dict(title='Relative Strength (RS Index)', zeroline=False, range=[-max_range, max_range]),
        yaxis=dict(title='Momentum (Mom Index)', zeroline=False, range=[-max_range, max_range]),
        height=600,
        showlegend=True,
        template="plotly_dark",
        hovermode="closest",
        plot_bgcolor="#0d1117", # Match Streamlit dark theme
        paper_bgcolor="#0d1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_cumulative(cumulative_df: pd.DataFrame, etf_info_map, title="Cumulative Performance"):
    """Generates a cumulative return plot."""
    # Convert cumulative returns to percentage growth (starting at 0%)
    plot_df = (cumulative_df / cumulative_df.iloc[0]) - 1.0
    plot_df = plot_df * 100 # Convert to percentage

    # Use Plotly Express for a simple, interactive line plot
    fig = px.line(
        plot_df,
        x=plot_df.index,
        y=plot_df.columns,
        title=title,
        labels={"value": "Cumulative Return (%)", "index": "Date", "variable": "Symbol"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Enhance layout
    fig.update_layout(
        height=500,
        yaxis=dict(tickformat=".0f", title="Cumulative Return (%)"),
        xaxis_title="Date",
        hovermode="x unified"
    )

    # Rename legends using ETF_INFO
    for trace in fig.data:
        symbol = trace.name
        sector_name = etf_info_map.get(symbol, (symbol, ""))[0]
        trace.name = f"{symbol} ({sector_name})"

    return fig


def plot_sharpe_volatility_scatter(scatter_data: pd.DataFrame):
    """
    Generates a scatter plot of Sharpe Ratio (Y) vs. Volatility (X).
    """
    if scatter_data.empty:
        return go.Figure().update_layout(title="Scatter Plot: No Data")

    # Use Plotly Express for the scatter plot
    fig = px.scatter(
        scatter_data,
        x='Volatility',
        y='Sharpe',
        color='3M_Return',  # Color by 3M return (momentum)
        size='Volatility',  # Size by volatility for emphasis
        hover_name='Sector',
        hover_data={'Symbol': True, 'Volatility': ':.2f', 'Sharpe': ':.2f', '3M_Return': ':.2f%'},
        template="plotly_dark",
        color_continuous_scale=px.colors.diverging.RdYlGn # Red/Yellow/Green scale for return
    )

    # Enhance layout and axes
    fig.update_layout(
        title="Sharpe Ratio vs. Volatility (Risk/Reward)",
        xaxis_title="Annualized Volatility (%) - Lower is better",
        yaxis_title="Sharpe Ratio - Higher is better",
        height=550,
        hovermode="closest",
        coloraxis_colorbar=dict(title="3M Return (%)")
    )
    
    # Add annotation for the "ideal" quadrant
    fig.add_annotation(
        x=scatter_data['Volatility'].min() * 1.1, 
        y=scatter_data['Sharpe'].max() * 0.9,
        text="Optimal Risk/Reward (High Sharpe, Low Volatility)",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color="#10b981", size=14)
    )

    return fig
