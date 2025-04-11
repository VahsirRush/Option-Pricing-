import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def create_output_directory(ticker: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{ticker}_{timestamp}"
    
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    
    return output_dir

def save_market_data(output_dir: str, market_data: dict):
    with open(f"{output_dir}/data/market_data.json", 'w') as f:
        json.dump(market_data, f, indent=4)

def save_option_chain(output_dir: str, option_chain: pd.DataFrame):
    option_chain.to_csv(f"{output_dir}/data/option_chain.csv")

def create_option_comparison_plot(ticker: str, theoretical_prices: dict, option_chain: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Option Prices Comparison', 'Implied Volatility Surface'),
                       vertical_spacing=0.2)
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=[theoretical_prices['monte_carlo']] * len(option_chain),
            name='Monte Carlo',
            line=dict(color='blue', dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=[theoretical_prices['black_scholes']] * len(option_chain),
            name='Black-Scholes',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=[theoretical_prices['binomial']] * len(option_chain),
            name='Binomial',
            line=dict(color='green', dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['lastPrice'],
            name='Market Price',
            mode='markers',
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=option_chain['strike'],
            y=option_chain['impliedVolatility'],
            name='Implied Volatility',
            mode='markers',
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'Option Analysis for {ticker}',
        xaxis_title='Strike Price',
        yaxis_title='Option Price',
        yaxis2_title='Implied Volatility',
        height=800,
        showlegend=True
    )
    
    return fig

def create_price_history_plot(ticker: str, hist_data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='OHLC'
        )
    )
    
    fig.update_layout(
        title=f'Price History for {ticker}',
        yaxis_title='Price',
        height=400
    )
    
    return fig

def create_volatility_plot(ticker: str, hist_data: pd.DataFrame) -> go.Figure:
    daily_returns = hist_data['Close'].pct_change()
    
    rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            name='20-day Rolling Volatility',
            line=dict(color='blue')
        )
    )
    
    fig.update_layout(
        title=f'Historical Volatility for {ticker}',
        yaxis_title='Volatility',
        height=400
    )
    
    return fig

def save_analysis_results(ticker: str, market_data: dict, option_chain: pd.DataFrame, 
                         theoretical_prices: dict, hist_data: pd.DataFrame):
    output_dir = create_output_directory(ticker)
    
    save_market_data(output_dir, market_data)
    
    save_option_chain(output_dir, option_chain)
    
    option_plot = create_option_comparison_plot(ticker, theoretical_prices, option_chain)
    price_plot = create_price_history_plot(ticker, hist_data)
    vol_plot = create_volatility_plot(ticker, hist_data)
    
    option_plot.write_html(f"{output_dir}/plots/option_analysis.html")
    price_plot.write_html(f"{output_dir}/plots/price_history.html")
    vol_plot.write_html(f"{output_dir}/plots/volatility.html")
    
    return output_dir 