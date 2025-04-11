import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import time
import requests

from src.mpc_option_pricing import MPCOptionPricing
from src.risk import risk_manager
from src.logger import logger
from data_cache import StockDataCache

# Set page config with dark theme
st.set_page_config(
    page_title="MPC Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply custom CSS
load_css()

# Get Alpha Vantage API key from environment variable
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')

# Initialize cache
cache = StockDataCache()

def get_stock_data(ticker: str) -> tuple:
    """Get stock data from multiple sources with fallback and caching."""
    max_retries = 3
    retry_delay = 2  # seconds
    
    # Try to get data from cache first
    cached_data = cache.get(ticker)
    if cached_data is not None:
        logger.info(f"Using cached data for {ticker}")
        hist = cached_data
    else:
        def try_yahoo_finance():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty:
                    return hist
                return None
            except Exception as e:
                logger.error(f"Yahoo Finance error: {str(e)}")
                return None
        
        def try_alpha_vantage():
            try:
                # Alpha Vantage API endpoint
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
                response = requests.get(url, timeout=10)
                
                # Check if the response is valid JSON
                try:
                    data = response.json()
                except ValueError as e:
                    logger.error(f"Invalid JSON response from Alpha Vantage: {str(e)}")
                    return None
                
                # Check for API errors
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"Alpha Vantage API note: {data['Note']}")
                    return None
                
                if "Time Series (Daily)" not in data:
                    logger.error(f"Missing time series data in Alpha Vantage response")
                    return None
                
                # Convert to DataFrame
                time_series = data["Time Series (Daily)"]
                if not time_series:
                    logger.error("Empty time series data from Alpha Vantage")
                    return None
                
                try:
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                except Exception as e:
                    logger.error(f"Error converting Alpha Vantage data to DataFrame: {str(e)}")
                    return None
                
                # Rename columns
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                # Convert string values to float
                for col in df.columns:
                    try:
                        df[col] = df[col].astype(float)
                    except Exception as e:
                        logger.error(f"Error converting {col} to float: {str(e)}")
                        return None
                
                # Sort by date
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Get last year of data
                one_year_ago = datetime.now() - timedelta(days=365)
                df = df[df.index >= one_year_ago]
                
                if df.empty:
                    logger.error("No data available for the last year")
                    return None
                    
                return df
                
            except requests.exceptions.Timeout:
                logger.error("Alpha Vantage API request timed out")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Alpha Vantage API request failed: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error with Alpha Vantage: {str(e)}")
                return None
        
        for attempt in range(max_retries):
            try:
                # Try Yahoo Finance first
                hist = try_yahoo_finance()
                
                # If Yahoo Finance fails, try Alpha Vantage
                if hist is None:
                    hist = try_alpha_vantage()
                
                if hist is None or hist.empty:
                    logger.warning(f"No historical data found for ticker {ticker} (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                # Cache the data if we got it successfully
                cache.set(ticker, hist)
                break
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts. Please check the ticker symbol and try again.")
    
    # Validate data
    if 'Close' not in hist.columns:
        logger.warning(f"Missing 'Close' price data for {ticker}")
        raise ValueError(f"Missing 'Close' price data for {ticker}")
    
    # Get the most recent valid closing price
    valid_prices = hist['Close'].dropna()
    if valid_prices.empty:
        logger.warning(f"No valid closing prices found for {ticker}")
        raise ValueError(f"No valid closing prices found for {ticker}")
    
    S0 = valid_prices.iloc[-1]
    
    # Calculate daily returns and volatility
    daily_returns = valid_prices.pct_change().dropna()
    if len(daily_returns) < 2:
        logger.warning(f"Insufficient data for volatility calculation for {ticker}")
        raise ValueError(f"Insufficient data for volatility calculation for {ticker}")
    
    sigma = daily_returns.std() * np.sqrt(252)
    
    r = 0.05  # Risk-free rate
    K = S0    # At-the-money strike
    T = 1.0   # 1 year to maturity
    
    return S0, K, r, sigma, T, hist

def create_price_plot(hist_data: pd.DataFrame) -> go.Figure:
    """Create price history plot."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='OHLC',
            increasing_line_color='#00ff9d',
            decreasing_line_color='#ff4d4d'
        )
    )
    
    fig.update_layout(
        title='Price History',
        yaxis_title='Price',
        height=400,
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

def create_volatility_plot(hist_data: pd.DataFrame) -> go.Figure:
    """Create volatility plot."""
    daily_returns = hist_data['Close'].pct_change()
    rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            name='20-day Rolling Volatility',
            line=dict(color='#00ff9d')
        )
    )
    
    fig.update_layout(
        title='Historical Volatility',
        yaxis_title='Volatility',
        height=400,
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

def create_option_price_comparison(mc_price, bs_price, bin_price) -> go.Figure:
    """Create option price comparison chart."""
    fig = go.Figure()
    
    methods = ['Monte Carlo', 'Black-Scholes', 'Binomial']
    prices = [mc_price, bs_price, bin_price]
    
    fig.add_trace(
        go.Bar(
            x=methods,
            y=prices,
            marker_color=['#00ff9d', '#00ccff', '#ff9d00'],
            text=[f"${price:.2f}" for price in prices],
            textposition='auto',
        )
    )
    
    fig.update_layout(
        title='Option Price Comparison',
        yaxis_title='Price ($)',
        height=400,
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

def main():
    # Header with logo
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>MPC Option Pricing</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #00ff9d;'>Secure Multi-Party Computation for Financial Derivatives</p>", unsafe_allow_html=True)
    
    # Sidebar with r3hpots.com style
    st.sidebar.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    
    try:
        # Get stock data
        S0, K, r, sigma, T, hist_data = get_stock_data(ticker)
        
        # Display market data
        st.sidebar.subheader("Market Data")
        st.sidebar.markdown(f"<div style='background-color: #2a2c33; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p><strong>Current Price:</strong> ${S0:.2f}</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p><strong>Strike Price:</strong> ${K:.2f}</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p><strong>Risk-free Rate:</strong> {r*100:.1f}%</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p><strong>Volatility:</strong> {sigma*100:.1f}%</p>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<p><strong>Time to Maturity:</strong> {T:.1f} years</p>", unsafe_allow_html=True)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Risk Analysis", "Historical Data"])
        
        with tab1:
            st.header("Price Analysis")
            
            # Calculate prices
            mpc_pricer = MPCOptionPricing()
            mpc_pricer.setup()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.subheader("Monte Carlo")
                mc_price = mpc_pricer.monte_carlo_option_price(
                    S0, K, r, sigma, T,
                    n_paths=10000,
                    n_steps=100
                )
                st.markdown(f"<h2 style='color: #00ff9d;'>${mc_price:.2f}</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.subheader("Black-Scholes")
                bs_price = mpc_pricer.black_scholes_option_price(
                    S0, K, r, sigma, T
                )
                st.markdown(f"<h2 style='color: #00ff9d;'>${bs_price:.2f}</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.subheader("Binomial")
                bin_price = mpc_pricer.binomial_option_price(
                    S0, K, r, sigma, T,
                    n_steps=100
                )
                st.markdown(f"<h2 style='color: #00ff9d;'>${bin_price:.2f}</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Price comparison chart
            price_comparison = create_option_price_comparison(mc_price, bs_price, bin_price)
            st.plotly_chart(price_comparison, use_container_width=True)
            
            mpc_pricer.shutdown()
        
        with tab2:
            st.header("Risk Analysis")
            
            # Perform stress testing
            stress_results = risk_manager.stress_test(S0, K, r, sigma, T)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.subheader("Stress Test Results")
                for scenario, results in stress_results.items():
                    st.markdown(f"<div style='background-color: #2a2c33; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: #00ff9d;'>{scenario}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Price:</strong> ${results['price']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>VaR:</strong> ${results['var']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>ES:</strong> ${results['es']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.subheader("Risk Metrics")
                # Add risk metrics visualization
                st.markdown("<p>Risk metrics visualization will be added here.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.header("Historical Data")
            
            # Create plots
            fig1 = create_price_plot(hist_data)
            fig2 = create_volatility_plot(hist_data)
            
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Error in Streamlit app: {str(e)}")

if __name__ == "__main__":
    main() 