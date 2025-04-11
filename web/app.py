import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta
from loguru import logger
import time
from dotenv import load_dotenv
from polygon import RESTClient
from polygon.rest.models import Agg
import json
from pathlib import Path
from scipy.stats import norm

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants for rate limiting
RATE_LIMIT_PER_MINUTE = 5  # Adjust based on your Polygon.io plan
CACHE_EXPIRY_HOURS = 24
API_USAGE_FILE = "cache/api_usage.json"

# Try to import the MPCOptionPricing class, but provide a fallback if it fails
try:
    from src.mpc_option_pricing import MPCOptionPricing
    has_mpc_pricer = True
except ImportError:
    has_mpc_pricer = False
    st.warning("MPCOptionPricing module not found. Using simplified calculations.")

# Define pricing functions for visualizations
def black_scholes_option_price(S0, K, r, sigma, T, option_type='call'):
    """Calculate Black-Scholes option price with consistent implementation"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if has_mpc_pricer:
        try:
            mpc_pricer = MPCOptionPricing()
            return mpc_pricer.black_scholes_option_price(S0, K, r, sigma, T)
        except Exception as e:
            st.error(f"Error in Black-Scholes calculation: {str(e)}")
            # Fallback to local calculation
            pass
    
    # Black-Scholes calculation with consistent parameters
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

def monte_carlo_option_price(S0, K, r, sigma, T, option_type='call', n_paths=10000, n_steps=100):
    """Calculate Monte Carlo option price with consistent implementation"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if has_mpc_pricer:
        try:
            mpc_pricer = MPCOptionPricing()
            return mpc_pricer.monte_carlo_option_price(S0, K, r, sigma, T, n_paths, n_steps)
        except Exception as e:
            st.error(f"Error in Monte Carlo calculation: {str(e)}")
            # Fallback to local calculation
            pass
    
    # Monte Carlo calculation with consistent parameters
    dt = T/n_steps
    paths = np.zeros((n_paths, n_steps+1))
    paths[:, 0] = S0
    
    # Generate all random numbers at once for better performance
    z = np.random.standard_normal((n_paths, n_steps))
    
    # Calculate drift and volatility terms
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # Calculate paths using vectorized operations
    for t in range(1, n_steps+1):
        paths[:, t] = paths[:, t-1] * np.exp(drift + vol * z[:, t-1])
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[:, -1], 0)
    
    # Calculate option price with discounting
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price

def binomial_option_price(S0, K, r, sigma, T, option_type='call', n_steps=100):
    """Calculate Binomial option price with consistent implementation"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if has_mpc_pricer:
        try:
            mpc_pricer = MPCOptionPricing()
            return mpc_pricer.binomial_option_price(S0, K, r, sigma, T, n_steps)
        except Exception as e:
            st.error(f"Error in Binomial calculation: {str(e)}")
            # Fallback to local calculation
            pass
    
    # Binomial calculation with consistent parameters
    dt = T/n_steps
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    
    # Build price tree
    price_tree = np.zeros((n_steps+1, n_steps+1))
    price_tree[0, 0] = S0
    
    for i in range(1, n_steps+1):
        price_tree[i, 0] = price_tree[i-1, 0]*u
        for j in range(1, i+1):
            price_tree[i, j] = price_tree[i-1, j-1]*d
    
    # Calculate option values
    option_values = np.zeros((n_steps+1, n_steps+1))
    for j in range(n_steps+1):
        if option_type == 'call':
            option_values[n_steps, j] = max(price_tree[n_steps, j] - K, 0)
        else:
            option_values[n_steps, j] = max(K - price_tree[n_steps, j], 0)
    
    # Backward induction
    for i in range(n_steps-1, -1, -1):
        for j in range(i+1):
            option_values[i, j] = np.exp(-r*dt)*(p*option_values[i+1, j] + (1-p)*option_values[i+1, j+1])
    
    return option_values[0, 0]

def find_implied_volatility(market_price, S0, K, r, T, option_type='call', tolerance=1e-5, max_iter=100):
    """Find implied volatility using Newton-Raphson method"""
    sigma = 0.5  # Initial guess
    
    for i in range(max_iter):
        price = black_scholes_option_price(S0, K, r, sigma, T, option_type)
        diff = price - market_price
        
        if abs(diff) < tolerance:
            return sigma
        
        # Calculate vega (derivative of price with respect to volatility)
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        
        # Update sigma
        sigma = sigma - diff/vega
        
        # Ensure sigma stays positive
        sigma = max(sigma, 0.001)
    
    return sigma  # Return best estimate if convergence fails

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self._load_usage()

    def _load_usage(self):
        """Load API usage history from file"""
        try:
            if os.path.exists(API_USAGE_FILE):
                with open(API_USAGE_FILE, 'r') as f:
                    data = json.load(f)
                    self.calls = [datetime.fromisoformat(t) for t in data['calls']]
        except Exception as e:
            logger.warning(f"Error loading API usage history: {e}")
            self.calls = []

    def _save_usage(self):
        """Save API usage history to file"""
        try:
            os.makedirs(os.path.dirname(API_USAGE_FILE), exist_ok=True)
            with open(API_USAGE_FILE, 'w') as f:
                json.dump({
                    'calls': [t.isoformat() for t in self.calls],
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Error saving API usage history: {e}")

    def wait_if_needed(self):
        """Wait if we've exceeded our rate limit"""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until the oldest call is 1 minute old
            sleep_time = 60 - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.calls = self.calls[1:]
        
        self.calls.append(now)
        self._save_usage()

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)

def get_cached_data(ticker: str) -> dict:
    """Get cached data if it exists and is not expired"""
    cache_dir = "cache"
    cache_file = os.path.join(cache_dir, f"{ticker}.csv")
    cache_meta_file = os.path.join(cache_dir, f"{ticker}.meta.json")
    
    if not (os.path.exists(cache_file) and os.path.exists(cache_meta_file)):
        return None
    
    try:
        # Check if cache is expired
        with open(cache_meta_file, 'r') as f:
            meta = json.load(f)
            cache_time = datetime.fromisoformat(meta['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                logger.info(f"Cache expired for {ticker}")
                return None
        
        # Load cached data
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        with open(cache_meta_file, 'r') as f:
            meta = json.load(f)
        
        logger.info(f"Using cached data for {ticker}")
        return {
            'hist': data,
            'S0': float(data['Close'].iloc[-1]),
            'sigma': meta['sigma'],
            'dates': data.index,
            'prices': data['Close'].values,
            'returns': data['Close'].pct_change().dropna(),
            'info': meta['info']
        }
    except Exception as e:
        logger.warning(f"Error loading cached data for {ticker}: {e}")
        return None

def save_to_cache(ticker: str, data: dict):
    """Save data to cache"""
    try:
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save DataFrame
        cache_file = os.path.join(cache_dir, f"{ticker}.csv")
        data['hist'].to_csv(cache_file)
        
        # Save metadata
        cache_meta_file = os.path.join(cache_dir, f"{ticker}.meta.json")
        with open(cache_meta_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'sigma': float(data['sigma']),
                'info': data['info']
            }, f)
        
        logger.info(f"Cached data for {ticker}")
    except Exception as e:
        logger.warning(f"Error caching data for {ticker}: {e}")

def fetch_stock_data(ticker: str, max_retries=3, retry_delay=2):
    """Fetch stock data from Polygon.io with retries."""
    # Check cache first
    cached_data = get_cached_data(ticker)
    if cached_data:
        return cached_data

    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        st.error("Polygon API key not found. Please set POLYGON_API_KEY environment variable.")
        return None
    
    # Log API key length for debugging (don't log the actual key)
    logger.info(f"API key length: {len(api_key)}")
    
    for attempt in range(max_retries):
        try:
            # Wait if we need to respect rate limits
            rate_limiter.wait_if_needed()
            
            # Initialize Polygon client
            client = RESTClient(api_key)
            
            # Get end date (today) and start date (1 year ago)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Ensure we're not requesting future data
            if end_date > datetime.now():
                end_date = datetime.now()
            
            logger.info(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            # First verify the ticker exists
            try:
                rate_limiter.wait_if_needed()
                ticker_details = client.get_ticker_details(ticker)
                if not ticker_details:
                    st.error(f"Ticker {ticker} not found. Please verify the symbol.")
                    return None
                logger.info(f"Verified ticker {ticker} exists")
            except Exception as e:
                if "NOT_FOUND" in str(e):
                    st.error(f"Ticker {ticker} not found. Please verify the symbol.")
                    return None
                logger.warning(f"Could not verify ticker {ticker}: {e}")
            
            # Fetch daily aggregates
            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    adjusted=True
                )
                
                if not aggs:
                    logger.warning(f"No data returned from Polygon API for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    st.error(f"No historical data found for {ticker}. Please verify the ticker symbol.")
                    return None
                
                logger.info(f"Retrieved {len(aggs)} days of data for {ticker}")
                
                # Convert to DataFrame
                data = pd.DataFrame([{
                    'Date': datetime.fromtimestamp(a.timestamp/1000),
                    'Open': float(a.open),
                    'High': float(a.high),
                    'Low': float(a.low),
                    'Close': float(a.close),
                    'Volume': float(a.volume)
                } for a in aggs])
                
                if data.empty:
                    logger.warning(f"Empty DataFrame created for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    st.error(f"Could not process data for {ticker}")
                    return None
                
                # Set Date as index
                data.set_index('Date', inplace=True)
                data.sort_index(ascending=True, inplace=True)
                
                # Calculate volatility using the last 252 trading days
                daily_returns = data['Close'].pct_change().dropna()
                if len(daily_returns) < 2:
                    logger.warning(f"Insufficient data for volatility calculation for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    st.error(f"Insufficient data for volatility calculation for {ticker}")
                    return None
                
                # Use the most recent closing price
                S0 = float(data['Close'].iloc[-1])
                
                # Calculate annualized volatility
                sigma = daily_returns.std() * np.sqrt(252)
                
                # Get company details
                try:
                    rate_limiter.wait_if_needed()  # Wait before making another API call
                    ticker_details = client.get_ticker_details(ticker)
                    info = {
                        'name': ticker_details.name,
                        'market': ticker_details.market,
                        'locale': ticker_details.locale,
                        'currency': ticker_details.currency_name,
                        'description': ticker_details.description
                    }
                    logger.info(f"Retrieved company details for {ticker}: {info['name']}")
                except Exception as e:
                    logger.warning(f"Could not fetch ticker details for {ticker}: {e}")
                    info = {'name': ticker}
                
                result = {
                    'hist': data,
                    'S0': S0,
                    'sigma': sigma,
                    'dates': data.index,
                    'prices': data['Close'].values,
                    'returns': daily_returns,
                    'info': info
                }
                
                # Cache the result
                save_to_cache(ticker, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing data for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                st.error(f"Error processing data for {ticker}: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                return None
    
    return None

# Set page config
st.set_page_config(
    page_title="Sen1 & Sen2 Option Pricing",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        max-width: 100%;
        padding: 2rem 2rem;
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ff9d;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #00ff9d;
        color: #0e1117;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 700;
    }
    .metric-container {
        background-color: #1a1c23;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-label {
        color: #fafafa;
        font-family: 'Arial', sans-serif;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #00ff9d;
        font-family: 'Arial', sans-serif;
        font-size: 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Sen1 Option Pricing Dashboard")

# Sidebar with two sections
st.sidebar.header("Sen1 Input Parameters")
ticker1 = st.sidebar.text_input("Stock Ticker", "AAPL", key="ticker1")
S0_1 = st.sidebar.number_input("Current Stock Price", value=100.0, step=1.0, key="S0_1")
K_1 = st.sidebar.number_input("Strike Price", value=100.0, step=1.0, key="K_1")
r_1 = st.sidebar.number_input("Risk-free Rate", value=0.05, step=0.01, format="%.2f", key="r_1")
sigma_1 = st.sidebar.number_input("Volatility", value=0.2, step=0.01, format="%.2f", key="sigma_1")
T_1 = st.sidebar.number_input("Time to Maturity (years)", value=1.0, step=0.1, format="%.1f", key="T_1")

# Add a divider in sidebar
st.sidebar.markdown("<hr style='border: 1px solid #2a2c33; margin: 2rem 0;'>", unsafe_allow_html=True)

# Sen2 parameters
st.sidebar.header("Sen2 Input Parameters")
ticker2 = st.sidebar.text_input("Stock Ticker", "AAPL", key="ticker2")
S0_2 = st.sidebar.number_input("Current Stock Price", value=100.0, step=1.0, key="S0_2")
K_2 = st.sidebar.number_input("Strike Price", value=100.0, step=1.0, key="K_2")
r_2 = st.sidebar.number_input("Risk-free Rate", value=0.05, step=0.01, format="%.2f", key="r_2")
sigma_2 = st.sidebar.number_input("Volatility", value=0.2, step=0.01, format="%.2f", key="sigma_2")
T_2 = st.sidebar.number_input("Time to Maturity (years)", value=1.0, step=0.1, format="%.1f", key="T_2")

# Fetch stock data for Sen1
stock_data1 = fetch_stock_data(ticker1)

if stock_data1:
    # Update Sen1 values based on fetched data
    S0_1 = float(stock_data1['S0'])
    K_1 = float(stock_data1['S0'])
    sigma_1 = float(stock_data1['sigma'])

# Fetch stock data for Sen2
stock_data2 = fetch_stock_data(ticker2)

if stock_data2:
    # Update Sen2 values based on fetched data
    S0_2 = float(stock_data2['S0'])
    K_2 = float(stock_data2['S0'])
    sigma_2 = float(stock_data2['sigma'])

# Create tabs for Sen1
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Price Analysis", "Volatility Analysis", "Payoff Analysis",
    "Historical Analysis", "Model Comparison"
])

def plot_volatility_smile(S0, K, r, T, option_type='call'):
    """Plot implied volatility smile"""
    # Create range of strike prices
    K_range = np.linspace(K * 0.8, K * 1.2, 50)
    sigma_range = np.linspace(0.1, 0.5, 50)
    
    # Calculate implied volatilities
    implied_vols = []
    for k in K_range:
        try:
            # Find sigma that matches market price
            market_price = black_scholes_option_price(S0, k, r, 0.2, T, option_type)
            implied_vol = find_implied_volatility(market_price, S0, k, r, T, option_type)
            implied_vols.append(implied_vol)
        except:
            implied_vols.append(np.nan)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K_range,
        y=implied_vols,
        name='Implied Volatility',
        line=dict(color='#00ff9d')
    ))
    
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

def plot_payoff_distribution(S0, K, r, sigma, T, option_type='call'):
    """Plot probability distribution of option payoffs"""
    # Generate price paths
    n_paths = 1000
    n_steps = 100
    dt = T / n_steps
    
    # Generate random walks
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[:, -1], 0)
    
    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=payoffs,
        nbinsx=50,
        name='Payoff Distribution',
        marker_color='#00ff9d'
    ))
    
    fig.update_layout(
        title='Option Payoff Distribution',
        xaxis_title='Payoff',
        yaxis_title='Frequency',
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

def plot_price_comparison(S0, K, r, sigma, T, n_paths=10000, n_steps=100):
    """Plot option prices across different stock prices using consistent parameters."""
    # Generate stock price range
    price_range = np.linspace(S0 * 0.5, S0 * 1.5, 50)
    
    # Calculate prices for each model
    mc_prices = []
    bs_prices = []
    bin_prices = []
    
    for price in price_range:
        # Monte Carlo price with consistent parameters
        mc_price = monte_carlo_option_price(price, K, r, sigma, T, 'call', n_paths, n_steps)
        mc_prices.append(mc_price)
        
        # Black-Scholes price
        bs_price = black_scholes_option_price(price, K, r, sigma, T, 'call')
        bs_prices.append(bs_price)
        
        # Binomial price with consistent parameters
        bin_price = binomial_option_price(price, K, r, sigma, T, 'call', n_steps)
        bin_prices.append(bin_price)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=price_range,
        y=mc_prices,
        name='Monte Carlo',
        line=dict(color='#00ff9d', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_range,
        y=bs_prices,
        name='Black-Scholes',
        line=dict(color='#ff9d00', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=price_range,
        y=bin_prices,
        name='Binomial',
        line=dict(color='#9d00ff', width=2)
    ))
    
    # Add vertical line for current stock price
    fig.add_vline(
        x=S0,
        line_dash="dash",
        line_color="white",
        annotation_text="Current Price",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title='Option Price Comparison',
        xaxis_title='Stock Price',
        yaxis_title='Option Price',
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def plot_historical_volatility(data, window=30):
    """Plot historical volatility analysis"""
    # Calculate rolling volatility
    returns = data['Close'].pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='#00ff9d')
    ))
    
    # Add volatility data
    fig.add_trace(go.Scatter(
        x=data.index[window:],
        y=rolling_vol[window:],
        name=f'{window}-day Rolling Volatility',
        line=dict(color='#00ccff'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Historical Price and Volatility',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volatility',
            overlaying='y',
            side='right'
        ),
        template='plotly_dark',
        paper_bgcolor='#1a1c23',
        plot_bgcolor='#1a1c23',
        font=dict(color='#fafafa'),
        xaxis=dict(gridcolor='#2a2c33'),
        yaxis=dict(gridcolor='#2a2c33')
    )
    
    return fig

# Function to calculate option prices with consistent parameters
def calculate_option_prices(S0, K, r, sigma, T, n_paths=10000, n_steps=100):
    """Calculate option prices using all models with consistent parameters"""
    # Calculate prices with consistent parameters
    mc_price = monte_carlo_option_price(S0, K, r, sigma, T, 'call', n_paths, n_steps)
    bs_price = black_scholes_option_price(S0, K, r, sigma, T, 'call')
    bin_price = binomial_option_price(S0, K, r, sigma, T, 'call', n_steps)
    
    return mc_price, bs_price, bin_price

# Display content in Sen1 tabs
with tab1:
    st.header("Price Analysis")
    
    # Calculate prices with consistent parameters
    mc_price, bs_price, bin_price = calculate_option_prices(S0_1, K_1, r_1, sigma_1, T_1)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Monte Carlo</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${mc_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Black-Scholes</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${bs_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Binomial</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${bin_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Price comparison chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    fig = plot_price_comparison(S0_1, K_1, r_1, sigma_1, T_1)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.header("Volatility Analysis")
    fig2 = plot_volatility_smile(S0_1, K_1, r_1, T_1)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Payoff Analysis")
    fig3 = plot_payoff_distribution(S0_1, K_1, r_1, sigma_1, T_1)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.header("Historical Analysis")
    if stock_data1:
        # Create price history plot
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=stock_data1['dates'],
                y=stock_data1['prices'],
                name='Price History',
                line=dict(color='#00ff9d')
            )
        )
        
        fig1.update_layout(
            title=f'Price History - {stock_data1["info"]["name"]}',
            yaxis_title='Price',
            height=400,
            template='plotly_dark',
            paper_bgcolor='#1a1c23',
            plot_bgcolor='#1a1c23',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#2a2c33'),
            yaxis=dict(gridcolor='#2a2c33')
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create volatility plot using rolling window
        window = 20  # 20-day rolling window
        rolling_vol = stock_data1['returns'].rolling(window=window).std() * np.sqrt(252)
        
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=stock_data1['dates'][window:],
                y=rolling_vol[window:],
                name='Rolling Volatility',
                line=dict(color='#00ff9d')
            )
        )
        
        fig2.update_layout(
            title=f'Historical Volatility - {stock_data1["info"]["name"]} ({window}-day Rolling)',
            yaxis_title='Volatility',
            height=400,
            template='plotly_dark',
            paper_bgcolor='#1a1c23',
            plot_bgcolor='#1a1c23',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#2a2c33'),
            yaxis=dict(gridcolor='#2a2c33')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No historical data available. Please enter a valid ticker symbol.")

with tab5:
    st.header("Model Comparison")
    fig5 = plot_price_comparison(S0_1, K_1, r_1, sigma_1, T_1)
    st.plotly_chart(fig5, use_container_width=True)

# Add a divider
st.markdown("<hr style='border: 1px solid #2a2c33; margin: 2rem 0;'>", unsafe_allow_html=True)

# Add Sen2 Option Pricing section
st.title("Sen2 Option Pricing Dashboard")

# Create tabs for Sen2
sen2_tab1, sen2_tab2, sen2_tab3, sen2_tab4, sen2_tab5 = st.tabs([
    "Price Analysis", "Volatility Analysis", "Payoff Analysis",
    "Historical Analysis", "Model Comparison"
])

# Display content in Sen2 tabs
with sen2_tab1:
    st.header("Price Analysis")
    
    # Calculate prices with consistent parameters
    mc_price, bs_price, bin_price = calculate_option_prices(S0_2, K_2, r_2, sigma_2, T_2)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Monte Carlo</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${mc_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Black-Scholes</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${bs_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Binomial</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${bin_price:.2f}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Price comparison chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Comparison")
    fig = plot_price_comparison(S0_2, K_2, r_2, sigma_2, T_2)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sen2_tab2:
    st.header("Volatility Analysis")
    fig2 = plot_volatility_smile(S0_2, K_2, r_2, T_2)
    st.plotly_chart(fig2, use_container_width=True)

with sen2_tab3:
    st.header("Payoff Analysis")
    fig3 = plot_payoff_distribution(S0_2, K_2, r_2, sigma_2, T_2)
    st.plotly_chart(fig3, use_container_width=True)

with sen2_tab4:
    st.header("Historical Analysis")
    if stock_data2:
        # Create price history plot
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=stock_data2['dates'],
                y=stock_data2['prices'],
                name='Price History',
                line=dict(color='#00ff9d')
            )
        )
        
        fig1.update_layout(
            title=f'Price History - {stock_data2["info"]["name"]}',
            yaxis_title='Price',
            height=400,
            template='plotly_dark',
            paper_bgcolor='#1a1c23',
            plot_bgcolor='#1a1c23',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#2a2c33'),
            yaxis=dict(gridcolor='#2a2c33')
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create volatility plot using rolling window
        window = 20  # 20-day rolling window
        rolling_vol = stock_data2['returns'].rolling(window=window).std() * np.sqrt(252)
        
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=stock_data2['dates'][window:],
                y=rolling_vol[window:],
                name='Rolling Volatility',
                line=dict(color='#00ff9d')
            )
        )
        
        fig2.update_layout(
            title=f'Historical Volatility - {stock_data2["info"]["name"]} ({window}-day Rolling)',
            yaxis_title='Volatility',
            height=400,
            template='plotly_dark',
            paper_bgcolor='#1a1c23',
            plot_bgcolor='#1a1c23',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#2a2c33'),
            yaxis=dict(gridcolor='#2a2c33')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No historical data available. Please enter a valid ticker symbol.")

with sen2_tab5:
    st.header("Model Comparison")
    fig5 = plot_price_comparison(S0_2, K_2, r_2, sigma_2, T_2)
    st.plotly_chart(fig5, use_container_width=True)

# Remove the main() function since we're not using it anymore
if __name__ == "__main__":
    pass 