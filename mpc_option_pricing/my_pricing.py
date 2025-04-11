from src.mpc_option_pricing import MPCOptionPricing
from src.visualization import save_analysis_results
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import streamlit as st
import plotly.graph_objects as go

def get_stock_data(ticker: str) -> tuple:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    
    S0 = hist['Close'].iloc[-1]
    
    daily_returns = hist['Close'].pct_change().dropna()
    sigma = daily_returns.std() * np.sqrt(252)
    
    r = 0.05
    
    K = S0
    
    T = 1.0
    
    return S0, K, r, sigma, T, hist

def get_option_chain(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    options = stock.option_chain()
    
    calls = options.calls
    puts = options.puts
    
    all_options = pd.concat([calls, puts])
    all_options = all_options.sort_values('strike')
    
    return all_options

async def main():
    os.makedirs("output", exist_ok=True)
    
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()
    
    try:
        print(f"\nFetching data for {ticker}...")
        S0, K, r, sigma, T, hist_data = get_stock_data(ticker)
        
        print("Fetching option chain data...")
        option_chain = get_option_chain(ticker)
        
        market_data = {
            "ticker": ticker,
            "current_price": float(S0),
            "strike_price": float(K),
            "risk_free_rate": float(r),
            "volatility": float(sigma),
            "time_to_maturity": float(T),
            "timestamp": datetime.now().isoformat()
        }
        
        print("\nMarket Data:")
        print("-" * 40)
        print(f"Current Stock Price: ${S0:.2f}")
        print(f"Strike Price:       ${K:.2f}")
        print(f"Risk-free Rate:     {r*100:.1f}%")
        print(f"Volatility:         {sigma*100:.1f}%")
        print(f"Time to Maturity:   {T:.1f} years")
        print("-" * 40)
        
        mpc_pricer = MPCOptionPricing()
        await mpc_pricer.setup()
        
        print("\nCalculating option prices...")
        
        print("\nUsing Monte Carlo simulation...")
        mc_price = await mpc_pricer.monte_carlo_option_price(
            S0, K, r, sigma, T,
            n_paths=10000,
            n_steps=100
        )
        
        print("\nUsing Black-Scholes model...")
        bs_price = await mpc_pricer.black_scholes_option_price(
            S0, K, r, sigma, T
        )
        
        print("\nUsing Binomial model...")
        bin_price = await mpc_pricer.binomial_option_price(
            S0, K, r, sigma, T,
            n_steps=100
        )
        
        print("\nResults:")
        print("-" * 40)
        print(f"Monte Carlo Price:    ${mc_price:.2f}")
        print(f"Black-Scholes Price: ${bs_price:.2f}")
        print(f"Binomial Price:      ${bin_price:.2f}")
        print("-" * 40)
        
        theoretical_prices = {
            'monte_carlo': float(mc_price),
            'black_scholes': float(bs_price),
            'binomial': float(bin_price)
        }
        
        print("\nSaving analysis results...")
        output_dir = save_analysis_results(
            ticker=ticker,
            market_data=market_data,
            option_chain=option_chain,
            theoretical_prices=theoretical_prices,
            hist_data=hist_data
        )
        
        print(f"\nAnalysis results saved in: {output_dir}")
        print("\nDirectory structure:")
        print(f"- {output_dir}/")
        print(f"  ├── data/")
        print(f"  │   ├── market_data.json")
        print(f"  │   └── option_chain.csv")
        print(f"  └── plots/")
        print(f"      ├── option_analysis.html")
        print(f"      ├── price_history.html")
        print(f"      └── volatility.html")
        
        await mpc_pricer.shutdown()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure the ticker symbol is valid and try again.")

if __name__ == "__main__":
    asyncio.run(main())

# Set page config
st.set_page_config(
    page_title="MPC Option Pricing",
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
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
    }
    .stButton > button {
        background-color: #00ff9d;
        color: #0e1117;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("MPC Option Pricing Dashboard")

# Sidebar
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
S0 = st.sidebar.number_input("Current Stock Price", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price", value=100.0, step=1.0)
r = st.sidebar.number_input("Risk-free Rate", value=0.05, step=0.01, format="%.2f")
sigma = st.sidebar.number_input("Volatility", value=0.2, step=0.01, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (years)", value=1.0, step=0.1, format="%.1f")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Price Analysis", "Risk Analysis", "Historical Data"])

with tab1:
    st.header("Price Analysis")
    
    # Calculate prices (simplified for demonstration)
    mc_price = S0 * 0.1
    bs_price = S0 * 0.12
    bin_price = S0 * 0.11
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader("Monte Carlo")
        st.markdown(f"<h2 style='color: #00ff9d;'>${mc_price:.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader("Black-Scholes")
        st.markdown(f"<h2 style='color: #00ff9d;'>${bs_price:.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div style='background-color: #1a1c23; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader("Binomial")
        st.markdown(f"<h2 style='color: #00ff9d;'>${bin_price:.2f}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Price comparison chart
    methods = ['Monte Carlo', 'Black-Scholes', 'Binomial']
    prices = [mc_price, bs_price, bin_price]
    
    fig = go.Figure()
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
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Risk Analysis")
    
    # Simulated stress test results
    stress_results = {
        "Base Case": {"price": 10.0, "var": 2.0, "es": 3.0},
        "Market Crash": {"price": 15.0, "var": 4.0, "es": 6.0},
        "Volatility Spike": {"price": 12.0, "var": 3.0, "es": 4.5}
    }
    
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
        st.markdown("<p>Risk metrics visualization will be added here.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.header("Historical Data")
    
    # Generate sample historical data
    dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
    prices = np.random.normal(100, 10, len(dates)).cumsum() + 1000
    
    # Create price history plot
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            name='Price History',
            line=dict(color='#00ff9d')
        )
    )
    
    fig1.update_layout(
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
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Generate sample volatility data
    volatility = np.random.normal(0.2, 0.05, len(dates))
    
    # Create volatility plot
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=dates,
            y=volatility,
            name='Volatility',
            line=dict(color='#00ff9d')
        )
    )
    
    fig2.update_layout(
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
    
    st.plotly_chart(fig2, use_container_width=True) 