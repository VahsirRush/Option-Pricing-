import pytest
import sys
import os

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MPCOptionPricing class
from src.mpc_option_pricing import MPCOptionPricing

def test_mpc_option_pricing_initialization():
    """Test that the MPCOptionPricing class can be initialized."""
    pricer = MPCOptionPricing()
    assert pricer is not None

def test_black_scholes_option_price():
    """Test the Black-Scholes option pricing method."""
    pricer = MPCOptionPricing()
    
    # Test parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    T = 1.0     # Time to maturity
    
    # Calculate option price
    price = pricer.black_scholes_option_price(S0, K, r, sigma, T)
    
    # Check that the price is a positive number
    assert price > 0
    
    # Check that the price is reasonable (for a call option with S0=K, price should be around 10-15)
    assert 5 < price < 20

def test_monte_carlo_option_price():
    """Test the Monte Carlo option pricing method."""
    pricer = MPCOptionPricing()
    
    # Test parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    T = 1.0     # Time to maturity
    
    # Calculate option price
    price = pricer.monte_carlo_option_price(S0, K, r, sigma, T, n_paths=1000, n_steps=50)
    
    # Check that the price is a positive number
    assert price > 0
    
    # Check that the price is reasonable (for a call option with S0=K, price should be around 10-15)
    assert 5 < price < 20

def test_binomial_option_price():
    """Test the Binomial option pricing method."""
    pricer = MPCOptionPricing()
    
    # Test parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    T = 1.0     # Time to maturity
    
    # Calculate option price
    price = pricer.binomial_option_price(S0, K, r, sigma, T, n_steps=50)
    
    # Check that the price is a positive number
    assert price > 0
    
    # Check that the price is reasonable (for a call option with S0=K, price should be around 10-15)
    assert 5 < price < 20 