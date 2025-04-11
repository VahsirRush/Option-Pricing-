import pytest
import numpy as np
from src.mpc_option_pricing import MPCOptionPricing
from src.risk import risk_manager
from src.gpu import gpu_accelerator

@pytest.fixture
async def mpc_pricer():
    """Create MPC pricer instance."""
    pricer = MPCOptionPricing()
    await pricer.setup()
    yield pricer
    await pricer.shutdown()

@pytest.mark.asyncio
async def test_monte_carlo_option_price(mpc_pricer):
    """Test Monte Carlo option pricing."""
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    price = await mpc_pricer.monte_carlo_option_price(
        S0, K, r, sigma, T,
        n_paths=1000,
        n_steps=100
    )
    
    assert isinstance(price, float)
    assert price > 0

@pytest.mark.asyncio
async def test_black_scholes_option_price(mpc_pricer):
    """Test Black-Scholes option pricing."""
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    price = await mpc_pricer.black_scholes_option_price(
        S0, K, r, sigma, T
    )
    
    assert isinstance(price, float)
    assert price > 0

@pytest.mark.asyncio
async def test_binomial_option_price(mpc_pricer):
    """Test Binomial option pricing."""
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    price = await mpc_pricer.binomial_option_price(
        S0, K, r, sigma, T,
        n_steps=100
    )
    
    assert isinstance(price, float)
    assert price > 0

def test_gpu_monte_carlo_paths():
    """Test GPU-accelerated Monte Carlo path generation."""
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_paths = 1000
    n_steps = 100
    
    paths = gpu_accelerator.monte_carlo_paths(
        S0, r, sigma, T,
        n_paths, n_steps
    )
    
    assert isinstance(paths, np.ndarray)
    assert paths.shape == (n_paths, n_steps + 1)
    assert np.all(paths > 0)

def test_risk_manager_stress_test():
    """Test risk manager stress testing."""
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    
    results = risk_manager.stress_test(S0, K, r, sigma, T)
    
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'market_crash',
        'volatility_spike',
        'interest_rate_shock',
        'combined_shock'
    ])
    
    for scenario, metrics in results.items():
        assert isinstance(metrics, dict)
        assert all(key in metrics for key in ['price', 'var', 'es'])

def test_risk_manager_var():
    """Test Value at Risk calculation."""
    returns = np.random.normal(0, 0.01, 1000)
    var = risk_manager.calculate_var(returns)
    
    assert isinstance(var, float)
    assert var < 0  # VaR should be negative for normal returns

def test_risk_manager_expected_shortfall():
    """Test Expected Shortfall calculation."""
    returns = np.random.normal(0, 0.01, 1000)
    es = risk_manager.calculate_expected_shortfall(returns)
    
    assert isinstance(es, float)
    assert es < 0  # ES should be negative for normal returns

def test_risk_manager_correlation_matrix():
    """Test correlation matrix calculation."""
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    corr_matrix = risk_manager.calculate_correlation_matrix(tickers)
    
    assert isinstance(corr_matrix, np.ndarray)
    assert corr_matrix.shape == (len(tickers), len(tickers))
    assert np.all(np.diag(corr_matrix) == 1.0)  # Diagonal should be 1.0
    assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)

def test_risk_manager_portfolio_metrics():
    """Test portfolio metrics calculation."""
    positions = {'AAPL': 100, 'MSFT': 200}
    prices = {'AAPL': 150.0, 'MSFT': 250.0}
    correlations = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    metrics = risk_manager.calculate_portfolio_metrics(
        positions, prices, correlations
    )
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        'portfolio_value',
        'portfolio_volatility',
        'position_weights'
    ])
    assert metrics['portfolio_value'] > 0
    assert metrics['portfolio_volatility'] >= 0
    assert len(metrics['position_weights']) == len(positions) 