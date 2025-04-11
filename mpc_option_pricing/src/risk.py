import numpy as np
from typing import Dict, List, Tuple
from .config import settings
from .logger import logger
from .gpu import gpu_accelerator

class RiskManager:
    """Risk management calculations for option pricing."""
    
    def __init__(self):
        self.confidence_level = 0.95  # 95% confidence level
    
    def calculate_var(self, 
                     returns: np.ndarray, 
                     confidence_level: float = None) -> float:
        """Calculate Value at Risk (VaR)."""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_expected_shortfall(self, 
                                   returns: np.ndarray, 
                                   confidence_level: float = None) -> float:
        """Calculate Expected Shortfall (ES)."""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def stress_test(self, 
                   S0: float, 
                   K: float, 
                   r: float, 
                   sigma: float, 
                   T: float,
                   scenarios: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """Perform stress testing with different scenarios."""
        if scenarios is None:
            scenarios = {
                'market_crash': {'S0': S0 * 0.7, 'sigma': sigma * 1.5},
                'volatility_spike': {'sigma': sigma * 2.0},
                'interest_rate_shock': {'r': r * 2.0},
                'combined_shock': {
                    'S0': S0 * 0.8,
                    'sigma': sigma * 1.5,
                    'r': r * 1.5
                }
            }
        
        results = {}
        for scenario_name, params in scenarios.items():
            # Create scenario parameters
            scenario_S0 = params.get('S0', S0)
            scenario_sigma = params.get('sigma', sigma)
            scenario_r = params.get('r', r)
            
            # Generate paths for scenario
            paths = gpu_accelerator.monte_carlo_paths(
                scenario_S0, scenario_r, scenario_sigma, T,
                n_paths=10000, n_steps=100
            )
            
            # Calculate payoffs
            payoffs = np.maximum(paths[:, -1] - K, 0)
            
            # Calculate scenario metrics
            results[scenario_name] = {
                'price': np.mean(payoffs) * np.exp(-scenario_r * T),
                'var': self.calculate_var(payoffs),
                'es': self.calculate_expected_shortfall(payoffs)
            }
        
        return results
    
    def calculate_correlation_matrix(self, 
                                   tickers: List[str], 
                                   period: str = '1y') -> np.ndarray:
        """Calculate correlation matrix for a list of tickers."""
        import yfinance as yf
        
        # Fetch historical data
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            data[ticker] = hist['Close'].pct_change().dropna()
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        return df.corr().values
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, float], 
                                  prices: Dict[str, float],
                                  correlations: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        # Calculate portfolio value
        portfolio_value = sum(positions[ticker] * prices[ticker] 
                            for ticker in positions)
        
        # Calculate position weights
        weights = np.array([positions[ticker] * prices[ticker] / portfolio_value 
                          for ticker in positions])
        
        # Calculate portfolio variance
        portfolio_variance = weights.T @ correlations @ weights
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_volatility': portfolio_volatility,
            'position_weights': dict(zip(positions.keys(), weights))
        }

# Create global risk manager instance
risk_manager = RiskManager() 