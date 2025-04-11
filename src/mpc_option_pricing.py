import numpy as np
import pandas as pd
from scipy.stats import norm

class MPCOptionPricing:
    """
    A class for pricing options using various methods including Monte Carlo,
    Black-Scholes, and Binomial models.
    """
    
    def __init__(self):
        """Initialize the MPCOptionPricing class."""
        pass
    
    def monte_carlo_option_price(self, S0, K, r, sigma, T, n_paths=10000, n_steps=100):
        """
        Calculate option price using Monte Carlo simulation.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity in years
        n_paths : int, optional
            Number of simulation paths
        n_steps : int, optional
            Number of time steps
        
        Returns:
        --------
        float
            Option price
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Monte Carlo calculation with consistent parameters
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate all random numbers at once for better performance
        z = np.random.standard_normal((n_paths, n_steps))
        
        # Calculate drift and volatility terms
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        # Calculate paths using vectorized operations
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + vol * z[:, t-1])
        
        # Calculate option price
        payoffs = np.maximum(paths[:, -1] - K, 0)
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return option_price
    
    def black_scholes_option_price(self, S0, K, r, sigma, T):
        """
        Calculate option price using Black-Scholes model.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity in years
        
        Returns:
        --------
        float
            Option price
        """
        # Calculate d1 and d2
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return option_price
    
    def binomial_option_price(self, S0, K, r, sigma, T, n_steps=100):
        """
        Calculate option price using Binomial model.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        K : float
            Strike price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity in years
        n_steps : int, optional
            Number of time steps
        
        Returns:
        --------
        float
            Option price
        """
        # Simplified implementation for demonstration
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize price tree
        price_tree = np.zeros((n_steps + 1, n_steps + 1))
        price_tree[0, 0] = S0
        
        # Fill price tree
        for i in range(1, n_steps + 1):
            for j in range(i + 1):
                if j == 0:
                    price_tree[i, j] = price_tree[i-1, j] * d
                else:
                    price_tree[i, j] = price_tree[i-1, j-1] * u
        
        # Calculate option price
        option_price = np.maximum(price_tree[n_steps, :] - K, 0)
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_price[j] = np.exp(-r * dt) * (p * option_price[j+1] + (1 - p) * option_price[j])
        
        return option_price[0] 