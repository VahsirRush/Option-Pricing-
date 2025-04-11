import numpy as np
from scipy.stats import norm
import mpyc
import asyncio
from typing import Tuple, List

class MPCOptionPricing:
    def __init__(self):
        """Initialize MPC runtime."""
        self.mpc = mpyc.runtime
        
    async def setup(self):
        """Setup MPC runtime."""
        await self.mpc.start()
        self.secfxp = self.mpc.SecFxp()
        
    async def shutdown(self):
        """Shutdown MPC runtime."""
        await self.mpc.shutdown()
        
    def _monte_carlo_path(self, S0: float, r: float, sigma: float, T: float, n_steps: int) -> List[float]:
        """Generate a single Monte Carlo path."""
        dt = T / n_steps
        path = [S0]
        for _ in range(n_steps):
            z = np.random.standard_normal()
            S = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            path.append(S)
        return path
    
    async def monte_carlo_option_price(self, 
                                     S0: float, 
                                     K: float, 
                                     r: float, 
                                     sigma: float, 
                                     T: float, 
                                     n_paths: int = 10000,
                                     n_steps: int = 100) -> float:
        """Calculate option price using Monte Carlo simulation with MPC."""
        # Generate paths
        paths = []
        for _ in range(n_paths):
            path = self._monte_carlo_path(S0, r, sigma, T, n_steps)
            paths.append(path)
        
        # Calculate payoffs
        payoffs = []
        for path in paths:
            ST = path[-1]
            payoff = max(ST - K, 0)  # Call option
            payoffs.append(self.secfxp(payoff))
        
        # Average payoffs and discount
        avg_payoff = sum(payoffs) / len(payoffs)
        discount = np.exp(-r * T)
        price = avg_payoff * self.secfxp(discount)
        
        return await self.mpc.output(price)
    
    async def black_scholes_option_price(self, 
                                       S0: float, 
                                       K: float, 
                                       r: float, 
                                       sigma: float, 
                                       T: float) -> float:
        """Calculate option price using Black-Scholes model with MPC."""
        # Calculate d1 and d2
        d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calculate option price components
        N1 = norm.cdf(d1)
        N2 = norm.cdf(d2)
        
        # Calculate price securely
        price = self.secfxp(S0)*self.secfxp(N1) - self.secfxp(K)*self.secfxp(np.exp(-r*T))*self.secfxp(N2)
        
        return await self.mpc.output(price)
    
    async def binomial_option_price(self, 
                                  S0: float, 
                                  K: float, 
                                  r: float, 
                                  sigma: float, 
                                  T: float, 
                                  n_steps: int = 100) -> float:
        """Calculate option price using Binomial model with MPC."""
        # Calculate binomial parameters
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp(r*dt) - d) / (u - d)
        
        # Calculate terminal stock prices
        stock_prices = []
        for j in range(n_steps + 1):
            price = S0 * (u ** (n_steps - j)) * (d ** j)
            stock_prices.append(price)
        
        # Calculate option values at maturity
        option_values = []
        for price in stock_prices:
            payoff = max(price - K, 0)
            option_values.append(self.secfxp(payoff))
        
        # Backward induction
        discount = np.exp(-r*dt)
        for i in range(n_steps - 1, -1, -1):
            new_values = []
            for j in range(i + 1):
                value = self.secfxp(discount) * (
                    self.secfxp(p) * option_values[j] + 
                    self.secfxp(1-p) * option_values[j+1]
                )
                new_values.append(value)
            option_values = new_values
        
        return await self.mpc.output(option_values[0])

async def main():
    # Example usage
    mpc_pricer = MPCOptionPricing()
    await mpc_pricer.setup()
    
    # Example parameters
    S0 = 100  # Initial stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0   # Time to maturity
    
    # Calculate prices using all three methods
    mc_price = await mpc_pricer.monte_carlo_option_price(S0, K, r, sigma, T)
    bs_price = await mpc_pricer.black_scholes_option_price(S0, K, r, sigma, T)
    bin_price = await mpc_pricer.binomial_option_price(S0, K, r, sigma, T)
    
    print(f"Monte Carlo Price: {mc_price}")
    print(f"Black-Scholes Price: {bs_price}")
    print(f"Binomial Price: {bin_price}")
    
    await mpc_pricer.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 