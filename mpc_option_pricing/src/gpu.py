import numpy as np
from typing import Optional, Tuple
from .config import settings
from .logger import logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("CuPy not available. GPU acceleration disabled.")

class GPUAccelerator:
    """GPU acceleration for option pricing calculations."""
    
    def __init__(self):
        self.use_gpu = settings.USE_GPU and GPU_AVAILABLE
        if self.use_gpu:
            logger.info("GPU acceleration enabled")
        else:
            logger.info("GPU acceleration disabled")
    
    def monte_carlo_paths(self, 
                         S0: float, 
                         r: float, 
                         sigma: float, 
                         T: float, 
                         n_paths: int, 
                         n_steps: int) -> np.ndarray:
        """Generate Monte Carlo paths using GPU if available."""
        if not self.use_gpu:
            return self._cpu_monte_carlo_paths(S0, r, sigma, T, n_paths, n_steps)
        
        try:
            dt = T / n_steps
            drift = (r - 0.5 * sigma**2) * dt
            vol = sigma * np.sqrt(dt)
            
            # Generate random numbers on GPU
            z = cp.random.standard_normal((n_paths, n_steps))
            
            # Calculate returns
            returns = drift + vol * z
            
            # Calculate paths
            paths = cp.exp(cp.cumsum(returns, axis=1))
            paths = cp.insert(paths, 0, 1.0, axis=1)
            paths = S0 * paths
            
            return cp.asnumpy(paths)
        except Exception as e:
            logger.warning(f"GPU Monte Carlo failed: {str(e)}")
            return self._cpu_monte_carlo_paths(S0, r, sigma, T, n_paths, n_steps)
    
    def _cpu_monte_carlo_paths(self, 
                              S0: float, 
                              r: float, 
                              sigma: float, 
                              T: float, 
                              n_paths: int, 
                              n_steps: int) -> np.ndarray:
        """Generate Monte Carlo paths on CPU."""
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        z = np.random.standard_normal((n_paths, n_steps))
        returns = drift + vol * z
        
        paths = np.exp(np.cumsum(returns, axis=1))
        paths = np.insert(paths, 0, 1.0, axis=1)
        paths = S0 * paths
        
        return paths
    
    def calculate_greeks(self, 
                        S0: float, 
                        K: float, 
                        r: float, 
                        sigma: float, 
                        T: float) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks using GPU if available."""
        if not self.use_gpu:
            return self._cpu_calculate_greeks(S0, K, r, sigma, T)
        
        try:
            # Implementation using GPU
            # This is a placeholder - actual implementation would use GPU-accelerated calculations
            return self._cpu_calculate_greeks(S0, K, r, sigma, T)
        except Exception as e:
            logger.warning(f"GPU Greeks calculation failed: {str(e)}")
            return self._cpu_calculate_greeks(S0, K, r, sigma, T)
    
    def _cpu_calculate_greeks(self, 
                             S0: float, 
                             K: float, 
                             r: float, 
                             sigma: float, 
                             T: float) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks on CPU."""
        # Implementation of Greeks calculations
        # This is a placeholder - actual implementation would calculate all Greeks
        return 0.0, 0.0, 0.0, 0.0, 0.0

# Create global GPU accelerator instance
gpu_accelerator = GPUAccelerator() 