# MPC Option Pricing

A comprehensive implementation of option pricing models using Multi-Party Computation (MPC).

## Features

- Secure computation of option prices using MPC
- Multiple pricing models:
  - Monte Carlo Simulation
  - Black-Scholes Model
  - Binomial Model
- Risk management features:
  - Value at Risk (VaR)
  - Expected Shortfall (ES)
  - Stress testing
  - Portfolio risk metrics
- GPU acceleration for performance
- REST API for integration
- Web interface for visualization
- Comprehensive testing and documentation

## Installation

### Using pip

```bash
pip install mpc-option-pricing
```

### From source

```bash
git clone https://github.com/yourusername/mpc-option-pricing.git
cd mpc-option-pricing
pip install -e .[dev]
```

## Usage

### Python API

```python
from mpc_option_pricing import MPCOptionPricing
import asyncio

async def main():
    # Initialize the MPC pricer
    mpc_pricer = MPCOptionPricing()
    await mpc_pricer.setup()
    
    # Set option parameters
    S0 = 100  # Initial stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0   # Time to maturity
    
    # Calculate prices using different methods
    mc_price = await mpc_pricer.monte_carlo_option_price(S0, K, r, sigma, T)
    bs_price = await mpc_pricer.black_scholes_option_price(S0, K, r, sigma, T)
    bin_price = await mpc_pricer.binomial_option_price(S0, K, r, sigma, T)
    
    print(f"Monte Carlo Price: {mc_price}")
    print(f"Black-Scholes Price: {bs_price}")
    print(f"Binomial Price: {bin_price}")
    
    await mpc_pricer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### REST API

Start the API server:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Web Interface

Start the web interface:

```bash
streamlit run web/app.py
```

## Docker

Build and run using Docker:

```bash
# Build the image
docker-compose build

# Start the services
docker-compose up -d

# View logs
docker-compose logs -f
```

## Development

### Setup

1. Clone the repository
2. Create a virtual environment
3. Install development dependencies:

```bash
pip install -e .[dev]
```

### Testing

Run tests:

```bash
pytest tests/
```

### Linting

Run linters:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Documentation

Build documentation:

```bash
cd docs && make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 