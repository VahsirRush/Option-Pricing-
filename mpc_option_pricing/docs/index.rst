Welcome to MPC Option Pricing's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   usage
   api
   models
   risk
   development
   contributing

Introduction
------------

MPC Option Pricing is a comprehensive implementation of option pricing models using Multi-Party Computation (MPC). The project provides:

- Secure computation of option prices using MPC
- Multiple pricing models (Monte Carlo, Black-Scholes, Binomial)
- Risk management features
- GPU acceleration
- REST API and web interface
- Comprehensive documentation and testing

Features
--------

* Secure computation using MPC
* Multiple pricing models
* Risk management tools
* GPU acceleration
* REST API
* Web interface
* Comprehensive testing
* Detailed documentation

Quick Start
----------

Install the package:

.. code-block:: bash

   pip install mpc-option-pricing

Basic usage:

.. code-block:: python

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 