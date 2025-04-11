Welcome to MPC Option Pricing's documentation!
============================================

MPC Option Pricing is a comprehensive suite for option pricing using various methods including Monte Carlo simulation, Black-Scholes model, and Binomial model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   models
   examples

Features
--------

* Monte Carlo simulation for option pricing
* Black-Scholes model implementation
* Binomial model for option pricing
* Risk analysis tools
* Historical data visualization
* REST API for integration with other systems

Installation
-----------

You can install MPC Option Pricing using pip:

.. code-block:: bash

   pip install mpc-option-pricing

Or clone the repository and install it manually:

.. code-block:: bash

   git clone https://github.com/yourusername/mpc-option-pricing.git
   cd mpc-option-pricing
   pip install -e .

Usage
-----

Basic usage example:

.. code-block:: python

   from src.mpc_option_pricing import MPCOptionPricing

   # Initialize the pricer
   pricer = MPCOptionPricing()

   # Calculate option price using Black-Scholes model
   price = pricer.black_scholes_option_price(
       S0=100.0,  # Initial stock price
       K=100.0,   # Strike price
       r=0.05,    # Risk-free rate
       sigma=0.2, # Volatility
       T=1.0      # Time to maturity
   )

   print(f"Option price: ${price:.2f}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 