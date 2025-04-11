from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the MPCOptionPricing class
try:
    from src.mpc_option_pricing import MPCOptionPricing
    has_mpc_pricer = True
except ImportError:
    has_mpc_pricer = False

app = FastAPI(
    title="MPC Option Pricing API",
    description="API for option pricing using Monte Carlo, Black-Scholes, and Binomial models",
    version="1.0.0"
)

class OptionRequest(BaseModel):
    S0: float
    K: float
    r: float
    sigma: float
    T: float
    method: str = "black_scholes"  # Default method
    n_paths: Optional[int] = 10000
    n_steps: Optional[int] = 100

class OptionResponse(BaseModel):
    price: float
    method: str
    parameters: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "Welcome to the MPC Option Pricing API"}

@app.post("/price", response_model=OptionResponse)
async def price_option(request: OptionRequest):
    if not has_mpc_pricer:
        raise HTTPException(status_code=500, detail="MPCOptionPricing module not available")
    
    try:
        mpc_pricer = MPCOptionPricing()
        
        if request.method == "monte_carlo":
            price = mpc_pricer.monte_carlo_option_price(
                request.S0, request.K, request.r, request.sigma, request.T,
                n_paths=request.n_paths, n_steps=request.n_steps
            )
        elif request.method == "black_scholes":
            price = mpc_pricer.black_scholes_option_price(
                request.S0, request.K, request.r, request.sigma, request.T
            )
        elif request.method == "binomial":
            price = mpc_pricer.binomial_option_price(
                request.S0, request.K, request.r, request.sigma, request.T,
                n_steps=request.n_steps
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return OptionResponse(
            price=price,
            method=request.method,
            parameters={
                "S0": request.S0,
                "K": request.K,
                "r": request.r,
                "sigma": request.sigma,
                "T": request.T,
                "n_paths": request.n_paths,
                "n_steps": request.n_steps
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating option price: {str(e)}")

@app.get("/methods")
async def get_methods():
    return {
        "methods": [
            {"name": "monte_carlo", "description": "Monte Carlo simulation"},
            {"name": "black_scholes", "description": "Black-Scholes model"},
            {"name": "binomial", "description": "Binomial model"}
        ]
    } 