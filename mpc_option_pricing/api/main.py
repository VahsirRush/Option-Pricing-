from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from ..src.database import get_db, init_db
from ..src.models import MarketData, OptionPrice, Greeks, RiskMetrics
from ..src.mpc_option_pricing import MPCOptionPricing
from ..src.risk import risk_manager
from ..src.logger import logger

app = FastAPI(
    title="MPC Option Pricing API",
    description="API for secure option pricing using Multi-Party Computation",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("API server started")

@app.get("/")
async def root():
    return {"message": "Welcome to MPC Option Pricing API"}

@app.post("/market-data/")
async def create_market_data(
    market_data: dict,
    db: Session = Depends(get_db)
):
    """Create new market data entry."""
    try:
        db_market_data = MarketData(**market_data)
        db.add(db_market_data)
        db.commit()
        db.refresh(db_market_data)
        return db_market_data
    except Exception as e:
        logger.error(f"Error creating market data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market-data/{ticker}")
async def get_market_data(
    ticker: str,
    db: Session = Depends(get_db)
):
    """Get market data for a specific ticker."""
    market_data = db.query(MarketData).filter(
        MarketData.ticker == ticker
    ).order_by(MarketData.timestamp.desc()).first()
    
    if not market_data:
        raise HTTPException(status_code=404, detail="Market data not found")
    
    return market_data

@app.post("/calculate-prices/")
async def calculate_prices(
    market_data: dict,
    db: Session = Depends(get_db)
):
    """Calculate option prices using different models."""
    try:
        # Initialize MPC pricer
        mpc_pricer = MPCOptionPricing()
        await mpc_pricer.setup()
        
        # Extract parameters
        S0 = market_data['current_price']
        K = market_data['strike_price']
        r = market_data['risk_free_rate']
        sigma = market_data['volatility']
        T = market_data['time_to_maturity']
        
        # Calculate prices
        mc_price = await mpc_pricer.monte_carlo_option_price(
            S0, K, r, sigma, T,
            n_paths=10000,
            n_steps=100
        )
        
        bs_price = await mpc_pricer.black_scholes_option_price(
            S0, K, r, sigma, T
        )
        
        bin_price = await mpc_pricer.binomial_option_price(
            S0, K, r, sigma, T,
            n_steps=100
        )
        
        # Store results
        db_market_data = MarketData(**market_data)
        db.add(db_market_data)
        db.commit()
        db.refresh(db_market_data)
        
        prices = [
            OptionPrice(
                market_data_id=db_market_data.id,
                model_type='monte_carlo',
                price=mc_price
            ),
            OptionPrice(
                market_data_id=db_market_data.id,
                model_type='black_scholes',
                price=bs_price
            ),
            OptionPrice(
                market_data_id=db_market_data.id,
                model_type='binomial',
                price=bin_price
            )
        ]
        
        db.add_all(prices)
        db.commit()
        
        await mpc_pricer.shutdown()
        
        return {
            "market_data": db_market_data,
            "prices": prices
        }
    except Exception as e:
        logger.error(f"Error calculating prices: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/calculate-risk/")
async def calculate_risk(
    market_data: dict,
    db: Session = Depends(get_db)
):
    """Calculate risk metrics for an option."""
    try:
        # Extract parameters
        S0 = market_data['current_price']
        K = market_data['strike_price']
        r = market_data['risk_free_rate']
        sigma = market_data['volatility']
        T = market_data['time_to_maturity']
        
        # Perform stress testing
        stress_results = risk_manager.stress_test(S0, K, r, sigma, T)
        
        # Store results
        db_market_data = MarketData(**market_data)
        db.add(db_market_data)
        db.commit()
        db.refresh(db_market_data)
        
        risk_metrics = RiskMetrics(
            market_data_id=db_market_data.id,
            stress_test_results=stress_results
        )
        
        db.add(risk_metrics)
        db.commit()
        
        return {
            "market_data": db_market_data,
            "risk_metrics": risk_metrics
        }
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 