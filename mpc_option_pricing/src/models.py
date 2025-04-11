from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .database import Base

class MarketData(Base):
    """Market data model."""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    current_price = Column(Float)
    strike_price = Column(Float)
    risk_free_rate = Column(Float)
    volatility = Column(Float)
    time_to_maturity = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    option_prices = relationship("OptionPrice", back_populates="market_data")
    greeks = relationship("Greeks", back_populates="market_data")
    risk_metrics = relationship("RiskMetrics", back_populates="market_data")

class OptionPrice(Base):
    """Option price model."""
    __tablename__ = "option_prices"

    id = Column(Integer, primary_key=True, index=True)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))
    model_type = Column(String)  # monte_carlo, black_scholes, binomial
    price = Column(Float)
    confidence_interval = Column(JSON)  # Store as JSON: {"lower": 0.0, "upper": 0.0}
    computation_time = Column(Float)  # in seconds
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="option_prices")

class Greeks(Base):
    """Option Greeks model."""
    __tablename__ = "greeks"

    id = Column(Integer, primary_key=True, index=True)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="greeks")

class RiskMetrics(Base):
    """Risk metrics model."""
    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True, index=True)
    market_data_id = Column(Integer, ForeignKey("market_data.id"))
    value_at_risk = Column(Float)
    expected_shortfall = Column(Float)
    stress_test_results = Column(JSON)  # Store multiple stress test scenarios
    correlation_matrix = Column(JSON)  # Store correlation matrix
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="risk_metrics")

class HistoricalData(Base):
    """Historical data model."""
    __tablename__ = "historical_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    date = Column(DateTime)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow) 