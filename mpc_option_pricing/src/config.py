import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./option_pricing.db")
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # MPC settings
    MPC_SECURITY_PARAMETER: int = int(os.getenv("MPC_SECURITY_PARAMETER", "128"))
    MPC_PARTIES: int = int(os.getenv("MPC_PARTIES", "3"))
    
    # Model settings
    MONTE_CARLO_PATHS: int = int(os.getenv("MONTE_CARLO_PATHS", "10000"))
    MONTE_CARLO_STEPS: int = int(os.getenv("MONTE_CARLO_STEPS", "100"))
    BINOMIAL_STEPS: int = int(os.getenv("BINOMIAL_STEPS", "100"))
    
    # Cache settings
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "cache"))
    CACHE_EXPIRY: int = int(os.getenv("CACHE_EXPIRY", "3600"))  # 1 hour
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Path = Path(os.getenv("LOG_FILE", "logs/option_pricing.log"))
    
    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure cache directory exists
settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True) 