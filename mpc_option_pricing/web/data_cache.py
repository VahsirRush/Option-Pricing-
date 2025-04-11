import os
import json
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

class StockDataCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, ticker):
        return os.path.join(self.cache_dir, f"{ticker}.json")
    
    def get(self, ticker):
        """Get cached data for a ticker if it exists and is not expired."""
        cache_path = self._get_cache_path(ticker)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check if cache is expired (older than 1 day)
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time > timedelta(days=1):
                logger.info(f"Cache expired for {ticker}")
                return None
            
            # Convert data back to DataFrame
            df = pd.DataFrame(data['data'])
            df.index = pd.to_datetime(df.index)
            return df
            
        except Exception as e:
            logger.error(f"Error reading cache for {ticker}: {str(e)}")
            return None
    
    def set(self, ticker, data):
        """Cache data for a ticker."""
        cache_path = self._get_cache_path(ticker)
        try:
            # Convert DataFrame to dict for JSON serialization
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data.to_dict(orient='index')
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            logger.info(f"Cached data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {str(e)}")
    
    def clear(self, ticker=None):
        """Clear cache for a specific ticker or all tickers."""
        if ticker:
            cache_path = self._get_cache_path(ticker)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for {ticker}")
        else:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cleared all cache") 