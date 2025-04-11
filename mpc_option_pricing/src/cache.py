import pickle
from pathlib import Path
import hashlib
import time
from typing import Any, Optional
from .config import settings
from .logger import logger

class Cache:
    """Simple file-based cache implementation."""
    
    def __init__(self):
        self.cache_dir = settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check if cache is expired
            if time.time() - data['timestamp'] > settings.CACHE_EXPIRY:
                cache_path.unlink()
                return None
                
            return data['value']
        except Exception as e:
            logger.warning(f"Cache read error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                'value': value,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error for key {key}: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")

# Create global cache instance
cache = Cache() 