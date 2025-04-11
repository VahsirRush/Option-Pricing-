import os
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv

def test_alpha_vantage_connection():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    print(f"Using API key: {api_key}")
    
    try:
        # Initialize the TimeSeries class
        ts = TimeSeries(key=api_key, output_format='pandas')
        
        # Try to get some data
        print("Attempting to fetch AAPL data...")
        data, meta_data = ts.get_daily(symbol='AAPL', outputsize='compact')
        
        if data is not None and not data.empty:
            print("Successfully retrieved data!")
            print("\nFirst few rows of data:")
            print(data.head())
            return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. Rate limit exceeded")
        print("3. Network connectivity issue")
        return False

if __name__ == "__main__":
    test_alpha_vantage_connection() 