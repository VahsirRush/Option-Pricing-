import os
from dotenv import load_dotenv
from polygon import RESTClient
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def test_polygon_api():
    """Test the Polygon.io API connection and key"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY not found in environment variables")
        return False
    
    print(f"API Key length: {len(api_key)}")
    
    try:
        # Initialize client
        client = RESTClient(api_key)
        
        # Test with a simple ticker
        ticker = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        print(f"Testing API with ticker {ticker} from {start_date.date()} to {end_date.date()}")
        
        # Try to get daily aggregates
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            adjusted=True
        )
        
        if aggs:
            print(f"Success! Retrieved {len(aggs)} days of data")
            # Print first day's data
            first_day = aggs[0]
            print("\nFirst day's data:")
            print(f"Date: {datetime.fromtimestamp(first_day.timestamp/1000)}")
            print(f"Open: {first_day.open}")
            print(f"High: {first_day.high}")
            print(f"Low: {first_day.low}")
            print(f"Close: {first_day.close}")
            print(f"Volume: {first_day.volume}")
            return True
        else:
            print("No data returned from API")
            return False
            
    except Exception as e:
        print(f"Error testing API: {str(e)}")
        if "NOT_AUTHORIZED" in str(e):
            print("API key is not authorized. Please check your Polygon.io subscription.")
        elif "rate limit" in str(e).lower():
            print("Rate limit exceeded. Please try again later.")
        return False

if __name__ == "__main__":
    test_polygon_api() 