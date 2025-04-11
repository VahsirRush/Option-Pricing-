import os
from polygon import RESTClient
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_polygon_api():
    """Test Polygon.io API functionality"""
    logger.info("Testing Polygon.io API...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key:
        logger.error("Polygon API key not found in environment variables")
        return False
    
    try:
        # Initialize Polygon client
        client = RESTClient(api_key)
        
        # Test 1: Get daily aggregates for AAPL
        logger.info("Testing daily aggregates for AAPL...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        aggs = client.get_aggs(
            ticker="AAPL",
            multiplier=1,
            timespan="day",
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
            adjusted=True
        )
        
        if not aggs:
            logger.error("No data returned from Polygon API for AAPL")
            return False
            
        logger.info(f"Successfully retrieved {len(aggs)} days of data for AAPL")
        
        # Test 2: Get company details
        logger.info("Testing company details retrieval...")
        ticker_details = client.get_ticker_details("AAPL")
        
        if not ticker_details:
            logger.error("No company details returned from Polygon API")
            return False
            
        logger.info(f"Successfully retrieved company details for AAPL: {ticker_details.name}")
        
        # Test 3: Get real-time quote
        logger.info("Testing real-time quote retrieval...")
        quote = client.get_last_quote("AAPL")
        
        if not quote:
            logger.error("No quote data returned from Polygon API")
            return False
            
        logger.info(f"Successfully retrieved quote for AAPL: ${quote.last.price}")
        
        logger.info("All Polygon.io API tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Polygon.io API: {str(e)}")
        return False

def main():
    """Run all API tests"""
    logger.info("Starting API tests...")
    
    # Test Polygon.io API
    polygon_success = test_polygon_api()
    
    # Summary
    logger.info("\nAPI Test Summary:")
    logger.info(f"Polygon.io API: {'✓' if polygon_success else '✗'}")
    
    # Overall status
    if polygon_success:
        logger.info("\nAll APIs are working correctly!")
        return 0
    else:
        logger.error("\nSome API tests failed. Please check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 