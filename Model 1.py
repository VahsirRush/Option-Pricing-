import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os
import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import random
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwinMomentumQuant:
    def __init__(self, ticker):
        self.ticker = ticker.lower()
        self.data = None
        self.live_price = None
        self.session = self._create_session()
        self.load_data()
    
    def _create_session(self):
        """Create a requests session with retry mechanism."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _validate_price(self, price):
        """Validate the scraped price."""
        try:
            price = float(price)
            if price <= 0:
                raise ValueError("Price must be positive")
            return price
        except (ValueError, TypeError):
            return None
    
    def fetch_live_price(self):
        """Fetch live stock price using web scraping from alternative sources."""
        sources = [
            f"https://www.bloomberg.com/quote/{self.ticker}:US",
            f"https://www.google.com/finance/quote/{self.ticker}:NASDAQ",
            f"https://finance.yahoo.com/quote/{self.ticker}"
        ]
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        
        for source in sources:
            try:
                headers = {"User-Agent": random.choice(user_agents)}
                time.sleep(random.uniform(2, 5))
                response = self.session.get(source, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Try multiple price selectors
                price_selectors = [
                    ("div", "price"),
                    ("span", "YMlKec fxKbKc"),
                    ("span", "Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)"),
                    ("span", "Fz(36px)")
                ]
                
                for tag, class_name in price_selectors:
                    price_tag = soup.find(tag, class_=class_name)
                    if price_tag:
                        price_text = price_tag.text.strip().replace(',', '').replace('$', '')
                        price = self._validate_price(price_text)
                        if price:
                            self.live_price = price
                            logger.info(f"Live price for {self.ticker.upper()}: {self.live_price}")
                            return
                
            except requests.RequestException as e:
                logger.error(f"Error fetching from {source}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error with {source}: {e}")
        
        logger.warning("Failed to retrieve live price from all sources.")
    
    def fetch_historical_data(self):
        """Fetch missing historical data using web scraping from Nasdaq."""
        logger.info(f"Fetching historical data for {self.ticker.upper()}...")
        url = f"https://www.nasdaq.com/market-activity/stocks/{self.ticker}/historical"
        
        try:
            headers = {"User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ])}
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            rows = soup.select("table.historical-data__table tr")
            if not rows:
                logger.error("No table rows found. Website structure may have changed.")
                return None
                
            data = []
            for row in rows[1:]:  # Skip header row
                cols = row.find_all("td")
                if len(cols) >= 2:
                    try:
                        date = cols[0].text.strip()
                        close_price = cols[1].text.strip().replace(',', '')
                        if close_price:
                            price = self._validate_price(close_price)
                            if price:
                                data.append([date, price])
                    except Exception as e:
                        logger.warning(f"Error processing row: {e}")
                        continue
            
            if not data:
                logger.error("No valid historical data found.")
                return None
            
            df = pd.DataFrame(data, columns=["Date", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            
            # Validate data
            if df.empty:
                logger.error("Empty DataFrame after processing")
                return None
                
            if df["Close"].isnull().any():
                logger.warning("Found missing values in Close prices")
                df = df.dropna()
            
            # Save data
            os.makedirs('data', exist_ok=True)
            df.to_csv(f"data/{self.ticker}.csv")
            logger.info(f"Successfully saved historical data for {self.ticker.upper()}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching historical data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data: {e}")
            return None
    
    def load_csv_data(self):
        """Load historical stock data from a local CSV file, fetch if missing."""
        file_path = f"data/{self.ticker}.csv"
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty and not df["Close"].isnull().all():
                    logger.info(f"Successfully loaded historical data for {self.ticker.upper()}")
                    return df
                else:
                    logger.warning("Loaded CSV file is empty or contains only null values")
            return self.fetch_historical_data()
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return self.fetch_historical_data()
    
    def load_data(self):
        """Fetch historical stock data from local CSV or scrape missing data."""
        self.data = self.load_csv_data()
        if self.data is not None:
            self.fetch_live_price()
            
            if self.live_price is not None:
                latest_date = self.data.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({"Close": [self.live_price]}, index=[latest_date])
                self.data = pd.concat([self.data, new_row])
                logger.info("Successfully updated data with live price")
    
    def run(self):
        """Execute the full quant analysis."""
        if self.data is None:
            logger.error("No data available for analysis")
            return {
                "Stock": self.ticker.upper(),
                "Status": "Error",
                "Message": "No data available for analysis"
            }
            
        return {
            "Stock": self.ticker.upper(),
            "Live Price": self.live_price if self.live_price else "Unavailable",
            "Historical Data Points": len(self.data),
            "Date Range": f"{self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}"
        }

if __name__ == "__main__":
    try:
        ticker = input("Enter the stock ticker: ").upper()
        quant_engine = TwinMomentumQuant(ticker)
        result = quant_engine.run()
        print("\nAnalysis Results:")
        for key, value in result.items():
            print(f"{key}: {value}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")








