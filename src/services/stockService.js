import axios from 'axios';

const API_KEY = process.env.REACT_APP_FINANCIAL_API_KEY;
const BASE_URL = 'https://financialmodelingprep.com/api/v3';

export const fetchStockData = async (symbol = 'AAPL') => {
  try {
    // Fetch current stock data
    const [quoteResponse, historicalResponse] = await Promise.all([
      axios.get(`${BASE_URL}/quote/${symbol}?apikey=${API_KEY}`),
      axios.get(`${BASE_URL}/historical-price-full/${symbol}?apikey=${API_KEY}`)
    ]);

    const quote = quoteResponse.data[0];
    const historical = historicalResponse.data.historical;

    // Calculate momentum scores
    const shortTermMomentum = calculateShortTermMomentum(historical);
    const longTermMomentum = calculateLongTermMomentum(historical);
    const overallScore = (shortTermMomentum + longTermMomentum) / 2;

    return {
      priceData: historical.slice(0, 30).map(day => ({
        date: day.date,
        price: day.close
      })),
      momentumData: {
        score: overallScore,
        shortTermMomentum,
        longTermMomentum
      },
      metrics: {
        peRatio: quote.pe,
        marketCap: quote.marketCap,
        fiftyTwoWeekHigh: quote.yearHigh,
        fiftyTwoWeekLow: quote.yearLow,
        volume: quote.volume,
        beta: quote.beta
      }
    };
  } catch (error) {
    console.error('Error fetching stock data:', error);
    throw new Error('Failed to fetch stock data');
  }
};

const calculateShortTermMomentum = (historical) => {
  if (historical.length < 20) return 0;
  const currentPrice = historical[0].close;
  const price20DaysAgo = historical[19].close;
  return ((currentPrice - price20DaysAgo) / price20DaysAgo) * 100;
};

const calculateLongTermMomentum = (historical) => {
  if (historical.length < 60) return 0;
  const currentPrice = historical[0].close;
  const price60DaysAgo = historical[59].close;
  return ((currentPrice - price60DaysAgo) / price60DaysAgo) * 100;
}; 