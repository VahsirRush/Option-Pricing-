# Option Pricing Model

A comprehensive option pricing model that uses the Black-Scholes model and Monte Carlo simulation to price options. The project includes a web interface for interactive analysis and an API for programmatic access.

## Features

- Black-Scholes option pricing model
- Monte Carlo simulation for option pricing
- Real-time stock data fetching using Polygon.io API
- Interactive web interface with Streamlit
- RESTful API for programmatic access
- Docker containerization for easy deployment
- Comprehensive documentation

## Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher (if running locally)
- Polygon.io API key

## Quick Start with Docker

1. Clone the repository:
```bash
git clone https://github.com/VahsirRush/Option-Pricing-.git
cd Option-Pricing-
```

2. Create a `.env` file in the root directory with your Polygon.io API key:
```bash
POLYGON_API_KEY=your_api_key_here
```

3. Start the application using Docker Compose:
```bash
docker-compose up -d
```

4. Access the services:
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- API Endpoint: http://localhost:8000

## Local Development Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export POLYGON_API_KEY=your_api_key_here  # On Windows: set POLYGON_API_KEY=your_api_key_here
```

4. Run the services:

For the API:
```bash
cd api
uvicorn main:app --reload
```

For the web interface:
```bash
cd web
streamlit run app.py
```

## Project Structure

```
.
├── api/                 # FastAPI backend
├── web/                # Streamlit frontend
├── src/                # Core pricing models
├── tests/              # Test suite
├── docs/               # Documentation
├── docker-compose.yml  # Docker configuration
└── requirements.txt    # Python dependencies
```

## API Endpoints

- `GET /api/stock/{symbol}`: Get stock data
- `POST /api/price/black-scholes`: Price options using Black-Scholes model
- `POST /api/price/monte-carlo`: Price options using Monte Carlo simulation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Polygon.io for providing market data
- Black-Scholes model for option pricing theory
- FastAPI and Streamlit for the web framework 