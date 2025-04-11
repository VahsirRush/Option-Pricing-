import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, CircularProgress, TextField, Button } from '@mui/material';
import { styled } from '@mui/material/styles';
import StockMetrics from './StockMetrics';
import MomentumScore from './MomentumScore';
import PriceChart from './PriceChart';
import { fetchStockData } from '../../services/stockService';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.paper,
  borderRadius: '12px',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: theme.spacing(2),
  marginBottom: theme.spacing(3),
  alignItems: 'center',
}));

const TwinMomentumModel = () => {
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');

  const loadData = async (stockSymbol) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchStockData(stockSymbol);
      setStockData(data);
    } catch (err) {
      setError(err.message);
      setStockData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData(symbol);
    const interval = setInterval(() => loadData(symbol), 60000); // Update every minute
    return () => clearInterval(interval);
  }, [symbol]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputSymbol.trim().toUpperCase() !== symbol) {
      setSymbol(inputSymbol.trim().toUpperCase());
    }
  };

  if (loading && !stockData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Twin Momentum Model
      </Typography>
      
      <InputContainer component="form" onSubmit={handleSubmit}>
        <TextField
          label="Stock Symbol"
          variant="outlined"
          value={inputSymbol}
          onChange={(e) => setInputSymbol(e.target.value.toUpperCase())}
          size="small"
          sx={{ width: '200px' }}
        />
        <Button 
          variant="contained" 
          type="submit"
          disabled={loading || inputSymbol.trim().toUpperCase() === symbol}
        >
          Update
        </Button>
      </InputContainer>

      {error && (
        <Typography color="error" gutterBottom>
          {error}
        </Typography>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <StyledPaper>
            <PriceChart data={stockData?.priceData} />
          </StyledPaper>
        </Grid>
        <Grid item xs={12} md={4}>
          <StyledPaper>
            <MomentumScore data={stockData?.momentumData} />
          </StyledPaper>
        </Grid>
        <Grid item xs={12}>
          <StyledPaper>
            <StockMetrics data={stockData?.metrics} />
          </StyledPaper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TwinMomentumModel; 