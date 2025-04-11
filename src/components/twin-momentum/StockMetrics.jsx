import React from 'react';
import { Grid, Typography, Box } from '@mui/material';
import { styled } from '@mui/material/styles';

const MetricCard = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  backgroundColor: theme.palette.background.default,
  borderRadius: '8px',
  border: `1px solid ${theme.palette.divider}`,
}));

const MetricValue = styled(Typography)(({ theme }) => ({
  fontSize: '1.5rem',
  fontWeight: 'bold',
  color: theme.palette.primary.main,
}));

const StockMetrics = ({ data }) => {
  if (!data) return null;

  const metrics = [
    { label: 'P/E Ratio', value: data.peRatio, format: (val) => val.toFixed(2) },
    { label: 'Market Cap', value: data.marketCap, format: (val) => `$${(val / 1e9).toFixed(2)}B` },
    { label: '52W High', value: data.fiftyTwoWeekHigh, format: (val) => `$${val.toFixed(2)}` },
    { label: '52W Low', value: data.fiftyTwoWeekLow, format: (val) => `$${val.toFixed(2)}` },
    { label: 'Volume', value: data.volume, format: (val) => (val / 1e6).toFixed(2) + 'M' },
    { label: 'Beta', value: data.beta, format: (val) => val.toFixed(2) },
  ];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Key Metrics
      </Typography>
      <Grid container spacing={2}>
        {metrics.map((metric) => (
          <Grid item xs={6} sm={4} md={2} key={metric.label}>
            <MetricCard>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                {metric.label}
              </Typography>
              <MetricValue>
                {metric.format(metric.value)}
              </MetricValue>
            </MetricCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default StockMetrics; 