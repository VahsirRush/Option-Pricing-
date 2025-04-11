import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const ScoreContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: theme.spacing(2),
}));

const ScoreCircle = styled(Box)(({ theme, score }) => ({
  width: '120px',
  height: '120px',
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  background: `conic-gradient(${theme.palette.primary.main} ${score}%, ${theme.palette.grey[200]} ${score}%)`,
  marginBottom: theme.spacing(2),
}));

const ScoreText = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 'bold',
  color: theme.palette.primary.main,
}));

const MomentumScore = ({ data }) => {
  if (!data) return null;

  const { score, shortTermMomentum, longTermMomentum } = data;

  return (
    <ScoreContainer>
      <Typography variant="h6" gutterBottom>
        Twin Momentum Score
      </Typography>
      <ScoreCircle score={score}>
        <ScoreText>{score}</ScoreText>
      </ScoreCircle>
      
      <Box width="100%" mb={2}>
        <Typography variant="body2" gutterBottom>
          Short-term Momentum
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={shortTermMomentum} 
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>

      <Box width="100%">
        <Typography variant="body2" gutterBottom>
          Long-term Momentum
        </Typography>
        <LinearProgress 
          variant="determinate" 
          value={longTermMomentum} 
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>
    </ScoreContainer>
  );
};

export default MomentumScore; 