import React from 'react';
import { Box, useTheme } from '@mui/material';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

// Register the chart components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface SkillRadarChartProps {
  skills: {
    codingStyle: number;
    projectComplexity: number;
    communityEngagement: number;
    documentation: number;
  };
}

const SkillRadarChart: React.FC<SkillRadarChartProps> = ({ skills }) => {
  const theme = useTheme();
  
  // Define chart data
  const data = {
    labels: [
      'Coding Style',
      'Project Complexity',
      'Community Engagement',
      'Documentation',
    ],
    datasets: [
      {
        label: 'Developer Skills',
        data: [
          skills.codingStyle,
          skills.projectComplexity, 
          skills.communityEngagement,
          skills.documentation,
        ],
        backgroundColor: `${theme.palette.primary.main}40`, // With opacity
        borderColor: theme.palette.primary.main,
        borderWidth: 2,
        pointBackgroundColor: theme.palette.secondary.main,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: theme.palette.secondary.main,
      },
    ],
  };

  // Chart options
  const options = {
    scales: {
      r: {
        angleLines: {
          display: true,
          color: theme.palette.divider,
        },
        grid: {
          color: theme.palette.divider,
        },
        pointLabels: {
          color: theme.palette.text.primary,
          font: {
            size: 12,
          },
        },
        suggestedMin: 0,
        suggestedMax: 5,
        ticks: {
          stepSize: 1,
          backdropColor: 'transparent',
          color: theme.palette.text.secondary,
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: theme.palette.background.paper,
        titleColor: theme.palette.text.primary,
        bodyColor: theme.palette.text.secondary,
        borderColor: theme.palette.divider,
        borderWidth: 1,
        padding: 10,
        boxPadding: 5,
      },
    },
    maintainAspectRatio: false,
  };

  return (
    <Box sx={{ height: 250, p: 2 }}>
      <Radar data={data} options={options} />
    </Box>
  );
};

export default SkillRadarChart;