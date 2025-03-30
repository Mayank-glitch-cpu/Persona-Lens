import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Chip, 
  Stack, 
  LinearProgress, 
  Grid,
  useTheme,
  Tooltip,
  IconButton
} from '@mui/material';
import StarIcon from '@mui/icons-material/Star';
import GitHubIcon from '@mui/icons-material/GitHub';
import CodeIcon from '@mui/icons-material/Code';

interface Project {
  name: string;
  stars: number;
  language: string;
}

interface RepositoryCardProps {
  repositories: Project[];
  username: string;
}

const RepositoryCard: React.FC<RepositoryCardProps> = ({ repositories, username }) => {
  const theme = useTheme();
  
  // Get relative star count for popularity bar
  const maxStars = Math.max(...repositories.map(repo => repo.stars), 100);
  
  // Map common languages to colors
  const getLanguageColor = (language: string) => {
    const colors: Record<string, string> = {
      'JavaScript': '#f1e05a',
      'TypeScript': '#3178c6',
      'Python': '#3572A5',
      'Java': '#b07219',
      'C#': '#178600',
      'PHP': '#4F5D95',
      'C++': '#f34b7d',
      'Ruby': '#701516',
      'Go': '#00ADD8',
      'Rust': '#dea584',
      'HTML': '#e34c26',
      'CSS': '#563d7c',
      'Swift': '#ffac45',
      'Kotlin': '#A97BFF',
    };
    
    return colors[language] || theme.palette.primary.main;
  };

  return (
    <Card variant="outlined" sx={{ 
      border: `1px solid ${theme.palette.divider}`,
      transition: 'all 0.3s ease',
      '&:hover': {
        boxShadow: theme.shadows[4],
        transform: 'translateY(-4px)'
      },
      mb: 2
    }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Notable Repositories
        </Typography>
        
        <Stack spacing={2}>
          {repositories.map((repo, index) => (
            <Grid container key={index} spacing={1} alignItems="center">
              <Grid item xs={12}>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 0.5
                }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                    <CodeIcon fontSize="small" sx={{ 
                      mr: 1, 
                      verticalAlign: 'text-bottom',
                      color: theme.palette.primary.main
                    }} />
                    {repo.name}
                  </Typography>
                  
                  <Box>
                    <Tooltip title={`${repo.stars} stars`}>
                      <Chip
                        icon={<StarIcon fontSize="small" />}
                        label={repo.stars}
                        size="small"
                        sx={{ 
                          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 215, 0, 0.1)' : 'rgba(255, 215, 0, 0.2)',
                          color: theme.palette.mode === 'dark' ? 'gold' : '#b3860b',
                          '& .MuiChip-icon': { 
                            color: theme.palette.mode === 'dark' ? 'gold' : '#b3860b' 
                          }
                        }}
                      />
                    </Tooltip>
                    
                    <Tooltip title="View on GitHub">
                      <IconButton 
                        size="small" 
                        sx={{ ml: 1 }}
                        href={`https://github.com/${username}/${repo.name}`}
                        target="_blank"
                      >
                        <GitHubIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Chip
                    label={repo.language}
                    size="small"
                    sx={{
                      mr: 2,
                      bgcolor: `${getLanguageColor(repo.language)}33`,
                      color: theme.palette.getContrastText(`${getLanguageColor(repo.language)}33`),
                      '&::before': {
                        content: '""',
                        display: 'inline-block',
                        width: '10px',
                        height: '10px',
                        borderRadius: '50%',
                        bgcolor: getLanguageColor(repo.language),
                        mr: 0.5
                      }
                    }}
                  />
                  
                  <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
                    <LinearProgress
                      variant="determinate"
                      value={(repo.stars / maxStars) * 100}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        bgcolor: theme.palette.background.paper,
                        width: '100%',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getLanguageColor(repo.language),
                          borderRadius: 4,
                        }
                      }}
                    />
                    <Typography variant="caption" sx={{ ml: 1, minWidth: '40px', color: theme.palette.text.secondary }}>
                      {Math.round((repo.stars / maxStars) * 100)}%
                    </Typography>
                  </Box>
                </Box>
              </Grid>
              
              {index < repositories.length - 1 && (
                <Grid item xs={12}>
                  <Box sx={{ borderBottom: `1px dashed ${theme.palette.divider}`, my: 1 }} />
                </Grid>
              )}
            </Grid>
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default RepositoryCard;