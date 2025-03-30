import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Avatar, 
  Chip, 
  Stack, 
  Grid,
  useTheme,
  Divider,
  Button,
  Tooltip,
  IconButton,
  Fade
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import BusinessIcon from '@mui/icons-material/Business';
import CodeIcon from '@mui/icons-material/Code';
import ForumIcon from '@mui/icons-material/Forum';
import EmailIcon from '@mui/icons-material/Email';
import WebIcon from '@mui/icons-material/Web';
import SkillRadarChart from './SkillRadarChart';

interface DeveloperProfile {
  name: string;
  username: string;
  avatar_url: string;
  company?: string;
  location?: string;
  bio?: string;
  email?: string;
  blog?: string;
  followers: number;
  following: number;
  public_repos: number;
  public_gists: number;
  languages: string[];
  skills: {
    codingStyle: number;
    projectComplexity: number;
    communityEngagement: number;
    documentation: number;
  };
}

interface ProfileCardProps {
  profile: DeveloperProfile;
}

const ProfileCard: React.FC<ProfileCardProps> = ({ profile }) => {
  const theme = useTheme();
  
  return (
    <Fade in={true} timeout={800}>
      <Card elevation={3} sx={{ 
        borderRadius: 2,
        overflow: 'visible',
        position: 'relative',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: theme.shadows[10]
        }
      }}>
        <Box sx={{ 
          height: 100, 
          bgcolor: theme.palette.primary.main,
          borderTopLeftRadius: 'inherit',
          borderTopRightRadius: 'inherit',
          position: 'relative',
          backgroundImage: 'linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px), linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }} />
        
        <CardContent sx={{ pb: 3, pt: 0 }}>
          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', sm: 'row' },
            alignItems: { xs: 'center', sm: 'flex-start' },
            mb: 2
          }}>
            <Avatar 
              src={profile.avatar_url} 
              alt={profile.name || profile.username}
              sx={{ 
                width: 120, 
                height: 120, 
                border: `4px solid ${theme.palette.background.paper}`,
                boxShadow: theme.shadows[3],
                mt: -8,
                mb: { xs: 2, sm: 0 },
                mr: { sm: 3 }
              }} 
            />
            
            <Box sx={{ flex: 1, textAlign: { xs: 'center', sm: 'left' } }}>
              <Typography variant="h5" component="h1" gutterBottom fontWeight="bold">
                {profile.name || profile.username}
              </Typography>
              
              <Typography 
                variant="subtitle1" 
                color="text.secondary" 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  justifyContent: { xs: 'center', sm: 'flex-start' }
                }}
              >
                <GitHubIcon fontSize="small" sx={{ mr: 0.5 }} />
                {profile.username}
              </Typography>
              
              {profile.bio && (
                <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  sx={{ 
                    mt: 1, 
                    fontStyle: 'italic',
                    maxWidth: '500px',
                    lineHeight: 1.6
                  }}
                >
                  "{profile.bio}"
                </Typography>
              )}
            </Box>
            
            <Box sx={{ 
              mt: { xs: 2, sm: 0 }, 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'flex-end'
            }}>
              <Button 
                variant="contained" 
                startIcon={<GitHubIcon />}
                href={`https://github.com/${profile.username}`}
                target="_blank"
                sx={{ mb: 1 }}
              >
                View Profile
              </Button>
              
              {profile.email && (
                <Tooltip title="Send Email">
                  <IconButton 
                    size="small" 
                    color="primary"
                    href={`mailto:${profile.email}`}
                    sx={{ mr: 1 }}
                  >
                    <EmailIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
              
              {profile.blog && (
                <Tooltip title="Visit Website">
                  <IconButton 
                    size="small" 
                    color="primary"
                    href={profile.blog.startsWith('http') ? profile.blog : `https://${profile.blog}`}
                    target="_blank"
                  >
                    <WebIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Box>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={6}>
              <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
                {profile.location && (
                  <Chip 
                    icon={<LocationOnIcon />} 
                    label={profile.location}
                    size="small"
                    sx={{ bgcolor: 'background.paper' }}
                  />
                )}
                
                {profile.company && (
                  <Chip 
                    icon={<BusinessIcon />} 
                    label={profile.company}
                    size="small"
                    sx={{ bgcolor: 'background.paper' }}
                  />
                )}
              </Stack>
              
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center', p: 1 }}>
                    <Typography variant="h6" color="primary.main">
                      {profile.followers}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Followers
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center', p: 1 }}>
                    <Typography variant="h6" color="primary.main">
                      {profile.following}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Following
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center', p: 1 }}>
                    <Typography variant="h6" color="primary.main">
                      {profile.public_repos}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Repos
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center', p: 1 }}>
                    <Typography variant="h6" color="primary.main">
                      {profile.public_gists}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Gists
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Top Languages
              </Typography>
              
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {profile.languages.map((lang, index) => (
                  <Chip 
                    key={index}
                    icon={<CodeIcon />}
                    label={lang}
                    size="small"
                    color={index === 0 ? "primary" : index === 1 ? "secondary" : "default"}
                    variant={index < 2 ? "filled" : "outlined"}
                    sx={{ mb: 1 }}
                  />
                ))}
              </Stack>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography variant="subtitle2" align="center" gutterBottom>
                <ForumIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                Skill Assessment
              </Typography>
              
              <SkillRadarChart skills={profile.skills} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Fade>
  );
};

export default ProfileCard;