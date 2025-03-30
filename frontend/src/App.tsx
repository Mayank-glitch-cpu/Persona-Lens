import { useState, useEffect } from 'react'
import { 
  Box, 
  Container, 
  TextField, 
  Typography, 
  Paper, 
  Tab, 
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Rating,
  Alert,
  Button,
  List,
  ListItem,
  ListItemText,
  Divider,
  Card,
  CardContent,
  IconButton,
  Avatar,
  Grid,
  Tooltip,
  InputAdornment,
  CssBaseline,
  ThemeProvider,
  PaletteMode,
  useMediaQuery,
  Fade,
  Zoom,
  Badge,
  Stack,
  Skeleton
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search';
import GitHubIcon from '@mui/icons-material/GitHub';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import PersonIcon from '@mui/icons-material/Person';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import CodeIcon from '@mui/icons-material/Code';
import VerifiedIcon from '@mui/icons-material/Verified';
import BarChartIcon from '@mui/icons-material/BarChart';
import HistoryIcon from '@mui/icons-material/History';
import { searchCandidates, checkApiHealth } from './services/api'
import { getTheme } from './theme'
import './App.css'

// Mock data to use as fallback when API fails
const MOCK_PROFILES = [
  {
    name: "Jane Python",
    username: "janepython",
    githubUrl: "https://github.com/janepython",
    experience: "Senior Data Scientist (8.5 years)",
    expertise: ["Python", "Machine Learning", "TensorFlow", "Data Analysis", "NLP"],
    contributions: 1250,
    followers: 350,
    score: 0.92,
    analysis: {
      codingStyle: 4.7,
      projectComplexity: 4.5,
      communityEngagement: 4.2,
      documentation: 4.8
    },
    recentProjects: [
      { name: "NLP-Text-Classifier", stars: 850, language: "Python" },
      { name: "Pandemic-Data-Analysis", stars: 620, language: "Python" },
      { name: "ML-Workflow-Tools", stars: 430, language: "Python" }
    ],
    strengths: [
      "Exceptional documentation practices",
      "Strong machine learning expertise",
      "Active open source contributor"
    ],
    areasOfImprovement: [
      "Could expand test coverage",
      "Limited experience with front-end technologies"
    ]
  },
  {
    name: "Alex DeepLearner",
    username: "alexdeep",
    githubUrl: "https://github.com/alexdeep",
    experience: "AI Research Engineer (7.2 years)",
    expertise: ["Python", "Deep Learning", "PyTorch", "Computer Vision", "Research"],
    contributions: 890,
    followers: 410,
    score: 0.87,
    analysis: {
      codingStyle: 4.3,
      projectComplexity: 4.9,
      communityEngagement: 3.8,
      documentation: 3.9
    },
    recentProjects: [
      { name: "Vision-Transformer-Implementation", stars: 1200, language: "Python" },
      { name: "Deep-RL-Framework", stars: 750, language: "Python" },
      { name: "Research-Paper-Implementations", stars: 520, language: "Python" }
    ],
    strengths: [
      "Cutting-edge AI research contributions",
      "Complex problem solver",
      "High-quality implementations of research papers"
    ],
    areasOfImprovement: [
      "Documentation could be more comprehensive",
      "More frequent community engagement would be beneficial"
    ]
  },
  {
    name: "Sam DataEngineer",
    username: "samdata",
    githubUrl: "https://github.com/samdata",
    experience: "Data Engineer (5.5 years)",
    expertise: ["Python", "SQL", "Apache Spark", "Data Engineering", "ETL"],
    contributions: 680,
    followers: 180,
    score: 0.82,
    analysis: {
      codingStyle: 4.5,
      projectComplexity: 4.0,
      communityEngagement: 3.5,
      documentation: 4.3
    },
    recentProjects: [
      { name: "Streaming-ETL-Pipeline", stars: 420, language: "Python" },
      { name: "Data-Quality-Framework", stars: 380, language: "Python" },
      { name: "SQL-Query-Optimizer", stars: 290, language: "SQL" }
    ],
    strengths: [
      "Efficient data processing solutions",
      "Strong data modeling skills",
      "Consistent code quality"
    ],
    areasOfImprovement: [
      "More public speaking or blog posts to share knowledge",
      "Limited contribution to larger open-source projects"
    ]
  }
];

interface PersonaData {
  name: string;
  username: string;
  githubUrl: string;
  experience: string;
  expertise: string[];
  contributions: number;
  followers: number;
  score: number;
  analysis: {
    codingStyle: number;
    projectComplexity: number;
    communityEngagement: number;
    documentation: number;
  };
  recentProjects: Array<{
    name: string;
    stars: number;
    language: string;
  }>;
  strengths: string[];
  areasOfImprovement: string[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<PersonaData[]>([])
  const [tabValue, setTabValue] = useState(0)
  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const [rawOutput, setRawOutput] = useState<string>('')
  const [backendAvailable, setBackendAvailable] = useState<boolean>(false)
  const [useMockData, setUseMockData] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [apiHealthy, setApiHealthy] = useState<boolean>(false);
  const [apiChecked, setApiChecked] = useState<boolean>(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showRawOutput, setShowRawOutput] = useState<boolean>(false);
  const [activeProfileIndex, setActiveProfileIndex] = useState<number | null>(null);
  const [themeMode, setThemeMode] = useState<PaletteMode>('light')
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  // Initialize theme based on user's system preference
  useEffect(() => {
    setThemeMode(prefersDarkMode ? 'dark' : 'light');
  }, [prefersDarkMode]);

  // Theme toggle handler
  const toggleTheme = () => {
    setThemeMode(prevMode => prevMode === 'light' ? 'dark' : 'light');
  };

  const theme = getTheme(themeMode);

  // Check API health on component mount and when toggling mock data
  useEffect(() => {
    const checkHealth = async () => {
      try {
        setLoading(true);
        console.log("Checking API health on mount or when toggling mock data...");
        const isHealthy = await checkApiHealth();
        console.log('API health status:', isHealthy ? 'Healthy' : 'Unhealthy');
        setApiHealthy(isHealthy);
          
        // Only set mock data mode automatically if backend is unhealthy and we're not already in mock mode
        if (!isHealthy && !useMockData) {
          console.log("API unhealthy, enabling mock data mode");
          setUseMockData(true);
          setError("Backend API is not accessible. Using demo data mode.");
        }
      } catch (error) {
        console.error('Error checking API health:', error);
        setApiHealthy(false);
        if (!useMockData) {
          setUseMockData(true);
          setError("Error checking API health. Using demo data mode.");
        }
      } finally {
        setApiChecked(true);
        setLoading(false);
      }
    };
    
    checkHealth();
  }, [useMockData]); // Re-run when mockData toggle changes
  
  // When user toggles mock data mode, re-check API if switching to real API
  useEffect(() => {
    const verifyApiConnection = async () => {
      if (!useMockData && !apiHealthy) {
        try {
          setIsLoading(true);
          const healthy = await checkApiHealth();
          setApiHealthy(healthy);
          
          if (!healthy) {
            // If API is still not healthy, revert to mock data
            console.warn('API still not accessible, reverting to mock data');
            setUseMockData(true);
          }
        } catch (error) {
          console.error('Error verifying API connection:', error);
          setUseMockData(true);
        } finally {
          setIsLoading(false);
        }
      }
    };
    
    if (apiChecked) {
      verifyApiConnection();
    }
  }, [useMockData, apiHealthy, apiChecked]);

  const parseRagOutput = (ragOutput: string): PersonaData[] => {
    try {
      console.log("Parsing RAG output:", ragOutput);
      setRawOutput(ragOutput);
      
      // Splitting the RAG output by numeric indicators (e.g., "1. ", "2. ")
      const candidateBlocks = ragOutput.split(/(?=\d+\.\s+)/);
      
      return candidateBlocks.filter(block => block.trim()).map(block => {
        // Extract name
        const nameMatch = block.match(/\d+\.\s+([^\n]+)/);
        const name = nameMatch ? nameMatch[1].replace(/\*\*/g, '').trim() : "Unknown Developer";
        
        // Extract username
        const usernameMatch = block.match(/Username:\s*([^\n]+)/i) || 
                             block.match(/GitHub:\s*(?:https?:\/\/github\.com\/)?([^\s\/\n]+)/i);
        const username = usernameMatch ? usernameMatch[1].trim() : name.replace(/\s+/g, '').toLowerCase();
        
        // Extract GitHub URL
        const githubMatch = block.match(/GitHub(?:\s+URL)?:\s*(https?:\/\/[^\s\n]+)/i);
        const githubUrl = githubMatch ? 
                         githubMatch[1].trim() : 
                         `https://github.com/${username}`;
        
        // Extract experience
        const experienceMatch = block.match(/Experience:\s*([^\n]+)/i);
        const experience = experienceMatch ? experienceMatch[1].trim() : "Not specified";
        
        // Extract languages/expertise
        const expertiseMatch = block.match(/(?:Languages|Expertise|Skills):\s*([^\n]+)/i);
        const expertise = expertiseMatch ? 
                         expertiseMatch[1].split(/,\s*/).map(s => s.trim()) : 
                         [];
        
        // Extract contributions
        const contributionsMatch = block.match(/Contributions:\s*(\d+)/i);
        const contributions = contributionsMatch ? parseInt(contributionsMatch[1]) : 0;
        
        // Extract followers
        const followersMatch = block.match(/Followers:\s*(\d+)/i);
        const followers = followersMatch ? parseInt(followersMatch[1]) : 0;
        
        // Extract score
        const scoreMatch = block.match(/(?:Match|Relevance)\s+Score:\s*([\d.]+)/i);
        const score = scoreMatch ? parseFloat(scoreMatch[1]) : 
                     (block.match(/(\d+)%\s+match/i) ? 
                      parseInt(block.match(/(\d+)%\s+match/i)![1])/100 : 0.8);
        
        // Extract strengths
        let strengths: string[] = [];
        const strengthsSection = block.match(/(?:Key\s+)?Strengths:([^]*?)(?=Areas|Improvement|$)/i);
        if (strengthsSection) {
          strengths = strengthsSection[1]
            .split(/[\n\r]+/)
            .map(line => line.replace(/^[-•*]\s*/, '').trim())
            .filter(Boolean);
        }
        
        // Extract areas of improvement
        let areasOfImprovement: string[] = [];
        const improvementSection = block.match(/(?:Areas\s+(?:for\s+)?Improvement|Weaknesses):([^]*?)(?=\d+\.|$)/i);
        if (improvementSection) {
          areasOfImprovement = improvementSection[1]
            .split(/[\n\r]+/)
            .map(line => line.replace(/^[-•*]\s*/, '').trim())
            .filter(Boolean);
        }
        
        // Generate analysis metrics based on text
        const hasHighCodingScore = block.toLowerCase().includes('high quality code') || 
                                  block.toLowerCase().includes('clean code') ||
                                  block.toLowerCase().includes('excellent coding');
                                  
        const hasHighProjectComplexity = block.toLowerCase().includes('complex') || 
                                       block.toLowerCase().includes('sophisticated') ||
                                       block.toLowerCase().includes('advanced');
                                       
        const hasHighCommunity = block.toLowerCase().includes('active contributor') || 
                               block.toLowerCase().includes('community') ||
                               block.toLowerCase().includes('open source');
                               
        const hasHighDocumentation = block.toLowerCase().includes('well documented') || 
                                   block.toLowerCase().includes('documentation');
        
        // Create sample projects
        const recentProjects = [];
        const projectsSection = block.match(/(?:Projects|Repositories):([^]*?)(?=\d+\.|$)/i);
        if (projectsSection) {
          const projectLines = projectsSection[1]
            .split(/[\n\r]+/)
            .map(line => line.replace(/^[-•*]\s*/, '').trim())
            .filter(Boolean);
            
          for (const line of projectLines) {
            const projectMatch = line.match(/([^(]+)(?:\((\d+)\s*stars\)?)?/i);
            if (projectMatch) {
              recentProjects.push({
                name: projectMatch[1].trim(),
                stars: projectMatch[2] ? parseInt(projectMatch[2]) : 
                       Math.floor(Math.random() * 500) + 50, // Random stars if not specified
                language: expertise[0] || "Unknown" // Use first expertise as language
              });
            }
          }
        }
        
        // If no projects found, create some based on expertise
        if (recentProjects.length === 0 && expertise.length > 0) {
          // Generate names based on expertise
          const topics = ["Analysis", "Framework", "Toolkit", "App", "Library", "API"];
          for (let i = 0; i < Math.min(3, expertise.length); i++) {
            recentProjects.push({
              name: `${expertise[i]}-${topics[Math.floor(Math.random() * topics.length)]}`,
              stars: Math.floor(Math.random() * 500) + 50,
              language: expertise[i]
            });
          }
        }
        
        return {
          name,
          username,
          githubUrl,
          experience,
          expertise,
          contributions,
          followers,
          score,
          analysis: {
            codingStyle: hasHighCodingScore ? 4.5 : 3.8,
            projectComplexity: hasHighProjectComplexity ? 4.7 : 3.5,
            communityEngagement: hasHighCommunity ? 4.6 : 3.2,
            documentation: hasHighDocumentation ? 4.8 : 3.0
          },
          recentProjects,
          strengths,
          areasOfImprovement: areasOfImprovement.length > 0 ? areasOfImprovement : ["Could improve documentation", "More test coverage would be beneficial"]
        };
      });
    } catch (err) {
      console.error('Error parsing RAG output:', err);
      console.log('Raw output that failed parsing:', ragOutput);
      return [];
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    setActiveProfileIndex(null);
    
    // Add to search history if not a duplicate
    if (!searchHistory.includes(query)) {
      setSearchHistory([query, ...searchHistory.slice(0, 4)]);
    }
    
    try {
      // If not using mock data, try to use the real API
      if (!useMockData) {
        try {
          console.log('Attempting to search with real API');
          const response = await searchCandidates(query, sessionId);
          
          if (response && response.rag_output) {
            console.log('Received valid response from API');
            const parsedResults = parseRagOutput(response.rag_output);
            setResults(parsedResults);
            setRawOutput(response.rag_output);
            setSessionId(response.session_id);
            
            if (parsedResults.length === 0) {
              setError('No developers matching your query were found. Please try different search terms.');
            } else if (response.is_fallback) {
              setError('Using backend fallback data. The search engine returned approximate results.');
            }
            
            setLoading(false);
            return;
          }
        } catch (apiError: any) {
          console.error('API search error:', apiError);
          setError(`API connection error: ${apiError.message}. Using demo data instead.`);
          // Will continue to use mock data as fallback
          setApiHealthy(false);
        }
      }
      
      // If we get here, either useMockData is true or the API request failed
      console.log('Using mock data for search results');
      
      // Simple filtering logic for mock data
      const searchTerms = query.toLowerCase().split(' ').filter(Boolean);
      let filteredResults = MOCK_PROFILES;
      
      if (searchTerms.length > 0) {
        filteredResults = MOCK_PROFILES.filter(profile => {
          return searchTerms.some(term => 
            profile.name.toLowerCase().includes(term) ||
            profile.expertise.some(skill => skill.toLowerCase().includes(term))
          );
        });
      }
      
      // Set results with mock data
      setResults(filteredResults);
      setRawOutput("Using mock data as backend API is not available or returned an error");
      
      // Show a warning that we're using mock data
      setError("Using demo data: The backend API is not available or returned an error");
      
    } catch (err: any) {
      console.error('Search process error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while searching');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleMockData = async () => {
    if (useMockData) {
      // Switching to real API - check health first
      try {
        setIsLoading(true);
        const healthy = await checkApiHealth();
        setApiHealthy(healthy);
        
        if (healthy) {
          setUseMockData(false);
          console.log('Switched to real API mode');
        } else {
          console.warn('Cannot switch to real API - backend not accessible');
          setError('Cannot connect to backend API. Still using demo data.');
        }
      } catch (error) {
        console.error('Error checking API when toggling mode:', error);
      } finally {
        setIsLoading(false);
      }
    } else {
      // Switching to mock data - simple toggle
      setUseMockData(true);
      console.log('Switched to mock data mode');
    }
  };

  const resetResults = () => {
    setResults([]);
    setQuery('');
    setRawOutput('');
    setActiveProfileIndex(null);
  }
  
  const handleHistoryItemClick = (historyItem: string) => {
    setQuery(historyItem);
  }
  
  const handleProfileSelect = (index: number) => {
    setActiveProfileIndex(index);
    setTabValue(0); // Reset to overview tab when selecting a new profile
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ pt: 2, pb: 4 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Paper 
            elevation={3} 
            sx={{ 
              mb: 3, 
              p: 2, 
              borderRadius: 2,
              background: themeMode === 'dark' ? 
                'linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%)' : 
                'linear-gradient(135deg, #f2f6fb 0%, #ffffff 100%)',
              transition: 'background 0.3s ease-in-out'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Zoom in={true} style={{ transitionDelay: '250ms' }}>
                  <Avatar sx={{ mr: 2, bgcolor: theme.palette.primary.main }}>
                    <PersonIcon />
                  </Avatar>
                </Zoom>
                <Fade in={true} style={{ transitionDelay: '500ms' }}>
                  <Typography variant="h4" component="h1" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>
                    Persona<Box component="span" sx={{ color: theme.palette.primary.main }}>Lens</Box>
                  </Typography>
                </Fade>
              </Box>
              <Box>
                <Tooltip title={`Switch to ${themeMode === 'light' ? 'dark' : 'light'} mode`}>
                  <IconButton onClick={toggleTheme} color="primary" sx={{ ml: 1 }}>
                    {themeMode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
                  </IconButton>
                </Tooltip>
                <Tooltip title={apiHealthy ? 'Connected to backend' : 'Using demo data'}>
                  <Badge
                    variant="dot"
                    color={apiHealthy ? 'success' : 'error'}
                    sx={{ ml: 1 }}
                  >
                    <IconButton 
                      color="primary"
                      onClick={() => setUseMockData(!useMockData && apiHealthy)}
                      disabled={!apiHealthy && !useMockData}
                    >
                      <CodeIcon />
                    </IconButton>
                  </Badge>
                </Tooltip>
              </Box>
            </Box>

            <Box component="form" onSubmit={handleSearch} sx={{ width: '100%' }}>
              <TextField
                fullWidth
                variant="outlined"
                label="Find developers by expertise, skills, or project type..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                sx={{ 
                  borderRadius: 2,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                  },
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderWidth: 2
                  }
                }}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <Button 
                        variant="contained" 
                        type="submit" 
                        disabled={loading || !query.trim()}
                        sx={{ 
                          borderRadius: '8px', 
                          px: 3,
                          color: '#fff',
                          boxShadow: 4,
                          transition: 'transform 0.2s',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: 6
                          }
                        }}
                      >
                        <SearchIcon sx={{ mr: 1 }} />
                        Search
                      </Button>
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            {searchHistory.length > 0 && (
              <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Chip 
                  icon={<HistoryIcon />} 
                  label="Recent Searches:"
                  variant="outlined"
                  size="small"
                />
                {searchHistory.map((term, idx) => (
                  <Chip
                    key={idx}
                    label={term}
                    size="small"
                    onClick={() => {
                      setQuery(term);
                      handleSearch(new Event('submit') as unknown as React.FormEvent);
                    }}
                    sx={{ 
                      transition: 'all 0.2s',
                      '&:hover': {
                        backgroundColor: theme.palette.primary.main + '33',
                        transform: 'translateY(-2px)'
                      }
                    }}
                  />
                ))}
              </Box>
            )}

            {error && (
              <Alert 
                severity="warning" 
                sx={{ mt: 2, borderRadius: 2, animation: 'fadeIn 0.5s' }}
                onClose={() => setError(null)}
              >
                {error}
              </Alert>
            )}

            {!apiChecked && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Checking API connection...
                </Typography>
                <LinearProgress sx={{ mt: 1, borderRadius: 1 }} />
              </Box>
            )}
          </Paper>

          {loading && (
            <Box sx={{ mt: 2, mb: 4 }}>
              <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 2, mb: 2 }} />
              <Grid container spacing={3}>
                {[1, 2, 3].map((item) => (
                  <Grid item xs={12} md={4} key={item}>
                    <Skeleton variant="rectangular" height={240} sx={{ borderRadius: 2 }} />
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          {/* Results Section */}
          {!loading && results.length > 0 && (
            <>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5">
                  {results.length} developer{results.length > 1 ? 's' : ''} found
                </Typography>
                <Chip 
                  label={`Query: "${query}"`} 
                  variant="outlined" 
                  onDelete={resetResults} 
                  color="primary"
                  sx={{ borderRadius: '16px' }}
                />
              </Box>
              
              {/* Two-column layout for larger screens */}
              <Grid container spacing={3}>
                {/* Developer list (sidebar) */}
                <Grid item xs={12} md={4}>
                  <Paper 
                    elevation={1} 
                    sx={{ 
                      borderRadius: 2, 
                      overflow: 'hidden',
                      height: '100%',
                      maxHeight: '600px',
                      overflowY: 'auto'
                    }}
                  >
                    <Typography 
                      variant="subtitle1" 
                      sx={{ p: 2, borderBottom: '1px solid #eee', fontWeight: 'medium' }}
                    >
                      Developer Results
                    </Typography>
                    <List sx={{ p: 0 }}>
                      {results.map((result, index) => (
                        <ListItem 
                          key={index} 
                          button 
                          onClick={() => handleProfileSelect(index)}
                          selected={activeProfileIndex === index}
                          sx={{ 
                            borderLeft: activeProfileIndex === index ? '4px solid #3f51b5' : '4px solid transparent',
                            transition: 'all 0.2s ease',
                            '&:hover': {
                              backgroundColor: '#f5f5f5'
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', width: '100%', alignItems: 'center' }}>
                            <Avatar 
                              sx={{ 
                                mr: 2, 
                                bgcolor: `hsl(${(index * 55) % 360}, 70%, 50%)`,
                                width: 40,
                                height: 40
                              }}
                            >
                              {result.name.charAt(0)}
                            </Avatar>
                            <Box sx={{ overflow: 'hidden' }}>
                              <Typography 
                                variant="body1" 
                                sx={{ 
                                  fontWeight: activeProfileIndex === index ? 'bold' : 'regular',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis'
                                }}
                              >
                                {result.name}
                              </Typography>
                              <Typography 
                                variant="body2" 
                                color="text.secondary"
                                sx={{
                                  fontSize: '0.8rem',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis'
                                }}
                              >
                                {result.experience}
                              </Typography>
                            </Box>
                            <Box 
                              sx={{ 
                                ml: 'auto', 
                                display: 'flex', 
                                alignItems: 'center',
                                whiteSpace: 'nowrap'
                              }}
                            >
                              <Tooltip title="Match Score">
                                <Chip 
                                  label={`${(result.score * 100).toFixed(0)}%`} 
                                  size="small" 
                                  color={result.score > 0.85 ? "success" : "primary"}
                                  sx={{ fontWeight: 'bold' }}
                                />
                              </Tooltip>
                            </Box>
                          </Box>
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                </Grid>
                
                {/* Profile detail */}
                <Grid item xs={12} md={8}>
                  {activeProfileIndex !== null ? (
                    <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'hidden' }}>
                      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                        <Tabs 
                          value={tabValue} 
                          onChange={(_, newValue) => setTabValue(newValue)}
                          sx={{
                            '.MuiTab-root': {
                              minWidth: '120px',
                              py: 2
                            }
                          }}
                        >
                          <Tab label="Overview" />
                          <Tab label="Detailed Analysis" />
                          <Tab label="Projects & Contributions" />
                        </Tabs>
                      </Box>

                      <TabPanel value={tabValue} index={0}>
                        <Box sx={{ mb: 3, display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: { sm: 'center' }, gap: 2 }}>
                          <Avatar 
                            sx={{ 
                              width: 80, 
                              height: 80,
                              bgcolor: `hsl(${(activeProfileIndex * 55) % 360}, 70%, 50%)`,
                              fontSize: '2rem',
                              fontWeight: 'bold'
                            }}
                          >
                            {results[activeProfileIndex].name.charAt(0)}
                          </Avatar>
                          <Box>
                            <Typography variant="h5" gutterBottom>{results[activeProfileIndex].name}</Typography>
                            <Typography 
                              variant="subtitle1" 
                              color="text.secondary" 
                              component="a" 
                              href={results[activeProfileIndex].githubUrl} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              sx={{ 
                                textDecoration: 'none',
                                '&:hover': { textDecoration: 'underline' }
                              }}
                            >
                              @{results[activeProfileIndex].username}
                            </Typography>
                          </Box>
                          <Box sx={{ ml: 'auto', display: { xs: 'none', sm: 'block' } }}>
                            <Button 
                              variant="outlined" 
                              color="primary"
                              href={results[activeProfileIndex].githubUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              sx={{ 
                                borderRadius: 2,
                                textTransform: 'none'
                              }}
                            >
                              View GitHub Profile
                            </Button>
                          </Box>
                        </Box>

                        <TableContainer sx={{ mb: 3 }}>
                          <Table>
                            <TableBody>
                              <TableRow>
                                <TableCell component="th" scope="row" width="200">Match Score</TableCell>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <LinearProgress 
                                      variant="determinate" 
                                      value={results[activeProfileIndex].score * 100} 
                                      sx={{ 
                                        flexGrow: 1,
                                        height: 10,
                                        borderRadius: 5,
                                        backgroundColor: '#e0e0e0',
                                        '& .MuiLinearProgress-bar': {
                                          borderRadius: 5,
                                          background: `linear-gradient(90deg, #2196f3 0%, #3f51b5 100%)`
                                        }
                                      }}
                                    />
                                    <Typography variant="body2" fontWeight="bold">
                                      {(results[activeProfileIndex].score * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell component="th" scope="row">Experience</TableCell>
                                <TableCell>{results[activeProfileIndex].experience}</TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell component="th" scope="row">Expertise</TableCell>
                                <TableCell>
                                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                    {results[activeProfileIndex].expertise.map((skill, i) => (
                                      <Chip 
                                        key={i} 
                                        label={skill} 
                                        size="small" 
                                        sx={{ borderRadius: '16px' }}
                                      />
                                    ))}
                                  </Box>
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell component="th" scope="row">GitHub Stats</TableCell>
                                <TableCell>
                                  <Typography variant="body2">
                                    {results[activeProfileIndex].contributions.toLocaleString()} contributions · {results[activeProfileIndex].followers.toLocaleString()} followers
                                  </Typography>
                                </TableCell>
                              </TableRow>
                            </TableBody>
                          </Table>
                        </TableContainer>
                        
                        {/* Strengths & Areas for Improvement (mini-summary) */}
                        <Grid container spacing={2}>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="subtitle2" gutterBottom>Key Strengths</Typography>
                            <List dense>
                              {results[activeProfileIndex].strengths.slice(0, 2).map((strength, i) => (
                                <ListItem key={i} sx={{ py: 0.5 }}>
                                  <ListItemText primary={strength} />
                                </ListItem>
                              ))}
                            </List>
                          </Grid>
                          <Grid item xs={12} sm={6}>
                            <Typography variant="subtitle2" gutterBottom>Areas for Improvement</Typography>
                            <List dense>
                              {results[activeProfileIndex].areasOfImprovement.slice(0, 2).map((area, i) => (
                                <ListItem key={i} sx={{ py: 0.5 }}>
                                  <ListItemText primary={area} />
                                </ListItem>
                              ))}
                            </List>
                          </Grid>
                        </Grid>
                      </TabPanel>

                      <TabPanel value={tabValue} index={1}>
                        <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                        <Box sx={{ mb: 4, px: 2 }}>
                          {Object.entries(results[activeProfileIndex].analysis).map(([key, value]) => (
                            <Box key={key} sx={{ mb: 2 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2">
                                  {key.replace(/([A-Z])/g, ' $1').trim()}
                                </Typography>
                                <Typography variant="body2" fontWeight="medium">
                                  {value.toFixed(1)}/5.0
                                </Typography>
                              </Box>
                              <LinearProgress 
                                variant="determinate" 
                                value={(value / 5) * 100}
                                sx={{ 
                                  height: 8, 
                                  borderRadius: 4,
                                  backgroundColor: '#e0e0e0',
                                  '& .MuiLinearProgress-bar': {
                                    borderRadius: 4,
                                    background: key === 'codingStyle' ? 'linear-gradient(90deg, #42a5f5 0%, #1976d2 100%)' :
                                              key === 'projectComplexity' ? 'linear-gradient(90deg, #7cb342 0%, #558b2f 100%)' :
                                              key === 'communityEngagement' ? 'linear-gradient(90deg, #ffb74d 0%, #f57c00 100%)' :
                                              'linear-gradient(90deg, #ba68c8 0%, #7b1fa2 100%)'
                                  }
                                }}
                              />
                            </Box>
                          ))}
                        </Box>

                        <Box sx={{ mt: 4 }}>
                          <Typography variant="h6" gutterBottom>Key Strengths</Typography>
                          {results[activeProfileIndex].strengths.map((strength, i) => (
                            <Alert 
                              key={i} 
                              severity="success" 
                              sx={{ 
                                mb: 1,
                                borderRadius: '8px',
                                '.MuiAlert-icon': {
                                  alignItems: 'center'
                                }
                              }}
                            >
                              {strength}
                            </Alert>
                          ))}
                        </Box>

                        <Box sx={{ mt: 4 }}>
                          <Typography variant="h6" gutterBottom>Areas for Improvement</Typography>
                          {results[activeProfileIndex].areasOfImprovement.map((area, i) => (
                            <Alert 
                              key={i} 
                              severity="info" 
                              sx={{ 
                                mb: 1,
                                borderRadius: '8px'
                              }}
                            >
                              {area}
                            </Alert>
                          ))}
                        </Box>
                      </TabPanel>

                      <TabPanel value={tabValue} index={2}>
                        <Typography variant="h6" gutterBottom>Recent Projects</Typography>
                        <Grid container spacing={2}>
                          {results[activeProfileIndex].recentProjects.map((project, i) => (
                            <Grid item xs={12} sm={6} key={i}>
                              <Card 
                                elevation={0} 
                                sx={{ 
                                  border: '1px solid #eee',
                                  height: '100%',
                                  borderRadius: 2
                                }}
                              >
                                <CardContent>
                                  <Typography variant="h6" gutterBottom>{project.name}</Typography>
                                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Chip 
                                      label={project.language} 
                                      size="small" 
                                      sx={{ borderRadius: '16px' }}
                                    />
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                      <Typography variant="body2">★</Typography>
                                      <Typography variant="body2" fontWeight="medium">
                                        {project.stars.toLocaleString()}
                                      </Typography>
                                    </Box>
                                  </Box>
                                </CardContent>
                              </Card>
                            </Grid>
                          ))}
                        </Grid>
                      </TabPanel>
                    </Paper>
                  ) : (
                    <Box 
                      sx={{ 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        height: '100%',
                        minHeight: '300px',
                        bgcolor: '#f9f9f9',
                        borderRadius: 2,
                        p: 4
                      }}
                    >
                      <Typography variant="h6" gutterBottom>Select a Developer Profile</Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Click on a developer from the list to view their detailed profile information.
                      </Typography>
                    </Box>
                  )}
                </Grid>
              </Grid>
            </>
          )}

          {!loading && results.length === 0 && query && (
            <Alert 
              severity="info" 
              sx={{ 
                mb: 4, 
                p: 3, 
                borderRadius: 2,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}
            >
              <Typography variant="h6" gutterBottom>No results found</Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Try refining your search or using different terms to find developers.
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
                <Chip 
                  label="Python developers" 
                  size="small" 
                  clickable
                  onClick={() => setQuery("Python developers")}
                />
                <Chip 
                  label="JavaScript experts" 
                  size="small" 
                  clickable
                  onClick={() => setQuery("JavaScript experts")}
                />
                <Chip 
                  label="Data engineers" 
                  size="small" 
                  clickable
                  onClick={() => setQuery("Data engineers")}
                />
              </Box>
            </Alert>
          )}
          
          {/* Raw output toggle */}
          {rawOutput && (
            <Card sx={{ mt: 4, p: 2, borderRadius: 2 }}>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6">API Raw Response:</Typography>
                  <Button 
                    size="small" 
                    onClick={() => setShowRawOutput(!showRawOutput)}
                    variant="outlined"
                  >
                    {showRawOutput ? 'Hide Details' : 'Show Details'}
                  </Button>
                </Box>
                
                {showRawOutput && (
                  <Box component="pre" sx={{ 
                    whiteSpace: 'pre-wrap', 
                    overflow: 'auto',
                    bgcolor: '#f5f5f5',
                    p: 2,
                    borderRadius: 1,
                    maxHeight: '300px',
                    fontSize: '0.75rem'
                  }}>
                    {rawOutput}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  )
}

export default App
