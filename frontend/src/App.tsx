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
  CardContent
} from '@mui/material'
import { searchCandidates } from './services/api'
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

  // Add a backend availability check on component mount
  useEffect(() => {
    const checkBackendAvailability = async () => {
      try {
        const response = await fetch('http://10.139.126.48:5000/api/health');
        if (response.ok) {
          console.log('Backend server is available');
          setBackendAvailable(true);
        } else {
          console.error('Backend server returned an error response');
          setBackendAvailable(false);
        }
      } catch (error) {
        console.error('Error checking backend availability:', error);
        setBackendAvailable(false);
      }
    };

    checkBackendAvailability();
  }, []);

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
    e.preventDefault()
    setLoading(true)
    setError(null)
    
    try {
      // First try to get real data from the API
      if (!useMockData) {
        try {
          // Call the actual backend API
          console.log("Sending search request for query:", query);
          const response = await searchCandidates(query, sessionId);
          setSessionId(response.session_id);
          
          console.log("API Response received:", response);
          
          // If the response is valid and has RAG output
          if (response && response.rag_output) {
            const parsedResults = parseRagOutput(response.rag_output);
            setResults(parsedResults);
            
            if (parsedResults.length === 0) {
              setError("No developers matching your query were found. Please try different search terms.");
            }
            
            // Done processing successfully
            setLoading(false);
            return;
          }
        } catch (apiError) {
          console.error("API error, falling back to mock data:", apiError);
          // Will continue to mock data below
        }
      }
      
      // Fallback to mock data if API failed or mock data was requested
      console.log("Using mock data as fallback");
      
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
      console.error("Error in search process:", err);
      setError(err instanceof Error ? err.message : 'An error occurred while searching');
      setResults([]);
    } finally {
      setLoading(false)
    }
  }

  const toggleMockData = () => {
    setUseMockData(!useMockData);
  }

  const resetResults = () => {
    setResults([]);
    setQuery('');
    setRawOutput('');
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'transparent' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Persona-Lens
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          AI-Powered GitHub Developer Profile Analysis
        </Typography>
        
        <Box component="form" onSubmit={handleSearch} sx={{ mt: 4, display: 'flex', gap: 2 }}>
          <TextField
            fullWidth
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter search query (e.g., 'experienced Python developer with ML background')"
            variant="outlined"
            sx={{ bgcolor: 'background.paper' }}
          />
          <Button 
            type="submit" 
            variant="contained" 
            color="primary"
            disabled={loading}
          >
            Search
          </Button>
          {results.length > 0 && (
            <Button 
              variant="outlined" 
              color="secondary"
              onClick={resetResults}
            >
              Clear
            </Button>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Button 
            variant="outlined" 
            size="small" 
            color={useMockData ? "success" : "info"}
            onClick={toggleMockData}
            sx={{ mb: 2 }}
          >
            {useMockData ? "Using Demo Data" : "Use Real API"}
          </Button>
          {!backendAvailable && (
            <Alert 
              severity="warning" 
              sx={{ ml: 2, flex: 1 }}
            >
              Backend connection issue detected. Using demo data instead.
            </Alert>
          )}
        </Box>
      </Paper>

      {loading && <LinearProgress sx={{ mb: 4 }} />}
      
      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {!loading && results.length > 0 && (
        <>
          <Typography variant="h5" gutterBottom>
            {results.length} developer{results.length > 1 ? 's' : ''} found matching your criteria
          </Typography>
          
          {results.map((result, index) => (
            <Paper key={index} elevation={1} sx={{ mb: 4, overflow: 'hidden' }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                  <Tab label="Overview" />
                  <Tab label="Detailed Analysis" />
                  <Tab label="Projects & Contributions" />
                </Tabs>
              </Box>

              <TabPanel value={tabValue} index={0}>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h5" gutterBottom>{result.name}</Typography>
                  <Typography variant="subtitle1" color="text.secondary" component="a" 
                    href={result.githubUrl} target="_blank" rel="noopener noreferrer">
                    {result.githubUrl}
                  </Typography>
                </Box>

                <TableContainer>
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell component="th" scope="row" width="200">Match Score</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <LinearProgress 
                              variant="determinate" 
                              value={result.score * 100} 
                              sx={{ flexGrow: 1 }}
                            />
                            <Typography variant="body2">
                              {(result.score * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell component="th" scope="row">Experience</TableCell>
                        <TableCell>{result.experience}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell component="th" scope="row">Expertise</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            {result.expertise.map((skill, i) => (
                              <Chip key={i} label={skill} size="small" />
                            ))}
                          </Box>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell component="th" scope="row">GitHub Stats</TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {result.contributions.toLocaleString()} contributions · {result.followers.toLocaleString()} followers
                          </Typography>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                <TableContainer>
                  <Table>
                    <TableBody>
                      {Object.entries(result.analysis).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell component="th" scope="row" width="200">
                            {key.replace(/([A-Z])/g, ' $1').trim()}
                          </TableCell>
                          <TableCell>
                            <Rating value={value} precision={0.1} readOnly />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" gutterBottom>Key Strengths</Typography>
                  {result.strengths.map((strength, i) => (
                    <Alert key={i} severity="success" sx={{ mb: 1 }}>
                      {strength}
                    </Alert>
                  ))}
                </Box>

                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" gutterBottom>Areas for Improvement</Typography>
                  {result.areasOfImprovement.map((area, i) => (
                    <Alert key={i} severity="info" sx={{ mb: 1 }}>
                      {area}
                    </Alert>
                  ))}
                </Box>
              </TabPanel>

              <TabPanel value={tabValue} index={2}>
                <Typography variant="h6" gutterBottom>Recent Projects</Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Project Name</TableCell>
                        <TableCell>Language</TableCell>
                        <TableCell align="right">Stars</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {result.recentProjects.map((project, i) => (
                        <TableRow key={i}>
                          <TableCell>{project.name}</TableCell>
                          <TableCell>
                            <Chip label={project.language} size="small" />
                          </TableCell>
                          <TableCell align="right">{project.stars}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>
            </Paper>
          ))}
        </>
      )}

      {!loading && results.length === 0 && query && (
        <Alert severity="info">No results found. Try refining your search.</Alert>
      )}
      
      {/* Raw output for debugging */}
      {rawOutput && (
        <Card sx={{ mt: 4, p: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Raw API Response:</Typography>
            <Box component="pre" sx={{ 
              whiteSpace: 'pre-wrap', 
              overflow: 'auto',
              bgcolor: '#f5f5f5',
              p: 2,
              borderRadius: 1,
              maxHeight: '300px'
            }}>
              {rawOutput}
            </Box>
          </CardContent>
        </Card>
      )}
    </Container>
  )
}

export default App
