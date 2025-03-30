import axios from 'axios';

// Try both IP address and localhost to increase reliability
const API_URLS = [
  'http://10.139.126.48:5000/api',  // IP address (primary)
  'http://localhost:5000/api'       // localhost (fallback)
];

let activeBaseUrl = API_URLS[0]; // Start with the first URL

export interface PersonaResponse {
  prompt: string;
  rag_output: string;
  session_id: string;
}

export interface DetailedProfileResponse {
  prompt: string;
  profile_data: string;
  session_id: string;
}

// Create axios instance with proper configuration
const apiClient = axios.create({
  baseURL: activeBaseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 20000, // 20 seconds timeout
});

// Function to try all available base URLs
const tryAllBaseUrls = async (endpoint: string, data: any) => {
  let lastError;
  
  for (const baseUrl of API_URLS) {
    try {
      console.log(`Attempting connection to: ${baseUrl}${endpoint}`);
      const response = await axios({
        method: 'post',
        url: `${baseUrl}${endpoint}`,
        data,
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 15000
      });
      
      // If successful, update the active base URL for future requests
      activeBaseUrl = baseUrl;
      apiClient.defaults.baseURL = baseUrl;
      console.log(`Successfully connected to: ${baseUrl}`);
      
      return response.data;
    } catch (err: any) {
      console.warn(`Connection failed for ${baseUrl}: ${err.message}`);
      lastError = err;
    }
  }
  
  // If we get here, all URLs failed
  throw lastError || new Error('Failed to connect to any API endpoint');
};

export const searchCandidates = async (query: string, sessionId?: string): Promise<PersonaResponse> => {
  try {
    console.log('Searching candidates with query:', query);
    
    // Try with all possible base URLs
    const data = await tryAllBaseUrls('/persona-lens/query', {
      query,
      session_id: sessionId
    });
    
    return data;
  } catch (error: any) {
    console.error('Error searching candidates:', error);
    
    // Detailed error logging to help debug network issues
    if (error.response) {
      console.error('Server responded with error:', {
        status: error.response.status,
        data: error.response.data
      });
      
      // If we received a 500 error with specific error message, include it
      if (error.response.status === 500 && error.response.data && error.response.data.error) {
        throw new Error(`Server error: ${error.response.data.error}`);
      }
    } else if (error.request) {
      console.error('No response received from server');
      throw new Error('Network error: Unable to connect to the server. Please check if the backend is running.');
    }
    
    throw new Error(error.message || 'Failed to search for candidates');
  }
};

export const getDetailedProfile = async (username: string, sessionId?: string): Promise<DetailedProfileResponse> => {
  try {
    // Try with all possible base URLs
    const data = await tryAllBaseUrls('/persona-lens/detailed-profile', {
      username,
      session_id: sessionId
    });
    
    return data;
  } catch (error: any) {
    console.error('Error fetching detailed profile:', error);
    
    // Include more details in the error if available
    if (error.response && error.response.data && error.response.data.error) {
      throw new Error(`Server error: ${error.response.data.error}`);
    }
    
    throw new Error(error.message || 'Failed to fetch detailed profile');
  }
};