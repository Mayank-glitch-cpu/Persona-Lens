import axios from 'axios';

// Using a more flexible URL configuration that works in various environments
const API_BASE_URL = 'http://localhost:5000/api';

// Fallback to window.location.origin for deployments
// const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:5000/api`;

export interface PersonaResponse {
  prompt: string;
  rag_output: string;
  session_id: string;
  is_fallback?: boolean;
}

export interface DetailedProfileResponse {
  prompt: string;
  profile_data: string;
  session_id: string;
}

// Create a custom axios instance with proper configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  timeout: 20000, // 20 second timeout
  withCredentials: false // Important: disable credentials for CORS
});

// Function to verify the API is accessible
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    console.log(`Testing API health at: ${API_BASE_URL}/health`);
    
    // Use native fetch to avoid any axios configuration issues
    const response = await fetch(`${API_BASE_URL}/health`, { 
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      },
      // These settings help avoid CORS and caching problems
      mode: 'cors',
      cache: 'no-cache',
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('Health check response:', data);
      
      if (data && data.status === 'ok') {
        console.log('API health check passed');
        return true;
      }
    }
    
    console.warn('API health check failed: Bad response');
    return false;
  } catch (error) {
    console.error('API health check failed with error:', error);
    return false;
  }
};

export const searchCandidates = async (query: string, sessionId?: string): Promise<PersonaResponse> => {
  try {
    // First check if the API is healthy
    const isHealthy = await checkApiHealth();
    if (!isHealthy) {
      throw new Error('API is not accessible');
    }
    
    console.log(`Sending search request to ${API_BASE_URL}/persona-lens/query`);
    
    // Use native fetch instead of axios for better CORS handling
    const response = await fetch(`${API_BASE_URL}/persona-lens/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        query,
        session_id: sessionId
      }),
      mode: 'cors',
      cache: 'no-cache'
    });
    
    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Search response received:', data);
    return data;
  } catch (error: any) {
    console.error('Error searching candidates:', error);
    
    // More detailed error logging
    throw new Error(`API connection failed: ${error.message || 'Unknown error'}`);
  }
};

export const getDetailedProfile = async (username: string, sessionId?: string): Promise<DetailedProfileResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/persona-lens/detailed-profile`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        username,
        session_id: sessionId
      }),
      mode: 'cors',
      cache: 'no-cache'
    });
    
    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error('Error fetching detailed profile:', error);
    throw new Error(`Failed to fetch profile: ${error.message || 'Unknown error'}`);
  }
};