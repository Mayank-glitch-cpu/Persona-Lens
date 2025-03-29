"""
Persona-Lens Gemini Client (Python 3.6 Compatible Version)
This script demonstrates how to integrate the Persona-Lens API with Google's Gemini API
using direct HTTP requests instead of the google-generativeai library
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Get API key from .env file
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1"

def list_available_models(api_key: str) -> list:
    """List all available Gemini models"""
    try:
        response = requests.get(
            f"{GEMINI_BASE_URL}/models",
            params={"key": api_key}
        )
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            print(f"Error listing models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Exception listing models: {str(e)}")
        return []

# Persona-Lens API configuration
PERSONA_LENS_API_URL = f"http://{os.environ.get('API_HOST', 'localhost')}:{os.environ.get('API_PORT', '5000')}/api/persona-lens"

class PersonaLensGeminiClient:
    """Client for connecting Persona-Lens with Gemini API (Python 3.6 compatible)"""
    
    def __init__(self, gemini_api_key: str = GEMINI_API_KEY, persona_lens_url: str = PERSONA_LENS_API_URL):
        """Initialize the client with API keys and configuration"""
        self.persona_lens_url = persona_lens_url
        self.session_id = None
        
        # Check for Gemini API key
        if not gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in .env file or environment variable.")
        
        self.gemini_api_key = gemini_api_key
        
        # Get available models
        models = list_available_models(gemini_api_key)
        if not models:
            raise ValueError("Unable to fetch available Gemini models. Please check your API key and internet connection.")
        
        # Find the best available model
        self.model_name = None
        preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        for preferred_model in preferred_models:
            for model in models:
                if preferred_model in model["name"]:
                    self.model_name = model["name"].split("/")[-1]
                    break
            if self.model_name:
                break
        
        if not self.model_name:
            raise ValueError("No suitable Gemini model found. Available models: " + ", ".join([m["name"] for m in models]))
        
        # Set the API URL with the confirmed model name
        self.api_url = f"{GEMINI_BASE_URL}/models/{self.model_name}:generateContent"
    
    def search_candidates(self, query: str) -> Dict[str, Any]:
        """
        Search for candidates using Persona-Lens and process the results with Gemini
        
        Args:
            query: The natural language query (e.g., "find me top 5 Python developers")
            
        Returns:
            Dict containing the Gemini response and session information
        """
        # Step 1: Call Persona-Lens API to get candidate information
        persona_lens_response = self._call_persona_lens_query(query)
        
        if not persona_lens_response:
            return {"error": "Failed to get response from Persona-Lens API"}
        
        # Save session ID for follow-up queries
        self.session_id = persona_lens_response.get("session_id")
        
        # Step 2: Send the formatted prompt to Gemini
        prompt = persona_lens_response.get("prompt", "")
        gemini_response = self._call_gemini_api(prompt)
        
        # Step 3: Return combined response
        return {
            "gemini_response": gemini_response,
            "session_id": self.session_id,
            "rag_output": persona_lens_response.get("rag_output", "")
        }
    
    def get_candidate_details(self, username: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific candidate and process it with Gemini
        
        Args:
            username: The username of the candidate to look up
            
        Returns:
            Dict containing the Gemini response and profile information
        """
        # Step 1: Call Persona-Lens API to get detailed profile
        profile_response = self._call_persona_lens_profile(username)
        
        if not profile_response:
            return {"error": f"Failed to get profile for {username} from Persona-Lens API"}
        
        # Update session ID if provided
        if profile_response.get("session_id"):
            self.session_id = profile_response.get("session_id")
        
        # Step 2: Send the formatted profile prompt to Gemini
        prompt = profile_response.get("prompt", "")
        gemini_response = self._call_gemini_api(prompt)
        
        # Step 3: Return combined response
        return {
            "gemini_response": gemini_response,
            "session_id": self.session_id,
            "profile_data": profile_response.get("profile_data", "")
        }
    
    def _call_persona_lens_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Call the Persona-Lens query API endpoint"""
        try:
            payload = {"query": query}
            if self.session_id:
                payload["session_id"] = self.session_id
                
            response = requests.post(
                f"{self.persona_lens_url}/query",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error calling Persona-Lens API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Exception calling Persona-Lens API: {str(e)}")
            return None
    
    def _call_persona_lens_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """Call the Persona-Lens detailed profile API endpoint"""
        try:
            payload = {"username": username}
            if self.session_id:
                payload["session_id"] = self.session_id
                
            response = requests.post(
                f"{self.persona_lens_url}/detailed-profile",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error calling Persona-Lens profile API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Exception calling Persona-Lens profile API: {str(e)}")
            return None
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call the Gemini API directly via HTTP request (Python 3.6 compatible)"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Extract the text from the response
                if (response_json.get("candidates") and 
                    len(response_json["candidates"]) > 0 and 
                    response_json["candidates"][0].get("content") and 
                    response_json["candidates"][0]["content"].get("parts") and 
                    len(response_json["candidates"][0]["content"]["parts"]) > 0):
                    
                    return response_json["candidates"][0]["content"]["parts"][0].get("text", "")
                else:
                    return "No response text from Gemini API"
            else:
                error_msg = f"Error from Gemini API: {response.status_code}"
                try:
                    error_details = response.json()
                    if "error" in error_details:
                        error_msg += f" - {error_details['error'].get('message', '')}"
                except:
                    error_msg += f" - {response.text}"
                
                print(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            print(f"Exception calling Gemini API: {str(e)}")
            return f"Error generating response: {str(e)}"

def main():
    """Example usage of the Persona-Lens Gemini client"""
    # API key is loaded from .env file
    
    try:
        client = PersonaLensGeminiClient()
        
        print("Persona-Lens Gemini Client (Python 3.6 Compatible)")
        print("=================================================")
        print("This client connects Persona-Lens with Google's Gemini API")
        print("Using direct HTTP requests instead of the google-generativeai library")
        print("Type 'exit' or 'quit' to end the session\n")
        
        while True:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
                
            # Check if it's a follow-up query about a specific candidate
            if query.lower().startswith(("tell me more about", "who is", "more details on", "more info on")):
                # Extract the username from the query
                parts = query.split()
                username_idx = -1
                for idx, word in enumerate(parts):
                    if word.lower() in ["about", "on"]:
                        username_idx = idx + 1
                        break
                
                if username_idx >= 0 and username_idx < len(parts):
                    username = parts[username_idx]
                    print(f"\nGetting details for {username}...")
                    result = client.get_candidate_details(username)
                else:
                    # If we can't extract a username, treat as regular query
                    print("\nProcessing your query...")
                    result = client.search_candidates(query)
            else:
                # Regular candidate search
                print("\nProcessing your query...")
                result = client.search_candidates(query)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                # Print the response from Gemini
                print("\nGemini Response:")
                print("----------------")
                print(result["gemini_response"])
            
            print("\n" + "="*50)
        
    except ValueError as ve:
        print(f"Configuration error: {str(ve)}")
        print("Make sure to set your Gemini API key in the .env file:")
        print("GEMINI_API_KEY=your_api_key_here")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()