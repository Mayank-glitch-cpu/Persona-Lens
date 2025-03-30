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
        self.mentioned_candidates = set()  # Track candidates mentioned in previous responses
        self.conversation_history = []  # Track conversation history
        self.last_query = None  # Store the last search query for "continue to iterate" functionality
        self.current_page = 1  # Track pagination for iterate functionality
        
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
    
    def _call_persona_lens_query(self, query: str, page: int = 1) -> Dict[str, Any]:
        """
        Call the Persona-Lens API with the user query
        
        Args:
            query: The natural language query
            page: Page number for pagination (default: 1)
            
        Returns:
            Dict containing the Persona-Lens response
        """
        try:
            response = requests.post(
                f"{self.persona_lens_url}/query",
                json={
                    "query": query,
                    "session_id": self.session_id,
                    "page": page  # Add pagination support
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error from Persona-Lens API: {response.status_code} - {response.text}")
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
            print("\n=== Sending request to Gemini API ===")
            print(f"Prompt being sent:\n{prompt}\n")
            
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
            
            print(f"Using Gemini model: {self.model_name}")
            print(f"API URL: {self.api_url}")
            
            response = requests.post(
                f"{self.api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            print(f"\nGemini API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print("\nRaw Gemini Response:")
                print(json.dumps(response_json, indent=2))
                
                if (response_json.get("candidates") and 
                    len(response_json["candidates"]) > 0 and 
                    response_json["candidates"][0].get("content") and 
                    response_json["candidates"][0]["content"].get("parts") and 
                    len(response_json["candidates"][0]["content"]["parts"]) > 0):
                    
                    response_text = response_json["candidates"][0]["content"]["parts"][0].get("text", "")
                    print("\nExtracted Response Text:")
                    print(response_text)
                    return response_text
                else:
                    print("\nWarning: Could not extract text from Gemini response structure")
                    return "No response text from Gemini API"
            else:
                error_msg = f"Error from Gemini API: {response.status_code}"
                try:
                    error_details = response.json()
                    if "error" in error_details:
                        error_msg += f" - {error_details['error'].get('message', '')}"
                except:
                    error_msg += f" - {response.text}"
                
                print(f"\nError: {error_msg}")
                return f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Exception calling Gemini API: {str(e)}"
            print(f"\nError: {error_msg}")
            return f"Error generating response: {str(e)}"

    def process_follow_up(self, query: str) -> Dict[str, Any]:
        """
        Process a follow-up question that might refer to a specific candidate
        or compare multiple candidates
        
        Args:
            query: The follow-up question from the user
            
        Returns:
            Dict containing the response
        """
        # Extract candidate names from the query
        candidate_names = self._extract_candidate_names(query)
        
        # If no candidates detected, treat as a regular query
        if not candidate_names:
            return self.search_candidates(query)
        
        # If there's only one candidate, get detailed profile
        if len(candidate_names) == 1:
            return self.get_candidate_details(candidate_names[0])
        
        # If multiple candidates, handle comparison
        if len(candidate_names) > 1:
            return self._handle_candidate_comparison(query, candidate_names)
        
        # Default fallback to regular search
        return self.search_candidates(query)
    
    def _extract_candidate_names(self, query: str) -> list:
        """
        Extract candidate names from a follow-up query
        
        Args:
            query: The follow-up question from the user
            
        Returns:
            List of extracted candidate names
        """
        # First check for common follow-up patterns
        lower_query = query.lower()
        
        # Patterns for direct mentions
        patterns = [
            # "tell me more about X"
            (["tell", "me", "more", "about"], 4),
            # "who is X"
            (["who", "is"], 2),
            # "more details on X"
            (["more", "details", "on"], 3),
            # "more info on X" 
            (["more", "info", "on"], 3),
            # "can you elaborate on X"
            (["can", "you", "elaborate", "on"], 4),
            # "what about X"
            (["what", "about"], 2)
        ]
        
        words = lower_query.split()
        
        # Check each pattern
        for pattern, idx_offset in patterns:
            if len(words) > idx_offset:
                # Check if the pattern matches the beginning of the query
                if words[:idx_offset] == pattern:
                    # The next word should be the candidate name
                    if idx_offset < len(words):
                        return [words[idx_offset]]
        
        # Check for comparison patterns
        if "compare" in lower_query and "and" in lower_query:
            # Try to extract names from "compare X and Y"
            compare_idx = words.index("compare")
            and_idx = words.index("and")
            
            if compare_idx < and_idx and and_idx < len(words) - 1:
                first_name = words[compare_idx + 1]
                second_name = words[and_idx + 1]
                return [first_name, second_name]
        
        # If we reach here, check if any of the previously mentioned candidates 
        # appear in the query
        candidates = []
        for candidate in self.mentioned_candidates:
            if candidate.lower() in lower_query:
                candidates.append(candidate)
        
        return candidates
    
    def _handle_candidate_comparison(self, query: str, candidate_names: list) -> Dict[str, Any]:
        """
        Handle a comparison between multiple candidates
        
        Args:
            query: The original query
            candidate_names: List of candidate names to compare
            
        Returns:
            Dict containing the comparison response
        """
        try:
            # First collect detailed profiles for each candidate
            profiles = {}
            for name in candidate_names:
                profile_response = self._call_persona_lens_profile(name)
                if profile_response and "profile_data" in profile_response:
                    profiles[name] = profile_response.get("profile_data", {})
            
            # If we couldn't get any profiles, return an error
            if not profiles:
                return {"error": "Could not retrieve profiles for the specified candidates"}
            
            # Construct a comparison prompt
            comparison_prompt = f"Compare the following GitHub developers: {', '.join(candidate_names)}.\n\n"
            
            for name, profile in profiles.items():
                comparison_prompt += f"--- {name} ---\n"
                if isinstance(profile, dict):
                    comparison_prompt += json.dumps(profile, indent=2)
                else:
                    comparison_prompt += str(profile)
                comparison_prompt += "\n\n"
            
            comparison_prompt += f"User query: {query}\n"
            comparison_prompt += "Please provide a detailed comparison of these developers based on their profiles."
            
            # Send to Gemini
            gemini_response = self._call_gemini_api(comparison_prompt)
            
            # Return the response
            return {
                "gemini_response": gemini_response,
                "session_id": self.session_id,
                "compared_candidates": candidate_names
            }
                
        except Exception as e:
            print(f"Exception during candidate comparison: {str(e)}")
            return {"error": f"Failed to compare candidates: {str(e)}"}
            
    def handle_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for handling all types of queries
        
        Args:
            query: The natural language query
            session_id: Optional session ID for follow-up queries
            
        Returns:
            Dict containing the response and session information
        """
        # Check if this is a follow-up query with session ID
        if session_id:
            self.session_id = session_id
            
            # Check for "continue to iterate" pattern
            if self._is_continue_iteration_query(query):
                return self._handle_continue_iteration()
                
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
        
        # Not a follow-up query, handle as a new search
        self.last_query = query  # Store the query for potential "continue" requests
        self.current_page = 1  # Reset pagination
        return self.search_candidates(query)
    
    def _is_continue_iteration_query(self, query: str) -> bool:
        """
        Detect if a query is asking to continue iterating through candidates
        
        Args:
            query: The query string to check
            
        Returns:
            Boolean indicating if this is a continue iteration query
        """
        query_lower = query.lower().strip()
        continue_patterns = [
            "continue to iterate",
            "continue iteration",
            "show more candidates",
            "show more results",
            "next page",
            "more candidates",
            "more results",
            "continue searching",
            "show additional",
            "more developers"
        ]
        
        return any(pattern in query_lower for pattern in continue_patterns)
    
    def _handle_continue_iteration(self) -> Dict[str, Any]:
        """
        Handle a request to continue iterating through candidates
        
        Returns:
            Dict containing the next page of results
        """
        if not self.last_query:
            return {
                "error": "No previous search query found to continue iterating from",
                "gemini_response": "I don't have any previous search results to continue from. Please provide a new search query.",
                "session_id": self.session_id
            }
        
        # Increment the page number
        self.current_page += 1
        
        # Call Persona-Lens API with pagination parameters
        persona_lens_response = self._call_persona_lens_query(
            self.last_query, 
            page=self.current_page
        )
        
        if not persona_lens_response:
            return {"error": "Failed to get response from Persona-Lens API"}
        
        # Send the formatted prompt to Gemini
        prompt = persona_lens_response.get("prompt", "")
        
        # Add context that this is a continuation
        continuation_prompt = f"""
This is a continuation of a previous search for "{self.last_query}".
Here are additional candidates (page {self.current_page}):

{prompt}
        """
        
        gemini_response = self._call_gemini_api(continuation_prompt)
        
        # Return combined response
        return {
            "gemini_response": gemini_response,
            "session_id": self.session_id,
            "rag_output": persona_lens_response.get("rag_output", ""),
            "is_continuation": True,
            "page": self.current_page
        }

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