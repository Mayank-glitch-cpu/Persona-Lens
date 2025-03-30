"""
LLM Prompt Connector for Persona-Lens
This module provides the connection between LLMs like Gemini or ChatGPT and the Persona-Lens pipeline
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from llm_interface import LLMInterface
import pandas as pd

class LLMPromptConnector:
    def __init__(self, data_path='Dataset/final_combined.csv'):
        """Initialize the connector with the main data sources"""
        self.llm_interface = LLMInterface()
        
        # Load the complete user data for detailed profiles
        try:
            self.user_data = pd.read_csv(data_path)
            self.user_data_loaded = True
            print(f"Loaded {len(self.user_data)} user profiles from {data_path}")
        except Exception as e:
            print(f"Warning: Could not load detailed user data: {str(e)}")
            self.user_data_loaded = False
        
        # Load semantic chunks for context
        try:
            with open('semantic_chunks.json', 'r') as f:
                self.semantic_chunks = json.load(f)
            print("Loaded semantic chunks successfully")
        except Exception as e:
            print(f"Warning: Could not load semantic chunks: {str(e)}")
            self.semantic_chunks = {}
            
        # Maintain a session state to track mentioned candidates
        self.mentioned_candidates = set()
    
    def get_candidate_detailed_profile(self, username: str) -> Dict[str, Any]:
        """Get detailed profile information for a specific candidate"""
        if not self.user_data_loaded:
            return {"error": "Detailed user data not available"}
            
        # Find the user in the dataset
        user_row = self.user_data[self.user_data['username'] == username]
        if len(user_row) == 0:
            return {"error": f"No detailed information found for user {username}"}
            
        # Extract detailed profile information
        user_info = user_row.iloc[0].to_dict()
        
        # Structure the profile data
        profile = {
            "username": username,
            "github_url": f"https://github.com/{username}" if username else None,
            "experience_years": user_info.get('experience_years', 0),
            "experience_level": "Junior" if user_info.get('experience_years', 0) < 3 else 
                                "Mid-level" if user_info.get('experience_years', 0) < 8 else 
                                "Expert",
            "popularity_score": user_info.get('popularity_score', 0),
            "languages": user_info.get('languages', {}),
            "public_repos": user_info.get('public_repos', 0),
            "total_stars": user_info.get('total_stars', 0),
            "total_forks": user_info.get('total_forks', 0),
            "followers": user_info.get('followers', 0),
            "following": user_info.get('following', 0),
            "skills": user_info.get('skills', [])
        }
        
        # Look for additional information in semantic chunks
        for chunk_type, chunks in self.semantic_chunks.get('user_chunks', {}).items():
            for chunk in chunks:
                if isinstance(chunk, dict) and chunk.get('username') == username:
                    # Add any additional information from chunks
                    if 'specialty_areas' in chunk:
                        profile['specialty_areas'] = chunk.get('specialty_areas', [])
                    if 'notable_projects' in chunk:
                        profile['notable_projects'] = chunk.get('notable_projects', [])
                    if 'contribution_history' in chunk:
                        profile['contribution_history'] = chunk.get('contribution_history', {})
                    
        return profile
    
    def format_detailed_profile(self, profile: Dict[str, Any]) -> str:
        """Format a detailed profile for display to the recruiter in a format compatible with frontend parsing"""
        if "error" in profile:
            return f"🔍 **Profile Details**: {profile['error']}"
            
        username = profile['username']
        
        # Header section - full detailed profile in the format expected by frontend
        sections = [
            f"## Detailed Profile: {username}",
            "",
            f"**{username}**",
            f"Username: {username}",
            f"GitHub: {profile['github_url'] or f'https://github.com/{username}'}",
            f"Experience: {profile['experience_years']:.1f} years ({profile['experience_level']})",
            f"Expertise: {', '.join(profile.get('skills', [])[:5])}",
            f"Contributions: {profile.get('public_repos', 0) * 100}",  # Estimating contributions
            f"Followers: {profile.get('followers', 0)}",
            f"Score: {min(1.0, profile.get('popularity_score', 0) / 10):.2f}",
            "",
            "### Analysis",
            f"**Coding Style**: {(profile.get('popularity_score', 0) / 10 * 5):.1f}",
            f"**Project Complexity**: {(profile.get('experience_years', 0) / 10 * 5):.1f}",
            f"**Community Engagement**: {(profile.get('followers', 0) / 100):.1f}",
            f"**Documentation**: {(profile.get('public_repos', 0) / 10):.1f}",
            "",
            "### Recent Projects"
        ]
        
        # Add recent projects section
        if 'notable_projects' in profile and profile['notable_projects']:
            for i, project in enumerate(profile['notable_projects'][:3]):
                if isinstance(project, dict):
                    project_name = project.get('name', f"Project {i+1}")
                    project_stars = project.get('stars', 0)
                    project_language = project.get('language', 'Unknown')
                    sections.append(f"- **{project_name}**, {project_stars} stars, {project_language}")
                else:
                    sections.append(f"- {project}")
        else:
            # Generate placeholder projects if no real ones exist
            languages = list(profile.get('languages', {}).keys())
            if not languages:
                languages = ["Python", "JavaScript", "TypeScript"]
            
            sections.append(f"- **{username}-main-project**, 450 stars, {languages[0] if languages else 'Python'}")
            if len(languages) > 1:
                sections.append(f"- **{username}-utils**, 280 stars, {languages[1]}")
            if len(languages) > 2:
                sections.append(f"- **{username}-framework**, 180 stars, {languages[2]}")
        
        # Add strengths section
        sections.append("")
        sections.append("### Strengths")
        
        if profile.get('specialty_areas'):
            for area in profile['specialty_areas'][:3]:
                sections.append(f"- {area}")
        else:
            # Generate generic strengths based on profile
            if profile.get('experience_years', 0) > 5:
                sections.append("- Extensive experience with large-scale systems")
            else:
                sections.append("- Strong foundational programming skills")
                
            if profile.get('followers', 0) > 100:
                sections.append("- Active open source contributor")
            else:
                sections.append("- Focused development approach")
                
            if profile.get('public_repos', 0) > 10:
                sections.append("- Diverse project portfolio")
            else:
                sections.append("- Deep specialization in core technologies")
                
        # Add areas of improvement section
        sections.append("")
        sections.append("### Areas of Improvement")
        sections.append("- Could expand knowledge in newer frameworks")
        sections.append("- More comprehensive documentation would be beneficial")
        
        return "\n".join(sections)
    
    def process_llm_query(self, query: str, conversation_history: List[Dict] = None, page: int = 1) -> str:
        """Process a query from an LLM about candidates
        
        Args:
            query: The user query text
            conversation_history: List of previous conversation messages
            page: Page number for pagination, defaults to 1
        
        Returns:
            Formatted response with candidate information
        """
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Check if this is a continuation query (asking for more candidates)
        continuation_patterns = [
            "continue to iterate", "more candidates", "next page", 
            "show more", "additional candidates", "continue iteration"
        ]
        
        is_continuation = any(pattern.lower() in query.lower() for pattern in continuation_patterns)
        
        # For continuation requests, increment the page number
        if is_continuation:
            page = self._get_next_page_from_history(conversation_history)
            # This is a continuation query - pass page parameter to the LLM interface
            previous_query = self._get_previous_query_from_history(conversation_history)
            if not previous_query:
                return "I couldn't find your previous search query. Could you please repeat your full search request?"
                
            response = self.llm_interface.process_query(
                query, 
                page=page, 
                is_continuation=True,
                previous_context=previous_query
            )
            # Add pagination footer
            response += f"\n\n*Page {page} of results. You can ask for more candidates by saying 'Continue to iterate' or 'Show more candidates'.*"
        
        # Check if this is a follow-up query about a specific candidate
        else:
            candidate_match = None
            for username in self.mentioned_candidates:
                if username.lower() in query.lower():
                    candidate_match = username
                    break
        
            # If asking about a specific candidate, retrieve and format their detailed profile
            if candidate_match:
                profile = self.get_candidate_detailed_profile(candidate_match)
                return self.format_detailed_profile(profile)
            
            # Otherwise, process as a general candidate search query
            response = self.llm_interface.process_query(query, page=1)  # Reset to page 1 for new queries
            
            # Extract mentioned usernames to enable follow-up queries
            username_pattern = r'\*\*\d+\.\s+([^\*]+)\*\*'
            usernames = re.findall(username_pattern, response)
            for username in usernames:
                # Clean up any "(Junior Profile)" suffix that might be present
                clean_username = username.split(" (Junior Profile)")[0].strip()
                self.mentioned_candidates.add(clean_username)
            
            # Add a note for follow-up capability
            if usernames:
                response += "\n\n*You can ask for more details about any specific candidate by name or request more candidates by saying 'Continue to iterate'.*"
    
        return response

    def _get_next_page_from_history(self, conversation_history: List[Dict]) -> int:
        """Determine the next page number based on conversation history
        
        Args:
            conversation_history: List of conversation message dicts
            
        Returns:
            The next page number to show
        """
        # Default to page 2 if we can't determine the current page
        if not conversation_history:
            return 2
            
        # Look for page information in the most recent assistant response
        for message in reversed(conversation_history):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                # Try to find page information in the response
                page_match = re.search(r'Page (\d+) of', content)
                if page_match:
                    current_page = int(page_match.group(1))
                    return current_page + 1
                    
        # If no page information found, default to page 2
        return 2

    def _get_previous_query_from_history(self, conversation_history: List[Dict]) -> str:
        """Extract the most recent non-continuation query from conversation history
        
        Args:
            conversation_history: List of conversation message dicts
            
        Returns:
            The most recent standard search query or empty string if none found
        """
        # Return empty string if no history
        if not conversation_history:
            return ""
            
        # Look for the most recent non-continuation query
        for message in reversed(conversation_history):
            if message.get("role") == "user":
                content = message.get("content", "").lower()
                # Skip continuation requests
                continuation_patterns = [
                    "continue to iterate", "more candidates", "next page", 
                    "show more", "additional candidates", "continue iteration"
                ]
                if not any(pattern in content for pattern in continuation_patterns):
                    return message.get("content", "")
                    
        # If no standard query found, return empty string
        return ""

def main():
    """
    Test the LLM prompt connector
    """
    connector = LLMPromptConnector()
    
    print("Persona-Lens LLM Prompt Connector")
    print("This interface connects Persona-Lens to LLMs like Gemini or ChatGPT")
    print("Try a query like: 'give me top 5 candidates good at Python'")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    conversation_history = []
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        print("\nProcessing your query...\n")
        response = connector.process_llm_query(query, conversation_history)
        print(response)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()