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
        """Format a detailed profile for display to the recruiter"""
        if "error" in profile:
            return f"🔍 **Profile Details**: {profile['error']}"
            
        sections = []
        
        # Header with name and basic info
        header = f"## 👤 Detailed Profile: {profile['username']}"
        if profile['github_url']:
            header += f" ([GitHub]({profile['github_url']}))"
        sections.append(header)
        
        # Summary section
        summary = [
            f"**Experience Level**: {profile['experience_level']} ({profile['experience_years']:.1f} years)",
            f"**Popularity Score**: {profile['popularity_score']:.1f}/10",
            f"**GitHub Stats**: {profile['public_repos']} repositories • {profile['total_stars']} stars • {profile['followers']} followers"
        ]
        sections.append("### Summary\n" + "\n".join(summary))
        
        # Programming Languages section
        if profile['languages'] and isinstance(profile['languages'], dict):
            lang_section = ["### 💻 Programming Languages"]
            languages = profile['languages']
            lang_items = []
            
            for lang, value in sorted(languages.items(), key=lambda x: (
                x[1]['count'] if isinstance(x[1], dict) and 'count' in x[1]
                else x[1] if isinstance(x[1], (int, float)) and x[1] > 1
                else 0
            ), reverse=True)[:10]:  # Show top 10 languages
                if isinstance(value, dict) and 'count' in value:
                    lang_items.append(f"- **{lang}**: {value['count']} repositories")
                elif isinstance(value, (int, float)):
                    if value > 1:  # Count
                        lang_items.append(f"- **{lang}**: {int(value)} repositories")
                    else:  # Percentage
                        lang_items.append(f"- **{lang}**: {value*100:.1f}% of code")
            
            lang_section.append("\n".join(lang_items))
            sections.append("\n".join(lang_section))
        
        # Skills section
        if profile.get('skills'):
            skills_section = ["### 🔧 Technical Skills"]
            if isinstance(profile['skills'], list):
                skills_section.append("\n".join([f"- {skill}" for skill in profile['skills'][:15]]))  # Show top 15 skills
            elif isinstance(profile['skills'], dict):
                skills_items = []
                for skill, value in sorted(profile['skills'].items(), key=lambda x: x[1], reverse=True)[:15]:
                    skills_items.append(f"- **{skill}**: {value}")
                skills_section.append("\n".join(skills_items))
            sections.append("\n".join(skills_section))
        
        # Notable projects section
        if profile.get('notable_projects'):
            projects_section = ["### 🚀 Notable Projects"]
            for project in profile['notable_projects'][:5]:  # Show top 5 projects
                if isinstance(project, dict):
                    proj_name = project.get('name', 'Unnamed Project')
                    proj_desc = project.get('description', 'No description available')
                    proj_stars = project.get('stars', 0)
                    proj_url = project.get('url', '')
                    
                    project_entry = f"- **{proj_name}**"
                    if proj_url:
                        project_entry += f" ([Link]({proj_url}))"
                    project_entry += f": {proj_desc} ({proj_stars} ⭐)"
                    projects_section.append(project_entry)
                else:
                    projects_section.append(f"- {project}")
            sections.append("\n".join(projects_section))
        
        # Join all sections with spacing
        return "\n\n".join(sections)
    
    def process_llm_query(self, query: str, conversation_history: List[str] = None) -> str:
        """Process a query from an LLM about candidates"""
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []
        
        # Check if this is a follow-up query about a specific candidate
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
        response = self.llm_interface.process_query(query)
        
        # Extract mentioned usernames to enable follow-up queries
        username_pattern = r'\*\*\d+\.\s+([^\*]+)\*\*'
        usernames = re.findall(username_pattern, response)
        for username in usernames:
            # Clean up any "(Junior Profile)" suffix that might be present
            clean_username = username.split(" (Junior Profile)")[0].strip()
            self.mentioned_candidates.add(clean_username)
        
        # Add a note for follow-up capability
        if usernames:
            response += "\n\n*You can ask for more details about any specific candidate by name.*"
            
        return response

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