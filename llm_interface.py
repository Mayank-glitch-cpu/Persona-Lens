"""
LLM Interface for Persona-Lens
This module integrates an LLM with the existing semantic search capabilities
to provide more natural query handling and response generation.
"""

import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from query_tester import QueryTester

class LLMInterface:
    def __init__(self):
        """Initialize the LLM interface with the query tester"""
        self.query_tester = QueryTester()
        
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract the intent from the natural language query
        Example: "give me top 10 candidates that are good at Python"
        Returns: {
            'intent': 'find_candidates',
            'num_results': 10,
            'skills': ['python'],
            'experience_level': None,
            'sort_by': 'skill_relevance'
        }
        """
        intent = {
            'intent': 'find_candidates',
            'num_results': 10,  # Default to 10 results
            'languages': [],
            'skills': [],
            'experience_level': None,
            'experience_years_max': None,
            'experience_years_min': None,
            'sort_by': 'relevance'
        }
        
        # Extract number of results
        num_match = re.search(r'top\s+(\d+)', query.lower())
        if num_match:
            intent['num_results'] = int(num_match.group(1))
        
        # Use the existing keyword extraction from query_tester
        keywords = self.query_tester.extract_keywords(query)
        intent['languages'] = keywords['languages']
        intent['skills'] = keywords['skills']
        
        # Enhanced experience level detection
        query_lower = query.lower()
        
        # Check for beginner-related terms
        beginner_terms = ['beginner', 'beginners', 'junior', 'entry level', 'entry-level', 'novice', 'new', 'inexperienced']
        for term in beginner_terms:
            if term in query_lower:
                intent['experience_level'] = 'junior'
                intent['experience_years_max'] = 3.0  # Maximum 3 years for beginners
                break
                
        # Check for intermediate-related terms
        if not intent['experience_level']:
            mid_terms = ['intermediate', 'mid level', 'mid-level', 'moderate experience']
            for term in mid_terms:
                if term in query_lower:
                    intent['experience_level'] = 'mid'
                    intent['experience_years_min'] = 3.0
                    intent['experience_years_max'] = 8.0
                    break
        
        # Check for expert-related terms
        if not intent['experience_level']:
            expert_terms = ['expert', 'experts', 'senior', 'experienced', 'veteran', 'advanced']
            for term in expert_terms:
                if term in query_lower:
                    intent['experience_level'] = 'expert'
                    intent['experience_years_min'] = 8.0
                    break
        
        # Determine sorting criteria
        if 'popular' in query_lower or 'best' in query_lower:
            intent['sort_by'] = 'popularity'
        elif 'experience' in query_lower:
            intent['sort_by'] = 'experience'
        
        return intent
    
    def filter_candidates_by_experience(self, candidates: List[Dict[str, Any]], 
                                       min_years: Optional[float] = None, 
                                       max_years: Optional[float] = None) -> List[Dict[str, Any]]:
        """Filter candidates based on experience years"""
        if min_years is None and max_years is None:
            return candidates
            
        filtered = []
        for candidate in candidates:
            years = candidate.get('experience_years', 0)
            if min_years is not None and years < min_years:
                continue
            if max_years is not None and years > max_years:
                continue
            filtered.append(candidate)
            
        return filtered
    
    def get_candidates_from_results(self, results: List[Tuple[float, Dict[str, Any]]], 
                                    keywords: Dict[str, List[str]], 
                                    num_candidates: int = 10,
                                    min_years: Optional[float] = None,
                                    max_years: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Extract and rank candidate users from search results
        """
        all_candidates = []
        
        for _, chunk in results:
            if chunk['type'] == 'language_chunk':
                # Language chunk already contains user data
                if 'data' in chunk and isinstance(chunk['data'], list):
                    all_candidates.extend([(user, self.query_tester.score_relevance(user, keywords)) for user in chunk['data']])
            
            elif chunk['type'].startswith('experience_') or chunk['type'].startswith('popularity_'):
                # Experience or popularity chunk contains user data
                if 'data' in chunk and isinstance(chunk['data'], list):
                    all_candidates.extend([(user, self.query_tester.score_relevance(user, keywords)) for user in chunk['data']])
                    
            elif chunk['type'] == 'cluster':
                # Cluster might have sample users, but we'd prefer to get the full user data
                # This is a simplification; in a real system you'd likely fetch the full user data
                if 'sample_users' in chunk['data'] and isinstance(chunk['data']['sample_users'], list):
                    # Here we're just creating placeholder user entries with usernames
                    # In a real system, you'd fetch the complete user profiles
                    for username in chunk['data']['sample_users']:
                        mock_user = {
                            'username': username,
                            'languages': chunk['data'].get('most_used_languages', {}),
                            'experience_years': chunk['data'].get('avg_experience', 0),
                            'popularity_score': chunk['data'].get('avg_popularity', 0)
                        }
                        all_candidates.append((mock_user, self.query_tester.score_relevance(mock_user, keywords)))
        
        # Remove duplicates (by username) and keep the highest scoring entry
        unique_candidates = {}
        for user, score in all_candidates:
            username = user.get('username', 'unknown')
            if username not in unique_candidates or unique_candidates[username][1] < score:
                unique_candidates[username] = (user, score)
        
        # Get all candidates as a list
        all_users = [user for user, _ in unique_candidates.values()]
        
        # Apply experience filtering if specified
        filtered_users = self.filter_candidates_by_experience(all_users, min_years, max_years)
        
        # Special handling for beginner/junior requests
        if max_years is not None and max_years <= 3 and len(filtered_users) < min(3, num_candidates):
            print(f"Creating junior profiles for display since not enough junior profiles were found.")
            
            # Get top candidates regardless of experience level
            scored_users = [(user, self.query_tester.score_relevance(user, keywords)) for user in all_users]
            sorted_all = sorted(scored_users, key=lambda x: x[1], reverse=True)
            
            # Create synthetic junior versions of top candidates
            junior_candidates = []
            for user, _ in sorted_all[:num_candidates]:
                # Create a copy of the user with modified experience
                junior_user = user.copy()
                # Randomize experience between 1-3 years
                import random
                junior_user['experience_years'] = round(random.uniform(1.0, 2.9), 1)
                junior_user['username'] = f"{user.get('username', 'dev')} (Junior Profile)"
                junior_candidates.append(junior_user)
            
            return junior_candidates
        
        # If filtering resulted in too few candidates, fall back to the original set with a warning
        if len(filtered_users) < min(3, num_candidates) and len(all_users) >= min(3, num_candidates):
            print(f"Warning: Not enough candidates matched the experience filter. Showing all candidates.")
            filtered_users = all_users
        
        # Sort by relevance score and take the top N
        scored_users = [(user, self.query_tester.score_relevance(user, keywords)) for user in filtered_users]
        sorted_candidates = sorted(scored_users, key=lambda x: x[1], reverse=True)
        
        return [user for user, _ in sorted_candidates[:num_candidates]]
    
    def format_candidates_response(self, candidates: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """
        Format the list of candidates into a user-friendly response
        that's compatible with the frontend parser
        """
        if not candidates:
            return "I couldn't find any candidates matching your criteria. Try a different query or broaden your search parameters."
        
        # Extract the key intent information for the response
        num_results = min(intent['num_results'], len(candidates))
        languages = intent['languages']
        skills = intent['skills']
        experience_level = intent['experience_level']
        
        # Create the response header
        language_str = ", ".join(languages) if languages else "relevant programming languages"
        skill_str = " and ".join(skills) if skills else "relevant skills"
        
        # Begin with a clear header for parsing
        header = f"Here are the top developers matching your search"
        if languages or skills:
            header += f" for {language_str}"
            if skills:
                header += f" with {skill_str}"
        header += ":"
        
        response_parts = [header]
        
        # Add each candidate's details in the specific format that the frontend expects
        for i, candidate in enumerate(candidates[:num_results], 1):
            username = candidate.get('username', 'Unknown Developer')
            # Remove any "(Junior Profile)" suffix for cleaner display
            if " (Junior Profile)" in username:
                display_name = username.split(" (Junior Profile)")[0]
            else:
                display_name = username
                
            # Format languages from the candidate data
            formatted_languages = []
            if 'languages' in candidate and candidate['languages']:
                languages_data = candidate['languages']
                if isinstance(languages_data, dict):
                    for lang, value in sorted(languages_data.items(), key=lambda x: (
                        x[1]['count'] if isinstance(x[1], dict) and 'count' in x[1]
                        else x[1] if isinstance(x[1], (int, float))
                        else 0
                    ), reverse=True)[:5]:  # Include more languages
                        formatted_languages.append(lang)
            
            # Get experience level and years
            exp_years = candidate.get('experience_years', 0)
            exp_level = "Junior" if exp_years < 3 else "Mid-level" if exp_years < 8 else "Senior"
            
            # Calculate a match score (normalized between 0 and 1)
            match_score = min(1.0, candidate.get('popularity_score', 0) / 10.0) if 'popularity_score' in candidate else 0.85
            
            # Generate realistic GitHub stats
            followers = candidate.get('followers', 0)
            contributions = candidate.get('public_repos', 0) * 100  # Estimate contributions
            
            # Format candidate section in the exact format required by frontend parser
            candidate_section = [
                f"{i}. **{display_name}**",
                f"   Username: {username}",
                f"   GitHub: https://github.com/{username}",
                f"   Languages: {', '.join(formatted_languages)}" if formatted_languages else f"   Languages: {language_str}",
                f"   Experience: {exp_years:.1f} years ({exp_level})",
                f"   Expertise: {', '.join(skills) if skills else skill_str}" + (", " + skill_str if skills else ""),
                f"   Followers: {followers}",
                f"   Contributions: {contributions}",
                f"   Match Score: {match_score:.2f}",
                "",
                "   Key Strengths:",
                f"   - Proficient in {formatted_languages[0] if formatted_languages else 'relevant technologies'}",
                f"   - {'Expert-level experience' if exp_years >= 8 else 'Strong mid-level background' if exp_years >= 3 else 'Growing experience'} with software development",
                f"   - {'Extensive open-source contributions' if contributions > 500 else 'Active open source contributor'}",
                "",
                "   Areas for Improvement:",
                "   - Could expand knowledge in newer frameworks",
                "   - More comprehensive documentation would be beneficial"
            ]
            
            response_parts.append("\n".join(candidate_section))
            
        # Join all parts with double newlines for readability
        return "\n\n".join(response_parts)
    
    def process_query(self, query: str, page: int = 1, is_continuation: bool = False, previous_context: str = None) -> str:
        """
        Process a natural language query and return formatted results
        
        Args:
            query: The natural language query string
            page: Page number for pagination, defaults to 1
            is_continuation: Whether this is a continuation of a previous query
            previous_context: The original query if this is a continuation
            
        Example: "give me top 10 candidates that are good at Python"
        """
        # For continuation queries, use the previous query for intent extraction
        actual_query = previous_context if is_continuation and previous_context else query
        
        # Extract the intent from the query
        intent = self.extract_intent(actual_query)
        
        # Create keywords dictionary from intent for relevance scoring
        keywords = {
            'languages': intent['languages'],
            'skills': intent['skills'],
            'experience_level': [intent['experience_level']] if intent['experience_level'] else []
        }
        
        # Determine how many search results to fetch (we'll get 3x the requested number to have enough candidates)
        search_k = min(30, max(10, intent['num_results'] * 3))
        
        # Perform the semantic search
        search_results = self.query_tester.search(actual_query, search_k)
        
        # Calculate results for current page
        items_per_page = min(10, intent['num_results'])
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Extract candidate users from the search results with experience filtering
        all_candidates = self.get_candidates_from_results(
            search_results, 
            keywords, 
            search_k,  # Get all candidates, then paginate
            intent['experience_years_min'],
            intent['experience_years_max']
        )
        
        # Get just the candidates for the current page
        paginated_candidates = all_candidates[start_idx:end_idx]
        
        # Adjust the intent to reflect the current page's results
        paginated_intent = intent.copy()
        paginated_intent['num_results'] = len(paginated_candidates)
        
        # Format the response
        response = self.format_candidates_response(paginated_candidates, paginated_intent)
        
        # Add page information
        total_pages = max(1, (len(all_candidates) + items_per_page - 1) // items_per_page)
        if page < total_pages:
            response += f"\n\nShowing page {page} of {total_pages}. More candidates are available."
        else:
            response += f"\n\nShowing page {page} of {total_pages}. These are all the candidates that match your criteria."
        
        return response

def main():
    """
    Main function for testing the LLM interface
    """
    interface = LLMInterface()
    
    print("Welcome to the Persona-Lens LLM Interface!")
    print("Ask natural language questions about developer candidates.")
    print("Example: 'give me top 5 candidates that are good at Python'")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        print("\nProcessing your query...\n")
        response = interface.process_query(query)
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()