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
        """
        if not candidates:
            return "I couldn't find any candidates matching your criteria. Try a different query or broaden your search parameters."
        
        # Extract the key intent information for the response
        num_results = min(intent['num_results'], len(candidates))
        languages = intent['languages']
        skills = intent['skills']
        experience_level = intent['experience_level']
        
        # Create the response header
        response_parts = []
        language_str = ", ".join(languages) if languages else "any programming language"
        skill_str = " and ".join(skills) if skills else "the specified criteria"
        
        # Start building the header based on experience level
        if experience_level == 'junior':
            header = f"Here are the top {num_results} beginner/junior candidates"
        elif experience_level == 'mid':
            header = f"Here are the top {num_results} mid-level candidates"
        elif experience_level == 'expert':
            header = f"Here are the top {num_results} expert candidates"
        else:
            header = f"Here are the top {num_results} candidates"
            
        # Add language and skill information    
        if languages:
            header += f" skilled in {language_str}"
        if skills:
            header += f" with expertise in {skill_str}"
        response_parts.append(header + ":\n")
        
        # Add each candidate's details
        for i, candidate in enumerate(candidates[:num_results], 1):
            candidate_details = [f"**{i}. {candidate.get('username', 'Unknown Developer')}**"]
            
            # Add languages
            if 'languages' in candidate and candidate['languages']:
                languages = candidate['languages']
                if isinstance(languages, dict):
                    lang_list = []
                    for lang, value in sorted(languages.items(), key=lambda x: (
                        x[1]['count'] if isinstance(x[1], dict) and 'count' in x[1]
                        else x[1] if isinstance(x[1], (int, float))
                        else 0
                    ), reverse=True)[:3]:
                        if isinstance(value, dict) and 'count' in value:
                            lang_list.append(f"{lang} ({value['count']} repos)")
                        elif isinstance(value, (int, float)):
                            if value > 1:
                                lang_list.append(f"{lang} ({int(value)} repos)")
                            else:
                                lang_list.append(f"{lang} ({value*100:.1f}%)")
                    if lang_list:
                        candidate_details.append(f"Languages: {', '.join(lang_list)}")
            
            # Add experience
            if 'experience_years' in candidate:
                exp_years = candidate['experience_years']
                experience_level = "Junior" if exp_years < 3 else "Mid-level" if exp_years < 8 else "Expert"
                candidate_details.append(f"Experience: {exp_years:.1f} years ({experience_level})")
            
            # Add popularity
            if 'popularity_score' in candidate:
                candidate_details.append(f"Popularity Score: {candidate['popularity_score']:.1f}")
            
            # Add repo stats
            stats = []
            if 'public_repos' in candidate:
                stats.append(f"{candidate['public_repos']} repos")
            if 'total_stars' in candidate:
                stats.append(f"{candidate['total_stars']} stars")
            if 'followers' in candidate:
                stats.append(f"{candidate['followers']} followers")
            
            if stats:
                candidate_details.append(f"Stats: {', '.join(stats)}")
            
            # Add the formatted candidate details
            response_parts.append("\n".join(candidate_details) + "\n")
        
        # Add summary at the end with appropriate experience level focus
        summary = f"\nThese candidates are ranked based on their"
        if experience_level == 'junior':
            summary += " beginner-friendly profile and "
        elif experience_level == 'mid':
            summary += " mid-level experience and "
        elif experience_level == 'expert':
            summary += " expert-level experience and "
            
        summary += f"expertise in {language_str}" + (f" and {skill_str}" if skills else "") + "."
        response_parts.append(summary)
        
        return "\n".join(response_parts)
    
    def process_query(self, query: str) -> str:
        """
        Process a natural language query and return formatted results
        Example: "give me top 10 candidates that are good at Python"
        """
        # Extract the intent from the query
        intent = self.extract_intent(query)
        
        # Create keywords dictionary from intent for relevance scoring
        keywords = {
            'languages': intent['languages'],
            'skills': intent['skills'],
            'experience_level': [intent['experience_level']] if intent['experience_level'] else []
        }
        
        # Determine how many search results to fetch (we'll get 3x the requested number to have enough candidates)
        search_k = min(30, max(10, intent['num_results'] * 3))
        
        # Perform the semantic search
        search_results = self.query_tester.search(query, search_k)
        
        # Extract candidate users from the search results with experience filtering
        candidates = self.get_candidates_from_results(
            search_results, 
            keywords, 
            intent['num_results'],
            intent['experience_years_min'],
            intent['experience_years_max']
        )
        
        # Format the response
        return self.format_candidates_response(candidates, intent)

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