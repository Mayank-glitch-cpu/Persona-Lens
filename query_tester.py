from embedding_indexer import ChunkEmbedder
from typing import List, Tuple, Dict, Any
import json
import re

class QueryTester:
    def __init__(self):
        """Initialize the query tester with the embedder"""
        self.embedder = ChunkEmbedder()
        # Load the existing index and mapping
        self.embedder.load()
        
        # Load the semantic chunks for reference
        with open('semantic_chunks.json', 'r') as f:
            self.semantic_chunks = json.load(f)

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract important keywords from the query"""
        keywords = {
            'languages': [],
            'skills': [],
            'experience_level': []
        }
        
        # Common programming languages
        languages = ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php']
        # Common experience levels
        experience_levels = ['junior', 'senior', 'expert', 'lead', 'experienced']
        # Common skills
        skills = ['machine learning', 'web development', 'backend', 'frontend', 'full stack', 
                 'devops', 'cloud', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'vue', 
                 'angular', 'node.js', 'django', 'flask', 'spring']
        
        query_lower = query.lower()
        
        # Find languages
        for lang in languages:
            if lang in query_lower:
                keywords['languages'].append(lang)
        
        # Find experience levels
        for level in experience_levels:
            if level in query_lower:
                keywords['experience_level'].append(level)
        
        # Find skills
        for skill in skills:
            if skill in query_lower:
                keywords['skills'].append(skill)
        
        return keywords

    def score_relevance(self, user_data: Dict[str, Any], keywords: Dict[str, List[str]]) -> float:
        """Score how relevant a user is to the search keywords"""
        score = 0.0
        
        # Score based on languages
        if keywords['languages'] and 'languages' in user_data:
            user_langs = {k.lower(): v for k, v in user_data['languages'].items()}
            for lang in keywords['languages']:
                if lang in user_langs:
                    value = user_langs[lang]
                    if isinstance(value, dict):
                        count = value.get('count', 0)
                        score += min(count / 5.0, 2.0)  # Cap at 2.0 points per language
                    elif isinstance(value, (int, float)):
                        if value > 1:  # Count
                            score += min(value / 5.0, 2.0)
                        else:  # Percentage
                            score += min(value * 2, 2.0)
        
        # Score based on experience level
        if keywords['experience_level'] and 'experience_years' in user_data:
            years = user_data['experience_years']
            if 'junior' in keywords['experience_level'] and years < 3:
                score += 1.5
            elif 'senior' in keywords['experience_level'] and years >= 5:
                score += 1.5
            elif ('expert' in keywords['experience_level'] or 'experienced' in keywords['experience_level']) and years >= 8:
                score += 2.0
        
        # Score based on popularity if searching for top developers
        if 'popularity_score' in user_data:
            pop_score = user_data['popularity_score']
            score += min(pop_score / 2.0, 1.0)  # Up to 1.0 point for popularity
        
        return score

    def get_user_details(self, user_data: Dict[str, Any], keywords: Dict[str, List[str]] = None) -> str:
        """Format user details into a readable string"""
        details = []
        
        # Add relevance score if keywords provided
        if keywords:
            relevance = self.score_relevance(user_data, keywords)
            if relevance > 0:
                details.append(f"Relevance Score: {relevance:.2f}")
        
        # Add username if available
        if 'username' in user_data:
            details.append(f"Username: {user_data['username']}")
        
        # Add languages if available
        if 'languages' in user_data:
            languages = user_data.get('languages', {})
            if isinstance(languages, dict):
                if languages:  # If not empty
                    lang_list = []
                    # Sort languages by usage (count or percentage)
                    sorted_langs = sorted(languages.items(), key=lambda x: 
                        x[1]['count'] if isinstance(x[1], dict) and 'count' in x[1]
                        else x[1] if isinstance(x[1], (int, float))
                        else 0, 
                        reverse=True
                    )
                    for lang, value in sorted_langs[:5]:
                        if isinstance(value, dict):
                            if 'count' in value:
                                lang_list.append(f"{lang} ({value['count']} repos)")
                            elif 'percentage' in value:
                                lang_list.append(f"{lang} ({value['percentage']*100:.1f}%)")
                        elif isinstance(value, (int, float)):
                            if value > 1:  # Likely a count
                                lang_list.append(f"{lang} ({int(value)} repos)")
                            else:  # Likely a percentage
                                lang_list.append(f"{lang} ({value*100:.1f}%)")
                    details.append(f"Primary Languages: {', '.join(lang_list)}")
                    if len(sorted_langs) > 5:
                        details.append(f"Total Languages: {len(sorted_langs)}")
        
        # Add experience if available
        if 'experience_years' in user_data:
            details.append(f"Experience: {user_data['experience_years']:.1f} years")
        
        # Add popularity metrics
        if 'popularity_score' in user_data:
            details.append(f"Popularity Score: {user_data['popularity_score']:.1f}")
        
        # Add repository stats
        if 'public_repos' in user_data:
            details.append(f"Public Repositories: {user_data['public_repos']:,}")
        if 'total_stars' in user_data:
            details.append(f"Total Stars: {user_data['total_stars']:,}")
        if 'total_forks' in user_data:
            details.append(f"Total Forks: {user_data['total_forks']:,}")
        
        # Add followers/following
        if 'followers' in user_data:
            details.append(f"Followers: {user_data['followers']:,}")
        if 'following' in user_data:
            details.append(f"Following: {user_data['following']:,}")
        
        return "\n".join(details) if details else "No detailed information available"

    def format_result(self, distance: float, chunk: Dict[str, Any], keywords: Dict[str, List[str]] = None) -> str:
        """Format a single search result for display"""
        result = [f"Distance: {distance:.2f}"]
        result.append(f"Type: {chunk['type']}")
        result.append(f"Name: {chunk['name']}")
        
        # Add specific details based on chunk type
        if chunk['type'] == 'cluster':
            data = chunk['data']
            result.append(f"Size: {data['size']}")
            result.append(f"Average Experience: {data['avg_experience']:.2f} years")
            result.append(f"Average Popularity: {data['avg_popularity']:.2f}")
            
            # Add language information
            if data.get('most_used_languages'):
                top_langs = sorted(data['most_used_languages'].items(), key=lambda x: x[1], reverse=True)[:5]
                result.append(f"Top Languages: {', '.join(f'{lang} ({count} repos)' for lang, count in top_langs)}")
            
            # Add skill information
            if data.get('most_common_skills'):
                top_skills = sorted(data['most_common_skills'].items(), key=lambda x: x[1], reverse=True)[:5]
                result.append(f"Top Skills: {', '.join(f'{skill} ({count})' for skill, count in top_skills)}")
                
            # If we have sample users, show them
            if data.get('sample_users'):
                result.append("\nSample Users:")
                for i, user in enumerate(data['sample_users'][:5], 1):
                    result.append(f"\nUser {i}: {user}")
                
        elif chunk['type'] == 'language_chunk':
            data = chunk.get('data', [])
            if isinstance(data, list) and data:
                # Score and sort users by relevance if keywords provided
                if keywords:
                    scored_users = [(user, self.score_relevance(user, keywords)) for user in data]
                    scored_users.sort(key=lambda x: x[1], reverse=True)
                    users = [user for user, _ in scored_users[:5]]
                else:
                    users = data[:5]
                
                result.append(f"\nTop {min(5, len(users))} Users in this Language Group:")
                for i, user in enumerate(users, 1):
                    result.append(f"\nUser {i}:")
                    result.append(self.get_user_details(user, keywords))
            
        elif chunk['type'].startswith('experience_') or chunk['type'].startswith('popularity_'):
            data = chunk.get('data', [])
            if isinstance(data, list) and data:
                # Score and sort users by relevance if keywords provided
                if keywords:
                    scored_users = [(user, self.score_relevance(user, keywords)) for user in data]
                    scored_users.sort(key=lambda x: x[1], reverse=True)
                    users = [user for user, _ in scored_users[:5]]
                else:
                    users = data[:5]
                
                result.append(f"\nTop {min(5, len(users))} Users in this Group:")
                for i, user in enumerate(users, 1):
                    result.append(f"\nUser {i}:")
                    result.append(self.get_user_details(user, keywords))
            
        return "\n".join(result)

    def search(self, query: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar chunks using the query"""
        return self.embedder.search(query, k)

    def interactive_search(self):
        """Run an interactive search session"""
        print("Welcome to the Developer Profile Query Tester!")
        print("Enter your queries to search the developer profiles database.")
        print("Example queries:")
        print("- 'top 10 applicants with exceptional java skills'")
        print("- 'experienced python developers with machine learning expertise'")
        print("- 'highly active JavaScript developers with good popularity'")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
                
            k = input("How many results would you like? (default: 3): ").strip()
            try:
                k = int(k) if k else 3
            except ValueError:
                k = 3
                print("Invalid number, using default of 3 results.")
            
            print(f"\nSearching for: {query}")
            print("=" * 50)
            
            # Extract keywords from query
            keywords = self.extract_keywords(query)
            
            results = self.search(query, k)
            for i, (distance, chunk) in enumerate(results, 1):
                print(f"\nResult {i}:")
                print("-" * 40)
                print(self.format_result(distance, chunk, keywords))
                print("-" * 40)
            
            print("\n")

def main():
    tester = QueryTester()
    tester.interactive_search()

if __name__ == "__main__":
    main()