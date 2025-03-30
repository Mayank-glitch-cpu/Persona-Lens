from embedding_indexer import ChunkEmbedder
from typing import List, Tuple, Dict, Any
import json
import re
import numpy as np

class QueryTester:
    def __init__(self):
        """Initialize the query tester with the embedder"""
        self.embedder = ChunkEmbedder()
        # Load the existing index and mapping
        self.embedder.load()
        
        # Load the semantic chunks for reference
        with open('semantic_chunks.json', 'r') as f:
            self.semantic_chunks = json.load(f)

        # ML-specific skills mapping
        self.ml_skill_categories = {
            'core_ml': {
                'machine learning', 'deep learning', 'neural networks', 'artificial intelligence',
                'statistical modeling', 'reinforcement learning'
            },
            'frameworks': {
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost', 'lightgbm',
                'fastai', 'mxnet', 'caffe'
            },
            'data_processing': {
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
                'jupyter', 'data visualization', 'feature engineering'
            },
            'specialized': {
                'computer vision', 'natural language processing', 'nlp', 'speech recognition',
                'recommendation systems', 'time series analysis', 'anomaly detection'
            },
            'ml_ops': {
                'mlflow', 'kubeflow', 'airflow', 'ml pipelines', 'model deployment',
                'model monitoring', 'feature store'
            }
        }

    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract important keywords from the query with enhanced ML focus"""
        keywords = {
            'languages': [],
            'skills': [],
            'experience_level': []
        }
        
        # Common programming languages with ML focus
        languages = ['python', 'r', 'julia', 'scala', 'java', 'c++', 
                    'javascript', 'typescript', 'go', 'rust']
        
        # Experience levels
        experience_levels = ['junior', 'senior', 'expert', 'lead', 'experienced']
        
        # ML-specific skills (consolidated from all categories)
        ml_skills = set()
        for category_skills in self.ml_skill_categories.values():
            ml_skills.update(category_skills)
        
        # Additional general skills
        general_skills = {
            'web development', 'backend', 'frontend', 'full stack', 
            'devops', 'cloud', 'aws', 'azure', 'docker', 'kubernetes',
            'data science', 'data engineering', 'data analysis',
            'big data', 'distributed systems', 'system design'
        }
        
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        # Find languages
        for lang in languages:
            if lang in query_lower:
                keywords['languages'].append(lang)
        
        # Find experience levels
        for level in experience_levels:
            if level in query_lower:
                keywords['experience_level'].append(level)
        
        # Find ML-specific skills (check for both exact and partial matches)
        for skill in ml_skills:
            if skill in query_lower or any(token in skill.split() for token in query_tokens):
                keywords['skills'].append(skill)
        
        # Find general skills
        for skill in general_skills:
            if skill in query_lower:
                keywords['skills'].append(skill)
        
        return keywords

    def score_relevance(self, user_data: Dict[str, Any], keywords: Dict[str, List[str]]) -> float:
        """Score how relevant a user is to search keywords with enhanced ML focus"""
        # Initialize scoring components
        language_score = 0.0
        ml_score = 0.0
        experience_score = 0.0
        impact_score = 0.0
        
        # ML-specific keywords that indicate expertise
        ml_frameworks = {'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost'}
        ml_concepts = {'machine learning', 'deep learning', 'neural networks', 'ai', 'artificial intelligence'}
        ml_domains = {'computer vision', 'nlp', 'natural language processing', 'reinforcement learning'}
        data_skills = {'pandas', 'numpy', 'data science', 'data analysis', 'jupyter'}
        
        # Score language expertise (30% weight)
        if keywords['languages'] and 'languages' in user_data:
            user_langs = {k.lower(): v for k, v in user_data['languages'].items()}
            for lang in keywords['languages']:
                if lang.lower() in user_langs:
                    value = user_langs[lang.lower()]
                    if isinstance(value, dict):
                        count = value.get('count', 0)
                        language_score += min(count / 5.0, 2.0)
                    elif isinstance(value, (int, float)):
                        if value > 1:
                            language_score += min(value / 5.0, 2.0)
                        else:
                            language_score += min(value * 2, 2.0)
        
        # Score ML expertise (40% weight)
        if 'skills' in user_data:
            user_skills = {s.lower() for s in user_data['skills']}
            
            # Framework expertise
            framework_matches = ml_frameworks.intersection(user_skills)
            ml_score += len(framework_matches) * 1.5
            
            # Core ML concepts
            concept_matches = ml_concepts.intersection(user_skills)
            ml_score += len(concept_matches) * 2.0
            
            # Specialized domains
            domain_matches = ml_domains.intersection(user_skills)
            ml_score += len(domain_matches) * 1.8
            
            # Data processing skills
            data_matches = data_skills.intersection(user_skills)
            ml_score += len(data_matches) * 1.0
        
        # Score experience level (15% weight)
        if 'experience_years' in user_data:
            years = user_data['experience_years']
            if years >= 8:  # Expert
                experience_score = 3.0
            elif years >= 5:  # Senior
                experience_score = 2.0
            elif years >= 3:  # Mid-level
                experience_score = 1.0
        
        # Score impact metrics (15% weight)
        stars = user_data.get('total_stars', 0)
        forks = user_data.get('total_forks', 0)
        repos = user_data.get('public_repos', 0)
        followers = user_data.get('followers', 0)
        
        # Use log scaling for better distribution
        impact_score += np.log1p(stars) * 0.3
        impact_score += np.log1p(forks) * 0.2
        impact_score += min(repos / 20.0, 1.0)
        impact_score += min(np.log1p(followers) * 0.1, 1.0)
        
        # Combine scores with weights
        total_score = (
            (language_score * 0.3) +
            (ml_score * 0.4) +
            (experience_score * 0.15) +
            (impact_score * 0.15)
        )
        
        return total_score

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