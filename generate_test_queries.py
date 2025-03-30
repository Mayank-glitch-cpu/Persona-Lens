import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Any
import ast

def load_features_dataset() -> pd.DataFrame:
    """Load the dataset with extracted features"""
    file_path = Path('Dataset/dataset_with_extracted_features.csv')
    df = pd.read_csv(file_path)
    # Convert string representations of lists and dicts to Python objects
    df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '[]' else [])
    df['language_stats_parsed'] = df['language_stats_parsed'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '{}' else {})
    return df

def get_primary_language(row):
    """Extract primary language from language_stats_parsed"""
    try:
        lang_stats = row['language_stats_parsed']
        if isinstance(lang_stats, list):
            # Convert list to dict
            lang_dict = {item['language']: item['percentage'] 
                        for item in lang_stats 
                        if isinstance(item, dict) and 'language' in item and 'percentage' in item}
            if lang_dict:
                return max(lang_dict.items(), key=lambda x: x[1])[0]
        elif isinstance(lang_stats, dict):
            if lang_stats:
                return max(lang_stats.items(), key=lambda x: x[1])[0]
    except (TypeError, AttributeError, KeyError):
        pass
    return None

def create_language_based_queries(df: pd.DataFrame) -> List[Dict[Any, Any]]:
    """Create test queries based on programming language expertise"""
    queries = []
    
    # Group users by their primary language
    for _, row in df.iterrows():
        try:
            lang_stats = row['language_stats_parsed']
            if not isinstance(lang_stats, dict) and not isinstance(lang_stats, list):
                continue
            
            # Convert to dict if it's a list
            if isinstance(lang_stats, list):
                lang_stats = {item['language']: item['percentage'] 
                            for item in lang_stats 
                            if isinstance(item, dict) and 'language' in item and 'percentage' in item}
            
            if not lang_stats:
                continue
                
            # Get primary language (highest percentage)
            primary_lang = max(lang_stats.items(), key=lambda x: x[1])
            
            # Find similar users (those who use the same primary language)
            similar_users = df[
                df['language_stats_parsed'].apply(
                    lambda x: (
                        isinstance(x, (dict, list)) and
                        (isinstance(x, dict) and primary_lang[0] in x or
                         isinstance(x, list) and any(
                             isinstance(item, dict) and
                             item.get('language') == primary_lang[0]
                             for item in x
                         ))
                    )
                )
            ]
            
            if len(similar_users) >= 5:  # Only create query if we have enough similar users
                queries.append({
                    'id': f"lang_{row['username']}",
                    'user_id': row['username'],
                    'type': 'language_expertise',
                    'primary_language': primary_lang[0],
                    'relevant_indices': similar_users.index.tolist()[:10]  # Top 10 similar users
                })
        except (TypeError, AttributeError, KeyError) as e:
            # Skip problematic rows
            continue
    
    return queries[:50]  # Limit to 50 queries for manageability

def create_skill_based_queries(df: pd.DataFrame) -> List[Dict[Any, Any]]:
    """Create test queries based on skill sets"""
    queries = []
    
    for _, row in df.iterrows():
        if not row['skills']:
            continue
            
        # Find users with similar skill sets
        user_skills = set(row['skills'])
        if len(user_skills) < 3:  # Skip users with too few skills
            continue
            
        # Calculate skill overlap with other users
        skill_similarities = []
        for idx, other_row in df.iterrows():
            other_skills = set(other_row['skills'])
            if other_skills:
                overlap = len(user_skills & other_skills) / len(user_skills | other_skills)
                skill_similarities.append((idx, overlap))
        
        # Sort by similarity and get top matches
        similar_users = sorted(skill_similarities, key=lambda x: x[1], reverse=True)[:10]
        if similar_users and similar_users[0][1] > 0.3:  # Only create query if we have good matches
            # Create query variations based on skill combinations
            skill_list = list(user_skills)
            top_skills = skill_list[:3]  # Use top 3 skills for variations
            
            variations = [
                f"developers skilled in {', '.join(top_skills)}",
                f"experts with {' and '.join(top_skills)} experience",
            ]
            
            # Add role-based variations
            role_skills = {
                'frontend': {'react', 'vue', 'angular', 'javascript', 'typescript', 'css'},
                'backend': {'python', 'java', 'node.js', 'django', 'spring'},
                'devops': {'docker', 'kubernetes', 'aws', 'jenkins', 'terraform'},
                'ml': {'machine learning', 'tensorflow', 'pytorch', 'data science'}
            }
            
            for role, role_skills_set in role_skills.items():
                if len(user_skills & role_skills_set) >= 2:
                    variations.append(f"{role} developers with {' and '.join(top_skills)} skills")
            
            for variation in variations:
                queries.append({
                    'id': f"skills_{row['username']}_{len(queries)}",
                    'user_id': row['username'],
                    'type': 'skill_based',
                    'query': variation,
                    'skills': list(user_skills),
                    'relevant_indices': [idx for idx, _ in similar_users]
                })
    
    return queries[:50]  # Limit to 50 queries for manageability

def create_popularity_based_queries(df: pd.DataFrame) -> List[Dict[Any, Any]]:
    """Create test queries based on user popularity metrics"""
    queries = []
    
    # Calculate popularity score based on multiple metrics
    df['popularity_score'] = (
        df['total_stars'].rank(pct=True) * 0.5 +
        df['followers'].rank(pct=True) * 0.5
    )
    
    # Extract primary language for each user
    df['primary_language'] = df.apply(get_primary_language, axis=1)
    
    popularity_ranges = [
        (0.8, float('inf'), 'outstanding'),
        (0.6, 0.8, 'notable'),
        (0.4, 0.6, 'established'),
        (0.2, 0.4, 'rising'),
        (0, 0.2, 'emerging')
    ]
    
    for _, row in df.iterrows():
        score = row['popularity_score']
        
        # Find popularity range
        for min_score, max_score, level in popularity_ranges:
            if min_score <= score < max_score:
                # Find similar users by popularity
                similar_users = df[
                    (df['popularity_score'] >= score - 0.1) &
                    (df['popularity_score'] <= score + 0.1)
                ]
                
                if len(similar_users) >= 5:
                    # Create query variations based on popularity and metrics
                    variations = [
                        f"{level} developers with high impact",
                        f"highly regarded {row['primary_language']} developers" if row['primary_language'] else None,
                        f"developers with significant {row['primary_language']} contributions" if row['primary_language'] else None,
                        f"{level} developers in {', '.join(row['skills'][:2])}" if row['skills'] else None
                    ]
                    
                    # Filter out None values
                    variations = [v for v in variations if v is not None]
                    
                    for variation in variations:
                        queries.append({
                            'id': f"popularity_{row['username']}_{len(queries)}",
                            'user_id': row['username'],
                            'type': 'popularity_based',
                            'query': variation,
                            'popularity_score': score,
                            'popularity_level': level,
                            'relevant_indices': similar_users.index.tolist()[:10]
                        })
                break
    
    return queries[:50]  # Limit to 50 queries for manageability

def main():
    try:
        # Load dataset
        df = load_features_dataset()
        print(f"Loaded dataset with {len(df)} records")
        
        # Generate different types of test queries
        language_queries = create_language_based_queries(df)
        skill_queries = create_skill_based_queries(df)
        popularity_queries = create_popularity_based_queries(df)
        
        # Combine all queries
        all_queries = {
            'language_based': language_queries,
            'skill_based': skill_queries,
            'popularity_based': popularity_queries
        }
        
        # Save queries to file
        output_file = 'test_queries.json'
        with open(output_file, 'w') as f:
            json.dump(all_queries, f, indent=2)
        
        print(f"\nGenerated test queries:")
        print(f"Language-based queries: {len(language_queries)}")
        print(f"Skill-based queries: {len(skill_queries)}")
        print(f"Popularity-based queries: {len(popularity_queries)}")
        print(f"\nSaved queries to {output_file}")
        
    except Exception as e:
        print(f"Error generating test queries: {str(e)}")

if __name__ == "__main__":
    main()