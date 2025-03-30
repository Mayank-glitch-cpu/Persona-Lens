import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import ast
from pathlib import Path
from typing import Dict, Any
from collections import Counter

def load_dataset():
    file_path = Path('Dataset/dataset_with_extracted_features.csv')
    df = pd.read_csv(file_path)
    # Convert string representations of lists and dicts to Python objects
    df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '[]' else [])
    df['language_stats_parsed'] = df['language_stats_parsed'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x != '{}' else {})
    return df

def analyze_user_data(df):
    """Analyze individual user data and create user profiles"""
    user_profiles = []
    
    for _, row in df.iterrows():
        # Calculate user's primary language
        primary_language = None
        max_percentage = 0
        if isinstance(row['language_stats_parsed'], list):
            for lang_info in row['language_stats_parsed']:
                if lang_info['percentage'] > max_percentage:
                    max_percentage = lang_info['percentage']
                    primary_language = lang_info['language']
        
        # Create user profile
        profile = {
            'username': row['username'],
            'experience_years': row['experience_years'],
            'popularity_score': row['popularity_score'],
            'primary_language': primary_language,
            'total_repos': row['public_repos'],
            'total_stars': row['total_stars'],
            'total_forks': row['total_forks'],
            'followers': row['followers'],
            'following': row['following'],
            'skills': row['skills'],
            'languages': row['language_stats_parsed']
        }
        user_profiles.append(profile)
    
    return user_profiles

def create_language_based_chunks(df: pd.DataFrame) -> Dict[str, Any]:
    """Create chunks based on primary programming language"""
    chunks = {}
    
    for _, row in df.iterrows():
        if not isinstance(row['language_stats_parsed'], list):
            continue
            
        # Convert language stats to dict
        lang_dict = {item['language']: item['percentage'] 
                    for item in row['language_stats_parsed'] 
                    if isinstance(item, dict) and 'language' in item}
        
        if not lang_dict:
            continue
            
        # Get primary language
        primary_lang = max(lang_dict.items(), key=lambda x: x[1])[0].lower()
        chunk_name = f"language_based_{primary_lang}"
        
        if chunk_name not in chunks:
            chunks[chunk_name] = {
                'users': [],
                'user_indices': [],
                'skills': set(),
                'experience_years': [],
                'popularity_scores': []
            }
            
        chunks[chunk_name]['users'].append(row['username'])
        chunks[chunk_name]['user_indices'].append(row.name)  # Add index
        if isinstance(row['skills'], list):
            chunks[chunk_name]['skills'].update(row['skills'])
        chunks[chunk_name]['experience_years'].append(row['experience_years'])
        chunks[chunk_name]['popularity_scores'].append(row['popularity_score'])
    
    # Convert sets to lists and calculate averages
    processed_chunks = {}
    for name, data in chunks.items():
        if len(data['users']) >= 5:  # Only keep chunks with enough users
            processed_chunks[name] = {
                'size': len(data['users']),
                'user_indices': data['user_indices'],  # Include indices
                'sample_users': data['users'][:5],
                'most_common_skills': Counter(list(data['skills'])).most_common(10),
                'avg_experience': np.mean(data['experience_years']),
                'avg_popularity': np.mean(data['popularity_scores'])
            }
    
    return processed_chunks

def create_skill_based_chunks(df: pd.DataFrame) -> Dict[str, Any]:
    """Create chunks based on skill sets"""
    chunks = {}
    
    for _, row in df.iterrows():
        if not isinstance(row['skills'], list):
            continue
            
        for skill in row['skills']:
            skill_lower = skill.lower()
            if skill_lower not in chunks:
                chunks[skill_lower] = {
                    'users': [],
                    'user_indices': [],
                    'languages': set(),
                    'experience_years': [],
                    'popularity_scores': []
                }
                
            chunks[skill_lower]['users'].append(row['username'])
            chunks[skill_lower]['user_indices'].append(row.name)  # Add index
            if isinstance(row['language_stats_parsed'], list):
                chunks[skill_lower]['languages'].update(
                    item['language'] for item in row['language_stats_parsed']
                    if isinstance(item, dict) and 'language' in item
                )
            chunks[skill_lower]['experience_years'].append(row['experience_years'])
            chunks[skill_lower]['popularity_scores'].append(row['popularity_score'])
    
    # Process chunks
    processed_chunks = {}
    for skill, data in chunks.items():
        if len(data['users']) >= 5:  # Only keep chunks with enough users
            chunk_name = f"skill_based_{skill}"
            processed_chunks[chunk_name] = {
                'size': len(data['users']),
                'user_indices': data['user_indices'],  # Include indices
                'sample_users': data['users'][:5],
                'most_used_languages': Counter(list(data['languages'])).most_common(5),
                'avg_experience': np.mean(data['experience_years']),
                'avg_popularity': np.mean(data['popularity_scores'])
            }
    
    return processed_chunks

def create_experience_based_chunks(df: pd.DataFrame) -> Dict[str, Any]:
    """Create chunks based on experience levels"""
    exp_ranges = [
        (0, 2, 'entry_level'),
        (2, 5, 'junior'),
        (5, 8, 'mid_level'),
        (8, 12, 'senior'),
        (12, float('inf'), 'expert')
    ]
    
    chunks = {level: {
        'users': [],
        'user_indices': [],
        'skills': set(),
        'languages': set(),
        'popularity_scores': []
    } for _, _, level in exp_ranges}
    
    for _, row in df.iterrows():
        exp_years = row['experience_years']
        for min_exp, max_exp, level in exp_ranges:
            if min_exp <= exp_years < max_exp:
                chunks[level]['users'].append(row['username'])
                chunks[level]['user_indices'].append(row.name)  # Add index
                if isinstance(row['skills'], list):
                    chunks[level]['skills'].update(row['skills'])
                if isinstance(row['language_stats_parsed'], list):
                    chunks[level]['languages'].update(
                        item['language'] for item in row['language_stats_parsed']
                        if isinstance(item, dict) and 'language' in item
                    )
                chunks[level]['popularity_scores'].append(row['popularity_score'])
                break
    
    # Process chunks
    processed_chunks = {}
    for level, data in chunks.items():
        if len(data['users']) >= 5:
            chunk_name = f"experience_{level}"
            processed_chunks[chunk_name] = {
                'size': len(data['users']),
                'user_indices': data['user_indices'],  # Include indices
                'sample_users': data['users'][:5],
                'most_common_skills': Counter(list(data['skills'])).most_common(10),
                'most_used_languages': Counter(list(data['languages'])).most_common(5),
                'avg_popularity': np.mean(data['popularity_scores'])
            }
    
    return processed_chunks

def create_popularity_based_chunks(df: pd.DataFrame) -> Dict[str, Any]:
    """Create chunks based on user popularity"""
    pop_ranges = [
        (0, 0.2, 'emerging'),
        (0.2, 0.4, 'established'),
        (0.4, 0.6, 'notable'),
        (0.6, 0.8, 'prominent'),
        (0.8, float('inf'), 'outstanding')
    ]
    
    chunks = {level: {
        'users': [],
        'user_indices': [],
        'skills': set(),
        'languages': set(),
        'experience_years': []
    } for _, _, level in pop_ranges}
    
    # Normalize popularity scores
    max_pop = df['popularity_score'].max()
    
    for _, row in df.iterrows():
        norm_pop = row['popularity_score'] / max_pop
        for min_pop, max_pop, level in pop_ranges:
            if min_pop <= norm_pop < max_pop:
                chunks[level]['users'].append(row['username'])
                chunks[level]['user_indices'].append(row.name)  # Add index
                if isinstance(row['skills'], list):
                    chunks[level]['skills'].update(row['skills'])
                if isinstance(row['language_stats_parsed'], list):
                    chunks[level]['languages'].update(
                        item['language'] for item in row['language_stats_parsed']
                        if isinstance(item, dict) and 'language' in item
                    )
                chunks[level]['experience_years'].append(row['experience_years'])
                break
    
    # Process chunks
    processed_chunks = {}
    for level, data in chunks.items():
        if len(data['users']) >= 5:
            chunk_name = f"popularity_{level}"
            processed_chunks[chunk_name] = {
                'size': len(data['users']),
                'user_indices': data['user_indices'],  # Include indices
                'sample_users': data['users'][:5],
                'most_common_skills': Counter(list(data['skills'])).most_common(10),
                'most_used_languages': Counter(list(data['languages'])).most_common(5),
                'avg_experience': np.mean(data['experience_years'])
            }
    
    return processed_chunks

def create_user_chunks(user_profiles, n_chunks=5):
    """Create semantic chunks of users based on their profiles"""
    chunks = {}
    
    # Group users by experience level with more granular ranges
    exp_ranges = [
        (0, 2, 'entry_level'),
        (2, 5, 'junior'),
        (5, 8, 'mid_level'),
        (8, 12, 'senior'),
        (12, float('inf'), 'expert')
    ]
    
    for min_exp, max_exp, level in exp_ranges:
        chunk_users = [
            profile for profile in user_profiles 
            if min_exp <= profile['experience_years'] < max_exp
        ]
        if chunk_users:
            chunks[f'experience_{level}'] = chunk_users
    
    # Group users by primary language - flattened structure
    for profile in user_profiles:
        if profile['primary_language']:
            chunk_name = f"language_based_{profile['primary_language'].lower()}"
            if chunk_name not in chunks:
                chunks[chunk_name] = []
            chunks[chunk_name].append(profile)
    
    # Keep only language groups with enough users
    chunks = {k: v for k, v in chunks.items() if not k.startswith('language_based_') or len(v) >= 5}
    
    # Group users by popularity score with more granular ranges
    popularity_ranges = [
        (0, 3, 'emerging'),
        (3, 6, 'established'),
        (6, 8, 'notable'),
        (8, 9, 'prominent'),
        (9, float('inf'), 'outstanding')
    ]
    
    for min_pop, max_pop, level in popularity_ranges:
        chunk_users = [
            profile for profile in user_profiles 
            if min_pop <= profile['popularity_score'] < max_pop
        ]
        if chunk_users:
            chunks[f'popularity_{level}'] = chunk_users
    
    # Add skill-based chunks
    skill_chunks = {}
    for profile in user_profiles:
        for skill in profile['skills']:
            if skill not in skill_chunks:
                skill_chunks[skill] = []
            skill_chunks[skill].append(profile)
    
    # Keep only skill groups with enough users and significant activity
    for skill, users in skill_chunks.items():
        if len(users) >= 5:  # Minimum 5 users per skill
            avg_popularity = sum(u['popularity_score'] for u in users) / len(users)
            if avg_popularity >= 3:  # Only include skills with decent activity
                chunks[f'skill_based_{skill.lower()}'] = users
    
    return chunks

def preprocess_numerical_features(df):
    numerical_features = ['public_repos', 'total_stars', 'total_forks', 
                         'followers', 'following', 'experience_years', 
                         'popularity_score']
    
    numerical_data = df[numerical_features].copy()
    numerical_data = numerical_data.fillna(numerical_data.mean())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)
    
    return pd.DataFrame(scaled_data, columns=numerical_features)

def analyze_skills(df):
    """Analyze and group users based on their skills"""
    all_skills = {}
    for skills in df['skills']:
        for skill in skills:
            all_skills[skill] = all_skills.get(skill, 0) + 1
    
    return {
        'total_unique_skills': len(all_skills),
        'top_skills': dict(sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10]),
        'skill_distribution': all_skills
    }

def analyze_languages(df):
    """Analyze programming language usage patterns"""
    language_stats = {}
    for stats in df['language_stats_parsed']:
        if isinstance(stats, list):
            for lang_info in stats:
                lang = lang_info.get('language', '')
                percentage = lang_info.get('percentage', 0)
                if lang:
                    if lang not in language_stats:
                        language_stats[lang] = {
                            'total_users': 0,
                            'avg_percentage': 0,
                            'total_percentage': 0
                        }
                    language_stats[lang]['total_users'] += 1
                    language_stats[lang]['total_percentage'] += percentage
    
    # Calculate averages
    for lang in language_stats:
        language_stats[lang]['avg_percentage'] = (
            language_stats[lang]['total_percentage'] / 
            language_stats[lang]['total_users']
        )
    
    return {
        'total_languages': len(language_stats),
        'language_stats': language_stats,
        'top_languages': dict(sorted(
            {k: v['total_users'] for k, v in language_stats.items()}.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
    }

def create_row_chunks(df, n_clusters=5):
    """Create semantic chunks of users based on their characteristics"""
    numerical_data = preprocess_numerical_features(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['user_cluster'] = kmeans.fit_predict(numerical_data)
    
    chunks = {}
    for cluster in range(n_clusters):
        chunk_data = df[df['user_cluster'] == cluster]
        # Analyze skills and languages for this cluster
        cluster_skills = analyze_skills(chunk_data)
        cluster_languages = analyze_languages(chunk_data)
        
        chunk_stats = {
            'size': len(chunk_data),
            'avg_popularity': chunk_data['popularity_score'].mean(),
            'avg_experience': chunk_data['experience_years'].mean(),
            'most_common_skills': cluster_skills['top_skills'],
            'most_used_languages': cluster_languages['top_languages'],
            'sample_users': chunk_data['username'].head(5).tolist()
        }
        chunks[f'cluster_{cluster}'] = chunk_stats
    
    return chunks

def create_column_chunks(df):
    """Create semantic chunks of features including skills and languages"""
    numerical_features = ['public_repos', 'total_stars', 'total_forks', 
                         'followers', 'following', 'experience_years', 
                         'popularity_score']
    
    # Calculate correlation matrix for numerical features
    corr_matrix = df[numerical_features].corr()
    
    # Use PCA to identify feature groups
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(corr_matrix)
    
    # Group features based on their primary PCA component
    feature_groups = {
        'engagement_metrics': [],
        'popularity_metrics': [],
        'activity_metrics': []
    }
    
    for idx, feature in enumerate(numerical_features):
        primary_component = np.argmax(np.abs(pca_result[idx]))
        if primary_component == 0:
            feature_groups['engagement_metrics'].append(feature)
        elif primary_component == 1:
            feature_groups['popularity_metrics'].append(feature)
        else:
            feature_groups['activity_metrics'].append(feature)
    
    # Add skill and language analysis
    feature_groups['skill_metrics'] = analyze_skills(df)
    feature_groups['language_metrics'] = analyze_languages(df)
    
    return feature_groups

def main():
    # Load the dataset
    df = load_dataset()
    
    # Create row-wise chunks (user clusters)
    user_clusters = create_row_chunks(df)
    
    # Create column-wise chunks (feature groups)
    feature_chunks = create_column_chunks(df)
    
    # Analyze individual user data
    user_profiles = analyze_user_data(df)
    
    # Create user-based chunks
    user_based_chunks = create_user_chunks(user_profiles)
    
    # Create language-based chunks
    language_chunks = create_language_based_chunks(df)
    
    # Create skill-based chunks
    skill_chunks = create_skill_based_chunks(df)
    
    # Create experience-based chunks
    experience_chunks = create_experience_based_chunks(df)
    
    # Create popularity-based chunks
    popularity_chunks = create_popularity_based_chunks(df)
    
    # Save the results
    results = {
        'user_clusters': user_clusters,
        'feature_groups': feature_chunks,
        'user_chunks': user_based_chunks,
        'language_chunks': language_chunks,
        'skill_chunks': skill_chunks,
        'experience_chunks': experience_chunks,
        'popularity_chunks': popularity_chunks
    }
    
    # Save to JSON file
    with open('semantic_chunks.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Semantic chunks have been created and saved to 'semantic_chunks.json'")
    
    # Print summaries
    print("\nFeature Groups:")
    for group_name, features in feature_chunks.items():
        if group_name in ['engagement_metrics', 'popularity_metrics', 'activity_metrics']:
            print(f"\n{group_name.replace('_', ' ').title()}:")
            print(", ".join(features))
        elif group_name == 'skill_metrics':
            print("\nTop Skills:")
            for skill, count in features['top_skills'].items():
                print(f"{skill}: {count} users")
        elif group_name == 'language_metrics':
            print("\nTop Programming Languages:")
            for lang, count in features['top_languages'].items():
                print(f"{lang}: {count} users")
    
    print("\nUser-based Chunks:")
    for chunk_type, users in user_based_chunks.items():
        if chunk_type != 'language_based':
            print(f"\n{chunk_type.replace('_', ' ').title()}:")
            print(f"Number of users: {len(users)}")
        else:
            print("\nLanguage-based Groups:")
            for lang, users_list in users.items():
                print(f"{lang}: {len(users_list)} users")

if __name__ == "__main__":
    main()