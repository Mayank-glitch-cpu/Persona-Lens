import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import ast
from pathlib import Path

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

def create_user_chunks(user_profiles, n_chunks=5):
    """Create semantic chunks of users based on their profiles"""
    chunks = {}
    
    # Group users by experience level
    exp_ranges = [
        (0, 5, 'junior'),
        (5, 10, 'mid_level'),
        (10, 15, 'senior'),
        (15, float('inf'), 'expert')
    ]
    
    for min_exp, max_exp, level in exp_ranges:
        chunks[f'experience_{level}'] = [
            profile for profile in user_profiles 
            if min_exp <= profile['experience_years'] < max_exp
        ]
    
    # Group users by primary language
    language_chunks = {}
    for profile in user_profiles:
        if profile['primary_language']:
            if profile['primary_language'] not in language_chunks:
                language_chunks[profile['primary_language']] = []
            language_chunks[profile['primary_language']].append(profile)
    
    # Keep only the top 10 language groups by size
    top_languages = sorted(
        language_chunks.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]
    
    chunks['language_based'] = {
        lang: users for lang, users in top_languages
    }
    
    # Group users by popularity score ranges
    popularity_ranges = [
        (0, 4, 'low'),
        (4, 7, 'medium'),
        (7, 9, 'high'),
        (9, float('inf'), 'very_high')
    ]
    
    for min_pop, max_pop, level in popularity_ranges:
        chunks[f'popularity_{level}'] = [
            profile for profile in user_profiles 
            if min_pop <= profile['popularity_score'] < max_pop
        ]
    
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
    
    # Save the results
    results = {
        'user_clusters': user_clusters,
        'feature_groups': feature_chunks,
        'user_chunks': user_based_chunks
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