import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Tuple
import pickle
from pathlib import Path
import os
import torch

# Force CPU usage to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.cuda.is_available = lambda: False

class ChunkEmbedder:
    def __init__(self, model_name: str = 'multi-qa-mpnet-base-dot-v1'):
        """Initialize the embedder with a specific transformer model"""
        # Force model to use CPU
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = 768  # Default dimension for mpnet-base models
        self.index = faiss.IndexFlatIP(self.dimension)  # Using Inner Product for better semantic matching
        self.chunk_mapping = {}
        self.chunk_counter = 0

    def chunk_to_text(self, chunk: Dict[str, Any]) -> str:
        """Convert a chunk to a text representation for embedding"""
        text_parts = []
        
        # Add type-specific context
        if chunk.get('type'):
            text_parts.append(f"Profile type: {chunk['type']}")
        
        # Add basic metrics with more context
        if 'size' in chunk:
            text_parts.append(f"Group size: {chunk['size']} developers")
        if 'avg_popularity' in chunk:
            text_parts.append(f"Popularity metrics: {chunk['avg_popularity']:.2f} average score")
        if 'avg_experience' in chunk:
            text_parts.append(f"Professional experience: {chunk['avg_experience']:.2f} years")
        
        # Enhanced skills representation
        if 'most_common_skills' in chunk:
            skills = list(chunk['most_common_skills'].items())
            skills_text = []
            for skill, count in skills:
                skill_level = "expert" if count > 10 else "proficient" if count > 5 else "familiar"
                skills_text.append(f"{skill} ({skill_level})")
            if skills_text:
                text_parts.append(f"Technical expertise: {', '.join(skills_text)}")
        
        # Enhanced language representation
        if 'most_used_languages' in chunk:
            languages = list(chunk['most_used_languages'].items())
            lang_text = []
            for lang, count in languages:
                proficiency = "primary" if count > 50 else "secondary" if count > 20 else "occasional"
                lang_text.append(f"{lang} ({proficiency})")
            if lang_text:
                text_parts.append(f"Programming languages: {', '.join(lang_text)}")
        
        # Project metrics
        if 'total_stars' in chunk:
            text_parts.append(f"Project impact: {chunk['total_stars']} stars")
        if 'total_forks' in chunk:
            text_parts.append(f"Community engagement: {chunk['total_forks']} forks")
        
        # Add sample users with their key strengths
        if 'sample_users' in chunk and chunk['sample_users']:
            user_text = []
            for user in chunk['sample_users'][:5]:
                if isinstance(user, dict):
                    strengths = []
                    if user.get('primary_language'):
                        strengths.append(f"expert in {user['primary_language']}")
                    if user.get('popularity_score', 0) > 7:
                        strengths.append("highly influential")
                    if user.get('experience_years', 0) > 8:
                        strengths.append("senior developer")
                    user_text.append(f"{user.get('username', 'Unknown')} ({', '.join(strengths)})")
                else:
                    user_text.append(user)
            text_parts.append(f"Notable members: {', '.join(user_text)}")
            
        return " | ".join(text_parts)

    def process_chunks(self, chunks_data: Dict[str, Any]) -> None:
        """Process all types of chunks and create embeddings with enhanced semantic context"""
        # Process user clusters with enhanced context
        if 'user_clusters' in chunks_data:
            for cluster_name, cluster_data in chunks_data['user_clusters'].items():
                # Add semantic role information
                role_context = self._infer_cluster_roles(cluster_data)
                cluster_data['inferred_roles'] = role_context
                
                # Generate rich text representation
                text = self.chunk_to_text(cluster_data)
                
                # Add language expertise levels
                if 'most_used_languages' in cluster_data:
                    languages = {lang: {'count': count, 
                                      'expertise': 'expert' if count > 50 else 'proficient' if count > 20 else 'familiar'}
                               for lang, count in cluster_data['most_used_languages'].items()}
                    cluster_data['languages'] = languages
                
                self.add_to_index(text, {'type': 'cluster', 'name': cluster_name, 'data': cluster_data})
        
        # Process feature groups with semantic context
        if 'feature_groups' in chunks_data:
            for group_name, group_data in chunks_data['feature_groups'].items():
                if isinstance(group_data, dict):
                    # Enhance feature group context
                    context = self._add_feature_context(group_name, group_data)
                    text = json.dumps(context)
                else:
                    text = str(group_data)
                self.add_to_index(text, {'type': 'feature_group', 'name': group_name, 'data': group_data})
        
        # Process user chunks with rich profiles
        if 'user_chunks' in chunks_data:
            for chunk_type, users in chunks_data['user_chunks'].items():
                if chunk_type.startswith('language_based_'):
                    # Enhanced language-based grouping
                    lang = chunk_type.replace('language_based_', '')
                    text_parts = [f"Developers specialized in {lang}"]
                    user_data = []
                    
                    for user in users:
                        if isinstance(user, dict):
                            # Create rich user profile
                            user_info = self._create_rich_user_profile(user)
                            user_data.append(user_info)
                            
                            # Add semantic expertise markers
                            if user_info.get('experience_years', 0) > 8:
                                text_parts.append(f"Senior {lang} developer: {user_info['username']}")
                            if user_info.get('popularity_score', 0) > 7:
                                text_parts.append(f"Influential {lang} developer: {user_info['username']}")
                    
                    text = " | ".join(text_parts)
                    self.add_to_index(text, {
                        'type': 'language_chunk',
                        'name': f'language_based_{lang}',
                        'data': user_data
                    })
                elif isinstance(users, list):
                    # Enhanced experience/popularity based grouping
                    text_parts = [f"{chunk_type}: {len(users)} developers"]
                    user_data = []
                    
                    for user in users:
                        if isinstance(user, dict):
                            user_info = self._create_rich_user_profile(user)
                            user_data.append(user_info)
                            
                            # Add key achievements
                            if user_info.get('total_stars', 0) > 1000:
                                text_parts.append(f"High impact developer: {user_info['username']}")
                    
                    text = " | ".join(text_parts)
                    self.add_to_index(text, {
                        'type': 'user_chunk',
                        'name': chunk_type,
                        'data': user_data
                    })

    def _create_rich_user_profile(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """Create an enhanced user profile with semantic context"""
        profile = {
            'username': user.get('username', 'Unknown'),
            'languages': user.get('languages', {}),
            'experience_years': user.get('experience_years', 0),
            'popularity_score': user.get('popularity_score', 0),
            'public_repos': user.get('public_repos', 0),
            'total_stars': user.get('total_stars', 0),
            'total_forks': user.get('total_forks', 0),
            'followers': user.get('followers', 0),
            'following': user.get('following', 0),
            'expertise_level': 'expert' if user.get('experience_years', 0) > 8 else 
                             'senior' if user.get('experience_years', 0) > 5 else 
                             'mid-level' if user.get('experience_years', 0) > 2 else 'junior',
            'impact_score': min(10, (user.get('total_stars', 0) / 1000 + user.get('total_forks', 0) / 500) / 2),
            'community_engagement': min(10, user.get('followers', 0) / 100)
        }
        
        # Add language expertise levels
        if 'repository_languages' in user:
            profile['language_expertise'] = {
                lang: {'count': count,
                      'level': 'expert' if count > 10 else 'proficient' if count > 5 else 'familiar'}
                for lang, count in user['repository_languages'].items()
            }
        
        return profile

    def _infer_cluster_roles(self, cluster_data: Dict[str, Any]) -> List[str]:
        """Infer potential roles based on cluster characteristics"""
        roles = []
        
        # Check for backend expertise
        backend_langs = {'Python', 'Java', 'Go', 'Ruby', 'PHP', 'C++', 'C#'}
        if any(lang in cluster_data.get('most_used_languages', {}) for lang in backend_langs):
            roles.append('backend_developer')
            
        # Check for frontend expertise
        frontend_langs = {'JavaScript', 'TypeScript', 'HTML', 'CSS'}
        if any(lang in cluster_data.get('most_used_languages', {}) for lang in frontend_langs):
            roles.append('frontend_developer')
            
        # Check for data science expertise
        if 'Python' in cluster_data.get('most_used_languages', {}) and \
           any(skill in cluster_data.get('most_common_skills', {}) 
               for skill in ['machine learning', 'data science', 'tensorflow', 'pytorch']):
            roles.append('data_scientist')
            
        # Check for DevOps expertise
        if any(skill in cluster_data.get('most_common_skills', {})
               for skill in ['docker', 'kubernetes', 'aws', 'devops', 'ci/cd']):
            roles.append('devops_engineer')
            
        return roles

    def _add_feature_context(self, group_name: str, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic context to feature groups"""
        context = group_data.copy()
        
        if 'skills' in group_name.lower():
            context['domain'] = 'technical_skills'
        elif 'language' in group_name.lower():
            context['domain'] = 'programming_languages'
        elif any(metric in group_name.lower() for metric in ['popularity', 'stars', 'forks']):
            context['domain'] = 'impact_metrics'
        elif 'experience' in group_name.lower():
            context['domain'] = 'professional_experience'
            
        return context

    def add_to_index(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a text chunk to the FAISS index with its metadata"""
        embedding = self.model.encode([text])[0]
        self.index.add(np.array([embedding]))
        self.chunk_mapping[self.chunk_counter] = metadata
        self.chunk_counter += 1

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar chunks using a query string"""
        query_vector = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_vector]), k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 indicates no match found
                results.append((float(distance), self.chunk_mapping[idx]))
        
        return results

    def save(self, base_path: str = '.') -> None:
        """Save the FAISS index and chunk mapping"""
        base_path = Path(base_path)
        embeddings_dir = base_path / 'embeddings'
        embeddings_dir.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(embeddings_dir / 'chunk_index.faiss'))
        with open(embeddings_dir / 'chunk_mapping.pkl', 'wb') as f:
            pickle.dump(self.chunk_mapping, f)
        print(f"Saved index to {embeddings_dir / 'chunk_index.faiss'}")
        print(f"Saved mapping to {embeddings_dir / 'chunk_mapping.pkl'}")

    def load(self, base_path: str = '.') -> None:
        """Load the FAISS index and chunk mapping"""
        base_path = Path(base_path)
        embeddings_dir = base_path / 'embeddings'
        
        # Check if files exist
        index_path = embeddings_dir / 'chunk_index.faiss'
        mapping_path = embeddings_dir / 'chunk_mapping.pkl'
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}. Did you run 'make embed' first?")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found at {mapping_path}. Did you run 'make embed' first?")
        
        # Load files
        self.index = faiss.read_index(str(index_path))
        with open(mapping_path, 'rb') as f:
            self.chunk_mapping = pickle.load(f)
        self.chunk_counter = len(self.chunk_mapping)
        print(f"Loaded index from {index_path}")
        print(f"Loaded mapping from {mapping_path}")

def main():
    # Load the chunks from the JSON file
    with open('semantic_chunks.json', 'r') as f:
        chunks_data = json.load(f)

    # Create and initialize the embedder
    embedder = ChunkEmbedder()

    # Process all chunks and create embeddings
    embedder.process_chunks(chunks_data)

    # Save the index and mapping
    embedder.save()

    # Example search
    print("\nTesting search functionality:")
    test_queries = [
        "experienced JavaScript developers",
        "machine learning experts with Python",
        "developers with high popularity",
        "junior developers with web development skills"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = embedder.search(query, k=3)
        for distance, chunk in results:
            print(f"Distance: {distance:.2f}")
            print(f"Type: {chunk['type']}")
            print(f"Name: {chunk['name']}")
            if chunk['type'] == 'cluster':
                print(f"Size: {chunk['data']['size']}")
                print(f"Avg Experience: {chunk['data']['avg_experience']:.2f} years")
            print("---")

if __name__ == "__main__":
    main()