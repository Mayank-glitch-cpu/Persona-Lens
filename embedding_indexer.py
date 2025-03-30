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
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with a specific transformer model"""
        # Force model to use CPU
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_mapping = {}
        self.chunk_counter = 0

    def chunk_to_text(self, chunk: Dict[str, Any]) -> str:
        """Convert a chunk to a text representation for embedding"""
        text_parts = []
        
        # Add basic metrics if they exist
        if 'size' in chunk:
            text_parts.append(f"Size: {chunk['size']}")
        if 'avg_popularity' in chunk:
            text_parts.append(f"Average popularity: {chunk['avg_popularity']:.2f}")
        if 'avg_experience' in chunk:
            text_parts.append(f"Average experience years: {chunk['avg_experience']:.2f}")
        
        # Add skills
        if 'most_common_skills' in chunk:
            skills = list(chunk['most_common_skills'].keys())
            if skills:
                text_parts.append(f"Skills: {', '.join(skills)}")
        
        # Add languages
        if 'most_used_languages' in chunk:
            languages = list(chunk['most_used_languages'].keys())
            if languages:
                text_parts.append(f"Languages: {', '.join(languages)}")
        
        # Add any sample users
        if 'sample_users' in chunk and chunk['sample_users']:
            text_parts.append(f"Sample users: {', '.join(chunk['sample_users'][:5])}")
            
        return " | ".join(text_parts)

    def process_chunks(self, chunks_data: Dict[str, Any]) -> None:
        """Process all types of chunks and create embeddings"""
        # Process user clusters
        if 'user_clusters' in chunks_data:
            for cluster_name, cluster_data in chunks_data['user_clusters'].items():
                text = self.chunk_to_text(cluster_data)
                # Add language information to cluster data
                if 'most_used_languages' in cluster_data:
                    languages = {lang: count for lang, count in cluster_data['most_used_languages'].items()}
                    cluster_data['languages'] = languages
                self.add_to_index(text, {'type': 'cluster', 'name': cluster_name, 'data': cluster_data})
        
        # Process feature groups
        if 'feature_groups' in chunks_data:
            for group_name, group_data in chunks_data['feature_groups'].items():
                if isinstance(group_data, dict):
                    text = json.dumps(group_data)
                else:
                    text = str(group_data)
                self.add_to_index(text, {'type': 'feature_group', 'name': group_name, 'data': group_data})
        
        # Process user chunks
        if 'user_chunks' in chunks_data:
            for chunk_type, users in chunks_data['user_chunks'].items():
                if chunk_type.startswith('language_based_'):
                    # Handle language-based chunks
                    lang = chunk_type.replace('language_based_', '')
                    text = f"Language {lang} developers"
                    user_data = []
                    for user in users:
                        if isinstance(user, dict):
                            # Extract user's languages from their repositories
                            user_info = {
                                'username': user.get('username', 'Unknown'),
                                'languages': user.get('languages', {}),
                                'experience_years': user.get('experience_years', 0),
                                'popularity_score': user.get('popularity_score', 0),
                                'public_repos': user.get('public_repos', 0),
                                'total_stars': user.get('total_stars', 0),
                                'total_forks': user.get('total_forks', 0),
                                'followers': user.get('followers', 0),
                                'following': user.get('following', 0)
                            }
                            # Add language counts if available
                            if 'repository_languages' in user:
                                user_info['languages'] = user['repository_languages']
                            user_data.append(user_info)
                    self.add_to_index(text, {
                        'type': 'language_chunk',
                        'name': f'language_based_{lang}',
                        'data': user_data
                    })
                elif isinstance(users, list):
                    # Handle experience and popularity based chunks
                    text = f"{chunk_type}: {len(users)} users"
                    user_data = []
                    for user in users:
                        if isinstance(user, dict):
                            user_info = {
                                'username': user.get('username', 'Unknown'),
                                'languages': user.get('languages', {}),
                                'experience_years': user.get('experience_years', 0),
                                'popularity_score': user.get('popularity_score', 0),
                                'public_repos': user.get('public_repos', 0),
                                'total_stars': user.get('total_stars', 0),
                                'total_forks': user.get('total_forks', 0),
                                'followers': user.get('followers', 0),
                                'following': user.get('following', 0)
                            }
                            if 'repository_languages' in user:
                                user_info['languages'] = user['repository_languages']
                            user_data.append(user_info)
                    self.add_to_index(text, {
                        'type': 'user_chunk',
                        'name': chunk_type,
                        'data': user_data
                    })

    def add_to_index(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a text chunk to the FAISS index with its metadata"""
        embedding = self.model.encode([text])[0]
        self.index.add(np.array([embedding]))
        self.chunk_mapping[self.chunk_counter] = metadata
        self.chunk_counter += 1

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar chunks using a query string"""
        try:
            query_vector = self.model.encode([query])[0]
            
            # Check if dimensions match
            if query_vector.shape[0] != self.dimension:
                print(f"Warning: Query vector dimension ({query_vector.shape[0]}) doesn't match index dimension ({self.dimension})")
                # Resize query vector to match index dimension
                if query_vector.shape[0] > self.dimension:
                    query_vector = query_vector[:self.dimension]  # Truncate
                else:
                    # Pad with zeros
                    padded_vector = np.zeros(self.dimension)
                    padded_vector[:query_vector.shape[0]] = query_vector
                    query_vector = padded_vector
            
            distances, indices = self.index.search(np.array([query_vector]), k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1:  # -1 indicates no match found
                    results.append((float(distance), self.chunk_mapping[idx]))
            
            return results
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            # Return mock results when search fails
            mock_results = []
            # Use the most popular chunks as fallback
            for idx in range(min(k, len(self.chunk_mapping))):
                if idx in self.chunk_mapping:
                    mock_results.append((999.0, self.chunk_mapping[idx]))
            
            if not mock_results and len(self.chunk_mapping) > 0:
                # Get first k available chunks
                available_keys = list(self.chunk_mapping.keys())[:k]
                for idx in available_keys:
                    mock_results.append((999.0, self.chunk_mapping[idx]))
            
            return mock_results

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