from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import Dict, List, Any, Tuple
import pickle
from pathlib import Path
<<<<<<< Updated upstream
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
=======
import json

class ChunkEmbedder:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """Initialize with a more powerful model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Create a flat IP index for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        
>>>>>>> Stashed changes
        self.chunk_mapping = {}
        self.chunk_counter = 0
        
        # Technical terminology mapping for better context
        self.tech_context = self._load_tech_context()
        
    def _load_tech_context(self) -> Dict[str, Dict[str, List[str]]]:
        """Load technical context for better semantic matching"""
        return {
            'languages': {
                'python': ['django', 'flask', 'fastapi', 'pytorch', 'tensorflow', 'numpy', 'pandas'],
                'javascript': ['node.js', 'react', 'vue', 'angular', 'express', 'typescript'],
                'java': ['spring', 'hibernate', 'maven', 'gradle', 'jakarta ee'],
                'cpp': ['boost', 'qt', 'cmake', 'stl', 'modern c++'],
                'go': ['gin', 'echo', 'kubernetes', 'docker', 'microservices'],
                'rust': ['actix', 'tokio', 'cargo', 'wasm', 'systems programming']
            },
            'domains': {
                'web': ['frontend', 'backend', 'fullstack', 'api', 'rest', 'graphql'],
                'ml': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'data science'],
                'devops': ['ci/cd', 'docker', 'kubernetes', 'aws', 'azure', 'terraform'],
                'mobile': ['android', 'ios', 'react native', 'flutter', 'mobile development'],
                'systems': ['operating systems', 'embedded', 'networking', 'distributed systems']
            },
            'skills': {
                'frontend': ['html', 'css', 'javascript', 'react', 'vue', 'angular'],
                'backend': ['api design', 'databases', 'microservices', 'cloud'],
                'devops': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                'ml': ['pytorch', 'tensorflow', 'scikit-learn', 'computer vision', 'nlp'],
                'architecture': ['system design', 'scalability', 'performance', 'security']
            }
        }

    def chunk_to_text(self, chunk: Dict[str, Any]) -> List[str]:
        """Convert chunk to multiple text representations with enhanced skill and popularity context"""
        text_representations = []
        
        chunk_type = chunk.get('type', '')
        data = chunk.get('data', {})
        
        # Base representations based on chunk type
        if chunk_type.startswith('language_based_'):
            lang = chunk_type.replace('language_based_', '')
            text_representations.extend([
                f"Expert {lang} developers and programmers",
                f"Professional {lang} software engineers",
                f"Experienced {lang} development specialists"
            ])
            
            # Add framework context
            if lang.lower() in self.tech_context['languages']:
                frameworks = self.tech_context['languages'][lang.lower()]
                text_representations.extend([
                    f"{lang} developer with {framework} expertise"
                    for framework in frameworks[:3]  # Top 3 frameworks
                ])
                
        elif chunk_type.startswith('skill_based_'):
            skill = chunk_type.replace('skill_based_', '')
            
            # Enhanced skill descriptions
            text_representations.extend([
                f"Developers specializing in {skill}",
                f"Technical experts in {skill}",
                f"Professional {skill} specialists",
                f"Software engineers with {skill} expertise",
                f"Developers proficient in {skill}"
            ])
            
            # Add domain-specific context
            for domain, skills in self.tech_context['skills'].items():
                if any(s.lower() in skill.lower() for s in skills):
                    text_representations.extend([
                        f"{domain} specialists with {skill} expertise",
                        f"{skill} experts in {domain} development",
                        f"{domain} developers proficient in {skill}"
                    ])
            
            # Add skill combinations from data
            if isinstance(data, dict) and 'most_common_skills' in data:
                skills = []
                if isinstance(data['most_common_skills'], dict):
                    skills = list(data['most_common_skills'].keys())
                elif isinstance(data['most_common_skills'], list):
                    skills = [item[0] for item in data['most_common_skills']]
                
                if skills:
                    text_representations.extend([
                        f"Developer with expertise in {', '.join(skills[:3])}",
                        f"Software engineer skilled in {' and '.join(skills[:2])}",
                        f"Technical professional with {skills[0]} focus"
                    ])
        
        # Enhanced popularity context
        if isinstance(data, dict):
            if 'popularity_score' in data or 'avg_popularity' in data:
                pop_score = float(data.get('popularity_score', data.get('avg_popularity', 0)))
                
                if pop_score > 0.8:
                    text_representations.extend([
                        "Outstanding developer with high community impact",
                        "Highly influential software engineer",
                        "Top-tier developer with significant contributions"
                    ])
                elif pop_score > 0.6:
                    text_representations.extend([
                        "Prominent developer in the community",
                        "Well-respected software engineer",
                        "Notable contributor with strong impact"
                    ])
                elif pop_score > 0.4:
                    text_representations.extend([
                        "Established developer with good reputation",
                        "Recognized software engineer",
                        "Active community contributor"
                    ])
            
            # Enhanced experience context
            if 'experience_years' in data or 'avg_experience' in data:
                exp_years = float(data.get('experience_years', data.get('avg_experience', 0)))
                
                if exp_years > 8:
                    text_representations.extend([
                        f"Senior developer with {int(exp_years)} years experience",
                        "Seasoned software engineer",
                        "Veteran developer"
                    ])
                elif exp_years > 4:
                    text_representations.extend([
                        f"Mid-level developer with {int(exp_years)} years experience",
                        "Experienced software engineer",
                        "Established developer"
                    ])
                else:
                    text_representations.extend([
                        f"Junior developer with {int(exp_years)} years experience",
                        "Early career software engineer",
                        "Growing developer"
                    ])
            
            # Enhanced language expertise context
            if 'most_used_languages' in data:
                languages = []
                if isinstance(data['most_used_languages'], dict):
                    languages = list(data['most_used_languages'].keys())
                elif isinstance(data['most_used_languages'], list):
                    languages = [item[0] for item in data['most_used_languages']]
                
                if languages:
                    text_representations.extend([
                        f"Developer proficient in {', '.join(languages[:3])}",
                        f"Software engineer with {' and '.join(languages[:2])} expertise",
                        f"Technical professional specializing in {languages[0]}"
                    ])
        
        return list(set(text_representations))  # Remove duplicates

    def add_to_index(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add text chunk to index with multiple representations"""
        # Get multiple text representations
        text_representations = self.chunk_to_text(metadata)
        if not text_representations:
            text_representations = [text]
        
        # Create embeddings for all representations
        try:
            embeddings = self.model.encode(text_representations, convert_to_numpy=True)
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Ensure correct shape and type
            if embeddings.shape[1] != self.dimension:
                print(f"Warning: Embedding dimension mismatch. Expected {self.dimension}, got {embeddings.shape[1]}")
                embeddings = np.pad(embeddings, ((0, 0), (0, self.dimension - embeddings.shape[1])))
            
            embeddings = embeddings.astype(np.float32)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Store mapping
            for _ in range(len(text_representations)):
                self.chunk_mapping[self.chunk_counter] = metadata
                self.chunk_counter += 1
                
        except Exception as e:
            print(f"Error adding embeddings for text: {str(e)}")
            return

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Enhanced search with query expansion and reranking"""
        # Generate query variations
        query_variations = self._expand_query(query)
        
        # Get embeddings for all variations
        query_embeddings = self.model.encode(query_variations)
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        faiss.normalize_L2(query_embeddings)
        
        # Search with all variations and collect unique results
        seen_indices = set()
        all_results = []
        
        for q_embed in query_embeddings:
            distances, indices = self.index.search(q_embed.reshape(1, -1), k * 2)
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx not in seen_indices:
                    seen_indices.add(idx)
                    result = (float(dist), self.chunk_mapping[idx])
                    all_results.append(result)
        
        # Rerank results using multiple criteria
        reranked_results = self._rerank_results(query, all_results)
        
        # Further diversify results based on query type
        query_lower = query.lower()
        if any(term in query_lower for term in ['skill', 'expert', 'specialist']):
            reranked_results = self._diversify_by_skills(reranked_results, k)
        elif any(term in query_lower for term in ['popular', 'influential', 'outstanding']):
            reranked_results = self._diversify_by_popularity(reranked_results, k)
        elif any(term in query_lower for term in ['language', 'programmer', 'developer']):
            reranked_results = self._diversify_by_languages(reranked_results, k)
            
        return reranked_results[:k]

<<<<<<< Updated upstream
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
=======
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with enhanced variations and technical context"""
        variations = [query]
        query_lower = query.lower()
        
        # Add technical context
        for domain, terms in self.tech_context.items():
            for category, items in terms.items():
                if category.lower() in query_lower:
                    variations.extend([
                        f"{query} with {item} expertise"
                        for item in items
                    ])
                    variations.extend([
                        f"{query} specializing in {item}"
                        for item in items
                    ])
                    variations.extend([
                        f"{query} proficient in {item}"
                        for item in items
                    ])
                for item in items:
                    if item.lower() in query_lower:
                        variations.extend([
                            f"{category} specialists with {item} expertise",
                            f"experienced {item} developers",
                            f"professional {item} engineers",
                            f"developers proficient in {item}",
                            f"software engineers skilled in {item}"
                        ])
        
        # Add experience level variations
        exp_variations = {
            'junior': ['entry level', 'early career', 'growing'],
            'mid-level': ['intermediate', 'experienced', 'established'],
            'senior': ['senior level', 'expert', 'veteran', 'seasoned']
        }
        
        for level, synonyms in exp_variations.items():
            if level in query_lower or any(syn in query_lower for syn in synonyms):
                base = query.replace(level, '').strip()
                variations.extend([
                    f"{syn} {base}" for syn in synonyms
                ])
        
        # Add popularity/influence variations
        pop_variations = {
            'outstanding': ['highly influential', 'top-tier', 'exceptional'],
            'prominent': ['well-respected', 'notable', 'recognized'],
            'established': ['respected', 'proven', 'reliable']
        }
        
        for level, synonyms in pop_variations.items():
            if level in query_lower or any(syn in query_lower for syn in synonyms):
                base = query.replace(level, '').strip()
                variations.extend([
                    f"{syn} {base}" for syn in synonyms
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in variations if not (x in seen or seen.add(x))]

    def _rerank_results(self, query: str, results: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        """Rerank results with improved skill matching and scoring balance"""
        reranked = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Enhanced focus detection with weighted terms
        query_focus = {
            'skill': 0.0,
            'language': 0.0,
            'popularity': 0.0,
            'experience': 0.0
        }
        
        focus_indicators = {
            'skill': {
                'high': ['specialist', 'expert', 'proficient', 'skilled in', 'expertise in'],
                'medium': ['skilled', 'experienced with', 'knowledge of'],
                'low': ['familiar', 'work with', 'used']
            },
            'language': {
                'high': ['developer', 'programmer', 'engineer', 'coding in'],
                'medium': ['coder', 'writing', 'developing in'],
                'low': ['using', 'with', 'knows']
            },
            'popularity': {
                'high': ['outstanding', 'top', 'best', 'highly regarded', 'influential'],
                'medium': ['popular', 'well-known', 'recognized'],
                'low': ['known', 'active', 'contributor']
            },
            'experience': {
                'high': ['senior', 'veteran', 'lead', 'principal'],
                'medium': ['mid-level', 'experienced', 'intermediate'],
                'low': ['junior', 'entry', 'beginning']
            }
        }
        
        # Improved query focus analysis
        for focus_type, weight_dict in focus_indicators.items():
            max_score = 0.0
            for weight, terms in weight_dict.items():
                weight_value = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[weight]
                # Check for both exact matches and phrase matches
                for term in terms:
                    if ' ' in term:
                        if term in query_lower:
                            max_score = max(max_score, weight_value * 1.2)  # Boost for phrase matches
                    elif term in query_terms:
                        max_score = max(max_score, weight_value)
            query_focus[focus_type] = max_score
        
        # Normalize focus scores while preserving relative importance
        total_focus = sum(query_focus.values()) or 1
        for key in query_focus:
            query_focus[key] = (query_focus[key] / total_focus) * 1.5  # Amplify differences
            
        for score, metadata in results:
            boost = 0.0
            data = metadata.get('data', {})
            chunk_type = metadata.get('type', '')
            
            if not isinstance(data, dict):
                continue
            
            # Enhanced skill matching with contextual scoring
            if 'most_common_skills' in data:
                skills = set()
                if isinstance(data['most_common_skills'], dict):
                    skills.update((k.lower(), v) for k, v in data['most_common_skills'].items())
                elif isinstance(data['most_common_skills'], list):
                    skills.update((item[0].lower(), item[1]) if isinstance(item, (list, tuple)) else (item.lower(), 1.0) 
                                for item in data['most_common_skills'])
                
                # Progressive skill matching with frequency weighting
                matched_skills = []
                for skill, freq in skills:
                    # Check for both exact and partial matches
                    if skill in query_lower or any(term in skill or skill in term for term in query_terms):
                        matched_skills.append((skill, freq))
                
                if matched_skills:
                    # Weight boost by skill frequency and number of matches
                    skill_boost = sum(freq for _, freq in matched_skills) / len(skills)
                    skill_boost *= (1 + 0.2 * len(matched_skills))  # Progressive bonus for multiple matches
                    boost += skill_boost * query_focus['skill']
            
            # Enhanced language expertise matching
            if 'most_used_languages' in data:
                languages = {}
                if isinstance(data['most_used_languages'], dict):
                    languages.update(data['most_used_languages'])
                elif isinstance(data['most_used_languages'], list):
                    for item in data['most_used_languages']:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            languages[item[0]] = item[1]
                        elif isinstance(item, str):
                            languages[item] = 1.0
                
                # Progressive language matching with usage weighting
                for lang, usage in languages.items():
                    lang_lower = lang.lower()
                    if lang_lower in query_lower:
                        boost += 0.3 * usage * query_focus['language']
                        if any(pattern.format(lang_lower) in query_lower 
                              for pattern in ["{} developer", "{} programmer", "{} engineer"]):
                            boost += 0.2 * usage  # Extra boost for role-specific matches
            
            # Improved popularity scoring with dynamic thresholds
            if ('popularity_score' in data or 'avg_popularity' in data) and query_focus['popularity'] > 0:
                pop_score = float(data.get('popularity_score', data.get('avg_popularity', 0)))
                
                # Dynamic popularity thresholds based on query focus
                if 'outstanding' in query_lower or 'top' in query_lower:
                    threshold = 0.8
                elif 'well-known' in query_lower or 'popular' in query_lower:
                    threshold = 0.6
                else:
                    threshold = 0.4
                
                # Progressive popularity boost with threshold consideration
                if pop_score >= threshold:
                    boost += (0.5 + (pop_score - threshold) * 0.5) * query_focus['popularity']
                else:
                    # Small boost for near-threshold scores
                    if pop_score >= threshold - 0.2:
                        boost += 0.2 * query_focus['popularity']
            
            # Enhanced experience matching with role context
            if 'experience_years' in data and query_focus['experience'] > 0:
                exp_years = float(data.get('experience_years', 0))
                exp_boost = 0.0
                
                # Dynamic experience thresholds
                senior_threshold = 8 if 'senior' in query_lower else 6
                mid_threshold = 4 if 'mid' in query_lower else 3
                
                if exp_years >= senior_threshold and ('senior' in query_lower or 'experienced' in query_lower):
                    exp_boost = 0.4
                elif mid_threshold <= exp_years < senior_threshold and ('mid' in query_lower or 'intermediate' in query_lower):
                    exp_boost = 0.3
                elif exp_years < mid_threshold and ('junior' in query_lower or 'entry' in query_lower):
                    exp_boost = 0.35
                
                boost += exp_boost * query_focus['experience']
            
            # Type-specific boosts with context consideration
            type_boosts = {
                'skill_based_': 0.3 if query_focus['skill'] > 0.3 else 0.1,
                'language_based_': 0.3 if query_focus['language'] > 0.3 else 0.1,
                'user_clusters': 0.25 if query_focus['popularity'] > 0.3 else 0.1
            }
            
            for type_prefix, type_boost in type_boosts.items():
                if chunk_type.startswith(type_prefix):
                    boost += type_boost
            
            # Normalize final score while preserving relative rankings
            final_score = (score + boost) / (1 + boost)  # Ensures score stays in [0,1]
            reranked.append((final_score, metadata))
        
        return sorted(reranked, key=lambda x: x[0], reverse=True)

    def _diversify_by_skills(self, results: List[Tuple[float, Dict[str, Any]]], k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Ensure diversity in skill-based results with progressive thresholds"""
        seen_skills = set()
        diversified = []
        remaining = []
        
        for score, metadata in results:
            data = metadata.get('data', {})
            if not isinstance(data, dict):
                continue
                
            skills = set()
            if 'most_common_skills' in data:
                if isinstance(data['most_common_skills'], dict):
                    skills.update(k.lower() for k in data['most_common_skills'].keys())
                elif isinstance(data['most_common_skills'], list):
                    skills.update(item[0].lower() for item in data['most_common_skills'])
            
            # Always include results with unique skills
            unique_skills = skills - seen_skills
            if unique_skills:
                seen_skills.update(skills)
                diversified.append((score * 1.1, metadata))  # Boost unique skill matches
            else:
                # Calculate overlap ratio
                overlap = len(skills & seen_skills) / len(skills) if skills else 1.0
                # Use progressive thresholds based on current diversity
                threshold = 0.9 if len(diversified) < k/2 else 0.7
                if overlap < threshold:
                    remaining.append((score * (1 - overlap * 0.3), metadata))
            
            if len(diversified) >= k:
                break
        
        # Add remaining results if needed
        if len(diversified) < k and remaining:
            remaining.sort(key=lambda x: x[0], reverse=True)
            diversified.extend(remaining[:k - len(diversified)])
        
        return sorted(diversified, key=lambda x: x[0], reverse=True)[:k]

    def _diversify_by_popularity(self, results: List[Tuple[float, Dict[str, Any]]], k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Ensure diversity in popularity-based results with improved clustering"""
        popularity_clusters = {}  # Map popularity ranges to best results
        diversified = []
        
        for score, metadata in results:
            data = metadata.get('data', {})
            if not isinstance(data, dict):
                continue
                
            # Prefer user clusters for popularity queries
            if metadata.get('type', '').startswith('user_clusters'):
                score *= 1.2
            
            pop_score = float(data.get('popularity_score', data.get('avg_popularity', 0)))
            
            # Create more granular ranges for better distribution
            if pop_score > 0.8:
                pop_range = 'very_high'
            elif pop_score > 0.6:
                pop_range = 'high'
            elif pop_score > 0.4:
                pop_range = 'medium'
            elif pop_score > 0.2:
                pop_range = 'low'
            else:
                pop_range = 'very_low'
            
            # Keep track of best result for each popularity range
            if pop_range not in popularity_clusters or score > popularity_clusters[pop_range][0]:
                popularity_clusters[pop_range] = (score, metadata)
        
        # Add results in order of popularity ranges
        for range_key in ['very_high', 'high', 'medium', 'low', 'very_low']:
            if range_key in popularity_clusters:
                diversified.append(popularity_clusters[range_key])
                if len(diversified) >= k:
                    break
                    
        return diversified[:k]

    def _diversify_by_languages(self, results: List[Tuple[float, Dict[str, Any]]], k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Ensure diversity in language-based results"""
        seen_languages = set()
        diversified = []
        
        for score, metadata in results:
            data = metadata.get('data', {})
            if not isinstance(data, dict):
                continue
                
            languages = set()
            if 'most_used_languages' in data:
                if isinstance(data['most_used_languages'], dict):
                    languages.update(k.lower() for k in data['most_used_languages'].keys())
                elif isinstance(data['most_used_languages'], list):
                    languages.update(item[0].lower() for item in data['most_used_languages'])
            
            # Add result if it has at least one new language
            if languages - seen_languages:
                seen_languages.update(languages)
                diversified.append((score, metadata))
                if len(diversified) >= k:
                    break
                    
        return diversified

    def save(self, directory: str = 'embeddings'):
        """Save index and mappings"""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / 'chunk_index.faiss'))
        
        # Save mapping
        with open(path / 'chunk_mapping.pkl', 'wb') as f:
            pickle.dump(self.chunk_mapping, f)
            
    def load(self, directory: str = 'embeddings'):
        """Load index and mappings"""
        path = Path(directory)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / 'chunk_index.faiss'))
        
        # Load mapping
        with open(path / 'chunk_mapping.pkl', 'rb') as f:
            self.chunk_mapping = pickle.load(f)

    def process_chunks(self, chunks_data: Dict) -> None:
        """Process and add all chunks to the index"""
        print("Processing chunks and creating embeddings...")
        
        def extract_indices(data):
            """Helper to safely extract indices from chunk data"""
            if isinstance(data, dict):
                return data.get('user_indices', [])
            elif isinstance(data, list):
                return [i for i, _ in enumerate(data)]
            return []
            
        for chunk_type, chunks in chunks_data.items():
            if isinstance(chunks, dict):  # Handle nested structure
                for chunk_name, chunk_data in chunks.items():
                    metadata = {
                        'type': chunk_type,
                        'name': chunk_name,
                        'data': chunk_data,
                        'indices': extract_indices(chunk_data)
                    }
                    self.add_to_index(chunk_name, metadata)
            elif isinstance(chunks, list):  # Handle list structure
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        metadata = {
                            'type': chunk_type,
                            'name': chunk.get('name', str(i)),
                            'data': chunk,
                            'indices': extract_indices(chunk)
                        }
                        self.add_to_index(metadata['name'], metadata)
                        
        print(f"Added {self.chunk_counter} chunks to index")
>>>>>>> Stashed changes

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