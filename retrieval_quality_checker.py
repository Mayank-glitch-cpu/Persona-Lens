import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import ndcg_score, average_precision_score, f1_score
from embedding_indexer import ChunkEmbedder
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval_debug.log'),
        logging.StreamHandler()
    ]
)

class RetrievalEvaluator:
    def __init__(self):
        self.embedder = ChunkEmbedder()
        self.embedder.load()
        
    def evaluate_queries(self, queries: List[Dict], k: int = 10) -> Dict[str, Any]:
        logging.info(f"Starting evaluation of {len(queries)} queries")
        
        results = {
            'detailed_results': [],
            'metrics_by_category': {}
        }
        
        for query in tqdm(queries, desc="Evaluating queries"):
            query_type = query.get('type', '')
            query_text = query.get('query', query.get('text', ''))
            logging.debug(f"Processing query: {query_text[:100]}... (type: {query_type})")
            
            # Get search results
            search_results = self.embedder.search(query_text, k=k)
            logging.debug(f"Got {len(search_results)} search results")
            
            retrieved_indices = []
            for _, metadata in search_results:
                if 'indices' in metadata:
                    retrieved_indices.extend(metadata['indices'])
                elif 'data' in metadata and isinstance(metadata['data'], dict) and 'user_indices' in metadata['data']:
                    retrieved_indices.extend(metadata['data']['user_indices'])
            
            logging.debug(f"Retrieved indices: {retrieved_indices[:k]}")
            
            # Get relevant indices
            relevant_indices = []
            if 'relevant_indices' in query:
                relevant_indices = query['relevant_indices']
            elif 'relevance' in query:
                relevant_indices = [idx for idx, is_relevant in enumerate(query['relevance']) if is_relevant]
                
            logging.debug(f"Relevant indices: {relevant_indices}")
            
            # Convert to sets for metric calculation
            retrieved_set = set(retrieved_indices[:k])
            relevant_set = set(relevant_indices)
            
            if relevant_set:  # Only calculate metrics if we have relevant indices
                # Calculate metrics
                precision_at_k = self._calculate_precision_at_k(list(retrieved_set), relevant_set, k)
                ndcg = self._calculate_ndcg(list(retrieved_set), relevant_set, k)
                map_score = self._calculate_map(list(retrieved_set), relevant_set)
                mrr = self._calculate_mrr(list(retrieved_set), relevant_set)
                f1 = self._calculate_f1(list(retrieved_set), relevant_set)
                recall = self._calculate_recall(list(retrieved_set), relevant_set)
                
                # Store individual query results
                query_result = {
                    'query_id': query.get('id', str(len(results['detailed_results']))),
                    'type': query_type,
                    'precision@k': precision_at_k,
                    'ndcg': ndcg,
                    'map': map_score,
                    'mrr': mrr,
                    'f1': f1,
                    'recall': recall,
                    'retrieved_indices': list(retrieved_set),
                    'relevant_indices': list(relevant_set)
                }
                results['detailed_results'].append(query_result)
                
                # Update type-specific metrics
                if query_type not in results['metrics_by_category']:
                    results['metrics_by_category'][query_type] = {
                        'precisions_at_k': [],
                        'ndcg_scores': [],
                        'map_scores': [],
                        'mrr_scores': [],
                        'f1_scores': [],
                        'recall_scores': [],
                        'count': 0
                    }
                metrics = results['metrics_by_category'][query_type]
                metrics['precisions_at_k'].append(precision_at_k)
                metrics['ndcg_scores'].append(ndcg)
                metrics['map_scores'].append(map_score)
                metrics['mrr_scores'].append(mrr)
                metrics['f1_scores'].append(f1)
                metrics['recall_scores'].append(recall)
                metrics['count'] += 1
        
        # Calculate aggregate metrics by category
        for query_type, metrics in results['metrics_by_category'].items():
            results['metrics_by_category'][query_type] = {
                'avg_precision@k': np.mean(metrics['precisions_at_k']) if metrics['count'] > 0 else 0.0,
                'avg_ndcg': np.mean(metrics['ndcg_scores']) if metrics['count'] > 0 else 0.0,
                'avg_map': np.mean(metrics['map_scores']) if metrics['count'] > 0 else 0.0,
                'avg_mrr': np.mean(metrics['mrr_scores']) if metrics['count'] > 0 else 0.0,
                'avg_f1': np.mean(metrics['f1_scores']) if metrics['count'] > 0 else 0.0,
                'avg_recall': np.mean(metrics['recall_scores']) if metrics['count'] > 0 else 0.0,
                'count': metrics['count']
            }
        
        return results
    
    def _calculate_precision_at_k(self, retrieved: List[int], relevant: set, k: int) -> float:
        """Calculate precision@k"""
        if not retrieved or not relevant:
            return 0.0
        return len(set(retrieved[:k]) & relevant) / k
    
    def _calculate_ndcg(self, retrieved: List[int], relevant: set, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not retrieved or not relevant:
            return 0.0
        
        relevance = np.zeros(k)
        for i, idx in enumerate(retrieved[:k]):
            if idx in relevant:
                relevance[i] = 1
                
        ideal_relevance = np.zeros(k)
        ideal_relevance[:len(relevant)] = 1
        
        return ndcg_score([ideal_relevance], [relevance])
    
    def _calculate_map(self, retrieved: List[int], relevant: set) -> float:
        """Calculate Mean Average Precision"""
        if not retrieved or not relevant:
            return 0.0
        
        y_true = [1 if idx in relevant else 0 for idx in retrieved]
        y_score = [1/(i+1) for i in range(len(retrieved))]
        
        return average_precision_score(y_true, y_score)
    
    def _calculate_mrr(self, retrieved: List[int], relevant: set) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not retrieved or not relevant:
            return 0.0
            
        for i, idx in enumerate(retrieved):
            if idx in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_f1(self, retrieved: List[int], relevant: set) -> float:
        """Calculate F1 Score"""
        if not retrieved or not relevant:
            return 0.0
            
        retrieved_set = set(retrieved)
        precision = len(retrieved_set & relevant) / len(retrieved_set)
        recall = len(retrieved_set & relevant) / len(relevant)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_recall(self, retrieved: List[int], relevant: set) -> float:
        """Calculate Recall"""
        if not retrieved or not relevant:
            return 0.0
        return len(set(retrieved) & relevant) / len(relevant)

def main():
    # Load test queries
    try:
        with open('test_queries.json', 'r') as f:
            query_data = json.load(f)
        
        logging.info(f"Loaded test queries: {len(query_data)} categories")
        logging.debug(f"Query categories: {list(query_data.keys())}")
        
        total_queries = sum(len(queries) for queries in query_data.values())
        logging.info(f"Total number of queries across all categories: {total_queries}")
        
        for category, queries in query_data.items():
            logging.debug(f"Category '{category}' has {len(queries)} queries")
            if len(queries) > 0:
                logging.debug(f"Sample query from '{category}': {queries[0]}")
    
    except FileNotFoundError:
        logging.error("test_queries.json not found")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing test_queries.json: {e}")
        return

    print(f"Loaded test queries: {len(query_data)} total queries")
    
    # Create evaluator
    evaluator = RetrievalEvaluator()
    
    # Evaluate each query type
    all_results = {
        'metrics_by_category': {},
        'detailed_results': []
    }
    
    for query_type, queries in query_data.items():
        print(f"\nEvaluating {query_type} queries...")
        results = evaluator.evaluate_queries(queries)
        
        # Merge results
        all_results['metrics_by_category'].update(results['metrics_by_category'])
        all_results['detailed_results'].extend(results['detailed_results'])
    
    print("\nEvaluation Results by Category:")
    for category, metrics in all_results['metrics_by_category'].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"Number of queries: {metrics['count']}")
        print(f"Average Precision@k: {metrics['avg_precision@k']:.3f}")
        print(f"Average NDCG: {metrics['avg_ndcg']:.3f}")
        print(f"Mean Average Precision: {metrics['avg_map']:.3f}")
        print(f"Mean Reciprocal Rank: {metrics['avg_mrr']:.3f}")
        print(f"Average F1 Score: {metrics['avg_f1']:.3f}")
        print(f"Average Recall: {metrics['avg_recall']:.3f}")
    
    # Save detailed results
    with open('retrieval_evaluation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved evaluation results to retrieval_evaluation.json")

if __name__ == "__main__":
    main()