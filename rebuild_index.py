import json
from embedding_indexer import ChunkEmbedder
import numpy as np

def main():
    print("Loading semantic chunks...")
    with open('semantic_chunks.json', 'r') as f:
        chunks_data = json.load(f)

    print("Creating embedder with enhanced model...")
    embedder = ChunkEmbedder(model_name='all-mpnet-base-v2')

    print("Processing chunks and creating embeddings...")
    embedder.process_chunks(chunks_data)

    print("Saving new index and mapping...")
    embedder.save()

    print("Testing search functionality with ML-specific queries...")
    test_queries = [
        "machine learning experts with Python experience",
        "deep learning specialists with research background",
        "data scientists with ML framework expertise",
        "AI engineers with production experience"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = embedder.search(query, k=3)
        for distance, chunk in results:
            print(f"Distance: {distance:.2f}")
            print(f"Type: {chunk['type']}")
            print(f"Name: {chunk['name']}")

if __name__ == "__main__":
    main()