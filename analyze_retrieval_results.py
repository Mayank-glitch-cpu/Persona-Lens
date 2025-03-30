import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import precision_recall_curve, auc

class RetrievalAnalyzer:
    def __init__(self):
        self.results = None
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        try:
            # Try to use seaborn-v0_8 style if available
            plt.style.use('seaborn-v0_8')
        except (ImportError, OSError):
            # Fallback to default style if seaborn style is not found
            plt.style.use('default')
            print("Note: Using default style as seaborn style was not found")
        
        # Set color palette
        sns.set_palette("deep")

    def load_results(self, filepath: str = 'retrieval_evaluation.json'):
        """Load the evaluation results"""
        try:
            with open(filepath, 'r') as f:
                self.results = json.load(f)
                print(f"Loaded {len(self.results['detailed_results'])} query results")
        except FileNotFoundError:
            print(f"Error: {filepath} not found")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {filepath}")
            raise

    def plot_retrieval_metrics_distribution(self):
        """Plot distribution of key retrieval metrics"""
        metrics = pd.DataFrame(self.results['detailed_results'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Retrieval Metrics', size=16)
        
        # Precision@k distribution
        sns.histplot(data=metrics, x='precision@k', ax=axes[0,0], bins=20)
        axes[0,0].set_title('Precision@k Distribution')
        
        # Recall distribution
        sns.histplot(data=metrics, x='recall', ax=axes[0,1], bins=20)
        axes[0,1].set_title('Recall Distribution')
        
        # F1 Score distribution
        sns.histplot(data=metrics, x='f1', ax=axes[1,0], bins=20)
        axes[1,0].set_title('F1 Score Distribution')
        
        # NDCG distribution
        sns.histplot(data=metrics, x='ndcg', ax=axes[1,1], bins=20)
        axes[1,1].set_title('NDCG Distribution')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metric_distributions.png')
        plt.close()

    def plot_query_type_performance(self):
        """Plot performance metrics by query type"""
        metrics = pd.DataFrame(self.results['detailed_results'])
        
        # Calculate mean metrics by query type
        type_metrics = metrics.groupby('type').agg({
            'precision@k': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'ndcg': 'mean'
        }).reset_index()
        
        # Create radar plot
        categories = type_metrics['type'].unique()
        metrics_cols = ['precision@k', 'recall', 'f1', 'ndcg']
        
        angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for category in categories:
            values = type_metrics[type_metrics['type'] == category][metrics_cols].values.flatten()
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=category)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_cols)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
        plt.title("Performance by Query Type")
        
        plt.savefig(self.plots_dir / 'query_type_performance.png')
        plt.close()

    def plot_precision_recall_curve(self):
        """Plot precision-recall curve for different query types"""
        metrics = pd.DataFrame(self.results['detailed_results'])
        
        plt.figure(figsize=(10, 6))
        
        for query_type in metrics['type'].unique():
            type_data = metrics[metrics['type'] == query_type]
            precision, recall, _ = precision_recall_curve(
                type_data['relevant_indices'].apply(lambda x: len(x) > 0),
                type_data['precision@k']
            )
            plt.plot(recall, precision, label=f'{query_type} (AUC={auc(recall, precision):.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves by Query Type')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plots_dir / 'precision_recall_curves.png')
        plt.close()

    def plot_retrieval_rank_analysis(self):
        """Analyze where relevant results appear in ranking"""
        metrics = pd.DataFrame(self.results['detailed_results'])
        
        rank_positions = []
        query_types = []
        
        for _, row in metrics.iterrows():
            retrieved = row['retrieved_indices']
            relevant = set(row['relevant_indices'])
            
            for rank, idx in enumerate(retrieved, 1):
                if idx in relevant:
                    rank_positions.append(rank)
                    query_types.append(row['type'])
        
        rank_df = pd.DataFrame({'Rank': rank_positions, 'Query Type': query_types})
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=rank_df, x='Query Type', y='Rank')
        plt.title('Distribution of Relevant Result Ranks by Query Type')
        plt.xticks(rotation=45)
        
        plt.savefig(self.plots_dir / 'rank_distribution.png')
        plt.close()

    def analyze_all(self):
        """Run complete analysis"""
        print("Starting retrieval quality analysis...")
        
        print("\nGenerating plots...")
        self.plot_retrieval_metrics_distribution()
        self.plot_query_type_performance()
        self.plot_precision_recall_curve()
        self.plot_retrieval_rank_analysis()
        
        print("\nAnalysis complete! The following plots were generated in the 'plots' directory:")
        print("- metric_distributions.png")
        print("- query_type_performance.png")
        print("- precision_recall_curves.png")
        print("- rank_distribution.png")

def main():
    analyzer = RetrievalAnalyzer()
    try:
        analyzer.load_results()
        analyzer.analyze_all()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()