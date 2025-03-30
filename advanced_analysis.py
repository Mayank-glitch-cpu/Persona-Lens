import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

class AdvancedAnalyzer:
    def __init__(self):
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        try:
            import seaborn as sns
            plt.style.use('seaborn-v0_8')
        except (ImportError, OSError):
            plt.style.use('default')
            print("Note: Using default style as seaborn style was not found")

    def load_evaluation_results(self) -> Dict:
        """Load evaluation results"""
        try:
            with open('retrieval_evaluation.json', 'r') as f:
                data = json.load(f)
                print(f"Loaded evaluation data with {len(data['detailed_results'])} results")
                return data
        except FileNotFoundError:
            print("Error: retrieval_evaluation.json not found")
            return {}
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in retrieval_evaluation.json")
            return {}

    def analyze_all(self):
        results = self.load_evaluation_results()
        if not results:
            return

        # Create plots directory if it doesn't exist
        self.plots_dir.mkdir(exist_ok=True)

        print("Generating analysis plots...")
        
        # Basic retrieval metrics
        self.plot_basic_metrics(results)
        
        # Query type analysis
        self.plot_query_type_performance(results)
        
        # Error analysis
        self.plot_error_distribution(results)
        
        print("\nAnalysis complete! Plots saved in 'plots' directory")

    def plot_basic_metrics(self, results: Dict):
        """Plot basic retrieval metrics"""
        metrics_df = pd.DataFrame(results['detailed_results'])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision distribution
        sns.histplot(data=metrics_df, x='precision@k', ax=ax1, bins=20)
        ax1.set_title('Precision@K Distribution')
        
        # Recall distribution
        sns.histplot(data=metrics_df, x='recall', ax=ax2, bins=20)
        ax2.set_title('Recall Distribution')
        
        plt.tight_layout()
        plt.savefig(str(self.plots_dir / 'basic_metrics.png'))
        plt.close()

    def plot_query_type_performance(self, results: Dict):
        """Plot performance by query type"""
        df = pd.DataFrame(results['detailed_results'])
        
        # Group by query type and calculate mean metrics
        type_metrics = df.groupby('type').agg({
            'precision@k': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'ndcg': 'mean'
        }).reset_index()
        
        # Create radar plot using matplotlib
        metrics = ['precision@k', 'recall', 'f1', 'ndcg']
        categories = type_metrics['type'].unique()
        
        # Compute angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        # Close the plot by appending first value
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for category in categories:
            values = type_metrics[type_metrics['type'] == category][metrics].values.flatten()
            values = np.concatenate((values, [values[0]]))  # Close the plot
            ax.plot(angles, values, 'o-', linewidth=2, label=category)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Performance by Query Type")
        
        plt.savefig(str(self.plots_dir / 'query_type_performance.png'))
        plt.close()

    def plot_error_distribution(self, results: Dict):
        """Plot error distribution analysis"""
        error_data = []
        for result in results['detailed_results']:
            retrieved = set(result['retrieved_indices'])
            relevant = set(result['relevant_indices'])
            
            error_data.append({
                'type': result['type'],
                'false_positives': len(retrieved - relevant),
                'false_negatives': len(relevant - retrieved),
                'precision': result['precision@k'],
                'recall': result['recall']
            })

        df = pd.DataFrame(error_data)

        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

        # False positives
        sns.boxplot(y=df['false_positives'], ax=ax1)
        ax1.set_title('False Positive Distribution')

        # False negatives
        sns.boxplot(y=df['false_negatives'], ax=ax2)
        ax2.set_title('False Negative Distribution')

        # Precision vs Recall scatter
        sns.scatterplot(data=df, x='recall', y='precision', ax=ax3)
        ax3.set_title('Precision vs Recall')

        # Error rates by query type
        error_by_type = df.groupby('type').mean()
        sns.barplot(x=error_by_type.index, y=error_by_type['false_positives'], ax=ax4)
        ax4.set_title('Error Rates by Query Type')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(str(self.plots_dir / 'error_analysis.png'))
        plt.close()

def main():
    analyzer = AdvancedAnalyzer()
    analyzer.analyze_all()

if __name__ == "__main__":
    main()