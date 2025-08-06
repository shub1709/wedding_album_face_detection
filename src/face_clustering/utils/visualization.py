"""
Visualization utilities for face clustering
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .logger import get_logger


class ClusteringVisualizer:
    """Visualization tools for clustering results"""
    
    def __init__(self, output_folder: str, style: str = 'default'):
        """
        Initialize visualizer
        
        Args:
            output_folder: Output directory for plots
            style: Matplotlib style to use
        """
        self.logger = get_logger(__name__)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        if style == 'default':
            sns.set_palette("husl")
    
    def plot_eps_optimization(self, optimization_results: List[Dict], 
                            save_path: Optional[str] = None) -> str:
        """
        Plot epsilon optimization results for DBSCAN
        
        Args:
            optimization_results: Results from epsilon optimization
            save_path: Custom save path
            
        Returns:
            Path to saved plot
        """
        if not optimization_results:
            raise ValueError("No optimization results provided")
        
        save_path = save_path or self.output_folder / "eps_optimization.png"
        
        try:
            eps_values = [r['eps'] for r in optimization_results]
            n_clusters = [r['n_clusters'] for r in optimization_results]
            n_noise = [r['n_noise'] for r in optimization_results]
            silhouette_scores = [r.get('silhouette_score') for r in optimization_results]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot clusters and noise
            ax1.plot(eps_values, n_clusters, 'b.-', label='Clusters', linewidth=2)
            ax1.plot(eps_values, n_noise, 'r.-', label='Noise Points', linewidth=2)
            ax1.set_xlabel('Epsilon')
            ax1.set_ylabel('Count')
            ax1.set_title('DBSCAN Parameter Optimization')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot silhouette scores
            valid_scores = [(eps, score) for eps, score in zip(eps_values, silhouette_scores) if score is not None]
            if valid_scores:
                eps_valid, scores_valid = zip(*valid_scores)
                ax2.plot(eps_valid, scores_valid, 'g.-', label='Silhouette Score', linewidth=2)
                ax2.set_xlabel('Epsilon')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Clustering Quality (Silhouette Score)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Highlight best score
                best_idx = np.argmax(scores_valid)
                ax2.scatter(eps_valid[best_idx], scores_valid[best_idx], 
                           color='red', s=100, zorder=5, label=f'Best (ε={eps_valid[best_idx]:.3f})')
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Eps optimization plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create eps optimization plot: {e}")
            raise
    
    def plot_cluster_distribution(self, labels: np.ndarray, 
                                save_path: Optional[str] = None) -> str:
        """
        Plot cluster size distribution
        
        Args:
            labels: Cluster labels
            save_path: Custom save path
            
        Returns:
            Path to saved plot
        """
        save_path = save_path or self.output_folder / "cluster_distribution.png"
        
        try:
            # Count cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Separate noise from clusters
            cluster_mask = unique_labels != -1
            cluster_labels = unique_labels[cluster_mask]
            cluster_counts = counts[cluster_mask]
            noise_count = counts[~cluster_mask][0] if np.any(~cluster_mask) else 0
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot of cluster sizes
            if len(cluster_labels) > 0:
                bars = ax1.bar(range(len(cluster_labels)), cluster_counts, 
                              color='skyblue', alpha=0.7, edgecolor='navy')
                ax1.set_xlabel('Cluster ID')
                ax1.set_ylabel('Number of Faces')
                ax1.set_title(f'Cluster Sizes ({len(cluster_labels)} clusters)')
                ax1.set_xticks(range(len(cluster_labels)))
                ax1.set_xticklabels([f'P{label}' for label in cluster_labels], rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, cluster_counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom')
                
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No clusters found', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=16)
                ax1.set_title('Cluster Sizes')
            
            # Pie chart of clustered vs unclustered
            total_clustered = np.sum(cluster_counts) if len(cluster_counts) > 0 else 0
            sizes = [total_clustered, noise_count]
            labels_pie = ['Clustered', 'Unclustered']
            colors = ['lightblue', 'lightcoral']
            
            if total_clustered > 0 or noise_count > 0:
                wedges, texts, autotexts = ax2.pie(sizes, labels=labels_pie, colors=colors,
                                                  autopct='%1.1f%%', startangle=90)
                ax2.set_title('Clustered vs Unclustered Faces')
            else:
                ax2.text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=16)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Cluster distribution plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster distribution plot: {e}")
            raise
    
    def plot_embedding_space(self, embeddings: np.ndarray, labels: np.ndarray,
                           method: str = 'tsne', save_path: Optional[str] = None) -> str:
        """
        Plot face embeddings in 2D space using dimensionality reduction
        
        Args:
            embeddings: Face embeddings
            labels: Cluster labels
            method: Dimensionality reduction method ('tsne', 'pca')
            save_path: Custom save path
            
        Returns:
            Path to saved plot
        """
        save_path = save_path or self.output_folder / f"embedding_space_{method}.png"
        
        try:
            self.logger.info(f"Creating {method.upper()} embedding visualization")
            
            # Reduce dimensionality
            if method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                reduced_embeddings = reducer.fit_transform(embeddings)
            elif method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
                explained_var = reducer.explained_variance_ratio_
            else:
                raise ValueError(f"Unsupported reduction method: {method}")
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            # Plot each cluster with different color
            unique_labels = set(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points
                    mask = labels == label
                    plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                              c='black', marker='x', s=50, alpha=0.6, label='Unclustered')
                else:
                    # Cluster points
                    mask = labels == label
                    plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                              c=[color], s=60, alpha=0.7, label=f'Person {label}')
            
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
            
            if method.lower() == 'pca':
                plt.title(f'Face Embeddings in {method.upper()} Space\n'
                         f'Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}')
            else:
                plt.title(f'Face Embeddings in {method.upper()} Space')
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Embedding space plot saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding space plot: {e}")
            raise
    
    def create_summary_dashboard(self, clustering_stats: Dict[str, Any],
                               optimization_results: List[Dict] = None,
                               embeddings: np.ndarray = None,
                               labels: np.ndarray = None) -> str:
        """
        Create a comprehensive dashboard with multiple plots
        
        Args:
            clustering_stats: Statistics from clustering
            optimization_results: Epsilon optimization results
            embeddings: Face embeddings for visualization
            labels: Cluster labels
            
        Returns:
            Path to saved dashboard
        """
        dashboard_path = self.output_folder / "clustering_dashboard.png"
        
        try:
            # Determine subplot layout
            n_plots = 2  # Always have cluster distribution and stats
            if optimization_results:
                n_plots += 1
            if embeddings is not None and labels is not None:
                n_plots += 1
            
            fig = plt.figure(figsize=(20, 5 * ((n_plots + 1) // 2)))
            
            plot_idx = 1
            
            # 1. Cluster distribution
            if labels is not None:
                ax = plt.subplot(2, 2, plot_idx)
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_mask = unique_labels != -1
                cluster_counts = counts[cluster_mask]
                
                if len(cluster_counts) > 0:
                    bars = ax.bar(range(len(cluster_counts)), cluster_counts, 
                                 color='skyblue', alpha=0.7)
                    ax.set_title('Cluster Sizes')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Count')
                    
                plot_idx += 1
            
            # 2. Summary statistics
            ax = plt.subplot(2, 2, plot_idx)
            ax.axis('off')
            
            stats_text = f"""
            Clustering Summary
            ==================
            Total Faces: {clustering_stats.get('total_faces', 'N/A')}
            Clusters: {clustering_stats.get('n_clusters', 'N/A')}
            Noise Points: {clustering_stats.get('n_noise', 'N/A')}
            Noise Ratio: {clustering_stats.get('noise_ratio', 0):.2%}
            """
            
            if 'silhouette_score' in clustering_stats:
                stats_text += f"Silhouette Score: {clustering_stats['silhouette_score']:.3f}"
            
            ax.text(0.1, 0.8, stats_text, fontsize=14, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plot_idx += 1
            
            # 3. Optimization results (if available)
            if optimization_results:
                ax = plt.subplot(2, 2, plot_idx)
                eps_values = [r['eps'] for r in optimization_results]
                n_clusters = [r['n_clusters'] for r in optimization_results]
                
                ax.plot(eps_values, n_clusters, 'b.-', linewidth=2)
                ax.set_title('Epsilon Optimization')
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Number of Clusters')
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
            
            # 4. Embedding visualization (if available)
            if embeddings is not None and labels is not None and len(embeddings) > 1:
                ax = plt.subplot(2, 2, plot_idx)
                
                # Quick PCA for dashboard
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)
                
                unique_labels = set(labels)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    mask = labels == label
                    if label == -1:
                        ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                                 c='black', marker='x', s=30, alpha=0.6)
                    else:
                        ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                                 c=[color], s=30, alpha=0.7)
                
                ax.set_title('Embedding Space (PCA)')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
            
            plt.tight_layout()
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Dashboard saved to: {dashboard_path}")
            return str(dashboard_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            raise