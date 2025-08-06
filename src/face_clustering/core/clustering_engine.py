"""
Clustering engine for face embeddings
"""
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter

from ..utils.logger import get_logger


class ClusteringEngine:
    """Face embedding clustering engine"""
    
    def __init__(self, algorithm: str = "DBSCAN", **kwargs):
        """
        Initialize clustering engine
        
        Args:
            algorithm: Clustering algorithm ('DBSCAN', 'KMeans', 'Agglomerative')
            **kwargs: Algorithm-specific parameters
        """
        self.logger = get_logger(__name__)
        self.algorithm = algorithm.upper()
        self.params = kwargs
        self.model = None
        self.labels_ = None
        self.embeddings_ = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize clustering model based on algorithm"""
        if self.algorithm == "DBSCAN":
            self.model = DBSCAN(
                eps=self.params.get('eps', 0.45),
                min_samples=self.params.get('min_samples', 2),
                metric=self.params.get('metric', 'cosine')
            )
        elif self.algorithm == "KMEANS":
            self.model = KMeans(
                n_clusters=self.params.get('n_clusters', 8),
                random_state=self.params.get('random_state', 42)
            )
        elif self.algorithm == "AGGLOMERATIVE":
            self.model = AgglomerativeClustering(
                n_clusters=self.params.get('n_clusters', 8),
                linkage=self.params.get('linkage', 'ward')
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
        
        self.logger.info(f"Initialized {self.algorithm} clustering model")
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and predict labels
        
        Args:
            embeddings: Face embeddings array
            
        Returns:
            Cluster labels
        """
        try:
            self.logger.info(f"Clustering {len(embeddings)} embeddings using {self.algorithm}")
            
            if len(embeddings) < 2:
                self.logger.warning("Insufficient data for clustering")
                return np.array([0] * len(embeddings))
            
            self.embeddings_ = embeddings
            self.labels_ = self.model.fit_predict(embeddings)
            
            # Log clustering results
            unique_labels = set(self.labels_)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(self.labels_).count(-1) if -1 in unique_labels else 0
            
            self.logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
            
            return self.labels_
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            raise
    
    def evaluate_clustering(self) -> Dict[str, float]:
        """
        Evaluate clustering performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.labels_ is None or self.embeddings_ is None:
            raise ValueError("No clustering results available. Run fit_predict first.")
        
        try:
            metrics = {}
            
            # Silhouette score (only if we have more than 1 cluster)
            unique_labels = set(self.labels_)
            if len(unique_labels) > 1:
                # Filter out noise points for silhouette score
                mask = self.labels_ != -1
                if np.sum(mask) > 1:
                    metrics['silhouette_score'] = silhouette_score(
                        self.embeddings_[mask], 
                        self.labels_[mask]
                    )
            
            # Basic statistics
            metrics['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
            metrics['n_noise'] = list(self.labels_).count(-1)
            metrics['noise_ratio'] = metrics['n_noise'] / len(self.labels_)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Clustering evaluation failed: {e}")
            return {}
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get detailed cluster statistics
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.labels_ is None:
            raise ValueError("No clustering results available")
        
        counter = Counter(self.labels_)
        stats = {
            'total_points': len(self.labels_),
            'cluster_counts': dict(counter),
            'largest_cluster': counter.most_common(1)[0] if counter else (None, 0),
            'smallest_cluster': counter.most_common()[-1] if counter else (None, 0)
        }
        
        return stats
    
    def optimize_eps(self, embeddings: np.ndarray, 
                    eps_range: Tuple[float, float] = (0.3, 0.6),
                    step: float = 0.05) -> List[Dict[str, Any]]:
        """
        Optimize epsilon parameter for DBSCAN
        
        Args:
            embeddings: Face embeddings
            eps_range: Range of epsilon values to test
            step: Step size for epsilon values
            
        Returns:
            List of results for different epsilon values
        """
        if self.algorithm != "DBSCAN":
            raise ValueError("Epsilon optimization only available for DBSCAN")
        
        results = []
        eps_values = np.arange(eps_range[0], eps_range[1] + step, step)
        
        self.logger.info(f"Optimizing epsilon in range {eps_range} with step {step}")
        
        for eps in eps_values:
            try:
                # Create temporary DBSCAN model
                temp_model = DBSCAN(
                    eps=eps,
                    min_samples=self.params.get('min_samples', 2),
                    metric=self.params.get('metric', 'cosine')
                )
                
                labels = temp_model.fit_predict(embeddings)
                
                # Calculate metrics
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # Calculate silhouette score if possible
                silhouette = None
                if n_clusters > 1:
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(embeddings[mask], labels[mask])
                
                results.append({
                    'eps': eps,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'silhouette_score': silhouette
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate eps={eps}: {e}")
        
        return results
    
    def update_parameters(self, **kwargs):
        """Update clustering parameters and reinitialize model"""
        self.params.update(kwargs)
        self._initialize_model()
        self.logger.info(f"Updated {self.algorithm} parameters: {kwargs}")
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers (only available for some algorithms)"""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        else:
            self.logger.warning(f"Cluster centers not available for {self.algorithm}")
            return np.array([])