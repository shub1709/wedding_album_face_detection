"""
Main pipeline for face clustering system
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from .core.face_detector import FaceDetector
from .core.clustering_engine import ClusteringEngine
from .core.file_organizer import FileOrganizer
from .utils.visualization import ClusteringVisualizer
from .utils.logger import get_logger


class FaceClusteringPipeline:
    """Main pipeline for face clustering operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face clustering pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Storage for pipeline data
        self.face_data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.clustering_stats: Dict[str, Any] = {}
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Face detector
            face_config = self.config.get('face_detection', {})
            self.face_detector = FaceDetector(
                model_name=face_config.get('model_name', 'buffalo_l'),
                providers=face_config.get('providers', ['CPUExecutionProvider']),
                det_size=tuple(face_config.get('detection_size', [640, 640]))
            )
            
            # Clustering engine
            cluster_config = self.config.get('clustering', {})
            self.clustering_engine = ClusteringEngine(
                algorithm=cluster_config.get('algorithm', 'DBSCAN'),
                **{k: v for k, v in cluster_config.items() if k != 'algorithm'}
            )
            
            # File organizer
            output_folder = self.config.get('paths', {}).get('output_folder', 'output')
            self.file_organizer = FileOrganizer(output_folder)
            
            # Visualizer
            self.visualizer = ClusteringVisualizer(output_folder)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def detect_faces(self, input_folder: str, 
                    supported_formats: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Detect faces in all images in input folder
        
        Args:
            input_folder: Path to input folder containing images
            supported_formats: List of supported image formats
            
        Returns:
            List of face data dictionaries
        """
        if supported_formats is None:
            supported_formats = self.config.get('image_processing', {}).get(
                'supported_formats', ['.jpg', '.jpeg', '.png']
            )
        
        min_face_size = self.config.get('face_detection', {}).get('min_face_size', 30)
        
        self.logger.info(f"Starting face detection in folder: {input_folder}")
        
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Get all image files
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No supported image files found in {input_folder}")
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        self.face_data = []
        
        for image_path in tqdm(image_files, desc="Detecting faces"):
            try:
                faces = self.face_detector.detect_faces_from_path(
                    str(image_path), min_face_size
                )
                
                for face in faces:
                    face_info = {
                        'image_path': str(image_path),
                        'embedding': face['embedding'],
                        'face_area': face['area'],
                        'bbox': face['bbox'],
                        'confidence': face['confidence']
                    }
                    self.face_data.append(face_info)
                    
            except Exception as e:
                self.logger.warning(f"Error processing {image_path.name}: {e}")
                continue
        
        self.logger.info(f"Face detection complete: {len(self.face_data)} faces found")
        
        if not self.face_data:
            raise ValueError("No faces found in the input images")
        
        return self.face_data
    
    def cluster_faces(self, optimize_params: bool = True) -> np.ndarray:
        """
        Cluster detected faces
        
        Args:
            optimize_params: Whether to optimize clustering parameters
            
        Returns:
            Cluster labels
        """
        if not self.face_data:
            raise ValueError("No face data available. Run detect_faces first.")
        
        self.logger.info("Starting face clustering")
        
        # Extract embeddings
        self.embeddings = np.vstack([face['embedding'] for face in self.face_data])
        
        # Optimize parameters if requested
        optimization_results = []
        if optimize_params and self.clustering_engine.algorithm == 'DBSCAN':
            cluster_config = self.config.get('clustering', {})
            eps_range = cluster_config.get('eps_range', [0.3, 0.6])
            eps_step = cluster_config.get('eps_step', 0.05)
            
            self.logger.info("Optimizing clustering parameters")
            optimization_results = self.clustering_engine.optimize_eps(
                self.embeddings, 
                tuple(eps_range), 
                eps_step
            )
            
            # Find best epsilon based on silhouette score
            valid_results = [r for r in optimization_results if r['silhouette_score'] is not None]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x['silhouette_score'])
                best_eps = best_result['eps']
                self.clustering_engine.update_parameters(eps=best_eps)
                self.logger.info(f"Using optimized epsilon: {best_eps}")
        
        # Perform clustering
        self.labels = self.clustering_engine.fit_predict(self.embeddings)
        
        # Get clustering statistics
        self.clustering_stats = self.clustering_engine.evaluate_clustering()
        
        # Create visualizations if debug mode is enabled
        if self.config.get('debug', {}).get('enabled', False):
            try:
                if optimization_results:
                    self.visualizer.plot_eps_optimization(optimization_results)
                
                self.visualizer.plot_cluster_distribution(self.labels)
                self.visualizer.plot_embedding_space(self.embeddings, self.labels, 'pca')
                self.visualizer.plot_embedding_space(self.embeddings, self.labels, 'tsne')
                
                # Create dashboard
                self.visualizer.create_summary_dashboard(
                    self.clustering_stats, optimization_results, 
                    self.embeddings, self.labels
                )
                
            except Exception as e:
                self.logger.warning(f"Visualization creation failed: {e}")
        
        self.logger.info(f"Clustering complete: {self.clustering_stats}")
        return self.labels
    
    def organize_results(self, copy_files: bool = True) -> Dict[str, Any]:
        """
        Organize clustered images into folders
        
        Args:
            copy_files: Whether to copy files or create symlinks
            
        Returns:
            Organization statistics
        """
        if self.labels is None or not self.face_data:
            raise ValueError("No clustering results available")
        
        self.logger.info("Organizing clustered images")
        
        # Organize files
        organization_stats = self.file_organizer.organize_clustered_images(
            self.face_data, self.labels, copy_files
        )
        
        # Create summary report
        summary_path = self.file_organizer.create_summary_report(
            organization_stats, self.face_data, self.labels
        )
        
        # Cleanup empty folders
        self.file_organizer.cleanup_empty_folders()
        
        self.logger.info(f"Organization complete. Summary saved to: {summary_path}")
        return organization_stats
    
    def run_full_pipeline(self, input_folder: str, 
                         optimize_params: bool = True,
                         copy_files: bool = True) -> Dict[str, Any]:
        """
        Run complete face clustering pipeline
        
        Args:
            input_folder: Path to input folder with images
            optimize_params: Whether to optimize clustering parameters
            copy_files: Whether to copy files or create symlinks
            
        Returns:
            Complete pipeline results
        """
        try:
            self.logger.info("Starting full face clustering pipeline")
            
            # Step 1: Detect faces
            face_data = self.detect_faces(input_folder)
            
            # Step 2: Cluster faces
            labels = self.cluster_faces(optimize_params)
            
            # Step 3: Organize results
            organization_stats = self.organize_results(copy_files)
            
            # Compile final results
            results = {
                'face_detection': {
                    'total_faces': len(face_data),
                    'unique_images': len(set(face['image_path'] for face in face_data))
                },
                'clustering': self.clustering_stats,
                'organization': organization_stats,
                'pipeline_status': 'completed'
            }
            
            self.logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results = {
                'pipeline_status': 'failed',
                'error': str(e)
            }
            return results
    
    def get_cluster_preview(self, cluster_id: int, max_images: int = 5) -> List[Dict[str, Any]]:
        """
        Get preview of images in a specific cluster
        
        Args:
            cluster_id: Cluster ID to preview
            max_images: Maximum number of images to return
            
        Returns:
            List of image information for the cluster
        """
        if self.labels is None or not self.face_data:
            raise ValueError("No clustering results available")
        
        cluster_faces = []
        for face_data, label in zip(self.face_data, self.labels):
            if label == cluster_id:
                cluster_faces.append(face_data)
                if len(cluster_faces) >= max_images:
                    break
        
        return cluster_faces
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of current pipeline state"""
        return {
            'face_data_loaded': len(self.face_data) if self.face_data else 0,
            'clustering_completed': self.labels is not None,
            'clustering_stats': self.clustering_stats,
            'config': self.config
        }
    
    def export_embeddings(self, output_path: str):
        """
        Export face embeddings and metadata
        
        Args:
            output_path: Path to save embeddings
        """
        if not self.face_data or self.embeddings is None:
            raise ValueError("No face data or embeddings available")
        
        export_data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'face_metadata': [
                {
                    'image_path': face['image_path'],
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'face_area': face['face_area']
                }
                for face in self.face_data
            ]
        }
        
        np.savez_compressed(output_path, **export_data)
        self.logger.info(f"Embeddings exported to: {output_path}")
    
    def load_embeddings(self, embeddings_path: str):
        """
        Load previously saved embeddings and metadata
        
        Args:
            embeddings_path: Path to embeddings file
        """
        try:
            data = np.load(embeddings_path, allow_pickle=True)
            
            self.embeddings = data['embeddings']
            self.labels = data.get('labels')
            face_metadata = data['face_metadata']
            
            # Reconstruct face_data
            self.face_data = []
            for i, metadata in enumerate(face_metadata):
                face_info = {
                    'image_path': str(metadata['image_path']),
                    'embedding': self.embeddings[i],
                    'bbox': metadata['bbox'],
                    'confidence': metadata['confidence'],
                    'face_area': metadata['face_area']
                }
                self.face_data.append(face_info)
            
            self.logger.info(f"Loaded {len(self.face_data)} faces from {embeddings_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            raise