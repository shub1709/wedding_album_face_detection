"""
File organization and management for clustered faces
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Set, Any
from collections import defaultdict, Counter

from ..utils.logger import get_logger


class FileOrganizer:
    """Organize files based on clustering results"""
    
    def __init__(self, output_folder: str):
        """
        Initialize file organizer
        
        Args:
            output_folder: Base output directory
        """
        self.logger = get_logger(__name__)
        self.output_folder = Path(output_folder)
        self.unclustered_folder = self.output_folder / "unclustered"
        
        # Create directories
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.unclustered_folder.mkdir(parents=True, exist_ok=True)
    
    def organize_clustered_images(self, face_data: List[Dict[str, Any]], 
                                labels: List[int],
                                copy_files: bool = True) -> Dict[str, Any]:
        """
        Organize images based on clustering results
        
        Args:
            face_data: List of face data dictionaries
            labels: Cluster labels for each face
            copy_files: Whether to copy files or create symlinks
            
        Returns:
            Organization statistics
        """
        if len(face_data) != len(labels):
            raise ValueError("Mismatch between face_data and labels length")
        
        self.logger.info(f"Organizing {len(face_data)} faces into clusters")
        
        stats = {
            'total_faces': len(face_data),
            'clusters_created': 0,
            'unclustered_faces': 0,
            'files_processed': 0,
            'errors': [],
            'cluster_sizes': Counter()
        }
        
        # Group faces by cluster
        cluster_groups = defaultdict(list)
        for face_info, label in zip(face_data, labels):
            cluster_groups[label].append(face_info)
        
        # Process each cluster
        for cluster_id, faces in cluster_groups.items():
            try:
                if cluster_id == -1:
                    # Handle unclustered faces
                    self._organize_unclustered_faces(faces, copy_files)
                    stats['unclustered_faces'] += len(faces)
                else:
                    # Handle clustered faces
                    cluster_folder = self.output_folder / f"person_{cluster_id}"
                    self._organize_cluster(faces, cluster_folder, copy_files)
                    stats['clusters_created'] += 1
                    stats['cluster_sizes'][cluster_id] = len(faces)
                
                stats['files_processed'] += len(faces)
                
            except Exception as e:
                error_msg = f"Error organizing cluster {cluster_id}: {e}"
                self.logger.error(error_msg)
                stats['errors'].append(error_msg)
        
        self.logger.info(f"Organization complete: {stats['clusters_created']} clusters, "
                        f"{stats['unclustered_faces']} unclustered faces")
        
        return stats
    
    def _organize_unclustered_faces(self, faces: List[Dict], copy_files: bool):
        """Organize unclustered faces"""
        for idx, face_info in enumerate(faces):
            try:
                source_path = Path(face_info['image_path'])
                
                # Create unique filename for unclustered face
                filename = f"{source_path.stem}_face{idx}{source_path.suffix}"
                dest_path = self.unclustered_folder / filename
                
                # Handle filename conflicts
                counter = 1
                while dest_path.exists():
                    filename = f"{source_path.stem}_face{idx}_{counter}{source_path.suffix}"
                    dest_path = self.unclustered_folder / filename
                    counter += 1
                
                self._copy_or_link_file(source_path, dest_path, copy_files)
                
            except Exception as e:
                self.logger.error(f"Error organizing unclustered face: {e}")
    
    def _organize_cluster(self, faces: List[Dict], cluster_folder: Path, copy_files: bool):
        """Organize faces within a cluster"""
        cluster_folder.mkdir(parents=True, exist_ok=True)
        
        # Keep track of processed images to avoid duplicates
        processed_images: Set[str] = set()
        
        for face_info in faces:
            try:
                source_path = Path(face_info['image_path'])
                
                # Skip if we've already processed this image for this cluster
                if str(source_path) in processed_images:
                    continue
                
                processed_images.add(str(source_path))
                
                # Create destination path
                dest_path = cluster_folder / source_path.name
                
                # Handle filename conflicts
                counter = 1
                while dest_path.exists():
                    stem = source_path.stem
                    suffix = source_path.suffix
                    dest_path = cluster_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                self._copy_or_link_file(source_path, dest_path, copy_files)
                
            except Exception as e:
                self.logger.error(f"Error organizing clustered face: {e}")
    
    def _copy_or_link_file(self, source: Path, dest: Path, copy_files: bool):
        """Copy or create symlink for file"""
        try:
            if copy_files:
                shutil.copy2(source, dest)
            else:
                # Create symlink (Unix-like systems)
                if hasattr(os, 'symlink'):
                    dest.symlink_to(source.absolute())
                else:
                    # Fallback to copy on Windows
                    shutil.copy2(source, dest)
                    
        except Exception as e:
            self.logger.error(f"Failed to copy/link {source} to {dest}: {e}")
            raise
    
    def create_summary_report(self, stats: Dict[str, Any], 
                            face_data: List[Dict], 
                            labels: List[int]) -> str:
        """
        Create a summary report of the organization results
        
        Args:
            stats: Organization statistics
            face_data: Original face data
            labels: Cluster labels
            
        Returns:
            Path to summary file
        """
        summary_path = self.output_folder / "clustering_summary.txt"
        
        try:
            unique_images = set(face['image_path'] for face in face_data)
            
            with open(summary_path, 'w') as f:
                f.write("Face Clustering Summary Report\n")
                f.write("=" * 40 + "\n\n")
                
                # General statistics
                f.write(f"Total unique images processed: {len(unique_images)}\n")
                f.write(f"Total faces detected: {stats['total_faces']}\n")
                f.write(f"Clusters created: {stats['clusters_created']}\n")
                f.write(f"Unclustered faces: {stats['unclustered_faces']}\n")
                f.write(f"Files processed successfully: {stats['files_processed']}\n")
                
                if stats['errors']:
                    f.write(f"Errors encountered: {len(stats['errors'])}\n")
                
                f.write("\n" + "-" * 30 + "\n")
                f.write("Cluster Details:\n")
                f.write("-" * 30 + "\n")
                
                # Cluster details
                for cluster_id, count in sorted(stats['cluster_sizes'].items()):
                    f.write(f"Person {cluster_id}: {count} faces\n")
                
                # Error details
                if stats['errors']:
                    f.write("\n" + "-" * 30 + "\n")
                    f.write("Errors:\n")
                    f.write("-" * 30 + "\n")
                    for error in stats['errors']:
                        f.write(f"- {error}\n")
                
                f.write(f"\nReport generated at: {summary_path}\n")
            
            self.logger.info(f"Summary report saved to: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create summary report: {e}")
            raise
    
    def cleanup_empty_folders(self):
        """Remove empty cluster folders"""
        try:
            for item in self.output_folder.iterdir():
                if item.is_dir() and item.name.startswith('person_'):
                    if not any(item.iterdir()):
                        item.rmdir()
                        self.logger.info(f"Removed empty folder: {item}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_organization_stats(self) -> Dict[str, Any]:
        """Get current organization statistics"""
        stats = {
            'output_folder': str(self.output_folder),
            'cluster_folders': [],
            'unclustered_count': 0,
            'total_organized_files': 0
        }
        
        try:
            # Count files in unclustered folder
            if self.unclustered_folder.exists():
                stats['unclustered_count'] = len(list(self.unclustered_folder.glob('*')))
            
            # Count cluster folders and files
            for item in self.output_folder.iterdir():
                if item.is_dir() and item.name.startswith('person_'):
                    file_count = len(list(item.glob('*')))
                    stats['cluster_folders'].append({
                        'name': item.name,
                        'file_count': file_count
                    })
                    stats['total_organized_files'] += file_count
            
            stats['total_organized_files'] += stats['unclustered_count']
            
        except Exception as e:
            self.logger.error(f"Error getting organization stats: {e}")
        
        return stats