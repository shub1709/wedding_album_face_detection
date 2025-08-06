#!/usr/bin/env python3
"""
Main entry point for Face Clustering System
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config.settings import Settings
from src.face_clustering.pipeline import FaceClusteringPipeline
from src.face_clustering.utils.logger import setup_logging


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Face Clustering System')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input folder containing images')
    parser.add_argument('--output', '-o', 
                       help='Output folder for results')
    parser.add_argument('--config', '-c', 
                       help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize clustering parameters')
    parser.add_argument('--no-copy', action='store_true',
                       help='Create symlinks instead of copying files')
    parser.add_argument('--export-embeddings', 
                       help='Export embeddings to specified path')
    parser.add_argument('--load-embeddings',
                       help='Load embeddings from specified path')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = args.config if args.config else None
        settings = Settings(config_path)
        
        # Override config with command line arguments
        if args.output:
            settings.set('paths.output_folder', args.output)
        
        # Setup logging
        setup_logging(settings._config)
        
        # Create necessary directories
        settings.create_directories()
        
        print("=" * 60)
        print("Face Clustering System")
        print("=" * 60)
        print(f"Input folder: {args.input}")
        print(f"Output folder: {settings.output_folder}")
        print(f"Min face size: {settings.min_face_size}")
        print(f"Clustering eps: {settings.eps}")
        print("=" * 60)
        
        # Initialize pipeline
        pipeline = FaceClusteringPipeline(settings._config)
        
        # Load embeddings if specified
        if args.load_embeddings:
            print(f"Loading embeddings from: {args.load_embeddings}")
            pipeline.load_embeddings(args.load_embeddings)
            
            # Only perform clustering and organization
            labels = pipeline.cluster_faces(args.optimize)
            results = pipeline.organize_results(not args.no_copy)
            
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(
                input_folder=args.input,
                optimize_params=args.optimize,
                copy_files=not args.no_copy
            )
        
        # Export embeddings if specified
        if args.export_embeddings:
            pipeline.export_embeddings(args.export_embeddings)
        
        # Print results
        print("\nPipeline Results:")
        print("-" * 30)
        
        if results.get('pipeline_status') == 'completed':
            face_stats = results.get('face_detection', {})
            cluster_stats = results.get('clustering', {})
            org_stats = results.get('organization', {})
            
            print(f"✓ Total images processed: {face_stats.get('unique_images', 0)}")
            print(f"✓ Total faces detected: {face_stats.get('total_faces', 0)}")
            print(f"✓ Clusters created: {cluster_stats.get('n_clusters', 0)}")
            print(f"✓ Unclustered faces: {cluster_stats.get('n_noise', 0)}")
            
            if 'silhouette_score' in cluster_stats:
                print(f"✓ Clustering quality: {cluster_stats['silhouette_score']:.3f}")
            
            print(f"✓ Files organized: {org_stats.get('files_processed', 0)}")
            
            if org_stats.get('errors'):
                print(f"⚠ Errors encountered: {len(org_stats['errors'])}")
            
            print(f"\n🎉 Results saved to: {settings.output_folder}")
            
        else:
            print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return 1
        
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())