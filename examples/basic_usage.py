#!/usr/bin/env python3
"""
Basic usage example for Face Clustering System
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from src.face_clustering.pipeline import FaceClusteringPipeline


def basic_example():
    """Basic face clustering example"""
    print("=" * 50)
    print("Face Clustering System - Basic Example")
    print("=" * 50)
    
    # Load default configuration
    settings = Settings()
    
    # Override some settings for this example
    settings.set('paths.input_folder', 'data/input')
    settings.set('paths.output_folder', 'data/output/basic_example')
    settings.set('face_detection.min_face_size', 40)
    settings.set('clustering.eps', 0.4)
    settings.set('debug.enabled', True)
    
    # Create directories
    settings.create_directories()
    
    print(f"Input folder: {settings.input_folder}")
    print(f"Output folder: {settings.output_folder}")
    print(f"Minimum face size: {settings.min_face_size}")
    print()
    
    try:
        # Initialize pipeline
        print("Initializing face clustering pipeline...")
        pipeline = FaceClusteringPipeline(settings._config)
        
        # Run the complete pipeline
        print("Running face clustering pipeline...")
        results = pipeline.run_full_pipeline(
            input_folder=settings.input_folder,
            optimize_params=False,  # Skip optimization for basic example
            copy_files=True
        )
        
        # Display results
        if results['pipeline_status'] == 'completed':
            print("\n✅ Pipeline completed successfully!")
            print("-" * 30)
            
            face_stats = results['face_detection']
            cluster_stats = results['clustering']
            org_stats = results['organization']
            
            print(f"📸 Images processed: {face_stats['unique_images']}")
            print(f"👥 Faces detected: {face_stats['total_faces']}")
            print(f"🎯 Clusters found: {cluster_stats['n_clusters']}")
            print(f"❓ Unclustered faces: {cluster_stats['n_noise']}")
            
            if cluster_stats['n_noise'] > 0:
                noise_ratio = cluster_stats['n_noise'] / face_stats['total_faces']
                print(f"📊 Noise ratio: {noise_ratio:.1%}")
            
            print(f"📁 Files organized: {org_stats['files_processed']}")
            
            if org_stats.get('errors'):
                print(f"⚠️  Errors: {len(org_stats['errors'])}")
            
            print(f"\n📂 Results saved to: {settings.output_folder}")
            
            # Show cluster breakdown
            if cluster_stats['n_clusters'] > 0:
                print("\n👥 Cluster breakdown:")
                cluster_counts = org_stats.get('cluster_sizes', {})
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"   Person {cluster_id}: {count} faces")
        
        else:
            print(f"❌ Pipeline failed: {results.get('error')}")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure to place some images in the 'data/input' folder")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def step_by_step_example():
    """Step-by-step clustering example"""
    print("\n" + "=" * 50)
    print("Face Clustering System - Step by Step Example")
    print("=" * 50)
    
    settings = Settings()
    settings.set('paths.input_folder', 'data/input')
    settings.set('paths.output_folder', 'data/output/step_by_step')
    settings.create_directories()
    
    try:
        # Initialize pipeline
        pipeline = FaceClusteringPipeline(settings._config)
        
        # Step 1: Face Detection
        print("Step 1: Detecting faces...")
        face_data = pipeline.detect_faces(settings.input_folder)
        print(f"   ✅ Found {len(face_data)} faces in images")
        
        # Step 2: Clustering
        print("Step 2: Clustering faces...")
        labels = pipeline.cluster_faces(optimize_params=True)
        print(f"   ✅ Created {pipeline.clustering_stats['n_clusters']} clusters")
        
        # Step 3: Get cluster preview
        print("Step 3: Previewing clusters...")
        for cluster_id in range(pipeline.clustering_stats['n_clusters']):
            preview = pipeline.get_cluster_preview(cluster_id, max_images=3)
            print(f"   👥 Person {cluster_id}: {len(preview)} faces")
            for i, face in enumerate(preview[:2]):  # Show first 2 images
                img_name = Path(face['image_path']).name
                print(f"      - {img_name}")
        
        # Step 4: Organization
        print("Step 4: Organizing results...")
        org_stats = pipeline.organize_results(copy_files=True)
        print(f"   ✅ Organized {org_stats['files_processed']} files")
        
        # Step 5: Export embeddings for future use
        embeddings_path = Path(settings.output_folder) / "embeddings.npz"
        print("Step 5: Exporting embeddings...")
        pipeline.export_embeddings(str(embeddings_path))
        print(f"   ✅ Embeddings saved to {embeddings_path}")
        
        print(f"\n🎉 All steps completed! Check results in {settings.output_folder}")
        
    except Exception as e:
        print(f"❌ Error in step-by-step example: {e}")


def clustering_comparison_example():
    """Compare different clustering algorithms"""
    print("\n" + "=" * 50)
    print("Clustering Algorithm Comparison")
    print("=" * 50)
    
    settings = Settings()
    settings.set('paths.input_folder', 'data/input')
    settings.set('debug.enabled', True)
    
    algorithms = ['DBSCAN', 'KMeans', 'Agglomerative']
    results_comparison = {}
    
    try:
        for algorithm in algorithms:
            print(f"\n🧪 Testing {algorithm}...")
            
            # Set output folder for this algorithm
            output_folder = f'data/output/comparison_{algorithm.lower()}'
            settings.set('paths.output_folder', output_folder)
            settings.set('clustering.algorithm', algorithm)
            
            if algorithm == 'KMeans' or algorithm == 'Agglomerative':
                settings.set('clustering.n_clusters', 5)  # Assume 5 people
            
            settings.create_directories()
            
            # Run pipeline
            pipeline = FaceClusteringPipeline(settings._config)
            results = pipeline.run_full_pipeline(
                settings.input_folder,
                optimize_params=False,  # Skip optimization for comparison
                copy_files=True
            )
            
            if results['pipeline_status'] == 'completed':
                cluster_stats = results['clustering']
                results_comparison[algorithm] = {
                    'n_clusters': cluster_stats['n_clusters'],
                    'n_noise': cluster_stats.get('n_noise', 0),
                    'silhouette_score': cluster_stats.get('silhouette_score', 'N/A')
                }
                print(f"   ✅ {algorithm}: {cluster_stats['n_clusters']} clusters")
            else:
                print(f"   ❌ {algorithm} failed: {results.get('error')}")
                results_comparison[algorithm] = {'error': results.get('error')}
        
        # Display comparison
        print("\n📊 Algorithm Comparison Results:")
        print("-" * 40)
        for algo, stats in results_comparison.items():
            if 'error' in stats:
                print(f"{algo:15}: Failed - {stats['error']}")
            else:
                silhouette = stats['silhouette_score']
                if isinstance(silhouette, float):
                    silhouette = f"{silhouette:.3f}"
                print(f"{algo:15}: {stats['n_clusters']} clusters, "
                      f"{stats['n_noise']} noise, "
                      f"quality: {silhouette}")
        
    except Exception as e:
        print(f"❌ Error in comparison example: {e}")


if __name__ == "__main__":
    # Run examples
    basic_example()
    step_by_step_example()
    clustering_comparison_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the 'data/output' folder for results.")
    print("=" * 50)