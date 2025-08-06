# Face Clustering System

A comprehensive, modular face clustering system using deep learning and computer vision techniques. This system automatically detects faces in images, generates embeddings, and groups similar faces together using advanced clustering algorithms.

## 🌟 Features

- **Advanced Face Detection**: Uses InsightFace (ArcFace-R100) for robust face detection and embedding generation
- **Multiple Clustering Algorithms**: Support for DBSCAN, K-means, and Agglomerative clustering
- **Parameter Optimization**: Automatic parameter tuning for optimal clustering results
- **Comprehensive Visualization**: Detailed plots and dashboards for analysis
- **Modular Architecture**: Clean, extensible codebase with separate components
- **Flexible Configuration**: YAML-based configuration management
- **Batch Processing**: Efficient processing of large image collections
- **Export/Import**: Save and load embeddings for reuse
- **Detailed Logging**: Comprehensive logging system with rotation

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

### Option 1: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/face-clustering-system.git
cd face-clustering-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Using pip

```bash
pip install face-clustering-system
```

## 🎯 Quick Start

### Basic Usage

```bash
# Run face clustering on your image folder
python main.py --input /path/to/images --output /path/to/results

# With parameter optimization
python main.py --input /path/to/images --output /path/to/results --optimize

# Create symlinks instead of copying files
python main.py --input /path/to/images --output /path/to/results --no-copy
```

### Python API

```python
from config.settings import Settings
from src.face_clustering.pipeline import FaceClusteringPipeline

# Load configuration
settings = Settings('config/config.yaml')

# Initialize pipeline
pipeline = FaceClusteringPipeline(settings._config)

# Run complete pipeline
results = pipeline.run_full_pipeline(
    input_folder='path/to/images',
    optimize_params=True,
    copy_files=True
)

print(f"Found {results['clustering']['n_clusters']} unique persons")
```

## ⚙️ Configuration

The system uses YAML configuration files. Edit `config/config.yaml` to customize:

```yaml
# Paths Configuration
paths:
  input_folder: "/path/to/input/folder"
  output_folder: "/path/to/output/folder"

# Face Detection Configuration
face_detection:
  model_name: "buffalo_l"
  min_face_size: 30
  providers: 
    - "CUDAExecutionProvider"  # For GPU
    - "CPUExecutionProvider"   # Fallback

# Clustering Configuration
clustering:
  algorithm: "DBSCAN"
  eps: 0.45
  min_samples: 2
  metric: "cosine"

# Debug and Visualization
debug:
  enabled: true
  save_plots: true
```

## 📁 Project Structure

```
face_clustering_system/
│
├── README.md
├── requirements.txt
├── setup.py
├── main.py                    # Main entry point
├── config/
│   ├── config.yaml           # Configuration file
│   └── settings.py           # Settings management
├── src/face_clustering/
│   ├── core/                 # Core components
│   │   ├── face_detector.py  # Face detection
│   │   ├── clustering_engine.py  # Clustering algorithms
│   │   └── file_organizer.py # File organization
│   ├── utils/               # Utilities
│   │   ├── logger.py        # Logging utilities
│   │   └── visualization.py # Plotting and visualization
│   └── pipeline.py          # Main pipeline
├── data/                    # Data directories
│   ├── input/
│   └── output/
├── logs/                    # Log files
└── tests/                   # Unit tests
```

## 🔧 Advanced Usage

### Custom Configuration

```python
# Create custom configuration
custom_config = {
    'face_detection': {
        'min_face_size': 50,
        'model_name': 'buffalo_l'
    },
    'clustering': {
        'algorithm': 'DBSCAN',
        'eps': 0.4,
        'min_samples': 3
    }
}

pipeline = FaceClusteringPipeline(custom_config)
```

### Export/Import Embeddings

```bash
# Export embeddings for later use
python main.py --input /path/to/images --export-embeddings embeddings.npz

# Load and cluster existing embeddings
python main.py --load-embeddings embeddings.npz --output /path/to/results
```

### Batch Processing with Different Parameters

```python
# Process multiple folders with different settings
folders = ['folder1', 'folder2', 'folder3']
eps_values = [0.3, 0.4, 0.5]

for folder, eps in zip(folders, eps_values):
    settings.set('clustering.eps', eps)
    pipeline = FaceClusteringPipeline(settings._config)
    results = pipeline.run_full_pipeline(folder)
```

## 📊 Output Structure

The system organizes results as follows:

```
output/
├── person_0/          # Cluster 0 images
│   ├── image1.jpg
│   └── image2.jpg
├── person_1/          # Cluster 1 images
│   ├── image3.jpg
│   └── image4.jpg
├── unclustered/       # Faces that couldn't be clustered
│   ├── image5_face0.jpg
│   └── image6_face0.jpg
├── clustering_summary.txt      # Text summary
├── clustering_dashboard.png    # Visual dashboard
├── cluster_distribution.png    # Cluster size distribution
├── embedding_space_pca.png     # PCA visualization
├── embedding_space_tsne.png    # t-SNE visualization
└── eps_optimization.png        # Parameter optimization plot
```

## 🎨 Visualization Features

- **Clustering Dashboard**: Comprehensive overview of results
- **Parameter Optimization Plots**: Visual guide for parameter tuning
- **Embedding Space Visualization**: PCA and t-SNE plots of face embeddings
- **Cluster Distribution**: Statistics and size distribution of clusters

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_detector.py
```

## 🚀 Performance Tips

1. **GPU Acceleration**: Install `onnxruntime-gpu` for faster inference
2. **Batch Size**: Adjust batch size in config for memory optimization
3. **Image Preprocessing**: Resize large images to reduce processing time
4. **Parameter Tuning**: Use optimization features to find best parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the excellent face recognition models
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms
- [OpenCV](https://opencv.org/) for image processing utilities

## 📞 Support

- Create an issue on GitHub for bug reports
- Check the [Wiki](https://github.com/yourusername/face-clustering-system/wiki) for documentation
- Email: your.email@example.com

## 🔄 Changelog

### v1.0.0
- Initial release with full clustering pipeline
- Support for multiple clustering algorithms
- Comprehensive visualization system
- Configuration management
- Export/import functionality

---

**Made with ❤️ for the computer vision community**