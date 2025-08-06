"""
Unit tests for FaceDetector class
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from face_clustering.core.face_detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector"""
    
    @pytest.fixture
    def mock_insightface(self):
        """Mock InsightFace app"""
        with patch('face_clustering.core.face_detector.FaceAnalysis') as mock_fa:
            mock_app = Mock()
            mock_fa.return_value = mock_app
            yield mock_app
    
    @pytest.fixture
    def face_detector(self, mock_insightface):
        """Create FaceDetector instance with mocked InsightFace"""
        return FaceDetector(
            model_name="buffalo_l",
            providers=['CPUExecutionProvider'],
            det_size=(640, 640)
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def mock_face_result(self):
        """Mock face detection result"""
        mock_face = Mock()
        mock_face.bbox = np.array([100, 100, 200, 200])  # x1, y1, x2, y2
        mock_face.embedding = np.random.random(512).astype(np.float32)
        mock_face.det_score = 0.95
        mock_face.landmark_2d_106 = np.random.random((106, 2))
        return mock_face
    
    def test_initialization(self, mock_insightface):
        """Test FaceDetector initialization"""
        detector = FaceDetector()
        
        assert detector.model_name == "buffalo_l"
        assert detector.providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']
        assert detector.det_size == (640, 640)
        assert detector.ctx_id == 0
        mock_insightface.prepare.assert_called_once()
    
    def test_initialization_with_custom_params(self, mock_insightface):
        """Test FaceDetector initialization with custom parameters"""
        custom_providers = ['CPUExecutionProvider']
        detector = FaceDetector(
            model_name="buffalo_s",
            providers=custom_providers,
            det_size=(320, 320),
            ctx_id=1
        )
        
        assert detector.model_name == "buffalo_s"
        assert detector.providers == custom_providers
        assert detector.det_size == (320, 320)
        assert detector.ctx_id == 1
    
    def test_detect_faces_success(self, face_detector, sample_image, mock_face_result):
        """Test successful face detection"""
        # Setup mock
        face_detector.app.get.return_value = [mock_face_result]
        
        # Test detection
        results = face_detector.detect_faces(sample_image, min_face_size=30)
        
        # Assertions
        assert len(results) == 1
        face_data = results[0]
        
        assert 'bbox' in face_data
        assert 'width' in face_data
        assert 'height' in face_data
        assert 'area' in face_data
        assert 'embedding' in face_data
        assert 'confidence' in face_data
        
        assert face_data['width'] == 100  # 200 - 100
        assert face_data['height'] == 100  # 200 - 100
        assert face_data['area'] == 10000  # 100 * 100
        assert face_data['confidence'] == 0.95
    
    def test_detect_faces_filter_small_faces(self, face_detector, sample_image):
        """Test filtering of small faces"""
        # Create small face mock
        small_face = Mock()
        small_face.bbox = np.array([100, 100, 120, 120])  # 20x20 face
        small_face.embedding = np.random.random(512).astype(np.float32)
        small_face.det_score = 0.95
        
        face_detector.app.get.return_value = [small_face]
        
        # Test with min_face_size=30 (should filter out 20x20 face)
        results = face_detector.detect_faces(sample_image, min_face_size=30)
        
        assert len(results) == 0
    
    def test_detect_faces_no_faces_found(self, face_detector, sample_image):
        """Test when no faces are detected"""
        face_detector.app.get.return_value = []
        
        results = face_detector.detect_faces(sample_image)
        
        assert len(results) == 0
    
    def test_detect_faces_exception_handling(self, face_detector, sample_image):
        """Test exception handling during face detection"""
        face_detector.app.get.side_effect = Exception("Detection failed")
        
        results = face_detector.detect_faces(sample_image)
        
        assert len(results) == 0
    
    @patch('face_clustering.core.face_detector.cv2.imread')
    def test_detect_faces_from_path_success(self, mock_imread, face_detector, 
                                          sample_image, mock_face_result):
        """Test face detection from image file path"""
        mock_imread.return_value = sample_image
        face_detector.app.get.return_value = [mock_face_result]
        
        results = face_detector.detect_faces_from_path("test_image.jpg")
        
        assert len(results) == 1
        mock_imread.assert_called_once_with("test_image.jpg")
    
    @patch('face_clustering.core.face_detector.cv2.imread')
    def test_detect_faces_from_path_invalid_image(self, mock_imread, face_detector):
        """Test face detection from invalid image path"""
        mock_imread.return_value = None  # Simulate failed image loading
        
        results = face_detector.detect_faces_from_path("invalid_image.jpg")
        
        assert len(results) == 0
    
    def test_extract_face_region_success(self, face_detector, sample_image):
        """Test successful face region extraction"""
        bbox = np.array([100, 100, 200, 200])
        
        face_region = face_detector.extract_face_region(sample_image, bbox, padding=0.1)
        
        assert face_region is not None
        assert face_region.shape[0] > 0  # Height
        assert face_region.shape[1] > 0  # Width
    
    def test_extract_face_region_with_padding(self, face_detector, sample_image):
        """Test face region extraction with padding"""
        bbox = np.array([100, 100, 200, 200])
        
        # Extract without padding
        face_no_pad = face_detector.extract_face_region(sample_image, bbox, padding=0.0)
        # Extract with padding
        face_with_pad = face_detector.extract_face_region(sample_image, bbox, padding=0.2)
        
        # Padded region should be larger
        assert face_with_pad.shape[0] >= face_no_pad.shape[0]
        assert face_with_pad.shape[1] >= face_no_pad.shape[1]
    
    def test_extract_face_region_boundary_cases(self, face_detector):
        """Test face region extraction at image boundaries"""
        small_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Face at edge of image
        bbox = np.array([90, 90, 100, 100])
        
        face_region = face_detector.extract_face_region(small_image, bbox, padding=0.5)
        
        # Should not crash and return valid region
        assert face_region is not None
        assert face_region.size > 0
    
    def test_extract_face_region_invalid_bbox(self, face_detector, sample_image):
        """Test face region extraction with invalid bbox"""
        invalid_bbox = np.array([300, 300, 250, 250])  # x2 < x1, y2 < y1
        
        face_region = face_detector.extract_face_region(sample_image, invalid_bbox)
        
        # Should handle gracefully
        assert face_region is None or face_region.size == 0
    
    def test_get_model_info(self, face_detector):
        """Test getting model information"""
        info = face_detector.get_model_info()
        
        assert 'model_name' in info
        assert 'providers' in info
        assert 'detection_size' in info
        assert 'context_id' in info
        
        assert info['model_name'] == "buffalo_l"
        assert info['providers'] == ['CPUExecutionProvider']
        assert info['detection_size'] == (640, 640)
        assert info['context_id'] == 0
    
    def test_app_not_initialized_error(self, mock_insightface):
        """Test error when app is not initialized"""
        detector = FaceDetector()
        detector.app = None
        
        sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Face detection model not initialized"):
            detector.detect_faces(sample_image)


if __name__ == "__main__":
    pytest.main([__file__])