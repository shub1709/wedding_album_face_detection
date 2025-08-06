"""
Face detection module using InsightFace
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import insightface
from insightface.app import FaceAnalysis

from ..utils.logger import get_logger


class FaceDetector:
    """Face detection class using InsightFace"""
    
    def __init__(self, model_name: str = "buffalo_l", 
                 providers: List[str] = None,
                 det_size: Tuple[int, int] = (640, 640),
                 ctx_id: int = 0):
        """
        Initialize face detector
        
        Args:
            model_name: InsightFace model name
            providers: Execution providers for ONNX
            det_size: Detection input size
            ctx_id: Context ID for GPU
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.det_size = det_size
        self.ctx_id = ctx_id
        
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the InsightFace model"""
        try:
            self.logger.info(f"Loading InsightFace model: {self.model_name}")
            self.app = FaceAnalysis(
                name=self.model_name, 
                providers=self.providers
            )
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self.logger.info("Face detection model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize face detection model: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, 
                    min_face_size: int = 30) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image as numpy array
            min_face_size: Minimum face size to consider
            
        Returns:
            List of face detection results
        """
        if self.app is None:
            raise ValueError("Face detection model not initialized")
        
        try:
            faces = self.app.get(image)
            valid_faces = []
            
            for face in faces:
                # Extract bounding box
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                width, height = x2 - x1, y2 - y1
                
                # Filter by minimum face size
                if width < min_face_size or height < min_face_size:
                    continue
                
                face_data = {
                    'bbox': box,
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'embedding': face.embedding,
                    'confidence': getattr(face, 'det_score', 1.0),
                    'landmarks': getattr(face, 'landmark_2d_106', None)
                }
                
                valid_faces.append(face_data)
            
            return valid_faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def detect_faces_from_path(self, image_path: str,
                              min_face_size: int = 30) -> List[Dict]:
        """
        Detect faces from image file path
        
        Args:
            image_path: Path to image file
            min_face_size: Minimum face size to consider
            
        Returns:
            List of face detection results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return []
            
            return self.detect_faces(image, min_face_size)
            
        except Exception as e:
            self.logger.error(f"Failed to detect faces in {image_path}: {e}")
            return []
    
    def extract_face_region(self, image: np.ndarray, 
                           bbox: np.ndarray, 
                           padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face region from image
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around face region
            
        Returns:
            Extracted face region or None
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Add padding
            width, height = x2 - x1, y2 - y1
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            # Calculate padded coordinates
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(image.shape[1], x2 + pad_w)
            y2_pad = min(image.shape[0], y2 + pad_h)
            
            # Extract face region
            face_region = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            return face_region if face_region.size > 0 else None
            
        except Exception as e:
            self.logger.error(f"Failed to extract face region: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'providers': self.providers,
            'detection_size': self.det_size,
            'context_id': self.ctx_id
        }