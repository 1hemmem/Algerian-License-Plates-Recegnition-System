import cv2
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from ultralytics import YOLO

    
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageQualityMetrics:
    """Store image quality metrics."""
    sharpness: float
    edge_count: int
    contrast: float
    noise: float
    total_score: float

class ImageProcessor:
    """Handle image processing operations."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the image processor.

        Args:
            target_size: Desired dimensions for processed images (width, height)
        """
        self.target_size = target_size
        self._denoising_params = {
            "h": 10,
            "hColor": 10,
            "templateWindowSize": 7,
            "searchWindowSize": 21,
        }
        self._sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image by resizing, denoising and sharpening.

        Args:
            image: Input image array

        Returns:
            Preprocessed image array
        """
        if image is None:
            raise ValueError("Input image is None")

        # Resize if needed
        current_height, current_width = image.shape[:2]
        if (current_height, current_width) != self.target_size:
            image = cv2.resize(image, self.target_size)

        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, **self._denoising_params)

        # Sharpen
        return cv2.filter2D(image, -1, self._sharpening_kernel)

    def calculate_quality_metrics(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Calculate various image quality metrics.

        Args:
            image: Input image array

        Returns:
            ImageQualityMetrics object containing all metrics
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.count_nonzero(edges)
            contrast = gray.max() - gray.min()
            noise = np.var(gray)

            # Calculate weighted score
            total_score = (
                0.4 * sharpness + 0.3 * edge_count + 0.2 * contrast - 0.1 * noise
            )

            return ImageQualityMetrics(
                sharpness=sharpness,
                edge_count=edge_count,
                contrast=contrast,
                noise=noise,
                total_score=total_score,
            )
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {str(e)}")
            raise


class LicensePlateDetector:
    """Detect and extract license plates from images."""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the detector with a YOLO model.
        
        Args:
            model_path: Path to the YOLO model weights
        """
        try:
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
            
    def detect_plate(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect and extract license plate from image.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (detection_success, plate_image)
        """
        try:
            results = self.model(image,conf=0.8)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box:
                        xyxy = box.xyxy.cpu().numpy()[0]
                        x1, y1, x2, y2 = map(int, xyxy)
                        coord_plate = [x1, y1, x2, y2]
                        plate_image = image[y1:y2, x1:x2]
                        return True, plate_image, coord_plate
                        
            return False, None, None
            
        except Exception as e:
            logger.error(f"Error during plate detection: {str(e)}")
            raise
        
