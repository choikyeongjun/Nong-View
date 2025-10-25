"""
Inference Engine - Core AI inference functionality
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import numpy as np
import torch
import cv2
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from .schemas import (
    Detection,
    InferenceResult,
    ModelType,
    DetectionType,
    CropClass,
    FacilityClass,
    LandUseClass,
    ModelConfig
)
from ..common.config import settings

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    AI inference engine for YOLOv11-based detection and segmentation
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            device: Device to run inference ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Inference engine initialized with device: {self.device}")
        
        # Model storage
        self.models: Dict[ModelType, YOLO] = {}
        self.model_configs: Dict[ModelType, ModelConfig] = {}
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        # Initialize class mappings
        self._init_class_mappings()
    
    def _init_class_mappings(self):
        """Initialize class name mappings for each model type"""
        self.class_mappings = {
            ModelType.CROP: {
                0: CropClass.IRG,
                1: CropClass.BARLEY,
                2: CropClass.WHEAT,
                3: CropClass.CORN_SILAGE,
                4: CropClass.HAY,
                5: CropClass.UNKNOWN
            },
            ModelType.FACILITY: {
                0: FacilityClass.GREENHOUSE_SINGLE,
                1: FacilityClass.GREENHOUSE_MULTI,
                2: FacilityClass.STORAGE,
                3: FacilityClass.LIVESTOCK,
                4: FacilityClass.SILO,
                5: FacilityClass.UNKNOWN
            },
            ModelType.LANDUSE: {
                0: LandUseClass.CULTIVATED,
                1: LandUseClass.FALLOW,
                2: LandUseClass.ABANDONED,
                3: LandUseClass.CONSTRUCTION,
                4: LandUseClass.WATER,
                5: LandUseClass.FOREST,
                6: LandUseClass.UNKNOWN
            }
        }
    
    def load_model(
        self,
        model_type: ModelType,
        model_path: str,
        config: Optional[ModelConfig] = None
    ):
        """
        Load a YOLO model
        
        Args:
            model_type: Type of model
            model_path: Path to model weights
            config: Optional model configuration
        """
        try:
            # Load YOLO model
            model = YOLO(model_path)
            model.to(self.device)
            
            # Store model
            self.models[model_type] = model
            
            # Create or update config
            if config:
                self.model_configs[model_type] = config
            else:
                # Create default config
                self.model_configs[model_type] = ModelConfig(
                    model_type=model_type,
                    model_path=model_path,
                    model_version="1.0.0",
                    confidence_threshold=settings.confidence_threshold,
                    nms_threshold=settings.nms_threshold,
                    classes=list(self.class_mappings[model_type].values()),
                    class_mapping={
                        k: v.value for k, v in self.class_mappings[model_type].items()
                    }
                )
            
            logger.info(f"Loaded {model_type} model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            raise
    
    def _preprocess_image(
        self,
        image_path: str,
        target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image
            target_size: Target size for model input
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if needed (YOLO handles this internally, but we can do it for consistency)
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def _postprocess_detections(
        self,
        results,
        model_type: ModelType,
        tile_id: UUID
    ) -> List[Detection]:
        """
        Postprocess YOLO results to Detection objects
        
        Args:
            results: YOLO prediction results
            model_type: Model type
            tile_id: Tile ID
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for r in results:
            if r.boxes is not None:
                # Process bounding boxes
                boxes = r.boxes
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # Get class and confidence
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Get class name
                    class_mapping = self.class_mappings[model_type]
                    if class_id in class_mapping:
                        class_name = class_mapping[class_id].value
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Calculate area (in pixels for now)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = Detection(
                        detection_type=DetectionType.BBOX,
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        area=area
                    )
                    detections.append(detection)
            
            if r.masks is not None:
                # Process segmentation masks
                masks = r.masks
                for i in range(len(masks)):
                    # Get mask polygon
                    mask = masks.xy[i]  # Polygon points
                    
                    # Get class and confidence
                    class_id = int(r.boxes.cls[i]) if r.boxes is not None else 0
                    confidence = float(r.boxes.conf[i]) if r.boxes is not None else 1.0
                    
                    # Get class name
                    class_mapping = self.class_mappings[model_type]
                    if class_id in class_mapping:
                        class_name = class_mapping[class_id].value
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Convert mask to list
                    segmentation = mask.tolist() if isinstance(mask, np.ndarray) else mask
                    
                    # Calculate area from polygon
                    if len(segmentation) > 0:
                        # Simple area calculation (shoelace formula)
                        area = self._calculate_polygon_area(segmentation)
                    else:
                        area = 0
                    
                    detection = Detection(
                        detection_type=DetectionType.SEGMENT,
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        segmentation=[segmentation],
                        area=area
                    )
                    detections.append(detection)
        
        return detections
    
    def _calculate_polygon_area(self, points: List[List[float]]) -> float:
        """
        Calculate area of polygon using shoelace formula
        
        Args:
            points: List of [x, y] points
            
        Returns:
            Area of polygon
        """
        if len(points) < 3:
            return 0
        
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2
    
    async def infer_single(
        self,
        tile_path: str,
        tile_id: UUID,
        model_type: ModelType
    ) -> InferenceResult:
        """
        Run inference on a single tile
        
        Args:
            tile_path: Path to tile image
            tile_id: Tile ID
            model_type: Model type to use
            
        Returns:
            InferenceResult object
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        start_time = time.time()
        
        try:
            # Get model and config
            model = self.models[model_type]
            config = self.model_configs[model_type]
            
            # Run inference
            results = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                model.predict,
                tile_path,
                config.confidence_threshold,
                config.nms_threshold
            )
            
            # Postprocess detections
            detections = self._postprocess_detections(results, model_type, tile_id)
            
            # Create result
            inference_time = time.time() - start_time
            
            result = InferenceResult(
                tile_id=tile_id,
                model_type=model_type,
                model_version=config.model_version,
                detections=detections,
                inference_time=inference_time,
                metadata={
                    'tile_path': tile_path,
                    'device': self.device
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    async def infer_batch(
        self,
        tile_paths: List[str],
        tile_ids: List[UUID],
        model_type: ModelType,
        batch_size: Optional[int] = None
    ) -> List[InferenceResult]:
        """
        Run inference on a batch of tiles
        
        Args:
            tile_paths: List of tile paths
            tile_ids: List of tile IDs
            model_type: Model type to use
            batch_size: Batch size for processing
            
        Returns:
            List of InferenceResult objects
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded")
        
        if batch_size is None:
            batch_size = settings.batch_size
        
        results = []
        total_tiles = len(tile_paths)
        
        # Process in batches
        for i in range(0, total_tiles, batch_size):
            batch_paths = tile_paths[i:i+batch_size]
            batch_ids = tile_ids[i:i+batch_size]
            
            # Run inference on batch
            batch_tasks = [
                self.infer_single(path, tile_id, model_type)
                for path, tile_id in zip(batch_paths, batch_ids)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}")
        
        return results
    
    def infer_multi_model(
        self,
        tile_path: str,
        tile_id: UUID,
        model_types: List[ModelType]
    ) -> Dict[ModelType, InferenceResult]:
        """
        Run multiple models on the same tile
        
        Args:
            tile_path: Path to tile image
            tile_id: Tile ID
            model_types: List of model types to run
            
        Returns:
            Dictionary of results by model type
        """
        async def run_multi():
            tasks = []
            for model_type in model_types:
                task = self.infer_single(tile_path, tile_id, model_type)
                tasks.append((model_type, task))
            
            results = {}
            for model_type, task in tasks:
                result = await task
                results[model_type] = result
            
            return results
        
        return asyncio.run(run_multi())
    
    def apply_nms(
        self,
        detections: List[Detection],
        threshold: float = 0.5
    ) -> List[Detection]:
        """
        Apply Non-Maximum Suppression to detections
        
        Args:
            detections: List of detections
            threshold: IOU threshold for NMS
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections
        
        # Group by class
        class_groups = {}
        for det in detections:
            if det.class_name not in class_groups:
                class_groups[det.class_name] = []
            class_groups[det.class_name].append(det)
        
        # Apply NMS per class
        filtered = []
        for class_name, class_dets in class_groups.items():
            if not class_dets:
                continue
            
            # Extract boxes and scores
            boxes = []
            scores = []
            for det in class_dets:
                if det.bbox:
                    boxes.append(det.bbox)
                    scores.append(det.confidence)
            
            if not boxes:
                filtered.extend(class_dets)
                continue
            
            # Apply NMS
            boxes_array = np.array(boxes)
            scores_array = np.array(scores)
            
            # Sort by score
            indices = np.argsort(scores_array)[::-1]
            
            keep = []
            while len(indices) > 0:
                current = indices[0]
                keep.append(current)
                
                if len(indices) == 1:
                    break
                
                # Calculate IOU with remaining boxes
                current_box = boxes_array[current]
                other_boxes = boxes_array[indices[1:]]
                
                ious = self._calculate_iou(current_box, other_boxes)
                
                # Keep boxes with IOU less than threshold
                indices = indices[1:][ious < threshold]
            
            # Keep selected detections
            for idx in keep:
                filtered.append(class_dets[idx])
        
        return filtered
    
    def _calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        boxes: np.ndarray
    ) -> np.ndarray:
        """
        Calculate IOU between one box and multiple boxes
        
        Args:
            box1: Reference box (x1, y1, x2, y2)
            boxes: Array of boxes
            
        Returns:
            Array of IOU values
        """
        # Calculate intersection
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box1_area + boxes_area - intersection
        
        # Calculate IOU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def get_model_info(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            model_type: Model type
            
        Returns:
            Model information dictionary
        """
        if model_type not in self.models:
            return {}
        
        model = self.models[model_type]
        config = self.model_configs[model_type]
        
        return {
            'model_type': model_type.value,
            'model_path': config.model_path,
            'model_version': config.model_version,
            'device': self.device,
            'input_size': config.input_size,
            'confidence_threshold': config.confidence_threshold,
            'nms_threshold': config.nms_threshold,
            'classes': config.classes
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
        
        # Clear GPU memory if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()