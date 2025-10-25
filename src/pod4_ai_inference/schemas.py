"""
Schemas for AI inference module
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4
from enum import Enum


class ModelType(str, Enum):
    """Model types"""
    CROP = "crop"
    FACILITY = "facility"
    LANDUSE = "landuse"


class DetectionType(str, Enum):
    """Detection types"""
    BBOX = "bbox"  # Bounding box
    SEGMENT = "segment"  # Segmentation mask
    CLASSIFICATION = "classification"  # Image classification


class CropClass(str, Enum):
    """Crop classification classes"""
    IRG = "IRG"  # Italian Ryegrass
    BARLEY = "barley"
    WHEAT = "wheat"
    CORN_SILAGE = "corn_silage"
    HAY = "hay"
    UNKNOWN = "unknown"


class FacilityClass(str, Enum):
    """Facility detection classes"""
    GREENHOUSE_SINGLE = "greenhouse_single"  # 단동 비닐하우스
    GREENHOUSE_MULTI = "greenhouse_multi"  # 연동 비닐하우스
    STORAGE = "storage"  # 저장시설
    LIVESTOCK = "livestock"  # 축사
    SILO = "silo"  # 사일로
    UNKNOWN = "unknown"


class LandUseClass(str, Enum):
    """Land use classification classes"""
    CULTIVATED = "cultivated"  # 경작지
    FALLOW = "fallow"  # 휴경지
    ABANDONED = "abandoned"  # 폐경지
    CONSTRUCTION = "construction"  # 공사중
    WATER = "water"  # 수역
    FOREST = "forest"  # 산림
    UNKNOWN = "unknown"


class Detection(BaseModel):
    """Single detection result"""
    detection_id: UUID = Field(default_factory=uuid4)
    detection_type: DetectionType
    class_name: str
    class_id: int
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    segmentation: Optional[List[List[float]]] = None  # Polygon points
    area: Optional[float] = None  # Area in square meters
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {v}")
        return v
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate bounding box"""
        if v is not None:
            x1, y1, x2, y2 = v
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid bounding box: {v}")
        return v


class InferenceResult(BaseModel):
    """Results from inference on a single tile"""
    result_id: UUID = Field(default_factory=uuid4)
    tile_id: UUID
    model_type: ModelType
    model_version: str
    detections: List[Detection]
    inference_time: float  # seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def detection_count(self) -> int:
        """Get number of detections"""
        return len(self.detections)
    
    def filter_by_confidence(self, threshold: float) -> List[Detection]:
        """Filter detections by confidence threshold"""
        return [d for d in self.detections if d.confidence >= threshold]
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get counts by class"""
        counts = {}
        for detection in self.detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts


class InferenceBatch(BaseModel):
    """Batch of tiles for inference"""
    batch_id: UUID = Field(default_factory=uuid4)
    tile_ids: List[UUID]
    tile_paths: List[str]
    model_type: ModelType
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def batch_size(self) -> int:
        """Get batch size"""
        return len(self.tile_ids)


class ModelConfig(BaseModel):
    """Model configuration"""
    model_id: UUID = Field(default_factory=uuid4)
    model_type: ModelType
    model_path: str
    model_version: str
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    classes: List[str]
    class_mapping: Dict[int, str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('input_size')
    def validate_input_size(cls, v):
        """Validate input size"""
        width, height = v
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid input size: {v}")
        return v


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_id: UUID
    total_inferences: int = 0
    total_detections: int = 0
    avg_inference_time: float = 0.0
    avg_confidence: float = 0.0
    class_distribution: Dict[str, int] = Field(default_factory=dict)
    error_count: int = 0
    last_inference: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    def update_metrics(self, result: InferenceResult):
        """Update metrics with new result"""
        self.total_inferences += 1
        self.total_detections += len(result.detections)
        
        # Update average inference time
        if self.total_inferences == 1:
            self.avg_inference_time = result.inference_time
        else:
            self.avg_inference_time = (
                (self.avg_inference_time * (self.total_inferences - 1) + 
                 result.inference_time) / self.total_inferences
            )
        
        # Update average confidence
        if result.detections:
            avg_conf = sum(d.confidence for d in result.detections) / len(result.detections)
            if self.total_detections == len(result.detections):
                self.avg_confidence = avg_conf
            else:
                self.avg_confidence = (
                    (self.avg_confidence * (self.total_detections - len(result.detections)) +
                     avg_conf * len(result.detections)) / self.total_detections
                )
        
        # Update class distribution
        for detection in result.detections:
            self.class_distribution[detection.class_name] = \
                self.class_distribution.get(detection.class_name, 0) + 1
        
        self.last_inference = datetime.now()


class InferenceJob(BaseModel):
    """Inference job tracking"""
    job_id: UUID = Field(default_factory=uuid4)
    image_id: UUID
    model_types: List[ModelType]
    tile_count: int
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    results: List[InferenceResult] = Field(default_factory=list)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('status')
    def validate_status(cls, v):
        """Validate job status"""
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v
    
    @validator('progress')
    def validate_progress(cls, v):
        """Validate progress"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Progress must be between 0 and 1: {v}")
        return v
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete"""
        return self.status in ['completed', 'failed']
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None