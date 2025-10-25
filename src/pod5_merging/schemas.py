"""
Schemas for merging module
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class MergeConfig(BaseModel):
    """Configuration for merging operation"""
    iou_threshold: float = Field(default=0.5, description="IOU threshold for duplicate removal")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence for detections")
    merge_strategy: str = Field(default="weighted_avg", description="Strategy for merging overlapping detections")
    boundary_extension: int = Field(default=10, description="Pixels to extend for boundary matching")
    min_area: float = Field(default=1.0, description="Minimum area in square meters")
    
    @validator('iou_threshold')
    def validate_iou(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"IOU threshold must be between 0 and 1: {v}")
        return v
    
    @validator('merge_strategy')
    def validate_strategy(cls, v):
        valid = ['weighted_avg', 'max_confidence', 'union', 'intersection']
        if v not in valid:
            raise ValueError(f"Invalid merge strategy: {v}")
        return v


class MergedDetection(BaseModel):
    """Merged detection across tiles"""
    detection_id: UUID = Field(default_factory=uuid4)
    class_name: str
    class_id: int
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # Global coordinates
    segmentation: Optional[List[List[float]]] = None
    area: float  # In square meters
    source_tiles: List[UUID]  # Tiles that contributed to this detection
    merge_count: int = 1  # Number of detections merged
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ParcelStatistics(BaseModel):
    """Statistics for a single parcel"""
    pnu: str
    total_area: float  # Square meters
    detected_area: float
    coverage_ratio: float
    class_areas: Dict[str, float]  # Area by class
    class_counts: Dict[str, int]  # Count by class
    dominant_class: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MergedResult(BaseModel):
    """Result of merging operation"""
    result_id: UUID = Field(default_factory=uuid4)
    image_id: UUID
    roi_bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    detections: List[MergedDetection]
    parcel_stats: Optional[List[ParcelStatistics]] = None
    total_tiles_processed: int
    merge_time: float  # seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def total_detections(self) -> int:
        return len(self.detections)
    
    def get_class_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics by class"""
        summary = {}
        for detection in self.detections:
            if detection.class_name not in summary:
                summary[detection.class_name] = {
                    'count': 0,
                    'total_area': 0.0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            
            summary[detection.class_name]['count'] += 1
            summary[detection.class_name]['total_area'] += detection.area
            summary[detection.class_name]['confidences'].append(detection.confidence)
        
        # Calculate averages
        for class_name, stats in summary.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            del stats['confidences']
        
        return summary