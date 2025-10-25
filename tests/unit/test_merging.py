"""
Unit tests for Merge Engine (POD5)
"""

import pytest
from uuid import uuid4

from src.pod5_merging import MergeEngine, MergeConfig
from src.pod5_merging.schemas import MergedDetection, MergedResult


class TestMergeEngine:
    """Test Merge Engine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create test merge engine"""
        config = MergeConfig(iou_threshold=0.5)
        return MergeEngine(config)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.config.iou_threshold == 0.5
        assert engine.config.merge_strategy == "weighted_avg"
    
    def test_calculate_iou(self, engine):
        """Test IOU calculation"""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        
        iou = engine._calculate_iou(bbox1, bbox2)
        assert 0 < iou < 1
        
        # Non-overlapping
        bbox3 = (20, 20, 30, 30)
        iou = engine._calculate_iou(bbox1, bbox3)
        assert iou == 0
    
    def test_calculate_bounds(self, engine):
        """Test bounds calculation"""
        detections = [
            MergedDetection(
                class_name="crop",
                class_id=0,
                confidence=0.8,
                bbox=(0, 0, 10, 10),
                area=100,
                source_tiles=[uuid4()]
            ),
            MergedDetection(
                class_name="crop",
                class_id=0,
                confidence=0.7,
                bbox=(5, 5, 15, 15),
                area=100,
                source_tiles=[uuid4()]
            )
        ]
        
        bounds = engine._calculate_bounds(detections)
        assert bounds == (0, 0, 15, 15)


class TestMergeConfig:
    """Test Merge Configuration"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = MergeConfig(
            iou_threshold=0.5,
            merge_strategy="weighted_avg"
        )
        assert config.iou_threshold == 0.5
    
    def test_invalid_iou_threshold(self):
        """Test invalid IOU threshold"""
        with pytest.raises(ValueError):
            MergeConfig(iou_threshold=1.5)
    
    def test_invalid_merge_strategy(self):
        """Test invalid merge strategy"""
        with pytest.raises(ValueError):
            MergeConfig(merge_strategy="invalid")


class TestMergedResult:
    """Test Merged Result"""
    
    def test_result_creation(self):
        """Test creating merged result"""
        result = MergedResult(
            image_id=uuid4(),
            roi_bounds=(0, 0, 100, 100),
            detections=[],
            total_tiles_processed=10,
            merge_time=1.5
        )
        
        assert result.total_detections == 0
        assert result.total_tiles_processed == 10
    
    def test_class_summary(self):
        """Test getting class summary"""
        detections = [
            MergedDetection(
                class_name="crop",
                class_id=0,
                confidence=0.8,
                bbox=(0, 0, 10, 10),
                area=100,
                source_tiles=[uuid4()]
            ),
            MergedDetection(
                class_name="crop",
                class_id=0,
                confidence=0.7,
                bbox=(20, 20, 30, 30),
                area=100,
                source_tiles=[uuid4()]
            ),
            MergedDetection(
                class_name="facility",
                class_id=1,
                confidence=0.9,
                bbox=(40, 40, 50, 50),
                area=100,
                source_tiles=[uuid4()]
            )
        ]
        
        result = MergedResult(
            image_id=uuid4(),
            roi_bounds=(0, 0, 100, 100),
            detections=detections,
            total_tiles_processed=10,
            merge_time=1.5
        )
        
        summary = result.get_class_summary()
        
        assert "crop" in summary
        assert summary["crop"]["count"] == 2
        assert summary["crop"]["total_area"] == 200
        assert summary["crop"]["avg_confidence"] == 0.75
        
        assert "facility" in summary
        assert summary["facility"]["count"] == 1