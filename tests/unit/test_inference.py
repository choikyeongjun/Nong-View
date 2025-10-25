"""
Unit tests for AI Inference Engine (POD4)
"""

import pytest
from uuid import uuid4
from datetime import datetime

from src.pod4_ai_inference import ModelManager, InferenceEngine
from src.pod4_ai_inference.schemas import (
    ModelType,
    ModelConfig,
    Detection,
    DetectionType,
    InferenceResult,
    ModelMetrics
)


class TestModelManager:
    """Test Model Manager functionality"""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create test model manager"""
        return ModelManager(models_dir=str(tmp_path))
    
    def test_manager_initialization(self, manager, tmp_path):
        """Test manager initialization"""
        assert manager.models_dir == tmp_path
        assert (tmp_path / "weights").exists()
        assert (tmp_path / "configs").exists()
        assert (tmp_path / "metrics").exists()
    
    def test_list_models_empty(self, manager):
        """Test listing models when empty"""
        models = manager.list_models()
        assert ModelType.CROP in models
        assert ModelType.FACILITY in models
        assert ModelType.LANDUSE in models
        assert all(len(v) == 0 for v in models.values())
    
    def test_get_active_model_none(self, manager):
        """Test getting active model when none set"""
        model = manager.get_active_model(ModelType.CROP)
        assert model is None


class TestInferenceEngine:
    """Test Inference Engine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create test inference engine"""
        return InferenceEngine(device='cpu')
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.device == 'cpu'
        assert len(engine.models) == 0
        assert len(engine.model_configs) == 0
    
    def test_class_mappings(self, engine):
        """Test class mappings initialization"""
        assert ModelType.CROP in engine.class_mappings
        assert ModelType.FACILITY in engine.class_mappings
        assert ModelType.LANDUSE in engine.class_mappings
    
    def test_calculate_iou(self, engine):
        """Test IOU calculation"""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        box3 = (20, 20, 30, 30)
        
        # Overlapping boxes
        iou = engine._calculate_iou(box1, np.array([box2]))
        assert 0 < iou[0] < 1
        
        # Non-overlapping boxes
        iou = engine._calculate_iou(box1, np.array([box3]))
        assert iou[0] == 0


class TestDetectionSchema:
    """Test Detection schemas"""
    
    def test_detection_creation(self):
        """Test creating detection"""
        detection = Detection(
            detection_type=DetectionType.BBOX,
            class_name="crop",
            class_id=0,
            confidence=0.8,
            bbox=(10, 10, 100, 100)
        )
        
        assert detection.detection_type == DetectionType.BBOX
        assert detection.confidence == 0.8
    
    def test_invalid_confidence(self):
        """Test invalid confidence value"""
        with pytest.raises(ValueError):
            Detection(
                detection_type=DetectionType.BBOX,
                class_name="crop",
                class_id=0,
                confidence=1.5,  # Invalid
                bbox=(10, 10, 100, 100)
            )
    
    def test_invalid_bbox(self):
        """Test invalid bounding box"""
        with pytest.raises(ValueError):
            Detection(
                detection_type=DetectionType.BBOX,
                class_name="crop",
                class_id=0,
                confidence=0.8,
                bbox=(100, 100, 10, 10)  # Invalid (x2 < x1)
            )


class TestInferenceResult:
    """Test Inference Result"""
    
    def test_result_creation(self):
        """Test creating inference result"""
        tile_id = uuid4()
        detections = [
            Detection(
                detection_type=DetectionType.BBOX,
                class_name="crop",
                class_id=0,
                confidence=0.8,
                bbox=(10, 10, 100, 100)
            )
        ]
        
        result = InferenceResult(
            tile_id=tile_id,
            model_type=ModelType.CROP,
            model_version="1.0.0",
            detections=detections,
            inference_time=0.5
        )
        
        assert result.detection_count == 1
        assert result.tile_id == tile_id
    
    def test_filter_by_confidence(self):
        """Test filtering detections by confidence"""
        detections = [
            Detection(
                detection_type=DetectionType.BBOX,
                class_name="crop",
                class_id=0,
                confidence=0.3,
                bbox=(10, 10, 100, 100)
            ),
            Detection(
                detection_type=DetectionType.BBOX,
                class_name="crop",
                class_id=0,
                confidence=0.8,
                bbox=(10, 10, 100, 100)
            )
        ]
        
        result = InferenceResult(
            tile_id=uuid4(),
            model_type=ModelType.CROP,
            model_version="1.0.0",
            detections=detections,
            inference_time=0.5
        )
        
        filtered = result.filter_by_confidence(0.5)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.8


class TestModelMetrics:
    """Test Model Metrics"""
    
    def test_metrics_update(self):
        """Test updating metrics"""
        metrics = ModelMetrics(model_id=uuid4())
        
        result = InferenceResult(
            tile_id=uuid4(),
            model_type=ModelType.CROP,
            model_version="1.0.0",
            detections=[
                Detection(
                    detection_type=DetectionType.BBOX,
                    class_name="crop",
                    class_id=0,
                    confidence=0.8,
                    bbox=(10, 10, 100, 100)
                )
            ],
            inference_time=0.5
        )
        
        metrics.update_metrics(result)
        
        assert metrics.total_inferences == 1
        assert metrics.total_detections == 1
        assert metrics.avg_inference_time == 0.5
        assert metrics.avg_confidence == 0.8