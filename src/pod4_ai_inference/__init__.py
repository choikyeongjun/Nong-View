"""
POD 4: AI Inference Module
Handles YOLO-based object detection and segmentation
"""

from .engine import InferenceEngine
from .model_manager import ModelManager
from .schemas import (
    Detection,
    InferenceResult,
    ModelConfig,
    InferenceBatch
)

__all__ = [
    "InferenceEngine",
    "ModelManager",
    "Detection",
    "InferenceResult",
    "ModelConfig",
    "InferenceBatch"
]