"""
Model Manager - Model version control and management system
"""

import logging
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
import yaml
import torch

from .schemas import (
    ModelConfig,
    ModelType,
    ModelMetrics,
    InferenceResult
)
from ..common.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model versions, configurations, and performance tracking
    Supports A/B testing and model rollback
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir or "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.weights_dir = self.models_dir / "weights"
        self.configs_dir = self.models_dir / "configs"
        self.metrics_dir = self.models_dir / "metrics"
        self.archive_dir = self.models_dir / "archive"
        
        for dir_path in [self.weights_dir, self.configs_dir, 
                         self.metrics_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[ModelType, Dict[str, ModelConfig]] = {
            ModelType.CROP: {},
            ModelType.FACILITY: {},
            ModelType.LANDUSE: {}
        }
        
        # Active models (current production models)
        self.active_models: Dict[ModelType, str] = {}
        
        # Model metrics
        self.metrics: Dict[Tuple[ModelType, str], ModelMetrics] = {}
        
        # Load existing models
        self._load_model_registry()
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_file = self.models_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load model configs
                    for model_type_str, versions in data.get('models', {}).items():
                        model_type = ModelType(model_type_str)
                        for version, config_data in versions.items():
                            config = ModelConfig(**config_data)
                            self.models[model_type][version] = config
                    
                    # Load active models
                    self.active_models = {
                        ModelType(k): v 
                        for k, v in data.get('active', {}).items()
                    }
                    
                logger.info(f"Loaded model registry with {sum(len(v) for v in self.models.values())} models")
                
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_file = self.models_dir / "registry.json"
        
        try:
            data = {
                'models': {
                    model_type.value: {
                        version: config.dict()
                        for version, config in versions.items()
                    }
                    for model_type, versions in self.models.items()
                },
                'active': {
                    model_type.value: version
                    for model_type, version in self.active_models.items()
                },
                'updated_at': datetime.now().isoformat()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug("Model registry saved")
            
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_model(
        self,
        model_type: ModelType,
        model_path: str,
        version: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> ModelConfig:
        """
        Register a new model
        
        Args:
            model_type: Type of model
            model_path: Path to model weights
            version: Model version (auto-generated if None)
            config: Additional configuration
            set_active: Set as active model
            
        Returns:
            ModelConfig object
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"v_{timestamp}"
        
        # Check if version already exists
        if version in self.models[model_type]:
            raise ValueError(f"Model version {version} already exists for {model_type}")
        
        # Copy model to weights directory
        dest_filename = f"{model_type.value}_{version}.pt"
        dest_path = self.weights_dir / dest_filename
        shutil.copy2(source_path, dest_path)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(dest_path)
        
        # Load model to get information
        try:
            model = torch.load(dest_path, map_location='cpu')
            model_info = {
                'architecture': model.get('model', {}).get('arch', 'unknown'),
                'parameters': model.get('model', {}).get('parameters', 0),
                'training_epochs': model.get('epoch', 0)
            }
        except:
            model_info = {}
        
        # Create model config
        model_config = ModelConfig(
            model_type=model_type,
            model_path=str(dest_path),
            model_version=version,
            confidence_threshold=config.get('confidence_threshold', 0.5) if config else 0.5,
            nms_threshold=config.get('nms_threshold', 0.5) if config else 0.5,
            classes=config.get('classes', []) if config else [],
            class_mapping=config.get('class_mapping', {}) if config else {},
            metadata={
                'hash': model_hash,
                'original_path': str(source_path),
                'model_info': model_info,
                'registered_at': datetime.now().isoformat()
            }
        )
        
        # Save config to file
        config_file = self.configs_dir / f"{model_type.value}_{version}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(model_config.dict(), f)
        
        # Add to registry
        self.models[model_type][version] = model_config
        
        # Set as active if requested
        if set_active:
            self.set_active_model(model_type, version)
        
        # Save registry
        self._save_model_registry()
        
        # Initialize metrics
        self.metrics[(model_type, version)] = ModelMetrics(
            model_id=model_config.model_id
        )
        
        logger.info(f"Registered model {model_type} version {version}")
        
        return model_config
    
    def set_active_model(self, model_type: ModelType, version: str):
        """
        Set active model for a model type
        
        Args:
            model_type: Type of model
            version: Model version to activate
        """
        if version not in self.models[model_type]:
            raise ValueError(f"Model version {version} not found for {model_type}")
        
        # Archive current active model if exists
        if model_type in self.active_models:
            old_version = self.active_models[model_type]
            logger.info(f"Archiving previous active model {model_type} {old_version}")
        
        # Set new active model
        self.active_models[model_type] = version
        
        # Update registry
        self._save_model_registry()
        
        logger.info(f"Set active model for {model_type}: {version}")
    
    def get_active_model(self, model_type: ModelType) -> Optional[ModelConfig]:
        """
        Get active model configuration
        
        Args:
            model_type: Type of model
            
        Returns:
            ModelConfig if active model exists
        """
        if model_type not in self.active_models:
            return None
        
        version = self.active_models[model_type]
        return self.models[model_type].get(version)
    
    def get_model(
        self,
        model_type: ModelType,
        version: Optional[str] = None
    ) -> Optional[ModelConfig]:
        """
        Get model configuration
        
        Args:
            model_type: Type of model
            version: Model version (uses active if None)
            
        Returns:
            ModelConfig if model exists
        """
        if version is None:
            return self.get_active_model(model_type)
        
        return self.models[model_type].get(version)
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None
    ) -> Dict[ModelType, List[str]]:
        """
        List available models
        
        Args:
            model_type: Filter by model type
            
        Returns:
            Dictionary of model versions by type
        """
        if model_type:
            return {model_type: list(self.models[model_type].keys())}
        
        return {
            mt: list(versions.keys())
            for mt, versions in self.models.items()
        }
    
    def archive_model(self, model_type: ModelType, version: str):
        """
        Archive a model version
        
        Args:
            model_type: Type of model
            version: Model version to archive
        """
        if version not in self.models[model_type]:
            raise ValueError(f"Model version {version} not found for {model_type}")
        
        config = self.models[model_type][version]
        
        # Move model file to archive
        source_path = Path(config.model_path)
        if source_path.exists():
            archive_path = self.archive_dir / source_path.name
            shutil.move(str(source_path), str(archive_path))
            
            # Update config
            config.model_path = str(archive_path)
            config.metadata['archived_at'] = datetime.now().isoformat()
        
        # Remove from active if it was active
        if self.active_models.get(model_type) == version:
            del self.active_models[model_type]
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Archived model {model_type} version {version}")
    
    def delete_model(self, model_type: ModelType, version: str):
        """
        Delete a model version
        
        Args:
            model_type: Type of model
            version: Model version to delete
        """
        if version not in self.models[model_type]:
            raise ValueError(f"Model version {version} not found for {model_type}")
        
        # Don't delete active model
        if self.active_models.get(model_type) == version:
            raise ValueError(f"Cannot delete active model {model_type} {version}")
        
        config = self.models[model_type][version]
        
        # Delete model file
        model_path = Path(config.model_path)
        if model_path.exists():
            model_path.unlink()
        
        # Delete config file
        config_file = self.configs_dir / f"{model_type.value}_{version}.yaml"
        if config_file.exists():
            config_file.unlink()
        
        # Remove from registry
        del self.models[model_type][version]
        
        # Remove metrics
        if (model_type, version) in self.metrics:
            del self.metrics[(model_type, version)]
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Deleted model {model_type} version {version}")
    
    def update_metrics(
        self,
        model_type: ModelType,
        version: str,
        result: InferenceResult
    ):
        """
        Update model metrics with inference result
        
        Args:
            model_type: Type of model
            version: Model version
            result: Inference result
        """
        key = (model_type, version)
        
        if key not in self.metrics:
            config = self.models[model_type].get(version)
            if not config:
                return
            
            self.metrics[key] = ModelMetrics(model_id=config.model_id)
        
        self.metrics[key].update_metrics(result)
        
        # Save metrics periodically
        if self.metrics[key].total_inferences % 100 == 0:
            self._save_metrics(model_type, version)
    
    def _save_metrics(self, model_type: ModelType, version: str):
        """Save metrics to disk"""
        key = (model_type, version)
        if key not in self.metrics:
            return
        
        metrics_file = self.metrics_dir / f"{model_type.value}_{version}_metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics[key].dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_metrics(
        self,
        model_type: ModelType,
        version: Optional[str] = None
    ) -> Optional[ModelMetrics]:
        """
        Get model metrics
        
        Args:
            model_type: Type of model
            version: Model version (uses active if None)
            
        Returns:
            ModelMetrics if available
        """
        if version is None:
            version = self.active_models.get(model_type)
            if not version:
                return None
        
        return self.metrics.get((model_type, version))
    
    def compare_models(
        self,
        model_type: ModelType,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            model_type: Type of model
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        metrics1 = self.get_metrics(model_type, version1)
        metrics2 = self.get_metrics(model_type, version2)
        
        if not metrics1 or not metrics2:
            return {}
        
        return {
            'version1': version1,
            'version2': version2,
            'inference_time_diff': metrics2.avg_inference_time - metrics1.avg_inference_time,
            'confidence_diff': metrics2.avg_confidence - metrics1.avg_confidence,
            'detection_rate_diff': (
                metrics2.total_detections / max(metrics2.total_inferences, 1) -
                metrics1.total_detections / max(metrics1.total_inferences, 1)
            ),
            'error_rate_diff': (
                metrics2.error_count / max(metrics2.total_inferences, 1) -
                metrics1.error_count / max(metrics1.total_inferences, 1)
            )
        }
    
    def rollback_model(self, model_type: ModelType):
        """
        Rollback to previous model version
        
        Args:
            model_type: Type of model
        """
        # Get version history
        versions = list(self.models[model_type].keys())
        
        if len(versions) < 2:
            raise ValueError(f"No previous version available for {model_type}")
        
        current = self.active_models.get(model_type)
        if not current:
            # No active model, activate the latest
            self.set_active_model(model_type, versions[-1])
            return
        
        # Find previous version
        try:
            current_idx = versions.index(current)
            if current_idx > 0:
                previous = versions[current_idx - 1]
                self.set_active_model(model_type, previous)
                logger.info(f"Rolled back {model_type} from {current} to {previous}")
            else:
                raise ValueError(f"No previous version for {model_type}")
        except ValueError as e:
            raise ValueError(f"Error during rollback: {e}")
    
    def export_model(
        self,
        model_type: ModelType,
        version: str,
        export_path: str
    ):
        """
        Export model with configuration
        
        Args:
            model_type: Type of model
            version: Model version
            export_path: Export directory path
        """
        if version not in self.models[model_type]:
            raise ValueError(f"Model version {version} not found for {model_type}")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        config = self.models[model_type][version]
        
        # Copy model file
        model_path = Path(config.model_path)
        if model_path.exists():
            dest_path = export_dir / model_path.name
            shutil.copy2(model_path, dest_path)
        
        # Export config
        config_file = export_dir / f"{model_type.value}_{version}_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config.dict(), f)
        
        # Export metrics if available
        metrics = self.get_metrics(model_type, version)
        if metrics:
            metrics_file = export_dir / f"{model_type.value}_{version}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics.dict(), f, indent=2, default=str)
        
        logger.info(f"Exported model {model_type} {version} to {export_path}")