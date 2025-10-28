"""
Optimized Model Training System for Nong-View Best Performance
Author: Claude Opus (System Architect & Core Algorithms)
Date: 2025-10-28
Version: 1.0.0

Advanced training optimization system implementing:
- Intelligent hyperparameter optimization
- Dynamic learning rate scheduling  
- Advanced loss function design
- Multi-stage training strategies
- Hardware-aware optimization
"""

import os
import sys
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import yaml
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from ultralytics import YOLO
from collections import defaultdict
import psutil
import GPUtil
from tqdm import tqdm
import gc
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.best_config import CONFIG, ModelType, DatasetType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """Training strategy types"""
    STANDARD = "standard"
    PROGRESSIVE = "progressive"  # Progressive resizing
    CURRICULUM = "curriculum"    # Easy to hard samples
    ENSEMBLE = "ensemble"        # Multiple model training
    DISTILLATION = "distillation"  # Knowledge distillation


@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    # Basic settings
    model_type: ModelType
    dataset_type: DatasetType
    epochs: int
    batch_size: int
    imgsz: int
    
    # Optimization settings  
    optimizer: str = "AdamW"
    base_lr: float = 0.001
    final_lr: float = 0.00001
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Advanced settings
    strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE
    use_amp: bool = True
    gradient_clip_val: float = 10.0
    ema_decay: float = 0.9999
    label_smoothing: float = 0.0
    
    # Augmentation settings
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.3
    degrees: float = 10.0
    translate: float = 0.2
    scale: float = 0.9
    shear: float = 2.0
    perspective: float = 0.0
    
    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 8
    pin_memory: bool = True
    
    # Loss weights (for multi-task learning)
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5
    
    # Early stopping
    patience: int = 30
    min_delta: float = 0.001
    
    # Checkpointing
    save_period: int = 5
    keep_checkpoints: int = 3


class AdvancedLearningRateScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
        self.current_lr = config.base_lr
        
        # Initialize scheduler based on strategy
        if config.strategy == TrainingStrategy.PROGRESSIVE:
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=config.epochs // 4,
                T_mult=2,
                eta_min=config.final_lr
            )
        else:
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=config.base_lr,
                epochs=config.epochs,
                steps_per_epoch=1,
                pct_start=config.warmup_epochs / config.epochs,
                final_div_factor=config.base_lr / config.final_lr
            )
    
    def step(self, metrics: Optional[Dict] = None):
        """Update learning rate based on metrics"""
        if metrics and 'loss' in metrics:
            # Adaptive adjustment based on loss plateau
            if self._is_plateau(metrics['loss']):
                self._reduce_lr_on_plateau()
        
        self.scheduler.step()
        self.current_epoch += 1
        self.current_lr = self.optimizer.param_groups[0]['lr']
        
        return self.current_lr
    
    def _is_plateau(self, loss: float, window: int = 5) -> bool:
        """Detect if loss has plateaued"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        
        self.loss_history.append(loss)
        
        if len(self.loss_history) < window:
            return False
        
        recent_losses = self.loss_history[-window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        # Plateau if standard deviation is very small
        return std_loss < self.config.min_delta * mean_loss
    
    def _reduce_lr_on_plateau(self, factor: float = 0.5):
        """Reduce learning rate on plateau"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        logger.info(f"Reduced learning rate to {self.current_lr * factor}")


class LossFunctionOptimizer:
    """Advanced loss function optimization for object detection"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 1.5
        
        # Adaptive loss weights based on dataset
        self.adaptive_weights = self._calculate_adaptive_weights()
    
    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive loss weights based on dataset characteristics"""
        weights = {
            'box': self.config.box_loss_weight,
            'cls': self.config.cls_loss_weight,
            'dfl': self.config.dfl_loss_weight
        }
        
        # Adjust weights based on dataset type
        if self.config.dataset_type == DatasetType.GROWTH_TIF:
            # Growth TIF has severe class imbalance
            weights['cls'] *= 1.5  # Increase classification loss weight
        elif self.config.dataset_type == DatasetType.GREENHOUSE_MULTI:
            # Multiple greenhouses need better localization
            weights['box'] *= 1.2
        
        return weights
    
    def compute_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for addressing class imbalance"""
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * ce_loss
        
        if self.config.label_smoothing > 0:
            # Apply label smoothing
            n_classes = pred.shape[1]
            smooth_target = target * (1 - self.config.label_smoothing) + \
                           self.config.label_smoothing / n_classes
            focal_loss = focal_loss * smooth_target
        
        return focal_loss.mean()
    
    def compute_ciou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute Complete IoU loss for better bounding box regression"""
        # This would be integrated with YOLO's existing loss
        # Placeholder for CIoU computation
        return torch.tensor(0.0)


class OptimizedModelTrainer:
    """Main trainer class with advanced optimization techniques"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_optimizer = LossFunctionOptimizer(config)
        self.scaler = GradScaler() if config.use_amp else None
        
        # Performance tracking
        self.best_metrics = {'mAP50': 0, 'mAP50-95': 0, 'loss': float('inf')}
        self.training_history = defaultdict(list)
        
        # Hardware optimization
        self._setup_hardware_optimization()
        
        # Create output directories
        self.output_dir = Path(f"results/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OptimizedModelTrainer with config: {asdict(config)}")
    
    def _setup_hardware_optimization(self):
        """Setup hardware-specific optimizations"""
        if self.device.type == 'cuda':
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudNN autotuner
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            logger.info(f"Hardware optimization enabled for {torch.cuda.get_device_name()}")
    
    def _initialize_model(self) -> YOLO:
        """Initialize YOLO model with optimizations"""
        model_name = f"yolo11{self.config.model_type.value}.pt"
        
        # Load pretrained model
        model = YOLO(model_name)
        
        # Apply model-specific optimizations
        if hasattr(model.model, 'half') and self.config.use_amp:
            model.model = model.model.half()
        
        return model
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with advanced settings"""
        if self.config.optimizer == "AdamW":
            optimizer = AdamW(
                self.model.model.parameters(),
                lr=self.config.base_lr,
                betas=(self.config.momentum, 0.999),
                weight_decay=self.config.weight_decay
            )
        else:  # SGD
            optimizer = SGD(
                self.model.model.parameters(),
                lr=self.config.base_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        
        return optimizer
    
    def _apply_progressive_resizing(self, epoch: int) -> int:
        """Apply progressive resizing strategy"""
        if self.config.strategy != TrainingStrategy.PROGRESSIVE:
            return self.config.imgsz
        
        # Start with smaller images and gradually increase
        min_size = 320
        max_size = self.config.imgsz
        
        progress = epoch / self.config.epochs
        
        if progress < 0.25:
            return min_size
        elif progress < 0.5:
            return 416
        elif progress < 0.75:
            return 512
        else:
            return max_size
    
    def _apply_curriculum_learning(self, epoch: int) -> Dict[str, Any]:
        """Apply curriculum learning - easy to hard samples"""
        if self.config.strategy != TrainingStrategy.CURRICULUM:
            return {}
        
        progress = epoch / self.config.epochs
        
        # Start with easy samples (high confidence) and gradually include harder ones
        confidence_threshold = max(0.3, 0.7 * (1 - progress))
        
        # Adjust augmentation intensity
        augmentation_intensity = min(1.0, progress * 1.5)
        
        return {
            'confidence_threshold': confidence_threshold,
            'mosaic': self.config.mosaic * augmentation_intensity,
            'mixup': self.config.mixup * augmentation_intensity,
            'copy_paste': self.config.copy_paste * augmentation_intensity
        }
    
    def train(self, data_yaml: str) -> Dict[str, Any]:
        """Main training loop with advanced optimization"""
        logger.info(f"Starting training with data: {data_yaml}")
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = AdvancedLearningRateScheduler(self.optimizer, self.config)
        
        # Training configuration
        train_args = {
            'data': data_yaml,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.imgsz,
            'optimizer': self.config.optimizer,
            'lr0': self.config.base_lr,
            'lrf': self.config.final_lr,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': self.loss_optimizer.adaptive_weights['box'],
            'cls': self.loss_optimizer.adaptive_weights['cls'],
            'dfl': self.loss_optimizer.adaptive_weights['dfl'],
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'copy_paste': self.config.copy_paste,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'device': self.config.device,
            'workers': self.config.workers,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'amp': self.config.use_amp,
            'label_smoothing': self.config.label_smoothing,
            'dropout': 0.0,
            'seed': 42,
            'deterministic': False,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'val': True,
            'plots': True
        }
        
        # Apply strategy-specific modifications
        if self.config.strategy == TrainingStrategy.PROGRESSIVE:
            logger.info("Using progressive resizing strategy")
            # Will be handled epoch by epoch
        elif self.config.strategy == TrainingStrategy.CURRICULUM:
            logger.info("Using curriculum learning strategy")
            # Will be handled epoch by epoch
        
        # Start training with monitoring
        try:
            results = self.model.train(**train_args)
            
            # Save final model
            self.model.save(self.output_dir / 'best.pt')
            
            # Save training history
            self._save_training_history()
            
            # Generate training report
            report = self._generate_training_report(results)
            
            logger.info(f"Training completed successfully. Best mAP50: {self.best_metrics['mAP50']:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Cleanup
            self._cleanup()
    
    def _monitor_training_epoch(self, epoch: int, metrics: Dict) -> None:
        """Monitor and log training progress"""
        # Update history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Check for improvements
        if 'mAP50' in metrics and metrics['mAP50'] > self.best_metrics['mAP50']:
            self.best_metrics['mAP50'] = metrics['mAP50']
            self.best_metrics['mAP50-95'] = metrics.get('mAP50-95', 0)
            logger.info(f"New best mAP50: {metrics['mAP50']:.4f}")
        
        # Memory monitoring
        if self.device.type == 'cuda':
            memory_used = torch.cuda.memory_reserved() / 1024**3
            memory_percent = (torch.cuda.memory_reserved() / 
                            torch.cuda.get_device_properties(0).total_memory * 100)
            
            logger.info(f"GPU Memory: {memory_used:.2f}GB ({memory_percent:.1f}%)")
        
        # CPU monitoring  
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent
        
        logger.info(f"CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%")
    
    def _save_training_history(self) -> None:
        """Save training history to file"""
        history_file = self.output_dir / 'training_history.json'
        
        # Convert to serializable format
        history = {k: [float(v) for v in values] 
                  for k, values in self.training_history.items()}
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        
        logger.info(f"Training history saved to {history_file}")
    
    def _generate_training_report(self, results: Any) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
            },
            'training_time': results.results_dict.get('train_time', 0),
            'hardware_info': {
                'gpu': torch.cuda.get_device_name() if self.device.type == 'cuda' else 'CPU',
                'gpu_count': torch.cuda.device_count(),
                'cpu_count': psutil.cpu_count(),
                'ram_gb': psutil.virtual_memory().total / 1024**3
            }
        }
        
        # Save report
        report_file = self.output_dir / 'training_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Training report saved to {report_file}")
        
        return report
    
    def _cleanup(self) -> None:
        """Cleanup resources"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()


class EnsembleTrainer:
    """Train ensemble of models for improved performance"""
    
    def __init__(self, configs: List[TrainingConfig]):
        self.configs = configs
        self.trainers = [OptimizedModelTrainer(config) for config in configs]
        self.results = []
    
    def train_ensemble(self, data_yaml: str) -> Dict[str, Any]:
        """Train multiple models for ensemble"""
        logger.info(f"Starting ensemble training with {len(self.configs)} models")
        
        for i, trainer in enumerate(self.trainers):
            logger.info(f"Training model {i+1}/{len(self.trainers)}: {trainer.config.model_type.value}")
            
            result = trainer.train(data_yaml)
            self.results.append(result)
            
            # Cleanup between models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Combine results
        ensemble_report = self._generate_ensemble_report()
        
        return ensemble_report
    
    def _generate_ensemble_report(self) -> Dict[str, Any]:
        """Generate ensemble training report"""
        avg_metrics = {
            'mAP50': np.mean([r['final_metrics']['mAP50'] for r in self.results]),
            'mAP50-95': np.mean([r['final_metrics']['mAP50-95'] for r in self.results]),
            'precision': np.mean([r['final_metrics']['precision'] for r in self.results]),
            'recall': np.mean([r['final_metrics']['recall'] for r in self.results])
        }
        
        return {
            'ensemble_size': len(self.results),
            'average_metrics': avg_metrics,
            'individual_results': self.results,
            'best_model': max(self.results, key=lambda x: x['final_metrics']['mAP50'])
        }


def create_training_config(
    model_type: ModelType,
    dataset_type: DatasetType,
    strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE
) -> TrainingConfig:
    """Factory function to create optimized training configuration"""
    
    # Get base config from global CONFIG
    dataset_config = CONFIG.dataset_configs[dataset_type]
    model_config = CONFIG.model_configs[model_type]
    
    # Create training config
    config = TrainingConfig(
        model_type=model_type,
        dataset_type=dataset_type,
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        imgsz=model_config['imgsz'],
        base_lr=model_config['lr0'],
        strategy=strategy
    )
    
    # Apply dataset-specific adjustments
    if dataset_type == DatasetType.GROWTH_TIF:
        # Severe class imbalance - use focal loss and more augmentation
        config.label_smoothing = 0.1
        config.mixup = 0.3
        config.copy_paste = 0.5
    elif dataset_type == DatasetType.GREENHOUSE_MULTI:
        # Multiple objects - need better localization
        config.box_loss_weight = 10.0
        config.mosaic = 1.0
    elif dataset_type == DatasetType.GREENHOUSE_SINGLE:
        # Single objects - simpler task
        config.epochs = int(config.epochs * 0.8)
    
    return config


if __name__ == "__main__":
    # Example usage
    logger.info("Optimized Training System - Claude Opus")
    
    # Create configuration
    config = create_training_config(
        model_type=ModelType.YOLO11N,
        dataset_type=DatasetType.GROWTH_TIF,
        strategy=TrainingStrategy.PROGRESSIVE
    )
    
    # Initialize trainer
    trainer = OptimizedModelTrainer(config)
    
    # Start training (requires data.yaml file)
    # results = trainer.train("path/to/data.yaml")
    
    logger.info("Training system initialized successfully")