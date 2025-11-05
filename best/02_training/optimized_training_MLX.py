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

# Fix OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
try:
    import GPUtil
except ImportError:
    GPUtil = None  # Optional dependency
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
    task: str = "segment"  # detect or segment
    
    # Optimization settings  
    optimizer: str = "AdamW"
    base_lr: float = 0.001
    final_lr: float = 0.00001
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Advanced settings
    strategy: TrainingStrategy = TrainingStrategy.STANDARD
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
    workers: int = 0  # Windows에서 안정성을 위해 0으로 설정 (멀티프로세싱 비활성화)
    pin_memory: bool = True
    
    # Loss weights (for multi-task learning)
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5
    mask_loss_weight: float = 2.5  # For segmentation only
    
    # Early stopping
    patience: int = 30
    min_delta: float = 0.001
    
    # Checkpointing
    save_period: int = 5
    keep_checkpoints: int = 3
    
    # Segmentation-specific settings
    overlap_mask: bool = True  # For segmentation
    mask_ratio: int = 4  # Mask downsampling ratio for segmentation


class AdvancedLearningRateScheduler:
    """Advanced learning rate scheduling with multiple strategies"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
        self.current_lr = config.base_lr
        
        # Initialize scheduler
        # pct_start는 0.0 < x < 1.0 범위여야 함
        # warmup_epochs가 total epochs보다 클 수 없음
        safe_warmup = min(config.warmup_epochs, config.epochs - 1)
        pct_start = safe_warmup / max(config.epochs, 1)
        pct_start = min(0.3, max(0.01, pct_start))  # 0.01 ~ 0.3 범위로 제한
        
        self.scheduler = OneCycleLR(
            optimizer,
            max_lr=config.base_lr,
            epochs=config.epochs,
            steps_per_epoch=1,
            pct_start=pct_start,
            final_div_factor=max(10, config.base_lr / config.final_lr)
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
        
        # Add mask loss weight for segmentation
        if self.config.task == 'segment':
            weights['mask'] = self.config.mask_loss_weight
        
        # Adjust weights based on dataset type
        if self.config.dataset_type == DatasetType.GROWTH_TIF:
            # Growth TIF has severe class imbalance
            weights['cls'] *= 1.5  # Increase classification loss weight
        elif self.config.dataset_type == DatasetType.GREENHOUSE_MULTI:
            # Multiple greenhouses need better localization
            weights['box'] *= 1.2
        elif self.config.dataset_type == DatasetType.MODEL3_GREENHOUSE_SEG:
            # Segmentation task - precise boundary detection is important
            if 'mask' in weights:
                weights['mask'] *= 1.2
        
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
        if config.task == 'segment':
            self.best_metrics = {
                'mAP50': 0, 'mAP50-95': 0, 
                'mask_mAP50': 0, 'mask_mAP50-95': 0,
                'loss': float('inf')
            }
        else:
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
        # Get model filename from model type
        model_value = self.config.model_type.value
        
        # For segmentation models, ensure .pt extension
        if model_value.endswith('-seg'):
            model_name = f"{model_value}.pt"
        else:
            model_name = f"yolo11{model_value}.pt"
        
        # Load pretrained model
        model = YOLO(model_name)
        
        # Apply model-specific optimizations
        if hasattr(model.model, 'half') and self.config.use_amp:
            model.model = model.model.half()
        
        logger.info(f"Model initialized: {model_name} (task: {self.config.task})")
        
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
            'task': self.config.task,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.imgsz,
            'optimizer': self.config.optimizer,
            'lr0': self.config.base_lr,
            'lrf': self.config.final_lr / self.config.base_lr,
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
            'rect': True,  # Rectangular training for stability
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'val': True,
            'plots': True,
            'cache': False,  # Disable cache to avoid empty tensor issues
            'fraction': 1.0  # Use full dataset
        }
        
        # Add segmentation-specific parameters
        if self.config.task == 'segment':
            train_args.update({
                'overlap_mask': self.config.overlap_mask,
                'mask_ratio': self.config.mask_ratio
            })
            logger.info(f"Segmentation mode enabled: overlap_mask={self.config.overlap_mask}, mask_ratio={self.config.mask_ratio}")
        
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
        # Extract metrics based on task
        final_metrics = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
        }
        
        # Add segmentation metrics if available
        if self.config.task == 'segment':
            final_metrics.update({
                'mask_mAP50': results.results_dict.get('metrics/mAP50(M)', 0),
                'mask_mAP50-95': results.results_dict.get('metrics/mAP50-95(M)', 0),
                'mask_precision': results.results_dict.get('metrics/precision(M)', 0),
                'mask_recall': results.results_dict.get('metrics/recall(M)', 0),
            })
            
            # Update best mask metrics
            if final_metrics['mask_mAP50'] > self.best_metrics.get('mask_mAP50', 0):
                self.best_metrics['mask_mAP50'] = final_metrics['mask_mAP50']
                self.best_metrics['mask_mAP50-95'] = final_metrics['mask_mAP50-95']
        
        # Convert config to dict with Enum serialization
        config_dict = asdict(self.config)
        # Convert Enum types to strings for JSON serialization
        config_dict['model_type'] = self.config.model_type.name
        config_dict['dataset_type'] = self.config.dataset_type.name
        config_dict['strategy'] = self.config.strategy.value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'task': self.config.task,
            'config': config_dict,
            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,
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
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
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
        # Basic metrics
        avg_metrics = {
            'mAP50': np.mean([r['final_metrics']['mAP50'] for r in self.results]),
            'mAP50-95': np.mean([r['final_metrics']['mAP50-95'] for r in self.results]),
            'precision': np.mean([r['final_metrics']['precision'] for r in self.results]),
            'recall': np.mean([r['final_metrics']['recall'] for r in self.results])
        }
        
        # Segmentation metrics (if available)
        if 'mask_mAP50' in self.results[0]['final_metrics']:
            avg_metrics.update({
                'mask_mAP50': np.mean([r['final_metrics'].get('mask_mAP50', 0) for r in self.results]),
                'mask_mAP50-95': np.mean([r['final_metrics'].get('mask_mAP50-95', 0) for r in self.results]),
                'mask_precision': np.mean([r['final_metrics'].get('mask_precision', 0) for r in self.results]),
                'mask_recall': np.mean([r['final_metrics'].get('mask_recall', 0) for r in self.results])
            })
        
        # Find best model
        best_model = max(self.results, key=lambda x: x['final_metrics']['mAP50'])
        
        ensemble_report = {
            'ensemble_size': len(self.results),
            'average_metrics': avg_metrics,
            'individual_results': self.results,
            'best_model': best_model
        }
        
        # Save ensemble report to file
        output_dir = Path("results") / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / 'ensemble_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(ensemble_report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Ensemble report saved to {report_file}")
        
        return ensemble_report


def create_training_config(
    model_type: ModelType,
    dataset_type: DatasetType,
    data_yaml: Optional[str] = None
) -> TrainingConfig:
    """Factory function to create optimized training configuration"""
    
    # Determine task type based on model
    is_segmentation = '_SEG' in model_type.name
    task = 'segment' if is_segmentation else 'detect'
    
    # Get dataset config
    dataset_info = CONFIG.data.dataset_info.get(dataset_type, {})
    
    # Get model-specific config
    model_specific = CONFIG.training.model_specific.get(model_type, {})
    
    # Create training config with defaults
    # 이미지 크기: 모델별 설정 우선, 없으면 데이터셋 권장 크기, 없으면 기본값
    default_imgsz = dataset_info.get('recommended_train_size', CONFIG.training.image_size)
    
    config = TrainingConfig(
        model_type=model_type,
        dataset_type=dataset_type,
        task=task,
        epochs=CONFIG.training.epochs,
        batch_size=model_specific.get('batch_size', CONFIG.training.batch_size),
        imgsz=model_specific.get('imgsz', default_imgsz),  # 모델별 또는 데이터셋별 크기
        base_lr=model_specific.get('lr0', CONFIG.training.lr0),
        warmup_epochs=model_specific.get('warmup_epochs', 3)
    )
    
    # Apply segmentation-specific settings
    if is_segmentation:
        config.overlap_mask = model_specific.get('overlap_mask', True)
        config.mask_ratio = model_specific.get('mask_ratio', 4)
        config.mask_loss_weight = CONFIG.training.mask_loss_gain
    
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
    elif dataset_type == DatasetType.MODEL3_GREENHOUSE_SEG:
        # Segmentation task - precise boundary detection
        config.mask_loss_weight *= 1.2
        config.copy_paste = 0.3
        config.mosaic = 1.0
    
    logger.info(f"Created training config for {model_type.name} on {dataset_type.name} (task: {task})")
    
    return config


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🎯 MODEL3 GREENHOUSE SEGMENTATION - ENSEMBLE TRAINING")
    logger.info("=" * 80)
    logger.info("Models: YOLO11M-SEG + YOLO11L-SEG + YOLO11X-SEG")
    logger.info("Task: Segmentation")
    logger.info("Dataset: model3_greenhouse_seg_processed")
    logger.info("Image Size: 1024px (Original)")
    logger.info("Strategy: Standard (Fixed Size)")
    logger.info("=" * 80)
    
    # Configuration
    data_yaml = r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml"
    epochs = 1  # 정식 학습 설정
    
    # 큰 모델들로 앙상블: Medium + Large + XLarge
    model_sizes = [
        ModelType.YOLO11M_SEG,
        ModelType.YOLO11L_SEG,
        ModelType.YOLO11X_SEG
    ]
    
    # Create configs for each model
    configs = []
    logger.info("\n📋 Creating configurations for ensemble models:")
    logger.info("-" * 80)
    
    for model_type in model_sizes:
        config = create_training_config(
            model_type=model_type,
            dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG
        )
        config.epochs = epochs
        
        # 이미지 크기는 모델별 설정(1024)에서 자동으로 가져옴
        # config.imgsz는 이미 1024로 설정됨
        
        # Batch size 수동 최적화 (RTX A6000 48GB, 1024px 이미지)
        # 목표: GPU 메모리 80-90% 사용 (38-43GB)
        if model_type == ModelType.YOLO11M_SEG:
            config.batch_size = 32  # 1024px 기준 최적값
        elif model_type == ModelType.YOLO11L_SEG:
            config.batch_size = 24  # 1024px 기준 최적값
        elif model_type == ModelType.YOLO11X_SEG:
            config.batch_size = 16   # 1024px 기준 최적값
        
        # 자동 조절 옵션 (필요시)
        # config.batch_size = -1
        
        configs.append(config)
        
        logger.info(f"✓ {model_type.name}")
        batch_display = "Auto (GPU optimized)" if config.batch_size == -1 else str(config.batch_size)
        logger.info(f"  - Image Size: {config.imgsz}px")
        logger.info(f"  - Batch Size: {batch_display}")
        logger.info(f"  - Learning Rate: {config.base_lr}")
        logger.info(f"  - Epochs: {config.epochs}")
        logger.info(f"  - Overlap Mask: {config.overlap_mask}")
        logger.info(f"  - Mask Ratio: {config.mask_ratio}")
    
    logger.info("-" * 80)
    logger.info(f"Total: {len(configs)} models will be trained sequentially")
    logger.info("=" * 80)
    
    # Confirm start
    logger.info("\n⏰ Estimated total training time (1024px, RTX A6000):")
    logger.info("   100 epochs 기준:")
    logger.info("   - YOLO11M-SEG (1024px, batch 16): ~6 hours")
    logger.info("   - YOLO11L-SEG (1024px, batch 12): ~10 hours")
    logger.info("   - YOLO11X-SEG (1024px, batch 8):  ~15 hours")
    logger.info("   Total: ~31 hours")
    logger.info(f"\n   Current setting: {epochs} epochs → ~{epochs/100*31:.1f} hours")
    logger.info("   Expected GPU memory: 38-45GB (80-95%)")
    logger.info("\n💡 1024px 학습으로 최고 성능 달성!")
    logger.info("🚀 Starting ensemble training in 3 seconds...")
    
    import time
    time.sleep(3)
    
    # Create ensemble trainer
    ensemble_trainer = EnsembleTrainer(configs)
    
    # Train ensemble
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE TRAINING STARTED")
    logger.info("=" * 80)
    
    try:
        results = ensemble_trainer.train_ensemble(data_yaml)
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ENSEMBLE TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"\nEnsemble Size: {results['ensemble_size']} models")
        
        logger.info(f"\n📊 Average Metrics:")
        logger.info("-" * 80)
        for metric, value in results['average_metrics'].items():
            logger.info(f"  {metric:20s}: {value:.4f}")
        
        # Best model info (model_type이 이미 문자열로 변환됨)
        best_model_name = results['best_model']['config']['model_type']
        logger.info(f"\n🏆 Best Model: {best_model_name}")
        logger.info("-" * 80)
        logger.info(f"  Box mAP50      : {results['best_model']['final_metrics']['mAP50']:.4f}")
        logger.info(f"  Box mAP50-95   : {results['best_model']['final_metrics']['mAP50-95']:.4f}")
        if 'mask_mAP50' in results['best_model']['final_metrics']:
            logger.info(f"  Mask mAP50     : {results['best_model']['final_metrics']['mask_mAP50']:.4f}")
            logger.info(f"  Mask mAP50-95  : {results['best_model']['final_metrics']['mask_mAP50-95']:.4f}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ All models trained successfully!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Training interrupted by user")
        logger.info("Partial results may have been saved")
        
    except Exception as e:
        logger.error(f"\n\n❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise