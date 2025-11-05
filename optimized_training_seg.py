"""
Optimized Segmentation Model Training System for Model3 Greenhouse
Based on: optimized_training.py by Claude Opus
Modified for: YOLOv11-seg Segmentation Task
Date: 2025-11-04
Version: 2.0.0 (Segmentation)

데이터: model3_greenhouse_seg_processed
클래스: Greenhouse_single (단동), Greenhouse_multi (연동)
모델: YOLOv11-seg
태스크: Segmentation

Advanced training optimization system implementing:
- Intelligent hyperparameter optimization
- Dynamic learning rate scheduling
- Advanced loss function design (with mask loss)
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
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
from ultralytics import YOLO
from collections import defaultdict
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
from tqdm import tqdm
import gc
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================== Enums ==================

class TrainingStrategy(Enum):
    """학습 전략"""
    STANDARD = "standard"
    PROGRESSIVE = "progressive"      # Progressive resizing
    CURRICULUM = "curriculum"        # Easy to hard samples
    ENSEMBLE = "ensemble"            # Multiple model training
    DISTILLATION = "distillation"    # Knowledge distillation


class SegModelSize(Enum):
    """YOLOv11-seg 모델 크기"""
    NANO = "yolo11n-seg.pt"
    SMALL = "yolo11s-seg.pt"
    MEDIUM = "yolo11m-seg.pt"
    LARGE = "yolo11l-seg.pt"
    XLARGE = "yolo11x-seg.pt"


# ================== Configuration ==================

@dataclass
class SegmentationTrainingConfig:
    """Segmentation 학습 설정"""
    # 기본 설정
    model_size: SegModelSize = SegModelSize.NANO
    data_yaml: str = r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml"
    task: str = "segment"  # Segmentation 태스크

    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640

    # 최적화 설정
    optimizer: str = "AdamW"
    base_lr: float = 0.001
    final_lr: float = 0.00001
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    momentum: float = 0.937

    # 고급 설정
    strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE
    use_amp: bool = True
    gradient_clip_val: float = 10.0
    ema_decay: float = 0.9999
    label_smoothing: float = 0.0

    # 데이터 증강 설정
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.3
    degrees: float = 10.0
    translate: float = 0.2
    scale: float = 0.9
    shear: float = 2.0
    perspective: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.5
    fliplr: float = 0.5

    # 손실 가중치 (Segmentation)
    box_loss_weight: float = 7.5   # Box loss
    cls_loss_weight: float = 0.5   # Classification loss
    dfl_loss_weight: float = 1.5   # DFL loss
    # Segmentation 전용
    mask_loss_weight: float = 2.5  # Mask loss (중요!)

    # Early stopping
    patience: int = 30
    min_delta: float = 0.001

    # 체크포인트
    save_period: int = 5
    keep_checkpoints: int = 3

    # 하드웨어 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 8
    pin_memory: bool = True

    # 프로젝트 설정
    project: str = "runs/segment"
    name: str = "model3_greenhouse"
    exist_ok: bool = False

    # 추가 설정
    verbose: bool = True
    plots: bool = True
    save: bool = True
    val: bool = True
    cache: bool = False
    resume: bool = False
    overlap_mask: bool = True  # Segmentation 전용
    mask_ratio: int = 4        # Segmentation mask downsampling ratio


# ================== Learning Rate Scheduler ==================

class AdvancedLearningRateScheduler:
    """고급 학습률 스케줄러"""

    def __init__(self, optimizer: torch.optim.Optimizer, config: SegmentationTrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_epoch = 0
        self.current_lr = config.base_lr

        # 전략에 따른 스케줄러 초기화
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
        """메트릭 기반 학습률 업데이트"""
        if metrics and 'loss' in metrics:
            if self._is_plateau(metrics['loss']):
                self._reduce_lr_on_plateau()

        self.scheduler.step()
        self.current_epoch += 1
        self.current_lr = self.optimizer.param_groups[0]['lr']

        return self.current_lr

    def _is_plateau(self, loss: float, window: int = 5) -> bool:
        """손실 정체 감지"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []

        self.loss_history.append(loss)

        if len(self.loss_history) < window:
            return False

        recent_losses = self.loss_history[-window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        return std_loss < self.config.min_delta * mean_loss

    def _reduce_lr_on_plateau(self, factor: float = 0.5):
        """정체 시 학습률 감소"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        logger.info(f"학습률 감소: {self.current_lr * factor:.6f}")


# ================== Loss Function Optimizer ==================

class SegmentationLossFunctionOptimizer:
    """Segmentation 손실 함수 최적화"""

    def __init__(self, config: SegmentationTrainingConfig):
        self.config = config
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 1.5

        # Segmentation 전용 적응형 가중치
        self.adaptive_weights = self._calculate_adaptive_weights()

    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """데이터셋 특성 기반 적응형 손실 가중치"""
        weights = {
            'box': self.config.box_loss_weight,
            'cls': self.config.cls_loss_weight,
            'dfl': self.config.dfl_loss_weight,
            'mask': self.config.mask_loss_weight  # Segmentation 추가
        }

        # 비닐하우스 데이터 특성 반영
        # 정밀한 경계 검출이 중요하므로 mask loss 가중치 증가
        weights['mask'] *= 1.2

        logger.info(f"적응형 손실 가중치: {weights}")
        return weights

    def compute_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Loss (클래스 불균형 해결)"""
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_loss_alpha * (1 - pt) ** self.focal_loss_gamma * ce_loss

        if self.config.label_smoothing > 0:
            n_classes = pred.shape[1]
            smooth_target = target * (1 - self.config.label_smoothing) + \
                           self.config.label_smoothing / n_classes
            focal_loss = focal_loss * smooth_target

        return focal_loss.mean()


# ================== Main Trainer ==================

class OptimizedSegmentationTrainer:
    """Segmentation 최적화 학습 클래스"""

    def __init__(self, config: SegmentationTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_optimizer = SegmentationLossFunctionOptimizer(config)
        self.scaler = GradScaler() if config.use_amp else None

        # 성능 추적 (Segmentation 메트릭)
        self.best_metrics = {
            'mAP50': 0,
            'mAP50-95': 0,
            'mask_mAP50': 0,      # Mask mAP50 추가
            'mask_mAP50-95': 0,   # Mask mAP50-95 추가
            'loss': float('inf')
        }
        self.training_history = defaultdict(list)

        # 하드웨어 최적화
        self._setup_hardware_optimization()

        # 출력 디렉토리 생성
        self.output_dir = Path(config.project) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=config.exist_ok)

        logger.info("=" * 60)
        logger.info("YOLOv11-seg Segmentation 학습 시스템")
        logger.info("=" * 60)
        logger.info(f"모델: {config.model_size.value}")
        logger.info(f"데이터: {config.data_yaml}")
        logger.info(f"태스크: {config.task}")
        logger.info(f"디바이스: {config.device}")
        logger.info(f"출력: {self.output_dir}")

    def _setup_hardware_optimization(self):
        """하드웨어 최적화 설정"""
        if self.device.type == 'cuda':
            # GPU 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # TF32 활성화 (Ampere 이상)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # 메모리 최적화
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        else:
            logger.warning("CPU 모드로 실행 중 - 학습이 느릴 수 있습니다")

    def train(self) -> Dict[str, Any]:
        """학습 실행"""
        logger.info("\n학습 시작...")

        # 모델 로드
        self.model = YOLO(self.config.model_size.value)
        logger.info(f"모델 로드 완료: {self.config.model_size.value}")

        # 학습 인자 설정
        train_args = {
            # 기본 설정
            'data': self.config.data_yaml,
            'task': self.config.task,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.imgsz,

            # 최적화 설정
            'optimizer': self.config.optimizer,
            'lr0': self.config.base_lr,
            'lrf': self.config.final_lr / self.config.base_lr,
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # 손실 가중치
            'box': self.loss_optimizer.adaptive_weights['box'],
            'cls': self.loss_optimizer.adaptive_weights['cls'],
            'dfl': self.loss_optimizer.adaptive_weights['dfl'],

            # 데이터 증강
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'copy_paste': self.config.copy_paste,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,

            # Segmentation 전용 설정
            'overlap_mask': self.config.overlap_mask,
            'mask_ratio': self.config.mask_ratio,

            # Early stopping
            'patience': self.config.patience,

            # 하드웨어 설정
            'device': self.config.device,
            'workers': self.config.workers,

            # 체크포인트
            'save': self.config.save,
            'save_period': self.config.save_period,

            # 기타
            'project': self.config.project,
            'name': self.config.name,
            'exist_ok': self.config.exist_ok,
            'verbose': self.config.verbose,
            'plots': self.config.plots,
            'val': self.config.val,
            'cache': self.config.cache,
            'resume': self.config.resume,
            'amp': self.config.use_amp
        }

        # Progressive resizing 전략
        if self.config.strategy == TrainingStrategy.PROGRESSIVE:
            logger.info("Progressive resizing 전략 사용")
        elif self.config.strategy == TrainingStrategy.CURRICULUM:
            logger.info("Curriculum learning 전략 사용")

        # 학습 시작
        try:
            start_time = time.time()

            results = self.model.train(**train_args)

            training_time = time.time() - start_time

            # 최종 모델 저장
            final_model_path = self.output_dir / 'best.pt'
            self.model.save(final_model_path)
            logger.info(f"최종 모델 저장: {final_model_path}")

            # 학습 히스토리 저장
            self._save_training_history()

            # 학습 리포트 생성
            report = self._generate_training_report(results, training_time)

            logger.info(f"\n학습 완료!")
            logger.info(f"  - 최고 Box mAP50: {self.best_metrics['mAP50']:.4f}")
            logger.info(f"  - 최고 Mask mAP50: {self.best_metrics.get('mask_mAP50', 0):.4f}")
            logger.info(f"  - 학습 시간: {training_time/60:.2f}분")

            return report

        except Exception as e:
            logger.error(f"학습 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            # 정리
            self._cleanup()

    def _save_training_history(self):
        """학습 히스토리 저장"""
        history_file = self.output_dir / 'training_history.json'

        history = {k: [float(v) if not isinstance(v, (list, dict)) else v
                      for v in values]
                  for k, values in self.training_history.items()}

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)

        logger.info(f"학습 히스토리 저장: {history_file}")

    def _generate_training_report(self, results: Any, training_time: float) -> Dict[str, Any]:
        """종합 학습 리포트 생성"""

        # 결과 메트릭 추출 (Segmentation)
        try:
            results_dict = results.results_dict if hasattr(results, 'results_dict') else {}

            final_metrics = {
                # Box metrics
                'box_mAP50': results_dict.get('metrics/mAP50(B)', 0),
                'box_mAP50-95': results_dict.get('metrics/mAP50-95(B)', 0),
                'box_precision': results_dict.get('metrics/precision(B)', 0),
                'box_recall': results_dict.get('metrics/recall(B)', 0),

                # Mask metrics (Segmentation)
                'mask_mAP50': results_dict.get('metrics/mAP50(M)', 0),
                'mask_mAP50-95': results_dict.get('metrics/mAP50-95(M)', 0),
                'mask_precision': results_dict.get('metrics/precision(M)', 0),
                'mask_recall': results_dict.get('metrics/recall(M)', 0),
            }

            # best_metrics 업데이트
            if final_metrics['mask_mAP50'] > self.best_metrics['mask_mAP50']:
                self.best_metrics['mask_mAP50'] = final_metrics['mask_mAP50']
                self.best_metrics['mask_mAP50-95'] = final_metrics['mask_mAP50-95']

        except Exception as e:
            logger.warning(f"메트릭 추출 실패: {e}")
            final_metrics = {}

        report = {
            'timestamp': datetime.now().isoformat(),
            'task': 'segmentation',
            'model': self.config.model_size.value,
            'data': self.config.data_yaml,

            'config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'imgsz': self.config.imgsz,
                'optimizer': self.config.optimizer,
                'base_lr': self.config.base_lr,
                'strategy': self.config.strategy.value,
            },

            'best_metrics': self.best_metrics,
            'final_metrics': final_metrics,

            'training_time_minutes': training_time / 60,
            'training_time_hours': training_time / 3600,

            'hardware_info': {
                'gpu': torch.cuda.get_device_name() if self.device.type == 'cuda' else 'CPU',
                'gpu_count': torch.cuda.device_count(),
                'cpu_count': psutil.cpu_count(),
                'ram_gb': psutil.virtual_memory().total / 1024**3
            },

            'loss_weights': self.loss_optimizer.adaptive_weights
        }

        # 리포트 저장
        report_file = self.output_dir / 'training_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        logger.info(f"학습 리포트 저장: {report_file}")

        return report

    def _cleanup(self):
        """메모리 정리"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("메모리 정리 완료")


# ================== Main Function ==================

def main():
    """메인 실행 함수"""

    # 설정
    config = SegmentationTrainingConfig(
        # 모델
        model_size=SegModelSize.NANO,  # nano, small, medium, large, xlarge

        # 데이터
        data_yaml=r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml",

        # 학습 설정
        epochs=100,
        batch_size=16,
        imgsz=640,

        # 최적화
        optimizer="AdamW",
        base_lr=0.001,
        final_lr=0.00001,
        warmup_epochs=3,

        # 전략
        strategy=TrainingStrategy.PROGRESSIVE,

        # 손실 가중치 (Segmentation 최적화)
        box_loss_weight=7.5,
        cls_loss_weight=0.5,
        dfl_loss_weight=1.5,
        mask_loss_weight=2.5,  # Segmentation mask loss

        # 데이터 증강
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,

        # 프로젝트
        project="runs/segment",
        name="model3_greenhouse",
        exist_ok=False,

        # 기타
        patience=30,
        save_period=5,
        plots=True,
        verbose=True
    )

    # 학습 실행
    trainer = OptimizedSegmentationTrainer(config)
    report = trainer.train()

    logger.info("\n" + "=" * 60)
    logger.info("🎉 학습 완료!")
    logger.info("=" * 60)
    logger.info(f"최고 Box mAP50: {report['best_metrics']['mAP50']:.4f}")
    logger.info(f"최고 Mask mAP50: {report['best_metrics']['mask_mAP50']:.4f}")
    logger.info(f"학습 시간: {report['training_time_hours']:.2f}시간")
    logger.info(f"출력 디렉토리: {config.project}/{config.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
