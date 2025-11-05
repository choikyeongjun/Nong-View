#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nong-View Best Performance Configuration
최적 성능을 위한 통합 설정 시스템

개발팀: Claude Opus (Architecture) + Claude Sonnet (Implementation)
버전: 1.0.0
날짜: 2025-10-28
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class ModelType(Enum):
    """지원하는 모델 타입"""
    # Detection 모델
    YOLO11N = "yolo11n"
    YOLO11S = "yolo11s"
    YOLO11M = "yolo11m"
    YOLO11L = "yolo11l"
    YOLO11X = "yolo11x"
    # Segmentation 모델
    YOLO11N_SEG = "yolo11n-seg"
    YOLO11S_SEG = "yolo11s-seg"
    YOLO11M_SEG = "yolo11m-seg"
    YOLO11L_SEG = "yolo11l-seg"
    YOLO11X_SEG = "yolo11x-seg"

class DatasetType(Enum):
    """지원하는 데이터셋 타입"""
    # Detection 데이터셋
    GREENHOUSE_MULTI = "greenhouse_multi"
    GREENHOUSE_SINGLE = "greenhouse_single"
    GROWTH_TIF = "growth_tif"
    # Segmentation 데이터셋
    MODEL3_GREENHOUSE_SEG = "model3_greenhouse_seg"

@dataclass
class HardwareConfig:
    """하드웨어 설정"""
    # GPU 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_memory_fraction: float = 0.95  # RTX A6000 48GB 기준
    cpu_workers: int = field(default_factory=lambda: min(16, os.cpu_count() or 8))
    
    # 메모리 최적화
    use_amp: bool = True  # Automatic Mixed Precision
    use_tf32: bool = True  # TensorFloat-32
    memory_efficient: bool = True
    
    def __post_init__(self):
        """하드웨어 최적화 설정 적용"""
        if self.device == "cuda":
            # TF32 활성화 (Ampere GPU 이상)
            if self.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # cuDNN 벤치마크 활성화
            torch.backends.cudnn.benchmark = True
            
            # 메모리 관리
            if self.memory_efficient:
                torch.cuda.empty_cache()

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    # 경로 설정
    namwon_data_path: str = "D:/Nong-View/Namwon_data"
    output_path: str = "D:/Nong-View/best/results"
    cache_path: str = "D:/Nong-View/best/cache"
    
    # 데이터셋별 정보
    dataset_info: Dict[DatasetType, Dict[str, Any]] = field(default_factory=lambda: {
        DatasetType.GREENHOUSE_MULTI: {
            "path": "dataset_greenhouse_multi",
            "classes": ["greenhouse_multi"],
            "total_images": 569,
            "total_objects": 669,
            "class_balance": {"greenhouse_multi": 1.0}
        },
        DatasetType.GREENHOUSE_SINGLE: {
            "path": "dataset_greenhouse_single", 
            "classes": ["greenhouse_single"],
            "total_images": 358,
            "total_objects": 584,
            "class_balance": {"greenhouse_single": 1.0}
        },
        DatasetType.GROWTH_TIF: {
            "path": "growth_tif_dataset",
            "classes": ["IRG_production", "Rye_production", "Greenhouse_single", 
                       "Greenhouse_multi", "Silage_bale"],
            "total_images": 1356,
            "total_objects": 1862,
            "class_balance": {
                "IRG_production": 0.25,     # 74% → 25% (다운 가중치)
                "Silage_bale": 1.0,         # 16% → 1.0 (기준)
                "Rye_production": 1.5,      # 8% → 1.5 (업 가중치)
                "Greenhouse_single": 3.0,   # 2% → 3.0 (강한 업 가중치)  
                "Greenhouse_multi": 5.0     # 1% → 5.0 (최강 업 가중치)
            }
        },
        DatasetType.MODEL3_GREENHOUSE_SEG: {
            "path": "model3_greenhouse_seg_processed",
            "classes": ["Greenhouse_single", "Greenhouse_multi"],
            "total_images": 3855,  # 원본 1483 + 증강 2372
            "original_images": 1483,
            "augmented_images": 2372,
            "total_objects": 3855,
            "task": "segment",
            "original_image_size": 1024,  # 원본 이미지 크기
            "recommended_train_size": 1024,  # 학습 권장 크기 (원본 크기 유지)
            "split_info": {
                "train": 3558,  # 원본 1186 + 증강 2372
                "val": 147,
                "test": 150
            },
            "augmentation_info": {
                "enabled": True,
                "factor": 3,
                "method": "stratified_split_segmentation"
            },
            "class_balance": {
                "Greenhouse_single": 1.0,
                "Greenhouse_multi": 1.0
            }
        }
    })
    
    # 데이터 분할 설정
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify_method: str = "class_distribution"
    random_seed: int = 42

@dataclass 
class TrainingConfig:
    """훈련 관련 설정"""
    # 기본 훈련 파라미터
    epochs: int = 100
    batch_size: int = 16  # 기본값 (자동 조정됨)
    image_size: int = 640
    patience: int = 30
    save_period: int = 10
    
    # 학습률 설정
    lr0: float = 0.01  # 초기 학습률
    lrf: float = 0.01  # 최종 학습률 비율
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # 옵티마이저 설정
    optimizer: str = "AdamW"  # SGD, Adam, AdamW
    scheduler: str = "cosine"  # linear, cosine
    
    # 손실 함수 가중치 (YOLO)
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    # Segmentation 전용
    mask_loss_gain: float = 2.5  # Mask loss weight for segmentation
    
    # 정규화 설정
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    # 데이터 증강 설정
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        # 색상 변환
        "hsv_h": 0.015,  # 색조
        "hsv_s": 0.7,    # 채도  
        "hsv_v": 0.4,    # 명도
        
        # 기하학적 변환
        "degrees": 10.0,    # 회전
        "translate": 0.1,   # 이동
        "scale": 0.5,       # 스케일
        "shear": 2.0,       # 전단
        "perspective": 0.0, # 원근
        
        # 플립
        "flipud": 0.5,      # 상하 반전
        "fliplr": 0.5,      # 좌우 반전
        
        # 고급 증강
        "mosaic": 0.5,      # 모자이크 (감소)
        "mixup": 0.1,       # 믹스업
        "copy_paste": 0.3,  # 카피-페이스트 (증가)
        
        # 품질 향상 기법
        "erasing": 0.4,     # 랜덤 지우기
        "crop_fraction": 1.0, # 크롭 비율
    })
    
    # 모델별 최적 설정
    model_specific: Dict[ModelType, Dict[str, Any]] = field(default_factory=lambda: {
        # Detection 모델
        ModelType.YOLO11N: {
            "batch_size": 32,
            "lr0": 0.01,
            "warmup_epochs": 3
        },
        ModelType.YOLO11S: {
            "batch_size": 24,
            "lr0": 0.01, 
            "warmup_epochs": 3
        },
        ModelType.YOLO11M: {
            "batch_size": 16,
            "lr0": 0.008,
            "warmup_epochs": 5
        },
        ModelType.YOLO11L: {
            "batch_size": 12,
            "lr0": 0.005,
            "warmup_epochs": 5  
        },
        ModelType.YOLO11X: {
            "batch_size": 8,
            "lr0": 0.003,
            "warmup_epochs": 5
        },
        # Segmentation 모델 (1024px 원본 데이터 최적화)
        ModelType.YOLO11N_SEG: {
            "batch_size": 8,   # 1024px 기준 (640px일 때 16)
            "imgsz": 1024,     # 원본 크기 유지
            "lr0": 0.001,
            "warmup_epochs": 3,
            "overlap_mask": True,
            "mask_ratio": 4
        },
        ModelType.YOLO11S_SEG: {
            "batch_size": 6,   # 1024px 기준 (640px일 때 12)
            "imgsz": 1024,
            "lr0": 0.001,
            "warmup_epochs": 3,
            "overlap_mask": True,
            "mask_ratio": 4
        },
        ModelType.YOLO11M_SEG: {
            "batch_size": 16,  # 1024px 기준, RTX A6000 최적화
            "imgsz": 1024,
            "lr0": 0.0008,
            "warmup_epochs": 5,
            "overlap_mask": True,
            "mask_ratio": 4
        },
        ModelType.YOLO11L_SEG: {
            "batch_size": 12,  # 1024px 기준, RTX A6000 최적화
            "imgsz": 1024,
            "lr0": 0.0005,
            "warmup_epochs": 5,
            "overlap_mask": True,
            "mask_ratio": 4
        },
        ModelType.YOLO11X_SEG: {
            "batch_size": 8,   # 1024px 기준, RTX A6000 최적화
            "imgsz": 1024,
            "lr0": 0.0003,
            "warmup_epochs": 5,
            "overlap_mask": True,
            "mask_ratio": 4
        }
    })

@dataclass
class InferenceConfig:
    """추론 관련 설정"""
    # 추론 파라미터
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 1000
    
    # 배치 처리
    batch_size: int = 16
    
    # 이미지 전처리
    image_size: int = 640
    normalize: bool = True
    
    # 후처리 최적화
    agnostic_nms: bool = False  # 클래스 무관 NMS
    multi_label: bool = False   # 다중 라벨
    
    # 성능 최적화
    half_precision: bool = True  # FP16 추론
    device: str = "cuda"
    
    # 결과 저장
    save_txt: bool = True
    save_conf: bool = True
    save_crop: bool = False
    
    # 시각화
    visualize: bool = True
    line_thickness: int = 2
    font_size: float = 1.0

@dataclass
class BenchmarkConfig:
    """벤치마크 관련 설정"""
    # 성능 측정 항목
    metrics: List[str] = field(default_factory=lambda: [
        "mAP@0.5", "mAP@0.75", "mAP@0.5:0.95",
        "precision", "recall", "f1_score",
        "inference_time", "preprocessing_time", "postprocessing_time",
        "gpu_memory", "cpu_usage", "throughput"
    ])
    
    # 벤치마크 설정
    num_runs: int = 5  # 평균 성능 계산용
    warmup_runs: int = 3  # 웜업 실행
    
    # 비교 대상
    baseline_models: List[ModelType] = field(default_factory=lambda: [
        ModelType.YOLO11N, ModelType.YOLO11S, ModelType.YOLO11M
    ])
    
    # 결과 저장
    save_detailed: bool = True
    save_plots: bool = True
    save_csv: bool = True

@dataclass
class BestConfig:
    """최적 성능 통합 설정"""
    # 하위 설정들
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig) 
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    
    # 프로젝트 정보
    project_name: str = "Nong-View-Best"
    version: str = "1.0.0"
    description: str = "최적 성능 농업 AI 객체탐지 시스템"
    
    # 로깅 설정
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 실험 추적
    experiment_name: str = field(default_factory=lambda: f"best_experiment_{os.getpid()}")
    save_checkpoints: bool = True
    
    def __post_init__(self):
        """설정 후처리"""
        # 경로 생성
        for path in [self.data.output_path, self.data.cache_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # 하드웨어 최적화 적용
        self.hardware.__post_init__()
    
    def get_model_config(self, model_type: ModelType) -> Dict[str, Any]:
        """모델별 최적 설정 반환"""
        base_config = {
            "epochs": self.training.epochs,
            "batch": self.training.batch_size,
            "imgsz": self.training.image_size,
            "lr0": self.training.lr0,
            "optimizer": self.training.optimizer,
        }
        
        # 모델별 특화 설정 적용
        if model_type in self.training.model_specific:
            base_config.update(self.training.model_specific[model_type])
        
        return base_config
    
    def get_dataset_config(self, dataset_type: DatasetType) -> Dict[str, Any]:
        """데이터셋별 설정 반환"""
        if dataset_type not in self.data.dataset_info:
            raise ValueError(f"지원되지 않는 데이터셋: {dataset_type}")
        
        return self.data.dataset_info[dataset_type]

# 전역 설정 인스턴스
CONFIG = BestConfig()

# 편의 함수들
def get_device():
    """현재 디바이스 반환"""
    return CONFIG.hardware.device

def get_output_path(subdir: str = "") -> Path:
    """출력 경로 반환"""
    path = Path(CONFIG.data.output_path)
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_cache_path(subdir: str = "") -> Path:
    """캐시 경로 반환"""
    path = Path(CONFIG.data.cache_path)
    if subdir:
        path = path / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path

def setup_logging():
    """로깅 설정"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level),
        format=CONFIG.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(get_output_path() / "best_performance.log")
        ]
    )
    
    return logging.getLogger(__name__)

# 로거 초기화
logger = setup_logging()

if __name__ == "__main__":
    # 설정 검증 및 출력
    print("🏆 Nong-View Best Performance Configuration")
    print("=" * 60)
    print(f"프로젝트: {CONFIG.project_name} v{CONFIG.version}")
    print(f"디바이스: {CONFIG.hardware.device}")
    print(f"GPU 메모리: {CONFIG.hardware.gpu_memory_fraction * 100:.0f}%")
    print(f"CPU 워커: {CONFIG.hardware.cpu_workers}개")
    print(f"출력 경로: {CONFIG.data.output_path}")
    
    print("\n📊 데이터셋 정보:")
    for dataset_type, info in CONFIG.data.dataset_info.items():
        print(f"  - {dataset_type.value}: {info['total_images']}개 이미지")
    
    print(f"\n🎯 성능 목표:")
    print(f"  - 정확도: mAP@0.5 > 95%")
    print(f"  - 속도: 추론 < 200ms/이미지")
    print(f"  - 메모리: GPU 활용률 > 90%")
    print("=" * 60)