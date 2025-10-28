#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nong-View Best Performance - Data Utilities
데이터 처리 관련 유틸리티 함수들

담당: Claude Sonnet (Data Processing & Integration)
개발 날짜: 2025-10-28
"""

import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from PIL import Image
import shutil
import random

# 설정 불러오기
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.best_config import CONFIG, DatasetType, logger

@dataclass
class ImageInfo:
    """이미지 정보 클래스"""
    filename: str
    filepath: str
    width: int
    height: int
    channels: int
    file_size: int
    format: str
    has_labels: bool = False
    label_count: int = 0
    class_distribution: Dict[str, int] = None
    quality_score: float = 0.0
    
    def __post_init__(self):
        if self.class_distribution is None:
            self.class_distribution = {}

@dataclass  
class DatasetStats:
    """데이터셋 통계 클래스"""
    total_images: int
    total_labels: int
    class_distribution: Dict[str, int]
    image_size_distribution: Dict[str, int]
    split_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    
class DataQualityAnalyzer:
    """데이터 품질 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image_quality(self, image_path: str) -> float:
        """이미지 품질 점수 계산 (0-1)"""
        try:
            # 이미지 로드
            if str(image_path).lower().endswith(('.tif', '.tiff')):
                with rasterio.open(image_path) as src:
                    image = src.read()
                    if len(image.shape) == 3:
                        image = np.transpose(image, (1, 2, 0))
                    if image.shape[-1] > 3:
                        image = image[:, :, :3]
            else:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            scores = []
            
            # 1. 명도 분포 점수 (0.3 가중치)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            
            # 히스토그램 엔트로피 (다양성)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
            brightness_score = min(entropy / 8.0, 1.0)  # 8bit 기준 정규화
            scores.append(('brightness', brightness_score, 0.3))
            
            # 2. 대비 점수 (0.25 가중치)  
            contrast = gray.std()
            contrast_score = min(contrast / 64.0, 1.0)  # 경험적 최대값
            scores.append(('contrast', contrast_score, 0.25))
            
            # 3. 선명도 점수 (0.25 가중치)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(sharpness / 1000.0, 1.0)  # 경험적 최대값
            scores.append(('sharpness', sharpness_score, 0.25))
            
            # 4. 노이즈 점수 (0.2 가중치)
            # 가우시안 블러 후 차이로 노이즈 추정
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            noise_score = max(0, 1.0 - noise / 25.0)  # 낮은 노이즈가 좋음
            scores.append(('noise', noise_score, 0.2))
            
            # 가중 평균 계산
            total_score = sum(score * weight for _, score, weight in scores)
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"이미지 품질 분석 실패 {image_path}: {e}")
            return 0.5  # 기본값
    
    def detect_outliers(self, image_stats: List[ImageInfo], 
                       method: str = "isolation_forest") -> List[bool]:
        """이상치 탐지"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # 특성 벡터 생성
            features = []
            for stat in image_stats:
                feature = [
                    stat.width,
                    stat.height, 
                    stat.channels,
                    stat.file_size,
                    stat.label_count,
                    stat.quality_score
                ]
                features.append(feature)
            
            features = np.array(features)
            
            # 표준화
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 이상치 탐지
            if method == "isolation_forest":
                detector = IsolationForest(contamination=0.1, random_state=CONFIG.data.random_seed)
                outliers = detector.fit_predict(features_scaled) == -1
            else:
                # 단순 통계적 방법
                z_scores = np.abs(stats.zscore(features_scaled, axis=0))
                outliers = np.any(z_scores > 3, axis=1)
            
            return outliers.tolist()
            
        except Exception as e:
            self.logger.error(f"이상치 탐지 실패: {e}")
            return [False] * len(image_stats)

class DatasetProcessor:
    """데이터셋 처리 및 분할 클래스"""
    
    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type
        self.dataset_config = CONFIG.get_dataset_config(dataset_type)
        self.dataset_path = Path(CONFIG.data.namwon_data_path) / self.dataset_config["path"]
        self.logger = logging.getLogger(__name__)
        self.quality_analyzer = DataQualityAnalyzer()
    
    def scan_dataset(self) -> Tuple[List[ImageInfo], DatasetStats]:
        """데이터셋 전체 스캔 및 분석"""
        self.logger.info(f"데이터셋 스캔 시작: {self.dataset_type.value}")
        
        image_infos = []
        class_counter = Counter()
        size_counter = Counter()
        total_labels = 0
        
        # 이미지 디렉토리 탐색
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists():
                self.logger.warning(f"이미지 디렉토리 없음: {images_dir}")
                continue
            
            # 이미지 파일 처리
            for img_file in images_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    try:
                        # 이미지 정보 추출
                        info = self._extract_image_info(img_file, labels_dir, split)
                        image_infos.append(info)
                        
                        # 통계 업데이트
                        total_labels += info.label_count
                        for class_name, count in info.class_distribution.items():
                            class_counter[class_name] += count
                        
                        size_key = f"{info.width}x{info.height}"
                        size_counter[size_key] += 1
                        
                    except Exception as e:
                        self.logger.error(f"이미지 처리 실패 {img_file}: {e}")
        
        # 품질 분석
        quality_scores = [info.quality_score for info in image_infos]
        quality_metrics = {
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores)
        }
        
        # 분할 통계
        split_counter = Counter()
        for info in image_infos:
            # 파일 경로에서 분할 추출
            if '/train/' in info.filepath:
                split_counter['train'] += 1
            elif '/val/' in info.filepath:
                split_counter['val'] += 1
            elif '/test/' in info.filepath:
                split_counter['test'] += 1
        
        # 데이터셋 통계 생성
        stats = DatasetStats(
            total_images=len(image_infos),
            total_labels=total_labels,
            class_distribution=dict(class_counter),
            image_size_distribution=dict(size_counter),
            split_distribution=dict(split_counter),
            quality_metrics=quality_metrics
        )
        
        self.logger.info(f"스캔 완료: {len(image_infos)}개 이미지, {total_labels}개 라벨")
        return image_infos, stats
    
    def _extract_image_info(self, img_path: Path, labels_dir: Path, split: str) -> ImageInfo:
        """단일 이미지 정보 추출"""
        # 기본 이미지 정보
        file_size = img_path.stat().st_size
        
        # 이미지 크기 정보
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            # GeoTIFF 처리
            with rasterio.open(img_path) as src:
                width, height = src.width, src.height
                channels = src.count
                img_format = 'TIFF'
        else:
            # 일반 이미지 처리
            with Image.open(img_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                img_format = img.format
        
        # 라벨 정보 추출
        label_path = labels_dir / f"{img_path.stem}.txt"
        has_labels = label_path.exists()
        label_count = 0
        class_distribution = {}
        
        if has_labels:
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    label_count = len(lines)
                    
                    # 클래스 분포 계산
                    class_ids = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_ids.append(int(parts[0]))
                    
                    # 클래스 이름 매핑
                    classes = self.dataset_config["classes"]
                    for class_id in class_ids:
                        if class_id < len(classes):
                            class_name = classes[class_id]
                            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                            
            except Exception as e:
                self.logger.error(f"라벨 파일 읽기 실패 {label_path}: {e}")
        
        # 품질 점수 계산
        quality_score = self.quality_analyzer.analyze_image_quality(img_path)
        
        return ImageInfo(
            filename=img_path.name,
            filepath=str(img_path),
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            format=img_format,
            has_labels=has_labels,
            label_count=label_count,
            class_distribution=class_distribution,
            quality_score=quality_score
        )
    
    def create_balanced_split(self, image_infos: List[ImageInfo], 
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1) -> Dict[str, List[ImageInfo]]:
        """균형잡힌 데이터 분할 생성"""
        self.logger.info("균형잡힌 데이터 분할 생성 중...")
        
        # 분할 비율 검증
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "분할 비율의 합이 1.0이 아닙니다"
        
        # 계층화를 위한 레이블 생성
        stratify_labels = []
        for info in image_infos:
            if info.class_distribution:
                # 주요 클래스 (가장 많은 객체를 가진 클래스)
                dominant_class = max(info.class_distribution, key=info.class_distribution.get)
                stratify_labels.append(dominant_class)
            else:
                stratify_labels.append('no_label')
        
        try:
            # 1차 분할: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                range(len(image_infos)),
                test_size=(val_ratio + test_ratio),
                stratify=stratify_labels,
                random_state=CONFIG.data.random_seed
            )
            
            # 2차 분할: val vs test
            temp_labels = [stratify_labels[i] for i in temp_indices]
            val_size = val_ratio / (val_ratio + test_ratio)
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                stratify=temp_labels,
                random_state=CONFIG.data.random_seed
            )
            
        except ValueError as e:
            self.logger.warning(f"계층화 분할 실패, 랜덤 분할 사용: {e}")
            # 계층화 불가능 시 랜덤 분할
            indices = list(range(len(image_infos)))
            random.Random(CONFIG.data.random_seed).shuffle(indices)
            
            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        # 결과 구성
        splits = {
            'train': [image_infos[i] for i in train_indices],
            'val': [image_infos[i] for i in val_indices],
            'test': [image_infos[i] for i in test_indices]
        }
        
        # 분할 결과 로그
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name}: {len(split_data)}개 이미지 "
                           f"({len(split_data)/len(image_infos)*100:.1f}%)")
        
        return splits

class DataAugmentation:
    """데이터 증강 클래스"""
    
    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type
        self.logger = logging.getLogger(__name__)
    
    def get_training_transforms(self, image_size: int = 640) -> A.Compose:
        """훈련용 데이터 증강 변환"""
        
        # 기본 증강 설정
        aug_config = CONFIG.training.augmentation
        
        transforms = [
            # 크기 조정
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, 
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            
            # 색상 변환
            A.HueSaturationValue(
                hue_shift_limit=int(aug_config["hsv_h"] * 180),
                sat_shift_limit=int(aug_config["hsv_s"] * 255),
                val_shift_limit=int(aug_config["hsv_v"] * 255),
                p=0.7
            ),
            
            # 기하학적 변환
            A.ShiftScaleRotate(
                shift_limit=aug_config["translate"],
                scale_limit=aug_config["scale"],
                rotate_limit=aug_config["degrees"],
                border_mode=cv2.BORDER_CONSTANT,
                p=0.8
            ),
            
            # 플립
            A.HorizontalFlip(p=aug_config["fliplr"]),
            A.VerticalFlip(p=aug_config["flipud"]),
            
            # 품질 개선
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # 정규화 및 텐서 변환
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def get_validation_transforms(self, image_size: int = 640) -> A.Compose:
        """검증용 변환 (증강 없음)"""
        transforms = [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']  
        ))

def export_dataset_yaml(dataset_type: DatasetType, output_path: Path, 
                       splits: Dict[str, List[ImageInfo]]) -> Path:
    """YOLO 형식 dataset.yaml 파일 생성"""
    
    dataset_config = CONFIG.get_dataset_config(dataset_type)
    
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(dataset_config["classes"]),
        'names': dataset_config["classes"],
        
        # 메타데이터
        'description': f'Nong-View Best Performance - {dataset_type.value}',
        'version': CONFIG.version,
        'created': pd.Timestamp.now().isoformat(),
        
        # 통계
        'dataset_stats': {
            'total_images': sum(len(split) for split in splits.values()),
            'train_images': len(splits.get('train', [])),
            'val_images': len(splits.get('val', [])),
            'test_images': len(splits.get('test', [])),
        }
    }
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"dataset.yaml 생성 완료: {yaml_path}")
    return yaml_path

def save_dataset_stats(stats: DatasetStats, output_path: Path) -> Path:
    """데이터셋 통계 저장"""
    stats_path = output_path / 'dataset_stats.json'
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
    
    logger.info(f"데이터셋 통계 저장 완료: {stats_path}")
    return stats_path

if __name__ == "__main__":
    # 테스트 코드
    print("🔧 Data Utils 테스트")
    
    # 각 데이터셋에 대해 간단한 스캔 테스트
    for dataset_type in DatasetType:
        try:
            processor = DatasetProcessor(dataset_type)
            print(f"\n📊 {dataset_type.value} 테스트 중...")
            
            # 데이터셋 경로 확인
            if processor.dataset_path.exists():
                print(f"  ✅ 경로 확인: {processor.dataset_path}")
                
                # 간단한 스캔 테스트
                image_infos, stats = processor.scan_dataset()
                print(f"  📈 이미지: {stats.total_images}개")
                print(f"  🏷️ 라벨: {stats.total_labels}개")
                print(f"  🎯 클래스: {list(stats.class_distribution.keys())}")
                print(f"  ⭐ 평균 품질: {stats.quality_metrics['mean_quality']:.3f}")
            else:
                print(f"  ❌ 경로 없음: {processor.dataset_path}")
                
        except Exception as e:
            print(f"  ❌ 에러: {e}")
    
    print("\n✅ Data Utils 테스트 완료")