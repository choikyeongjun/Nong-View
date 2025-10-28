#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nong-View Best Performance - 최적화된 데이터 전처리 시스템

기존 실제 스크립트의 검증된 기능 + Jupyter 노트북의 고급 알고리즘 통합
- 3개 데이터셋 통합 처리 (greenhouse_multi, greenhouse_single, growth_tif)
- 클래스 불균형 해결 (growth_tif 74% IRG 지배 문제)
- 계층화 분할 (stratified split)
- 데이터 품질 자동 관리
- 농업 특화 데이터 증강

담당: Claude Sonnet (Data Processing & Integration)
개발 날짜: 2025-10-28
"""

import os
import json
import yaml
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import logging

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import rasterio
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A

# 설정 불러오기
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.best_config import CONFIG, DatasetType, ModelType, logger
from utils.data_utils import DatasetProcessor, DataAugmentation, DataQualityAnalyzer, ImageInfo, DatasetStats

@dataclass
class ProcessingStats:
    """처리 통계 클래스"""
    original_images: int = 0
    processed_images: int = 0
    augmented_images: int = 0
    filtered_images: int = 0
    total_objects: int = 0
    class_distribution: Dict[str, int] = None
    quality_metrics: Dict[str, float] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.class_distribution is None:
            self.class_distribution = {}
        if self.quality_metrics is None:
            self.quality_metrics = {}

class OptimizedDataProcessor:
    """최적화된 데이터 처리 메인 클래스"""
    
    def __init__(self, output_dir: str = None, enable_quality_filter: bool = True):
        self.output_dir = Path(output_dir) if output_dir else Path(CONFIG.data.output_path) / "processed_datasets"
        self.enable_quality_filter = enable_quality_filter
        self.logger = logging.getLogger(__name__)
        
        # 하위 클래스 초기화
        self.quality_analyzer = DataQualityAnalyzer()
        self.balanced_creator = BalancedDatasetCreator()
        self.advanced_augmenter = AdvancedDataAugmentation()
        
        # 통계 초기화
        self.stats = defaultdict(ProcessingStats)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"최적화된 데이터 처리기 초기화: {self.output_dir}")
    
    def process_all_datasets(self, target_dataset_types: List[DatasetType] = None) -> Dict[str, ProcessingStats]:
        """모든 데이터셋 통합 처리"""
        if target_dataset_types is None:
            target_dataset_types = list(DatasetType)
        
        self.logger.info(f"통합 데이터 처리 시작: {len(target_dataset_types)}개 데이터셋")
        
        all_results = {}
        
        for dataset_type in target_dataset_types:
            try:
                self.logger.info(f"처리 시작: {dataset_type.value}")
                result = self.process_single_dataset(dataset_type)
                all_results[dataset_type.value] = result
                self.logger.info(f"처리 완료: {dataset_type.value}")
                
            except Exception as e:
                self.logger.error(f"데이터셋 처리 실패 {dataset_type.value}: {e}")
                continue
        
        # 통합 결과 저장
        self._save_processing_report(all_results)
        
        return all_results
    
    def process_single_dataset(self, dataset_type: DatasetType) -> ProcessingStats:
        """단일 데이터셋 최적화 처리"""
        import time
        start_time = time.time()
        
        # 1. 데이터셋 스캔 및 분석
        processor = DatasetProcessor(dataset_type)
        image_infos, dataset_stats = processor.scan_dataset()
        
        if not image_infos:
            raise ValueError(f"이미지가 없습니다: {dataset_type.value}")
        
        self.logger.info(f"데이터셋 스캔 완료: {len(image_infos)}개 이미지")
        
        # 2. 데이터 품질 필터링
        if self.enable_quality_filter:
            image_infos = self._filter_by_quality(image_infos, dataset_type)
        
        # 3. 클래스 불균형 해결
        balanced_infos = self.balanced_creator.create_balanced_dataset(
            image_infos, dataset_type, dataset_stats
        )
        
        # 4. 계층화 분할
        splits = self._create_stratified_splits(balanced_infos, dataset_type)
        
        # 5. 출력 디렉토리 구성
        output_path = self.output_dir / f"best_{dataset_type.value}"
        self._setup_output_structure(output_path)
        
        # 6. 데이터 복사 및 증강
        processed_stats = self._copy_and_augment_data(splits, output_path, dataset_type)
        
        # 7. YAML 및 메타데이터 생성
        self._generate_dataset_files(splits, output_path, dataset_type, processed_stats)
        
        # 처리 시간 기록
        processed_stats.processing_time = time.time() - start_time
        
        self.logger.info(f"데이터셋 처리 완료: {dataset_type.value} ({processed_stats.processing_time:.2f}초)")
        
        return processed_stats
    
    def _filter_by_quality(self, image_infos: List[ImageInfo], dataset_type: DatasetType) -> List[ImageInfo]:
        """데이터 품질 기반 필터링"""
        self.logger.info("데이터 품질 필터링 시작...")
        
        # 품질 점수 임계값 (데이터셋별 조정)
        quality_thresholds = {
            DatasetType.GREENHOUSE_MULTI: 0.4,   # 상대적으로 높은 품질 요구
            DatasetType.GREENHOUSE_SINGLE: 0.4,  
            DatasetType.GROWTH_TIF: 0.3          # TIF는 상대적으로 낮은 임계값
        }
        
        threshold = quality_thresholds.get(dataset_type, 0.35)
        
        # 품질 점수가 없는 경우 계산
        for info in tqdm(image_infos, desc="품질 점수 계산"):
            if info.quality_score == 0.0:
                info.quality_score = self.quality_analyzer.analyze_image_quality(info.filepath)
        
        # 이상치 탐지
        outliers = self.quality_analyzer.detect_outliers(image_infos)
        
        # 필터링 적용
        filtered_infos = []
        filtered_count = 0
        
        for i, info in enumerate(image_infos):
            # 품질 점수 및 이상치 여부 확인
            if info.quality_score >= threshold and not outliers[i]:
                filtered_infos.append(info)
            else:
                filtered_count += 1
                self.logger.debug(f"필터링됨: {info.filename} (품질: {info.quality_score:.3f}, 이상치: {outliers[i]})")
        
        self.logger.info(f"품질 필터링 완료: {len(filtered_infos)}개 유지, {filtered_count}개 제거")
        
        return filtered_infos
    
    def _create_stratified_splits(self, image_infos: List[ImageInfo], dataset_type: DatasetType) -> Dict[str, List[ImageInfo]]:
        """계층화 분할 생성"""
        self.logger.info("계층화 데이터 분할 시작...")
        
        # 분할을 위한 레이블 생성
        stratify_labels = []
        for info in image_infos:
            if info.class_distribution:
                # 가장 많은 객체를 가진 클래스를 대표 클래스로 사용
                dominant_class = max(info.class_distribution, key=info.class_distribution.get)
                stratify_labels.append(dominant_class)
            else:
                stratify_labels.append('no_label')
        
        # 클래스 분포 확인
        class_counts = Counter(stratify_labels)
        self.logger.info(f"클래스 분포: {dict(class_counts)}")
        
        # 최소 클래스 샘플 수 확인
        min_samples = min(class_counts.values()) if class_counts else 0
        
        try:
            if min_samples >= 3:  # 계층화 분할 가능
                # StratifiedShuffleSplit 사용
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=CONFIG.data.val_ratio + CONFIG.data.test_ratio,
                    random_state=CONFIG.data.random_seed
                )
                
                train_idx, temp_idx = next(splitter.split(range(len(image_infos)), stratify_labels))
                
                # 검증/테스트 분할
                temp_labels = [stratify_labels[i] for i in temp_idx]
                temp_class_counts = Counter(temp_labels)
                
                if min(temp_class_counts.values()) >= 2:
                    val_ratio = CONFIG.data.val_ratio / (CONFIG.data.val_ratio + CONFIG.data.test_ratio)
                    
                    val_splitter = StratifiedShuffleSplit(
                        n_splits=1,
                        test_size=(1 - val_ratio),
                        random_state=CONFIG.data.random_seed
                    )
                    
                    val_idx_temp, test_idx_temp = next(val_splitter.split(temp_idx, temp_labels))
                    val_idx = [temp_idx[i] for i in val_idx_temp]
                    test_idx = [temp_idx[i] for i in test_idx_temp]
                else:
                    # 단순 분할
                    mid_point = len(temp_idx) // 2
                    val_idx = temp_idx[:mid_point]
                    test_idx = temp_idx[mid_point:]
                
                self.logger.info("계층화 분할 성공")
                
            else:
                raise ValueError("샘플 수 부족으로 계층화 불가능")
                
        except Exception as e:
            self.logger.warning(f"계층화 분할 실패, 랜덤 분할 사용: {e}")
            
            # 랜덤 분할
            indices = list(range(len(image_infos)))
            random.Random(CONFIG.data.random_seed).shuffle(indices)
            
            n_train = int(len(indices) * CONFIG.data.train_ratio)
            n_val = int(len(indices) * CONFIG.data.val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
        
        # 결과 구성
        splits = {
            'train': [image_infos[i] for i in train_idx],
            'val': [image_infos[i] for i in val_idx],
            'test': [image_infos[i] for i in test_idx]
        }
        
        # 분할 결과 로그
        for split_name, split_data in splits.items():
            class_dist = defaultdict(int)
            for info in split_data:
                for class_name, count in info.class_distribution.items():
                    class_dist[class_name] += count
            
            self.logger.info(f"{split_name}: {len(split_data)}개 이미지, 클래스 분포: {dict(class_dist)}")
        
        return splits
    
    def _setup_output_structure(self, output_path: Path):
        """출력 디렉토리 구조 생성"""
        # 기존 디렉토리 제거 (있는 경우)
        if output_path.exists():
            shutil.rmtree(output_path)
        
        # YOLO 형식 디렉토리 구조 생성
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def _copy_and_augment_data(self, splits: Dict[str, List[ImageInfo]], 
                             output_path: Path, dataset_type: DatasetType) -> ProcessingStats:
        """데이터 복사 및 증강"""
        stats = ProcessingStats()
        stats.original_images = sum(len(split) for split in splits.values())
        
        self.logger.info(f"데이터 복사 및 증강 시작: {stats.original_images}개 이미지")
        
        total_processed = 0
        total_augmented = 0
        class_counter = Counter()
        
        for split_name, image_infos in splits.items():
            self.logger.info(f"{split_name} 처리 중: {len(image_infos)}개 이미지")
            
            images_dir = output_path / 'images' / split_name
            labels_dir = output_path / 'labels' / split_name
            
            # 증강 설정 (훈련 데이터만)
            apply_augmentation = (split_name == 'train')
            augmentation_factor = self._get_augmentation_factor(dataset_type, split_name)
            
            for info in tqdm(image_infos, desc=f"복사 중 ({split_name})"):
                try:
                    # 원본 이미지 복사
                    self._copy_single_image(info, images_dir, labels_dir)
                    total_processed += 1
                    
                    # 클래스 분포 업데이트
                    for class_name, count in info.class_distribution.items():
                        class_counter[class_name] += count
                    
                    # 데이터 증강 (훈련 데이터만)
                    if apply_augmentation and augmentation_factor > 1:
                        augmented_count = self._augment_single_image(
                            info, images_dir, labels_dir, 
                            augmentation_factor - 1, dataset_type
                        )
                        total_augmented += augmented_count
                        
                        # 증강된 이미지의 클래스 분포도 업데이트
                        for class_name, count in info.class_distribution.items():
                            class_counter[class_name] += count * augmented_count
                    
                except Exception as e:
                    self.logger.error(f"이미지 처리 실패 {info.filename}: {e}")
                    continue
        
        # 통계 업데이트
        stats.processed_images = total_processed
        stats.augmented_images = total_augmented
        stats.total_objects = sum(class_counter.values())
        stats.class_distribution = dict(class_counter)
        
        self.logger.info(f"데이터 복사 완료: {total_processed}개 처리, {total_augmented}개 증강")
        
        return stats
    
    def _get_augmentation_factor(self, dataset_type: DatasetType, split_name: str) -> int:
        """데이터셋별 증강 배수 결정"""
        if split_name != 'train':
            return 1
        
        # 데이터셋 크기에 따른 증강 배수
        augmentation_factors = {
            DatasetType.GREENHOUSE_MULTI: 2,   # 569개 → 적당한 증강
            DatasetType.GREENHOUSE_SINGLE: 3,  # 358개 → 더 많은 증강  
            DatasetType.GROWTH_TIF: 1          # 1356개 → 증강 없음 (불균형 해결이 우선)
        }
        
        return augmentation_factors.get(dataset_type, 2)
    
    def _copy_single_image(self, info: ImageInfo, images_dir: Path, labels_dir: Path):
        """단일 이미지 및 라벨 복사"""
        source_img_path = Path(info.filepath)
        
        # 이미지 복사
        target_img_path = images_dir / source_img_path.name
        shutil.copy2(source_img_path, target_img_path)
        
        # 라벨 복사 (있는 경우)
        source_label_path = source_img_path.parent.parent / 'labels' / source_img_path.parent.name / f"{source_img_path.stem}.txt"
        
        if source_label_path.exists():
            target_label_path = labels_dir / f"{source_img_path.stem}.txt"
            shutil.copy2(source_label_path, target_label_path)
    
    def _augment_single_image(self, info: ImageInfo, images_dir: Path, 
                            labels_dir: Path, augment_count: int, dataset_type: DatasetType) -> int:
        """단일 이미지 데이터 증강"""
        if augment_count <= 0:
            return 0
        
        source_img_path = Path(info.filepath)
        source_label_path = source_img_path.parent.parent / 'labels' / source_img_path.parent.name / f"{source_img_path.stem}.txt"
        
        if not source_label_path.exists():
            return 0  # 라벨이 없으면 증강하지 않음
        
        try:
            # 이미지 로드
            if source_img_path.suffix.lower() in ['.tif', '.tiff']:
                with rasterio.open(source_img_path) as src:
                    image = src.read()
                    if len(image.shape) == 3:
                        image = np.transpose(image, (1, 2, 0))
                    if image.shape[-1] > 3:
                        image = image[:, :, :3]
                    image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = cv2.imread(str(source_img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 라벨 로드
            bboxes, class_labels = self._load_yolo_labels(source_label_path)
            
            if not bboxes:
                return 0
            
            # 증강 변환 가져오기
            augmenter = DataAugmentation(dataset_type)
            transform = augmenter.get_training_transforms(image_size=max(image.shape[:2]))
            
            successful_augmentations = 0
            
            for i in range(augment_count):
                try:
                    # 증강 적용
                    augmented = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    if not aug_bboxes:  # 박스가 손실된 경우 스킵
                        continue
                    
                    # 증강된 이미지 저장
                    aug_img_name = f"{source_img_path.stem}_aug{i+1}{source_img_path.suffix}"
                    aug_img_path = images_dir / aug_img_name
                    
                    # 텐서를 numpy로 변환
                    if hasattr(aug_image, 'numpy'):
                        aug_image = aug_image.numpy()
                    
                    # 정규화 해제 및 저장
                    if aug_image.max() <= 1.0:
                        aug_image = (aug_image * 255).astype(np.uint8)
                    
                    aug_image_pil = Image.fromarray(aug_image.transpose(1, 2, 0))
                    aug_image_pil.save(aug_img_path)
                    
                    # 증강된 라벨 저장
                    aug_label_path = labels_dir / f"{source_img_path.stem}_aug{i+1}.txt"
                    self._save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)
                    
                    successful_augmentations += 1
                    
                except Exception as e:
                    self.logger.debug(f"증강 실패 {source_img_path.name} #{i+1}: {e}")
                    continue
            
            return successful_augmentations
            
        except Exception as e:
            self.logger.error(f"이미지 증강 실패 {source_img_path.name}: {e}")
            return 0
    
    def _load_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """YOLO 형식 라벨 로드"""
        bboxes = []
        class_labels = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        except Exception as e:
            self.logger.error(f"라벨 로드 실패 {label_path}: {e}")
        
        return bboxes, class_labels
    
    def _save_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List):
        """YOLO 형식 라벨 저장"""
        try:
            with open(label_path, 'w') as f:
                for bbox, class_label in zip(bboxes, class_labels):
                    f.write(f"{class_label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        except Exception as e:
            self.logger.error(f"라벨 저장 실패 {label_path}: {e}")
    
    def _generate_dataset_files(self, splits: Dict[str, List[ImageInfo]], 
                              output_path: Path, dataset_type: DatasetType, 
                              stats: ProcessingStats):
        """데이터셋 YAML 및 메타데이터 파일 생성"""
        
        # dataset.yaml 생성
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
            'processor': 'OptimizedDataProcessor',
            
            # 통계
            'dataset_stats': {
                'original_images': stats.original_images,
                'processed_images': stats.processed_images,
                'augmented_images': stats.augmented_images,
                'total_images': stats.processed_images + stats.augmented_images,
                'total_objects': stats.total_objects,
                'train_images': len(splits.get('train', [])),
                'val_images': len(splits.get('val', [])),
                'test_images': len(splits.get('test', [])),
                'class_distribution': stats.class_distribution,
                'processing_time_seconds': stats.processing_time
            }
        }
        
        yaml_path = output_path / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        # 상세 통계 JSON 저장
        stats_path = output_path / 'processing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"데이터셋 파일 생성 완료: {yaml_path}")
    
    def _save_processing_report(self, all_results: Dict[str, ProcessingStats]):
        """전체 처리 리포트 저장"""
        report_path = self.output_dir / 'processing_report.json'
        
        # 통계 변환
        report_data = {}
        for dataset_name, stats in all_results.items():
            report_data[dataset_name] = asdict(stats)
        
        # 전체 요약 추가
        total_original = sum(stats.original_images for stats in all_results.values())
        total_processed = sum(stats.processed_images for stats in all_results.values())
        total_augmented = sum(stats.augmented_images for stats in all_results.values())
        
        report_data['summary'] = {
            'total_datasets': len(all_results),
            'total_original_images': total_original,
            'total_processed_images': total_processed,
            'total_augmented_images': total_augmented,
            'total_final_images': total_processed + total_augmented,
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'processor_version': CONFIG.version
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"전체 처리 리포트 저장: {report_path}")


class BalancedDatasetCreator:
    """클래스 불균형 해결 전용 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_balanced_dataset(self, image_infos: List[ImageInfo], 
                              dataset_type: DatasetType, 
                              dataset_stats: DatasetStats) -> List[ImageInfo]:
        """클래스 불균형 해결된 데이터셋 생성"""
        
        if dataset_type != DatasetType.GROWTH_TIF:
            # growth_tif_dataset이 아니면 원본 그대로 반환
            return image_infos
        
        self.logger.info("클래스 불균형 해결 시작 (growth_tif_dataset)")
        
        # 클래스별 이미지 분류
        class_images = defaultdict(list)
        for info in image_infos:
            for class_name in info.class_distribution.keys():
                class_images[class_name].append(info)
        
        # 현재 클래스 분포 분석
        class_counts = {class_name: len(images) for class_name, images in class_images.items()}
        self.logger.info(f"현재 클래스 분포: {class_counts}")
        
        # 타겟 클래스 개수 계산 (중간값 기준)
        sorted_counts = sorted(class_counts.values())
        target_count = sorted_counts[len(sorted_counts) // 2]  # 중앙값 사용
        
        self.logger.info(f"타겟 클래스 개수: {target_count}")
        
        # 균형잡힌 데이터셋 생성
        balanced_infos = []
        
        for class_name, images in class_images.items():
            current_count = len(images)
            
            if current_count <= target_count:
                # 부족한 클래스: 모든 이미지 포함
                balanced_infos.extend(images)
                self.logger.info(f"{class_name}: {current_count}개 모두 포함")
            else:
                # 과다한 클래스: 다운샘플링
                sampled_images = random.sample(images, target_count)
                balanced_infos.extend(sampled_images)
                self.logger.info(f"{class_name}: {current_count}개 → {target_count}개 다운샘플링")
        
        # 결과 분석
        final_class_counts = defaultdict(int)
        for info in balanced_infos:
            for class_name in info.class_distribution.keys():
                final_class_counts[class_name] += 1
        
        self.logger.info(f"균형 조정 후 클래스 분포: {dict(final_class_counts)}")
        self.logger.info(f"총 이미지 개수: {len(image_infos)} → {len(balanced_infos)}")
        
        return balanced_infos


class AdvancedDataAugmentation:
    """농업 특화 고급 데이터 증강 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_agricultural_transforms(self, dataset_type: DatasetType, image_size: int = 640) -> A.Compose:
        """농업 특화 데이터 증강 변환"""
        
        # 데이터셋별 특화 증강
        if dataset_type == DatasetType.GROWTH_TIF:
            # TIF 데이터: 계절 변화 시뮬레이션
            transforms = self._get_seasonal_transforms(image_size)
        elif dataset_type in [DatasetType.GREENHOUSE_MULTI, DatasetType.GREENHOUSE_SINGLE]:
            # 온실 데이터: 조명 및 구조물 변화
            transforms = self._get_greenhouse_transforms(image_size)
        else:
            # 기본 변환
            transforms = self._get_basic_transforms(image_size)
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _get_seasonal_transforms(self, image_size: int) -> List:
        """계절 변화 시뮬레이션 변환"""
        return [
            # 크기 조정
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, value=0),
            
            # 계절별 색상 변화
            A.OneOf([
                # 봄: 신선한 녹색
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                # 여름: 강한 햇빛
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                # 가을: 갈색/황색 톤
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=1.0),
                # 겨울: 어두운 톤
                A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=0.1, p=1.0),
            ], p=0.8),
            
            # 기상 조건 시뮬레이션
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),  # 안개
                A.RandomRain(blur_value=2, brightness_coefficient=0.9, p=0.3),  # 비
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.2),  # 햇빛
            ], p=0.4),
            
            # 기하학적 변환 (보수적)
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT, p=0.7
            ),
            
            # 플립 (농업 환경에서 자연스러움)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
        ]
    
    def _get_greenhouse_transforms(self, image_size: int) -> List:
        """온실 특화 변환"""
        return [
            # 크기 조정
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, value=0),
            
            # 온실 조명 변화
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.8),
            
            # 색온도 변화 (LED 조명 변화)
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.7),
            
            # 유리/플라스틱 효과
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=0.2),
            ], p=0.3),
            
            # 기하학적 변환
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, p=0.8
            ),
            
            # 플립
            A.HorizontalFlip(p=0.5),
        ]
    
    def _get_basic_transforms(self, image_size: int) -> List:
        """기본 변환"""
        return [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, value=0),
            
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT, p=0.7
            ),
            
            A.HorizontalFlip(p=0.5),
        ]


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Nong-View 최적화된 데이터 전처리")
    
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=[dt.value for dt in DatasetType],
        default=[dt.value for dt in DatasetType],
        help='처리할 데이터셋 선택'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='출력 디렉토리 (기본값: CONFIG에서 가져옴)'
    )
    
    parser.add_argument(
        '--disable-quality-filter',
        action='store_true',
        help='품질 필터링 비활성화'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 로그 출력'
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 데이터셋 타입 변환
    target_datasets = [DatasetType(name) for name in args.datasets]
    
    # 프로세서 초기화
    processor = OptimizedDataProcessor(
        output_dir=args.output_dir,
        enable_quality_filter=not args.disable_quality_filter
    )
    
    # 처리 시작
    logger.info("=" * 60)
    logger.info("🏆 Nong-View 최적화된 데이터 전처리 시작")
    logger.info("=" * 60)
    
    try:
        results = processor.process_all_datasets(target_datasets)
        
        logger.info("=" * 60)
        logger.info("✅ 모든 데이터셋 처리 완료!")
        
        # 결과 요약
        for dataset_name, stats in results.items():
            logger.info(f"📊 {dataset_name}:")
            logger.info(f"  - 원본: {stats.original_images}개")
            logger.info(f"  - 처리: {stats.processed_images}개")
            logger.info(f"  - 증강: {stats.augmented_images}개")
            logger.info(f"  - 총 객체: {stats.total_objects}개")
            logger.info(f"  - 처리 시간: {stats.processing_time:.2f}초")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 처리 실패: {e}")
        raise


if __name__ == "__main__":
    main()