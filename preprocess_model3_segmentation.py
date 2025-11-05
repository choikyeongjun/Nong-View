#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model3 Greenhouse - YOLOv11-seg 전용 전처리
Segmentation (polygon) 형식 지원

작성: Claude Sonnet
날짜: 2025-11-04
"""

import os
import json
import yaml
import shutil
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # INFO -> DEBUG로 변경하여 모든 디버그 정보 출력
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================== 설정 ==================

@dataclass
class SegConfig:
    """Segmentation 전용 설정"""
    source_dir: str = r"D:\Nong-View\model3_greenhouse"
    output_dir: str = r"D:\model3_greenhouse_seg_balanced_5000"

    # 원본 클래스 ID: 9=단동, 10=연동
    original_class_ids: List[int] = field(default_factory=lambda: [9, 10])
    # YOLO 클래스명 (0=단동, 1=연동으로 재매핑)
    classes: List[str] = field(default_factory=lambda: ['Greenhouse_single', 'Greenhouse_multi'])
    nc: int = 2

    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    enable_augmentation: bool = True
    augmentation_factor: int = 3
    balance_classes: bool = True  # 클래스 균형 맞추기
    target_train_per_class: int = 5000  # 각 클래스당 train 목표

    random_seed: int = 42


@dataclass
class ImageInfo:
    """이미지 정보"""
    filepath: Path
    filename: str
    classes: List[int] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    dominant_class: int = 0


@dataclass
class ProcessingStats:
    """처리 통계"""
    original_images: int = 0
    processed_images: int = 0
    augmented_images: int = 0
    total_objects: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0


# ================== 증강 클래스 ==================

class SegmentationAugmenter:
    """Segmentation 전용 증강 (다양한 기법)"""
    
    def __init__(self):
        self.aug_methods = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_90,
            self.rotate_180,
            self.brightness_contrast,
            self.add_noise,
            self.flip_brightness,
            self.rotate_contrast
        ]

    def augment_image_and_polygons(self, image: np.ndarray, polygons: List[List[float]],
                                   class_labels: List[int], method_idx: int = 0) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """이미지와 polygon 증강"""
        method = self.aug_methods[method_idx % len(self.aug_methods)]
        return method(image, polygons, class_labels)
    
    def horizontal_flip(self, image: np.ndarray, polygons: List[List[float]],
                       class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """좌우 반전"""
        try:
            aug_image = cv2.flip(image, 1)
            aug_polygons = []

            for poly in polygons:
                aug_poly = []
                for i in range(0, len(poly), 2):
                    x = poly[i]
                    y = poly[i + 1]
                    aug_poly.append(1.0 - x)
                    aug_poly.append(y)
                aug_polygons.append(aug_poly)

            return aug_image, aug_polygons, class_labels.copy()
        except Exception as e:
            logger.error(f"증강 에러: {e}")
            return image, polygons, class_labels
    
    def vertical_flip(self, image: np.ndarray, polygons: List[List[float]],
                     class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """상하 반전"""
        try:
            aug_image = cv2.flip(image, 0)
            aug_polygons = []

            for poly in polygons:
                aug_poly = []
                for i in range(0, len(poly), 2):
                    x = poly[i]
                    y = poly[i + 1]
                    aug_poly.append(x)
                    aug_poly.append(1.0 - y)
                aug_polygons.append(aug_poly)

            return aug_image, aug_polygons, class_labels.copy()
        except Exception as e:
            return image, polygons, class_labels
    
    def rotate_90(self, image: np.ndarray, polygons: List[List[float]],
                 class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """90도 회전"""
        try:
            aug_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            aug_polygons = []

            for poly in polygons:
                aug_poly = []
                for i in range(0, len(poly), 2):
                    x = poly[i]
                    y = poly[i + 1]
                    aug_poly.append(y)
                    aug_poly.append(1.0 - x)
                aug_polygons.append(aug_poly)

            return aug_image, aug_polygons, class_labels.copy()
        except Exception as e:
            return image, polygons, class_labels
    
    def rotate_180(self, image: np.ndarray, polygons: List[List[float]],
                  class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """180도 회전"""
        try:
            aug_image = cv2.rotate(image, cv2.ROTATE_180)
            aug_polygons = []

            for poly in polygons:
                aug_poly = []
                for i in range(0, len(poly), 2):
                    x = poly[i]
                    y = poly[i + 1]
                    aug_poly.append(1.0 - x)
                    aug_poly.append(1.0 - y)
                aug_polygons.append(aug_poly)

            return aug_image, aug_polygons, class_labels.copy()
        except Exception as e:
            return image, polygons, class_labels
    
    def brightness_contrast(self, image: np.ndarray, polygons: List[List[float]],
                           class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """밝기/대비 조정"""
        try:
            aug_image = image.copy()
            
            brightness = random.uniform(0.7, 1.3)
            aug_image = np.clip(aug_image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            
            contrast = random.uniform(0.8, 1.2)
            aug_image = np.clip((aug_image.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)

            return aug_image, [p.copy() for p in polygons], class_labels.copy()
        except Exception as e:
            return image, polygons, class_labels
    
    def add_noise(self, image: np.ndarray, polygons: List[List[float]],
                 class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """노이즈 추가"""
        try:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            aug_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return aug_image, [p.copy() for p in polygons], class_labels.copy()
        except Exception as e:
            return image, polygons, class_labels
    
    def flip_brightness(self, image: np.ndarray, polygons: List[List[float]],
                       class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """좌우반전 + 밝기"""
        aug_img, aug_polys, aug_labels = self.horizontal_flip(image, polygons, class_labels)
        brightness = random.uniform(0.8, 1.2)
        aug_img = np.clip(aug_img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        return aug_img, aug_polys, aug_labels
    
    def rotate_contrast(self, image: np.ndarray, polygons: List[List[float]],
                       class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """회전 + 대비"""
        aug_img, aug_polys, aug_labels = self.rotate_180(image, polygons, class_labels)
        contrast = random.uniform(0.9, 1.1)
        aug_img = np.clip((aug_img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
        return aug_img, aug_polys, aug_labels


# ================== 메인 전처리 클래스 ==================

class SegmentationPreprocessor:
    """YOLOv11-seg 전용 전처리"""

    def __init__(self, config: SegConfig):
        self.config = config
        self.augmenter = SegmentationAugmenter()

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        logger.info("=" * 60)
        logger.info("YOLOv11-seg Segmentation 전처리 시작")
        logger.info("=" * 60)
        logger.info(f"소스: {config.source_dir}")
        logger.info(f"출력: {config.output_dir}")
        logger.info(f"증강: {config.augmentation_factor}배")

    def run(self) -> ProcessingStats:
        """전체 프로세스 실행"""
        start_time = time.time()

        logger.info("\n[1/5] 데이터 수집 중...")
        image_infos = self._collect_images()
        logger.info(f"✓ {len(image_infos)}개 이미지 수집")

        logger.info("\n[2/5] 계층화 분할 중...")
        splits = self._stratified_split(image_infos)
        logger.info(f"✓ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        logger.info("\n[3/5] 출력 디렉토리 생성 중...")
        output_path = Path(self.config.output_dir)
        self._setup_output_structure(output_path)
        logger.info(f"✓ {output_path}")

        logger.info("\n[4/5] 데이터 복사 및 증강 중...")
        stats = self._copy_and_augment(splits, output_path)
        logger.info(f"✓ 처리: {stats.processed_images}개, 증강: {stats.augmented_images}개")

        logger.info("\n[5/5] 메타데이터 생성 중...")
        self._create_yaml(splits, output_path, stats)
        self._save_stats(stats, output_path)
        logger.info("✓ 완료")

        stats.processing_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("✅ 전처리 완료!")
        logger.info("=" * 60)
        self._print_summary(stats)

        return stats

    def _collect_images(self) -> List[ImageInfo]:
        """이미지 수집 (images 폴더에서 직접)"""
        source_path = Path(self.config.source_dir)
        all_images = []

        # images 폴더에서 직접 수집 (train/val/test 분할 전)
        images_dir = source_path / 'images'
        labels_dir = source_path / 'labels'

        if not images_dir.exists():
            logger.error(f"이미지 디렉토리 없음: {images_dir}")
            return all_images
        
        if not labels_dir.exists():
            logger.error(f"라벨 디렉토리 없음: {labels_dir}")
            return all_images

        # 모든 TIF 이미지 수집
        for img_path in list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff')):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                classes, class_dist = self._parse_seg_label(label_path)

                if classes:
                    # 주요 클래스 찾기
                    class_counts = Counter(classes)
                    dominant_class = class_counts.most_common(1)[0][0]

                    info = ImageInfo(
                        filepath=img_path,
                        filename=img_path.name,
                        classes=classes,
                        class_distribution=class_dist,
                        dominant_class=dominant_class
                    )
                    all_images.append(info)

        return all_images

    def _remap_class_id(self, original_id: int) -> int:
        """원본 클래스 ID를 YOLO ID로 재매핑 (9->0, 10->1)"""
        if original_id in self.config.original_class_ids:
            return self.config.original_class_ids.index(original_id)
        return original_id
    
    def _parse_seg_label(self, label_path: Path) -> Tuple[List[int], Dict[str, int]]:
        """Segmentation 라벨 파싱 (클래스 ID 재매핑 포함)"""
        classes = []
        class_dist = defaultdict(int)

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        original_class_id = int(parts[0])
                        
                        # 클래스 ID 재매핑 (9->0, 10->1)
                        class_id = self._remap_class_id(original_class_id)
                        classes.append(class_id)

                        if 0 <= class_id < len(self.config.classes):
                            class_name = self.config.classes[class_id]
                        else:
                            class_name = f"Class_{class_id}"

                        class_dist[class_name] += 1
        except Exception as e:
            logger.error(f"라벨 파싱 실패 {label_path.name}: {e}")

        return classes, dict(class_dist)

    def _stratified_split(self, image_infos: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
        """계층화 분할"""
        class_groups = defaultdict(list)
        for info in image_infos:
            class_groups[info.dominant_class].append(info)

        train_images = []
        val_images = []
        test_images = []

        for class_id, class_images in class_groups.items():
            random.shuffle(class_images)

            n = len(class_images)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)

            train_images.extend(class_images[:n_train])
            val_images.extend(class_images[n_train:n_train + n_val])
            test_images.extend(class_images[n_train + n_val:])

            if 0 <= class_id < len(self.config.classes):
                class_name = self.config.classes[class_id]
            else:
                class_name = f"Class_{class_id}"

            logger.info(f"  {class_name}: Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")

        random.shuffle(train_images)
        random.shuffle(val_images)
        random.shuffle(test_images)

        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

    def _setup_output_structure(self, output_path: Path):
        """출력 디렉토리 생성"""
        if output_path.exists():
            logger.warning(f"기존 디렉토리 삭제: {output_path}")
            shutil.rmtree(output_path)

        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _calculate_class_augmentation_factors(self, train_images: List[ImageInfo]) -> Dict[int, int]:
        """클래스별 증강 배수 계산"""
        # 클래스별 train 이미지 수 계산
        class_counts = Counter()
        for info in train_images:
            class_counts[info.dominant_class] += 1
        
        logger.info(f"\nTrain 데이터 클래스 분포 (재매핑 후):")
        for class_id, count in sorted(class_counts.items()):
            if 0 <= class_id < len(self.config.classes):
                class_name = self.config.classes[class_id]
            else:
                class_name = f"Class_{class_id}"
            logger.info(f"  - Class {class_id} ({class_name}): {count}개")
        
        if not class_counts:
            logger.warning("클래스 카운트가 비어있습니다!")
            return {}
        
        # 클래스별 증강 배수 계산
        aug_factors = {}
        
        logger.info("\n클래스별 증강 전략:")
        for class_id in sorted(class_counts.keys()):
            if 0 <= class_id < len(self.config.classes):
                class_name = self.config.classes[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            current_count = class_counts[class_id]
            target_count = self.config.target_train_per_class
            
            # 필요한 증강 개수 = 목표 - 현재
            needed = max(0, target_count - current_count)
            aug_factors[class_id] = needed
            
            logger.info(f"  - {class_name:20s}: {current_count:4d}개 → (+{needed:4d}) → {current_count + needed:4d}개")
        
        logger.info(f"\n증강 factors: {aug_factors}")
        return aug_factors

    def _copy_and_augment(self, splits: Dict[str, List[ImageInfo]], output_path: Path) -> ProcessingStats:
        """데이터 복사 및 클래스별 균형 증강"""
        stats = ProcessingStats()
        stats.original_images = sum(len(images) for images in splits.values())

        total_processed = 0
        total_augmented = 0
        class_counter = Counter()
        
        # 클래스별 증강 배수 계산 (balance_classes=True일 때만)
        aug_factors = {}
        if self.config.enable_augmentation and self.config.balance_classes:
            aug_factors = self._calculate_class_augmentation_factors(splits['train'])

        for split_name, image_infos in splits.items():
            images_dir = output_path / 'images' / split_name
            labels_dir = output_path / 'labels' / split_name

            apply_augmentation = (split_name == 'train' and self.config.enable_augmentation)

            # Val/Test는 간단히 복사만
            if not apply_augmentation:
                for info in tqdm(image_infos, desc=f"{split_name} 처리", ncols=70):
                    self._copy_single_file(info, images_dir, labels_dir)
                    total_processed += 1
                    
                    for class_name, count in info.class_distribution.items():
                        class_counter[class_name] += count
                continue
            
            # Train은 클래스별로 처리
            if self.config.balance_classes and aug_factors:
                # 클래스별로 분리
                class_images = defaultdict(list)
                for info in image_infos:
                    class_images[info.dominant_class].append(info)
                
                # 클래스별로 복사 및 증강
                for class_id in sorted(class_images.keys()):
                    class_name = self.config.classes[class_id] if class_id < len(self.config.classes) else f"Class_{class_id}"
                    class_imgs = class_images[class_id]
                    
                    logger.info(f"\n{class_name} 처리 중...")
                    
                    # 원본 복사
                    for info in tqdm(class_imgs, desc=f"  원본 복사", ncols=70, 
                                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
                        self._copy_single_file(info, images_dir, labels_dir)
                        total_processed += 1
                        
                        for cls_name, count in info.class_distribution.items():
                            class_counter[cls_name] += count
                    
                    # 증강
                    needed = aug_factors.get(class_id, 0)
                    if needed > 0:
                        augmented = self._augment_class_batch(
                            class_imgs, images_dir, labels_dir, class_id, needed
                        )
                        total_augmented += augmented
                        
                        # 증강 데이터 클래스 분포 업데이트
                        for info in class_imgs[:1]:  # 대표 이미지의 분포 사용
                            for cls_name, count in info.class_distribution.items():
                                class_counter[cls_name] += count * augmented
            
            else:
                # 기존 방식 (모든 이미지에 동일 배수)
                for info in tqdm(image_infos, desc=f"{split_name} 처리", ncols=70):
                    self._copy_single_file(info, images_dir, labels_dir)
                    total_processed += 1

                    for class_name, count in info.class_distribution.items():
                        class_counter[class_name] += count

                    aug_count = self._augment_single(
                        info, images_dir, labels_dir, self.config.augmentation_factor - 1, 0
                    )
                    total_augmented += aug_count
                    
                    for class_name, count in info.class_distribution.items():
                        class_counter[class_name] += count * aug_count

        stats.processed_images = total_processed
        stats.augmented_images = total_augmented
        stats.total_objects = sum(class_counter.values())
        stats.class_distribution = dict(class_counter)

        return stats
    
    def _augment_class_batch(self, class_images: List[ImageInfo], images_dir: Path,
                            labels_dir: Path, class_id: int, needed: int) -> int:
        """특정 클래스를 배치로 증강"""
        augmented = 0
        class_name = self.config.classes[class_id] if class_id < len(self.config.classes) else f"Class_{class_id}"
        
        pbar = tqdm(total=needed, desc=f"  증강 생성", ncols=70,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        
        method_idx = 0
        while augmented < needed:
            source_img = random.choice(class_images)
            
            try:
                if self._augment_single(source_img, images_dir, labels_dir, 1, method_idx, 
                                       aug_prefix=f"{class_name[:6]}_aug{augmented:04d}"):
                    augmented += 1
                    pbar.update(1)
                
                method_idx += 1
            except:
                continue
        
        pbar.close()
        return augmented

    def _copy_single_file(self, info: ImageInfo, images_dir: Path, labels_dir: Path):
        """단일 파일 복사"""
        # 이미지 복사
        shutil.copy2(info.filepath, images_dir / info.filename)

        # 라벨 복사 - 직접 경로
        # info.filepath: .../model3_greenhouse/images/xxx.png
        # source_label: .../model3_greenhouse/labels/xxx.txt
        source_data_root = info.filepath.parent.parent  # .../model3_greenhouse
        source_label = source_data_root / 'labels' / f"{info.filepath.stem}.txt"
        
        if source_label.exists():
            shutil.copy2(source_label, labels_dir / f"{info.filepath.stem}.txt")
        else:
            logger.warning(f"라벨 파일 없음 (복사): {source_label}")

    def _augment_single(self, info: ImageInfo, images_dir: Path, labels_dir: Path, 
                       count: int, method_idx: int = 0, aug_prefix: str = None) -> int:
        """단일 이미지 증강 (Segmentation)"""
        if count <= 0:
            return 0

        # 라벨 경로 (직접)
        source_data_root = info.filepath.parent.parent  # .../model3_greenhouse
        source_label = source_data_root / 'labels' / f"{info.filepath.stem}.txt"

        if not source_label.exists():
            return 0

        try:
            # 이미지 로드
            image = cv2.imread(str(info.filepath))
            if image is None:
                return 0

            # Segmentation 라벨 로드
            polygons, class_labels = self._load_seg_labels(source_label)
            if not polygons:
                return 0

            successful = 0

            for i in range(count):
                try:
                    # 증강 적용 (다양한 기법)
                    aug_image, aug_polygons, aug_labels = self.augmenter.augment_image_and_polygons(
                        image, polygons, class_labels, method_idx + i
                    )

                    if not aug_polygons:
                        continue

                    # 파일명 결정 (PNG로 저장)
                    if aug_prefix:
                        aug_img_name = f"{aug_prefix}.png"
                        aug_label_name = f"{aug_prefix}.txt"
                    else:
                        aug_img_name = f"{info.filepath.stem}_aug{i+1}.png"
                        aug_label_name = f"{info.filepath.stem}_aug{i+1}.txt"
                    
                    # 증강 이미지 저장 (PNG로 변환)
                    aug_img_path = images_dir / aug_img_name
                    result = cv2.imwrite(str(aug_img_path), aug_image)
                    if not result:
                        continue

                    # 증강 라벨 저장
                    aug_label_path = labels_dir / aug_label_name
                    self._save_seg_labels(aug_label_path, aug_polygons, aug_labels)

                    # 저장 확인
                    if aug_img_path.exists() and aug_label_path.exists():
                        successful += 1
                    
                except Exception as e:
                    logger.debug(f"증강 실패: {e}")
                    continue

            return successful

        except Exception as e:
            return 0

    def _load_seg_labels(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Segmentation 라벨 로드 (클래스 ID 재매핑 포함)"""
        polygons = []
        class_labels = []

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        original_class_id = int(parts[0])
                        polygon = [float(x) for x in parts[1:]]
                        
                        # 클래스 ID 재매핑 (9->0, 10->1)
                        class_id = self._remap_class_id(original_class_id)

                        polygons.append(polygon)
                        class_labels.append(class_id)
        except Exception as e:
            logger.error(f"라벨 로드 실패 {label_path}: {e}")

        return polygons, class_labels

    def _save_seg_labels(self, label_path: Path, polygons: List[List[float]], class_labels: List[int]):
        """Segmentation 라벨 저장"""
        try:
            with open(label_path, 'w') as f:
                for polygon, class_id in zip(polygons, class_labels):
                    # class_id + polygon 좌표들
                    coords_str = ' '.join([f"{coord:.10f}" for coord in polygon])
                    f.write(f"{class_id} {coords_str}\n")
        except Exception as e:
            logger.error(f"라벨 저장 실패 {label_path}: {e}")

    def _create_yaml(self, splits: Dict[str, List[ImageInfo]], output_path: Path, stats: ProcessingStats):
        """YAML 파일 생성"""
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config.nc,
            'names': self.config.classes,

            'task': 'segment',  # Segmentation 명시

            'dataset_info': {
                'total': stats.original_images,
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test']),
                'processed': stats.processed_images,
                'augmented': stats.augmented_images
            },

            'preprocessing': {
                'method': 'balanced_class_augmentation' if self.config.balance_classes else 'stratified_split_segmentation',
                'augmentation': self.config.enable_augmentation,
                'augmentation_factor': self.config.augmentation_factor,
                'balance_classes': self.config.balance_classes,
                'target_train_per_class': self.config.target_train_per_class if self.config.balance_classes else None,
                'random_seed': self.config.random_seed
            }
        }

        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"YAML: {yaml_path}")

    def _save_stats(self, stats: ProcessingStats, output_path: Path):
        """통계 저장"""
        stats_path = output_path / 'processing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2, ensure_ascii=False)

        logger.info(f"통계: {stats_path}")

    def _print_summary(self, stats: ProcessingStats):
        """요약 출력"""
        logger.info(f"\n📊 처리 통계:")
        logger.info(f"  - 원본 이미지: {stats.original_images}개")
        logger.info(f"  - 처리된 이미지: {stats.processed_images}개")
        logger.info(f"  - 증강된 이미지: {stats.augmented_images}개")
        logger.info(f"  - 총 이미지: {stats.processed_images + stats.augmented_images}개")
        logger.info(f"  - 증강 비율: {(stats.augmented_images/stats.processed_images if stats.processed_images > 0 else 0):.2f}배")
        logger.info(f"  - 총 객체: {stats.total_objects}개")
        logger.info(f"  - 처리 시간: {stats.processing_time:.2f}초")

        logger.info(f"\n📊 클래스 분포:")
        for class_name, count in sorted(stats.class_distribution.items()):
            percentage = (count / stats.total_objects) * 100 if stats.total_objects > 0 else 0
            logger.info(f"  - {class_name}: {count}개 ({percentage:.1f}%)")

        logger.info(f"\n✨ 출력: {self.config.output_dir}")
        
        # 증강 확인
        if self.config.enable_augmentation and stats.augmented_images == 0:
            logger.warning(f"\n⚠️  경고: 증강이 활성화되었지만 증강된 이미지가 없습니다!")
            logger.warning(f"   - augmentation_factor: {self.config.augmentation_factor}")
            logger.warning(f"   - 증강 대상: train 데이터만")


# ================== 메인 함수 ==================

def main():
    """메인 실행"""
    config = SegConfig(
        source_dir=r"D:\Nong-View\model3_greenhouse",
        output_dir=r"D:\model3_greenhouse_seg_balanced_5000",
        original_class_ids=[9, 10],  # 원본: 9=단동, 10=연동
        classes=['Greenhouse_single', 'Greenhouse_multi'],  # YOLO: 0=단동, 1=연동
        nc=2,
        enable_augmentation=True,
        augmentation_factor=3,
        balance_classes=True,
        target_train_per_class=5000,
        random_seed=42
    )

    preprocessor = SegmentationPreprocessor(config)
    stats = preprocessor.run()

    logger.info("\n" + "=" * 60)
    logger.info(">> YOLOv11-seg 전처리 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
