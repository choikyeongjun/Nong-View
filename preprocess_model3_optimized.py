#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model3 Greenhouse 데이터셋 최적화 전처리
optimized_preprocessing.py 구조 기반

클래스:
- 0: Greenhouse_single (단동)
- 1: Greenhouse_multi (연동)

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
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================== 설정 클래스 ==================

@dataclass
class Model3Config:
    """Model3 Greenhouse 전용 설정"""
    # 경로
    source_dir: str = r"C:\Users\LX\Nong-View\model3_greenhouse"
    output_dir: str = r"C:\Users\LX\Nong-View\model3_greenhouse_best_processed"

    # 클래스 정보
    classes: List[str] = field(default_factory=lambda: ['Greenhouse_single', 'Greenhouse_multi'])
    nc: int = 2

    # 데이터 분할 비율
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 전처리 옵션
    enable_quality_filter: bool = True
    enable_augmentation: bool = True
    augmentation_factor: int = 2  # 훈련 데이터만

    # 품질 필터링 임계값
    quality_threshold: float = 0.4

    # 랜덤 시드
    random_seed: int = 42


# ================== 데이터 클래스 ==================

@dataclass
class ImageInfo:
    """이미지 정보 클래스"""
    filepath: Path
    filename: str
    width: int = 0
    height: int = 0
    classes: List[int] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    quality_score: float = 0.0
    dominant_class: int = -1


@dataclass
class ProcessingStats:
    """처리 통계 클래스"""
    original_images: int = 0
    processed_images: int = 0
    augmented_images: int = 0
    filtered_images: int = 0
    total_objects: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    processing_time: float = 0.0


# ================== 데이터 품질 분석기 ==================

class DataQualityAnalyzer:
    """데이터 품질 분석 클래스"""

    def analyze_image_quality(self, image_path: Path) -> float:
        """이미지 품질 점수 계산 (0.0 ~ 1.0)"""
        try:
            # 이미지 로드
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0

            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. 흐림 감지 (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 500.0, 1.0)  # 정규화

            # 2. 밝기 평가
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5

            # 3. 대비 평가
            contrast = gray.std()
            contrast_score = min(contrast / 64.0, 1.0)

            # 종합 점수 (가중 평균)
            quality_score = (
                blur_score * 0.5 +
                brightness_score * 0.3 +
                contrast_score * 0.2
            )

            return quality_score

        except Exception as e:
            logger.error(f"품질 분석 실패 {image_path.name}: {e}")
            return 0.5  # 중간 점수

    def detect_outliers(self, image_infos: List[ImageInfo]) -> List[bool]:
        """이상치 탐지 (간단한 IQR 방법)"""
        if len(image_infos) < 4:
            return [False] * len(image_infos)

        # 품질 점수 수집
        scores = [info.quality_score for info in image_infos]
        scores_sorted = sorted(scores)

        # IQR 계산
        q1_idx = len(scores_sorted) // 4
        q3_idx = (3 * len(scores_sorted)) // 4
        q1 = scores_sorted[q1_idx]
        q3 = scores_sorted[q3_idx]
        iqr = q3 - q1

        # 이상치 기준
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 이상치 탐지
        outliers = [
            score < lower_bound or score > upper_bound
            for score in scores
        ]

        return outliers


# ================== 데이터 증강 ==================

class GreenhouseDataAugmentation:
    """온실 특화 데이터 증강 (간단하고 확실하게)"""

    def augment_image(self, image: np.ndarray, bboxes: List, class_labels: List) -> Tuple[np.ndarray, List, List]:
        """이미지 증강 적용 - 간단하고 확실한 변환만 사용"""
        try:
            aug_image = image.copy()
            aug_bboxes = [bbox.copy() if isinstance(bbox, list) else list(bbox) for bbox in bboxes]
            aug_class_labels = class_labels.copy()

            h, w = aug_image.shape[:2]

            # 1. 좌우 반전 (항상 적용 - 가장 안전)
            aug_image = cv2.flip(aug_image, 1)
            # bbox x 좌표 반전
            for bbox in aug_bboxes:
                bbox[0] = 1.0 - bbox[0]

            # 2. 밝기 조정
            brightness_factor = random.uniform(0.85, 1.15)
            aug_image = np.clip(aug_image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

            # 3. 대비 조정
            contrast_factor = random.uniform(0.9, 1.1)
            aug_image = np.clip((aug_image.astype(np.float32) - 127.5) * contrast_factor + 127.5, 0, 255).astype(np.uint8)

            return aug_image, aug_bboxes, aug_class_labels

        except Exception as e:
            logger.error(f"증강 중 에러: {e}")
            # 실패 시 원본 반환
            return image, bboxes, class_labels


# ================== 메인 전처리 클래스 ==================

class Model3OptimizedPreprocessor:
    """Model3 Greenhouse 최적화 전처리 클래스"""

    def __init__(self, config: Model3Config = None):
        self.config = config if config else Model3Config()
        self.quality_analyzer = DataQualityAnalyzer()
        self.augmenter = GreenhouseDataAugmentation()

        # 랜덤 시드 설정
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        logger.info("=" * 60)
        logger.info("Model3 Greenhouse 최적화 전처리 초기화")
        logger.info("=" * 60)
        logger.info(f"소스: {self.config.source_dir}")
        logger.info(f"출력: {self.config.output_dir}")
        logger.info(f"품질 필터링: {self.config.enable_quality_filter}")
        logger.info(f"데이터 증강: {self.config.enable_augmentation}")

    def run(self) -> ProcessingStats:
        """전체 전처리 프로세스 실행"""
        start_time = time.time()

        logger.info("\n[1/6] 데이터 수집 중...")
        image_infos = self._collect_images()
        logger.info(f"✓ {len(image_infos)}개 이미지 수집 완료")

        logger.info("\n[2/6] 품질 분석 중...")
        if self.config.enable_quality_filter:
            image_infos = self._filter_by_quality(image_infos)
        logger.info(f"✓ {len(image_infos)}개 이미지 (품질 필터링 후)")

        logger.info("\n[3/6] 계층화 분할 중...")
        splits = self._stratified_split(image_infos)
        logger.info(f"✓ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        logger.info("\n[4/6] 출력 디렉토리 생성 중...")
        output_path = Path(self.config.output_dir)
        self._setup_output_structure(output_path)
        logger.info(f"✓ {output_path}")

        logger.info("\n[5/6] 데이터 복사 및 증강 중...")
        stats = self._copy_and_augment_data(splits, output_path)
        logger.info(f"✓ 처리: {stats.processed_images}개, 증강: {stats.augmented_images}개")

        logger.info("\n[6/6] 메타데이터 생성 중...")
        self._create_yaml_file(splits, output_path, stats)
        self._save_statistics(stats, output_path)
        logger.info("✓ 완료")

        stats.processing_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("✅ 전처리 완료!")
        logger.info("=" * 60)
        self._print_summary(stats)

        return stats

    def _collect_images(self) -> List[ImageInfo]:
        """이미지 수집"""
        source_path = Path(self.config.source_dir)
        all_images = []

        # train/val/test 폴더에서 수집
        for split in ['train', 'val', 'test']:
            images_dir = source_path / 'images' / split
            labels_dir = source_path / 'labels' / split

            if not images_dir.exists():
                logger.warning(f"디렉토리 없음: {images_dir}")
                continue

            # 이미지 파일 찾기
            for img_path in images_dir.glob('*.png'):
                label_path = labels_dir / f"{img_path.stem}.txt"

                if label_path.exists():
                    # 라벨 파싱
                    classes, class_dist = self._parse_label(label_path)

                    if classes:
                        # 이미지 크기 읽기
                        try:
                            img = Image.open(img_path)
                            width, height = img.size
                        except:
                            width, height = 0, 0

                        # 주요 클래스 찾기 (가장 많은 객체를 가진 클래스 ID)
                        if classes:
                            class_counts = Counter(classes)
                            dominant_class = class_counts.most_common(1)[0][0]
                        else:
                            dominant_class = 0

                        info = ImageInfo(
                            filepath=img_path,
                            filename=img_path.name,
                            width=width,
                            height=height,
                            classes=classes,
                            class_distribution=class_dist,
                            dominant_class=dominant_class
                        )
                        all_images.append(info)

        return all_images

    def _parse_label(self, label_path: Path) -> Tuple[List[int], Dict[str, int]]:
        """YOLO 라벨 파싱"""
        classes = []
        class_dist = defaultdict(int)

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        classes.append(class_id)

                        # 클래스 이름 가져오기 (안전하게)
                        if 0 <= class_id < len(self.config.classes):
                            class_name = self.config.classes[class_id]
                        else:
                            class_name = f"Class_{class_id}"

                        class_dist[class_name] += 1
        except Exception as e:
            logger.error(f"라벨 파싱 실패 {label_path.name}: {e}")

        return classes, dict(class_dist)

    def _filter_by_quality(self, image_infos: List[ImageInfo]) -> List[ImageInfo]:
        """품질 기반 필터링"""
        # 품질 점수 계산
        for info in tqdm(image_infos, desc="품질 분석"):
            info.quality_score = self.quality_analyzer.analyze_image_quality(info.filepath)

        # 이상치 탐지
        outliers = self.quality_analyzer.detect_outliers(image_infos)

        # 필터링
        filtered = []
        for i, info in enumerate(image_infos):
            if info.quality_score >= self.config.quality_threshold and not outliers[i]:
                filtered.append(info)

        logger.info(f"품질 필터링: {len(image_infos)} → {len(filtered)} ({len(image_infos) - len(filtered)}개 제거)")

        return filtered

    def _stratified_split(self, image_infos: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
        """계층화 분할"""
        # 클래스별 그룹화
        class_groups = defaultdict(list)
        for info in image_infos:
            class_groups[info.dominant_class].append(info)

        train_images = []
        val_images = []
        test_images = []

        # 각 클래스별로 분할
        for class_id, class_images in class_groups.items():
            random.shuffle(class_images)

            n = len(class_images)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)

            train_images.extend(class_images[:n_train])
            val_images.extend(class_images[n_train:n_train + n_val])
            test_images.extend(class_images[n_train + n_val:])

            # 클래스 이름 가져오기 (안전하게)
            if 0 <= class_id < len(self.config.classes):
                class_name = self.config.classes[class_id]
            else:
                class_name = f"Class_{class_id}"

            logger.info(f"  {class_name}: Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")

        # 최종 셔플
        random.shuffle(train_images)
        random.shuffle(val_images)
        random.shuffle(test_images)

        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

    def _setup_output_structure(self, output_path: Path):
        """출력 디렉토리 구조 생성"""
        if output_path.exists():
            logger.warning(f"기존 디렉토리 삭제: {output_path}")
            shutil.rmtree(output_path)

        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _copy_and_augment_data(self, splits: Dict[str, List[ImageInfo]], output_path: Path) -> ProcessingStats:
        """데이터 복사 및 증강"""
        stats = ProcessingStats()
        stats.original_images = sum(len(images) for images in splits.values())

        total_processed = 0
        total_augmented = 0
        class_counter = Counter()

        for split_name, image_infos in splits.items():
            images_dir = output_path / 'images' / split_name
            labels_dir = output_path / 'labels' / split_name

            # 증강 여부 결정
            apply_augmentation = (split_name == 'train' and self.config.enable_augmentation)

            for info in tqdm(image_infos, desc=f"{split_name} 처리"):
                # 원본 복사
                self._copy_single_file(info, images_dir, labels_dir)
                total_processed += 1

                # 클래스 분포 업데이트
                for class_name, count in info.class_distribution.items():
                    class_counter[class_name] += count

                # 증강 (훈련 데이터만)
                if apply_augmentation:
                    aug_count = self._augment_single_image(
                        info, images_dir, labels_dir, self.config.augmentation_factor - 1
                    )
                    total_augmented += aug_count

                    # 증강 데이터 클래스 분포
                    for class_name, count in info.class_distribution.items():
                        class_counter[class_name] += count * aug_count

        stats.processed_images = total_processed
        stats.augmented_images = total_augmented
        stats.total_objects = sum(class_counter.values())
        stats.class_distribution = dict(class_counter)

        return stats

    def _copy_single_file(self, info: ImageInfo, images_dir: Path, labels_dir: Path):
        """단일 파일 복사"""
        # 이미지 복사
        shutil.copy2(info.filepath, images_dir / info.filename)

        # 라벨 복사
        source_label = info.filepath.parent.parent / 'labels' / info.filepath.parent.name / f"{info.filepath.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, labels_dir / f"{info.filepath.stem}.txt")

    def _augment_single_image(self, info: ImageInfo, images_dir: Path, labels_dir: Path, count: int) -> int:
        """단일 이미지 증강 - 개선된 버전"""
        if count <= 0:
            return 0

        # 라벨 파일 경로
        source_label = info.filepath.parent.parent / 'labels' / info.filepath.parent.name / f"{info.filepath.stem}.txt"

        if not source_label.exists():
            logger.debug(f"라벨 없음: {source_label}")
            return 0

        try:
            # 이미지 로드
            image = cv2.imread(str(info.filepath))
            if image is None:
                logger.warning(f"이미지 로드 실패: {info.filepath}")
                return 0

            # 라벨 로드
            bboxes, class_labels = self._load_yolo_labels(source_label)
            if not bboxes:
                logger.debug(f"bbox 없음: {source_label}")
                return 0

            successful = 0

            for i in range(count):
                try:
                    # 증강 적용
                    aug_image, aug_bboxes, aug_labels = self.augmenter.augment_image(
                        image, bboxes, class_labels
                    )

                    if not aug_bboxes or len(aug_bboxes) == 0:
                        logger.warning(f"증강 후 bbox 손실: {info.filename} #{i+1}")
                        continue

                    # 증강 이미지 저장
                    aug_img_name = f"{info.filepath.stem}_aug{i+1}{info.filepath.suffix}"
                    aug_img_path = images_dir / aug_img_name

                    result = cv2.imwrite(str(aug_img_path), aug_image)
                    if not result:
                        logger.error(f"이미지 저장 실패: {aug_img_path}")
                        continue

                    # 증강 라벨 저장
                    aug_label_path = labels_dir / f"{info.filepath.stem}_aug{i+1}.txt"
                    self._save_yolo_labels(aug_label_path, aug_bboxes, aug_labels)

                    # 저장 확인
                    if aug_img_path.exists() and aug_label_path.exists():
                        successful += 1
                        if i == 0:  # 첫 번째 증강만 로그
                            logger.info(f"증강 성공: {aug_img_name}")
                    else:
                        logger.error(f"파일 확인 실패: {aug_img_name}")

                except Exception as e:
                    logger.error(f"증강 실패 {info.filename} #{i+1}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue

            if successful == 0:
                logger.warning(f"모든 증강 실패: {info.filename} (시도: {count}회)")

            return successful

        except Exception as e:
            logger.error(f"이미지 증강 실패 {info.filename}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    def _load_yolo_labels(self, label_path: Path) -> Tuple[List, List]:
        """YOLO 라벨 로드"""
        bboxes = []
        class_labels = []

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id)
        except Exception as e:
            logger.error(f"라벨 로드 실패 {label_path}: {e}")

        return bboxes, class_labels

    def _save_yolo_labels(self, label_path: Path, bboxes: List, class_labels: List):
        """YOLO 라벨 저장"""
        try:
            with open(label_path, 'w') as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        except Exception as e:
            logger.error(f"라벨 저장 실패 {label_path}: {e}")

    def _create_yaml_file(self, splits: Dict[str, List[ImageInfo]], output_path: Path, stats: ProcessingStats):
        """YAML 설정 파일 생성"""
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config.nc,
            'names': self.config.classes,

            'dataset_info': {
                'total': stats.original_images,
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test']),
                'processed': stats.processed_images,
                'augmented': stats.augmented_images,
                'filtered': stats.filtered_images
            },

            'preprocessing': {
                'method': 'optimized_stratified_split',
                'quality_filtering': self.config.enable_quality_filter,
                'quality_threshold': self.config.quality_threshold,
                'augmentation': self.config.enable_augmentation,
                'augmentation_factor': self.config.augmentation_factor,
                'random_seed': self.config.random_seed
            }
        }

        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"YAML 파일: {yaml_path}")

    def _save_statistics(self, stats: ProcessingStats, output_path: Path):
        """통계 저장"""
        stats_path = output_path / 'processing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, indent=2, ensure_ascii=False)

        logger.info(f"통계 파일: {stats_path}")

    def _print_summary(self, stats: ProcessingStats):
        """최종 요약 출력"""
        logger.info(f"\n📊 처리 통계:")
        logger.info(f"  - 원본 이미지: {stats.original_images}개")
        logger.info(f"  - 처리된 이미지: {stats.processed_images}개")
        logger.info(f"  - 증강된 이미지: {stats.augmented_images}개")
        logger.info(f"  - 총 이미지: {stats.processed_images + stats.augmented_images}개")
        logger.info(f"  - 총 객체: {stats.total_objects}개")
        logger.info(f"  - 처리 시간: {stats.processing_time:.2f}초")

        logger.info(f"\n📊 클래스 분포:")
        for class_name, count in sorted(stats.class_distribution.items()):
            percentage = (count / stats.total_objects) * 100 if stats.total_objects > 0 else 0
            logger.info(f"  - {class_name}: {count}개 ({percentage:.1f}%)")

        logger.info(f"\n✨ 출력 디렉토리: {self.config.output_dir}")


# ================== 메인 함수 ==================

def main():
    """메인 실행 함수"""
    # 설정
    config = Model3Config(
        source_dir=r"C:\Users\LX\Nong-View\model3_greenhouse",
        output_dir=r"C:\Users\LX\Nong-View\model3_greenhouse_best_processed",
        classes=['Greenhouse_single', 'Greenhouse_multi'],
        nc=2,
        enable_quality_filter=False,  # 품질 검수 이미 완료
        enable_augmentation=True,
        augmentation_factor=3,
        quality_threshold=0.4,
        random_seed=42
    )

    # 전처리기 생성 및 실행
    preprocessor = Model3OptimizedPreprocessor(config)
    stats = preprocessor.run()

    logger.info("\n" + "=" * 60)
    logger.info("🎉 전처리 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
