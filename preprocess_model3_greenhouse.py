#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model3 Greenhouse 데이터셋 전처리 스크립트
과학적 데이터 분할 및 재구성

기능:
- 임의로 분할된 train/val/test 데이터를 통합
- 계층화 분할 (Stratified Split)을 통한 과학적 재분할
- 클래스 균형 유지
- YOLO 형식 디렉토리 구조 생성
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """이미지 정보 클래스"""
    image_path: Path
    label_path: Path
    filename: str
    classes: List[int]
    dominant_class: int  # 가장 많은 객체의 클래스


class Model3GreenhousePreprocessor:
    """Model3 Greenhouse 데이터 전처리 클래스"""

    def __init__(self, source_dir: str, output_dir: str,
                 train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Args:
            source_dir: 현재 model3_greenhouse 디렉토리
            output_dir: 재구성된 데이터셋 출력 디렉토리
            train_ratio: 훈련 데이터 비율 (기본값: 0.8)
            val_ratio: 검증 데이터 비율 (기본값: 0.1)
            test_ratio: 테스트 데이터 비율 (기본값: 0.1)
            random_seed: 랜덤 시드 (재현성)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # 랜덤 시드 설정
        random.seed(random_seed)

        # 통계
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'class_distribution': Counter(),
            'split_distribution': {}
        }

        logger.info(f"전처리기 초기화 완료")
        logger.info(f"  - 소스: {self.source_dir}")
        logger.info(f"  - 출력: {self.output_dir}")
        logger.info(f"  - 분할 비율: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")

    def run(self):
        """전체 전처리 프로세스 실행"""
        logger.info("=" * 60)
        logger.info("🚀 Model3 Greenhouse 데이터 전처리 시작")
        logger.info("=" * 60)

        # 1. 데이터 수집
        logger.info("\n[1/5] 데이터 수집 중...")
        all_images = self._collect_all_images()
        logger.info(f"  ✓ 총 {len(all_images)}개 이미지 수집 완료")

        # 2. 통계 분석
        logger.info("\n[2/5] 데이터 분석 중...")
        self._analyze_data(all_images)
        self._print_statistics()

        # 3. 계층화 분할
        logger.info("\n[3/5] 계층화 분할 중...")
        train_images, val_images, test_images = self._stratified_split(all_images)
        logger.info(f"  ✓ Train: {len(train_images)}개")
        logger.info(f"  ✓ Val: {len(val_images)}개")
        logger.info(f"  ✓ Test: {len(test_images)}개")

        # 4. 출력 디렉토리 생성
        logger.info("\n[4/5] 출력 디렉토리 생성 중...")
        self._setup_output_structure()

        # 5. 데이터 복사
        logger.info("\n[5/5] 데이터 복사 중...")
        self._copy_split_data('train', train_images)
        self._copy_split_data('val', val_images)
        self._copy_split_data('test', test_images)

        # 6. YAML 파일 생성
        logger.info("\nYAML 설정 파일 생성 중...")
        self._create_yaml_file(len(train_images), len(val_images), len(test_images))

        logger.info("\n" + "=" * 60)
        logger.info("✅ 전처리 완료!")
        logger.info("=" * 60)
        self._print_final_summary()

    def _collect_all_images(self) -> List[ImageInfo]:
        """모든 이미지 수집 (train/val/test 통합)"""
        all_images = []

        # train, val, test 폴더에서 모든 이미지 수집
        for split in ['train', 'val', 'test']:
            images_dir = self.source_dir / 'images' / split
            labels_dir = self.source_dir / 'labels' / split

            if not images_dir.exists():
                logger.warning(f"  ⚠ 디렉토리 없음: {images_dir}")
                continue

            # 이미지 파일 찾기
            image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))

            for img_path in image_files:
                # 대응하는 라벨 파일 찾기
                label_path = labels_dir / f"{img_path.stem}.txt"

                if label_path.exists():
                    # 라벨 파싱
                    classes = self._parse_label_file(label_path)

                    if classes:
                        # 가장 많은 객체를 가진 클래스 찾기
                        class_counts = Counter(classes)
                        dominant_class = class_counts.most_common(1)[0][0]

                        image_info = ImageInfo(
                            image_path=img_path,
                            label_path=label_path,
                            filename=img_path.name,
                            classes=classes,
                            dominant_class=dominant_class
                        )
                        all_images.append(image_info)
                else:
                    logger.warning(f"  ⚠ 라벨 없음: {img_path.name}")

        return all_images

    def _parse_label_file(self, label_path: Path) -> List[int]:
        """YOLO 형식 라벨 파일 파싱"""
        classes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id x_center y_center width height
                        class_id = int(parts[0])
                        classes.append(class_id)
        except Exception as e:
            logger.error(f"  ✗ 라벨 파싱 실패 {label_path}: {e}")

        return classes

    def _analyze_data(self, images: List[ImageInfo]):
        """데이터 통계 분석"""
        self.stats['total_images'] = len(images)

        for img_info in images:
            # 객체 개수
            self.stats['total_objects'] += len(img_info.classes)

            # 클래스 분포
            for class_id in img_info.classes:
                self.stats['class_distribution'][class_id] += 1

    def _print_statistics(self):
        """통계 출력"""
        logger.info(f"  📊 통계:")
        logger.info(f"    - 총 이미지: {self.stats['total_images']}개")
        logger.info(f"    - 총 객체: {self.stats['total_objects']}개")
        logger.info(f"    - 클래스 분포:")

        class_names = {0: 'Greenhouse_single', 1: 'Greenhouse_multi'}
        for class_id, count in sorted(self.stats['class_distribution'].items()):
            class_name = class_names.get(class_id, f'Class_{class_id}')
            percentage = (count / self.stats['total_objects']) * 100
            logger.info(f"      • {class_name}: {count}개 ({percentage:.1f}%)")

    def _stratified_split(self, images: List[ImageInfo]) -> Tuple[List[ImageInfo], List[ImageInfo], List[ImageInfo]]:
        """계층화 분할 (Stratified Split)"""
        # 클래스별로 이미지 그룹화
        class_groups = defaultdict(list)
        for img_info in images:
            class_groups[img_info.dominant_class].append(img_info)

        train_images = []
        val_images = []
        test_images = []

        # 각 클래스별로 분할
        for class_id, class_images in class_groups.items():
            # 셔플
            random.shuffle(class_images)

            n_total = len(class_images)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)

            # 분할
            train_images.extend(class_images[:n_train])
            val_images.extend(class_images[n_train:n_train + n_val])
            test_images.extend(class_images[n_train + n_val:])

            class_name = {0: 'Greenhouse_single', 1: 'Greenhouse_multi'}.get(class_id, f'Class_{class_id}')
            logger.info(f"  • {class_name}: Train={n_train}, Val={n_val}, Test={n_total - n_train - n_val}")

        # 최종 셔플
        random.shuffle(train_images)
        random.shuffle(val_images)
        random.shuffle(test_images)

        return train_images, val_images, test_images

    def _setup_output_structure(self):
        """출력 디렉토리 구조 생성"""
        # 기존 디렉토리 삭제 (있는 경우)
        if self.output_dir.exists():
            logger.info(f"  ⚠ 기존 출력 디렉토리 삭제: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        # YOLO 형식 디렉토리 구조 생성
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        logger.info(f"  ✓ 출력 디렉토리 생성 완료: {self.output_dir}")

    def _copy_split_data(self, split_name: str, images: List[ImageInfo]):
        """분할된 데이터 복사"""
        images_dir = self.output_dir / 'images' / split_name
        labels_dir = self.output_dir / 'labels' / split_name

        # 클래스 분포 추적
        class_dist = Counter()

        for img_info in tqdm(images, desc=f"  {split_name} 복사"):
            # 이미지 복사
            shutil.copy2(img_info.image_path, images_dir / img_info.filename)

            # 라벨 복사
            shutil.copy2(img_info.label_path, labels_dir / f"{Path(img_info.filename).stem}.txt")

            # 클래스 분포 업데이트
            for class_id in img_info.classes:
                class_dist[class_id] += 1

        # 분할별 통계 저장
        self.stats['split_distribution'][split_name] = {
            'images': len(images),
            'objects': sum(class_dist.values()),
            'class_distribution': dict(class_dist)
        }

        logger.info(f"  ✓ {split_name}: {len(images)}개 이미지, {sum(class_dist.values())}개 객체")

    def _create_yaml_file(self, n_train: int, n_val: int, n_test: int):
        """데이터셋 YAML 파일 생성"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 2,
            'names': ['Greenhouse_single', 'Greenhouse_multi'],

            # 메타데이터
            'dataset_info': {
                'total': self.stats['total_images'],
                'train': n_train,
                'val': n_val,
                'test': n_test,
                'augmented': 0
            },

            'preprocessing': {
                'method': 'stratified_split',
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'random_seed': self.random_seed
            }
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(f"  ✓ YAML 파일 생성: {yaml_path}")

    def _print_final_summary(self):
        """최종 요약 출력"""
        logger.info("\n📊 최종 데이터 분포:")

        class_names = {0: 'Greenhouse_single', 1: 'Greenhouse_multi'}

        for split_name in ['train', 'val', 'test']:
            if split_name in self.stats['split_distribution']:
                split_info = self.stats['split_distribution'][split_name]
                logger.info(f"\n  📁 {split_name.upper()}:")
                logger.info(f"    - 이미지: {split_info['images']}개")
                logger.info(f"    - 객체: {split_info['objects']}개")
                logger.info(f"    - 클래스 분포:")

                for class_id, count in sorted(split_info['class_distribution'].items()):
                    class_name = class_names.get(class_id, f'Class_{class_id}')
                    percentage = (count / split_info['objects']) * 100
                    logger.info(f"      • {class_name}: {count}개 ({percentage:.1f}%)")

        logger.info(f"\n✨ 재구성된 데이터셋 위치: {self.output_dir}")
        logger.info(f"✨ 설정 파일: {self.output_dir / 'data.yaml'}")


def main():
    """메인 함수"""
    # 경로 설정
    source_dir = r"C:\Users\LX\Nong-View\model3_greenhouse"
    output_dir = r"C:\Users\LX\Nong-View\model3_greenhouse_processed"

    # 전처리기 생성 및 실행
    preprocessor = Model3GreenhousePreprocessor(
        source_dir=source_dir,
        output_dir=output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
