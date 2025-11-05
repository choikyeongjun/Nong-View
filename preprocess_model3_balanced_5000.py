#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model3 Greenhouse - 클래스 균형 전처리 (각 5000개)
YOLOv11-seg 전용, Segmentation (polygon) 형식 지원

작성: Claude Sonnet
날짜: 2025-11-04
"""

import os
import json
import yaml
import shutil
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np
from tqdm import tqdm

# ================== 설정 ==================

@dataclass
class BalancedSegConfig:
    """균형 증강 전용 설정"""
    source_dir: str = r"D:\Nong-View\model3_greenhouse"
    output_dir: str = r"D:\model3_greenhouse_balanced_5000"
    
    classes: List[str] = field(default_factory=lambda: ['Greenhouse_single', 'Greenhouse_multi'])
    nc: int = 2
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    target_train_per_class: int = 5000  # 각 클래스당 train 목표 개수
    
    random_seed: int = 42


@dataclass
class ImageInfo:
    """이미지 정보"""
    filepath: Path
    filename: str
    classes: List[int] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    dominant_class: int = 0


# ================== 고급 증강 클래스 ==================

class AdvancedAugmenter:
    """다양한 증강 기법 (polygon 지원)"""
    
    def __init__(self):
        self.methods = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.brightness,
            self.contrast,
            self.noise,
            self.flip_brightness,
            self.rotate_contrast
        ]
    
    def augment(self, image: np.ndarray, polygons: List[List[float]], 
                class_labels: List[int], method_idx: int) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """증강 수행"""
        method = self.methods[method_idx % len(self.methods)]
        return method(image, polygons, class_labels)
    
    def horizontal_flip(self, image, polygons, labels):
        aug_img = cv2.flip(image, 1)
        aug_polys = [[1.0 - polygons[i][j] if j % 2 == 0 else polygons[i][j] 
                      for j in range(len(polygons[i]))] for i in range(len(polygons))]
        return aug_img, aug_polys, labels.copy()
    
    def vertical_flip(self, image, polygons, labels):
        aug_img = cv2.flip(image, 0)
        aug_polys = [[polygons[i][j] if j % 2 == 0 else 1.0 - polygons[i][j] 
                      for j in range(len(polygons[i]))] for i in range(len(polygons))]
        return aug_img, aug_polys, labels.copy()
    
    def rotate_90(self, image, polygons, labels):
        aug_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        aug_polys = [[polygons[i][j+1] if j % 2 == 0 else 1.0 - polygons[i][j-1] 
                      for j in range(len(polygons[i]))] for i in range(len(polygons))]
        return aug_img, aug_polys, labels.copy()
    
    def rotate_180(self, image, polygons, labels):
        aug_img = cv2.rotate(image, cv2.ROTATE_180)
        aug_polys = [[1.0 - polygons[i][j] for j in range(len(polygons[i]))] 
                     for i in range(len(polygons))]
        return aug_img, aug_polys, labels.copy()
    
    def rotate_270(self, image, polygons, labels):
        aug_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        aug_polys = [[1.0 - polygons[i][j+1] if j % 2 == 0 else polygons[i][j-1] 
                      for j in range(len(polygons[i]))] for i in range(len(polygons))]
        return aug_img, aug_polys, labels.copy()
    
    def brightness(self, image, polygons, labels):
        b = random.uniform(0.7, 1.3)
        aug_img = np.clip(image.astype(np.float32) * b, 0, 255).astype(np.uint8)
        return aug_img, [p.copy() for p in polygons], labels.copy()
    
    def contrast(self, image, polygons, labels):
        c = random.uniform(0.8, 1.2)
        aug_img = np.clip((image.astype(np.float32) - 127.5) * c + 127.5, 0, 255).astype(np.uint8)
        return aug_img, [p.copy() for p in polygons], labels.copy()
    
    def noise(self, image, polygons, labels):
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        aug_img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return aug_img, [p.copy() for p in polygons], labels.copy()
    
    def flip_brightness(self, image, polygons, labels):
        aug_img, aug_polys, aug_labels = self.horizontal_flip(image, polygons, labels)
        b = random.uniform(0.8, 1.2)
        aug_img = np.clip(aug_img.astype(np.float32) * b, 0, 255).astype(np.uint8)
        return aug_img, aug_polys, aug_labels
    
    def rotate_contrast(self, image, polygons, labels):
        aug_img, aug_polys, aug_labels = self.rotate_180(image, polygons, labels)
        c = random.uniform(0.9, 1.1)
        aug_img = np.clip((aug_img.astype(np.float32) - 127.5) * c + 127.5, 0, 255).astype(np.uint8)
        return aug_img, aug_polys, aug_labels


# ================== 메인 전처리 클래스 ==================

class BalancedSegmentationPreprocessor:
    """클래스 균형 전처리 (train 각 5000개)"""
    
    def __init__(self, config: BalancedSegConfig):
        self.config = config
        self.augmenter = AdvancedAugmenter()
        
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        print("=" * 70)
        print(">> Model3 Greenhouse 클래스 균형 전처리")
        print("=" * 70)
        print(f"원본: {config.source_dir}")
        print(f"출력: {config.output_dir}")
        print(f"목표: 각 클래스 train {config.target_train_per_class}개")
        print("=" * 70)
    
    def run(self):
        """전체 프로세스 실행"""
        start_time = time.time()
        
        # 1. 데이터 수집
        print("\n[1/6] 데이터 수집 중...", end=" ")
        image_infos = self._collect_images()
        print(f"OK ({len(image_infos)}개)")
        
        # 2. 계층화 분할
        print("[2/6] 계층화 분할 중...", end=" ")
        splits = self._stratified_split(image_infos)
        print(f"OK (train:{len(splits['train'])}, val:{len(splits['val'])}, test:{len(splits['test'])})")
        
        # 3. 출력 구조
        print("[3/6] 디렉토리 생성...", end=" ")
        output_path = Path(self.config.output_dir)
        self._setup_output_structure(output_path)
        print("OK")
        
        # 4. Val/Test 복사
        print("[4/6] Val/Test 복사 중...")
        self._copy_val_test(splits, output_path)
        print("OK")
        
        # 5. Train 클래스별 증강
        print(f"\n[5/6] Train 데이터 클래스별 균형 증강 (각 {self.config.target_train_per_class}개)")
        stats = self._balanced_augment_train(splits['train'], output_path)
        
        # 6. 메타데이터
        print("\n[6/6] 메타데이터 생성...", end=" ")
        self._create_yaml(splits, output_path, stats)
        self._save_stats(stats, output_path)
        print("OK")
        
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(">> 전처리 완료!")
        print("=" * 70)
        self._print_summary(stats, processing_time)
    
    def _collect_images(self) -> List[ImageInfo]:
        """이미지 수집"""
        source_path = Path(self.config.source_dir)
        all_images = []
        
        for split in ['train', 'val', 'test']:
            images_dir = source_path / 'images' / split
            labels_dir = source_path / 'labels' / split
            
            if not images_dir.exists():
                continue
            
            for img_path in images_dir.glob('*.png'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if label_path.exists():
                    classes, class_dist = self._parse_seg_label(label_path)
                    
                    if classes:
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
    
    def _parse_seg_label(self, label_path: Path) -> Tuple[List[int], Dict[str, int]]:
        """Segmentation 라벨 파싱"""
        classes = []
        class_dist = defaultdict(int)
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        classes.append(class_id)
                        
                        if 0 <= class_id < len(self.config.classes):
                            class_name = self.config.classes[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        
                        class_dist[class_name] += 1
        except Exception as e:
            pass
        
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
            shutil.rmtree(output_path)
        
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def _copy_val_test(self, splits: Dict, output_path: Path):
        """Val/Test 데이터 복사"""
        for split_name in ['val', 'test']:
            images_dir = output_path / 'images' / split_name
            labels_dir = output_path / 'labels' / split_name
            
            for info in tqdm(splits[split_name], desc=f"   {split_name:4s}", ncols=70, 
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
                self._copy_single_file(info, images_dir, labels_dir)
    
    def _balanced_augment_train(self, train_images: List[ImageInfo], output_path: Path) -> Dict:
        """Train 데이터 클래스별 균형 증강"""
        # 클래스별로 분리
        class_images = defaultdict(list)
        for info in train_images:
            class_images[info.dominant_class].append(info)
        
        print()
        for class_id, images in class_images.items():
            class_name = self.config.classes[class_id]
            print(f"   - {class_name}: 원본 {len(images)}개")
        
        # 클래스별 증강 전략 계산
        aug_strategy = {}
        for class_id, images in class_images.items():
            needed = self.config.target_train_per_class - len(images)
            aug_strategy[class_id] = max(0, needed)
            class_name = self.config.classes[class_id]
            print(f"   - {class_name}: {needed}개 증강 필요 -> 최종 {len(images) + needed}개")
        
        # 클래스별 증강 실행
        all_stats = {}
        images_dir = output_path / 'images' / 'train'
        labels_dir = output_path / 'labels' / 'train'
        
        print()
        for class_id, images in class_images.items():
            class_name = self.config.classes[class_id]
            
            # 원본 복사
            for info in images:
                self._copy_single_file(info, images_dir, labels_dir)
            
            # 증강
            needed = aug_strategy[class_id]
            if needed > 0:
                print(f"\n   >> {class_name} 증강 ({needed}개)...")
                augmented = self._augment_class(class_id, images, images_dir, labels_dir, needed)
                all_stats[class_id] = {
                    'original': len(images),
                    'augmented': augmented,
                    'total': len(images) + augmented
                }
            else:
                all_stats[class_id] = {
                    'original': len(images),
                    'augmented': 0,
                    'total': len(images)
                }
        
        return all_stats
    
    def _augment_class(self, class_id: int, source_images: List[ImageInfo], 
                      images_dir: Path, labels_dir: Path, needed: int) -> int:
        """특정 클래스 증강"""
        augmented = 0
        
        pbar = tqdm(total=needed, desc=f"      {self.config.classes[class_id][:12]:12s}",
                   ncols=70, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        
        method_idx = 0
        while augmented < needed:
            source_img = random.choice(source_images)
            
            try:
                if self._augment_single(source_img, images_dir, labels_dir, class_id, augmented, method_idx):
                    augmented += 1
                    pbar.update(1)
                
                method_idx += 1
            except:
                continue
        
        pbar.close()
        return augmented
    
    def _augment_single(self, source_img: ImageInfo, images_dir: Path, labels_dir: Path,
                       class_id: int, aug_idx: int, method_idx: int) -> bool:
        """단일 이미지 증강"""
        try:
            # 이미지 로드
            image = cv2.imread(str(source_img.filepath))
            if image is None:
                return False
            
            # 라벨 로드
            source_label = source_img.filepath.parent.parent / 'labels' / \
                          source_img.filepath.parent.name / f"{source_img.filepath.stem}.txt"
            
            if not source_label.exists():
                return False
            
            polygons, class_labels = self._load_seg_labels(source_label)
            if not polygons:
                return False
            
            # 증강
            aug_img, aug_polys, aug_labels = self.augmenter.augment(
                image, polygons, class_labels, method_idx
            )
            
            if not aug_polys:
                return False
            
            # 저장
            class_prefix = self.config.classes[class_id][:6]
            aug_name = f"{class_prefix}_aug{aug_idx:04d}"
            
            aug_img_path = images_dir / f"{aug_name}.png"
            if not cv2.imwrite(str(aug_img_path), aug_img):
                return False
            
            aug_label_path = labels_dir / f"{aug_name}.txt"
            self._save_seg_labels(aug_label_path, aug_polys, aug_labels)
            
            return aug_img_path.exists() and aug_label_path.exists()
            
        except:
            return False
    
    def _copy_single_file(self, info: ImageInfo, images_dir: Path, labels_dir: Path):
        """단일 파일 복사"""
        shutil.copy2(info.filepath, images_dir / info.filename)
        
        source_label = info.filepath.parent.parent / 'labels' / \
                      info.filepath.parent.name / f"{info.filepath.stem}.txt"
        
        if source_label.exists():
            shutil.copy2(source_label, labels_dir / f"{info.filepath.stem}.txt")
    
    def _load_seg_labels(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """Segmentation 라벨 로드"""
        polygons = []
        class_labels = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        polygon = [float(x) for x in parts[1:]]
                        polygons.append(polygon)
                        class_labels.append(class_id)
        except:
            pass
        
        return polygons, class_labels
    
    def _save_seg_labels(self, label_path: Path, polygons: List[List[float]], class_labels: List[int]):
        """Segmentation 라벨 저장"""
        try:
            with open(label_path, 'w') as f:
                for polygon, class_id in zip(polygons, class_labels):
                    coords_str = ' '.join([f"{coord:.10f}" for coord in polygon])
                    f.write(f"{class_id} {coords_str}\n")
        except:
            pass
    
    def _create_yaml(self, splits: Dict, output_path: Path, stats: Dict):
        """YAML 파일 생성"""
        # Train 총 개수 계산
        train_total = sum(s['total'] for s in stats.values())
        
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config.nc,
            'names': self.config.classes,
            'task': 'segment',
            
            'dataset_info': {
                'total_train': train_total,
                'val': len(splits['val']),
                'test': len(splits['test']),
                'total': train_total + len(splits['val']) + len(splits['test'])
            },
            
            'preprocessing': {
                'method': 'balanced_class_augmentation',
                'target_per_class': self.config.target_train_per_class,
                'class_stats': {}
            }
        }
        
        for class_id, class_stats in stats.items():
            class_name = self.config.classes[class_id]
            yaml_content['preprocessing']['class_stats'][class_name] = class_stats
        
        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def _save_stats(self, stats: Dict, output_path: Path):
        """통계 저장"""
        stats_path = output_path / 'processing_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self, stats: Dict, processing_time: float):
        """요약 출력"""
        print("\n>> 최종 통계:")
        
        total_original = 0
        total_augmented = 0
        total_final = 0
        
        for class_id, class_stats in stats.items():
            class_name = self.config.classes[class_id]
            print(f"\n  [{class_name}]")
            print(f"     - 원본: {class_stats['original']}개")
            print(f"     - 증강: {class_stats['augmented']}개")
            print(f"     - 최종: {class_stats['total']}개")
            
            total_original += class_stats['original']
            total_augmented += class_stats['augmented']
            total_final += class_stats['total']
        
        print(f"\n  [전체 합계]")
        print(f"     - 원본: {total_original}개")
        print(f"     - 증강: {total_augmented}개")
        print(f"     - 최종: {total_final}개")
        
        print(f"\n처리 시간: {processing_time:.1f}초")
        print(f"출력 경로: {self.config.output_dir}")


# ================== 메인 함수 ==================

def main():
    """메인 실행"""
    config = BalancedSegConfig(
        source_dir=r"D:\Nong-View\model3_greenhouse",
        output_dir=r"D:\model3_greenhouse_balanced_5000",
        target_train_per_class=5000,
        random_seed=42
    )
    
    preprocessor = BalancedSegmentationPreprocessor(config)
    preprocessor.run()
    
    print("\n" + "=" * 70)
    print(">> 모든 작업 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()

