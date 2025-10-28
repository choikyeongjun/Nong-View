#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIF 파일 직접 사용을 위한 데이터셋 준비 스크립트
GeoJSON 라벨을 YOLO 형식으로 변환하되, 이미지는 TIF 그대로 사용
"""

import json
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import rowcol
from tqdm import tqdm
import shutil
import random
from collections import Counter
import yaml

# 클래스 매핑 (ID -> YOLO 클래스 인덱스)
CLASS_MAPPING = {
    2: 0,  # IRG(생산기) -> 0
    4: 1,  # 호밀(생산기) -> 1
    9: 2,  # 비닐하우스(단동) -> 2
    10: 3, # 비닐하우스(다동) -> 3
    11: 4, # 곤포사일리지 -> 4
}

CLASS_NAMES = {
    0: "IRG_production",
    1: "Rye_production",
    2: "Greenhouse_single",
    3: "Greenhouse_multi",
    4: "Silage_bale"
}

def geojson_to_yolo(json_path, tif_path):
    """GeoJSON 좌표를 YOLO 형식으로 변환"""
    yolo_labels = []

    try:
        # GeoJSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # TIF 파일의 지리참조 정보 읽기
        with rasterio.open(tif_path) as src:
            transform = src.transform
            img_width = src.width
            img_height = src.height

            for feature in data.get('features', []):
                class_id = feature['properties'].get('id')

                # 클래스 ID 확인
                if class_id not in CLASS_MAPPING:
                    continue

                yolo_class = CLASS_MAPPING[class_id]
                geometry = feature['geometry']

                if geometry['type'] == 'MultiPolygon':
                    for polygon in geometry['coordinates']:
                        for ring in polygon:
                            # 좌표를 픽셀 좌표로 변환
                            pixel_coords = []
                            for coord in ring:
                                row, col = rowcol(transform, coord[0], coord[1])
                                # 정규화 (0-1 범위)
                                x_norm = col / img_width
                                y_norm = row / img_height

                                # 범위 체크
                                if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                                    pixel_coords.extend([x_norm, y_norm])

                            if len(pixel_coords) >= 6:  # 최소 3개 점
                                label_line = f"{yolo_class} " + " ".join(f"{c:.6f}" for c in pixel_coords)
                                yolo_labels.append(label_line)

                elif geometry['type'] == 'Polygon':
                    for ring in geometry['coordinates']:
                        pixel_coords = []
                        for coord in ring:
                            row, col = rowcol(transform, coord[0], coord[1])
                            x_norm = col / img_width
                            y_norm = row / img_height

                            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                                pixel_coords.extend([x_norm, y_norm])

                        if len(pixel_coords) >= 6:
                            label_line = f"{yolo_class} " + " ".join(f"{c:.6f}" for c in pixel_coords)
                            yolo_labels.append(label_line)

    except Exception as e:
        print(f"Error processing {json_path}: {e}")

    return yolo_labels

def process_dataset(input_dirs, output_dir):
    """TIF 파일 직접 사용 데이터셋 처리"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 디렉토리 구조 생성 (TIF용)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    all_data = []
    class_counts = Counter()

    print("\n=== 데이터 수집 시작 (TIF 직접 사용) ===")

    for input_dir in input_dirs:
        input_path = Path(input_dir)
        img_dir = input_path / '이미지데이터'
        label_dir = input_path / '라벨링데이터' / 'editjson'

        print(f"\n처리 중: {input_dir}")

        # TIF 파일 목록
        tif_files = list(img_dir.glob('*.tif'))
        print(f"발견된 TIF 파일: {len(tif_files)}개")

        for tif_path in tqdm(tif_files, desc="라벨 변환 중"):
            stem = tif_path.stem
            json_path = label_dir / f"{stem}.json"

            if not json_path.exists():
                continue

            # YOLO 라벨 생성
            yolo_labels = geojson_to_yolo(json_path, tif_path)

            if yolo_labels:
                # 클래스 카운트
                for label in yolo_labels:
                    class_id = int(label.split()[0])
                    class_counts[class_id] += 1

                all_data.append({
                    'tif_path': tif_path,
                    'labels': yolo_labels,
                    'stem': stem
                })

    print(f"\n총 유효 데이터: {len(all_data)}개")
    print("\n클래스별 객체 수:")
    for cls_id, count in sorted(class_counts.items()):
        print(f"  {CLASS_NAMES[cls_id]}: {count}개")

    if len(all_data) == 0:
        print("변환할 데이터가 없습니다!")
        return

    # 데이터 분할 (8:1:1)
    random.seed(42)  # 재현 가능한 분할
    random.shuffle(all_data)
    n_total = len(all_data)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]

    print(f"\n데이터 분할:")
    print(f"  Train: {len(train_data)}개")
    print(f"  Val: {len(val_data)}개")
    print(f"  Test: {len(test_data)}개")

    # 파일 복사 및 라벨 저장
    print("\n파일 처리 중...")

    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        for item in tqdm(split_data, desc=f"{split_name} 처리"):
            # TIF 파일 복사 (심볼릭 링크로 공간 절약)
            img_name = f"{item['stem']}.tif"
            img_dest = output_path / 'images' / split_name / img_name

            # 하드링크 또는 복사
            if not img_dest.exists():
                try:
                    # Windows에서는 복사
                    shutil.copy2(item['tif_path'], img_dest)
                except Exception as e:
                    print(f"Error copying {item['tif_path']}: {e}")
                    continue

            # 라벨 저장
            label_dest = output_path / 'labels' / split_name / f"{item['stem']}.txt"
            with open(label_dest, 'w') as f:
                f.write('\n'.join(item['labels']))

    # dataset.yaml 생성 (TIF 지원 설정 포함)
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': list(CLASS_NAMES.values()),

        # TIF 관련 설정
        'format': 'tif',
        'img_size': [1024, 1024],
        'normalize': True,

        # 데이터 통계
        'dataset_stats': {
            'total': n_total,
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data),
            'class_counts': {CLASS_NAMES[k]: v for k, v in class_counts.items()}
        }
    }

    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ 변환 완료!")
    print(f"📁 출력 디렉토리: {output_path}")
    print(f"📄 설정 파일: {yaml_path}")

    # 통계 저장
    stats = {
        'total_images': n_total,
        'train_images': len(train_data),
        'val_images': len(val_data),
        'test_images': len(test_data),
        'class_counts': {CLASS_NAMES[k]: v for k, v in class_counts.items()},
        'image_format': 'tif',
        'image_size': '1024x1024 (mostly)'
    }

    stats_path = output_path / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"📊 통계 파일: {stats_path}")

def main():
    # 입력 디렉토리
    input_dirs = [
        '251014_데이터',
        '251020_데이터'
    ]

    # 출력 디렉토리 (TIF 전용)
    output_dir = 'growth_tif_dataset'

    # 처리 실행
    process_dataset(input_dirs, output_dir)

if __name__ == "__main__":
    main()