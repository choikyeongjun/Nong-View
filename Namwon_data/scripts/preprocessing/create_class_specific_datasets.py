#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
클래스별 YOLOv11 Segmentation 데이터셋 생성
각 클래스(비닐하우스 단동, 비닐하우스 다동, 곤포사일리지)별로 별도 모델 학습용 데이터셋 생성
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm

# 타겟 클래스 정의
TARGET_CLASSES = {
    9: {
        'name': '비닐하우스_단동',
        'output_dir': 'dataset_greenhouse_single',
        'description': 'Greenhouse Single (단동 비닐하우스)'
    },
    10: {
        'name': '비닐하우스_다동',
        'output_dir': 'dataset_greenhouse_multi',
        'description': 'Greenhouse Multi (다동 비닐하우스)'
    },
    11: {
        'name': '곤포사일리지',
        'output_dir': 'dataset_silage_bale',
        'description': 'Silage Bale (곤포사일리지)'
    }
}

# 데이터 폴더 리스트
DATA_FOLDERS = [
    '250903-데이터',
    '250910-최종납품',
    '250911-최종납품',
    '250917-최종납품',
    '250918-최종납품',
    '251014_데이터',
    '251020_데이터'
]

def collect_class_data(target_class_id):
    """특정 클래스의 이미지-라벨 쌍 수집"""

    print(f"\n📂 클래스 {target_class_id} ({TARGET_CLASSES[target_class_id]['name']}) 데이터 수집 중...")

    all_pairs = []

    for folder_name in DATA_FOLDERS:
        folder_path = Path(folder_name)

        label_dir = folder_path / '라벨링데이터' / 'pixeljson'
        image_dir = folder_path / '이미지데이터'

        if not label_dir.exists() or not image_dir.exists():
            continue

        # 라벨 파일 탐색
        label_files = list(label_dir.glob('*.txt'))

        for label_file in label_files:
            # 라벨 파일에서 타겟 클래스 포함 여부 확인
            has_target_class = False
            label_lines = []

            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 3:
                                class_id = int(parts[0])
                                if class_id == target_class_id:
                                    has_target_class = True
                                    label_lines.append(line)

                if has_target_class:
                    # 대응하는 TIF 이미지 찾기
                    image_name = label_file.stem
                    image_path = None

                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                        candidate = image_dir / f"{image_name}{ext}"
                        if candidate.exists():
                            image_path = candidate
                            break

                    if image_path and image_path.exists():
                        all_pairs.append({
                            'image_path': image_path,
                            'label_lines': label_lines,
                            'stem': image_name,
                            'source_folder': folder_name
                        })

            except Exception as e:
                print(f"   ⚠️  파일 처리 오류 {label_file}: {e}")

    print(f"   ✅ 수집 완료: {len(all_pairs)}개 이미지")
    return all_pairs

def split_dataset(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """데이터셋을 train:val:test로 분할"""

    random.shuffle(pairs)

    total = len(pairs)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]

    print(f"\n📊 데이터 분할:")
    print(f"   Train: {len(train_pairs)}개 ({len(train_pairs)/total*100:.1f}%)")
    print(f"   Val:   {len(val_pairs)}개 ({len(val_pairs)/total*100:.1f}%)")
    print(f"   Test:  {len(test_pairs)}개 ({len(test_pairs)/total*100:.1f}%)")

    return train_pairs, val_pairs, test_pairs

def create_dataset(target_class_id):
    """특정 클래스의 데이터셋 생성"""

    class_info = TARGET_CLASSES[target_class_id]
    output_dir = Path(class_info['output_dir'])

    print("\n" + "="*80)
    print(f"🚀 {class_info['name']} 데이터셋 생성 시작")
    print(f"   출력 디렉토리: {output_dir}")
    print("="*80)

    # 출력 디렉토리 구조 생성
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 데이터 수집
    all_pairs = collect_class_data(target_class_id)

    if len(all_pairs) == 0:
        print(f"❌ {class_info['name']} 데이터가 없습니다!")
        return None

    # 데이터 분할
    random.seed(42)  # 재현 가능성
    train_pairs, val_pairs, test_pairs = split_dataset(all_pairs)

    # 파일 복사 및 라벨 생성
    print("\n📋 파일 처리 중...")

    total_objects = 0

    for split_name, split_data in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        print(f"\n   {split_name.upper()} 처리 중...")

        for item in tqdm(split_data, desc=f"   {split_name}"):
            # 이미지 복사
            img_dest = output_dir / 'images' / split_name / f"{item['stem']}.tif"

            try:
                if not img_dest.exists():
                    shutil.copy2(item['image_path'], img_dest)
            except Exception as e:
                print(f"      ⚠️  이미지 복사 실패 {item['stem']}: {e}")
                continue

            # 라벨 저장 (클래스 ID를 0으로 변경)
            label_dest = output_dir / 'labels' / split_name / f"{item['stem']}.txt"

            yolo_labels = []
            for label_line in item['label_lines']:
                # 클래스 ID를 0으로 변경 (단일 클래스 모델)
                parts = label_line.split()
                parts[0] = '0'  # 클래스 ID를 0으로
                yolo_labels.append(' '.join(parts))
                total_objects += 1

            with open(label_dest, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_labels))

    # dataset.yaml 생성
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # 단일 클래스
        'names': [class_info['name']],

        # 메타데이터
        'description': class_info['description'],
        'original_class_id': target_class_id,
        'task': 'segment',
        'image_format': 'tif',

        # 통계
        'dataset_stats': {
            'total_images': len(all_pairs),
            'total_objects': total_objects,
            'train_images': len(train_pairs),
            'val_images': len(val_pairs),
            'test_images': len(test_pairs),
        }
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print("\n" + "="*80)
    print(f"✅ {class_info['name']} 데이터셋 생성 완료!")
    print("="*80)
    print(f"📁 출력 디렉토리: {output_dir.absolute()}")
    print(f"📄 설정 파일: {yaml_path}")
    print(f"📊 총 이미지: {len(all_pairs)}개")
    print(f"📊 총 객체: {total_objects}개")
    print(f"   - Train: {len(train_pairs)}개 이미지")
    print(f"   - Val:   {len(val_pairs)}개 이미지")
    print(f"   - Test:  {len(test_pairs)}개 이미지")

    # 통계 JSON 저장
    import json
    stats_path = output_dir / 'dataset_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(yaml_content['dataset_stats'], f, ensure_ascii=False, indent=2)

    print(f"📈 통계 파일: {stats_path}")

    return output_dir

def main():
    """메인 함수 - 3개 클래스 모두 처리"""

    print("="*80)
    print("🌟 YOLOv11 Segmentation 클래스별 데이터셋 생성")
    print("="*80)
    print("\n📋 생성할 데이터셋:")
    for class_id, info in TARGET_CLASSES.items():
        print(f"   {class_id}. {info['name']} -> {info['output_dir']}/")

    print(f"\n📂 소스 데이터 폴더: {len(DATA_FOLDERS)}개")
    for folder in DATA_FOLDERS:
        print(f"   - {folder}")

    print("\n⚙️  설정:")
    print("   - 데이터 분할: Train:Val:Test = 8:1:1")
    print("   - 각 클래스: nc=1 (단일 클래스 모델)")
    print("   - 이미지 형식: TIF (GeoTIFF)")
    print("   - 라벨 형식: YOLO Segmentation (polygon)")

    # 각 클래스별 데이터셋 생성
    results = {}

    for class_id in TARGET_CLASSES.keys():
        try:
            output_dir = create_dataset(class_id)
            results[class_id] = output_dir
        except Exception as e:
            print(f"\n❌ 클래스 {class_id} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()

    # 최종 요약
    print("\n" + "="*80)
    print("🎉 전체 데이터셋 생성 완료!")
    print("="*80)

    for class_id, output_dir in results.items():
        if output_dir:
            class_info = TARGET_CLASSES[class_id]
            print(f"\n✅ {class_info['name']}")
            print(f"   📁 {output_dir}/")
            print(f"   📄 {output_dir}/dataset.yaml")

    print("\n📝 다음 단계:")
    print("   1. 각 데이터셋의 통계 확인")
    print("   2. YOLOv11 seg 모델 학습:")
    for class_id, output_dir in results.items():
        if output_dir:
            print(f"      - yolo segment train data={output_dir}/dataset.yaml model=yolo11x-seg.pt")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
