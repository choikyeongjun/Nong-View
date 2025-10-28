#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 YOLO 데이터셋 생성 스크립트
모든 폴더의 이미지와 라벨을 수집하여 train:val:test = 8:1:1로 분할
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

def find_image_label_pairs():
    """모든 폴더에서 이미지-라벨 쌍을 찾아 반환"""
    
    # 소스 폴더 정의
    source_folders = [
        {
            'name': 'data',
            'image_path': 'data/image',
            'label_path': 'data/label/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250827_납품데이터',
            'image_path': '250827_납품데이터/이미지데이터',
            'label_path': '250827_납품데이터/라벨링데이터/Pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250903-데이터',
            'image_path': '250903-데이터/이미지데이터',
            'label_path': '250903-데이터/라벨링데이터/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250910-최종납품',
            'image_path': '250910-최종납품/이미지데이터',
            'label_path': '250910-최종납품/라벨링데이터/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250911-최종납품',
            'image_path': '250911-최종납품/이미지데이터',
            'label_path': '250911-최종납품/라벨링데이터/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250917-최종납품',
            'image_path': '250917-최종납품/이미지데이터',
            'label_path': '250917-최종납품/라벨링데이터/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        },
        {
            'name': '250918-최종납품',
            'image_path': '250918-최종납품/이미지데이터',
            'label_path': '250918-최종납품/라벨링데이터/pixeljson',
            'image_ext': '.tif',
            'label_ext': '.txt'
        }
    ]
    
    all_pairs = []
    
    for folder_info in source_folders:
        print(f"\n📁 {folder_info['name']} 폴더 처리 중...")
        
        image_dir = Path(folder_info['image_path'])
        label_dir = Path(folder_info['label_path'])
        
        if not image_dir.exists():
            print(f"   ⚠️  이미지 폴더가 존재하지 않습니다: {image_dir}")
            continue
            
        if not label_dir.exists():
            print(f"   ⚠️  라벨 폴더가 존재하지 않습니다: {label_dir}")
            continue
        
        # 이미지 파일 목록 수집
        image_files = {}
        for img_file in image_dir.glob(f"*{folder_info['image_ext']}"):
            base_name = img_file.stem
            image_files[base_name] = img_file
        
        # 라벨 파일 목록 수집
        label_files = {}
        for label_file in label_dir.glob(f"*{folder_info['label_ext']}"):
            base_name = label_file.stem
            label_files[base_name] = label_file
        
        # 매칭되는 쌍 찾기
        matched_pairs = 0
        for base_name in image_files:
            if base_name in label_files:
                all_pairs.append({
                    'base_name': base_name,
                    'image_path': image_files[base_name],
                    'label_path': label_files[base_name],
                    'source_folder': folder_info['name']
                })
                matched_pairs += 1
        
        print(f"   ✅ 이미지: {len(image_files)}개, 라벨: {len(label_files)}개, 매칭: {matched_pairs}개")
    
    return all_pairs

def split_dataset(pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """데이터셋을 train:val:test로 분할"""
    
    # 랜덤 셔플
    random.shuffle(pairs)
    
    total = len(pairs)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]
    
    print(f"\n📊 데이터셋 분할 결과:")
    print(f"   🚂 Train: {len(train_pairs)}개 ({len(train_pairs)/total*100:.1f}%)")
    print(f"   🔍 Val: {len(val_pairs)}개 ({len(val_pairs)/total*100:.1f}%)")
    print(f"   🧪 Test: {len(test_pairs)}개 ({len(test_pairs)/total*100:.1f}%)")
    print(f"   📊 Total: {total}개")
    
    return train_pairs, val_pairs, test_pairs

def copy_files(pairs, split_name, output_dir):
    """파일들을 해당 분할 폴더로 복사"""
    
    image_dir = output_dir / 'images' / split_name
    label_dir = output_dir / 'labels' / split_name
    
    print(f"\n📋 {split_name.upper()} 데이터 복사 중... ({len(pairs)}개)")
    
    for i, pair in enumerate(pairs):
        # 이미지 파일 복사 (확장자를 .jpg로 변경)
        src_image = pair['image_path']
        dst_image = image_dir / f"{pair['base_name']}.jpg"
        
        # TIFF를 JPG로 변환하여 복사 (PIL 사용)
        try:
            from PIL import Image
            with Image.open(src_image) as img:
                # RGBA를 RGB로 변환 (필요한 경우)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(dst_image, 'JPEG', quality=95)
        except Exception as e:
            print(f"   ⚠️  이미지 변환 실패 {src_image}: {e}")
            # 실패 시 원본 복사
            shutil.copy2(src_image, dst_image)
        
        # 라벨 파일 복사
        src_label = pair['label_path']
        dst_label = label_dir / f"{pair['base_name']}.txt"
        shutil.copy2(src_label, dst_label)
        
        if (i + 1) % 500 == 0:
            print(f"   진행률: {i+1}/{len(pairs)} ({(i+1)/len(pairs)*100:.1f}%)")
    
    print(f"   ✅ {split_name.upper()} 완료: {len(pairs)}개 파일 복사")

def create_yaml_config(output_dir, class_names):
    """YOLO 학습용 YAML 설정 파일 생성"""
    
    config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n📄 YAML 설정 파일 생성: {yaml_path}")
    return yaml_path

def analyze_class_distribution(pairs):
    """클래스 분포 분석"""
    
    class_counts = defaultdict(int)
    
    for pair in pairs:
        try:
            with open(pair['label_path'], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
        except Exception as e:
            print(f"   ⚠️  라벨 파일 읽기 실패 {pair['label_path']}: {e}")
    
    return class_counts

def main():
    """메인 함수"""
    
    print("🚀 통합 YOLO 데이터셋 생성 시작")
    print("=" * 50)
    
    # 랜덤 시드 설정
    random.seed(42)
    
    # 출력 디렉토리
    output_dir = Path('unified_yolo_dataset')
    
    # 1. 모든 이미지-라벨 쌍 수집
    print("\n1️⃣ 데이터 수집 중...")
    all_pairs = find_image_label_pairs()
    
    if not all_pairs:
        print("❌ 매칭되는 이미지-라벨 쌍을 찾을 수 없습니다!")
        return
    
    print(f"\n✅ 총 {len(all_pairs)}개의 이미지-라벨 쌍을 찾았습니다.")
    
    # 2. 클래스 분포 분석
    print("\n2️⃣ 클래스 분포 분석 중...")
    class_counts = analyze_class_distribution(all_pairs)
    
    class_names = [
        'IRG(생육기)',      # 0 -> 1
        'IRG(생산기)',      # 1 -> 2  
        '호밀(생육기)',     # 2 -> 3
        '호밀(생산기)',     # 3 -> 4
        '옥수수(생육기)',   # 4 -> 5
        '옥수수(생산기)',   # 5 -> 6
        '수단그라스(생육기)', # 6 -> 7
        '수단그라스(생산기)', # 7 -> 8
        '비닐하우스(단동)', # 8 -> 9
        '비닐하우스(다동)', # 9 -> 10
        '곤포사일리지',     # 10 -> 11
        '경작지',          # 11 -> 12
        '비경작지'         # 12 -> 13
    ]
    
    print("\n📊 클래스별 객체 수:")
    for class_id in sorted(class_counts.keys()):
        if class_id < len(class_names):
            print(f"   {class_id+1:2d}. {class_names[class_id]:15s}: {class_counts[class_id]:,}개")
        else:
            print(f"   {class_id+1:2d}. {'Unknown':15s}: {class_counts[class_id]:,}개")
    
    # 3. 데이터셋 분할
    print("\n3️⃣ 데이터셋 분할 중...")
    train_pairs, val_pairs, test_pairs = split_dataset(all_pairs)
    
    # 4. 파일 복사
    print("\n4️⃣ 파일 복사 중...")
    copy_files(train_pairs, 'train', output_dir)
    copy_files(val_pairs, 'val', output_dir)
    copy_files(test_pairs, 'test', output_dir)
    
    # 5. YAML 설정 파일 생성
    print("\n5️⃣ YAML 설정 파일 생성 중...")
    yaml_path = create_yaml_config(output_dir, class_names)
    
    print("\n" + "=" * 50)
    print("🎉 통합 YOLO 데이터셋 생성 완료!")
    print(f"📁 출력 폴더: {output_dir.absolute()}")
    print(f"📄 설정 파일: {yaml_path}")
    print("\n📋 최종 구조:")
    print(f"   🚂 Train: {len(train_pairs):,}개")
    print(f"   🔍 Val: {len(val_pairs):,}개") 
    print(f"   🧪 Test: {len(test_pairs):,}개")
    print(f"   📊 Total: {len(all_pairs):,}개")

if __name__ == "__main__":
    main()
