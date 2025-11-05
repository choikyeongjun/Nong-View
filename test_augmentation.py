#!/usr/bin/env python3
"""증강 테스트 스크립트"""

import cv2
import numpy as np
from pathlib import Path

# 테스트 이미지 로드
test_img_path = Path(r"C:\Users\LX\Nong-View\model3_greenhouse\images\train\1F001D40001.png")
test_label_path = Path(r"C:\Users\LX\Nong-View\model3_greenhouse\labels\train\1F001D40001.txt")

print(f"이미지 경로: {test_img_path}")
print(f"이미지 존재: {test_img_path.exists()}")
print(f"라벨 경로: {test_label_path}")
print(f"라벨 존재: {test_label_path.exists()}")

# 이미지 로드 테스트
if test_img_path.exists():
    img = cv2.imread(str(test_img_path))
    if img is not None:
        print(f"✓ 이미지 로드 성공: {img.shape}")
    else:
        print("✗ 이미지 로드 실패")

# 라벨 로드 테스트
if test_label_path.exists():
    with open(test_label_path, 'r') as f:
        lines = f.readlines()
    print(f"✓ 라벨 로드 성공: {len(lines)}개 객체")
    for i, line in enumerate(lines[:3]):
        print(f"  {i+1}: {line.strip()}")

# 증강 테스트
print("\n증강 테스트 중...")
if test_img_path.exists() and test_label_path.exists():
    # 라벨 파싱
    bboxes = []
    class_labels = []
    with open(test_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)

    print(f"  Bboxes: {len(bboxes)}개")
    print(f"  Classes: {class_labels}")

    # 간단한 증강 (좌우 반전)
    img = cv2.imread(str(test_img_path))
    aug_img = cv2.flip(img, 1)
    aug_bboxes = [[1.0 - bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes]

    # 저장 테스트
    test_output = Path(r"C:\Users\LX\Nong-View\test_aug.png")
    success = cv2.imwrite(str(test_output), aug_img)
    print(f"  이미지 저장: {'✓' if success else '✗'} {test_output}")

    test_label_output = Path(r"C:\Users\LX\Nong-View\test_aug.txt")
    with open(test_label_output, 'w') as f:
        for bbox, cls in zip(aug_bboxes, class_labels):
            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    print(f"  라벨 저장: ✓ {test_label_output}")

    print("\n✓ 증강 테스트 완료!")
