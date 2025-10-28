#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가장자리 검출 개선 추론 스크립트
마스크 가장자리 확장 후처리 적용
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
from datetime import datetime
import time

class EdgeEnhancedInference:
    def __init__(self, model_path, dataset_path='production_dataset_balanced'):
        """초기화"""
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.test_images_path = self.dataset_path / 'images' / 'test'
        self.test_labels_path = self.dataset_path / 'labels' / 'test'
        
        # 클래스 정보
        self.class_names = ['IRG_production', 'Rye_production', 
                           'Corn_production', 'SudanGrass_production']
        self.class_colors = plt.cm.rainbow(np.linspace(0, 1, len(self.class_names)))
        
        # 모델 로드
        self.model = YOLO(self.model_path)
        print(f"✅ 모델 로드 완료: {self.model_path}")
    
    def expand_edge_masks(self, masks, image_shape, expand_pixels=25, iterations=3):
        """가장자리 마스크 균형잡힌 확장 처리"""
        h, w = image_shape[:2]
        expanded_masks = []
        
        for mask in masks:
            mask_np = mask.cpu().numpy()
            
            # 원본 마스크 크기를 이미지 크기로 리사이즈
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 이진화 (적절한 임계값)
            mask_binary = (mask_resized > 0.3).astype(np.uint8)
            
            # 중간 크기 커널
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            
            # 적당한 확장
            mask_dilated = cv2.dilate(mask_binary, kernel_small, iterations=iterations)
            
            # 가장자리별 제한적 처리
            edge_check_distance = 20  # 가장자리에서 20픽셀 내에 마스크가 있는지 확인
            edge_extend_distance = 10  # 실제 확장은 10픽셀까지만
            
            # 상단 가장자리 처리 (조건부)
            if np.any(mask_dilated[:edge_check_distance, :] > 0):
                # 상단 가장자리 근처에만 마스크가 많이 있을 때만 확장
                top_density = np.mean(mask_dilated[:edge_check_distance, :])
                if top_density > 0.5:  # 50% 이상 채워져 있을 때만
                    top_area = mask_dilated[:edge_extend_distance, :]
                    top_expanded = cv2.dilate(top_area, kernel_medium, iterations=2)
                    mask_dilated[:edge_extend_distance, :] = np.maximum(
                        mask_dilated[:edge_extend_distance, :], top_expanded
                    )
                    # 최상단 5픽셀만 채우기
                    if top_density > 0.7:  # 70% 이상일 때만
                        mask_dilated[:5, :] = np.maximum(mask_dilated[:5, :], 
                                                         mask_dilated[5:10, :])
            
            # 하단 가장자리 처리 (조건부)
            if np.any(mask_dilated[-edge_check_distance:, :] > 0):
                bottom_density = np.mean(mask_dilated[-edge_check_distance:, :])
                if bottom_density > 0.5:
                    bottom_area = mask_dilated[-edge_extend_distance:, :]
                    bottom_expanded = cv2.dilate(bottom_area, kernel_medium, iterations=2)
                    mask_dilated[-edge_extend_distance:, :] = np.maximum(
                        mask_dilated[-edge_extend_distance:, :], bottom_expanded
                    )
                    if bottom_density > 0.7:
                        mask_dilated[-5:, :] = np.maximum(mask_dilated[-5:, :], 
                                                          mask_dilated[-10:-5, :])
            
            # 좌측 가장자리 처리 (조건부)
            if np.any(mask_dilated[:, :edge_check_distance] > 0):
                left_density = np.mean(mask_dilated[:, :edge_check_distance])
                if left_density > 0.5:
                    left_area = mask_dilated[:, :edge_extend_distance]
                    left_expanded = cv2.dilate(left_area, kernel_medium, iterations=2)
                    mask_dilated[:, :edge_extend_distance] = np.maximum(
                        mask_dilated[:, :edge_extend_distance], left_expanded
                    )
                    if left_density > 0.7:
                        mask_dilated[:, :5] = np.maximum(mask_dilated[:, :5], 
                                                         mask_dilated[:, 5:10])
            
            # 우측 가장자리 처리 (조건부)
            if np.any(mask_dilated[:, -edge_check_distance:] > 0):
                right_density = np.mean(mask_dilated[:, -edge_check_distance:])
                if right_density > 0.5:
                    right_area = mask_dilated[:, -edge_extend_distance:]
                    right_expanded = cv2.dilate(right_area, kernel_medium, iterations=2)
                    mask_dilated[:, -edge_extend_distance:] = np.maximum(
                        mask_dilated[:, -edge_extend_distance:], right_expanded
                    )
                    if right_density > 0.7:
                        mask_dilated[:, -5:] = np.maximum(mask_dilated[:, -5:], 
                                                          mask_dilated[:, -10:-5])
            
            # Closing 연산으로 작은 구멍 채우기
            mask_closed = cv2.morphologyEx(mask_dilated.astype(np.float32), 
                                          cv2.MORPH_CLOSE, kernel_small)
            
            # 부드럽게 처리 (약하게)
            mask_enhanced = cv2.GaussianBlur(mask_closed, (3, 3), 1)
            
            # 값 범위 조정 (0-1)
            mask_enhanced = np.clip(mask_enhanced, 0, 1)
            
            expanded_masks.append(mask_enhanced)
        
        return expanded_masks
    
    def predict_with_edge_fix(self, image_path, conf=0.25, show_comparison=True):
        """가장자리 개선 추론"""
        
        # 이미지 로드
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        print(f"\n🖼️ 추론 중: {Path(image_path).name}")
        
        # 1. 추론 실행 (augment 제거)
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=0.45,  # 약간 낮춰서 더 많은 검출
            imgsz=1024,
            retina_masks=True,  # 고해상도 마스크
            verbose=False
        )
        
        result = results[0]
        
        # 2. 마스크 후처리 적용
        if result.masks is not None and len(result.masks) > 0:
            print(f"   검출: {len(result.masks)}개 객체")
            
            # 원본 마스크 백업
            original_masks = result.masks.data.clone()
            
            # 가장자리 확장 적용 (균형잡힌 설정)
            expanded_masks = self.expand_edge_masks(
                result.masks.data.clone(),  # clone 사용
                img.shape,
                expand_pixels=25,  # 25픽셀까지 확장 (중간)
                iterations=3  # 적당한 반복
            )
            
            # 확장된 마스크로 새로운 텐서 생성 (inplace 수정 대신)
            new_masks = []
            for exp_mask in expanded_masks:
                new_masks.append(torch.from_numpy(exp_mask).to(result.masks.data.device))
            
            # 새로운 텐서로 교체
            result.masks.data = torch.stack(new_masks)
            
            if show_comparison:
                self.visualize_before_after(img_rgb, original_masks, result.masks.data, 
                                           result.boxes, Path(image_path).name)
        else:
            print("   검출된 객체 없음")
        
        return result
    
    def visualize_before_after(self, img, original_masks, enhanced_masks, boxes, image_name):
        """개선 전후 비교 시각화"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 원본 이미지
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # 2. 개선 전 마스크
        overlay_before = img.copy()
        for i, mask in enumerate(original_masks):
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
            cls_id = int(boxes.cls[i])
            
            mask_bool = mask_resized > 0.5
            color = (self.class_colors[cls_id][:3] * 255).astype(int)
            overlay_before[mask_bool] = overlay_before[mask_bool] * 0.5 + color * 0.5
        
        axes[1].imshow(overlay_before.astype(np.uint8))
        axes[1].set_title('Before Edge Enhancement', fontsize=12)
        axes[1].axis('off')
        
        # 3. 개선 후 마스크
        overlay_after = img.copy()
        for i, mask in enumerate(enhanced_masks):
            mask_np = mask.cpu().numpy()
            if len(mask_np.shape) == 3:
                mask_np = mask_np.squeeze()
            
            # 이미 expand_edge_masks에서 리사이즈됨
            cls_id = int(boxes.cls[i])
            
            mask_bool = mask_np > 0.5
            color = (self.class_colors[cls_id][:3] * 255).astype(int)
            overlay_after[mask_bool] = overlay_after[mask_bool] * 0.5 + color * 0.5
        
        axes[2].imshow(overlay_after.astype(np.uint8))
        axes[2].set_title('After Edge Enhancement', fontsize=12)
        axes[2].axis('off')
        
        fig.suptitle(f'Edge Enhancement: {image_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def process_test_set(self, num_samples=10, save_dir='edge_enhanced_results'):
        """Test 셋 처리 및 비교"""
        
        print("\n" + "="*60)
        print("🔧 가장자리 개선 추론")
        print("="*60)
        
        # Test 이미지 목록
        test_images = list(self.test_images_path.glob('*.jpg'))[:num_samples]
        
        # 저장 디렉토리
        save_path = Path(save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 {len(test_images)}개 이미지 처리")
        
        # GT와 비교를 위한 통계
        edge_improvements = []
        
        for i, img_path in enumerate(test_images, 1):
            print(f"\n[{i}/{len(test_images)}] {img_path.name}")
            
            # 가장자리 개선 추론
            result = self.predict_with_edge_fix(
                img_path, 
                conf=0.25,
                show_comparison=False  # 시각화 비활성화 (빠른 처리)
            )
            
            # 결과 저장
            if result.masks is not None:
                # 마스크 이미지 저장
                img = cv2.imread(str(img_path))
                overlay = img.copy()
                
                for j, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    if len(mask_np.shape) == 3:
                        mask_np = mask_np.squeeze()
                    
                    cls_id = int(result.boxes.cls[j])
                    mask_bool = mask_np > 0.5
                    color = (self.class_colors[cls_id][:3] * 255).astype(int)
                    overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
                
                save_file = save_path / f'enhanced_{img_path.name}'
                cv2.imwrite(str(save_file), overlay)
                
                # 가장자리 픽셀 개선도 측정
                edge_pixels = self.measure_edge_coverage(mask_np)
                edge_improvements.append(edge_pixels)
                print(f"   가장자리 커버리지: {edge_pixels:.1f}%")
        
        print(f"\n📊 전체 결과:")
        print(f"   평균 가장자리 커버리지: {np.mean(edge_improvements):.1f}%")
        print(f"   결과 저장: {save_path}")
    
    def measure_edge_coverage(self, mask, edge_width=10):
        """가장자리 커버리지 측정"""
        h, w = mask.shape[:2]
        
        # 가장자리 영역의 마스크 비율 계산
        edge_coverage = []
        
        # 각 가장자리 확인
        top_coverage = np.mean(mask[:edge_width, :] > 0.5) * 100
        bottom_coverage = np.mean(mask[-edge_width:, :] > 0.5) * 100
        left_coverage = np.mean(mask[:, :edge_width] > 0.5) * 100
        right_coverage = np.mean(mask[:, -edge_width:] > 0.5) * 100
        
        # 평균 커버리지
        avg_coverage = np.mean([top_coverage, bottom_coverage, left_coverage, right_coverage])
        
        return avg_coverage

def main():
    """메인 실행 - 하드코딩된 버전"""
    
    print("\n🚀 가장자리 개선 추론 시작")
    
    # 설정 (하드코딩)
    model_path = 'runs/quick_test/quick_test_5ep_20250930_0135/weights/best.pt'
    
    # 추론 엔진 생성
    engine = EdgeEnhancedInference(model_path)
    
    # Test 셋 20개 이미지 자동 처리
    print("\n📌 Test 셋 20개 이미지 자동 처리 시작...")
    engine.process_test_set(num_samples=20)

if __name__ == "__main__":
    main()