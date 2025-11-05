"""
Segmentation Mask Fusion Ensemble System
여러 Segmentation 모델의 예측을 결합하여 더 정확한 결과 생성

Author: Claude Sonnet
Date: 2025-11-04
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import json

from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    model_paths: List[str] = field(default_factory=list)
    model_weights: List[float] = field(default_factory=list)  # 각 모델의 가중치
    fusion_method: str = "weighted_average"  # weighted_average, voting, union, intersection
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SegmentationEnsemble:
    """Segmentation Mask Fusion 앙상블"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        
        # 가중치 기본값 설정 (균등)
        if not config.model_weights:
            config.model_weights = [1.0 / len(config.model_paths)] * len(config.model_paths)
        
        # 가중치 정규화
        total_weight = sum(config.model_weights)
        self.weights = [w / total_weight for w in config.model_weights]
        
        # 모델 로드
        self._load_models()
        
        logger.info(f"✓ {len(self.models)}개 모델 로드 완료")
        logger.info(f"✓ Fusion 방법: {config.fusion_method}")
        logger.info(f"✓ 모델 가중치: {[f'{w:.3f}' for w in self.weights]}")
    
    def _load_models(self):
        """모든 모델 로드"""
        for model_path in self.config.model_paths:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일 없음: {model_path}")
            
            model = YOLO(model_path)
            model.to(self.config.device)
            self.models.append(model)
            logger.info(f"  - 로드: {Path(model_path).name}")
    
    def predict_single(self, image_path: str, save: bool = False, 
                      save_dir: Optional[str] = None) -> Dict:
        """단일 이미지 예측"""
        # 각 모델의 예측
        all_predictions = []
        
        for i, model in enumerate(self.models):
            results = model.predict(
                image_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                verbose=False
            )[0]
            
            all_predictions.append({
                'boxes': results.boxes,
                'masks': results.masks,
                'model_idx': i,
                'weight': self.weights[i]
            })
        
        # 마스크 융합
        fused_result = self._fuse_masks(all_predictions, image_path)
        
        # 저장
        if save and save_dir:
            self._save_result(fused_result, image_path, save_dir)
        
        return fused_result
    
    def predict_batch(self, image_paths: List[str], save_dir: Optional[str] = None) -> List[Dict]:
        """배치 예측"""
        results = []
        
        for img_path in tqdm(image_paths, desc="앙상블 예측"):
            result = self.predict_single(img_path, save=True, save_dir=save_dir)
            results.append(result)
        
        return results
    
    def _fuse_masks(self, predictions: List[Dict], image_path: str) -> Dict:
        """마스크 융합"""
        # 이미지 크기 가져오기
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        if self.config.fusion_method == "weighted_average":
            return self._fuse_weighted_average(predictions, h, w)
        elif self.config.fusion_method == "voting":
            return self._fuse_voting(predictions, h, w)
        elif self.config.fusion_method == "union":
            return self._fuse_union(predictions, h, w)
        elif self.config.fusion_method == "intersection":
            return self._fuse_intersection(predictions, h, w)
        else:
            raise ValueError(f"알 수 없는 fusion 방법: {self.config.fusion_method}")
    
    def _fuse_weighted_average(self, predictions: List[Dict], h: int, w: int) -> Dict:
        """가중 평균 방식의 마스크 융합"""
        # 클래스별로 마스크 누적
        class_masks = {}  # {class_id: accumulated_mask}
        class_boxes = {}  # {class_id: list of boxes}
        class_confidences = {}  # {class_id: list of confidences}
        
        for pred in predictions:
            if pred['masks'] is None:
                continue
            
            weight = pred['weight']
            boxes = pred['boxes']
            masks = pred['masks']
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                box = boxes.xyxy[i].cpu().numpy()
                
                # 마스크를 이미지 크기로 리사이즈
                mask = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # 클래스별로 누적
                if cls not in class_masks:
                    class_masks[cls] = np.zeros((h, w), dtype=np.float32)
                    class_boxes[cls] = []
                    class_confidences[cls] = []
                
                class_masks[cls] += mask_resized * weight * conf
                class_boxes[cls].append((box, conf, weight))
                class_confidences[cls].append(conf * weight)
        
        # 최종 결과 생성
        fused_masks = []
        fused_boxes = []
        fused_classes = []
        fused_confidences = []
        
        for cls, accumulated_mask in class_masks.items():
            # 임계값 적용
            binary_mask = (accumulated_mask > 0.5).astype(np.uint8)
            
            if binary_mask.sum() == 0:
                continue
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # 가장 큰 컨투어 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 바운딩 박스 계산
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            box = np.array([x, y, x + w_box, y + h_box], dtype=np.float32)
            
            # 평균 confidence
            avg_conf = np.mean(class_confidences[cls])
            
            fused_masks.append(accumulated_mask)
            fused_boxes.append(box)
            fused_classes.append(cls)
            fused_confidences.append(avg_conf)
        
        return {
            'masks': fused_masks,
            'boxes': fused_boxes,
            'classes': fused_classes,
            'confidences': fused_confidences,
            'fusion_method': 'weighted_average',
            'num_models': len(predictions)
        }
    
    def _fuse_voting(self, predictions: List[Dict], h: int, w: int) -> Dict:
        """투표 방식의 마스크 융합 (픽셀별 다수결)"""
        # 클래스별로 투표 맵 생성
        class_votes = {}
        
        for pred in predictions:
            if pred['masks'] is None:
                continue
            
            boxes = pred['boxes']
            masks = pred['masks']
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # confidence가 낮으면 건너뛰기
                if conf < self.config.conf_threshold:
                    continue
                
                mask = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                if cls not in class_votes:
                    class_votes[cls] = np.zeros((h, w), dtype=np.int32)
                
                # 마스크가 있는 픽셀에 투표
                class_votes[cls] += (mask_resized > 0.5).astype(np.int32)
        
        # 과반수 이상 투표된 마스크만 유지
        threshold_votes = len(predictions) // 2 + 1
        
        fused_masks = []
        fused_boxes = []
        fused_classes = []
        fused_confidences = []
        
        for cls, votes in class_votes.items():
            # 과반수 투표
            binary_mask = (votes >= threshold_votes).astype(np.uint8)
            
            if binary_mask.sum() == 0:
                continue
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            box = np.array([x, y, x + w_box, y + h_box], dtype=np.float32)
            
            # Confidence는 투표 비율
            conf = votes.max() / len(predictions)
            
            fused_masks.append(binary_mask.astype(np.float32))
            fused_boxes.append(box)
            fused_classes.append(cls)
            fused_confidences.append(conf)
        
        return {
            'masks': fused_masks,
            'boxes': fused_boxes,
            'classes': fused_classes,
            'confidences': fused_confidences,
            'fusion_method': 'voting',
            'num_models': len(predictions)
        }
    
    def _fuse_union(self, predictions: List[Dict], h: int, w: int) -> Dict:
        """합집합 방식의 마스크 융합"""
        class_masks = {}
        
        for pred in predictions:
            if pred['masks'] is None:
                continue
            
            boxes = pred['boxes']
            masks = pred['masks']
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                mask = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                if cls not in class_masks:
                    class_masks[cls] = np.zeros((h, w), dtype=np.uint8)
                
                # OR 연산 (합집합)
                class_masks[cls] = np.logical_or(class_masks[cls], binary_mask).astype(np.uint8)
        
        fused_masks = []
        fused_boxes = []
        fused_classes = []
        fused_confidences = []
        
        for cls, mask in class_masks.items():
            if mask.sum() == 0:
                continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            box = np.array([x, y, x + w_box, y + h_box], dtype=np.float32)
            
            fused_masks.append(mask.astype(np.float32))
            fused_boxes.append(box)
            fused_classes.append(cls)
            fused_confidences.append(1.0)  # Union은 confidence 의미 없음
        
        return {
            'masks': fused_masks,
            'boxes': fused_boxes,
            'classes': fused_classes,
            'confidences': fused_confidences,
            'fusion_method': 'union',
            'num_models': len(predictions)
        }
    
    def _fuse_intersection(self, predictions: List[Dict], h: int, w: int) -> Dict:
        """교집합 방식의 마스크 융합 (가장 보수적)"""
        class_masks = {}
        class_counts = {}
        
        for pred in predictions:
            if pred['masks'] is None:
                continue
            
            boxes = pred['boxes']
            masks = pred['masks']
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                mask = masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                
                if cls not in class_masks:
                    class_masks[cls] = np.ones((h, w), dtype=np.uint8)
                    class_counts[cls] = 0
                
                # AND 연산 (교집합)
                class_masks[cls] = np.logical_and(class_masks[cls], binary_mask).astype(np.uint8)
                class_counts[cls] += 1
        
        fused_masks = []
        fused_boxes = []
        fused_classes = []
        fused_confidences = []
        
        for cls, mask in class_masks.items():
            # 모든 모델이 동의한 영역만 유지
            if class_counts[cls] < len(predictions):
                continue
            
            if mask.sum() == 0:
                continue
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            box = np.array([x, y, x + w_box, y + h_box], dtype=np.float32)
            
            fused_masks.append(mask.astype(np.float32))
            fused_boxes.append(box)
            fused_classes.append(cls)
            fused_confidences.append(1.0)
        
        return {
            'masks': fused_masks,
            'boxes': fused_boxes,
            'classes': fused_classes,
            'confidences': fused_confidences,
            'fusion_method': 'intersection',
            'num_models': len(predictions)
        }
    
    def _save_result(self, result: Dict, image_path: str, save_dir: str):
        """결과 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 로드
        img = cv2.imread(image_path)
        overlay = img.copy()
        
        # 클래스 색상
        colors = [
            (0, 255, 0),    # Green for class 0
            (255, 0, 0),    # Blue for class 1
            (0, 0, 255),    # Red for class 2
        ]
        
        # 마스크 그리기
        for i, (mask, box, cls, conf) in enumerate(zip(
            result['masks'], result['boxes'], result['classes'], result['confidences']
        )):
            color = colors[cls % len(colors)]
            
            # 마스크 적용
            mask_binary = (mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            colored_mask[mask_binary > 0] = color
            
            # 반투명 오버레이
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
            
            # 바운딩 박스
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # 라벨
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 저장
        filename = Path(image_path).stem
        output_path = save_path / f"{filename}_ensemble.png"
        cv2.imwrite(str(output_path), overlay)
        
        # 메타데이터 저장
        meta_path = save_path / f"{filename}_ensemble.json"
        meta = {
            'fusion_method': result['fusion_method'],
            'num_models': result['num_models'],
            'num_detections': len(result['masks']),
            'classes': [int(c) for c in result['classes']],
            'confidences': [float(c) for c in result['confidences']]
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def evaluate(self, test_images: List[str], ground_truth_dir: str) -> Dict:
        """앙상블 성능 평가"""
        # TODO: 구현 필요
        logger.warning("evaluate 메서드는 아직 구현되지 않았습니다.")
        return {}


def create_ensemble_from_training(training_result_dirs: List[str], 
                                  fusion_method: str = "weighted_average",
                                  model_weights: Optional[List[float]] = None) -> SegmentationEnsemble:
    """학습 결과로부터 앙상블 생성"""
    model_paths = []
    
    for result_dir in training_result_dirs:
        result_path = Path(result_dir)
        
        # best.pt 찾기
        best_model = result_path / "train" / "weights" / "best.pt"
        if not best_model.exists():
            # 다른 경로 시도
            best_model = result_path / "best.pt"
        
        if best_model.exists():
            model_paths.append(str(best_model))
            logger.info(f"✓ 모델 발견: {best_model}")
        else:
            logger.warning(f"⚠ 모델 없음: {result_dir}")
    
    if not model_paths:
        raise FileNotFoundError("사용 가능한 모델이 없습니다.")
    
    config = EnsembleConfig(
        model_paths=model_paths,
        model_weights=model_weights if model_weights else [],
        fusion_method=fusion_method
    )
    
    return SegmentationEnsemble(config)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Segmentation Mask Fusion Ensemble")
    logger.info("=" * 80)
    
    # 예제: 3개 모델로 앙상블 생성
    # 실제 사용 시 학습된 모델 경로로 변경
    model_paths = [
        r"results/training_20251104_120000/train/weights/best.pt",
        r"results/training_20251104_130000/train/weights/best.pt",
        r"results/training_20251104_140000/train/weights/best.pt",
    ]
    
    # 모델 가중치 (mAP50 기준으로 설정 가능)
    # 예: Model1(0.85), Model2(0.90), Model3(0.88) mAP50
    model_weights = [0.85, 0.90, 0.88]
    
    config = EnsembleConfig(
        model_paths=model_paths,
        model_weights=model_weights,
        fusion_method="weighted_average",  # or "voting", "union", "intersection"
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    try:
        ensemble = SegmentationEnsemble(config)
        
        # 테스트 이미지 예측
        test_images = [
            r"path/to/test/image1.png",
            r"path/to/test/image2.png"
        ]
        
        results = ensemble.predict_batch(
            test_images, 
            save_dir="results/ensemble_predictions"
        )
        
        logger.info(f"\n✅ {len(results)}개 이미지 앙상블 예측 완료!")
        
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("\n💡 사용법:")
        logger.info("1. 여러 모델을 학습하세요")
        logger.info("2. 학습된 모델 경로를 model_paths에 설정하세요")
        logger.info("3. 스크립트를 실행하세요")

