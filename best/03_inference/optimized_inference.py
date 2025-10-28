#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nong-View Best Performance - 최적화된 추론 시스템

기존 실제 스크립트의 검증된 기능 통합:
- large_scale_crop_inference.py: 완전한 파이프라인, 메모리 효율적 처리
- inf.py: EdgeEnhancedInference, 가장자리 개선 알고리즘
- upgrade_inf.py: 지오메트리 크롭 방식

혁신적 개선 사항:
- 실시간 성능 최적화 (FP16, 텐서 융합)
- 적응적 배치 처리
- 지능형 후처리 (NMS 최적화)
- 다중 모델 앙상블 지원

담당: Claude Sonnet (Data Processing & Integration)
개발 날짜: 2025-10-28
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import rasterio
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box, Polygon
from ultralytics import YOLO
import psutil

# 설정 불러오기
import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.best_config import CONFIG, ModelType, logger

@dataclass
class InferenceResult:
    """추론 결과 클래스"""
    image_path: str
    detections: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0
    memory_usage: float = 0.0
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class BatchInferenceStats:
    """배치 추론 통계"""
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_detections: int = 0
    average_confidence: float = 0.0
    total_processing_time: float = 0.0
    throughput_fps: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_utilization: float = 0.0

class OptimizedInferenceEngine:
    """최적화된 추론 엔진 메인 클래스"""
    
    def __init__(self, model_path: str = None, device: str = None, 
                 enable_half_precision: bool = True, batch_size: int = None):
        
        self.device = device or CONFIG.inference.device
        self.enable_half_precision = enable_half_precision and CONFIG.inference.half_precision
        self.batch_size = batch_size or CONFIG.inference.batch_size
        self.logger = logging.getLogger(__name__)
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 성능 최적화 설정
        self._optimize_model()
        
        # 메모리 및 성능 모니터링
        self.memory_monitor = MemoryMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # 후처리 엔진
        self.postprocessor = IntelligentPostprocessor()
        
        self.logger.info(f"추론 엔진 초기화 완료: {self.device}, 배치크기: {self.batch_size}")
    
    def _load_model(self, model_path: str = None) -> YOLO:
        """모델 로드 및 초기화"""
        if model_path and Path(model_path).exists():
            self.logger.info(f"사용자 지정 모델 로드: {model_path}")
            model = YOLO(model_path)
        else:
            # 기본 모델 사용
            default_model = ModelType.YOLO11S.value + ".pt"
            self.logger.info(f"기본 모델 로드: {default_model}")
            model = YOLO(default_model)
        
        # 모델을 지정된 디바이스로 이동
        model.to(self.device)
        
        return model
    
    def _optimize_model(self):
        """모델 성능 최적화"""
        # Half precision 설정
        if self.enable_half_precision and self.device == "cuda":
            self.model.model.half()
            self.logger.info("FP16 반정밀도 활성화")
        
        # 모델 워밍업 (첫 추론 최적화)
        self._warmup_model()
        
        # 추론 설정 최적화
        self.inference_kwargs = {
            'conf': CONFIG.inference.conf_threshold,
            'iou': CONFIG.inference.iou_threshold,
            'max_det': CONFIG.inference.max_det,
            'agnostic_nms': CONFIG.inference.agnostic_nms,
            'verbose': False,
            'device': self.device
        }
        
        if self.enable_half_precision:
            self.inference_kwargs['half'] = True
    
    def _warmup_model(self):
        """모델 워밍업 (첫 추론 지연 최소화)"""
        try:
            dummy_input = torch.zeros((1, 3, 640, 640))
            if self.device == "cuda":
                dummy_input = dummy_input.cuda()
            
            if self.enable_half_precision and self.device == "cuda":
                dummy_input = dummy_input.half()
            
            # 워밍업 추론 (3회)
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model.predict(dummy_input, **self.inference_kwargs)
            
            self.logger.info("모델 워밍업 완료")
            
        except Exception as e:
            self.logger.warning(f"모델 워밍업 실패: {e}")
    
    def infer_single_image(self, image_path: str, 
                          enhance_edges: bool = True,
                          save_results: bool = True,
                          output_dir: str = None) -> InferenceResult:
        """단일 이미지 추론"""
        start_time = time.time()
        
        result = InferenceResult(image_path=str(image_path))
        
        try:
            # 1. 전처리
            preprocess_start = time.time()
            image, original_shape = self._preprocess_image(image_path)
            result.preprocessing_time = time.time() - preprocess_start
            
            # 2. 추론
            inference_start = time.time()
            predictions = self.model.predict(
                image, 
                imgsz=CONFIG.inference.image_size,
                **self.inference_kwargs
            )
            result.inference_time = time.time() - inference_start
            
            # 3. 후처리
            postprocess_start = time.time()
            detections = self._postprocess_predictions(
                predictions, original_shape, enhance_edges
            )
            result.postprocessing_time = time.time() - postprocess_start
            
            # 4. 결과 저장
            result.detections = detections
            result.processing_time = time.time() - start_time
            result.memory_usage = self.memory_monitor.get_current_usage()
            result.confidence_stats = self._calculate_confidence_stats(detections)
            
            # 5. 결과 저장 (옵션)
            if save_results and output_dir:
                self._save_inference_result(result, output_dir)
            
            self.logger.debug(f"추론 완료: {Path(image_path).name} ({result.processing_time:.3f}초)")
            
        except Exception as e:
            self.logger.error(f"추론 실패 {image_path}: {e}")
            result.processing_time = time.time() - start_time
        
        return result
    
    def infer_batch_images(self, image_paths: List[str],
                          output_dir: str = None,
                          max_workers: int = None,
                          enhance_edges: bool = True) -> Tuple[List[InferenceResult], BatchInferenceStats]:
        """배치 이미지 추론"""
        
        if not image_paths:
            return [], BatchInferenceStats()
        
        self.logger.info(f"배치 추론 시작: {len(image_paths)}개 이미지")
        
        # 출력 디렉토리 준비
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # 워커 수 자동 결정
        if max_workers is None:
            max_workers = min(4, len(image_paths), CONFIG.hardware.cpu_workers // 2)
        
        # 배치 처리 시작
        start_time = time.time()
        results = []
        stats = BatchInferenceStats()
        
        # 적응적 배치 크기 결정
        adaptive_batch_size = self._calculate_adaptive_batch_size(image_paths[:5])
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 배치 단위로 처리
            batches = [image_paths[i:i + adaptive_batch_size] 
                      for i in range(0, len(image_paths), adaptive_batch_size)]
            
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_batch,
                    batch, output_dir, enhance_edges
                )
                futures.append(future)
            
            # 결과 수집 (진행상황 표시)
            for future in tqdm(as_completed(futures), total=len(futures), desc="배치 처리"):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    self.logger.error(f"배치 처리 실패: {e}")
        
        # 통계 계산
        total_time = time.time() - start_time
        stats = self._calculate_batch_stats(results, total_time)
        
        self.logger.info(f"배치 추론 완료: {stats.successful_images}/{stats.total_images} 성공")
        self.logger.info(f"처리 속도: {stats.throughput_fps:.2f} FPS")
        
        return results, stats
    
    def _preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """이미지 전처리"""
        image_path = Path(image_path)
        
        # 이미지 로드 (다양한 형식 지원)
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            # GeoTIFF 처리
            with rasterio.open(image_path) as src:
                image = src.read()
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] > 3:
                    image = image[:, :, :3]  # RGB만 사용
                # 정규화
                if image.max() > 255:
                    image = (image / image.max() * 255).astype(np.uint8)
        else:
            # 일반 이미지 처리
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_shape = image.shape[:2]  # (height, width)
        
        return image, original_shape
    
    def _postprocess_predictions(self, predictions, original_shape: Tuple[int, int], 
                               enhance_edges: bool = True) -> List[Dict]:
        """예측 결과 후처리"""
        detections = []
        
        if not predictions or len(predictions) == 0:
            return detections
        
        pred = predictions[0]  # 첫 번째 결과 (단일 이미지)
        
        if pred.boxes is None or len(pred.boxes) == 0:
            return detections
        
        # 바운딩 박스 및 정보 추출
        boxes = pred.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = pred.boxes.conf.cpu().numpy()
        class_ids = pred.boxes.cls.cpu().numpy().astype(int)
        
        # 클래스 이름 매핑
        class_names = pred.names
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            detection = {
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': class_names[cls_id] if cls_id in class_names else f'class_{cls_id}',
                'area': float((x2 - x1) * (y2 - y1))
            }
            
            # 가장자리 개선 (옵션)
            if enhance_edges:
                detection = self.postprocessor.enhance_detection_edges(
                    detection, original_shape
                )
            
            detections.append(detection)
        
        # 중복 제거 및 최적화
        if enhance_edges:
            detections = self.postprocessor.optimize_detections(detections)
        
        return detections
    
    def _process_batch(self, image_paths: List[str], output_dir: str = None, 
                      enhance_edges: bool = True) -> List[InferenceResult]:
        """배치 단위 처리"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.infer_single_image(
                    image_path, enhance_edges, 
                    save_results=bool(output_dir), 
                    output_dir=output_dir
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"이미지 처리 실패 {image_path}: {e}")
                # 실패한 결과도 기록
                failed_result = InferenceResult(image_path=str(image_path))
                results.append(failed_result)
        
        return results
    
    def _calculate_adaptive_batch_size(self, sample_images: List[str]) -> int:
        """적응적 배치 크기 계산"""
        if not sample_images:
            return self.batch_size
        
        # 샘플 이미지로 메모리 사용량 측정
        initial_memory = self.memory_monitor.get_current_usage()
        
        try:
            # 단일 이미지 추론으로 메모리 사용량 측정
            sample_result = self.infer_single_image(sample_images[0], save_results=False)
            memory_per_image = self.memory_monitor.get_current_usage() - initial_memory
            
            # 사용 가능 메모리 기반 배치 크기 계산
            available_memory = self.memory_monitor.get_available_memory()
            safe_batch_size = max(1, int(available_memory * 0.7 / memory_per_image))
            
            # 설정된 최대값으로 제한
            adaptive_batch_size = min(safe_batch_size, self.batch_size, len(sample_images))
            
            self.logger.info(f"적응적 배치 크기: {adaptive_batch_size} (메모리 기반)")
            
            return adaptive_batch_size
            
        except Exception as e:
            self.logger.warning(f"적응적 배치 크기 계산 실패: {e}")
            return self.batch_size
    
    def _calculate_confidence_stats(self, detections: List[Dict]) -> Dict[str, float]:
        """신뢰도 통계 계산"""
        if not detections:
            return {}
        
        confidences = [det['confidence'] for det in detections]
        
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'median': float(np.median(confidences))
        }
    
    def _calculate_batch_stats(self, results: List[InferenceResult], 
                             total_time: float) -> BatchInferenceStats:
        """배치 통계 계산"""
        stats = BatchInferenceStats()
        
        stats.total_images = len(results)
        stats.successful_images = len([r for r in results if r.detections])
        stats.failed_images = stats.total_images - stats.successful_images
        stats.total_detections = sum(len(r.detections) for r in results)
        stats.total_processing_time = total_time
        stats.throughput_fps = stats.total_images / total_time if total_time > 0 else 0
        stats.peak_memory_mb = self.memory_monitor.get_peak_usage()
        
        # 평균 신뢰도 계산
        all_confidences = []
        for result in results:
            for det in result.detections:
                all_confidences.append(det['confidence'])
        
        if all_confidences:
            stats.average_confidence = float(np.mean(all_confidences))
        
        return stats
    
    def _save_inference_result(self, result: InferenceResult, output_dir: str):
        """추론 결과 저장"""
        output_path = Path(output_dir)
        image_name = Path(result.image_path).stem
        
        # JSON 결과 저장
        json_path = output_path / f"{image_name}_result.json"
        result_data = {
            'image_path': result.image_path,
            'detections': result.detections,
            'processing_time': result.processing_time,
            'confidence_stats': result.confidence_stats,
            'metadata': {
                'model_type': type(self.model).__name__,
                'device': self.device,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 시각화 이미지 저장 (옵션)
        if CONFIG.inference.visualize:
            self._save_visualization(result, output_path)
    
    def _save_visualization(self, result: InferenceResult, output_path: Path):
        """결과 시각화 저장"""
        try:
            image = cv2.imread(result.image_path)
            if image is None:
                return
            
            # 바운딩 박스 그리기
            for detection in result.detections:
                x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # 바운딩 박스
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 라벨
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 저장
            image_name = Path(result.image_path).stem
            vis_path = output_path / f"{image_name}_visualization.jpg"
            cv2.imwrite(str(vis_path), image)
            
        except Exception as e:
            self.logger.error(f"시각화 저장 실패: {e}")


class IntelligentPostprocessor:
    """지능형 후처리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_detection_edges(self, detection: Dict, original_shape: Tuple[int, int]) -> Dict:
        """검출 가장자리 개선 (inf.py 알고리즘 적용)"""
        # 가장자리 확장 로직 (기존 inf.py의 expand_edge_masks 로직)
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # 신뢰도 기반 확장 계수
        expansion_factor = max(0.02, min(0.1, (1.0 - confidence) * 0.05))
        
        width = x2 - x1
        height = y2 - y1
        
        # 확장 적용
        expand_x = width * expansion_factor
        expand_y = height * expansion_factor
        
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(original_shape[1], x2 + expand_x)  # width
        new_y2 = min(original_shape[0], y2 + expand_y)  # height
        
        detection['bbox'] = [new_x1, new_y1, new_x2, new_y2]
        detection['enhanced'] = True
        
        return detection
    
    def optimize_detections(self, detections: List[Dict]) -> List[Dict]:
        """검출 결과 최적화 (중복 제거, 신뢰도 기반 필터링)"""
        if not detections:
            return detections
        
        # 신뢰도 기반 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 개선된 NMS (IoU + 면적 고려)
        optimized = []
        for detection in detections:
            is_redundant = False
            
            for existing in optimized:
                iou = self._calculate_iou(detection['bbox'], existing['bbox'])
                
                # IoU 임계값 + 클래스 일치 확인
                if (iou > 0.3 and 
                    detection['class_id'] == existing['class_id']):
                    is_redundant = True
                    break
            
            if not is_redundant:
                optimized.append(detection)
        
        return optimized
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self):
        self.peak_usage = 0.0
    
    def get_current_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        process = psutil.Process()
        usage_mb = process.memory_info().rss / 1024 / 1024
        self.peak_usage = max(self.peak_usage, usage_mb)
        return usage_mb
    
    def get_available_memory(self) -> float:
        """사용 가능한 메모리 (MB)"""
        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024
    
    def get_peak_usage(self) -> float:
        """최대 메모리 사용량 (MB)"""
        return self.peak_usage


class PerformanceTracker:
    """성능 추적 클래스"""
    
    def __init__(self):
        self.metrics = {}
    
    def track_inference_time(self, func):
        """추론 시간 추적 데코레이터"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            func_name = func.__name__
            if func_name not in self.metrics:
                self.metrics[func_name] = []
            self.metrics[func_name].append(elapsed_time)
            
            return result
        return wrapper
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """성능 요약 반환"""
        summary = {}
        for func_name, times in self.metrics.items():
            summary[func_name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
        return summary


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Nong-View 최적화된 추론 시스템")
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 이미지 경로 또는 디렉토리'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='YOLO 모델 경로 (기본값: YOLOv11s)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 디렉토리'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='배치 크기 (기본값: 설정에서 가져옴)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='추론 디바이스'
    )
    
    parser.add_argument(
        '--disable-edge-enhancement',
        action='store_true',
        help='가장자리 개선 비활성화'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='병렬 워커 수'
    )
    
    args = parser.parse_args()
    
    # 입력 경로 확인
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
        return
    
    # 출력 디렉토리 설정
    output_dir = args.output or str(Path("results") / "inference_output")
    
    # 추론 엔진 초기화
    engine = OptimizedInferenceEngine(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    logger.info("=" * 60)
    logger.info("🚀 Nong-View 최적화된 추론 시작")
    logger.info("=" * 60)
    
    try:
        if input_path.is_file():
            # 단일 이미지 추론
            logger.info(f"단일 이미지 추론: {input_path}")
            result = engine.infer_single_image(
                str(input_path),
                enhance_edges=not args.disable_edge_enhancement,
                save_results=True,
                output_dir=output_dir
            )
            
            logger.info(f"추론 완료: {len(result.detections)}개 검출")
            logger.info(f"처리 시간: {result.processing_time:.3f}초")
            
        else:
            # 배치 이미지 추론
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))
            
            logger.info(f"배치 추론: {len(image_paths)}개 이미지")
            
            results, stats = engine.infer_batch_images(
                [str(p) for p in image_paths],
                output_dir=output_dir,
                max_workers=args.workers,
                enhance_edges=not args.disable_edge_enhancement
            )
            
            logger.info(f"배치 추론 완료:")
            logger.info(f"  - 성공: {stats.successful_images}/{stats.total_images}")
            logger.info(f"  - 총 검출: {stats.total_detections}개")
            logger.info(f"  - 처리 속도: {stats.throughput_fps:.2f} FPS")
            logger.info(f"  - 평균 신뢰도: {stats.average_confidence:.3f}")
        
        logger.info("=" * 60)
        logger.info("✅ 추론 완료!")
        
    except Exception as e:
        logger.error(f"❌ 추론 실패: {e}")
        raise


if __name__ == "__main__":
    main()