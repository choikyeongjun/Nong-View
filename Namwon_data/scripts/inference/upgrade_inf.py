#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, field
import gc
import warnings
warnings.filterwarnings('ignore')

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon, box

from ultralytics import YOLO
import cv2
import torch

import psutil
from tqdm import tqdm

from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.morphology import remove_small_objects, remove_small_holes

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """설정 클래스"""
    model_path: str = "best.pt"
    device: str = "auto"
    conf_threshold: float = 0.05
    iou_threshold: float = 0.45
    min_area_pixels: int = 100
    morphology_kernel_size: int = 5
    smoothing_sigma: float = 1.0
    use_cache: bool = False  # 캐시 사용 여부 (기본값: False)
    
    # 학습된 4개 클래스 정의
    class_info: Dict[int, Tuple[str, str]] = field(default_factory=lambda: {
        0: ("IRG(생육기)", "#4CAF50"),      # 초록색
        1: ("호밀(생육기)", "#FFC107"),      # 황색
        2: ("옥수수(생육기)", "#2196F3"),    # 파란색
        3: ("수단그라스(생육기)", "#9C27B0") # 보라색
    })

class PostProcessor:
    """마스크 후처리 클래스"""
    def __init__(self, config: Config):
        self.config = config
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size)
        )
    
    def refine_mask(self, mask):
        """마스크 정제"""
        # 구멍 채우기
        mask = binary_fill_holes(mask)
        
        # 모폴로지 연산
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # 작은 객체 제거
        mask = remove_small_objects(mask.astype(bool), min_size=self.config.min_area_pixels)
        mask = remove_small_holes(mask, area_threshold=self.config.min_area_pixels)
        
        # 스무딩
        if self.config.smoothing_sigma > 0:
            mask = gaussian_filter(mask.astype(float), sigma=self.config.smoothing_sigma)
            mask = (mask > 0.5).astype(np.uint8)
        
        return mask

class YOLOInference:
    """YOLO 추론 클래스"""
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self._load_model()
    
    def _setup_device(self):
        """디바이스 설정"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("CPU 사용")
            return device
        return self.config.device
    
    def _load_model(self):
        """모델 로드"""
        try:
            logger.info(f"모델 로딩 중: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            
            if 'cuda' in self.device:
                self.model.to(self.device)
                torch.backends.cudnn.benchmark = True
                
            logger.info("✅ 모델 로드 완료")
            logger.info(f"탐지 클래스: {', '.join([info[0] for info in self.config.class_info.values()])}")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def predict(self, image):
        """단일 이미지 추론"""
        try:
            # YOLO 추론
            results = self.model.predict(
                image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            predictions = {}
            confidences = {}
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
                    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                    
                    h, w = result.orig_shape[:2]
                    
                    for mask, cls, conf in zip(masks, classes, confs):
                        cls = int(cls)
                        
                        # 학습된 클래스만 처리
                        if cls not in self.config.class_info:
                            continue
                        
                        # 마스크 리사이즈
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        
                        # 클래스별 최고 신뢰도 마스크 저장
                        if cls not in predictions or conf > confidences[cls]:
                            predictions[cls] = mask
                            confidences[cls] = float(conf)
            
            return predictions, confidences
            
        except Exception as e:
            logger.error(f"추론 실패: {e}")
            return {}, {}

class CropSegmentationPipeline:
    """메인 파이프라인 클래스"""
    def __init__(self):
        self.config = Config()
        self.setup_directories()
        self.post_processor = PostProcessor(self.config)
        self.yolo = YOLOInference(self.config)
        
        # 통계 초기화
        self.stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
    
    def setup_directories(self):
        """필수 디렉토리 생성"""
        directories = ['input', 'gpkg', 'output']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
        
        # 캐시 디렉토리 정리 (있으면)
        cache_dir = Path('cache')
        if cache_dir.exists() and not self.config.use_cache:
            logger.info("캐시 디렉토리 정리 중...")
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    def check_polygon_overlap(self, tif_path, polygon):
        """폴리곤과 래스터의 겹침 확인"""
        try:
            with rasterio.open(tif_path) as src:
                # 래스터 경계 상자
                raster_bounds = box(*src.bounds)
                
                # 폴리곤과 래스터의 교집합 확인
                if polygon.intersects(raster_bounds):
                    intersection = polygon.intersection(raster_bounds)
                    # 교집합 면적이 충분한지 확인
                    if intersection.area > 0:
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"범위 확인 실패: {e}")
            return False
    
    def crop_and_segment(self, tif_path, polygon, polygon_id):
        """폴리곤 영역을 크롭하고 세그멘테이션 수행"""
        try:
            # 폴리곤이 래스터 범위 내에 있는지 확인
            if not self.check_polygon_overlap(tif_path, polygon):
                logger.warning(f"⚠️ {polygon_id}: 래스터 범위 밖 - 건너뜀")
                self.stats['skipped'] += 1
                return None
            
            with rasterio.open(tif_path) as src:
                # 폴리곤 형태로 크롭 (지오메트리 마스킹)
                try:
                    out_image, out_transform = mask(src, [polygon], crop=True, filled=True)
                except ValueError as e:
                    logger.warning(f"⚠️ {polygon_id}: 크롭 실패 - {e}")
                    self.stats['skipped'] += 1
                    return None
                
                # RGB 채널만 사용
                if out_image.shape[0] > 3:
                    out_image = out_image[:3]
                
                # CHW -> HWC 변환
                cropped_image = np.transpose(out_image, (1, 2, 0))
                
                # 유효 픽셀 마스크
                if src.nodata is not None:
                    valid_mask = ~(out_image[0] == src.nodata)
                else:
                    valid_mask = np.ones(out_image.shape[1:], dtype=bool)
                
                # 유효 픽셀이 없으면 건너뛰기
                if not np.any(valid_mask):
                    logger.warning(f"⚠️ {polygon_id}: 유효 픽셀 없음 - 건너뜀")
                    self.stats['skipped'] += 1
                    return None
                
                # YOLO 추론
                logger.debug(f"추론 중: {polygon_id}")
                predictions, confidences = self.yolo.predict(cropped_image)
                
                if not predictions:
                    logger.info(f"ℹ️ {polygon_id}: 탐지된 객체 없음")
                    self.stats['processed'] += 1
                    return None
                
                # 후처리 및 결과 정리
                results = []
                for cls_id, pred_mask in predictions.items():
                    # 폴리곤 영역 외부 마스킹
                    pred_mask = pred_mask * valid_mask.astype(np.float32)
                    
                    # 마스크 정제
                    refined_mask = self.post_processor.refine_mask(pred_mask)
                    
                    if np.any(refined_mask):
                        # 지오메트리 변환
                        from rasterio.features import shapes
                        
                        for geom, value in shapes(refined_mask.astype(np.uint8), transform=out_transform):
                            if value == 1:
                                class_name, color = self.config.class_info[cls_id]
                                poly = Polygon(geom['coordinates'][0])
                                
                                results.append({
                                    'geometry': poly,
                                    'class_id': cls_id,
                                    'class_name': class_name,
                                    'confidence': round(confidences[cls_id], 4),
                                    'color': color,
                                    'area_m2': round(poly.area, 2)
                                })
                
                if results:
                    # GeoDataFrame 생성 및 저장
                    gdf = gpd.GeoDataFrame(results, crs=src.crs)
                    output_path = Path("output") / f"{polygon_id}.gpkg"
                    gdf.to_file(output_path, driver='GPKG')
                    logger.info(f"✅ {polygon_id}: {len(results)}개 객체 저장")
                    self.stats['processed'] += 1
                else:
                    logger.info(f"ℹ️ {polygon_id}: 후처리 후 객체 없음")
                    self.stats['processed'] += 1
                
                return results
                
        except Exception as e:
            logger.error(f"❌ {polygon_id} 처리 실패: {e}")
            self.stats['failed'] += 1
            return None
    
    def check_memory(self):
        """메모리 체크 및 정리"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        if available_gb < 2.0:
            logger.warning(f"⚠️ 메모리 부족: {available_gb:.2f}GB - 정리 중...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run(self):
        """파이프라인 실행"""
        print("\n" + "="*60)
        print(" YOLOv11 농작물 세그멘테이션 파이프라인 ")
        print(" (지오메트리 크롭 버전) ")
        print("="*60 + "\n")
        
        # 입력 파일 확인
        tif_files = list(Path("input").glob("*.tif")) + list(Path("input").glob("*.tiff"))
        gpkg_files = list(Path("gpkg").glob("*.gpkg"))
        
        if not tif_files:
            logger.error("❌ TIF 파일이 없습니다. 'input/' 폴더에 TIF 파일을 넣어주세요.")
            return
        
        if not gpkg_files:
            logger.error("❌ GPKG 파일이 없습니다. 'gpkg/' 폴더에 GPKG 파일을 넣어주세요.")
            return
        
        logger.info(f"📁 입력 파일: TIF {len(tif_files)}개, GPKG {len(gpkg_files)}개")
        
        # 작업 목록 생성
        tasks = []
        logger.info("작업 목록 생성 중...")
        
        for tif_file in tif_files:
            logger.info(f"  - {tif_file.name}")
            for gpkg_file in gpkg_files:
                logger.info(f"    + {gpkg_file.name}")
                try:
                    gdf = gpd.read_file(gpkg_file)
                    for idx, row in gdf.iterrows():
                        tasks.append({
                            'tif_path': str(tif_file),
                            'tif_name': tif_file.stem,
                            'gpkg_name': gpkg_file.stem,
                            'polygon': row.geometry,
                            'polygon_id': f"{tif_file.stem}_{gpkg_file.stem}_{idx:03d}"
                        })
                except Exception as e:
                    logger.error(f"    ❌ GPKG 읽기 실패: {e}")
        
        self.stats['total'] = len(tasks)
        logger.info(f"\n📊 총 {len(tasks)}개 작업 시작\n")
        
        # 작업 실행
        start_time = datetime.now()
        
        with tqdm(total=len(tasks), desc="전체 진행률", unit="polygon") as pbar:
            for i, task in enumerate(tasks, 1):
                # 진행 상황 업데이트
                pbar.set_postfix({
                    '처리': self.stats['processed'],
                    '건너뜀': self.stats['skipped'],
                    '실패': self.stats['failed']
                })
                
                # 작업 실행
                self.crop_and_segment(
                    task['tif_path'],
                    task['polygon'],
                    task['polygon_id']
                )
                
                # 진행률 업데이트
                pbar.update(1)
                
                # 주기적 메모리 정리
                if i % 10 == 0:
                    self.check_memory()
        
        # 완료 보고
        elapsed_time = datetime.now() - start_time
        
        print("\n" + "="*60)
        print(" 처리 완료! ")
        print("="*60)
        print(f"\n📊 처리 결과:")
        print(f"  - 전체: {self.stats['total']}개")
        print(f"  - 처리됨: {self.stats['processed']}개")
        print(f"  - 건너뜀: {self.stats['skipped']}개")
        print(f"  - 실패: {self.stats['failed']}개")
        print(f"  - 소요 시간: {elapsed_time}")
        print(f"\n📁 결과 파일: output/ 폴더")
        print("="*60 + "\n")

def main():
    """메인 함수"""
    try:
        # 파이프라인 실행
        pipeline = CropSegmentationPipeline()
        pipeline.run()
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ 사용자에 의해 중단됨")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()