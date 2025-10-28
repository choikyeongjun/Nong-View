#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
import gc
import warnings
warnings.filterwarnings('ignore')

import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Polygon

from ultralytics import YOLO
import cv2
import torch
import torch.nn.functional as F

import psutil
from tqdm import tqdm
import time

from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.morphology import remove_small_objects, remove_small_holes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_path: str = "best.pt"
    device: str = "auto"
    batch_size: int = 8
    conf_threshold: float = 0.05
    iou_threshold: float = 0.45
    tile_size: int = 1024
    overlap_ratio: float = 0.5
    min_tile_size: int = 512
    max_workers: int = field(default_factory=lambda: min(8, os.cpu_count()))
    memory_limit_gb: float = 16.0
    cache_dir: str = "cache"
    min_area_pixels: int = 100
    morphology_kernel_size: int = 5
    smoothing_sigma: float = 1.0
    
    # 실제 학습된 4개 클래스만 정의
    class_info: Dict[int, Tuple[str, str]] = field(default_factory=lambda: {
        0: ("IRG(생육기)", "#4CAF50"),      # 초록색
        1: ("호밀(생육기)", "#FFC107"),      # 황색
        2: ("옥수수(생육기)", "#2196F3"),    # 파란색
        3: ("수단그라스(생육기)", "#9C27B0") # 보라색
    })

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class SimpleCache:
    def __init__(self, cache_dir="cache", enabled=True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def get_key(self, *args):
        key_str = '_'.join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def exists(self, key):
        if not self.enabled:
            return False
        return (self.cache_dir / f"{key}.pkl").exists()
    
    def load(self, key):
        if not self.enabled:
            return None
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def save(self, key, data):
        if not self.enabled:
            return
        try:
            with open(self.cache_dir / f"{key}.pkl", 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

class TileProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.step_size = int(config.tile_size * (1 - config.overlap_ratio))
    
    def generate_tiles(self, shape):
        h, w = shape[:2]
        tiles = []
        
        if h <= self.config.tile_size and w <= self.config.tile_size:
            tiles.append((0, 0, w, h))
            return tiles
        
        for y in range(0, h, self.step_size):
            for x in range(0, w, self.step_size):
                tile_w = min(self.config.tile_size, w - x)
                tile_h = min(self.config.tile_size, h - y)
                
                if tile_w < self.config.min_tile_size:
                    x = max(0, w - self.config.tile_size)
                    tile_w = min(self.config.tile_size, w)
                
                if tile_h < self.config.min_tile_size:
                    y = max(0, h - self.config.tile_size)
                    tile_h = min(self.config.tile_size, h)
                
                tiles.append((x, y, tile_w, tile_h))
        
        return tiles
    
    def merge_predictions(self, predictions, coords, shape):
        h, w = shape[:2]
        class_masks = {}
        class_weights = {}
        
        for pred, (x, y, tw, th) in zip(predictions, coords):
            for cls_id, mask in pred.items():
                if cls_id not in class_masks:
                    class_masks[cls_id] = np.zeros((h, w), dtype=np.float32)
                    class_weights[cls_id] = np.zeros((h, w), dtype=np.float32)
                
                if mask.shape[:2] != (th, tw):
                    mask = cv2.resize(mask.astype(np.float32), (tw, th))
                
                weight = self._create_weight_map(th, tw)
                class_masks[cls_id][y:y+th, x:x+tw] += mask * weight
                class_weights[cls_id][y:y+th, x:x+tw] += weight
        
        results = {}
        for cls_id in class_masks:
            mask = np.divide(class_masks[cls_id], class_weights[cls_id], 
                           where=class_weights[cls_id] > 0)
            mask = (mask > self.config.conf_threshold).astype(np.uint8)
            if np.any(mask):
                results[cls_id] = mask
        
        return results
    
    def _create_weight_map(self, h, w):
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        
        sigma = min(h, w) / 4
        weight = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
        
        fade_ratio = 0.1
        fade_pixels = int(min(w, h) * fade_ratio)
        for i in range(fade_pixels):
            alpha = i / fade_pixels
            weight[i, :] *= alpha
            weight[-i-1, :] *= alpha
            weight[:, i] *= alpha
            weight[:, -i-1] *= alpha
        
        return weight

class PostProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size)
        )
    
    def refine_mask(self, mask):
        mask = binary_fill_holes(mask)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = remove_small_objects(mask.astype(bool), min_size=self.config.min_area_pixels)
        mask = remove_small_holes(mask, area_threshold=self.config.min_area_pixels)
        
        if self.config.smoothing_sigma > 0:
            mask = gaussian_filter(mask.astype(float), sigma=self.config.smoothing_sigma)
            mask = (mask > 0.5).astype(np.uint8)
        
        return mask

class YOLOInference:
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self._load_model()
    
    def _setup_device(self):
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _load_model(self):
        try:
            self.model = YOLO(self.config.model_path)
            if 'cuda' in self.device:
                self.model.to(self.device)
                torch.backends.cudnn.benchmark = True
            logger.info(f"모델 로드 완료: {self.config.model_path} on {self.device}")
            logger.info(f"탐지 클래스: {', '.join([info[0] for info in self.config.class_info.values()])}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def predict_batch(self, images):
        if not images:
            return []
        
        try:
            results = self.model.predict(
                images,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                device=self.device,
                batch=min(len(images), self.config.batch_size),
                verbose=False
            )
            
            batch_predictions = []
            for result in results:
                pred_dict = {}
                
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
                    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                    
                    h, w = result.orig_shape[:2]
                    
                    for mask, cls, conf in zip(masks, classes, confidences):
                        cls = int(cls)
                        
                        # 학습된 클래스만 처리
                        if cls not in self.config.class_info:
                            logger.warning(f"알 수 없는 클래스 {cls} 무시")
                            continue
                        
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask.astype(np.float32), (w, h))
                        
                        if cls not in pred_dict or conf > pred_dict[cls][1]:
                            pred_dict[cls] = (mask, conf)
                    
                    for cls in pred_dict:
                        pred_dict[cls] = pred_dict[cls][0]
                
                batch_predictions.append(pred_dict)
            
            return batch_predictions
        except Exception as e:
            logger.error(f"배치 추론 실패: {e}")
            return [{}] * len(images)

class Pipeline:
    def __init__(self):
        self.config = Config()
        self.setup_directories()
        
        self.cache = SimpleCache(self.config.cache_dir)
        self.tile_processor = TileProcessor(self.config)
        self.post_processor = PostProcessor(self.config)
        self.yolo = YOLOInference(self.config)
    
    def setup_directories(self):
        for dir_name in ['input', 'gpkg', 'output', 'cache', 'temp']:
            Path(dir_name).mkdir(exist_ok=True)
    
    def check_memory(self):
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        if available_gb < 2.0:
            logger.warning(f"메모리 부족: {available_gb:.2f}GB")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_polygon(self, tif_path, polygon, polygon_id):
        cache_key = self.cache.get_key(tif_path, polygon.wkt, polygon_id)
        
        if self.cache.exists(cache_key):
            logger.info(f"캐시에서 로드: {polygon_id}")
            return self.cache.load(cache_key)
        
        try:
            bounds = polygon.bounds
            
            with rasterio.open(tif_path) as src:
                window = rasterio.windows.from_bounds(*bounds, src.transform)
                height = int(window.height)
                width = int(window.width)
                
                tiles = self.tile_processor.generate_tiles((height, width))
                
                all_predictions = []
                all_coords = []
                batch_images = []
                batch_coords = []
                
                for x, y, tw, th in tiles:
                    tile_window = Window(
                        window.col_off + x,
                        window.row_off + y,
                        tw, th
                    )
                    
                    tile_data = src.read(window=tile_window)
                    if tile_data.shape[0] > 3:
                        tile_data = tile_data[:3]
                    tile_image = np.transpose(tile_data, (1, 2, 0))
                    
                    batch_images.append(tile_image)
                    batch_coords.append((x, y, tw, th))
                    
                    if len(batch_images) >= self.config.batch_size:
                        predictions = self.yolo.predict_batch(batch_images)
                        all_predictions.extend(predictions)
                        all_coords.extend(batch_coords)
                        batch_images = []
                        batch_coords = []
                        self.check_memory()
                
                if batch_images:
                    predictions = self.yolo.predict_batch(batch_images)
                    all_predictions.extend(predictions)
                    all_coords.extend(batch_coords)
                
                merged_results = self.tile_processor.merge_predictions(
                    all_predictions, all_coords, (height, width)
                )
                
                final_results = {}
                for cls_id, mask in merged_results.items():
                    refined_mask = self.post_processor.refine_mask(mask)
                    if np.any(refined_mask):
                        final_results[cls_id] = refined_mask
                
                self.cache.save(cache_key, final_results)
                
                return final_results
                
        except Exception as e:
            logger.error(f"폴리곤 {polygon_id} 처리 실패: {e}")
            return None
    
    def save_results(self, results, polygon, output_path, tif_path, bounds):
        if not results:
            return
        
        try:
            with rasterio.open(tif_path) as src:
                window = rasterio.windows.from_bounds(*bounds, src.transform)
                transform = rasterio.windows.transform(window, src.transform)
                crs = src.crs
            
            geometries = []
            
            for cls_id, mask in results.items():
                from rasterio.features import shapes
                
                for geom, value in shapes(mask.astype(np.uint8), transform=transform):
                    if value == 1:
                        class_name, color = self.config.class_info.get(cls_id, (f"Class_{cls_id}", "#000000"))
                        
                        geometries.append({
                            'geometry': Polygon(geom['coordinates'][0]),
                            'class_id': cls_id,
                            'class_name': class_name,
                            'color': color,
                            'area_m2': Polygon(geom['coordinates'][0]).area
                        })
            
            if geometries:
                gdf = gpd.GeoDataFrame(geometries, crs=crs)
                gdf.to_file(output_path, driver='GPKG')
                logger.info(f"결과 저장: {output_path} - {len(geometries)}개 객체")
        
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    def run(self):
        logger.info("="*50)
        logger.info("YOLOv11 농작물 세그멘테이션 파이프라인 시작")
        logger.info(f"탐지 대상: {', '.join([info[0] for info in self.config.class_info.values()])}")
        logger.info("="*50)
        
        tif_files = list(Path("input").glob("*.tif")) + list(Path("input").glob("*.tiff"))
        gpkg_files = list(Path("gpkg").glob("*.gpkg"))
        
        if not tif_files or not gpkg_files:
            logger.error("입력 파일이 없습니다.")
            logger.error(f"TIF 파일은 'input/' 폴더에, GPKG 파일은 'gpkg/' 폴더에 넣어주세요.")
            return
        
        logger.info(f"TIF 파일: {len(tif_files)}개, GPKG 파일: {len(gpkg_files)}개")
        
        tasks = []
        for tif_file in tif_files:
            for gpkg_file in gpkg_files:
                gdf = gpd.read_file(gpkg_file)
                for idx, row in gdf.iterrows():
                    tasks.append({
                        'tif_path': str(tif_file),
                        'polygon': row.geometry,
                        'polygon_id': f"{tif_file.stem}_{gpkg_file.stem}_{idx}"
                    })
        
        logger.info(f"총 {len(tasks)}개 작업")
        
        with tqdm(total=len(tasks), desc="처리 진행률") as pbar:
            if self.config.max_workers > 1:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    for task in tasks:
                        future = executor.submit(self._process_task, task)
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"작업 실패: {e}")
                        pbar.update(1)
            else:
                for task in tasks:
                    self._process_task(task)
                    pbar.update(1)
        
        logger.info("="*50)
        logger.info("파이프라인 완료!")
        logger.info("결과는 'output/' 폴더에서 확인하세요.")
        logger.info("="*50)
    
    def _process_task(self, task):
        results = self.process_polygon(
            task['tif_path'],
            task['polygon'],
            task['polygon_id']
        )
        
        if results:
            output_path = Path("output") / f"{task['polygon_id']}.gpkg"
            self.save_results(
                results,
                task['polygon'],
                str(output_path),
                task['tif_path'],
                task['polygon'].bounds
            )

def main():
    pipeline = Pipeline()
    pipeline.run()

if __name__ == "__main__":
    main()