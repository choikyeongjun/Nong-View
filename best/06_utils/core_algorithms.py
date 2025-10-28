"""
Core Algorithms for Nong-View Best Performance
Author: Claude Opus (System Architect & Core Algorithms)
Date: 2025-10-28
Version: 1.0.0

Core algorithmic implementations for:
- Advanced tiling strategies
- Optimized inference pipelines
- Intelligent result merging
- Spatial indexing and optimization
- Performance-critical utilities
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import cv2
import torch
from shapely.geometry import box, Polygon
from shapely.strtree import STRtree
from rtree import index
import rasterio
from rasterio.windows import Window
from collections import defaultdict
import logging
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
import itertools

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a single tile"""
    x: int
    y: int
    width: int
    height: int
    row: int
    col: int
    image_id: str
    overlap_ratio: float = 0.2
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get tile bounds (minx, miny, maxx, maxy)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    @property
    def polygon(self) -> Polygon:
        """Get tile as shapely polygon"""
        return box(*self.bounds)


@dataclass
class Detection:
    """Single detection result"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    tile_info: Optional[TileInfo] = None
    
    @property
    def area(self) -> float:
        """Calculate bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bbox"""
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def polygon(self) -> Polygon:
        """Get detection as shapely polygon"""
        return box(*self.bbox)


class AdvancedTilingStrategy:
    """Advanced tiling algorithms for optimal coverage and performance"""
    
    def __init__(
        self,
        tile_size: int = 640,
        overlap: float = 0.2,
        min_tile_size: int = 320,
        adaptive: bool = True
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tile_size = min_tile_size
        self.adaptive = adaptive
        
    def generate_tiles(
        self,
        image_width: int,
        image_height: int,
        roi: Optional[Polygon] = None
    ) -> List[TileInfo]:
        """Generate optimal tile layout for image"""
        
        if self.adaptive and roi:
            return self._adaptive_tiling(image_width, image_height, roi)
        else:
            return self._regular_tiling(image_width, image_height)
    
    def _regular_tiling(
        self,
        width: int,
        height: int
    ) -> List[TileInfo]:
        """Generate regular grid tiling with overlap"""
        tiles = []
        stride = int(self.tile_size * (1 - self.overlap))
        
        row = 0
        for y in range(0, height, stride):
            col = 0
            for x in range(0, width, stride):
                # Calculate actual tile dimensions
                tile_width = min(self.tile_size, width - x)
                tile_height = min(self.tile_size, height - y)
                
                # Skip tiles that are too small
                if tile_width < self.min_tile_size or tile_height < self.min_tile_size:
                    # Extend previous tile if possible
                    if tiles and tiles[-1].row == row:
                        tiles[-1].width = width - tiles[-1].x
                    continue
                
                tile = TileInfo(
                    x=x, y=y,
                    width=tile_width,
                    height=tile_height,
                    row=row, col=col,
                    image_id="",
                    overlap_ratio=self.overlap
                )
                tiles.append(tile)
                col += 1
            row += 1
        
        return tiles
    
    def _adaptive_tiling(
        self,
        width: int,
        height: int,
        roi: Polygon
    ) -> List[TileInfo]:
        """Generate adaptive tiling based on ROI"""
        tiles = []
        
        # Get ROI bounds
        minx, miny, maxx, maxy = roi.bounds
        roi_width = maxx - minx
        roi_height = maxy - miny
        
        # Calculate optimal tile size for ROI
        optimal_tile_size = self._calculate_optimal_tile_size(roi_width, roi_height)
        
        # Generate tiles covering ROI
        stride = int(optimal_tile_size * (1 - self.overlap))
        
        row = 0
        for y in np.arange(miny, maxy, stride):
            col = 0
            for x in np.arange(minx, maxx, stride):
                # Create candidate tile
                tile_poly = box(x, y, 
                              min(x + optimal_tile_size, maxx),
                              min(y + optimal_tile_size, maxy))
                
                # Check intersection with ROI
                intersection = tile_poly.intersection(roi)
                if intersection.area / tile_poly.area > 0.3:  # At least 30% overlap
                    tile = TileInfo(
                        x=int(x), y=int(y),
                        width=int(min(optimal_tile_size, maxx - x)),
                        height=int(min(optimal_tile_size, maxy - y)),
                        row=row, col=col,
                        image_id="",
                        overlap_ratio=self.overlap
                    )
                    tiles.append(tile)
                    col += 1
            row += 1
        
        return tiles
    
    def _calculate_optimal_tile_size(self, roi_width: float, roi_height: float) -> int:
        """Calculate optimal tile size based on ROI dimensions"""
        # Find tile size that minimizes waste
        min_dim = min(roi_width, roi_height)
        
        if min_dim <= self.tile_size:
            return int(min_dim)
        
        # Find divisor that gives minimal remainder
        best_size = self.tile_size
        min_waste = float('inf')
        
        for size in range(self.tile_size, self.min_tile_size, -32):
            waste_w = roi_width % size
            waste_h = roi_height % size
            total_waste = waste_w + waste_h
            
            if total_waste < min_waste:
                min_waste = total_waste
                best_size = size
        
        return best_size


class OptimizedInferencePipeline:
    """Optimized inference pipeline with batching and caching"""
    
    def __init__(
        self,
        batch_size: int = 8,
        use_amp: bool = True,
        cache_size: int = 100,
        device: str = 'cuda'
    ):
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.cache_size = cache_size
        self.device = device
        self.cache = {}
        
    def batch_inference(
        self,
        model: Any,
        tiles: List[np.ndarray],
        tile_infos: List[TileInfo]
    ) -> List[Detection]:
        """Perform batched inference on tiles"""
        all_detections = []
        
        # Process in batches
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles = tiles[i:i + self.batch_size]
            batch_infos = tile_infos[i:i + self.batch_size]
            
            # Stack tiles for batch processing
            batch_tensor = self._prepare_batch(batch_tiles)
            
            # Check cache
            batch_key = self._compute_batch_key(batch_tensor)
            if batch_key in self.cache:
                batch_results = self.cache[batch_key]
            else:
                # Run inference
                batch_results = self._run_inference(model, batch_tensor)
                
                # Update cache
                if len(self.cache) < self.cache_size:
                    self.cache[batch_key] = batch_results
            
            # Process results
            for result, tile_info in zip(batch_results, batch_infos):
                detections = self._process_result(result, tile_info)
                all_detections.extend(detections)
        
        return all_detections
    
    def _prepare_batch(self, tiles: List[np.ndarray]) -> torch.Tensor:
        """Prepare batch tensor from tiles"""
        # Normalize and convert to tensor
        batch = []
        for tile in tiles:
            # Resize if needed
            if tile.shape[:2] != (self.batch_size, self.batch_size):
                tile = cv2.resize(tile, (640, 640))
            
            # Normalize
            tile = tile.astype(np.float32) / 255.0
            
            # Convert to CHW format
            tile = np.transpose(tile, (2, 0, 1))
            batch.append(tile)
        
        batch_tensor = torch.from_numpy(np.stack(batch))
        return batch_tensor.to(self.device)
    
    def _compute_batch_key(self, batch: torch.Tensor) -> str:
        """Compute cache key for batch"""
        # Simple hash based on tensor shape and sum
        return f"{batch.shape}_{batch.sum().item():.2f}"
    
    def _run_inference(self, model: Any, batch: torch.Tensor) -> List:
        """Run model inference on batch"""
        with torch.no_grad():
            if self.use_amp and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    results = model(batch)
            else:
                results = model(batch)
        
        return results
    
    def _process_result(
        self,
        result: Any,
        tile_info: TileInfo
    ) -> List[Detection]:
        """Process single inference result to detections"""
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                # Convert to global coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Adjust for tile position
                x1 += tile_info.x
                y1 += tile_info.y
                x2 += tile_info.x
                y2 += tile_info.y
                
                detection = Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=str(box.cls[0]),  # Would map to actual name
                    tile_info=tile_info
                )
                detections.append(detection)
        
        return detections


class IntelligentMergingAlgorithm:
    """Advanced algorithms for merging overlapping detections"""
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.25,
        use_nms: bool = True,
        use_wbf: bool = False  # Weighted Boxes Fusion
    ):
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.use_nms = use_nms
        self.use_wbf = use_wbf
        
    def merge_detections(
        self,
        detections: List[Detection],
        strategy: str = 'nms'
    ) -> List[Detection]:
        """Merge overlapping detections using specified strategy"""
        
        if not detections:
            return []
        
        # Filter by confidence
        detections = [d for d in detections if d.confidence >= self.confidence_threshold]
        
        if strategy == 'nms':
            return self._nms_merge(detections)
        elif strategy == 'wbf':
            return self._wbf_merge(detections)
        elif strategy == 'soft_nms':
            return self._soft_nms_merge(detections)
        elif strategy == 'cluster':
            return self._cluster_merge(detections)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
    
    def _nms_merge(self, detections: List[Detection]) -> List[Detection]:
        """Standard Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det.class_id].append(det)
        
        merged = []
        
        for class_id, class_detections in class_groups.items():
            # Sort by confidence
            class_detections.sort(key=lambda x: x.confidence, reverse=True)
            
            # Apply NMS
            keep = []
            while class_detections:
                # Take highest confidence detection
                best = class_detections.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                class_detections = [
                    det for det in class_detections
                    if self._calculate_iou(best.bbox, det.bbox) < self.iou_threshold
                ]
            
            merged.extend(keep)
        
        return merged
    
    def _soft_nms_merge(
        self,
        detections: List[Detection],
        sigma: float = 0.5
    ) -> List[Detection]:
        """Soft-NMS: gradually decrease confidence of overlapping boxes"""
        if not detections:
            return []
        
        # Group by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det.class_id].append(det)
        
        merged = []
        
        for class_id, class_detections in class_groups.items():
            # Process with Soft-NMS
            processed = []
            
            while class_detections:
                # Find max confidence detection
                max_idx = max(range(len(class_detections)),
                            key=lambda i: class_detections[i].confidence)
                best = class_detections.pop(max_idx)
                processed.append(best)
                
                # Update confidence of overlapping boxes
                for i, det in enumerate(class_detections):
                    iou = self._calculate_iou(best.bbox, det.bbox)
                    if iou > 0:
                        # Gaussian decay
                        class_detections[i].confidence *= np.exp(-(iou * iou) / sigma)
                
                # Remove low confidence detections
                class_detections = [
                    det for det in class_detections
                    if det.confidence >= self.confidence_threshold
                ]
            
            merged.extend(processed)
        
        return merged
    
    def _wbf_merge(
        self,
        detections: List[Detection],
        weights: Optional[List[float]] = None
    ) -> List[Detection]:
        """Weighted Boxes Fusion - combine overlapping boxes"""
        if not detections:
            return []
        
        # Group overlapping detections
        clusters = self._cluster_detections(detections)
        
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Weighted average of boxes
                weights = [d.confidence for d in cluster] if weights is None else weights
                total_weight = sum(weights)
                
                # Calculate weighted average bbox
                weighted_bbox = [0, 0, 0, 0]
                for det, weight in zip(cluster, weights):
                    for i in range(4):
                        weighted_bbox[i] += det.bbox[i] * weight / total_weight
                
                # Create merged detection
                merged_det = Detection(
                    bbox=weighted_bbox,
                    confidence=sum(d.confidence for d in cluster) / len(cluster),
                    class_id=cluster[0].class_id,
                    class_name=cluster[0].class_name
                )
                merged.append(merged_det)
        
        return merged
    
    def _cluster_merge(self, detections: List[Detection]) -> List[Detection]:
        """Merge using clustering approach"""
        if not detections:
            return []
        
        # Build spatial index
        idx = index.Index()
        for i, det in enumerate(detections):
            idx.insert(i, det.bbox)
        
        # Find clusters of overlapping detections
        clusters = []
        processed = set()
        
        for i, det in enumerate(detections):
            if i in processed:
                continue
            
            # Find all overlapping detections
            cluster = [det]
            processed.add(i)
            
            # Get candidates from spatial index
            candidates = list(idx.intersection(det.bbox))
            
            for j in candidates:
                if j != i and j not in processed:
                    if self._calculate_iou(det.bbox, detections[j].bbox) >= self.iou_threshold:
                        cluster.append(detections[j])
                        processed.add(j)
            
            clusters.append(cluster)
        
        # Merge each cluster
        merged = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Take detection with highest confidence
                best = max(cluster, key=lambda x: x.confidence)
                merged.append(best)
        
        return merged
    
    def _cluster_detections(
        self,
        detections: List[Detection]
    ) -> List[List[Detection]]:
        """Cluster overlapping detections"""
        clusters = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j not in used:
                    if self._calculate_iou(det1.bbox, det2.bbox) >= self.iou_threshold:
                        cluster.append(det2)
                        used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class SpatialIndexOptimizer:
    """Spatial indexing for efficient geometric operations"""
    
    def __init__(self, detections: List[Detection]):
        self.detections = detections
        self.rtree_idx = None
        self.str_tree = None
        self.kd_tree = None
        
    def build_rtree_index(self) -> index.Index:
        """Build R-tree spatial index"""
        idx = index.Index()
        for i, det in enumerate(self.detections):
            idx.insert(i, det.bbox)
        self.rtree_idx = idx
        return idx
    
    def build_str_tree(self) -> STRtree:
        """Build STRtree for shapely geometries"""
        geometries = [det.polygon for det in self.detections]
        self.str_tree = STRtree(geometries)
        return self.str_tree
    
    def build_kd_tree(self) -> KDTree:
        """Build KD-tree for nearest neighbor queries"""
        centers = np.array([det.center for det in self.detections])
        self.kd_tree = KDTree(centers)
        return self.kd_tree
    
    def find_overlapping(
        self,
        query_box: List[float],
        min_overlap: float = 0.1
    ) -> List[int]:
        """Find all detections overlapping with query box"""
        if self.rtree_idx is None:
            self.build_rtree_index()
        
        # Get candidates from spatial index
        candidates = list(self.rtree_idx.intersection(query_box))
        
        # Filter by actual overlap
        overlapping = []
        query_poly = box(*query_box)
        
        for idx in candidates:
            det_poly = self.detections[idx].polygon
            intersection = query_poly.intersection(det_poly)
            
            if intersection.area / det_poly.area >= min_overlap:
                overlapping.append(idx)
        
        return overlapping
    
    def find_nearest(
        self,
        point: Tuple[float, float],
        k: int = 1
    ) -> List[int]:
        """Find k nearest detections to a point"""
        if self.kd_tree is None:
            self.build_kd_tree()
        
        distances, indices = self.kd_tree.query(point, k=k)
        
        if k == 1:
            return [indices] if indices < len(self.detections) else []
        else:
            return indices.tolist()


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_memory_layout(data: np.ndarray) -> np.ndarray:
        """Optimize memory layout for better cache performance"""
        if not data.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(data)
        return data
    
    @staticmethod
    def vectorize_iou_calculation(
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Vectorized IoU calculation for multiple boxes"""
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
        boxes2 = boxes2[np.newaxis, :, :]  # (1, M, 4)
        
        # Calculate intersection
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = np.where(union > 0, intersection / union, 0)
        
        return iou
    
    @staticmethod
    def parallel_process_tiles(
        tiles: List[TileInfo],
        process_func: callable,
        num_workers: int = 4
    ) -> List:
        """Parallel processing of tiles"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_func, tiles))
        
        return results
    
    @staticmethod
    def optimize_batch_size(
        available_memory: int,
        model_memory_per_sample: int,
        overhead: float = 0.1
    ) -> int:
        """Calculate optimal batch size based on available memory"""
        usable_memory = available_memory * (1 - overhead)
        batch_size = int(usable_memory / model_memory_per_sample)
        
        # Round to power of 2 for better performance
        return 2 ** int(np.log2(batch_size))


class GeometricAlgorithms:
    """Advanced geometric algorithms for spatial operations"""
    
    @staticmethod
    def douglas_peucker_simplify(
        points: np.ndarray,
        epsilon: float = 1.0
    ) -> np.ndarray:
        """Simplify polygon using Douglas-Peucker algorithm"""
        if len(points) < 3:
            return points
        
        # Find point with maximum distance
        dmax = 0
        index = 0
        
        for i in range(1, len(points) - 1):
            d = GeometricAlgorithms._point_line_distance(
                points[i],
                points[0],
                points[-1]
            )
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = GeometricAlgorithms.douglas_peucker_simplify(
                points[:index + 1], epsilon
            )
            rec_results2 = GeometricAlgorithms.douglas_peucker_simplify(
                points[index:], epsilon
            )
            
            # Build result
            result = np.vstack((rec_results1[:-1], rec_results2))
        else:
            result = np.vstack((points[0], points[-1]))
        
        return result
    
    @staticmethod
    def _point_line_distance(
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """Calculate perpendicular distance from point to line"""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        
        point_vec_scaled = point_vec / line_len
        
        t = np.dot(line_unitvec, point_vec_scaled)
        t = max(0.0, min(1.0, t))
        
        nearest = line_vec * t
        dist = np.linalg.norm(point_vec - nearest)
        
        return dist
    
    @staticmethod
    def convex_hull_graham_scan(points: np.ndarray) -> np.ndarray:
        """Compute convex hull using Graham scan algorithm"""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = sorted(points.tolist())
        if len(points) <= 1:
            return np.array(points)
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        return np.array(lower[:-1] + upper[:-1])
    
    @staticmethod
    def minimum_bounding_rectangle(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Find minimum area bounding rectangle using rotating calipers"""
        hull = GeometricAlgorithms.convex_hull_graham_scan(points)
        
        min_area = float('inf')
        min_rect = None
        
        for i in range(len(hull)):
            # Get edge vector
            edge = hull[(i + 1) % len(hull)] - hull[i]
            angle = np.arctan2(edge[1], edge[0])
            
            # Rotate hull
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated = hull @ rotation_matrix.T
            
            # Get bounding box
            min_x, min_y = rotated.min(axis=0)
            max_x, max_y = rotated.max(axis=0)
            
            area = (max_x - min_x) * (max_y - min_y)
            
            if area < min_area:
                min_area = area
                
                # Create rectangle corners
                corners = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ])
                
                # Rotate back
                inv_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                min_rect = corners @ inv_rotation.T
        
        return min_rect, min_area


class HungarianMatcher:
    """Hungarian algorithm for optimal assignment problems"""
    
    @staticmethod
    def match_detections(
        detections1: List[Detection],
        detections2: List[Detection],
        cost_threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """Match detections between two sets using Hungarian algorithm"""
        
        if not detections1 or not detections2:
            return []
        
        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(detections1), len(detections2)))
        
        for i, det1 in enumerate(detections1):
            for j, det2 in enumerate(detections2):
                iou = HungarianMatcher._calculate_iou(det1.bbox, det2.bbox)
                cost_matrix[i, j] = 1 - iou
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter by cost threshold
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < cost_threshold:
                matches.append((row, col))
        
        return matches
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


# Utility functions
def calculate_metrics(
    predictions: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    
    if not ground_truth:
        return {
            'precision': 1.0 if not predictions else 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    if not predictions:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Match predictions to ground truth
    matches = HungarianMatcher.match_detections(
        predictions, ground_truth, 1 - iou_threshold
    )
    
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


if __name__ == "__main__":
    # Example usage
    logger.info("Core Algorithms Module - Claude Opus")
    
    # Test tiling
    tiler = AdvancedTilingStrategy(tile_size=640, overlap=0.2)
    tiles = tiler.generate_tiles(2048, 2048)
    logger.info(f"Generated {len(tiles)} tiles")
    
    # Test merging
    merger = IntelligentMergingAlgorithm(iou_threshold=0.5)
    
    # Create sample detections
    detections = [
        Detection([100, 100, 200, 200], 0.9, 0, "class1"),
        Detection([150, 150, 250, 250], 0.8, 0, "class1"),
        Detection([300, 300, 400, 400], 0.85, 1, "class2")
    ]
    
    merged = merger.merge_detections(detections, strategy='nms')
    logger.info(f"Merged {len(detections)} detections to {len(merged)}")
    
    logger.info("Core algorithms initialized successfully")