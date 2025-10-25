"""
Merge Engine - Core merging functionality for tile-based detections
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from uuid import UUID
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import geopandas as gpd
from rtree import index

from .schemas import (
    MergeConfig,
    MergedDetection,
    MergedResult,
    ParcelStatistics
)
from ..pod3_tiling.schemas import TileMetadata
from ..pod4_ai_inference.schemas import InferenceResult, Detection

logger = logging.getLogger(__name__)


class MergeEngine:
    """
    Engine for merging tile-based detection results
    Handles overlapping detections, boundary cases, and spatial aggregation
    """
    
    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize merge engine
        
        Args:
            config: Merge configuration
        """
        self.config = config or MergeConfig()
        
    def merge_tile_results(
        self,
        tile_results: List[InferenceResult],
        tile_metadata: List[TileMetadata],
        image_id: UUID,
        roi_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> MergedResult:
        """
        Merge detection results from multiple tiles
        
        Args:
            tile_results: List of inference results
            tile_metadata: List of tile metadata
            image_id: Source image ID
            roi_bounds: Optional ROI bounds for filtering
            
        Returns:
            MergedResult object
        """
        start_time = time.time()
        
        # Create tile lookup
        tile_lookup = {tile.tile_id: tile for tile in tile_metadata}
        
        # Convert detections to global coordinates
        global_detections = self._convert_to_global_coords(
            tile_results, tile_lookup
        )
        
        # Build spatial index
        spatial_idx = self._build_spatial_index(global_detections)
        
        # Merge overlapping detections
        merged_detections = self._merge_overlapping_detections(
            global_detections, spatial_idx
        )
        
        # Filter by ROI if provided
        if roi_bounds:
            merged_detections = self._filter_by_roi(merged_detections, roi_bounds)
        
        # Calculate overall bounds
        if not roi_bounds and merged_detections:
            roi_bounds = self._calculate_bounds(merged_detections)
        
        merge_time = time.time() - start_time
        
        result = MergedResult(
            image_id=image_id,
            roi_bounds=roi_bounds or (0, 0, 0, 0),
            detections=merged_detections,
            total_tiles_processed=len(tile_results),
            merge_time=merge_time
        )
        
        logger.info(f"Merged {len(tile_results)} tile results into {len(merged_detections)} detections in {merge_time:.2f}s")
        
        return result
    
    def _convert_to_global_coords(
        self,
        tile_results: List[InferenceResult],
        tile_lookup: Dict[UUID, TileMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Convert tile-local detections to global coordinates
        
        Args:
            tile_results: List of inference results
            tile_lookup: Tile metadata lookup
            
        Returns:
            List of detections with global coordinates
        """
        global_detections = []
        
        for result in tile_results:
            tile = tile_lookup.get(result.tile_id)
            if not tile:
                logger.warning(f"Tile metadata not found for {result.tile_id}")
                continue
            
            # Get tile offset in pixels
            pixel_offset_x = tile.bounds.pixel_bounds[0]
            pixel_offset_y = tile.bounds.pixel_bounds[1]
            
            # Get geographic bounds
            geo_bounds = tile.bounds.geo_bounds
            geo_width = geo_bounds[2] - geo_bounds[0]
            geo_height = geo_bounds[3] - geo_bounds[1]
            
            # Calculate scale factors
            tile_width = tile.bounds.pixel_bounds[2] - tile.bounds.pixel_bounds[0]
            tile_height = tile.bounds.pixel_bounds[3] - tile.bounds.pixel_bounds[1]
            
            scale_x = geo_width / tile_width if tile_width > 0 else 1
            scale_y = geo_height / tile_height if tile_height > 0 else 1
            
            for detection in result.detections:
                # Convert to global coordinates
                if detection.bbox:
                    x1, y1, x2, y2 = detection.bbox
                    
                    # Convert to geographic coordinates
                    global_x1 = geo_bounds[0] + x1 * scale_x
                    global_y1 = geo_bounds[1] + y1 * scale_y
                    global_x2 = geo_bounds[0] + x2 * scale_x
                    global_y2 = geo_bounds[1] + y2 * scale_y
                    
                    global_bbox = (global_x1, global_y1, global_x2, global_y2)
                else:
                    global_bbox = None
                
                # Convert segmentation if available
                if detection.segmentation:
                    global_segmentation = []
                    for polygon in detection.segmentation:
                        global_polygon = []
                        for i in range(0, len(polygon), 2):
                            if i + 1 < len(polygon):
                                x = geo_bounds[0] + polygon[i] * scale_x
                                y = geo_bounds[1] + polygon[i + 1] * scale_y
                                global_polygon.extend([x, y])
                        global_segmentation.append(global_polygon)
                else:
                    global_segmentation = None
                
                # Calculate area in square meters
                if global_bbox:
                    area = (global_bbox[2] - global_bbox[0]) * (global_bbox[3] - global_bbox[1])
                else:
                    area = detection.area * scale_x * scale_y if detection.area else 0
                
                global_detections.append({
                    'detection': detection,
                    'tile_id': result.tile_id,
                    'global_bbox': global_bbox,
                    'global_segmentation': global_segmentation,
                    'area': area,
                    'processed': False
                })
        
        return global_detections
    
    def _build_spatial_index(
        self,
        detections: List[Dict[str, Any]]
    ) -> index.Index:
        """
        Build spatial index for efficient overlap detection
        
        Args:
            detections: List of detections with global coordinates
            
        Returns:
            R-tree spatial index
        """
        idx = index.Index()
        
        for i, det in enumerate(detections):
            if det['global_bbox']:
                idx.insert(i, det['global_bbox'])
        
        return idx
    
    def _merge_overlapping_detections(
        self,
        detections: List[Dict[str, Any]],
        spatial_idx: index.Index
    ) -> List[MergedDetection]:
        """
        Merge overlapping detections using configured strategy
        
        Args:
            detections: List of detections with global coordinates
            spatial_idx: Spatial index
            
        Returns:
            List of merged detections
        """
        merged = []
        processed = set()
        
        for i, det in enumerate(detections):
            if i in processed or not det['global_bbox']:
                continue
            
            # Find overlapping detections
            overlapping = self._find_overlapping(i, detections, spatial_idx)
            
            if overlapping:
                # Merge overlapping detections
                merged_det = self._merge_detection_group(
                    [detections[j] for j in overlapping]
                )
                merged.append(merged_det)
                processed.update(overlapping)
            else:
                # Convert single detection
                merged_det = self._create_merged_detection(det)
                merged.append(merged_det)
                processed.add(i)
        
        return merged
    
    def _find_overlapping(
        self,
        idx: int,
        detections: List[Dict[str, Any]],
        spatial_idx: index.Index
    ) -> Set[int]:
        """
        Find detections overlapping with given detection
        
        Args:
            idx: Detection index
            detections: List of all detections
            spatial_idx: Spatial index
            
        Returns:
            Set of overlapping detection indices
        """
        det = detections[idx]
        if not det['global_bbox']:
            return {idx}
        
        overlapping = {idx}
        to_check = [idx]
        checked = set()
        
        while to_check:
            current = to_check.pop()
            if current in checked:
                continue
            checked.add(current)
            
            current_det = detections[current]
            current_bbox = current_det['global_bbox']
            
            # Find potentially overlapping detections
            candidates = list(spatial_idx.intersection(current_bbox))
            
            for candidate in candidates:
                if candidate in overlapping:
                    continue
                
                cand_det = detections[candidate]
                
                # Check if same class
                if current_det['detection'].class_name != cand_det['detection'].class_name:
                    continue
                
                # Calculate IOU
                iou = self._calculate_iou(current_bbox, cand_det['global_bbox'])
                
                if iou >= self.config.iou_threshold:
                    overlapping.add(candidate)
                    to_check.append(candidate)
        
        return overlapping
    
    def _calculate_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate Intersection over Union for two bboxes
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IOU value
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _merge_detection_group(
        self,
        group: List[Dict[str, Any]]
    ) -> MergedDetection:
        """
        Merge a group of overlapping detections
        
        Args:
            group: List of overlapping detections
            
        Returns:
            Merged detection
        """
        if len(group) == 1:
            return self._create_merged_detection(group[0])
        
        # Extract properties
        class_name = group[0]['detection'].class_name
        class_id = group[0]['detection'].class_id
        
        # Merge based on strategy
        if self.config.merge_strategy == 'weighted_avg':
            merged_bbox, confidence = self._merge_weighted_avg(group)
        elif self.config.merge_strategy == 'max_confidence':
            merged_bbox, confidence = self._merge_max_confidence(group)
        elif self.config.merge_strategy == 'union':
            merged_bbox, confidence = self._merge_union(group)
        else:  # intersection
            merged_bbox, confidence = self._merge_intersection(group)
        
        # Calculate merged area
        if merged_bbox:
            area = (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1])
        else:
            area = sum(d['area'] for d in group) / len(group)
        
        # Collect source tiles
        source_tiles = list(set(d['tile_id'] for d in group))
        
        return MergedDetection(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            bbox=merged_bbox,
            area=area,
            source_tiles=source_tiles,
            merge_count=len(group)
        )
    
    def _merge_weighted_avg(
        self,
        group: List[Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """
        Merge using weighted average by confidence
        """
        if not any(d['global_bbox'] for d in group):
            return None, 0.0
        
        total_conf = sum(d['detection'].confidence for d in group)
        
        if total_conf == 0:
            return None, 0.0
        
        weighted_bbox = [0, 0, 0, 0]
        
        for det in group:
            if det['global_bbox']:
                weight = det['detection'].confidence / total_conf
                for i in range(4):
                    weighted_bbox[i] += det['global_bbox'][i] * weight
        
        avg_confidence = total_conf / len(group)
        
        return tuple(weighted_bbox), avg_confidence
    
    def _merge_max_confidence(
        self,
        group: List[Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """
        Use detection with maximum confidence
        """
        max_det = max(group, key=lambda d: d['detection'].confidence)
        return max_det['global_bbox'], max_det['detection'].confidence
    
    def _merge_union(
        self,
        group: List[Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """
        Merge using union of all bboxes
        """
        bboxes = [d['global_bbox'] for d in group if d['global_bbox']]
        
        if not bboxes:
            return None, 0.0
        
        min_x = min(b[0] for b in bboxes)
        min_y = min(b[1] for b in bboxes)
        max_x = max(b[2] for b in bboxes)
        max_y = max(b[3] for b in bboxes)
        
        avg_confidence = sum(d['detection'].confidence for d in group) / len(group)
        
        return (min_x, min_y, max_x, max_y), avg_confidence
    
    def _merge_intersection(
        self,
        group: List[Dict[str, Any]]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """
        Merge using intersection of all bboxes
        """
        bboxes = [d['global_bbox'] for d in group if d['global_bbox']]
        
        if not bboxes:
            return None, 0.0
        
        max_x1 = max(b[0] for b in bboxes)
        max_y1 = max(b[1] for b in bboxes)
        min_x2 = min(b[2] for b in bboxes)
        min_y2 = min(b[3] for b in bboxes)
        
        # Check if valid intersection
        if max_x1 >= min_x2 or max_y1 >= min_y2:
            # Fall back to union if no intersection
            return self._merge_union(group)
        
        avg_confidence = sum(d['detection'].confidence for d in group) / len(group)
        
        return (max_x1, max_y1, min_x2, min_y2), avg_confidence
    
    def _create_merged_detection(
        self,
        det: Dict[str, Any]
    ) -> MergedDetection:
        """
        Create merged detection from single detection
        """
        return MergedDetection(
            class_name=det['detection'].class_name,
            class_id=det['detection'].class_id,
            confidence=det['detection'].confidence,
            bbox=det['global_bbox'],
            segmentation=det['global_segmentation'],
            area=det['area'],
            source_tiles=[det['tile_id']],
            merge_count=1
        )
    
    def _filter_by_roi(
        self,
        detections: List[MergedDetection],
        roi_bounds: Tuple[float, float, float, float]
    ) -> List[MergedDetection]:
        """
        Filter detections by ROI bounds
        """
        filtered = []
        roi_box = box(*roi_bounds)
        
        for det in detections:
            if det.bbox:
                det_box = box(*det.bbox)
                if roi_box.intersects(det_box):
                    # Calculate intersection ratio
                    intersection = roi_box.intersection(det_box)
                    if intersection.area / det_box.area > 0.5:  # Keep if >50% inside ROI
                        filtered.append(det)
            else:
                filtered.append(det)  # Keep if no bbox
        
        return filtered
    
    def _calculate_bounds(
        self,
        detections: List[MergedDetection]
    ) -> Tuple[float, float, float, float]:
        """
        Calculate overall bounds from detections
        """
        bboxes = [d.bbox for d in detections if d.bbox]
        
        if not bboxes:
            return (0, 0, 0, 0)
        
        min_x = min(b[0] for b in bboxes)
        min_y = min(b[1] for b in bboxes)
        max_x = max(b[2] for b in bboxes)
        max_y = max(b[3] for b in bboxes)
        
        return (min_x, min_y, max_x, max_y)
    
    def calculate_parcel_statistics(
        self,
        merged_result: MergedResult,
        parcel_geometries: gpd.GeoDataFrame
    ) -> List[ParcelStatistics]:
        """
        Calculate statistics per parcel
        
        Args:
            merged_result: Merged detection result
            parcel_geometries: GeoDataFrame with parcel boundaries
            
        Returns:
            List of parcel statistics
        """
        stats = []
        
        for _, parcel in parcel_geometries.iterrows():
            pnu = parcel.get('PNU', '')
            geometry = parcel.geometry
            
            if not geometry or not pnu:
                continue
            
            # Find detections within parcel
            parcel_detections = []
            class_areas = {}
            class_counts = {}
            
            for det in merged_result.detections:
                if det.bbox:
                    det_box = box(*det.bbox)
                    
                    if geometry.intersects(det_box):
                        intersection = geometry.intersection(det_box)
                        intersection_area = intersection.area
                        
                        # Update statistics
                        if det.class_name not in class_areas:
                            class_areas[det.class_name] = 0
                            class_counts[det.class_name] = 0
                        
                        class_areas[det.class_name] += intersection_area
                        class_counts[det.class_name] += 1
                        parcel_detections.append(det)
            
            # Calculate totals
            total_area = geometry.area
            detected_area = sum(class_areas.values())
            coverage_ratio = detected_area / total_area if total_area > 0 else 0
            
            # Find dominant class
            dominant_class = None
            if class_areas:
                dominant_class = max(class_areas.items(), key=lambda x: x[1])[0]
            
            parcel_stat = ParcelStatistics(
                pnu=pnu,
                total_area=total_area,
                detected_area=detected_area,
                coverage_ratio=coverage_ratio,
                class_areas=class_areas,
                class_counts=class_counts,
                dominant_class=dominant_class
            )
            
            stats.append(parcel_stat)
        
        return stats