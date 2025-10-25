"""
Coordinate system validation and geometry verification
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.validation import explain_validity, make_valid
from pyproj import CRS, Transformer
import numpy as np

logger = logging.getLogger(__name__)


class CoordinateValidator:
    """
    Coordinate system validator and transformer
    Ensures all data uses consistent CRS (EPSG:5186 - Korea 2000)
    """
    
    DEFAULT_CRS = "EPSG:5186"  # Korea 2000 / Central Belt 2010
    SUPPORTED_CRS = [
        "EPSG:5186",  # Korea 2000 / Central Belt 2010
        "EPSG:5174",  # Korea 2000 / Unified CS
        "EPSG:4326",  # WGS84
        "EPSG:3857",  # Web Mercator
        "EPSG:5179",  # Korea 2000 / UTM Zone 52N
    ]
    
    def __init__(self, target_crs: str = DEFAULT_CRS):
        """
        Initialize validator with target CRS
        
        Args:
            target_crs: Target coordinate reference system
        """
        self.target_crs = target_crs
        self._crs = CRS(target_crs)
        self._transformers: Dict[str, Transformer] = {}
        
    def validate_crs(self, crs_string: str) -> bool:
        """
        Validate CRS string format
        
        Args:
            crs_string: CRS string (e.g., "EPSG:5186")
            
        Returns:
            True if valid CRS
        """
        try:
            CRS(crs_string)
            return True
        except Exception as e:
            logger.error(f"Invalid CRS: {crs_string} - {e}")
            return False
    
    def is_supported_crs(self, crs_string: str) -> bool:
        """Check if CRS is in supported list"""
        return crs_string in self.SUPPORTED_CRS
    
    def get_transformer(self, source_crs: str) -> Transformer:
        """
        Get or create transformer for CRS conversion
        
        Args:
            source_crs: Source coordinate reference system
            
        Returns:
            Transformer object for coordinate transformation
        """
        if source_crs not in self._transformers:
            self._transformers[source_crs] = Transformer.from_crs(
                source_crs, 
                self.target_crs, 
                always_xy=True
            )
        return self._transformers[source_crs]
    
    def transform_point(
        self, 
        x: float, 
        y: float, 
        source_crs: str
    ) -> Tuple[float, float]:
        """
        Transform a single point to target CRS
        
        Args:
            x: X coordinate
            y: Y coordinate
            source_crs: Source CRS
            
        Returns:
            Transformed (x, y) coordinates
        """
        if source_crs == self.target_crs:
            return x, y
            
        transformer = self.get_transformer(source_crs)
        return transformer.transform(x, y)
    
    def transform_bounds(
        self, 
        bounds: List[float], 
        source_crs: str
    ) -> List[float]:
        """
        Transform bounding box to target CRS
        
        Args:
            bounds: [minx, miny, maxx, maxy]
            source_crs: Source CRS
            
        Returns:
            Transformed bounds
        """
        if source_crs == self.target_crs:
            return bounds
            
        minx, miny, maxx, maxy = bounds
        transformer = self.get_transformer(source_crs)
        
        # Transform all four corners
        corners = [
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy)
        ]
        
        transformed = [transformer.transform(x, y) for x, y in corners]
        
        # Calculate new bounds from transformed corners
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        
        return [min(xs), min(ys), max(xs), max(ys)]
    
    def transform_geometry(
        self, 
        geometry: Dict[str, Any], 
        source_crs: str
    ) -> Dict[str, Any]:
        """
        Transform geometry to target CRS
        
        Args:
            geometry: GeoJSON geometry dict
            source_crs: Source CRS
            
        Returns:
            Transformed geometry
        """
        if source_crs == self.target_crs:
            return geometry
            
        transformer = self.get_transformer(source_crs)
        geom = shape(geometry)
        
        # Transform based on geometry type
        if geom.geom_type == 'Point':
            x, y = transformer.transform(geom.x, geom.y)
            transformed = shape({'type': 'Point', 'coordinates': [x, y]})
        else:
            # For complex geometries, transform all coordinates
            def transform_coords(coords):
                if isinstance(coords[0], (list, tuple)):
                    return [transform_coords(c) for c in coords]
                else:
                    return transformer.transform(coords[0], coords[1])
            
            geom_dict = mapping(geom)
            geom_dict['coordinates'] = transform_coords(geom_dict['coordinates'])
            transformed = shape(geom_dict)
        
        return mapping(transformed)


class GeometryValidator:
    """
    Validate and repair geometries
    """
    
    def __init__(self):
        """Initialize geometry validator"""
        self.min_area_threshold = 1.0  # square meters
        self.simplify_tolerance = 0.1  # meters
        
    def validate_geometry(self, geometry: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate geometry
        
        Args:
            geometry: GeoJSON geometry dict
            
        Returns:
            Tuple of (is_valid, explanation)
        """
        try:
            geom = shape(geometry)
            is_valid = geom.is_valid
            
            if not is_valid:
                explanation = explain_validity(geom)
                return False, explanation
                
            # Additional checks
            if geom.is_empty:
                return False, "Geometry is empty"
                
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                if geom.area < self.min_area_threshold:
                    return False, f"Area too small: {geom.area} m²"
                    
            return True, "Valid geometry"
            
        except Exception as e:
            return False, f"Error validating geometry: {e}"
    
    def repair_geometry(self, geometry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair invalid geometry
        
        Args:
            geometry: GeoJSON geometry dict
            
        Returns:
            Repaired geometry or None if unrepairable
        """
        try:
            geom = shape(geometry)
            
            if not geom.is_valid:
                # Try to make valid
                repaired = make_valid(geom)
                
                # Check if repair was successful
                if repaired.is_valid:
                    logger.info("Successfully repaired geometry")
                    return mapping(repaired)
                else:
                    logger.warning("Could not repair geometry")
                    return None
                    
            return geometry
            
        except Exception as e:
            logger.error(f"Error repairing geometry: {e}")
            return None
    
    def simplify_geometry(
        self, 
        geometry: Dict[str, Any], 
        tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simplify geometry to reduce complexity
        
        Args:
            geometry: GeoJSON geometry dict
            tolerance: Simplification tolerance in meters
            
        Returns:
            Simplified geometry
        """
        if tolerance is None:
            tolerance = self.simplify_tolerance
            
        try:
            geom = shape(geometry)
            simplified = geom.simplify(tolerance, preserve_topology=True)
            return mapping(simplified)
            
        except Exception as e:
            logger.error(f"Error simplifying geometry: {e}")
            return geometry
    
    def check_polygon_integrity(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check polygon integrity (holes, self-intersection, etc.)
        
        Args:
            geometry: GeoJSON polygon geometry
            
        Returns:
            Validation results dict
        """
        results = {
            'is_valid': False,
            'has_holes': False,
            'is_simple': False,
            'is_closed': False,
            'area': 0.0,
            'perimeter': 0.0,
            'issues': []
        }
        
        try:
            geom = shape(geometry)
            
            if geom.geom_type not in ['Polygon', 'MultiPolygon']:
                results['issues'].append(f"Not a polygon: {geom.geom_type}")
                return results
                
            results['is_valid'] = geom.is_valid
            results['is_simple'] = geom.is_simple
            results['area'] = geom.area
            results['perimeter'] = geom.length
            
            if geom.geom_type == 'Polygon':
                results['has_holes'] = len(geom.interiors) > 0
                results['is_closed'] = geom.exterior.is_ring
            elif geom.geom_type == 'MultiPolygon':
                results['has_holes'] = any(len(p.interiors) > 0 for p in geom.geoms)
                results['is_closed'] = all(p.exterior.is_ring for p in geom.geoms)
                
            if not results['is_valid']:
                results['issues'].append(explain_validity(geom))
                
            if not results['is_simple']:
                results['issues'].append("Geometry is not simple (self-intersecting)")
                
            if results['area'] < self.min_area_threshold:
                results['issues'].append(f"Area too small: {results['area']} m²")
                
        except Exception as e:
            results['issues'].append(f"Error checking polygon: {e}")
            
        return results
    
    def detect_gaps_overlaps(
        self, 
        geometries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect gaps and overlaps between geometries
        
        Args:
            geometries: List of GeoJSON geometries
            
        Returns:
            Analysis results
        """
        results = {
            'total_geometries': len(geometries),
            'overlaps': [],
            'gaps': [],
            'total_area': 0.0,
            'union_area': 0.0
        }
        
        try:
            shapes = [shape(g) for g in geometries]
            
            # Calculate total area
            results['total_area'] = sum(s.area for s in shapes)
            
            # Calculate union area
            union = shapes[0]
            for s in shapes[1:]:
                union = union.union(s)
            results['union_area'] = union.area
            
            # Check for overlaps
            for i in range(len(shapes)):
                for j in range(i + 1, len(shapes)):
                    if shapes[i].intersects(shapes[j]):
                        intersection = shapes[i].intersection(shapes[j])
                        if intersection.area > 0:
                            results['overlaps'].append({
                                'indices': [i, j],
                                'area': intersection.area
                            })
            
            # Gap detection (simplified - actual implementation would be more complex)
            if results['union_area'] < results['total_area'] * 0.95:
                results['gaps'].append({
                    'type': 'potential_gaps',
                    'area_difference': results['total_area'] - results['union_area']
                })
                
        except Exception as e:
            logger.error(f"Error detecting gaps/overlaps: {e}")
            
        return results