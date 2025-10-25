"""
Data Registry - Central data management system for the pipeline
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor

import rasterio
from rasterio.crs import CRS as RasterioCRS
import geopandas as gpd
from shapely.geometry import mapping

from .schemas import (
    ImageMetadata, 
    ShapeMetadata, 
    DataRegistryEntry, 
    VersionInfo,
    Bounds,
    SourceInfo
)
from .validators import CoordinateValidator, GeometryValidator
from ..common.config import settings

logger = logging.getLogger(__name__)


class DataRegistry:
    """
    Central data registry for managing all input data
    Handles versioning, validation, and indexing
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize data registry
        
        Args:
            storage_path: Base path for data storage
        """
        self.storage_path = Path(storage_path or settings.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_path = self.storage_path / "images"
        self.shapes_path = self.storage_path / "shapes"
        self.metadata_path = self.storage_path / "metadata"
        
        for path in [self.images_path, self.shapes_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize validators
        self.coord_validator = CoordinateValidator()
        self.geom_validator = GeometryValidator()
        
        # In-memory registry (would be database in production)
        self._registry: Dict[UUID, DataRegistryEntry] = {}
        self._versions: Dict[UUID, List[VersionInfo]] = {}
        self._index: Dict[str, List[UUID]] = {}  # Various indices
        
        # Thread pool for I/O operations
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
        # Load existing registry if exists
        self._load_registry()
    
    def _load_registry(self):
        """Load existing registry from disk"""
        registry_file = self.metadata_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get('entries', []):
                        entry = DataRegistryEntry(**entry_data)
                        self._registry[entry.registry_id] = entry
                        
                logger.info(f"Loaded {len(self._registry)} registry entries")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.metadata_path / "registry.json"
        try:
            data = {
                'entries': [entry.dict() for entry in self._registry.values()],
                'updated_at': datetime.now().isoformat()
            }
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.debug("Registry saved to disk")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_image_metadata(self, file_path: Path) -> ImageMetadata:
        """
        Extract metadata from image file
        
        Args:
            file_path: Path to image file
            
        Returns:
            ImageMetadata object
        """
        try:
            with rasterio.open(file_path) as src:
                # Extract bounds
                bounds = Bounds(
                    minx=src.bounds.left,
                    miny=src.bounds.bottom,
                    maxx=src.bounds.right,
                    maxy=src.bounds.top
                )
                
                # Transform bounds if needed
                if src.crs and str(src.crs) != self.coord_validator.target_crs:
                    bounds_list = self.coord_validator.transform_bounds(
                        [bounds.minx, bounds.miny, bounds.maxx, bounds.maxy],
                        str(src.crs)
                    )
                    bounds = Bounds(
                        minx=bounds_list[0],
                        miny=bounds_list[1],
                        maxx=bounds_list[2],
                        maxy=bounds_list[3]
                    )
                
                # Calculate resolution (average of x and y)
                res_x = (src.bounds.right - src.bounds.left) / src.width
                res_y = (src.bounds.top - src.bounds.bottom) / src.height
                resolution = (res_x + res_y) / 2
                
                # Get capture date from tags if available
                capture_date = datetime.now()
                if 'TIFFTAG_DATETIME' in src.tags():
                    try:
                        capture_date = datetime.strptime(
                            src.tags()['TIFFTAG_DATETIME'], 
                            '%Y:%m:%d %H:%M:%S'
                        )
                    except:
                        pass
                
                metadata = ImageMetadata(
                    file_path=str(file_path),
                    capture_date=capture_date,
                    crs=str(src.crs) if src.crs else "EPSG:5186",
                    resolution=resolution,
                    bounds=bounds,
                    file_size=file_path.stat().st_size,
                    format=src.driver,
                    bands=src.count,
                    width=src.width,
                    height=src.height,
                    metadata=dict(src.tags())
                )
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
            raise
    
    def _extract_shape_metadata(self, file_path: Path) -> List[ShapeMetadata]:
        """
        Extract metadata from shapefile
        
        Args:
            file_path: Path to shapefile
            
        Returns:
            List of ShapeMetadata objects
        """
        try:
            gdf = gpd.read_file(file_path)
            
            # Transform to target CRS if needed
            if gdf.crs and str(gdf.crs) != self.coord_validator.target_crs:
                gdf = gdf.to_crs(self.coord_validator.target_crs)
            
            metadata_list = []
            
            for idx, row in gdf.iterrows():
                # Get PNU from properties
                pnu = row.get('PNU', '').zfill(19)
                if not pnu or pnu == '0' * 19:
                    logger.warning(f"Missing PNU for shape at index {idx}")
                    continue
                
                # Get geometry bounds
                geom_bounds = row.geometry.bounds
                bounds = Bounds(
                    minx=geom_bounds[0],
                    miny=geom_bounds[1],
                    maxx=geom_bounds[2],
                    maxy=geom_bounds[3]
                )
                
                # Validate geometry
                is_valid, explanation = self.geom_validator.validate_geometry(
                    mapping(row.geometry)
                )
                
                if not is_valid:
                    logger.warning(f"Invalid geometry for PNU {pnu}: {explanation}")
                    # Try to repair
                    repaired = self.geom_validator.repair_geometry(
                        mapping(row.geometry)
                    )
                    if repaired:
                        row.geometry = gpd.GeoSeries.from_wkt([row.geometry.wkt])[0]
                
                metadata = ShapeMetadata(
                    file_path=str(file_path),
                    pnu=pnu,
                    geometry_type=row.geometry.geom_type,
                    crs=str(gdf.crs) if gdf.crs else "EPSG:5186",
                    bounds=bounds,
                    properties=dict(row.drop('geometry')),
                    area=row.geometry.area,
                    perimeter=row.geometry.length
                )
                
                metadata_list.append(metadata)
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Error extracting shape metadata: {e}")
            raise
    
    async def register_image(
        self, 
        file_path: str,
        source_info: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Register an image in the data registry
        
        Args:
            file_path: Path to image file
            source_info: Optional source information
            
        Returns:
            Registry ID
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Extract metadata
        metadata = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._extract_image_metadata,
            path
        )
        
        # Add source info if provided
        if source_info:
            metadata.source = SourceInfo(**source_info)
        
        # Create registry entry
        entry = DataRegistryEntry(
            image_metadata=metadata,
            processing_status="pending"
        )
        
        # Add to registry
        self._registry[entry.registry_id] = entry
        
        # Update indices
        self._update_indices(entry)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered image: {entry.registry_id}")
        return entry.registry_id
    
    async def register_shapes(
        self, 
        file_path: str,
        registry_id: Optional[UUID] = None
    ) -> UUID:
        """
        Register shapes in the data registry
        
        Args:
            file_path: Path to shapefile
            registry_id: Optional existing registry ID to associate with
            
        Returns:
            Registry ID
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Shapefile not found: {file_path}")
        
        # Extract metadata
        metadata_list = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._extract_shape_metadata,
            path
        )
        
        if registry_id and registry_id in self._registry:
            # Add to existing entry
            entry = self._registry[registry_id]
            entry.shape_metadata = metadata_list
            entry.updated_at = datetime.now()
        else:
            # Create new entry
            entry = DataRegistryEntry(
                shape_metadata=metadata_list,
                processing_status="pending"
            )
            self._registry[entry.registry_id] = entry
        
        # Update indices
        self._update_indices(entry)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered {len(metadata_list)} shapes: {entry.registry_id}")
        return entry.registry_id
    
    def _update_indices(self, entry: DataRegistryEntry):
        """Update various indices for fast lookup"""
        # Date index
        if entry.image_metadata:
            date_key = entry.image_metadata.capture_date.strftime('%Y%m%d')
            if date_key not in self._index:
                self._index[date_key] = []
            self._index[date_key].append(entry.registry_id)
        
        # PNU index
        if entry.shape_metadata:
            for shape in entry.shape_metadata:
                pnu_key = f"pnu_{shape.pnu}"
                if pnu_key not in self._index:
                    self._index[pnu_key] = []
                self._index[pnu_key].append(entry.registry_id)
    
    def get_entry(self, registry_id: UUID) -> Optional[DataRegistryEntry]:
        """Get registry entry by ID"""
        return self._registry.get(registry_id)
    
    def find_by_date(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[DataRegistryEntry]:
        """Find entries by date range"""
        results = []
        for entry in self._registry.values():
            if entry.image_metadata:
                if start_date <= entry.image_metadata.capture_date <= end_date:
                    results.append(entry)
        return results
    
    def find_by_pnu(self, pnu: str) -> List[DataRegistryEntry]:
        """Find entries by PNU"""
        key = f"pnu_{pnu}"
        if key in self._index:
            return [self._registry[id] for id in self._index[key]]
        return []
    
    def find_by_bounds(
        self, 
        bounds: Bounds
    ) -> List[DataRegistryEntry]:
        """Find entries intersecting with bounds"""
        results = []
        for entry in self._registry.values():
            if entry.image_metadata:
                if entry.image_metadata.bounds.intersects(bounds):
                    results.append(entry)
        return results
    
    def create_version(
        self, 
        registry_id: UUID,
        description: Optional[str] = None
    ) -> UUID:
        """
        Create a version snapshot for time series tracking
        
        Args:
            registry_id: Registry entry to version
            description: Optional version description
            
        Returns:
            Version ID
        """
        if registry_id not in self._registry:
            raise ValueError(f"Registry ID not found: {registry_id}")
        
        entry = self._registry[registry_id]
        
        # Get or create version list
        if registry_id not in self._versions:
            self._versions[registry_id] = []
        
        # Create version info
        version_number = len(self._versions[registry_id]) + 1
        version = VersionInfo(
            registry_id=registry_id,
            version_number=version_number,
            capture_date=entry.image_metadata.capture_date if entry.image_metadata else datetime.now(),
            description=description
        )
        
        self._versions[registry_id].append(version)
        
        logger.info(f"Created version {version_number} for {registry_id}")
        return version.version_id
    
    def get_versions(self, registry_id: UUID) -> List[VersionInfo]:
        """Get all versions for a registry entry"""
        return self._versions.get(registry_id, [])
    
    def update_status(
        self, 
        registry_id: UUID, 
        status: str,
        error_message: Optional[str] = None
    ):
        """
        Update processing status of registry entry
        
        Args:
            registry_id: Registry ID
            status: New status
            error_message: Optional error message
        """
        if registry_id not in self._registry:
            raise ValueError(f"Registry ID not found: {registry_id}")
        
        entry = self._registry[registry_id]
        entry.processing_status = status
        entry.error_message = error_message
        entry.updated_at = datetime.now()
        
        self._save_registry()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_images = sum(
            1 for e in self._registry.values() 
            if e.image_metadata
        )
        total_shapes = sum(
            len(e.shape_metadata) for e in self._registry.values() 
            if e.shape_metadata
        )
        
        status_counts = {}
        for entry in self._registry.values():
            status = entry.processing_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_entries': len(self._registry),
            'total_images': total_images,
            'total_shapes': total_shapes,
            'status_counts': status_counts,
            'total_versions': sum(len(v) for v in self._versions.values())
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
        self._save_registry()