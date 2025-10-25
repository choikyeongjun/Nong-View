"""
Data schemas for POD1
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class Bounds(BaseModel):
    """Spatial bounds"""
    minx: float
    miny: float
    maxx: float
    maxy: float
    
    def to_list(self) -> List[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]
    
    def intersects(self, other: 'Bounds') -> bool:
        """Check if bounds intersect with another bounds"""
        return not (
            self.maxx < other.minx or
            self.minx > other.maxx or
            self.maxy < other.miny or
            self.miny > other.maxy
        )


class SourceInfo(BaseModel):
    """Source information for imagery"""
    drone_model: Optional[str] = None
    camera: Optional[str] = None
    altitude: Optional[float] = None
    overlap: Optional[float] = None
    flight_date: Optional[datetime] = None


class ImageMetadata(BaseModel):
    """Image metadata schema"""
    image_id: UUID = Field(default_factory=uuid4)
    file_path: str
    capture_date: datetime
    crs: str = "EPSG:5186"
    resolution: float  # meters per pixel
    bounds: Bounds
    source: Optional[SourceInfo] = None
    file_size: Optional[int] = None
    format: Optional[str] = None
    bands: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('crs')
    def validate_crs(cls, v):
        """Validate coordinate reference system"""
        if not v.startswith('EPSG:'):
            raise ValueError(f"Invalid CRS format: {v}")
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution is positive"""
        if v <= 0:
            raise ValueError(f"Resolution must be positive: {v}")
        return v
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class ShapeMetadata(BaseModel):
    """Shapefile metadata schema"""
    shape_id: UUID = Field(default_factory=uuid4)
    file_path: str
    pnu: str  # Parcel number (19 digits)
    geometry_type: str
    crs: str = "EPSG:5186"
    bounds: Bounds
    properties: Dict[str, Any] = Field(default_factory=dict)
    area: Optional[float] = None  # square meters
    perimeter: Optional[float] = None  # meters
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('pnu')
    def validate_pnu(cls, v):
        """Validate PNU format (19 digits)"""
        if not v.isdigit() or len(v) != 19:
            raise ValueError(f"Invalid PNU format: {v}")
        return v
    
    @validator('geometry_type')
    def validate_geometry_type(cls, v):
        """Validate geometry type"""
        valid_types = ['Point', 'LineString', 'Polygon', 
                      'MultiPoint', 'MultiLineString', 'MultiPolygon']
        if v not in valid_types:
            raise ValueError(f"Invalid geometry type: {v}")
        return v


class DataRegistryEntry(BaseModel):
    """Data registry entry combining image and shape data"""
    registry_id: UUID = Field(default_factory=uuid4)
    image_metadata: Optional[ImageMetadata] = None
    shape_metadata: Optional[List[ShapeMetadata]] = None
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('processing_status')
    def validate_status(cls, v):
        """Validate processing status"""
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v


class VersionInfo(BaseModel):
    """Version information for time series data"""
    version_id: UUID = Field(default_factory=uuid4)
    registry_id: UUID
    version_number: int
    capture_date: datetime
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }