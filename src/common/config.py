"""
Configuration management for Nong-View
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/nongview",
        description="PostgreSQL connection URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # Storage
    storage_path: str = Field(
        default="./storage",
        description="Base path for file storage"
    )
    minio_endpoint: Optional[str] = Field(
        default=None,
        description="MinIO endpoint URL"
    )
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    
    # GIS
    default_crs: str = Field(
        default="EPSG:5186",
        description="Default coordinate reference system (Korea 2000)"
    )
    
    # Tiling
    tile_size: int = Field(
        default=640,
        description="Default tile size in pixels"
    )
    tile_overlap: float = Field(
        default=0.2,
        description="Tile overlap ratio (0.0-1.0)"
    )
    
    # AI/ML
    gpu_device: int = Field(
        default=0,
        description="GPU device ID"
    )
    batch_size: int = Field(
        default=16,
        description="Inference batch size"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Detection confidence threshold"
    )
    nms_threshold: float = Field(
        default=0.5,
        description="Non-maximum suppression threshold"
    )
    
    # Models
    crop_model_path: str = Field(
        default="models/weights/yolov11_crop.pt",
        description="Path to crop detection model"
    )
    facility_model_path: str = Field(
        default="models/weights/yolov11_facility.pt",
        description="Path to facility detection model"
    )
    landuse_model_path: str = Field(
        default="models/weights/yolov11_landuse.pt",
        description="Path to land use classification model"
    )
    
    # Performance
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads"
    )
    chunk_size: int = Field(
        default=1024,
        description="File reading chunk size"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default="logs/nongview.log",
        description="Log file path"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()