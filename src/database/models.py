from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid

def generate_uuid():
    return str(uuid.uuid4())

class Image(Base):
    __tablename__ = "images"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False, unique=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    
    width = Column(Integer)
    height = Column(Integer)
    bands = Column(Integer)
    crs = Column(String)
    bounds = Column(JSON)
    
    metadata = Column(JSON)
    status = Column(String, default="uploaded")
    
    analyses = relationship("Analysis", back_populates="image", cascade="all, delete-orphan")
    tiles = relationship("Tile", back_populates="image", cascade="all, delete-orphan")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("images.id", ondelete="CASCADE"))
    job_id = Column(String, unique=True)
    
    analysis_type = Column(String, nullable=False)
    model_name = Column(String)
    model_version = Column(String)
    
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    parameters = Column(JSON)
    error_message = Column(Text)
    
    image = relationship("Image", back_populates="analyses")
    results = relationship("Result", back_populates="analysis", cascade="all, delete-orphan")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Result(Base):
    __tablename__ = "results"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"))
    
    class_name = Column(String)
    confidence = Column(Float)
    
    geometry_type = Column(String)
    geometry = Column(JSON)
    
    bbox = Column(JSON)
    area = Column(Float)
    
    attributes = Column(JSON)
    
    analysis = relationship("Analysis", back_populates="results")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Tile(Base):
    __tablename__ = "tiles"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("images.id", ondelete="CASCADE"))
    
    tile_index = Column(Integer)
    row = Column(Integer)
    col = Column(Integer)
    
    x_min = Column(Float)
    y_min = Column(Float)
    x_max = Column(Float)
    y_max = Column(Float)
    
    width = Column(Integer)
    height = Column(Integer)
    
    filepath = Column(String)
    processed = Column(Boolean, default=False)
    
    image = relationship("Image", back_populates="tiles")
    tile_results = relationship("TileResult", back_populates="tile", cascade="all, delete-orphan")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TileResult(Base):
    __tablename__ = "tile_results"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    tile_id = Column(String, ForeignKey("tiles.id", ondelete="CASCADE"))
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"))
    
    detections = Column(JSON)
    inference_time = Column(Float)
    
    tile = relationship("Tile", back_populates="tile_results")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Parcel(Base):
    __tablename__ = "parcels"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    pnu = Column(String, unique=True, nullable=False)
    
    address = Column(String)
    owner_name = Column(String)
    
    geometry = Column(JSON)
    area = Column(Float)
    
    land_use = Column(String)
    crop_type = Column(String)
    cultivation_status = Column(String)
    
    statistics = relationship("ParcelStatistics", back_populates="parcel", cascade="all, delete-orphan")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ParcelStatistics(Base):
    __tablename__ = "parcel_statistics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    parcel_id = Column(String, ForeignKey("parcels.id", ondelete="CASCADE"))
    analysis_id = Column(String, ForeignKey("analyses.id", ondelete="CASCADE"))
    
    crop_coverage_percent = Column(Float)
    facility_count = Column(Integer)
    cultivation_area = Column(Float)
    fallow_area = Column(Float)
    
    detailed_stats = Column(JSON)
    
    parcel = relationship("Parcel", back_populates="statistics")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Export(Base):
    __tablename__ = "exports"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    export_type = Column(String, nullable=False)
    
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    
    filepath = Column(String)
    file_size = Column(Integer)
    
    parameters = Column(JSON)
    error_message = Column(Text)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())