"""
Unit tests for Data Registry (POD1)
"""

import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
import asyncio

from src.pod1_data_ingestion import DataRegistry, ImageMetadata, CoordinateValidator
from src.pod1_data_ingestion.schemas import Bounds


class TestDataRegistry:
    """Test Data Registry functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def registry(self, temp_dir):
        """Create test registry"""
        return DataRegistry(storage_path=str(temp_dir))
    
    def test_registry_initialization(self, registry, temp_dir):
        """Test registry initialization"""
        assert registry.storage_path == temp_dir
        assert (temp_dir / "images").exists()
        assert (temp_dir / "shapes").exists()
        assert (temp_dir / "metadata").exists()
    
    def test_bounds_intersection(self):
        """Test bounds intersection calculation"""
        bounds1 = Bounds(minx=0, miny=0, maxx=10, maxy=10)
        bounds2 = Bounds(minx=5, miny=5, maxx=15, maxy=15)
        bounds3 = Bounds(minx=20, miny=20, maxx=30, maxy=30)
        
        assert bounds1.intersects(bounds2)
        assert not bounds1.intersects(bounds3)
    
    @pytest.mark.asyncio
    async def test_register_image_not_found(self, registry):
        """Test registering non-existent image"""
        with pytest.raises(FileNotFoundError):
            await registry.register_image("non_existent.tif")
    
    def test_get_statistics(self, registry):
        """Test getting registry statistics"""
        stats = registry.get_statistics()
        
        assert 'total_entries' in stats
        assert 'total_images' in stats
        assert 'total_shapes' in stats
        assert stats['total_entries'] == 0


class TestCoordinateValidator:
    """Test Coordinate Validator"""
    
    @pytest.fixture
    def validator(self):
        """Create test validator"""
        return CoordinateValidator()
    
    def test_validate_crs(self, validator):
        """Test CRS validation"""
        assert validator.validate_crs("EPSG:5186")
        assert validator.validate_crs("EPSG:4326")
        assert not validator.validate_crs("INVALID:CRS")
    
    def test_is_supported_crs(self, validator):
        """Test supported CRS check"""
        assert validator.is_supported_crs("EPSG:5186")
        assert validator.is_supported_crs("EPSG:4326")
        assert not validator.is_supported_crs("EPSG:9999")
    
    def test_transform_point(self, validator):
        """Test point transformation"""
        # Test identity transformation
        x, y = validator.transform_point(100, 200, "EPSG:5186")
        assert x == 100
        assert y == 200
    
    def test_transform_bounds(self, validator):
        """Test bounds transformation"""
        bounds = [0, 0, 100, 100]
        # Test identity transformation
        result = validator.transform_bounds(bounds, "EPSG:5186")
        assert result == bounds