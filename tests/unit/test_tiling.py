"""
Unit tests for Tiling Engine (POD3)
"""

import pytest
import numpy as np
from uuid import uuid4

from src.pod3_tiling import TilingEngine, TilingConfig, TileIndexer
from src.pod3_tiling.schemas import TilePosition, TileBounds


class TestTilingEngine:
    """Test Tiling Engine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create test tiling engine"""
        config = TilingConfig(tile_size=640, overlap=0.2)
        return TilingEngine(config)
    
    def test_calculate_grid_size(self, engine):
        """Test grid size calculation"""
        # Test exact fit
        rows, cols = engine.calculate_grid_size(1280, 1280)
        assert rows == 2
        assert cols == 2
        
        # Test with remainder
        rows, cols = engine.calculate_grid_size(1500, 1500)
        assert rows == 3
        assert cols == 3
    
    def test_stride_calculation(self):
        """Test stride calculation from overlap"""
        config = TilingConfig(tile_size=640, overlap=0.2)
        assert config.stride == 512  # 640 * 0.8
        
        config = TilingConfig(tile_size=640, overlap=0.0)
        assert config.stride == 640
    
    def test_tile_position_string(self):
        """Test tile position string formatting"""
        pos = TilePosition(row=1, col=2)
        assert pos.to_string() == "001_002"
    
    def test_tile_bounds_properties(self):
        """Test tile bounds properties"""
        bounds = TileBounds(
            pixel_bounds=(0, 0, 640, 640),
            geo_bounds=(100.0, 200.0, 200.0, 300.0)
        )
        assert bounds.width == 640
        assert bounds.height == 640


class TestTileIndexer:
    """Test Tile Indexer functionality"""
    
    @pytest.fixture
    def indexer(self):
        """Create test tile indexer"""
        return TileIndexer()
    
    def test_create_index(self, indexer):
        """Test creating spatial index"""
        image_id = uuid4()
        idx = indexer.create_index(image_id)
        
        assert image_id in indexer.indices
        assert image_id in indexer.tile_metadata
    
    def test_get_statistics_empty(self, indexer):
        """Test statistics for empty index"""
        image_id = uuid4()
        stats = indexer.get_statistics(image_id)
        assert stats == {}
    
    def test_clear_index(self, indexer):
        """Test clearing index"""
        image_id = uuid4()
        indexer.create_index(image_id)
        indexer.clear_index(image_id)
        
        assert image_id not in indexer.indices
        assert image_id not in indexer.tile_metadata


class TestTilingConfig:
    """Test Tiling Configuration"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = TilingConfig(tile_size=640, overlap=0.2)
        assert config.tile_size == 640
        assert config.overlap == 0.2
    
    def test_invalid_tile_size(self):
        """Test invalid tile size"""
        with pytest.raises(ValueError):
            TilingConfig(tile_size=-100)
        
        with pytest.raises(ValueError):
            TilingConfig(tile_size=641)  # Not divisible by 32
    
    def test_invalid_overlap(self):
        """Test invalid overlap"""
        with pytest.raises(ValueError):
            TilingConfig(overlap=-0.1)
        
        with pytest.raises(ValueError):
            TilingConfig(overlap=1.0)