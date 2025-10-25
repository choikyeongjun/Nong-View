"""
Schemas for tiling module
"""

from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class TilePosition(BaseModel):
    """Position of tile in the grid"""
    row: int
    col: int
    
    def to_string(self) -> str:
        """Convert to string format for naming"""
        return f"{self.row:03d}_{self.col:03d}"


class TileBounds(BaseModel):
    """Tile boundary information"""
    pixel_bounds: Tuple[int, int, int, int]  # minx, miny, maxx, maxy in pixels
    geo_bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy in CRS units
    
    @property
    def width(self) -> int:
        """Get tile width in pixels"""
        return self.pixel_bounds[2] - self.pixel_bounds[0]
    
    @property
    def height(self) -> int:
        """Get tile height in pixels"""
        return self.pixel_bounds[3] - self.pixel_bounds[1]


class TileMetadata(BaseModel):
    """Metadata for a single tile"""
    tile_id: UUID = Field(default_factory=uuid4)
    parent_image_id: UUID
    position: TilePosition
    bounds: TileBounds
    file_path: Optional[str] = None
    size: Tuple[int, int]  # width, height
    overlap_info: Dict[str, Any] = Field(default_factory=dict)
    processing_status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('processing_status')
    def validate_status(cls, v):
        """Validate processing status"""
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v
    
    def get_neighbors(self) -> Dict[str, TilePosition]:
        """Get positions of neighboring tiles"""
        row, col = self.position.row, self.position.col
        return {
            'top': TilePosition(row=row-1, col=col),
            'bottom': TilePosition(row=row+1, col=col),
            'left': TilePosition(row=row, col=col-1),
            'right': TilePosition(row=row, col=col+1),
            'top_left': TilePosition(row=row-1, col=col-1),
            'top_right': TilePosition(row=row-1, col=col+1),
            'bottom_left': TilePosition(row=row+1, col=col-1),
            'bottom_right': TilePosition(row=row+1, col=col+1)
        }


class TilingConfig(BaseModel):
    """Configuration for tiling operation"""
    tile_size: int = Field(default=640, description="Tile size in pixels")
    overlap: float = Field(default=0.2, description="Overlap ratio (0.0-1.0)")
    padding_mode: str = Field(default="constant", description="Padding mode for edge tiles")
    padding_value: int = Field(default=0, description="Padding value if mode is constant")
    output_format: str = Field(default="TIFF", description="Output format for tiles")
    compression: Optional[str] = Field(default="LZW", description="Compression method")
    include_partial: bool = Field(default=True, description="Include partial tiles at edges")
    min_tile_coverage: float = Field(default=0.1, description="Minimum coverage ratio for partial tiles")
    
    @validator('tile_size')
    def validate_tile_size(cls, v):
        """Validate tile size"""
        if v <= 0:
            raise ValueError(f"Tile size must be positive: {v}")
        if v % 32 != 0:
            raise ValueError(f"Tile size should be divisible by 32 for optimal performance: {v}")
        return v
    
    @validator('overlap')
    def validate_overlap(cls, v):
        """Validate overlap ratio"""
        if not 0.0 <= v < 1.0:
            raise ValueError(f"Overlap must be between 0.0 and 1.0: {v}")
        return v
    
    @validator('padding_mode')
    def validate_padding_mode(cls, v):
        """Validate padding mode"""
        valid_modes = ['constant', 'edge', 'reflect', 'symmetric']
        if v not in valid_modes:
            raise ValueError(f"Invalid padding mode: {v}")
        return v
    
    @property
    def stride(self) -> int:
        """Calculate stride based on tile size and overlap"""
        return int(self.tile_size * (1 - self.overlap))


class TilingResult(BaseModel):
    """Result of tiling operation"""
    source_image_id: UUID
    config: TilingConfig
    tiles: List[TileMetadata]
    grid_size: Tuple[int, int]  # rows, cols
    total_tiles: int
    processing_time: float  # seconds
    created_at: datetime = Field(default_factory=datetime.now)
    
    def get_tile_by_position(self, row: int, col: int) -> Optional[TileMetadata]:
        """Get tile by grid position"""
        for tile in self.tiles:
            if tile.position.row == row and tile.position.col == col:
                return tile
        return None
    
    def get_coverage_map(self) -> Dict[str, Any]:
        """Get coverage statistics"""
        covered_area = sum(
            (tile.bounds.width * tile.bounds.height) 
            for tile in self.tiles
        )
        
        # Calculate unique coverage (accounting for overlaps)
        # This is simplified - actual implementation would be more complex
        unique_coverage = covered_area * (1 - self.config.overlap) ** 2
        
        return {
            'total_tiles': self.total_tiles,
            'grid_size': self.grid_size,
            'covered_area': covered_area,
            'unique_coverage': unique_coverage,
            'overlap_ratio': self.config.overlap
        }


class TileIndex(BaseModel):
    """Spatial index for tiles"""
    image_id: UUID
    tiles: Dict[str, TileMetadata]  # key: "row_col"
    spatial_index: Optional[Any] = None  # R-tree index
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_tile(self, tile: TileMetadata):
        """Add tile to index"""
        key = f"{tile.position.row}_{tile.position.col}"
        self.tiles[key] = tile
    
    def get_tiles_in_bounds(
        self, 
        bounds: Tuple[float, float, float, float]
    ) -> List[TileMetadata]:
        """Get tiles intersecting with bounds"""
        result = []
        for tile in self.tiles.values():
            tile_bounds = tile.bounds.geo_bounds
            # Check intersection
            if not (tile_bounds[2] < bounds[0] or 
                   tile_bounds[0] > bounds[2] or
                   tile_bounds[3] < bounds[1] or
                   tile_bounds[1] > bounds[3]):
                result.append(tile)
        return result
    
    def get_adjacent_tiles(
        self, 
        tile: TileMetadata
    ) -> Dict[str, Optional[TileMetadata]]:
        """Get adjacent tiles"""
        neighbors = tile.get_neighbors()
        result = {}
        for direction, pos in neighbors.items():
            key = f"{pos.row}_{pos.col}"
            result[direction] = self.tiles.get(key)
        return result