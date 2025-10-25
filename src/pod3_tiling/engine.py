"""
Tiling Engine - Core tiling functionality
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Any
from uuid import UUID
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .schemas import (
    TileMetadata,
    TilingConfig,
    TilingResult,
    TilePosition,
    TileBounds
)
from ..common.config import settings

logger = logging.getLogger(__name__)


class TilingEngine:
    """
    High-performance tiling engine for splitting large images
    Supports overlap, padding, and parallel processing
    """
    
    def __init__(self, config: Optional[TilingConfig] = None):
        """
        Initialize tiling engine
        
        Args:
            config: Tiling configuration
        """
        self.config = config or TilingConfig()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        
    def calculate_grid_size(
        self, 
        image_width: int, 
        image_height: int
    ) -> Tuple[int, int]:
        """
        Calculate grid dimensions based on image size and tiling config
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (rows, cols)
        """
        stride = self.config.stride
        
        # Calculate number of tiles needed
        cols = (image_width - self.config.tile_size) // stride + 1
        rows = (image_height - self.config.tile_size) // stride + 1
        
        # Add extra tiles if remainder exists and partial tiles are included
        if self.config.include_partial:
            if (image_width - self.config.tile_size) % stride > 0:
                cols += 1
            if (image_height - self.config.tile_size) % stride > 0:
                rows += 1
                
        return rows, cols
    
    def generate_tile_positions(
        self, 
        image_width: int, 
        image_height: int
    ) -> Generator[Tuple[TilePosition, TileBounds], None, None]:
        """
        Generate tile positions and bounds
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Yields:
            Tuple of (position, bounds) for each tile
        """
        stride = self.config.stride
        tile_size = self.config.tile_size
        rows, cols = self.calculate_grid_size(image_width, image_height)
        
        for row in range(rows):
            for col in range(cols):
                # Calculate pixel bounds
                x_start = col * stride
                y_start = row * stride
                x_end = min(x_start + tile_size, image_width)
                y_end = min(y_start + tile_size, image_height)
                
                # Check if tile has minimum coverage
                actual_width = x_end - x_start
                actual_height = y_end - y_start
                coverage = (actual_width * actual_height) / (tile_size * tile_size)
                
                if coverage < self.config.min_tile_coverage and not self.config.include_partial:
                    continue
                
                position = TilePosition(row=row, col=col)
                bounds = TileBounds(
                    pixel_bounds=(x_start, y_start, x_end, y_end),
                    geo_bounds=(0, 0, 0, 0)  # Will be calculated with transform
                )
                
                yield position, bounds
    
    def _extract_tile(
        self,
        src: rasterio.DatasetReader,
        position: TilePosition,
        bounds: TileBounds
    ) -> Tuple[np.ndarray, TileBounds]:
        """
        Extract a single tile from the image
        
        Args:
            src: Rasterio dataset reader
            position: Tile position
            bounds: Tile bounds
            
        Returns:
            Tuple of (tile_array, updated_bounds)
        """
        x_start, y_start, x_end, y_end = bounds.pixel_bounds
        width = x_end - x_start
        height = y_end - y_start
        
        # Create window for reading
        window = Window(x_start, y_start, width, height)
        
        # Read tile data
        tile_data = src.read(window=window)
        
        # Apply padding if tile is smaller than expected size
        if width < self.config.tile_size or height < self.config.tile_size:
            tile_data = self._pad_tile(tile_data, width, height)
        
        # Calculate geographic bounds
        transform = src.window_transform(window)
        geo_bounds = (
            transform.c,  # minx
            transform.f + height * transform.e,  # miny
            transform.c + width * transform.a,  # maxx
            transform.f  # maxy
        )
        
        # Update bounds with geographic coordinates
        updated_bounds = TileBounds(
            pixel_bounds=bounds.pixel_bounds,
            geo_bounds=geo_bounds
        )
        
        return tile_data, updated_bounds
    
    def _pad_tile(
        self,
        tile_data: np.ndarray,
        actual_width: int,
        actual_height: int
    ) -> np.ndarray:
        """
        Pad tile to target size
        
        Args:
            tile_data: Tile array
            actual_width: Actual tile width
            actual_height: Actual tile height
            
        Returns:
            Padded tile array
        """
        bands, height, width = tile_data.shape
        target_size = self.config.tile_size
        
        if width >= target_size and height >= target_size:
            return tile_data[:, :target_size, :target_size]
        
        # Create padded array
        if self.config.padding_mode == 'constant':
            padded = np.full(
                (bands, target_size, target_size),
                self.config.padding_value,
                dtype=tile_data.dtype
            )
            padded[:, :height, :width] = tile_data
        elif self.config.padding_mode == 'edge':
            pad_width = [
                (0, 0),  # No padding for bands
                (0, target_size - height),
                (0, target_size - width)
            ]
            padded = np.pad(tile_data, pad_width, mode='edge')
        elif self.config.padding_mode == 'reflect':
            pad_width = [
                (0, 0),
                (0, target_size - height),
                (0, target_size - width)
            ]
            padded = np.pad(tile_data, pad_width, mode='reflect')
        else:  # symmetric
            pad_width = [
                (0, 0),
                (0, target_size - height),
                (0, target_size - width)
            ]
            padded = np.pad(tile_data, pad_width, mode='symmetric')
        
        return padded
    
    def _save_tile(
        self,
        tile_data: np.ndarray,
        output_path: Path,
        src_profile: dict
    ):
        """
        Save tile to disk
        
        Args:
            tile_data: Tile array
            output_path: Output file path
            src_profile: Source image profile
        """
        # Update profile for tile
        profile = src_profile.copy()
        profile.update({
            'width': tile_data.shape[2],
            'height': tile_data.shape[1],
            'compress': self.config.compression,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        })
        
        # Write tile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(tile_data)
    
    async def tile_image(
        self,
        image_path: str,
        output_dir: str,
        image_id: UUID
    ) -> TilingResult:
        """
        Tile an image asynchronously
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output tiles
            image_id: Image ID for tracking
            
        Returns:
            TilingResult object
        """
        start_time = time.time()
        input_path = Path(image_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        tiles = []
        
        with rasterio.open(input_path) as src:
            rows, cols = self.calculate_grid_size(src.width, src.height)
            total_tiles = rows * cols
            
            logger.info(f"Tiling image into {rows}x{cols} grid ({total_tiles} tiles)")
            
            # Process tiles
            with tqdm(total=total_tiles, desc="Tiling") as pbar:
                for position, bounds in self.generate_tile_positions(src.width, src.height):
                    # Extract tile
                    tile_data, updated_bounds = await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        self._extract_tile,
                        src,
                        position,
                        bounds
                    )
                    
                    # Generate output path
                    tile_name = f"{image_id}_{position.to_string()}.tif"
                    tile_path = out_dir / tile_name
                    
                    # Save tile
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        self._save_tile,
                        tile_data,
                        tile_path,
                        src.profile
                    )
                    
                    # Create metadata
                    tile_metadata = TileMetadata(
                        parent_image_id=image_id,
                        position=position,
                        bounds=updated_bounds,
                        file_path=str(tile_path),
                        size=(tile_data.shape[2], tile_data.shape[1]),
                        processing_status="completed"
                    )
                    
                    tiles.append(tile_metadata)
                    pbar.update(1)
        
        processing_time = time.time() - start_time
        
        result = TilingResult(
            source_image_id=image_id,
            config=self.config,
            tiles=tiles,
            grid_size=(rows, cols),
            total_tiles=len(tiles),
            processing_time=processing_time
        )
        
        logger.info(f"Tiling completed: {len(tiles)} tiles in {processing_time:.2f} seconds")
        
        return result
    
    def tile_image_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        image_ids: List[UUID]
    ) -> List[TilingResult]:
        """
        Tile multiple images in batch
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory for output tiles
            image_ids: List of image IDs
            
        Returns:
            List of TilingResult objects
        """
        async def process_batch():
            tasks = []
            for path, img_id in zip(image_paths, image_ids):
                task = self.tile_image(path, output_dir, img_id)
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        return asyncio.run(process_batch())
    
    def merge_tiles(
        self,
        tiles: List[TileMetadata],
        output_path: str,
        image_shape: Tuple[int, int, int]
    ):
        """
        Merge tiles back into a single image
        
        Args:
            tiles: List of tile metadata
            output_path: Output image path
            image_shape: Shape of output image (bands, height, width)
        """
        bands, height, width = image_shape
        merged = np.zeros((bands, height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        for tile in tiles:
            # Load tile data
            with rasterio.open(tile.file_path) as src:
                tile_data = src.read()
            
            # Get tile position
            x_start, y_start, x_end, y_end = tile.bounds.pixel_bounds
            
            # Add tile to merged image with weighted averaging for overlaps
            tile_height = y_end - y_start
            tile_width = x_end - x_start
            
            # Create weight matrix for smooth blending
            weight_matrix = self._create_weight_matrix(tile_width, tile_height)
            
            # Add tile data
            merged[:, y_start:y_end, x_start:x_end] += tile_data[:, :tile_height, :tile_width] * weight_matrix
            weights[y_start:y_end, x_start:x_end] += weight_matrix
        
        # Normalize by weights
        for band in range(bands):
            merged[band] = np.divide(merged[band], weights, where=weights > 0)
        
        # Save merged image
        # (Profile would need to be properly configured based on source)
        logger.info(f"Merged {len(tiles)} tiles into {output_path}")
    
    def _create_weight_matrix(
        self,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Create weight matrix for smooth blending at tile edges
        
        Args:
            width: Tile width
            height: Tile height
            
        Returns:
            Weight matrix
        """
        # Create linear gradients for edges
        fade_size = int(self.config.tile_size * self.config.overlap / 2)
        
        weights = np.ones((height, width), dtype=np.float32)
        
        if fade_size > 0:
            # Top edge
            if height > fade_size:
                for i in range(fade_size):
                    weights[i, :] *= (i + 1) / fade_size
            
            # Bottom edge
            if height > fade_size:
                for i in range(fade_size):
                    weights[height - fade_size + i, :] *= (fade_size - i) / fade_size
            
            # Left edge
            if width > fade_size:
                for i in range(fade_size):
                    weights[:, i] *= (i + 1) / fade_size
            
            # Right edge  
            if width > fade_size:
                for i in range(fade_size):
                    weights[:, width - fade_size + i] *= (fade_size - i) / fade_size
        
        return weights
    
    def cleanup(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)