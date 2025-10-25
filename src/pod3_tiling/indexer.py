"""
Tile Indexer - Spatial indexing for efficient tile retrieval
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from uuid import UUID
import pickle
from pathlib import Path
import numpy as np
from rtree import index

from .schemas import TileMetadata, TileIndex, TilePosition

logger = logging.getLogger(__name__)


class TileIndexer:
    """
    Spatial indexer for tiles using R-tree
    Enables efficient spatial queries for tile retrieval
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize tile indexer
        
        Args:
            index_path: Optional path to persist index
        """
        self.index_path = Path(index_path) if index_path else None
        self.indices: Dict[UUID, index.Index] = {}
        self.tile_metadata: Dict[UUID, Dict[int, TileMetadata]] = {}
        
        # R-tree properties
        self.properties = index.Property()
        self.properties.dimension = 2
        self.properties.variant = index.RT_Variant.RT_Quadratic
        self.properties.fill_factor = 0.7
        
        if self.index_path:
            self._load_indices()
    
    def create_index(self, image_id: UUID) -> index.Index:
        """
        Create a new spatial index for an image
        
        Args:
            image_id: Image ID
            
        Returns:
            R-tree index
        """
        idx = index.Index(properties=self.properties)
        self.indices[image_id] = idx
        self.tile_metadata[image_id] = {}
        return idx
    
    def add_tiles(self, image_id: UUID, tiles: List[TileMetadata]):
        """
        Add tiles to spatial index
        
        Args:
            image_id: Image ID
            tiles: List of tile metadata
        """
        if image_id not in self.indices:
            self.create_index(image_id)
        
        idx = self.indices[image_id]
        metadata = self.tile_metadata[image_id]
        
        for i, tile in enumerate(tiles):
            # Use geographic bounds for spatial indexing
            bounds = tile.bounds.geo_bounds
            idx.insert(i, bounds)
            metadata[i] = tile
            
        logger.info(f"Added {len(tiles)} tiles to index for image {image_id}")
    
    def find_tiles_by_bounds(
        self,
        image_id: UUID,
        bounds: Tuple[float, float, float, float],
        mode: str = 'intersects'
    ) -> List[TileMetadata]:
        """
        Find tiles by spatial bounds
        
        Args:
            image_id: Image ID
            bounds: Query bounds (minx, miny, maxx, maxy)
            mode: Query mode ('intersects' or 'contains')
            
        Returns:
            List of matching tiles
        """
        if image_id not in self.indices:
            logger.warning(f"No index found for image {image_id}")
            return []
        
        idx = self.indices[image_id]
        metadata = self.tile_metadata[image_id]
        
        if mode == 'intersects':
            hits = list(idx.intersection(bounds))
        elif mode == 'contains':
            # Find tiles completely contained within bounds
            hits = []
            for i in idx.intersection(bounds):
                tile_bounds = metadata[i].bounds.geo_bounds
                if (tile_bounds[0] >= bounds[0] and 
                    tile_bounds[1] >= bounds[1] and
                    tile_bounds[2] <= bounds[2] and
                    tile_bounds[3] <= bounds[3]):
                    hits.append(i)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return [metadata[i] for i in hits]
    
    def find_tiles_by_position(
        self,
        image_id: UUID,
        position: TilePosition
    ) -> Optional[TileMetadata]:
        """
        Find tile by grid position
        
        Args:
            image_id: Image ID
            position: Tile position
            
        Returns:
            Tile metadata if found
        """
        if image_id not in self.tile_metadata:
            return None
        
        metadata = self.tile_metadata[image_id]
        
        for tile in metadata.values():
            if (tile.position.row == position.row and 
                tile.position.col == position.col):
                return tile
        
        return None
    
    def find_neighboring_tiles(
        self,
        image_id: UUID,
        tile: TileMetadata,
        distance: float = 0.0
    ) -> Dict[str, Optional[TileMetadata]]:
        """
        Find neighboring tiles
        
        Args:
            image_id: Image ID
            tile: Reference tile
            distance: Additional search distance
            
        Returns:
            Dictionary of neighboring tiles by direction
        """
        neighbors = {}
        neighbor_positions = tile.get_neighbors()
        
        for direction, position in neighbor_positions.items():
            neighbor = self.find_tiles_by_position(image_id, position)
            neighbors[direction] = neighbor
        
        return neighbors
    
    def get_tile_coverage_map(
        self,
        image_id: UUID,
        resolution: int = 100
    ) -> np.ndarray:
        """
        Generate coverage map showing tile distribution
        
        Args:
            image_id: Image ID
            resolution: Resolution of coverage map
            
        Returns:
            2D array showing coverage
        """
        if image_id not in self.tile_metadata:
            return np.zeros((resolution, resolution))
        
        metadata = self.tile_metadata[image_id]
        
        # Find overall bounds
        all_bounds = [tile.bounds.geo_bounds for tile in metadata.values()]
        if not all_bounds:
            return np.zeros((resolution, resolution))
        
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        
        # Create coverage map
        coverage = np.zeros((resolution, resolution), dtype=np.int32)
        
        for tile in metadata.values():
            bounds = tile.bounds.geo_bounds
            
            # Convert to map coordinates
            x1 = int((bounds[0] - min_x) / (max_x - min_x) * (resolution - 1))
            y1 = int((bounds[1] - min_y) / (max_y - min_y) * (resolution - 1))
            x2 = int((bounds[2] - min_x) / (max_x - min_x) * (resolution - 1))
            y2 = int((bounds[3] - min_y) / (max_y - min_y) * (resolution - 1))
            
            coverage[y1:y2+1, x1:x2+1] += 1
        
        return coverage
    
    def calculate_overlap_matrix(
        self,
        image_id: UUID
    ) -> Dict[Tuple[int, int], float]:
        """
        Calculate overlap between tiles
        
        Args:
            image_id: Image ID
            
        Returns:
            Dictionary of tile pair overlaps
        """
        if image_id not in self.tile_metadata:
            return {}
        
        metadata = self.tile_metadata[image_id]
        overlaps = {}
        
        tiles = list(metadata.values())
        
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                tile1 = tiles[i]
                tile2 = tiles[j]
                
                # Check if tiles overlap
                b1 = tile1.bounds.geo_bounds
                b2 = tile2.bounds.geo_bounds
                
                # Calculate intersection
                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                
                if x_overlap > 0 and y_overlap > 0:
                    overlap_area = x_overlap * y_overlap
                    tile1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    tile2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    
                    # Calculate overlap ratio (IoU)
                    union_area = tile1_area + tile2_area - overlap_area
                    iou = overlap_area / union_area if union_area > 0 else 0
                    
                    overlaps[(i, j)] = iou
        
        return overlaps
    
    def optimize_tile_selection(
        self,
        image_id: UUID,
        bounds: Tuple[float, float, float, float],
        max_tiles: Optional[int] = None
    ) -> List[TileMetadata]:
        """
        Optimize tile selection to minimize redundancy
        
        Args:
            image_id: Image ID
            bounds: Query bounds
            max_tiles: Maximum number of tiles to return
            
        Returns:
            Optimized list of tiles
        """
        # Get all intersecting tiles
        tiles = self.find_tiles_by_bounds(image_id, bounds)
        
        if not tiles or (max_tiles and len(tiles) <= max_tiles):
            return tiles
        
        # Calculate coverage score for each tile
        scores = []
        for tile in tiles:
            tile_bounds = tile.bounds.geo_bounds
            
            # Calculate intersection area with query bounds
            x_overlap = max(0, min(tile_bounds[2], bounds[2]) - max(tile_bounds[0], bounds[0]))
            y_overlap = max(0, min(tile_bounds[3], bounds[3]) - max(tile_bounds[1], bounds[1]))
            intersection_area = x_overlap * y_overlap
            
            tile_area = (tile_bounds[2] - tile_bounds[0]) * (tile_bounds[3] - tile_bounds[1])
            
            # Score based on coverage ratio
            score = intersection_area / tile_area if tile_area > 0 else 0
            scores.append((score, tile))
        
        # Sort by score and select top tiles
        scores.sort(reverse=True, key=lambda x: x[0])
        
        if max_tiles:
            selected = [tile for _, tile in scores[:max_tiles]]
        else:
            # Select tiles that provide good coverage
            selected = []
            covered_area = 0
            target_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            
            for score, tile in scores:
                selected.append(tile)
                covered_area += score * target_area
                
                # Stop when we have good coverage
                if covered_area >= target_area * 0.95:
                    break
        
        return selected
    
    def get_statistics(self, image_id: UUID) -> Dict[str, Any]:
        """
        Get index statistics
        
        Args:
            image_id: Image ID
            
        Returns:
            Statistics dictionary
        """
        if image_id not in self.tile_metadata:
            return {}
        
        metadata = self.tile_metadata[image_id]
        
        if not metadata:
            return {}
        
        # Calculate statistics
        all_bounds = [tile.bounds.geo_bounds for tile in metadata.values()]
        
        stats = {
            'total_tiles': len(metadata),
            'min_x': min(b[0] for b in all_bounds),
            'min_y': min(b[1] for b in all_bounds),
            'max_x': max(b[2] for b in all_bounds),
            'max_y': max(b[3] for b in all_bounds),
            'avg_tile_area': np.mean([
                (b[2] - b[0]) * (b[3] - b[1]) for b in all_bounds
            ]),
            'coverage_area': sum(
                (b[2] - b[0]) * (b[3] - b[1]) for b in all_bounds
            )
        }
        
        # Add grid statistics
        rows = set(tile.position.row for tile in metadata.values())
        cols = set(tile.position.col for tile in metadata.values())
        
        stats.update({
            'grid_rows': len(rows),
            'grid_cols': len(cols),
            'grid_completeness': len(metadata) / (len(rows) * len(cols))
        })
        
        return stats
    
    def save_index(self, image_id: UUID):
        """
        Save index to disk
        
        Args:
            image_id: Image ID
        """
        if not self.index_path:
            return
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = self.index_path / f"{image_id}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.tile_metadata.get(image_id, {}), f)
        
        logger.info(f"Saved index for image {image_id}")
    
    def load_index(self, image_id: UUID):
        """
        Load index from disk
        
        Args:
            image_id: Image ID
        """
        if not self.index_path:
            return
        
        metadata_file = self.index_path / f"{image_id}_metadata.pkl"
        
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Rebuild R-tree index
            idx = self.create_index(image_id)
            for i, tile in metadata.items():
                idx.insert(i, tile.bounds.geo_bounds)
                self.tile_metadata[image_id][i] = tile
            
            logger.info(f"Loaded index for image {image_id}")
    
    def _load_indices(self):
        """Load all indices from disk"""
        if not self.index_path or not self.index_path.exists():
            return
        
        for metadata_file in self.index_path.glob("*_metadata.pkl"):
            image_id = UUID(metadata_file.stem.replace("_metadata", ""))
            self.load_index(image_id)
    
    def clear_index(self, image_id: UUID):
        """
        Clear index for an image
        
        Args:
            image_id: Image ID
        """
        if image_id in self.indices:
            del self.indices[image_id]
        if image_id in self.tile_metadata:
            del self.tile_metadata[image_id]
        
        # Remove from disk if exists
        if self.index_path:
            metadata_file = self.index_path / f"{image_id}_metadata.pkl"
            if metadata_file.exists():
                metadata_file.unlink()