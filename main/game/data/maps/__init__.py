"""
Maps Package - Skeleton Graph Road Generation System

This package implements a hierarchical, sparse road network generation system
based on skeleton graphs. All roads are EXACTLY 1 TILE WIDE.

MODULES:
--------
config.py
    Complete map generation configuration with all tunable parameters.
    Use MapConfig dataclass to customize generation.

utils.py
    Core utilities: grid operations, pathfinding, graph algorithms,
    line rasterization (Bresenham for 1-tile roads), noise generation.

skeleton_graph.py
    Skeleton graph road generation: node placement, graph construction,
    1-tile wide road rasterization, highway portal marking.

mapGen.py
    Main MapGenerator class orchestrating the complete pipeline:
    terrain → elevation → biomes → skeleton roads → local roads → urban.

terrain.py
    Terrain type definitions with gameplay properties.

tile.py
    Individual tile class for grid cells.

USAGE:
------
```python
from main.game.data.maps.config import MapConfig
from main.game.data.maps.mapGen import MapGenerator

# Create configuration
config = MapConfig(
    width=40,
    height=40,
    seed=12345,
    # Customize any parameter...
    node_min_spacing=8,
    skeleton_extra_edges=0.15,
    local_road_density=0.08,
)

# Generate map
generator = MapGenerator(config)
grid = generator.generate()

# Get statistics
stats = generator.get_statistics()
print(f"Generated {stats['highways']} highway tiles")
print(f"Generated {stats['local_roads']} local road tiles")
```

ARCHITECTURE:
-------------
1. **Node Generation**: Strategic placement using Poisson-disc sampling
2. **Graph Construction**: Delaunay → MST → add loops for connectivity
3. **Highway Rasterization**: Bresenham's algorithm for 1-tile wide roads
4. **Local Roads**: Block subdivision with grid patterns (1-tile wide)
5. **Urban Detection**: Find road-enclosed blocks and urbanize based on heat
6. **Buildings**: Scatter in urban blocks with spacing constraints

KEY FEATURES:
-------------
- ALL roads are exactly 1 tile wide (no multi-tile roads)
- Hierarchical structure (highways → local roads → blocks)
- Sparse, strategic networks (not dense/complete graphs)
- Heat map integration for strategic placement
- Biome-aware terrain generation
- Elevation system with automatic ramps
- Connectivity guarantees

DESIGN PHILOSOPHY:
------------------
Roads should be sparse and strategic, not dense networks. The skeleton graph
creates a backbone of highways, then local roads fill in blocks. This mimics
real urban planning: major arteries connect cities, local streets fill blocks.
"""

from main.game.data.maps.config import MapConfig
from main.game.data.maps.mapGen import MapGenerator, Biome
from main.game.data.maps.terrain import (
    terrain, plains, forest, urban, mountains,
    road, highway, debris, river, floodplain,
    bridge, wetland, ALL_TERRAIN_TYPES
)
from main.game.data.maps.tile import tile

__all__ = [
    # Configuration
    'MapConfig',

    # Generator
    'MapGenerator',
    'Biome',

    # Terrain
    'terrain',
    'plains',
    'forest',
    'urban',
    'mountains',
    'road',
    'highway',
    'debris',
    'river',
    'floodplain',
    'bridge',
    'wetland',
    'ALL_TERRAIN_TYPES',

    # Tile
    'tile',
]