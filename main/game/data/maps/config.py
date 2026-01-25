"""
Map generation configuration with skeleton-graph road system.

ROAD GENERATION PHILOSOPHY:
- Skeleton graph creates sparse, strategic highway network (1-tile wide)
- Local roads fill in blocks with grid patterns (1-tile wide)
- Urban blocks detected from road-enclosed regions
- ALL ROADS ARE EXACTLY 1 TILE WIDE

This configuration exposes every tunable parameter for the generation system.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np


@dataclass
class MapConfig:
    """
    Complete map generation configuration.

    All parameters are exposed and documented for fine-tuning generation.
    """

    # ========================================================================
    # BASIC MAP PARAMETERS
    # ========================================================================

    width: int = 20  # Map width in tiles
    height: int = 15  # Map height in tiles
    seed: Optional[int] = None  # Random seed (None = random)

    # ========================================================================
    # TERRAIN DENSITY TARGETS (fraction of total tiles)
    # ========================================================================

    forest_density: float = 0.20  # Forest coverage (0.0-0.6)
    urban_density: float = 0.08  # Urban area coverage (0.0-0.4)
    mountain_density: float = 0.06  # Mountain coverage (0.0-0.25)
    road_density: float = 0.05  # Road coverage target (0.0-0.2)
    debris_density: float = 0.04  # Debris coverage (0.0-0.15)

    # ========================================================================
    # ELEVATION SYSTEM
    # ========================================================================

    elevation_density: float = 0.15  # Fraction of map with elevation > 0
    max_elevation: int = 3  # Maximum elevation level (1-5)
    min_support_neighbors: int = 2  # Min neighbors at level N-1 to promote to N
    min_ramp_neighbors: int = 1  # Min ramps needed to promote tile
    ramp_placement_probability: float = 0.75  # Chance to place ramp (0.0-1.0)

    # Road elevation handling
    road_elevation_bypass_threshold: int = 2  # Max elevation diff roads ignore
    road_force_ramp: bool = True  # Auto-place ramps on roads
    road_max_elevation_cross: int = 1  # Max elevation change roads can cross

    # ========================================================================
    # STRATEGIC ELEMENTS
    # ========================================================================

    building_density: float = 0.04  # Building coverage (0.0-0.2)
    control_zone_count: int = 5  # Number of control zones (3-11, prefer odd)
    min_building_spacing: int = 3  # Min tiles between buildings

    # Building distribution
    building_heat_influence: float = 0.6  # Heat map influence on buildings
    building_variety_bonus: float = 0.4  # Bonus for building diversity

    # ========================================================================
    # SMOOTHING AND CLUSTERING
    # ========================================================================

    smoothing_passes: int = 2  # Terrain smoothing iterations (0-6)
    cluster_min_size: int = 3  # Min cluster size before removal

    # ========================================================================
    # BIOME GENERATION
    # ========================================================================

    biome_count_min: int = 2  # Minimum biomes (1-6)
    biome_count_max: int = 4  # Maximum biomes (min-8)
    biome_noise_scale: float = 0.07  # Noise scale for blending (0.0-1.0)
    biome_noise_octaves: int = 2  # Noise octaves (1-6)
    biome_min_radius: int = 4  # Minimum biome radius in tiles
    biome_heat_scaling: bool = True  # Scale biome size with heat
    biome_heat_scale_factor: float = 0.5  # Heat scaling strength (0.0-1.0)

    # ========================================================================
    # SKELETON GRAPH - NODE GENERATION
    # ========================================================================

    use_skeleton_graph: bool = True  # Enable skeleton graph system

    # Node placement method: "poisson_disc", "grid_noise", "city_districts"
    node_generation_method: str = "poisson_disc"

    # Node spacing (-1 = auto-calculate from map size)
    node_min_spacing: int = -1

    # Target node count (-1 = auto-calculate from map size)
    node_count_target: int = -1

    # Node placement acceptance probability range
    node_heat_preference: float = 0.5  # Prefer heat=0.5 (moderate zones)
    node_heat_tolerance: float = 0.3  # Acceptance range around preference

    # ========================================================================
    # SKELETON GRAPH - GRAPH CONSTRUCTION
    # ========================================================================

    # Construction method: "delaunay_mst", "nearest_neighbor", "astar_mesh"
    skeleton_method: str = "delaunay_mst"

    # Edges to add back after MST (creates loops) (0.0-0.5)
    skeleton_extra_edges: float = 0.15

    # Prefer straight edges (1.0 = strong preference) (0.0-1.0)
    skeleton_straightness_bias: float = 0.90

    # Penalize diagonal edges (higher = stronger penalty) (1.0-3.0)
    skeleton_diagonal_penalty: float = 1.5

    # K-nearest neighbors for local connectivity (2-8)
    skeleton_k_nearest: int = 4

    # ========================================================================
    # SKELETON GRAPH - RASTERIZATION (1-TILE WIDE)
    # ========================================================================

    # Rasterization method: "bresenham", "straight_only"
    highway_rasterization_method: str = "bresenham"

    # Smooth sharp corners in skeleton edges
    highway_curve_smoothing: bool = True

    # Corner smoothing radius (tiles)
    highway_corner_radius: int = 2

    # IMPORTANT: All skeleton roads are EXACTLY 1 tile wide
    # No width parameter - this ensures strategic, sparse networks

    # ========================================================================
    # LOCAL ROADS (CONSTRAINED, 1-TILE WIDE)
    # ========================================================================

    generate_local_roads: bool = True  # Enable local road generation

    # Local road density target (fraction of non-highway tiles)
    local_road_density: float = 0.08  # (0.0-0.15)

    # Minimum spacing between parallel local roads
    local_road_min_spacing: int = 3  # (2-8 tiles)

    # Minimum distance from local roads to highways
    local_road_to_highway_spacing: int = 2  # (1-5 tiles)

    # Can local roads only connect at designated portals?
    local_road_intersection_portals: bool = True

    # Spacing between highway connection portals
    local_road_portal_spacing: int = 8  # (4-20 tiles)

    # Local road generation method: "block_grid", "random_walk"
    local_fill_method: str = "block_grid"

    # Block subdivision for grid method
    local_grid_spacing: int = 5  # Grid spacing in tiles (3-10)

    # ========================================================================
    # URBAN BLOCK GENERATION
    # ========================================================================

    # Urban generation method: "block_fill", "road_growth"
    urban_generation_method: str = "block_fill"

    # Fraction of suitable blocks to urbanize (0.0-1.0)
    urban_block_fill_density: float = 0.6

    # Minimum block size to consider for urbanization
    urban_block_min_size: int = 4  # (3-15 tiles)

    # Chance to place building in urban block tile (0.0-1.0)
    urban_building_in_block_chance: float = 0.3

    # Urban preference for high-heat areas
    urban_heat_bias: float = 0.7  # (0.0-1.0)

    # ========================================================================
    # HEAT MAP CONFIGURATION
    # ========================================================================

    use_heat_map: bool = True  # Enable heat map generation
    heat_decay_rate: float = 0.15  # Heat decay rate (0.05-0.5)
    heat_influence_radius: int = -1  # Heat influence radius (-1 = auto)

    # ========================================================================
    # PATHFINDING AND PERFORMANCE
    # ========================================================================

    smoothing_enabled: bool = True  # Enable terrain smoothing
    seed_retry_attempts: int = 100  # Max retries for element placement

    # ========================================================================
    # SPAWN POINTS
    # ========================================================================

    spawn_point_1: Optional[Tuple[int, int]] = None
    spawn_point_2: Optional[Tuple[int, int]] = None

    # ========================================================================
    # ADVANCED: BIOME DEFINITIONS OVERRIDE
    # ========================================================================

    biome_definitions: Optional[List[Dict]] = field(default=None)

    # ========================================================================
    # VALIDATION AND AUTO-CALCULATION
    # ========================================================================

    def __post_init__(self):
        """Validate and auto-calculate configuration values."""

        # Helper for integer clipping with type preservation
        def clip_int(value, lo, hi):
            return int(np.clip(value, lo, hi))

        # Helper for float clipping
        def clip_float(value, lo, hi):
            return float(np.clip(value, lo, hi))

        # Clamp basic dimensions
        self.width = clip_int(self.width, 8, 256)
        self.height = clip_int(self.height, 8, 256)

        # Clamp densities (use numpy.clip for concise clamping)
        self.forest_density = clip_float(self.forest_density, 0.0, 0.6)
        self.urban_density = clip_float(self.urban_density, 0.0, 0.4)
        self.mountain_density = clip_float(self.mountain_density, 0.0, 0.25)
        self.road_density = clip_float(self.road_density, 0.0, 0.2)
        self.debris_density = clip_float(self.debris_density, 0.0, 0.15)
        self.elevation_density = clip_float(self.elevation_density, 0.0, 0.6)
        self.building_density = clip_float(self.building_density, 0.0, 0.2)

        # Elevation parameters
        self.max_elevation = clip_int(self.max_elevation, 1, 5)
        self.min_support_neighbors = clip_int(self.min_support_neighbors, 1, 4)
        self.min_ramp_neighbors = clip_int(self.min_ramp_neighbors, 0, 3)
        self.ramp_placement_probability = clip_float(self.ramp_placement_probability, 0.0, 1.0)
        self.road_elevation_bypass_threshold = clip_int(self.road_elevation_bypass_threshold, 1, 5)
        self.road_max_elevation_cross = clip_int(self.road_max_elevation_cross, 0, 3)

        # Control zones (prefer odd)
        if self.control_zone_count % 2 == 0:
            self.control_zone_count += 1
        self.control_zone_count = clip_int(self.control_zone_count, 3, 11)

        # Spacing and smoothing
        self.min_building_spacing = clip_int(self.min_building_spacing, 2, 6)
        self.smoothing_passes = clip_int(self.smoothing_passes, 0, 6)
        self.cluster_min_size = clip_int(self.cluster_min_size, 1, 12)

        # Building distribution
        self.building_heat_influence = clip_float(self.building_heat_influence, 0.0, 1.0)
        self.building_variety_bonus = clip_float(self.building_variety_bonus, 0.0, 1.0)

        # Biomes
        self.biome_count_min = clip_int(self.biome_count_min, 1, 6)
        self.biome_count_max = max(self.biome_count_min, clip_int(self.biome_count_max, 1, 8))
        self.biome_noise_scale = clip_float(self.biome_noise_scale, 0.0, 1.0)
        self.biome_noise_octaves = clip_int(self.biome_noise_octaves, 1, 6)
        self.biome_min_radius = clip_int(self.biome_min_radius, 3, 20)
        self.biome_heat_scale_factor = clip_float(self.biome_heat_scale_factor, 0.0, 1.0)

        # Auto-calculate skeleton graph parameters
        if self.node_min_spacing == -1:
            # ~6-8% of average dimension
            avg_dim = (self.width + self.height) / 2
            self.node_min_spacing = max(6, int(avg_dim * 0.07))
        else:
            self.node_min_spacing = clip_int(self.node_min_spacing, 4, 30)

        if self.node_count_target == -1:
            # Based on map area and spacing
            area = self.width * self.height
            self.node_count_target = max(4, int(area / (self.node_min_spacing ** 2)))
        else:
            self.node_count_target = clip_int(self.node_count_target, 3, 50)

        # Skeleton graph parameters
        self.skeleton_extra_edges = clip_float(self.skeleton_extra_edges, 0.0, 0.5)
        self.skeleton_straightness_bias = clip_float(self.skeleton_straightness_bias, 0.0, 1.0)
        self.skeleton_diagonal_penalty = clip_float(self.skeleton_diagonal_penalty, 1.0, 3.0)
        self.skeleton_k_nearest = clip_int(self.skeleton_k_nearest, 2, 8)
        self.node_heat_preference = clip_float(self.node_heat_preference, 0.0, 1.0)
        self.node_heat_tolerance = clip_float(self.node_heat_tolerance, 0.0, 1.0)

        # Highway rasterization
        self.highway_corner_radius = clip_int(self.highway_corner_radius, 1, 5)

        # Local roads
        self.local_road_density = clip_float(self.local_road_density, 0.0, 0.15)
        self.local_road_min_spacing = clip_int(self.local_road_min_spacing, 2, 8)
        self.local_road_to_highway_spacing = clip_int(self.local_road_to_highway_spacing, 1, 5)
        self.local_road_portal_spacing = clip_int(self.local_road_portal_spacing, 4, 20)
        self.local_grid_spacing = clip_int(self.local_grid_spacing, 3, 10)

        # Urban blocks
        self.urban_block_fill_density = clip_float(self.urban_block_fill_density, 0.0, 1.0)
        self.urban_block_min_size = clip_int(self.urban_block_min_size, 3, 15)
        self.urban_building_in_block_chance = clip_float(self.urban_building_in_block_chance, 0.0, 1.0)
        self.urban_heat_bias = clip_float(self.urban_heat_bias, 0.0, 1.0)

        # Heat map
        self.heat_decay_rate = clip_float(self.heat_decay_rate, 0.05, 0.5)
        if self.heat_influence_radius == -1:
            self.heat_influence_radius = max(self.width, self.height) // 2

        # Validate method strings (keep as-is if invalid)
        valid_node_methods = ["poisson_disc", "grid_noise", "city_districts"]
        if self.node_generation_method not in valid_node_methods:
            self.node_generation_method = "poisson_disc"

        valid_skeleton_methods = ["delaunay_mst", "nearest_neighbor", "astar_mesh"]
        if self.skeleton_method not in valid_skeleton_methods:
            self.skeleton_method = "delaunay_mst"

        valid_raster_methods = ["bresenham", "straight_only"]
        if self.highway_rasterization_method not in valid_raster_methods:
            self.highway_rasterization_method = "bresenham"

        valid_fill_methods = ["block_grid", "random_walk"]
        if self.local_fill_method not in valid_fill_methods:
            self.local_fill_method = "block_grid"

        valid_urban_methods = ["block_fill", "road_growth"]
        if self.urban_generation_method not in valid_urban_methods:
            self.urban_generation_method = "block_fill"

        # Default spawn points
        if self.spawn_point_1 is None:
            self.spawn_point_1 = (self.width // 6, self.height // 6)
        if self.spawn_point_2 is None:
            self.spawn_point_2 = (5 * self.width // 6, 5 * self.height // 6)

        # Validate spawn points (use numpy.clip for clarity)
        x1 = int(np.clip(self.spawn_point_1[0], 0, self.width - 1))
        y1 = int(np.clip(self.spawn_point_1[1], 0, self.height - 1))
        x2 = int(np.clip(self.spawn_point_2[0], 0, self.width - 1))
        y2 = int(np.clip(self.spawn_point_2[1], 0, self.height - 1))
        self.spawn_point_1 = (x1, y1)
        self.spawn_point_2 = (x2, y2)

    @property
    def area(self) -> int:
        """Total map area in tiles."""
        return self.width * self.height
