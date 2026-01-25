"""
Enhanced MapConfig with Skeleton-Graph Road Generation.

HIERARCHICAL GENERATION:
1. Skeleton graph (highways/major roads) - sparse, strategic
2. Rasterization (graph → tiles) - clean conversion
3. Local road fill (constrained) - prevents parallel spam
4. Urban/building fill (block-based) - realistic city structure
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class MapConfig:
    # ---------------------------
    # Basic map dimensions & seed
    # ---------------------------
    width: int = 20
    height: int = 15
    seed: Optional[int] = None

    # ---------------------------
    # Global terrain density targets (REDUCED)
    # ---------------------------
    forest_density: float = 0.20
    urban_density: float = 0.08
    mountain_density: float = 0.06
    road_density: float = 0.05
    debris_density: float = 0.04

    # ---------------------------
    # Elevation / ramp parameters
    # ---------------------------
    elevation_density: float = 0.15
    max_elevation: int = 3
    min_support_neighbors: int = 2
    min_ramp_neighbors: int = 1
    ramp_placement_probability: float = 0.75

    # Road elevation handling
    road_elevation_bypass_threshold: int = 2
    road_force_ramp: bool = True
    road_max_elevation_cross: int = 1

    # ---------------------------
    # Strategic elements
    # ---------------------------
    building_density: float = 0.04
    control_zone_count: int = 5
    min_building_spacing: int = 3

    # Building distribution
    building_heat_influence: float = 0.6
    building_variety_bonus: float = 0.4

    # ---------------------------
    # Smoothing and clustering
    # ---------------------------
    smoothing_passes: int = 2
    cluster_min_size: int = 3

    # ---------------------------
    # Biome generation (DYNAMIC SIZING)
    # ---------------------------
    biome_count_min: int = 2
    biome_count_max: int = 4
    biome_noise_scale: float = 0.07
    biome_noise_octaves: int = 2

    # Heat-based biome sizing
    biome_min_radius: int = 4
    biome_heat_scaling: bool = True
    biome_heat_scale_factor: float = 0.5

    # ---------------------------
    # SKELETON GRAPH GENERATION (NEW)
    # ---------------------------

    # Node generation
    use_skeleton_graph: bool = True                    # Enable skeleton-graph system
    node_generation_method: str = "poisson_disc"       # "poisson_disc", "grid_noise", "city_districts"
    node_min_spacing: int = -1                         # Auto-calculate if -1 (based on map size)
    node_count_target: int = -1                        # Auto-calculate if -1

    # Skeleton graph construction
    skeleton_method: str = "delaunay_mst"              # "delaunay_mst", "astar_mesh"
    skeleton_extra_edges: float = 0.15                 # % of MST edges to add back for loops
    skeleton_straightness_bias: float = 0.90           # Prefer straight skeleton edges

    # Highway rasterization
    highway_width: int = 2
    highway_rasterization_method: str = "antialiased" # "bresenham", "antialiased", "spline"
    highway_curve_smoothing: bool = True               # Smooth sharp corners

    # Local road generation (constrained)
    generate_local_roads: bool = True
    local_road_density: float = 0.12                   # Target % of tiles as local roads
    local_road_min_spacing: int = 3                    # Min tiles between parallel local roads
    local_road_to_highway_spacing: int = 2             # Min tiles from local to highway
    local_road_intersection_portals: bool = True       # Locals can only join highways at portals
    local_road_portal_spacing: int = 8                 # Distance between allowed portals

    # Local road fill strategy
    local_fill_method: str = "block_subdivision"       # "cellular_walker", "astar_pois", "block_subdivision"
    local_road_hierarchy: bool = True                  # Create collectors vs residential roads

    # ---------------------------
    # Urban center generation (BLOCK-BASED)
    # ---------------------------
    urban_generation_method: str = "block_fill"        # "block_fill", "road_growth"
    urban_block_fill_density: float = 0.6              # % of blocks to fill with urban
    urban_block_min_size: int = 4                      # Minimum block size to urbanize
    urban_building_in_block_chance: float = 0.3        # Chance of building in urban block

    # ---------------------------
    # Heat map configuration
    # ---------------------------
    use_heat_map: bool = True
    heat_decay_rate: float = 0.15
    heat_influence_radius: int = -1

    # ---------------------------
    # Legacy road parameters (DEPRECATED - kept for compatibility)
    # ---------------------------
    city_base_count: float = 0.001
    city_growth_factor: float = 0.4
    city_max_fraction_of_area: float = 0.04
    urban_center_road_bias: float = 0.8
    urban_cluster_tightness: float = 0.7
    urban_building_density: float = 0.3
    highway_factor: float = 0.0012
    highway_straightness_bias: float = 0.85
    highway_max_turns: int = 3
    road_urban_bias: float = 2.5
    road_adjacency_prevention: bool = True
    road_min_spacing: int = 2

    # ---------------------------
    # Diagonal road discouragement
    # ---------------------------
    discourage_diagonal_roads: bool = True
    diagonal_road_penalty: float = 0.85
    keep_city_diagonal_roads: bool = False
    diagonal_detection_radius: int = 1

    # ---------------------------
    # Pathfinding & performance
    # ---------------------------
    smoothing_enabled: bool = True
    seed_retry_attempts: int = 100

    # ---------------------------
    # Spawn points
    # ---------------------------
    spawn_point_1: Optional[Tuple[int, int]] = None
    spawn_point_2: Optional[Tuple[int, int]] = None

    # ---------------------------
    # Biome definitions override
    # ---------------------------
    biome_definitions: Optional[List[Dict]] = field(default=None)

    def __post_init__(self):
        """Validate and clamp config values."""
        # Clamp dimensions
        self.width = max(8, min(256, self.width))
        self.height = max(8, min(256, self.height))

        # Clamp densities
        self.forest_density = max(0.0, min(0.6, self.forest_density))
        self.urban_density = max(0.0, min(0.4, self.urban_density))
        self.mountain_density = max(0.0, min(0.25, self.mountain_density))
        self.road_density = max(0.0, min(0.2, self.road_density))
        self.debris_density = max(0.0, min(0.15, self.debris_density))
        self.elevation_density = max(0.0, min(0.6, self.elevation_density))
        self.building_density = max(0.0, min(0.2, self.building_density))

        # Elevation params
        self.max_elevation = max(1, min(5, self.max_elevation))
        self.min_support_neighbors = max(1, min(4, self.min_support_neighbors))
        self.min_ramp_neighbors = max(0, min(3, self.min_ramp_neighbors))
        self.ramp_placement_probability = max(0.0, min(1.0, self.ramp_placement_probability))

        # Road elevation
        self.road_elevation_bypass_threshold = max(1, min(5, self.road_elevation_bypass_threshold))
        self.road_max_elevation_cross = max(0, min(3, self.road_max_elevation_cross))

        # Control zones (prefer odd)
        if self.control_zone_count % 2 == 0:
            self.control_zone_count += 1
        self.control_zone_count = max(3, min(11, self.control_zone_count))

        # Spacing and smoothing
        self.min_building_spacing = max(2, min(6, self.min_building_spacing))
        self.smoothing_passes = max(0, min(6, self.smoothing_passes))
        self.cluster_min_size = max(1, min(12, self.cluster_min_size))

        # Building distribution
        self.building_heat_influence = max(0.0, min(1.0, self.building_heat_influence))
        self.building_variety_bonus = max(0.0, min(1.0, self.building_variety_bonus))

        # Biomes
        self.biome_count_min = max(1, min(6, self.biome_count_min))
        self.biome_count_max = max(self.biome_count_min, min(8, self.biome_count_max))
        self.biome_noise_scale = max(0.0, min(1.0, self.biome_noise_scale))
        self.biome_noise_octaves = max(1, min(6, self.biome_noise_octaves))
        self.biome_min_radius = max(3, min(20, self.biome_min_radius))
        self.biome_heat_scale_factor = max(0.0, min(1.0, self.biome_heat_scale_factor))

        # Skeleton graph
        if self.node_min_spacing == -1:
            # Auto-calculate: ~5-8% of average dimension
            avg_dim = (self.width + self.height) / 2
            self.node_min_spacing = max(6, int(avg_dim * 0.07))
        else:
            self.node_min_spacing = max(4, min(30, self.node_min_spacing))

        if self.node_count_target == -1:
            # Auto-calculate based on map area and spacing
            area = self.width * self.height
            self.node_count_target = max(4, int(area / (self.node_min_spacing ** 2)))
        else:
            self.node_count_target = max(3, min(50, self.node_count_target))

        self.skeleton_extra_edges = max(0.0, min(0.5, self.skeleton_extra_edges))
        self.skeleton_straightness_bias = max(0.0, min(1.0, self.skeleton_straightness_bias))

        self.highway_width = max(1, min(6, self.highway_width))

        # Local roads
        self.local_road_density = max(0.0, min(0.1, self.local_road_density))
        self.local_road_min_spacing = max(2, min(8, self.local_road_min_spacing))
        self.local_road_to_highway_spacing = max(1, min(5, self.local_road_to_highway_spacing))
        self.local_road_portal_spacing = max(4, min(20, self.local_road_portal_spacing))

        # Urban blocks
        self.urban_block_fill_density = max(0.0, min(1.0, self.urban_block_fill_density))
        self.urban_block_min_size = max(3, min(15, self.urban_block_min_size))
        self.urban_building_in_block_chance = max(0.0, min(1.0, self.urban_building_in_block_chance))

        # Legacy params
        self.city_base_count = max(0.0001, min(0.01, self.city_base_count))
        self.city_growth_factor = max(0.0, min(1.5, self.city_growth_factor))
        self.city_max_fraction_of_area = max(0.01, min(0.25, self.city_max_fraction_of_area))
        self.urban_center_road_bias = max(0.0, min(1.0, self.urban_center_road_bias))
        self.urban_cluster_tightness = max(0.0, min(1.0, self.urban_cluster_tightness))
        self.urban_building_density = max(0.0, min(1.0, self.urban_building_density))
        self.highway_straightness_bias = max(0.0, min(1.0, self.highway_straightness_bias))
        self.highway_max_turns = max(1, min(10, self.highway_max_turns))
        self.road_urban_bias = max(0.0, min(10.0, self.road_urban_bias))
        self.road_min_spacing = max(1, min(5, self.road_min_spacing))
        self.diagonal_road_penalty = max(0.0, min(1.0, self.diagonal_road_penalty))
        self.diagonal_detection_radius = max(1, min(3, self.diagonal_detection_radius))

        # Heat map
        self.heat_decay_rate = max(0.05, min(0.5, self.heat_decay_rate))
        if self.heat_influence_radius == -1:
            self.heat_influence_radius = max(self.width, self.height) // 2

        # Validate skeleton methods
        valid_node_methods = ["poisson_disc", "grid_noise", "city_districts"]
        if self.node_generation_method not in valid_node_methods:
            self.node_generation_method = "poisson_disc"

        valid_skeleton_methods = ["delaunay_mst", "astar_mesh"]
        if self.skeleton_method not in valid_skeleton_methods:
            self.skeleton_method = "delaunay_mst"

        valid_raster_methods = ["bresenham", "antialiased", "spline"]
        if self.highway_rasterization_method not in valid_raster_methods:
            self.highway_rasterization_method = "antialiased"

        valid_fill_methods = ["cellular_walker", "astar_pois", "block_subdivision"]
        if self.local_fill_method not in valid_fill_methods:
            self.local_fill_method = "block_subdivision"

        valid_urban_methods = ["block_fill", "road_growth"]
        if self.urban_generation_method not in valid_urban_methods:
            self.urban_generation_method = "block_fill"

        # Default spawn points
        if self.spawn_point_1 is None:
            self.spawn_point_1 = (self.width // 6, self.height // 6)
        if self.spawn_point_2 is None:
            self.spawn_point_2 = (5 * self.width // 6, 5 * self.height // 6)

        # Validate spawn points
        self.spawn_point_1 = (
            max(0, min(self.width - 1, self.spawn_point_1[0])),
            max(0, min(self.height - 1, self.spawn_point_1[1])),
        )
        self.spawn_point_2 = (
            max(0, min(self.width - 1, self.spawn_point_2[0])),
            max(0, min(self.height - 1, self.spawn_point_2[1])),
        )

    @property
    def area(self) -> int:
        """Total map area."""
        return self.width * self.height