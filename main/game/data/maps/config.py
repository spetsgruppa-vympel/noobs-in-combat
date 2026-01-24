"""
Configuration dataclass for procedural map generation.

All generator knobs are exposed here. Defaults aim to produce realistic
small-to-medium sized maps with cities, highways, roads and biomes.
Values are clamped/validated in __post_init__.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from main.config import MAP_WIDTH, MAP_HEIGHT


@dataclass
class MapConfig:
    # ---------------------------
    # Basic map dimensions & seed
    # ---------------------------
    width: int = MAP_WIDTH
    height: int = MAP_HEIGHT
    seed: Optional[int] = None

    # ---------------------------
    # Global terrain density targets (fractions)
    # ---------------------------
    forest_density: float = 0.25
    urban_density: float = 0.15
    mountain_density: float = 0.10
    road_density: float = 0.06
    debris_density: float = 0.05

    # ---------------------------
    # Elevation / ramp parameters
    # ---------------------------
    elevation_density: float = 0.18
    max_elevation: int = 3
    min_support_neighbors: int = 2
    min_ramp_neighbors: int = 1
    ramp_placement_probability: float = 0.75

    # ---------------------------
    # Strategic elements (control/building)
    # ---------------------------
    building_density: float = 0.08        # fraction of tiles to attempt as buildings
    control_zone_count: int = 5           # will be rounded to odd number in __post_init__
    min_building_spacing: int = 3         # Manhattan distance between buildings

    # ---------------------------
    # Smoothing and clustering
    # ---------------------------
    smoothing_passes: int = 2
    cluster_min_size: int = 3             # minimum cluster size to keep (non-plains)

    # ---------------------------
    # Biome generation & blending
    # ---------------------------
    biome_count_min: int = 2
    biome_count_max: int = 4
    biome_noise_scale: float = 0.07       # controls noise influence on biome boundaries
    biome_noise_octaves: int = 2

    # ---------------------------
    # Cities & roads (scale with map size)
    # ---------------------------
    city_base_count: float = 0.0015       # city_count = max(1, int(area * city_base_count))
    city_growth_factor: float = 0.6       # how strongly city influence converts tiles to urban
    highway_factor: float = 0.0012        # (reserved for future use — exposed for tuning)
    highway_width: int = 2
    road_urban_bias: float = 2.5          # how much urban/city influence favors roads
    city_max_fraction_of_area: float = 0.06

    # ---------------------------
    # Diagonal / parallel road discouragement
    # ---------------------------
    discourage_diagonal_roads: bool = True
    diagonal_road_penalty: float = 0.85   # probability to remove an isolated diagonal road tile (0..1)
    keep_city_diagonal_roads: bool = False
    diagonal_detection_radius: int = 1    # radius to check neighbouring roads for "parallel" patterns

    # ---------------------------
    # Pathfinding & performance
    # ---------------------------
    smoothing_enabled: bool = True
    seed_retry_attempts: int = 100

    # ---------------------------
    # Spawn points (defaults set in __post_init__)
    # ---------------------------
    spawn_point_1: Optional[Tuple[int, int]] = None
    spawn_point_2: Optional[Tuple[int, int]] = None

    # ---------------------------
    # Biome definitions override (optional)
    # ---------------------------
    biome_definitions: Optional[List[Dict]] = field(default=None)

    # ---------------------------
    # Internal read-only convenience properties:
    # ---------------------------
    def __post_init__(self):
        """Validate and clamp config values for safety and predictability."""
        # Clamp dimensions to reasonable bounds
        self.width = max(8, min(256, self.width))
        self.height = max(8, min(256, self.height))

        # Clamp densities and fractions
        self.forest_density = max(0.0, min(0.6, self.forest_density))
        self.urban_density = max(0.0, min(0.6, self.urban_density))
        self.mountain_density = max(0.0, min(0.35, self.mountain_density))
        self.road_density = max(0.0, min(0.2, self.road_density))
        self.debris_density = max(0.0, min(0.15, self.debris_density))
        self.elevation_density = max(0.0, min(0.6, self.elevation_density))
        self.building_density = max(0.0, min(0.3, self.building_density))

        # Elevation params
        self.max_elevation = max(1, min(5, self.max_elevation))
        self.min_support_neighbors = max(1, min(4, self.min_support_neighbors))
        self.min_ramp_neighbors = max(0, min(3, self.min_ramp_neighbors))
        self.ramp_placement_probability = max(0.0, min(1.0, self.ramp_placement_probability))

        # Control zone count rounding (prefer odd)
        if self.control_zone_count % 2 == 0:
            self.control_zone_count += 1
        self.control_zone_count = max(3, min(11, self.control_zone_count))

        # Spacing and smoothing clamps
        self.min_building_spacing = max(2, min(6, self.min_building_spacing))
        self.smoothing_passes = max(0, min(6, self.smoothing_passes))
        self.cluster_min_size = max(1, min(12, self.cluster_min_size))

        # Biome counts
        self.biome_count_min = max(1, min(6, self.biome_count_min))
        self.biome_count_max = max(self.biome_count_min, min(8, self.biome_count_max))
        self.biome_noise_scale = max(0.0, min(1.0, self.biome_noise_scale))
        self.biome_noise_octaves = max(1, min(6, self.biome_noise_octaves))

        # City & road clamps
        self.city_base_count = max(0.0001, min(0.01, self.city_base_count))
        self.city_growth_factor = max(0.0, min(1.5, self.city_growth_factor))
        self.highway_width = max(1, min(6, self.highway_width))
        self.road_urban_bias = max(0.0, min(10.0, self.road_urban_bias))
        self.city_max_fraction_of_area = max(0.01, min(0.25, self.city_max_fraction_of_area))

        # Diagonal road settings
        self.discourage_diagonal_roads = bool(self.discourage_diagonal_roads)
        self.diagonal_road_penalty = max(0.0, min(1.0, self.diagonal_road_penalty))
        self.keep_city_diagonal_roads = bool(self.keep_city_diagonal_roads)
        self.diagonal_detection_radius = max(1, min(3, self.diagonal_detection_radius))

        # Default spawn points if none provided — positioned near/equidistant corners (tunable)
        if self.spawn_point_1 is None:
            self.spawn_point_1 = (self.width // 6, self.height // 6)
        if self.spawn_point_2 is None:
            self.spawn_point_2 = (5 * self.width // 6, 5 * self.height // 6)

        # Ensure spawn points are valid coordinates
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
        """Total map area (width * height)."""
        return self.width * self.height
