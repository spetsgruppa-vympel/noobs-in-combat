"""
Map generation configuration with validation.

Provides a validated dataclass for all map generation parameters.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from main.config import MAP_WIDTH, MAP_HEIGHT


@dataclass
class MapConfig:
    """
    Configuration for map generation with automatic validation.

    All density values are fractions (0.0 to 1.0) representing target coverage.
    Invalid values are automatically clamped to valid ranges.
    """

    # Map dimensions
    width: int = MAP_WIDTH
    height: int = MAP_HEIGHT
    seed: Optional[int] = None

    # Terrain densities (0.0 to 1.0)
    forest_density: float = 0.25
    urban_density: float = 0.15
    mountain_density: float = 0.10
    road_density: float = 0.06
    debris_density: float = 0.05

    # Elevation parameters
    elevation_density: float = 0.18
    max_elevation: int = 3
    min_support_neighbors: int = 2
    min_ramp_neighbors: int = 1
    ramp_placement_probability: float = 0.75

    # Strategic elements
    building_density: float = 0.08
    control_zone_count: int = 5
    min_building_spacing: int = 3

    # Generation parameters
    smoothing_passes: int = 2
    spawn_point_1: Optional[Tuple[int, int]] = None
    spawn_point_2: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Validate and auto-correct configuration values."""
        # Clamp dimensions
        self.width = max(8, min(256, self.width))
        self.height = max(8, min(256, self.height))

        # Clamp densities
        self.forest_density = max(0.0, min(0.5, self.forest_density))
        self.urban_density = max(0.0, min(0.3, self.urban_density))
        self.mountain_density = max(0.0, min(0.25, self.mountain_density))
        self.road_density = max(0.0, min(0.15, self.road_density))
        self.debris_density = max(0.0, min(0.1, self.debris_density))
        self.elevation_density = max(0.0, min(0.4, self.elevation_density))
        self.building_density = max(0.0, min(0.2, self.building_density))

        # Clamp elevation parameters
        self.max_elevation = max(1, min(5, self.max_elevation))
        self.min_support_neighbors = max(1, min(4, self.min_support_neighbors))
        self.min_ramp_neighbors = max(0, min(3, self.min_ramp_neighbors))
        self.ramp_placement_probability = max(0.0, min(1.0, self.ramp_placement_probability))

        # Auto-correct control zone count to nearest odd number
        if self.control_zone_count % 2 == 0:
            self.control_zone_count += 1
        self.control_zone_count = max(3, min(11, self.control_zone_count))

        # Clamp spacing and smoothing
        self.min_building_spacing = max(2, min(6, self.min_building_spacing))
        self.smoothing_passes = max(0, min(5, self.smoothing_passes))

        # Set default spawn points if not provided
        if self.spawn_point_1 is None:
            self.spawn_point_1 = (self.width // 4, self.height // 4)
        if self.spawn_point_2 is None:
            self.spawn_point_2 = (3 * self.width // 4, 3 * self.height // 4)

        # Clamp spawn points to valid coordinates
        self.spawn_point_1 = (
            max(0, min(self.width - 1, self.spawn_point_1[0])),
            max(0, min(self.height - 1, self.spawn_point_1[1]))
        )
        self.spawn_point_2 = (
            max(0, min(self.width - 1, self.spawn_point_2[0])),
            max(0, min(self.height - 1, self.spawn_point_2[1]))
        )

    @property
    def area(self) -> int:
        """Total map area in tiles."""
        return self.width * self.height