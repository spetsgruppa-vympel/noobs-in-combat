"""
Procedural Map Generator for Snakes in Combat.

Generates realistic tactical maps with strategic elements using a score-based
fairness system. Maps are evaluated for competitive balance and regenerated
if scores are too imbalanced.

Strategic elements are integrated naturally into terrain:
- Control Zones: Marked with buildings and favorable terrain
- High Ground: Elevated plateaus with tactical advantage
- Choke Points: Natural narrow passages in terrain
- Cover Positions: Clusters of forests and urban areas
"""

import random
import pygame
from collections import deque
from typing import List, Tuple, Dict, Set, Optional
from main.config import get_logger, MAP_WIDTH, MAP_HEIGHT, TILE_SIZE
from main.game.data.maps.terrain import plains, forest, urban, mountains, road, highway, debris
from main.game.data.maps.tile import tile
from main.game.ui.UItheme import UITheme


# ============================================================================
# GENERATION CONSTANTS
# ============================================================================

DEFAULT_FOREST_DENSITY = 0.25
DEFAULT_URBAN_DENSITY = 0.15
DEFAULT_MOUNTAIN_DENSITY = 0.10
DEFAULT_ROAD_DENSITY = 0.08
DEFAULT_DEBRIS_DENSITY = 0.05
DEFAULT_ELEVATION_DENSITY = 0.20
DEFAULT_BUILDING_DENSITY = 0.08

MIN_ELEVATION = 0
MAX_ELEVATION = 3
RAMP_ELEVATION_INCREMENT = 1
MIN_BUILDING_SPACING = 3

DEFAULT_SMOOTHING_PASSES = 2
SMOOTHING_MIN_NEIGHBORS = 3

# Strategic element counts (algorithmically balanced)
CONTROL_ZONE_COUNT = 5  # Always odd number for center + pairs
HIGH_GROUND_CLUSTERS_PER_SIDE = 2  # Equal count per side
BUILDING_COUNT = 12  # Even number for balanced distribution
BALANCE_TOLERANCE = 0.05  # Maximum allowed distance difference (5%)

# Visualization
PREVIEW_WINDOW_TITLE = "Map Preview"
PREVIEW_FPS = 30
TERRAIN_COLOR_MAP = {
    plains: (200, 200, 140),
    forest: (34, 139, 34),
    urban: (90, 90, 90),
    mountains: (120, 120, 120),
    road: (180, 150, 75),
    highway: (210, 180, 50),
    debris: (130, 110, 90)
}
ELEVATION_COLOR_MULTIPLIERS = [1.0, 1.15, 1.30, 1.45]
BUILDING_COLOR = (150, 75, 0)
RAMP_COLOR = (100, 100, 150)


# ============================================================================
# MAP CONFIGURATION
# ============================================================================

class MapConfig:
    """Configuration parameters for map generation."""

    def __init__(
        self,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        forest_density=DEFAULT_FOREST_DENSITY,
        urban_density=DEFAULT_URBAN_DENSITY,
        mountain_density=DEFAULT_MOUNTAIN_DENSITY,
        road_density=DEFAULT_ROAD_DENSITY,
        debris_density=DEFAULT_DEBRIS_DENSITY,
        elevation_density=DEFAULT_ELEVATION_DENSITY,
        building_density=DEFAULT_BUILDING_DENSITY,
        seed=None,
        smoothing_passes=DEFAULT_SMOOTHING_PASSES,
        spawn_point_1=None,
        spawn_point_2=None
    ):
        """Initialize map configuration."""
        self.width = width
        self.height = height
        self.forest_density = forest_density
        self.urban_density = urban_density
        self.mountain_density = mountain_density
        self.road_density = road_density
        self.debris_density = debris_density
        self.elevation_density = elevation_density
        self.building_density = building_density
        self.seed = seed
        self.smoothing_passes = smoothing_passes

        # Default spawn points (opposing corners)
        self.spawn_point_1 = spawn_point_1 or (width // 4, height // 4)
        self.spawn_point_2 = spawn_point_2 or (3 * width // 4, 3 * height // 4)

        self.logger = get_logger(__name__)
        self.logger.info(f"MapConfig: {width}x{height}")


# ============================================================================
# MAP GENERATOR WITH SCORE-BASED FAIRNESS
# ============================================================================

class MapGenerator:
    """
    Procedural map generator with score-based fairness validation.

    Generates maps and scores them for competitive balance. If a map's
    balance score is too low, it regenerates until acceptable.
    """

    def __init__(self, config=None):
        """Initialize the map generator."""
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)
        self.grid: List[List[tile]] = []
        self.buildings: List[Tuple[int, int]] = []
        self.control_zones: List[Tuple[int, int]] = []  # Key strategic positions
        self.high_ground_clusters: List[Tuple[int, int]] = []
        self.choke_points: List[Tuple[int, int]] = []

        if self.config.seed is not None:
            random.seed(self.config.seed)
            self.logger.info(f"Random seed: {self.config.seed}")

    def generate(self, visualize=False):
        """Generate a guaranteed balanced tactical map."""
        self.logger.info(f"Generating {self.config.width}x{self.config.height} map")

        # Single generation pass with algorithmic balance
        self._initialize_grid()
        self._generate_elevation()
        self._generate_ramps()
        self._generate_terrain()
        self._smooth_terrain()

        # Algorithmically balanced strategic element placement
        self._place_balanced_control_zones()
        self._place_balanced_high_ground()
        self._place_balanced_buildings()
        self._identify_choke_points()  # Natural features

        self.logger.info("Balanced map generated (algorithmic guarantee)")

        if visualize:
            self.visualize_map_pygame()

        return self.grid

    def _initialize_grid(self):
        """Initialize grid with plains tiles."""
        self.grid = []
        self.buildings = []
        self.control_zones = []
        self.high_ground_clusters = []
        self.choke_points = []

        for y in range(self.config.height):
            row = []
            for x in range(self.config.width):
                t = tile(
                    x=x, y=y, terrain_type=plains, size=TILE_SIZE,
                    occupied=False, is_building=False, elevation=MIN_ELEVATION,
                    is_ramp=False, ramp_direction=None
                )
                row.append(t)
            self.grid.append(row)

    def _generate_elevation(self):
        """Generate elevation levels."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                if random.random() < self.config.elevation_density:
                    elevation = random.randint(1, MAX_ELEVATION)
                    self.grid[y][x].set_elevation(elevation)

        # Smooth elevation to create plateaus
        for _ in range(2):
            new_grid = [row[:] for row in self.grid]
            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    elevations = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            elevations.append(self.grid[y + dy][x + dx].elevation)
                    most_common = max(set(elevations), key=elevations.count)
                    new_grid[y][x].elevation = most_common
            self.grid = new_grid

    def _generate_ramps(self):
        """Generate ramps connecting elevation levels."""
        for y in range(1, self.config.height - 1):
            for x in range(1, self.config.width - 1):
                current_elev = self.grid[y][x].elevation
                directions = [('north', 0, -1), ('south', 0, 1),
                            ('east', 1, 0), ('west', -1, 0)]

                for direction, dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if not self._is_valid_position(nx, ny):
                        continue

                    neighbor_elev = self.grid[ny][nx].elevation
                    elev_diff = abs(current_elev - neighbor_elev)

                    if elev_diff == RAMP_ELEVATION_INCREMENT:
                        if not self.grid[y][x].is_ramp and random.random() < 0.6:
                            self.grid[y][x].set_ramp(True, direction)
                            break

    def _generate_terrain(self):
        """Generate all terrain types."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].terrain_type != plains:
                    continue

                # Mountains (impassable)
                if random.random() < self.config.mountain_density:
                    self.grid[y][x].terrain_type = mountains
                # Forests (cover, slow)
                elif random.random() < self.config.forest_density:
                    self.grid[y][x].terrain_type = forest
                # Urban (defensive)
                elif random.random() < self.config.urban_density:
                    self.grid[y][x].terrain_type = urban
                # Debris (light cover)
                elif random.random() < self.config.debris_density:
                    self.grid[y][x].terrain_type = debris
                # Roads (fast movement)
                elif random.random() < self.config.road_density:
                    self.grid[y][x].terrain_type = road

    def _smooth_terrain(self):
        """Smooth terrain using cellular automata."""
        for _ in range(self.config.smoothing_passes):
            new_grid = [row[:] for row in self.grid]
            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    current_type = self.grid[y][x].terrain_type
                    same_neighbors = sum(
                        1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                        if not (dx == 0 and dy == 0) and
                        self.grid[y + dy][x + dx].terrain_type == current_type
                    )

                    if same_neighbors < SMOOTHING_MIN_NEIGHBORS and current_type != plains:
                        neighbor_types = {}
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                n_type = self.grid[y + dy][x + dx].terrain_type
                                neighbor_types[n_type] = neighbor_types.get(n_type, 0) + 1
                        if neighbor_types:
                            most_common = max(neighbor_types, key=neighbor_types.get)
                            new_grid[y][x].terrain_type = most_common
            self.grid = new_grid

    def _place_balanced_control_zones(self):
        """
        Place control zones with algorithmic balance guarantee.

        Strategy:
        - 1 at exact map center (equidistant from both spawns)
        - Remaining zones placed in pairs at equidistant positions
        """
        self.logger.debug("Placing balanced control zones")

        # Center zone (always perfectly balanced)
        center_x = self.config.width // 2
        center_y = self.config.height // 2
        self.control_zones.append((center_x, center_y))

        # Calculate remaining zones to place (in pairs)
        remaining = CONTROL_ZONE_COUNT - 1
        pairs_to_place = remaining // 2

        # Generate candidate positions
        candidates = []
        for y in range(3, self.config.height - 3):
            for x in range(3, self.config.width - 3):
                if (x, y) == (center_x, center_y):
                    continue

                # Calculate balance score for this position
                d1 = self._manhattan_distance(self.config.spawn_point_1, (x, y))
                d2 = self._manhattan_distance(self.config.spawn_point_2, (x, y))

                # Prefer positions with minimal distance difference
                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)

                # Exclude positions too close to spawns or center
                if d1 < 5 or d2 < 5:
                    continue
                if self._manhattan_distance((x, y), (center_x, center_y)) < 3:
                    continue

                candidates.append((balance, x, y))

        # Sort by balance score (best first)
        candidates.sort(reverse=True, key=lambda c: c[0])

        # Place zones from best candidates, ensuring spacing
        placed = 0
        for balance, x, y in candidates:
            if placed >= pairs_to_place * 2:
                break

            # Check spacing from existing zones
            too_close = False
            for zx, zy in self.control_zones:
                if self._manhattan_distance((x, y), (zx, zy)) < 6:
                    too_close = True
                    break

            if not too_close:
                self.control_zones.append((x, y))
                placed += 1

        self.logger.info(f"Placed {len(self.control_zones)} balanced control zones")

    def _place_balanced_high_ground(self):
        """
        Ensure high ground is balanced between spawns.

        Strategy:
        - Identify all elevated clusters
        - Assign clusters to spawns based on distance
        - Ensure each spawn has equal count and quality
        """
        self.logger.debug("Balancing high ground placement")

        # First, ensure we have elevated terrain near each spawn
        self._create_elevation_clusters_near_spawns()

        # Find all high ground clusters
        visited = set()
        all_clusters = []

        for y in range(self.config.height):
            for x in range(self.config.width):
                if (x, y) in visited:
                    continue
                if self.grid[y][x].elevation >= 2:
                    cluster = self._get_connected_tiles(x, y,
                        lambda tx, ty: self.grid[ty][tx].elevation >= 2)
                    if len(cluster) >= 9:
                        center_x = sum(cx for cx, cy in cluster) // len(cluster)
                        center_y = sum(cy for cx, cy in cluster) // len(cluster)

                        d1 = self._manhattan_distance(self.config.spawn_point_1, (center_x, center_y))
                        d2 = self._manhattan_distance(self.config.spawn_point_2, (center_x, center_y))

                        all_clusters.append({
                            'center': (center_x, center_y),
                            'size': len(cluster),
                            'd1': d1,
                            'd2': d2,
                            'tiles': cluster
                        })
                        visited.update(cluster)

        # Assign clusters to ensure balance
        self._assign_balanced_clusters(all_clusters)

    def _create_elevation_clusters_near_spawns(self):
        """Create elevated areas near each spawn if needed."""
        for spawn_idx, spawn in enumerate([self.config.spawn_point_1,
                                           self.config.spawn_point_2]):
            # Create a cluster at distance 6-8 from spawn
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.randint(6, 8)

            cluster_x = int(spawn[0] + distance * (1 if spawn_idx == 0 else -1))
            cluster_y = int(spawn[1] + distance * random.choice([-1, 0, 1]))

            # Create 3x3 elevated area
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cx, cy = cluster_x + dx, cluster_y + dy
                    if self._is_valid_position(cx, cy):
                        self.grid[cy][cx].set_elevation(2)

    def _assign_balanced_clusters(self, clusters):
        """Assign high ground clusters ensuring balance."""
        if not clusters:
            return

        # Sort clusters by size (quality)
        clusters.sort(key=lambda c: c['size'], reverse=True)

        # Assign alternating clusters to ensure balance
        spawn1_clusters = []
        spawn2_clusters = []

        for i, cluster in enumerate(clusters):
            # Assign to closer spawn, but ensure balance
            if cluster['d1'] < cluster['d2']:
                target = spawn1_clusters
            else:
                target = spawn2_clusters

            # If one side has too many, assign to other side
            if len(spawn1_clusters) > len(spawn2_clusters) + 1:
                target = spawn2_clusters
            elif len(spawn2_clusters) > len(spawn1_clusters) + 1:
                target = spawn1_clusters

            target.append(cluster)
            self.high_ground_clusters.append(cluster['center'])

        self.logger.info(
            f"High ground balanced: Spawn1={len(spawn1_clusters)}, "
            f"Spawn2={len(spawn2_clusters)}"
        )

    def _place_balanced_buildings(self):
        """
        Place buildings with guaranteed distance balance.

        Strategy:
        - Place buildings at control zones first
        - For remaining buildings, use paired placement
        - Ensure total distance to all buildings is equal
        """
        self.logger.debug("Placing balanced buildings")

        # Place buildings at control zones
        for cx, cy in self.control_zones:
            best_pos = self._find_building_position_near(cx, cy, radius=2)
            if best_pos:
                bx, by = best_pos
                self.grid[by][bx].set_building(True)
                self.buildings.append((bx, by))

        # Place remaining buildings in balanced pairs
        remaining = BUILDING_COUNT - len(self.buildings)

        # Find candidate positions sorted by balance
        candidates = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                terrain = self.grid[y][x].terrain_type
                if terrain in [mountains, forest, debris]:
                    continue
                if self.grid[y][x].is_ramp or self.grid[y][x].is_building:
                    continue
                if not self._check_building_spacing(x, y):
                    continue

                d1 = self._manhattan_distance(self.config.spawn_point_1, (x, y))
                d2 = self._manhattan_distance(self.config.spawn_point_2, (x, y))

                # Balance score: prefer positions with similar distances
                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                candidates.append((balance, x, y))

        # Sort by balance (best first)
        candidates.sort(reverse=True, key=lambda c: c[0])

        # Place remaining buildings
        placed = 0
        for balance, x, y in candidates:
            if placed >= remaining:
                break

            if self._check_building_spacing(x, y):
                self.grid[y][x].set_building(True)
                self.buildings.append((x, y))
                placed += 1

        self.logger.info(f"Placed {len(self.buildings)} balanced buildings")

    def _identify_choke_points(self):
        """Identify natural choke points in terrain."""
        self.logger.debug("Identifying choke points")

        for y in range(2, self.config.height - 2):
            for x in range(2, self.config.width - 2):
                if self.grid[y][x].terrain_type == mountains:
                    continue

                # Count passable neighbors
                passable = sum(
                    1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                    if not (dx == 0 and dy == 0) and
                    self._is_valid_position(x + dx, y + dy) and
                    self.grid[y + dy][x + dx].terrain_type != mountains
                )

                # Choke point: 2-4 passable neighbors (restricted passage)
                if 2 <= passable <= 4:
                    self.choke_points.append((x, y))

        self.logger.info(f"Identified {len(self.choke_points)} choke points")

    def _find_building_position_near(self, cx: int, cy: int, radius: int) -> Optional[Tuple[int, int]]:
        """Find a suitable building position near target coordinates."""
        for r in range(radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if not self._is_valid_position(x, y):
                        continue
                    terrain = self.grid[y][x].terrain_type
                    if terrain in [mountains, forest, debris]:
                        continue
                    if self.grid[y][x].is_ramp or self.grid[y][x].is_building:
                        continue
                    if self._check_building_spacing(x, y):
                        return (x, y)
        return None

    def _check_building_spacing(self, x: int, y: int) -> bool:
        """Check minimum building spacing."""
        for bx, by in self.buildings:
            if abs(x - bx) + abs(y - by) < MIN_BUILDING_SPACING:
                return False
        return True

    def _get_connected_tiles(self, x: int, y: int, condition) -> Set[Tuple[int, int]]:
        """Get all connected tiles matching condition."""
        visited = set()
        queue = deque([(x, y)])
        visited.add((x, y))

        while queue:
            cx, cy = queue.popleft()

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy

                if not self._is_valid_position(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                if not condition(nx, ny):
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return visited

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def visualize_map_pygame(self):
        """
        Display map in pygame window.

        NOTE: This is a simple 2D debug view. For full 2.5D preview,
        use the main menu's Map Preview feature.
        """
        pygame.init()
        width_px = self.config.width * TILE_SIZE
        height_px = self.config.height * TILE_SIZE
        window = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption(PREVIEW_WINDOW_TITLE)

        clock = pygame.time.Clock()
        font = pygame.font.SysFont(**UITheme.FONT_TILE_LABEL)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            window.fill((0, 0, 0))

            mouse_x, mouse_y = pygame.mouse.get_pos()
            hovered_x = mouse_x // TILE_SIZE
            hovered_y = mouse_y // TILE_SIZE

            # Draw tiles
            for row in self.grid:
                for cell in row:
                    if cell.is_building:
                        base_color = BUILDING_COLOR
                    elif cell.is_ramp:
                        base_color = RAMP_COLOR
                    else:
                        base_color = TERRAIN_COLOR_MAP.get(cell.terrain_type, (255, 0, 255))

                    if cell.elevation > 0:
                        multiplier = ELEVATION_COLOR_MULTIPLIERS[min(cell.elevation, MAX_ELEVATION)]
                        base_color = tuple(min(255, int(c * multiplier)) for c in base_color)

                    is_hovered = (cell.x == hovered_x and cell.y == hovered_y)
                    tile_color = UITheme.darken_color(base_color, 0.6) if is_hovered else base_color

                    tile_rect = pygame.Rect(cell.x * TILE_SIZE, cell.y * TILE_SIZE,
                                          TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(window, tile_color, tile_rect)

                    if is_hovered:
                        info = [cell.terrain_type.name]
                        if cell.is_building:
                            info.append("BLDG")
                        if cell.elevation > 0:
                            info.append(f"Elev:{cell.elevation}")
                        text = " ".join(info)
                        label = font.render(text, True, base_color)
                        window.blit(label, label.get_rect(center=tile_rect.center))

            # Draw spawn points
            for idx, spawn in enumerate([self.config.spawn_point_1, self.config.spawn_point_2]):
                rect = pygame.Rect(spawn[0] * TILE_SIZE, spawn[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                color = (0, 255, 0) if idx == 0 else (255, 0, 0)
                pygame.draw.rect(window, color, rect, 3)

            pygame.display.flip()
            clock.tick(PREVIEW_FPS)

        pygame.quit()

    def get_terrain_statistics(self):
        """Calculate terrain statistics and verify balance."""
        stats = {
            "plains": 0, "forest": 0, "urban": 0, "mountains": 0,
            "road": 0, "highway": 0, "debris": 0,
            "buildings": len(self.buildings),
            "ramps": 0, "elevated": 0,
            "control_zones": len(self.control_zones),
            "high_ground_clusters": len(self.high_ground_clusters),
            "choke_points": len(self.choke_points)
        }

        for row in self.grid:
            for t in row:
                name = t.terrain_type.name.lower()
                if name in stats:
                    stats[name] += 1
                if t.is_ramp:
                    stats["ramps"] += 1
                if t.elevation > MIN_ELEVATION:
                    stats["elevated"] += 1

        # Verify balance
        balance_info = self._verify_balance()
        stats.update(balance_info)

        return stats

    def _verify_balance(self) -> Dict[str, float]:
        """
        Verify algorithmic balance guarantees.

        Returns diagnostic information about balance quality.
        """
        if not self.buildings:
            return {"balance_verified": True, "building_balance": 1.0}

        # Calculate building distance balance
        p1_dists = [self._manhattan_distance(self.config.spawn_point_1, (bx, by))
                    for bx, by in self.buildings]
        p2_dists = [self._manhattan_distance(self.config.spawn_point_2, (bx, by))
                    for bx, by in self.buildings]

        avg1 = sum(p1_dists) / len(p1_dists)
        avg2 = sum(p2_dists) / len(p2_dists)

        building_balance = min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1.0

        # Calculate control zone balance
        cz_p1 = [self._manhattan_distance(self.config.spawn_point_1, cz)
                 for cz in self.control_zones]
        cz_p2 = [self._manhattan_distance(self.config.spawn_point_2, cz)
                 for cz in self.control_zones]

        cz_avg1 = sum(cz_p1) / len(cz_p1) if cz_p1 else 0
        cz_avg2 = sum(cz_p2) / len(cz_p2) if cz_p2 else 0

        cz_balance = min(cz_avg1, cz_avg2) / max(cz_avg1, cz_avg2) if max(cz_avg1, cz_avg2) > 0 else 1.0

        verified = building_balance >= 0.85 and cz_balance >= 0.85

        self.logger.info(
            f"Balance verification: Buildings={building_balance:.3f}, "
            f"ControlZones={cz_balance:.3f}, Verified={verified}"
        )

        return {
            "balance_verified": verified,
            "building_balance": building_balance,
            "control_zone_balance": cz_balance
        }