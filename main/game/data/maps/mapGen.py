"""
Procedural Map Generator for Snakes in Combat.

Generates realistic but playable tactical maps with varied terrain, strategic positioning,
building placements, elevation systems with ramps, and fairness validation. Uses cellular
automata and noise-based algorithms for natural terrain distribution while maintaining
gameplay balance.

Current Implementation: PHASE 1-4 (Terrain, Elevation, Buildings, Fairness)
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

# Default terrain density values (probability 0.0-1.0)
DEFAULT_FOREST_DENSITY = 0.25
DEFAULT_URBAN_DENSITY = 0.15
DEFAULT_MOUNTAIN_DENSITY = 0.10
DEFAULT_ROAD_DENSITY = 0.08
DEFAULT_DEBRIS_DENSITY = 0.05

# Elevation system
DEFAULT_ELEVATION_DENSITY = 0.20  # Probability of elevated terrain
MIN_ELEVATION = 0                 # Ground level
MAX_ELEVATION = 3                 # Maximum elevation level
RAMP_ELEVATION_INCREMENT = 1      # Elevation change per ramp tile

# Building generation
DEFAULT_BUILDING_DENSITY = 0.08   # Probability of building placement
MIN_BUILDING_SPACING = 3          # Minimum tiles between buildings

# Smoothing algorithm parameters
DEFAULT_SMOOTHING_PASSES = 2
SMOOTHING_MIN_NEIGHBORS = 3       # Minimum same-type neighbors to avoid conversion

# Cluster analysis
DEFAULT_MIN_CLUSTER_SIZE = 3

# Fairness validation
MAX_REROLL_ATTEMPTS = 5           # Maximum map regeneration attempts
PATH_LENGTH_TOLERANCE = 0.3       # Allowed path length difference (30%)
RESOURCE_BALANCE_TOLERANCE = 0.2  # Allowed resource imbalance (20%)

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================

# Preview window settings
PREVIEW_WINDOW_TITLE = "Snakes in Combat - Tactical Map Preview"
PREVIEW_FPS = 30
PREVIEW_BACKGROUND_COLOR = (0, 0, 0)

# Terrain color mapping (base colors)
TERRAIN_COLOR_MAP = {
    plains: (200, 200, 140),
    forest: (34, 139, 34),
    urban: (90, 90, 90),
    mountains: (120, 120, 120),
    road: (180, 150, 75),
    highway: (210, 180, 50),
    debris: (130, 110, 90)
}

# Elevation color modifiers (multiply base color)
ELEVATION_COLOR_MULTIPLIERS = [
    1.0,   # Level 0: normal color
    1.15,  # Level 1: slightly brighter
    1.30,  # Level 2: brighter
    1.45   # Level 3: brightest
]

# Building and ramp colors
BUILDING_COLOR = (150, 75, 0)      # Brown
RAMP_COLOR = (100, 100, 150)       # Blue-gray

# ASCII symbols for console map preview
ASCII_TERRAIN_SYMBOLS = {
    plains: ".",
    forest: "T",
    urban: "#",
    mountains: "^",
    road: "=",
    highway: "≡",
    debris: "x"
}


class MapConfig:
    """Configuration parameters for procedural map generation."""

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
            min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
            validate_fairness=True,
            spawn_point_1=None,
            spawn_point_2=None
    ):
        """Initialize map configuration with specified or default parameters."""
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
        self.min_cluster_size = min_cluster_size
        self.validate_fairness = validate_fairness

        # Auto-generate spawn points if not provided
        self.spawn_point_1 = spawn_point_1 or (width // 4, height // 2)
        self.spawn_point_2 = spawn_point_2 or (3 * width // 4, height // 2)

        self.logger = get_logger(__name__)
        self.logger.info(
            f"MapConfig created: {width}x{height}, "
            f"forest={forest_density}, urban={urban_density}, "
            f"elevation={elevation_density}, buildings={building_density}"
        )


class MapGenerator:
    """Procedural map generator with elevation, buildings, and fairness validation."""

    def __init__(self, config=None):
        """Initialize the map generator."""
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)
        self.grid: List[List[tile]] = []
        self.buildings: List[Tuple[int, int]] = []  # Building positions

        if self.config.seed is not None:
            random.seed(self.config.seed)
            self.logger.info(f"Random seed set to {self.config.seed}")

        self.logger.info("MapGenerator initialized")

    def generate(self, visualize=False):
        """Generate a complete tactical map with validation."""
        self.logger.info(f"Starting map generation: {self.config.width}x{self.config.height}")

        if self.config.validate_fairness:
            # Try to generate a fair map, rerolling if necessary
            for attempt in range(MAX_REROLL_ATTEMPTS):
                self.logger.info(f"Generation attempt {attempt + 1}/{MAX_REROLL_ATTEMPTS}")
                self._generate_map_internal()

                if self._validate_fairness():
                    self.logger.info(f"Fair map generated on attempt {attempt + 1}")
                    break
                else:
                    self.logger.warning(f"Map failed fairness validation, rerolling...")
            else:
                self.logger.warning(f"Could not generate fair map after {MAX_REROLL_ATTEMPTS} attempts")
        else:
            self._generate_map_internal()

        self.logger.info("Map generation complete")

        if visualize:
            self.visualize_map_pygame()

        return self.grid

    def _generate_map_internal(self):
        """Internal map generation process."""
        self._initialize_grid()
        self._generate_elevation()
        self._generate_ramps()
        self._generate_mountains()
        self._generate_forests()
        self._generate_urban_areas()
        self._generate_debris()
        self._generate_roads()
        self._smooth_terrain()
        self._generate_buildings()

    def _initialize_grid(self):
        """Initialize the map grid with default plains tiles."""
        self.logger.debug("Initializing grid with plains")
        self.grid = []
        self.buildings = []

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

        self.logger.debug(f"Grid initialized: {len(self.grid)}x{len(self.grid[0])}")

    def _generate_elevation(self):
        """Generate elevation levels across the map."""
        self.logger.debug("Generating elevation")
        elevated_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if random.random() < self.config.elevation_density:
                    # Random elevation between 1 and MAX_ELEVATION
                    elevation = random.randint(1, MAX_ELEVATION)
                    self.grid[y][x].set_elevation(elevation)
                    elevated_count += 1

        # Smooth elevation to create plateaus
        self._smooth_elevation()

        self.logger.info(f"Generated elevation for {elevated_count} tiles")

    def _smooth_elevation(self):
        """Smooth elevation to create natural plateaus."""
        for _ in range(2):  # Two smoothing passes
            new_grid = [row[:] for row in self.grid]

            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    # Get neighbor elevations
                    elevations = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            elevations.append(self.grid[y + dy][x + dx].elevation)

                    # Set to most common elevation
                    most_common = max(set(elevations), key=elevations.count)
                    new_grid[y][x].elevation = most_common

            self.grid = new_grid

    def _generate_ramps(self):
        """Generate ramps to connect different elevation levels."""
        self.logger.debug("Generating ramps")
        ramp_count = 0

        for y in range(1, self.config.height - 1):
            for x in range(1, self.config.width - 1):
                current_elev = self.grid[y][x].elevation

                # Check all four directions for elevation differences
                directions = [
                    ('north', 0, -1),
                    ('south', 0, 1),
                    ('east', 1, 0),
                    ('west', -1, 0)
                ]

                for direction, dx, dy in directions:
                    neighbor_x = x + dx
                    neighbor_y = y + dy

                    if not self._is_valid_position(neighbor_x, neighbor_y):
                        continue

                    neighbor_elev = self.grid[neighbor_y][neighbor_x].elevation
                    elev_diff = abs(current_elev - neighbor_elev)

                    # Create ramps for elevation differences
                    if elev_diff == RAMP_ELEVATION_INCREMENT:
                        # Determine if we should place a ramp here
                        if not self.grid[y][x].is_ramp and random.random() < 0.6:
                            self.grid[y][x].set_ramp(True, direction)
                            ramp_count += 1
                            break  # Only one ramp per tile

        self.logger.info(f"Generated {ramp_count} ramp tiles")

    def _generate_mountains(self):
        """Place mountain tiles (only on ground level)."""
        self.logger.debug("Generating mountains")
        mountain_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                # Mountains only on ground level
                if self.grid[y][x].elevation == MIN_ELEVATION:
                    if random.random() < self.config.mountain_density:
                        self.grid[y][x].terrain_type = mountains
                        mountain_count += 1

        self.logger.info(f"Placed {mountain_count} mountain tiles")

    def _generate_forests(self):
        """Place forest tiles on non-mountain terrain."""
        self.logger.debug("Generating forests")
        forest_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].terrain_type == mountains:
                    continue
                if random.random() < self.config.forest_density:
                    self.grid[y][x].terrain_type = forest
                    forest_count += 1

        self.logger.info(f"Placed {forest_count} forest tiles")

    def _generate_urban_areas(self):
        """Place urban tiles on plains (avoiding mountains and forests)."""
        self.logger.debug("Generating urban areas")
        urban_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].terrain_type in [mountains, forest]:
                    continue
                if random.random() < self.config.urban_density:
                    self.grid[y][x].terrain_type = urban
                    urban_count += 1

        self.logger.info(f"Placed {urban_count} urban tiles")

    def _generate_debris(self):
        """Place debris tiles on remaining plains."""
        self.logger.debug("Generating debris")
        debris_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].terrain_type != plains:
                    continue
                if random.random() < self.config.debris_density:
                    self.grid[y][x].terrain_type = debris
                    debris_count += 1

        self.logger.info(f"Placed {debris_count} debris tiles")

    def _generate_roads(self):
        """Place road tiles on plains and debris."""
        self.logger.debug("Generating roads")
        road_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].terrain_type not in [plains, debris]:
                    continue
                if random.random() < self.config.road_density:
                    self.grid[y][x].terrain_type = road
                    road_count += 1

        self.logger.info(f"Placed {road_count} road tiles")

    def _smooth_terrain(self):
        """Smooth terrain using cellular automata algorithm."""
        self.logger.debug(f"Smoothing terrain ({self.config.smoothing_passes} passes)")

        for pass_num in range(self.config.smoothing_passes):
            new_grid = [row[:] for row in self.grid]

            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    current_type = self.grid[y][x].terrain_type
                    same_neighbors = 0

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            neighbor = self.grid[y + dy][x + dx]
                            if neighbor.terrain_type == current_type:
                                same_neighbors += 1

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
            self.logger.debug(f"Smoothing pass {pass_num + 1} complete")

    def _generate_buildings(self):
        """Generate buildings on suitable terrain."""
        self.logger.debug("Generating buildings")
        building_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                # Buildings on plains, urban, or elevated plains
                terrain = self.grid[y][x].terrain_type
                is_ramp = self.grid[y][x].is_ramp

                # Skip unsuitable terrain
                if terrain in [mountains, forest, debris] or is_ramp:
                    continue

                # Check minimum spacing from other buildings
                if not self._check_building_spacing(x, y):
                    continue

                # Place building with configured probability
                if random.random() < self.config.building_density:
                    self.grid[y][x].set_building(True)
                    self.buildings.append((x, y))
                    building_count += 1

        self.logger.info(f"Placed {building_count} buildings")

    def _check_building_spacing(self, x: int, y: int) -> bool:
        """Check if a position maintains minimum building spacing."""
        for bx, by in self.buildings:
            distance = abs(x - bx) + abs(y - by)  # Manhattan distance
            if distance < MIN_BUILDING_SPACING:
                return False
        return True

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within map bounds."""
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _validate_fairness(self) -> bool:
        """Validate map fairness for competitive play."""
        self.logger.debug("Validating map fairness")

        # Check 1: Both spawn points accessible
        if not self._check_spawn_accessibility():
            self.logger.warning("Spawn points not fully accessible")
            return False

        # Check 2: Balanced path lengths
        if not self._check_path_balance():
            self.logger.warning("Path lengths imbalanced")
            return False

        # Check 3: Balanced building access
        if not self._check_building_balance():
            self.logger.warning("Building access imbalanced")
            return False

        # Check 4: Balanced high ground access
        if not self._check_elevation_balance():
            self.logger.warning("High ground access imbalanced")
            return False

        self.logger.debug("Map passed all fairness checks")
        return True

    def _check_spawn_accessibility(self) -> bool:
        """Check if both spawn points can access the entire map."""
        sp1_reachable = self._get_reachable_tiles(self.config.spawn_point_1)
        sp2_reachable = self._get_reachable_tiles(self.config.spawn_point_2)

        total_tiles = self.config.width * self.config.height
        accessible_from_both = len(sp1_reachable.intersection(sp2_reachable))

        # At least 80% of map should be accessible from both spawns
        return accessible_from_both >= 0.8 * total_tiles

    def _check_path_balance(self) -> bool:
        """Check if path lengths between spawns and key points are balanced."""
        # Find path from spawn1 to spawn2
        path1 = self._find_path(self.config.spawn_point_1, self.config.spawn_point_2)
        path2 = self._find_path(self.config.spawn_point_2, self.config.spawn_point_1)

        if not path1 or not path2:
            return False

        # Paths should be similar length
        length_diff = abs(len(path1) - len(path2))
        avg_length = (len(path1) + len(path2)) / 2

        return length_diff <= PATH_LENGTH_TOLERANCE * avg_length

    def _check_building_balance(self) -> bool:
        """Check if both players have balanced access to buildings."""
        if len(self.buildings) == 0:
            return True  # No buildings to balance

        sp1_distances = []
        sp2_distances = []

        for building_pos in self.buildings:
            dist1 = self._manhattan_distance(self.config.spawn_point_1, building_pos)
            dist2 = self._manhattan_distance(self.config.spawn_point_2, building_pos)
            sp1_distances.append(dist1)
            sp2_distances.append(dist2)

        avg_dist1 = sum(sp1_distances) / len(sp1_distances)
        avg_dist2 = sum(sp2_distances) / len(sp2_distances)

        # Average distances should be similar
        diff = abs(avg_dist1 - avg_dist2)
        avg = (avg_dist1 + avg_dist2) / 2

        return diff <= RESOURCE_BALANCE_TOLERANCE * avg

    def _check_elevation_balance(self) -> bool:
        """Check if both players have balanced access to high ground."""
        sp1_high_ground = self._count_nearby_high_ground(self.config.spawn_point_1)
        sp2_high_ground = self._count_nearby_high_ground(self.config.spawn_point_2)

        if sp1_high_ground == 0 and sp2_high_ground == 0:
            return True  # No high ground on map

        total_high_ground = sp1_high_ground + sp2_high_ground
        if total_high_ground == 0:
            return True

        # High ground access should be balanced
        balance = min(sp1_high_ground, sp2_high_ground) / max(sp1_high_ground, sp2_high_ground)

        return balance >= (1.0 - RESOURCE_BALANCE_TOLERANCE)

    def _get_reachable_tiles(self, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Get all tiles reachable from a starting position using BFS."""
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()

            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if not self._is_valid_position(nx, ny):
                    continue

                if (nx, ny) in visited:
                    continue

                # Check if tile is passable
                tile_obj = self.grid[ny][nx]
                if tile_obj.terrain_type == mountains:
                    continue

                # Check elevation (can only traverse with ramps or same level)
                if not self._can_traverse(x, y, nx, ny):
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return visited

    def _can_traverse(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if a unit can traverse from one tile to another."""
        tile1 = self.grid[y1][x1]
        tile2 = self.grid[y2][x2]

        elev_diff = abs(tile1.elevation - tile2.elevation)

        # Same elevation: always passable
        if elev_diff == 0:
            return True

        # Different elevation: need ramp
        if elev_diff <= RAMP_ELEVATION_INCREMENT:
            return tile1.is_ramp or tile2.is_ramp

        # Too steep without multiple ramps
        return False

    def _find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path between two points using A* algorithm."""
        from heapq import heappush, heappop

        def heuristic(pos):
            return self._manhattan_distance(pos, goal)

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor

                if not self._is_valid_position(nx, ny):
                    continue

                if self.grid[ny][nx].terrain_type == mountains:
                    continue

                if not self._can_traverse(x, y, nx, ny):
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _count_nearby_high_ground(self, position: Tuple[int, int], radius: int = 8) -> int:
        """Count elevated tiles within radius of a position."""
        count = 0
        x, y = position

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if self._is_valid_position(nx, ny):
                    if self.grid[ny][nx].elevation > MIN_ELEVATION:
                        count += 1

        return count

    def get_terrain_statistics(self):
        """Calculate terrain type distribution statistics."""
        stats = {
            "plains": 0, "forest": 0, "urban": 0,
            "mountains": 0, "road": 0, "highway": 0,
            "debris": 0, "buildings": len(self.buildings),
            "ramps": 0, "elevated": 0
        }

        for row in self.grid:
            for tile_obj in row:
                name = tile_obj.terrain_type.name.lower()
                if name in stats:
                    stats[name] += 1
                if tile_obj.is_ramp:
                    stats["ramps"] += 1
                if tile_obj.elevation > MIN_ELEVATION:
                    stats["elevated"] += 1

        total = self.config.width * self.config.height
        self.logger.info(f"Terrain stats: {stats} (total: {total})")
        return stats

    def export_map_ascii(self):
        """Export the map as ASCII art for console viewing."""
        lines = []
        for row in self.grid:
            line = ""
            for t in row:
                if t.is_building:
                    line += "B"
                elif t.is_ramp:
                    line += "R"
                elif t.elevation > 0:
                    line += str(t.elevation)
                else:
                    line += ASCII_TERRAIN_SYMBOLS.get(t.terrain_type, "?")
            lines.append(line)
        return "\n".join(lines)

    def visualize_map_pygame(self):
        """Display the generated map in a pygame window with all features."""
        if pygame is None:
            print("Pygame not installed. Run `pip install pygame` to enable visualization.")
            return

        pygame.init()
        width_px = self.config.width * TILE_SIZE
        height_px = self.config.height * TILE_SIZE
        window = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption(PREVIEW_WINDOW_TITLE)

        clock = pygame.time.Clock()

        label_font = pygame.font.SysFont(
            UITheme.FONT_TILE_LABEL["name"],
            UITheme.FONT_TILE_LABEL["size"],
            UITheme.FONT_TILE_LABEL["bold"]
        )

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            window.fill(PREVIEW_BACKGROUND_COLOR)

            mouse_x, mouse_y = pygame.mouse.get_pos()
            hovered_tile_x = mouse_x // TILE_SIZE
            hovered_tile_y = mouse_y // TILE_SIZE

            for row in self.grid:
                for cell in row:
                    # Determine base color
                    if cell.is_building:
                        base_color = BUILDING_COLOR
                    elif cell.is_ramp:
                        base_color = RAMP_COLOR
                    else:
                        base_color = TERRAIN_COLOR_MAP.get(cell.terrain_type, (255, 0, 255))

                    # Apply elevation brightness
                    if cell.elevation > 0:
                        multiplier = ELEVATION_COLOR_MULTIPLIERS[min(cell.elevation, MAX_ELEVATION)]
                        base_color = tuple(min(255, int(c * multiplier)) for c in base_color)

                    is_hovered = (cell.x == hovered_tile_x and cell.y == hovered_tile_y)

                    if is_hovered:
                        tile_color = UITheme.darken_color(base_color, UITheme.TILE_HOVER_DARKEN_FACTOR)
                    else:
                        tile_color = base_color

                    tile_rect = pygame.Rect(
                        cell.x * TILE_SIZE,
                        cell.y * TILE_SIZE,
                        TILE_SIZE,
                        TILE_SIZE
                    )
                    pygame.draw.rect(window, tile_color, tile_rect)

                    if is_hovered:
                        # Show detailed info
                        info_parts = [cell.terrain_type.name]
                        if cell.is_building:
                            info_parts.append("BLDG")
                        if cell.is_ramp:
                            info_parts.append(f"Ramp({cell.ramp_direction})")
                        if cell.elevation > 0:
                            info_parts.append(f"Elev:{cell.elevation}")

                        info_text = " ".join(info_parts)

                        label_surface = label_font.render(
                            info_text,
                            True,
                            base_color
                        )

                        label_rect = label_surface.get_rect(center=tile_rect.center)
                        window.blit(label_surface, label_rect)

            # Draw spawn points
            for spawn_idx, spawn_pos in enumerate([self.config.spawn_point_1, self.config.spawn_point_2]):
                spawn_rect = pygame.Rect(
                    spawn_pos[0] * TILE_SIZE,
                    spawn_pos[1] * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE
                )
                color = (0, 255, 0) if spawn_idx == 0 else (255, 0, 0)
                pygame.draw.rect(window, color, spawn_rect, 3)

            pygame.display.flip()
            clock.tick(PREVIEW_FPS)

        pygame.quit()


def display_console_menu():
    """Display the main console menu for map generation."""
    print("\n" + "="*60)
    print("   SNAKES IN COMBAT - MAP GENERATOR")
    print("="*60)
    print("\nGeneration Options:")
    print("  [1] Generate with Default Settings (with preview)")
    print("  [2] Generate with Custom Settings (with preview)")
    print("  [3] Quick Generate (no preview)")
    print("  [Q] Quit")
    print("-"*60)

    choice = input("\nSelect option [1/2/3/Q]: ").strip().lower()
    return choice


def generate_map_from_console():
    """Interactive console-based map generation tool with full configuration."""
    logger = get_logger(__name__)
    logger.info("Starting console map generation with custom configuration")

    print("\n" + "-"*60)
    print("   CUSTOM MAP CONFIGURATION")
    print("-"*60 + "\n")

    try:
        # Map dimensions
        print("Map Dimensions:")
        width = int(input(f"  Width (default {MAP_WIDTH}): ") or MAP_WIDTH)
        height = int(input(f"  Height (default {MAP_HEIGHT}): ") or MAP_HEIGHT)

        # Terrain densities
        print("\nTerrain Densities (0.0 - 1.0):")
        forest = float(input(f"  Forest (default {DEFAULT_FOREST_DENSITY}): ")
                      or DEFAULT_FOREST_DENSITY)
        urban = float(input(f"  Urban (default {DEFAULT_URBAN_DENSITY}): ")
                     or DEFAULT_URBAN_DENSITY)
        mountain = float(input(f"  Mountain (default {DEFAULT_MOUNTAIN_DENSITY}): ")
                        or DEFAULT_MOUNTAIN_DENSITY)
        road_dens = float(input(f"  Road (default {DEFAULT_ROAD_DENSITY}): ")
                         or DEFAULT_ROAD_DENSITY)
        debris_dens = float(input(f"  Debris (default {DEFAULT_DEBRIS_DENSITY}): ")
                           or DEFAULT_DEBRIS_DENSITY)

        # Elevation and buildings
        print("\nElevation & Buildings:")
        elevation = float(input(f"  Elevation density (default {DEFAULT_ELEVATION_DENSITY}): ")
                         or DEFAULT_ELEVATION_DENSITY)
        building = float(input(f"  Building density (default {DEFAULT_BUILDING_DENSITY}): ")
                        or DEFAULT_BUILDING_DENSITY)

        # Advanced options
        print("\nAdvanced Options:")
        smoothing = int(input(f"  Smoothing passes (default {DEFAULT_SMOOTHING_PASSES}): ")
                       or DEFAULT_SMOOTHING_PASSES)

        fairness_input = input("  Validate fairness? (Y/n): ").strip().lower()
        validate_fairness = fairness_input != 'n'

        seed_input = input("  Random seed (leave blank for random): ")
        seed = int(seed_input) if seed_input.strip() else None

        # Create configuration
        config = MapConfig(
            width=width, height=height,
            forest_density=forest, urban_density=urban,
            mountain_density=mountain, road_density=road_dens,
            debris_density=debris_dens, seed=seed,
            smoothing_passes=smoothing,
            elevation_density=elevation,
            building_density=building,
            validate_fairness=validate_fairness
        )

        logger.info("Configuration accepted, generating map")
        print("\n" + "="*60)
        print("Generating map with custom settings...")
        print("="*60 + "\n")

        # Generate map
        generator = MapGenerator(config)
        game_map = generator.generate()

        # Display statistics
        stats = generator.get_terrain_statistics()
        print("\nTerrain Distribution:")
        for terrain_type, count in stats.items():
            if terrain_type in ["buildings", "ramps", "elevated"]:
                print(f"  {terrain_type.capitalize():<12} {count:>4}")
            else:
                pct = (count / (width * height)) * 100
                print(f"  {terrain_type.capitalize():<12} {count:>4} tiles ({pct:>5.1f}%)")

        # Display ASCII preview
        print("\n" + "-"*60)
        print("ASCII Map Preview:")
        print("Legend: . = plains, T = forest, # = urban, ^ = mountains")
        print("        = = road, x = debris, B = building, R = ramp")
        print("        0-3 = elevation level")
        print("-"*60)
        print(generator.export_map_ascii())
        print("-"*60)

        # Show pygame preview
        print("\nLaunching interactive preview window...")
        print("(Hover over tiles to see details, ESC to close)")
        print("Green square = Player 1 spawn, Red square = Player 2 spawn")
        generator.visualize_map_pygame()

        logger.info("Console map generation complete")
        return game_map

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"\nError: Invalid input - {e}")
        print("Please enter valid numeric values.")
        return None
    except KeyboardInterrupt:
        logger.info("Map generation cancelled by user")
        print("\n\nCancelled.")
        return None


def generate_default_map(visualize=False):
    """Generate a map using default configuration settings."""
    logger = get_logger(__name__)
    logger.info("Generating default map")

    generator = MapGenerator()
    game_map = generator.generate()

    # Display statistics
    stats = generator.get_terrain_statistics()
    total_tiles = MAP_WIDTH * MAP_HEIGHT

    print("\n" + "="*60)
    print("   DEFAULT MAP GENERATED")
    print("="*60)
    print(f"\nDimensions: {MAP_WIDTH}x{MAP_HEIGHT} ({total_tiles} tiles)")
    print("\nTerrain Distribution:")
    for terrain_type, count in stats.items():
        if terrain_type in ["buildings", "ramps", "elevated"]:
            print(f"  {terrain_type.capitalize():<12} {count:>4}")
        else:
            pct = (count / total_tiles) * 100
            print(f"  {terrain_type.capitalize():<12} {count:>4} tiles ({pct:>5.1f}%)")

    if visualize:
        print("\n" + "-"*60)
        print("ASCII Map Preview:")
        print("Legend: . = plains, T = forest, # = urban, ^ = mountains")
        print("        = = road, x = debris, B = building, R = ramp")
        print("        0-3 = elevation level")
        print("-"*60)
        print(generator.export_map_ascii())
        print("-"*60)
        print("\nLaunching interactive preview window...")
        print("(Hover over tiles to see details, ESC to close)")
        print("Green square = Player 1 spawn, Red square = Player 2 spawn")
        generator.visualize_map_pygame()

    logger.info("Default map generation complete")
    return game_map


def generate_quick_map():
    """Generate a default map without visualization or detailed output."""
    logger = get_logger(__name__)
    logger.info("Generating map in quick mode (no preview)")

    generator = MapGenerator()
    game_map = generator.generate(visualize=False)

    print("\nMap generated successfully (no preview)")
    logger.info("Quick map generation complete")
    return game_map


def run_interactive_menu():
    """Run the interactive console menu for map generation."""
    logger = get_logger(__name__)
    logger.info("Starting interactive map generator menu")

    while True:
        choice = display_console_menu()

        if choice == '1':
            # Default with preview
            print("\nGenerating map with default settings...")
            generate_default_map(visualize=True)
            input("\nPress Enter to return to menu...")

        elif choice == '2':
            # Custom with preview
            game_map = generate_map_from_console()
            if game_map:
                input("\nPress Enter to return to menu...")

        elif choice == '3':
            # Quick generation
            generate_quick_map()
            input("\nPress Enter to return to menu...")

        elif choice == 'q':
            print("\nExiting map generator. Goodbye!")
            logger.info("User exited map generator")
            break

        else:
            print("\nInvalid option. Please choose 1, 2, 3, or Q.")


if __name__ == "__main__":
    # Main execution: Run interactive menu
    run_interactive_menu()