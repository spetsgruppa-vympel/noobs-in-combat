"""
Procedural Map Generator for Snakes in Combat.

Generates realistic but playable tactical maps with varied terrain, strategic positioning,
and building placements. Uses cellular automata and noise-based algorithms for natural
terrain distribution while maintaining gameplay balance.

Map Generation Roadmap:
=======================

PHASE 1 - Basic Terrain Generation (CURRENT):
    - Grid initialization with plains as default
    - Noise-based terrain distribution
    - Configurable map dimensions and terrain densities
    - Console-based configuration

PHASE 2 - Advanced Terrain Features:
    - River/water body generation with flow paths
    - Mountain ranges with realistic elevation
    - Road networks connecting key points
    - Forest clusters with natural boundaries
    - Urban area generation with building foundations

PHASE 3 - Strategic Elements:
    - Spawn point placement for balanced starts
    - Capturable building placement
    - Resource point distribution
    - Chokepoint identification and balancing
    - Line of sight analysis for fairness

PHASE 4 - Building System Integration:
    - Building type selection (HQ, factories, etc.)
    - Building health and capture mechanics
    - Building benefits and unit production
    - Victory condition integration

PHASE 5 - Playability Optimization:
    - Path validation (all areas reachable)
    - Balance scoring system
    - Automatic reroll for unbalanced maps
    - Difficulty scaling
    - Competitive balance metrics

PHASE 6 - Advanced Features:
    - Biome system (desert, arctic, jungle, etc.)
    - Weather effects integration
    - Dynamic map events
    - Multi-layer elevation system
    - Campaign map linking

Current Implementation: PHASE 1 + partial PHASE 2
"""

import random
import pygame
from main.config import get_logger, MAP_WIDTH, MAP_HEIGHT, TILE_SIZE
from main.game.data.maps.terrain import plains, forest, urban, mountains, road, highway, debris
from main.game.data.maps.tile import tile


class MapConfig:
    def __init__(
            self,
            width=MAP_WIDTH,
            height=MAP_HEIGHT,
            forest_density=0.25,
            urban_density=0.15,
            mountain_density=0.10,
            road_density=0.08,
            debris_density=0.05,
            seed=None,
            smoothing_passes=2,
            min_cluster_size=3
    ):
        self.width = width
        self.height = height
        self.forest_density = forest_density
        self.urban_density = urban_density
        self.mountain_density = mountain_density
        self.road_density = road_density
        self.debris_density = debris_density
        self.seed = seed
        self.smoothing_passes = smoothing_passes
        self.min_cluster_size = min_cluster_size

        self.logger = get_logger(__name__)
        self.logger.info(
            f"MapConfig created: {width}x{height}, "
            f"forest={forest_density}, urban={urban_density}, "
            f"mountain={mountain_density}, seed={seed}"
        )


class MapGenerator:
    def __init__(self, config=None):
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)
        self.grid = []

        if self.config.seed is not None:
            random.seed(self.config.seed)
            self.logger.info(f"Random seed set to {self.config.seed}")

        self.logger.info("MapGenerator initialized")

    def generate(self, visualize=False):
        self.logger.info(f"Starting map generation: {self.config.width}x{self.config.height}")

        self._initialize_grid()
        self._generate_mountains()
        self._generate_forests()
        self._generate_urban_areas()
        self._generate_debris()
        self._generate_roads()
        self._smooth_terrain()

        self.logger.info("Map generation complete")

        if visualize:
            self.visualize_map_pygame()

        return self.grid

    def _initialize_grid(self):
        self.logger.debug("Initializing grid with plains")
        self.grid = []

        for y in range(self.config.height):
            row = []
            for x in range(self.config.width):
                t = tile(x=x, y=y, terrain_type=plains, size=TILE_SIZE, occupied=False)
                row.append(t)
            self.grid.append(row)

        self.logger.debug(f"Grid initialized: {len(self.grid)}x{len(self.grid[0])}")

    def _generate_mountains(self):
        self.logger.debug("Generating mountains")
        mountain_count = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                if random.random() < self.config.mountain_density:
                    self.grid[y][x].terrain_type = mountains
                    mountain_count += 1

        self.logger.info(f"Placed {mountain_count} mountain tiles")

    def _generate_forests(self):
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

                    if same_neighbors < 3 and current_type != plains:
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

    def get_terrain_statistics(self):
        stats = {
            "plains": 0, "forest": 0, "urban": 0,
            "mountains": 0, "road": 0,
            "highway": 0, "debris": 0
        }

        for row in self.grid:
            for tile_obj in row:
                name = tile_obj.terrain_type.name.lower()
                if name in stats:
                    stats[name] += 1

        total = self.config.width * self.config.height
        self.logger.info(f"Terrain stats: {stats} (total: {total})")
        return stats

    def export_map_ascii(self):
        symbols = {
            plains: ".", forest: "T", urban: "#",
            mountains: "^", road: "=", highway: "≡",
            debris: "x"
        }
        lines = []
        for row in self.grid:
            lines.append("".join(symbols.get(t.terrain_type, "?") for t in row))
        return "\n".join(lines)

    def visualize_map_pygame(self):
        if pygame is None:
            print("Pygame not installed. Run `pip install pygame` to enable visualization.")
            return

        pygame.init()
        width_px = self.config.width * TILE_SIZE
        height_px = self.config.height * TILE_SIZE
        window = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Snakes in Combat - Tactical Map Preview")

        clock = pygame.time.Clock()

        COLOR_MAP = {
            plains: (200, 200, 140), forest: (34, 139, 34),
            urban: (90, 90, 90), mountains: (120, 120, 120),
            road: (180, 150, 75), highway: (210, 180, 50),
            debris: (130, 110, 90)
        }

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            window.fill((0, 0, 0))

            for row in self.grid:
                for cell in row:
                    color = COLOR_MAP.get(cell.terrain_type, (255, 0, 255))
                    pygame.draw.rect(
                        window, color,
                        pygame.Rect(cell.x * TILE_SIZE, cell.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    )

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


def generate_map_from_console():
    logger = get_logger(__name__)
    logger.info("Starting console map generation")
    print("\n=== Snakes in Combat - Map Generator ===\n")

    try:
        width = int(input(f"Map width (default {MAP_WIDTH}): ") or MAP_WIDTH)
        height = int(input(f"Map height (default {MAP_HEIGHT}): ") or MAP_HEIGHT)
        forest = float(input("Forest density 0.0-1.0 (default 0.25): ") or 0.25)
        urban = float(input("Urban density 0.0-1.0 (default 0.15): ") or 0.15)
        mountain = float(input("Mountain density 0.0-1.0 (default 0.10): ") or 0.10)
        road_dens = float(input("Road density 0.0-1.0 (default 0.08): ") or 0.08)
        debris_dens = float(input("Debris density 0.0-1.0 (default 0.05): ") or 0.05)

        seed_input = input("Random seed (leave blank for random): ")
        seed = int(seed_input) if seed_input.strip() else None

        config = MapConfig(
            width=width, height=height,
            forest_density=forest, urban_density=urban,
            mountain_density=mountain, road_density=road_dens,
            debris_density=debris_dens, seed=seed
        )

        logger.info("Configuration accepted, generating map")
        print("\nGenerating map...\n")

        generator = MapGenerator(config)
        game_map = generator.generate()

        stats = generator.get_terrain_statistics()
        print("\nTerrain Distribution:")
        for terrain_type, count in stats.items():
            pct = (count / (width * height)) * 100
            print(f"  {terrain_type.capitalize()}: {count} ({pct:.1f}%)")

        print("\nMap Preview (ASCII):")
        print(generator.export_map_ascii())
        logger.info("Console map generation complete")
        return game_map

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"\nError: Invalid input - {e}")
        return None
    except KeyboardInterrupt:
        logger.info("Map generation cancelled by user")
        print("\n\nCancelled.")
        return None


def generate_default_map(visualize=False):
    logger = get_logger(__name__)
    logger.info("Generating default map")
    generator = MapGenerator()
    game_map = generator.generate(visualize=visualize)
    logger.info("Default map generation complete")
    return game_map


if __name__ == "__main__":
    AUTO_VISUALIZE = True
    if AUTO_VISUALIZE:
        generate_default_map(visualize=True)
    else:
        generate_map_from_console()
