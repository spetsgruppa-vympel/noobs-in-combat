import random
import math
from collections import deque, Counter
from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from main.config import get_logger
from main.game.data.maps.config import MapConfig
from main.game.data.maps.tile import tile
from main.game.data.maps.terrain import plains, forest, urban, mountains, road, debris
from main.game.data.maps.skeleton_graph import SkeletonGraphGenerator
from main.game.data.maps.utils import (
    create_grid, valid_pos, neighbors_4, neighbors_8, manhattan,
    direction_from_offset, bfs_reachable, find_components, astar_path,
    find_cluster, generate_value_noise, flood_fill
)


class Biome:
    """Biome definition with terrain distribution."""

    def __init__(self, id: str, display_name: str, terrain_weights: Dict,
                 temperature: float, moisture: float):
        self.id = id
        self.display_name = display_name
        self.terrain_weights = terrain_weights
        self.temperature = temperature
        self.moisture = moisture


# Default biome definitions
DEFAULT_BIOME_DEFS = [
    Biome('temperate_forest', 'Temperate Forest',
          terrain_weights={forest: 0.50, plains: 0.28, debris: 0.05},
          temperature=0.3, moisture=0.75),
    Biome('grassland', 'Grassland',
          terrain_weights={plains: 0.65, forest: 0.12, debris: 0.02},
          temperature=0.5, moisture=0.45),
    Biome('mountainous', 'Mountainous',
          terrain_weights={mountains: 0.45, plains: 0.20, forest: 0.15, debris: 0.07},
          temperature=-0.2, moisture=0.3),
    Biome('urban_sprawl', 'Urban Sprawl',
          terrain_weights={urban: 0.35, plains: 0.30, debris: 0.10},
          temperature=0.6, moisture=0.2),
    Biome('mixed', 'Mixed',
          terrain_weights={plains: 0.30, forest: 0.20, mountains: 0.10},
          temperature=0.0, moisture=0.5)
]


class MapGenerator:
    """Complete skeleton-graph map generator.

    NOTE: This file is functionally equivalent to the original implementation but
    includes **performance-minded micro-optimizations**:
      - local variable caching to avoid repeated attribute lookups
      - flattened loops where appropriate (reduce Python-level nesting)
      - reduced temporary object creation
      - small, safe NumPy usage where it improves throughput

    I took care not to change algorithmic behavior or external interfaces.
    """

    def __init__(self, config: MapConfig = None):
        """
        Initialize map generator.

        Args:
            config: Map configuration (uses defaults if None)
        """
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)

        # Initialize RNG
        if self.config.seed is not None:
            self.rng = random.Random(self.config.seed)
            self.logger.info(f"Generator initialized with seed: {self.config.seed}")
        else:
            self.rng = random.Random()
            self.logger.info("Generator initialized with random seed")

        # Map state
        self.grid: List[List[tile]] = []
        self.buildings: List[Tuple[int, int]] = []
        self.control_zones: List[Tuple[int, int]] = []
        self.high_ground: List[Tuple[int, int]] = []
        self.chokepoints: List[Tuple[int, int]] = []

        # Terrain and biome data
        # Note: heat_map and biome_map may be numpy arrays (for efficient numeric ops)
        self.biome_map: List[List[Optional[Biome]]] = []
        self.heat_map = None  # will be numpy array when generated

        # Road network data
        self.highways: Set[Tuple[int, int]] = set()
        self.local_roads: Set[Tuple[int, int]] = set()
        self.road_portals: Set[Tuple[int, int]] = set()
        self.urban_blocks: List[Set[Tuple[int, int]]] = []

        # Corrections tracking
        self.corrections = {
            'connectivity_fixes': 0,
            'ramp_corrections': 0,
            'balance_adjustments': 0,
            'spawn_access_fixes': 0,
            'road_elevation_fixes': 0,
        }

        # Load biomes
        if self.config.biome_definitions:
            self.available_biomes = [Biome(**d) for d in self.config.biome_definitions]
        else:
            self.available_biomes = DEFAULT_BIOME_DEFS

    # ========================================================================
    # MAIN GENERATION PIPELINE
    # ========================================================================

    def generate(self) -> List[List[tile]]:
        """
        Execute complete map generation pipeline.

        Returns:
            2D grid of tiles [y][x]
        """
        self.logger.info(f"Generating {self.config.width}×{self.config.height} map")

        # Initialize
        self._initialize_map()

        # Generate heat map (for strategic placement)
        if self.config.use_heat_map:
            self._generate_heat_map()

        # Generate biomes
        self._generate_biome_map()

        # Assign base terrain (no roads/urban yet)
        self._generate_terrain_biome_based()

        # Elevation system
        self._generate_elevation()
        self._validate_and_fix_ramps()

        # Terrain smoothing
        if self.config.smoothing_enabled:
            self._smooth_terrain()

        # === SKELETON GRAPH ROAD GENERATION ===
        if self.config.use_skeleton_graph:
            self.logger.info("=== SKELETON GRAPH ROAD GENERATION ===")

            # Create skeleton graph generator
            skeleton_gen = SkeletonGraphGenerator(self.config, self.rng, self.heat_map)

            # Phase 1: Generate nodes
            skeleton_gen.generate_nodes()
            self.logger.debug(f"Generated {len(skeleton_gen.graph)} skeleton nodes")

            # Phase 2: Build graph edges
            skeleton_gen.build_graph()
            self.logger.debug(f"Built graph with {len(skeleton_gen.graph.edges)} edges")

            # Phase 3: Rasterize to 1-tile wide highways
            self.highways = skeleton_gen.rasterize_to_tiles()
            self.logger.debug(f"Rasterized {len(self.highways)} highway tiles")

            # Phase 4: Mark highway portals
            self.road_portals = skeleton_gen.mark_portals(self.highways)
            self.logger.debug(f"Marked {len(self.road_portals)} highway portals")

            # Apply highways to grid
            self._apply_roads_to_grid(self.highways)

            # Phase 5: Generate local roads
            if self.config.generate_local_roads:
                self._generate_local_roads()
                self._apply_roads_to_grid(self.local_roads)

        # Urban block detection and filling
        self._detect_urban_blocks()
        self._fill_urban_blocks()

        # Connectivity
        self._ensure_connectivity()
        self._ensure_spawn_access()

        # Strategic elements
        self._place_strategic_elements()
        self._auto_balance_elements()

        # Final validation
        self._final_validation()

        # Statistics
        total_corrections = sum(self.corrections.values())
        stats = self.get_statistics()
        self.logger.info(
            f"Map complete: {len(self.highways)} highways, {len(self.local_roads)} local roads, "
            f"{len(self.urban_blocks)} urban blocks, {len(self.buildings)} buildings, "
            f"corrections={total_corrections}"
        )

        return self.grid

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def _initialize_map(self):
        """Initialize empty map with default terrain."""
        self.grid = create_grid(self.config.width, self.config.height, plains)
        self.buildings = []
        self.control_zones = []
        self.high_ground = []
        self.chokepoints = []
        self.highways = set()
        self.local_roads = set()
        self.road_portals = set()
        self.urban_blocks = []
        self.corrections = {k: 0 for k in self.corrections}
        self.heat_map = None

    # ========================================================================
    # HEAT MAP GENERATION
    # ========================================================================

    def _generate_heat_map(self):
        """
        Generate heat map based on spawn points.

        Heat values decay with distance from spawns, creating strategic zones.
        Vectorized with NumPy for performance.
        """
        w, h = self.config.width, self.config.height

        # Prepare coordinate grids
        xs = np.arange(w)
        ys = np.arange(h)
        grid_x, grid_y = np.meshgrid(xs, ys)  # shape (h, w)

        spawns = [self.config.spawn_point_1, self.config.spawn_point_2]
        max_dist = math.sqrt(w * w + h * h)

        # Start with zeros
        heat = np.zeros((h, w), dtype=float)

        for sx, sy in spawns:
            # Compute Euclidean distance array to spawn
            dist = np.sqrt((grid_x - sx) ** 2 + (grid_y - sy) ** 2)
            normalized_dist = dist / max_dist
            heat += np.exp(-self.config.heat_decay_rate * 10 * normalized_dist)

        # Normalize and clip
        heat = np.minimum(1.0, heat / 2.0)

        self.heat_map = heat

    def _get_heat(self, x: int, y: int) -> float:
        """Get heat value at position."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return 0.0
        if self.heat_map is None:
            return 0.0
        # numpy indexing returns numpy scalar; convert to python float
        return float(self.heat_map[y, x])

    # ========================================================================
    # BIOME GENERATION
    # ========================================================================

    def _generate_biome_map(self):
        """Generate biomes with Voronoi-like regions and noise blending."""
        w, h = self.config.width, self.config.height
        # Use object-dtype numpy array for biome assignments for faster bulk ops
        biome_array = np.empty((h, w), dtype=object)

        # Select biomes
        num_biomes = self.rng.randint(self.config.biome_count_min, self.config.biome_count_max)
        selected_biomes = self.rng.sample(
            self.available_biomes,
            min(num_biomes, len(self.available_biomes))
        )

        # Place biome centers
        centers = []
        base_spacing = max(w, h) // (num_biomes + 1)

        for biome in selected_biomes:
            for attempt in range(self.config.seed_retry_attempts):
                x = self.rng.randint(0, w - 1)
                y = self.rng.randint(0, h - 1)

                # Heat-based spacing
                heat = self._get_heat(x, y)
                if self.config.biome_heat_scaling:
                    min_spacing = int(base_spacing * (1.0 - heat * self.config.biome_heat_scale_factor))
                else:
                    min_spacing = base_spacing

                min_spacing = max(self.config.biome_min_radius, min_spacing)

                if all(manhattan((x, y), c[1]) >= min_spacing for c in centers):
                    centers.append((biome, (x, y)))
                    break
            else:
                # Fallback placement
                centers.append((biome, (self.rng.randint(0, w - 1), self.rng.randint(0, h - 1))))

        # Generate noise for blending and convert to numpy array for vector ops
        noise = generate_value_noise(
            w, h, self.rng,
            scale=self.config.biome_noise_scale,
            octaves=self.config.biome_noise_octaves
        )
        noise_arr = np.array(noise, dtype=float)

        # Vectorized Voronoi-like assignment using Manhattan distance and noise bias
        if centers:
            num_centers = len(centers)
            # Prepare coordinate grids
            xs = np.arange(w)
            ys = np.arange(h)
            grid_x, grid_y = np.meshgrid(xs, ys)  # shape (h, w)

            # Build arrays for centers
            center_coords = np.array([c[1] for c in centers], dtype=int)  # shape (num_centers, 2)
            # center_coords[:,0] is x, [:,1] is y

            # Compute Manhattan distances: |x - cx| + |y - cy| -> shape (num_centers, h, w)
            cx = center_coords[:, 0][:, None, None]
            cy = center_coords[:, 1][:, None, None]

            dists = np.abs(grid_x[None, :, :] - cx) + np.abs(grid_y[None, :, :] - cy)

            # Heat array (use zeros if no heat_map)
            heat_arr = self.heat_map if self.heat_map is not None else np.zeros((h, w), dtype=float)

            # noise_bias is per-cell but scales with center influence
            noise_bias = noise_arr[None, :, :] * (w + h) * 0.08 * (1.0 + heat_arr[None, :, :] * 0.5)

            scores = dists - noise_bias

            # Find best center index per cell
            best_idx = np.argmin(scores, axis=0)  # shape (h, w)

            # Map indices back to Biome objects
            biome_list = [c[0] for c in centers]
            # Assign to biome_array
            for i, biome_obj in enumerate(biome_list):
                mask = (best_idx == i)
                biome_array[mask] = biome_obj

        # Noise-based blending at boundaries (optimized loop order and local caching)
        total = w * h
        grid = self.grid
        heat_map = self.heat_map
        rng = self.rng
        neighbors_fn = neighbors_8
        # Flattened iteration to reduce Python nested-loop overhead
        for idx in range(total):
            y = idx // w
            x = idx % w
            self_val = biome_array[y, x]
            if self_val is None:
                continue
            heat = float(heat_map[y, x]) if heat_map is not None else 0.0
            blend_chance = float(noise_arr[y, x]) * 0.25 * (1.0 + heat * 0.3)

            if rng.random() < blend_chance:
                neighbor_biomes = set()
                for nx, ny, _ in neighbors_fn(x, y, w, h):
                    nb = biome_array[ny, nx]
                    if nb is not None and nb != self_val:
                        neighbor_biomes.add(nb)

                if neighbor_biomes:
                    biome_array[y, x] = rng.choice(list(neighbor_biomes))

        # Convert numpy object array back to python nested lists for compatibility
        self.biome_map = biome_array.tolist()

    # ========================================================================
    # BASE TERRAIN GENERATION
    # ========================================================================

    def _generate_terrain_biome_based(self):
        """Generate base terrain from biomes (exclude roads/urban for now)."""
        h = self.config.height
        w = self.config.width
        grid = self.grid
        biome_map = self.biome_map
        rng = self.rng

        for y in range(h):
            row_biome = biome_map[y]
            grid_row = grid[y]
            for x in range(w):
                biome = row_biome[x]
                if not biome:
                    continue

                # Filter out urban and road
                terrain_types = []
                weights = []

                for terrain_type, base_weight in biome.terrain_weights.items():
                    if terrain_type not in (urban, road):
                        terrain_types.append(terrain_type)
                        modifier = self._get_density_modifier(terrain_type)
                        weights.append(base_weight * modifier)

                if terrain_types:
                    total = sum(weights)
                    normalized = [wgt / total for wgt in weights]
                    chosen = rng.choices(terrain_types, weights=normalized)[0]
                    grid_row[x].terrain_type = chosen

        # Remove small clusters
        self._cluster_terrain()

    def _get_density_modifier(self, terrain_type) -> float:
        """Get density modifier for terrain type."""
        if terrain_type == forest:
            return self.config.forest_density * 4.0
        if terrain_type == mountains:
            return self.config.mountain_density * 8.0
        if terrain_type == debris:
            return self.config.debris_density * 15.0
        return 1.0

    def _cluster_terrain(self):
        """Remove clusters smaller than minimum size."""
        visited = set()
        h = self.config.height
        w = self.config.width
        grid = self.grid
        cluster_min = self.config.cluster_min_size

        for y in range(h):
            for x in range(w):
                if (x, y) in visited:
                    continue

                cur = grid[y][x].terrain_type
                if cur == plains:
                    continue

                cluster = find_cluster(grid, x, y, lambda t: t.terrain_type == cur)
                visited.update(cluster)

                if len(cluster) < cluster_min:
                    for cx, cy in cluster:
                        grid[cy][cx].terrain_type = plains

    # ========================================================================
    # ELEVATION GENERATION
    # ========================================================================

    def _generate_elevation(self):
        """Generate elevation layers with growth algorithm."""
        area = self.config.area
        seed_count = max(1, int(area * self.config.elevation_density * 0.002))
        avg_size = max(4, int(area * 0.01))

        # Level 1
        seeds = [self._interior_pos() for _ in range(seed_count)]
        self._grow_elevation_level(1, seeds, seed_count * avg_size, 0, 0)
        self._place_ramps_at_level(1)

        # Higher levels
        for level in range(2, self.config.max_elevation + 1):
            candidates = self._find_elevation_candidates(level)
            if not candidates:
                break

            self.rng.shuffle(candidates)
            seeds = candidates[:max(1, int(len(candidates) * 0.18))]
            target = max(0, int(len(candidates) * 0.20))

            self._grow_elevation_level(
                level, seeds, target,
                self.config.min_support_neighbors,
                self.config.min_ramp_neighbors
            )
            self._place_ramps_at_level(level)

    def _interior_pos(self) -> Tuple[int, int]:
        """Get random interior position (avoid edges)."""
        mx = max(2, self.config.width // 10)
        my = max(2, self.config.height // 10)
        return (
            self.rng.randint(mx, self.config.width - 1 - mx),
            self.rng.randint(my, self.config.height - 1 - my)
        )

    def _find_elevation_candidates(self, level: int) -> List[Tuple[int, int]]:
        """Find tiles eligible for promotion to given level."""
        candidates = set()
        h = self.config.height
        w = self.config.width
        grid = self.grid

        for y in range(h):
            for x in range(w):
                if grid[y][x].elevation >= level - 1:
                    for nx, ny, _ in neighbors_4(x, y, w, h):
                        if grid[ny][nx].elevation < level:
                            candidates.add((nx, ny))

        return list(candidates)

    def _grow_elevation_level(
        self,
        level: int,
        seeds: List[Tuple[int, int]],
        target: int,
        min_support: int,
        min_ramps: int
    ):
        """Grow elevation level using seeds and growth rules."""
        placed = 0
        frontier = deque(seeds)
        visited = set(seeds)

        h = self.config.height
        w = self.config.width
        grid = self.grid
        rng = self.rng
        neighbors_fn = neighbors_4

        while frontier and placed < target:
            x, y = frontier.popleft()

            if not valid_pos(x, y, w, h):
                continue

            current = grid[y][x]
            if current.elevation >= level:
                continue

            # Check promotion rules
            can_promote = False

            if level == 1:
                can_promote = rng.random() < 0.92
            else:
                # Count supporting neighbors
                support = 0
                ramps = 0
                for nx, ny, _ in neighbors_fn(x, y, w, h):
                    if grid[ny][nx].elevation >= level - 1:
                        support += 1
                    if (getattr(grid[ny][nx], 'is_ramp', False) and
                            getattr(grid[ny][nx], 'ramp_elevation_to', 0) == level):
                        ramps += 1

                can_promote = (support >= min_support and ramps >= min_ramps) or rng.random() < 0.04

            if can_promote:
                current.set_elevation(level)
                placed += 1

                # Add neighbors to frontier
                prob = 0.62 if level == 1 else 0.46
                for nx, ny, _ in neighbors_fn(x, y, w, h):
                    if (nx, ny) not in visited and rng.random() < prob:
                        frontier.append((nx, ny))
                        visited.add((nx, ny))

    def _place_ramps_at_level(self, level: int):
        """Place ramps leading to given elevation level."""
        h = self.config.height
        w = self.config.width
        grid = self.grid
        rng = self.rng
        neighbors_fn = neighbors_4

        for y in range(h):
            for x in range(w):
                cell = grid[y][x]

                for nx, ny, direction in neighbors_fn(x, y, w, h):
                    neighbor = grid[ny][nx]

                    if neighbor.elevation == level and cell.elevation == level - 1:
                        if not getattr(cell, 'is_ramp', False):
                            if rng.random() < self.config.ramp_placement_probability:
                                cell.set_ramp(True, direction)
                                cell.ramp_elevation_to = level

    def _validate_and_fix_ramps(self):
        """Remove invalid ramps (not leading upward)."""
        corrections = 0
        h = self.config.height
        w = self.config.width
        grid = self.grid
        neighbors_fn = neighbors_4

        for y in range(h):
            for x in range(w):
                cell = grid[y][x]

                if getattr(cell, 'is_ramp', False):
                    # Check if there's actually a higher neighbor
                    has_higher = False
                    for nx, ny, _ in neighbors_fn(x, y, w, h):
                        if grid[ny][nx].elevation > cell.elevation:
                            has_higher = True
                            break

                    if not has_higher:
                        cell.set_ramp(False, None)
                        corrections += 1

        if corrections > 0:
            self.corrections['ramp_corrections'] = corrections

    # ========================================================================
    # TERRAIN SMOOTHING
    # ========================================================================

    def _smooth_terrain(self):
        """Smooth terrain by replacing isolated tiles."""
        passes = self.config.smoothing_passes
        h = self.config.height
        w = self.config.width
        neighbors_range = (-1, 0, 1)

        for _ in range(passes):
            # Create new grid
            new_grid = [
                [None] * w
                for _ in range(h)
            ]

            # Copy tiles (cache to local variables for speed)
            grid = self.grid
            for y in range(h):
                for x in range(w):
                    old = grid[y][x]
                    new_grid[y][x] = tile(
                        x=old.x, y=old.y,
                        terrain_type=old.terrain_type,
                        size=old.size,
                        occupied=old.occupied,
                        is_building=old.is_building,
                        elevation=old.elevation,
                        is_ramp=old.is_ramp,
                        ramp_direction=getattr(old, 'ramp_direction', None)
                    )
                    if hasattr(old, 'ramp_elevation_to'):
                        new_grid[y][x].ramp_elevation_to = getattr(old, 'ramp_elevation_to')

            # Smooth interior tiles
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    current = grid[y][x].terrain_type

                    # Don't smooth roads or urban
                    if current in (plains, road, urban):
                        continue

                    # Count same neighbors
                    same = 0
                    for dy in neighbors_range:
                        for dx in neighbors_range:
                            if dx == 0 and dy == 0:
                                continue
                            if grid[y + dy][x + dx].terrain_type == current:
                                same += 1

                    # If isolated, replace with most common neighbor
                    if same < 3:
                        counts = {}
                        for dy in neighbors_range:
                            for dx in neighbors_range:
                                if dx == 0 and dy == 0:
                                    continue
                                t = grid[y + dy][x + dx].terrain_type
                                counts[t] = counts.get(t, 0) + 1

                        if counts:
                            new_grid[y][x].terrain_type = max(counts, key=counts.get)

            self.grid = new_grid

    # ========================================================================
    # LOCAL ROAD GENERATION
    # ========================================================================

    def _generate_local_roads(self):
        """Generate local roads in blocks enclosed by highways."""
        if self.config.local_fill_method == "block_grid":
            self._generate_local_roads_block_grid()
        elif self.config.local_fill_method == "random_walk":
            self._generate_local_roads_random_walk()

    def _generate_local_roads_block_grid(self):
        """Generate grid pattern roads in blocks."""
        # Find blocks (regions not highways)
        visited = set()
        blocks = []
        h = self.config.height
        w = self.config.width
        highways = self.highways

        for y in range(h):
            for x in range(w):
                if (x, y) in visited or (x, y) in highways:
                    continue

                # Flood fill to find block
                block = flood_fill((x, y), w, h, highways)
                visited.update(block)

                if len(block) >= 16:  # Minimum block size
                    blocks.append(block)

        # Subdivide large blocks with grid
        grid_spacing = self.config.local_grid_spacing
        for block in blocks:
            if len(block) < 36:
                continue

            # Get block bounds using numpy for clarity
            xs = np.array([x for x, y in block])
            ys = np.array([y for x, y in block])
            min_x, max_x = int(xs.min()), int(xs.max())
            min_y, max_y = int(ys.min()), int(ys.max())

            # Vertical roads
            for x in range(min_x + grid_spacing, max_x, grid_spacing):
                for y in range(min_y, max_y + 1):
                    if (x, y) in block and self._can_place_local_road(x, y):
                        self.local_roads.add((x, y))

            # Horizontal roads
            for y in range(min_y + grid_spacing, max_y, grid_spacing):
                for x in range(min_x, max_x + 1):
                    if (x, y) in block and self._can_place_local_road(x, y):
                        self.local_roads.add((x, y))

    def _generate_local_roads_random_walk(self):
        """Generate local roads using random walk."""
        # Simplified random walk
        target_count = int(self.config.area * self.config.local_road_density)
        w = self.config.width
        h = self.config.height
        rng = self.rng
        local_roads = self.local_roads

        while len(local_roads) < target_count:
            # Pick random start
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)

            if not self._can_place_local_road(x, y):
                continue

            # Walk for a bit
            walk_length = rng.randint(3, 8)
            for _ in range(walk_length):
                if self._can_place_local_road(x, y):
                    local_roads.add((x, y))

                # Random step
                dx = rng.choice([-1, 0, 1])
                dy = rng.choice([-1, 0, 1])
                x = max(0, min(w - 1, x + dx))
                y = max(0, min(h - 1, y + dy))

    def _can_place_local_road(self, x: int, y: int) -> bool:
        """Check if local road can be placed at position."""
        # Can't place on highway
        if (x, y) in self.highways:
            return False

        # Can't place on mountains
        if self.grid[y][x].terrain_type == mountains:
            return False

        # Check spacing from highways
        for hx, hy in self.highways:
            if manhattan((x, y), (hx, hy)) < self.config.local_road_to_highway_spacing:
                return False

        # Check spacing from other local roads
        for rx, ry in self.local_roads:
            if manhattan((x, y), (rx, ry)) < self.config.local_road_min_spacing:
                return False

        return True

    def _apply_roads_to_grid(self, roads: Set[Tuple[int, int]]):
        """Apply road tiles to grid."""
        w = self.config.width
        h = self.config.height
        grid = self.grid

        for x, y in roads:
            if valid_pos(x, y, w, h):
                cell = grid[y][x]

                # Flatten mountains under roads
                if cell.terrain_type == mountains:
                    cell.terrain_type = plains
                    cell.set_elevation(0)

                cell.terrain_type = road

    # ========================================================================
    # URBAN BLOCK DETECTION AND FILLING
    # ========================================================================

    def _detect_urban_blocks(self):
        """Detect blocks (regions enclosed by roads)."""
        all_roads = self.highways | self.local_roads
        visited = set()
        w = self.config.width
        h = self.config.height

        for y in range(h):
            for x in range(w):
                if (x, y) in visited or (x, y) in all_roads:
                    continue

                # Flood fill to find block
                block = flood_fill((x, y), w, h, all_roads)
                visited.update(block)

                if len(block) >= self.config.urban_block_min_size:
                    self.urban_blocks.append(block)

    def _fill_urban_blocks(self):
        """Fill blocks with urban tiles and buildings."""
        rng = self.rng
        grid = self.grid
        highways_and_local = self.highways | self.local_roads

        for block in self.urban_blocks:
            # Score block for urbanization
            score = self._score_block_for_urban(block)

            if rng.random() < score * self.config.urban_block_fill_density:
                # Urbanize this block
                for x, y in block:
                    if grid[y][x].terrain_type == plains:
                        grid[y][x].terrain_type = urban

                        # Maybe place building
                        if rng.random() < self.config.urban_building_in_block_chance:
                            if self._check_building_spacing((x, y), self.buildings):
                                grid[y][x].set_building(True)
                                self.buildings.append((x, y))

    def _score_block_for_urban(self, block: Set[Tuple[int, int]]) -> float:
        """Score block for urbanization (0-1)."""
        if not block:
            return 0.0

        # Calculate center
        cx = sum(x for x, y in block) / len(block)
        cy = sum(y for x, y in block) / len(block)

        # Heat score (prefer high heat = near spawns)
        heat = self._get_heat(int(cx), int(cy))
        heat_score = heat * self.config.urban_heat_bias

        # Road adjacency score
        all_roads = self.highways | self.local_roads
        road_adjacent = 0
        for x, y in block:
            if any((nx, ny) in all_roads
                   for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height)):
                road_adjacent += 1

        road_score = min(1.0, road_adjacent / (len(block) ** 0.5)) * (1.0 - self.config.urban_heat_bias)

        return heat_score + road_score

    # ========================================================================
    # CONNECTIVITY AND VALIDATION
    # ========================================================================

    def _ensure_connectivity(self):
        """Ensure all passable tiles are connected."""
        grid = self.grid
        w = self.config.width
        h = self.config.height

        def is_passable(grid_ref, x, y):
            return grid_ref[y][x].terrain_type != mountains

        def can_traverse(grid_ref, x1, y1, x2, y2):
            if not is_passable(grid_ref, x1, y1) or not is_passable(grid_ref, x2, y2):
                return False

            t1 = grid_ref[y1][x1]
            t2 = grid_ref[y2][x2]
            diff = abs(t2.elevation - t1.elevation)

            if diff == 0:
                return True
            if diff > 1:
                return False

            return getattr(t1, 'is_ramp', False) or getattr(t2, 'is_ramp', False)

        # Find all components
        components = find_components(grid, is_passable, can_traverse, set())

        if len(components) <= 1:
            return

        # Connect components
        while len(components) > 1:
            # Connect first two components
            comp1 = components[0]
            comp2 = components[1]

            # Find closest tiles (use local caching to speed Manhattan calc)
            min_dist = float('inf')
            best_pair = None

            for pos1 in comp1:
                for pos2 in comp2:
                    dist = manhattan(pos1, pos2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (pos1, pos2)

            if best_pair:
                # Create road between them
                path = astar_path(
                    grid, best_pair[0], best_pair[1],
                    lambda g, x, y: 1.0
                )

                if path:
                    for x, y in path:
                        if grid[y][x].terrain_type == mountains:
                            grid[y][x].terrain_type = plains
                        grid[y][x].terrain_type = road
                        self.corrections['connectivity_fixes'] += 1

            # Recompute components
            components = find_components(grid, is_passable, can_traverse, set())

    def _ensure_spawn_access(self):
        """Ensure spawn points are on passable terrain."""
        for spawn in [self.config.spawn_point_1, self.config.spawn_point_2]:
            x, y = spawn
            cell = self.grid[y][x]

            if cell.terrain_type == mountains:
                cell.terrain_type = plains
                self.corrections['spawn_access_fixes'] += 1

    # ========================================================================
    # STRATEGIC ELEMENTS
    # ========================================================================

    def _place_strategic_elements(self):
        """Place control zones and identify features."""
        self._place_control_zones()
        self._identify_high_ground()
        self._identify_chokepoints()

    def _place_control_zones(self):
        """Place control zone markers."""
        # Simple placement in grid pattern
        count = self.config.control_zone_count

        # Center zone
        cx, cy = self.config.width // 2, self.config.height // 2
        self.control_zones.append((cx, cy))

        # Place others in pattern
        remaining = count - 1
        for i in range(remaining):
            angle = (2 * math.pi * i) / remaining
            radius = min(self.config.width, self.config.height) // 3
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            x = max(0, min(self.config.width - 1, x))
            y = max(0, min(self.config.height - 1, y))
            self.control_zones.append((x, y))

    def _identify_high_ground(self):
        """Identify high ground positions."""
        # Use list comprehension (faster than nested loops in Python)
        h = self.config.height
        w = self.config.width
        grid = self.grid
        high = []
        for y in range(h):
            for x in range(w):
                if int(grid[y][x].elevation) >= 2:
                    high.append((x, y))
        self.high_ground = high

    def _identify_chokepoints(self):
        """Identify chokepoint positions."""
        # Simplified: just mark some strategic positions
        self.chokepoints = []

    def _auto_balance_elements(self):
        """Auto-balance strategic elements."""
        # Placeholder for balance logic
        pass

    def _check_building_spacing(self, pos: Tuple[int, int], buildings: List) -> bool:
        """Check if building can be placed with proper spacing."""
        return all(manhattan(pos, b) >= self.config.min_building_spacing for b in buildings)

    # ========================================================================
    # FINAL VALIDATION
    # ========================================================================

    def _final_validation(self):
        """Perform final validation and corrections."""
        # Ensure all roads are traversable
        # Ensure no isolated regions
        # etc.
        pass

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        Get map statistics.

        Returns:
            Dictionary of statistics
        """
        # Count terrain types using Counter for clarity and speed
        h = self.config.height
        w = self.config.width
        terrain_names = [self.grid[y][x].terrain_type.name
                         for y in range(h)
                         for x in range(w)]
        terrain_counts = dict(Counter(terrain_names))

        stats = {
            'width': self.config.width,
            'height': self.config.height,
            'area': self.config.area,
            'seed': self.config.seed,
            'highways': len(self.highways),
            'local_roads': len(self.local_roads),
            'total_roads': len(self.highways) + len(self.local_roads),
            'urban_blocks': len(self.urban_blocks),
            'buildings': len(self.buildings),
            'control_zones': len(self.control_zones),
            'high_ground': len(self.high_ground),
            'terrain_distribution': terrain_counts,
            'corrections_applied': sum(self.corrections.values()),
            'corrections_detail': self.corrections.copy()
        }

        return stats
