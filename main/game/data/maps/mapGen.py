"""
Advanced map generator with biomes, cities, highways, smoothing, elevation and
automatic validation.

This module implements MapGenerator which exposes:
    config = MapConfig(...)
    generator = MapGenerator(config)
    grid = generator.generate()
    stats = generator.get_statistics()
"""
import random
from collections import deque
from typing import List, Tuple, Dict, Set, Optional
from main.config import get_logger
from main.game.data.maps.config import MapConfig
from main.game.data.maps.tile import tile
from main.game.data.maps.terrain import plains, forest, urban, mountains, road, debris
from main.game.data.maps.utils import (
    create_grid, valid_pos, neighbors_4, neighbors_8, manhattan,
    direction_from_offset, bfs_reachable, find_components, astar_path,
    manhattan_path, find_cluster, generate_value_noise
)


# -------------------------
# Biome object (simple)
# -------------------------
class Biome:
    """Simple biome descriptor: id, display_name and a terrain weight map."""
    def __init__(self, id: str, display_name: str, terrain_weights: Dict, temperature: float, moisture: float):
        self.id = id
        self.display_name = display_name
        self.terrain_weights = terrain_weights
        self.temperature = temperature
        self.moisture = moisture


# Default set of biomes (data-driven; easy to replace via MapConfig.biome_definitions)
DEFAULT_BIOME_DEFS = [
    Biome('temperate_forest', 'Temperate Forest',
          terrain_weights={forest: 0.50, plains: 0.28, debris: 0.05, road: 0.03, urban: 0.04},
          temperature=0.3, moisture=0.75),
    Biome('grassland', 'Grassland',
          terrain_weights={plains: 0.65, forest: 0.12, road: 0.10, debris: 0.02, urban: 0.06},
          temperature=0.5, moisture=0.45),
    Biome('mountainous', 'Mountainous',
          terrain_weights={mountains: 0.45, plains: 0.20, forest: 0.15, debris: 0.07},
          temperature=-0.2, moisture=0.3),
    Biome('urban_sprawl', 'Urban Sprawl',
          terrain_weights={urban: 0.55, road: 0.20, debris: 0.10, plains: 0.10},
          temperature=0.6, moisture=0.2),
    Biome('mixed', 'Mixed',
          terrain_weights={plains: 0.30, forest: 0.20, urban: 0.20, mountains: 0.10, road: 0.05},
          temperature=0.0, moisture=0.5)
]


class MapGenerator:
    """
    Map generator class.

    - Uses MapConfig for all configurable parameters.
    - Uses deterministic RNG (config.seed) for reproducibility.
    - Public method `generate()` returns a validated, corrected grid of tiles.
    """
    def __init__(self, config: MapConfig = None):
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)

        # Deterministic RNG
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

        # Biome and city data
        self.biome_map: List[List[Optional[Biome]]] = []
        self.city_centers: List[Tuple[int, int]] = []
        self.roads: Set[Tuple[int, int]] = set()

        # Corrections / auto-fix counters
        self.corrections = {
            'connectivity_fixes': 0,
            'ramp_corrections': 0,
            'balance_adjustments': 0,
            'spawn_access_fixes': 0
        }

        # Load biome definitions from config override or defaults
        if self.config.biome_definitions:
            # Expect a list of dicts convertible to Biome(...) parameters
            self.available_biomes = [Biome(**d) for d in self.config.biome_definitions]
        else:
            self.available_biomes = DEFAULT_BIOME_DEFS

    # -------------------------
    # Entry point
    # -------------------------
    def generate(self) -> List[List[tile]]:
        """
        Run the full generation pipeline and return the final grid.

        The pipeline is documented inline in the method comments. Major stages:
          - initialization
          - biome + noise generation
          - city & highway seeding
          - biome-aware terrain assignment
          - elevation & ramp generation
          - smoothing (deep-copy)
          - road carving (highways + local roads)
          - diagonal-road cleanup (configurable)
          - connectivity / spawn access auto-correction
          - strategic element placement & auto-balancing
          - final validation
        """
        self.logger.info(f"Generating {self.config.width}×{self.config.height} map")

        # Phase 1: initialize fresh tile grid (all plains initially)
        self.grid = create_grid(self.config.width, self.config.height, plains)
        self.buildings = []
        self.control_zones = []
        self.high_ground = []
        self.chokepoints = []
        self.city_centers = []
        self.roads = set()
        self.corrections = {k: 0 for k in self.corrections}

        # Phase 2: biome map (Voronoi + cheap value-noise blending)
        self._generate_biome_map()

        # Phase 3: seed cities and highway connections (scales with map size)
        self._place_cities_and_highways()

        # Phase 4: assign terrain based on biome + city influence (urban bias)
        self._generate_terrain_biome_based()

        # Phase 5: elevation and ramp placement with validation
        self._generate_elevation()
        self._validate_and_fix_ramps()

        # Phase 6: smoothing pass(s) with deep-cloning to avoid aliasing
        if self.config.smoothing_enabled:
            self._smooth_terrain()

        # Phase 7: carve highways and road network
        self._carve_road_network()

        # Phase 8: optionally remove/discourage diagonal road artifacts
        if self.config.discourage_diagonal_roads:
            self._remove_diagonal_road_artifacts()

        # Phase 9: connectivity fixes and spawn access assurance
        self._ensure_connectivity()
        self._ensure_spawn_access()

        # Phase 10: strategic placement, balancing and final check
        self._place_strategic_elements()
        self._auto_balance_elements()
        self._final_validation()

        # Log summary and return grid
        total_corrections = sum(self.corrections.values())
        stats = self.get_statistics()
        self.logger.info(
            f"Map complete: {len(self.buildings)} buildings, {len(self.control_zones)} control zones, "
            f"balance={stats['overall_balance']:.2%}, corrections={total_corrections}"
        )
        return self.grid

    # -------------------------
    # Biome generation helpers
    # -------------------------
    def _generate_biome_map(self):
        """Generate a Voronoi-style assignment of biomes, blended by cheap value-noise."""
        w, h = self.config.width, self.config.height
        self.biome_map = [[None for _ in range(w)] for _ in range(h)]

        # pick number of biomes within configured min/max
        num_biomes = self.rng.randint(self.config.biome_count_min, self.config.biome_count_max)
        selected_biomes = self.rng.sample(self.available_biomes, min(num_biomes, len(self.available_biomes)))

        # place centers with a simple minimal spacing to avoid too many tiny regions
        centers = []
        min_distance = max(w, h) // 3
        for biome in selected_biomes:
            for attempt in range(self.config.seed_retry_attempts):
                x = self.rng.randint(0, w - 1)
                y = self.rng.randint(0, h - 1)
                if all(manhattan((x, y), c[1]) >= min_distance for c in centers):
                    centers.append((biome, (x, y)))
                    break
            else:
                centers.append((biome, (self.rng.randint(0, w - 1), self.rng.randint(0, h - 1))))

        # cheap value-noise map to break perfect Voronoi linear boundaries
        noise = generate_value_noise(w, h, self.rng, scale=self.config.biome_noise_scale, octaves=self.config.biome_noise_octaves)

        # assign each tile to the biome whose center minimizes (distance - noise_bias)
        for y in range(h):
            for x in range(w):
                best = None
                best_score = float('inf')
                for biome, center in centers:
                    dist = manhattan((x, y), center)
                    # apply noise bias so that tiles sometimes prefer other centers near boundary
                    bias = noise[y][x] * (w + h) * 0.08
                    score = dist - bias
                    if score < best_score:
                        best_score = score
                        best = biome
                self.biome_map[y][x] = best

        # small blending pass: where noise is high, allow neighbor biomes to replace current
        for y in range(h):
            for x in range(w):
                if self.rng.random() < noise[y][x] * 0.25:
                    neighbor_biomes = set()
                    for nx, ny, _ in neighbors_8(x, y, w, h):
                        nb = self.biome_map[ny][nx]
                        if nb is not None and nb != self.biome_map[y][x]:
                            neighbor_biomes.add(nb)
                    if neighbor_biomes:
                        self.biome_map[y][x] = self.rng.choice(list(neighbor_biomes))

        self._biome_centers = centers
        self.logger.debug(f"Biome map: placed {len(centers)} biome centers")

    # -------------------------
    # City & highway seeding
    # -------------------------
    def _place_cities_and_highways(self):
        """
        Determine city centers and simple highway connections.

        City count and city maximum size scale with map area.
        """
        area = self.config.area
        # number of cities = proportional to area (tunable via config)
        city_count = max(1, int(area * self.config.city_base_count))
        city_count = min(city_count, max(1, area // 256 + 1) * 3)

        candidate_scores = []
        for y in range(2, self.config.height - 2):
            for x in range(2, self.config.width - 2):
                b = self.biome_map[y][x]
                if b is None:
                    continue
                score = 0.0
                # prefer urban-friendly biomes
                if b.id == 'urban_sprawl':
                    score += 1.0
                elif b.id in ('grassland', 'mixed'):
                    score += 0.6
                elif b.id == 'temperate_forest':
                    score += 0.3
                # avoid mountain tiles
                if self.grid[y][x].terrain_type == mountains:
                    score -= 10.0
                candidate_scores.append((score + (self.rng.random() * 0.1), x, y))

        candidate_scores.sort(reverse=True)
        selected = []
        for score, x, y in candidate_scores:
            if len(selected) >= city_count:
                break
            spacing = max(6, min(self.config.width, self.config.height) // 8)
            if all(manhattan((x, y), c) >= spacing for c in selected):
                selected.append((x, y))
        self.city_centers = selected

        # Connect city centers + spawns via simple greedy MST-like connections (highways)
        hubs = list(self.city_centers) + [self.config.spawn_point_1, self.config.spawn_point_2]
        connections = []
        if hubs:
            remaining = set(hubs)
            connected = {hubs[0]}
            remaining.remove(hubs[0])
            while remaining:
                best_pair = None
                best_dist = float('inf')
                for a in connected:
                    for b in list(remaining):
                        d = manhattan(a, b)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (a, b)
                if best_pair:
                    connections.append(best_pair)
                    connected.add(best_pair[1])
                    remaining.remove(best_pair[1])

        self._highway_connections = connections

    # -------------------------
    # Terrain assignment (biome + city influence)
    # -------------------------
    def _generate_terrain_biome_based(self):
        """
        Convert biome map into actual terrain tiles. City centers strongly
        bias tiles near them toward 'urban'. Roads are favored near cities.
        """
        w, h = self.config.width, self.config.height

        # precompute city influence map (higher near each city center)
        city_influence = [[0.0] * w for _ in range(h)]
        for cx, cy in self.city_centers:
            max_radius = max(4, int(min(self.config.width, self.config.height) * 0.12))
            for y in range(max(0, cy - max_radius), min(h, cy + max_radius + 1)):
                for x in range(max(0, cx - max_radius), min(w, cx + max_radius + 1)):
                    d = manhattan((x, y), (cx, cy))
                    if d <= max_radius:
                        influence = (1.0 - (d / (max_radius + 0.01))) * self.config.city_growth_factor
                        city_influence[y][x] += influence

        # assign tiles by sampling each biome's terrain weight table and applying density modifiers
        for y in range(h):
            for x in range(w):
                biome = self.biome_map[y][x]
                if not biome:
                    continue
                terrain_types = list(biome.terrain_weights.keys())
                weights = list(biome.terrain_weights.values())
                adjusted = []
                for terrain_type, base in zip(terrain_types, weights):
                    modifier = self._get_density_modifier(terrain_type)
                    # boost urban and road probabilities near cities
                    if terrain_type == urban:
                        adjusted.append(base * modifier * (1.0 + city_influence[y][x] * 3.0))
                    elif terrain_type == road:
                        adjusted.append(base * modifier * (1.0 + city_influence[y][x] * self.config.road_urban_bias))
                    else:
                        adjusted.append(base * modifier)
                total = sum(adjusted)
                if total > 0:
                    normalized = [v / total for v in adjusted]
                    chosen = self.rng.choices(terrain_types, weights=normalized)[0]
                    self.grid[y][x].terrain_type = chosen

        # enforce city cores to be urban and grow them aggressively
        for cx, cy in self.city_centers:
            self._grow_city(cx, cy)

        # remove tiny non-plains speckles (configurable threshold)
        self._cluster_terrain()

    def _get_density_modifier(self, terrain_type) -> float:
        """Map terrain types to density-modifiers derived from config densities."""
        if terrain_type == forest:
            return self.config.forest_density * 4.0
        if terrain_type == urban:
            return self.config.urban_density * 6.0
        if terrain_type == mountains:
            return self.config.mountain_density * 8.0
        if terrain_type == road:
            return self.config.road_density * 12.0
        if terrain_type == debris:
            return self.config.debris_density * 15.0
        return 1.0

    def _grow_city(self, cx: int, cy: int):
        """
        Grow an urban blob from the city center.
        - Max city area constrained by config.city_max_fraction_of_area.
        - Probabilistic flood for organic boundaries.
        """
        max_area = int(self.config.area * self.config.city_max_fraction_of_area)
        frontier = deque([(cx, cy)])
        visited = set(frontier)
        placed = 0
        while frontier and placed < max_area:
            x, y = frontier.popleft()
            if not valid_pos(x, y, self.config.width, self.config.height):
                continue
            self.grid[y][x].terrain_type = urban
            placed += 1
            for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height):
                if (nx, ny) not in visited and self.rng.random() < 0.62:
                    visited.add((nx, ny))
                    frontier.append((nx, ny))

    def _cluster_terrain(self):
        """
        Remove tiny isolated clusters of the same non-plains terrain to avoid speckles.
        Size threshold is configurable in MapConfig.cluster_min_size.
        """
        visited = set()
        for y in range(self.config.height):
            for x in range(self.config.width):
                if (x, y) in visited:
                    continue
                cur = self.grid[y][x].terrain_type
                if cur == plains:
                    continue
                cluster = self._find_terrain_cluster(x, y, cur, visited)
                if len(cluster) < self.config.cluster_min_size:
                    for cx, cy in cluster:
                        self.grid[cy][cx].terrain_type = plains

    def _find_terrain_cluster(self, start_x: int, start_y: int, terrain_type, visited: Set) -> Set[Tuple[int, int]]:
        """Return an orthogonally connected cluster of tiles with the same terrain_type."""
        cluster = set()
        queue = deque([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            if self.grid[y][x].terrain_type != terrain_type:
                continue
            visited.add((x, y))
            cluster.add((x, y))
            for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                if (nx, ny) not in visited:
                    queue.append((nx, ny))
        return cluster

    # -------------------------
    # Smoothing (deep-clone tile objects)
    # -------------------------
    def _smooth_terrain(self):
        """Smooth terrain using neighborhood majority rules while avoiding aliasing by deep-copying tile state."""
        for _ in range(self.config.smoothing_passes):
            # create new tiles with copied state (avoid keeping references to old tile objects)
            new_grid: List[List[tile]] = [[None] * self.config.width for _ in range(self.config.height)]
            for y in range(self.config.height):
                for x in range(self.config.width):
                    old = self.grid[y][x]
                    new = tile(x=old.x, y=old.y, terrain_type=old.terrain_type, size=old.size,
                               occupied=old.occupied, is_building=old.is_building, elevation=old.elevation,
                               is_ramp=old.is_ramp, ramp_direction=getattr(old, 'ramp_direction', None))
                    if hasattr(old, 'ramp_elevation_to'):
                        new.ramp_elevation_to = getattr(old, 'ramp_elevation_to')
                    new_grid[y][x] = new

            # smoothing rule: if a non-plains tile has fewer than 3 same-type neighbors,
            # switch it to the most frequent neighbor type
            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    current = self.grid[y][x].terrain_type
                    if current == plains:
                        continue
                    same = 0
                    counts: Dict = {}
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            t = self.grid[y + dy][x + dx].terrain_type
                            if t == current:
                                same += 1
                            counts[t] = counts.get(t, 0) + 1
                    if same < 3 and counts:
                        new_grid[y][x].terrain_type = max(counts, key=counts.get)
            self.grid = new_grid

    # -------------------------
    # Elevation generation and ramp validation
    # -------------------------
    def _generate_elevation(self):
        """Create elevation levels by seeded growth respecting support & ramp constraints."""
        area = self.config.area
        seed_count = max(1, int(area * self.config.elevation_density * 0.002))
        avg_size = max(4, int(area * 0.01))

        # initial seeds & grow
        seeds = [self._interior_pos() for _ in range(seed_count)]
        self._grow_elevation_level(1, seeds, seed_count * avg_size, 0, 0)
        self._place_ramps_at_level(1)

        # higher elevation levels require neighboring support
        for level in range(2, self.config.max_elevation + 1):
            candidates = self._find_elevation_candidates(level)
            if not candidates:
                break
            self.rng.shuffle(candidates)
            seeds = candidates[:max(1, int(len(candidates) * 0.18))]
            target = max(0, int(len(candidates) * 0.20))
            self._grow_elevation_level(level, seeds, target,
                                      self.config.min_support_neighbors,
                                      self.config.min_ramp_neighbors)
            self._place_ramps_at_level(level)

    def _interior_pos(self) -> Tuple[int, int]:
        """Return a random position away from map edges (reduces edge plateauing)."""
        mx = max(2, self.config.width // 10)
        my = max(2, self.config.height // 10)
        return (self.rng.randint(mx, self.config.width - 1 - mx),
                self.rng.randint(my, self.config.height - 1 - my))

    def _find_elevation_candidates(self, level: int) -> List[Tuple[int, int]]:
        """Return tiles adjacent to tiles of elevation >= level-1 (candidates for raising)."""
        candidates = set()
        for y in range(self.config.height):
            for x in range(self.config.width):
                if self.grid[y][x].elevation >= level - 1:
                    for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                        if self.grid[ny][nx].elevation < level:
                            candidates.add((nx, ny))
        return list(candidates)

    def _grow_elevation_level(self, level: int, seeds: List[Tuple[int, int]],
                             target: int, min_support: int, min_ramps: int):
        """Probabilistic expansion from seed positions with support & ramp checks for higher levels."""
        placed = 0
        frontier = deque(seeds)
        visited = set(seeds)
        while frontier and placed < target:
            x, y = frontier.popleft()
            if not valid_pos(x, y, self.config.width, self.config.height):
                continue
            current = self.grid[y][x]
            if current.elevation >= level:
                continue
            can_promote = False
            if level == 1:
                can_promote = self.rng.random() < 0.92
            else:
                support = sum(1 for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height)
                              if self.grid[ny][nx].elevation >= level - 1)
                ramps = sum(1 for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height)
                            if getattr(self.grid[ny][nx], 'is_ramp', False) and
                            getattr(self.grid[ny][nx], 'ramp_elevation_to', 0) == level)
                can_promote = (support >= min_support and ramps >= min_ramps) or self.rng.random() < 0.04
            if can_promote:
                current.set_elevation(level)
                placed += 1
                prob = 0.62 if level == 1 else 0.46
                for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                    if (nx, ny) not in visited and self.rng.random() < prob:
                        frontier.append((nx, ny))
                        visited.add((nx, ny))

    def _place_ramps_at_level(self, level: int):
        """Place ramps between level-1 and level tiles with configured probability."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                cell = self.grid[y][x]
                for nx, ny, direction in neighbors_4(x, y, self.config.width, self.config.height):
                    neighbor = self.grid[ny][nx]
                    if neighbor.elevation == level and cell.elevation == level - 1:
                        if not getattr(cell, 'is_ramp', False) and self.rng.random() < self.config.ramp_placement_probability:
                            cell.set_ramp(True, direction)
                            cell.ramp_elevation_to = level
                        elif getattr(cell, 'is_ramp', False):
                            existing = getattr(cell, 'ramp_elevation_to', None)
                            if existing is None or level > existing:
                                cell.ramp_elevation_to = level

    def _validate_and_fix_ramps(self):
        """Auto-fix inconsistent ramps (remove ramps that lead nowhere; add ramps where large diffs exist)."""
        corrections = 0
        for y in range(self.config.height):
            for x in range(self.config.width):
                cell = self.grid[y][x]
                if getattr(cell, 'is_ramp', False):
                    # check if ramp leads to any higher neighbor; if not, remove it
                    has_higher = False
                    for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                        if self.grid[ny][nx].elevation > cell.elevation:
                            has_higher = True
                            break
                    if not has_higher:
                        cell.set_ramp(False, None)
                        if hasattr(cell, 'ramp_elevation_to'):
                            try:
                                delattr(cell, 'ramp_elevation_to')
                            except Exception:
                                try:
                                    del cell.ramp_elevation_to
                                except Exception:
                                    pass
                        corrections += 1

                # fix neighbors with large elevation difference lacking ramps (add ramp to lower tile)
                for nx, ny, direction in neighbors_4(x, y, self.config.width, self.config.height):
                    neighbor = self.grid[ny][nx]
                    diff = abs(cell.elevation - neighbor.elevation)
                    if diff > 1 and not getattr(cell, 'is_ramp', False) and not getattr(neighbor, 'is_ramp', False):
                        if cell.elevation < neighbor.elevation:
                            cell.set_ramp(True, direction)
                            cell.ramp_elevation_to = neighbor.elevation
                            corrections += 1
                        elif neighbor.elevation < cell.elevation:
                            reverse_dir = direction_from_offset(x - nx, y - ny)
                            if reverse_dir:
                                neighbor.set_ramp(True, reverse_dir)
                                neighbor.ramp_elevation_to = cell.elevation
                                corrections += 1
        if corrections > 0:
            self.logger.debug(f"Auto-corrected {corrections} ramp issues")
            self.corrections['ramp_corrections'] = corrections

    # -------------------------
    # Roads & highways generation
    # -------------------------
    def _carve_road_network(self):
        """
        Carve highways first (connecting city hubs and spawns) then local roads that
        connect city centers to control zones/spawns. Tracks roads in self.roads set.
        """
        # highways
        for a, b in getattr(self, '_highway_connections', []):
            path = astar_path(self.grid, a, b, self._highway_cost)
            if not path:
                path = manhattan_path(a, b, self.rng)
            self._carve_path(path, road_type=road, width=self.config.highway_width)

        # local roads: connect city centers to nearest control/spawn hub
        nodes = list(self.city_centers) + [self.config.spawn_point_1, self.config.spawn_point_2]
        for c in self.city_centers:
            target = self._nearest_control_or_spawn(c)
            if target:
                path = astar_path(self.grid, c, target, self._terrain_cost)
                if not path:
                    path = manhattan_path(c, target, self.rng)
                self._carve_path(path, road_type=road, width=1)

        # record road tiles for quick checks elsewhere
        self.roads = {(x, y) for y in range(self.config.height) for x in range(self.config.width) if self.grid[y][x].terrain_type == road}

    def _carve_path(self, path: List[Tuple[int, int]], road_type, width: int = 1):
        """
        Carve a path (list of tile coords) into a road. We carve width by creating
        a small square around each path tile (width parameter) and converting some
        plains to road. Mountain tiles are flattened to plains first.
        """
        if not path:
            return
        for (x, y) in path:
            for dy in range(-width + 1, width):
                for dx in range(-width + 1, width):
                    nx, ny = x + dx, y + dy
                    if not valid_pos(nx, ny, self.config.width, self.config.height):
                        continue
                    cell = self.grid[ny][nx]
                    # flatten mountains under roads to keep highways passable
                    if cell.terrain_type == mountains:
                        cell.terrain_type = plains
                    # carving rules:
                    if cell.terrain_type in (plains, road):
                        cell.terrain_type = road
                    # if urban, keep it urban but road may still be recorded
        # update roads set after carving path
        self.roads.update({(x, y) for x, y in path if valid_pos(x, y, self.config.width, self.config.height)})

    def _highway_cost(self, grid, x, y):
        """Cost function specialized for highway routing (prefers plains / low-cost tiles)."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return 999.0
        t = grid[y][x].terrain_type
        if t == plains:
            return 0.6
        if t == road:
            return 0.5
        if t == urban:
            return 0.7
        if t == mountains:
            return 8.0
        return 1.5

    def _terrain_cost(self, grid, x, y):
        """Default terrain cost used for pathfinding when carving local roads."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return 999.0
        terrain = grid[y][x].terrain_type
        if terrain == plains:
            return 1.0
        elif terrain == road:
            return 0.5
        elif terrain == mountains:
            return 10.0
        elif terrain == urban:
            return 0.9
        else:
            return 2.0

    def _nearest_control_or_spawn(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Return nearest control zone or spawn point from pos (used to tie road endpoints)."""
        best = None
        best_d = float('inf')
        candidates = list(self.control_zones) + [self.config.spawn_point_1, self.config.spawn_point_2]
        if not candidates:
            return None
        for c in candidates:
            d = manhattan(pos, c)
            if d < best_d:
                best_d = d
                best = c
        return best

    # -------------------------
    # Diagonal road artifact removal (new)
    # -------------------------
    def _remove_diagonal_road_artifacts(self):
        """
        Post-process to discourage 'diagonal-only' or parallel offset roads that would be
        unrealistic in real life (two parallel highways offset by one tile diagonally).

        Strategy:
            - Find road tiles that have no orthogonal road neighbors but have diagonal road neighbors.
            - Remove the tile (set to plains) with probability = diagonal_road_penalty, unless
              it is inside a city core and keep_city_diagonal_roads=True.
            - Optionally, repair by inserting an orthogonal connecting road instead of the diagonal tile.
        This behavior is configurable via MapConfig:
            - discourage_diagonal_roads (bool)
            - diagonal_road_penalty (float 0..1)
            - keep_city_diagonal_roads (bool)
        """
        to_remove = []
        w, h = self.config.width, self.config.height

        for y in range(h):
            for x in range(w):
                if self.grid[y][x].terrain_type != road:
                    continue
                # count orthogonal and diagonal neighboring roads
                orth_road = 0
                diag_road = 0
                orth_neighbors = []
                diag_neighbors = []
                for nx, ny, direction in neighbors_8(x, y, w, h):
                    if self.grid[ny][nx].terrain_type == road:
                        # determine if orthogonal vs diagonal by comparing manhattan distance
                        if abs(nx - x) + abs(ny - y) == 1:
                            orth_road += 1
                            orth_neighbors.append((nx, ny))
                        else:
                            diag_road += 1
                            diag_neighbors.append((nx, ny))

                # If this road tile is ONLY diagonally connected (no orth neighbors), it's suspicious
                if orth_road == 0 and diag_road > 0:
                    # If tile is inside a city core we may optionally keep it
                    inside_city = any(manhattan((x, y), c) <= max(3, min(w, h) // 10) for c in self.city_centers)
                    if inside_city and self.config.keep_city_diagonal_roads:
                        continue
                    # mark for removal with probability penalty
                    if self.rng.random() < self.config.diagonal_road_penalty:
                        to_remove.append((x, y))

        # Remove or repair marked tiles
        for x, y in to_remove:
            # attempt conservative repair: try to add an orthogonal connector between adjacent diagonal roads
            repaired = False
            # look for two diagonal road neighbors that could be connected by an orthogonal tile
            diag_neighs = [(nx, ny) for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height) if self.grid[ny][nx].terrain_type == road and abs(nx - x) + abs(ny - y) == 2]
            for (ax, ay) in diag_neighs:
                # find intermediate orthogonal positions between (x,y) and (ax,ay)
                mid1 = (ax, y)   # same x as diagonal neighbor, same y as original
                mid2 = (x, ay)
                # If either mid is inside bounds and not mountain, create orthogonal road there instead of the diagonal
                for mx, my in (mid1, mid2):
                    if not valid_pos(mx, my, self.config.width, self.config.height):
                        continue
                    if self.grid[my][mx].terrain_type != mountains:
                        # create small orthogonal connector
                        self.grid[my][mx].terrain_type = road
                        self.roads.add((mx, my))
                        repaired = True
                        break
                if repaired:
                    break
            if not repaired:
                # fallback: remove diagonal tile (turn it back to plains)
                self.grid[y][x].terrain_type = plains
                if (x, y) in self.roads:
                    self.roads.remove((x, y))

    # -------------------------
    # Connectivity & spawn access (unchanged core logic but increased robustness)
    # -------------------------
    def _ensure_connectivity(self):
        """
        Ensure all passable tiles are connected by carving corridors between disconnected components.
        Limits number of attempts to avoid infinite loops.
        """
        max_fixes = 8
        attempts = 0
        while attempts < max_fixes:
            main = bfs_reachable(self.grid, self.config.spawn_point_1, self._is_passable, self._can_traverse)
            components = find_components(self.grid, self._is_passable, self._can_traverse, main)
            if not components:
                break
            self.logger.debug(f"Found {len(components)} disconnected regions (attempt {attempts})")
            for component in components:
                # pick near pair (component tile, main tile) by sampling
                best_dist = float('inf')
                best_pair = None
                comp_sample = list(component)[:80]
                main_sample = list(main)[:80]
                for c_pos in comp_sample:
                    for m_pos in main_sample:
                        dist = manhattan(c_pos, m_pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (c_pos, m_pos)
                if best_pair:
                    self._create_corridor(best_pair[0], best_pair[1])
                    self.corrections['connectivity_fixes'] += 1
            attempts += 1
        if attempts >= max_fixes:
            self.logger.warning(f"Connectivity fixes exhausted ({max_fixes} attempts)")

    def _ensure_spawn_access(self):
        """Ensure spawn tiles themselves are passable and reachable from each other."""
        corrections = 0
        for spawn in [self.config.spawn_point_1, self.config.spawn_point_2]:
            sx, sy = spawn
            if self.grid[sy][sx].terrain_type == mountains:
                self.grid[sy][sx].terrain_type = plains
                corrections += 1
        reachable = bfs_reachable(self.grid, self.config.spawn_point_1, self._is_passable, self._can_traverse)
        if self.config.spawn_point_2 not in reachable:
            self._create_corridor(self.config.spawn_point_1, self.config.spawn_point_2)
            corrections += 1
        if corrections > 0:
            self.logger.debug(f"Auto-fixed {corrections} spawn access issues")
            self.corrections['spawn_access_fixes'] = corrections

    def _is_passable(self, grid, x: int, y: int) -> bool:
        """Tile-level passable test (non-mountain tiles are considered passable)."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return False
        return grid[y][x].terrain_type != mountains

    def _can_traverse(self, grid, x1: int, y1: int, x2: int, y2: int) -> bool:
        """
        Movement rule for BFS: tiles must be passable and elevation difference must be <= 1,
        and ramps must be present to cross equal-to-1 elevation differences.
        """
        if not self._is_passable(grid, x1, y1) or not self._is_passable(grid, x2, y2):
            return False
        t1 = grid[y1][x1]
        t2 = grid[y2][x2]
        diff = abs(t2.elevation - t1.elevation)
        if diff == 0:
            return True
        if diff > 1:
            return False
        # diff == 1: require at least one tile to be a ramp
        return getattr(t1, 'is_ramp', False) or getattr(t2, 'is_ramp', False)

    def _create_corridor(self, start: Tuple[int, int], end: Tuple[int, int]):
        """
        Create a corridor / road between start & end (tries A* then Manhattan fallback).
        Carving rules: mountains->plains, plains->road. Also add ramps where elevation changes.
        """
        path = astar_path(self.grid, start, end, self._terrain_cost)
        if not path:
            path = manhattan_path(start, end, self.rng)
        for x, y in path:
            cell = self.grid[y][x]
            if cell.terrain_type == mountains:
                cell.terrain_type = plains
            if cell.terrain_type == plains:
                cell.terrain_type = road
        # add ramps for elevation differences
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            t1 = self.grid[y1][x1]
            t2 = self.grid[y2][x2]
            if t2.elevation == t1.elevation + 1 and not getattr(t1, 'is_ramp', False):
                direction = direction_from_offset(x2 - x1, y2 - y1)
                if direction:
                    t1.set_ramp(True, direction)
                    t1.ramp_elevation_to = t2.elevation
            elif t1.elevation == t2.elevation + 1 and not getattr(t2, 'is_ramp', False):
                direction = direction_from_offset(x1 - x2, y1 - y2)
                if direction:
                    t2.set_ramp(True, direction)
                    t2.ramp_elevation_to = t1.elevation

    # -------------------------
    # Strategic element placement and balancing
    # -------------------------
    def _place_strategic_elements(self):
        """Top-level placement of control zones, buildings, and feature identification."""
        self._place_control_zones()
        target = int(self.config.area * self.config.building_density)
        self._place_buildings(target)
        self._identify_high_ground()
        self._identify_chokepoints()

    def _place_control_zones(self):
        """
        Place control zones balanced between spawn points and biased toward
        points of contact: road intersections, biome edges and city boundaries.
        """
        zones = []
        cx, cy = self.config.width // 2, self.config.height // 2
        zones.append((cx, cy))
        remaining = self.config.control_zone_count - 1
        candidates = []
        for y in range(2, self.config.height - 2):
            for x in range(2, self.config.width - 2):
                if (x, y) == (cx, cy):
                    continue
                if self.grid[y][x].terrain_type == mountains:
                    continue
                d1 = manhattan(self.config.spawn_point_1, (x, y))
                d2 = manhattan(self.config.spawn_point_2, (x, y))
                if d1 < 4 or d2 < 4:
                    continue
                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                # points of contact:
                road_near = any(((nx, ny) in self.roads) for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height))
                biome_contact = False
                b = self.biome_map[y][x]
                for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height):
                    if self.biome_map[ny][nx] != b:
                        biome_contact = True
                        break
                score = balance + (0.45 if road_near else 0.0) + (0.35 if biome_contact else 0.0)
                candidates.append((score, x, y))
        candidates.sort(reverse=True)
        for score, x, y in candidates:
            if len(zones) >= self.config.control_zone_count:
                break
            if all(manhattan((x, y), z) >= 6 for z in zones):
                zones.append((x, y))
        self.control_zones = zones

    def _place_buildings(self, target: int):
        """
        Place buildings preferentially:
            - near roads,
            - inside city cores,
            - adjacent to control zones and spawn neighborhoods,
            - while respecting min_building_spacing.
        """
        buildings: List[Tuple[int, int]] = []
        candidates = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                t = self.grid[y][x].terrain_type
                if t in [mountains, forest, debris]:
                    continue
                cell = self.grid[y][x]
                if getattr(cell, 'is_ramp', False) or getattr(cell, 'is_building', False):
                    continue
                # scoring heuristics
                road_near = any(((nx, ny) in self.roads) for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height))
                city_near = any(manhattan((x, y), c) <= max(3, min(self.config.width, self.config.height) // 10) for c in self.city_centers)
                cz_near = any(manhattan((x, y), cz) <= 4 for cz in self.control_zones)
                d1 = manhattan(self.config.spawn_point_1, (x, y))
                d2 = manhattan(self.config.spawn_point_2, (x, y))
                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                score = balance + (0.8 if road_near else 0.0) + (0.95 if city_near else 0.0) + (0.6 if cz_near else 0.0)
                candidates.append((score, x, y))
        candidates.sort(reverse=True)
        for score, x, y in candidates:
            if len(buildings) >= target:
                break
            if self._check_building_spacing((x, y), buildings):
                self.grid[y][x].set_building(True)
                buildings.append((x, y))
        self.buildings = buildings

    def _find_building_near(self, cx: int, cy: int, existing: List, radius: int) -> Optional[Tuple[int, int]]:
        """Find a valid tile for a building near (cx,cy) with bounds checking."""
        for r in range(radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if not valid_pos(x, y, self.config.width, self.config.height):
                        continue
                    terrain = self.grid[y][x].terrain_type
                    if terrain in [mountains, forest, debris]:
                        continue
                    if getattr(self.grid[y][x], 'is_ramp', False) or getattr(self.grid[y][x], 'is_building', False):
                        continue
                    if self._check_building_spacing((x, y), existing):
                        return (x, y)
        return None

    def _check_building_spacing(self, pos: Tuple[int, int], buildings: List) -> bool:
        """Return True if pos is at least min_building_spacing from all existing buildings."""
        return all(manhattan(pos, b) >= self.config.min_building_spacing for b in buildings)

    def _identify_high_ground(self):
        """Find high-elevation clusters and record their centroid positions."""
        visited = set()
        clusters = []
        for y in range(self.config.height):
            for x in range(self.config.width):
                if (x, y) in visited or self.grid[y][x].elevation < 2:
                    continue
                cluster = find_cluster(self.grid, x, y, lambda t: t.elevation >= 2)
                if len(cluster) >= 6:
                    cx = sum(px for px, py in cluster) // len(cluster)
                    cy = sum(py for px, py in cluster) // len(cluster)
                    clusters.append((cx, cy))
                visited.update(cluster)
        self.high_ground = clusters

    def _identify_chokepoints(self):
        """
        Identify tiles that naturally act as chokepoints: few passable neighbors
        in the 8-neighborhood (2..4 passable neighbors).
        """
        chokepoints = []
        for y in range(2, self.config.height - 2):
            for x in range(2, self.config.width - 2):
                if self.grid[y][x].terrain_type == mountains:
                    continue
                passable = sum(1 for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height)
                             if self.grid[ny][nx].terrain_type != mountains)
                if 2 <= passable <= 4:
                    chokepoints.append((x, y))
        self.chokepoints = chokepoints

    def _auto_balance_elements(self):
        """Auto-remove or relocate buildings if building placement is heavily imbalanced."""
        metrics = self._calculate_balance()
        if metrics['overall_balance'] < 0.85 and len(self.buildings) > 0:
            self.logger.debug(f"Balance suboptimal ({metrics['overall_balance']:.2%}), auto-adjusting")
            while metrics['overall_balance'] < 0.85 and len(self.buildings) > 3:
                worst_idx = self._find_worst_building()
                if worst_idx is not None:
                    bx, by = self.buildings.pop(worst_idx)
                    self.grid[by][bx].set_building(False)
                    self.corrections['balance_adjustments'] += 1
                    metrics = self._calculate_balance()
                else:
                    break
            if self.corrections['balance_adjustments'] > 0:
                self.logger.debug(f"Removed {self.corrections['balance_adjustments']} buildings to improve balance")

    def _find_worst_building(self) -> Optional[int]:
        """Return index of building that most increases imbalance (largest d1-d2)."""
        if not self.buildings:
            return None
        worst_idx = None
        worst_imbalance = 0.0
        for i, (bx, by) in enumerate(self.buildings):
            d1 = manhattan(self.config.spawn_point_1, (bx, by))
            d2 = manhattan(self.config.spawn_point_2, (bx, by))
            imbalance = abs(d1 - d2)
            if imbalance > worst_imbalance:
                worst_imbalance = imbalance
                worst_idx = i
        return worst_idx

    def _calculate_balance(self) -> Dict[str, float]:
        """Compute simple balance metrics for buildings and control zones relative to spawn points."""
        metrics = {}
        if self.buildings:
            p1_dists = [manhattan(self.config.spawn_point_1, b) for b in self.buildings]
            p2_dists = [manhattan(self.config.spawn_point_2, b) for b in self.buildings]
            avg1 = sum(p1_dists) / len(p1_dists)
            avg2 = sum(p2_dists) / len(p2_dists)
            metrics['building_balance'] = min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1.0
        else:
            metrics['building_balance'] = 1.0
        if self.control_zones:
            cz1 = [manhattan(self.config.spawn_point_1, cz) for cz in self.control_zones]
            cz2 = [manhattan(self.config.spawn_point_2, cz) for cz in self.control_zones]
            avg1 = sum(cz1) / len(cz1)
            avg2 = sum(cz2) / len(cz2)
            metrics['control_zone_balance'] = min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1.0
        else:
            metrics['control_zone_balance'] = 1.0
        metrics['overall_balance'] = min(metrics['building_balance'], metrics['control_zone_balance'])
        metrics['balance_verified'] = metrics['overall_balance'] >= 0.85
        return metrics

    def _final_validation(self):
        """Final validation step logs any remaining unreachable passable tiles."""
        reachable = bfs_reachable(self.grid, self.config.spawn_point_1, self._is_passable, self._can_traverse)
        all_passable = sum(1 for y in range(self.config.height) for x in range(self.config.width) if self._is_passable(self.grid, x, y))
        if len(reachable) != all_passable:
            self.logger.warning(f"Final check: {all_passable - len(reachable)} tiles still unreachable")

    # -------------------------
    # Statistics helpers
    # -------------------------
    def get_statistics(self) -> Dict:
        """Return a dictionary of useful summary statistics about the generated map."""
        stats = {
            'width': self.config.width,
            'height': self.config.height,
            'area': self.config.area,
            'seed': self.config.seed,
        }
        counts = {'plains': 0, 'forest': 0, 'urban': 0, 'mountains': 0, 'road': 0, 'debris': 0, 'ramps': 0, 'elevated': 0}
        for row in self.grid:
            for t in row:
                name = getattr(t.terrain_type, 'name', type(t.terrain_type).__name__).lower()
                if name in counts:
                    counts[name] += 1
                if getattr(t, 'is_ramp', False):
                    counts['ramps'] += 1
                if t.elevation > 0:
                    counts['elevated'] += 1
        stats.update(counts)
        stats['buildings'] = len(self.buildings)
        stats['control_zones'] = len(self.control_zones)
        stats['high_ground_clusters'] = len(self.high_ground)
        stats['chokepoints'] = len(self.chokepoints)
        stats.update(self._calculate_balance())
        stats['corrections_applied'] = sum(self.corrections.values())
        stats['correction_details'] = self.corrections.copy()
        return stats

    # -------------------------
    # Back-compat convenience
    # -------------------------
    def get_terrain_statistics(self) -> Dict:
        """Legacy name preserved for compatibility."""
        return self.get_statistics()
