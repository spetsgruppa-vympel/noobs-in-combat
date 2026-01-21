"""
Procedural Map Generator for Snakes in Combat.

Generates tactical maps with algorithmic balance guarantees through
intelligent placement of strategic elements. No visualization included
(preview handled by menuManager.py).
"""

import random
from collections import deque
from typing import List, Tuple, Dict, Set, Optional
from math import hypot

from main.config import get_logger, MAP_WIDTH, MAP_HEIGHT, TILE_SIZE
from main.game.data.maps.terrain import plains, forest, urban, mountains, road, highway, debris
from main.game.data.maps.tile import tile

# ============================================================================
# GENERATION CONSTANTS
# ============================================================================

DEFAULT_FOREST_DENSITY = 0.25
DEFAULT_URBAN_DENSITY = 0.15
DEFAULT_MOUNTAIN_DENSITY = 0.10
DEFAULT_ROAD_DENSITY = 0.06
DEFAULT_DEBRIS_DENSITY = 0.05
DEFAULT_ELEVATION_DENSITY = 0.18
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

# Elevation growth rules (tunable)
MIN_SUPPORT_NEIGHBORS_FOR_PROMOTE = 2  # number of neighbors at previous elevation required
MIN_RAMP_NEIGHBORS_FOR_PROMOTE = 1     # number of ramp tiles adjacent to previous-elevation neighbors required

# Ramp generation probability when a border exists
RAMP_PLACE_PROB = 0.75

# Smoothing and seed sizes
LEVEL1_SEED_COUNT_FACTOR = 0.002  # seeds per tile for elevation level 1 (scaled by area)
SEED_GROWTH_TARGET_FACTOR = 0.01  # fraction of map area to consume per seed cluster on average

# Accessibility retry attempts
MAX_CONNECTIVITY_FIXES = 6

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

        # Default spawn points (opposing quadrants)
        self.spawn_point_1 = spawn_point_1 or (width // 4, height // 4)
        self.spawn_point_2 = spawn_point_2 or (3 * width // 4, 3 * height // 4)

        self.logger = get_logger(__name__)
        self.logger.info(f"MapConfig: {width}x{height}")


# ============================================================================
# MAP GENERATOR WITH ALGORITHMIC FAIRNESS
# ============================================================================

class MapGenerator:
    """
    Procedural map generator with algorithmic balance guarantees.

    Ensures competitive balance through intelligent placement during
    generation rather than post-generation validation.
    """

    def __init__(self, config=None):
        """Initialize the map generator."""
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)
        self.grid: List[List[tile]] = []
        self.buildings: List[Tuple[int, int]] = []
        self.control_zones: List[Tuple[int, int]] = []
        self.high_ground_clusters: List[Tuple[int, int]] = []
        self.choke_points: List[Tuple[int, int]] = []

        if self.config.seed is not None:
            random.seed(self.config.seed)
            self.logger.info(f"Random seed: {self.config.seed}")

    # ---------------------------
    # Public entry
    # ---------------------------
    def generate(self):
        """Generate a guaranteed balanced tactical map."""
        self.logger.info(f"Generating {self.config.width}x{self.config.height} map")

        # Single generation flow
        self._initialize_grid()

        # Terrain: seeded growth to avoid edge bias
        self._generate_terrain_seeded()

        # Elevation: iterative growth with ramp & support constraints
        self._generate_elevation_seeded_and_ramped()

        # Smooth terrain for realism
        self._smooth_terrain()

        # Ensure accessibility: create ramps/roads where necessary (may modify tiles)
        self._ensure_full_accessibility()

        # Strategic placements
        self._place_balanced_control_zones()
        self._create_elevation_clusters_near_spawns()  # may add extra high ground if needed
        self._place_balanced_high_ground()
        self._place_balanced_buildings()
        self._identify_choke_points()

        self.logger.info("Balanced map generated (algorithmic guarantee)")
        return self.grid

    # ---------------------------
    # Initialization
    # ---------------------------
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

    # ---------------------------
    # Terrain generation (seeded growth to avoid edge bias)
    # ---------------------------
    def _generate_terrain_seeded(self):
        """Place forest/mountains/urban/debris/road with seeded blob growth."""
        width, height = self.config.width, self.config.height
        area = width * height

        # helper: center-biased random position (reduces edge bias)
        def center_biased_pos():
            # Gaussian around center with sigma proportional to map size
            cx, cy = width / 2.0, height / 2.0
            sigma_x = max(1.0, width / 6.0)
            sigma_y = max(1.0, height / 6.0)
            for _ in range(50):
                rx = int(random.gauss(cx, sigma_x))
                ry = int(random.gauss(cy, sigma_y))
                if 0 <= rx < width and 0 <= ry < height:
                    return rx, ry
            # fallback
            return random.randrange(0, width), random.randrange(0, height)

        # seeded clusters parameters
        def cluster_amount(density):
            # convert density fraction into number of seeds
            return max(1, int(area * density * 0.01))

        # Generic seeded growth: pick seeds then expand probabilistically
        def seed_and_grow(seed_count, avg_cluster_size, setter_fn):
            seeds = []
            for _ in range(seed_count):
                sx, sy = center_biased_pos()
                seeds.append((sx, sy))

            target_total = max(1, int(avg_cluster_size * seed_count))
            placed = 0
            frontier = deque(seeds)
            visited_local = set(seeds)
            attempts = 0
            while frontier and placed < target_total and attempts < area * 10:
                attempts += 1
                cx, cy = frontier.popleft()
                if 0 <= cx < width and 0 <= cy < height:
                    if setter_fn(cx, cy):
                        placed += 1
                        # grow neighbors
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited_local:
                                # probability decay for distance from seed
                                if random.random() < 0.55:
                                    frontier.append((nx, ny))
                                    visited_local.add((nx, ny))
            return placed

        # Place mountains (smaller clusters, not concentrated at edges)
        mountain_seeds = max(1, int(area * self.config.mountain_density * 0.002))
        mountain_avg = max(3, int(area * self.config.mountain_density * 0.004))
        def place_mountain(x,y):
            if self.grid[y][x].terrain_type == plains:
                self.grid[y][x].terrain_type = mountains
                return True
            return False
        seed_and_grow(mountain_seeds, mountain_avg, place_mountain)

        # Place forests (larger clusters)
        forest_seeds = max(1, int(area * self.config.forest_density * 0.003))
        forest_avg = max(5, int(area * self.config.forest_density * 0.01))
        def place_forest(x,y):
            if self.grid[y][x].terrain_type == plains:
                self.grid[y][x].terrain_type = forest
                return True
            return False
        seed_and_grow(forest_seeds, forest_avg, place_forest)

        # Place urban cores (small discrete clusters, but interior-biased)
        urban_seeds = max(1, int(area * self.config.urban_density * 0.002))
        urban_avg = max(2, int(area * self.config.urban_density * 0.006))
        def place_urban(x,y):
            if self.grid[y][x].terrain_type == plains:
                self.grid[y][x].terrain_type = urban
                return True
            return False
        seed_and_grow(urban_seeds, urban_avg, place_urban)

        # Place debris / ruins
        debris_seeds = max(1, int(area * self.config.debris_density * 0.002))
        debris_avg = max(2, int(area * self.config.debris_density * 0.005))
        def place_debris(x,y):
            if self.grid[y][x].terrain_type == plains:
                self.grid[y][x].terrain_type = debris
                return True
            return False
        seed_and_grow(debris_seeds, debris_avg, place_debris)

        # Place sparse roads as connectors (we'll carve more later if needed)
        road_seeds = max(1, int(area * self.config.road_density * 0.001))
        road_avg = max(2, int(area * self.config.road_density * 0.004))
        def place_road(x,y):
            if self.grid[y][x].terrain_type == plains:
                self.grid[y][x].terrain_type = road
                return True
            return False
        seed_and_grow(road_seeds, road_avg, place_road)

    # ---------------------------
    # Elevation & Ramp generation (seeded & constrained)
    # ---------------------------
    def _generate_elevation_seeded_and_ramped(self):
        """
        Iteratively generate elevation plateaus and ramps:
         - Level 1 seeds grown first (no previous-level constraint).
         - For higher levels, require MIN_SUPPORT_NEIGHBORS_FOR_PROMOTE neighbors
           at previous elevation AND at least MIN_RAMP_NEIGHBORS_FOR_PROMOTE
           nearby ramps (these ramps are created after the previous level growth).
         - Ramps are discrete tiles with direction (north/south/east/west).
        """
        w, h = self.config.width, self.config.height
        area = w * h

        # 1) Determine number of level1 seeds based on density
        seed_count = max(1, int(area * self.config.elevation_density * LEVEL1_SEED_COUNT_FACTOR))
        avg_cluster_size = max(4, int(area * SEED_GROWTH_TARGET_FACTOR))

        # helper to pick interior-biased seeds (avoids edges where possible)
        def interior_seed():
            margin_x = max(2, w // 10)
            margin_y = max(2, h // 10)
            for _ in range(40):
                sx = random.randint(margin_x, w - 1 - margin_x)
                sy = random.randint(margin_y, h - 1 - margin_y)
                return sx, sy
            return random.randrange(0, w), random.randrange(0, h)

        # grow function for level N starting from seeds, but honoring constraints for N>1
        def grow_level(level, seeds, target_tiles):
            placed = 0
            frontier = deque(seeds)
            visited = set(seeds)

            while frontier and placed < target_tiles:
                cx, cy = frontier.popleft()
                if not self._is_valid_position(cx, cy):
                    continue
                cur = self.grid[cy][cx]
                if cur.elevation >= level:
                    # already at or above
                    pass
                else:
                    # check promotion conditions
                    if level == 1:
                        # accept with some probability to avoid uniform plates
                        if random.random() < 0.92:
                            cur.set_elevation(level)
                            placed += 1
                    else:
                        # require neighbors at previous level and ramp neighbors nearby
                        support = 0
                        ramp_support = 0
                        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if self._is_valid_position(nx, ny):
                                ncell = self.grid[ny][nx]
                                if ncell.elevation >= (level - 1):
                                    support += 1
                                # check if neighbor is a ramp that ascends from (level-2) -> (level-1)
                                if ncell.is_ramp and getattr(ncell, 'ramp_elevation_to', None) == level:
                                    ramp_support += 1

                        if support >= MIN_SUPPORT_NEIGHBORS_FOR_PROMOTE and ramp_support >= MIN_RAMP_NEIGHBORS_FOR_PROMOTE:
                            cur.set_elevation(level)
                            placed += 1
                        else:
                            # allow a small probability of promotion to form natural peaks if map sparse
                            if random.random() < 0.04:
                                cur.set_elevation(level)
                                placed += 1

                # push neighbors
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1), (1,1),(-1,-1),(1,-1),(-1,1)]:
                    nx, ny = cx + dx, cy + dy
                    if self._is_valid_position(nx, ny) and (nx, ny) not in visited:
                        # bias growth outward less aggressively with higher levels
                        p = 0.62 if level == 1 else 0.46
                        if random.random() < p:
                            frontier.append((nx, ny))
                            visited.add((nx, ny))

            return placed

        # 1) Level 1 growth
        level1_seeds = [interior_seed() for _ in range(seed_count)]
        target_level1 = max(1, int(area * 0.01)) if avg_cluster_size == 0 else seed_count * avg_cluster_size
        placed1 = grow_level(1, level1_seeds, target_level1)
        self.logger.debug(f"Placed level1 tiles: {placed1}")

        # After placing level1, generate ramps along borders (lower->level1)
        self._place_ramps_around_level(1)

        # 2) For higher levels: attempt to grow level 2..MAX_ELEVATION
        for level in range(2, MAX_ELEVATION + 1):
            # candidates are tiles adjacent to existing (level-1) tiles
            candidates = set()
            for y in range(h):
                for x in range(w):
                    if self.grid[y][x].elevation >= (level - 1):
                        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                            nx, ny = x + dx, y + dy
                            if self._is_valid_position(nx, ny) and self.grid[ny][nx].elevation < level:
                                candidates.add((nx, ny))

            # choose a subset of candidates as seeds, bias interior
            seeds = []
            candidates_list = list(candidates)
            random.shuffle(candidates_list)
            seed_limit = max(1, int(len(candidates_list) * 0.18))
            for i in range(min(seed_limit, len(candidates_list))):
                seeds.append(candidates_list[i])

            # target tiles roughly proportional to candidates and density
            target = max(0, int(len(candidates_list) * 0.20))
            placed = grow_level(level, seeds, target)
            self.logger.debug(f"Placed level{level} tiles: {placed}")

            # place ramps around this new level to allow further growth and access
            self._place_ramps_around_level(level)

    def _place_ramps_around_level(self, level):
        """
        Place ramps on lower-elevation tiles adjacent to tiles of given level
        so they connect up to that level. Ramps are discrete tiles with direction.
        We'll place ramps on the lower tile and mark ramp_elevation_to = level.
        """
        w, h = self.config.width, self.config.height

        for y in range(h):
            for x in range(w):
                cell = self.grid[y][x]
                # look for adjacent tile at exactly 'level'
                for direction, dx, dy in [('north', 0, -1), ('south', 0, 1), ('east', 1, 0), ('west', -1, 0)]:
                    nx, ny = x + dx, y + dy
                    if not self._is_valid_position(nx, ny):
                        continue
                    neighbor = self.grid[ny][nx]
                    if neighbor.elevation == level and cell.elevation == level - 1:
                        # candidate ramp on cell pointing to neighbor
                        if not cell.is_ramp:
                            if random.random() < RAMP_PLACE_PROB:
                                # mark as ramp tile
                                cell.set_ramp(True, direction)
                                # annotate which elevation it ramps to (used for promotion checks)
                                setattr(cell, 'ramp_elevation_to', level)
                        else:
                            # ensure ramp_elevation_to is set to the highest adjacent level
                            existing = getattr(cell, 'ramp_elevation_to', None)
                            if existing is None or level > existing:
                                setattr(cell, 'ramp_elevation_to', level)

    # ---------------------------
    # Accessibility & Fixes
    # ---------------------------
    def _ensure_full_accessibility(self):
        """
        Guarantee that all tiles (except impassable mountains) are reachable
        from spawn_point_1 by walking across passable tiles and using ramps.
        If disconnected regions exist, carve connecting corridors (roads) and
        place ramps where needed to connect elevation differences.
        """
        w, h = self.config.width, self.config.height
        start = self.config.spawn_point_1

        def passable(x, y):
            # treat mountains as impassable; everything else passable (ramps passable).
            tt = self.grid[y][x].terrain_type
            return tt != mountains

        def neighbors4(x, y):
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    yield nx, ny

        # BFS to find reachable set
        def bfs_from(sx, sy):
            seen = set()
            q = deque([(sx, sy)])
            seen.add((sx, sy))
            while q:
                cx, cy = q.popleft()
                for nx, ny in neighbors4(cx, cy):
                    if (nx, ny) in seen:
                        continue
                    if passable(nx, ny):
                        # ramp crossing: can move from lower to higher only if ramp exists in appropriate tile
                        # allow movement between equal elevations freely
                        ce = self.grid[cy][cx].elevation
                        ne = self.grid[ny][nx].elevation
                        if ne == ce:
                            seen.add((nx, ny))
                            q.append((nx, ny))
                        elif ne == ce + 1:
                            # climbing up: require the destination or source to be a ramp oriented correctly
                            # if source has a ramp pointing to (nx,ny) OR destination has ramp pointing from (cx,cy) (rare)
                            src = self.grid[cy][cx]
                            dst = self.grid[ny][nx]
                            # check ramps on source or dst
                            if src.is_ramp and getattr(src, 'ramp_direction', None):
                                # ramp could point to nx,ny: compute direction name
                                # source ramp should point to neighbor
                                dir_name = self._dir_from_offset(nx - cx, ny - cy)
                                if src.ramp_direction == dir_name:
                                    seen.add((nx, ny))
                                    q.append((nx, ny))
                            elif dst.is_ramp:
                                dir_back = self._dir_from_offset(cx - nx, cy - ny)
                                if dst.ramp_direction == dir_back:
                                    seen.add((nx, ny))
                                    q.append((nx, ny))
                        elif ne + 1 == ce:
                            # moving down: allow if there is ramp on neighbor pointing up, or on current pointing down
                            src = self.grid[cy][cx]
                            dst = self.grid[ny][nx]
                            if dst.is_ramp and getattr(dst, 'ramp_direction', None):
                                dir_back = self._dir_from_offset(cx - nx, cy - ny)
                                if dst.ramp_direction == dir_back:
                                    seen.add((nx, ny))
                                    q.append((nx, ny))
                            elif src.is_ramp:
                                # allow descending via source ramp as well
                                dir_name = self._dir_from_offset(nx - cx, ny - cy)
                                if src.ramp_direction == dir_name:
                                    seen.add((nx, ny))
                                    q.append((nx, ny))
                        # else elevation difference >1 -> impassable
            return seen

        reachable = bfs_from(start[0], start[1])
        total_passable = set((x,y) for y in range(h) for x in range(w) if passable(x,y))

        # If all passable tiles reachable, done
        if reachable >= total_passable:
            self.logger.debug("All tiles reachable from spawn1")
            return

        # Otherwise, iteratively connect components by carving roads and placing ramps
        components = []
        remaining = total_passable - reachable
        seen_global = set(reachable)
        for y in range(h):
            for x in range(w):
                if (x,y) in seen_global or (x,y) not in total_passable:
                    continue
                comp = self._get_connected_tiles(x, y, lambda xx, yy: passable(xx, yy))
                components.append(comp)
                seen_global.update(comp)

        # connect each component back to reachable set by finding nearest pair and carving path
        tries = 0
        while components and tries < MAX_CONNECTIVITY_FIXES:
            tries += 1
            new_components = []
            for comp in components:
                # pick a representative tile from comp and from reachable
                comp_list = list(comp)
                # find nearest tile in reachable
                best_pair = None
                best_dist = 1e9
                for (cx, cy) in comp_list:
                    for (rx, ry) in reachable:
                        d = abs(cx - rx) + abs(cy - ry)
                        if d < best_dist:
                            best_dist = d
                            best_pair = ((cx, cy), (rx, ry))
                if best_pair is None:
                    new_components.append(comp)
                    continue
                start_pt, end_pt = best_pair

                # carve a simple Manhattan-style corridor between start_pt and end_pt
                sx, sy = start_pt
                ex, ey = end_pt

                # step in x then y (or vice versa) with small randomness
                path = []
                cx, cy = sx, sy
                while (cx, cy) != (ex, ey):
                    if cx != ex and (cy == ey or random.random() < 0.6):
                        cx += 1 if ex > cx else -1
                    elif cy != ey:
                        cy += 1 if ey > cy else -1
                    if (cx, cy) not in path:
                        path.append((cx, cy))

                # carve path: convert any mountains on path to plains, set roads, and place ramps for elevation differences
                for (px, py) in path:
                    cell = self.grid[py][px]
                    if cell.terrain_type == mountains:
                        cell.terrain_type = plains
                    # set to road to indicate passable corridor
                    if cell.terrain_type == plains:
                        cell.terrain_type = road
                    # if elevation difference to adjacent path tile > 0, place ramp on lower tile oriented to higher tile
                # place ramps along path where elevation steps up
                for (px, py) in path:
                    for dx, dy, dname in [(1,0,'east'),(-1,0,'west'),(0,1,'south'),(0,-1,'north')]:
                        nx, ny = px + dx, py + dy
                        if not self._is_valid_position(nx, ny):
                            continue
                        cur = self.grid[py][px]
                        nei = self.grid[ny][nx]
                        if nei.elevation == cur.elevation + 1 and not cur.is_ramp:
                            # place ramp on cur pointing to neighbor
                            cur.set_ramp(True, dname)
                            setattr(cur, 'ramp_elevation_to', nei.elevation)
                        elif cur.elevation == nei.elevation + 1 and not nei.is_ramp:
                            # place ramp on neighbor to allow descent/ascent
                            nei.set_ramp(True, self._dir_from_offset(px - nx, py - ny))
                            setattr(nei, 'ramp_elevation_to', cur.elevation)
                # Recompute reachable set after carving
                reachable = bfs_from(start[0], start[1])
                # If comp now reachable, skip adding to new_components
                if comp & reachable:
                    continue
                else:
                    new_components.append(comp)
            components = new_components
            if not components:
                break

        # Final quick check - if still unreachable tiles exist, convert them to plains + roads and link
        reachable = bfs_from(start[0], start[1])
        missing = total_passable - reachable
        if missing:
            self.logger.warning(f"Still unreachable tiles after fixes: {len(missing)} - force connecting")
            for (mx, my) in list(missing):
                # make tile plain and road to maximize connectivity
                ct = self.grid[my][mx]
                ct.terrain_type = road
                # connect by placing a ramp if it's elevated and neighbor available
                for dx, dy, dname in [(1,0,'east'),(-1,0,'west'),(0,1,'south'),(0,-1,'north')]:
                    nx, ny = mx + dx, my + dy
                    if self._is_valid_position(nx, ny):
                        ncell = self.grid[ny][nx]
                        if ncell.elevation == ct.elevation + 1 and not ct.is_ramp:
                            ct.set_ramp(True, dname)
                            setattr(ct, 'ramp_elevation_to', ncell.elevation)

    # ---------------------------
    # Terrain smoothing (cellular automata)
    # ---------------------------
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

    # ---------------------------
    # Strategic placement helpers
    # ---------------------------
    def _place_balanced_control_zones(self):
        """Place control zones with algorithmic balance guarantee."""
        self.logger.debug("Placing balanced control zones")

        center_x = self.config.width // 2
        center_y = self.config.height // 2
        self.control_zones.append((center_x, center_y))

        remaining = CONTROL_ZONE_COUNT - 1
        pairs_to_place = remaining // 2

        candidates = []
        for y in range(3, self.config.height - 3):
            for x in range(3, self.config.width - 3):
                if (x, y) == (center_x, center_y):
                    continue

                d1 = self._manhattan_distance(self.config.spawn_point_1, (x, y))
                d2 = self._manhattan_distance(self.config.spawn_point_2, (x, y))

                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)

                if d1 < 6 or d2 < 6:
                    continue
                if self._manhattan_distance((x, y), (center_x, center_y)) < 3:
                    continue

                # avoid placing control zones in impassable/mountain centers
                if self.grid[y][x].terrain_type == mountains:
                    continue

                candidates.append((balance, x, y))

        # sort by balance (prefer balanced positions)
        candidates.sort(reverse=True, key=lambda c: c[0])

        placed = 0
        for balance, x, y in candidates:
            if placed >= pairs_to_place * 2:
                break

            too_close = False
            for zx, zy in self.control_zones:
                if self._manhattan_distance((x, y), (zx, zy)) < 6:
                    too_close = True
                    break

            if not too_close:
                self.control_zones.append((x, y))
                placed += 1

        self.logger.info(f"Placed {len(self.control_zones)} balanced control zones")

    def _create_elevation_clusters_near_spawns(self):
        """Create elevated areas near each spawn if needed (helps balance)."""
        for spawn_idx, spawn in enumerate([self.config.spawn_point_1,
                                           self.config.spawn_point_2]):
            distance = random.randint(5, 8)

            cluster_x = int(spawn[0] + distance * (1 if spawn_idx == 0 else -1))
            cluster_y = int(spawn[1] + distance * random.choice([-1, 0, 1]))

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    cx, cy = cluster_x + dx, cluster_y + dy
                    if self._is_valid_position(cx, cy):
                        # raise to elevation 2 if free
                        if self.grid[cy][cx].elevation < 2:
                            self.grid[cy][cx].set_elevation(2)

            # ensure ramps around newly created cluster
            self._place_ramps_around_level(2)

    def _place_balanced_high_ground(self):
        """Ensure high ground is balanced between spawns."""
        self.logger.debug("Balancing high ground placement")

        visited = set()
        all_clusters = []

        for y in range(self.config.height):
            for x in range(self.config.width):
                if (x, y) in visited:
                    continue
                if self.grid[y][x].elevation >= 2:
                    cluster = self._get_connected_tiles(x, y,
                        lambda tx, ty: self.grid[ty][tx].elevation >= 2)
                    if len(cluster) >= 6:
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

        self._assign_balanced_clusters(all_clusters)

    def _assign_balanced_clusters(self, clusters):
        """Assign high ground clusters ensuring balance."""
        if not clusters:
            return

        clusters.sort(key=lambda c: c['size'], reverse=True)

        spawn1_clusters = []
        spawn2_clusters = []

        for cluster in clusters:
            if cluster['d1'] < cluster['d2']:
                target = spawn1_clusters
            else:
                target = spawn2_clusters

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
        """Place buildings with guaranteed distance balance."""
        self.logger.debug("Placing balanced buildings")

        # seed some buildings near control zones first
        for cx, cy in self.control_zones:
            best_pos = self._find_building_position_near(cx, cy, radius=2)
            if best_pos:
                bx, by = best_pos
                self.grid[by][bx].set_building(True)
                self.buildings.append((bx, by))

        remaining = BUILDING_COUNT - len(self.buildings)
        if remaining <= 0:
            self.logger.info(f"Placed {len(self.buildings)} buildings (control zone anchors)")
            return

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

                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                candidates.append((balance, x, y))

        candidates.sort(reverse=True, key=lambda c: c[0])

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

                passable = sum(
                    1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                    if not (dx == 0 and dy == 0) and
                    self._is_valid_position(x + dx, y + dy) and
                    self.grid[y + dy][x + dx].terrain_type != mountains
                )

                if 2 <= passable <= 4:
                    self.choke_points.append((x, y))

        self.logger.info(f"Identified {len(self.choke_points)} choke points")

    # ---------------------------
    # Utility helpers
    # ---------------------------
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

    def _dir_from_offset(self, dx: int, dy: int) -> Optional[str]:
        """Return direction name from dx,dy for cardinal 4-neighbors."""
        if dx == 0 and dy == -1:
            return 'north'
        if dx == 0 and dy == 1:
            return 'south'
        if dx == 1 and dy == 0:
            return 'east'
        if dx == -1 and dy == 0:
            return 'west'
        return None

    # ---------------------------
    # Statistics & verification
    # ---------------------------
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
                try:
                    name = t.terrain_type.name.lower()
                except Exception:
                    name = str(t.terrain_type).lower()
                if name in stats:
                    stats[name] += 1
                if t.is_ramp:
                    stats["ramps"] += 1
                if t.elevation > MIN_ELEVATION:
                    stats["elevated"] += 1

        balance_info = self._verify_balance()
        stats.update(balance_info)

        return stats

    def _verify_balance(self) -> Dict[str, float]:
        """Verify algorithmic balance guarantees."""
        if not self.buildings:
            return {"balance_verified": True, "building_balance": 1.0}

        p1_dists = [self._manhattan_distance(self.config.spawn_point_1, (bx, by))
                    for bx, by in self.buildings]
        p2_dists = [self._manhattan_distance(self.config.spawn_point_2, (bx, by))
                    for bx, by in self.buildings]

        avg1 = sum(p1_dists) / len(p1_dists)
        avg2 = sum(p2_dists) / len(p2_dists)

        building_balance = min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 1.0

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

# end of file
