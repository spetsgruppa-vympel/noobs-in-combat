"""
Procedural map generator with automatic validation and correction.

This generator includes built-in validators that run during generation
and automatically fix issues like disconnected regions, imbalanced
strategic elements, and ramp placement errors.

Public API:
    config = MapConfig(width=40, height=30, seed=12345)
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
    manhattan_path, find_cluster
)


class MapGenerator:
    """
    Self-correcting procedural map generator.

    Automatically validates and fixes:
    - Disconnected regions (creates corridors)
    - Imbalanced strategic elements (adjusts placement)
    - Invalid ramp placement (corrects elevation/ramp rules)
    - Unreachable spawn points (carves access paths)

    Example:
        >>> config = MapConfig(width=40, height=30, seed=42)
        >>> generator = MapGenerator(config)
        >>> grid = generator.generate()  # Automatically corrected
        >>> stats = generator.get_statistics()
        >>> print(f"Corrections made: {stats.get('corrections_applied', 0)}")
    """

    def __init__(self, config: MapConfig = None):
        """Initialize generator with configuration."""
        self.config = config or MapConfig()
        self.logger = get_logger(__name__)

        # Initialize deterministic RNG
        if self.config.seed is not None:
            self.rng = random.Random(self.config.seed)
            self.logger.info(f"Generator initialized with seed: {self.config.seed}")
        else:
            self.rng = random.Random()
            self.logger.info("Generator initialized with random seed")

        # Map data
        self.grid: List[List[tile]] = []
        self.buildings: List[Tuple[int, int]] = []
        self.control_zones: List[Tuple[int, int]] = []
        self.high_ground: List[Tuple[int, int]] = []
        self.chokepoints: List[Tuple[int, int]] = []

        # Correction tracking
        self.corrections = {
            'connectivity_fixes': 0,
            'ramp_corrections': 0,
            'balance_adjustments': 0,
            'spawn_access_fixes': 0
        }

    def generate(self) -> List[List[tile]]:
        """
        Generate complete map with automatic validation and correction.

        Generation phases:
        1. Initialize grid
        2. Generate terrain
        3. Generate elevation with validation
        4. Smooth terrain
        5. Auto-correct connectivity
        6. Place strategic elements with auto-balancing
        7. Validate and fix all invariants

        Returns:
            Validated and corrected grid
        """
        self.logger.info(f"Generating {self.config.width}×{self.config.height} map")

        # Phase 1: Initialize
        self.grid = create_grid(self.config.width, self.config.height, plains)
        self.buildings = []
        self.control_zones = []
        self.high_ground = []
        self.chokepoints = []
        self.corrections = {k: 0 for k in self.corrections}

        # Phase 2: Terrain generation
        self._generate_terrain()

        # Phase 3: Elevation with auto-validation
        self._generate_elevation()
        self._validate_and_fix_ramps()  # Auto-correct ramp issues

        # Phase 4: Smoothing
        self._smooth_terrain()

        # Phase 5: Auto-correct connectivity
        self._ensure_connectivity()
        self._ensure_spawn_access()  # Guarantee spawn accessibility

        # Phase 6: Strategic elements with auto-balancing
        self._place_strategic_elements()
        self._auto_balance_elements()  # Adjust if imbalanced

        # Phase 7: Final validation
        self._final_validation()

        # Log results
        total_corrections = sum(self.corrections.values())
        stats = self.get_statistics()
        self.logger.info(
            f"Map complete: {len(self.buildings)} buildings, "
            f"{len(self.control_zones)} zones, balance={stats['overall_balance']:.2%}, "
            f"corrections={total_corrections}"
        )

        return self.grid

    # ========================================================================
    # TERRAIN GENERATION
    # ========================================================================

    def _generate_terrain(self):
        """Generate terrain using seeded blob growth."""
        # Place terrain types in order for realistic adjacencies
        for terrain_type, density, seed_f, growth_f, adj_bonus in [
            (mountains, self.config.mountain_density, 0.002, 0.004, {}),
            (forest, self.config.forest_density, 0.003, 0.01, {forest: 0.2, plains: 0.1}),
            (urban, self.config.urban_density, 0.002, 0.006, {urban: 0.25, road: 0.15}),
            (debris, self.config.debris_density, 0.002, 0.005, {urban: 0.2}),
            (road, self.config.road_density, 0.001, 0.004, {urban: 0.2, road: 0.15})
        ]:
            self._place_terrain_type(terrain_type, density, seed_f, growth_f, adj_bonus)

    def _place_terrain_type(self, terrain_type, density: float, seed_factor: float,
                           growth_factor: float, adjacency_bonus: dict):
        """Place single terrain type using seeded blob growth."""
        area = self.config.area
        seed_count = max(1, int(area * density * seed_factor))
        avg_size = max(2, int(area * density * growth_factor))
        target = seed_count * avg_size

        # Generate center-biased seeds
        seeds = [self._center_biased_pos() for _ in range(seed_count)]

        # Grow from seeds
        placed = 0
        frontier = deque(seeds)
        visited = set(seeds)

        while frontier and placed < target and len(visited) < area:
            x, y = frontier.popleft()
            if not valid_pos(x, y, self.config.width, self.config.height):
                continue

            current = self.grid[y][x]
            if current.terrain_type == plains:
                # Calculate placement probability
                place_prob = 0.92
                for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height):
                    n_type = self.grid[ny][nx].terrain_type
                    if n_type in adjacency_bonus:
                        place_prob += adjacency_bonus[n_type] / 8.0

                if self.rng.random() < min(1.0, place_prob):
                    current.terrain_type = terrain_type
                    placed += 1

                    # Expand to neighbors
                    for nx, ny, _ in neighbors_8(x, y, self.config.width, self.config.height):
                        if (nx, ny) not in visited and self.rng.random() < 0.55:
                            frontier.append((nx, ny))
                            visited.add((nx, ny))

    def _center_biased_pos(self) -> Tuple[int, int]:
        """Generate position biased toward map center."""
        cx = self.config.width / 2.0
        cy = self.config.height / 2.0
        sigma_x = max(1.0, self.config.width / 6.0)
        sigma_y = max(1.0, self.config.height / 6.0)

        for _ in range(50):
            x = int(self.rng.gauss(cx, sigma_x))
            y = int(self.rng.gauss(cy, sigma_y))
            if valid_pos(x, y, self.config.width, self.config.height):
                return (x, y)

        return (self.rng.randrange(self.config.width),
                self.rng.randrange(self.config.height))

    def _smooth_terrain(self):
        """Smooth terrain using cellular automata."""
        for _ in range(self.config.smoothing_passes):
            new_grid = [row[:] for row in self.grid]

            for y in range(1, self.config.height - 1):
                for x in range(1, self.config.width - 1):
                    current = self.grid[y][x].terrain_type
                    if current == plains:
                        continue

                    # Count same-type neighbors
                    same = sum(1 for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                             if not (dx == 0 and dy == 0) and
                             self.grid[y+dy][x+dx].terrain_type == current)

                    # Convert if too few neighbors
                    if same < 3:
                        types = {}
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                t = self.grid[y+dy][x+dx].terrain_type
                                types[t] = types.get(t, 0) + 1
                        if types:
                            new_grid[y][x].terrain_type = max(types, key=types.get)

            self.grid = new_grid

    # ========================================================================
    # ELEVATION GENERATION WITH VALIDATION
    # ========================================================================

    def _generate_elevation(self):
        """Generate elevation levels with support requirements."""
        area = self.config.area
        seed_count = max(1, int(area * self.config.elevation_density * 0.002))
        avg_size = max(4, int(area * 0.01))

        # Level 1: Initial seeds
        seeds = [self._interior_pos() for _ in range(seed_count)]
        self._grow_elevation_level(1, seeds, seed_count * avg_size, 0, 0)
        self._place_ramps_at_level(1)

        # Higher levels with support requirements
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
        """Generate position away from edges."""
        mx = max(2, self.config.width // 10)
        my = max(2, self.config.height // 10)
        return (self.rng.randint(mx, self.config.width - 1 - mx),
                self.rng.randint(my, self.config.height - 1 - my))

    def _find_elevation_candidates(self, level: int) -> List[Tuple[int, int]]:
        """Find tiles adjacent to previous elevation level."""
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
        """Grow single elevation level from seeds."""
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

            # Check promotion conditions
            can_promote = False
            if level == 1:
                can_promote = self.rng.random() < 0.92
            else:
                support = sum(1 for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height)
                            if self.grid[ny][nx].elevation >= level - 1)
                ramps = sum(1 for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height)
                          if (self.grid[ny][nx].is_ramp and
                              getattr(self.grid[ny][nx], 'ramp_elevation_to', 0) == level))

                can_promote = (support >= min_support and ramps >= min_ramps) or self.rng.random() < 0.04

            if can_promote:
                current.set_elevation(level)
                placed += 1

                # Expand
                prob = 0.62 if level == 1 else 0.46
                for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                    if (nx, ny) not in visited and self.rng.random() < prob:
                        frontier.append((nx, ny))
                        visited.add((nx, ny))

    def _place_ramps_at_level(self, level: int):
        """Place ramps at borders of elevation level."""
        for y in range(self.config.height):
            for x in range(self.config.width):
                cell = self.grid[y][x]
                for nx, ny, direction in neighbors_4(x, y, self.config.width, self.config.height):
                    neighbor = self.grid[ny][nx]
                    if neighbor.elevation == level and cell.elevation == level - 1:
                        if not cell.is_ramp and self.rng.random() < self.config.ramp_placement_probability:
                            cell.set_ramp(True, direction)
                            cell.ramp_elevation_to = level
                        elif cell.is_ramp:
                            existing = getattr(cell, 'ramp_elevation_to', None)
                            if existing is None or level > existing:
                                cell.ramp_elevation_to = level

    def _validate_and_fix_ramps(self):
        """AUTO-VALIDATOR: Check and fix ramp placement issues."""
        corrections = 0

        for y in range(self.config.height):
            for x in range(self.config.width):
                cell = self.grid[y][x]

                if cell.is_ramp:
                    # Check if ramp has higher neighbor
                    has_higher = False
                    for nx, ny, _ in neighbors_4(x, y, self.config.width, self.config.height):
                        if self.grid[ny][nx].elevation > cell.elevation:
                            has_higher = True
                            break

                    # FIX: Remove ramp if no higher neighbor
                    if not has_higher:
                        cell.set_ramp(False, None)
                        if hasattr(cell, 'ramp_elevation_to'):
                            delattr(cell, 'ramp_elevation_to')
                        corrections += 1

                # Check for large elevation differences without ramps
                for nx, ny, direction in neighbors_4(x, y, self.config.width, self.config.height):
                    neighbor = self.grid[ny][nx]
                    diff = abs(cell.elevation - neighbor.elevation)

                    if diff > 1 and not cell.is_ramp and not neighbor.is_ramp:
                        # FIX: Add ramp on lower tile
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

    # ========================================================================
    # CONNECTIVITY AUTO-CORRECTION
    # ========================================================================

    def _ensure_connectivity(self):
        """AUTO-VALIDATOR: Ensure all passable tiles are connected."""
        max_fixes = 6
        attempts = 0

        while attempts < max_fixes:
            # Find disconnected components
            main = bfs_reachable(self.grid, self.config.spawn_point_1,
                               self._is_passable, self._can_traverse)

            components = find_components(self.grid, self._is_passable,
                                        self._can_traverse, main)

            if not components:
                break  # All connected

            self.logger.debug(f"Found {len(components)} disconnected regions")

            # Connect each component
            for component in components:
                # Find nearest pair
                best_dist = float('inf')
                best_pair = None

                comp_sample = list(component)[:50]
                main_sample = list(main)[:50]

                for c_pos in comp_sample:
                    for m_pos in main_sample:
                        dist = manhattan(c_pos, m_pos)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (c_pos, m_pos)

                # Create corridor
                if best_pair:
                    self._create_corridor(best_pair[0], best_pair[1])
                    self.corrections['connectivity_fixes'] += 1

            attempts += 1

        if attempts >= max_fixes:
            self.logger.warning(f"Connectivity fixes exhausted ({max_fixes} attempts)")

    def _ensure_spawn_access(self):
        """AUTO-VALIDATOR: Guarantee spawn points are accessible."""
        corrections = 0

        # Make spawn tiles passable
        for spawn in [self.config.spawn_point_1, self.config.spawn_point_2]:
            sx, sy = spawn
            if self.grid[sy][sx].terrain_type == mountains:
                self.grid[sy][sx].terrain_type = plains
                corrections += 1

        # Verify spawns can reach each other
        reachable = bfs_reachable(self.grid, self.config.spawn_point_1,
                                 self._is_passable, self._can_traverse)

        if self.config.spawn_point_2 not in reachable:
            # FIX: Create corridor between spawns
            self._create_corridor(self.config.spawn_point_1, self.config.spawn_point_2)
            corrections += 1

        if corrections > 0:
            self.logger.debug(f"Auto-fixed {corrections} spawn access issues")
            self.corrections['spawn_access_fixes'] = corrections

    def _is_passable(self, grid, x: int, y: int) -> bool:
        """Check if tile is passable."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return False
        return grid[y][x].terrain_type != mountains

    def _can_traverse(self, grid, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if movement between tiles is possible."""
        if not self._is_passable(grid, x1, y1) or not self._is_passable(grid, x2, y2):
            return False

        t1 = grid[y1][x1]
        t2 = grid[y2][x2]
        diff = abs(t2.elevation - t1.elevation)

        if diff == 0:
            return True
        if diff > 1:
            return False

        # Check for ramps
        return t1.is_ramp or t2.is_ramp

    def _create_corridor(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Create corridor between two points."""
        # Try A* first
        path = astar_path(self.grid, start, end, self._terrain_cost)

        # Fallback to Manhattan
        if not path:
            path = manhattan_path(start, end, self.rng)

        # Carve corridor
        for x, y in path:
            cell = self.grid[y][x]
            if cell.terrain_type == mountains:
                cell.terrain_type = plains
            if cell.terrain_type == plains:
                cell.terrain_type = road

        # Add ramps where needed
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            t1 = self.grid[y1][x1]
            t2 = self.grid[y2][x2]

            if t2.elevation == t1.elevation + 1 and not t1.is_ramp:
                direction = direction_from_offset(x2 - x1, y2 - y1)
                if direction:
                    t1.set_ramp(True, direction)
                    t1.ramp_elevation_to = t2.elevation
            elif t1.elevation == t2.elevation + 1 and not t2.is_ramp:
                direction = direction_from_offset(x1 - x2, y1 - y2)
                if direction:
                    t2.set_ramp(True, direction)
                    t2.ramp_elevation_to = t1.elevation

    def _terrain_cost(self, grid, x: int, y: int) -> float:
        """Get movement cost for terrain."""
        if not valid_pos(x, y, self.config.width, self.config.height):
            return 999.0

        terrain = grid[y][x].terrain_type
        if terrain == plains:
            return 1.0
        elif terrain == road:
            return 0.8
        elif terrain == mountains:
            return 10.0
        else:
            return 2.0

    # ========================================================================
    # STRATEGIC PLACEMENT WITH AUTO-BALANCING
    # ========================================================================

    def _place_strategic_elements(self):
        """Place control zones, buildings, and identify features."""
        # Control zones
        self._place_control_zones()

        # Buildings
        target = int(self.config.area * self.config.building_density)
        self._place_buildings(target)

        # Identify features
        self._identify_high_ground()
        self._identify_chokepoints()

    def _place_control_zones(self):
        """Place balanced control zones."""
        zones = []

        # Center zone
        cx, cy = self.config.width // 2, self.config.height // 2
        zones.append((cx, cy))

        # Balanced pairs
        remaining = self.config.control_zone_count - 1
        pairs = remaining // 2

        # Find candidates
        candidates = []
        for y in range(3, self.config.height - 3):
            for x in range(3, self.config.width - 3):
                if (x, y) == (cx, cy):
                    continue
                if self.grid[y][x].terrain_type == mountains:
                    continue

                d1 = manhattan(self.config.spawn_point_1, (x, y))
                d2 = manhattan(self.config.spawn_point_2, (x, y))

                if d1 < 6 or d2 < 6:
                    continue
                if manhattan((x, y), (cx, cy)) < 3:
                    continue

                balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                candidates.append((balance, x, y))

        candidates.sort(reverse=True)

        # Place with spacing
        for balance, x, y in candidates:
            if len(zones) >= self.config.control_zone_count:
                break

            if all(manhattan((x, y), z) >= 6 for z in zones):
                zones.append((x, y))

        self.control_zones = zones

    def _place_buildings(self, target: int):
        """Place balanced buildings."""
        buildings = []

        # Near control zones first
        for cx, cy in self.control_zones:
            pos = self._find_building_near(cx, cy, buildings, 2)
            if pos:
                bx, by = pos
                self.grid[by][bx].set_building(True)
                buildings.append((bx, by))

        # Remaining with balance
        remaining = target - len(buildings)
        if remaining > 0:
            candidates = []
            for y in range(self.config.height):
                for x in range(self.config.width):
                    terrain = self.grid[y][x].terrain_type
                    if terrain in [mountains, forest, debris]:
                        continue
                    if self.grid[y][x].is_ramp or self.grid[y][x].is_building:
                        continue
                    if not self._check_building_spacing((x, y), buildings):
                        continue

                    d1 = manhattan(self.config.spawn_point_1, (x, y))
                    d2 = manhattan(self.config.spawn_point_2, (x, y))
                    balance = 1.0 - abs(d1 - d2) / max(d1 + d2, 1)
                    candidates.append((balance, x, y))

            candidates.sort(reverse=True)

            for balance, x, y in candidates[:remaining]:
                if self._check_building_spacing((x, y), buildings):
                    self.grid[y][x].set_building(True)
                    buildings.append((x, y))

        self.buildings = buildings

    def _find_building_near(self, cx: int, cy: int, existing: List, radius: int) -> Optional[Tuple[int, int]]:
        """Find building position near target."""
        for r in range(radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if not valid_pos(x, y, self.config.width, self.config.height):
                        continue

                    terrain = self.grid[y][x].terrain_type
                    if terrain in [mountains, forest, debris]:
                        continue
                    if self.grid[y][x].is_ramp or self.grid[y][x].is_building:
                        continue
                    if self._check_building_spacing((x, y), existing):
                        return (x, y)
        return None

    def _check_building_spacing(self, pos: Tuple[int, int], buildings: List) -> bool:
        """Check minimum building spacing."""
        return all(manhattan(pos, b) >= self.config.min_building_spacing for b in buildings)

    def _identify_high_ground(self):
        """Identify high ground clusters."""
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
        """Identify natural chokepoints."""
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
        """AUTO-VALIDATOR: Adjust elements if balance is poor."""
        metrics = self._calculate_balance()

        # If balance is poor, adjust by removing worst offenders
        if metrics['overall_balance'] < 0.85 and len(self.buildings) > 0:
            self.logger.debug(f"Balance suboptimal ({metrics['overall_balance']:.2%}), auto-adjusting")

            # Remove buildings that hurt balance most
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
        """Find building contributing most to imbalance."""
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
        """Calculate balance metrics."""
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
        """Final validation pass - log any remaining issues."""
        # Check connectivity one more time
        reachable = bfs_reachable(self.grid, self.config.spawn_point_1,
                                 self._is_passable, self._can_traverse)

        all_passable = sum(1 for y in range(self.config.height)
                          for x in range(self.config.width)
                          if self._is_passable(self.grid, x, y))

        if len(reachable) != all_passable:
            self.logger.warning(f"Final check: {all_passable - len(reachable)} tiles still unreachable")

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> Dict:
        """Get comprehensive map statistics including corrections."""
        stats = {
            'width': self.config.width,
            'height': self.config.height,
            'area': self.config.area,
            'seed': self.config.seed,
        }

        # Count terrain
        counts = {'plains': 0, 'forest': 0, 'urban': 0, 'mountains': 0,
                 'road': 0, 'debris': 0, 'ramps': 0, 'elevated': 0}

        for row in self.grid:
            for t in row:
                name = t.terrain_type.name.lower()
                if name in counts:
                    counts[name] += 1
                if t.is_ramp:
                    counts['ramps'] += 1
                if t.elevation > 0:
                    counts['elevated'] += 1

        stats.update(counts)

        # Strategic elements
        stats['buildings'] = len(self.buildings)
        stats['control_zones'] = len(self.control_zones)
        stats['high_ground_clusters'] = len(self.high_ground)
        stats['chokepoints'] = len(self.chokepoints)

        # Balance metrics
        stats.update(self._calculate_balance())

        # Corrections applied
        stats['corrections_applied'] = sum(self.corrections.values())
        stats['correction_details'] = self.corrections.copy()

        return stats

    def get_terrain_statistics(self) -> Dict:
        """Legacy compatibility method."""
        return self.get_statistics()


# Legacy exports for backward compatibility
MapConfig = MapConfig