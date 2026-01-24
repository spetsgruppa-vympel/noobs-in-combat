"""
Utility helpers for grid operations, pathfinding, clustering and fast value-noise.

This module provides:
- create_grid: construct initial tile grid
- neighbor iteration helpers (4- and 8-neighborhood)
- manhattan and direction helpers
- BFS reachability and connected components helpers
- A* pathfinding (4-neighborhood)
- simple multi-octave value-noise used for biome blending
"""

import heapq
from collections import deque
from typing import List, Tuple, Set, Optional, Callable, Iterator
from main.game.data.maps.tile import tile
from main.config import TILE_SIZE

# Cardinal and diagonal directions used by many algorithms
DIRECTIONS_4 = [
    ('north', 0, -1), ('south', 0, 1),
    ('east', 1, 0), ('west', -1, 0)
]
DIRECTIONS_8 = DIRECTIONS_4 + [
    ('northeast', 1, -1), ('southeast', 1, 1),
    ('southwest', -1, 1), ('northwest', -1, -1)
]


def create_grid(width: int, height: int, terrain_type) -> List[List[tile]]:
    """
    Create a grid (height x width) filled with new tile objects.

    Each tile is a freshly constructed tile instance to avoid aliasing.
    """
    grid: List[List[tile]] = []
    for y in range(height):
        row: List[tile] = []
        for x in range(width):
            t = tile(x=x, y=y, terrain_type=terrain_type, size=TILE_SIZE,
                     occupied=False, is_building=False, elevation=0,
                     is_ramp=False, ramp_direction=None)
            row.append(t)
        grid.append(row)
    return grid


def valid_pos(x: int, y: int, width: int, height: int) -> bool:
    """Return True if (x,y) is inside the rectangular grid bounds."""
    return 0 <= x < width and 0 <= y < height


def neighbors_4(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Yield cardinal neighbors (nx, ny, direction) that fall inside bounds."""
    for direction, dx, dy in DIRECTIONS_4:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


def neighbors_8(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Yield 8-connected neighbors (including diagonals)."""
    for direction, dx, dy in DIRECTIONS_8:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


def manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance between integer positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def direction_from_offset(dx: int, dy: int) -> Optional[str]:
    """Return direction name (from DIRECTIONS_8) for given offset, or None if not exact match."""
    for direction, ddx, ddy in DIRECTIONS_8:
        if dx == ddx and dy == ddy:
            return direction
    return None


def bfs_reachable(grid: List[List[tile]], start: Tuple[int, int],
                  passable_fn: Callable, can_move_fn: Callable) -> Set[Tuple[int, int]]:
    """
    Breadth-First Search to compute the set of tiles reachable from start.

    passable_fn(grid, x, y) -> bool: tile-level passability (e.g., tile not mountain)
    can_move_fn(grid, x1, y1, x2, y2) -> bool: movement rules between two adjacent tiles
    """
    if not passable_fn(grid, start[0], start[1]):
        return set()

    width = len(grid[0]) if grid else 0
    height = len(grid)

    reachable = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny, _ in neighbors_4(x, y, width, height):
            if (nx, ny) in reachable:
                continue
            if can_move_fn(grid, x, y, nx, ny):
                reachable.add((nx, ny))
                queue.append((nx, ny))
    return reachable


def find_components(grid: List[List[tile]], passable_fn: Callable,
                    can_move_fn: Callable, exclude: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    """
    Return a list of connected components (sets of positions) for passable tiles,
    excluding any tiles that are in `exclude`.
    """
    width = len(grid[0]) if grid else 0
    height = len(grid)
    all_passable = set()
    for y in range(height):
        for x in range(width):
            if passable_fn(grid, x, y):
                all_passable.add((x, y))

    remaining = all_passable - exclude
    components: List[Set[Tuple[int, int]]] = []
    while remaining:
        seed = next(iter(remaining))
        component = bfs_reachable(grid, seed, passable_fn, can_move_fn)
        components.append(component)
        remaining -= component
    return components


def astar_path(grid: List[List[tile]], start: Tuple[int, int], goal: Tuple[int, int],
               cost_fn: Callable) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on 4-neighborhood. Returns None if no path found.

    cost_fn(grid, x, y) -> float should return traversal cost for entering tile (x,y).
    Heuristic uses Manhattan distance which is admissible for 4-neighborhood.
    """
    width = len(grid[0]) if grid else 0
    height = len(grid)

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            # Reconstruct path
            path: List[Tuple[int, int]] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = current
        for nx, ny, _ in neighbors_4(x, y, width, height):
            terrain_cost = float(cost_fn(grid, nx, ny))
            new_cost = cost_so_far[current] + terrain_cost
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost + manhattan((nx, ny), goal)
                heapq.heappush(frontier, (priority, (nx, ny)))
                came_from[(nx, ny)] = current
    return None


def manhattan_path(start: Tuple[int, int], end: Tuple[int, int], rng) -> List[Tuple[int, int]]:
    """
    Simple Manhattan-biased path generator (fallback for A*). Moves preferentially in x
    (or y) direction with some randomness to avoid rigid straight lines.
    """
    path: List[Tuple[int, int]] = []
    x, y = start
    ex, ey = end
    while (x, y) != (ex, ey):
        if x != ex and (y == ey or rng.random() < 0.6):
            x += 1 if ex > x else -1
        elif y != ey:
            y += 1 if ey > y else -1
        if (x, y) not in path:
            path.append((x, y))
    return path


def find_cluster(grid: List[List[tile]], start_x: int, start_y: int,
                 predicate: Callable) -> Set[Tuple[int, int]]:
    """
    Find an orthogonally-connected cluster of tiles starting at (start_x,start_y)
    where predicate(tile) is True. Returns a set of positions in the cluster.
    """
    width = len(grid[0]) if grid else 0
    height = len(grid)

    cluster: Set[Tuple[int, int]] = {(start_x, start_y)}
    queue = deque([(start_x, start_y)])
    while queue:
        x, y = queue.popleft()
        for nx, ny, _ in neighbors_4(x, y, width, height):
            if (nx, ny) in cluster:
                continue
            if predicate(grid[ny][nx]):
                cluster.add((nx, ny))
                queue.append((nx, ny))
    return cluster


# --------------------------
# Lightweight value noise
# --------------------------
def generate_value_noise(width: int, height: int, rng, scale: float = 0.08, octaves: int = 2):
    """
    Generate a simple multi-octave value noise map in range [0,1] suitable for
    inexpensive biome blending. Not Perlin — but cheap and deterministic from RNG.
    """
    # base random seed field
    base = [[rng.random() for _ in range(width)] for _ in range(height)]

    def smooth_once(src):
        out = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                s = 0.0
                c = 0
                # average 3x3 neighborhood (clamped)
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        s += src[ny][nx]
                        c += 1
                out[y][x] = s / c
        return out

    result = [[0.0] * width for _ in range(height)]
    amplitude = 1.0
    total_amp = 0.0

    for _ in range(max(1, octaves)):
        sm = smooth_once(base)
        for y in range(height):
            for x in range(width):
                result[y][x] += sm[y][x] * amplitude
        total_amp += amplitude
        amplitude *= 0.5
        base = smooth_once(base)  # further smooth to simulate higher octave

    # Normalize
    for y in range(height):
        for x in range(width):
            result[y][x] /= max(1e-9, total_amp)
    return result
