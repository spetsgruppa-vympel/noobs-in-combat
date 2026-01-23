"""
Grid and tile utility functions.

Provides helper functions for grid manipulation, pathfinding, and spatial operations.
"""

import heapq
from collections import deque
from typing import List, Tuple, Set, Optional, Callable, Iterator
from main.game.data.maps.tile import tile
from main.config import TILE_SIZE

# Direction mappings
DIRECTIONS_4 = [
    ('north', 0, -1), ('south', 0, 1),
    ('east', 1, 0), ('west', -1, 0)
]

DIRECTIONS_8 = DIRECTIONS_4 + [
    ('northeast', 1, -1), ('southeast', 1, 1),
    ('southwest', -1, 1), ('northwest', -1, -1)
]


def create_grid(width: int, height: int, terrain_type) -> List[List[tile]]:
    """Create empty grid filled with terrain type."""
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            t = tile(x=x, y=y, terrain_type=terrain_type, size=TILE_SIZE,
                     occupied=False, is_building=False, elevation=0,
                     is_ramp=False, ramp_direction=None)
            row.append(t)
        grid.append(row)
    return grid


def valid_pos(x: int, y: int, width: int, height: int) -> bool:
    """Check if position is within bounds."""
    return 0 <= x < width and 0 <= y < height


def neighbors_4(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Get 4-connected neighbors with direction names."""
    for direction, dx, dy in DIRECTIONS_4:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


def neighbors_8(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Get 8-connected neighbors with direction names."""
    for direction, dx, dy in DIRECTIONS_8:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


def manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance between positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def direction_from_offset(dx: int, dy: int) -> Optional[str]:
    """Get direction name from offset."""
    for direction, ddx, ddy in DIRECTIONS_8:
        if dx == ddx and dy == ddy:
            return direction
    return None


def bfs_reachable(grid: List[List[tile]], start: Tuple[int, int],
                  passable_fn: Callable, can_move_fn: Callable) -> Set[Tuple[int, int]]:
    """
    Find all tiles reachable from start using BFS.

    Args:
        grid: Map grid
        start: Starting position
        passable_fn: Function(grid, x, y) -> bool for passability
        can_move_fn: Function(grid, x1, y1, x2, y2) -> bool for movement rules
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
    """Find disconnected components excluding already-visited tiles."""
    width = len(grid[0]) if grid else 0
    height = len(grid)

    all_passable = set()
    for y in range(height):
        for x in range(width):
            if passable_fn(grid, x, y):
                all_passable.add((x, y))

    remaining = all_passable - exclude
    components = []

    while remaining:
        seed = next(iter(remaining))
        component = bfs_reachable(grid, seed, passable_fn, can_move_fn)
        components.append(component)
        remaining -= component

    return components


def astar_path(grid: List[List[tile]], start: Tuple[int, int], goal: Tuple[int, int],
               cost_fn: Callable) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding.

    Args:
        grid: Map grid
        start: Start position
        goal: Goal position
        cost_fn: Function(grid, x, y) -> float for terrain cost
    """
    width = len(grid[0]) if grid else 0
    height = len(grid)

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        x, y = current
        for nx, ny, _ in neighbors_4(x, y, width, height):
            terrain_cost = cost_fn(grid, nx, ny)
            new_cost = cost_so_far[current] + terrain_cost

            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost + manhattan((nx, ny), goal)
                heapq.heappush(frontier, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    return None


def manhattan_path(start: Tuple[int, int], end: Tuple[int, int], rng) -> List[Tuple[int, int]]:
    """Create Manhattan-style path as fallback."""
    path = []
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
    """Find connected cluster of tiles matching predicate."""
    width = len(grid[0]) if grid else 0
    height = len(grid)

    cluster = {(start_x, start_y)}
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