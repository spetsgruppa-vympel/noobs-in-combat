"""
Utility functions for map generation (NumPy-accelerated).

Changes from original:
- Added NumPy and vectorized k-nearest neighbor distance computations.
- Rewrote generate_value_noise to use NumPy operations and fast 3x3 smoothing
  (edge behavior uses 'edge' padding which keeps values reasonable near borders).

Behavior and public API are preserved (functions still accept/return the same
types). NumPy is only used where it yields a meaningful performance/clarity
improvement for numeric-heavy operations.
"""

import heapq
import math
from collections import deque
from typing import List, Tuple, Set, Optional, Callable, Iterator
import numpy as np
from main.game.data.maps.tile import tile
from main.config import TILE_SIZE


# ============================================================================
# DIRECTION DEFINITIONS
# ============================================================================

DIRECTIONS_4 = [
    ('north', 0, -1), ('south', 0, 1),
    ('east', 1, 0), ('west', -1, 0)
]

DIRECTIONS_8 = DIRECTIONS_4 + [
    ('northeast', 1, -1), ('southeast', 1, 1),
    ('southwest', -1, 1), ('northwest', -1, -1)
]


# ============================================================================
# GRID OPERATIONS
# ============================================================================

def create_grid(width: int, height: int, terrain_type) -> List[List[tile]]:
    """
    Create a grid (height x width) filled with new tile objects.

    Args:
        width: Grid width in tiles
        height: Grid height in tiles
        terrain_type: Default terrain type for all tiles

    Returns:
        2D list of tile objects [y][x]
    """
    grid: List[List[tile]] = []
    for y in range(height):
        row: List[tile] = []
        for x in range(width):
            t = tile(
                x=x, y=y,
                terrain_type=terrain_type,
                size=TILE_SIZE,
                occupied=False,
                is_building=False,
                elevation=0,
                is_ramp=False,
                ramp_direction=None
            )
            row.append(t)
        grid.append(row)
    return grid


def valid_pos(x: int, y: int, width: int, height: int) -> bool:
    """Check if position is within grid bounds."""
    return 0 <= x < width and 0 <= y < height


def neighbors_4(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Yield cardinal neighbors (nx, ny, direction) within bounds."""
    for direction, dx, dy in DIRECTIONS_4:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


def neighbors_8(x: int, y: int, width: int, height: int) -> Iterator[Tuple[int, int, str]]:
    """Yield 8-connected neighbors (including diagonals) within bounds."""
    for direction, dx, dy in DIRECTIONS_8:
        nx, ny = x + dx, y + dy
        if valid_pos(nx, ny, width, height):
            yield (nx, ny, direction)


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

def manhattan(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan (L1) distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Euclidean (L2) distance between two positions."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return math.sqrt(dx * dx + dy * dy)


def euclidean_squared(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Squared Euclidean distance (faster, for comparisons)."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return dx * dx + dy * dy


def direction_from_offset(dx: int, dy: int) -> Optional[str]:
    """
    Return direction name for given offset.

    Args:
        dx: X offset
        dy: Y offset

    Returns:
        Direction string or None if not exact match
    """
    for direction, ddx, ddy in DIRECTIONS_8:
        if dx == ddx and dy == ddy:
            return direction
    return None


# ============================================================================
# LINE RASTERIZATION (FOR 1-TILE WIDE ROADS)
# ============================================================================

def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm for 1-tile wide paths.

    This is the core algorithm for rasterizing skeleton edges into
    single-tile roads.

    Args:
        start: Starting position (x, y)
        end: Ending position (x, y)

    Returns:
        List of positions forming a line from start to end
    """
    path = []
    x0, y0 = start
    x1, y1 = end

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        path.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return path


def smooth_path(path: List[Tuple[int, int]], radius: int = 2) -> List[Tuple[int, int]]:
    """
    Smooth sharp corners in a path by cutting corners.

    Args:
        path: Original path
        radius: Corner cutting radius

    Returns:
        Smoothed path (may be shorter)
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 1

    while i < len(path) - 1:
        # Check if this is a corner
        prev = path[i - 1]
        curr = path[i]
        next_pos = path[i + 1]

        # Calculate direction change
        dir1 = (curr[0] - prev[0], curr[1] - prev[1])
        dir2 = (next_pos[0] - curr[0], next_pos[1] - curr[1])

        if dir1 != dir2:
            # This is a corner - try to cut it
            # Skip ahead by radius
            skip = min(radius, len(path) - i - 1)
            if skip > 0:
                i += skip
                smoothed.append(path[i])
            else:
                smoothed.append(curr)
        else:
            smoothed.append(curr)

        i += 1

    smoothed.append(path[-1])
    return smoothed


# ============================================================================
# GRAPH ALGORITHMS
# ============================================================================

def kruskal_mst(nodes: List[Tuple[int, int]], edges: List[Tuple[float, int, int]]) -> List[Tuple[int, int]]:
    """
    Kruskal's algorithm for Minimum Spanning Tree.

    Args:
        nodes: List of node positions
        edges: List of (weight, node_idx_a, node_idx_b)

    Returns:
        List of edges in MST as (node_idx_a, node_idx_b)
    """
    n = len(nodes)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Sort edges by weight
    sorted_edges = sorted(edges)

    mst_edges = []
    for weight, i, j in sorted_edges:
        if union(i, j):
            mst_edges.append((i, j))
            if len(mst_edges) == n - 1:
                break

    return mst_edges


def k_nearest_neighbors(nodes: List[Tuple[int, int]], k: int) -> List[Tuple[float, int, int]]:
    """
    Find k-nearest neighbor edges for all nodes.

    Args:
        nodes: List of node positions
        k: Number of nearest neighbors

    Returns:
        List of edges as (distance, node_idx_a, node_idx_b)

    Vectorized with NumPy for clarity and speed on medium/large node sets.
    """
    edges = []
    n = len(nodes)
    if n == 0 or k <= 0:
        return edges

    pts = np.array(nodes, dtype=float)  # shape (n,2)

    # Compute full pairwise distance matrix (n x n)
    # dists[i, j] = distance from i to j
    diff_x = pts[:, 0][:, None] - pts[:, 0][None, :]
    diff_y = pts[:, 1][:, None] - pts[:, 1][None, :]
    dists = np.hypot(diff_x, diff_y)

    # Set diagonal to inf to ignore self
    np.fill_diagonal(dists, np.inf)

    # For each node, select k nearest neighbors
    k = min(k, n - 1)
    for i in range(n):
        # argpartition to get k smallest distances (unsorted)
        nn = np.argpartition(dists[i], k)[:k]
        # sort selected neighbors for deterministic ordering
        nn = nn[np.argsort(dists[i, nn])]
        for j in nn:
            # avoid duplicates: only include if i < j
            if i < int(j):
                edges.append((float(dists[i, int(j)]), i, int(j)))

    return edges


# ============================================================================
# PATHFINDING
# ============================================================================

def bfs_reachable(
    grid: List[List[tile]],
    start: Tuple[int, int],
    passable_fn: Callable,
    can_move_fn: Callable
) -> Set[Tuple[int, int]]:
    """
    Breadth-First Search to find all reachable tiles from start.
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


def find_components(
    grid: List[List[tile]],
    passable_fn: Callable,
    can_move_fn: Callable,
    exclude: Set[Tuple[int, int]]
) -> List[Set[Tuple[int, int]]]:
    """
    Find all connected components in the grid.
    """
    width = len(grid[0]) if grid else 0
    height = len(grid)

    # Find all passable tiles
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


def astar_path(
    grid: List[List[tile]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cost_fn: Callable
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on 4-neighborhood grid.
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


def manhattan_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    rng
) -> List[Tuple[int, int]]:
    """
    Simple Manhattan-biased path (fallback for A*).
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


# ============================================================================
# CLUSTERING
# ============================================================================

def find_cluster(
    grid: List[List[tile]],
    start_x: int,
    start_y: int,
    predicate: Callable
) -> Set[Tuple[int, int]]:
    """
    Find orthogonally-connected cluster matching predicate.
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


def flood_fill(
    start: Tuple[int, int],
    width: int,
    height: int,
    blocked: Set[Tuple[int, int]]
) -> Set[Tuple[int, int]]:
    """
    Flood fill to find contiguous region avoiding blocked tiles.
    """
    region = set()
    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        if (x, y) in blocked:
            continue

        region.add((x, y))

        for nx, ny, _ in neighbors_4(x, y, width, height):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return region


# ============================================================================
# NOISE GENERATION
# ============================================================================

def generate_value_noise(
    width: int,
    height: int,
    rng,
    scale: float = 0.08,
    octaves: int = 2
) -> List[List[float]]:
    """
    Generate multi-octave value noise map in range [0, 1].

    This implementation uses NumPy to accelerate smoothing (3x3 average)
    and accumulation across octaves. The RNG passed in can be any object
    exposing .random() — we build the initial field via Python calls and
    then operate on NumPy arrays for the heavy lifting.

    Returns:
        2D list of noise values [y][x]
    """
    # Build base random field as NumPy array (height x width)
    base = np.empty((height, width), dtype=float)
    for y in range(height):
        for x in range(width):
            base[y, x] = rng.random()

    def smooth_once_np(src: np.ndarray) -> np.ndarray:
        """3x3 average smoothing using edge padding."""
        # pad with edge values so that each interior/outside cell uses 9 values
        p = np.pad(src, pad_width=1, mode='edge')
        # sum of 3x3 neighborhoods using shifted views
        s = (
            p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:]
            + p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
        )
        return s / 9.0

    result = np.zeros((height, width), dtype=float)
    amplitude = 1.0
    total_amp = 0.0

    for _ in range(max(1, int(octaves))):
        sm = smooth_once_np(base)
        result += sm * amplitude
        total_amp += amplitude
        amplitude *= 0.5
        # prepare base for next octave by smoothing further and optionally
        # downsampling could be used, but we simply smooth again to add detail
        base = smooth_once_np(base)

    # Normalize to [0, 1]
    if total_amp > 0:
        result /= total_amp

    # Convert back to list-of-lists for compatibility
    return result.tolist()


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def calculate_angle(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    """
    # Vectors from p2
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    # Dot product and magnitudes
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp for numerical stability

    return math.acos(cos_angle)


def is_diagonal(dx: int, dy: int) -> bool:
    """Check if direction offset is diagonal."""
    return dx != 0 and dy != 0


def cardinal_direction_bias(dx: int, dy: int, bias: float = 0.9) -> float:
    """
    Calculate weight penalty for non-cardinal directions.
    """
    if dx != 0 and dy != 0:
        # Diagonal - apply penalty
        return 1.0 / max(bias, 0.1)
    return 1.0
