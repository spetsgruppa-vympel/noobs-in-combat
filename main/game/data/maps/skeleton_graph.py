"""
Skeleton Graph Road Generation System.

This module implements the hierarchical road network generation:
1. Generate strategic nodes (cities, junctions)
2. Build sparse skeleton graph (MST + loops)
3. Rasterize to 1-tile wide highways
4. Generate local roads in blocks
5. Detect and fill urban areas

ALL ROADS ARE EXACTLY 1 TILE WIDE - no exceptions.
"""

import math
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
from main.game.data.maps.config import MapConfig
from main.game.data.maps.utils import (
    euclidean, euclidean_squared, manhattan,
    bresenham_line, smooth_path,
    kruskal_mst, k_nearest_neighbors,
    flood_fill, is_diagonal, cardinal_direction_bias
)


class SkeletonGraph:
    """
    Graph structure for road skeleton.

    Represents the strategic highway network before rasterization.
    Nodes are cities/junctions, edges are highways.
    """

    def __init__(self):
        """Initialize empty skeleton graph."""
        self.nodes: List[Tuple[int, int]] = []  # Node positions (x, y)
        self.edges: List[Tuple[int, int]] = []  # Edge pairs (node_idx_a, node_idx_b)
        self.node_types: Dict[int, str] = {}  # node_idx -> "spawn"/"city"/"junction"

    def add_node(self, pos: Tuple[int, int], node_type: str = "junction") -> int:
        """
        Add node to graph.

        Args:
            pos: Node position (x, y)
            node_type: Node type classification

        Returns:
            Index of newly added node
        """
        idx = len(self.nodes)
        self.nodes.append(pos)
        self.node_types[idx] = node_type
        return idx

    def add_edge(self, a: int, b: int):
        """
        Add edge between two nodes.

        Args:
            a: First node index
            b: Second node index
        """
        if (a, b) not in self.edges and (b, a) not in self.edges:
            self.edges.append((a, b))

    def get_neighbors(self, node_idx: int) -> List[int]:
        """
        Get all nodes connected to given node.

        Args:
            node_idx: Node index

        Returns:
            List of connected node indices
        """
        neighbors = []
        for a, b in self.edges:
            if a == node_idx:
                neighbors.append(b)
            elif b == node_idx:
                neighbors.append(a)
        return neighbors

    def __len__(self):
        """Return number of nodes."""
        return len(self.nodes)


class SkeletonGraphGenerator:
    """
    Generator for skeleton graph road networks.

    Orchestrates the entire pipeline from node placement to road rasterization.
    """

    def __init__(self, config: MapConfig, rng, heat_map: List[List[float]]):
        """
        Initialize skeleton graph generator.

        Args:
            config: Map configuration
            rng: Random number generator
            heat_map: Heat map for strategic placement
        """
        self.config = config
        self.rng = rng
        # Keep original heat_map for compatibility but also create a NumPy view
        self.heat_map = heat_map
        self.heat_np = np.array(heat_map) if heat_map is not None else None
        self.graph: Optional[SkeletonGraph] = None

    # ========================================================================
    # PHASE 1: NODE GENERATION
    # ========================================================================

    def generate_nodes(self) -> SkeletonGraph:
        """
        Generate skeleton graph nodes using configured method.

        Returns:
            SkeletonGraph with nodes (no edges yet)
        """
        self.graph = SkeletonGraph()

        # Always add spawn points as nodes
        self.graph.add_node(self.config.spawn_point_1, "spawn")
        self.graph.add_node(self.config.spawn_point_2, "spawn")

        # Generate additional nodes based on method
        if self.config.node_generation_method == "poisson_disc":
            self._generate_poisson_disc_nodes()
        elif self.config.node_generation_method == "grid_noise":
            self._generate_grid_noise_nodes()
        elif self.config.node_generation_method == "city_districts":
            self._generate_city_district_nodes()

        return self.graph

    def _generate_poisson_disc_nodes(self):
        """
        Poisson-disc sampling for evenly distributed nodes.

        Uses Bridson's algorithm to ensure minimum spacing between nodes
        while maintaining organic, non-grid-like placement.
        """
        min_dist = self.config.node_min_spacing
        target_count = self.config.node_count_target
        w, h = self.config.width, self.config.height

        # Grid for fast spatial lookup
        cell_size = min_dist / math.sqrt(2)
        grid_w = int(math.ceil(w / cell_size))
        grid_h = int(math.ceil(h / cell_size))
        grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]

        def grid_pos(x, y):
            return (int(x / cell_size), int(y / cell_size))

        # Initialize with spawn points
        active = []
        for node_pos in self.graph.nodes:
            gx, gy = grid_pos(node_pos[0], node_pos[1])
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                grid[gy][gx] = node_pos
                active.append(node_pos)

        # Generate points
        attempts_per_point = 30
        while active and len(self.graph) < target_count:
            idx = self.rng.randint(0, len(active) - 1)
            point = active[idx]

            found = False
            for _ in range(attempts_per_point):
                # Random point in annulus around current point
                angle = self.rng.random() * 2 * math.pi
                radius = min_dist * (1 + self.rng.random())
                new_x = point[0] + radius * math.cos(angle)
                new_y = point[1] + radius * math.sin(angle)

                if not (0 <= new_x < w and 0 <= new_y < h):
                    continue

                new_point = (int(new_x), int(new_y))
                gx, gy = grid_pos(new_point[0], new_point[1])

                if not (0 <= gx < grid_w and 0 <= gy < grid_h):
                    continue

                # Check neighboring grid cells for conflicts
                valid = True
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ngx, ngy = gx + dx, gy + dy
                        if 0 <= ngx < grid_w and 0 <= ngy < grid_h:
                            neighbor = grid[ngy][ngx]
                            if neighbor is not None:
                                if euclidean(new_point, neighbor) < min_dist:
                                    valid = False
                                    break
                    if not valid:
                        break

                if valid:
                    # Heat-based acceptance
                    heat = self._get_heat(new_point[0], new_point[1])
                    heat_diff = abs(heat - self.config.node_heat_preference)
                    acceptance = 1.0 - (heat_diff / max(0.01, self.config.node_heat_tolerance))
                    acceptance = max(0.0, min(1.0, acceptance))

                    if self.rng.random() < acceptance:
                        grid[gy][gx] = new_point
                        active.append(new_point)
                        self.graph.add_node(new_point, "junction")
                        found = True
                        break

            if not found:
                active.pop(idx)

    def _generate_grid_noise_nodes(self):
        """
        Grid-based nodes with noise perturbation.

        Creates a regular grid then adds random offsets for organic feel.
        Vectorizes scoring using NumPy for faster heat-map ranking.
        """
        spacing = self.config.node_min_spacing
        target = self.config.node_count_target

        # Create grid points (regular grid, with random integer offsets)
        grid_points = []
        # keep offsets integer so positions are tile indices
        max_offset = spacing // 3
        for y in range(spacing, self.config.height, spacing):
            for x in range(spacing, self.config.width, spacing):
                offset_x = self.rng.randint(-max_offset, max_offset)
                offset_y = self.rng.randint(-max_offset, max_offset)
                nx = max(0, min(self.config.width - 1, x + offset_x))
                ny = max(0, min(self.config.height - 1, y + offset_y))
                grid_points.append((nx, ny))

        if not grid_points:
            return

        # Vectorized heat scoring
        pts = np.array(grid_points, dtype=int)
        xs = pts[:, 0]
        ys = pts[:, 1]

        if self.heat_np is not None:
            # clamp coordinates to array bounds (defensive)
            xs_clamped = np.clip(xs, 0, self.heat_np.shape[1] - 1)
            ys_clamped = np.clip(ys, 0, self.heat_np.shape[0] - 1)
            heats = self.heat_np[ys_clamped, xs_clamped].astype(float)
        else:
            heats = np.zeros(len(xs), dtype=float)

        pref = float(self.config.node_heat_preference)
        scores = 1.0 - np.abs(heats - pref)

        # select top scoring points while preserving unique positions
        # get indices sorted descending by score
        order = np.argsort(-scores)

        needed = max(0, target - len(self.graph))
        chosen = []
        seen = set()
        for idx in order:
            if len(chosen) >= needed:
                break
            pos = (int(xs[idx]), int(ys[idx]))
            if pos in seen:
                continue
            seen.add(pos)
            chosen.append(pos)

        for pos in chosen:
            self.graph.add_node(pos, "junction")

    def _generate_city_district_nodes(self):
        """
        Place nodes at biome centers and boundaries.

        Creates nodes at natural map features for more realistic placement.
        """
        # This is a simplified version - full implementation would need biome data
        # For now, just place nodes in a pattern around map center
        cx, cy = self.config.width // 2, self.config.height // 2
        radius = min(self.config.width, self.config.height) // 4

        # Place nodes in circle around center
        num_radial = self.config.node_count_target - len(self.graph)
        for i in range(num_radial):
            angle = (2 * math.pi * i) / num_radial
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            x = max(0, min(self.config.width - 1, x))
            y = max(0, min(self.config.height - 1, y))
            self.graph.add_node((x, y), "city")

    # ========================================================================
    # PHASE 2: GRAPH CONSTRUCTION
    # ========================================================================

    def build_graph(self):
        """
        Build skeleton graph edges using configured method.

        Creates the sparse network of connections between nodes.
        """
        if not self.graph:
            raise RuntimeError("Must generate nodes before building graph")

        if self.config.skeleton_method == "delaunay_mst":
            self._build_delaunay_mst()
        elif self.config.skeleton_method == "nearest_neighbor":
            self._build_nearest_neighbor()
        elif self.config.skeleton_method == "astar_mesh":
            self._build_astar_mesh()

    def _build_delaunay_mst(self):
        """
        Delaunay triangulation → MST → add loops.
        """
        nodes = self.graph.nodes
        n = len(nodes)

        if n < 2:
            return

        # Approximate Delaunay with k-nearest neighbors
        k_nearest = self.config.skeleton_k_nearest
        all_edges = k_nearest_neighbors(nodes, min(k_nearest, n - 1))

        # Apply penalties for undesirable edges
        weighted_edges = []
        for dist, i, j in all_edges:
            weight = dist

            # Straightness bias (penalize non-straight edges)
            dx = abs(nodes[i][0] - nodes[j][0])
            dy = abs(nodes[i][1] - nodes[j][1])

            # Penalize diagonals that aren't clean 45-degree angles
            if dx > 0 and dy > 0:
                # This is diagonal
                if abs(dx - dy) > 2:
                    # Not close to 45 degrees
                    weight *= (1.0 + (1.0 - self.config.skeleton_straightness_bias))
                # Additional diagonal penalty
                weight *= self.config.skeleton_diagonal_penalty

            weighted_edges.append((weight, i, j))

        # Remove duplicates
        edge_set = set()
        unique_edges = []
        for weight, i, j in weighted_edges:
            edge = tuple(sorted([i, j]))
            if edge not in edge_set:
                edge_set.add(edge)
                unique_edges.append((weight, edge[0], edge[1]))

        # Build MST
        mst_edges = kruskal_mst(nodes, unique_edges)

        # Add MST edges to graph
        for i, j in mst_edges:
            self.graph.add_edge(i, j)

        # Add extra edges for loops
        extra_count = int(len(mst_edges) * self.config.skeleton_extra_edges)

        # Find edges not in MST
        mst_set = set((min(i, j), max(i, j)) for i, j in mst_edges)
        remaining = [(w, i, j) for w, i, j in unique_edges
                     if (min(i, j), max(i, j)) not in mst_set]
        remaining.sort()  # Sort by weight

        # Add best remaining edges
        for _, i, j in remaining[:extra_count]:
            self.graph.add_edge(i, j)

    def _build_nearest_neighbor(self):
        """
        Simple nearest-neighbor connection.

        Each node connects to its k nearest neighbors.
        Vectorized distance computation with NumPy for improved clarity and
        performance on medium/large node counts.
        """
        nodes = self.graph.nodes
        n = len(nodes)
        if n < 2:
            return
        k = min(self.config.skeleton_k_nearest, n - 1)

        # Convert node list to NumPy array of shape (n, 2)
        pts = np.array(nodes, dtype=int)

        for i in range(n):
            # compute vectorized distances to all other nodes
            diff = pts - pts[i]  # shape (n,2)
            # Euclidean distances
            dists = np.hypot(diff[:, 0], diff[:, 1])
            dists[i] = np.inf  # ignore self

            # get indices of k smallest distances
            nn_idx = np.argpartition(dists, k)[:k]
            # argpartition does not sort; sort the selected indices for determinism
            nn_idx = nn_idx[np.argsort(dists[nn_idx])]

            for j in nn_idx:
                self.graph.add_edge(i, int(j))

    def _build_astar_mesh(self):
        """
        Connect nodes with A*-like paths.

        Creates more natural-looking curved connections.
        """
        # For now, use nearest neighbor
        # Full A* implementation would require terrain awareness
        self._build_nearest_neighbor()

    # ========================================================================
    # PHASE 3: RASTERIZATION (1-TILE WIDE)
    # ========================================================================

    def rasterize_to_tiles(self) -> Set[Tuple[int, int]]:
        """
        Convert skeleton edges to 1-tile wide highway tiles.

        Returns:
            Set of highway tile positions
        """
        if not self.graph:
            raise RuntimeError("Must build graph before rasterization")

        highways = set()

        for i, j in self.graph.edges:
            start = self.graph.nodes[i]
            end = self.graph.nodes[j]

            # Rasterize edge
            if self.config.highway_rasterization_method == "bresenham":
                path = bresenham_line(start, end)
            else:  # "straight_only"
                path = self._straight_only_line(start, end)

            # Apply smoothing if enabled
            if self.config.highway_curve_smoothing:
                path = smooth_path(path, self.config.highway_corner_radius)

            # Add to highways (all roads are 1 tile wide!)
            for pos in path:
                highways.add(pos)

        return highways

    def _straight_only_line(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Create path using only cardinal directions (no diagonals).

        Args:
            start: Starting position
            end: Ending position

        Returns:
            List of positions forming Manhattan path
        """
        path = []
        x, y = start
        ex, ey = end

        # Move in X direction first
        while x != ex:
            path.append((x, y))
            x += 1 if ex > x else -1

        # Then Y direction
        while y != ey:
            path.append((x, y))
            y += 1 if ey > y else -1

        path.append((ex, ey))
        return path

    # ========================================================================
    # PHASE 4: HIGHWAY PORTALS
    # ========================================================================

    def mark_portals(self, highways: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Mark where local roads can connect to highways.

        Args:
            highways: Set of highway tile positions

        Returns:
            Set of portal positions
        """
        if not self.config.local_road_intersection_portals:
            # All highway tiles are portals
            return highways.copy()

        portals = set()

        # Mark skeleton nodes as portals
        for node_pos in self.graph.nodes:
            if node_pos in highways:
                portals.add(node_pos)

        # Mark regular intervals along highways
        spacing = max(1, self.config.local_road_portal_spacing)
        highway_list = sorted(list(highways))

        # Use slicing for regular intervals (more concise)
        for pos in highway_list[::spacing]:
            portals.add(pos)

        return portals

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _get_heat(self, x: int, y: int) -> float:
        """Get heat map value at position."""
        if self.heat_np is not None:
            if 0 <= x < self.heat_np.shape[1] and 0 <= y < self.heat_np.shape[0]:
                return float(self.heat_np[y, x])
            return 0.0

        # fallback to original list-of-lists behavior
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            return self.heat_map[y][x]
        return 0.0
