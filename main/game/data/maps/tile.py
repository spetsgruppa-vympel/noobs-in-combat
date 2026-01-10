"""
Tile class definition for map grid.

Each tile represents a single grid cell on the game map, storing its position,
terrain type, size, and occupation status. Tiles are the fundamental building
blocks of the game map.
"""


class tile:
    """
    Individual map tile with position, terrain, and state information.

    Tiles form the grid-based game map. Each tile has a terrain type that affects
    gameplay and can be occupied by a unit. Tiles also store their pixel size
    for rendering calculations.

    Attributes:
        x: Grid X coordinate (column index)
        y: Grid Y coordinate (row index)
        terrain_type: terrain object defining gameplay properties
        size: Tile size in pixels for rendering
        occupied: Boolean indicating if a unit is on this tile
    """

    def __init__(self, x, y, terrain_type, size, occupied):
        """
        Initialize a map tile.

        Args:
            x: X coordinate in grid (0-based column index)
            y: Y coordinate in grid (0-based row index)
            terrain_type: terrain object defining this tile's properties
            size: Pixel size for rendering (typically square)
            occupied: Boolean, True if a unit occupies this tile
        """
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.size = size
        self.occupied = occupied

    def get_terrain_type(self):
        """
        Get the terrain type of this tile.

        Returns:
            terrain object representing this tile's terrain
        """
        return self.terrain_type

    def get_occupation(self):
        """
        Check if this tile is occupied by a unit.

        Returns:
            Boolean indicating occupation status
        """
        return self.occupied

    def change_occupation(self):
        """
        Toggle the occupation status of this tile.

        Used when units move onto or off of this tile.
        """
        self.occupied = not self.occupied

    def set_occupation(self, is_occupied):
        """
        Explicitly set the occupation status.

        Args:
            is_occupied: Boolean occupation status to set
        """
        self.occupied = is_occupied

    def get_coordinates(self):
        """
        Get the grid coordinates of this tile.

        Returns:
            Tuple of (x, y) grid coordinates
        """
        return (self.x, self.y)

    def __repr__(self):
        """String representation for debugging."""
        occupied_str = "occupied" if self.occupied else "empty"
        return (
            f"tile(x={self.x}, y={self.y}, terrain={self.terrain_type.name}, "
            f"{occupied_str})"
        )