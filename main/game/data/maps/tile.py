"""
Tile class definition for map grid.

Each tile represents a single grid cell on the game map, storing its position,
terrain type, size, occupation status, building information, and elevation data.
Tiles are the fundamental building blocks of the game map.
"""


class tile:
    """
    Individual map tile with position, terrain, elevation, and state information.

    Tiles form the grid-based game map. Each tile has a terrain type that affects
    gameplay and can be occupied by a unit. Tiles also store their pixel size
    for rendering calculations, can host buildings for strategic gameplay, and
    have elevation values for height-based tactics.

    Attributes:
        x: Grid X coordinate (column index)
        y: Grid Y coordinate (row index)
        terrain_type: terrain object defining gameplay properties
        size: Tile size in pixels for rendering
        occupied: Boolean indicating if a unit is on this tile
        is_building: Boolean indicating if this tile contains a building
        elevation: Numeric elevation level (0 = ground level)
        is_ramp: Boolean indicating if this tile is a ramp
        ramp_direction: Direction ramp connects (e.g., 'north', 'south', etc.)
    """

    def __init__(
        self,
        x,
        y,
        terrain_type,
        size,
        occupied,
        is_building=False,
        elevation=0,
        is_ramp=False,
        ramp_direction=None
    ):
        """
        Initialize a map tile.

        Args:
            x: X coordinate in grid (0-based column index)
            y: Y coordinate in grid (0-based row index)
            terrain_type: terrain object defining this tile's properties
            size: Pixel size for rendering (typically square)
            occupied: Boolean, True if a unit occupies this tile
            is_building: Boolean, True if a building exists on this tile (default: False)
            elevation: Numeric elevation level, 0 = ground (default: 0)
            is_ramp: Boolean, True if this is a ramp tile (default: False)
            ramp_direction: String direction of ramp connection (default: None)
        """
        self.x = x
        self.y = y
        self.terrain_type = terrain_type
        self.size = size
        self.occupied = occupied
        self.is_building = is_building
        self.elevation = elevation
        self.is_ramp = is_ramp
        self.ramp_direction = ramp_direction

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

    def get_building_status(self):
        """
        Check if this tile contains a building.

        Returns:
            Boolean indicating if a building exists on this tile
        """
        return self.is_building

    def set_building(self, has_building):
        """
        Set whether this tile contains a building.

        Args:
            has_building: Boolean, True if building should exist on this tile
        """
        self.is_building = has_building

    def get_elevation(self):
        """
        Get the elevation level of this tile.

        Returns:
            Integer elevation level (0 = ground level)
        """
        return self.elevation

    def set_elevation(self, elevation_level):
        """
        Set the elevation level of this tile.

        Args:
            elevation_level: Integer elevation level to set
        """
        self.elevation = elevation_level

    def get_ramp_status(self):
        """
        Check if this tile is a ramp.

        Returns:
            Boolean indicating if this is a ramp tile
        """
        return self.is_ramp

    def set_ramp(self, is_ramp, direction=None):
        """
        Set whether this tile is a ramp and its direction.

        Args:
            is_ramp: Boolean, True if this should be a ramp tile
            direction: String direction of ramp ('north', 'south', 'east', 'west')
        """
        self.is_ramp = is_ramp
        self.ramp_direction = direction if is_ramp else None

    def get_ramp_direction(self):
        """
        Get the direction of this ramp tile.

        Returns:
            String direction ('north', 'south', 'east', 'west') or None if not a ramp
        """
        return self.ramp_direction

    def get_coordinates(self):
        """
        Get the grid coordinates of this tile.

        Returns:
            Tuple of (x, y) grid coordinates
        """
        return (self.x, self.y)

    def __repr__(self):
        """
        String representation for debugging.

        Returns:
            String showing tile coordinates, terrain type, occupation, building, and elevation
        """
        occupied_str = "occupied" if self.occupied else "empty"
        building_str = ", building" if self.is_building else ""
        ramp_str = f", ramp({self.ramp_direction})" if self.is_ramp else ""
        elev_str = f", elev={self.elevation}"
        return (
            f"tile(x={self.x}, y={self.y}, terrain={self.terrain_type.name}, "
            f"{occupied_str}{building_str}{ramp_str}{elev_str})"
        )