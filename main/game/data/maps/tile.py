class tile:
    def __init__(self, x, y, terrain_type, size, occupied):

        # x coordinate of the tile
        self.x = x

        # y coordinate of the tile
        self.y = y

        # the type of terrain (urban, forest, plains, road, etc) the tile is
        self.terrain_type = terrain_type

        # the size of the tile in pixels
        self.size = size

        # whether the tile is occupied
        self.occupied = occupied

    def get_terrain_type(self):
        terrain_type = self.terrain_type
        return terrain_type

    def get_occupied(self):
        is_occupied = self.occupied
        return is_occupied

    def change_occupation(self):  # changes the occupation status of the tile
        self.occupied = not self.occupied