class tile:
    def __init__(self, x, y, terrainType, size):

        # x coordinate of the tile
        self.x = x

        # y coordinate of the tile
        self.y = y

        # the type of terrain (urban, forest, plains, road, etc) the tile is
        self.terrainType = terrainType

        # the size of the tile in pixels
        self.size = size

    def get_terrain_type(self):
