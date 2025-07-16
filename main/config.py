debug = True

# ---------------------------
# MAP CONFIGURATION
# ---------------------------

TILE_SIZE = 40
MAP_WIDTH = 20   # number of tiles horizontally
MAP_HEIGHT = 15  # number of tiles vertically

SCREEN_WIDTH = TILE_SIZE * MAP_WIDTH
SCREEN_HEIGHT = TILE_SIZE * MAP_HEIGHT


def debug_print(text):
        if debug:
            print(text)