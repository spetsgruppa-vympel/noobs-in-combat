import os

import pygame
from pygame import display

#||||||||||||||||||||||||||||
# CONFIGURATION
#||||||||||||||||||||||||||||

# ---------------------------
# MAP CONFIG
# ---------------------------

TILE_SIZE = 40  # tile size in pixels
MAP_WIDTH = 20  # number of tiles horizontally
MAP_HEIGHT = 15  # number of tiles vertically

# ---------------------------
# SCREEN CONFIG
# ---------------------------

pygame.init()
info = display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
REFERENCE_SCREEN_WIDTH, REFERENCE_SCREEN_HEIGHT = 1920, 1080
SCREEN_HEIGHT_CONSTANT = 0
SCREEN_WIDTH_CONSTANT = 0

#||||||||||||||||||||||||||||
# CUSTOM EXCEPTIONS
#||||||||||||||||||||||||||||


#||||||||||||||||||||||||||||
# UTIL FUNCTIONS
#||||||||||||||||||||||||||||

# ---------------------------
# COLORED PRINT STATEMENT
# ---------------------------

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
debug = True


def color_print(text, flag=None):
    if flag == "ERROR":
        style = f"{RED}{BOLD}"
    elif flag == "WARNING":
        style = f"{YELLOW}{BOLD}"
    elif flag == "IMPORTANT":
        style = f"{GREEN}{BOLD}"
    else:
        style = ""  # default console style

    print(f"{style}{text}{RESET}")


# ---------------------------
# CONVERT FROM NUMBER TO SCREEN MULTIPLIER
# ---------------------------

def resolution_converter(coord, axis):
    global SCREEN_WIDTH, SCREEN_HEIGHT, REFERENCE_SCREEN_WIDTH, REFERENCE_SCREEN_WIDTH, SCREEN_HEIGHT_CONSTANT, SCREEN_WIDTH_CONSTANT
    if axis == 'x':
        if SCREEN_WIDTH_CONSTANT != 0:  # if not done before, cache the screen width constant for easier things
            SCREEN_WIDTH_CONSTANT = SCREEN_WIDTH / REFERENCE_SCREEN_WIDTH
            color_print(f"Calculated SCREEN_WIDTH_CONSTANT: {SCREEN_WIDTH_CONSTANT} and sending the multiplication.")
            return SCREEN_WIDTH_CONSTANT * coord

    elif axis == 'y':
        multiplier = coord / REFERENCE_SCREEN_HEIGHT * SCREEN_HEIGHT
        print(f"Calculated resolution rescaling, original is {coord}, multiplier is {multiplier}")
        return int(multiplier)
    else:
        raise ValueError("Axis must be 'x' or 'y'")


# ---------------------------
# GET PROJECT ROOT
# ---------------------------
_cached_root = None  # module-level cache for the root dir


def get_project_root(marker="main"):
    # automatically finds the project root by looking for a folder named `marker`.
    # Caches the result for future calls.
    global _cached_root
    if _cached_root:
        return _cached_root

    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, marker)):
            _cached_root = current_dir
            return _cached_root
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # reached filesystem root
            raise FileNotFoundError(f"Could not find project root containing '{marker}'")
        current_dir = parent_dir
