# ---------------------------
# DEBUG
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
# MAP CONFIGURATION
# ---------------------------

TILE_SIZE = 40  # tile size in pixels
MAP_WIDTH = 20   # number of tiles horizontally
MAP_HEIGHT = 15  # number of tiles vertically

# ---------------------------
# CUSTOM EXCEPTIONS
# ---------------------------
