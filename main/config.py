import os
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import shutil

import pygame
from pygame import display

# ||||||||||||||||||||||||||||
# CONFIGURATION
# ||||||||||||||||||||||||||||

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
# cache the constants
SCREEN_HEIGHT_CONSTANT = False
SCREEN_WIDTH_CONSTANT = False


# ||||||||||||||||||||||||||||
# LOGGING CONFIGURATION
# ||||||||||||||||||||||||||||

def setup_logging(project_root):
    """
    Set up comprehensive logging with automatic file management.

    Args:
        project_root: Path to the project root directory
    """
    # Create log directories
    log_dir = Path(project_root) / "log_dump"
    old_log_dir = Path(project_root) / "old_log_dump"

    log_dir.mkdir(exist_ok=True)
    old_log_dir.mkdir(exist_ok=True)

    # Move old log files (older than 1 day) to old_log_dump
    _cleanup_old_logs(log_dir, old_log_dir)

    # Create log filename with timestamp
    log_filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    # Configure root logger with custom format
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )

    # Create and return logger
    logger = logging.getLogger('SnakesInCombat')
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger


def _cleanup_old_logs(log_dir, old_log_dir):
    """Move log files older than 1 day to old_log_dump directory."""
    now = datetime.now()
    cutoff_time = now - timedelta(days=1)

    for log_file in log_dir.glob("*.log"):
        # Get file modification time
        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

        # Move if older than 1 day
        if file_mtime < cutoff_time:
            dest = old_log_dir / log_file.name
            shutil.move(str(log_file), str(dest))
            print(f"Moved old log file: {log_file.name} to old_log_dump")


def get_logger(name=None):
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        logging.Logger instance
    """
    return logging.getLogger(name or 'SnakesInCombat')


# ||||||||||||||||||||||||||||
# UTIL FUNCTIONS
# ||||||||||||||||||||||||||||

# ---------------------------
# CONVERT FROM NUMBER TO SCREEN MULTIPLIER
# ---------------------------

def resolution_converter(coord, axis):
    """
    Convert from reference size to actual screen size.

    Args:
        coord: Coordinate value to convert
        axis: 'x' or 'y' axis

    Returns:
        Converted coordinate value
    """
    global SCREEN_WIDTH, SCREEN_HEIGHT, REFERENCE_SCREEN_WIDTH, REFERENCE_SCREEN_HEIGHT
    global SCREEN_HEIGHT_CONSTANT, SCREEN_WIDTH_CONSTANT

    logger = get_logger(__name__)

    if axis == 'x':
        if not SCREEN_WIDTH_CONSTANT:
            SCREEN_WIDTH_CONSTANT = SCREEN_WIDTH / REFERENCE_SCREEN_WIDTH
            logger.debug(f"Calculated SCREEN_WIDTH_CONSTANT: {SCREEN_WIDTH_CONSTANT}")
        return SCREEN_WIDTH_CONSTANT * coord

    elif axis == 'y':
        if not SCREEN_HEIGHT_CONSTANT:
            SCREEN_HEIGHT_CONSTANT = SCREEN_HEIGHT / REFERENCE_SCREEN_HEIGHT
            logger.debug(f"Calculated SCREEN_HEIGHT_CONSTANT: {SCREEN_HEIGHT_CONSTANT}")
        return SCREEN_HEIGHT_CONSTANT * coord

    else:
        logger.error(f"Invalid axis parameter: {axis}. Must be 'x' or 'y'")
        raise ValueError("Axis must be 'x' or 'y'")


# ---------------------------
# GET PROJECT ROOT
# ---------------------------
_cached_root = None  # module-level cache for the root dir


def get_project_root(marker="main"):
    """
    Automatically find the project root by looking for a folder named `marker`.
    Caches the result for future calls.

    Args:
        marker: Directory name to search for (default: "main")

    Returns:
        Path to project root directory

    Raises:
        FileNotFoundError: If project root cannot be found
    """
    global _cached_root

    logger = get_logger(__name__)

    if _cached_root:
        logger.debug(f"Using cached project root: {_cached_root}")
        return _cached_root

    current_dir = os.path.abspath(os.path.dirname(__file__))
    logger.debug(f"Starting project root search from: {current_dir}")

    while True:
        marker_path = os.path.join(current_dir, marker)
        if os.path.exists(marker_path):
            _cached_root = current_dir
            logger.info(f"Project root found: {_cached_root}")
            return _cached_root

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # reached filesystem root
            logger.error(f"Could not find project root containing '{marker}'")
            raise FileNotFoundError(f"Could not find project root containing '{marker}'")
        current_dir = parent_dir