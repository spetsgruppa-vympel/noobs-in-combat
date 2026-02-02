"""
Configuration and utility functions for Snakes in Combat.

This module provides:
- Global configuration constants (screen size, map dimensions)
- Logging setup with automatic file rotation
- Resolution scaling utilities
- Project root path discovery
- Performance monitoring utilities

The configuration system is designed to be imported early and provide
foundational utilities used throughout the application.
"""

import os
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import shutil

import pygame
from pygame import display

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Map dimensions in tiles
TILE_SIZE = 40  # Size of each tile in pixels
MAP_WIDTH = 20  # Horizontal tile count
MAP_HEIGHT = 15  # Vertical tile count

# Screen resolution detection and caching
pygame.init()
info = display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h
REFERENCE_SCREEN_WIDTH, REFERENCE_SCREEN_HEIGHT = 1920, 1080

# Cached resolution multipliers (computed on first use)
_SCREEN_WIDTH_MULTIPLIER = None
_SCREEN_HEIGHT_MULTIPLIER = None


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(project_root):
    """
    Configure comprehensive logging with automatic file management.

    Creates two directories:
    - log_dump/: Current logs (files less than 1 day old)
    - old_log_dump/: Archived logs (files older than 1 day)

    Log files are named: game_YYYYMMDD_HHMMSS.log

    Args:
        project_root (Path): Path to the project root directory

    Returns:
        logging.Logger: Configured logger instance for the application
    """
    # Create log directories
    log_dir = Path(project_root) / "log_dump"
    old_log_dir = Path(project_root) / "old_log_dump"

    log_dir.mkdir(exist_ok=True)
    old_log_dir.mkdir(exist_ok=True)

    # Archive old log files before starting new session
    _archive_old_logs(log_dir, old_log_dir)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"game_{timestamp}.log"
    log_path = log_dir / log_filename

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create application logger
    logger = logging.getLogger('SnakesInCombat')
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger


def _archive_old_logs(log_dir, archive_dir):
    """
    Move log files older than 1 day to archive directory.

    This prevents the log directory from becoming cluttered with old logs
    while preserving them for later analysis if needed.

    Args:
        log_dir (Path): Directory containing current logs
        archive_dir (Path): Directory for archived logs
    """
    cutoff_time = datetime.now() - timedelta(days=1)

    for log_file in log_dir.glob("*.log"):
        # Check file modification time
        file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

        if file_mtime < cutoff_time:
            # Move old log to archive
            dest = archive_dir / log_file.name
            shutil.move(str(log_file), str(dest))
            print(f"Archived old log: {log_file.name}")


def get_logger(name=None):
    """
    Get a logger instance for a specific module.

    This should be called at the top of each module:
        logger = get_logger(__name__)

    Args:
        name (str, optional): Logger name, typically __name__

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name or 'SnakesInCombat')


def get_map_logger():
    """
    Get a specialized logger for map generation.

    This logger is used specifically for map generation operations to
    separate map generation logs from general application logs.

    Returns:
        logging.Logger: Map generation logger instance
    """
    return logging.getLogger('SnakesInCombat.MapGeneration')


class PerformanceTimer:
    """
    Context manager for timing and logging operation duration.

    Usage:
        with PerformanceTimer(logger, "Operation name"):
            # ... code to time ...

    The timer will automatically log the elapsed time when the context exits.

    Attributes:
        logger: Logger instance to use for output
        operation_name: Name of the operation being timed
        start_time: Time when context was entered
    """

    def __init__(self, logger, operation_name):
        """
        Initialize performance timer.

        Args:
            logger: Logger instance for output
            operation_name (str): Name of operation to time
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        """Start timing when entering context."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time when exiting context."""
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(f"Completed: {self.operation_name} in {elapsed:.3f}s")
        return False


def log_memory_usage(logger, label="Memory usage"):
    """
    Log current memory usage.

    Attempts to use psutil for accurate memory reporting. Falls back to
    a simple message if psutil is not available.

    Args:
        logger: Logger instance for output
        label (str): Label prefix for the log message
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        logger.debug(f"{label}: {mem_mb:.1f} MB")
    except ImportError:
        # psutil not available - skip detailed memory logging
        logger.debug(f"{label}: (psutil not available)")
    except Exception as e:
        logger.debug(f"{label}: Error reading memory: {e}")


# ============================================================================
# RESOLUTION UTILITIES
# ============================================================================

def resolution_converter(coord, axis):
    """
    Convert reference coordinates to actual screen coordinates.

    This allows UI elements to be defined using reference dimensions
    (1920x1080) and automatically scale to the actual display resolution.

    The multipliers are cached after first computation for performance.

    Args:
        coord (float): Coordinate value in reference resolution
        axis (str): 'x' for horizontal, 'y' for vertical

    Returns:
        float: Scaled coordinate for actual screen resolution

    Raises:
        ValueError: If axis is not 'x' or 'y'

    Example:
        # Convert 960 pixels (half of 1920) to current screen width
        center_x = resolution_converter(960, 'x')
    """
    global _SCREEN_WIDTH_MULTIPLIER, _SCREEN_HEIGHT_MULTIPLIER

    logger = get_logger(__name__)

    if axis == 'x':
        if _SCREEN_WIDTH_MULTIPLIER is None:
            _SCREEN_WIDTH_MULTIPLIER = SCREEN_WIDTH / REFERENCE_SCREEN_WIDTH
            logger.debug(f"Computed width multiplier: {_SCREEN_WIDTH_MULTIPLIER:.4f}")
        return _SCREEN_WIDTH_MULTIPLIER * coord

    elif axis == 'y':
        if _SCREEN_HEIGHT_MULTIPLIER is None:
            _SCREEN_HEIGHT_MULTIPLIER = SCREEN_HEIGHT / REFERENCE_SCREEN_HEIGHT
            logger.debug(f"Computed height multiplier: {_SCREEN_HEIGHT_MULTIPLIER:.4f}")
        return _SCREEN_HEIGHT_MULTIPLIER * coord

    else:
        logger.error(f"Invalid axis: {axis} (must be 'x' or 'y')")
        raise ValueError("Axis must be 'x' or 'y'")


# ============================================================================
# PROJECT ROOT DISCOVERY
# ============================================================================

_CACHED_PROJECT_ROOT = None  # Module-level cache


def get_project_root(marker="main"):
    """
    Automatically find the project root directory.

    Searches upward from the current file location until it finds a directory
    containing the marker folder. The result is cached for subsequent calls.

    Args:
        marker (str): Directory name to search for (default: "main")

    Returns:
        str: Absolute path to project root directory

    Raises:
        FileNotFoundError: If project root cannot be found

    Example:
        root = get_project_root()  # Finds directory containing "main/" folder
        assets_path = os.path.join(root, "assets", "ui")
    """
    global _CACHED_PROJECT_ROOT

    logger = get_logger(__name__)

    # Return cached value if available
    if _CACHED_PROJECT_ROOT:
        logger.debug(f"Using cached project root: {_CACHED_PROJECT_ROOT}")
        return _CACHED_PROJECT_ROOT

    # Start search from current file's directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    logger.debug(f"Searching for project root from: {current_dir}")

    # Walk up directory tree
    while True:
        marker_path = os.path.join(current_dir, marker)

        if os.path.exists(marker_path):
            _CACHED_PROJECT_ROOT = current_dir
            logger.info(f"Project root found: {_CACHED_PROJECT_ROOT}")
            return _CACHED_PROJECT_ROOT

        # Move to parent directory
        parent_dir = os.path.dirname(current_dir)

        # Check if we've reached filesystem root
        if parent_dir == current_dir:
            logger.error(f"Project root not found (searched for '{marker}' directory)")
            raise FileNotFoundError(
                f"Could not find project root containing '{marker}' directory"
            )

        current_dir = parent_dir