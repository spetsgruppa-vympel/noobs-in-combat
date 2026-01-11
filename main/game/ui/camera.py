"""
Camera System for Snakes in Combat.

Provides a flexible, view-agnostic camera system that supports panning, zooming,
and target following. Works seamlessly with both 2D orthographic and 2.5D/isometric
projections for map previews and in-game rendering.

The camera system:
    - Transforms world coordinates to screen coordinates
    - Supports smooth panning and zooming
    - Can follow units or tiles automatically
    - Handles viewport boundaries and clamping
    - Provides both 2D and isometric coordinate conversions

Usage:
    # Create camera
    camera = Camera(screen_width=1280, screen_height=720)

    # Pan to position
    camera.pan_to(world_x=100, world_y=100)

    # Follow a unit
    camera.follow_target(unit_x=50, unit_y=50)

    # Transform world to screen
    screen_x, screen_y = camera.world_to_screen(world_x, world_y)

    # Transform screen to world
    world_x, world_y = camera.screen_to_world(screen_x, screen_y)
"""

import math
import pygame
from typing import Tuple, Optional

# ============================================================================
# CAMERA CONSTANTS
# ============================================================================

# Zoom settings
MIN_ZOOM = 0.25  # Minimum zoom level (25% of original size)
MAX_ZOOM = 4.0  # Maximum zoom level (400% of original size)
DEFAULT_ZOOM = 1.0  # Default zoom level (100%)
ZOOM_STEP = 0.1  # Zoom increment per step

# Panning settings
PAN_SMOOTH_FACTOR = 0.15  # Lower = smoother but slower (0.0-1.0)
PAN_THRESHOLD = 0.5  # Distance threshold to stop smooth panning

# Follow settings
FOLLOW_OFFSET_X = 0  # X offset when following target (in world units)
FOLLOW_OFFSET_Y = 0  # Y offset when following target (in world units)

# Isometric projection settings
ISO_TILE_WIDTH = 64  # Width of isometric tile in pixels
ISO_TILE_HEIGHT = 32  # Height of isometric tile in pixels


class CameraMode:
    """
    Enumeration of camera projection modes.

    Attributes:
        ORTHOGRAPHIC: Standard 2D top-down view
        ISOMETRIC: 2.5D isometric projection (diamond view)
    """
    ORTHOGRAPHIC = "orthographic"
    ISOMETRIC = "isometric"


class Camera:
    """
    Flexible camera system supporting multiple projection modes.

    The camera manages viewport transformations between world space (tile
    coordinates) and screen space (pixel coordinates). It supports panning,
    zooming, and automatic target following with smooth interpolation.

    The camera can operate in two modes:
        - ORTHOGRAPHIC: Direct 2D mapping (world units = screen pixels)
        - ISOMETRIC: 2.5D diamond projection for pseudo-3D appearance

    Attributes:
        screen_width: Viewport width in pixels
        screen_height: Viewport height in pixels
        x: Camera center X position in world space
        y: Camera center Y position in world space
        zoom: Current zoom level (1.0 = 100%)
        mode: Projection mode (orthographic or isometric)
        target_x: Target X position for smooth panning (None if not following)
        target_y: Target Y position for smooth panning (None if not following)
        following: Whether camera is following a target
        min_x: Minimum allowed camera X position (None = no limit)
        max_x: Maximum allowed camera X position (None = no limit)
        min_y: Minimum allowed camera Y position (None = no limit)
        max_y: Maximum allowed camera Y position (None = no limit)
    """

    def __init__(
            self,
            screen_width: int,
            screen_height: int,
            start_x: float = 0.0,
            start_y: float = 0.0,
            zoom: float = DEFAULT_ZOOM,
            mode: str = CameraMode.ORTHOGRAPHIC
    ):
        """
        Initialize the camera system.

        Args:
            screen_width: Width of viewport in pixels
            screen_height: Height of viewport in pixels
            start_x: Initial camera X position in world space (default: 0)
            start_y: Initial camera Y position in world space (default: 0)
            zoom: Initial zoom level (default: 1.0)
            mode: Projection mode - orthographic or isometric (default: orthographic)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = start_x
        self.y = start_y
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        self.mode = mode

        # Smooth panning targets
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        self.following = False

        # Boundary limits (None = unlimited)
        self.min_x: Optional[float] = None
        self.max_x: Optional[float] = None
        self.min_y: Optional[float] = None
        self.max_y: Optional[float] = None

    def set_bounds(
            self,
            min_x: Optional[float] = None,
            max_x: Optional[float] = None,
            min_y: Optional[float] = None,
            max_y: Optional[float] = None
    ):
        """
        Set camera movement boundaries in world space.

        Args:
            min_x: Minimum X position (None = no limit)
            max_x: Maximum X position (None = no limit)
            min_y: Minimum Y position (None = no limit)
            max_y: Maximum Y position (None = no limit)
        """
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self._clamp_position()

    def _clamp_position(self):
        """Clamp camera position to defined boundaries."""
        if self.min_x is not None:
            self.x = max(self.x, self.min_x)
        if self.max_x is not None:
            self.x = min(self.x, self.max_x)
        if self.min_y is not None:
            self.y = max(self.y, self.min_y)
        if self.max_y is not None:
            self.y = min(self.y, self.max_y)

    def pan(self, dx: float, dy: float):
        """
        Pan camera by relative offset.

        Args:
            dx: X offset in world space
            dy: Y offset in world space
        """
        self.x += dx
        self.y += dy
        self._clamp_position()
        self.following = False

    def pan_to(self, x: float, y: float, smooth: bool = False):
        """
        Pan camera to absolute position.

        Args:
            x: Target X position in world space
            y: Target Y position in world space
            smooth: If True, interpolate smoothly to position
        """
        if smooth:
            self.target_x = x
            self.target_y = y
        else:
            self.x = x
            self.y = y
            self._clamp_position()
        self.following = False

    def follow_target(self, x: float, y: float):
        """
        Make camera follow a target position continuously.

        The camera will smoothly track the target position until follow
        mode is disabled by panning or calling stop_following().

        Args:
            x: Target X position in world space
            y: Target Y position in world space
        """
        self.target_x = x + FOLLOW_OFFSET_X
        self.target_y = y + FOLLOW_OFFSET_Y
        self.following = True

    def stop_following(self):
        """Stop following the current target."""
        self.following = False
        self.target_x = None
        self.target_y = None

    def zoom_in(self, steps: int = 1):
        """
        Zoom in by specified steps.

        Args:
            steps: Number of zoom steps (default: 1)
        """
        self.set_zoom(self.zoom + (ZOOM_STEP * steps))

    def zoom_out(self, steps: int = 1):
        """
        Zoom out by specified steps.

        Args:
            steps: Number of zoom steps (default: 1)
        """
        self.set_zoom(self.zoom - (ZOOM_STEP * steps))

    def set_zoom(self, zoom: float):
        """
        Set absolute zoom level.

        Args:
            zoom: Zoom level (clamped between MIN_ZOOM and MAX_ZOOM)
        """
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))

    def update(self, delta_time: float = 1.0):
        """
        Update camera state (smooth panning interpolation).

        Call this each frame to enable smooth camera movement. The delta_time
        parameter allows for frame-rate independent movement.

        Args:
            delta_time: Time since last update in seconds (default: 1.0)
        """
        if self.target_x is not None and self.target_y is not None:
            # Calculate distance to target
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx * dx + dy * dy)

            # If close enough, snap to target
            if distance < PAN_THRESHOLD:
                self.x = self.target_x
                self.y = self.target_y
                if not self.following:
                    self.target_x = None
                    self.target_y = None
            else:
                # Smooth interpolation
                self.x += dx * PAN_SMOOTH_FACTOR * delta_time
                self.y += dy * PAN_SMOOTH_FACTOR * delta_time

            self._clamp_position()

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Transform world coordinates to screen coordinates.

        Applies camera position, zoom, and projection mode to convert
        world space coordinates to screen pixel coordinates.

        Args:
            world_x: X position in world space
            world_y: Y position in world space

        Returns:
            Tuple of (screen_x, screen_y) in pixel coordinates
        """
        if self.mode == CameraMode.ISOMETRIC:
            return self._world_to_screen_isometric(world_x, world_y)
        else:
            return self._world_to_screen_orthographic(world_x, world_y)

    def _world_to_screen_orthographic(
            self,
            world_x: float,
            world_y: float
    ) -> Tuple[float, float]:
        """
        Orthographic (2D) world to screen transformation.

        Args:
            world_x: X position in world space
            world_y: Y position in world space

        Returns:
            Tuple of (screen_x, screen_y) in pixel coordinates
        """
        # Apply camera offset and zoom
        relative_x = (world_x - self.x) * self.zoom
        relative_y = (world_y - self.y) * self.zoom

        # Center on screen
        screen_x = relative_x + self.screen_width / 2
        screen_y = relative_y + self.screen_height / 2

        return (screen_x, screen_y)

    def _world_to_screen_isometric(
            self,
            world_x: float,
            world_y: float
    ) -> Tuple[float, float]:
        """
        Isometric (2.5D diamond) world to screen transformation.

        Converts grid coordinates to isometric diamond projection:
            screen_x = (world_x - world_y) * tile_width/2
            screen_y = (world_x + world_y) * tile_height/2

        Args:
            world_x: X position in world space (grid column)
            world_y: Y position in world space (grid row)

        Returns:
            Tuple of (screen_x, screen_y) in pixel coordinates
        """
        # Isometric projection formula
        iso_x = (world_x - world_y) * (ISO_TILE_WIDTH / 2)
        iso_y = (world_x + world_y) * (ISO_TILE_HEIGHT / 2)

        # Apply camera offset and zoom
        relative_x = (iso_x - self.x) * self.zoom
        relative_y = (iso_y - self.y) * self.zoom

        # Center on screen
        screen_x = relative_x + self.screen_width / 2
        screen_y = relative_y + self.screen_height / 2

        return (screen_x, screen_y)

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Transform screen coordinates to world coordinates.

        Inverse operation of world_to_screen. Useful for mouse picking
        and UI interaction with world objects.

        Args:
            screen_x: X position in screen pixels
            screen_y: Y position in screen pixels

        Returns:
            Tuple of (world_x, world_y) in world space coordinates
        """
        if self.mode == CameraMode.ISOMETRIC:
            return self._screen_to_world_isometric(screen_x, screen_y)
        else:
            return self._screen_to_world_orthographic(screen_x, screen_y)

    def _screen_to_world_orthographic(
            self,
            screen_x: float,
            screen_y: float
    ) -> Tuple[float, float]:
        """
        Orthographic (2D) screen to world transformation.

        Args:
            screen_x: X position in screen pixels
            screen_y: Y position in screen pixels

        Returns:
            Tuple of (world_x, world_y) in world space coordinates
        """
        # Remove screen center offset
        relative_x = screen_x - self.screen_width / 2
        relative_y = screen_y - self.screen_height / 2

        # Remove zoom and add camera position
        world_x = (relative_x / self.zoom) + self.x
        world_y = (relative_y / self.zoom) + self.y

        return (world_x, world_y)

    def _screen_to_world_isometric(
            self,
            screen_x: float,
            screen_y: float
    ) -> Tuple[float, float]:
        """
        Isometric (2.5D diamond) screen to world transformation.

        Inverse of isometric projection:
            world_x = (screen_x / tile_width + screen_y / tile_height)
            world_y = (screen_y / tile_height - screen_x / tile_width)

        Args:
            screen_x: X position in screen pixels
            screen_y: Y position in screen pixels

        Returns:
            Tuple of (world_x, world_y) in world space (grid coordinates)
        """
        # Remove screen center offset
        relative_x = screen_x - self.screen_width / 2
        relative_y = screen_y - self.screen_height / 2

        # Remove zoom and camera offset
        iso_x = (relative_x / self.zoom) + self.x
        iso_y = (relative_y / self.zoom) + self.y

        # Inverse isometric projection
        world_x = (iso_x / (ISO_TILE_WIDTH / 2) + iso_y / (ISO_TILE_HEIGHT / 2)) / 2
        world_y = (iso_y / (ISO_TILE_HEIGHT / 2) - iso_x / (ISO_TILE_WIDTH / 2)) / 2

        return (world_x, world_y)

    def is_visible(
            self,
            world_x: float,
            world_y: float,
            margin: float = 0.0
    ) -> bool:
        """
        Check if a world position is visible in the current viewport.

        Useful for culling off-screen objects to improve performance.

        Args:
            world_x: X position in world space
            world_y: Y position in world space
            margin: Extra margin around viewport (default: 0)

        Returns:
            True if position is visible, False otherwise
        """
        screen_x, screen_y = self.world_to_screen(world_x, world_y)

        return (
                -margin <= screen_x <= self.screen_width + margin and
                -margin <= screen_y <= self.screen_height + margin
        )

    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get current viewport boundaries in world space.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in world coordinates
        """
        # Get world coordinates of screen corners
        top_left = self.screen_to_world(0, 0)
        bottom_right = self.screen_to_world(self.screen_width, self.screen_height)

        return (
            top_left[0],
            top_left[1],
            bottom_right[0],
            bottom_right[1]
        )

    def set_mode(self, mode: str):
        """
        Change camera projection mode.

        Args:
            mode: New projection mode (CameraMode.ORTHOGRAPHIC or CameraMode.ISOMETRIC)
        """
        if mode in [CameraMode.ORTHOGRAPHIC, CameraMode.ISOMETRIC]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid camera mode: {mode}")

    def __repr__(self):
        """String representation for debugging."""
        return (
            f"Camera(pos=({self.x:.1f}, {self.y:.1f}), "
            f"zoom={self.zoom:.2f}, mode={self.mode}, "
            f"following={self.following})"
        )