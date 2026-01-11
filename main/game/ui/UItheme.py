"""
UI Theme configuration for Snakes in Combat.

Provides centralized styling constants for consistent UI appearance across
all menus and game interfaces. Uses a dark theme with high contrast for
readability and visual clarity.

The theme system ensures:
- Consistent colors throughout the application
- Easy theme modifications (change once, apply everywhere)
- Accessibility through high contrast
- Professional appearance

Usage:
    from UItheme import UITheme

    button.background_colour = UITheme.PRIMARY
    button.border_colour = UITheme.SECONDARY
    label.text_colour = UITheme.TEXT
"""

import pygame


class UITheme:
    """
    Centralized UI theme with dark color palette.

    This class serves as a namespace for UI constants. All attributes are
    class-level for easy access without instantiation.

    Color Philosophy:
        - Dark backgrounds reduce eye strain during long play sessions
        - High contrast ensures readability
        - Grayscale palette keeps focus on gameplay, not flashy UI
        - Subtle gradations (6 shades of gray) provide depth

    Color Palette (darkest to lightest):
        BACKGROUND  (#1E1E1E) - Near-black base layer
        DISABLED    (#2A2A2A) - Very dark for disabled elements
        PRIMARY     (#3C3C3C) - Dark gray for panels/buttons
        HOVER       (#505050) - Mid-gray for hover feedback
        SECONDARY   (#5A5A5A) - Light-mid gray for borders
        HIGHLIGHT   (#707070) - Light gray for active elements
        TEXT        (#FFFFFF) - Pure white for maximum contrast
    """

    # ========================================================================
    # COLOR DEFINITIONS
    # ========================================================================

    # Base layer - main window background
    BACKGROUND = pygame.Color("#1E1E1E")

    # UI element backgrounds and surfaces
    PRIMARY = pygame.Color("#3C3C3C")      # Buttons, panels, dropdowns

    # Borders, dividers, and outlines
    SECONDARY = pygame.Color("#5A5A5A")

    # Active/focused element state
    HIGHLIGHT = pygame.Color("#707070")

    # Hover state (between PRIMARY and HIGHLIGHT)
    HOVER = pygame.Color("#505050")

    # Text color (maximum contrast against dark backgrounds)
    TEXT = pygame.Color("#FFFFFF")

    # Disabled element state (subtle, receding)
    DISABLED = pygame.Color("#2A2A2A")

    # ========================================================================
    # TILE HOVER EFFECTS
    # ========================================================================

    # Darkening factor for tile hover effect (0.0 = black, 1.0 = no change)
    TILE_HOVER_DARKEN_FACTOR = 0.6

    @staticmethod
    def darken_color(color, factor=0.6):
        """
        Darken a color by a specified factor.

        Multiplies each RGB component by the factor to create a darker shade.
        Used for tile hover effects in the map visualization.

        Args:
            color: pygame.Color or tuple (r, g, b) to darken
            factor: Float between 0.0 (black) and 1.0 (original color)

        Returns:
            pygame.Color object with darkened RGB values

        Example:
            >>> dark_green = UITheme.darken_color((0, 255, 0), 0.5)
            >>> # Returns Color(0, 127, 0)
        """
        if isinstance(color, tuple):
            color = pygame.Color(*color)

        r = int(color.r * factor)
        g = int(color.g * factor)
        b = int(color.b * factor)

        return pygame.Color(r, g, b)

    # ========================================================================
    # TYPOGRAPHY
    # ========================================================================

    # Font configurations stored as dictionaries for easy unpacking
    # Usage: pygame.font.Font(**UITheme.FONT_MAIN)

    # Main title font (large, bold)
    FONT_MAIN = {
        "name": "Arial",
        "size": 24,
        "bold": True
    }

    # Menu option font (medium, regular)
    FONT_MENU = {
        "name": "Arial",
        "size": 20,
        "bold": False
    }

    # Button label font (medium-large, bold)
    FONT_BUTTON = {
        "name": "Arial",
        "size": 22,
        "bold": True
    }

    # Tile label font (small, bold) - for terrain type labels on hover
    FONT_TILE_LABEL = {
        "name": "Arial",
        "size": 12,
        "bold": True
    }

    # ========================================================================
    # DIMENSIONS AND SPACING
    # ========================================================================

    # Border width for panels, buttons, and other UI elements
    BORDER_WIDTH = 2  # pixels

    # Corner radius for rounded rectangles
    # Set to 0 for sharp corners, increase for more rounded appearance
    RADIUS = 5  # pixels

