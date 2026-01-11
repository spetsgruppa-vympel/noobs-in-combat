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

    # ========================================================================
    # DIMENSIONS AND SPACING
    # ========================================================================

    # Border width for panels, buttons, and other UI elements
    BORDER_WIDTH = 2  # pixels

    # Corner radius for rounded rectangles
    # Set to 0 for sharp corners, increase for more rounded appearance
    RADIUS = 5  # pixels

    # ========================================================================
    # USAGE EXAMPLES
    # ========================================================================
    """
    # Creating a styled button
    button_rect = pygame.Rect(100, 100, 200, 50)
    pygame.draw.rect(screen, UITheme.PRIMARY, button_rect)
    pygame.draw.rect(screen, UITheme.SECONDARY, button_rect, UITheme.BORDER_WIDTH)
    
    # Hover effect
    if button_hovered:
        pygame.draw.rect(screen, UITheme.HOVER, button_rect)
    
    # Active/selected state
    if button_selected:
        pygame.draw.rect(screen, UITheme.HIGHLIGHT, button_rect)
    
    # Disabled state
    if button_disabled:
        pygame.draw.rect(screen, UITheme.DISABLED, button_rect)
    """