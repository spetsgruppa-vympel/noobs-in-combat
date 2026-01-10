"""
UI Theme configuration for Snakes in Combat.

Centralizes all UI styling constants including colors, fonts, and dimensions.
Uses a dark theme with high contrast for readability and visual appeal.
"""

import pygame


class UITheme:
    """
    Centralized UI theme constants for consistent styling across all menus.

    All colors use pygame.Color for consistent color management.
    Font configurations are stored as dictionaries for easy passing to UI elements.

    Color Palette:
        - BACKGROUND: Near-black for main background (#1E1E1E)
        - PRIMARY: Dark gray for panels and buttons (#3C3C3C)
        - SECONDARY: Mid-gray for borders and dividers (#5A5A5A)
        - HIGHLIGHT: Light gray for focused/active elements (#707070)
        - HOVER: In-between gray for hover states (#505050)
        - TEXT: White for maximum contrast (#FFFFFF)
        - DISABLED: Very dark gray for disabled elements (#2A2A2A)
    """

    # Core color palette
    BACKGROUND = pygame.Color("#1E1E1E")  # Main background
    PRIMARY = pygame.Color("#3C3C3C")     # Panels and buttons
    SECONDARY = pygame.Color("#5A5A5A")   # Borders and lines
    HIGHLIGHT = pygame.Color("#707070")   # Active/focused state
    HOVER = pygame.Color("#505050")       # Hover state
    TEXT = pygame.Color("#FFFFFF")        # Text color
    DISABLED = pygame.Color("#2A2A2A")    # Disabled elements

    # Font configurations (name, size, bold flag)
    FONT_MAIN = {"name": "Arial", "size": 24, "bold": True}
    FONT_MENU = {"name": "Arial", "size": 20, "bold": False}
    FONT_BUTTON = {"name": "Arial", "size": 22, "bold": True}

    # UI element dimensions and styling
    BORDER_WIDTH = 2  # Standard border width in pixels
    RADIUS = 5        # Corner radius for rounded elements in pixels