import pygame

class UITheme:

    # UI color scheme for menus, theme.json refused to work soo... *shrug* %$$$%$$$%$$$%$$$%$$$%$$$%$$$%$$$%$$$

    BACKGROUND = pygame.Color("#1E1E1E")  # almost black, but not harsh
    PRIMARY = pygame.Color("#3C3C3C")  # dark gray for panels/buttons
    SECONDARY = pygame.Color("#5A5A5A")  # mid-gray for borders/lines
    HIGHLIGHT = pygame.Color("#707070")  # lighter gray for focus/active state
    HOVER = pygame.Color("#505050")  # in-between for hover
    TEXT = pygame.Color("#FFFFFF")  # white text for contrast
    DISABLED = pygame.Color("#2A2A2A")  # very dark gray for disabled

    FONT_MAIN = {"name": "Arial", "size": 24, "bold": True}
    FONT_MENU = {"name": "Arial", "size": 20, "bold": False}
    FONT_BUTTON = {"name": "Arial", "size": 22, "bold": True}

    BORDER_WIDTH = 2
    RADIUS = 5