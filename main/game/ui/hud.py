import os

import pygame
import ctypes
import platform
import pygame_gui
from main.config import color_print


def run_hud():
    pygame.init()

    color_print("Initializing HUD...", "IMPORTANT")

    # get current monitor resolution
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h

    # create a resizable window with monitor's resolution
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    pygame.display.set_caption("Noobs in Combat")

    script_dir = os.path.dirname(os.path.abspath(__file__))  # folder where script is located
    theme_path = os.path.join(script_dir, "theme.json")

    # initialize pygame_gui UIManager with calculated resolution
    manager = pygame_gui.UIManager((screen_width, screen_height), theme_path)
    print(theme_path)

    # maximize window via OS API
    if platform.system() == "Windows":
        wm_info = pygame.display.get_wm_info()
        hwnd = wm_info.get("window")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            color_print("Window maximized successfully.", "IMPORTANT")
        else:
            color_print("Could not obtain window handle to maximize.", "ERROR")
    else:  # raise error if non-windows OS
        color_print("OS NOT WINDOWS", "ERROR")
        raise OSError("ONLY WORKS ON WINDOWS")

    running = True

    # clock for delta time and FPS limiting
    clock = pygame.time.Clock()

    while running:
        # dt in seconds
        dt = clock.tick(60) / 1000.0  # delta time, limits FPS to 60

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                color_print("Exit requested by user.", "IMPORTANT")
                running = False

        # fill screen with background
        screen.fill((30, 30, 30))
        # everything else about the HUD should be put here
        # update display
        pygame.display.flip()

    color_print("Shutting down HUD.", "IMPORTANT")

# map element names to their pygame_gui classes
ELEMENT_MAP = {
    "button": pygame_gui.elements.UIButton,
    "slider": pygame_gui.elements.UIHorizontalSlider,
    "label": pygame_gui.elements.UITextBox,
    "dropdown": pygame_gui.elements.UIDropDownMenu
}

def init_ui(          # initializes UI elements to be drawn in main loop
    element_type,     # 'button', 'slider', 'label', etc.
    rect,             # tuple (x, y, width, height)
    manager,          # the pygame_gui UIManager
    **kwargs          # additional type-specific arguments
):
    ui_class = ELEMENT_MAP[element_type]

    # include rect and manager in the forwarded arguments
    params = {
        "relative_rect": rect,
        "manager": manager,
        **kwargs
    }

    return ui_class(**params)