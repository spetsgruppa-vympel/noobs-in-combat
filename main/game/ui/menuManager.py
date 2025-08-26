import os
import pygame
import pygame_gui
from main.config import resolution_converter, color_print, get_project_root

# ------------------------
# BUTTON CALLBACK FUNCTIONS
# ------------------------

def singleplayer_press():
    pass
    # TODO: starting singleplayer logic here

def multiplayer_press():
    pass
    # TODO: starting multiplayer logic here

def loadout_press():
    pass
    # TODO: starting loadout logic here

# ------------------------
# MENUMANAGER CLASS
# ------------------------

class MenuManager:
    # fraction of screen height above center for logo placement
    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color_print = color_print

        # GUI manager with theme
        theme_path = os.path.join(get_project_root(), "assets", "theme.json")
        self.manager = pygame_gui.UIManager((screen_width, screen_height), theme_path)
        color_print("GUI Manager initialized with theme.json", "IMPORTANT")

        # placeholders for lazy loaded assets and UI
        self._logo = None
        self._background = None
        self.menu_panel = None
        self._singleplayer_btn = None
        self._multiplayer_btn = None
        self._loadout_btn = None
        self.quit_dialog = None

        # cache static menu surface
        self.static_surface = pygame.Surface((screen_width, screen_height))

        # create UI
        self.create_main_menu()
        color_print("Main menu created.", "IMPORTANT")
        self.draw_static()  # pre-draw static elements

    # ---------------------------
    # Lazy-loaded assets
    # ---------------------------
    @property
    def logo(self):
        if self._logo is None:
            logo_path = os.path.join(get_project_root(), "assets", "ui", "game_icon.jpg")
            self._logo = pygame.image.load(logo_path).convert_alpha()
            self.color_print("Logo loaded.", "IMPORTANT")
            # scale logo to 20% of screen width
            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
            self.color_print(f"Logo scaled to {w}x{h}", "IMPORTANT")
        return self._logo

    @property
    def background(self):
        if self._background is None:
            bg_path = os.path.join(get_project_root(), "assets", "ui", "menu_background.jpg")
            bg = pygame.image.load(bg_path).convert()
            self.color_print("Background loaded.", "IMPORTANT")
            # scale + blur background
            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
            self.color_print("Background scaled and blurred.", "IMPORTANT")
        return self._background

    # ---------------------------
    # Lazy-loaded buttons
    # ---------------------------
    @property
    def singleplayer_btn(self):
        return self._singleplayer_btn

    @property
    def multiplayer_btn(self):
        return self._multiplayer_btn

    @property
    def loadout_btn(self):
        return self._loadout_btn

    # ---------------------------
    # Menu creation
    # ---------------------------
    def create_main_menu(self):
        # ---------------------------
        # PANEL (reference: 400x350 centered on 1920x1080)
        # ---------------------------
        panel_width = resolution_converter(400, 'x')
        panel_height = resolution_converter(350, 'y')
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2

        self.menu_panel = self.init_ui(
            "panel",
            pygame.Rect(panel_x, panel_y, panel_width, panel_height),
            self.manager,
            object_id="#menu_panel"
        )
        self.menu_panel.background_colour = pygame.Color(50, 50, 50, 200)

        # ---------------------------
        # BUTTONS
        # Reference resolution values:
        # width=300, height=60, margin_y=40
        # ---------------------------

        button_width = resolution_converter(300, 'x')
        button_height = resolution_converter(60, 'y')
        button_x = (panel_width - button_width) // 2
        start_y = resolution_converter(40, 'y')

        # Singleplayer button
        self._singleplayer_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y, button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Singleplayer",
            object_id="#singleplayer_btn"
        )
        self.color_print("Singleplayer button created.", "IMPORTANT")

        # Multiplayer button
        self._multiplayer_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y + int(button_height * 1.1),
                        button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Multiplayer",
            object_id="#multiplayer_btn"
        )
        self.color_print("Multiplayer button created.", "IMPORTANT")

        # Loadout button
        self._loadout_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y + int(button_height * 2.2),
                        button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Loadout",
            object_id="#loadout_btn"
        )
        self.color_print("Loadout button created.", "IMPORTANT")

    # ---------------------------
    # UI INITIALIZATION HELPER
    # ---------------------------

    @staticmethod
    def init_ui(element_type, rect, manager, container=None, **kwargs):
        import pygame_gui.elements as elements
        element_map = {
            "button": elements.UIButton,
            "slider": elements.UIHorizontalSlider,
            "label": elements.UITextBox,
            "dropdown": elements.UIDropDownMenu,
            "panel": elements.UIPanel
        }
        ui_class = element_map[element_type]
        params = {
            "relative_rect": rect,
            "manager": manager,
            **kwargs
        }
        if container:
            params["container"] = container
        return ui_class(**params)

    # ---------------------------
    # DRAW MENU
    # ---------------------------

    def draw_static(self):
        # draws static background + logo once onto a surface
        self.static_surface.blit(self.background, (0, 0))
        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 - logo_offset)
        )
        self.static_surface.blit(self.logo, logo_rect)

    def draw(self):
        # blit pre-rendered background/logo + GUI elements.
        self.screen.blit(self.static_surface, (0, 0))
        self.manager.draw_ui(self.screen)

    # ---------------------------
    # EVENT PROCESSING
    # ---------------------------
    def process_events(self, events):
        # process Pygame events through GUI manager
        for event in events:
            self.manager.process_events(event)