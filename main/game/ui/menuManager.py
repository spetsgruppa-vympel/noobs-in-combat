import os
import pygame
import pygame_gui
from main.config import resolution_converter, color_print, get_project_root

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
        self.manager = pygame_gui.UIManager((screen_width, screen_height))
        color_print("GUI Manager initialized with code-based theme.", "IMPORTANT")

        # placeholders for lazy loaded assets and UI
        self._logo = None
        self._background = None
        self.menu_panel = None
        self._singleplayer_btn = None
        self._multiplayer_btn = None
        self._loadout_btn = None
        self._quit_dialog = None

        # cache static menu surface
        self.static_surface = pygame.Surface((screen_width, screen_height))

        # store UI elements in a list
        self.ui_elements = []  # track created elements

        # create UI
        self.create_main_menu()
        color_print("Main menu created.", "IMPORTANT")
        self.draw_static()  # pre-draw static elements


    # ---------------------------
    # loaded assets
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
    # lazy-loaded buttons
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
    # menu creation
    # ---------------------------



    def create_main_menu(self):
        # ---------------------------
        # PANEL (reference: 400x350 centered on 1920x1080)
        # ---------------------------
        panel_width = resolution_converter(400, 'x')
        panel_height = resolution_converter(350, 'y')
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 1.5

        self.menu_panel = self.init_ui(
            "panel",
            pygame.Rect(panel_x, panel_y, panel_width, panel_height),
            self.manager,
            object_id="#menu_panel"
        )

        # ---------------------------
        # BUTTONS (1920x1080)
        # reference resolution values:
        # width=300, height=60, margin_y=40
        # ---------------------------

        button_width = resolution_converter(300, 'x')
        button_height = resolution_converter(60, 'y')
        button_x = (panel_width - button_width) // 2
        start_y = resolution_converter(40, 'y')

        # singleplayer button
        self._singleplayer_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y, button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Singleplayer",
            object_id="#singleplayer_btn"
        )
        self.color_print("Singleplayer button created.", "IMPORTANT")

        # multiplayer button
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

        # loadout button
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

    def init_ui(self, element_type, rect, manager, container=None, **kwargs):
        import pygame_gui.elements as elements
        import pygame_gui.windows as windows
        from .UItheme import UITheme

        element_map = {
            "button": elements.UIButton,
            "slider": elements.UIHorizontalSlider,
            "label": elements.UILabel,
            "dropdown": elements.UIDropDownMenu,
            "panel": elements.UIPanel,
            "textbox": elements.UITextBox,
            "confirmation_dialog": windows.UIConfirmationDialog
        }

        ui_class = element_map[element_type]

        # confirmation dialogs use different params than normal elements
        if element_type == "confirmation_dialog":
            element = ui_class(
                rect=rect,
                manager=manager,
                **kwargs
            )
            # style background / border like panels
            element.background_colour = UITheme.BACKGROUND
            element.border_colour = UITheme.SECONDARY
            element.rebuild()
            return element

        # --- standard element creation ---
        params = {
            "relative_rect": rect,
            "manager": manager,
            **kwargs
        }
        if container:
            params["container"] = container

        element = ui_class(**params)
        self.ui_elements.append(element)

        # ---- apply theme styling ----
        if element_type == "button":
            element.colours.update({
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY,
                "hovered_border": UITheme.HIGHLIGHT
            })
            element.rebuild()

        elif element_type in ("dropdown", "slider"):
            element.colours.update({
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY
            })
            element.rebuild()

        elif element_type == "panel":
            element.background_colour = UITheme.BACKGROUND
            element.border_colour = UITheme.SECONDARY
            element.rebuild()

        elif element_type == "label":
            if "text" in kwargs:
                element.set_text(
                    f'<font color="{UITheme.TEXT}">{kwargs["text"]}</font>'
                )

        elif element_type == "textbox":
            element.set_text(
                f'<font color="{UITheme.TEXT}">{kwargs.get("html_text", "")}</font>'
            )

        return element

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

    # ------------------------
    # CLEAR UI DEFINITION
    # ------------------------

    def clear_element(self,
                 element_clear, # which element(s) to be cleared
                 ):
        if element_clear == "all":
            for element in self.ui_elements:
                element.kill()
            self.ui_elements.clear()

            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
            self.menu_panel = None
            self._quit_dialog = None

    # ------------------------
    # BUTTON CALLBACK FUNCTIONS
    # ------------------------

    def singleplayer_press(self):
        color_print("singleplayer button pressed", "IMPORTANT")
        pass
        # TODO: starting singleplayer logic here

    def multiplayer_press(self):
        color_print("multiplayer button pressed", "IMPORTANT")
        pass
        # TODO: starting multiplayer logic here

    def loadout_press(self):
        color_print("loadout button pressed", "IMPORTANT")
        pass
        # TODO: starting loadout logic here


    # ---------------------------
    # EVENT PROCESSING
    # ---------------------------

    def process_events(self, events):

        # quit dialog sizes
        dialog_w, dialog_h = resolution_converter(330, "x"), resolution_converter(200, "y")
        dialog_x = (self.screen_width - dialog_w) // 2
        dialog_y = (self.screen_height - dialog_h) // 2

        def init_quit_dialog():
            if self._quit_dialog is None:
                self._quit_dialog = self.init_ui(
                    "confirmation_dialog",
                    pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h),
                    self.manager,
                    window_title="Confirm Quit",
                    action_long_desc="Are you sure you want to quit?",
                    action_short_name="Quit",
                    blocking=True
                )

        for event in events:
            self.manager.process_events(event)

            # button presses
            if event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self._singleplayer_btn:
                    self.singleplayer_press()
                elif event.ui_element == self._multiplayer_btn:
                    self.multiplayer_press()
                elif event.ui_element == self._loadout_btn:
                    self.loadout_press()

            elif event.type in (pygame.KEYDOWN, pygame.QUIT):
                if event.type == pygame.QUIT:
                    # noinspection PyUnreachableCode
                    if self._quit_dialog:
                        # this code IS in fact accessible pycharm you stupid-
                        print("quit")
                        return "quit"
                if event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                    continue
                init_quit_dialog()

            # quit confirmed
            elif event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None
                    return "quit"

            # quit dialog closed without confirming
            elif event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None

        return None