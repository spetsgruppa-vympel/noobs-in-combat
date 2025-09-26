import os
import pygame
import pygame_gui
from main.config import resolution_converter, color_print, get_project_root

class MenuManager:
    # fraction of screen height above center where the logo will be placed
    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height, theme=None):
        # store display surface and dimensions
        self.current_menu = None
        self.menu_stack = []
        self.singleplayer_main_spec = None
        self.menu_spec = None
        self.theme = theme
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height

        # keep handy logging function
        self.color_print = color_print

        # create the pygame_gui UI manager for this screen size
        # allow passing a custom theme file/manager later (kept for compatibility)
        self.manager = pygame_gui.UIManager((screen_width, screen_height))
        self.color_print("GUI Manager initialized with code-based theme.", "IMPORTANT")

        # lazy-loaded assets (None until first accessed)
        self._logo = None
        self._background = None
        self._icon = None

        # top-level containers / important widget refs (set when created)
        self.menu_panel = None
        self._quit_dialog = None

        # placeholders for convenience (can be populated by spec creation)
        self._singleplayer_btn = None
        self._multiplayer_btn = None
        self._loadout_btn = None

        # Surface used to cache static drawing (background + logo)
        self.static_surface = pygame.Surface((screen_width, screen_height))

        # list of UI elements for bulk operations (kill/hide)
        self.ui_elements = []

        # element instance -> callback mapping for fast event dispatch
        self.element_callbacks = {}

        # create the UI using a data-driven spec approach
        # original behavior preserved: create_main_menu will build a spec and call create_from_spec
        self.create_main_menu()
        self.color_print("Main menu created (spec)", "IMPORTANT")

        pygame.display.set_icon(self.logo)
        self.color_print("Window icon set from logo", "IMPORTANT")

        # pre-render static content once
        self.draw_static()

    # ---------------------------
    # ASSETS (lazy-loaded)
    # ---------------------------
    @property
    def logo(self):
        # load and cache the logo image when first accessed
        if self._logo is None:
            logo_path = os.path.join(get_project_root(), "assets", "ui", "game_icon.jpg")
            self._logo = pygame.image.load(logo_path).convert_alpha()
            self.color_print("Logo loaded.", "IMPORTANT")

            # scale to 20% of screen width while preserving aspect ratio
            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
            self.color_print(f"Logo scaled to {w}x{h}", "IMPORTANT")
        return self._logo

    @property
    def background(self):
        # load and cache the background image when first accessed
        if self._background is None:
            bg_path = os.path.join(get_project_root(), "assets", "ui", "menu_background.jpg")
            bg = pygame.image.load(bg_path).convert()
            self.color_print("Background loaded", "IMPORTANT")

            # scale to screen size then cheap-blur via downscale/upscale
            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
            self.color_print("Background scaled and blurred", "IMPORTANT")
        return self._background

    # ---------------------------
    # BACKWARD-COMPATIBLE ACCESSORS
    # ---------------------------
    @property
    def singleplayer_btn(self):
        # convenience accessor for backward compatibility
        return self._singleplayer_btn

    @property
    def multiplayer_btn(self):
        return self._multiplayer_btn

    @property
    def loadout_btn(self):
        return self._loadout_btn

    # ---------------------------
    # CENTRALIZED UI FACTORY
    # ---------------------------
    def init_ui(self, element_type, rect, manager, container=None, **kwargs):
        # import locally to avoid import-time side-effects
        import pygame_gui.elements as elements
        import pygame_gui.windows as windows
        from .UItheme import UITheme

        # map string keys to concrete pygame_gui classes
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

        # handle confirmation dialogs differently (their API expects rect & manager etc.)
        if element_type == "confirmation_dialog":
            element = ui_class(
                rect=rect,
                manager=manager,
                **kwargs
            )
            # style the dialog to match theme
            element.background_colour = UITheme.BACKGROUND
            element.border_colour = UITheme.SECONDARY
            element.rebuild()
            return element

        # build common params for standard widgets
        params = {
            "relative_rect": rect,
            "manager": manager,
            **kwargs
        }

        # parent container if provided
        if container:
            params["container"] = container

        # instantiate the element
        element = ui_class(**params)

        # keep reference for bulk operations
        self.ui_elements.append(element)

        # apply theme-based styling per element type
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
                element.set_text(f'<font color="{UITheme.TEXT}">{kwargs["text"]}</font>')

        elif element_type == "textbox":
            element.set_text(f'<font color="{UITheme.TEXT}">{kwargs.get("html_text", "")}</font>')

        return element

    # ---------------------------
    # create_from_spec
    # ---------------------------
    def create_from_spec(self, spec):
        # 'spec' is a dictionary describing a ui tree: root + children
        # root 'rect' must be an iterable [x, y, w, h]

        # determine root element type (default to panel)
        elem_type = spec.get("type", "panel")

        # create pygame.Rect for root
        root_rect = pygame.Rect(*spec["rect"])  # raises if missing/wrong

        # object_id string for theming, if provided
        obj_id = f'#{spec.get("id", "")}' if spec.get("id") else None

        # create the root element via init_ui
        root_element = self.init_ui(elem_type, root_rect, self.manager, object_id=obj_id)

        # attach root to self by id for convenience (e.g. self.menu_panel)
        if spec.get("id"):
            setattr(self, spec["id"], root_element)

        # iterate children and create each, registering callbacks where provided
        for child in spec.get("children", []):
            # child rects are relative to parent (pygame_gui container convention)
            child_rect = pygame.Rect(*child["rect"])
            child_obj_id = f'#{child.get("id", "")}' if child.get("id") else None

            # assemble kwargs for init_ui (text, options_list, html_text etc.)
            kwargs = {}
            if "text" in child:
                kwargs["text"] = child["text"]
            if "options" in child:
                kwargs["options_list"] = child["options"]
            if "html_text" in child:
                kwargs["html_text"] = child["html_text"]

            # instantiate the child element and parent it to root_element
            element = self.init_ui(child["type"], child_rect, self.manager,
                                   container=root_element, object_id=child_obj_id, **kwargs)

            # if the spec provides an id, set an attribute on self for easy access
            if child.get("id"):
                setattr(self, child["id"], element)

            # if a callback is provided (string name or callable), register it
            cb = child.get("callback")
            if cb:
                if isinstance(cb, str):
                    callback_fn = getattr(self, cb, None)
                else:
                    callback_fn = cb
                if callback_fn:
                    self.element_callbacks[element] = callback_fn

        # return the created root so caller may keep it if desired
        return root_element


    # LOAD MENU AND ALLOW FOR BACK BUTTON TO WORK
    def load_menu(self, spec):
        # clear current UI before pushing new one
        if self.current_menu:
            self.clear_element("all")

        # create new menu root
        root = self.create_from_spec(spec)

        # push spec + root for back navigation
        self.menu_stack.append((spec))
        self.current_menu = root
        return root

    # ---------------------------
    # DRAWING
    # ---------------------------
    def draw_static(self):
        # pre-render the background once onto static_surface
        self.static_surface.blit(self.background, (0, 0))

        # compute logo placement and blit it once as well
        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - logo_offset))
        self.static_surface.blit(self.logo, logo_rect)

    def draw(self):
        # blit pre-rendered background + logo; then draw GUI widgets
        self.screen.blit(self.static_surface, (0, 0))
        self.manager.draw_ui(self.screen)

    # ---------------------------
    # CLEAR / KILL UI
    # ---------------------------
    def clear_element(self, element_clear):
        # support 'all' to kill everything created
        if element_clear == "all":
            for element in list(self.ui_elements):
                # removes element from pygame_gui manager
                element.kill()
            self.ui_elements.clear()

            # reset refs
            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
            self.menu_panel = None
            self._quit_dialog = None

            # clear callback mapping to avoid memory leaks (we need to comply with the rust police)
            self.element_callbacks.clear()

    # ---------------------------
    # MAIN MENU CREATION (now spec-driven)
    # ---------------------------
    def create_main_menu(self):
        # compute panel geometry using resolution converters (keeps layout scalable)
        panel_width = resolution_converter(450, 'x')
        panel_height = resolution_converter(350, 'y')
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = int(self.screen_height - panel_height) // 1.5

        # compute button geometry relative to panel
        button_width = resolution_converter(300, 'x')
        button_height = resolution_converter(60, 'y')
        button_x = (panel_width - button_width) // 2
        start_y = resolution_converter(40, 'y')

        back_button_x = resolution_converter(10, 'x')
        back_button_y = resolution_converter(40, 'y')

        back_button_width = resolution_converter(60, 'x')
        back_button_height = resolution_converter(60, 'y')

        # build the menu spec dictionary describing the panel and its children

        # back button def
        def back_button_spec():
            return {
                "type": "button",
                "id": "_back_btn",
                "rect": [back_button_x, back_button_y, back_button_width, back_button_height],
                "text": "Back",
                "callback": "back_press"
            }

        self.menu_spec = {
            "type": "panel",
            "id": "menu_panel",
            "rect": [panel_x, panel_y, panel_width, panel_height],
            "children": [
                {
                    "type": "button",
                    "id": "_singleplayer_btn",
                    # child rects are relative to the parent panel
                    "rect": [button_x, start_y, button_width, button_height],
                    "text": "Singleplayer",
                    "callback": "singleplayer_press"
                },
                {
                    "type": "button",
                    "id": "_multiplayer_btn",
                    "rect": [button_x, start_y + int(button_height * 1.1), button_width, button_height],
                    "text": "Multiplayer",
                    "callback": "multiplayer_press"
                },
                {
                    "type": "button",
                    "id": "_loadout_btn",
                    "rect": [button_x, start_y + int(button_height * 2.2), button_width, button_height],
                    "text": "Loadout",
                    "callback": "loadout_press"
                },

            ]
        }

        # define spec for the singleplayer main menu
        self.singleplayer_main_spec = {
            "type": "panel",
            "id": "singleplayer_main_spec",
            "rect": [panel_x, panel_y, panel_width, panel_height],
            "children": [
                {
                    "type": "button",
                    "id": "_singleplayer_start_btn",
                    "rect": [button_x, start_y, button_width, button_height],
                    "text": "Start",
                    "callback": ""
                },
                back_button_spec()
            ]
        }

        # loadout_main_spec = {
        #     "type": "panel",
        #     "id": "loadout_main_spec",
        #     "rect": [panel_x, panel_y, panel_width, panel_height],
        #     "children": [
        #         #
        #         {
        #             "type": "button",
        #             "id": "_quit_dialog",
        #             "rect": [panel_x, panel_y, panel_width, panel_height],
        #             "text": "Quit"
        #         }
        #     ]
        # }

        # create the UI tree from the spec and keep the returned root
        root = self.load_menu(self.menu_spec)

        # convenience: keep attributes without the leading underscore for external code
        # if the spec used ids with leading underscores, the created attributes are set
        # above. ensure the internal underscored attributes are populated from the
        # ones the spec created. DO NOT assign to the read-only public @property
        # names (singleplayer_btn, multiplayer_btn, loadout_btn) because those
        # are defined as properties that return the underscored attributes.
        self.menu_panel = getattr(self, 'menu_panel', root)
        # populate the underscored attributes if they exist on the instance
        if hasattr(self, '_singleplayer_btn'):
            self._singleplayer_btn = getattr(self, '_singleplayer_btn')
        if hasattr(self, '_multiplayer_btn'):
            self._multiplayer_btn = getattr(self, '_multiplayer_btn')
        if hasattr(self, '_loadout_btn'):
            self._loadout_btn = getattr(self, '_loadout_btn')

    # ---------------------------
    # CALLBACKS
    # ---------------------------
    def singleplayer_press(self):
        self.color_print("Singleplayer button pressed", "IMPORTANT")
        self.load_menu(self.singleplayer_main_spec)
        # TODO: implement singleplayer logic

    def multiplayer_press(self):
        color_print("Multiplayer button pressed", "IMPORTANT")
        # TODO: implement multiplayer logic

    def loadout_press(self):
        color_print("Loadout button pressed", "IMPORTANT")
        # TODO: implement loadout logic

    def back_press(self):
        color_print("Back button pressed", "IMPORTANT")
        if len(self.menu_stack) > 1:
            # pop current menu
            self.clear_element("all")
            self.menu_stack.pop()

            # restore previous
            spec = self.menu_stack[-1]
            self.current_menu = self.create_from_spec(spec)
            self.color_print("")
        else:
            self.color_print("No previous menu to go back to!", "WARNING")

    # ---------------------------
    # EVENT LOOP (simplified using mapping)
    # ---------------------------
    def process_events(self, events):
        # compute actual sizes
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

        # process all incoming events once per frame
        for event in events:
            # let the GUI manager process the event first
            self.manager.process_events(event)

            if event.type == pygame.USEREVENT and getattr(event, 'user_type', None) == pygame_gui.UI_BUTTON_PRESSED:
                # use direct object lookup into the mapping; O(1) dispatch
                callback = self.element_callbacks.get(event.ui_element)
                if callback:
                    callback()
                    continue

            # handle quit / escape logic
            elif event.type in (pygame.KEYDOWN, pygame.QUIT):
                if event.type == pygame.QUIT:
                    # if quit dialog currently exists and user closed window
                    # noinspection PyUnreachableCode
                    # SCREW YOU DUMB PYCHARM!!!!!! jetbrains more like NOBRAINS KILL URSELF PLZ
                    if self._quit_dialog:
                        print("quit")
                        return "quit"

                # only show quit dialog on ESC (ignore other keys)
                if event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                    continue
                init_quit_dialog()

            # user confirmed the quit dialog
            elif event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None
                    return "quit"

            # user closed the quit dialog without confirming
            elif event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None

        # no special action requested
        return None
