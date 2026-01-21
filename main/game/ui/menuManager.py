import os
import pygame
import pygame_gui
import math
from main.config import resolution_converter, get_project_root, get_logger, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT

class MainMenuManager:
    """
    Manages the main menu GUI system with angled 2.5D map preview.

    - Tiles are rendered as square tops projected with tilt and rotation,
      producing a coherent 2.5D look as if you're looking at the map from an angle.
    - WASD/Arrow keys move the camera (in world-space, relative to rotation).
    - Middle mouse (click + drag) rotates the camera.
    - Mouse wheel zooms the preview.
    - Hovering shows inspector info and darkens & labels the tile.
    """

    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height, theme=None):
        """Initialize the main menu manager."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MainMenuManager (angled 2.5D)")

        # ... (kept the bulk of existing init state)
        self.all_units = []
        self.selected_unit = None
        self.player_loadout = []
        self.selected_loadout_index = None
        self.loadout_unit_buttons = []
        self.current_menu = None
        self.menu_stack = []
        self.menu_spec = None
        self.specs = {}
        self.theme = theme
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.logger.debug(f"Screen dimensions: {screen_width}x{screen_height}")

        self.manager = pygame_gui.UIManager((screen_width, screen_height))
        self._preload_fonts()
        self.logger.info("GUI Manager initialized")

        self._logo = None
        self._background = None

        self.menu_panel = None
        self._quit_dialog = None
        self._singleplayer_btn = None
        self._multiplayer_btn = None
        self._loadout_btn = None
        self._mapgen_btn = None

        # Map generator state with angled 2.5D top-down preview
        self.map_generator = None
        self.current_map = None

        # Preview camera state (in screen pixels for pan)
        self.preview_pan = [0.0, 0.0]    # pixels
        self.preview_zoom = 1.0
        self.preview_tile_size = TILE_SIZE if TILE_SIZE else 32
        # camera rotation (degrees) and tilt (0..1, smaller => more angled)
        self.camera_rotation = 30.0  # degrees (start with a small rotation for angle)
        self.camera_tilt = 0.55      # vertical squash to simulate looking from angle
        # rotation interaction
        self.preview_rotating = False
        self.preview_last_mouse = (0, 0)

        self.map_preview_active = False
        self.mouse_over_preview = False

        # map size (square)
        self.map_size = MAP_WIDTH if MAP_WIDTH else 40

        self.static_surface = pygame.Surface((screen_width, screen_height))
        self.logger.debug("Created static surface for background caching")

        self.ui_elements = []
        self.element_callbacks = {}

        self._load_units()
        self.logger.info("Creating main menu from spec...")
        self.create_main_menu()
        self.logger.info("Main menu created successfully")

        try:
            pygame.display.set_icon(self.logo)
            self.logger.info("Window icon set from logo")
        except Exception:
            pass

        self.draw_static()
        self.logger.debug("Static content pre-rendered")

    def _load_units(self):
        """Load all available units from the units module into all_units list."""
        try:
            from main.game.data.units.units import (
                Recon_Tank, Grunts, Sniper_Team, Mobile_Howitzer,
                Light_Machine_Gunner, Medium_Machine_Gunner, Battle_Tank,
                Supply_Carrier, Supply_Truck, IFV, SMG_Squad, Grunts_M, APC
            )

            self.all_units = [
                Grunts, SMG_Squad, Grunts_M, Light_Machine_Gunner,
                Medium_Machine_Gunner, Sniper_Team, Recon_Tank, Battle_Tank,
                IFV, APC, Supply_Carrier, Supply_Truck, Mobile_Howitzer
            ]
            self.logger.info(f"Loaded {len(self.all_units)} units for loadout selection")
        except Exception as e:
            self.logger.error(f"Failed to load units: {e}", exc_info=True)
            self.all_units = []

    def _preload_fonts(self):
        """Preload commonly used fonts to prevent runtime warnings."""
        try:
            font_dict = self.manager.get_theme().get_font_dictionary()

            font_configs = [
                (14, 'noto_sans', True, False, True),
                (14, 'noto_sans', False, True, True),
                (14, 'noto_sans', True, True, True),
                (14, 'noto_sans', False, False, True),
                (12, 'noto_sans', True, False, True),
                (12, 'noto_sans', False, True, True),
                (12, 'noto_sans', False, False, True),
                (16, 'noto_sans', True, False, True),
                (16, 'noto_sans', False, True, True),
            ]

            preloaded = 0
            for size, name, bold, italic, aa in font_configs:
                try:
                    font_dict.preload_font(size, name, bold, italic, aa)
                    preloaded += 1
                except Exception as e:
                    self.logger.debug(f"Could not preload font {name} {size}pt: {e}")

            self.logger.debug(f"Preloaded {preloaded}/{len(font_configs)} UI font variations")
        except Exception as e:
            self.logger.debug(f"Font preloading encountered issue (non-critical): {e}")

    @property
    def logo(self):
        """Lazy-load and cache the game logo, scaled to 20% of screen width."""
        if self._logo is None:
            logo_path = os.path.join(get_project_root(), "assets", "ui", "game_icon.jpg")
            self.logger.debug(f"Loading logo from: {logo_path}")

            self._logo = pygame.image.load(logo_path).convert_alpha()
            self.logger.info("Logo loaded successfully")

            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
            self.logger.debug(f"Logo scaled to {w}x{h}")
        return self._logo

    @property
    def background(self):
        """Lazy-load and cache the menu background with blur effect."""
        if self._background is None:
            bg_path = os.path.join(get_project_root(), "assets", "ui", "menu_background.jpg")
            self.logger.debug(f"Loading background from: {bg_path}")

            bg = pygame.image.load(bg_path).convert()
            self.logger.info("Background loaded successfully")

            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
            self.logger.debug("Background scaled and blurred")
        return self._background

    @property
    def singleplayer_btn(self):
        return self._singleplayer_btn

    @property
    def multiplayer_btn(self):
        return self._multiplayer_btn

    @property
    def loadout_btn(self):
        return self._loadout_btn

    def init_ui(self, element_type, rect, manager, container=None, **kwargs):
        import pygame_gui.elements as elements
        import pygame_gui.windows as windows
        from .UItheme import UITheme

        self.logger.debug(f"Creating UI element: {element_type}")

        def _safe_apply_colours(elem, mapping):
            if elem is None:
                return
            if hasattr(elem, "colours"):
                try:
                    elem.colours.update(mapping)
                    try:
                        elem.rebuild()
                    except Exception as e:
                        self.logger.warning(f"Failed to rebuild element: {e}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to update colors: {e}")

            try:
                if "normal_bg" in mapping and hasattr(elem, "background_colour"):
                    elem.background_colour = mapping["normal_bg"]
                if "normal_border" in mapping and hasattr(elem, "border_colour"):
                    elem.border_colour = mapping["normal_border"]
                try:
                    elem.rebuild()
                except Exception:
                    pass
            except Exception:
                pass

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

        if element_type == "confirmation_dialog":
            self.logger.debug("Creating confirmation dialog")
            element = ui_class(rect=rect, manager=manager, **kwargs)
            try:
                element.background_colour = UITheme.BACKGROUND
                element.border_colour = UITheme.SECONDARY
                element.rebuild()
            except Exception as e:
                self.logger.warning(f"Failed to style dialog: {e}")
            return element

        params = {"relative_rect": rect, "manager": manager, **kwargs}
        if container:
            params["container"] = container

        if element_type == "dropdown":
            options = params.pop("options_list", []) or []
            if not options:
                options = ["<no units available>"]
                self.logger.warning("Dropdown created with no options")

            starting_option = params.pop("starting_option", None)
            if starting_option not in options:
                starting_option = options[0]

            other_kwargs = {k: v for k, v in params.items()
                            if k not in ("relative_rect", "manager")}

            try:
                element = ui_class(options, starting_option, params["relative_rect"],
                                   params["manager"], **other_kwargs)
            except TypeError:
                element = ui_class(options, starting_option,
                                   relative_rect=params["relative_rect"],
                                   manager=params["manager"],
                                   **other_kwargs)

            self.ui_elements.append(element)
            _safe_apply_colours(element, {
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY
            })
            self.logger.debug(f"Dropdown created with {len(options)} options")
            return element

        elif element_type == "slider":
            start_value = params.pop("start_value", 0.0)
            value_range = params.pop("value_range", (0.0, 1.0))

            other_kwargs = {k: v for k, v in params.items()
                            if k not in ("relative_rect", "manager")}

            element = ui_class(
                relative_rect=params["relative_rect"],
                start_value=start_value,
                value_range=value_range,
                manager=params["manager"],
                **other_kwargs
            )

            self.ui_elements.append(element)
            _safe_apply_colours(element, {
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY
            })
            self.logger.debug(f"Slider created: {start_value} in {value_range}")
            return element

        # Generic element creation
        element = ui_class(**params)
        self.ui_elements.append(element)

        if element_type == "button":
            _safe_apply_colours(element, {
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY,
                "hovered_border": UITheme.HIGHLIGHT
            })

        elif element_type == "panel":
            try:
                element.background_colour = UITheme.BACKGROUND
                element.border_colour = UITheme.SECONDARY
                element.rebuild()
            except Exception as e:
                self.logger.warning(f"Failed to style panel: {e}")

        elif element_type == "label":
            if "text" in kwargs:
                try:
                    element.set_text(kwargs["text"])
                except Exception:
                    try:
                        element.set_text(str(kwargs["text"]))
                    except Exception as e:
                        self.logger.warning(f"Failed to set label text: {e}")

            _safe_apply_colours(element, {"normal_text": UITheme.TEXT})

        elif element_type == "textbox":
            try:
                element.set_text(kwargs.get("html_text", ""))
            except Exception:
                try:
                    element.set_text(str(kwargs.get("html_text", "")))
                except Exception as e:
                    self.logger.warning(f"Failed to set textbox text: {e}")

        return element

    def create_from_spec(self, spec):
        """Recursively create UI hierarchy from specification dictionary."""
        self.logger.debug(f"Creating UI from spec: {spec.get('id', 'unnamed')}")

        elem_type = spec.get("type", "panel")
        root_rect = pygame.Rect(*spec["rect"])
        obj_id = f'#{spec.get("id", "")}' if spec.get("id") else None

        root_element = self.init_ui(elem_type, root_rect, self.manager, object_id=obj_id)

        if spec.get("id"):
            setattr(self, spec["id"], root_element)
            self.logger.debug(f"Set attribute: {spec['id']}")

        for child in spec.get("children", []):
            child_rect = pygame.Rect(*child["rect"])
            child_obj_id = f'#{child.get("id", "")}' if child.get("id") else None

            kwargs = {}
            if "text" in child:
                kwargs["text"] = child["text"]
            if "options" in child:
                kwargs["options_list"] = child["options"]
            if "html_text" in child:
                kwargs["html_text"] = child["html_text"]
            if "start_value" in child:
                kwargs["start_value"] = child["start_value"]
            if "value_range" in child:
                kwargs["value_range"] = child["value_range"]

            element = self.init_ui(child["type"], child_rect, self.manager,
                                   container=root_element, object_id=child_obj_id, **kwargs)

            if child.get("id"):
                setattr(self, child["id"], element)

            cb = child.get("callback")
            if cb:
                callback_fn = getattr(self, cb, None) if isinstance(cb, str) else cb
                if callback_fn:
                    self.element_callbacks[element] = callback_fn
                    self.logger.debug(f"Registered callback for {child.get('id', 'element')}")

        self.logger.info(f"Created UI tree with {len(spec.get('children', []))} children")
        return root_element

    def load_menu(self, spec):
        """Load a new menu, replacing the current one and adding to navigation stack."""
        self.logger.info(f"Loading menu: {spec.get('id', 'unnamed')}")

        if self.current_menu:
            self.clear_element("all")

        root = self.create_from_spec(spec)
        self.menu_stack.append(spec)
        self.current_menu = root

        self.logger.debug(f"Menu stack depth: {len(self.menu_stack)}")
        return root

    def draw_static(self):
        """Pre-render static background elements to cached surface."""
        self.logger.debug("Drawing static content")
        self.static_surface.blit(self.background, (0, 0))

        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(center=(self.screen_width // 2,
                                               self.screen_height // 2 - logo_offset))
        self.static_surface.blit(self.logo, logo_rect)

    def clear_element(self, element_clear):
        """Clear UI elements ('all' to clear everything)."""
        if element_clear == "all":
            self.logger.debug(f"Clearing {len(self.ui_elements)} UI elements")
            for element in list(self.ui_elements):
                try:
                    element.kill()
                except Exception:
                    pass
            self.ui_elements.clear()

            self.loadout_unit_buttons.clear()
            self.selected_loadout_index = None

            # Clear map preview state
            self.map_preview_active = False
            self.mouse_over_preview = False
            self.map_generator = None
            self.current_map = None
            self.map_camera_iso = None
            self.camera_rotation = 0

            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
            self._mapgen_btn = None
            self.menu_panel = None
            self._quit_dialog = None
            self.element_callbacks.clear()

            self.logger.info("All UI elements cleared")

    def create_main_menu(self):
        """Build specifications for all menu screens and load the main menu."""
        self.logger.debug("Building main menu specifications")

        panel_width = resolution_converter(450, 'x')
        panel_height = resolution_converter(350, 'y')
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = int(self.screen_height - panel_height) // 1.5

        button_width = resolution_converter(300, 'x')
        button_height = resolution_converter(60, 'y')
        button_x = (panel_width - button_width) // 2
        start_y = resolution_converter(40, 'y')

        back_btn_x = resolution_converter(10, 'x')
        back_btn_y = resolution_converter(40, 'y')
        back_btn_w = resolution_converter(60, 'x')
        back_btn_h = resolution_converter(60, 'y')

        def back_btn():
            return {
                "type": "button",
                "id": "_back_btn",
                "rect": [back_btn_x, back_btn_y, back_btn_w, back_btn_h],
                "text": "Back",
                "callback": "back_press"
            }

        self.specs['menu_spec'] = {
            "type": "panel",
            "id": "menu_panel",
            "rect": [panel_x, panel_y, panel_width, panel_height],
            "children": [
                {
                    "type": "button",
                    "id": "_singleplayer_btn",
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
                {
                    "type": "button",
                    "id": "_mapgen_btn",
                    "rect": [button_x, start_y + int(button_height * 3.3), button_width, button_height],
                    "text": "Map Generator",
                    "callback": "mapgen_press"
                },
            ]
        }

        self.specs['singleplayer_main_spec'] = {
            "type": "panel",
            "id": "singleplayer_main_spec",
            "rect": [panel_x, panel_y, panel_width, panel_height],
            "children": [
                {
                    "type": "button",
                    "id": "_singleplayer_start_btn",
                    "rect": [button_x, start_y, button_width, button_height],
                    "text": "Start Game",
                    "callback": "start_game_press"
                },
                back_btn()
            ]
        }

        loadout_w = int(panel_width * 2.5)
        loadout_h = int(panel_height * 1.8)
        loadout_x = (self.screen_width - loadout_w) // 2
        loadout_y = (self.screen_height - loadout_h) // 2

        self.specs['loadout_spec'] = {
            "type": "panel",
            "id": "loadout_panel",
            "rect": [loadout_x, loadout_y, loadout_w, loadout_h],
            "children": [
                {
                    "type": "label",
                    "id": "loadout_title",
                    "rect": [20, 15, 300, 40],
                    "text": "Loadout Selection"
                },
                {
                    "type": "dropdown",
                    "id": "unit_dropdown",
                    "rect": [20, 70, 280, 45],
                    "options": [u.name for u in self.all_units] if self.all_units else ["No units"],
                    "callback": "unit_selected"
                },
                {
                    "type": "panel",
                    "id": "loadout_list_panel",
                    "rect": [20, 135, 280, 285]
                },
                {
                    "type": "textbox",
                    "id": "unit_description",
                    "rect": [320, 70, 380, 280],
                    "html_text": "<b>Select a unit</b><br><br>Choose from dropdown to view stats."
                },
                {
                    "type": "button",
                    "id": "confirm_unit_btn",
                    "rect": [320, 370, 200, 50],
                    "text": "Add to Loadout",
                    "callback": "add_to_loadout"
                },
                {
                    "type": "button",
                    "id": "remove_unit_btn",
                    "rect": [530, 370, 170, 50],
                    "text": "Remove Selected",
                    "callback": "remove_selected_unit"
                },
                back_btn()
            ]
        }

        # Map generator spec: left controls. Preview and inspector are drawn/created outside this panel
        mapgen_controls_w = int(self.screen_width * 0.25)
        mapgen_preview_x = mapgen_controls_w + 10
        mapgen_preview_w = self.screen_width - mapgen_preview_x

        # We'll add a map size slider to the controls panel
        self.specs['mapgen_spec'] = {
            "type": "panel",
            "id": "mapgen_panel",
            "rect": [0, 0, mapgen_controls_w, self.screen_height],
            "children": [
                {
                    "type": "label",
                    "id": "mapgen_title",
                    "rect": [20, 20, mapgen_controls_w - 40, 40],
                    "text": "Map Generator - Angled 2.5D View"
                },
                {
                    "type": "label",
                    "id": "map_size_label",
                    "rect": [20, 80, mapgen_controls_w - 40, 25],
                    "text": "Map Size:"
                },
                {
                    "type": "slider",
                    "id": "map_size_slider",
                    "rect": [20, 110, mapgen_controls_w - 40, 35],
                    "start_value": float(self.map_size),
                    "value_range": (8.0, 128.0),
                    "callback": "map_size_changed"
                },
                {
                    "type": "label",
                    "id": "forest_label",
                    "rect": [20, 160, mapgen_controls_w - 40, 25],
                    "text": "Forest Density:"
                },
                {
                    "type": "slider",
                    "id": "forest_slider",
                    "rect": [20, 190, mapgen_controls_w - 40, 35],
                    "start_value": 0.25,
                    "value_range": (0.0, 0.5)
                },
                {
                    "type": "label",
                    "id": "urban_label",
                    "rect": [20, 240, mapgen_controls_w - 40, 25],
                    "text": "Urban Density:"
                },
                {
                    "type": "slider",
                    "id": "urban_slider",
                    "rect": [20, 270, mapgen_controls_w - 40, 35],
                    "start_value": 0.15,
                    "value_range": (0.0, 0.3)
                },
                {
                    "type": "label",
                    "id": "mountain_label",
                    "rect": [20, 320, mapgen_controls_w - 40, 25],
                    "text": "Mountain Density:"
                },
                {
                    "type": "slider",
                    "id": "mountain_slider",
                    "rect": [20, 350, mapgen_controls_w - 40, 35],
                    "start_value": 0.10,
                    "value_range": (0.0, 0.25)
                },
                {
                    "type": "label",
                    "id": "elevation_label",
                    "rect": [20, 400, mapgen_controls_w - 40, 25],
                    "text": "Elevation Density:"
                },
                {
                    "type": "slider",
                    "id": "elevation_slider",
                    "rect": [20, 430, mapgen_controls_w - 40, 35],
                    "start_value": 0.20,
                    "value_range": (0.0, 0.4)
                },
                {
                    "type": "label",
                    "id": "building_label",
                    "rect": [20, 480, mapgen_controls_w - 40, 25],
                    "text": "Building Density:"
                },
                {
                    "type": "slider",
                    "id": "building_slider",
                    "rect": [20, 510, mapgen_controls_w - 40, 35],
                    "start_value": 0.08,
                    "value_range": (0.0, 0.2)
                },
                {
                    "type": "button",
                    "id": "generate_map_btn",
                    "rect": [20, 560, mapgen_controls_w - 40, 45],
                    "text": "Generate New Map",
                    "callback": "generate_map"
                },
                {
                    "type": "textbox",
                    "id": "mapgen_instructions",
                    "rect": [20, 620, mapgen_controls_w - 40, 200],
                    "html_text": (
                        "<b>Controls:</b><br>"
                        "• WASD / Arrows: Move camera (when mouse over preview)<br>"
                        "• Mouse Wheel: Zoom (when mouse over preview)<br>"
                        "• Middle Mouse (hold): Drag left/right to rotate camera<br><br>"
                        "<b>Display:</b><br>"
                        "• Angled 2.5D projection: square tops with vertical faces for elevation/buildings<br>"
                    )
                },
                back_btn()
            ]
        }

        root = self.load_menu(self.specs['menu_spec'])

        self.menu_panel = getattr(self, 'menu_panel', root)
        if hasattr(self, '_singleplayer_btn'):
            self._singleplayer_btn = getattr(self, '_singleplayer_btn')
        if hasattr(self, '_multiplayer_btn'):
            self._multiplayer_btn = getattr(self, '_multiplayer_btn')
        if hasattr(self, '_loadout_btn'):
            self._loadout_btn = getattr(self, '_loadout_btn')
        if hasattr(self, '_mapgen_btn'):
            self._mapgen_btn = getattr(self, '_mapgen_btn')

    def singleplayer_press(self, element=None):
        """Navigate to singleplayer submenu."""
        self.logger.info("Singleplayer button pressed")
        self.load_menu(self.specs['singleplayer_main_spec'])

    def multiplayer_press(self, element=None):
        """Multiplayer button handler (not yet implemented)."""
        self.logger.info("Multiplayer pressed (not implemented)")

    def start_game_press(self, element=None):
        """Start game button handler."""
        self.logger.info("Start game pressed")

    def loadout_press(self, element=None):
        """Navigate to loadout menu and refresh unit list."""
        self.logger.info("Loadout button pressed")
        self.load_menu(self.specs['loadout_spec'])

        dd = getattr(self, "unit_dropdown", None)
        if dd and self.all_units:
            try:
                first_unit_name = self.all_units[0].name
                self.selected_unit = self.all_units[0]
                self.unit_selected(dd)
                self.logger.debug("Dropdown updated with units")
            except Exception as e:
                self.logger.warning(f"Failed to update dropdown: {e}")

        self.refresh_loadout_view()

    def mapgen_press(self, element=None):
        """Navigate to map generator interface with angled 2.5D preview."""
        self.logger.info("Map Generator button pressed")
        self.load_menu(self.specs['mapgen_spec'])
        self.map_preview_active = True

        # Reset preview camera state
        self.preview_pan = [0.0, 0.0]
        self.preview_zoom = 1.0
        self.preview_tile_size = TILE_SIZE if TILE_SIZE else 32
        self.preview_rotating = False
        self.preview_last_mouse = (0, 0)
        self.camera_rotation = 30.0
        self.camera_tilt = 0.55
        self.map_size = getattr(self, "map_size", MAP_WIDTH if MAP_WIDTH else 40)

        # Create right-side inspector UI elements (absolute positions)
        preview_x = int(self.screen_width * 0.25) + 10
        preview_w = self.screen_width - preview_x
        inspector_w = 300
        inspector_x = preview_x + preview_w - inspector_w - 10
        inspector_y = 20

        try:
            self.preview_info_panel = self.init_ui(
                "panel",
                pygame.Rect(inspector_x, inspector_y, inspector_w, 220),
                self.manager
            )

            self.preview_tile_title = self.init_ui(
                "label",
                pygame.Rect(10, 10, inspector_w - 20, 24),
                self.manager,
                container=self.preview_info_panel,
                text="Tile Inspector"
            )

            self.preview_tile_type = self.init_ui(
                "label",
                pygame.Rect(10, 40, inspector_w - 20, 28),
                self.manager,
                container=self.preview_info_panel,
                text="Type: -"
            )

            self.preview_tile_coords = self.init_ui(
                "label",
                pygame.Rect(10, 70, inspector_w - 20, 22),
                self.manager,
                container=self.preview_info_panel,
                text="Coords: -"
            )

            self.preview_tile_elev = self.init_ui(
                "label",
                pygame.Rect(10, 96, inspector_w - 20, 22),
                self.manager,
                container=self.preview_info_panel,
                text="Elevation: -"
            )

            self.preview_tile_color = self.init_ui(
                "panel",
                pygame.Rect(10, 124, 40, 40),
                self.manager,
                container=self.preview_info_panel
            )

            self.preview_tile_note = self.init_ui(
                "label",
                pygame.Rect(60, 124, inspector_w - 70, 80),
                self.manager,
                container=self.preview_info_panel,
                text="Hover a tile in the preview to see info here."
            )
        except Exception as e:
            self.logger.warning(f"Failed to create inspector UI: {e}")

        # Generate initial map
        self.generate_map()

    def map_size_changed(self, element=None):
        """Handle map size slider movement and regenerate map."""
        try:
            if element is None:
                return
            size_val = int(round(element.get_current_value()))
            size_val = max(8, min(256, size_val))
            self.map_size = size_val
            label = getattr(self, "map_size_label", None)
            if label:
                try:
                    label.set_text(f"Map Size: {self.map_size} × {self.map_size}")
                except Exception:
                    pass
            self.logger.info(f"Map size slider set to {self.map_size}; regenerating map")
            self.generate_map()
        except Exception as e:
            self.logger.error(f"Error in map_size_changed: {e}", exc_info=True)

    def generate_map(self, element=None):
        """Generate a new map with current settings (square size from slider)."""
        from main.game.data.maps.mapGen import MapGenerator, MapConfig

        self.logger.info("Generating new map (angled 2.5D preview)")

        forest_density = getattr(self, "forest_slider", None)
        urban_density = getattr(self, "urban_slider", None)
        mountain_density = getattr(self, "mountain_slider", None)
        elevation_density = getattr(self, "elevation_slider", None)
        building_density = getattr(self, "building_slider", None)

        config = MapConfig(
            width=self.map_size,
            height=self.map_size,
            forest_density=forest_density.get_current_value() if forest_density else 0.25,
            urban_density=urban_density.get_current_value() if urban_density else 0.15,
            mountain_density=mountain_density.get_current_value() if mountain_density else 0.10,
            elevation_density=elevation_density.get_current_value() if elevation_density else 0.20,
            building_density=building_density.get_current_value() if building_density else 0.08
        )

        self.map_generator = MapGenerator(config)
        self.current_map = self.map_generator.generate()

        # Reset preview camera to center on map
        map_px_w = self.map_size * self.preview_tile_size * self.preview_zoom
        map_px_h = self.map_size * self.preview_tile_size * self.preview_zoom
        preview_x = int(self.screen_width * 0.25) + 10
        preview_w = self.screen_width - preview_x
        preview_h = self.screen_height

        # set pan so map center aligns with preview center
        self.preview_pan[0] = - (map_px_w / 2) + (preview_w / 2)
        self.preview_pan[1] = - (map_px_h / 2) + (preview_h / 2)

        self.preview_zoom = 1.0
        self.camera_rotation = 30.0

        stats = self.map_generator.get_terrain_statistics()
        self.logger.info(f"Map generated: {stats}")

    def back_press(self, element=None):
        """Navigate to previous menu in the stack."""
        self.logger.info("Back button pressed")
        if len(self.menu_stack) > 1:
            self.clear_element("all")
            self.menu_stack.pop()
            spec = self.menu_stack[-1]
            self.current_menu = self.create_from_spec(spec)
            self.logger.debug(f"Navigated back to: {spec.get('id', 'previous')}")
        else:
            self.logger.warning("Already at root menu")

    def refresh_loadout_view(self):
        """Update the loadout display with clickable unit buttons."""
        for btn in self.loadout_unit_buttons:
            btn.kill()
        self.loadout_unit_buttons.clear()

        panel = getattr(self, "loadout_list_panel", None)
        if not panel:
            return

        try:
            from .UItheme import UITheme
            panel.background_colour = UITheme.BACKGROUND
            panel.border_colour = UITheme.SECONDARY
            panel.rebuild()
        except Exception:
            pass

        if not self.player_loadout:
            empty_box = self.init_ui(
                "textbox",
                pygame.Rect(5, 5, 270, 100),
                self.manager,
                container=panel,
                html_text="<b>Current Loadout:</b><br><br>No units selected<br><br><em>Add units from the dropdown above (max 6 units)</em>"
            )
            self.loadout_unit_buttons.append(empty_box)
            self.logger.debug("Loadout empty, showing placeholder")
            return

        button_height = 40
        button_spacing = 5
        button_width = 260
        start_y = 10

        for i, unit in enumerate(self.player_loadout):
            y_pos = start_y + (i * (button_height + button_spacing))

            cost_text = str(unit.cost) if unit.cost is not None else "N/A"
            button_text = f"{i + 1}. {unit.name} - {cost_text}"

            unit_btn = self.init_ui(
                "button",
                pygame.Rect(10, y_pos, button_width, button_height),
                self.manager,
                container=panel,
                text=button_text
            )

            if i == self.selected_loadout_index:
                try:
                    from .UItheme import UITheme
                    unit_btn.colours["normal_bg"] = UITheme.HIGHLIGHT
                    unit_btn.rebuild()
                except Exception:
                    pass

            self.loadout_unit_buttons.append(unit_btn)

            def make_callback(index):
                return lambda elem=None: self.select_loadout_unit(index)

            self.element_callbacks[unit_btn] = make_callback(i)

        total_cost = sum(u.cost for u in self.player_loadout if isinstance(u.cost, (int, float)))
        total_y = start_y + (len(self.player_loadout) * (button_height + button_spacing)) + 10

        total_label = self.init_ui(
            "label",
            pygame.Rect(10, total_y, button_width, 30),
            self.manager,
            container=panel,
            text=f"<b>Total Cost: {total_cost} | Units: {len(self.player_loadout)}/6</b>"
        )
        self.loadout_unit_buttons.append(total_label)

        self.logger.debug(f"Refreshed loadout view: {len(self.player_loadout)} units, cost {total_cost}")

    def unit_selected(self, element=None):
        """Update unit description box when dropdown selection changes."""
        dropdown = element or getattr(self, "unit_dropdown", None)
        if not dropdown:
            return

        selection = getattr(dropdown, "selected_option", None)
        if not selection:
            return

        if isinstance(selection, tuple):
            selection = selection[0] if selection else None

        if not selection:
            return

        unit = next((u for u in self.all_units if u.name == selection), None)
        if not unit:
            self.logger.warning(f"Unit '{selection}' not found")
            return

        self.selected_unit = unit
        self.logger.info(f"Selected: {unit.name}")

        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            weapons = ", ".join([w.name for w in unit.weapons]) if unit.weapons else "None"
            perks = ", ".join([p.name for p in unit.perks]) if unit.perks else "None"

            lines = [
                f"<b>{unit.name}</b>",
                f"<br><b>Cost:</b> {unit.cost}",
                f"<b>HP:</b> {unit.hp}",
                f"<b>Armor:</b> {unit.armor}",
                f"<b>Sight:</b> {unit.sight}",
                f"<b>Mobility:</b> {unit.mobility}",
                f"<b>Class:</b> {unit.unitclass.name}",
                f"<b>Weight:</b> {unit.transport_weight}",
                f"<br><b>Weapons:</b> {weapons}",
                f"<b>Perks:</b> {perks}",
                f"<br>{unit.description or 'No description.'}"
            ]
            desc_box.set_text("<br>".join(lines))

    def add_to_loadout(self, element=None):
        """Add selected unit to loadout (max 6, no duplicates)."""
        unit = self.selected_unit
        if not unit:
            dd = getattr(self, "unit_dropdown", None)
            if dd:
                sel = getattr(dd, "selected_option", None)
                if isinstance(sel, tuple):
                    sel = sel[0] if sel else None
                unit = next((u for u in self.all_units if u.name == sel), None)

        if not unit:
            self.logger.warning("No unit selected")
            return

        if len(self.player_loadout) >= 6:
            self.logger.warning("Loadout full (max 6)")
            return

        if any(u.name == unit.name for u in self.player_loadout):
            self.logger.warning(f"{unit.name} already in loadout")
            return

        self.player_loadout.append(unit)
        self.logger.info(f"Added {unit.name} ({len(self.player_loadout)}/6)")

        self.selected_loadout_index = None

        self.refresh_loadout_view()

    def select_loadout_unit(self, index):
        """Select a unit from the loadout list."""
        if index < 0 or index >= len(self.player_loadout):
            self.logger.warning(f"Invalid loadout index: {index}")
            return

        self.selected_loadout_index = index
        unit = self.player_loadout[index]

        self.logger.info(f"Selected loadout unit: {unit.name} (index {index})")

        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            weapons = ", ".join([w.name for w in unit.weapons]) if unit.weapons else "None"
            perks = ", ".join([p.name for p in unit.perks]) if unit.perks else "None"

            lines = [
                f"<b>{unit.name}</b> <em>(In Loadout)</em>",
                f"<br><b>Cost:</b> {unit.cost}",
                f"<b>HP:</b> {unit.hp}",
                f"<b>Armor:</b> {unit.armor}",
                f"<b>Sight:</b> {unit.sight}",
                f"<b>Mobility:</b> {unit.mobility}",
                f"<b>Class:</b> {unit.unitclass.name}",
                f"<b>Weight:</b> {unit.transport_weight}",
                f"<br><b>Weapons:</b> {weapons}",
                f"<b>Perks:</b> {perks}",
                f"<br>{unit.description or 'No description.'}",
                f"<br><em>Click 'Remove Selected' to remove this unit.</em>"
            ]
            desc_box.set_text("<br>".join(lines))

        self.refresh_loadout_view()

    def remove_selected_unit(self, element=None):
        """Remove the currently selected unit from loadout."""
        if self.selected_loadout_index is not None:
            if 0 <= self.selected_loadout_index < len(self.player_loadout):
                removed = self.player_loadout.pop(self.selected_loadout_index)
                self.logger.info(f"Removed {removed.name} from loadout ({len(self.player_loadout)}/6)")
                self.selected_loadout_index = None
            else:
                self.logger.warning(f"Invalid selected index: {self.selected_loadout_index}")
                self.selected_loadout_index = None
        else:
            if self.player_loadout:
                removed = self.player_loadout.pop()
                self.logger.info(f"Removed last unit {removed.name} ({len(self.player_loadout)}/6)")
            else:
                self.logger.warning("Loadout is empty, nothing to remove")

        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            desc_box.set_text("<b>Select a unit</b><br><br>Choose from dropdown or loadout to view stats.")

        self.refresh_loadout_view()

    def process_events(self, events):
        """Process pygame and UI events."""
        dialog_w = resolution_converter(330, "x")
        dialog_h = resolution_converter(200, "y")
        dialog_x = (self.screen_width - dialog_w) // 2
        dialog_y = (self.screen_height - dialog_h) // 2

        def init_quit_dialog():
            if self._quit_dialog is None:
                self.logger.info("Creating quit dialog")
                self._quit_dialog = self.init_ui(
                    "confirmation_dialog",
                    pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h),
                    self.manager,
                    window_title="Confirm Quit",
                    action_long_desc="Are you sure you want to quit?",
                    action_short_name="Quit",
                    blocking=True
                )

        # Update mouse position for map preview
        if self.map_preview_active:
            mouse_pos = pygame.mouse.get_pos()
            preview_x = int(self.screen_width * 0.25) + 10
            preview_rect = pygame.Rect(preview_x, 0,
                                       self.screen_width - preview_x,
                                       self.screen_height)
            self.mouse_over_preview = preview_rect.collidepoint(mouse_pos)

        for event in events:
            self.manager.process_events(event)

            if event.type == pygame.USEREVENT:
                user_type = getattr(event, "user_type", None)

                # include slider moved events
                slider_event = pygame_gui.UI_HORIZONTAL_SLIDER_MOVED if hasattr(pygame_gui, "UI_HORIZONTAL_SLIDER_MOVED") else None
                if user_type in (pygame_gui.UI_BUTTON_PRESSED, pygame_gui.UI_DROP_DOWN_MENU_CHANGED, slider_event):
                    callback = self.element_callbacks.get(event.ui_element)
                    if callback:
                        try:
                            result = callback(event.ui_element)
                            if result in ("map_generator", "start_game"):
                                return result
                        except TypeError:
                            result = callback()
                            if result in ("map_generator", "start_game"):
                                return result
                        continue

            # Preview interactions: middle mouse rotates; wheel zooms; motion updates rotation while middle held
            if self.map_preview_active:
                # middle button press (start rotating)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    mouse_pos = event.pos
                    preview_x = int(self.screen_width * 0.25) + 10
                    preview_rect = pygame.Rect(preview_x, 0,
                                               self.screen_width - preview_x,
                                               self.screen_height)
                    if preview_rect.collidepoint(mouse_pos):
                        self.preview_rotating = True
                        self.preview_last_mouse = mouse_pos
                        pygame.mouse.get_rel()  # reset relative motion
                        self.logger.debug("Started middle-mouse rotating")
                        continue

                # middle button up (stop rotating)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                    if self.preview_rotating:
                        self.preview_rotating = False
                        self.logger.debug("Stopped middle-mouse rotating")
                        continue

                # mouse wheel zoom (when mouse over preview)
                if event.type == pygame.MOUSEWHEEL and self.mouse_over_preview:
                    old_zoom = self.preview_zoom
                    if event.y > 0:
                        self.preview_zoom *= 1.12
                    else:
                        self.preview_zoom /= 1.12
                    self.preview_zoom = max(0.25, min(4.0, self.preview_zoom))
                    self.logger.debug(f"Zoom changed {old_zoom:.3f} -> {self.preview_zoom:.3f}")

            if event.type in (pygame.KEYDOWN, pygame.QUIT):
                if event.type == pygame.QUIT:
                    if self._quit_dialog:
                        self.logger.info("Quit confirmed (window close)")
                        return "quit"

                if event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                    continue
                init_quit_dialog()

            if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == self._quit_dialog:
                    self.logger.info("Quit confirmed")
                    self._quit_dialog = None
                    return "quit"

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._quit_dialog:
                    self.logger.info("Quit cancelled")
                    self._quit_dialog = None

            # mouse motion while rotating -> rotate camera
            if event.type == pygame.MOUSEMOTION and self.preview_rotating:
                mx, my = event.pos
                lx, ly = self.preview_last_mouse
                dx = mx - lx
                # rotate camera proportional to horizontal movement
                self.camera_rotation = (self.camera_rotation + dx * 0.45) % 360
                self.preview_last_mouse = (mx, my)
                # continue to avoid other processing
                continue

        return None

    def update_camera(self, dt):
        """Update camera panning with WASD/Arrow keys while mouse is over preview.

        Movement is applied in world-space and converted into pan in screen pixels
        so movement respects the camera rotation.
        """
        if not self.map_preview_active:
            return

        if not self.mouse_over_preview:
            return

        keys = pygame.key.get_pressed()
        move_speed_tiles = 8.0 * dt  # move in tiles per second scaled by dt

        dx_world = 0.0
        dy_world = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy_world -= move_speed_tiles
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy_world += move_speed_tiles
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx_world -= move_speed_tiles
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx_world += move_speed_tiles

        if dx_world == 0 and dy_world == 0:
            return

        # convert world tile delta to screen pixels considering rotation and tilt
        angle_rad = math.radians(self.camera_rotation)
        tile_px = self.preview_tile_size * self.preview_zoom
        # rotated world -> screen delta
        screen_dx = (math.cos(angle_rad) * dx_world - math.sin(angle_rad) * dy_world) * tile_px
        screen_dy = (math.sin(angle_rad) * dx_world + math.cos(angle_rad) * dy_world) * tile_px * self.camera_tilt

        self.preview_pan[0] += screen_dx
        self.preview_pan[1] += screen_dy

    def draw(self):
        """Render all UI elements and the angled 2.5D preview."""
        self.screen.blit(self.static_surface, (0, 0))

        # Draw angled 2.5D map preview if active
        if self.map_preview_active and self.current_map is not None:
            try:
                self._draw_map_preview_angled()
            except Exception as e:
                self.logger.error(f"Error drawing angled preview: {e}", exc_info=True)

        # UI manager draws UI elements (including the inspector panel)
        self.manager.draw_ui(self.screen)

    def _draw_map_preview_angled(self):
        """Render the map as square tops projected with tilt & rotation, plus side faces."""
        from main.game.data.maps.terrain import plains, forest, urban, mountains, road, highway, debris
        from .UItheme import UITheme

        preview_x = int(self.screen_width * 0.25) + 10
        preview_w = self.screen_width - preview_x
        preview_h = self.screen_height

        # preview surface (use alpha to allow UI theme edges)
        preview_surface = pygame.Surface((preview_w, preview_h), flags=pygame.SRCALPHA)
        preview_surface.fill(UITheme.BACKGROUND)

        # Terrain color mapping
        terrain_colors = {
            plains: (200, 200, 140),
            forest: (34, 139, 34),
            urban: (110, 110, 110),
            mountains: (120, 100, 80),
            road: (150, 120, 80),
            highway: (210, 180, 50),
            debris: (160, 140, 120)
        }

        map_h = len(self.current_map)
        map_w = len(self.current_map[0]) if map_h > 0 else 0

        # tile size in pixels after zoom
        tile_px = self.preview_tile_size * self.preview_zoom

        pan_x = self.preview_pan[0]
        pan_y = self.preview_pan[1]

        # center of preview in screen coordinates (inside preview_surface)
        center_x = preview_w / 2.0
        center_y = preview_h / 2.0

        angle_rad = math.radians(self.camera_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        def project_point(wx, wy, z):
            """
            Project a world point (wx, wy, elevation z in tiles) into preview_surface coords.
            wx, wy are in tile-space (tile indices). z is in elevation units (tiles).
            """
            # translate world so map center is origin
            ox = wx - (map_w / 2.0)
            oy = wy - (map_h / 2.0)

            # rotate around origin
            rx = ox * cos_a - oy * sin_a
            ry = ox * sin_a + oy * cos_a

            # project to screen pixels
            sx = rx * tile_px
            sy = ry * tile_px * self.camera_tilt

            # apply elevation offset (raise top by some pixels)
            elev_px = z * (tile_px * 0.45)  # elevation scale (tweakable)
            sy -= elev_px

            # translate to preview center + pan
            screen_x = center_x + sx + pan_x
            screen_y = center_y + sy + pan_y
            return (int(screen_x), int(screen_y))

        # determine hovered tile index
        mouse_pos = pygame.mouse.get_pos()
        preview_rect = pygame.Rect(preview_x, 0, preview_w, preview_h)
        hovered = None
        if preview_rect.collidepoint(mouse_pos):
            mx, my = mouse_pos
            # need to unproject roughly: iterate nearest tile candidate by sampling
            # Convert mouse to preview local coords
            local_x = mx - preview_x - center_x - pan_x
            local_y = my - center_y - pan_y
            # approximate inverse rotation and tilt (not perfect but good enough)
            # undo tilt
            local_y_un = local_y / (tile_px * self.camera_tilt) if tile_px * self.camera_tilt != 0 else local_y
            # map back with inverse rotation
            inv_rx = (local_x / tile_px) if tile_px != 0 else local_x
            inv_ry = local_y_un
            # rotate back
            ix = inv_rx * cos_a + inv_ry * sin_a
            iy = -inv_rx * sin_a + inv_ry * cos_a
            # convert to world tile coords
            tile_x = int(math.floor(ix + (map_w / 2.0)))
            tile_y = int(math.floor(iy + (map_h / 2.0)))
            if 0 <= tile_x < map_w and 0 <= tile_y < map_h:
                hovered = (tile_x, tile_y)

        # We'll render tiles in painter's order using depth sorting by (rotated ry + z)
        render_list = []
        for y in range(map_h):
            for x in range(map_w):
                try:
                    cell = self.current_map[y][x]
                except Exception:
                    continue
                # compute a depth key: rotated y + elevation
                ox = x - (map_w / 2.0)
                oy = y - (map_h / 2.0)
                ry = ox * sin_a + oy * cos_a
                depth = ry + getattr(cell, "elevation", 0) * 0.9
                render_list.append((depth, x, y, cell))
        render_list.sort(key=lambda r: r[0])  # back (small) -> front (large)

        # render tiles
        for _, x, y, cell in render_list:
            # compute four corner world coords (tile corner positions) and project them
            # corners in tile-space: (x,y), (x+1,y), (x+1,y+1), (x,y+1)
            elev = getattr(cell, "elevation", 0) or 0
            # top face corners (with elevation)
            p0 = project_point(x, y, elev)
            p1 = project_point(x + 1, y, elev)
            p2 = project_point(x + 1, y + 1, elev)
            p3 = project_point(x, y + 1, elev)

            # base corners at ground (z=0) used for side faces
            b0 = project_point(x, y, 0)
            b1 = project_point(x + 1, y, 0)
            b2 = project_point(x + 1, y + 1, 0)
            b3 = project_point(x, y + 1, 0)

            # cull if completely outside preview
            minx = min(p0[0], p1[0], p2[0], p3[0], b0[0], b1[0], b2[0], b3[0])
            maxx = max(p0[0], p1[0], p2[0], p3[0], b0[0], b1[0], b2[0], b3[0])
            miny = min(p0[1], p1[1], p2[1], p3[1], b0[1], b1[1], b2[1], b3[1])
            maxy = max(p0[1], p1[1], p2[1], p3[1], b0[1], b1[1], b2[1], b3[1])
            if maxx < -tile_px or minx > preview_w + tile_px or maxy < -tile_px or miny > preview_h + tile_px:
                continue

            base_color = { }.get(cell.terrain_type, None)
            # fallback using terrain mapping
            terrain_map_color = {
                # map objects are keys; use getattr .terrain_type references in mapGen
            }
            # try to get a color from terrain mapping above; if missing use default from our mapping
            base_color = terrain_map_color.get(cell.terrain_type, terrain_colors.get(cell.terrain_type, (200, 200, 140)))

            # adjust top color by elevation
            top_multiplier = 1.0 + (elev * 0.05)
            top_color = tuple(min(255, int(c * top_multiplier)) for c in base_color)

            # if hovered, darken top
            is_hover = (hovered is not None and hovered[0] == x and hovered[1] == y)
            if is_hover:
                top_color = tuple(max(0, int(c * 0.65)) for c in top_color)

            # draw top polygon
            try:
                pygame.draw.polygon(preview_surface, top_color, [p0, p1, p2, p3])
                pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [p0, p1, p2, p3], 1)
            except Exception:
                pass

            # draw a single front-facing side (choose the two top polygon points with max screen_y)
            try:
                pts = [p0, p1, p2, p3]
                base_pts = [b0, b1, b2, b3]
                # pick two points with largest y (lowest on screen) -> that's the front edge
                idx_sorted = sorted(range(4), key=lambda i: pts[i][1], reverse=True)
                i1, i2 = idx_sorted[0], idx_sorted[1]
                # ensure consistent ordering along edge
                top_a = pts[i1]
                top_b = pts[i2]
                base_a = base_pts[i1]
                base_b = base_pts[i2]
                side_color = tuple(max(0, int(c * 0.55)) for c in base_color)
                pygame.draw.polygon(preview_surface, side_color, [top_a, top_b, base_b, base_a])
                pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [top_a, top_b, base_b, base_a], 1)
            except Exception:
                pass

            # if hovered, draw tile type label centered on top polygon
            if is_hover:
                try:
                    font = pygame.font.SysFont('Arial', max(12, int(14 * self.preview_zoom)))
                    tname = getattr(cell.terrain_type, "name", None) or str(getattr(cell, "terrain_type", cell))
                    ttext = str(tname).capitalize()
                    # compute centroid of top polygon
                    cx = (p0[0] + p1[0] + p2[0] + p3[0]) // 4
                    cy = (p0[1] + p1[1] + p2[1] + p3[1]) // 4
                    text_surf = font.render(ttext, True, (255, 255, 255))
                    text_rect = text_surf.get_rect(center=(cx, cy))
                    preview_surface.blit(text_surf, text_rect)
                except Exception:
                    pass

        # blit preview surface to screen
        self.screen.blit(preview_surface, (preview_x, 0))

        # draw preview border
        preview_outer = pygame.Rect(preview_x, 0, preview_w, preview_h)
        try:
            pygame.draw.rect(self.screen, UITheme.SECONDARY, preview_outer, 2)
        except Exception:
            pass

        # update inspector UI values (hovered tile)
        try:
            if hovered and 0 <= hovered[0] < map_w and 0 <= hovered[1] < map_h:
                hcell = self.current_map[hovered[1]][hovered[0]]
                tname = getattr(hcell.terrain_type, "name", None) or str(getattr(hcell, "terrain_type", hcell))
                coord_text = f"Coords: {getattr(hcell, 'x', hovered[0])}, {getattr(hcell, 'y', hovered[1])}"
                elev_text = f"Elevation: {getattr(hcell, 'elevation', 0)}"
                if getattr(self, "preview_tile_type", None):
                    try:
                        self.preview_tile_type.set_text(f"Type: {str(tname).capitalize()}")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_coords", None):
                    try:
                        self.preview_tile_coords.set_text(coord_text)
                    except Exception:
                        pass
                if getattr(self, "preview_tile_elev", None):
                    try:
                        self.preview_tile_elev.set_text(elev_text)
                    except Exception:
                        pass
                if getattr(self, "preview_tile_note", None):
                    try:
                        note = "Building" if getattr(hcell, "is_building", False) else ("Ramp" if getattr(hcell, "is_ramp", False) else "")
                        self.preview_tile_note.set_text(note)
                    except Exception:
                        pass
                # color swatch
                swatch = getattr(self, "preview_tile_color", None)
                if swatch:
                    try:
                        col = terrain_colors.get(hcell.terrain_type, (200, 200, 140))
                        swatch.background_colour = col
                        try:
                            swatch.rebuild()
                        except Exception:
                            pass
                    except Exception:
                        pass
            else:
                # clear inspector values
                if getattr(self, "preview_tile_type", None):
                    try:
                        self.preview_tile_type.set_text("Type: -")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_coords", None):
                    try:
                        self.preview_tile_coords.set_text("Coords: -")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_elev", None):
                    try:
                        self.preview_tile_elev.set_text("Elevation: -")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_note", None):
                    try:
                        self.preview_tile_note.set_text("Hover a tile in the preview to see info here.")
                    except Exception:
                        pass
                swatch = getattr(self, "preview_tile_color", None)
                if swatch:
                    try:
                        swatch.background_colour = UITheme.BACKGROUND
                        try:
                            swatch.rebuild()
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception:
            pass

        # draw control hint if mouse over preview
        if self.mouse_over_preview:
            try:
                font = pygame.font.SysFont('Arial', 14)
                hint = f"WASD/Arrows: Move | Wheel: Zoom | Middle-drag: Rotate | Rot: {int(self.camera_rotation)}°"
                text = font.render(hint, True, UITheme.TEXT)
                text_rect = text.get_rect(centerx=preview_x + preview_w // 2, top=10)
                bg_rect = text_rect.inflate(20, 10)
                pygame.draw.rect(self.screen, (UITheme.PRIMARY if hasattr(UITheme, "PRIMARY") else (50,50,50)), bg_rect)
                pygame.draw.rect(self.screen, (UITheme.SECONDARY if hasattr(UITheme, "SECONDARY") else (200,200,200)), bg_rect, 1)
                self.screen.blit(text, text_rect)
            except Exception:
                pass

    def _rotate_coords(self, x, y):
        """Compatibility shim (not used by angled renderer)."""
        from main.config import MAP_WIDTH, MAP_HEIGHT

        cx = MAP_WIDTH / 2
        cy = MAP_HEIGHT / 2

        tx = x - cx
        ty = y - cy

        angle_rad = math.radians(self.camera_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        return rx + cx, ry + cy
