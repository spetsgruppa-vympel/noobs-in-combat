import os
import pygame
import pygame_gui
import math
from main.config import resolution_converter, get_project_root, get_logger, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT
from main.game.ui.camera import Camera, CameraMode


class MainMenuManager:
    """
    Main menu manager with complete map generator UI and 3D preview.

    FEATURES:
    - All customization sliders (Task 2: Complete)
    - Fixed camera using camera.py: WASD=move, middle-drag=rotate, wheel=zoom (Task 3: Complete)
    - 3D buildings and 45-degree ramps (Task 4: Complete)
    """

    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height, theme=None):
        """Initialize the main menu manager."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MainMenuManager with camera system")

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

        # Map generator state
        self.map_generator = None
        self.current_map = None

        # TASK 3: Camera integration
        self.camera = None
        self.preview_tile_size = TILE_SIZE if TILE_SIZE else 32
        self.camera_rotation = 30.0  # Initial rotation
        self.preview_rotating = False
        self.preview_last_mouse = (0, 0)

        self.map_preview_active = False
        self.mouse_over_preview = False
        self.map_size = MAP_WIDTH if MAP_WIDTH else 40

        self.static_surface = pygame.Surface((screen_width, screen_height))
        self.logger.debug("Created static surface")

        self.ui_elements = []
        self.element_callbacks = {}

        self._load_units()
        self.logger.info("Creating main menu...")
        self.create_main_menu()
        self.logger.info("Main menu created")

        try:
            pygame.display.set_icon(self.logo)
        except Exception:
            pass

        self.draw_static()

    def _load_units(self):
        """Load all available units."""
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
        except Exception as e:
            self.logger.error(f"Failed to load units: {e}")
            self.all_units = []

    def _preload_fonts(self):
        """Preload commonly used fonts."""
        try:
            font_dict = self.manager.get_theme().get_font_dictionary()
            font_configs = [
                (14, 'noto_sans', True, False, True),
                (14, 'noto_sans', False, True, True),
                (12, 'noto_sans', False, False, True),
            ]
            for size, name, bold, italic, aa in font_configs:
                try:
                    font_dict.preload_font(size, name, bold, italic, aa)
                except Exception:
                    pass
        except Exception:
            pass

    @property
    def logo(self):
        """Lazy-load logo."""
        if self._logo is None:
            logo_path = os.path.join(get_project_root(), "assets", "ui", "game_icon.jpg")
            self._logo = pygame.image.load(logo_path).convert_alpha()
            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
        return self._logo

    @property
    def background(self):
        """Lazy-load background."""
        if self._background is None:
            bg_path = os.path.join(get_project_root(), "assets", "ui", "menu_background.jpg")
            bg = pygame.image.load(bg_path).convert()
            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
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
        from main.game.ui.UItheme import UITheme

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
            element = ui_class(rect=rect, manager=manager, **kwargs)
            return element

        params = {"relative_rect": rect, "manager": manager, **kwargs}
        if container:
            params["container"] = container

        if element_type == "dropdown":
            options = params.pop("options_list", []) or []
            if not options:
                options = ["<no units available>"]
            starting_option = params.pop("starting_option", None)
            if starting_option not in options:
                starting_option = options[0]
            other_kwargs = {k: v for k, v in params.items() if k not in ("relative_rect", "manager")}
            element = ui_class(options, starting_option, params["relative_rect"], params["manager"], **other_kwargs)
            self.ui_elements.append(element)
            return element

        elif element_type == "slider":
            start_value = params.pop("start_value", 0.0)
            value_range = params.pop("value_range", (0.0, 1.0))
            other_kwargs = {k: v for k, v in params.items() if k not in ("relative_rect", "manager")}
            element = ui_class(
                relative_rect=params["relative_rect"],
                start_value=start_value,
                value_range=value_range,
                manager=params["manager"],
                **other_kwargs
            )
            self.ui_elements.append(element)
            return element

        element = ui_class(**params)
        self.ui_elements.append(element)
        return element

    def create_from_spec(self, spec):
        """Recursively create UI from spec."""
        elem_type = spec.get("type", "panel")
        root_rect = pygame.Rect(*spec["rect"])
        obj_id = f'#{spec.get("id", "")}' if spec.get("id") else None

        root_element = self.init_ui(elem_type, root_rect, self.manager, object_id=obj_id)

        if spec.get("id"):
            setattr(self, spec["id"], root_element)

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

        return root_element

    def load_menu(self, spec):
        """Load new menu."""
        if self.current_menu:
            self.clear_element("all")

        root = self.create_from_spec(spec)
        self.menu_stack.append(spec)
        self.current_menu = root
        return root

    def draw_static(self):
        """Pre-render static background."""
        self.static_surface.blit(self.background, (0, 0))
        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - logo_offset))
        self.static_surface.blit(self.logo, logo_rect)

    def clear_element(self, element_clear):
        """Clear UI elements."""
        if element_clear == "all":
            for element in list(self.ui_elements):
                try:
                    element.kill()
                except Exception:
                    pass
            self.ui_elements.clear()
            self.loadout_unit_buttons.clear()
            self.selected_loadout_index = None
            self.map_preview_active = False
            self.mouse_over_preview = False
            self.map_generator = None
            self.current_map = None
            self.camera = None
            self.camera_rotation = 30.0
            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
            self._mapgen_btn = None
            self.menu_panel = None
            self._quit_dialog = None
            self.element_callbacks.clear()

    def create_main_menu(self):
        """Build menu specifications."""
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

        # Main menu
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

        # Singleplayer menu
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

        # Loadout menu
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
                    "html_text": "<b>Select a unit</b>"
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

        # TASK 2: Complete map generator UI with ALL settings
        mapgen_controls_w = int(self.screen_width * 0.22)

        y_pos = 60
        spacing = 50

        children = [
            {"type": "label", "id": "mapgen_title", "rect": [10, 20, mapgen_controls_w - 20, 30],
             "text": "Map Generator"},

            # Map size
            {"type": "label", "id": "map_size_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": f"Map Size: {self.map_size}"},
            {"type": "slider", "id": "map_size_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": float(self.map_size), "value_range": (20.0, 100.0), "callback": "map_size_changed"},
        ]
        y_pos += spacing + 10

        # Terrain density sliders
        terrain_settings = [
            ("forest", "Forest", 0.25, (0.0, 0.5)),
            ("urban", "Urban", 0.15, (0.0, 0.3)),
            ("mountain", "Mountain", 0.10, (0.0, 0.25)),
            ("road", "Road", 0.06, (0.0, 0.15)),
            ("debris", "Debris", 0.05, (0.0, 0.1)),
        ]

        for setting_id, label, default, value_range in terrain_settings:
            children.extend([
                {"type": "label", "id": f"{setting_id}_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
                 "text": f"{label}: {int(default * 100)}%"},
                {"type": "slider", "id": f"{setting_id}_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
                 "start_value": default, "value_range": value_range, "callback": f"{setting_id}_changed"},
            ])
            y_pos += spacing

        # Elevation settings
        children.extend([
            {"type": "label", "id": "elevation_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": "Elevation: 20%"},
            {"type": "slider", "id": "elevation_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": 0.20, "value_range": (0.0, 0.4), "callback": "elevation_changed"},
        ])
        y_pos += spacing

        children.extend([
            {"type": "label", "id": "max_elev_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": "Max Elevation: 3"},
            {"type": "slider", "id": "max_elev_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": 3.0, "value_range": (1.0, 5.0), "callback": "max_elev_changed"},
        ])
        y_pos += spacing

        # Building settings
        children.extend([
            {"type": "label", "id": "building_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": "Buildings: 8%"},
            {"type": "slider", "id": "building_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": 0.08, "value_range": (0.0, 0.2), "callback": "building_changed"},
        ])
        y_pos += spacing

        children.extend([
            {"type": "label", "id": "control_zones_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": "Control Zones: 5"},
            {"type": "slider", "id": "control_zones_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": 5.0, "value_range": (3.0, 11.0), "callback": "control_zones_changed"},
        ])
        y_pos += spacing

        # Smoothing
        children.extend([
            {"type": "label", "id": "smoothing_label", "rect": [10, y_pos, mapgen_controls_w - 20, 20],
             "text": "Smoothing: 2"},
            {"type": "slider", "id": "smoothing_slider", "rect": [10, y_pos + 25, mapgen_controls_w - 20, 20],
             "start_value": 2.0, "value_range": (0.0, 5.0), "callback": "smoothing_changed"},
        ])
        y_pos += spacing + 10

        # Generate button
        children.extend([
            {"type": "button", "id": "generate_map_btn", "rect": [10, y_pos, mapgen_controls_w - 20, 40],
             "text": "Generate Map", "callback": "generate_map"},
        ])
        y_pos += 50

        # Instructions
        children.extend([
            {"type": "textbox", "id": "mapgen_instructions", "rect": [10, y_pos, mapgen_controls_w - 20, 180],
             "html_text": "<b>Controls:</b><br>WASD: Move camera<br>Wheel: Zoom<br>Middle-drag: Rotate<br><br><b>3D View</b><br>• Buildings shown as boxes<br>• Ramps shown as slopes"},
        ])

        children.append(back_btn())

        self.specs['mapgen_spec'] = {
            "type": "panel",
            "id": "mapgen_panel",
            "rect": [0, 0, mapgen_controls_w, self.screen_height],
            "children": children
        }

        root = self.load_menu(self.specs['menu_spec'])
        self.menu_panel = getattr(self, 'menu_panel', root)

    # Callback methods
    def singleplayer_press(self, element=None):
        """Navigate to singleplayer."""
        self.load_menu(self.specs['singleplayer_main_spec'])

    def multiplayer_press(self, element=None):
        """Multiplayer (not implemented)."""
        pass

    def start_game_press(self, element=None):
        """Start game."""
        pass

    def loadout_press(self, element=None):
        """Navigate to loadout menu."""
        self.load_menu(self.specs['loadout_spec'])
        dd = getattr(self, "unit_dropdown", None)
        if dd and self.all_units:
            try:
                self.selected_unit = self.all_units[0]
                self.unit_selected(dd)
            except Exception:
                pass
        self.refresh_loadout_view()

    def mapgen_press(self, element=None):
        """Navigate to map generator with camera initialization."""
        self.load_menu(self.specs['mapgen_spec'])
        self.map_preview_active = True

        # TASK 3: Initialize camera system
        preview_x = int(self.screen_width * 0.22) + 10
        preview_w = self.screen_width - preview_x

        self.camera = Camera(
            screen_width=preview_w,
            screen_height=self.screen_height,
            start_x=self.map_size * self.preview_tile_size / 2.0,
            start_y=self.map_size * self.preview_tile_size / 2.0,
            zoom=1.0,
            mode=CameraMode.ORTHOGRAPHIC
        )

        self.camera_rotation = 30.0
        self.preview_rotating = False

        # Create inspector UI
        inspector_w = 300
        inspector_x = preview_x + preview_w - inspector_w - 10

        try:
            self.preview_info_panel = self.init_ui("panel", pygame.Rect(inspector_x, 20, inspector_w, 200),
                                                   self.manager)
            self.preview_tile_title = self.init_ui("label", pygame.Rect(10, 10, inspector_w - 20, 24),
                                                   self.manager, container=self.preview_info_panel,
                                                   text="Tile Inspector")
            self.preview_tile_type = self.init_ui("label", pygame.Rect(10, 40, inspector_w - 20, 28),
                                                  self.manager, container=self.preview_info_panel, text="Type: -")
            self.preview_tile_coords = self.init_ui("label", pygame.Rect(10, 70, inspector_w - 20, 22),
                                                    self.manager, container=self.preview_info_panel, text="Coords: -")
            self.preview_tile_elev = self.init_ui("label", pygame.Rect(10, 96, inspector_w - 20, 22),
                                                  self.manager, container=self.preview_info_panel, text="Elevation: -")
            self.preview_tile_color = self.init_ui("panel", pygame.Rect(10, 124, 40, 40),
                                                   self.manager, container=self.preview_info_panel)
            self.preview_tile_note = self.init_ui("label", pygame.Rect(60, 124, inspector_w - 70, 60),
                                                  self.manager, container=self.preview_info_panel,
                                                  text="Hover over map")
        except Exception as e:
            self.logger.warning(f"Failed to create inspector: {e}")

        self.generate_map()

    # TASK 2: All slider change handlers
    def map_size_changed(self, element=None):
        """Handle map size change."""
        if element:
            size = int(round(element.get_current_value()))
            self.map_size = max(20, min(100, size))
            label = getattr(self, "map_size_label", None)
            if label:
                try:
                    label.set_text(f"Map Size: {self.map_size}")
                except Exception:
                    pass
            self.generate_map()

    def forest_changed(self, element=None):
        """Handle forest density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "forest_label", None)
            if label:
                try:
                    label.set_text(f"Forest: {int(val * 100)}%")
                except Exception:
                    pass

    def urban_changed(self, element=None):
        """Handle urban density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "urban_label", None)
            if label:
                try:
                    label.set_text(f"Urban: {int(val * 100)}%")
                except Exception:
                    pass

    def mountain_changed(self, element=None):
        """Handle mountain density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "mountain_label", None)
            if label:
                try:
                    label.set_text(f"Mountain: {int(val * 100)}%")
                except Exception:
                    pass

    def road_changed(self, element=None):
        """Handle road density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "road_label", None)
            if label:
                try:
                    label.set_text(f"Road: {int(val * 100)}%")
                except Exception:
                    pass

    def debris_changed(self, element=None):
        """Handle debris density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "debris_label", None)
            if label:
                try:
                    label.set_text(f"Debris: {int(val * 100)}%")
                except Exception:
                    pass

    def elevation_changed(self, element=None):
        """Handle elevation density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "elevation_label", None)
            if label:
                try:
                    label.set_text(f"Elevation: {int(val * 100)}%")
                except Exception:
                    pass

    def max_elev_changed(self, element=None):
        """Handle max elevation change."""
        if element:
            val = int(round(element.get_current_value()))
            label = getattr(self, "max_elev_label", None)
            if label:
                try:
                    label.set_text(f"Max Elevation: {val}")
                except Exception:
                    pass

    def building_changed(self, element=None):
        """Handle building density change."""
        if element:
            val = element.get_current_value()
            label = getattr(self, "building_label", None)
            if label:
                try:
                    label.set_text(f"Buildings: {int(val * 100)}%")
                except Exception:
                    pass

    def control_zones_changed(self, element=None):
        """Handle control zones change."""
        if element:
            val = int(round(element.get_current_value()))
            # Make odd
            if val % 2 == 0:
                val += 1
            label = getattr(self, "control_zones_label", None)
            if label:
                try:
                    label.set_text(f"Control Zones: {val}")
                except Exception:
                    pass

    def smoothing_changed(self, element=None):
        """Handle smoothing change."""
        if element:
            val = int(round(element.get_current_value()))
            label = getattr(self, "smoothing_label", None)
            if label:
                try:
                    label.set_text(f"Smoothing: {val}")
                except Exception:
                    pass

    def generate_map(self, element=None):
        """Generate map with current settings."""
        from main.game.data.maps.config import MapConfig
        from main.game.data.maps.mapGen import MapGenerator

        # Gather all settings from sliders
        config_params = {
            'width': self.map_size,
            'height': self.map_size,
        }

        # Get slider values
        sliders = {
            'forest_slider': 'forest_density',
            'urban_slider': 'urban_density',
            'mountain_slider': 'mountain_density',
            'road_slider': 'road_density',
            'debris_slider': 'debris_density',
            'elevation_slider': 'elevation_density',
            'building_slider': 'building_density',
            'max_elev_slider': 'max_elevation',
            'control_zones_slider': 'control_zone_count',
            'smoothing_slider': 'smoothing_passes',
        }

        for slider_id, param_name in sliders.items():
            slider = getattr(self, slider_id, None)
            if slider:
                val = slider.get_current_value()
                if param_name in ['max_elevation', 'control_zone_count', 'smoothing_passes']:
                    val = int(round(val))
                    if param_name == 'control_zone_count' and val % 2 == 0:
                        val += 1
                config_params[param_name] = val

        config = MapConfig(**config_params)
        self.map_generator = MapGenerator(config)
        self.current_map = self.map_generator.generate()

        # TASK 3: Reset camera to center of map
        if self.camera:
            map_center_world = self.map_size * self.preview_tile_size / 2.0
            self.camera.pan_to(map_center_world, map_center_world, smooth=False)
            self.camera.set_zoom(1.0)

        stats = self.map_generator.get_statistics()
        self.logger.info(f"Map generated: {stats}")

    def back_press(self, element=None):
        """Navigate back."""
        if len(self.menu_stack) > 1:
            self.clear_element("all")
            self.menu_stack.pop()
            spec = self.menu_stack[-1]
            self.current_menu = self.create_from_spec(spec)

    def refresh_loadout_view(self):
        """Update loadout display."""
        for btn in self.loadout_unit_buttons:
            btn.kill()
        self.loadout_unit_buttons.clear()

        panel = getattr(self, "loadout_list_panel", None)
        if not panel:
            return

        if not self.player_loadout:
            empty_box = self.init_ui("textbox", pygame.Rect(5, 5, 270, 100), self.manager,
                                     container=panel, html_text="<b>No units selected</b>")
            self.loadout_unit_buttons.append(empty_box)
            return

        button_height = 40
        button_spacing = 5
        start_y = 10

        for i, unit in enumerate(self.player_loadout):
            y_pos = start_y + (i * (button_height + button_spacing))
            button_text = f"{i + 1}. {unit.name} - {unit.cost if unit.cost else 'N/A'}"
            unit_btn = self.init_ui("button", pygame.Rect(10, y_pos, 260, button_height),
                                    self.manager, container=panel, text=button_text)
            self.loadout_unit_buttons.append(unit_btn)

            def make_callback(index):
                return lambda elem=None: self.select_loadout_unit(index)

            self.element_callbacks[unit_btn] = make_callback(i)

    def unit_selected(self, element=None):
        """Update unit description when dropdown changes."""
        dropdown = element or getattr(self, "unit_dropdown", None)
        if not dropdown:
            return

        selection = getattr(dropdown, "selected_option", None)
        if isinstance(selection, tuple):
            selection = selection[0] if selection else None

        if not selection:
            return

        unit = next((u for u in self.all_units if u.name == selection), None)
        if not unit:
            return

        self.selected_unit = unit
        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            weapons = ", ".join([w.name for w in unit.weapons]) if unit.weapons else "None"
            perks = ", ".join([p.name for p in unit.perks]) if unit.perks else "None"
            lines = [
                f"<b>{unit.name}</b>",
                f"<br><b>Cost:</b> {unit.cost}",
                f"<b>HP:</b> {unit.hp}",
                f"<b>Armor:</b> {unit.armor}",
                f"<br><b>Weapons:</b> {weapons}",
                f"<b>Perks:</b> {perks}",
            ]
            desc_box.set_text("<br>".join(lines))

    def add_to_loadout(self, element=None):
        """Add selected unit to loadout."""
        unit = self.selected_unit
        if not unit or len(self.player_loadout) >= 6:
            return
        if any(u.name == unit.name for u in self.player_loadout):
            return
        self.player_loadout.append(unit)
        self.refresh_loadout_view()

    def select_loadout_unit(self, index):
        """Select unit from loadout."""
        if 0 <= index < len(self.player_loadout):
            self.selected_loadout_index = index
            unit = self.player_loadout[index]
            desc_box = getattr(self, "unit_description", None)
            if desc_box:
                weapons = ", ".join([w.name for w in unit.weapons]) if unit.weapons else "None"
                lines = [
                    f"<b>{unit.name}</b> <em>(In Loadout)</em>",
                    f"<br><b>Cost:</b> {unit.cost}",
                    f"<b>HP:</b> {unit.hp}",
                    f"<br><b>Weapons:</b> {weapons}",
                ]
                desc_box.set_text("<br>".join(lines))
            self.refresh_loadout_view()

    def remove_selected_unit(self, element=None):
        """Remove selected unit from loadout."""
        if self.selected_loadout_index is not None and 0 <= self.selected_loadout_index < len(self.player_loadout):
            self.player_loadout.pop(self.selected_loadout_index)
            self.selected_loadout_index = None
        elif self.player_loadout:
            self.player_loadout.pop()
        self.refresh_loadout_view()

    def process_events(self, events):
        """Process events including camera controls."""
        # Update mouse position for preview
        if self.map_preview_active:
            mouse_pos = pygame.mouse.get_pos()
            preview_x = int(self.screen_width * 0.22) + 10
            preview_rect = pygame.Rect(preview_x, 0, self.screen_width - preview_x, self.screen_height)
            self.mouse_over_preview = preview_rect.collidepoint(mouse_pos)

        for event in events:
            self.manager.process_events(event)

            if event.type == pygame.USEREVENT:
                user_type = getattr(event, "user_type", None)
                slider_event = pygame_gui.UI_HORIZONTAL_SLIDER_MOVED if hasattr(pygame_gui,
                                                                                "UI_HORIZONTAL_SLIDER_MOVED") else None
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

            # TASK 3: Camera controls - Middle mouse rotation
            if self.map_preview_active and self.camera:
                # Middle mouse press (start rotating)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                    if self.mouse_over_preview:
                        self.preview_rotating = True
                        self.preview_last_mouse = event.pos
                        pygame.mouse.get_rel()

                # Middle mouse release (stop rotating)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                    self.preview_rotating = False

                # Mouse wheel zoom
                if event.type == pygame.MOUSEWHEEL and self.mouse_over_preview:
                    if event.y > 0:
                        self.camera.zoom_in(steps=1)
                    else:
                        self.camera.zoom_out(steps=1)

                # Mouse motion while rotating
                if event.type == pygame.MOUSEMOTION and self.preview_rotating:
                    mx, my = event.pos
                    lx, ly = self.preview_last_mouse
                    dx = mx - lx
                    self.camera_rotation = (self.camera_rotation + dx * 0.45) % 360
                    self.preview_last_mouse = (mx, my)

            # Quit dialog
            if event.type in (pygame.KEYDOWN, pygame.QUIT):
                if event.type == pygame.QUIT:
                    if self._quit_dialog:
                        return "quit"
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self._quit_dialog is None:
                        dialog_w = resolution_converter(330, "x")
                        dialog_h = resolution_converter(200, "y")
                        dialog_x = (self.screen_width - dialog_w) // 2
                        dialog_y = (self.screen_height - dialog_h) // 2
                        self._quit_dialog = self.init_ui(
                            "confirmation_dialog",
                            pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h),
                            self.manager,
                            window_title="Confirm Quit",
                            action_long_desc="Are you sure?",
                            action_short_name="Quit",
                            blocking=True
                        )

            if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None
                    return "quit"

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._quit_dialog:
                    self._quit_dialog = None

        return None

    def update_camera(self, dt):
        """TASK 3: Update camera with WASD movement using camera.py."""
        if not self.map_preview_active or not self.mouse_over_preview or not self.camera:
            return

        keys = pygame.key.get_pressed()
        move_speed = 300.0 * dt  # Pixels per second

        # WASD camera panning
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.camera.pan(0, -move_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.camera.pan(0, move_speed)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.camera.pan(-move_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.camera.pan(move_speed, 0)

        # Update camera smooth interpolation
        self.camera.update(dt)

    def draw(self):
        """Draw all UI and map preview."""
        self.screen.blit(self.static_surface, (0, 0))

        # TASK 4: Draw 3D map preview with buildings and ramps
        if self.map_preview_active and self.current_map is not None and self.camera:
            try:
                self._draw_map_preview_3d()
            except Exception as e:
                self.logger.error(f"Error drawing 3D preview: {e}", exc_info=True)

        self.manager.draw_ui(self.screen)

    def _draw_map_preview_3d(self):
        """TASK 4: Render map with 3D buildings and 45-degree ramps using camera.py."""
        from main.game.data.maps.terrain import plains, forest, urban, mountains, road, highway, debris
        from main.game.ui.UItheme import UITheme

        preview_x = int(self.screen_width * 0.22) + 10
        preview_w = self.screen_width - preview_x
        preview_h = self.screen_height

        preview_surface = pygame.Surface((preview_w, preview_h), flags=pygame.SRCALPHA)
        preview_surface.fill(UITheme.BACKGROUND)

        terrain_colors = {
            plains: (200, 200, 140),
            forest: (34, 139, 34),
            urban: (110, 110, 110),
            mountains: (120, 100, 80),
            road: (150, 120, 80),
            debris: (160, 140, 120)
        }

        map_h = len(self.current_map)
        map_w = len(self.current_map[0]) if map_h > 0 else 0

        tile_px = self.preview_tile_size * self.camera.zoom

        # Camera tilt for pseudo-3D
        camera_tilt = 0.55

        angle_rad = math.radians(self.camera_rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        def project_point(wx, wy, z):
            """Project world point to screen using camera transform."""
            # Convert tile coordinates to world pixels
            world_x = wx * self.preview_tile_size
            world_y = wy * self.preview_tile_size

            # Apply elevation
            world_z = z * (self.preview_tile_size * 0.45)

            # Transform to screen using camera
            screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

            # Apply rotation
            offset_x = screen_x - preview_w / 2.0
            offset_y = screen_y - preview_h / 2.0

            rx = offset_x * cos_a - offset_y * sin_a
            ry = offset_x * sin_a + offset_y * cos_a

            # Apply tilt and elevation
            ry = ry * camera_tilt - world_z * self.camera.zoom

            final_x = int(preview_w / 2.0 + rx)
            final_y = int(preview_h / 2.0 + ry)

            return (final_x, final_y)

        # Find hovered tile
        mouse_pos = pygame.mouse.get_pos()
        hovered = None
        if self.mouse_over_preview:
            mx, my = mouse_pos
            # Simple raycast to find tile under mouse
            # Test all tiles and find closest to mouse
            min_dist = float('inf')
            for y in range(map_h):
                for x in range(map_w):
                    try:
                        cell = self.current_map[y][x]
                        elev = getattr(cell, "elevation", 0) or 0
                        # Get center of tile
                        cx, cy = project_point(x + 0.5, y + 0.5, elev)
                        dist = math.sqrt((mx - preview_x - cx) ** 2 + (my - cy) ** 2)
                        if dist < min_dist and dist < tile_px:
                            min_dist = dist
                            hovered = (x, y)
                    except Exception:
                        continue

        # Render tiles in painter's order (back to front)
        render_list = []
        for y in range(map_h):
            for x in range(map_w):
                try:
                    cell = self.current_map[y][x]
                except Exception:
                    continue

                # Calculate depth for sorting
                elev = getattr(cell, "elevation", 0) or 0
                rx = (x - map_w / 2.0) * cos_a - (y - map_h / 2.0) * sin_a
                ry = (x - map_w / 2.0) * sin_a + (y - map_h / 2.0) * cos_a
                depth = ry + elev * 0.9
                render_list.append((depth, x, y, cell))

        render_list.sort(key=lambda r: r[0])

        for _, x, y, cell in render_list:
            elev = getattr(cell, "elevation", 0) or 0

            # Top face corners
            p0 = project_point(x, y, elev)
            p1 = project_point(x + 1, y, elev)
            p2 = project_point(x + 1, y + 1, elev)
            p3 = project_point(x, y + 1, elev)

            # Base corners (ground level)
            b0 = project_point(x, y, 0)
            b1 = project_point(x + 1, y, 0)
            b2 = project_point(x + 1, y + 1, 0)
            b3 = project_point(x, y + 1, 0)

            # Culling
            minx = min(p0[0], p1[0], p2[0], p3[0], b0[0], b1[0], b2[0], b3[0])
            maxx = max(p0[0], p1[0], p2[0], p3[0], b0[0], b1[0], b2[0], b3[0])
            miny = min(p0[1], p1[1], p2[1], p3[1], b0[1], b1[1], b2[1], b3[1])
            maxy = max(p0[1], p1[1], p2[1], p3[1], b0[1], b1[1], b2[1], b3[1])
            if maxx < -tile_px or minx > preview_w + tile_px or maxy < -tile_px or miny > preview_h + tile_px:
                continue

            base_color = terrain_colors.get(cell.terrain_type, (200, 200, 140))
            top_multiplier = 1.0 + (elev * 0.05)
            top_color = tuple(min(255, int(c * top_multiplier)) for c in base_color)

            is_hover = (hovered is not None and hovered[0] == x and hovered[1] == y)
            if is_hover:
                top_color = tuple(max(0, int(c * 0.65)) for c in top_color)

            # TASK 4: Draw ramps as 45-degree slopes
            if cell.is_ramp:
                ramp_dir = getattr(cell, 'ramp_direction', None)
                ramp_to_elev = getattr(cell, 'ramp_elevation_to', elev + 1)

                # Get elevated edge based on direction
                if ramp_dir == 'north':
                    ramp_top = [project_point(x, y, ramp_to_elev), project_point(x + 1, y, ramp_to_elev)]
                    ramp_bottom = [p3, p2]
                elif ramp_dir == 'south':
                    ramp_top = [project_point(x, y + 1, ramp_to_elev), project_point(x + 1, y + 1, ramp_to_elev)]
                    ramp_bottom = [p0, p1]
                elif ramp_dir == 'east':
                    ramp_top = [project_point(x + 1, y, ramp_to_elev), project_point(x + 1, y + 1, ramp_to_elev)]
                    ramp_bottom = [p0, p3]
                elif ramp_dir == 'west':
                    ramp_top = [project_point(x, y, ramp_to_elev), project_point(x, y + 1, ramp_to_elev)]
                    ramp_bottom = [p1, p2]
                else:
                    ramp_top = [p0, p1]
                    ramp_bottom = [p3, p2]

                # Draw ramp surface (angled quad)
                ramp_color = tuple(max(0, int(c * 0.8)) for c in base_color)
                try:
                    pygame.draw.polygon(preview_surface, ramp_color,
                                        ramp_top + list(reversed(ramp_bottom)))
                    pygame.draw.polygon(preview_surface, UITheme.SECONDARY,
                                        ramp_top + list(reversed(ramp_bottom)), 2)
                except Exception:
                    pass

            # TASK 4: Draw buildings as 3D boxes
            elif cell.is_building:
                building_height = 2.0  # Extra height units

                # Building top corners (elevated)
                bt0 = project_point(x + 0.2, y + 0.2, elev + building_height)
                bt1 = project_point(x + 0.8, y + 0.2, elev + building_height)
                bt2 = project_point(x + 0.8, y + 0.8, elev + building_height)
                bt3 = project_point(x + 0.2, y + 0.8, elev + building_height)

                # Building base corners
                bb0 = project_point(x + 0.2, y + 0.2, elev)
                bb1 = project_point(x + 0.8, y + 0.2, elev)
                bb2 = project_point(x + 0.8, y + 0.8, elev)
                bb3 = project_point(x + 0.2, y + 0.8, elev)

                # Building colors
                building_color = (80, 80, 100)
                building_top_color = (100, 100, 120)

                # Draw visible wall (front-facing)
                try:
                    pts = [bt0, bt1, bt2, bt3]
                    base_pts = [bb0, bb1, bb2, bb3]

                    # Find front edge (furthest back in screen space)
                    idx_sorted = sorted(range(4), key=lambda i: pts[i][1], reverse=True)
                    i1, i2 = idx_sorted[0], idx_sorted[1]

                    top_a = pts[i1]
                    top_b = pts[i2]
                    base_a = base_pts[i1]
                    base_b = base_pts[i2]

                    # Draw wall quad
                    pygame.draw.polygon(preview_surface, building_color, [top_a, top_b, base_b, base_a])
                    pygame.draw.polygon(preview_surface, (60, 60, 80), [top_a, top_b, base_b, base_a], 1)
                except Exception:
                    pass

                # Draw building top
                try:
                    pygame.draw.polygon(preview_surface, building_top_color, [bt0, bt1, bt2, bt3])
                    pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [bt0, bt1, bt2, bt3], 1)
                except Exception:
                    pass

                # Draw ground tile underneath
                try:
                    pygame.draw.polygon(preview_surface, top_color, [p0, p1, p2, p3])
                    pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [p0, p1, p2, p3], 1)
                except Exception:
                    pass

            else:
                # Regular tile - draw top face
                try:
                    pygame.draw.polygon(preview_surface, top_color, [p0, p1, p2, p3])
                    pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [p0, p1, p2, p3], 1)
                except Exception:
                    pass

                # Draw side face for elevated tiles
                if elev > 0:
                    try:
                        pts = [p0, p1, p2, p3]
                        base_pts = [b0, b1, b2, b3]

                        # Find visible edge
                        idx_sorted = sorted(range(4), key=lambda i: pts[i][1], reverse=True)
                        i1, i2 = idx_sorted[0], idx_sorted[1]

                        top_a = pts[i1]
                        top_b = pts[i2]
                        base_a = base_pts[i1]
                        base_b = base_pts[i2]

                        side_color = tuple(max(0, int(c * 0.55)) for c in base_color)
                        pygame.draw.polygon(preview_surface, side_color, [top_a, top_b, base_b, base_a])
                        pygame.draw.polygon(preview_surface, UITheme.SECONDARY, [top_a, top_b, base_b, base_a], 1)
                    except Exception:
                        pass

            # Draw tile type label if hovered
            if is_hover:
                try:
                    font = pygame.font.SysFont('Arial', max(10, int(12 * self.camera.zoom)))
                    tname = getattr(cell.terrain_type, "name", str(cell.terrain_type)).capitalize()
                    cx = (p0[0] + p1[0] + p2[0] + p3[0]) // 4
                    cy = (p0[1] + p1[1] + p2[1] + p3[1]) // 4
                    text_surf = font.render(tname, True, (255, 255, 255))
                    text_rect = text_surf.get_rect(center=(cx, cy))
                    preview_surface.blit(text_surf, text_rect)
                except Exception:
                    pass

        # Blit preview to screen
        self.screen.blit(preview_surface, (preview_x, 0))

        # Draw border
        preview_outer = pygame.Rect(preview_x, 0, preview_w, preview_h)
        try:
            pygame.draw.rect(self.screen, UITheme.SECONDARY, preview_outer, 2)
        except Exception:
            pass

        # Update inspector
        try:
            if hovered and 0 <= hovered[0] < map_w and 0 <= hovered[1] < map_h:
                hcell = self.current_map[hovered[1]][hovered[0]]
                tname = getattr(hcell.terrain_type, "name", str(hcell.terrain_type)).capitalize()
                if getattr(self, "preview_tile_type", None):
                    try:
                        self.preview_tile_type.set_text(f"Type: {tname}")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_coords", None):
                    try:
                        self.preview_tile_coords.set_text(f"Coords: {hovered[0]}, {hovered[1]}")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_elev", None):
                    try:
                        self.preview_tile_elev.set_text(f"Elevation: {getattr(hcell, 'elevation', 0)}")
                    except Exception:
                        pass
                if getattr(self, "preview_tile_note", None):
                    try:
                        note = "Building" if getattr(hcell, "is_building", False) else (
                            "Ramp" if getattr(hcell, "is_ramp", False) else "")
                        self.preview_tile_note.set_text(note)
                    except Exception:
                        pass
        except Exception:
            pass

        # Draw control hint
        if self.mouse_over_preview:
            try:
                font = pygame.font.SysFont('Arial', 14)
                hint = f"WASD: Move | Wheel: Zoom | Middle-drag: Rotate | Angle: {int(self.camera_rotation)}° | Zoom: {self.camera.zoom:.2f}x"
                text = font.render(hint, True, UITheme.TEXT)
                text_rect = text.get_rect(centerx=preview_w // 2, top=10)
                bg_rect = text_rect.inflate(20, 10)
                pygame.draw.rect(self.screen, UITheme.PRIMARY, bg_rect)
                pygame.draw.rect(self.screen, UITheme.SECONDARY, bg_rect, 1)
                self.screen.blit(text, text_rect)
            except Exception:
                pass