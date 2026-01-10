import os
import pygame
import pygame_gui
from main.config import resolution_converter, get_project_root, get_logger


class MainMenuManager:
    """
    Manages the main menu GUI system for the game.

    Handles menu creation, navigation, UI element styling, and user interactions
    for all out-of-game menus including main menu, loadout selection, and settings.

    Attributes:
        screen: pygame.Surface for rendering
        screen_width: Display width in pixels
        screen_height: Display height in pixels
        manager: pygame_gui.UIManager instance
        all_units: List of available unit objects for loadout selection
        player_loadout: List of units selected by the player
        current_menu: Currently active menu panel
        menu_stack: Stack of menu specifications for navigation history
        specs: Dictionary of menu specifications
        ui_elements: List of all active UI elements
        element_callbacks: Dict mapping UI elements to their callback functions
    """

    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height, theme=None):
        """
        Initialize the main menu manager.

        Args:
            screen: pygame.Surface to render UI on
            screen_width: Display width in pixels
            screen_height: Display height in pixels
            theme: Optional UI theme (currently unused, uses code-based theme)
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MainMenuManager")

        self.all_units = []
        self.selected_unit = None
        self.player_loadout = []
        self.selected_loadout_index = None  # Track which loadout unit is selected
        self.loadout_unit_buttons = []  # Track dynamically created buttons
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

        self.static_surface = pygame.Surface((screen_width, screen_height))
        self.logger.debug("Created static surface for background caching")

        self.ui_elements = []
        self.element_callbacks = {}

        self._load_units()
        self.logger.info("Creating main menu from spec...")
        self.create_main_menu()
        self.logger.info("Main menu created successfully")

        pygame.display.set_icon(self.logo)
        self.logger.info("Window icon set from logo")

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
        """
        Preload commonly used fonts to prevent runtime warnings.

        pygame_gui issues warnings when fonts are loaded dynamically during
        rendering. This method preloads all font variations we use.
        """
        try:
            font_dict = self.manager.get_theme().get_font_dictionary()

            # Preload fonts using the exact method signature pygame_gui expects
            # The font_dictionary.preload_font takes: point_size, font_name, bold, italic, antialiased
            font_configs = [
                # (size, name, bold, italic, antialiased)
                (14, 'noto_sans', True, False, True),  # bold
                (14, 'noto_sans', False, True, True),  # italic
                (14, 'noto_sans', True, True, True),  # bold italic
                (14, 'noto_sans', False, False, True),  # regular
                (12, 'noto_sans', True, False, True),  # small bold
                (12, 'noto_sans', False, True, True),  # small italic
                (12, 'noto_sans', False, False, True),  # small regular
                (16, 'noto_sans', True, False, True),  # large bold
                (16, 'noto_sans', False, True, True),  # large italic
            ]

            preloaded = 0
            for size, name, bold, italic, aa in font_configs:
                try:
                    # Call with positional arguments in correct order
                    font_dict.preload_font(size, name, bold, italic, aa)
                    preloaded += 1
                except Exception as e:
                    # Font might not exist or already loaded
                    self.logger.debug(f"Could not preload font {name} {size}pt: {e}")

            self.logger.debug(f"Preloaded {preloaded}/{len(font_configs)} UI font variations")
        except Exception as e:
            # Non-critical, just log and continue
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

        # generic element creation
        element = ui_class(**params)
        self.ui_elements.append(element)

        # style buttons, sliders, panels similarly as before
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

        elif element_type == "slider":
            _safe_apply_colours(element, {
                "normal_bg": UITheme.PRIMARY,
                "hovered_bg": UITheme.HOVER,
                "disabled_bg": UITheme.DISABLED,
                "normal_text": UITheme.TEXT,
                "hovered_text": UITheme.TEXT,
                "disabled_text": UITheme.SECONDARY,
                "normal_border": UITheme.SECONDARY
            })

        elif element_type == "panel":
            try:
                element.background_colour = UITheme.BACKGROUND
                element.border_colour = UITheme.SECONDARY
                element.rebuild()
            except Exception as e:
                self.logger.warning(f"Failed to style panel: {e}")

        elif element_type == "label":
            # UILabel doesn't parse HTML tags. Set plain text and use colours for color.
            if "text" in kwargs:
                try:
                    element.set_text(kwargs["text"])
                except Exception:
                    try:
                        element.set_text(str(kwargs["text"]))
                    except Exception as e:
                        self.logger.warning(f"Failed to set label text: {e}")

            # Apply text colour via colours dict if available
            _safe_apply_colours(element, {"normal_text": UITheme.TEXT})

        elif element_type == "textbox":
            # UITextBox supports limited HTML (bold/italic/linebreaks). Don't wrap in <font>.
            try:
                element.set_text(kwargs.get("html_text", ""))
            except Exception:
                try:
                    element.set_text(str(kwargs.get("html_text", "")))
                except Exception as e:
                    self.logger.warning(f"Failed to set textbox text: {e}")

        return element

    def create_from_spec(self, spec):
        """
        Recursively create UI hierarchy from specification dictionary.

        Args:
            spec: Dict containing 'type', 'rect', 'id', 'children', 'callback'

        Returns:
            Root UI element
        """
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
        """Pre-render static background elements (background image + logo) to cached surface."""
        self.logger.debug("Drawing static content")
        self.static_surface.blit(self.background, (0, 0))

        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(center=(self.screen_width // 2,
                                               self.screen_height // 2 - logo_offset))
        self.static_surface.blit(self.logo, logo_rect)

    def draw(self):
        """Render all UI elements: blit static surface then draw dynamic UI on top."""
        self.screen.blit(self.static_surface, (0, 0))
        self.manager.draw_ui(self.screen)

    def clear_element(self, element_clear):
        """Clear UI elements ('all' to clear everything)."""
        if element_clear == "all":
            self.logger.debug(f"Clearing {len(self.ui_elements)} UI elements")
            for element in list(self.ui_elements):
                element.kill()
            self.ui_elements.clear()

            # Clear loadout-specific tracking
            self.loadout_unit_buttons.clear()
            self.selected_loadout_index = None

            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
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

        root = self.load_menu(self.specs['menu_spec'])

        self.menu_panel = getattr(self, 'menu_panel', root)
        if hasattr(self, '_singleplayer_btn'):
            self._singleplayer_btn = getattr(self, '_singleplayer_btn')
        if hasattr(self, '_multiplayer_btn'):
            self._multiplayer_btn = getattr(self, '_multiplayer_btn')
        if hasattr(self, '_loadout_btn'):
            self._loadout_btn = getattr(self, '_loadout_btn')

    def singleplayer_press(self, element=None):
        """Navigate to singleplayer submenu."""
        self.logger.info("Singleplayer button pressed")
        self.load_menu(self.specs['singleplayer_main_spec'])

    def multiplayer_press(self, element=None):
        """Multiplayer button handler (not yet implemented)."""
        self.logger.info("Multiplayer pressed (not implemented)")

    def start_game_press(self, element=None):
        """Start game button handler (will trigger map generation)."""
        self.logger.info("Start game pressed")
        # TODO: Implement game start

    def loadout_press(self, element=None):
        """Navigate to loadout menu and refresh unit list."""
        self.logger.info("Loadout button pressed")
        self.load_menu(self.specs['loadout_spec'])

        dd = getattr(self, "unit_dropdown", None)
        if dd and self.all_units:
            try:
                # Manually trigger selection for first unit
                first_unit_name = self.all_units[0].name
                self.selected_unit = self.all_units[0]
                self.unit_selected(dd)
                self.logger.debug("Dropdown updated with units")
            except Exception as e:
                self.logger.warning(f"Failed to update dropdown: {e}")

        self.refresh_loadout_view()

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
        """
        Update the loadout display with clickable unit buttons.

        Creates individual buttons for each unit in the loadout that can be
        clicked to view details and select for removal.
        """
        # Clear existing loadout buttons
        for btn in self.loadout_unit_buttons:
            btn.kill()
        self.loadout_unit_buttons.clear()

        # Get the loadout list panel
        panel = getattr(self, "loadout_list_panel", None)
        if not panel:
            return

        # Clear background styling for the panel
        try:
            from .UItheme import UITheme
            panel.background_colour = UITheme.BACKGROUND
            panel.border_colour = UITheme.SECONDARY
            panel.rebuild()
        except Exception:
            pass

        if not self.player_loadout:
            # Show empty message as a textbox instead of label (better for HTML)
            import pygame
            from .UItheme import UITheme

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

        # Create button for each unit in loadout
        import pygame
        button_height = 40
        button_spacing = 5
        button_width = 260
        start_y = 10

        for i, unit in enumerate(self.player_loadout):
            y_pos = start_y + (i * (button_height + button_spacing))

            # Determine button text and styling
            cost_text = str(unit.cost) if unit.cost is not None else "N/A"
            button_text = f"{i + 1}. {unit.name} - {cost_text}"

            # Create button for this loadout entry
            unit_btn = self.init_ui(
                "button",
                pygame.Rect(10, y_pos, button_width, button_height),
                self.manager,
                container=panel,
                text=button_text
            )

            # Highlight if this unit is selected
            if i == self.selected_loadout_index:
                try:
                    from .UItheme import UITheme
                    unit_btn.colours["normal_bg"] = UITheme.HIGHLIGHT
                    unit_btn.rebuild()
                except Exception:
                    pass

            # Store button and register callback
            self.loadout_unit_buttons.append(unit_btn)

            # Create closure to capture index
            def make_callback(index):
                return lambda elem=None: self.select_loadout_unit(index)

            self.element_callbacks[unit_btn] = make_callback(i)

        # Add total cost label at bottom
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

        # Handle both string and tuple cases (pygame_gui returns tuples sometimes)
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
                # Handle tuple case
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

        # Clear loadout selection when adding new unit
        self.selected_loadout_index = None

        self.refresh_loadout_view()

    def select_loadout_unit(self, index):
        """
        Select a unit from the loadout list.

        Shows the unit's details and highlights it for removal.

        Args:
            index: Index of unit in player_loadout list
        """
        if index < 0 or index >= len(self.player_loadout):
            self.logger.warning(f"Invalid loadout index: {index}")
            return

        # Update selected index
        self.selected_loadout_index = index
        unit = self.player_loadout[index]

        self.logger.info(f"Selected loadout unit: {unit.name} (index {index})")

        # Update the description box with this unit's info
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

        # Refresh the loadout view to show highlighting
        self.refresh_loadout_view()

    def remove_selected_unit(self, element=None):
        """
        Remove the currently selected unit from loadout.

        If no unit is selected from the loadout, removes the last unit.
        """
        if self.selected_loadout_index is not None:
            # Remove the selected unit
            if 0 <= self.selected_loadout_index < len(self.player_loadout):
                removed = self.player_loadout.pop(self.selected_loadout_index)
                self.logger.info(f"Removed {removed.name} from loadout ({len(self.player_loadout)}/6)")
                self.selected_loadout_index = None
            else:
                self.logger.warning(f"Invalid selected index: {self.selected_loadout_index}")
                self.selected_loadout_index = None
        else:
            # Fall back to removing last unit
            if self.player_loadout:
                removed = self.player_loadout.pop()
                self.logger.info(f"Removed last unit {removed.name} ({len(self.player_loadout)}/6)")
            else:
                self.logger.warning("Loadout is empty, nothing to remove")

        # Clear the description box
        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            desc_box.set_text("<b>Select a unit</b><br><br>Choose from dropdown or loadout to view stats.")

        self.refresh_loadout_view()

    def process_events(self, events):
        """Process pygame and UI events, return 'quit' if user confirms exit."""
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

        for event in events:
            self.manager.process_events(event)

            if event.type == pygame.USEREVENT:
                user_type = getattr(event, "user_type", None)

                if user_type in (pygame_gui.UI_BUTTON_PRESSED, pygame_gui.UI_DROP_DOWN_MENU_CHANGED):
                    callback = self.element_callbacks.get(event.ui_element)
                    if callback:
                        try:
                            callback(event.ui_element)
                        except TypeError:
                            callback()
                        continue

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

        return None