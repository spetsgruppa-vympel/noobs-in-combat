import os
import pygame
import pygame_gui
from main.config import resolution_converter, get_project_root, get_logger


class MainMenuManager:
    """Manages GUI while not in-game"""

    logo_vertical_factor = 0.25

    def __init__(self, screen, screen_width, screen_height, theme=None):
        self.logger = get_logger(__name__)
        self.logger.info("Initializing MainMenuManager")

        # store display surface and dimensions
        self.all_units = []
        self.selected_unit = None
        self.player_loadout = []
        self.current_menu = None
        self.menu_stack = []
        self.singleplayer_main_spec = None
        self.menu_spec = None
        self.specs = {}
        self.theme = theme
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.logger.debug(f"Screen dimensions: {screen_width}x{screen_height}")

        # create the pygame_gui UI manager for this screen size
        self.manager = pygame_gui.UIManager((screen_width, screen_height))
        self.logger.info("GUI Manager initialized with code-based theme")

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
        self.logger.debug("Created static surface for background caching")

        # list of UI elements for bulk operations (kill/hide)
        self.ui_elements = []

        # element instance -> callback mapping for fast event dispatch
        self.element_callbacks = {}

        # create the UI
        self.logger.info("Creating main menu from spec...")
        self.create_main_menu()
        self.logger.info("Main menu created successfully")

        pygame.display.set_icon(self.logo)
        self.logger.info("Window icon set from logo")

        # pre-render static content once
        self.draw_static()
        self.logger.debug("Static content pre-rendered")

    # ---------------------------
    # ASSETS (lazy-loaded)
    # ---------------------------
    @property
    def logo(self):
        """Load and cache the logo image when first accessed"""
        if self._logo is None:
            logo_path = os.path.join(get_project_root(), "assets", "ui", "game_icon.jpg")
            self.logger.debug(f"Loading logo from: {logo_path}")

            self._logo = pygame.image.load(logo_path).convert_alpha()
            self.logger.info("Logo loaded successfully")

            # scale to 20% of screen width while preserving aspect ratio
            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
            self.logger.debug(f"Logo scaled to {w}x{h}")
        return self._logo

    @property
    def background(self):
        """Load and cache the background image when first accessed"""
        if self._background is None:
            bg_path = os.path.join(get_project_root(), "assets", "ui", "menu_background.jpg")
            self.logger.debug(f"Loading background from: {bg_path}")

            bg = pygame.image.load(bg_path).convert()
            self.logger.info("Background loaded successfully")

            # scale to screen size then cheap-blur via downscale/upscale
            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
            self.logger.debug("Background scaled and blurred")
        return self._background

    # ---------------------------
    # BACKWARD-COMPATIBLE ACCESSORS
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
    # CENTRALIZED UI FACTORY
    # ---------------------------
    def init_ui(self, element_type, rect, manager, container=None, **kwargs):
        """
        Create and style UI elements centrally.

        Args:
            element_type: Type of element to create (button, slider, label, etc.)
            rect: pygame.Rect defining element position and size
            manager: pygame_gui.UIManager instance
            container: Parent container (optional)
            **kwargs: Additional arguments for element creation

        Returns:
            Created UI element
        """
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
                        self.logger.warning(f"Failed to rebuild element after color update: {e}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to update element colors: {e}")
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

        # handle confirmation dialogs
        if element_type == "confirmation_dialog":
            self.logger.debug("Creating confirmation dialog")
            element = ui_class(rect=rect, manager=manager, **kwargs)
            try:
                element.background_colour = UITheme.BACKGROUND
                element.border_colour = UITheme.SECONDARY
                element.rebuild()
            except Exception as e:
                self.logger.warning(f"Failed to style confirmation dialog: {e}")
            return element

        # build common params for standard widgets
        params = {"relative_rect": rect, "manager": manager, **kwargs}
        if container:
            params["container"] = container

        if element_type == "dropdown":
            options = params.pop("options_list", []) or []
            if not options:
                options = ["<no units available>"]
                self.logger.warning("Dropdown created with no options, using placeholder")

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

        # instantiate the element for non-dropdowns
        element = ui_class(**params)
        self.ui_elements.append(element)

        # apply theme-based styling per element type
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
            if "text" in kwargs:
                try:
                    element.set_text(f'<font color="{UITheme.TEXT}">{kwargs["text"]}</font>')
                except Exception:
                    try:
                        element.set_text(kwargs["text"])
                    except Exception as e:
                        self.logger.warning(f"Failed to set label text: {e}")
        elif element_type == "textbox":
            try:
                element.set_text(f'<font color="{UITheme.TEXT}">{kwargs.get("html_text", "")}</font>')
            except Exception:
                try:
                    element.set_text(kwargs.get("html_text", ""))
                except Exception as e:
                    self.logger.warning(f"Failed to set textbox text: {e}")

        return element

    # ---------------------------
    # create_from_spec
    # ---------------------------
    def create_from_spec(self, spec):
        """
        Create UI tree from specification dictionary.

        Args:
            spec: Dictionary describing UI structure with root and children

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

        # iterate children and create each
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
                if isinstance(cb, str):
                    callback_fn = getattr(self, cb, None)
                else:
                    callback_fn = cb
                if callback_fn:
                    self.element_callbacks[element] = callback_fn
                    self.logger.debug(f"Registered callback for {child.get('id', 'element')}")

        self.logger.info(f"Created UI tree with {len(spec.get('children', []))} children")
        return root_element

    def load_menu(self, spec):
        """Load a new menu from specification"""
        self.logger.info(f"Loading menu: {spec.get('id', 'unnamed')}")

        if self.current_menu:
            self.clear_element("all")

        root = self.create_from_spec(spec)
        self.menu_stack.append(spec)
        self.current_menu = root

        self.logger.debug(f"Menu stack depth: {len(self.menu_stack)}")
        return root

    # ---------------------------
    # DRAWING
    # ---------------------------
    def draw_static(self):
        """Pre-render static background elements"""
        self.logger.debug("Drawing static content")
        self.static_surface.blit(self.background, (0, 0))

        logo_offset = int(self.screen_height * self.logo_vertical_factor)
        logo_rect = self.logo.get_rect(center=(self.screen_width // 2,
                                               self.screen_height // 2 - logo_offset))
        self.static_surface.blit(self.logo, logo_rect)

    def draw(self):
        """Draw all UI elements"""
        self.screen.blit(self.static_surface, (0, 0))
        self.manager.draw_ui(self.screen)

    # ---------------------------
    # CLEAR / KILL UI
    # ---------------------------
    def clear_element(self, element_clear):
        """Clear UI elements"""
        if element_clear == "all":
            self.logger.debug(f"Clearing all UI elements ({len(self.ui_elements)} total)")
            for element in list(self.ui_elements):
                element.kill()
            self.ui_elements.clear()

            self._singleplayer_btn = None
            self._multiplayer_btn = None
            self._loadout_btn = None
            self.menu_panel = None
            self._quit_dialog = None
            self.element_callbacks.clear()

            self.logger.info("All UI elements cleared")

    # ---------------------------
    # MAIN MENU CREATION
    # ---------------------------
    def create_main_menu(self):
        """Create main menu UI structure"""
        self.logger.debug("Building main menu specifications")

        panel_width = resolution_converter(450, 'x')
        panel_height = resolution_converter(350, 'y')
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = int(self.screen_height - panel_height) // 1.5

        button_width = resolution_converter(300, 'x')
        button_height = resolution_converter(60, 'y')
        button_x = (panel_width - button_width) // 2
        start_y = resolution_converter(40, 'y')

        back_button_x = resolution_converter(10, 'x')
        back_button_y = resolution_converter(40, 'y')
        back_button_width = resolution_converter(60, 'x')
        back_button_height = resolution_converter(60, 'y')

        def back_button_spec():
            return {
                "type": "button",
                "id": "_back_btn",
                "rect": [back_button_x, back_button_y, back_button_width, back_button_height],
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
                    "text": "Start",
                    "callback": ""
                },
                back_button_spec()
            ]
        }

        self.specs['loadout_spec'] = {
            "type": "panel",
            "id": "loadout_panel",
            "rect": [panel_x, panel_y, int(panel_width * 2), int(panel_height * 1.5)],
            "children": [
                {
                    "type": "label",
                    "id": "loadout_title",
                    "rect": [20, 20, 200, 40],
                    "text": "Loadout Selection"
                },
                {
                    "type": "dropdown",
                    "id": "unit_dropdown",
                    "rect": [20, 80, 200, 40],
                    "options": [u.name for u in self.all_units],
                    "callback": "unit_selected"
                },
                {
                    "type": "textbox",
                    "id": "unit_description",
                    "rect": [250, 80, 300, 200],
                    "html_text": "Select a unit to see details..."
                },
                {
                    "type": "button",
                    "id": "confirm_unit_btn",
                    "rect": [250, 300, 200, 50],
                    "text": "Add to Loadout",
                    "callback": "add_to_loadout"
                },
                {
                    "type": "textbox",
                    "id": "current_loadout",
                    "rect": [20, 140, 220, 220],
                    "html_text": "Current loadout:\n<em>empty</em>"
                },
                {
                    "type": "button",
                    "id": "_back_btn",
                    "rect": [10, 10, back_button_width, back_button_height],
                    "text": "Back",
                    "callback": "back_press"
                }
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

    # ---------------------------
    # CALLBACKS
    # ---------------------------
    def singleplayer_press(self, element=None):
        self.logger.info("Singleplayer button pressed")
        self.load_menu(self.specs['singleplayer_main_spec'])

    def multiplayer_press(self, element=None):
        self.logger.info("Multiplayer button pressed")

    def loadout_press(self, element=None):
        self.logger.info("Loadout button pressed")
        self.load_menu(self.specs['loadout_spec'])

        dd = getattr(self, "unit_dropdown", None)
        if dd and hasattr(dd, "set_options_list"):
            try:
                dd.set_options_list([u.name for u in self.all_units])
                self.logger.debug("Updated dropdown options")
            except Exception as e:
                self.logger.warning(f"Failed to update dropdown: {e}")

        self.refresh_loadout_view()

    def back_press(self, element=None):
        self.logger.info("Back button pressed")
        if len(self.menu_stack) > 1:
            self.clear_element("all")
            self.menu_stack.pop()
            spec = self.menu_stack[-1]
            self.current_menu = self.create_from_spec(spec)
            self.logger.debug(f"Navigated back to: {spec.get('id', 'previous menu')}")
        else:
            self.logger.warning("No previous menu to navigate to")

    # ---------------------------
    # Loadout callbacks & helpers
    # ---------------------------
    def refresh_loadout_view(self):
        """Update the current_loadout textbox"""
        box = getattr(self, "current_loadout", None)
        if not box:
            return
        if not self.player_loadout:
            box.set_text("Current loadout:\n<em>empty</em>")
            return

        lines = ["<b>Current loadout:</b>"]
        total_cost = 0
        for i, u in enumerate(self.player_loadout, start=1):
            cost_text = u.cost if u.cost is not None else "N/A"
            lines.append(f"{i}. {u.name} — cost: {cost_text}")
            if isinstance(u.cost, (int, float)):
                total_cost += u.cost
        lines.append(f"<br><b>Total cost:</b> {total_cost}")
        box.set_text("<br>".join(lines))

        self.logger.debug(f"Loadout updated: {len(self.player_loadout)} units, total cost: {total_cost}")

    def unit_selected(self, element=None):
        """Handle unit dropdown selection"""
        dropdown = element or getattr(self, "unit_dropdown", None)
        if dropdown is None:
            return

        selection = getattr(dropdown, "selected_option", None)
        if not selection:
            return

        unit = next((u for u in self.all_units if u.name == selection), None)
        if not unit:
            self.logger.warning(f"Selected unit '{selection}' not found in unit list")
            return

        self.selected_unit = unit
        self.logger.info(f"Unit selected: {unit.name}")

        desc_box = getattr(self, "unit_description", None)
        if desc_box:
            desc_lines = [
                f"<b>{unit.name}</b>",
                f"Cost: {unit.cost}",
                f"HP: {unit.hp}",
                f"Armor: {unit.armor}",
                f"Sight: {unit.sight}",
                f"Mobility: {unit.mobility}",
                "",
                f"{unit.description or ''}"
            ]
            desc_box.set_text("<br>".join(str(x) for x in desc_lines))

    def add_to_loadout(self, element=None):
        """Add selected unit to player loadout"""
        unit = self.selected_unit
        if unit is None:
            dd = getattr(self, "unit_dropdown", None)
            if dd:
                sel = getattr(dd, "selected_option", None)
                unit = next((u for u in self.all_units if u.name == sel), None)

        if unit is None:
            self.logger.warning("Attempted to add unit to loadout but none selected")
            return

        MAX_LOADOUT = 6
        if len(self.player_loadout) >= MAX_LOADOUT:
            self.logger.warning(f"Loadout is full (max {MAX_LOADOUT})")
            return

        if any(u.name == unit.name for u in self.player_loadout):
            self.logger.warning(f"{unit.name} is already in the loadout")
            return

        self.player_loadout.append(unit)
        self.logger.info(f"Added {unit.name} to loadout (total: {len(self.player_loadout)})")
        self.refresh_loadout_view()

    # ---------------------------
    # EVENT LOOP
    # ---------------------------
    def process_events(self, events):
        """Process pygame and UI events"""
        dialog_w, dialog_h = resolution_converter(330, "x"), resolution_converter(200, "y")
        dialog_x = (self.screen_width - dialog_w) // 2
        dialog_y = (self.screen_height - dialog_h) // 2

        def init_quit_dialog():
            if self._quit_dialog is None:
                self.logger.info("Creating quit confirmation dialog")
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
                        self.logger.info("Quit confirmed via window close")
                        return "quit"

                if event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                    continue
                init_quit_dialog()

            if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == self._quit_dialog:
                    self.logger.info("Quit confirmed via dialog")
                    self._quit_dialog = None
                    return "quit"

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == self._quit_dialog:
                    self.logger.info("Quit dialog cancelled")
                    self._quit_dialog = None

        return None