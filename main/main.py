import asyncio
import pygame
import pygame_gui
from .config import color_print
from main.game.ui.mainMenu import singleplayer_press, multiplayer_press  # your updated location

# ------------------------
# Main Game Class
# ------------------------
class Game:
    def __init__(self):
        self.menu_panel = None
        self.multiplayer_btn = None
        self.singleplayer_btn = None
        self.quit_dialog = None
        pygame.init()
        color_print("Initializing HUD...", "IMPORTANT")

        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Noobs in Combat")
        color_print(f"Screen initialized: {self.screen_width}x{self.screen_height}", "IMPORTANT")

        # load theme
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(script_dir, "theme.json")
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height), theme_path)
        color_print("GUI Manager initialized with theme.json", "IMPORTANT")

        self.clock = pygame.time.Clock()
        self.running = True

        # placeholders for lazy loaded assets
        self._logo = None
        self._background = None

        # create UI elements
        self.create_main_menu()
        color_print("Main menu created.", "IMPORTANT")

    logo_vertical_factor = 0.25  # fraction of screen height above center

    @property
    def logo(self):
        if self._logo is None:
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logo_path = os.path.join(root_dir, "assets", "ui", "game_icon.jpg")
            self._logo = pygame.image.load(logo_path).convert_alpha()
            color_print("Logo loaded from assets/ui/game_icon.jpg", "IMPORTANT")
            # scale logo to 40% screen width
            w = int(self.screen_width * 0.2)
            h = int(self._logo.get_height() * (w / self._logo.get_width()))
            self._logo = pygame.transform.smoothscale(self._logo, (w, h))
            color_print(f"Logo scaled to {w}x{h}", "IMPORTANT")
        return self._logo

    @property
    def background(self):
        if self._background is None:
            import os
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bg_path = os.path.join(root_dir, "assets", "ui", "menu_background.jpg")
            bg = pygame.image.load(bg_path).convert()
            color_print("Background loaded from assets/ui/menu_background.jpg", "IMPORTANT")
            bg = pygame.transform.smoothscale(bg, (self.screen_width, self.screen_height))
            small = pygame.transform.smoothscale(bg, (self.screen_width // 10, self.screen_height // 10))
            self._background = pygame.transform.smoothscale(small, (self.screen_width, self.screen_height))
            color_print("Background scaled and blurred", "IMPORTANT")
        return self._background

    def create_main_menu(self):
        # --- create menu panel ---
        panel_width = 400
        panel_height = 250
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2 + 50

        self.menu_panel = self.init_ui(
            "panel",
            pygame.Rect(panel_x, panel_y, panel_width, panel_height),
            self.manager,
            object_id="#menu_panel"
        )

        # set panel background color
        self.menu_panel.background_colour = pygame.Color(50, 50, 50, 200)

        # --- Button dimensions ---
        button_width = 300
        button_height = 60
        button_x = (panel_width - button_width) // 2
        start_y = 40  # margin from top of panel

        # singleplayer button
        self.singleplayer_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y, button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Singleplayer",
            object_id="#singleplayer_btn"
        )
        color_print("Singleplayer button created.", "IMPORTANT")

        # multiplayer button
        self.multiplayer_btn = self.init_ui(
            "button",
            pygame.Rect(button_x, start_y + button_height + 20, button_width, button_height),
            self.manager,
            container=self.menu_panel,
            text="Multiplayer",
            object_id="#multiplayer_btn"
        )
        color_print("Multiplayer button created.", "IMPORTANT")

    #   ---ui initialization function---
    @staticmethod
    def init_ui(element_type, rect, manager, container=None, **kwargs):
        element_map = {
            "button": pygame_gui.elements.UIButton,
            "slider": pygame_gui.elements.UIHorizontalSlider,
            "label": pygame_gui.elements.UITextBox,
            "dropdown": pygame_gui.elements.UIDropDownMenu,
            "panel": pygame_gui.elements.UIPanel
        }
        ui_class = element_map[element_type]
        params = {
            "relative_rect": rect,
            "manager": manager,
            **kwargs
        }
        if container is not None:
            params["container"] = container  # attach to panel/container
        color_print(f"UI element initialized: {element_type}", "IMPORTANT")
        return ui_class(**params)

    # ---main loop---
    async def main_loop(self):
        color_print("starting main loop...", "IMPORTANT")
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            events = pygame.event.get()

            # guard clause for quit
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                    color_print("exit requested by user.", "IMPORTANT")
                    continue

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    color_print("esc pressed, asking for confirmation...", "WARNING")
                    dialog_w, dialog_h = 330, 200
                    # sets esc dialog box dimensions
                    dialog_x = (self.screen_width - dialog_w) // 2  # sets dialog box positions
                    dialog_y = (self.screen_height - dialog_h) // 2

                    self.quit_dialog = pygame_gui.windows.UIConfirmationDialog(
                        rect=pygame.Rect(dialog_x, dialog_y, dialog_w, dialog_h),
                        manager=self.manager,
                        window_title="Confirm Quit",
                        action_long_desc="Are you sure you want to quit?",
                        action_short_name="Quit",
                        blocking=True
                    )
                    continue

                if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                    if event.ui_element == getattr(self, "quit_dialog", None):
                        color_print("quit confirmed by user.", "IMPORTANT")
                        self.running = False
                        continue

                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.singleplayer_btn:
                        color_print("singleplayer button pressed.", "IMPORTANT")
                        singleplayer_press()
                    elif event.ui_element == self.multiplayer_btn:
                        color_print("multiplayer button pressed.", "IMPORTANT")
                        multiplayer_press()
                    continue

                self.manager.process_events(event)

            # update gui manager
            self.manager.update(dt)

            # draw background
            self.screen.blit(self.background, (0, 0))

            # responsive logo placement (25% of screen height above center)
            logo_offset = int(self.screen_height * 0.25)
            logo_rect = self.logo.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 - logo_offset)
            )
            self.screen.blit(self.logo, logo_rect)

            # keep menu panel centered responsively
            panel_width, panel_height = self.menu_panel.get_relative_rect().size
            self.menu_panel.set_relative_position((
                (self.screen_width - panel_width) // 2,
                (self.screen_height - panel_height) // 2 + 50
            ))

            # draw gui
            self.manager.draw_ui(self.screen)
            pygame.display.flip()
            await asyncio.sleep(0)


# ------------------------
# entry point
# ------------------------
if __name__ == "__main__":
    async def main():
        color_print("Launching game...", "IMPORTANT")
        game = Game()
        await game.main_loop()
        pygame.quit()
        color_print("Pygame quit successfully.", "IMPORTANT")

    asyncio.run(main())
