import asyncio
import ctypes
import os
import platform
import pygame
import pygame_gui
from main.config import color_print


class Game:
    def __init__(self):
        pygame.init()  # initialize pygame modules
        color_print("Initializing HUD...", "IMPORTANT")

        # get current screen resolution
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h

        # create a resizable window with full screen resolution
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.RESIZABLE
        )

        # set window title
        pygame.display.set_caption("Noobs in Combat")

        # setup GUI manager with theme
        script_dir = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(script_dir, "theme.json")
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height), theme_path)

        # maximize window only on windows
        # skill issue on you if you don't understand os api :tongue_out:
        if platform.system() == "Windows":
            wm_info = pygame.display.get_wm_info()
            hwnd = wm_info.get("window")
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
                color_print("Window maximized successfully.", "IMPORTANT")
            else:
                color_print("Could not obtain window handle to maximize.", "ERROR")
        else:
            color_print("OS NOT WINDOWS", "ERROR")
            raise OSError("ONLY WORKS ON WINDOWS")

        self.clock = pygame.time.Clock()  # clock for FPS control
        self.running = True  # main loop flag

    async def main_loop(self):
        # main game loop
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # limit FPS to 60 and get delta time

            # get all events once
            events = pygame.event.get()

            # handle events
            for event in events:
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    color_print("Exit requested by user.", "IMPORTANT")
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    print(f"Key pressed: {pygame.key.name(event.key)}")
                    # future input logic
                    # inputHandler.process(event)

            # fill screen with dark background
            self.screen.fill((30, 30, 30))

            # update display (draw everything)
            pygame.display.flip()

            # yield control to asyncio event loop
            await asyncio.sleep(0)

    @staticmethod
    def init_ui(element_type, rect, manager, **kwargs):
        # generic UI element initializer
        ELEMENT_MAP = {
            "button": pygame_gui.elements.UIButton,
            "slider": pygame_gui.elements.UIHorizontalSlider,
            "label": pygame_gui.elements.UITextBox,
            "dropdown": pygame_gui.elements.UIDropDownMenu
        }
        ui_class = ELEMENT_MAP[element_type]
        params = {
            "relative_rect": rect,
            "manager": manager,
            **kwargs
        }
        return ui_class(**params)


if __name__ == "__main__":
    async def main():
        # import HUD logic, might do additional setup

        # create game instance
        game = Game()

        # run main loop
        await game.main_loop()

        # cleanup pygame when done
        pygame.quit()

    # start asyncio event loop
    asyncio.run(main())
