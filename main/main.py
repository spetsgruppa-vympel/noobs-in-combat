
class Game:
    def __init__(self):

        # ---------------------------
        # LAZY IMPORTS
        # ---------------------------

        import pygame
        from main.game.ui.menuManager import MenuManager  # lazy import here

        # ---------------------------
        # SCREEN INIT
        # ---------------------------

        # dynamically get current display resolution
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h


        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Noobs in Combat")
        self.clock = pygame.time.Clock()
        self.running = True

        # ---------------------------
        # MENU MANAGER
        # ---------------------------

        self.menu_manager = MenuManager(self.screen, self.screen_width, self.screen_height)

    async def main_loop(self):
        import asyncio
        import pygame

        while self.running:
            dt = self.clock.tick(60) / 1000.0
            events = pygame.event.get()

            # process menu events (includes ESC + window close handling)
            result = self.menu_manager.process_events(events)
            if result == "quit":
                self.running = False

            # update
            self.menu_manager.manager.update(dt)

            # draw
            self.menu_manager.draw()
            pygame.display.flip()
            await asyncio.sleep(0)

# ------------------------ # ENTRY POINT # ------------------------

if __name__ == "__main__":
    import asyncio
    from main.config import color_print
    async def main():
        import pygame
        color_print("Launching game...", "IMPORTANT")
        game = Game()
        await game.main_loop()
        pygame.quit()
        color_print("Pygame quit successfully.", "IMPORTANT")

    asyncio.run(main())