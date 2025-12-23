import logging


class Game:
    def __init__(self):
        # ---------------------------
        # LAZY IMPORTS
        # ---------------------------
        import pygame
        from main.game.ui.menuManager import MainMenuManager
        from main.config import get_logger

        self.logger = get_logger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("Game initialization started")

        # ---------------------------
        # SCREEN INIT
        # ---------------------------

        # dynamically get current display resolution
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h

        self.logger.info(f"Display resolution: {self.screen_width}x{self.screen_height}")

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Noobs in Combat")
        self.logger.info("Display initialized in fullscreen mode")

        self.clock = pygame.time.Clock()
        self.running = True

        # ---------------------------
        # MENU MANAGER
        # ---------------------------

        self.logger.info("Initializing menu manager...")
        self.menu_manager = MainMenuManager(self.screen, self.screen_width, self.screen_height)
        self.logger.info("Game initialization completed successfully")

    async def main_loop(self):
        import asyncio
        import pygame

        self.logger.info("Entering main game loop")
        frame_count = 0

        while self.running:
            dt = self.clock.tick(60) / 1000.0
            frame_count += 1

            # Log every 300 frames (5 seconds at 60 FPS)
            if frame_count % 300 == 0:
                self.logger.debug(f"Frame: {frame_count}, Delta time: {dt:.4f}s")

            events = pygame.event.get()

            # Log significant events
            for event in events:
                if event.type == pygame.QUIT:
                    self.logger.info("Quit event received")
                elif event.type == pygame.KEYDOWN:
                    self.logger.debug(f"Key pressed: {pygame.key.name(event.key)}")

            # process menu events (includes ESC + window close handling)
            result = self.menu_manager.process_events(events)
            if result == "quit":
                self.logger.info("Quit action confirmed by user")
                self.running = False

            # update
            self.menu_manager.manager.update(dt)

            # draw
            self.menu_manager.draw()
            pygame.display.flip()
            await asyncio.sleep(0)

        self.logger.info(f"Main loop exited after {frame_count} frames")


# ------------------------ # ENTRY POINT # ------------------------

if __name__ == "__main__":
    import asyncio
    from main.config import get_project_root, setup_logging


    async def main():
        import pygame

        # Initialize logging first
        project_root = get_project_root()
        logger = setup_logging(project_root)

        logger.info("=" * 60)
        logger.info("Application started")
        logger.info(f"Project root: {project_root}")

        try:
            logger.info("Creating Game instance...")
            game = Game()

            logger.info("Starting main loop...")
            await game.main_loop()

            logger.info("Shutting down pygame...")
            pygame.quit()
            logger.info("Pygame shut down successfully")

        except Exception as e:
            logger.critical(f"Critical error in main: {e}", exc_info=True)
            raise
        finally:
            logger.info("Application terminated")
            logger.info("=" * 60)


    asyncio.run(main())