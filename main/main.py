"""
Main entry point for Snakes in Combat.

Initializes pygame, creates the game window, and runs the main game loop.
Handles high-level game state management and coordinates between menu system
and gameplay systems.
"""

import asyncio
import pygame
from main.config import get_logger, get_project_root, setup_logging


class Game:
    """
    Main game controller class.

    Manages the game loop, display initialization, and high-level state transitions
    between menus and gameplay. Coordinates all major game systems.

    Attributes:
        screen: pygame.Surface for rendering
        screen_width: Display width in pixels
        screen_height: Display height in pixels
        clock: pygame.Clock for frame rate control
        running: Boolean flag for main loop control
        menu_manager: MainMenuManager instance for menu UI
        game_state: Current game state ('menu', 'playing', 'paused')
    """

    def __init__(self):
        """
        Initialize the game instance.

        Sets up pygame, creates the display window, and initializes all
        major game systems including menu manager and logging.
        """
        from main.game.ui.menuManager import MainMenuManager

        self.logger = get_logger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("Game initialization started")

        # Get current display resolution for fullscreen mode
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h

        self.logger.info(f"Display resolution: {self.screen_width}x{self.screen_height}")

        # Create fullscreen display surface
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Snakes in Combat")
        self.logger.info("Display initialized in fullscreen mode")

        # Initialize frame rate control
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_state = "menu"  # Possible states: 'menu', 'playing', 'paused'

        # Initialize menu system
        self.logger.info("Initializing menu manager...")
        self.menu_manager = MainMenuManager(
            self.screen, self.screen_width, self.screen_height
        )
        self.logger.info("Game initialization completed successfully")

    async def main_loop(self):
        """
        Main game loop using asyncio for smooth frame pacing.

        Handles event processing, state updates, and rendering at 60 FPS.
        Logs periodic statistics for debugging and performance monitoring.
        """
        self.logger.info("Entering main game loop")
        frame_count = 0
        target_fps = 60

        while self.running:
            # Calculate delta time for frame-independent movement
            dt = self.clock.tick(target_fps) / 1000.0
            frame_count += 1

            # Log performance metrics every 5 seconds
            if frame_count % (target_fps * 5) == 0:
                actual_fps = self.clock.get_fps()
                self.logger.debug(
                    f"Frame: {frame_count}, FPS: {actual_fps:.1f}, "
                    f"Delta: {dt:.4f}s, State: {self.game_state}"
                )

            # Collect all pygame events
            events = pygame.event.get()

            # Log significant user events
            for event in events:
                if event.type == pygame.QUIT:
                    self.logger.info("Quit event received from window")
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    self.logger.debug(f"Key pressed: {key_name}")

            # Process events based on current game state
            if self.game_state == "menu":
                result = self.menu_manager.process_events(events)
                if result == "quit":
                    self.logger.info("Quit action confirmed by user")
                    self.running = False
                elif result == "start_game":
                    self.logger.info("Transitioning from menu to gameplay")
                    self.game_state = "playing"
                    # TODO: Initialize gameplay systems

                # Update menu UI elements
                self.menu_manager.manager.update(dt)

                # Render menu
                self.menu_manager.draw()

            elif self.game_state == "playing":
                # TODO: Handle gameplay event processing
                # TODO: Update game systems (units, AI, etc.)
                # TODO: Render game world
                pass

            elif self.game_state == "paused":
                # TODO: Handle pause menu
                pass

            # Flip display buffers
            pygame.display.flip()

            # Yield control to allow other async operations
            await asyncio.sleep(0)

        self.logger.info(f"Main loop exited after {frame_count} frames")

    def cleanup(self):
        """
        Clean up resources before shutdown.

        Ensures all systems are properly shut down and resources are released.
        """
        self.logger.info("Performing cleanup operations")

        try:
            pygame.quit()
            self.logger.info("Pygame shut down successfully")
        except Exception as e:
            self.logger.error(f"Error during pygame shutdown: {e}", exc_info=True)


async def main():
    """
    Async main entry point.

    Initializes logging, creates game instance, runs main loop, and handles cleanup.
    Catches and logs any critical errors that occur during execution.
    """
    # Initialize logging system first
    project_root = get_project_root()
    logger = setup_logging(project_root)

    logger.info("=" * 60)
    logger.info("Application started")
    logger.info(f"Project root: {project_root}")

    game = None

    try:
        logger.info("Creating Game instance...")
        game = Game()

        logger.info("Starting main loop...")
        await game.main_loop()

    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        raise

    finally:
        if game:
            game.cleanup()

        logger.info("Application terminated")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())