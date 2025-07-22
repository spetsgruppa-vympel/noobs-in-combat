# initialize everything and handle input
import asyncio
import pygame

async def main():  # starts up everything
    from .game.core.time.input import inputHandler

    pygame.init()
    await inputHandler.handle_input()
    pygame.quit()

asyncio.run(main())
if __name__ == "__main__":
    pygame.init()
    # noinspection PyUnresolvedReferences
    from main.game.ui import hud  # runs the hud

