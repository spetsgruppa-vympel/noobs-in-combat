import pygame
import asyncio

async def handle_input():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return  # Exit loop
            elif event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")

        await asyncio.sleep(0.01)  # Yield control to other tasks