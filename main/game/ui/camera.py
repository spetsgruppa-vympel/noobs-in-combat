import pygame
import main.config

pygame.init()
info = pygame.display.Info()

# Get actual screen size
screen_w = info.current_w
screen_h = info.current_h

# Set game window to ~50% of screen size
SCREEN_WIDTH = screen_w // 2
SCREEN_HEIGHT = screen_h // 2

# Tile size and map dimensions
TILE_SIZE = 40
MAP_WIDTH = (screen_w // TILE_SIZE)  # Full world size
MAP_HEIGHT = (screen_h // TILE_SIZE)

cameraOffset = [0, 0]  # x, y in pixels
cameraSpeed = 10

# Create window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def draw(self, surface, drawCameraOffset):
    draw_rect = self.rect.move(-drawCameraOffset[0], -drawCameraOffset[1])
    pygame.draw.rect(surface, self.color, draw_rect)
    if self.highlight == "selected":
        pygame.draw.rect(surface, (255, 255, 0), draw_rect, 3)


# Inside event loop or key handling:
keys = pygame.key.get_pressed()
if keys[pygame.K_LEFT]:
    cameraOffset[0] -= cameraSpeed
if keys[pygame.K_RIGHT]:
    cameraOffset[0] += cameraSpeed
if keys[pygame.K_UP]:
    cameraOffset[1] -= cameraSpeed
if keys[pygame.K_DOWN]:
    cameraOffset[1] += cameraSpeed