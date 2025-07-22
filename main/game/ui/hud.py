# main display file
import pygame
import sys

# get current screen resolution
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h

# create a fullscreen window
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Noobs in Combat")

# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            running = False


pygame.quit()
sys.exit()