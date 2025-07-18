# initialize everything and handle input

import pygame.key
from . import config
from .game.ui import camera
from .game.data.units.helperClasses import footMobility
pygame.key.start_text_input()
pygame.init()
print(footMobility.forestCost)