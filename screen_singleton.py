import pygame
from settings import WIDTH, HEIGHT

class ScreenSingleton:
    _instance = None
    screen = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Rogue-like Game')
        return cls._instance

screen_singleton = ScreenSingleton().screen
