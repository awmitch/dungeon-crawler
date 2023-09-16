import pygame
import random
random.seed(43)
from utilities import item_colors
from typing import Any
from screen_singleton import screen_singleton as screen

class Item(pygame.sprite.Sprite):
    def __init__(self, x, y, item_type):
        super().__init__()
        
        # Set a triangle appearance for the item
        self.image = pygame.Surface([25, 25], pygame.SRCALPHA)  # Added pygame.SRCALPHA for transparency
        pygame.draw.polygon(self.image, item_colors[item_type], [(12, 0), (0, 25), (25, 25)])
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.item_boosts = {
            "fire_rate":100,
            "bullet_speed":2,
            "speed":1,
            "damage":10,
            "max_health":15,
            "health_pack":30
        }
        self.item_type = item_type

    def pick_up(self, player):
        if self.item_type == 'health_pack':
            player.stats["health"] += self.item_boosts[self.item_type]  # heal the player
            if player.stats["health"] > player.stats["max_health"]:
                player.stats["health"] = player.stats["max_health"]
        else:
            player.power_ups[self.item_type] += 1
            player.stats[self.item_type] += self.item_boosts[self.item_type]
        self.kill()  # Remove the item after applying the effect

    def update(self, *args: Any, **kwargs: Any) -> None:
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)