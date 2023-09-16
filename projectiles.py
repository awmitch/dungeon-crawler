from settings import GREEN, RED
import pygame
import math
from typing import Any
from screen_singleton import screen_singleton as screen

class Projectile(pygame.sprite.Sprite):
    def __init__(self, start_x, start_y, dest_x, dest_y, owner):
        super().__init__()
        self.image = pygame.Surface([10, 10])
        self.owner = owner  # This refers to the entity (enemy or player) that shot the bullet

        if self.owner.is_player:
            self.image.fill(GREEN)
        else:
            self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = start_x
        self.rect.y = start_y
        # Calculate the angle to the target (in radians)
        angle = math.atan2(dest_y - start_y, dest_x - start_x)
        # Calculate velocity components based on angle and speed
        self.change_x = math.cos(angle) * self.owner.stats["bullet_speed"]
        self.change_y = math.sin(angle) * self.owner.stats["bullet_speed"]

    def update(self, *args: Any, **kwargs: Any) -> None:
        
        # Check for collisions between the projectile and the walls
        wall_hits = pygame.sprite.spritecollide(self, self.owner.current_room.walls, False)
        # Check for collisions between the projectile and the doors
        door_hits = pygame.sprite.spritecollide(self, self.owner.current_room.doors, False)

        if (len(wall_hits) == 0) and (len(door_hits) == 0):
            self.rect.x += self.change_x * self.owner.current_room.get_time_modifier()
            self.rect.y += self.change_y * self.owner.current_room.get_time_modifier()
        else:
            self.kill()
            
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)
        
        
