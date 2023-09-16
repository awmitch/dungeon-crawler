import pygame
from settings import GREEN
from utilities import set_safe_position
from screen_singleton import screen_singleton as screen
from typing import Any

class AbilityItem(pygame.sprite.Sprite):
    def __init__(self, player_x, player_y, ability, init=False):
        super().__init__()

        # Load the ability image
        ability_image_path = ability_paths[ability]
        # Load the ability image
        original_image = pygame.image.load(ability_image_path).convert_alpha()
        
        # Scale the image (for example, to half its original size)
        scale_factor = 0.1  # 10%
        width, height = original_image.get_size()
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        
        self.image = pygame.transform.scale(original_image, (scaled_width, scaled_height))
        
        self.rect = self.image.get_rect()
        if init:
            x, y = set_safe_position(player_x, player_y, scaled_width, scaled_height)
        else:
            x, y = player_x, player_y
        self.rect.x = x
        self.rect.y = y
        self.ability_name = ability

    def pick_up(self, player):
        # If player has an active ability, drop it
        if player.active_ability:
            ability = player.abilities[player.active_ability]
            if ability.is_active:
                ability.is_active = False
                ability.on_deactivate(player)
            ability.end_time = ability.current_time - ability.cooldown_time
            ability.start_time = ability.current_time - ability.duration_time
            player_x, player_y = player.get_player_pos(center=True)
            dropped_ability = AbilityItem(player_x, player_y, player.active_ability)
            player.current_room.all_sprites.add(dropped_ability)
            player.current_room.items.add(dropped_ability)

        # Set the new ability
        player.set_ability(self.ability_name, self.image)

        # Remove this sprite
        self.kill()

    def update(self, *args: Any, **kwargs: Any) -> None:
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)

class Ability:
    def __init__(self, cooldown: int, duration: int):
        self.cooldown_time = cooldown
        self.duration_time = duration
        self.remaining_duration = duration  # Starts with the full duration
        self.end_time = -self.cooldown_time
        self.start_time = 0
        self.current_cooldown = 0
        self.is_active = False
        self.current_time = 0

    def activate(self, player):
        self.current_time = pygame.time.get_ticks()
        if ((self.current_time - self.end_time) < self.cooldown_time) and self.is_overused:
            return
        else:
            self.is_overused = False

        if self.remaining_duration > 0:
            self.is_active = True
            self.start_time = self.current_time - (self.duration_time - self.remaining_duration)
            self.end_time = self.start_time + self.duration_time
            self.on_activate(player)

    def deactivate_early(self, player):
        self.current_time = pygame.time.get_ticks()
        # Calculate elapsed time since activation
        elapsed_time = self.current_time - self.start_time

        # Adjust cooldown based on the proportion of time the ability was active
        proportional_cooldown = (1 - elapsed_time / self.duration_time) * self.cooldown_time

        self.remaining_duration = self.duration_time - elapsed_time

        # Set the end_time based on the new proportional cooldown
        self.end_time = self.current_time - proportional_cooldown
        self.is_active = False
        self.on_deactivate(player)

    def update(self, player):
        self.current_time = pygame.time.get_ticks()
        if self.is_active:
            if (self.current_time - self.start_time) >= self.duration_time:
                self.is_active = False
                self.is_overused = True
                self.on_deactivate(player)
        else:
            # Adjust duration based on the proportion of time the ability has cooled down
            proportional_duration = ((self.current_time - self.end_time) / self.cooldown_time) * self.duration_time
            if self.remaining_duration < self.duration_time:
                self.remaining_duration = proportional_duration

    def on_activate(self, player):
        pass

    def on_deactivate(self, player):
        pass


class Sprint(Ability):
    def on_activate(self, player):
        player.stats["speed"] *= 1.5

    def on_deactivate(self, player):
        player.stats["speed"] /= 1.5

class Dodge(Ability):
    def __init__(self, *args, **kwargs):
        super(Dodge, self).__init__(*args, **kwargs)
        self.angle = 0  # initial rotation angle
        self.dodge_direction = (0, 0)  # Initial dodge direction as a tuple (x, y)
        self.speed_modifier = 1.5

    def on_activate(self, player):
        player.is_dodging = True

        player.original_image = player.image.copy()  
        self.angle = 0

        # Set the dodge direction based on player's current movement direction
        self.dodge_direction = (
            self.speed_modifier*player.target_velocity_x, 
            self.speed_modifier*player.target_velocity_y
        )
    
    def get_direction(self):
        return self.dodge_direction
    
    def update(self, player):
        super().update(player)
        if self.is_active:
            # Determine the direction of rotation based on dodge direction
            self.angle = 360*((self.current_time - self.start_time)/self.duration_time)
            if self.angle >= 360:  
                self.angle = 0
            if self.dodge_direction[0] <= 0:  # If moving horizontally left
                player.image = pygame.transform.rotate(player.original_image, self.angle)
            else:
                player.image = pygame.transform.rotate(player.original_image, -self.angle)
            
    def on_deactivate(self, player):
        player.is_dodging = False
        player.image = player.original_image
        self.dodge_direction = (0, 0)

class Deflect(Ability):
    def on_activate(self, player):
        player.is_deflecting = True

    def on_deactivate(self, player):
        player.is_deflecting = False

class Fortify(Ability):
    def on_activate(self, player):
        player.armor += 50

    def on_deactivate(self, player):
        player.armor -= 50
        if player.armor < 0:
            player.armor = 0

class Invisibility(Ability):
    def on_activate(self, player):
        # Save the original image in case it's needed for deactivation
        self.original_image = player.image.copy()

        # Get player size
        width, height = player.image.get_size()

        # Create a fully transparent surface the size of the player
        outline_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        outline_surface.fill((0, 0, 0, 0))  # Ensure it's transparent

        # Draw only the outline
        outline_color = GREEN
        outline_thickness = 2  # Adjust as needed
        pygame.draw.rect(outline_surface, outline_color, outline_surface.get_rect(), outline_thickness)

        # Set this as the player's image
        player.image = outline_surface

        player.is_invisible = True

    def on_deactivate(self, player):
        # Restore the original image
        player.image = self.original_image
        player.is_invisible = False

class Adrenaline(Ability):
    def on_activate(self, player):
        player.stats["fire_rate"] /= 2
        player.stats["speed"] *= 1.25

    def on_deactivate(self, player):
        player.stats["fire_rate"] *= 2
        player.stats["speed"] /= 1.25

class BulletTime(Ability):
    def on_activate(self, player):
        player.time_modifier = 0.5

    def on_deactivate(self, player):
        player.time_modifier = 1.0


ability_paths = {
    'sprint': './sprites/items/abilities/sprint.png',
    'dodge': './sprites/items/abilities/dodge.png',
    'deflect': './sprites/items/abilities/deflect.png',
    'fortify': './sprites/items/abilities/fortify.png',
    'invisibility': './sprites/items/abilities/invisibility.png',
    'adrenaline': './sprites/items/abilities/adrenaline.png',
    'bullettime': './sprites/items/abilities/bullettime.png',
}