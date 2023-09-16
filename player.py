from settings import WIDTH,HEIGHT,WALL_THICKNESS,GREEN, ORANGE, GRAY, BLUE_GRAY
from gamedata import DifficultyScaler, CentralizedDQN
from map import Direction
from utilities import get_fire_direction, check_boundaries, move_with_collision, game_over_screen
from abilities import (
    AbilityItem, 
    Sprint, 
    Dodge, 
    Deflect, 
    Fortify, 
    Invisibility, 
    Adrenaline, 
    BulletTime
)
from projectiles import Projectile
from weapons import Sword
from item import Item
from typing import Any
from screen_singleton import screen_singleton as screen
from gamedata import dqn_singleton as central_dqn
from gamedata import diff_scaler_singleton as difficulty_scaler
import pygame
import math
# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()

        self.enemy_cnt = 0
        self.start_time = pygame.time.get_ticks()

        self.base_stats = {
            "speed": 4,
            "bullet_speed": 10,
            "fire_rate": 200, # milliseconds
            "damage": 25,
            "max_health": 100,  # example value
            "health":100
        }
        self.stats = self.base_stats.copy()

        self.current_velocity_x = 0
        self.current_velocity_y = 0
        self.target_velocity_x = 0
        self.target_velocity_y = 0
        self.acceleration = 1  # This value can be adjusted as per requirement

        self.is_player = True
        self.power_ups = {
            'fire_rate': 0,
            'bullet_speed': 0,
            'speed': 0,
            'damage': 0,
            'max_health': 0,
        }
        self.abilities = {
            'sprint': Sprint(cooldown=5000, duration=2000),
            'dodge': Dodge(cooldown=500, duration=500),
            'deflect': Deflect(cooldown=1500, duration=2000),
            'fortify': Fortify(cooldown=2000, duration=5000),
            'invisibility': Invisibility(cooldown=3000, duration=3000),
            'adrenaline': Adrenaline(cooldown=2000, duration=5000),
            'bullettime': BulletTime(cooldown=2000, duration=5000)
        }
        self.active_ability = None
        self.active_ability_image = None
        self.last_pickup_time = -1000  # set to a value such that initial delay is negated
        self.pickup_delay = 1000  # 1 second delay
        self.time_modifier = 1.0

        self.is_dodging = False
        self.is_deflecting = False
        self.is_invisible = False
        self.armor = 0  # Add armor to player stats

        self.sword = Sword(self)
        self.last_action_time = 0
        self.lunge_distance = 150
        self.lunge_speed = 3 * self.stats["speed"]
        self.lunge_rate_mod = 2
        self.melee_rate_mod = 3

    def update(self, *args: Any, **kwargs: Any) -> None:
        current_tick = pygame.time.get_ticks()
        if self.current_room.enemies:  # Check if there are enemies left in the room
            self.elapsed_time += current_tick - self.last_tick  # Increment the elapsed time
        self.last_tick = current_tick

        # Check for collisions between items and the player
        item_hits = pygame.sprite.spritecollide(self, self.current_room.items, False)  # False means the item won't be removed upon collision
        for item in item_hits:
            if (
                isinstance(item, Item) or
                isinstance(item, AbilityItem) and (current_tick - self.last_pickup_time) > self.pickup_delay
            ):
                item.pick_up(self)
                self.last_pickup_time = current_tick

        # Check for collisions between enemy projectiles and the player
        player_hits = pygame.sprite.spritecollide(self, self.current_room.enemy_projectiles, True)  # True means projectiles will be removed after collision
        for bullet in player_hits:
            if self.is_deflecting:
                dest_x = bullet.owner.rect.centerx
                dest_y = bullet.owner.rect.centery
                bullet = Projectile(self.rect.centerx, self.rect.centery, dest_x, dest_y, self)
                self.current_room.projectiles.add(bullet)
                self.current_room.all_sprites.add(bullet)
            else:
                game_over = self.take_damage(bullet.owner.stats["damage"])  # example damage value
                central_dqn.add_to_reward(bullet.owner.enemy_id, 1)  # Increase reward because the enemy hit the player
                if game_over:
                    return
        
        door_hits = pygame.sprite.spritecollide(self, self.current_room.doors, False)
        if door_hits and (len(self.current_room.enemies) == 0):
            self.current_room.reset()
            # Identify which door was used
            door_direction = door_hits[0].direction  # [0] because spritecollide returns a list

            # Reflect player's position
            if door_direction == Direction.UP:  # North
                self.update_room(door_direction)
                self.rect.y = HEIGHT-(WALL_THICKNESS+(self.rect.bottom-self.rect.y))
            elif door_direction == Direction.DOWN:  # South
                self.update_room(door_direction)
                self.rect.y = WALL_THICKNESS+(self.rect.y-self.rect.top)
            elif door_direction == Direction.LEFT:  # West
                self.update_room(door_direction)
                self.rect.x = WIDTH-(WALL_THICKNESS+(self.rect.right-self.rect.x))
            elif door_direction == Direction.RIGHT:  # East
                self.update_room(door_direction)
                self.rect.x = WALL_THICKNESS+(self.rect.x-self.rect.left)

            return

        if (
            self.current_room.trapdoor is not None 
            and self.rect.colliderect(self.current_room.trapdoor.rect)
        ):
            self.reset()

        # Handle movement
        keys = pygame.key.get_pressed()
        self.fire_projectile(keys)
        self.use_ability(keys)
        if not self.is_dodging:
            angle = self.sword.get_angle_to_target()
            if self.sword.is_lunging:
                # Continue the lunge in the direction of the player
                self.target_velocity_x = self.lunge_speed * math.cos(angle)
                self.target_velocity_y = self.lunge_speed * math.sin(angle)
            else:
                self.move_with_keys(keys)
            self.current_velocity_x = self.handle_velocity(self.target_velocity_x, self.current_velocity_x, self.acceleration)
            self.current_velocity_y =  self.handle_velocity(self.target_velocity_y, self.current_velocity_y, self.acceleration)
        else:
            self.current_velocity_x,self.current_velocity_y = self.abilities["dodge"].get_direction()
        self.change_x = self.current_velocity_x
        self.change_y = self.current_velocity_y
        self.change_x *= self.current_room.get_time_modifier()
        self.change_y *= self.current_room.get_time_modifier()

        # Boundary checks
        self.check_boundaries()

        # Collisions
        collidables = pygame.sprite.Group()
        collidables.add(self.current_room.walls)
        collidables.add(self.current_room.enemies)
        if len(self.current_room.enemies) > 0:
            collidables.add(self.current_room.doors)
        wall_hit = move_with_collision(self, collidables)
        # Draw
        screen.blit(self.image, self.rect.topleft)
        self.decide_melee_attack(keys)

        self.draw_health(screen)
        self.draw_ability_cooldown(screen)
        if self.active_ability:
            screen.blit(self.active_ability_image, (WIDTH - self.active_ability_image.get_width(), HEIGHT - self.active_ability_image.get_height()))

        return super().update(*args, **kwargs)
    
    def decide_melee_attack(self,keys):
        if not self.sword.is_lunging:
            dx, dy = self.get_rel_pos()
            self.sword.set_angle_to_target(dx, dy)
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - self.last_action_time
        if pygame.mouse.get_pressed()[2]:
            melee_rate = self.melee_rate_mod*self.stats["fire_rate"] / self.current_room.get_time_modifier()
            if elapsed_time >= melee_rate:
                if self.sword.swing(): self.last_action_time = current_time
        elif keys[pygame.K_SPACE]:
            lunge_rate = self.melee_rate_mod*self.lunge_rate_mod*self.stats["fire_rate"] / self.current_room.get_time_modifier()
            if elapsed_time >= lunge_rate:
                if self.sword.lunge(): self.last_action_time = current_time + lunge_rate
        self.sword.update(self.current_room.enemies)

    def get_rel_pos(self):
        dest_x, dest_y = pygame.mouse.get_pos()
        dx = dest_x - self.rect.centerx
        dy = dest_y - self.rect.centery
        return dx, dy
    
    def use_ability(self, keys):
        # Check for ability activation
        if self.active_ability:
            ability = self.abilities[self.active_ability]
            if keys[pygame.K_LSHIFT]:
                if not ability.is_active:
                    ability.activate(self)
            else:
                # If the ability is 'dodge', skip the deactivation process
                if self.active_ability in ['dodge','fortify', 'invisibility']:
                    pass
                elif ability.is_active:
                    ability.deactivate_early(self)

        # Update abilities
        for ability in self.abilities.values():
            ability.update(self)
    
    def set_ability(self, ability, ability_image):
            self.active_ability = ability
            self.active_ability_image = ability_image

    def fire_projectile(self, keys):
        now = pygame.time.get_ticks()
        fire_rate = self.stats["fire_rate"] / self.current_room.get_time_modifier()
        if now - self.last_shot > fire_rate:
            self.last_shot = now
            # Check if the left mouse button is pressed
            if any([keys[key] for key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]]):
                fire = True
                fire_direction = get_fire_direction(keys)
            if pygame.mouse.get_pressed()[0]:
                fire = True
                dest_x, dest_y = pygame.mouse.get_pos()
                dir_x, dir_y = dest_x - self.rect.centerx, dest_y - self.rect.centery
                magnitude = (dir_x**2 + dir_y**2) ** 0.5
                fire_direction = dir_x/magnitude, dir_y/magnitude
            else:
                fire = False

            if fire and fire_direction:
                dest_x = self.rect.centerx + fire_direction[0] * 1000  # Arbitrarily large value
                dest_y = self.rect.centery + fire_direction[1] * 1000
                bullet = Projectile(self.rect.centerx, self.rect.centery, dest_x, dest_y, self)
                self.current_room.projectiles.add(bullet)
                self.current_room.all_sprites.add(bullet)
    def handle_velocity(self, target_velocity, current_velocity, acceleration):
        difference = target_velocity - current_velocity
        if abs(difference) < acceleration:
            return target_velocity
        else:
            return current_velocity + acceleration * (1 if difference > 0 else -1)

    def move_with_keys(self,keys):
        if keys[pygame.K_a]:
            self.go_left()
        elif keys[pygame.K_d]:
            self.go_right()
        else:
            self.stop_horizontal()

        if keys[pygame.K_w]:
            self.go_up()
        elif keys[pygame.K_s]:
            self.go_down()
        else:
            self.stop_vertical()

    def check_boundaries(self, width=WIDTH, height=HEIGHT, wall_thickness=WALL_THICKNESS):
        self.rect.left = max(self.rect.left, wall_thickness)
        self.rect.right = min(self.rect.right, width - wall_thickness)
        self.rect.top = max(self.rect.top, wall_thickness)
        self.rect.bottom = min(self.rect.bottom, height - wall_thickness)

    def go_left(self):
        self.target_velocity_x = -self.stats["speed"]

    def go_right(self):
        self.target_velocity_x = self.stats["speed"]

    def go_up(self):
        self.target_velocity_y = -self.stats["speed"]

    def go_down(self):
        self.target_velocity_y = self.stats["speed"]

    def stop(self):
        self.target_velocity_x = 0
        self.target_velocity_y = 0

    def stop_horizontal(self):
        self.target_velocity_x = 0

    def stop_vertical(self):
        self.target_velocity_y = 0

    def take_damage(self, damage):
        difficulty_scaler.game_data.update_data(
            damage_taken=damage, 
        )
        # I-frames when dodging
        if self.is_dodging:
            return 0
        if self.armor > damage:
            self.armor -= damage
            damage = 0
        elif self.armor > 0:
            damage -= self.armor
            self.armor = 0
        self.stats["health"] -= damage

        if self.stats["health"] <= 0:
            return 1
        else:
            return 0

    def draw_health(self, screen):
        # Set up the dimensions and position for the health bar
        bar_length = 50
        bar_height = 5
        pos = (self.rect.x, self.rect.y - 15)  # 15 pixels above the player
        fill = (self.stats["health"] / self.stats["max_health"]) * bar_length
        # Calculate armor fill length as a proportion of max health
        armor_fill = (self.armor / self.stats["max_health"]) * bar_length
        
        # Draw health bar background (gray)
        pygame.draw.rect(screen, GRAY, (pos[0], pos[1], bar_length, bar_height))
        # Draw health bar fill (green)
        pygame.draw.rect(screen, (0, 255, 0), (pos[0], pos[1], fill, bar_height))
        # Draw armor bar fill (blue/gray) over health bar
        pygame.draw.rect(screen, BLUE_GRAY, (pos[0], pos[1], armor_fill, bar_height))  # SteelBlue color (70, 130, 180)

    def draw_ability_cooldown(self, screen):
        if not self.active_ability:
            return

        ability = self.abilities[self.active_ability]

        # Set up the dimensions and position for the ability cooldown bar
        bar_length = 50
        bar_height = 5
        pos = (self.rect.x, self.rect.y - 10)  # 10 pixels above the player, below health bar

        current_time = pygame.time.get_ticks()
        elapsed_since_use = current_time - ability.end_time
        
        if ability.is_active:
            fill = ((ability.duration_time - (current_time - ability.start_time)) / ability.duration_time) * bar_length
        else:
            fill = (elapsed_since_use / ability.cooldown_time) * bar_length

        # Clip to maximum bar_length
        fill = min(fill, bar_length)

        # Draw ability cooldown bar background (gray)
        pygame.draw.rect(screen, (128, 128, 128), (pos[0], pos[1], bar_length, bar_height))
        # Draw ability cooldown bar fill (blue for differentiation)
        pygame.draw.rect(screen, ORANGE, (pos[0], pos[1], fill, bar_height))

    def reset_position(self):
        self.rect.x = WIDTH // 2
        self.rect.y = HEIGHT // 2
        self.change_x = 0
        self.change_y = 0

    def reset(self, game_over=False, init=False):
        # Reset game state
        
        if game_over:
            self.reset_position()  # Assuming you create a method to reset player's position
            self.stats = self.base_stats.copy()
            for key in self.power_ups.keys():
                self.power_ups[key] = 0
            if self.active_ability:
                ability = self.abilities[self.active_ability]
                if ability.is_active:
                    ability.is_active = False
                    ability.on_deactivate(self)
                self.active_ability = None
            self.next_level = False
            self.elapsed_time = 0
            self.last_tick = pygame.time.get_ticks()
            self.last_shot = self.last_tick
        else:
            self.next_level = True
        if not init:
            self.current_room.reset(game_over)
        
    def update_room(self, direction):
        self.current_room = self.current_room.neighbors[direction]
        self.add_sprite()

    def add_sprite(self):
        self.current_room.all_sprites.add(self)
        
    def get_player_pos(self, center=False):
        if center:
            return self.rect.centerx, self.rect.centery
        else:
            return self.rect.x, self.rect.y
    
    def get_player_vel(self):
        return self.current_velocity_x, self.current_velocity_y

    def get_time_modifier(self):
        return self.time_modifier
    
    def get_health_frac(self):
        return self.stats["health"] / self.stats["max_health"]

    def set_current_room(self, room_node):
        self.current_room = room_node