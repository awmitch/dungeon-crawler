from settings import (
    WIDTH,HEIGHT,
    WALL_THICKNESS,
    RED, 
    PURPLE,
    ORANGE,
    BLACK,
    DARK_GRAY,
    BATCH_SIZE,
    ENEMY_SIZE,
)
from typing import Any
from projectiles import Projectile
from utilities import set_safe_position, check_boundaries, move_with_collision
from screen_singleton import screen_singleton as screen
from gamedata import dqn_singleton as central_dqn
from gamedata import diff_scaler_singleton as difficulty_scaler
from item import Item
from weapons import Sword
import pygame
import torch
import math
import random
random.seed(43)
class Enemy(pygame.sprite.Sprite):
    def __init__(self, enemy_id, room_node, x, y):
        super().__init__()
        self.image = pygame.Surface([40, 40])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.change_x = 0
        self.change_y = 0
        self.current_velocity_x = 0
        self.current_velocity_y = 0
        self.target_velocity_x = 0
        self.target_velocity_y = 0
        self.acceleration = 0.5  # This value can be adjusted as per requirement
        self.base_stats = {
            "speed": 3,
            "bullet_speed": 10,
            "fire_rate": 500, # milliseconds
            "damage": 10,
            "max_health": 50,  # example value
            "health": 50
        }
        self.stats = self.base_stats.copy()
        self.last_shot = pygame.time.get_ticks()
        self.enemy_id = enemy_id  # Unique ID for each enemy
        self.current_room = room_node
        # Safely set the enemy's position based on player's position
        #player_x, player_y = self.current_room.get_player_pos()
        self.rect.x, self.rect.y = x, y #set_safe_position(player_x, player_y)
        self.is_boss = False
        self.is_player = False
        self.chance_to_drop = 0.25  # example 25% chance to drop an item
        # Movement mapping
        self.movement_functions = {
            0: self.go_left,
            1: self.go_right,
            2: self.go_up,
            3: self.go_down
        }

    def adjust_stats(self):
        self.stats["fire_rate"] *= difficulty_scaler.global_multiplier_fire_rate
        self.stats["speed"] *= difficulty_scaler.global_multiplier_speed
        self.stats["max_health"] *= difficulty_scaler.global_multiplier_health
        self.stats["health"] *= difficulty_scaler.global_multiplier_health

    def take_damage(self, damage):
        self.stats["health"] -= damage
        difficulty_scaler.game_data.update_data(
            damage_dealt=damage, 
        )
        if self.stats["health"] <= 0:
            #central_dqn.add_to_reward(self.enemy_id, -5)  # Deduct reward when enemy is killed
            self.kill()  # Remove this enemy instance
            self.on_death()

    def draw_health(self, screen):
        # Similar to the player's health bar drawing method but adjust position or size if necessary
        bar_length = 40
        bar_height = 5
        pos = (self.rect.x, self.rect.y - 10)
        fill = (self.stats["health"] / self.stats["max_health"]) * bar_length
        pygame.draw.rect(screen, (128, 128, 128), (pos[0], pos[1], bar_length, bar_height))
        pygame.draw.rect(screen, (255, 0, 0), (pos[0], pos[1], fill, bar_height))
    
    def get_state(self):
        player_x, player_y = self.current_room.get_player_pos(center=True)
        player_vel_x, player_vel_y = self.current_room.get_player_vel()
        enemy_vel_x, enemy_vel_y = self.current_velocity_x, self.current_velocity_y

        # State now includes velocities for both player and enemy
        return torch.tensor([
            self.rect.x, self.rect.y, 
            player_x, player_y, 
            player_vel_x, player_vel_y, 
            enemy_vel_x, enemy_vel_y, 
            self.enemy_id
            ], dtype=torch.float32).unsqueeze(0)

    def pathing(self, contribute_state=True):
        # Capture the current state before taking any action
        current_state = self.get_state()

        action = central_dqn.select_action(current_state, self.enemy_id)  # Assuming each enemy has a unique 'id' attribute
        # Convert action tensor into scalar
        action_scalar = action.item()

        self.movement_functions[action_scalar]()

        self.adjust_velocity()
        # Collision handling and boundary checks
        self.handle_collisions_and_boundaries()

        if contribute_state:
            next_state = self.get_state()

            reward = central_dqn.get_reward(self.enemy_id)  
            central_dqn.store_transition(self.enemy_id, current_state, action, reward, next_state)
            central_dqn.reset_reward(self.enemy_id)
            central_dqn.optimize_model()
            
    def adjust_velocity(self):
        # Adjust velocity based on acceleration
        difference_x = self.target_velocity_x - self.current_velocity_x
        difference_y = self.target_velocity_y - self.current_velocity_y

        self.current_velocity_x += self.acceleration * (1 if difference_x > 0 else -1) if abs(difference_x) >= self.acceleration else difference_x
        self.current_velocity_y += self.acceleration * (1 if difference_y > 0 else -1) if abs(difference_y) >= self.acceleration else difference_y

        self.change_x, self.change_y = self.current_velocity_x, self.current_velocity_y
        self.change_x *= self.current_room.get_time_modifier()
        self.change_y *= self.current_room.get_time_modifier()

    def handle_collisions_and_boundaries(self):
        collidables = pygame.sprite.Group()
        collidables.add(self.current_room.walls)
        collidables.add(self.current_room.doors)
        collidables.add(self.current_room.player)
        enemies_minus_self = self.current_room.enemies.copy()
        enemies_minus_self.remove(self)
        collidables.add(enemies_minus_self)
        
        wall_hit = move_with_collision(self, collidables)
        if wall_hit:
            central_dqn.add_to_reward(self.enemy_id, -100)  # Penalty for collision

        self.rect.clamp_ip(pygame.Rect(WALL_THICKNESS, WALL_THICKNESS, WIDTH - WALL_THICKNESS - ENEMY_SIZE, HEIGHT - WALL_THICKNESS - ENEMY_SIZE))

    def go_left(self):
        self.target_velocity_x = -self.stats["speed"]

    def go_right(self):
        self.target_velocity_x = self.stats["speed"]

    def go_up(self):
        self.target_velocity_y = -self.stats["speed"]

    def go_down(self):
        self.target_velocity_y = self.stats["speed"]

    def update(self, *args: Any, **kwargs: Any) -> None:

        # Check for collisions between player projectiles and enemies
        enemy_hits = pygame.sprite.spritecollide(self, self.current_room.projectiles, True)

        for bullet in enemy_hits:
            self.take_damage(bullet.owner.stats["damage"])  # example damage value
            #central_dqn.add_to_reward(self.enemy_id, -1)  # Decrease reward because the player hit the enemy

        # Draw
        screen.blit(self.image, self.rect)
        self.draw_health(screen)
        return super().update(*args, **kwargs)
    
    def on_death(self):
        if self.is_boss: 
            self.current_room.generate_trapdoor()
            item = Item(WIDTH//2, HEIGHT//2, 'health_pack')
            self.current_room.items.add(item)
            self.current_room.all_sprites.add(item)
        if random.random() < self.chance_to_drop:
            # Determine the type of boost this item will provide
            type = random.choice([
                'fire_rate', 
                'bullet_speed', 
                'speed', 
                'damage', 
                'max_health', 
            ])
            item = Item(self.rect.x, self.rect.y, type)
            self.current_room.items.add(item)
            self.current_room.all_sprites.add(item)

class MeleeEnemy(Enemy):
    def __init__(self, enemy_id, room_node, x, y):
        super().__init__(enemy_id, room_node, x, y)
        self.sword = Sword(self)
        
        # self.stats["speed"] *= 1.5
        self.stats["max_health"] *= 1.5
        self.stats["health"] *= 1.5
        # self.stats["fire_rate"] *= 1.5
        self.image.fill(ORANGE)
        self.last_action_time = 0
        self.lunge_distance = 150
        self.lunge_speed = 3 * self.stats["speed"]
        self.lunge_rate_mod = 2

    def melee_pathing(self):
        
        angle = self.sword.get_angle_to_target()
        if self.sword.is_lunging:
            # Continue the lunge in the direction of the player
            self.target_velocity_x = self.lunge_speed * math.cos(angle)
            self.target_velocity_y = self.lunge_speed * math.sin(angle)
        else:
            # Set movement to approach the player
            self.target_velocity_x = self.stats["speed"] * math.cos(angle)
            self.target_velocity_y = self.stats["speed"] * math.sin(angle)
        
        # Adjust velocity and handle collisions
        self.adjust_velocity()
        self.handle_collisions_and_boundaries()

    # Check if player is near the enemy
    def is_player_near(self, check_distance):
        player_rect = self.current_room.player.rect
        # Calculate distance from all four corners of the player to the enemy's center
        distances = [
            math.sqrt((self.rect.centerx - x)**2 + (self.rect.centery - y)**2)
            for x, y in [
                (player_rect.topleft), (player_rect.topright),
                (player_rect.bottomleft), (player_rect.bottomright)
            ]
        ]
        
        # Check if any distance is less than the sword's arc radius
        return any(distance <= check_distance for distance in distances)
    
    def decide_melee_attack(self):
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - self.last_action_time
        if self.is_player_near(self.sword.length):
            if elapsed_time >= self.stats["fire_rate"] / self.current_room.get_time_modifier():
                if self.sword.swing(): self.last_action_time = current_time
        elif self.is_player_near(self.lunge_distance):
            lunge_rate = self.lunge_rate_mod*self.stats["fire_rate"] / self.current_room.get_time_modifier()
            if elapsed_time >= lunge_rate:
                if self.sword.lunge(): self.last_action_time = current_time + lunge_rate
        self.sword.update(self.current_room.player)

    def update(self):
        super().update()
        if not self.sword.is_lunging:
            dx, dy = self.get_rel_pos()
            self.sword.set_angle_to_target(dx, dy)

        if self.current_room.detect_player():
            self.melee_pathing()
        else:
            self.pathing(contribute_state=False)
        
        # If the player is farther than the sword arc radius but within lunging distance
        self.decide_melee_attack()
        
    def get_rel_pos(self):
        # Get player position
        player_x, player_y = self.current_room.get_player_pos(center=True)

        # Calculate the distance and angle to the player from the enemy
        dx = player_x - self.rect.centerx
        dy = player_y - self.rect.centery
        return dx, dy

class RangedEnemy(Enemy):
    def __init__(self, enemy_id, room_node, x, y):
        super().__init__(enemy_id, room_node, x, y)

    def fire_projectile(self):
        # Shoot based on fire rate
        now = pygame.time.get_ticks()
        fire_rate = self.stats["fire_rate"] / self.current_room.get_time_modifier()
        if now - self.last_shot > fire_rate:
            self.last_shot = now

            player_x, player_y = self.current_room.get_player_pos(center=True)

            # Calculate distance between player and enemy
            dx = player_x - self.rect.centerx
            dy = player_y - self.rect.centery

            player_vel_x, player_vel_y = self.current_room.get_player_vel()

            # Quadratic coefficients
            a = (player_vel_x**2 + player_vel_y**2 - self.stats["bullet_speed"]**2)
            b = 2 * (dx * player_vel_x + dy * player_vel_y)
            c = dx**2 + dy**2

            # Discriminant
            D = b**2 - 4*a*c

            if D >= 0:  # means there's a solution, bullet can hit the player
                # We only take the smaller root because it's the sooner interception time
                if a == 0: a = 1e-6
                time_to_reach = (-b - math.sqrt(D)) / (2*a)
                

                # Calculate predicted player position based on velocity
                predicted_x = player_x + player_vel_x * time_to_reach
                predicted_y = player_y + player_vel_y * time_to_reach

                # Clamp predicted values within map bounds considering the WALL_THICKNESS
                predicted_x = max(2*WALL_THICKNESS, min(predicted_x, WIDTH - 2*WALL_THICKNESS))
                predicted_y = max(2*WALL_THICKNESS, min(predicted_y, HEIGHT - 2*WALL_THICKNESS))

                bullet = Projectile(self.rect.centerx, self.rect.centery, predicted_x, predicted_y, self)
                self.current_room.enemy_projectiles.add(bullet)
                self.current_room.all_sprites.add(bullet)

    def update(self):
        super().update()  # Use the update method from the base class

        # Add a small time penalty
        central_dqn.add_to_reward(self.enemy_id, -0.01)

        # Handle movement
        self.pathing()

        if self.current_room.detect_player():
            self.fire_projectile()

class RangedBoss(RangedEnemy):
    def __init__(self, enemy_id, room_node, x, y):
        super().__init__(enemy_id, room_node, x, y)
        self.image.fill(PURPLE)  # Making boss visually different, maybe purple or another color
        self.stats["max_health"] *= 3  # Boss will have more health than regular enemy
        self.stats["health"] = self.stats["max_health"]
        self.stats["fire_rate"] /= 2  # Boss shoots more frequently
        self.is_boss = True
        self.chance_to_drop = 1  # example 25% chance to drop an item

        # Additional attributes for the boss can be added here

class MeleeBoss(MeleeEnemy):
    def __init__(self, enemy_id, room_node, x, y):
        super().__init__(enemy_id, room_node, x, y)
        self.image.fill(PURPLE)  # Making boss visually different, maybe purple or another color
        self.stats["max_health"] *= 3  # Boss will have more health than regular enemy
        self.stats["health"] = self.stats["max_health"]
        # self.stats["fire_rate"] /= 1.5  # Boss shoots more frequently
        self.stats["speed"] *= 1.5  # Boss is FAST
        self.is_boss = True
        self.chance_to_drop = 1  # example 25% chance to drop an item
        self.sword.length *= 1.25
        # Additional attributes for the boss can be added here
