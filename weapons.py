import pygame
import math
from settings import (
    DARK_GRAY,
    BLACK
)
from screen_singleton import screen_singleton as screen

class Sword:
    def __init__(self, owner):
        self.owner = owner
        self.length = 70
        self.swing_angle = math.radians(80) # Initialize at the start angle
        self.current_swing_angle = self.swing_angle
        self.swing_direction = 1
        self.angle_to_target = 0
        self.last_swing_time = 0
        self.is_lunging = False
        self.last_lunge_time = 0
        self.last_action_time = 0
        self.lunge_duration = 500
        self.swing_duration = 300

        self.handle_length = 15
        self.crossguard_length = 15
        self.offset = 10
        self.base_offset = self.offset + self.handle_length
        self.is_swinging = False
        self.damaged_enemies = set()  # set to store enemies that have been damaged during an action
        
        # Define sword attributes
        self.image = pygame.Surface((self.length, 10))
        self.image.fill(DARK_GRAY)
        
        # For the crossguard
        self.crossguard_image = pygame.Surface((20, 5))
        self.crossguard_image.fill(BLACK)

    def update(self, enemies):
        if not isinstance(enemies, pygame.sprite.Group):
            enemy = enemies
            enemies = pygame.sprite.Group()
            enemies.add(enemy)
        # Stop the lunge after a certain time
        current_time = pygame.time.get_ticks()
        if self.is_lunging:
            if current_time - self.last_lunge_time > self.lunge_duration:
                self.stop_lunge()
            self.check_lunge_collision(enemies)

        if self.is_swinging:
            self.update_swing(enemies)

        self.draw()

    def swing(self):
        if not self.is_swinging:
            self.last_swing_time = pygame.time.get_ticks()
            self.damaged_enemies.clear()  # clear the set when starting a new swing
            self.current_swing_angle = self.swing_direction*self.swing_angle  # Initialize at the start angle
            self.is_swinging = True
            return True
        else:
            return False

    def update_swing(self, enemies):
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - self.last_swing_time
        swing_rate = self.swing_duration / self.owner.current_room.get_time_modifier()
        if 0 <= elapsed_time < swing_rate:
            swing_progress = elapsed_time / swing_rate
            start_swing_angle = self.swing_direction*self.swing_angle
            # Calculate the change in swing angle based on progress
            self.current_swing_angle = start_swing_angle * (1.0 - 2.0*swing_progress)

            crescent_points = self.get_polygon()
            polygon_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            # Draw the crescent shape
            pygame.draw.polygon(polygon_surface, (255, 255, 255, 50), crescent_points)
            screen.blit(polygon_surface, (0, 0))
            polygon = pygame.draw.polygon(screen, (255, 255, 255, 50), self.get_polygon())
            for enemy in enemies:
                if polygon.colliderect(enemy.rect) and enemy not in self.damaged_enemies:
                    enemy.take_damage(self.owner.stats["damage"])
                    self.damaged_enemies.add(enemy)  # Add the damaged enemy to the set

        else:
            self.is_swinging = False
            self.last_swing_time = current_time
            self.swing_direction *= -1

    def lunge(self):
        if not self.is_lunging and not self.is_swinging:
            self.damaged_enemies.clear()  # clear the set when starting a new lunge
            self.is_lunging = True
            self.last_lunge_time = pygame.time.get_ticks()
            return True
        else:
            return False

    def stop_lunge(self):
        self.is_lunging = False

    def check_lunge_collision(self, enemies):
        # Calculate the tip of the sword
        tip_x = self.owner.rect.centerx + self.length * math.cos(self.angle_to_target)
        tip_y = self.owner.rect.centery + self.length * math.sin(self.angle_to_target)

        # Check if the tip of the sword is inside the player's rectangle
        for enemy in enemies:
            if enemy.rect.collidepoint(tip_x, tip_y) and enemy not in self.damaged_enemies:
                enemy.take_damage(self.owner.stats["damage"])
                self.damaged_enemies.add(enemy)  # Add the damaged enemy to the set

    def draw_stabbing(self):
        self.draw_elements(self.angle_to_target)
    
    def draw_elements(self, angle):
        # Calculate sword's endpoints
        tip_x = self.owner.rect.centerx + self.length * math.cos(angle)
        tip_y = self.owner.rect.centery + self.length * math.sin(angle)

        # Adjusted sword base position
        base_x = self.owner.rect.centerx + self.base_offset * math.cos(angle)
        base_y = self.owner.rect.centery + self.base_offset * math.sin(angle)
        
        # Calculate handle endpoints
        handle_start_x = self.owner.rect.centerx + self.offset * math.cos(angle)
        handle_start_y = self.owner.rect.centery + self.offset * math.sin(angle)
        
        # Draw the handle
        pygame.draw.line(screen, BLACK, (handle_start_x, handle_start_y), (base_x, base_y), 10)

        # Draw the sword
        pygame.draw.line(screen, DARK_GRAY, (base_x, base_y), (tip_x, tip_y), 10)
        
        # Calculate crossguard endpoints
        cg_start_x = base_x - self.crossguard_length * math.sin(angle)
        cg_start_y = base_y + self.crossguard_length * math.cos(angle)
        cg_end_x = base_x + self.crossguard_length * math.sin(angle)
        cg_end_y = base_y - self.crossguard_length * math.cos(angle)
        
        # Draw the crossguard
        pygame.draw.line(screen, BLACK, (cg_start_x, cg_start_y), (cg_end_x, cg_end_y), 5)

    def draw_resting(self):
        # Just an example, adjust as needed. Perhaps a slight angle for the resting state.
        #resting_angle = math.radians(45)  
        self.draw_elements(self.angle_to_target + self.current_swing_angle )

    def draw_swinging(self):
        angle_to_target = self.angle_to_target + self.current_swing_angle
        self.draw_elements(angle_to_target)

    # Example:
    def draw(self):
        if self.is_lunging:
            self.draw_stabbing()
        elif self.is_swinging:
            self.draw_swinging()
        else:
            self.draw_resting()

    def get_polygon(self):
        start_angle = self.angle_to_target + self.swing_direction*self.swing_angle
        end_angle = self.angle_to_target + self.current_swing_angle

        num_points = 10
        # Outer polygon (sword tip)
        wedge_points_outer = []
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = self.owner.rect.centerx + self.length * math.cos(angle)
            y = self.owner.rect.centery + self.length * math.sin(angle)
            wedge_points_outer.append((x, y))

        # Inner polygon (sword base)
        wedge_points_inner = []
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = self.owner.rect.centerx + self.base_offset * math.cos(angle)
            y = self.owner.rect.centery + self.base_offset * math.sin(angle)
            wedge_points_inner.append((x, y))

        # Combine the outer and inner points
        crescent_points = wedge_points_outer + wedge_points_inner[::-1]

        return crescent_points
    
    def set_angle_to_target(self, rel_x, rel_y):
        self.angle_to_target = math.atan2(rel_y, rel_x)

    def get_angle_to_target(self, rad=True):
        if rad:
            return self.angle_to_target
        else:
            return math.degrees(self.angle_to_target)