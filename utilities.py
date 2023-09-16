import pygame
import random
random.seed(43)
import math
from settings import (
    WIDTH,
    HEIGHT,
    WALL_THICKNESS, 
    FONT_SIZE, 
    GREEN, 
    RED,
    WHITE,
    ORANGE,
    PURPLE,
    YELLOW,
    BLUE,
    LIGHT_GRAY,
    DOOR_HEIGHT,
    DOOR_WIDTH,
    ENEMY_SIZE
)
from gamedata import diff_scaler_singleton as difficulty_scaler
from enum import Enum
import math
pygame.font.init()
game_over_font = pygame.font.Font(None, 74)
game_over_color = (255, 0, 0)
item_colors = {
    "fire_rate":YELLOW,
    "bullet_speed":PURPLE,
    "speed":BLUE,
    "damage":ORANGE,
    "max_health":RED,
    "health_pack":GREEN
}

def game_over_screen(screen):
    screen.fill(WHITE)
    game_over_text = game_over_font.render('Game Over', True, game_over_color)
    text_rect = game_over_text.get_rect(center=(WIDTH/2, HEIGHT/2))
    screen.blit(game_over_text, text_rect)
    pygame.display.flip()
    waiting_for_restart = True
    while waiting_for_restart:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_r:  # Press R to restart
                    waiting_for_restart = False

def start_screen(screen):
    font_title = pygame.font.Font(None, 74)
    font_instructions = pygame.font.Font(None, 36)

    title = font_title.render('Game', True, GREEN)
    instructions = font_instructions.render('Press Enter to start', True, GREEN)

    screen.fill(LIGHT_GRAY)

    title_rect = title.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 50))
    instructions_rect = instructions.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 50))

    screen.blit(title, title_rect)
    screen.blit(instructions, instructions_rect)

    pygame.display.flip()

    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    waiting_for_input = False


def check_boundaries(sprite):
    # Check left boundary
    if sprite.rect.left < 0:
        sprite.rect.left = 0
        sprite.change_x = 0
        
    # Check right boundary
    if sprite.rect.right > WIDTH:
        sprite.rect.right = WIDTH
        sprite.change_x = 0
        
    # Check top boundary
    if sprite.rect.top < 0:
        sprite.rect.top = 0
        sprite.change_y = 0
        
    # Check bottom boundary
    if sprite.rect.bottom > HEIGHT:
        sprite.rect.bottom = HEIGHT
        sprite.change_y = 0

def move_with_collision(sprite, walls_group):
    original_change_x = sprite.change_x
    original_change_y = sprite.change_y
    wall_hit = False

    # Add max iterations to prevent infinite loops
    max_iterations = abs(sprite.change_x) + abs(sprite.change_y)
    iterations = 0

    # Horizontal movement
    while sprite.change_x != 0 and iterations < max_iterations:
        if sprite.change_x > 0:
            sprite.rect.x += 1
            sprite.change_x -= 1
        else:
            sprite.rect.x -= 1
            sprite.change_x += 1

        wall_hits = pygame.sprite.spritecollide(sprite, walls_group, False)
        if wall_hits:
            wall_hit = True
            if original_change_x > 0:
                sprite.rect.right = wall_hits[0].rect.left
            else:
                sprite.rect.left = wall_hits[0].rect.right
            sprite.change_x = 0
        
        iterations += 1

    # Reset iterations for vertical movement
    iterations = 0

    # Vertical movement
    while sprite.change_y != 0 and iterations < max_iterations:
        if sprite.change_y > 0:
            sprite.rect.y += 1
            sprite.change_y -= 1
        else:
            sprite.rect.y -= 1
            sprite.change_y += 1

        wall_hits = pygame.sprite.spritecollide(sprite, walls_group, False)
        if wall_hits:
            wall_hit = True
            if original_change_y > 0:
                sprite.rect.bottom = wall_hits[0].rect.top
            else:
                sprite.rect.top = wall_hits[0].rect.bottom
            sprite.change_y = 0
        
        iterations += 1
    
    return wall_hit
def reflect(value, axis_value):
    return axis_value + (axis_value - value)

def set_safe_position(player_x, player_y, size_x, size_y, safe_radius=100):
    """Sets the position to a safe distance from the player."""

    # Get a random position
    x = random.randint(WALL_THICKNESS, WIDTH - WALL_THICKNESS - size_x)
    y = random.randint(WALL_THICKNESS, HEIGHT - WALL_THICKNESS - size_y)
    # Keep regenerating until the position is safe
    while is_too_close(x, y, player_x, player_y, safe_radius):
        x = random.randint(WALL_THICKNESS, WIDTH - WALL_THICKNESS - size_x)
        y = random.randint(WALL_THICKNESS, HEIGHT - WALL_THICKNESS - size_y)
    return x,y

@staticmethod
def is_too_close(x1, y1, x2, y2, radius):
    """Check if the two positions are closer than the given radius."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < radius


def get_fire_direction(keys):
    dx, dy = 0, 0
    if keys[pygame.K_UP]:
        dy -= 1
    if keys[pygame.K_DOWN]:
        dy += 1
    if keys[pygame.K_LEFT]:
        dx -= 1
    if keys[pygame.K_RIGHT]:
        dx += 1

    # If we are not pressing any arrow keys, return None
    if dx == 0 and dy == 0:
        return None

    # Normalize the vector for diagonals (so that diagonal shots don't move faster)
    magnitude = (dx**2 + dy**2) ** 0.5
    return dx/magnitude, dy/magnitude

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def opposite(self):
        return {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }[self]
    
def draw_timer(screen, elapsed_time, num_enemies):
    # Convert the elapsed time into minutes and seconds
    minutes = elapsed_time // 60000  # 60000 milliseconds in a minute
    seconds = (elapsed_time % 60000) // 1000  # Convert remainder to seconds

    # Render the time and draw it on the screen
    font = pygame.font.SysFont(None, FONT_SIZE)
    if num_enemies > 0:
        color = RED
    else:
        color = GREEN
    time_text = font.render(f'Time: {minutes:02}:{seconds:02}', True, color)  # Change color if necessary
    screen.blit(time_text, (WIDTH-200, WALL_THICKNESS // 2 - FONT_SIZE // 3))  # Adjust position as needed

def draw_level(screen, level):
    # Render the level and draw it on the screen
    font = pygame.font.SysFont(None, FONT_SIZE)
    level_text = font.render(f'Level: {level:02}', True, WHITE)  # Change color if necessary
    screen.blit(level_text, (WIDTH-200, HEIGHT- WALL_THICKNESS // 2 - FONT_SIZE // 3))  # Adjust position as needed

def draw_difficulty(screen):
    # Render the level and draw it on the screen
    font = pygame.font.SysFont(None, FONT_SIZE)
    difficulty = difficulty_scaler.average_difficulty_scaling()
    level_text = font.render(f'Difficulty: {difficulty:.2f}x', True, WHITE)  # Change color if necessary
    screen.blit(level_text, (WALL_THICKNESS // 2 - FONT_SIZE // 3, HEIGHT- WALL_THICKNESS // 2 - FONT_SIZE // 3))  # Adjust position as needed

def draw_power_ups(screen, power_ups):
    # Render the level and draw it on the screen
    font = pygame.font.SysFont(None, 2*FONT_SIZE//3)
    fire_rate = font.render(f'F: {power_ups["fire_rate"]:01}', True, item_colors["fire_rate"])  # Change color if necessary
    bullet_speed = font.render(f'B: {power_ups["bullet_speed"]:01}', True, item_colors["bullet_speed"])  # Change color if necessary
    speed = font.render(f'S: {power_ups["speed"]:01}', True, item_colors["speed"])  # Change color if necessary
    damage = font.render(f'D: {power_ups["damage"]:01}', True, item_colors["damage"])  # Change color if necessary
    max_health = font.render(f'H: {power_ups["max_health"]:01}', True, item_colors["max_health"])  # Change color if necessary
    screen.blit(fire_rate, (WALL_THICKNESS // 2 - FONT_SIZE // 2, HEIGHT- WALL_THICKNESS - FONT_SIZE))  # Adjust position as needed
    screen.blit(bullet_speed, (WALL_THICKNESS // 2 - FONT_SIZE // 2, HEIGHT- WALL_THICKNESS - 2*FONT_SIZE))  # Adjust position as needed
    screen.blit(speed, (WALL_THICKNESS // 2 - FONT_SIZE // 2, HEIGHT- WALL_THICKNESS - 3*FONT_SIZE))  # Adjust position as needed
    screen.blit(damage, (WALL_THICKNESS // 2 - FONT_SIZE // 2, HEIGHT- WALL_THICKNESS - 4*FONT_SIZE))  # Adjust position as needed
    screen.blit(max_health, (WALL_THICKNESS // 2 - FONT_SIZE // 2, HEIGHT- WALL_THICKNESS - 5*FONT_SIZE))  # Adjust position as needed

def is_too_close_to_door(x, y, doors, safe_radius=100):
    for door in doors:
        door_center_x = door.rect.x + door.rect.width // 2
        door_center_y = door.rect.y + door.rect.height // 2
        
        if door.direction == Direction.UP or door.direction == Direction.DOWN:
            if abs(x - door_center_x) <= door.rect.width // 2 and abs(y - door_center_y) <= safe_radius:
                return True
        elif door.direction == Direction.LEFT or door.direction == Direction.RIGHT:
            if abs(y - door_center_y) <= door.rect.height // 2 and abs(x - door_center_x) <= safe_radius:
                return True
    return False

def generate_valid_enemy_position(doors, safe_radius=100):
    while True:
        x = random.randint(WALL_THICKNESS, WIDTH - WALL_THICKNESS - ENEMY_SIZE)
        y = random.randint(WALL_THICKNESS, HEIGHT - WALL_THICKNESS - ENEMY_SIZE)
        if not is_too_close_to_door(x, y, doors, safe_radius):
            return x, y


