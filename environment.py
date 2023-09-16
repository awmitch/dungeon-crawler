from typing import Any
from settings import (
    WIDTH, 
    HEIGHT, 
    WALL_THICKNESS, 
    DOOR_HEIGHT, 
    DOOR_WIDTH, 
    BROWN, 
    PURPLE,
    TRAP_SIZE
)

from enemy import RangedEnemy, MeleeEnemy, RangedBoss, MeleeBoss
from utilities import set_safe_position, Direction, generate_valid_enemy_position
from screen_singleton import screen_singleton as screen
from gamedata import dqn_singleton as central_dqn
from gamedata import diff_scaler_singleton as difficulty_scaler
from abilities import AbilityItem
from item import Item
import pygame
import random
random.seed(43)
class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill((0, 0, 0))  # black walls
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    def update(self, *args: Any, **kwargs: Any) -> None:
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)
    
class RoomNode:
    def __init__(
            self,
            x, y,
            player,
        ):

        self.x = x
        self.y = y
        self.visited = False
        self.neighbors = dict()  # Dictionary with Direction as key
        self.is_end = False
        self.is_start = False
        self.distance_from_start = 0  # To help pick the most distant end node

        # Groups initialization
        self.all_sprites = pygame.sprite.Group()
        self.projectiles = pygame.sprite.Group()
        self.enemy_projectiles = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.doors = pygame.sprite.Group()
        self.walls = pygame.sprite.Group()
        self.items = pygame.sprite.Group()
        
        self.player = player
        self.trapdoor = None

    def has_neighbor(self, direction):
        return direction in self.neighbors
    
    def generate_enemies(self, number):
        if self.is_end:
            x, y = generate_valid_enemy_position(self.doors)
            chosen_enemy_type = random.choice([RangedBoss, MeleeBoss])
            boss = chosen_enemy_type(self.player.enemy_cnt, self, x, y)
            boss.adjust_stats()
            self.enemies.add(boss)
            self.player.enemy_cnt += 1
            central_dqn.reset_reward(boss.enemy_id)
        else:
            for _ in range(number):
                # Randomly choose between RangedEnemy and MeleeEnemy
                chosen_enemy_type = random.choice([RangedEnemy, MeleeEnemy])
                x, y = generate_valid_enemy_position(self.doors)
                enemy = chosen_enemy_type(
                    self.player.enemy_cnt,
                    self,
                    x, y
                )
                enemy.adjust_stats()
                self.enemies.add(enemy)
                self.player.enemy_cnt += 1
                central_dqn.reset_reward(enemy.enemy_id)


    def generate_doors(self):
        if self.has_neighbor(Direction.UP):
            if self.neighbors[Direction.UP].is_end:
                #door.image = pygame.image.load('path_to_skull_image.png')  # Load skull image or any other image marking the boss room
                door_color = PURPLE
            else:
                door_color = BROWN
            door = Door(WIDTH // 2 - DOOR_WIDTH // 2, 0, DOOR_WIDTH, WALL_THICKNESS, Direction.UP, door_color)
            self.doors.add(door)

        if self.has_neighbor(Direction.DOWN):
            if self.neighbors[Direction.DOWN].is_end:
                #door.image = pygame.image.load('path_to_skull_image.png')  # Load skull image or any other image marking the boss room
                door_color = PURPLE
            else:
                door_color = BROWN
            door = Door(WIDTH // 2 - DOOR_WIDTH // 2, HEIGHT - WALL_THICKNESS, DOOR_WIDTH, WALL_THICKNESS, Direction.DOWN, door_color)
            self.doors.add(door)

        if self.has_neighbor(Direction.LEFT):
            if self.neighbors[Direction.LEFT].is_end:
                #door.image = pygame.image.load('path_to_skull_image.png')  # Load skull image or any other image marking the boss room
                door_color = PURPLE
            else:
                door_color = BROWN
            door = Door(0, HEIGHT // 2 - DOOR_HEIGHT // 2, WALL_THICKNESS, DOOR_HEIGHT, Direction.LEFT, door_color)
            self.doors.add(door)

        if self.has_neighbor(Direction.RIGHT):
            if self.neighbors[Direction.RIGHT].is_end:
                #door.image = pygame.image.load('path_to_skull_image.png')  # Load skull image or any other image marking the boss room
                door_color = PURPLE
            else:
                door_color = BROWN
            door = Door(WIDTH - WALL_THICKNESS, HEIGHT // 2 - DOOR_HEIGHT // 2, WALL_THICKNESS, DOOR_HEIGHT, Direction.RIGHT, door_color)
            self.doors.add(door)

    def generate_walls(self):
        # Top walls
        if not self.has_neighbor(Direction.UP):
            self.walls.add(Wall(0, 0, WIDTH, WALL_THICKNESS))
        else:
            self.walls.add(Wall(0, 0, WIDTH // 2 - DOOR_WIDTH // 2, WALL_THICKNESS))  # left of the top door
            self.walls.add(Wall(WIDTH // 2 + DOOR_WIDTH // 2, 0, WIDTH // 2 - DOOR_WIDTH // 2, WALL_THICKNESS))  # right of the top door

        # Bottom walls
        if not self.has_neighbor(Direction.DOWN):
            self.walls.add(Wall(0, HEIGHT - WALL_THICKNESS, WIDTH, WALL_THICKNESS))
        else:
            self.walls.add(Wall(0, HEIGHT - WALL_THICKNESS, WIDTH // 2 - DOOR_WIDTH // 2, WALL_THICKNESS))  # left of the bottom door
            self.walls.add(Wall(WIDTH // 2 + DOOR_WIDTH // 2, HEIGHT - WALL_THICKNESS, WIDTH // 2 - DOOR_WIDTH // 2, WALL_THICKNESS))  # right of the bottom door

        # Left walls
        if not self.has_neighbor(Direction.LEFT):
            self.walls.add(Wall(0, 0, WALL_THICKNESS, HEIGHT))
        else:
            self.walls.add(Wall(0, 0, WALL_THICKNESS, HEIGHT // 2 - DOOR_HEIGHT // 2))  # above the left door
            self.walls.add(Wall(0, HEIGHT // 2 + DOOR_HEIGHT // 2, WALL_THICKNESS, HEIGHT // 2 - DOOR_HEIGHT // 2))  # below the left door

        # Right walls
        if not self.has_neighbor(Direction.RIGHT):
            self.walls.add(Wall(WIDTH - WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT))
        else:
            self.walls.add(Wall(WIDTH - WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT // 2 - DOOR_HEIGHT // 2))  # above the right door
            self.walls.add(Wall(WIDTH - WALL_THICKNESS, HEIGHT // 2 + DOOR_HEIGHT // 2, WALL_THICKNESS, HEIGHT // 2 - DOOR_HEIGHT // 2))  # below the right door

    def generate_ability(self):
        player_x, player_y = self.get_player_pos(center=True)
        ability = self.get_player_ability()
        while ability == self.get_player_ability():
            ability_id = random.randint(0, len(self.player.abilities)-1)
            ability = list(self.player.abilities.keys())[ability_id]
        ability_item = AbilityItem(player_x, player_y, ability, init=True)
        self.all_sprites.add(ability_item)
        self.items.add(ability_item)

    def generate_fresh(self):
        self.generate_walls()
        self.generate_doors()
        if self.is_start:
            self.generate_ability()
        else:
            hp_drop_chance = self.get_hp_chance()
            if random.random() < hp_drop_chance:
                item = Item(WIDTH//2, HEIGHT//2, 'health_pack')
                self.items.add(item)
                self.all_sprites.add(item)
            self.generate_enemies(random.randint(1, 2))
        self.all_sprites.add(
            self.enemies, 
            self.walls, 
            self.doors
        )

    def generate_trapdoor(self):
        player_x, player_y = self.get_player_pos(center=True)
        self.trapdoor = Trapdoor(player_x, player_y)
        self.all_sprites.add(self.trapdoor)
        
    def get_player_ability(self):
        return self.player.active_ability
    
    def get_player_pos(self, center=False):
        return self.player.get_player_pos(center)
    
    def get_player_vel(self):
        return self.player.get_player_vel()

    def get_time_modifier(self):
        return self.player.get_time_modifier()

    def get_hp_chance(self):
        return (1 - self.player.get_health_frac())

    def detect_player(self):
        return not self.player.is_invisible
    
    def reset(self, game_over=False):
        self.projectiles.empty()
        self.enemy_projectiles.empty()
        self.all_sprites.remove(self.player)
        if game_over: 
            difficulty_scaler.reset()
        elif not self.visited:
            difficulty_scaler.game_data.update_data(
                rooms_cleared=1, 
                time_alive=self.player.elapsed_time
            )
            difficulty_scaler.collect_data()
            self.visited = True
    
    def update(self):
        self.all_sprites.update()
        
class Door(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, direction, door_color):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(door_color)  # Brownish color
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.direction = direction
    def update(self, *args: Any, **kwargs: Any) -> None:
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)
class Trapdoor(pygame.sprite.Sprite):
    def __init__(self, player_x, player_y):
        super().__init__()
        #self.image = pygame.image.load('path_to_trapdoor_image.png')
        self.image = pygame.Surface([TRAP_SIZE, TRAP_SIZE])
        self.image.fill(BROWN)  # Brownish color
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = set_safe_position(player_x, player_y, TRAP_SIZE, TRAP_SIZE)
    def update(self, *args: Any, **kwargs: Any) -> None:
        # Draw
        screen.blit(self.image, self.rect)
        return super().update(*args, **kwargs)