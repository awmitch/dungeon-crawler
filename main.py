
from settings import WIDTH, HEIGHT, WHITE, WALL_THICKNESS, START_ROOMS, LIGHT_GRAY
from utilities import game_over_screen, draw_timer, draw_level, draw_difficulty, draw_power_ups, start_screen
from screen_singleton import screen_singleton as screen
from gamedata import diff_scaler_singleton as difficulty_scaler
from map import Dungeon
from player import Player
import pygame

# Initialize pygame
pygame.init()

# Screen setup
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption('Rogue-like Game')

# Main loop
running = True

player = Player()
fire = False
fire_direction = None

dungeon = Dungeon(player)

# Call start screen before entering main game loop
start_screen(screen)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(LIGHT_GRAY)
    player.current_room.update()
    #player.current_room.all_sprites.draw(screen)
    if (player.stats["health"] <= 0):
        game_over_screen(screen)
        # Reset game state
        dungeon.reset()

    if player.next_level:
        difficulty_scaler.adjust_difficulty()
        # Generate new level
        dungeon.generate_level()
        player.next_level = False

    draw_timer(screen, player.elapsed_time, len(player.current_room.enemies))
    draw_level(screen, len(dungeon.level_nodes))
    draw_difficulty(screen)
    draw_power_ups(screen, player.power_ups)

    dungeon.level_nodes[-1].draw_minimap(screen, player.current_room)
    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()

