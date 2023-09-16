from settings import (
    MINIMAP_PADDING, MINIMAP_CELL_SIZE, GREEN, PURPLE,
    MINIMAP_POSITION_X, MINIMAP_POSITION_Y, START_ROOMS, ROOM_GROWTH
)
from environment import RoomNode
from utilities import Direction
from gamedata import diff_scaler_singleton as difficulty_scaler
import pygame
import random
random.seed(43)

def get_min_max_coordinates(nodes):
    min_x = min(node.x for node in nodes if node)
    max_x = max(node.x for node in nodes if node)
    
    min_y = min(node.y for node in nodes if node)
    max_y = max(node.y for node in nodes if node)

    return min_x, max_x, min_y, max_y

def should_draw_node(player_node, node):
    if node == player_node:
        return True
    if node.visited:
        return True
    for neighbor in node.neighbors.values():
        if neighbor and (neighbor == player_node or neighbor.visited):
            return True
    return False

def draw_door(minimap_surface, node, neighbor, direction, offset_x, offset_y):
    door_width = MINIMAP_CELL_SIZE // 3
    door_height = MINIMAP_CELL_SIZE // 3
    
    # Calculate the base draw position based on the node's x and y values and considering the offsets
    draw_x = MINIMAP_PADDING + (node.x - offset_x) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING)
    draw_y = MINIMAP_PADDING + (node.y - offset_y) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING)
    
    if direction == Direction.UP:
        pygame.draw.rect(minimap_surface, (255, 255, 0), (draw_x + (MINIMAP_CELL_SIZE - door_width) // 2, draw_y - door_height, door_width, door_height))
    elif direction == Direction.DOWN:
        pygame.draw.rect(minimap_surface, (255, 255, 0), (draw_x + (MINIMAP_CELL_SIZE - door_width) // 2, draw_y + MINIMAP_CELL_SIZE, door_width, door_height))
    elif direction == Direction.LEFT:
        pygame.draw.rect(minimap_surface, (255, 255, 0), (draw_x - door_width, draw_y + (MINIMAP_CELL_SIZE - door_height) // 2, door_width, door_height))
    elif direction == Direction.RIGHT:
        pygame.draw.rect(minimap_surface, (255, 255, 0), (draw_x + MINIMAP_CELL_SIZE, draw_y + (MINIMAP_CELL_SIZE - door_height) // 2, door_width, door_height))

def update_distance_from_start(start_node):
    """Updates distance_from_start for all nodes using BFS."""
    queue = [(start_node, 0)]  # node and its distance from start
    visited = set()

    while queue:
        current_node, distance = queue.pop(0)
        current_node.distance_from_start = distance
        visited.add(current_node)

        for neighbor in current_node.neighbors.values():
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))

class LevelNode:
    def __init__(self, size, player):
        self.start_node = None
        self.room_nodes = []
        self.player = player

        self.prim_generate_map(size)
        for room in self.room_nodes:
            room.generate_fresh()

    def prim_generate_map(self, size):
        self.start_node = RoomNode(0, 0, self.player)
        self.start_node.is_start = True
        self.room_nodes.append(self.start_node)
        frontier = [(self.start_node, direction) for direction in Direction]

        while len(self.room_nodes) < size and frontier:
            node, direction = random.choice(frontier)
            dx, dy = direction.value
            x, y = node.x + dx, node.y + dy

            existing_node = next((n for n in self.room_nodes if n.x == x and n.y == y), None)

            # If the node does not exist in our list of nodes, create a new one.
            if not existing_node:
                new_node = RoomNode(x, y, self.player)
                self.room_nodes.append(new_node)
                
                new_node.neighbors[direction.opposite()] = node
                node.neighbors[direction] = new_node

                for d in Direction:
                    nx, ny = x + d.value[0], y + d.value[1]
                    if not any(n.x == nx and n.y == ny for n in self.room_nodes):
                        frontier.append((new_node, d))
            # If the node already exists, and there's no connection in the chosen direction, establish the connection.
            else:
                if not node.neighbors.get(direction):
                    node.neighbors[direction] = existing_node
                    existing_node.neighbors[direction.opposite()] = node
            frontier.remove((node, direction))

        update_distance_from_start(self.start_node)
        end_node = max(self.room_nodes, key=lambda node: node.distance_from_start)
        end_node.is_end = True

    def draw_minimap(self, screen, player_node):

        # Calculate the minimum x and y values among all nodes
        min_x, max_x, min_y, max_y = get_min_max_coordinates(self.room_nodes)

        minimap_width = (max_x - min_x + 1) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING) + MINIMAP_PADDING
        minimap_height = (max_y - min_y + 1) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING) + MINIMAP_PADDING

        minimap_surface = pygame.Surface((minimap_width, minimap_height), pygame.SRCALPHA)  # SRCALPHA makes sure the surface supports transparency
        TRANSPARENT_BACKGROUND_COLOR = (50, 50, 50, 128)  # Last value is alpha (0 = fully transparent, 255 = fully opaque)
        minimap_surface.fill(TRANSPARENT_BACKGROUND_COLOR)

        start_node = self.room_nodes[0]  # The starting node is the first node in the list

        for node in self.room_nodes:
            if node and should_draw_node(player_node, node):  # Check if the node exists
                color = (200, 200, 200) if node.visited else (50, 50, 50)
                if node == player_node:
                    color = GREEN
                elif node.is_end:
                    color = PURPLE

                # Calculate draw position based on node's x and y values, adjusted by the min_x and min_y
                draw_x = MINIMAP_PADDING + (node.x - min_x) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING)
                draw_y = MINIMAP_PADDING + (node.y - min_y) * (MINIMAP_CELL_SIZE + MINIMAP_PADDING)

                pygame.draw.rect(minimap_surface, color, (draw_x, draw_y, MINIMAP_CELL_SIZE, MINIMAP_CELL_SIZE))

                # If the current node is the start_node, draw an additional mark
                if node == start_node:
                    # Let's draw a smaller blue rectangle within the starting node as an indicator
                    # Adjust the size and position based on your needs
                    inner_padding = MINIMAP_CELL_SIZE // 4
                    pygame.draw.rect(minimap_surface, (0, 0, 255), 
                                    (draw_x + inner_padding, draw_y + inner_padding, 
                                    MINIMAP_CELL_SIZE - 2 * inner_padding, 
                                    MINIMAP_CELL_SIZE - 2 * inner_padding))
                    
                # Draw doors between rooms
                for direction, neighbor in node.neighbors.items():
                    if neighbor:  # Check if the neighbor exists
                        draw_door(minimap_surface, node, neighbor, direction, min_x, min_y)
                        

        screen.blit(minimap_surface, (MINIMAP_POSITION_X, MINIMAP_POSITION_Y))
class Dungeon:
    def __init__(self, player):
        self.player = player
        self.reset(init=True)

    def reset(self, init=False):
        self.start_level = None
        self.level_nodes = []
        self.num_rooms = START_ROOMS
        self.player.reset(game_over=True, init=init)
        self.generate_level()

    def generate_level(self):
        if not self.level_nodes:
            level_node = LevelNode(self.num_rooms, self.player)
            self.start_level = level_node
        else:
            self.num_rooms += ROOM_GROWTH
            level_node = LevelNode(self.num_rooms, self.player)

        self.level_nodes.append(level_node)
        self.player.set_current_room(self.level_nodes[-1].start_node)
        self.player.add_sprite()
    



        