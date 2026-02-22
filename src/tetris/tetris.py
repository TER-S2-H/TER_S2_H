from src.maze import Maze
import src.tetris.shape as sh
import networkx as nx
import random

# Define the outter cells as closed to create a pathway around the maze
def clear_outter_pathway(maze):
    for x in range(31):
        for y in range(28):
            if x == 0 or y == 0:
                maze.close_cell((x, y))

def clear_central_block(maze):
    shape = sh.central_block((14, 12), maze)
    shape.place_shape()

# Create a random shape at the given node if it satisfies the constraints of being placed in the maze
def create_random_shape(maze, node):
    shapes = list(sh.Shape.__subclasses__())
    shapes = random.sample(shapes, len(shapes))
    for shape_cls in shapes:
        shape = shape_cls(node, maze)
        if shape.satisfies_constraints() and shape.__class__.__name__ != "central_block":
            return shape
    return None

def build_tetris_maze(maze):
    clear_outter_pathway(maze)

    clear_central_block(maze)

    for x in range(31):
        for y in range(28):
            if maze.is_cell_free((x, y)):
                shape = create_random_shape(maze, (x, y))
                if shape:
                    shape.place_shape()