from src.maze import Maze
import src.tetris.shape as sh
import networkx as nx
import random

# Define the outter cells as closed to create a pathway around the maze
def clear_static_pathways(maze):
    for x in range(31):
        for y in range(28):
            if x == 0 or x == 15 or y == 0:
                maze.close_cell((x, y))

def clear_static_blocks(maze):
    # central block
    shape = sh.central_block((14, 10), maze)
    shape.place_shape()

    # left blocks
    shape = sh.side_block((12, 0), maze)
    shape.place_shape()

    shape = sh.side_block((16, 0), maze)
    shape.place_shape()

    # right blocks
    shape = sh.side_block((12, 21), maze)
    shape.place_shape()

    shape = sh.side_block((16, 21), maze)
    shape.place_shape()

# Create a random shape at the given node if it satisfies the constraints of being placed in the maze
def create_random_shape(maze, node):
    shapes = list(sh.Shape.__subclasses__())
    shapes = random.sample(shapes, len(shapes))
    for shape_cls in shapes:
        shape = shape_cls(node, maze)
        if shape.satisfies_constraints() and shape.__class__.__name__ != "central_block" and shape.__class__.__name__ != "side_block":
            return shape
    return None

def static_setup(maze):
    clear_static_pathways(maze)
    clear_static_blocks(maze)

    maze.graph.add_edge((15, 0), (15, 27))

def build_tetris_maze(maze):
    static_setup(maze)

    for y in range(28):
        for x in range(31):
            if maze.is_cell_free((x, y)):
                shape = create_random_shape(maze, (x, y))
                if shape:
                    shape.place_shape()