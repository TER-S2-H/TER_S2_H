from src.maze import Maze
from src.tetris.shape import Shape, WallSH
import networkx as nx

if __name__ == "__main__":
    maze = Maze()
    shape = WallSH((5, 5), maze)
    print(shape.is_available())
    shape.place_shape()
    maze.display()