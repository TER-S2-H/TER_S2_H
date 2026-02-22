from src.maze import Maze
import src.tetris.shape as sh
import src.tetris.tetris as tetris
import networkx as nx

if __name__ == "__main__":
    maze = Maze()
    tetris.build_tetris_maze(maze)
    maze.display()