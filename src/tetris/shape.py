from src.maze import Maze
import networkx as nx
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, node, maze):
        self.maze = maze
        self.init_node = node
        self.nodes = [self.init_node]
        self.define_nodes()

    @abstractmethod
    def define_nodes(self):
        pass
    
    # Check if the shape can be placed at the initial node without overlapping with existing walls or pathways
    def is_available(self):
        for node in self.nodes:
            if not self.maze.is_cell_free(node):
                return False
            elif not self.maze.is_cell_free((node[0], node[1] + 1)):
                return False
            elif not self.maze.is_cell_free((node[0] + 1, node[1])):
                return False
            elif not self.maze.is_cell_free((node[0] + 1, node[1] + 1)):
                return False
        return True
    
    @abstractmethod
    def satisfies_constraints(self):
        pass

    # Mark the cells occupied by the shape as walls in the maze and clear pathways around it
    def close_surrounding_cells(self, node):
        self.maze.close_cell(node)
        self.maze.close_cell((node[0], node[1] + 1))
        self.maze.close_cell((node[0] + 1, node[1]))
        self.maze.close_cell((node[0] + 1, node[1] + 1))

    # Remove edges between the shape's nodes and their neighbors to integrate the shape into the maze structure
    def place_shape(self):
        for node in self.nodes:
            for neighbor in list(self.maze.graph.neighbors(node)):
                if neighbor not in self.nodes:
                    self.maze.graph.remove_edge(node, neighbor)

            self.close_surrounding_cells(node)




# Two cell horizontal wall
class WallSH(Shape):
    def define_nodes(self):
        x, y = self.init_node
        self.nodes.append((x, y + 1))
        self.nodes.append((x, y + 2))
    
    def satisfies_constraints(self, maze):
        return self.is_available()
    
# Two cell vertical wall
class WallSV(Shape):
    def define_nodes(self):
        x, y = self.init_node
        self.nodes.append((x + 1, y))
        self.nodes.append((x + 2, y))
    
    def satisfies_constraints(self, maze):
        return self.is_available()