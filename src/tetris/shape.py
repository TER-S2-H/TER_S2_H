from src.maze import Maze
import networkx as nx
from abc import ABC, abstractmethod

class Shape(ABC):
    def __init__(self, node):
        self.init_node = node
        self.nodes = [self.init_node]
        self.define_nodes()

    @abstractmethod
    def define_nodes(self):
        pass
    
    @abstractmethod
    def satisfies_constraints(self, maze):
        pass
    
    @abstractmethod
    def place_shape(self, maze):
        for node in self.nodes:
            maze.open_cell(node)
