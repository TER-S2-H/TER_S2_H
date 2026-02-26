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

    # Remove edges between the shape's nodes and their neighbors to integrate the shape into the maze structure
    def place_shape(self):
        for node in self.nodes:
            for neighbor in list(self.maze.graph.neighbors(node)):
                if neighbor not in self.nodes:
                    self.maze.graph.remove_edge(node, neighbor)

            self.maze.close_surrounding_cells(node)
            self.maze.add_wall(node)

class central_block(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for xi in range(5):
            for yi in range(8):
                if xi != 0 or yi != 0:
                    self.nodes.append((x + xi, y + yi))
    
    def satisfies_constraints(self):
        return self.is_available()

class side_block(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for xi in range(5):
            for yi in range(5):
                if xi != 0 or yi != 0:
                    self.nodes.append((x + xi, y + yi))
    
    def satisfies_constraints(self):
        return self.is_available()

# 3x2 horizontal wall
class WallSH(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(2):
            for j in range(3):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
# 3x2 vertical wall
class WallSV(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(3):
            for j in range(2):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
# 2x4 horizontal wall
class WallLH(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(2):
            for j in range(4):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()

# 2x4 vertical wall
class WallLV(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(4):
            for j in range(2):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
# 3x4 horizontal block
class BlockSH(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(3):
            for j in range(4):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
# 3x4 vertical block
class BlockSV(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(4):
            for j in range(3):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()

# 3x5 horizontal block
class BlockLH(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(3):
            for j in range(5):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
# 3x5 vertical block
class BlockLV(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(5):
            for j in range(3):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()

#  L shape down
class L_shapeD(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(3):
            for j in range(2):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
        
        
        for i in range(2):
            for j in range(5):
                self.nodes.append((x + 3 + i, y + j))
    
    def satisfies_constraints(self):
        return self.is_available()
    
#  L shape up
class L_shapeU(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(5):
            for j in range(2):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
        
        
        for i in range(2):
            for j in range(3):
                self.nodes.append((x + i, y + 2 + j))
    
    def satisfies_constraints(self):
        return self.is_available()

# T shape right
class T_shapeR(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(8):
            for j in range(2):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
        
        for i in range(2):
            for j in range(3):
                self.nodes.append((x + 3 + i, y + 2 + j))
    
    def satisfies_constraints(self):
        return self.is_available()

# T shape down
class T_shapeD(Shape):
    def define_nodes(self):
        x, y = self.init_node
        for i in range(2):
            for j in range(8):
                if i != 0 or j != 0:
                    self.nodes.append((x + i, y + j))
        
        for i in range(3):
            for j in range(2):
                self.nodes.append((x + 2 + i, y + 3 + j))
    
    def satisfies_constraints(self):
        return self.is_available()