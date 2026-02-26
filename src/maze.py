import networkx as nx
from rich import print

class Maze:
    def __init__(self):
        self.graph = nx.grid_2d_graph(31, 28)

        for node in self.graph.nodes:
            self.graph.nodes[node]["free"] = True
            self.graph.nodes[node]["wall"] = False

    def add_wall(self, node):
        self.graph.nodes[node]["wall"] = True

    def remove_wall(self, node):
        self.graph.nodes[node]["wall"] = False
    
    def is_wall(self, node):
        if node not in self.graph.nodes:
            return False
        return self.graph.nodes[node]["wall"]

    def add_walls(self, node1, orientation, length):
        for i in range(length):
            if orientation == 0:  # horizontal
                self.add_wall((node1[0] + i, node1[1]), (node1[0] + i, node1[1] + 1))
            elif orientation == 1:  # vertical
                self.add_wall((node1[0], node1[1] + i), (node1[0] + 1, node1[1] + i))

    def close_cell(self, node):
        if node in self.graph.nodes:
            self.graph.nodes[node]["free"] = False
    
    def open_cell(self, node):
        if node in self.graph.nodes:
            self.graph.nodes[node]["free"] = True

    def is_cell_free(self, node):
        if node not in self.graph.nodes:
            return False
        return self.graph.nodes[node]["free"]
    
    def close_surrounding_cells(self, node):
        self.close_cell(node)
        self.close_cell((node[0], node[1] + 1))
        self.close_cell((node[0] + 1, node[1]))
        self.close_cell((node[0] + 1, node[1] + 1))

    def create_block(self, node1, node2):
        for x in range(node1[0], node2[0] + 1):
            self.add_wall((x, node1[1]), (x, node1[1]-1))
            self.add_wall((x, node2[1]), (x, node2[1]+1))
        
        for y in range(node1[1], node2[1] + 1):
            self.add_wall((node1[0], y), (node1[0]-1, y))
            self.add_wall((node2[0], y), (node2[0]+1, y))

        for x in range(node1[0], node2[0] + 1):
            for y in range(node1[1], node2[1] + 1):
                self.close_surrounding_cells((x, y))

        
    
    def display(self):
        for x in range(31):
            for y in range(28):
                if self.is_wall((x, y)):
                    print("[bright_blue]██[/bright_blue]", end="")

                else:
                    print("[black]██[/black]", end="")
            print()