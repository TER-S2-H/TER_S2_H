import networkx as nx

class Maze:
    def __init__(self):
        self.graph = nx.grid_2d_graph(31, 28)

        for node in self.graph.nodes:
            self.graph.nodes[node]["free"] = True

        self.create_block((14, 12), (16, 15))

    def add_wall(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

    def remove_wall(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2)

    def add_walls(self, node1, orientation, length):
        for i in range(length):
            if orientation == 0:  # horizontal
                self.add_wall((node1[0] + i, node1[1]), (node1[0] + i, node1[1] + 1))
            elif orientation == 1:  # vertical
                self.add_wall((node1[0], node1[1] + i), (node1[0] + 1, node1[1] + i))

    def close_cell(self, node):
        self.graph.nodes[node]["free"] = False
    
    def open_cell(self, node):
        self.graph.nodes[node]["free"] = True

    def is_cell_free(self, node):
        return self.graph.nodes[node]["free"]

    def create_block(self, node1, node2):
        for x in range(node1[0], node2[0] + 1):
            self.add_wall((x, node1[1]), (x, node1[1]-1))
            self.add_wall((x, node2[1]), (x, node2[1]+1))
        
        for y in range(node1[1], node2[1] + 1):
            self.add_wall((node1[0], y), (node1[0]-1, y))
            self.add_wall((node2[0], y), (node2[0]+1, y))

        for x in range(node1[0] - 1, node2[0] + 1):
            for y in range(node1[1] - 1, node2[1] + 1):
                self.close_cell((x, y))

        
    
    def display(self):
        for x in range(31):
            for y in range(28):
                cell = "|_"
                if self.graph.has_edge((x, y), (x, y-1)):
                    cell = cell.replace("|", " ")
                if self.graph.has_edge((x, y), (x+1, y)):
                    cell = cell.replace("_", " ")
                print(cell, end="")
            print("|")