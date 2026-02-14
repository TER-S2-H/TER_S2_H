import networkx as nx

class Maze:
    def __init__(self):
        self.graph = nx.grid_2d_graph(31, 28)

        self.create_block((15, 12), (15, 15))

    def add_wall(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

    def remove_wall(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2)

    def create_block(self, node1, node2):
        for x in range(node1[0], node2[0] + 1):
            self.add_wall((x, node1[1]), (x, node1[1]-1))
            self.add_wall((x, node2[1]), (x, node2[1]+1))
        
        for y in range(node1[1], node2[1] + 1):
            self.add_wall((node1[0], y), (node1[0]-1, y))
            self.add_wall((node2[0], y), (node2[0]+1, y))

        
    
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