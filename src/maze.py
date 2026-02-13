import networkx as nx

class Maze:
    def __init__(self):
        self.graph = nx.grid_2d_graph(31, 28)

        self.surround_with_walls((15, 12))
        self.surround_with_walls((15, 13))
        self.surround_with_walls((15, 14))
        self.surround_with_walls((15, 15))

    def add_wall(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

    def remove_wall(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            self.graph.add_edge(node1, node2)
    
    def surround_with_walls(self, node):
        node_edges = self.graph.edges(node)
        for edge in node_edges:
            self.graph.remove_edge(*edge)