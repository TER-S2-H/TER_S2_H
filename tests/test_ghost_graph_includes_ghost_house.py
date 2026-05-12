"""Ghost topology must include the ghost house and differ from Pac-Man's graph only there."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _undirected_edges(graph):
    edges = set()
    for a, nbrs in graph.items():
        for b in nbrs:
            edges.add(tuple(sorted((a, b))))
    return edges


def test_ghost_graph_includes_ghost_house(maze):
    pac_nodes = set(maze.pacman_graph)
    ghost_nodes = set(maze.ghost_graph)

    assert set(maze.ghost_house) <= ghost_nodes
    assert ghost_nodes - pac_nodes == set(maze.ghost_house)

    pac_edges = _undirected_edges(maze.pacman_graph)
    ghost_edges = _undirected_edges(maze.ghost_graph)
    extra_edges = ghost_edges - pac_edges
    ghost_house_edges = {
        tuple(sorted((a, b)))
        for a, nbrs in maze.ghost_graph.items()
        for b in nbrs
        if a in maze.ghost_house or b in maze.ghost_house
    }
    assert extra_edges == ghost_house_edges
