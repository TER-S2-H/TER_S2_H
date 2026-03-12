"""
Tests des caractéristiques du maze
====================================
On vérifie les propriétés mathématiques du maze généré :
connectivité, dead ends, structure du graphe, etc.

Pour lancer :
    pytest tests/test_maze_properties.py -v
"""

import pytest
import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


# -------------------------------------------------------------------
# Fixture
# -------------------------------------------------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def maze_data(client):
    """Récupère un maze avec seed fixe pour avoir des tests reproductibles."""
    response = client.get("/maze?seed=42")
    return response.get_json()


# -------------------------------------------------------------------
# Helpers : construction du graphe en mémoire
# -------------------------------------------------------------------
def build_adjacency(maze_data):
    """
    Construit un dictionnaire d'adjacence à partir des edges du JSON.
    Exemple : {"4,23": ["4,24", "5,23"], ...}
    """
    adjacency = {}
    for node in maze_data["nodes"]:
        adjacency[node["id"]] = []

    for edge in maze_data["edges"]:
        a, b = edge["from"], edge["to"]
        if a in adjacency:
            adjacency[a].append(b)
        if b in adjacency:
            adjacency[b].append(a)

    return adjacency


def bfs_connected(adjacency, start):
    """
    Parcours BFS depuis un noeud de départ.
    Retourne l'ensemble de tous les noeuds accessibles.
    """
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    return visited


# -------------------------------------------------------------------
# Tests de structure du JSON
# -------------------------------------------------------------------
class TestStructureJSON:

    def test_nodes_ont_coordonnees(self, maze_data):
        """Chaque noeud doit avoir des coordonnées x et y."""
        for node in maze_data["nodes"]:
            assert "x" in node, f"Noeud sans x : {node}"
            assert "y" in node, f"Noeud sans y : {node}"

    def test_nodes_ont_id(self, maze_data):
        """Chaque noeud doit avoir un identifiant unique."""
        ids = [node["id"] for node in maze_data["nodes"]]
        assert len(ids) == len(set(ids)), "Des noeuds ont des IDs dupliqués !"

    def test_edges_referencent_noeuds_existants(self, maze_data):
        """
        Chaque arête doit référencer des noeuds qui existent dans la liste.
        Un edge vers un noeud inexistant = bug dans le générateur.
        """
        node_ids = {node["id"] for node in maze_data["nodes"]}
        for edge in maze_data["edges"]:
            assert edge["from"] in node_ids, \
                f"Edge 'from' pointe vers un noeud inexistant : {edge['from']}"
            assert edge["to"] in node_ids, \
                f"Edge 'to' pointe vers un noeud inexistant : {edge['to']}"

    def test_walls_ont_coordonnees(self, maze_data):
        """Chaque mur doit avoir des coordonnées x et y."""
        for wall in maze_data["walls"]:
            assert "x" in wall
            assert "y" in wall

    def test_coordonnees_dans_les_bornes(self, maze_data):
        """
        Toutes les coordonnées des noeuds doivent être dans les limites
        du maze (0 <= x < width, 0 <= y < height).
        """
        width = maze_data["metadata"]["width"]
        height = maze_data["metadata"]["height"]
        for node in maze_data["nodes"]:
            assert 0 <= node["x"] < width, \
                f"Noeud hors limites en x : {node}"
            assert 0 <= node["y"] < height, \
                f"Noeud hors limites en y : {node}"


# -------------------------------------------------------------------
# Tests de connectivité
# -------------------------------------------------------------------
class TestConnectivite:

    def test_maze_est_connexe(self, maze_data):
        """
        Tous les couloirs doivent être accessibles depuis n'importe quel point.
        Un maze avec des zones isolées serait injouable.
        
        On fait un BFS depuis le premier noeud et on vérifie qu'on
        atteint tous les autres.
        """
        adjacency = build_adjacency(maze_data)
        nodes = list(adjacency.keys())

        if not nodes:
            pytest.skip("Aucun noeud dans le maze")

        start = nodes[0]
        visited = bfs_connected(adjacency, start)

        assert len(visited) == len(nodes), (
            f"Le maze n'est pas connexe : "
            f"{len(visited)} noeuds accessibles sur {len(nodes)} au total"
        )

    def test_chaque_noeud_a_au_moins_un_voisin(self, maze_data):
        """
        Aucun noeud ne doit être complètement isolé (0 voisins).
        """
        adjacency = build_adjacency(maze_data)
        for node_id, neighbors in adjacency.items():
            assert len(neighbors) > 0, \
                f"Noeud isolé (aucun voisin) : {node_id}"


# -------------------------------------------------------------------
# Tests des dead ends (impasses)
# -------------------------------------------------------------------
class TestDeadEnds:

    def test_dead_ends_existent(self, maze_data):
        """
        Un maze Pac-Man doit avoir des impasses (noeuds avec 1 seul voisin).
        S'il n'y en a aucune, le maze serait trop ouvert.
        """
        adjacency = build_adjacency(maze_data)
        dead_ends = [n for n, voisins in adjacency.items() if len(voisins) == 1]
        assert len(dead_ends) > 0, "Le maze ne contient aucune impasse !"

    def test_dead_ends_raisonnables(self, maze_data):
        """
        Il ne doit pas y avoir trop d'impasses non plus — sinon le maze
        serait un simple arbre et pas un labyrinthe Pac-Man.
        On fixe arbitrairement une limite à 30% des noeuds.
        """
        adjacency = build_adjacency(maze_data)
        total = len(adjacency)
        dead_ends = [n for n, v in adjacency.items() if len(v) == 1]

        ratio = len(dead_ends) / total if total > 0 else 0
        assert ratio < 0.30, (
            f"Trop d'impasses : {len(dead_ends)}/{total} "
            f"({ratio:.0%}) — maze trop linéaire"
        )

    def test_compte_dead_ends(self, maze_data):
        """
        Affiche le nombre d'impasses — informatif pour la présentation.
        Ce test ne peut pas échouer, il documente juste la valeur.
        """
        adjacency = build_adjacency(maze_data)
        dead_ends = [n for n, v in adjacency.items() if len(v) == 1]
        print(f"\n  → Dead ends : {len(dead_ends)} sur {len(adjacency)} noeuds")
        # Ce test est toujours vrai, son but est l'affichage avec pytest -v -s
        assert True


# -------------------------------------------------------------------
# Tests des intersections
# -------------------------------------------------------------------
class TestIntersections:

    def test_intersections_existent(self, maze_data):
        """
        Un maze Pac-Man doit avoir des intersections (noeuds avec 3+ voisins).
        Sans intersections, il n'y a pas de choix de chemin possible.
        """
        adjacency = build_adjacency(maze_data)
        intersections = [n for n, v in adjacency.items() if len(v) >= 3]
        assert len(intersections) > 0, "Aucune intersection dans le maze !"
