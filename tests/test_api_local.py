"""
Tests fonctionnels — API Flask en local
========================================
Ces tests utilisent le client de test Flask : pas besoin de lancer
le serveur. Flask simule les requêtes HTTP directement en mémoire.

Pour lancer :
    pytest tests/test_api_local.py -v
"""

import json
import pytest
import sys
import os

# Permet d'importer app.py depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


# -------------------------------------------------------------------
# Fixture : le client de test Flask
# -------------------------------------------------------------------
@pytest.fixture
def client():
    """
    Crée un client de test Flask.
    app.config["TESTING"] = True désactive la gestion d'erreurs interne
    de Flask pour que les exceptions remontent jusqu'aux tests.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# -------------------------------------------------------------------
# Tests sur GET /
# -------------------------------------------------------------------
class TestIndex:

    def test_status_200(self, client):
        """La route / doit répondre avec un code 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_contient_endpoints(self, client):
        """La route / doit lister les endpoints disponibles."""
        response = client.get("/")
        data = response.get_json()
        assert "endpoints" in data

    def test_contient_route_maze(self, client):
        """La documentation doit mentionner la route /maze."""
        response = client.get("/")
        data = response.get_json()
        assert "GET /maze" in data["endpoints"]


# -------------------------------------------------------------------
# Tests sur GET /health
# -------------------------------------------------------------------
class TestHealth:

    def test_status_200(self, client):
        """Le health check doit répondre avec un code 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok(self, client):
        """Le health check doit retourner {"status": "ok"}."""
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "ok"


# -------------------------------------------------------------------
# Tests sur GET /maze
# -------------------------------------------------------------------
class TestMaze:

    def test_status_200(self, client):
        """GET /maze doit répondre avec un code 200."""
        response = client.get("/maze")
        assert response.status_code == 200

    def test_contient_cles_obligatoires(self, client):
        """La réponse doit contenir toutes les clés attendues."""
        response = client.get("/maze")
        data = response.get_json()
        for key in ["seed", "metadata", "nodes", "edges", "walls", "ghost_walls"]:
            assert key in data, f"Clé manquante dans la réponse : {key}"

    def test_metadata_contient_dimensions(self, client):
        """La metadata doit contenir width et height."""
        response = client.get("/maze")
        data = response.get_json()
        meta = data["metadata"]
        assert "width" in meta
        assert "height" in meta

    def test_dimensions_pac_man(self, client):
        response = client.get("/maze")
        data = response.get_json()
        meta = data["metadata"]
        assert meta["width"] == 28, f"Largeur attendue 28, obtenu {meta['width']}"
        assert meta["height"] == 31, f"Hauteur attendue 31, obtenu {meta['height']}"

    def test_seed_present_dans_reponse(self, client):
        """La réponse doit inclure le seed utilisé pour générer le maze."""
        response = client.get("/maze")
        data = response.get_json()
        assert "seed" in data
        assert isinstance(data["seed"], int)

    def test_seed_fixe_reproductible(self, client):
        """
        Avec le même seed, on doit obtenir exactement le même maze.
                        """
        response1 = client.get("/maze?seed=42")
        response2 = client.get("/maze?seed=42")
        data1 = response1.get_json()
        data2 = response2.get_json()

        # Les deux mazes doivent avoir le même nombre de noeuds et d'arêtes
        assert data1["metadata"]["node_count"] == data2["metadata"]["node_count"]
        assert data1["metadata"]["edge_count"] == data2["metadata"]["edge_count"]

    def test_seeds_differents_donnent_mazes_differents(self, client):
        """
        Deux seeds différents doivent produire des mazes différents
        (au moins une différence dans les arêtes).
        """
        response1 = client.get("/maze?seed=1")
        response2 = client.get("/maze?seed=9999")
        data1 = response1.get_json()
        data2 = response2.get_json()

        # Les edges ne doivent pas être identiques
        edges1 = sorted([f"{e['from']}-{e['to']}" for e in data1["edges"]])
        edges2 = sorted([f"{e['from']}-{e['to']}" for e in data2["edges"]])
        assert edges1 != edges2

    def test_nodes_non_vide(self, client):
        """Le maze doit avoir des noeuds (couloirs)."""
        response = client.get("/maze")
        data = response.get_json()
        assert len(data["nodes"]) > 0

    def test_edges_non_vide(self, client):
        """Le maze doit avoir des arêtes (connexions entre couloirs)."""
        response = client.get("/maze")
        data = response.get_json()
        assert len(data["edges"]) > 0

    def test_walls_non_vide(self, client):
        """Le maze doit avoir des murs."""
        response = client.get("/maze")
        data = response.get_json()
        assert len(data["walls"]) > 0

    def test_seed_invalide_retourne_400(self, client):
        """Un seed non entier doit retourner une erreur 400."""
        response = client.get("/maze?seed=abc")
        assert response.status_code == 400

    def test_seed_invalide_retourne_message_erreur(self, client):
        """Le message d'erreur doit être explicite."""
        response = client.get("/maze?seed=abc")
        data = response.get_json()
        assert "error" in data
