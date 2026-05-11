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
import unittest.mock

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
        for key in [
            "seed",
            "metadata",
            "nodes",
            "edges",
            "walls",
            "ghost_walls",
            "energizers",
        ]:
            assert key in data, f"Clé manquante dans la réponse : {key}"

    def test_metadata_contient_dimensions(self, client):
        """La metadata doit contenir width et height."""
        response = client.get("/maze")
        data = response.get_json()
        meta = data["metadata"]
        assert "width" in meta
        assert "height" in meta

    def test_metadata_contient_compteur_energizers(self, client):
        """La metadata doit inclure le nombre d'energizers."""
        response = client.get("/maze")
        data = response.get_json()
        meta = data["metadata"]
        assert "energizer_count" in meta
        assert meta["energizer_count"] == len(data["energizers"])

    def test_energizers_attendus(self, client):
        """Le maze contient 2 ou 4 energizers selon les ranges valides."""
        response = client.get("/maze")
        data = response.get_json()
        assert len(data["energizers"]) in {2, 4}

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


# -------------------------------------------------------------------
# Tests sur GET /maze/<seed>
# -------------------------------------------------------------------

_MAZE_DOC_42 = {
    "_id": 42,
    "seed": 42,
    "maze_data": {
        "seed": 42,
        "metadata": {"width": 28, "height": 31, "node_count": 10, "edge_count": 9,
                     "energizer_count": 2},
        "nodes": [{"x": 1, "y": 1, "type": "PATH"}],
        "edges": [{"from": "1,1", "to": "2,1"}],
        "walls": [{"x": 0, "y": 0}],
        "ghost_walls": [{"x": 13, "y": 12}],
        "energizers": [{"x": 1, "y": 3}],
    },
}


class TestGetMazeBySeed:

    def test_status_200_seed_existant(self, client, mock_mongodb):
        """GET /maze/<seed> avec un seed stocké → 200."""
        mock_mongodb.find_one.return_value = _MAZE_DOC_42
        response = client.get("/maze/42")
        assert response.status_code == 200

    def test_contient_maze_data(self, client, mock_mongodb):
        """La réponse doit contenir les données du maze."""
        mock_mongodb.find_one.return_value = _MAZE_DOC_42
        response = client.get("/maze/42")
        data = response.get_json()
        assert "seed" in data
        assert data["seed"] == 42

    def test_seed_inconnu_retourne_404(self, client, mock_mongodb):
        """GET /maze/<seed> avec un seed inexistant → 404."""
        mock_mongodb.find_one.return_value = None
        response = client.get("/maze/999999")
        assert response.status_code == 404

    def test_404_contient_message_erreur(self, client, mock_mongodb):
        """Le message 404 doit être explicite."""
        mock_mongodb.find_one.return_value = None
        response = client.get("/maze/999999")
        data = response.get_json()
        assert "error" in data


# -------------------------------------------------------------------
# Tests sur GET /mazes
# -------------------------------------------------------------------
class TestListMazes:

    def test_status_200(self, client):
        """GET /mazes → 200."""
        response = client.get("/mazes")
        assert response.status_code == 200

    def test_retourne_liste(self, client):
        """GET /mazes doit retourner une liste JSON."""
        response = client.get("/mazes")
        data = response.get_json()
        assert isinstance(data, list)

    def test_filtre_limit(self, client):
        """Le paramètre limit est accepté sans erreur."""
        response = client.get("/mazes?limit=5")
        assert response.status_code == 200

    def test_filtre_rated_only(self, client):
        """Le paramètre rated_only=true est accepté sans erreur."""
        response = client.get("/mazes?rated_only=true")
        assert response.status_code == 200

    def test_filtre_min_score(self, client):
        """Le paramètre min_score=3 est accepté sans erreur."""
        response = client.get("/mazes?min_score=3")
        assert response.status_code == 200

    def test_limit_invalide_retourne_400(self, client):
        """Un paramètre limit non numérique → 400."""
        response = client.get("/mazes?limit=abc")
        assert response.status_code == 400

    def test_min_score_invalide_retourne_400(self, client):
        """Un paramètre min_score non numérique → 400."""
        response = client.get("/mazes?min_score=xyz")
        assert response.status_code == 400


# -------------------------------------------------------------------
# Tests sur PATCH /maze/<seed>/rate
# -------------------------------------------------------------------
class TestRateMaze:

    def test_rating_valide_retourne_200(self, client, mock_mongodb):
        """PATCH /maze/<seed>/rate avec rating=3 → 200."""
        mock_mongodb.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
        response = client.patch("/maze/42/rate",
                                json={"rating": 3},
                                content_type="application/json")
        assert response.status_code == 200

    def test_reponse_contient_seed_et_rating(self, client, mock_mongodb):
        """La réponse 200 doit confirmer le seed et la note."""
        mock_mongodb.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
        response = client.patch("/maze/42/rate",
                                json={"rating": 3},
                                content_type="application/json")
        data = response.get_json()
        assert data["seed"] == 42
        assert data["rating"] == 3

    def test_rating_zero_valide(self, client, mock_mongodb):
        """rating=0 est une valeur limite valide → 200."""
        mock_mongodb.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
        response = client.patch("/maze/42/rate",
                                json={"rating": 0},
                                content_type="application/json")
        assert response.status_code == 200

    def test_rating_cinq_valide(self, client, mock_mongodb):
        """rating=5 est la valeur maximale valide → 200."""
        mock_mongodb.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
        response = client.patch("/maze/42/rate",
                                json={"rating": 5},
                                content_type="application/json")
        assert response.status_code == 200

    def test_rating_trop_eleve_retourne_400(self, client):
        """rating=6 dépasse le maximum → 400."""
        response = client.patch("/maze/42/rate",
                                json={"rating": 6},
                                content_type="application/json")
        assert response.status_code == 400

    def test_rating_negatif_retourne_400(self, client):
        """rating=-1 est négatif → 400."""
        response = client.patch("/maze/42/rate",
                                json={"rating": -1},
                                content_type="application/json")
        assert response.status_code == 400

    def test_rating_manquant_retourne_400(self, client):
        """Corps JSON sans champ rating → 400."""
        response = client.patch("/maze/42/rate",
                                json={},
                                content_type="application/json")
        assert response.status_code == 400

    def test_rating_non_entier_retourne_400(self, client):
        """rating flottant (2.5) → 400 (doit être un entier)."""
        response = client.patch("/maze/42/rate",
                                json={"rating": 2.5},
                                content_type="application/json")
        assert response.status_code == 400

    def test_seed_inconnu_retourne_404(self, client, mock_mongodb):
        """Seed inexistant (matched_count=0) → 404."""
        mock_mongodb.update_one.return_value = unittest.mock.MagicMock(matched_count=0)
        response = client.patch("/maze/999999/rate",
                                json={"rating": 3},
                                content_type="application/json")
        assert response.status_code == 404
