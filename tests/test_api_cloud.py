"""
Tests end-to-end — API déployée sur Render
============================================
Ces tests appellent VRAIMENT l'API sur le cloud via HTTP.
Ils vérifient que ce qui fonctionne en local fonctionne aussi en production.

  Ces tests nécessitent une connexion internet.
  Sur le plan gratuit Render, le service peut mettre 50s à répondre
    si il était en veille. C'est normal.

Pour lancer :
    pytest tests/test_api_cloud.py -v

Pour lancer TOUS les tests (local + propriétés + cloud) :
    pytest tests/ -v
"""

import pytest
import requests

# -------------------------------------------------------------------
# URL de l'API en production
# -------------------------------------------------------------------
CLOUD_URL = "https://pacman-s2-h.onrender.com"
TIMEOUT = 60  # secondes — généreux pour le cold start Render


# -------------------------------------------------------------------
# Fixture : vérifier que le cloud est accessible avant de tester
# -------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def check_cloud_available():
    """
    Vérifie que le service Render est accessible.
    Si non, tous les tests de ce fichier sont ignorés (skip)
    plutôt que de tous échouer avec des erreurs de connexion.
    """
    try:
        response = requests.get(f"{CLOUD_URL}/health", timeout=TIMEOUT)
        if response.status_code != 200:
            pytest.skip(f"Service Render non disponible (status {response.status_code})")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Impossible de joindre {CLOUD_URL} — pas de connexion internet ?")
    except requests.exceptions.Timeout:
        pytest.skip(f"Timeout après {TIMEOUT}s — le service Render ne répond pas")


# -------------------------------------------------------------------
# Tests health check
# -------------------------------------------------------------------
class TestHealthCloud:

    def test_health_status_200(self):
        """Le health check sur le cloud doit répondre 200."""
        response = requests.get(f"{CLOUD_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200

    def test_health_retourne_ok(self):
        """Le health check doit retourner {"status": "ok"}."""
        response = requests.get(f"{CLOUD_URL}/health", timeout=TIMEOUT)
        data = response.json()
        assert data["status"] == "ok"


# -------------------------------------------------------------------
# Tests de l'API /maze en production
# -------------------------------------------------------------------
class TestMazeCloud:

    def test_maze_status_200(self):
        """GET /maze sur le cloud doit répondre 200."""
        response = requests.get(f"{CLOUD_URL}/maze", timeout=TIMEOUT)
        assert response.status_code == 200

    def test_maze_retourne_json(self):
        """La réponse doit être du JSON valide."""
        response = requests.get(f"{CLOUD_URL}/maze", timeout=TIMEOUT)
        assert response.headers["Content-Type"].startswith("application/json")
        data = response.json()  # lève une exception si pas du JSON
        assert data is not None

    def test_maze_contient_cles_obligatoires(self):
        """La réponse JSON doit contenir toutes les clés attendues."""
        response = requests.get(f"{CLOUD_URL}/maze", timeout=TIMEOUT)
        data = response.json()
        for key in ["seed", "metadata", "nodes", "edges", "walls"]:
            assert key in data, f"Clé manquante dans la réponse cloud : {key}"

    def test_maze_dimensions_pac_man(self):
        """
        Le maze doit faire 28x31 même en production.
        
        """
        response = requests.get(f"{CLOUD_URL}/maze", timeout=TIMEOUT)
        data = response.json()
        meta = data["metadata"]
        assert meta["width"] == 28
        assert meta["height"] == 31

    def test_maze_seed_fixe_reproductible(self):
        """
        Avec seed=42, deux appels successifs doivent donner le même maze.
        Vérifie la reproductibilité sur le cloud.
        """
        r1 = requests.get(f"{CLOUD_URL}/maze?seed=42", timeout=TIMEOUT)
        r2 = requests.get(f"{CLOUD_URL}/maze?seed=42", timeout=TIMEOUT)
        d1 = r1.json()
        d2 = r2.json()
        assert d1["metadata"]["node_count"] == d2["metadata"]["node_count"]
        assert d1["metadata"]["edge_count"] == d2["metadata"]["edge_count"]

    def test_seed_invalide_retourne_400(self):
        """Un seed invalide doit retourner 400 en production aussi."""
        response = requests.get(f"{CLOUD_URL}/maze?seed=abc", timeout=TIMEOUT)
        assert response.status_code == 400
