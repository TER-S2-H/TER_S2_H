# conftest.py — Configuration pytest
# Ce fichier est automatiquement lu par pytest.
# Il permet à pytest de trouver les modules depuis la racine du projet.
import sys
import os
import unittest.mock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True, scope="session")
def mock_mongodb():
    """
    Mock de la connexion MongoDB pour tous les tests.

    Sans ce mock, les tests échouent en CI (pas de MONGODB_URI) et en
    développement sur réseau universitaire (port 27017 bloqué).

    On patche app.get_collection (la référence importée dans app.py via
    `from src.database.db import get_collection`) pour court-circuiter
    toute tentative de connexion réseau.
    """
    mock_col = unittest.mock.MagicMock()
    # update_one : utilisé dans /maze (upsert) et /maze/<seed>/rate
    mock_col.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
    with unittest.mock.patch("app.get_collection", return_value=mock_col):
        yield mock_col
