# conftest.py — Configuration pytest
# Ce fichier est automatiquement lu par pytest.
# Il permet à pytest de trouver les modules depuis la racine du projet.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
