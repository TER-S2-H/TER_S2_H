#!/usr/bin/env python3
"""
API Flask — Générateur de maze Pac-Man
Expose le générateur via HTTP pour pouvoir l'appeler depuis n'importe où.
"""

from flask import Flask, jsonify, request
from src.pacman_map import PacmanMap
import random

# -------------------------------------------------------------------
# Création de l'application Flask
# -------------------------------------------------------------------
app = Flask(__name__)


# -------------------------------------------------------------------
# Route principale : GET /maze
# -------------------------------------------------------------------
@app.route("/maze", methods=["GET"])
def generate_maze():
    """
    Génère un maze et retourne sa représentation en JSON.

    Paramètres de requête (tous optionnels) :
      - seed  (int)  : graine pour la génération (reproductible)
                       Si absent, une graine aléatoire est utilisée.

    Exemple d'appel :
      GET /maze
      GET /maze?seed=42

    Réponse JSON :
      {
        "seed": 42,
        "metadata": { "width": ..., "height": ..., ... },
        "nodes": [...],
        "edges": [...],
        "walls": [...],
        "ghost_walls": [...]
      }
    """
    # --- Lecture du paramètre seed ---
    seed_param = request.args.get("seed")

    if seed_param is not None:
        try:
            seed = int(seed_param)
        except ValueError:
            # Si le seed n'est pas un entier valide, on renvoie une erreur claire
            return jsonify({"error": "Le paramètre 'seed' doit être un entier."}), 400
    else:
        seed = random.randint(0, 999_999)

    # --- Génération du maze ---
    try:
        maze = PacmanMap(seed=seed)
        maze.generate()
        tile_map = maze.get_tile_map()
        maze_json = maze.export_to_graph_json(tile_map=tile_map)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération : {str(e)}"}), 500

    # --- Construction de la réponse ---
    # On ajoute le seed utilisé dans la réponse pour que le client
    # puisse reproduire exactement le même maze plus tard.
    response = {"seed": seed, **maze_json}

    return jsonify(response), 200


# -------------------------------------------------------------------
# Route de santé : GET /health
# -------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """
    Vérifie que le serveur tourne correctement.
    Utile pour Render et pour les tests de bout-en-bout (end-to-end).

    Exemple d'appel :
      GET /health

    Réponse JSON :
      { "status": "ok" }
    """
    return jsonify({"statut": "ok"}), 200


# -------------------------------------------------------------------
# Route d'accueil : GET /
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Page d'accueil avec la documentation de l'API.
    """
    return jsonify({
        "service": "Pac-Man Maze Generator API",
        "version": "1.0",
        "endpoints": {
            "GET /maze": {
                "description": "Génère un maze Pac-Man",
                "params": {
                    "seed (optionnel)": "Entier — graine pour la génération reproductible"
                },
                "example": "/maze?seed=42"
            },
            "GET /health": {
                "description": "Vérification que le service est en ligne"
            }
        }
    }), 200


# -------------------------------------------------------------------
# Lancement du serveur (uniquement si on exécute ce fichier directement)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True : recharge automatiquement le serveur quand tu modifies le code
    # host="0.0.0.0" : accepte les connexions de toutes les interfaces réseau
    #                   (nécessaire pour Render et Docker)
    # port=5000 : port par défaut de Flask
    app.run(debug=True, host="0.0.0.0", port=5000)