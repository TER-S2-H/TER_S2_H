#!/usr/bin/env python3
from flask import Flask, jsonify, request
from src.pacman_map import PacmanMap
from src.maze_complexity import compute_complexity_score
import random

app = Flask(__name__)

@app.route("/maze", methods=["GET"])
def generate_maze():
    seed_param = request.args.get("seed")
    if seed_param is not None:
        try:
            seed = int(seed_param)
        except ValueError:
            return jsonify({"error": "Le paramètre 'seed' doit être un entier."}), 400
    else:
        seed = random.randint(0, 999_999)

    try:
        maze = PacmanMap(seed=seed)
        maze.generate()
        tile_map = maze.get_tile_map()
        maze_json = maze.export_to_graph_json(tile_map=tile_map)

        # Ajout du complexity_score dans metadata
        score_data = compute_complexity_score(
            maze_data=maze_json,
            start=(13, 23),
            wrap_tunnels=True,
        )
        maze_json["metadata"]["complexity_score"] = score_data

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération : {str(e)}"}), 500

    response = {"seed": seed, **maze_json}
    return jsonify(response), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Pac-Man Maze Generator API",
        "version": "1.0",
        "endpoints": {
            "GET /maze": {
                "description": "Génère un maze Pac-Man",
                "params": {"seed (optionnel)": "Entier — graine pour la génération reproductible"},
                "example": "/maze?seed=42"
            },
            "GET /health": {"description": "Vérification que le service est en ligne"}
        }
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)