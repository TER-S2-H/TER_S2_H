#!/usr/bin/env python3
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
from src.pacman_map import PacmanMap
from src.maze_complexity import compute_complexity_score
from src.database.db import get_collection
from datetime import datetime, timezone
from pymongo.errors import PyMongoError
import random
import logging

app = Flask(__name__)

# ---------------------------------------------------------------------------
# GET /maze  — génération + upsert MongoDB
# ---------------------------------------------------------------------------

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

        score_data = compute_complexity_score(
            maze_data=maze_json,
            start=(13, 23),
            wrap_tunnels=True,
        )
        maze_json["metadata"]["complexity_score"] = score_data

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération : {str(e)}"}), 500

    response = {"seed": seed, **maze_json}

    # Upsert MongoDB — ne jamais écraser un rating existant
    try:
        col = get_collection()
        col.update_one(
            {"_id": seed},
            {
                "$setOnInsert": {
                    "_id": seed,
                    "seed": seed,
                    "maze_data": response,
                    "complexity_score": maze_json["metadata"]["complexity_score"]["score"],
                    "created_at": datetime.now(timezone.utc),
                    "rating": None,
                    "rated_at": None,
                }
            },
            upsert=True,
        )
    except PyMongoError as e:
        logging.error("MongoDB upsert failed for seed %s: %s", seed, e)

    return jsonify(response), 200


# ---------------------------------------------------------------------------
# PATCH /maze/<seed>/rate  — noter un maze
# ---------------------------------------------------------------------------

@app.route("/maze/<int:seed>/rate", methods=["PATCH"])
def rate_maze(seed):
    body = request.get_json(silent=True) or {}
    rating = body.get("rating")

    if rating is None or not isinstance(rating, int) or not (0 <= rating <= 5):
        return jsonify({"error": "Le champ 'rating' doit être un entier entre 0 et 5."}), 400

    try:
        col = get_collection()
        result = col.update_one(
            {"_id": seed},
            {
                "$set": {
                    "rating": rating,
                    "rated_at": datetime.now(timezone.utc),
                }
            },
        )
    except PyMongoError as e:
        logging.error("MongoDB rate failed for seed %s: %s", seed, e)
        return jsonify({"error": "Service de base de données indisponible."}), 503

    if result.matched_count == 0:
        return jsonify({"error": f"Aucun maze trouvé pour le seed {seed}."}), 404

    return jsonify({"seed": seed, "rating": rating, "message": "rated successfully"}), 200


# ---------------------------------------------------------------------------
# GET /mazes  — liste paginée (sans maze_data)
# ---------------------------------------------------------------------------

@app.route("/mazes", methods=["GET"])
def list_mazes():
    try:
        limit = int(request.args.get("limit", 20))
        rated_only = request.args.get("rated_only", "false").lower() == "true"
        min_score = float(request.args.get("min_score", 0))
    except ValueError:
        return jsonify({"error": "Paramètres invalides."}), 400

    query = {"complexity_score": {"$gte": min_score}}
    if rated_only:
        query["rating"] = {"$ne": None}

    try:
        col = get_collection()
        cursor = (
            col.find(query, {"maze_data": 0})
               .sort("created_at", -1)
               .limit(limit)
        )
        mazes = []
        for doc in cursor:
            created_at = doc.get("created_at")
            mazes.append({
                "seed": doc["seed"],
                "complexity_score": doc.get("complexity_score"),
                "rating": doc.get("rating"),
                "created_at": created_at.isoformat() if created_at else None,
            })
    except PyMongoError as e:
        logging.error("MongoDB list_mazes failed: %s", e)
        return jsonify({"error": "Service de base de données indisponible."}), 503

    return jsonify(mazes), 200


# ---------------------------------------------------------------------------
# GET /maze/<seed>  — récupérer un maze stocké
# ---------------------------------------------------------------------------

@app.route("/maze/<int:seed>", methods=["GET"])
def get_maze(seed):
    try:
        col = get_collection()
        doc = col.find_one({"_id": seed})
    except PyMongoError as e:
        logging.error("MongoDB get_maze failed for seed %s: %s", seed, e)
        return jsonify({"error": "Service de base de données indisponible."}), 503

    if doc is None:
        return jsonify({"error": f"Aucun maze trouvé pour le seed {seed}."}), 404

    return jsonify(doc["maze_data"]), 200


# ---------------------------------------------------------------------------
# Routes utilitaires
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Pac-Man Maze Generator API",
        "version": "2.0",
        "endpoints": {
            "GET /maze": {
                "description": "Génère (et stocke) un maze Pac-Man",
                "params": {"seed (optionnel)": "Entier — graine reproductible"},
                "example": "/maze?seed=42"
            },
            "GET /maze/<seed>": {
                "description": "Récupère un maze stocké depuis la base"
            },
            "PATCH /maze/<seed>/rate": {
                "description": "Note un maze (0-5)",
                "body": {"rating": "entier 0-5"}
            },
            "GET /mazes": {
                "description": "Liste des mazes stockés",
                "params": {
                    "limit": "nombre max (défaut 20)",
                    "rated_only": "true/false",
                    "min_score": "score minimum de complexité"
                }
            },
            "GET /health": {"description": "Vérification que le service est en ligne"}
        }
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
