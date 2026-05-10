# TER S2 — Pac-Man Maze Generator — Groupe H

Générateur de maze Pac-Man exposé en tant que web service HTTP, déployé sur Render,
avec persistence MongoDB Atlas, jeu Pac-Man jouable et visualiseur de labyrinthe.

**URL du service :** https://pacman-s2-h.onrender.com

---

## Installation

Python 3.6+ requis.

```bash
# Dépendances serveur (API Flask + générateur + base de données)
pip install -r requirements.txt

# Dépendances client local (visualisation Pygame + jeu)
pip install -r requirements-client.txt
```

> `requirements.txt` est utilisé par Render pour le déploiement — il ne contient
> pas pygame qui n'est utile qu'en local. C'est pourquoi il y a deux fichiers séparés.

### Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/pacman_ter?appName=...
```

Sur Render : ajouter `MONGODB_URI` dans **Dashboard → Environment**.

---

## Structure du projet

```
├── app.py                          # API Flask (serveur web)
├── client.py                       # Visualisation Pygame simple (client local)
├── generate.py                     # Générateur en ligne de commande (wrapper)
├── maze_complexity.py              # Score de complexité (wrapper)
├── maze_complexity_visualize.py    # Visualisation du chemin (wrapper)
├── render_json.py                  # Rendu depuis JSON (wrapper)
├── Procfile                        # Configuration Render
├── requirements.txt                # Dépendances serveur (Render)
├── requirements-client.txt         # Dépendances client local
├── .env                            # Variables d'environnement locales (non versionné)
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD GitHub Actions
├── src/
│   ├── pacman_map.py               # Logique de génération du maze
│   ├── maze_renderer.py            # Rendu PNG
│   ├── maze_complexity.py          # Calcul du score de complexité
│   ├── maze_complexity_visualize.py
│   ├── maze_viewer.py              # Visualiseur pygame (sans personnages)
│   ├── maze_seed_search.py         # Recherche de seeds par critères
│   ├── pacman_simulator.py         # Simulateur headless de parties
│   ├── read_sessions.py            # Lecture des sessions enregistrées
│   ├── visualize_session.py        # Visualisation d'une session rejouée
│   ├── generate.py
│   ├── render_json.py
│   ├── database/
│   │   └── db.py                   # Connexion MongoDB Atlas (get_collection)
│   └── game/
│       ├── game.py                 # Boucle principale du jeu (Pac-Man jouable)
│       ├── environment.py          # État du plateau (grille, pellets, collisions)
│       ├── pacman_agent.py         # Agent Pac-Man (clavier)
│       ├── ghost.py                # IA des 4 fantômes
│       ├── pellet.py               # Modèle des pellets
│       ├── recorder.py             # Enregistrement de parties en JSON
│       └── agent.py                # Types partagés (Direction, vecteurs)
└── tests/
    ├── test_api_local.py           # Tests fonctionnels (Flask test client)
    ├── test_maze_properties.py     # Tests caractéristiques du maze
    ├── test_api_cloud.py           # Tests end-to-end (Render)
    ├── curl_examples.sh            # Exemples de commandes curl
    └── conftest.py                 # Configuration pytest
```

---

## Web Service (API HTTP)

### Lancer le serveur en local

```bash
python app.py
```

Le serveur tourne sur http://localhost:5000

### Routes disponibles

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Documentation de l'API |
| `/health` | GET | Vérification que le service est en ligne |
| `/maze` | GET | Génère et stocke un maze (seed aléatoire) |
| `/maze?seed=42` | GET | Génère et stocke un maze reproductible |
| `/maze/<seed>` | GET | Récupère un maze stocké en base |
| `/maze/<seed>/rate` | PATCH | Note un maze (0 à 5) |
| `/mazes` | GET | Liste des mazes stockés |

### Format de réponse `/maze`

```json
{
  "seed": 42,
  "metadata": {
    "width": 28,
    "height": 31,
    "node_count": 326,
    "edge_count": 347,
    "complexity_score": {
      "score": 2.89,
      "steps": 403,
      "longest_trap": 13,
      "avg_trap_len": 10.32,
      "num_traps": 19,
      "num_tunnels": 1
    }
  },
  "nodes": [{"x": 1, "y": 1, "type": "PATH"}, "..."],
  "edges": [{"from": "1,1", "to": "2,1"}, "..."],
  "walls": [{"x": 0, "y": 0}, "..."],
  "ghost_walls": [{"x": 13, "y": 12}, "..."],
  "energizers": [{"x": 26, "y": 5}, {"x": 26, "y": 25}]
}
```

### Détail des nouvelles routes

**`GET /maze/<seed>`** — Récupère un maze déjà généré depuis MongoDB :
```bash
curl https://pacman-s2-h.onrender.com/maze/42
```

**`PATCH /maze/<seed>/rate`** — Note un maze entre 0 et 5 :
```bash
curl -X PATCH https://pacman-s2-h.onrender.com/maze/42/rate \
     -H "Content-Type: application/json" \
     -d '{"rating": 4}'
# → {"seed": 42, "rating": 4, "message": "rated successfully"}
```

**`GET /mazes`** — Liste les mazes stockés, triés par date décroissante :
```bash
# 20 derniers mazes
curl https://pacman-s2-h.onrender.com/mazes

# Avec filtres
curl "https://pacman-s2-h.onrender.com/mazes?limit=5&rated_only=true&min_score=6.0"
```

Paramètres optionnels : `limit` (défaut 20), `rated_only` (true/false), `min_score` (float).

---

## Jeu Pac-Man

### Lancer le jeu complet

```bash
# Seed aléatoire, API Render
python -m src.game.game

# Seed fixe
python -m src.game.game --seed 42

# API locale (lancer app.py d'abord)
python -m src.game.game --local
```

**Contrôles :** `←↑→↓` ou `ZQSD` — déplacer Pac-Man | `P` — pause | `R` — rejouer | `Q / Échap` — quitter

### Visualiseur de labyrinthe (sans personnages)

Affiche uniquement les murs, pellets et power pellets, sans Pac-Man ni fantômes :

```bash
# Seed aléatoire
python -m src.maze_viewer

# Seed fixe
python -m src.maze_viewer --seed 42

# API locale
python -m src.maze_viewer --seed 42 --local
```

### Intégration notation en fin de partie

Le seed est retourné par `GET /maze` dès le chargement. Après une partie,
`main.py` peut noter le maze ainsi :

```python
import requests

seed = maze_data["seed"]   # récupéré au chargement
requests.patch(
    f"https://pacman-s2-h.onrender.com/maze/{seed}/rate",
    json={"rating": 4},
    timeout=5,
)
```

---

## Visualisation Pygame simple (client)

```bash
# Depuis le cloud Render
python client.py

# Depuis le serveur local
python client.py --local

# Avec seed fixe
python client.py --seed 42

# Changer la taille des cases
python client.py --tile-size 16
```

**Contrôles :** `ESPACE` → nouveau maze | `Q` → quitter

---

## Jeu Pac-Man

```bash
# Labyrinthe aléatoire (API Render)
python main.py

# Seed fixe
python main.py --seed 42

# API locale (lancer app.py d'abord)
python main.py --local

# Seed fixe en local
python main.py --local --seed 42

# Enregistrer la partie
python main.py --record
```

**Contrôles :** `Flèches / WASD` → déplacer Pac-Man | `P` → pause | `Q / Échap` → quitter

> Sur le plan gratuit Render, le service se met en veille après 15 min
> d'inactivité. Le premier appel peut prendre 50 secondes.

---

## Génération en ligne de commande (local)

```bash
# Générer un maze (JSON + PNG + TXT)
python generate.py

# Avec seed fixe
python generate.py --seed 42

# JSON uniquement
python generate.py --json-only
```

Options complètes :
- `-o, --output PATH` — dossier de sortie (défaut : `maze_[seed]/`)
- `--seed SEED` — graine pour la reproductibilité
- `-s, --size SIZE` — taille des tuiles en pixels (défaut : 20)
- `--json-only` / `--png-only` / `--text-only`
- `-v, --verbose`

### Score de complexité

```bash
python maze_complexity.py maze_42/maze.json
```

Le score est calculé via un algorithme greedy nearest-neighbor avec optimisation 2-opt.  
**Formule :** `score = (food_count / steps) × 10` (plafonné à 10)

---

## Tests

```bash
pip install pytest requests

# Tous les tests
pytest tests/ -v

# Tests fonctionnels uniquement (pas besoin de connexion internet)
pytest tests/test_api_local.py -v

# Tests des propriétés mathématiques du maze
pytest tests/test_maze_properties.py -v

# Tests end-to-end sur Render (nécessite connexion internet)
pytest tests/test_api_cloud.py -v
```

---

## Exemples curl

```bash
BASE=https://pacman-s2-h.onrender.com

curl $BASE/health
curl $BASE/maze
curl "$BASE/maze?seed=42"
curl $BASE/maze/42
curl -X PATCH $BASE/maze/42/rate -H "Content-Type: application/json" -d '{"rating": 4}'
curl "$BASE/mazes?limit=5&rated_only=true"
```

---

## CI/CD

Le fichier `.github/workflows/deploy.yml` automatise les déploiements sur Render.

À chaque push sur `main` :
1. GitHub Actions lance les tests automatiquement (`test_api_local.py` + `test_maze_properties.py`)
2. Si tous les tests passent → déploiement sur Render déclenché
3. Si un test échoue → le déploiement est bloqué

---

## License

See LICENSE file for details.

---

*Université Nice Côte d'Azur — M1 Informatique — TER S2*  
*Encadrant : Pr. Gilles Menez*  
*Membres : THEOLIEN Bierhoff, Hugo VIANA, JULES Stevenson*
