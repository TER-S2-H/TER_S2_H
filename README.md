# TER S2 — Pac-Man Maze Generator — Groupe H

Générateur de maze Pac-Man exposé en tant que web service HTTP, déployé sur Render.

**URL du service :** https://pacman-s2-h.onrender.com

---

## Installation

Python 3.6+ requis.

```bash
# Dépendances serveur (API Flask + générateur)
pip install -r requirements.txt

# Dépendances client local (visualisation Pygame)
pip install -r requirements-client.txt
```

> `requirements.txt` est utilisé par Render pour le déploiement — il ne contient
> pas pygame qui n'est utile qu'en local. C'est pourquoi il y a deux fichiers séparés.

---

## Structure du projet

```
├── app.py                        # API Flask (serveur web)
├── client.py                     # Visualisation Pygame (client local)
├── generate.py                   # Générateur en ligne de commande (wrapper)
├── maze_complexity.py            # Score de complexité (wrapper)
├── maze_complexity_visualize.py  # Visualisation du chemin (wrapper)
├── render_json.py                # Rendu depuis JSON (wrapper)
├── Procfile                      # Configuration Render
├── requirements.txt              # Dépendances serveur (Render)
├── requirements-client.txt       # Dépendances client local
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD GitHub Actions
├── src/
│   ├── pacman_map.py             # Logique de génération du maze
│   ├── maze_renderer.py          # Rendu PNG
│   ├── maze_complexity.py        # Calcul du score de complexité
│   ├── maze_complexity_visualize.py
│   ├── generate.py
│   └── render_json.py
└── tests/
    ├── test_api_local.py         # Tests fonctionnels (Flask test client)
    ├── test_maze_properties.py   # Tests caractéristiques du maze
    ├── test_api_cloud.py         # Tests end-to-end (Render)
    ├── curl_examples.sh          # Exemples de commandes curl
    └── conftest.py               # Configuration pytest
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
| `/maze` | GET | Génère un maze (seed aléatoire) |
| `/maze?seed=42` | GET | Génère un maze reproductible |

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
  "nodes": [{"id": "1,1", "x": 1, "y": 1, "type": "PATH"}, "..."],
  "edges": [{"from": "1,1", "to": "2,1"}, "..."],
  "walls": [{"x": 0, "y": 0}, "..."],
  "ghost_walls": [{"x": 13, "y": 12}, "..."]
}
```

### Visualisation Pygame (client local)

```bash
# Depuis le cloud Render
python client.py

# Depuis le serveur local (lancer app.py d'abord)
python client.py --local

# Avec un seed fixe
python client.py --seed 42

# Changer la taille des cases
python client.py --tile-size 16
```

**Contrôles :** `ESPACE` → nouveau maze | `Q` → quitter

> Sur le plan gratuit Render, le service se met en veille après 15 min
> d'inactivité. Le premier appel peut prendre 50 secondes.

---

## Génération en ligne de commande (local)

```bash
# Générer un maze (JSON + PNG + TXT)
python generate.py

# Avec seed fixe
python generate.py --seed 42

# Changer la taille des tuiles
python generate.py -s 30

# JSON uniquement
python generate.py --json-only
```

Options complètes :
- `-o, --output PATH` — dossier de sortie (défaut : `maze_[seed]/`)
- `--seed SEED` — graine pour la reproductibilité
- `-s, --size SIZE` — taille des tuiles en pixels (défaut : 20)
- `--json-only` / `--png-only` / `--text-only`
- `-v, --verbose`

### Rendu depuis un JSON existant

```bash
python render_json.py maze_42/maze.json
python render_json.py maze_42/maze.json -o output.png -s 30
python render_json.py maze_42/maze.json --show-graph
```

### Score de complexité

```bash
python maze_complexity.py maze_42/maze.json
```

Le score est calculé via un algorithme greedy nearest-neighbor avec optimisation 2-opt.  
**Formule :** `score = (food_count / steps) × 10` (plafonné à 10)

### Visualisation animée du chemin

```bash
python maze_complexity_visualize.py maze_42/maze.json
```

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

Voir `tests/curl_examples.sh` pour la liste complète.

```bash
curl https://pacman-s2-h.onrender.com/health
curl https://pacman-s2-h.onrender.com/maze
curl https://pacman-s2-h.onrender.com/maze?seed=42
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
