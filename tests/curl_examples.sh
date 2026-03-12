# =============================================================
# Exemples de commandes curl
# =============================================================
# Remplace BASE_URL par l'URL cloud ou locale selon le contexte
#
# Cloud  : https://pacman-s2-h.onrender.com
# Local  : http://localhost:5000
# =============================================================


# 1. Documentation de l'API
curl https://pacman-s2-h.onrender.com/

# 2. Vérifier que le service est en ligne (health check)
curl https://pacman-s2-h.onrender.com/health

# 3. Générer un maze aléatoire
curl https://pacman-s2-h.onrender.com/maze

# 4. Générer un maze avec un seed fixe (reproductible)
curl https://pacman-s2-h.onrender.com/maze?seed=42

# 5. Même seed = même maze (vérifier la reproductibilité)
curl https://pacman-s2-h.onrender.com/maze?seed=42
curl https://pacman-s2-h.onrender.com/maze?seed=42

# 6. Seed invalide; doit retourner une erreur 400
curl https://pacman-s2-h.onrender.com/maze?seed=abc

# =============================================================
# Mêmes commandes en local (avec le serveur lancé : python app.py)
# =============================================================

curl http://localhost:5000/
curl http://localhost:5000/health
curl http://localhost:5000/maze
curl http://localhost:5000/maze?seed=42
curl http://localhost:5000/maze?seed=abc
