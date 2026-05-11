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

# 7. Récupérer un maze déjà stocké par son seed
curl https://pacman-s2-h.onrender.com/maze/42

# 8. Récupérer un seed inexistant → 404
curl https://pacman-s2-h.onrender.com/maze/999999

# 9. Lister les mazes stockés (20 derniers par défaut)
curl https://pacman-s2-h.onrender.com/mazes

# 10. Lister avec filtres : 5 résultats, seulement les notés, score min 3
curl "https://pacman-s2-h.onrender.com/mazes?limit=5&rated_only=true&min_score=3"

# 11. Noter un maze : rating valide (0-5)
curl -X PATCH https://pacman-s2-h.onrender.com/maze/42/rate \
     -H "Content-Type: application/json" \
     -d '{"rating": 4}'

# 12. Rating invalide (> 5) → 400
curl -X PATCH https://pacman-s2-h.onrender.com/maze/42/rate \
     -H "Content-Type: application/json" \
     -d '{"rating": 6}'

# =============================================================
# Mêmes commandes en local (avec le serveur lancé : python app.py)
# =============================================================

curl http://localhost:5000/
curl http://localhost:5000/health
curl http://localhost:5000/maze
curl http://localhost:5000/maze?seed=42
curl http://localhost:5000/maze?seed=abc
curl http://localhost:5000/maze/42
curl http://localhost:5000/maze/999999
curl http://localhost:5000/mazes
curl "http://localhost:5000/mazes?limit=5&rated_only=true&min_score=3"
curl -X PATCH http://localhost:5000/maze/42/rate \
     -H "Content-Type: application/json" \
     -d '{"rating": 4}'
curl -X PATCH http://localhost:5000/maze/42/rate \
     -H "Content-Type: application/json" \
     -d '{"rating": 6}'
