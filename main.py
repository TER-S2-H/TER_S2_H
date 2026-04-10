#!/usr/bin/env python3
"""
Point d'entrée du jeu Pac-Man.

Usage
-----
  python main.py                    # labyrinthe aléatoire (API Render)
  python main.py --seed 42          # seed fixe
  python main.py --local            # serveur Flask local (localhost:5000)
  python main.py --local --seed 42  # seed fixe en local
  python main.py --record           # enregistre la partie dans recording.json

Contrôles en jeu
----------------
  Flèches / WASD  : déplacer Pac-Man
  P               : pause
  Q / Échap        : quitter
"""

import argparse
from src.game.game import Game


def main():
    parser = argparse.ArgumentParser(description="Pac-Man — TER S2-H")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Seed pour la génération du labyrinthe")
    parser.add_argument("--local",  action="store_true",
                        help="Utiliser le serveur Flask local (localhost:5000)")
    parser.add_argument("--record", action="store_true",
                        help="Enregistrer la partie dans recording.json")
    args = parser.parse_args()

    game = Game(seed=args.seed, local=args.local, record=args.record)
    game.run()


if __name__ == "__main__":
    main()
