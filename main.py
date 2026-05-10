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
  python main.py --no-ghosts        # jouer sans fantômes
  python main.py --replay recording.json   # rejouer une partie enregistrée
  python main.py --ai random        # choisir l'IA des fantômes (défaut : random)

Contrôles en jeu
----------------
  Flèches / WASD  : déplacer Pac-Man
  P               : pause
  Q / Échap        : quitter
"""

import argparse
import sys
from src.game.game import Game
from src.game.recorder import Recorder
from src.game.pacman_agent import ReplayPacManAgent


def main():
    parser = argparse.ArgumentParser(description="Pac-Man — TER S2-H")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Seed pour la génération du labyrinthe")
    parser.add_argument("--local",  action="store_true",
                        help="Utiliser le serveur Flask local (localhost:5000)")
    parser.add_argument("--record", action="store_true",
                        help="Enregistrer la partie dans recording.json")
    parser.add_argument("--no-ghosts", action="store_true",
                        help="Désactiver les fantômes (mode solo sans adversaires)")
    parser.add_argument("--replay", type=str, default=None, metavar="FILE",
                        help="Rejouer une partie enregistrée (fichier JSON)")
    parser.add_argument("--ai", type=str, default="random", metavar="NAME",
                        help="IA des fantômes : 'random' (défaut)")
    args = parser.parse_args()

    pacman_agent = None
    seed = args.seed

    if args.replay:
        try:
            rec = Recorder.load(args.replay)
        except FileNotFoundError:
            print(f"Erreur : fichier de replay introuvable : {args.replay}")
            sys.exit(1)
        seed = rec.seed
        actions = [frame["actions"]["pacman"] for frame in rec.frames]
        pacman_agent = ReplayPacManAgent(actions)
        print(f"[Replay] {len(actions)} actions chargées — seed={seed}")

    game = Game(seed=seed, local=args.local, record=args.record,
                no_ghosts=args.no_ghosts, pacman_agent=pacman_agent,
                ai=args.ai)
    game.run()


if __name__ == "__main__":
    main()
