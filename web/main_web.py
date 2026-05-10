#!/usr/bin/env python3
"""
Point d'entrée Pygbag — version navigateur du jeu Pac-Man.

Pygbag exige :
  1. Une fonction async def main()
  2. asyncio.run(main()) comme dernière instruction
  3. Pas de sys.exit() ni de boucle bloquante

Pas d'argparse : les paramètres sont figés pour la version web.
Pour changer le seed ou les options, modifier les constantes ci-dessous.
"""
import asyncio
import sys
import os

# Permet d'importer src_web depuis le même dossier que ce fichier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src_web.game.game import Game


async def main():
    # seed=None → labyrinthe aléatoire à chaque partie
    game = Game(seed=None, no_ghosts=False, ai="random")
    await game.run()


asyncio.run(main())
