import json
from datetime import datetime
from typing import Optional

from .agent import Direction


class Recorder:
    """
    Enregistre le déroulement d'une partie frame par frame.

    Format de chaque frame enregistrée :
        {
            "step"   : int,
            "actions": {"pacman": "UP", "ghost_0": "LEFT", …},
            "grid"   : [[char, …], …]   # snapshot 2D du plateau
        }

    Le fichier JSON final contient également le seed, ce qui permet
    de rejouer la partie à l'identique en rechargeant le même labyrinthe.

    Usage
    -----
        rec = Recorder(seed=env.seed)
        rec.record(actions_dict, env.get_grid_snapshot())
        rec.save("ma_partie.json")

        # Relecture
        rec2 = Recorder.load("ma_partie.json")
        print(rec2.seed)        # seed pour recharger le même maze
        print(rec2.frames[0])   # première frame
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed:   Optional[int] = seed
        self.frames: list = []
        self._step:  int  = 0
        self.coups:  int  = 0   # mouvements effectifs de Pac-Man, renseigné par Game

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def record(self, actions_dict: dict, grid_state: list) -> None:
        """
        Enregistre une frame.

        Paramètres
        ----------
        actions_dict : {"pacman": Direction, "ghost_0": Direction, …}
        grid_state   : env.get_grid_snapshot() – liste 2D de caractères
        """
        serialized_actions = {
            key: (val.name if isinstance(val, Direction) else str(val))
            for key, val in actions_dict.items()
        }
        self.frames.append({
            "step":    self._step,
            "actions": serialized_actions,
            "grid":    grid_state,
        })
        self._step += 1

    # ------------------------------------------------------------------
    # Sauvegarde / chargement
    # ------------------------------------------------------------------

    def save(self, filename: str) -> None:
        """Sauvegarde la partie enregistrée dans un fichier JSON."""
        data = {
            "seed":        self.seed,
            "recorded_at": datetime.now().isoformat(),
            "total_steps": self._step,
            "total_coups": self.coups,
            "frames":      self.frames,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Recorder] Partie sauvegardée : {filename} ({self._step} ticks, {self.coups} coups)")

    @classmethod
    def load(cls, filename: str) -> "Recorder":
        """Charge un fichier de replay et retourne un Recorder peuplé."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        rec = cls(seed=data.get("seed"))
        rec.frames = data.get("frames", [])
        rec._step  = data.get("total_steps", len(rec.frames))
        rec.coups  = data.get("total_coups", 0)
        return rec

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f"Recorder(seed={self.seed}, frames={len(self.frames)})"
