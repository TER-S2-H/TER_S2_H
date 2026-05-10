import json
import platform
from datetime import datetime
from typing import Optional

from .agent import Direction


class Recorder:
    """
    Enregistre le déroulement d'une partie frame par frame.
    Version web : save() déclenche un téléchargement navigateur au lieu d'écrire
    dans le filesystem (indisponible dans le sandbox WebAssembly).
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed:   Optional[int] = seed
        self.frames: list = []
        self._step:  int  = 0
        self.coups:  int  = 0

    def record(self, actions_dict: dict, grid_state: list) -> None:
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

    def save(self, filename: str) -> None:
        data = {
            "seed":        self.seed,
            "recorded_at": datetime.now().isoformat(),
            "total_steps": self._step,
            "total_coups": self.coups,
            "frames":      self.frames,
        }

        if platform.system() == "Emscripten":
            # Navigateur : déclencher un téléchargement via l'API JS
            try:
                from pyodide.ffi import to_js
                from js import document, URL, Blob  # type: ignore
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                blob = Blob.new(to_js([json_str]), to_js({"type": "application/json"}))
                url = URL.createObjectURL(blob)
                a = document.createElement("a")
                a.href = url
                a.setAttribute("download", "recording.json")
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
                URL.revokeObjectURL(url)
                print(f"[Recorder] Téléchargement lancé ({self._step} ticks, {self.coups} coups)")
            except Exception as e:
                print(f"[Recorder] Téléchargement impossible ({e}) — {self._step} ticks ignorés")
        else:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Recorder] Partie sauvegardée : {filename} ({self._step} ticks, {self.coups} coups)")

    @classmethod
    def load(cls, filename: str) -> "Recorder":
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        rec = cls(seed=data.get("seed"))
        rec.frames = data.get("frames", [])
        rec._step  = data.get("total_steps", len(rec.frames))
        rec.coups  = data.get("total_coups", 0)
        return rec

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return f"Recorder(seed={self.seed}, frames={len(self.frames)})"
