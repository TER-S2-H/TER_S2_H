PELLET_SCORES = {
    "normal": 10,
    "power":  50,
}


class Pellet:
    """
    Représente un pellet (gomme) sur le plateau.

    Attributs
    ---------
    x, y      : position dans la grille
    type      : "normal" (10 pts) ou "power" (50 pts, active le mode frightened)
    score     : points accordés à la collecte
    """

    def __init__(self, x: int, y: int, pellet_type: str = "normal"):
        if pellet_type not in PELLET_SCORES:
            raise ValueError(f"Type de pellet invalide : {pellet_type!r}")
        self.x = x
        self.y = y
        self.type = pellet_type
        self.score = PELLET_SCORES[pellet_type]

    def __repr__(self):
        return f"Pellet({self.type!r}, x={self.x}, y={self.y})"
