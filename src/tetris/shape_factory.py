from enum import Enum
from typing import List, Tuple

class ShapeType(Enum):
    I = "I"
    O = "O"
    T = "T"
    S = "S"
    Z = "Z"
    J = "J"
    L = "L"

class Shape:
    def __init__(self, blocks: List[Tuple[int, int]], shape_type: ShapeType):
        self.blocks = blocks
        self.shape_type = shape_type
    
    def rotate(self) -> None:
        """Rotate shape 90 degrees clockwise"""
        self.blocks = [(-y, x) for x, y in self.blocks]

class ShapeFactory:
    @staticmethod
    def create_shape(shape_type: ShapeType) -> Shape:
        """Factory method to create Tetris shapes"""
        shapes = {
            ShapeType.I: [(0, 0), (1, 0), (2, 0), (3, 0)],
            ShapeType.O: [(0, 0), (1, 0), (0, 1), (1, 1)],
            ShapeType.T: [(0, 0), (1, 0), (2, 0), (1, 1)],
            ShapeType.S: [(1, 0), (2, 0), (0, 1), (1, 1)],
            ShapeType.Z: [(0, 0), (1, 0), (1, 1), (2, 1)],
            ShapeType.J: [(0, 0), (0, 1), (1, 1), (2, 1)],
            ShapeType.L: [(2, 0), (0, 1), (1, 1), (2, 1)],
        }
        return Shape(shapes[shape_type], shape_type)