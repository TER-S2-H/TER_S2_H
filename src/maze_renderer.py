"""
Maze Renderer - Python Version
Renders Pacman-style mazes to PNG images using PIL/Pillow
"""

from PIL import Image, ImageDraw
from typing import List, Tuple
import os


class MazeRenderer:
    """Renders mazes to PNG images"""
    
    def __init__(self, tile_size: int = 20):
        """
        Create a new MazeRenderer
        
        Args:
            tile_size: Size of each tile in pixels
        """
        self.tile_size = tile_size
        self.background_color = (0, 0, 0)  # Black
        self.wall_color = (33, 33, 222)  # Blue
        self.ghost_wall_color = (255, 255, 255)  # White
    
    def set_colors(self, background: Tuple[int, int, int] = None,
                   wall: Tuple[int, int, int] = None,
                   ghost_wall: Tuple[int, int, int] = None):
        """Set custom colors for maze elements"""
        if background:
            self.background_color = background
        if wall:
            self.wall_color = wall
        if ghost_wall:
            self.ghost_wall_color = ghost_wall
    
    def set_tile_size(self, size: int):
        """Set the tile size in pixels"""
        self.tile_size = size
    
    def render_to_image(self, tile_map: List[List]) -> Image.Image:
        """
        Render maze to a PIL Image
        
        Args:
            tile_map: 2D array of Tile objects
            
        Returns:
            PIL Image containing the rendered maze
        """
        from .pacman_map import Tile
        
        if not tile_map or not tile_map[0]:
            raise ValueError("TileMap cannot be empty")
        
        width = len(tile_map) * self.tile_size
        height = len(tile_map[0]) * self.tile_size
        
        # Create image with background color
        image = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(image)
        
        # Draw each tile
        for i in range(len(tile_map)):
            for j in range(len(tile_map[0])):
                self._draw_tile(draw, tile_map[i][j],
                                i * self.tile_size,
                                j * self.tile_size)

        return image
    
    def _draw_tile(self, draw: ImageDraw.ImageDraw, tile, x: int, y: int):
        """Draw a single tile"""
        from .pacman_map import Tile
        
        if tile.state == Tile.WALL:
            # Draw wall as filled rectangle
            draw.rectangle(
                [x, y, x + self.tile_size - 1, y + self.tile_size - 1],
                fill=self.wall_color
            )
        
        elif tile.state == Tile.GHOSTWALL:
            # Draw ghost wall as horizontal line
            line_y = y + self.tile_size // 2
            draw.rectangle(
                [x, line_y - 1, x + self.tile_size - 1, line_y + 1],
                fill=self.ghost_wall_color
            )
        
        elif tile.state == Tile.GHOSTWALL:
            # Draw ghost wall as horizontal bar
            wall_height = max(self.tile_size // 4, 2)
            wall_offset = (self.tile_size - wall_height) // 2
            draw.rectangle(
                [x, y + wall_offset, x + self.tile_size - 1, y + wall_offset + wall_height],
                fill=self.ghost_wall_color
            )
    
    def render_to_file(self, tile_map: List[List], filename: str, 
                       format: str = 'PNG'):
        """
        Render maze and save to file
        
        Args:
            tile_map: 2D array of Tile objects
            filename: Output filename
            format: Image format (PNG, JPEG, BMP, etc.)
        """
        image = self.render_to_image(tile_map)
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', 
                   exist_ok=True)
        
        # For JPEG, ensure RGB mode
        if format.upper() in ['JPEG', 'JPG']:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        image.save(filename, format=format.upper())
        print(f"Maze rendered to: {os.path.abspath(filename)}")
    
    @staticmethod
    def render_to_text(tile_map: List[List]) -> str:
        """
        Create a text representation of the maze
        
        Args:
            tile_map: 2D array of Tile objects
            
        Returns:
            String representation using ASCII/Unicode characters
        """
        from .pacman_map import Tile
        
        lines = []
        for j in range(len(tile_map[0])):
            line = []
            for i in range(len(tile_map)):
                tile = tile_map[i][j]
                if tile.state == Tile.WALL:
                    line.append('█')
                elif tile.state == Tile.GHOSTWALL:
                    line.append('─')
                elif tile.state == Tile.PATH:
                    line.append('·')
                else:
                    line.append(' ')
            lines.append(''.join(line))
        
        return '\n'.join(lines)
    
    @staticmethod
    def render_text_to_file(tile_map: List[List], filename: str):
        """
        Render text representation to file
        
        Args:
            tile_map: 2D array of Tile objects
            filename: Output filename
        """
        text = MazeRenderer.render_to_text(tile_map)
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', 
                   exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Text maze rendered to: {os.path.abspath(filename)}")

