#!/usr/bin/env python3
"""
JSON Maze Renderer
Renders a maze directly from its JSON graph representation
"""

import argparse
import json
import os
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Set


class JSONMazeRenderer:
    """Renders mazes from JSON graph data"""
    
    def __init__(self, tile_size: int = 20):
        """
        Create a new JSON maze renderer
        
        Args:
            tile_size: Size of each tile in pixels
        """
        self.tile_size = tile_size
        self.background_color = (0, 0, 0)  # Black
        self.wall_color = (33, 33, 222)  # Blue
        self.ghost_wall_color = (255, 255, 255)  # White
        self.node_color = (255, 255, 0)  # Yellow (for visualization)
        self.edge_color = (100, 100, 100)  # Gray (for visualization)
    
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
    
    def load_json(self, json_path: str) -> Dict:
        """Load maze data from JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def render_to_image(self, maze_data: Dict, show_graph: bool = False) -> Image.Image:
        """
        Render maze from JSON data to a PIL Image
        
        Args:
            maze_data: Dictionary containing maze graph data
            show_graph: If True, visualize nodes and edges on top of maze
            
        Returns:
            PIL Image containing the rendered maze
        """
        width = maze_data['metadata']['width']
        height = maze_data['metadata']['height']
        
        img_width = width * self.tile_size
        img_height = height * self.tile_size
        
        # Create image with background color
        image = Image.new('RGB', (img_width, img_height), self.background_color)
        draw = ImageDraw.Draw(image)
        
        # Draw walls
        for wall in maze_data['walls']:
            x = wall['x'] * self.tile_size
            y = wall['y'] * self.tile_size
            draw.rectangle(
                [x, y, x + self.tile_size - 1, y + self.tile_size - 1],
                fill=self.wall_color
            )
        
        # Draw ghost walls
        for ghost_wall in maze_data['ghost_walls']:
            x = ghost_wall['x'] * self.tile_size
            y = ghost_wall['y'] * self.tile_size
            line_y = y + self.tile_size // 2
            draw.rectangle(
                [x, line_y - 1, x + self.tile_size - 1, line_y + 1],
                fill=self.ghost_wall_color
            )
        
        # Optionally draw graph visualization
        if show_graph:
            def parse_node_id(node_id: str) -> Tuple[int, int]:
                x_str, y_str = node_id.split(',')
                return int(x_str), int(y_str)

            # Draw edges first (so they appear behind nodes)
            for edge in maze_data['edges']:
                from_pos = parse_node_id(edge['from'])
                to_pos = parse_node_id(edge['to'])

                from_x = from_pos[0] * self.tile_size + self.tile_size // 2
                from_y = from_pos[1] * self.tile_size + self.tile_size // 2
                to_x = to_pos[0] * self.tile_size + self.tile_size // 2
                to_y = to_pos[1] * self.tile_size + self.tile_size // 2
                
                draw.line([from_x, from_y, to_x, to_y], fill=self.edge_color, width=1)
            
            # Draw nodes
            for node in maze_data['nodes']:
                x = node['x'] * self.tile_size + self.tile_size // 2
                y = node['y'] * self.tile_size + self.tile_size // 2
                radius = max(2, self.tile_size // 8)
                
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=self.node_color
                )
        
        return image
    
    def render_to_file(self, maze_data: Dict, output_path: str, show_graph: bool = False):
        """
        Render maze to file
        
        Args:
            maze_data: Dictionary containing maze graph data
            output_path: Path to save the image
            show_graph: If True, visualize nodes and edges
        """
        image = self.render_to_image(maze_data, show_graph)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        
        image.save(output_path)
        print(f"Maze rendered to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Render a maze from JSON graph data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('json_file',
                       help='Path to JSON maze file')
    
    parser.add_argument('-o', '--output',
                       help='Output image file (default: [json_name]_render.png)')
    
    parser.add_argument('-s', '--size',
                       type=int,
                       default=20,
                       help='Tile size in pixels (default: 20)')
    
    parser.add_argument('--show-graph',
                       action='store_true',
                       help='Show nodes and edges on the maze')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.json_file)[0]
        output_file = f"{base_name}_render.png"
    
    # Load and render
    if args.verbose:
        print(f"Loading JSON from: {args.json_file}")
    
    renderer = JSONMazeRenderer(tile_size=args.size)
    maze_data = renderer.load_json(args.json_file)
    
    if args.verbose:
        print(f"Maze size: {maze_data['metadata']['width']}x{maze_data['metadata']['height']}")
        print(f"Nodes: {maze_data['metadata']['node_count']}")
        print(f"Edges: {maze_data['metadata']['edge_count']}")
        print(f"Walls: {maze_data['metadata']['wall_count']}")
        print(f"Ghost walls: {maze_data['metadata']['ghost_wall_count']}")
        print(f"Rendering to: {output_file}")
    
    renderer.render_to_file(maze_data, output_file, show_graph=args.show_graph)
    
    if args.verbose:
        print("Done!")


if __name__ == "__main__":
    main()
