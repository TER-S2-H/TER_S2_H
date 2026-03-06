"""
Pacman Maze Generator - Python Version
Ported from Java implementation by Ciro Duran
"""

import random
import json
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field


@dataclass
class IntTuple:
    """Position tuple with direction"""
    x: int = 0
    y: int = 0
    direction: int = -1
    
    def equals(self, other: 'IntTuple') -> bool:
        return self.x == other.x and self.y == other.y


class Tile:
    """Represents a tile in the maze"""
    BLANK = 0
    PATH = 1
    PATHBLANK = 2
    WALL = 3
    GHOSTWALL = 4
    ENERGIZER = 5
    GHOSTSPACE = 6
    
    def __init__(self, x: int, y: int):
        self.position = IntTuple(x, y)
        self.state = Tile.BLANK
        self.cell = None


class Cell:
    """Represents a cell in the generation grid"""
    num_filled = 0
    
    # Direction constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, x: int, y: int):
        self.filled = False
        self.position = IntTuple(x, y)
        self.connect = [False, False, False, False]
        self.next = [None, None, None, None]
        self.number = -1
        self.group = -1
        
        self.is_ghost_space = False
        
        self.is_raise_height_candidate = False
        self.raise_height = False
        self.is_shrink_width_candidate = False
        self.shrink_width = False
        
        self.is_join_candidate = False
        
        self.is_edge_tunnel_candidate = False
        self.is_void_tunnel_candidate = False
        self.is_single_dead_end_candidate = False
        self.single_dead_end_dir = -1
        self.is_double_dead_end_candidate = False
        self.top_tunnel = False
        
        self.tile_position = IntTuple()
        self.tile_size = IntTuple()
    
    def is_open(self, dir: int, prev_dir: int = -1, size: int = -1) -> bool:
        """Check if a direction is open for movement"""
        if ((self.position.y == 6 and self.position.x == 0 and dir == Cell.DOWN) or
            (self.position.y == 7 and self.position.x == 0 and dir == Cell.UP)):
            return False
        
        if size == 2 and (dir == prev_dir or ((dir + 2) % 4) == prev_dir):
            return False
        
        if self.next[dir] is not None and not self.next[dir].filled:
            if not (self.next[dir].next[Cell.LEFT] is not None and 
                   not self.next[dir].next[Cell.LEFT].filled):
                return True
        
        return False
    
    def get_open_cells(self, prev_dir: int, size: int) -> List[int]:
        """Get list of open directions"""
        open_cells = []
        for i in range(4):
            if self.is_open(i, prev_dir, size):
                open_cells.append(i)
        return open_cells
    
    def connect_to(self, dir: int):
        """Connect this cell to a neighbor"""
        self.connect[dir] = True
        self.next[dir].connect[(dir + 2) % 4] = True
        if self.position.x == 0 and dir == Cell.RIGHT:
            self.connect[Cell.LEFT] = True
    
    def fill(self, num_group: int):
        """Fill this cell as part of a group"""
        self.filled = True
        self.number = Cell.num_filled
        Cell.num_filled += 1
        self.group = num_group
    
    def is_cross_center(self) -> bool:
        """Check if cell is a 4-way intersection"""
        return all(self.connect)


class PacmanMap:
    """Generates Pacman-style mazes"""
    
    def __init__(self, seed: Optional[int] = None):
        self.cell_map_size = IntTuple(5, 9)
        self.cell_map: List[List[Cell]] = []
        self.tile_map: List[List[Tile]] = []
        self.tall_rows: List[int] = []
        self.narrow_cols: List[int] = []
        self.gen_count = 0
        
        if seed is not None:
            random.seed(seed)
        
        self.reset()
    
    def reset(self):
        """Reset the map for a new generation"""
        Cell.num_filled = 0
        self.cell_map = [[Cell(i, j) for j in range(self.cell_map_size.y)] 
                         for i in range(self.cell_map_size.x)]
        
        # Set up neighbor references
        for j in range(self.cell_map_size.y):
            for i in range(self.cell_map_size.x):
                c = self.cell_map[i][j]
                c.next[Cell.LEFT] = self.cell_map[i-1][j] if i > 0 else None
                c.next[Cell.RIGHT] = self.cell_map[i+1][j] if i < self.cell_map_size.x - 1 else None
                c.next[Cell.UP] = self.cell_map[i][j-1] if j > 0 else None
                c.next[Cell.DOWN] = self.cell_map[i][j+1] if j < self.cell_map_size.y - 1 else None
        
        # Set up ghost space
        for pos in [(0, 3), (1, 3), (0, 4), (1, 4)]:
            i, j = pos
            self.cell_map[i][j].is_ghost_space = True
            self.cell_map[i][j].filled = True
        
        self.cell_map[0][3].connect[Cell.LEFT] = True
        self.cell_map[0][3].connect[Cell.RIGHT] = True
        self.cell_map[0][3].connect[Cell.DOWN] = True
        
        self.cell_map[1][3].connect[Cell.LEFT] = True
        self.cell_map[1][3].connect[Cell.DOWN] = True
        
        self.cell_map[0][4].connect[Cell.LEFT] = True
        self.cell_map[0][4].connect[Cell.RIGHT] = True
        self.cell_map[0][4].connect[Cell.UP] = True
        
        self.cell_map[1][4].connect[Cell.LEFT] = True
        self.cell_map[1][4].connect[Cell.UP] = True
        
        self.tall_rows = [-1] * self.cell_map_size.x
        self.narrow_cols = [-1] * self.cell_map_size.y
    
    def generate(self):
        """Generate a new maze"""
        while True:
            self.reset()
            self.attempt_generate()
            self.gen_count += 1
            
            if not self.is_desirable():
                continue
            
            self.set_up_scale_coords()
            self.join_walls()
            
            if not self.create_tunnels():
                continue
            
            break
    
    def get_cell_map(self) -> List[List[Cell]]:
        """Get the cell map"""
        return self.cell_map
    
    def get_tile_map(self) -> List[List[Tile]]:
        """Generate and return the tile map"""
        sub_x = self.cell_map_size.x * 3 - 1 + 2
        sub_y = self.cell_map_size.y * 3 + 1 + 3
        mid_x = sub_x - 2
        full_x = (sub_x - 2) * 2
        
        # Create tile map
        result = [[Tile(i, j) for j in range(sub_y)] for i in range(full_x)]
        
        # Map cells to tiles
        for j in range(self.cell_map_size.y):
            for i in range(self.cell_map_size.x):
                c = self.cell_map[i][j]
                
                for x0 in range(c.tile_size.x):
                    for y0 in range(c.tile_size.y):
                        result[c.tile_position.x + x0][c.tile_position.y + 1 + y0].cell = c
                        if c.is_ghost_space:
                            self._set_tile(result, c.tile_position.x + x0, 
                                         c.tile_position.y + y0, Tile.GHOSTSPACE, 
                                         sub_x, sub_y, mid_x)
        
        # Generate paths
        for j in range(sub_y):
            for i in range(sub_x):
                c = result[i][j].cell if result[i][j] else None
                cl = result[i-1][j].cell if i > 0 else None
                cu = result[i][j-1].cell if j > 0 else None
                
                if c is not None:
                    if ((cl is not None and c.group != cl.group) or
                        (cu is not None and c.group != cu.group) or
                        (cu is None and not c.connect[Cell.UP])):
                        self._set_tile(result, i, j, Tile.PATH, sub_x, sub_y, mid_x)
                else:
                    if ((cl is not None and 
                         (not cl.connect[Cell.RIGHT] or 
                          self._get_tile_state(result, i-1, j, sub_x, sub_y, mid_x) == Tile.PATH)) or
                        (cu is not None and
                         (not cu.connect[Cell.DOWN] or 
                          self._get_tile_state(result, i, j-1, sub_x, sub_y, mid_x) == Tile.PATH))):
                        self._set_tile(result, i, j, Tile.PATH, sub_x, sub_y, mid_x)
                
                if (self._get_tile_state(result, i-1, j, sub_x, sub_y, mid_x) == Tile.PATH and
                    self._get_tile_state(result, i, j-1, sub_x, sub_y, mid_x) == Tile.PATH and
                    self._get_tile_state(result, i-1, j-1, sub_x, sub_y, mid_x) == Tile.BLANK):
                    self._set_tile(result, i, j, Tile.PATH, sub_x, sub_y, mid_x)
        
        # Create tunnels
        c = self.cell_map[self.cell_map_size.x - 1][0]
        while c is not None:
            if c.top_tunnel:
                y = c.tile_position.y + 1
                self._set_tile(result, sub_x - 1, y, Tile.PATH, sub_x, sub_y, mid_x)
                self._set_tile(result, sub_x - 2, y, Tile.PATH, sub_x, sub_y, mid_x)
            c = c.next[Cell.DOWN]
        
        # Generate walls
        for j in range(sub_y):
            for i in range(sub_x):
                if self._get_tile_state(result, i, j, sub_x, sub_y, mid_x) != Tile.PATH:
                    adjacent_to_path = any([
                        self._get_tile_state(result, i+dx, j+dy, sub_x, sub_y, mid_x) == Tile.PATH
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (1,1), (-1,1)]
                    ])
                    if adjacent_to_path:
                        self._set_tile(result, i, j, Tile.WALL, sub_x, sub_y, mid_x)
        
        # Add ghost wall
        self._set_tile(result, 2, 12, Tile.GHOSTWALL, sub_x, sub_y, mid_x)
        
        # Add energizers
        range_top = self._get_top_energizer_range(result, sub_x, sub_y, mid_x)
        if range_top:
            y = int(range_top[0] + random.random() * range_top[1])
            self._set_tile(result, sub_x - 2, y, Tile.ENERGIZER, sub_x, sub_y, mid_x)
        
        range_bot = self._get_bot_energizer_range(result, sub_x, sub_y, mid_x)
        if range_bot:
            y = int(range_bot[0] + random.random() * range_bot[1])
            self._set_tile(result, sub_x - 2, y, Tile.ENERGIZER, sub_x, sub_y, mid_x)
        
        # Erase dead ends at edge
        for y in range(sub_y):
            x = sub_x - 1
            if self._get_tile_state(result, x, y, sub_x, sub_y, mid_x) == Tile.PATH:
                self._erase_until_intersection(result, x, y, sub_x, sub_y, mid_x)
        
        # Clear spawn area
        self._set_tile(result, 1, sub_y - 8, Tile.PATHBLANK, sub_x, sub_y, mid_x)
        
        for i in range(7):
            y = sub_y - 14
            self._set_tile(result, i, y, Tile.PATHBLANK, sub_x, sub_y, mid_x)
            j = 1
            while (self._get_tile_state(result, i, y+j, sub_x, sub_y, mid_x) == Tile.PATH and
                   self._get_tile_state(result, i-1, y+j, sub_x, sub_y, mid_x) == Tile.WALL and
                   self._get_tile_state(result, i+1, y+j, sub_x, sub_y, mid_x) == Tile.WALL):
                self._set_tile(result, i, y+j, Tile.PATHBLANK, sub_x, sub_y, mid_x)
                j += 1
            
            y = sub_y - 20
            self._set_tile(result, i, y, Tile.PATHBLANK, sub_x, sub_y, mid_x)
            j = 1
            while (self._get_tile_state(result, i, y-j, sub_x, sub_y, mid_x) == Tile.PATH and
                   self._get_tile_state(result, i-1, y-j, sub_x, sub_y, mid_x) == Tile.WALL and
                   self._get_tile_state(result, i+1, y-j, sub_x, sub_y, mid_x) == Tile.WALL):
                self._set_tile(result, i, y-j, Tile.PATHBLANK, sub_x, sub_y, mid_x)
                j += 1
        
        for i in range(7):
            x = 6
            y = sub_y - 14 - i
            self._set_tile(result, x, y, Tile.PATHBLANK, sub_x, sub_y, mid_x)
            j = 1
            while (self._get_tile_state(result, x+j, y, sub_x, sub_y, mid_x) == Tile.PATH and
                   self._get_tile_state(result, x+j, y-1, sub_x, sub_y, mid_x) == Tile.WALL and
                   self._get_tile_state(result, x+j, y+1, sub_x, sub_y, mid_x) == Tile.WALL):
                self._set_tile(result, x+j, y, Tile.PATHBLANK, sub_x, sub_y, mid_x)
                j += 1
        
        self.tile_map = result
        return result
    
    def _set_tile(self, tile_map, x, y, state, w, h, mid_x):
        """Set a tile in the map with mirroring"""
        if x < 0 or x >= w or y < 0 or y >= h:
            return
        x -= 2
        tile_map[mid_x + x][y].state = state
        tile_map[mid_x - 1 - x][y].state = state
    
    def _get_tile_state(self, tile_map, x, y, w, h, mid_x) -> int:
        """Get tile state with bounds checking"""
        if x < 0 or x >= w or y < 0 or y >= h:
            return -1
        x -= 2
        return tile_map[mid_x + x][y].state
    
    def _get_top_energizer_range(self, tile_map, w, h, mid_x) -> Optional[Tuple[int, int]]:
        """Get valid range for top energizer placement"""
        x = w - 2
        min_y = 0
        max_y = h // 2
        
        for y in range(1, max_y):
            if (self._get_tile_state(tile_map, x, y, w, h, mid_x) == Tile.PATH and
                self._get_tile_state(tile_map, x, y+1, w, h, mid_x) == Tile.PATH):
                min_y = y + 1
                break
        
        max_y = min(max_y, min_y + 7)
        for y in range(min_y + 1, max_y):
            if self._get_tile_state(tile_map, x-1, y, w, h, mid_x) == Tile.PATH:
                max_y = y
                break
        
        return (min_y, max_y - 1) if max_y > min_y else None
    
    def _get_bot_energizer_range(self, tile_map, w, h, mid_x) -> Optional[Tuple[int, int]]:
        """Get valid range for bottom energizer placement"""
        x = w - 2
        min_y = h // 2
        max_y = 0
        
        for y in range(h - 3, min_y - 1, -1):
            if (self._get_tile_state(tile_map, x, y, w, h, mid_x) == Tile.PATH and
                self._get_tile_state(tile_map, x, y+1, w, h, mid_x) == Tile.PATH):
                max_y = y
                break
        
        min_y = max(min_y, max_y - 7)
        for y in range(max_y - 1, min_y, -1):
            if self._get_tile_state(tile_map, x-1, y, w, h, mid_x) == Tile.PATH:
                min_y = y + 1
                break
        
        return (min_y, max_y - 1) if max_y > min_y else None
    
    def _erase_until_intersection(self, tile_map, x, y, w, h, mid_x):
        """Erase path tiles until reaching an intersection"""
        while True:
            adj = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self._get_tile_state(tile_map, x+dx, y+dy, w, h, mid_x) == Tile.PATH:
                    adj.append((x+dx, y+dy))
            
            if len(adj) == 1:
                self._set_tile(tile_map, x, y, Tile.PATHBLANK, w, h, mid_x)
                x, y = adj[0]
            else:
                break
    
    def get_left_most_empty_cells(self) -> List[Cell]:
        """Get leftmost empty cells"""
        result = []
        for i in range(self.cell_map_size.x):
            for j in range(self.cell_map_size.y):
                c = self.cell_map[i][j]
                if not c.filled:
                    result.append(c)
            if result:
                break
        return result
    
    def attempt_generate(self):
        """Attempt to generate the maze structure"""
        single_count = [0, 0]
        prob_stop_growing_at_size = [0.0, 0.0, 0.1, 0.5, 0.75, 1.0]
        prob_top_and_bot_single_cell_join = 0.35
        prob_extend_at_size_2 = 1.0
        prob_extend_at_size_3_or_4 = 0.5
        
        long_pieces = 0
        MAX_LONG_PIECES = 1
        dir = -1
        
        num_groups = 0
        while True:
            open_cells = self.get_left_most_empty_cells()
            num_open_cells = len(open_cells)
            
            if num_open_cells == 0:
                break
            
            first_cell = cell = random.choice(open_cells)
            cell.fill(num_groups)
            
            if (cell.position.x < self.cell_map_size.x - 1 and
                (cell.position.y == 0 or cell.position.y == self.cell_map_size.y - 1) and
                random.random() < prob_top_and_bot_single_cell_join):
                single_count_pos = 0 if cell.position.y == 0 else 1
                if single_count[single_count_pos] == 0:
                    cell.connect[Cell.UP if cell.position.y == 0 else Cell.DOWN] = True
                    single_count[single_count_pos] += 1
                    num_groups += 1
                    continue
            
            size = 1
            
            if cell.position.x == self.cell_map_size.x - 1:
                cell.connect[Cell.RIGHT] = True
                cell.is_raise_height_candidate = True
            else:
                while size < 5:
                    stop = False
                    
                    if size == 2:
                        c = first_cell
                        if (c.position.x > 0 and c.connect[Cell.RIGHT] and
                            c.next[Cell.RIGHT] is not None and
                            c.next[Cell.RIGHT].next[Cell.RIGHT] is not None):
                            if long_pieces < MAX_LONG_PIECES and random.random() < prob_extend_at_size_2:
                                chosen_dir = -1
                                c = c.next[Cell.RIGHT].next[Cell.RIGHT]
                                dirs = [False] * 4
                                
                                if c.is_open(Cell.UP):
                                    dirs[Cell.UP] = True
                                if c.is_open(Cell.DOWN):
                                    dirs[Cell.DOWN] = True
                                
                                if dirs[Cell.UP] and dirs[Cell.DOWN]:
                                    chosen_dir = Cell.UP if random.random() < 0.5 else Cell.DOWN
                                elif dirs[Cell.UP]:
                                    chosen_dir = Cell.UP
                                elif dirs[Cell.DOWN]:
                                    chosen_dir = Cell.DOWN
                                
                                if chosen_dir != -1:
                                    c.connect_to(Cell.LEFT)
                                    c.fill(num_groups)
                                    c.connect_to(chosen_dir)
                                    c.next[chosen_dir].fill(num_groups)
                                    long_pieces += 1
                                    size += 2
                                    stop = True
                    
                    if not stop:
                        le_open_cells = cell.get_open_cells(dir, size)
                        num_le_open_cells = len(le_open_cells)
                        
                        if num_le_open_cells == 0 and size == 2:
                            cell = new_cell
                            le_open_cells = cell.get_open_cells(dir, size)
                            num_le_open_cells = len(le_open_cells)
                        
                        if num_le_open_cells == 0:
                            stop = True
                        else:
                            dir = random.choice(le_open_cells)
                            new_cell = cell.next[dir]
                            
                            cell.connect_to(dir)
                            new_cell.fill(num_groups)
                            size += 1
                            
                            if first_cell.position.x == 0 and size == 3:
                                stop = True
                            
                            if random.random() < prob_stop_growing_at_size[size]:
                                stop = True
                    
                    if stop:
                        if size == 1:
                            pass
                        elif size == 2:
                            c = first_cell
                            if c.position.x == self.cell_map_size.x - 1:
                                if c.connect[Cell.UP]:
                                    c = c.next[Cell.UP]
                                c.connect[Cell.RIGHT] = True
                                c.next[Cell.DOWN].connect[Cell.RIGHT] = True
                        elif size == 3 or size == 4:
                            if (long_pieces < MAX_LONG_PIECES and
                                first_cell.position.x > 0 and
                                random.random() <= prob_extend_at_size_3_or_4):
                                dirs = []
                                for i in range(4):
                                    if (cell.connect[i] and cell.next[i] is not None and
                                        cell.next[i].is_open(i)):
                                        dirs.append(i)
                                if dirs:
                                    d = random.choice(dirs)
                                    c = cell.next[d]
                                    c.connect_to(d)
                                    c.next[d].fill(num_groups)
                                    long_pieces += 1
                        break
            
            num_groups += 1
        
        self.set_resize_candidates()
    
    def set_resize_candidates(self):
        """Mark cells that can be resized"""
        for j in range(self.cell_map_size.y):
            for i in range(self.cell_map_size.x):
                c = self.cell_map[i][j]
                q = c.connect
                
                if ((c.position.x == 0 or not q[Cell.LEFT]) and
                    (c.position.x == self.cell_map_size.x - 1 or not q[Cell.RIGHT]) and
                    q[Cell.UP] != q[Cell.DOWN]):
                    c.is_raise_height_candidate = True
                
                c2 = c.next[Cell.RIGHT]
                if c2 is not None:
                    q2 = c2.connect
                    if (((c.position.x == 0 or not q[Cell.LEFT]) and 
                         not q[Cell.UP] and not q[Cell.DOWN]) and
                        ((c2.position.x == self.cell_map_size.x - 1 or not q2[Cell.RIGHT]) and
                         not q2[Cell.UP] and not q2[Cell.DOWN])):
                        c.is_raise_height_candidate = True
                        c2.is_raise_height_candidate = True
                
                if c.position.x == self.cell_map_size.x - 1 and q[Cell.RIGHT]:
                    c.is_shrink_width_candidate = True
                
                if ((c.position.y == 0 or not q[Cell.UP]) and
                    (c.position.y == self.cell_map_size.y - 1 or not q[Cell.DOWN]) and
                    q[Cell.LEFT] != q[Cell.RIGHT]):
                    c.is_shrink_width_candidate = True
    
    def is_desirable(self) -> bool:
        """Check if the generated maze meets requirements"""
        c = self.cell_map[self.cell_map_size.x - 1][0]
        if c.connect[Cell.UP] or c.connect[Cell.RIGHT]:
            return False
        
        c = self.cell_map[self.cell_map_size.x - 1][self.cell_map_size.y - 1]
        if c.connect[Cell.DOWN] or c.connect[Cell.RIGHT]:
            return False
        
        for y in range(self.cell_map_size.y - 1):
            for x in range(self.cell_map_size.x - 1):
                if ((self.is_stacked_horizontal(x, y) and self.is_stacked_horizontal(x, y + 1)) or
                    (self.is_stacked_vertical(x, y) and self.is_stacked_vertical(x + 1, y))):
                    if x == 0:
                        return False
                    
                    g = self.cell_map[x][y].group
                    self.cell_map[x][y].connect[Cell.DOWN] = True
                    self.cell_map[x][y].connect[Cell.RIGHT] = True
                    
                    self.cell_map[x + 1][y].connect[Cell.DOWN] = True
                    self.cell_map[x + 1][y].connect[Cell.LEFT] = True
                    self.cell_map[x + 1][y].group = g
                    
                    self.cell_map[x][y + 1].connect[Cell.UP] = True
                    self.cell_map[x][y + 1].connect[Cell.RIGHT] = True
                    self.cell_map[x][y + 1].group = g
                    
                    self.cell_map[x + 1][y + 1].connect[Cell.UP] = True
                    self.cell_map[x + 1][y + 1].connect[Cell.LEFT] = True
                    self.cell_map[x + 1][y + 1].group = g
        
        if not self.choose_tall_rows():
            return False
        
        if not self.choose_narrow_cols():
            return False
        
        return True
    
    def is_stacked_horizontal(self, x: int, y: int) -> bool:
        """Check if cells are stacked horizontally"""
        q1 = self.cell_map[x][y].connect
        q2 = self.cell_map[x + 1][y].connect
        return (not q1[Cell.UP] and not q1[Cell.DOWN] and 
                (x == 0 or not q1[Cell.LEFT]) and q1[Cell.RIGHT] and
                not q2[Cell.UP] and not q2[Cell.DOWN] and 
                q2[Cell.LEFT] and not q2[Cell.RIGHT])
    
    def is_stacked_vertical(self, x: int, y: int) -> bool:
        """Check if cells are stacked vertically"""
        q1 = self.cell_map[x][y].connect
        q2 = self.cell_map[x][y + 1].connect
        
        if x == self.cell_map_size.x - 1:
            return (not q1[Cell.LEFT] and not q1[Cell.UP] and not q1[Cell.DOWN] and
                    not q2[Cell.LEFT] and not q2[Cell.UP] and not q2[Cell.DOWN])
        
        return (not q1[Cell.LEFT] and not q1[Cell.RIGHT] and not q1[Cell.UP] and
                q1[Cell.DOWN] and not q2[Cell.LEFT] and not q2[Cell.RIGHT] and
                q2[Cell.UP] and not q2[Cell.DOWN])
    
    def choose_tall_rows(self) -> bool:
        """Choose rows to make taller"""
        for y in range(3):
            c = self.cell_map[0][y]
            if c.is_raise_height_candidate and self.can_raise_height(0, y):
                c.raise_height = True
                self.tall_rows[c.position.x] = c.position.y
                return True
        return False
    
    def can_raise_height(self, x: int, y: int) -> bool:
        """Check if height can be raised"""
        if x == self.cell_map_size.x - 1:
            return True
        
        c2 = None
        for y0 in range(y, -1, -1):
            c = self.cell_map[x][y0]
            c2 = c.next[Cell.RIGHT]
            if ((not c.connect[Cell.UP] or c.is_cross_center()) and
                (not c2.connect[Cell.UP] or c2.is_cross_center())):
                break
        
        candidates = []
        while c2 is not None:
            if c2.is_raise_height_candidate:
                candidates.append(c2)
            
            if ((not c2.connect[Cell.DOWN] or c2.is_cross_center()) and
                (not c2.next[Cell.LEFT].connect[Cell.DOWN] or 
                 c2.next[Cell.LEFT].is_cross_center())):
                break
            c2 = c2.next[Cell.DOWN]
        
        random.shuffle(candidates)
        
        for c2 in candidates:
            if self.can_raise_height(c2.position.x, c2.position.y):
                c2.raise_height = True
                self.tall_rows[c2.position.x] = c2.position.y
                return True
        
        return False
    
    def choose_narrow_cols(self) -> bool:
        """Choose columns to make narrower"""
        for x in range(self.cell_map_size.x - 1, -1, -1):
            c = self.cell_map[x][0]
            if c.is_shrink_width_candidate and self.can_shrink_width(x, 0):
                c.shrink_width = True
                self.narrow_cols[c.position.y] = c.position.x
                return True
        return False
    
    def can_shrink_width(self, x: int, y: int) -> bool:
        """Check if width can be shrunk"""
        if y == self.cell_map_size.y - 1:
            return True
        
        c2 = None
        for x0 in range(x, self.cell_map_size.x):
            c = self.cell_map[x0][y]
            c2 = c.next[Cell.DOWN]
            if ((not c.connect[Cell.RIGHT] or c.is_cross_center()) and
                (not c2.connect[Cell.RIGHT] or c2.is_cross_center())):
                break
        
        candidates = []
        while c2 is not None:
            if c2.is_shrink_width_candidate:
                candidates.append(c2)
            
            if ((not c2.connect[Cell.LEFT] or c2.is_cross_center()) and
                (not c2.next[Cell.UP].connect[Cell.LEFT] or 
                 c2.next[Cell.UP].is_cross_center())):
                break
            c2 = c2.next[Cell.LEFT]
        
        random.shuffle(candidates)
        
        for c2 in candidates:
            if self.can_shrink_width(c2.position.x, c2.position.y):
                c2.shrink_width = True
                self.narrow_cols[c2.position.y] = c2.position.x
                return True
        
        return False
    
    def set_up_scale_coords(self):
        """Set up tile coordinates for cells"""
        for j in range(self.cell_map_size.y):
            for i in range(self.cell_map_size.x):
                c = self.cell_map[i][j]
                
                c.tile_position.x = c.position.x * 3
                if self.narrow_cols[c.position.y] < c.position.x:
                    c.tile_position.x -= 1
                
                c.tile_position.y = c.position.y * 3
                if self.tall_rows[c.position.x] < c.position.y:
                    c.tile_position.y += 1
                
                c.tile_size.x = 2 if c.shrink_width else 3
                c.tile_size.y = 4 if c.raise_height else 3
    
    def join_walls(self):
        """Join walls where appropriate"""
        for x in range(self.cell_map_size.x):
            c = self.cell_map[x][0]
            if (not c.connect[Cell.LEFT] and not c.connect[Cell.RIGHT] and
                not c.connect[Cell.UP] and
                (not c.connect[Cell.DOWN] or not c.next[Cell.DOWN].connect[Cell.DOWN])):
                
                if ((c.next[Cell.LEFT] is None or not c.next[Cell.LEFT].connect[Cell.UP]) and
                    (c.next[Cell.RIGHT] is not None and not c.next[Cell.RIGHT].connect[Cell.UP])):
                    
                    if not (c.next[Cell.DOWN] is not None and
                           c.next[Cell.DOWN].connect[Cell.RIGHT] and
                           c.next[Cell.DOWN].next[Cell.RIGHT].connect[Cell.RIGHT]):
                        c.is_join_candidate = True
                        if random.random() <= 0.25:
                            c.connect[Cell.UP] = True
        
        for x in range(self.cell_map_size.x):
            c = self.cell_map[x][self.cell_map_size.y - 1]
            if (not c.connect[Cell.LEFT] and not c.connect[Cell.RIGHT] and
                not c.connect[Cell.DOWN] and
                (not c.connect[Cell.UP] or not c.next[Cell.UP].connect[Cell.UP])):
                
                if ((c.next[Cell.LEFT] is None or not c.next[Cell.LEFT].connect[Cell.DOWN]) and
                    (c.next[Cell.RIGHT] is not None and not c.next[Cell.RIGHT].connect[Cell.DOWN])):
                    
                    if not (c.next[Cell.UP] is not None and
                           c.next[Cell.UP].connect[Cell.RIGHT] and
                           c.next[Cell.UP].next[Cell.RIGHT].connect[Cell.RIGHT]):
                        c.is_join_candidate = True
                        if random.random() <= 0.25:
                            c.connect[Cell.DOWN] = True
        
        for y in range(1, self.cell_map_size.y - 1):
            c = self.cell_map[self.cell_map_size.x - 1][y]
            
            if c.raise_height:
                continue
            
            if (not c.connect[Cell.RIGHT] and not c.connect[Cell.UP] and
                not c.connect[Cell.DOWN] and
                not c.next[Cell.UP].connect[Cell.RIGHT] and
                not c.next[Cell.DOWN].connect[Cell.RIGHT]):
                
                if c.connect[Cell.LEFT]:
                    c2 = c.next[Cell.LEFT]
                    if (not c2.connect[Cell.UP] and not c2.connect[Cell.DOWN] and
                        not c2.connect[Cell.LEFT]):
                        c.is_join_candidate = True
                        if random.random() <= 0.5:
                            c.connect[Cell.RIGHT] = True
    
    def create_tunnels(self) -> bool:
        """Create wrap-around tunnels"""
        single_dead_end_cells = []
        top_single_dead_end_cells = []
        bot_single_dead_end_cells = []
        
        void_tunnel_cells = []
        top_void_tunnel_cells = []
        bot_void_tunnel_cells = []
        
        edge_tunnel_cells = []
        top_edge_tunnel_cells = []
        bot_edge_tunnel_cells = []
        
        double_dead_end_cells = []
        
        for y in range(self.cell_map_size.y):
            c = self.cell_map[self.cell_map_size.x - 1][y]
            if c.connect[Cell.UP]:
                continue
            
            if c.position.y > 1 and c.position.y < self.cell_map_size.y - 2:
                c.is_edge_tunnel_candidate = True
                edge_tunnel_cells.append(c)
                if c.position.y <= 2:
                    top_edge_tunnel_cells.append(c)
                elif c.position.y >= self.cell_map_size.y - 4:
                    bot_edge_tunnel_cells.append(c)
            
            up_dead = (c.next[Cell.UP] is None or c.next[Cell.UP].connect[Cell.RIGHT])
            down_dead = (c.next[Cell.DOWN] is None or c.next[Cell.DOWN].connect[Cell.RIGHT])
            
            if c.connect[Cell.RIGHT]:
                if up_dead:
                    c.is_void_tunnel_candidate = True
                    void_tunnel_cells.append(c)
                    if c.position.y <= 2:
                        top_void_tunnel_cells.append(c)
                    elif c.position.y >= self.cell_map_size.y - 3:
                        bot_void_tunnel_cells.append(c)
            else:
                if c.connect[Cell.DOWN]:
                    continue
                
                if up_dead != down_dead:
                    if (not c.raise_height and y < self.cell_map_size.y - 1 and
                        not c.next[Cell.LEFT].connect[Cell.LEFT]):
                        single_dead_end_cells.append(c)
                        c.is_single_dead_end_candidate = True
                        c.single_dead_end_dir = Cell.UP if up_dead else Cell.DOWN
                        offset = 1 if up_dead else 0
                        if c.position.y <= 1 + offset:
                            top_single_dead_end_cells.append(c)
                        elif c.position.y >= self.cell_map_size.y - 4 + offset:
                            bot_single_dead_end_cells.append(c)
                elif up_dead and down_dead:
                    if y > 0 and y < self.cell_map_size.y - 1:
                        if (c.next[Cell.LEFT].connect[Cell.UP] and
                            c.next[Cell.LEFT].connect[Cell.DOWN]):
                            c.is_double_dead_end_candidate = True
                            if c.position.y >= 2 and c.position.y <= self.cell_map_size.y - 4:
                                double_dead_end_cells.append(c)
        
        num_tunnels_desired = 2 if random.random() <= 0.45 else 1
        
        if num_tunnels_desired == 1:
            if void_tunnel_cells:
                random.choice(void_tunnel_cells).top_tunnel = True
            elif single_dead_end_cells:
                self.select_single_dead_end(random.choice(single_dead_end_cells))
            elif edge_tunnel_cells:
                random.choice(edge_tunnel_cells).top_tunnel = True
            else:
                return False
        elif num_tunnels_desired == 2:
            if double_dead_end_cells:
                c = random.choice(double_dead_end_cells)
                c.connect[Cell.RIGHT] = True
                c.top_tunnel = True
                c.next[Cell.DOWN].top_tunnel = True
            else:
                num_tunnels_created = 1
                if top_void_tunnel_cells:
                    random.choice(top_void_tunnel_cells).top_tunnel = True
                elif top_single_dead_end_cells:
                    self.select_single_dead_end(random.choice(top_single_dead_end_cells))
                elif top_edge_tunnel_cells:
                    random.choice(top_edge_tunnel_cells).top_tunnel = True
                else:
                    num_tunnels_created = 0
                
                if bot_void_tunnel_cells:
                    random.choice(bot_void_tunnel_cells).top_tunnel = True
                elif bot_single_dead_end_cells:
                    self.select_single_dead_end(random.choice(bot_single_dead_end_cells))
                elif bot_edge_tunnel_cells:
                    random.choice(bot_edge_tunnel_cells).top_tunnel = True
                else:
                    if num_tunnels_created == 0:
                        return False
        
        # Verify tunnels don't create invalid exits
        for y in range(self.cell_map_size.y):
            c = self.cell_map[self.cell_map_size.x - 1][y]
            if c.top_tunnel:
                exit = True
                top_y = c.tile_position.y
                while c.next[Cell.LEFT] is not None:
                    c = c.next[Cell.LEFT]
                    if not c.connect[Cell.UP] and c.tile_position.y == top_y:
                        continue
                    else:
                        exit = False
                        break
                if exit:
                    return False
        
        # Connect void tunnels
        for c in void_tunnel_cells:
            if not c.top_tunnel:
                self.replace_group(c.group, c.next[Cell.UP].group)
                c.connect[Cell.UP] = True
                c.next[Cell.UP].connect[Cell.DOWN] = True
        
        return True
    
    def select_single_dead_end(self, c: Cell):
        """Select a single dead end for tunnel"""
        c.connect[Cell.RIGHT] = True
        if c.single_dead_end_dir == Cell.UP:
            c.top_tunnel = True
        else:
            c.next[Cell.DOWN].top_tunnel = True
    
    def replace_group(self, old_group: int, new_group: int):
        """Replace group numbers"""
        for j in range(self.cell_map_size.y):
            for i in range(self.cell_map_size.x):
                c = self.cell_map[i][j]
                if c.group == old_group:
                    c.group = new_group
    
    def export_to_graph_json(self, output_file: Optional[str] = None, tile_map: Optional[List[List]] = None) -> Dict:
        """
        Export maze as a graph representation in JSON format.
        
        Graph representation:
        - Nodes: All walkable tiles (simplified, no pellet/energizer distinction)
        - Edges: Adjacency between adjacent walkable tiles
        - Walls: Non-walkable tiles
        - Ghost walls: Ghost house door
        
        Args:
            output_file: Optional file path to save JSON. If None, returns dict.
            tile_map: Optional tile map to export. If None, generates a new one.
            
        Returns:
            Dictionary containing graph representation
        """
        # Step 1: Get tile map (use provided or generate new)
        if tile_map is None:
            tile_map = self.get_tile_map()
        height = len(tile_map[0])
        width = len(tile_map)
        
        # Simplified tile type names (no pellet/energizer distinction)
        tile_type_names = {
            Tile.BLANK: "BLANK",
            Tile.PATH: "PATH",
            Tile.PATHBLANK: "PATHBLANK",
            Tile.WALL: "WALL",
            Tile.GHOSTWALL: "GHOSTWALL",
            Tile.ENERGIZER: "PATH",  # Treat energizers as regular paths
            Tile.GHOSTSPACE: "GHOSTSPACE"
        }
        
        # Step 2: Identify all tiles with their types
        nodes: Dict[str, Dict] = {}  # id -> {id, x, y, type}
        walkable_positions: Set[Tuple[int, int]] = set()  # for quick lookup
        walls = []
        ghost_walls = []
        
        for y in range(height):
            for x in range(width):
                state = tile_map[x][y].state
                
                # Walkable tiles become nodes (exclude ghost house)
                if state in [Tile.PATH, Tile.PATHBLANK, Tile.ENERGIZER]:
                    node_id = f"{x},{y}"
                    nodes[node_id] = {
                        "id": node_id,
                        "x": x,
                        "y": y,
                        "type": tile_type_names[state]
                    }
                    walkable_positions.add((x, y))
                
                # Track walls
                elif state == Tile.WALL:
                    walls.append({"x": x, "y": y})
                elif state == Tile.GHOSTWALL:
                    ghost_walls.append({"x": x, "y": y})

        
        # Step 3: Build edges (adjacency between walkable tiles)
        edges_set: Set[Tuple[str, str]] = set()  # to deduplicate
        
        for node_id in nodes.keys():
            x, y = map(int, node_id.split(','))
            
            # Check all 4 adjacent neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < width and 0 <= ny < height:
                    # Check if neighbor is also walkable
                    if (nx, ny) in walkable_positions:
                        neighbor_id = f"{nx},{ny}"
                        # Create edge (use sorted tuple to handle deduplication)
                        edge = tuple(sorted([node_id, neighbor_id]))
                        edges_set.add(edge)
        
        # Step 4: Convert edge set to list
        edges = [{"from": edge[0], "to": edge[1]} for edge in edges_set]
        
        # Step 5: Build JSON structure
        graph_json = {
            "metadata": {
                "width": width,
                "height": height,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
            "nodes": list(nodes.values()),
            "edges": edges,
            "walls": walls,
            "ghost_walls": ghost_walls,
        }
        
        # Optional: save to file
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(graph_json, f, indent=2)
            print(f"Graph JSON exported to {output_file}")
        
        return graph_json


if __name__ == "__main__":
    # Test the maze generator
    print("Generating Pacman maze...")
    maze = PacmanMap()
    maze.generate()
    
    print(f"Generated after {maze.gen_count} attempts")
    
    # Print a simple ASCII representation
    tile_map = maze.get_tile_map()
    print(f"Tile map size: {len(tile_map)}x{len(tile_map[0])}")
    for j in range(len(tile_map[0])):
        for i in range(len(tile_map)):
            state = tile_map[i][j].state
            if state == Tile.WALL:
                print('█', end='')
            elif state == Tile.PATH:
                print('·', end='')
            elif state == Tile.ENERGIZER:
                print('●', end='')
            elif state == Tile.GHOSTWALL:
                print('─', end='')
            else:
                print(' ', end='')
        print()
    
    # Export to graph JSON
    print("\nExporting maze to graph JSON...")
    graph_json = maze.export_to_graph_json("maze_graph.json")
    print(f"Nodes: {graph_json['metadata']['node_count']}")
    print(f"Edges: {graph_json['metadata']['edge_count']}")
