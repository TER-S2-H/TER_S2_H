# conftest.py — Configuration pytest
import glob
import sys
import os
import unittest.mock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.game.maze import Maze as _Maze
    _MAZE_DIR = os.path.join(os.path.dirname(__file__), "..", "generated_maze")

    def _find_mazes():
        paths = sorted(glob.glob(os.path.join(_MAZE_DIR, "**", "maze.json"), recursive=True))
        if not paths:
            paths = sorted(glob.glob(os.path.join(_MAZE_DIR, "*.json")))
        return paths

    @pytest.fixture(params=_find_mazes(), ids=lambda p: os.path.basename(os.path.dirname(p)))
    def maze(request):
        return _Maze.load(request.param)

    @pytest.fixture
    def example_maze():
        paths = _find_mazes()
        assert paths, "No maze files found under generated_maze/"
        return _Maze.load(paths[0])

except ImportError:
    pass  # Bierhoff game module not present — maze fixtures unavailable


@pytest.fixture(autouse=True, scope="session")
def mock_mongodb():
    """Mock MongoDB so JEPA API tests work without a live connection or dotenv."""
    mock_col = unittest.mock.MagicMock()
    mock_col.update_one.return_value = unittest.mock.MagicMock(matched_count=1)
    try:
        with unittest.mock.patch("app.get_collection", return_value=mock_col):
            yield mock_col
    except (ModuleNotFoundError, ImportError):
        yield mock_col
