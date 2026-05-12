"""Registry role discovery should expose all AI for benchmark selection."""

from src.agents import get_agent_names


def test_agent_registry_roles_cover_all_agents():
    assert set(get_agent_names("pacman")) == {
        "heuristic_pacman",
        "mcts_pacman",
        "random_pacman",
    }
    assert set(get_agent_names("ghosts")) == {
        "heuristic_ghosts",
        "iceguct",
        "random_ghosts",
    }


def test_agent_registry_names_default_to_all():
    names = get_agent_names()
    assert "mcts_pacman" in names
    assert "iceguct" in names
    assert names == sorted(names)