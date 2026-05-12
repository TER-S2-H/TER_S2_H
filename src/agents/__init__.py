from .base import Agent, AgentFactory
from .random_baseline import RandomPacmanAgent, RandomGhostsAgent
from .heuristic_baseline import HeuristicPacman, HeuristicGhosts

# Registry: maps string name → factory callable
AGENT_REGISTRY: dict[str, AgentFactory] = {}
AGENT_ROLES: dict[str, str] = {}

def register(name: str, factory, role: str | None = None):
    AGENT_REGISTRY[name] = factory
    if role is not None:
        AGENT_ROLES[name] = role

def get_agent(name: str) -> "AgentFactory":
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]


def get_agent_names(role: str | None = None) -> list[str]:
    if role is None:
        return sorted(AGENT_REGISTRY)
    return sorted(name for name, agent_role in AGENT_ROLES.items() if agent_role == role)

# Register built-ins
register("random_pacman",    lambda: RandomPacmanAgent(), role="pacman")
register("random_ghosts",    lambda: RandomGhostsAgent(), role="ghosts")
register("heuristic_pacman", lambda: HeuristicPacman(), role="pacman")
register("heuristic_ghosts", lambda: HeuristicGhosts(), role="ghosts")

# MCTS / ICEgUCT agents
def _register_mcts():
    from .mcts_pacman import MCTSPacmanAgent
    from .mcts_ghosts import MCTSGhostsAgent
    from .iceguct_ghosts import IceGuctGhosts
    register("mcts_pacman",  lambda: MCTSPacmanAgent(),  role="pacman")
    register("mcts_ghosts",  lambda: MCTSGhostsAgent(),  role="ghosts")
    register("iceguct",      lambda: IceGuctGhosts(),    role="ghosts")

_register_mcts()

# JEPA agents (requires torch; silently skipped if unavailable)
def _register_jepa():
    try:
        from .jepa_agents import (
            JEPAHierPacmanAgent, JEPAL1PacmanAgent,
            JEPAHierGhostsAgent, JEPAL1GhostsAgent,
        )
        register("jepa_hier_pacman", lambda: JEPAHierPacmanAgent(), role="pacman")
        register("jepa_l1_pacman",   lambda: JEPAL1PacmanAgent(),   role="pacman")
        register("jepa_hier_ghosts", lambda: JEPAHierGhostsAgent(), role="ghosts")
        register("jepa_l1_ghosts",   lambda: JEPAL1GhostsAgent(),   role="ghosts")
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("jepa agents not registered: %s", exc)

_register_jepa()
