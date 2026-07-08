"""
Agent initialization for the food-transition ABM.

Each node in the social network represents one agent.

Each agent has six Theory-of-Planned-Behavior-style variables:

Environmental / planetary-health dimension:
    - env_attitudes
    - env_norms
    - env_PBC

Individual-health dimension:
    - health_attitudes
    - health_norms
    - health_PBC

All variables are continuous values between 0 and 1.

Interpretation:
    0 = very low
    1 = very high

The model later combines these variables into two intention coordinates:

    intention_env
    intention_health

These coordinates can be plotted in a 2D space.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentState:
    """
    Stores all agent-level variables as NumPy arrays.

    Each array has length n_agents.
    The value at index i belongs to agent i.
    """

    env_attitudes: np.ndarray
    env_norms: np.ndarray
    env_PBC: np.ndarray

    health_attitudes: np.ndarray
    health_norms: np.ndarray
    health_PBC: np.ndarray

    intention_env: np.ndarray
    intention_health: np.ndarray

    individual_health_status: np.ndarray


def random_grid_values(
    n_agents: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate random values between 0 and 1 in steps of 0.1.

    Example possible values:
        0.0, 0.1, 0.2, ..., 1.0
    """

    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
    return rng.choice(grid, size=n_agents)


def initialize_agents(
    n_agents: int,
    seed: int | None = None,
) -> AgentState:
    """
    Initialize all agent variables randomly.

    Parameters
    ----------
    n_agents:
        Number of agents in the model.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    AgentState
        Dataclass containing all initialized agent variables.
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    rng = np.random.default_rng(seed)

    env_attitudes = random_grid_values(n_agents, seed=rng.integers(1_000_000))
    env_norms = random_grid_values(n_agents, seed=rng.integers(1_000_000))
    env_PBC = random_grid_values(n_agents, seed=rng.integers(1_000_000))

    health_attitudes = random_grid_values(n_agents, seed=rng.integers(1_000_000))
    health_norms = random_grid_values(n_agents, seed=rng.integers(1_000_000))
    health_PBC = random_grid_values(n_agents, seed=rng.integers(1_000_000))

    intention_env = np.zeros(n_agents)
    intention_health = np.zeros(n_agents)

    # Initial individual health status.
    # For now, this is random. Later, we can initialize it from empirical data.
    individual_health_status = random_grid_values(
        n_agents,
        seed=rng.integers(1_000_000),
    )

    return AgentState(
        env_attitudes=env_attitudes,
        env_norms=env_norms,
        env_PBC=env_PBC,
        health_attitudes=health_attitudes,
        health_norms=health_norms,
        health_PBC=health_PBC,
        intention_env=intention_env,
        intention_health=intention_health,
        individual_health_status=individual_health_status,
    )