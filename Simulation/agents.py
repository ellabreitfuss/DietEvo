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
    env_attitudes: np.ndarray
    env_norms: np.ndarray
    env_PBC: np.ndarray

    health_attitudes: np.ndarray
    health_norms: np.ndarray
    health_PBC: np.ndarray

    intention_env: np.ndarray
    intention_health: np.ndarray

    behavior_env: np.ndarray
    behavior_health: np.ndarray

    individual_health_status: np.ndarray


def random_grid_values(n_agents: int, seed: int | None = None) -> np.ndarray:
    """
    Generate random values from:
    0.0, 0.1, 0.2, ..., 1.0
    """

    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
    return rng.choice(grid, size=n_agents)


def initialize_agents(n_agents: int, seed: int | None = None) -> AgentState:
    """
    Initialize agents with normally distributed attitudes and norms.

    Attitudes and norms are sampled from a truncated normal distribution
    centered at 0.5 (σ = 0.2), clipped to [0, 1], and rounded to the
    nearest 0.1.
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be positive.")

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # Initial attitude and norm distribution
    # ----------------------------------------------------------

    INITIAL_MEAN = 0.2
    INITIAL_STD = 0.1
    GRID = np.arange(0.0, 1.01, 0.1)

    def sample_initial_values() -> np.ndarray:
        """Sample values from a truncated Gaussian distribution."""
        values = np.clip(
            rng.normal(
                loc=INITIAL_MEAN,
                scale=INITIAL_STD,
                size=n_agents,
            ),
            0.0,
            1.0,
        )

        # Map every value to the nearest grid point (0.0, 0.1, ..., 1.0)
        return GRID[np.abs(values[:, None] - GRID).argmin(axis=1)]

    env_attitudes = sample_initial_values()
    env_norms = sample_initial_values()

    health_attitudes = sample_initial_values()
    health_norms = sample_initial_values()

    # ----------------------------------------------------------
    # Perceived behavioural control
    # ----------------------------------------------------------

    env_PBC = np.full(n_agents, 0.5)
    health_PBC = np.full(n_agents, 0.8)

    # ----------------------------------------------------------
    # Dynamic variables
    # ----------------------------------------------------------

    intention_env = np.zeros(n_agents)
    intention_health = np.zeros(n_agents)

    behavior_env = np.zeros(n_agents)
    behavior_health = np.zeros(n_agents)

    # Calculated during model initialization
    individual_health_status = np.zeros(n_agents)

    return AgentState(
        env_attitudes=env_attitudes,
        env_norms=env_norms,
        env_PBC=env_PBC,
        health_attitudes=health_attitudes,
        health_norms=health_norms,
        health_PBC=health_PBC,
        intention_env=intention_env,
        intention_health=intention_health,
        behavior_env=behavior_env,
        behavior_health=behavior_health,
        individual_health_status=individual_health_status,
    )