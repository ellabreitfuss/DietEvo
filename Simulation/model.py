"""
Core ABM logic for the food-transition model.

The model currently contains:

1. Intention formation
2. Norm updating through household and friendship networks
3. Attitude updating through planetary and individual health status
4. Individual health-status updating
5. Repeated simulation over time

No diet categories are defined.

Instead, agents are represented as points in a 2D intention space:

    x-axis = environmental / planetary-health intention
    y-axis = individual-health intention
"""

from dataclasses import dataclass
import numpy as np
import networkx as nx

from Simulation.agents import AgentState


@dataclass
class ModelParameters:
    """
    Stores all behavioral parameters of the ABM.

    intention_weight_attitude:
        Weight of attitudes in intention formation.

    intention_weight_norm:
        Weight of perceived norms in intention formation.

    intention_weight_PBC:
        Weight of perceived behavioral control in intention formation.

    household_weight:
        Importance of household members for norm updating.

    friendship_weight:
        Importance of friends for norm updating.

    norm_learning_rate:
        Speed at which perceived norms move toward neighbors' intentions.

    attitude_learning_rate:
        Speed at which attitudes respond to health system states.

    health_status_learning_rate:
        Speed at which individual health status responds to behavior.
    """

    intention_weight_attitude: float = 0.4
    intention_weight_norm: float = 0.4
    intention_weight_PBC: float = 0.2

    household_weight: float = 0.8
    friendship_weight: float = 0.2

    # learning rate instead transformation from intention to behavior?
    norm_learning_rate: float = 0.25
    attitude_learning_rate: float = 0.10
    health_status_learning_rate: float = 0.10


@dataclass
class FoodTransitionModel:
    """
    Full ABM object.

    agents:
        Agent-level variables.

    household_graph:
        NetworkX graph containing household ties.

    friendship_graph:
        NetworkX graph containing friendship ties.

    planetary_health_status:
        Global system-level variable.

        Scale:
            0 = very bad planetary health
            1 = very good planetary health
    """

    agents: AgentState
    household_graph: nx.Graph
    friendship_graph: nx.Graph
    parameters: ModelParameters
    planetary_health_status: float = 0.2


def calculate_intentions(model: FoodTransitionModel) -> None:
    """
    Calculate intention coordinates for every agent.

    Intention follows a TPB-style weighted sum:

        intention = 0.4 * attitude + 0.4 * norm + 0.2 * PBC

    This is calculated separately for:

        - environmental / planetary-health intention
        - individual-health intention
    """

    p = model.parameters
    a = model.agents

    a.intention_env = (
        p.intention_weight_attitude * a.env_attitudes
        + p.intention_weight_norm * a.env_norms
        + p.intention_weight_PBC * a.env_PBC
    )

    a.intention_health = (
        p.intention_weight_attitude * a.health_attitudes
        + p.intention_weight_norm * a.health_norms
        + p.intention_weight_PBC * a.health_PBC
    )


def average_neighbor_intention(
    graph: nx.Graph,
    intentions: np.ndarray,
    n_agents: int,
) -> np.ndarray:
    """
    Calculate the average intention of neighbors in one graph layer.

    """

    averages = np.zeros(n_agents)

    for agent in range(n_agents):
        neighbors = list(graph.neighbors(agent))

        if len(neighbors) == 0:
            averages[agent] = intentions[agent]
        else:
            averages[agent] = np.mean(intentions[neighbors])

    return averages

def update_norms(model: FoodTransitionModel) -> None:
    """
    Update perceived norms through household and friendship influence. This is where ABM and the social Network are interacting. 

    The model first calculates the average intention in each social layer:

        H_i = average household intention
        F_i = average friendship intention

    Then it combines them:

        neighbor_intention_i = 0.8 * H_i + 0.2 * F_i

    Finally, perceived norms move gradually toward that neighborhood intention:

        N_i(t+1) = (1 - alpha) * N_i(t) + alpha * neighbor_intention_i

    This is done separately for the environmental and health dimensions.
    """

    p = model.parameters
    a = model.agents
    n_agents = len(a.intention_env)

    household_env = average_neighbor_intention(
        model.household_graph,
        a.intention_env,
        n_agents,
    )

    friendship_env = average_neighbor_intention(
        model.friendship_graph,
        a.intention_env,
        n_agents,
    )

    neighbor_env = (
        p.household_weight * household_env
        + p.friendship_weight * friendship_env
    )

    household_health = average_neighbor_intention(
        model.household_graph,
        a.intention_health,
        n_agents,
    )

    friendship_health = average_neighbor_intention(
        model.friendship_graph,
        a.intention_health,
        n_agents,
    )

    neighbor_health = (
        p.household_weight * household_health
        + p.friendship_weight * friendship_health
    )

    alpha = p.norm_learning_rate

    a.env_norms = (1 - alpha) * a.env_norms + alpha * neighbor_env
    a.health_norms = (1 - alpha) * a.health_norms + alpha * neighbor_health

    a.env_norms = np.clip(a.env_norms, 0, 1)
    a.health_norms = np.clip(a.health_norms, 0, 1)


def update_attitudes(model: FoodTransitionModel) -> None:
    """
    Update attitudes based on planetary and individual health status.

    Environmental attitude:
        If planetary health is bad, environmental concern increases.

        planetary_concern = 1 - planetary_health_status

    Health attitude:
        If individual health status is bad, health concern increases.

        health_concern_i = 1 - individual_health_status_i
    """

    p = model.parameters
    a = model.agents

    mu = p.attitude_learning_rate

    planetary_concern = 1 - model.planetary_health_status
    health_concern = 1 - a.individual_health_status

    a.env_attitudes = (
        (1 - mu) * a.env_attitudes
        + mu * planetary_concern
    )

    a.health_attitudes = (
        (1 - mu) * a.health_attitudes
        + mu * health_concern
    )

    a.env_attitudes = np.clip(a.env_attitudes, 0, 1)
    a.health_attitudes = np.clip(a.health_attitudes, 0, 1)


def update_individual_health_status(model: FoodTransitionModel) -> None:
    """
    Update individual health status.

    For now, we assume that stronger environmental intention also represents
    healthier behavior.

    This is a simplifying assumption from your current model idea:

        behavior is represented by the environmental intention coordinate.

    Therefore:

        individual_health_status moves toward intention_env

    Later, this can be replaced by a more detailed behavior-health function.
    """

    p = model.parameters
    a = model.agents

    rho = p.health_status_learning_rate

    a.individual_health_status = (
        (1 - rho) * a.individual_health_status
        + rho * a.intention_env
    )

    a.individual_health_status = np.clip(a.individual_health_status, 0, 1)


def step(model: FoodTransitionModel) -> None:
    """
    Run one simulation step.

    Order of events:

    1. Calculate intentions from current attitudes, norms, and PBC.
    2. Update norms based on connected agents' intentions.
    3. Update health status based on behavior proxy.
    4. Update attitudes based on planetary and individual health status.
    5. Recalculate intentions after all updates.

    The final recalculation makes the stored intention values consistent
    with the updated state at the end of the step.
    """

    calculate_intentions(model)
    update_norms(model)
    update_individual_health_status(model)
    update_attitudes(model)
    calculate_intentions(model)


def run_model(
    model: FoodTransitionModel,
    n_steps: int,
) -> dict[str, list[float]]:
    """
    Run the ABM for multiple steps.

    Returns a history dictionary with aggregate values over time.
    """

    history = {
        "mean_intention_env": [],
        "mean_intention_health": [],
        "mean_env_attitude": [],
        "mean_health_attitude": [],
        "mean_env_norm": [],
        "mean_health_norm": [],
        "mean_individual_health_status": [],
    }

    for _ in range(n_steps):
        step(model)

        a = model.agents

        history["mean_intention_env"].append(float(np.mean(a.intention_env)))
        history["mean_intention_health"].append(float(np.mean(a.intention_health)))
        history["mean_env_attitude"].append(float(np.mean(a.env_attitudes)))
        history["mean_health_attitude"].append(float(np.mean(a.health_attitudes)))
        history["mean_env_norm"].append(float(np.mean(a.env_norms)))
        history["mean_health_norm"].append(float(np.mean(a.health_norms)))
        history["mean_individual_health_status"].append(
            float(np.mean(a.individual_health_status))
        )

    return history