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


from dataclasses import dataclass, field
import numpy as np
import networkx as nx

from Simulation.agents import AgentState


@dataclass
class ModelParameters:
    """
    Behavioral and system parameters.
    """

    # Theory of Planned Behavior
    intention_weight_attitude: float = 0.40
    intention_weight_norm: float =  0.25
    intention_weight_PBC: float = 0.35

    # Intention-behavior thresholds
    threshold_health: float =  0.08  #0.08 if low: IH status higher than env status
    threshold_env: float =  0.1    #0.10 if high: IH higher

    # Attitude update weights
    attitude_weight_health: float = 0.10
    attitude_weight_env: float =  0.10

    # Norm update weights
    household_weight: float = 0.80 #change doesnt change the outcome if random 
    friendship_weight: float = 0.20
    motivation_to_comply: float = 0.25   #0.80   #influence on norms

    # Health-status update weights
    individual_health_weight_previous: float = 0.50
    individual_health_weight_behavior: float = 0.30
    individual_health_weight_environment: float = 0.15
    individual_health_weight_noise: float = 0.05

    # Environmental-health update weights
    env_health_weight_previous: float =  0.60
    env_health_weight_behavior: float = 0.35
    env_health_weight_noise: float =  0.05

    # Noise
    noise_mean: float = 0.5   #0.5 if over 0 EH and PH are bigger 
    noise_sd: float =  0.1  #0.1


@dataclass
class FoodTransitionModel:
    """
    Full ABM object.
    """

    agents: AgentState
    household_graph: nx.Graph
    friendship_graph: nx.Graph
    parameters: ModelParameters
    environmental_health_status: float = 0.2

    # Needed because your environmental attitude equation compares EH_t to EH_t-5.
    environmental_health_history: list[float] = field(default_factory=list)


def validate_parameters(p: ModelParameters) -> None:
    """
    Validate parameter consistency.
    """

    if not np.isclose(
        p.intention_weight_attitude
        + p.intention_weight_norm
        + p.intention_weight_PBC,
        1.0,
    ):
        raise ValueError("Intention weights must sum to 1.")

    if not np.isclose(p.household_weight + p.friendship_weight, 1.0):
        raise ValueError("Household and friendship weights must sum to 1.")

    if not np.isclose(
        p.individual_health_weight_previous
        + p.individual_health_weight_behavior
        + p.individual_health_weight_environment
        + p.individual_health_weight_noise,
        1.0,
    ):
        raise ValueError("Individual health-status weights must sum to 1.")

    if not np.isclose(
        p.env_health_weight_previous
        + p.env_health_weight_behavior
        + p.env_health_weight_noise,
        1.0,
    ):
        raise ValueError("Environmental health-status weights must sum to 1.")


def calculate_intentions(model: FoodTransitionModel) -> None:
    """
    Calculate intention from attitude, norm, and PBC.

    I = w_A * A + w_N * N + w_PBC * PBC
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

    a.intention_env = np.clip(a.intention_env, 0, 1)
    a.intention_health = np.clip(a.intention_health, 0, 1)


def initialize_behavior_from_intention(model: FoodTransitionModel) -> None:
    """
    Initial behavior calculation.

    According to your team decision:

    B(0) = I(0) * PBC
    """

    a = model.agents

    a.behavior_env = a.intention_env * a.env_PBC
    a.behavior_health = a.intention_health * a.health_PBC

    a.behavior_env = np.clip(a.behavior_env, 0, 1)
    a.behavior_health = np.clip(a.behavior_health, 0, 1)

def initialize_individual_health_status(model: FoodTransitionModel) -> None:
    """
    Calculate initial individual health status from initial behavior,
    environmental health status, and noise.

    This avoids randomly setting IH at the beginning.

    Initial version:

        IH_0 =
            w_behavior * B_IH,0
            + w_environment * EH_0
            + w_noise * noise

    The previous-health term is not used at initialization because
    there is no IH_{t-1} yet.
    """

    p = model.parameters
    a = model.agents

    rng = np.random.default_rng()
    noise = rng.normal(
        p.noise_mean,
        p.noise_sd,
        size=len(a.behavior_health),
    )
    noise = np.clip(noise, 0, 1)

    total_weight = (
        p.individual_health_weight_behavior
        + p.individual_health_weight_environment
        + p.individual_health_weight_noise
    )

    a.individual_health_status = (
        p.individual_health_weight_behavior * a.behavior_health
        + p.individual_health_weight_environment * model.environmental_health_status
        + p.individual_health_weight_noise * noise
    ) / total_weight

    a.individual_health_status = np.clip(a.individual_health_status, 0, 1)


def initialize_model_state(model: FoodTransitionModel) -> None:
    """
    Initialize model state before the first simulation step.

    Order:
    1. Validate parameters.
    2. Store initial environmental health.
    3. Calculate initial intentions.
    4. Calculate initial behavior.
    5. Calculate initial individual health status.

    Important:
    Individual health is calculated.
    """

    validate_parameters(model.parameters)

    calculate_intentions(model)
    initialize_behavior_from_intention(model)
    initialize_individual_health_status(model)

def average_neighbor_value(
    graph: nx.Graph,
    values: np.ndarray,
    n_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate average neighbor value for one network layer.

    Returns
    -------
    averages:
        Average value of neighbors.

    has_neighbors:
        Boolean array.
        True if the agent has at least one neighbor in this layer.
        False if the agent has no neighbors in this layer.
    """

    averages = np.zeros(n_agents)
    has_neighbors = np.zeros(n_agents, dtype=bool)

    for agent in range(n_agents):
        neighbors = list(graph.neighbors(agent))

        if len(neighbors) > 0:
            averages[agent] = np.mean(values[neighbors])
            has_neighbors[agent] = True

    return averages, has_neighbors


def update_norms(model: FoodTransitionModel) -> None:
    """
    Update subjective norms from connected agents' observed behavior.

    Important:
    Norms are updated from behavior, not intention.

    If an agent has no household and no friendship neighbors,
    its norm remains unchanged.
    """

    p = model.parameters
    a = model.agents
    n_agents = len(a.env_norms)

    h_env, has_h_env = average_neighbor_value(
        model.household_graph,
        a.behavior_env,
        n_agents,
    )

    f_env, has_f_env = average_neighbor_value(
        model.friendship_graph,
        a.behavior_env,
        n_agents,
    )

    h_health, has_h_health = average_neighbor_value(
        model.household_graph,
        a.behavior_health,
        n_agents,
    )

    f_health, has_f_health = average_neighbor_value(
        model.friendship_graph,
        a.behavior_health,
        n_agents,
    )

    has_any_env_neighbor = has_h_env | has_f_env
    has_any_health_neighbor = has_h_health | has_f_health

    descriptive_norm_env = (
        p.household_weight * h_env
        + p.friendship_weight * f_env
    )

    descriptive_norm_health = (
        p.household_weight * h_health
        + p.friendship_weight * f_health
    )

    m = p.motivation_to_comply

    new_env_norms = a.env_norms.copy()
    new_health_norms = a.health_norms.copy()

    new_env_norms[has_any_env_neighbor] = (
        (1 - m) * a.env_norms[has_any_env_neighbor]
        + m * descriptive_norm_env[has_any_env_neighbor]
    )

    new_health_norms[has_any_health_neighbor] = (
        (1 - m) * a.health_norms[has_any_health_neighbor]
        + m * descriptive_norm_health[has_any_health_neighbor]
    )

    a.env_norms = np.clip(new_env_norms, 0, 1)
    a.health_norms = np.clip(new_health_norms, 0, 1)


def update_individual_health_status(model: FoodTransitionModel) -> None:
    """
    Update individual health status.

    IH_t =
        w0 * IH_t-1
        + w1 * B_IH,t-1
        + w2 * EH_t
        + w3 * noise
    """

    p = model.parameters
    a = model.agents

    rng = np.random.default_rng()
    noise = rng.normal(p.noise_mean, p.noise_sd, size=len(a.individual_health_status))
    noise = np.clip(noise, 0, 1)

    a.individual_health_status = (
        p.individual_health_weight_previous * a.individual_health_status
        + p.individual_health_weight_behavior * a.behavior_health
        + p.individual_health_weight_environment * model.environmental_health_status
        + p.individual_health_weight_noise * noise
    )

    a.individual_health_status = np.clip(a.individual_health_status, 0, 1)


def update_environmental_health_status(model: FoodTransitionModel) -> None:
    """
    Update global environmental health status.

    EH_t =
        w0 * EH_t-1
        + w1 * mean(B_EH)
        + w2 * noise

    """

    p = model.parameters
    a = model.agents

    rng = np.random.default_rng()
    noise = float(np.clip(rng.normal(p.noise_mean, p.noise_sd), 0, 1))

    mean_env_behavior = float(np.mean(a.behavior_env))

    model.environmental_health_status = (
        p.env_health_weight_previous * model.environmental_health_status
        + p.env_health_weight_behavior * mean_env_behavior
        + p.env_health_weight_noise * noise
    )

    model.environmental_health_status = float(
        np.clip(model.environmental_health_status, 0, 1)
    )

    model.environmental_health_history.append(model.environmental_health_status)


def update_attitudes(model: FoodTransitionModel) -> None:
    """
    Update individual-health and environmental attitudes.

    Individual-health attitude
    --------------------------
    The IH attitude follows:

        A_IH,i,t =
            A_IH,i,t-1
            + w_A,IH
              * (1 - A_IH,i,t-1)
              * (1 - IH_i,t)

    Interpretation:
        - Poor individual health increases health concern.
        - Good individual health causes little or no increase.
        - Health attitude cannot decrease.

    Environmental attitude
    ----------------------
    Environmental attitude follows a target-seeking equation:

        A_EH,i,t =
            A_EH,i,t-1
            + w_A,EH
              * ((1 - EH_t) - A_EH,i,t-1)

    Interpretation:
        - Poor environmental health increases environmental concern.
        - Improving environmental health reduces environmental concern.
    """

    p = model.parameters
    a = model.agents

    # ==========================================================
    # Individual-health attitudes
    # ==========================================================

    a.health_attitudes = (
        a.health_attitudes
        + p.attitude_weight_health
        * (1.0 - a.health_attitudes)
        * (1.0 - a.individual_health_status)
    )

    # ==========================================================
    # Environmental attitudes
    # ==========================================================

    target_env_attitude = 1.0 - model.environmental_health_status

    a.env_attitudes = (
        a.env_attitudes
        + p.attitude_weight_env
        * (target_env_attitude - a.env_attitudes)
    )

    # Keep values within [0, 1]
    a.health_attitudes = np.clip(a.health_attitudes, 0.0, 1.0)
    a.env_attitudes = np.clip(a.env_attitudes, 0.0, 1.0)


def update_behavior(model: FoodTransitionModel) -> None:
    """
    Update dietary behavior with threshold rule.

    Team decision:

    B_target = I * PBC

    Behavior only changes if the difference between previous behavior
    and current intention is large enough.

    For implementation, we use the target behavior in the comparison:

        abs(B_previous - B_target) >= threshold

    This is more internally consistent than comparing B_previous to I only,
    because actual target behavior includes PBC.
    """

    p = model.parameters
    a = model.agents

    target_env = a.intention_env * a.env_PBC
    target_health = a.intention_health * a.health_PBC

    change_env = np.abs(a.behavior_env - target_env) >= p.threshold_env
    change_health = np.abs(a.behavior_health - target_health) >= p.threshold_health

    a.behavior_env[change_env] = target_env[change_env]
    a.behavior_health[change_health] = target_health[change_health]

    a.behavior_env = np.clip(a.behavior_env, 0, 1)
    a.behavior_health = np.clip(a.behavior_health, 0, 1)

def step(model: FoodTransitionModel) -> None:
    """
    Run one model step.

    Order:
    1. Environmental health status
    2. Individual health status
    3. Attitudes
    4. Norms
    5. Intentions
    6. Behavior
    """

    update_environmental_health_status(model)
    update_individual_health_status(model)
    update_attitudes(model)
    update_norms(model)
    calculate_intentions(model)
    update_behavior(model)


def run_model(model: FoodTransitionModel, n_steps: int) -> dict[str, list[float]]:
    """
    Run model for n_steps and store aggregate outputs.
    """

    initialize_model_state(model)

    history = {
        "environmental_health_status": [],
        "mean_individual_health_status": [],
        "mean_env_attitude": [],
        "mean_health_attitude": [],
        "mean_env_norm": [],
        "mean_health_norm": [],
        "mean_intention_env": [],
        "mean_intention_health": [],
        "mean_behavior_env": [],
        "mean_behavior_health": [],
    }

    for _ in range(n_steps):
        step(model)

        a = model.agents

        history["environmental_health_status"].append(
            model.environmental_health_status
        )
        history["mean_individual_health_status"].append(
            float(np.mean(a.individual_health_status))
        )
        history["mean_env_attitude"].append(float(np.mean(a.env_attitudes)))
        history["mean_health_attitude"].append(float(np.mean(a.health_attitudes)))
        history["mean_env_norm"].append(float(np.mean(a.env_norms)))
        history["mean_health_norm"].append(float(np.mean(a.health_norms)))
        history["mean_intention_env"].append(float(np.mean(a.intention_env)))
        history["mean_intention_health"].append(float(np.mean(a.intention_health)))
        history["mean_behavior_env"].append(float(np.mean(a.behavior_env)))
        history["mean_behavior_health"].append(float(np.mean(a.behavior_health)))

    return history