"""
Household network generation for the food-transition ABM.

Purpose
-------
This module creates a synthetic household layer for an agent-based model.

The household layer represents strong social influence between people who
live together. In the dietary-transition model, household ties are important
because household members often share:

- food purchases
- cooking routines
- meals
- dietary habits
- norms around acceptable foods

The output of this module is a NetworkX graph in which:
- each node represents one agent
- each edge connects two agents in the same household
- each household is represented as a fully connected clique
- each household edge has a fixed influence weight

Example
-------
If household = [0, 3, 7], then the graph contains:

0 -- 3
0 -- 7
3 -- 7

All three agents can influence each other through the household layer.
"""

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np


@dataclass
class HouseholdNetwork:
    """
    Container for the generated household network.

    Attributes
    ----------
    households:
        List of households.

        Each household is represented as a list of agent IDs.

        Example:
            [
                [0, 4],
                [1],
                [2, 3, 5]
            ]

        This means:
            - household 0 contains agents 0 and 4
            - household 1 contains agent 1
            - household 2 contains agents 2, 3, and 5

    graph:
        NetworkX graph representing household ties.

        Nodes:
            Agent IDs from 0 to n_agents - 1.

        Edges:
            Undirected links between agents in the same household.

        Edge attributes:
            weight:
                Strength of household influence.

            layer:
                Always set to "household" for this module.

            household_id:
                ID of the household to which the edge belongs.

    household_id:
        Dictionary mapping each agent ID to the household ID.

        Example:
            {
                0: 0,
                4: 0,
                1: 1,
                2: 2,
                3: 2,
                5: 2
            }
    """

    households: List[List[int]]
    graph: nx.Graph
    household_id: Dict[int, int]


def generate_household_sizes(
    n_agents: int,
    household_size_probs: Dict[int, float],
    seed: int | None = None,
) -> List[int]:
    """
    Generate a list of household sizes.

    The function repeatedly samples household sizes from a probability
    distribution until all agents can be assigned to a household.

    Parameters
    ----------
    n_agents:
        Total number of agents in the synthetic population.

        Must be a positive integer.

    household_size_probs:
        Probability distribution over possible household sizes.

        Example:
            {
                1: 0.30,
                2: 0.35,
                3: 0.15,
                4: 0.15,
                5: 0.05
            }

        Keys:
            Household sizes.

        Values:
            Relative probabilities.

        The probabilities do not need to sum exactly to 1.
        They are normalized internally.

    seed:
        Optional random seed for reproducibility.

        Use the same seed to generate the same household size sequence.

    Returns
    -------
    household_sizes:
        List of sampled household sizes.

        The sum of this list is exactly equal to n_agents.

    Notes
    -----
    If the final sampled household would exceed n_agents, its size is reduced
    so that the total population size is exactly n_agents.

    Example
    -------
    If n_agents = 10 and sampled sizes are:

        [3, 4, 4]

    the final household is truncated to size 3, giving:

        [3, 4, 3]
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    if not household_size_probs:
        raise ValueError("household_size_probs must not be empty.")

    rng = np.random.default_rng(seed)

    sizes = np.array(list(household_size_probs.keys()), dtype=int)
    probs = np.array(list(household_size_probs.values()), dtype=float)

    if np.any(sizes <= 0):
        raise ValueError("All household sizes must be positive integers.")

    if np.any(probs < 0):
        raise ValueError("Household size probabilities must be non-negative.")

    if probs.sum() == 0:
        raise ValueError("At least one household size probability must be positive.")

    probs = probs / probs.sum()

    household_sizes: List[int] = []
    assigned_agents = 0

    while assigned_agents < n_agents:
        sampled_size = int(rng.choice(sizes, p=probs))

        remaining_agents = n_agents - assigned_agents

        household_size = min(sampled_size, remaining_agents)

        household_sizes.append(household_size)
        assigned_agents += household_size

    return household_sizes


def assign_agents_to_households(
    n_agents: int,
    household_sizes: List[int],
    seed: int | None = None,
) -> List[List[int]]:
    """
    Randomly assign agents to households of predefined sizes.

    Parameters
    ----------
    n_agents:
        Total number of agents.

        Agents are represented by integer IDs:

            0, 1, 2, ..., n_agents - 1

    household_sizes:
        List of household sizes.

        The sum of this list must equal n_agents.

        Example:
            [2, 1, 3]

        means:
            - first household has 2 agents
            - second household has 1 agent
            - third household has 3 agents

    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    households:
        List of households, where each household is a list of agent IDs.

    Example
    -------
    household_sizes = [2, 1, 3]

    Possible output:

        [
            [4, 0],
            [2],
            [1, 5, 3]
        ]

    Notes
    -----
    Agent assignment is random. The household sizes are fixed, but the specific
    agents inside each household depend on the random shuffle.
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    if sum(household_sizes) != n_agents:
        raise ValueError("The sum of household_sizes must equal n_agents.")

    if any(size <= 0 for size in household_sizes):
        raise ValueError("All household sizes must be positive integers.")

    rng = np.random.default_rng(seed)

    agents = np.arange(n_agents)
    rng.shuffle(agents)

    households: List[List[int]] = []
    start_index = 0

    for household_size in household_sizes:
        end_index = start_index + household_size
        household = agents[start_index:end_index].tolist()
        households.append(household)
        start_index = end_index

    return households


def build_household_graph(
    n_agents: int,
    households: List[List[int]],
    weight: float = 0.8,
) -> tuple[nx.Graph, Dict[int, int]]:
    """
    Build the household graph from household assignments.

    Each household is converted into a fully connected subgraph.

    Parameters
    ----------
    n_agents:
        Total number of agents.

    households:
        List of households.

        Each household is a list of agent IDs.

    weight:
        Influence weight assigned to every household edge.

        Higher values mean stronger household influence.

        Example interpretation:
            weight = 0.8 means household influence is strong.

    Returns
    -------
    graph:
        Undirected NetworkX graph containing household ties.

    household_id:
        Dictionary mapping each agent to its household ID.

    Edge Construction
    -----------------
    A household with size n creates:

        n * (n - 1) / 2

    undirected edges.

    Examples:
        size 1 household -> 0 edges
        size 2 household -> 1 edge
        size 3 household -> 3 edges
        size 4 household -> 6 edges

    Notes
    -----
    Single-person households generate no edges but are still included as nodes.
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    if weight < 0:
        raise ValueError("weight must be non-negative.")

    graph = nx.Graph()
    graph.add_nodes_from(range(n_agents))

    household_id: Dict[int, int] = {}

    for h_id, household in enumerate(households):
        for agent in household:
            if agent < 0 or agent >= n_agents:
                raise ValueError(f"Invalid agent ID: {agent}")

            if agent in household_id:
                raise ValueError(f"Agent {agent} appears in more than one household.")

            household_id[agent] = h_id

        for i, agent_a in enumerate(household):
            for agent_b in household[i + 1:]:
                graph.add_edge(
                    agent_a,
                    agent_b,
                    weight=weight,
                    layer="household",
                    household_id=h_id,
                )

    if len(household_id) != n_agents:
        raise ValueError("Not all agents were assigned to a household.")

    return graph, household_id


def create_household_network(
    n_agents: int,
    household_size_probs: Dict[int, float] | None = None,
    weight: float = 0.8,
    seed: int | None = None,
) -> HouseholdNetwork:
    """
    Create a complete synthetic household network.

    This is the main function you will usually call from outside this module.

    It performs three steps:

    1. Generate household sizes.
    2. Randomly assign agents to households.
    3. Build the household graph.

    Parameters
    ----------
    n_agents:
        Total number of agents.

    household_size_probs:
        Probability distribution over household sizes.

        If None, the following default distribution is used:

            1-person household: 30%
            2-person household: 35%
            3-person household: 15%
            4-person household: 15%
            5-person household: 5%

        These are placeholder values.
        For a research model, replace them with empirical household-size
        data from the target country or region.

    weight:
        Influence weight assigned to household ties.

        This weight will later be combined with friendship or other social
        network layers.

    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    HouseholdNetwork:
        Dataclass containing:
            - households
            - graph
            - household_id
    """

    if household_size_probs is None:
        household_size_probs = {
            1: 0.30,
            2: 0.35,
            3: 0.15,
            4: 0.15,
            5: 0.05,
        }

    household_sizes = generate_household_sizes(
        n_agents=n_agents,
        household_size_probs=household_size_probs,
        seed=seed,
    )

    households = assign_agents_to_households(
        n_agents=n_agents,
        household_sizes=household_sizes,
        seed=seed,
    )

    graph, household_id = build_household_graph(
        n_agents=n_agents,
        households=households,
        weight=weight,
    )

    return HouseholdNetwork(
        households=households,
        graph=graph,
        household_id=household_id,
    )


def summarize_households(households: List[List[int]]) -> Dict[str, float]:
    """
    Calculate summary statistics for generated households.

    Parameters
    ----------
    households:
        List of generated households.

    Returns
    -------
    summary:
        Dictionary with basic household statistics.

    Contains:
        n_households:
            Number of households.

        mean_size:
            Average household size.

        min_size:
            Smallest household size.

        max_size:
            Largest household size.

        share_single_person:
            Share of households with exactly one person.
    """

    household_sizes = np.array([len(household) for household in households])

    return {
        "n_households": float(len(households)),
        "mean_size": float(np.mean(household_sizes)),
        "min_size": float(np.min(household_sizes)),
        "max_size": float(np.max(household_sizes)),
        "share_single_person": float(np.mean(household_sizes == 1)),
    }


if __name__ == "__main__":
    household_network = create_household_network(
        n_agents=1000,
        weight=0.8,
        seed=42,
    )

    summary = summarize_households(household_network.households)

    print("Household network summary")
    print("-------------------------")
    print(f"Number of agents: {household_network.graph.number_of_nodes()}")
    print(f"Number of households: {int(summary['n_households'])}")
    print(f"Number of household ties: {household_network.graph.number_of_edges()}")
    print(f"Average household size: {summary['mean_size']:.2f}")
    print(f"Minimum household size: {int(summary['min_size'])}")
    print(f"Maximum household size: {int(summary['max_size'])}")
    print(f"Share single-person households: {summary['share_single_person']:.2%}")
