"""
Friendship network generation for the food-transition ABM.

Purpose
-------
This module creates a synthetic friendship layer.

The friendship layer represents medium-strength social influence between agents
who are socially connected but do not necessarily live together.

In the dietary-transition model, friendship ties matter because friends can
influence:

- perceived social norms
- attitudes toward plant-based diets
- exposure to new meals or ideas
- identity-related food choices
- willingness to try dietary change

This module keeps friendship ties separate from household ties.
They will later be combined in multiplex.py.
"""

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np


@dataclass
class FriendshipNetwork:
    """
    Container for the generated friendship network.

    Attributes
    ----------
    graph:
        NetworkX graph representing friendship ties.

        Nodes:
            Agent IDs from 0 to n_agents - 1.

        Edges:
            Undirected friendship links.

        Edge attributes:
            weight:
                Strength of friendship influence.

            layer:
                Always set to "friendship".

    degree:
        Dictionary mapping each agent ID to its number of friends.

        Example:
            {
                0: 6,
                1: 8,
                2: 4
            }
    """

    graph: nx.Graph
    degree: Dict[int, int]


def create_friendship_network(
    n_agents: int,
    average_friends: int = 8,
    rewiring_probability: float = 0.1,
    weight: float = 0.3,
    seed: int | None = None,
) -> FriendshipNetwork:
    """
    Create a synthetic friendship network using the Watts-Strogatz small-world model.

    The Watts-Strogatz model is useful because real friendship networks often have:

    - local clusters
    - short paths between people
    - moderate randomness
    - more structure than a purely random graph

    Parameters
    ----------
    n_agents:
        Total number of agents.

    average_friends:
        Approximate number of friendship ties per agent.

        In NetworkX's Watts-Strogatz implementation, this corresponds to k,
        the number of nearest neighbors each node is initially connected to.

        Important:
            This value must be even.
            If an odd number is provided, the function increases it by 1.

    rewiring_probability:
        Probability of rewiring each edge.

        Interpretation:
            0.0 = completely regular local network
            1.0 = highly random network

        A value around 0.1 is a common starting point for small-world networks.

    weight:
        Influence weight assigned to each friendship edge.

        This should usually be lower than the household weight.

        Example:
            household weight = 0.8
            friendship weight = 0.3

    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    FriendshipNetwork:
        Dataclass containing:
            - graph
            - degree dictionary
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    if average_friends <= 0:
        raise ValueError("average_friends must be positive.")

    if average_friends >= n_agents:
        raise ValueError("average_friends must be smaller than n_agents.")

    if not 0 <= rewiring_probability <= 1:
        raise ValueError("rewiring_probability must be between 0 and 1.")

    if weight < 0:
        raise ValueError("weight must be non-negative.")

    if average_friends % 2 != 0:
        average_friends += 1

    graph = nx.watts_strogatz_graph(
        n=n_agents,
        k=average_friends,
        p=rewiring_probability,
        seed=seed,
    )

    for agent_a, agent_b in graph.edges():
        graph[agent_a][agent_b]["weight"] = weight
        graph[agent_a][agent_b]["layer"] = "friendship"

    degree = dict(graph.degree())

    return FriendshipNetwork(
        graph=graph,
        degree=degree,
    )


def summarize_friendships(graph: nx.Graph) -> Dict[str, float]:
    """
    Calculate summary statistics for the friendship network.

    Parameters
    ----------
    graph:
        NetworkX friendship graph.

    Returns
    -------
    summary:
        Dictionary with basic friendship-network statistics.

    Contains:
        n_agents:
            Number of nodes.

        n_friendship_ties:
            Number of friendship edges.

        mean_degree:
            Average number of friends per agent.

        min_degree:
            Smallest number of friends.

        max_degree:
            Largest number of friends.

        clustering:
            Average clustering coefficient.

        connected:
            Whether the graph is fully connected.
    """

    degrees = np.array([degree for _, degree in graph.degree()])

    return {
        "n_agents": float(graph.number_of_nodes()),
        "n_friendship_ties": float(graph.number_of_edges()),
        "mean_degree": float(np.mean(degrees)),
        "min_degree": float(np.min(degrees)),
        "max_degree": float(np.max(degrees)),
        "clustering": float(nx.average_clustering(graph)),
        "connected": float(nx.is_connected(graph)),
    }


if __name__ == "__main__":
    friendship_network = create_friendship_network(
        n_agents=1000,
        average_friends=8,
        rewiring_probability=0.1,
        weight=0.3,
        seed=42,
    )

    summary = summarize_friendships(friendship_network.graph)

    print("Friendship network summary")
    print("--------------------------")
    print(f"Number of agents: {int(summary['n_agents'])}")
    print(f"Number of friendship ties: {int(summary['n_friendship_ties'])}")
    print(f"Average friends per agent: {summary['mean_degree']:.2f}")
    print(f"Minimum friends: {int(summary['min_degree'])}")
    print(f"Maximum friends: {int(summary['max_degree'])}")
    print(f"Average clustering: {summary['clustering']:.3f}")
    print(f"Connected: {bool(summary['connected'])}")
    