"""
Multiplex network construction for the food-transition ABM.

Purpose
-------
This module combines separate social network layers into one weighted network.

Current layers:
- household network
- friendship network

The model keeps layers separate for documentation and interpretation, but
combines them into one graph and one influence matrix for simulation.

The final influence matrix W can be used for:

- Friedkin-Johnsen opinion updates
- perceived social norm calculation
- neighborhood exposure calculations
"""

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np


@dataclass
class MultiplexNetwork:
    """
    Container for the combined social network.

    Attributes
    ----------
    graph:
        Combined weighted NetworkX graph.

    adjacency_matrix:
        Weighted adjacency matrix A.

        A[i, j] is the raw influence weight from agent j to agent i.

    influence_matrix:
        Row-normalized influence matrix W.

        W[i, j] is the normalized influence of agent j on agent i.
        Each row sums to 1 if the agent has at least one social tie.

    layer_edge_counts:
        Number of edges contributed by each layer.
    """

    graph: nx.Graph
    adjacency_matrix: np.ndarray
    influence_matrix: np.ndarray
    layer_edge_counts: Dict[str, int]


def combine_network_layers(
    graphs: List[nx.Graph],
    n_agents: int,
) -> nx.Graph:
    """
    Combine multiple weighted network layers into one graph.

    If the same edge exists in multiple layers, the weights are added.

    Example
    -------
    If agents 4 and 7 are both household members and friends:

        household weight = 0.8
        friendship weight = 0.3

    Then combined edge weight:

        total weight = 1.1

    The edge attribute "layers" records all layers contributing to the tie.
    """

    if n_agents <= 0:
        raise ValueError("n_agents must be a positive integer.")

    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(range(n_agents))

    for graph in graphs:
        for agent_a, agent_b, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            layer = data.get("layer", "unknown")

            if combined_graph.has_edge(agent_a, agent_b):
                combined_graph[agent_a][agent_b]["weight"] += weight
                combined_graph[agent_a][agent_b]["layers"].append(layer)
            else:
                combined_graph.add_edge(
                    agent_a,
                    agent_b,
                    weight=weight,
                    layers=[layer],
                )

    return combined_graph


def create_adjacency_matrix(
    graph: nx.Graph,
    n_agents: int,
) -> np.ndarray:
    """
    Create a weighted adjacency matrix from the combined graph.

    Rows and columns follow agent IDs:

        0, 1, 2, ..., n_agents - 1

    A[i, j] gives the raw weight of the tie between agent i and agent j.
    """

    adjacency_matrix = nx.to_numpy_array(
        graph,
        nodelist=list(range(n_agents)),
        weight="weight",
    )

    return adjacency_matrix


def row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Row-normalize a matrix.

    This converts raw tie weights into relative influence weights.

    Example
    -------
    If agent i has three social ties with weights:

        [0.8, 0.3, 0.3]

    then the normalized weights become:

        [0.571, 0.214, 0.214]

    This means the household member has stronger influence than each friend,
    but all influences together sum to 1.

    Rows with no social ties remain all zeros.
    """

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )

    return normalized_matrix


def count_layer_edges(graph: nx.Graph) -> Dict[str, int]:
    """
    Count how many combined edges contain each layer type.

    Important
    ---------
    If one edge belongs to both household and friendship layers, it is counted
    once for household and once for friendship.

    This is intentional because the function counts layer participation, not
    unique graph edges.
    """

    counts: Dict[str, int] = {}

    for _, _, data in graph.edges(data=True):
        layers = data.get("layers", ["unknown"])

        for layer in layers:
            counts[layer] = counts.get(layer, 0) + 1

    return counts


def create_multiplex_network(
    graphs: List[nx.Graph],
    n_agents: int,
) -> MultiplexNetwork:
    """
    Create the full multiplex network.

    Steps
    -----
    1. Combine all network layers into one weighted graph.
    2. Convert the graph into a weighted adjacency matrix.
    3. Row-normalize the adjacency matrix to create the influence matrix.
    4. Count layer-specific edge contributions.

    Returns
    -------
    MultiplexNetwork:
        Dataclass containing:
            - combined graph
            - raw adjacency matrix
            - normalized influence matrix
            - layer edge counts
    """

    combined_graph = combine_network_layers(
        graphs=graphs,
        n_agents=n_agents,
    )

    adjacency_matrix = create_adjacency_matrix(
        graph=combined_graph,
        n_agents=n_agents,
    )

    influence_matrix = row_normalize_matrix(adjacency_matrix)

    layer_edge_counts = count_layer_edges(combined_graph)

    return MultiplexNetwork(
        graph=combined_graph,
        adjacency_matrix=adjacency_matrix,
        influence_matrix=influence_matrix,
        layer_edge_counts=layer_edge_counts,
    )


def summarize_multiplex_network(network: MultiplexNetwork) -> Dict[str, float]:
    """
    Calculate summary statistics for the combined network.
    """

    graph = network.graph
    degrees = np.array([degree for _, degree in graph.degree()])

    row_sums = network.influence_matrix.sum(axis=1)
    isolated_agents = np.sum(row_sums == 0)

    return {
        "n_agents": float(graph.number_of_nodes()),
        "n_edges": float(graph.number_of_edges()),
        "mean_degree": float(np.mean(degrees)),
        "min_degree": float(np.min(degrees)),
        "max_degree": float(np.max(degrees)),
        "average_clustering": float(nx.average_clustering(graph)),
        "connected": float(nx.is_connected(graph)),
        "isolated_agents": float(isolated_agents),
    }


if __name__ == "__main__":
    from households import create_household_network
    from friendships import create_friendship_network

    N_AGENTS = 1000

    household_network = create_household_network(
        n_agents=N_AGENTS,
        weight=0.8,
        seed=42,
    )

    friendship_network = create_friendship_network(
        n_agents=N_AGENTS,
        average_friends=8,
        rewiring_probability=0.1,
        weight=0.3,
        seed=42,
    )

    multiplex_network = create_multiplex_network(
        graphs=[
            household_network.graph,
            friendship_network.graph,
        ],
        n_agents=N_AGENTS,
    )

    summary = summarize_multiplex_network(multiplex_network)

    print("Multiplex network summary")
    print("-------------------------")
    print(f"Number of agents: {int(summary['n_agents'])}")
    print(f"Number of total edges: {int(summary['n_edges'])}")
    print(f"Average degree: {summary['mean_degree']:.2f}")
    print(f"Minimum degree: {int(summary['min_degree'])}")
    print(f"Maximum degree: {int(summary['max_degree'])}")
    print(f"Average clustering: {summary['average_clustering']:.3f}")
    print(f"Connected: {bool(summary['connected'])}")
    print(f"Isolated agents: {int(summary['isolated_agents'])}")
    print(f"Layer edge counts: {multiplex_network.layer_edge_counts}")

    print()
    print("Influence matrix check")
    print("----------------------")
    print(f"Shape: {multiplex_network.influence_matrix.shape}")
    print(f"Mean row sum: {multiplex_network.influence_matrix.sum(axis=1).mean():.3f}")