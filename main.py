# main.py

from Network.households import create_household_network, summarize_households
from Network.friendships import create_friendship_network, summarize_friendships
from Network.multiplex import create_multiplex_network, summarize_multiplex_network


def main() -> None:
    """
    Build the first complete social network for the food-transition ABM.

    Current network layers:
    1. Household network
    2. Friendship network

    These layers are created separately and then combined into one weighted
    influence matrix for later simulation.
    """

    n_agents = 1000
    seed = 42

    household_network = create_household_network(
        n_agents=n_agents,
        weight=0.8,
        seed=seed,
    )

    friendship_network = create_friendship_network(
        n_agents=n_agents,
        average_friends=8,
        rewiring_probability=0.1,
        weight=0.3,
        seed=seed,
    )

    multiplex_network = create_multiplex_network(
        graphs=[
            household_network.graph,
            friendship_network.graph,
        ],
        n_agents=n_agents,
    )

    household_summary = summarize_households(household_network.households)
    friendship_summary = summarize_friendships(friendship_network.graph)
    multiplex_summary = summarize_multiplex_network(multiplex_network)

    print("\nHOUSEHOLD NETWORK")
    print("-----------------")
    print(f"Number of households: {int(household_summary['n_households'])}")
    print(f"Average household size: {household_summary['mean_size']:.2f}")
    print(f"Share single-person households: {household_summary['share_single_person']:.2%}")

    print("\nFRIENDSHIP NETWORK")
    print("------------------")
    print(f"Friendship ties: {int(friendship_summary['n_friendship_ties'])}")
    print(f"Average friends per agent: {friendship_summary['mean_degree']:.2f}")
    print(f"Average clustering: {friendship_summary['clustering']:.3f}")

    print("\nMULTIPLEX NETWORK")
    print("-----------------")
    print(f"Total edges: {int(multiplex_summary['n_edges'])}")
    print(f"Average degree: {multiplex_summary['mean_degree']:.2f}")
    print(f"Connected: {bool(multiplex_summary['connected'])}")
    print(f"Layer edge counts: {multiplex_network.layer_edge_counts}")

    print("\nINFLUENCE MATRIX")
    print("----------------")
    print(f"Shape: {multiplex_network.influence_matrix.shape}")
    print(f"Mean row sum: {multiplex_network.influence_matrix.sum(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()