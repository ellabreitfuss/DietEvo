import matplotlib.pyplot as plt

from Network.households import create_household_network
from Network.friendships import create_friendship_network

from Simulation.agents import initialize_agents
from Simulation.model import (
    FoodTransitionModel,
    ModelParameters,
    run_model,
)


def main() -> None:
    """
    Build the network, initialize agents, run the ABM,
    and plot final agent intentions.
    """

    n_agents = 1000
    seed = 42
    n_steps = 50

    household_network = create_household_network(
        n_agents=n_agents,
        weight=1.0,
        seed=seed,
    )

    friendship_network = create_friendship_network(
        n_agents=n_agents,
        average_friends=8,
        rewiring_probability=0,
        weight=1.0,
        seed=seed,
    )

    agents = initialize_agents(
        n_agents=n_agents,
        seed=seed,
    )

    parameters = ModelParameters(
        intention_weight_attitude=0.4,
        intention_weight_norm=0.4,
        intention_weight_PBC=0.2,
        household_weight=0.8,
        friendship_weight=0.2,
        norm_learning_rate=0.25,
        attitude_learning_rate=0.10,
        health_status_learning_rate=0.10,
    )

    model = FoodTransitionModel(
        agents=agents,
        household_graph=household_network.graph,
        friendship_graph=friendship_network.graph,
        parameters=parameters,
        planetary_health_status=0.2,
    )

    history = run_model(
        model=model,
        n_steps=n_steps,
    )

    print("Simulation finished.")
    print(f"Mean environmental intention: {history['mean_intention_env'][-1]:.3f}")
    print(f"Mean health intention: {history['mean_intention_health'][-1]:.3f}")

    plt.figure()
    plt.scatter(
        model.agents.intention_env,
        model.agents.intention_health,
        alpha=0.5,
    )
    plt.xlabel("Environmental / planetary-health intention")
    plt.ylabel("Individual-health intention")
    plt.title("Agent intentions after simulation")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["mean_intention_env"], label="Environmental intention")
    plt.plot(history["mean_intention_health"], label="Health intention")
    plt.xlabel("Simulation step")
    plt.ylabel("Mean intention")
    plt.title("Mean intentions over time")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
