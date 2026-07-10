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
    n_agents = 100
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

    parameters = ModelParameters()

    model = FoodTransitionModel(
        agents=agents,
        household_graph=household_network.graph,
        friendship_graph=friendship_network.graph,
        parameters=parameters,
        environmental_health_status= 0.9,
    )

    history = run_model(
        model=model,
        n_steps=n_steps,
    )

    print("Simulation finished.")
    print(f"Final EH status: {history['environmental_health_status'][-1]:.3f}")
    print(f"Final mean IH status: {history['mean_individual_health_status'][-1]:.3f}")
    print(f"Final mean env behavior: {history['mean_behavior_env'][-1]:.3f}")
    print(f"Final mean health behavior: {history['mean_behavior_health'][-1]:.3f}")

    plt.figure()
    plt.scatter(
        model.agents.behavior_env,
        model.agents.behavior_health,
        alpha=0.5,
    )
    plt.xlabel("Environmental behavior")
    plt.ylabel("Individual-health behavior")
    plt.title("Agent behavior after simulation")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history["environmental_health_status"], label="Environmental health status")
    plt.plot(history["mean_individual_health_status"], label="Mean individual health status")
    plt.xlabel("Simulation step")
    plt.ylabel("Status")
    plt.title("Health-status dynamics")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

