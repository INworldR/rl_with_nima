import gymnasium as gym
import numpy as np
import pandas as pd
import pickle  # Import pickle module


def collect_interaction_history(
    env_name="CartPole-v1", episodes=100, output_file="random_agent_history.pkl"
):
    """
    Interacts with the environment for a given number of episodes, collects state-action-next state pairs,
    and saves them to a pickle file.

    Parameters:
    - env_name: str, name of the environment (default is "CartPole-v1")
    - episodes: int, number of episodes to run
    - output_file: str, name of the output pickle file to save the history

    Returns:
    - history: list of dictionaries, each containing (state, action, next_state)
    """

    # Create the environment
    env = gym.make(env_name, render_mode="rgb_array")

    # List to store the states, actions, and next states
    history = []

    # Run the agent through several episodes
    for episode in range(episodes):
        terminated = False
        truncated = False
        total_reward = 0

        # Reset the environment at the beginning of each episode
        initial_state = env.reset()
        current_state = initial_state[0]

        while not terminated and not truncated:
            # Randomly choose an action from the action space
            action = np.random.randint(0, env.action_space.n)

            # Take the chosen action and get the next state, reward, etc.
            next_state, reward, terminated, truncated, info = env.step(action)

            # Append the data as a dictionary (state, action, next_state) to the history
            history.append(
                {"state": current_state, "action": action, "next_state": next_state}
            )

            # Move to the next state
            current_state = next_state

    # Save the history to a pickle file for further analysis
    with open(output_file, "wb") as f:
        pickle.dump(pd.DataFrame(history), f)

    return history


# Example usage
history = collect_interaction_history(
    episodes=1000, output_file="data/random_agent_history.pkl"
)

# Print the first few entries in the history
print("History of interactions:")
print(history[:5])
