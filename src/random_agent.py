import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create an environment for the agent to interact with
env = gym.make("CartPole-v1", render_mode="rgb_array")

# History list to store the states, actions, and next states
history = []

# Number of episodes to run
episodes = 100

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

        # Append the (initial_state, action, next_state) to the history
        history.append((current_state, action, next_state))

        # Move to the next state
        current_state = next_state

# Print out the history of interactions (states, actions, next states)
print("History of interactions between the agent and environment:")
for initial_state, action, next_state in history:
    print(f"Initial State: {initial_state}, Action: {action}, Next State: {next_state}")

# Save the history to a CSV file for further analysis
df = pd.DataFrame(history, columns=["Initial_State", "Action", "Next_State"])
df.to_csv("random_agent_history.csv", index=False)
