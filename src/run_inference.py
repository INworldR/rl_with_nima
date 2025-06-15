"""Run inference with a trained DQN model and save transitions to pickle file."""

import os
import pickle
import gymnasium as gym
import numpy as np
import pandas as pd
from ray.rllib.algorithms.dqn import DQNConfig
import tensorflow as tf

# Make sure TF2 is configured correctly
tf.compat.v1.enable_resource_variables()

# Number of episodes to run
episodes = 100


def collect_transitions_with_agent(checkpoint_dir, num_episodes=100):
    """Load agent from checkpoint and collect transitions."""
    print(f"Loading agent from checkpoint: {checkpoint_dir}")

    # Build a DQN config with the same settings used during training
    config = DQNConfig()
    config.environment(env="CartPole-v1").framework(
        framework="tf2", eager_tracing=True
    ).rollouts(
        num_rollout_workers=0
    )  # Use 0 workers for inference

    # Build the agent from the config
    agent = config.build()

    # Restore from checkpoint
    agent.restore(checkpoint_dir)
    print("Agent successfully restored")

    # Create environment for inference (without rendering)
    env = gym.make("CartPole-v1", render_mode=None)

    # Collect data
    history = []
    rewards = []

    for episode in range(num_episodes):
        initial_state, _ = env.reset()
        current_state = initial_state
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            # Get action from the trained agent (with exploration disabled)
            action = agent.compute_single_action(
                observation=current_state, explore=False
            )

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Save transition information - keep numpy arrays intact for pickle
            history.append((current_state, int(action), next_state))

            # Update current state
            current_state = next_state

        rewards.append(total_reward)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")

    env.close()

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

    return history, rewards


def main():
    """Main function to run inference and save data."""
    checkpoint_dir = "./cartpole_dqn_checkpoints"

    try:
        # Collect transitions using the trained agent
        history, rewards = collect_transitions_with_agent(checkpoint_dir, episodes)

        # Export transitions as pickle file
        with open("trained_agent_history.pkl", "wb") as f:
            pickle.dump(history, f)

        # Also save the episode rewards as pickle
        with open("trained_agent_rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)

        print(f"Saved {len(history)} transitions to trained_agent_history.pkl")
        print(f"Saved {len(rewards)} episode rewards to trained_agent_rewards.pkl")

    except Exception as e:
        print(f"Error running inference with trained agent: {e}")
        print("Falling back to random agent for data collection")

        # Fallback to random agent if loading fails
        random_history, random_rewards = collect_random_transitions(episodes)

        # Save random agent data as pickle
        with open("random_agent_history.pkl", "wb") as f:
            pickle.dump(random_history, f)

        with open("random_agent_rewards.pkl", "wb") as f:
            pickle.dump(random_rewards, f)

        print(f"Saved {len(random_history)} transitions to random_agent_history.pkl")
        print(
            f"Saved {len(random_rewards)} episode rewards to random_agent_rewards.pkl"
        )


def collect_random_transitions(num_episodes=100):
    """Collect transitions using a random policy."""
    env = gym.make("CartPole-v1", render_mode=None)

    history = []
    rewards = []

    for episode in range(num_episodes):
        initial_state, _ = env.reset()
        current_state = initial_state
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Keep numpy arrays intact for pickle
            history.append((current_state, int(action), next_state))

            current_state = next_state

        rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward} (random policy)"
            )

    env.close()

    avg_reward = sum(rewards) / len(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

    return history, rewards


if __name__ == "__main__":
    main()
