"""Learning agent for the learned CartPole environment."""
import gymnasium as gym
from matplotlib import pyplot as plt
from plot_util import visualize_env
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from tqdm import tqdm

from learned_cartpole import CartPole_Learned

# Initialize Ray
ray.init()

# Register our custom environment with RLlib
def env_creator(env_config):
    return CartPole_Learned(**env_config)

register_env("CartPole-Learned-v0", env_creator)

# 1 - Build an agent
# 1.1 - Get the default config of DQN
config = DQNConfig()

# 1.2 - Examine the config by converting it to a dict via .to_dict() method
config_as_dict = config.to_dict()

# 1.3 - Modify the config if needed
# 1.4 - Introduce the environment to the agent's config
config.environment(env="CartPole-Learned-v0", env_config={}).framework(
    framework="tf2", eager_tracing=True
).rollouts(num_rollout_workers=4, num_envs_per_worker=2).evaluation(
    evaluation_config={"explore": False},
    evaluation_duration=10,
    evaluation_interval=1,
    evaluation_duration_unit="episodes",
)

# 1.5 - Build the agent from the config
agent = config.build()

# 3 - Run training loop
nr_trainings = 100
mean_rewards = []
print("Starting training...")

# Create progress bar
pbar = tqdm(total=nr_trainings, desc="Training Progress")

for i in range(nr_trainings):
    result = agent.train()
    mean_reward = agent.evaluate()["env_runners"]["episode_reward_mean"]
    mean_rewards.append(mean_reward)
    
    # Update progress bar with current reward
    pbar.set_postfix({"mean_reward": f"{mean_reward:.2f}"})
    pbar.update(1)

pbar.close()

# Plot the mean rewards
plt.figure(figsize=(10, 6))
plt.plot(mean_rewards)
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.title("Mean reward vs. training rounds")
plt.grid(True)
plt.savefig("mean_reward_vs_training_rounds_learned.png")
plt.close()

print("\nTraining completed. Starting visualization...")

# 4 - Visualize the trained agent
env = CartPole_Learned(render_mode="rgb_array")
s, _ = env.reset()
done = False
cumulative_reward = 0

while not done:
    # Let the agent choose an action
    a = agent.compute_single_action(observation=s, explore=False)
    # Pass it to the environment
    s, r, terminated, truncated, info = env.step(action=a)

    # Keep track of the score
    cumulative_reward += r
    done = terminated or truncated
    
    # Visualize the agent
    visualize_env(env=env, pause_sec=0.1)

print("Total reward:", cumulative_reward)
env.close()

# Shutdown Ray
ray.shutdown()
print("Good-bye.") 