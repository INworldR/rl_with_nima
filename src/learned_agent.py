"""Learning agent for the learned CartPole environment."""
import gymnasium as gym
from matplotlib import pyplot as plt
from plot_util import visualize_env
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from tqdm.auto import tqdm
import logging
import warnings
import os
import sys
import time
import signal
from contextlib import contextmanager
from gymnasium.wrappers import TimeLimit

from learned_cartpole2 import CartPole_Learned2

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["RAY_DEDUP_LOGS"] = "0"  # Disable log deduplication

# Set logging to ERROR level only
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Timeout context manager
@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Redirect stdout to suppress Ray's output
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Initialize Ray with minimal logging
with SuppressOutput():
    ray.init(
        logging_level=logging.ERROR,
        log_to_driver=False,
        include_dashboard=False,
        ignore_reinit_error=True,
        local_mode=True,
        num_cpus=1
    )

# Register our custom environment with RLlib
def env_creator(env_config):
    env = CartPole_Learned2(**env_config)
    env.spec = gym.spec("CartPole-v1")
    #env.spec.max_episode_steps = 200  # Set max episode stepsenv = TimeLimit(env, max_episode_steps=1000)
    env = TimeLimit(env, max_episode_steps=200)
    return env

# Register environment
register_env("CartPole-Learned-v0", env_creator)

# Configure DQN with minimal settings
config = DQNConfig()
config = config.environment(env="CartPole-Learned-v0", env_config={})
config = config.framework(framework="tf2")
config = config.rollouts(
    num_rollout_workers=0,
    num_envs_per_worker=1,
    batch_mode="complete_episodes",
    rollout_fragment_length=10
)
config = config.training(
    train_batch_size=20,
    lr=0.001,
    gamma=0.99,
    target_network_update_freq=20,
    replay_buffer_config={
        "type": "MultiAgentReplayBuffer",
        "capacity": 100
    },
    n_step=1,
    double_q=True,
    dueling=True
)
config = config.evaluation(
    evaluation_config={"explore": False},
    evaluation_duration=1,
    evaluation_interval=1,
    evaluation_duration_unit="episodes",
)

# Build agent
with SuppressOutput():
    agent = config.build()

# Training loop
nr_trainings = 100
mean_rewards = []

pbar = tqdm(
    total=nr_trainings,
    desc="Training Progress",
    bar_format='{l_bar}{bar:30}{r_bar}',
    ncols=100,
    position=0,
    leave=True
)

try:
    for i in range(nr_trainings):
        # Train with timeout
        try:
            with timeout(30):
                result = agent.train()
        except TimeoutError:
            raise
        
        # Evaluate with timeout
        try:
            with timeout(10):
                eval_result = agent.evaluate()
                mean_reward = eval_result["evaluation"]["sampler_results"]["episode_reward_mean"]
        except TimeoutError:
            raise
        
        mean_rewards.append(mean_reward)
        
        # Update progress
        pbar.set_postfix({
            "mean_reward": f"{mean_reward:.2f}",
            "iteration": f"{i+1}/{nr_trainings}"
        })
        pbar.update(1)
        pbar.refresh()
        
except Exception as e:
    raise e
finally:
    pbar.close()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(mean_rewards)
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.title("Mean reward vs. training rounds")
plt.grid(True)
plt.savefig("mean_reward_vs_training_rounds_learned.png")
plt.close()

# Visualize trained agent
env = CartPole_Learned2(render_mode="rgb_array")
s, _ = env.reset()
done = False
cumulative_reward = 0

while not done:
    a = agent.compute_single_action(observation=s, explore=False)
    s, r, terminated, truncated, info = env.step(action=a)
    cumulative_reward += r
    done = terminated or truncated
    visualize_env(env=env, pause_sec=0.1)

env.close()

# Cleanup
ray.shutdown() 