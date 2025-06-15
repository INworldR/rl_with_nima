#!/home/marc/miniconda3/envs/jupyter-ai/bin/python
"""
Main entry point for the CartPole RL experiments.

This module provides the primary interface for training and evaluating
reinforcement learning agents on the CartPole environment.
"""

import gymnasium as gym
import numpy as np
import argparse
import logging
from typing import Optional, Dict, Any


class CartPoleExperiment:
    """Main experiment class for CartPole RL training."""
    
    def __init__(self, 
                 env_name: str = "CartPole-v1",
                 seed: Optional[int] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the CartPole experiment.
        
        Args:
            env_name: Name of the Gymnasium environment
            seed: Random seed for reproducibility
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.env_name = env_name
        self.seed = seed
        self.render_mode = render_mode
        self.env = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def setup_environment(self) -> gym.Env:
        """Create and configure the CartPole environment."""
        self.env = gym.make(self.env_name, render_mode=self.render_mode)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.logger.info(f"Environment created: {self.env_name}")
        self.logger.info(f"Action space: {self.env.action_space}")
        self.logger.info(f"Observation space: {self.env.observation_space}")
        
        return self.env
        
    def run_random_episode(self) -> Dict[str, Any]:
        """
        Run a single episode with random actions.
        
        Returns:
            Dictionary containing episode statistics
        """
        if self.env is None:
            self.setup_environment()
            
        observation, info = self.env.reset(seed=self.seed)
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
        self.logger.info(f"Episode finished: {steps} steps, total reward: {total_reward}")
        
        return {
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated
        }
        
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
            self.logger.info("Environment closed")


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="CartPole RL Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    experiment = CartPoleExperiment(seed=args.seed, render_mode=render_mode)
    
    try:
        experiment.setup_environment()
        
        results = []
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            result = experiment.run_random_episode()
            results.append(result)
            
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        
        print(f"\nSummary over {args.episodes} episodes:")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.2f}")
        
    finally:
        experiment.close()


if __name__ == "__main__":
    main()